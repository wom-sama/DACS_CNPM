from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from trkh.core.config import load_data_spec
from trkh.data.dataset import PairedViewTrainDataset
from trkh.tools.evaluate_embedding_retrieval import (
    OrderedFolderDataset,
    OrderedYoloObjectDataset,
)
from trkh.tools.export_aidt_predictions import SquarePad
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)
from trkh.tools.remap_classification_teacher_to_yolo import _classification_key


RESNET_NAME = "resnet50.a1_in1k"
VIT_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
IMAGE_SIZE = 224
BATCH_SIZE = 32
YOLO_CONTEXT_MARGIN_RATIO = 0.50
EXPECTED_ROWS = {"train": 9215, "val": 2606}
LITERATURE = (
    "https://huggingface.co/timm/resnet50.a1_in1k",
    "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "https://github.com/huggingface/pytorch-image-models",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen raw-pretrained ResNet50+ViT-B/16 features using the "
            "exact AIDT square-pad evaluation geometry. Only train/val are read."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resnet", type=str, default=RESNET_NAME)
    parser.add_argument("--vit", type=str, default=VIT_NAME)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument(
        "--yolo-crop-margin-ratio",
        type=float,
        default=YOLO_CONTEXT_MARGIN_RATIO,
    )
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().casefold()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _selected_pretrained_config(config: Mapping[str, object]) -> Dict[str, object]:
    keys = (
        "architecture",
        "tag",
        "hf_hub_id",
        "input_size",
        "interpolation",
        "crop_pct",
        "mean",
        "std",
        "num_classes",
    )
    output: Dict[str, object] = {}
    for key in keys:
        value = config.get(key)
        if isinstance(value, tuple):
            value = list(value)
        if value is not None:
            output[key] = value
    return output


def _load_backbone(
    name: str,
    *,
    global_pool: Optional[str],
) -> Tuple[nn.Module, Dict[str, object]]:
    kwargs: Dict[str, object] = {"pretrained": True, "num_classes": 0}
    if global_pool is not None:
        kwargs["global_pool"] = str(global_pool)
    model = timm.create_model(str(name), **kwargs)
    data_config = dict(timm.data.resolve_model_data_config(model))
    metadata = {
        "name": str(name),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "feature_dim": int(getattr(model, "num_features")),
        "state_dict_sha256": state_dict_sha256(model),
        "pretrained_config": _selected_pretrained_config(
            dict(getattr(model, "pretrained_cfg", {}) or {})
        ),
        "resolved_data_config": _selected_pretrained_config(data_config),
    }
    return model, metadata


def _normalization_tensors(
    config: Mapping[str, object],
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    mean = tuple(float(value) for value in config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(value) for value in config.get("std", (0.229, 0.224, 0.225)))
    return (
        torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1),
        torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1),
    )


def _dataset_sample_key(
    dataset: OrderedFolderDataset | OrderedYoloObjectDataset,
    *,
    sample_index: int,
    path_text: str,
    label: int,
) -> Tuple[str, int]:
    if isinstance(dataset, OrderedYoloObjectDataset):
        source_stem, object_index, source_label = PairedViewTrainDataset._sample_key(
            dataset.dataset,
            int(sample_index),
        )
        if int(source_label) != int(label):
            raise ValueError(
                "YOLO sample-key label differs from extracted label: "
                f"sample_index={sample_index}, {source_label} != {label}"
            )
        return str(source_stem), int(object_index)
    return _classification_key(str(path_text))


def _extract_split(
    *,
    split: str,
    dataset: OrderedFolderDataset | OrderedYoloObjectDataset,
    loader: DataLoader,
    resnet: nn.Module,
    vit: nn.Module,
    resnet_normalization: Tuple[Tensor, Tensor],
    vit_normalization: Tuple[Tensor, Tensor],
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    resnet.eval()
    vit.eval()
    feature_rows: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    sample_indices: List[int] = []
    source_stems: List[str] = []
    object_indices: List[int] = []
    start = time.time()
    resnet_mean, resnet_std = resnet_normalization
    vit_mean, vit_std = vit_normalization

    with torch.inference_mode():
        for images, batch_labels, batch_paths, batch_indices, _batch_stems in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp) and device.type == "cuda",
            ):
                resnet_features = resnet((images - resnet_mean) / resnet_std)
                vit_features = vit((images - vit_mean) / vit_std)
            if resnet_features.ndim != 2 or vit_features.ndim != 2:
                raise RuntimeError("AIDT backbones did not return pooled 2D features")
            combined = torch.cat((resnet_features.float(), vit_features.float()), dim=1)
            feature_rows.append(combined.cpu().numpy())
            labels.extend(int(value) for value in batch_labels.tolist())
            paths.extend(str(value) for value in batch_paths)
            sample_indices.extend(int(value) for value in batch_indices.tolist())
            for path_text, sample_index, label in zip(
                batch_paths,
                batch_indices.tolist(),
                batch_labels.tolist(),
            ):
                source_stem, object_index = _dataset_sample_key(
                    dataset,
                    sample_index=int(sample_index),
                    path_text=str(path_text),
                    label=int(label),
                )
                source_stems.append(str(source_stem))
                object_indices.append(int(object_index))

    features = np.concatenate(feature_rows, axis=0).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    if features.shape[0] != len(dataset) or label_array.shape != (len(dataset),):
        raise RuntimeError(f"{split} extraction row count does not match its dataset")
    if not np.isfinite(features).all():
        raise RuntimeError(f"{split} extraction produced non-finite features")
    if len(set(sample_indices)) != len(dataset):
        raise RuntimeError(f"{split} extraction has duplicate dataset indices")

    return {
        "features": features,
        "labels": label_array,
        "paths": np.asarray(paths, dtype=object),
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
        "source_stem": np.asarray(source_stems, dtype=object),
        "object_index": np.asarray(object_indices, dtype=np.int64),
        "classes": np.asarray(dataset.class_names, dtype=object),
        "seconds": float(time.time() - start),
        "source_groups": int(len(set(source_stems))),
        "class_counts": np.bincount(label_array, minlength=len(dataset.class_names)).tolist(),
    }


def run_extraction(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.image_size) <= 0 or int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("image-size/batch-size/workers are invalid")
    if int(args.max_samples_per_class) < 0:
        raise ValueError("max-samples-per-class must be non-negative")
    if float(args.yolo_crop_margin_ratio) < 0.0:
        raise ValueError("yolo-crop-margin-ratio must be non-negative")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(str(args.device))
    amp_dtype = torch.bfloat16 if str(args.amp_dtype) == "bf16" else torch.float16

    data_spec = load_data_spec(Path(args.data))
    data_format = str(data_spec.data_format).strip().casefold()
    if data_format not in {"classification_folder", "yolo"}:
        raise ValueError("raw AIDT feature extraction requires classification_folder or yolo data")
    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((int(args.image_size), int(args.image_size)), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    resnet, resnet_metadata = _load_backbone(str(args.resnet), global_pool="avg")
    vit, vit_metadata = _load_backbone(str(args.vit), global_pool=None)
    resnet.to(device=device).eval()
    vit.to(device=device).eval()
    resnet_normalization = _normalization_tensors(
        resnet_metadata["resolved_data_config"], device=device
    )
    vit_normalization = _normalization_tensors(
        vit_metadata["resolved_data_config"], device=device
    )

    split_payloads: Dict[str, Dict[str, object]] = {}
    split_summaries: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val"):
        if data_format == "classification_folder":
            dataset = OrderedFolderDataset(
                data_spec=data_spec,
                split=split,
                transform=transform,
                max_samples_per_class=int(args.max_samples_per_class),
            )
        else:
            dataset = OrderedYoloObjectDataset(
                data_spec=data_spec,
                split=split,
                transform=transform,
                max_samples_per_class=int(args.max_samples_per_class),
                crop_margin_ratio=float(args.yolo_crop_margin_ratio),
            )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.workers) > 0,
        )
        payload = _extract_split(
            split=split,
            dataset=dataset,
            loader=loader,
            resnet=resnet,
            vit=vit,
            resnet_normalization=resnet_normalization,
            vit_normalization=vit_normalization,
            device=device,
            amp=bool(args.amp),
            amp_dtype=amp_dtype,
        )
        output_path = output_dir / f"features_{split}_raw_aidt_resnet50_vitb16.npz"
        np.savez_compressed(
            output_path,
            features=payload["features"],
            labels=payload["labels"],
            paths=payload["paths"],
            sample_index=payload["sample_index"],
            source_stem=payload["source_stem"],
            object_index=payload["object_index"],
            classes=payload["classes"],
            resnet_dim=np.asarray([int(resnet_metadata["feature_dim"])], dtype=np.int64),
            vit_dim=np.asarray([int(vit_metadata["feature_dim"])], dtype=np.int64),
            resnet_name=np.asarray([str(args.resnet)], dtype=object),
            vit_name=np.asarray([str(args.vit)], dtype=object),
            resnet_state_dict_sha256=np.asarray(
                [str(resnet_metadata["state_dict_sha256"])], dtype=object
            ),
            vit_state_dict_sha256=np.asarray(
                [str(vit_metadata["state_dict_sha256"])], dtype=object
            ),
            raw_pretrained=np.asarray([True], dtype=np.bool_),
            data_format=np.asarray([data_format], dtype=object),
            yolo_crop_margin_ratio=np.asarray(
                [float(args.yolo_crop_margin_ratio) if data_format == "yolo" else -1.0],
                dtype=np.float32,
            ),
        )
        split_payloads[split] = payload
        split_summaries[split] = {
            "rows": int(len(payload["labels"])),
            "expected_full_rows": int(EXPECTED_ROWS[split]),
            "full_support": bool(
                int(args.max_samples_per_class) == 0
                and len(payload["labels"]) == EXPECTED_ROWS[split]
            ),
            "source_groups": int(payload["source_groups"]),
            "class_counts": list(payload["class_counts"]),
            "feature_shape": list(np.asarray(payload["features"]).shape),
            "seconds": float(payload["seconds"]),
            "output": str(output_path),
        }

    train_sources = {
        str(value).casefold()
        for value in np.asarray(split_payloads["train"]["source_stem"], dtype=object)
    }
    val_sources = {
        str(value).casefold()
        for value in np.asarray(split_payloads["val"]["source_stem"], dtype=object)
    }
    source_overlap = len(train_sources.intersection(val_sources))
    summary = {
        "method": "raw_pretrained_aidt_resnet50_vitb16_feature_extraction",
        "literature": list(LITERATURE),
        "data": str(Path(args.data).resolve()),
        "data_format": data_format,
        "view": "wide_yolo_bbox_context" if data_format == "yolo" else "classification_object_crop",
        "split_usage": {"train": True, "val": True, "test": False},
        "geometry": (
            "SquarePad(fill=0) then direct "
            f"Resize({int(args.image_size)},{int(args.image_size)}), matching AIDT eval"
        ),
        "branch_normalization": "each backbone uses its own resolved timm mean/std",
        "models": {"resnet": resnet_metadata, "vit": vit_metadata},
        "combined_feature_dim": int(
            int(resnet_metadata["feature_dim"]) + int(vit_metadata["feature_dim"])
        ),
        "splits": split_summaries,
        "train_val_source_overlap": int(source_overlap),
        "max_samples_per_class": int(args.max_samples_per_class),
        "yolo_crop_margin_ratio": (
            float(args.yolo_crop_margin_ratio) if data_format == "yolo" else None
        ),
        "device": str(device),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "compute_precision": str(args.amp_dtype) if bool(args.amp) else "fp32",
        "timm_version": str(timm.__version__),
        "torch_version": str(torch.__version__),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        "# Raw-Pretrained AIDT Backbone Feature Cache",
        "",
        f"- Train/val rows: `{split_summaries['train']['rows']}/{split_summaries['val']['rows']}`",
        f"- Feature dim: `{summary['combined_feature_dim']}`",
        f"- Train/val source overlap: `{source_overlap}`",
        f"- Compute precision: `{summary['compute_precision']}`",
        f"- ResNet state SHA-256: `{resnet_metadata['state_dict_sha256']}`",
        f"- ViT state SHA-256: `{vit_metadata['state_dict_sha256']}`",
        "",
        "The encoders remain frozen and are loaded from their original timm pretrained weights. No test input, raw-data edit, classifier, model, or checkpoint is written.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(
        output_dir,
        mode="raw_pretrained_aidt_feature_cache_manifest",
    )
    summary["artifact_manifest"] = {key: value for key, value in manifest.items() if key != "files"}
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_extraction(parse_args(argv))
    print(
        json.dumps(
            {
                "splits": summary["splits"],
                "models": summary["models"],
                "train_val_source_overlap": summary["train_val_source_overlap"],
                "artifact_manifest": summary["artifact_manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

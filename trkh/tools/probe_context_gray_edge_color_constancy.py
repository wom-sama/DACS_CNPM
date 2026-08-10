from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.models.photometric_invariant import (
    apply_diagonal_color_constancy,
    estimate_gray_edge_illuminant,
)
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _effective_rank,
    _fit_grouped_readout,
    _l2_normalize,
    _load_rgb_cache,
    _normalize_path,
    _source_stems,
    _transition_summary,
    _write_prediction_audit,
)


LITERATURE = [
    "https://staff.science.uva.nl/th.gevers/pub/GeversTIP07.pdf",
    "https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008",
    "https://www.sciencedirect.com/science/article/abs/pii/0016003280900587",
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate illuminant from raw yolo_f source context with Gray-Edge, "
            "correct only the object crop, and audit frozen TRKH complementarity."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--gray-edge-sigma", type=float, default=2.0)
    parser.add_argument("--minkowski-p", type=float, default=6.0)
    parser.add_argument("--bbox-exclusion-margin", type=float, default=0.08)
    parser.add_argument("--min-gain", type=float, default=0.5)
    parser.add_argument("--max-gain", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--logistic-c", type=float, default=0.3)
    parser.add_argument("--logistic-max-iter", type=int, default=600)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _context_dataset(
    *,
    data: Path,
    split: str,
    checkpoint: Mapping[str, object],
    class_name_mode: str,
    max_samples: int,
) -> tuple[Dataset, List[str]]:
    dataset, class_names = _build_dataset(
        data_yaml=Path(data),
        split=str(split),
        checkpoint=checkpoint,
        class_name_mode=str(class_name_mode),
        max_samples=int(max_samples),
    )
    if not isinstance(dataset, MangoYOLOCropDataset):
        raise TypeError("Context Gray-Edge precheck requires object-level yolo_f data")
    dataset.classification_source_context_aux = True
    dataset.classification_source_context_layout = "full"
    dataset.classification_source_context_mode = "gray"
    dataset.classification_source_context_background_alpha = 1.0
    return dataset, class_names


def _angle_degrees(first: Tensor, second: Tensor) -> Tensor:
    first_normalized = first.float() / first.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
    second_normalized = second.float() / second.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
    cosine = (first_normalized * second_normalized).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _extract_split(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    mean: Sequence[float],
    std: Sequence[float],
    gray_edge_sigma: float,
    minkowski_p: float,
    bbox_exclusion_margin: float,
    min_gain: float,
    max_gain: float,
    split: str,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in dataset_paths_fn()] if callable(dataset_paths_fn) else []
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    neutral = torch.full((1, 3), 1.0 / math.sqrt(3.0), device=device, dtype=torch.float32)

    embeddings: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    paths: List[str] = []
    illuminants: List[np.ndarray] = []
    gains_all: List[np.ndarray] = []
    support_all: List[np.ndarray] = []
    context_crop_angle_all: List[np.ndarray] = []
    neutral_angle_all: List[np.ndarray] = []
    clipped_gain_all: List[np.ndarray] = []
    preview: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    seen = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"context-gray-edge-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                raise ValueError("Context Gray-Edge requires tensor metadata")
            required = {
                "source_context_image",
                "source_context_image_mask",
                "source_context_bbox",
            }
            missing = sorted(required.difference(metadata))
            if missing:
                raise ValueError(f"Missing source-context metadata: {missing}")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            context_images = metadata["source_context_image"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            context_mask = metadata["source_context_image_mask"].to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )
            context_bbox = metadata["source_context_bbox"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            crop_mask = None
            if torch.is_tensor(metadata.get("image_mask")):
                crop_mask = metadata["image_mask"].to(device=device, dtype=torch.bool)
            bbox_metadata = None
            if torch.is_tensor(metadata.get("bbox")):
                bbox_metadata = metadata["bbox"].to(device=device, dtype=torch.float32)

            crop_rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            context_rgb = (context_images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            context_illuminant, support = estimate_gray_edge_illuminant(
                context_rgb,
                valid_mask=context_mask,
                excluded_bbox=context_bbox,
                bbox_margin_ratio=float(bbox_exclusion_margin),
                sigma=float(gray_edge_sigma),
                minkowski_p=float(minkowski_p),
            )
            crop_illuminant, _ = estimate_gray_edge_illuminant(
                crop_rgb,
                valid_mask=crop_mask,
                sigma=float(gray_edge_sigma),
                minkowski_p=float(minkowski_p),
            )
            corrected_rgb, gains = apply_diagonal_color_constancy(
                crop_rgb,
                context_illuminant,
                valid_mask=crop_mask,
                min_gain=float(min_gain),
                max_gain=float(max_gain),
            )
            corrected_images = (corrected_rgb - mean_tensor) / std_tensor

            illuminants.append(context_illuminant.detach().float().cpu().numpy())
            gains_all.append(gains.detach().float().cpu().numpy())
            support_all.append(support.detach().float().cpu().numpy())
            context_crop_angle_all.append(
                _angle_degrees(context_illuminant, crop_illuminant).cpu().numpy()
            )
            neutral_angle_all.append(
                _angle_degrees(context_illuminant, neutral.expand_as(context_illuminant)).cpu().numpy()
            )
            clipped_gain_all.append(
                (
                    (gains <= float(min_gain) + 1e-6)
                    | (gains >= float(max_gain) - 1e-6)
                ).float().mean(dim=1).cpu().numpy()
            )

            for row_index, target in enumerate(labels.detach().cpu().tolist()):
                if int(target) in preview:
                    continue
                original = (
                    crop_rgb[row_index].cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
                ).round().astype(np.uint8)
                context = (
                    context_rgb[row_index].cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
                ).round().astype(np.uint8)
                corrected = (
                    corrected_rgb[row_index].cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
                ).round().astype(np.uint8)
                preview[int(target)] = (original, context, corrected)

            with autocast_context(device, bool(amp)):
                features = model.forward_features(
                    corrected_images,
                    image_valid_mask=crop_mask,
                    bbox_token_prior=bbox_metadata,
                )
                if bbox_metadata is not None:
                    features["bbox"] = bbox_metadata
                if hasattr(model, "forward_heads"):
                    output = model.forward_heads(features)
                else:
                    output = classification_logits_from_features(model, features)
                logits, _, _ = extract_detection_from_model_output(output)
                head_input = extract_head_input_from_features(model, features)
            embeddings.append(head_input.detach().float().cpu().numpy())
            probabilities.append(logits.detach().float().softmax(dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

            raw_paths = metadata.get("paths", [])
            batch_paths = [str(path) for path in raw_paths] if isinstance(raw_paths, Sequence) else []
            batch_count = int(images.size(0))
            if dataset_paths and (
                len(batch_paths) != batch_count
                or not any(str(path).strip() for path in batch_paths)
            ):
                batch_paths = dataset_paths[seen : seen + batch_count]
            seen += batch_count
            paths.extend(batch_paths)
    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32, copy=False),
        "probabilities": np.concatenate(probabilities, axis=0).astype(np.float32, copy=False),
        "labels": np.concatenate(labels_all, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "illuminants": np.concatenate(illuminants, axis=0).astype(np.float32, copy=False),
        "gains": np.concatenate(gains_all, axis=0).astype(np.float32, copy=False),
        "support_fraction": np.concatenate(support_all, axis=0).astype(np.float32, copy=False),
        "context_crop_angle": np.concatenate(context_crop_angle_all, axis=0).astype(np.float32, copy=False),
        "neutral_angle": np.concatenate(neutral_angle_all, axis=0).astype(np.float32, copy=False),
        "clipped_gain_fraction": np.concatenate(clipped_gain_all, axis=0).astype(np.float32, copy=False),
        "preview": preview,
    }


def _write_preview(
    path: Path,
    preview: Mapping[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> None:
    tile = 192
    resampling = getattr(Image, "Resampling", Image)
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    for class_index in range(len(class_names)):
        if class_index not in preview:
            continue
        row = Image.new("RGB", (tile * 3, tile), color=(255, 255, 255))
        for column, array in enumerate(preview[class_index]):
            image = Image.fromarray(array).resize((tile, tile), resampling.BILINEAR)
            row.paste(image, (column * tile, 0))
        rows.append(row)
        manifest.append(
            {
                "row": len(rows) - 1,
                "class_index": class_index,
                "class_name": str(class_names[class_index]),
                "columns": ["object_crop", "raw_source_context", "context_corrected_crop"],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 3, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_feature_cache(
    path: Path,
    *,
    payload: Mapping[str, object],
    sample_index: np.ndarray,
    class_names: Sequence[str],
) -> None:
    np.savez_compressed(
        path,
        features=np.asarray(payload["embeddings"], dtype=np.float32),
        probabilities=np.asarray(payload["probabilities"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        paths=np.asarray(payload["paths"], dtype=object),
        sample_index=np.asarray(sample_index, dtype=np.int64),
        source_stem=_source_stems(np.asarray(payload["paths"], dtype=object)),
        classes=np.asarray(class_names, dtype=object),
        illuminants=np.asarray(payload["illuminants"], dtype=np.float32),
        gains=np.asarray(payload["gains"], dtype=np.float32),
        support_fraction=np.asarray(payload["support_fraction"], dtype=np.float32),
        view=np.asarray(["context_gray_edge_corrected_crop"], dtype=object),
    )


def assess_context_color_constancy_readiness(
    *,
    train_samples: int,
    val_samples: int,
    alignment_ok: bool,
    mean_support_fraction: float,
    clipped_gain_fraction: float,
    rgb_effective_rank: float,
    corrected_effective_rank: float,
    base_metrics: Mapping[str, object],
    corrected_metrics: Mapping[str, object],
    corrected_transitions: Mapping[str, int],
    rgb_oof_metrics: Mapping[str, object],
    combined_oof_metrics: Mapping[str, object],
    rgb_val_metrics: Mapping[str, object],
    combined_val_metrics: Mapping[str, object],
    combined_transitions: Mapping[str, int],
    class1_error_delta_auc: float,
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_support_fraction": 0.15,
        "max_clipped_gain_fraction": 0.05,
        "min_corrected_to_rgb_effective_rank_ratio": 0.75,
        "min_class1_f1": 0.70,
        "max_macro_drop": 0.001,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_error_delta_auc": 0.60,
    }
    common = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "val_support_complete": int(val_samples) == int(thresholds["required_val_samples"]),
        "cache_alignment": bool(alignment_ok),
        "illuminant_support": float(mean_support_fraction)
        >= float(thresholds["min_support_fraction"]),
        "gain_not_clipped": float(clipped_gain_fraction)
        <= float(thresholds["max_clipped_gain_fraction"]),
        "corrected_rank_preserved": float(corrected_effective_rank)
        / max(float(rgb_effective_rank), 1e-8)
        >= float(thresholds["min_corrected_to_rgb_effective_rank_ratio"]),
    }
    direct_macro_drop = float(base_metrics["macro_f1"]) - float(corrected_metrics["macro_f1"])
    direct = {
        "direct_class1_milestone": float(corrected_metrics["focus_f1"])
        >= float(thresholds["min_class1_f1"]),
        "direct_macro_preserved": direct_macro_drop <= float(thresholds["max_macro_drop"]),
        "direct_net_corrections": int(corrected_transitions["corrections"])
        >= int(corrected_transitions["harms"]),
        "direct_class1_recall_protected": int(corrected_transitions["class1_fn_rescued"])
        >= int(corrected_transitions["class1_tp_broken"]),
        "direct_class1_fp_control": int(corrected_transitions["class1_fp_removed"])
        >= int(corrected_transitions["class1_fp_created"]),
    }
    oof_macro_gain = float(combined_oof_metrics["macro_f1"]) - float(rgb_oof_metrics["macro_f1"])
    oof_class1_gain = float(combined_oof_metrics["focus_f1"]) - float(rgb_oof_metrics["focus_f1"])
    val_macro_gain = float(combined_val_metrics["macro_f1"]) - float(rgb_val_metrics["macro_f1"])
    val_class1_gain = float(combined_val_metrics["focus_f1"]) - float(rgb_val_metrics["focus_f1"])
    fusion_macro_drop = float(base_metrics["macro_f1"]) - float(combined_val_metrics["macro_f1"])
    fusion = {
        "fusion_oof_macro_gain": oof_macro_gain >= float(thresholds["min_oof_macro_gain"]),
        "fusion_oof_class1_gain": oof_class1_gain >= float(thresholds["min_oof_class1_gain"]),
        "fusion_val_macro_gain": val_macro_gain >= float(thresholds["min_val_macro_gain"]),
        "fusion_val_class1_gain": val_class1_gain >= float(thresholds["min_val_class1_gain"]),
        "fusion_class1_milestone": float(combined_val_metrics["focus_f1"])
        >= float(thresholds["min_class1_f1"]),
        "fusion_macro_preserved": fusion_macro_drop <= float(thresholds["max_macro_drop"]),
        "fusion_net_corrections": int(combined_transitions["corrections"])
        >= int(combined_transitions["harms"]),
        "fusion_class1_recall_protected": int(combined_transitions["class1_fn_rescued"])
        >= int(combined_transitions["class1_tp_broken"]),
        "fusion_class1_fp_control": int(combined_transitions["class1_fp_removed"])
        >= int(combined_transitions["class1_fp_created"]),
        "fusion_error_direction_separable": float(class1_error_delta_auc)
        >= float(thresholds["min_error_delta_auc"]),
    }
    common_ready = all(common.values())
    direct_ready = common_ready and all(direct.values())
    fusion_ready = common_ready and all(fusion.values())
    return {
        "context_color_constancy_target_ready": bool(direct_ready or fusion_ready),
        "smoke_ready": bool(direct_ready or fusion_ready),
        "smoke_permission": bool(direct_ready or fusion_ready),
        "full_train_permission": False,
        "selected_path": "direct_preprocess" if direct_ready else ("dual_view_fusion" if fusion_ready else "none"),
        "common_checks": common,
        "direct_path_checks": direct,
        "fusion_path_checks": fusion,
        "failed_common_checks": [name for name, passed in common.items() if not passed],
        "failed_direct_checks": [name for name, passed in direct.items() if not passed],
        "failed_fusion_checks": [name for name, passed in fusion.items() if not passed],
        "observed": {
            "direct_macro_drop": direct_macro_drop,
            "oof_macro_gain": oof_macro_gain,
            "oof_class1_gain": oof_class1_gain,
            "val_macro_gain": val_macro_gain,
            "val_class1_gain": val_class1_gain,
            "fusion_macro_drop": fusion_macro_drop,
            "class1_error_delta_auc": float(class1_error_delta_auc),
            "rgb_effective_rank": float(rgb_effective_rank),
            "corrected_effective_rank": float(corrected_effective_rank),
            "corrected_to_rgb_effective_rank_ratio": float(corrected_effective_rank)
            / max(float(rgb_effective_rank), 1e-8),
        },
        "thresholds": thresholds,
    }


def _write_illuminant_rows(
    path: Path,
    *,
    labels: np.ndarray,
    sample_index: np.ndarray,
    paths: np.ndarray,
    payload: Mapping[str, object],
) -> None:
    illuminants = np.asarray(payload["illuminants"], dtype=np.float32)
    gains = np.asarray(payload["gains"], dtype=np.float32)
    support = np.asarray(payload["support_fraction"], dtype=np.float32)
    context_crop_angle = np.asarray(payload["context_crop_angle"], dtype=np.float32)
    neutral_angle = np.asarray(payload["neutral_angle"], dtype=np.float32)
    fields = [
        "sample_index",
        "path",
        "target",
        "illuminant_r",
        "illuminant_g",
        "illuminant_b",
        "gain_r",
        "gain_g",
        "gain_b",
        "support_fraction",
        "context_crop_angle_deg",
        "neutral_angle_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(labels.shape[0]):
            writer.writerow(
                {
                    "sample_index": int(sample_index[index]),
                    "path": str(paths[index]),
                    "target": int(labels[index]),
                    "illuminant_r": f"{float(illuminants[index, 0]):.10g}",
                    "illuminant_g": f"{float(illuminants[index, 1]):.10g}",
                    "illuminant_b": f"{float(illuminants[index, 2]):.10g}",
                    "gain_r": f"{float(gains[index, 0]):.10g}",
                    "gain_g": f"{float(gains[index, 1]):.10g}",
                    "gain_b": f"{float(gains[index, 2]):.10g}",
                    "support_fraction": f"{float(support[index]):.10g}",
                    "context_crop_angle_deg": f"{float(context_crop_angle[index]):.10g}",
                    "neutral_angle_deg": f"{float(neutral_angle[index]):.10g}",
                }
            )


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device).eval()
    mean, std = checkpoint_input_normalization(checkpoint)
    max_samples = {"train": int(args.max_train_samples), "val": int(args.max_val_samples)}
    cache_paths = {
        "train": Path(args.rgb_cache_dir) / "train_embeddings.npz",
        "val": Path(args.rgb_cache_dir) / "val_embeddings.npz",
    }
    rgb_cache = {
        split: _load_rgb_cache(path, max_samples=max_samples[split])
        for split, path in cache_paths.items()
    }

    start = time.perf_counter()
    payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    for split in ("train", "val"):
        dataset, class_names = _context_dataset(
            data=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples[split],
        )
        payloads[split] = _extract_split(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            mean=mean,
            std=std,
            gray_edge_sigma=float(args.gray_edge_sigma),
            minkowski_p=float(args.minkowski_p),
            bbox_exclusion_margin=float(args.bbox_exclusion_margin),
            min_gain=float(args.min_gain),
            max_gain=float(args.max_gain),
            split=split,
        )

    alignment: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val"):
        cached_labels = np.asarray(rgb_cache[split]["labels"], dtype=np.int64)
        labels = np.asarray(payloads[split]["labels"], dtype=np.int64)
        cached_paths = np.asarray(rgb_cache[split]["paths"], dtype=object)
        paths = np.asarray(payloads[split]["paths"], dtype=object)
        row_count_equal = int(cached_labels.shape[0]) == int(labels.shape[0])
        labels_equal = bool(np.array_equal(cached_labels, labels))
        paths_equal = bool(
            len(cached_paths) == len(paths)
            and all(_normalize_path(left) == _normalize_path(right) for left, right in zip(cached_paths, paths))
        )
        alignment[split] = {
            "rows": int(labels.shape[0]),
            "row_count_equal": row_count_equal,
            "labels_equal": labels_equal,
            "paths_equal": paths_equal,
        }
        if not (row_count_equal and labels_equal and paths_equal):
            raise ValueError(f"RGB/context-corrected alignment failed for {split}: {alignment[split]}")

    train_labels = np.asarray(rgb_cache["train"]["labels"], dtype=np.int64)
    val_labels = np.asarray(rgb_cache["val"]["labels"], dtype=np.int64)
    train_rgb = _l2_normalize(np.asarray(rgb_cache["train"]["embeddings"], dtype=np.float32))
    val_rgb = _l2_normalize(np.asarray(rgb_cache["val"]["embeddings"], dtype=np.float32))
    train_corrected = _l2_normalize(np.asarray(payloads["train"]["embeddings"], dtype=np.float32))
    val_corrected = _l2_normalize(np.asarray(payloads["val"]["embeddings"], dtype=np.float32))
    train_features = {
        "rgb": train_rgb,
        "corrected": train_corrected,
        "rgb_corrected": np.concatenate((train_rgb, train_corrected), axis=1),
    }
    val_features = {
        "rgb": val_rgb,
        "corrected": val_corrected,
        "rgb_corrected": np.concatenate((val_rgb, val_corrected), axis=1),
    }
    train_groups = _source_stems(np.asarray(rgb_cache["train"]["paths"], dtype=object))
    readouts: Dict[str, Dict[str, object]] = {}
    for name in ("rgb", "corrected", "rgb_corrected"):
        print(f"fitting grouped OOF readout: {name}", flush=True)
        result = _fit_grouped_readout(
            train_features=train_features[name],
            train_labels=train_labels,
            train_groups=train_groups,
            val_features=val_features[name],
            folds=int(args.folds),
            c_value=float(args.logistic_c),
            max_iter=int(args.logistic_max_iter),
            seed=42,
            class_names=class_names,
        )
        result["val_metrics"] = _classification_metrics(
            val_labels,
            np.asarray(result["val_probabilities"], dtype=np.float32),
            class_names=class_names,
        )
        readouts[name] = result

    base_train_probabilities = np.asarray(rgb_cache["train"]["probabilities"], dtype=np.float32)
    base_val_probabilities = np.asarray(rgb_cache["val"]["probabilities"], dtype=np.float32)
    corrected_val_probabilities = np.asarray(payloads["val"]["probabilities"], dtype=np.float32)
    combined_val_probabilities = np.asarray(readouts["rgb_corrected"]["val_probabilities"], dtype=np.float32)
    base_metrics = _classification_metrics(val_labels, base_val_probabilities, class_names=class_names)
    corrected_metrics = _classification_metrics(
        val_labels,
        corrected_val_probabilities,
        class_names=class_names,
    )
    corrected_transitions = _transition_summary(
        val_labels,
        base_val_probabilities,
        corrected_val_probabilities,
    )
    combined_transitions = _transition_summary(
        val_labels,
        base_val_probabilities,
        combined_val_probabilities,
    )

    base_predictions = base_val_probabilities.argmax(axis=1)
    recall_mask = (val_labels == 1) & (base_predictions != 1)
    false_positive_mask = (val_labels != 1) & (base_predictions == 1)
    error_mask = recall_mask | false_positive_mask
    error_labels = recall_mask[error_mask].astype(np.int64)
    delta_p1 = combined_val_probabilities[:, 1] - np.asarray(
        readouts["rgb"]["val_probabilities"], dtype=np.float32
    )[:, 1]
    class1_error_delta_auc = (
        float(roc_auc_score(error_labels, delta_p1[error_mask]))
        if int(np.unique(error_labels).size) == 2
        else 0.5
    )

    all_support = np.concatenate(
        [np.asarray(payloads[split]["support_fraction"], dtype=np.float32) for split in ("train", "val")]
    )
    all_clipped = np.concatenate(
        [np.asarray(payloads[split]["clipped_gain_fraction"], dtype=np.float32) for split in ("train", "val")]
    )
    rgb_effective_rank = _effective_rank(train_rgb)
    corrected_effective_rank = _effective_rank(train_corrected)
    alignment_ok = all(
        bool(item["row_count_equal"] and item["labels_equal"] and item["paths_equal"])
        for item in alignment.values()
    )
    gate = assess_context_color_constancy_readiness(
        train_samples=int(train_labels.shape[0]),
        val_samples=int(val_labels.shape[0]),
        alignment_ok=alignment_ok,
        mean_support_fraction=float(all_support.mean()),
        clipped_gain_fraction=float(all_clipped.mean()),
        rgb_effective_rank=rgb_effective_rank,
        corrected_effective_rank=corrected_effective_rank,
        base_metrics=base_metrics,
        corrected_metrics=corrected_metrics,
        corrected_transitions=corrected_transitions,
        rgb_oof_metrics=readouts["rgb"]["oof_metrics"],
        combined_oof_metrics=readouts["rgb_corrected"]["oof_metrics"],
        rgb_val_metrics=readouts["rgb"]["val_metrics"],
        combined_val_metrics=readouts["rgb_corrected"]["val_metrics"],
        combined_transitions=combined_transitions,
        class1_error_delta_auc=class1_error_delta_auc,
    )

    for split in ("train", "val"):
        _save_feature_cache(
            output_dir / f"{split}_context_corrected_embeddings.npz",
            payload=payloads[split],
            sample_index=np.asarray(rgb_cache[split]["sample_index"], dtype=np.int64),
            class_names=class_names,
        )
        base_probabilities = base_train_probabilities if split == "train" else base_val_probabilities
        variants = {
            "corrected_direct": np.asarray(payloads[split]["probabilities"], dtype=np.float32),
            "rgb_readout": np.asarray(
                readouts["rgb"]["oof_probabilities" if split == "train" else "val_probabilities"],
                dtype=np.float32,
            ),
            "corrected_readout": np.asarray(
                readouts["corrected"]["oof_probabilities" if split == "train" else "val_probabilities"],
                dtype=np.float32,
            ),
            "rgb_corrected_readout": np.asarray(
                readouts["rgb_corrected"]["oof_probabilities" if split == "train" else "val_probabilities"],
                dtype=np.float32,
            ),
        }
        _write_prediction_audit(
            output_dir / f"{split}_readout_predictions.csv",
            labels=np.asarray(rgb_cache[split]["labels"], dtype=np.int64),
            sample_index=np.asarray(rgb_cache[split]["sample_index"], dtype=np.int64),
            paths=np.asarray(rgb_cache[split]["paths"], dtype=object),
            base_probabilities=base_probabilities,
            variants=variants,
        )
        _write_illuminant_rows(
            output_dir / f"{split}_illuminant_audit.csv",
            labels=np.asarray(rgb_cache[split]["labels"], dtype=np.int64),
            sample_index=np.asarray(rgb_cache[split]["sample_index"], dtype=np.int64),
            paths=np.asarray(rgb_cache[split]["paths"], dtype=object),
            payload=payloads[split],
        )
    _write_preview(output_dir / "context_gray_edge_preview_train.png", payloads["train"]["preview"], class_names)

    compact_readouts = {
        name: {
            "feature_dim": int(train_features[name].shape[1]),
            "oof_metrics": result["oof_metrics"],
            "val_metrics": result["val_metrics"],
            "fold_iterations": result["fold_iterations"],
            "final_iterations": result["final_iterations"],
        }
        for name, result in readouts.items()
    }
    diagnostic_stats = {}
    for split in ("train", "val"):
        gains = np.asarray(payloads[split]["gains"], dtype=np.float32)
        diagnostic_stats[split] = {
            "support_fraction_mean": float(
                np.asarray(payloads[split]["support_fraction"], dtype=np.float32).mean()
            ),
            "support_fraction_p01": float(
                np.quantile(np.asarray(payloads[split]["support_fraction"], dtype=np.float32), 0.01)
            ),
            "neutral_angle_mean_deg": float(
                np.asarray(payloads[split]["neutral_angle"], dtype=np.float32).mean()
            ),
            "context_crop_angle_mean_deg": float(
                np.asarray(payloads[split]["context_crop_angle"], dtype=np.float32).mean()
            ),
            "gain_mean": gains.mean(axis=0).tolist(),
            "gain_p01": np.quantile(gains, 0.01, axis=0).tolist(),
            "gain_p99": np.quantile(gains, 0.99, axis=0).tolist(),
            "clipped_gain_fraction": float(
                np.asarray(payloads[split]["clipped_gain_fraction"], dtype=np.float32).mean()
            ),
        }
    summary = {
        "mode": "context_gray_edge_color_constancy_precheck",
        "guardrail": (
            "Raw yolo_f context estimates nuisance illumination; only the object crop is corrected. "
            "Frozen keeper plus grouped OOF train readouts, no test/model training/raw-data edit."
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "rgb_cache_dir": str(Path(args.rgb_cache_dir).resolve()),
        "protocol": {
            "estimator": "first_order_gray_edge",
            "sigma": float(args.gray_edge_sigma),
            "minkowski_p": float(args.minkowski_p),
            "bbox_exclusion_margin": float(args.bbox_exclusion_margin),
            "min_gain": float(args.min_gain),
            "max_gain": float(args.max_gain),
            "context_background_alpha": 1.0,
            "object_color_excluded_from_context_estimate": True,
            "rgb_branch_preserved": True,
            "candidate_sweep": False,
        },
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "source_group_count": int(np.unique(train_groups).size),
        "alignment": alignment,
        "class_names": class_names,
        "diagnostic_stats": diagnostic_stats,
        "rgb_effective_rank": rgb_effective_rank,
        "corrected_effective_rank": corrected_effective_rank,
        "corrected_to_rgb_effective_rank_ratio": corrected_effective_rank
        / max(rgb_effective_rank, 1e-8),
        "base_val_metrics": base_metrics,
        "corrected_direct_val_metrics": corrected_metrics,
        "readout": {
            "folds": int(args.folds),
            "logistic_c": float(args.logistic_c),
            "selection": "none; one predeclared Gray-Edge protocol and readout",
            "variants": compact_readouts,
        },
        "transitions": {
            "base_to_corrected_direct": corrected_transitions,
            "base_to_rgb_corrected_readout": combined_transitions,
            "base_class1_fn_support": int(recall_mask.sum()),
            "base_class1_fp_support": int(false_positive_mask.sum()),
            "fusion_class1_error_delta_auc": class1_error_delta_auc,
        },
        "gate": gate,
        "elapsed_seconds": float(time.perf_counter() - start),
        "literature": LITERATURE,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_written": False,
        "trainable_manifest_written": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    combined_metrics = compact_readouts["rgb_corrected"]["val_metrics"]
    readme = [
        "# Context Gray-Edge Color-Constancy Precheck",
        "",
        f"- Train/validation rows: `{train_labels.shape[0]}/{val_labels.shape[0]}`",
        f"- Base macro/class1 F1: `{float(base_metrics['macro_f1']):.6f}/{float(base_metrics['focus_f1']):.6f}`",
        f"- Corrected-direct macro/class1 F1: `{float(corrected_metrics['macro_f1']):.6f}/{float(corrected_metrics['focus_f1']):.6f}`",
        f"- RGB+corrected readout macro/class1 F1: `{float(combined_metrics['macro_f1']):.6f}/{float(combined_metrics['focus_f1']):.6f}`",
        f"- Mean illuminant support: `{float(all_support.mean()):.6f}`",
        f"- Smoke ready/path: `{str(bool(gate['smoke_ready'])).lower()}/{gate['selected_path']}`",
        "",
        "No test split, model training, checkpoint, trainable manifest, or raw-data edit was used.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

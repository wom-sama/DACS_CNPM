from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from trkh.evaluation.attention_viz import analyze_tensor, resolve_layer_index
from trkh.core.config import IMAGENET_MEAN, default_data_yaml, load_data_spec, to_serializable
from trkh.data.dataset import MangoYOLOCropDataset, PseudoVideoAugmenter, build_eval_transform
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.models.feature_hooks import count_attention_layers
from trkh.inference.inference import load_model
from trkh.models.model import (
    classification_logits_from_features,
    extract_bbox_from_model_output,
)
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    maybe_enable_dataset_image_cache,
    set_seed,
    summarize_token_norms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Danh gia robustness va fail-cases cho ViT-Registers.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default=None,
        help="Cach xu ly names trong data.yaml; dung raw cho dataset tuy bien hoac >4 lop.",
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=0,
        help="Neu > 0, validate so class trong data.yaml truoc robustness eval.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--num-fail-cases", type=int, default=3)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--head-reduction", choices=("mean", "max"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occlusion-ratio", type=float, default=0.22)
    return parser.parse_args()


def imagenet_fill(mean) -> tuple[int, int, int]:
    return tuple(int(round(channel * 255.0)) for channel in mean)


class CenterOcclusion:
    def __init__(self, ratio: float = 0.22, fill=(127, 127, 127)) -> None:
        self.ratio = ratio
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.array(image.convert("RGB"), copy=True)
        height, width = array.shape[:2]
        occ_width = max(1, int(round(width * self.ratio)))
        occ_height = max(1, int(round(height * self.ratio)))
        left = max(0, (width - occ_width) // 2)
        top = max(0, (height - occ_height) // 2)
        right = min(width, left + occ_width)
        bottom = min(height, top + occ_height)
        array[top:bottom, left:right] = np.array(self.fill, dtype=array.dtype)
        return Image.fromarray(array)


class LightingShift:
    def __init__(self, brightness: float = 1.0, contrast: float = 1.0) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(self.brightness)
        image = ImageEnhance.Contrast(image).enhance(self.contrast)
        return image


class IdentityCorruption:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.copy()


def _unpack_classification_sample(sample: object) -> Tuple[Image.Image, int, Mapping[str, object], torch.Tensor]:
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise ValueError(
            "Robustness evaluation requires classification samples as "
            "(PIL image, label, metadata)."
        )
    image, label, metadata = sample
    if not isinstance(image, Image.Image) or not isinstance(metadata, Mapping):
        raise ValueError("Invalid robustness classification sample types.")
    bbox = metadata.get("bbox")
    if not torch.is_tensor(bbox) or bbox.numel() != 4:
        raise ValueError("Robustness classification sample requires a four-value bbox tensor.")
    return image, int(label), metadata, bbox.reshape(4).to(dtype=torch.float32)


def _transform_classification_image(
    image: Image.Image,
    *,
    label: int,
    metadata: Mapping[str, object],
    bbox: torch.Tensor,
    transform,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    crop_bbox = metadata.get("crop_bbox")
    if not torch.is_tensor(crop_bbox) or crop_bbox.numel() != 4:
        crop_bbox = bbox
    target = {
        "labels": torch.tensor([int(label)], dtype=torch.long),
        "boxes": crop_bbox.reshape(1, 4).to(dtype=torch.float32),
    }
    transformed = transform(image, target=target)
    if not isinstance(transformed, (tuple, list)) or len(transformed) != 2:
        raise ValueError("Robustness transform must return (tensor, transformed_target).")
    tensor, transformed_target = transformed
    if not torch.is_tensor(tensor) or not isinstance(transformed_target, Mapping):
        raise ValueError("Invalid robustness transform output types.")
    transformed_boxes = transformed_target.get("boxes")
    if not torch.is_tensor(transformed_boxes) or transformed_boxes.ndim != 2 or transformed_boxes.size(0) < 1:
        raise ValueError("Robustness transform removed the classification bbox.")
    output_metadata = {
        "bbox": bbox.to(dtype=torch.float32),
        "crop_bbox": transformed_boxes[0].to(dtype=torch.float32),
    }
    image_mask = transformed_target.get("image_mask")
    if torch.is_tensor(image_mask):
        output_metadata["image_mask"] = image_mask.to(dtype=torch.bool)
    return tensor, output_metadata


def _forward_classification_with_metadata(
    model,
    images: torch.Tensor,
    metadata: Optional[Mapping[str, object]],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    bbox = metadata.get("bbox") if isinstance(metadata, Mapping) else None
    image_mask = metadata.get("image_mask") if isinstance(metadata, Mapping) else None
    if torch.is_tensor(bbox):
        bbox = bbox.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        bbox = None
    if torch.is_tensor(image_mask):
        image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    else:
        image_mask = None

    features = None
    if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
        features = model.forward_features(
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=bbox,
        )
        if bbox is not None:
            features["bbox"] = bbox
        output = model.forward_heads(features)
    elif hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
        features = model.forward_features(
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=bbox,
        )
        if bbox is not None:
            features["bbox"] = bbox
        output = classification_logits_from_features(model, features)
    else:
        output = model(images)
    logits, _ = extract_bbox_from_model_output(output)
    return logits, features


class CorruptedDataset(Dataset):
    def __init__(
        self,
        base_dataset: MangoYOLOCropDataset,
        corruption: Callable[[Image.Image], Image.Image],
        transform,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption = corruption
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def enable_image_cache(self, *args, **kwargs) -> None:
        enable_cache = getattr(self.base_dataset, "enable_image_cache", None)
        if callable(enable_cache):
            enable_cache(*args, **kwargs)

    def image_cache_stats(self) -> Dict[str, object]:
        stats_fn = getattr(self.base_dataset, "image_cache_stats", None)
        return stats_fn() if callable(stats_fn) else {}

    def __getitem__(self, index: int):
        image, label, targets, bbox = _unpack_classification_sample(self.base_dataset[index])
        image = self.corruption(image)
        tensor, transformed_metadata = _transform_classification_image(
            image,
            label=label,
            metadata=targets,
            bbox=bbox,
            transform=self.transform,
        )
        return tensor, label, transformed_metadata


def evaluate_condition(
    model,
    dataset: Dataset,
    class_names: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int,
) -> Dict[str, object]:
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=device.type == "cuda",
        context="robustness",
        prefetch_factor=2,
        persistent_workers=True,
    )
    cache_summary = maybe_enable_dataset_image_cache(
        dataset,
        enabled=int(dataloader_summary["effective_num_workers"]) == 0,
        context="robustness",
    )
    print(
        "DataLoader setup:",
        {
            "loader": dataloader_summary,
            "image_cache": cache_summary,
        },
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    all_targets = []
    all_predictions = []
    artifact_totals = {
        "patch_norm_mean": 0.0,
        "patch_norm_std": 0.0,
        "patch_norm_max": 0.0,
        "register_norm_mean": 0.0,
        "register_norm_max": 0.0,
        "register_to_patch_ratio": 0.0,
        "high_norm_patch_fraction": 0.0,
    }
    artifact_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images, labels = batch[0], batch[1]
            metadata = batch[2] if len(batch) == 3 and isinstance(batch[2], Mapping) else None
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, features = _forward_classification_with_metadata(
                model,
                images,
                metadata,
                device=device,
            )
            predictions = logits.argmax(dim=1)
            all_targets.append(labels.detach().cpu())
            all_predictions.append(predictions.detach().cpu())
            if features is not None:
                batch_stats = summarize_token_norms(features)
                for key, value in batch_stats.items():
                    artifact_totals[key] += float(value)
                if batch_stats:
                    artifact_batches += 1

    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    predictions = (
        torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
    )

    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        confusion[int(target), int(prediction)] += 1

    confusion_f = confusion.to(torch.float32)
    support = confusion_f.sum(dim=1)
    predicted_support = confusion_f.sum(dim=0)
    tp = confusion_f.diag()
    precision = tp / predicted_support.clamp(min=1.0)
    recall = tp / support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    accuracy = tp.sum() / confusion_f.sum().clamp(min=1.0)

    return {
        "accuracy": float(accuracy.item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_f1": float((f1 * support / support.sum().clamp(min=1.0)).sum().item()),
        "confusion_matrix": confusion.tolist(),
        "artifact_stats": {
            key: value / max(1, artifact_batches) for key, value in artifact_totals.items()
        },
    }


def collect_fail_cases(
    model,
    checkpoint,
    base_dataset: MangoYOLOCropDataset,
    corruption: Callable[[Image.Image], Image.Image],
    transform,
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
    num_fail_cases: int,
    layer_index: int,
    head_reduction: str,
    alpha: float,
) -> List[Dict[str, object]]:
    fail_cases = []

    for index in range(len(base_dataset)):
        if len(fail_cases) >= num_fail_cases:
            break

        crop_image, label, targets, bbox = _unpack_classification_sample(base_dataset[index])
        corrupted_image = corruption(crop_image)
        transformed_tensor, transformed_metadata = _transform_classification_image(
            corrupted_image,
            label=label,
            metadata=targets,
            bbox=bbox,
            transform=transform,
        )
        tensor = transformed_tensor.unsqueeze(0).to(device)
        batched_metadata = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in transformed_metadata.items()
        }
        with torch.no_grad():
            logits, _ = _forward_classification_with_metadata(
                model,
                tensor,
                batched_metadata,
                device=device,
            )
            probabilities = F.softmax(logits, dim=1)[0].detach().cpu()
            prediction = int(probabilities.argmax().item())

        if prediction == int(label):
            continue

        sample = base_dataset.samples[index]
        case_dir = ensure_dir(output_dir / f"case_{len(fail_cases) + 1:02d}")
        viz_summary = analyze_tensor(
            model=model,
            class_names=class_names,
            crop_image=corrupted_image,
            tensor=tensor,
            layer_index=layer_index,
            head_reduction=head_reduction,
            alpha=alpha,
            top_k=3,
            output_dir=case_dir,
            method="both",
            target_class=prediction,
        )
        record = {
            "image_path": str(sample.image_path.resolve()),
            "label_index": int(label),
            "label_name": class_names[int(label)],
            "prediction_index": prediction,
            "prediction_name": class_names[prediction],
            "prediction_probability": float(probabilities[prediction].item()),
            "bbox": [float(value) for value in bbox.tolist()],
            "viz": viz_summary,
        }
        json_dump(case_dir / "fail_case.json", to_serializable(record))
        fail_cases.append(record)

    return fail_cases


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(
        args.checkpoint,
        device,
        override_image_size=args.override_image_size,
    )
    image_size = int(args.override_image_size or checkpoint["model_config"]["image_size"])
    resize_mode = checkpoint.get("augmentation_config", {}).get("resize_mode", "pad")
    temporal_frames = int(checkpoint.get("model_config", {}).get("temporal_frames", 1))
    crop_margin_ratio = float(
        checkpoint.get("augmentation_config", {}).get("crop_margin_ratio", 0.05)
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.resolve().parent.parent / "robustness"
    ensure_dir(output_dir)

    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="val",
        transform=None,
        crop_margin_ratio=crop_margin_ratio,
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    augmentation_config = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    base_transform = build_eval_transform(
        image_size=image_size,
        resize_mode=resize_mode,
        illumination_normalization=bool(augmentation_config.get("illumination_normalization", False)),
        illumination_normalization_strength=float(augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0),
        foreground_crop_mode=str(augmentation_config.get("foreground_crop_mode", "none") or "none"),
        foreground_crop_margin_ratio=float(augmentation_config.get("foreground_crop_margin_ratio", 0.08) or 0.08),
        foreground_crop_min_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_min_mask_area_ratio", 0.03) or 0.03
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_max_mask_area_ratio", 0.92) or 0.92
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation_config.get("foreground_crop_max_crop_area_ratio", 0.98) or 0.98
        ),
        background_suppression_mode=str(augmentation_config.get("background_suppression_mode", "none") or "none"),
        background_suppression_margin=float(augmentation_config.get("background_suppression_margin", 0.08) or 0.08),
        background_suppression_blur_radius=float(augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0),
        surface_detail_amplification_mode=str(
            augmentation_config.get("surface_detail_amplification_mode", "none") or "none"
        ),
        surface_detail_amplification_strength=float(
            augmentation_config.get("surface_detail_amplification_strength", 0.0) or 0.0
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation_config.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation_config.get("surface_detail_amplification_foreground_weight", 0.85) or 0.85
        ),
        eval_surface_detail_amplification=bool(
            augmentation_config.get("eval_surface_detail_amplification", False)
        ),
    )
    if temporal_frames > 1:
        transform = PseudoVideoAugmenter(
            frame_transform=base_transform,
            temporal_frames=temporal_frames,
            deterministic=True,
        )
    else:
        transform = base_transform
    attention_depth = count_attention_layers(model)
    layer_index = resolve_layer_index(args.layer, attention_depth) if attention_depth > 0 else 0

    corruptions = {
        "clean": IdentityCorruption(),
        "occlusion_center": CenterOcclusion(
            ratio=args.occlusion_ratio,
            fill=imagenet_fill(IMAGENET_MEAN),
        ),
        "lighting_dim": LightingShift(brightness=0.7, contrast=0.9),
        "lighting_bright": LightingShift(brightness=1.25, contrast=1.1),
        "low_contrast": LightingShift(brightness=1.0, contrast=0.65),
    }

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "image_size": image_size,
        "temporal_frames": temporal_frames,
        "attention_depth": attention_depth,
        "conditions": {},
    }

    for condition_name, corruption in corruptions.items():
        condition_dir = ensure_dir(output_dir / condition_name)
        condition_dataset = CorruptedDataset(
            base_dataset=base_dataset,
            corruption=corruption,
            transform=transform,
        )
        metrics = evaluate_condition(
            model=model,
            dataset=condition_dataset,
            class_names=class_names,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
        )
        fail_cases = collect_fail_cases(
            model=model,
            checkpoint=checkpoint,
            base_dataset=base_dataset,
            corruption=corruption,
            transform=transform,
            class_names=class_names,
            device=device,
            output_dir=condition_dir / "fail_cases",
            num_fail_cases=args.num_fail_cases,
            layer_index=layer_index,
            head_reduction=args.head_reduction,
            alpha=args.alpha,
        )
        condition_payload = {
            "metrics": metrics,
            "fail_cases": fail_cases,
        }
        json_dump(condition_dir / "summary.json", to_serializable(condition_payload))
        payload["conditions"][condition_name] = condition_payload

    json_dump(output_dir / "robustness_summary.json", to_serializable(payload))
    print(json.dumps(to_serializable(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()

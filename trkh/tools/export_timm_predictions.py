from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from trkh.data.dataset import build_eval_transform
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predictions/probabilities from a TIMM ImageFolder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with train/val/test subdirectories.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required with --split test so the sealed test split cannot be opened accidentally.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for quick smoke/debug export. 0 means full split.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--weight-source",
        choices=("auto", "model", "ema"),
        default="auto",
        help=(
            "Select serialized model_state or EMA weights. auto reproduces the "
            "checkpoint's recorded validation/deployment weight source."
        ),
    )
    parser.add_argument(
        "--surface-pair-branch-off",
        action="store_true",
        help=(
            "For a trained Surface Pair Hybrid V3 checkpoint, evaluate the same "
            "co-adapted DINO backbone/head with only its pair residual disabled."
        ),
    )
    parser.add_argument(
        "--audit-surface-pair-routing",
        action="store_true",
        help=(
            "On a clean non-TTA V3 export, record full-split routing and actual "
            "residual/token telemetry."
        ),
    )
    parser.add_argument("--model", default="", help="Override model name if checkpoint args are missing.")
    parser.add_argument(
        "--tta-horizontal-flip",
        action="store_true",
        help="Average logits from the original image and a horizontal flip.",
    )
    parser.add_argument(
        "--tta-brightness-deltas",
        type=str,
        default="",
        help="Comma-separated RGB-space brightness deltas, e.g. -0.06,0.06. Applied after normalization is undone.",
    )
    parser.add_argument(
        "--tta-contrast-scales",
        type=str,
        default="",
        help="Comma-separated RGB-space contrast scales around 0.5, e.g. 0.9,1.1.",
    )
    parser.add_argument(
        "--tta-saturation-scales",
        type=str,
        default="",
        help="Comma-separated RGB-space saturation scales, e.g. 0.85,1.15.",
    )
    parser.add_argument(
        "--tta-gamma-values",
        type=str,
        default="",
        help="Comma-separated RGB-space gamma values; <1 brightens, >1 darkens, e.g. 0.9,1.1.",
    )
    parser.add_argument(
        "--tta-sharpness-amounts",
        type=str,
        default="",
        help="Comma-separated unsharp-mask amounts, e.g. 0.25,0.50.",
    )
    parser.add_argument(
        "--tta-zoom-scales",
        type=str,
        default="",
        help="Comma-separated center zoom scales applied on normalized tensors, e.g. 1.06,1.12.",
    )
    parser.add_argument(
        "--tta-spatial-crop-fractions",
        type=str,
        default="",
        help=(
            "Comma-separated crop fractions in (0,1). For each value, add five "
            "resized crops: top-left, top-right, center, bottom-left, bottom-right."
        ),
    )
    parser.add_argument(
        "--tta-channel-stretch-percentiles",
        type=str,
        default="",
        help=(
            "Comma-separated low:high percentile pairs, e.g. 0.01:0.99,0.03:0.97. "
            "Each variant performs per-image per-channel RGB contrast stretching."
        ),
    )
    parser.add_argument(
        "--tta-luma-stretch-percentiles",
        type=str,
        default="",
        help=(
            "Comma-separated low:high percentile pairs, e.g. 0.01:0.99. "
            "Each variant performs per-image luminance contrast stretching."
        ),
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("--split test requires the explicit --allow-test acknowledgement")
    return args


def _parse_float_list(value: str) -> List[float]:
    output: List[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        output.append(float(item))
    return output


def _parse_percentile_pairs(value: str) -> List[tuple[float, float]]:
    output: List[tuple[float, float]] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Percentile pair must use low:high format, got {item!r}.")
        low_text, high_text = item.split(":", 1)
        low = float(low_text.strip())
        high = float(high_text.strip())
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Percentiles must satisfy 0 <= low < high <= 1, got {item!r}.")
        output.append((low, high))
    return output


def build_transform_config(model_name: str) -> tuple[transforms.Compose, tuple[float, ...], tuple[float, ...]]:
    cfg = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=False))
    size = cfg.get("input_size", (3, 224, 224))[-2:]
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    transform = transforms.Compose(
        [
            transforms.Resize(size, antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ]
    )
    return transform, tuple(float(value) for value in mean), tuple(float(value) for value in std)


def build_transforms(model_name: str) -> transforms.Compose:
    transform, _, _ = build_transform_config(model_name)
    return transform


def _normalize_from_rgb(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images - mean) / std


def _rgb_from_normalized(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images * std + mean).clamp(0.0, 1.0)


def _center_zoom(images: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if scale <= 1.0 + 1e-6:
        return images
    height, width = int(images.shape[-2]), int(images.shape[-1])
    crop_height = max(1, min(height, int(round(height / scale))))
    crop_width = max(1, min(width, int(round(width / scale))))
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    cropped = images[..., top : top + crop_height, left : left + crop_width]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def _spatial_resized_crops(images: torch.Tensor, fraction: float) -> List[torch.Tensor]:
    fraction = float(fraction)
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"Spatial crop fraction must be in (0, 1), got {fraction}.")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    crop_height = max(1, min(height, int(round(height * fraction))))
    crop_width = max(1, min(width, int(round(width * fraction))))
    positions = [
        (0, 0),
        (0, width - crop_width),
        ((height - crop_height) // 2, (width - crop_width) // 2),
        (height - crop_height, 0),
        (height - crop_height, width - crop_width),
    ]
    crops: List[torch.Tensor] = []
    for top, left in positions:
        top = max(0, int(top))
        left = max(0, int(left))
        cropped = images[..., top : top + crop_height, left : left + crop_width]
        crops.append(F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False))
    return crops


def _percentile_stretch(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    flat = values.flatten(start_dim=2)
    q_low = torch.quantile(flat, float(low), dim=-1, keepdim=True).view(values.shape[0], values.shape[1], 1, 1)
    q_high = torch.quantile(flat, float(high), dim=-1, keepdim=True).view(values.shape[0], values.shape[1], 1, 1)
    scale = (q_high - q_low).clamp_min(1e-4)
    return ((values - q_low) / scale).clamp(0.0, 1.0)


def _tta_batches(
    images: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
    saturation_scales: Sequence[float],
    gamma_values: Sequence[float],
    sharpness_amounts: Sequence[float],
    zoom_scales: Sequence[float],
    spatial_crop_fractions: Sequence[float],
    channel_stretch_percentiles: Sequence[tuple[float, float]],
    luma_stretch_percentiles: Sequence[tuple[float, float]],
) -> List[torch.Tensor]:
    batches = [images]
    if horizontal_flip:
        batches.append(torch.flip(images, dims=(-1,)))

    for scale in zoom_scales:
        if abs(float(scale) - 1.0) <= 1e-12:
            continue
        batches.append(_center_zoom(images, float(scale)))

    for fraction in spatial_crop_fractions:
        batches.extend(_spatial_resized_crops(images, float(fraction)))

    if (
        brightness_deltas
        or contrast_scales
        or saturation_scales
        or gamma_values
        or sharpness_amounts
        or channel_stretch_percentiles
        or luma_stretch_percentiles
    ):
        mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        rgb_images = _rgb_from_normalized(images, mean_tensor, std_tensor)
        for delta in brightness_deltas:
            if abs(float(delta)) <= 1e-12:
                continue
            adjusted = (rgb_images + float(delta)).clamp(0.0, 1.0)
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
        for scale in contrast_scales:
            if abs(float(scale) - 1.0) <= 1e-12:
                continue
            adjusted = ((rgb_images - 0.5) * float(scale) + 0.5).clamp(0.0, 1.0)
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
        for scale in saturation_scales:
            if abs(float(scale) - 1.0) <= 1e-12:
                continue
            luminance = (
                rgb_images[:, 0:1] * 0.299
                + rgb_images[:, 1:2] * 0.587
                + rgb_images[:, 2:3] * 0.114
            )
            adjusted = (luminance + (rgb_images - luminance) * float(scale)).clamp(0.0, 1.0)
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
        for gamma in gamma_values:
            gamma = float(gamma)
            if gamma <= 0.0:
                raise ValueError(f"Gamma must be > 0, got {gamma}.")
            if abs(gamma - 1.0) <= 1e-12:
                continue
            adjusted = rgb_images.clamp(0.0, 1.0).pow(gamma)
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
        for amount in sharpness_amounts:
            amount = float(amount)
            if abs(amount) <= 1e-12:
                continue
            blurred = F.avg_pool2d(rgb_images, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            adjusted = (rgb_images + amount * (rgb_images - blurred)).clamp(0.0, 1.0)
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
        for low, high in channel_stretch_percentiles:
            adjusted = _percentile_stretch(rgb_images, float(low), float(high))
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
        for low, high in luma_stretch_percentiles:
            luminance = (
                rgb_images[:, 0:1] * 0.299
                + rgb_images[:, 1:2] * 0.587
                + rgb_images[:, 2:3] * 0.114
            )
            luma_flat = luminance.flatten(start_dim=2)
            q_low = torch.quantile(luma_flat, float(low), dim=-1, keepdim=True).view(luminance.shape[0], 1, 1, 1)
            q_high = torch.quantile(luma_flat, float(high), dim=-1, keepdim=True).view(luminance.shape[0], 1, 1, 1)
            scale = (q_high - q_low).clamp_min(1e-4)
            adjusted = ((rgb_images - q_low) / scale).clamp(0.0, 1.0)
            batches.append(_normalize_from_rgb(adjusted, mean_tensor, std_tensor))
    return batches


def _forward_tta(
    model: nn.Module,
    images: torch.Tensor,
    *,
    autocast_device: str,
    amp: bool,
    mean: Sequence[float],
    std: Sequence[float],
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
    saturation_scales: Sequence[float],
    gamma_values: Sequence[float],
    sharpness_amounts: Sequence[float],
    zoom_scales: Sequence[float],
    spatial_crop_fractions: Sequence[float],
    channel_stretch_percentiles: Sequence[tuple[float, float]],
    luma_stretch_percentiles: Sequence[tuple[float, float]],
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    tta_images = _tta_batches(
        images,
        mean=mean,
        std=std,
        horizontal_flip=horizontal_flip,
        brightness_deltas=brightness_deltas,
        contrast_scales=contrast_scales,
        saturation_scales=saturation_scales,
        gamma_values=gamma_values,
        sharpness_amounts=sharpness_amounts,
        zoom_scales=zoom_scales,
        spatial_crop_fractions=spatial_crop_fractions,
        channel_stretch_percentiles=channel_stretch_percentiles,
        luma_stretch_percentiles=luma_stretch_percentiles,
    )
    for variant_images in tta_images:
        with torch.autocast(device_type=autocast_device, enabled=bool(amp) and images.device.type == "cuda"):
            variant_logits = model(variant_images)
        logits_sum = variant_logits.float() if logits_sum is None else logits_sum + variant_logits.float()
    if logits_sum is None:
        raise RuntimeError("No TTA variants were generated.")
    return logits_sum / float(len(tta_images))


def metrics_from_predictions(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str]) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    confusion = [[0] * len(classes) for _ in classes]
    for target, prediction in zip(y_true, y_pred):
        confusion[int(target)][int(prediction)] += 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion,
        "per_class": [
            {
                "class_index": index,
                "class_name": str(classes[index]),
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(classes))
        ],
    }


def checkpoint_export_metadata(
    checkpoint: object,
    *,
    weight_source: str = "auto",
) -> Tuple[List[str], str, Mapping[str, torch.Tensor], Dict[str, str]]:
    """Resolve both legacy TIMM exports and current TRKH trainer checkpoints."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("TIMM export checkpoint must be a mapping.")

    raw_classes = checkpoint.get("classes") or checkpoint.get("class_names") or []
    classes = [str(value) for value in raw_classes]

    model_name = ""
    for container_key in ("args", "model_config"):
        container = checkpoint.get(container_key, {})
        if not isinstance(container, Mapping):
            continue
        for name_key in ("model", "timm_model_name"):
            candidate = str(container.get(name_key, "") or "").strip()
            if candidate:
                model_name = candidate
                break
        if model_name:
            break

    requested = str(weight_source).strip().lower()
    if requested not in {"auto", "model", "ema"}:
        raise ValueError(f"Unsupported weight_source: {weight_source!r}.")
    legacy_state = checkpoint.get("model")
    model_state = checkpoint.get("model_state")
    ema_state = checkpoint.get("ema_model_state")
    checkpoint_source = str(checkpoint.get("checkpoint_weight_source", "")).lower()
    validation_source = str(checkpoint.get("validation_weight_source", "")).lower()

    selected_key = ""
    if isinstance(legacy_state, Mapping) and not isinstance(model_state, Mapping):
        if requested == "ema":
            raise ValueError("Legacy checkpoint has no EMA weights.")
        selected_key = "model"
        state_dict = legacy_state
    elif requested == "model":
        selected_key = "model_state"
        state_dict = model_state
    elif requested == "ema":
        if checkpoint_source == "ema" and isinstance(model_state, Mapping):
            selected_key = "model_state"
            state_dict = model_state
        else:
            selected_key = "ema_model_state"
            state_dict = ema_state
    elif checkpoint_source == "ema" and isinstance(model_state, Mapping):
        selected_key = "model_state"
        state_dict = model_state
    elif validation_source == "ema" and isinstance(ema_state, Mapping):
        selected_key = "ema_model_state"
        state_dict = ema_state
    else:
        selected_key = "model_state"
        state_dict = model_state

    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint missing model/model_state weights.")
    selection = {
        "requested": requested,
        "resolved_state_key": selected_key,
        "checkpoint_weight_source": checkpoint_source,
        "validation_weight_source": validation_source,
    }
    return classes, model_name, state_dict, selection


def checkpoint_eval_transform(
    checkpoint: object,
    model_name: str,
) -> tuple[object, tuple[float, ...], tuple[float, ...], Dict[str, Any]]:
    """Rebuild the checkpoint's validation preprocessing when TRKH metadata exists."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("TIMM export checkpoint must be a mapping.")

    if "model_config" not in checkpoint:
        transform, mean, std = build_transform_config(model_name)
        return transform, mean, std, {
            "source": "timm_model_config",
            "input_mean": [float(value) for value in mean],
            "input_std": [float(value) for value in std],
        }

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("TRKH checkpoint model_config must be a mapping.")
    augmentation_config = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation_config, Mapping):
        raise ValueError("TRKH checkpoint augmentation_config must be a mapping.")

    image_size = int(model_config.get("image_size", 0) or 0)
    if image_size <= 0:
        raise ValueError("TRKH checkpoint model_config.image_size must be positive.")
    temporal_frames = int(model_config.get("temporal_frames", 1) or 1)
    if temporal_frames != 1:
        raise ValueError(
            "TIMM ImageFolder export currently requires model_config.temporal_frames=1; "
            f"got {temporal_frames}."
        )

    mean, std = checkpoint_input_normalization(checkpoint)

    def config_float(key: str, default: float) -> float:
        value = augmentation_config.get(key, default)
        return float(default if value is None else value)

    transform_config: Dict[str, Any] = {
        "image_size": image_size,
        "resize_mode": str(augmentation_config.get("resize_mode", "pad") or "pad"),
        "illumination_normalization": bool(
            augmentation_config.get("illumination_normalization", False)
        ),
        "illumination_normalization_strength": config_float(
            "illumination_normalization_strength", 0.0
        ),
        "foreground_crop_mode": str(
            augmentation_config.get("foreground_crop_mode", "none") or "none"
        ),
        "foreground_crop_margin_ratio": config_float(
            "foreground_crop_margin_ratio", 0.08
        ),
        "foreground_crop_min_mask_area_ratio": config_float(
            "foreground_crop_min_mask_area_ratio", 0.03
        ),
        "foreground_crop_max_mask_area_ratio": config_float(
            "foreground_crop_max_mask_area_ratio", 0.92
        ),
        "foreground_crop_max_crop_area_ratio": config_float(
            "foreground_crop_max_crop_area_ratio", 0.98
        ),
        "background_suppression_mode": str(
            augmentation_config.get("background_suppression_mode", "none") or "none"
        ),
        "background_suppression_margin": config_float(
            "background_suppression_margin", 0.08
        ),
        "background_suppression_blur_radius": config_float(
            "background_suppression_blur_radius", 7.0
        ),
        "surface_detail_amplification_mode": str(
            augmentation_config.get("surface_detail_amplification_mode", "none") or "none"
        ),
        "surface_detail_amplification_strength": config_float(
            "surface_detail_amplification_strength", 0.0
        ),
        "surface_detail_amplification_blur_radius": config_float(
            "surface_detail_amplification_blur_radius", 1.25
        ),
        "surface_detail_amplification_foreground_weight": config_float(
            "surface_detail_amplification_foreground_weight", 0.85
        ),
        "eval_surface_detail_amplification": bool(
            augmentation_config.get("eval_surface_detail_amplification", False)
        ),
    }
    transform = build_eval_transform(
        **transform_config,
        mean=mean,
        std=std,
    )
    preprocessing = {
        "source": "trkh_checkpoint",
        **transform_config,
        "input_mean": [float(value) for value in mean],
        "input_std": [float(value) for value in std],
    }
    return transform, mean, std, preprocessing


def align_imagefolder_class_order(
    dataset: ImageFolder,
    checkpoint_classes: Sequence[str],
) -> None:
    """Remap ImageFolder's alphabetical targets to the checkpoint label order."""
    expected = [str(value) for value in checkpoint_classes]
    observed = [str(value) for value in dataset.classes]
    if observed == expected:
        return
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError(
            "Class names differ between checkpoint and dataset: "
            f"checkpoint={expected}, dataset={observed}"
        )

    old_to_new = {
        int(old_index): int(expected.index(class_name))
        for old_index, class_name in enumerate(observed)
    }
    dataset.samples = [
        (path, old_to_new[int(target)]) for path, target in dataset.samples
    ]
    dataset.imgs = dataset.samples
    dataset.targets = [int(target) for _, target in dataset.samples]
    dataset.classes = expected
    dataset.class_to_idx = {
        class_name: class_index for class_index, class_name in enumerate(expected)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash the exact selected weights, independent of checkpoint packaging."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not torch.is_tensor(value):
            raise TypeError(f"State-dict entry {name!r} is not a tensor.")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, byteorder="little", signed=False))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, byteorder="little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def _sample_identity_hashes(
    samples: Sequence[Tuple[str, int]],
    targets: Sequence[int],
    *,
    data_root: Path,
) -> Dict[str, str]:
    if len(samples) != len(targets):
        raise ValueError("Prediction rows and dataset samples are misaligned.")
    root = Path(data_root).resolve()
    identity_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    for (raw_path, _), target in zip(samples, targets):
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Sample escaped the declared data root: {path}") from error
        identity_digest.update(f"{relative}\t{int(target)}\n".encode("utf-8"))
        content_digest.update(
            f"{relative}\t{int(target)}\t{_sha256(path)}\n".encode("utf-8")
        )
    return {
        "sample_identity_sha256": identity_digest.hexdigest(),
        "evaluated_split_content_sha256": content_digest.hexdigest(),
    }


def _distribution_summary(values: Sequence[np.ndarray]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Surface-pair routing audit contains non-finite values.")
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def build_export_model(
    checkpoint: Mapping[str, Any],
    *,
    model_name: str,
    num_classes: int,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[nn.Module, str]:
    """Rebuild current TRKH models exactly; retain the legacy TIMM path."""
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping) and str(
        model_config.get("model_type", "") or ""
    ).strip():
        resolved_checkpoint = dict(checkpoint)
        resolved_checkpoint["model_state"] = state_dict
        return (
            build_model_from_checkpoint(
                resolved_checkpoint,
                num_classes=num_classes,
            ),
            "trkh_checkpoint_architecture",
        )
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model, "legacy_timm_architecture"


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    (
        classes,
        checkpoint_model_name,
        state_dict,
        weight_selection,
    ) = checkpoint_export_metadata(
        checkpoint,
        weight_source=args.weight_source,
    )
    model_name = str(args.model or checkpoint_model_name).strip()
    if not model_name:
        raise ValueError("Cannot infer TIMM model name. Pass --model.")
    if not classes:
        raise ValueError(f"Checkpoint missing classes: {args.checkpoint}")

    transform, model_mean, model_std, preprocessing = checkpoint_eval_transform(
        checkpoint,
        model_name,
    )
    preprocessing_sha256 = _canonical_json_sha256(preprocessing)
    selected_state_dict_sha256 = _state_dict_sha256(state_dict)
    dataset = ImageFolder(str(args.data / args.split), transform=transform)
    align_imagefolder_class_order(dataset, classes)
    max_samples = max(0, int(args.max_samples))
    if max_samples > 0:
        dataset.samples = list(dataset.samples[:max_samples])
        dataset.imgs = dataset.samples
        dataset.targets = [int(target) for _, target in dataset.samples]

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.workers) > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_rebuild_source = build_export_model(
        checkpoint,
        model_name=model_name,
        num_classes=len(classes),
        state_dict=state_dict,
    )
    model.to(device).eval()

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    brightness_deltas = _parse_float_list(args.tta_brightness_deltas)
    contrast_scales = _parse_float_list(args.tta_contrast_scales)
    saturation_scales = _parse_float_list(args.tta_saturation_scales)
    gamma_values = _parse_float_list(args.tta_gamma_values)
    sharpness_amounts = _parse_float_list(args.tta_sharpness_amounts)
    zoom_scales = _parse_float_list(args.tta_zoom_scales)
    spatial_crop_fractions = _parse_float_list(args.tta_spatial_crop_fractions)
    channel_stretch_percentiles = _parse_percentile_pairs(args.tta_channel_stretch_percentiles)
    luma_stretch_percentiles = _parse_percentile_pairs(args.tta_luma_stretch_percentiles)
    model_config = checkpoint.get("model_config")
    model_config = dict(model_config) if isinstance(model_config, Mapping) else {}
    is_surface_pair_v3 = bool(
        getattr(model, "is_pretrained_surface_pair_hybrid_v3", False)
    )
    if bool(args.surface_pair_branch_off) and not is_surface_pair_v3:
        raise ValueError("--surface-pair-branch-off requires a V3 checkpoint.")
    tta_enabled = bool(
        args.tta_horizontal_flip
        or brightness_deltas
        or contrast_scales
        or saturation_scales
        or gamma_values
        or sharpness_amounts
        or zoom_scales
        or spatial_crop_fractions
        or channel_stretch_percentiles
        or luma_stretch_percentiles
    )
    if bool(args.audit_surface_pair_routing):
        if not is_surface_pair_v3:
            raise ValueError("--audit-surface-pair-routing requires a V3 checkpoint.")
        if bool(args.surface_pair_branch_off) or tta_enabled:
            raise ValueError(
                "Surface-pair routing audit requires the active branch and no TTA."
            )
    inference_model = model.backbone if bool(args.surface_pair_branch_off) else model
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities: List[List[float]] = []
    routing_values: Dict[str, List[np.ndarray]] = {
        "residual_to_token_ratio": [],
        "routing_strength_proxy": [],
        "absolute_signed_pair_coefficient": [],
        "competitor_entropy": [],
        "competitor_max_weight": [],
        "non_class3_mass": [],
    }
    routing_by_class: Dict[int, Dict[str, List[np.ndarray]]] = {
        class_index: {name: [] for name in routing_values}
        for class_index in range(len(classes))
    }
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"{model_name} {args.split}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if bool(args.audit_surface_pair_routing):
                with torch.autocast(
                    device_type=autocast_device,
                    enabled=bool(args.amp) and device.type == "cuda",
                ):
                    tokens, trace = model.forward_features_with_fusion_trace(images)
                    logits = model.forward_head(tokens)
                residual_ratio = trace["gated_residual_norm_ratio"].float()
                routing_strength = trace["pair_routing_strength"].float().squeeze(-1)
                signed_coefficients = trace["signed_pair_coefficients"].float()
                competitor_weights = trace["pair_competitor_weights"].float()
                competitor_entropy = -(
                    competitor_weights
                    * competitor_weights.clamp_min(1e-12).log()
                ).sum(dim=-1)
                audit_batch = {
                    "residual_to_token_ratio": residual_ratio,
                    "routing_strength_proxy": routing_strength,
                    "absolute_signed_pair_coefficient": signed_coefficients.abs(),
                    "competitor_entropy": competitor_entropy,
                    "competitor_max_weight": competitor_weights.amax(dim=-1),
                    "non_class3_mass": trace["pair_non_class3_mass"].float().squeeze(-1),
                }
                labels_cpu = labels.detach().cpu().tolist()
                for name, values in audit_batch.items():
                    values_cpu = values.detach().cpu().numpy()
                    routing_values[name].append(values_cpu)
                    flattened = values_cpu.reshape(values_cpu.shape[0], -1)
                    per_image = (
                        np.quantile(flattened, 0.95, axis=1)
                        if name == "residual_to_token_ratio"
                        else flattened.mean(axis=1)
                    )
                    for row_index, class_index in enumerate(labels_cpu):
                        routing_by_class[int(class_index)][name].append(
                            np.asarray([per_image[row_index]], dtype=np.float64)
                        )
            else:
                logits = _forward_tta(
                    inference_model,
                    images,
                    autocast_device=autocast_device,
                    amp=bool(args.amp),
                    mean=model_mean,
                    std=model_std,
                    horizontal_flip=bool(args.tta_horizontal_flip),
                    brightness_deltas=brightness_deltas,
                    contrast_scales=contrast_scales,
                    saturation_scales=saturation_scales,
                    gamma_values=gamma_values,
                    sharpness_amounts=sharpness_amounts,
                    zoom_scales=zoom_scales,
                    spatial_crop_fractions=spatial_crop_fractions,
                    channel_stretch_percentiles=channel_stretch_percentiles,
                    luma_stretch_percentiles=luma_stretch_percentiles,
                )
            loss = criterion(logits, labels)
            probs = logits.softmax(dim=1)
            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    elapsed = time.perf_counter() - start

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"predictions_{args.split}.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "y_true",
            "y_pred",
            "true_name",
            "pred_name",
            "confidence",
            *[f"prob_{index}" for index in range(len(classes))],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (path, _), target, prediction, probs in zip(dataset.samples, y_true, y_pred, probabilities):
            row = {
                "path": str(path),
                "y_true": int(target),
                "y_pred": int(prediction),
                "true_name": classes[int(target)],
                "pred_name": classes[int(prediction)],
                "confidence": float(probs[int(prediction)]),
            }
            row.update({f"prob_{index}": float(probs[index]) for index in range(len(classes))})
            writer.writerow(row)

    routing_audit: Dict[str, object] | None = None
    if bool(args.audit_surface_pair_routing):
        routing_audit = {
            "schema_version": 1,
            "scope": "full_export_split",
            "statistics": {
                name: _distribution_summary(values)
                for name, values in routing_values.items()
            },
            "per_true_class_image_level": {
                str(class_index): {
                    name: _distribution_summary(values)
                    for name, values in by_metric.items()
                }
                for class_index, by_metric in routing_by_class.items()
            },
        }

    train_config = checkpoint.get("train_config")
    train_config = dict(train_config) if isinstance(train_config, Mapping) else {}
    experiment_protocol_id = str(
        train_config.get(
            "experiment_protocol_id",
            checkpoint.get(
                "experiment_protocol_id",
                model_config.get("experiment_protocol_id", ""),
            ),
        )
    )
    sample_identity_hashes = _sample_identity_hashes(
        dataset.samples,
        y_true,
        data_root=args.data,
    )
    metrics = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "samples": len(y_true),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "inference_time_ms_per_image": float(elapsed * 1000.0 / max(1, len(y_true))),
        "model_rebuild_source": model_rebuild_source,
        "weight_selection": weight_selection,
        "selected_state_dict_sha256": selected_state_dict_sha256,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "prediction_file_sha256": _sha256(prediction_path),
        **sample_identity_hashes,
        "dataset_image_tree_sha256": str(
            train_config.get("dataset_image_tree_sha256", "")
        ).lower(),
        "model_type": str(model_config.get("model_type", "") or ""),
        "dinov3_surface_hybrid_mode": str(
            model_config.get("dinov3_surface_hybrid_mode", "") or ""
        ),
        "experiment_protocol_id": experiment_protocol_id,
        "surface_pair_branch_off": bool(args.surface_pair_branch_off),
        "surface_pair_routing_audit": routing_audit,
        "amp_enabled": bool(args.amp),
        "preprocessing": preprocessing,
        "preprocessing_sha256": preprocessing_sha256,
        "tta": {
            "horizontal_flip": bool(args.tta_horizontal_flip),
            "brightness_deltas": [float(value) for value in brightness_deltas],
            "contrast_scales": [float(value) for value in contrast_scales],
            "saturation_scales": [float(value) for value in saturation_scales],
            "gamma_values": [float(value) for value in gamma_values],
            "sharpness_amounts": [float(value) for value in sharpness_amounts],
            "zoom_scales": [float(value) for value in zoom_scales],
            "spatial_crop_fractions": [float(value) for value in spatial_crop_fractions],
            "channel_stretch_percentiles": [[float(low), float(high)] for low, high in channel_stretch_percentiles],
            "luma_stretch_percentiles": [[float(low), float(high)] for low, high in luma_stretch_percentiles],
        },
        "classes": classes,
        **metrics_from_predictions(y_true, y_pred, classes),
    }
    with (output_dir / f"metrics_{args.split}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(
        {
            "output_dir": str(output_dir),
            "split": args.split,
            "samples": len(y_true),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "predictions": str(prediction_path),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

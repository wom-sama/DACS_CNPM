from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predictions/probabilities from a TIMM ImageFolder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with train/val/test subdirectories.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
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
    return parser.parse_args()


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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
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


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    classes = list(checkpoint.get("classes", [])) if isinstance(checkpoint, dict) else []
    model_name = str(args.model or ckpt_args.get("model", "")).strip()
    if not model_name:
        raise ValueError("Cannot infer TIMM model name. Pass --model.")
    if not classes:
        raise ValueError(f"Checkpoint missing classes: {args.checkpoint}")

    transform, model_mean, model_std = build_transform_config(model_name)
    dataset = ImageFolder(str(args.data / args.split), transform=transform)
    if list(dataset.classes) != classes:
        raise ValueError(
            "Class order mismatch between checkpoint and dataset: "
            f"checkpoint={classes}, dataset={dataset.classes}"
        )
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
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint missing model state: {args.checkpoint}")
    model.load_state_dict(state_dict)
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
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities: List[List[float]] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"{model_name} {args.split}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = _forward_tta(
                model,
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

    metrics = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "samples": len(y_true),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "inference_time_ms_per_image": float(elapsed * 1000.0 / max(1, len(y_true))),
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

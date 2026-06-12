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
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    output: List[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        output.append(float(item))
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


def _tta_batches(
    images: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
) -> List[torch.Tensor]:
    batches = [images]
    if horizontal_flip:
        batches.append(torch.flip(images, dims=(-1,)))

    if brightness_deltas or contrast_scales:
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
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    tta_images = _tta_batches(
        images,
        mean=mean,
        std=std,
        horizontal_flip=horizontal_flip,
        brightness_deltas=brightness_deltas,
        contrast_scales=contrast_scales,
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

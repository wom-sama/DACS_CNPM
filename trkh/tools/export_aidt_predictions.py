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
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from tqdm import tqdm


class SquarePad:
    def __init__(self, fill: int = 0) -> None:
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return transforms.functional.pad(image, [left, top, right, bottom], fill=self.fill)


class RGBImageFolder(ImageFolder):
    def __init__(self, root: str, image_size: int) -> None:
        super().__init__(
            root=root,
            transform=transforms.Compose(
                [
                    SquarePad(),
                    transforms.Resize((int(image_size), int(image_size)), antialias=True),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            ),
        )


class ResNet50ViTEnsemble(nn.Module):
    def __init__(self, num_classes: int, resnet_name: str, vit_name: str) -> None:
        super().__init__()
        self.resnet = timm.create_model(resnet_name, pretrained=False, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        resnet_dim = int(self.resnet.num_features)
        vit_dim = int(self.vit.num_features)
        self.classifier = nn.Sequential(
            nn.Linear(resnet_dim + vit_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, resnet_images: torch.Tensor, vit_images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(resnet_images, vit_images))

    def forward_features(self, resnet_images: torch.Tensor, vit_images: torch.Tensor) -> torch.Tensor:
        resnet_features = self.resnet(resnet_images)
        vit_features = self.vit(vit_images)
        return torch.cat((resnet_features, vit_features), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export prediction probabilities from an AIDT ResNet50+ViT checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with train/val/test.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--tta-horizontal-flip", action="store_true")
    parser.add_argument("--tta-brightness-deltas", type=str, default="")
    parser.add_argument("--tta-contrast-scales", type=str, default="")
    parser.add_argument("--tta-saturation-scales", type=str, default="")
    parser.add_argument("--tta-gamma-values", type=str, default="")
    parser.add_argument("--tta-sharpness-amounts", type=str, default="")
    parser.add_argument("--tta-zoom-scales", type=str, default="")
    parser.add_argument(
        "--export-features-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ output for clean-view concatenated ResNet+ViT features. "
            "Probabilities may still use TTA; features are exported from the unaugmented view."
        ),
    )
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    output: List[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            output.append(float(item))
    return output


def _model_config(model_name: str, *, global_pool: str | None = None) -> Dict[str, object]:
    kwargs = {"pretrained": False, "num_classes": 0}
    if global_pool is not None:
        kwargs["global_pool"] = global_pool
    model = timm.create_model(model_name, **kwargs)
    cfg = timm.data.resolve_model_data_config(model)
    return {
        "mean": tuple(float(value) for value in cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(float(value) for value in cfg.get("std", (0.229, 0.224, 0.225))),
    }


def _normalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    return (images - mean_tensor) / std_tensor


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


def _tta_rgb_batches(
    images: torch.Tensor,
    *,
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
    saturation_scales: Sequence[float],
    gamma_values: Sequence[float],
    sharpness_amounts: Sequence[float],
    zoom_scales: Sequence[float],
) -> List[torch.Tensor]:
    rgb_images = images.clamp(0.0, 1.0)
    batches = [rgb_images]
    if horizontal_flip:
        batches.append(torch.flip(rgb_images, dims=(-1,)))
    for scale in zoom_scales:
        if abs(float(scale) - 1.0) > 1e-12:
            batches.append(_center_zoom(rgb_images, float(scale)))
    for delta in brightness_deltas:
        if abs(float(delta)) > 1e-12:
            batches.append((rgb_images + float(delta)).clamp(0.0, 1.0))
    for scale in contrast_scales:
        if abs(float(scale) - 1.0) > 1e-12:
            batches.append(((rgb_images - 0.5) * float(scale) + 0.5).clamp(0.0, 1.0))
    for scale in saturation_scales:
        if abs(float(scale) - 1.0) <= 1e-12:
            continue
        luminance = (
            rgb_images[:, 0:1] * 0.299
            + rgb_images[:, 1:2] * 0.587
            + rgb_images[:, 2:3] * 0.114
        )
        batches.append((luminance + (rgb_images - luminance) * float(scale)).clamp(0.0, 1.0))
    for gamma in gamma_values:
        gamma = float(gamma)
        if gamma <= 0.0:
            raise ValueError(f"Gamma must be > 0, got {gamma}.")
        if abs(gamma - 1.0) > 1e-12:
            batches.append(rgb_images.pow(gamma).clamp(0.0, 1.0))
    for amount in sharpness_amounts:
        amount = float(amount)
        if abs(amount) <= 1e-12:
            continue
        blurred = F.avg_pool2d(rgb_images, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        batches.append((rgb_images + amount * (rgb_images - blurred)).clamp(0.0, 1.0))
    return batches


def _forward_tta(
    model: nn.Module,
    images: torch.Tensor,
    *,
    autocast_device: str,
    amp: bool,
    amp_dtype: torch.dtype,
    resnet_mean: Sequence[float],
    resnet_std: Sequence[float],
    vit_mean: Sequence[float],
    vit_std: Sequence[float],
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
    saturation_scales: Sequence[float],
    gamma_values: Sequence[float],
    sharpness_amounts: Sequence[float],
    zoom_scales: Sequence[float],
) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    for rgb_variant in _tta_rgb_batches(
        images,
        horizontal_flip=horizontal_flip,
        brightness_deltas=brightness_deltas,
        contrast_scales=contrast_scales,
        saturation_scales=saturation_scales,
        gamma_values=gamma_values,
        sharpness_amounts=sharpness_amounts,
        zoom_scales=zoom_scales,
    ):
        resnet_images = _normalize(rgb_variant, resnet_mean, resnet_std)
        vit_images = _normalize(rgb_variant, vit_mean, vit_std)
        with torch.autocast(
            device_type=autocast_device,
            dtype=amp_dtype,
            enabled=bool(amp) and images.device.type == "cuda",
        ):
            logits = model(resnet_images, vit_images)
        logits_sum = logits.float() if logits_sum is None else logits_sum + logits.float()
    if logits_sum is None:
        raise RuntimeError("No TTA variants were generated.")
    return logits_sum / float(
        len(
            _tta_rgb_batches(
                images[:1],
                horizontal_flip=horizontal_flip,
                brightness_deltas=brightness_deltas,
                contrast_scales=contrast_scales,
                saturation_scales=saturation_scales,
                gamma_values=gamma_values,
                sharpness_amounts=sharpness_amounts,
                zoom_scales=zoom_scales,
            )
        )
    )


def _forward_clean_features(
    model: ResNet50ViTEnsemble,
    images: torch.Tensor,
    *,
    autocast_device: str,
    amp: bool,
    amp_dtype: torch.dtype,
    resnet_mean: Sequence[float],
    resnet_std: Sequence[float],
    vit_mean: Sequence[float],
    vit_std: Sequence[float],
) -> torch.Tensor:
    rgb_images = images.clamp(0.0, 1.0)
    resnet_images = _normalize(rgb_images, resnet_mean, resnet_std)
    vit_images = _normalize(rgb_images, vit_mean, vit_std)
    with torch.autocast(
        device_type=autocast_device,
        dtype=amp_dtype,
        enabled=bool(amp) and images.device.type == "cuda",
    ):
        return model.forward_features(resnet_images, vit_images).float()


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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = list(checkpoint.get("classes", [])) if isinstance(checkpoint, dict) else []
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not classes:
        raise ValueError(f"Checkpoint missing classes: {args.checkpoint}")
    resnet_name = str(ckpt_args.get("resnet", "resnet50.a1_in1k"))
    vit_name = str(ckpt_args.get("vit", "vit_base_patch16_224.augreg2_in21k_ft_in1k"))
    image_size = int(ckpt_args.get("image_size", 224))

    dataset = RGBImageFolder(str(args.data / args.split), image_size=image_size)
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
    model = ResNet50ViTEnsemble(len(classes), resnet_name=resnet_name, vit_name=vit_name)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint missing model state: {args.checkpoint}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    resnet_cfg = _model_config(resnet_name, global_pool="avg")
    vit_cfg = _model_config(vit_name)
    amp_dtype = torch.bfloat16 if str(args.amp_dtype) == "bf16" else torch.float16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    tta_options = {
        "horizontal_flip": bool(args.tta_horizontal_flip),
        "brightness_deltas": _parse_float_list(args.tta_brightness_deltas),
        "contrast_scales": _parse_float_list(args.tta_contrast_scales),
        "saturation_scales": _parse_float_list(args.tta_saturation_scales),
        "gamma_values": _parse_float_list(args.tta_gamma_values),
        "sharpness_amounts": _parse_float_list(args.tta_sharpness_amounts),
        "zoom_scales": _parse_float_list(args.tta_zoom_scales),
    }
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities: List[List[float]] = []
    feature_vectors: List[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"aidt {args.split}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = _forward_tta(
                model,
                images,
                autocast_device=autocast_device,
                amp=bool(args.amp),
                amp_dtype=amp_dtype,
                resnet_mean=resnet_cfg["mean"],
                resnet_std=resnet_cfg["std"],
                vit_mean=vit_cfg["mean"],
                vit_std=vit_cfg["std"],
                **tta_options,
            )
            loss = criterion(logits, labels)
            probs = logits.softmax(dim=1)
            if args.export_features_npz is not None:
                clean_features = _forward_clean_features(
                    model,
                    images,
                    autocast_device=autocast_device,
                    amp=bool(args.amp),
                    amp_dtype=amp_dtype,
                    resnet_mean=resnet_cfg["mean"],
                    resnet_std=resnet_cfg["std"],
                    vit_mean=vit_cfg["mean"],
                    vit_std=vit_cfg["std"],
                )
                feature_vectors.append(clean_features.cpu().numpy().astype(np.float32))
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
        "model": "aidt_resnet50_vit_b16",
        "checkpoint": str(args.checkpoint),
        "resnet": resnet_name,
        "vit": vit_name,
        "split": str(args.split),
        "samples": len(y_true),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "inference_time_ms_per_image": float(elapsed * 1000.0 / max(1, len(y_true))),
        "tta": tta_options,
        "classes": classes,
        **metrics_from_predictions(y_true, y_pred, classes),
    }
    with (output_dir / f"metrics_{args.split}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    feature_npz_path = None
    if args.export_features_npz is not None:
        feature_npz_path = Path(args.export_features_npz)
        feature_npz_path.parent.mkdir(parents=True, exist_ok=True)
        feature_array = (
            np.concatenate(feature_vectors, axis=0).astype(np.float32)
            if feature_vectors
            else np.zeros((0, 0), dtype=np.float32)
        )
        if int(feature_array.shape[0]) != len(dataset.samples):
            raise RuntimeError(
                "Feature export row count mismatch: "
                f"features={feature_array.shape[0]} samples={len(dataset.samples)}"
            )
        np.savez_compressed(
            feature_npz_path,
            features=feature_array,
            paths=np.asarray([str(path) for path, _ in dataset.samples], dtype=object),
            labels=np.asarray(y_true, dtype=np.int64),
            predictions=np.asarray(y_pred, dtype=np.int64),
            probabilities=np.asarray(probabilities, dtype=np.float32),
            classes=np.asarray(classes, dtype=object),
            feature_source=np.asarray(["aidt_resnet50_vit_concat_clean"], dtype=object),
        )
    print(
        {
            "output_dir": str(output_dir),
            "split": args.split,
            "samples": len(y_true),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "predictions": str(prediction_path),
            "features_npz": str(feature_npz_path) if feature_npz_path is not None else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

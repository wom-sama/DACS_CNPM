from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import load_data_spec, project_dir
from trkh.core.utils import load_checkpoint
from trkh.data.dataset import ClassificationFolderDataset, build_eval_transform
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe train-only ordinal maturity direction tren embedding cua checkpoint; "
            "tune logit scale tren val va chi danh gia test sau khi scale da chot."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--ordered-classes", type=str, default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-scale", type=float, default=0.6)
    parser.add_argument("--scale-steps", type=int, default=25)
    return parser.parse_args()


def _parse_ordered_classes(value: str, num_classes: int) -> List[int]:
    classes: List[int] = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        class_index = int(item)
        if class_index < 0 or class_index >= int(num_classes):
            raise ValueError(f"ordered class ngoai khoang: {class_index}")
        if class_index not in classes:
            classes.append(class_index)
    if len(classes) < 2:
        raise ValueError("--ordered-classes can it nhat 2 class.")
    return classes


def _build_dataset(data_spec, split: str, augmentation_config: Dict[str, object]):
    transform = build_eval_transform(
        image_size=int(augmentation_config.get("image_size", 0) or 0),
        resize_mode=str(augmentation_config.get("resize_mode", "pad") or "pad"),
        illumination_normalization=bool(
            augmentation_config.get("illumination_normalization", False)
        ),
        illumination_normalization_strength=float(
            augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0
        ),
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
        background_suppression_mode=str(
            augmentation_config.get("background_suppression_mode", "none") or "none"
        ),
        background_suppression_margin=float(
            augmentation_config.get("background_suppression_margin", 0.08) or 0.08
        ),
        background_suppression_blur_radius=float(
            augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0
        ),
    )
    return ClassificationFolderDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=transform,
        class_aware_augmentation=False,
    )


@torch.inference_mode()
def _extract_split(
    *,
    model,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    split: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=False,
        persistent_workers=False,
    )
    feature_parts: List[Tensor] = []
    logit_parts: List[Tensor] = []
    target_parts: List[Tensor] = []
    for images, targets in tqdm(loader, desc=f"Extract {split}", leave=False):
        images = images.to(device)
        features = model.forward_features(images)
        logits = classification_logits_from_features(model, features)
        head_input = extract_head_input_from_features(model, features)
        feature_parts.append(head_input.detach().float().cpu())
        logit_parts.append(logits.detach().float().cpu())
        target_parts.append(targets.detach().long().cpu())
    return (
        torch.cat(feature_parts, dim=0),
        torch.cat(logit_parts, dim=0),
        torch.cat(target_parts, dim=0),
    )


def _fit_ridge_ordinal(
    features: Tensor,
    targets: Tensor,
    ordered_classes: Sequence[int],
    ridge: float,
) -> Dict[str, Tensor]:
    valid = torch.zeros_like(targets, dtype=torch.bool)
    ordinal_targets = torch.zeros_like(targets, dtype=torch.float64)
    anchors = torch.arange(len(ordered_classes), dtype=torch.float64)
    anchors = anchors - anchors.mean()
    for rank, class_index in enumerate(ordered_classes):
        class_mask = targets == int(class_index)
        valid = valid | class_mask
        ordinal_targets[class_mask] = anchors[rank]
    x = features[valid].to(torch.float64)
    y = ordinal_targets[valid]
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp(min=1e-6)
    x = (x - mean) / std
    design = torch.cat((x, torch.ones((x.size(0), 1), dtype=x.dtype)), dim=1)
    regularizer = torch.eye(design.size(1), dtype=design.dtype) * max(0.0, float(ridge))
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(
        design.transpose(0, 1) @ design + regularizer,
        design.transpose(0, 1) @ y,
    )
    return {
        "mean": mean.float(),
        "std": std.float(),
        "weights": weights.float(),
        "anchors": anchors.float(),
    }


def _ordinal_scores(features: Tensor, probe: Dict[str, Tensor]) -> Tensor:
    normalized = (features.float() - probe["mean"]) / probe["std"]
    design = torch.cat(
        (normalized, torch.ones((normalized.size(0), 1), dtype=normalized.dtype)),
        dim=1,
    )
    return (design @ probe["weights"]).clamp(min=-3.0, max=3.0)


def _adjust_logits(
    logits: Tensor,
    scores: Tensor,
    ordered_classes: Sequence[int],
    anchors: Tensor,
    scale: float,
) -> Tensor:
    adjusted = logits.clone()
    for rank, class_index in enumerate(ordered_classes):
        adjusted[:, int(class_index)] += float(scale) * scores * anchors[rank]
    return adjusted


def _metrics(logits: Tensor, targets: Tensor, class_names: Sequence[str]) -> Dict[str, object]:
    probabilities = torch.softmax(logits.float(), dim=1)
    predictions = probabilities.argmax(dim=1)
    return build_metrics(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=int(args.expected_num_classes),
    )
    checkpoint_names = list(checkpoint.get("class_names", []))
    if checkpoint_names and checkpoint_names != list(data_spec.class_names):
        raise ValueError("Class names trong checkpoint khong khop dataset.")
    ordered_classes = _parse_ordered_classes(args.ordered_classes, data_spec.num_classes)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model_from_checkpoint(
        checkpoint,
        num_classes=data_spec.num_classes,
    ).to(device).eval()
    augmentation_config = dict(checkpoint.get("augmentation_config", {}))
    model_config = dict(checkpoint.get("model_config", {}))
    augmentation_config["image_size"] = int(model_config.get("image_size", 224))

    extracted: Dict[str, Tuple[Tensor, Tensor, Tensor]] = {}
    for split in ("train", "val", "test"):
        dataset = _build_dataset(data_spec, split, augmentation_config)
        extracted[split] = _extract_split(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            split=split,
        )

    train_features, _, train_targets = extracted["train"]
    probe = _fit_ridge_ordinal(
        train_features,
        train_targets,
        ordered_classes,
        ridge=args.ridge,
    )
    val_features, val_logits, val_targets = extracted["val"]
    val_scores = _ordinal_scores(val_features, probe)
    scales = torch.linspace(
        0.0,
        max(0.0, float(args.max_scale)),
        steps=max(2, int(args.scale_steps)),
    ).tolist()
    scale_rows: List[Dict[str, float]] = []
    best_scale = 0.0
    best_key = (-1.0, -1.0)
    for scale in scales:
        metrics = _metrics(
            _adjust_logits(
                val_logits,
                val_scores,
                ordered_classes,
                probe["anchors"],
                scale,
            ),
            val_targets,
            data_spec.class_names,
        )
        class1_f1 = float(metrics["per_class"][1]["f1"])
        macro_f1 = float(metrics["macro_f1"])
        scale_rows.append(
            {
                "scale": float(scale),
                "macro_f1": macro_f1,
                "class1_f1": class1_f1,
                "accuracy": float(metrics["accuracy"]),
            }
        )
        key = (macro_f1, class1_f1)
        if key > best_key:
            best_key = key
            best_scale = float(scale)

    _, test_logits, test_targets = extracted["test"]
    test_scores = _ordinal_scores(extracted["test"][0], probe)
    baseline_val = _metrics(val_logits, val_targets, data_spec.class_names)
    adjusted_val = _metrics(
        _adjust_logits(
            val_logits,
            val_scores,
            ordered_classes,
            probe["anchors"],
            best_scale,
        ),
        val_targets,
        data_spec.class_names,
    )
    baseline_test = _metrics(test_logits, test_targets, data_spec.class_names)
    adjusted_test = _metrics(
        _adjust_logits(
            test_logits,
            test_scores,
            ordered_classes,
            probe["anchors"],
            best_scale,
        ),
        test_targets,
        data_spec.class_names,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "val_scale_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scale", "macro_f1", "class1_f1", "accuracy"))
        writer.writeheader()
        writer.writerows(scale_rows)
    torch.save(probe, output_dir / "ordinal_ridge_probe.pt")
    summary = {
        "data_yaml": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "fit_split": "train",
        "scale_selection_split": "val",
        "final_report_split": "test",
        "ordered_classes": list(ordered_classes),
        "ridge": float(args.ridge),
        "selected_scale": float(best_scale),
        "baseline_val": baseline_val,
        "adjusted_val": adjusted_val,
        "baseline_test": baseline_test,
        "adjusted_test": adjusted_test,
        "score_summary": {
            split: {
                "mean": float(_ordinal_scores(values[0], probe).mean().item()),
                "std": float(_ordinal_scores(values[0], probe).std(unbiased=False).item()),
            }
            for split, values in extracted.items()
        },
        "elapsed_seconds": float(time.time() - started),
        "leakage_control": (
            "Ridge direction fit on train only; scale selected on val only; "
            "test metrics computed after scale selection."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "selected_scale": best_scale,
                "test_macro_f1_before": baseline_test["macro_f1"],
                "test_macro_f1_after": adjusted_test["macro_f1"],
                "test_class1_f1_before": baseline_test["per_class"][1]["f1"],
                "test_class1_f1_after": adjusted_test["per_class"][1]["f1"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

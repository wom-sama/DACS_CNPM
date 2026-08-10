from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.core.utils import autocast_context, ensure_dir, load_checkpoint, set_seed
from trkh.data.dataset import MangoSample
from trkh.evaluation.xai_audit import (
    _apply_masked_background,
    _build_dataset,
    _forward_logits_with_optional_bbox,
)
from trkh.inference.inference import load_model


@dataclass(frozen=True)
class GuardConfig:
    drop_threshold: float
    max_clean_margin: float
    max_clean_confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a narrow object-desaturation counterfactual guard for class-1 "
            "predictions. This is diagnostic only and does not edit raw data or checkpoints."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default=None)
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--focus-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="bbox")
    parser.add_argument(
        "--apply-config",
        type=str,
        default="",
        help="Optional fixed config drop,max_margin,max_confidence for applying to this split.",
    )
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _probability_columns(row: Mapping[str, object], prefix: str) -> List[Tuple[int, str]]:
    columns: List[Tuple[int, str]] = []
    expected = f"{prefix}_prob_"
    for key in row:
        text = str(key)
        if not text.startswith(expected):
            continue
        try:
            columns.append((int(text[len(expected) :]), text))
        except ValueError:
            continue
    return sorted(columns, key=lambda item: item[0])


def _row_probabilities(row: Mapping[str, object], prefix: str, num_classes: int) -> List[float]:
    values = [0.0 for _ in range(int(num_classes))]
    found = False
    for index, column in _probability_columns(row, prefix):
        if 0 <= index < int(num_classes):
            values[index] = _safe_float(row.get(column), 0.0)
            found = True
    if found:
        return values
    pred = _safe_int(row.get(f"{prefix}_pred"), -1)
    if 0 <= pred < int(num_classes):
        values[pred] = 1.0
    return values


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    num_classes: int,
) -> Dict[str, object]:
    confusion = [[0 for _ in range(int(num_classes))] for _ in range(int(num_classes))]
    for target, prediction in zip(targets, predictions):
        if 0 <= int(target) < int(num_classes) and 0 <= int(prediction) < int(num_classes):
            confusion[int(target)][int(prediction)] += 1
    per_class: List[Dict[str, float]] = []
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(int(num_classes)))
    for index in range(int(num_classes)):
        tp = float(confusion[index][index])
        support = float(sum(confusion[index]))
        predicted = float(sum(confusion[row][index] for row in range(int(num_classes))))
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class.append(
            {
                "class_index": int(index),
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return {
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "macro_f1": float(sum(item["f1"] for item in per_class) / max(1, int(num_classes))),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def apply_guard_predictions(
    rows: Sequence[Mapping[str, object]],
    *,
    config: GuardConfig,
    focus_class: int,
    num_classes: int,
) -> List[int]:
    output: List[int] = []
    for row in rows:
        clean_pred = _safe_int(row.get("clean_pred"), -1)
        desat_pred = _safe_int(row.get("desat_pred"), -1)
        clean_probs = _row_probabilities(row, "clean", num_classes)
        desat_probs = _row_probabilities(row, "desat", num_classes)
        clean_focus = float(clean_probs[int(focus_class)])
        desat_focus = float(desat_probs[int(focus_class)])
        drop = clean_focus - desat_focus
        margin = _safe_float(row.get("clean_margin"), 0.0)
        confidence = _safe_float(row.get("clean_confidence"), 0.0)
        route = (
            clean_pred == int(focus_class)
            and desat_pred != int(focus_class)
            and 0 <= desat_pred < int(num_classes)
            and drop >= float(config.drop_threshold)
            and margin <= float(config.max_clean_margin)
            and confidence <= float(config.max_clean_confidence)
        )
        output.append(int(desat_pred if route else clean_pred))
    return output


def _score_metrics(metrics: Mapping[str, object], *, focus_class: int, focus_weight: float) -> float:
    per_class = metrics.get("per_class", [])
    focus_f1 = 0.0
    if isinstance(per_class, list) and 0 <= int(focus_class) < len(per_class):
        item = per_class[int(focus_class)]
        if isinstance(item, Mapping):
            focus_f1 = _safe_float(item.get("f1"), 0.0)
    return _safe_float(metrics.get("macro_f1"), 0.0) + float(focus_weight) * focus_f1


def sweep_guard_configs(
    rows: Sequence[Mapping[str, object]],
    *,
    focus_class: int,
    num_classes: int,
    focus_weight: float = 0.25,
    drop_thresholds: Optional[Sequence[float]] = None,
    max_margins: Optional[Sequence[float]] = None,
    max_confidences: Optional[Sequence[float]] = None,
) -> Tuple[GuardConfig, Dict[str, object]]:
    if drop_thresholds is None:
        drop_thresholds = [round(value * 0.01, 4) for value in range(0, 21, 2)]
    if max_margins is None:
        max_margins = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 1.0]
    if max_confidences is None:
        max_confidences = [0.34, 0.38, 0.42, 0.46, 1.0]
    targets = [_safe_int(row.get("target_index"), -1) for row in rows]
    best_config = GuardConfig(0.0, 1.0, 1.0)
    best_metrics: Dict[str, object] = {}
    best_score = -1.0
    for drop in drop_thresholds:
        for margin in max_margins:
            for confidence in max_confidences:
                config = GuardConfig(float(drop), float(margin), float(confidence))
                preds = apply_guard_predictions(
                    rows,
                    config=config,
                    focus_class=focus_class,
                    num_classes=num_classes,
                )
                metrics = compute_classification_metrics(targets, preds, num_classes=num_classes)
                score = _score_metrics(metrics, focus_class=focus_class, focus_weight=focus_weight)
                if score > best_score:
                    best_score = float(score)
                    best_config = config
                    best_metrics = metrics
    return best_config, best_metrics


def _parse_config(text: str) -> Optional[GuardConfig]:
    if not str(text or "").strip():
        return None
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError("--apply-config must be 'drop,max_margin,max_confidence'.")
    return GuardConfig(float(parts[0]), float(parts[1]), float(parts[2]))


def _source_group_for_row(row: Mapping[str, object]) -> str:
    value = str(row.get("source_stem", "") or "").strip()
    if value:
        return value
    path = str(row.get("image_path", "") or "").strip()
    return Path(path).stem if path else "unknown"


def train_holdout_guard_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    focus_class: int,
    num_classes: int,
    focus_weight: float,
    folds: int,
    seed: int,
) -> Dict[str, object]:
    if len(rows) < 2:
        metrics = compute_classification_metrics([], [], num_classes=num_classes)
        return {"enabled": False, "reason": "not_enough_rows", "metrics": metrics}
    targets = [_safe_int(row.get("target_index"), -1) for row in rows]
    groups = [_source_group_for_row(row) for row in rows]
    unique_groups = len(set(groups))
    n_splits = max(2, min(int(folds), unique_groups, len(rows)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    oof_predictions = [-1 for _ in rows]
    fold_summaries: List[Dict[str, object]] = []
    for fold_index, (train_indices, holdout_indices) in enumerate(splitter.split(rows, targets, groups)):
        train_rows = [rows[int(index)] for index in train_indices]
        holdout_rows = [rows[int(index)] for index in holdout_indices]
        config, train_metrics = sweep_guard_configs(
            train_rows,
            focus_class=focus_class,
            num_classes=num_classes,
            focus_weight=focus_weight,
        )
        holdout_preds = apply_guard_predictions(
            holdout_rows,
            config=config,
            focus_class=focus_class,
            num_classes=num_classes,
        )
        for source_index, prediction in zip(holdout_indices, holdout_preds):
            oof_predictions[int(source_index)] = int(prediction)
        holdout_metrics = compute_classification_metrics(
            [_safe_int(row.get("target_index"), -1) for row in holdout_rows],
            holdout_preds,
            num_classes=num_classes,
        )
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "train_rows": int(len(train_rows)),
                "holdout_rows": int(len(holdout_rows)),
                "config": config.__dict__,
                "train_macro_f1": train_metrics["macro_f1"],
                "train_focus_f1": train_metrics["per_class"][int(focus_class)]["f1"],
                "holdout_macro_f1": holdout_metrics["macro_f1"],
                "holdout_focus_f1": holdout_metrics["per_class"][int(focus_class)]["f1"],
            }
        )
    valid_pairs = [(target, pred) for target, pred in zip(targets, oof_predictions) if int(pred) >= 0]
    oof_metrics = compute_classification_metrics(
        [target for target, _ in valid_pairs],
        [pred for _, pred in valid_pairs],
        num_classes=num_classes,
    )
    full_config, full_metrics = sweep_guard_configs(
        rows,
        focus_class=focus_class,
        num_classes=num_classes,
        focus_weight=focus_weight,
    )
    clean_metrics = compute_classification_metrics(
        targets,
        [_safe_int(row.get("clean_pred"), -1) for row in rows],
        num_classes=num_classes,
    )
    return {
        "enabled": True,
        "folds": int(n_splits),
        "clean_metrics": clean_metrics,
        "oof_metrics": oof_metrics,
        "folds_detail": fold_summaries,
        "full_train_config": full_config.__dict__,
        "full_train_metrics": full_metrics,
    }


def _clone_target(target: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for key, value in target.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _crop_image_and_target(dataset, index: int) -> Tuple[Image.Image, Dict[str, torch.Tensor], MangoSample]:
    sample = dataset.samples[int(index)]
    image = dataset._load_rgb_image(sample.image_path)
    labels = torch.tensor([obj.label for obj in sample.objects], dtype=torch.long)
    boxes = torch.tensor([obj.bbox for obj in sample.objects], dtype=torch.float32)
    if getattr(dataset, "crop_to_primary_object", False):
        image, target = dataset._crop_to_primary_object(image=image, labels=labels, boxes=boxes, sample=sample)
    else:
        target = {"labels": labels, "boxes": boxes}
    target["augmentation_scale"] = torch.tensor(
        [dataset._augmentation_scale_for_labels(target["labels"])],
        dtype=torch.float32,
    )
    return image.convert("RGB"), target, sample


def _transform_image(dataset, image: Image.Image, target: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if dataset.transform is None:
        raise ValueError("Counterfactual guard requires tensor eval transform.")
    transformed = dataset.transform(image, target=_clone_target(target))
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise TypeError("Expected eval transform to return (image_tensor, target).")
    tensor, transformed_target = transformed
    if not torch.is_tensor(tensor):
        raise TypeError("Eval transform did not return an image tensor.")
    return tensor, dict(transformed_target)


def _metadata_from_transformed(
    *,
    dataset,
    sample: MangoSample,
    transformed_target: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    primary_object = dataset._select_sample_primary_object(sample)
    metadata: Dict[str, torch.Tensor] = {
        "bbox": torch.tensor(primary_object.bbox, dtype=torch.float32),
    }
    transformed_boxes = transformed_target.get("boxes")
    if torch.is_tensor(transformed_boxes) and transformed_boxes.ndim == 2 and transformed_boxes.size(0) > 0:
        areas = transformed_boxes[:, 2].clamp(min=0.0) * transformed_boxes[:, 3].clamp(min=0.0)
        crop_bbox = transformed_boxes[int(torch.argmax(areas).item())]
    else:
        crop_bbox = torch.tensor((0.5, 0.5, 1.0, 1.0), dtype=torch.float32)
    metadata["crop_bbox"] = crop_bbox.to(dtype=torch.float32).clamp(0.0, 1.0)
    image_mask = transformed_target.get("image_mask")
    if torch.is_tensor(image_mask):
        metadata["image_mask"] = image_mask.to(dtype=torch.bool)
    return metadata


def _forward_batch(
    *,
    model,
    tensors: Sequence[torch.Tensor],
    metadata_items: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
    amp: bool,
    bbox_token_prior_source: str,
) -> torch.Tensor:
    images = torch.stack([tensor.to(dtype=torch.float32) for tensor in tensors], dim=0).to(device=device)
    bbox = torch.stack([item["bbox"] for item in metadata_items], dim=0).to(device=device, dtype=torch.float32)
    crop_bbox = torch.stack([item["crop_bbox"] for item in metadata_items], dim=0).to(device=device, dtype=torch.float32)
    masks = [item.get("image_mask") for item in metadata_items]
    image_mask = None
    if all(torch.is_tensor(mask) for mask in masks):
        image_mask = torch.stack([mask for mask in masks if torch.is_tensor(mask)], dim=0).to(
            device=device,
            dtype=torch.bool,
        )
    bbox_prior = crop_bbox if str(bbox_token_prior_source).lower() == "crop_bbox" else bbox
    with torch.no_grad():
        with autocast_context(device, amp):
            logits = _forward_logits_with_optional_bbox(
                model,
                images,
                bbox,
                bbox_prior,
                image_valid_mask=image_mask,
            )
    return F.softmax(logits.float(), dim=1).detach().cpu()


def _top2(probabilities: torch.Tensor) -> Tuple[int, float, int, float, float]:
    values, indices = torch.topk(probabilities, k=min(2, int(probabilities.numel())))
    top1_idx = int(indices[0].item())
    top1_value = float(values[0].item())
    top2_idx = int(indices[1].item()) if values.numel() > 1 else -1
    top2_value = float(values[1].item()) if values.numel() > 1 else 0.0
    return top1_idx, top1_value, top2_idx, top2_value, top1_value - top2_value


def collect_counterfactual_rows(
    *,
    model,
    checkpoint: Mapping[str, object],
    dataset,
    split: str,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_samples: int,
    amp: bool,
    bbox_token_prior_source: str,
) -> List[Dict[str, object]]:
    del checkpoint
    rows: List[Dict[str, object]] = []
    clean_tensors: List[torch.Tensor] = []
    desat_tensors: List[torch.Tensor] = []
    metadata_items: List[Dict[str, torch.Tensor]] = []
    pending_rows: List[Dict[str, object]] = []
    total = len(dataset) if int(max_samples) <= 0 else min(len(dataset), int(max_samples))
    model.eval()

    def flush() -> None:
        if not pending_rows:
            return
        clean_probs = _forward_batch(
            model=model,
            tensors=clean_tensors,
            metadata_items=metadata_items,
            device=device,
            amp=amp,
            bbox_token_prior_source=bbox_token_prior_source,
        )
        desat_probs = _forward_batch(
            model=model,
            tensors=desat_tensors,
            metadata_items=metadata_items,
            device=device,
            amp=amp,
            bbox_token_prior_source=bbox_token_prior_source,
        )
        for row, clean_prob, desat_prob in zip(pending_rows, clean_probs, desat_probs):
            clean_pred, clean_conf, clean_top2, clean_top2_conf, clean_margin = _top2(clean_prob)
            desat_pred, desat_conf, desat_top2, desat_top2_conf, desat_margin = _top2(desat_prob)
            row.update(
                {
                    "clean_pred": int(clean_pred),
                    "clean_confidence": float(clean_conf),
                    "clean_top2": int(clean_top2),
                    "clean_top2_confidence": float(clean_top2_conf),
                    "clean_margin": float(clean_margin),
                    "desat_pred": int(desat_pred),
                    "desat_confidence": float(desat_conf),
                    "desat_top2": int(desat_top2),
                    "desat_top2_confidence": float(desat_top2_conf),
                    "desat_margin": float(desat_margin),
                    "changed_prediction": int(clean_pred != desat_pred),
                }
            )
            for class_index in range(len(class_names)):
                row[f"clean_prob_{class_index}"] = float(clean_prob[class_index].item())
                row[f"desat_prob_{class_index}"] = float(desat_prob[class_index].item())
            row["focus_drop"] = float(
                row.get(f"clean_prob_{focus_class}", 0.0)
                - row.get(f"desat_prob_{focus_class}", 0.0)
            )
            rows.append(row)
        clean_tensors.clear()
        desat_tensors.clear()
        metadata_items.clear()
        pending_rows.clear()

    for index in range(total):
        crop_image, target, sample = _crop_image_and_target(dataset, index)
        desat_image = _apply_masked_background(crop_image, "object_desaturate")
        clean_tensor, clean_target = _transform_image(dataset, crop_image, target)
        desat_tensor, _ = _transform_image(dataset, desat_image, target)
        metadata = _metadata_from_transformed(dataset=dataset, sample=sample, transformed_target=clean_target)
        label = int(sample.primary_label)
        primary_object = dataset._select_sample_primary_object(sample)
        clean_tensors.append(clean_tensor)
        desat_tensors.append(desat_tensor)
        metadata_items.append(metadata)
        pending_rows.append(
            {
                "split": str(split),
                "sample_index": int(index),
                "image_path": str(sample.image_path),
                "source_stem": str(sample.image_path.stem),
                "object_index": int(primary_object.object_index),
                "target_index": int(label),
                "target_name": str(class_names[label]) if 0 <= label < len(class_names) else str(label),
            }
        )
        if len(pending_rows) >= max(1, int(batch_size)):
            flush()
    flush()
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _prediction_change_summary(rows: Sequence[Mapping[str, object]], *, num_classes: int) -> Dict[str, object]:
    clean_targets = [_safe_int(row.get("target_index"), -1) for row in rows]
    clean_preds = [_safe_int(row.get("clean_pred"), -1) for row in rows]
    desat_preds = [_safe_int(row.get("desat_pred"), -1) for row in rows]
    changed = [row for row in rows if _safe_int(row.get("clean_pred"), -1) != _safe_int(row.get("desat_pred"), -1)]
    correction = 0
    harm = 0
    for row in changed:
        target = _safe_int(row.get("target_index"), -1)
        clean = _safe_int(row.get("clean_pred"), -1)
        desat = _safe_int(row.get("desat_pred"), -1)
        if clean != target and desat == target:
            correction += 1
        if clean == target and desat != target:
            harm += 1
    return {
        "clean_metrics": compute_classification_metrics(clean_targets, clean_preds, num_classes=num_classes),
        "desat_metrics": compute_classification_metrics(clean_targets, desat_preds, num_classes=num_classes),
        "changed_predictions": int(len(changed)),
        "corrections": int(correction),
        "harms": int(harm),
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = load_checkpoint(args.checkpoint, map_location="cpu")
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
    )
    if data_spec.data_format != "yolo":
        raise ValueError("Counterfactual color guard diagnostic currently expects YOLO-format yolo_f data.")
    model, checkpoint, class_names = load_model(args.checkpoint, device)
    dataset = _build_dataset(
        data_yaml=args.data,
        classification_folder_yolo_data=None,
        split=args.split,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
        checkpoint=checkpoint,
    )
    output_dir = ensure_dir(args.output_dir)
    rows = collect_counterfactual_rows(
        model=model,
        checkpoint=checkpoint_payload,
        dataset=dataset,
        split=args.split,
        class_names=class_names,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        max_samples=max(0, int(args.max_samples)),
        amp=not bool(args.disable_amp),
        bbox_token_prior_source=args.bbox_token_prior_source,
    )
    predictions_csv = output_dir / f"counterfactual_color_guard_{args.split}.csv"
    _write_rows(predictions_csv, rows)
    num_classes = len(class_names)
    summary: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": str(args.split),
        "rows": int(len(rows)),
        "class_names": list(class_names),
        "prediction_csv": str(predictions_csv.resolve()),
        "change_summary": _prediction_change_summary(rows, num_classes=num_classes),
        "focus_class": int(args.focus_class),
        "sources": [
            "https://arxiv.org/abs/2006.16705",
            "https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf",
            "https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf",
        ],
    }
    fixed_config = _parse_config(args.apply_config)
    if fixed_config is not None:
        guarded_preds = apply_guard_predictions(
            rows,
            config=fixed_config,
            focus_class=int(args.focus_class),
            num_classes=num_classes,
        )
        targets = [_safe_int(row.get("target_index"), -1) for row in rows]
        summary["applied_config"] = fixed_config.__dict__
        summary["guarded_metrics"] = compute_classification_metrics(
            targets,
            guarded_preds,
            num_classes=num_classes,
        )
    if args.split == "train":
        summary["train_holdout_guard"] = train_holdout_guard_summary(
            rows,
            focus_class=int(args.focus_class),
            num_classes=num_classes,
            focus_weight=float(args.focus_weight),
            folds=int(args.folds),
            seed=int(args.seed),
        )
    else:
        config, metrics = sweep_guard_configs(
            rows,
            focus_class=int(args.focus_class),
            num_classes=num_classes,
            focus_weight=float(args.focus_weight),
        )
        summary["val_or_split_best_diagnostic"] = {
            "config": config.__dict__,
            "metrics": metrics,
            "note": "Diagnostic only; do not use as final rule unless selected from train/OOF evidence.",
        }
    summary_path = output_dir / f"counterfactual_color_guard_{args.split}_summary.json"
    summary_path.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")
    print(json.dumps(to_serializable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

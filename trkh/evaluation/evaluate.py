from __future__ import annotations

import argparse
import csv
import math
import time
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torchvision.ops import batched_nms as _torchvision_batched_nms
except Exception:  # pragma: no cover - optional compiled torchvision op
    _torchvision_batched_nms = None

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.training.debug_and_optimization import TestTimeAugmentation
from trkh.inference.inference import resolve_confidence_threshold
from trkh.training.loss import HybridDetectionClassificationLoss, box_iou_from_xywh, generalized_iou
from trkh.training.matcher import HungarianMatcher
from trkh.evaluation.metrics import (
    build_metrics,
    build_thresholded_acceptance_summary,
    plot_confusion_matrix,
    plot_detection_confidence_curve,
    plot_per_class_pr_panels,
    plot_per_class_metrics,
    plot_pr_curve,
)
from trkh.models.model import (
    build_model_from_checkpoint,
    extract_bbox_from_model_output,
    extract_detection_from_model_output,
    extract_head_input_from_features,
)
from trkh.core.utils import (
    autocast_context,
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    maybe_enable_dataset_image_cache,
    set_seed,
    summarize_token_norms,
)

DETECTION_MODEL_TYPES = {"detr_vit_registers", "vit_registers_hybrid"}
DETECTION_SCORE_MODES = (
    "foreground",
    "foreground_times_not_background",
    "foreground_minus_background",
    "class_only",
    "objectness",
    "class_sqrt_objectness",
    "objectness_sqrt_class",
    "class_objectness_mean",
    "class_objectness_min",
    "quality",
    "class_sqrt_quality",
    "quality_sqrt_class",
    "objectness_quality",
    "class_sqrt_objectness_quality",
)


def resolve_crop_to_primary_object(
    checkpoint: Dict[str, object],
    *,
    full_image_detection: bool = False,
    crop_to_primary_object: bool = False,
) -> bool:
    if full_image_detection and crop_to_primary_object:
        raise ValueError("Chi duoc chon mot trong --full-image-detection hoac --crop-to-primary-object.")
    if full_image_detection:
        return False
    if crop_to_primary_object:
        return True
    data_summary = checkpoint.get("data_summary", {})
    if isinstance(data_summary, dict) and "crop_to_primary_object" in data_summary:
        return bool(data_summary["crop_to_primary_object"])
    return True


def resolve_classification_object_crops(
    checkpoint: Dict[str, object],
    *,
    disable_classification_object_crops: bool = False,
) -> bool:
    if disable_classification_object_crops:
        return False
    data_summary = checkpoint.get("data_summary", {})
    if isinstance(data_summary, dict) and "classification_object_crops" in data_summary:
        return bool(data_summary["classification_object_crops"])
    return True


def _is_detection_targets(targets) -> bool:
    if isinstance(targets, list):
        return all(isinstance(item, dict) and "labels" in item and "boxes" in item for item in targets)
    return False


def _move_targets_to_device(targets, device: torch.device):
    return [
        {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _stack_image_masks_from_targets(targets) -> Optional[torch.Tensor]:
    if not _is_detection_targets(targets):
        return None
    masks = []
    for target in targets:
        image_mask = target.get("image_mask")
        if not torch.is_tensor(image_mask):
            return None
        masks.append(image_mask.to(dtype=torch.bool))
    if not masks:
        return None
    return torch.stack(masks, dim=0)


def _dataset_sample_paths(dataset) -> List[str]:
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    if callable(sample_paths_fn):
        return [str(path) for path in sample_paths_fn()]
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return []
    paths: List[str] = []
    for sample in samples:
        image_path = getattr(sample, "image_path", None)
        paths.append("" if image_path is None else str(image_path))
    return paths


def _build_prediction_records(
    *,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    class_names: Sequence[str],
    sample_paths: Sequence[str],
) -> List[Dict[str, object]]:
    targets = targets.detach().cpu().to(dtype=torch.long).view(-1)
    predictions = predictions.detach().cpu().to(dtype=torch.long).view(-1)
    probabilities = probabilities.detach().cpu().to(dtype=torch.float32)
    class_count = len(class_names)
    record_count = min(int(targets.numel()), int(predictions.numel()), int(probabilities.shape[0]))
    if sample_paths:
        record_count = min(record_count, len(sample_paths))

    records: List[Dict[str, object]] = []
    for sample_index in range(record_count):
        target_index = int(targets[sample_index].item())
        prediction_index = int(predictions[sample_index].item())
        row_probabilities = probabilities[sample_index]
        top_k = max(1, min(5, class_count, int(row_probabilities.numel())))
        top_values, top_indices = torch.topk(row_probabilities, k=top_k)
        record: Dict[str, object] = {
            "sample_index": int(sample_index),
            "image_path": str(sample_paths[sample_index]) if sample_paths else "",
            "target_index": target_index,
            "target_name": str(class_names[target_index]) if 0 <= target_index < class_count else "",
            "prediction_index": prediction_index,
            "prediction_name": str(class_names[prediction_index]) if 0 <= prediction_index < class_count else "",
            "confidence": float(row_probabilities[prediction_index].item())
            if 0 <= prediction_index < int(row_probabilities.numel())
            else 0.0,
            "correct": int(target_index == prediction_index),
        }
        for rank, (score, class_index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            class_index = int(class_index)
            record[f"top{rank}_index"] = class_index
            record[f"top{rank}_name"] = str(class_names[class_index]) if 0 <= class_index < class_count else ""
            record[f"top{rank}_probability"] = float(score)
        for class_index, class_name in enumerate(class_names):
            if class_index < int(row_probabilities.numel()):
                record[f"prob_{class_index}_{class_name}"] = float(row_probabilities[class_index].item())
        records.append(record)
    return records


def _classification_metric_summary(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Sequence[str],
) -> Dict[str, float]:
    metrics = build_metrics(
        targets=targets.to(dtype=torch.long).view(-1),
        predictions=predictions.to(dtype=torch.long).view(-1),
        class_names=class_names,
    )
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_precision": float(metrics.get("macro_precision", 0.0)),
        "macro_recall": float(metrics.get("macro_recall", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
    }


def _bootstrap_ci_from_labels(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Sequence[str],
    *,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> Dict[str, List[float]]:
    targets = targets.to(dtype=torch.long).view(-1).cpu()
    predictions = predictions.to(dtype=torch.long).view(-1).cpu()
    if int(targets.numel()) < 2 or int(predictions.numel()) < 2:
        return {}

    sample_count = min(int(targets.numel()), int(predictions.numel()))
    targets = targets[:sample_count]
    predictions = predictions[:sample_count]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values: Dict[str, List[float]] = {
        "accuracy": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
    }
    for _ in range(max(1, int(n_bootstrap))):
        indices = torch.randint(0, sample_count, (sample_count,), generator=generator)
        sampled = _classification_metric_summary(
            targets=targets[indices],
            predictions=predictions[indices],
            class_names=class_names,
        )
        for key in values:
            values[key].append(float(sampled[key]))

    intervals: Dict[str, List[float]] = {}
    for key, metric_values in values.items():
        tensor = torch.tensor(metric_values, dtype=torch.float32)
        intervals[key] = [
            float(torch.quantile(tensor, 0.025).item()),
            float(torch.quantile(tensor, 0.975).item()),
        ]
    return intervals


def _comparison_class_names_for_data_spec(
    data_spec,
    split: str,
    fallback_class_names: Sequence[str],
) -> List[str]:
    fallback = [str(name) for name in fallback_class_names]
    if getattr(data_spec, "data_format", "yolo") != "classification_folder":
        return fallback
    split_dir = data_spec.split_images_dir(split)
    if not split_dir.exists():
        return fallback
    folder_class_names = [
        path.name
        for path in sorted(split_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]
    if len(folder_class_names) == len(fallback) and set(folder_class_names) == set(fallback):
        return folder_class_names
    return fallback


def _record_class_name(
    record: Dict[str, object],
    *,
    name_key: str,
    index_key: str,
    class_names: Sequence[str],
) -> str:
    value = record.get(name_key, "")
    if isinstance(value, str) and value:
        return value
    try:
        index = int(record.get(index_key, -1))
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(class_names):
        return str(class_names[index])
    return ""


def _build_baseline_comparison_prediction_rows(
    prediction_records: Sequence[Dict[str, object]],
    *,
    class_names: Sequence[str],
    comparison_class_names: Sequence[str],
) -> List[Dict[str, object]]:
    class_to_index = {str(name): index for index, name in enumerate(comparison_class_names)}
    rows: List[Dict[str, object]] = []
    for record in prediction_records:
        if not isinstance(record, dict):
            continue
        true_name = _record_class_name(
            record,
            name_key="target_name",
            index_key="target_index",
            class_names=class_names,
        )
        pred_name = _record_class_name(
            record,
            name_key="prediction_name",
            index_key="prediction_index",
            class_names=class_names,
        )
        if true_name not in class_to_index or pred_name not in class_to_index:
            continue
        rows.append(
            {
                "path": str(record.get("image_path", "")),
                "y_true": int(class_to_index[true_name]),
                "y_pred": int(class_to_index[pred_name]),
                "true_name": true_name,
                "pred_name": pred_name,
            }
        )
    rows.sort(key=lambda item: str(item.get("path", "")))
    return rows


def _build_baseline_comparison_summary(
    *,
    rows: Sequence[Dict[str, object]],
    comparison_class_names: Sequence[str],
    metrics: Dict[str, object],
    model: nn.Module,
    checkpoint: Dict[str, object],
    paper_name: str,
    family: str,
    seed: int,
) -> Dict[str, object]:
    targets = torch.tensor([int(row["y_true"]) for row in rows], dtype=torch.long)
    predictions = torch.tensor([int(row["y_pred"]) for row in rows], dtype=torch.long)
    metric_summary = _classification_metric_summary(
        targets=targets,
        predictions=predictions,
        class_names=comparison_class_names,
    ) if len(rows) else {
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
    }
    timing = metrics.get("timing", {})
    eval_loop_seconds = float(timing.get("eval_loop_seconds", 0.0)) if isinstance(timing, dict) else 0.0
    model_config = checkpoint.get("model_config", {})
    model_name = ""
    if isinstance(model_config, dict):
        model_name = str(model_config.get("model_type", "")).strip()
    if not model_name:
        model_name = str(checkpoint.get("model_type", "") or model.__class__.__name__)
    best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", ""))
    return {
        "paper_name": str(paper_name),
        "family": str(family),
        "backend": "trkh",
        "model": model_name,
        "pretrained": False,
        "test_size": int(len(rows)),
        "test_loss": float(metrics.get("loss", 0.0)),
        "metrics": metric_summary,
        "ci95": _bootstrap_ci_from_labels(
            targets=targets,
            predictions=predictions,
            class_names=comparison_class_names,
            seed=seed,
        ) if len(rows) else {},
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "inference_time_ms_per_image": float(eval_loop_seconds / max(1, len(rows)) * 1000.0),
        "best_epoch": best_epoch,
        "classes": [str(name) for name in comparison_class_names],
    }


def _pairwise_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    boxes1 = boxes1.to(dtype=torch.float32)
    boxes2 = boxes2.to(dtype=torch.float32)
    boxes1_xyxy = torch.cat((boxes1[:, :2] - boxes1[:, 2:] / 2.0, boxes1[:, :2] + boxes1[:, 2:] / 2.0), dim=-1)
    boxes2_xyxy = torch.cat((boxes2[:, :2] - boxes2[:, 2:] / 2.0, boxes2[:, :2] + boxes2[:, 2:] / 2.0), dim=-1)
    boxes1_xyxy = boxes1_xyxy.clamp(0.0, 1.0)
    boxes2_xyxy = boxes2_xyxy.clamp(0.0, 1.0)

    top_left = torch.maximum(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    bottom_right = torch.minimum(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0.0).prod(dim=-1)
    area1 = (boxes1_xyxy[:, 2:] - boxes1_xyxy[:, :2]).clamp(min=0.0).prod(dim=-1)
    area2 = (boxes2_xyxy[:, 2:] - boxes2_xyxy[:, :2]).clamp(min=0.0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def _score_detection_queries(
    record: Dict[str, torch.Tensor],
    score_mode: str,
) -> torch.Tensor:
    score_mode = str(score_mode or "foreground").strip().lower()
    foreground_scores = record["scores"].to(dtype=torch.float32)
    objectness_scores = record.get("objectness_scores")
    quality_scores = record.get("quality_scores")
    class_scores = record.get("class_scores")
    if class_scores is None:
        class_scores = foreground_scores
    class_scores = class_scores.to(dtype=torch.float32).clamp(0.0, 1.0)
    if quality_scores is not None:
        quality_scores = quality_scores.to(dtype=torch.float32).clamp(0.0, 1.0)
        if score_mode == "quality":
            return quality_scores
        if score_mode == "class_sqrt_quality":
            return class_scores * torch.sqrt(quality_scores.clamp(min=0.0))
        if score_mode == "quality_sqrt_class":
            return quality_scores * torch.sqrt(class_scores.clamp(min=0.0))
    if objectness_scores is not None:
        objectness_scores = objectness_scores.to(dtype=torch.float32).clamp(0.0, 1.0)
        if score_mode == "class_only":
            return class_scores
        if score_mode == "objectness":
            return objectness_scores
        if score_mode == "class_sqrt_objectness":
            return class_scores * torch.sqrt(objectness_scores.clamp(min=0.0))
        if score_mode == "objectness_sqrt_class":
            return objectness_scores * torch.sqrt(class_scores.clamp(min=0.0))
        if score_mode == "class_objectness_mean":
            return 0.5 * (class_scores + objectness_scores)
        if score_mode == "class_objectness_min":
            return torch.minimum(class_scores, objectness_scores)
        if quality_scores is not None:
            if score_mode == "objectness_quality":
                return objectness_scores * quality_scores
            if score_mode == "class_sqrt_objectness_quality":
                return class_scores * torch.sqrt((objectness_scores * quality_scores).clamp(min=0.0))
        return foreground_scores
    if score_mode == "foreground_times_not_background":
        background_scores = record.get("background_scores")
        if background_scores is None:
            return foreground_scores
        return foreground_scores * (1.0 - background_scores.to(dtype=torch.float32).clamp(0.0, 1.0))
    if score_mode == "foreground_minus_background":
        background_scores = record.get("background_scores")
        if background_scores is None:
            return foreground_scores
        return (foreground_scores - background_scores.to(dtype=torch.float32) + 1.0) / 2.0
    return foreground_scores


def _extract_count_logits(model_output) -> Optional[torch.Tensor]:
    if isinstance(model_output, dict):
        return model_output.get("count_logits")
    if isinstance(model_output, (tuple, list)) and len(model_output) >= 4:
        return model_output[3]
    return None


def _extract_quality_logits(model_output) -> Optional[torch.Tensor]:
    if isinstance(model_output, dict):
        return model_output.get("quality_logits")
    if isinstance(model_output, (tuple, list)) and len(model_output) >= 5:
        return model_output[4]
    return None


def _resolve_adaptive_max_detections(
    record: Dict[str, torch.Tensor],
    *,
    adaptive_max_detections: bool = False,
    count_source: str = "auto",
    count_margin: int = 1,
    min_detections: int = 1,
    max_detections_per_image: Optional[int] = None,
) -> Optional[int]:
    hard_cap = max_detections_per_image
    if hard_cap is not None:
        hard_cap = max(1, int(hard_cap))
    if not adaptive_max_detections:
        return hard_cap

    source = str(count_source or "auto").strip().lower()
    count_value: Optional[float] = None
    if source in {"auto", "count_head"}:
        count_prediction = record.get("count_prediction")
        if count_prediction is not None:
            if torch.is_tensor(count_prediction):
                count_value = float(count_prediction.reshape(-1)[0].item())
            else:
                count_value = float(count_prediction)
    if count_value is None and source in {"auto", "objectness"}:
        objectness_scores = record.get("objectness_scores")
        if objectness_scores is not None:
            count_value = float(objectness_scores.to(dtype=torch.float32).clamp(0.0, 1.0).sum().item())
    if count_value is None:
        return hard_cap

    q_cap = int(record.get("scores", torch.empty(0)).numel())
    if q_cap <= 0:
        q_cap = hard_cap or 1
    upper_cap = min(q_cap, hard_cap) if hard_cap is not None else q_cap
    estimate = int(round(max(0.0, count_value))) + int(count_margin)
    return min(max(int(min_detections), estimate), max(1, upper_cap))


def _nms_query_indices(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: Optional[float],
    max_detections: Optional[int],
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    order = torch.argsort(scores.to(dtype=torch.float32), descending=True)
    if iou_threshold is None or float(iou_threshold) <= 0.0:
        if max_detections is not None:
            order = order[: max(1, int(max_detections))]
        return order.to(dtype=torch.long)

    boxes_xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2.0, boxes[:, :2] + boxes[:, 2:] / 2.0), dim=-1)
    boxes_xyxy = boxes_xyxy.to(dtype=torch.float32).clamp(0.0, 1.0)
    classes = classes.to(dtype=torch.long)
    if _torchvision_batched_nms is not None:
        keep = _torchvision_batched_nms(
            boxes_xyxy,
            scores.to(dtype=torch.float32),
            classes,
            float(iou_threshold),
        )
        if max_detections is not None:
            keep = keep[: max(1, int(max_detections))]
        return keep.to(dtype=torch.long, device=boxes.device)

    kept: List[int] = []
    for index in order.tolist():
        should_keep = True
        for kept_index in kept:
            if int(classes[index].item()) != int(classes[kept_index].item()):
                continue
            iou = _pairwise_iou_xywh(boxes[index : index + 1], boxes[kept_index : kept_index + 1])
            if float(iou.item()) > float(iou_threshold):
                should_keep = False
                break
        if should_keep:
            kept.append(int(index))
            if max_detections is not None and len(kept) >= max(1, int(max_detections)):
                break
    return torch.as_tensor(kept, dtype=torch.long)


def _compute_detection_metrics_at_threshold(
    records: Sequence[Dict[str, torch.Tensor]],
    threshold: float,
    iou_threshold: float = 0.5,
    nms_iou_threshold: Optional[float] = 0.5,
    max_detections_per_image: Optional[int] = None,
    adaptive_max_detections: bool = False,
    adaptive_count_source: str = "auto",
    adaptive_count_margin: int = 1,
    adaptive_min_detections: int = 1,
    require_foreground_argmax: bool = False,
    score_mode: str = "foreground",
) -> Dict[str, object]:
    predicted_objects = 0
    ground_truth_objects = 0
    true_positive = 0

    for record in records:
        scores = _score_detection_queries(record, score_mode=score_mode)
        classes = record["classes"].to(dtype=torch.long)
        boxes = record["boxes"].to(dtype=torch.float32)
        target_labels = record["target_labels"].to(dtype=torch.long)
        target_boxes = record["target_boxes"].to(dtype=torch.float32)
        ground_truth_objects += int(target_labels.numel())

        keep = scores >= float(threshold)
        if require_foreground_argmax and "is_background_argmax" in record:
            keep = keep & (~record["is_background_argmax"].to(dtype=torch.bool))
        if keep.sum().item() == 0:
            continue
        kept_scores = scores[keep]
        kept_classes = classes[keep]
        kept_boxes = boxes[keep]
        resolved_max_detections = _resolve_adaptive_max_detections(
            record,
            adaptive_max_detections=adaptive_max_detections,
            count_source=adaptive_count_source,
            count_margin=adaptive_count_margin,
            min_detections=adaptive_min_detections,
            max_detections_per_image=max_detections_per_image,
        )
        order = _nms_query_indices(
            kept_boxes,
            kept_classes,
            kept_scores,
            iou_threshold=nms_iou_threshold,
            max_detections=resolved_max_detections,
        )
        kept_scores = kept_scores[order]
        kept_classes = kept_classes[order]
        kept_boxes = kept_boxes[order]
        predicted_objects += int(kept_classes.numel())

        if target_labels.numel() == 0:
            continue
        iou_matrix = _pairwise_iou_xywh(kept_boxes, target_boxes)
        matched_targets = torch.zeros((target_labels.numel(),), dtype=torch.bool)
        for prediction_index in range(kept_classes.numel()):
            same_class = target_labels == kept_classes[prediction_index]
            available = same_class & (~matched_targets)
            if not available.any():
                continue
            candidate_indices = torch.nonzero(available, as_tuple=False).flatten()
            candidate_iou = iou_matrix[prediction_index, candidate_indices]
            best_local = int(torch.argmax(candidate_iou).item())
            best_target = int(candidate_indices[best_local].item())
            if float(candidate_iou[best_local].item()) >= float(iou_threshold):
                matched_targets[best_target] = True
                true_positive += 1

    precision = true_positive / float(max(1, predicted_objects))
    recall = true_positive / float(max(1, ground_truth_objects))
    f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "predicted_objects": int(predicted_objects),
        "ground_truth_objects": int(ground_truth_objects),
        "true_positive_50": int(true_positive),
        "precision_50": float(precision),
        "recall_50": float(recall),
        "f1_50": float(f1),
        "nms_iou_threshold": None if nms_iou_threshold is None else float(nms_iou_threshold),
        "max_detections_per_image": None
        if max_detections_per_image is None
        else int(max_detections_per_image),
        "adaptive_max_detections": bool(adaptive_max_detections),
        "adaptive_count_source": str(adaptive_count_source),
        "adaptive_count_margin": int(adaptive_count_margin),
        "adaptive_min_detections": int(adaptive_min_detections),
        "require_foreground_argmax": bool(require_foreground_argmax),
        "score_mode": str(score_mode),
    }


def _prepare_detection_records_for_confidence_curve(
    records: Sequence[Dict[str, torch.Tensor]],
    nms_iou_threshold: Optional[float] = 0.5,
    max_detections_per_image: Optional[int] = None,
    adaptive_max_detections: bool = False,
    adaptive_count_source: str = "auto",
    adaptive_count_margin: int = 1,
    adaptive_min_detections: int = 1,
    require_foreground_argmax: bool = False,
    score_mode: str = "foreground",
) -> List[Dict[str, torch.Tensor]]:
    prepared: List[Dict[str, torch.Tensor]] = []
    for record in records:
        scores = _score_detection_queries(record, score_mode=score_mode)
        classes = record["classes"].to(dtype=torch.long)
        boxes = record["boxes"].to(dtype=torch.float32)
        target_labels = record["target_labels"].to(dtype=torch.long)
        target_boxes = record["target_boxes"].to(dtype=torch.float32)

        keep = torch.ones_like(scores, dtype=torch.bool)
        if require_foreground_argmax and "is_background_argmax" in record:
            keep = keep & (~record["is_background_argmax"].to(dtype=torch.bool))

        kept_scores = scores[keep].to(dtype=torch.float32)
        kept_classes = classes[keep]
        kept_boxes = boxes[keep]
        resolved_max_detections = _resolve_adaptive_max_detections(
            record,
            adaptive_max_detections=adaptive_max_detections,
            count_source=adaptive_count_source,
            count_margin=adaptive_count_margin,
            min_detections=adaptive_min_detections,
            max_detections_per_image=max_detections_per_image,
        )
        if kept_scores.numel() > 0:
            order = _nms_query_indices(
                kept_boxes,
                kept_classes,
                kept_scores,
                iou_threshold=nms_iou_threshold,
                max_detections=resolved_max_detections,
            )
            kept_scores = kept_scores[order]
            kept_classes = kept_classes[order]
            kept_boxes = kept_boxes[order]

        if kept_boxes.numel() > 0 and target_boxes.numel() > 0:
            iou_matrix = _pairwise_iou_xywh(kept_boxes, target_boxes)
        else:
            iou_matrix = torch.zeros((kept_boxes.shape[0], target_labels.numel()), dtype=torch.float32)
        prepared.append(
            {
                "scores": kept_scores,
                "classes": kept_classes,
                "boxes": kept_boxes,
                "target_labels": target_labels,
                "target_boxes": target_boxes,
                "iou_matrix": iou_matrix,
                "resolved_max_detections": torch.tensor(
                    -1 if resolved_max_detections is None else int(resolved_max_detections),
                    dtype=torch.long,
                ),
            }
        )
    return prepared


def _compute_prepared_detection_metrics_at_threshold(
    prepared_records: Sequence[Dict[str, torch.Tensor]],
    threshold: float,
    iou_threshold: float = 0.5,
    nms_iou_threshold: Optional[float] = 0.5,
    max_detections_per_image: Optional[int] = None,
    adaptive_max_detections: bool = False,
    adaptive_count_source: str = "auto",
    adaptive_count_margin: int = 1,
    adaptive_min_detections: int = 1,
    require_foreground_argmax: bool = False,
    score_mode: str = "foreground",
) -> Dict[str, object]:
    predicted_objects = 0
    ground_truth_objects = 0
    true_positive = 0

    for record in prepared_records:
        scores = record["scores"].to(dtype=torch.float32)
        classes = record["classes"].to(dtype=torch.long)
        target_labels = record["target_labels"].to(dtype=torch.long)
        iou_matrix = record["iou_matrix"].to(dtype=torch.float32)
        ground_truth_objects += int(target_labels.numel())

        keep = scores >= float(threshold)
        if keep.sum().item() == 0:
            continue
        kept_classes = classes[keep]
        kept_iou = iou_matrix[keep]
        predicted_objects += int(kept_classes.numel())

        if target_labels.numel() == 0:
            continue
        matched_targets = torch.zeros((target_labels.numel(),), dtype=torch.bool)
        for prediction_index in range(kept_classes.numel()):
            same_class = target_labels == kept_classes[prediction_index]
            available = same_class & (~matched_targets)
            if not available.any():
                continue
            candidate_indices = torch.nonzero(available, as_tuple=False).flatten()
            candidate_iou = kept_iou[prediction_index, candidate_indices]
            best_local = int(torch.argmax(candidate_iou).item())
            best_target = int(candidate_indices[best_local].item())
            if float(candidate_iou[best_local].item()) >= float(iou_threshold):
                matched_targets[best_target] = True
                true_positive += 1

    precision = true_positive / float(max(1, predicted_objects))
    recall = true_positive / float(max(1, ground_truth_objects))
    f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "predicted_objects": int(predicted_objects),
        "ground_truth_objects": int(ground_truth_objects),
        "true_positive_50": int(true_positive),
        "precision_50": float(precision),
        "recall_50": float(recall),
        "f1_50": float(f1),
        "nms_iou_threshold": None if nms_iou_threshold is None else float(nms_iou_threshold),
        "max_detections_per_image": None
        if max_detections_per_image is None
        else int(max_detections_per_image),
        "adaptive_max_detections": bool(adaptive_max_detections),
        "adaptive_count_source": str(adaptive_count_source),
        "adaptive_count_margin": int(adaptive_count_margin),
        "adaptive_min_detections": int(adaptive_min_detections),
        "require_foreground_argmax": bool(require_foreground_argmax),
        "score_mode": str(score_mode),
    }


def _cumulative_detection_true_positive_counts(
    classes: torch.Tensor,
    target_labels: torch.Tensor,
    iou_matrix: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    if classes.numel() == 0:
        return torch.zeros((1,), dtype=torch.long)
    cumulative = torch.zeros((classes.numel() + 1,), dtype=torch.long)
    if target_labels.numel() == 0:
        return cumulative

    matched_targets = torch.zeros((target_labels.numel(),), dtype=torch.bool)
    true_positive = 0
    for prediction_index in range(classes.numel()):
        same_class = target_labels == classes[prediction_index]
        available = same_class & (~matched_targets)
        if available.any():
            candidate_indices = torch.nonzero(available, as_tuple=False).flatten()
            candidate_iou = iou_matrix[prediction_index, candidate_indices]
            best_local = int(torch.argmax(candidate_iou).item())
            best_target = int(candidate_indices[best_local].item())
            if float(candidate_iou[best_local].item()) >= float(iou_threshold):
                matched_targets[best_target] = True
                true_positive += 1
        cumulative[prediction_index + 1] = int(true_positive)
    return cumulative


def _build_detection_confidence_curve(
    records: Sequence[Dict[str, torch.Tensor]],
    num_thresholds: int = 101,
    nms_iou_threshold: Optional[float] = 0.5,
    max_detections_per_image: Optional[int] = None,
    adaptive_max_detections: bool = False,
    adaptive_count_source: str = "auto",
    adaptive_count_margin: int = 1,
    adaptive_min_detections: int = 1,
    require_foreground_argmax: bool = False,
    score_mode: str = "foreground",
) -> Dict[str, object]:
    prepared_records = _prepare_detection_records_for_confidence_curve(
        records,
        nms_iou_threshold=nms_iou_threshold,
        max_detections_per_image=max_detections_per_image,
        adaptive_max_detections=adaptive_max_detections,
        adaptive_count_source=adaptive_count_source,
        adaptive_count_margin=adaptive_count_margin,
        adaptive_min_detections=adaptive_min_detections,
        require_foreground_argmax=require_foreground_argmax,
        score_mode=score_mode,
    )
    return _build_detection_confidence_curve_from_prepared(
        prepared_records,
        num_thresholds=num_thresholds,
        nms_iou_threshold=nms_iou_threshold,
        max_detections_per_image=max_detections_per_image,
        adaptive_max_detections=adaptive_max_detections,
        adaptive_count_source=adaptive_count_source,
        adaptive_count_margin=adaptive_count_margin,
        adaptive_min_detections=adaptive_min_detections,
        require_foreground_argmax=require_foreground_argmax,
        score_mode=score_mode,
    )


def _build_detection_confidence_curve_from_prepared(
    prepared_records: Sequence[Dict[str, torch.Tensor]],
    num_thresholds: int = 101,
    nms_iou_threshold: Optional[float] = 0.5,
    max_detections_per_image: Optional[int] = None,
    adaptive_max_detections: bool = False,
    adaptive_count_source: str = "auto",
    adaptive_count_margin: int = 1,
    adaptive_min_detections: int = 1,
    require_foreground_argmax: bool = False,
    score_mode: str = "foreground",
) -> Dict[str, object]:
    thresholds = torch.linspace(0.0, 1.0, steps=max(2, int(num_thresholds)), dtype=torch.float32)

    predicted_objects_by_threshold = torch.zeros_like(thresholds, dtype=torch.long)
    true_positive_by_threshold = torch.zeros_like(thresholds, dtype=torch.long)
    ground_truth_objects = 0
    for record in prepared_records:
        scores = record["scores"].to(dtype=torch.float32)
        classes = record["classes"].to(dtype=torch.long)
        target_labels = record["target_labels"].to(dtype=torch.long)
        iou_matrix = record["iou_matrix"].to(dtype=torch.float32)
        ground_truth_objects += int(target_labels.numel())
        if scores.numel() == 0:
            continue

        selected_counts = (scores.unsqueeze(0) >= thresholds.view(-1, 1)).sum(dim=1).to(dtype=torch.long)
        prefix_true_positive = _cumulative_detection_true_positive_counts(
            classes=classes,
            target_labels=target_labels,
            iou_matrix=iou_matrix,
        )
        predicted_objects_by_threshold += selected_counts
        true_positive_by_threshold += prefix_true_positive[selected_counts]

    precision = true_positive_by_threshold.to(dtype=torch.float32) / predicted_objects_by_threshold.clamp(min=1).to(dtype=torch.float32)
    recall = true_positive_by_threshold.to(dtype=torch.float32) / float(max(1, ground_truth_objects))
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    precision_values = precision.tolist()
    recall_values = recall.tolist()
    f1_values = f1.tolist()
    predicted_objects_values = [int(value) for value in predicted_objects_by_threshold.tolist()]
    true_positive_values = [int(value) for value in true_positive_by_threshold.tolist()]
    best_index = int(torch.argmax(f1).item()) if f1.numel() else 0

    curve = {
        "thresholds": thresholds.tolist(),
        "predicted_objects": predicted_objects_values,
        "ground_truth_objects": int(ground_truth_objects),
        "true_positive_50": true_positive_values,
        "precision_50": precision_values,
        "recall_50": recall_values,
        "f1_50": f1_values,
        "best_f1_50": max(0.0, float(f1[best_index].item())) if f1.numel() else 0.0,
        "best_f1_50_confidence": float(thresholds[best_index].item()),
        "nms_iou_threshold": None if nms_iou_threshold is None else float(nms_iou_threshold),
        "max_detections_per_image": None
        if max_detections_per_image is None
        else int(max_detections_per_image),
        "adaptive_max_detections": bool(adaptive_max_detections),
        "adaptive_count_source": str(adaptive_count_source),
        "adaptive_count_margin": int(adaptive_count_margin),
        "adaptive_min_detections": int(adaptive_min_detections),
        "require_foreground_argmax": bool(require_foreground_argmax),
        "score_mode": str(score_mode),
    }
    curve["best_metrics"] = _detection_curve_metrics_at_index(curve, best_index)
    return curve


def _detection_curve_metrics_at_index(curve: Dict[str, object], index: int) -> Dict[str, object]:
    thresholds = list(curve.get("thresholds", []))
    if not thresholds:
        index = 0
    else:
        index = min(max(0, int(index)), len(thresholds) - 1)
    predicted_objects = list(curve.get("predicted_objects", []))
    true_positive = list(curve.get("true_positive_50", []))
    precision = list(curve.get("precision_50", []))
    recall = list(curve.get("recall_50", []))
    f1 = list(curve.get("f1_50", []))
    return {
        "threshold": float(thresholds[index]) if thresholds else 0.0,
        "predicted_objects": int(predicted_objects[index]) if index < len(predicted_objects) else 0,
        "ground_truth_objects": int(curve.get("ground_truth_objects", 0) or 0),
        "true_positive_50": int(true_positive[index]) if index < len(true_positive) else 0,
        "precision_50": float(precision[index]) if index < len(precision) else 0.0,
        "recall_50": float(recall[index]) if index < len(recall) else 0.0,
        "f1_50": float(f1[index]) if index < len(f1) else 0.0,
        "nms_iou_threshold": curve.get("nms_iou_threshold"),
        "max_detections_per_image": curve.get("max_detections_per_image"),
        "adaptive_max_detections": bool(curve.get("adaptive_max_detections", False)),
        "adaptive_count_source": str(curve.get("adaptive_count_source", "auto")),
        "adaptive_count_margin": int(
            curve.get("adaptive_count_margin", 1)
            if curve.get("adaptive_count_margin", 1) is not None
            else 1
        ),
        "adaptive_min_detections": int(
            curve.get("adaptive_min_detections", 1)
            if curve.get("adaptive_min_detections", 1) is not None
            else 1
        ),
        "require_foreground_argmax": bool(curve.get("require_foreground_argmax", False)),
        "score_mode": str(curve.get("score_mode", "")),
    }


def _build_background_aware_classification_metrics(
    targets: torch.Tensor,
    predictions_with_background: torch.Tensor,
    class_names: Sequence[str],
    threshold: Optional[float] = None,
    score_mode: str = "",
) -> Dict[str, object]:
    num_classes = len(class_names)
    background_index = num_classes
    targets = targets.to(dtype=torch.long).view(-1)
    predictions_with_background = predictions_with_background.to(dtype=torch.long).view(-1)
    support = torch.zeros((num_classes,), dtype=torch.float32)
    predicted_support = torch.zeros((num_classes,), dtype=torch.float32)
    true_positive = torch.zeros((num_classes,), dtype=torch.float32)
    background_predictions = int((predictions_with_background == background_index).sum().item())

    for class_index in range(num_classes):
        target_mask = targets == class_index
        prediction_mask = predictions_with_background == class_index
        support[class_index] = target_mask.sum().to(dtype=torch.float32)
        predicted_support[class_index] = prediction_mask.sum().to(dtype=torch.float32)
        true_positive[class_index] = (target_mask & prediction_mask).sum().to(dtype=torch.float32)

    precision = true_positive / predicted_support.clamp(min=1.0)
    recall = true_positive / support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    accuracy = true_positive.sum() / targets.numel() if targets.numel() else torch.tensor(0.0)
    per_class = [
        {
            "class_index": int(index),
            "class_name": str(class_name),
            "support": int(support[index].item()),
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
        }
        for index, class_name in enumerate(class_names)
    ]
    return {
        "accuracy": float(accuracy.item()),
        "macro_precision": float(precision.mean().item()) if precision.numel() else 0.0,
        "macro_recall": float(recall.mean().item()) if recall.numel() else 0.0,
        "macro_f1": float(f1.mean().item()) if f1.numel() else 0.0,
        "weighted_f1": float((f1 * support / support.sum().clamp(min=1.0)).sum().item()) if f1.numel() else 0.0,
        "threshold": None if threshold is None else float(threshold),
        "score_mode": str(score_mode),
        "background_predictions": background_predictions,
        "background_prediction_rate": background_predictions / float(max(1, targets.numel())),
        "per_class": per_class,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    max_batches: Optional[int] = None,
    collect_artifact_stats: bool = True,
    tta: bool = False,
    tta_brightness_delta: float = 0.08,
    confidence_threshold: Optional[float] = None,
    detection_nms_iou_threshold: Optional[float] = 0.5,
    max_detections_per_image: Optional[int] = None,
    adaptive_max_detections: bool = False,
    adaptive_count_source: str = "auto",
    adaptive_count_margin: int = 1,
    adaptive_min_detections: int = 1,
    require_foreground_argmax: bool = False,
    detection_score_mode: str = "foreground",
    collect_prediction_records: bool = False,
) -> Dict[str, object]:
    eval_start = time.perf_counter()
    model.eval()
    losses = []
    loss_components = {
        "cls_loss": 0.0,
        "objectness_loss": 0.0,
        "bbox_l1_loss": 0.0,
        "bbox_giou_loss": 0.0,
        "cardinality_loss": 0.0,
        "count_loss": 0.0,
        "quality_loss": 0.0,
        "count_objectness_consistency_loss": 0.0,
        "auxiliary_loss": 0.0,
    }
    all_targets = []
    all_predictions = []
    all_full_predictions = []
    all_detection_scores = []
    all_probabilities = []
    all_bbox_targets = []
    all_bbox_predictions = []
    detection_records: List[Dict[str, torch.Tensor]] = []
    detection_totals = {
        "predicted_objects": 0,
        "ground_truth_objects": 0,
        "true_positive_50": 0,
    }
    count_abs_error_sum = 0.0
    count_squared_error_sum = 0.0
    count_samples = 0
    detection_mode = False
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
    total_batches = len(dataloader)
    batch_iterator = dataloader
    tta_runner = None
    if tta:
        tta_runner = TestTimeAugmentation(
            brightness_delta=tta_brightness_delta,
            contrast_delta=tta_brightness_delta,
            saturation_delta=tta_brightness_delta,
            num_aug=4,
        )
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        batch_iterator = islice(dataloader, total_batches)

    loop_start = time.perf_counter()
    with torch.inference_mode():
        with tqdm(
            batch_iterator,
            desc="Validation",
            leave=False,
            total=total_batches,
            dynamic_ncols=True,
        ) as pbar:
            for batch in pbar:
                if len(batch) == 2 and _is_detection_targets(batch[1]):
                    images, batch_targets = batch
                    labels = None
                    target_boxes = None
                    detection_targets = _move_targets_to_device(batch_targets, device)
                    detection_mode = True
                elif len(batch) == 3:
                    images, labels, targets = batch
                    target_boxes = targets["bbox"]
                    detection_targets = None
                elif len(batch) == 2:
                    images, labels = batch
                    target_boxes = None
                    detection_targets = None
                else:
                    raise ValueError("Eval dataloader phai tra ve 2 hoac 3 phan tu.")
                images = images.to(device, non_blocking=True)
                if labels is not None:
                    labels = labels.to(device, non_blocking=True)
                    metric_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
                else:
                    metric_labels = None
                if target_boxes is not None:
                    target_boxes = target_boxes.to(device, non_blocking=True)

                base_features = None
                base_output = None
                image_valid_mask = _stack_image_masks_from_targets(detection_targets)
                with autocast_context(device, amp):
                    if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
                        base_features = model.forward_features(images, image_valid_mask=image_valid_mask)
                        base_output = model.forward_heads(base_features)
                    elif hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
                        base_features = model.forward_features(images, image_valid_mask=image_valid_mask)
                        base_output = model.head(extract_head_input_from_features(model, base_features))
                    else:
                        base_output = model(images)
                    base_logits, pred_boxes, base_objectness_logits = extract_detection_from_model_output(base_output)
                    base_count_logits = _extract_count_logits(base_output)
                    base_quality_logits = _extract_quality_logits(base_output)
                    metric_logits = base_logits
                    metric_pred_boxes = pred_boxes
                    metric_objectness_logits = base_objectness_logits
                    metric_count_logits = base_count_logits
                    metric_quality_logits = base_quality_logits

                    if tta_runner is not None:
                        tta_output = tta_runner.forward(model, images)
                        metric_logits = tta_output["logits"]
                        metric_pred_boxes = tta_output.get("boxes", pred_boxes)
                        metric_objectness_logits = tta_output.get("objectness_logits", base_objectness_logits)
                        metric_count_logits = tta_output.get("count_logits", base_count_logits)
                        metric_quality_logits = tta_output.get("quality_logits", base_quality_logits)
                        if metric_pred_boxes is None:
                            metric_pred_boxes = pred_boxes

                    if criterion is not None:
                        if detection_targets is not None:
                            loss_result = criterion(
                                base_output,
                                detection_targets,
                                return_details=True,
                            )
                            loss, details = loss_result
                            for key in loss_components:
                                loss_components[key] += float(details.get(key, 0.0))
                        elif target_boxes is not None and hasattr(criterion, "bbox_loss"):
                            loss_result = criterion(
                                base_output,
                                labels,
                                {"bbox": target_boxes},
                                return_details=True,
                            )
                            loss, details = loss_result
                            for key in loss_components:
                                loss_components[key] += float(details.get(key, 0.0))
                        else:
                            loss = criterion(base_logits, labels)
                        loss_value = float(loss.detach().cpu().item())
                        losses.append(loss_value)
                        pbar.set_postfix(loss=f"{loss_value:.4f}")

                probabilities = F.softmax(metric_logits.float(), dim=-1)
                objectness_scores = (
                    torch.sigmoid(metric_objectness_logits.float())
                    if metric_objectness_logits is not None
                    else None
                )
                quality_scores = (
                    torch.sigmoid(metric_quality_logits.float())
                    if metric_quality_logits is not None
                    else None
                )
                count_predictions = (
                    F.softplus(metric_count_logits.float()).detach()
                    if metric_count_logits is not None
                    else None
                )

                if detection_targets is not None:
                    matcher = getattr(criterion, "matcher", HungarianMatcher())
                    matcher_output = {"logits": metric_logits.float(), "boxes": metric_pred_boxes.float()}
                    if metric_objectness_logits is not None:
                        matcher_output["objectness_logits"] = metric_objectness_logits.float()
                    indices = matcher(matcher_output, detection_targets)
                    uses_separate_objectness = objectness_scores is not None
                    background_index = len(class_names)
                    foreground_scores, foreground_classes = probabilities.max(dim=-1)
                    if uses_separate_objectness:
                        detection_scores = foreground_scores * objectness_scores
                        predicted_labels_with_background = foreground_classes.clone()
                        predicted_labels_with_background = torch.where(
                            objectness_scores >= 0.5,
                            predicted_labels_with_background,
                            torch.full_like(predicted_labels_with_background, background_index),
                        )
                        background_scores = 1.0 - objectness_scores
                    else:
                        background_index = probabilities.shape[-1] - 1
                        predicted_labels_with_background = probabilities.argmax(dim=-1)
                        foreground_scores, foreground_classes = probabilities[..., :-1].max(dim=-1)
                        detection_scores = foreground_scores
                        background_scores = probabilities[..., background_index]
                    detection_totals["predicted_objects"] += int(
                        (predicted_labels_with_background != background_index).sum().item()
                    )
                    detection_totals["ground_truth_objects"] += int(
                        sum(int(target["labels"].numel()) for target in detection_targets)
                    )
                    for batch_index, target in enumerate(detection_targets):
                        detection_records.append(
                            {
                                "scores": detection_scores[batch_index].detach().cpu(),
                                "class_scores": foreground_scores[batch_index].detach().cpu(),
                                "classes": foreground_classes[batch_index].detach().cpu(),
                                "boxes": metric_pred_boxes[batch_index].detach().cpu().to(torch.float32),
                                "objectness_scores": objectness_scores[batch_index].detach().cpu().to(torch.float32)
                                if objectness_scores is not None
                                else None,
                                "quality_scores": quality_scores[batch_index].detach().cpu().to(torch.float32)
                                if quality_scores is not None
                                else None,
                                "background_scores": background_scores[batch_index].detach().cpu().to(torch.float32),
                                "count_prediction": count_predictions[batch_index].detach().cpu().to(torch.float32)
                                if count_predictions is not None
                                else None,
                                "is_background_argmax": (
                                    predicted_labels_with_background[batch_index] == background_index
                                )
                                .detach()
                                .cpu(),
                                "target_labels": target["labels"].detach().cpu().to(torch.long),
                                "target_boxes": target["boxes"].detach().cpu().to(torch.float32),
                            }
                        )
                        if count_predictions is not None:
                            target_count = float(target["labels"].numel())
                            prediction_count = float(count_predictions[batch_index].detach().cpu().item())
                            count_abs_error_sum += abs(prediction_count - target_count)
                            count_squared_error_sum += (prediction_count - target_count) ** 2
                            count_samples += 1

                    for batch_index, (query_indices, target_indices) in enumerate(indices):
                        if query_indices.numel() == 0:
                            continue
                        matched_class_probabilities = (
                            probabilities[batch_index, query_indices]
                            if uses_separate_objectness
                            else probabilities[batch_index, query_indices, :-1]
                        ).detach().cpu()
                        matched_predictions = matched_class_probabilities.argmax(dim=-1)
                        if uses_separate_objectness:
                            matched_objectness = objectness_scores[batch_index, query_indices].detach().cpu()
                            matched_predictions_with_background = torch.where(
                                matched_objectness >= 0.5,
                                matched_predictions,
                                torch.full_like(matched_predictions, background_index),
                            )
                        else:
                            matched_full_probabilities = probabilities[batch_index, query_indices].detach().cpu()
                            matched_predictions_with_background = matched_full_probabilities.argmax(dim=-1)
                        matched_targets = detection_targets[batch_index]["labels"][target_indices].detach().cpu()
                        matched_pred_boxes = metric_pred_boxes[batch_index, query_indices].detach().cpu()
                        matched_target_boxes = detection_targets[batch_index]["boxes"][target_indices].detach().cpu()
                        matched_detection_scores = detection_scores[batch_index, query_indices].detach().cpu()

                        all_targets.append(matched_targets)
                        all_predictions.append(matched_predictions)
                        all_full_predictions.append(matched_predictions_with_background)
                        all_detection_scores.append(matched_detection_scores)
                        all_probabilities.append(matched_class_probabilities)
                        all_bbox_predictions.append(matched_pred_boxes)
                        all_bbox_targets.append(matched_target_boxes)

                        iou = box_iou_from_xywh(matched_pred_boxes, matched_target_boxes)
                        valid_predictions = matched_predictions_with_background != background_index
                        class_correct = matched_predictions_with_background == matched_targets
                        detection_totals["true_positive_50"] += int(
                            ((iou >= 0.5) & class_correct & valid_predictions).sum().item()
                        )
                else:
                    predictions = probabilities.argmax(dim=1)
                    if metric_labels is None:
                        raise ValueError("Classification evaluation thieu label target.")
                    all_targets.append(metric_labels.detach().cpu())
                    all_predictions.append(predictions.detach().cpu())
                    all_probabilities.append(probabilities.detach().cpu())
                    if metric_pred_boxes is not None and target_boxes is not None:
                        all_bbox_predictions.append(metric_pred_boxes.detach().cpu())
                        all_bbox_targets.append(target_boxes.detach().cpu())

                if collect_artifact_stats and base_features is not None:
                    batch_stats = summarize_token_norms(base_features)
                    for key, value in batch_stats.items():
                        artifact_totals[key] += float(value)
                    artifact_batches += 1

    eval_loop_seconds = time.perf_counter() - loop_start
    metrics_start = time.perf_counter()
    detection_prepare_seconds = 0.0
    detection_threshold_seconds = 0.0
    detection_curve_seconds = 0.0
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    predictions = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)
    full_predictions = (
        torch.cat(all_full_predictions)
        if all_full_predictions
        else torch.empty(0, dtype=torch.long)
    )
    matched_detection_scores = (
        torch.cat(all_detection_scores).to(dtype=torch.float32)
        if all_detection_scores
        else torch.empty(0, dtype=torch.float32)
    )
    probabilities = (
        torch.cat(all_probabilities)
        if all_probabilities
        else torch.empty((0, len(class_names)), dtype=torch.float32)
    )
    metrics = build_metrics(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
        probabilities=probabilities,
    )
    if bool(collect_prediction_records) and not detection_mode and targets.numel() and predictions.numel():
        metrics["prediction_records"] = _build_prediction_records(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
            sample_paths=_dataset_sample_paths(getattr(dataloader, "dataset", None)),
        )
    if detection_mode and targets.numel() and full_predictions.numel():
        metrics["foreground_classification"] = {
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class": metrics.get("per_class", []),
        }
        metrics["background_aware_classification_raw_objectness_05"] = (
            _build_background_aware_classification_metrics(
                targets=targets,
                predictions_with_background=full_predictions,
                class_names=class_names,
                threshold=0.5,
                score_mode="objectness_argmax",
            )
        )
    metrics["loss"] = float(sum(losses) / max(1, len(losses))) if losses else 0.0
    if losses:
        metrics["loss_components"] = {
            key: value / max(1, len(losses)) for key, value in loss_components.items()
        }
    metrics["tta_enabled"] = bool(tta)
    metrics["tta_brightness_delta"] = float(tta_brightness_delta)
    if all_bbox_predictions and all_bbox_targets:
        pred_boxes = torch.cat(all_bbox_predictions, dim=0).to(torch.float32)
        target_boxes = torch.cat(all_bbox_targets, dim=0).to(torch.float32)
        bbox_giou = generalized_iou(pred_boxes, target_boxes)
        bbox_iou = box_iou_from_xywh(pred_boxes, target_boxes)
        metrics["bbox"] = {
            "mean_giou": float(bbox_giou.mean().item()),
            "mean_iou": float(bbox_iou.mean().item()),
            "mean_l1": float(torch.nn.functional.l1_loss(pred_boxes, target_boxes).item()),
        }
    if detection_mode:
        predicted_objects = max(0, int(detection_totals["predicted_objects"]))
        ground_truth_objects = max(0, int(detection_totals["ground_truth_objects"]))
        true_positive = max(0, int(detection_totals["true_positive_50"]))
        precision = true_positive / float(max(1, predicted_objects))
        recall = true_positive / float(max(1, ground_truth_objects))
        f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
        metrics["detection"] = {
            "predicted_objects": predicted_objects,
            "ground_truth_objects": ground_truth_objects,
            "true_positive_50": true_positive,
            "precision_50": precision,
            "recall_50": recall,
            "f1_50": f1,
        }
        if detection_records:
            metrics["raw_query_detection"] = metrics["detection"]
            prepare_start = time.perf_counter()
            prepared_detection_records = _prepare_detection_records_for_confidence_curve(
                detection_records,
                nms_iou_threshold=detection_nms_iou_threshold,
                max_detections_per_image=max_detections_per_image,
                adaptive_max_detections=adaptive_max_detections,
                adaptive_count_source=adaptive_count_source,
                adaptive_count_margin=adaptive_count_margin,
                adaptive_min_detections=adaptive_min_detections,
                require_foreground_argmax=require_foreground_argmax,
                score_mode=detection_score_mode,
            )
            detection_prepare_seconds = time.perf_counter() - prepare_start

            curve_start = time.perf_counter()
            detection_curve = _build_detection_confidence_curve_from_prepared(
                prepared_detection_records,
                nms_iou_threshold=detection_nms_iou_threshold,
                max_detections_per_image=max_detections_per_image,
                adaptive_max_detections=adaptive_max_detections,
                adaptive_count_source=adaptive_count_source,
                adaptive_count_margin=adaptive_count_margin,
                adaptive_min_detections=adaptive_min_detections,
                require_foreground_argmax=require_foreground_argmax,
                score_mode=detection_score_mode,
            )
            detection_curve_seconds = time.perf_counter() - curve_start
            if confidence_threshold is None:
                metrics["detection"] = _detection_curve_metrics_at_index(detection_curve, 0)
            else:
                threshold_start = time.perf_counter()
                metrics["detection"] = _compute_prepared_detection_metrics_at_threshold(
                    prepared_detection_records,
                    threshold=float(confidence_threshold),
                    nms_iou_threshold=detection_nms_iou_threshold,
                    max_detections_per_image=max_detections_per_image,
                    adaptive_max_detections=adaptive_max_detections,
                    adaptive_count_source=adaptive_count_source,
                    adaptive_count_margin=adaptive_count_margin,
                    adaptive_min_detections=adaptive_min_detections,
                    require_foreground_argmax=require_foreground_argmax,
                    score_mode=detection_score_mode,
                )
                detection_threshold_seconds = time.perf_counter() - threshold_start
            metrics["detection_confidence_curve"] = detection_curve
            metrics["detection_best_threshold"] = detection_curve.get("best_metrics", {})
            if confidence_threshold is not None:
                metrics["detection_calibrated"] = dict(metrics["detection"])
        if targets.numel() and predictions.numel() and matched_detection_scores.numel():
            background_index = len(class_names)
            background_threshold = confidence_threshold
            if background_threshold is None:
                background_threshold = (
                    metrics.get("detection_confidence_curve", {}).get("best_f1_50_confidence")
                    if isinstance(metrics.get("detection_confidence_curve", {}), dict)
                    else None
                )
            if background_threshold is None:
                background_threshold = 0.5
            calibrated_full_predictions = torch.where(
                matched_detection_scores >= float(background_threshold),
                predictions.to(dtype=torch.long),
                torch.full_like(predictions.to(dtype=torch.long), background_index),
            )
            metrics["background_aware_classification"] = _build_background_aware_classification_metrics(
                targets=targets,
                predictions_with_background=calibrated_full_predictions,
                class_names=class_names,
                threshold=float(background_threshold),
                score_mode=detection_score_mode,
            )
        if count_samples > 0:
            metrics["count"] = {
                "mae": float(count_abs_error_sum / max(1, count_samples)),
                "rmse": float(math.sqrt(count_squared_error_sum / max(1, count_samples))),
                "samples": int(count_samples),
                "source": "count_head",
            }
    if confidence_threshold is not None:
        metrics["calibrated"] = build_thresholded_acceptance_summary(
            targets=targets,
            probabilities=probabilities,
            class_names=class_names,
            threshold=confidence_threshold,
        )
    if artifact_batches > 0:
        metrics["artifact_stats"] = {
            key: value / artifact_batches for key, value in artifact_totals.items()
        }
    metric_seconds = time.perf_counter() - metrics_start
    metrics["timing"] = {
        "eval_loop_seconds": float(eval_loop_seconds),
        "metric_seconds": float(metric_seconds),
        "detection_prepare_seconds": float(detection_prepare_seconds),
        "detection_threshold_seconds": float(detection_threshold_seconds),
        "detection_curve_seconds": float(detection_curve_seconds),
        "total_seconds": float(time.perf_counter() - eval_start),
    }
    return metrics


def save_evaluation_artifacts(
    metrics: Dict[str, object],
    class_names: Sequence[str],
    output_dir: Path,
    *,
    comparison_summary: Optional[Dict[str, object]] = None,
    comparison_prediction_rows: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    ensure_dir(output_dir)
    prediction_records = metrics.get("prediction_records", [])
    metrics_payload = dict(metrics)
    metrics_payload.pop("prediction_records", None)
    json_dump(
        output_dir / ("metrics_detailed.json" if comparison_summary is not None else "metrics.json"),
        to_serializable(metrics_payload),
    )
    if comparison_summary is not None:
        json_dump(output_dir / "metrics.json", to_serializable(comparison_summary))
    if comparison_prediction_rows is not None:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "y_true", "y_pred", "true_name", "pred_name"],
            )
            writer.writeheader()
            for row in comparison_prediction_rows:
                writer.writerow(
                    {
                        "path": row.get("path", ""),
                        "y_true": row.get("y_true", ""),
                        "y_pred": row.get("y_pred", ""),
                        "true_name": row.get("true_name", ""),
                        "pred_name": row.get("pred_name", ""),
                    }
                )
    if isinstance(prediction_records, list) and prediction_records:
        fieldnames: List[str] = []
        for record in prediction_records:
            if not isinstance(record, dict):
                continue
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(str(key))
        predictions_path = output_dir / (
            "predictions_detailed.csv" if comparison_prediction_rows is not None else "predictions.csv"
        )
        with predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in prediction_records:
                if isinstance(record, dict):
                    writer.writerow({key: record.get(key, "") for key in fieldnames})
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_per_class_metrics(
        metrics.get("per_class", []),
        output_dir / "per_class_metrics.png",
    )
    if "confidence_curves" in metrics:
        plot_pr_curve(
            metrics["confidence_curves"],
            class_names,
            output_dir / "pr_curve.png",
        )
        plot_per_class_pr_panels(
            metrics["confidence_curves"],
            class_names,
            output_dir / "pr_curve_per_class.png",
        )
    if "detection_confidence_curve" in metrics:
        plot_detection_confidence_curve(
            metrics["detection_confidence_curve"],
            output_dir / "detection_confidence_curve.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Danh gia checkpoint DETR ViT-Registers cho mango multi-object detection.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Duong dan den best.pt")
    parser.add_argument("--data", type=Path, default=default_data_yaml(), help="data.yaml")
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
        help="Neu > 0, validate so class trong data.yaml truoc khi evaluate.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--paper-name",
        default="TRKH-ViTReg-224",
        help="Ten hien thi trong metrics.json kieu baseline.",
    )
    parser.add_argument(
        "--family",
        default="TRKH",
        help="Nhom model hien thi trong bang so sanh baseline.",
    )
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--eval-tta", dest="tta", action="store_true", default=False)
    parser.add_argument("--tta-brightness-delta", type=float, default=0.08)
    parser.add_argument("--full-image-detection", action="store_true", default=False)
    parser.add_argument("--crop-to-primary-object", action="store_true", default=False)
    parser.add_argument("--disable-classification-object-crops", action="store_true", default=False)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--disable-calibration", action="store_true", default=False)
    parser.add_argument("--detection-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections-per-image", type=int, default=0)
    parser.add_argument(
        "--detection-score-mode",
        choices=DETECTION_SCORE_MODES,
        default="foreground",
    )
    parser.add_argument("--require-foreground-argmax", action="store_true", default=False)
    parser.add_argument(
        "--adaptive-max-detections",
        action="store_true",
        default=False,
        help="Dung count head/objectness de gioi han so detection moi anh.",
    )
    parser.add_argument(
        "--adaptive-count-source",
        choices=("auto", "count_head", "objectness"),
        default="auto",
    )
    parser.add_argument("--adaptive-count-margin", type=int, default=1)
    parser.add_argument("--adaptive-min-detections", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.adaptive_count_margin < 0:
        raise ValueError("--adaptive-count-margin phai >= 0.")
    if args.adaptive_min_detections < 0:
        raise ValueError("--adaptive-min-detections phai >= 0.")
    set_seed(args.seed, deterministic=False)
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if len(class_names) != data_spec.num_classes:
        raise ValueError("So lop trong checkpoint khong khop data.yaml")
    checkpoint_model_type = str(checkpoint.get("model_config", {}).get("model_type", "")).strip().lower()
    checkpoint_detection_mode = checkpoint_model_type in DETECTION_MODEL_TYPES
    crop_to_primary_object = resolve_crop_to_primary_object(
        checkpoint,
        full_image_detection=args.full_image_detection,
        crop_to_primary_object=args.crop_to_primary_object,
    )
    classification_object_crops = resolve_classification_object_crops(
        checkpoint,
        disable_classification_object_crops=args.disable_classification_object_crops,
    )

    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        num_classes=len(class_names),
        override_image_size=args.override_image_size,
    )
    model.to(device)
    model.eval()

    image_size = int(args.override_image_size or checkpoint.get("model_config", {}).get("image_size", 224))
    eval_transform = build_eval_transform(
        image_size=image_size,
        resize_mode=checkpoint.get("augmentation_config", {}).get("resize_mode", "pad"),
    )
    if data_spec.data_format == "classification_folder":
        if checkpoint_detection_mode:
            raise ValueError("format=classification_folder khong ho tro evaluate checkpoint detection.")
        dataset = ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split=args.split,
            transform=eval_transform,
            class_aware_augmentation=False,
        )
    else:
        dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split=args.split,
            transform=eval_transform,
            crop_margin_ratio=float(
                checkpoint.get("augmentation_config", {}).get("crop_margin_ratio", 0.05)
            ),
            crop_to_primary_object=crop_to_primary_object,
            classification_target=not checkpoint_detection_mode,
            classification_object_crops=classification_object_crops,
        )
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=args.num_workers,
        requested_pin_memory=device.type == "cuda",
        context=f"evaluate_{args.split}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    cache_summary = maybe_enable_dataset_image_cache(
        dataset,
        enabled=int(dataloader_summary["effective_num_workers"]) == 0,
        context=f"evaluate_{args.split}",
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
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(dataset.num_classes or 1)),
            batch_mix_probability=0.0,
        ),
        **dataloader_kwargs,
    )
    confidence_threshold = resolve_confidence_threshold(
        checkpoint=checkpoint,
        explicit_threshold=args.confidence_threshold,
        disable_calibration=args.disable_calibration,
    )
    if checkpoint_detection_mode:
        criterion = HybridDetectionClassificationLoss(
            num_classes=data_spec.num_classes,
            label_smoothing=float(checkpoint.get("train_config", {}).get("label_smoothing", 0.0)),
            cls_weight=float(checkpoint.get("train_config", {}).get("cls_loss_weight", 1.0)),
            bbox_l1_weight=float(checkpoint.get("train_config", {}).get("bbox_l1_loss_weight", 1.0)),
            bbox_giou_weight=float(checkpoint.get("train_config", {}).get("bbox_giou_loss_weight", 0.5)),
            background_weight=float(checkpoint.get("train_config", {}).get("background_loss_weight", 0.3)),
            objectness_weight=float(checkpoint.get("train_config", {}).get("objectness_loss_weight", 5.0)),
            objectness_focal_alpha=float(checkpoint.get("train_config", {}).get("objectness_focal_alpha", 0.75)),
            objectness_focal_gamma=float(checkpoint.get("train_config", {}).get("objectness_focal_gamma", 0.5)),
            matcher_class_cost=float(checkpoint.get("train_config", {}).get("matcher_class_cost", 1.0)),
            matcher_objectness_cost=float(checkpoint.get("train_config", {}).get("matcher_objectness_cost", 1.0)),
            cardinality_weight=float(checkpoint.get("train_config", {}).get("cardinality_loss_weight", 0.0)),
            count_weight=float(checkpoint.get("train_config", {}).get("count_loss_weight", 0.0)),
        )
    else:
        criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        device=device,
        class_names=class_names,
        criterion=criterion,
        amp=args.amp,
        max_batches=args.max_batches or None,
        collect_artifact_stats=True,
        tta=args.tta,
        tta_brightness_delta=args.tta_brightness_delta,
        confidence_threshold=confidence_threshold,
        detection_nms_iou_threshold=args.detection_nms_iou_threshold,
        max_detections_per_image=(
            int(args.max_detections_per_image)
            if int(args.max_detections_per_image) > 0
            else None
        ),
        require_foreground_argmax=args.require_foreground_argmax,
        detection_score_mode=args.detection_score_mode,
        adaptive_max_detections=args.adaptive_max_detections,
        adaptive_count_source=args.adaptive_count_source,
        adaptive_count_margin=args.adaptive_count_margin,
        adaptive_min_detections=args.adaptive_min_detections,
        collect_prediction_records=True,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.resolve().parent.parent / f"eval_{args.split}"
    comparison_summary = None
    comparison_prediction_rows = None
    prediction_records = metrics.get("prediction_records", [])
    if not checkpoint_detection_mode and isinstance(prediction_records, list) and prediction_records:
        comparison_class_names = _comparison_class_names_for_data_spec(
            data_spec,
            args.split,
            class_names,
        )
        comparison_prediction_rows = _build_baseline_comparison_prediction_rows(
            prediction_records,
            class_names=class_names,
            comparison_class_names=comparison_class_names,
        )
        comparison_summary = _build_baseline_comparison_summary(
            rows=comparison_prediction_rows,
            comparison_class_names=comparison_class_names,
            metrics=metrics,
            model=model,
            checkpoint=checkpoint,
            paper_name=args.paper_name,
            family=args.family,
            seed=args.seed,
        )
    save_evaluation_artifacts(
        metrics,
        class_names,
        output_dir,
        comparison_summary=comparison_summary,
        comparison_prediction_rows=comparison_prediction_rows,
    )

    summary = {
        "split": args.split,
        "samples": len(dataset),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "loss": metrics["loss"],
        "tta": args.tta,
        "confidence_threshold": confidence_threshold,
    }
    if "bbox" in metrics:
        summary["bbox"] = metrics["bbox"]
    if "confidence_curves" in metrics:
        summary["best_macro_f1_confidence"] = metrics["confidence_curves"][
            "best_macro_f1_confidence"
        ]
    if "detection_confidence_curve" in metrics:
        summary["best_detection_f1_50"] = metrics["detection_confidence_curve"].get("best_f1_50", 0.0)
        summary["best_detection_f1_confidence"] = metrics["detection_confidence_curve"].get(
            "best_f1_50_confidence",
            0.0,
        )
    if "count" in metrics:
        summary["count"] = metrics["count"]
    if "calibrated" in metrics:
        summary["calibrated_coverage"] = metrics["calibrated"]["coverage"]
        summary["calibrated_macro_f1"] = metrics["calibrated"]["accepted_metrics"].get("macro_f1", 0.0)
    if "detection_calibrated" in metrics:
        summary["calibrated_detection_f1_50"] = metrics["detection_calibrated"].get("f1_50", 0.0)
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    set_seed,
)
from trkh.data.dataset import MangoYOLOCropDataset, bbox_xywh_to_xyxy
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _forward_logits,
    _select_bbox_token_prior,
)


TILE_FRACTION = 0.70
CANDIDATE_TILE_WEIGHT = 0.50
TILE_POSITIONS: Tuple[Tuple[str, float, float], ...] = (
    ("top_left", 0.0, 0.0),
    ("top_right", 1.0, 0.0),
    ("bottom_left", 0.0, 1.0),
    ("bottom_right", 1.0, 1.0),
    ("center", 0.5, 0.5),
)
EXPECTED_KEEPER_VAL_MACRO = 0.882925
EXPECTED_KEEPER_VAL_CLASS1 = 0.678261


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked no-test audit of five original-resolution object-surface tiles "
            "against the keeper's exact global crop."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--allow-preflight",
        action="store_true",
        default=False,
        help="Allow capped train/validation prefixes for implementation checks only.",
    )
    return parser.parse_args()


def fixed_tile_boxes(
    width: int,
    height: int,
    fraction: float = TILE_FRACTION,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    if width < 2 or height < 2:
        raise ValueError("Object crop must be at least 2x2 pixels")
    if not 0.5 <= float(fraction) < 1.0:
        raise ValueError("tile fraction must be in [0.5, 1.0)")
    tile_width = min(width, max(2, int(round(width * float(fraction)))))
    tile_height = min(height, max(2, int(round(height * float(fraction)))))
    travel_x = max(0, width - tile_width)
    travel_y = max(0, height - tile_height)
    boxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for name, x_position, y_position in TILE_POSITIONS:
        left = int(round(travel_x * float(x_position)))
        top = int(round(travel_y * float(y_position)))
        boxes.append((name, (left, top, left + tile_width, top + tile_height)))
    return boxes


def object_crop_geometry(
    *,
    image_size: Sequence[int],
    bbox: Sequence[float],
    margin_ratio: float,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    width, height = int(image_size[0]), int(image_size[1])
    if width < 1 or height < 1:
        raise ValueError("image size must be positive")
    x1, y1, x2, y2 = bbox_xywh_to_xyxy(tuple(float(v) for v in bbox), width, height)
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    margin_x = box_width * max(0.0, float(margin_ratio))
    margin_y = box_height * max(0.0, float(margin_ratio))
    left = max(0, int(math.floor(float(x1) - margin_x)))
    top = max(0, int(math.floor(float(y1) - margin_y)))
    right = min(width, int(math.ceil(float(x2) + margin_x)))
    bottom = min(height, int(math.ceil(float(y2) + margin_y)))
    if right <= left or bottom <= top:
        raise ValueError("invalid object crop geometry")
    object_xyxy = (
        max(0.0, float(x1) - left),
        max(0.0, float(y1) - top),
        min(float(right - left), float(x2) - left),
        min(float(bottom - top), float(y2) - top),
    )
    return (left, top, right, bottom), object_xyxy


def clipped_bbox_for_tile(
    object_xyxy: Sequence[float],
    tile_xyxy: Sequence[int],
) -> Tuple[float, float, float, float]:
    tile_left, tile_top, tile_right, tile_bottom = [float(v) for v in tile_xyxy]
    object_left, object_top, object_right, object_bottom = [float(v) for v in object_xyxy]
    left = max(tile_left, object_left)
    top = max(tile_top, object_top)
    right = min(tile_right, object_right)
    bottom = min(tile_bottom, object_bottom)
    if right <= left or bottom <= top:
        raise ValueError("fixed surface tile does not intersect the object")
    tile_width = max(1.0, tile_right - tile_left)
    tile_height = max(1.0, tile_bottom - tile_top)
    return (
        ((left + right) * 0.5 - tile_left) / tile_width,
        ((top + bottom) * 0.5 - tile_top) / tile_height,
        (right - left) / tile_width,
        (bottom - top) / tile_height,
    )


def fuse_global_and_tiles(
    global_probabilities: Tensor,
    tile_probabilities: Tensor,
    tile_weight: float = CANDIDATE_TILE_WEIGHT,
) -> Tensor:
    if global_probabilities.ndim != 2:
        raise ValueError("global probabilities must have shape [B, C]")
    if tile_probabilities.ndim != 3:
        raise ValueError("tile probabilities must have shape [B, V, C]")
    if tile_probabilities.shape[0] != global_probabilities.shape[0]:
        raise ValueError("global/tile batch mismatch")
    if tile_probabilities.shape[2] != global_probabilities.shape[1]:
        raise ValueError("global/tile class mismatch")
    if not 0.0 <= float(tile_weight) <= 1.0:
        raise ValueError("tile weight must be in [0, 1]")
    tile_mean = tile_probabilities.mean(dim=1)
    fused = (1.0 - float(tile_weight)) * global_probabilities + float(tile_weight) * tile_mean
    return fused / fused.sum(dim=1, keepdim=True).clamp(min=1e-8)


def _normalize_mask(mask: Optional[Tensor], image: Tensor) -> Tensor:
    if not torch.is_tensor(mask):
        return torch.ones(image.shape[-2:], dtype=torch.bool)
    value = mask.to(dtype=torch.bool)
    if value.ndim == 3 and value.size(0) == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"image mask must be 2D, got {tuple(value.shape)}")
    return value


class OriginalResolutionSurfaceTileDataset(Dataset):
    def __init__(
        self,
        base_dataset: MangoYOLOCropDataset,
        *,
        max_samples: int = 0,
    ) -> None:
        if not isinstance(base_dataset, MangoYOLOCropDataset):
            raise TypeError("surface-tile audit requires a YOLO crop dataset")
        self.base_dataset = base_dataset
        self.max_samples = max(0, int(max_samples))

    def __len__(self) -> int:
        length = len(self.base_dataset)
        return min(length, self.max_samples) if self.max_samples > 0 else length

    def _source_views(
        self,
        index: int,
    ) -> Tuple[Image.Image, List[Tuple[str, Image.Image]], Dict[str, object]]:
        sample = self.base_dataset.samples[int(index)]
        source_image = self.base_dataset._load_rgb_image(sample.image_path)
        primary = self.base_dataset._select_sample_primary_object(sample)
        crop_box, object_xyxy = object_crop_geometry(
            image_size=source_image.size,
            bbox=primary.bbox,
            margin_ratio=float(self.base_dataset._crop_margin_ratio_for_sample(sample)),
        )
        object_crop = source_image.crop(crop_box).convert("RGB")
        named_tiles: List[Tuple[str, Image.Image]] = []
        tile_boxes = fixed_tile_boxes(*object_crop.size)
        tile_bboxes: List[Tuple[float, float, float, float]] = []
        for name, tile_box in tile_boxes:
            named_tiles.append((name, object_crop.crop(tile_box).convert("RGB")))
            tile_bboxes.append(clipped_bbox_for_tile(object_xyxy, tile_box))
        geometry: Dict[str, object] = {
            "sample_index": int(index),
            "image_path": str(sample.image_path),
            "source_stem": sample.image_path.stem,
            "label": int(primary.label),
            "source_bbox": tuple(float(v) for v in primary.bbox),
            "crop_box": crop_box,
            "crop_width": int(object_crop.width),
            "crop_height": int(object_crop.height),
            "crop_min_side": int(min(object_crop.size)),
            "tile_boxes": [box for _, box in tile_boxes],
            "tile_bboxes": tile_bboxes,
        }
        return object_crop, named_tiles, geometry

    def render_views(self, index: int) -> Tuple[List[Tuple[str, Image.Image]], Dict[str, object]]:
        object_crop, tiles, geometry = self._source_views(index)
        return [("global", object_crop)] + tiles, geometry

    def __getitem__(self, index: int) -> Dict[str, object]:
        global_image, label, global_metadata = self.base_dataset[int(index)]
        if not isinstance(global_metadata, dict):
            raise TypeError("classification bbox metadata is required")
        _, named_tiles, geometry = self._source_views(index)
        source_bbox = global_metadata.get("bbox")
        global_crop_bbox = global_metadata.get("crop_bbox")
        if not torch.is_tensor(source_bbox) or not torch.is_tensor(global_crop_bbox):
            raise ValueError("global bbox/crop_bbox metadata is missing")

        images = [global_image]
        source_bboxes = [source_bbox.to(dtype=torch.float32)]
        crop_bboxes = [global_crop_bbox.to(dtype=torch.float32)]
        masks = [_normalize_mask(global_metadata.get("image_mask"), global_image)]
        tile_bboxes = geometry["tile_bboxes"]
        for (_, tile), tile_bbox in zip(named_tiles, tile_bboxes):
            target = {
                "labels": torch.tensor([int(label)], dtype=torch.long),
                "boxes": torch.tensor([tile_bbox], dtype=torch.float32),
                "augmentation_scale": torch.ones(1, dtype=torch.float32),
            }
            transformed = self.base_dataset.transform(tile, target=target)
            if not isinstance(transformed, tuple) or len(transformed) != 2:
                raise TypeError("evaluation transform must return (image, target)")
            tile_tensor, transformed_target = transformed
            transformed_boxes = transformed_target.get("boxes")
            if not torch.is_tensor(transformed_boxes) or transformed_boxes.numel() < 4:
                raise ValueError("tile transform lost its object bbox")
            images.append(tile_tensor)
            source_bboxes.append(source_bbox.to(dtype=torch.float32))
            crop_bboxes.append(transformed_boxes[0].to(dtype=torch.float32).clamp(0.0, 1.0))
            masks.append(_normalize_mask(transformed_target.get("image_mask"), tile_tensor))

        return {
            "images": torch.stack(images, dim=0),
            "bbox": torch.stack(source_bboxes, dim=0),
            "crop_bbox": torch.stack(crop_bboxes, dim=0),
            "image_mask": torch.stack(masks, dim=0),
            "label": int(label),
            "sample_index": int(index),
            "image_path": str(geometry["image_path"]),
            "source_stem": str(geometry["source_stem"]),
            "crop_width": int(geometry["crop_width"]),
            "crop_height": int(geometry["crop_height"]),
            "crop_min_side": int(geometry["crop_min_side"]),
        }


def collate_surface_tiles(samples: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "images": torch.stack([sample["images"] for sample in samples], dim=0),
        "bbox": torch.stack([sample["bbox"] for sample in samples], dim=0),
        "crop_bbox": torch.stack([sample["crop_bbox"] for sample in samples], dim=0),
        "image_mask": torch.stack([sample["image_mask"] for sample in samples], dim=0),
        "labels": torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long),
        "sample_indices": [int(sample["sample_index"]) for sample in samples],
        "image_paths": [str(sample["image_path"]) for sample in samples],
        "source_stems": [str(sample["source_stem"]) for sample in samples],
        "crop_widths": [int(sample["crop_width"]) for sample in samples],
        "crop_heights": [int(sample["crop_height"]) for sample in samples],
        "crop_min_sides": [int(sample["crop_min_side"]) for sample in samples],
    }


def _focus_metrics(metrics: Mapping[str, object], focus_class: int = 1) -> Dict[str, float]:
    for row in metrics.get("per_class", []):
        if int(row.get("class_index", -1)) == int(focus_class):
            return {
                "precision": float(row.get("precision", 0.0) or 0.0),
                "recall": float(row.get("recall", 0.0) or 0.0),
                "f1": float(row.get("f1", 0.0) or 0.0),
            }
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _compact_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "class1": _focus_metrics(metrics),
        "per_class": metrics.get("per_class", []),
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    values = np.asarray(labels, dtype=np.int64)
    if values.size < 2 or np.unique(values).size < 2:
        return None
    return float(roc_auc_score(values, np.asarray(scores, dtype=np.float64)))


def transition_audit(
    targets: Tensor,
    global_probabilities: Tensor,
    candidate_probabilities: Tensor,
) -> Dict[str, object]:
    targets = targets.to(dtype=torch.long)
    global_predictions = global_probabilities.argmax(dim=1)
    candidate_predictions = candidate_probabilities.argmax(dim=1)
    global_correct = global_predictions.eq(targets)
    candidate_correct = candidate_predictions.eq(targets)
    changed = global_predictions.ne(candidate_predictions)
    corrections = changed & ~global_correct & candidate_correct
    harms = changed & global_correct & ~candidate_correct
    neutral = changed & ~global_correct & ~candidate_correct
    focus = 1
    global_fn = targets.eq(focus) & global_predictions.ne(focus)
    global_tp = targets.eq(focus) & global_predictions.eq(focus)
    global_fp = targets.ne(focus) & global_predictions.eq(focus)
    global_tn = targets.ne(focus) & global_predictions.ne(focus)
    return {
        "changed": int(changed.sum().item()),
        "corrections": int(corrections.sum().item()),
        "harms": int(harms.sum().item()),
        "neutral": int(neutral.sum().item()),
        "class1_fn_rescued": int((global_fn & candidate_predictions.eq(focus)).sum().item()),
        "class1_tp_broken": int((global_tp & candidate_predictions.ne(focus)).sum().item()),
        "class1_fp_removed": int((global_fp & candidate_predictions.ne(focus)).sum().item()),
        "class1_fp_created": int((global_tn & candidate_predictions.eq(focus)).sum().item()),
    }


def direction_audit(
    targets: Tensor,
    global_probabilities: Tensor,
    tile_probabilities: Tensor,
) -> Dict[str, object]:
    global_predictions = global_probabilities.argmax(dim=1)
    tile_mean = tile_probabilities.mean(dim=1)
    delta = tile_mean[:, 1] - global_probabilities[:, 1]
    fn_mask = targets.eq(1) & global_predictions.ne(1)
    fp_mask = targets.ne(1) & global_predictions.eq(1)
    direction_labels = torch.cat(
        (
            torch.ones(int(fn_mask.sum().item()), dtype=torch.long),
            torch.zeros(int(fp_mask.sum().item()), dtype=torch.long),
        )
    )
    direction_scores = torch.cat((delta[fn_mask], delta[fp_mask]))
    auc = _safe_auc(direction_labels.tolist(), direction_scores.tolist())
    return {
        "fn_count": int(fn_mask.sum().item()),
        "fp_count": int(fp_mask.sum().item()),
        "fn_mean_tile_minus_global_p1": (
            float(delta[fn_mask].mean().item()) if bool(fn_mask.any()) else None
        ),
        "fp_mean_tile_minus_global_p1": (
            float(delta[fp_mask].mean().item()) if bool(fp_mask.any()) else None
        ),
        "fn_vs_fp_delta_auroc": auc,
    }


def _geometry_summary(rows: Sequence[Mapping[str, object]], image_size: int) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, selected in (
        ("all", list(rows)),
        ("class1", [row for row in rows if int(row["target_index"]) == 1]),
    ):
        values = np.asarray([float(row["crop_min_side"]) for row in selected], dtype=np.float64)
        if values.size == 0:
            result[key] = {"count": 0}
            continue
        result[key] = {
            "count": int(values.size),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "fraction_at_least_model_input": float(np.mean(values >= int(image_size))),
            "fraction_at_least_384": float(np.mean(values >= 384)),
        }
    return result


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        return
    fieldnames: List[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


@torch.inference_mode()
def collect_split(
    *,
    split: str,
    dataset: OriginalResolutionSurfaceTileDataset,
    model: nn.Module,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    bbox_token_prior_source: str,
) -> Dict[str, object]:
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context=f"highres_surface_tiles_{split}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_surface_tiles,
        **loader_kwargs,
    )
    all_targets: List[Tensor] = []
    all_global: List[Tensor] = []
    all_tiles: List[Tensor] = []
    rows: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc=f"surface-tiles:{split}", unit="batch"):
        images = batch["images"].to(device=device, non_blocking=True)
        bbox = batch["bbox"].to(device=device, non_blocking=True)
        crop_bbox = batch["crop_bbox"].to(device=device, non_blocking=True)
        image_mask = batch["image_mask"].to(device=device, non_blocking=True)
        view_logits: List[Tensor] = []
        for view_index in range(images.size(1)):
            prior = _select_bbox_token_prior(
                bbox=bbox[:, view_index],
                crop_bbox=crop_bbox[:, view_index],
                source=bbox_token_prior_source,
            )
            logits, _ = _forward_logits(
                model=model,
                images=images[:, view_index],
                image_valid_mask=image_mask[:, view_index],
                bbox_metadata=bbox[:, view_index],
                bbox_token_prior=prior,
                amp=bool(amp),
                device=device,
            )
            view_logits.append(logits.cpu())
        probabilities = torch.softmax(torch.stack(view_logits, dim=1).float(), dim=2)
        global_probabilities = probabilities[:, 0]
        tile_probabilities = probabilities[:, 1:]
        candidate_probabilities = fuse_global_and_tiles(global_probabilities, tile_probabilities)
        targets = batch["labels"].to(dtype=torch.long)
        global_predictions = global_probabilities.argmax(dim=1)
        candidate_predictions = candidate_probabilities.argmax(dim=1)
        tile_mean = tile_probabilities.mean(dim=1)
        tile_predictions = tile_probabilities.argmax(dim=2)
        for row_index in range(targets.numel()):
            row: Dict[str, object] = {
                "split": split,
                "sample_index": int(batch["sample_indices"][row_index]),
                "image_path": str(batch["image_paths"][row_index]),
                "source_stem": str(batch["source_stems"][row_index]),
                "target_index": int(targets[row_index].item()),
                "global_prediction_index": int(global_predictions[row_index].item()),
                "candidate_prediction_index": int(candidate_predictions[row_index].item()),
                "global_correct": int(global_predictions[row_index].eq(targets[row_index]).item()),
                "candidate_correct": int(candidate_predictions[row_index].eq(targets[row_index]).item()),
                "crop_width": int(batch["crop_widths"][row_index]),
                "crop_height": int(batch["crop_heights"][row_index]),
                "crop_min_side": int(batch["crop_min_sides"][row_index]),
                "global_p1": float(global_probabilities[row_index, 1].item()),
                "tile_mean_p1": float(tile_mean[row_index, 1].item()),
                "tile_minus_global_p1": float(
                    (tile_mean[row_index, 1] - global_probabilities[row_index, 1]).item()
                ),
                "candidate_p1": float(candidate_probabilities[row_index, 1].item()),
                "any_tile_correct": int(
                    tile_predictions[row_index].eq(targets[row_index]).any().item()
                ),
            }
            for class_index in range(len(class_names)):
                row[f"global_prob_{class_index}"] = float(
                    global_probabilities[row_index, class_index].item()
                )
                row[f"tile_mean_prob_{class_index}"] = float(
                    tile_mean[row_index, class_index].item()
                )
                row[f"candidate_prob_{class_index}"] = float(
                    candidate_probabilities[row_index, class_index].item()
                )
            for tile_index, (tile_name, _, _) in enumerate(TILE_POSITIONS):
                row[f"{tile_name}_prediction_index"] = int(
                    tile_predictions[row_index, tile_index].item()
                )
                row[f"{tile_name}_p1"] = float(
                    tile_probabilities[row_index, tile_index, 1].item()
                )
            rows.append(row)
        all_targets.append(targets)
        all_global.append(global_probabilities)
        all_tiles.append(tile_probabilities)

    targets = torch.cat(all_targets, dim=0)
    global_probabilities = torch.cat(all_global, dim=0)
    tile_probabilities = torch.cat(all_tiles, dim=0)
    candidate_probabilities = fuse_global_and_tiles(global_probabilities, tile_probabilities)
    global_metrics = build_metrics(
        targets,
        global_probabilities.argmax(dim=1),
        class_names,
        probabilities=global_probabilities,
    )
    tile_metrics = build_metrics(
        targets,
        tile_probabilities.mean(dim=1).argmax(dim=1),
        class_names,
        probabilities=tile_probabilities.mean(dim=1),
    )
    candidate_metrics = build_metrics(
        targets,
        candidate_probabilities.argmax(dim=1),
        class_names,
        probabilities=candidate_probabilities,
    )
    return {
        "split": split,
        "sample_count": int(targets.numel()),
        "source_group_count": int(len({str(row["source_stem"]) for row in rows})),
        "loader": loader_summary,
        "global_metrics": _compact_metrics(global_metrics),
        "tile_mean_metrics": _compact_metrics(tile_metrics),
        "candidate_metrics": _compact_metrics(candidate_metrics),
        "transitions": transition_audit(targets, global_probabilities, candidate_probabilities),
        "direction": direction_audit(targets, global_probabilities, tile_probabilities),
        "geometry": _geometry_summary(rows, int(images.shape[-1])),
        "any_tile_correct_fraction": float(
            np.mean([int(row["any_tile_correct"]) for row in rows])
        ),
        "rows": rows,
        "targets": targets,
        "global_probabilities": global_probabilities,
        "tile_probabilities": tile_probabilities,
        "candidate_probabilities": candidate_probabilities,
    }


def assess_smoke_permission(
    *,
    train_result: Mapping[str, object],
    val_result: Mapping[str, object],
    source_overlap_count: int,
    expected_train_count: int,
    expected_val_count: int,
    preflight: bool,
) -> Dict[str, object]:
    val_global = val_result["global_metrics"]
    val_candidate = val_result["candidate_metrics"]
    global_focus = val_global["class1"]
    candidate_focus = val_candidate["class1"]
    transitions = val_result["transitions"]
    train_auc = train_result["direction"].get("fn_vs_fp_delta_auroc")
    val_auc = val_result["direction"].get("fn_vs_fp_delta_auroc")
    checks = {
        "full_train_coverage": int(train_result["sample_count"]) == int(expected_train_count),
        "full_val_coverage": int(val_result["sample_count"]) == int(expected_val_count),
        "no_train_val_source_overlap": int(source_overlap_count) == 0,
        "global_keeper_macro_reproduced": (
            abs(float(val_global["macro_f1"]) - EXPECTED_KEEPER_VAL_MACRO) <= 0.006
        ),
        "global_keeper_class1_reproduced": (
            abs(float(global_focus["f1"]) - EXPECTED_KEEPER_VAL_CLASS1) <= 0.012
        ),
        "candidate_macro_noninferior": (
            float(val_candidate["macro_f1"]) >= float(val_global["macro_f1"]) - 0.001
        ),
        "candidate_class1_gain": (
            float(candidate_focus["f1"]) >= float(global_focus["f1"]) + 0.005
        ),
        "candidate_class1_milestone": float(candidate_focus["f1"]) >= 0.70,
        "candidate_class1_recall_preserved": (
            float(candidate_focus["recall"]) >= float(global_focus["recall"])
        ),
        "more_corrections_than_harms": (
            int(transitions["corrections"]) > int(transitions["harms"])
        ),
        "class1_fp_conservative": (
            int(transitions["class1_fp_removed"]) >= int(transitions["class1_fp_created"])
        ),
        "class1_recall_action_positive": (
            int(transitions["class1_fn_rescued"]) > int(transitions["class1_tp_broken"])
        ),
        "train_delta_direction": train_auc is not None and float(train_auc) >= 0.60,
        "val_delta_direction": val_auc is not None and float(val_auc) >= 0.60,
        "not_preflight": not bool(preflight),
    }
    return {
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": int(len(checks)),
        "smoke_permission": bool(all(checks.values())),
    }


def _plot_direction(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        rows = result["rows"]
        fn = [
            float(row["tile_minus_global_p1"])
            for row in rows
            if int(row["target_index"]) == 1 and int(row["global_prediction_index"]) != 1
        ]
        fp = [
            float(row["tile_minus_global_p1"])
            for row in rows
            if int(row["target_index"]) != 1 and int(row["global_prediction_index"]) == 1
        ]
        bins = np.linspace(-0.35, 0.35, 36)
        axis.hist(fn, bins=bins, alpha=0.65, label=f"class1 FN (n={len(fn)})")
        axis.hist(fp, bins=bins, alpha=0.65, label=f"class1 FP (n={len(fp)})")
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(str(result["split"]))
        axis.set_xlabel("tile mean p1 - global p1")
        axis.set_ylabel("count")
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_metrics(path: Path, val_result: Mapping[str, object]) -> None:
    methods = ("global", "tile_mean", "candidate")
    payloads = (
        val_result["global_metrics"],
        val_result["tile_mean_metrics"],
        val_result["candidate_metrics"],
    )
    macro = [float(payload["macro_f1"]) for payload in payloads]
    focus = [float(payload["class1"]["f1"]) for payload in payloads]
    x = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.bar(x - 0.18, macro, width=0.36, label="macro F1")
    axis.bar(x + 0.18, focus, width=0.36, label="class1 F1")
    axis.set_xticks(x, methods)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_contact_sheet(
    path: Path,
    dataset: OriginalResolutionSurfaceTileDataset,
    rows: Sequence[Mapping[str, object]],
    max_rows: int = 12,
) -> None:
    focus_rows = [
        row
        for row in rows
        if (
            int(row["target_index"]) == 1
            and int(row["global_prediction_index"]) != 1
        )
        or (
            int(row["target_index"]) != 1
            and int(row["global_prediction_index"]) == 1
        )
    ]
    focus_rows.sort(key=lambda row: abs(float(row["tile_minus_global_p1"])), reverse=True)
    selected = focus_rows[: int(max_rows)]
    if not selected:
        return
    thumb_size = 144
    header_height = 34
    column_names = ("global",) + tuple(name for name, _, _ in TILE_POSITIONS)
    canvas = Image.new(
        "RGB",
        (thumb_size * len(column_names), (thumb_size + header_height) * len(selected)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(selected):
        views, _ = dataset.render_views(int(row["sample_index"]))
        y = row_index * (thumb_size + header_height)
        title = (
            f"idx={int(row['sample_index'])} y={int(row['target_index'])} "
            f"g={int(row['global_prediction_index'])} c={int(row['candidate_prediction_index'])} "
            f"dp1={float(row['tile_minus_global_p1']):+.3f}"
        )
        draw.text((4, y + 2), title, fill="black")
        for column_index, (name, image) in enumerate(views):
            thumb = image.copy()
            thumb.thumbnail((thumb_size - 4, thumb_size - 18), Image.Resampling.LANCZOS)
            x = column_index * thumb_size
            image_y = y + header_height + 16
            canvas.paste(thumb, (x + (thumb_size - thumb.width) // 2, image_y))
            draw.text((x + 3, y + header_height), name, fill="black")
    canvas.save(path)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    files = []
    for path in sorted(output_dir.rglob("*"), key=lambda value: str(value).casefold()):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "name": str(path.relative_to(output_dir)).replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    aggregate = hashlib.sha256()
    for row in files:
        aggregate.update(str(row["name"]).encode("utf-8"))
        aggregate.update(str(row["sha256"]).encode("ascii"))
    manifest = {
        "mode": "highres_surface_tiles_readiness_evidence_manifest",
        "payload_count": int(len(files)),
        "payload_size_bytes": int(sum(int(row["size_bytes"]) for row in files)),
        "payload_manifest_sha256": aggregate.hexdigest(),
        "files": files,
        "contains_checkpoint": any(str(row["name"]).lower().endswith(".pt") for row in files),
        "contains_model_binary": any(
            str(row["name"]).lower().endswith((".pt", ".pth", ".engine")) for row in files
        ),
        "contains_test_payload": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _public_result(result: Mapping[str, object]) -> Dict[str, object]:
    excluded = {
        "rows",
        "targets",
        "global_probabilities",
        "tile_probabilities",
        "candidate_probabilities",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1 or int(args.num_workers) < 0:
        raise ValueError("invalid batch/worker settings")
    capped = int(args.max_train_samples) > 0 or int(args.max_val_samples) > 0
    if capped and not bool(args.allow_preflight):
        raise ValueError("sample caps require --allow-preflight")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)
    set_seed(int(args.seed))

    checkpoint_path = Path(args.checkpoint).resolve()
    data_path = Path(args.data).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if class_names != list(data_spec.class_names):
        raise ValueError("checkpoint and dataset class orders differ")
    if str(checkpoint.get("model_config", {}).get("model_type", "")).lower() not in {
        "vit_registers",
        "trkh",
    }:
        raise ValueError("surface-tile audit is locked to a TRKH classification checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(dict(checkpoint), num_classes=len(class_names))
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    transform = _build_eval_transform_from_checkpoint(dict(checkpoint), image_size=image_size)
    base_datasets: Dict[str, MangoYOLOCropDataset] = {}
    datasets: Dict[str, OriginalResolutionSurfaceTileDataset] = {}
    caps = {"train": int(args.max_train_samples), "val": int(args.max_val_samples)}
    for split in ("train", "val"):
        base = _build_classification_dataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            checkpoint=dict(checkpoint),
        )
        if not isinstance(base, MangoYOLOCropDataset):
            raise TypeError("expected MangoYOLOCropDataset")
        base_datasets[split] = base
        datasets[split] = OriginalResolutionSurfaceTileDataset(base, max_samples=caps[split])

    bbox_token_prior_source = str(
        checkpoint.get("model_config", {}).get("bbox_token_prior_source", "crop_bbox")
        or "crop_bbox"
    )
    train_result = collect_split(
        split="train",
        dataset=datasets["train"],
        model=model,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
        bbox_token_prior_source=bbox_token_prior_source,
    )
    val_result = collect_split(
        split="val",
        dataset=datasets["val"],
        model=model,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
        bbox_token_prior_source=bbox_token_prior_source,
    )
    train_sources = {str(row["source_stem"]) for row in train_result["rows"]}
    val_sources = {str(row["source_stem"]) for row in val_result["rows"]}
    source_overlap = sorted(train_sources & val_sources)
    gate = assess_smoke_permission(
        train_result=train_result,
        val_result=val_result,
        source_overlap_count=len(source_overlap),
        expected_train_count=len(base_datasets["train"]),
        expected_val_count=len(base_datasets["val"]),
        preflight=capped,
    )

    _write_csv(output_dir / "train_predictions.csv", train_result["rows"])
    _write_csv(output_dir / "val_predictions.csv", val_result["rows"])
    metric_rows: List[Dict[str, object]] = []
    for result in (train_result, val_result):
        for method_key, method_name in (
            ("global_metrics", "global"),
            ("tile_mean_metrics", "tile_mean"),
            ("candidate_metrics", "candidate"),
        ):
            metrics = result[method_key]
            metric_rows.append(
                {
                    "split": result["split"],
                    "method": method_name,
                    "macro_f1": metrics["macro_f1"],
                    "class1_precision": metrics["class1"]["precision"],
                    "class1_recall": metrics["class1"]["recall"],
                    "class1_f1": metrics["class1"]["f1"],
                }
            )
    _write_csv(output_dir / "metrics_comparison.csv", metric_rows)
    _plot_direction(output_dir / "class1_fn_fp_tile_direction.png", (train_result, val_result))
    _plot_metrics(output_dir / "validation_metrics.png", val_result)
    _render_contact_sheet(
        output_dir / "validation_surface_tile_contact_sheet.png",
        datasets["val"],
        val_result["rows"],
    )

    summary = {
        "mode": "highres_surface_tiles_readiness",
        "protocol_locked": True,
        "preflight": bool(capped),
        "test_split_used": False,
        "raw_dataset_modified": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "data": str(data_path),
        "device": str(device),
        "amp": bool(args.amp),
        "image_size": int(image_size),
        "class_names": class_names,
        "protocol": {
            "global_view": "exact checkpoint object crop and evaluation transform",
            "local_view_source": "original-resolution bbox crop before model resize",
            "tile_fraction": TILE_FRACTION,
            "tile_positions": [name for name, _, _ in TILE_POSITIONS],
            "candidate": "0.50 global probabilities + 0.50 mean of five tile probabilities",
            "candidate_tile_weight": CANDIDATE_TILE_WEIGHT,
            "bbox_token_prior_source": bbox_token_prior_source,
            "model_fit": False,
            "threshold_or_weight_sweep": False,
            "train_probability_caveat": (
                "keeper train predictions are in-sample; train view deltas are transfer support, "
                "not current-keeper OOF evidence"
            ),
            "primary_sources": [
                "https://openaccess.thecvf.com/content_ECCV_2018/html/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.html",
                "https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3399_ECCV_2020_paper.php",
            ],
            "fastvit_no_run_triage": {
                "paper": "https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html",
                "local_timm_variant": "fastvit_sa12",
                "layers": [2, 2, 6, 2],
                "token_mixers": ["repmixer", "repmixer", "repmixer", "attention"],
                "repmixer_kernel_size": 3,
                "decision": (
                    "no launch: another staged stock hybrid does not provide a new class1-positive "
                    "supervision source after the documented stock-backbone closure"
                ),
            },
        },
        "source_overlap": {
            "count": int(len(source_overlap)),
            "examples": source_overlap[:10],
        },
        "artifact_manifest_path": "artifact_manifest.json",
        "train": _public_result(train_result),
        "val": _public_result(val_result),
        "gate": gate,
    }
    json_dump(output_dir / "summary.json", summary)
    val_global = summary["val"]["global_metrics"]
    val_candidate = summary["val"]["candidate_metrics"]
    readme_lines = [
        "# Original-Resolution Surface-Tile Readiness Audit",
        "",
        "- No test split, model fitting, threshold sweep, checkpoint output, or raw-data edit.",
        f"- Global val macro/class1: `{val_global['macro_f1']:.6f}/{val_global['class1']['f1']:.6f}`.",
        f"- Candidate val macro/class1: `{val_candidate['macro_f1']:.6f}/{val_candidate['class1']['f1']:.6f}`.",
        f"- Candidate class1 P/R: `{val_candidate['class1']['precision']:.6f}/{val_candidate['class1']['recall']:.6f}`.",
        f"- Val transitions: `{summary['val']['transitions']}`.",
        f"- Train/val FN-vs-FP delta AUROC: `{summary['train']['direction']['fn_vs_fp_delta_auroc']}` / `{summary['val']['direction']['fn_vs_fp_delta_auroc']}`.",
        f"- Smoke permission: `{gate['smoke_permission']}` ({gate['passed']}/{gate['total']} checks).",
        "",
        "The train keeper probabilities are in-sample. They are reported only to test whether the fixed view delta transfers in sign; they are not described as OOF evidence.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    _write_artifact_manifest(output_dir)
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()

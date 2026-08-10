from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    set_seed,
)
from trkh.data.dataset import (
    MangoYOLOCropDataset,
    PairedViewTrainDataset,
    build_train_collate_fn,
)
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _checkpoint_augmentation_config,
    _class1_f1,
    _forward_logits,
    _select_bbox_token_prior,
)


@dataclass(frozen=True)
class BBoxAlignedPatchMatch:
    primary_local_indices: Tensor
    paired_local_indices: Tensor
    control_paired_local_indices: Tensor
    paired_candidate_local_indices: Tensor
    target_candidate_positions: Tensor
    geometric_distances: Tensor


def patch_centers_from_indices(indices: Tensor, grid_size: Sequence[int]) -> Tensor:
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if len(grid_size) != 2:
        raise ValueError("grid_size must contain height and width")
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_size values must be positive")
    values = indices.to(dtype=torch.long)
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= grid_h * grid_w):
        raise ValueError("patch index is outside the grid")
    x = (values.remainder(grid_w).to(dtype=torch.float32) + 0.5) / float(grid_w)
    y = (torch.div(values, grid_w, rounding_mode="floor").to(dtype=torch.float32) + 0.5) / float(
        grid_h
    )
    return torch.stack((x, y), dim=1)


def _xywh_bounds(bbox: Tensor) -> Tuple[Tensor, Tensor]:
    value = bbox.to(dtype=torch.float32).flatten()
    if value.numel() < 4:
        raise ValueError("bbox must contain cx, cy, width, height")
    center = value[:2].clamp(0.0, 1.0)
    size = value[2:4].clamp(min=1e-4, max=1.0)
    lower = (center - 0.5 * size).clamp(0.0, 1.0)
    upper = (center + 0.5 * size).clamp(0.0, 1.0)
    size = (upper - lower).clamp(min=1e-4)
    return lower, size


def build_bbox_aligned_patch_match(
    *,
    primary_patch_indices: Tensor,
    paired_patch_indices: Tensor,
    primary_grid_size: Sequence[int],
    paired_grid_size: Sequence[int],
    primary_bbox: Tensor,
    paired_bbox: Tensor,
    primary_valid_mask: Optional[Tensor] = None,
    paired_valid_mask: Optional[Tensor] = None,
    max_pairs: int = 64,
    max_geometric_distance: float = 0.12,
) -> Optional[BBoxAlignedPatchMatch]:
    primary_centers = patch_centers_from_indices(primary_patch_indices, primary_grid_size)
    paired_centers = patch_centers_from_indices(paired_patch_indices, paired_grid_size)
    device = primary_centers.device
    paired_centers = paired_centers.to(device=device)

    primary_valid = torch.ones(primary_centers.size(0), dtype=torch.bool, device=device)
    if torch.is_tensor(primary_valid_mask):
        if primary_valid_mask.numel() != primary_valid.numel():
            raise ValueError("primary_valid_mask shape mismatch")
        primary_valid &= primary_valid_mask.to(device=device, dtype=torch.bool).flatten()
    paired_valid = torch.ones(paired_centers.size(0), dtype=torch.bool, device=device)
    if torch.is_tensor(paired_valid_mask):
        if paired_valid_mask.numel() != paired_valid.numel():
            raise ValueError("paired_valid_mask shape mismatch")
        paired_valid &= paired_valid_mask.to(device=device, dtype=torch.bool).flatten()

    primary_lower, primary_size = _xywh_bounds(primary_bbox.to(device=device))
    paired_lower, paired_size = _xywh_bounds(paired_bbox.to(device=device))
    primary_upper = primary_lower + primary_size
    paired_upper = paired_lower + paired_size
    primary_inside = (
        (primary_centers[:, 0] >= primary_lower[0])
        & (primary_centers[:, 0] <= primary_upper[0])
        & (primary_centers[:, 1] >= primary_lower[1])
        & (primary_centers[:, 1] <= primary_upper[1])
    )
    paired_inside = (
        (paired_centers[:, 0] >= paired_lower[0])
        & (paired_centers[:, 0] <= paired_upper[0])
        & (paired_centers[:, 1] >= paired_lower[1])
        & (paired_centers[:, 1] <= paired_upper[1])
    )
    primary_local = torch.nonzero(primary_valid & primary_inside, as_tuple=False).flatten()
    paired_candidates = torch.nonzero(paired_valid & paired_inside, as_tuple=False).flatten()
    if primary_local.numel() == 0 or paired_candidates.numel() == 0:
        return None

    if int(max_pairs) > 0 and primary_local.numel() > int(max_pairs):
        selected = torch.linspace(
            0,
            primary_local.numel() - 1,
            steps=int(max_pairs),
            device=device,
        ).round().to(dtype=torch.long)
        primary_local = primary_local.index_select(0, selected)

    relative = (
        primary_centers.index_select(0, primary_local) - primary_lower.unsqueeze(0)
    ) / primary_size.unsqueeze(0)
    relative = relative.clamp(0.0, 1.0)
    target_centers = paired_lower.unsqueeze(0) + relative * paired_size.unsqueeze(0)
    candidate_centers = paired_centers.index_select(0, paired_candidates)
    distances = torch.cdist(target_centers, candidate_centers, p=2)
    nearest_distance, nearest_candidate_position = distances.min(dim=1)
    keep = nearest_distance <= float(max_geometric_distance)
    if not bool(keep.any()):
        return None
    primary_local = primary_local[keep]
    nearest_distance = nearest_distance[keep]
    nearest_candidate_position = nearest_candidate_position[keep]
    paired_local = paired_candidates.index_select(0, nearest_candidate_position)

    candidate_count = int(paired_candidates.numel())
    if candidate_count > 1:
        shift = max(1, candidate_count // 2)
        control_positions = (nearest_candidate_position + shift).remainder(candidate_count)
    else:
        control_positions = nearest_candidate_position
    control_local = paired_candidates.index_select(0, control_positions)
    return BBoxAlignedPatchMatch(
        primary_local_indices=primary_local,
        paired_local_indices=paired_local,
        control_paired_local_indices=control_local,
        paired_candidate_local_indices=paired_candidates,
        target_candidate_positions=nearest_candidate_position,
        geometric_distances=nearest_distance,
    )


def correspondence_statistics(
    primary_tokens: Tensor,
    paired_tokens: Tensor,
    match: BBoxAlignedPatchMatch,
) -> Dict[str, float]:
    if primary_tokens.ndim != 2 or paired_tokens.ndim != 2:
        raise ValueError("tokens must have shape [token, channel]")
    if primary_tokens.size(1) != paired_tokens.size(1):
        raise ValueError("primary and paired token dimensions must match")
    primary = F.normalize(
        primary_tokens.index_select(0, match.primary_local_indices).float(),
        dim=1,
        eps=1e-6,
    )
    paired = F.normalize(
        paired_tokens.index_select(0, match.paired_local_indices).float(),
        dim=1,
        eps=1e-6,
    )
    control = F.normalize(
        paired_tokens.index_select(0, match.control_paired_local_indices).float(),
        dim=1,
        eps=1e-6,
    )
    candidates = F.normalize(
        paired_tokens.index_select(0, match.paired_candidate_local_indices).float(),
        dim=1,
        eps=1e-6,
    )
    matched_cosine = (primary * paired).sum(dim=1)
    control_cosine = (primary * control).sum(dim=1)
    similarity = primary @ candidates.transpose(0, 1)
    target_similarity = similarity.gather(1, match.target_candidate_positions.unsqueeze(1)).squeeze(1)
    ranks = 1 + (similarity > target_similarity.unsqueeze(1)).sum(dim=1)
    top1 = (ranks == 1).to(dtype=torch.float32)
    reciprocal_rank = ranks.to(dtype=torch.float32).reciprocal()
    random_top1 = 1.0 / float(max(1, candidates.size(0)))
    return {
        "pair_count": float(primary.size(0)),
        "paired_candidate_count": float(candidates.size(0)),
        "matched_cosine": float(matched_cosine.mean().item()),
        "control_cosine": float(control_cosine.mean().item()),
        "correspondence_margin": float((matched_cosine - control_cosine).mean().item()),
        "feature_top1_rate": float(top1.mean().item()),
        "random_top1_rate": float(random_top1),
        "feature_top1_lift": float(top1.mean().item() - random_top1),
        "mean_reciprocal_rank": float(reciprocal_rank.mean().item()),
        "geometric_distance": float(match.geometric_distances.float().mean().item()),
        "geometric_distance_max": float(match.geometric_distances.float().max().item()),
    }


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        return
    fieldnames: List[str] = []
    for row in records:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


_CORRESPONDENCE_FIELDS = (
    "pair_count",
    "paired_candidate_count",
    "matched_cosine",
    "control_cosine",
    "correspondence_margin",
    "feature_top1_rate",
    "random_top1_rate",
    "feature_top1_lift",
    "mean_reciprocal_rank",
    "geometric_distance",
    "geometric_distance_max",
)


def summarize_correspondence_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    valid = [row for row in rows if int(row.get("match_valid", 0) or 0) == 1]
    summary: Dict[str, object] = {
        "sample_count": int(len(rows)),
        "valid_sample_count": int(len(valid)),
        "valid_sample_fraction": float(len(valid) / max(1, len(rows))),
    }
    for field in _CORRESPONDENCE_FIELDS:
        values = [float(row[field]) for row in valid if row.get(field, "") not in {"", None}]
        summary[f"mean_{field}"] = float(sum(values) / max(1, len(values)))
    return summary


def _row_groups(row: Mapping[str, object]) -> List[str]:
    target = int(row.get("target", -1))
    primary_prediction = int(row.get("primary_prediction", -1))
    groups = ["all", f"target_{target}"]
    groups.append("primary_correct" if target == primary_prediction else "primary_error")
    if target == 1 and primary_prediction == 1:
        groups.append("class1_tp")
    elif target == 1:
        groups.append("class1_fn")
    elif primary_prediction == 1:
        groups.append("class1_fp")
    return groups


def build_group_summaries(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    names: List[str] = []
    for row in rows:
        for name in _row_groups(row):
            if name not in names:
                names.append(name)
    summaries: List[Dict[str, object]] = []
    for name in names:
        subset = [row for row in rows if name in _row_groups(row)]
        summary = summarize_correspondence_rows(subset)
        summaries.append({"group": name, **summary})
    return summaries


def build_prediction_support(
    targets: Tensor,
    primary_predictions: Tensor,
    paired_predictions: Tensor,
) -> Dict[str, object]:
    targets = targets.to(dtype=torch.long).flatten()
    primary = primary_predictions.to(dtype=torch.long).flatten()
    paired = paired_predictions.to(dtype=torch.long).flatten()
    if targets.numel() != primary.numel() or targets.numel() != paired.numel():
        raise ValueError("prediction support inputs must have equal length")
    primary_correct = primary == targets
    paired_correct = paired == targets
    target_focus = targets == 1
    primary_focus = primary == 1
    paired_focus = paired == 1
    primary_fn = target_focus & (~primary_focus)
    primary_tp = target_focus & primary_focus
    primary_fp = (~target_focus) & primary_focus
    paired_fp = (~target_focus) & paired_focus
    fn_corrected = primary_fn & paired_focus
    tp_broken = primary_tp & (~paired_focus)
    fp_suppressed = primary_fp & (~paired_focus)
    fp_created = (~target_focus) & (~primary_focus) & paired_focus
    return {
        "sample_count": int(targets.numel()),
        "primary_correct": int(primary_correct.sum().item()),
        "paired_correct": int(paired_correct.sum().item()),
        "paired_corrections": int(((~primary_correct) & paired_correct).sum().item()),
        "paired_harms": int((primary_correct & (~paired_correct)).sum().item()),
        "class1_target_count": int(target_focus.sum().item()),
        "primary_class1_tp": int(primary_tp.sum().item()),
        "primary_class1_fn": int(primary_fn.sum().item()),
        "primary_class1_fp": int(primary_fp.sum().item()),
        "paired_class1_fp": int(paired_fp.sum().item()),
        "class1_fn_corrected_by_paired": int(fn_corrected.sum().item()),
        "class1_tp_broken_by_paired": int(tp_broken.sum().item()),
        "class1_fp_suppressed_by_paired": int(fp_suppressed.sum().item()),
        "class1_fp_created_by_paired": int(fp_created.sum().item()),
        "class1_fn_correction_rate": float(fn_corrected.sum().item() / max(1, primary_fn.sum().item())),
        "class1_tp_break_rate": float(tp_broken.sum().item() / max(1, primary_tp.sum().item())),
    }


def assess_correspondence_gate(
    *,
    overall: Mapping[str, object],
    prediction_support: Mapping[str, object],
    support_complete: bool,
    min_valid_sample_fraction: float = 0.95,
    min_correspondence_margin: float = 0.02,
    min_feature_top1_lift: float = 0.05,
    max_mean_geometric_distance: float = 0.08,
    min_class1_fn_corrections: int = 5,
    min_class1_fn_correction_rate: float = 0.15,
    max_class1_tp_break_rate: float = 0.08,
) -> Dict[str, object]:
    def number(values: Mapping[str, object], key: str, default: float) -> float:
        value = values.get(key)
        return float(default if value is None else value)

    geometry_checks = {
        "support_complete": bool(support_complete),
        "valid_sample_fraction": number(overall, "valid_sample_fraction", 0.0)
        >= float(min_valid_sample_fraction),
        "correspondence_margin": number(overall, "mean_correspondence_margin", 0.0)
        >= float(min_correspondence_margin),
        "feature_top1_lift": number(overall, "mean_feature_top1_lift", 0.0)
        >= float(min_feature_top1_lift),
        "geometric_distance": number(overall, "mean_geometric_distance", 1.0)
        <= float(max_mean_geometric_distance),
    }
    fn_corrections = int(prediction_support.get("class1_fn_corrected_by_paired", 0) or 0)
    tp_breaks = int(prediction_support.get("class1_tp_broken_by_paired", 0) or 0)
    fp_suppressed = int(prediction_support.get("class1_fp_suppressed_by_paired", 0) or 0)
    fp_created = int(prediction_support.get("class1_fp_created_by_paired", 0) or 0)
    class1_checks = {
        "minimum_fn_corrections": fn_corrections >= int(min_class1_fn_corrections),
        "fn_correction_rate": number(prediction_support, "class1_fn_correction_rate", 0.0)
        >= float(min_class1_fn_correction_rate),
        "tp_break_rate": number(prediction_support, "class1_tp_break_rate", 1.0)
        <= float(max_class1_tp_break_rate),
        "recall_budget_positive": fn_corrections > tp_breaks,
        "false_positive_budget_nonnegative": fp_suppressed >= fp_created,
    }
    geometry_ready = all(geometry_checks.values())
    class1_ready = all(class1_checks.values())
    failed = [
        f"geometry:{name}" for name, passed in geometry_checks.items() if not passed
    ] + [f"class1:{name}" for name, passed in class1_checks.items() if not passed]
    return {
        "geometry_signal_ready": bool(geometry_ready),
        "class1_teacher_signal_ready": bool(class1_ready),
        "smoke_ready": bool(geometry_ready and class1_ready),
        "training_permission": bool(geometry_ready and class1_ready),
        "geometry_checks": geometry_checks,
        "class1_checks": class1_checks,
        "failed_checks": failed,
        "thresholds": {
            "min_valid_sample_fraction": float(min_valid_sample_fraction),
            "min_correspondence_margin": float(min_correspondence_margin),
            "min_feature_top1_lift": float(min_feature_top1_lift),
            "max_mean_geometric_distance": float(max_mean_geometric_distance),
            "min_class1_fn_corrections": int(min_class1_fn_corrections),
            "min_class1_fn_correction_rate": float(min_class1_fn_correction_rate),
            "max_class1_tp_break_rate": float(max_class1_tp_break_rate),
        },
    }


def _compact_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: metrics.get(key)
        for key in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "per_class",
            "confusion_matrix",
        )
        if key in metrics
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact bbox-aligned patch correspondence between paired yolo_f and "
            "class_f views before adding a trainable dense alignment loss."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--paired-data", type=Path, required=True)
    parser.add_argument("--paired-classification-folder-yolo-data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--primary-view-mode",
        choices=("checkpoint_crop", "raw_context"),
        default="raw_context",
    )
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-pairs-per-sample", type=int, default=64)
    parser.add_argument("--max-geometric-distance", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-sample-fraction", type=float, default=0.95)
    parser.add_argument("--min-correspondence-margin", type=float, default=0.02)
    parser.add_argument("--min-feature-top1-lift", type=float, default=0.05)
    parser.add_argument("--max-mean-geometric-distance", type=float, default=0.08)
    parser.add_argument("--min-class1-fn-corrections", type=int, default=5)
    parser.add_argument("--min-class1-fn-correction-rate", type=float, default=0.15)
    parser.add_argument("--max-class1-tp-break-rate", type=float, default=0.08)
    return parser.parse_args()


def _build_primary_dataset(
    *,
    mode: str,
    data_spec,
    split: str,
    transform,
    checkpoint: Mapping[str, object],
):
    if mode == "checkpoint_crop":
        return _build_classification_dataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            checkpoint=dict(checkpoint),
        )
    if data_spec.data_format == "classification_folder":
        raise ValueError("raw_context primary view requires YOLO-format data")
    augmentation = _checkpoint_augmentation_config(dict(checkpoint))
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=transform,
        crop_margin_ratio=float(augmentation.get("crop_margin_ratio", 0.05) or 0.05),
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        class_aware_augmentation=False,
        classification_source_context=True,
        classification_source_context_mode="desaturate_blur",
        classification_source_context_layout="full",
        classification_source_context_margin_ratio=0.12,
        classification_source_context_background_alpha=1.0,
        classification_source_context_blur_radius=7.0,
        classification_bbox_metadata=True,
    )


def main() -> None:
    args = parse_args()
    if args.max_pairs_per_sample < 1:
        raise ValueError("--max-pairs-per-sample must be positive")
    if args.max_geometric_distance <= 0.0:
        raise ValueError("--max-geometric-distance must be positive")
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    primary_spec = load_data_spec(args.data, class_name_mode="raw", expected_num_classes=5)
    paired_spec = load_data_spec(
        args.paired_data,
        class_name_mode="raw",
        expected_num_classes=primary_spec.num_classes,
    )
    paired_yolo_spec = load_data_spec(
        args.paired_classification_folder_yolo_data,
        class_name_mode="raw",
        expected_num_classes=primary_spec.num_classes,
    )
    if paired_spec.class_names != primary_spec.class_names:
        raise ValueError("primary and paired class names do not match")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    class_names = list(checkpoint.get("class_names", primary_spec.class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(checkpoint=checkpoint, num_classes=len(class_names))
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 224))
    transform = _build_eval_transform_from_checkpoint(checkpoint, image_size=image_size)
    primary_dataset = _build_primary_dataset(
        mode=args.primary_view_mode,
        data_spec=primary_spec,
        split=args.split,
        transform=transform,
        checkpoint=checkpoint,
    )
    paired_dataset = _build_classification_dataset(
        data_spec=paired_spec,
        split=args.split,
        transform=transform,
        checkpoint=checkpoint,
        paired_yolo_data_spec=paired_yolo_spec,
    )
    dataset = PairedViewTrainDataset(
        primary_dataset=primary_dataset,
        paired_dataset=paired_dataset,
        primary_name=f"primary:{primary_spec.data_yaml.parent.name}:{args.primary_view_mode}",
        paired_name=f"paired:{paired_spec.data_yaml.parent.name}",
        require_all_matched=True,
    )
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=args.num_workers,
        requested_pin_memory=device.type == "cuda",
        context=f"paired_bbox_correspondence_{args.split}",
        prefetch_factor=2,
        persistent_workers=True,
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

    paths = [str(path) for path in dataset.sample_paths()]
    rows: List[Dict[str, object]] = []
    target_batches: List[Tensor] = []
    primary_probability_batches: List[Tensor] = []
    paired_probability_batches: List[Tensor] = []
    total_batches = len(loader)
    iterator = loader
    if args.max_batches > 0:
        total_batches = min(total_batches, int(args.max_batches))
        iterator = islice(loader, total_batches)
    sample_offset = 0
    with torch.inference_mode():
        for images, labels, metadata in tqdm(
            iterator,
            total=total_batches,
            desc=f"BBox-aligned correspondence ({args.primary_view_mode})",
            dynamic_ncols=True,
        ):
            metadata = metadata if isinstance(metadata, dict) else {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            targets = labels.argmax(dim=1) if labels.ndim == 2 else labels.to(dtype=torch.long)
            paired_images = metadata.get("paired_view_image")
            if not torch.is_tensor(paired_images):
                raise ValueError("paired_view_image is missing")
            paired_images = paired_images.to(device=device, non_blocking=True)

            def move(name: str, dtype: torch.dtype) -> Optional[Tensor]:
                value = metadata.get(name)
                return (
                    value.to(device=device, dtype=dtype, non_blocking=True)
                    if torch.is_tensor(value)
                    else None
                )

            bbox = move("bbox", torch.float32)
            crop_bbox = move("crop_bbox", torch.float32)
            image_mask = move("image_mask", torch.bool)
            paired_bbox = move("paired_view_bbox", torch.float32)
            paired_crop_bbox = move("paired_view_crop_bbox", torch.float32)
            paired_mask = move("paired_view_image_mask", torch.bool)
            primary_prior = _select_bbox_token_prior(
                bbox=bbox,
                crop_bbox=crop_bbox,
                source=args.bbox_token_prior_source,
            )
            paired_prior = _select_bbox_token_prior(
                bbox=paired_bbox,
                crop_bbox=paired_crop_bbox,
                source=args.bbox_token_prior_source,
            )
            primary_logits, primary_features = _forward_logits(
                model=model,
                images=images,
                image_valid_mask=image_mask,
                bbox_metadata=bbox,
                bbox_token_prior=primary_prior,
                amp=args.amp,
                device=device,
            )
            paired_logits, paired_features = _forward_logits(
                model=model,
                images=paired_images,
                image_valid_mask=paired_mask,
                bbox_metadata=paired_bbox,
                bbox_token_prior=paired_prior,
                amp=args.amp,
                device=device,
            )
            if not isinstance(primary_features, dict) or not isinstance(paired_features, dict):
                raise TypeError("model did not return paired feature dictionaries")
            primary_patches = primary_features.get("patches")
            paired_patches = paired_features.get("patches")
            primary_indices = primary_features.get("patch_indices")
            paired_indices = paired_features.get("patch_indices")
            if not all(
                torch.is_tensor(value)
                for value in (primary_patches, paired_patches, primary_indices, paired_indices)
            ):
                raise TypeError("model features do not expose patches and patch_indices")
            if crop_bbox is None or paired_crop_bbox is None:
                raise ValueError("crop_bbox metadata is required for geometric matching")
            primary_key_padding = primary_features.get("memory_key_padding_mask")
            paired_key_padding = paired_features.get("memory_key_padding_mask")
            primary_probabilities = torch.softmax(primary_logits.float(), dim=1)
            paired_probabilities = torch.softmax(paired_logits.float(), dim=1)
            primary_predictions = primary_probabilities.argmax(dim=1)
            paired_predictions = paired_probabilities.argmax(dim=1)

            for local_index in range(images.size(0)):
                primary_valid = None
                paired_valid = None
                if torch.is_tensor(primary_key_padding):
                    primary_valid = ~primary_key_padding[local_index].to(dtype=torch.bool)
                if torch.is_tensor(paired_key_padding):
                    paired_valid = ~paired_key_padding[local_index].to(dtype=torch.bool)
                match = build_bbox_aligned_patch_match(
                    primary_patch_indices=primary_indices[local_index],
                    paired_patch_indices=paired_indices[local_index],
                    primary_grid_size=primary_features["grid_size"],
                    paired_grid_size=paired_features["grid_size"],
                    primary_bbox=crop_bbox[local_index],
                    paired_bbox=paired_crop_bbox[local_index],
                    primary_valid_mask=primary_valid,
                    paired_valid_mask=paired_valid,
                    max_pairs=args.max_pairs_per_sample,
                    max_geometric_distance=args.max_geometric_distance,
                )
                global_index = sample_offset + local_index
                target = int(targets[local_index].item())
                primary_prediction = int(primary_predictions[local_index].item())
                paired_prediction = int(paired_predictions[local_index].item())
                row: Dict[str, object] = {
                    "sample_index": int(global_index),
                    "image_path": paths[global_index] if global_index < len(paths) else "",
                    "target": target,
                    "primary_prediction": primary_prediction,
                    "paired_prediction": paired_prediction,
                    "primary_correct": int(primary_prediction == target),
                    "paired_correct": int(paired_prediction == target),
                    "primary_confidence": float(primary_probabilities[local_index].max().item()),
                    "paired_confidence": float(paired_probabilities[local_index].max().item()),
                    "primary_bbox_cx": float(crop_bbox[local_index, 0].item()),
                    "primary_bbox_cy": float(crop_bbox[local_index, 1].item()),
                    "primary_bbox_w": float(crop_bbox[local_index, 2].item()),
                    "primary_bbox_h": float(crop_bbox[local_index, 3].item()),
                    "paired_bbox_w": float(paired_crop_bbox[local_index, 2].item()),
                    "paired_bbox_h": float(paired_crop_bbox[local_index, 3].item()),
                    "match_valid": int(match is not None),
                }
                if match is not None:
                    row.update(
                        correspondence_statistics(
                            primary_patches[local_index],
                            paired_patches[local_index],
                            match,
                        )
                    )
                rows.append(row)
            sample_offset += int(images.size(0))
            target_batches.append(targets.detach().cpu())
            primary_probability_batches.append(primary_probabilities.detach().cpu())
            paired_probability_batches.append(paired_probabilities.detach().cpu())

    targets = torch.cat(target_batches, dim=0)
    primary_probabilities = torch.cat(primary_probability_batches, dim=0)
    paired_probabilities = torch.cat(paired_probability_batches, dim=0)
    primary_predictions = primary_probabilities.argmax(dim=1)
    paired_predictions = paired_probabilities.argmax(dim=1)
    primary_metrics = build_metrics(
        targets=targets,
        predictions=primary_predictions,
        probabilities=primary_probabilities,
        class_names=class_names,
    )
    paired_metrics = build_metrics(
        targets=targets,
        predictions=paired_predictions,
        probabilities=paired_probabilities,
        class_names=class_names,
    )
    overall = summarize_correspondence_rows(rows)
    group_summaries = build_group_summaries(rows)
    prediction_support = build_prediction_support(targets, primary_predictions, paired_predictions)
    support_complete = int(targets.numel()) == int(len(dataset))
    gate = assess_correspondence_gate(
        overall=overall,
        prediction_support=prediction_support,
        support_complete=support_complete,
        min_valid_sample_fraction=args.min_valid_sample_fraction,
        min_correspondence_margin=args.min_correspondence_margin,
        min_feature_top1_lift=args.min_feature_top1_lift,
        max_mean_geometric_distance=args.max_mean_geometric_distance,
        min_class1_fn_corrections=args.min_class1_fn_corrections,
        min_class1_fn_correction_rate=args.min_class1_fn_correction_rate,
        max_class1_tp_break_rate=args.max_class1_tp_break_rate,
    )
    summary = {
        "mode": "paired_bbox_aligned_patch_correspondence_precheck",
        "guardrail": (
            "No train, no test, no raw-data edit, no trainable manifest. A future loss is "
            "permitted only when the full-support gate passes."
        ),
        "checkpoint": str(args.checkpoint),
        "data": str(primary_spec.data_yaml),
        "paired_data": str(paired_spec.data_yaml),
        "paired_yolo_data": str(paired_yolo_spec.data_yaml),
        "split": args.split,
        "primary_view_mode": args.primary_view_mode,
        "raw_context_definition": (
            "classification_source_context full layout with background_alpha=1.0; source pixels "
            "remain unchanged while transformed bbox metadata identifies the object"
            if args.primary_view_mode == "raw_context"
            else None
        ),
        "sample_count": int(targets.numel()),
        "dataset_sample_count": int(len(dataset)),
        "support_complete": bool(support_complete),
        "bbox_token_prior_source": args.bbox_token_prior_source,
        "max_pairs_per_sample": int(args.max_pairs_per_sample),
        "max_geometric_distance": float(args.max_geometric_distance),
        "dataloader": dataloader_summary,
        "primary_dataset_report": primary_dataset.quality_report(),
        "paired_dataset_report": paired_dataset.quality_report(),
        "paired_dataset_report_effective": dataset.quality_report(),
        "primary_metrics": _compact_metrics(primary_metrics),
        "paired_metrics": _compact_metrics(paired_metrics),
        "primary_class1_f1": float(_class1_f1(primary_metrics, 1)),
        "paired_class1_f1": float(_class1_f1(paired_metrics, 1)),
        "correspondence": overall,
        "prediction_support": prediction_support,
        "gate": gate,
        "literature": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract.html",
            "https://openaccess.thecvf.com/content/WACV2024/papers/Caron_Location-Aware_Self-Supervised_Transformers_for_Semantic_Segmentation_WACV_2024_paper.pdf",
            "https://openaccess.thecvf.com/content/CVPR2023/papers/Tao_Siamese_Image_Modeling_for_Self-Supervised_Vision_Representation_Learning_CVPR_2023_paper.pdf",
        ],
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
    }
    _write_rows_csv(output_dir / "per_sample_correspondence.csv", rows)
    _write_rows_csv(output_dir / "group_summary.csv", group_summaries)
    json_dump(output_dir / "summary.json", to_serializable(summary))
    readme = [
        "# Paired BBox-Aligned Patch Correspondence Precheck",
        "",
        f"- Primary view: `{args.primary_view_mode}`",
        f"- Split/support: `{args.split}` / `{int(targets.numel())}/{len(dataset)}`",
        f"- Geometry ready: `{str(gate['geometry_signal_ready']).lower()}`",
        f"- Class1 teacher ready: `{str(gate['class1_teacher_signal_ready']).lower()}`",
        f"- Smoke ready: `{str(gate['smoke_ready']).lower()}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        "",
        "This artifact is diagnostic only. It does not authorize test access or training when the gate is false.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(
        {
            "samples": int(targets.numel()),
            "primary_macro_f1": float(primary_metrics["macro_f1"]),
            "primary_class1_f1": float(_class1_f1(primary_metrics, 1)),
            "paired_macro_f1": float(paired_metrics["macro_f1"]),
            "paired_class1_f1": float(_class1_f1(paired_metrics, 1)),
            "correspondence_margin": overall.get("mean_correspondence_margin", 0.0),
            "feature_top1_lift": overall.get("mean_feature_top1_lift", 0.0),
            "smoke_ready": gate["smoke_ready"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

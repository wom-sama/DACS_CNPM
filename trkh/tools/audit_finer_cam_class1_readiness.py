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
import torch.nn.functional as F
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
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.metrics import build_metrics
from trkh.models.feature_hooks import resolve_feature_hook
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _forward_logits,
    _select_bbox_token_prior,
)
from trkh.tools.probe_embedding_prototypes import _collate_classification


SEED = 20260712
FOCUS_CLASS = 1
REFERENCE_COUNT = 3
FINER_ALPHA = 1.0
MASK_FRACTION = 0.05
INTERIOR_ERODE_RATIO = 0.15
FEATURE_SOURCE = "auto"
EXPECTED_KEEPER_VAL_MACRO = 0.882925
EXPECTED_KEEPER_VAL_CLASS1 = 0.678261

PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2025/html/"
    "Zhang_Finer-CAM_Spotting_the_Difference_Reveals_Finer_Details_"
    "for_Visual_Explanation_CVPR_2025_paper.html"
)
OFFICIAL_REPO_URL = "https://github.com/Imageomics/Finer-CAM"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train/validation-only Finer-CAM class-1 readiness audit. "
            "It never reads test, changes raw data, or writes a checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--allow-preflight",
        action="store_true",
        default=False,
        help="Allow capped train/validation prefixes for implementation checks only.",
    )
    return parser.parse_args(argv)


class PrefixDataset(Dataset):
    def __init__(self, dataset: Dataset, max_samples: int = 0) -> None:
        self.dataset = dataset
        self.length = min(len(dataset), int(max_samples)) if int(max_samples) > 0 else len(dataset)

    def __len__(self) -> int:
        return int(self.length)

    def __getitem__(self, index: int):
        return self.dataset[int(index)]

    def sample_paths(self) -> List[str]:
        sample_paths = getattr(self.dataset, "sample_paths", None)
        if not callable(sample_paths):
            raise TypeError("wrapped dataset must expose sample_paths()")
        return [str(value) for value in sample_paths()[: self.length]]


def select_reference_categories(
    logits: Tensor,
    *,
    target_class: int = FOCUS_CLASS,
    reference_count: int = REFERENCE_COUNT,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B,C]")
    class_count = int(logits.size(1))
    if not 0 <= int(target_class) < class_count:
        raise ValueError("target_class is outside the classifier range")
    count = min(max(1, int(reference_count)), class_count - 1)
    differences = (logits - logits[:, int(target_class) : int(target_class) + 1]).abs()
    differences = differences.clone()
    differences[:, int(target_class)] = float("inf")
    return differences.topk(k=count, dim=1, largest=False, sorted=True).indices


def finer_weighted_target(
    logits: Tensor,
    references: Tensor,
    *,
    target_class: int = FOCUS_CLASS,
    alpha: float = FINER_ALPHA,
) -> Tensor:
    if logits.ndim != 2 or references.ndim != 2:
        raise ValueError("logits/references must have shape [B,C] and [B,K]")
    if int(logits.size(0)) != int(references.size(0)):
        raise ValueError("logits/references batch sizes differ")
    if references.numel() == 0:
        return logits[:, int(target_class)]
    probabilities = torch.softmax(logits, dim=1)
    reference_probabilities = probabilities.gather(1, references)
    reference_logits = logits.gather(1, references)
    target_logits = logits[:, int(target_class)].unsqueeze(1)
    numerator = (
        reference_probabilities * (target_logits - float(alpha) * reference_logits)
    ).sum(dim=1)
    return numerator / reference_probabilities.sum(dim=1).clamp(min=1e-9)


def batched_gradcam(activations: Tensor, gradients: Tensor) -> Tensor:
    if activations.ndim != 4 or gradients.ndim != 4:
        raise ValueError("Grad-CAM activations/gradients must be [B,C,H,W]")
    if tuple(activations.shape) != tuple(gradients.shape):
        raise ValueError("Grad-CAM activations/gradients must have identical shapes")
    weights = gradients.float().mean(dim=(2, 3), keepdim=True)
    heatmaps = torch.relu((weights * activations.float()).sum(dim=1))
    flat = heatmaps.flatten(1)
    maxima = flat.max(dim=1).values.view(-1, 1, 1)
    return heatmaps / maxima.clamp(min=1e-9)


def upsample_heatmaps(heatmaps: Tensor, output_size: Tuple[int, int]) -> Tensor:
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape [B,H,W]")
    return F.interpolate(
        heatmaps.unsqueeze(1),
        size=(int(output_size[0]), int(output_size[1])),
        mode="bilinear",
        align_corners=False,
    )[:, 0].clamp(min=0.0)


def top_fraction_mask(
    heatmaps: Tensor,
    valid_mask: Tensor,
    *,
    fraction: float = MASK_FRACTION,
) -> Tensor:
    if heatmaps.ndim != 3 or valid_mask.ndim != 3:
        raise ValueError("heatmaps/valid_mask must have shape [B,H,W]")
    if tuple(heatmaps.shape) != tuple(valid_mask.shape):
        raise ValueError("heatmaps/valid_mask shapes differ")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be in (0,1)")
    result = torch.zeros_like(valid_mask, dtype=torch.bool)
    flat_heatmaps = heatmaps.flatten(1)
    flat_valid = valid_mask.to(dtype=torch.bool).flatten(1)
    flat_result = result.flatten(1)
    for row_index in range(int(heatmaps.size(0))):
        valid_indices = torch.nonzero(flat_valid[row_index], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        values = flat_heatmaps[row_index, valid_indices]
        if float(values.max().item()) <= 1e-9:
            continue
        count = max(1, int(math.ceil(float(valid_indices.numel()) * float(fraction))))
        chosen = values.topk(k=min(count, int(values.numel())), largest=True).indices
        flat_result[row_index, valid_indices[chosen]] = True
    return result


def normalized_reference_weights(probabilities: Tensor, references: Tensor) -> Tensor:
    weights = probabilities.gather(1, references).detach().float()
    return weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)


def weighted_reference_probability(
    probabilities: Tensor,
    references: Tensor,
    reference_weights: Tensor,
) -> Tensor:
    return (probabilities.gather(1, references).float() * reference_weights.float()).sum(dim=1)


def relative_confidence_drop(
    original_probabilities: Tensor,
    masked_probabilities: Tensor,
    references: Tensor,
    reference_weights: Tensor,
    *,
    target_class: int = FOCUS_CLASS,
) -> Tensor:
    original_reference = weighted_reference_probability(
        original_probabilities,
        references,
        reference_weights,
    )
    masked_reference = weighted_reference_probability(
        masked_probabilities,
        references,
        reference_weights,
    )
    target_drop = (
        original_probabilities[:, int(target_class)]
        - masked_probabilities[:, int(target_class)]
    )
    return target_drop - (original_reference - masked_reference)


def apply_fixed_contrast_gain(
    probabilities: Tensor,
    contrast_gain: Tensor,
    *,
    target_class: int = FOCUS_CLASS,
) -> Tensor:
    if probabilities.ndim != 2 or contrast_gain.ndim != 1:
        raise ValueError("probabilities/gain must have shape [B,C] and [B]")
    if int(probabilities.size(0)) != int(contrast_gain.size(0)):
        raise ValueError("probabilities/gain batch sizes differ")
    target_in_top2 = probabilities.topk(k=min(2, int(probabilities.size(1))), dim=1).indices.eq(
        int(target_class)
    ).any(dim=1)
    scores = probabilities.float().clone()
    adjusted = (scores[:, int(target_class)] + contrast_gain.float()).clamp(min=1e-9)
    scores[:, int(target_class)] = torch.where(
        target_in_top2,
        adjusted,
        scores[:, int(target_class)],
    )
    return scores / scores.sum(dim=1, keepdim=True).clamp(min=1e-9)


def transition_audit(
    targets: Tensor,
    global_probabilities: Tensor,
    candidate_probabilities: Tensor,
) -> Dict[str, int]:
    targets = targets.to(dtype=torch.long)
    global_predictions = global_probabilities.argmax(dim=1)
    candidate_predictions = candidate_probabilities.argmax(dim=1)
    global_correct = global_predictions.eq(targets)
    candidate_correct = candidate_predictions.eq(targets)
    changed = global_predictions.ne(candidate_predictions)
    focus = int(FOCUS_CLASS)
    return {
        "changed": int(changed.sum().item()),
        "corrections": int((changed & ~global_correct & candidate_correct).sum().item()),
        "harms": int((changed & global_correct & ~candidate_correct).sum().item()),
        "neutral": int((changed & ~global_correct & ~candidate_correct).sum().item()),
        "class1_fn_rescued": int(
            (targets.eq(focus) & global_predictions.ne(focus) & candidate_predictions.eq(focus))
            .sum()
            .item()
        ),
        "class1_tp_broken": int(
            (targets.eq(focus) & global_predictions.eq(focus) & candidate_predictions.ne(focus))
            .sum()
            .item()
        ),
        "class1_fp_removed": int(
            (targets.ne(focus) & global_predictions.eq(focus) & candidate_predictions.ne(focus))
            .sum()
            .item()
        ),
        "class1_fp_created": int(
            (targets.ne(focus) & global_predictions.ne(focus) & candidate_predictions.eq(focus))
            .sum()
            .item()
        ),
    }


def _focus_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    for row in metrics.get("per_class", []):
        if int(row.get("class_index", -1)) == FOCUS_CLASS:
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


def _direction_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    fn_rows = [
        row
        for row in rows
        if int(row["target_index"]) == FOCUS_CLASS
        and int(row["global_prediction_index"]) != FOCUS_CLASS
    ]
    fp_rows = [
        row
        for row in rows
        if int(row["target_index"]) != FOCUS_CLASS
        and int(row["global_prediction_index"]) == FOCUS_CLASS
    ]
    combined = fn_rows + fp_rows
    labels = [1] * len(fn_rows) + [0] * len(fp_rows)
    score_keys = (
        "standard_relative_drop",
        "finer_relative_drop",
        "finer_minus_standard_relative_drop",
        "finer_minus_standard_interior_mass",
    )
    aucs = {
        key: _safe_auc(labels, [float(row[key]) for row in combined])
        for key in score_keys
    }
    return {
        "fn_count": int(len(fn_rows)),
        "fp_count": int(len(fp_rows)),
        "fn_means": _mean_fields(fn_rows, score_keys),
        "fp_means": _mean_fields(fp_rows, score_keys),
        "fn_vs_fp_auroc": aucs,
    }


def _mean_fields(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> Dict[str, Optional[float]]:
    return {
        key: (float(np.mean([float(row[key]) for row in rows])) if rows else None)
        for key in keys
    }


def _group_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    fields = (
        "standard_relative_drop",
        "finer_relative_drop",
        "finer_minus_standard_relative_drop",
        "standard_bbox_mass",
        "finer_bbox_mass",
        "standard_interior_mass",
        "finer_interior_mass",
        "standard_border_mass",
        "finer_border_mass",
        "top5_jaccard",
    )
    groups = {
        "all": list(rows),
        "class1_fn": [
            row
            for row in rows
            if int(row["target_index"]) == FOCUS_CLASS
            and int(row["global_prediction_index"]) != FOCUS_CLASS
        ],
        "class1_fp": [
            row
            for row in rows
            if int(row["target_index"]) != FOCUS_CLASS
            and int(row["global_prediction_index"]) == FOCUS_CLASS
        ],
        "class1_tp": [
            row
            for row in rows
            if int(row["target_index"]) == FOCUS_CLASS
            and int(row["global_prediction_index"]) == FOCUS_CLASS
        ],
        "class1_tn": [
            row
            for row in rows
            if int(row["target_index"]) != FOCUS_CLASS
            and int(row["global_prediction_index"]) != FOCUS_CLASS
        ],
    }
    return {
        name: {"count": int(len(selected)), **_mean_fields(selected, fields)}
        for name, selected in groups.items()
    }


def _bbox_masks(
    crop_bbox: Tensor,
    valid_mask: Tensor,
    *,
    erode_ratio: float = INTERIOR_ERODE_RATIO,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, height, width = valid_mask.shape
    yy = (torch.arange(height, device=valid_mask.device, dtype=torch.float32) + 0.5) / float(height)
    xx = (torch.arange(width, device=valid_mask.device, dtype=torch.float32) + 0.5) / float(width)
    y_grid = yy.view(1, height, 1)
    x_grid = xx.view(1, 1, width)
    boxes = crop_bbox[:, :4].to(device=valid_mask.device, dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, box_width, box_height = boxes.unbind(dim=1)
    left = (cx - 0.5 * box_width).view(batch_size, 1, 1)
    right = (cx + 0.5 * box_width).view(batch_size, 1, 1)
    top = (cy - 0.5 * box_height).view(batch_size, 1, 1)
    bottom = (cy + 0.5 * box_height).view(batch_size, 1, 1)
    bbox_mask = (
        (x_grid >= left)
        & (x_grid <= right)
        & (y_grid >= top)
        & (y_grid <= bottom)
        & valid_mask
    )
    scale = 1.0 - 2.0 * float(erode_ratio)
    interior_left = (cx - 0.5 * box_width * scale).view(batch_size, 1, 1)
    interior_right = (cx + 0.5 * box_width * scale).view(batch_size, 1, 1)
    interior_top = (cy - 0.5 * box_height * scale).view(batch_size, 1, 1)
    interior_bottom = (cy + 0.5 * box_height * scale).view(batch_size, 1, 1)
    interior_mask = (
        (x_grid >= interior_left)
        & (x_grid <= interior_right)
        & (y_grid >= interior_top)
        & (y_grid <= interior_bottom)
        & valid_mask
    )
    border_size = max(1, int(round(min(height, width) * 0.08)))
    border = torch.zeros_like(valid_mask, dtype=torch.bool)
    border[:, :border_size, :] = True
    border[:, -border_size:, :] = True
    border[:, :, :border_size] = True
    border[:, :, -border_size:] = True
    return bbox_mask, interior_mask, border & valid_mask


def _heatmap_stats(
    heatmaps: Tensor,
    valid_mask: Tensor,
    bbox_mask: Tensor,
    interior_mask: Tensor,
    border_mask: Tensor,
) -> Dict[str, Tensor]:
    valid_heat = heatmaps.float() * valid_mask.float()
    totals = valid_heat.flatten(1).sum(dim=1)
    mass = valid_heat / totals.view(-1, 1, 1).clamp(min=1e-9)
    flat_mass = mass.flatten(1)
    entropy = -(
        flat_mass * torch.log(flat_mass.clamp(min=1e-12))
    ).sum(dim=1) / math.log(float(max(2, flat_mass.size(1))))
    peak = valid_heat.flatten(1).argmax(dim=1)
    peak_y = torch.div(peak, int(heatmaps.size(2)), rounding_mode="floor")
    peak_x = peak.remainder(int(heatmaps.size(2)))
    peak_inside_bbox = bbox_mask[
        torch.arange(int(heatmaps.size(0)), device=heatmaps.device), peak_y, peak_x
    ]
    return {
        "bbox_mass": (mass * bbox_mask.float()).flatten(1).sum(dim=1),
        "interior_mass": (mass * interior_mask.float()).flatten(1).sum(dim=1),
        "border_mass": (mass * border_mask.float()).flatten(1).sum(dim=1),
        "entropy": entropy,
        "peak_inside_bbox": peak_inside_bbox.float(),
        "nonzero": totals.gt(1e-9).float(),
    }


def _disable_inplace_modules(model: nn.Module) -> List[str]:
    changed: List[str] = []
    for name, module in model.named_modules():
        if getattr(module, "inplace", False) is not True:
            continue
        module.inplace = False
        changed.append(name or "<root>")
    return changed


def _masked_probabilities(
    *,
    model: nn.Module,
    images: Tensor,
    masks: Tensor,
    image_valid_mask: Tensor,
    bbox_metadata: Tensor,
    bbox_token_prior: Tensor,
    device: torch.device,
) -> Tensor:
    masked_images = images.masked_fill(masks.unsqueeze(1), 0.0)
    with torch.inference_mode():
        logits, _ = _forward_logits(
            model=model,
            images=masked_images,
            image_valid_mask=image_valid_mask,
            bbox_metadata=bbox_metadata,
            bbox_token_prior=bbox_token_prior,
            amp=False,
            device=device,
        )
    return torch.softmax(logits.float(), dim=1)


def collect_split(
    *,
    split: str,
    dataset: Dataset,
    model: nn.Module,
    feature_module: nn.Module,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    bbox_token_prior_source: str,
) -> Dict[str, object]:
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context=f"finer_cam_class1_{split}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate_classification,
        **loader_kwargs,
    )
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    if not callable(sample_paths_fn):
        raise TypeError("classification dataset must expose sample_paths()")
    dataset_paths = [str(value) for value in sample_paths_fn()]
    if len(dataset_paths) != len(dataset):
        raise ValueError("dataset sample_paths coverage is incomplete")
    capture: Dict[str, object] = {"enabled": False, "activation": None}

    def hook(_module, _inputs, output) -> None:
        if bool(capture["enabled"]) and torch.is_tensor(output):
            capture["activation"] = output

    handle = feature_module.register_forward_hook(hook)
    all_targets: List[Tensor] = []
    all_global: List[Tensor] = []
    all_candidate: List[Tensor] = []
    all_standard_cam: List[np.ndarray] = []
    all_finer_cam: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    row_offset = 0
    try:
        for images, targets, metadata in tqdm(loader, desc=f"finer-cam:{split}", unit="batch"):
            if not isinstance(metadata, Mapping):
                raise ValueError("classification metadata is required")
            required = ("bbox", "crop_bbox", "image_mask")
            if any(not torch.is_tensor(metadata.get(key)) for key in required):
                raise ValueError("bbox/crop_bbox/image_mask metadata is required")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            bbox = metadata["bbox"].to(device=device, dtype=torch.float32, non_blocking=True)
            crop_bbox = metadata["crop_bbox"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            image_valid_mask = metadata["image_mask"].to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )
            if image_valid_mask.ndim == 4 and int(image_valid_mask.size(1)) == 1:
                image_valid_mask = image_valid_mask[:, 0]
            bbox_token_prior = _select_bbox_token_prior(
                bbox=bbox,
                crop_bbox=crop_bbox,
                source=bbox_token_prior_source,
            )
            if not torch.is_tensor(bbox_token_prior):
                raise ValueError("bbox token prior is required")

            model.zero_grad(set_to_none=True)
            capture["activation"] = None
            capture["enabled"] = True
            logits, _ = _forward_logits(
                model=model,
                images=images,
                image_valid_mask=image_valid_mask,
                bbox_metadata=bbox,
                bbox_token_prior=bbox_token_prior,
                amp=False,
                device=device,
            )
            capture["enabled"] = False
            activations = capture.get("activation")
            if not torch.is_tensor(activations) or activations.ndim != 4:
                raise RuntimeError("selected Finer-CAM feature hook did not capture a 4D tensor")
            references = select_reference_categories(logits)
            standard_gradients = torch.autograd.grad(
                logits[:, FOCUS_CLASS].sum(),
                activations,
                retain_graph=True,
            )[0]
            finer_targets = finer_weighted_target(logits, references)
            finer_gradients = torch.autograd.grad(
                finer_targets.sum(),
                activations,
            )[0]
            standard_cam = batched_gradcam(activations.detach(), standard_gradients.detach())
            finer_cam = batched_gradcam(activations.detach(), finer_gradients.detach())
            probabilities = torch.softmax(logits.detach().float(), dim=1)
            reference_weights = normalized_reference_weights(probabilities, references)

            output_size = (int(images.size(2)), int(images.size(3)))
            standard_full = upsample_heatmaps(standard_cam, output_size)
            finer_full = upsample_heatmaps(finer_cam, output_size)
            standard_mask = top_fraction_mask(standard_full, image_valid_mask)
            finer_mask = top_fraction_mask(finer_full, image_valid_mask)
            standard_masked_probabilities = _masked_probabilities(
                model=model,
                images=images,
                masks=standard_mask,
                image_valid_mask=image_valid_mask,
                bbox_metadata=bbox,
                bbox_token_prior=bbox_token_prior,
                device=device,
            )
            finer_masked_probabilities = _masked_probabilities(
                model=model,
                images=images,
                masks=finer_mask,
                image_valid_mask=image_valid_mask,
                bbox_metadata=bbox,
                bbox_token_prior=bbox_token_prior,
                device=device,
            )
            standard_drop = relative_confidence_drop(
                probabilities,
                standard_masked_probabilities,
                references,
                reference_weights,
            )
            finer_drop = relative_confidence_drop(
                probabilities,
                finer_masked_probabilities,
                references,
                reference_weights,
            )
            contrast_gain = finer_drop - standard_drop
            candidate_probabilities = apply_fixed_contrast_gain(probabilities, contrast_gain)

            bbox_mask, interior_mask, border_mask = _bbox_masks(crop_bbox, image_valid_mask)
            standard_stats = _heatmap_stats(
                standard_full,
                image_valid_mask,
                bbox_mask,
                interior_mask,
                border_mask,
            )
            finer_stats = _heatmap_stats(
                finer_full,
                image_valid_mask,
                bbox_mask,
                interior_mask,
                border_mask,
            )
            intersection = (standard_mask & finer_mask).flatten(1).sum(dim=1).float()
            union = (standard_mask | finer_mask).flatten(1).sum(dim=1).float()
            top5_jaccard = intersection / union.clamp(min=1.0)
            global_predictions = probabilities.argmax(dim=1)
            candidate_predictions = candidate_probabilities.argmax(dim=1)
            class1_ranks = (
                probabilities.argsort(dim=1, descending=True).eq(FOCUS_CLASS).to(torch.int64).argmax(dim=1)
                + 1
            )
            original_reference = weighted_reference_probability(
                probabilities,
                references,
                reference_weights,
            )
            standard_masked_reference = weighted_reference_probability(
                standard_masked_probabilities,
                references,
                reference_weights,
            )
            finer_masked_reference = weighted_reference_probability(
                finer_masked_probabilities,
                references,
                reference_weights,
            )
            paths = dataset_paths[row_offset : row_offset + int(targets.numel())]
            if len(paths) != int(targets.numel()):
                raise ValueError("dataset path slice coverage is incomplete")
            for batch_index in range(int(targets.numel())):
                row: Dict[str, object] = {
                    "split": split,
                    "sample_index": int(row_offset + batch_index),
                    "image_path": paths[batch_index],
                    "source_stem": Path(paths[batch_index]).stem.casefold(),
                    "target_index": int(targets[batch_index].item()),
                    "global_prediction_index": int(global_predictions[batch_index].item()),
                    "candidate_prediction_index": int(candidate_predictions[batch_index].item()),
                    "class1_rank": int(class1_ranks[batch_index].item()),
                    "reference_indices": "|".join(
                        str(int(value)) for value in references[batch_index].tolist()
                    ),
                    "global_p1": float(probabilities[batch_index, FOCUS_CLASS].item()),
                    "global_reference_probability": float(original_reference[batch_index].item()),
                    "standard_masked_p1": float(
                        standard_masked_probabilities[batch_index, FOCUS_CLASS].item()
                    ),
                    "standard_masked_reference_probability": float(
                        standard_masked_reference[batch_index].item()
                    ),
                    "finer_masked_p1": float(
                        finer_masked_probabilities[batch_index, FOCUS_CLASS].item()
                    ),
                    "finer_masked_reference_probability": float(
                        finer_masked_reference[batch_index].item()
                    ),
                    "standard_relative_drop": float(standard_drop[batch_index].item()),
                    "finer_relative_drop": float(finer_drop[batch_index].item()),
                    "finer_minus_standard_relative_drop": float(
                        contrast_gain[batch_index].item()
                    ),
                    "standard_bbox_mass": float(standard_stats["bbox_mass"][batch_index].item()),
                    "finer_bbox_mass": float(finer_stats["bbox_mass"][batch_index].item()),
                    "standard_interior_mass": float(
                        standard_stats["interior_mass"][batch_index].item()
                    ),
                    "finer_interior_mass": float(finer_stats["interior_mass"][batch_index].item()),
                    "finer_minus_standard_interior_mass": float(
                        (
                            finer_stats["interior_mass"][batch_index]
                            - standard_stats["interior_mass"][batch_index]
                        ).item()
                    ),
                    "standard_border_mass": float(standard_stats["border_mass"][batch_index].item()),
                    "finer_border_mass": float(finer_stats["border_mass"][batch_index].item()),
                    "standard_entropy": float(standard_stats["entropy"][batch_index].item()),
                    "finer_entropy": float(finer_stats["entropy"][batch_index].item()),
                    "standard_peak_inside_bbox": int(
                        standard_stats["peak_inside_bbox"][batch_index].item()
                    ),
                    "finer_peak_inside_bbox": int(
                        finer_stats["peak_inside_bbox"][batch_index].item()
                    ),
                    "standard_cam_nonzero": int(standard_stats["nonzero"][batch_index].item()),
                    "finer_cam_nonzero": int(finer_stats["nonzero"][batch_index].item()),
                    "top5_jaccard": float(top5_jaccard[batch_index].item()),
                }
                for class_index in range(len(class_names)):
                    row[f"global_prob_{class_index}"] = float(
                        probabilities[batch_index, class_index].item()
                    )
                    row[f"candidate_prob_{class_index}"] = float(
                        candidate_probabilities[batch_index, class_index].item()
                    )
                rows.append(row)
            row_offset += int(targets.numel())
            all_targets.append(targets.detach().cpu())
            all_global.append(probabilities.detach().cpu())
            all_candidate.append(candidate_probabilities.detach().cpu())
            all_standard_cam.append(standard_cam.detach().cpu().numpy().astype(np.float16))
            all_finer_cam.append(finer_cam.detach().cpu().numpy().astype(np.float16))
    finally:
        handle.remove()

    targets = torch.cat(all_targets, dim=0)
    global_probabilities = torch.cat(all_global, dim=0)
    candidate_probabilities = torch.cat(all_candidate, dim=0)
    global_metrics = build_metrics(
        targets,
        global_probabilities.argmax(dim=1),
        class_names,
        probabilities=global_probabilities,
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
        "candidate_metrics": _compact_metrics(candidate_metrics),
        "transitions": transition_audit(targets, global_probabilities, candidate_probabilities),
        "direction": _direction_summary(rows),
        "groups": _group_summary(rows),
        "rows": rows,
        "targets": targets,
        "global_probabilities": global_probabilities,
        "candidate_probabilities": candidate_probabilities,
        "standard_cams": np.concatenate(all_standard_cam, axis=0),
        "finer_cams": np.concatenate(all_finer_cam, axis=0),
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
    train_auc = train_result["direction"]["fn_vs_fp_auroc"].get(
        "finer_minus_standard_relative_drop"
    )
    val_auc = val_result["direction"]["fn_vs_fp_auroc"].get(
        "finer_minus_standard_relative_drop"
    )
    train_all = train_result["groups"]["all"]
    val_all = val_result["groups"]["all"]
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
        "train_finer_relative_drop_gain": (
            float(train_all["finer_relative_drop"])
            > float(train_all["standard_relative_drop"])
        ),
        "val_finer_relative_drop_gain": (
            float(val_all["finer_relative_drop"])
            > float(val_all["standard_relative_drop"])
        ),
        "train_fn_fp_direction": train_auc is not None and float(train_auc) >= 0.60,
        "val_fn_fp_direction": val_auc is not None and float(val_auc) >= 0.60,
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
        "not_preflight": not bool(preflight),
    }
    return {
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": int(len(checks)),
        "smoke_permission": bool(all(checks.values())),
    }


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


def _tensor_to_image(tensor: Tensor, mean: Sequence[float], std: Sequence[float]) -> Image.Image:
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    rgb = (tensor.detach().cpu().float() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    array = rgb.mul(255.0).round().to(dtype=torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = torch.from_numpy(np.asarray(heatmap, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        heat,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0).numpy()
    color = plt.get_cmap("jet")(resized)[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    mixed = np.clip((1.0 - float(alpha)) * base + float(alpha) * color, 0.0, 1.0)
    return Image.fromarray(np.round(mixed * 255.0).astype(np.uint8))


def _masked_preview(
    image: Image.Image,
    heatmap: np.ndarray,
    valid_mask: Tensor,
    mean: Sequence[float],
) -> Image.Image:
    full = upsample_heatmaps(
        torch.from_numpy(np.asarray(heatmap, dtype=np.float32)).unsqueeze(0),
        (image.height, image.width),
    )
    mask = top_fraction_mask(full, valid_mask.unsqueeze(0).to(dtype=torch.bool))[0].numpy()
    array = np.asarray(image, dtype=np.uint8).copy()
    fill = np.round(np.asarray(mean, dtype=np.float32) * 255.0).astype(np.uint8)
    array[mask] = fill
    return Image.fromarray(array)


def _preview_indices(rows: Sequence[Mapping[str, object]], max_rows: int = 12) -> List[int]:
    predicates = (
        lambda row: int(row["target_index"]) == FOCUS_CLASS
        and int(row["global_prediction_index"]) != FOCUS_CLASS,
        lambda row: int(row["target_index"]) != FOCUS_CLASS
        and int(row["global_prediction_index"]) == FOCUS_CLASS,
        lambda row: int(row["target_index"]) == FOCUS_CLASS
        and int(row["global_prediction_index"]) == FOCUS_CLASS,
        lambda row: int(row["target_index"]) != FOCUS_CLASS
        and int(row["global_prediction_index"]) != FOCUS_CLASS
        and int(row["class1_rank"]) <= 2,
    )
    selected: List[int] = []
    per_group = max(1, int(max_rows) // len(predicates))
    for predicate in predicates:
        candidates = [row for row in rows if predicate(row)]
        candidates.sort(
            key=lambda row: abs(float(row["finer_minus_standard_relative_drop"])),
            reverse=True,
        )
        selected.extend(int(row["sample_index"]) for row in candidates[:per_group])
    return selected[: int(max_rows)]


def _render_contact_sheet(
    path: Path,
    *,
    dataset: Dataset,
    result: Mapping[str, object],
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    rows = result["rows"]
    indices = _preview_indices(rows)
    if not indices:
        return
    row_by_index = {int(row["sample_index"]): row for row in rows}
    standard_cams = result["standard_cams"]
    finer_cams = result["finer_cams"]
    thumb = 150
    header = 38
    columns = ("input", "Grad-CAM", "Finer-CAM", "Grad mask 5%", "Finer mask 5%")
    canvas = Image.new("RGB", (thumb * len(columns), (thumb + header) * len(indices)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_index in enumerate(indices):
        item = dataset[int(sample_index)]
        image_tensor, _, metadata = item
        image = _tensor_to_image(image_tensor, mean, std)
        valid_mask = metadata.get("image_mask")
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.ones(image.height, image.width, dtype=torch.bool)
        if valid_mask.ndim == 3 and int(valid_mask.size(0)) == 1:
            valid_mask = valid_mask[0]
        standard = np.asarray(standard_cams[int(sample_index)], dtype=np.float32)
        finer = np.asarray(finer_cams[int(sample_index)], dtype=np.float32)
        views = (
            image,
            _overlay(image, standard),
            _overlay(image, finer),
            _masked_preview(image, standard, valid_mask, mean),
            _masked_preview(image, finer, valid_mask, mean),
        )
        row = row_by_index[int(sample_index)]
        y = row_index * (thumb + header)
        title = (
            f"idx={sample_index} y={int(row['target_index'])} "
            f"g={int(row['global_prediction_index'])} c={int(row['candidate_prediction_index'])} "
            f"dRD={float(row['finer_minus_standard_relative_drop']):+.3f}"
        )
        draw.text((3, y + 2), title, fill="black")
        for column_index, (name, view) in enumerate(zip(columns, views)):
            rendered = view.copy()
            rendered.thumbnail((thumb - 4, thumb - 20), Image.Resampling.LANCZOS)
            x = column_index * thumb
            draw.text((x + 3, y + header), name, fill="black")
            image_y = y + header + 18
            canvas.paste(rendered, (x + (thumb - rendered.width) // 2, image_y))
    canvas.save(path)


def _plot_direction(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results):
        rows = result["rows"]
        fn = [
            float(row["finer_minus_standard_relative_drop"])
            for row in rows
            if int(row["target_index"]) == FOCUS_CLASS
            and int(row["global_prediction_index"]) != FOCUS_CLASS
        ]
        fp = [
            float(row["finer_minus_standard_relative_drop"])
            for row in rows
            if int(row["target_index"]) != FOCUS_CLASS
            and int(row["global_prediction_index"]) == FOCUS_CLASS
        ]
        axis.hist(fn, bins=30, alpha=0.65, label=f"class1 FN (n={len(fn)})")
        axis.hist(fp, bins=30, alpha=0.65, label=f"class1 FP (n={len(fp)})")
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(str(result["split"]))
        axis.set_xlabel("Finer RD - Grad-CAM RD")
        axis.set_ylabel("count")
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_metrics(path: Path, val_result: Mapping[str, object]) -> None:
    payloads = (val_result["global_metrics"], val_result["candidate_metrics"])
    macro = [float(payload["macro_f1"]) for payload in payloads]
    class1 = [float(payload["class1"]["f1"]) for payload in payloads]
    x = np.arange(2)
    figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    axis.bar(x - 0.18, macro, width=0.36, label="macro F1")
    axis.bar(x + 0.18, class1, width=0.36, label="class1 F1")
    axis.set_xticks(x, ("keeper", "fixed contrast gain"))
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


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
        "mode": "finer_cam_class1_readiness_evidence_manifest",
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
        "candidate_probabilities",
        "standard_cams",
        "finer_cams",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1 or int(args.num_workers) < 0:
        raise ValueError("invalid batch/worker settings")
    capped = int(args.max_train_samples) > 0 or int(args.max_val_samples) > 0
    if capped and not bool(args.allow_preflight):
        raise ValueError("sample caps require --allow-preflight")
    output_dir = Path(args.output_dir).resolve()
    data_path = Path(args.data).resolve()
    if _is_relative_to(output_dir, data_path.parent):
        raise ValueError("output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)
    set_seed(int(args.seed))

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        raise ValueError("Finer-CAM readiness is locked to bbox-aware yolo_f")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if class_names != list(data_spec.class_names):
        raise ValueError("checkpoint and dataset class orders differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(dict(checkpoint), num_classes=len(class_names))
    model.to(device)
    model.eval()
    disabled_inplace_modules = _disable_inplace_modules(model)
    feature_spec = resolve_feature_hook(model, feature_source=FEATURE_SOURCE)
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    transform = _build_eval_transform_from_checkpoint(dict(checkpoint), image_size=image_size)
    base_datasets: Dict[str, Dataset] = {}
    datasets: Dict[str, Dataset] = {}
    caps = {"train": int(args.max_train_samples), "val": int(args.max_val_samples)}
    for split in ("train", "val"):
        base = _build_classification_dataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            checkpoint=dict(checkpoint),
        )
        base_datasets[split] = base
        datasets[split] = PrefixDataset(base, caps[split])
    bbox_token_prior_source = str(
        checkpoint.get("model_config", {}).get("bbox_token_prior_source", "crop_bbox")
        or "crop_bbox"
    )
    train_result = collect_split(
        split="train",
        dataset=datasets["train"],
        model=model,
        feature_module=feature_spec.module,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        bbox_token_prior_source=bbox_token_prior_source,
    )
    val_result = collect_split(
        split="val",
        dataset=datasets["val"],
        model=model,
        feature_module=feature_spec.module,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
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
    metrics_rows: List[Dict[str, object]] = []
    for result in (train_result, val_result):
        for key, name in (("global_metrics", "keeper"), ("candidate_metrics", "fixed_contrast_gain")):
            metrics = result[key]
            metrics_rows.append(
                {
                    "split": result["split"],
                    "method": name,
                    "macro_f1": metrics["macro_f1"],
                    "class1_precision": metrics["class1"]["precision"],
                    "class1_recall": metrics["class1"]["recall"],
                    "class1_f1": metrics["class1"]["f1"],
                }
            )
    _write_csv(output_dir / "metrics_comparison.csv", metrics_rows)
    _plot_direction(output_dir / "class1_fn_fp_contrast_gain.png", (train_result, val_result))
    _plot_metrics(output_dir / "validation_metrics.png", val_result)
    mean, std = checkpoint_input_normalization(checkpoint)
    _render_contact_sheet(
        output_dir / "validation_finer_cam_contact_sheet.png",
        dataset=datasets["val"],
        result=val_result,
        mean=mean,
        std=std,
    )

    summary = {
        "mode": "finer_cam_class1_readiness",
        "protocol_locked": True,
        "preflight": bool(capped),
        "test_split_used": False,
        "raw_dataset_modified": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "data": str(data_path),
        "device": str(device),
        "image_size": int(image_size),
        "class_names": class_names,
        "protocol": {
            "paper": PAPER_URL,
            "official_repository": OFFICIAL_REPO_URL,
            "feature_source_requested": FEATURE_SOURCE,
            "feature_source_resolved": feature_spec.source,
            "target_class": FOCUS_CLASS,
            "reference_selection": (
                "three non-target classes with smallest absolute logit distance to class1"
            ),
            "reference_count": REFERENCE_COUNT,
            "finer_objective": (
                "sum_i p_i * (logit_class1 - alpha * logit_ref_i) / sum_i p_i"
            ),
            "alpha": FINER_ALPHA,
            "mask_fraction": MASK_FRACTION,
            "mask_fill": "zero in normalized input space (checkpoint mean RGB)",
            "relative_drop": "(p1-p1_masked) - (weighted_pref-pref_masked)",
            "fixed_candidate": (
                "when class1 is global top2, add Finer-RD minus Grad-CAM-RD to p1, "
                "clamp positive, renormalize; otherwise preserve keeper"
            ),
            "model_fit": False,
            "threshold_layer_or_weight_sweep": False,
            "ground_truth_used_for_attribution": False,
            "train_probability_caveat": (
                "keeper train outputs are in-sample; train direction is transfer support, not OOF"
            ),
            "disabled_inplace_modules_for_gradient_hooks": disabled_inplace_modules,
        },
        "source_overlap": {"count": int(len(source_overlap)), "examples": source_overlap[:10]},
        "artifact_manifest_path": "artifact_manifest.json",
        "train": _public_result(train_result),
        "val": _public_result(val_result),
        "gate": gate,
    }
    json_dump(output_dir / "summary.json", summary)
    val_global = summary["val"]["global_metrics"]
    val_candidate = summary["val"]["candidate_metrics"]
    readme = [
        "# Finer-CAM Class-1 Readiness Audit",
        "",
        "- No test access, raw-data edit, model fit, threshold/layer sweep, or checkpoint output.",
        f"- Feature source: `{feature_spec.source}`; class1 target; three logit-nearest references.",
        f"- Keeper val macro/class1: `{val_global['macro_f1']:.6f}/{val_global['class1']['f1']:.6f}`.",
        f"- Fixed candidate val macro/class1: `{val_candidate['macro_f1']:.6f}/{val_candidate['class1']['f1']:.6f}`.",
        f"- Candidate class1 P/R: `{val_candidate['class1']['precision']:.6f}/{val_candidate['class1']['recall']:.6f}`.",
        f"- Val transitions: `{summary['val']['transitions']}`.",
        f"- Train/val FN-vs-FP contrast-gain AUROC: `{summary['train']['direction']['fn_vs_fp_auroc']['finer_minus_standard_relative_drop']}` / `{summary['val']['direction']['fn_vs_fp_auroc']['finer_minus_standard_relative_drop']}`.",
        f"- Smoke permission: `{gate['smoke_permission']}` ({gate['passed']}/{gate['total']} checks).",
        "",
        "The train keeper outputs are in-sample and are not described as current-keeper OOF evidence.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    _write_artifact_manifest(output_dir)
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()

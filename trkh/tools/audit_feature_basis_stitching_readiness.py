from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.audit_pdisco_semantic_part_readiness import (
    _balanced_subset_indices,
    _build_split_dataset,
    classification_metrics,
    dataset_sample_paths,
    transition_counts,
)
from trkh.tools.probe_embedding_prototypes import _collate_classification


SEED = 20260714
FOCUS_CLASS = 1
FORKS = (5, 6, 7)
RIDGE_RATIO = 1e-4
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
EXPECTED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
EXPECTED_SCRATCH_SHA256 = (
    "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
)

OOF_THRESHOLDS = {
    "patch_intersection_ratio": 0.85,
    "token_r2": 0.50,
    "prefix_r2": 0.35,
    "patch_r2": 0.50,
    "cnn_r2": 0.50,
    "std_ratio_min": 0.80,
    "std_ratio_max": 1.20,
    "probability_mae": 0.04,
    "argmax_agreement": 0.75,
}

VAL_THRESHOLDS = {
    "probability_mae": 0.03,
    "argmax_agreement": 0.85,
    "macro_f1": 0.887,
    "focus_precision": 0.65,
    "focus_recall": 0.70,
    "focus_f1": 0.69,
    "focus_false_positives": 60,
    "focus_tp_break": 10,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@dataclass
class AffineMap:
    weight: Tensor
    bias: Tensor
    ridge: float

    def apply(self, values: Tensor) -> Tensor:
        return values @ self.weight.to(device=values.device, dtype=values.dtype) + self.bias.to(
            device=values.device,
            dtype=values.dtype,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "weight": self.weight.detach().cpu().float(),
            "bias": self.bias.detach().cpu().float(),
            "ridge": float(self.ridge),
        }


class LinearPairStats:
    """Streaming sufficient statistics for centered affine direct matching."""

    def __init__(self, input_dim: int, output_dim: Optional[int] = None) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim if output_dim is not None else input_dim)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("pair-stat dimensions must be positive")
        self.count = 0
        self.sum_x = torch.zeros(self.input_dim, dtype=torch.float64)
        self.sum_y = torch.zeros(self.output_dim, dtype=torch.float64)
        self.xtx = torch.zeros((self.input_dim, self.input_dim), dtype=torch.float64)
        self.xty = torch.zeros((self.input_dim, self.output_dim), dtype=torch.float64)
        self.yty = torch.zeros((self.output_dim, self.output_dim), dtype=torch.float64)

    def update(self, x: Tensor, y: Tensor) -> None:
        if x.ndim != 2 or y.ndim != 2 or int(x.size(0)) != int(y.size(0)):
            raise ValueError("direct-match pairs must be aligned matrices")
        if int(x.size(1)) != self.input_dim or int(y.size(1)) != self.output_dim:
            raise ValueError("direct-match pair dimensions differ from accumulator")
        if int(x.size(0)) == 0:
            return
        x32 = x.detach().float()
        y32 = y.detach().float()
        if not bool(torch.isfinite(x32).all().item()) or not bool(torch.isfinite(y32).all().item()):
            raise ValueError("direct-match pairs contain nonfinite values")
        self.count += int(x32.size(0))
        self.sum_x += x32.sum(dim=0, dtype=torch.float64).cpu()
        self.sum_y += y32.sum(dim=0, dtype=torch.float64).cpu()
        # Matmul stays on the source device so the large token fit uses the GPU.
        self.xtx += (x32.transpose(0, 1) @ x32).double().cpu()
        self.xty += (x32.transpose(0, 1) @ y32).double().cpu()
        self.yty += (y32.transpose(0, 1) @ y32).double().cpu()

    def add_(self, other: "LinearPairStats", scale: float = 1.0) -> "LinearPairStats":
        if (self.input_dim, self.output_dim) != (other.input_dim, other.output_dim):
            raise ValueError("cannot combine pair stats with different dimensions")
        if scale not in (-1.0, 1.0):
            raise ValueError("pair-stat combination scale must be -1 or 1")
        self.count += int(scale * other.count)
        self.sum_x += other.sum_x * scale
        self.sum_y += other.sum_y * scale
        self.xtx += other.xtx * scale
        self.xty += other.xty * scale
        self.yty += other.yty * scale
        if self.count < 0:
            raise ValueError("pair-stat subtraction produced a negative count")
        return self

    def clone(self) -> "LinearPairStats":
        output = LinearPairStats(self.input_dim, self.output_dim)
        output.count = int(self.count)
        output.sum_x = self.sum_x.clone()
        output.sum_y = self.sum_y.clone()
        output.xtx = self.xtx.clone()
        output.xty = self.xty.clone()
        output.yty = self.yty.clone()
        return output

    def centered(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.count <= 1:
            raise ValueError("at least two pairs are required")
        divisor = float(self.count)
        cxx = self.xtx - torch.outer(self.sum_x, self.sum_x) / divisor
        cxy = self.xty - torch.outer(self.sum_x, self.sum_y) / divisor
        cyy = self.yty - torch.outer(self.sum_y, self.sum_y) / divisor
        return cxx, cxy, cyy

    def solve(self, ridge_ratio: float = RIDGE_RATIO) -> AffineMap:
        if self.count <= 1:
            raise ValueError("at least two direct-match pairs are required")
        cxx, cxy, _ = self.centered()
        scale = float(torch.trace(cxx).item()) / float(max(1, self.input_dim))
        ridge = max(float(ridge_ratio) * max(scale, 1e-12), 1e-12)
        regularized = cxx + torch.eye(self.input_dim, dtype=torch.float64) * ridge
        weight = torch.linalg.solve(regularized, cxy)
        mean_x = self.sum_x / float(self.count)
        mean_y = self.sum_y / float(self.count)
        bias = mean_y - mean_x @ weight
        if not bool(torch.isfinite(weight).all().item()) or not bool(
            torch.isfinite(bias).all().item()
        ):
            raise RuntimeError("direct affine solve produced nonfinite parameters")
        return AffineMap(weight=weight.float(), bias=bias.float(), ridge=ridge)

    def cka(self) -> float:
        cxx, cxy, cyy = self.centered()
        numerator = float(cxy.square().sum().item())
        denominator = math.sqrt(
            max(float(cxx.square().sum().item()), 0.0)
            * max(float(cyy.square().sum().item()), 0.0)
        )
        return float(numerator / denominator) if denominator > 0.0 else 0.0

    def identity_r2(self) -> Optional[float]:
        if self.input_dim != self.output_dim or self.count <= 1:
            return None
        _, _, cyy = self.centered()
        target_sst = float(torch.trace(cyy).item())
        sse = float(torch.trace(self.xtx + self.yty - self.xty - self.xty.T).item())
        return float(1.0 - sse / target_sst) if target_sst > 0.0 else None

    def compact_summary(self) -> Dict[str, object]:
        cxx, _, cyy = self.centered()
        x_std = math.sqrt(max(float(torch.trace(cxx).item()), 0.0) / float(self.count * self.input_dim))
        y_std = math.sqrt(
            max(float(torch.trace(cyy).item()), 0.0) / float(self.count * self.output_dim)
        )
        return {
            "pairs": int(self.count),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "linear_cka": self.cka(),
            "identity_r2": self.identity_r2(),
            "input_std": float(x_std),
            "target_std": float(y_std),
        }


class DirectMatchMetrics:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.sum_prediction = torch.zeros(self.dim, dtype=torch.float64)
        self.sum_target = torch.zeros(self.dim, dtype=torch.float64)
        self.prediction_square_sum = 0.0
        self.target_square_sum = 0.0
        self.sse = 0.0
        self.cosine_sum = 0.0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("mapped predictions and targets must be aligned matrices")
        if int(prediction.size(1)) != self.dim:
            raise ValueError("mapped metric dimension mismatch")
        if int(prediction.size(0)) == 0:
            return
        pred = prediction.detach().float()
        truth = target.detach().float()
        if not bool(torch.isfinite(pred).all().item()) or not bool(
            torch.isfinite(truth).all().item()
        ):
            raise ValueError("mapped metrics received nonfinite values")
        self.count += int(pred.size(0))
        self.sum_prediction += pred.sum(dim=0, dtype=torch.float64).cpu()
        self.sum_target += truth.sum(dim=0, dtype=torch.float64).cpu()
        self.prediction_square_sum += float(pred.square().sum().item())
        self.target_square_sum += float(truth.square().sum().item())
        self.sse += float((pred - truth).square().sum().item())
        self.cosine_sum += float(
            torch.nn.functional.cosine_similarity(pred, truth, dim=1, eps=1e-8).sum().item()
        )

    def summary(self) -> Dict[str, object]:
        if self.count <= 1:
            raise ValueError("mapped direct metrics require at least two rows")
        target_center = float(self.sum_target.square().sum().item()) / float(self.count)
        prediction_center = float(self.sum_prediction.square().sum().item()) / float(self.count)
        target_sst = max(self.target_square_sum - target_center, 0.0)
        prediction_sst = max(self.prediction_square_sum - prediction_center, 0.0)
        target_std = math.sqrt(target_sst / float(self.count * self.dim))
        prediction_std = math.sqrt(prediction_sst / float(self.count * self.dim))
        return {
            "pairs": int(self.count),
            "r2": float(1.0 - self.sse / target_sst) if target_sst > 0.0 else 0.0,
            "rmse": math.sqrt(self.sse / float(self.count * self.dim)),
            "mean_cosine": float(self.cosine_sum / float(self.count)),
            "prediction_std": float(prediction_std),
            "target_std": float(target_std),
            "std_ratio": float(prediction_std / target_std) if target_std > 0.0 else 0.0,
        }


def aligned_token_pairs(
    keeper_tokens: Tensor,
    keeper_patch_indices: Tensor,
    scratch_tokens: Tensor,
    scratch_patch_indices: Tensor,
    *,
    prefix_count: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if keeper_tokens.ndim != 2 or scratch_tokens.ndim != 2:
        raise ValueError("aligned token matching expects one sample at a time")
    if int(keeper_tokens.size(1)) != int(scratch_tokens.size(1)):
        raise ValueError("keeper and scratch embedding dimensions differ")
    if keeper_patch_indices.ndim != 1 or scratch_patch_indices.ndim != 1:
        raise ValueError("patch indices must be vectors")
    if int(keeper_tokens.size(0)) != int(prefix_count + keeper_patch_indices.numel()):
        raise ValueError("keeper token/index geometry mismatch")
    if int(scratch_tokens.size(0)) != int(prefix_count + scratch_patch_indices.numel()):
        raise ValueError("scratch token/index geometry mismatch")
    if torch.unique(keeper_patch_indices).numel() != keeper_patch_indices.numel():
        raise ValueError("keeper patch indices contain duplicates")
    if torch.unique(scratch_patch_indices).numel() != scratch_patch_indices.numel():
        raise ValueError("scratch patch indices contain duplicates")

    prefix_x = keeper_tokens[:prefix_count]
    prefix_y = scratch_tokens[:prefix_count]
    sorted_scratch, scratch_order = torch.sort(scratch_patch_indices)
    positions = torch.searchsorted(sorted_scratch, keeper_patch_indices)
    safe_positions = positions.clamp(max=max(0, int(sorted_scratch.numel()) - 1))
    valid = positions < int(sorted_scratch.numel())
    if bool(valid.any().item()):
        valid = valid & (sorted_scratch[safe_positions] == keeper_patch_indices)
    keeper_patch_tokens = keeper_tokens[prefix_count:]
    scratch_patch_tokens = scratch_tokens[prefix_count:]
    patch_x = keeper_patch_tokens[valid]
    patch_y = scratch_patch_tokens[scratch_order[safe_positions[valid]]]
    return prefix_x, prefix_y, patch_x, patch_y


def frozen_precision_rule(
    keeper_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> np.ndarray:
    keeper = np.asarray(keeper_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if keeper.shape != candidate.shape or keeper.ndim != 2:
        raise ValueError("precision-rule probability matrices must align")
    if not np.isfinite(keeper).all() or not np.isfinite(candidate).all():
        raise ValueError("precision-rule probabilities must be finite")
    probabilities = keeper * 0.60 + candidate * 0.40
    probabilities[:, FOCUS_CLASS] = np.maximum(
        probabilities[:, FOCUS_CLASS] - 0.034,
        1e-8,
    )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if bool((row_sums <= 0.0).any()):
        raise ValueError("precision-rule normalization is invalid")
    return probabilities / row_sums


def choose_oof_fork(
    summaries: Mapping[int, Mapping[str, object]],
    *,
    tie_tolerance: float = 1e-4,
) -> Optional[int]:
    eligible = [
        int(fork)
        for fork, summary in summaries.items()
        if bool(summary.get("eligible", False))
    ]
    if not eligible:
        return None
    best_mae = min(float(summaries[fork]["functional"]["probability_mae"]) for fork in eligible)
    tied = [
        fork
        for fork in eligible
        if float(summaries[fork]["functional"]["probability_mae"])
        <= best_mae + float(tie_tolerance)
    ]
    return max(tied)


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent = _dataset_labels(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, Sequence) or len(samples) != len(dataset):
        raise ValueError("dataset samples are unavailable for source-grouped labels")
    labels = [int(getattr(sample, "primary_label")) for sample in samples]
    return np.asarray(labels, dtype=np.int64)


def _dataset_paths_and_groups(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    paths = np.asarray(dataset_sample_paths(dataset), dtype=str)
    if int(paths.size) != int(len(dataset)) or any(
        not str(path).strip() for path in paths.tolist()
    ):
        raise ValueError("dataset paths are missing or misaligned")
    groups = np.asarray([Path(path).stem.casefold() for path in paths.tolist()], dtype=str)
    return paths, groups


def assign_source_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    source = np.asarray(groups).astype(str).reshape(-1)
    if target.size != source.size:
        raise ValueError("labels and source groups do not align")
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    assignments = np.full(target.size, -1, dtype=np.int64)
    rows: List[Dict[str, object]] = []
    total_overlap = 0
    for fold_index, (fit_rows, holdout_rows) in enumerate(
        splitter.split(np.zeros(target.size), target, source)
    ):
        fit_sources = set(source[fit_rows].tolist())
        holdout_sources = set(source[holdout_rows].tolist())
        overlap = fit_sources.intersection(holdout_sources)
        total_overlap += len(overlap)
        assignments[holdout_rows] = int(fold_index)
        rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(fit_rows.size),
                "holdout_rows": int(holdout_rows.size),
                "fit_sources": int(len(fit_sources)),
                "holdout_sources": int(len(holdout_sources)),
                "source_overlap": int(len(overlap)),
                "holdout_class_counts": np.bincount(
                    target[holdout_rows],
                    minlength=int(target.max(initial=0) + 1),
                ).tolist(),
            }
        )
    if bool((assignments < 0).any()):
        raise RuntimeError("source-grouped OOF assignment is incomplete")
    return assignments, {
        "fold_count": int(folds),
        "source_overlap": int(total_overlap),
        "assignment_complete": True,
        "assignment_counts": np.bincount(assignments, minlength=int(folds)).tolist(),
        "folds": rows,
    }


class ForkInputCapture:
    def __init__(self, model: nn.Module, forks: Sequence[int]) -> None:
        self.model = model
        self.forks = tuple(int(fork) for fork in forks)
        self.values: Dict[int, Tuple[Tensor, Tensor]] = {}
        self.handles = []
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise ValueError("model does not expose a transformer block list")
        for fork in self.forks:
            if not 1 <= int(fork) < len(blocks):
                raise ValueError(f"invalid fork after block {fork}")

            def hook(module, args, kwargs, *, fork_number: int = fork):
                del module
                tokens = args[0] if args else None
                patch_indices = kwargs.get("patch_indices")
                if not torch.is_tensor(tokens) or not torch.is_tensor(patch_indices):
                    raise RuntimeError("fork hook did not receive tokens and patch indices")
                self.values[fork_number] = (tokens, patch_indices)

            self.handles.append(
                blocks[fork].register_forward_pre_hook(hook, with_kwargs=True)
            )

    def clear(self) -> None:
        self.values.clear()

    def consume(self) -> Dict[int, Tuple[Tensor, Tensor]]:
        missing = sorted(set(self.forks) - set(self.values))
        if missing:
            raise RuntimeError(f"fork hooks were not reached: {missing}")
        return dict(self.values)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _forward_with_capture(
    model: nn.Module,
    capture: ForkInputCapture,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> Tuple[Dict[str, Tensor], Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
    crop_bbox = metadata.get("crop_bbox")
    image_mask = metadata.get("image_mask")
    bbox = metadata.get("bbox")
    if not torch.is_tensor(crop_bbox) or not torch.is_tensor(image_mask) or not torch.is_tensor(bbox):
        raise ValueError("bbox, crop_bbox, and image_mask metadata are required")
    crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
    image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    bbox = bbox.to(device=device, dtype=torch.float32, non_blocking=True)
    capture.clear()
    features = model.forward_features(
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=crop_bbox,
    )
    features["bbox"] = bbox[:, :4]
    logits = classification_logits_from_features(model, features)
    return features, logits, capture.consume()


def _receiver_logits_from_fork(
    receiver: nn.Module,
    *,
    fork: int,
    tokens: Tensor,
    patch_indices: Tensor,
    base_features: Mapping[str, Tensor],
    cnn_pooled: Tensor,
) -> Tensor:
    output_tokens = tokens
    grid_size = base_features.get("grid_size")
    if not isinstance(grid_size, Sequence) or len(grid_size) != 2:
        raise ValueError("receiver replay requires grid_size")
    for block_index in range(int(fork), len(receiver.blocks)):
        output_tokens = receiver.blocks[block_index](
            output_tokens,
            grid_size=tuple(int(value) for value in grid_size),
            prefix_count=int(receiver.num_prefix_tokens),
            patch_indices=patch_indices,
        )
    output_tokens = receiver.norm(output_tokens)
    register_end = 1 + int(receiver.num_registers)
    branch_end = register_end + int(receiver.num_branch_tokens)
    patches = output_tokens[:, branch_end:]
    if int(patches.size(1)) != int(patch_indices.size(1)):
        raise ValueError("receiver replay token/index geometry mismatch")

    features: Dict[str, Tensor] = {
        "tokens": output_tokens,
        "cls": output_tokens[:, 0],
        "registers": output_tokens[:, 1:register_end],
        "branch_tokens": output_tokens[:, register_end:branch_end],
        "patches": patches,
        "patch_indices": patch_indices,
        "pooled": receiver.pool_tokens_for_head(
            output_tokens[:, 0],
            output_tokens[:, 1:register_end],
            output_tokens[:, register_end:branch_end],
        ),
        "grid_size": tuple(int(value) for value in grid_size),
        "cnn_pooled": cnn_pooled,
    }
    for key in ("bbox", "memory_key_padding_mask", "patch_bbox_prior", "patch_foreground_prior"):
        value = base_features.get(key)
        if torch.is_tensor(value):
            features[key] = value
    return classification_logits_from_features(receiver, features)


def apply_fold_maps(
    values: Tensor,
    fold_ids: np.ndarray,
    maps: Mapping[int, AffineMap],
) -> Tensor:
    folds = np.asarray(fold_ids, dtype=np.int64).reshape(-1)
    if int(values.size(0)) != int(folds.size):
        raise ValueError("fold assignments do not align with the batch")
    output = torch.empty(
        (int(values.size(0)), *tuple(int(value) for value in values.shape[1:-1]), maps[next(iter(maps))].bias.numel()),
        device=values.device,
        dtype=values.dtype,
    )
    for fold in np.unique(folds).tolist():
        if int(fold) not in maps:
            raise KeyError(f"missing affine map for fold {fold}")
        row_mask = torch.as_tensor(folds == int(fold), device=values.device, dtype=torch.bool)
        output[row_mask] = maps[int(fold)].apply(values[row_mask])
    return output


def _sum_stats(stats: Iterable[LinearPairStats]) -> LinearPairStats:
    values = list(stats)
    if not values:
        raise ValueError("cannot sum an empty pair-stat sequence")
    total = LinearPairStats(values[0].input_dim, values[0].output_dim)
    for value in values:
        total.add_(value)
    return total


def _model_compatibility(
    keeper: nn.Module,
    scratch: nn.Module,
    keeper_checkpoint: Mapping[str, object],
    scratch_checkpoint: Mapping[str, object],
) -> Dict[str, object]:
    keeper_names = list(keeper_checkpoint.get("class_names", []))
    scratch_names = list(scratch_checkpoint.get("class_names", []))
    checks = {
        "class_names_equal": keeper_names == scratch_names and len(keeper_names) == 5,
        "model_type_equal": type(keeper) is type(scratch),
        "depth_equal": len(keeper.blocks) == len(scratch.blocks) == 8,
        "embed_dim_equal": int(keeper.embed_dim) == int(scratch.embed_dim),
        "prefix_count_equal": int(keeper.num_prefix_tokens) == int(scratch.num_prefix_tokens),
        "register_count_equal": int(keeper.num_registers) == int(scratch.num_registers),
        "branch_count_equal": int(keeper.num_branch_tokens) == int(scratch.num_branch_tokens),
        "pruning_schedule_equal": dict(keeper.token_prune_schedule)
        == dict(scratch.token_prune_schedule),
        "pruning_finishes_by_block5": max(
            (int(index) + 1 for index in keeper.token_prune_schedule),
            default=0,
        )
        <= 5,
        "cnn_features_enabled": bool(keeper.cnn_feature_fusion)
        and bool(scratch.cnn_feature_fusion),
        "late_member_disabled": not bool(getattr(keeper, "late_member_enabled", False))
        and not bool(getattr(scratch, "late_member_enabled", False)),
    }
    if not all(checks.values()):
        raise ValueError(f"keeper/scratch models are not stitch-compatible: {checks}")
    return {
        "checks": checks,
        "embed_dim": int(keeper.embed_dim),
        "prefix_count": int(keeper.num_prefix_tokens),
        "depth": int(len(keeper.blocks)),
        "token_prune_schedule": {
            str(int(index) + 1): float(rate)
            for index, rate in keeper.token_prune_schedule.items()
        },
        "class_names": keeper_names,
    }


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        persistent_workers=bool(int(workers) > 0),
        collate_fn=_collate_classification,
    )


def _collect_fit_stats(
    *,
    keeper: nn.Module,
    scratch: nn.Module,
    dataset: Dataset,
    fold_assignments: np.ndarray,
    folds: int,
    forks: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[
    Dict[int, Dict[int, Dict[str, LinearPairStats]]],
    Dict[int, LinearPairStats],
    Dict[int, Dict[str, int]],
    float,
]:
    token_dim = int(keeper.embed_dim)
    token_stats = {
        int(fork): {
            int(fold): {
                role: LinearPairStats(token_dim)
                for role in ("all", "prefix", "patch")
            }
            for fold in range(int(folds))
        }
        for fork in forks
    }
    cnn_stats = {int(fold): LinearPairStats(token_dim) for fold in range(int(folds))}
    intersections = {
        int(fork): {
            "matched": 0,
            "keeper_total": 0,
            "scratch_total": 0,
            "rows": 0,
        }
        for fork in forks
    }
    loader = _build_loader(dataset, batch_size=batch_size, workers=workers)
    keeper_capture = ForkInputCapture(keeper, forks)
    scratch_capture = ForkInputCapture(scratch, forks)
    row_offset = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc="stitch-fit-stats", dynamic_ncols=True)
            for images, labels, metadata in iterator:
                del labels
                if not isinstance(metadata, Mapping):
                    raise ValueError("classification metadata is required")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                batch_rows = int(images.size(0))
                batch_folds = np.asarray(
                    fold_assignments[row_offset : row_offset + batch_rows],
                    dtype=np.int64,
                )
                keeper_features, _, keeper_forks = _forward_with_capture(
                    keeper,
                    keeper_capture,
                    images,
                    metadata,
                    device=device,
                )
                scratch_features, _, scratch_forks = _forward_with_capture(
                    scratch,
                    scratch_capture,
                    images,
                    metadata,
                    device=device,
                )
                keeper_cnn = keeper_features.get("cnn_pooled")
                scratch_cnn = scratch_features.get("cnn_pooled")
                if not torch.is_tensor(keeper_cnn) or not torch.is_tensor(scratch_cnn):
                    raise ValueError("both models must expose CNN pooled vectors")
                for fold in np.unique(batch_folds).tolist():
                    mask = torch.as_tensor(
                        batch_folds == int(fold),
                        device=device,
                        dtype=torch.bool,
                    )
                    cnn_stats[int(fold)].update(keeper_cnn[mask], scratch_cnn[mask])

                prefix_count = int(keeper.num_prefix_tokens)
                for fork in forks:
                    keeper_tokens, keeper_indices = keeper_forks[int(fork)]
                    scratch_tokens, scratch_indices = scratch_forks[int(fork)]
                    for fold in np.unique(batch_folds).tolist():
                        prefix_x_rows: List[Tensor] = []
                        prefix_y_rows: List[Tensor] = []
                        patch_x_rows: List[Tensor] = []
                        patch_y_rows: List[Tensor] = []
                        local_rows = np.flatnonzero(batch_folds == int(fold)).tolist()
                        for row in local_rows:
                            prefix_x, prefix_y, patch_x, patch_y = aligned_token_pairs(
                                keeper_tokens[row],
                                keeper_indices[row],
                                scratch_tokens[row],
                                scratch_indices[row],
                                prefix_count=prefix_count,
                            )
                            prefix_x_rows.append(prefix_x)
                            prefix_y_rows.append(prefix_y)
                            patch_x_rows.append(patch_x)
                            patch_y_rows.append(patch_y)
                            intersections[int(fork)]["matched"] += int(patch_x.size(0))
                            intersections[int(fork)]["keeper_total"] += int(
                                keeper_indices[row].numel()
                            )
                            intersections[int(fork)]["scratch_total"] += int(
                                scratch_indices[row].numel()
                            )
                            intersections[int(fork)]["rows"] += 1
                        prefix_x_cat = torch.cat(prefix_x_rows, dim=0)
                        prefix_y_cat = torch.cat(prefix_y_rows, dim=0)
                        patch_x_cat = torch.cat(patch_x_rows, dim=0)
                        patch_y_cat = torch.cat(patch_y_rows, dim=0)
                        role_stats = token_stats[int(fork)][int(fold)]
                        role_stats["prefix"].update(prefix_x_cat, prefix_y_cat)
                        role_stats["patch"].update(patch_x_cat, patch_y_cat)
                        role_stats["all"].update(
                            torch.cat((prefix_x_cat, patch_x_cat), dim=0),
                            torch.cat((prefix_y_cat, patch_y_cat), dim=0),
                        )
                row_offset += batch_rows
    finally:
        keeper_capture.close()
        scratch_capture.close()
    if row_offset != len(dataset):
        raise RuntimeError(f"fit-stat row mismatch: {row_offset}/{len(dataset)}")
    return token_stats, cnn_stats, intersections, float(time.perf_counter() - started)


def _solve_maps(
    token_stats: Mapping[int, Mapping[int, Mapping[str, LinearPairStats]]],
    cnn_stats: Mapping[int, LinearPairStats],
    *,
    folds: int,
) -> Tuple[
    Dict[int, Dict[int, AffineMap]],
    Dict[int, AffineMap],
    Dict[int, AffineMap],
    AffineMap,
    Dict[int, Dict[str, object]],
]:
    oof_token_maps: Dict[int, Dict[int, AffineMap]] = {}
    full_token_maps: Dict[int, AffineMap] = {}
    raw_summaries: Dict[int, Dict[str, object]] = {}
    for fork, fold_stats in token_stats.items():
        global_roles = {
            role: _sum_stats(fold_stats[fold][role] for fold in range(int(folds)))
            for role in ("all", "prefix", "patch")
        }
        full_token_maps[int(fork)] = global_roles["all"].solve()
        oof_token_maps[int(fork)] = {}
        for holdout_fold in range(int(folds)):
            fit_stats = global_roles["all"].clone().add_(
                fold_stats[holdout_fold]["all"],
                scale=-1.0,
            )
            oof_token_maps[int(fork)][holdout_fold] = fit_stats.solve()
        raw_summaries[int(fork)] = {
            role: global_roles[role].compact_summary()
            for role in ("all", "prefix", "patch")
        }

    global_cnn = _sum_stats(cnn_stats.values())
    full_cnn_map = global_cnn.solve()
    oof_cnn_maps: Dict[int, AffineMap] = {}
    for holdout_fold in range(int(folds)):
        fit_stats = global_cnn.clone().add_(cnn_stats[holdout_fold], scale=-1.0)
        oof_cnn_maps[holdout_fold] = fit_stats.solve()
    for summary in raw_summaries.values():
        summary["cnn"] = global_cnn.compact_summary()
    return oof_token_maps, oof_cnn_maps, full_token_maps, full_cnn_map, raw_summaries


def _probability_comparison(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, object]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("probability comparison matrices do not align")
    return {
        "probability_mae": float(np.abs(left - right).mean()),
        "probability_rmse": float(np.sqrt(np.square(left - right).mean())),
        "argmax_agreement": float((left.argmax(axis=1) == right.argmax(axis=1)).mean()),
    }


def _focus_row(metrics: Mapping[str, object]) -> Mapping[str, object]:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Sequence) or len(per_class) <= FOCUS_CLASS:
        raise ValueError("classification metrics do not contain focus class")
    row = per_class[FOCUS_CLASS]
    if not isinstance(row, Mapping):
        raise ValueError("focus-class metric row is invalid")
    return row


def _write_prediction_csv(
    path: Path,
    *,
    labels: np.ndarray,
    paths: np.ndarray,
    fold_ids: Optional[np.ndarray],
    probabilities: Mapping[str, np.ndarray],
) -> None:
    names = list(probabilities)
    if not names:
        raise ValueError("prediction CSV requires at least one probability matrix")
    rows = int(np.asarray(labels).size)
    class_count = int(np.asarray(probabilities[names[0]]).shape[1])
    fieldnames = ["row_index", "image_path", "source_stem", "target_index"]
    if fold_ids is not None:
        fieldnames.append("oof_fold")
    for name in names:
        fieldnames.append(f"{name}_prediction")
        fieldnames.extend(f"{name}_prob_{index}" for index in range(class_count))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(rows):
            row: Dict[str, object] = {
                "row_index": int(row_index),
                "image_path": str(paths[row_index]),
                "source_stem": Path(str(paths[row_index])).stem.casefold(),
                "target_index": int(labels[row_index]),
            }
            if fold_ids is not None:
                row["oof_fold"] = int(fold_ids[row_index])
            for name in names:
                values = np.asarray(probabilities[name])[row_index]
                row[f"{name}_prediction"] = int(values.argmax())
                for class_index, value in enumerate(values.tolist()):
                    row[f"{name}_prob_{class_index}"] = float(value)
            writer.writerow(row)


def _run_oof_evaluation(
    *,
    keeper: nn.Module,
    scratch: nn.Module,
    dataset: Dataset,
    labels: np.ndarray,
    paths: np.ndarray,
    fold_assignments: np.ndarray,
    forks: Sequence[int],
    oof_token_maps: Mapping[int, Mapping[int, AffineMap]],
    oof_cnn_maps: Mapping[int, AffineMap],
    raw_summaries: Mapping[int, Mapping[str, object]],
    intersections: Mapping[int, Mapping[str, int]],
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    workers: int,
    output_dir: Path,
) -> Tuple[Dict[int, Dict[str, object]], float]:
    rows = int(len(dataset))
    class_count = len(class_names)
    keeper_probabilities = np.zeros((rows, class_count), dtype=np.float32)
    scratch_probabilities = np.zeros((rows, class_count), dtype=np.float32)
    fork_probabilities = {
        int(fork): {
            name: np.zeros((rows, class_count), dtype=np.float32)
            for name in ("identity", "oracle", "mapped", "fused")
        }
        for fork in forks
    }
    direct_metrics = {
        int(fork): {
            role: DirectMatchMetrics(int(keeper.embed_dim))
            for role in ("all", "prefix", "patch", "cnn")
        }
        for fork in forks
    }
    oracle_logit_max_error = {int(fork): 0.0 for fork in forks}
    loader = _build_loader(dataset, batch_size=batch_size, workers=workers)
    keeper_capture = ForkInputCapture(keeper, forks)
    scratch_capture = ForkInputCapture(scratch, forks)
    row_offset = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc="stitch-oof-eval", dynamic_ncols=True)
            for images, _, metadata in iterator:
                if not isinstance(metadata, Mapping):
                    raise ValueError("classification metadata is required")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                batch_rows = int(images.size(0))
                row_slice = slice(row_offset, row_offset + batch_rows)
                batch_folds = np.asarray(fold_assignments[row_slice], dtype=np.int64)
                keeper_features, keeper_logits, keeper_forks = _forward_with_capture(
                    keeper,
                    keeper_capture,
                    images,
                    metadata,
                    device=device,
                )
                scratch_features, scratch_logits, scratch_forks = _forward_with_capture(
                    scratch,
                    scratch_capture,
                    images,
                    metadata,
                    device=device,
                )
                keeper_probabilities[row_slice] = keeper_logits.softmax(dim=1).cpu().numpy()
                scratch_probabilities[row_slice] = scratch_logits.softmax(dim=1).cpu().numpy()
                keeper_cnn = keeper_features.get("cnn_pooled")
                scratch_cnn = scratch_features.get("cnn_pooled")
                if not torch.is_tensor(keeper_cnn) or not torch.is_tensor(scratch_cnn):
                    raise ValueError("CNN pooled vectors are missing")
                mapped_cnn = apply_fold_maps(keeper_cnn, batch_folds, oof_cnn_maps)

                for fork in forks:
                    token_maps = oof_token_maps[int(fork)]
                    keeper_tokens, keeper_indices = keeper_forks[int(fork)]
                    scratch_tokens, scratch_indices = scratch_forks[int(fork)]
                    mapped_tokens = apply_fold_maps(keeper_tokens, batch_folds, token_maps)
                    identity_logits = _receiver_logits_from_fork(
                        scratch,
                        fork=int(fork),
                        tokens=keeper_tokens,
                        patch_indices=keeper_indices,
                        base_features=keeper_features,
                        cnn_pooled=keeper_cnn,
                    )
                    oracle_logits = _receiver_logits_from_fork(
                        scratch,
                        fork=int(fork),
                        tokens=scratch_tokens,
                        patch_indices=scratch_indices,
                        base_features=scratch_features,
                        cnn_pooled=scratch_cnn,
                    )
                    mapped_logits = _receiver_logits_from_fork(
                        scratch,
                        fork=int(fork),
                        tokens=mapped_tokens,
                        patch_indices=keeper_indices,
                        base_features=keeper_features,
                        cnn_pooled=mapped_cnn,
                    )
                    oracle_logit_max_error[int(fork)] = max(
                        oracle_logit_max_error[int(fork)],
                        float((oracle_logits - scratch_logits).abs().max().item()),
                    )
                    identity_prob = identity_logits.softmax(dim=1).cpu().numpy()
                    oracle_prob = oracle_logits.softmax(dim=1).cpu().numpy()
                    mapped_prob = mapped_logits.softmax(dim=1).cpu().numpy()
                    fused_prob = frozen_precision_rule(
                        keeper_logits.softmax(dim=1).cpu().numpy(),
                        mapped_prob,
                    )
                    fork_probabilities[int(fork)]["identity"][row_slice] = identity_prob
                    fork_probabilities[int(fork)]["oracle"][row_slice] = oracle_prob
                    fork_probabilities[int(fork)]["mapped"][row_slice] = mapped_prob
                    fork_probabilities[int(fork)]["fused"][row_slice] = fused_prob

                    prefix_count = int(keeper.num_prefix_tokens)
                    for local_row, fold in enumerate(batch_folds.tolist()):
                        prefix_x, prefix_y, patch_x, patch_y = aligned_token_pairs(
                            keeper_tokens[local_row],
                            keeper_indices[local_row],
                            scratch_tokens[local_row],
                            scratch_indices[local_row],
                            prefix_count=prefix_count,
                        )
                        affine = token_maps[int(fold)]
                        mapped_prefix = affine.apply(prefix_x)
                        mapped_patch = affine.apply(patch_x)
                        direct_metrics[int(fork)]["prefix"].update(mapped_prefix, prefix_y)
                        direct_metrics[int(fork)]["patch"].update(mapped_patch, patch_y)
                        direct_metrics[int(fork)]["all"].update(
                            torch.cat((mapped_prefix, mapped_patch), dim=0),
                            torch.cat((prefix_y, patch_y), dim=0),
                        )
                    direct_metrics[int(fork)]["cnn"].update(mapped_cnn, scratch_cnn)
                row_offset += batch_rows
    finally:
        keeper_capture.close()
        scratch_capture.close()
    if row_offset != rows:
        raise RuntimeError(f"OOF evaluation row mismatch: {row_offset}/{rows}")

    keeper_metrics = classification_metrics(labels, keeper_probabilities, class_names=class_names)
    scratch_metrics = classification_metrics(labels, scratch_probabilities, class_names=class_names)
    summaries: Dict[int, Dict[str, object]] = {}
    for fork in forks:
        probabilities = fork_probabilities[int(fork)]
        functional = _probability_comparison(probabilities["mapped"], scratch_probabilities)
        role_metrics = {
            role: direct_metrics[int(fork)][role].summary()
            for role in ("all", "prefix", "patch", "cnn")
        }
        intersection = intersections[int(fork)]
        keeper_ratio = float(intersection["matched"] / max(1, intersection["keeper_total"]))
        scratch_ratio = float(intersection["matched"] / max(1, intersection["scratch_total"]))
        std_values = [role_metrics["all"]["std_ratio"], role_metrics["cnn"]["std_ratio"]]
        gates = {
            "patch_intersection_keeper": keeper_ratio
            >= OOF_THRESHOLDS["patch_intersection_ratio"],
            "patch_intersection_scratch": scratch_ratio
            >= OOF_THRESHOLDS["patch_intersection_ratio"],
            "token_r2": float(role_metrics["all"]["r2"]) >= OOF_THRESHOLDS["token_r2"],
            "prefix_r2": float(role_metrics["prefix"]["r2"])
            >= OOF_THRESHOLDS["prefix_r2"],
            "patch_r2": float(role_metrics["patch"]["r2"]) >= OOF_THRESHOLDS["patch_r2"],
            "cnn_r2": float(role_metrics["cnn"]["r2"]) >= OOF_THRESHOLDS["cnn_r2"],
            "activation_std": all(
                OOF_THRESHOLDS["std_ratio_min"]
                <= float(value)
                <= OOF_THRESHOLDS["std_ratio_max"]
                for value in std_values
            ),
            "probability_mae": float(functional["probability_mae"])
            <= OOF_THRESHOLDS["probability_mae"],
            "argmax_agreement": float(functional["argmax_agreement"])
            >= OOF_THRESHOLDS["argmax_agreement"],
            "oracle_replay_exact": float(oracle_logit_max_error[int(fork)]) <= 1e-6,
        }
        summaries[int(fork)] = {
            "fork_after_block": int(fork),
            "raw_representation": dict(raw_summaries[int(fork)]),
            "direct_match": role_metrics,
            "patch_intersection": {
                **{key: int(value) for key, value in intersection.items()},
                "keeper_ratio": keeper_ratio,
                "scratch_ratio": scratch_ratio,
            },
            "functional": functional,
            "oracle_logit_max_error": float(oracle_logit_max_error[int(fork)]),
            "metrics": {
                "keeper": keeper_metrics,
                "scratch": scratch_metrics,
                "identity": classification_metrics(
                    labels, probabilities["identity"], class_names=class_names
                ),
                "mapped": classification_metrics(
                    labels, probabilities["mapped"], class_names=class_names
                ),
                "fused": classification_metrics(
                    labels, probabilities["fused"], class_names=class_names
                ),
            },
            "fused_transitions_vs_keeper": transition_counts(
                labels,
                keeper_probabilities,
                probabilities["fused"],
            ),
            "gates": gates,
            "eligible": bool(all(gates.values())),
        }
        _write_prediction_csv(
            output_dir / f"oof_predictions_fork{int(fork)}.csv",
            labels=labels,
            paths=paths,
            fold_ids=fold_assignments,
            probabilities={
                "keeper": keeper_probabilities,
                "scratch": scratch_probabilities,
                **probabilities,
            },
        )
    return summaries, float(time.perf_counter() - started)


def _run_validation(
    *,
    keeper: nn.Module,
    scratch: nn.Module,
    dataset: Dataset,
    labels: np.ndarray,
    paths: np.ndarray,
    fork: int,
    token_map: AffineMap,
    cnn_map: AffineMap,
    raw_summary: Mapping[str, object],
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    workers: int,
    output_dir: Path,
) -> Tuple[Dict[str, object], float]:
    rows = int(len(dataset))
    class_count = len(class_names)
    probabilities = {
        name: np.zeros((rows, class_count), dtype=np.float32)
        for name in ("keeper", "scratch", "identity", "oracle", "mapped", "fused")
    }
    metrics = {
        role: DirectMatchMetrics(int(keeper.embed_dim))
        for role in ("all", "prefix", "patch", "cnn")
    }
    raw_stats = {
        role: LinearPairStats(int(keeper.embed_dim))
        for role in ("all", "prefix", "patch", "cnn")
    }
    intersection = {"matched": 0, "keeper_total": 0, "scratch_total": 0, "rows": 0}
    oracle_max_error = 0.0
    loader = _build_loader(dataset, batch_size=batch_size, workers=workers)
    keeper_capture = ForkInputCapture(keeper, (fork,))
    scratch_capture = ForkInputCapture(scratch, (fork,))
    row_offset = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc=f"stitch-val-fork{fork}", dynamic_ncols=True)
            for images, _, metadata in iterator:
                if not isinstance(metadata, Mapping):
                    raise ValueError("classification metadata is required")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                batch_rows = int(images.size(0))
                row_slice = slice(row_offset, row_offset + batch_rows)
                keeper_features, keeper_logits, keeper_forks = _forward_with_capture(
                    keeper, keeper_capture, images, metadata, device=device
                )
                scratch_features, scratch_logits, scratch_forks = _forward_with_capture(
                    scratch, scratch_capture, images, metadata, device=device
                )
                keeper_tokens, keeper_indices = keeper_forks[int(fork)]
                scratch_tokens, scratch_indices = scratch_forks[int(fork)]
                keeper_cnn = keeper_features.get("cnn_pooled")
                scratch_cnn = scratch_features.get("cnn_pooled")
                if not torch.is_tensor(keeper_cnn) or not torch.is_tensor(scratch_cnn):
                    raise ValueError("CNN pooled vectors are missing")
                mapped_tokens = token_map.apply(keeper_tokens)
                mapped_cnn = cnn_map.apply(keeper_cnn)
                identity_logits = _receiver_logits_from_fork(
                    scratch,
                    fork=fork,
                    tokens=keeper_tokens,
                    patch_indices=keeper_indices,
                    base_features=keeper_features,
                    cnn_pooled=keeper_cnn,
                )
                oracle_logits = _receiver_logits_from_fork(
                    scratch,
                    fork=fork,
                    tokens=scratch_tokens,
                    patch_indices=scratch_indices,
                    base_features=scratch_features,
                    cnn_pooled=scratch_cnn,
                )
                mapped_logits = _receiver_logits_from_fork(
                    scratch,
                    fork=fork,
                    tokens=mapped_tokens,
                    patch_indices=keeper_indices,
                    base_features=keeper_features,
                    cnn_pooled=mapped_cnn,
                )
                oracle_max_error = max(
                    oracle_max_error,
                    float((oracle_logits - scratch_logits).abs().max().item()),
                )
                probabilities["keeper"][row_slice] = keeper_logits.softmax(dim=1).cpu().numpy()
                probabilities["scratch"][row_slice] = scratch_logits.softmax(dim=1).cpu().numpy()
                probabilities["identity"][row_slice] = identity_logits.softmax(dim=1).cpu().numpy()
                probabilities["oracle"][row_slice] = oracle_logits.softmax(dim=1).cpu().numpy()
                probabilities["mapped"][row_slice] = mapped_logits.softmax(dim=1).cpu().numpy()
                probabilities["fused"][row_slice] = frozen_precision_rule(
                    probabilities["keeper"][row_slice],
                    probabilities["mapped"][row_slice],
                )
                prefix_count = int(keeper.num_prefix_tokens)
                for local_row in range(batch_rows):
                    prefix_x, prefix_y, patch_x, patch_y = aligned_token_pairs(
                        keeper_tokens[local_row],
                        keeper_indices[local_row],
                        scratch_tokens[local_row],
                        scratch_indices[local_row],
                        prefix_count=prefix_count,
                    )
                    mapped_prefix = token_map.apply(prefix_x)
                    mapped_patch = token_map.apply(patch_x)
                    metrics["prefix"].update(mapped_prefix, prefix_y)
                    metrics["patch"].update(mapped_patch, patch_y)
                    metrics["all"].update(
                        torch.cat((mapped_prefix, mapped_patch), dim=0),
                        torch.cat((prefix_y, patch_y), dim=0),
                    )
                    raw_stats["prefix"].update(prefix_x, prefix_y)
                    raw_stats["patch"].update(patch_x, patch_y)
                    raw_stats["all"].update(
                        torch.cat((prefix_x, patch_x), dim=0),
                        torch.cat((prefix_y, patch_y), dim=0),
                    )
                    intersection["matched"] += int(patch_x.size(0))
                    intersection["keeper_total"] += int(keeper_indices[local_row].numel())
                    intersection["scratch_total"] += int(scratch_indices[local_row].numel())
                    intersection["rows"] += 1
                metrics["cnn"].update(mapped_cnn, scratch_cnn)
                raw_stats["cnn"].update(keeper_cnn, scratch_cnn)
                row_offset += batch_rows
    finally:
        keeper_capture.close()
        scratch_capture.close()
    if row_offset != rows:
        raise RuntimeError(f"validation row mismatch: {row_offset}/{rows}")

    mapped_comparison = _probability_comparison(probabilities["mapped"], probabilities["scratch"])
    fused_metrics = classification_metrics(labels, probabilities["fused"], class_names=class_names)
    focus = _focus_row(fused_metrics)
    transitions = transition_counts(labels, probabilities["keeper"], probabilities["fused"])
    confusion = np.asarray(fused_metrics["confusion_matrix"], dtype=np.int64)
    false_positives = int(confusion[:, FOCUS_CLASS].sum() - confusion[FOCUS_CLASS, FOCUS_CLASS])
    direct = {role: metrics[role].summary() for role in metrics}
    gates = {
        "full_support": rows == EXPECTED_VAL_ROWS,
        "probability_mae": float(mapped_comparison["probability_mae"])
        <= VAL_THRESHOLDS["probability_mae"],
        "argmax_agreement": float(mapped_comparison["argmax_agreement"])
        >= VAL_THRESHOLDS["argmax_agreement"],
        "macro_f1": float(fused_metrics["macro_f1"]) >= VAL_THRESHOLDS["macro_f1"],
        "focus_precision": float(focus["precision"]) >= VAL_THRESHOLDS["focus_precision"],
        "focus_recall": float(focus["recall"]) >= VAL_THRESHOLDS["focus_recall"],
        "focus_f1": float(focus["f1"]) >= VAL_THRESHOLDS["focus_f1"],
        "focus_false_positives": false_positives
        <= VAL_THRESHOLDS["focus_false_positives"],
        "corrections_not_fewer_than_harms": int(transitions["corrections"])
        >= int(transitions["harms"]),
        "focus_tp_break": int(transitions["focus_tp_break"])
        <= VAL_THRESHOLDS["focus_tp_break"],
        "token_r2": float(direct["all"]["r2"]) >= OOF_THRESHOLDS["token_r2"],
        "prefix_r2": float(direct["prefix"]["r2"]) >= OOF_THRESHOLDS["prefix_r2"],
        "patch_r2": float(direct["patch"]["r2"]) >= OOF_THRESHOLDS["patch_r2"],
        "cnn_r2": float(direct["cnn"]["r2"]) >= OOF_THRESHOLDS["cnn_r2"],
        "activation_std": all(
            OOF_THRESHOLDS["std_ratio_min"]
            <= float(direct[role]["std_ratio"])
            <= OOF_THRESHOLDS["std_ratio_max"]
            for role in ("all", "cnn")
        ),
        "oracle_replay_exact": oracle_max_error <= 1e-6,
    }
    summary = {
        "fork_after_block": int(fork),
        "rows": rows,
        "raw_representation": {
            role: raw_stats[role].compact_summary() for role in raw_stats
        },
        "train_raw_representation": dict(raw_summary),
        "direct_match": direct,
        "patch_intersection": {
            **intersection,
            "keeper_ratio": float(intersection["matched"] / max(1, intersection["keeper_total"])),
            "scratch_ratio": float(intersection["matched"] / max(1, intersection["scratch_total"])),
        },
        "functional": mapped_comparison,
        "oracle_logit_max_error": oracle_max_error,
        "metrics": {
            name: classification_metrics(labels, values, class_names=class_names)
            for name, values in probabilities.items()
        },
        "fused_transitions_vs_keeper": transitions,
        "focus_false_positives": false_positives,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }
    _write_prediction_csv(
        output_dir / "validation_predictions.csv",
        labels=labels,
        paths=paths,
        fold_ids=None,
        probabilities=probabilities,
    )
    return summary, float(time.perf_counter() - started)


def _adapter_state(
    *,
    keeper_sha256: str,
    scratch_sha256: str,
    fold_assignments: np.ndarray,
    oof_token_maps: Mapping[int, Mapping[int, AffineMap]],
    oof_cnn_maps: Mapping[int, AffineMap],
    full_token_maps: Mapping[int, AffineMap],
    full_cnn_map: AffineMap,
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "direct_feature_basis_stitching",
        "keeper_checkpoint_sha256": keeper_sha256,
        "scratch_checkpoint_sha256": scratch_sha256,
        "ridge_ratio": RIDGE_RATIO,
        "fold_assignments": torch.as_tensor(fold_assignments, dtype=torch.int64),
        "oof_token_maps": {
            str(fork): {str(fold): value.state_dict() for fold, value in maps.items()}
            for fork, maps in oof_token_maps.items()
        },
        "oof_cnn_maps": {
            str(fold): value.state_dict() for fold, value in oof_cnn_maps.items()
        },
        "full_token_maps": {
            str(fork): value.state_dict() for fork, value in full_token_maps.items()
        },
        "full_cnn_map": full_cnn_map.state_dict(),
        "test_data_used": False,
        "raw_data_modified": False,
    }


def _write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Feature-Basis Stitching Readiness",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Train rows: `{summary['train_rows']}`",
        f"- Selected fork: `{summary.get('selected_fork')}`",
        f"- Test data used: `{summary['test_data_used']}`",
        f"- Raw data modified: `{summary['raw_data_modified']}`",
        "",
        "## Train OOF Forks",
        "",
        "| Fork | Token R2 | Prefix R2 | Patch R2 | CNN R2 | Scratch MAE | Agreement | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    oof = summary.get("oof", {})
    if isinstance(oof, Mapping):
        for key in sorted(oof, key=lambda value: int(value)):
            row = oof[key]
            if not isinstance(row, Mapping):
                continue
            direct = row["direct_match"]
            functional = row["functional"]
            lines.append(
                "| {fork} | {token:.4f} | {prefix:.4f} | {patch:.4f} | {cnn:.4f} | "
                "{mae:.5f} | {agreement:.4f} | {eligible} |".format(
                    fork=key,
                    token=float(direct["all"]["r2"]),
                    prefix=float(direct["prefix"]["r2"]),
                    patch=float(direct["patch"]["r2"]),
                    cnn=float(direct["cnn"]["r2"]),
                    mae=float(functional["probability_mae"]),
                    agreement=float(functional["argmax_agreement"]),
                    eligible=bool(row["eligible"]),
                )
            )
    validation = summary.get("validation")
    if isinstance(validation, Mapping):
        fused = validation["metrics"]["fused"]
        focus = _focus_row(fused)
        lines.extend(
            [
                "",
                "## Validation",
                "",
                f"- Macro F1: `{float(fused['macro_f1']):.6f}`",
                f"- Class-1 P/R/F1: `{float(focus['precision']):.6f}/"
                f"{float(focus['recall']):.6f}/{float(focus['f1']):.6f}`",
                f"- Focus false positives: `{validation['focus_false_positives']}`",
                f"- Passed: `{validation['passed']}`",
            ]
        )
    lines.extend(
        [
            "",
            "No labels entered the affine fit. Test rows and raw-data writes are forbidden.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked train/validation-only direct feature-basis stitching "
            "readiness audit for the TRKH keeper and scratch complement."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    keeper_path = Path(args.keeper).expanduser().resolve()
    scratch_path = Path(args.scratch).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    for path in (keeper_path, scratch_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    keeper_sha256 = sha256_file(keeper_path)
    scratch_sha256 = sha256_file(scratch_path)
    if keeper_sha256 != EXPECTED_KEEPER_SHA256:
        raise ValueError(
            f"keeper hash differs from locked protocol: {keeper_sha256}"
        )
    if scratch_sha256 != EXPECTED_SCRATCH_SHA256:
        raise ValueError(
            f"scratch hash differs from locked protocol: {scratch_sha256}"
        )
    output_dir.mkdir(parents=True)
    _seed_everything(SEED)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = _resolve_device(args.device)
    started = time.perf_counter()

    keeper_checkpoint = torch.load(keeper_path, map_location="cpu", weights_only=False)
    scratch_checkpoint = torch.load(scratch_path, map_location="cpu", weights_only=False)
    if not isinstance(keeper_checkpoint, Mapping) or not isinstance(scratch_checkpoint, Mapping):
        raise ValueError("checkpoint payloads must be mappings")
    keeper = build_model_from_checkpoint(dict(keeper_checkpoint)).to(
        device=device, dtype=torch.float32
    )
    scratch = build_model_from_checkpoint(dict(scratch_checkpoint)).to(
        device=device, dtype=torch.float32
    )
    keeper.eval()
    scratch.eval()
    compatibility = _model_compatibility(
        keeper,
        scratch,
        keeper_checkpoint,
        scratch_checkpoint,
    )
    class_names = list(compatibility["class_names"])

    train_dataset, train_names = _build_split_dataset(
        data_yaml=data_path,
        split="train",
        checkpoint=keeper_checkpoint,
        preflight=False,
    )
    if list(train_names) != class_names:
        raise ValueError("train dataset class order differs from checkpoint")
    if bool(args.preflight):
        train_dataset = Subset(train_dataset, _balanced_subset_indices(train_dataset, 128))
    train_labels = _dataset_labels(train_dataset)
    train_paths, train_groups = _dataset_paths_and_groups(train_dataset)
    if not bool(args.preflight) and len(train_dataset) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"full train support differs from protocol: {len(train_dataset)}/{EXPECTED_TRAIN_ROWS}"
        )
    fold_count = 2 if bool(args.preflight) else 5
    fold_assignments, fold_summary = assign_source_folds(
        train_labels,
        train_groups,
        folds=fold_count,
        seed=SEED,
    )

    token_stats, cnn_stats, intersections, fit_seconds = _collect_fit_stats(
        keeper=keeper,
        scratch=scratch,
        dataset=train_dataset,
        fold_assignments=fold_assignments,
        folds=fold_count,
        forks=FORKS,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    (
        oof_token_maps,
        oof_cnn_maps,
        full_token_maps,
        full_cnn_map,
        raw_summaries,
    ) = _solve_maps(token_stats, cnn_stats, folds=fold_count)
    adapter_path = output_dir / "adapter_state.pt"
    torch.save(
        _adapter_state(
            keeper_sha256=keeper_sha256,
            scratch_sha256=scratch_sha256,
            fold_assignments=fold_assignments,
            oof_token_maps=oof_token_maps,
            oof_cnn_maps=oof_cnn_maps,
            full_token_maps=full_token_maps,
            full_cnn_map=full_cnn_map,
        ),
        adapter_path,
    )

    oof_summaries, oof_seconds = _run_oof_evaluation(
        keeper=keeper,
        scratch=scratch,
        dataset=train_dataset,
        labels=train_labels,
        paths=train_paths,
        fold_assignments=fold_assignments,
        forks=FORKS,
        oof_token_maps=oof_token_maps,
        oof_cnn_maps=oof_cnn_maps,
        raw_summaries=raw_summaries,
        intersections=intersections,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        output_dir=output_dir,
    )
    selected_fork = None if bool(args.preflight) else choose_oof_fork(oof_summaries)
    validation_summary: Optional[Dict[str, object]] = None
    validation_seconds = 0.0
    train_val_source_overlap: Optional[int] = None
    if selected_fork is not None:
        val_dataset, val_names = _build_split_dataset(
            data_yaml=data_path,
            split="val",
            checkpoint=keeper_checkpoint,
            preflight=False,
        )
        if list(val_names) != class_names:
            raise ValueError("validation dataset class order differs from checkpoint")
        val_labels = _dataset_labels(val_dataset)
        val_paths, val_groups = _dataset_paths_and_groups(val_dataset)
        train_val_source_overlap = len(set(train_groups.tolist()).intersection(val_groups.tolist()))
        if train_val_source_overlap != 0:
            raise ValueError("train and validation source groups overlap")
        validation_summary, validation_seconds = _run_validation(
            keeper=keeper,
            scratch=scratch,
            dataset=val_dataset,
            labels=val_labels,
            paths=val_paths,
            fork=int(selected_fork),
            token_map=full_token_maps[int(selected_fork)],
            cnn_map=full_cnn_map,
            raw_summary=raw_summaries[int(selected_fork)],
            class_names=class_names,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            output_dir=output_dir,
        )

    summary: Dict[str, object] = {
        "schema_version": 1,
        "mode": "preflight" if bool(args.preflight) else "full",
        "protocol": "direct_feature_basis_stitching_readiness_20260714",
        "data_yaml": str(data_path),
        "keeper_checkpoint": str(keeper_path),
        "keeper_checkpoint_sha256": keeper_sha256,
        "scratch_checkpoint": str(scratch_path),
        "scratch_checkpoint_sha256": scratch_sha256,
        "compatibility": compatibility,
        "seed": SEED,
        "forks": list(FORKS),
        "ridge_ratio": RIDGE_RATIO,
        "train_rows": int(len(train_dataset)),
        "train_source_groups": int(np.unique(train_groups).size),
        "folds": fold_summary,
        "oof_thresholds": OOF_THRESHOLDS,
        "validation_thresholds": VAL_THRESHOLDS,
        "oof": {str(fork): value for fork, value in oof_summaries.items()},
        "selected_fork": selected_fork,
        "validation": validation_summary,
        "train_val_source_overlap": train_val_source_overlap,
        "adapter_state": str(adapter_path),
        "adapter_state_sha256": sha256_file(adapter_path),
        "timing_seconds": {
            "fit_stats": fit_seconds,
            "oof_evaluation": oof_seconds,
            "validation": validation_seconds,
            "total": float(time.perf_counter() - started),
        },
        "test_data_used": False,
        "raw_data_modified": False,
        "current_best_command_updated": False,
        "decision": (
            "runtime_preflight_only"
            if bool(args.preflight)
            else (
                "adapter_checkpoint_parity_allowed"
                if validation_summary is not None and bool(validation_summary["passed"])
                else (
                    "linear_feature_basis_stitching_rejected_before_validation"
                    if selected_fork is None
                    else "linear_feature_basis_stitching_rejected_on_validation"
                )
            )
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    artifact_manifest = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "files": artifact_manifest,
                "test_data_used": False,
                "raw_data_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run(_parse_args(argv))
    print(
        {
            "mode": summary["mode"],
            "train_rows": summary["train_rows"],
            "selected_fork": summary["selected_fork"],
            "decision": summary["decision"],
            "output_dir": str(Path(summary["adapter_state"]).parent),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _direction_auc,
    _load_cache,
    _transition_stats,
    _write_csv,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
)


SEED = 20260712
FOLDS = 5
ROI_SIZE = 128
INTERIOR_ERODE_RATIO = 0.15
LBP_SCALES = ((8, 1.0), (16, 2.0))
RESIDUAL_C = 0.3
MAX_ITERATIONS = 300
DESCRIPTOR_THREADS = 8
FOCUS_CLASS_INDEX = 1
FOCUS_MILESTONE = 0.70
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
LITERATURE = (
    "https://doi.org/10.1109/TPAMI.2002.1017623",
    "https://citeseerx.ist.psu.edu/document?doi=8e01f162182365c7a275fb6b7ecaafe7b9719673&repid=rep1&type=pdf",
    "https://library.imaging.org/cgiv/articles/3/1/art00012",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only readiness audit for multi-resolution uniform "
            "LBP on the eroded object interior. It never reads test, edits raw "
            "data, or writes a model/checkpoint."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--descriptor-threads", type=int, default=DESCRIPTOR_THREADS)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--roi-size", type=int, default=ROI_SIZE)
    parser.add_argument(
        "--interior-erode-ratio",
        type=float,
        default=INTERIOR_ERODE_RATIO,
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--residual-c", type=float, default=RESIDUAL_C)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_protocol(args: argparse.Namespace) -> None:
    if int(args.roi_size) < 32:
        raise ValueError("roi-size must be at least 32")
    if not 0.0 <= float(args.interior_erode_ratio) < 0.45:
        raise ValueError("interior-erode-ratio must be in [0, 0.45)")
    if int(args.folds) < 3:
        raise ValueError("At least three source-grouped folds are required")
    if float(args.residual_c) <= 0.0:
        raise ValueError("residual-c must be positive")
    if int(args.max_iterations) < 50:
        raise ValueError("max-iterations must be at least 50")
    if (
        int(args.batch_size) <= 0
        or int(args.workers) < 0
        or int(args.descriptor_threads) <= 0
    ):
        raise ValueError("batch-size/workers/descriptor-threads are invalid")
    output_dir = Path(args.output_dir).resolve()
    dataset_root = Path(args.data).resolve().parent
    if _is_relative_to(output_dir, dataset_root):
        raise ValueError("Output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")


def interior_roi_boxes(
    crop_bbox: Tensor,
    *,
    image_height: int,
    image_width: int,
    erode_ratio: float = INTERIOR_ERODE_RATIO,
) -> Tensor:
    if crop_bbox.ndim != 2 or int(crop_bbox.size(1)) < 4:
        raise ValueError("crop_bbox must be [B,4]")
    erode = float(erode_ratio)
    if not 0.0 <= erode < 0.45:
        raise ValueError("erode_ratio must be in [0, 0.45)")
    values = crop_bbox[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = values.unbind(dim=1)
    half_width = 0.5 * width * (1.0 - 2.0 * erode)
    half_height = 0.5 * height * (1.0 - 2.0 * erode)
    x1 = ((cx - half_width) * float(image_width)).clamp(0.0, float(image_width - 1))
    y1 = ((cy - half_height) * float(image_height)).clamp(0.0, float(image_height - 1))
    x2 = ((cx + half_width) * float(image_width)).clamp(1.0, float(image_width))
    y2 = ((cy + half_height) * float(image_height)).clamp(1.0, float(image_height))
    x2 = torch.maximum(x2, x1 + 1.0)
    y2 = torch.maximum(y2, y1 + 1.0)
    batch_index = torch.arange(
        int(values.size(0)),
        device=values.device,
        dtype=torch.float32,
    )
    return torch.stack((batch_index, x1, y1, x2, y2), dim=1)


def uniform_lbp_descriptor(
    gray_uint8: np.ndarray,
    *,
    scales: Sequence[Tuple[int, float]] = LBP_SCALES,
) -> Tuple[np.ndarray, list[np.ndarray]]:
    gray = np.asarray(gray_uint8)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("gray_uint8 must be a non-empty 2D image")
    if gray.dtype != np.uint8:
        raise ValueError("gray_uint8 must use uint8 values")
    descriptor_parts = []
    maps = []
    for points, radius in scales:
        points = int(points)
        radius = float(radius)
        if points < 4 or radius <= 0.0:
            raise ValueError("LBP scale is invalid")
        code = local_binary_pattern(gray, points, radius, method="uniform")
        margin = int(math.ceil(radius))
        if int(code.shape[0]) <= 2 * margin or int(code.shape[1]) <= 2 * margin:
            raise ValueError("LBP image is too small for the requested radius")
        valid_code = code[margin:-margin, margin:-margin]
        bins = points + 2
        histogram = np.bincount(
            np.asarray(valid_code, dtype=np.int64).reshape(-1),
            minlength=bins,
        ).astype(np.float64)
        histogram /= max(float(histogram.sum()), 1.0)
        descriptor_parts.append(histogram.astype(np.float32))
        maps.append(np.asarray(code, dtype=np.float32))
    descriptor = np.concatenate(descriptor_parts, axis=0).astype(np.float32)
    if not np.isfinite(descriptor).all():
        raise ValueError("LBP descriptor contains non-finite values")
    return descriptor, maps


def _normalize_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).resolve(strict=False)).replace("/", "\\").casefold()


def _assert_dataset_cache_alignment(dataset, cache: Mapping[str, np.ndarray]) -> None:
    rows = int(cache["labels"].shape[0])
    if len(dataset) != rows:
        raise ValueError(f"Dataset/cache row mismatch: {len(dataset)} != {rows}")
    sample_indices = np.asarray(cache["sample_index"], dtype=np.int64)
    if not np.array_equal(sample_indices, np.arange(rows, dtype=np.int64)):
        raise ValueError("Cache sample_index must match deterministic dataset order")
    labels_fn = getattr(dataset, "labels", None)
    paths_fn = getattr(dataset, "sample_paths", None)
    if not callable(labels_fn) or not callable(paths_fn):
        raise ValueError("Dataset must expose labels() and sample_paths()")
    dataset_labels = np.asarray(labels_fn(), dtype=np.int64)
    dataset_paths = np.asarray([_normalize_path(value) for value in paths_fn()], dtype=object)
    cache_paths = np.asarray(
        [_normalize_path(value) for value in cache["paths"]],
        dtype=object,
    )
    if not np.array_equal(dataset_labels, np.asarray(cache["labels"], dtype=np.int64)):
        raise ValueError("Dataset/cache labels differ")
    if not np.array_equal(dataset_paths, cache_paths):
        raise ValueError("Dataset/cache paths differ")


def _preview_kind(label: int, prediction: int) -> str:
    if label == FOCUS_CLASS_INDEX and prediction != FOCUS_CLASS_INDEX:
        return "class1_fn"
    if label != FOCUS_CLASS_INDEX and prediction == FOCUS_CLASS_INDEX:
        return "class1_fp"
    return "correct" if label == prediction else "other_error"


def _want_preview(
    *,
    label: int,
    prediction: int,
    correct_classes: set[int],
    fn_count: int,
    fp_count: int,
) -> bool:
    kind = _preview_kind(label, prediction)
    if kind == "correct" and label not in correct_classes:
        return True
    if kind == "class1_fn" and fn_count < 4:
        return True
    return kind == "class1_fp" and fp_count < 4


def _extract_lbp_split(
    *,
    dataset,
    cache: Mapping[str, np.ndarray],
    checkpoint: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    workers: int,
    roi_size: int,
    erode_ratio: float,
    descriptor_threads: int,
    split: str,
) -> Dict[str, object]:
    _assert_dataset_cache_alignment(dataset, cache)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(workers) > 0),
        collate_fn=_collate_classification,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    descriptors = []
    labels = []
    previews = []
    correct_classes: set[int] = set()
    fn_count = 0
    fp_count = 0
    row_offset = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(descriptor_threads))) as executor:
        for images, targets, metadata in tqdm(loader, desc=f"lbp-{split}", dynamic_ncols=True):
            if not isinstance(metadata, Mapping) or not torch.is_tensor(metadata.get("crop_bbox")):
                raise ValueError("classification crop_bbox metadata is required")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            crop_bbox = metadata["crop_bbox"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            boxes = interior_roi_boxes(
                crop_bbox,
                image_height=int(rgb.size(2)),
                image_width=int(rgb.size(3)),
                erode_ratio=erode_ratio,
            )
            roi = roi_align(
                rgb,
                boxes,
                output_size=(int(roi_size), int(roi_size)),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
            gray = (
                0.299 * roi[:, 0]
                + 0.587 * roi[:, 1]
                + 0.114 * roi[:, 2]
            )
            gray_uint8 = (
                gray.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8).cpu().numpy()
            )
            rgb_uint8 = (
                roi.mul(255.0)
                .round()
                .clamp(0.0, 255.0)
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            batch_labels = targets.cpu().numpy().astype(np.int64, copy=False)
            batch_predictions = np.asarray(
                cache["probabilities"][row_offset : row_offset + len(batch_labels)].argmax(axis=1),
                dtype=np.int64,
            )
            batch_lbp = list(executor.map(uniform_lbp_descriptor, gray_uint8))
            for local_index, (lbp_result, rgb_image, label, prediction) in enumerate(
                zip(batch_lbp, rgb_uint8, batch_labels, batch_predictions)
            ):
                descriptor, maps = lbp_result
                descriptors.append(descriptor)
                labels.append(int(label))
                if _want_preview(
                    label=int(label),
                    prediction=int(prediction),
                    correct_classes=correct_classes,
                    fn_count=fn_count,
                    fp_count=fp_count,
                ):
                    kind = _preview_kind(int(label), int(prediction))
                    previews.append(
                        {
                            "split": split,
                            "row_index": int(row_offset + local_index),
                            "sample_index": int(cache["sample_index"][row_offset + local_index]),
                            "source_stem": str(cache["source_stems"][row_offset + local_index]),
                            "image_path": str(cache["paths"][row_offset + local_index]),
                            "target": int(label),
                            "keeper_prediction": int(prediction),
                            "kind": kind,
                            "rgb": np.asarray(rgb_image, dtype=np.uint8),
                            "maps": maps,
                        }
                    )
                    if kind == "correct":
                        correct_classes.add(int(label))
                    elif kind == "class1_fn":
                        fn_count += 1
                    elif kind == "class1_fp":
                        fp_count += 1
            row_offset += len(batch_labels)
    if row_offset != len(dataset):
        raise RuntimeError(f"Incomplete LBP extraction: {row_offset} != {len(dataset)}")
    feature_array = np.stack(descriptors, axis=0).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if not np.array_equal(label_array, np.asarray(cache["labels"], dtype=np.int64)):
        raise RuntimeError("Extracted labels do not match cache")
    scale_sums = []
    offset = 0
    for points, _radius in LBP_SCALES:
        width = int(points) + 2
        scale_sums.append(feature_array[:, offset : offset + width].sum(axis=1))
        offset += width
    histogram_error = float(
        max(np.max(np.abs(values - 1.0)) for values in scale_sums)
    )
    return {
        "features": feature_array,
        "labels": label_array,
        "previews": previews,
        "seconds": float(time.time() - start),
        "histogram_sum_max_abs_error": histogram_error,
    }


def _fit_scaler(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def _apply_scaler(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (
        (np.asarray(features, dtype=np.float64) - mean) / scale
    ).astype(np.float64, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def apply_offset_residual(
    base_probabilities: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_probabilities, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if base.ndim != 2 or x.ndim != 2 or w.shape != (x.shape[1], base.shape[1]):
        raise ValueError("Offset residual arrays are not aligned")
    if b.shape != (base.shape[1],):
        raise ValueError("Offset residual bias has the wrong shape")
    logits = np.log(np.clip(base, 1e-7, 1.0)) + x @ w + b
    return _softmax(logits)


def fit_offset_residual(
    *,
    fit_features: np.ndarray,
    fit_base_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_base_probabilities: np.ndarray,
    c_value: float,
    max_iterations: int,
    fit_bias: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x_fit = np.asarray(fit_features, dtype=np.float64)
    y_fit = np.asarray(fit_labels, dtype=np.int64)
    base_fit = np.asarray(fit_base_probabilities, dtype=np.float64)
    x_eval = np.asarray(eval_features, dtype=np.float64)
    base_eval = np.asarray(eval_base_probabilities, dtype=np.float64)
    rows, feature_dim = x_fit.shape
    class_count = int(base_fit.shape[1])
    if rows == 0 or base_fit.shape[0] != rows or y_fit.shape[0] != rows:
        raise ValueError("Residual fit arrays are empty or misaligned")
    if x_eval.shape[1] != feature_dim or base_eval.shape[1] != class_count:
        raise ValueError("Residual evaluation arrays are misaligned")
    if not 0.0 < float(c_value) or int(max_iterations) <= 0:
        raise ValueError("Residual optimizer settings are invalid")
    base_logits = np.log(np.clip(base_fit, 1e-7, 1.0))
    bias_count = class_count if bool(fit_bias) else 0
    parameter_count = feature_dim * class_count + bias_count

    def objective(parameters: np.ndarray):
        weights = parameters[: feature_dim * class_count].reshape(feature_dim, class_count)
        bias = (
            parameters[feature_dim * class_count :]
            if bool(fit_bias)
            else np.zeros(class_count, dtype=np.float64)
        )
        logits = base_logits + x_fit @ weights + bias
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        log_normalizer = np.log(exponent.sum(axis=1)) + logits.max(axis=1)
        loss = float(
            np.mean(log_normalizer - logits[np.arange(rows), y_fit])
            + 0.5 * np.square(weights).sum() / float(c_value)
        )
        gradient_logits = probabilities
        gradient_logits[np.arange(rows), y_fit] -= 1.0
        gradient_logits /= float(rows)
        gradient_weights = x_fit.T @ gradient_logits + weights / float(c_value)
        gradient_parts = [gradient_weights.reshape(-1)]
        if bool(fit_bias):
            gradient_parts.append(gradient_logits.sum(axis=0))
        gradient = np.concatenate(gradient_parts, axis=0)
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(parameter_count, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 50,
        },
    )
    weights = result.x[: feature_dim * class_count].reshape(feature_dim, class_count)
    bias = (
        result.x[feature_dim * class_count :]
        if bool(fit_bias)
        else np.zeros(class_count, dtype=np.float64)
    )
    probabilities = apply_offset_residual(base_eval, x_eval, weights, bias)
    return probabilities, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "weight_l2": float(np.linalg.norm(weights)),
        "bias_l2": float(np.linalg.norm(bias)),
        "fit_bias": bool(fit_bias),
    }


def _focus(metrics: Mapping[str, object], focus_class_index: int) -> Mapping[str, object]:
    return metrics["per_class"][int(focus_class_index)]


def assess_lbp_readiness(
    *,
    train_rows: int,
    val_rows: int,
    fold_source_overlap: int,
    train_val_source_overlap: int,
    histogram_sum_max_abs_error: float,
    descriptor_effective_rank: float,
    all_optimizers_converged: bool,
    folds_with_focus_gain: int,
    fold_count: int,
    oof_keeper: Mapping[str, object],
    oof_candidate: Mapping[str, object],
    val_keeper: Mapping[str, object],
    val_lbp: Mapping[str, object],
    val_candidate: Mapping[str, object],
    oof_transitions: Mapping[str, int],
    val_transitions: Mapping[str, int],
    oof_direction: Mapping[str, float],
    val_direction: Mapping[str, float],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    oof_keeper_focus = _focus(oof_keeper, focus)
    oof_candidate_focus = _focus(oof_candidate, focus)
    keeper_focus = _focus(val_keeper, focus)
    val_lbp_focus = _focus(val_lbp, focus)
    candidate_focus = _focus(val_candidate, focus)
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "fold_source_overlap": int(fold_source_overlap),
        "train_val_source_overlap": int(train_val_source_overlap),
        "histogram_sum_max_abs_error": float(histogram_sum_max_abs_error),
        "descriptor_effective_rank": float(descriptor_effective_rank),
        "all_optimizers_converged": bool(all_optimizers_converged),
        "folds_with_focus_gain": int(folds_with_focus_gain),
        "fold_count": int(fold_count),
        "oof_macro_gain": float(oof_candidate["macro_f1"] - oof_keeper["macro_f1"]),
        "oof_focus_gain": float(oof_candidate_focus["f1"] - oof_keeper_focus["f1"]),
        "val_lbp_macro_f1": float(val_lbp["macro_f1"]),
        "val_lbp_focus_f1": float(val_lbp_focus["f1"]),
        "keeper_macro_f1": float(val_keeper["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "keeper_focus_recall": float(keeper_focus["recall"]),
        "candidate_macro_f1": float(val_candidate["macro_f1"]),
        "candidate_focus_f1": float(candidate_focus["f1"]),
        "candidate_focus_precision": float(candidate_focus["precision"]),
        "candidate_focus_recall": float(candidate_focus["recall"]),
        "val_macro_gain": float(val_candidate["macro_f1"] - val_keeper["macro_f1"]),
        "val_focus_gain": float(candidate_focus["f1"] - keeper_focus["f1"]),
        "oof_transitions": dict(oof_transitions),
        "val_transitions": dict(val_transitions),
        "oof_direction_auc": float(oof_direction["auc_fn_positive"]),
        "val_direction_auc": float(val_direction["auc_fn_positive"]),
        "direction_auc_gap": float(
            abs(oof_direction["auc_fn_positive"] - val_direction["auc_fn_positive"])
        ),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "source_group_folds_disjoint": observed["fold_source_overlap"] == 0,
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "lbp_histograms_normalized": observed["histogram_sum_max_abs_error"] <= 1e-5,
        "descriptor_effective_rank_ge_5": observed["descriptor_effective_rank"] >= 5.0,
        "all_offset_optimizers_converged": bool(all_optimizers_converged),
        "focus_gain_in_at_least_3_folds": observed["folds_with_focus_gain"]
        >= min(3, observed["fold_count"]),
        "oof_macro_nonnegative": observed["oof_macro_gain"] >= 0.0,
        "oof_focus_gain_ge_0p005": observed["oof_focus_gain"] >= 0.005,
        "oof_corrections_ge_harms": int(oof_transitions["corrections"])
        >= int(oof_transitions["harms"]),
        "oof_focus_fp_removed_ge_created": int(
            oof_transitions["focus_false_positive_removed"]
        )
        >= int(oof_transitions["focus_false_positive_created"]),
        "oof_focus_fn_rescued_ge_tp_broken": int(
            oof_transitions["focus_false_negative_rescued"]
        )
        >= int(oof_transitions["focus_true_positive_broken"]),
        "standalone_lbp_val_focus_ge_0p50": observed["val_lbp_focus_f1"] >= 0.50,
        "val_macro_preserves_keeper_within_0p001": observed["val_macro_gain"] >= -0.001,
        "val_focus_reaches_0p70": observed["candidate_focus_f1"] >= FOCUS_MILESTONE,
        "val_focus_improves_keeper_by_0p01": observed["val_focus_gain"] >= 0.01,
        "val_focus_recall_preserved_within_0p01": observed["candidate_focus_recall"]
        >= observed["keeper_focus_recall"] - 0.01,
        "val_corrections_ge_harms": int(val_transitions["corrections"])
        >= int(val_transitions["harms"]),
        "val_focus_fp_removed_ge_created": int(
            val_transitions["focus_false_positive_removed"]
        )
        >= int(val_transitions["focus_false_positive_created"]),
        "val_focus_fn_rescued_ge_tp_broken": int(
            val_transitions["focus_false_negative_rescued"]
        )
        >= int(val_transitions["focus_true_positive_broken"]),
        "oof_direction_auc_ge_0p60": observed["oof_direction_auc"] >= 0.60,
        "val_direction_auc_ge_0p60": observed["val_direction_auc"] >= 0.60,
        "direction_auc_gap_le_0p15": observed["direction_auc_gap"] <= 0.15,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "image_smoke_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _effective_rank(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    energy = np.square(singular_values)
    probability = energy / max(float(energy.sum()), 1e-12)
    return float(math.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))))


def _write_preview(previews: Sequence[Mapping[str, object]], output_dir: Path) -> Dict[str, object]:
    if not previews:
        return {"rows": 0, "path": ""}
    tile = 160
    label_height = 42
    columns = 3
    canvas = Image.new(
        "RGB",
        (columns * tile, len(previews) * (tile + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    json_rows = []
    for row_index, item in enumerate(previews):
        y = row_index * (tile + label_height)
        rgb = Image.fromarray(np.asarray(item["rgb"], dtype=np.uint8)).resize(
            (tile, tile),
            Image.Resampling.BILINEAR,
        )
        canvas.paste(rgb, (0, y))
        for map_index, code in enumerate(item["maps"]):
            points = int(LBP_SCALES[map_index][0])
            normalized = np.asarray(code, dtype=np.float32) / float(points + 1)
            gray = Image.fromarray(
                (normalized.clip(0.0, 1.0) * 255.0).round().astype(np.uint8),
            ).convert("RGB")
            canvas.paste(gray.resize((tile, tile), Image.Resampling.NEAREST), ((map_index + 1) * tile, y))
        label = (
            f"{item['kind']} row={item['row_index']} "
            f"target={item['target']} pred={item['keeper_prediction']} "
            f"source={item['source_stem']}"
        )
        draw.text((4, y + tile + 3), label[:95], fill=(0, 0, 0), font=font)
        draw.text((4, y + tile + 19), "RGB | LBP(8,1) | LBP(16,2)", fill=(0, 0, 0), font=font)
        json_rows.append({key: value for key, value in item.items() if key not in {"rgb", "maps"}})
    image_path = output_dir / "lbp_surface_preview.png"
    json_path = output_dir / "lbp_surface_preview.json"
    canvas.save(image_path)
    json_path.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    return {"rows": int(len(previews)), "path": str(image_path), "json": str(json_path)}


def _prediction_rows(
    *,
    split: str,
    cache: Mapping[str, np.ndarray],
    fold_assignment: np.ndarray,
    standalone: np.ndarray,
    candidate: np.ndarray,
) -> list[Dict[str, object]]:
    rows = []
    for index in range(int(cache["labels"].shape[0])):
        row: Dict[str, object] = {
            "split": split,
            "sample_index": int(cache["sample_index"][index]),
            "fold": int(fold_assignment[index]),
            "source_stem": str(cache["source_stems"][index]),
            "image_path": str(cache["paths"][index]),
            "target_index": int(cache["labels"][index]),
            "keeper_prediction_index": int(cache["probabilities"][index].argmax()),
            "lbp_prediction_index": int(standalone[index].argmax()),
            "candidate_prediction_index": int(candidate[index].argmax()),
        }
        for class_index in range(int(candidate.shape[1])):
            row[f"keeper_prob_{class_index}"] = float(cache["probabilities"][index, class_index])
            row[f"lbp_prob_{class_index}"] = float(standalone[index, class_index])
            row[f"candidate_prob_{class_index}"] = float(candidate[index, class_index])
        rows.append(row)
    return rows


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
        "mode": "lbp_surface_texture_readiness_evidence_manifest",
        "payload_count": int(len(files)),
        "payload_size_bytes": int(sum(int(row["size_bytes"]) for row in files)),
        "payload_manifest_sha256": aggregate.hexdigest(),
        "files": files,
        "contains_checkpoint": any(str(row["name"]).lower().endswith(".pt") for row in files),
        "contains_model_binary": any(
            str(row["name"]).lower().endswith((".pt", ".pth", ".engine"))
            for row in files
        ),
        "contains_test_payload": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_protocol(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    train_cache_path = Path(args.train_cache).resolve()
    val_cache_path = Path(args.val_cache).resolve()
    train = _load_cache(train_cache_path, split="train")
    val = _load_cache(val_cache_path, split="val")
    if int(args.max_train_samples) > 0:
        limit = min(int(args.max_train_samples), int(train["labels"].shape[0]))
        train = {key: np.asarray(value)[:limit] for key, value in train.items()}
    if int(args.max_val_samples) > 0:
        limit = min(int(args.max_val_samples), int(val["labels"].shape[0]))
        val = {key: np.asarray(value)[:limit] for key, value in val.items()}
    train_dataset, class_names = _build_dataset(
        data_yaml=Path(args.data),
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=int(args.max_train_samples),
    )
    val_dataset, val_class_names = _build_dataset(
        data_yaml=Path(args.data),
        split="val",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=int(args.max_val_samples),
    )
    if list(class_names) != list(val_class_names):
        raise ValueError("Train/validation class order differs")
    device = _resolve_device(str(args.device))
    extracted_train = _extract_lbp_split(
        dataset=train_dataset,
        cache=train,
        checkpoint=checkpoint,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        roi_size=int(args.roi_size),
        erode_ratio=float(args.interior_erode_ratio),
        descriptor_threads=int(args.descriptor_threads),
        split="train",
    )
    extracted_val = _extract_lbp_split(
        dataset=val_dataset,
        cache=val,
        checkpoint=checkpoint,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        roi_size=int(args.roi_size),
        erode_ratio=float(args.interior_erode_ratio),
        descriptor_threads=int(args.descriptor_threads),
        split="val",
    )
    x_train = np.asarray(extracted_train["features"], dtype=np.float32)
    x_val = np.asarray(extracted_val["features"], dtype=np.float32)
    y_train = np.asarray(train["labels"], dtype=np.int64)
    y_val = np.asarray(val["labels"], dtype=np.int64)
    train_groups = np.asarray(train["source_stems"], dtype=object)
    val_groups = np.asarray(val["source_stems"], dtype=object)
    train_val_source_overlap = len(set(train_groups.tolist()).intersection(set(val_groups.tolist())))
    class_count = int(train["probabilities"].shape[1])
    focus_class_index = int(args.focus_class_index)
    if class_count != len(class_names) or not 0 <= focus_class_index < class_count:
        raise ValueError("Class count/focus index is invalid")
    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    oof_standalone = np.zeros((len(y_train), class_count), dtype=np.float32)
    oof_candidate = np.zeros_like(oof_standalone)
    fold_assignment = np.full(len(y_train), -1, dtype=np.int64)
    fold_rows = []
    fold_protocols = []
    maximum_fold_overlap = 0
    all_optimizers_converged = True
    uniform_train = np.full_like(train["probabilities"], 1.0 / float(class_count))
    for fold_index, (fit_indices, hold_indices) in enumerate(
        splitter.split(x_train, y_train, groups=train_groups)
    ):
        fit_sources = set(train_groups[fit_indices].tolist())
        hold_sources = set(train_groups[hold_indices].tolist())
        overlap = len(fit_sources.intersection(hold_sources))
        maximum_fold_overlap = max(maximum_fold_overlap, overlap)
        mean, scale = _fit_scaler(x_train[fit_indices])
        fit_features = _apply_scaler(x_train[fit_indices], mean, scale)
        hold_features = _apply_scaler(x_train[hold_indices], mean, scale)
        standalone, standalone_fit = fit_offset_residual(
            fit_features=fit_features,
            fit_base_probabilities=uniform_train[fit_indices],
            fit_labels=y_train[fit_indices],
            eval_features=hold_features,
            eval_base_probabilities=uniform_train[hold_indices],
            c_value=float(args.residual_c),
            max_iterations=int(args.max_iterations),
            fit_bias=True,
        )
        candidate, candidate_fit = fit_offset_residual(
            fit_features=fit_features,
            fit_base_probabilities=train["probabilities"][fit_indices],
            fit_labels=y_train[fit_indices],
            eval_features=hold_features,
            eval_base_probabilities=train["probabilities"][hold_indices],
            c_value=float(args.residual_c),
            max_iterations=int(args.max_iterations),
            fit_bias=False,
        )
        oof_standalone[hold_indices] = standalone
        oof_candidate[hold_indices] = candidate
        fold_assignment[hold_indices] = int(fold_index)
        keeper_metrics = _classification_metrics(y_train[hold_indices], train["probabilities"][hold_indices])
        standalone_metrics = _classification_metrics(y_train[hold_indices], standalone)
        candidate_metrics = _classification_metrics(y_train[hold_indices], candidate)
        focus_gain = float(
            _focus(candidate_metrics, focus_class_index)["f1"]
            - _focus(keeper_metrics, focus_class_index)["f1"]
        )
        fold_rows.append(
            {
                "fold": int(fold_index),
                "rows": int(len(hold_indices)),
                "source_overlap": int(overlap),
                "keeper_macro_f1": float(keeper_metrics["macro_f1"]),
                "keeper_focus_f1": float(_focus(keeper_metrics, focus_class_index)["f1"]),
                "lbp_macro_f1": float(standalone_metrics["macro_f1"]),
                "lbp_focus_f1": float(_focus(standalone_metrics, focus_class_index)["f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "candidate_focus_f1": float(_focus(candidate_metrics, focus_class_index)["f1"]),
                "candidate_focus_gain": focus_gain,
            }
        )
        fold_protocols.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": int(overlap),
                "standalone_fit": standalone_fit,
                "candidate_fit": candidate_fit,
            }
        )
        all_optimizers_converged = all_optimizers_converged and bool(
            standalone_fit["success"] and candidate_fit["success"]
        )
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF assignment is incomplete")
    mean, scale = _fit_scaler(x_train)
    train_scaled = _apply_scaler(x_train, mean, scale)
    val_scaled = _apply_scaler(x_val, mean, scale)
    uniform_val = np.full_like(val["probabilities"], 1.0 / float(class_count))
    val_standalone, standalone_full_fit = fit_offset_residual(
        fit_features=train_scaled,
        fit_base_probabilities=uniform_train,
        fit_labels=y_train,
        eval_features=val_scaled,
        eval_base_probabilities=uniform_val,
        c_value=float(args.residual_c),
        max_iterations=int(args.max_iterations),
        fit_bias=True,
    )
    val_candidate, candidate_full_fit = fit_offset_residual(
        fit_features=train_scaled,
        fit_base_probabilities=train["probabilities"],
        fit_labels=y_train,
        eval_features=val_scaled,
        eval_base_probabilities=val["probabilities"],
        c_value=float(args.residual_c),
        max_iterations=int(args.max_iterations),
        fit_bias=False,
    )
    all_optimizers_converged = all_optimizers_converged and bool(
        standalone_full_fit["success"] and candidate_full_fit["success"]
    )
    metrics = {
        "train_oof": {
            "keeper_in_sample_reference": _classification_metrics(y_train, train["probabilities"]),
            "lbp_standalone": _classification_metrics(y_train, oof_standalone),
            "keeper_lbp_residual": _classification_metrics(y_train, oof_candidate),
        },
        "val": {
            "keeper": _classification_metrics(y_val, val["probabilities"]),
            "lbp_standalone": _classification_metrics(y_val, val_standalone),
            "keeper_lbp_residual": _classification_metrics(y_val, val_candidate),
        },
    }
    transitions = {
        "train_oof_candidate_vs_keeper": _transition_stats(
            y_train,
            train["probabilities"],
            oof_candidate,
            focus_class_index=focus_class_index,
        ),
        "val_candidate_vs_keeper": _transition_stats(
            y_val,
            val["probabilities"],
            val_candidate,
            focus_class_index=focus_class_index,
        ),
    }
    direction = {
        "train_oof_candidate_vs_keeper": _direction_auc(
            y_train,
            train["probabilities"],
            oof_candidate,
            focus_class_index=focus_class_index,
        ),
        "val_candidate_vs_keeper": _direction_auc(
            y_val,
            val["probabilities"],
            val_candidate,
            focus_class_index=focus_class_index,
        ),
    }
    folds_with_focus_gain = sum(float(row["candidate_focus_gain"]) > 0.0 for row in fold_rows)
    histogram_error = max(
        float(extracted_train["histogram_sum_max_abs_error"]),
        float(extracted_val["histogram_sum_max_abs_error"]),
    )
    descriptor_effective_rank = _effective_rank(train_scaled)
    gate = assess_lbp_readiness(
        train_rows=len(y_train),
        val_rows=len(y_val),
        fold_source_overlap=maximum_fold_overlap,
        train_val_source_overlap=train_val_source_overlap,
        histogram_sum_max_abs_error=histogram_error,
        descriptor_effective_rank=descriptor_effective_rank,
        all_optimizers_converged=all_optimizers_converged,
        folds_with_focus_gain=folds_with_focus_gain,
        fold_count=int(args.folds),
        oof_keeper=metrics["train_oof"]["keeper_in_sample_reference"],
        oof_candidate=metrics["train_oof"]["keeper_lbp_residual"],
        val_keeper=metrics["val"]["keeper"],
        val_lbp=metrics["val"]["lbp_standalone"],
        val_candidate=metrics["val"]["keeper_lbp_residual"],
        oof_transitions=transitions["train_oof_candidate_vs_keeper"],
        val_transitions=transitions["val_candidate_vs_keeper"],
        oof_direction=direction["train_oof_candidate_vs_keeper"],
        val_direction=direction["val_candidate_vs_keeper"],
        focus_class_index=focus_class_index,
        test_split_used=False,
    )
    preview = _write_preview(extracted_val["previews"], output_dir)
    np.savez_compressed(
        output_dir / "lbp_descriptors_train_val.npz",
        train_features=x_train,
        train_labels=y_train,
        train_sample_index=train["sample_index"],
        train_paths=train["paths"],
        val_features=x_val,
        val_labels=y_val,
        val_sample_index=val["sample_index"],
        val_paths=val["paths"],
    )
    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            cache=train,
            fold_assignment=fold_assignment,
            standalone=oof_standalone,
            candidate=oof_candidate,
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            cache=val,
            fold_assignment=np.full(len(y_val), -1, dtype=np.int64),
            standalone=val_standalone,
            candidate=val_candidate,
        ),
    )
    protocol = {
        "method": "multiresolution_uniform_lbp_interior_keeper_offset_readiness",
        "scope": (
            "A fixed grayscale microtexture precheck on the eroded runtime object ROI. "
            "It is not an image-model branch or a claim that labels are clean."
        ),
        "literature": list(LITERATURE),
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path),
        "train_cache": str(train_cache_path),
        "val_cache": str(val_cache_path),
        "split_usage": {"train": True, "val": True, "test": False},
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_source_groups": int(np.unique(train_groups).size),
        "val_source_groups": int(np.unique(val_groups).size),
        "train_val_source_overlap": int(train_val_source_overlap),
        "roi_size": int(args.roi_size),
        "interior_erode_ratio": float(args.interior_erode_ratio),
        "lbp_method": "uniform",
        "lbp_scales": [
            {"points": int(points), "radius": float(radius), "bins": int(points) + 2}
            for points, radius in LBP_SCALES
        ],
        "descriptor_dim": int(x_train.shape[1]),
        "grayscale": "BT.601 luminance from checkpoint-denormalized RGB",
        "residual": (
            "zero-initialized linear LBP residual added to fixed keeper log probabilities; "
            "no residual bias; natural-frequency mean CE plus L2/(2*C)"
        ),
        "descriptor_threads": int(args.descriptor_threads),
        "residual_c": float(args.residual_c),
        "max_iterations": int(args.max_iterations),
        "folds": int(args.folds),
        "source_grouped": True,
        "underlying_keeper_train_cache_is_not_oof": True,
        "validation_hyperparameter_tuning": False,
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
        "test_split_used": False,
    }
    summary = {
        "protocol": protocol,
        "extraction": {
            "train_seconds": float(extracted_train["seconds"]),
            "val_seconds": float(extracted_val["seconds"]),
            "histogram_sum_max_abs_error": histogram_error,
            "descriptor_effective_rank": descriptor_effective_rank,
            "preview": preview,
        },
        "fold_protocols": fold_protocols,
        "full_fit_protocol": {
            "standalone_fit": standalone_full_fit,
            "candidate_fit": candidate_full_fit,
        },
        "metrics": metrics,
        "transitions": transitions,
        "focus_direction": direction,
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
        "seconds": float(time.time() - start),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    observed = gate["observed"]
    readme = [
        "# Interior Multi-Resolution LBP Readiness",
        "",
        f"- Train in-sample keeper/grouped-OOF residual macro-class1: `{float(metrics['train_oof']['keeper_in_sample_reference']['macro_f1']):.6f}/{float(_focus(metrics['train_oof']['keeper_in_sample_reference'], focus_class_index)['f1']):.6f} -> {float(metrics['train_oof']['keeper_lbp_residual']['macro_f1']):.6f}/{float(_focus(metrics['train_oof']['keeper_lbp_residual'], focus_class_index)['f1']):.6f}`",
        f"- Val keeper/residual macro-class1: `{float(metrics['val']['keeper']['macro_f1']):.6f}/{float(_focus(metrics['val']['keeper'], focus_class_index)['f1']):.6f} -> {float(metrics['val']['keeper_lbp_residual']['macro_f1']):.6f}/{float(_focus(metrics['val']['keeper_lbp_residual'], focus_class_index)['f1']):.6f}`",
        f"- Val standalone LBP macro-class1: `{float(metrics['val']['lbp_standalone']['macro_f1']):.6f}/{float(_focus(metrics['val']['lbp_standalone'], focus_class_index)['f1']):.6f}`",
        f"- Standardized descriptor effective rank: `{float(observed['descriptor_effective_rank']):.6f}`",
        f"- OOF/val direction AUROC: `{float(observed['oof_direction_auc']):.6f}/{float(observed['val_direction_auc']):.6f}`",
        f"- Val FP removed/created: `{int(observed['val_transitions']['focus_false_positive_removed'])}/{int(observed['val_transitions']['focus_false_positive_created'])}`",
        f"- Val FN rescued/TP broken: `{int(observed['val_transitions']['focus_false_negative_rescued'])}/{int(observed['val_transitions']['focus_true_positive_broken'])}`",
        f"- Image smoke permission: `{str(bool(gate['image_smoke_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "Decision: reject before image smoke when any fixed check fails. Do not sweep LBP channels, scales, ROI erosion, residual C, or fold seeds on this evidence.",
        "",
        "This diagnostic uses frozen train/validation caches and immutable runtime crops. It reads no test split and writes no model/checkpoint or training manifest.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(output_dir)
    summary["artifact_manifest"] = {key: value for key, value in manifest.items() if key != "files"}
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(args)
    print(
        json.dumps(
            {
                "metrics": summary["metrics"],
                "transitions": summary["transitions"],
                "focus_direction": summary["focus_direction"],
                "gate": summary["gate"],
                "artifact_manifest": summary["artifact_manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

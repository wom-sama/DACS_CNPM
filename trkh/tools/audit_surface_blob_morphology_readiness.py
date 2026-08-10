from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.color import rgb2hsv, rgb2lab
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.tools.audit_lbp_surface_texture_readiness import (
    _apply_scaler,
    _assert_dataset_cache_alignment,
    _effective_rank,
    _fit_scaler,
    _focus,
    _is_relative_to,
    _resolve_device,
    fit_offset_residual,
    interior_roi_boxes,
)
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
BLOB_SIGMAS = (1.2, 2.4, 4.8)
BACKGROUND_SIGMA = 6.0
PEAK_MAD_MULTIPLIER = 2.5
HIGHLIGHT_QUANTILE = 0.95
CONTROL_DIM = 25
MORPHOLOGY_DIM = 45
RESIDUAL_C = 0.3
MAX_ITERATIONS = 300
DESCRIPTOR_THREADS = 8
FOCUS_CLASS_INDEX = 1
FOCUS_MILESTONE = 0.70
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
LITERATURE = (
    "https://openaccess.thecvf.com/content_cvpr_2013/html/Kim_Specular_Reflection_Separation_2013_CVPR_paper.html",
    "https://openaccess.thecvf.com/content_cvpr_2015/html/Liu_Saturation-Preserving_Specular_Reflection_2015_CVPR_paper.html",
    "https://doi.org/10.1016/j.scienta.2011.11.018",
    "https://doi.org/10.1016/j.jafr.2022.100477",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only readiness audit for scale-normalized interior "
            "surface-blob morphology after fixed achromatic-highlight exclusion. "
            "The highlight map is a proxy, not a physical reflection separation. It never reads "
            "test, edits raw data, or writes a model/checkpoint."
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
    parser.add_argument("--interior-erode-ratio", type=float, default=INTERIOR_ERODE_RATIO)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--residual-c", type=float, default=RESIDUAL_C)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _validate_protocol(args: argparse.Namespace) -> None:
    if int(args.roi_size) < 64:
        raise ValueError("roi-size must be at least 64")
    if not 0.0 <= float(args.interior_erode_ratio) < 0.45:
        raise ValueError("interior-erode-ratio must be in [0, 0.45)")
    if int(args.folds) < 3:
        raise ValueError("At least three source-grouped folds are required")
    if float(args.residual_c) <= 0.0 or int(args.max_iterations) < 50:
        raise ValueError("Residual optimizer settings are invalid")
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


def _robust_location_scale(values: np.ndarray, valid: np.ndarray) -> Tuple[float, float]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if selected.size == 0:
        return 0.0, 1.0
    center = float(np.median(selected))
    mad = float(np.median(np.abs(selected - center)))
    return center, max(1.4826 * mad, 1e-6)


def achromatic_highlight_mask_from_rgb(
    rgb: np.ndarray,
    *,
    quantile: float = HIGHLIGHT_QUANTILE,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 or values.size == 0:
        raise ValueError("rgb must be a non-empty HxWx3 array")
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("quantile must be in [0.5, 1.0)")
    hsv = rgb2hsv(values.clip(0.0, 1.0))
    score = hsv[..., 2] * (1.0 - hsv[..., 1])
    threshold = float(np.quantile(score, float(quantile)))
    mask = (score >= threshold) & (hsv[..., 2] >= float(np.median(hsv[..., 2])))
    return np.asarray(mask, dtype=bool), np.asarray(score, dtype=np.float32)


def _masked_mean_std(values: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if selected.ndim == 1:
        selected = selected[:, None]
    if selected.size == 0:
        width = int(values.shape[-1]) if np.asarray(values).ndim == 3 else 1
        return np.zeros(width, dtype=np.float64), np.zeros(width, dtype=np.float64)
    return selected.mean(axis=0), selected.std(axis=0)


def _quantiles(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if selected.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.quantile(selected, (0.10, 0.50, 0.90)).astype(np.float64)


def diffuse_color_control_descriptor(
    rgb: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32).clip(0.0, 1.0)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 3 or mask.shape != values.shape[:2]:
        raise ValueError("RGB and diffuse mask shapes differ")
    lab = rgb2lab(values).astype(np.float32)
    hsv = rgb2hsv(values).astype(np.float32)
    rgb_mean, rgb_std = _masked_mean_std(values, mask)
    lab_mean, lab_std = _masked_mean_std(lab, mask)
    hue = hsv[..., 0][mask]
    if hue.size:
        radians = 2.0 * math.pi * hue.astype(np.float64)
        hue_sin = float(np.sin(radians).mean())
        hue_cos = float(np.cos(radians).mean())
        hue_concentration = float(math.hypot(hue_sin, hue_cos))
    else:
        hue_sin = hue_cos = hue_concentration = 0.0
    saturation_mean, saturation_std = _masked_mean_std(hsv[..., 1], mask)
    value_mean, value_std = _masked_mean_std(hsv[..., 2], mask)
    luminance = 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]
    chroma = np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2]))
    descriptor = np.concatenate(
        (
            rgb_mean,
            rgb_std,
            lab_mean,
            lab_std,
            np.asarray((hue_sin, hue_cos, hue_concentration), dtype=np.float64),
            np.asarray(
                (
                    float(saturation_mean[0]),
                    float(saturation_std[0]),
                    float(value_mean[0]),
                    float(value_std[0]),
                ),
                dtype=np.float64,
            ),
            _quantiles(luminance, mask),
            _quantiles(chroma, mask),
        ),
        axis=0,
    ).astype(np.float32)
    if descriptor.shape != (CONTROL_DIM,) or not np.isfinite(descriptor).all():
        raise RuntimeError("Diffuse color control descriptor is invalid")
    return descriptor


def _peak_features(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(score, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    radius = max(1, int(math.ceil(float(sigma))))
    local_max = ndimage.maximum_filter(values, size=2 * radius + 1, mode="nearest")
    peaks = mask & (values >= float(PEAK_MAD_MULTIPLIER)) & (values >= local_max - 1e-6)
    labels, count = ndimage.label(peaks)
    peak_values = []
    radii = []
    peak_map = np.zeros_like(values, dtype=np.float32)
    height, width = values.shape
    center_y = 0.5 * float(height - 1)
    center_x = 0.5 * float(width - 1)
    normalizer = max(math.hypot(center_x, center_y), 1.0)
    for label_index in range(1, int(count) + 1):
        ys, xs = np.nonzero(labels == label_index)
        if ys.size == 0:
            continue
        local_values = values[ys, xs]
        best = int(np.argmax(local_values))
        peak_values.append(float(local_values[best]))
        peak_map[int(ys[best]), int(xs[best])] = float(local_values[best])
        radii.append(float(math.hypot(float(xs[best]) - center_x, float(ys[best]) - center_y) / normalizer))
    valid_count = max(int(mask.sum()), 1)
    selected = values[peaks]
    return (
        np.asarray(
            (
                float(len(peak_values)) / float(valid_count),
                float(peaks.sum()) / float(valid_count),
                float(np.mean(peak_values)) if peak_values else 0.0,
                float(np.quantile(selected, 0.90)) if selected.size else 0.0,
                float(np.mean(radii)) if radii else 0.0,
            ),
            dtype=np.float32,
        ),
        peak_map,
    )


def surface_blob_descriptor(
    rgb_uint8: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    image = np.asarray(rgb_uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb_uint8 must be a non-empty uint8 HxWx3 image")
    rgb = image.astype(np.float32) / 255.0
    highlight, highlight_score = achromatic_highlight_mask_from_rgb(rgb)
    valid = ~highlight
    border = int(math.ceil(max(BLOB_SIGMAS) * 2.0))
    if image.shape[0] <= 2 * border or image.shape[1] <= 2 * border:
        raise ValueError("ROI is too small for the locked blob scales")
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    control = diffuse_color_control_descriptor(rgb, valid)

    lab = rgb2lab(rgb).astype(np.float32)
    lightness = lab[..., 0] / 100.0
    chroma = np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2])) / 181.0
    maps: Dict[str, np.ndarray] = {
        "highlight": highlight.astype(np.float32),
        "highlight_score": highlight_score,
    }
    morphology_parts = []
    aggregate_dark = np.zeros_like(lightness, dtype=np.float32)
    aggregate_bright = np.zeros_like(lightness, dtype=np.float32)
    aggregate_chroma = np.zeros_like(lightness, dtype=np.float32)

    for source_name, source in (("lightness", lightness), ("chroma", chroma)):
        background = ndimage.gaussian_filter(source, sigma=BACKGROUND_SIGMA, mode="reflect")
        cleaned = np.where(highlight, background, source)
        for sigma in BLOB_SIGMAS:
            response = (float(sigma) ** 2) * ndimage.gaussian_laplace(
                cleaned,
                sigma=float(sigma),
                mode="reflect",
            )
            center, scale = _robust_location_scale(response, valid)
            standardized = (response - center) / scale
            if source_name == "lightness":
                dark_score = np.maximum(standardized, 0.0).astype(np.float32)
                bright_score = np.maximum(-standardized, 0.0).astype(np.float32)
                dark_features, dark_peaks = _peak_features(dark_score, valid, sigma=float(sigma))
                bright_features, bright_peaks = _peak_features(bright_score, valid, sigma=float(sigma))
                morphology_parts.append(dark_features)
                morphology_parts.append(bright_features)
                aggregate_dark = np.maximum(aggregate_dark, dark_peaks)
                aggregate_bright = np.maximum(aggregate_bright, bright_peaks)
            else:
                chroma_score = np.abs(standardized).astype(np.float32)
                chroma_features, chroma_peaks = _peak_features(chroma_score, valid, sigma=float(sigma))
                morphology_parts.append(chroma_features)
                aggregate_chroma = np.maximum(aggregate_chroma, chroma_peaks)

    morphology = np.concatenate(morphology_parts, axis=0).astype(np.float32)
    if morphology.shape != (MORPHOLOGY_DIM,) or not np.isfinite(morphology).all():
        raise RuntimeError("Surface morphology descriptor is invalid")
    maps.update(
        {
            "dark_blob": aggregate_dark,
            "bright_blob": aggregate_bright,
            "chroma_blob": aggregate_chroma,
            "valid": valid.astype(np.float32),
        }
    )
    telemetry = {
        "highlight_fraction": float(highlight.mean()),
        "valid_fraction": float(valid.mean()),
        "dark_peak_density_sum": float(sum(morphology[offset] for offset in range(0, 30, 10))),
        "bright_peak_density_sum": float(sum(morphology[offset] for offset in range(5, 30, 10))),
        "chroma_peak_density_sum": float(sum(morphology[offset] for offset in range(30, 45, 5))),
    }
    return control, morphology, maps, telemetry


def _preview_kind(label: int, prediction: int) -> str:
    if label == FOCUS_CLASS_INDEX and prediction != FOCUS_CLASS_INDEX:
        return "class1_fn"
    if label != FOCUS_CLASS_INDEX and prediction == FOCUS_CLASS_INDEX:
        return "class1_fp"
    return "correct" if label == prediction else "other_error"


def _extract_split(
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
    controls = []
    morphologies = []
    telemetry_rows = []
    previews = []
    preview_correct: set[int] = set()
    preview_counts = {"class1_fn": 0, "class1_fp": 0}
    row_offset = 0
    start = time.time()
    process_count = max(1, int(descriptor_threads))
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for images, targets, metadata in tqdm(loader, desc=f"surface-blob-{split}", dynamic_ncols=True):
            if not isinstance(metadata, Mapping) or not torch.is_tensor(metadata.get("crop_bbox")):
                raise ValueError("classification crop_bbox metadata is required")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            crop_bbox = metadata["crop_bbox"].to(device=device, dtype=torch.float32, non_blocking=True)
            rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            boxes = interior_roi_boxes(
                crop_bbox,
                image_height=int(rgb.size(2)),
                image_width=int(rgb.size(3)),
                erode_ratio=float(erode_ratio),
            )
            roi = roi_align(
                rgb,
                boxes,
                output_size=(int(roi_size), int(roi_size)),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
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
            chunk_size = max(1, int(math.ceil(len(rgb_uint8) / float(process_count * 2))))
            batch_results = list(
                executor.map(surface_blob_descriptor, rgb_uint8, chunksize=chunk_size)
            )
            for local_index, (result, rgb_image, label, prediction) in enumerate(
                zip(batch_results, rgb_uint8, batch_labels, batch_predictions)
            ):
                control, morphology, maps, telemetry = result
                controls.append(control)
                morphologies.append(morphology)
                telemetry_rows.append(telemetry)
                if split == "val":
                    kind = _preview_kind(int(label), int(prediction))
                    keep = False
                    if kind == "correct" and int(label) not in preview_correct:
                        preview_correct.add(int(label))
                        keep = True
                    elif kind in preview_counts and preview_counts[kind] < 4:
                        preview_counts[kind] += 1
                        keep = True
                    if keep:
                        previews.append(
                            {
                                "row_index": int(row_offset + local_index),
                                "sample_index": int(cache["sample_index"][row_offset + local_index]),
                                "source_stem": str(cache["source_stems"][row_offset + local_index]),
                                "image_path": str(cache["paths"][row_offset + local_index]),
                                "target": int(label),
                                "keeper_prediction": int(prediction),
                                "kind": kind,
                                "rgb": np.asarray(rgb_image, dtype=np.uint8),
                                "maps": maps,
                                "telemetry": telemetry,
                            }
                        )
            row_offset += len(batch_labels)
    if row_offset != len(dataset):
        raise RuntimeError(f"Incomplete descriptor extraction: {row_offset} != {len(dataset)}")
    control_array = np.stack(controls, axis=0).astype(np.float32)
    morphology_array = np.stack(morphologies, axis=0).astype(np.float32)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    if control_array.shape != (len(dataset), CONTROL_DIM):
        raise RuntimeError("Control descriptor matrix has the wrong shape")
    if morphology_array.shape != (len(dataset), MORPHOLOGY_DIM):
        raise RuntimeError("Morphology descriptor matrix has the wrong shape")
    telemetry_summary = {
        key: {
            "mean": float(np.mean([row[key] for row in telemetry_rows])),
            "p10": float(np.quantile([row[key] for row in telemetry_rows], 0.10)),
            "p90": float(np.quantile([row[key] for row in telemetry_rows], 0.90)),
        }
        for key in telemetry_rows[0]
    }
    return {
        "control": control_array,
        "morphology": morphology_array,
        "labels": labels,
        "previews": previews,
        "telemetry": telemetry_summary,
        "finite": bool(np.isfinite(control_array).all() and np.isfinite(morphology_array).all()),
        "seconds": float(time.time() - start),
    }


def _fit_four_readouts(
    *,
    fit_control: np.ndarray,
    fit_candidate: np.ndarray,
    fit_base: np.ndarray,
    fit_labels: np.ndarray,
    eval_control: np.ndarray,
    eval_candidate: np.ndarray,
    eval_base: np.ndarray,
    c_value: float,
    max_iterations: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    class_count = int(fit_base.shape[1])
    uniform_fit = np.full_like(fit_base, 1.0 / float(class_count))
    uniform_eval = np.full_like(eval_base, 1.0 / float(class_count))
    probabilities: Dict[str, np.ndarray] = {}
    fits: Dict[str, object] = {}
    for name, fit_features, eval_features, fit_probabilities, eval_probabilities, fit_bias in (
        ("control_standalone", fit_control, eval_control, uniform_fit, uniform_eval, True),
        ("candidate_standalone", fit_candidate, eval_candidate, uniform_fit, uniform_eval, True),
        ("keeper_control_residual", fit_control, eval_control, fit_base, eval_base, False),
        ("keeper_candidate_residual", fit_candidate, eval_candidate, fit_base, eval_base, False),
    ):
        probabilities[name], fits[name] = fit_offset_residual(
            fit_features=fit_features,
            fit_base_probabilities=fit_probabilities,
            fit_labels=fit_labels,
            eval_features=eval_features,
            eval_base_probabilities=eval_probabilities,
            c_value=float(c_value),
            max_iterations=int(max_iterations),
            fit_bias=bool(fit_bias),
        )
    return probabilities, fits


def assess_surface_blob_readiness(
    *,
    train_rows: int,
    val_rows: int,
    fold_source_overlap: int,
    train_val_source_overlap: int,
    descriptors_finite: bool,
    morphology_effective_rank: float,
    mean_highlight_fraction: float,
    all_optimizers_converged: bool,
    folds_with_residual_focus_gain: int,
    fold_count: int,
    oof_metrics: Mapping[str, Mapping[str, object]],
    val_metrics: Mapping[str, Mapping[str, object]],
    oof_incremental_transitions: Mapping[str, int],
    val_incremental_transitions: Mapping[str, int],
    val_keeper_transitions: Mapping[str, int],
    oof_incremental_direction: Mapping[str, float],
    val_incremental_direction: Mapping[str, float],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    oof_control = oof_metrics["keeper_control_residual"]
    oof_candidate = oof_metrics["keeper_candidate_residual"]
    val_keeper = val_metrics["keeper"]
    val_control = val_metrics["keeper_control_residual"]
    val_candidate = val_metrics["keeper_candidate_residual"]
    oof_control_standalone = oof_metrics["control_standalone"]
    oof_candidate_standalone = oof_metrics["candidate_standalone"]
    val_control_standalone = val_metrics["control_standalone"]
    val_candidate_standalone = val_metrics["candidate_standalone"]
    raw_oof_direction_auc = oof_incremental_direction.get("auc_fn_positive")
    raw_val_direction_auc = val_incremental_direction.get("auc_fn_positive")
    directions_available = bool(
        raw_oof_direction_auc is not None
        and raw_val_direction_auc is not None
        and np.isfinite(float(raw_oof_direction_auc))
        and np.isfinite(float(raw_val_direction_auc))
    )
    oof_direction_auc = float(raw_oof_direction_auc) if directions_available else -1.0
    val_direction_auc = float(raw_val_direction_auc) if directions_available else -1.0
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "fold_source_overlap": int(fold_source_overlap),
        "train_val_source_overlap": int(train_val_source_overlap),
        "descriptors_finite": bool(descriptors_finite),
        "morphology_effective_rank": float(morphology_effective_rank),
        "mean_highlight_fraction": float(mean_highlight_fraction),
        "all_optimizers_converged": bool(all_optimizers_converged),
        "folds_with_residual_focus_gain": int(folds_with_residual_focus_gain),
        "fold_count": int(fold_count),
        "oof_standalone_macro_gain": float(oof_candidate_standalone["macro_f1"] - oof_control_standalone["macro_f1"]),
        "oof_standalone_focus_gain": float(_focus(oof_candidate_standalone, focus)["f1"] - _focus(oof_control_standalone, focus)["f1"]),
        "val_standalone_macro_gain": float(val_candidate_standalone["macro_f1"] - val_control_standalone["macro_f1"]),
        "val_standalone_focus_gain": float(_focus(val_candidate_standalone, focus)["f1"] - _focus(val_control_standalone, focus)["f1"]),
        "oof_incremental_macro_gain": float(oof_candidate["macro_f1"] - oof_control["macro_f1"]),
        "oof_incremental_focus_gain": float(_focus(oof_candidate, focus)["f1"] - _focus(oof_control, focus)["f1"]),
        "val_incremental_macro_gain": float(val_candidate["macro_f1"] - val_control["macro_f1"]),
        "val_incremental_focus_gain": float(_focus(val_candidate, focus)["f1"] - _focus(val_control, focus)["f1"]),
        "keeper_macro_f1": float(val_keeper["macro_f1"]),
        "keeper_focus_f1": float(_focus(val_keeper, focus)["f1"]),
        "keeper_focus_recall": float(_focus(val_keeper, focus)["recall"]),
        "candidate_macro_f1": float(val_candidate["macro_f1"]),
        "candidate_focus_f1": float(_focus(val_candidate, focus)["f1"]),
        "candidate_focus_precision": float(_focus(val_candidate, focus)["precision"]),
        "candidate_focus_recall": float(_focus(val_candidate, focus)["recall"]),
        "val_keeper_macro_gain": float(val_candidate["macro_f1"] - val_keeper["macro_f1"]),
        "val_keeper_focus_gain": float(_focus(val_candidate, focus)["f1"] - _focus(val_keeper, focus)["f1"]),
        "oof_incremental_transitions": dict(oof_incremental_transitions),
        "val_incremental_transitions": dict(val_incremental_transitions),
        "val_keeper_transitions": dict(val_keeper_transitions),
        "incremental_directions_available": directions_available,
        "oof_incremental_direction_auc": oof_direction_auc,
        "val_incremental_direction_auc": val_direction_auc,
        "direction_auc_gap": float(abs(oof_direction_auc - val_direction_auc)),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "source_group_folds_disjoint": observed["fold_source_overlap"] == 0,
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "descriptors_finite": bool(descriptors_finite),
        "highlight_exclusion_fraction_locked": 0.03 <= observed["mean_highlight_fraction"] <= 0.08,
        "morphology_effective_rank_ge_5": observed["morphology_effective_rank"] >= 5.0,
        "all_offset_optimizers_converged": bool(all_optimizers_converged),
        "residual_focus_gain_in_at_least_3_folds": observed["folds_with_residual_focus_gain"] >= min(3, observed["fold_count"]),
        "oof_standalone_macro_nonnegative": observed["oof_standalone_macro_gain"] >= 0.0,
        "oof_standalone_focus_gain_ge_0p005": observed["oof_standalone_focus_gain"] >= 0.005,
        "val_standalone_macro_nonnegative": observed["val_standalone_macro_gain"] >= 0.0,
        "val_standalone_focus_gain_ge_0p01": observed["val_standalone_focus_gain"] >= 0.01,
        "oof_incremental_macro_nonnegative": observed["oof_incremental_macro_gain"] >= 0.0,
        "oof_incremental_focus_gain_ge_0p005": observed["oof_incremental_focus_gain"] >= 0.005,
        "val_incremental_macro_nonnegative": observed["val_incremental_macro_gain"] >= 0.0,
        "val_incremental_focus_gain_ge_0p005": observed["val_incremental_focus_gain"] >= 0.005,
        "oof_incremental_corrections_ge_harms": int(oof_incremental_transitions["corrections"]) >= int(oof_incremental_transitions["harms"]),
        "oof_incremental_fp_removed_ge_created": int(oof_incremental_transitions["focus_false_positive_removed"]) >= int(oof_incremental_transitions["focus_false_positive_created"]),
        "oof_incremental_fn_rescued_ge_tp_broken": int(oof_incremental_transitions["focus_false_negative_rescued"]) >= int(oof_incremental_transitions["focus_true_positive_broken"]),
        "val_incremental_corrections_ge_harms": int(val_incremental_transitions["corrections"]) >= int(val_incremental_transitions["harms"]),
        "val_incremental_fp_removed_ge_created": int(val_incremental_transitions["focus_false_positive_removed"]) >= int(val_incremental_transitions["focus_false_positive_created"]),
        "val_incremental_fn_rescued_ge_tp_broken": int(val_incremental_transitions["focus_false_negative_rescued"]) >= int(val_incremental_transitions["focus_true_positive_broken"]),
        "incremental_directions_available": directions_available,
        "oof_incremental_direction_auc_ge_0p60": observed["oof_incremental_direction_auc"] >= 0.60,
        "val_incremental_direction_auc_ge_0p60": observed["val_incremental_direction_auc"] >= 0.60,
        "direction_auc_gap_le_0p15": observed["direction_auc_gap"] <= 0.15,
        "val_macro_preserves_keeper_within_0p001": observed["val_keeper_macro_gain"] >= -0.001,
        "val_focus_reaches_0p70": observed["candidate_focus_f1"] >= FOCUS_MILESTONE,
        "val_focus_improves_keeper_by_0p01": observed["val_keeper_focus_gain"] >= 0.01,
        "val_focus_recall_preserved_within_0p01": observed["candidate_focus_recall"] >= observed["keeper_focus_recall"] - 0.01,
        "val_keeper_corrections_ge_harms": int(val_keeper_transitions["corrections"]) >= int(val_keeper_transitions["harms"]),
        "val_keeper_fp_removed_ge_created": int(val_keeper_transitions["focus_false_positive_removed"]) >= int(val_keeper_transitions["focus_false_positive_created"]),
        "val_keeper_fn_rescued_ge_tp_broken": int(val_keeper_transitions["focus_false_negative_rescued"]) >= int(val_keeper_transitions["focus_true_positive_broken"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "image_smoke_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _normalize_map(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    high = float(np.quantile(finite, 0.98))
    return np.clip(array / max(high, 1e-6), 0.0, 1.0)


def _heatmap(values: np.ndarray) -> Image.Image:
    normalized = _normalize_map(values)
    rgb = np.stack(
        (
            normalized,
            np.sqrt(normalized) * 0.72,
            (1.0 - normalized) * 0.20,
        ),
        axis=-1,
    )
    return Image.fromarray((rgb.clip(0.0, 1.0) * 255.0).round().astype(np.uint8))


def _write_preview(previews: Sequence[Mapping[str, object]], output_dir: Path) -> Dict[str, object]:
    if not previews:
        return {"rows": 0, "path": ""}
    tile = 150
    label_height = 46
    columns = 5
    canvas = Image.new("RGB", (columns * tile, len(previews) * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    json_rows = []
    for row_index, item in enumerate(previews):
        y = row_index * (tile + label_height)
        rgb = Image.fromarray(np.asarray(item["rgb"], dtype=np.uint8))
        canvas.paste(rgb.resize((tile, tile), Image.Resampling.BILINEAR), (0, y))
        maps = item["maps"]
        highlight_overlay = np.asarray(item["rgb"], dtype=np.float32).copy()
        mask = np.asarray(maps["highlight"], dtype=bool)
        highlight_overlay[mask] = 0.45 * highlight_overlay[mask] + 0.55 * np.asarray((255.0, 40.0, 40.0), dtype=np.float32)
        canvas.paste(Image.fromarray(highlight_overlay.clip(0, 255).astype(np.uint8)).resize((tile, tile)), (tile, y))
        for column, key in enumerate(("dark_blob", "bright_blob", "chroma_blob"), start=2):
            canvas.paste(_heatmap(np.asarray(maps[key])).resize((tile, tile), Image.Resampling.BILINEAR), (column * tile, y))
        telemetry = item["telemetry"]
        draw.text(
            (4, y + tile + 3),
            f"{item['kind']} row={item['row_index']} target={item['target']} pred={item['keeper_prediction']} source={item['source_stem']}"[:120],
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (4, y + tile + 20),
            f"RGB | highlight-proxy={telemetry['highlight_fraction']:.3f} | dark LoG | bright LoG | chroma LoG",
            fill=(0, 0, 0),
            font=font,
        )
        json_rows.append({key: value for key, value in item.items() if key not in {"rgb", "maps"}})
    image_path = output_dir / "surface_blob_morphology_preview.png"
    json_path = output_dir / "surface_blob_morphology_preview.json"
    canvas.save(image_path)
    json_path.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    return {"rows": int(len(previews)), "path": str(image_path), "json": str(json_path)}


def _prediction_rows(
    *,
    split: str,
    cache: Mapping[str, np.ndarray],
    fold_assignment: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
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
        }
        for name, values in probabilities.items():
            row[f"{name}_prediction_index"] = int(values[index].argmax())
            for class_index in range(int(values.shape[1])):
                row[f"{name}_prob_{class_index}"] = float(values[index, class_index])
        rows.append(row)
    return rows


def _write_artifact_manifest(
    output_dir: Path,
    *,
    mode: str = "surface_blob_morphology_readiness_evidence_manifest",
) -> Dict[str, object]:
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
        "mode": str(mode),
        "payload_count": int(len(files)),
        "payload_size_bytes": int(sum(int(row["size_bytes"]) for row in files)),
        "payload_manifest_sha256": aggregate.hexdigest(),
        "files": files,
        "contains_checkpoint": any(str(row["name"]).lower().endswith(".pt") for row in files),
        "contains_model_binary": any(str(row["name"]).lower().endswith((".pt", ".pth", ".engine")) for row in files),
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
    extracted_train = _extract_split(
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
    extracted_val = _extract_split(
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
    x_train_control = np.asarray(extracted_train["control"], dtype=np.float64)
    x_val_control = np.asarray(extracted_val["control"], dtype=np.float64)
    x_train_morphology = np.asarray(extracted_train["morphology"], dtype=np.float64)
    x_val_morphology = np.asarray(extracted_val["morphology"], dtype=np.float64)
    x_train_candidate = np.concatenate((x_train_control, x_train_morphology), axis=1)
    x_val_candidate = np.concatenate((x_val_control, x_val_morphology), axis=1)
    y_train = np.asarray(train["labels"], dtype=np.int64)
    y_val = np.asarray(val["labels"], dtype=np.int64)
    train_groups = np.asarray(train["source_stems"], dtype=object)
    val_groups = np.asarray(val["source_stems"], dtype=object)
    train_val_source_overlap = len(set(map(str, train_groups)).intersection(map(str, val_groups)))

    prediction_names = (
        "control_standalone",
        "candidate_standalone",
        "keeper_control_residual",
        "keeper_candidate_residual",
    )
    oof_predictions = {
        name: np.zeros_like(train["probabilities"], dtype=np.float32)
        for name in prediction_names
    }
    fold_assignment = np.full(len(y_train), -1, dtype=np.int64)
    fold_rows = []
    fold_protocols = []
    maximum_fold_overlap = 0
    all_optimizers_converged = True
    folds_with_focus_gain = 0
    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    for fold_index, (fit_indices, hold_indices) in enumerate(
        splitter.split(x_train_control, y_train, train_groups)
    ):
        fit_sources = set(map(str, train_groups[fit_indices]))
        hold_sources = set(map(str, train_groups[hold_indices]))
        overlap = len(fit_sources.intersection(hold_sources))
        maximum_fold_overlap = max(maximum_fold_overlap, overlap)
        control_mean, control_scale = _fit_scaler(x_train_control[fit_indices])
        candidate_mean, candidate_scale = _fit_scaler(x_train_candidate[fit_indices])
        fold_probabilities, fold_fits = _fit_four_readouts(
            fit_control=_apply_scaler(x_train_control[fit_indices], control_mean, control_scale),
            fit_candidate=_apply_scaler(x_train_candidate[fit_indices], candidate_mean, candidate_scale),
            fit_base=train["probabilities"][fit_indices],
            fit_labels=y_train[fit_indices],
            eval_control=_apply_scaler(x_train_control[hold_indices], control_mean, control_scale),
            eval_candidate=_apply_scaler(x_train_candidate[hold_indices], candidate_mean, candidate_scale),
            eval_base=train["probabilities"][hold_indices],
            c_value=float(args.residual_c),
            max_iterations=int(args.max_iterations),
        )
        for name in prediction_names:
            oof_predictions[name][hold_indices] = fold_probabilities[name]
        fold_assignment[hold_indices] = int(fold_index)
        control_metrics = _classification_metrics(y_train[hold_indices], fold_probabilities["keeper_control_residual"])
        candidate_metrics = _classification_metrics(y_train[hold_indices], fold_probabilities["keeper_candidate_residual"])
        focus_gain = float(_focus(candidate_metrics, int(args.focus_class_index))["f1"] - _focus(control_metrics, int(args.focus_class_index))["f1"])
        folds_with_focus_gain += int(focus_gain > 0.0)
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "source_overlap": int(overlap),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "control_focus_f1": float(_focus(control_metrics, int(args.focus_class_index))["f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "candidate_focus_f1": float(_focus(candidate_metrics, int(args.focus_class_index))["f1"]),
                "candidate_focus_gain": float(focus_gain),
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
                "fits": fold_fits,
            }
        )
        all_optimizers_converged = all_optimizers_converged and all(bool(value["success"]) for value in fold_fits.values())
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF assignment is incomplete")

    control_mean, control_scale = _fit_scaler(x_train_control)
    candidate_mean, candidate_scale = _fit_scaler(x_train_candidate)
    val_predictions, full_fits = _fit_four_readouts(
        fit_control=_apply_scaler(x_train_control, control_mean, control_scale),
        fit_candidate=_apply_scaler(x_train_candidate, candidate_mean, candidate_scale),
        fit_base=train["probabilities"],
        fit_labels=y_train,
        eval_control=_apply_scaler(x_val_control, control_mean, control_scale),
        eval_candidate=_apply_scaler(x_val_candidate, candidate_mean, candidate_scale),
        eval_base=val["probabilities"],
        c_value=float(args.residual_c),
        max_iterations=int(args.max_iterations),
    )
    all_optimizers_converged = all_optimizers_converged and all(bool(value["success"]) for value in full_fits.values())
    oof_metrics = {
        "keeper_in_sample_reference": _classification_metrics(y_train, train["probabilities"]),
        **{name: _classification_metrics(y_train, oof_predictions[name]) for name in prediction_names},
    }
    val_metrics = {
        "keeper": _classification_metrics(y_val, val["probabilities"]),
        **{name: _classification_metrics(y_val, val_predictions[name]) for name in prediction_names},
    }
    transitions = {
        "train_oof_candidate_vs_control": _transition_stats(
            y_train,
            oof_predictions["keeper_control_residual"],
            oof_predictions["keeper_candidate_residual"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_control": _transition_stats(
            y_val,
            val_predictions["keeper_control_residual"],
            val_predictions["keeper_candidate_residual"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_keeper": _transition_stats(
            y_val,
            val["probabilities"],
            val_predictions["keeper_candidate_residual"],
            focus_class_index=int(args.focus_class_index),
        ),
    }
    direction = {
        "train_oof_candidate_vs_control": _direction_auc(
            y_train,
            oof_predictions["keeper_control_residual"],
            oof_predictions["keeper_candidate_residual"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_control": _direction_auc(
            y_val,
            val_predictions["keeper_control_residual"],
            val_predictions["keeper_candidate_residual"],
            focus_class_index=int(args.focus_class_index),
        ),
    }
    morphology_mean, morphology_scale = _fit_scaler(x_train_morphology)
    morphology_rank = _effective_rank(_apply_scaler(x_train_morphology, morphology_mean, morphology_scale))
    descriptors_finite = bool(extracted_train["finite"] and extracted_val["finite"])
    mean_highlight_fraction = float(
        0.5 * extracted_train["telemetry"]["highlight_fraction"]["mean"]
        + 0.5 * extracted_val["telemetry"]["highlight_fraction"]["mean"]
    )
    gate = assess_surface_blob_readiness(
        train_rows=len(y_train),
        val_rows=len(y_val),
        fold_source_overlap=maximum_fold_overlap,
        train_val_source_overlap=train_val_source_overlap,
        descriptors_finite=descriptors_finite,
        morphology_effective_rank=morphology_rank,
        mean_highlight_fraction=mean_highlight_fraction,
        all_optimizers_converged=all_optimizers_converged,
        folds_with_residual_focus_gain=folds_with_focus_gain,
        fold_count=int(args.folds),
        oof_metrics=oof_metrics,
        val_metrics=val_metrics,
        oof_incremental_transitions=transitions["train_oof_candidate_vs_control"],
        val_incremental_transitions=transitions["val_candidate_vs_control"],
        val_keeper_transitions=transitions["val_candidate_vs_keeper"],
        oof_incremental_direction=direction["train_oof_candidate_vs_control"],
        val_incremental_direction=direction["val_candidate_vs_control"],
        focus_class_index=int(args.focus_class_index),
        test_split_used=False,
    )
    preview = _write_preview(extracted_val["previews"], output_dir)
    np.savez_compressed(
        output_dir / "surface_blob_descriptors_train_val.npz",
        train_control=x_train_control.astype(np.float32),
        train_morphology=x_train_morphology.astype(np.float32),
        train_labels=y_train,
        train_sample_index=train["sample_index"],
        train_paths=train["paths"],
        val_control=x_val_control.astype(np.float32),
        val_morphology=x_val_morphology.astype(np.float32),
        val_labels=y_val,
        val_sample_index=val["sample_index"],
        val_paths=val["paths"],
    )
    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(split="train_oof", cache=train, fold_assignment=fold_assignment, probabilities=oof_predictions),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            cache=val,
            fold_assignment=np.full(len(y_val), -1, dtype=np.int64),
            probabilities=val_predictions,
        ),
    )
    protocol = {
        "method": "surface_blob_morphology_achromatic_highlight_proxy_readiness",
        "scope": (
            "Fixed scale-normalized LoG morphology on the eroded runtime object ROI. "
            "The matched control uses the same diffuse color moments without morphology."
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
        "highlight_proxy": (
            "top quantile of HSV value * (1 - saturation); used only as an "
            "exclusion mask and not claimed as physical specular separation"
        ),
        "highlight_quantile": float(HIGHLIGHT_QUANTILE),
        "blob_sigmas": list(BLOB_SIGMAS),
        "background_sigma": float(BACKGROUND_SIGMA),
        "peak_mad_multiplier": float(PEAK_MAD_MULTIPLIER),
        "control_dim": int(CONTROL_DIM),
        "morphology_dim": int(MORPHOLOGY_DIM),
        "candidate_dim": int(CONTROL_DIM + MORPHOLOGY_DIM),
        "residual": "zero-init no-bias linear offset on fixed keeper log probabilities",
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
            "train_telemetry": extracted_train["telemetry"],
            "val_telemetry": extracted_val["telemetry"],
            "descriptors_finite": descriptors_finite,
            "morphology_effective_rank": float(morphology_rank),
            "preview": preview,
        },
        "fold_protocols": fold_protocols,
        "full_fit_protocol": full_fits,
        "metrics": {"train_oof": oof_metrics, "val": val_metrics},
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
    readme = [
        "# Surface-Blob Morphology Readiness",
        "",
        f"- Val keeper macro/class1: `{float(val_metrics['keeper']['macro_f1']):.6f}/{float(_focus(val_metrics['keeper'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val matched control residual: `{float(val_metrics['keeper_control_residual']['macro_f1']):.6f}/{float(_focus(val_metrics['keeper_control_residual'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val morphology residual: `{float(val_metrics['keeper_candidate_residual']['macro_f1']):.6f}/{float(_focus(val_metrics['keeper_candidate_residual'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val standalone control/morphology class1: `{float(_focus(val_metrics['control_standalone'], int(args.focus_class_index))['f1']):.6f}/{float(_focus(val_metrics['candidate_standalone'], int(args.focus_class_index))['f1']):.6f}`",
        f"- OOF/val incremental direction AUROC: `{float(gate['observed']['oof_incremental_direction_auc']):.6f}/{float(gate['observed']['val_incremental_direction_auc']):.6f}`",
        f"- Morphology effective rank: `{float(morphology_rank):.6f}`",
        f"- Image smoke permission: `{str(bool(gate['image_smoke_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "Decision: reject before image smoke when any fixed check fails. Do not sweep blob scales, thresholds, ROI erosion, highlight-proxy quantile, residual C, or fold seeds on this evidence.",
        "",
        "This diagnostic reads immutable train/validation crops and frozen keeper caches. It reads no test split and writes no model/checkpoint or training manifest.",
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

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.config import to_serializable
from trkh.core.utils import autocast_context, load_checkpoint
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.probe_context_gray_edge_color_constancy import _context_dataset
from trkh.tools.probe_embedding_prototypes import _collate_classification
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _transition_summary,
)


SEED = 20260711
CASTS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("clean", (1.00, 1.00, 1.00)),
    ("dim", (0.72, 0.72, 0.72)),
    ("bright", (1.28, 1.28, 1.28)),
    ("red_cast", (1.18, 0.94, 0.94)),
    ("green_cast", (0.94, 1.18, 0.94)),
    ("blue_cast", (0.94, 0.94, 1.18)),
    ("warm_cast", (1.14, 1.02, 0.88)),
    ("cool_cast", (0.88, 1.02, 1.14)),
)
LITERATURE = (
    "https://openaccess.thecvf.com/content/CVPR2021/html/Lo_CLCC_Contrastive_Learning_for_Color_Constancy_CVPR_2021_paper.html",
    "https://openaccess.thecvf.com/content_cvpr_2015/html/Cheng_Effective_Learning-Based_Illuminant_2015_CVPR_paper.html",
    "https://openaccess.thecvf.com/content_iccv_2015/html/Barron_Convolutional_Color_Constancy_ICCV_2015_paper.html",
    "https://openaccess.thecvf.com/content_cvpr_2017/html/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.html",
    "https://openaccess.thecvf.com/content_cvpr_workshops_2015/W03/html/Bianco_Color_Constancy_Using_2015_CVPR_paper.html",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only/source-grouped readiness audit for a learned source-context "
            "illuminant estimator and object-crop canonicalization. Test is forbidden."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-train-sources", type=int, default=0)
    parser.add_argument("--max-val-sources", type=int, default=0)
    parser.add_argument("--max-val-objects", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _path_key(value: object) -> str:
    return str(Path(str(value)).resolve()).replace("/", "\\").lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _UniqueSourceContextDataset(Dataset):
    def __init__(self, base_dataset: Dataset, *, max_sources: int = 0) -> None:
        samples = getattr(base_dataset, "samples", None)
        if not isinstance(samples, list):
            raise TypeError("Context illuminant audit requires a YOLO sample list")
        seen: set[str] = set()
        indices: List[int] = []
        for index, sample in enumerate(samples):
            image_path = getattr(sample, "image_path", None)
            if image_path is None:
                raise ValueError("YOLO sample is missing image_path")
            key = _path_key(image_path)
            if key in seen:
                continue
            seen.add(key)
            indices.append(index)
            if int(max_sources) > 0 and len(indices) >= int(max_sources):
                break
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = self.base_dataset.samples[self.indices[index]]
        image = self.base_dataset._load_rgb_image(sample.image_path)
        labels = torch.tensor([int(obj.label) for obj in sample.objects], dtype=torch.long)
        boxes = torch.tensor([obj.bbox for obj in sample.objects], dtype=torch.float32)
        target = {
            "labels": labels,
            "boxes": boxes,
            "augmentation_scale": torch.ones((1,), dtype=torch.float32),
        }
        transformed = self.base_dataset.transform(image, target=target)
        if not isinstance(transformed, (tuple, list)) or len(transformed) != 2:
            raise ValueError("Context transform must return (tensor, target)")
        tensor, transformed_target = transformed
        if not torch.is_tensor(tensor) or not isinstance(transformed_target, Mapping):
            raise ValueError("Invalid context transform output")
        transformed_boxes = transformed_target.get("boxes")
        if not torch.is_tensor(transformed_boxes):
            raise ValueError("Context transform did not preserve boxes")
        image_mask = transformed_target.get("image_mask")
        if not torch.is_tensor(image_mask):
            image_mask = torch.ones(tensor.shape[-2:], dtype=torch.bool)
        return (
            tensor,
            transformed_boxes.to(dtype=torch.float32),
            image_mask.to(dtype=torch.bool),
            str(sample.image_path),
        )


def _collate_source_context(batch):
    images, boxes, masks, paths = zip(*batch)
    image_shapes = {tuple(image.shape) for image in images}
    mask_shapes = {tuple(mask.shape) for mask in masks}
    if len(image_shapes) != 1 or len(mask_shapes) != 1:
        raise ValueError("Source-context audit expects fixed-size transformed tensors")
    return torch.stack(images), list(boxes), torch.stack(masks), list(paths)


def _context_mask(
    valid_mask: np.ndarray,
    boxes: np.ndarray,
    *,
    margin_ratio: float = 0.08,
) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool).copy()
    height, width = mask.shape
    for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        cx, cy, box_width, box_height = (float(value) for value in box)
        half_width = box_width * (0.5 + float(margin_ratio))
        half_height = box_height * (0.5 + float(margin_ratio))
        left = max(0, int(math.floor((cx - half_width) * width)))
        right = min(width, int(math.ceil((cx + half_width) * width)))
        top = max(0, int(math.floor((cy - half_height) * height)))
        bottom = min(height, int(math.ceil((cy + half_height) * height)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = False
    if int(mask.sum()) < 64:
        return np.asarray(valid_mask, dtype=bool).copy()
    return mask


def context_color_descriptor(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    boxes: np.ndarray,
    gain: Sequence[float],
    *,
    max_pixels: int = 4096,
) -> Tuple[np.ndarray, float]:
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("Context RGB must have shape (3, H, W)")
    gain_array = np.asarray(gain, dtype=np.float32).reshape(3, 1, 1)
    cast = np.clip(values * gain_array, 1.0 / 255.0, 1.0)
    clipped_fraction = float(((values * gain_array) >= 1.0).mean())
    mask = _context_mask(valid_mask, boxes)
    pixels = cast[:, mask]
    if pixels.shape[1] > int(max_pixels):
        indices = np.linspace(0, pixels.shape[1] - 1, num=int(max_pixels), dtype=np.int64)
        pixels = pixels[:, indices]

    percentiles = np.percentile(
        pixels,
        (1, 5, 10, 25, 50, 75, 90, 95, 99),
        axis=1,
    ).T.reshape(-1)
    channel_stats = np.concatenate((pixels.mean(axis=1), pixels.std(axis=1), percentiles))
    log_pixels = np.log(pixels + 1e-4)
    chroma = np.stack((log_pixels[0] - log_pixels[1], log_pixels[2] - log_pixels[1]))
    chroma_percentiles = np.percentile(
        chroma,
        (1, 5, 10, 25, 50, 75, 90, 95, 99),
        axis=1,
    ).T.reshape(-1)
    luminance = 0.2126 * pixels[0] + 0.7152 * pixels[1] + 0.0722 * pixels[2]
    maximum = pixels.max(axis=0)
    saturation = (maximum - pixels.min(axis=0)) / np.maximum(maximum, 1e-4)
    auxiliary = np.concatenate(
        (
            chroma.mean(axis=1),
            chroma.std(axis=1),
            np.percentile(luminance, (1, 10, 25, 50, 75, 90, 99)),
            np.percentile(saturation, (10, 25, 50, 75, 90)),
        )
    )
    descriptor = np.concatenate((channel_stats, chroma_percentiles, auxiliary)).astype(
        np.float32,
        copy=False,
    )
    if descriptor.shape != (67,) or not np.isfinite(descriptor).all():
        raise ValueError(f"Invalid context descriptor: shape={descriptor.shape}")
    return descriptor, clipped_fraction


def apply_gain_and_correction(
    rgb: torch.Tensor,
    applied_gain: torch.Tensor,
    predicted_gain: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rgb.ndim != 4 or rgb.size(1) != 3:
        raise ValueError("RGB batch must have shape (B, 3, H, W)")
    applied = applied_gain.to(device=rgb.device, dtype=rgb.dtype).reshape(-1, 3, 1, 1)
    predicted = predicted_gain.to(device=rgb.device, dtype=rgb.dtype).reshape(-1, 3, 1, 1)
    cast = (rgb * applied).clamp(0.0, 1.0)
    corrected = (cast / predicted.clamp_min(1e-4)).clamp(0.0, 1.0)
    return cast, corrected


def _estimator() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=96,
        min_samples_leaf=3,
        max_features=0.75,
        n_jobs=-1,
        random_state=SEED,
    )


def _regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    angular_target = np.exp(targets)
    angular_prediction = np.exp(predictions)
    cosine = np.sum(angular_target * angular_prediction, axis=1) / np.maximum(
        np.linalg.norm(angular_target, axis=1)
        * np.linalg.norm(angular_prediction, axis=1),
        1e-12,
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "r2_per_channel": r2_score(targets, predictions, multioutput="raw_values").tolist(),
        "log_gain_mae_per_channel": np.abs(targets - predictions).mean(axis=0).tolist(),
        "log_gain_mae": float(np.abs(targets - predictions).mean()),
        "angular_error_mean_degrees": float(angles.mean()),
        "angular_error_p95_degrees": float(np.percentile(angles, 95)),
    }


def _extract_context_descriptors(
    *,
    dataset: Dataset,
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int,
    workers: int,
    split: str,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        collate_fn=_collate_source_context,
        persistent_workers=bool(int(workers) > 0),
    )
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    groups: List[str] = []
    source_paths: List[str] = []
    cast_indices: List[int] = []
    clipped: List[float] = []
    iterator = tqdm(loader, desc=f"context-illuminant-{split}", dynamic_ncols=True)
    for images, boxes, masks, paths in iterator:
        rgb = (images.numpy() * std_array + mean_array).clip(0.0, 1.0)
        mask_array = masks.numpy().astype(bool)
        for row_index, path in enumerate(paths):
            key = _path_key(path)
            box_array = boxes[row_index].numpy()
            for cast_index, (_, gain) in enumerate(CASTS):
                descriptor, clipped_fraction = context_color_descriptor(
                    rgb[row_index],
                    mask_array[row_index],
                    box_array,
                    gain,
                )
                features.append(descriptor)
                targets.append(np.log(np.asarray(gain, dtype=np.float32)))
                groups.append(key)
                source_paths.append(key)
                cast_indices.append(cast_index)
                clipped.append(clipped_fraction)
    return {
        "features": np.stack(features).astype(np.float32, copy=False),
        "targets": np.stack(targets).astype(np.float32, copy=False),
        "groups": np.asarray(groups, dtype=object),
        "source_paths": np.asarray(source_paths, dtype=object),
        "cast_indices": np.asarray(cast_indices, dtype=np.int64),
        "clipped_fraction": np.asarray(clipped, dtype=np.float32),
        "source_count": int(len(dataset)),
    }


def _fit_estimator(
    train: Mapping[str, object],
    val: Mapping[str, object],
    *,
    folds: int,
) -> Dict[str, object]:
    train_features = np.asarray(train["features"], dtype=np.float32)
    train_targets = np.asarray(train["targets"], dtype=np.float32)
    train_groups = np.asarray(train["groups"], dtype=object)
    val_features = np.asarray(val["features"], dtype=np.float32)
    val_targets = np.asarray(val["targets"], dtype=np.float32)
    splitter = GroupKFold(n_splits=int(folds))
    oof = np.zeros_like(train_targets)
    fold_rows: List[Dict[str, int]] = []
    template = _estimator()
    for fold_index, (fit_indices, holdout_indices) in enumerate(
        splitter.split(train_features, groups=train_groups)
    ):
        estimator = clone(template)
        estimator.fit(train_features[fit_indices], train_targets[fit_indices])
        oof[holdout_indices] = estimator.predict(train_features[holdout_indices])
        fold_rows.append(
            {
                "fold": int(fold_index + 1),
                "fit_rows": int(len(fit_indices)),
                "holdout_rows": int(len(holdout_indices)),
            }
        )
        print(f"illuminant estimator fold {fold_index + 1}/{folds} complete", flush=True)
    final_estimator = clone(template)
    final_estimator.fit(train_features, train_targets)
    val_predictions = final_estimator.predict(val_features).astype(np.float32, copy=False)
    return {
        "oof_predictions": oof,
        "val_predictions": val_predictions,
        "oof_metrics": _regression_metrics(train_targets, oof),
        "val_metrics": _regression_metrics(val_targets, val_predictions),
        "folds": fold_rows,
        "estimator": final_estimator,
    }


def _gain_lookup(
    val_context: Mapping[str, object],
    predictions: np.ndarray,
) -> Dict[Tuple[str, int], np.ndarray]:
    paths = np.asarray(val_context["source_paths"], dtype=object)
    cast_indices = np.asarray(val_context["cast_indices"], dtype=np.int64)
    output: Dict[Tuple[str, int], np.ndarray] = {}
    for path, cast_index, log_gain in zip(paths, cast_indices, predictions):
        output[(str(path), int(cast_index))] = np.exp(log_gain).astype(np.float32)
    return output


def _class1_error_counts(metrics: Mapping[str, object]) -> Tuple[int, int]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    false_positives = int(matrix[:, 1].sum() - matrix[1, 1])
    false_negatives = int(matrix[1, :].sum() - matrix[1, 1])
    return false_positives, false_negatives


def assess_context_illuminant_readiness(
    *,
    train_sources: int,
    val_sources: int,
    val_objects: int,
    oof_regression: Mapping[str, object],
    val_regression: Mapping[str, object],
    conditions: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    thresholds = {
        "required_train_sources": 8064,
        "required_val_sources": 2577,
        "required_val_objects": 2606,
        "min_regression_r2": 0.95,
        "max_log_gain_mae": 0.015,
        "max_clean_macro_drop": 0.002,
        "max_clean_class1_drop": 0.005,
        "min_mean_macro_recovery": 0.80,
        "min_mean_class1_recovery": 0.70,
        "max_corrected_macro_gap_from_clean": 0.010,
        "max_corrected_class1_gap_from_clean": 0.015,
        "max_class1_fp_over_clean": 8,
        "max_class1_fn_over_clean": 5,
    }
    clean = conditions["clean"]
    clean_raw = clean["raw_metrics"]
    clean_corrected = clean["corrected_metrics"]
    nonneutral = [value for key, value in conditions.items() if key != "clean"]

    macro_recovery: List[float] = []
    class1_recovery: List[float] = []
    corrected_macro: List[float] = []
    corrected_class1: List[float] = []
    all_conditions_improve = True
    class1_error_budget = True
    clean_fp, clean_fn = _class1_error_counts(clean_raw)
    for condition in nonneutral:
        raw = condition["raw_metrics"]
        corrected = condition["corrected_metrics"]
        macro_denominator = float(clean_raw["macro_f1"]) - float(raw["macro_f1"])
        class1_denominator = float(clean_raw["focus_f1"]) - float(raw["focus_f1"])
        if macro_denominator > 0.01:
            macro_recovery.append(
                (float(corrected["macro_f1"]) - float(raw["macro_f1"]))
                / macro_denominator
            )
        if class1_denominator > 0.01:
            class1_recovery.append(
                (float(corrected["focus_f1"]) - float(raw["focus_f1"]))
                / class1_denominator
            )
        corrected_macro.append(float(corrected["macro_f1"]))
        corrected_class1.append(float(corrected["focus_f1"]))
        all_conditions_improve = all_conditions_improve and (
            float(corrected["macro_f1"]) >= float(raw["macro_f1"])
            and float(corrected["focus_f1"]) >= float(raw["focus_f1"]) - 0.005
            and int(condition["raw_to_corrected"]["corrections"])
            >= int(condition["raw_to_corrected"]["harms"])
        )
        corrected_fp, corrected_fn = _class1_error_counts(corrected)
        class1_error_budget = class1_error_budget and (
            corrected_fp <= clean_fp + int(thresholds["max_class1_fp_over_clean"])
            and corrected_fn <= clean_fn + int(thresholds["max_class1_fn_over_clean"])
        )

    observed = {
        "min_oof_r2": float(min(oof_regression["r2_per_channel"])),
        "min_val_r2": float(min(val_regression["r2_per_channel"])),
        "max_oof_log_gain_mae": float(max(oof_regression["log_gain_mae_per_channel"])),
        "max_val_log_gain_mae": float(max(val_regression["log_gain_mae_per_channel"])),
        "clean_macro_drop": float(clean_raw["macro_f1"])
        - float(clean_corrected["macro_f1"]),
        "clean_class1_drop": float(clean_raw["focus_f1"])
        - float(clean_corrected["focus_f1"]),
        "mean_macro_recovery": float(np.mean(macro_recovery)) if macro_recovery else 0.0,
        "mean_class1_recovery": float(np.mean(class1_recovery)) if class1_recovery else 0.0,
        "mean_corrected_macro_f1": float(np.mean(corrected_macro)),
        "mean_corrected_class1_f1": float(np.mean(corrected_class1)),
        "clean_class1_false_positives": clean_fp,
        "clean_class1_false_negatives": clean_fn,
    }
    checks = {
        "full_train_source_support": int(train_sources)
        == int(thresholds["required_train_sources"]),
        "full_val_source_support": int(val_sources)
        == int(thresholds["required_val_sources"]),
        "full_val_object_support": int(val_objects)
        == int(thresholds["required_val_objects"]),
        "oof_regression_transfer": observed["min_oof_r2"]
        >= float(thresholds["min_regression_r2"])
        and observed["max_oof_log_gain_mae"] <= float(thresholds["max_log_gain_mae"]),
        "val_regression_transfer": observed["min_val_r2"]
        >= float(thresholds["min_regression_r2"])
        and observed["max_val_log_gain_mae"] <= float(thresholds["max_log_gain_mae"]),
        "clean_macro_preserved": observed["clean_macro_drop"]
        <= float(thresholds["max_clean_macro_drop"]),
        "clean_class1_preserved": observed["clean_class1_drop"]
        <= float(thresholds["max_clean_class1_drop"]),
        "synthetic_conditions_improve": bool(all_conditions_improve),
        "mean_macro_recovery": observed["mean_macro_recovery"]
        >= float(thresholds["min_mean_macro_recovery"]),
        "mean_class1_recovery": observed["mean_class1_recovery"]
        >= float(thresholds["min_mean_class1_recovery"]),
        "corrected_macro_near_clean": observed["mean_corrected_macro_f1"]
        >= float(clean_raw["macro_f1"])
        - float(thresholds["max_corrected_macro_gap_from_clean"]),
        "corrected_class1_near_clean": observed["mean_corrected_class1_f1"]
        >= float(clean_raw["focus_f1"])
        - float(thresholds["max_corrected_class1_gap_from_clean"]),
        "class1_error_budget": bool(class1_error_budget),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "context_illuminant_target_ready": not failed,
        "smoke_ready": not failed,
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _evaluate_object_conditions(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    gain_lookup: Mapping[Tuple[str, int], np.ndarray],
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        collate_fn=_collate_classification,
        persistent_workers=bool(int(workers) > 0),
    )
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in dataset_paths_fn()] if callable(dataset_paths_fn) else []
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    probabilities: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"raw": [], "corrected": []} for name, _ in CASTS
    }
    labels_all: List[np.ndarray] = []
    paths_all: List[str] = []
    pixel_errors: Dict[str, Dict[str, List[float]]] = {
        name: {"raw": [], "corrected": []} for name, _ in CASTS
    }
    preview: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    seen = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc="context-illuminant-classifier-val", dynamic_ncols=True)
        for images, labels, metadata in iterator:
            batch_count = int(images.size(0))
            batch_paths: List[str] = []
            if isinstance(metadata, Mapping) and isinstance(metadata.get("paths"), Sequence):
                batch_paths = [str(path) for path in metadata["paths"]]
            if len(batch_paths) != batch_count or not any(path.strip() for path in batch_paths):
                batch_paths = dataset_paths[seen : seen + batch_count]
            seen += batch_count
            labels_all.append(labels.numpy())
            paths_all.extend(batch_paths)
            clean_rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            for cast_index, (cast_name, gain) in enumerate(CASTS):
                predicted = np.stack(
                    [gain_lookup[(_path_key(path), cast_index)] for path in batch_paths]
                )
                applied_gain = torch.tensor(gain, dtype=torch.float32).view(1, 3).expand(batch_count, -1)
                predicted_gain = torch.from_numpy(predicted)
                raw_rgb, corrected_rgb = apply_gain_and_correction(
                    clean_rgb,
                    applied_gain,
                    predicted_gain,
                )
                pixel_errors[cast_name]["raw"].extend(
                    (raw_rgb - clean_rgb).abs().flatten(1).mean(dim=1).tolist()
                )
                pixel_errors[cast_name]["corrected"].extend(
                    (corrected_rgb - clean_rgb).abs().flatten(1).mean(dim=1).tolist()
                )
                if cast_name == "warm_cast":
                    for row_index, target in enumerate(labels.tolist()):
                        if int(target) in preview:
                            continue
                        arrays = []
                        for tensor in (clean_rgb[row_index], raw_rgb[row_index], corrected_rgb[row_index]):
                            arrays.append(
                                (tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0)
                                .round()
                                .astype(np.uint8)
                            )
                        preview[int(target)] = tuple(arrays)
                raw_input = ((raw_rgb - mean_tensor) / std_tensor).to(device=device)
                corrected_input = ((corrected_rgb - mean_tensor) / std_tensor).to(device=device)
                with autocast_context(device, bool(amp)):
                    raw_logits, _ = _forward_classification_with_metadata(
                        model,
                        raw_input,
                        metadata,
                        device=device,
                    )
                    corrected_logits, _ = _forward_classification_with_metadata(
                        model,
                        corrected_input,
                        metadata,
                        device=device,
                    )
                probabilities[cast_name]["raw"].append(
                    raw_logits.float().softmax(dim=1).cpu().numpy()
                )
                probabilities[cast_name]["corrected"].append(
                    corrected_logits.float().softmax(dim=1).cpu().numpy()
                )

    labels = np.concatenate(labels_all).astype(np.int64, copy=False)
    condition_results: Dict[str, object] = {}
    prediction_rows: List[Dict[str, object]] = []
    clean_raw_probabilities = np.concatenate(probabilities["clean"]["raw"])
    for cast_name, _ in CASTS:
        raw = np.concatenate(probabilities[cast_name]["raw"])
        corrected = np.concatenate(probabilities[cast_name]["corrected"])
        raw_metrics = _classification_metrics(labels, raw, class_names=class_names)
        corrected_metrics = _classification_metrics(labels, corrected, class_names=class_names)
        condition_results[cast_name] = {
            "raw_metrics": raw_metrics,
            "corrected_metrics": corrected_metrics,
            "raw_to_corrected": _transition_summary(labels, raw, corrected),
            "clean_to_corrected": _transition_summary(labels, clean_raw_probabilities, corrected),
            "raw_pixel_mae": float(np.mean(pixel_errors[cast_name]["raw"])),
            "corrected_pixel_mae": float(np.mean(pixel_errors[cast_name]["corrected"])),
        }
        raw_predictions = raw.argmax(axis=1)
        corrected_predictions = corrected.argmax(axis=1)
        for row_index, path in enumerate(paths_all):
            row: Dict[str, object] = {
                "path": path,
                "target": int(labels[row_index]),
                "cast": cast_name,
                "raw_prediction": int(raw_predictions[row_index]),
                "corrected_prediction": int(corrected_predictions[row_index]),
            }
            for class_index in range(len(class_names)):
                row[f"raw_probability_{class_index}"] = float(raw[row_index, class_index])
                row[f"corrected_probability_{class_index}"] = float(
                    corrected[row_index, class_index]
                )
            prediction_rows.append(row)
    return condition_results, prediction_rows, preview


def _write_preview(
    path: Path,
    preview: Mapping[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> None:
    tile = 192
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index in range(len(class_names)):
        if class_index not in preview:
            continue
        row = Image.new("RGB", (tile * 3, tile), color=(255, 255, 255))
        for column, array in enumerate(preview[class_index]):
            row.paste(Image.fromarray(array).resize((tile, tile), resampling.BILINEAR), (column * tile, 0))
        rows.append(row)
        manifest.append(
            {
                "row": len(rows) - 1,
                "class_index": class_index,
                "class_name": str(class_names[class_index]),
                "columns": ["clean_crop", "warm_cast", "estimated_correction"],
            }
        )
    canvas = Image.new("RGB", (tile * 3, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if int(args.folds) != 5:
        raise ValueError("Protocol is locked to five source-grouped folds")
    if int(args.batch_size) < 1 or int(args.workers) < 0:
        raise ValueError("Invalid batch/workers setting")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    mean, std = checkpoint_input_normalization(checkpoint)
    device = _resolve_device(args.device)

    started = time.time()
    train_base, class_names = _context_dataset(
        data=Path(args.data),
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    val_base, val_class_names = _context_dataset(
        data=Path(args.data),
        split="val",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=int(args.max_val_objects),
    )
    if list(class_names) != list(val_class_names):
        raise ValueError("Train/validation class names differ")
    train_sources = _UniqueSourceContextDataset(
        train_base,
        max_sources=int(args.max_train_sources),
    )
    val_sources = _UniqueSourceContextDataset(
        val_base,
        max_sources=int(args.max_val_sources),
    )
    train_context = _extract_context_descriptors(
        dataset=train_sources,
        mean=mean,
        std=std,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        split="train",
    )
    val_context = _extract_context_descriptors(
        dataset=val_sources,
        mean=mean,
        std=std,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        split="val",
    )
    estimator_result = _fit_estimator(train_context, val_context, folds=int(args.folds))
    gain_lookup = _gain_lookup(val_context, estimator_result["val_predictions"])

    train_base.classification_source_context_aux = False
    val_base.classification_source_context_aux = False
    model = build_model_from_checkpoint(checkpoint).to(device)
    condition_results, prediction_rows, preview = _evaluate_object_conditions(
        model=model,
        dataset=val_base,
        gain_lookup=gain_lookup,
        class_names=class_names,
        mean=mean,
        std=std,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        amp=bool(args.amp),
    )
    gate = assess_context_illuminant_readiness(
        train_sources=int(train_context["source_count"]),
        val_sources=int(val_context["source_count"]),
        val_objects=int(len(val_base)),
        oof_regression=estimator_result["oof_metrics"],
        val_regression=estimator_result["val_metrics"],
        conditions=condition_results,
    )

    fieldnames = list(prediction_rows[0])
    with (output_dir / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    _write_preview(output_dir / "warm_cast_correction_preview.png", preview, class_names)

    clean_cast_rows = np.asarray(val_context["cast_indices"]) == 0
    clean_gain_predictions = np.exp(
        np.asarray(estimator_result["val_predictions"])[clean_cast_rows]
    )
    summary = {
        "mode": "context_illuminant_estimator_readiness_precheck",
        "protocol": {
            "seed": SEED,
            "casts": [{"name": name, "gain": list(gain)} for name, gain in CASTS],
            "descriptor": "67D context-outside-all-bboxes RGB/log-chroma/luminance/saturation statistics",
            "estimator": "ExtraTreesRegressor(n=96,min_leaf=3,max_features=0.75)",
            "folds": 5,
            "grouping": "source image path; all casts of one source stay in one fold",
            "selection": "none; fixed protocol and fixed gate",
            "saved_estimator": False,
        },
        "guardrails": {
            "raw_dataset_touched": False,
            "test_split_used": False,
            "trainable_manifest_written": False,
            "checkpoint_created": False,
            "full_train_permission": False,
        },
        "inputs": {
            "data": str(Path(args.data).resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "class_names": list(class_names),
        },
        "support": {
            "train_sources": int(train_context["source_count"]),
            "val_sources": int(val_context["source_count"]),
            "val_objects": int(len(val_base)),
            "train_descriptor_rows": int(np.asarray(train_context["features"]).shape[0]),
            "val_descriptor_rows": int(np.asarray(val_context["features"]).shape[0]),
        },
        "regression": {
            "oof": estimator_result["oof_metrics"],
            "validation": estimator_result["val_metrics"],
            "folds": estimator_result["folds"],
            "mean_clipped_fraction_train": float(
                np.asarray(train_context["clipped_fraction"]).mean()
            ),
            "mean_clipped_fraction_val": float(
                np.asarray(val_context["clipped_fraction"]).mean()
            ),
            "clean_predicted_gain_mean": clean_gain_predictions.mean(axis=0).tolist(),
            "clean_predicted_gain_p05": np.percentile(clean_gain_predictions, 5, axis=0).tolist(),
            "clean_predicted_gain_p95": np.percentile(clean_gain_predictions, 95, axis=0).tolist(),
        },
        "conditions": condition_results,
        "gate": gate,
        "literature": list(LITERATURE),
        "elapsed_seconds": float(time.time() - started),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(to_serializable(summary), indent=2),
        encoding="utf-8",
    )
    decision = "PASS pre-smoke readiness" if gate["smoke_permission"] else "REJECT before GPU smoke"
    readme = [
        "# Context Illuminant Estimator Readiness",
        "",
        f"Decision: **{decision}**.",
        "",
        "This fixed diagnostic learns only synthetic diagonal illumination gains from",
        "full-frame context outside transformed object boxes. It applies estimated",
        "correction to the object crop and audits the frozen keeper on full validation.",
        "It does not use test, alter raw data, save an estimator, create a checkpoint,",
        "or grant full-train permission.",
        "",
        f"- Train/val sources: `{train_context['source_count']}/{val_context['source_count']}`",
        f"- Validation objects: `{len(val_base)}`",
        f"- Failed gates: `{gate['failed_checks']}`",
        f"- Clean raw macro/class1: `{condition_results['clean']['raw_metrics']['macro_f1']:.6f}/"
        f"{condition_results['clean']['raw_metrics']['focus_f1']:.6f}`",
        f"- Clean corrected macro/class1: `{condition_results['clean']['corrected_metrics']['macro_f1']:.6f}/"
        f"{condition_results['clean']['corrected_metrics']['focus_f1']:.6f}`",
        "",
        "See `summary.json`, `validation_predictions.csv`, and",
        "`warm_cast_correction_preview.png` for the complete audit.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(to_serializable({"output_dir": str(output_dir), "gate": gate}), indent=2))


if __name__ == "__main__":
    main()

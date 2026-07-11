from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import kymatio
import numpy as np
import torch
import torch.nn.functional as F
from kymatio.torch import Scattering2D
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from tqdm import tqdm

from trkh.tools.audit_multistage_teacher_feature_readiness import (
    _alignment_plan,
    _class1_error_direction_auc,
    _compact_alignment,
    _heat_overlay,
    _imagefolder_rows,
    _load_base_predictions,
    _projected_effective_rank,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _transition_summary,
    _write_prediction_audit,
)


LITERATURE = [
    "https://pubmed.ncbi.nlm.nih.gov/23787341/",
    "https://arxiv.org/abs/1203.1513",
    "https://openaccess.thecvf.com/content_cvpr_2013/html/Sifre_Rotation_Scaling_and_2013_CVPR_paper.html",
    "https://www.jmlr.org/papers/v21/19-047.html",
    "https://www.kymat.io/",
    "https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.html",
    "https://openaccess.thecvf.com/content/CVPR2022/papers/Gauthier_Parametric_Scattering_Networks_CVPR_2022_paper.pdf",
]

IMAGE_SIZE = 128
SCATTERING_J = 3
SCATTERING_L = 4
SCATTERING_MAX_ORDER = 2
SPATIAL_POOL_SIZE = 2
PCA_COMPONENTS = 128
PCA_WHITEN = True
LOGISTIC_C = 0.3
RBF_SVC_C = 3.0
SEED = 20260711
CHANNEL_NAMES = ("red", "green", "blue", "luminance", "red_green", "blue_yellow")
READOUT_NAMES = ("linear_logistic", "rbf_svc")
PRIMARY_READOUT = "rbf_svc"


class PathImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, int(target), str(self.samples[int(index)][0])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only audit of fixed wavelet-scattering crop features "
            "as a source-grouped complement to the wide-context TRKH keeper."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--base-train-csv", type=Path, required=True)
    parser.add_argument("--base-val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Defaults to zero for deterministic Windows/VS Code execution.",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--save-descriptors", action="store_true", default=False)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_output_outside_datasets(
    output_dir: Path,
    *,
    classification_root: Path,
    yolo_data: Path,
) -> None:
    output = Path(output_dir).resolve(strict=False)
    dataset_roots = (
        Path(classification_root).resolve(strict=False),
        Path(yolo_data).resolve(strict=False).parent,
    )
    for root in dataset_roots:
        if output == root or _is_relative_to(output, root):
            raise ValueError(f"Output directory must stay outside raw dataset root: {root}")


def build_scattering_channels(images: Tensor) -> Tensor:
    if images.ndim != 4 or int(images.size(1)) != 3:
        raise ValueError(f"Expected RGB tensor [B,3,H,W], got {tuple(images.shape)}")
    rgb = images.float().clamp(0.0, 1.0)
    red, green, blue = rgb.unbind(dim=1)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    red_green = 0.5 * (red - green + 1.0)
    blue_yellow = 0.5 * (blue - 0.5 * (red + green) + 1.0)
    opponent = torch.stack((luminance, red_green, blue_yellow), dim=1).clamp(0.0, 1.0)
    return torch.cat((rgb, opponent), dim=1)


def scattering_order_slices(
    coefficient_count: int,
    *,
    j: int = SCATTERING_J,
    l: int = SCATTERING_L,
) -> Dict[str, slice]:
    first_count = int(j) * int(l)
    second_count = int(l) * int(l) * int(j) * (int(j) - 1) // 2
    expected = 1 + first_count + second_count
    if int(coefficient_count) != expected:
        raise ValueError(
            f"Unexpected scattering coefficient count: {coefficient_count}; expected {expected}"
        )
    return {
        "zero": slice(0, 1),
        "first": slice(1, 1 + first_count),
        "second": slice(1 + first_count, expected),
    }


def scattering_descriptor(scattered: Tensor, *, spatial_pool_size: int = 2) -> Tensor:
    if scattered.ndim != 5:
        raise ValueError(
            "Expected scattering output [B,C,K,H,W], "
            f"got {tuple(scattered.shape)}"
        )
    pool_size = max(1, int(spatial_pool_size))
    values = scattered.float()
    mean = values.mean(dim=(-2, -1))
    std = values.std(dim=(-2, -1), unbiased=False)
    batch, channels, coefficients, height, width = values.shape
    pooled = F.adaptive_avg_pool2d(
        values.reshape(batch * channels * coefficients, 1, height, width),
        (pool_size, pool_size),
    ).reshape(batch, channels, coefficients, pool_size, pool_size)
    return torch.cat(
        (mean.flatten(1), std.flatten(1), pooled.flatten(1)),
        dim=1,
    )


def _stable_softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return (exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)).astype(
        np.float32
    )


def aligned_decision_probabilities(
    estimator,
    features: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    raw = np.asarray(estimator.decision_function(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if raw.ndim == 1:
        if classes.size != 2:
            raise ValueError("One-dimensional decision scores require two estimator classes")
        raw = np.stack((-raw, raw), axis=1)
    if raw.shape[1] != classes.size:
        raise ValueError(
            f"Decision/class column mismatch: scores={raw.shape[1]}, classes={classes.size}"
        )
    aligned = np.full((raw.shape[0], int(class_count)), -1e9, dtype=np.float64)
    for column, class_index in enumerate(classes):
        aligned[:, int(class_index)] = raw[:, int(column)]
    return _stable_softmax(aligned)


def _aligned_predict_proba(
    estimator,
    features: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    aligned = np.zeros((raw.shape[0], int(class_count)), dtype=np.float64)
    for column, class_index in enumerate(classes):
        aligned[:, int(class_index)] = raw[:, int(column)]
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned.astype(np.float32)


def _make_readouts(seed: int) -> Dict[str, object]:
    return {
        "linear_logistic": LogisticRegression(
            C=LOGISTIC_C,
            class_weight="balanced",
            max_iter=500,
            random_state=int(seed),
            solver="lbfgs",
        ),
        "rbf_svc": SVC(
            C=RBF_SVC_C,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
            decision_function_shape="ovr",
            cache_size=2048,
        ),
    }


def _readout_probabilities(
    name: str,
    estimator,
    features: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    if name == "linear_logistic":
        return _aligned_predict_proba(estimator, features, class_count=class_count)
    if name == "rbf_svc":
        return aligned_decision_probabilities(estimator, features, class_count=class_count)
    raise KeyError(name)


def _estimator_telemetry(estimator) -> Dict[str, object]:
    telemetry: Dict[str, object] = {}
    if hasattr(estimator, "n_iter_"):
        telemetry["iterations"] = [int(value) for value in np.asarray(estimator.n_iter_).ravel()]
    if hasattr(estimator, "n_support_"):
        telemetry["support_vectors_per_class"] = [
            int(value) for value in np.asarray(estimator.n_support_).ravel()
        ]
        telemetry["support_vectors_total"] = int(np.asarray(estimator.n_support_).sum())
    return telemetry


def fit_grouped_scattering_readouts(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    class_names: Sequence[str],
    folds: int,
    seed: int,
) -> Dict[str, object]:
    train_features = np.asarray(train_features, dtype=np.float32)
    val_features = np.asarray(val_features, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    class_count = len(class_names)
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    split_indices = list(
        splitter.split(np.zeros(train_labels.shape[0]), train_labels, groups=train_groups)
    )
    oof = {
        name: np.zeros((train_labels.size, class_count), dtype=np.float32)
        for name in READOUT_NAMES
    }
    fold_telemetry: List[Dict[str, object]] = []
    for fold_index, (fit_indices, holdout_indices) in enumerate(split_indices):
        fold_start = time.perf_counter()
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(train_features[fit_indices])
        holdout_scaled = scaler.transform(train_features[holdout_indices])
        component_count = min(
            PCA_COMPONENTS,
            int(fit_scaled.shape[0]) - 1,
            int(fit_scaled.shape[1]),
        )
        if component_count < PCA_COMPONENTS:
            raise ValueError(
                f"Fold {fold_index + 1} cannot support fixed PCA-{PCA_COMPONENTS}: "
                f"available={component_count}"
            )
        pca = PCA(
            n_components=PCA_COMPONENTS,
            whiten=PCA_WHITEN,
            svd_solver="randomized",
            random_state=int(seed) + int(fold_index),
        )
        fit_projected = pca.fit_transform(fit_scaled)
        holdout_projected = pca.transform(holdout_scaled)
        readout_telemetry: Dict[str, object] = {}
        for name, estimator in _make_readouts(int(seed) + int(fold_index)).items():
            estimator.fit(fit_projected, train_labels[fit_indices])
            oof[name][holdout_indices] = _readout_probabilities(
                name,
                estimator,
                holdout_projected,
                class_count=class_count,
            )
            readout_telemetry[name] = _estimator_telemetry(estimator)
        fold_telemetry.append(
            {
                "fold": int(fold_index + 1),
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "elapsed_seconds": float(time.perf_counter() - fold_start),
                "readouts": readout_telemetry,
            }
        )
        print(
            f"descriptor readout fold {fold_index + 1}/{len(split_indices)} complete; "
            f"fit={fit_indices.size} holdout={holdout_indices.size} "
            f"elapsed={fold_telemetry[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )

    final_start = time.perf_counter()
    final_scaler = StandardScaler()
    train_scaled = final_scaler.fit_transform(train_features)
    val_scaled = final_scaler.transform(val_features)
    final_pca = PCA(
        n_components=PCA_COMPONENTS,
        whiten=PCA_WHITEN,
        svd_solver="randomized",
        random_state=int(seed) + 1000,
    )
    train_projected = final_pca.fit_transform(train_scaled)
    val_projected = final_pca.transform(val_scaled)
    results: Dict[str, Dict[str, object]] = {}
    for name, estimator in _make_readouts(int(seed) + 1000).items():
        estimator.fit(train_projected, train_labels)
        val_probabilities = _readout_probabilities(
            name,
            estimator,
            val_projected,
            class_count=class_count,
        )
        results[name] = {
            "oof_probabilities": oof[name],
            "val_probabilities": val_probabilities,
            "oof_metrics": _classification_metrics(
                train_labels,
                oof[name],
                class_names=class_names,
            ),
            "val_metrics": _classification_metrics(
                val_labels,
                val_probabilities,
                class_names=class_names,
            ),
            "final_estimator": _estimator_telemetry(estimator),
        }
    return {
        "readouts": results,
        "folds": fold_telemetry,
        "final_pca_explained_variance": float(final_pca.explained_variance_ratio_.sum()),
        "final_fit_elapsed_seconds": float(time.perf_counter() - final_start),
    }


def _binary_focus_oracle(
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class: int = 1,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    target = labels == int(focus_class)
    base = np.asarray(base_probabilities).argmax(axis=1) == int(focus_class)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1) == int(focus_class)
    oracle = base.copy()
    oracle[(base != target) & (candidate == target)] = candidate[(base != target) & (candidate == target)]
    true_positive = int((target & oracle).sum())
    false_positive = int((~target & oracle).sum())
    false_negative = int((target & ~oracle).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def assess_wavelet_scattering_readiness(
    *,
    train_samples: int,
    val_samples: int,
    source_groups: int,
    descriptor_effective_rank: float,
    descriptor_finite: bool,
    direct_val_metrics: Mapping[str, object],
    candidate_oof_metrics: Mapping[str, object],
    candidate_val_metrics: Mapping[str, object],
    binary_oracle: Mapping[str, float],
    train_direction_auc: float,
    val_direction_auc: float,
    val_transitions: Mapping[str, int],
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_source_groups": 8000,
        "min_descriptor_effective_rank": 24.0,
        "min_oof_macro_f1": 0.82,
        "min_oof_class1_f1": 0.55,
        "min_val_macro_f1": 0.82,
        "min_val_class1_f1": 0.60,
        "max_oof_val_class1_gap": 0.08,
        "min_binary_oracle_class1_gain": 0.02,
        "min_direction_auc": 0.60,
        "max_direction_auc_gap": 0.15,
    }
    observed = {
        "descriptor_effective_rank": float(descriptor_effective_rank),
        "oof_macro_f1": float(candidate_oof_metrics["macro_f1"]),
        "oof_class1_f1": float(candidate_oof_metrics["focus_f1"]),
        "val_macro_f1": float(candidate_val_metrics["macro_f1"]),
        "val_class1_f1": float(candidate_val_metrics["focus_f1"]),
        "oof_val_class1_gap": abs(
            float(candidate_oof_metrics["focus_f1"])
            - float(candidate_val_metrics["focus_f1"])
        ),
        "binary_oracle_class1_gain": float(binary_oracle["f1"])
        - float(direct_val_metrics["focus_f1"]),
        "train_direction_auc": float(train_direction_auc),
        "val_direction_auc": float(val_direction_auc),
        "direction_auc_gap": abs(float(train_direction_auc) - float(val_direction_auc)),
    }
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "validation_support": int(val_samples) == int(thresholds["required_val_samples"]),
        "source_group_support": int(source_groups) >= int(thresholds["min_source_groups"]),
        "descriptor_finite": bool(descriptor_finite),
        "descriptor_rank": float(descriptor_effective_rank)
        >= float(thresholds["min_descriptor_effective_rank"]),
        "oof_macro_signal": observed["oof_macro_f1"]
        >= float(thresholds["min_oof_macro_f1"]),
        "oof_class1_signal": observed["oof_class1_f1"]
        >= float(thresholds["min_oof_class1_f1"]),
        "validation_macro_signal": observed["val_macro_f1"]
        >= float(thresholds["min_val_macro_f1"]),
        "validation_class1_signal": observed["val_class1_f1"]
        >= float(thresholds["min_val_class1_f1"]),
        "oof_validation_stability": observed["oof_val_class1_gap"]
        <= float(thresholds["max_oof_val_class1_gap"]),
        "binary_oracle_complementarity": observed["binary_oracle_class1_gain"]
        >= float(thresholds["min_binary_oracle_class1_gain"]),
        "train_error_direction": float(train_direction_auc)
        >= float(thresholds["min_direction_auc"]),
        "validation_error_direction": float(val_direction_auc)
        >= float(thresholds["min_direction_auc"]),
        "error_direction_stability": observed["direction_auc_gap"]
        <= float(thresholds["max_direction_auc_gap"]),
        "validation_net_corrections": int(val_transitions["corrections"])
        >= int(val_transitions["harms"]),
        "validation_class1_recall_protected": int(val_transitions["class1_fn_rescued"])
        >= int(val_transitions["class1_tp_broken"]),
        "validation_class1_fp_control": int(val_transitions["class1_fp_removed"])
        >= int(val_transitions["class1_fp_created"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "wavelet_scattering_ready": not failed,
        "smoke_ready": not failed,
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _extract_split(
    *,
    scattering: Scattering2D,
    classification_root: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    transform = v2.Compose(
        (
            v2.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        )
    )
    dataset = PathImageFolder(Path(classification_root) / str(split), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(workers) > 0),
    )
    order_slices: Optional[Dict[str, slice]] = None
    descriptors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    paths: List[str] = []
    order_absolute_sum = {"zero": 0.0, "first": 0.0, "second": 0.0}
    order_element_count = {"zero": 0, "first": 0, "second": 0}
    preview: Dict[int, Dict[str, object]] = {}
    with torch.inference_mode():
        for images, targets, batch_paths in tqdm(loader, desc=f"wavelet-{split}"):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            channels = build_scattering_channels(images)
            scattered = scattering(channels)
            if scattered.ndim != 5:
                raise ValueError(f"Unexpected Kymatio output shape: {tuple(scattered.shape)}")
            if order_slices is None:
                order_slices = scattering_order_slices(int(scattered.size(2)))
            descriptor = scattering_descriptor(scattered)
            descriptors.append(descriptor.cpu().numpy().astype(np.float32, copy=False))
            labels.append(targets.numpy().astype(np.int64, copy=False))
            paths.extend(str(path) for path in batch_paths)
            for name, coefficient_slice in order_slices.items():
                values = scattered[:, :, coefficient_slice]
                order_absolute_sum[name] += float(values.abs().sum().item())
                order_element_count[name] += int(values.numel())
            for row_index, target in enumerate(targets.tolist()):
                class_index = int(target)
                if class_index in preview:
                    continue
                rgb = (
                    images[row_index]
                    .detach()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                    .clip(0.0, 1.0)
                )
                preview[class_index] = {
                    "rgb": (rgb * 255.0).round().astype(np.uint8),
                    "path": str(batch_paths[row_index]),
                    "first": scattered[row_index, :, order_slices["first"]]
                    .square()
                    .mean(dim=(0, 1))
                    .detach()
                    .cpu()
                    .numpy(),
                    "second": scattered[row_index, :, order_slices["second"]]
                    .square()
                    .mean(dim=(0, 1))
                    .detach()
                    .cpu()
                    .numpy(),
                }
    if order_slices is None:
        raise ValueError(f"Empty ImageFolder split: {classification_root / split}")
    feature_array = np.concatenate(descriptors, axis=0).astype(np.float32, copy=False)
    label_array = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    total_absolute_sum = max(sum(order_absolute_sum.values()), 1e-12)
    order_stats = {
        name: {
            "coefficient_count": int(order_slices[name].stop - order_slices[name].start),
            "mean_absolute": float(
                order_absolute_sum[name] / max(order_element_count[name], 1)
            ),
            "absolute_mass_ratio": float(order_absolute_sum[name] / total_absolute_sum),
        }
        for name in ("zero", "first", "second")
    }
    return {
        "descriptors": feature_array,
        "labels": label_array,
        "paths": np.asarray(paths, dtype=object),
        "folder_classes": list(dataset.classes),
        "order_stats": order_stats,
        "preview": preview,
    }


def _align_payload(payload: Mapping[str, object], plan: Mapping[str, object]) -> Dict[str, object]:
    indices = np.asarray(plan["row_indices"], dtype=np.int64)
    return {
        "descriptors": np.asarray(payload["descriptors"], dtype=np.float32)[indices],
        "labels": np.asarray(plan["labels"], dtype=np.int64),
        "sample_index": np.asarray(plan["sample_index"], dtype=np.int64),
        "paths": np.asarray(plan["paths"], dtype=object),
        "source_stems": np.asarray(plan["source_stems"], dtype=object),
        "class_names": list(plan["class_names"]),
        "folder_classes": list(payload["folder_classes"]),
        "order_stats": payload["order_stats"],
        "preview": payload["preview"],
    }


def _write_preview(
    path: Path,
    *,
    preview: Mapping[int, Mapping[str, object]],
    folder_classes: Sequence[str],
) -> None:
    tile = 176
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index, class_name in enumerate(folder_classes):
        if int(class_index) not in preview:
            continue
        item = preview[int(class_index)]
        rgb = np.asarray(item["rgb"], dtype=np.uint8)
        row = Image.new("RGB", (tile * 3, tile), color=(255, 255, 255))
        row.paste(Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR), (0, 0))
        first = _heat_overlay(rgb, np.asarray(item["first"], dtype=np.float32))
        second = _heat_overlay(rgb, np.asarray(item["second"], dtype=np.float32))
        row.paste(Image.fromarray(first).resize((tile, tile), resampling.BILINEAR), (tile, 0))
        row.paste(
            Image.fromarray(second).resize((tile, tile), resampling.BILINEAR),
            (tile * 2, 0),
        )
        rows.append(row)
        manifest.append(
            {
                "row": int(len(rows) - 1),
                "class_index": int(class_index),
                "class_name": str(class_name),
                "path": str(item["path"]),
                "columns": ["rgb", "first_order_energy", "second_order_energy"],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 3, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _compact_readout(result: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in ("oof_probabilities", "val_probabilities")
    }


def _write_readout_metrics(path: Path, readouts: Mapping[str, Mapping[str, object]]) -> None:
    fields = (
        "readout",
        "split",
        "accuracy",
        "macro_f1",
        "class1_f1",
        "class1_precision",
        "class1_recall",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in READOUT_NAMES:
            for split, metric_key in (("train_oof", "oof_metrics"), ("val", "val_metrics")):
                metrics = readouts[name][metric_key]
                writer.writerow(
                    {
                        "readout": name,
                        "split": split,
                        "accuracy": f"{float(metrics['accuracy']):.10g}",
                        "macro_f1": f"{float(metrics['macro_f1']):.10g}",
                        "class1_f1": f"{float(metrics['focus_f1']):.10g}",
                        "class1_precision": f"{float(metrics['focus_precision']):.10g}",
                        "class1_recall": f"{float(metrics['focus_recall']):.10g}",
                    }
                )


def _build_alignment(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    plans: Dict[str, Dict[str, object]] = {}
    classes_by_split: Dict[str, List[str]] = {}
    for split in ("train", "val"):
        rows, classes = _imagefolder_rows(Path(args.classification_root), split)
        classes_by_split[split] = classes
        plans[split] = _alignment_plan(
            classification_rows=rows,
            classification_classes=classes,
            yolo_data=Path(args.yolo_data),
            split=split,
            class_name_mode=str(args.class_name_mode),
        )
    if list(plans["train"]["class_names"]) != list(plans["val"]["class_names"]):
        raise ValueError("Train/validation target class order mismatch")
    if len(plans["train"]["class_names"]) != 5:
        raise ValueError("Wavelet readiness audit is locked to the five-class TRKH task")
    return plans, classes_by_split


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.folds) != 5:
        raise ValueError("Protocol is locked to five source-grouped folds; no fold sweep is allowed")
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers must be non-negative")
    for required in (
        Path(args.classification_root) / "train",
        Path(args.classification_root) / "val",
        Path(args.yolo_data),
        Path(args.base_train_csv),
        Path(args.base_val_csv),
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    _assert_output_outside_datasets(
        Path(args.output_dir),
        classification_root=Path(args.classification_root),
        yolo_data=Path(args.yolo_data),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans, classes_by_split = _build_alignment(args)
    preflight = {
        "mode": "wavelet_scattering_readiness_preflight",
        "classification_root": str(Path(args.classification_root).resolve()),
        "yolo_data": str(Path(args.yolo_data).resolve()),
        "base_train_csv": str(Path(args.base_train_csv).resolve()),
        "base_val_csv": str(Path(args.base_val_csv).resolve()),
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "protocol": {
            "image_size": IMAGE_SIZE,
            "channels": list(CHANNEL_NAMES),
            "scattering_j": SCATTERING_J,
            "scattering_l": SCATTERING_L,
            "scattering_max_order": SCATTERING_MAX_ORDER,
            "descriptor": "mean+std+2x2_spatial_pool",
            "pca_components": PCA_COMPONENTS,
            "pca_whiten": PCA_WHITEN,
            "linear_logistic_c": LOGISTIC_C,
            "rbf_svc_c": RBF_SVC_C,
            "rbf_svc_gamma": "scale",
            "primary_readout": PRIMARY_READOUT,
            "folds": int(args.folds),
            "seed": SEED,
            "candidate_selection_uses_validation": False,
        },
        "kymatio_version": str(kymatio.__version__),
        "test_split_used": False,
        "raw_dataset_touched": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
    }
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if bool(args.preflight_only):
        return preflight

    start = time.perf_counter()
    device = _resolve_device(str(args.device or ""))
    scattering = Scattering2D(
        J=SCATTERING_J,
        shape=(IMAGE_SIZE, IMAGE_SIZE),
        L=SCATTERING_L,
        max_order=SCATTERING_MAX_ORDER,
    ).to(device)
    extracted: Dict[str, Dict[str, object]] = {}
    aligned: Dict[str, Dict[str, object]] = {}
    extraction_seconds: Dict[str, float] = {}
    for split in ("train", "val"):
        split_start = time.perf_counter()
        extracted[split] = _extract_split(
            scattering=scattering,
            classification_root=Path(args.classification_root),
            split=split,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
        )
        extraction_seconds[split] = float(time.perf_counter() - split_start)
        if list(extracted[split]["folder_classes"]) != classes_by_split[split]:
            raise ValueError(f"ImageFolder class order changed during extraction for {split}")
        aligned[split] = _align_payload(extracted[split], plans[split])

    train = aligned["train"]
    val = aligned["val"]
    class_names = list(train["class_names"])
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    train_features = np.asarray(train["descriptors"], dtype=np.float32)
    val_features = np.asarray(val["descriptors"], dtype=np.float32)
    descriptor_finite = bool(
        np.isfinite(train_features).all() and np.isfinite(val_features).all()
    )
    if not descriptor_finite:
        raise ValueError("Wavelet descriptor contains non-finite values")
    raw_descriptor_effective_rank = _projected_effective_rank(
        train_features,
        seed=SEED + 991,
    )
    standardized_train_features = StandardScaler().fit_transform(train_features)
    descriptor_effective_rank = _projected_effective_rank(
        standardized_train_features,
        seed=SEED + 991,
    )
    del standardized_train_features
    fit = fit_grouped_scattering_readouts(
        train_features=train_features,
        train_labels=train_labels,
        train_groups=np.asarray(train["source_stems"], dtype=object),
        val_features=val_features,
        val_labels=val_labels,
        class_names=class_names,
        folds=int(args.folds),
        seed=SEED,
    )
    readouts = fit["readouts"]
    base_train = _load_base_predictions(
        Path(args.base_train_csv),
        expected_split="train",
        expected_labels=train_labels,
        class_count=len(class_names),
    )
    base_val = _load_base_predictions(
        Path(args.base_val_csv),
        expected_split="val",
        expected_labels=val_labels,
        class_count=len(class_names),
    )
    candidate_train = np.asarray(
        readouts[PRIMARY_READOUT]["oof_probabilities"],
        dtype=np.float32,
    )
    candidate_val = np.asarray(
        readouts[PRIMARY_READOUT]["val_probabilities"],
        dtype=np.float32,
    )
    direct_train_metrics = _classification_metrics(
        train_labels,
        base_train,
        class_names=class_names,
    )
    direct_val_metrics = _classification_metrics(
        val_labels,
        base_val,
        class_names=class_names,
    )
    train_transitions = _transition_summary(train_labels, base_train, candidate_train)
    val_transitions = _transition_summary(val_labels, base_val, candidate_val)
    train_direction_auc = _class1_error_direction_auc(
        train_labels,
        base_train,
        candidate_train,
    )
    val_direction_auc = _class1_error_direction_auc(
        val_labels,
        base_val,
        candidate_val,
    )
    binary_oracle = _binary_focus_oracle(val_labels, base_val, candidate_val)
    gate = assess_wavelet_scattering_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_groups=int(plans["train"]["source_groups"]),
        descriptor_effective_rank=descriptor_effective_rank,
        descriptor_finite=descriptor_finite,
        direct_val_metrics=direct_val_metrics,
        candidate_oof_metrics=readouts[PRIMARY_READOUT]["oof_metrics"],
        candidate_val_metrics=readouts[PRIMARY_READOUT]["val_metrics"],
        binary_oracle=binary_oracle,
        train_direction_auc=train_direction_auc,
        val_direction_auc=val_direction_auc,
        val_transitions=val_transitions,
    )

    _write_readout_metrics(output_dir / "readout_metrics.csv", readouts)
    _write_prediction_audit(
        output_dir / "train_oof_prediction_audit.csv",
        labels=train_labels,
        sample_index=np.asarray(train["sample_index"], dtype=np.int64),
        paths=np.asarray(train["paths"], dtype=object),
        base_probabilities=base_train,
        variants={
            name: np.asarray(readouts[name]["oof_probabilities"], dtype=np.float32)
            for name in READOUT_NAMES
        },
    )
    _write_prediction_audit(
        output_dir / "val_prediction_audit.csv",
        labels=val_labels,
        sample_index=np.asarray(val["sample_index"], dtype=np.int64),
        paths=np.asarray(val["paths"], dtype=object),
        base_probabilities=base_val,
        variants={
            name: np.asarray(readouts[name]["val_probabilities"], dtype=np.float32)
            for name in READOUT_NAMES
        },
    )
    _write_preview(
        output_dir / "wavelet_order_energy_preview.png",
        preview=val["preview"],
        folder_classes=val["folder_classes"],
    )
    if bool(args.save_descriptors):
        for split, payload in (("train", train), ("val", val)):
            np.savez_compressed(
                output_dir / f"{split}_wavelet_scattering_descriptors.npz",
                descriptors=np.asarray(payload["descriptors"], dtype=np.float32),
                labels=np.asarray(payload["labels"], dtype=np.int64),
                sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
                source_stem=np.asarray(payload["source_stems"], dtype=object),
                paths=np.asarray(payload["paths"], dtype=object),
                classes=np.asarray(class_names, dtype=object),
            )

    summary = {
        "mode": "wavelet_scattering_crop_to_context_readiness",
        "guardrail": (
            "Fixed class_f crop scattering aligned to yolo_f object rows; train/validation "
            "only, no raw-data edit, test access, checkpoint, target manifest, or model training."
        ),
        "literature": LITERATURE,
        "protocol": preflight["protocol"],
        "runtime": {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "torch_threads": int(args.torch_threads),
            "kymatio_version": str(kymatio.__version__),
            "extraction_seconds": extraction_seconds,
            "readout_fold_telemetry": fit["folds"],
            "final_fit_elapsed_seconds": fit["final_fit_elapsed_seconds"],
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "descriptor": {
            "dimension": int(train_features.shape[1]),
            "finite": descriptor_finite,
            "raw_effective_rank_projected128": raw_descriptor_effective_rank,
            "standardized_effective_rank_projected128": descriptor_effective_rank,
            "final_pca_explained_variance": fit["final_pca_explained_variance"],
            "order_stats": {
                split: aligned[split]["order_stats"] for split in ("train", "val")
            },
            "saved": bool(args.save_descriptors),
        },
        "direct_base": {
            "train_metrics": direct_train_metrics,
            "val_metrics": direct_val_metrics,
        },
        "readouts": {
            name: _compact_readout(readouts[name]) for name in READOUT_NAMES
        },
        "class1_complementarity": {
            "binary_validation_oracle": binary_oracle,
            "train_oof_direction_auc": train_direction_auc,
            "val_direction_auc": val_direction_auc,
            "train_transitions_vs_base": train_transitions,
            "val_transitions_vs_base": val_transitions,
        },
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
        "decision": (
            "Proceed to one fixed TRKH-native wavelet-branch smoke."
            if bool(gate["smoke_permission"])
            else "Reject fixed wavelet-scattering branch before GPU training."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    primary_oof = readouts[PRIMARY_READOUT]["oof_metrics"]
    primary_val = readouts[PRIMARY_READOUT]["val_metrics"]
    (output_dir / "README.md").write_text(
        "# Wavelet-scattering readiness audit\n\n"
        "- View: fixed `class_f` crop descriptors strictly aligned to `yolo_f` objects.\n"
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`; test is closed.\n"
        f"- Primary grouped-OOF macro/class1 F1: "
        f"`{primary_oof['macro_f1']:.6f}/{primary_oof['focus_f1']:.6f}`.\n"
        f"- Primary validation macro/class1 F1: "
        f"`{primary_val['macro_f1']:.6f}/{primary_val['focus_f1']:.6f}`.\n"
        f"- Base validation macro/class1 F1: "
        f"`{direct_val_metrics['macro_f1']:.6f}/{direct_val_metrics['focus_f1']:.6f}`.\n"
        f"- FN-vs-FP direction AUROC train/val: "
        f"`{train_direction_auc:.6f}/{val_direction_auc:.6f}`.\n"
        f"- Smoke permission: `{bool(gate['smoke_permission'])}`.\n"
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`.\n\n"
        "This artifact is diagnostic only and writes no training target or checkpoint.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_audit(args)
    if bool(args.preflight_only):
        payload = summary
    else:
        payload = {
            "output_dir": str(Path(args.output_dir).resolve()),
            "smoke_permission": bool(summary["gate"]["smoke_permission"]),
            "failed_checks": summary["gate"]["failed_checks"],
            "decision": summary["decision"],
        }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

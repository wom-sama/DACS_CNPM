from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import StandardScaler
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
from trkh.tools.audit_wavelet_scattering_readiness import (
    PCA_COMPONENTS,
    PRIMARY_READOUT,
    READOUT_NAMES,
    _assert_output_outside_datasets,
    _binary_focus_oracle,
    _compact_readout,
    _write_readout_metrics,
    fit_grouped_scattering_readouts,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _transition_summary,
    _write_prediction_audit,
)


IMAGE_SIZE = 128
ORIENTATION_BINS = 9
CELL_SIZE = 8
READOUT_GRID = IMAGE_SIZE // CELL_SIZE
SEED = 20260711
LITERATURE = (
    "https://openaccess.thecvf.com/content/CVPR2022/html/"
    "Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_"
    "Pre-Training_CVPR_2022_paper.html",
    "https://openaccess.thecvf.com/content/CVPR2022/papers/"
    "Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_"
    "Pre-Training_CVPR_2022_paper.pdf",
    "https://openaccess.thecvf.com/content/CVPR2022/html/"
    "He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html",
)


class PathImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, int(target), str(self.samples[int(index)][0])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only readiness audit for an RGB-HOG MaskFeat target. "
            "The audit writes no target manifest, checkpoint, or dataset file."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--base-train-csv", type=Path, required=True)
    parser.add_argument("--base-val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rgb_hog_cell_map(
    images: Tensor,
    *,
    orientation_bins: int = ORIENTATION_BINS,
    cell_size: int = CELL_SIZE,
    epsilon: float = 1e-6,
) -> Tensor:
    """Dense per-RGB-channel unsigned HOG with one L2-normalized vector per cell."""

    if images.ndim != 4 or int(images.size(1)) != 3:
        raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(images.shape)}")
    bins = int(orientation_bins)
    cell = int(cell_size)
    if bins < 2 or cell < 1 or float(epsilon) <= 0.0:
        raise ValueError("orientation_bins/cell_size/epsilon are invalid")
    height, width = int(images.size(2)), int(images.size(3))
    if height % cell or width % cell:
        raise ValueError(f"Image shape {(height, width)} must be divisible by cell_size={cell}")

    values = images.float()
    padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
    gradient_x = 0.5 * (
        padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
    )
    gradient_y = 0.5 * (
        padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
    )
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)
    orientation = torch.remainder(torch.atan2(gradient_y, gradient_x), math.pi)
    bin_coordinate = orientation * (float(bins) / math.pi)
    lower_float = torch.floor(bin_coordinate)
    lower = lower_float.to(dtype=torch.long).remainder(bins)
    upper = (lower + 1).remainder(bins)
    upper_weight = bin_coordinate - lower_float
    lower_weight = 1.0 - upper_weight

    histogram = values.new_zeros(
        (int(values.size(0)), int(values.size(1)), bins, height, width)
    )
    histogram.scatter_add_(
        2,
        lower.unsqueeze(2),
        (magnitude * lower_weight).unsqueeze(2),
    )
    histogram.scatter_add_(
        2,
        upper.unsqueeze(2),
        (magnitude * upper_weight).unsqueeze(2),
    )
    batch, channels = int(values.size(0)), int(values.size(1))
    cells_y, cells_x = height // cell, width // cell
    pooled = F.avg_pool2d(
        histogram.reshape(batch * channels * bins, 1, height, width),
        kernel_size=cell,
        stride=cell,
    ).reshape(batch, channels, bins, cells_y, cells_x)
    norm = torch.sqrt(pooled.square().sum(dim=2, keepdim=True) + float(epsilon))
    return pooled / norm


def hog_readout_descriptor(
    cell_map: Tensor,
    *,
    readout_grid: int = READOUT_GRID,
) -> Tensor:
    if cell_map.ndim != 5:
        raise ValueError(f"Expected HOG [B,C,K,H,W], got {tuple(cell_map.shape)}")
    grid = int(readout_grid)
    if grid < 1:
        raise ValueError("readout_grid must be positive")
    batch, channels, bins, height, width = cell_map.shape
    pooled = F.adaptive_avg_pool2d(
        cell_map.reshape(batch, channels * bins, height, width),
        output_size=(grid, grid),
    )
    return pooled.flatten(1)


def _build_alignment(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    plans: Dict[str, Dict[str, object]] = {}
    classes_by_split: Dict[str, List[str]] = {}
    for split in ("train", "val"):
        rows, classes = _imagefolder_rows(Path(args.classification_root), split)
        plans[split] = _alignment_plan(
            classification_rows=rows,
            classification_classes=classes,
            yolo_data=Path(args.yolo_data),
            split=split,
            class_name_mode=str(args.class_name_mode),
        )
        classes_by_split[split] = classes
    if list(plans["train"]["class_names"]) != list(plans["val"]["class_names"]):
        raise ValueError("Train/validation target class order differs")
    return plans, classes_by_split


def _extract_split(
    *,
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
    dataset = PathImageFolder(Path(classification_root) / split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(workers) > 0),
    )
    descriptors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    paths: List[str] = []
    preview: Dict[int, Dict[str, object]] = {}
    normalized_cell_sum = 0.0
    normalized_cell_count = 0
    zero_cell_count = 0
    with torch.inference_mode():
        for images, targets, batch_paths in tqdm(loader, desc=f"maskfeat-hog-{split}"):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            cell_map = rgb_hog_cell_map(images)
            descriptor = hog_readout_descriptor(cell_map)
            descriptors.append(descriptor.cpu().numpy().astype(np.float32, copy=False))
            labels.append(targets.numpy().astype(np.int64, copy=False))
            paths.extend(str(path) for path in batch_paths)
            cell_norm = torch.sqrt(cell_map.square().sum(dim=2))
            normalized_cell_sum += float(cell_norm.sum().item())
            normalized_cell_count += int(cell_norm.numel())
            zero_cell_count += int((cell_norm < 1e-3).sum().item())
            for row_index, target in enumerate(targets.tolist()):
                class_index = int(target)
                if class_index in preview:
                    continue
                rgb = (
                    images[row_index]
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                    .clip(0.0, 1.0)
                )
                preview[class_index] = {
                    "rgb": (rgb * 255.0).round().astype(np.uint8),
                    "path": str(batch_paths[row_index]),
                    "channel_energy": cell_map[row_index]
                    .square()
                    .sum(dim=1)
                    .cpu()
                    .numpy(),
                }
    if not descriptors:
        raise ValueError(f"Empty split: {classification_root / split}")
    return {
        "descriptors": np.concatenate(descriptors, axis=0).astype(np.float32, copy=False),
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "folder_classes": list(dataset.classes),
        "preview": preview,
        "cell_stats": {
            "mean_normalized_cell_norm": float(
                normalized_cell_sum / max(1, normalized_cell_count)
            ),
            "zero_cell_fraction": float(zero_cell_count / max(1, normalized_cell_count)),
        },
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
        "preview": payload["preview"],
        "cell_stats": payload["cell_stats"],
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
        energy = np.asarray(item["channel_energy"], dtype=np.float32)
        row = Image.new("RGB", (tile * 4, tile), color=(255, 255, 255))
        row.paste(Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR), (0, 0))
        for channel_index in range(3):
            overlay = _heat_overlay(rgb, energy[channel_index])
            row.paste(
                Image.fromarray(overlay).resize((tile, tile), resampling.BILINEAR),
                ((channel_index + 1) * tile, 0),
            )
        rows.append(row)
        manifest.append(
            {
                "row": int(len(rows) - 1),
                "class_index": int(class_index),
                "class_name": str(class_name),
                "path": str(item["path"]),
                "columns": ["rgb", "red_hog_energy", "green_hog_energy", "blue_hog_energy"],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 4, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def assess_maskfeat_hog_readiness(
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
        "descriptor_rank": observed["descriptor_effective_rank"]
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
        "train_error_direction": observed["train_direction_auc"]
        >= float(thresholds["min_direction_auc"]),
        "validation_error_direction": observed["val_direction_auc"]
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
        "maskfeat_hog_target_ready": not failed,
        "smoke_ready": not failed,
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.folds) != 5:
        raise ValueError("Protocol is locked to five source-grouped folds")
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
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
        "mode": "maskfeat_hog_target_readiness_preflight",
        "classification_root": str(Path(args.classification_root).resolve()),
        "yolo_data": str(Path(args.yolo_data).resolve()),
        "base_train_csv": str(Path(args.base_train_csv).resolve()),
        "base_val_csv": str(Path(args.base_val_csv).resolve()),
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "protocol": {
            "image_size": IMAGE_SIZE,
            "channels": "rgb_separate_histograms",
            "unsigned_orientation_bins": ORIENTATION_BINS,
            "cell_size": CELL_SIZE,
            "cell_normalization": "l2",
            "readout_grid": READOUT_GRID,
            "descriptor_dimension": 3 * ORIENTATION_BINS * READOUT_GRID * READOUT_GRID,
            "pca_components": PCA_COMPONENTS,
            "primary_readout": PRIMARY_READOUT,
            "folds": int(args.folds),
            "seed": SEED,
            "candidate_selection_uses_validation": False,
            "paper_epoch_caveat": (
                "MaskFeat image ablations use hundreds of pretrain epochs; this project "
                "requires <=30 total epochs, so a strong fixed-target readiness signal is mandatory."
            ),
        },
        "literature": list(LITERATURE),
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
    extracted: Dict[str, Dict[str, object]] = {}
    aligned: Dict[str, Dict[str, object]] = {}
    extraction_seconds: Dict[str, float] = {}
    for split in ("train", "val"):
        split_start = time.perf_counter()
        extracted[split] = _extract_split(
            classification_root=Path(args.classification_root),
            split=split,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
        )
        extraction_seconds[split] = float(time.perf_counter() - split_start)
        if list(extracted[split]["folder_classes"]) != classes_by_split[split]:
            raise ValueError(f"ImageFolder class order changed for {split}")
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
        raise ValueError("HOG descriptor contains non-finite values")
    raw_rank = _projected_effective_rank(train_features, seed=SEED + 991)
    standardized = StandardScaler().fit_transform(train_features)
    standardized_rank = _projected_effective_rank(standardized, seed=SEED + 991)
    del standardized
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
        readouts[PRIMARY_READOUT]["oof_probabilities"], dtype=np.float32
    )
    candidate_val = np.asarray(
        readouts[PRIMARY_READOUT]["val_probabilities"], dtype=np.float32
    )
    direct_train_metrics = _classification_metrics(
        train_labels, base_train, class_names=class_names
    )
    direct_val_metrics = _classification_metrics(
        val_labels, base_val, class_names=class_names
    )
    train_transitions = _transition_summary(train_labels, base_train, candidate_train)
    val_transitions = _transition_summary(val_labels, base_val, candidate_val)
    train_direction_auc = _class1_error_direction_auc(
        train_labels, base_train, candidate_train
    )
    val_direction_auc = _class1_error_direction_auc(val_labels, base_val, candidate_val)
    binary_oracle = _binary_focus_oracle(val_labels, base_val, candidate_val)
    gate = assess_maskfeat_hog_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_groups=int(plans["train"]["source_groups"]),
        descriptor_effective_rank=standardized_rank,
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
        output_dir / "maskfeat_hog_rgb_energy_preview.png",
        preview=val["preview"],
        folder_classes=val["folder_classes"],
    )
    summary = {
        "mode": "maskfeat_hog_target_readiness",
        "guardrail": (
            "Fixed RGB-HOG crop descriptor strictly aligned to yolo_f object rows; "
            "train/validation only, no raw-data edit, test, target manifest, checkpoint, "
            "or image-model training."
        ),
        "literature": list(LITERATURE),
        "protocol": preflight["protocol"],
        "runtime": {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "torch_threads": int(args.torch_threads),
            "extraction_seconds": extraction_seconds,
            "readout_fold_telemetry": fit["folds"],
            "final_fit_elapsed_seconds": fit["final_fit_elapsed_seconds"],
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "descriptor": {
            "dimension": int(train_features.shape[1]),
            "finite": descriptor_finite,
            "raw_effective_rank_projected128": raw_rank,
            "standardized_effective_rank_projected128": standardized_rank,
            "final_pca_explained_variance": fit["final_pca_explained_variance"],
            "cell_stats": {
                split: aligned[split]["cell_stats"] for split in ("train", "val")
            },
            "saved": False,
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
        "image_model_trained": False,
        "decision": (
            "Proceed to one fixed short MaskFeat-HOG auxiliary smoke."
            if bool(gate["smoke_permission"])
            else "Reject MaskFeat-HOG target before image-model training."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    primary_oof = readouts[PRIMARY_READOUT]["oof_metrics"]
    primary_val = readouts[PRIMARY_READOUT]["val_metrics"]
    (output_dir / "README.md").write_text(
        "# MaskFeat-HOG target readiness audit\n\n"
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`; test is closed.\n"
        f"- RGB HOG: `{ORIENTATION_BINS}` unsigned bins, `{CELL_SIZE}x{CELL_SIZE}` cells, L2 normalization.\n"
        f"- Primary grouped-OOF macro/class1 F1: `{primary_oof['macro_f1']:.6f}/{primary_oof['focus_f1']:.6f}`.\n"
        f"- Primary validation macro/class1 F1: `{primary_val['macro_f1']:.6f}/{primary_val['focus_f1']:.6f}`.\n"
        f"- Base validation macro/class1 F1: `{direct_val_metrics['macro_f1']:.6f}/{direct_val_metrics['focus_f1']:.6f}`.\n"
        f"- FN-vs-FP direction AUROC train/val: `{train_direction_auc:.6f}/{val_direction_auc:.6f}`.\n"
        f"- Smoke permission: `{bool(gate['smoke_permission'])}`.\n"
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`.\n\n"
        "This artifact writes no MaskFeat target, model, or checkpoint.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(
        output_dir,
        mode="maskfeat_hog_target_readiness_evidence_manifest",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(args)
    if bool(summary.get("mode") == "maskfeat_hog_target_readiness_preflight"):
        payload = summary
    else:
        payload = {
            "output_dir": str(Path(args.output_dir).resolve()),
            "smoke_permission": bool(summary["gate"]["smoke_permission"]),
            "failed_checks": summary["gate"]["failed_checks"],
            "decision": summary["decision"],
        }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

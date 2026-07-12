from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
from huggingface_hub import try_to_load_from_cache
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from trkh.core.utils import autocast_context
from trkh.tools.audit_dinov2_dense_patch_readiness import (
    FEATURE_NAMES,
    PathImageFolder,
    SPATIAL_POOL_SIZE,
    _align_payload,
    _build_alignment,
    _extract_split,
    _resolve_device,
    _save_descriptor_cache,
    _write_preview,
    dino_dense_patch_descriptors,
)
from trkh.tools.audit_multistage_teacher_feature_readiness import (
    _class1_error_direction_auc,
    _compact_alignment,
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


DINO_V2_MODEL = "vit_small_patch14_dinov2.lvd142m"
DINO_V2_HF_REPO = "timm/vit_small_patch14_dinov2.lvd142m"
DINO_V2_WEIGHT_SHA256 = (
    "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
)
DINO_V2_PARAMETER_COUNT = 21_628_800
DINO_V3_MODEL = "vit_small_patch16_dinov3.lvd1689m"
DINO_V3_HF_REPO = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_V3_WEIGHT_SHA256 = (
    "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
)
DINO_V3_PARAMETER_COUNT = 21_586_944
IMAGE_SIZE = 256
PATCH_SIZE = 16
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE
PREFIX_TOKENS = 5
EMBEDDING_DIMENSION = 384
SEED = 20260712
MODEL_SPECS = {
    "dinov2": {
        "model_name": DINO_V2_MODEL,
        "hf_repo": DINO_V2_HF_REPO,
        "weight_sha256": DINO_V2_WEIGHT_SHA256,
        "parameter_count": DINO_V2_PARAMETER_COUNT,
        "image_size": 224,
        "patch_size": 14,
        "patch_grid": (16, 16),
        "prefix_tokens": 1,
    },
    "dinov3": {
        "model_name": DINO_V3_MODEL,
        "hf_repo": DINO_V3_HF_REPO,
        "weight_sha256": DINO_V3_WEIGHT_SHA256,
        "parameter_count": DINO_V3_PARAMETER_COUNT,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "patch_grid": (PATCH_GRID, PATCH_GRID),
        "prefix_tokens": PREFIX_TOKENS,
    },
}
LITERATURE = (
    "https://arxiv.org/abs/2508.10104",
    "https://github.com/facebookresearch/dinov3",
    "https://huggingface.co/collections/facebook/dinov3",
    "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m",
    "https://arxiv.org/abs/2304.07193",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train/validation-only comparison of frozen DINOv3-S and "
            "the retained matched DINOv2-S dense-patch audit."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--base-train-csv", type=Path, required=True)
    parser.add_argument("--base-val-csv", type=Path, required=True)
    parser.add_argument("--dinov2-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DINO_V3_MODEL)
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--save-descriptors", action="store_true", default=False)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path(value: object) -> str:
    return str(value).replace("/", "\\").casefold()


def _cached_weight_provenance(
    *,
    hf_repo: str,
    expected_sha256: str,
) -> Dict[str, object]:
    cached = try_to_load_from_cache(str(hf_repo), "model.safetensors")
    if not isinstance(cached, str) or not Path(cached).is_file():
        raise FileNotFoundError(
            f"Exact cached weight is unavailable: {hf_repo}/model.safetensors"
        )
    path = Path(cached)
    digest = _sha256_file(path)
    if digest != str(expected_sha256):
        raise ValueError(
            f"Frozen DINO weight hash changed: {digest} != {expected_sha256}"
        )
    snapshot = path.parent.name if path.parent.parent.name == "snapshots" else ""
    return {
        "hf_repo": str(hf_repo),
        "filename": "model.safetensors",
        "cached_path": str(path.resolve()),
        "snapshot": snapshot,
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def _build_model_and_transform(
    model_key: str,
    *,
    device: torch.device,
) -> Tuple[torch.nn.Module, object, Dict[str, object]]:
    spec = MODEL_SPECS[str(model_key)]
    image_size = int(spec["image_size"])
    model = timm.create_model(
        str(spec["model_name"]),
        pretrained=True,
        num_classes=0,
        img_size=image_size,
    )
    grid_size = tuple(int(value) for value in model.patch_embed.grid_size)
    patch_size = tuple(int(value) for value in model.patch_embed.patch_size)
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
    embedding_dimension = int(getattr(model, "num_features", 0))
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    observed = {
        "patch_grid": grid_size,
        "patch_size": patch_size,
        "prefix_tokens": prefix_tokens,
        "embedding_dimension": embedding_dimension,
        "parameter_count": parameter_count,
    }
    expected = {
        "patch_grid": tuple(spec["patch_grid"]),
        "patch_size": (int(spec["patch_size"]), int(spec["patch_size"])),
        "prefix_tokens": int(spec["prefix_tokens"]),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "parameter_count": int(spec["parameter_count"]),
    }
    if observed != expected:
        raise ValueError(f"Unexpected locked {model_key} architecture: {observed} != {expected}")
    config = timm.data.resolve_model_data_config(model)
    mean = tuple(float(value) for value in config["mean"])
    std = tuple(float(value) for value in config["std"])
    transform = v2.Compose(
        (
            v2.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        )
    )
    weight = _cached_weight_provenance(
        hf_repo=str(spec["hf_repo"]),
        expected_sha256=str(spec["weight_sha256"]),
    )
    model.to(device).eval()
    metadata = {
        "model_name": str(spec["model_name"]),
        "parameter_count": parameter_count,
        "embedding_dimension": embedding_dimension,
        "prefix_tokens": prefix_tokens,
        "register_tokens": prefix_tokens - 1,
        "patch_size": int(spec["patch_size"]),
        "patch_grid": list(grid_size),
        "image_size": image_size,
        "mean": list(mean),
        "std": list(std),
        "hf_hub_id": str(getattr(model, "default_cfg", {}).get("hf_hub_id", "")),
        "weight": weight,
        "timm_version": str(timm.__version__),
        "torch_version": str(torch.__version__),
    }
    return model, transform, metadata


def _numerical_precision_probe(
    *,
    model: torch.nn.Module,
    transform,
    classification_root: Path,
    device: torch.device,
    model_metadata: Mapping[str, object],
) -> Dict[str, object]:
    dataset = PathImageFolder(Path(classification_root) / "train", transform=transform)
    sample_count = min(16, len(dataset))
    if sample_count < 2:
        raise ValueError("Numerical precision probe requires at least two train images")
    images = torch.stack([dataset[index][0] for index in range(sample_count)]).to(
        device=device, dtype=torch.float32
    )
    midpoint = sample_count // 2

    def descriptors(batch: torch.Tensor, *, amp: bool) -> Dict[str, torch.Tensor]:
        with torch.inference_mode(), autocast_context(device, amp):
            tokens = model.forward_features(batch)
        values = dino_dense_patch_descriptors(
            tokens,
            grid_size=tuple(int(value) for value in model_metadata["patch_grid"]),
            prefix_tokens=int(model_metadata["prefix_tokens"]),
        )
        return {
            name: values[name].detach().float().cpu()
            for name in FEATURE_NAMES
        }

    fp32_full = descriptors(images, amp=False)
    fp32_left = descriptors(images[:midpoint], amp=False)
    fp32_right = descriptors(images[midpoint:], amp=False)
    amp_full = descriptors(images, amp=True)
    amp_left = descriptors(images[:midpoint], amp=True)
    amp_right = descriptors(images[midpoint:], amp=True)
    telemetry: Dict[str, object] = {}
    for name in FEATURE_NAMES:
        fp32_split = torch.cat((fp32_left[name], fp32_right[name]), dim=0)
        amp_split = torch.cat((amp_left[name], amp_right[name]), dim=0)
        amp_precision_delta = (amp_full[name] - fp32_full[name]).abs()
        amp_geometry_delta = (amp_full[name] - amp_split).abs()
        fp32_geometry_delta = (fp32_full[name] - fp32_split).abs()
        telemetry[name] = {
            "amp_vs_fp32_mean_abs": float(amp_precision_delta.mean().item()),
            "amp_vs_fp32_max_abs": float(amp_precision_delta.max().item()),
            "amp_batch_geometry_mean_abs": float(amp_geometry_delta.mean().item()),
            "amp_batch_geometry_max_abs": float(amp_geometry_delta.max().item()),
            "fp32_batch_geometry_mean_abs": float(fp32_geometry_delta.mean().item()),
            "fp32_batch_geometry_max_abs": float(fp32_geometry_delta.max().item()),
        }
        if float(fp32_geometry_delta.max().item()) > 1e-6:
            raise ValueError(
                f"FP32 descriptor changes with batch geometry for {name}: "
                f"{float(fp32_geometry_delta.max().item())}"
            )
    return {
        "sample_count": sample_count,
        "split_geometry": [sample_count, midpoint, sample_count - midpoint],
        "selected_precision": "fp32",
        "reason": (
            "FP32 is batch-geometry stable; AMP is measured only as a rejected "
            "runtime alternative and is never used for full descriptors."
        ),
        "features": telemetry,
    }


def _validate_dinov2_summary(
    summary: Mapping[str, object],
    *,
    expected_train_rows: int,
    expected_val_rows: int,
) -> None:
    protocol = summary.get("protocol", {})
    model = summary.get("model", {})
    alignment = summary.get("alignment", {})
    checks = {
        "mode": summary.get("mode") == "dinov2_dense_patch_readiness",
        "model": model.get("model_name") == DINO_V2_MODEL,
        "protocol_model": protocol.get("model") == DINO_V2_MODEL,
        "folds": int(protocol.get("folds", -1)) == 5,
        "seed": int(protocol.get("seed", -1)) == SEED,
        "matched_source_folds": bool(protocol.get("matched_source_folds")),
        "primary_readout": protocol.get("primary_readout") == PRIMARY_READOUT,
        "test_closed": summary.get("test_split_used") is False,
        "train_rows": int(alignment.get("train", {}).get("yolo_rows", -1))
        == int(expected_train_rows),
        "val_rows": int(alignment.get("val", {}).get("yolo_rows", -1))
        == int(expected_val_rows),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(f"Retained DINOv2 audit is not protocol-compatible: {failed}")


def _load_variant_probabilities(
    path: Path,
    *,
    variant: str,
    expected_labels: np.ndarray,
    expected_sample_index: np.ndarray,
    expected_paths: np.ndarray,
    class_count: int,
) -> np.ndarray:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    probability_fields = [f"{variant}_prob_{index}" for index in range(class_count)]
    required = {"sample_index", "target", "path", *probability_fields}
    missing_fields = sorted(required.difference(fields))
    if missing_fields:
        raise ValueError(f"Missing {path} fields for {variant}: {missing_fields}")
    indexed: Dict[int, Mapping[str, str]] = {}
    for row in rows:
        sample_index = int(row["sample_index"])
        if sample_index in indexed:
            raise ValueError(f"Duplicate sample_index={sample_index} in {path}")
        indexed[sample_index] = row
    expected_indices = np.asarray(expected_sample_index, dtype=np.int64)
    if sorted(indexed) != expected_indices.tolist():
        raise ValueError(f"Strict sample_index mismatch in {path}")
    probabilities = np.zeros((expected_indices.size, class_count), dtype=np.float32)
    for position, sample_index in enumerate(expected_indices.tolist()):
        row = indexed[int(sample_index)]
        if int(row["target"]) != int(expected_labels[position]):
            raise ValueError(f"Target mismatch at sample_index={sample_index} in {path}")
        if _normalize_path(row["path"]) != _normalize_path(expected_paths[position]):
            raise ValueError(f"Path mismatch at sample_index={sample_index} in {path}")
        probabilities[position] = np.asarray(
            [float(row[field]) for field in probability_fields], dtype=np.float32
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError(f"Invalid {variant} probabilities in {path}")
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities


def _selection_score(metrics: Mapping[str, object]) -> float:
    return float(metrics["macro_f1"]) + float(metrics["focus_f1"])


def select_oof_feature(
    probabilities: Mapping[str, np.ndarray],
    *,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    metrics = {
        name: _classification_metrics(labels, values, class_names=class_names)
        for name, values in probabilities.items()
    }
    ordered = sorted(
        metrics,
        key=lambda name: (-_selection_score(metrics[name]), FEATURE_NAMES.index(name)),
    )
    return {
        "selected_feature": ordered[0],
        "criterion": "train_oof_macro_f1_plus_class1_f1",
        "scores": {name: _selection_score(metrics[name]) for name in FEATURE_NAMES},
        "metrics": metrics,
    }


def _fold_comparison(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows: List[Dict[str, object]] = []
    for fold_index, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(labels.size), labels, groups=groups), start=1
    ):
        if set(groups[fit_indices]).intersection(set(groups[holdout_indices])):
            raise ValueError(f"Source leakage in fold {fold_index}")
        control = _classification_metrics(
            labels[holdout_indices],
            control_probabilities[holdout_indices],
            class_names=class_names,
        )
        candidate = _classification_metrics(
            labels[holdout_indices],
            candidate_probabilities[holdout_indices],
            class_names=class_names,
        )
        rows.append(
            {
                "fold": fold_index,
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "fit_groups": int(np.unique(groups[fit_indices]).size),
                "holdout_groups": int(np.unique(groups[holdout_indices]).size),
                "source_overlap": 0,
                "control_macro_f1": float(control["macro_f1"]),
                "control_class1_f1": float(control["focus_f1"]),
                "candidate_macro_f1": float(candidate["macro_f1"]),
                "candidate_class1_f1": float(candidate["focus_f1"]),
                "macro_gain": float(candidate["macro_f1"])
                - float(control["macro_f1"]),
                "class1_gain": float(candidate["focus_f1"])
                - float(control["focus_f1"]),
            }
        )
    return rows


def assess_dinov3_readiness(
    *,
    train_samples: int,
    val_samples: int,
    source_groups: int,
    cross_split_source_overlap: int,
    descriptor_finite: bool,
    dense_effective_rank: float,
    energy_entropy: float,
    energy_border_mass: float,
    control_oof_metrics: Mapping[str, object],
    candidate_oof_metrics: Mapping[str, object],
    control_val_metrics: Mapping[str, object],
    candidate_val_metrics: Mapping[str, object],
    keeper_val_metrics: Mapping[str, object],
    fold_comparison: Sequence[Mapping[str, object]],
    keeper_oracle: Mapping[str, float],
    train_direction_auc: float,
    val_direction_auc: float,
    keeper_transitions: Mapping[str, int],
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_source_groups": 8000,
        "min_dense_effective_rank": 24.0,
        "min_energy_entropy": 0.35,
        "max_energy_border_mass": 0.65,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_class1_fold_wins": 4,
        "min_val_class1_f1": 0.70,
        "min_keeper_macro_gain": 0.002,
        "min_keeper_class1_gain": 0.015,
        "max_keeper_class1_recall_drop": 0.02,
        "min_keeper_oracle_class1_gain": 0.02,
        "min_direction_auc": 0.58,
        "max_direction_auc_gap": 0.12,
    }
    class1_fold_wins = sum(float(row["class1_gain"]) > 0.0 for row in fold_comparison)
    observed = {
        "train_samples": int(train_samples),
        "val_samples": int(val_samples),
        "source_groups": int(source_groups),
        "cross_split_source_overlap": int(cross_split_source_overlap),
        "dense_effective_rank": float(dense_effective_rank),
        "energy_entropy": float(energy_entropy),
        "energy_border_mass": float(energy_border_mass),
        "oof_macro_gain": float(candidate_oof_metrics["macro_f1"])
        - float(control_oof_metrics["macro_f1"]),
        "oof_class1_gain": float(candidate_oof_metrics["focus_f1"])
        - float(control_oof_metrics["focus_f1"]),
        "val_macro_gain": float(candidate_val_metrics["macro_f1"])
        - float(control_val_metrics["macro_f1"]),
        "val_class1_gain": float(candidate_val_metrics["focus_f1"])
        - float(control_val_metrics["focus_f1"]),
        "class1_fold_wins": int(class1_fold_wins),
        "keeper_macro_gain": float(candidate_val_metrics["macro_f1"])
        - float(keeper_val_metrics["macro_f1"]),
        "keeper_class1_gain": float(candidate_val_metrics["focus_f1"])
        - float(keeper_val_metrics["focus_f1"]),
        "keeper_class1_recall_drop": float(keeper_val_metrics["focus_recall"])
        - float(candidate_val_metrics["focus_recall"]),
        "keeper_oracle_class1_gain": float(keeper_oracle["f1"])
        - float(keeper_val_metrics["focus_f1"]),
        "train_direction_auc": float(train_direction_auc),
        "val_direction_auc": float(val_direction_auc),
        "direction_auc_gap": abs(float(train_direction_auc) - float(val_direction_auc)),
    }
    checks = {
        "train_support": observed["train_samples"] >= thresholds["min_train_samples"],
        "validation_support": observed["val_samples"] == thresholds["required_val_samples"],
        "source_group_support": observed["source_groups"] >= thresholds["min_source_groups"],
        "cross_split_source_isolation": observed["cross_split_source_overlap"] == 0,
        "descriptor_finite": bool(descriptor_finite),
        "dense_rank": observed["dense_effective_rank"]
        >= thresholds["min_dense_effective_rank"],
        "spatial_energy_noncollapsed": observed["energy_entropy"]
        >= thresholds["min_energy_entropy"],
        "spatial_energy_not_border_dominated": observed["energy_border_mass"]
        <= thresholds["max_energy_border_mass"],
        "oof_macro_gain": observed["oof_macro_gain"] >= thresholds["min_oof_macro_gain"],
        "oof_class1_gain": observed["oof_class1_gain"]
        >= thresholds["min_oof_class1_gain"],
        "validation_macro_gain": observed["val_macro_gain"]
        >= thresholds["min_val_macro_gain"],
        "validation_class1_gain": observed["val_class1_gain"]
        >= thresholds["min_val_class1_gain"],
        "class1_fold_consistency": observed["class1_fold_wins"]
        >= thresholds["min_class1_fold_wins"],
        "validation_class1_milestone": float(candidate_val_metrics["focus_f1"])
        >= thresholds["min_val_class1_f1"],
        "keeper_macro_gain": observed["keeper_macro_gain"]
        >= thresholds["min_keeper_macro_gain"],
        "keeper_class1_gain": observed["keeper_class1_gain"]
        >= thresholds["min_keeper_class1_gain"],
        "keeper_class1_recall_preserved": observed["keeper_class1_recall_drop"]
        <= thresholds["max_keeper_class1_recall_drop"],
        "keeper_binary_oracle_complementarity": observed["keeper_oracle_class1_gain"]
        >= thresholds["min_keeper_oracle_class1_gain"],
        "train_error_direction": observed["train_direction_auc"]
        >= thresholds["min_direction_auc"],
        "validation_error_direction": observed["val_direction_auc"]
        >= thresholds["min_direction_auc"],
        "error_direction_stability": observed["direction_auc_gap"]
        <= thresholds["max_direction_auc_gap"],
        "keeper_validation_net_corrections": int(keeper_transitions["corrections"])
        >= int(keeper_transitions["harms"]),
        "keeper_validation_class1_recall_protected": int(
            keeper_transitions["class1_fn_rescued"]
        )
        >= int(keeper_transitions["class1_tp_broken"]),
        "keeper_validation_class1_fp_control": int(
            keeper_transitions["class1_fp_removed"]
        )
        >= int(keeper_transitions["class1_fp_created"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "dinov3_representation_ready": ready,
        "smoke_ready": ready,
        "smoke_permission": ready,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _protocol() -> Dict[str, object]:
    dense_dimension = (
        2 + SPATIAL_POOL_SIZE * SPATIAL_POOL_SIZE
    ) * EMBEDDING_DIMENSION
    return {
        "control_model": DINO_V2_MODEL,
        "candidate_model": DINO_V3_MODEL,
        "control_pretrained": True,
        "candidate_pretrained": True,
        "control_image_size": 224,
        "control_patch_size": 14,
        "control_patch_grid": [16, 16],
        "control_prefix_tokens": 1,
        "candidate_image_size": IMAGE_SIZE,
        "candidate_patch_size": PATCH_SIZE,
        "candidate_patch_grid": [PATCH_GRID, PATCH_GRID],
        "candidate_prefix_tokens": PREFIX_TOKENS,
        "candidate_register_tokens": PREFIX_TOKENS - 1,
        "features": list(FEATURE_NAMES),
        "dense_patch_signature": "patch_mean+patch_std+2x2_spatial_means",
        "control_dimension": EMBEDDING_DIMENSION,
        "dense_patch_dimension": dense_dimension,
        "candidate_dimension": EMBEDDING_DIMENSION + dense_dimension,
        "pca_components": PCA_COMPONENTS,
        "primary_readout": PRIMARY_READOUT,
        "readouts": list(READOUT_NAMES),
        "feature_selection": "train_oof_macro_f1_plus_class1_f1",
        "descriptor_precision": "fp32",
        "amp_rejected_by_batch_geometry_probe": True,
        "folds": 5,
        "seed": SEED,
        "matched_source_folds": True,
        "candidate_selection_uses_validation": False,
        "descriptor_grid_sweep": False,
        "test_selection": False,
        "comparison_scope": (
            "Released model+preprocessing recipe; DINOv2 is re-extracted at "
            "224/patch14 and DINOv3 at 256/patch16, both in FP32. Both yield a "
            "16x16 patch grid and the same descriptor/readout dimensions."
        ),
    }


def _prepare_output_dir(path: Path) -> None:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(
        child.name for child in output_dir.iterdir() if child.name != "preflight.json"
    )
    if stale:
        raise FileExistsError(
            f"Output directory contains stale audit payloads: {stale[:8]}"
        )


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.folds) != 5:
        raise ValueError("Protocol is locked to five source-grouped folds")
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    if str(args.model) != DINO_V3_MODEL:
        raise ValueError(f"Locked protocol requires --model {DINO_V3_MODEL!r}")
    if bool(args.amp):
        raise ValueError(
            "This matched protocol is locked to FP32; AMP changed descriptors "
            "with batch geometry in both DINOv2 and DINOv3 preflights"
        )
    for required in (
        Path(args.classification_root) / "train",
        Path(args.classification_root) / "val",
        Path(args.yolo_data),
        Path(args.base_train_csv),
        Path(args.base_val_csv),
        Path(args.dinov2_audit_dir) / "summary.json",
        Path(args.dinov2_audit_dir) / "train_oof_prediction_audit.csv",
        Path(args.dinov2_audit_dir) / "val_prediction_audit.csv",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    _assert_output_outside_datasets(
        Path(args.output_dir),
        classification_root=Path(args.classification_root),
        yolo_data=Path(args.yolo_data),
    )
    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir)
    plans, classes_by_split = _build_alignment(args)
    train_groups = set(str(value) for value in plans["train"]["source_stems"])
    val_groups = set(str(value) for value in plans["val"]["source_stems"])
    cross_split_source_overlap = len(train_groups.intersection(val_groups))
    if cross_split_source_overlap:
        raise ValueError(f"Train/validation source overlap: {cross_split_source_overlap}")
    dinov2_summary_path = Path(args.dinov2_audit_dir) / "summary.json"
    dinov2_summary = json.loads(dinov2_summary_path.read_text(encoding="utf-8"))
    _validate_dinov2_summary(
        dinov2_summary,
        expected_train_rows=int(plans["train"]["yolo_rows"]),
        expected_val_rows=int(plans["val"]["yolo_rows"]),
    )
    dinov2_inputs = {
        name: {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for name, path in {
            "summary": dinov2_summary_path,
            "train_predictions": Path(args.dinov2_audit_dir)
            / "train_oof_prediction_audit.csv",
            "val_predictions": Path(args.dinov2_audit_dir) / "val_prediction_audit.csv",
        }.items()
    }
    preflight = {
        "mode": "dinov3_dense_patch_readiness_preflight",
        "classification_root": str(Path(args.classification_root).resolve()),
        "yolo_data": str(Path(args.yolo_data).resolve()),
        "base_train_csv": str(Path(args.base_train_csv).resolve()),
        "base_val_csv": str(Path(args.base_val_csv).resolve()),
        "dinov2_reference": dinov2_inputs,
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "cross_split_source_overlap": cross_split_source_overlap,
        "protocol": _protocol(),
        "expected_weight_sha256": {
            "dinov2": DINO_V2_WEIGHT_SHA256,
            "dinov3": DINO_V3_WEIGHT_SHA256,
        },
        "literature": list(LITERATURE),
        "test_split_used": False,
        "raw_dataset_touched": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if bool(args.preflight_only):
        return preflight

    start = time.perf_counter()
    device = _resolve_device(str(args.device or ""))
    aligned_by_model: Dict[str, Dict[str, Dict[str, object]]] = {}
    model_metadata: Dict[str, Dict[str, object]] = {}
    numerical_precision: Dict[str, Dict[str, object]] = {}
    extraction_seconds: Dict[str, Dict[str, float]] = {}
    for model_key in ("dinov2", "dinov3"):
        model, transform, metadata = _build_model_and_transform(
            model_key, device=device
        )
        model_metadata[model_key] = metadata
        numerical_precision[model_key] = _numerical_precision_probe(
            model=model,
            transform=transform,
            classification_root=Path(args.classification_root),
            device=device,
            model_metadata=metadata,
        )
        aligned_by_model[model_key] = {}
        extraction_seconds[model_key] = {}
        for split in ("train", "val"):
            split_start = time.perf_counter()
            extracted = _extract_split(
                model=model,
                transform=transform,
                classification_root=Path(args.classification_root),
                split=split,
                device=device,
                batch_size=int(args.batch_size),
                workers=int(args.workers),
                amp=False,
                model_metadata=metadata,
                progress_label=f"{model_key}-fp32-dense",
            )
            extraction_seconds[model_key][split] = float(
                time.perf_counter() - split_start
            )
            if list(extracted["folder_classes"]) != classes_by_split[split]:
                raise ValueError(f"ImageFolder class order changed for {model_key}/{split}")
            aligned_by_model[model_key][split] = _align_payload(
                extracted, plans[split]
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train = aligned_by_model["dinov3"]["train"]
    val = aligned_by_model["dinov3"]["val"]
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    train_group_array = np.asarray(train["source_stems"], dtype=object)
    class_names = list(train["class_names"])
    for split, labels in (("train", train_labels), ("val", val_labels)):
        control = aligned_by_model["dinov2"][split]
        candidate = aligned_by_model["dinov3"][split]
        for key in ("sample_index", "paths", "source_stems"):
            if not np.array_equal(np.asarray(control[key]), np.asarray(candidate[key])):
                raise ValueError(f"DINOv2/DINOv3 {split} identity drift for {key}")
        if not np.array_equal(np.asarray(control["labels"], dtype=np.int64), labels):
            raise ValueError(f"DINOv2/DINOv3 {split} target drift")
    descriptor_finite = all(
        np.isfinite(
            np.asarray(aligned_by_model[model_key][split][name], dtype=np.float32)
        ).all()
        for model_key in ("dinov2", "dinov3")
        for split in ("train", "val")
        for name in FEATURE_NAMES
    )
    if not descriptor_finite:
        raise ValueError("Frozen DINO descriptor contains non-finite values")
    dense_effective_rank: Dict[str, float] = {}
    for model_key in ("dinov2", "dinov3"):
        dense_train = np.asarray(
            aligned_by_model[model_key]["train"]["cls_plus_dense_patch"],
            dtype=np.float32,
        )[:, EMBEDDING_DIMENSION:]
        dense_standardized = StandardScaler().fit_transform(dense_train)
        dense_effective_rank[model_key] = _projected_effective_rank(
            dense_standardized,
            seed=SEED + 991,
        )
        del dense_standardized

    readout_fits: Dict[str, Dict[str, Dict[str, object]]] = {}
    for model_key in ("dinov2", "dinov3"):
        readout_fits[model_key] = {}
        for name in FEATURE_NAMES:
            print(f"fitting matched {model_key} FP32 readouts: {name}", flush=True)
            readout_fits[model_key][name] = fit_grouped_scattering_readouts(
                train_features=np.asarray(
                    aligned_by_model[model_key]["train"][name], dtype=np.float32
                ),
                train_labels=train_labels,
                train_groups=train_group_array,
                val_features=np.asarray(
                    aligned_by_model[model_key]["val"][name], dtype=np.float32
                ),
                val_labels=val_labels,
                class_names=class_names,
                folds=int(args.folds),
                seed=SEED,
            )
            _write_readout_metrics(
                output_dir / f"{model_key}_{name}_readout_metrics.csv",
                readout_fits[model_key][name]["readouts"],
            )

    model_probabilities: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for model_key in ("dinov2", "dinov3"):
        model_probabilities[model_key] = {"train": {}, "val": {}}
        for name in FEATURE_NAMES:
            result = readout_fits[model_key][name]["readouts"][PRIMARY_READOUT]
            model_probabilities[model_key]["train"][name] = np.asarray(
                result["oof_probabilities"], dtype=np.float32
            )
            model_probabilities[model_key]["val"][name] = np.asarray(
                result["val_probabilities"], dtype=np.float32
            )
    dinov2_train = model_probabilities["dinov2"]["train"]
    dinov2_val = model_probabilities["dinov2"]["val"]
    dinov3_train = model_probabilities["dinov3"]["train"]
    dinov3_val = model_probabilities["dinov3"]["val"]
    historical_dinov2_train = {
        name: _load_variant_probabilities(
            Path(args.dinov2_audit_dir) / "train_oof_prediction_audit.csv",
            variant=name,
            expected_labels=train_labels,
            expected_sample_index=np.asarray(train["sample_index"], dtype=np.int64),
            expected_paths=np.asarray(train["paths"], dtype=object),
            class_count=len(class_names),
        )
        for name in FEATURE_NAMES
    }
    historical_dinov2_val = {
        name: _load_variant_probabilities(
            Path(args.dinov2_audit_dir) / "val_prediction_audit.csv",
            variant=name,
            expected_labels=val_labels,
            expected_sample_index=np.asarray(val["sample_index"], dtype=np.int64),
            expected_paths=np.asarray(val["paths"], dtype=object),
            class_count=len(class_names),
        )
        for name in FEATURE_NAMES
    }
    dinov2_selection = select_oof_feature(
        dinov2_train, labels=train_labels, class_names=class_names
    )
    dinov3_selection = select_oof_feature(
        dinov3_train, labels=train_labels, class_names=class_names
    )
    historical_dinov2_selection = select_oof_feature(
        historical_dinov2_train, labels=train_labels, class_names=class_names
    )
    control_name = str(dinov2_selection["selected_feature"])
    candidate_name = str(dinov3_selection["selected_feature"])
    historical_control_name = str(
        historical_dinov2_selection["selected_feature"]
    )
    control_train = dinov2_train[control_name]
    control_val = dinov2_val[control_name]
    candidate_train = dinov3_train[candidate_name]
    candidate_val = dinov3_val[candidate_name]
    control_oof_metrics = _classification_metrics(
        train_labels, control_train, class_names=class_names
    )
    control_val_metrics = _classification_metrics(
        val_labels, control_val, class_names=class_names
    )
    candidate_oof_metrics = _classification_metrics(
        train_labels, candidate_train, class_names=class_names
    )
    candidate_val_metrics = _classification_metrics(
        val_labels, candidate_val, class_names=class_names
    )
    historical_control_train = historical_dinov2_train[historical_control_name]
    historical_control_val = historical_dinov2_val[historical_control_name]
    historical_control_oof_metrics = _classification_metrics(
        train_labels, historical_control_train, class_names=class_names
    )
    historical_control_val_metrics = _classification_metrics(
        val_labels, historical_control_val, class_names=class_names
    )
    prior_control = dinov2_summary["matched_readouts"][historical_control_name][
        "readouts"
    ][PRIMARY_READOUT]
    for split, current, prior in (
        ("oof", historical_control_oof_metrics, prior_control["oof_metrics"]),
        ("val", historical_control_val_metrics, prior_control["val_metrics"]),
    ):
        for key in ("macro_f1", "focus_f1", "focus_precision", "focus_recall"):
            if not np.isclose(float(current[key]), float(prior[key]), atol=1e-7, rtol=0.0):
                raise ValueError(f"DINOv2 {split} metric drift for {key}")
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
    keeper_train_metrics = _classification_metrics(
        train_labels, base_train, class_names=class_names
    )
    keeper_val_metrics = _classification_metrics(
        val_labels, base_val, class_names=class_names
    )
    fold_comparison = _fold_comparison(
        labels=train_labels,
        groups=train_group_array,
        control_probabilities=control_train,
        candidate_probabilities=candidate_train,
        class_names=class_names,
    )
    control_to_candidate_train = _transition_summary(
        train_labels, control_train, candidate_train
    )
    control_to_candidate_val = _transition_summary(
        val_labels, control_val, candidate_val
    )
    keeper_to_candidate_train = _transition_summary(
        train_labels, base_train, candidate_train
    )
    keeper_to_candidate_val = _transition_summary(
        val_labels, base_val, candidate_val
    )
    historical_to_fp32_train = _transition_summary(
        train_labels, historical_control_train, control_train
    )
    historical_to_fp32_val = _transition_summary(
        val_labels, historical_control_val, control_val
    )
    train_direction_auc = _class1_error_direction_auc(
        train_labels, control_train, candidate_train
    )
    val_direction_auc = _class1_error_direction_auc(
        val_labels, control_val, candidate_val
    )
    keeper_direction_auc = _class1_error_direction_auc(
        val_labels, base_val, candidate_val
    )
    control_oracle = _binary_focus_oracle(val_labels, control_val, candidate_val)
    keeper_oracle = _binary_focus_oracle(val_labels, base_val, candidate_val)
    gate = assess_dinov3_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_groups=int(plans["train"]["source_groups"]),
        cross_split_source_overlap=cross_split_source_overlap,
        descriptor_finite=descriptor_finite,
        dense_effective_rank=dense_effective_rank["dinov3"],
        energy_entropy=float(val["energy_stats"]["entropy_mean"]),
        energy_border_mass=float(val["energy_stats"]["border_mass_mean"]),
        control_oof_metrics=control_oof_metrics,
        candidate_oof_metrics=candidate_oof_metrics,
        control_val_metrics=control_val_metrics,
        candidate_val_metrics=candidate_val_metrics,
        keeper_val_metrics=keeper_val_metrics,
        fold_comparison=fold_comparison,
        keeper_oracle=keeper_oracle,
        train_direction_auc=train_direction_auc,
        val_direction_auc=val_direction_auc,
        keeper_transitions=keeper_to_candidate_val,
    )

    train_variants = {
        "dinov2_cls": dinov2_train["cls"],
        "dinov2_dense": dinov2_train["cls_plus_dense_patch"],
        "dinov2_selected": control_train,
        "dinov2_amp_reference": historical_control_train,
        "dinov3_cls": dinov3_train["cls"],
        "dinov3_dense": dinov3_train["cls_plus_dense_patch"],
        "dinov3_selected": candidate_train,
    }
    val_variants = {
        "dinov2_cls": dinov2_val["cls"],
        "dinov2_dense": dinov2_val["cls_plus_dense_patch"],
        "dinov2_selected": control_val,
        "dinov2_amp_reference": historical_control_val,
        "dinov3_cls": dinov3_val["cls"],
        "dinov3_dense": dinov3_val["cls_plus_dense_patch"],
        "dinov3_selected": candidate_val,
    }
    _write_prediction_audit(
        output_dir / "train_oof_prediction_audit.csv",
        labels=train_labels,
        sample_index=np.asarray(train["sample_index"], dtype=np.int64),
        paths=np.asarray(train["paths"], dtype=object),
        base_probabilities=base_train,
        variants=train_variants,
    )
    _write_prediction_audit(
        output_dir / "val_prediction_audit.csv",
        labels=val_labels,
        sample_index=np.asarray(val["sample_index"], dtype=np.int64),
        paths=np.asarray(val["paths"], dtype=object),
        base_probabilities=base_val,
        variants=val_variants,
    )
    _write_preview(
        output_dir / "dinov3_patch_energy_preview.png",
        preview=aligned_by_model["dinov3"]["val"]["preview"],
        folder_classes=aligned_by_model["dinov3"]["val"]["folder_classes"],
        energy_label="dinov3_patch_deviation_energy",
    )
    if bool(args.save_descriptors):
        for model_key in ("dinov2", "dinov3"):
            for split in ("train", "val"):
                _save_descriptor_cache(
                    output_dir / f"{model_key}_{split}_descriptors.npz",
                    payload=aligned_by_model[model_key][split],
                )

    compact_readouts = {
        model_key: {
            feature_name: {
                "dimension": int(
                    np.asarray(
                        aligned_by_model[model_key]["train"][feature_name]
                    ).shape[1]
                ),
                "final_pca_explained_variance": float(
                    readout_fits[model_key][feature_name][
                        "final_pca_explained_variance"
                    ]
                ),
                "fold_telemetry": readout_fits[model_key][feature_name]["folds"],
                "final_fit_elapsed_seconds": float(
                    readout_fits[model_key][feature_name]["final_fit_elapsed_seconds"]
                ),
                "readouts": {
                    name: _compact_readout(
                        readout_fits[model_key][feature_name]["readouts"][name]
                    )
                    for name in READOUT_NAMES
                },
            }
            for feature_name in FEATURE_NAMES
        }
        for model_key in ("dinov2", "dinov3")
    }
    summary = {
        "mode": "dinov3_dense_patch_readiness",
        "guardrail": (
            "Frozen DINOv3-S versus freshly re-extracted FP32 DINOv2-S, using "
            "identical fixed descriptors, PCA/readout, source folds, and OOF-only "
            "feature selection. The retained AMP audit is provenance-only. No "
            "test, raw-data edit, target manifest, checkpoint, or image training."
        ),
        "literature": list(LITERATURE),
        "protocol": preflight["protocol"],
        "models": model_metadata,
        "numerical_precision_preflight": numerical_precision,
        "runtime": {
            "device": str(device),
            "descriptor_precision": "fp32",
            "amp": False,
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "torch_threads": int(args.torch_threads),
            "extraction_seconds": extraction_seconds,
            "peak_cuda_memory_mib": {
                model_key: {
                    split: float(
                        aligned_by_model[model_key][split]["peak_cuda_memory_mib"]
                    )
                    for split in ("train", "val")
                }
                for model_key in ("dinov2", "dinov3")
            },
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "cross_split_source_overlap": cross_split_source_overlap,
        "dinov2_reference": dinov2_inputs,
        "descriptor": {
            "finite": descriptor_finite,
            "dense_effective_rank_projected128": dense_effective_rank,
            "energy_stats": {
                model_key: {
                    split: aligned_by_model[model_key][split]["energy_stats"]
                    for split in ("train", "val")
                }
                for model_key in ("dinov2", "dinov3")
            },
            "saved": bool(args.save_descriptors),
        },
        "selection": {
            "dinov2": dinov2_selection,
            "dinov3": dinov3_selection,
            "historical_dinov2_amp": historical_dinov2_selection,
            "validation_used": False,
        },
        "keeper": {
            "train_metrics": keeper_train_metrics,
            "val_metrics": keeper_val_metrics,
        },
        "dinov2_selected": {
            "feature": control_name,
            "precision": "fp32",
            "oof_metrics": control_oof_metrics,
            "val_metrics": control_val_metrics,
        },
        "historical_dinov2_amp_reference": {
            "feature": historical_control_name,
            "oof_metrics": historical_control_oof_metrics,
            "val_metrics": historical_control_val_metrics,
            "amp_to_fp32_train_oof": historical_to_fp32_train,
            "amp_to_fp32_val": historical_to_fp32_val,
        },
        "matched_readouts": compact_readouts,
        "dinov3_selected": {
            "feature": candidate_name,
            "precision": "fp32",
            "oof_metrics": candidate_oof_metrics,
            "val_metrics": candidate_val_metrics,
        },
        "fold_comparison": fold_comparison,
        "class1_complementarity": {
            "control_candidate_binary_oracle_val": control_oracle,
            "keeper_candidate_binary_oracle_val": keeper_oracle,
            "control_error_direction_auc_train_oof": train_direction_auc,
            "control_error_direction_auc_val": val_direction_auc,
            "keeper_error_direction_auc_val": keeper_direction_auc,
            "control_to_candidate_train_oof": control_to_candidate_train,
            "control_to_candidate_val": control_to_candidate_val,
            "keeper_to_candidate_train_oof": keeper_to_candidate_train,
            "keeper_to_candidate_val": keeper_to_candidate_val,
            "historical_dinov2_amp_to_fp32_train_oof": historical_to_fp32_train,
            "historical_dinov2_amp_to_fp32_val": historical_to_fp32_val,
        },
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
        "image_model_trained": False,
        "decision": (
            "Proceed to one fixed short DINOv3 representation-transfer smoke."
            if bool(gate["smoke_permission"])
            else "Reject DINOv3 representation transfer before image-model training."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# DINOv3 frozen-representation readiness audit\n\n"
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`; test is closed.\n"
        f"- Selected DINOv2 feature: `{control_name}`; OOF macro/class1 "
        f"`{control_oof_metrics['macro_f1']:.6f}/{control_oof_metrics['focus_f1']:.6f}`; "
        f"validation `{control_val_metrics['macro_f1']:.6f}/{control_val_metrics['focus_f1']:.6f}`.\n"
        f"- Selected DINOv3 feature: `{candidate_name}`; OOF macro/class1 "
        f"`{candidate_oof_metrics['macro_f1']:.6f}/{candidate_oof_metrics['focus_f1']:.6f}`; "
        f"validation `{candidate_val_metrics['macro_f1']:.6f}/{candidate_val_metrics['focus_f1']:.6f}`.\n"
        f"- Keeper validation macro/class1: `{keeper_val_metrics['macro_f1']:.6f}/{keeper_val_metrics['focus_f1']:.6f}`.\n"
        f"- DINOv3 class1 fold wins: `{gate['observed']['class1_fold_wins']}/5`.\n"
        f"- FN-vs-FP direction AUROC train/val: `{train_direction_auc:.6f}/{val_direction_auc:.6f}`.\n"
        f"- Smoke permission: `{bool(gate['smoke_permission'])}`.\n"
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`.\n\n"
        "Feature choice is fixed by train OOF only; validation is used only for the locked gate.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(
        output_dir,
        mode="dinov3_dense_patch_readiness_evidence_manifest",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(args)
    if summary.get("mode") == "dinov3_dense_patch_readiness_preflight":
        payload = summary
    else:
        payload = {
            "output_dir": str(Path(args.output_dir).resolve()),
            "selected_feature": summary["dinov3_selected"]["feature"],
            "smoke_permission": bool(summary["gate"]["smoke_permission"]),
            "failed_checks": summary["gate"]["failed_checks"],
            "decision": summary["decision"],
        }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

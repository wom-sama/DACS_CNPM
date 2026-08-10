from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.models.model import build_model_from_checkpoint
from trkh.tools.audit_dinov3_pair_patch_stat_readiness import (
    FOCUS_CLASS,
    RIVAL_CLASSES,
    _assert_output_outside_train,
    _binary_metrics,
    _model_contract,
    _positive_probabilities,
    _sha256,
    _write_csv,
    normalized_source_group,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)


MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
FOLDS = 5
SEED = 20260731
PCA_COMPONENTS = 128
LOGISTIC_C = 0.30
SPATIAL_POOL_SIZE = 2


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only OOF precheck for raw DINOv3 patch moments. "
            "It compares capacity-matched pooled and pooled+dense descriptors "
            "without constructing validation or test datasets."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def raw_patch_moment_descriptors(
    tokens: Tensor,
    *,
    prefix_tokens: int,
    spatial_pool_size: int = SPATIAL_POOL_SIZE,
) -> Dict[str, Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,D], got {tuple(tokens.shape)}")
    prefix = int(prefix_tokens)
    pool_size = int(spatial_pool_size)
    if prefix < 1 or prefix >= int(tokens.size(1)) or pool_size < 1:
        raise ValueError("prefix_tokens/spatial_pool_size are invalid")
    patches = tokens.float()[:, prefix:]
    side = int(round(float(patches.size(1)) ** 0.5))
    if side * side != int(patches.size(1)):
        raise ValueError(
            f"Patch count {int(patches.size(1))} is not a square spatial grid"
        )
    pooled = patches.mean(dim=1)
    patch_std = patches.std(dim=1, unbiased=False)
    patch_grid = patches.reshape(
        int(patches.size(0)), side, side, int(patches.size(2))
    ).permute(0, 3, 1, 2)
    spatial_means = F.adaptive_avg_pool2d(
        patch_grid,
        output_size=(pool_size, pool_size),
    ).flatten(1)
    dense = torch.cat((pooled, patch_std, spatial_means), dim=1)
    return {
        "pooled": pooled,
        "patch_std": patch_std,
        "spatial_means": spatial_means,
        "pooled_plus_patch_moments": dense,
        "patch_grid": torch.as_tensor(
            [side, side], device=tokens.device, dtype=torch.int64
        ),
    }


def _extract_descriptors(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    if not callable(sample_paths_fn):
        raise TypeError("Classification dataset must expose sample_paths()")
    paths = [str(path) for path in sample_paths_fn()]
    pooled_batches: List[np.ndarray] = []
    dense_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    maximum_pool_parity_error = 0.0
    patch_grid = (0, 0)
    seen = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc="dinov3-raw-patch-fp32", dynamic_ncols=True)
        for images, labels, _metadata in iterator:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            raw_tokens = model.forward_features(images)
            if not torch.is_tensor(raw_tokens):
                raise TypeError("DINOv3 forward_features must return one tensor")
            descriptors = raw_patch_moment_descriptors(
                raw_tokens,
                prefix_tokens=int(getattr(model, "num_prefix_tokens", 0) or 0),
            )
            model_pooled = model.forward_head(raw_tokens, pre_logits=True).float()
            descriptor_pooled = descriptors["pooled"]
            parity_error = float(
                (model_pooled - descriptor_pooled).abs().max().detach().cpu().item()
            )
            maximum_pool_parity_error = max(maximum_pool_parity_error, parity_error)
            if parity_error > 1e-6:
                raise RuntimeError(
                    "Raw patch mean does not match the deployed average pool: "
                    f"max_abs={parity_error}"
                )
            grid_tensor = descriptors["patch_grid"].detach().cpu().tolist()
            patch_grid = (int(grid_tensor[0]), int(grid_tensor[1]))
            pooled_batches.append(model_pooled.detach().cpu().numpy())
            dense_batches.append(
                descriptors["pooled_plus_patch_moments"].detach().cpu().numpy()
            )
            label_batches.append(labels.detach().cpu().numpy())
            seen += int(labels.numel())
    if seen != len(paths):
        raise RuntimeError(f"Descriptor rows/path rows differ: {seen} != {len(paths)}")
    payload: Dict[str, object] = {
        "pooled": np.concatenate(pooled_batches, axis=0).astype(np.float32, copy=False),
        "dense": np.concatenate(dense_batches, axis=0).astype(np.float32, copy=False),
        "labels": np.concatenate(label_batches, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "source_groups": np.asarray(
            [normalized_source_group(path) for path in paths],
            dtype=object,
        ),
        "patch_grid": list(patch_grid),
        "maximum_pool_parity_error": float(maximum_pool_parity_error),
    }
    for key in ("pooled", "dense", "labels", "paths", "source_groups"):
        if int(np.asarray(payload[key]).shape[0]) != seen:
            raise RuntimeError(f"Descriptor key {key} has inconsistent row count")
    if not np.isfinite(np.asarray(payload["pooled"], dtype=np.float32)).all():
        raise ValueError("Pooled descriptors contain non-finite values")
    if not np.isfinite(np.asarray(payload["dense"], dtype=np.float32)).all():
        raise ValueError("Dense descriptors contain non-finite values")
    return payload


def assign_global_source_folds(
    labels: np.ndarray,
    source_groups: np.ndarray,
    *,
    folds: int = FOLDS,
    seed: int = SEED,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    if labels.size != groups.size:
        raise ValueError("labels/source_groups must have equal length")
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    assignments = np.full(labels.size, -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    for fold_index, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(labels.size, dtype=np.int8), labels, groups=groups)
    ):
        overlap = set(groups[fit_indices].tolist()) & set(
            groups[holdout_indices].tolist()
        )
        if bool((assignments[holdout_indices] >= 0).any()):
            raise RuntimeError("Global source folds assign one row more than once")
        assignments[holdout_indices] = int(fold_index)
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_samples": int(fit_indices.size),
                "holdout_samples": int(holdout_indices.size),
                "fit_source_groups": int(np.unique(groups[fit_indices]).size),
                "holdout_source_groups": int(np.unique(groups[holdout_indices]).size),
                "source_overlap": int(len(overlap)),
                "holdout_class_counts": np.bincount(
                    labels[holdout_indices], minlength=5
                ).astype(int).tolist(),
            }
        )
    if bool((assignments < 0).any()):
        raise RuntimeError("Global source fold assignment is incomplete")
    return assignments, fold_rows


def _capacity_matched_readout(seed: int):
    return make_pipeline(
        StandardScaler(),
        PCA(
            n_components=PCA_COMPONENTS,
            whiten=True,
            svd_solver="randomized",
            random_state=int(seed),
        ),
        LogisticRegression(
            C=LOGISTIC_C,
            class_weight="balanced",
            max_iter=500,
            random_state=int(seed),
            solver="lbfgs",
        ),
    )


def run_matched_pair_oof(
    *,
    control_features: np.ndarray,
    candidate_features: np.ndarray,
    labels: np.ndarray,
    source_groups: np.ndarray,
    fold_assignments: np.ndarray,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    control = np.asarray(control_features, dtype=np.float32)
    candidate = np.asarray(candidate_features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    assignments = np.asarray(fold_assignments, dtype=np.int64).reshape(-1)
    if not (
        control.shape[0]
        == candidate.shape[0]
        == labels.size
        == groups.size
        == assignments.size
    ):
        raise ValueError("Matched OOF arrays have inconsistent row counts")
    pair_rows: List[Dict[str, object]] = []
    fold_metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for rival in RIVAL_CLASSES:
        pair_mask = np.logical_or(labels == int(rival), labels == FOCUS_CLASS)
        pair_indices = np.flatnonzero(pair_mask)
        pair_targets = (labels[pair_indices] == FOCUS_CLASS).astype(np.int64)
        control_oof = np.full(pair_indices.size, np.nan, dtype=np.float64)
        candidate_oof = np.full(pair_indices.size, np.nan, dtype=np.float64)
        pair_folds = assignments[pair_indices]
        for fold_index in range(FOLDS):
            fit_local = np.flatnonzero(pair_folds != int(fold_index))
            hold_local = np.flatnonzero(pair_folds == int(fold_index))
            if fit_local.size <= PCA_COMPONENTS or hold_local.size == 0:
                raise ValueError(
                    f"Insufficient pair support for fold {fold_index}, rival {rival}"
                )
            fit_sources = set(groups[pair_indices[fit_local]].tolist())
            hold_sources = set(groups[pair_indices[hold_local]].tolist())
            overlap = fit_sources & hold_sources
            control_model = clone(_capacity_matched_readout(seed=SEED + fold_index))
            candidate_model = clone(_capacity_matched_readout(seed=SEED + fold_index))
            control_model.fit(
                control[pair_indices[fit_local]], pair_targets[fit_local]
            )
            candidate_model.fit(
                candidate[pair_indices[fit_local]], pair_targets[fit_local]
            )
            control_probabilities = _positive_probabilities(
                control_model, control[pair_indices[hold_local]]
            )
            candidate_probabilities = _positive_probabilities(
                candidate_model, candidate[pair_indices[hold_local]]
            )
            control_oof[hold_local] = control_probabilities
            candidate_oof[hold_local] = candidate_probabilities
            control_metrics = _binary_metrics(
                pair_targets[hold_local], control_probabilities
            )
            candidate_metrics = _binary_metrics(
                pair_targets[hold_local], candidate_probabilities
            )
            fold_row: Dict[str, object] = {
                "pair": f"{int(rival)}-1",
                "rival_class": int(rival),
                "fold": int(fold_index),
                "fit_samples": int(fit_local.size),
                "holdout_samples": int(hold_local.size),
                "source_overlap": int(len(overlap)),
            }
            for prefix, metrics in (
                ("control", control_metrics),
                ("candidate", candidate_metrics),
            ):
                for key, value in metrics.items():
                    fold_row[f"{prefix}_{key}"] = float(value)
            fold_row["delta_auroc"] = float(
                candidate_metrics["auroc"] - control_metrics["auroc"]
            )
            fold_row["delta_balanced_accuracy"] = float(
                candidate_metrics["balanced_accuracy"]
                - control_metrics["balanced_accuracy"]
            )
            fold_metric_rows.append(fold_row)
        if not np.isfinite(control_oof).all() or not np.isfinite(candidate_oof).all():
            raise RuntimeError(f"OOF probabilities are incomplete for rival {rival}")
        control_metrics = _binary_metrics(pair_targets, control_oof)
        candidate_metrics = _binary_metrics(pair_targets, candidate_oof)
        pair_row: Dict[str, object] = {
            "pair": f"{int(rival)}-1",
            "rival_class": int(rival),
            "samples": int(pair_targets.size),
            "class1_samples": int(pair_targets.sum()),
            "rival_samples": int((pair_targets == 0).sum()),
            "source_groups": int(np.unique(groups[pair_indices]).size),
        }
        for prefix, metrics in (
            ("control", control_metrics),
            ("candidate", candidate_metrics),
        ):
            for key, value in metrics.items():
                pair_row[f"{prefix}_{key}"] = float(value)
        for key in (
            "balanced_accuracy",
            "auroc",
            "precision_class1",
            "recall_class1",
            "specificity_rival",
            "f1_class1",
        ):
            pair_row[f"delta_{key}"] = float(
                candidate_metrics[key] - control_metrics[key]
            )
        pair_rows.append(pair_row)
        for local_index, sample_index in enumerate(pair_indices):
            prediction_rows.append(
                {
                    "sample_index": int(sample_index),
                    "source_group": str(groups[sample_index]),
                    "fold": int(pair_folds[local_index]),
                    "target_index": int(labels[sample_index]),
                    "pair": f"{int(rival)}-1",
                    "control_probability_class1": float(control_oof[local_index]),
                    "candidate_probability_class1": float(candidate_oof[local_index]),
                }
            )
    return pair_rows, fold_metric_rows, prediction_rows


def assess_raw_patch_readiness(
    *,
    pair_rows: Sequence[Mapping[str, object]],
    fold_metric_rows: Sequence[Mapping[str, object]],
    global_fold_rows: Sequence[Mapping[str, object]],
    train_samples: int,
    source_groups: int,
    pool_parity_error: float,
) -> Dict[str, object]:
    pairs = [dict(row) for row in pair_rows]
    fold_metrics = [dict(row) for row in fold_metric_rows]
    global_folds = [dict(row) for row in global_fold_rows]
    thresholds = {
        "required_train_samples": 8278,
        "min_source_groups": 7000,
        "required_pair_folds": len(RIVAL_CLASSES) * FOLDS,
        "max_source_overlap": 0,
        "max_pool_parity_error": 1e-6,
        "min_mean_auroc_gain": 0.010,
        "min_positive_pair_fold_gains": 10,
        "min_pairs_with_auroc_gain": 2,
        "min_pair_auroc_gain": 0.010,
        "max_pair_auroc_loss": 0.005,
        "min_pairs_with_f1_gain": 2,
        "min_pair_f1_gain": 0.010,
        "max_pair_recall_loss": 0.020,
        "min_class1_tp_retention": 0.970,
        "min_rival_fp_reduction": 0.100,
    }
    mean_auroc_gain = float(
        np.mean([float(row["delta_auroc"]) for row in pairs])
    )
    positive_fold_gains = int(
        sum(float(row["delta_auroc"]) > 0.0 for row in fold_metrics)
    )
    pairs_with_auroc_gain = int(
        sum(
            float(row["delta_auroc"])
            >= float(thresholds["min_pair_auroc_gain"])
            for row in pairs
        )
    )
    pairs_with_f1_gain = int(
        sum(
            float(row["delta_f1_class1"])
            >= float(thresholds["min_pair_f1_gain"])
            for row in pairs
        )
    )
    minimum_pair_auroc_delta = float(
        min(float(row["delta_auroc"]) for row in pairs)
    )
    minimum_pair_recall_delta = float(
        min(float(row["delta_recall_class1"]) for row in pairs)
    )
    control_tp = float(sum(float(row["control_tp"]) for row in pairs))
    candidate_tp = float(sum(float(row["candidate_tp"]) for row in pairs))
    control_fp = float(sum(float(row["control_fp"]) for row in pairs))
    candidate_fp = float(sum(float(row["candidate_fp"]) for row in pairs))
    tp_retention = candidate_tp / max(1.0, control_tp)
    fp_reduction = (control_fp - candidate_fp) / max(1.0, control_fp)
    maximum_overlap = int(
        max(
            [int(row["source_overlap"]) for row in global_folds]
            + [int(row["source_overlap"]) for row in fold_metrics]
        )
    )
    observed = {
        "mean_auroc_gain": mean_auroc_gain,
        "positive_pair_fold_gains": positive_fold_gains,
        "pairs_with_auroc_gain": pairs_with_auroc_gain,
        "pairs_with_f1_gain": pairs_with_f1_gain,
        "minimum_pair_auroc_delta": minimum_pair_auroc_delta,
        "minimum_pair_recall_delta": minimum_pair_recall_delta,
        "class1_tp_retention": float(tp_retention),
        "rival_fp_reduction": float(fp_reduction),
        "control_tp": int(control_tp),
        "candidate_tp": int(candidate_tp),
        "control_fp": int(control_fp),
        "candidate_fp": int(candidate_fp),
        "maximum_source_overlap": maximum_overlap,
        "pool_parity_error": float(pool_parity_error),
    }
    checks = {
        "canonical_train_support": int(train_samples)
        == int(thresholds["required_train_samples"]),
        "source_group_support": int(source_groups)
        >= int(thresholds["min_source_groups"]),
        "complete_pair_fold_coverage": len(fold_metrics)
        == int(thresholds["required_pair_folds"]),
        "source_group_folds_disjoint": maximum_overlap
        <= int(thresholds["max_source_overlap"]),
        "deployed_pool_parity": float(pool_parity_error)
        <= float(thresholds["max_pool_parity_error"]),
        "mean_auroc_gain": mean_auroc_gain
        >= float(thresholds["min_mean_auroc_gain"]),
        "fold_direction_stability": positive_fold_gains
        >= int(thresholds["min_positive_pair_fold_gains"]),
        "pair_auroc_support": pairs_with_auroc_gain
        >= int(thresholds["min_pairs_with_auroc_gain"]),
        "no_pair_auroc_collapse": minimum_pair_auroc_delta
        >= -float(thresholds["max_pair_auroc_loss"]),
        "pair_f1_support": pairs_with_f1_gain
        >= int(thresholds["min_pairs_with_f1_gain"]),
        "class1_recall_protected": minimum_pair_recall_delta
        >= -float(thresholds["max_pair_recall_loss"]),
        "class1_tp_retained": tp_retention
        >= float(thresholds["min_class1_tp_retention"]),
        "rival_false_positives_reduced": fp_reduction
        >= float(thresholds["min_rival_fp_reduction"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "raw_patch_moment_specialist_ready": bool(ready),
        "implementation_permission": bool(ready),
        "smoke_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, object],
    model_contract: Mapping[str, object],
) -> Dict[str, object]:
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint model_config is missing")
    observed = {
        "model_type": str(model_config.get("model_type", "")),
        "timm_model_name": str(model_config.get("timm_model_name", "")),
        "checkpoint_weight_source": str(
            checkpoint.get("checkpoint_weight_source", "")
        ),
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "global_pool": str(model_contract.get("global_pool", "")),
    }
    expected = {
        "model_type": "timm_classifier",
        "timm_model_name": MODEL_NAME,
        "checkpoint_weight_source": "ema",
        "epoch": 4,
        "best_epoch": 4,
        "global_pool": "avg",
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"Locked B2 checkpoint contract mismatch: {mismatches}")
    return {"expected": expected, "observed": observed, "mismatches": mismatches}


def run_precheck(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    data_path = Path(args.data).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for required in (data_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model_contract = _model_contract(model)
    checkpoint_contract = _validate_checkpoint_contract(checkpoint, model_contract)
    dataset, class_names = _build_dataset(
        data_yaml=data_path,
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    if len(class_names) != 5:
        raise ValueError(f"Locked precheck requires five classes, got {len(class_names)}")
    _assert_output_outside_train(output_dir, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema_version": 1,
        "mode": "dinov3_raw_patch_moments_train_only_oof",
        "data": str(data_path),
        "data_sha256": _sha256(data_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "output_dir": str(output_dir),
        "class_names": [str(name) for name in class_names],
        "train_samples": int(len(dataset)),
        "model_contract": model_contract,
        "checkpoint_contract": checkpoint_contract,
        "locked_protocol": {
            "focus_class": FOCUS_CLASS,
            "rival_classes": list(RIVAL_CLASSES),
            "folds": FOLDS,
            "seed": SEED,
            "control_descriptor": "deployed_average_pooled_patch_tokens_384d",
            "candidate_descriptor": (
                "pooled_384d_plus_patch_std_384d_plus_2x2_spatial_means_1536d"
            ),
            "capacity_match": (
                "fold_fit_standard_scaler_plus_whitened_randomized_pca128"
            ),
            "logistic_c": LOGISTIC_C,
            "threshold": 0.5,
            "fp32_extraction": True,
            "candidate_selection_uses_validation": False,
            "descriptor_or_threshold_sweep": False,
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "raw_dataset_touched": False,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if bool(args.preflight_only):
        return preflight

    device = _resolve_device(str(args.device or ""))
    model.to(device=device, dtype=torch.float32)
    start = time.perf_counter()
    descriptors = _extract_descriptors(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    labels = np.asarray(descriptors["labels"], dtype=np.int64)
    groups = np.asarray(descriptors["source_groups"], dtype=object)
    assignments, global_fold_rows = assign_global_source_folds(labels, groups)
    pair_rows, fold_metric_rows, prediction_rows = run_matched_pair_oof(
        control_features=np.asarray(descriptors["pooled"], dtype=np.float32),
        candidate_features=np.asarray(descriptors["dense"], dtype=np.float32),
        labels=labels,
        source_groups=groups,
        fold_assignments=assignments,
    )
    readiness = assess_raw_patch_readiness(
        pair_rows=pair_rows,
        fold_metric_rows=fold_metric_rows,
        global_fold_rows=global_fold_rows,
        train_samples=int(labels.size),
        source_groups=int(np.unique(groups).size),
        pool_parity_error=float(descriptors["maximum_pool_parity_error"]),
    )
    path_values = np.asarray(descriptors["paths"], dtype=object)
    for row in prediction_rows:
        row["image_path"] = str(path_values[int(row["sample_index"])])
    assignment_rows = [
        {
            "sample_index": int(index),
            "image_path": str(path_values[index]),
            "source_group": str(groups[index]),
            "target_index": int(labels[index]),
            "fold": int(assignments[index]),
        }
        for index in range(labels.size)
    ]
    _write_csv(output_dir / "global_fold_assignments.csv", assignment_rows)
    _write_csv(output_dir / "pair_fold_metrics.csv", fold_metric_rows)
    _write_csv(output_dir / "pair_metrics.csv", pair_rows)
    _write_csv(output_dir / "train_oof_pair_predictions.csv", prediction_rows)
    summary = {
        **preflight,
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "feature_dimensions": {
            "control_before_pca": int(np.asarray(descriptors["pooled"]).shape[1]),
            "candidate_before_pca": int(np.asarray(descriptors["dense"]).shape[1]),
            "both_after_fold_fit_pca": PCA_COMPONENTS,
        },
        "patch_grid": list(descriptors["patch_grid"]),
        "maximum_pool_parity_error": float(
            descriptors["maximum_pool_parity_error"]
        ),
        "source_groups": int(np.unique(groups).size),
        "global_folds": global_fold_rows,
        "pair_results": pair_rows,
        "readiness": readiness,
        "artifacts": {
            "global_fold_assignments": str(
                output_dir / "global_fold_assignments.csv"
            ),
            "pair_fold_metrics": str(output_dir / "pair_fold_metrics.csv"),
            "pair_metrics": str(output_dir / "pair_metrics.csv"),
            "oof_predictions": str(output_dir / "train_oof_pair_predictions.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_precheck(args)
    readiness = summary.get("readiness")
    if isinstance(readiness, Mapping):
        output = {
            "raw_patch_moment_specialist_ready": bool(
                readiness.get("raw_patch_moment_specialist_ready", False)
            ),
            "failed_checks": list(readiness.get("failed_checks", [])),
            "observed": dict(readiness.get("observed", {})),
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    else:
        output = {
            "preflight_only": bool(args.preflight_only),
            "train_samples": int(summary.get("train_samples", 0)),
            "validation_split_used": bool(
                summary.get("validation_split_used", False)
            ),
            "test_split_used": bool(summary.get("test_split_used", False)),
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

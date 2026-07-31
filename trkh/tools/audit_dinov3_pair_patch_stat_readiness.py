from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.models.model import (
    build_model_from_checkpoint,
    patch_evidence_summary_features,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)


FOCUS_CLASS = 1
RIVAL_CLASSES = (0, 2, 4)
PATCH_MARGIN_PAIRS = ((0, 1), (2, 1), (4, 1))
FOLDS = 5
SEED = 42
TOP_K = 4
LOGISTIC_C = 0.10
SOURCE_BOX_SUFFIX = re.compile(r"_box\d+$", re.IGNORECASE)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only, source-grouped readiness audit for a lightweight DINOv3 "
            "pairwise patch-statistic specialist. The matched control uses pooled "
            "features only; validation and test are never opened."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def normalized_source_group(path: object) -> str:
    stem = Path(str(path)).stem.strip().casefold()
    return SOURCE_BOX_SUFFIX.sub("", stem)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_output_outside_train(output_dir: Path, dataset: Dataset) -> None:
    root = Path(getattr(dataset, "root_dir", "")).resolve()
    output = Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("output-dir must be outside the canonical train image tree")


def _model_contract(model: torch.nn.Module) -> Dict[str, object]:
    if not bool(getattr(model, "is_timm_classifier", False)):
        raise TypeError("This locked audit requires a TIMM classifier checkpoint")
    for attribute in ("forward_features", "forward_head", "head"):
        if not callable(getattr(model, attribute, None)):
            raise TypeError(f"TIMM classifier is missing callable {attribute}")
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 0) or 0)
    if prefix_tokens < 1:
        raise ValueError("DINOv3 audit requires at least one prefix token")
    classifier = getattr(model, "head")
    out_features = int(getattr(classifier, "out_features", 0) or 0)
    if out_features != 5:
        raise ValueError(f"Locked five-class audit received classifier width {out_features}")
    return {
        "class_name": type(model).__name__,
        "is_timm_classifier": True,
        "num_prefix_tokens": prefix_tokens,
        "global_pool": str(getattr(model, "global_pool", "")),
        "classifier_out_features": out_features,
    }


def _local_patch_statistics(local_logits: Tensor) -> Tensor:
    return patch_evidence_summary_features(
        local_logits,
        pairs=PATCH_MARGIN_PAIRS,
        top_k=TOP_K,
    )


def _extract_train_features(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
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
    dataset_paths = [str(path) for path in sample_paths_fn()]
    pooled_batches: List[np.ndarray] = []
    logit_batches: List[np.ndarray] = []
    patch_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    seen = 0
    token_shape: Optional[Tuple[int, int]] = None
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc="dinov3-patch-stat-train", dynamic_ncols=True)
        for images, labels, _metadata in iterator:
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            with autocast_context(device, bool(amp)):
                raw_tokens = model.forward_features(images)
                if not torch.is_tensor(raw_tokens) or raw_tokens.ndim != 3:
                    raise ValueError(
                        "DINOv3 forward_features must return tokens [B,N,D], got "
                        f"{getattr(raw_tokens, 'shape', None)}"
                    )
                prefix_tokens = int(getattr(model, "num_prefix_tokens", 0) or 0)
                patches = raw_tokens[:, prefix_tokens:]
                if patches.ndim != 3 or int(patches.size(1)) < TOP_K:
                    raise ValueError("DINOv3 patch tokens are absent or shorter than top-k")
                pooled = model.forward_head(raw_tokens, pre_logits=True)
                logits = model.forward_head(raw_tokens, pre_logits=False)
                if not torch.is_tensor(pooled) or pooled.ndim != 2:
                    raise ValueError("DINOv3 pre-logits must have shape [B,D]")
                if not torch.is_tensor(logits) or logits.shape != (images.size(0), 5):
                    raise ValueError("DINOv3 logits must have shape [B,5]")
                flat_local = model.head(
                    patches.reshape(int(patches.size(0) * patches.size(1)), int(patches.size(2)))
                )
                local_logits = flat_local.reshape(
                    int(patches.size(0)), int(patches.size(1)), 5
                )
                patch_statistics = _local_patch_statistics(local_logits)
            token_shape = (int(patches.size(1)), int(patches.size(2)))
            pooled_batches.append(pooled.detach().float().cpu().numpy())
            logit_batches.append(logits.detach().float().cpu().numpy())
            patch_batches.append(patch_statistics.detach().float().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            seen += int(labels.numel())
    if seen != len(dataset_paths):
        raise RuntimeError(f"Feature rows/path rows differ: {seen} != {len(dataset_paths)}")
    payload: Dict[str, object] = {
        "pooled": np.concatenate(pooled_batches, axis=0).astype(np.float32, copy=False),
        "global_logits": np.concatenate(logit_batches, axis=0).astype(np.float32, copy=False),
        "patch_statistics": np.concatenate(patch_batches, axis=0).astype(
            np.float32, copy=False
        ),
        "labels": np.concatenate(label_batches, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(dataset_paths, dtype=object),
        "source_groups": np.asarray(
            [normalized_source_group(path) for path in dataset_paths],
            dtype=object,
        ),
        "patch_token_shape": list(token_shape or (0, 0)),
    }
    row_count = int(np.asarray(payload["labels"]).shape[0])
    for key in ("pooled", "global_logits", "patch_statistics", "paths", "source_groups"):
        if int(np.asarray(payload[key]).shape[0]) != row_count:
            raise RuntimeError(f"Extracted key {key} has inconsistent row count")
    for key in ("pooled", "global_logits", "patch_statistics"):
        if not np.isfinite(np.asarray(payload[key], dtype=np.float32)).all():
            raise ValueError(f"Extracted feature block {key} contains non-finite values")
    if any(not str(value).strip() for value in np.asarray(payload["source_groups"])):
        raise ValueError("Source grouping produced an empty identifier")
    return payload


def _readout(seed: int = SEED):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=LOGISTIC_C,
            class_weight="balanced",
            max_iter=1000,
            random_state=int(seed),
            solver="liblinear",
        ),
    )


def _positive_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    estimator = list(model.named_steps.values())[-1]
    classes = np.asarray(getattr(estimator, "classes_", []), dtype=np.int64)
    matches = np.flatnonzero(classes == 1)
    if matches.size != 1:
        raise ValueError(f"Binary readout classes do not contain positive class 1: {classes}")
    return probabilities[:, int(matches[0])]


def _binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(np.logical_and(targets == 1, predictions == 1).sum())
    fp = int(np.logical_and(targets == 0, predictions == 1).sum())
    tn = int(np.logical_and(targets == 0, predictions == 0).sum())
    fn = int(np.logical_and(targets == 1, predictions == 0).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "precision_class1": float(precision),
        "recall_class1": float(recall),
        "specificity_rival": float(specificity),
        "f1_class1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _pair_oof_comparison(
    *,
    pooled_features: np.ndarray,
    patch_features: np.ndarray,
    labels: np.ndarray,
    source_groups: np.ndarray,
    rivals: Sequence[int] = RIVAL_CLASSES,
    folds: int = FOLDS,
    seed: int = SEED,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    pooled = np.asarray(pooled_features, dtype=np.float32)
    patch = np.asarray(patch_features, dtype=np.float32)
    if pooled.ndim != 2 or patch.ndim != 2 or pooled.shape[0] != labels.size:
        raise ValueError("OOF feature blocks must be 2D and aligned with labels")
    if patch.shape[0] != labels.size or groups.size != labels.size:
        raise ValueError("OOF patch/groups rows must align with labels")
    pair_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for rival in rivals:
        pair_mask = np.logical_or(labels == int(rival), labels == FOCUS_CLASS)
        pair_indices = np.flatnonzero(pair_mask)
        pair_targets = (labels[pair_indices] == FOCUS_CLASS).astype(np.int64)
        pair_groups = groups[pair_indices]
        control_features = pooled[pair_indices]
        candidate_features = np.concatenate(
            (control_features, patch[pair_indices]), axis=1
        ).astype(np.float32, copy=False)
        splitter = StratifiedGroupKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(seed),
        )
        split_indices = list(
            splitter.split(
                np.zeros(pair_targets.shape[0], dtype=np.int8),
                pair_targets,
                groups=pair_groups,
            )
        )
        control_oof = np.full(pair_targets.shape[0], np.nan, dtype=np.float64)
        candidate_oof = np.full(pair_targets.shape[0], np.nan, dtype=np.float64)
        max_overlap = 0
        for fit_indices, holdout_indices in split_indices:
            overlap = set(pair_groups[fit_indices].tolist()) & set(
                pair_groups[holdout_indices].tolist()
            )
            max_overlap = max(max_overlap, len(overlap))
            control_model = clone(_readout(seed=int(seed)))
            candidate_model = clone(_readout(seed=int(seed)))
            control_model.fit(control_features[fit_indices], pair_targets[fit_indices])
            candidate_model.fit(candidate_features[fit_indices], pair_targets[fit_indices])
            control_oof[holdout_indices] = _positive_probabilities(
                control_model, control_features[holdout_indices]
            )
            candidate_oof[holdout_indices] = _positive_probabilities(
                candidate_model, candidate_features[holdout_indices]
            )
        if not np.isfinite(control_oof).all() or not np.isfinite(candidate_oof).all():
            raise RuntimeError(f"OOF predictions are incomplete for rival class {rival}")
        control_metrics = _binary_metrics(pair_targets, control_oof)
        candidate_metrics = _binary_metrics(pair_targets, candidate_oof)
        row: Dict[str, object] = {
            "pair": f"{int(rival)}-1",
            "rival_class": int(rival),
            "samples": int(pair_targets.size),
            "class1_samples": int(pair_targets.sum()),
            "rival_samples": int((pair_targets == 0).sum()),
            "source_groups": int(np.unique(pair_groups).size),
            "folds": int(len(split_indices)),
            "max_source_overlap": int(max_overlap),
        }
        for prefix, metrics in (
            ("control", control_metrics),
            ("candidate", candidate_metrics),
        ):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = float(value)
        for key in (
            "balanced_accuracy",
            "auroc",
            "precision_class1",
            "recall_class1",
            "specificity_rival",
            "f1_class1",
        ):
            row[f"delta_{key}"] = float(candidate_metrics[key] - control_metrics[key])
        pair_rows.append(row)
        for local_index, sample_index in enumerate(pair_indices):
            prediction_rows.append(
                {
                    "sample_index": int(sample_index),
                    "source_group": str(groups[sample_index]),
                    "target_index": int(labels[sample_index]),
                    "pair": f"{int(rival)}-1",
                    "control_probability_class1": float(control_oof[local_index]),
                    "candidate_probability_class1": float(candidate_oof[local_index]),
                }
            )
    return pair_rows, prediction_rows


def assess_patch_stat_readiness(
    *,
    pair_rows: Sequence[Mapping[str, object]],
    train_samples: int,
    source_groups: int,
    feature_finite: bool,
) -> Dict[str, object]:
    rows = [dict(row) for row in pair_rows]
    if not rows:
        raise ValueError("Readiness requires at least one pair result")
    by_rival = {int(row["rival_class"]): row for row in rows}
    if 2 not in by_rival:
        raise ValueError("Readiness requires the locked 2-1 boundary")
    thresholds = {
        "required_train_samples": 8278,
        "min_source_groups": 7000,
        "required_pairs": len(RIVAL_CLASSES),
        "min_mean_balanced_accuracy_gain": 0.005,
        "min_mean_auroc_gain": 0.005,
        "min_mean_specificity_gain": 0.010,
        "max_pair_recall_loss": 0.020,
        "min_class2_balanced_accuracy_gain": 0.005,
        "min_class2_specificity_gain": 0.020,
        "max_source_overlap": 0,
    }
    mean_ba_gain = float(
        np.mean([float(row["delta_balanced_accuracy"]) for row in rows])
    )
    mean_auc_gain = float(np.mean([float(row["delta_auroc"]) for row in rows]))
    mean_specificity_gain = float(
        np.mean([float(row["delta_specificity_rival"]) for row in rows])
    )
    minimum_recall_delta = float(
        min(float(row["delta_recall_class1"]) for row in rows)
    )
    class2 = by_rival[2]
    observed = {
        "mean_balanced_accuracy_gain": mean_ba_gain,
        "mean_auroc_gain": mean_auc_gain,
        "mean_specificity_gain": mean_specificity_gain,
        "minimum_pair_recall_delta": minimum_recall_delta,
        "class2_balanced_accuracy_gain": float(class2["delta_balanced_accuracy"]),
        "class2_specificity_gain": float(class2["delta_specificity_rival"]),
        "max_source_overlap": int(max(int(row["max_source_overlap"]) for row in rows)),
    }
    checks = {
        "canonical_train_support": int(train_samples)
        == int(thresholds["required_train_samples"]),
        "source_group_support": int(source_groups)
        >= int(thresholds["min_source_groups"]),
        "locked_pair_coverage": len(rows) == int(thresholds["required_pairs"])
        and set(by_rival) == set(RIVAL_CLASSES),
        "feature_finite": bool(feature_finite),
        "source_group_folds_disjoint": observed["max_source_overlap"]
        <= int(thresholds["max_source_overlap"]),
        "mean_balanced_accuracy_gain": mean_ba_gain
        >= float(thresholds["min_mean_balanced_accuracy_gain"]),
        "mean_auroc_gain": mean_auc_gain
        >= float(thresholds["min_mean_auroc_gain"]),
        "mean_rival_rejection_gain": mean_specificity_gain
        >= float(thresholds["min_mean_specificity_gain"]),
        "class1_recall_protected": minimum_recall_delta
        >= -float(thresholds["max_pair_recall_loss"]),
        "class2_boundary_gain": float(class2["delta_balanced_accuracy"])
        >= float(thresholds["min_class2_balanced_accuracy_gain"]),
        "class2_false_positive_rejection": float(class2["delta_specificity_rival"])
        >= float(thresholds["min_class2_specificity_gain"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "patch_stat_specialist_ready": bool(ready),
        "implementation_permission": bool(ready),
        "smoke_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: List[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
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
    contract = _model_contract(model)
    dataset, class_names = _build_dataset(
        data_yaml=data_path,
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    if len(class_names) != 5:
        raise ValueError(f"Locked audit requires five classes, got {len(class_names)}")
    _assert_output_outside_train(output_dir, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema_version": 1,
        "mode": "dinov3_pair_patch_stat_train_only_readiness",
        "data": str(data_path),
        "data_sha256": _sha256(data_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "output_dir": str(output_dir),
        "class_names": [str(name) for name in class_names],
        "train_samples": int(len(dataset)),
        "model_contract": contract,
        "locked_protocol": {
            "focus_class": FOCUS_CLASS,
            "rival_classes": list(RIVAL_CLASSES),
            "patch_margin_pairs": [list(pair) for pair in PATCH_MARGIN_PAIRS],
            "folds": FOLDS,
            "seed": SEED,
            "top_k": TOP_K,
            "logistic_c": LOGISTIC_C,
            "control": "pooled_prelogits_plus_global_logits",
            "candidate": "control_plus_local_patch_logit_statistics",
            "matched_source_group_folds": True,
            "candidate_selection_uses_validation": False,
            "threshold_sweep": False,
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
    model.to(device)
    start = time.perf_counter()
    extracted = _extract_train_features(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        amp=bool(args.amp),
    )
    pooled_control = np.concatenate(
        (
            np.asarray(extracted["pooled"], dtype=np.float32),
            np.asarray(extracted["global_logits"], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    pair_rows, prediction_rows = _pair_oof_comparison(
        pooled_features=pooled_control,
        patch_features=np.asarray(extracted["patch_statistics"], dtype=np.float32),
        labels=np.asarray(extracted["labels"], dtype=np.int64),
        source_groups=np.asarray(extracted["source_groups"], dtype=object),
    )
    feature_finite = bool(
        np.isfinite(pooled_control).all()
        and np.isfinite(np.asarray(extracted["patch_statistics"], dtype=np.float32)).all()
    )
    readiness = assess_patch_stat_readiness(
        pair_rows=pair_rows,
        train_samples=int(np.asarray(extracted["labels"]).size),
        source_groups=int(np.unique(np.asarray(extracted["source_groups"])).size),
        feature_finite=feature_finite,
    )
    _write_csv(output_dir / "pair_metrics.csv", pair_rows)
    path_lookup = np.asarray(extracted["paths"], dtype=object)
    for row in prediction_rows:
        row["image_path"] = str(path_lookup[int(row["sample_index"])])
    _write_csv(output_dir / "train_oof_pair_predictions.csv", prediction_rows)
    np.savez_compressed(
        output_dir / "train_feature_cache.npz",
        pooled=np.asarray(extracted["pooled"], dtype=np.float32),
        global_logits=np.asarray(extracted["global_logits"], dtype=np.float32),
        patch_statistics=np.asarray(extracted["patch_statistics"], dtype=np.float32),
        labels=np.asarray(extracted["labels"], dtype=np.int64),
        paths=np.asarray(extracted["paths"], dtype=object),
        source_groups=np.asarray(extracted["source_groups"], dtype=object),
        class_names=np.asarray(class_names, dtype=object),
    )
    summary = {
        **preflight,
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "amp": bool(args.amp),
        "feature_dimensions": {
            "pooled": int(np.asarray(extracted["pooled"]).shape[1]),
            "global_logits": int(np.asarray(extracted["global_logits"]).shape[1]),
            "control": int(pooled_control.shape[1]),
            "patch_statistics": int(np.asarray(extracted["patch_statistics"]).shape[1]),
            "candidate": int(
                pooled_control.shape[1]
                + np.asarray(extracted["patch_statistics"]).shape[1]
            ),
        },
        "patch_token_shape": list(extracted["patch_token_shape"]),
        "source_groups": int(np.unique(np.asarray(extracted["source_groups"])).size),
        "pair_results": pair_rows,
        "readiness": readiness,
        "artifacts": {
            "pair_metrics": str(output_dir / "pair_metrics.csv"),
            "oof_predictions": str(output_dir / "train_oof_pair_predictions.csv"),
            "feature_cache": str(output_dir / "train_feature_cache.npz"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(args)
    readiness = summary.get("readiness", {})
    if isinstance(readiness, Mapping) and "patch_stat_specialist_ready" in readiness:
        print(
            json.dumps(
                {
                    "patch_stat_specialist_ready": bool(
                        readiness.get("patch_stat_specialist_ready", False)
                    ),
                    "failed_checks": list(readiness.get("failed_checks", [])),
                    "observed": dict(readiness.get("observed", {})),
                    "output_dir": str(Path(args.output_dir).resolve()),
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "preflight_only": bool(args.preflight_only),
                    "train_samples": int(summary.get("train_samples", 0)),
                    "validation_split_used": bool(
                        summary.get("validation_split_used", False)
                    ),
                    "test_split_used": bool(summary.get("test_split_used", False)),
                    "output_dir": str(Path(args.output_dir).resolve()),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from trkh.tools.audit_two_stage_reedl_readiness import (
    _apply_scaler,
    _classification_metrics,
    _direction_auc,
    _fit_scaler,
    _load_cache,
    _transition_stats,
    _write_csv,
)


SEED = 20260712
FOLDS = 5
LOGISTIC_C = 0.3
MAX_ITERATIONS = 2000
INFERENCE_ALPHA = 0.5
FOCUS_CLASS_INDEX = 1
FOCUS_MILESTONE = 0.70
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only BBN classifier-space readiness audit over a "
            "frozen TRKH embedding cache. It never reads test or writes a model."
        )
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--logistic-c", type=float, default=LOGISTIC_C)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--inference-alpha", type=float, default=INFERENCE_ALPHA)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def bbn_reversed_sample_weights(
    labels: np.ndarray,
    *,
    class_count: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the loss weights equivalent to BBN's reversed class sampler."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    counts = np.bincount(labels, minlength=int(class_count)).astype(np.float64)
    if counts.size != int(class_count) or np.any(counts <= 0.0):
        raise ValueError(f"Every class needs support, observed counts={counts.tolist()}")

    maximum = float(counts.max())
    class_raw = maximum / counts
    class_probabilities = class_raw / class_raw.sum()

    # BBN first samples a class with probability proportional to 1/N_i and
    # then samples uniformly within that class. Applied as a weighted empirical
    # loss over natural rows, the equivalent per-row weight is proportional to
    # 1/N_i^2 because each class contributes N_i rows.
    per_class_row_weight = np.square(maximum / counts)
    sample_weights = per_class_row_weight[labels]
    sample_weights /= sample_weights.mean()
    realized_class_mass = np.asarray(
        [sample_weights[labels == index].sum() for index in range(int(class_count))],
        dtype=np.float64,
    )
    realized_class_probabilities = realized_class_mass / realized_class_mass.sum()
    if not np.allclose(realized_class_probabilities, class_probabilities, atol=1e-10):
        raise RuntimeError("Derived row weights do not reproduce the reversed class sampler")
    return sample_weights.astype(np.float64), {
        "class_counts": counts.astype(np.int64).tolist(),
        "class_raw_inverse_frequency": class_raw.tolist(),
        "target_class_sampling_probabilities": class_probabilities.tolist(),
        "per_class_row_weights_before_mean_normalization": per_class_row_weight.tolist(),
        "realized_weighted_class_probabilities": realized_class_probabilities.tolist(),
        "sample_weight_mean": float(sample_weights.mean()),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def fuse_bilateral_logits(
    conventional_logits: np.ndarray,
    reversed_logits: np.ndarray,
    *,
    alpha: float = INFERENCE_ALPHA,
) -> np.ndarray:
    conventional_logits = np.asarray(conventional_logits, dtype=np.float64)
    reversed_logits = np.asarray(reversed_logits, dtype=np.float64)
    if conventional_logits.shape != reversed_logits.shape or conventional_logits.ndim != 2:
        raise ValueError("Bilateral logits must be aligned 2D arrays")
    coefficient = float(alpha)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return (
        coefficient * conventional_logits + (1.0 - coefficient) * reversed_logits
    ).astype(np.float32)


def _fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    logistic_c: float,
    max_iterations: int,
    workers: int,
    sample_weights: Optional[np.ndarray],
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(logistic_c),
        max_iter=int(max_iterations),
        solver="lbfgs",
        n_jobs=max(1, int(workers)),
        tol=1e-5,
    )
    model.fit(
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        sample_weight=(
            None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
        ),
    )
    expected = np.arange(int(np.max(labels)) + 1, dtype=np.int64)
    if not np.array_equal(model.classes_, expected):
        raise ValueError(f"Classifier class order is unsafe: {model.classes_.tolist()}")
    return model


def _branch_logits(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    logits = np.asarray(model.decision_function(features), dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Expected multiclass logits, got {logits.shape}")
    return logits


def _fit_bilateral_pair(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    logistic_c: float,
    max_iterations: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mean, scale = _fit_scaler(fit_features)
    fit_scaled = _apply_scaler(fit_features, mean, scale)
    eval_scaled = _apply_scaler(eval_features, mean, scale)
    class_count = int(np.max(fit_labels)) + 1
    reversed_weights, weighting = bbn_reversed_sample_weights(
        fit_labels,
        class_count=class_count,
    )
    conventional = _fit_classifier(
        fit_scaled,
        fit_labels,
        logistic_c=logistic_c,
        max_iterations=max_iterations,
        workers=workers,
        sample_weights=None,
    )
    reversed_branch = _fit_classifier(
        fit_scaled,
        fit_labels,
        logistic_c=logistic_c,
        max_iterations=max_iterations,
        workers=workers,
        sample_weights=reversed_weights,
    )
    convergence = {
        "conventional_iterations": [int(value) for value in conventional.n_iter_.tolist()],
        "reversed_iterations": [int(value) for value in reversed_branch.n_iter_.tolist()],
        "max_iterations": int(max_iterations),
        "converged": bool(
            np.max(conventional.n_iter_) < int(max_iterations)
            and np.max(reversed_branch.n_iter_) < int(max_iterations)
        ),
    }
    return (
        _branch_logits(conventional, eval_scaled),
        _branch_logits(reversed_branch, eval_scaled),
        {"weighting": weighting, "convergence": convergence},
    )


def _focus(metrics: Mapping[str, object], focus_class_index: int) -> Mapping[str, object]:
    return metrics["per_class"][int(focus_class_index)]


def assess_bbn_classifier_readiness(
    *,
    train_rows: int,
    val_rows: int,
    fold_source_overlap: int,
    train_val_source_overlap: int,
    folds_with_focus_gain: int,
    fold_count: int,
    all_models_converged: bool,
    oof_conventional: Mapping[str, object],
    oof_bilateral: Mapping[str, object],
    val_keeper: Mapping[str, object],
    val_conventional: Mapping[str, object],
    val_bilateral: Mapping[str, object],
    transitions_vs_keeper: Mapping[str, int],
    transitions_vs_conventional: Mapping[str, int],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    oof_control_focus = _focus(oof_conventional, focus)
    oof_candidate_focus = _focus(oof_bilateral, focus)
    keeper_focus = _focus(val_keeper, focus)
    val_control_focus = _focus(val_conventional, focus)
    val_candidate_focus = _focus(val_bilateral, focus)
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "fold_source_overlap": int(fold_source_overlap),
        "train_val_source_overlap": int(train_val_source_overlap),
        "folds_with_focus_gain": int(folds_with_focus_gain),
        "fold_count": int(fold_count),
        "all_models_converged": bool(all_models_converged),
        "oof_conventional_macro_f1": float(oof_conventional["macro_f1"]),
        "oof_conventional_focus_f1": float(oof_control_focus["f1"]),
        "oof_bilateral_macro_f1": float(oof_bilateral["macro_f1"]),
        "oof_bilateral_focus_f1": float(oof_candidate_focus["f1"]),
        "oof_macro_gain": float(
            oof_bilateral["macro_f1"] - oof_conventional["macro_f1"]
        ),
        "oof_focus_gain": float(oof_candidate_focus["f1"] - oof_control_focus["f1"]),
        "keeper_macro_f1": float(val_keeper["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "keeper_focus_recall": float(keeper_focus["recall"]),
        "val_conventional_macro_f1": float(val_conventional["macro_f1"]),
        "val_conventional_focus_f1": float(val_control_focus["f1"]),
        "val_bilateral_macro_f1": float(val_bilateral["macro_f1"]),
        "val_bilateral_focus_f1": float(val_candidate_focus["f1"]),
        "val_bilateral_focus_precision": float(val_candidate_focus["precision"]),
        "val_bilateral_focus_recall": float(val_candidate_focus["recall"]),
        "val_focus_gain_vs_conventional": float(
            val_candidate_focus["f1"] - val_control_focus["f1"]
        ),
        "val_macro_gain_vs_keeper": float(
            val_bilateral["macro_f1"] - val_keeper["macro_f1"]
        ),
        "val_focus_gain_vs_keeper": float(val_candidate_focus["f1"] - keeper_focus["f1"]),
        "transitions_vs_keeper": dict(transitions_vs_keeper),
        "transitions_vs_conventional": dict(transitions_vs_conventional),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "source_group_folds_disjoint": observed["fold_source_overlap"] == 0,
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "all_logistic_models_converged": bool(all_models_converged),
        "oof_macro_gain_ge_0p002": observed["oof_macro_gain"] >= 0.002,
        "oof_focus_gain_ge_0p005": observed["oof_focus_gain"] >= 0.005,
        "focus_gain_in_at_least_3_folds": observed["folds_with_focus_gain"]
        >= min(3, observed["fold_count"]),
        "val_macro_preserves_keeper_within_0p001": observed["val_macro_gain_vs_keeper"]
        >= -0.001,
        "val_focus_reaches_0p70": observed["val_bilateral_focus_f1"]
        >= FOCUS_MILESTONE,
        "val_focus_improves_keeper_by_0p01": observed["val_focus_gain_vs_keeper"]
        >= 0.01,
        "val_focus_gain_vs_conventional_ge_0p005": observed[
            "val_focus_gain_vs_conventional"
        ]
        >= 0.005,
        "val_focus_recall_preserved_within_0p01": observed[
            "val_bilateral_focus_recall"
        ]
        >= observed["keeper_focus_recall"] - 0.01,
        "candidate_corrections_ge_harms_vs_keeper": int(
            transitions_vs_keeper["corrections"]
        )
        >= int(transitions_vs_keeper["harms"]),
        "candidate_focus_fp_removed_ge_created_vs_keeper": int(
            transitions_vs_keeper["focus_false_positive_removed"]
        )
        >= int(transitions_vs_keeper["focus_false_positive_created"]),
        "candidate_focus_fn_rescued_ge_tp_broken_vs_keeper": int(
            transitions_vs_keeper["focus_false_negative_rescued"]
        )
        >= int(transitions_vs_keeper["focus_true_positive_broken"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "image_smoke_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": {
            "oof_macro_gain": 0.002,
            "oof_focus_gain": 0.005,
            "minimum_positive_folds": min(3, int(fold_count)),
            "val_macro_tolerance_vs_keeper": -0.001,
            "focus_milestone": FOCUS_MILESTONE,
            "focus_gain_vs_keeper": 0.01,
            "focus_gain_vs_conventional": 0.005,
            "focus_recall_tolerance_vs_keeper": -0.01,
        },
    }


def _method_row(
    *,
    split: str,
    method: str,
    metrics: Mapping[str, object],
    focus_class_index: int,
) -> Dict[str, object]:
    focus = _focus(metrics, focus_class_index)
    return {
        "split": split,
        "method": method,
        "samples": int(metrics["samples"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "focus_f1": float(focus["f1"]),
        "focus_precision": float(focus["precision"]),
        "focus_recall": float(focus["recall"]),
        "nll": float(metrics["nll"]),
        "brier": float(metrics["brier"]),
    }


def _prediction_rows(
    *,
    split: str,
    cache: Mapping[str, np.ndarray],
    fold_assignment: np.ndarray,
    conventional: np.ndarray,
    reversed_branch: np.ndarray,
    bilateral: np.ndarray,
) -> list[Dict[str, object]]:
    output = []
    for row_index in range(int(cache["labels"].shape[0])):
        row: Dict[str, object] = {
            "split": split,
            "sample_index": int(cache["sample_index"][row_index]),
            "fold": int(fold_assignment[row_index]),
            "source_stem": str(cache["source_stems"][row_index]),
            "image_path": str(cache["paths"][row_index]),
            "target_index": int(cache["labels"][row_index]),
            "keeper_prediction_index": int(cache["probabilities"][row_index].argmax()),
            "conventional_prediction_index": int(conventional[row_index].argmax()),
            "reversed_prediction_index": int(reversed_branch[row_index].argmax()),
            "bilateral_prediction_index": int(bilateral[row_index].argmax()),
        }
        for class_index in range(int(conventional.shape[1])):
            row[f"keeper_prob_{class_index}"] = float(
                cache["probabilities"][row_index, class_index]
            )
            row[f"conventional_prob_{class_index}"] = float(
                conventional[row_index, class_index]
            )
            row[f"reversed_prob_{class_index}"] = float(
                reversed_branch[row_index, class_index]
            )
            row[f"bilateral_prob_{class_index}"] = float(
                bilateral[row_index, class_index]
            )
        output.append(row)
    return output


def _plot_method_metrics(
    method_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, split in zip(axes, ("train_oof", "val")):
        rows = {str(row["method"]): row for row in method_rows if row["split"] == split}
        methods = (
            ["conventional", "reversed", "bilateral"]
            if split == "train_oof"
            else ["keeper", "conventional", "reversed", "bilateral"]
        )
        x = np.arange(len(methods))
        macro = [float(rows[name]["macro_f1"]) for name in methods]
        focus = [float(rows[name]["focus_f1"]) for name in methods]
        axis.bar(x - 0.18, macro, width=0.36, label="macro F1")
        axis.bar(x + 0.18, focus, width=0.36, label="class1 F1")
        axis.set_xticks(x, methods, rotation=18, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(split)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


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
        "mode": "bbn_bilateral_classifier_readiness_evidence_manifest",
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


def _validate_protocol(args: argparse.Namespace) -> None:
    if int(args.folds) < 3:
        raise ValueError("At least three source-grouped folds are required")
    if float(args.logistic_c) <= 0.0:
        raise ValueError("logistic-c must be positive")
    if int(args.max_iterations) < 100:
        raise ValueError("max-iterations must be at least 100")
    if not 0.0 <= float(args.inference_alpha) <= 1.0:
        raise ValueError("inference-alpha must be in [0, 1]")
    if int(args.workers) <= 0:
        raise ValueError("workers must be positive")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_protocol(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache_path = Path(args.train_cache).resolve()
    val_cache_path = Path(args.val_cache).resolve()
    train = _load_cache(train_cache_path, split="train")
    val = _load_cache(val_cache_path, split="val")
    if train["embeddings"].shape[1] != val["embeddings"].shape[1]:
        raise ValueError("Train and validation embedding dimensions differ")
    class_count = int(train["probabilities"].shape[1])
    if val["probabilities"].shape[1] != class_count:
        raise ValueError("Train and validation class counts differ")
    focus_class_index = int(args.focus_class_index)
    if not 0 <= focus_class_index < class_count:
        raise ValueError("focus-class-index is outside the class range")

    train_rows = int(train["labels"].shape[0])
    val_rows = int(val["labels"].shape[0])
    train_sources = set(train["source_stems"].tolist())
    val_sources = set(val["source_stems"].tolist())
    train_val_source_overlap = len(train_sources.intersection(val_sources))
    oof_conventional_logits = np.zeros((train_rows, class_count), dtype=np.float32)
    oof_reversed_logits = np.zeros_like(oof_conventional_logits)
    fold_assignment = np.full(train_rows, -1, dtype=np.int64)
    fold_rows = []
    fold_protocols = []
    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    maximum_source_overlap = 0
    all_models_converged = True
    for fold_index, (fit_indices, hold_indices) in enumerate(
        splitter.split(train["embeddings"], train["labels"], train["source_stems"])
    ):
        fit_sources = set(train["source_stems"][fit_indices].tolist())
        hold_sources = set(train["source_stems"][hold_indices].tolist())
        source_overlap = len(fit_sources.intersection(hold_sources))
        maximum_source_overlap = max(maximum_source_overlap, source_overlap)
        conventional_logits, reversed_logits, fit_protocol = _fit_bilateral_pair(
            train["embeddings"][fit_indices],
            train["labels"][fit_indices],
            train["embeddings"][hold_indices],
            logistic_c=float(args.logistic_c),
            max_iterations=int(args.max_iterations),
            workers=int(args.workers),
        )
        oof_conventional_logits[hold_indices] = conventional_logits
        oof_reversed_logits[hold_indices] = reversed_logits
        fold_assignment[hold_indices] = int(fold_index)
        conventional_probabilities = _softmax(conventional_logits)
        reversed_probabilities = _softmax(reversed_logits)
        bilateral_probabilities = _softmax(
            fuse_bilateral_logits(
                conventional_logits,
                reversed_logits,
                alpha=float(args.inference_alpha),
            )
        )
        fold_conventional = _classification_metrics(
            train["labels"][hold_indices],
            conventional_probabilities,
        )
        fold_reversed = _classification_metrics(
            train["labels"][hold_indices],
            reversed_probabilities,
        )
        fold_bilateral = _classification_metrics(
            train["labels"][hold_indices],
            bilateral_probabilities,
        )
        all_models_converged = all_models_converged and bool(
            fit_protocol["convergence"]["converged"]
        )
        fold_protocols.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": int(source_overlap),
                **fit_protocol,
            }
        )
        for method, metrics in (
            ("conventional", fold_conventional),
            ("reversed", fold_reversed),
            ("bilateral", fold_bilateral),
        ):
            focus_metrics = _focus(metrics, focus_class_index)
            fold_rows.append(
                {
                    "fold": int(fold_index),
                    "method": method,
                    "rows": int(metrics["samples"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "focus_f1": float(focus_metrics["f1"]),
                    "focus_precision": float(focus_metrics["precision"]),
                    "focus_recall": float(focus_metrics["recall"]),
                }
            )
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF assignment is incomplete")

    oof_conventional = _softmax(oof_conventional_logits)
    oof_reversed = _softmax(oof_reversed_logits)
    oof_bilateral = _softmax(
        fuse_bilateral_logits(
            oof_conventional_logits,
            oof_reversed_logits,
            alpha=float(args.inference_alpha),
        )
    )
    val_conventional_logits, val_reversed_logits, full_fit_protocol = _fit_bilateral_pair(
        train["embeddings"],
        train["labels"],
        val["embeddings"],
        logistic_c=float(args.logistic_c),
        max_iterations=int(args.max_iterations),
        workers=int(args.workers),
    )
    all_models_converged = all_models_converged and bool(
        full_fit_protocol["convergence"]["converged"]
    )
    val_conventional = _softmax(val_conventional_logits)
    val_reversed = _softmax(val_reversed_logits)
    val_bilateral = _softmax(
        fuse_bilateral_logits(
            val_conventional_logits,
            val_reversed_logits,
            alpha=float(args.inference_alpha),
        )
    )

    split_probabilities = {
        "train_oof": {
            "keeper_in_sample_reference": train["probabilities"],
            "conventional": oof_conventional,
            "reversed": oof_reversed,
            "bilateral": oof_bilateral,
        },
        "val": {
            "keeper": val["probabilities"],
            "conventional": val_conventional,
            "reversed": val_reversed,
            "bilateral": val_bilateral,
        },
    }
    split_caches = {"train_oof": train, "val": val}
    metrics: Dict[str, Dict[str, object]] = {}
    method_rows = []
    for split, probabilities_by_method in split_probabilities.items():
        metrics[split] = {}
        for method, probabilities in probabilities_by_method.items():
            method_metrics = _classification_metrics(
                split_caches[split]["labels"],
                probabilities,
            )
            metrics[split][method] = method_metrics
            method_rows.append(
                _method_row(
                    split=split,
                    method=method,
                    metrics=method_metrics,
                    focus_class_index=focus_class_index,
                )
            )

    fold_lookup = {
        (int(row["fold"]), str(row["method"])): row for row in fold_rows
    }
    folds_with_focus_gain = sum(
        float(fold_lookup[(fold, "bilateral")]["focus_f1"])
        > float(fold_lookup[(fold, "conventional")]["focus_f1"])
        for fold in range(int(args.folds))
    )
    transitions = {
        "train_oof_bilateral_vs_conventional": _transition_stats(
            train["labels"],
            oof_conventional,
            oof_bilateral,
            focus_class_index=focus_class_index,
        ),
        "train_oof_bilateral_vs_keeper": _transition_stats(
            train["labels"],
            train["probabilities"],
            oof_bilateral,
            focus_class_index=focus_class_index,
        ),
        "val_bilateral_vs_conventional": _transition_stats(
            val["labels"],
            val_conventional,
            val_bilateral,
            focus_class_index=focus_class_index,
        ),
        "val_bilateral_vs_keeper": _transition_stats(
            val["labels"],
            val["probabilities"],
            val_bilateral,
            focus_class_index=focus_class_index,
        ),
    }
    direction = {
        "train_oof_bilateral_vs_conventional": _direction_auc(
            train["labels"],
            oof_conventional,
            oof_bilateral,
            focus_class_index=focus_class_index,
        ),
        "val_bilateral_vs_conventional": _direction_auc(
            val["labels"],
            val_conventional,
            val_bilateral,
            focus_class_index=focus_class_index,
        ),
    }
    gate = assess_bbn_classifier_readiness(
        train_rows=train_rows,
        val_rows=val_rows,
        fold_source_overlap=maximum_source_overlap,
        train_val_source_overlap=train_val_source_overlap,
        folds_with_focus_gain=folds_with_focus_gain,
        fold_count=int(args.folds),
        all_models_converged=all_models_converged,
        oof_conventional=metrics["train_oof"]["conventional"],
        oof_bilateral=metrics["train_oof"]["bilateral"],
        val_keeper=metrics["val"]["keeper"],
        val_conventional=metrics["val"]["conventional"],
        val_bilateral=metrics["val"]["bilateral"],
        transitions_vs_keeper=transitions["val_bilateral_vs_keeper"],
        transitions_vs_conventional=transitions["val_bilateral_vs_conventional"],
        focus_class_index=focus_class_index,
        test_split_used=False,
    )

    protocol = {
        "method": "bbn_bilateral_classifier_space_readiness",
        "scope": (
            "A frozen-embedding precheck of natural and BBN-reversed classifier "
            "branches. It is BBN-inspired readiness evidence, not an image-model reproduction."
        ),
        "primary_source": (
            "https://openaccess.thecvf.com/content_CVPR_2020/html/"
            "Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_"
            "Long-Tailed_Visual_Recognition_CVPR_2020_paper.html"
        ),
        "train_cache": str(train_cache_path),
        "val_cache": str(val_cache_path),
        "split_usage": {"train": True, "val": True, "test": False},
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_source_groups": int(len(train_sources)),
        "val_source_groups": int(len(val_sources)),
        "train_val_source_overlap": int(train_val_source_overlap),
        "embedding_dim": int(train["embeddings"].shape[1]),
        "class_count": class_count,
        "folds": int(args.folds),
        "source_grouped": True,
        "logistic_c": float(args.logistic_c),
        "max_iterations": int(args.max_iterations),
        "scaler": "fit rows only; common to both branches",
        "conventional_branch": "natural-frequency multinomial logistic readout",
        "reversed_branch": (
            "BBN reversed class probability proportional to 1/N_i, represented "
            "as per-row loss weight proportional to 1/N_i^2"
        ),
        "inference_fusion": (
            f"raw branch logits alpha={float(args.inference_alpha):.6f}, "
            f"1-alpha={1.0 - float(args.inference_alpha):.6f}"
        ),
        "validation_hyperparameter_tuning": False,
        "underlying_keeper_train_cache_is_not_oof": True,
        "probability_direction_scope": (
            "Only conventional-versus-bilateral logistic readouts share a comparable "
            "probability scale. Keeper comparisons use discrete transitions, not p1 deltas."
        ),
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
        "test_split_used": False,
    }
    summary = {
        "protocol": protocol,
        "fold_protocols": fold_protocols,
        "full_fit_protocol": full_fit_protocol,
        "metrics": metrics,
        "transitions": transitions,
        "focus_direction": direction,
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }
    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(output_dir / "method_metrics.csv", method_rows)
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            cache=train,
            fold_assignment=fold_assignment,
            conventional=oof_conventional,
            reversed_branch=oof_reversed,
            bilateral=oof_bilateral,
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            cache=val,
            fold_assignment=np.full(val_rows, -1, dtype=np.int64),
            conventional=val_conventional,
            reversed_branch=val_reversed,
            bilateral=val_bilateral,
        ),
    )
    _plot_method_metrics(method_rows, output_dir / "method_metrics.png")
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    readme = [
        "# BBN Bilateral Classifier-Space Readiness",
        "",
        f"- OOF conventional macro/class1: `{float(metrics['train_oof']['conventional']['macro_f1']):.6f}/{float(_focus(metrics['train_oof']['conventional'], focus_class_index)['f1']):.6f}`",
        f"- OOF bilateral macro/class1: `{float(metrics['train_oof']['bilateral']['macro_f1']):.6f}/{float(_focus(metrics['train_oof']['bilateral'], focus_class_index)['f1']):.6f}`",
        f"- Val keeper macro/class1: `{float(metrics['val']['keeper']['macro_f1']):.6f}/{float(_focus(metrics['val']['keeper'], focus_class_index)['f1']):.6f}`",
        f"- Val conventional macro/class1: `{float(metrics['val']['conventional']['macro_f1']):.6f}/{float(_focus(metrics['val']['conventional'], focus_class_index)['f1']):.6f}`",
        f"- Val reversed macro/class1: `{float(metrics['val']['reversed']['macro_f1']):.6f}/{float(_focus(metrics['val']['reversed'], focus_class_index)['f1']):.6f}`",
        f"- Val bilateral macro/class1: `{float(metrics['val']['bilateral']['macro_f1']):.6f}/{float(_focus(metrics['val']['bilateral'], focus_class_index)['f1']):.6f}`",
        f"- OOF bilateral gains versus conventional macro/class1: `{float(metrics['train_oof']['bilateral']['macro_f1'] - metrics['train_oof']['conventional']['macro_f1']):+.6f}/{float(_focus(metrics['train_oof']['bilateral'], focus_class_index)['f1'] - _focus(metrics['train_oof']['conventional'], focus_class_index)['f1']):+.6f}`",
        f"- Val bilateral gains versus keeper macro/class1: `{float(metrics['val']['bilateral']['macro_f1'] - metrics['val']['keeper']['macro_f1']):+.6f}/{float(_focus(metrics['val']['bilateral'], focus_class_index)['f1'] - _focus(metrics['val']['keeper'], focus_class_index)['f1']):+.6f}`",
        f"- Same-scale class1 FN-versus-FP direction AUROC, OOF -> val: `{float(direction['train_oof_bilateral_vs_conventional']['auc_fn_positive']):.6f} -> {float(direction['val_bilateral_vs_conventional']['auc_fn_positive']):.6f}`",
        f"- Val bilateral versus keeper class1 FP removed/created: `{int(transitions['val_bilateral_vs_keeper']['focus_false_positive_removed'])}/{int(transitions['val_bilateral_vs_keeper']['focus_false_positive_created'])}`",
        f"- Val bilateral versus keeper class1 FN rescued/TP broken: `{int(transitions['val_bilateral_vs_keeper']['focus_false_negative_rescued'])}/{int(transitions['val_bilateral_vs_keeper']['focus_true_positive_broken'])}`",
        f"- Image smoke permission: `{str(bool(gate['image_smoke_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "Decision: reject before image smoke. Do not sweep branch alpha, logistic C, reversed weighting, or folds on this frozen representation.",
        "",
        "This diagnostic reads frozen train/validation caches only. It does not read test, modify raw data, or write a model/checkpoint.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(output_dir)
    summary["artifact_manifest"] = {
        key: value for key, value in manifest.items() if key != "files"
    }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(args)
    print(
        json.dumps(
            {
                "metrics": summary["metrics"],
                "transitions": summary["transitions"],
                "gate": summary["gate"],
                "artifact_manifest": summary["artifact_manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

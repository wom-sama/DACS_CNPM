from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from trkh.tools.audit_teacher_uncertainty_critic_readiness import (
    PredictionTable,
    _load_prediction_table,
    _roc_auc,
)


def adaptive_lambda_ratio(probabilities: np.ndarray, *, lambda_ref: float, alpha: float) -> np.ndarray:
    """Return the official one-step ALDR lambda/lambda_ref update."""
    if lambda_ref <= 0.0:
        raise ValueError("lambda_ref must be positive.")
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1 so every adaptive lambda stays positive.")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    logits = np.log(probabilities)
    scaled = logits / float(lambda_ref)
    scaled -= scaled.max(axis=1, keepdims=True)
    dw_probabilities = np.exp(scaled)
    dw_probabilities /= dw_probabilities.sum(axis=1, keepdims=True)
    class_count = int(probabilities.shape[1])
    log_class_count = math.log(class_count)
    kl_uniform = np.sum(
        dw_probabilities * np.log(np.clip(class_count * dw_probabilities, 1e-12, None)),
        axis=1,
    )
    return 1.0 - kl_uniform / (float(alpha) * log_class_count)


def target_gradient_strength(
    probabilities: np.ndarray,
    targets: np.ndarray,
    lambdas: np.ndarray,
    *,
    margin: float,
) -> np.ndarray:
    """Magnitude of the detached-lambda ALDR gradient on the target logit."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    targets = np.asarray(targets, dtype=np.int64)
    lambdas = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    if len(probabilities) != len(targets) or len(targets) != len(lambdas):
        raise ValueError("Probability, target, and lambda rows must align.")
    if np.any(lambdas <= 0.0):
        raise ValueError("Every lambda must be positive.")
    logits = np.log(probabilities)
    target_logits = logits[np.arange(len(targets)), targets][:, None]
    differences = logits - target_logits + float(margin)
    differences[np.arange(len(targets)), targets] = 0.0
    scaled = differences / lambdas[:, None]
    scaled -= scaled.max(axis=1, keepdims=True)
    adversarial_weights = np.exp(scaled)
    adversarial_weights /= adversarial_weights.sum(axis=1, keepdims=True)
    return 1.0 - adversarial_weights[np.arange(len(targets)), targets]


def _group_id(row: Mapping[str, str], sample_index: int) -> str:
    source_stem = str(row.get("source_stem", "") or "").strip()
    if source_stem:
        return source_stem
    image_path = str(row.get("image_path", row.get("path", "")) or "").strip()
    if image_path:
        normalized = image_path.replace("\\", "/")
        return normalized.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return f"sample_{sample_index}"


def _describe(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Dict[str, object]:
    point = _roc_auc(labels, scores)
    if point is None or replicates <= 0:
        return {"auc": point, "bootstrap_replicates": 0, "ci95_low": None, "ci95_high": None}
    unique_groups, group_inverse = np.unique(groups.astype(str), return_inverse=True)
    group_rows = [np.flatnonzero(group_inverse == index) for index in range(len(unique_groups))]
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(replicates)):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        indices = np.concatenate([group_rows[index] for index in sampled])
        value = _roc_auc(labels[indices], scores[indices])
        if value is not None:
            values.append(float(value))
    if not values:
        return {"auc": point, "bootstrap_replicates": 0, "ci95_low": None, "ci95_high": None}
    return {
        "auc": point,
        "bootstrap_replicates": int(len(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _auc_task(
    mask: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, object]:
    masked_labels = labels[mask]
    masked_scores = scores[mask]
    payload = _bootstrap_auc(
        masked_labels,
        masked_scores,
        groups[mask],
        replicates=bootstrap_replicates,
        seed=seed,
    )
    positive = masked_labels == 1
    negative = ~positive
    payload.update(
        {
            "samples": int(mask.sum()),
            "positives": int(positive.sum()),
            "negatives": int(negative.sum()),
            "positive_mean": float(masked_scores[positive].mean()) if positive.any() else None,
            "negative_mean": float(masked_scores[negative].mean()) if negative.any() else None,
        }
    )
    return payload


def _audit_split(
    name: str,
    path: Path,
    *,
    focus_class_index: int,
    lambda_ref: float,
    alpha: float,
    margin: float,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[Dict[str, object], list[dict[str, object]]]:
    table: PredictionTable = _load_prediction_table(name, Path(path))
    if not 0 <= int(focus_class_index) < table.probabilities.shape[1]:
        raise ValueError("focus_class_index is outside the prediction class range.")
    targets = table.targets
    predictions = table.predictions
    probabilities = table.probabilities
    groups = np.asarray(
        [_group_id(row, int(index)) for row, index in zip(table.rows, table.sample_indices)],
        dtype=object,
    )
    lambda_ratio = adaptive_lambda_ratio(probabilities, lambda_ref=lambda_ref, alpha=alpha)
    adaptive_lambda = float(lambda_ref) * lambda_ratio
    fixed_gradient = target_gradient_strength(
        probabilities,
        targets,
        np.full(len(targets), float(lambda_ref), dtype=np.float64),
        margin=margin,
    )
    adaptive_gradient = target_gradient_strength(
        probabilities,
        targets,
        adaptive_lambda,
        margin=margin,
    )
    gradient_ratio = adaptive_gradient / np.clip(fixed_gradient, 1e-12, None)
    gradient_change = np.abs(gradient_ratio - 1.0)
    wrong = predictions != targets
    focus = int(focus_class_index)
    focus_tp = (targets == focus) & (predictions == focus)
    focus_fn = (targets == focus) & (predictions != focus)
    focus_fp = (targets != focus) & (predictions == focus)

    masks = {
        "all": np.ones(len(targets), dtype=bool),
        "correct": ~wrong,
        "wrong": wrong,
        "focus_true_positive": focus_tp,
        "focus_false_negative": focus_fn,
        "focus_false_positive": focus_fp,
        "nonfocus_true_negative": (targets != focus) & (predictions != focus),
    }
    category_stats = {}
    for category, mask in masks.items():
        category_stats[category] = {
            "samples": int(mask.sum()),
            "lambda_ratio": _describe(lambda_ratio[mask]),
            "fixed_target_gradient": _describe(fixed_gradient[mask]),
            "adaptive_target_gradient": _describe(adaptive_gradient[mask]),
            "target_gradient_ratio": _describe(gradient_ratio[mask]),
        }

    all_mask = masks["all"]
    fp_or_tp = predictions == focus
    fn_or_tp = targets == focus
    fp_or_fn = focus_fp | focus_fn
    auc_tasks = {
        "error_detection_all": _auc_task(
            all_mask,
            wrong.astype(np.int64),
            lambda_ratio,
            groups,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "predicted_focus_fp_detection": _auc_task(
            fp_or_tp,
            focus_fp.astype(np.int64),
            lambda_ratio,
            groups,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 1,
        ),
        "true_focus_fn_detection": _auc_task(
            fn_or_tp,
            focus_fn.astype(np.int64),
            lambda_ratio,
            groups,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 2,
        ),
        "focus_fp_vs_fn_direction": _auc_task(
            fp_or_fn,
            focus_fp.astype(np.int64),
            lambda_ratio,
            groups,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 3,
        ),
    }

    lambda_p10 = float(np.quantile(lambda_ratio, 0.10))
    lambda_p90 = float(np.quantile(lambda_ratio, 0.90))
    rows = []
    for row_index, sample_index in enumerate(table.sample_indices.tolist()):
        category = "correct"
        if focus_fn[row_index]:
            category = "focus_false_negative"
        elif focus_fp[row_index]:
            category = "focus_false_positive"
        elif focus_tp[row_index]:
            category = "focus_true_positive"
        elif wrong[row_index]:
            category = "other_error"
        rows.append(
            {
                "split": name,
                "sample_index": int(sample_index),
                "source_group": str(groups[row_index]),
                "target_index": int(targets[row_index]),
                "prediction_index": int(predictions[row_index]),
                "category": category,
                "lambda_ratio": float(lambda_ratio[row_index]),
                "adaptive_lambda": float(adaptive_lambda[row_index]),
                "fixed_target_gradient": float(fixed_gradient[row_index]),
                "adaptive_target_gradient": float(adaptive_gradient[row_index]),
                "target_gradient_ratio": float(gradient_ratio[row_index]),
            }
        )

    return (
        {
            "split": name,
            "source_csv": str(Path(path).resolve()),
            "samples": int(len(targets)),
            "class_count": int(probabilities.shape[1]),
            "source_groups": int(len(np.unique(groups))),
            "error_count": int(wrong.sum()),
            "focus_false_positive_count": int(focus_fp.sum()),
            "focus_false_negative_count": int(focus_fn.sum()),
            "auc_tasks": auc_tasks,
            "lambda_ratio": {
                **_describe(lambda_ratio),
                "p90_minus_p10": float(lambda_p90 - lambda_p10),
            },
            "absolute_target_gradient_change": _describe(gradient_change),
            "categories": category_stats,
        },
        rows,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_adaptive_ldr_readiness(
    *,
    train_csv: Path,
    validation_csv: Path,
    output_dir: Path,
    focus_class_index: int = 1,
    lambda_ref: float = 1.0,
    alpha: float = 2.0,
    margin: float = 1.0,
    bootstrap_replicates: int = 500,
    seed: int = 42,
    min_error_auc: float = 0.75,
    min_focus_error_auc: float = 0.70,
    min_fp_vs_fn_auc: float = 0.40,
    min_lambda_p90_p10_span: float = 0.05,
    min_median_gradient_change: float = 0.01,
) -> Dict[str, object]:
    train, train_rows = _audit_split(
        "train",
        train_csv,
        focus_class_index=focus_class_index,
        lambda_ref=lambda_ref,
        alpha=alpha,
        margin=margin,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    validation, validation_rows = _audit_split(
        "validation",
        validation_csv,
        focus_class_index=focus_class_index,
        lambda_ref=lambda_ref,
        alpha=alpha,
        margin=margin,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 100,
    )
    if train["class_count"] != validation["class_count"]:
        raise ValueError("Train and validation class counts do not match.")

    blockers = []
    checks: Dict[str, object] = {}
    for split_name, payload in (("train", train), ("validation", validation)):
        split_checks: Dict[str, object] = {}
        for task, threshold in (
            ("error_detection_all", min_error_auc),
            ("predicted_focus_fp_detection", min_focus_error_auc),
            ("true_focus_fn_detection", min_focus_error_auc),
            ("focus_fp_vs_fn_direction", min_fp_vs_fn_auc),
        ):
            auc = payload["auc_tasks"][task]["auc"]
            split_checks[task] = {"auc": auc, "minimum": float(threshold)}
            if auc is None or float(auc) < float(threshold):
                blockers.append(f"{split_name}_{task}_auc_below_min:{auc}<{threshold}")
        span = float(payload["lambda_ratio"]["p90_minus_p10"])
        median_gradient_change = float(payload["absolute_target_gradient_change"]["median"])
        split_checks["lambda_p90_p10_span"] = {
            "value": span,
            "minimum": float(min_lambda_p90_p10_span),
        }
        split_checks["median_absolute_target_gradient_change"] = {
            "value": median_gradient_change,
            "minimum": float(min_median_gradient_change),
        }
        if span < float(min_lambda_p90_p10_span):
            blockers.append(
                f"{split_name}_lambda_span_below_min:{span}<{min_lambda_p90_p10_span}"
            )
        if median_gradient_change < float(min_median_gradient_change):
            blockers.append(
                f"{split_name}_gradient_change_below_min:"
                f"{median_gradient_change}<{min_median_gradient_change}"
            )
        checks[split_name] = split_checks

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "sample_adaptive_ldr_diagnostics.csv", [*train_rows, *validation_rows])
    smoke_gate_ready = not blockers
    summary: Dict[str, object] = {
        "mode": "adaptive_ldr_readiness",
        "research_formula": (
            "lambda_i/lambda_ref = 1 - KL(softmax(logits/lambda_ref)||uniform) / "
            "(alpha*log(K)); lambda is detached for the model-gradient step"
        ),
        "note": (
            "No-test, no-train diagnostic. Exported probabilities recover logits up to an additive "
            "constant. Train predictions are in-sample and are used only to test the stage-2 update; "
            "validation is the primary generalization gate."
        ),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "parameters": {
            "focus_class_index": int(focus_class_index),
            "lambda_ref": float(lambda_ref),
            "alpha": float(alpha),
            "margin": float(margin),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
        },
        "thresholds": {
            "min_error_auc": float(min_error_auc),
            "min_focus_error_auc": float(min_focus_error_auc),
            "min_fp_vs_fn_auc": float(min_fp_vs_fn_auc),
            "min_lambda_p90_p10_span": float(min_lambda_p90_p10_span),
            "min_median_gradient_change": float(min_median_gradient_change),
        },
        "train": train,
        "validation": validation,
        "gate_checks": checks,
        "smoke_gate_ready": bool(smoke_gate_ready),
        "blocking_reasons": blockers,
        "decision": (
            "Eligible for one bounded adaptive-LDR smoke; this does not approve a probe or test use."
            if smoke_gate_ready
            else "Do not implement or smoke adaptive LDR on the current keeper. Its uncertainty ordering "
            "is not class1-FP-safe and/or the adaptive lambda and gradient differ too little from the "
            "already rejected fixed LDR route."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    (output_dir / "README.md").write_text(
        "# Adaptive LDR readiness\n\n"
        "This no-test diagnostic reproduces the official one-step ALDR lambda update from aligned "
        "probabilities and measures error ranking, class-1 FN/FP ordering, lambda spread, and the "
        "resulting target-gradient change. It writes no training manifest and does not modify data.\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the official adaptive LDR update is worth a bounded TRKH smoke."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--lambda-ref", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, object]:
    return build_adaptive_ldr_readiness(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        output_dir=args.output_dir,
        focus_class_index=args.focus_class_index,
        lambda_ref=args.lambda_ref,
        alpha=args.alpha,
        margin=args.margin,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "smoke_gate_ready": summary["smoke_gate_ready"],
                "blocking_reasons": summary["blocking_reasons"],
                "output": "summary.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

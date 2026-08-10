from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _sha256,
    _write_artifact_manifest,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _transition_summary,
    _write_prediction_audit,
)


SEED = 20260711
FOCUS_CLASS = 1
ALPHA_GRID = np.linspace(-1.0, 1.0, 201, dtype=np.float64)
LITERATURE = (
    "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
    "69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html",
    "https://proceedings.mlr.press/v139/kumar21b.html",
    "https://proceedings.mlr.press/v97/cotter19b.html",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the matched PWCA-minus-control log-probability residual "
            "can improve the frozen keeper under train-OOF recall and FP constraints."
        )
    )
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _guard_non_test_path(path: Path, *, label: str) -> None:
    normalized = str(Path(path).resolve()).replace("\\", "/").casefold()
    if "/test/" in normalized or "predictions_test" in normalized:
        raise ValueError(f"{label} must not reference test data: {path}")


def _read_prediction_csv(
    path: Path,
    *,
    control_prefix: str,
    pwca_prefix: str,
) -> Dict[str, np.ndarray]:
    _guard_non_test_path(path, label="prediction CSV")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"sample_index", "path", "target", "base_prediction"}
    for prefix in ("base", control_prefix, pwca_prefix):
        required.update(f"{prefix}_prob_{index}" for index in range(5))
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns {missing}: {path}")

    sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if np.unique(sample_index).size != sample_index.size:
        raise ValueError(f"sample_index is not unique: {path}")
    paths = np.asarray([row["path"] for row in rows], dtype=object)
    if any("/test/" in str(value).replace("\\", "/").casefold() for value in paths):
        raise ValueError(f"Prediction CSV contains test rows: {path}")
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)

    def probabilities(prefix: str) -> np.ndarray:
        values = np.asarray(
            [
                [float(row[f"{prefix}_prob_{index}"]) for index in range(5)]
                for row in rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or bool((values < 0.0).any()):
            raise ValueError(f"Invalid probabilities for {prefix}: {path}")
        totals = values.sum(axis=1, keepdims=True)
        if bool((totals <= 0.0).any()):
            raise ValueError(f"Zero probability sum for {prefix}: {path}")
        return values / totals

    return {
        "sample_index": sample_index,
        "paths": paths,
        "labels": labels,
        "base": probabilities("base"),
        "control": probabilities(control_prefix),
        "pwca": probabilities(pwca_prefix),
    }


def apply_keeper_relative_residual(
    base_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    pwca_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    base = np.asarray(base_probabilities, dtype=np.float64)
    control = np.asarray(control_probabilities, dtype=np.float64)
    pwca = np.asarray(pwca_probabilities, dtype=np.float64)
    if base.shape != control.shape or base.shape != pwca.shape or base.ndim != 2:
        raise ValueError("base/control/PWCA probabilities must have the same [N,C] shape")
    epsilon = 1e-8
    logits = np.log(np.clip(base, epsilon, 1.0)) + float(alpha) * (
        np.log(np.clip(pwca, epsilon, 1.0))
        - np.log(np.clip(control, epsilon, 1.0))
    )
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return (values / values.sum(axis=1, keepdims=True)).astype(np.float32)


def _focus_rates(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(probabilities).argmax(axis=1)
    positives = targets == FOCUS_CLASS
    predicted_positive = predictions == FOCUS_CLASS
    true_positive = int(np.sum(positives & predicted_positive))
    false_negative = int(np.sum(positives & ~predicted_positive))
    false_positive = int(np.sum(~positives & predicted_positive))
    true_negative = int(np.sum(~positives & ~predicted_positive))
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "recall": float(true_positive / max(1, true_positive + false_negative)),
        "false_positive_rate": float(false_positive / max(1, false_positive + true_negative)),
    }


def select_oof_residual_alpha(
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    pwca_probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
    alpha_grid: np.ndarray = ALPHA_GRID,
) -> Tuple[float, List[Dict[str, object]]]:
    base_metrics = _classification_metrics(labels, base_probabilities, class_names=class_names)
    base_rates = _focus_rates(labels, base_probabilities)
    rows: List[Dict[str, object]] = []
    eligible: List[Dict[str, object]] = []
    for alpha in np.asarray(alpha_grid, dtype=np.float64).reshape(-1):
        probabilities = apply_keeper_relative_residual(
            base_probabilities,
            control_probabilities,
            pwca_probabilities,
            float(alpha),
        )
        metrics = _classification_metrics(labels, probabilities, class_names=class_names)
        rates = _focus_rates(labels, probabilities)
        constraints = {
            "macro_preserved": float(metrics["macro_f1"])
            >= float(base_metrics["macro_f1"]) - 1e-12,
            "class1_f1_preserved": float(metrics["focus_f1"])
            >= float(base_metrics["focus_f1"]) - 1e-12,
            "class1_recall_preserved": float(rates["recall"])
            >= float(base_rates["recall"]) - 1e-12,
            "class1_fp_not_increased": int(rates["false_positive"])
            <= int(base_rates["false_positive"]),
        }
        row: Dict[str, object] = {
            "alpha": float(alpha),
            "macro_f1": float(metrics["macro_f1"]),
            "class1_f1": float(metrics["focus_f1"]),
            "class1_recall": float(rates["recall"]),
            "class1_false_positive": int(rates["false_positive"]),
            "eligible": bool(all(constraints.values())),
            "constraints": constraints,
        }
        rows.append(row)
        if bool(row["eligible"]):
            eligible.append(row)
    if not eligible:
        raise RuntimeError("The fixed alpha grid did not contain an eligible keeper identity")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["class1_f1"]),
            float(row["macro_f1"]),
            -abs(float(row["alpha"])),
        ),
    )
    return float(selected["alpha"]), rows


def _prediction_oracle(
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
) -> Dict[str, object]:
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    base_prediction = np.asarray(base_probabilities).argmax(axis=1)
    candidate_prediction = np.asarray(candidate_probabilities).argmax(axis=1)
    oracle_prediction = base_prediction.copy()
    fixable = (base_prediction != targets) & (candidate_prediction == targets)
    oracle_prediction[fixable] = candidate_prediction[fixable]
    probabilities = np.eye(len(class_names), dtype=np.float32)[oracle_prediction]
    return {
        "metrics": _classification_metrics(targets, probabilities, class_names=class_names),
        "fixable_rows": int(fixable.sum()),
        "candidate_harmable_rows": int(
            np.sum((base_prediction == targets) & (candidate_prediction != targets))
        ),
        "label_assisted": True,
        "selection_permission": False,
    }


def _gate(
    *,
    selected_alpha: float,
    train_base_metrics: Mapping[str, object],
    train_selected_metrics: Mapping[str, object],
    val_base_metrics: Mapping[str, object],
    val_selected_metrics: Mapping[str, object],
    train_base_rates: Mapping[str, float],
    train_selected_rates: Mapping[str, float],
    val_base_rates: Mapping[str, float],
    val_selected_rates: Mapping[str, float],
    train_transitions: Mapping[str, int],
    val_transitions: Mapping[str, int],
    val_samples: int,
) -> Dict[str, object]:
    thresholds = {
        "required_val_samples": 2606,
        "min_abs_nonzero_alpha": 1e-12,
        "min_oof_macro_gain": 0.001,
        "min_oof_class1_gain": 0.005,
        "min_val_macro_gain": 0.001,
        "min_val_class1_gain": 0.005,
    }
    observed = {
        "selected_alpha": float(selected_alpha),
        "oof_macro_gain": float(train_selected_metrics["macro_f1"])
        - float(train_base_metrics["macro_f1"]),
        "oof_class1_gain": float(train_selected_metrics["focus_f1"])
        - float(train_base_metrics["focus_f1"]),
        "val_macro_gain": float(val_selected_metrics["macro_f1"])
        - float(val_base_metrics["macro_f1"]),
        "val_class1_gain": float(val_selected_metrics["focus_f1"])
        - float(val_base_metrics["focus_f1"]),
    }
    checks = {
        "val_support_complete": int(val_samples) == int(thresholds["required_val_samples"]),
        "nonzero_residual_selected": abs(float(selected_alpha))
        >= float(thresholds["min_abs_nonzero_alpha"]),
        "oof_macro_gain": observed["oof_macro_gain"] >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": observed["oof_class1_gain"]
        >= float(thresholds["min_oof_class1_gain"]),
        "val_macro_gain": observed["val_macro_gain"] >= float(thresholds["min_val_macro_gain"]),
        "val_class1_gain": observed["val_class1_gain"]
        >= float(thresholds["min_val_class1_gain"]),
        "oof_class1_recall_preserved": float(train_selected_rates["recall"])
        >= float(train_base_rates["recall"]),
        "val_class1_recall_preserved": float(val_selected_rates["recall"])
        >= float(val_base_rates["recall"]),
        "oof_class1_fp_control": int(train_selected_rates["false_positive"])
        <= int(train_base_rates["false_positive"]),
        "val_class1_fp_control": int(val_selected_rates["false_positive"])
        <= int(val_base_rates["false_positive"]),
        "oof_net_corrections": int(train_transitions["corrections"])
        >= int(train_transitions["harms"]),
        "val_net_corrections": int(val_transitions["corrections"])
        >= int(val_transitions["harms"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "keeper_relative_pwca_residual_ready": not failed,
        "trainable_adapter_precheck_passed": not failed,
        "smoke_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    train_path = Path(args.train_predictions)
    val_path = Path(args.val_predictions)
    output_dir = Path(args.output_dir)
    _guard_non_test_path(output_dir, label="output directory")
    train = _read_prediction_csv(
        train_path,
        control_prefix="control_oof",
        pwca_prefix="pwca_oof",
    )
    val = _read_prediction_csv(
        val_path,
        control_prefix="control",
        pwca_prefix="pwca",
    )
    class_names = [str(index) for index in range(5)]
    selected_alpha, alpha_rows = select_oof_residual_alpha(
        train["labels"],
        train["base"],
        train["control"],
        train["pwca"],
        class_names=class_names,
    )
    train_selected = apply_keeper_relative_residual(
        train["base"], train["control"], train["pwca"], selected_alpha
    )
    val_selected = apply_keeper_relative_residual(
        val["base"], val["control"], val["pwca"], selected_alpha
    )
    train_base_metrics = _classification_metrics(
        train["labels"], train["base"], class_names=class_names
    )
    train_selected_metrics = _classification_metrics(
        train["labels"], train_selected, class_names=class_names
    )
    val_base_metrics = _classification_metrics(
        val["labels"], val["base"], class_names=class_names
    )
    val_selected_metrics = _classification_metrics(
        val["labels"], val_selected, class_names=class_names
    )
    train_base_rates = _focus_rates(train["labels"], train["base"])
    train_selected_rates = _focus_rates(train["labels"], train_selected)
    val_base_rates = _focus_rates(val["labels"], val["base"])
    val_selected_rates = _focus_rates(val["labels"], val_selected)
    train_transitions = _transition_summary(
        train["labels"], train["base"], train_selected
    )
    val_transitions = _transition_summary(val["labels"], val["base"], val_selected)
    gate = _gate(
        selected_alpha=selected_alpha,
        train_base_metrics=train_base_metrics,
        train_selected_metrics=train_selected_metrics,
        val_base_metrics=val_base_metrics,
        val_selected_metrics=val_selected_metrics,
        train_base_rates=train_base_rates,
        train_selected_rates=train_selected_rates,
        val_base_rates=val_base_rates,
        val_selected_rates=val_selected_rates,
        train_transitions=train_transitions,
        val_transitions=val_transitions,
        val_samples=int(val["labels"].size),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_prediction_audit(
        output_dir / "train_oof_selected_predictions.csv",
        labels=train["labels"],
        sample_index=train["sample_index"],
        paths=train["paths"],
        base_probabilities=train["base"],
        variants={"selected_residual": train_selected},
    )
    _write_prediction_audit(
        output_dir / "val_selected_predictions.csv",
        labels=val["labels"],
        sample_index=val["sample_index"],
        paths=val["paths"],
        base_probabilities=val["base"],
        variants={"selected_residual": val_selected},
    )
    (output_dir / "alpha_oof_audit.json").write_text(
        json.dumps(alpha_rows, indent=2), encoding="utf-8"
    )
    summary: Dict[str, object] = {
        "mode": "pwca_keeper_relative_residual_readiness",
        "guardrail": (
            "Alpha is selected only from train OOF. Validation is transfer evaluation only. "
            "No test, image training, checkpoint, deploy artifact, or raw-data edit."
        ),
        "inputs": {
            "train_predictions": str(train_path.resolve()),
            "val_predictions": str(val_path.resolve()),
            "sha256": {"train": _sha256(train_path), "val": _sha256(val_path)},
        },
        "protocol": {
            "formula": "log(p_keeper) + alpha * (log(p_pwca) - log(p_control))",
            "alpha_grid": {
                "minimum": float(ALPHA_GRID.min()),
                "maximum": float(ALPHA_GRID.max()),
                "count": int(ALPHA_GRID.size),
                "step": float(ALPHA_GRID[1] - ALPHA_GRID[0]),
            },
            "selection_split": "train_oof_only",
            "selection_constraints": [
                "macro_f1>=keeper",
                "class1_f1>=keeper",
                "class1_recall>=keeper",
                "class1_false_positive_count<=keeper",
            ],
            "selected_alpha": float(selected_alpha),
            "candidate_sweep": False,
            "validation_selection": False,
            "seed": SEED,
        },
        "support": {
            "train_rows": int(train["labels"].size),
            "val_rows": int(val["labels"].size),
            "train_unique_sample_index": int(np.unique(train["sample_index"]).size),
            "val_unique_sample_index": int(np.unique(val["sample_index"]).size),
        },
        "metrics": {
            "train_oof_keeper": train_base_metrics,
            "train_oof_selected": train_selected_metrics,
            "val_keeper": val_base_metrics,
            "val_selected": val_selected_metrics,
        },
        "focus_rates": {
            "train_oof_keeper": train_base_rates,
            "train_oof_selected": train_selected_rates,
            "val_keeper": val_base_rates,
            "val_selected": val_selected_rates,
        },
        "transitions": {
            "train_oof_keeper_to_selected": train_transitions,
            "val_keeper_to_selected": val_transitions,
        },
        "base_or_pwca_prediction_oracle": {
            "train_oof": _prediction_oracle(
                train["labels"],
                train["base"],
                train["pwca"],
                class_names=class_names,
            ),
            "val": _prediction_oracle(
                val["labels"], val["base"], val["pwca"], class_names=class_names
            ),
        },
        "eligible_alpha_count": int(sum(bool(row["eligible"]) for row in alpha_rows)),
        "gate": gate,
        "literature": list(LITERATURE),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "image_model_trained": False,
        "model_written": False,
        "checkpoint_written": False,
        "trainable_manifest_written": False,
    }
    summary["artifact_manifest_path"] = str(
        (output_dir / "artifact_manifest.json").resolve()
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    readme = [
        "# Keeper-Relative PWCA Residual Readiness",
        "",
        f"- Selected train-OOF alpha: `{selected_alpha:.6f}`",
        f"- Eligible alpha count: `{summary['eligible_alpha_count']}`",
        f"- Train OOF keeper/selected macro-class1: `{float(train_base_metrics['macro_f1']):.6f}/{float(train_base_metrics['focus_f1']):.6f}` -> `{float(train_selected_metrics['macro_f1']):.6f}/{float(train_selected_metrics['focus_f1']):.6f}`",
        f"- Validation keeper/selected macro-class1: `{float(val_base_metrics['macro_f1']):.6f}/{float(val_base_metrics['focus_f1']):.6f}` -> `{float(val_selected_metrics['macro_f1']):.6f}/{float(val_selected_metrics['focus_f1']):.6f}`",
        f"- Trainable-adapter precheck passed: `{str(bool(gate['trainable_adapter_precheck_passed'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "This closes only scalar post-hoc residual transfer. It does not train or reject a new token adapter.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    _write_artifact_manifest(
        output_dir,
        mode="pwca_keeper_relative_residual_readiness_evidence_manifest",
    )
    print(
        json.dumps(
            {
                "selected_alpha": selected_alpha,
                "eligible_alpha_count": summary["eligible_alpha_count"],
                "metrics": summary["metrics"],
                "focus_rates": summary["focus_rates"],
                "gate": gate,
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_audit(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

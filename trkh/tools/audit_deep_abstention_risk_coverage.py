from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


DEFAULT_COVERAGES = (1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.50)


def _reject_test_input(path: Path, split: str) -> None:
    if str(split).strip().lower() not in {"val", "validation"}:
        raise ValueError("Deep-abstention model selection is validation-only; --split must be val.")
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if re.search(r"(^|[/_.-])test([/_.-]|$)", normalized):
        raise ValueError(f"Refusing test-derived predictions: {path}")


def _first_present(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _read_prediction_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"Prediction CSV is empty: {path}")

    rows: List[Dict[str, object]] = []
    for row_index, row in enumerate(source_rows):
        target = _first_present(row, ("target_index", "y_true", "target", "label"))
        prediction = _first_present(
            row,
            ("prediction_index", "y_pred", "prediction", "pred"),
        )
        abstention = _first_present(
            row,
            ("abstention_probability", "deep_abstention_probability"),
        )
        if not target or not prediction or not abstention:
            raise ValueError(
                "Prediction CSV must contain target, prediction, and "
                f"abstention probability columns; invalid row {row_index}."
            )
        q = float(abstention)
        if not math.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(f"Invalid abstention probability at row {row_index}: {q}")
        normalized: Dict[str, object] = dict(row)
        normalized["row_index"] = int(row_index)
        normalized["target_index"] = int(target)
        normalized["prediction_index"] = int(prediction)
        normalized["abstention_probability"] = q
        normalized["error"] = int(int(target) != int(prediction))
        rows.append(normalized)
    return rows


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_indices: Sequence[int],
) -> Tuple[float, Dict[int, Dict[str, float]]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(class_indices),
        zero_division=0,
    )
    per_class = {
        int(class_index): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_index in enumerate(class_indices)
    }
    return float(np.mean(f1)), per_class


def analyze_risk_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    focus_class: int = 1,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    if not rows:
        raise ValueError("At least one prediction row is required.")
    targets = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    predictions = np.asarray(
        [int(row["prediction_index"]) for row in rows], dtype=np.int64
    )
    abstention = np.asarray(
        [float(row["abstention_probability"]) for row in rows], dtype=np.float64
    )
    errors = (targets != predictions).astype(np.int64)
    class_indices = sorted(set(targets.tolist()) | set(predictions.tolist()))
    order = np.argsort(abstention, kind="stable")
    cumulative_errors = np.cumsum(errors[order])
    retained_counts = np.arange(1, len(rows) + 1, dtype=np.int64)
    selective_risk = cumulative_errors / retained_counts
    full_curve = [
        {
            "retained_count": int(count),
            "abstained_count": int(len(rows) - count),
            "coverage": float(count / len(rows)),
            "risk": float(selective_risk[count - 1]),
            "max_retained_abstention_probability": float(abstention[order[count - 1]]),
        }
        for count in retained_counts
    ]

    checkpoints: List[Dict[str, object]] = []
    for requested_coverage in coverages:
        requested_coverage = min(max(float(requested_coverage), 0.0), 1.0)
        retained_count = max(1, min(len(rows), int(math.ceil(len(rows) * requested_coverage))))
        retained = order[:retained_count]
        macro_f1, per_class = _classification_metrics(
            targets[retained],
            predictions[retained],
            class_indices,
        )
        focus_metrics = per_class.get(
            int(focus_class),
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        )
        checkpoints.append(
            {
                "requested_coverage": requested_coverage,
                "coverage": float(retained_count / len(rows)),
                "retained_count": int(retained_count),
                "abstained_count": int(len(rows) - retained_count),
                "risk": float(errors[retained].mean()),
                "accuracy": float(1.0 - errors[retained].mean()),
                "macro_f1": macro_f1,
                "focus_class": int(focus_class),
                "focus_precision": float(focus_metrics["precision"]),
                "focus_recall": float(focus_metrics["recall"]),
                "focus_f1": float(focus_metrics["f1"]),
                "focus_support": int(focus_metrics["support"]),
                "min_per_class_f1": float(
                    min(float(values["f1"]) for values in per_class.values())
                ),
                "per_class": per_class,
            }
        )

    base_macro_f1, base_per_class = _classification_metrics(
        targets,
        predictions,
        class_indices,
    )
    error_count = int(errors.sum())
    if 0 < error_count < len(rows):
        error_auroc = float(roc_auc_score(errors, abstention))
        error_average_precision = float(average_precision_score(errors, abstention))
    else:
        error_auroc = 0.5
        error_average_precision = float(error_count / len(rows))
    descending = np.argsort(-abstention, kind="stable")
    top_tenth_count = max(1, int(math.ceil(0.10 * len(rows))))
    top_tenth = descending[:top_tenth_count]
    captured_errors = int(errors[top_tenth].sum())
    ranked_rows: List[Dict[str, object]] = []
    for rank, row_index in enumerate(descending, start=1):
        ranked = dict(rows[int(row_index)])
        ranked["error"] = int(errors[int(row_index)])
        ranked["abstention_rank"] = int(rank)
        ranked_rows.append(ranked)

    gate_coverages = [
        float(checkpoint["coverage"])
        for checkpoint in checkpoints
        if float(checkpoint["min_per_class_f1"]) >= 0.98
    ]
    summary: Dict[str, object] = {
        "sample_count": int(len(rows)),
        "error_count": error_count,
        "error_rate": float(errors.mean()),
        "base_macro_f1": base_macro_f1,
        "base_per_class": base_per_class,
        "focus_class": int(focus_class),
        "abstention": {
            "mean": float(abstention.mean()),
            "max": float(abstention.max()),
            "p50": float(np.quantile(abstention, 0.50)),
            "p90": float(np.quantile(abstention, 0.90)),
            "p99": float(np.quantile(abstention, 0.99)),
            "mean_correct": float(abstention[errors == 0].mean())
            if bool(np.any(errors == 0))
            else 0.0,
            "mean_incorrect": float(abstention[errors == 1].mean())
            if bool(np.any(errors == 1))
            else 0.0,
        },
        "error_detection": {
            "auroc": error_auroc,
            "average_precision": error_average_precision,
            "top_10_percent_error_capture": float(captured_errors / max(1, error_count)),
            "top_10_percent_error_precision": float(captured_errors / top_tenth_count),
        },
        "aurc": float(selective_risk.mean()),
        "coverage_checkpoints": checkpoints,
        "max_checkpoint_coverage_all_classes_f1_ge_098": (
            max(gate_coverages) if gate_coverages else 0.0
        ),
    }
    return summary, full_curve, ranked_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                continue
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_curve(curve: Sequence[Mapping[str, object]], path: Path) -> None:
    coverage = [float(row["coverage"]) for row in curve]
    risk = [float(row["risk"]) for row in curve]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(coverage, risk, color="#155E75", linewidth=2.0)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk")
    axis.set_xlim(0.0, 1.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only risk-coverage audit for a TRKH DAC checkpoint."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--focus-class", type=int, default=1)
    args = parser.parse_args()

    _reject_test_input(args.predictions, args.split)
    rows = _read_prediction_rows(args.predictions)
    summary, curve, ranked_rows = analyze_risk_coverage(
        rows,
        focus_class=args.focus_class,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["input_predictions"] = str(args.predictions.resolve())
    summary["split"] = "val"
    with (args.output_dir / "risk_coverage_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    _write_csv(args.output_dir / "risk_coverage_curve.csv", curve)
    _write_csv(args.output_dir / "abstention_ranked_cases.csv", ranked_rows)
    _write_csv(
        args.output_dir / "risk_coverage_checkpoints.csv",
        summary["coverage_checkpoints"],
    )
    _plot_curve(curve, args.output_dir / "risk_coverage_curve.png")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from trkh.evaluation.metrics import build_metrics


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_(.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a small class-logit bias on validation predictions, freeze it, "
            "then evaluate the frozen calibration on test predictions."
        ),
    )
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=("macro", "macro_class1", "fair"),
        default="macro_class1",
    )
    parser.add_argument("--class1-index", type=int, default=1)
    parser.add_argument("--class1-weight", type=float, default=0.25)
    parser.add_argument("--reference-class", type=int, default=-1)
    parser.add_argument("--max-abs-bias", type=float, default=1.0)
    parser.add_argument(
        "--search-steps",
        type=str,
        default="0.20,0.10,0.05,0.02,0.01",
        help="Coordinate-descent step sizes, coarse to fine.",
    )
    parser.add_argument("--max-passes-per-step", type=int, default=10)
    parser.add_argument("--fair-min-weight", type=float, default=0.25)
    parser.add_argument("--fair-gap-penalty", type=float, default=1.5)
    parser.add_argument("--fair-gap-target", type=float, default=0.05)
    return parser.parse_args()


def _probability_columns(fieldnames: Sequence[str]) -> Tuple[List[str], List[str]]:
    indexed: Dict[int, Tuple[str, str]] = {}
    for column in fieldnames:
        match = PROBABILITY_COLUMN.match(str(column))
        if match is None:
            continue
        class_index = int(match.group(1))
        class_name = str(match.group(2) or class_index)
        if class_index in indexed:
            raise ValueError(f"Duplicate probability column for class {class_index}.")
        indexed[class_index] = (str(column), class_name)
    if not indexed:
        raise ValueError("Prediction CSV does not contain prob_<index> columns.")
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ValueError(
            f"Probability class indices must be contiguous from zero: {sorted(indexed)}"
        )
    return (
        [indexed[index][0] for index in expected],
        [indexed[index][1] for index in expected],
    )


def read_prediction_csv(
    path: Path,
) -> Tuple[List[Dict[str, str]], List[str], Tensor, Tensor]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    if "target_index" not in fieldnames:
        raise ValueError(f"Prediction CSV is missing target_index: {path}")

    probability_columns, class_names = _probability_columns(fieldnames)
    targets = torch.tensor(
        [int(row["target_index"]) for row in rows],
        dtype=torch.int64,
    )
    probabilities = torch.tensor(
        [
            [float(row[column]) for column in probability_columns]
            for row in rows
        ],
        dtype=torch.float64,
    )
    if not torch.isfinite(probabilities).all():
        raise ValueError(f"Prediction CSV contains non-finite probabilities: {path}")
    probabilities = probabilities.clamp(min=1e-12)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp(min=1e-12)
    if targets.min().item() < 0 or targets.max().item() >= len(class_names):
        raise ValueError(f"target_index is outside probability columns: {path}")
    return rows, class_names, targets, probabilities


def metrics_for_bias(
    log_probabilities: Tensor,
    targets: Tensor,
    class_names: Sequence[str],
    bias: Sequence[float],
) -> Dict[str, object]:
    bias_tensor = torch.as_tensor(
        list(bias),
        dtype=log_probabilities.dtype,
    ).view(1, -1)
    probabilities = torch.softmax(log_probabilities + bias_tensor, dim=1)
    predictions = probabilities.argmax(dim=1)
    return build_metrics(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )


def objective_score(
    metrics: Mapping[str, object],
    *,
    objective: str,
    class1_index: int,
    class1_weight: float,
    fair_min_weight: float,
    fair_gap_penalty: float,
    fair_gap_target: float,
) -> float:
    macro_f1 = float(metrics["macro_f1"])
    per_class = list(metrics["per_class"])
    class_f1 = [float(item["f1"]) for item in per_class]
    class1_f1 = class_f1[int(class1_index)]
    if objective == "macro":
        return macro_f1
    if objective == "macro_class1":
        return macro_f1 + max(0.0, float(class1_weight)) * class1_f1
    if objective == "fair":
        class_gap = max(class_f1) - min(class_f1)
        return (
            macro_f1
            + max(0.0, float(fair_min_weight)) * min(class_f1)
            - max(0.0, float(fair_gap_penalty))
            * max(0.0, class_gap - max(0.0, float(fair_gap_target)))
        )
    raise ValueError(f"Unsupported objective: {objective}")


def _candidate_key(
    metrics: Mapping[str, object],
    score: float,
    class1_index: int,
    bias: Sequence[float],
) -> Tuple[float, float, float, float, float]:
    class1_f1 = float(list(metrics["per_class"])[int(class1_index)]["f1"])
    return (
        float(score),
        float(metrics["macro_f1"]),
        class1_f1,
        float(metrics["accuracy"]),
        -sum(abs(float(value)) for value in bias),
    )


def fit_logit_bias(
    log_probabilities: Tensor,
    targets: Tensor,
    class_names: Sequence[str],
    *,
    objective: str = "macro_class1",
    class1_index: int = 1,
    class1_weight: float = 0.25,
    reference_class: int = -1,
    max_abs_bias: float = 1.0,
    search_steps: Sequence[float] = (0.2, 0.1, 0.05, 0.02, 0.01),
    max_passes_per_step: int = 10,
    fair_min_weight: float = 0.25,
    fair_gap_penalty: float = 1.5,
    fair_gap_target: float = 0.05,
) -> Tuple[List[float], Dict[str, object], List[Dict[str, object]]]:
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Logit calibration requires at least two classes.")
    if not 0 <= int(class1_index) < num_classes:
        raise ValueError("class1_index is outside the class range.")
    if int(reference_class) < 0:
        reference_class = num_classes - 1
    if not 0 <= int(reference_class) < num_classes:
        raise ValueError("reference_class is outside the class range.")
    steps = [float(step) for step in search_steps if float(step) > 0.0]
    if not steps:
        raise ValueError("search_steps must contain at least one positive value.")

    max_abs_bias = max(0.0, float(max_abs_bias))
    bias = [0.0 for _ in range(num_classes)]
    best_metrics = metrics_for_bias(log_probabilities, targets, class_names, bias)
    best_score = objective_score(
        best_metrics,
        objective=objective,
        class1_index=class1_index,
        class1_weight=class1_weight,
        fair_min_weight=fair_min_weight,
        fair_gap_penalty=fair_gap_penalty,
        fair_gap_target=fair_gap_target,
    )
    trace: List[Dict[str, object]] = []

    for step in steps:
        for pass_index in range(max(1, int(max_passes_per_step))):
            pass_changed = False
            for class_index in range(num_classes):
                if class_index == int(reference_class):
                    continue
                local_bias = list(bias)
                local_metrics = best_metrics
                local_score = best_score
                local_key = _candidate_key(
                    local_metrics,
                    local_score,
                    class1_index,
                    local_bias,
                )
                center = float(bias[class_index])
                for offset in range(-5, 6):
                    candidate = list(bias)
                    candidate[class_index] = min(
                        max_abs_bias,
                        max(-max_abs_bias, center + float(offset) * step),
                    )
                    candidate_metrics = metrics_for_bias(
                        log_probabilities,
                        targets,
                        class_names,
                        candidate,
                    )
                    candidate_score = objective_score(
                        candidate_metrics,
                        objective=objective,
                        class1_index=class1_index,
                        class1_weight=class1_weight,
                        fair_min_weight=fair_min_weight,
                        fair_gap_penalty=fair_gap_penalty,
                        fair_gap_target=fair_gap_target,
                    )
                    candidate_key = _candidate_key(
                        candidate_metrics,
                        candidate_score,
                        class1_index,
                        candidate,
                    )
                    if candidate_key > local_key:
                        local_bias = candidate
                        local_metrics = candidate_metrics
                        local_score = candidate_score
                        local_key = candidate_key
                if not math.isclose(
                    local_bias[class_index],
                    bias[class_index],
                    abs_tol=1e-12,
                ):
                    bias = local_bias
                    best_metrics = local_metrics
                    best_score = local_score
                    pass_changed = True

            trace.append(
                {
                    "step": float(step),
                    "pass": int(pass_index + 1),
                    "changed": bool(pass_changed),
                    "objective_score": float(best_score),
                    "macro_f1": float(best_metrics["macro_f1"]),
                    "class1_f1": float(
                        list(best_metrics["per_class"])[int(class1_index)]["f1"]
                    ),
                    "accuracy": float(best_metrics["accuracy"]),
                    "bias": [float(value) for value in bias],
                }
            )
            if not pass_changed:
                break
    return bias, best_metrics, trace


def _calibrated_probabilities(log_probabilities: Tensor, bias: Sequence[float]) -> Tensor:
    bias_tensor = torch.as_tensor(
        list(bias),
        dtype=log_probabilities.dtype,
    ).view(1, -1)
    return torch.softmax(log_probabilities + bias_tensor, dim=1)


def write_calibrated_predictions(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    class_names: Sequence[str],
    probabilities: Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    base_fields = list(rows[0].keys())
    extra_fields = [
        "calibrated_prediction_index",
        "calibrated_prediction_name",
        "calibrated_confidence",
        "calibrated_correct",
        *[
            f"calibrated_prob_{index}_{class_name}"
            for index, class_name in enumerate(class_names)
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base_fields, *extra_fields])
        writer.writeheader()
        for row_index, source_row in enumerate(rows):
            prediction = int(predictions[row_index].item())
            target = int(source_row["target_index"])
            row = dict(source_row)
            row.update(
                {
                    "calibrated_prediction_index": prediction,
                    "calibrated_prediction_name": class_names[prediction],
                    "calibrated_confidence": float(confidence[row_index].item()),
                    "calibrated_correct": int(prediction == target),
                }
            )
            row.update(
                {
                    f"calibrated_prob_{index}_{class_name}": float(
                        probabilities[row_index, index].item()
                    )
                    for index, class_name in enumerate(class_names)
                }
            )
            writer.writerow(row)


def _parse_steps(value: str) -> List[float]:
    steps = [
        float(item.strip())
        for item in str(value).replace(";", ",").split(",")
        if item.strip()
    ]
    if not steps or any(step <= 0.0 for step in steps):
        raise ValueError("--search-steps must contain positive comma-separated values.")
    return steps


def main() -> None:
    args = parse_args()
    val_rows, class_names, val_targets, val_probabilities = read_prediction_csv(
        args.val_predictions
    )
    reference_class = (
        len(class_names) - 1
        if int(args.reference_class) < 0
        else int(args.reference_class)
    )
    search_steps = _parse_steps(args.search_steps)
    val_log_probabilities = val_probabilities.log()
    baseline_val = metrics_for_bias(
        val_log_probabilities,
        val_targets,
        class_names,
        [0.0 for _ in class_names],
    )
    bias, adjusted_val, trace = fit_logit_bias(
        val_log_probabilities,
        val_targets,
        class_names,
        objective=args.objective,
        class1_index=args.class1_index,
        class1_weight=args.class1_weight,
        reference_class=reference_class,
        max_abs_bias=args.max_abs_bias,
        search_steps=search_steps,
        max_passes_per_step=args.max_passes_per_step,
        fair_min_weight=args.fair_min_weight,
        fair_gap_penalty=args.fair_gap_penalty,
        fair_gap_target=args.fair_gap_target,
    )

    # The test file is intentionally loaded only after validation has frozen the bias.
    test_rows, test_class_names, test_targets, test_probabilities = read_prediction_csv(
        args.test_predictions
    )
    if test_class_names != class_names:
        raise ValueError(
            f"Validation/test class columns differ: {class_names} vs {test_class_names}"
        )
    test_log_probabilities = test_probabilities.log()
    baseline_test = metrics_for_bias(
        test_log_probabilities,
        test_targets,
        class_names,
        [0.0 for _ in class_names],
    )
    adjusted_test = metrics_for_bias(
        test_log_probabilities,
        test_targets,
        class_names,
        bias,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_calibrated_probabilities = _calibrated_probabilities(
        val_log_probabilities,
        bias,
    )
    test_calibrated_probabilities = _calibrated_probabilities(
        test_log_probabilities,
        bias,
    )
    write_calibrated_predictions(
        output_dir / "val_predictions_calibrated.csv",
        val_rows,
        class_names,
        val_calibrated_probabilities,
    )
    write_calibrated_predictions(
        output_dir / "test_predictions_calibrated.csv",
        test_rows,
        class_names,
        test_calibrated_probabilities,
    )
    with (output_dir / "val_search_trace.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "step",
            "pass",
            "changed",
            "objective_score",
            "macro_f1",
            "class1_f1",
            "accuracy",
            *[f"bias_{index}" for index in range(len(class_names))],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in trace:
            writer.writerow(
                {
                    key: value
                    for key, value in item.items()
                    if key != "bias"
                }
                | {
                    f"bias_{index}": float(value)
                    for index, value in enumerate(item["bias"])
                }
            )

    summary = {
        "mode": "validation_only_class_logit_bias",
        "val_predictions": str(Path(args.val_predictions).resolve()),
        "test_predictions": str(Path(args.test_predictions).resolve()),
        "class_names": list(class_names),
        "objective": str(args.objective),
        "class1_index": int(args.class1_index),
        "class1_weight": float(args.class1_weight),
        "reference_class": int(reference_class),
        "max_abs_bias": float(args.max_abs_bias),
        "search_steps": search_steps,
        "selected_bias": [float(value) for value in bias],
        "baseline_val": baseline_val,
        "adjusted_val": adjusted_val,
        "baseline_test": baseline_test,
        "adjusted_test": adjusted_test,
        "leakage_control": (
            "Bias search uses validation predictions only. Test predictions are loaded "
            "after the bias is frozen and are never used to select a parameter."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "selected_bias": summary["selected_bias"],
                "val_macro_f1": adjusted_val["macro_f1"],
                "val_class1_f1": adjusted_val["per_class"][int(args.class1_index)]["f1"],
                "test_macro_f1_before": baseline_test["macro_f1"],
                "test_macro_f1_after": adjusted_test["macro_f1"],
                "test_class1_f1_before": baseline_test["per_class"][int(args.class1_index)]["f1"],
                "test_class1_f1_after": adjusted_test["per_class"][int(args.class1_index)]["f1"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

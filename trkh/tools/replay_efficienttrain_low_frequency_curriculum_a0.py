from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


CONDITIONS = ("clean", "lighting_dim", "lighting_bright", "low_contrast")
SHIFTED_CONDITIONS = CONDITIONS[1:]
VIEWS = ("early_b176", "middle_b224")
FIT_FOLDS = (1, 2, 3, 4)
EXPECTED_ROWS = 607
EXPECTED_TP = 421
EXPECTED_FP = 186
KNOWN_NEAR_TIE_INDEX = 3657
KNOWN_NEAR_TIE_COHORT_POSITION = 337
SENSITIVITY_ROWS = EXPECTED_ROWS - 1
TP_QUANTILE = 0.97


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empirical_quantile_higher(values: Sequence[float], quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError("The empirical quantile requires a nonempty vector.")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("Quantile must be in [0, 1].")
    index = int(math.ceil(float(quantile) * float(ordered.size - 1)))
    return float(ordered[index])


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if label_values.shape != score_values.shape or label_values.ndim != 1:
        raise ValueError("AUROC labels and scores must be aligned vectors.")
    positives = label_values == 1
    negatives = label_values == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both classes.")

    order = np.argsort(score_values, kind="mergesort")
    sorted_scores = score_values[order]
    ranks = np.empty(score_values.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (float(start + 1) + float(end)) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[positives].sum())
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _read_rows(path: Path) -> list[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _condition_rows(
    rows: Sequence[Mapping[str, str]],
    condition: str,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> list[Mapping[str, str]]:
    selected = [row for row in rows if str(row["condition"]) == condition]
    selected.sort(key=lambda row: int(row["cohort_position"]))
    if len(selected) != int(expected_rows):
        raise ValueError(
            f"Condition {condition} has {len(selected)} rows, expected {expected_rows}."
        )
    positions = [int(row["cohort_position"]) for row in selected]
    if positions != _expected_cohort_positions(expected_rows):
        raise ValueError(f"Condition {condition} has noncanonical cohort positions.")
    return selected


def _expected_cohort_positions(expected_rows: int) -> list[int]:
    if int(expected_rows) == EXPECTED_ROWS:
        return list(range(EXPECTED_ROWS))
    if int(expected_rows) == SENSITIVITY_ROWS:
        return [
            position
            for position in range(EXPECTED_ROWS)
            if position != KNOWN_NEAR_TIE_COHORT_POSITION
        ]
    raise ValueError(f"Unsupported expected cohort row count: {expected_rows}.")


def _thresholds_from_clean(
    clean: Sequence[Mapping[str, str]], view: str
) -> Dict[int, float]:
    score_key = f"{view}_suppression_score"
    thresholds: Dict[int, float] = {}
    for held_fold in FIT_FOLDS:
        fit_values = [
            float(row[score_key])
            for row in clean
            if str(row["cohort"]) == "tp" and int(row["fold"]) != held_fold
        ]
        thresholds[held_fold] = empirical_quantile_higher(
            fit_values, TP_QUANTILE
        )
    return thresholds


def _hard_decisions(
    clean: Sequence[Mapping[str, str]], view: str
) -> Dict[str, object]:
    prediction_key = f"{view}_prediction"
    corrections = 0
    harms = 0
    tp_breaks = 0
    fp_removals = 0
    transitions: Counter[str] = Counter()
    for row in clean:
        target = int(row["target"])
        native = int(row["native_b256_prediction"])
        candidate = int(row[prediction_key])
        transitions[f"{native}->{candidate}"] += 1
        corrections += int(native != target and candidate == target)
        harms += int(native == target and candidate != target)
        tp_breaks += int(
            str(row["cohort"]) == "tp" and native == 1 and candidate != 1
        )
        fp_removals += int(
            str(row["cohort"]) == "fp" and native == 1 and candidate != 1
        )
    return {
        "corrections": corrections,
        "harms": harms,
        "net_corrections": corrections - harms,
        "class1_tp_breaks": tp_breaks,
        "restricted_fp_removals": fp_removals,
        "transitions": dict(sorted(transitions.items())),
    }


def _view_metrics(
    rows: Sequence[Mapping[str, str]],
    view: str,
    *,
    fixed_thresholds: Optional[Mapping[int, float]] = None,
    expected_rows: int = EXPECTED_ROWS,
) -> Dict[str, object]:
    by_condition = {
        condition: _condition_rows(
            rows, condition, expected_rows=expected_rows
        )
        for condition in CONDITIONS
    }
    clean = by_condition["clean"]
    thresholds = (
        {int(key): float(value) for key, value in fixed_thresholds.items()}
        if fixed_thresholds is not None
        else _thresholds_from_clean(clean, view)
    )
    score_key = f"{view}_suppression_score"
    threshold_key = f"{view}_oof_threshold"
    rejected_key = f"{view}_oof_rejected"
    condition_metrics: Dict[str, object] = {}
    stored_score_error = 0.0
    stored_threshold_error = 0.0
    stored_rejection_mismatches = 0
    for condition in CONDITIONS:
        selected = by_condition[condition]
        labels = [int(str(row["cohort"]) == "fp") for row in selected]
        scores = [float(row[score_key]) for row in selected]
        decisions: list[bool] = []
        for row, score in zip(selected, scores):
            fold = int(row["fold"])
            threshold = thresholds[fold]
            recomputed_score = float(row["native_b256_prob_1"]) - float(
                row[f"{view}_prob_1"]
            )
            stored_score_error = max(
                stored_score_error, abs(score - recomputed_score)
            )
            stored_threshold_error = max(
                stored_threshold_error, abs(float(row[threshold_key]) - threshold)
            )
            rejected = score > threshold
            stored_rejection_mismatches += int(
                int(row[rejected_key]) != int(rejected)
            )
            decisions.append(rejected)
        tp_mask = np.asarray(labels, dtype=np.int64) == 0
        fp_mask = ~tp_mask
        decision_array = np.asarray(decisions, dtype=np.bool_)
        score_array = np.asarray(scores, dtype=np.float64)
        condition_metrics[condition] = {
            "rows": len(selected),
            "auroc_fp_vs_tp": binary_auroc(labels, scores),
            "tp_retention": float(np.mean(~decision_array[tp_mask])),
            "restricted_fp_rejection": float(
                np.mean(decision_array[fp_mask])
            ),
            "tp_breaks": int(decision_array[tp_mask].sum()),
            "restricted_fp_removals": int(decision_array[fp_mask].sum()),
            "tp_suppression_median": float(np.median(score_array[tp_mask])),
            "fp_suppression_median": float(np.median(score_array[fp_mask])),
        }

    fold_metrics: Dict[str, object] = {}
    for fold in FIT_FOLDS:
        selected = [row for row in clean if int(row["fold"]) == fold]
        tp = [row for row in selected if str(row["cohort"]) == "tp"]
        fp = [row for row in selected if str(row["cohort"]) == "fp"]
        threshold = thresholds[fold]
        tp_breaks = sum(float(row[score_key]) > threshold for row in tp)
        fp_removals = sum(float(row[score_key]) > threshold for row in fp)
        fold_metrics[str(fold)] = {
            "threshold": threshold,
            "fit_tp_rows": sum(
                str(row["cohort"]) == "tp" and int(row["fold"]) != fold
                for row in clean
            ),
            "holdout_tp_rows": len(tp),
            "holdout_fp_rows": len(fp),
            "tp_breaks": int(tp_breaks),
            "restricted_fp_removals": int(fp_removals),
            "tp_retention": float(1.0 - tp_breaks / len(tp)),
            "restricted_fp_rejection": float(fp_removals / len(fp)),
        }
    return {
        "threshold_quantile": TP_QUANTILE,
        "threshold_method": "higher",
        "fold_thresholds": {str(key): value for key, value in thresholds.items()},
        "conditions": condition_metrics,
        "clean_folds": fold_metrics,
        "positive_clean_fp_rejection_fold_count": sum(
            float(value["restricted_fp_rejection"]) > 0.0
            for value in fold_metrics.values()
        ),
        "hard_decisions_clean": _hard_decisions(clean, view),
        "csv_reconstruction": {
            "maximum_suppression_score_error": stored_score_error,
            "maximum_threshold_error": stored_threshold_error,
            "rejection_mismatches": stored_rejection_mismatches,
        },
    }


def assess_information_gate(view_metrics: Mapping[str, object]) -> Dict[str, object]:
    early = view_metrics["early_b176"]
    middle = view_metrics["middle_b224"]
    conditions = early["conditions"]
    clean = conditions["clean"]
    hard = early["hard_decisions_clean"]
    checks = {
        "clean_auroc_gte_0p65": float(clean["auroc_fp_vs_tp"]) >= 0.65,
        "every_shifted_auroc_gte_0p60": all(
            float(conditions[name]["auroc_fp_vs_tp"]) >= 0.60
            for name in SHIFTED_CONDITIONS
        ),
        "four_condition_mean_auroc_gte_0p65": float(
            np.mean(
                [
                    float(conditions[name]["auroc_fp_vs_tp"])
                    for name in CONDITIONS
                ]
            )
        )
        >= 0.65,
        "clean_tp_retention_gte_0p97": float(clean["tp_retention"]) >= 0.97,
        "clean_fp_rejection_gte_0p10": float(
            clean["restricted_fp_rejection"]
        )
        >= 0.10,
        "every_shifted_tp_retention_gte_0p93": all(
            float(conditions[name]["tp_retention"]) >= 0.93
            for name in SHIFTED_CONDITIONS
        ),
        "every_shifted_fp_rejection_gte_0p05": all(
            float(conditions[name]["restricted_fp_rejection"]) >= 0.05
            for name in SHIFTED_CONDITIONS
        ),
        "positive_clean_fp_rejection_in_gte_3_folds": int(
            early["positive_clean_fp_rejection_fold_count"]
        )
        >= 3,
        "clean_fp_removals_exceed_tp_breaks_by_gte_5": int(
            clean["restricted_fp_removals"]
        )
        - int(clean["tp_breaks"])
        >= 5,
        "early_clean_auroc_not_below_middle": float(clean["auroc_fp_vs_tp"])
        >= float(middle["conditions"]["clean"]["auroc_fp_vs_tp"]),
        "early_clean_fp_rejection_not_below_middle": float(
            clean["restricted_fp_rejection"]
        )
        >= float(
            middle["conditions"]["clean"]["restricted_fp_rejection"]
        ),
        "hard_corrections_exceed_harms": int(hard["corrections"])
        > int(hard["harms"]),
        "hard_class1_tp_breaks_lte_2": int(hard["class1_tp_breaks"]) <= 2,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }


def _mechanism_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> Dict[str, object]:
    expected = int(expected_rows) * len(CONDITIONS) * len(VIEWS)
    if len(rows) != expected:
        raise ValueError(f"Mechanism CSV has {len(rows)} rows, expected {expected}.")
    views: Dict[str, object] = {}
    for view in VIEWS:
        condition_values: Dict[str, object] = {}
        for condition in CONDITIONS:
            selected = [
                row
                for row in rows
                if str(row["view"]) == view
                and str(row["condition"]) == condition
            ]
            selected.sort(key=lambda row: int(row["cohort_position"]))
            if len(selected) != int(expected_rows):
                raise ValueError(
                    f"Mechanism {view}/{condition} has {len(selected)} rows."
                )
            positions = [int(row["cohort_position"]) for row in selected]
            if positions != _expected_cohort_positions(expected_rows):
                raise ValueError(
                    f"Mechanism {view}/{condition} has noncanonical cohort positions."
                )
            labels = [int(str(row["cohort"]) == "fp") for row in selected]
            object_scores = [float(row["object_residual_rms"]) for row in selected]
            outside_scores = [float(row["outside_residual_rms"]) for row in selected]
            label_array = np.asarray(labels, dtype=np.int64)
            object_array = np.asarray(object_scores, dtype=np.float64)
            outside_array = np.asarray(outside_scores, dtype=np.float64)
            object_tp = float(np.median(object_array[label_array == 0]))
            object_fp = float(np.median(object_array[label_array == 1]))
            outside_tp = float(np.median(outside_array[label_array == 0]))
            outside_fp = float(np.median(outside_array[label_array == 1]))
            condition_values[condition] = {
                "rows": len(selected),
                "all_regions_nonempty": all(
                    int(row["object_pixels"]) > 0
                    and int(row["outside_pixels"]) > 0
                    for row in selected
                ),
                "object_auroc_fp_vs_tp": binary_auroc(labels, object_scores),
                "outside_auroc_fp_vs_tp": binary_auroc(labels, outside_scores),
                "object_tp_median": object_tp,
                "object_fp_median": object_fp,
                "outside_tp_median": outside_tp,
                "outside_fp_median": outside_fp,
                "object_fp_minus_tp_gap": object_fp - object_tp,
                "outside_fp_minus_tp_gap": outside_fp - outside_tp,
            }
        views[view] = {"conditions": condition_values}
    return {"views": views}


def assess_mechanism_gate(summary: Mapping[str, object]) -> Dict[str, object]:
    conditions = summary["views"]["early_b176"]["conditions"]
    positive_gap_count = sum(
        float(value["object_fp_minus_tp_gap"]) > 0.0
        and float(value["object_fp_minus_tp_gap"])
        > float(value["outside_fp_minus_tp_gap"])
        for value in conditions.values()
    )
    checks = {
        "every_row_has_object_and_outside": all(
            bool(value["all_regions_nonempty"]) for value in conditions.values()
        ),
        "every_condition_object_auroc_gte_0p60": all(
            float(value["object_auroc_fp_vs_tp"]) >= 0.60
            for value in conditions.values()
        ),
        "every_condition_object_auroc_exceeds_outside_by_0p03": all(
            float(value["object_auroc_fp_vs_tp"])
            - float(value["outside_auroc_fp_vs_tp"])
            >= 0.03
            for value in conditions.values()
        ),
        "object_gap_positive_and_exceeds_outside_in_gte_3_conditions": positive_gap_count
        >= 3,
    }
    return {
        "positive_object_gap_condition_count": int(positive_gap_count),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }


def replay_artifacts(
    predictions_path: Path, mechanism_path: Path
) -> Dict[str, object]:
    prediction_rows = _read_rows(predictions_path)
    expected_predictions = EXPECTED_ROWS * len(CONDITIONS)
    if len(prediction_rows) != expected_predictions:
        raise ValueError(
            f"Prediction CSV has {len(prediction_rows)} rows, "
            f"expected {expected_predictions}."
        )
    identities = None
    for condition in CONDITIONS:
        selected = _condition_rows(prediction_rows, condition)
        current = [
            (
                int(row["sample_index"]),
                int(row["target"]),
                str(row["cohort"]),
                int(row["fold"]),
            )
            for row in selected
        ]
        if identities is None:
            identities = current
        elif current != identities:
            raise ValueError(f"Condition {condition} changed cohort identity.")
    if identities is None:
        raise RuntimeError("Prediction identities are missing.")
    tp_count = sum(row[2] == "tp" for row in identities)
    fp_count = sum(row[2] == "fp" for row in identities)
    if (tp_count, fp_count) != (EXPECTED_TP, EXPECTED_FP):
        raise ValueError(f"Cohort count differs: TP={tp_count}, FP={fp_count}.")

    view_metrics = {
        view: _view_metrics(prediction_rows, view) for view in VIEWS
    }
    information_gate = assess_information_gate(view_metrics)
    mechanism_summary = _mechanism_summary(_read_rows(mechanism_path))
    mechanism_gate = assess_mechanism_gate(mechanism_summary)
    filtered_predictions = [
        row
        for row in prediction_rows
        if int(row["sample_index"]) != KNOWN_NEAR_TIE_INDEX
    ]
    sensitivity_views: Dict[str, object] = {}
    for view in VIEWS:
        thresholds = {
            int(key): float(value)
            for key, value in view_metrics[view]["fold_thresholds"].items()
        }
        sensitivity_views[view] = _view_metrics(
            filtered_predictions,
            view,
            fixed_thresholds=thresholds,
            expected_rows=SENSITIVITY_ROWS,
        )
    prediction_sensitivity = {
        "excluded_sample_index": KNOWN_NEAR_TIE_INDEX,
        "rows_per_condition": SENSITIVITY_ROWS,
        "thresholds_refit": False,
        "view_metrics": sensitivity_views,
        "information_gate": assess_information_gate(sensitivity_views),
    }
    mechanism_rows = _read_rows(mechanism_path)
    filtered_mechanism = [
        row
        for row in mechanism_rows
        if int(row["sample_index"]) != KNOWN_NEAR_TIE_INDEX
    ]
    sensitivity_mechanism_summary = _mechanism_summary(
        filtered_mechanism, expected_rows=SENSITIVITY_ROWS
    )
    mechanism_sensitivity = {
        "excluded_sample_index": KNOWN_NEAR_TIE_INDEX,
        "rows_per_condition_view": SENSITIVITY_ROWS,
        "mechanism_summary": sensitivity_mechanism_summary,
        "mechanism_gate": assess_mechanism_gate(sensitivity_mechanism_summary),
    }
    return {
        "prediction_rows": len(prediction_rows),
        "mechanism_rows": EXPECTED_ROWS * len(CONDITIONS) * len(VIEWS),
        "cohort_tp": tp_count,
        "cohort_fp": fp_count,
        "view_metrics": view_metrics,
        "information_gate": information_gate,
        "mechanism_summary": mechanism_summary,
        "mechanism_gate": mechanism_gate,
        "known_near_tie_exclusion_sensitivity": {
            "prediction": prediction_sensitivity,
            "mechanism": mechanism_sensitivity,
        },
    }


def canonical_close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            canonical_close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            canonical_close(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=float(tolerance)
        )
    return left == right


def verify_summary(
    replay: Mapping[str, object], summary: Mapping[str, object]
) -> Dict[str, bool]:
    return {
        "view_metrics_exact_within_1e12": canonical_close(
            replay["view_metrics"], summary["view_metrics"]
        ),
        "information_gate_exact": canonical_close(
            replay["information_gate"], summary["information_gate"]
        ),
        "mechanism_summary_exact_within_1e12": canonical_close(
            replay["mechanism_summary"], summary["mechanism_summary"]
        ),
        "mechanism_gate_exact": canonical_close(
            replay["mechanism_gate"], summary["mechanism_gate"]
        ),
        "known_near_tie_exclusion_exact_within_1e12": canonical_close(
            replay["known_near_tie_exclusion_sensitivity"],
            summary["known_near_tie_exclusion_sensitivity"],
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent CSV replay for locked EfficientTrain A0."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    replay = replay_artifacts(args.predictions, args.mechanism)
    result: Dict[str, object] = {"replay": replay}
    if args.summary is not None:
        summary_path = Path(args.summary).resolve()
        observed_sha = _sha256(summary_path)
        expected_sha = str(args.expected_summary_sha256).strip().casefold()
        if expected_sha and observed_sha != expected_sha:
            raise ValueError(
                f"Summary SHA-256 mismatch: {observed_sha} != {expected_sha}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks = verify_summary(replay, summary)
        result["summary_sha256"] = observed_sha
        result["summary_checks"] = checks
        result["summary_exact"] = all(checks.values())
        if not bool(result["summary_exact"]):
            raise RuntimeError("Independent replay differs from formal summary.")
    if args.output is not None:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f"Replay output already exists: {output}")
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from trkh.tools.soft_ensemble_predictions import (
    PredictionTable,
    _metrics,
    _probability_columns,
    _read_table,
    _source_path,
    _target_index,
    _target_name,
)


def _reject_test_input(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if re.search(r"(^|[/_.-])test([/_.-]|$)", normalized):
        raise ValueError(f"Readiness fitting is validation-only; refusing test input: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_grid(start: float, stop: float, step: float) -> np.ndarray:
    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step):
        raise ValueError("Grid bounds must be finite.")
    if step <= 0.0 or stop < start:
        raise ValueError(f"Invalid grid: start={start}, stop={stop}, step={step}")
    count = int(round((stop - start) / step))
    values = np.asarray([start + index * step for index in range(count + 1)], dtype=np.float64)
    if values.size == 0 or values[-1] > stop + 1e-9:
        raise ValueError("Failed to construct the requested grid.")
    return values


def _sort_key(value: str) -> Tuple[int, object]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _source_group(row: Mapping[str, str]) -> str:
    value = str(row.get("source_stem", "") or "").strip()
    if value:
        return value
    path = _source_path(row)
    if path:
        return Path(path).stem
    raise ValueError("Prediction row has neither source_stem nor image path.")


def _validate_prediction_row_split(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_split: str,
) -> None:
    expected = "val" if str(expected_split).strip().lower() in {"val", "validation"} else str(
        expected_split
    ).strip().lower()
    if expected not in {"val", "test"}:
        raise ValueError(f"Unsupported expected split: {expected_split!r}")
    mismatches: List[Tuple[int, str, str]] = []
    for index, row in enumerate(rows):
        source = _source_path(row)
        parts = [part.lower() for part in re.split(r"[/\\]+", source) if part]
        declared = "test" if "test" in parts else "val" if any(
            part in {"val", "validation"} for part in parts
        ) else ""
        if declared != expected:
            mismatches.append((index, declared or "unknown", source))
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError(
            f"Prediction rows do not belong to split={expected}: examples={mismatches}"
        )


def _validated_row_probabilities(
    row: Mapping[str, str],
    columns: Sequence[str],
) -> List[float]:
    values: List[float] = []
    for column in columns:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid probability in column {column!r}.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite probability in column {column!r}.")
        if value < 0.0:
            raise ValueError(f"Negative probability in column {column!r}: {value}")
        values.append(value)
    total = float(sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Prediction probabilities must have a finite positive sum.")
    return [value / total for value in values]


def _aligned_arrays(
    keeper: PredictionTable,
    candidate: PredictionTable,
) -> Tuple[
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Mapping[str, str]],
]:
    if keeper.class_names != candidate.class_names:
        raise ValueError(
            f"Class order mismatch: keeper={keeper.class_names}, candidate={candidate.class_names}"
        )
    keeper_keys = set(keeper.rows_by_key)
    candidate_keys = set(candidate.rows_by_key)
    if keeper_keys != candidate_keys:
        raise ValueError(
            "Prediction key mismatch: "
            f"keeper_only={len(keeper_keys - candidate_keys)}, "
            f"candidate_only={len(candidate_keys - keeper_keys)}"
        )
    keys = sorted(keeper_keys, key=_sort_key)
    keeper_columns = _probability_columns(
        next(iter(keeper.rows_by_key.values())).keys()
    )[0]
    candidate_columns = _probability_columns(
        next(iter(candidate.rows_by_key.values())).keys()
    )[0]
    targets: List[int] = []
    groups: List[str] = []
    keeper_probabilities: List[List[float]] = []
    candidate_probabilities: List[List[float]] = []
    base_rows: List[Mapping[str, str]] = []
    for key in keys:
        keeper_row = keeper.rows_by_key[key]
        candidate_row = candidate.rows_by_key[key]
        keeper_target = _target_index(keeper_row)
        candidate_target = _target_index(candidate_row)
        if keeper_target != candidate_target:
            raise ValueError(
                f"Target mismatch for {key}: keeper={keeper_target}, candidate={candidate_target}"
            )
        keeper_group = _source_group(keeper_row)
        candidate_group = _source_group(candidate_row)
        if keeper_group != candidate_group:
            raise ValueError(
                f"Source-group mismatch for {key}: keeper={keeper_group!r}, "
                f"candidate={candidate_group!r}"
            )
        targets.append(keeper_target)
        groups.append(keeper_group)
        keeper_probabilities.append(
            _validated_row_probabilities(keeper_row, keeper_columns)
        )
        candidate_probabilities.append(
            _validated_row_probabilities(candidate_row, candidate_columns)
        )
        base_rows.append(keeper_row)
    arrays = (
        np.asarray(targets, dtype=np.int64),
        np.asarray(groups, dtype=object),
        np.asarray(keeper_probabilities, dtype=np.float64),
        np.asarray(candidate_probabilities, dtype=np.float64),
    )
    for array in arrays[2:]:
        if not bool(np.isfinite(array).all()):
            raise ValueError("Prediction probabilities contain non-finite values.")
    return keys, *arrays, base_rows


def apply_precision_ensemble(
    keeper_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    candidate_weight: float,
    focus_margin_offset: float,
    focus_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keeper_probabilities = np.asarray(keeper_probabilities, dtype=np.float64)
    candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    if keeper_probabilities.shape != candidate_probabilities.shape:
        raise ValueError(
            "Keeper and candidate probability arrays must have the same shape: "
            f"{keeper_probabilities.shape} vs {candidate_probabilities.shape}"
        )
    if keeper_probabilities.ndim != 2 or keeper_probabilities.shape[1] < 2:
        raise ValueError("Probability arrays must have shape [samples, classes].")
    for name, probabilities in (
        ("keeper", keeper_probabilities),
        ("candidate", candidate_probabilities),
    ):
        if not bool(np.isfinite(probabilities).all()):
            raise ValueError(f"{name} probabilities contain non-finite values.")
        if bool(np.any(probabilities < 0.0)):
            raise ValueError(f"{name} probabilities contain negative values.")
        if bool(np.any(probabilities.sum(axis=1) <= 0.0)):
            raise ValueError(f"{name} probability rows must have a positive sum.")
    weight = float(candidate_weight)
    offset = float(focus_margin_offset)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"candidate_weight must be in [0, 1], got {weight}")
    if offset < 0.0 or not math.isfinite(offset):
        raise ValueError(f"focus_margin_offset must be finite and >= 0, got {offset}")
    if not 0 <= int(focus_class) < keeper_probabilities.shape[1]:
        raise ValueError(f"Invalid focus_class={focus_class}")
    blended = (1.0 - weight) * keeper_probabilities + weight * candidate_probabilities
    decision_scores = blended.copy()
    decision_scores[:, int(focus_class)] -= offset
    predictions = decision_scores.argmax(axis=1).astype(np.int64)
    other_scores = np.delete(blended, int(focus_class), axis=1)
    focus_margin = (
        blended[:, int(focus_class)] - other_scores.max(axis=1) - offset
    )
    return blended, predictions, focus_margin


def _metric_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    focus_class: int,
) -> Dict[str, object]:
    metrics = _metrics(targets.tolist(), predictions.tolist(), class_names)
    focus = dict(metrics["per_class"][int(focus_class)])
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "focus": focus,
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _selection_key(
    metrics: Mapping[str, object],
    *,
    candidate_weight: float,
    focus_margin_offset: float,
    precision_weight: float,
) -> Tuple[float, float, float, float, float, float]:
    focus = metrics["focus"]
    assert isinstance(focus, Mapping)
    macro_f1 = float(metrics["macro_f1"])
    focus_f1 = float(focus["f1"])
    focus_precision = float(focus["precision"])
    return (
        macro_f1 + focus_f1 + float(precision_weight) * focus_precision,
        macro_f1,
        focus_f1,
        focus_precision,
        -float(candidate_weight),
        -float(focus_margin_offset),
    )


def _choose_fold_parameters(
    targets: np.ndarray,
    keeper_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    fit_indices: np.ndarray,
    *,
    class_names: Sequence[str],
    focus_class: int,
    candidate_weights: Sequence[float],
    focus_margin_offsets: Sequence[float],
    minimum_focus_precision: float,
    macro_tolerance: float,
    precision_weight: float,
) -> Tuple[float, float, Dict[str, object], Dict[str, object], int]:
    fit_targets = targets[fit_indices]
    keeper_predictions = keeper_probabilities[fit_indices].argmax(axis=1)
    keeper_metrics = _metric_summary(
        fit_targets,
        keeper_predictions,
        class_names,
        focus_class,
    )
    keeper_focus = keeper_metrics["focus"]
    assert isinstance(keeper_focus, Mapping)
    feasible: List[Tuple[Tuple[float, ...], float, float, Dict[str, object]]] = []
    for candidate_weight in candidate_weights:
        blended = (
            (1.0 - float(candidate_weight)) * keeper_probabilities[fit_indices]
            + float(candidate_weight) * candidate_probabilities[fit_indices]
        )
        for focus_margin_offset in focus_margin_offsets:
            scores = blended.copy()
            scores[:, int(focus_class)] -= float(focus_margin_offset)
            predictions = scores.argmax(axis=1)
            metrics = _metric_summary(
                fit_targets,
                predictions,
                class_names,
                focus_class,
            )
            focus = metrics["focus"]
            assert isinstance(focus, Mapping)
            if float(focus["precision"]) < float(minimum_focus_precision):
                continue
            if float(metrics["macro_f1"]) < float(keeper_metrics["macro_f1"]) - float(
                macro_tolerance
            ):
                continue
            if float(focus["f1"]) < float(keeper_focus["f1"]):
                continue
            feasible.append(
                (
                    _selection_key(
                        metrics,
                        candidate_weight=float(candidate_weight),
                        focus_margin_offset=float(focus_margin_offset),
                        precision_weight=float(precision_weight),
                    ),
                    float(candidate_weight),
                    float(focus_margin_offset),
                    metrics,
                )
            )
    if not feasible:
        raise RuntimeError("No precision-constrained ensemble setting passed the fit-fold gates.")
    _key, weight, offset, metrics = max(feasible, key=lambda item: item[0])
    return weight, offset, metrics, keeper_metrics, len(feasible)


def _transition_audit(
    targets: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    focus_class: int,
) -> Tuple[Dict[str, int], np.ndarray]:
    changed = reference_predictions != candidate_predictions
    reference_correct = reference_predictions == targets
    candidate_correct = candidate_predictions == targets
    categories = np.full(len(targets), "unchanged", dtype=object)
    categories[changed & ~reference_correct & candidate_correct] = "correction"
    categories[changed & reference_correct & ~candidate_correct] = "harm"
    categories[changed & ~reference_correct & ~candidate_correct] = "neutral_changed"
    summary = {
        "changed": int(changed.sum()),
        "corrections": int(np.sum(categories == "correction")),
        "harms": int(np.sum(categories == "harm")),
        "neutral_changed": int(np.sum(categories == "neutral_changed")),
        "focus_fn_rescued": int(
            np.sum(
                (targets == int(focus_class))
                & (reference_predictions != int(focus_class))
                & (candidate_predictions == int(focus_class))
            )
        ),
        "focus_tp_broken": int(
            np.sum(
                (targets == int(focus_class))
                & (reference_predictions == int(focus_class))
                & (candidate_predictions != int(focus_class))
            )
        ),
        "focus_fp_removed": int(
            np.sum(
                (targets != int(focus_class))
                & (reference_predictions == int(focus_class))
                & (candidate_predictions != int(focus_class))
            )
        ),
        "focus_fp_created": int(
            np.sum(
                (targets != int(focus_class))
                & (reference_predictions != int(focus_class))
                & (candidate_predictions == int(focus_class))
            )
        ),
    }
    return summary, categories


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _prediction_rows(
    *,
    keys: Sequence[str],
    base_rows: Sequence[Mapping[str, str]],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    focus_margins: np.ndarray,
    class_names: Sequence[str],
    folds: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, key in enumerate(keys):
        base = base_rows[index]
        prediction = int(predictions[index])
        target = int(targets[index])
        order = np.argsort(-probabilities[index], kind="stable")
        record: Dict[str, object] = {
            "sample_index": key,
            "image_path": _source_path(base),
            "source_stem": _source_group(base),
            "target_index": target,
            "target_name": _target_name(base, class_names),
            "prediction_index": prediction,
            "prediction_name": class_names[prediction],
            "confidence": float(probabilities[index, prediction]),
            "correct": int(prediction == target),
            "focus_decision_margin": float(focus_margins[index]),
            "top1_index_raw_blend": int(order[0]),
            "top2_index_raw_blend": int(order[1]),
            "top1_probability_raw_blend": float(probabilities[index, order[0]]),
            "top2_probability_raw_blend": float(probabilities[index, order[1]]),
        }
        if folds is not None:
            record["oof_fold"] = int(folds[index])
        for class_index, class_name in enumerate(class_names):
            record[f"prob_{class_index}_{class_name}"] = float(
                probabilities[index, class_index]
            )
        rows.append(record)
    return rows


def _priority_xai_rows(
    transition_rows: Sequence[Mapping[str, object]],
    *,
    limit_per_category: int = 6,
) -> List[Dict[str, object]]:
    priorities = (
        "focus_fp_removed",
        "harm",
        "retained_focus_fp",
        "focus_fn",
        "correction",
    )
    selected: List[Dict[str, object]] = []
    used = set()
    for priority in priorities:
        rows = [row for row in transition_rows if row.get("xai_category") == priority]
        rows.sort(
            key=lambda row: (
                -abs(float(row.get("focus_decision_margin", 0.0))),
                int(row.get("sample_index", 0)),
            )
        )
        for row in rows[: max(0, int(limit_per_category))]:
            key = str(row["sample_index"])
            if key in used:
                continue
            used.add(key)
            selected.append(dict(row))
    return selected


def run_precision_ensemble_readiness(
    *,
    keeper_csv: Path,
    candidate_csv: Path,
    output_dir: Path,
    key_column: str = "sample_index",
    expected_rows: int = 2606,
    focus_class: int = 1,
    folds: int = 5,
    seed: int = 20260714,
    minimum_focus_precision: float = 0.65,
    macro_tolerance: float = 0.002,
    precision_weight: float = 0.10,
    candidate_weight_start: float = 0.0,
    candidate_weight_stop: float = 1.0,
    candidate_weight_step: float = 0.05,
    focus_margin_start: float = 0.0,
    focus_margin_stop: float = 0.10,
    focus_margin_step: float = 0.002,
) -> Dict[str, object]:
    keeper_csv = Path(keeper_csv)
    candidate_csv = Path(candidate_csv)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Readiness output already exists and is nonempty: {output_dir}"
            )
    _reject_test_input(keeper_csv)
    _reject_test_input(candidate_csv)
    keeper = _read_table("keeper", keeper_csv, key_column)
    candidate = _read_table("candidate", candidate_csv, key_column)
    keys, targets, groups, keeper_probs, candidate_probs, base_rows = _aligned_arrays(
        keeper,
        candidate,
    )
    _validate_prediction_row_split(base_rows, expected_split="val")
    if int(expected_rows) > 0 and len(keys) != int(expected_rows):
        raise ValueError(f"Expected {expected_rows} rows, found {len(keys)}")
    class_names = list(keeper.class_names)
    if not 0 <= int(focus_class) < len(class_names):
        raise ValueError(f"Invalid focus_class={focus_class} for {len(class_names)} classes")
    candidate_weights = _float_grid(
        candidate_weight_start,
        candidate_weight_stop,
        candidate_weight_step,
    )
    focus_margin_offsets = _float_grid(
        focus_margin_start,
        focus_margin_stop,
        focus_margin_step,
    )

    keeper_predictions = keeper_probs.argmax(axis=1)
    candidate_predictions = candidate_probs.argmax(axis=1)
    keeper_metrics = _metric_summary(
        targets,
        keeper_predictions,
        class_names,
        focus_class,
    )
    candidate_metrics = _metric_summary(
        targets,
        candidate_predictions,
        class_names,
        focus_class,
    )
    candidate_transitions, _ = _transition_audit(
        targets,
        keeper_predictions,
        candidate_predictions,
        focus_class=focus_class,
    )

    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    oof_predictions = np.full(len(keys), -1, dtype=np.int64)
    oof_probabilities = np.zeros_like(keeper_probs)
    oof_focus_margins = np.zeros(len(keys), dtype=np.float64)
    oof_fold = np.zeros(len(keys), dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    selected_weights: List[float] = []
    selected_offsets: List[float] = []
    source_overlap = 0
    for fold_index, (fit_indices, heldout_indices) in enumerate(
        splitter.split(np.zeros(len(targets)), targets, groups),
        start=1,
    ):
        fit_groups = set(groups[fit_indices].tolist())
        heldout_groups = set(groups[heldout_indices].tolist())
        overlap = len(fit_groups & heldout_groups)
        source_overlap += overlap
        if overlap:
            raise RuntimeError(f"Source-group leakage in fold {fold_index}: {overlap}")
        weight, offset, fit_metrics, fit_keeper_metrics, feasible_count = (
            _choose_fold_parameters(
                targets,
                keeper_probs,
                candidate_probs,
                fit_indices,
                class_names=class_names,
                focus_class=focus_class,
                candidate_weights=candidate_weights,
                focus_margin_offsets=focus_margin_offsets,
                minimum_focus_precision=minimum_focus_precision,
                macro_tolerance=macro_tolerance,
                precision_weight=precision_weight,
            )
        )
        heldout_probabilities, heldout_predictions, heldout_margins = (
            apply_precision_ensemble(
                keeper_probs[heldout_indices],
                candidate_probs[heldout_indices],
                candidate_weight=weight,
                focus_margin_offset=offset,
                focus_class=focus_class,
            )
        )
        oof_predictions[heldout_indices] = heldout_predictions
        oof_probabilities[heldout_indices] = heldout_probabilities
        oof_focus_margins[heldout_indices] = heldout_margins
        oof_fold[heldout_indices] = int(fold_index)
        heldout_metrics = _metric_summary(
            targets[heldout_indices],
            heldout_predictions,
            class_names,
            focus_class,
        )
        heldout_keeper_metrics = _metric_summary(
            targets[heldout_indices],
            keeper_predictions[heldout_indices],
            class_names,
            focus_class,
        )
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "heldout_rows": int(len(heldout_indices)),
                "fit_source_groups": int(len(fit_groups)),
                "heldout_source_groups": int(len(heldout_groups)),
                "source_overlap": int(overlap),
                "candidate_weight": float(weight),
                "focus_margin_offset": float(offset),
                "feasible_settings": int(feasible_count),
                "fit_keeper_macro_f1": float(fit_keeper_metrics["macro_f1"]),
                "fit_keeper_focus_precision": float(fit_keeper_metrics["focus"]["precision"]),
                "fit_keeper_focus_f1": float(fit_keeper_metrics["focus"]["f1"]),
                "fit_selected_macro_f1": float(fit_metrics["macro_f1"]),
                "fit_selected_focus_precision": float(fit_metrics["focus"]["precision"]),
                "fit_selected_focus_recall": float(fit_metrics["focus"]["recall"]),
                "fit_selected_focus_f1": float(fit_metrics["focus"]["f1"]),
                "heldout_keeper_macro_f1": float(heldout_keeper_metrics["macro_f1"]),
                "heldout_keeper_focus_precision": float(
                    heldout_keeper_metrics["focus"]["precision"]
                ),
                "heldout_keeper_focus_recall": float(
                    heldout_keeper_metrics["focus"]["recall"]
                ),
                "heldout_keeper_focus_f1": float(heldout_keeper_metrics["focus"]["f1"]),
                "heldout_selected_macro_f1": float(heldout_metrics["macro_f1"]),
                "heldout_selected_focus_precision": float(
                    heldout_metrics["focus"]["precision"]
                ),
                "heldout_selected_focus_recall": float(heldout_metrics["focus"]["recall"]),
                "heldout_selected_focus_f1": float(heldout_metrics["focus"]["f1"]),
            }
        )
        selected_weights.append(float(weight))
        selected_offsets.append(float(offset))
    if bool(np.any(oof_predictions < 0)) or bool(np.any(oof_fold <= 0)):
        raise RuntimeError("OOF predictions are incomplete.")

    median_weight = float(np.median(np.asarray(selected_weights, dtype=np.float64)))
    median_offset = float(np.median(np.asarray(selected_offsets, dtype=np.float64)))
    locked_weight = float(candidate_weights[np.argmin(np.abs(candidate_weights - median_weight))])
    locked_offset = float(
        focus_margin_offsets[np.argmin(np.abs(focus_margin_offsets - median_offset))]
    )
    locked_probabilities, locked_predictions, locked_focus_margins = (
        apply_precision_ensemble(
            keeper_probs,
            candidate_probs,
            candidate_weight=locked_weight,
            focus_margin_offset=locked_offset,
            focus_class=focus_class,
        )
    )
    oof_metrics = _metric_summary(
        targets,
        oof_predictions,
        class_names,
        focus_class,
    )
    locked_metrics = _metric_summary(
        targets,
        locked_predictions,
        class_names,
        focus_class,
    )
    oof_transitions, oof_categories = _transition_audit(
        targets,
        keeper_predictions,
        oof_predictions,
        focus_class=focus_class,
    )
    locked_transitions, locked_categories = _transition_audit(
        targets,
        keeper_predictions,
        locked_predictions,
        focus_class=focus_class,
    )
    locked_vs_candidate, _ = _transition_audit(
        targets,
        candidate_predictions,
        locked_predictions,
        focus_class=focus_class,
    )

    keeper_focus = keeper_metrics["focus"]
    oof_focus = oof_metrics["focus"]
    locked_focus = locked_metrics["focus"]
    assert isinstance(keeper_focus, Mapping)
    assert isinstance(oof_focus, Mapping)
    assert isinstance(locked_focus, Mapping)
    heldout_precision_values = [
        float(row["heldout_selected_focus_precision"]) for row in fold_rows
    ]
    stability = {
        "candidate_weight_min": float(min(selected_weights)),
        "candidate_weight_max": float(max(selected_weights)),
        "candidate_weight_range": float(max(selected_weights) - min(selected_weights)),
        "focus_margin_offset_min": float(min(selected_offsets)),
        "focus_margin_offset_max": float(max(selected_offsets)),
        "focus_margin_offset_range": float(max(selected_offsets) - min(selected_offsets)),
        "minimum_heldout_focus_precision": float(min(heldout_precision_values)),
    }
    gates = {
        "validation_rows_exact": len(keys) == int(expected_rows) if int(expected_rows) > 0 else True,
        "source_group_overlap_zero": source_overlap == 0,
        "fold_parameter_stability": (
            float(stability["candidate_weight_range"]) <= 0.10 + 1e-12
            and float(stability["focus_margin_offset_range"]) <= 0.02 + 1e-12
        ),
        "oof_macro_not_below_keeper": float(oof_metrics["macro_f1"])
        >= float(keeper_metrics["macro_f1"]),
        "oof_focus_f1_not_below_keeper": float(oof_focus["f1"])
        >= float(keeper_focus["f1"]),
        "oof_focus_precision_floor": float(oof_focus["precision"])
        >= float(minimum_focus_precision),
        "oof_focus_precision_gain_003": float(oof_focus["precision"])
        >= float(keeper_focus["precision"]) + 0.03,
        "minimum_heldout_focus_precision_060": min(heldout_precision_values) >= 0.60,
        "locked_macro_not_below_keeper": float(locked_metrics["macro_f1"])
        >= float(keeper_metrics["macro_f1"]),
        "locked_focus_f1_not_below_keeper": float(locked_focus["f1"])
        >= float(keeper_focus["f1"]),
        "locked_focus_precision_floor": float(locked_focus["precision"])
        >= float(minimum_focus_precision),
        "locked_focus_recall_floor_070": float(locked_focus["recall"]) >= 0.70,
        "locked_corrections_not_below_harms": int(locked_transitions["corrections"])
        >= int(locked_transitions["harms"]),
        "locked_focus_fp_net_reduction": int(locked_transitions["focus_fp_removed"])
        > int(locked_transitions["focus_fp_created"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fold_selections.csv", fold_rows)
    oof_rows = _prediction_rows(
        keys=keys,
        base_rows=base_rows,
        targets=targets,
        predictions=oof_predictions,
        probabilities=oof_probabilities,
        focus_margins=oof_focus_margins,
        class_names=class_names,
        folds=oof_fold,
    )
    locked_rows = _prediction_rows(
        keys=keys,
        base_rows=base_rows,
        targets=targets,
        predictions=locked_predictions,
        probabilities=locked_probabilities,
        focus_margins=locked_focus_margins,
        class_names=class_names,
    )
    _write_csv(output_dir / "oof_predictions_detailed.csv", oof_rows)
    _write_csv(output_dir / "locked_val_predictions_detailed.csv", locked_rows)

    transition_rows: List[Dict[str, object]] = []
    for index, key in enumerate(keys):
        target = int(targets[index])
        keeper_prediction = int(keeper_predictions[index])
        candidate_prediction = int(candidate_predictions[index])
        locked_prediction = int(locked_predictions[index])
        locked_category = str(locked_categories[index])
        if target != int(focus_class) and locked_prediction == int(focus_class):
            xai_category = "retained_focus_fp"
        elif target == int(focus_class) and locked_prediction != int(focus_class):
            xai_category = "focus_fn"
        elif (
            target != int(focus_class)
            and keeper_prediction == int(focus_class)
            and locked_prediction != int(focus_class)
        ):
            xai_category = "focus_fp_removed"
        elif locked_category == "harm":
            xai_category = "harm"
        elif locked_category == "correction":
            xai_category = "correction"
        else:
            xai_category = ""
        if locked_category == "unchanged" and not xai_category:
            continue
        row = {
            "sample_index": key,
            "image_path": _source_path(base_rows[index]),
            "source_stem": _source_group(base_rows[index]),
            "target_index": target,
            "target_name": class_names[target],
            "keeper_prediction_index": keeper_prediction,
            "candidate_prediction_index": candidate_prediction,
            "locked_prediction_index": locked_prediction,
            "locked_transition": locked_category,
            "oof_transition": str(oof_categories[index]),
            "xai_category": xai_category,
            "keeper_focus_probability": float(keeper_probs[index, int(focus_class)]),
            "candidate_focus_probability": float(candidate_probs[index, int(focus_class)]),
            "locked_focus_probability": float(locked_probabilities[index, int(focus_class)]),
            "focus_decision_margin": float(locked_focus_margins[index]),
        }
        transition_rows.append(row)
    _write_csv(output_dir / "locked_transition_rows.csv", transition_rows)
    priority_rows = _priority_xai_rows(transition_rows)
    _write_csv(output_dir / "xai_priority_cases.csv", priority_rows)

    summary: Dict[str, object] = {
        "mode": "validation_only_precision_ensemble_readiness",
        "inputs": {
            "keeper_csv": str(keeper_csv.resolve()),
            "keeper_sha256": _sha256(keeper_csv),
            "candidate_csv": str(candidate_csv.resolve()),
            "candidate_sha256": _sha256(candidate_csv),
            "test_data_used": False,
        },
        "rows": int(len(keys)),
        "source_groups": int(len(set(groups.tolist()))),
        "class_names": class_names,
        "focus_class": int(focus_class),
        "protocol": {
            "folds": int(folds),
            "seed": int(seed),
            "minimum_focus_precision": float(minimum_focus_precision),
            "macro_tolerance": float(macro_tolerance),
            "precision_weight": float(precision_weight),
            "candidate_weight_grid": [float(value) for value in candidate_weights],
            "focus_margin_offset_grid": [float(value) for value in focus_margin_offsets],
            "selection": (
                "fit-fold precision floor; macro within tolerance of fit keeper; "
                "focus F1 not below fit keeper; maximize macro+focus_f1+precision_weight*precision"
            ),
            "lock_rule": "median fold winner snapped to the declared grids",
        },
        "models": {
            "keeper": keeper_metrics,
            "candidate": candidate_metrics,
            "candidate_vs_keeper": candidate_transitions,
        },
        "fold_selections": fold_rows,
        "oof": {
            "metrics": oof_metrics,
            "vs_keeper": oof_transitions,
        },
        "locked": {
            "candidate_weight": locked_weight,
            "keeper_weight": float(1.0 - locked_weight),
            "focus_margin_offset": locked_offset,
            "metrics": locked_metrics,
            "vs_keeper": locked_transitions,
            "vs_candidate": locked_vs_candidate,
        },
        "stability": stability,
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "artifacts": {
            "fold_selections": "fold_selections.csv",
            "oof_predictions": "oof_predictions_detailed.csv",
            "locked_val_predictions": "locked_val_predictions_detailed.csv",
            "transition_rows": "locked_transition_rows.csv",
            "xai_priority_cases": "xai_priority_cases.csv",
        },
        "guardrail": (
            "This is a validation-only readiness audit. The locked parameters may be applied "
            "to final test exactly once only after all gates pass; test must not alter them."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only source-grouped readiness audit for a precision-constrained "
            "keeper/candidate probability ensemble."
        )
    )
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-column", default="sample_index")
    parser.add_argument("--expected-rows", type=int, default=2606)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--minimum-focus-precision", type=float, default=0.65)
    parser.add_argument("--macro-tolerance", type=float, default=0.002)
    parser.add_argument("--precision-weight", type=float, default=0.10)
    parser.add_argument("--candidate-weight-start", type=float, default=0.0)
    parser.add_argument("--candidate-weight-stop", type=float, default=1.0)
    parser.add_argument("--candidate-weight-step", type=float, default=0.05)
    parser.add_argument("--focus-margin-start", type=float, default=0.0)
    parser.add_argument("--focus-margin-stop", type=float, default=0.10)
    parser.add_argument("--focus-margin-step", type=float, default=0.002)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_precision_ensemble_readiness(
        keeper_csv=args.keeper,
        candidate_csv=args.candidate,
        output_dir=args.output_dir,
        key_column=args.key_column,
        expected_rows=args.expected_rows,
        focus_class=args.focus_class,
        folds=args.folds,
        seed=args.seed,
        minimum_focus_precision=args.minimum_focus_precision,
        macro_tolerance=args.macro_tolerance,
        precision_weight=args.precision_weight,
        candidate_weight_start=args.candidate_weight_start,
        candidate_weight_stop=args.candidate_weight_stop,
        candidate_weight_step=args.candidate_weight_step,
        focus_margin_start=args.focus_margin_start,
        focus_margin_stop=args.focus_margin_stop,
        focus_margin_step=args.focus_margin_step,
    )
    focus = summary["locked"]["metrics"]["focus"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "rows": summary["rows"],
                "all_gates_passed": summary["all_gates_passed"],
                "candidate_weight": summary["locked"]["candidate_weight"],
                "focus_margin_offset": summary["locked"]["focus_margin_offset"],
                "macro_f1": summary["locked"]["metrics"]["macro_f1"],
                "focus_precision": focus["precision"],
                "focus_recall": focus["recall"],
                "focus_f1": focus["f1"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

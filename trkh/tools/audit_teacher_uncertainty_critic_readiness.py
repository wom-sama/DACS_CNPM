from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_.*)?$")
DEFAULT_CRITICAL_TRANSITIONS = ("0->1", "2->1", "4->1", "1->0", "1->2", "1->4")


@dataclass(frozen=True)
class PredictionTable:
    name: str
    path: Path
    sample_indices: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    rows: Sequence[Mapping[str, str]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    return rows


def _assert_not_test_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if re.search(r"(^|[_-])test(?:[_-]|\.|$)", Path(path).name.lower()):
        raise ValueError(f"Locked test artifact is not allowed in this audit: {path}")
    for row in rows:
        split = str(row.get("split", "") or "").strip().lower()
        if split == "test":
            raise ValueError(f"Locked test row is not allowed in this audit: {path}")
        image_path = str(row.get("image_path") or row.get("path") or "").replace("\\", "/").lower()
        if "/test/" in image_path:
            raise ValueError(f"Locked test image is not allowed in this audit: {image_path}")


def _class_count(rows: Sequence[Mapping[str, str]]) -> int:
    indices = {
        int(match.group(1))
        for key in rows[0]
        for match in [PROBABILITY_COLUMN.match(str(key))]
        if match is not None
    }
    if not indices:
        raise ValueError("Prediction CSV has no prob_* columns.")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(f"Probability class indices are not contiguous: {sorted(indices)}")
    return len(expected)


def _row_probability(row: Mapping[str, str], class_index: int) -> float:
    exact = f"prob_{class_index}"
    value = str(row.get(exact, "") or "").strip()
    if value:
        return float(value)
    candidates = sorted(key for key in row if str(key).startswith(f"{exact}_"))
    for key in candidates:
        value = str(row.get(key, "") or "").strip()
        if value:
            return float(value)
    raise ValueError(f"Missing probability for class {class_index}.")


def _load_prediction_table(name: str, path: Path, *, expected_class_count: int | None = None) -> PredictionTable:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    rows = _read_csv(path)
    _assert_not_test_rows(path, rows)
    class_count = _class_count(rows)
    if expected_class_count is not None and class_count != int(expected_class_count):
        raise ValueError(
            f"Class-count mismatch for {name}: expected {expected_class_count}, found {class_count}"
        )

    indexed_rows: list[tuple[int, Mapping[str, str]]] = []
    seen: set[int] = set()
    for row in rows:
        value = str(row.get("sample_index", "") or "").strip()
        if not value:
            raise ValueError(f"Strict sample_index is required for {name}: {path}")
        sample_index = int(value)
        if sample_index in seen:
            raise ValueError(f"Duplicate sample_index={sample_index} in {name}: {path}")
        seen.add(sample_index)
        indexed_rows.append((sample_index, row))
    indexed_rows.sort(key=lambda item: item[0])

    sample_indices = np.asarray([item[0] for item in indexed_rows], dtype=np.int64)
    probabilities = np.asarray(
        [[_row_probability(row, index) for index in range(class_count)] for _, row in indexed_rows],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    targets = []
    predictions = []
    sorted_rows = []
    for row_index, (_, row) in enumerate(indexed_rows):
        target_text = str(row.get("target_index", row.get("y_true", "")) or "").strip()
        if not target_text:
            raise ValueError(f"Missing target_index for sample_index={sample_indices[row_index]} in {name}")
        targets.append(int(target_text))
        prediction_text = str(
            row.get("prediction_index", row.get("y_pred", row.get("pred_index", ""))) or ""
        ).strip()
        prediction = int(prediction_text) if prediction_text else int(np.argmax(probabilities[row_index]))
        probability_prediction = int(np.argmax(probabilities[row_index]))
        if prediction != probability_prediction:
            raise ValueError(
                f"prediction_index/probability mismatch for {name} sample_index={sample_indices[row_index]}: "
                f"{prediction}!={probability_prediction}"
            )
        predictions.append(prediction)
        sorted_rows.append(row)
    return PredictionTable(
        name=name,
        path=path,
        sample_indices=sample_indices,
        targets=np.asarray(targets, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
        probabilities=probabilities,
        rows=sorted_rows,
    )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    class_count = int(probabilities.shape[1])
    scale = math.log(max(2, class_count))
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=1) / scale


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    left_term = np.sum(left * np.log(np.clip(left / midpoint, 1e-12, None)), axis=1)
    right_term = np.sum(right * np.log(np.clip(right / midpoint, 1e-12, None)), axis=1)
    return 0.5 * (left_term + right_term) / math.log(2.0)


def _mean_pairwise_js(expert_probabilities: np.ndarray) -> np.ndarray:
    expert_count = int(expert_probabilities.shape[0])
    if expert_count < 2:
        return np.zeros(expert_probabilities.shape[1], dtype=np.float64)
    values = [
        _js_divergence(expert_probabilities[left], expert_probabilities[right])
        for left in range(expert_count)
        for right in range(left + 1, expert_count)
    ]
    return np.mean(np.stack(values, axis=0), axis=0)


def _vote_disagreement(expert_probabilities: np.ndarray, class_count: int) -> np.ndarray:
    votes = np.argmax(expert_probabilities, axis=2)
    fractions = np.stack([(votes == index).mean(axis=0) for index in range(class_count)], axis=0)
    return 1.0 - np.max(fractions, axis=0)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positives].sum())
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _auc_payload(mask: np.ndarray, labels: np.ndarray, score: np.ndarray) -> Dict[str, object]:
    masked_labels = labels[mask]
    masked_scores = score[mask]
    positive = masked_labels == 1
    negative = ~positive
    return {
        "auc": _roc_auc(masked_labels, masked_scores),
        "samples": int(mask.sum()),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "positive_mean": float(masked_scores[positive].mean()) if positive.any() else None,
        "negative_mean": float(masked_scores[negative].mean()) if negative.any() else None,
    }


def _classification_metrics(targets: np.ndarray, predictions: np.ndarray, class_count: int) -> Dict[str, object]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    per_class = []
    f1_values = []
    for index in range(class_count):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": int(index),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "samples": int(len(targets)),
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _action_stats(
    mask: np.ndarray,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    routed_predictions: np.ndarray,
) -> Dict[str, object]:
    changed = mask & (routed_predictions != base_predictions)
    corrected = changed & (base_predictions != targets) & (routed_predictions == targets)
    harmed = changed & (base_predictions == targets) & (routed_predictions != targets)
    wrong_to_wrong = changed & (base_predictions != targets) & (routed_predictions != targets)
    changes = int(changed.sum())
    return {
        "eligible": int(mask.sum()),
        "changes": changes,
        "corrections": int(corrected.sum()),
        "harms": int(harmed.sum()),
        "wrong_to_wrong": int(wrong_to_wrong.sum()),
        "correction_precision": float(corrected.sum() / changes) if changes else None,
    }


def _route_summary(
    name: str,
    probabilities: np.ndarray,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    focus_class_index: int,
) -> Dict[str, object]:
    predictions = np.argmax(probabilities, axis=1)
    focus = int(focus_class_index)
    masks = {
        "all_changes": np.ones(len(targets), dtype=bool),
        "focus_suppress": (base_predictions == focus) & (predictions != focus),
        "focus_rescue": (base_predictions != focus) & (predictions == focus),
        "true_focus": targets == focus,
        "focus_boundary": (base_predictions == focus) | (predictions == focus) | (targets == focus),
    }
    return {
        "name": name,
        "metrics": _classification_metrics(targets, predictions, probabilities.shape[1]),
        "actions": {
            key: _action_stats(mask, targets, base_predictions, predictions)
            for key, mask in masks.items()
        },
    }


def _transition_support(
    targets: np.ndarray,
    base_predictions: np.ndarray,
    teacher_probabilities: np.ndarray,
    expert_probabilities: np.ndarray,
    transitions: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    teacher_predictions = np.argmax(teacher_probabilities, axis=1)
    expert_predictions = np.argmax(expert_probabilities, axis=2)
    class_count = int(teacher_probabilities.shape[1])
    output: Dict[str, Dict[str, object]] = {}
    for transition in transitions:
        left, right = (int(value) for value in transition.split("->", 1))
        if not (0 <= left < class_count and 0 <= right < class_count):
            raise ValueError(
                f"Transition {transition!r} is outside the configured {class_count}-class problem."
            )
        mask = (targets == left) & (base_predictions == right)
        support = int(mask.sum())
        changed = mask & (teacher_predictions != base_predictions)
        corrected = mask & (teacher_predictions == targets)
        wrong_to_wrong = changed & (teacher_predictions != targets)
        target_probability = teacher_probabilities[mask, left]
        target_vote_fraction = (expert_predictions[:, mask] == left).mean(axis=0) if support else np.asarray([])
        output[transition] = {
            "transition": transition,
            "support": support,
            "teacher_changes": int(changed.sum()),
            "teacher_corrections": int(corrected.sum()),
            "teacher_wrong_to_wrong": int(wrong_to_wrong.sum()),
            "correction_precision_among_changes": (
                float(corrected.sum() / changed.sum()) if changed.any() else None
            ),
            "mean_teacher_target_probability": (
                float(target_probability.mean()) if target_probability.size else None
            ),
            "mean_expert_target_vote_fraction": (
                float(target_vote_fraction.mean()) if target_vote_fraction.size else None
            ),
        }
    return output


def _risk_coverage(
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    *,
    score_name: str,
    focus_class_index: int,
    class_count: int,
) -> Dict[str, object]:
    order = np.argsort(uncertainty, kind="mergesort")
    cumulative_errors = np.cumsum(predictions[order] != targets[order])
    aurc = float(np.mean(cumulative_errors / np.arange(1, len(order) + 1)))
    rows = []
    for coverage in (0.50, 0.70, 0.80, 0.90, 0.95, 1.00):
        count = min(len(order), max(1, int(math.ceil(coverage * len(order)))))
        selected = order[:count]
        metrics = _classification_metrics(
            targets[selected], predictions[selected], int(class_count)
        )
        focus_metrics = metrics["per_class"][int(focus_class_index)]
        rows.append(
            {
                "score": score_name,
                "coverage": float(count / len(order)),
                "accepted": int(count),
                "risk": float(np.mean(predictions[selected] != targets[selected])),
                "macro_f1": float(metrics["macro_f1"]),
                "focus_f1": float(focus_metrics["f1"]),
                "focus_support": int(focus_metrics["support"]),
            }
        )
    return {"aurc": aurc, "rows": rows}


def _audit_split(
    split_name: str,
    base_path: Path,
    expert_paths: Mapping[str, Path],
    *,
    focus_class_index: int,
    critical_transitions: Sequence[str],
) -> tuple[Dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    base = _load_prediction_table("base", Path(base_path))
    class_count = int(base.probabilities.shape[1])
    if not 0 <= int(focus_class_index) < class_count:
        raise ValueError(f"focus_class_index={focus_class_index} is outside 0..{class_count - 1}")
    experts = [
        _load_prediction_table(name, Path(path), expected_class_count=class_count)
        for name, path in expert_paths.items()
    ]
    if not experts:
        raise ValueError(f"At least one expert is required for {split_name}.")
    for expert in experts:
        if not np.array_equal(expert.sample_indices, base.sample_indices):
            missing = sorted(set(base.sample_indices.tolist()) - set(expert.sample_indices.tolist()))[:10]
            extra = sorted(set(expert.sample_indices.tolist()) - set(base.sample_indices.tolist()))[:10]
            raise ValueError(
                f"Strict sample_index mismatch for {split_name}/{expert.name}; missing={missing}, extra={extra}"
            )
        if not np.array_equal(expert.targets, base.targets):
            mismatch = np.flatnonzero(expert.targets != base.targets)[:10].tolist()
            raise ValueError(f"Target mismatch for {split_name}/{expert.name} at rows {mismatch}")

    expert_probabilities = np.stack([expert.probabilities for expert in experts], axis=0)
    teacher_mean = np.mean(expert_probabilities, axis=0)
    teacher_mean /= teacher_mean.sum(axis=1, keepdims=True)
    sorted_base = np.sort(base.probabilities, axis=1)
    base_margin = sorted_base[:, -1] - sorted_base[:, -2]
    expected_teacher_entropy = np.mean(
        np.stack([_entropy(probabilities) for probabilities in expert_probabilities], axis=0), axis=0
    )
    teacher_entropy = _entropy(teacher_mean)
    features = {
        "base_entropy": _entropy(base.probabilities),
        "base_inverse_margin": 1.0 - base_margin,
        "base_one_minus_max": 1.0 - np.max(base.probabilities, axis=1),
        "teacher_mean_entropy": teacher_entropy,
        "teacher_expected_entropy": expected_teacher_entropy,
        "teacher_mutual_information": teacher_entropy - expected_teacher_entropy,
        "teacher_pairwise_js": _mean_pairwise_js(expert_probabilities),
        "base_teacher_js": _js_divergence(base.probabilities, teacher_mean),
        "teacher_vote_disagreement": _vote_disagreement(expert_probabilities, class_count),
        "teacher_focus_probability": teacher_mean[:, int(focus_class_index)],
        "teacher_nonfocus_probability": 1.0 - teacher_mean[:, int(focus_class_index)],
    }

    targets = base.targets
    predictions = base.predictions
    focus = int(focus_class_index)
    any_error = predictions != targets
    focus_fp = (predictions == focus) & (targets != focus)
    focus_fn = (targets == focus) & (predictions != focus)
    task_specs = {
        "error_detection_all": (np.ones(len(targets), dtype=bool), any_error.astype(np.int64)),
        "predicted_focus_fp_detection": (predictions == focus, focus_fp.astype(np.int64)),
        "true_focus_fn_detection": (targets == focus, focus_fn.astype(np.int64)),
        "focus_error_detection": (
            (targets == focus) | (predictions == focus),
            (focus_fp | focus_fn).astype(np.int64),
        ),
        "focus_fp_vs_fn_direction": (focus_fp | focus_fn, focus_fp.astype(np.int64)),
    }
    auc_tasks = {
        task_name: {
            feature_name: _auc_payload(mask, labels, values)
            for feature_name, values in features.items()
        }
        for task_name, (mask, labels) in task_specs.items()
    }

    routes = {
        expert.name: _route_summary(
            expert.name, expert.probabilities, targets, predictions, focus_class_index
        )
        for expert in experts
    }
    routes["teacher_mean"] = _route_summary(
        "teacher_mean", teacher_mean, targets, predictions, focus_class_index
    )
    transition_support = _transition_support(
        targets,
        predictions,
        teacher_mean,
        expert_probabilities,
        critical_transitions,
    )
    risk_coverage = {
        score_name: _risk_coverage(
            targets,
            predictions,
            features[score_name],
            score_name=score_name,
            focus_class_index=focus_class_index,
            class_count=class_count,
        )
        for score_name in (
            "base_entropy",
            "base_inverse_margin",
            "teacher_mean_entropy",
            "teacher_vote_disagreement",
        )
    }

    teacher_predictions = np.argmax(teacher_mean, axis=1)
    diagnostic_rows = []
    for row_index, sample_index in enumerate(base.sample_indices.tolist()):
        row: dict[str, object] = {
            "split": split_name,
            "sample_index": int(sample_index),
            "target_index": int(targets[row_index]),
            "base_prediction_index": int(predictions[row_index]),
            "teacher_mean_prediction_index": int(teacher_predictions[row_index]),
            "base_correct": int(predictions[row_index] == targets[row_index]),
            "teacher_mean_correct": int(teacher_predictions[row_index] == targets[row_index]),
            "focus_false_positive": int(focus_fp[row_index]),
            "focus_false_negative": int(focus_fn[row_index]),
            "transition": f"{targets[row_index]}->{predictions[row_index]}",
        }
        for feature_name, values in features.items():
            row[feature_name] = float(values[row_index])
        diagnostic_rows.append(row)

    risk_rows = [
        {"split": split_name, **row}
        for payload in risk_coverage.values()
        for row in payload["rows"]
    ]
    summary = {
        "split": split_name,
        "base_csv": str(base.path.resolve()),
        "expert_csvs": {expert.name: str(expert.path.resolve()) for expert in experts},
        "samples": int(len(targets)),
        "class_count": class_count,
        "focus_class_index": focus,
        "base_metrics": _classification_metrics(targets, predictions, class_count),
        "error_counts": {
            "all": int(any_error.sum()),
            "focus_false_positive": int(focus_fp.sum()),
            "focus_false_negative": int(focus_fn.sum()),
            "focus_true_positive": int(((targets == focus) & (predictions == focus)).sum()),
        },
        "auc_tasks": auc_tasks,
        "routes": routes,
        "transition_support": transition_support,
        "risk_coverage": risk_coverage,
    }
    return summary, diagnostic_rows, risk_rows


def _metric_auc(split: Mapping[str, object], task: str, feature: str) -> float | None:
    tasks = split.get("auc_tasks", {})
    if not isinstance(tasks, Mapping):
        return None
    task_payload = tasks.get(task, {})
    if not isinstance(task_payload, Mapping):
        return None
    feature_payload = task_payload.get(feature, {})
    if not isinstance(feature_payload, Mapping):
        return None
    value = feature_payload.get("auc")
    return float(value) if value is not None else None


def _action_precision(split: Mapping[str, object], action: str) -> float | None:
    routes = split.get("routes", {})
    if not isinstance(routes, Mapping):
        return None
    teacher = routes.get("teacher_mean", {})
    if not isinstance(teacher, Mapping):
        return None
    actions = teacher.get("actions", {})
    if not isinstance(actions, Mapping):
        return None
    payload = actions.get(action, {})
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("correction_precision")
    return float(value) if value is not None else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_teacher_uncertainty_critic_readiness(
    *,
    base_train: Path,
    train_experts: Mapping[str, Path],
    base_val: Path,
    val_experts: Mapping[str, Path],
    output_dir: Path,
    focus_class_index: int = 1,
    critical_transitions: Sequence[str] = DEFAULT_CRITICAL_TRANSITIONS,
    min_detection_auc: float = 0.75,
    min_direction_auc: float = 0.60,
    min_train_fp_transition_corrections: int = 10,
    min_train_fn_transition_corrections: int = 5,
    min_train_action_precision: float = 0.80,
    min_val_action_precision: float = 0.70,
) -> Dict[str, object]:
    if not train_experts or not val_experts:
        raise ValueError("Both train OOF and validation expert inputs are required.")
    for name, path in train_experts.items():
        if "oof" not in str(Path(path)).lower():
            raise ValueError(f"Train expert must be explicitly OOF/fold-safe ({name}): {path}")

    train, train_rows, train_risk = _audit_split(
        "train",
        Path(base_train),
        train_experts,
        focus_class_index=focus_class_index,
        critical_transitions=critical_transitions,
    )
    val, val_rows, val_risk = _audit_split(
        "validation",
        Path(base_val),
        val_experts,
        focus_class_index=focus_class_index,
        critical_transitions=critical_transitions,
    )
    if int(train["class_count"]) != int(val["class_count"]):
        raise ValueError("Train/validation class counts do not match.")

    blockers: list[str] = []
    auc_checks = {}
    for task in ("predicted_focus_fp_detection", "true_focus_fn_detection"):
        train_auc = _metric_auc(train, task, "teacher_nonfocus_probability")
        val_auc = _metric_auc(val, task, "teacher_nonfocus_probability")
        auc_checks[task] = {"train_auc": train_auc, "validation_auc": val_auc}
        if train_auc is None or train_auc < float(min_detection_auc):
            blockers.append(f"train_oof_detection_auc_below_min:{task}:{train_auc}")
        if val_auc is None or val_auc < float(min_detection_auc):
            blockers.append(f"validation_detection_auc_below_min:{task}:{val_auc}")

    train_direction_auc = _metric_auc(train, "focus_fp_vs_fn_direction", "teacher_nonfocus_probability")
    val_direction_auc = _metric_auc(val, "focus_fp_vs_fn_direction", "teacher_nonfocus_probability")
    auc_checks["focus_fp_vs_fn_direction"] = {
        "train_auc": train_direction_auc,
        "validation_auc": val_direction_auc,
    }
    if train_direction_auc is None or train_direction_auc < float(min_direction_auc):
        blockers.append(f"train_oof_direction_auc_below_min:{train_direction_auc}")
    if val_direction_auc is None or val_direction_auc < float(min_direction_auc):
        blockers.append(f"validation_direction_auc_below_min:{val_direction_auc}")
    if (
        train_direction_auc is not None
        and val_direction_auc is not None
        and (train_direction_auc - 0.5) * (val_direction_auc - 0.5) <= 0.0
    ):
        blockers.append(
            f"focus_fp_vs_fn_direction_sign_flip:train={train_direction_auc:.6f}:val={val_direction_auc:.6f}"
        )

    transition_checks = {}
    train_transition_support = train["transition_support"]
    for transition in critical_transitions:
        payload = train_transition_support[transition]
        minimum = (
            int(min_train_fp_transition_corrections)
            if transition.endswith(f"->{int(focus_class_index)}")
            else int(min_train_fn_transition_corrections)
        )
        corrections = int(payload["teacher_corrections"])
        transition_checks[transition] = {
            "train_oof_corrections": corrections,
            "minimum": minimum,
            "validation_corrections": int(val["transition_support"][transition]["teacher_corrections"]),
        }
        if corrections < minimum:
            blockers.append(f"train_oof_transition_support_below_min:{transition}:{corrections}<{minimum}")

    action_checks = {}
    for action in ("focus_suppress", "focus_rescue"):
        train_precision = _action_precision(train, action)
        val_precision = _action_precision(val, action)
        action_checks[action] = {
            "train_correction_precision": train_precision,
            "validation_correction_precision": val_precision,
        }
        if train_precision is None or train_precision < float(min_train_action_precision):
            blockers.append(f"train_oof_action_precision_below_min:{action}:{train_precision}")
        if val_precision is None or val_precision < float(min_val_action_precision):
            blockers.append(f"validation_action_precision_below_min:{action}:{val_precision}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transition_rows = []
    for split_name, payload in (("train", train), ("validation", val)):
        for transition, values in payload["transition_support"].items():
            transition_rows.append({"split": split_name, **values})
    _write_csv(output_dir / "sample_uncertainty_diagnostics.csv", [*train_rows, *val_rows])
    _write_csv(output_dir / "risk_coverage.csv", [*train_risk, *val_risk])
    _write_csv(output_dir / "transition_support.csv", transition_rows)

    smoke_gate_ready = not blockers
    summary: Dict[str, object] = {
        "mode": "teacher_uncertainty_critic_readiness",
        "note": (
            "No-test diagnostic. Train reliability comes only from explicitly OOF expert CSVs; "
            "the base train predictions may be in-sample and are used only to define student error candidates."
        ),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "focus_class_index": int(focus_class_index),
        "critical_transitions": list(critical_transitions),
        "thresholds": {
            "min_detection_auc": float(min_detection_auc),
            "min_direction_auc": float(min_direction_auc),
            "min_train_fp_transition_corrections": int(min_train_fp_transition_corrections),
            "min_train_fn_transition_corrections": int(min_train_fn_transition_corrections),
            "min_train_action_precision": float(min_train_action_precision),
            "min_validation_action_precision": float(min_val_action_precision),
        },
        "train": train,
        "validation": val,
        "gate_checks": {
            "auc": auc_checks,
            "transitions": transition_checks,
            "actions": action_checks,
        },
        "smoke_gate_ready": bool(smoke_gate_ready),
        "blocking_reasons": blockers,
        "decision": (
            "Eligible for one bounded uncertainty-critic smoke; this is not approval for full training or test use."
            if smoke_gate_ready
            else "Do not train a teacher-supervised uncertainty/evidential critic from the current caches. "
            "Uncertainty may rank ambiguous cases, but transition-safe correction support does not transfer."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    (output_dir / "README.md").write_text(
        "# Teacher uncertainty critic readiness\n\n"
        "This artifact is diagnostic-only. It reads train OOF and validation predictions, refuses test rows, "
        "writes no train manifest, and does not modify raw data. See `summary.json` for the fixed gate.\n",
        encoding="utf-8",
    )
    return summary


def _parse_named_paths(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expert input must be NAME=CSV: {value!r}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"Expert name is empty or duplicated: {name!r}")
        output[name] = Path(path_text.strip())
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether train-OOF teacher uncertainty supports a transition-safe class-1 critic. "
            "Locked test rows are always rejected."
        )
    )
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--train-expert", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--base-val", type=Path, required=True)
    parser.add_argument("--val-expert", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, object]:
    return build_teacher_uncertainty_critic_readiness(
        base_train=args.base_train,
        train_experts=_parse_named_paths(args.train_expert),
        base_val=args.base_val,
        val_experts=_parse_named_paths(args.val_expert),
        output_dir=args.output_dir,
        focus_class_index=args.focus_class_index,
    )


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        "smoke_gate_ready": summary["smoke_gate_ready"],
        "blocking_reasons": summary["blocking_reasons"],
        "output": "summary.json",
    }, indent=2))


if __name__ == "__main__":
    main()

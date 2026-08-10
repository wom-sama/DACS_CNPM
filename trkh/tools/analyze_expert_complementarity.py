from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_(.*))?$")


@dataclass(frozen=True)
class PredictionTable:
    name: str
    path: Path
    rows: Mapping[str, Mapping[str, str]]
    probabilities: Mapping[str, np.ndarray]
    predictions: Mapping[str, int]
    targets: Mapping[str, int]
    image_paths: Mapping[str, str]
    class_names: Sequence[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze object-level complementarity between expert prediction CSVs. "
            "Use validation only for diagnosis; do not treat oracle numbers as deployable metrics."
        )
    )
    parser.add_argument("--input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=None,
        help="Optional teacher/ensemble probability CSV used as the baseline prediction.",
    )
    parser.add_argument(
        "--base-name",
        default="base",
        help="Name for --base-csv in summaries.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Fallback weighted average if --base-csv is omitted, as NAME:WEIGHT comma list.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--top-cases", type=int, default=80)
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CSV, got: {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Empty expert name in {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return name, path


def _parse_weights(value: str, names: Sequence[str]) -> Dict[str, float]:
    weights = {name: 1.0 for name in names}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Weight item must be NAME:WEIGHT, got {item!r}")
        name, weight_text = item.split(":", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"Weight specified for unknown expert {name!r}")
        weight = float(weight_text)
        if weight < 0.0:
            raise ValueError(f"Weight must be non-negative for {name!r}: {weight}")
        weights[name] = weight
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("At least one expert weight must be positive.")
    return {name: float(value / total) for name, value in weights.items()}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _row_key(row: Mapping[str, str], path: Path) -> str:
    sample_index = str(row.get("sample_index", "") or "").strip()
    if sample_index:
        try:
            return f"sample:{int(sample_index)}"
        except ValueError as exc:
            raise ValueError(f"Invalid sample_index {sample_index!r} in {path}") from exc
    image_path = str(row.get("path") or row.get("image_path") or "").strip()
    if not image_path:
        raise ValueError(f"CSV lacks sample_index and path/image_path: {path}")
    return f"path:{image_path.lower()}"


def _probability_columns(row: Mapping[str, str]) -> List[Tuple[int, str, str]]:
    columns: List[Tuple[int, str, str]] = []
    for key in row:
        match = PROBABILITY_COLUMN.match(str(key))
        if match is None:
            continue
        suffix = str(match.group(2) or "")
        columns.append((int(match.group(1)), key, suffix))
    return sorted(columns, key=lambda item: item[0])


def _infer_class_names(rows: Sequence[Mapping[str, str]]) -> List[str]:
    first_columns = _probability_columns(rows[0])
    if first_columns:
        max_index = max(index for index, _, _ in first_columns)
        names = [str(index) for index in range(max_index + 1)]
        for index, _, suffix in first_columns:
            if suffix:
                names[index] = suffix
        for row in rows:
            for index_key, name_key in (
                ("target_index", "target_name"),
                ("prediction_index", "prediction_name"),
                ("y_true", "true_name"),
                ("y_pred", "pred_name"),
            ):
                index_text = str(row.get(index_key, "") or "").strip()
                name = str(row.get(name_key, "") or "").strip()
                if index_text and name:
                    names[int(index_text)] = name
        return names

    indexed_names: Dict[int, str] = {}
    for row in rows:
        for index_key, name_key in (
            ("target_index", "target_name"),
            ("prediction_index", "prediction_name"),
            ("y_true", "true_name"),
            ("y_pred", "pred_name"),
        ):
            index_text = str(row.get(index_key, "") or "").strip()
            name = str(row.get(name_key, "") or "").strip()
            if index_text and name:
                indexed_names[int(index_text)] = name
    if not indexed_names:
        raise ValueError("Cannot infer class names.")
    expected = list(range(max(indexed_names) + 1))
    if sorted(indexed_names) != expected:
        raise ValueError(f"Non-contiguous class indices: {sorted(indexed_names)}")
    return [indexed_names[index] for index in expected]


def _target_index(row: Mapping[str, str]) -> int | None:
    for key in ("target_index", "y_true", "true_index"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(value)
    return None


def _prediction_index(row: Mapping[str, str]) -> int | None:
    for key in ("prediction_index", "y_pred", "pred_index", "teacher_pred_index"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(value)
    return None


def _row_probabilities(row: Mapping[str, str], class_count: int) -> np.ndarray:
    probabilities = np.zeros(class_count, dtype=np.float64)
    found = False
    for index, column, _ in _probability_columns(row):
        if index >= class_count:
            continue
        value = str(row.get(column, "") or "").strip()
        if value:
            probabilities[index] = float(value)
            found = True
    if not found:
        prediction = _prediction_index(row)
        if prediction is None:
            raise ValueError("Row has neither prob_* columns nor prediction_index.")
        probabilities[int(prediction)] = 1.0
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("Probability row sums to zero.")
    return probabilities / total


def _load_prediction_table(name: str, path: Path, class_count: int | None = None) -> PredictionTable:
    rows = _read_csv(path)
    class_names = _infer_class_names(rows)
    if class_count is None:
        class_count = len(class_names)
    if len(class_names) < class_count:
        class_names = list(class_names) + [str(index) for index in range(len(class_names), class_count)]

    rows_by_key: Dict[str, Mapping[str, str]] = {}
    probabilities: Dict[str, np.ndarray] = {}
    predictions: Dict[str, int] = {}
    targets: Dict[str, int] = {}
    image_paths: Dict[str, str] = {}
    duplicate_keys = 0
    for row in rows:
        key = _row_key(row, path)
        if key in rows_by_key:
            duplicate_keys += 1
        rows_by_key[key] = row
        probability = _row_probabilities(row, class_count)
        probabilities[key] = probability
        predictions[key] = int(np.argmax(probability))
        target = _target_index(row)
        if target is not None:
            targets[key] = int(target)
        image_paths[key] = str(row.get("path") or row.get("image_path") or "")
    if duplicate_keys:
        raise ValueError(f"{path} has duplicate keys after sample_index/path mapping: {duplicate_keys}")
    return PredictionTable(
        name=name,
        path=path,
        rows=rows_by_key,
        probabilities=probabilities,
        predictions=predictions,
        targets=targets,
        image_paths=image_paths,
        class_names=class_names[:class_count],
    )


def _sorted_keys(keys: Iterable[str]) -> List[str]:
    def sort_key(value: str) -> Tuple[int, object]:
        if value.startswith("sample:"):
            return (0, int(value.split(":", 1)[1]))
        return (1, value)

    return sorted(keys, key=sort_key)


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
    class_names: Sequence[str],
) -> Dict[str, object]:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if 0 <= target < class_count and 0 <= prediction < class_count:
            matrix[target, prediction] += 1
    per_class = []
    f1_values = []
    for index in range(class_count):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_support = int(matrix[:, index].sum())
        precision = float(tp / predicted_support) if predicted_support else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": index,
                "class_name": str(class_names[index]) if index < len(class_names) else str(index),
                "support": support,
                "predicted_support": predicted_support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _top2_margin(probabilities: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(probabilities, axis=1)
    if probabilities.shape[1] < 2:
        return sorted_values[:, -1]
    return sorted_values[:, -1] - sorted_values[:, -2]


def _vote_entropy(predictions: np.ndarray, class_count: int) -> np.ndarray:
    entropies = []
    for row in predictions.T:
        counts = np.bincount(row, minlength=class_count).astype(np.float64)
        fractions = counts / max(1.0, float(counts.sum()))
        non_zero = fractions[fractions > 0.0]
        entropies.append(float(-np.sum(non_zero * np.log(non_zero))))
    return np.asarray(entropies, dtype=np.float64)


def _array_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _pattern_key(names: Sequence[str]) -> str:
    return "+".join(names) if names else "none"


def _write_case_csv(
    path: Path,
    keys: Sequence[str],
    targets: np.ndarray,
    base_predictions: np.ndarray,
    base_probabilities: np.ndarray,
    expert_names: Sequence[str],
    expert_predictions: np.ndarray,
    expert_probabilities: np.ndarray,
    image_paths: Mapping[str, str],
    focus_class: int,
    case_kind: str,
    limit: int,
) -> None:
    rows = []
    for row_index, key in enumerate(keys):
        if case_kind == "focus_fn":
            if not (int(targets[row_index]) == focus_class and int(base_predictions[row_index]) != focus_class):
                continue
            helpers = [
                name
                for expert_index, name in enumerate(expert_names)
                if int(expert_predictions[expert_index, row_index]) == focus_class
            ]
        elif case_kind == "focus_fp":
            if not (int(targets[row_index]) != focus_class and int(base_predictions[row_index]) == focus_class):
                continue
            helpers = [
                name
                for expert_index, name in enumerate(expert_names)
                if int(expert_predictions[expert_index, row_index]) == int(targets[row_index])
            ]
        else:
            raise ValueError(f"Unknown case kind: {case_kind}")

        focus_votes = float(np.mean(expert_predictions[:, row_index] == focus_class))
        margin = float(_top2_margin(base_probabilities[row_index : row_index + 1])[0])
        rows.append(
            {
                "key": key,
                "sample_index": key.split(":", 1)[1] if key.startswith("sample:") else "",
                "image_path": image_paths.get(key, ""),
                "target_index": int(targets[row_index]),
                "base_prediction_index": int(base_predictions[row_index]),
                "base_focus_probability": float(base_probabilities[row_index, focus_class]),
                "base_margin": margin,
                "expert_focus_vote_fraction": focus_votes,
                "helpers": _pattern_key(helpers),
                **{
                    f"{name}_prediction": int(expert_predictions[expert_index, row_index])
                    for expert_index, name in enumerate(expert_names)
                },
                **{
                    f"{name}_focus_probability": float(
                        expert_probabilities[expert_index, row_index, focus_class]
                    )
                    for expert_index, name in enumerate(expert_names)
                },
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["expert_focus_vote_fraction"]) if case_kind == "focus_fn" else float(row["expert_focus_vote_fraction"]),
            -float(row["base_focus_probability"]) if case_kind == "focus_fp" else float(row["base_focus_probability"]),
            float(row["base_margin"]),
        )
    )
    rows = rows[: max(0, int(limit))]
    fieldnames = [
        "key",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction_index",
        "base_focus_probability",
        "base_margin",
        "expert_focus_vote_fraction",
        "helpers",
    ]
    for name in expert_names:
        fieldnames.append(f"{name}_prediction")
    for name in expert_names:
        fieldnames.append(f"{name}_focus_probability")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    named_paths = [_parse_named_path(value) for value in args.input]
    expert_names = [name for name, _ in named_paths]
    if len(set(expert_names)) != len(expert_names):
        raise ValueError(f"Duplicate expert names: {expert_names}")

    first_rows = _read_csv(named_paths[0][1])
    class_names = _infer_class_names(first_rows)
    class_count = len(class_names)
    if not (0 <= int(args.focus_class_index) < class_count):
        raise ValueError(f"--focus-class-index out of range for {class_count} classes.")

    experts = [
        _load_prediction_table(name=name, path=path, class_count=class_count)
        for name, path in named_paths
    ]
    common_keys = set(experts[0].rows)
    for table in experts[1:]:
        common_keys &= set(table.rows)
    if not common_keys:
        raise ValueError("No common samples across expert CSVs.")

    keys = _sorted_keys(common_keys)
    targets = np.asarray([experts[0].targets[key] for key in keys], dtype=np.int64)
    for table in experts[1:]:
        mismatches = [key for key in keys if int(table.targets.get(key, -999)) != int(experts[0].targets[key])]
        if mismatches:
            raise ValueError(f"Target mismatch in {table.name}; first key: {mismatches[0]}")

    expert_probabilities = np.stack(
        [
            np.stack([table.probabilities[key] for key in keys], axis=0)
            for table in experts
        ],
        axis=0,
    )
    expert_predictions = expert_probabilities.argmax(axis=2)

    if args.base_csv is not None:
        base_table = _load_prediction_table(
            name=str(args.base_name),
            path=Path(args.base_csv),
            class_count=class_count,
        )
        missing = [key for key in keys if key not in base_table.probabilities]
        if missing:
            raise ValueError(f"Base CSV misses {len(missing)} samples; first: {missing[0]}")
        base_probabilities = np.stack([base_table.probabilities[key] for key in keys], axis=0)
        base_name = str(args.base_name)
    else:
        weights = _parse_weights(str(args.weights), expert_names)
        weight_array = np.asarray([weights[name] for name in expert_names], dtype=np.float64)
        base_probabilities = np.tensordot(weight_array, expert_probabilities, axes=(0, 0))
        base_probabilities = base_probabilities / np.clip(
            base_probabilities.sum(axis=1, keepdims=True), 1e-12, None
        )
        base_name = "weighted_average"
    base_predictions = base_probabilities.argmax(axis=1)

    any_expert_correct = (expert_predictions == targets.reshape(1, -1)).any(axis=0)
    oracle_predictions = base_predictions.copy()
    oracle_predictions[any_expert_correct] = targets[any_expert_correct]
    focus = int(args.focus_class_index)
    focus_fn = (targets == focus) & (base_predictions != focus)
    focus_fp = (targets != focus) & (base_predictions == focus)
    focus_related_base_error = focus_fn | focus_fp
    focus_oracle_predictions = base_predictions.copy()
    focus_oracle_mask = focus_related_base_error & any_expert_correct
    focus_oracle_predictions[focus_oracle_mask] = targets[focus_oracle_mask]

    metrics_by_model: Dict[str, object] = {
        base_name: _classification_metrics(targets, base_predictions, class_count, class_names),
        "any_expert_oracle": _classification_metrics(
            targets, oracle_predictions, class_count, class_names
        ),
        "focus_error_only_oracle": _classification_metrics(
            targets, focus_oracle_predictions, class_count, class_names
        ),
    }
    for expert_index, name in enumerate(expert_names):
        metrics_by_model[name] = _classification_metrics(
            targets, expert_predictions[expert_index], class_count, class_names
        )

    rescue_counts: Dict[str, int] = {}
    fp_block_correct_counts: Dict[str, int] = {}
    fp_block_nonfocus_counts: Dict[str, int] = {}
    exclusive_fn_rescue = Counter()
    exclusive_fp_block = Counter()
    fn_patterns = Counter()
    fp_patterns = Counter()
    for row_index in range(len(keys)):
        fn_rescuers = [
            name
            for expert_index, name in enumerate(expert_names)
            if focus_fn[row_index] and int(expert_predictions[expert_index, row_index]) == focus
        ]
        fp_correct_blockers = [
            name
            for expert_index, name in enumerate(expert_names)
            if focus_fp[row_index]
            and int(expert_predictions[expert_index, row_index]) == int(targets[row_index])
        ]
        fp_nonfocus_blockers = [
            name
            for expert_index, name in enumerate(expert_names)
            if focus_fp[row_index] and int(expert_predictions[expert_index, row_index]) != focus
        ]
        if focus_fn[row_index]:
            fn_patterns[_pattern_key(fn_rescuers)] += 1
            if len(fn_rescuers) == 1:
                exclusive_fn_rescue[fn_rescuers[0]] += 1
            for name in fn_rescuers:
                rescue_counts[name] = rescue_counts.get(name, 0) + 1
        if focus_fp[row_index]:
            fp_patterns[_pattern_key(fp_correct_blockers)] += 1
            if len(fp_correct_blockers) == 1:
                exclusive_fp_block[fp_correct_blockers[0]] += 1
            for name in fp_correct_blockers:
                fp_block_correct_counts[name] = fp_block_correct_counts.get(name, 0) + 1
            for name in fp_nonfocus_blockers:
                fp_block_nonfocus_counts[name] = fp_block_nonfocus_counts.get(name, 0) + 1

    base_margins = _top2_margin(base_probabilities)
    focus_vote_fraction = np.mean(expert_predictions == focus, axis=0)
    vote_entropy = _vote_entropy(expert_predictions, class_count)
    masks = {
        "all": np.ones(len(keys), dtype=bool),
        "base_correct": base_predictions == targets,
        "base_wrong": base_predictions != targets,
        "focus_fn": focus_fn,
        "focus_fp": focus_fp,
        "focus_related_base_error": focus_related_base_error,
    }
    group_stats = {}
    for name, mask in masks.items():
        group_stats[name] = {
            "count": int(mask.sum()),
            "base_focus_probability": _array_stats(base_probabilities[mask, focus]),
            "base_margin": _array_stats(base_margins[mask]),
            "expert_focus_vote_fraction": _array_stats(focus_vote_fraction[mask]),
            "expert_vote_entropy": _array_stats(vote_entropy[mask]),
        }

    image_paths = experts[0].image_paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_case_csv(
        args.output_dir / "focus_false_negative_cases.csv",
        keys=keys,
        targets=targets,
        base_predictions=base_predictions,
        base_probabilities=base_probabilities,
        expert_names=expert_names,
        expert_predictions=expert_predictions,
        expert_probabilities=expert_probabilities,
        image_paths=image_paths,
        focus_class=focus,
        case_kind="focus_fn",
        limit=int(args.top_cases),
    )
    _write_case_csv(
        args.output_dir / "focus_false_positive_cases.csv",
        keys=keys,
        targets=targets,
        base_predictions=base_predictions,
        base_probabilities=base_probabilities,
        expert_names=expert_names,
        expert_predictions=expert_predictions,
        expert_probabilities=expert_probabilities,
        image_paths=image_paths,
        focus_class=focus,
        case_kind="focus_fp",
        limit=int(args.top_cases),
    )

    summary = {
        "base_name": base_name,
        "samples": int(len(keys)),
        "focus_class_index": focus,
        "focus_class_name": str(class_names[focus]),
        "experts": expert_names,
        "metrics": metrics_by_model,
        "focus_error_counts": {
            "base_focus_false_negative": int(focus_fn.sum()),
            "base_focus_false_positive": int(focus_fp.sum()),
            "focus_error_any_expert_correct": int((focus_related_base_error & any_expert_correct).sum()),
            "focus_false_negative_any_expert_predicts_focus": int(
                ((expert_predictions == focus).any(axis=0) & focus_fn).sum()
            ),
            "focus_false_positive_any_expert_correct_target": int(
                ((expert_predictions == targets.reshape(1, -1)).any(axis=0) & focus_fp).sum()
            ),
        },
        "focus_false_negative_rescue_counts": dict(sorted(rescue_counts.items())),
        "focus_false_positive_correct_block_counts": dict(sorted(fp_block_correct_counts.items())),
        "focus_false_positive_nonfocus_block_counts": dict(sorted(fp_block_nonfocus_counts.items())),
        "exclusive_focus_false_negative_rescue": dict(sorted(exclusive_fn_rescue.items())),
        "exclusive_focus_false_positive_correct_block": dict(sorted(exclusive_fp_block.items())),
        "top_focus_false_negative_rescue_patterns": fn_patterns.most_common(20),
        "top_focus_false_positive_correct_block_patterns": fp_patterns.most_common(20),
        "group_stats": group_stats,
        "outputs": {
            "focus_false_negative_cases": str(args.output_dir / "focus_false_negative_cases.csv"),
            "focus_false_positive_cases": str(args.output_dir / "focus_false_positive_cases.csv"),
        },
        "notes": [
            "Oracle metrics use validation labels to expose complementarity only.",
            "Do not report oracle numbers as deployable model performance.",
            "Do not tune thresholds on test; any router should be trained from train/OOF signals.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

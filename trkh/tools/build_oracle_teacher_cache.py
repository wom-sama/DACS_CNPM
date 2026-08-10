from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only oracle-style teacher cache from multiple prediction CSVs. "
            "For each sample, probabilities from experts that already predict the hard label "
            "are averaged; if none are correct, all experts are averaged. Use only on train."
        )
    )
    parser.add_argument("--input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--selection",
        choices=("correct_mean", "correct_best_confidence"),
        default="correct_mean",
    )
    parser.add_argument("--hard-target-blend", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in str(value):
        raise ValueError(f"Expected NAME=CSV, got {value!r}")
    name, path_text = str(value).split("=", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Empty expert name in {value!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def _sample_key(row: Mapping[str, str], *, source: Path) -> str:
    sample_index = str(row.get("sample_index", "") or "").strip()
    if sample_index:
        return f"sample:{int(sample_index)}"
    path = str(row.get("path") or row.get("image_path") or "").strip()
    if not path:
        raise ValueError(f"Prediction row in {source} lacks sample_index and path/image_path.")
    return f"path:{path.lower()}"


def _probability_columns(row: Mapping[str, str]) -> List[Tuple[int, str]]:
    columns: List[Tuple[int, str]] = []
    for key in row:
        if not key.startswith("prob_"):
            continue
        parts = str(key).split("_", 2)
        if len(parts) < 2:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        columns.append((index, key))
    columns.sort(key=lambda item: item[0])
    return columns


def _read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    if not _probability_columns(rows[0]):
        raise ValueError(f"Prediction CSV has no prob_* columns: {path}")
    output: Dict[str, Dict[str, str]] = {}
    duplicate_keys: List[str] = []
    for row in rows:
        key = _sample_key(row, source=path)
        if key in output:
            duplicate_keys.append(key)
            continue
        output[key] = row
    if duplicate_keys:
        raise ValueError(f"{path} has duplicate sample keys: {duplicate_keys[:5]}")
    return output


def _row_probabilities(row: Mapping[str, str]) -> np.ndarray:
    columns = _probability_columns(row)
    probabilities = np.asarray([float(row[column]) for _, column in columns], dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum()


def _int_value(row: Mapping[str, str], names: Sequence[str]) -> int:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return int(value)
    raise ValueError(f"Missing integer column among {names}")


def _name_value(row: Mapping[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(1e-6, float(temperature))
    if abs(temperature - 1.0) <= 1e-8:
        return probabilities
    output = np.clip(probabilities, 1e-12, None) ** (1.0 / temperature)
    return output / max(1e-12, float(output.sum()))


def _metrics(rows: Sequence[Mapping[str, object]], class_count: int) -> Dict[str, object]:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for row in rows:
        matrix[int(row["target_index"]), int(row["teacher_pred_index"])] += 1
    per_class: List[Dict[str, object]] = []
    for index in range(class_count):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class.append(
            {
                "class_index": index,
                "support": support,
                "predicted_support": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(sum(float(item["f1"]) for item in per_class) / max(1, class_count)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def build_oracle_teacher(
    named_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    *,
    selection: str,
    hard_target_blend: float,
    temperature: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    names = list(named_rows)
    keys = sorted(set.intersection(*(set(rows) for rows in named_rows.values())))
    if not keys:
        raise ValueError("Input prediction CSVs do not share any sample keys.")
    hard_target_blend = min(1.0, max(0.0, float(hard_target_blend)))
    records: List[Dict[str, object]] = []
    selection_counter: Counter[str] = Counter()
    expert_use_counter: Counter[str] = Counter()
    for key in keys:
        rows = {name: named_rows[name][key] for name in names}
        reference = rows[names[0]]
        target_index = _int_value(reference, ("target_index", "y_true"))
        target_name = _name_value(reference, ("target_name", "true_name"))
        image_path = str(reference.get("image_path") or reference.get("path") or "")
        sample_index = str(reference.get("sample_index", "") or "").strip()
        expert_probabilities: List[Tuple[str, np.ndarray, float, bool]] = []
        for name, row in rows.items():
            probabilities = _row_probabilities(row)
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            expert_probabilities.append((name, probabilities, confidence, prediction == target_index))
        correct_experts = [item for item in expert_probabilities if item[3]]
        selected = correct_experts if correct_experts else expert_probabilities
        if correct_experts:
            selection_counter["correct_expert"] += 1
        else:
            selection_counter["fallback_all_wrong"] += 1
        if selection == "correct_best_confidence" and correct_experts:
            selected = [max(correct_experts, key=lambda item: item[2])]
        probability = np.stack([item[1] for item in selected], axis=0).mean(axis=0)
        probability = probability / max(1e-12, float(probability.sum()))
        if hard_target_blend > 0.0:
            one_hot = np.zeros_like(probability)
            one_hot[target_index] = 1.0
            probability = (1.0 - hard_target_blend) * probability + hard_target_blend * one_hot
            probability = probability / max(1e-12, float(probability.sum()))
        probability = _apply_temperature(probability, temperature)
        pred_index = int(np.argmax(probability))
        selected_names = [item[0] for item in selected]
        expert_use_counter.update(selected_names)
        record: Dict[str, object] = {
            "sample_index": sample_index,
            "path": image_path,
            "true_name": target_name,
            "target_index": int(target_index),
            "teacher_pred_index": pred_index,
            "teacher_pred_name": target_name if pred_index == target_index else str(pred_index),
            "teacher_confidence": float(probability[pred_index]),
            "selection_reason": "correct_expert" if correct_experts else "fallback_all_wrong",
            "selected_experts": ",".join(selected_names),
        }
        for index, value in enumerate(probability.tolist()):
            record[f"prob_{index}"] = float(value)
        records.append(record)
    summary = {
        "samples": len(records),
        "selection_counts": dict(selection_counter),
        "expert_use_counts": dict(expert_use_counter),
        "class_count": len(_row_probabilities(next(iter(next(iter(named_rows.values())).values())))),
    }
    return records, summary


def main() -> None:
    args = parse_args()
    named_paths = [_parse_named_path(value) for value in args.input]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate expert names: {names}")
    named_rows = {name: _read_csv(path) for name, path in named_paths}
    records, summary = build_oracle_teacher(
        named_rows,
        selection=str(args.selection),
        hard_target_blend=float(args.hard_target_blend),
        temperature=float(args.temperature),
    )
    class_count = int(summary["class_count"])
    metrics = _metrics(records, class_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"teacher_probs_{args.split}.csv"
    fieldnames = [
        "sample_index",
        "path",
        "true_name",
        "target_index",
        "teacher_pred_index",
        "teacher_pred_name",
        "teacher_confidence",
        "selection_reason",
        "selected_experts",
        *[f"prob_{index}" for index in range(class_count)],
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    output_json = args.output_dir / f"teacher_probs_{args.split}_metrics.json"
    output_json.write_text(
        json.dumps(
            {
                "mode": "oracle_teacher_cache",
                "split": str(args.split),
                "inputs": {name: str(path) for name, path in named_paths},
                "selection": str(args.selection),
                "hard_target_blend": float(args.hard_target_blend),
                "temperature": float(args.temperature),
                **summary,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_csv": str(output_csv), "metrics": metrics, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

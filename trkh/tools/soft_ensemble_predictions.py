from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PredictionTable:
    name: str
    path: Path
    rows_by_key: Dict[str, Mapping[str, str]]
    class_names: List[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blend prediction CSV probabilities with fixed weights. "
            "Use a frozen validation-selected weight before any final test audit."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Prediction CSV with prob_<index> columns. Can be repeated.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma list NAME:WEIGHT. Missing names use weight 1.0.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-column", default="sample_index")
    parser.add_argument("--min-models", type=int, default=0)
    return parser.parse_args(argv)


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in str(value):
        raise ValueError(f"--input must have NAME=CSV format, got {value!r}")
    name, path_text = str(value).split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--input has empty name: {value!r}")
    path = Path(path_text.strip())
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return name, path


def _parse_weights(value: str, names: Sequence[str]) -> Dict[str, float]:
    weights = {str(name): 1.0 for name in names}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Weight item must have NAME:WEIGHT format: {item!r}")
        name, weight_text = item.split(":", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"Weight provided for unknown input {name!r}")
        weight = float(weight_text)
        if weight < 0.0:
            raise ValueError(f"Weight must be >= 0 for {name!r}")
        weights[name] = weight
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("At least one ensemble weight must be positive.")
    return {name: float(weight / total) for name, weight in weights.items()}


def _probability_columns(fieldnames: Sequence[str]) -> Tuple[List[str], List[str]]:
    by_index: Dict[int, Tuple[str, str]] = {}
    for field in fieldnames:
        text = str(field)
        if not text.startswith("prob_"):
            continue
        rest = text[len("prob_") :]
        index_text, _, class_name = rest.partition("_")
        if not index_text.isdigit():
            continue
        index = int(index_text)
        if index in by_index:
            continue
        by_index[index] = (text, class_name or str(index))
    if not by_index:
        raise ValueError("Prediction CSV does not contain prob_<index> columns.")
    expected = list(range(len(by_index)))
    if sorted(by_index) != expected:
        raise ValueError(f"Probability columns must be contiguous from 0: {sorted(by_index)}")
    return (
        [by_index[index][0] for index in expected],
        [by_index[index][1] for index in expected],
    )


def _read_table(name: str, path: Path, key_column: str) -> PredictionTable:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV has no header: {path}")
        if key_column not in reader.fieldnames:
            raise ValueError(f"Prediction CSV missing key column {key_column!r}: {path}")
        prob_columns, class_names = _probability_columns(reader.fieldnames)
        rows_by_key: Dict[str, Mapping[str, str]] = {}
        duplicate_keys = 0
        for row in reader:
            key = str(row.get(key_column, "")).strip()
            if not key:
                continue
            if key in rows_by_key:
                duplicate_keys += 1
            # Keep only rows with parseable probabilities so failures are early.
            _row_probabilities(row, prob_columns)
            rows_by_key[key] = dict(row)
    if duplicate_keys:
        raise ValueError(f"Duplicate {key_column} values in {path}: {duplicate_keys}")
    if not rows_by_key:
        raise ValueError(f"Prediction CSV has no keyed rows: {path}")
    return PredictionTable(name=name, path=path, rows_by_key=rows_by_key, class_names=class_names)


def _row_probabilities(row: Mapping[str, str], columns: Sequence[str]) -> List[float]:
    probabilities = [max(0.0, float(row[column])) for column in columns]
    total = sum(probabilities)
    if total <= 0.0:
        return [1.0 / float(len(probabilities)) for _ in probabilities]
    return [float(value / total) for value in probabilities]


def _target_index(row: Mapping[str, str]) -> int:
    for key in ("target_index", "y_true"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(value)
    raise ValueError("Row has no target_index/y_true column.")


def _target_name(row: Mapping[str, str], class_names: Sequence[str]) -> str:
    for key in ("target_name", "true_name"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return class_names[_target_index(row)]


def _source_path(row: Mapping[str, str]) -> str:
    for key in ("image_path", "path"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    class_count = len(class_names)
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    per_class = []
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(class_count))
    for index, class_name in enumerate(class_names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_support = sum(matrix[row][index] for row in range(class_count))
        precision = float(tp / predicted_support) if predicted_support else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class.append(
            {
                "class_index": index,
                "class_name": str(class_name),
                "support": int(support),
                "predicted_support": int(predicted_support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_f1": float(sum(item["f1"] for item in per_class) / class_count)
        if class_count
        else 0.0,
        "weighted_f1": float(
            sum(item["f1"] * item["support"] for item in per_class) / max(1, total)
        ),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def run_soft_ensemble(
    *,
    inputs: Sequence[str],
    weights_text: str,
    output_dir: Path,
    key_column: str = "sample_index",
    min_models: int = 0,
) -> Dict[str, object]:
    named_paths = [_parse_named_path(value) for value in inputs]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate input names are not allowed: {names}")
    tables = [_read_table(name, path, key_column) for name, path in named_paths]
    class_names = list(tables[0].class_names)
    for table in tables[1:]:
        if len(table.class_names) != len(class_names):
            raise ValueError(
                f"Class count mismatch for {table.name}: {table.class_names} vs {class_names}"
            )
    weights = _parse_weights(weights_text, names)
    min_models = int(min_models) if int(min_models) > 0 else len(tables)

    keys = sorted(set().union(*(set(table.rows_by_key) for table in tables)), key=lambda v: int(v) if v.isdigit() else v)
    records: List[Dict[str, object]] = []
    targets: List[int] = []
    predictions: List[int] = []
    prob_columns_by_name = {
        table.name: _probability_columns(next(iter(table.rows_by_key.values())).keys())[0]
        for table in tables
    }
    for key in keys:
        present = [table for table in tables if key in table.rows_by_key]
        if len(present) < min_models:
            continue
        base_row = present[0].rows_by_key[key]
        target = _target_index(base_row)
        probability = [0.0 for _ in class_names]
        used_weight = 0.0
        model_predictions: Dict[str, int] = {}
        for table in present:
            row = table.rows_by_key[key]
            row_prob = _row_probabilities(row, prob_columns_by_name[table.name])
            weight = float(weights[table.name])
            used_weight += weight
            for index, value in enumerate(row_prob):
                probability[index] += weight * float(value)
            model_predictions[table.name] = int(max(range(len(row_prob)), key=row_prob.__getitem__))
        if used_weight <= 0.0:
            continue
        probability = [float(value / used_weight) for value in probability]
        prediction = int(max(range(len(probability)), key=probability.__getitem__))
        targets.append(target)
        predictions.append(prediction)
        record: Dict[str, object] = {
            key_column: key,
            "path": _source_path(base_row),
            "target_index": target,
            "target_name": _target_name(base_row, class_names),
            "prediction_index": prediction,
            "prediction_name": class_names[prediction],
            "confidence": probability[prediction],
            "correct": int(prediction == target),
            "models_used": len(present),
        }
        for index, class_name in enumerate(class_names):
            record[f"prob_{index}_{class_name}"] = probability[index]
        for name, model_prediction in model_predictions.items():
            record[f"{name}_prediction_index"] = model_prediction
            record[f"{name}_prediction_name"] = class_names[model_prediction]
        records.append(record)

    if not records:
        raise ValueError("No ensemble records were produced.")
    metrics = _metrics(targets, predictions, class_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with (output_dir / "predictions_detailed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    summary = {
        "mode": "soft_probability_ensemble",
        "key_column": str(key_column),
        "inputs": {table.name: str(table.path.resolve()) for table in tables},
        "weights": weights,
        "records": len(records),
        "class_names": class_names,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_soft_ensemble(
        inputs=args.input,
        weights_text=args.weights,
        output_dir=args.output_dir,
        key_column=args.key_column,
        min_models=args.min_models,
    )
    focus = summary["metrics"]["per_class"][1]
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "records": summary["records"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "class1_f1": focus["f1"],
                "class1_precision": focus["precision"],
                "class1_recall": focus["recall"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

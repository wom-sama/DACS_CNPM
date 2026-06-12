from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from trkh.core.config import load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-image soft teacher probabilities from multiple pretrained "
            "prediction CSVs. Output probabilities are mapped to the target "
            "TRKH class order from data.yaml."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Prediction CSV from export_timm_predictions.py. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional comma list NAME:WEIGHT. Missing names use weight 1.0.",
    )
    parser.add_argument("--min-models", type=int, default=1)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Optional probability smoothing/sharpening before averaging. 1 keeps original probs.",
    )
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--input must have NAME=CSV format, got: {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--input has empty model name: {value!r}")
    path = Path(path_text.strip())
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return name, path


def _parse_weights(value: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Weight item must be NAME:WEIGHT, got: {item!r}")
        name, weight_text = item.split(":", 1)
        weight = float(weight_text)
        if weight <= 0.0:
            raise ValueError(f"Weight must be > 0 for {name!r}, got {weight}")
        weights[name.strip()] = float(weight)
    return weights


def _read_prediction_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"path", "true_name"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV missing columns {sorted(missing)}: {path}")
    return rows


def _read_source_classes(prediction_csv: Path, split: str) -> List[str]:
    candidates = [
        prediction_csv.with_name(f"metrics_{split}.json"),
        prediction_csv.with_name("metrics.json"),
    ]
    for metrics_path in candidates:
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        classes = payload.get("classes")
        if isinstance(classes, list) and classes:
            return [str(value) for value in classes]
    raise FileNotFoundError(
        "Khong tim thay metrics JSON de doc class order cho "
        f"{prediction_csv}. Can metrics_{split}.json hoac metrics.json."
    )


def _row_probabilities(
    row: Mapping[str, str],
    *,
    source_classes: Sequence[str],
    target_classes: Sequence[str],
    temperature: float,
) -> torch.Tensor:
    source_probabilities = []
    for source_index in range(len(source_classes)):
        key = f"prob_{source_index}"
        if key not in row:
            raise ValueError(f"Prediction CSV thieu cot {key}.")
        source_probabilities.append(float(row[key]))
    source = torch.tensor(source_probabilities, dtype=torch.float32).clamp(min=0.0)
    source = source / source.sum().clamp(min=1e-12)
    mapped = torch.tensor(
        [
            float(source[source_classes.index(str(class_name))].item())
            for class_name in target_classes
        ],
        dtype=torch.float32,
    )
    temperature = max(1e-4, float(temperature))
    if abs(temperature - 1.0) > 1e-6:
        mapped = mapped.clamp(min=1e-12).pow(1.0 / temperature)
        mapped = mapped / mapped.sum().clamp(min=1e-12)
    return mapped


def _metrics(records: Sequence[Mapping[str, str]], class_names: Sequence[str]) -> Dict[str, object]:
    matrix: Dict[str, Counter[str]] = {name: Counter() for name in class_names}
    for record in records:
        matrix[str(record["true_name"])][str(record["teacher_pred_name"])] += 1
    per_class = []
    total = 0
    correct = 0
    for class_name in class_names:
        tp = int(matrix[class_name][class_name])
        support = int(sum(matrix[class_name].values()))
        predicted = int(sum(matrix[target][class_name] for target in class_names))
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        total += support
        correct += tp
        per_class.append(
            {
                "class_name": class_name,
                "support": support,
                "predicted_support": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_f1": float(sum(item["f1"] for item in per_class) / max(1, len(per_class))),
        "per_class": per_class,
        "confusion_matrix": [
            [int(matrix[target][prediction]) for prediction in class_names]
            for target in class_names
        ],
    }


def main() -> None:
    args = parse_args()
    data_spec = load_data_spec(args.data)
    target_classes = [str(value) for value in data_spec.class_names]
    named_paths = [_parse_named_path(value) for value in args.input]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names are not allowed: {names}")
    weights = _parse_weights(args.weights)
    min_models = max(1, int(args.min_models))

    rows_by_name: Dict[str, Dict[str, Dict[str, str]]] = {}
    source_classes_by_name: Dict[str, List[str]] = {}
    for name, path in named_paths:
        source_classes = _read_source_classes(path, args.split)
        missing_classes = sorted(set(target_classes) - set(source_classes))
        if missing_classes:
            raise ValueError(
                f"Source classes cua {name} khong khop target: missing={missing_classes}"
            )
        source_classes_by_name[name] = source_classes
        rows = _read_prediction_csv(path)
        rows_by_name[name] = {str(row["path"]): row for row in rows}

    path_counts = Counter()
    for rows in rows_by_name.values():
        path_counts.update(rows.keys())
    selected_paths = sorted(path for path, count in path_counts.items() if count >= min_models)
    if not selected_paths:
        raise ValueError("Khong co sample nao du min_models.")

    output_rows: List[Dict[str, object]] = []
    for path in selected_paths:
        weighted_probability = torch.zeros(len(target_classes), dtype=torch.float32)
        weight_sum = 0.0
        true_names = []
        used_models = []
        for name in names:
            row = rows_by_name[name].get(path)
            if row is None:
                continue
            probability = _row_probabilities(
                row,
                source_classes=source_classes_by_name[name],
                target_classes=target_classes,
                temperature=float(args.temperature),
            )
            weight = float(weights.get(name, 1.0))
            weighted_probability += probability * weight
            weight_sum += weight
            true_names.append(str(row.get("true_name", "")))
            used_models.append(name)
        if weight_sum <= 0.0:
            continue
        weighted_probability = weighted_probability / weight_sum
        weighted_probability = weighted_probability / weighted_probability.sum().clamp(min=1e-12)
        pred_index = int(weighted_probability.argmax().item())
        true_name = next((name for name in true_names if name), "")
        output_row: Dict[str, object] = {
            "path": path,
            "true_name": true_name,
            "teacher_pred_index": pred_index,
            "teacher_pred_name": target_classes[pred_index],
            "teacher_confidence": float(weighted_probability[pred_index].item()),
            "model_count": len(used_models),
            "models": ",".join(used_models),
        }
        for index, value in enumerate(weighted_probability.tolist()):
            output_row[f"prob_{index}"] = float(value)
        output_rows.append(output_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"teacher_probs_{args.split}.csv"
    fieldnames = [
        "path",
        "true_name",
        "teacher_pred_index",
        "teacher_pred_name",
        "teacher_confidence",
        "model_count",
        "models",
        *[f"prob_{index}" for index in range(len(target_classes))],
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    metrics = {
        "split": str(args.split),
        "samples": len(output_rows),
        "target_classes": target_classes,
        "members": [
            {
                "name": name,
                "path": str(path),
                "source_classes": source_classes_by_name[name],
                "weight": float(weights.get(name, 1.0)),
            }
            for name, path in named_paths
        ],
        "min_models": min_models,
        "temperature": float(args.temperature),
        **_metrics(output_rows, target_classes),
    }
    output_json = args.output_dir / f"teacher_probs_{args.split}_metrics.json"
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        {
            "output_csv": str(output_csv),
            "metrics": str(output_json),
            "samples": len(output_rows),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

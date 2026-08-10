from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from trkh.core.config import load_data_spec
from trkh.evaluation.metrics import build_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch fold/val TIMM prediction CSVs into one train-only OOF CSV. "
            "The output path is rewritten to the original source train image and probabilities "
            "are mapped to data.yaml class order."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="FOLD=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="train", choices=("train", "val"))
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected FOLD=CSV, got {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Empty fold name in {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return name, path


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"path", "true_name", "pred_name"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV missing columns {sorted(missing)}: {path}")
    return rows


def _read_source_classes(prediction_csv: Path) -> List[str]:
    candidates = [
        prediction_csv.with_name("metrics_val.json"),
        prediction_csv.with_name("metrics.json"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        values = payload.get("classes")
        if isinstance(values, list) and values:
            return [str(value) for value in values]
    raise FileNotFoundError(f"Cannot infer source class order for {prediction_csv}")


def _probabilities_in_target_order(
    row: Mapping[str, str],
    *,
    source_classes: Sequence[str],
    target_classes: Sequence[str],
) -> List[float]:
    source_probabilities = []
    for source_index in range(len(source_classes)):
        key = f"prob_{source_index}"
        if key not in row:
            raise ValueError(f"Prediction row missing {key}.")
        source_probabilities.append(max(0.0, float(row[key])))
    total = sum(source_probabilities)
    if total <= 0.0:
        source_probabilities = [1.0 / float(len(source_probabilities)) for _ in source_probabilities]
    else:
        source_probabilities = [value / total for value in source_probabilities]
    return [
        float(source_probabilities[source_classes.index(str(class_name))])
        for class_name in target_classes
    ]


def _original_path(source_root: Path, fold_path: Path) -> Path:
    class_name = fold_path.parent.name
    return source_root / class_name / fold_path.name


def stitch_oof_predictions(
    *,
    data_yaml: Path,
    inputs: Sequence[str],
    output_dir: Path,
    source_split: str = "train",
) -> Dict[str, object]:
    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    if data_spec.data_format != "classification_folder":
        raise ValueError("stitch_timm_oof_predictions expects classification_folder data.")
    target_classes = [str(value) for value in data_spec.class_names]
    target_index = {class_name: index for index, class_name in enumerate(target_classes)}
    source_root = data_spec.split_images_dir(source_split)

    output_rows: List[Dict[str, object]] = []
    seen_paths: Dict[str, str] = {}
    fold_counts: Dict[str, int] = {}
    for fold_name, csv_path in [_parse_named_path(value) for value in inputs]:
        rows = _read_rows(csv_path)
        source_classes = _read_source_classes(csv_path)
        if set(source_classes) != set(target_classes):
            raise ValueError(
                f"Class names differ for {csv_path}: source={source_classes}, target={target_classes}"
            )
        fold_counts[fold_name] = len(rows)
        for row in rows:
            fold_image_path = Path(str(row["path"]))
            original = _original_path(source_root, fold_image_path)
            if not original.is_file():
                raise FileNotFoundError(f"Cannot map fold image back to source split: {fold_image_path} -> {original}")
            original_key = str(original.resolve()).lower()
            if original_key in seen_paths:
                raise ValueError(
                    f"Duplicate OOF prediction for {original}; first={seen_paths[original_key]}, second={fold_name}"
                )
            seen_paths[original_key] = fold_name
            true_name = str(row["true_name"])
            pred_name = str(row["pred_name"])
            if true_name not in target_index:
                raise ValueError(f"Unknown true_name {true_name!r} in {csv_path}")
            if pred_name not in target_index:
                raise ValueError(f"Unknown pred_name {pred_name!r} in {csv_path}")
            probabilities = _probabilities_in_target_order(
                row,
                source_classes=source_classes,
                target_classes=target_classes,
            )
            prediction = max(range(len(probabilities)), key=lambda index: probabilities[index])
            output_row: Dict[str, object] = {
                "path": str(original),
                "fold": fold_name,
                "target_index": int(target_index[true_name]),
                "true_name": true_name,
                "prediction_index": int(prediction),
                "pred_name": target_classes[int(prediction)],
                "confidence": float(probabilities[int(prediction)]),
            }
            output_row.update(
                {
                    f"prob_{index}_{class_name}": float(probabilities[index])
                    for index, class_name in enumerate(target_classes)
                }
            )
            output_rows.append(output_row)

    output_rows.sort(key=lambda row: str(row["path"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions_train_oof.csv"
    fieldnames = [
        "path",
        "fold",
        "target_index",
        "true_name",
        "prediction_index",
        "pred_name",
        "confidence",
        *[f"prob_{index}_{class_name}" for index, class_name in enumerate(target_classes)],
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    targets = torch.tensor([int(row["target_index"]) for row in output_rows], dtype=torch.int64)
    predictions = torch.tensor([int(row["prediction_index"]) for row in output_rows], dtype=torch.int64)
    metrics = build_metrics(targets=targets, predictions=predictions, class_names=target_classes)
    summary = {
        "data": str(data_yaml),
        "source_split": str(source_split),
        "output_csv": str(output_csv),
        "rows": int(len(output_rows)),
        "fold_counts": fold_counts,
        "class_names": target_classes,
        "metrics": metrics,
        "leakage_note": "Rows are stitched from fold validation predictions and mapped back to source train paths.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = stitch_oof_predictions(
        data_yaml=args.data,
        inputs=args.input,
        output_dir=args.output_dir,
        source_split=str(args.source_split),
    )
    print(
        json.dumps(
            {
                "output_csv": summary["output_csv"],
                "rows": summary["rows"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "class1_f1": summary["metrics"]["per_class"][1]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

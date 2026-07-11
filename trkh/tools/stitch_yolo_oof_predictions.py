from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from trkh.core.config import load_data_spec
from trkh.evaluation.metrics import build_metrics


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SourceObject:
    sample_index: int
    image_path: Path
    label_path: Path
    source_stem: str
    object_index: int
    target_index: int
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch fold/val YOLO object-crop prediction CSVs into one train-only OOF CSV. "
            "Rows are mapped by source_stem + object_index to avoid collapsing multi-object images."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="FOLD=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="train", choices=("train", "val"))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing source objects, useful only for smoke/debug stitching.",
    )
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
    required = {"target_index", "prediction_index"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV missing columns {sorted(missing)}: {path}")
    return rows


def _index_image_paths(images_dir: Path) -> Dict[str, Path]:
    image_paths: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.setdefault(path.stem, path)
    return image_paths


def _parse_source_objects(data_yaml: Path, *, source_split: str) -> tuple[List[SourceObject], List[str]]:
    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    if data_spec.data_format == "classification_folder":
        raise ValueError("stitch_yolo_oof_predictions expects YOLO-format data.")
    class_names = [str(value) for value in data_spec.class_names]
    images_dir = data_spec.split_images_dir(source_split)
    labels_dir = data_spec.split_labels_dir(source_split)
    image_paths = _index_image_paths(images_dir)
    objects: List[SourceObject] = []

    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda item: item.name):
        image_path = image_paths.get(label_path.stem)
        if image_path is None:
            continue
        for object_index, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                label = int(parts[0])
                bbox = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            except ValueError:
                continue
            if label < 0 or label >= len(class_names):
                continue
            if any(value < 0.0 or value > 1.0 for value in bbox) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue
            objects.append(
                SourceObject(
                    sample_index=len(objects),
                    image_path=image_path,
                    label_path=label_path,
                    source_stem=label_path.stem,
                    object_index=int(object_index),
                    target_index=int(label),
                    bbox=bbox,
                )
            )
    if not objects:
        raise ValueError(f"No source YOLO objects found in {labels_dir}")
    return objects, class_names


def _object_key(source_stem: str, object_index: int) -> str:
    return f"{source_stem}#{int(object_index)}"


def _source_stem_from_row(row: Mapping[str, str]) -> str:
    value = str(row.get("source_stem", "") or "").strip()
    if value:
        return value
    image_path = str(row.get("image_path", "") or "").strip()
    if image_path:
        return Path(image_path).stem
    raise ValueError("Prediction row lacks source_stem/image_path.")


def _int_field(row: Mapping[str, str], key: str) -> int:
    value = str(row.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"Prediction row missing {key}.")
    return int(float(value))


def _probabilities_in_target_order(row: Mapping[str, str], *, class_names: Sequence[str]) -> List[float]:
    probabilities: List[float] = []
    for class_index, class_name in enumerate(class_names):
        named_key = f"prob_{class_index}_{class_name}"
        plain_key = f"prob_{class_index}"
        if named_key in row and str(row[named_key]).strip() != "":
            value = float(row[named_key])
        elif plain_key in row and str(row[plain_key]).strip() != "":
            value = float(row[plain_key])
        else:
            raise ValueError(f"Prediction row missing probability column {named_key} or {plain_key}.")
        probabilities.append(max(0.0, value))
    total = sum(probabilities)
    if total <= 0.0:
        return [1.0 / float(len(class_names)) for _ in class_names]
    return [float(value / total) for value in probabilities]


def stitch_oof_predictions(
    *,
    data_yaml: Path,
    inputs: Sequence[str],
    output_dir: Path,
    source_split: str = "train",
    allow_partial: bool = False,
) -> Dict[str, object]:
    source_objects, class_names = _parse_source_objects(data_yaml, source_split=source_split)
    object_by_key = {
        _object_key(obj.source_stem, obj.object_index): obj
        for obj in source_objects
    }
    if len(object_by_key) != len(source_objects):
        raise ValueError("Source YOLO objects produced duplicate source_stem/object_index keys.")

    output_rows: List[Dict[str, object]] = []
    seen_keys: Dict[str, str] = {}
    fold_counts: Dict[str, int] = {}
    for fold_name, csv_path in [_parse_named_path(value) for value in inputs]:
        rows = _read_rows(csv_path)
        fold_counts[fold_name] = len(rows)
        for row in rows:
            source_stem = _source_stem_from_row(row)
            object_index = _int_field(row, "object_index")
            key = _object_key(source_stem, object_index)
            source_object = object_by_key.get(key)
            if source_object is None:
                raise KeyError(f"Cannot map fold prediction to source object: {key} from {csv_path}")
            if key in seen_keys:
                raise ValueError(
                    f"Duplicate OOF prediction for {key}; first={seen_keys[key]}, second={fold_name}"
                )
            seen_keys[key] = fold_name
            target_index = _int_field(row, "target_index")
            if target_index != int(source_object.target_index):
                raise ValueError(
                    f"Target mismatch for {key}: csv={target_index}, source={source_object.target_index}"
                )
            probabilities = _probabilities_in_target_order(row, class_names=class_names)
            prediction_index = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))
            output_row: Dict[str, object] = {
                "sample_index": int(source_object.sample_index),
                "source_stem": source_object.source_stem,
                "object_index": int(source_object.object_index),
                "image_path": str(source_object.image_path),
                "label_path": str(source_object.label_path),
                "fold": fold_name,
                "target_index": int(source_object.target_index),
                "true_name": class_names[int(source_object.target_index)],
                "prediction_index": int(prediction_index),
                "pred_name": class_names[int(prediction_index)],
                "confidence": float(probabilities[int(prediction_index)]),
                "correct": int(prediction_index == int(source_object.target_index)),
                "bbox_x": float(source_object.bbox[0]),
                "bbox_y": float(source_object.bbox[1]),
                "bbox_w": float(source_object.bbox[2]),
                "bbox_h": float(source_object.bbox[3]),
            }
            output_row.update(
                {
                    f"prob_{index}_{class_name}": float(probabilities[index])
                    for index, class_name in enumerate(class_names)
                }
            )
            output_rows.append(output_row)

    missing_keys = sorted(set(object_by_key).difference(seen_keys))
    if missing_keys and not allow_partial:
        raise ValueError(
            f"Missing {len(missing_keys)} source objects in OOF predictions. "
            f"First missing keys: {missing_keys[:10]}"
        )

    output_rows.sort(key=lambda row: int(row["sample_index"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions_train_oof.csv"
    fieldnames = [
        "sample_index",
        "source_stem",
        "object_index",
        "image_path",
        "label_path",
        "fold",
        "target_index",
        "true_name",
        "prediction_index",
        "pred_name",
        "confidence",
        "correct",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        *[f"prob_{index}_{class_name}" for index, class_name in enumerate(class_names)],
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    targets = torch.tensor([int(row["target_index"]) for row in output_rows], dtype=torch.int64)
    predictions = torch.tensor([int(row["prediction_index"]) for row in output_rows], dtype=torch.int64)
    metrics = build_metrics(targets=targets, predictions=predictions, class_names=class_names)
    summary = {
        "data": str(data_yaml),
        "source_split": str(source_split),
        "output_csv": str(output_csv),
        "rows": int(len(output_rows)),
        "source_object_count": int(len(source_objects)),
        "missing_source_object_count": int(len(missing_keys)),
        "fold_counts": fold_counts,
        "class_names": class_names,
        "metrics": metrics,
        "leakage_note": (
            "Rows are stitched from fold validation predictions and keyed by source_stem/object_index. "
            "The output is train-only OOF and must not include final val/test predictions for router training."
        ),
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
        allow_partial=bool(args.allow_partial),
    )
    print(
        json.dumps(
            {
                "output_csv": summary["output_csv"],
                "rows": summary["rows"],
                "source_object_count": summary["source_object_count"],
                "missing_source_object_count": summary["missing_source_object_count"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "class1_f1": summary["metrics"]["per_class"][1]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

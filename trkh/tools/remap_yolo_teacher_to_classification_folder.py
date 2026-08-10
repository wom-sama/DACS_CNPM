from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from trkh.core.config import load_data_spec
from trkh.data.dataset import IMAGE_EXTENSIONS


CLASSIFICATION_CROP_NAME = re.compile(r"^(?P<source>.+)_box(?P<box>\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remap a YOLO object-level teacher CSV to classification_folder sample order. "
            "The source CSV is expected to be in YOLO dataset order with duplicate source "
            "image paths for multi-object images. The output uses class_f crop paths and "
            "sample_index values matching ClassificationFolderDataset."
        )
    )
    parser.add_argument("--classification-data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--yolo-teacher-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--strict", action="store_true", default=False)
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Teacher CSV has no header: {path}")
        return [dict(row) for row in reader]


def _probability_columns(row: Mapping[str, str], class_count: int) -> List[str]:
    columns: List[str] = []
    missing: List[str] = []
    for index in range(class_count):
        exact = f"prob_{index}"
        if exact in row:
            columns.append(exact)
            continue
        prefix = f"prob_{index}_"
        candidates = sorted(str(column) for column in row if str(column).startswith(prefix))
        if candidates:
            columns.append(candidates[0])
        else:
            missing.append(exact)
    if missing:
        raise ValueError(f"Teacher CSV missing probability columns: {missing}")
    return columns


def _source_id_from_teacher_row(row: Mapping[str, str]) -> str:
    path_text = str(row.get("path", "") or row.get("image_path", "")).strip()
    if not path_text:
        raise ValueError("Teacher row has neither path nor image_path")
    return Path(path_text).stem


def _build_yolo_teacher_index(
    rows: Sequence[Mapping[str, str]],
    *,
    class_count: int,
) -> Tuple[Dict[Tuple[str, int], Mapping[str, str]], Dict[str, object]]:
    if not rows:
        raise ValueError("Teacher CSV is empty")
    _probability_columns(rows[0], class_count)

    occurrence_counter: Counter[str] = Counter()
    indexed: Dict[Tuple[str, int], Mapping[str, str]] = {}
    duplicate_keys = 0
    for row in rows:
        source_id = _source_id_from_teacher_row(row)
        object_index = int(occurrence_counter[source_id])
        key = (source_id, object_index)
        if key in indexed:
            duplicate_keys += 1
        indexed[key] = row
        occurrence_counter[source_id] += 1

    return indexed, {
        "teacher_rows": len(rows),
        "teacher_source_images": len(occurrence_counter),
        "teacher_multi_object_sources": int(sum(1 for count in occurrence_counter.values() if count > 1)),
        "teacher_max_objects_per_source": int(max(occurrence_counter.values()) if occurrence_counter else 0),
        "duplicate_source_box_keys": int(duplicate_keys),
    }


def _classification_samples(
    *,
    data_yaml: Path,
    split: str,
    class_name_mode: str,
    expected_num_classes: int,
) -> Tuple[List[Tuple[Path, int, str, int]], List[str]]:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes or None,
    )
    if data_spec.data_format != "classification_folder":
        raise ValueError("--classification-data must point to classification_folder data.yaml")

    samples: List[Tuple[Path, int, str, int]] = []
    root_dir = data_spec.split_images_dir(split)
    for class_index, class_name in enumerate(data_spec.class_names):
        class_dir = root_dir / str(class_name)
        if not class_dir.is_dir():
            continue
        image_paths = [
            path
            for path in sorted(class_dir.rglob("*"), key=lambda item: str(item).lower())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in image_paths:
            match = CLASSIFICATION_CROP_NAME.match(image_path.stem)
            if match is None:
                raise ValueError(f"Cannot parse classification crop name: {image_path.name}")
            source_id = str(match.group("source"))
            box_index = int(match.group("box"))
            samples.append((image_path, int(class_index), source_id, box_index))
    return samples, list(data_spec.class_names)


def _safe_probability(row: Mapping[str, str], column: str) -> float:
    value = float(row[column])
    if value < 0.0:
        return 0.0
    return value


def main() -> None:
    args = parse_args()
    samples, class_names = _classification_samples(
        data_yaml=args.classification_data,
        split=args.split,
        class_name_mode=args.class_name_mode,
        expected_num_classes=int(args.expected_num_classes),
    )
    class_count = len(class_names)
    teacher_rows = _read_rows(args.yolo_teacher_csv)
    teacher_index, teacher_summary = _build_yolo_teacher_index(
        teacher_rows,
        class_count=class_count,
    )
    source_probability_columns = _probability_columns(teacher_rows[0], class_count)
    output_probability_columns = [f"prob_{index}" for index in range(class_count)]

    missing: List[Dict[str, object]] = []
    outputs: List[Dict[str, object]] = []
    prediction_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    correct = 0
    source_box_hits: Counter[str] = Counter()
    for sample_index, (image_path, class_index, source_id, box_index) in enumerate(samples):
        target_counts[int(class_index)] += 1
        teacher_row = teacher_index.get((source_id, int(box_index)))
        if teacher_row is None:
            missing.append(
                {
                    "sample_index": int(sample_index),
                    "path": str(image_path),
                    "source_id": source_id,
                    "box_index": int(box_index),
                }
            )
            continue

        probabilities = [
            _safe_probability(teacher_row, column)
            for column in source_probability_columns
        ]
        total = sum(probabilities)
        if total <= 0.0:
            probabilities = [1.0 / float(class_count) for _ in range(class_count)]
        else:
            probabilities = [float(value / total) for value in probabilities]
        prediction = int(max(range(class_count), key=lambda index: probabilities[index]))
        prediction_counts[prediction] += 1
        correct += int(prediction == int(class_index))
        source_box_hits[source_id] += 1
        target_name = str(class_names[int(class_index)])
        prediction_name = str(class_names[int(prediction)])
        confidence = float(probabilities[int(prediction)])

        row: Dict[str, object] = {
            "sample_index": int(sample_index),
            "image_path": str(image_path),
            "path": str(image_path),
            "target_index": int(class_index),
            "target_name": target_name,
            "true_name": target_name,
            "prediction_index": int(prediction),
            "prediction_name": prediction_name,
            "pred_name": prediction_name,
            "confidence": f"{confidence:.10f}",
        }
        for class_index_value, probability in enumerate(probabilities):
            row[f"prob_{class_index_value}"] = f"{probability:.10f}"
        outputs.append(row)

    if missing and bool(args.strict):
        raise ValueError(f"Missing {len(missing)} teacher mappings; first missing: {missing[:5]}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "path",
        "target_index",
        "target_name",
        "true_name",
        "prediction_index",
        "prediction_name",
        "pred_name",
        "confidence",
    ] + output_probability_columns
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in outputs:
            writer.writerow(row)

    summary = {
        "classification_data": str(args.classification_data),
        "split": str(args.split),
        "yolo_teacher_csv": str(args.yolo_teacher_csv),
        "output_csv": str(args.output_csv),
        "class_names": class_names,
        "classification_samples": int(len(samples)),
        "mapped_rows": int(len(outputs)),
        "missing_rows": int(len(missing)),
        "teacher_label_agreement": float(correct / max(1, len(outputs))),
        "target_counts": {str(index): int(target_counts[index]) for index in range(class_count)},
        "teacher_prediction_counts": {
            str(index): int(prediction_counts[index]) for index in range(class_count)
        },
        "source_images_matched": int(len(source_box_hits)),
        "missing_examples": missing[:10],
        **teacher_summary,
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

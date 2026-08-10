from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from trkh.core.config import load_data_spec
from trkh.data.dataset import MangoYOLOCropDataset, PairedViewTrainDataset


CLASSIFICATION_CROP_NAME = re.compile(r"^(?P<source>.+)_box(?P<box>\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remap classification_folder teacher probabilities to YOLO object-crop "
            "sample_index order. This is intended for class_f teachers such as AIDT "
            "when training a yolo_f TRKH student."
        )
    )
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--source-class-names", default="")
    parser.add_argument("--source-metrics-json", type=Path, default=None)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--strict", action="store_true", default=False)
    return parser.parse_args()


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Teacher CSV not found: {path}")
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Teacher CSV has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _parse_source_class_names(value: str) -> List[str]:
    names = [part.strip() for part in str(value or "").split(",") if part.strip()]
    return names


def _source_class_names_from_metrics(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"source metrics JSON has no classes list: {path}")
    return [str(name) for name in classes]


def _probability_columns(fieldnames: Sequence[str], class_count: int) -> List[str]:
    columns: List[str] = []
    missing: List[str] = []
    for index in range(int(class_count)):
        exact = f"prob_{index}"
        if exact in fieldnames:
            columns.append(exact)
            continue
        prefix = f"prob_{index}_"
        candidates = sorted(str(name) for name in fieldnames if str(name).startswith(prefix))
        if candidates:
            columns.append(candidates[0])
            continue
        missing.append(exact)
    if missing:
        raise ValueError(f"Teacher CSV missing probability columns: {missing}")
    return columns


def _infer_source_class_names_from_columns(columns: Sequence[str]) -> List[str]:
    names: List[str] = []
    for index, column in enumerate(columns):
        prefix = f"prob_{index}_"
        if not str(column).startswith(prefix):
            return []
        names.append(str(column)[len(prefix) :])
    return names


def _resolve_source_class_names(
    *,
    fieldnames: Sequence[str],
    source_class_names: str,
    source_metrics_json: Optional[Path],
    class_count: int,
) -> List[str]:
    explicit = _parse_source_class_names(source_class_names)
    from_metrics = _source_class_names_from_metrics(source_metrics_json)
    prob_columns = _probability_columns(fieldnames, class_count)
    inferred = _infer_source_class_names_from_columns(prob_columns)
    names = explicit or from_metrics or inferred
    if len(names) != int(class_count):
        raise ValueError(
            "Cannot resolve teacher source class names. Pass --source-class-names "
            "or --source-metrics-json when probability columns are only prob_0..N."
        )
    return [str(name) for name in names]


def _classification_key(path_text: str) -> Tuple[str, int]:
    stem = Path(str(path_text)).stem
    match = CLASSIFICATION_CROP_NAME.match(stem)
    if match is None:
        raise ValueError(f"Cannot parse classification crop filename: {path_text}")
    return str(match.group("source")), int(match.group("box"))


def _teacher_probabilities_by_key(
    rows: Sequence[Mapping[str, str]],
    *,
    fieldnames: Sequence[str],
    source_class_names: Sequence[str],
    target_class_names: Sequence[str],
) -> Tuple[Dict[Tuple[str, int], List[float]], Dict[str, object]]:
    class_count = len(target_class_names)
    source_columns = _probability_columns(fieldnames, class_count)
    target_index_by_name = {str(name): int(index) for index, name in enumerate(target_class_names)}
    source_to_target: List[int] = []
    for source_name in source_class_names:
        if str(source_name) not in target_index_by_name:
            raise ValueError(f"Teacher class not found in YOLO data names: {source_name}")
        source_to_target.append(target_index_by_name[str(source_name)])

    indexed: Dict[Tuple[str, int], List[float]] = {}
    duplicate_keys = 0
    teacher_prediction_counts: Counter[int] = Counter()
    for row in rows:
        path_text = str(row.get("path", "") or row.get("image_path", "") or "").strip()
        if not path_text:
            continue
        key = _classification_key(path_text)
        probabilities = [0.0 for _ in range(class_count)]
        for source_index, target_index in enumerate(source_to_target):
            value = max(0.0, float(row[source_columns[source_index]]))
            probabilities[int(target_index)] = value
        total = sum(probabilities)
        if total <= 0.0:
            probabilities = [1.0 / float(class_count) for _ in range(class_count)]
        else:
            probabilities = [float(value / total) for value in probabilities]
        if key in indexed:
            duplicate_keys += 1
        indexed[key] = probabilities
        prediction = int(max(range(class_count), key=lambda index: probabilities[index]))
        teacher_prediction_counts[prediction] += 1

    return indexed, {
        "teacher_rows": int(len(rows)),
        "teacher_key_count": int(len(indexed)),
        "duplicate_teacher_keys": int(duplicate_keys),
        "teacher_prediction_counts": {
            str(index): int(teacher_prediction_counts[index]) for index in range(class_count)
        },
        "source_class_names": [str(name) for name in source_class_names],
        "target_class_names": [str(name) for name in target_class_names],
    }


def _build_yolo_dataset(
    *,
    yolo_data: Path,
    split: str,
    class_name_mode: str,
    expected_num_classes: int,
):
    data_spec = load_data_spec(
        yolo_data,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes or None,
    )
    if data_spec.data_format != "yolo":
        raise ValueError("--yolo-data must point to YOLO data.yaml")
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=None,
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        class_aware_augmentation=False,
        classification_bbox_metadata=True,
    )
    return dataset, list(data_spec.class_names)


def remap_classification_teacher_to_yolo(
    *,
    teacher_csv: Path,
    yolo_data: Path,
    split: str,
    output_csv: Path,
    source_class_names: str = "",
    source_metrics_json: Optional[Path] = None,
    class_name_mode: str = "raw",
    expected_num_classes: int = 5,
    strict: bool = False,
) -> Dict[str, object]:
    dataset, target_class_names = _build_yolo_dataset(
        yolo_data=yolo_data,
        split=split,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes,
    )
    rows, fieldnames = _read_csv(teacher_csv)
    source_names = _resolve_source_class_names(
        fieldnames=fieldnames,
        source_class_names=source_class_names,
        source_metrics_json=source_metrics_json,
        class_count=len(target_class_names),
    )
    teacher_by_key, teacher_summary = _teacher_probabilities_by_key(
        rows,
        fieldnames=fieldnames,
        source_class_names=source_names,
        target_class_names=target_class_names,
    )

    paths = [Path(path) for path in dataset.sample_paths()]
    labels = [int(label) for label in dataset.labels()]
    outputs: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    correct = 0
    target_counts: Counter[int] = Counter()
    prediction_counts: Counter[int] = Counter()
    for sample_index in range(len(dataset)):
        source_id, object_index, fallback_label = PairedViewTrainDataset._sample_key(dataset, sample_index)
        target_index = int(labels[sample_index]) if sample_index < len(labels) else int(fallback_label)
        target_counts[target_index] += 1
        probabilities = teacher_by_key.get((str(source_id), int(object_index)))
        if probabilities is None:
            missing.append(
                {
                    "sample_index": int(sample_index),
                    "path": str(paths[sample_index]) if sample_index < len(paths) else "",
                    "source_stem": str(source_id),
                    "object_index": int(object_index),
                    "target_index": int(target_index),
                }
            )
            continue
        prediction = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))
        prediction_counts[prediction] += 1
        correct += int(prediction == int(target_index))
        target_name = str(target_class_names[target_index])
        prediction_name = str(target_class_names[prediction])
        path = paths[sample_index] if sample_index < len(paths) else Path(str(source_id))
        row: Dict[str, object] = {
            "sample_index": int(sample_index),
            "path": str(path),
            "image_path": str(path),
            "source_stem": str(source_id),
            "object_index": int(object_index),
            "target_index": int(target_index),
            "target_name": target_name,
            "true_name": target_name,
            "prediction_index": int(prediction),
            "prediction_name": prediction_name,
            "pred_name": prediction_name,
            "confidence": f"{float(probabilities[prediction]):.10f}",
        }
        for class_index, probability in enumerate(probabilities):
            row[f"prob_{class_index}"] = f"{float(probability):.10f}"
            row[f"prob_{class_index}_{target_class_names[class_index]}"] = f"{float(probability):.10f}"
        outputs.append(row)

    if missing and strict:
        raise ValueError(f"Missing {len(missing)} teacher mappings; preview={missing[:5]}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    probability_fields = [f"prob_{index}" for index in range(len(target_class_names))]
    named_probability_fields = [
        f"prob_{index}_{target_class_names[index]}" for index in range(len(target_class_names))
    ]
    fieldnames_out = [
        "sample_index",
        "path",
        "image_path",
        "source_stem",
        "object_index",
        "target_index",
        "target_name",
        "true_name",
        "prediction_index",
        "prediction_name",
        "pred_name",
        "confidence",
    ] + probability_fields + named_probability_fields
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(outputs)

    summary = {
        "teacher_csv": str(Path(teacher_csv).resolve()),
        "yolo_data": str(Path(yolo_data).resolve()),
        "split": str(split),
        "output_csv": str(output_csv.resolve()),
        "yolo_samples": int(len(dataset)),
        "mapped_rows": int(len(outputs)),
        "missing_rows": int(len(missing)),
        "teacher_label_agreement": float(correct / max(1, len(outputs))),
        "target_counts": {str(index): int(target_counts[index]) for index in range(len(target_class_names))},
        "mapped_prediction_counts": {
            str(index): int(prediction_counts[index]) for index in range(len(target_class_names))
        },
        "missing_examples": missing[:10],
        **teacher_summary,
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = remap_classification_teacher_to_yolo(
        teacher_csv=args.teacher_csv,
        yolo_data=args.yolo_data,
        split=args.split,
        output_csv=args.output_csv,
        source_class_names=args.source_class_names,
        source_metrics_json=args.source_metrics_json,
        class_name_mode=args.class_name_mode,
        expected_num_classes=int(args.expected_num_classes),
        strict=bool(args.strict),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

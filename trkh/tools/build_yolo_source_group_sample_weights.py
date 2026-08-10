from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trkh.core.config import load_data_spec


ObjectRow = Tuple[int, Tuple[float, float, float, float]]


def _resolve_data_yaml(data_path: Path) -> Path:
    resolved = data_path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "data.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"data.yaml not found: {resolved}")
    return resolved


def _parse_yolo_objects(label_path: Path, num_classes: int) -> Tuple[List[ObjectRow], int]:
    objects: List[ObjectRow] = []
    invalid_rows = 0
    for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid_rows += 1
            continue
        try:
            label = int(parts[0])
            bbox = tuple(float(value) for value in parts[1:5])
        except ValueError:
            invalid_rows += 1
            continue
        if label < 0 or label >= int(num_classes):
            invalid_rows += 1
            continue
        if any(value < 0.0 or value > 1.0 for value in bbox) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            invalid_rows += 1
            continue
        objects.append((label, bbox))  # type: ignore[arg-type]
    return objects, invalid_rows


def _find_image_path(images_dir: Path, stem: str) -> str:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _join_labels(labels: Iterable[int]) -> str:
    return " ".join(str(int(label)) for label in sorted(set(int(label) for label in labels)))


def _weight_for_object(
    *,
    target_index: int,
    object_count: int,
    unique_label_count: int,
    mixed_label_weight: float,
    mixed_label_class1_weight: float,
    same_label_multi_weight: float,
    same_label_class4_weight: float,
) -> Tuple[float, str]:
    if object_count <= 1:
        return 1.0, "single_object"
    if unique_label_count > 1:
        if int(target_index) == 1:
            return float(mixed_label_class1_weight), "mixed_label_class1_protected"
        return float(mixed_label_weight), "mixed_label_source"
    if int(target_index) == 4:
        return float(same_label_class4_weight), "same_label_multi_class4"
    return float(same_label_multi_weight), "same_label_multi_source"


def build_manifest(
    *,
    data: Path,
    output_dir: Path,
    split: str = "train",
    mixed_label_weight: float = 0.65,
    mixed_label_class1_weight: float = 0.85,
    same_label_multi_weight: float = 0.85,
    same_label_class4_weight: float = 0.75,
    default_weight: float = 1.0,
    write_default_rows: bool = False,
) -> Dict[str, object]:
    split_name = str(split or "train").strip().lower()
    if split_name != "train":
        raise ValueError("Source-group sample weights must be built from train split only.")
    data_yaml = _resolve_data_yaml(data)
    data_spec = load_data_spec(data_yaml, class_name_mode="raw")
    labels_dir = data_spec.split_labels_dir(split_name)
    images_dir = data_spec.split_images_dir(split_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_weight = max(0.0, float(default_weight))
    weights = {
        "mixed_label_weight": max(0.0, float(mixed_label_weight)),
        "mixed_label_class1_weight": max(0.0, float(mixed_label_class1_weight)),
        "same_label_multi_weight": max(0.0, float(same_label_multi_weight)),
        "same_label_class4_weight": max(0.0, float(same_label_class4_weight)),
    }

    manifest_path = output_dir / "source_group_sample_weights_train_only.csv"
    rows: List[Dict[str, object]] = []
    sample_index = 0
    invalid_rows = 0
    objects_total = 0
    weighted_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    observed_weights: List[float] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        objects, file_invalid_rows = _parse_yolo_objects(label_path, data_spec.num_classes)
        invalid_rows += int(file_invalid_rows)
        if not objects:
            continue
        source_labels = [int(label) for label, _ in objects]
        object_count = len(objects)
        unique_label_count = len(set(source_labels))
        unique_labels = _join_labels(source_labels)
        image_path = _find_image_path(images_dir, label_path.stem)
        for object_index, (target_index, _bbox) in enumerate(objects):
            weight, reason = _weight_for_object(
                target_index=int(target_index),
                object_count=int(object_count),
                unique_label_count=int(unique_label_count),
                mixed_label_weight=weights["mixed_label_weight"],
                mixed_label_class1_weight=weights["mixed_label_class1_weight"],
                same_label_multi_weight=weights["same_label_multi_weight"],
                same_label_class4_weight=weights["same_label_class4_weight"],
            )
            objects_total += 1
            class_counter[str(int(target_index))] += 1
            observed_weights.append(float(weight))
            if bool(write_default_rows) or abs(float(weight) - default_weight) > 1e-12:
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "image_path": image_path,
                        "label_path": str(label_path.resolve()),
                        "target_index": int(target_index),
                        "object_index": int(object_index),
                        "source_object_count": int(object_count),
                        "source_unique_label_count": int(unique_label_count),
                        "source_unique_labels": unique_labels,
                        "sample_weight": f"{float(weight):.10g}",
                        "reason": reason,
                    }
                )
                weighted_counter[str(int(target_index))] += 1
                reason_counter[reason] += 1
            sample_index += 1

    fieldnames = [
        "sample_index",
        "image_path",
        "label_path",
        "target_index",
        "object_index",
        "source_object_count",
        "source_unique_label_count",
        "source_unique_labels",
        "sample_weight",
        "reason",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data_yaml": str(data_yaml.resolve()),
        "split": split_name,
        "output_manifest": str(manifest_path.resolve()),
        "objects_total": int(objects_total),
        "sample_indices_total": int(sample_index),
        "written_rows": int(len(rows)),
        "invalid_rows": int(invalid_rows),
        "default_weight": float(default_weight),
        "weights": weights,
        "write_default_rows": bool(write_default_rows),
        "class_counts": dict(sorted(class_counter.items())),
        "weighted_class_counts": dict(sorted(weighted_counter.items())),
        "reason_counts": dict(sorted(reason_counter.items())),
        "mean_source_group_weight": float(mean(observed_weights)) if observed_weights else 0.0,
        "min_source_group_weight": float(min(observed_weights)) if observed_weights else 0.0,
        "max_source_group_weight": float(max(observed_weights)) if observed_weights else 0.0,
        "leakage_guard": "train split only; sample_index order mirrors MangoYOLOCropDataset classification_object_crops",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not rows:
        raise ValueError("No sample-weight rows were written; check weights or write_default_rows.")
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-only sample weights from YOLO same-source object groups."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--mixed-label-weight", type=float, default=0.65)
    parser.add_argument("--mixed-label-class1-weight", type=float, default=0.85)
    parser.add_argument("--same-label-multi-weight", type=float, default=0.85)
    parser.add_argument("--same-label-class4-weight", type=float, default=0.75)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--write-default-rows", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_manifest(
        data=Path(args.data),
        output_dir=Path(args.output_dir),
        split=str(args.split),
        mixed_label_weight=float(args.mixed_label_weight),
        mixed_label_class1_weight=float(args.mixed_label_class1_weight),
        same_label_multi_weight=float(args.same_label_multi_weight),
        same_label_class4_weight=float(args.same_label_class4_weight),
        default_weight=float(args.default_weight),
        write_default_rows=bool(args.write_default_rows),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

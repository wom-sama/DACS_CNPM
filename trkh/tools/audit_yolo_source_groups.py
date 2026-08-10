from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from trkh.core.config import load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit YOLO label-file source groups for same-image consistency risk. "
            "This diagnostic reads labels only and never modifies the dataset."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml or dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max-examples", type=int, default=80)
    return parser.parse_args()


def _resolve_data_yaml(data_path: Path) -> Path:
    resolved = data_path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "data.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"data.yaml not found: {resolved}")
    return resolved


def _parse_yolo_labels(label_path: Path, num_classes: int) -> Tuple[List[int], int]:
    labels: List[int] = []
    invalid_rows = 0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            invalid_rows += 1
            continue
        try:
            label = int(float(parts[0]))
        except ValueError:
            invalid_rows += 1
            continue
        if label < 0 or label >= int(num_classes):
            invalid_rows += 1
            continue
        labels.append(label)
    return labels, invalid_rows


def _counter_to_str_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: item[0])}


def _label_counts_text(labels: Sequence[int]) -> str:
    return ";".join(f"{label}:{count}" for label, count in sorted(Counter(labels).items()))


def _example_row(split: str, label_path: Path, labels: Sequence[int]) -> Dict[str, object]:
    unique_labels = sorted(set(int(label) for label in labels))
    return {
        "split": split,
        "source_stem": label_path.stem,
        "label_path": str(label_path),
        "object_count": int(len(labels)),
        "unique_label_count": int(len(unique_labels)),
        "unique_labels": " ".join(str(label) for label in unique_labels),
        "labels": " ".join(str(int(label)) for label in labels),
        "class_counts": _label_counts_text(labels),
    }


def _iter_requested_splits(requested_splits: Iterable[str], has_test_split: bool) -> List[str]:
    splits: List[str] = []
    for raw_split in requested_splits:
        split = raw_split.strip().lower()
        if not split:
            continue
        if split == "test" and not has_test_split:
            continue
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {raw_split}")
        splits.append(split)
    return splits


def _audit_split(
    *,
    split: str,
    labels_dir: Path,
    num_classes: int,
    max_examples: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    label_files = sorted(labels_dir.glob("*.txt"))
    class_counts: Counter = Counter()
    multi_object_class_counts: Counter = Counter()
    same_label_multi_object_class_counts: Counter = Counter()
    mixed_label_object_class_counts: Counter = Counter()
    group_size_counts: Counter = Counter()
    mixed_label_pair_counts: Counter = Counter()
    invalid_rows = 0
    empty_label_files = 0
    object_count = 0
    single_object_images = 0
    multi_object_images = 0
    same_label_multi_object_images = 0
    mixed_label_images = 0
    class1_images = 0
    class1_multi_object_images = 0
    class1_same_label_multi_object_images = 0
    class1_mixed_label_images = 0
    mixed_examples: List[Dict[str, object]] = []
    multi_examples: List[Dict[str, object]] = []

    for label_path in label_files:
        labels, file_invalid_rows = _parse_yolo_labels(label_path, num_classes)
        invalid_rows += int(file_invalid_rows)
        if not labels:
            empty_label_files += 1
            continue

        label_counter = Counter(labels)
        unique_labels = sorted(label_counter)
        group_size = len(labels)
        object_count += group_size
        class_counts.update(labels)
        group_size_counts.update([group_size])

        if 1 in label_counter:
            class1_images += 1

        if group_size == 1:
            single_object_images += 1
            continue

        multi_object_images += 1
        multi_object_class_counts.update(labels)
        if 1 in label_counter:
            class1_multi_object_images += 1
        if len(multi_examples) < max_examples:
            multi_examples.append(_example_row(split, label_path, labels))

        if len(unique_labels) == 1:
            same_label_multi_object_images += 1
            same_label_multi_object_class_counts.update(labels)
            if unique_labels[0] == 1:
                class1_same_label_multi_object_images += 1
            continue

        mixed_label_images += 1
        mixed_label_object_class_counts.update(labels)
        for left, right in combinations(unique_labels, 2):
            mixed_label_pair_counts.update([f"{left}-{right}"])
        if 1 in label_counter:
            class1_mixed_label_images += 1
        if len(mixed_examples) < max_examples:
            mixed_examples.append(_example_row(split, label_path, labels))

    images_with_objects = single_object_images + multi_object_images
    summary = {
        "labels_dir": str(labels_dir),
        "label_files": int(len(label_files)),
        "empty_label_files": int(empty_label_files),
        "invalid_rows": int(invalid_rows),
        "images_with_objects": int(images_with_objects),
        "objects": int(object_count),
        "class_counts": _counter_to_str_dict(class_counts),
        "single_object_images": int(single_object_images),
        "multi_object_images": int(multi_object_images),
        "same_label_multi_object_images": int(same_label_multi_object_images),
        "mixed_label_images": int(mixed_label_images),
        "group_size_counts": _counter_to_str_dict(group_size_counts),
        "multi_object_class_counts": _counter_to_str_dict(multi_object_class_counts),
        "same_label_multi_object_class_counts": _counter_to_str_dict(same_label_multi_object_class_counts),
        "mixed_label_object_class_counts": _counter_to_str_dict(mixed_label_object_class_counts),
        "mixed_label_pair_counts": _counter_to_str_dict(mixed_label_pair_counts),
        "class1_images": int(class1_images),
        "class1_multi_object_images": int(class1_multi_object_images),
        "class1_same_label_multi_object_images": int(class1_same_label_multi_object_images),
        "class1_mixed_label_images": int(class1_mixed_label_images),
    }
    return summary, mixed_examples, multi_examples


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "split",
        "source_stem",
        "label_path",
        "object_count",
        "unique_label_count",
        "unique_labels",
        "labels",
        "class_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def audit_yolo_source_groups(
    data: Path,
    output_dir: Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    max_examples: int = 80,
) -> Dict[str, object]:
    data_yaml = _resolve_data_yaml(data)
    data_spec = load_data_spec(data_yaml, class_name_mode="raw")
    if data_spec.data_format != "yolo":
        raise ValueError(f"Source-group audit expects YOLO data, got: {data_spec.data_format}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_splits = _iter_requested_splits(splits, data_spec.has_test_split)

    split_summaries: Dict[str, object] = {}
    mixed_examples: List[Dict[str, object]] = []
    multi_examples: List[Dict[str, object]] = []
    for split in requested_splits:
        split_summary, split_mixed_examples, split_multi_examples = _audit_split(
            split=split,
            labels_dir=data_spec.split_labels_dir(split),
            num_classes=data_spec.num_classes,
            max_examples=int(max_examples),
        )
        split_summaries[split] = split_summary
        mixed_examples.extend(split_mixed_examples)
        multi_examples.extend(split_multi_examples)

    train_summary = split_summaries.get("train", {})
    val_summary = split_summaries.get("val", {})
    test_summary = split_summaries.get("test", {})
    decision_hints = [
        "Global same-image consistency is risky when mixed-label source images exist.",
        "Same-label peer consistency is only a narrow signal if val/test are mostly single-object.",
    ]
    if int(train_summary.get("mixed_label_images", 0)) > 0:
        decision_hints.append(
            "Train contains mixed-label multi-object images; never force all objects from one source image together."
        )
    if int(val_summary.get("multi_object_images", 0)) == 0 and int(test_summary.get("multi_object_images", 0)) == 0:
        decision_hints.append(
            "Val/test have no multi-object coverage, so source-peer consistency cannot be validated directly."
        )
    elif int(test_summary.get("multi_object_images", 0)) == 0:
        decision_hints.append(
            "Test is single-object, so source-peer consistency must improve object-level boundary decisions to matter."
        )

    summary: Dict[str, object] = {
        "data_yaml": str(data_yaml),
        "class_names": list(data_spec.class_names),
        "splits": split_summaries,
        "decision_hints": decision_hints,
    }

    (output_dir / "source_group_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "mixed_label_examples.csv", mixed_examples[: int(max_examples)])
    _write_csv(output_dir / "multi_object_examples.csv", multi_examples[: int(max_examples)])
    return summary


def main() -> None:
    args = parse_args()
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    summary = audit_yolo_source_groups(
        args.data,
        args.output_dir,
        splits=splits,
        max_examples=args.max_examples,
    )
    print(json.dumps(summary["splits"], indent=2, ensure_ascii=False))
    for hint in summary["decision_hints"]:
        print(f"- {hint}")


if __name__ == "__main__":
    main()

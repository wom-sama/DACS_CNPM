from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from trkh.core.config import load_data_spec
from trkh.core.utils import ensure_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ClassificationRecord:
    source_path: Path
    original_split: str
    class_id: int
    class_name: str
    source_id: str
    image_number: Optional[int]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a grouped/sequence-safe classification-folder split from an existing split."
    )
    parser.add_argument("--data", type=Path, required=True, help="Source classification-folder data.yaml.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument(
        "--near-id-window",
        type=int,
        default=3,
        help=(
            "All Image_N files with ID distance <= this value stay in the same split, "
            "including label transitions inside one capture sequence."
        ),
    )
    parser.add_argument(
        "--disable-group-exact-duplicates",
        action="store_true",
        help="Do not force byte-identical images into the same split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-report-examples", type=int, default=30)
    return parser.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float) -> float:
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("--train-ratio phai nam trong (0, 1).")
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("--val-ratio phai nam trong [0, 1).")
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    if test_ratio <= 0.0:
        raise ValueError("--train-ratio + --val-ratio phai < 1.")
    return test_ratio


def _source_id_from_stem(stem: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"(.+?)_box\d+$", stem)
    source_id = match.group(1) if match else stem
    number_match = re.match(r"Image_(\d+)$", source_id, flags=re.IGNORECASE)
    image_number = int(number_match.group(1)) if number_match else None
    return source_id, image_number


def collect_records(
    data_yaml: Path,
    *,
    class_name_mode: str,
    expected_num_classes: int,
) -> Tuple[List[ClassificationRecord], List[str]]:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes or None,
    )
    if data_spec.data_format != "classification_folder":
        raise ValueError("Tool nay chi ho tro data.yaml format=classification_folder.")

    split_roots = {
        "train": data_spec.train_images,
        "val": data_spec.val_images,
        "test": data_spec.test_images,
    }
    class_names = list(data_spec.class_names)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    records: List[ClassificationRecord] = []
    for split, root in split_roots.items():
        if root is None:
            continue
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                source_id, image_number = _source_id_from_stem(path.stem)
                records.append(
                    ClassificationRecord(
                        source_path=path,
                        original_split=split,
                        class_id=class_to_id[class_name],
                        class_name=class_name,
                        source_id=source_id,
                        image_number=image_number,
                    )
                )
    return records, class_names


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_groups(
    records: Sequence[ClassificationRecord],
    near_id_window: int,
    *,
    group_exact_duplicates: bool = True,
) -> List[List[int]]:
    union_find = UnionFind(len(records))

    by_source: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_source[record.source_id].append(index)
    for indices in by_source.values():
        first = indices[0]
        for index in indices[1:]:
            union_find.union(first, index)

    if group_exact_duplicates:
        by_sha1: Dict[str, List[int]] = defaultdict(list)
        for index, record in enumerate(records):
            by_sha1[_sha1_file(record.source_path)].append(index)
        for indices in by_sha1.values():
            first = indices[0]
            for index in indices[1:]:
                union_find.union(first, index)

    numbered_records: List[Tuple[int, int]] = []
    for index, record in enumerate(records):
        if record.image_number is not None:
            numbered_records.append((int(record.image_number), index))
    window = max(0, int(near_id_window))
    if window > 0:
        numbered_records.sort(key=lambda item: item[0])
        for (previous_number, previous_index), (number, index) in zip(
            numbered_records, numbered_records[1:]
        ):
            if number - previous_number <= window:
                union_find.union(previous_index, index)

    grouped: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union_find.find(index)].append(index)
    return list(grouped.values())


def _group_count_vector(group: Sequence[int], records: Sequence[ClassificationRecord], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for index in group:
        counts[records[index].class_id] += 1
    return counts


def assign_groups(
    groups: Sequence[Sequence[int]],
    records: Sequence[ClassificationRecord],
    *,
    class_names: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    ratios = {"train": float(train_ratio), "val": float(val_ratio), "test": test_ratio}
    num_classes = len(class_names)
    total_counts = [0] * num_classes
    group_payloads = []
    for group in groups:
        vector = _group_count_vector(group, records, num_classes)
        for class_id, count in enumerate(vector):
            total_counts[class_id] += count
        group_payloads.append({"indices": list(group), "counts": vector, "size": len(group)})

    rng = random.Random(seed)
    rng.shuffle(group_payloads)
    group_payloads.sort(key=lambda item: (int(item["size"]), max(item["counts"])), reverse=True)
    target_counts = {
        split: [count * ratio for count in total_counts]
        for split, ratio in ratios.items()
    }
    assigned = {split: [] for split in SPLITS}
    assigned_counts = {split: [0] * num_classes for split in SPLITS}

    def score_after(split: str, vector: Sequence[int]) -> float:
        score = 0.0
        for candidate_split in SPLITS:
            candidate_counts = list(assigned_counts[candidate_split])
            if candidate_split == split:
                candidate_counts = [
                    candidate_counts[class_id] + int(vector[class_id])
                    for class_id in range(num_classes)
                ]
            for class_id in range(num_classes):
                denom = max(1.0, float(total_counts[class_id]))
                target = target_counts[candidate_split][class_id]
                score += ((candidate_counts[class_id] - target) / denom) ** 2
        total_assigned = sum(sum(values) for values in assigned_counts.values()) + sum(vector)
        total_target = sum(total_counts)
        if total_target > 0:
            for candidate_split in SPLITS:
                split_total = sum(assigned_counts[candidate_split]) + (
                    sum(vector) if candidate_split == split else 0
                )
                split_target = total_target * ratios[candidate_split]
                score += 0.25 * ((split_total - split_target) / total_target) ** 2
            score += 0.0 * total_assigned
        return score

    for payload in group_payloads:
        vector = payload["counts"]
        split = min(SPLITS, key=lambda candidate: (score_after(candidate, vector), len(assigned[candidate])))
        assigned[split].extend(payload["indices"])
        for class_id, count in enumerate(vector):
            assigned_counts[split][class_id] += int(count)
    return assigned


def _prepare_output(output_dir: Path, class_names: Sequence[str], overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def _materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if link_mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def write_split(
    output_dir: Path,
    assigned: Dict[str, List[int]],
    records: Sequence[ClassificationRecord],
    *,
    class_names: Sequence[str],
    link_mode: str,
) -> List[Dict[str, object]]:
    manifest = []
    used_names: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for split in SPLITS:
        for index in sorted(assigned[split], key=lambda item: str(records[item].source_path)):
            record = records[index]
            name_counter = used_names[(split, record.class_name)]
            output_name = record.source_path.name
            if name_counter[output_name] > 0:
                output_name = f"{record.source_path.stem}_{name_counter[record.source_path.name]}{record.source_path.suffix}"
            name_counter[record.source_path.name] += 1
            dst = output_dir / split / record.class_name / output_name
            _materialize_file(record.source_path, dst, link_mode)
            manifest.append(
                {
                    "split": split,
                    "original_split": record.original_split,
                    "class_id": record.class_id,
                    "class_name": record.class_name,
                    "source_id": record.source_id,
                    "image_number": record.image_number,
                    "source_path": str(record.source_path.resolve()),
                    "output_path": str(dst.resolve()),
                }
            )
    return manifest


def write_data_yaml(output_dir: Path, class_names: Sequence[str]) -> None:
    payload = {
        "format": "classification_folder",
        "path": ".",
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(class_names),
        "class_name_mode": "raw",
        "canbang_yaml": "canbang.yaml",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_canbang_yaml(
    output_dir: Path,
    manifest: Sequence[Dict[str, object]],
    *,
    class_names: Sequence[str],
    train_ratio: float,
    val_ratio: float,
) -> None:
    counts = Counter(int(row["class_id"]) for row in manifest)
    total = sum(counts.values())
    classes = {}
    for class_id, class_name in enumerate(class_names):
        count = int(counts.get(class_id, 0))
        ratio = count / max(1, total)
        classes[str(class_id)] = {
            "name": class_name,
            "count": count,
            "ratio": round(ratio, 6),
            "percent": round(ratio * 100.0, 2),
        }
    payload = {
        "dataset_balance": {
            "version_note": "Grouped sequence-safe split from cls_crops; no pretrain, no added images",
            "total_images": total,
            "total_objects": total,
            "train_ratio_config": float(train_ratio),
            "val_ratio_config": float(val_ratio),
            "test_ratio_config": float(1.0 - train_ratio - val_ratio),
            "classes": classes,
        }
    }
    (output_dir / "canbang.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, manifest: Sequence[Dict[str, object]]) -> None:
    fields = [
        "split",
        "original_split",
        "class_id",
        "class_name",
        "source_id",
        "image_number",
        "source_path",
        "output_path",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_report(
    *,
    records: Sequence[ClassificationRecord],
    groups: Sequence[Sequence[int]],
    assigned: Dict[str, List[int]],
    manifest: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    near_id_window: int,
    train_ratio: float,
    val_ratio: float,
    max_examples: int,
) -> Dict[str, object]:
    per_split = {
        split: dict(Counter(int(manifest_row["class_id"]) for manifest_row in manifest if manifest_row["split"] == split))
        for split in SPLITS
    }
    group_sizes = sorted((len(group) for group in groups), reverse=True)
    examples = []
    for group in sorted(groups, key=len, reverse=True)[:max(0, max_examples)]:
        split = next(split_name for split_name, indices in assigned.items() if group[0] in set(indices))
        sample_records = [records[index] for index in group[:8]]
        examples.append(
            {
                "split": split,
                "size": len(group),
                "class_counts": dict(Counter(record.class_id for record in (records[index] for index in group))),
                "examples": [
                    {
                        "class_id": record.class_id,
                        "class_name": record.class_name,
                        "source_id": record.source_id,
                        "path": str(record.source_path.resolve()),
                    }
                    for record in sample_records
                ],
            }
        )
    return {
        "total_images": len(records),
        "class_names": list(class_names),
        "source_split_counts": dict(Counter(record.original_split for record in records)),
        "target_split_counts": {split: len(assigned[split]) for split in SPLITS},
        "target_class_counts_by_split": per_split,
        "group_count": len(groups),
        "largest_group_sizes": group_sizes[:20],
        "near_id_window": int(near_id_window),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "largest_group_examples": examples,
    }


def write_report_markdown(output_dir: Path, report: Dict[str, object]) -> None:
    lines = [
        "# Grouped Classification Split Report",
        "",
        f"- Total images: `{report['total_images']}`",
        f"- Group count: `{report['group_count']}`",
        f"- Near-ID window: `{report['near_id_window']}`",
        f"- Target split counts: `{report['target_split_counts']}`",
        f"- Largest groups: `{report['largest_group_sizes'][:10]}`",
        "",
        "## Class Counts By Split",
        "",
        "| Split | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        counts = {int(k): int(v) for k, v in report["target_class_counts_by_split"][split].items()}
        total = sum(counts.values())
        lines.append(
            "| {split} | {c0} | {c1} | {c2} | {c3} | {c4} | {total} |".format(
                split=split,
                c0=counts.get(0, 0),
                c1=counts.get(1, 0),
                c2=counts.get(2, 0),
                c3=counts.get(3, 0),
                c4=counts.get(4, 0),
                total=total,
            )
        )
    lines.extend(
        [
            "",
            (
                "This split keeps same-source crops and every near-neighbor `Image_N` "
                "capture sequence in one split, including transitions between class "
                "labels. It is intended for publication-grade evaluation when no more "
                "data can be collected."
            ),
            "",
            "## Largest Group Examples",
            "",
        ]
    )
    for example in report["largest_group_examples"][:10]:
        lines.append(f"- split `{example['split']}`, size `{example['size']}`, class_counts `{example['class_counts']}`")
        for item in example["examples"][:4]:
            lines.append(f"  - `{item['class_name']}` / `{item['source_id']}` / `{item['path']}`")
    (output_dir / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio)
    records, class_names = collect_records(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
    )
    if not records:
        raise ValueError("Khong tim thay anh classification nao trong data.yaml nguon.")
    groups = build_groups(
        records,
        args.near_id_window,
        group_exact_duplicates=not bool(args.disable_group_exact_duplicates),
    )
    assigned = assign_groups(
        groups,
        records,
        class_names=class_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    output_dir = ensure_dir(args.output_dir)
    _prepare_output(output_dir, class_names, overwrite=bool(args.overwrite))
    manifest = write_split(
        output_dir,
        assigned,
        records,
        class_names=class_names,
        link_mode=args.link_mode,
    )
    write_data_yaml(output_dir, class_names)
    write_canbang_yaml(
        output_dir,
        manifest,
        class_names=class_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_manifest(output_dir, manifest)
    report = build_report(
        records=records,
        groups=groups,
        assigned=assigned,
        manifest=manifest,
        class_names=class_names,
        near_id_window=args.near_id_window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_examples=args.max_report_examples,
    )
    (output_dir / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_markdown(output_dir, report)
    print(
        {
            "output_dir": str(output_dir.resolve()),
            "total_images": len(records),
            "groups": len(groups),
            "target_split_counts": report["target_split_counts"],
            "largest_group_sizes": report["largest_group_sizes"][:5],
        }
    )


if __name__ == "__main__":
    main()

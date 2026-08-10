from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from trkh.core.config import load_data_spec
from trkh.core.utils import ensure_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit classification-folder split leakage.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default=None)
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ahash-size", type=int, default=8)
    parser.add_argument(
        "--near-id-window",
        type=int,
        default=3,
        help="Soft leak check: report same-class Image_N files across splits with numeric IDs this close.",
    )
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def _iter_records(data_yaml: Path, class_name_mode: Optional[str], expected_num_classes: int) -> List[Dict[str, object]]:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes or None,
    )
    if data_spec.data_format != "classification_folder":
        raise ValueError("audit_classification_split chi ho tro format=classification_folder.")
    records: List[Dict[str, object]] = []
    for split, root in (("train", data_spec.train_images), ("val", data_spec.val_images), ("test", data_spec.test_images)):
        if root is None:
            continue
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = image_path.stem
            source_match = re.match(r"(.+?)_box\d+$", stem)
            source_id = source_match.group(1) if source_match else stem
            image_number_match = re.match(r"Image_(\d+)$", source_id, flags=re.IGNORECASE)
            records.append(
                {
                    "split": split,
                    "class_name": image_path.parent.name,
                    "path": image_path,
                    "source_id": source_id,
                    "image_number": int(image_number_match.group(1)) if image_number_match else None,
                }
            )
    return records


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(path: Path, size: int) -> int:
    size = max(4, int(size))
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        values = list(image.getdata())
    average = sum(values) / max(1, len(values))
    bits = 0
    for index, value in enumerate(values):
        if value >= average:
            bits |= 1 << index
    return bits


def _cross_split_groups(groups: Dict[object, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    results = []
    for key, items in groups.items():
        split_names = sorted({str(item["split"]) for item in items})
        if len(split_names) <= 1:
            continue
        results.append(
            {
                "key": key,
                "splits": split_names,
                "items": [
                    {
                        "split": str(item["split"]),
                        "class_name": str(item["class_name"]),
                        "path": str(Path(item["path"]).resolve()),
                    }
                    for item in items
                ],
            }
        )
    return sorted(results, key=lambda item: len(item["items"]), reverse=True)


def _near_id_cross_split_pairs(
    records: List[Dict[str, object]],
    *,
    window: int,
    max_examples: int,
) -> Tuple[int, int, List[Dict[str, object]]]:
    window = max(0, int(window))
    if window <= 0:
        return 0, 0, []
    by_class: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("image_number") is not None:
            by_class[str(record["class_name"])].append(record)

    pair_count = 0
    paths = set()
    examples: List[Dict[str, object]] = []
    for class_name, items in by_class.items():
        items = sorted(items, key=lambda item: int(item["image_number"]))
        for left_index, left in enumerate(items):
            left_number = int(left["image_number"])
            for right in items[left_index + 1 :]:
                right_number = int(right["image_number"])
                distance = right_number - left_number
                if distance > window:
                    break
                if str(left["split"]) == str(right["split"]):
                    continue
                pair_count += 1
                paths.add(str(Path(left["path"]).resolve()))
                paths.add(str(Path(right["path"]).resolve()))
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "key": f"{class_name}: Image_{left_number} ~ Image_{right_number}",
                            "distance": distance,
                            "items": [
                                {
                                    "split": str(left["split"]),
                                    "class_name": class_name,
                                    "path": str(Path(left["path"]).resolve()),
                                },
                                {
                                    "split": str(right["split"]),
                                    "class_name": class_name,
                                    "path": str(Path(right["path"]).resolve()),
                                },
                            ],
                        }
                    )
    return pair_count, len(paths), examples


def _write_markdown(path: Path, report: Dict[str, object], max_examples: int) -> None:
    lines = [
        "# Classification Split Leak Audit",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Total images: `{report['total_images']}`",
        "",
        "| Check | Groups | Files |",
        "|---|---:|---:|",
        (
            f"| Exact SHA1 duplicate across splits | {report['exact_cross_split_groups']} | "
            f"{report['exact_cross_split_files']} |"
        ),
        (
            f"| Same source stem across splits | {report['source_cross_split_groups']} | "
            f"{report['source_cross_split_files']} |"
        ),
        (
            f"| Same average-hash bucket across splits | {report['ahash_cross_split_groups']} | "
            f"{report['ahash_cross_split_files']} |"
        ),
        (
            f"| Same-class near numeric image ID across splits | {report['near_id_cross_split_pairs']} pairs | "
            f"{report['near_id_cross_split_files']} |"
        ),
        "",
        "Exact SHA1 and same source stem are hard leak signals. Average-hash overlap and near numeric image ID are soft signals: they can indicate near-duplicate/sequence leakage, but also occur when many similar mango crops were collected in a row.",
        "",
    ]
    for section_key, title in (
        ("source_cross_split_examples", "Source Stem Examples"),
        ("exact_cross_split_examples", "Exact Duplicate Examples"),
        ("near_id_cross_split_examples", "Near Numeric Image ID Examples"),
        ("ahash_cross_split_examples", "Average-Hash Examples"),
    ):
        examples = list(report.get(section_key, []))[:max_examples]
        lines.extend([f"## {title}", ""])
        if not examples:
            lines.append("- None")
            lines.append("")
            continue
        for example in examples:
            lines.append(f"- `{example['key']}`:")
            for item in example["items"][:8]:
                lines.append(f"  - `{item['split']}` / `{item['class_name']}` / `{item['path']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = _iter_records(args.data, args.class_name_mode, args.expected_num_classes)
    by_sha1: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_source: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_ahash: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        path = Path(record["path"])
        by_sha1[_sha1_file(path)].append(record)
        by_source[str(record["source_id"])].append(record)
        by_ahash[hex(_average_hash(path, int(args.ahash_size)))].append(record)

    exact_cross = _cross_split_groups(by_sha1)
    source_cross = _cross_split_groups(by_source)
    ahash_cross = _cross_split_groups(by_ahash)
    near_id_pairs, near_id_files, near_id_examples = _near_id_cross_split_pairs(
        records,
        window=int(args.near_id_window),
        max_examples=max(0, int(args.max_examples)),
    )
    output_dir = ensure_dir(args.output_dir)
    report = {
        "dataset": str(args.data.resolve()),
        "total_images": len(records),
        "counts_by_split": dict(Counter(str(record["split"]) for record in records)),
        "exact_cross_split_groups": len(exact_cross),
        "exact_cross_split_files": sum(len(group["items"]) for group in exact_cross),
        "source_cross_split_groups": len(source_cross),
        "source_cross_split_files": sum(len(group["items"]) for group in source_cross),
        "ahash_cross_split_groups": len(ahash_cross),
        "ahash_cross_split_files": sum(len(group["items"]) for group in ahash_cross),
        "near_id_window": int(args.near_id_window),
        "near_id_cross_split_pairs": near_id_pairs,
        "near_id_cross_split_files": near_id_files,
        "exact_cross_split_examples": exact_cross[: max(0, int(args.max_examples))],
        "source_cross_split_examples": source_cross[: max(0, int(args.max_examples))],
        "near_id_cross_split_examples": near_id_examples,
        "ahash_cross_split_examples": ahash_cross[: max(0, int(args.max_examples))],
    }
    (output_dir / "leak_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(output_dir / "leak_audit.md", report, max_examples=max(0, int(args.max_examples)))
    print(
        {
            "output_dir": str(output_dir.resolve()),
            "exact_cross_split_groups": report["exact_cross_split_groups"],
            "source_cross_split_groups": report["source_cross_split_groups"],
            "ahash_cross_split_groups": report["ahash_cross_split_groups"],
            "near_id_cross_split_pairs": report["near_id_cross_split_pairs"],
        }
    )


if __name__ == "__main__":
    main()

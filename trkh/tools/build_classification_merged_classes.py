from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from trkh.core.config import load_data_spec
from trkh.core.utils import ensure_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a classification-folder ablation dataset by merging one or more source classes. "
            "The original train/val/test split is preserved."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Source classification-folder data.yaml.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument(
        "--merge-classes",
        required=True,
        help="Comma-separated source class IDs to merge into new class 0, for example: 0,1.",
    )
    parser.add_argument(
        "--merged-name",
        default="",
        help="Name for the new merged class. Defaults to source names joined with _MERGED_.",
    )
    parser.add_argument("--link-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_class_ids(value: str) -> List[int]:
    class_ids = []
    for raw_item in str(value).split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        class_ids.append(int(raw_item))
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("--merge-classes khong duoc lap class ID.")
    if len(class_ids) < 2:
        raise ValueError("--merge-classes can it nhat 2 class ID de gop.")
    return class_ids


def _materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
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


def _prepare_output(output_dir: Path, class_names: Sequence[str], overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Thu muc output da ton tai va khong rong: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def _build_mapping(
    class_names: Sequence[str],
    merge_classes: Sequence[int],
    *,
    merged_name: str,
) -> Tuple[Dict[int, int], List[str], Dict[int, List[int]]]:
    num_classes = len(class_names)
    merge_set = set(int(class_id) for class_id in merge_classes)
    for class_id in merge_set:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"Class ID nam ngoai khoang: {class_id}, num_classes={num_classes}")

    new_class_names: List[str] = [
        str(merged_name).strip()
        or "_MERGED_".join(str(class_names[class_id]) for class_id in sorted(merge_set))
    ]
    old_to_new: Dict[int, int] = {}
    new_to_old: Dict[int, List[int]] = {0: sorted(merge_set)}
    for class_id in sorted(merge_set):
        old_to_new[class_id] = 0

    for class_id, class_name in enumerate(class_names):
        if class_id in merge_set:
            continue
        new_class_id = len(new_class_names)
        old_to_new[class_id] = new_class_id
        new_to_old[new_class_id] = [class_id]
        new_class_names.append(str(class_name))
    return old_to_new, new_class_names, new_to_old


def _iter_source_images(root: Path) -> List[Path]:
    if root is None or not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _write_data_yaml(output_dir: Path, class_names: Sequence[str]) -> None:
    payload = {
        "format": "classification_folder",
        "path": ".",
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(class_names),
        "class_name_mode": "raw",
        "canbang_yaml": "canbang.yaml",
        "names": {index: class_name for index, class_name in enumerate(class_names)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_canbang_yaml(
    output_dir: Path,
    manifest: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    *,
    source_data_yaml: Path,
    new_to_old: Dict[int, List[int]],
) -> None:
    counts = Counter(int(row["new_class_id"]) for row in manifest)
    total = sum(counts.values())
    per_split = {
        split: sum(1 for row in manifest if str(row["split"]) == split)
        for split in SPLITS
    }
    classes = {}
    for class_id, class_name in enumerate(class_names):
        count = int(counts.get(class_id, 0))
        ratio = count / max(1, total)
        classes[str(class_id)] = {
            "name": class_name,
            "count": count,
            "ratio": round(ratio, 6),
            "percent": round(ratio * 100.0, 2),
            "source_class_ids": [int(value) for value in new_to_old.get(class_id, [])],
        }
    payload = {
        "dataset_balance": {
            "version_note": (
                "Classification-folder ablation dataset with merged classes; "
                "original split preserved; no pretrain, no added images"
            ),
            "source_data_yaml": str(source_data_yaml.resolve()),
            "total_images": total,
            "total_objects": total,
            "train_ratio_config": per_split["train"] / max(1, total),
            "val_ratio_config": per_split["val"] / max(1, total),
            "test_ratio_config": per_split["test"] / max(1, total),
            "classes": classes,
        }
    }
    (output_dir / "canbang.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_manifest(output_dir: Path, manifest: Sequence[Dict[str, object]]) -> None:
    fields = [
        "split",
        "original_class_id",
        "original_class_name",
        "new_class_id",
        "new_class_name",
        "source_path",
        "output_path",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_summary(
    output_dir: Path,
    manifest: Sequence[Dict[str, object]],
    *,
    source_data_yaml: Path,
    old_to_new: Dict[int, int],
    new_to_old: Dict[int, List[int]],
    class_names: Sequence[str],
) -> Dict[str, object]:
    per_split = {
        split: dict(Counter(int(row["new_class_id"]) for row in manifest if str(row["split"]) == split))
        for split in SPLITS
    }
    summary = {
        "source_data_yaml": str(source_data_yaml.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_images": len(manifest),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "old_to_new": {str(key): int(value) for key, value in sorted(old_to_new.items())},
        "new_to_old": {str(key): [int(item) for item in value] for key, value in sorted(new_to_old.items())},
        "counts": dict(Counter(int(row["new_class_id"]) for row in manifest)),
        "per_split_counts": per_split,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_merged_classification_dataset(
    data_yaml: Path,
    output_dir: Path,
    *,
    merge_classes: Sequence[int],
    merged_name: str = "",
    class_name_mode: str = "raw",
    expected_num_classes: int = 0,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> Dict[str, object]:
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes or None,
    )
    if data_spec.data_format != "classification_folder":
        raise ValueError("Tool nay chi ho tro format=classification_folder.")

    old_class_names = list(data_spec.class_names)
    old_to_new, new_class_names, new_to_old = _build_mapping(
        old_class_names,
        merge_classes,
        merged_name=merged_name,
    )
    _prepare_output(output_dir, new_class_names, overwrite)

    split_roots = {
        "train": data_spec.train_images,
        "val": data_spec.val_images,
        "test": data_spec.test_images,
    }
    used_names: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    manifest: List[Dict[str, object]] = []
    old_name_to_id = {name: index for index, name in enumerate(old_class_names)}
    for split, root in split_roots.items():
        if root is None:
            continue
        for source_path in _iter_source_images(root):
            original_class_name = source_path.parent.name
            if original_class_name not in old_name_to_id:
                continue
            original_class_id = old_name_to_id[original_class_name]
            new_class_id = old_to_new[original_class_id]
            new_class_name = new_class_names[new_class_id]
            name_counter = used_names[(split, new_class_name)]
            output_name = source_path.name
            if name_counter[output_name] > 0:
                output_name = f"{source_path.stem}_{name_counter[source_path.name]}{source_path.suffix}"
            name_counter[source_path.name] += 1
            output_path = output_dir / split / new_class_name / output_name
            _materialize_file(source_path, output_path, link_mode)
            manifest.append(
                {
                    "split": split,
                    "original_class_id": int(original_class_id),
                    "original_class_name": original_class_name,
                    "new_class_id": int(new_class_id),
                    "new_class_name": new_class_name,
                    "source_path": str(source_path.resolve()),
                    "output_path": str(output_path.resolve()),
                }
            )

    _write_data_yaml(output_dir, new_class_names)
    _write_canbang_yaml(output_dir, manifest, new_class_names, source_data_yaml=data_yaml, new_to_old=new_to_old)
    _write_manifest(output_dir, manifest)
    return _write_summary(
        output_dir,
        manifest,
        source_data_yaml=data_yaml,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        class_names=new_class_names,
    )


def main() -> None:
    args = parse_args()
    summary = build_merged_classification_dataset(
        args.data,
        args.output_dir,
        merge_classes=_parse_class_ids(args.merge_classes),
        merged_name=args.merged_name,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    ensure_dir(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

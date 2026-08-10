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


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a classification-folder dataset that keeps only samples whose source_id "
            "appears once in the source manifest. This removes crops from multi-object source images."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Source classification-folder data.yaml.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None, help="Defaults to <data root>/manifest.csv.")
    parser.add_argument("--link-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def _read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Khong tim thay manifest.csv: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        "names": {index: name for index, name in enumerate(class_names)},
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
) -> None:
    counts = Counter(int(row["class_id"]) for row in manifest)
    total = sum(counts.values())
    per_split = {split: sum(1 for row in manifest if str(row["split"]) == split) for split in SPLITS}
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
            "version_note": (
                "Grouped sequence-safe classification split filtered to source_id count == 1; "
                "multi-object source crops removed; no pretrain, no added images"
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


def _write_report(
    output_dir: Path,
    *,
    source_data_yaml: Path,
    source_manifest: Path,
    source_rows: Sequence[Dict[str, str]],
    kept_rows: Sequence[Dict[str, object]],
    removed_source_ids: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, object]:
    per_split = {
        split: dict(Counter(int(row["class_id"]) for row in kept_rows if str(row["split"]) == split))
        for split in SPLITS
    }
    source_counts = Counter(row.get("source_id", "") for row in source_rows)
    report = {
        "source_data_yaml": str(source_data_yaml.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_names": list(class_names),
        "source_rows": len(source_rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(source_rows) - len(kept_rows),
        "source_ids": len(source_counts),
        "single_source_ids": sum(1 for count in source_counts.values() if count == 1),
        "multi_source_ids": sum(1 for count in source_counts.values() if count > 1),
        "removed_source_ids": len(removed_source_ids),
        "target_split_counts": dict(Counter(str(row["split"]) for row in kept_rows)),
        "target_class_counts_by_split": per_split,
        "target_class_counts": dict(Counter(int(row["class_id"]) for row in kept_rows)),
    }
    (output_dir / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Single-Source Classification Filter",
        "",
        f"- Source data: `{source_data_yaml.resolve()}`",
        f"- Source rows: `{report['source_rows']}`",
        f"- Kept rows: `{report['kept_rows']}`",
        f"- Removed rows: `{report['removed_rows']}`",
        f"- Removed multi-source IDs: `{report['removed_source_ids']}`",
        "",
        "## Class Counts By Split",
        "",
        "| Split | Class | Count |",
        "| --- | --- | ---: |",
    ]
    for split in SPLITS:
        for class_id, class_name in enumerate(class_names):
            count = int(per_split.get(split, {}).get(class_id, 0))
            lines.append(f"| {split} | {class_id} - {class_name} | {count} |")
    (output_dir / "filter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_single_source_classification_dataset(
    data_yaml: Path,
    output_dir: Path,
    *,
    class_name_mode: str = "raw",
    expected_num_classes: int = 0,
    manifest_path: Path | None = None,
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
    class_names = list(data_spec.class_names)
    manifest_path = Path(manifest_path) if manifest_path is not None else data_spec.root / "manifest.csv"
    source_rows = _read_manifest(manifest_path)

    source_counts = Counter(str(row.get("source_id", "")) for row in source_rows)
    kept_source_ids = {source_id for source_id, count in source_counts.items() if count == 1 and source_id}
    removed_source_ids = [source_id for source_id, count in source_counts.items() if count > 1]
    _prepare_output(output_dir, class_names, overwrite)

    used_names: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    kept_rows: List[Dict[str, object]] = []
    for row in source_rows:
        source_id = str(row.get("source_id", ""))
        if source_id not in kept_source_ids:
            continue
        split = str(row.get("split", "")).strip()
        class_id = int(row.get("class_id", 0))
        class_name = str(row.get("class_name", class_names[class_id] if 0 <= class_id < len(class_names) else ""))
        if split not in SPLITS:
            raise ValueError(f"Split khong hop le trong manifest: {split!r}")
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"class_id khong hop le trong manifest: {class_id}")
        if class_name != class_names[class_id]:
            class_name = class_names[class_id]
        source_path = Path(str(row.get("output_path") or row.get("source_path") or ""))
        if not source_path.is_file():
            raise FileNotFoundError(f"Khong tim thay file source trong manifest: {source_path}")
        name_counter = used_names[(split, class_name)]
        output_name = source_path.name
        if name_counter[output_name] > 0:
            output_name = f"{source_path.stem}_{name_counter[source_path.name]}{source_path.suffix}"
        name_counter[source_path.name] += 1
        output_path = output_dir / split / class_name / output_name
        _materialize_file(source_path, output_path, link_mode)
        kept_rows.append(
            {
                "split": split,
                "original_split": row.get("original_split", ""),
                "class_id": class_id,
                "class_name": class_name,
                "source_id": source_id,
                "image_number": row.get("image_number", ""),
                "source_path": str(source_path.resolve()),
                "output_path": str(output_path.resolve()),
            }
        )

    _write_data_yaml(output_dir, class_names)
    _write_canbang_yaml(output_dir, kept_rows, class_names, source_data_yaml=data_yaml)
    _write_manifest(output_dir, kept_rows)
    return _write_report(
        output_dir,
        source_data_yaml=data_yaml,
        source_manifest=manifest_path,
        source_rows=source_rows,
        kept_rows=kept_rows,
        removed_source_ids=removed_source_ids,
        class_names=class_names,
    )


def main() -> None:
    args = parse_args()
    report = build_single_source_classification_dataset(
        args.data,
        args.output_dir,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
        manifest_path=args.manifest,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

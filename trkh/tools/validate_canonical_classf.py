from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, Mapping, Sequence

from trkh.core.config import load_data_spec, to_serializable
from trkh.data.dataset import IMAGE_EXTENSIONS


CANONICAL_CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
CANONICAL_FILE_SHA256 = {
    "data.yaml": "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8",
    "manifest.csv": "59cf846b69f73c72bc415feaa3ce19fae13ca99f2c58ddb1119c0aaacc77a870",
    "stats.json": "17d39f7de0e84b578a2ba77c1958ed3de9c73fe9715bc7ebd6c19b465e0a8273",
}
CANONICAL_IMAGE_TREE_SHA256 = (
    "70a1b7d2b4c6f80e28fe3f0f714f1ba3e8ab654a90ce50b1e9cae8e4dac4a503"
)
CANONICAL_SPLIT_CLASS_COUNTS = {
    "train": (1987, 497, 1326, 2080, 2388),
    "val": (558, 158, 380, 494, 889),
    "test": (278, 83, 180, 260, 461),
}
CANONICAL_SOURCE_IMAGE_COUNTS = {
    "train": 7751,
    "val": 2471,
    "test": 1262,
}
CANONICAL_LEAKAGE_GROUP_COUNTS = {
    "train": 5361,
    "val": 2471,
    "test": 1262,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_count(path: Path) -> int:
    return sum(
        1
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_relative_paths(root: Path) -> list[str]:
    return sorted(
        candidate.relative_to(root).as_posix()
        for split in CANONICAL_SPLIT_CLASS_COUNTS
        for class_name in CANONICAL_CLASS_NAMES
        for candidate in (root / split / class_name).rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_tree_sha256(root: Path, relative_paths: Sequence[str]) -> str:
    """Hash ordered relative paths and each image's content digest."""

    digest = hashlib.sha256()
    for relative_path in sorted(str(path) for path in relative_paths):
        image_path = root / Path(relative_path)
        file_digest = _sha256(image_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_relative_output_path(row: Mapping[str, str]) -> str:
    split = str(row["split"]).strip().lower()
    class_name = str(row["class_name"])
    output_parts = PureWindowsPath(str(row["output_image"])).parts
    if len(output_parts) < 3:
        raise ValueError(
            f"Manifest output_image has no split/class suffix: {row['output_image']!r}."
        )
    observed_suffix = tuple(output_parts[-3:-1])
    expected_suffix = (split, class_name)
    _require_equal(
        observed_suffix,
        expected_suffix,
        "manifest output split/class suffix",
    )
    return f"{split}/{class_name}/{output_parts[-1]}"


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ValueError(
            f"Canonical class_f contract failed for {label}: "
            f"observed={observed!r}, expected={expected!r}."
        )


def validate_canonical_classf(
    data_yaml: Path,
    *,
    require_exact_file_hashes: bool = True,
    require_exact_image_tree_hash: bool = True,
) -> Dict[str, object]:
    """Validate class_f v1 without model inference or test-metric evaluation."""

    data_yaml = Path(data_yaml).resolve()
    root = data_yaml.parent
    files = {
        "data.yaml": data_yaml,
        "manifest.csv": root / "manifest.csv",
        "stats.json": root / "stats.json",
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Canonical class_f file is missing: {name} -> {path}")
    file_hashes = {name: _sha256(path) for name, path in files.items()}
    if require_exact_file_hashes:
        _require_equal(file_hashes, CANONICAL_FILE_SHA256, "file SHA-256 map")

    data_spec = load_data_spec(
        data_yaml,
        class_name_mode="raw",
        expected_num_classes=5,
    )
    _require_equal(
        data_spec.data_format,
        "classification_folder",
        "data format",
    )
    _require_equal(
        tuple(data_spec.class_names),
        CANONICAL_CLASS_NAMES,
        "class order",
    )

    split_class_counts: Dict[str, list[int]] = {}
    for split, expected_counts in CANONICAL_SPLIT_CLASS_COUNTS.items():
        split_root = data_spec.split_images_dir(split)
        if not split_root.is_dir():
            raise FileNotFoundError(f"Canonical class_f split is missing: {split_root}")
        observed_counts = [
            _image_count(split_root / class_name)
            for class_name in CANONICAL_CLASS_NAMES
        ]
        _require_equal(
            tuple(observed_counts),
            tuple(expected_counts),
            f"{split} class counts",
        )
        split_class_counts[split] = observed_counts

    manifest_rows = []
    with files["manifest.csv"].open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "split",
            "source_image",
            "output_image",
            "class_id",
            "class_name",
            "leakage_group",
        }
        missing_columns = sorted(required_columns.difference(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(
                f"Canonical manifest is missing columns: {missing_columns!r}."
            )
        manifest_rows = [dict(row) for row in reader]

    expected_total = sum(
        sum(counts) for counts in CANONICAL_SPLIT_CLASS_COUNTS.values()
    )
    _require_equal(len(manifest_rows), expected_total, "manifest row count")
    output_paths = [str(row["output_image"]) for row in manifest_rows]
    _require_equal(
        len(set(output_paths)),
        len(output_paths),
        "unique manifest output paths",
    )
    manifest_relative_paths = [
        _manifest_relative_output_path(row) for row in manifest_rows
    ]
    _require_equal(
        len(set(manifest_relative_paths)),
        len(manifest_relative_paths),
        "unique manifest relative output paths",
    )
    image_relative_paths = _image_relative_paths(data_spec.root)
    _require_equal(
        image_relative_paths,
        sorted(manifest_relative_paths),
        "manifest/filesystem image inventory",
    )
    image_tree_sha256 = _image_tree_sha256(
        data_spec.root,
        image_relative_paths,
    )
    if require_exact_image_tree_hash:
        _require_equal(
            image_tree_sha256,
            CANONICAL_IMAGE_TREE_SHA256,
            "image-tree SHA-256",
        )

    manifest_split_class_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    source_images: Dict[str, set[str]] = defaultdict(set)
    leakage_groups: Dict[str, set[str]] = defaultdict(set)
    leakage_group_splits: Dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        split = str(row["split"]).strip().lower()
        class_id = int(row["class_id"])
        if split not in CANONICAL_SPLIT_CLASS_COUNTS:
            raise ValueError(f"Unexpected manifest split: {split!r}.")
        if not 0 <= class_id < len(CANONICAL_CLASS_NAMES):
            raise ValueError(f"Unexpected manifest class_id: {class_id}.")
        _require_equal(
            str(row["class_name"]),
            CANONICAL_CLASS_NAMES[class_id],
            "manifest class id/name mapping",
        )
        manifest_split_class_counts[split][class_id] += 1
        source_images[split].add(str(row["source_image"]))
        leakage_group = str(row["leakage_group"])
        leakage_groups[split].add(leakage_group)
        leakage_group_splits[leakage_group].add(split)

    cross_split_groups = sorted(
        group
        for group, splits in leakage_group_splits.items()
        if len(splits) > 1
    )
    _require_equal(cross_split_groups, [], "cross-split leakage groups")
    for split, expected_counts in CANONICAL_SPLIT_CLASS_COUNTS.items():
        observed_counts = tuple(
            int(manifest_split_class_counts[split][class_index])
            for class_index in range(len(CANONICAL_CLASS_NAMES))
        )
        _require_equal(
            observed_counts,
            tuple(expected_counts),
            f"manifest {split} class counts",
        )
        _require_equal(
            len(source_images[split]),
            CANONICAL_SOURCE_IMAGE_COUNTS[split],
            f"manifest {split} source-image count",
        )
        _require_equal(
            len(leakage_groups[split]),
            CANONICAL_LEAKAGE_GROUP_COUNTS[split],
            f"manifest {split} leakage-group count",
        )

    return {
        "schema_version": 1,
        "contract": "TRKH_CLASS_F_CANONICAL_V1_20260730",
        "data_yaml": str(data_yaml),
        "root": str(data_spec.root),
        "data_format": data_spec.data_format,
        "class_names": list(data_spec.class_names),
        "file_sha256": file_hashes,
        "image_tree_sha256": image_tree_sha256,
        "image_tree_file_count": len(image_relative_paths),
        "split_class_counts": split_class_counts,
        "split_totals": {
            split: int(sum(counts))
            for split, counts in split_class_counts.items()
        },
        "source_image_counts": {
            split: len(source_images[split])
            for split in CANONICAL_SPLIT_CLASS_COUNTS
        },
        "leakage_group_counts": {
            split: len(leakage_groups[split])
            for split in CANONICAL_SPLIT_CLASS_COUNTS
        },
        "cross_split_leakage_group_count": 0,
        "test_metadata_used_for_static_integrity": True,
        "test_image_bytes_hashed_for_static_integrity": True,
        "test_model_inference_performed": False,
        "test_metrics_read": False,
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable canonical TRKH class_f dataset contract."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allow-file-hash-drift",
        action="store_true",
        default=False,
        help="Engineering-only: validate structure/counts but do not accept as canonical v1.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = validate_canonical_classf(
        args.data,
        require_exact_file_hashes=not bool(args.allow_file_hash_drift),
    )
    text = json.dumps(to_serializable(payload), ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

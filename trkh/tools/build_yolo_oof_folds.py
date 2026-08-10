from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set

import yaml
from sklearn.model_selection import StratifiedGroupKFold

from trkh.core.config import load_data_spec


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class YoloImageSample:
    image_path: Path
    label_path: Path
    stem: str
    labels: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build grouped stratified train-only OOF folds for YOLO object-level classification. "
            "Images are grouped by source image so the same image cannot appear in both fold/train "
            "and fold/val; fold/test mirrors fold/val for generic runners with --skip-test."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-split", default="train", choices=("train", "val"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="auto tries hardlinks and falls back to copying individual files if needed.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional smoke cap on source images after sorting. 0 keeps all source-split images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _index_image_paths(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    image_paths: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.setdefault(path.stem, path)
    return image_paths


def _parse_valid_labels(label_path: Path, *, num_classes: int) -> tuple[int, ...]:
    labels: List[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            label = int(parts[0])
            x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        if label < 0 or label >= int(num_classes):
            continue
        if any(value < 0.0 or value > 1.0 for value in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
            continue
        labels.append(int(label))
    return tuple(labels)


def _apply_balanced_image_cap(
    samples: Sequence[YoloImageSample],
    *,
    max_images: int,
    num_classes: int,
) -> List[YoloImageSample]:
    if max_images <= 0 or max_images >= len(samples):
        return list(samples)

    by_class: Dict[int, List[int]] = {class_index: [] for class_index in range(int(num_classes))}
    for sample_index, sample in enumerate(samples):
        for label in sorted(set(sample.labels)):
            if 0 <= int(label) < int(num_classes):
                by_class[int(label)].append(int(sample_index))

    selected: Set[int] = set()
    cursors = {class_index: 0 for class_index in by_class}
    while len(selected) < int(max_images):
        added_this_round = False
        for class_index in range(int(num_classes)):
            indices = by_class.get(class_index, [])
            cursor = cursors[class_index]
            while cursor < len(indices) and indices[cursor] in selected:
                cursor += 1
            cursors[class_index] = cursor
            if cursor >= len(indices):
                continue
            selected.add(indices[cursor])
            cursors[class_index] = cursor + 1
            added_this_round = True
            if len(selected) >= int(max_images):
                break
        if not added_this_round:
            break

    if len(selected) < int(max_images):
        for sample_index in range(len(samples)):
            selected.add(int(sample_index))
            if len(selected) >= int(max_images):
                break

    return [samples[index] for index in sorted(selected)]


def _scan_yolo_samples(
    data_yaml: Path,
    *,
    source_split: str,
    max_images: int = 0,
) -> tuple[List[YoloImageSample], List[str], Path, Path]:
    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    if data_spec.data_format == "classification_folder":
        raise ValueError("build_yolo_oof_folds expects YOLO-format data, not classification_folder.")

    images_dir = data_spec.split_images_dir(source_split)
    labels_dir = data_spec.split_labels_dir(source_split)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")

    image_paths = _index_image_paths(images_dir)
    class_names = [str(value) for value in data_spec.class_names]
    samples: List[YoloImageSample] = []
    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda item: item.name):
        image_path = image_paths.get(label_path.stem)
        if image_path is None:
            continue
        labels = _parse_valid_labels(label_path, num_classes=len(class_names))
        if not labels:
            continue
        samples.append(
            YoloImageSample(
                image_path=image_path,
                label_path=label_path,
                stem=label_path.stem,
                labels=labels,
            )
        )
    if not samples:
        raise ValueError(f"No valid YOLO image/object samples found in {labels_dir}")
    samples = _apply_balanced_image_cap(
        samples,
        max_images=int(max_images),
        num_classes=len(class_names),
    )
    return samples, class_names, images_dir, labels_dir


def _prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists. Pass --overwrite to replace: {path}")
        resolved = path.resolve()
        if len(resolved.parts) < 3:
            raise ValueError(f"Refusing to remove suspiciously short output path: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_file(source: Path, destination: Path, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    if link_mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def _write_fold_data_yaml(fold_root: Path, class_names: Sequence[str]) -> None:
    fold_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(fold_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": list(class_names),
        "format": "yolo",
        "class_name_mode": "raw",
        "note": (
            "OOF fold built from train-only YOLO images. Images are group-split, and fold/test "
            "mirrors fold/val only to satisfy generic runners when --skip-test is used."
        ),
    }
    (fold_root / "data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _class_counts(samples: Sequence[YoloImageSample], indices: Set[int], num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for sample_index in sorted(indices):
        for label in samples[int(sample_index)].labels:
            counts[int(label)] += 1
    return counts


def _materialize_split(
    *,
    fold_root: Path,
    split_name: str,
    samples: Sequence[YoloImageSample],
    indices: Set[int],
    link_mode: str,
    materialized_counts: Dict[str, int],
) -> None:
    for sample_index in sorted(indices):
        sample = samples[int(sample_index)]
        image_destination = fold_root / "images" / split_name / sample.image_path.name
        label_destination = fold_root / "labels" / split_name / sample.label_path.name
        image_mode = _materialize_file(sample.image_path, image_destination, link_mode)
        label_mode = _materialize_file(sample.label_path, label_destination, link_mode)
        materialized_counts[image_mode] = int(materialized_counts.get(image_mode, 0)) + 1
        materialized_counts[label_mode] = int(materialized_counts.get(label_mode, 0)) + 1


def _expanded_object_targets(samples: Sequence[YoloImageSample]) -> tuple[List[int], List[int], List[int]]:
    x: List[int] = []
    y: List[int] = []
    groups: List[int] = []
    for sample_index, sample in enumerate(samples):
        for label in sample.labels:
            x.append(int(sample_index))
            y.append(int(label))
            groups.append(int(sample_index))
    return x, y, groups


def build_folds(
    *,
    data_yaml: Path,
    output_root: Path,
    source_split: str,
    folds: int,
    seed: int,
    link_mode: str,
    max_images: int = 0,
    overwrite: bool = False,
) -> Dict[str, object]:
    samples, class_names, images_dir, labels_dir = _scan_yolo_samples(
        data_yaml,
        source_split=source_split,
        max_images=max_images,
    )
    x, y, groups = _expanded_object_targets(samples)
    min_class_count = min(y.count(index) for index in range(len(class_names)))
    n_splits = min(int(folds), int(min_class_count), len(samples))
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 folds; images={len(samples)}, min object class count={min_class_count}."
        )

    _prepare_output_root(output_root, overwrite=overwrite)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    materialized_counts: Dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    all_indices = set(range(len(samples)))
    fold_summaries: List[Dict[str, object]] = []

    for fold_index, (_, val_object_indices) in enumerate(splitter.split(x, y, groups)):
        val_indices = {int(groups[int(object_index)]) for object_index in val_object_indices}
        train_indices = all_indices.difference(val_indices)
        if train_indices.intersection(val_indices):
            raise ValueError(f"Fold {fold_index} has image leakage between train and val.")

        fold_root = output_root / f"fold_{fold_index:02d}"
        _write_fold_data_yaml(fold_root, class_names)
        for split_name, indices in (("train", train_indices), ("val", val_indices), ("test", val_indices)):
            _materialize_split(
                fold_root=fold_root,
                split_name=split_name,
                samples=samples,
                indices=indices,
                link_mode=link_mode,
                materialized_counts=materialized_counts,
            )

        train_class_counts = _class_counts(samples, train_indices, len(class_names))
        val_class_counts = _class_counts(samples, val_indices, len(class_names))
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "root": str(fold_root),
                "data_yaml": str(fold_root / "data.yaml"),
                "train_image_count": int(len(train_indices)),
                "val_image_count": int(len(val_indices)),
                "test_image_count": int(len(val_indices)),
                "train_object_count": int(sum(train_class_counts)),
                "val_object_count": int(sum(val_class_counts)),
                "test_object_count": int(sum(val_class_counts)),
                "train_class_object_counts": train_class_counts,
                "val_class_object_counts": val_class_counts,
                "test_class_object_counts": val_class_counts,
                "val_stems": [samples[index].stem for index in sorted(val_indices)],
            }
        )

    source_class_counts = _class_counts(samples, all_indices, len(class_names))
    summary = {
        "data": str(data_yaml),
        "source_split": str(source_split),
        "source_images_dir": str(images_dir),
        "source_labels_dir": str(labels_dir),
        "output_root": str(output_root),
        "folds": int(n_splits),
        "seed": int(seed),
        "link_mode": str(link_mode),
        "class_names": list(class_names),
        "source_image_count": int(len(samples)),
        "source_object_count": int(sum(source_class_counts)),
        "source_class_object_counts": source_class_counts,
        "max_images": int(max_images),
        "materialized_counts": materialized_counts,
        "folds_detail": fold_summaries,
        "leakage_note": (
            "All fold images come from the source split only. StratifiedGroupKFold groups by image, "
            "so every source image belongs to exactly one fold/val set."
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_folds(
        data_yaml=args.data,
        output_root=args.output_root,
        source_split=str(args.source_split),
        folds=int(args.folds),
        seed=int(args.seed),
        link_mode=str(args.link_mode),
        max_images=int(args.max_images),
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "folds": summary["folds"],
                "source_image_count": summary["source_image_count"],
                "source_object_count": summary["source_object_count"],
                "source_class_object_counts": summary["source_class_object_counts"],
                "materialized_counts": summary["materialized_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

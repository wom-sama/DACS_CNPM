from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from sklearn.model_selection import StratifiedKFold

from trkh.core.config import load_data_spec


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    class_index: int
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build stratified train-only OOF ImageFolder folds from a classification-folder data.yaml. "
            "Each fold can be used with the TIMM baseline trainer: train on fold/train and export fold/val."
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
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Optional smoke cap before splitting. 0 keeps all source-split samples.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scan_samples(data_yaml: Path, split: str, max_samples_per_class: int = 0) -> tuple[List[Sample], List[str], Path]:
    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    if data_spec.data_format != "classification_folder":
        raise ValueError("build_classification_oof_folds expects classification_folder data.")
    split_dir = data_spec.split_images_dir(split)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    class_names = [str(value) for value in data_spec.class_names]
    samples: List[Sample] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        class_samples = [
            Sample(path=path, class_index=class_index, class_name=class_name)
            for path in sorted(class_dir.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_samples_per_class > 0:
            class_samples = class_samples[: int(max_samples_per_class)]
        samples.extend(class_samples)
    if not samples:
        raise ValueError(f"No image samples found in {split_dir}")
    return samples, class_names, data_spec.root


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
        "train": "train",
        "val": "val",
        "test": "test",
        "names": {index: name for index, name in enumerate(class_names)},
        "format": "classification_folder",
        "note": (
            "OOF fold built from train-only source samples. The fold/test directory mirrors "
            "fold/val only to satisfy generic trainers when --skip-test is used."
        ),
    }
    (fold_root / "data.yaml").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _destination_for(fold_root: Path, split: str, sample: Sample) -> Path:
    return fold_root / split / sample.class_name / sample.path.name


def build_folds(
    *,
    data_yaml: Path,
    output_root: Path,
    source_split: str,
    folds: int,
    seed: int,
    link_mode: str,
    max_samples_per_class: int = 0,
    overwrite: bool = False,
) -> Dict[str, object]:
    samples, class_names, source_root = _scan_samples(
        data_yaml,
        source_split,
        max_samples_per_class=max_samples_per_class,
    )
    labels = [sample.class_index for sample in samples]
    min_class_count = min(labels.count(index) for index in range(len(class_names)))
    n_splits = min(int(folds), int(min_class_count))
    if n_splits < 2:
        raise ValueError(f"Need at least 2 folds; min class count is {min_class_count}.")

    _prepare_output_root(output_root, overwrite=overwrite)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    materialized_counts: Dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    fold_summaries: List[Dict[str, object]] = []
    for fold_index, (train_indices, val_indices) in enumerate(splitter.split(samples, labels)):
        fold_root = output_root / f"fold_{fold_index:02d}"
        _write_fold_data_yaml(fold_root, class_names)
        fold_counts = {
            "train": [0 for _ in class_names],
            "val": [0 for _ in class_names],
            "test": [0 for _ in class_names],
        }
        for split_name, indices in (("train", train_indices), ("val", val_indices), ("test", val_indices)):
            for sample_index in indices:
                sample = samples[int(sample_index)]
                mode = _materialize_file(
                    sample.path,
                    _destination_for(fold_root, split_name, sample),
                    link_mode=link_mode,
                )
                materialized_counts[mode] = int(materialized_counts.get(mode, 0)) + 1
                fold_counts[split_name][sample.class_index] += 1
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "root": str(fold_root),
                "data_yaml": str(fold_root / "data.yaml"),
                "train_count": int(len(train_indices)),
                "val_count": int(len(val_indices)),
                "test_count": int(len(val_indices)),
                "class_counts": fold_counts,
            }
        )

    summary = {
        "data": str(data_yaml),
        "source_root": str(source_root),
        "source_split": str(source_split),
        "output_root": str(output_root),
        "folds": int(n_splits),
        "seed": int(seed),
        "link_mode": str(link_mode),
        "class_names": list(class_names),
        "source_sample_count": int(len(samples)),
        "max_samples_per_class": int(max_samples_per_class),
        "materialized_counts": materialized_counts,
        "folds_detail": fold_summaries,
        "leakage_note": (
            "All train/val fold samples come from the source split only. Use fold/val predictions "
            "as OOF train probabilities; do not mix final test into this workflow."
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
        max_samples_per_class=int(args.max_samples_per_class),
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "folds": summary["folds"],
                "source_sample_count": summary["source_sample_count"],
                "materialized_counts": summary["materialized_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from trkh.data.dataset import PairedViewTrainDataset
from trkh.tools.remap_classification_teacher_to_yolo import (
    _build_yolo_dataset,
    _classification_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remap classification_folder teacher feature NPZ to YOLO object-crop "
            "sample_index order for feature/RKD distillation."
        )
    )
    parser.add_argument("--teacher-feature-npz", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--strict", action="store_true", default=False)
    return parser.parse_args()


def _string_array(values: object) -> List[str]:
    array = np.asarray(values)
    return [str(value) for value in array.reshape(-1).tolist()]


def _load_feature_rows(npz_path: Path) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "features" not in data:
            raise ValueError(f"Teacher feature NPZ thieu 'features': {npz_path}")
        features = np.asarray(data["features"], dtype=np.float32)
        paths: List[str] = []
        for name in ("paths", "path", "image_paths", "image_path"):
            if name in data:
                paths = _string_array(data[name])
                break
        classes = _string_array(data["classes"]) if "classes" in data else []
        feature_source = _string_array(data["feature_source"]) if "feature_source" in data else []
    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError(f"Teacher feature NPZ features phai co shape [N, D], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Teacher feature NPZ co feature khong huu han: {npz_path}")
    if len(paths) != int(features.shape[0]):
        raise ValueError(
            f"Teacher feature NPZ paths length khong khop features: {len(paths)} != {features.shape[0]}"
        )
    return features, paths, {
        "teacher_rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "classes": classes,
        "feature_source": feature_source,
    }


def _features_by_classification_key(
    features: np.ndarray,
    paths: List[str],
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[str, object]]:
    by_key: Dict[Tuple[str, int], np.ndarray] = {}
    duplicate_keys = 0
    object_counts: Counter[int] = Counter()
    for feature, path_text in zip(features, paths):
        key = _classification_key(path_text)
        if key in by_key:
            duplicate_keys += 1
        by_key[key] = np.asarray(feature, dtype=np.float32)
        object_counts[int(key[1])] += 1
    return by_key, {
        "teacher_key_count": int(len(by_key)),
        "duplicate_teacher_keys": int(duplicate_keys),
        "teacher_object_index_counts": {
            str(index): int(count) for index, count in sorted(object_counts.items())
        },
    }


def remap_classification_features_to_yolo(
    *,
    teacher_feature_npz: Path,
    yolo_data: Path,
    split: str,
    output_npz: Path,
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
    teacher_features, teacher_paths, source_summary = _load_feature_rows(teacher_feature_npz)
    by_key, key_summary = _features_by_classification_key(teacher_features, teacher_paths)

    paths = [Path(path) for path in dataset.sample_paths()]
    labels = [int(label) for label in dataset.labels()]
    output_features: List[np.ndarray] = []
    output_sample_indices: List[int] = []
    output_labels: List[int] = []
    output_paths: List[str] = []
    output_source_stems: List[str] = []
    output_object_indices: List[int] = []
    missing: List[Dict[str, object]] = []

    for sample_index in range(len(dataset)):
        source_id, object_index, fallback_label = PairedViewTrainDataset._sample_key(dataset, sample_index)
        feature = by_key.get((str(source_id), int(object_index)))
        target_index = int(labels[sample_index]) if sample_index < len(labels) else int(fallback_label)
        path = paths[sample_index] if sample_index < len(paths) else Path(str(source_id))
        if feature is None:
            missing.append(
                {
                    "sample_index": int(sample_index),
                    "path": str(path),
                    "source_stem": str(source_id),
                    "object_index": int(object_index),
                    "target_index": int(target_index),
                }
            )
            continue
        output_features.append(np.asarray(feature, dtype=np.float32))
        output_sample_indices.append(int(sample_index))
        output_labels.append(int(target_index))
        output_paths.append(str(path))
        output_source_stems.append(str(source_id))
        output_object_indices.append(int(object_index))

    if missing and strict:
        raise ValueError(f"Missing {len(missing)} teacher feature mappings; preview={missing[:5]}")
    if not output_features:
        raise ValueError("No teacher feature rows mapped to YOLO samples.")

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        features=np.stack(output_features, axis=0).astype(np.float32),
        sample_index=np.asarray(output_sample_indices, dtype=np.int64),
        labels=np.asarray(output_labels, dtype=np.int64),
        paths=np.asarray(output_paths, dtype=object),
        source_stem=np.asarray(output_source_stems, dtype=object),
        object_index=np.asarray(output_object_indices, dtype=np.int64),
        classes=np.asarray(target_class_names, dtype=object),
        source_feature_npz=np.asarray([str(Path(teacher_feature_npz).resolve())], dtype=object),
    )

    label_counts: Counter[int] = Counter(output_labels)
    summary = {
        "teacher_feature_npz": str(Path(teacher_feature_npz).resolve()),
        "yolo_data": str(Path(yolo_data).resolve()),
        "split": str(split),
        "output_npz": str(output_npz.resolve()),
        "yolo_samples": int(len(dataset)),
        "mapped_rows": int(len(output_features)),
        "missing_rows": int(len(missing)),
        "target_class_names": [str(name) for name in target_class_names],
        "mapped_label_counts": {
            str(index): int(label_counts[index]) for index in range(len(target_class_names))
        },
        "missing_examples": missing[:10],
        **source_summary,
        **key_summary,
    }
    output_npz.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = remap_classification_features_to_yolo(
        teacher_feature_npz=args.teacher_feature_npz,
        yolo_data=args.yolo_data,
        split=args.split,
        output_npz=args.output_npz,
        class_name_mode=args.class_name_mode,
        expected_num_classes=int(args.expected_num_classes),
        strict=bool(args.strict),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

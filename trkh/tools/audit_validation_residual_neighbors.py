from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


IMAGE_NUMBER = re.compile(r"^Image_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class FeatureSpaceSpec:
    name: str
    train_path: Path
    val_path: Path


@dataclass(frozen=True)
class FeatureCache:
    features: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    sample_indices: np.ndarray
    source_stems: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit validation ceiling residuals against train neighbors and numeric "
            "source sequences in multiple feature spaces. Train+val only; test is rejected."
        )
    )
    parser.add_argument("--residual-csv", type=Path, required=True)
    parser.add_argument(
        "--space",
        action="append",
        required=True,
        metavar="NAME=TRAIN_NPZ,VAL_NPZ",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--sequence-window", type=int, default=5)
    parser.add_argument("--expected-train-samples", type=int, default=9215)
    parser.add_argument("--expected-val-samples", type=int, default=2606)
    return parser.parse_args()


def parse_space_spec(value: str) -> FeatureSpaceSpec:
    if "=" not in value:
        raise ValueError(f"Feature space must be NAME=TRAIN_NPZ,VAL_NPZ: {value!r}")
    name, paths = value.split("=", 1)
    path_parts = [Path(part.strip()) for part in paths.split(",") if part.strip()]
    if not name.strip() or len(path_parts) != 2:
        raise ValueError(f"Feature space must be NAME=TRAIN_NPZ,VAL_NPZ: {value!r}")
    for path in path_parts:
        if not path.is_file():
            raise FileNotFoundError(f"Feature cache does not exist: {path}")
    return FeatureSpaceSpec(name=name.strip(), train_path=path_parts[0], val_path=path_parts[1])


def _path_split(path_text: str) -> str | None:
    parts = [part for part in re.split(r"[\\/]", str(path_text).lower()) if part]
    found = [split for split in ("train", "val", "test") if split in parts]
    if len(found) > 1:
        raise ValueError(f"Ambiguous split path: {path_text}")
    return found[0] if found else None


def _source_stem(path_text: str) -> str:
    stem = Path(str(path_text)).stem
    crop_match = re.match(r"(.+?)_box\d+$", stem, flags=re.IGNORECASE)
    return crop_match.group(1) if crop_match else stem


def _feature_key(cache: np.lib.npyio.NpzFile, path: Path) -> str:
    for key in ("features", "embeddings"):
        if key in cache.files:
            return key
    raise ValueError(f"Feature cache has neither features nor embeddings: {path}")


def _load_feature_cache(
    path: Path,
    *,
    required_split: str,
    expected_samples: int,
) -> FeatureCache:
    with np.load(path, allow_pickle=True) as cache:
        for key in ("labels", "paths"):
            if key not in cache.files:
                raise ValueError(f"Feature cache missing {key}: {path}")
        feature_values = np.asarray(cache[_feature_key(cache, path)], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=np.int64)
        paths = np.asarray(cache["paths"], dtype=object)
        sample_indices = (
            np.asarray(cache["sample_index"], dtype=np.int64)
            if "sample_index" in cache.files
            else np.arange(labels.shape[0], dtype=np.int64)
        )
        source_stems = (
            np.asarray(cache["source_stem"], dtype=object)
            if "source_stem" in cache.files
            else np.asarray([_source_stem(value) for value in paths], dtype=object)
        )
    row_count = int(labels.shape[0])
    if feature_values.ndim != 2:
        raise ValueError(f"Features must be 2D: {path} -> {feature_values.shape}")
    for name, values in (
        ("features", feature_values),
        ("paths", paths),
        ("sample_index", sample_indices),
        ("source_stem", source_stems),
    ):
        if int(values.shape[0]) != row_count:
            raise ValueError(f"Feature cache {name} length mismatch: {path}")
    if expected_samples > 0 and row_count != int(expected_samples):
        raise ValueError(f"Expected {expected_samples} {required_split} rows, found {row_count}: {path}")
    order = np.argsort(sample_indices)
    sorted_indices = sample_indices[order]
    if not np.array_equal(sorted_indices, np.arange(row_count, dtype=np.int64)):
        raise ValueError(f"sample_index must be contiguous from 0: {path}")
    paths = paths[order]
    for row_index, path_text in enumerate(paths.tolist()):
        split = _path_split(str(path_text))
        if split != required_split:
            raise ValueError(
                f"Train+val-only guard rejected {path} row {row_index}: "
                f"required={required_split}, found={split}, path={path_text}"
            )
    return FeatureCache(
        features=feature_values[order],
        labels=labels[order],
        paths=paths,
        sample_indices=sorted_indices,
        source_stems=source_stems[order],
    )


def _read_residuals(path: Path, val_labels: np.ndarray) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Residual CSV is empty: {path}")
    residuals: list[dict[str, object]] = []
    seen: set[int] = set()
    for row in rows:
        sample_index = int(row["sample_index"])
        target_index = int(row["target_index"])
        if sample_index in seen:
            raise ValueError(f"Duplicate residual sample_index: {sample_index}")
        if not 0 <= sample_index < val_labels.shape[0]:
            raise ValueError(f"Residual sample_index out of range: {sample_index}")
        if target_index != int(val_labels[sample_index]):
            raise ValueError(
                f"Residual target mismatch at sample {sample_index}: "
                f"{target_index} != {val_labels[sample_index]}"
            )
        seen.add(sample_index)
        residuals.append(
            {
                **row,
                "sample_index": sample_index,
                "target_index": target_index,
                "base_prediction_index": int(row["base_prediction_index"]),
            }
        )
    return residuals


def _normalize(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def _image_number(source_stem: str) -> int | None:
    match = IMAGE_NUMBER.match(str(source_stem))
    return int(match.group(1)) if match else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_validation_residual_neighbors(
    *,
    residual_csv: Path,
    feature_spaces: Sequence[FeatureSpaceSpec],
    output_dir: Path,
    top_k: int = 15,
    sequence_window: int = 5,
    expected_train_samples: int = 9215,
    expected_val_samples: int = 2606,
) -> Dict[str, object]:
    if not feature_spaces:
        raise ValueError("At least one feature space is required")
    if len({space.name for space in feature_spaces}) != len(feature_spaces):
        raise ValueError("Feature space names must be unique")
    if not 0 < int(top_k) < int(expected_train_samples or 2**31):
        raise ValueError("top_k must be positive and smaller than train rows")
    if int(sequence_window) < 1:
        raise ValueError("sequence_window must be positive")

    caches: dict[str, tuple[FeatureCache, FeatureCache]] = {}
    reference_train_labels: np.ndarray | None = None
    reference_val_labels: np.ndarray | None = None
    reference_train_paths: np.ndarray | None = None
    reference_val_paths: np.ndarray | None = None
    for space in feature_spaces:
        train = _load_feature_cache(
            space.train_path,
            required_split="train",
            expected_samples=int(expected_train_samples),
        )
        val = _load_feature_cache(
            space.val_path,
            required_split="val",
            expected_samples=int(expected_val_samples),
        )
        if int(top_k) >= train.labels.shape[0]:
            raise ValueError(f"top_k must be smaller than train rows for {space.name}")
        if reference_train_labels is None:
            reference_train_labels = train.labels
            reference_val_labels = val.labels
            reference_train_paths = train.paths
            reference_val_paths = val.paths
        else:
            if not np.array_equal(train.labels, reference_train_labels):
                raise ValueError(f"Train target mismatch across feature spaces: {space.name}")
            if not np.array_equal(val.labels, reference_val_labels):
                raise ValueError(f"Validation target mismatch across feature spaces: {space.name}")
            if not np.array_equal(train.paths.astype(str), reference_train_paths.astype(str)):
                raise ValueError(f"Train path/order mismatch across feature spaces: {space.name}")
            if not np.array_equal(val.paths.astype(str), reference_val_paths.astype(str)):
                raise ValueError(f"Validation path/order mismatch across feature spaces: {space.name}")
        caches[space.name] = (train, val)
    assert reference_train_labels is not None
    assert reference_val_labels is not None
    residuals = _read_residuals(Path(residual_csv), reference_val_labels)

    train_neighbor_rows: list[dict[str, object]] = []
    residual_summary_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    per_space_summary: dict[str, dict[str, object]] = {}
    residual_by_index = {int(row["sample_index"]): row for row in residuals}

    for space in feature_spaces:
        train, val = caches[space.name]
        train_features = _normalize(train.features)
        val_features = _normalize(val.features)
        sample_indices = np.asarray(sorted(residual_by_index), dtype=np.int64)
        similarity = val_features[sample_indices] @ train_features.T
        for query_row, sample_index in enumerate(sample_indices.tolist()):
            same_source = np.asarray(
                [
                    str(stem).lower() == str(val.source_stems[sample_index]).lower()
                    for stem in train.source_stems
                ],
                dtype=bool,
            )
            similarity[query_row, same_source] = -np.inf
        candidate_indices = np.argpartition(-similarity, kth=int(top_k) - 1, axis=1)[:, : int(top_k)]
        candidate_scores = np.take_along_axis(similarity, candidate_indices, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        neighbor_indices = np.take_along_axis(candidate_indices, order, axis=1)
        neighbor_scores = np.take_along_axis(candidate_scores, order, axis=1)

        target_fractions: list[float] = []
        predicted_fractions: list[float] = []
        zero_target_support = 0
        for query_row, sample_index in enumerate(sample_indices.tolist()):
            residual = residual_by_index[sample_index]
            target = int(residual["target_index"])
            base_prediction = int(residual["base_prediction_index"])
            labels = train.labels[neighbor_indices[query_row]]
            target_fraction = float(np.mean(labels == target))
            predicted_fraction = float(np.mean(labels == base_prediction))
            target_fractions.append(target_fraction)
            predicted_fractions.append(predicted_fraction)
            zero_target_support += int(target_fraction == 0.0)
            counts = np.bincount(labels, minlength=int(max(train.labels.max(), val.labels.max())) + 1)
            majority = int(np.argmax(counts))
            residual_summary_rows.append(
                {
                    "space": space.name,
                    "sample_index": sample_index,
                    "target_index": target,
                    "base_prediction_index": base_prediction,
                    "residual_kind": residual.get("focus_binary_residual_kind", ""),
                    "top_k": int(top_k),
                    "neighbor_majority_index": majority,
                    "neighbor_target_fraction": target_fraction,
                    "neighbor_base_prediction_fraction": predicted_fraction,
                    "nearest_similarity": float(neighbor_scores[query_row, 0]),
                    "mean_similarity": float(np.mean(neighbor_scores[query_row])),
                }
            )
            for rank, neighbor_index in enumerate(neighbor_indices[query_row].tolist(), start=1):
                train_neighbor_rows.append(
                    {
                        "space": space.name,
                        "sample_index": sample_index,
                        "target_index": target,
                        "base_prediction_index": base_prediction,
                        "rank": rank,
                        "similarity": float(neighbor_scores[query_row, rank - 1]),
                        "neighbor_sample_index": int(train.sample_indices[neighbor_index]),
                        "neighbor_label": int(train.labels[neighbor_index]),
                        "neighbor_matches_target": int(train.labels[neighbor_index] == target),
                        "neighbor_matches_base_prediction": int(
                            train.labels[neighbor_index] == base_prediction
                        ),
                        "neighbor_source_stem": str(train.source_stems[neighbor_index]),
                        "neighbor_path": str(train.paths[neighbor_index]),
                    }
                )

        combined = (("train", train), ("val", val))
        for sample_index, residual in residual_by_index.items():
            query_number = _image_number(str(val.source_stems[sample_index]))
            if query_number is None:
                continue
            target = int(residual["target_index"])
            query_feature = val_features[sample_index]
            for split, cache in combined:
                normalized = train_features if split == "train" else val_features
                for candidate_index, source_stem in enumerate(cache.source_stems.tolist()):
                    candidate_number = _image_number(str(source_stem))
                    if candidate_number is None:
                        continue
                    distance = abs(candidate_number - query_number)
                    if distance == 0 or distance > int(sequence_window):
                        continue
                    candidate_label = int(cache.labels[candidate_index])
                    sequence_rows.append(
                        {
                            "space": space.name,
                            "sample_index": sample_index,
                            "target_index": target,
                            "base_prediction_index": int(residual["base_prediction_index"]),
                            "query_source_stem": str(val.source_stems[sample_index]),
                            "neighbor_split": split,
                            "neighbor_sample_index": int(cache.sample_indices[candidate_index]),
                            "neighbor_source_stem": str(source_stem),
                            "numeric_distance": distance,
                            "neighbor_label": candidate_label,
                            "label_conflict": int(candidate_label != target),
                            "similarity": float(query_feature @ normalized[candidate_index]),
                            "neighbor_path": str(cache.paths[candidate_index]),
                        }
                    )
        per_space_summary[space.name] = {
            "residual_rows": len(residuals),
            "top_k": int(top_k),
            "mean_neighbor_target_fraction": float(np.mean(target_fractions)),
            "mean_neighbor_base_prediction_fraction": float(np.mean(predicted_fractions)),
            "zero_target_neighbor_support_rows": int(zero_target_support),
            "majority_matches_base_prediction_rows": int(
                sum(
                    int(row["neighbor_majority_index"] == row["base_prediction_index"])
                    for row in residual_summary_rows
                    if row["space"] == space.name
                )
            ),
        }

    unique_sequence_pairs: dict[tuple[int, str, int], dict[str, object]] = {}
    for row in sequence_rows:
        key = (
            int(row["sample_index"]),
            str(row["neighbor_split"]),
            int(row["neighbor_sample_index"]),
        )
        payload = unique_sequence_pairs.setdefault(
            key,
            {
                "sample_index": int(row["sample_index"]),
                "target_index": int(row["target_index"]),
                "neighbor_split": str(row["neighbor_split"]),
                "neighbor_sample_index": int(row["neighbor_sample_index"]),
                "neighbor_label": int(row["neighbor_label"]),
                "label_conflict": int(row["label_conflict"]),
                "max_similarity": float(row["similarity"]),
            },
        )
        payload["max_similarity"] = max(float(payload["max_similarity"]), float(row["similarity"]))
    conflict_pairs = [row for row in unique_sequence_pairs.values() if int(row["label_conflict"]) == 1]
    conflict_residuals = {int(row["sample_index"]) for row in conflict_pairs}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "residual_neighbor_summary.csv", residual_summary_rows)
    _write_csv(output_dir / "train_neighbors.csv", train_neighbor_rows)
    _write_csv(output_dir / "sequence_neighbors.csv", sequence_rows)
    _write_csv(output_dir / "sequence_label_conflicts_unique.csv", conflict_pairs)

    summary: Dict[str, object] = {
        "audit_mode": "train_val_only_validation_ceiling_residual_neighbors",
        "residual_csv": str(Path(residual_csv).resolve()),
        "residual_rows": len(residuals),
        "residual_sample_indices": sorted(residual_by_index),
        "feature_spaces": [
            {
                "name": space.name,
                "train_path": str(space.train_path.resolve()),
                "val_path": str(space.val_path.resolve()),
            }
            for space in feature_spaces
        ],
        "per_space": per_space_summary,
        "sequence_window": int(sequence_window),
        "unique_sequence_neighbors": len(unique_sequence_pairs),
        "cross_label_sequence_neighbors": len(conflict_pairs),
        "residuals_with_cross_label_sequence_neighbor": len(conflict_residuals),
        "cross_label_sequence_pairs_by_max_similarity": {
            str(threshold): int(
                sum(float(row["max_similarity"]) >= threshold for row in conflict_pairs)
            )
            for threshold in (0.90, 0.95, 0.98)
        },
        "guardrails": {
            "raw_dataset_touched": False,
            "test_split_used": False,
            "test_paths_accepted": False,
            "trainable_manifest_written": False,
            "training_permission": False,
            "threshold_fitted": False,
        },
        "outputs": {
            "residual_neighbor_summary": "residual_neighbor_summary.csv",
            "train_neighbors": "train_neighbors.csv",
            "sequence_neighbors": "sequence_neighbors.csv",
            "sequence_label_conflicts_unique": "sequence_label_conflicts_unique.csv",
        },
        "notes": [
            "Numeric Image_N proximity is acquisition metadata, not a deployable visual feature.",
            "Neighbor labels and sequence conflicts diagnose ambiguity; they do not authorize relabeling.",
            "This audit reads train and validation feature caches only and rejects test paths.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = audit_validation_residual_neighbors(
        residual_csv=Path(args.residual_csv),
        feature_spaces=[parse_space_spec(value) for value in args.space],
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        sequence_window=int(args.sequence_window),
        expected_train_samples=int(args.expected_train_samples),
        expected_val_samples=int(args.expected_val_samples),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from trkh.tools.audit_validation_residual_neighbors import (
    FeatureSpaceSpec,
    audit_validation_residual_neighbors,
)


def _write_cache(
    path: Path,
    *,
    split: str,
    features: np.ndarray,
    labels: list[int],
    image_numbers: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        paths=np.asarray(
            [f"D:/dataset/images/{split}/Image_{number}.jpg" for number in image_numbers],
            dtype=object,
        ),
        source_stem=np.asarray([f"Image_{number}" for number in image_numbers], dtype=object),
        sample_index=np.arange(len(labels), dtype=np.int64),
    )


def _write_residual(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "target_index",
                "base_prediction_index",
                "focus_binary_residual_kind",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_index": 0,
                "target_index": 1,
                "base_prediction_index": 2,
                "focus_binary_residual_kind": "fn",
            }
        )


def test_residual_neighbor_audit_reports_cross_label_sequence(tmp_path: Path) -> None:
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    residual_path = tmp_path / "residual.csv"
    _write_cache(
        train_path,
        split="train",
        features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
        labels=[2, 2, 1],
        image_numbers=[101, 150, 300],
    )
    _write_cache(
        val_path,
        split="val",
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        labels=[1, 0],
        image_numbers=[100, 400],
    )
    _write_residual(residual_path)

    summary = audit_validation_residual_neighbors(
        residual_csv=residual_path,
        feature_spaces=[FeatureSpaceSpec("toy", train_path, val_path)],
        output_dir=tmp_path / "out",
        top_k=2,
        sequence_window=2,
        expected_train_samples=3,
        expected_val_samples=2,
    )

    assert summary["guardrails"]["test_split_used"] is False
    assert summary["per_space"]["toy"]["majority_matches_base_prediction_rows"] == 1
    assert summary["residuals_with_cross_label_sequence_neighbor"] == 1
    assert (tmp_path / "out" / "sequence_label_conflicts_unique.csv").is_file()


def test_residual_neighbor_audit_rejects_test_cache(tmp_path: Path) -> None:
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    residual_path = tmp_path / "residual.csv"
    _write_cache(
        train_path,
        split="train",
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        labels=[0, 1],
        image_numbers=[1, 2],
    )
    _write_cache(
        test_path,
        split="test",
        features=np.asarray([[1.0, 0.0]]),
        labels=[1],
        image_numbers=[3],
    )
    _write_residual(residual_path)

    with pytest.raises(ValueError, match=r"Train\+val-only guard"):
        audit_validation_residual_neighbors(
            residual_csv=residual_path,
            feature_spaces=[FeatureSpaceSpec("toy", train_path, test_path)],
            output_dir=tmp_path / "out",
            top_k=1,
            sequence_window=2,
            expected_train_samples=2,
            expected_val_samples=1,
        )

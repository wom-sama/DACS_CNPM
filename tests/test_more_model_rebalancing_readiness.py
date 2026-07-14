from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from trkh.models.model_rebalancing import inject_model_rebalancing_convs
from trkh.tools.audit_more_model_rebalancing_readiness import (
    _base_parameter_sha256,
    _ordered_index_sha256,
    _prepare_output_dir,
    _read_clean_train_rows,
)


def _write_rows(path: Path) -> None:
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target_index",
        "keeper_prediction",
        "keeper_prob_0",
        "keeper_prob_1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "condition": "clean",
                "sample_index": 0,
                "source_stem": "image_0",
                "image_path": str(path.parent / "train" / "Image_0.jpg"),
                "fold": 0,
                "target_index": 0,
                "keeper_prediction": 0,
                "keeper_prob_0": 0.8,
                "keeper_prob_1": 0.2,
            }
        )
        writer.writerow(
            {
                "condition": "clean",
                "sample_index": 1,
                "source_stem": "image_1",
                "image_path": str(path.parent / "train" / "Image_1.jpg"),
                "fold": 1,
                "target_index": 1,
                "keeper_prediction": 1,
                "keeper_prob_0": 0.1,
                "keeper_prob_1": 0.9,
            }
        )
        writer.writerow(
            {
                "condition": "lighting_dim",
                "sample_index": 0,
                "source_stem": "image_0",
                "image_path": str(path.parent / "train" / "Image_0.jpg"),
                "fold": 0,
                "target_index": 0,
                "keeper_prediction": 0,
                "keeper_prob_0": 0.7,
                "keeper_prob_1": 0.3,
            }
        )


def test_read_clean_rows_validates_order_counts_and_train_paths(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_rows(path)
    rows = _read_clean_train_rows(
        path,
        expected_rows=2,
        expected_class_counts=(1, 1),
        expected_fold_counts=(1, 1),
    )
    assert [row.sample_index for row in rows] == [0, 1]
    assert [row.fold for row in rows] == [0, 1]


def test_read_clean_rows_rejects_non_normalized_probabilities(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_rows(path)
    text = path.read_text(encoding="utf-8").replace("0.8,0.2", "0.8,0.4", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not normalized"):
        _read_clean_train_rows(
            path,
            expected_rows=2,
            expected_class_counts=(1, 1),
            expected_fold_counts=(1, 1),
        )


def test_output_dir_must_be_empty(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    assert _prepare_output_dir(output) == output.resolve()
    (output / "existing.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        _prepare_output_dir(output)


def test_ordered_index_hash_is_order_sensitive() -> None:
    assert _ordered_index_sha256([1, 2, 3]) != _ordered_index_sha256([3, 2, 1])


def test_base_parameter_hash_ignores_tail_but_detects_base_change() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1))
    inject_model_rebalancing_convs(model, rank_ratio=0.2, seed=42)
    before = _base_parameter_sha256(model)
    with torch.no_grad():
        model[0].tail_b.weight.add_(1.0)
    assert _base_parameter_sha256(model) == before
    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert _base_parameter_sha256(model) != before

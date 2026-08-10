from __future__ import annotations

import numpy as np
import torch

from trkh.tools.audit_dinov3_pair_patch_stat_readiness import (
    _local_patch_statistics,
    _pair_oof_comparison,
    assess_patch_stat_readiness,
    normalized_source_group,
)


def test_normalized_source_group_removes_crop_suffix() -> None:
    assert normalized_source_group(r"C:\data\Image_123_box000.JPG") == "image_123"
    assert normalized_source_group(r"C:\data\Image_123_box004.png") == "image_123"
    assert normalized_source_group(r"C:\data\plain_name.jpg") == "plain_name"


def test_local_patch_statistics_are_finite_and_fixed_width() -> None:
    logits = torch.randn(3, 16, 5)
    features = _local_patch_statistics(logits)
    assert features.shape == (3, 70)
    assert torch.isfinite(features).all()


def test_patch_candidate_uses_disjoint_groups_and_adds_signal() -> None:
    rng = np.random.default_rng(42)
    labels = np.tile(np.asarray([0, 1, 2, 1, 4, 1], dtype=np.int64), 40)
    groups = np.asarray([f"source_{index}" for index in range(labels.size)], dtype=object)
    pooled = rng.normal(size=(labels.size, 4)).astype(np.float32)
    patch = rng.normal(scale=0.15, size=(labels.size, 3)).astype(np.float32)
    patch[:, 0] += np.where(labels == 1, 2.0, -2.0).astype(np.float32)
    rows, predictions = _pair_oof_comparison(
        pooled_features=pooled,
        patch_features=patch,
        labels=labels,
        source_groups=groups,
    )
    assert len(rows) == 3
    assert len(predictions) == sum(
        int(np.logical_or(labels == rival, labels == 1).sum()) for rival in (0, 2, 4)
    )
    assert all(int(row["max_source_overlap"]) == 0 for row in rows)
    assert all(float(row["delta_balanced_accuracy"]) > 0.20 for row in rows)


def test_readiness_gate_rejects_recall_harm() -> None:
    rows = []
    for rival in (0, 2, 4):
        rows.append(
            {
                "rival_class": rival,
                "delta_balanced_accuracy": 0.02,
                "delta_auroc": 0.02,
                "delta_specificity_rival": 0.04,
                "delta_recall_class1": -0.03 if rival == 4 else 0.0,
                "max_source_overlap": 0,
            }
        )
    result = assess_patch_stat_readiness(
        pair_rows=rows,
        train_samples=8278,
        source_groups=7751,
        feature_finite=True,
    )
    assert not result["patch_stat_specialist_ready"]
    assert "class1_recall_protected" in result["failed_checks"]


def test_readiness_gate_allows_matched_patch_gain() -> None:
    rows = []
    for rival in (0, 2, 4):
        rows.append(
            {
                "rival_class": rival,
                "delta_balanced_accuracy": 0.015,
                "delta_auroc": 0.012,
                "delta_specificity_rival": 0.03,
                "delta_recall_class1": 0.0,
                "max_source_overlap": 0,
            }
        )
    result = assess_patch_stat_readiness(
        pair_rows=rows,
        train_samples=8278,
        source_groups=7751,
        feature_finite=True,
    )
    assert result["patch_stat_specialist_ready"]
    assert result["failed_checks"] == []

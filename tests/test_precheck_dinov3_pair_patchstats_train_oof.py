from __future__ import annotations

import numpy as np
import torch

from trkh.tools.precheck_dinov3_pair_patchstats_train_oof import (
    assess_raw_patch_readiness,
    assign_global_source_folds,
    raw_patch_moment_descriptors,
)


def _passing_rows():
    pairs = []
    folds = []
    for rival in (0, 2, 4):
        pairs.append(
            {
                "rival_class": rival,
                "delta_auroc": 0.02,
                "delta_f1_class1": 0.02,
                "delta_recall_class1": 0.0,
                "control_tp": 100,
                "candidate_tp": 100,
                "control_fp": 40,
                "candidate_fp": 30,
            }
        )
        for fold in range(5):
            folds.append(
                {
                    "rival_class": rival,
                    "fold": fold,
                    "delta_auroc": 0.01,
                    "source_overlap": 0,
                }
            )
    global_folds = [{"fold": fold, "source_overlap": 0} for fold in range(5)]
    return pairs, folds, global_folds


def test_raw_patch_moment_descriptor_shape_and_pool() -> None:
    tokens = torch.randn(2, 261, 384)
    result = raw_patch_moment_descriptors(tokens, prefix_tokens=5)
    assert result["pooled"].shape == (2, 384)
    assert result["patch_std"].shape == (2, 384)
    assert result["spatial_means"].shape == (2, 1536)
    assert result["pooled_plus_patch_moments"].shape == (2, 2304)
    assert torch.equal(result["patch_grid"], torch.tensor([16, 16]))
    assert torch.allclose(result["pooled"], tokens[:, 5:].mean(dim=1))


def test_global_source_folds_keep_mixed_label_group_together() -> None:
    labels = np.asarray([0, 1, 2, 3, 4] * 20, dtype=np.int64)
    groups = np.asarray([f"group_{index}" for index in range(labels.size)], dtype=object)
    groups[0] = "shared"
    groups[1] = "shared"
    assignments, rows = assign_global_source_folds(labels, groups)
    assert assignments[0] == assignments[1]
    assert set(assignments.tolist()) == set(range(5))
    assert all(int(row["source_overlap"]) == 0 for row in rows)


def test_raw_patch_readiness_passes_locked_gain() -> None:
    pairs, folds, global_folds = _passing_rows()
    result = assess_raw_patch_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_folds,
        train_samples=8278,
        source_groups=7751,
        pool_parity_error=0.0,
    )
    assert result["raw_patch_moment_specialist_ready"]
    assert result["failed_checks"] == []


def test_raw_patch_readiness_rejects_false_positive_tradeoff() -> None:
    pairs, folds, global_folds = _passing_rows()
    for row in pairs:
        row["candidate_fp"] = 39
    result = assess_raw_patch_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_folds,
        train_samples=8278,
        source_groups=7751,
        pool_parity_error=0.0,
    )
    assert not result["raw_patch_moment_specialist_ready"]
    assert "rival_false_positives_reduced" in result["failed_checks"]

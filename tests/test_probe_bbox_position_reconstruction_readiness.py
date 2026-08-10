from __future__ import annotations

import pytest
import torch

from trkh.tools.probe_bbox_position_reconstruction_readiness import (
    RidgeAccumulator,
    assess_position_readiness,
    bbox_relative_patch_targets,
    position_metrics,
    ridge_predict,
    scale_patch_position_embeddings,
)


def test_patch_position_scaling_preserves_prefix_tokens() -> None:
    values = torch.arange(7, dtype=torch.float32).view(1, 7, 1)
    scaled = scale_patch_position_embeddings(values, (2, 2), 0.0)
    assert torch.equal(scaled[:, :3], values[:, :3])
    assert torch.count_nonzero(scaled[:, 3:]) == 0


def test_bbox_relative_targets_follow_patch_centers() -> None:
    local, targets = bbox_relative_patch_targets(
        patch_indices=torch.arange(16),
        grid_size=(4, 4),
        bbox=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        max_tokens=16,
    )
    assert local.tolist() == [5, 6, 9, 10]
    assert torch.allclose(
        targets,
        torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
    )


def test_ridge_accumulator_recovers_linear_targets() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((512, 3), generator=generator)
    weights = torch.tensor([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.25]])
    bias = torch.tensor([0.1, -0.2])
    targets = features @ weights + bias
    ridge = RidgeAccumulator(feature_dim=3)
    ridge.update(features, targets)
    coefficients = ridge.solve(l2=1e-8)
    predictions = ridge_predict(features, coefficients)
    assert torch.allclose(predictions, targets.to(dtype=torch.float64), atol=1e-6)


def test_position_metrics_report_perfect_prediction() -> None:
    targets = torch.tensor([[0.1, 0.2], [0.8, 0.7], [0.4, 0.6]])
    metrics = position_metrics(targets, targets)
    assert metrics["mean_mae"] == pytest.approx(0.0)
    assert metrics["mean_r2"] == pytest.approx(1.0)
    assert metrics["coarse_joint_accuracy"] == pytest.approx(1.0)


def test_position_gate_requires_decodability_and_noncatastrophic_ablation() -> None:
    ready = assess_position_readiness(
        metrics={
            "token_count": 80000,
            "mean_r2": 0.25,
            "coarse_joint_accuracy": 0.22,
            "mean_mae": 0.18,
        },
        class1_sample_mean_mae=0.20,
        baseline_macro_f1=0.88,
        ablated_macro_f1=0.70,
        changed_predictions=200,
        train_token_count=80000,
        val_support_complete=True,
    )
    assert ready["smoke_ready"] is True

    rejected = assess_position_readiness(
        metrics={
            "token_count": 80000,
            "mean_r2": -0.05,
            "coarse_joint_accuracy": 0.07,
            "mean_mae": 0.35,
        },
        class1_sample_mean_mae=0.50,
        baseline_macro_f1=0.88,
        ablated_macro_f1=0.40,
        changed_predictions=500,
        train_token_count=80000,
        val_support_complete=True,
    )
    assert rejected["smoke_ready"] is False
    assert "mean_r2" in rejected["failed_checks"]
    assert "ablation_not_catastrophic" in rejected["failed_checks"]

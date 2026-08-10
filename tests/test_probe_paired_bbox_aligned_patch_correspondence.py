from __future__ import annotations

import pytest
import torch

from trkh.tools.probe_paired_bbox_aligned_patch_correspondence import (
    assess_correspondence_gate,
    build_bbox_aligned_patch_match,
    correspondence_statistics,
    patch_centers_from_indices,
)


def test_patch_centers_use_xy_order() -> None:
    centers = patch_centers_from_indices(torch.tensor([0, 1, 4, 15]), (4, 4))
    assert torch.allclose(
        centers,
        torch.tensor([[0.125, 0.125], [0.375, 0.125], [0.125, 0.375], [0.875, 0.875]]),
    )


def test_full_bbox_identity_maps_same_grid_indices() -> None:
    indices = torch.arange(16)
    match = build_bbox_aligned_patch_match(
        primary_patch_indices=indices,
        paired_patch_indices=indices,
        primary_grid_size=(4, 4),
        paired_grid_size=(4, 4),
        primary_bbox=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        paired_bbox=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        max_pairs=16,
        max_geometric_distance=0.01,
    )
    assert match is not None
    assert torch.equal(match.primary_local_indices, match.paired_local_indices)
    assert float(match.geometric_distances.max()) == pytest.approx(0.0)


def test_bbox_relative_mapping_expands_object_crop_to_full_crop() -> None:
    indices = torch.arange(16)
    match = build_bbox_aligned_patch_match(
        primary_patch_indices=indices,
        paired_patch_indices=indices,
        primary_grid_size=(4, 4),
        paired_grid_size=(4, 4),
        primary_bbox=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        paired_bbox=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        max_pairs=16,
        max_geometric_distance=0.2,
    )
    assert match is not None
    assert match.primary_local_indices.tolist() == [5, 6, 9, 10]
    # Mapped coordinates land midway between paired patch centers; argmin keeps
    # the lower deterministic index for each tie.
    assert match.paired_local_indices.tolist() == [0, 2, 8, 10]


def test_correspondence_statistics_reward_true_spatial_match() -> None:
    indices = torch.arange(4)
    match = build_bbox_aligned_patch_match(
        primary_patch_indices=indices,
        paired_patch_indices=indices,
        primary_grid_size=(2, 2),
        paired_grid_size=(2, 2),
        primary_bbox=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        paired_bbox=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        max_pairs=4,
        max_geometric_distance=0.01,
    )
    assert match is not None
    tokens = torch.eye(4)
    stats = correspondence_statistics(tokens, tokens.clone(), match)
    assert stats["matched_cosine"] == pytest.approx(1.0)
    assert stats["correspondence_margin"] > 0.9
    assert stats["feature_top1_rate"] == pytest.approx(1.0)


def test_gate_requires_geometry_and_recall_safe_teacher_support() -> None:
    overall = {
        "valid_sample_fraction": 0.99,
        "mean_correspondence_margin": 0.08,
        "mean_feature_top1_lift": 0.20,
        "mean_geometric_distance": 0.03,
    }
    support = {
        "class1_fn_corrected_by_paired": 8,
        "class1_tp_broken_by_paired": 2,
        "class1_fp_suppressed_by_paired": 9,
        "class1_fp_created_by_paired": 3,
        "class1_fn_correction_rate": 0.25,
        "class1_tp_break_rate": 0.02,
    }
    passed = assess_correspondence_gate(
        overall=overall,
        prediction_support=support,
        support_complete=True,
    )
    assert passed["smoke_ready"] is True

    unsafe = dict(support)
    unsafe["class1_tp_broken_by_paired"] = 12
    unsafe["class1_tp_break_rate"] = 0.15
    rejected = assess_correspondence_gate(
        overall=overall,
        prediction_support=unsafe,
        support_complete=True,
    )
    assert rejected["geometry_signal_ready"] is True
    assert rejected["class1_teacher_signal_ready"] is False
    assert rejected["training_permission"] is False


def test_gate_preserves_exact_zero_rates() -> None:
    result = assess_correspondence_gate(
        overall={
            "valid_sample_fraction": 1.0,
            "mean_correspondence_margin": 0.10,
            "mean_feature_top1_lift": 0.20,
            "mean_geometric_distance": 0.0,
        },
        prediction_support={
            "class1_fn_corrected_by_paired": 6,
            "class1_tp_broken_by_paired": 0,
            "class1_fp_suppressed_by_paired": 3,
            "class1_fp_created_by_paired": 0,
            "class1_fn_correction_rate": 0.20,
            "class1_tp_break_rate": 0.0,
        },
        support_complete=True,
    )
    assert result["geometry_checks"]["geometric_distance"] is True
    assert result["class1_checks"]["tp_break_rate"] is True
    assert result["smoke_ready"] is True

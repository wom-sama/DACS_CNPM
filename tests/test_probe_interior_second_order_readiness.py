import numpy as np
import torch

from trkh.tools.probe_interior_second_order_readiness import (
    assess_interior_second_order_readiness,
    build_fixed_orthogonal_projection,
    compact_second_order_descriptor,
    token_region_masks,
)


def _metrics(macro: float, focus: float):
    return {"macro_f1": macro, "focus_f1": focus}


def test_fixed_projection_is_deterministic_and_orthogonal():
    first = build_fixed_orthogonal_projection(32, 8, seed=17)
    second = build_fixed_orthogonal_projection(32, 8, seed=17)
    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(8), atol=1e-5)


def test_region_masks_erode_full_bbox_to_four_token_core():
    bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    indices = torch.arange(16).view(1, 16)
    core, ring, context, valid, stats = token_region_masks(
        crop_bbox=bbox,
        patch_indices=indices,
        grid_size=(4, 4),
        core_erode_ratio=0.25,
    )
    assert int(core.sum()) == 4
    assert int(ring.sum()) == 12
    assert int(context.sum()) == 0
    assert int(valid.sum()) == 16
    assert not bool(stats["core_fallback"].item())
    assert bool(stats["context_fallback"].item())


def test_compact_second_order_descriptor_is_finite_and_permutation_invariant():
    torch.manual_seed(3)
    tokens = torch.randn(2, 12, 16)
    weights = torch.rand(2, 12)
    weights = weights / weights.sum(dim=1, keepdim=True)
    projection = build_fixed_orthogonal_projection(16, 6, seed=9)
    descriptor, mean = compact_second_order_descriptor(tokens, weights, projection)
    permutation = torch.randperm(12)
    permuted, permuted_mean = compact_second_order_descriptor(
        tokens[:, permutation],
        weights[:, permutation],
        projection,
    )
    assert descriptor.shape == (2, 27)
    assert mean.shape == (2, 6)
    assert torch.isfinite(descriptor).all()
    assert torch.allclose(descriptor, permuted, atol=1e-5)
    assert torch.allclose(mean, permuted_mean, atol=1e-5)


def test_readiness_gate_requires_geometry_and_class1_safety():
    transitions = {
        "corrections": 15,
        "harms": 7,
        "class1_fn_rescued": 5,
        "class1_tp_broken": 2,
        "class1_fp_removed": 8,
        "class1_fp_created": 2,
    }
    ready = assess_interior_second_order_readiness(
        train_samples=9215,
        val_samples=2606,
        mean_core_tokens=32.0,
        core_fallback_fraction=0.0,
        direct_metrics=_metrics(0.884, 0.686),
        head_oof_metrics=_metrics(0.930, 0.790),
        candidate_oof_metrics=_metrics(0.935, 0.805),
        head_val_metrics=_metrics(0.880, 0.670),
        candidate_val_metrics=_metrics(0.890, 0.710),
        all_oof_metrics=_metrics(0.80, 0.70),
        core_oof_metrics=_metrics(0.82, 0.72),
        all_val_metrics=_metrics(0.78, 0.60),
        core_val_metrics=_metrics(0.81, 0.63),
        direct_transitions=transitions,
        class1_error_delta_auc=0.68,
    )
    assert ready["smoke_ready"] is True

    rejected = assess_interior_second_order_readiness(
        train_samples=9215,
        val_samples=2606,
        mean_core_tokens=32.0,
        core_fallback_fraction=0.0,
        direct_metrics=_metrics(0.884, 0.686),
        head_oof_metrics=_metrics(0.930, 0.790),
        candidate_oof_metrics=_metrics(0.929, 0.780),
        head_val_metrics=_metrics(0.880, 0.670),
        candidate_val_metrics=_metrics(0.875, 0.650),
        all_oof_metrics=_metrics(0.80, 0.70),
        core_oof_metrics=_metrics(0.79, 0.68),
        all_val_metrics=_metrics(0.78, 0.60),
        core_val_metrics=_metrics(0.77, 0.58),
        direct_transitions={**transitions, "corrections": 2, "harms": 9},
        class1_error_delta_auc=0.45,
    )
    assert rejected["smoke_ready"] is False
    assert "oof_class1_gain" in rejected["failed_checks"]
    assert "net_corrections_nonnegative" in rejected["failed_checks"]

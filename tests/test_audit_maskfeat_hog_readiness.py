from __future__ import annotations

import numpy as np
import torch

from trkh.tools.audit_maskfeat_hog_readiness import (
    assess_maskfeat_hog_readiness,
    hog_readout_descriptor,
    rgb_hog_cell_map,
)


def test_rgb_hog_cell_map_shape_and_local_normalization() -> None:
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, :, 16:] = 1.0
    cell_map = rgb_hog_cell_map(image, orientation_bins=9, cell_size=8)
    assert tuple(cell_map.shape) == (1, 3, 9, 4, 4)
    norms = torch.sqrt(cell_map.square().sum(dim=2))
    assert bool((norms <= 1.000001).all())
    assert int((norms > 0.99).sum()) > 0
    descriptor = hog_readout_descriptor(cell_map, readout_grid=2)
    assert tuple(descriptor.shape) == (1, 3 * 9 * 2 * 2)
    assert torch.isfinite(descriptor).all()


def test_rgb_hog_local_normalization_is_gain_stable() -> None:
    generator = torch.Generator().manual_seed(7)
    image = torch.rand(2, 3, 32, 32, generator=generator) * 0.4
    original = rgb_hog_cell_map(image, orientation_bins=9, cell_size=8)
    gained = rgb_hog_cell_map(image * 2.0, orientation_bins=9, cell_size=8)
    np.testing.assert_allclose(
        original.numpy(),
        gained.numpy(),
        atol=2e-4,
        rtol=2e-4,
    )


def test_maskfeat_gate_rejects_weak_class1_signal() -> None:
    metrics_base = {"macro_f1": 0.88, "focus_f1": 0.68}
    metrics_weak = {"macro_f1": 0.70, "focus_f1": 0.30}
    transitions = {
        "corrections": 2,
        "harms": 20,
        "class1_fn_rescued": 1,
        "class1_tp_broken": 8,
        "class1_fp_removed": 2,
        "class1_fp_created": 9,
    }
    result = assess_maskfeat_hog_readiness(
        train_samples=9215,
        val_samples=2606,
        source_groups=8064,
        descriptor_effective_rank=50.0,
        descriptor_finite=True,
        direct_val_metrics=metrics_base,
        candidate_oof_metrics=metrics_weak,
        candidate_val_metrics=metrics_weak,
        binary_oracle={"f1": 0.69},
        train_direction_auc=0.51,
        val_direction_auc=0.52,
        val_transitions=transitions,
    )
    assert result["smoke_permission"] is False
    assert "validation_class1_signal" in result["failed_checks"]
    assert "validation_class1_recall_protected" in result["failed_checks"]

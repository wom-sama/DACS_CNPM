from __future__ import annotations

import numpy as np
import torch

from trkh.tools.audit_dinov2_dense_patch_readiness import (
    MODEL_NAME,
    _parse_args,
    assess_dinov2_dense_patch_readiness,
    dinov2_dense_patch_descriptors,
)


def test_dinov2_dense_patch_descriptor_shape_and_fixed_moments() -> None:
    cls = torch.tensor([[[10.0, 20.0]]])
    patches = torch.arange(1, 33, dtype=torch.float32).reshape(1, 16, 2)
    tokens = torch.cat((cls, patches), dim=1)
    descriptors = dinov2_dense_patch_descriptors(
        tokens,
        grid_size=(4, 4),
        prefix_tokens=1,
        spatial_pool_size=2,
    )
    assert tuple(descriptors["cls"].shape) == (1, 2)
    assert tuple(descriptors["dense_patch"].shape) == (1, 12)
    assert tuple(descriptors["cls_plus_dense_patch"].shape) == (1, 14)
    np.testing.assert_allclose(descriptors["cls"].numpy(), [[10.0, 20.0]])
    np.testing.assert_allclose(
        descriptors["dense_patch"][:, :2].numpy(),
        patches.mean(dim=1).numpy(),
    )
    np.testing.assert_allclose(
        descriptors["dense_patch"][:, 2:4].numpy(),
        patches.std(dim=1, unbiased=False).numpy(),
    )
    assert torch.isfinite(descriptors["cls_plus_dense_patch"]).all()


def test_dinov2_dense_patch_descriptor_rejects_grid_mismatch() -> None:
    tokens = torch.zeros(2, 17, 8)
    try:
        dinov2_dense_patch_descriptors(tokens, grid_size=(5, 5), prefix_tokens=1)
    except ValueError as error:
        assert "Token/grid mismatch" in str(error)
    else:
        raise AssertionError("Expected a token/grid mismatch")


def test_dinov2_dense_patch_protocol_defaults_are_locked() -> None:
    args = _parse_args(
        (
            "--classification-root",
            "class_f",
            "--yolo-data",
            "data.yaml",
            "--base-train-csv",
            "train.csv",
            "--base-val-csv",
            "val.csv",
            "--output-dir",
            "output",
        )
    )
    assert args.model == MODEL_NAME
    assert args.folds == 5
    assert args.batch_size == 128


def test_dinov2_dense_patch_gate_rejects_non_incremental_candidate() -> None:
    base = {"macro_f1": 0.884, "focus_f1": 0.686}
    control = {"macro_f1": 0.84, "focus_f1": 0.53}
    candidate = {"macro_f1": 0.841, "focus_f1": 0.535}
    transitions = {
        "corrections": 12,
        "harms": 40,
        "class1_fn_rescued": 4,
        "class1_tp_broken": 20,
        "class1_fp_removed": 8,
        "class1_fp_created": 15,
    }
    result = assess_dinov2_dense_patch_readiness(
        train_samples=9215,
        val_samples=2606,
        source_groups=8064,
        dense_effective_rank=50.0,
        descriptor_finite=True,
        energy_entropy=0.8,
        energy_border_mass=0.25,
        direct_val_metrics=base,
        control_oof_metrics=control,
        candidate_oof_metrics=candidate,
        control_val_metrics=control,
        candidate_val_metrics=candidate,
        binary_oracle={"f1": 0.70},
        train_direction_auc=0.55,
        val_direction_auc=0.56,
        direct_val_transitions=transitions,
    )
    assert result["smoke_permission"] is False
    assert "oof_class1_gain" in result["failed_checks"]
    assert "validation_class1_milestone" in result["failed_checks"]
    assert "direct_validation_class1_recall_protected" in result["failed_checks"]

from __future__ import annotations

import numpy as np
import torch

from trkh.tools.audit_multistage_teacher_feature_readiness import (
    STAGE_NAMES,
    _class1_error_direction_auc,
    assess_multistage_teacher_readiness,
    build_fixed_projection,
    multistage_signature_from_activations,
)


def _metrics(macro: float, focus: float):
    return {
        "macro_f1": macro,
        "focus_f1": focus,
        "focus_precision": focus,
        "focus_recall": focus,
    }


def _transitions():
    return {
        "changed": 20,
        "corrections": 12,
        "harms": 8,
        "neutral": 0,
        "class1_fn_rescued": 6,
        "class1_tp_broken": 3,
        "class1_fp_removed": 8,
        "class1_fp_created": 2,
    }


def test_fixed_projection_is_deterministic_and_orthonormal() -> None:
    first = build_fixed_projection(32, 8, seed=17)
    second = build_fixed_projection(32, 8, seed=17)
    assert torch.equal(first, second)
    identity = first.transpose(0, 1) @ first
    assert torch.allclose(identity, torch.eye(8), atol=1e-5)


def test_multistage_signature_has_stable_shapes_and_finite_values() -> None:
    channels = {"stage1": 6, "stage2": 8, "stage4": 10, "stage5": 12, "head": 14}
    activations = {
        name: torch.randn(3, channel, 9 - index, 9 - index)
        for index, (name, channel) in enumerate(channels.items())
    }
    head, signature, energy, projections = multistage_signature_from_activations(
        activations,
        projection_dim=4,
        spatial_size=3,
        seed=29,
    )
    assert head.shape == (3, 14)
    assert signature.shape == (3, len(STAGE_NAMES) * (4 + 9))
    assert set(energy) == set(STAGE_NAMES)
    assert set(projections) == set(STAGE_NAMES)
    assert torch.isfinite(head).all()
    assert torch.isfinite(signature).all()


def test_class1_error_direction_auc_rewards_fn_increase_and_fp_decrease() -> None:
    labels = np.asarray([1, 1, 0, 2], dtype=np.int64)
    base = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    candidate = base.copy()
    candidate[0, 1] += 0.5
    candidate[2:, 1] -= 0.5
    assert _class1_error_direction_auc(labels, base, candidate) == 1.0


def test_readiness_gate_passes_only_with_signal_and_class1_safety() -> None:
    passed = assess_multistage_teacher_readiness(
        train_samples=9215,
        val_samples=2606,
        source_groups=8064,
        signature_effective_rank=40.0,
        mean_energy_entropy=0.75,
        max_energy_border_mass=0.40,
        direct_val_metrics=_metrics(0.884, 0.686),
        head_oof_metrics=_metrics(0.88, 0.70),
        candidate_oof_metrics=_metrics(0.89, 0.73),
        head_val_metrics=_metrics(0.88, 0.68),
        candidate_val_metrics=_metrics(0.889, 0.72),
        train_direction_auc=0.70,
        val_direction_auc=0.68,
        val_transitions=_transitions(),
    )
    assert passed["smoke_permission"] is True

    failed = assess_multistage_teacher_readiness(
        train_samples=9215,
        val_samples=2606,
        source_groups=8064,
        signature_effective_rank=40.0,
        mean_energy_entropy=0.75,
        max_energy_border_mass=0.40,
        direct_val_metrics=_metrics(0.884, 0.686),
        head_oof_metrics=_metrics(0.88, 0.70),
        candidate_oof_metrics=_metrics(0.881, 0.701),
        head_val_metrics=_metrics(0.88, 0.68),
        candidate_val_metrics=_metrics(0.879, 0.681),
        train_direction_auc=0.70,
        val_direction_auc=0.45,
        val_transitions={**_transitions(), "class1_tp_broken": 9},
    )
    assert failed["smoke_permission"] is False
    assert "validation_error_direction" in failed["failed_checks"]
    assert "validation_class1_recall_protected" in failed["failed_checks"]

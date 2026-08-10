from __future__ import annotations

import numpy as np
import torch

from trkh.tools.audit_wavelet_scattering_readiness import (
    assess_wavelet_scattering_readiness,
    aligned_decision_probabilities,
    build_scattering_channels,
    scattering_descriptor,
    scattering_order_slices,
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


def test_scattering_channels_are_bounded_and_deterministic() -> None:
    rgb = torch.tensor([[[[1.0]], [[0.0]], [[0.5]]]], dtype=torch.float32)
    channels = build_scattering_channels(rgb)
    assert channels.shape == (1, 6, 1, 1)
    assert torch.allclose(channels[0, :3, 0, 0], torch.tensor([1.0, 0.0, 0.5]))
    assert torch.allclose(channels[0, 3:, 0, 0], torch.tensor([0.356, 1.0, 0.5]))
    assert float(channels.min()) >= 0.0
    assert float(channels.max()) <= 1.0


def test_scattering_descriptor_has_fixed_mean_std_and_spatial_shape() -> None:
    values = torch.arange(2 * 6 * 61 * 4 * 4, dtype=torch.float32).reshape(2, 6, 61, 4, 4)
    descriptor = scattering_descriptor(values, spatial_pool_size=2)
    assert descriptor.shape == (2, 6 * 61 * (1 + 1 + 4))
    assert torch.isfinite(descriptor).all()


def test_scattering_order_slices_cover_every_coefficient_once() -> None:
    order = scattering_order_slices(61, j=3, l=4)
    assert order["zero"] == slice(0, 1)
    assert order["first"] == slice(1, 13)
    assert order["second"] == slice(13, 61)


def test_aligned_decision_probabilities_restore_class_column_order() -> None:
    class Estimator:
        classes_ = np.asarray([2, 0, 1], dtype=np.int64)

        @staticmethod
        def decision_function(_features):
            return np.asarray([[3.0, 1.0, 2.0]], dtype=np.float64)

    probabilities = aligned_decision_probabilities(
        Estimator(),
        np.zeros((1, 2), dtype=np.float32),
        class_count=3,
    )
    assert probabilities.shape == (1, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert int(probabilities.argmax(axis=1)[0]) == 2
    assert probabilities[0, 2] > probabilities[0, 1] > probabilities[0, 0]


def test_wavelet_gate_requires_fold_safe_class1_corrections() -> None:
    passed = assess_wavelet_scattering_readiness(
        train_samples=9215,
        val_samples=2606,
        source_groups=8064,
        descriptor_effective_rank=40.0,
        descriptor_finite=True,
        direct_val_metrics=_metrics(0.884, 0.686),
        candidate_oof_metrics=_metrics(0.85, 0.66),
        candidate_val_metrics=_metrics(0.84, 0.64),
        binary_oracle={"f1": 0.73},
        train_direction_auc=0.70,
        val_direction_auc=0.68,
        val_transitions=_transitions(),
    )
    assert passed["smoke_permission"] is True

    failed = assess_wavelet_scattering_readiness(
        train_samples=9215,
        val_samples=2606,
        source_groups=8064,
        descriptor_effective_rank=40.0,
        descriptor_finite=True,
        direct_val_metrics=_metrics(0.884, 0.686),
        candidate_oof_metrics=_metrics(0.83, 0.58),
        candidate_val_metrics=_metrics(0.81, 0.49),
        binary_oracle={"f1": 0.69},
        train_direction_auc=0.70,
        val_direction_auc=0.45,
        val_transitions={**_transitions(), "class1_tp_broken": 9},
    )
    assert failed["smoke_permission"] is False
    assert "validation_class1_signal" in failed["failed_checks"]
    assert "validation_error_direction" in failed["failed_checks"]
    assert "validation_class1_recall_protected" in failed["failed_checks"]

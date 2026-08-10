from __future__ import annotations

import numpy as np
import torch

from trkh.tools.probe_context_illuminant_estimator_readiness import (
    _context_mask,
    apply_gain_and_correction,
    assess_context_illuminant_readiness,
    context_color_descriptor,
)


def _metrics(macro: float, focus: float, *, fp: int = 4, fn: int = 3):
    matrix = np.eye(5, dtype=np.int64) * 20
    matrix[0, 1] = fp
    matrix[1, 0] = fn
    return {
        "macro_f1": macro,
        "focus_f1": focus,
        "confusion_matrix": matrix.tolist(),
    }


def test_context_mask_excludes_every_bbox() -> None:
    mask = _context_mask(
        np.ones((20, 20), dtype=bool),
        np.asarray([[0.25, 0.25, 0.20, 0.20], [0.75, 0.75, 0.20, 0.20]]),
        margin_ratio=0.0,
    )
    assert not mask[5, 5]
    assert not mask[15, 15]
    assert mask[5, 15]


def test_context_descriptor_is_finite_and_cast_sensitive() -> None:
    rgb = np.stack(
        np.meshgrid(
            np.linspace(0.1, 0.9, 16, dtype=np.float32),
            np.linspace(0.2, 0.8, 16, dtype=np.float32),
        ),
        axis=0,
    )
    rgb = np.concatenate((rgb, rgb[:1] * 0.5 + 0.2), axis=0)
    clean, _ = context_color_descriptor(rgb, np.ones((16, 16), bool), np.zeros((0, 4)), (1, 1, 1))
    warm, _ = context_color_descriptor(rgb, np.ones((16, 16), bool), np.zeros((0, 4)), (1.2, 1.0, 0.8))
    assert clean.shape == (67,)
    assert np.isfinite(clean).all()
    assert not np.allclose(clean, warm)


def test_gain_correction_recovers_unclipped_rgb() -> None:
    rgb = torch.full((2, 3, 4, 4), 0.4)
    gain = torch.tensor([[1.2, 0.9, 0.8], [0.7, 1.1, 1.2]])
    cast, corrected = apply_gain_and_correction(rgb, gain, gain)
    assert not torch.allclose(cast, rgb)
    assert torch.allclose(corrected, rgb, atol=1e-6)


def test_readiness_gate_requires_full_support_and_clean_preservation() -> None:
    regression = {
        "r2_per_channel": [0.98, 0.98, 0.98],
        "log_gain_mae_per_channel": [0.005, 0.006, 0.005],
    }
    clean = {
        "raw_metrics": _metrics(0.885, 0.680),
        "corrected_metrics": _metrics(0.884, 0.678),
        "raw_to_corrected": {"corrections": 2, "harms": 1},
    }
    shifted = {
        "raw_metrics": _metrics(0.70, 0.45, fp=20, fn=20),
        "corrected_metrics": _metrics(0.882, 0.675, fp=5, fn=4),
        "raw_to_corrected": {"corrections": 30, "harms": 2},
    }
    passed = assess_context_illuminant_readiness(
        train_sources=8064,
        val_sources=2577,
        val_objects=2606,
        oof_regression=regression,
        val_regression=regression,
        conditions={"clean": clean, "warm": shifted},
    )
    assert passed["smoke_permission"] is True
    rejected = assess_context_illuminant_readiness(
        train_sources=100,
        val_sources=50,
        val_objects=50,
        oof_regression=regression,
        val_regression=regression,
        conditions={"clean": clean, "warm": shifted},
    )
    assert rejected["smoke_permission"] is False
    assert "full_train_source_support" in rejected["failed_checks"]

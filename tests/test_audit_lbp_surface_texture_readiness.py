from __future__ import annotations

import math

import numpy as np
import torch

from trkh.tools.audit_lbp_surface_texture_readiness import (
    FOCUS_MILESTONE,
    FOLDS,
    DESCRIPTOR_THREADS,
    INTERIOR_ERODE_RATIO,
    LBP_SCALES,
    MAX_ITERATIONS,
    RESIDUAL_C,
    ROI_SIZE,
    SEED,
    apply_offset_residual,
    assess_lbp_readiness,
    interior_roi_boxes,
    parse_args,
    uniform_lbp_descriptor,
)


def test_protocol_defaults_are_locked() -> None:
    args = parse_args(
        [
            "--data",
            "data.yaml",
            "--checkpoint",
            "best.pt",
            "--train-cache",
            "train.npz",
            "--val-cache",
            "val.npz",
            "--output-dir",
            "out",
        ]
    )
    assert args.roi_size == ROI_SIZE == 128
    assert args.interior_erode_ratio == INTERIOR_ERODE_RATIO == 0.15
    assert args.folds == FOLDS == 5
    assert args.descriptor_threads == DESCRIPTOR_THREADS == 8
    assert args.residual_c == RESIDUAL_C == 0.3
    assert args.max_iterations == MAX_ITERATIONS == 300
    assert args.seed == SEED == 20260712
    assert LBP_SCALES == ((8, 1.0), (16, 2.0))
    assert FOCUS_MILESTONE == 0.70


def test_interior_roi_boxes_erode_centered_box() -> None:
    bbox = torch.tensor([[0.5, 0.5, 0.8, 0.6]], dtype=torch.float32)
    boxes = interior_roi_boxes(
        bbox,
        image_height=100,
        image_width=200,
        erode_ratio=0.25,
    )
    assert boxes.shape == (1, 5)
    assert torch.allclose(
        boxes[0],
        torch.tensor([0.0, 60.0, 35.0, 140.0, 65.0]),
        atol=1e-5,
    )


def test_uniform_lbp_descriptor_is_normalized_and_shift_invariant() -> None:
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    shifted = image + np.uint8(20)
    descriptor, maps = uniform_lbp_descriptor(image)
    shifted_descriptor, shifted_maps = uniform_lbp_descriptor(shifted)
    assert descriptor.shape == (28,)
    assert len(maps) == len(shifted_maps) == 2
    assert math.isclose(float(descriptor[:10].sum()), 1.0, abs_tol=1e-7)
    assert math.isclose(float(descriptor[10:].sum()), 1.0, abs_tol=1e-7)
    assert np.array_equal(descriptor, shifted_descriptor)


def test_zero_offset_residual_preserves_base_probabilities() -> None:
    base = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32)
    features = np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    probabilities = apply_offset_residual(
        base,
        features,
        np.zeros((2, 3), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    assert np.allclose(probabilities, base, atol=1e-7)


def _metrics(macro: float, focus_f1: float, focus_recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.85, "recall": 0.85},
            {"f1": focus_f1, "precision": 0.72, "recall": focus_recall},
        ],
    }


def _transitions() -> dict:
    return {
        "changed": 12,
        "corrections": 8,
        "harms": 3,
        "neutral": 1,
        "focus_false_positive_removed": 4,
        "focus_false_positive_created": 1,
        "focus_false_negative_rescued": 3,
        "focus_true_positive_broken": 1,
    }


def _direction(auc: float) -> dict:
    return {
        "auc_fn_positive": auc,
        "samples": 20,
        "false_positives": 10,
        "false_negatives": 10,
    }


def test_gate_accepts_transferable_recall_safe_texture_signal() -> None:
    result = assess_lbp_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=0,
        train_val_source_overlap=0,
        histogram_sum_max_abs_error=1e-7,
        descriptor_effective_rank=9.0,
        all_optimizers_converged=True,
        folds_with_focus_gain=4,
        fold_count=5,
        oof_keeper=_metrics(0.93, 0.80, 0.81),
        oof_candidate=_metrics(0.932, 0.81, 0.82),
        val_keeper=_metrics(0.884, 0.686, 0.781),
        val_lbp=_metrics(0.70, 0.55, 0.58),
        val_candidate=_metrics(0.887, 0.705, 0.79),
        oof_transitions=_transitions(),
        val_transitions=_transitions(),
        oof_direction=_direction(0.68),
        val_direction=_direction(0.66),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["image_smoke_permission"]
    assert result["failed_checks"] == []


def test_gate_rejects_texture_suppressor_and_source_overlap() -> None:
    transitions = _transitions()
    transitions.update(
        {
            "corrections": 2,
            "harms": 9,
            "focus_false_negative_rescued": 0,
            "focus_true_positive_broken": 8,
        }
    )
    result = assess_lbp_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=1,
        train_val_source_overlap=1,
        histogram_sum_max_abs_error=0.1,
        descriptor_effective_rank=2.0,
        all_optimizers_converged=False,
        folds_with_focus_gain=1,
        fold_count=5,
        oof_keeper=_metrics(0.93, 0.80, 0.81),
        oof_candidate=_metrics(0.92, 0.78, 0.78),
        val_keeper=_metrics(0.884, 0.686, 0.781),
        val_lbp=_metrics(0.50, 0.30, 0.25),
        val_candidate=_metrics(0.87, 0.62, 0.60),
        oof_transitions=transitions,
        val_transitions=transitions,
        oof_direction=_direction(0.40),
        val_direction=_direction(0.80),
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["image_smoke_permission"]
    assert "source_group_folds_disjoint" in result["failed_checks"]
    assert "train_val_sources_disjoint" in result["failed_checks"]
    assert "lbp_histograms_normalized" in result["failed_checks"]
    assert "val_focus_recall_preserved_within_0p01" in result["failed_checks"]
    assert "val_focus_fn_rescued_ge_tp_broken" in result["failed_checks"]
    assert "direction_auc_gap_le_0p15" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]

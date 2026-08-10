from __future__ import annotations

import numpy as np

from trkh.tools.audit_surface_blob_morphology_readiness import (
    BLOB_SIGMAS,
    CONTROL_DIM,
    FOLDS,
    INTERIOR_ERODE_RATIO,
    MORPHOLOGY_DIM,
    RESIDUAL_C,
    ROI_SIZE,
    HIGHLIGHT_QUANTILE,
    assess_surface_blob_readiness,
    parse_args,
    achromatic_highlight_mask_from_rgb,
    surface_blob_descriptor,
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
    assert args.residual_c == RESIDUAL_C == 0.3
    assert HIGHLIGHT_QUANTILE == 0.95
    assert BLOB_SIGMAS == (1.2, 2.4, 4.8)
    assert CONTROL_DIM == 25
    assert MORPHOLOGY_DIM == 45


def test_highlight_proxy_prefers_bright_achromatic_pixels() -> None:
    rgb = np.zeros((20, 20, 3), dtype=np.float32)
    rgb[:] = (0.15, 0.55, 0.10)
    rgb[2:7, 2:7] = (0.95, 0.95, 0.95)
    rgb[12:17, 12:17] = (0.95, 0.85, 0.05)
    mask, score = achromatic_highlight_mask_from_rgb(rgb)
    assert score[3, 3] > score[13, 13]
    assert mask[3, 3]
    assert not mask[13, 13]


def _synthetic_surface(with_spots: bool) -> np.ndarray:
    yy, xx = np.mgrid[:128, :128]
    base = np.zeros((128, 128, 3), dtype=np.float32)
    base[..., 0] = 0.48 + 0.03 * xx / 127.0
    base[..., 1] = 0.68 + 0.02 * yy / 127.0
    base[..., 2] = 0.20
    if with_spots:
        for cy, cx, radius in ((38, 42, 3), (64, 76, 4), (91, 51, 2), (80, 95, 3)):
            spot = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            base[spot] *= 0.35
    return (base.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


def test_surface_blob_descriptor_is_finite_and_detects_added_spots() -> None:
    smooth_control, smooth_morphology, smooth_maps, smooth_telemetry = surface_blob_descriptor(
        _synthetic_surface(False)
    )
    spot_control, spot_morphology, spot_maps, spot_telemetry = surface_blob_descriptor(
        _synthetic_surface(True)
    )
    assert smooth_control.shape == spot_control.shape == (CONTROL_DIM,)
    assert smooth_morphology.shape == spot_morphology.shape == (MORPHOLOGY_DIM,)
    assert np.isfinite(spot_morphology).all()
    assert spot_telemetry["dark_peak_density_sum"] > smooth_telemetry["dark_peak_density_sum"]
    assert set(spot_maps) == {
        "highlight",
        "highlight_score",
        "dark_blob",
        "bright_blob",
        "chroma_blob",
        "valid",
    }


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


def _metric_bundle(*, winner: bool) -> tuple[dict, dict]:
    candidate_focus = 0.71 if winner else 0.67
    candidate_macro = 0.887 if winner else 0.875
    oof = {
        "control_standalone": _metrics(0.70, 0.52, 0.55),
        "candidate_standalone": _metrics(0.712 if winner else 0.69, 0.54 if winner else 0.50, 0.57),
        "keeper_control_residual": _metrics(0.930, 0.800, 0.81),
        "keeper_candidate_residual": _metrics(0.932 if winner else 0.925, 0.810 if winner else 0.790, 0.82),
    }
    val = {
        "keeper": _metrics(0.884, 0.686, 0.781),
        "control_standalone": _metrics(0.70, 0.52, 0.55),
        "candidate_standalone": _metrics(0.712 if winner else 0.69, 0.54 if winner else 0.50, 0.57),
        "keeper_control_residual": _metrics(0.884, 0.690, 0.78),
        "keeper_candidate_residual": _metrics(candidate_macro, candidate_focus, 0.79 if winner else 0.70),
    }
    return oof, val


def test_gate_accepts_incremental_recall_safe_morphology() -> None:
    oof, val = _metric_bundle(winner=True)
    result = assess_surface_blob_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=0,
        train_val_source_overlap=0,
        descriptors_finite=True,
        morphology_effective_rank=12.0,
        mean_highlight_fraction=0.05,
        all_optimizers_converged=True,
        folds_with_residual_focus_gain=4,
        fold_count=5,
        oof_metrics=oof,
        val_metrics=val,
        oof_incremental_transitions=_transitions(),
        val_incremental_transitions=_transitions(),
        val_keeper_transitions=_transitions(),
        oof_incremental_direction=_direction(0.68),
        val_incremental_direction=_direction(0.66),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["image_smoke_permission"]
    assert result["failed_checks"] == []


def test_gate_rejects_nonincremental_signal_and_source_overlap() -> None:
    oof, val = _metric_bundle(winner=False)
    bad_transitions = _transitions()
    bad_transitions.update(
        {
            "corrections": 2,
            "harms": 9,
            "focus_false_negative_rescued": 0,
            "focus_true_positive_broken": 8,
        }
    )
    result = assess_surface_blob_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=1,
        train_val_source_overlap=1,
        descriptors_finite=False,
        morphology_effective_rank=2.0,
        mean_highlight_fraction=0.20,
        all_optimizers_converged=False,
        folds_with_residual_focus_gain=1,
        fold_count=5,
        oof_metrics=oof,
        val_metrics=val,
        oof_incremental_transitions=bad_transitions,
        val_incremental_transitions=bad_transitions,
        val_keeper_transitions=bad_transitions,
        oof_incremental_direction=_direction(0.40),
        val_incremental_direction=_direction(0.80),
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["image_smoke_permission"]
    assert "source_group_folds_disjoint" in result["failed_checks"]
    assert "train_val_sources_disjoint" in result["failed_checks"]
    assert "descriptors_finite" in result["failed_checks"]
    assert "highlight_exclusion_fraction_locked" in result["failed_checks"]
    assert "val_focus_reaches_0p70" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]


def test_gate_fails_closed_when_direction_auc_is_unavailable() -> None:
    oof, val = _metric_bundle(winner=True)
    unavailable = _direction(0.0)
    unavailable["auc_fn_positive"] = None
    result = assess_surface_blob_readiness(
        train_rows=256,
        val_rows=128,
        fold_source_overlap=0,
        train_val_source_overlap=0,
        descriptors_finite=True,
        morphology_effective_rank=12.0,
        mean_highlight_fraction=0.05,
        all_optimizers_converged=True,
        folds_with_residual_focus_gain=4,
        fold_count=5,
        oof_metrics=oof,
        val_metrics=val,
        oof_incremental_transitions=_transitions(),
        val_incremental_transitions=_transitions(),
        val_keeper_transitions=_transitions(),
        oof_incremental_direction=unavailable,
        val_incremental_direction=_direction(0.66),
        focus_class_index=1,
        test_split_used=False,
    )
    assert not result["image_smoke_permission"]
    assert not result["observed"]["incremental_directions_available"]
    assert "incremental_directions_available" in result["failed_checks"]

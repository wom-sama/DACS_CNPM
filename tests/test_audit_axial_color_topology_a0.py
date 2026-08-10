from __future__ import annotations

import numpy as np
import pytest

from trkh.tools.audit_axial_color_topology_a0 import (
    AXIAL_DIM,
    CONTROL_DIM,
    EXPECTED_CLASS_COUNTS,
    TOTAL_DIM,
    assemble_role_features,
    assess_axial_color_topology_a0,
    axial_color_topology_descriptor,
    fit_offset_model,
    source_derangement,
)
from trkh.tools.audit_lbp_surface_texture_readiness import apply_offset_residual


def _surface(*, horizontal: bool = True) -> np.ndarray:
    size = 128
    coordinate = np.linspace(0.0, 1.0, size, dtype=np.float32)
    field = coordinate[None, :] if horizontal else coordinate[:, None]
    field = np.broadcast_to(field, (size, size))
    red = 70.0 + 145.0 * field
    green = 190.0 - 55.0 * field
    blue = 55.0 + 20.0 * (1.0 - field)
    image = np.stack((red, green, blue), axis=2)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _metrics(
    macro: float,
    precision: float,
    recall: float,
    focus_f1: float,
    nonfocus: float = 0.8,
) -> dict[str, object]:
    f1 = [nonfocus] * 5
    p = [nonfocus] * 5
    r = [nonfocus] * 5
    f1[1] = focus_f1
    p[1] = precision
    r[1] = recall
    return {
        "macro_f1": macro,
        "per_class_f1": f1,
        "per_class_precision": p,
        "per_class_recall": r,
        "support": [100] * 5,
        "predicted_support": [100] * 5,
        "confusion_matrix": [[0] * 5 for _ in range(5)],
        "accuracy": macro,
    }


def _bundle(*, shifted: bool = False) -> dict[str, object]:
    control = _metrics(0.80, 0.60, 0.70, 0.646)
    candidate = _metrics(0.81, 0.63, 0.70, 0.663)
    placebo = _metrics(0.805, 0.61, 0.70, 0.652)
    return {
        "brightness": 1.0,
        "contrast": 1.0,
        "metrics": {
            "keeper": control,
            "control": control,
            "candidate": candidate,
            "placebo": placebo,
        },
        "delta": {
            "macro_f1": 0.010,
            "class1_precision": 0.030,
            "class1_recall": 0.0,
            "class1_f1": 0.017,
            "candidate_vs_placebo_macro_f1": 0.005,
            "candidate_vs_placebo_class1_f1": 0.011,
            "maximum_nonfocus_f1_drop": 0.0,
        },
        "transitions": {
            "changed": 12,
            "corrections": 8,
            "harms": 2,
            "neutral": 2,
            "focus_fn_rescue": 3,
            "focus_tp_break": 1,
            "focus_fp_remove_correct": 6,
            "focus_fp_create": 1,
            "candidate_correction": 8,
            "candidate_harm": 2,
        },
        "restricted_fp": {
            "control": 20,
            "candidate": 14,
            "removed": 7,
            "created": 1,
            "net_reduction": 6 if not shifted else 3,
        },
        "direction": {
            "auc": 0.70 if not shifted else 0.62,
            "positive_keeper_fn": 10,
            "negative_restricted_keeper_fp": 20,
            "rows": 30,
            "mean_positive_score": 0.1,
            "mean_negative_score": -0.1,
        },
    }


def test_axial_descriptor_contract_is_finite() -> None:
    control, axial, telemetry = axial_color_topology_descriptor(_surface())
    assert control.shape == (CONTROL_DIM,)
    assert axial.shape == (AXIAL_DIM,)
    assert np.isfinite(control).all()
    assert np.isfinite(axial).all()
    assert telemetry["minimum_bin_support"] >= 128


def test_axial_descriptor_is_reflection_invariant() -> None:
    image = _surface()
    _control, original, _telemetry = axial_color_topology_descriptor(image)
    _control, reflected, _telemetry = axial_color_topology_descriptor(image[:, ::-1].copy())
    assert np.allclose(original, reflected, atol=2e-4, rtol=2e-4)


def test_axial_descriptor_is_quarter_turn_invariant() -> None:
    image = _surface(horizontal=True)
    _control, horizontal, _telemetry = axial_color_topology_descriptor(image)
    _control, vertical, _telemetry = axial_color_topology_descriptor(np.rot90(image).copy())
    assert np.allclose(horizontal, vertical, atol=2e-4, rtol=2e-4)


def test_axial_descriptor_detects_spatial_gradient() -> None:
    uniform = np.full((128, 128, 3), (145, 155, 70), dtype=np.uint8)
    _control, uniform_axial, _telemetry = axial_color_topology_descriptor(uniform)
    _control, gradient_axial, _telemetry = axial_color_topology_descriptor(_surface())
    assert float(np.linalg.norm(gradient_axial - uniform_axial)) > 1.0


def test_role_features_have_matched_dimensions_and_zero_control_tail() -> None:
    rng = np.random.default_rng(3)
    control = rng.normal(size=(6, CONTROL_DIM)).astype(np.float32)
    axial = rng.normal(size=(6, AXIAL_DIM)).astype(np.float32)
    control_role = assemble_role_features(control, axial, role="control")
    candidate_role = assemble_role_features(control, axial, role="candidate")
    placebo_role = assemble_role_features(
        control, axial, role="placebo", axial_indices=np.arange(5, -1, -1)
    )
    assert control_role.shape == candidate_role.shape == placebo_role.shape == (6, TOTAL_DIM)
    assert np.count_nonzero(control_role[:, CONTROL_DIM:]) == 0
    assert np.array_equal(candidate_role[:, :CONTROL_DIM], control)


def test_source_derangement_never_reuses_source() -> None:
    sources = np.asarray([f"source_{index // 2}" for index in range(40)], dtype=object)
    indices = np.arange(40, dtype=np.int64)
    permutation = source_derangement(indices, sources, seed=11)
    assert sorted(permutation.tolist()) == indices.tolist()
    assert np.all(sources[indices] != sources[permutation])


def test_fit_offset_model_is_finite_and_applies_to_eval() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, TOTAL_DIM)).astype(np.float32)
    labels = np.tile(np.arange(len(EXPECTED_CLASS_COUNTS)), 24)
    base = np.full((120, 5), 0.2, dtype=np.float32)
    weights, record = fit_offset_model(features, base, labels)
    probabilities = apply_offset_residual(
        base[:10], features[:10], weights, np.zeros(5, dtype=np.float64)
    )
    assert weights.shape == (TOTAL_DIM, 5)
    assert np.isfinite(weights).all()
    assert probabilities.shape == (10, 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert record["iterations"] < 300


def test_gate_accepts_only_complete_precision_safe_signal() -> None:
    structural = {
        "provenance_hashes_exact": True,
        "train_rows_exact": True,
        "class_counts_exact": True,
        "source_folds_exact": True,
        "source_overlap_zero": True,
        "four_conditions_exact": True,
        "descriptor_dimensions_exact": True,
        "descriptors_finite": True,
        "minimum_bin_support_gte_128": True,
        "axial_effective_rank_gte_12": True,
        "all_15_readouts_converged": True,
        "placebo_source_deranged": True,
        "raw_dataset_metadata_unchanged": True,
        "validation_data_unused": True,
        "test_data_unused": True,
        "model_or_checkpoint_written_false": True,
    }
    conditions = {
        "clean": _bundle(),
        "lighting_dim": _bundle(shifted=True),
        "lighting_bright": _bundle(shifted=True),
        "low_contrast": _bundle(shifted=True),
    }
    gate = assess_axial_color_topology_a0(
        structural_checks=structural, condition_results=conditions
    )
    assert gate["automatic_gates_passed"] is True
    assert gate["visual_review_required"] is True
    assert gate["stage_b_smoke_authorized"] is False


def test_gate_rejects_precision_regression() -> None:
    structural = {"structural": True}
    conditions = {
        "clean": _bundle(),
        "lighting_dim": _bundle(shifted=True),
        "lighting_bright": _bundle(shifted=True),
        "low_contrast": _bundle(shifted=True),
    }
    conditions["clean"]["delta"]["class1_precision"] = -0.01
    gate = assess_axial_color_topology_a0(
        structural_checks=structural, condition_results=conditions
    )
    assert gate["automatic_gates_passed"] is False
    assert "clean_class1_precision_delta_gte_0p010" in gate["failed_checks"]


def test_descriptor_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="requires uint8"):
        axial_color_topology_descriptor(np.zeros((64, 64, 3), dtype=np.uint8))

import numpy as np
import torch

from trkh.tools.audit_highres_shifted_window_bridge_readiness import (
    DETAIL_DESCRIPTOR_DIM,
    SHARED_DESCRIPTOR_DIM,
    assess_highres_bridge_readiness,
    fit_residual_readout,
    haar_bands_from_projected_grid,
    pool_spatial_descriptor,
    source_derangement,
    standardize_fit_holdout,
)


def _metrics(macro: float, f1: float, precision: float, recall: float):
    return {
        "macro_f1": macro,
        "focus_f1": f1,
        "focus_precision": precision,
        "focus_recall": recall,
        "per_class": [
            {"f1": macro + 0.01},
            {"f1": f1},
            {"f1": macro},
            {"f1": macro - 0.01},
            {"f1": macro},
        ],
    }


def test_haar_bands_match_orthonormal_two_by_two_definition() -> None:
    grid = torch.tensor([[[[1.0], [2.0]], [[3.0], [5.0]]]])
    bands = haar_bands_from_projected_grid(grid)
    assert bands.shape == (1, 1, 1, 3)
    torch.testing.assert_close(bands.flatten(), torch.tensor([-1.5, -2.5, 0.5]))


def test_spatial_pooling_has_locked_dimensions_and_coverage() -> None:
    feature_map = torch.arange(1, 33, dtype=torch.float32).view(1, 4, 4, 2)
    mask = torch.ones((1, 16), dtype=torch.bool)
    pooled, coverage = pool_spatial_descriptor(feature_map, mask)
    assert pooled.shape == (1, 12)
    assert coverage.shape == (1, 5)
    torch.testing.assert_close(coverage, torch.ones_like(coverage))
    torch.testing.assert_close(pooled[:, :2], feature_map.mean(dim=(1, 2)))


def test_source_derangement_is_bijective_and_source_disjoint() -> None:
    indices = np.arange(12, dtype=np.int64)
    groups = np.asarray([f"group_{index // 2}" for index in range(12)], dtype=object)
    permuted = source_derangement(indices, groups, seed=17)
    assert set(permuted.tolist()) == set(indices.tolist())
    assert not np.any(groups[indices] == groups[permuted])


def test_standardization_handles_protocol_zero_padding() -> None:
    fit = np.asarray([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    holdout = np.asarray([[7.0, 0.0]])
    fit_standardized, holdout_standardized, audit = standardize_fit_holdout(
        fit, holdout
    )
    assert audit == {"constant_columns": 1, "finite": True}
    np.testing.assert_allclose(fit_standardized[:, 1], 0.0)
    np.testing.assert_allclose(holdout_standardized[:, 1], 0.0)


def test_zero_feature_residual_readout_preserves_offset_decisions() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=np.float64
    )
    features = np.zeros((4, 3), dtype=np.float64)
    result = fit_residual_readout(
        fit_features=features,
        fit_labels=labels,
        fit_base_probabilities=probabilities,
        holdout_features=features,
        holdout_base_probabilities=probabilities,
        class_count=2,
        max_iter=20,
    )
    assert result["finite"] is True
    assert result["parameter_count"] == 8
    np.testing.assert_array_equal(
        result["probabilities"].argmax(axis=1), probabilities.argmax(axis=1)
    )


def test_readiness_gate_requires_behavior_and_placebo_wins() -> None:
    keeper = _metrics(0.90, 0.80, 0.75, 0.86)
    control = _metrics(0.90, 0.80, 0.75, 0.86)
    candidate = _metrics(0.904, 0.808, 0.758, 0.858)
    placebo = _metrics(0.902, 0.804, 0.754, 0.854)
    fold_rows = [
        {
            "candidate_control_class1_f1_delta": 0.001 if index < 4 else -0.001,
            "candidate_control_class1_precision_delta": (
                0.001 if index < 4 else -0.001
            ),
        }
        for index in range(5)
    ]
    result = assess_highres_bridge_readiness(
        structural_checks={"example": True},
        keeper_metrics=keeper,
        metrics={"control": control, "candidate": candidate, "placebo": placebo},
        candidate_control_transitions={
            "corrections": 4,
            "harms": 2,
            "class1_fn_rescued": 2,
            "class1_tp_broken": 1,
        },
        control_restricted_fp=10,
        candidate_restricted_fp=8,
        fold_rows=fold_rows,
    )
    assert result["architecture_implementation_authorized"] is True
    assert result["failed_checks"] == []


def test_locked_descriptor_dimensions_are_stable() -> None:
    assert SHARED_DESCRIPTOR_DIM == 197
    assert DETAIL_DESCRIPTOR_DIM == 288

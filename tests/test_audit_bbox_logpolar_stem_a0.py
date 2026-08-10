from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from trkh.models.model import HybridConvStem
from trkh.tools.audit_bbox_logpolar_stem_a0 import (
    DESCRIPTOR_DIM,
    EXPECTED_HASHES,
    _maximum_numeric_difference,
    assess_bbox_logpolar_a0,
    build_bbox_support_grid,
    build_polar_grid,
    direction_auc,
    periodic_stem_forward,
    radial_band_descriptor,
    sample_bbox_support,
    source_derangement,
)


def test_protocol_hash_is_locked() -> None:
    assert EXPECTED_HASHES["protocol"] == (
        "dc04f38ea27f813dc9f1866e8bc9a325ca834475ae7c2b0a9ae1c45c60896f76"
    )


def test_bbox_support_grid_uses_pixel_centers_and_identity_geometry() -> None:
    bbox = torch.tensor([[0.5, 0.5, 1.0 / 1.10, 1.0 / 1.10]])
    grid = build_bbox_support_grid(bbox, output_size=8)
    expected = (torch.arange(8, dtype=torch.float32) + 0.5) / 8.0 * 2.0 - 1.0
    assert grid.shape == (1, 8, 8, 2)
    assert torch.allclose(grid[0, 0, :, 0], expected, atol=1e-7)
    assert torch.allclose(grid[0, :, 0, 1], expected, atol=1e-7)

    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    sampled = sample_bbox_support(image, bbox, output_size=8)
    assert torch.allclose(sampled, image, atol=2e-5)


def test_polar_grid_has_angle_rows_radius_columns_and_locked_endpoints() -> None:
    log_grid = build_polar_grid(2, output_size=8, mode="log")
    linear_grid = build_polar_grid(1, output_size=8, mode="linear")
    assert log_grid.shape == (2, 8, 8, 2)
    assert torch.equal(log_grid[0], log_grid[1])

    log_radius = torch.linalg.vector_norm(log_grid[0], dim=-1)
    linear_radius = torch.linalg.vector_norm(linear_grid[0], dim=-1)
    assert torch.allclose(log_radius[0], log_radius[3], atol=1e-6)
    assert torch.allclose(linear_radius[0], linear_radius[7], atol=1e-6)
    assert torch.all(torch.diff(log_radius[0]) > 0)
    assert torch.all(torch.diff(linear_radius[0]) > 0)
    assert float(log_radius[0, 0]) < float(linear_radius[0, 0])
    assert float(log_radius[0, -1]) < 1.0
    assert float(linear_radius[0, -1]) == pytest.approx(0.9375, abs=1e-6)
    assert not torch.allclose(log_grid[0, 0], log_grid[0, 1])


def test_radial_descriptor_order_and_population_std() -> None:
    features = torch.zeros(2, 256, 32, 32)
    for band in range(4):
        start = band * 8
        features[:, :, :, start : start + 4] = float(band)
        features[:, :, :, start + 4 : start + 8] = float(band + 2)
    descriptor = radial_band_descriptor(features)
    assert descriptor.shape == (2, DESCRIPTOR_DIM)
    for band in range(4):
        offset = band * 512
        assert torch.allclose(
            descriptor[:, offset : offset + 256],
            torch.full((2, 256), float(band + 1)),
        )
        assert torch.allclose(
            descriptor[:, offset + 256 : offset + 512],
            torch.ones(2, 256),
        )


def test_periodic_stem_is_equivariant_to_pool_aligned_angular_roll() -> None:
    torch.manual_seed(7)
    stem = HybridConvStem(in_channels=3, stem_channels=4, embed_dim=256).eval()
    values = torch.randn(2, 3, 64, 64)
    output = periodic_stem_forward(stem, values)
    shifted = periodic_stem_forward(stem, torch.roll(values, shifts=8, dims=2))
    assert output.shape == (2, 256, 8, 8)
    assert torch.allclose(shifted, torch.roll(output, shifts=1, dims=2), atol=2e-6)


def test_periodic_stem_matches_native_when_wrapped_boundary_is_zero() -> None:
    torch.manual_seed(11)
    stem = HybridConvStem(in_channels=3, stem_channels=4, embed_dim=256).eval()
    values = torch.randn(1, 3, 64, 64)
    values[:, :, :16] = 0.0
    values[:, :, -16:] = 0.0
    native = stem(values)
    periodic = periodic_stem_forward(stem, values)
    assert torch.allclose(periodic, native, atol=2e-6)


def test_source_derangement_is_deterministic_and_changes_every_source() -> None:
    indices = np.arange(12, dtype=np.int64)
    sources = np.asarray([f"source_{index}" for index in indices], dtype=object)
    first = source_derangement(indices, sources, seed=19)
    second = source_derangement(indices, sources, seed=19)
    assert np.array_equal(first, second)
    assert np.all(sources[first] != sources[indices])
    assert sorted(first.tolist()) == indices.tolist()


def test_direction_auc_uses_all_true_class1_and_restricted_keeper_fp() -> None:
    targets = np.asarray([1, 1, 0, 2, 4, 3], dtype=np.int64)
    keeper = np.asarray(
        [
            [0.1, 0.8, 0.1, 0.0, 0.0],
            [0.6, 0.3, 0.1, 0.0, 0.0],
            [0.2, 0.7, 0.1, 0.0, 0.0],
            [0.2, 0.6, 0.2, 0.0, 0.0],
            [0.1, 0.6, 0.0, 0.0, 0.3],
            [0.0, 0.1, 0.0, 0.9, 0.0],
        ]
    )
    control = keeper.copy()
    candidate = keeper.copy()
    candidate[:2, 1] += 0.1
    candidate[2:5, 1] -= 0.1
    report = direction_auc(targets, keeper, control, candidate)
    assert report["positive_true_class1"] == 2
    assert report["positive_keeper_tp"] == 1
    assert report["positive_keeper_fn"] == 1
    assert report["negative_restricted_keeper_fp"] == 3
    assert report["auc"] == pytest.approx(1.0)


def _passing_results() -> dict[str, object]:
    role_metrics = {
        "macro_f1": 0.91,
        "per_class_precision": [0.9, 0.75, 0.9, 0.9, 0.9],
        "per_class_recall": [0.9, 0.80, 0.9, 0.9, 0.9],
        "per_class_f1": [0.9, 0.775, 0.9, 0.9, 0.9],
    }
    condition = {
        "metrics": {
            "keeper": copy.deepcopy(role_metrics),
            "cartesian": copy.deepcopy(role_metrics),
            "candidate": copy.deepcopy(role_metrics),
            "linear": copy.deepcopy(role_metrics),
            "source_placebo": copy.deepcopy(role_metrics),
        },
        "deltas": {
            "candidate_vs_cartesian": {
                "macro_f1": 0.01,
                "class1_precision": 0.03,
                "class1_recall": 0.01,
                "class1_f1": 0.02,
            },
            "candidate_vs_linear": {
                "macro_f1": 0.003,
                "class1_precision": 0.006,
                "class1_recall": 0.0,
                "class1_f1": 0.006,
            },
            "candidate_vs_source_placebo": {
                "macro_f1": 0.003,
                "class1_precision": 0.006,
                "class1_recall": 0.0,
                "class1_f1": 0.006,
            },
        },
        "comparisons": {
            "candidate_vs_cartesian": {
                "corrections": 20,
                "harms": 10,
                "focus_fn_rescue": 4,
                "focus_tp_break": 2,
                "restricted_fp": {"net_reduction": 12},
            }
        },
        "directions": {
            "candidate": {"auc": 0.70},
            "linear": {"auc": 0.64},
            "source_placebo": {"auc": 0.63},
        },
    }
    return {
        "clean": copy.deepcopy(condition),
        "lighting_dim": copy.deepcopy(condition),
        "lighting_bright": copy.deepcopy(condition),
        "low_contrast": copy.deepcopy(condition),
    }


def test_gate_is_conjunctive_and_rejects_precision_failure() -> None:
    structural = {"structure": True}
    folds = [
        {
            "candidate_vs_cartesian": {
                "macro_f1": 0.01,
                "class1_precision": 0.02,
                "class1_recall": 0.0,
                "class1_f1": 0.01,
            }
        }
        for _ in range(5)
    ]
    passing = assess_bbox_logpolar_a0(
        structural_checks=structural,
        results=_passing_results(),
        fold_metrics=folds,
    )
    assert passing["automatic_gates_passed"] is True

    failing_results = _passing_results()
    failing_results["clean"]["deltas"]["candidate_vs_cartesian"][
        "class1_precision"
    ] = 0.014
    failing = assess_bbox_logpolar_a0(
        structural_checks=structural,
        results=failing_results,
        fold_metrics=folds,
    )
    assert failing["automatic_gates_passed"] is False
    assert "clean_class1_precision_delta_gte_0p015" in failing["failed_checks"]
    assert failing["full_train_authorized"] is False


def test_maximum_numeric_difference_checks_nested_replay() -> None:
    left = {"a": [1.0, {"b": True}], "c": "fixed"}
    right = {"a": [1.0 + 1e-9, {"b": True}], "c": "fixed"}
    assert _maximum_numeric_difference(left, right) == pytest.approx(1e-9)
    with pytest.raises(ValueError, match="mapping keys"):
        _maximum_numeric_difference(left, {"different": 1})


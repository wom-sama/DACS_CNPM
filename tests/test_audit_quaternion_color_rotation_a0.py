from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from trkh.tools.audit_hamburger_nmf_surface_a0 import (
    _global_rng_equal,
    _global_rng_snapshot,
)
from trkh.tools.audit_more_model_rebalancing_readiness import _read_clean_train_rows
from trkh.tools.audit_quaternion_color_rotation_a0 import (
    BASE_DIM,
    COVARIANCE_DIM,
    EXPECTED_ORDERED_INDEX_SHA256,
    FOLDS,
    REPO_ROOT,
    ROLE_NAMES,
    QuaternionConv2d,
    _cohort_index_sha256,
    _parameter_sha256,
    _sha256,
    _write_csv,
    _write_json,
    _write_manifest,
    _verify_manifest,
    build_fold_models,
    channel_dephase,
    crop_inset_rgb,
    effective_rank,
    engineering_checks,
    finalize_visual_review,
    gray_rotation_matrix_torch,
    locked_cohort,
    parameter_count,
    positive_threshold,
    replay_artifacts,
    rgb_grid_covariance_features,
    rodrigues_rotation_matrix_numpy,
    role_metrics,
)


def test_gray_equation_matches_independent_rodrigues_fp64() -> None:
    generator = np.random.default_rng(5)
    scale = generator.uniform(-2.0, 2.0, size=(3, 2, 4))
    theta = generator.uniform(-math.pi, math.pi, size=scale.shape)
    expected = rodrigues_rotation_matrix_numpy(scale, theta)
    observed = gray_rotation_matrix_torch(
        torch.from_numpy(scale), torch.from_numpy(theta)
    ).numpy()
    assert np.max(np.abs(expected - observed)) <= 1e-12


def test_gray_projection_grayscale_and_cyclic_symmetry() -> None:
    scale = torch.tensor([0.4, 1.2], dtype=torch.float64)
    theta = torch.tensor([-1.1, 0.7], dtype=torch.float64)
    matrices = gray_rotation_matrix_torch(scale, theta).numpy()
    gray = np.ones(3) / math.sqrt(3.0)
    cyclic = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    vector = np.asarray([0.2, -0.7, 1.4])
    for matrix, magnitude in zip(matrices, scale.numpy()):
        assert gray @ (matrix @ vector) == pytest.approx(
            magnitude * (gray @ vector), abs=1e-12
        )
        assert np.max(np.abs(matrix @ gray - magnitude * gray)) <= 1e-12
        assert np.max(np.abs(matrix @ cyclic - cyclic @ matrix)) <= 1e-12


def test_quaternion_conv_matches_oracle_and_static_multigroup() -> None:
    module = QuaternionConv2d(
        2,
        3,
        3,
        stride=2,
        padding=1,
        generator=torch.Generator().manual_seed(41),
    ).double()
    value = torch.randn(2, 6, 11, 13, dtype=torch.float64)
    blocks = rodrigues_rotation_matrix_numpy(
        module.scale.detach().numpy(), module.theta.detach().numpy()
    )
    weight = blocks.transpose(0, 4, 1, 5, 2, 3).reshape(9, 6, 3, 3)
    expected = F.conv2d(
        value,
        torch.from_numpy(weight),
        module.bias,
        stride=2,
        padding=1,
    )
    observed = module(value)
    assert torch.max(torch.abs(expected - observed)).item() <= 1e-12
    assert torch.equal(observed, module.materialized_conv()(value))
    assert module(value[:1]).shape[0] == 1


def test_quaternion_invalid_inputs_and_finite_difference_gradient() -> None:
    with pytest.raises(ValueError):
        gray_rotation_matrix_torch(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        gray_rotation_matrix_torch(torch.tensor([float("nan")]), torch.ones(1))
    module = QuaternionConv2d(1, 1, 3, generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError):
        module(torch.ones(1, 4, 8, 8))
    scale = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    theta = torch.tensor([0.31], dtype=torch.float64, requires_grad=True)
    value = gray_rotation_matrix_torch(scale, theta).square().sum()
    value.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()


def test_dephase_is_deterministic_marginal_preserving_and_rng_free() -> None:
    value = torch.arange(2 * 3 * 17 * 19, dtype=torch.float32).reshape(2, 3, 17, 19)
    before = _global_rng_snapshot()
    first = channel_dephase(value, (13, 29))
    second = channel_dephase(value, (13, 29))
    after = _global_rng_snapshot()
    assert torch.equal(first, second)
    assert _global_rng_equal(before, after)
    for row in range(2):
        for channel in range(3):
            assert torch.equal(
                torch.sort(first[row, channel].flatten()).values,
                torch.sort(value[row, channel].flatten()).values,
            )


def test_crop_geometry_is_floor_ceil_inset_and_resize_exact() -> None:
    rgb = torch.linspace(0.0, 1.0, 3 * 100 * 200).reshape(1, 3, 100, 200)
    bbox = torch.tensor([[0.5, 0.5, 0.5, 0.4]], dtype=torch.float32)
    mask = torch.ones(1, 1, 100, 200, dtype=torch.bool)
    crop, geometry = crop_inset_rgb(rgb, bbox, mask)
    assert crop.shape == (1, 3, 64, 64)
    assert geometry[0] == {
        "original_x0": 50,
        "original_y0": 29,
        "original_x1_exclusive": 150,
        "original_y1_exclusive": 71,
        "inset_x0": 60,
        "inset_y0": 34,
        "inset_x1_exclusive": 140,
        "inset_y1_exclusive": 66,
        "valid_fraction": 1.0,
    }


def test_crop_rejects_low_valid_coverage() -> None:
    rgb = torch.ones(1, 3, 64, 64)
    bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid fraction"):
        crop_inset_rgb(rgb, bbox, mask)


def test_rgb_grid_covariance_dimension_and_finiteness() -> None:
    cache = np.arange(3 * 3 * 64 * 64, dtype=np.uint8).reshape(3, 3, 64, 64)
    features = rgb_grid_covariance_features(cache)
    assert features.shape == (3, COVARIANCE_DIM)
    assert np.isfinite(features).all()


def test_fold_models_are_matched_rng_restoring_and_parameter_matched() -> None:
    before = _global_rng_snapshot()
    models = build_fold_models()
    after = _global_rng_snapshot()
    assert _global_rng_equal(before, after)
    candidate = models["quaternion_gray_base"]
    hashes = {
        _parameter_sha256(models[role].trunk)
        for role in (
            "quaternion_gray_base",
            "quaternion_red_axis_base",
            "quaternion_dephased_base",
            "quaternion_gray_only",
        )
    }
    assert len(hashes) == 1
    assert torch.equal(
        candidate.head.weight[:, :48],
        models["quaternion_gray_only"].head.weight[:, :48],
    )
    relative = abs(
        parameter_count(candidate) - parameter_count(models["real_cnn_base"])
    ) / parameter_count(candidate)
    assert relative <= 0.05


def test_static_surface_model_is_exact() -> None:
    model = build_fold_models()["quaternion_gray_base"].eval()
    static = model.materialized().eval()
    rgb = torch.rand(2, 3, 64, 64)
    base = torch.rand(2, BASE_DIM)
    with torch.inference_mode():
        assert torch.max(torch.abs(model(rgb, base) - static(rgb, base))).item() <= 1e-6


def test_positive_threshold_retains_locked_fit_fraction() -> None:
    positive = np.linspace(0.01, 0.99, 100)
    scores = np.concatenate((positive, np.linspace(0.0, 1.0, 50)))
    labels = np.concatenate((np.ones(100, dtype=np.int64), np.zeros(50, dtype=np.int64)))
    threshold = positive_threshold(scores, labels)
    assert np.mean(positive >= threshold) >= 0.97
    assert threshold == pytest.approx(positive[3])


def test_role_metrics_and_effective_rank() -> None:
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 10)
    scores = np.linspace(0.0, 1.0, 20)
    actions = scores >= 0.5
    folds = np.repeat(np.arange(5), 4)
    result = role_metrics(scores, actions, labels, folds)
    assert 0.0 <= result["auroc"] <= 1.0
    assert result["corrections"] == result["fp_rejected"]
    matrix = np.eye(20, 16)
    assert effective_rank(matrix) >= 15.0


def test_locked_train_cohort_order() -> None:
    path = (
        REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv"
    )
    cohort = locked_cohort(_read_clean_train_rows(path))
    assert len(cohort) == 750
    assert _cohort_index_sha256([row.sample_index for row in cohort]) == EXPECTED_ORDERED_INDEX_SHA256


def test_engineering_check_bundle_passes() -> None:
    result = engineering_checks()
    assert result["passed"]
    assert all(result["checks"].values())


def test_replay_reconstructs_scores_and_actions(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv"
    )
    cohort = locked_cohort(_read_clean_train_rows(source))
    features = {}
    heads = {}
    metadata = {"folds": {}}
    rows = []
    positions_by_fold = {
        fold: [index for index, row in enumerate(cohort) if row.fold == fold]
        for fold in FOLDS
    }
    for role_index, role in enumerate(ROLE_NAMES):
        matrix = np.stack(
            (
                np.linspace(-1.0, 1.0, len(cohort)),
                np.full(len(cohort), 0.01 * role_index),
            ),
            axis=1,
        )
        features[role] = matrix
    for fold in FOLDS:
        role_metadata = {}
        for role_index, role in enumerate(ROLE_NAMES):
            weight_key = f"fold{fold}__{role}__weight"
            bias_key = f"fold{fold}__{role}__bias"
            heads[weight_key] = np.asarray([[0.5, -0.25]], dtype=np.float64)
            heads[bias_key] = np.asarray([0.01 * fold], dtype=np.float64)
            role_metadata[role] = {
                "held_positions": positions_by_fold[fold],
                "threshold": 0.5,
                "weight_key": weight_key,
                "bias_key": bias_key,
            }
        metadata["folds"][str(fold)] = {"roles": role_metadata}
    for position, row in enumerate(cohort):
        record = {
            "position": position,
            "sample_index": row.sample_index,
            "source_stem": row.source_stem,
            "fold": row.fold,
            "target": row.target,
            "binary_label": int(row.target == 1),
        }
        for role in ROLE_NAMES:
            weight = heads[f"fold{row.fold}__{role}__weight"]
            bias = heads[f"fold{row.fold}__{role}__bias"][0]
            logit = float(features[role][position] @ weight.reshape(-1) + bias)
            score = 1.0 / (1.0 + math.exp(-logit))
            record[f"score__{role}"] = score
            record[f"action__{role}"] = int(score >= 0.5)
        rows.append(record)
    np.savez_compressed(tmp_path / "oof_replay_features.npz", **features)
    np.savez_compressed(tmp_path / "oof_head_states.npz", **heads)
    _write_json(tmp_path / "oof_state_metadata.json", metadata)
    _write_csv(tmp_path / "oof_predictions.csv", rows)
    replay = replay_artifacts(tmp_path)
    assert replay["passed"]
    assert replay["max_score_error"] <= 1e-15
    assert replay["exact_actions"]


def test_manifest_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "value.txt"
    artifact.write_text("locked", encoding="utf-8")
    _write_manifest(tmp_path)
    assert _verify_manifest(tmp_path)["passed"]
    artifact.write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        _verify_manifest(tmp_path)


def test_visual_pass_cannot_rescue_failed_automatic_gate(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {
            "status": "complete_pending_visual_review",
            "pre_manual_passed": False,
            "manual_visual_review": {"decision": "pending"},
            "passed": False,
            "advancement_authorized": False,
        },
    )
    _write_manifest(tmp_path)
    expected = _sha256(summary_path)
    result = finalize_visual_review(
        summary_path=summary_path,
        decision="pass",
        expected_summary_sha256=expected,
    )
    assert not result["passed"]
    assert not result["advancement_authorized"]

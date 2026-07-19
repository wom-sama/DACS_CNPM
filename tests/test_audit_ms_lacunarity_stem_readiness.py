from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.audit_ms_lacunarity_stem_readiness import (
    EXPECTED_CANDIDATE_DIM,
    EXPECTED_CANDIDATE_ONLY_DIM,
    EXPECTED_CONTROL_DIM,
    _aligned_block_correlations,
    _effective_rank,
    _fit_readout,
    _restricted_fp_transitions,
    assess_ms_lacunarity_readiness,
    extract_ms_lacunarity_descriptors,
    gaussian_pyrdown,
    weighted_mean_lacunarity,
)


def _metrics(
    *,
    macro_f1: float,
    focus_precision: float,
    focus_recall: float,
    focus_f1: float,
    nonfocus_f1: float = 0.8,
) -> dict[str, object]:
    per_class = []
    for index in range(5):
        if index == 1:
            precision = focus_precision
            recall = focus_recall
            f1 = focus_f1
        else:
            precision = nonfocus_f1
            recall = nonfocus_f1
            f1 = nonfocus_f1
        per_class.append(
            {
                "class_index": index,
                "support": 100,
                "predicted": 100,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"macro_f1": macro_f1, "per_class": per_class}


def _passing_gate_kwargs() -> dict[str, object]:
    return {
        "train_rows": 7372,
        "holdout_rows": 1843,
        "train_holdout_source_overlap": 0,
        "fold_source_overlap": 0,
        "features_finite": True,
        "control_dim": EXPECTED_CONTROL_DIM,
        "candidate_dim": EXPECTED_CANDIDATE_DIM,
        "candidate_only_effective_rank": 12.0,
        "all_converged": True,
        "peak_cuda_memory_mib": 5000.0,
        "folds_with_focus_gain": 4,
        "oof_metrics": {
            "control": _metrics(
                macro_f1=0.700,
                focus_precision=0.60,
                focus_recall=0.70,
                focus_f1=0.646,
            ),
            "candidate": _metrics(
                macro_f1=0.705,
                focus_precision=0.63,
                focus_recall=0.70,
                focus_f1=0.663,
            ),
        },
        "holdout_metrics": {
            "control": _metrics(
                macro_f1=0.700,
                focus_precision=0.60,
                focus_recall=0.70,
                focus_f1=0.646,
            ),
            "candidate": _metrics(
                macro_f1=0.710,
                focus_precision=0.65,
                focus_recall=0.70,
                focus_f1=0.674,
            ),
        },
        "holdout_transitions": {
            "changed": 20,
            "corrections": 10,
            "harms": 5,
            "neutral": 5,
            "focus_false_positive_removed": 7,
            "focus_false_positive_created": 1,
            "focus_false_negative_rescued": 3,
            "focus_true_positive_broken": 1,
        },
        "restricted_fp": {
            "control": 20,
            "candidate": 15,
            "removed": 6,
            "created": 1,
        },
        "oof_direction_auc": 0.70,
        "holdout_direction_auc": 0.72,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }


def test_gaussian_pyrdown_preserves_uniform_values() -> None:
    values = torch.full((2, 3, 8, 8), 7.5)
    output = gaussian_pyrdown(values)
    assert output.shape == (2, 3, 4, 4)
    assert torch.allclose(output, torch.full_like(output, 7.5), atol=1e-6)


def test_weighted_lacunarity_matches_independent_direct_moments() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.rand((2, 4, 7, 7), generator=generator) * 255.0
    mask = (torch.rand((2, 1, 7, 7), generator=generator) > 0.25).float()
    mean, lacunarity, support = weighted_mean_lacunarity(values, mask)

    expected_support = mask.sum(dim=(2, 3)).clamp_min(1.0)
    expected_mean = (values * mask).sum(dim=(2, 3)) / expected_support
    expected_second = (values.square() * mask).sum(dim=(2, 3)) / expected_support
    expected_lacunarity = expected_second / (expected_mean.square() + 1e-5) - 1.0
    assert torch.allclose(support, expected_support, atol=0.0, rtol=0.0)
    assert torch.allclose(mean, expected_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(lacunarity, expected_lacunarity, atol=1e-6, rtol=1e-6)


def test_uniform_map_has_zero_lacunarity_and_clustered_map_is_positive() -> None:
    mask = torch.ones((1, 1, 8, 8))
    uniform = torch.full((1, 1, 8, 8), 127.5)
    clustered = torch.zeros((1, 1, 8, 8))
    clustered[:, :, :4] = 255.0
    _, uniform_lacunarity, _ = weighted_mean_lacunarity(uniform, mask)
    _, clustered_lacunarity, _ = weighted_mean_lacunarity(clustered, mask)
    assert abs(float(uniform_lacunarity.item())) <= 1e-6
    assert float(clustered_lacunarity.item()) > 0.9


def test_descriptor_contract_is_finite_and_exact() -> None:
    generator = torch.Generator().manual_seed(11)
    stem = torch.randn((2, 256, 32, 32), generator=generator)
    crop_bbox = torch.tensor(
        [[0.50, 0.50, 0.75, 0.60], [0.45, 0.55, 0.50, 0.70]],
        dtype=torch.float32,
    )
    valid = torch.ones((2, 256, 256), dtype=torch.bool)
    payload = extract_ms_lacunarity_descriptors(stem, crop_bbox, valid)
    assert payload["control"].shape == (2, EXPECTED_CONTROL_DIM)
    assert payload["candidate_only"].shape == (2, EXPECTED_CANDIDATE_ONLY_DIM)
    assert payload["candidate"].shape == (2, EXPECTED_CANDIDATE_DIM)
    assert torch.equal(
        payload["candidate"][:, :EXPECTED_CONTROL_DIM], payload["control"]
    )
    assert all(torch.isfinite(value).all() for value in payload.values())
    assert float(payload["object_support"].min()) >= 9.0


def test_descriptor_rejects_wrong_stem_channels() -> None:
    with pytest.raises(ValueError, match="256"):
        extract_ms_lacunarity_descriptors(
            torch.zeros((1, 64, 32, 32)),
            torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            None,
        )


def test_effective_rank_and_aligned_correlations_are_finite() -> None:
    rng = np.random.default_rng(13)
    control = rng.normal(size=(128, EXPECTED_CONTROL_DIM)).astype(np.float32)
    candidate_only = rng.normal(
        size=(128, EXPECTED_CANDIDATE_ONLY_DIM)
    ).astype(np.float32)
    rank = _effective_rank(candidate_only, maximum_rows=128)
    correlations = _aligned_block_correlations(control, candidate_only)
    assert rank > 20.0
    assert 0.0 <= correlations["all"]["mean_abs"] <= 1.0
    assert 0.0 <= correlations["all"]["max_abs"] <= 1.0


def test_readout_restores_absent_class_probability_column() -> None:
    rng = np.random.default_rng(17)
    fit_features = rng.normal(size=(60, 8)).astype(np.float32)
    fit_labels = np.repeat(np.asarray([0, 1, 2, 4], dtype=np.int64), 15)
    eval_features = rng.normal(size=(7, 8)).astype(np.float32)
    probabilities, record = _fit_readout(
        fit_features,
        fit_labels,
        eval_features,
        c_value=0.3,
        max_iterations=2000,
        seed=17,
    )
    assert probabilities.shape == (7, 5)
    assert np.all(probabilities[:, 3] == 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert record["converged"] is True


def test_restricted_fp_transitions_exclude_class_three() -> None:
    targets = np.asarray([0, 2, 4, 3, 1, 0], dtype=np.int64)
    control_predictions = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.int64)
    candidate_predictions = np.asarray([0, 1, 1, 0, 1, 1], dtype=np.int64)
    control = np.eye(5, dtype=np.float32)[control_predictions]
    candidate = np.eye(5, dtype=np.float32)[candidate_predictions]
    result = _restricted_fp_transitions(targets, control, candidate)
    assert result == {"control": 2, "candidate": 3, "removed": 1, "created": 2}


def test_gate_accepts_precision_first_candidate() -> None:
    gate = assess_ms_lacunarity_readiness(**_passing_gate_kwargs())
    assert gate["condition_replay_permission"] is True
    assert gate["image_smoke_permission"] is False
    assert gate["full_train_permission"] is False
    assert gate["failed_checks"] == []


def test_gate_rejects_precision_and_recall_regression() -> None:
    kwargs = _passing_gate_kwargs()
    kwargs["holdout_metrics"] = {
        "control": _metrics(
            macro_f1=0.700,
            focus_precision=0.65,
            focus_recall=0.70,
            focus_f1=0.674,
        ),
        "candidate": _metrics(
            macro_f1=0.701,
            focus_precision=0.61,
            focus_recall=0.66,
            focus_f1=0.634,
        ),
    }
    gate = assess_ms_lacunarity_readiness(**kwargs)
    assert gate["condition_replay_permission"] is False
    assert "holdout_focus_precision_gain_ge_0p025" in gate["failed_checks"]
    assert "holdout_focus_recall_preserved" in gate["failed_checks"]
    assert "holdout_focus_f1_gain_ge_0p015" in gate["failed_checks"]

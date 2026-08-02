from __future__ import annotations

import numpy as np
import torch

from trkh.tools.audit_dinov3_depth_trajectory_b12 import (
    DEFAULT_BATCH_SIZE,
    EMBED_DIM,
    FEATURES_PER_ARM,
    PATCH_COUNT,
    RIVALS,
    _pair_metrics,
    assess_readiness,
    depth_trajectory_features,
    pair_features,
    summarize_pair_maps,
)


def test_extraction_batch_matches_locked_a0_cache_batch() -> None:
    assert DEFAULT_BATCH_SIZE == 32


def test_locked_map_summary_uses_linear_patch_quantiles() -> None:
    base = torch.arange(PATCH_COUNT, dtype=torch.float32)
    maps = base.reshape(1, 1, 1, PATCH_COUNT).repeat(2, 2, len(RIVALS), 1)
    output = summarize_pair_maps(maps).reshape(2, 2, len(RIVALS), 5)
    assert output.shape == (2, 2, 3, 5)
    assert torch.allclose(output[..., 2], torch.full_like(output[..., 2], 25.5))
    assert torch.allclose(output[..., 3], torch.full_like(output[..., 3], 127.5))
    assert torch.allclose(output[..., 4], torch.full_like(output[..., 4], 229.5))
    assert torch.allclose(output[..., 1], torch.full_like(output[..., 1], base.std(unbiased=False)))


def test_depth_trajectory_arms_are_finite_equal_width_and_distinct() -> None:
    generator = torch.Generator().manual_seed(7)
    x2 = torch.randn(2, PATCH_COUNT, EMBED_DIM, generator=generator)
    spatial = torch.linspace(-1.0, 1.0, PATCH_COUNT).reshape(1, PATCH_COUNT, 1)
    direction = torch.zeros(len(RIVALS), EMBED_DIM)
    direction[:, : len(RIVALS)] = torch.eye(len(RIVALS))
    x5 = x2 + spatial * direction[0].reshape(1, 1, -1)
    x8 = x5 + spatial.square() * direction[1].reshape(1, 1, -1)
    arms = depth_trajectory_features(x2, x5, x8, direction)
    assert set(arms) == {"candidate", "latest", "deranged"}
    assert all(value.shape == (2, FEATURES_PER_ARM) for value in arms.values())
    assert all(torch.isfinite(value).all() for value in arms.values())
    assert not torch.equal(arms["candidate"], arms["latest"])
    assert not torch.equal(arms["candidate"], arms["deranged"])


def test_pair_feature_selection_preserves_two_maps_and_five_stats() -> None:
    features = np.arange(4 * FEATURES_PER_ARM, dtype=np.float32).reshape(4, -1)
    selected = pair_features(features, 1)
    shaped = features.reshape(4, 2, 3, 5)
    np.testing.assert_array_equal(selected, shaped[:, :, 1, :].reshape(4, 10))


def test_pair_metrics_treats_class1_as_positive_at_zero() -> None:
    metrics = _pair_metrics(
        np.asarray([1, 1, 0, 0], dtype=np.int64),
        np.asarray([2.0, -1.0, 1.0, -2.0], dtype=np.float64),
    )
    assert metrics["class1_tp"] == 1
    assert metrics["class1_fn"] == 1
    assert metrics["rival_fp"] == 1
    assert metrics["class1_f1"] == 0.5


def _mock_summary(mean_auc: float, tp: int, fp: int, pair_delta: float = 0.0):
    pairs = {
        str(rival): {"auroc": mean_auc + pair_delta, "class1_tp": tp // 3, "rival_fp": fp // 3}
        for rival in RIVALS
    }
    folds = [
        {"fold": fold, "rival": rival, "auroc": mean_auc + pair_delta}
        for fold in range(5)
        for rival in RIVALS
    ]
    return {
        "mean_pair_auroc": mean_auc + pair_delta,
        "aggregate_class1_tp": tp,
        "aggregate_rival_fp": fp,
        "pairs": pairs,
        "fold_pairs": folds,
    }


def test_readiness_is_conjunctive_and_never_opens_validation() -> None:
    summaries = {
        "base": _mock_summary(0.9950, 1400, 90),
        "latest": _mock_summary(0.9952, 1400, 80),
        "deranged": _mock_summary(0.9951, 1390, 85),
        "candidate": _mock_summary(0.9965, 1390, 75),
    }
    deltas = [
        {"fold": fold, "rival": rival, "candidate_minus_latest_auroc": 0.0013}
        for fold in range(5)
        for rival in RIVALS
    ]
    bootstrap = {
        "candidate_vs_latest": {"lower": 0.0001},
        "candidate_vs_deranged": {"lower": 0.0001},
        "candidate_vs_base": {"lower": 0.0001},
    }
    result = assess_readiness(
        summaries=summaries,
        bootstrap=bootstrap,
        pair_fold_deltas=deltas,
        integrity_ok=True,
    )
    assert result["signal_gate_passed"] is True
    assert result["b12_model_preregistration_permission"] is True
    assert result["validation_permission"] is False
    assert result["test_permission"] is False

    bootstrap["candidate_vs_base"]["lower"] = -1e-6
    failed = assess_readiness(
        summaries=summaries,
        bootstrap=bootstrap,
        pair_fold_deltas=deltas,
        integrity_ok=True,
    )
    assert failed["signal_gate_passed"] is False
    assert failed["exact_b12_signal_family_closed"] is True

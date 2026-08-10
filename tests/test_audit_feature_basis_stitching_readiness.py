from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.audit_feature_basis_stitching_readiness import (
    AffineMap,
    DirectMatchMetrics,
    LinearPairStats,
    aligned_token_pairs,
    apply_fold_maps,
    choose_oof_fork,
    frozen_precision_rule,
)


def test_linear_pair_stats_recovers_affine_map_and_supports_subtraction() -> None:
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(512, 4, generator=generator)
    weight = torch.tensor(
        [
            [1.2, -0.2, 0.0],
            [0.1, 0.8, 0.3],
            [-0.4, 0.2, 1.1],
            [0.5, 0.0, -0.1],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.3, -0.7, 0.2])
    y = x @ weight + bias

    first = LinearPairStats(4, 3)
    second = LinearPairStats(4, 3)
    first.update(x[:256], y[:256])
    second.update(x[256:], y[256:])
    total = first.clone().add_(second)
    recovered = total.solve(ridge_ratio=1e-10)

    assert recovered.apply(x).allclose(y, atol=2e-5, rtol=2e-5)
    restored = total.clone().add_(second, scale=-1.0)
    assert restored.count == first.count
    assert torch.allclose(restored.xtx, first.xtx)
    assert 0.0 < total.cka() <= 1.0


def test_aligned_token_pairs_matches_prefix_and_patch_indices_in_keeper_order() -> None:
    keeper_tokens = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2)
    scratch_tokens = (torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2) + 100)
    keeper_indices = torch.tensor([9, 2, 5, 7])
    scratch_indices = torch.tensor([5, 9, 1, 2])

    prefix_x, prefix_y, patch_x, patch_y = aligned_token_pairs(
        keeper_tokens,
        keeper_indices,
        scratch_tokens,
        scratch_indices,
        prefix_count=3,
    )

    assert torch.equal(prefix_x, keeper_tokens[:3])
    assert torch.equal(prefix_y, scratch_tokens[:3])
    assert torch.equal(patch_x, keeper_tokens[3:][torch.tensor([0, 1, 2])])
    assert torch.equal(patch_y, scratch_tokens[3:][torch.tensor([1, 3, 0])])


def test_apply_fold_maps_routes_rows_without_changing_extra_dimensions() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    maps = {
        0: AffineMap(torch.eye(2), torch.tensor([1.0, 1.0]), 0.0),
        1: AffineMap(torch.eye(2) * 2.0, torch.tensor([-1.0, -1.0]), 0.0),
    }

    output = apply_fold_maps(values, np.asarray([0, 1]), maps)

    assert torch.equal(output[0], values[0] + 1.0)
    assert torch.equal(output[1], values[1] * 2.0 - 1.0)


def test_direct_match_metrics_reports_perfect_fit() -> None:
    values = torch.randn(20, 5, generator=torch.Generator().manual_seed(3))
    metrics = DirectMatchMetrics(5)
    metrics.update(values, values)
    summary = metrics.summary()
    assert summary["r2"] == pytest.approx(1.0)
    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["std_ratio"] == pytest.approx(1.0)


def test_frozen_precision_rule_applies_locked_weight_offset_and_normalization() -> None:
    keeper = np.asarray([[0.2, 0.5, 0.1, 0.1, 0.1]], dtype=np.float64)
    candidate = np.asarray([[0.1, 0.7, 0.05, 0.1, 0.05]], dtype=np.float64)
    result = frozen_precision_rule(keeper, candidate)
    expected = keeper * 0.60 + candidate * 0.40
    expected[:, 1] -= 0.034
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(result, expected)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_choose_oof_fork_uses_mae_then_later_fork_tie_break() -> None:
    summaries = {
        5: {"eligible": True, "functional": {"probability_mae": 0.02000}},
        6: {"eligible": True, "functional": {"probability_mae": 0.02005}},
        7: {"eligible": False, "functional": {"probability_mae": 0.001}},
    }
    assert choose_oof_fork(summaries) == 6
    summaries[6]["functional"]["probability_mae"] = 0.021
    assert choose_oof_fork(summaries) == 5
    summaries[5]["eligible"] = False
    summaries[6]["eligible"] = False
    assert choose_oof_fork(summaries) is None

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trkh.tools.audit_bi_level_routing_attention_preflight import (
    _attention_integrity,
    _baseline_config,
    _bbox_masks,
    _bra_config,
    _dense_mhsa_parity,
    _condition_summary,
    _independent_sparse_scatter,
    _official_equation_replay,
    _route_set_comparison,
    _single_route_geometry,
)
from trkh.models.bi_level_routing_attention import BiLevelRoutingAttention


OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\biformer-cvpr2023")


def test_bra_preflight_configs_lock_only_the_causal_topk_difference() -> None:
    source = {
        "model_type": "vit_registers",
        "early_token_mask_keep_rate": 0.5,
        "deformable_spatial_attention": True,
    }
    control = _bra_config(source, topk=16)
    candidate = _bra_config(source, topk=4)
    differing = {
        key for key in set(control).union(candidate) if control.get(key) != candidate.get(key)
    }
    assert differing == {"bi_level_routing_attention_topk"}
    assert control["bi_level_routing_attention_layers"] == "2"
    assert control["bi_level_routing_attention_regions_per_axis"] == 4
    assert control["bi_level_routing_attention_local_context_kernel_size"] == 5
    assert control["early_token_mask_keep_rate"] == 1.0
    assert not control["deformable_spatial_attention"]
    baseline = _baseline_config(source)
    assert not baseline["bi_level_routing_attention"]


def test_bbox_masks_use_token_centers_and_one_patch_far_margin() -> None:
    foreground, far = _bbox_masks(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert int(foreground.sum()) == 64
    assert int(far.sum()) == 156
    assert not bool((foreground & far).any())


def test_route_geometry_reports_foreground_selection_and_query_diversity() -> None:
    module = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(16, 16),
        num_heads=4,
        regions_per_axis=4,
        topk=4,
        local_context_kernel_size=5,
    )
    routes = torch.tensor(
        [[index, (index + 1) % 16, (index + 4) % 16, (index + 5) % 16] for index in range(16)]
    )
    summary = _single_route_geometry(
        routes,
        torch.tensor([0.5, 0.5, 0.5, 0.5]),
        module.region_token_indices,
    )
    assert summary["valid_object_query"]
    assert summary["object_token_count"] == 64
    assert summary["foreground_gain"] > 0.0
    assert summary["far_background_reduction"] > 0.0
    assert summary["distinct_route_sets"] == 16
    assert 0.0 <= summary["pairwise_route_jaccard"] <= 1.0
    assert summary["nonlocal_route_fraction"] > 0.0


def test_route_set_comparison_is_order_invariant() -> None:
    left = torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    right = torch.tensor([[[3, 2, 1, 0], [4, 5, 8, 9]]])
    summary = _route_set_comparison(left, right)
    assert summary["exact_fraction"] == pytest.approx(0.5)
    assert summary["mean_jaccard"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert summary["disagreement_count"] == 1


def test_condition_summary_reports_valid_geometry_without_hiding_invalid_rows() -> None:
    base = {
        "valid_object_query": True,
        "foreground_gain": 0.04,
        "far_background_reduction": 0.03,
        "routed_foreground_fraction": 0.30,
        "all_region_foreground_fraction": 0.26,
        "routed_far_background_fraction": 0.55,
        "all_region_far_background_fraction": 0.58,
        "distinct_route_sets": 6,
        "pairwise_route_jaccard": 0.5,
        "nonlocal_route_fraction": 0.4,
        "probability_mae": 0.003,
        "clean_route_jaccard": "",
        "selected_affinity_margin": 1.0,
    }
    invalid = dict(base)
    invalid.update(
        {
            "valid_object_query": False,
            "foreground_gain": float("nan"),
            "far_background_reduction": float("nan"),
            "routed_foreground_fraction": float("nan"),
            "routed_far_background_fraction": float("nan"),
        }
    )
    summary = _condition_summary([base, invalid])
    assert summary["rows"] == 2
    assert summary["valid_object_query_rows"] == 1
    assert summary["finite_rows"] == 1
    assert summary["foreground_gain_mean"] == pytest.approx(0.04)
    assert summary["far_background_reduction_mean"] == pytest.approx(0.03)
    assert summary["positive_foreground_gain_fraction"] == 1.0


def test_independent_sparse_scatter_reconstructs_module_dense_attention() -> None:
    torch.manual_seed(5)
    module = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=4,
        topk=4,
        local_context_kernel_size=5,
    ).eval()
    inputs = torch.randn(2, 21, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    with torch.inference_mode():
        module(
            inputs,
            return_attention=True,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=indices,
            collect_trace=True,
        )
    trace = module.trace()
    replay = _independent_sparse_scatter(
        trace["native_sparse_attention"],
        trace["route_indices"],
        module.region_token_indices,
        prefix_count=5,
        patch_count=16,
    )
    torch.testing.assert_close(replay, trace["dense_attention"][:, :, 5:])
    integrity = _attention_integrity(module)
    assert integrity["scatter_replay_maximum_absolute_error"] == 0.0
    assert integrity["dense_row_sum_maximum_error"] <= 1e-6


def test_bra_preflight_equation_and_dense_oracles_pass() -> None:
    if not (OFFICIAL_ROOT / "ops" / "bra_nchw.py").is_file():
        pytest.skip("Locked official BiFormer source is absent.")
    assert _official_equation_replay(OFFICIAL_ROOT)["passed"]
    assert _dense_mhsa_parity()["passed"]


def test_bra_wrapper_locks_preflight_review_and_pair_modes() -> None:
    script = Path("scripts/run_trkh_bi_level_routing_attention_a1.ps1").read_text(
        encoding="utf-8"
    )
    assert "[switch]$PreflightOnly" in script
    assert "[switch]$FinalizeVisualReview" in script
    assert "[switch]$RunPair" in script
    assert "formal_pair_permission" in script
    assert "-BiLevelRoutingAttentionTopK 16" in script
    assert "-BiLevelRoutingAttentionTopK 4" in script
    assert "SkipFinalTest = $true" in script
    assert "DisableBalancedEpochSampling = $true" in script

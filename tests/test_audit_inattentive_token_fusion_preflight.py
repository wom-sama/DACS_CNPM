from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trkh.tools.audit_inattentive_token_fusion_preflight import (
    CONDITIONS,
    EXPECTED_HOLDOUT_ROWS,
    _bbox_patch_overlap,
    _behavior_gate_checks,
    _fusion_config,
    _fusion_lineage,
    _official_equation_replay,
    _resolve_tiny_edge_cohort,
    _set_jaccard,
)


def test_fusion_preflight_configs_lock_only_the_feature_flag() -> None:
    source = {
        "model_type": "vit_registers",
        "pretrained": True,
        "early_token_mask_keep_rate": 0.5,
        "token_pruning": True,
        "token_prune_layers": "2,5",
        "token_keep_rates": "0.85,0.65",
    }
    control = _fusion_config(source, enabled=False)
    candidate = _fusion_config(source, enabled=True)
    differing = {
        key
        for key in set(control).union(candidate)
        if control.get(key) != candidate.get(key)
    }
    assert differing == {"inattentive_token_fusion"}
    assert not control["pretrained"]
    assert control["early_token_mask_keep_rate"] == 1.0


def test_official_equation_oracle_replays_both_fusion_stages() -> None:
    summary = _official_equation_replay()
    assert summary["passed"]
    assert summary["maximum_absolute_error"] == 0.0
    assert summary["first_dropped_partition_valid"]
    assert summary["later_dropped_partition_valid"]


def test_bbox_patch_overlap_uses_fractional_area_not_only_centers() -> None:
    full = _bbox_patch_overlap(torch.tensor([0.5, 0.5, 1.0, 1.0]))
    assert full.shape == (256,)
    torch.testing.assert_close(full, torch.ones_like(full))

    one_patch = _bbox_patch_overlap(
        torch.tensor([1.0 / 32.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 16.0])
    )
    assert one_patch[0] == pytest.approx(1.0)
    assert int(torch.count_nonzero(one_patch)) == 1


def test_second_stage_lineage_refolds_previous_context_coefficients() -> None:
    stage1_kept = torch.arange(2, 256).view(1, -1)
    stage2_kept = torch.arange(3, 256).view(1, -1)
    pruning = [
        {
            "kept_indices": stage1_kept,
            "dropped_indices": torch.tensor([[0, 1]]),
            "fusion_weights": torch.tensor([[0.2, 0.3]]),
            "context_attention_mass": torch.tensor([[0.5]]),
            "previous_context_attention_mass": torch.tensor([[0.0]]),
        },
        {
            "kept_indices": stage2_kept,
            "dropped_indices": torch.tensor([[2]]),
            "fusion_weights": torch.tensor([[0.4]]),
            "context_attention_mass": torch.tensor([[0.9]]),
            "previous_context_attention_mass": torch.tensor([[0.5]]),
        },
    ]
    overlap = torch.zeros(256)
    overlap[[0, 2]] = 1.0
    summary = _fusion_lineage(pruning, 0, overlap)
    expected = torch.zeros(256, dtype=torch.float64)
    expected[0] = 0.1
    expected[1] = 0.15
    expected[2] = 0.4
    torch.testing.assert_close(summary["lineage"], expected)
    assert summary["lineage_total"] == pytest.approx(0.65)
    assert summary["object_mass"] == pytest.approx(0.5)
    assert summary["weighted_object_fraction"] == pytest.approx(0.5 / 0.65)
    assert summary["partitions_valid"]
    assert max(row["mass_replay_error"] for row in summary["stages"]) < 1e-7


def test_set_jaccard_is_order_invariant() -> None:
    left = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    right = torch.tensor([[3, 2, 1, 0], [4, 5, 8, 9]])
    observed = _set_jaccard(left, right)
    assert observed.tolist() == pytest.approx([1.0, 1.0 / 3.0])


def test_tiny_edge_cohort_is_metadata_only_and_deterministic() -> None:
    geometries = []
    rows = []
    for index in range(EXPECTED_HOLDOUT_ROWS):
        geometries.append(
            {
                "bbox_area": 0.01 + index / 100_000.0,
                "bbox_edge_gap": 0.01 if index % 10 == 0 else 0.20,
            }
        )
        rows.append(SimpleNamespace(sample_index=10_000 + index))
    first = _resolve_tiny_edge_cohort(geometries, rows)
    second = _resolve_tiny_edge_cohort(geometries, rows)
    assert first == second
    assert first["tiny_count"] >= EXPECTED_HOLDOUT_ROWS // 4
    assert first["edge_count"] == 185
    assert first["cohort_count"] > first["tiny_count"]
    assert len(first["ordered_sample_index_sha256"]) == 64


def _passing_condition_summary(*, clean: bool) -> dict[str, object]:
    return {
        "rows": EXPECTED_HOLDOUT_ROWS,
        "metric_deltas": {
            "macro_f1": 0.0,
            "class1_f1": 0.0,
            "class1_precision": 0.0,
        },
        "class1_tp_retention": 1.0,
        "probability_mae_mean": 0.001,
        "tiny_edge_nonzero_object_mass_fraction": 0.5,
        "weighted_object_gain_mean": 0.02,
        "outside_context_fraction_mean": 0.4,
        "first_prune_exact_fraction": 1.0,
        "second_prune_jaccard_mean": 0.99,
        "partition_valid_fraction": 1.0,
        "stage1_positive_finite_mass_fraction": 1.0,
        "stage2_positive_finite_mass_fraction": 1.0,
        "maximum_mass_replay_error": 0.0,
        "restricted_fp_delta": 0,
        "finite_rows": EXPECTED_HOLDOUT_ROWS,
        "standard_trace_control_maximum_error": 0.0,
        "standard_trace_candidate_maximum_error": 0.0,
        "clean_context_cosine_mean": None if clean else 0.8,
    }


def test_behavior_gate_rejects_a_shift_precision_regression() -> None:
    summaries = {
        name: _passing_condition_summary(clean=name == "clean")
        for name, _, _ in CONDITIONS
    }
    passing = _behavior_gate_checks(
        summaries, row_count=EXPECTED_HOLDOUT_ROWS * len(CONDITIONS)
    )
    assert all(passing.values())
    summaries["lighting_dim"]["metric_deltas"]["class1_precision"] = -0.011
    rejected = _behavior_gate_checks(
        summaries, row_count=EXPECTED_HOLDOUT_ROWS * len(CONDITIONS)
    )
    assert not rejected["lighting_dim_class1_precision_safe"]


def test_fusion_wrapper_locks_preflight_review_and_pair_modes() -> None:
    script = Path("scripts/run_trkh_inattentive_token_fusion_a0.ps1").read_text(
        encoding="utf-8"
    )
    assert "[switch]$PreflightOnly" in script
    assert "[switch]$ReplayEngineeringCorrection" in script
    assert "[switch]$FinalizeVisualReview" in script
    assert "[switch]$RunPair" in script
    assert "ExpectedSummarySha256" in script
    assert "formal_pair_permission" in script
    assert "postflight_replay.runtime_files_unchanged" in script
    assert "-InattentiveTokenFusion $false" in script
    assert "-InattentiveTokenFusion $true" in script
    assert "SkipFinalTest = $true" in script
    assert "DisableBalancedEpochSampling = $true" in script

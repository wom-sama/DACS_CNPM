from __future__ import annotations

import math

import pytest
import torch

from trkh.tools.train_joint_spatial_pcgrad_smoke import (
    AUXILIARY_GRADIENT_RATIO,
    BATCH_SIZE,
    DETECTOR_HEAD_LEARNING_RATE,
    FOCUS_MILESTONE,
    HEAD_WARMUP_BATCHES,
    HEAD_WARMUP_BATCH_SIZE,
    MAX_TRAIN_BATCHES,
    PRIMARY_LEARNING_RATE,
    SAMPLES_PER_CLASS,
    SEED,
    STEP_PARAMETER_RATIO,
    assess_smoke_gate,
    componentwise_spatial_pcgrad,
    parse_args,
    run_smoke,
)


def test_smoke_protocol_defaults_are_locked() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "keeper.pt",
            "--data-yaml",
            "data.yaml",
            "--output-dir",
            "out",
        ]
    )
    assert args.max_train_batches == MAX_TRAIN_BATCHES == 120
    assert args.update_mode == "normalized_macrostep"
    assert not args.allow_rejected_adamw_ablation
    assert args.samples_per_class == SAMPLES_PER_CLASS == 12
    assert args.step_parameter_ratio == STEP_PARAMETER_RATIO == 1e-4
    assert args.batch_size == BATCH_SIZE == 4
    assert args.head_warmup_batches == HEAD_WARMUP_BATCHES == 120
    assert args.head_warmup_batch_size == HEAD_WARMUP_BATCH_SIZE == 8
    assert args.primary_learning_rate == PRIMARY_LEARNING_RATE == 1e-5
    assert args.detector_head_learning_rate == DETECTOR_HEAD_LEARNING_RATE == 2e-4
    assert args.auxiliary_gradient_ratio == AUXILIARY_GRADIENT_RATIO == 0.25
    assert args.seed == SEED == 20260712
    assert FOCUS_MILESTONE == 0.70


def test_rejected_adamw_ablation_requires_explicit_opt_in() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "keeper.pt",
            "--data-yaml",
            "data.yaml",
            "--output-dir",
            "out",
            "--update-mode",
            "adamw_microbatch",
        ]
    )
    with pytest.raises(ValueError, match="rejected ablation"):
        run_smoke(args)

    unlocked = parse_args(
        [
            "--checkpoint",
            "keeper.pt",
            "--data-yaml",
            "data.yaml",
            "--output-dir",
            "out",
            "--update-mode",
            "adamw_microbatch",
            "--allow-rejected-adamw-ablation",
        ]
    )
    assert unlocked.allow_rejected_adamw_ablation


def test_componentwise_spatial_pcgrad_projects_and_normalizes() -> None:
    primary = [torch.tensor([1.0, 0.0])]
    objectness = [torch.tensor([-1.0, 1.0])]
    localization = [torch.tensor([0.0, 1.0])]
    combined, telemetry = componentwise_spatial_pcgrad(
        primary,
        objectness,
        localization,
        auxiliary_ratio=0.25,
    )
    expected = torch.tensor([1.0, 0.125 * (1.0 + 1.0 / math.sqrt(2.0))])
    expected = expected / expected.norm()
    assert torch.allclose(combined[0], expected, atol=1e-6)
    assert math.isclose(float(combined[0].norm()), 1.0, rel_tol=1e-6)
    assert math.isclose(
        telemetry["objectness_retained_norm_ratio"],
        1.0 / math.sqrt(2.0),
        rel_tol=1e-6,
    )
    assert telemetry["localization_retained_norm_ratio"] == 1.0
    assert abs(telemetry["objectness_projected_dot"]) < 1e-7


def _metrics(macro: float, focus_f1: float, focus_recall: float) -> dict:
    return {
        "rows": 2606,
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "recall": 0.80},
            {"f1": focus_f1, "recall": focus_recall},
        ],
    }


def _transitions() -> dict:
    return {
        "changed": 8,
        "corrections": 5,
        "harms": 2,
        "neutral": 1,
        "focus_false_positive_removed": 3,
        "focus_false_positive_created": 1,
        "focus_false_negative_rescued": 2,
        "focus_true_positive_broken": 1,
    }


def test_smoke_gate_accepts_real_candidate_gain() -> None:
    result = assess_smoke_gate(
        keeper_metrics=_metrics(0.884, 0.686, 0.78),
        control_metrics=_metrics(0.884, 0.690, 0.78),
        candidate_metrics=_metrics(0.887, 0.705, 0.79),
        transitions_vs_control=_transitions(),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["probe_permission"]
    assert result["failed_checks"] == []


def test_smoke_gate_rejects_sub_milestone_and_harmful_candidate() -> None:
    transitions = _transitions()
    transitions.update(
        {
            "corrections": 1,
            "harms": 5,
            "focus_false_positive_removed": 1,
            "focus_false_positive_created": 4,
            "focus_false_negative_rescued": 0,
            "focus_true_positive_broken": 3,
        }
    )
    result = assess_smoke_gate(
        keeper_metrics=_metrics(0.884, 0.686, 0.78),
        control_metrics=_metrics(0.884, 0.690, 0.78),
        candidate_metrics=_metrics(0.880, 0.680, 0.74),
        transitions_vs_control=transitions,
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["probe_permission"]
    assert "candidate_focus_reaches_0p70" in result["failed_checks"]
    assert "candidate_corrections_ge_harms_vs_control" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]

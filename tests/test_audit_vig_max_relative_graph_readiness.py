from __future__ import annotations

import hashlib
from pathlib import Path

from trkh.tools.audit_vig_max_relative_graph_readiness import (
    EXPECTED_ADDED_PARAMETERS,
    EXPECTED_GRAPH_LAYERS,
    LOCKED_PROTOCOL_SHA256,
    _candidate_config,
    _expected_state_additions,
    assess_vig_graph_stage_a,
    parse_args,
)


def _comparison(*, good: bool) -> dict[str, object]:
    if good:
        delta = {
            "macro_f1": 0.001,
            "class1_f1": 0.004,
            "class1_precision": 0.008,
            "class1_recall": 0.0,
        }
        transitions = {
            "changed": 5,
            "restricted_focus_fp_reduction": 3,
            "focus_tp_break": 0,
            "focus_fn_rescue": 1,
            "candidate_correction": 4,
            "candidate_harm": 1,
            "new_nonfocus_3_to_2_harms": 0,
            "focus_fp_remove_correct": 3,
            "focus_fp_create": 0,
        }
    else:
        delta = {
            "macro_f1": -0.02,
            "class1_f1": -0.03,
            "class1_precision": -0.04,
            "class1_recall": -0.05,
        }
        transitions = {
            "changed": 1,
            "restricted_focus_fp_reduction": -1,
            "focus_tp_break": 5,
            "focus_fn_rescue": 0,
            "candidate_correction": 0,
            "candidate_harm": 1,
            "new_nonfocus_3_to_2_harms": 3,
            "focus_fp_remove_correct": 0,
            "focus_fp_create": 1,
        }
    return {
        "delta": delta,
        "transitions": transitions,
        "maximum_nonfocus_f1_drop": 0.0 if good else 0.03,
    }


def test_locked_defaults_and_candidate_config() -> None:
    args = parse_args(["--output-dir", "runs/unit_vig_graph"])
    assert args.graph_layers == "2,5"
    assert args.graph_bottleneck_dim == 64
    assert args.graph_k == 9
    assert args.max_train_batches == 30
    assert args.learning_rate == 3e-4
    assert EXPECTED_GRAPH_LAYERS == (2, 5)
    assert EXPECTED_ADDED_PARAMETERS == 99_840
    config = _candidate_config({"depth": 8}, args)
    assert config["dynamic_graph_mixer"] is True
    assert config["dynamic_graph_mixer_layers"] == "2,5"
    assert config["dynamic_graph_mixer_bottleneck_dim"] == 64
    assert config["dynamic_graph_mixer_k"] == 9
    assert config["pretrained"] is False


def test_locked_protocol_hash_matches_precommitted_file() -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "TRKH_5CLASS_VIG_MAX_RELATIVE_GRAPH_READINESS_PROTOCOL_20260715.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == LOCKED_PROTOCOL_SHA256


def test_expected_checkpoint_additions_are_exact() -> None:
    additions = _expected_state_additions((2, 5))
    assert len(additions) == 10
    assert "blocks.1.dynamic_graph_mixer.norm.weight" in additions
    assert "blocks.1.dynamic_graph_mixer.input_projection.weight" in additions
    assert "blocks.4.dynamic_graph_mixer.output_projection.bias" in additions


def test_gate_passes_only_when_structural_and_decision_checks_pass() -> None:
    good = _comparison(good=True)
    lighting = [
        {**_comparison(good=True), "condition": condition}
        for condition in ("lighting_dim", "lighting_bright", "low_contrast")
    ]
    passed = assess_vig_graph_stage_a(
        structural_checks={"mechanism": True},
        unadapted=good,
        adapted=good,
        illumination=lighting,
    )
    assert passed["all_gates_passed"] is True
    assert passed["stage_b_smoke_authorized"] is True
    assert passed["full_train_authorized"] is False

    failed = assess_vig_graph_stage_a(
        structural_checks={"mechanism": False},
        unadapted=_comparison(good=False),
        adapted=_comparison(good=False),
        illumination=[
            {**_comparison(good=False), "condition": condition}
            for condition in ("lighting_dim", "lighting_bright", "low_contrast")
        ],
    )
    assert failed["all_gates_passed"] is False
    assert failed["stage_b_smoke_authorized"] is False
    assert "mechanism" in failed["failed_checks"]


def test_launcher_is_fail_closed_and_train_only() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_vig_graph_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "trkh.tools.audit_vig_max_relative_graph_readiness" in script
    assert "--official-vig-root" in script
    assert "validation_used = $false" in script
    assert "test_used = $false" in script
    assert "$LASTEXITCODE -ne 0" in script

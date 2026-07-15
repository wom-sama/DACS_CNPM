from __future__ import annotations

import hashlib
from pathlib import Path

from trkh.tools.audit_deep_class_prompt_readiness import (
    EXPECTED_ADDED_PARAMETERS,
    EXPECTED_BLOCKS,
    EXPECTED_PROMPT_STATE_KEYS,
    LOCKED_PROTOCOL_SHA256,
    _candidate_config,
    assess_deep_class_prompt_stage_a,
    parse_args,
)


def _comparison(*, good: bool) -> dict[str, object]:
    if good:
        delta = {
            "macro_f1": 0.001,
            "class1_f1": 0.006,
            "class1_precision": 0.007,
            "class1_recall": 0.0,
        }
        transitions = {
            "restricted_focus_fp_corrected_to_target": 2,
            "focus_fn_rescue": 2,
            "focus_tp_break": 1,
            "candidate_correction": 4,
            "candidate_harm": 1,
        }
        nonfocus_drop = 0.001
    else:
        delta = {
            "macro_f1": -0.01,
            "class1_f1": -0.02,
            "class1_precision": -0.03,
            "class1_recall": -0.04,
        }
        transitions = {
            "restricted_focus_fp_corrected_to_target": 0,
            "focus_fn_rescue": 0,
            "focus_tp_break": 3,
            "candidate_correction": 0,
            "candidate_harm": 2,
        }
        nonfocus_drop = 0.02
    return {
        "delta": delta,
        "transitions": transitions,
        "maximum_nonfocus_f1_drop": nonfocus_drop,
    }


def test_locked_defaults_and_candidate_config() -> None:
    args = parse_args(["--output-dir", "runs/unit_deep_class_prompt"])
    assert args.batch_size == 32
    assert args.max_train_batches == 60
    assert args.learning_rate == 0.005
    assert args.momentum == 0.9
    assert args.weight_decay == 0.001
    assert args.logit_scale == 0.10
    assert args.prompt_init_seed == 20260715
    assert EXPECTED_BLOCKS == 8
    assert EXPECTED_ADDED_PARAMETERS == 11_009
    assert len(EXPECTED_PROMPT_STATE_KEYS) == 5
    config = _candidate_config({"depth": 8}, args)
    assert config["deep_class_prompt"] is True
    assert config["deep_class_prompt_logit_scale"] == 0.10
    assert config["deep_class_prompt_init_seed"] == 20260715
    assert config["pretrained"] is False


def test_locked_protocol_hash_matches_precommitted_file() -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "TRKH_5CLASS_DEEP_CLASS_PROMPT_READINESS_PROTOCOL_20260715.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == LOCKED_PROTOCOL_SHA256


def test_gate_requires_every_structural_decision_and_lighting_check() -> None:
    adapted = _comparison(good=True)
    lighting = [
        {**_comparison(good=True), "condition": condition}
        for condition in ("lighting_dim", "lighting_bright", "low_contrast")
    ]
    passed = assess_deep_class_prompt_stage_a(
        structural_checks={"mechanism": True},
        adapted=adapted,
        illumination=lighting,
    )
    assert passed["all_gates_passed"] is True
    assert passed["stage_b_smoke_authorized"] is True
    assert passed["full_train_authorized"] is False

    failed = assess_deep_class_prompt_stage_a(
        structural_checks={"mechanism": False},
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
        / "run_trkh_deep_class_prompt_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "trkh.tools.audit_deep_class_prompt_readiness" in script
    assert "--prompt-cam-root" in script
    assert "--mctformer-root" in script
    assert "validation_used = $false" in script
    assert "test_used = $false" in script
    assert "$LASTEXITCODE -ne 0" in script

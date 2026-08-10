from __future__ import annotations

from pathlib import Path

import pytest

from trkh.tools.audit_xca_dual_axis_readiness import (
    EXPECTED_XCA_LAYERS,
    STATIC_EXPORT_BATCH_SIZE,
    _candidate_config,
    _expected_missing_keys,
    assess_xca_stage_a,
    parse_args,
)


def _comparison_payload() -> dict[str, object]:
    return {
        "delta": {
            "macro_f1": 0.003,
            "class1_f1": 0.004,
            "class1_precision": 0.006,
            "class1_recall": -0.005,
        },
        "transitions": {
            "changed": 5,
            "focus_tp_break": 1,
            "focus_fn_rescue": 1,
            "candidate_correction": 4,
            "candidate_harm": 1,
            "restricted_focus_fp_reduction": 2,
            "new_nonfocus_3_to_2_harms": 1,
        },
        "maximum_nonfocus_f1_drop": 0.010,
    }


def test_stage_a_assessment_requires_every_locked_gate() -> None:
    passing = assess_xca_stage_a(
        structural_checks={"structure": True, "resource": True},
        unadapted=_comparison_payload(),
        adapted=_comparison_payload(),
    )
    assert passing["all_gates_passed"] is True
    assert passing["stage_b_smoke_authorized"] is True
    assert passing["full_train_authorized"] is False

    failing_payload = _comparison_payload()
    failing_payload["delta"]["class1_precision"] = 0.0049
    failing = assess_xca_stage_a(
        structural_checks={"structure": True},
        unadapted=_comparison_payload(),
        adapted=failing_payload,
    )
    assert failing["all_gates_passed"] is False
    assert "class1_precision_delta_gte_0p005" in failing["failed_checks"]


def test_unadapted_intervention_guard_is_fail_closed() -> None:
    unadapted = _comparison_payload()
    unadapted["delta"]["class1_recall"] = -0.021
    unadapted["transitions"]["focus_tp_break"] = 5
    unadapted["transitions"]["focus_fn_rescue"] = 1
    result = assess_xca_stage_a(
        structural_checks={"structure": True},
        unadapted=unadapted,
        adapted=_comparison_payload(),
    )
    assert result["all_gates_passed"] is False
    assert "class1_recall_delta_gte_minus_0p020" in result["failed_checks"]
    assert "net_tp_breaks_at_most_3" in result["failed_checks"]


def test_candidate_config_and_missing_keys_are_exact() -> None:
    args = parse_args(["--output-dir", "unused"])
    config = _candidate_config(
        {
            "model_type": "vit_registers",
            "token_pruning": True,
            "visual_contrast_attention": False,
        },
        args,
    )
    assert config["cross_covariance_attention"] is True
    assert config["cross_covariance_attention_layers"] == "2,5"
    assert config["cross_covariance_attention_residual_scale"] == pytest.approx(0.10)
    assert config["token_pruning"] is True
    assert _expected_missing_keys(EXPECTED_XCA_LAYERS) == {
        "blocks.1.cross_covariance_attention.temperature",
        "blocks.1.cross_covariance_attention.norm.weight",
        "blocks.1.cross_covariance_attention.norm.bias",
        "blocks.4.cross_covariance_attention.temperature",
        "blocks.4.cross_covariance_attention.norm.weight",
        "blocks.4.cross_covariance_attention.norm.bias",
    }


def test_stage_a_defaults_and_powershell_wrapper_are_locked() -> None:
    args = parse_args(["--output-dir", "unused"])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.max_train_batches == 20
    assert args.learning_rate == pytest.approx(8e-5)
    assert args.weight_decay == pytest.approx(0.05)
    assert args.xca_layers == "2,5"
    assert args.residual_scale == pytest.approx(0.10)
    assert STATIC_EXPORT_BATCH_SIZE == 2

    audit_source = (
        Path(__file__).resolve().parents[1]
        / "trkh"
        / "tools"
        / "audit_xca_dual_axis_readiness.py"
    ).read_text(encoding="utf-8")
    assert "dynamic_axes={" not in audit_source

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_xca_dual_axis_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "trkh.tools.audit_xca_dual_axis_readiness" in script
    assert '"--checkpoint", $Checkpoint' in script
    assert '"--data", $DataYaml' in script
    assert '"--output-dir", $OutputDir' in script
    assert "validation_used = $false" in script
    assert "test_used = $false" in script
    assert "2>&1" not in script

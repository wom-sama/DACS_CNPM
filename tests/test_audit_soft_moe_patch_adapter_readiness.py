from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from trkh.tools.audit_soft_moe_patch_adapter_readiness import (
    CANDIDATE_TRAINABLE,
    CONTROL_TRAINABLE,
    EXPECTED_ADDED_PARAMETERS,
    EXPECTED_LAYERS,
    EXPECTED_STATE_KEYS,
    LOCKED_PROTOCOL_SHA256,
    STATIC_EXPORT_BATCH_SIZE,
    _candidate_config,
    _metadata_for_batch,
    assess_soft_moe_stage_a,
    parse_args,
)
from trkh.tools.correct_soft_moe_stage_a_equation_assessment import (
    EQUATION_CHECK,
    build_correction,
)


def _comparison(*, passing: bool = True) -> dict[str, object]:
    if passing:
        return {
            "delta": {
                "macro_f1": 0.001,
                "class1_f1": 0.006,
                "class1_precision": 0.007,
                "class1_recall": 0.0,
            },
            "transitions": {
                "restricted_focus_fp_corrected_to_target": 2,
                "focus_fn_rescue": 2,
                "focus_tp_break": 1,
                "candidate_correction": 4,
                "candidate_harm": 1,
            },
            "maximum_nonfocus_f1_drop": 0.001,
        }
    return {
        "delta": {
            "macro_f1": -0.01,
            "class1_f1": -0.02,
            "class1_precision": -0.03,
            "class1_recall": -0.04,
        },
        "transitions": {
            "restricted_focus_fp_corrected_to_target": 0,
            "focus_fn_rescue": 0,
            "focus_tp_break": 3,
            "candidate_correction": 0,
            "candidate_harm": 2,
        },
        "maximum_nonfocus_f1_drop": 0.02,
    }


def _lighting(*, passing: bool = True) -> list[dict[str, object]]:
    return [
        {**_comparison(passing=passing), "condition": condition}
        for condition in ("lighting_dim", "lighting_bright", "low_contrast")
    ]


def test_locked_defaults_config_and_state_contract() -> None:
    args = parse_args(["--output-dir", "runs/unit_soft_moe"])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 5
    assert args.max_train_batches == 60
    assert args.learning_rate == pytest.approx(0.005)
    assert args.momentum == pytest.approx(0.9)
    assert args.weight_decay == pytest.approx(0.001)
    assert args.layers == "2,5"
    assert args.hidden_dim == 64
    assert args.num_experts == 4
    assert args.residual_scale == pytest.approx(0.10)
    assert args.router_scale_init == pytest.approx(10.0)
    assert args.init_seed == 20260715
    assert args.max_runtime_ratio == pytest.approx(1.25)
    assert args.max_peak_vram_gib == pytest.approx(3.25)
    assert STATIC_EXPORT_BATCH_SIZE == 1
    assert EXPECTED_LAYERS == (2, 5)
    assert EXPECTED_ADDED_PARAMETERS == 266_754
    assert len(EXPECTED_STATE_KEYS) == 36
    assert CANDIDATE_TRAINABLE == EXPECTED_STATE_KEYS
    assert len(CONTROL_TRAINABLE) == 16

    config = _candidate_config({"depth": 8, "token_pruning": True}, args)
    assert config["soft_moe_patch_adapter"] is True
    assert config["soft_moe_patch_adapter_layers"] == "2,5"
    assert config["soft_moe_hidden_dim"] == 64
    assert config["soft_moe_num_experts"] == 4
    assert config["pretrained"] is False
    assert config["token_pruning"] is True


def test_locked_protocol_hash_matches_precommitted_file() -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "TRKH_5CLASS_SOFT_MOE_PATCH_ADAPTER_READINESS_PROTOCOL_20260715.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == LOCKED_PROTOCOL_SHA256


def test_fp32_gradient_metadata_is_sliced_to_the_image_batch() -> None:
    metadata = {
        "bbox": torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8),
        "image_mask": torch.ones(32, 4, 4, dtype=torch.bool),
        "sample_index": torch.arange(32),
        "path": [f"row_{index}" for index in range(32)],
    }
    sliced = _metadata_for_batch(metadata, device=torch.device("cpu"), count=5)
    assert sliced["bbox"].shape == (5, 8)
    assert sliced["image_mask"].shape == (5, 4, 4)
    assert sliced["sample_index"].tolist() == [0, 1, 2, 3, 4]
    assert sliced["path"] == metadata["path"]

    with pytest.raises(ValueError, match="positive"):
        _metadata_for_batch(metadata, device=torch.device("cpu"), count=0)


def test_gate_requires_both_references_and_every_lighting_condition() -> None:
    passed = assess_soft_moe_stage_a(
        structural_checks={"mechanism": True, "resource": True},
        candidate_vs_raw=_comparison(),
        candidate_vs_control=_comparison(),
        illumination_vs_raw=_lighting(),
        illumination_vs_control=_lighting(),
    )
    assert passed["all_gates_passed"] is True
    assert passed["stage_b_smoke_authorized"] is True
    assert passed["full_train_authorized"] is False

    failed_control = _comparison()
    failed_control["delta"]["class1_precision"] = 0.0049
    failed = assess_soft_moe_stage_a(
        structural_checks={"mechanism": True},
        candidate_vs_raw=_comparison(),
        candidate_vs_control=failed_control,
        illumination_vs_raw=_lighting(),
        illumination_vs_control=_lighting(),
    )
    assert failed["all_gates_passed"] is False
    assert failed["stage_b_smoke_authorized"] is False
    assert (
        "candidate_vs_control_class1_precision_delta_gte_0p005"
        in failed["failed_checks"]
    )


def test_auditor_and_launcher_are_train_only_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    audit_source = (
        root / "trkh" / "tools" / "audit_soft_moe_patch_adapter_readiness.py"
    ).read_text(encoding="utf-8")
    assert 'split="val"' not in audit_source
    assert 'split="test"' not in audit_source

    script = (
        root / "scripts" / "run_trkh_soft_moe_patch_adapter_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "trkh.tools.audit_soft_moe_patch_adapter_readiness" in script
    assert '"--checkpoint", $Checkpoint' in script
    assert '"--vmoe-root", $VmoeRoot' in script
    assert '"--soft-moe-paper", $SoftMoePaper' in script
    assert '"--sweet-spot-paper", $SweetSpotPaper' in script
    assert "validation_used = $false" in script
    assert "test_used = $false" in script
    assert "$LASTEXITCODE -ne 0" in script
    assert "2>&1" not in script


def test_equation_correction_cannot_change_decision_failures() -> None:
    summary = {
        "validation_data_used": False,
        "test_data_used": False,
        "gate": {
            "structural_checks": {EQUATION_CHECK: False, "mechanism": True},
            "decision_checks": {"class1_precision": False},
            "illumination_checks": {"lighting": True},
            "failed_checks": [EQUATION_CHECK, "class1_precision"],
            "all_gates_passed": False,
            "stage_b_smoke_authorized": False,
            "full_train_authorized": False,
        },
        "adapted_candidate_vs_raw": {"delta": {"class1_precision": 0.0}},
        "adapted_candidate_vs_control": {"delta": {"class1_precision": 0.0}},
        "illumination_candidate_vs_raw": [],
        "illumination_candidate_vs_control": [],
    }
    equation = {
        precision: {
            "dispatch_sum_max_error": 1e-7,
            "combine_sum_max_error": 1e-7,
            "slot_max_error": 0.0,
            "residual_max_error": 0.0,
            "update_max_error": 0.0,
            "prefix_bit_exact": True,
            "finite": True,
        }
        for precision in ("fp32", "bf16")
    }
    correction = build_correction(
        summary,
        equation=equation,
        source_summary_sha256="source",
        original_auditor_sha256="original",
        corrected_auditor_sha256="corrected",
    )
    assert correction["corrected_gate"]["structural_checks"][EQUATION_CHECK]
    assert correction["corrected_gate"]["failed_checks"] == ["class1_precision"]
    assert correction["corrected_gate"]["stage_b_smoke_authorized"] is False
    assert correction["decision_evidence_changed"] is False
    assert correction["outcome_changed"] is False

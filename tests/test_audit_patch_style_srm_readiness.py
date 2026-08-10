from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trkh.models.patch_style_recalibration import PatchStyleRecalibration

from trkh.tools.audit_patch_style_srm_readiness import (
    EXPECTED_SRM_LAYERS,
    STATIC_EXPORT_BATCH_SIZE,
    _candidate_config,
    _expected_state_additions,
    _onnx_compare,
    _SRMEquationExportWrapper,
    assess_patch_style_srm_stage_a,
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
            "focus_fp_remove_correct": 3,
            "focus_fp_create": 1,
            "candidate_correction": 4,
            "candidate_harm": 1,
            "restricted_focus_fp_reduction": 2,
            "new_nonfocus_3_to_2_harms": 1,
        },
        "maximum_nonfocus_f1_drop": 0.010,
    }


def _illumination_payload() -> list[dict[str, object]]:
    rows = []
    for condition in ("lighting_dim", "lighting_bright", "low_contrast"):
        row = _comparison_payload()
        row["condition"] = condition
        rows.append(row)
    return rows


def test_stage_a_assessment_requires_every_locked_gate() -> None:
    passing = assess_patch_style_srm_stage_a(
        structural_checks={"structure": True, "resource": True},
        unadapted=_comparison_payload(),
        adapted=_comparison_payload(),
        illumination=_illumination_payload(),
    )
    assert passing["all_gates_passed"] is True
    assert passing["stage_b_smoke_authorized"] is True
    assert passing["full_train_authorized"] is False

    failing_payload = _comparison_payload()
    failing_payload["delta"]["class1_precision"] = 0.0049
    failing = assess_patch_style_srm_stage_a(
        structural_checks={"structure": True},
        unadapted=_comparison_payload(),
        adapted=failing_payload,
        illumination=_illumination_payload(),
    )
    assert failing["all_gates_passed"] is False
    assert "class1_precision_delta_gte_0p005" in failing["failed_checks"]


def test_illumination_assessment_is_fail_closed() -> None:
    illumination = _illumination_payload()
    illumination[0]["delta"]["class1_precision"] = -0.011
    illumination[1]["delta"]["class1_precision"] = -0.012
    illumination[2]["transitions"]["focus_fp_create"] = 20
    result = assess_patch_style_srm_stage_a(
        structural_checks={"structure": True},
        unadapted=_comparison_payload(),
        adapted=_comparison_payload(),
        illumination=illumination,
    )
    assert result["all_gates_passed"] is False
    assert "lighting_precision_nonnegative_at_least_2_of_3" in result["failed_checks"]
    assert "worst_lighting_precision_delta_gte_minus_0p010" in result["failed_checks"]
    assert "aggregate_focus_fp_removals_gt_creations" in result["failed_checks"]


def test_candidate_config_and_state_additions_are_exact() -> None:
    args = parse_args(["--output-dir", "unused"])
    config = _candidate_config(
        {
            "model_type": "vit_registers",
            "token_pruning": True,
            "locally_enhanced_ffn": False,
        },
        args,
    )
    assert config["patch_style_recalibration"] is True
    assert config["patch_style_recalibration_layers"] == "2,5"
    assert config["token_pruning"] is True
    additions = _expected_state_additions(EXPECTED_SRM_LAYERS)
    assert len(additions) == 12
    assert "blocks.1.mlp.style_recalibration.cfc" in additions
    assert "blocks.4.mlp.style_recalibration.bn.num_batches_tracked" in additions


def test_stage_a_defaults_and_powershell_wrapper_are_locked() -> None:
    args = parse_args(["--output-dir", "unused"])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 5
    assert args.max_train_batches == 30
    assert args.learning_rate == pytest.approx(3e-4)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.srm_layers == "2,5"
    assert args.max_runtime_ratio == pytest.approx(1.15)
    assert STATIC_EXPORT_BATCH_SIZE == 1

    audit_source = (
        Path(__file__).resolve().parents[1]
        / "trkh"
        / "tools"
        / "audit_patch_style_srm_readiness.py"
    ).read_text(encoding="utf-8")
    assert "dynamic_axes={" not in audit_source
    assert 'split="val"' not in audit_source
    assert 'split="test"' not in audit_source

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_patch_style_srm_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "trkh.tools.audit_patch_style_srm_readiness" in script
    assert '"--checkpoint", $Checkpoint' in script
    assert '"--data", $DataYaml' in script
    assert '"--output-dir", $OutputDir' in script
    assert "validation_used = $false" in script
    assert "test_used = $false" in script
    assert "2>&1" not in script


def test_static_batch_one_equation_export_keeps_batch_norm_in_eval(tmp_path) -> None:
    module = PatchStyleRecalibration(16).train()
    wrapper = _SRMEquationExportWrapper(module, prefix_count=2)
    assert wrapper.training is False
    assert wrapper.module.bn.training is False
    result = _onnx_compare(
        wrapper=wrapper,
        inputs=(torch.randn(1, 7, 16),),
        input_names=("hidden",),
        path=tmp_path / "srm.onnx",
    )
    assert result["succeeded"] is True
    assert result["batch_contract"] == "static_batch_1"
    assert result["maximum_absolute_error"] <= 1e-6
    assert wrapper.module.bn.training is False

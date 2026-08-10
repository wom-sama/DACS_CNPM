import json
from pathlib import Path

from trkh.models.model import HybridConvStem
from trkh.tools.audit_validity_partial_conv_stem_readiness import (
    MAX_EQUATION_ERROR,
    _config_differences,
    _model_configs,
    _module_type_differences,
    _official_equation_oracle,
    assess_readiness,
    parse_args,
    replay_summary,
)


def test_locked_cli_defaults() -> None:
    args = parse_args(["--output-dir", "runs/unit_validity_partial"])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.num_workers == 4
    assert args.benchmark_repeats == 3
    assert args.seed == 42
    assert args.device == "cuda"


def test_model_configs_change_only_stem_convolution() -> None:
    source = {
        "model_type": "vit_registers",
        "stem_architecture": "conv_pool",
        "pretrained": False,
    }
    control, candidate = _model_configs(source)
    assert _config_differences(control, candidate) == ["stem_convolution"]
    assert control["stem_convolution"] == "standard"
    assert candidate["stem_convolution"] == "validity_partial"
    assert candidate["stem_pooling_mode"] == "max"


def test_module_type_diff_is_exactly_three_convolutions() -> None:
    control = HybridConvStem(3, 4, 16, convolution="standard")
    candidate = HybridConvStem(3, 4, 16, convolution="validity_partial")
    differences = _module_type_differences(control, candidate)
    assert differences == [
        {
            "name": f"blocks.{index}.block.conv",
            "control": "Conv2d",
            "candidate": "ValidityPartialConv2d",
        }
        for index in range(3)
    ]


def test_official_equation_oracle_matches_locked_source() -> None:
    source = Path(
        r"D:\DataAI\external_sources\official\NVIDIA_partialconv\models\partialconv2d.py"
    )
    result = _official_equation_oracle(source)
    assert result["interior_output_max_abs_error"] <= MAX_EQUATION_ERROR
    assert result["interior_mask_exact"] is True
    assert result["deliberate_tensor_border_delta"] > 1e-6
    assert result["finite"] is True


def test_gate_requires_every_check_and_resource_limit() -> None:
    passed = assess_readiness(
        checks={"schema": True, "oracle": True},
        peak_vram_gib=2.0,
        runtime_ratio=1.05,
    )
    assert passed["smoke_permission"] is True
    assert passed["full_train_permission"] is False

    failed = assess_readiness(
        checks={"schema": True, "oracle": False},
        peak_vram_gib=8.0,
        runtime_ratio=1.36,
    )
    assert failed["smoke_permission"] is False
    assert set(failed["failed_checks"]) == {
        "oracle",
        "peak_vram_within_budget",
        "runtime_ratio_within_budget",
    }


def test_summary_replay_detects_gate_input_drift(tmp_path: Path) -> None:
    gate_inputs = tmp_path / "gate_inputs.json"
    payload = {
        "checks": {"schema": True, "oracle": True},
        "peak_vram_gib": 2.0,
        "runtime_ratio": 1.05,
    }
    gate_inputs.write_text(json.dumps(payload), encoding="utf-8")
    expected = assess_readiness(
        checks=payload["checks"],
        peak_vram_gib=payload["peak_vram_gib"],
        runtime_ratio=payload["runtime_ratio"],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "gate_inputs_path": str(gate_inputs.resolve()),
                "gate": expected,
            }
        ),
        encoding="utf-8",
    )
    assert replay_summary(summary)["exact"] is True

    payload["checks"]["oracle"] = False
    gate_inputs.write_text(json.dumps(payload), encoding="utf-8")
    assert replay_summary(summary)["exact"] is False

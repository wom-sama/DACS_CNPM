from pathlib import Path

from trkh.models.model import InstanceBatchNorm2d
from trkh.tools.audit_ibn_a_shallow_stem_readiness import (
    MAX_EQUATION_ERROR,
    MAX_ONNX_ERROR,
    MAX_STANDARDIZED_MEAN,
    MAX_STANDARDIZED_VARIANCE_ERROR,
    MAX_STYLE_ERROR_RATIO,
    _config_differences,
    _export_norm,
    _model_configs,
    _oracle,
    assess_readiness,
    parse_args,
)


def test_locked_cli_defaults() -> None:
    args = parse_args(["--output-dir", "runs/unit_ibn_a"])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.num_workers == 4
    assert args.benchmark_repeats == 3
    assert args.seed == 42
    assert args.device == "cuda"


def test_model_configs_change_only_stem_normalization() -> None:
    source = {
        "model_type": "vit_registers",
        "stem_architecture": "conv_pool",
        "pretrained": False,
    }
    control, candidate = _model_configs(source)
    assert _config_differences(control, candidate) == ["stem_normalization"]
    assert control["stem_normalization"] == "batch"
    assert candidate["stem_normalization"] == "ibn_a_first"


def test_numerical_oracle_passes_locked_thresholds() -> None:
    result = _oracle()
    assert result["equation_max_abs_error"] <= MAX_EQUATION_ERROR
    assert result["standardized_mean_max_abs"] <= MAX_STANDARDIZED_MEAN
    assert (
        result["standardized_variance_max_error"]
        <= MAX_STANDARDIZED_VARIANCE_ERROR
    )
    assert result["style_error_ratio"] <= MAX_STYLE_ERROR_RATIO
    assert result["batch_half_bit_identical"] is True
    assert result["finite"] is True


def test_onnx_export_uses_no_custom_operator(tmp_path: Path) -> None:
    result = _export_norm(InstanceBatchNorm2d(32).eval(), tmp_path)
    assert result["custom_operators"] == []
    assert result["maximum_abs_error"] <= MAX_ONNX_ERROR
    assert result["passed"] is True
    assert Path(result["path"]).is_file()


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
        runtime_ratio=1.31,
    )
    assert failed["smoke_permission"] is False
    assert set(failed["failed_checks"]) == {
        "oracle",
        "peak_vram_within_budget",
        "runtime_ratio_within_budget",
    }

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest
import timm
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from trkh.inference.surfacefold_deployment import (
    benchmark_onnx_vs_ort_cpu,
    check_ort_mobile_usability,
    compare_pytorch_ort,
    convert_fixed_ort_arm,
    export_fixed_opset17_onnx,
    onnx_topology_fingerprint,
)
from trkh.models.swiftformer_surfacefold_b17 import (
    SURFACEFOLD_MEAN_CONTROL_B17_MODE,
    SURFACEFOLD_OFF_B17_MODE,
    SURFACEFOLD_SPATIAL_B17_MODE,
    SwiftFormerSurfaceFoldB17,
)


class TinyClassifier(nn.Module):
    def __init__(self, *, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=stride, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, 5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.activation(self.conv(images))).flatten(1)
        return self.head(features)


def _sample(seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        (1, 3, 16, 16), dtype=np.float32
    )


def _export(tmp_path: Path, name: str = "tiny.onnx") -> tuple[TinyClassifier, Path]:
    torch.manual_seed(11)
    model = TinyClassifier().eval()
    path = tmp_path / name
    result = export_fixed_opset17_onnx(
        model,
        torch.from_numpy(_sample()),
        path,
    )
    assert result["opset"] == 17
    assert result["input_shape"] == [1, 3, 16, 16]
    assert result["output_shape"] == [1, 5]
    assert len(result["sha256"]) == 64
    return model, path


def test_fixed_export_and_pytorch_ort_parity(tmp_path: Path) -> None:
    model, path = _export(tmp_path)
    result = compare_pytorch_ort(
        model,
        path,
        [_sample(1), _sample(2)],
        providers=("CPUExecutionProvider",),
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert result["passed"] is True
    assert result["samples"] == 2
    assert result["argmax_mismatches"] == 0
    assert result["max_abs_error"] <= 1.0e-5


def test_export_rejects_non_batch_one_and_existing_output(tmp_path: Path) -> None:
    model = TinyClassifier().eval()
    path = tmp_path / "tiny.onnx"
    with pytest.raises(ValueError, match="batch size exactly one"):
        export_fixed_opset17_onnx(model, torch.zeros(2, 3, 16, 16), path)

    export_fixed_opset17_onnx(model, torch.zeros(1, 3, 16, 16), path)
    with pytest.raises(FileExistsError):
        export_fixed_opset17_onnx(model, torch.zeros(1, 3, 16, 16), path)


def test_topology_fingerprint_ignores_weights_but_detects_structure(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1)
    first = TinyClassifier().eval()
    torch.manual_seed(2)
    second = TinyClassifier().eval()
    changed = TinyClassifier(stride=2).eval()
    first_path = tmp_path / "first.onnx"
    second_path = tmp_path / "second.onnx"
    changed_path = tmp_path / "changed.onnx"
    sample = torch.from_numpy(_sample())
    export_fixed_opset17_onnx(first, sample, first_path)
    export_fixed_opset17_onnx(second, sample, second_path)
    export_fixed_opset17_onnx(changed, sample, changed_path)

    first_hash = onnx_topology_fingerprint(first_path)["sha256"]
    second_hash = onnx_topology_fingerprint(second_path)["sha256"]
    changed_hash = onnx_topology_fingerprint(changed_path)["sha256"]
    assert first_hash == second_hash
    assert changed_hash != first_hash


def _nested_if_model(path: Path, *, then_op: str) -> None:
    condition = numpy_helper.from_array(np.asarray(True, dtype=np.bool_), name="cond")
    value = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="value")
    then_output = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1])
    else_output = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [1])
    then_graph = helper.make_graph(
        [helper.make_node(then_op, ["value"], ["then_out"], name="then_node")],
        "then_graph",
        [],
        [then_output],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["value"], ["else_out"], name="else_node")],
        "else_graph",
        [],
        [else_output],
    )
    output = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["cond"],
                ["result"],
                name="if_node",
                then_branch=then_graph,
                else_branch=else_graph,
            )
        ],
        "nested",
        [],
        [output],
        [condition, value],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_topology_fingerprint_recurses_into_subgraphs(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity_if.onnx"
    neg_path = tmp_path / "neg_if.onnx"
    _nested_if_model(identity_path, then_op="Identity")
    _nested_if_model(neg_path, then_op="Neg")

    assert (
        onnx_topology_fingerprint(identity_path)["sha256"]
        != onnx_topology_fingerprint(neg_path)["sha256"]
    )


def test_topology_fingerprint_includes_tensor_attributes(tmp_path: Path) -> None:
    paths = [tmp_path / "constant_1.onnx", tmp_path / "constant_2.onnx"]
    for path, value in zip(paths, (1.0, 2.0)):
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        tensor = helper.make_tensor("constant", TensorProto.FLOAT, [1], [value])
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["output"], value=tensor)],
            "constant_graph",
            [],
            [output],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=10,
        )
        onnx.checker.check_model(model)
        onnx.save(model, path)

    assert (
        onnx_topology_fingerprint(paths[0])["sha256"]
        != onnx_topology_fingerprint(paths[1])["sha256"]
    )


def test_topology_rejects_duplicate_default_opset_imports(tmp_path: Path) -> None:
    _, path = _export(tmp_path)
    model = onnx.load(path)
    model.opset_import.append(helper.make_opsetid("", 16))
    duplicate_path = tmp_path / "duplicate_opset.onnx"
    onnx.save(model, duplicate_path)

    with pytest.raises(ValueError, match="only ai.onnx opset 17"):
        onnx_topology_fingerprint(duplicate_path)


def test_fixed_arm_ort_conversion_and_reload_parity(tmp_path: Path) -> None:
    _, onnx_path = _export(tmp_path)
    output_dir = tmp_path / "arm_package"
    result = convert_fixed_ort_arm(
        onnx_path,
        output_dir,
        [_sample(3), _sample(4)],
        intra_op_threads=1,
        inter_op_threads=1,
    )

    assert result["passed"] is True
    assert result["optimization_style"] == "Fixed"
    assert result["target_platform"] == "arm"
    assert result["type_reduction"] is True
    assert result["parity"]["argmax_mismatches"] == 0
    assert Path(result["ort_path"]).is_file()
    assert Path(result["config_path"]).is_file()
    with pytest.raises(FileExistsError):
        convert_fixed_ort_arm(onnx_path, output_dir, [_sample(3)])


def test_conversion_rejects_nonfinite_tolerance_before_writing(tmp_path: Path) -> None:
    _, onnx_path = _export(tmp_path)
    output_dir = tmp_path / "invalid_tolerance"

    with pytest.raises(ValueError, match="finite and non-negative"):
        convert_fixed_ort_arm(
            onnx_path,
            output_dir,
            [_sample(3)],
            max_abs_error=float("nan"),
        )
    assert not output_dir.exists()


def test_mobile_usability_captures_structural_results_and_log(tmp_path: Path) -> None:
    _, path = _export(tmp_path)
    result = check_ort_mobile_usability(path)

    assert type(result["prebuilt_mobile_package_supported"]) is bool
    assert type(result["nnapi_or_coreml_may_help"]) is bool
    assert result["checker_is_structural_not_device_certification"] is True
    assert "ORT Mobile" in result["log"]


def test_alternating_cpu_latency_summary(tmp_path: Path) -> None:
    _, onnx_path = _export(tmp_path)
    package = convert_fixed_ort_arm(
        onnx_path,
        tmp_path / "arm_package",
        [_sample(5)],
    )
    result = benchmark_onnx_vs_ort_cpu(
        onnx_path,
        package["ort_path"],
        _sample(6),
        intra_op_threads=1,
        inter_op_threads=1,
        warmups=2,
        trials=2,
        iterations_per_trial=3,
    )

    assert result["provider"] == "CPUExecutionProvider"
    assert result["order_rule"] == "alternate by (trial + iteration) modulo 2"
    assert result["pre_benchmark_parity"]["passed"] is True
    for name in ("onnx", "ort"):
        assert len(result["arms"][name]["trials_ms"]) == 2
        assert all(len(row) == 3 for row in result["arms"][name]["trials_ms"])
        assert result["arms"][name]["median_ms"] > 0.0
        assert result["arms"][name]["p95_ms"] > 0.0
    assert result["ort_to_onnx_ratio"]["median"] > 0.0
    assert result["ort_to_onnx_ratio"]["p95"] > 0.0


def test_latency_rejects_self_comparison_and_fake_ort(tmp_path: Path) -> None:
    _, onnx_path = _export(tmp_path)
    with pytest.raises(ValueError, match="one .onnx source and one .ort target"):
        benchmark_onnx_vs_ort_cpu(
            onnx_path,
            onnx_path,
            _sample(6),
            warmups=1,
            trials=1,
            iterations_per_trial=1,
        )

    fake_ort = tmp_path / "fake.ort"
    fake_ort.write_bytes(onnx_path.read_bytes())
    with pytest.raises(ValueError, match="ORT-format identifier"):
        benchmark_onnx_vs_ort_cpu(
            onnx_path,
            fake_ort,
            _sample(6),
            warmups=1,
            trials=1,
            iterations_per_trial=1,
        )


def test_parity_fails_closed_for_unavailable_provider(tmp_path: Path) -> None:
    model, path = _export(tmp_path)
    with pytest.raises(RuntimeError, match="providers are unavailable"):
        compare_pytorch_ort(
            model,
            path,
            [_sample()],
            providers=("DefinitelyMissingExecutionProvider",),
        )


def test_real_b17_folded_arms_share_stock_topology_and_ort_parity(
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    base = timm.create_model(
        "swiftformer_xs",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    arms = {
        "stock": SwiftFormerSurfaceFoldB17(
            copy.deepcopy(base), SURFACEFOLD_OFF_B17_MODE
        ),
        "control": SwiftFormerSurfaceFoldB17(
            copy.deepcopy(base), SURFACEFOLD_MEAN_CONTROL_B17_MODE
        ),
        "candidate": SwiftFormerSurfaceFoldB17(
            copy.deepcopy(base), SURFACEFOLD_SPATIAL_B17_MODE
        ),
    }
    sample = torch.zeros(1, 3, 224, 224)
    arrays = [sample.numpy()]
    fingerprints: dict[str, str] = {}
    for name, arm in arms.items():
        deployed = arm.fold_to_deploy().eval()
        assert sum(parameter.numel() for parameter in deployed.parameters()) == 3_035_570
        assert all("surfacefold" not in key and "factor_" not in key for key in deployed.state_dict())
        path = tmp_path / f"{name}.onnx"
        export_fixed_opset17_onnx(deployed, sample, path)
        fingerprints[name] = str(onnx_topology_fingerprint(path)["sha256"])
        parity = compare_pytorch_ort(
            deployed,
            path,
            arrays,
            providers=("CPUExecutionProvider",),
            intra_op_threads=1,
            inter_op_threads=1,
        )
        assert parity["max_abs_error"] <= 1.0e-5
        assert parity["argmax_mismatches"] == 0

    assert len(set(fingerprints.values())) == 1

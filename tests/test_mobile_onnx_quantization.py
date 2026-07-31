from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from trkh.inference.mobile_onnx_quantization import (
    CalibrationDataReader,
    compare_onnx_logits,
    onnx_operator_summary,
    quantize_mobile_onnx_qdq,
)
from trkh.inference.deploy import build_mobile_proxy_gate, evaluate_mobile_onnx_pair


class _TinyConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(32 * 16 * 16, 5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.conv(images))
        return self.classifier(features.flatten(1))


def _export_tiny_model(path: Path) -> nn.Module:
    torch.manual_seed(20260729)
    model = _TinyConvClassifier().eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 16, 16),
        path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    return model


def test_calibration_reader_is_train_only_and_rewindable() -> None:
    arrays = [
        np.zeros((2, 3, 4, 4), dtype=np.float32),
        np.ones((1, 3, 4, 4), dtype=np.float32),
    ]
    reader = CalibrationDataReader(
        (array for array in arrays),
        calibration_split="train",
        provenance={"dataset_split": "train", "test_used": False},
    )

    assert reader.provenance["sample_count"] == 3
    assert reader.provenance["calibration_split"] == "train"
    assert reader.get_next()["images"].shape[0] == 2
    assert reader.get_next()["images"].shape[0] == 1
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next()["images"].shape[0] == 2

    with pytest.raises(ValueError, match="train-only"):
        CalibrationDataReader(arrays, calibration_split="test")
    with pytest.raises(ValueError, match="train-only"):
        CalibrationDataReader(
            arrays,
            provenance={"nested": {"source_split": "test"}},
        )


def test_static_qdq_quantization_and_logit_comparison(tmp_path: Path) -> None:
    fp32_path = tmp_path / "tiny_fp32.onnx"
    int8_path = tmp_path / "tiny_int8_qdq.onnx"
    _export_tiny_model(fp32_path)

    rng = np.random.default_rng(20260729)
    calibration_batches = [
        {"images": rng.normal(size=(2, 3, 16, 16)).astype(np.float32)}
        for _ in range(5)
    ]
    reader = CalibrationDataReader(
        calibration_batches,
        calibration_split="train",
        provenance={"dataset": "synthetic_unit_test", "dataset_split": "train"},
    )
    quantization = quantize_mobile_onnx_qdq(
        fp32_path,
        int8_path,
        reader,
        provenance={"purpose": "unit_test", "calibration_split": "train"},
    )

    assert int8_path.is_file()
    assert quantization["format"] == "QDQ"
    assert quantization["activation_type"] == "QUInt8"
    assert quantization["weight_type"] == "QInt8"
    assert quantization["per_channel"] is True
    assert quantization["size_reduction_bytes"] > 0

    fp32_operators = onnx_operator_summary(fp32_path)
    int8_operators = onnx_operator_summary(int8_path)
    assert fp32_operators["conv"] >= 1
    assert fp32_operators["gemm"] + fp32_operators["matmul"] >= 1
    assert int8_operators["quantize_linear"] > 0
    assert int8_operators["dequantize_linear"] > 0
    assert int8_operators["quantized_target_op_coverage"] >= 0.90
    assert int8_operators["fully_quantized_target_op_coverage"] >= 0.90
    assert int8_operators["size_bytes"] < fp32_operators["size_bytes"]

    comparison_batches = [
        {"images": rng.normal(size=(3, 3, 16, 16)).astype(np.float32)},
        {"images": rng.normal(size=(1, 3, 16, 16)).astype(np.float32)},
    ]
    comparison = compare_onnx_logits(
        fp32_path,
        int8_path,
        comparison_batches,
    )
    assert comparison["total"] == 4
    assert comparison["batches"] == 2
    assert np.isfinite(comparison["max_abs_logit_error"])
    assert np.isfinite(comparison["max_abs_probability_error"])
    assert 0 <= comparison["argmax_mismatches"] <= comparison["total"]


def test_quantizer_rejects_test_provenance(tmp_path: Path) -> None:
    fp32_path = tmp_path / "tiny_fp32.onnx"
    _export_tiny_model(fp32_path)
    reader = CalibrationDataReader(
        [np.zeros((1, 3, 16, 16), dtype=np.float32)],
        calibration_split="train",
    )

    with pytest.raises(ValueError, match="train-only"):
        quantize_mobile_onnx_qdq(
            fp32_path,
            tmp_path / "forbidden.onnx",
            reader,
            provenance={"source_split": "test"},
        )
    assert not (tmp_path / "forbidden.onnx").exists()


def test_mobile_proxy_aligned_accuracy_and_gate(tmp_path: Path) -> None:
    fp32_path = tmp_path / "tiny_fp32.onnx"
    int8_path = tmp_path / "tiny_int8_qdq.onnx"
    model = _export_tiny_model(fp32_path)
    rng = np.random.default_rng(9)
    calibration = [
        rng.normal(size=(2, 3, 16, 16)).astype(np.float32)
        for _ in range(5)
    ]
    quantization = quantize_mobile_onnx_qdq(
        fp32_path,
        int8_path,
        CalibrationDataReader(
            calibration,
            provenance={"dataset_split": "train", "test_used": False},
        ),
        provenance={"calibration_split": "train"},
    )
    images = torch.from_numpy(
        rng.normal(size=(8, 3, 16, 16)).astype(np.float32)
    )
    labels = torch.tensor([0, 1, 2, 3, 4, 1, 0, 2], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=False)

    evaluation = evaluate_mobile_onnx_pair(
        reference_model=model,
        fp32_onnx_path=fp32_path,
        int8_onnx_path=int8_path,
        loader=loader,
        class_names=[f"class_{index}" for index in range(5)],
    )
    gate = build_mobile_proxy_gate(
        evaluation=evaluation,
        quantization=quantization,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        max_fp32_probability_error=1e-5,
        max_macro_f1_drop=1.0,
        max_class1_f1_drop=1.0,
        max_int8_mib=1.0,
        max_parameters=1_000_000,
    )

    assert evaluation["samples"] == 8
    assert evaluation["torch_to_onnx_fp32"]["argmax_mismatches"] == 0
    assert gate["status"] == "passed"
    assert gate["target_device_certified"] is False

    weak_quality_gate = build_mobile_proxy_gate(
        evaluation=evaluation,
        quantization=quantization,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        max_fp32_probability_error=1e-5,
        max_macro_f1_drop=1.0,
        max_class1_f1_drop=1.0,
        max_int8_mib=1.0,
        max_parameters=1_000_000,
        min_int8_macro_f1=1.0,
        min_int8_class1_f1=1.0,
        min_int8_class1_precision=1.0,
        min_int8_class1_recall=1.0,
        min_qdq_target_op_coverage=0.90,
    )
    assert weak_quality_gate["status"] == "failed"
    assert weak_quality_gate["checks"]["int8_absolute_macro_f1"] is False

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, helper
from onnxruntime.quantization import (
    CalibrationDataReader as OrtCalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


ArrayInput = Union[np.ndarray, Mapping[str, np.ndarray]]
_TRAIN_SPLIT = "train"
_SPLIT_KEYS = {
    "split",
    "data_split",
    "dataset_split",
    "source_split",
    "calibration_split",
}
_TEST_ACCESS_KEYS = {"test_used", "uses_test", "test_accessed"}
_QUANTIZED_OP_TYPES = ("Conv", "MatMul", "Gemm")


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_split(value: object) -> str:
    return str(value or "").strip().lower()


def _validate_train_only_provenance(
    provenance: Optional[Mapping[str, object]],
    *,
    context: str,
) -> Dict[str, object]:
    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise TypeError(f"{context} provenance must be a mapping.")

    normalized = {str(key): value for key, value in provenance.items()}

    def visit(mapping: Mapping[str, object], prefix: str) -> None:
        for raw_key, value in mapping.items():
            key = str(raw_key).strip().lower()
            qualified_key = f"{prefix}.{key}" if prefix else key
            if key in _SPLIT_KEYS or key.endswith("_split"):
                split = _normalized_split(value)
                if split != _TRAIN_SPLIT:
                    raise ValueError(
                        f"{context} is train-only; {qualified_key}={value!r} is forbidden."
                    )
            if key in _TEST_ACCESS_KEYS and bool(value):
                raise ValueError(
                    f"{context} is train-only; {qualified_key}=True is forbidden."
                )
            if isinstance(value, Mapping):
                visit(value, qualified_key)

    visit(normalized, "")
    return normalized


def _freeze_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Calibration input {name!r} must be a numpy.ndarray.")
    if value.ndim < 1 or value.size <= 0:
        raise ValueError(
            f"Calibration input {name!r} must have a non-empty batch dimension."
        )
    if value.dtype.kind not in "biuf":
        raise TypeError(
            f"Calibration input {name!r} has unsupported dtype {value.dtype}."
        )
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"Calibration input {name!r} contains non-finite values.")
    return np.ascontiguousarray(value).copy()


def _normalize_record(record: ArrayInput, *, input_name: str) -> Dict[str, np.ndarray]:
    if isinstance(record, np.ndarray):
        mapping: Mapping[str, object] = {input_name: record}
    elif isinstance(record, Mapping):
        if not record:
            raise ValueError("Calibration input mapping must not be empty.")
        mapping = record
    else:
        raise TypeError(
            "Calibration samples must be numpy arrays or name-to-array mappings."
        )

    normalized: Dict[str, np.ndarray] = {}
    for raw_name, value in mapping.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Calibration input names must not be empty.")
        if name in normalized:
            raise ValueError(f"Duplicate calibration input name after normalization: {name!r}.")
        normalized[name] = _freeze_array(value, name=name)

    batch_sizes = {int(value.shape[0]) for value in normalized.values()}
    if len(batch_sizes) != 1:
        raise ValueError(
            "All tensors in one calibration record must share the same batch size."
        )
    return normalized


class CalibrationDataReader(OrtCalibrationDataReader):
    """Materialized, rewindable ONNX calibration reader restricted to train data.

    Each record may be a single NumPy array (bound to ``input_name``) or a
    mapping from ONNX input name to NumPy array. Materialization deliberately
    freezes generator output and makes ``rewind`` deterministic for ORT.
    """

    def __init__(
        self,
        samples: Iterable[ArrayInput],
        *,
        input_name: str = "images",
        calibration_split: str = _TRAIN_SPLIT,
        provenance: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        normalized_input_name = str(input_name or "").strip()
        if not normalized_input_name:
            raise ValueError("input_name must not be empty.")
        split = _normalized_split(calibration_split)
        if split != _TRAIN_SPLIT:
            raise ValueError(
                f"Mobile INT8 calibration is train-only; split={calibration_split!r} is forbidden."
            )
        user_provenance = _validate_train_only_provenance(
            provenance,
            context="Calibration reader",
        )

        records = [
            _normalize_record(record, input_name=normalized_input_name)
            for record in samples
        ]
        if not records:
            raise ValueError("At least one train calibration record is required.")

        expected_names = tuple(sorted(records[0]))
        expected_specs = {
            name: (records[0][name].dtype.str, tuple(records[0][name].shape[1:]))
            for name in expected_names
        }
        for index, record in enumerate(records[1:], start=1):
            names = tuple(sorted(record))
            if names != expected_names:
                raise ValueError(
                    "Calibration input names must be identical for every record: "
                    f"record_0={expected_names}, record_{index}={names}."
                )
            for name in expected_names:
                spec = (record[name].dtype.str, tuple(record[name].shape[1:]))
                if spec != expected_specs[name]:
                    raise ValueError(
                        f"Calibration input {name!r} dtype/trailing shape changed at "
                        f"record {index}: expected={expected_specs[name]}, got={spec}."
                    )

        sample_count = sum(int(record[expected_names[0]].shape[0]) for record in records)
        self.calibration_split = split
        self._records = records
        self._cursor = 0
        self._provenance: Dict[str, object] = {
            **user_provenance,
            "calibration_split": _TRAIN_SPLIT,
            "test_used": False,
            "record_count": int(len(records)),
            "sample_count": int(sample_count),
            "input_names": list(expected_names),
            "input_specs": {
                name: {
                    "dtype": str(records[0][name].dtype),
                    "trailing_shape": list(records[0][name].shape[1:]),
                }
                for name in expected_names
            },
        }

    @property
    def provenance(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._provenance, ensure_ascii=False))

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._cursor >= len(self._records):
            return None
        record = self._records[self._cursor]
        self._cursor += 1
        return record

    def rewind(self) -> None:
        self._cursor = 0


def _iter_graph_nodes(graph: GraphProto) -> Iterable[NodeProto]:
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from _iter_graph_nodes(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    yield from _iter_graph_nodes(nested_graph)


def onnx_operator_summary(path: Union[str, Path]) -> Dict[str, object]:
    """Return file size and standard/mobile-relevant ONNX operator counts."""

    model_path = Path(path).expanduser().absolute()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    nodes = list(_iter_graph_nodes(model.graph))
    counts = Counter(node.op_type for node in nodes)
    producer_op = {
        output_name: node.op_type
        for node in nodes
        for output_name in node.output
        if output_name
    }
    target_nodes = [node for node in nodes if node.op_type in _QUANTIZED_OP_TYPES]
    quantized_target_ops = 0
    fully_quantized_target_ops = 0
    for node in target_nodes:
        quantized_inputs = sum(
            1
            for input_name in list(node.input)[:2]
            if producer_op.get(input_name) == "DequantizeLinear"
        )
        if quantized_inputs >= 1:
            quantized_target_ops += 1
        if quantized_inputs >= min(2, len(node.input)):
            fully_quantized_target_ops += 1
    target_op_count = len(target_nodes)
    operators = {name: int(counts[name]) for name in sorted(counts)}
    return {
        "path": str(model_path),
        "size_bytes": int(model_path.stat().st_size),
        "sha256": _sha256_file(model_path),
        "node_count": int(sum(counts.values())),
        "initializer_count": int(len(model.graph.initializer)),
        "conv": int(counts["Conv"]),
        "matmul": int(counts["MatMul"]),
        "gemm": int(counts["Gemm"]),
        "quantize_linear": int(counts["QuantizeLinear"]),
        "dequantize_linear": int(counts["DequantizeLinear"]),
        "target_op_count": int(target_op_count),
        "quantized_target_op_count": int(quantized_target_ops),
        "fully_quantized_target_op_count": int(fully_quantized_target_ops),
        "quantized_target_op_coverage": float(
            quantized_target_ops / max(1, target_op_count)
        ),
        "fully_quantized_target_op_coverage": float(
            fully_quantized_target_ops / max(1, target_op_count)
        ),
        "operators": operators,
    }


def _reader_provenance(calibration_reader: OrtCalibrationDataReader) -> Dict[str, object]:
    split = _normalized_split(getattr(calibration_reader, "calibration_split", ""))
    if split != _TRAIN_SPLIT:
        raise ValueError(
            "Quantization requires a calibration reader with calibration_split='train'."
        )
    value = getattr(calibration_reader, "provenance", None)
    if callable(value):
        value = value()
    if not isinstance(value, Mapping):
        raise ValueError(
            "Quantization requires explicit train-only calibration provenance."
        )
    normalized = _validate_train_only_provenance(
        value,
        context="Calibration reader",
    )
    if _normalized_split(normalized.get("calibration_split")) != _TRAIN_SPLIT:
        raise ValueError("Calibration provenance must record calibration_split='train'.")
    if bool(normalized.get("test_used", False)):
        raise ValueError("Calibration provenance must record test_used=False.")
    return normalized


def _set_quantization_metadata(
    model: ModelProto,
    *,
    fp32_summary: Mapping[str, object],
    calibration_provenance: Mapping[str, object],
    extra_provenance: Mapping[str, object],
) -> None:
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata.update(
        {
            "trkh.mobile_proxy": "onnxruntime_static_qdq_int8",
            "trkh.quantization.format": "QDQ",
            "trkh.quantization.activation_type": "QUInt8",
            "trkh.quantization.weight_type": "QInt8",
            "trkh.quantization.per_channel": "true",
            "trkh.quantization.calibration_split": _TRAIN_SPLIT,
            "trkh.quantization.test_used": "false",
            "trkh.quantization.source_sha256": str(fp32_summary["sha256"]),
            "trkh.quantization.calibration_provenance": json.dumps(
                calibration_provenance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "trkh.quantization.extra_provenance": json.dumps(
                extra_provenance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    helper.set_model_props(model, metadata)


def quantize_mobile_onnx_qdq(
    fp32_path: Union[str, Path],
    int8_path: Union[str, Path],
    calibration_reader: OrtCalibrationDataReader,
    *,
    provenance: Optional[Mapping[str, object]] = None,
    overwrite: bool = False,
    require_size_reduction: bool = True,
    min_target_op_coverage: float = 0.90,
) -> Dict[str, object]:
    """Create a train-calibrated, per-channel QDQ INT8 mobile proxy.

    The artifact is written to a temporary sibling and promoted atomically only
    after ONNX validation, Q/DQ presence, provenance, and optional size gates.
    This is a desktop engineering proxy; target-device certification remains a
    separate requirement.
    """

    source = Path(fp32_path).expanduser().absolute()
    destination = Path(int8_path).expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(f"FP32 ONNX model does not exist: {source}")
    if source == destination:
        raise ValueError("FP32 and INT8 ONNX paths must be different.")
    if destination.exists() and not bool(overwrite):
        raise FileExistsError(
            f"INT8 output already exists; pass overwrite=True explicitly: {destination}"
        )
    if not isinstance(calibration_reader, OrtCalibrationDataReader):
        raise TypeError("calibration_reader must implement ORT CalibrationDataReader.")

    calibration_provenance = _reader_provenance(calibration_reader)
    extra_provenance = _validate_train_only_provenance(
        provenance,
        context="Quantization",
    )
    source_summary = onnx_operator_summary(source)
    if (
        int(source_summary["conv"])
        + int(source_summary["matmul"])
        + int(source_summary["gemm"])
        <= 0
    ):
        raise ValueError("FP32 model contains no Conv, MatMul, or Gemm operators to quantize.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp.onnx"
    )
    calibration_reader.rewind()
    try:
        quantize_static(
            model_input=source,
            model_output=temporary,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=list(_QUANTIZED_OP_TYPES),
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
        quantized_model = onnx.load(str(temporary), load_external_data=True)
        _set_quantization_metadata(
            quantized_model,
            fp32_summary=source_summary,
            calibration_provenance=calibration_provenance,
            extra_provenance=extra_provenance,
        )
        onnx.checker.check_model(quantized_model)
        onnx.save_model(quantized_model, str(temporary))

        temporary_summary = onnx_operator_summary(temporary)
        if int(temporary_summary["quantize_linear"]) <= 0:
            raise RuntimeError("Quantized model contains no QuantizeLinear nodes.")
        if int(temporary_summary["dequantize_linear"]) <= 0:
            raise RuntimeError("Quantized model contains no DequantizeLinear nodes.")
        observed_coverage = float(
            temporary_summary.get("fully_quantized_target_op_coverage", 0.0)
        )
        if observed_coverage < float(min_target_op_coverage):
            raise RuntimeError(
                "Fully quantized QDQ target-op coverage is too low: "
                f"observed={observed_coverage:.6f}, "
                f"required={float(min_target_op_coverage):.6f}."
            )
        source_size = int(source_summary["size_bytes"])
        quantized_size = int(temporary_summary["size_bytes"])
        if bool(require_size_reduction) and quantized_size >= source_size:
            raise RuntimeError(
                "QDQ artifact did not reduce model size: "
                f"fp32={source_size} bytes, int8={quantized_size} bytes."
            )

        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    int8_summary = onnx_operator_summary(destination)
    reduction_bytes = int(source_summary["size_bytes"]) - int(int8_summary["size_bytes"])
    reduction_ratio = float(reduction_bytes / max(1, int(source_summary["size_bytes"])))
    return {
        "status": "completed",
        "artifact_role": "desktop_mobile_proxy_not_target_device_certification",
        "format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "calibration": calibration_provenance,
        "provenance": extra_provenance,
        "fp32": source_summary,
        "int8": int8_summary,
        "size_reduction_bytes": int(reduction_bytes),
        "size_reduction_ratio": reduction_ratio,
        "minimum_target_op_coverage": float(min_target_op_coverage),
    }


def _session_output_name(
    fp32_session: ort.InferenceSession,
    int8_session: ort.InferenceSession,
    explicit_name: Optional[str],
) -> str:
    fp32_names = [output.name for output in fp32_session.get_outputs()]
    int8_names = [output.name for output in int8_session.get_outputs()]
    if explicit_name is not None:
        name = str(explicit_name).strip()
        if not name or name not in fp32_names or name not in int8_names:
            raise ValueError(
                f"Requested output {explicit_name!r} is not shared by both models."
            )
        return name
    if "logits" in fp32_names and "logits" in int8_names:
        return "logits"
    if not fp32_names or not int8_names or fp32_names[0] != int8_names[0]:
        raise ValueError(
            f"FP32/INT8 output schemas differ: fp32={fp32_names}, int8={int8_names}."
        )
    return fp32_names[0]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.maximum(
        exponentials.sum(axis=1, keepdims=True),
        np.finfo(exponentials.dtype).tiny,
    )


def compare_onnx_logits(
    fp32_path: Union[str, Path],
    int8_path: Union[str, Path],
    inputs: Iterable[Mapping[str, np.ndarray]],
    *,
    output_name: Optional[str] = None,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> Dict[str, object]:
    """Compare FP32 and INT8 classification logits on exactly aligned inputs."""

    source = Path(fp32_path).expanduser().absolute()
    quantized = Path(int8_path).expanduser().absolute()
    if not source.is_file() or not quantized.is_file():
        raise FileNotFoundError(
            f"Both ONNX models must exist: fp32={source}, int8={quantized}."
        )
    provider_list = [str(provider).strip() for provider in providers if str(provider).strip()]
    if not provider_list:
        raise ValueError("At least one ONNX Runtime provider is required.")

    fp32_session = ort.InferenceSession(str(source), providers=provider_list)
    int8_session = ort.InferenceSession(str(quantized), providers=provider_list)
    fp32_inputs = {entry.name for entry in fp32_session.get_inputs()}
    int8_inputs = {entry.name for entry in int8_session.get_inputs()}
    if fp32_inputs != int8_inputs:
        raise ValueError(
            f"FP32/INT8 input schemas differ: fp32={sorted(fp32_inputs)}, "
            f"int8={sorted(int8_inputs)}."
        )
    selected_output = _session_output_name(
        fp32_session,
        int8_session,
        output_name,
    )

    maximum_logit_error = 0.0
    maximum_probability_error = 0.0
    mismatch_count = 0
    total = 0
    batch_count = 0
    for batch_index, raw_feed in enumerate(inputs):
        if not isinstance(raw_feed, Mapping):
            raise TypeError(f"Comparison batch {batch_index} must be an input mapping.")
        feed = _normalize_record(raw_feed, input_name="images")
        feed_names = set(feed)
        if feed_names != fp32_inputs:
            raise ValueError(
                f"Comparison batch {batch_index} input names differ: "
                f"expected={sorted(fp32_inputs)}, got={sorted(feed_names)}."
            )
        fp32_logits = np.asarray(
            fp32_session.run([selected_output], feed)[0],
            dtype=np.float64,
        )
        int8_logits = np.asarray(
            int8_session.run([selected_output], feed)[0],
            dtype=np.float64,
        )
        if fp32_logits.ndim == 1:
            fp32_logits = fp32_logits[None, :]
        if int8_logits.ndim == 1:
            int8_logits = int8_logits[None, :]
        if fp32_logits.ndim != 2 or int8_logits.ndim != 2:
            raise ValueError(
                "Classification logits must have shape [B, C]: "
                f"fp32={fp32_logits.shape}, int8={int8_logits.shape}."
            )
        if fp32_logits.shape != int8_logits.shape:
            raise ValueError(
                f"FP32/INT8 logit shapes differ: {fp32_logits.shape} vs {int8_logits.shape}."
            )
        if not np.isfinite(fp32_logits).all() or not np.isfinite(int8_logits).all():
            raise ValueError("FP32/INT8 logits contain non-finite values.")

        logit_error = float(np.max(np.abs(fp32_logits - int8_logits)))
        fp32_probabilities = _softmax(fp32_logits)
        int8_probabilities = _softmax(int8_logits)
        probability_error = float(
            np.max(np.abs(fp32_probabilities - int8_probabilities))
        )
        maximum_logit_error = max(maximum_logit_error, logit_error)
        maximum_probability_error = max(
            maximum_probability_error,
            probability_error,
        )
        mismatch_count += int(
            np.count_nonzero(
                np.argmax(fp32_logits, axis=1) != np.argmax(int8_logits, axis=1)
            )
        )
        total += int(fp32_logits.shape[0])
        batch_count += 1

    if total <= 0:
        raise ValueError("At least one comparison sample is required.")
    if not math.isfinite(maximum_logit_error) or not math.isfinite(
        maximum_probability_error
    ):
        raise RuntimeError("Computed ONNX comparison errors are non-finite.")
    return {
        "fp32_path": str(source),
        "int8_path": str(quantized),
        "output_name": selected_output,
        "providers": provider_list,
        "batches": int(batch_count),
        "total": int(total),
        "max_abs_logit_error": float(maximum_logit_error),
        "max_abs_probability_error": float(maximum_probability_error),
        "argmax_mismatches": int(mismatch_count),
        "argmax_mismatch_rate": float(mismatch_count / total),
    }


__all__ = [
    "CalibrationDataReader",
    "compare_onnx_logits",
    "onnx_operator_summary",
    "quantize_mobile_onnx_qdq",
]

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import shutil
import time
import uuid
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import AttributeProto, GraphProto, ModelProto, SparseTensorProto, TensorProto
from onnxruntime.tools.convert_onnx_models_to_ort import (
    OptimizationStyle,
    convert_onnx_models_to_ort,
)
from onnxruntime.tools.mobile_helpers import (
    check_model_can_use_ort_mobile_pkg,
    usability_checker,
)
from torch import Tensor, nn


_CPU_PROVIDER = "CPUExecutionProvider"
_STANDARD_DOMAINS = frozenset(("", "ai.onnx"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_model(path: Path) -> ModelProto:
    resolved = path.expanduser().absolute()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {resolved}")
    model = onnx.load(str(resolved), load_external_data=True)
    onnx.checker.check_model(model)
    return model


def _require_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _float_token(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"ONNX topology contains a non-finite float attribute: {number!r}")
    return number.hex()


def _tensor_structure(
    tensor: TensorProto, *, include_value: bool = False
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": str(tensor.name),
        "data_type": int(tensor.data_type),
        "dims": [int(value) for value in tensor.dims],
    }
    if include_value:
        result["value_sha256"] = hashlib.sha256(
            tensor.SerializeToString(deterministic=True)
        ).hexdigest()
    return result


def _sparse_tensor_structure(
    tensor: SparseTensorProto, *, include_value: bool = False
) -> dict[str, object]:
    return {
        "dims": [int(value) for value in tensor.dims],
        "values": _tensor_structure(tensor.values, include_value=include_value),
        "indices": _tensor_structure(tensor.indices, include_value=include_value),
    }


def _type_structure(value: onnx.ValueInfoProto) -> dict[str, object]:
    return {
        "name": str(value.name),
        "type_proto": value.type.SerializeToString(deterministic=True).hex(),
    }


def _attribute_structure(attribute: AttributeProto) -> dict[str, object]:
    result: dict[str, object] = {
        "name": str(attribute.name),
        "type": int(attribute.type),
        "ref_attr_name": str(attribute.ref_attr_name),
    }
    kind = attribute.type
    if kind == AttributeProto.FLOAT:
        result["value"] = _float_token(attribute.f)
    elif kind == AttributeProto.INT:
        result["value"] = int(attribute.i)
    elif kind == AttributeProto.STRING:
        result["value"] = bytes(attribute.s).hex()
    elif kind == AttributeProto.TENSOR:
        result["value"] = _tensor_structure(attribute.t, include_value=True)
    elif kind == AttributeProto.GRAPH:
        result["value"] = _graph_structure(attribute.g)
    elif kind == AttributeProto.SPARSE_TENSOR:
        result["value"] = _sparse_tensor_structure(
            attribute.sparse_tensor, include_value=True
        )
    elif kind == AttributeProto.TYPE_PROTO:
        result["value"] = attribute.tp.SerializeToString(deterministic=True).hex()
    elif kind == AttributeProto.FLOATS:
        result["value"] = [_float_token(value) for value in attribute.floats]
    elif kind == AttributeProto.INTS:
        result["value"] = [int(value) for value in attribute.ints]
    elif kind == AttributeProto.STRINGS:
        result["value"] = [bytes(value).hex() for value in attribute.strings]
    elif kind == AttributeProto.TENSORS:
        result["value"] = [
            _tensor_structure(value, include_value=True) for value in attribute.tensors
        ]
    elif kind == AttributeProto.GRAPHS:
        result["value"] = [_graph_structure(value) for value in attribute.graphs]
    elif kind == AttributeProto.SPARSE_TENSORS:
        result["value"] = [
            _sparse_tensor_structure(value, include_value=True)
            for value in attribute.sparse_tensors
        ]
    elif kind == AttributeProto.TYPE_PROTOS:
        result["value"] = [
            value.SerializeToString(deterministic=True).hex()
            for value in attribute.type_protos
        ]
    elif kind != AttributeProto.UNDEFINED:
        raise ValueError(
            f"Unsupported ONNX attribute type {kind} for {attribute.name!r}."
        )
    return result


def _graph_structure(graph: GraphProto) -> dict[str, object]:
    return {
        "name": str(graph.name),
        "inputs": [_type_structure(value) for value in graph.input],
        "outputs": [_type_structure(value) for value in graph.output],
        "value_info": [_type_structure(value) for value in graph.value_info],
        "initializers": [_tensor_structure(value) for value in graph.initializer],
        "sparse_initializers": [
            _sparse_tensor_structure(value) for value in graph.sparse_initializer
        ],
        "nodes": [
            {
                "name": str(node.name),
                "domain": str(node.domain),
                "op_type": str(node.op_type),
                "inputs": [str(value) for value in node.input],
                "outputs": [str(value) for value in node.output],
                "attributes": [
                    _attribute_structure(attribute) for attribute in node.attribute
                ],
            }
            for node in graph.node
        ],
    }


def _iter_graphs(graph: GraphProto) -> Iterable[GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from _iter_graphs(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from _iter_graphs(nested)


def _standard_opset17_contract(model: ModelProto) -> None:
    imports = [
        (str(item.domain) if item.domain else "ai.onnx", int(item.version))
        for item in model.opset_import
    ]
    if imports != [("ai.onnx", 17)]:
        raise ValueError(f"Expected only ai.onnx opset 17, got {imports}.")
    if model.functions or model.training_info:
        raise ValueError("Deployment ONNX must not contain local functions or training info.")
    domains = {
        str(node.domain)
        for graph in _iter_graphs(model.graph)
        for node in graph.node
    }
    unsupported = sorted(domain for domain in domains if domain not in _STANDARD_DOMAINS)
    if unsupported:
        raise ValueError(f"Deployment ONNX contains non-standard domains: {unsupported}.")


def onnx_topology_fingerprint(path: str | Path) -> dict[str, object]:
    """Hash recursive graph structure while deliberately excluding tensor values."""

    model_path = Path(path).expanduser().absolute()
    model = _checked_model(model_path)
    _standard_opset17_contract(model)
    descriptor = {
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {
                "domain": str(value.domain) if value.domain else "ai.onnx",
                "version": int(value.version),
            }
            for value in model.opset_import
        ],
        "graph": _graph_structure(model.graph),
    }
    encoded = json.dumps(
        descriptor,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "descriptor": descriptor,
    }


def _static_shape(value: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value.type.tensor_type
    if not value.type.HasField("tensor_type") or not tensor_type.HasField("shape"):
        raise ValueError(f"ONNX value {value.name!r} is not a shaped tensor.")
    shape: list[int] = []
    for dimension in tensor_type.shape.dim:
        if dimension.WhichOneof("value") != "dim_value" or dimension.dim_value <= 0:
            raise ValueError(f"ONNX value {value.name!r} has a dynamic/invalid shape.")
        shape.append(int(dimension.dim_value))
    return shape


def export_fixed_opset17_onnx(
    model: nn.Module,
    example_input: Tensor,
    output_path: str | Path,
    *,
    input_name: str = "images",
    output_name: str = "logits",
    overwrite: bool = False,
) -> dict[str, object]:
    """Export one static batch-1 tensor input/output using standard opset 17."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(example_input, Tensor) or example_input.ndim < 1:
        raise TypeError("example_input must be a non-scalar torch.Tensor.")
    if int(example_input.shape[0]) != 1:
        raise ValueError("Deployment export requires batch size exactly one.")
    if not torch.isfinite(example_input).all():
        raise ValueError("example_input contains non-finite values.")
    input_name = str(input_name).strip()
    output_name = str(output_name).strip()
    if not input_name or not output_name:
        raise ValueError("input_name and output_name must not be blank.")

    destination = Path(output_path).expanduser().absolute()
    if destination.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output must use the .onnx suffix: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"ONNX output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.partial.onnx"
    )
    export_model = copy.deepcopy(model).cpu().eval()
    sample = example_input.detach().cpu().contiguous()
    try:
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                export_model,
                sample,
                str(temporary),
                export_params=True,
                input_names=[input_name],
                output_names=[output_name],
                opset_version=17,
                do_constant_folding=True,
                dynamic_axes=None,
                keep_initializers_as_inputs=False,
            )
        exported = _checked_model(temporary)
        _standard_opset17_contract(exported)
        if len(exported.graph.input) != 1 or len(exported.graph.output) != 1:
            raise ValueError("Deployment ONNX must have exactly one input and one output.")
        observed_input = exported.graph.input[0]
        observed_output = exported.graph.output[0]
        input_shape = _static_shape(observed_input)
        output_shape = _static_shape(observed_output)
        if observed_input.name != input_name or input_shape != list(sample.shape):
            raise ValueError(
                f"Export input contract drifted: {observed_input.name!r}, {input_shape}."
            )
        if observed_output.name != output_name or not output_shape or output_shape[0] != 1:
            raise ValueError(
                f"Export output contract drifted: {observed_output.name!r}, {output_shape}."
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    fingerprint = onnx_topology_fingerprint(destination)
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": int(destination.stat().st_size),
        "opset": 17,
        "input_name": input_name,
        "input_shape": list(sample.shape),
        "output_name": output_name,
        "output_shape": output_shape,
        "topology_sha256": fingerprint["sha256"],
    }


def _session_options(*, intra_op_threads: int, inter_op_threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = _require_positive_int(
        intra_op_threads, name="intra_op_threads"
    )
    options.inter_op_num_threads = _require_positive_int(
        inter_op_threads, name="inter_op_threads"
    )
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def _providers(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError("providers must be a sequence of provider names, not a string.")
    providers = [str(provider).strip() for provider in value]
    if not providers or any(not provider for provider in providers):
        raise ValueError("providers must contain at least one non-blank provider.")
    if len(set(providers)) != len(providers):
        raise ValueError(f"providers contains duplicates: {providers}.")
    unavailable = sorted(set(providers) - set(ort.get_available_providers()))
    if unavailable:
        raise RuntimeError(f"Requested ONNX Runtime providers are unavailable: {unavailable}.")
    return providers


def _ort_session(
    path: Path,
    *,
    providers: Sequence[str],
    intra_op_threads: int,
    inter_op_threads: int,
) -> ort.InferenceSession:
    requested = _providers(providers)
    session = ort.InferenceSession(
        str(path),
        sess_options=_session_options(
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        ),
        providers=requested,
    )
    if session.get_providers() != requested:
        raise RuntimeError(
            f"ONNX Runtime provider contract drifted: requested={requested}, "
            f"active={session.get_providers()}."
        )
    return session


def _validated_arrays(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, ...]:
    frozen: list[np.ndarray] = []
    for index, value in enumerate(arrays):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Synthetic input {index} must be a numpy.ndarray.")
        if value.ndim < 1 or value.shape[0] != 1 or value.size <= 0:
            raise ValueError(f"Synthetic input {index} must be a non-empty batch-1 array.")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Synthetic input {index} has invalid dtype or non-finite values.")
        frozen.append(np.ascontiguousarray(value).copy())
    if not frozen:
        raise ValueError("At least one synthetic input array is required.")
    return tuple(frozen)


def _session_schema(session: ort.InferenceSession) -> tuple[str, str, tuple[int, ...]]:
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Parity requires one ONNX input and one ONNX output.")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if not all(type(value) is int and value > 0 for value in input_meta.shape):
        raise ValueError(f"Parity requires a fixed ONNX input shape: {input_meta.shape}.")
    return input_meta.name, output_meta.name, tuple(input_meta.shape)


def compare_pytorch_ort(
    model: nn.Module,
    onnx_path: str | Path,
    arrays: Iterable[np.ndarray],
    *,
    providers: Sequence[str] = (_CPU_PROVIDER,),
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
    max_abs_error: float = 1.0e-5,
    require_argmax_match: bool = True,
) -> dict[str, object]:
    """Fail closed unless PyTorch and ORT agree on aligned synthetic arrays."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    tolerance = float(max_abs_error)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("max_abs_error must be finite and non-negative.")
    samples = _validated_arrays(arrays)
    path = Path(onnx_path).expanduser().absolute()
    onnx_model = _checked_model(path)
    _standard_opset17_contract(onnx_model)
    session = _ort_session(
        path,
        providers=providers,
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    input_name, output_name, expected_shape = _session_schema(session)
    torch_model = copy.deepcopy(model).cpu().eval()
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    mismatches = 0
    with torch.inference_mode():
        for index, array in enumerate(samples):
            if array.shape != expected_shape:
                raise ValueError(
                    f"Synthetic input {index} shape drifted: expected={expected_shape}, "
                    f"got={array.shape}."
                )
            torch_output = torch_model(torch.from_numpy(array.copy()))
            if not isinstance(torch_output, Tensor):
                raise TypeError("PyTorch deployment model must return one Tensor.")
            expected = torch_output.detach().cpu().numpy()
            observed = np.asarray(session.run([output_name], {input_name: array})[0])
            if expected.shape != observed.shape:
                raise ValueError(
                    f"PyTorch/ORT output shapes differ: {expected.shape} vs {observed.shape}."
                )
            if not np.isfinite(expected).all() or not np.isfinite(observed).all():
                raise FloatingPointError("PyTorch/ORT comparison produced non-finite output.")
            difference = np.abs(expected.astype(np.float64) - observed.astype(np.float64))
            maximum = max(maximum, float(difference.max(initial=0.0)))
            absolute_sum += float(difference.sum())
            element_count += int(difference.size)
            expected_rows = expected.reshape(expected.shape[0], -1)
            observed_rows = observed.reshape(observed.shape[0], -1)
            mismatches += int(
                np.count_nonzero(
                    np.argmax(expected_rows, axis=1) != np.argmax(observed_rows, axis=1)
                )
            )
    if maximum > tolerance or (require_argmax_match and mismatches != 0):
        raise RuntimeError(
            "PyTorch/ORT parity gate failed: "
            f"max_abs={maximum:.9g}, tolerance={tolerance:.9g}, "
            f"argmax_mismatches={mismatches}."
        )
    return {
        "passed": True,
        "onnx_path": str(path),
        "providers": list(providers),
        "intra_op_threads": int(intra_op_threads),
        "inter_op_threads": int(inter_op_threads),
        "samples": len(samples),
        "max_abs_error": maximum,
        "mean_abs_error": float(absolute_sum / max(1, element_count)),
        "argmax_mismatches": mismatches,
    }


def _compare_ort_artifacts(
    first_path: Path,
    second_path: Path,
    arrays: Iterable[np.ndarray],
    *,
    intra_op_threads: int,
    inter_op_threads: int,
    max_abs_error: float,
) -> dict[str, object]:
    tolerance = float(max_abs_error)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("max_abs_error must be finite and non-negative.")
    first_path = Path(first_path).expanduser().absolute()
    second_path = Path(second_path).expanduser().absolute()
    if first_path.suffix.lower() != ".onnx" or second_path.suffix.lower() != ".ort":
        raise ValueError("Artifact parity requires one .onnx source and one .ort target.")
    if not first_path.is_file() or not second_path.is_file():
        raise FileNotFoundError("Both ONNX and ORT-format artifacts must exist.")
    if first_path.samefile(second_path):
        raise ValueError("ONNX and ORT-format artifacts must be distinct files.")
    with second_path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8 or header[4:8] != b"ORTM":
        raise ValueError("The .ort target does not contain the ORT-format identifier.")
    _standard_opset17_contract(_checked_model(first_path))
    samples = _validated_arrays(arrays)
    first = _ort_session(
        first_path,
        providers=(_CPU_PROVIDER,),
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    second = _ort_session(
        second_path,
        providers=(_CPU_PROVIDER,),
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    first_input, first_output, expected_shape = _session_schema(first)
    second_input, second_output, second_shape = _session_schema(second)
    if expected_shape != second_shape:
        raise ValueError(f"ONNX/ORT input shapes differ: {expected_shape} vs {second_shape}.")
    maximum = 0.0
    mismatches = 0
    for index, array in enumerate(samples):
        if array.shape != expected_shape:
            raise ValueError(
                f"Synthetic input {index} shape drifted: expected={expected_shape}, got={array.shape}."
            )
        expected = np.asarray(first.run([first_output], {first_input: array})[0])
        observed = np.asarray(second.run([second_output], {second_input: array})[0])
        if expected.shape != observed.shape:
            raise ValueError(f"ONNX/ORT output shapes differ: {expected.shape} vs {observed.shape}.")
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise FloatingPointError("ONNX/ORT comparison produced non-finite output.")
        maximum = max(
            maximum,
            float(
                np.max(
                    np.abs(expected.astype(np.float64) - observed.astype(np.float64)),
                    initial=0.0,
                )
            ),
        )
        mismatches += int(
            np.count_nonzero(
                np.argmax(expected.reshape(expected.shape[0], -1), axis=1)
                != np.argmax(observed.reshape(observed.shape[0], -1), axis=1)
            )
        )
    if maximum > tolerance or mismatches:
        raise RuntimeError(
            "ONNX/ORT-format reload parity failed: "
            f"max_abs={maximum:.9g}, tolerance={tolerance:.9g}, "
            f"argmax_mismatches={mismatches}."
        )
    return {
        "passed": True,
        "samples": len(samples),
        "max_abs_error": maximum,
        "argmax_mismatches": mismatches,
    }


def convert_fixed_ort_arm(
    onnx_path: str | Path,
    output_dir: str | Path,
    arrays: Iterable[np.ndarray],
    *,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
    max_abs_error: float = 1.0e-5,
) -> dict[str, object]:
    """Atomically create a type-reduced ARM Fixed-style ORT-format package."""

    tolerance = float(max_abs_error)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("max_abs_error must be finite and non-negative.")
    source = Path(onnx_path).expanduser().absolute()
    source_model = _checked_model(source)
    _standard_opset17_contract(source_model)
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"ORT output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    stage.mkdir()
    try:
        convert_onnx_models_to_ort(
            source,
            output_dir=stage,
            optimization_styles=[OptimizationStyle.Fixed],
            custom_op_library_path=None,
            target_platform="arm",
            save_optimized_onnx_model=False,
            allow_conversion_failures=False,
            enable_type_reduction=True,
        )
        ort_files = list(stage.glob("*.ort"))
        config_files = list(stage.glob("*.config"))
        if len(ort_files) != 1 or len(config_files) != 1:
            raise RuntimeError(
                "ORT conversion produced an unexpected artifact set: "
                f"ort={ort_files}, config={config_files}."
            )
        parity = _compare_ort_artifacts(
            source,
            ort_files[0],
            arrays,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
            max_abs_error=tolerance,
        )
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    ort_path = destination / ort_files[0].name
    config_path = destination / config_files[0].name
    return {
        "passed": True,
        "source_onnx_path": str(source),
        "ort_path": str(ort_path),
        "ort_sha256": _sha256_file(ort_path),
        "ort_size_bytes": int(ort_path.stat().st_size),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "optimization_style": "Fixed",
        "target_platform": "arm",
        "type_reduction": True,
        "providers": [_CPU_PROVIDER],
        "parity": parity,
    }


def check_ort_mobile_usability(onnx_path: str | Path) -> dict[str, object]:
    """Capture ORT Mobile package and NNAPI/CoreML heuristic checker results."""

    path = Path(onnx_path).expanduser().absolute()
    model = _checked_model(path)
    _standard_opset17_contract(model)
    stream = io.StringIO()
    logger = logging.getLogger(f"trkh.surfacefold.mobile.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("ORT Mobile prebuilt-package compatibility")
        package_supported = check_model_can_use_ort_mobile_pkg.run_check(
            path,
            check_model_can_use_ort_mobile_pkg.get_default_config_path(),
            logger,
        )
        logger.info("ORT Mobile NNAPI/CoreML usability heuristic")
        accelerator_recommended = usability_checker.analyze_model(
            path,
            skip_optimize=False,
            logger=logger,
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
    captured = stream.getvalue()
    if not captured.strip():
        raise RuntimeError("ORT Mobile usability checker produced no diagnostic log.")
    return {
        "onnx_path": str(path),
        "prebuilt_mobile_package_supported": bool(package_supported),
        "nnapi_or_coreml_may_help": bool(accelerator_recommended),
        "checker_is_structural_not_device_certification": True,
        "log": captured,
    }


def _latency_distribution(trials: list[list[float]]) -> dict[str, object]:
    values = np.asarray([value for trial in trials for value in trial], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise RuntimeError("Latency benchmark produced empty, non-finite, or non-positive data.")
    return {
        "trials_ms": trials,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95, method="linear")),
    }


def benchmark_onnx_vs_ort_cpu(
    onnx_path: str | Path,
    ort_path: str | Path,
    array: np.ndarray,
    *,
    intra_op_threads: int = 4,
    inter_op_threads: int = 1,
    warmups: int = 15,
    trials: int = 5,
    iterations_per_trial: int = 100,
    max_abs_error: float = 1.0e-5,
) -> dict[str, object]:
    """Benchmark ONNX and ORT format with a deterministic alternating order."""

    warmup_count = _require_positive_int(warmups, name="warmups")
    trial_count = _require_positive_int(trials, name="trials")
    iteration_count = _require_positive_int(
        iterations_per_trial, name="iterations_per_trial"
    )
    sample = _validated_arrays((array,))[0]
    paths = {
        "onnx": Path(onnx_path).expanduser().absolute(),
        "ort": Path(ort_path).expanduser().absolute(),
    }
    tolerance = float(max_abs_error)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("max_abs_error must be finite and non-negative.")
    if paths["onnx"].suffix.lower() != ".onnx" or paths["ort"].suffix.lower() != ".ort":
        raise ValueError("Latency requires one .onnx source and one .ort target.")
    if not paths["onnx"].is_file() or not paths["ort"].is_file():
        raise FileNotFoundError("Both latency artifacts must exist.")
    if paths["onnx"].samefile(paths["ort"]):
        raise ValueError("Latency artifacts must be distinct files.")
    parity = _compare_ort_artifacts(
        paths["onnx"],
        paths["ort"],
        (sample,),
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
        max_abs_error=tolerance,
    )
    sessions = {
        name: _ort_session(
            path,
            providers=(_CPU_PROVIDER,),
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        )
        for name, path in paths.items()
    }
    schemas = {name: _session_schema(session) for name, session in sessions.items()}
    if schemas["onnx"][2] != schemas["ort"][2] or sample.shape != schemas["onnx"][2]:
        raise ValueError(
            f"Latency input shapes differ: sample={sample.shape}, schemas={schemas}."
        )

    def run(name: str) -> float:
        input_name, output_name, _ = schemas[name]
        started = time.perf_counter_ns()
        output = sessions[name].run([output_name], {input_name: sample})[0]
        elapsed = float((time.perf_counter_ns() - started) / 1.0e6)
        if not np.isfinite(output).all():
            raise FloatingPointError(f"{name} latency inference produced non-finite output.")
        return elapsed

    names = ("onnx", "ort")
    for index in range(warmup_count):
        order = names[index % 2 :] + names[: index % 2]
        for name in order:
            run(name)

    measurements: dict[str, list[list[float]]] = {name: [] for name in names}
    for trial in range(trial_count):
        rows: dict[str, list[float]] = {name: [] for name in names}
        for iteration in range(iteration_count):
            offset = (trial + iteration) % 2
            order = names[offset:] + names[:offset]
            for name in order:
                rows[name].append(run(name))
        for name in names:
            measurements[name].append(rows[name])

    summaries = {
        name: _latency_distribution(measurements[name]) for name in names
    }
    return {
        "provider": _CPU_PROVIDER,
        "intra_op_threads": int(intra_op_threads),
        "inter_op_threads": int(inter_op_threads),
        "execution_mode": "ORT_SEQUENTIAL",
        "warmups": warmup_count,
        "trials": trial_count,
        "iterations_per_trial": iteration_count,
        "order_rule": "alternate by (trial + iteration) modulo 2",
        "pre_benchmark_parity": parity,
        "arms": summaries,
        "ort_to_onnx_ratio": {
            "median": float(
                summaries["ort"]["median_ms"] / summaries["onnx"]["median_ms"]
            ),
            "p95": float(
                summaries["ort"]["p95_ms"] / summaries["onnx"]["p95_ms"]
            ),
        },
    }


__all__ = [
    "benchmark_onnx_vs_ort_cpu",
    "check_ort_mobile_usability",
    "compare_pytorch_ort",
    "convert_fixed_ort_arm",
    "export_fixed_opset17_onnx",
    "onnx_topology_fingerprint",
]

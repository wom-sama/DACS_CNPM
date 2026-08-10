from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Dict, List, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from trkh.tools.pair_surface_ddf_a0_engine import (
    ROLE_NAMES,
    apply_ddf_standard,
    initialize_sidecar,
    numpy_ddf,
    parameter_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")
ENGINE_PATH = (
    REPOSITORY_ROOT
    / "trkh"
    / "tools"
    / "pair_surface_ddf_a0_engine.py"
)
ENGINE_TEST_PATH = (
    REPOSITORY_ROOT / "tests" / "test_pair_surface_ddf_a0_engine.py"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_pair_surface_ddf_a0_engineering_preflight_20260725"
    / "summary.json"
)

EXPECTED_PARAMETERS = {
    "ddf_full": 9380,
    "static_matched": 9435,
    "ddf_spatial_only": 9380,
    "ddf_channel_only": 9380,
    "ddf_full_repeat": 9380,
}
MAX_ORACLE_ERROR = 1e-10
MAX_ORT_ERROR = 1e-5
MAX_BATCH1_MEAN_MS = 2.0
MAX_BATCH1_P95_MS = 2.5
MAX_BATCH32_MEAN_MS = 2.5
MAX_BATCH32_P95_MS = 3.0
MAX_PEAK_CUDA_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _read_lock() -> tuple[Mapping[str, object], str]:
    sidecar = LOCK_SHA_PATH.read_text(encoding="ascii").strip().split()
    if len(sidecar) != 2 or sidecar[1] != LOCK_PATH.name:
        raise ValueError("DDF lock SHA sidecar is malformed")
    observed = _sha256(LOCK_PATH)
    if observed != sidecar[0]:
        raise ValueError("DDF lock SHA sidecar differs")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected_state = "prospective_no_candidate_implementation_or_observation"
    if lock["state"] != expected_state:
        raise ValueError(f"DDF lock state differs: {lock['state']}")
    return lock, observed


def _equation_oracle() -> Dict[str, object]:
    rng = np.random.default_rng(20260725)
    inputs = rng.normal(size=(2, 3, 5, 6))
    channel = rng.normal(size=(2, 3, 9))
    spatial = rng.normal(size=(2, 9, 5, 6))
    observed = apply_ddf_standard(
        torch.from_numpy(inputs),
        torch.from_numpy(channel),
        torch.from_numpy(spatial),
    ).numpy()
    expected = numpy_ddf(inputs, channel, spatial)
    maximum = float(np.max(np.abs(observed - expected)))
    return {
        "dtype": "float64",
        "input_shape": list(inputs.shape),
        "maximum_absolute_error": maximum,
        "limit": MAX_ORACLE_ERROR,
        "passed": maximum <= MAX_ORACLE_ERROR,
    }


def _role_contracts() -> Dict[str, object]:
    contracts: Dict[str, object] = {}
    common_states: Dict[str, Mapping[str, torch.Tensor]] = {}
    for role in ROLE_NAMES:
        model = initialize_sidecar(role, fold=0).eval()
        common_states[role] = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        contract = parameter_contract(model)
        inputs = torch.randn(2, 3, 64, 64)
        trace = model.forward_with_trace(inputs)
        contract["outputs_finite"] = bool(
            torch.isfinite(trace["scores"]).all()
            and torch.isfinite(trace["attention_maps"]).all()
        )
        contract["output_shape"] = list(trace["scores"].shape)
        contract["parameter_count_passed"] = (
            contract["parameter_count"] == EXPECTED_PARAMETERS[role]
        )
        contracts[role] = contract

    primary = (
        "ddf_full",
        "static_matched",
        "ddf_spatial_only",
        "ddf_channel_only",
    )
    shared_names = set.intersection(
        *(set(common_states[role]) for role in primary)
    )
    shared_names = {
        name
        for name in shared_names
        if all(
            common_states[role][name].shape
            == common_states["ddf_full"][name].shape
            for role in primary
        )
    }
    shared_exact = all(
        torch.equal(
            common_states["ddf_full"][name],
            common_states[role][name],
        )
        for name in shared_names
        for role in primary[1:]
    )
    return {
        "roles": contracts,
        "matching_primary_tensors": len(shared_names),
        "matching_primary_tensors_bit_exact": shared_exact,
        "passed": bool(
            shared_exact
            and all(
                row["parameter_count_passed"] and row["outputs_finite"]
                for row in contracts.values()
            )
        ),
    }


def _onnx_and_tensorrt() -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    model = initialize_sidecar("ddf_full", fold=0).eval()
    example = torch.randn(2, 3, 64, 64)
    with tempfile.TemporaryDirectory(prefix="trkh_ddf_a0_") as directory:
        onnx_path = Path(directory) / "pair_surface_ddf.onnx"
        with torch.inference_mode():
            reference = model(example).numpy()
        torch.onnx.export(
            model,
            example,
            onnx_path,
            opset_version=17,
            input_names=["images"],
            output_names=["scores"],
            dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
        graph = onnx.load(str(onnx_path))
        onnx.checker.check_model(graph)
        domains = sorted({str(node.domain) for node in graph.graph.node})
        operators = sorted({str(node.op_type) for node in graph.graph.node})
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        observed = session.run(None, {"images": example.numpy()})[0]
        ort_error = float(np.max(np.abs(observed - reference)))
        dynamic_shapes: Dict[str, List[int]] = {}
        dynamic_errors: Dict[str, float] = {}
        for batch in (1, 32):
            values = torch.randn(batch, 3, 64, 64)
            expected = model(values).detach().numpy()
            actual = session.run(None, {"images": values.numpy()})[0]
            dynamic_shapes[str(batch)] = list(actual.shape)
            dynamic_errors[str(batch)] = float(
                np.max(np.abs(actual - expected))
            )

        trt_available = False
        trt_version = "not-installed"
        parse_passed = False
        build_passed = False
        engine_bytes = 0
        parser_errors: List[str] = []
        try:
            import tensorrt as trt

            trt_available = True
            trt_version = str(trt.__version__)
            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            parse_passed = bool(parser.parse(onnx_path.read_bytes()))
            parser_errors = [
                str(parser.get_error(index))
                for index in range(parser.num_errors)
            ]
            if parse_passed:
                config = builder.create_builder_config()
                config.set_memory_pool_limit(
                    trt.MemoryPoolType.WORKSPACE,
                    1 << 30,
                )
                config.set_flag(trt.BuilderFlag.FP16)
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    "images",
                    (1, 3, 64, 64),
                    (8, 3, 64, 64),
                    (32, 3, 64, 64),
                )
                config.add_optimization_profile(profile)
                serialized = builder.build_serialized_network(network, config)
                build_passed = serialized is not None
                if serialized is not None:
                    engine_bytes = len(bytes(serialized))
        except Exception as error:  # pragma: no cover - environment dependent
            parser_errors.append(repr(error))

        onnx_record = {
            "opset": 17,
            "nodes": len(graph.graph.node),
            "domains": domains,
            "operators": operators,
            "standard_domains_only": all(
                value in {"", "ai.onnx"} for value in domains
            ),
            "sha256_ephemeral": _sha256(onnx_path),
            "bytes_ephemeral": onnx_path.stat().st_size,
            "retained": False,
            "ort_maximum_absolute_error": ort_error,
            "ort_dynamic_shapes": dynamic_shapes,
            "ort_dynamic_errors": dynamic_errors,
        }
    return {
        "onnx": onnx_record,
        "tensorrt": {
            "available": trt_available,
            "version": trt_version,
            "parse_passed": parse_passed,
            "build_passed": build_passed,
            "engine_bytes_ephemeral": engine_bytes,
            "parser_errors": parser_errors,
            "retained": False,
        },
        "passed": bool(
            onnx_record["standard_domains_only"]
            and ort_error <= MAX_ORT_ERROR
            and all(
                value <= MAX_ORT_ERROR for value in dynamic_errors.values()
            )
            and trt_available
            and parse_passed
            and build_passed
        ),
    }


def _benchmark_batch(
    batch_size: int,
    *,
    iterations: int,
) -> Dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = initialize_sidecar(
        "ddf_full",
        fold=0,
        device="cuda",
    ).eval().to(dtype=torch.float16)
    inputs = torch.randn(
        batch_size,
        3,
        64,
        64,
        device="cuda",
        dtype=torch.float16,
    )
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(50):
            model(inputs)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    with torch.inference_mode():
        eager_output = model(inputs).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        graph_output = model(inputs)
    graph.replay()
    torch.cuda.synchronize()
    maximum_eager_graph_error = float(
        (eager_output - graph_output).abs().max()
    )
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    values = np.asarray(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )
    result = {
        "backend": "torch_cuda_graph_fp16_static_binding",
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "current_cuda_bytes": int(torch.cuda.memory_allocated()),
        "outputs_finite": bool(torch.isfinite(graph_output).all()),
        "maximum_eager_graph_error": maximum_eager_graph_error,
    }
    del graph, graph_output, eager_output, inputs, model
    torch.cuda.empty_cache()
    return result


def _cuda_benchmark() -> Dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False, "passed": False}
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    batch1 = _benchmark_batch(1, iterations=300)
    batch32 = _benchmark_batch(32, iterations=200)
    maximum_peak = max(
        int(batch1["peak_cuda_bytes"]),
        int(batch32["peak_cuda_bytes"]),
    )
    checks = {
        "batch1_mean": batch1["mean_ms"] <= MAX_BATCH1_MEAN_MS,
        "batch1_p95": batch1["p95_ms"] <= MAX_BATCH1_P95_MS,
        "batch32_mean": batch32["mean_ms"] <= MAX_BATCH32_MEAN_MS,
        "batch32_p95": batch32["p95_ms"] <= MAX_BATCH32_P95_MS,
        "peak_cuda": maximum_peak <= MAX_PEAK_CUDA_BYTES,
        "finite": bool(batch1["outputs_finite"] and batch32["outputs_finite"]),
        "cuda_graph_exact": bool(
            batch1["maximum_eager_graph_error"] == 0.0
            and batch32["maximum_eager_graph_error"] == 0.0
        ),
    }
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "measurement_scope": (
            "sidecar_compute_only_static_cuda_binding_no_host_or_preprocess_copy"
        ),
        "batch1": batch1,
        "batch32": batch32,
        "maximum_peak_cuda_bytes": maximum_peak,
        "limits": {
            "batch1_mean_ms": MAX_BATCH1_MEAN_MS,
            "batch1_p95_ms": MAX_BATCH1_P95_MS,
            "batch32_mean_ms": MAX_BATCH32_MEAN_MS,
            "batch32_p95_ms": MAX_BATCH32_P95_MS,
            "peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _nvidia_snapshot() -> Dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-gpu=name,driver_version,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu,pstate,clocks.sm,clocks.mem"
            ),
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_preflight(output_path: Path) -> Dict[str, object]:
    started = time.perf_counter()
    lock, lock_sha = _read_lock()
    equation = _equation_oracle()
    roles = _role_contracts()
    nvidia_before_benchmark = _nvidia_snapshot()
    benchmark = _cuda_benchmark()
    nvidia_after_benchmark = _nvidia_snapshot()
    export = _onnx_and_tensorrt()
    nvidia_after_export = _nvidia_snapshot()
    checks = {
        "prospective_lock": (
            lock["access"]["formal_runs_authorized"] == 0
            and lock["access"]["replay_runs_authorized"] == 0
        ),
        "equation_oracle": equation["passed"],
        "role_contracts": roles["passed"],
        "onnx_ort_tensorrt": export["passed"],
        "cuda_benchmark": benchmark["passed"],
        "no_candidate_data": True,
    }
    summary = {
        "protocol_id": "trkh_pair_surface_ddf_a0_20260725",
        "state": "synthetic_engineering_preflight",
        "created_utc": "2026-07-25T00:00:00Z",
        "lock": {"path": str(LOCK_PATH), "sha256": lock_sha},
        "sources": {
            "engine": {
                "path": str(ENGINE_PATH),
                "sha256": _sha256(ENGINE_PATH),
            },
            "engine_test": {
                "path": str(ENGINE_TEST_PATH),
                "sha256": _sha256(ENGINE_TEST_PATH),
            },
            "preflight": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "equation": equation,
        "roles": roles,
        "export": export,
        "benchmark": benchmark,
        "checks": checks,
        "passed": all(checks.values()),
        "candidate_data": {
            "cache_read": False,
            "raw_dataset_read": False,
            "labels_read": False,
            "validation_read": False,
            "test_read": False,
            "candidate_metric_created": False,
        },
        "resource": {
            "elapsed_seconds": time.perf_counter() - started,
            "temporary_onnx_retained": False,
            "temporary_engine_retained": False,
            "nvidia_before_benchmark": nvidia_before_benchmark,
            "nvidia_after_benchmark": nvidia_after_benchmark,
            "nvidia_after_export": nvidia_after_export,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "onnx": _package_version("onnx"),
            "onnxruntime": _package_version("onnxruntime-gpu"),
            "tensorrt": _package_version("tensorrt"),
        },
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Preflight output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    if not summary["passed"]:
        raise RuntimeError(f"Pair-Surface DDF preflight failed: {checks}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic Pair-Surface DDF A0 engineering preflight."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Unique summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(args.output)
    compact = {
        "passed": summary["passed"],
        "equation_max_abs": summary["equation"]["maximum_absolute_error"],
        "batch1_mean_p95_ms": [
            summary["benchmark"]["batch1"]["mean_ms"],
            summary["benchmark"]["batch1"]["p95_ms"],
        ],
        "batch32_mean_p95_ms": [
            summary["benchmark"]["batch32"]["mean_ms"],
            summary["benchmark"]["batch32"]["p95_ms"],
        ],
        "onnxruntime_max_abs": summary["export"]["onnx"][
            "ort_maximum_absolute_error"
        ],
        "tensorrt_engine_bytes": summary["export"]["tensorrt"][
            "engine_bytes_ephemeral"
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

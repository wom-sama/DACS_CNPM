from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import timm
import torch

from trkh.tools import audit_swiftformer_surfacefold_b18_preflight as b18


def _latency(onnx_ms: float = 10.0, ort_ms: float = 10.4) -> dict[str, object]:
    def arm(value: float) -> dict[str, object]:
        rows = [[value for _ in range(100)] for _ in range(5)]
        return {"trials_ms": rows, "median_ms": value, "p95_ms": value}

    return {
        "provider": "CPUExecutionProvider",
        "intra_op_threads": 4,
        "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "warmups": 15,
        "trials": 5,
        "iterations_per_trial": 100,
        "order_rule": "alternate by (trial + iteration) modulo 2",
        "pre_benchmark_parity": {
            "passed": True,
            "samples": 1,
            "max_abs_error": 1.0e-6,
            "argmax_mismatches": 0,
        },
        "arms": {"onnx": arm(onnx_ms), "ort": arm(ort_ms)},
        "ort_to_onnx_ratio": {
            "median": ort_ms / onnx_ms,
            "p95": ort_ms / onnx_ms,
        },
    }


def _target(target_platform: str) -> dict[str, object]:
    package_name = "host_native_amd64" if target_platform == "amd64" else "android_arm"
    manifest = {
        "optimization_style": "Fixed",
        "target_platform": target_platform,
        "enable_type_reduction": True,
        "save_optimized_onnx_model": False,
        "custom_op_library_path": None,
        "allow_conversion_failures": False,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "passed": True,
        "source_onnx_sha256": "b" * 64,
        "artifact_name": f"{package_name}.ort",
        "ort_sha256": "c" * 64,
        "ort_size_bytes": 10_500_000,
        "config_sha256": "d" * 64,
        "config_size_bytes": 1234,
        "optimization_style": "Fixed",
        "target_platform": target_platform,
        "type_reduction": True,
        "providers": ["CPUExecutionProvider"],
        "parity": {
            "passed": True,
            "samples": 64,
            "max_abs_error": 1.0e-6,
            "argmax_mismatches": 0,
        },
        "conversion_manifest": manifest,
        "conversion_manifest_sha256": manifest_sha256,
    }


def _deployment(*, stock: bool, topology: str = "a" * 64) -> dict[str, object]:
    return {
        "arm": "stock" if stock else "candidate",
        "parameters": b18.STOCK_PARAMETERS,
        "onnx": {
            "sha256": "b" * 64,
            "size_bytes": 10_000_000,
            "opset": 17,
            "input_name": "x",
            "input_shape": [1, 3, 224, 224],
            "output_name": "logits",
            "output_shape": [1, 5],
            "topology_sha256": topology,
        },
        "topology_sha256": topology,
        "parity": {
            "passed": True,
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": 4,
            "inter_op_threads": 1,
            "samples": 64,
            "max_abs_error": 1.0e-6,
            "mean_abs_error": 5.0e-7,
            "argmax_mismatches": 0,
        },
        "targets": {
            "host_native_amd64": _target("amd64"),
            "android_arm": _target("arm"),
        },
        "android_arm_mobile": {
            "prebuilt_mobile_package_supported": True,
            "nnapi_or_coreml_may_help": False,
            "checker_is_structural_not_device_certification": True,
            "log": "ORT Mobile prebuilt-package compatibility\nORT Mobile NNAPI/CoreML usability heuristic\n",
        },
        "native_amd64_latency": _latency(),
        "arm_host_diagnostic": {
            "status": "NOT_APPLICABLE_TARGET_MISMATCH",
            "host_architecture": "AMD64",
            "artifact_target": "arm",
            "executed": False,
            "gate": False,
        },
        "forbidden_identifiers": [],
        "native_amd64_ratio_is_gate": stock,
        "checks": {name: True for name in b18.DEPLOYMENT_CHECKS},
        "passed": True,
    }


def _mechanism() -> dict[str, object]:
    def role(name: str) -> dict[str, object]:
        names = {
            "head": ["backbone.head.weight", "backbone.head.bias"],
            "head_dist": ["backbone.head_dist.weight", "backbone.head_dist.bias"],
            "p": ["factor_p"],
            "d": ["factor_d"],
        }[name]
        return {"names": names, "present": True, "finite": True, "nonzero": True}

    relation = {
        "control": {
            "loss": 0.2,
            "p": role("p"),
            "d": role("d"),
            "candidate_centered_d_nonzero": False,
            "control_d_spatially_equal": True,
            "s3_gradient_nonzero": True,
            "teacher_gradient_absent": True,
        },
        "candidate": {
            "loss": 0.2,
            "p": role("p"),
            "d": role("d"),
            "candidate_centered_d_nonzero": True,
            "control_d_spatially_equal": False,
            "s3_gradient_nonzero": True,
            "teacher_gradient_absent": True,
        },
    }
    total = {}
    for name in ("stock", "control", "candidate"):
        roles = {key: role(key) for key in ("head", "head_dist")}
        if name != "stock":
            roles.update({key: role(key) for key in ("p", "d")})
        total[name] = {
            "ce": 1.0,
            "relation": 0.2,
            "total": 1.02,
            "roles": roles,
            "s3_gradient_nonzero": True,
            "teacher_gradient_absent": True,
        }
    return {
        "counts": {
            "stock": {"total": b18.STOCK_PARAMETERS, "trainable": b18.STOCK_TRAINABLE},
            "control": {"total": b18.ACTIVE_PARAMETERS, "trainable": b18.ACTIVE_TRAINABLE},
            "candidate": {"total": b18.ACTIVE_PARAMETERS, "trainable": b18.ACTIVE_TRAINABLE},
        },
        "relation_only_gradients": relation,
        "total_objective_gradients": total,
        "float64_oracle_max_abs": {"control": 1.0e-12, "candidate": 1.0e-12},
        "fold_parity_max_abs": {
            "control": {"pre_bn": 0.0, "features": 1.0e-7, "logits": 1.0e-7},
            "candidate": {"pre_bn": 0.0, "features": 1.0e-7, "logits": 1.0e-7},
        },
        "w0_b0_sha256": {"stock": "e" * 64, "control": "f" * 64, "candidate": "1" * 64},
        "checks": {name: True for name in b18.MECHANISM_CHECKS},
        "passed": True,
    }


def _cuda() -> dict[str, object]:
    def role(name: str) -> dict[str, object]:
        names = {
            "head": ["backbone.head.weight", "backbone.head.bias"],
            "head_dist": ["backbone.head_dist.weight", "backbone.head_dist.bias"],
            "p": ["factor_p"],
            "d": ["factor_d"],
        }[name]
        return {"names": names, "present": True, "finite": True, "nonzero": True}

    gradients = {}
    for arm in ("stock", "control", "candidate"):
        roles = {name: role(name) for name in ("head", "head_dist")}
        if arm != "stock":
            roles.update({name: role(name) for name in ("p", "d")})
        gradients[arm] = {
            "all": {
                "tensors": 10,
                "missing": [],
                "nonfinite": [],
                "all_present_finite": True,
            },
            "roles": roles,
            "s3_nonzero": True,
            "logits_dtype": "torch.bfloat16",
            "s3_dtype": "torch.bfloat16",
        }
    optimizer = {
        "trainable_tensors": 10,
        "state_tensors": 10,
        "parameter_set_exact": True,
        "finite": True,
        "shapes_exact": True,
        "steps_one": True,
    }
    w0 = {"stock": "7" * 64, "control": "8" * 64, "candidate": "9" * 64}
    return {
        "settings": {
            "batch": 16,
            "teacher_precision": "fp32_inference_once",
            "student_precision": "bf16_autocast",
            "arm_order": ["stock", "control", "candidate"],
            "optimizer": "AdamW_one_real_step_each",
            "gradient_clip": 0.7,
            "synthetic_ce_weights": list(b18.SYNTHETIC_CE_WEIGHTS),
            "tf32": False,
        },
        "exposure_sha256": "2" * 64,
        "teacher": {
            "calls": 1,
            "tokens_shape": [16, 261, 384],
            "raw_map_shape": [16, 384, 16, 16],
            "target_shape": [16, 84],
            "target_dtype": "torch.float32",
            "target_requires_grad": False,
            "parameter_gradients_present": 0,
        },
        "runtime_state": {
            "deterministic_algorithms": True,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "losses": {
            arm: {"ce": 1.0, "relation": 0.2, "total": 1.02}
            for arm in ("stock", "control", "candidate")
        },
        "gradients": gradients,
        "optimizer_states": {
            arm: copy.deepcopy(optimizer)
            for arm in ("stock", "control", "candidate")
        },
        "w0_b0_before": w0,
        "w0_b0_after": copy.deepcopy(w0),
        "memory": {
            "baseline_allocated": 100_000_000,
            "baseline_reserved": 200_000_000,
            "peak_allocated": 600_000_000,
            "peak_reserved": 700_000_000,
            "limit_allocated": b18.MAX_CUDA_ALLOCATED_BYTES,
        },
        "checks": {name: True for name in b18.CUDA_CHECKS},
        "passed": True,
    }


def _payload() -> dict[str, object]:
    stock = _deployment(stock=True)
    folded = {
        "stock_topology_sha256": stock["topology_sha256"],
        "arms": {
            "control": _deployment(stock=False),
            "candidate": _deployment(stock=False),
        },
        "checks": {name: True for name in b18.FOLDED_CHECKS},
        "passed": True,
    }
    sources = {name: "3" * 64 for name in b18._source_paths()}
    sources.update(b18.SOURCE_LOCKS)
    git = {
        "head": "4" * 40,
        "branch": b18.EXPECTED_BRANCH,
        "status": "",
        "clean": True,
        "head_is_commit": True,
    }
    def asset(lock: b18.AssetLock) -> dict[str, object]:
        return {
            "role": lock.role,
            "repo_id": lock.repo_id,
            "revision": lock.revision,
            "filename": lock.filename,
            "path": f"C:/cache/{lock.filename}",
            "bytes": lock.byte_count,
            "sha256": lock.sha256,
            "license": lock.license,
            "offline_cache_only": True,
        }

    weights = {"student": asset(b18.STUDENT_LOCK), "teacher": asset(b18.DINO_LOCK)}
    timm_sources = {
        "paths": {name: f"C:/site-packages/timm/{name}" for name in b18.EXPECTED_TIMM_HASHES},
        "sha256": dict(b18.EXPECTED_TIMM_HASHES),
    }
    nodeids = [f"test_node_{index}" for index in range(b18.EXPECTED_FOCUSED_TEST_COUNT)]
    child = {
        "schema_version": 1,
        "guard_root": os.path.abspath(os.fspath(b18.DATASET_ROOT)),
        "test_paths": list(b18.FOCUSED_TEST_PATHS),
        "audit": {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
        "collected_nodeids": nodeids,
        "executed_nodeids": copy.deepcopy(nodeids),
        "pytest_returncode": 0,
        "error": None,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": b18.PROTOCOL_ID,
        "protocol_sha256": b18.PROTOCOL_SHA256,
        "mode": "offline_label_free_stock_first_preflight",
        "created_at_unix": 1.0,
        "git": git,
        "source_hashes": sources,
        "runtime": dict(b18.EXPECTED_RUNTIME),
        "timm_sources": timm_sources,
        "device": {"requested": "cuda", "resolved": "cuda:0", **b18.EXPECTED_CUDA},
        "weights": weights,
        "strict_load": {
            "caller_rng_preserved": True,
            "student": {
                "runtime_class": "timm.models.swiftformer.SwiftFormer",
                "stage_channels": [48, 56, 112, 220],
                "parameters": b18.STOCK_PARAMETERS,
                "heads": [[220, 5], [220, 5]],
                "distilled_training": False,
                "nonzero_dropout_or_path": [],
                "strict_load": True,
                "head_reset": {
                    "seed": b18.SEED,
                    "initializer": "timm_trunc_normal_0p02_head_then_head_dist",
                    "bias_zero": True,
                    "state_sha256": b18.EXPECTED_HEAD_STATE_SHA256,
                    "caller_rng_preserved": True,
                },
                "asset_keys": 316,
            },
            "teacher": {
                "runtime_class": "timm.models.eva.Eva",
                "features": 384,
                "prefix_tokens": 5,
                "grid": [16, 16],
                "all_frozen": True,
                "strict_load": True,
                "asset_keys": 162,
            },
        },
        "focused_tests": {
            "python_executable": str(Path(sys.executable).resolve()),
            "test_paths": list(b18.FOCUSED_TEST_PATHS),
            "bootstrap_sha256": hashlib.sha256(
                b18._focused_test_bootstrap().encode("utf-8")
            ).hexdigest(),
            "returncode": 0,
            "pytest_returncode": 0,
            "passed_count": b18.EXPECTED_FOCUSED_TEST_COUNT,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "child_guard": child,
            "stdout": f"{b18.EXPECTED_FOCUSED_TEST_COUNT} passed in 1.0s\n",
            "stderr": "",
            "passed": True,
        },
        "stock_deployment": stock,
        "mechanism": _mechanism(),
        "folded_deployment": folded,
        "cuda": _cuda(),
        "isolation": {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
        "offline": {
            "environment": dict(b18.OFFLINE_ENV),
            "cublas_workspace_config": ":4096:8",
        },
        "end_rehash": {
            "git": git,
            "source_hashes": sources,
            "runtime": dict(b18.EXPECTED_RUNTIME),
            "timm_sources": timm_sources,
            "weights": weights,
        },
        "permissions": dict(b18.SUCCESS_PERMISSIONS),
        "authorization": dict(b18.AUTHORIZATION),
        "checks": {name: True for name in b18.TOP_CHECKS},
        "passed": True,
    }
    return payload


def test_locked_sources_and_parser_have_no_data_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    assert b18._sha256(root / "docs/TRKH_PRETRAINED_CLASSF_B18_TARGETMATCH_SURFACEFOLD_PROTOCOL_20260805.md") == b18.PROTOCOL_SHA256
    assert b18._sha256(root / "trkh/models/swiftformer_surfacefold_b17.py") == b18.MODEL_SHA256
    assert b18._sha256(root / "trkh/inference/surfacefold_deployment.py") == b18.DEPLOYMENT_SHA256
    arguments = b18._parse_args(["--output-dir", "runs/preflight_b18_surfacefold_xs_test"])
    assert set(vars(arguments)) == {"output_dir", "student_weight", "dino_weight", "device"}
    assert all("data" not in name and "label" not in name for name in vars(arguments))


def test_universal_source_map_is_unambiguous() -> None:
    assert set(b18._source_paths()) == {
        "protocol",
        "model",
        "model_test",
        "deployment",
        "deployment_test",
        "preflight_runner",
        "preflight_test",
        "formal_runner",
        "formal_runner_test",
    }


def test_output_scope_is_fresh_direct_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(b18, "_root", lambda: tmp_path)
    valid = tmp_path / "runs" / f"{b18.OUTPUT_PREFIX}case"
    assert b18._validated_output(valid) == valid
    with pytest.raises(ValueError):
        b18._validated_output(tmp_path / "elsewhere" / f"{b18.OUTPUT_PREFIX}case")
    valid.mkdir()
    with pytest.raises(FileExistsError):
        b18._validated_output(valid)


def test_isolation_guard_blocks_network_and_dataset() -> None:
    with pytest.raises(RuntimeError, match="isolation guard observed"):
        with b18._isolation_guard():
            with pytest.raises(RuntimeError, match="blocked network"):
                socket.getaddrinfo("example.com", 443)
    with pytest.raises(RuntimeError, match="isolation guard observed"):
        with b18._isolation_guard():
            with pytest.raises(RuntimeError, match="blocked dataset"):
                open(b18.DATASET_ROOT / "data.yaml", "rb")


def test_guarded_pytest_blocks_child_dataset_network_and_process(tmp_path: Path) -> None:
    if os.environ.get("TRKH_B18_IN_GUARDED_PYTEST") == "1":
        with pytest.raises(RuntimeError, match="isolation guard observed"):
            with b18._isolation_guard(deny_process=True):
                with pytest.raises(RuntimeError, match="blocked child process"):
                    subprocess.run([sys.executable, "-c", "pass"])
        return
    guarded_root = tmp_path / "guarded_data"
    guarded_root.mkdir()
    (guarded_root / "sentinel.txt").write_text("secret", encoding="utf-8")
    dataset_probe = tmp_path / "test_dataset_escape.py"
    dataset_probe.write_text(
        "import os\n"
        "def test_dataset():\n"
        "    open(os.path.join(os.environ['TRKH_B18_CHILD_GUARD_ROOT'], 'sentinel.txt')).read()\n"
        ,
        encoding="utf-8",
    )
    dataset_evidence = b18._run_guarded_pytest(
        (str(dataset_probe),), guard_root=guarded_root, pytest_root=tmp_path
    )
    assert dataset_evidence["returncode"] == 86
    assert dataset_evidence["child_guard"]["audit"]["dataset_attempts"]

    process_probe = tmp_path / "test_network_process_escape.py"
    process_probe.write_text(
        "import socket, subprocess, sys\n"
        "def test_network():\n"
        "    socket.getaddrinfo('example.com', 443)\n"
        "def test_process():\n"
        "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n",
        encoding="utf-8",
    )
    evidence = b18._run_guarded_pytest(
        (str(process_probe),),
        guard_root=tmp_path / "unused_guard_root",
        pytest_root=tmp_path,
    )
    audit = evidence["child_guard"]["audit"]
    assert evidence["returncode"] == 86
    assert audit["network_attempts"]
    assert audit["process_attempts"]


def test_head_reset_and_arm_builder_are_rng_safe_and_reproducible() -> None:
    first = timm.create_model("swiftformer_xs", pretrained=False, num_classes=1000)
    second = copy.deepcopy(first)
    before = torch.random.get_rng_state().clone()
    first_evidence = b18.reset_five_class_heads(first)
    second_evidence = b18.reset_five_class_heads(second)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert first_evidence["state_sha256"] == second_evidence["state_sha256"]
    bundle = b18.build_b18_arms(first)
    assert bundle.rng_preserved
    assert b18._common_state_equal(bundle.control, bundle.candidate)
    assert b18._state_storage_disjoint(bundle.control, bundle.candidate)
    assert [int(model._surfacefold_mode_id) for _, model in bundle.items()] == [0, 1, 2]


def test_initialized_cuda_rng_regression_and_b18_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert torch.cuda.is_available(), "B18 focused suite requires the locked CUDA host"
    torch.cuda.init()
    caller_cpu = torch.random.get_rng_state().clone()
    caller_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    try:
        before = b18._rng_digest()
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(b18.SEED + 991)
        retired = b18._rng_digest()
        assert retired["cpu"] == before["cpu"]
        assert retired["cuda"] != before["cuda"]
        torch.random.set_rng_state(caller_cpu)
        torch.cuda.set_rng_state_all(caller_cuda)

        student_template = timm.create_model(
            b18.STUDENT_TIMM_ID,
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        teacher_template = timm.create_model(
            b18.DINO_TIMM_ID,
            pretrained=False,
            num_classes=0,
            img_size=b18.DINO_IMAGE_SIZE,
        )
        student_state = student_template.state_dict()
        teacher_state = teacher_template.state_dict()

        def create_model(model_id: str, **kwargs):
            del kwargs
            if model_id == b18.STUDENT_TIMM_ID:
                return copy.deepcopy(student_template)
            if model_id == b18.DINO_TIMM_ID:
                return copy.deepcopy(teacher_template)
            raise AssertionError(model_id)

        def load_file(path: str, *, device: str):
            assert device == "cpu"
            return student_state if path.endswith("student.safetensors") else teacher_state

        monkeypatch.setattr(b18.timm, "create_model", create_model)
        monkeypatch.setattr(b18, "load_file", load_file)
        before_strict = b18._rng_digest()
        student, stock, _, evidence = b18._strict_load_models(
            Path("student.safetensors"), Path("teacher.safetensors")
        )
        assert evidence["caller_rng_preserved"] is True
        assert b18._rng_digest() == before_strict

        before_arms = b18._rng_digest()
        bundle = b18.build_active_arms_after_stock(student, stock)
        assert bundle.rng_preserved is True
        assert b18._rng_digest() == before_arms

        head_model = copy.deepcopy(student)
        before_head = b18._rng_digest()
        b18.reset_five_class_heads(head_model, seed=b18.SEED + 1)
        assert b18._rng_digest() == before_head
    finally:
        torch.random.set_rng_state(caller_cpu)
        torch.cuda.set_rng_state_all(caller_cuda)


def test_relation_only_helper_proves_candidate_and_control_gradient_geometry() -> None:
    base = timm.create_model("swiftformer_xs", pretrained=False, num_classes=5)
    bundle = b18.build_b18_arms(base)
    candidate = b18._relation_only_gradients(bundle.candidate)
    control = b18._relation_only_gradients(bundle.control)
    assert candidate["p"]["nonzero"] and candidate["d"]["nonzero"]
    assert candidate["candidate_centered_d_nonzero"]
    assert control["p"]["nonzero"] and control["d"]["nonzero"]
    assert control["control_d_spatially_equal"]
    assert candidate["teacher_gradient_absent"] and control["teacher_gradient_absent"]


def test_valid_payload_recomputes_all_nested_gates() -> None:
    checks = b18._payload_checks(_payload())
    assert checks and all(checks.values()), checks


def test_dual_target_export_uses_one_onnx_and_native_only_latency_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = torch.zeros(1, 3, 224, 224).numpy()
    arrays = tuple(sample for _ in range(64))
    conversion_calls: list[tuple[str, str]] = []
    benchmark_artifacts: list[str] = []

    def export(model, tensor, path, **kwargs):
        del model, tensor, kwargs
        path.write_bytes(b"one-exported-onnx")
        return {
            "path": str(path),
            "sha256": b18._sha256(path),
            "size_bytes": path.stat().st_size,
            "opset": 17,
            "input_name": "x",
            "input_shape": [1, 3, 224, 224],
            "output_name": "logits",
            "output_shape": [1, 5],
            "topology_sha256": "a" * 64,
        }

    def convert(onnx_path, output_dir, samples, *, target_platform, **kwargs):
        del samples, kwargs
        source = Path(onnx_path)
        manifest = {
            "optimization_style": "Fixed",
            "target_platform": target_platform,
            "enable_type_reduction": True,
            "save_optimized_onnx_model": False,
            "custom_op_library_path": None,
            "allow_conversion_failures": False,
        }
        conversion_calls.append((source.name, target_platform))
        return {
            "passed": True,
            "source_onnx_path": str(source),
            "source_onnx_sha256": b18._sha256(source),
            "ort_path": str(Path(output_dir) / f"{source.stem}.ort"),
            "ort_sha256": "c" * 64,
            "ort_size_bytes": 16,
            "config_path": str(Path(output_dir) / "ops.config"),
            "config_sha256": "d" * 64,
            "config_size_bytes": 8,
            "optimization_style": "Fixed",
            "target_platform": target_platform,
            "type_reduction": True,
            "providers": ["CPUExecutionProvider"],
            "parity": {
                "passed": True,
                "samples": 64,
                "max_abs_error": 1.0e-6,
                "argmax_mismatches": 0,
            },
            "conversion_manifest": manifest,
            "conversion_manifest_sha256": hashlib.sha256(
                json.dumps(
                    manifest,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def benchmark(onnx_path, ort_path, sample, **kwargs):
        del onnx_path, sample, kwargs
        benchmark_artifacts.append(Path(ort_path).name)
        return _latency()

    monkeypatch.setattr(b18, "STOCK_PARAMETERS", 0)
    monkeypatch.setattr(b18, "export_fixed_opset17_onnx", export)
    monkeypatch.setattr(
        b18,
        "onnx_topology_fingerprint",
        lambda path: {"sha256": "a" * 64, "descriptor": {"nodes": []}},
    )
    monkeypatch.setattr(
        b18,
        "compare_pytorch_ort",
        lambda *args, **kwargs: {
            "passed": True,
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": 4,
            "inter_op_threads": 1,
            "samples": 64,
            "max_abs_error": 1.0e-6,
            "mean_abs_error": 5.0e-7,
            "argmax_mismatches": 0,
            "onnx_path": "ignored",
        },
    )
    monkeypatch.setattr(
        b18,
        "check_ort_mobile_usability",
        lambda path: {
            "onnx_path": str(path),
            "prebuilt_mobile_package_supported": True,
            "nnapi_or_coreml_may_help": False,
            "checker_is_structural_not_device_certification": True,
            "log": "ORT Mobile prebuilt-package compatibility\nORT Mobile NNAPI/CoreML usability heuristic\n",
        },
    )
    monkeypatch.setattr(b18, "convert_fixed_ort", convert)
    monkeypatch.setattr(b18, "benchmark_onnx_vs_ort_cpu", benchmark)

    evidence = b18._portable_deployment_evidence(
        torch.nn.Identity(),
        "stock",
        tmp_path,
        arrays,
        require_host_ratio=True,
    )
    assert evidence["passed"] is True
    assert conversion_calls == [
        ("host_native_amd64.onnx", "amd64"),
        ("android_arm.onnx", "arm"),
    ]
    assert benchmark_artifacts == ["host_native_amd64.ort"]
    assert {
        package["source_onnx_sha256"] for package in evidence["targets"].values()
    } == {evidence["onnx"]["sha256"]}
    assert evidence["arm_host_diagnostic"]["gate"] is False

    stock = _deployment(stock=True)
    stock["native_amd64_latency"] = _latency(10.0, 10.6)
    assert b18._deployment_payload_ok(stock, stock=True) is False
    active = _deployment(stock=False)
    active["native_amd64_latency"] = _latency(10.0, 10.6)
    assert b18._deployment_payload_ok(active, stock=False) is True


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda p: p["permissions"].update(formal_train_permission=False), "permission_exact"),
        (lambda p: p["source_hashes"].update(protocol="0" * 64), "source_hashes_exact"),
        (lambda p: p["stock_deployment"]["onnx"].update(size_bytes=b18.MAX_ONNX_BYTES + 1), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["native_amd64_latency"]["ort_to_onnx_ratio"].update(p95=1.2), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["native_amd64_latency"]["arms"]["ort"]["trials_ms"][0].pop(), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["native_amd64_latency"]["arms"]["ort"]["trials_ms"][0].__setitem__(0, True), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["native_amd64_latency"].update(execution_mode="ORT_PARALLEL"), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["native_amd64_latency"].update(order_rule="forged"), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["android_arm_mobile"].update(log="checker ran"), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["targets"]["android_arm"].update(source_onnx_sha256="0" * 64), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["targets"]["host_native_amd64"].update(target_platform="arm"), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["arm_host_diagnostic"].update(gate=True), "stock_deployment_exact"),
        (lambda p: p["mechanism"]["float64_oracle_max_abs"].update(candidate=1.0e-4), "mechanism_exact"),
        (lambda p: p["mechanism"]["relation_only_gradients"]["candidate"]["p"].update(nonzero=False), "mechanism_exact"),
        (lambda p: p["mechanism"]["total_objective_gradients"]["stock"].update(s3_gradient_nonzero=False), "mechanism_exact"),
        (lambda p: p["folded_deployment"]["arms"]["candidate"].update(topology_sha256="9" * 64), "folded_deployment_exact"),
        (lambda p: p["cuda"]["memory"].update(peak_allocated=b18.MAX_CUDA_ALLOCATED_BYTES + 1), "cuda_exact"),
        (lambda p: p["cuda"].update(losses={}), "cuda_exact"),
        (lambda p: p["cuda"]["gradients"]["candidate"]["roles"]["p"].update(nonzero=False), "cuda_exact"),
        (lambda p: p["cuda"].update(optimizer_states={}), "cuda_exact"),
        (lambda p: p["isolation"]["network_attempts"].append("x"), "isolation_exact"),
        (lambda p: p["isolation"]["process_attempts"].append("x"), "isolation_exact"),
        (lambda p: p["strict_load"]["student"].update(stage_channels=[48, 56, 111, 220]), "strict_load_exact"),
        (lambda p: p["strict_load"]["student"]["head_reset"].update(seed=1), "strict_load_exact"),
        (lambda p: p["focused_tests"]["child_guard"]["audit"]["process_attempts"].append("x"), "focused_tests_exact"),
        (lambda p: p["focused_tests"].update(passed_count=1), "focused_tests_exact"),
        (lambda p: p.update(created_at_unix=True), "identity_exact"),
        (lambda p: p["device"].update(name="forged"), "device_exact"),
        (lambda p: p["checks"].update(cuda_batch16_contract=False), "top_checks_exact"),
        (lambda p: p.update(passed=False), "passed_exact"),
    ],
)
def test_payload_rejects_bypass_mutations(mutation, failed_check: str) -> None:
    payload = _payload()
    mutation(payload)
    assert b18._payload_checks(payload)[failed_check] is False


def test_payload_rejects_boolean_numeric_spoof() -> None:
    payload = _payload()
    payload["stock_deployment"]["onnx"]["size_bytes"] = True
    assert b18._payload_checks(payload)["stock_deployment_exact"] is False


def test_atomic_publish_is_canonical_hashed_and_non_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    result = b18._publish(destination, "preflight.json", {"z": 1, "a": True})
    raw = (destination / "preflight.json").read_bytes()
    assert raw == b18._canonical_bytes({"z": 1, "a": True})
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    assert (destination / "preflight.sha256").read_text() == f"{result['sha256']}\n"
    with pytest.raises(FileExistsError):
        b18._publish(destination, "preflight.json", {})


def test_stock_gate_failure_attaches_full_raw_timing_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _deployment(stock=True)
    evidence["native_amd64_latency"] = _latency(10.0, 14.0)
    evidence["checks"]["native_amd64_host_format_ratio"] = False
    evidence["passed"] = False

    class Stock:
        def fold_to_deploy(self):
            return object()

    monkeypatch.setattr(b18, "_synthetic_arrays", lambda: (object(),))
    monkeypatch.setattr(
        b18,
        "_portable_deployment_evidence",
        lambda *args, **kwargs: copy.deepcopy(evidence),
    )
    with pytest.raises(RuntimeError, match="stock deployment gate failed") as caught:
        b18._stock_deployment_contract(Stock())
    attached = caught.value.b18_gate_evidence
    assert attached["gate"] == "stock_deployment_first"
    assert attached["evidence"]["native_amd64_latency"]["arms"]["onnx"]["trials_ms"]
    assert len(attached["evidence"]["native_amd64_latency"]["arms"]["ort"]["trials_ms"]) == 5
    assert all(
        len(row) == 100
        for row in attached["evidence"]["native_amd64_latency"]["arms"]["ort"]["trials_ms"]
    )


def test_failure_artifact_atomically_hashes_attached_gate_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate_evidence = {
        "gate": "stock_deployment_first",
        "evidence": {"native_amd64_latency": _latency(10.0, 14.0), "passed": False},
    }
    failure_root = tmp_path / "failed"
    monkeypatch.setattr(b18, "_failure_path", lambda requested: failure_root)
    error = RuntimeError("closed")
    error.b18_gate_evidence = gate_evidence
    try:
        raise error
    except RuntimeError as caught:
        result = b18._write_failure(tmp_path / "requested", "stock", caught)
    payload = json.loads((failure_root / "failure.json").read_text(encoding="utf-8"))
    assert payload["gate_evidence"] == gate_evidence
    assert payload["gate_evidence_sha256"] == hashlib.sha256(
        b18._canonical_bytes(gate_evidence)
    ).hexdigest()
    assert payload["permissions"] == b18.NO_ACCESS_PERMISSIONS
    assert payload["passed"] is False
    assert result["sha256"] == hashlib.sha256(
        (failure_root / "failure.json").read_bytes()
    ).hexdigest()


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"x":1,"x":2}', object_pairs_hook=b18._no_duplicate_keys)


def test_build_preflight_is_stock_first_and_candidate_never_runs_after_stock_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    order: list[str] = []

    @contextmanager
    def isolation():
        yield {"dataset_attempts": [], "network_attempts": []}

    git = {"head": "a" * 40, "branch": b18.EXPECTED_BRANCH, "status": "", "clean": True, "head_is_commit": True}
    monkeypatch.setattr(b18, "_validated_output", lambda value: output)
    monkeypatch.setattr(b18, "_offline_environment", lambda: {"environment": b18.OFFLINE_ENV, "cublas_workspace_config": ":4096:8"})
    monkeypatch.setattr(b18, "_isolation_guard", isolation)
    monkeypatch.setattr(b18, "_git_contract", lambda: git)
    monkeypatch.setattr(b18, "_source_hashes", lambda: {**{name: "1" * 64 for name in b18._source_paths()}, **b18.SOURCE_LOCKS})
    monkeypatch.setattr(b18, "_runtime_contract", lambda: dict(b18.EXPECTED_RUNTIME))
    monkeypatch.setattr(b18, "_timm_contract", lambda: {"paths": {}, "sha256": b18.EXPECTED_TIMM_HASHES})
    monkeypatch.setattr(b18, "_device_contract", lambda value: (object(), {"requested": "cuda"}))
    monkeypatch.setattr(b18, "_resolve_asset", lambda lock, path: (tmp_path / lock.filename, {"offline_cache_only": True}))
    monkeypatch.setattr(b18, "_focused_tests", lambda: {"passed": True})
    monkeypatch.setattr(b18, "_strict_load_models", lambda a, c: (object(), object(), object(), {"student": {"strict_load": True}, "teacher": {"strict_load": True}}))

    def stock(model):
        del model
        order.append("stock")
        raise RuntimeError("stock closed")

    monkeypatch.setattr(b18, "_stock_deployment_contract", stock)
    monkeypatch.setattr(
        b18,
        "build_active_arms_after_stock",
        lambda base, stock: order.append("construct_active"),
    )
    monkeypatch.setattr(b18, "_mechanism_contract", lambda bundle: order.append("mechanism"))
    monkeypatch.setattr(b18, "_write_failure", lambda *args: {"artifact": "failure"})
    args = argparse.Namespace(output_dir=output, student_weight=None, dino_weight=None, device="cuda")
    with pytest.raises(RuntimeError, match="stock closed"):
        b18.build_preflight(args)
    assert order == ["stock"]


def test_main_turns_keyboard_interrupt_into_nonzero_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(b18, "build_preflight", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert b18.main(["--output-dir", "runs/preflight_b18_surfacefold_xs_interrupt"]) == 1
    assert '"passed": false' in capsys.readouterr().err

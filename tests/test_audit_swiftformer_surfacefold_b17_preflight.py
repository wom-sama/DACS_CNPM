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

from trkh.tools import audit_swiftformer_surfacefold_b17_preflight as b17


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


def _deployment(*, stock: bool, topology: str = "a" * 64) -> dict[str, object]:
    return {
        "arm": "stock" if stock else "candidate",
        "parameters": b17.STOCK_PARAMETERS,
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
        "mobile": {
            "prebuilt_mobile_package_supported": True,
            "nnapi_or_coreml_may_help": False,
            "checker_is_structural_not_device_certification": True,
            "log": "ORT Mobile prebuilt-package compatibility\nORT Mobile NNAPI/CoreML usability heuristic\n",
        },
        "ort": {
            "passed": True,
            "ort_sha256": "c" * 64,
            "ort_size_bytes": 10_500_000,
            "config_sha256": "d" * 64,
            "optimization_style": "Fixed",
            "target_platform": "arm",
            "type_reduction": True,
            "providers": ["CPUExecutionProvider"],
            "parity": {
                "passed": True,
                "samples": 64,
                "max_abs_error": 1.0e-6,
                "argmax_mismatches": 0,
            },
        },
        "latency": _latency(),
        "forbidden_identifiers": [],
        "host_ratio_is_gate": stock,
        "checks": {name: True for name in b17.DEPLOYMENT_CHECKS},
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
            "stock": {"total": b17.STOCK_PARAMETERS, "trainable": b17.STOCK_TRAINABLE},
            "control": {"total": b17.ACTIVE_PARAMETERS, "trainable": b17.ACTIVE_TRAINABLE},
            "candidate": {"total": b17.ACTIVE_PARAMETERS, "trainable": b17.ACTIVE_TRAINABLE},
        },
        "relation_only_gradients": relation,
        "total_objective_gradients": total,
        "float64_oracle_max_abs": {"control": 1.0e-12, "candidate": 1.0e-12},
        "fold_parity_max_abs": {
            "control": {"pre_bn": 0.0, "features": 1.0e-7, "logits": 1.0e-7},
            "candidate": {"pre_bn": 0.0, "features": 1.0e-7, "logits": 1.0e-7},
        },
        "w0_b0_sha256": {"stock": "e" * 64, "control": "f" * 64, "candidate": "1" * 64},
        "checks": {name: True for name in b17.MECHANISM_CHECKS},
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
            "synthetic_ce_weights": list(b17.SYNTHETIC_CE_WEIGHTS),
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
            "limit_allocated": b17.MAX_CUDA_ALLOCATED_BYTES,
        },
        "checks": {name: True for name in b17.CUDA_CHECKS},
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
        "checks": {name: True for name in b17.FOLDED_CHECKS},
        "passed": True,
    }
    sources = {name: "3" * 64 for name in b17._source_paths()}
    sources.update(b17.SOURCE_LOCKS)
    git = {
        "head": "4" * 40,
        "branch": b17.EXPECTED_BRANCH,
        "status": "",
        "clean": True,
        "head_is_commit": True,
    }
    def asset(lock: b17.AssetLock) -> dict[str, object]:
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

    weights = {"student": asset(b17.STUDENT_LOCK), "teacher": asset(b17.DINO_LOCK)}
    timm_sources = {
        "paths": {name: f"C:/site-packages/timm/{name}" for name in b17.EXPECTED_TIMM_HASHES},
        "sha256": dict(b17.EXPECTED_TIMM_HASHES),
    }
    nodeids = [f"test_node_{index}" for index in range(b17.EXPECTED_FOCUSED_TEST_COUNT)]
    child = {
        "schema_version": 1,
        "guard_root": os.path.abspath(os.fspath(b17.DATASET_ROOT)),
        "test_paths": list(b17.FOCUSED_TEST_PATHS),
        "audit": {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
        "collected_nodeids": nodeids,
        "executed_nodeids": copy.deepcopy(nodeids),
        "pytest_returncode": 0,
        "error": None,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": b17.PROTOCOL_ID,
        "protocol_sha256": b17.PROTOCOL_SHA256,
        "mode": "offline_label_free_stock_first_preflight",
        "created_at_unix": 1.0,
        "git": git,
        "source_hashes": sources,
        "runtime": dict(b17.EXPECTED_RUNTIME),
        "timm_sources": timm_sources,
        "device": {"requested": "cuda", "resolved": "cuda:0", **b17.EXPECTED_CUDA},
        "weights": weights,
        "strict_load": {
            "caller_rng_preserved": True,
            "student": {
                "runtime_class": "timm.models.swiftformer.SwiftFormer",
                "stage_channels": [48, 56, 112, 220],
                "parameters": b17.STOCK_PARAMETERS,
                "heads": [[220, 5], [220, 5]],
                "distilled_training": False,
                "nonzero_dropout_or_path": [],
                "strict_load": True,
                "head_reset": {
                    "seed": b17.SEED,
                    "initializer": "timm_trunc_normal_0p02_head_then_head_dist",
                    "bias_zero": True,
                    "state_sha256": b17.EXPECTED_HEAD_STATE_SHA256,
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
            "test_paths": list(b17.FOCUSED_TEST_PATHS),
            "bootstrap_sha256": hashlib.sha256(
                b17._focused_test_bootstrap().encode("utf-8")
            ).hexdigest(),
            "returncode": 0,
            "pytest_returncode": 0,
            "passed_count": b17.EXPECTED_FOCUSED_TEST_COUNT,
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
            "child_guard": child,
            "stdout": f"{b17.EXPECTED_FOCUSED_TEST_COUNT} passed in 1.0s\n",
            "stderr": "",
            "passed": True,
        },
        "stock_deployment": stock,
        "mechanism": _mechanism(),
        "folded_deployment": folded,
        "cuda": _cuda(),
        "isolation": {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
        "offline": {
            "environment": dict(b17.OFFLINE_ENV),
            "cublas_workspace_config": ":4096:8",
        },
        "end_rehash": {
            "git": git,
            "source_hashes": sources,
            "runtime": dict(b17.EXPECTED_RUNTIME),
            "timm_sources": timm_sources,
            "weights": weights,
        },
        "permissions": dict(b17.SUCCESS_PERMISSIONS),
        "authorization": dict(b17.AUTHORIZATION),
        "checks": {name: True for name in b17.TOP_CHECKS},
        "passed": True,
    }
    return payload


def test_locked_sources_and_parser_have_no_data_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    assert b17._sha256(root / "docs/TRKH_PRETRAINED_CLASSF_B17_SURFACEFOLD_XS_PROTOCOL_20260805.md") == b17.PROTOCOL_SHA256
    assert b17._sha256(root / "trkh/models/swiftformer_surfacefold_b17.py") == b17.MODEL_SHA256
    assert b17._sha256(root / "trkh/inference/surfacefold_deployment.py") == b17.DEPLOYMENT_SHA256
    arguments = b17._parse_args(["--output-dir", "runs/preflight_b17_surfacefold_xs_test"])
    assert set(vars(arguments)) == {"output_dir", "student_weight", "dino_weight", "device"}
    assert all("data" not in name and "label" not in name for name in vars(arguments))


def test_universal_source_map_is_unambiguous() -> None:
    assert set(b17._source_paths()) == {
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
    monkeypatch.setattr(b17, "_root", lambda: tmp_path)
    valid = tmp_path / "runs" / f"{b17.OUTPUT_PREFIX}case"
    assert b17._validated_output(valid) == valid
    with pytest.raises(ValueError):
        b17._validated_output(tmp_path / "elsewhere" / f"{b17.OUTPUT_PREFIX}case")
    valid.mkdir()
    with pytest.raises(FileExistsError):
        b17._validated_output(valid)


def test_isolation_guard_blocks_network_and_dataset() -> None:
    with pytest.raises(RuntimeError, match="isolation guard observed"):
        with b17._isolation_guard():
            with pytest.raises(RuntimeError, match="blocked network"):
                socket.getaddrinfo("example.com", 443)
    with pytest.raises(RuntimeError, match="isolation guard observed"):
        with b17._isolation_guard():
            with pytest.raises(RuntimeError, match="blocked dataset"):
                open(b17.DATASET_ROOT / "data.yaml", "rb")


def test_guarded_pytest_blocks_child_dataset_network_and_process(tmp_path: Path) -> None:
    if os.environ.get("TRKH_B17_IN_GUARDED_PYTEST") == "1":
        with pytest.raises(RuntimeError, match="isolation guard observed"):
            with b17._isolation_guard(deny_process=True):
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
        "    open(os.path.join(os.environ['TRKH_B17_CHILD_GUARD_ROOT'], 'sentinel.txt')).read()\n"
        ,
        encoding="utf-8",
    )
    dataset_evidence = b17._run_guarded_pytest(
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
    evidence = b17._run_guarded_pytest(
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
    first_evidence = b17.reset_five_class_heads(first)
    second_evidence = b17.reset_five_class_heads(second)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert first_evidence["state_sha256"] == second_evidence["state_sha256"]
    bundle = b17.build_b17_arms(first)
    assert bundle.rng_preserved
    assert b17._common_state_equal(bundle.control, bundle.candidate)
    assert b17._state_storage_disjoint(bundle.control, bundle.candidate)
    assert [int(model._surfacefold_mode_id) for _, model in bundle.items()] == [0, 1, 2]


def test_relation_only_helper_proves_candidate_and_control_gradient_geometry() -> None:
    base = timm.create_model("swiftformer_xs", pretrained=False, num_classes=5)
    bundle = b17.build_b17_arms(base)
    candidate = b17._relation_only_gradients(bundle.candidate)
    control = b17._relation_only_gradients(bundle.control)
    assert candidate["p"]["nonzero"] and candidate["d"]["nonzero"]
    assert candidate["candidate_centered_d_nonzero"]
    assert control["p"]["nonzero"] and control["d"]["nonzero"]
    assert control["control_d_spatially_equal"]
    assert candidate["teacher_gradient_absent"] and control["teacher_gradient_absent"]


def test_valid_payload_recomputes_all_nested_gates() -> None:
    checks = b17._payload_checks(_payload())
    assert checks and all(checks.values()), checks


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda p: p["permissions"].update(formal_train_permission=False), "permission_exact"),
        (lambda p: p["source_hashes"].update(protocol="0" * 64), "source_hashes_exact"),
        (lambda p: p["stock_deployment"]["onnx"].update(size_bytes=b17.MAX_ONNX_BYTES + 1), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["latency"]["ort_to_onnx_ratio"].update(p95=1.2), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["latency"]["arms"]["ort"]["trials_ms"][0].pop(), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["latency"]["arms"]["ort"]["trials_ms"][0].__setitem__(0, True), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["latency"].update(execution_mode="ORT_PARALLEL"), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["latency"].update(order_rule="forged"), "stock_deployment_exact"),
        (lambda p: p["stock_deployment"]["mobile"].update(log="checker ran"), "stock_deployment_exact"),
        (lambda p: p["mechanism"]["float64_oracle_max_abs"].update(candidate=1.0e-4), "mechanism_exact"),
        (lambda p: p["mechanism"]["relation_only_gradients"]["candidate"]["p"].update(nonzero=False), "mechanism_exact"),
        (lambda p: p["mechanism"]["total_objective_gradients"]["stock"].update(s3_gradient_nonzero=False), "mechanism_exact"),
        (lambda p: p["folded_deployment"]["arms"]["candidate"].update(topology_sha256="9" * 64), "folded_deployment_exact"),
        (lambda p: p["cuda"]["memory"].update(peak_allocated=b17.MAX_CUDA_ALLOCATED_BYTES + 1), "cuda_exact"),
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
    assert b17._payload_checks(payload)[failed_check] is False


def test_payload_rejects_boolean_numeric_spoof() -> None:
    payload = _payload()
    payload["stock_deployment"]["onnx"]["size_bytes"] = True
    assert b17._payload_checks(payload)["stock_deployment_exact"] is False


def test_atomic_publish_is_canonical_hashed_and_non_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    result = b17._publish(destination, "preflight.json", {"z": 1, "a": True})
    raw = (destination / "preflight.json").read_bytes()
    assert raw == b17._canonical_bytes({"z": 1, "a": True})
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    assert (destination / "preflight.sha256").read_text() == f"{result['sha256']}\n"
    with pytest.raises(FileExistsError):
        b17._publish(destination, "preflight.json", {})


def test_stock_gate_failure_attaches_full_raw_timing_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _deployment(stock=True)
    evidence["latency"] = _latency(10.0, 14.0)
    evidence["checks"]["host_format_ratio"] = False
    evidence["passed"] = False

    class Stock:
        def fold_to_deploy(self):
            return object()

    monkeypatch.setattr(b17, "_synthetic_arrays", lambda: (object(),))
    monkeypatch.setattr(
        b17,
        "_portable_deployment_evidence",
        lambda *args, **kwargs: copy.deepcopy(evidence),
    )
    with pytest.raises(RuntimeError, match="stock deployment gate failed") as caught:
        b17._stock_deployment_contract(Stock())
    attached = caught.value.b17_gate_evidence
    assert attached["gate"] == "stock_deployment_first"
    assert attached["evidence"]["latency"]["arms"]["onnx"]["trials_ms"]
    assert len(attached["evidence"]["latency"]["arms"]["ort"]["trials_ms"]) == 5
    assert all(
        len(row) == 100
        for row in attached["evidence"]["latency"]["arms"]["ort"]["trials_ms"]
    )


def test_failure_artifact_atomically_hashes_attached_gate_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate_evidence = {
        "gate": "stock_deployment_first",
        "evidence": {"latency": _latency(10.0, 14.0), "passed": False},
    }
    failure_root = tmp_path / "failed"
    monkeypatch.setattr(b17, "_failure_path", lambda requested: failure_root)
    error = RuntimeError("closed")
    error.b17_gate_evidence = gate_evidence
    try:
        raise error
    except RuntimeError as caught:
        result = b17._write_failure(tmp_path / "requested", "stock", caught)
    payload = json.loads((failure_root / "failure.json").read_text(encoding="utf-8"))
    assert payload["gate_evidence"] == gate_evidence
    assert payload["gate_evidence_sha256"] == hashlib.sha256(
        b17._canonical_bytes(gate_evidence)
    ).hexdigest()
    assert payload["permissions"] == b17.NO_ACCESS_PERMISSIONS
    assert payload["passed"] is False
    assert result["sha256"] == hashlib.sha256(
        (failure_root / "failure.json").read_bytes()
    ).hexdigest()


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"x":1,"x":2}', object_pairs_hook=b17._no_duplicate_keys)


def test_build_preflight_is_stock_first_and_candidate_never_runs_after_stock_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    order: list[str] = []

    @contextmanager
    def isolation():
        yield {"dataset_attempts": [], "network_attempts": []}

    git = {"head": "a" * 40, "branch": b17.EXPECTED_BRANCH, "status": "", "clean": True, "head_is_commit": True}
    monkeypatch.setattr(b17, "_validated_output", lambda value: output)
    monkeypatch.setattr(b17, "_offline_environment", lambda: {"environment": b17.OFFLINE_ENV, "cublas_workspace_config": ":4096:8"})
    monkeypatch.setattr(b17, "_isolation_guard", isolation)
    monkeypatch.setattr(b17, "_git_contract", lambda: git)
    monkeypatch.setattr(b17, "_source_hashes", lambda: {**{name: "1" * 64 for name in b17._source_paths()}, **b17.SOURCE_LOCKS})
    monkeypatch.setattr(b17, "_runtime_contract", lambda: dict(b17.EXPECTED_RUNTIME))
    monkeypatch.setattr(b17, "_timm_contract", lambda: {"paths": {}, "sha256": b17.EXPECTED_TIMM_HASHES})
    monkeypatch.setattr(b17, "_device_contract", lambda value: (object(), {"requested": "cuda"}))
    monkeypatch.setattr(b17, "_resolve_asset", lambda lock, path: (tmp_path / lock.filename, {"offline_cache_only": True}))
    monkeypatch.setattr(b17, "_focused_tests", lambda: {"passed": True})
    monkeypatch.setattr(b17, "_strict_load_models", lambda a, c: (object(), object(), object(), {"student": {"strict_load": True}, "teacher": {"strict_load": True}}))

    def stock(model):
        del model
        order.append("stock")
        raise RuntimeError("stock closed")

    monkeypatch.setattr(b17, "_stock_deployment_contract", stock)
    monkeypatch.setattr(
        b17,
        "build_active_arms_after_stock",
        lambda base, stock: order.append("construct_active"),
    )
    monkeypatch.setattr(b17, "_mechanism_contract", lambda bundle: order.append("mechanism"))
    monkeypatch.setattr(b17, "_write_failure", lambda *args: {"artifact": "failure"})
    args = argparse.Namespace(output_dir=output, student_weight=None, dino_weight=None, device="cuda")
    with pytest.raises(RuntimeError, match="stock closed"):
        b17.build_preflight(args)
    assert order == ["stock"]


def test_main_turns_keyboard_interrupt_into_nonzero_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(b17, "build_preflight", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert b17.main(["--output-dir", "runs/preflight_b17_surfacefold_xs_interrupt"]) == 1
    assert '"passed": false' in capsys.readouterr().err

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import socket
from pathlib import Path

import numpy as np
import onnx
import pytest
import timm
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from trkh.tools import audit_swiftformer_surface_b16_preflight as b16


def _asset(lock: b16.AssetLock) -> dict[str, object]:
    return {
        "role": lock.role,
        "repo_id": lock.repo_id,
        "revision": lock.revision,
        "filename": lock.filename,
        "path": f"C:/cache/{lock.role}/{lock.filename}",
        "bytes": lock.byte_count,
        "sha256": lock.sha256,
        "license": lock.license,
        "offline_cache_only": True,
    }


def _export(*, fp32: bool, byte_count: int, marker: str) -> dict[str, object]:
    common: dict[str, object] = {
        "bytes": byte_count,
        "sha256": marker * 64,
        "opset_imports": [{"domain": "ai.onnx", "version": 17}],
        "local_function_count": 0,
        "training_info_count": 0,
        "graph_count": 1,
        "initializer_count": 10,
        "domains": [""],
        "operators": ["Conv", "Gemm"],
        "input_name": "images",
        "input_shape": [1, 3, 224, 224],
        "output_name": "logits",
        "output_shape": [1, 5],
        "forbidden_identifiers": [],
    }
    if fp32:
        common["parity"] = {"max_abs": 1.0e-7, "argmax_mismatches": 0}
    else:
        common["qdq"] = {
            "format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "calibration_samples": 32,
            "quantize_linear": 20,
            "dequantize_linear": 30,
        }
        common["coverage"] = {
            "eligible_initializers": 10,
            "eligible_weight_elements": 1000,
            "covered_initializers": 9,
            "qdq_covered_weight_elements": 950,
            "ratio": 0.95,
        }
        common["comparison"] = {
            "provider": "CPUExecutionProvider",
            "samples": 16,
            "finite": True,
            "output_minimum": -1.0,
            "output_maximum": 1.0,
            "max_abs_logit_error": 0.02,
            "argmax_matches": 15,
            "argmax_mismatches": 1,
        }
    return common


def _latency_arm(value: float) -> dict[str, object]:
    trials = [[value for _ in range(b16.ORT_ITERATIONS)] for _ in range(b16.ORT_TRIALS)]
    return {"trials_ms": trials, "median_ms": value, "p95_ms": value}


def _latency_precision() -> dict[str, object]:
    arms = {
        "stock": _latency_arm(5.0),
        "control": _latency_arm(5.2),
        "candidate": _latency_arm(5.3),
    }
    return {
        "arms": arms,
        "ratios": {
            "control": {"median": 1.04, "p95": 1.04},
            "candidate": {"median": 1.06, "p95": 1.06},
        },
    }


def _deployment() -> dict[str, object]:
    stock_fp32 = 12 * 1024**2
    return {
        "fp32": {
            "stock": _export(fp32=True, byte_count=stock_fp32, marker="1"),
            "control": _export(
                fp32=True, byte_count=stock_fp32 + 100_000, marker="2"
            ),
            "candidate": _export(
                fp32=True, byte_count=stock_fp32 + 100_000, marker="3"
            ),
        },
        "int8": {
            "stock": _export(fp32=False, byte_count=4 * 1024**2, marker="4"),
            "control": _export(fp32=False, byte_count=4 * 1024**2, marker="5"),
            "candidate": _export(fp32=False, byte_count=4 * 1024**2, marker="6"),
        },
        "latency": {
            "settings": {
                "provider": "CPUExecutionProvider",
                "threads": 4,
                "inter_op_threads": 1,
                "execution_mode": "ORT_SEQUENTIAL",
                "batch": 1,
                "warmups": 15,
                "trials": 5,
                "iterations_per_trial": 100,
                "quantile_method": "linear",
            },
            "precisions": {
                "fp32": _latency_precision(),
                "int8": _latency_precision(),
            },
            "passed": True,
        },
        "checks": {name: True for name in b16.DEPLOYMENT_CHECK_NAMES},
        "passed": True,
    }


def _synthetic() -> dict[str, object]:
    return {
        "formal_data_or_labels_read": False,
        "parameters": {
            "stock": b16.STOCK_PARAMETERS,
            "branch": b16.BRANCH_PARAMETERS,
            "control": b16.CANDIDATE_PARAMETERS,
            "candidate": b16.CANDIDATE_PARAMETERS,
        },
        "state": {
            "candidate_control_equal": True,
            "candidate_control_storage_disjoint": True,
            "caller_rng_preserved": True,
        },
        "formula": {
            "branch_max_abs": 0.0,
            "control_max_abs": 0.0,
            "branch_spatial_variance": 0.01,
        },
        "relation": {
            "sample_shape": [2, 364],
            "student_14x14_shape": [2, 364],
            "numpy_max_abs": 1.0e-7,
            "minimum": -0.8,
            "maximum": 0.9,
            "teacher_loss_formula_max_abs": 0.0,
            "horizontal_flip_max_abs": 1.0e-7,
        },
        "shared_geometry": {
            "raw_shape": [277, 319, 3],
            "crop": {
                "top": 19,
                "left": 27,
                "height": 231,
                "width": 257,
                "horizontal_flip": True,
            },
            "student_shape": [3, 224, 224],
            "teacher_shape": [3, 256, 256],
            "student_reference_max_abs": 0.0,
            "teacher_reference_max_abs": 0.0,
        },
        "checks": {name: True for name in b16.SYNTHETIC_CHECK_NAMES},
        "passed": True,
    }


def _cuda() -> dict[str, object]:
    gradients = {
        name: {
            "all_parameter_gradients_finite": True,
            "head_nonzero": True,
            "head_dist_nonzero": True,
            "branch_nonzero": None if name == "stock" else True,
            "s2_nonzero": None if name == "stock" else True,
        }
        for name in b16.ARMS
    }
    optimizer_states = {
        name: {
            "trainable_parameters": b16.EXPECTED_TRAINABLE_PARAMETER_TENSORS[name],
            "state_parameters": b16.EXPECTED_TRAINABLE_PARAMETER_TENSORS[name],
            "state_tensors": 3 * b16.EXPECTED_TRAINABLE_PARAMETER_TENSORS[name],
            "state_keys": ["exp_avg", "exp_avg_sq", "step"],
            "every_trainable_parameter_present": True,
            "keys_exact": True,
            "shapes_exact": True,
            "finite": True,
            "steps_exact": True,
        }
        for name in b16.ARMS
    }
    return {
        "settings": {
            "batch": 16,
            "student_precision": "bfloat16_autocast",
            "teacher_precision": "float32_no_grad",
            "arm_schedule": ["stock", "control", "candidate"],
            "optimizer": "AdamW_real_step",
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
        },
        "teacher_calls": 1,
        "losses": {name: 1.0 for name in b16.ARMS},
        "gradients": gradients,
        "optimizer_states": optimizer_states,
        "memory": {
            "baseline_allocated": 1_000,
            "baseline_reserved": 2_000,
            "peak_allocated": 3_000,
            "peak_reserved": 4_000,
            "limit_allocated": b16.MAX_CUDA_ALLOCATED_BYTES,
        },
        "checks": {name: True for name in b16.CUDA_CHECK_NAMES},
        "passed": True,
    }


def _valid_payload() -> dict[str, object]:
    source_hashes = {name: "a" * 64 for name in b16._source_paths()}
    source_hashes["protocol"] = b16.PROTOCOL_SHA256
    payload = {
        "schema_version": 1,
        "protocol_id": b16.PROTOCOL_ID,
        "mode": "synthetic_preflight_no_dataset",
        "created_at_unix": 1.0,
        "git": {
            "head": "b" * 40,
            "branch": b16.EXPECTED_BRANCH,
            "status": "",
            "tracked_worktree_clean": True,
            "head_is_commit": True,
        },
        "source_hashes": source_hashes,
        "runtime": copy.deepcopy(b16.EXPECTED_RUNTIME),
        "timm_sources": {
            "files": {
                name: f"C:/site-packages/timm/models/{name}"
                for name in b16.EXPECTED_TIMM_SOURCE_HASHES
            },
            "sha256": copy.deepcopy(b16.EXPECTED_TIMM_SOURCE_HASHES),
        },
        "device": {
            "requested": "cuda",
            "resolved": "cuda:0",
            "index": 0,
            "name": b16.EXPECTED_CUDA_NAME,
            "total_memory": b16.EXPECTED_CUDA_MEMORY_BYTES,
            "capability": list(b16.EXPECTED_CUDA_CAPABILITY),
            "bf16_supported": True,
        },
        "weights": {
            "student": _asset(b16.STUDENT_LOCK),
            "teacher": _asset(b16.DINO_LOCK),
        },
        "strict_load": {
            "student_load": {
                "runtime_class": "timm.models.swiftformer.SwiftFormer",
                "stages": [48, 56, 112, 220],
                "head": [220, 5],
                "head_dist": [220, 5],
                "parameters_5class": b16.STOCK_PARAMETERS,
                "strict_load": True,
                "head_reset": {
                    "seed": b16.SEED,
                    "initializer": "timm.trunc_normal_std_0p02_sequential_head_then_head_dist",
                    "bias_zero": True,
                    "state_sha256": "7" * 64,
                },
            },
            "teacher_load": {
                "runtime_class": "timm.models.eva.Eva",
                "features": 384,
                "prefix_tokens": 5,
                "grid": [16, 16],
                "all_requires_grad_false": True,
                "forbidden_classf_state_keys": [],
                "strict_load": True,
            },
        },
        "focused_tests": {
            "command": b16._focused_test_command(),
            "returncode": 0,
            "output": "99 passed",
        },
        "synthetic": _synthetic(),
        "cuda_fit": _cuda(),
        "deployment": _deployment(),
        "offline": {
            "environment": copy.deepcopy(b16.EXPECTED_OFFLINE_ENVIRONMENT),
            "socket_connect_denied": True,
            "network_attempts": 0,
        },
        "permissions": copy.deepcopy(b16.PERMISSIONS),
        "checks": {name: True for name in b16.TOP_CHECK_NAMES},
        "passed": True,
    }
    payload["end_rehash"] = {
        "git": copy.deepcopy(payload["git"]),
        "source_hashes": copy.deepcopy(payload["source_hashes"]),
        "timm_sources": copy.deepcopy(payload["timm_sources"]),
        "weights": copy.deepcopy(payload["weights"]),
    }
    return payload


def test_protocol_hash_and_parser_have_no_dataset_surface() -> None:
    protocol = (
        b16._repository_root()
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B16_SWIFTSURFACE_XS_PROTOCOL_20260805.md"
    )
    assert b16._sha256(protocol) == b16.PROTOCOL_SHA256
    args = b16._parse_args(["--output-dir", "runs/preflight_b16_swiftsurface_xs_x"])
    assert set(vars(args)) == {"output_dir", "student_weight", "dino_weight", "device"}
    assert "quantize_mobile_onnx_qdq" not in inspect.getsource(b16)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert {
        name: os.environ[name] for name in b16.EXPECTED_OFFLINE_ENVIRONMENT
    } == b16.EXPECTED_OFFLINE_ENVIRONMENT
    assert "formal_train_runner" in b16._source_paths()
    assert "formal_train_runner_test" in b16._source_paths()


def test_output_scope_is_fresh_direct_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(b16, "_repository_root", lambda: tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    valid = runs / f"{b16.PREFLIGHT_OUTPUT_PREFIX}abc_r1"
    assert b16._validated_output_path(valid) == valid.resolve()
    with pytest.raises(ValueError, match="direct child"):
        b16._validated_output_path(runs / "nested" / valid.name)
    valid.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        b16._validated_output_path(valid)


def test_arm_builder_preserves_rng_and_copies_equal_disjoint_state() -> None:
    torch.manual_seed(731)
    stock = timm.create_model(
        "swiftformer_xs.dist_in1k",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    before = torch.random.get_rng_state().clone()
    bundle = b16.build_b16_arms(stock, seed=733)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert bundle.rng_preserved is True
    assert b16._state_equal(bundle.candidate, bundle.control)
    assert b16._state_storage_disjoint(bundle.candidate, bundle.control)
    assert sum(p.numel() for p in bundle.stock.parameters()) == b16.STOCK_PARAMETERS
    assert sum(p.numel() for p in bundle.candidate.parameters()) == b16.CANDIDATE_PARAMETERS


def test_synthetic_contract_executes_14x14_relation_and_shared_crop_oracle() -> None:
    stock = timm.create_model(
        "swiftformer_xs.dist_in1k",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    payload = b16._synthetic_mechanism_contract(b16.build_b16_arms(stock))
    assert payload["relation"]["sample_shape"] == [2, 364]
    assert payload["relation"]["horizontal_flip_max_abs"] <= 1e-6
    assert payload["shared_geometry"]["student_shape"] == [3, 224, 224]
    assert payload["shared_geometry"]["teacher_shape"] == [3, 256, 256]
    assert payload["checks"]["shared_raw_crop_flip_views_exact"] is True


def test_head_reset_is_truncated_normal_zero_bias_reproducible_and_rng_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(739)
    source = timm.create_model(
        "swiftformer_xs.dist_in1k", pretrained=False, num_classes=1000
    )
    left = copy.deepcopy(source)
    right = copy.deepcopy(source)
    cuda_seed_calls: list[int] = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: cuda_seed_calls.append(int(seed)),
    )
    before = torch.random.get_rng_state().clone()
    left_contract = b16.reset_swiftformer_five_class_heads(left, seed=b16.SEED)
    after = torch.random.get_rng_state()
    right_contract = b16.reset_swiftformer_five_class_heads(right, seed=b16.SEED)

    assert torch.equal(before, after)
    assert left_contract == right_contract
    assert b16._hex64(left_contract["state_sha256"])
    assert left_contract["bias_zero"] is True
    assert torch.equal(left.head.weight, right.head.weight)
    assert torch.equal(left.head_dist.weight, right.head_dist.weight)
    assert torch.count_nonzero(left.head.bias) == 0
    assert torch.count_nonzero(left.head_dist.bias) == 0
    assert 0.015 < float(left.head.weight.std()) < 0.025
    assert 0.015 < float(left.head_dist.weight.std()) < 0.025
    assert cuda_seed_calls == []
    assert "torch.manual_seed(" not in inspect.getsource(b16.build_b16_arms)
    assert "torch.manual_seed(" not in inspect.getsource(
        b16.reset_swiftformer_five_class_heads
    )


def test_offline_asset_resolver_is_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(b"locked")
    digest = b16._sha256(path)
    lock = b16.AssetLock("tiny", "repo/tiny", "revision", "tiny.safetensors", 6, digest, "MIT")
    resolved, payload = b16._resolve_offline_asset(lock, path)
    assert resolved == path.resolve()
    assert payload["sha256"] == digest
    assert payload["offline_cache_only"] is False
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset changed"):
        b16._resolve_offline_asset(lock, path)


def _save_weight_graphs(fp32_path: Path, qdq_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w1 = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="w1")
    w2 = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="w2")
    fp32_graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["a"], name="matmul_w1"),
            helper.make_node("MatMul", ["a", "w2"], ["y"], name="matmul_w2"),
        ],
        "fp32",
        [x],
        [y],
        [w1, w2],
    )
    onnx.save(helper.make_model(fp32_graph), fp32_path)

    q1 = numpy_helper.from_array(np.eye(2, dtype=np.int8), name="q1")
    scale = numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="zero")
    qdq_graph = helper.make_graph(
        [
            helper.make_node(
                "DequantizeLinear",
                ["q1", "scale", "zero"],
                ["w1_dq"],
                name="dequantize_w1",
            ),
            helper.make_node(
                "MatMul", ["x", "w1_dq"], ["a"], name="matmul_w1"
            ),
            helper.make_node("MatMul", ["a", "w2"], ["y"], name="matmul_w2"),
        ],
        "qdq",
        [x],
        [y],
        [q1, scale, zero, w2],
    )
    onnx.save(helper.make_model(qdq_graph), qdq_path)


def test_qdq_coverage_counts_weight_elements_not_operator_rows(tmp_path: Path) -> None:
    fp32 = tmp_path / "fp32.onnx"
    qdq = tmp_path / "qdq.onnx"
    _save_weight_graphs(fp32, qdq)
    coverage = b16.qdq_weight_element_coverage(fp32, qdq)
    assert coverage == {
        "eligible_initializers": 2,
        "eligible_weight_elements": 8,
        "covered_initializers": 1,
        "qdq_covered_weight_elements": 4,
        "ratio": 0.5,
    }


def test_qdq_coverage_ignores_unrelated_quantized_weight_elements(
    tmp_path: Path,
) -> None:
    fp32_path = tmp_path / "fp32_unrelated.onnx"
    qdq_path = tmp_path / "qdq_unrelated.onnx"
    x2 = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 2])
    x10 = helper.make_tensor_value_info("x10", TensorProto.FLOAT, [1, 10])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 2])
    y10 = helper.make_tensor_value_info("y10", TensorProto.FLOAT, [1, 10])
    unrelated = helper.make_tensor_value_info("unrelated", TensorProto.FLOAT, [1, 10])
    w4 = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="w4")
    w100 = numpy_helper.from_array(np.eye(10, dtype=np.float32), name="w100")
    fp32_graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x2", "w4"], ["y2"], name="eligible_w4"),
            helper.make_node(
                "MatMul", ["x10", "w100"], ["y10"], name="eligible_w100"
            ),
        ],
        "fp32_unrelated",
        [x2, x10],
        [y2, y10],
        [w4, w100],
    )
    onnx.save(helper.make_model(fp32_graph), fp32_path)

    q100 = numpy_helper.from_array(np.eye(10, dtype=np.int8), name="q100")
    scale = numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="zero")
    qdq_graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x2", "w4"], ["y2"], name="eligible_w4"),
            helper.make_node(
                "MatMul", ["x10", "w100"], ["y10"], name="eligible_w100"
            ),
            helper.make_node(
                "DequantizeLinear",
                ["q100", "scale", "zero"],
                ["q100_dq"],
                name="dequantize_unrelated",
            ),
            helper.make_node(
                "MatMul",
                ["x10", "q100_dq"],
                ["unrelated"],
                name="unrelated_quantized_target",
            ),
        ],
        "qdq_unrelated",
        [x2, x10],
        [y2, y10, unrelated],
        [w4, w100, q100, scale, zero],
    )
    onnx.save(helper.make_model(qdq_graph), qdq_path)

    coverage = b16.qdq_weight_element_coverage(fp32_path, qdq_path)
    assert coverage == {
        "eligible_initializers": 2,
        "eligible_weight_elements": 104,
        "covered_initializers": 0,
        "qdq_covered_weight_elements": 0,
        "ratio": 0.0,
    }


def test_qdq_coverage_scopes_duplicate_nested_node_and_initializer_names(
    tmp_path: Path,
) -> None:
    fp32_path = tmp_path / "fp32_nested.onnx"
    qdq_path = tmp_path / "qdq_nested.onnx"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    condition = numpy_helper.from_array(np.asarray(True), name="condition")

    def branch(*, quantized: bool, graph_name: str):
        output = helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [1, 2])
        if not quantized:
            weight = numpy_helper.from_array(
                np.eye(2, dtype=np.float32), name="shared_weight"
            )
            nodes = [
                helper.make_node(
                    "MatMul",
                    ["x", "shared_weight"],
                    ["branch_y"],
                    name="shared_target",
                )
            ]
            initializers = [weight]
        else:
            weight = numpy_helper.from_array(
                np.eye(2, dtype=np.int8), name="shared_weight_quantized"
            )
            scale = numpy_helper.from_array(
                np.asarray(0.1, dtype=np.float32), name="shared_scale"
            )
            zero = numpy_helper.from_array(
                np.asarray(0, dtype=np.int8), name="shared_zero"
            )
            nodes = [
                helper.make_node(
                    "DequantizeLinear",
                    ["shared_weight_quantized", "shared_scale", "shared_zero"],
                    ["shared_weight_dq"],
                    name="shared_dequantize",
                ),
                helper.make_node(
                    "MatMul",
                    ["x", "shared_weight_dq"],
                    ["branch_y"],
                    name="shared_target",
                ),
            ]
            initializers = [weight, scale, zero]
        return helper.make_graph(nodes, graph_name, [], [output], initializers)

    fp32_if = helper.make_node(
        "If",
        ["condition"],
        ["y"],
        name="scoped_if",
        then_branch=branch(quantized=False, graph_name="then_fp32"),
        else_branch=branch(quantized=False, graph_name="else_fp32"),
    )
    onnx.save(
        helper.make_model(
            helper.make_graph([fp32_if], "fp32_nested", [x], [y], [condition])
        ),
        fp32_path,
    )
    qdq_if = helper.make_node(
        "If",
        ["condition"],
        ["y"],
        name="scoped_if",
        then_branch=branch(quantized=True, graph_name="then_qdq"),
        else_branch=branch(quantized=False, graph_name="else_qdq"),
    )
    onnx.save(
        helper.make_model(
            helper.make_graph([qdq_if], "qdq_nested", [x], [y], [condition])
        ),
        qdq_path,
    )

    assert b16.qdq_weight_element_coverage(fp32_path, qdq_path) == {
        "eligible_initializers": 2,
        "eligible_weight_elements": 8,
        "covered_initializers": 1,
        "qdq_covered_weight_elements": 4,
        "ratio": 0.5,
    }


def test_onnx_contract_rejects_wrong_opset(tmp_path: Path) -> None:
    path = tmp_path / "wrong_opset.onnx"
    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    weight = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="scale")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["images", "scale"], ["logits"], name="scale")],
        "wrong_opset",
        [images],
        [logits],
        [weight],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
    )
    with pytest.raises(ValueError, match="opset 17"):
        b16._onnx_graph_contract(path)


def test_onnx_contract_scans_forbidden_nested_initializers(tmp_path: Path) -> None:
    path = tmp_path / "nested_forbidden.onnx"
    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    condition = numpy_helper.from_array(np.asarray(True), name="condition")

    def identity_branch(name: str, *, forbidden: bool):
        output = helper.make_tensor_value_info(
            "branch_logits", TensorProto.FLOAT, [1, 3, 224, 224]
        )
        initializers = []
        if forbidden:
            initializers.append(
                numpy_helper.from_array(
                    np.asarray(1.0, dtype=np.float32),
                    name="teacher_relation_nested_weight",
                )
            )
        return helper.make_graph(
            [
                helper.make_node(
                    "Identity", ["images"], ["branch_logits"], name=f"{name}_identity"
                )
            ],
            name,
            [],
            [output],
            initializers,
        )

    choose = helper.make_node(
        "If",
        ["condition"],
        ["logits"],
        name="choose_branch",
        then_branch=identity_branch("then_branch", forbidden=True),
        else_branch=identity_branch("else_branch", forbidden=False),
    )
    graph = helper.make_graph(
        [choose], "nested_forbidden", [images], [logits], [condition]
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path
    )
    with pytest.raises(ValueError, match="forbidden recursive identifiers"):
        b16._onnx_graph_contract(path)


def test_onnx_contract_scans_sparse_tensor_attribute_identifiers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sparse_forbidden.onnx"
    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    values = numpy_helper.from_array(
        np.asarray([1.0], dtype=np.float32), name="teacher_relation_hidden"
    )
    indices = numpy_helper.from_array(
        np.asarray([[0, 0]], dtype=np.int64), name="sparse_indices"
    )
    sparse = helper.make_sparse_tensor(values, indices, [1, 1])
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["unused_sparse"], name="sparse", sparse_value=sparse
            ),
            helper.make_node("Identity", ["images"], ["logits"], name="identity"),
        ],
        "sparse_forbidden",
        [images],
        [logits],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path
    )
    with pytest.raises(ValueError, match="forbidden recursive identifiers"):
        b16._onnx_graph_contract(path)


def test_onnx_contract_rejects_local_function_baggage(tmp_path: Path) -> None:
    path = tmp_path / "local_function.onnx"
    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["logits"], name="identity")],
        "local_function",
        [images],
        [logits],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    function = helper.make_function(
        "custom",
        "UnusedHidden",
        ["X"],
        ["Y"],
        [
            helper.make_node(
                "Identity", ["X"], ["Y"], name="teacher_relation_hidden"
            )
        ],
        [helper.make_opsetid("", 17)],
    )
    model.functions.append(function)
    onnx.save(model, path)
    with pytest.raises(ValueError, match="FunctionProto"):
        b16._onnx_graph_contract(path)


@pytest.mark.parametrize(
    ("field", "forged"),
    (("eligible_initializers", "10"), ("covered_initializers", -999)),
)
def test_payload_rejects_forged_qdq_initializer_counts(
    field: str, forged: object
) -> None:
    payload = _valid_payload()
    payload["deployment"]["int8"]["candidate"]["coverage"][field] = forged
    assert not b16._preflight_payload_structure_checks(payload)["deployment_exact"]


@pytest.mark.parametrize(
    ("eligible_initializers", "covered_initializers", "eligible", "covered"),
    ((10, 0, 1000, 950), (10, 10, 1000, 950), (1001, 10, 1000, 950)),
)
def test_payload_rejects_impossible_qdq_initializer_element_aggregates(
    eligible_initializers: int,
    covered_initializers: int,
    eligible: int,
    covered: int,
) -> None:
    payload = _valid_payload()
    coverage = payload["deployment"]["int8"]["candidate"]["coverage"]
    coverage.update(
        {
            "eligible_initializers": eligible_initializers,
            "covered_initializers": covered_initializers,
            "eligible_weight_elements": eligible,
            "qdq_covered_weight_elements": covered,
            "ratio": covered / eligible,
        }
    )
    assert not b16._preflight_payload_structure_checks(payload)["deployment_exact"]


def test_relation_oracle_uses_actual_14x14_all_edge_vector_and_flip() -> None:
    features = torch.randn(2, 7, 14, 14, generator=torch.Generator().manual_seed(91))
    relation = b16.cosine_neighbor_relation_field(features)
    numpy_relation = b16._numpy_relation(features.numpy())
    flipped = b16.cosine_neighbor_relation_field(features.flip(-1))
    expected_flip = b16._horizontal_flip_relation_expected(relation, 14, 14)
    assert relation.shape == (2, 364)
    assert np.allclose(relation.numpy(), numpy_relation, rtol=1e-6, atol=1e-6)
    assert torch.allclose(flipped, expected_flip, rtol=1e-6, atol=1e-6)


class _GradientProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.surface_branch = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 3)
        self.head_dist = nn.Linear(4, 3)

    def forward(self, inputs: torch.Tensor):
        s2 = self.surface_branch(inputs)
        s2.retain_grad()
        return self.head(s2) + self.head_dist(s2), s2


def test_gradient_and_adamw_state_contracts_are_role_and_parameter_exact() -> None:
    model = _GradientProbe()
    optimizer = torch.optim.AdamW(model.parameters())
    logits, s2 = model(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    logits.square().mean().backward()
    gradients = b16._gradient_contract(model, s2, require_branch=True)
    assert gradients == {
        "all_parameter_gradients_finite": True,
        "head_nonzero": True,
        "head_dist_nonzero": True,
        "branch_nonzero": True,
        "s2_nonzero": True,
    }
    optimizer.step()
    state = b16._optimizer_state_contract(model, optimizer)
    assert state["trainable_parameters"] == len(list(model.parameters()))
    assert state["state_parameters"] == len(list(model.parameters()))
    assert state["state_tensors"] == 3 * len(list(model.parameters()))
    assert state["state_keys"] == ["exp_avg", "exp_avg_sq", "step"]
    assert all(
        state[key] is True
        for key in (
            "every_trainable_parameter_present",
            "keys_exact",
            "shapes_exact",
            "finite",
            "steps_exact",
        )
    )

    model.head_dist.weight.grad.zero_()
    model.head_dist.bias.grad.zero_()
    assert b16._gradient_contract(model, s2, require_branch=True)[
        "head_dist_nonzero"
    ] is False


def test_calibration_bytes_follow_locked_float32_recipe() -> None:
    records = b16._synthetic_calibration()
    assert len(records) == 32
    rng = np.random.default_rng(b16.SEED)
    pixels = rng.random((32, 3, 224, 224), dtype=np.float32)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    expected = np.ascontiguousarray((pixels - mean) / std, dtype=np.float32)
    observed = np.concatenate([record["images"] for record in records], axis=0)
    assert observed.dtype == np.float32
    assert observed.flags.c_contiguous
    assert observed.tobytes() == expected.tobytes()


def test_valid_payload_recomputes_every_nested_gate() -> None:
    checks = b16._preflight_payload_structure_checks(_valid_payload())
    assert checks
    assert all(checks.values()), checks


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda payload: payload["deployment"]["latency"]["precisions"]["fp32"]["ratios"]["candidate"].__setitem__("p95", 1.01),
            "deployment_exact",
        ),
        (
            lambda payload: payload["deployment"]["int8"]["candidate"]["coverage"].__setitem__("ratio", 0.99),
            "deployment_exact",
        ),
        (
            lambda payload: payload["cuda_fit"]["memory"].__setitem__("peak_allocated", b16.MAX_CUDA_ALLOCATED_BYTES + 1),
            "cuda_exact",
        ),
        (
            lambda payload: payload["cuda_fit"]["gradients"]["candidate"].__setitem__("branch_nonzero", False),
            "cuda_exact",
        ),
        (
            lambda payload: payload["cuda_fit"]["gradients"]["stock"].__setitem__("s2_nonzero", True),
            "cuda_exact",
        ),
        (
            lambda payload: payload["cuda_fit"]["optimizer_states"]["control"].__setitem__("state_parameters", 1),
            "cuda_exact",
        ),
        (
            lambda payload: payload["synthetic"]["checks"].pop("relation_bounds"),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["synthetic"]["relation"].__setitem__("sample_shape", [2, 31]),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["synthetic"]["relation"].__setitem__("horizontal_flip_max_abs", 0.1),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["synthetic"]["shared_geometry"]["crop"].__setitem__("left", 28),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["deployment"]["fp32"]["stock"].__setitem__("opset_imports", [{"domain": "ai.onnx", "version": 18}]),
            "deployment_exact",
        ),
        (
            lambda payload: payload["offline"].__setitem__("network_attempts", 1),
            "offline_exact",
        ),
        (
            lambda payload: payload["end_rehash"]["source_hashes"].__setitem__("runner", "f" * 64),
            "end_rehash_exact",
        ),
        (
            lambda payload: payload["permissions"].__setitem__("test_constructed", True),
            "permissions_exact",
        ),
    ],
)
def test_payload_rejects_tampering_behind_passed_true(mutation, failed_check: str) -> None:
    payload = _valid_payload()
    mutation(payload)
    assert payload["passed"] is True
    assert not b16._preflight_payload_structure_checks(payload)[failed_check]


def test_payload_rejects_raw_latency_and_qdq_type_tampering() -> None:
    latency = _valid_payload()
    latency["deployment"]["latency"]["precisions"]["int8"]["arms"]["candidate"]["trials_ms"][0] = [
        11.0 for _ in range(b16.ORT_ITERATIONS)
    ]
    assert not b16._preflight_payload_structure_checks(latency)["deployment_exact"]

    quantized = _valid_payload()
    quantized["deployment"]["int8"]["stock"]["qdq"]["activation_type"] = "QUInt8"
    assert not b16._preflight_payload_structure_checks(quantized)["deployment_exact"]

    for field, forged in (("bytes", "4194304"), ("bytes", True)):
        typed = _valid_payload()
        typed["deployment"]["int8"]["stock"][field] = forged
        assert not b16._preflight_payload_structure_checks(typed)["deployment_exact"]

    qdq_count = _valid_payload()
    qdq_count["deployment"]["int8"]["stock"]["qdq"]["quantize_linear"] = True
    assert not b16._preflight_payload_structure_checks(qdq_count)["deployment_exact"]


def test_nested_check_maps_reject_empty_false_and_extra() -> None:
    for section, failed in (
        ("synthetic", "synthetic_exact"),
        ("cuda_fit", "cuda_exact"),
        ("deployment", "deployment_exact"),
    ):
        for replacement in ({}, {"forged": True}, {"forged": False}):
            payload = _valid_payload()
            payload[section]["checks"] = replacement
            assert not b16._preflight_payload_structure_checks(payload)[failed]


def test_offline_guard_blocks_socket_and_records_attempt() -> None:
    attempts: list[str] | None = None
    with pytest.raises(RuntimeError, match="blocked network access"):
        with b16._offline_network_guard() as observed:
            attempts = observed
            socket.create_connection(("example.invalid", 443))
    assert attempts
    assert b16._offline_environment_contract()["network_attempts"] == 0

    datagram = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        datagram_attempts: list[str] | None = None
        with pytest.raises(RuntimeError, match="blocked network access"):
            with b16._offline_network_guard() as observed:
                datagram_attempts = observed
                datagram.sendto(b"x", ("127.0.0.1", 9))
        assert datagram_attempts
    finally:
        datagram.close()


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda payload: payload.__setitem__("schema_version", True), "identity_exact"),
        (lambda payload: payload["device"].__setitem__("index", False), "device_exact"),
        (
            lambda payload: payload["deployment"]["fp32"]["stock"]["parity"].__setitem__("argmax_mismatches", False),
            "deployment_exact",
        ),
        (
            lambda payload: payload["deployment"]["int8"]["stock"]["comparison"].__setitem__("output_minimum", False),
            "deployment_exact",
        ),
        (
            lambda payload: payload["synthetic"]["relation"].__setitem__("minimum", False),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["synthetic"]["formula"].__setitem__("branch_max_abs", False),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["deployment"]["latency"]["settings"].__setitem__("inter_op_threads", True),
            "deployment_exact",
        ),
        (
            lambda payload: payload["focused_tests"].__setitem__("returncode", False),
            "focused_tests_exact",
        ),
        (
            lambda payload: payload["deployment"]["fp32"]["stock"].__setitem__("input_shape", [True, 3, 224, 224]),
            "deployment_exact",
        ),
        (
            lambda payload: payload["synthetic"]["shared_geometry"]["crop"].__setitem__("horizontal_flip", 1),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["synthetic"]["state"].__setitem__("caller_rng_preserved", 1),
            "synthetic_exact",
        ),
        (
            lambda payload: payload["permissions"].__setitem__("network_used", 0),
            "permissions_exact",
        ),
        (
            lambda payload: payload["strict_load"]["teacher_load"].__setitem__("strict_load", 1),
            "strict_load_exact",
        ),
        (
            lambda payload: payload["end_rehash"]["weights"]["student"].__setitem__("offline_cache_only", 1),
            "end_rehash_exact",
        ),
    ],
)
def test_payload_rejects_boolean_number_spoofs(mutation, failed_check: str) -> None:
    payload = _valid_payload()
    mutation(payload)
    assert not b16._preflight_payload_structure_checks(payload)[failed_check]


def test_end_rehash_rejects_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload()
    monkeypatch.setattr(b16, "_git_contract", lambda: copy.deepcopy(payload["git"]))
    drifted_sources = copy.deepcopy(payload["source_hashes"])
    drifted_sources["runner"] = "f" * 64
    monkeypatch.setattr(b16, "_source_hashes", lambda: drifted_sources)
    monkeypatch.setattr(
        b16,
        "_timm_source_contract",
        lambda: copy.deepcopy(payload["timm_sources"]),
    )

    def asset(lock, explicit):
        del explicit
        return Path(f"C:/{lock.role}.safetensors"), _asset(lock)

    monkeypatch.setattr(b16, "_resolve_offline_asset", asset)
    with pytest.raises(RuntimeError, match="end rehash drifted"):
        b16._end_rehash_contract(
            start_git=payload["git"],
            start_sources=payload["source_hashes"],
            start_timm_sources=payload["timm_sources"],
            start_weights=payload["weights"],
            student_path=Path("student.safetensors"),
            dino_path=Path("teacher.safetensors"),
        )


def test_atomic_directory_promotion_and_failure_artifact_are_hashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(b16, "_repository_root", lambda: tmp_path)
    output = runs / f"{b16.PREFLIGHT_OUTPUT_PREFIX}atomic"
    published = b16._publish_artifact_directory(
        output, "preflight.json", _valid_payload()
    )
    assert output.is_dir()
    assert not output.with_name(output.name + ".partial").exists()
    raw = (output / "preflight.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == published["sha256"]
    assert (output / "preflight.sha256").read_text(encoding="ascii") == (
        published["sha256"] + "\n"
    )

    requested = runs / f"{b16.PREFLIGHT_OUTPUT_PREFIX}interrupted"
    failure = b16._write_failure_artifact(
        requested,
        stage="deployment",
        error=KeyboardInterrupt("interrupted"),
        traceback_text="KeyboardInterrupt: interrupted",
    )
    failure_path = Path(failure["artifact"])
    failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure_path.parent.name.startswith(b16.FAILURE_OUTPUT_PREFIX)
    assert failure_payload["passed"] is False
    assert failure_payload["error_type"] == "KeyboardInterrupt"
    assert hashlib.sha256(failure_path.read_bytes()).hexdigest() == failure["sha256"]
    with pytest.raises(ValueError, match="canonical repo/runs"):
        b16.validate_accepted_preflight(failure_path, failure["sha256"])


def _patch_accepted_environment(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    monkeypatch.setattr(b16, "_git_contract", lambda: copy.deepcopy(payload["git"]))
    monkeypatch.setattr(
        b16, "_source_hashes", lambda: copy.deepcopy(payload["source_hashes"])
    )
    monkeypatch.setattr(
        b16, "_runtime_contract", lambda: copy.deepcopy(payload["runtime"])
    )
    monkeypatch.setattr(
        b16,
        "_timm_source_contract",
        lambda: copy.deepcopy(payload["timm_sources"]),
    )
    monkeypatch.setattr(
        b16,
        "_device_contract",
        lambda device: (torch.device("cpu"), copy.deepcopy(payload["device"])),
    )

    def asset(lock, explicit):
        del explicit
        return Path(f"C:/{lock.role}.safetensors"), copy.deepcopy(
            payload["weights"][lock.role]
        )

    monkeypatch.setattr(b16, "_resolve_offline_asset", asset)


def test_accepted_validator_requires_canonical_direct_child_bytes_and_no_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(b16, "_repository_root", lambda: tmp_path)
    payload = _valid_payload()
    _patch_accepted_environment(monkeypatch, payload)
    output = runs / f"{b16.PREFLIGHT_OUTPUT_PREFIX}accepted"
    published = b16._publish_artifact_directory(output, "preflight.json", payload)
    artifact = Path(published["artifact"])
    accepted = b16.validate_accepted_preflight(artifact, published["sha256"])
    assert accepted["sha256"] == published["sha256"]

    noncanonical = json.dumps(payload, sort_keys=True).encode("utf-8")
    artifact.write_bytes(noncanonical)
    digest = hashlib.sha256(noncanonical).hexdigest()
    artifact.with_name("preflight.sha256").write_bytes(
        (digest + "\n").encode("ascii")
    )
    with pytest.raises(ValueError, match="not canonical"):
        b16.validate_accepted_preflight(artifact, digest)

    duplicate = b'{"schema_version":1,' + b16._canonical_json_bytes(payload)[1:]
    artifact.write_bytes(duplicate)
    digest = hashlib.sha256(duplicate).hexdigest()
    artifact.with_name("preflight.sha256").write_bytes(
        (digest + "\n").encode("ascii")
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        b16.validate_accepted_preflight(artifact, digest)

    outside = tmp_path / f"{b16.PREFLIGHT_OUTPUT_PREFIX}outside"
    outside.mkdir()
    outside_artifact = outside / "preflight.json"
    outside_artifact.write_bytes(b16._canonical_json_bytes(payload))
    outside_digest = hashlib.sha256(outside_artifact.read_bytes()).hexdigest()
    outside_artifact.with_name("preflight.sha256").write_bytes(
        (outside_digest + "\n").encode("ascii")
    )
    with pytest.raises(ValueError, match="canonical repo/runs"):
        b16.validate_accepted_preflight(outside_artifact, outside_digest)


def test_main_reports_interruption_as_nonzero_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = KeyboardInterrupt("stopped")
    setattr(error, "b16_failure_artifact", {"sha256": "a" * 64})

    def interrupted(args):
        del args
        raise error

    monkeypatch.setattr(b16, "build_preflight", interrupted)
    assert b16.main(
        ["--output-dir", "runs/preflight_b16_swiftsurface_xs_interrupted"]
    ) == 1
    reported = json.loads(capsys.readouterr().err)
    assert reported["passed"] is False
    assert reported["error_type"] == "KeyboardInterrupt"
    assert reported["failure_artifact"]["sha256"] == "a" * 64


def test_build_preflight_hashes_keyboard_interrupt_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(b16, "_repository_root", lambda: tmp_path)

    def interrupt():
        raise KeyboardInterrupt("power interruption")

    monkeypatch.setattr(b16, "_git_contract", interrupt)
    requested = runs / f"{b16.PREFLIGHT_OUTPUT_PREFIX}build_interrupt"
    args = b16._parse_args(["--output-dir", str(requested)])
    with pytest.raises(KeyboardInterrupt, match="power interruption") as caught:
        b16.build_preflight(args)
    failure = getattr(caught.value, "b16_failure_artifact")
    failure_path = Path(failure["artifact"])
    assert failure_path.name == "failure.json"
    assert failure_path.parent.name == f"{b16.FAILURE_OUTPUT_PREFIX}build_interrupt"
    assert hashlib.sha256(failure_path.read_bytes()).hexdigest() == failure["sha256"]

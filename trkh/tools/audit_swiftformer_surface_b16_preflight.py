from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from onnx import AttributeProto, GraphProto, NodeProto
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from safetensors.torch import load_file
from timm.layers import trunc_normal_
from torch import Tensor, nn

from trkh.inference.mobile_onnx_quantization import (
    CalibrationDataReader,
    onnx_operator_summary,
)
from trkh.models.swiftformer_surface_b16 import (
    SURFACE_MEAN_CONTROL_B16_MODE,
    SURFACE_OFF_B16_MODE,
    SURFACE_SPATIAL_B16_MODE,
    SwiftFormerSurfaceB16,
    cosine_neighbor_relation_field,
    surface_relation_distillation_loss,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B16_SWIFTSURFACE_XS_20260805"
PROTOCOL_SHA256 = (
    "4184395eb2c599d5bbd8b0cfa0a44cd20a73864b2b509f8602b0c73573c99cde"
)
EXPECTED_BRANCH = "research/pretrained-classf-b1"
SEED = 20260805

STUDENT_MODEL_ID = "timm/swiftformer_xs.dist_in1k"
STUDENT_TIMM_ID = "swiftformer_xs.dist_in1k"
STUDENT_REVISION = "ac0196f198e58c82183a67f5f5c0952421b3e6ac"
STUDENT_FILENAME = "model.safetensors"
STUDENT_BYTES = 13_957_760
STUDENT_SHA256 = (
    "c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c"
)
STUDENT_LICENSE = "apache-2.0"

DINO_MODEL_ID = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_TIMM_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINO_FILENAME = "model.safetensors"
DINO_BYTES = 86_362_376
DINO_SHA256 = (
    "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
)
DINO_LICENSE = "dinov3-license"
DINO_PREFIX_TOKENS = 5
DINO_FEATURE_DIM = 384
DINO_IMAGE_SIZE = 256

STOCK_PARAMETERS = 3_035_570
BRANCH_PARAMETERS = 26_092
CANDIDATE_PARAMETERS = 3_061_662
MAX_PARAMETERS = 3_200_000
MAX_FP32_BYTES = int(13.5 * 1024**2)
MAX_FP32_STOCK_DELTA_BYTES = int(0.25 * 1024**2)
MAX_INT8_BYTES = 6 * 1024**2
MIN_QDQ_WEIGHT_ELEMENT_COVERAGE = 0.90
MAX_ONNX_ABS_ERROR = 1.0e-5
MAX_LATENCY_RATIO = 1.10
MAX_P95_MS = 12.0
MAX_CUDA_ALLOCATED_BYTES = 7 * 1024**3

ORT_THREADS = 4
ORT_WARMUPS = 15
ORT_TRIALS = 5
ORT_ITERATIONS = 100
CALIBRATION_SAMPLES = 32
PREFLIGHT_OUTPUT_PREFIX = "preflight_b16_swiftsurface_xs_"
FAILURE_OUTPUT_PREFIX = "failed_b16_swiftsurface_xs_"

EXPECTED_CUDA_NAME = "NVIDIA GeForce RTX 4060 Laptop GPU"
EXPECTED_CUDA_CAPABILITY = [8, 9]
EXPECTED_CUDA_MEMORY_BYTES = 8_585_216_000
EXPECTED_TRAINABLE_PARAMETER_TENSORS = {
    "stock": 238,
    "control": 243,
    "candidate": 243,
}
EXPECTED_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

EXPECTED_RUNTIME = {
    "python": "3.9.11",
    "numpy": "1.26.4",
    "onnx": "1.19.1",
    "onnxruntime": {"distribution": "onnxruntime-gpu", "version": "1.19.2"},
    "safetensors": "0.7.0",
    "huggingface_hub": "0.36.2",
    "timm": "1.0.27",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "torch_cuda_runtime": "12.4",
    "cudnn": 90100,
}

EXPECTED_TIMM_SOURCE_HASHES = {
    "swiftformer.py": "e7f79c9bfb3750636ada4cd776c9ab9da41d321ba56ada28201a7dc480b654a7",
    "eva.py": "23314ef536d7ce9cc3737e54f424841b46f426e93af651c357b4a75a8b00a12d",
    "_factory.py": "30a6eecdaba750af470cfae3196186fd647052c72338a21c928914aac06163e4",
    "_builder.py": "424afd527cb1780d73c50f2302cd0945f3b1589a860427c45221d4f136e29708",
}

ARMS = ("stock", "control", "candidate")
PRECISIONS = ("fp32", "int8")
TARGET_OPS = frozenset(("Conv", "Gemm", "MatMul"))
FORBIDDEN_EXPORT_TERMS = ("teacher", "dino", "relation", "smoothl1")

PERMISSIONS = {
    "dataset_yaml_read": False,
    "dataset_path_enumerated": False,
    "train_image_read": False,
    "real_label_read": False,
    "prior_prediction_read": False,
    "validation_constructed": False,
    "test_constructed": False,
    "network_used": False,
    "formal_train_permission": False,
}

TOP_CHECK_NAMES = (
    "canonical_branch_clean_committed_head",
    "protocol_and_sources_exact_tracked",
    "runtime_and_timm_sources_exact",
    "offline_assets_exact_strict",
    "focused_tests_passed_without_skip",
    "synthetic_mechanism_contract",
    "cuda_fit_contract",
    "deployment_contract",
    "no_dataset_or_network_access",
    "end_rehash_matches_start",
)

SYNTHETIC_CHECK_NAMES = (
    "parameter_counts_exact",
    "candidate_control_state_equal_disjoint",
    "caller_rng_preserved",
    "branch_off_stock_bit_exact",
    "branch_formula_exact",
    "control_formula_exact",
    "stage_geometry_exact",
    "relation_numpy_parity",
    "relation_bounds",
    "relation_all_edges_exact",
    "teacher_resize_before_relation_exact",
    "horizontal_flip_relation_equivariant",
    "shared_raw_crop_flip_views_exact",
    "teacher_not_registered",
)

CUDA_CHECK_NAMES = (
    "hardware_exact_bf16",
    "batch_and_precision_exact",
    "teacher_called_once_geometry_exact",
    "teacher_gradient_free",
    "all_losses_finite",
    "student_gradients_finite_nonzero",
    "optimizer_states_materialized",
    "peak_allocated_lte_7gib",
)

DEPLOYMENT_CHECK_NAMES = (
    "all_fp32_static_standard_teacher_free",
    "all_fp32_torch_ort_parity",
    "all_fp32_size_gates",
    "all_int8_static_qdq_standard",
    "all_int8_size_gates",
    "all_int8_weight_element_coverage",
    "all_latency_gates",
)


@dataclass(frozen=True)
class AssetLock:
    role: str
    repo_id: str
    revision: str
    filename: str
    byte_count: int
    sha256: str
    license: str


STUDENT_LOCK = AssetLock(
    role="student",
    repo_id=STUDENT_MODEL_ID,
    revision=STUDENT_REVISION,
    filename=STUDENT_FILENAME,
    byte_count=STUDENT_BYTES,
    sha256=STUDENT_SHA256,
    license=STUDENT_LICENSE,
)
DINO_LOCK = AssetLock(
    role="teacher",
    repo_id=DINO_MODEL_ID,
    revision=DINO_REVISION,
    filename=DINO_FILENAME,
    byte_count=DINO_BYTES,
    sha256=DINO_SHA256,
    license=DINO_LICENSE,
)


@dataclass
class ArmBundle:
    stock: nn.Module
    control: SwiftFormerSurfaceB16
    candidate: SwiftFormerSurfaceB16
    ablation: SwiftFormerSurfaceB16
    rng_preserved: bool


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_bytes)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive_finite(value: object) -> bool:
    return (
        _finite_number(value)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_finite(value: object) -> bool:
    return (
        _finite_number(value)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _exact_int_sequence(value: object, expected: Sequence[int]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(expected)
        and all(type(item) is int for item in value)
        and tuple(value) == tuple(expected)
    )


def _strict_json_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_strict_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return bool(
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_strict_json_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _exact_true_checks(value: object, names: Sequence[str]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == set(names)
        and all(value[name] is True for name in names)
    )


def _source_paths() -> dict[str, Path]:
    root = _repository_root()
    return {
        "runner": Path(__file__).resolve(),
        "runner_test": root
        / "tests"
        / "test_audit_swiftformer_surface_b16_preflight.py",
        "protocol": root
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B16_SWIFTSURFACE_XS_PROTOCOL_20260805.md",
        "model": root / "trkh" / "models" / "swiftformer_surface_b16.py",
        "model_test": root / "tests" / "test_swiftformer_surface_b16.py",
        "mobile_quantization": root
        / "trkh"
        / "inference"
        / "mobile_onnx_quantization.py",
        "mobile_quantization_test": root
        / "tests"
        / "test_mobile_onnx_quantization.py",
        "formal_train_runner": root
        / "trkh"
        / "tools"
        / "run_swiftformer_surface_b16_train_oof.py",
        "formal_train_runner_test": root
        / "tests"
        / "test_run_swiftformer_surface_b16_train_oof.py",
    }


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = _source_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"B16 bound source is missing: {name}={path}")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"B16 bound source is not tracked: {name}={path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes["protocol"] != PROTOCOL_SHA256:
        raise RuntimeError("B16 protocol SHA-256 changed")
    return hashes


def _git_contract() -> dict[str, object]:
    root = _repository_root()

    def output(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = output("status", "--porcelain", "--untracked-files=all")
    head = output("rev-parse", "HEAD")
    return {
        "head": head,
        "branch": output("branch", "--show-current"),
        "status": status,
        "tracked_worktree_clean": status == "",
        "head_is_commit": bool(re.fullmatch(r"[0-9a-f]{40}", head)),
    }


def _offline_environment_contract() -> dict[str, object]:
    observed = {
        name: str(os.environ.get(name, ""))
        for name in EXPECTED_OFFLINE_ENVIRONMENT
    }
    if observed != EXPECTED_OFFLINE_ENVIRONMENT:
        raise RuntimeError(f"B16 offline environment drifted: {observed}")
    return {
        "environment": observed,
        "socket_connect_denied": True,
        "network_attempts": 0,
    }


@contextmanager
def _offline_network_guard() -> Iterator[list[str]]:
    attempts: list[str] = []

    def blocked(*args: object, **kwargs: object) -> None:
        del kwargs
        target = repr(args[-1]) if args else "unknown"
        attempts.append(target)
        raise RuntimeError(f"B16 offline preflight blocked network access: {target}")

    socket_methods = ("connect", "connect_ex", "send", "sendall", "sendto")
    optional_methods = tuple(
        name for name in ("sendmsg", "sendmsg_afalg") if hasattr(socket.socket, name)
    )
    with ExitStack() as stack:
        for name in (*socket_methods, *optional_methods):
            stack.enter_context(mock.patch.object(socket.socket, name, blocked))
        stack.enter_context(mock.patch.object(socket, "create_connection", blocked))
        stack.enter_context(mock.patch.object(socket, "getaddrinfo", blocked))
        yield attempts
    if attempts:
        raise RuntimeError(f"B16 offline preflight observed network attempts: {attempts}")


def _runtime_contract() -> dict[str, object]:
    ort_distribution: dict[str, str] | None = None
    for distribution in ("onnxruntime-gpu", "onnxruntime"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        ort_distribution = {"distribution": distribution, "version": version}
        break
    if ort_distribution is None:
        raise importlib.metadata.PackageNotFoundError("ONNX Runtime is not installed")
    observed: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "onnx": importlib.metadata.version("onnx"),
        "onnxruntime": ort_distribution,
        "safetensors": importlib.metadata.version("safetensors"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "timm": importlib.metadata.version("timm"),
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "torch_cuda_runtime": str(torch.version.cuda or ""),
        "cudnn": int(torch.backends.cudnn.version() or 0),
    }
    if observed != EXPECTED_RUNTIME:
        raise RuntimeError(f"B16 runtime drifted: {observed}")
    return observed


def _timm_source_contract() -> dict[str, object]:
    import timm.models._builder as builder
    import timm.models._factory as factory
    import timm.models.eva as eva
    import timm.models.swiftformer as swiftformer

    paths = {
        "swiftformer.py": Path(swiftformer.__file__).resolve(),
        "eva.py": Path(eva.__file__).resolve(),
        "_factory.py": Path(factory.__file__).resolve(),
        "_builder.py": Path(builder.__file__).resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_TIMM_SOURCE_HASHES:
        raise RuntimeError(f"B16 timm runtime source drifted: {hashes}")
    return {
        "files": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
    }


def _device_contract(requested: str) -> tuple[torch.device, dict[str, object]]:
    if requested.strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B16 preflight requires the locked CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "requested": "cuda",
        "resolved": str(device),
        "index": int(device.index or 0),
        "name": str(properties.name),
        "total_memory": int(properties.total_memory),
        "capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    expected = {
        "name": EXPECTED_CUDA_NAME,
        "total_memory": EXPECTED_CUDA_MEMORY_BYTES,
        "capability": EXPECTED_CUDA_CAPABILITY,
        "bf16_supported": True,
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise RuntimeError(f"B16 CUDA hardware drifted: {payload}")
    return device, payload


def _resolve_offline_asset(lock: AssetLock, explicit: Path | None) -> tuple[Path, dict[str, object]]:
    if explicit is None:
        cached = try_to_load_from_cache(
            lock.repo_id,
            lock.filename,
            revision=lock.revision,
        )
        if not isinstance(cached, str):
            raise FileNotFoundError(
                f"Locked offline {lock.role} asset is absent: "
                f"{lock.repo_id}@{lock.revision}/{lock.filename}"
            )
        path = Path(cached).expanduser().resolve()
    else:
        path = explicit.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Locked {lock.role} asset is missing: {path}")
    observed_bytes = int(path.stat().st_size)
    observed_sha = _sha256(path)
    if observed_bytes != lock.byte_count or observed_sha != lock.sha256:
        raise ValueError(
            f"Locked {lock.role} asset changed: bytes={observed_bytes}, sha={observed_sha}"
        )
    return path, {
        "role": lock.role,
        "repo_id": lock.repo_id,
        "revision": lock.revision,
        "filename": lock.filename,
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "license": lock.license,
        "offline_cache_only": explicit is None,
    }


@contextmanager
def _preserve_torch_rng() -> Iterator[None]:
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _rng_digest() -> dict[str, object]:
    payload: dict[str, object] = {
        "cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    if torch.cuda.is_initialized():
        payload["cuda"] = [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def build_b16_arms(base_five_class: nn.Module, *, seed: int = SEED) -> ArmBundle:
    before = _rng_digest()
    with _preserve_torch_rng():
        torch.random.default_generator.manual_seed(int(seed))
        stock = copy.deepcopy(base_five_class)
        template = SwiftFormerSurfaceB16(
            copy.deepcopy(base_five_class), SURFACE_SPATIAL_B16_MODE
        )
        candidate = template
        control = copy.deepcopy(template)
        control.mode = SURFACE_MEAN_CONTROL_B16_MODE
        ablation = copy.deepcopy(template)
        ablation.mode = SURFACE_OFF_B16_MODE
    after = _rng_digest()
    return ArmBundle(
        stock=stock,
        control=control,
        candidate=candidate,
        ablation=ablation,
        rng_preserved=before == after,
    )


def _tensor_state_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def reset_swiftformer_five_class_heads(
    model: nn.Module, *, seed: int = SEED
) -> dict[str, object]:
    existing = (getattr(model, "head", None), getattr(model, "head_dist", None))
    if not all(isinstance(layer, nn.Linear) for layer in existing):
        raise TypeError("B16 head reset requires both pretrained SwiftFormer linear heads")
    if not all(int(layer.in_features) == 220 for layer in existing):
        raise ValueError("B16 head reset requires two 220-channel heads")
    device = existing[0].weight.device
    dtype = existing[0].weight.dtype
    with _preserve_torch_rng():
        torch.random.default_generator.manual_seed(int(seed))
        for name in ("head", "head_dist"):
            layer = nn.Linear(220, 5, device=device, dtype=dtype)
            trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
            setattr(model, name, layer)
    model.num_classes = 5
    values = {
        "head.weight": model.head.weight,
        "head.bias": model.head.bias,
        "head_dist.weight": model.head_dist.weight,
        "head_dist.bias": model.head_dist.bias,
    }
    return {
        "seed": int(seed),
        "initializer": "timm.trunc_normal_std_0p02_sequential_head_then_head_dist",
        "bias_zero": bool(
            torch.count_nonzero(model.head.bias) == 0
            and torch.count_nonzero(model.head_dist.bias) == 0
        ),
        "state_sha256": _tensor_state_sha256(values),
    }


def _strict_load_models(
    student_path: Path,
    dino_path: Path,
) -> tuple[ArmBundle, nn.Module, dict[str, object]]:
    with _preserve_torch_rng():
        torch.random.default_generator.manual_seed(SEED)
        student = timm.create_model(
            STUDENT_TIMM_ID,
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        student_state = load_file(str(student_path), device="cpu")
        student.load_state_dict(student_state, strict=True)
        head_reset = reset_swiftformer_five_class_heads(student, seed=SEED)
        bundle = build_b16_arms(student, seed=SEED)

        teacher = timm.create_model(
            DINO_TIMM_ID,
            pretrained=False,
            num_classes=0,
            img_size=DINO_IMAGE_SIZE,
        )
        teacher_state = load_file(str(dino_path), device="cpu")
        teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    teacher.requires_grad_(False)

    forbidden_teacher_keys = sorted(
        key
        for key in teacher_state
        if any(term in key.casefold() for term in ("class_f", "b9", "classifier_5"))
    )
    observed_student = {
        "runtime_class": f"{type(student).__module__}.{type(student).__name__}",
        "stages": [
            int(stage["num_chs"])
            for stage in getattr(student, "feature_info").get_dicts()
        ],
        "head": [int(student.head.in_features), int(student.head.out_features)],
        "head_dist": [
            int(student.head_dist.in_features),
            int(student.head_dist.out_features),
        ],
        "parameters_5class": sum(parameter.numel() for parameter in student.parameters()),
    }
    expected_student = {
        "runtime_class": "timm.models.swiftformer.SwiftFormer",
        "stages": [48, 56, 112, 220],
        "head": [220, 5],
        "head_dist": [220, 5],
        "parameters_5class": STOCK_PARAMETERS,
    }
    observed_teacher = {
        "runtime_class": f"{type(teacher).__module__}.{type(teacher).__name__}",
        "features": int(getattr(teacher, "num_features", -1)),
        "prefix_tokens": int(getattr(teacher, "num_prefix_tokens", -1)),
        "grid": [int(value) for value in teacher.patch_embed.grid_size],
        "all_requires_grad_false": all(
            not parameter.requires_grad for parameter in teacher.parameters()
        ),
        "forbidden_classf_state_keys": forbidden_teacher_keys,
    }
    expected_teacher = {
        "runtime_class": "timm.models.eva.Eva",
        "features": DINO_FEATURE_DIM,
        "prefix_tokens": DINO_PREFIX_TOKENS,
        "grid": [16, 16],
        "all_requires_grad_false": True,
        "forbidden_classf_state_keys": [],
    }
    if observed_student != expected_student or observed_teacher != expected_teacher:
        raise RuntimeError(
            f"B16 strict model contract failed: student={observed_student}, "
            f"teacher={observed_teacher}"
        )
    return bundle, teacher, {
        "student_load": {
            **observed_student,
            "strict_load": True,
            "head_reset": head_reset,
        },
        "teacher_load": {**observed_teacher, "strict_load": True},
    }


def _state_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return bool(
        left_state.keys() == right_state.keys()
        and all(torch.equal(left_state[name], right_state[name]) for name in left_state)
    )


def _state_storage_disjoint(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return bool(
        left_state.keys() == right_state.keys()
        and all(left_state[name].data_ptr() != right_state[name].data_ptr() for name in left_state)
    )


def _numpy_relation(features: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    norms = np.sqrt(np.sum(features.astype(np.float64) ** 2, axis=1, keepdims=True))
    normalized = features.astype(np.float64) / np.maximum(norms, float(eps))
    right = np.sum(normalized[:, :, :, :-1] * normalized[:, :, :, 1:], axis=1)
    down = np.sum(normalized[:, :, :-1, :] * normalized[:, :, 1:, :], axis=1)
    return np.concatenate((right.reshape(right.shape[0], -1), down.reshape(down.shape[0], -1)), axis=1)


def _horizontal_flip_relation_expected(field: Tensor, height: int, width: int) -> Tensor:
    right_count = height * (width - 1)
    right = field[:, :right_count].reshape(-1, height, width - 1)
    down = field[:, right_count:].reshape(-1, height - 1, width)
    return torch.cat((right.flip(-1).flatten(1), down.flip(-1).flatten(1)), dim=1)


def _synthetic_mechanism_contract(bundle: ArmBundle) -> dict[str, object]:
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    from trkh.tools.run_swiftformer_surface_b16_train_oof import (
        CropParameters,
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_shared_train_views,
    )

    stock_count = sum(parameter.numel() for parameter in bundle.stock.parameters())
    control_count = sum(parameter.numel() for parameter in bundle.control.parameters())
    candidate_count = sum(parameter.numel() for parameter in bundle.candidate.parameters())
    branch_count = sum(
        parameter.numel() for parameter in bundle.candidate.surface_branch.parameters()
    )

    generator = torch.Generator().manual_seed(SEED + 11)
    s2 = torch.randn(2, 112, 14, 14, generator=generator)
    branch = copy.deepcopy(bundle.candidate.surface_branch).eval()
    with torch.inference_mode():
        spatial = branch(s2)
        direct = branch.pointwise(branch.activation(branch.norm(branch.depthwise(s2))))
        direct = direct * branch.layer_scale
        globalized = branch(s2, mean_broadcast=True)
        direct_globalized = spatial.mean(dim=(2, 3), keepdim=True).expand_as(spatial)

    image = torch.randn(1, 3, 64, 64, generator=generator)
    bundle.stock.eval()
    bundle.ablation.eval()
    with torch.inference_mode():
        stock_logits = bundle.stock(image)
        ablation_logits = bundle.ablation(image)

    geometry_image = torch.randn(1, 3, 224, 224, generator=generator)
    bundle.candidate.eval()
    with torch.inference_mode():
        geometry_logits, geometry_s2 = bundle.candidate(
            geometry_image, return_s2=True
        )
        geometry_final = bundle.candidate.forward_features(geometry_image)

    relation_features = torch.randn(2, 7, 14, 14, generator=generator)
    torch_relation = cosine_neighbor_relation_field(relation_features)
    numpy_relation = _numpy_relation(relation_features.numpy())
    flipped_relation = cosine_neighbor_relation_field(relation_features.flip(-1))
    expected_flipped = _horizontal_flip_relation_expected(torch_relation, 14, 14)

    student = torch.randn(2, 11, 14, 14, generator=generator)
    teacher = torch.randn(2, 13, 16, 16, generator=generator)
    student_relation = cosine_neighbor_relation_field(student)
    actual_loss = surface_relation_distillation_loss(student, teacher)
    resized_teacher = F.interpolate(
        teacher.float(), size=(14, 14), mode="bilinear", align_corners=False
    )
    manual_loss = F.smooth_l1_loss(
        cosine_neighbor_relation_field(student),
        cosine_neighbor_relation_field(resized_teacher),
        beta=0.1,
        reduction="mean",
    )

    rows, columns = np.indices((277, 319), dtype=np.int32)
    raw_rgb = np.stack(
        (columns % 251, rows % 251, (3 * columns + 5 * rows) % 251), axis=-1
    ).astype(np.uint8)
    raw_image = Image.fromarray(raw_rgb)
    crop_parameters = CropParameters(19, 27, 231, 257, True)
    shared_student, shared_teacher = build_shared_train_views(
        raw_image, crop_parameters
    )
    reference_crop = raw_image.crop(
        (
            crop_parameters.left,
            crop_parameters.top,
            crop_parameters.left + crop_parameters.width,
            crop_parameters.top + crop_parameters.height,
        )
    ).transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def reference_view(size: int) -> Tensor:
        resized = TF.resize(
            reference_crop,
            [size, size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        tensor = TF.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    expected_student = reference_view(224)
    expected_teacher = reference_view(256)
    student_view_max_abs = float((shared_student - expected_student).abs().max())
    teacher_view_max_abs = float((shared_teacher - expected_teacher).abs().max())

    candidate_control_equal = _state_equal(bundle.candidate, bundle.control)
    candidate_control_disjoint = _state_storage_disjoint(
        bundle.candidate, bundle.control
    )
    teacher_names = [name.casefold() for name, _ in bundle.candidate.named_parameters()]
    teacher_names.extend(
        name.casefold() for name, _ in bundle.candidate.named_buffers()
    )
    checks = {
        "parameter_counts_exact": (
            stock_count == STOCK_PARAMETERS
            and branch_count == BRANCH_PARAMETERS
            and control_count == candidate_count == CANDIDATE_PARAMETERS
            and candidate_count <= MAX_PARAMETERS
        ),
        "candidate_control_state_equal_disjoint": candidate_control_equal
        and candidate_control_disjoint,
        "caller_rng_preserved": bundle.rng_preserved,
        "branch_off_stock_bit_exact": torch.equal(stock_logits, ablation_logits),
        "branch_formula_exact": torch.equal(spatial, direct),
        "control_formula_exact": torch.equal(globalized, direct_globalized),
        "stage_geometry_exact": (
            tuple(geometry_logits.shape) == (1, 5)
            and tuple(geometry_s2.shape) == (1, 112, 14, 14)
            and tuple(geometry_final.shape) == (1, 220, 7, 7)
        ),
        "relation_numpy_parity": bool(
            np.allclose(torch_relation.numpy(), numpy_relation, rtol=1e-6, atol=1e-6)
        ),
        "relation_bounds": bool(
            torch.isfinite(torch_relation).all()
            and float(torch_relation.min()) >= -1.0 - 1e-6
            and float(torch_relation.max()) <= 1.0 + 1e-6
        ),
        "relation_all_edges_exact": tuple(torch_relation.shape) == (2, 364)
        and tuple(student_relation.shape) == (2, 364),
        "teacher_resize_before_relation_exact": torch.equal(actual_loss, manual_loss),
        "horizontal_flip_relation_equivariant": torch.allclose(
            flipped_relation, expected_flipped, rtol=1e-6, atol=1e-6
        ),
        "shared_raw_crop_flip_views_exact": (
            tuple(shared_student.shape) == (3, 224, 224)
            and tuple(shared_teacher.shape) == (3, 256, 256)
            and student_view_max_abs == 0.0
            and teacher_view_max_abs == 0.0
        ),
        "teacher_not_registered": not any(
            any(term in name for term in FORBIDDEN_EXPORT_TERMS)
            for name in teacher_names
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"B16 synthetic mechanism contract failed: {checks}")
    return {
        "formal_data_or_labels_read": False,
        "parameters": {
            "stock": stock_count,
            "branch": branch_count,
            "control": control_count,
            "candidate": candidate_count,
        },
        "state": {
            "candidate_control_equal": candidate_control_equal,
            "candidate_control_storage_disjoint": candidate_control_disjoint,
            "caller_rng_preserved": bundle.rng_preserved,
        },
        "formula": {
            "branch_max_abs": float((spatial - direct).abs().max()),
            "control_max_abs": float((globalized - direct_globalized).abs().max()),
            "branch_spatial_variance": float(spatial.float().var(dim=(2, 3)).mean()),
        },
        "relation": {
            "sample_shape": list(torch_relation.shape),
            "student_14x14_shape": list(student_relation.shape),
            "numpy_max_abs": float(
                np.max(np.abs(torch_relation.numpy() - numpy_relation))
            ),
            "minimum": float(torch_relation.min()),
            "maximum": float(torch_relation.max()),
            "teacher_loss_formula_max_abs": float((actual_loss - manual_loss).abs()),
            "horizontal_flip_max_abs": float(
                (flipped_relation - expected_flipped).abs().max()
            ),
        },
        "shared_geometry": {
            "raw_shape": [277, 319, 3],
            "crop": {
                "top": crop_parameters.top,
                "left": crop_parameters.left,
                "height": crop_parameters.height,
                "width": crop_parameters.width,
                "horizontal_flip": crop_parameters.horizontal_flip,
            },
            "student_shape": list(shared_student.shape),
            "teacher_shape": list(shared_teacher.shape),
            "student_reference_max_abs": student_view_max_abs,
            "teacher_reference_max_abs": teacher_view_max_abs,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _iter_graph_records(
    graph: GraphProto, path: tuple[str, ...] = ("root",)
) -> Iterator[tuple[tuple[str, ...], GraphProto]]:
    yield path, graph
    nested_owners: set[str] = set()
    for node_index, node in enumerate(graph.node):
        nested_attributes = [
            attribute
            for attribute in node.attribute
            if attribute.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS}
        ]
        if not nested_attributes:
            continue
        owner = str(node.name).strip()
        if not owner or owner in nested_owners:
            raise ValueError(
                "B16 recursive ONNX graph owner identity is blank or duplicated: "
                f"graph_path={path!r}, node_index={node_index}, name={node.name!r}"
            )
        nested_owners.add(owner)
        for attribute in nested_attributes:
            if attribute.type == AttributeProto.GRAPH:
                child = f"{owner}:{attribute.name}:0"
                yield from _iter_graph_records(attribute.g, path + (child,))
            elif attribute.type == AttributeProto.GRAPHS:
                for graph_index, nested in enumerate(attribute.graphs):
                    child = f"{owner}:{attribute.name}:{graph_index}"
                    yield from _iter_graph_records(nested, path + (child,))


def _iter_graphs(graph: GraphProto) -> Iterator[GraphProto]:
    for _, nested in _iter_graph_records(graph):
        yield nested


def _iter_nodes(graph: GraphProto) -> Iterator[NodeProto]:
    for nested in _iter_graphs(graph):
        yield from nested.node


def _tensor_shape(value_info: Any) -> list[int]:
    dimensions = value_info.type.tensor_type.shape.dim
    result = []
    for dimension in dimensions:
        if not dimension.HasField("dim_value"):
            raise ValueError("B16 ONNX export must use fully static dimensions")
        result.append(int(dimension.dim_value))
    return result


def _onnx_graph_contract(path: Path) -> dict[str, object]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    if model.functions:
        raise ValueError("B16 ONNX must not contain local FunctionProto baggage")
    if model.training_info:
        raise ValueError("B16 ONNX must not contain training_info baggage")
    opset_imports: dict[str, int] = {}
    for item in model.opset_import:
        domain = str(item.domain) or "ai.onnx"
        if domain in opset_imports:
            raise ValueError(f"B16 ONNX has duplicate opset domain: {domain!r}")
        opset_imports[domain] = int(item.version)
    if opset_imports != {"ai.onnx": 17}:
        raise ValueError(f"B16 ONNX must persist exact ai.onnx opset 17: {opset_imports}")

    graphs = list(_iter_graphs(model.graph))
    nodes = list(_iter_nodes(model.graph))
    domains = sorted({str(node.domain) for node in nodes})
    operators = sorted({str(node.op_type) for node in nodes})
    initializers = [initializer for graph in graphs for initializer in graph.initializer]
    identifiers = [value for graph in graphs for value in (graph.name, graph.doc_string)]
    identifiers.extend(initializer.name for initializer in initializers)
    for graph in graphs:
        identifiers.extend(value.name for value in graph.input)
        identifiers.extend(value.name for value in graph.output)
        identifiers.extend(value.name for value in graph.value_info)
        for sparse in graph.sparse_initializer:
            identifiers.extend((sparse.values.name, sparse.indices.name))
    identifiers.extend(
        value
        for node in nodes
        for value in (node.name, node.op_type, node.domain, node.doc_string)
    )
    identifiers.extend(name for node in nodes for name in (*node.input, *node.output))
    for node in nodes:
        for attribute in node.attribute:
            identifiers.append(attribute.name)
            if attribute.type == AttributeProto.TENSOR:
                identifiers.append(attribute.t.name)
            elif attribute.type == AttributeProto.TENSORS:
                identifiers.extend(tensor.name for tensor in attribute.tensors)
            elif attribute.type == AttributeProto.STRING:
                identifiers.append(attribute.s.decode("utf-8", errors="replace"))
            elif attribute.type == AttributeProto.STRINGS:
                identifiers.extend(
                    value.decode("utf-8", errors="replace")
                    for value in attribute.strings
                )
            elif attribute.type == AttributeProto.SPARSE_TENSOR:
                identifiers.extend(
                    (
                        attribute.sparse_tensor.values.name,
                        attribute.sparse_tensor.indices.name,
                    )
                )
            elif attribute.type == AttributeProto.SPARSE_TENSORS:
                for sparse in attribute.sparse_tensors:
                    identifiers.extend((sparse.values.name, sparse.indices.name))
    searchable = "\n".join(
        identifier for identifier in identifiers if identifier
    ).casefold()
    forbidden = sorted(term for term in FORBIDDEN_EXPORT_TERMS if term in searchable)
    if any(domain not in {"", "ai.onnx"} for domain in domains):
        raise ValueError(f"B16 ONNX contains nonstandard node domains: {domains}")
    if forbidden:
        raise ValueError(f"B16 ONNX contains forbidden recursive identifiers: {forbidden}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("B16 deployment graph must have one input and one output")
    return {
        "opset_imports": [{"domain": "ai.onnx", "version": 17}],
        "local_function_count": 0,
        "training_info_count": 0,
        "graph_count": len(graphs),
        "initializer_count": len(initializers),
        "domains": domains,
        "operators": operators,
        "input_name": str(model.graph.input[0].name),
        "input_shape": _tensor_shape(model.graph.input[0]),
        "output_name": str(model.graph.output[0].name),
        "output_shape": _tensor_shape(model.graph.output[0]),
        "forbidden_identifiers": forbidden,
    }


class _ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)


def _ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"B16 ORT provider drifted: {session.get_providers()}")
    return session


def _export_fp32(model: nn.Module, sample: Tensor, path: Path) -> dict[str, object]:
    export_model = _ExportWrapper(copy.deepcopy(model).cpu().eval()).eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            export_model,
            sample,
            str(path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    graph = _onnx_graph_contract(path)
    with torch.inference_mode():
        torch_output = export_model(sample).numpy()
    session = _ort_session(path)
    ort_output = session.run(None, {"images": sample.numpy()})[0]
    parity = {
        "max_abs": float(np.max(np.abs(torch_output - ort_output))),
        "argmax_mismatches": int(
            np.count_nonzero(torch_output.argmax(axis=1) != ort_output.argmax(axis=1))
        ),
    }
    return {
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        **graph,
        "parity": parity,
    }


def _initializer_specs(
    graph: GraphProto,
) -> dict[tuple[tuple[str, ...], str], tuple[int, int]]:
    result: dict[tuple[tuple[str, ...], str], tuple[int, int]] = {}
    for graph_path, nested in _iter_graph_records(graph):
        for initializer in nested.initializer:
            key = (graph_path, str(initializer.name))
            if not initializer.name or key in result:
                raise ValueError(
                    "B16 ONNX initializer identity is blank or duplicated in one "
                    f"graph scope: path={graph_path!r}, name={initializer.name!r}"
                )
            result[key] = (
                len(initializer.dims),
                int(np.prod(initializer.dims, dtype=np.int64)),
            )
    return result


def _resolve_initializer_spec(
    specs: Mapping[tuple[tuple[str, ...], str], tuple[int, int]],
    graph_path: tuple[str, ...],
    name: str,
) -> tuple[tuple[tuple[str, ...], str] | None, tuple[int, int]]:
    for length in range(len(graph_path), 0, -1):
        key = (graph_path[:length], name)
        if key in specs:
            return key, specs[key]
    return None, (0, 0)


def _weight_input_indices(node: NodeProto) -> tuple[int, ...]:
    if node.op_type in {"Conv", "Gemm"}:
        return (1,)
    if node.op_type == "MatMul":
        return (0, 1)
    return ()


def _target_nodes_by_identity(
    graph: GraphProto, *, graph_label: str
) -> dict[tuple[tuple[str, ...], str], NodeProto]:
    targets: dict[tuple[tuple[str, ...], str], NodeProto] = {}
    for graph_path, nested in _iter_graph_records(graph):
        for node in nested.node:
            if node.op_type not in TARGET_OPS:
                continue
            identity = str(node.name)
            key = (graph_path, identity)
            if not identity.strip():
                raise ValueError(
                    f"B16 {graph_label} target node has blank identity at "
                    f"{graph_path!r}; QDQ coverage cannot be mapped fail-closed"
                )
            if key in targets:
                raise ValueError(
                    f"B16 {graph_label} target node identity is duplicated in "
                    f"one graph scope: path={graph_path!r}, name={identity!r}"
                )
            targets[key] = node
    return targets


def qdq_weight_element_coverage(fp32_path: Path, int8_path: Path) -> dict[str, object]:
    fp32 = onnx.load(str(fp32_path), load_external_data=True)
    int8 = onnx.load(str(int8_path), load_external_data=True)
    fp32_specs = _initializer_specs(fp32.graph)
    int8_specs = _initializer_specs(int8.graph)
    fp32_targets = _target_nodes_by_identity(fp32.graph, graph_label="FP32")
    int8_targets = _target_nodes_by_identity(int8.graph, graph_label="QDQ")

    eligible: dict[
        tuple[tuple[str, ...], str],
        tuple[int, set[tuple[tuple[str, ...], str, str, int]]],
    ] = {}
    for (graph_path, identity), node in fp32_targets.items():
        for index in _weight_input_indices(node):
            if index >= len(node.input):
                continue
            name = node.input[index]
            initializer_key, (rank, elements) = _resolve_initializer_spec(
                fp32_specs, graph_path, name
            )
            if rank >= 2:
                if initializer_key is None:
                    raise RuntimeError("B16 eligible initializer resolution failed")
                if initializer_key not in eligible:
                    eligible[initializer_key] = (elements, set())
                elif eligible[initializer_key][0] != elements:
                    raise ValueError(
                        f"B16 FP32 initializer {name!r} has inconsistent element counts"
                    )
                eligible[initializer_key][1].add(
                    (graph_path, identity, str(node.op_type), index)
                )

    producers: dict[tuple[tuple[str, ...], str], NodeProto] = {}
    for graph_path, nested in _iter_graph_records(int8.graph):
        for node in nested.node:
            for output in node.output:
                if not output:
                    continue
                key = (graph_path, str(output))
                if key in producers:
                    raise ValueError(
                        "B16 QDQ tensor has multiple producers in one graph scope: "
                        f"path={graph_path!r}, name={output!r}"
                    )
                producers[key] = node

    covered: dict[tuple[tuple[str, ...], str], int] = {}
    for initializer_key, (elements, usages) in eligible.items():
        all_usages_covered = bool(usages)
        for graph_path, identity, op_type, index in usages:
            target = int8_targets.get((graph_path, identity))
            if (
                target is None
                or target.op_type != op_type
                or index >= len(target.input)
            ):
                all_usages_covered = False
                break
            producer = producers.get((graph_path, target.input[index]))
            if (
                producer is None
                or producer.op_type != "DequantizeLinear"
                or not producer.input
            ):
                all_usages_covered = False
                break
            quantized_name = producer.input[0]
            _, (rank, quantized_elements) = _resolve_initializer_spec(
                int8_specs, graph_path, quantized_name
            )
            if rank < 2 or quantized_elements != elements:
                all_usages_covered = False
                break
        if all_usages_covered:
            covered[initializer_key] = elements

    eligible_elements = int(sum(elements for elements, _ in eligible.values()))
    covered_elements = int(sum(covered.values()))
    if eligible_elements <= 0:
        raise RuntimeError("B16 FP32 graph has no eligible constant weight elements")
    ratio = float(covered_elements / eligible_elements)
    return {
        "eligible_initializers": len(eligible),
        "eligible_weight_elements": eligible_elements,
        "covered_initializers": len(covered),
        "qdq_covered_weight_elements": covered_elements,
        "ratio": ratio,
    }


def _synthetic_calibration() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    pixels = rng.random(
        (CALIBRATION_SAMPLES, 3, 224, 224), dtype=np.float32
    )
    normalized = np.ascontiguousarray((pixels - mean) / std, dtype=np.float32)
    return [
        {"images": np.ascontiguousarray(normalized[index : index + 1])}
        for index in range(CALIBRATION_SAMPLES)
    ]


def _synthetic_comparison() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED + 1)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    pixels = rng.random((16, 3, 224, 224), dtype=np.float32)
    normalized = np.ascontiguousarray((pixels - mean) / std, dtype=np.float32)
    return [
        {"images": np.ascontiguousarray(normalized[index : index + 1])}
        for index in range(16)
    ]


def _compare_fp32_qdq(
    fp32_path: Path,
    int8_path: Path,
    records: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, object]:
    fp32_session = _ort_session(fp32_path)
    int8_session = _ort_session(int8_path)
    maximum_error = 0.0
    mismatches = 0
    output_minimum = math.inf
    output_maximum = -math.inf
    for record in records:
        feed = {"images": np.ascontiguousarray(record["images"], dtype=np.float32)}
        fp32_output = np.asarray(fp32_session.run(None, feed)[0], dtype=np.float32)
        int8_output = np.asarray(int8_session.run(None, feed)[0], dtype=np.float32)
        if not np.isfinite(fp32_output).all() or not np.isfinite(int8_output).all():
            raise FloatingPointError("B16 FP32/QDQ comparison produced non-finite logits")
        maximum_error = max(
            maximum_error, float(np.max(np.abs(fp32_output - int8_output)))
        )
        mismatches += int(
            np.count_nonzero(
                fp32_output.argmax(axis=1) != int8_output.argmax(axis=1)
            )
        )
        output_minimum = min(output_minimum, float(int8_output.min()))
        output_maximum = max(output_maximum, float(int8_output.max()))
    return {
        "provider": "CPUExecutionProvider",
        "samples": len(records),
        "finite": True,
        "output_minimum": output_minimum,
        "output_maximum": output_maximum,
        "max_abs_logit_error": maximum_error,
        "argmax_matches": len(records) - mismatches,
        "argmax_mismatches": mismatches,
    }


def _quantize_graph(
    fp32_path: Path,
    int8_path: Path,
    calibration: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, object]:
    reader = CalibrationDataReader(
        calibration,
        calibration_split="train",
        provenance={
            "source_kind": "synthetic_rng_no_dataset",
            "dataset_accessed": False,
            "labels_used": False,
            "test_used": False,
            "seed": SEED,
        },
    )
    reader.rewind()
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        use_external_data_format=False,
        extra_options={},
    )
    quantized = onnx.load(str(int8_path), load_external_data=True)
    onnx.checker.check_model(quantized)
    graph = _onnx_graph_contract(int8_path)
    operators = onnx_operator_summary(int8_path)
    coverage = qdq_weight_element_coverage(fp32_path, int8_path)
    comparison = _compare_fp32_qdq(
        fp32_path, int8_path, _synthetic_comparison()
    )
    return {
        "bytes": int(int8_path.stat().st_size),
        "sha256": _sha256(int8_path),
        **graph,
        "qdq": {
            "format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "calibration_samples": CALIBRATION_SAMPLES,
            "quantize_linear": int(operators["quantize_linear"]),
            "dequantize_linear": int(operators["dequantize_linear"]),
        },
        "coverage": coverage,
        "comparison": comparison,
    }


def _timed_ort_call(session: ort.InferenceSession, array: np.ndarray) -> float:
    started = time.perf_counter_ns()
    session.run(None, {"images": array})
    return float((time.perf_counter_ns() - started) / 1e6)


def _benchmark_precision(paths: Mapping[str, Path], array: np.ndarray) -> dict[str, object]:
    if set(paths) != set(ARMS):
        raise ValueError("B16 latency benchmark requires exactly three arms")
    sessions = {name: _ort_session(paths[name]) for name in ARMS}
    for warmup in range(ORT_WARMUPS):
        order = ARMS[warmup % len(ARMS) :] + ARMS[: warmup % len(ARMS)]
        for name in order:
            sessions[name].run(None, {"images": array})

    trials: dict[str, list[list[float]]] = {name: [] for name in ARMS}
    for trial in range(ORT_TRIALS):
        rows = {name: [] for name in ARMS}
        for iteration in range(ORT_ITERATIONS):
            offset = (trial + iteration) % len(ARMS)
            order = ARMS[offset:] + ARMS[:offset]
            for name in order:
                rows[name].append(_timed_ort_call(sessions[name], array))
        for name in ARMS:
            trials[name].append(rows[name])

    arms: dict[str, object] = {}
    for name in ARMS:
        flattened = np.asarray(
            [value for trial in trials[name] for value in trial], dtype=np.float64
        )
        arms[name] = {
            "trials_ms": trials[name],
            "median_ms": float(np.median(flattened)),
            "p95_ms": float(np.quantile(flattened, 0.95, method="linear")),
        }
    ratios = {}
    for name in ("control", "candidate"):
        ratios[name] = {
            "median": float(arms[name]["median_ms"] / arms["stock"]["median_ms"]),
            "p95": float(arms[name]["p95_ms"] / arms["stock"]["p95_ms"]),
        }
    return {"arms": arms, "ratios": ratios}


def _latency_contract(paths: Mapping[str, Mapping[str, Path]], array: np.ndarray) -> dict[str, object]:
    if set(paths) != set(PRECISIONS):
        raise ValueError("B16 latency contract requires FP32 and INT8 graphs")
    precisions = {
        precision: _benchmark_precision(paths[precision], array)
        for precision in PRECISIONS
    }
    passed = True
    for precision in PRECISIONS:
        row = precisions[precision]
        passed = passed and all(
            float(row["arms"][name]["p95_ms"]) <= MAX_P95_MS for name in ARMS
        )
        passed = passed and all(
            float(row["ratios"][name][metric]) <= MAX_LATENCY_RATIO
            for name in ("control", "candidate")
            for metric in ("median", "p95")
        )
    return {
        "settings": {
            "provider": "CPUExecutionProvider",
            "threads": ORT_THREADS,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "batch": 1,
            "warmups": ORT_WARMUPS,
            "trials": ORT_TRIALS,
            "iterations_per_trial": ORT_ITERATIONS,
            "quantile_method": "linear",
        },
        "precisions": precisions,
        "passed": bool(passed),
    }


def _deployment_contract(bundle: ArmBundle) -> dict[str, object]:
    models = {
        "stock": bundle.stock.cpu().eval(),
        "control": bundle.control.cpu().eval(),
        "candidate": bundle.candidate.cpu().eval(),
    }
    generator = torch.Generator().manual_seed(SEED + 31)
    sample = torch.randn(1, 3, 224, 224, generator=generator)
    calibration = _synthetic_calibration()
    with tempfile.TemporaryDirectory(prefix="trkh_b16_deploy_") as temporary:
        root = Path(temporary)
        fp32_paths = {name: root / f"{name}_fp32.onnx" for name in ARMS}
        int8_paths = {name: root / f"{name}_int8_qdq.onnx" for name in ARMS}
        fp32 = {
            name: _export_fp32(models[name], sample, fp32_paths[name]) for name in ARMS
        }
        int8 = {
            name: _quantize_graph(fp32_paths[name], int8_paths[name], calibration)
            for name in ARMS
        }
        latency = _latency_contract(
            {"fp32": fp32_paths, "int8": int8_paths}, sample.numpy()
        )

    checks = {
        "all_fp32_static_standard_teacher_free": all(
            value["domains"]
            and all(domain in {"", "ai.onnx"} for domain in value["domains"])
            and value["input_name"] == "images"
            and value["input_shape"] == [1, 3, 224, 224]
            and value["output_name"] == "logits"
            and value["output_shape"] == [1, 5]
            and value["forbidden_identifiers"] == []
            for value in fp32.values()
        ),
        "all_fp32_torch_ort_parity": all(
            float(value["parity"]["max_abs"]) <= MAX_ONNX_ABS_ERROR
            and int(value["parity"]["argmax_mismatches"]) == 0
            for value in fp32.values()
        ),
        "all_fp32_size_gates": all(
            int(value["bytes"]) <= MAX_FP32_BYTES for value in fp32.values()
        )
        and all(
            int(fp32[name]["bytes"]) - int(fp32["stock"]["bytes"])
            <= MAX_FP32_STOCK_DELTA_BYTES
            for name in ("control", "candidate")
        ),
        "all_int8_static_qdq_standard": all(
            value["domains"]
            and all(domain in {"", "ai.onnx"} for domain in value["domains"])
            and value["input_shape"] == [1, 3, 224, 224]
            and value["output_shape"] == [1, 5]
            and value["forbidden_identifiers"] == []
            and value["qdq"]["format"] == "QDQ"
            and value["qdq"]["quantize_linear"] > 0
            and value["qdq"]["dequantize_linear"] > 0
            and value["comparison"]["finite"] is True
            and value["comparison"]["provider"] == "CPUExecutionProvider"
            for value in int8.values()
        ),
        "all_int8_size_gates": all(
            int(value["bytes"]) <= MAX_INT8_BYTES for value in int8.values()
        ),
        "all_int8_weight_element_coverage": all(
            float(value["coverage"]["ratio"])
            >= MIN_QDQ_WEIGHT_ELEMENT_COVERAGE
            and int(value["coverage"]["qdq_covered_weight_elements"])
            <= int(value["coverage"]["eligible_weight_elements"])
            for value in int8.values()
        ),
        "all_latency_gates": latency["passed"] is True,
    }
    return {
        "fp32": fp32,
        "int8": int8,
        "latency": latency,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _optimizer_groups(model: nn.Module) -> list[dict[str, object]]:
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.casefold()
        is_task = (
            lower.startswith(("head.", "head_dist."))
            or lower.startswith("surface_branch.")
            or lower.startswith(("backbone.head.", "backbone.head_dist."))
        )
        lr = 3.0e-4 if is_task else 3.0e-5
        no_decay = (
            parameter.ndim <= 1
            or lower.endswith("bias")
            or "norm" in lower
            or "layer_scale" in lower
        )
        weight_decay = 0.0 if no_decay else 0.05
        groups.setdefault((lr, weight_decay), []).append(parameter)
    return [
        {"params": parameters, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), parameters in sorted(groups.items())
    ]


def _gradient_contract(
    model: nn.Module, s2: Tensor | None, *, require_branch: bool
) -> dict[str, object]:
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )

    def role_nonzero(role: str) -> bool:
        selected = []
        for name, parameter in named_trainable:
            lower = name.casefold()
            if role == "head":
                matched = lower.startswith("head.") or lower.startswith(
                    "backbone.head."
                )
            elif role == "head_dist":
                matched = lower.startswith("head_dist.") or lower.startswith(
                    "backbone.head_dist."
                )
            elif role == "branch":
                matched = lower.startswith("surface_branch.")
            else:
                raise ValueError(f"Unknown B16 gradient role: {role}")
            if matched:
                selected.append(parameter)
        return bool(
            selected
            and any(
                parameter.grad is not None
                and int(torch.count_nonzero(parameter.grad)) > 0
                for parameter in selected
            )
        )

    s2_nonzero: bool | None = None
    branch_nonzero: bool | None = None
    if require_branch:
        branch_nonzero = role_nonzero("branch")
        s2_nonzero = bool(
            s2 is not None
            and s2.grad is not None
            and torch.isfinite(s2.grad).all()
            and int(torch.count_nonzero(s2.grad)) > 0
        )
    elif s2 is not None:
        raise ValueError("B16 stock gradient contract must not receive S2")
    return {
        "all_parameter_gradients_finite": finite,
        "head_nonzero": role_nonzero("head"),
        "head_dist_nonzero": role_nonzero("head_dist"),
        "branch_nonzero": branch_nonzero,
        "s2_nonzero": s2_nonzero,
    }


def _optimizer_state_contract(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, object]:
    named = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_parameters = set(named.values())
    observed_parameters = set(optimizer.state)
    keys_exact = True
    shapes_exact = True
    finite = True
    steps_exact = True
    tensor_count = 0
    for parameter in expected_parameters:
        state = optimizer.state.get(parameter, {})
        keys_exact = keys_exact and set(state) == {"step", "exp_avg", "exp_avg_sq"}
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            shapes_exact = False
            continue
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        tensor_count += sum(torch.is_tensor(value) for value in state.values())
        finite = bool(
            finite
            and all(
                torch.is_tensor(value) and bool(torch.isfinite(value).all())
                for value in state.values()
            )
        )
        steps_exact = bool(
            steps_exact
            and torch.is_tensor(step)
            and step.numel() == 1
            and float(step.detach().cpu()) == 1.0
        )
        shapes_exact = bool(
            shapes_exact
            and torch.is_tensor(step)
            and tuple(step.shape) == ()
            and torch.is_tensor(exp_avg)
            and tuple(exp_avg.shape) == tuple(parameter.shape)
            and torch.is_tensor(exp_avg_sq)
            and tuple(exp_avg_sq.shape) == tuple(parameter.shape)
        )
    return {
        "trainable_parameters": len(expected_parameters),
        "state_parameters": len(observed_parameters),
        "state_tensors": int(tensor_count),
        "state_keys": ["exp_avg", "exp_avg_sq", "step"],
        "every_trainable_parameter_present": observed_parameters
        == expected_parameters,
        "keys_exact": bool(keys_exact),
        "shapes_exact": bool(shapes_exact),
        "finite": bool(finite),
        "steps_exact": bool(steps_exact),
    }


def _cuda_fit_contract(
    bundle: ArmBundle,
    teacher: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("B16 deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    previous_tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_tf32_cudnn = bool(torch.backends.cudnn.allow_tf32)
    previous_deterministic = bool(torch.are_deterministic_algorithms_enabled())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    models = {
        "stock": bundle.stock.to(device).train(),
        "control": bundle.control.to(device).train(),
        "candidate": bundle.candidate.to(device).train(),
    }
    teacher = teacher.to(device).eval()
    optimizers = {
        name: torch.optim.AdamW(_optimizer_groups(model))
        for name, model in models.items()
    }
    losses: dict[str, float] = {}
    gradients: dict[str, dict[str, object]] = {}
    optimizer_states: dict[str, dict[str, object]] = {}
    teacher_calls = 0
    raw_teacher_map: Tensor | None = None
    try:
        generator = torch.Generator().manual_seed(SEED + 41)
        base = torch.rand(16, 3, 256, 256, generator=generator)
        student_images = F.interpolate(
            base, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
        )
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        student_images = ((student_images - mean) / std).to(device)
        teacher_images = ((base - mean) / std).to(device)
        targets = (torch.arange(16, device=device) % 5).long()

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            tokens = teacher.forward_features(teacher_images.float())
            teacher_calls += 1
            if not isinstance(tokens, Tensor) or tuple(tokens.shape) != (
                16,
                261,
                DINO_FEATURE_DIM,
            ):
                raise RuntimeError(f"B16 DINO token geometry changed: {getattr(tokens, 'shape', None)}")
            raw_teacher_map = (
                tokens[:, DINO_PREFIX_TOKENS:]
                .reshape(16, 16, 16, DINO_FEATURE_DIM)
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
            )
        del tokens

        for name in ARMS:
            model = models[name]
            optimizer = optimizers[name]
            s2: Tensor | None = None
            relation: Tensor | None = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if name == "stock":
                    logits = model(student_images)
                    loss = F.cross_entropy(logits.float(), targets)
                else:
                    logits, s2 = model(student_images, return_s2=True)
                    s2.retain_grad()
                    relation = surface_relation_distillation_loss(s2, raw_teacher_map)
                    loss = F.cross_entropy(logits.float(), targets) + 0.10 * relation
            loss.backward()
            gradients[name] = _gradient_contract(
                model, s2, require_branch=name != "stock"
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            losses[name] = float(loss.detach())
            optimizer_states[name] = _optimizer_state_contract(model, optimizer)
            optimizer.zero_grad(set_to_none=True)
            del logits, loss, s2, relation

        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        teacher_gradient_free = all(
            parameter.grad is None for parameter in teacher.parameters()
        )
        checks = {
            "hardware_exact_bf16": bool(torch.cuda.is_bf16_supported()),
            "batch_and_precision_exact": tuple(student_images.shape) == (16, 3, 224, 224),
            "teacher_called_once_geometry_exact": teacher_calls == 1
            and raw_teacher_map is not None
            and tuple(raw_teacher_map.shape) == (16, 384, 16, 16),
            "teacher_gradient_free": teacher_gradient_free,
            "all_losses_finite": all(math.isfinite(value) for value in losses.values()),
            "student_gradients_finite_nonzero": all(
                gradients[name]["all_parameter_gradients_finite"] is True
                and gradients[name]["head_nonzero"] is True
                and gradients[name]["head_dist_nonzero"] is True
                and (
                    gradients[name]["branch_nonzero"] is True
                    and gradients[name]["s2_nonzero"] is True
                    if name != "stock"
                    else gradients[name]["branch_nonzero"] is None
                    and gradients[name]["s2_nonzero"] is None
                )
                for name in ARMS
            ),
            "optimizer_states_materialized": all(
                optimizer_states[name]["trainable_parameters"]
                == EXPECTED_TRAINABLE_PARAMETER_TENSORS[name]
                and optimizer_states[name]["state_parameters"]
                == EXPECTED_TRAINABLE_PARAMETER_TENSORS[name]
                and optimizer_states[name]["state_tensors"]
                == 3 * EXPECTED_TRAINABLE_PARAMETER_TENSORS[name]
                and optimizer_states[name]["every_trainable_parameter_present"]
                is True
                and optimizer_states[name]["keys_exact"] is True
                and optimizer_states[name]["shapes_exact"] is True
                and optimizer_states[name]["finite"] is True
                and optimizer_states[name]["steps_exact"] is True
                for name in ARMS
            ),
            "peak_allocated_lte_7gib": peak_allocated <= MAX_CUDA_ALLOCATED_BYTES,
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
            "teacher_calls": teacher_calls,
            "losses": losses,
            "gradients": gradients,
            "optimizer_states": optimizer_states,
            "memory": {
                "baseline_allocated": baseline_allocated,
                "baseline_reserved": baseline_reserved,
                "peak_allocated": peak_allocated,
                "peak_reserved": peak_reserved,
                "limit_allocated": MAX_CUDA_ALLOCATED_BYTES,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
        torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn
        torch.use_deterministic_algorithms(previous_deterministic)
        del optimizers, models, teacher, raw_teacher_map
        gc.collect()
        torch.cuda.empty_cache()


def _focused_test_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "tests/test_swiftformer_surface_b16.py",
        "tests/test_audit_swiftformer_surface_b16_preflight.py",
        "tests/test_mobile_onnx_quantization.py",
        "tests/test_run_swiftformer_surface_b16_train_oof.py",
    ]


def _run_focused_tests() -> dict[str, object]:
    command = _focused_test_command()
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        command,
        cwd=_repository_root(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "skipped" in output.casefold():
        raise RuntimeError(f"B16 focused tests failed or skipped:\n{output}")
    return {"command": command, "returncode": result.returncode, "output": output}


def _valid_export_payload(value: object, *, fp32: bool) -> bool:
    common_keys = {
        "bytes",
        "sha256",
        "opset_imports",
        "local_function_count",
        "training_info_count",
        "graph_count",
        "initializer_count",
        "domains",
        "operators",
        "input_name",
        "input_shape",
        "output_name",
        "output_shape",
        "forbidden_identifiers",
    }
    expected_keys = common_keys | (
        {"parity"} if fp32 else {"qdq", "coverage", "comparison"}
    )
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        return False
    common = bool(
        type(value.get("bytes")) is int
        and value["bytes"] > 0
        and _hex64(value.get("sha256"))
        and value.get("opset_imports")
        == [{"domain": "ai.onnx", "version": 17}]
        and type(value.get("local_function_count")) is int
        and value["local_function_count"] == 0
        and type(value.get("training_info_count")) is int
        and value["training_info_count"] == 0
        and type(value.get("graph_count")) is int
        and value["graph_count"] >= 1
        and type(value.get("initializer_count")) is int
        and value["initializer_count"] > 0
        and isinstance(value.get("domains"), list)
        and bool(value["domains"])
        and all(domain in {"", "ai.onnx"} for domain in value["domains"])
        and isinstance(value.get("operators"), list)
        and bool(value["operators"])
        and value.get("input_name") == "images"
        and _exact_int_sequence(value.get("input_shape"), (1, 3, 224, 224))
        and value.get("output_name") == "logits"
        and _exact_int_sequence(value.get("output_shape"), (1, 5))
        and value.get("forbidden_identifiers") == []
    )
    if not common:
        return False
    if fp32:
        parity = value.get("parity")
        return bool(
            isinstance(parity, Mapping)
            and set(parity) == {"max_abs", "argmax_mismatches"}
            and _nonnegative_finite(parity.get("max_abs"))
            and float(parity["max_abs"]) <= MAX_ONNX_ABS_ERROR
            and type(parity.get("argmax_mismatches")) is int
            and parity["argmax_mismatches"] == 0
        )
    qdq = value.get("qdq")
    coverage = value.get("coverage")
    if not isinstance(qdq, Mapping) or set(qdq) != {
        "format",
        "activation_type",
        "weight_type",
        "per_channel",
        "calibration_samples",
        "quantize_linear",
        "dequantize_linear",
    }:
        return False
    if not (
        qdq.get("format") == "QDQ"
        and qdq.get("activation_type") == "QInt8"
        and qdq.get("weight_type") == "QInt8"
        and qdq.get("per_channel") is True
        and qdq.get("calibration_samples") == CALIBRATION_SAMPLES
        and type(qdq.get("quantize_linear")) is int
        and qdq["quantize_linear"] > 0
        and type(qdq.get("dequantize_linear")) is int
        and qdq["dequantize_linear"] > 0
    ):
        return False
    if not isinstance(coverage, Mapping) or set(coverage) != {
        "eligible_initializers",
        "eligible_weight_elements",
        "covered_initializers",
        "qdq_covered_weight_elements",
        "ratio",
    }:
        return False
    eligible_initializers = coverage.get("eligible_initializers")
    covered_initializers = coverage.get("covered_initializers")
    eligible = coverage.get("eligible_weight_elements")
    covered = coverage.get("qdq_covered_weight_elements")
    if not (
        type(eligible_initializers) is int
        and eligible_initializers > 0
        and type(covered_initializers) is int
        and 0 <= covered_initializers <= eligible_initializers
        and type(eligible) is int
        and eligible > 0
        and type(covered) is int
        and 0 <= covered <= eligible
        and eligible_initializers <= eligible
        and covered_initializers <= covered
        and (covered_initializers == 0) == (covered == 0)
        and (covered_initializers == eligible_initializers)
        == (covered == eligible)
        and (
            covered in {0, eligible}
            or 0 < covered_initializers < eligible_initializers
        )
        and _nonnegative_finite(coverage.get("ratio"))
    ):
        return False
    expected_ratio = covered / eligible
    comparison = value.get("comparison")
    return bool(
        np.isclose(float(coverage["ratio"]), expected_ratio, rtol=0.0, atol=1e-12)
        and expected_ratio >= MIN_QDQ_WEIGHT_ELEMENT_COVERAGE
        and isinstance(comparison, Mapping)
        and set(comparison)
        == {
            "provider",
            "samples",
            "finite",
            "output_minimum",
            "output_maximum",
            "max_abs_logit_error",
            "argmax_matches",
            "argmax_mismatches",
        }
        and comparison.get("provider") == "CPUExecutionProvider"
        and comparison.get("samples") == 16
        and comparison.get("finite") is True
        and _finite_number(comparison.get("output_minimum"))
        and math.isfinite(float(comparison["output_minimum"]))
        and _finite_number(comparison.get("output_maximum"))
        and math.isfinite(float(comparison["output_maximum"]))
        and float(comparison["output_minimum"])
        <= float(comparison["output_maximum"])
        and _nonnegative_finite(comparison.get("max_abs_logit_error"))
        and type(comparison.get("argmax_matches")) is int
        and type(comparison.get("argmax_mismatches")) is int
        and comparison["argmax_matches"] >= 0
        and comparison["argmax_mismatches"] >= 0
        and comparison["argmax_matches"] + comparison["argmax_mismatches"] == 16
    )


def _latency_arm_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "trials_ms",
        "median_ms",
        "p95_ms",
    }:
        return False
    trials = value.get("trials_ms")
    if not (
        isinstance(trials, list)
        and len(trials) == ORT_TRIALS
        and all(isinstance(row, list) and len(row) == ORT_ITERATIONS for row in trials)
        and all(_positive_finite(sample) for row in trials for sample in row)
    ):
        return False
    flattened = np.asarray([sample for row in trials for sample in row], dtype=np.float64)
    expected_median = float(np.median(flattened))
    expected_p95 = float(np.quantile(flattened, 0.95, method="linear"))
    return bool(
        _positive_finite(value.get("median_ms"))
        and _positive_finite(value.get("p95_ms"))
        and np.isclose(float(value["median_ms"]), expected_median, rtol=0.0, atol=1e-12)
        and np.isclose(float(value["p95_ms"]), expected_p95, rtol=0.0, atol=1e-12)
        and expected_p95 <= MAX_P95_MS
    )


def _latency_payload_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"settings", "precisions", "passed"}:
        return False
    expected_settings = {
        "provider": "CPUExecutionProvider",
        "threads": ORT_THREADS,
        "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "batch": 1,
        "warmups": ORT_WARMUPS,
        "trials": ORT_TRIALS,
        "iterations_per_trial": ORT_ITERATIONS,
        "quantile_method": "linear",
    }
    settings = value.get("settings")
    precisions = value.get("precisions")
    if (
        settings != expected_settings
        or not isinstance(settings, Mapping)
        or any(
            type(settings.get(name)) is not int
            for name in (
                "threads",
                "inter_op_threads",
                "batch",
                "warmups",
                "trials",
                "iterations_per_trial",
            )
        )
        or value.get("passed") is not True
    ):
        return False
    if not isinstance(precisions, Mapping) or set(precisions) != set(PRECISIONS):
        return False
    for precision in PRECISIONS:
        row = precisions[precision]
        if not isinstance(row, Mapping) or set(row) != {"arms", "ratios"}:
            return False
        arms = row.get("arms")
        ratios = row.get("ratios")
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            return False
        if not all(_latency_arm_contract(arms[name]) for name in ARMS):
            return False
        if not isinstance(ratios, Mapping) or set(ratios) != {"control", "candidate"}:
            return False
        for name in ("control", "candidate"):
            ratio = ratios[name]
            if not isinstance(ratio, Mapping) or set(ratio) != {"median", "p95"}:
                return False
            expected_median = float(arms[name]["median_ms"]) / float(
                arms["stock"]["median_ms"]
            )
            expected_p95 = float(arms[name]["p95_ms"]) / float(
                arms["stock"]["p95_ms"]
            )
            if not (
                _positive_finite(ratio.get("median"))
                and _positive_finite(ratio.get("p95"))
                and np.isclose(float(ratio["median"]), expected_median, rtol=0.0, atol=1e-12)
                and np.isclose(float(ratio["p95"]), expected_p95, rtol=0.0, atol=1e-12)
                and expected_median <= MAX_LATENCY_RATIO
                and expected_p95 <= MAX_LATENCY_RATIO
            ):
                return False
    return True


def _deployment_payload_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "fp32",
        "int8",
        "latency",
        "checks",
        "passed",
    }:
        return False
    fp32 = value.get("fp32")
    int8 = value.get("int8")
    if not isinstance(fp32, Mapping) or set(fp32) != set(ARMS):
        return False
    if not isinstance(int8, Mapping) or set(int8) != set(ARMS):
        return False
    if not all(_valid_export_payload(fp32[name], fp32=True) for name in ARMS):
        return False
    if not all(_valid_export_payload(int8[name], fp32=False) for name in ARMS):
        return False
    if not all(int(fp32[name]["bytes"]) <= MAX_FP32_BYTES for name in ARMS):
        return False
    if not all(
        int(fp32[name]["bytes"]) - int(fp32["stock"]["bytes"])
        <= MAX_FP32_STOCK_DELTA_BYTES
        for name in ("control", "candidate")
    ):
        return False
    if not all(int(int8[name]["bytes"]) <= MAX_INT8_BYTES for name in ARMS):
        return False
    return bool(
        _latency_payload_contract(value.get("latency"))
        and _exact_true_checks(value.get("checks"), DEPLOYMENT_CHECK_NAMES)
        and value.get("passed") is True
    )


def _synthetic_payload_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "formal_data_or_labels_read",
        "parameters",
        "state",
        "formula",
        "relation",
        "shared_geometry",
        "checks",
        "passed",
    }:
        return False
    parameters = value.get("parameters")
    state = value.get("state")
    formula = value.get("formula")
    relation = value.get("relation")
    shared_geometry = value.get("shared_geometry")
    if parameters != {
        "stock": STOCK_PARAMETERS,
        "branch": BRANCH_PARAMETERS,
        "control": CANDIDATE_PARAMETERS,
        "candidate": CANDIDATE_PARAMETERS,
    }:
        return False
    if not (
        isinstance(state, Mapping)
        and set(state)
        == {
            "candidate_control_equal",
            "candidate_control_storage_disjoint",
            "caller_rng_preserved",
        }
        and all(item is True for item in state.values())
    ):
        return False
    if not isinstance(formula, Mapping) or set(formula) != {
        "branch_max_abs",
        "control_max_abs",
        "branch_spatial_variance",
    }:
        return False
    if not (
        _nonnegative_finite(formula.get("branch_max_abs"))
        and formula.get("branch_max_abs") == 0.0
        and _nonnegative_finite(formula.get("control_max_abs"))
        and formula.get("control_max_abs") == 0.0
        and _positive_finite(formula.get("branch_spatial_variance"))
    ):
        return False
    if not isinstance(relation, Mapping) or set(relation) != {
        "sample_shape",
        "student_14x14_shape",
        "numpy_max_abs",
        "minimum",
        "maximum",
        "teacher_loss_formula_max_abs",
        "horizontal_flip_max_abs",
    }:
        return False
    if not (
        _exact_int_sequence(relation.get("sample_shape"), (2, 364))
        and _exact_int_sequence(relation.get("student_14x14_shape"), (2, 364))
        and _nonnegative_finite(relation.get("numpy_max_abs"))
        and float(relation["numpy_max_abs"]) <= 1e-6
        and _finite_number(relation.get("minimum"))
        and math.isfinite(float(relation["minimum"]))
        and float(relation["minimum"]) >= -1.0 - 1e-6
        and _finite_number(relation.get("maximum"))
        and math.isfinite(float(relation["maximum"]))
        and float(relation["maximum"]) <= 1.0 + 1e-6
        and _nonnegative_finite(relation.get("teacher_loss_formula_max_abs"))
        and relation.get("teacher_loss_formula_max_abs") == 0.0
        and _nonnegative_finite(relation.get("horizontal_flip_max_abs"))
        and float(relation["horizontal_flip_max_abs"]) <= 1e-6
    ):
        return False
    if not isinstance(shared_geometry, Mapping) or set(shared_geometry) != {
        "raw_shape",
        "crop",
        "student_shape",
        "teacher_shape",
        "student_reference_max_abs",
        "teacher_reference_max_abs",
    }:
        return False
    if not (
        _exact_int_sequence(shared_geometry.get("raw_shape"), (277, 319, 3))
        and isinstance(shared_geometry.get("crop"), Mapping)
        and set(shared_geometry["crop"])
        == {"top", "left", "height", "width", "horizontal_flip"}
        and type(shared_geometry["crop"].get("top")) is int
        and shared_geometry["crop"]["top"] == 19
        and type(shared_geometry["crop"].get("left")) is int
        and shared_geometry["crop"]["left"] == 27
        and type(shared_geometry["crop"].get("height")) is int
        and shared_geometry["crop"]["height"] == 231
        and type(shared_geometry["crop"].get("width")) is int
        and shared_geometry["crop"]["width"] == 257
        and shared_geometry["crop"].get("horizontal_flip") is True
        and _exact_int_sequence(shared_geometry.get("student_shape"), (3, 224, 224))
        and _exact_int_sequence(shared_geometry.get("teacher_shape"), (3, 256, 256))
        and _nonnegative_finite(shared_geometry.get("student_reference_max_abs"))
        and shared_geometry.get("student_reference_max_abs") == 0.0
        and _nonnegative_finite(shared_geometry.get("teacher_reference_max_abs"))
        and shared_geometry.get("teacher_reference_max_abs") == 0.0
    ):
        return False
    return bool(
        value.get("formal_data_or_labels_read") is False
        and _exact_true_checks(value.get("checks"), SYNTHETIC_CHECK_NAMES)
        and value.get("passed") is True
    )


def _cuda_payload_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "settings",
        "teacher_calls",
        "losses",
        "gradients",
        "optimizer_states",
        "memory",
        "checks",
        "passed",
    }:
        return False
    expected_settings = {
        "batch": 16,
        "student_precision": "bfloat16_autocast",
        "teacher_precision": "float32_no_grad",
        "arm_schedule": ["stock", "control", "candidate"],
        "optimizer": "AdamW_real_step",
        "tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    losses = value.get("losses")
    gradients = value.get("gradients")
    states = value.get("optimizer_states")
    memory = value.get("memory")
    settings = value.get("settings")
    if (
        settings != expected_settings
        or not isinstance(settings, Mapping)
        or type(settings.get("batch")) is not int
        or settings.get("tf32") is not False
        or type(value.get("teacher_calls")) is not int
        or value.get("teacher_calls") != 1
    ):
        return False
    if not isinstance(losses, Mapping) or set(losses) != set(ARMS):
        return False
    if not all(_positive_finite(losses[name]) for name in ARMS):
        return False
    if not isinstance(gradients, Mapping) or set(gradients) != set(ARMS):
        return False
    gradient_keys = {
        "all_parameter_gradients_finite",
        "head_nonzero",
        "head_dist_nonzero",
        "branch_nonzero",
        "s2_nonzero",
    }
    for name in ARMS:
        gradient = gradients[name]
        if not (
            isinstance(gradient, Mapping)
            and set(gradient) == gradient_keys
            and gradient.get("all_parameter_gradients_finite") is True
            and gradient.get("head_nonzero") is True
            and gradient.get("head_dist_nonzero") is True
        ):
            return False
        if name == "stock":
            if not (
                gradient.get("branch_nonzero") is None
                and gradient.get("s2_nonzero") is None
            ):
                return False
        elif not (
            gradient.get("branch_nonzero") is True
            and gradient.get("s2_nonzero") is True
        ):
            return False
    if not isinstance(states, Mapping) or set(states) != set(ARMS):
        return False
    state_keys = {
        "trainable_parameters",
        "state_parameters",
        "state_tensors",
        "state_keys",
        "every_trainable_parameter_present",
        "keys_exact",
        "shapes_exact",
        "finite",
        "steps_exact",
    }
    for name in ARMS:
        state = states[name]
        expected = EXPECTED_TRAINABLE_PARAMETER_TENSORS[name]
        if not (
            isinstance(state, Mapping)
            and set(state) == state_keys
            and type(state.get("trainable_parameters")) is int
            and state["trainable_parameters"] == expected
            and type(state.get("state_parameters")) is int
            and state["state_parameters"] == expected
            and type(state.get("state_tensors")) is int
            and state["state_tensors"] == 3 * expected
            and state.get("state_keys") == ["exp_avg", "exp_avg_sq", "step"]
            and state.get("every_trainable_parameter_present") is True
            and state.get("keys_exact") is True
            and state.get("shapes_exact") is True
            and state.get("finite") is True
            and state.get("steps_exact") is True
        ):
            return False
    if not isinstance(memory, Mapping) or set(memory) != {
        "baseline_allocated",
        "baseline_reserved",
        "peak_allocated",
        "peak_reserved",
        "limit_allocated",
    }:
        return False
    if not all(type(memory[name]) is int and memory[name] >= 0 for name in memory):
        return False
    if not (
        memory["limit_allocated"] == MAX_CUDA_ALLOCATED_BYTES
        and memory["peak_allocated"] <= memory["limit_allocated"]
        and memory["peak_allocated"] >= memory["baseline_allocated"]
        and memory["peak_reserved"] >= memory["baseline_reserved"]
    ):
        return False
    return bool(
        _exact_true_checks(value.get("checks"), CUDA_CHECK_NAMES)
        and value.get("passed") is True
    )


def _asset_payload_contract(value: object, lock: AssetLock) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "role",
        "repo_id",
        "revision",
        "filename",
        "path",
        "bytes",
        "sha256",
        "license",
        "offline_cache_only",
    }:
        return False
    return bool(
        value.get("role") == lock.role
        and value.get("repo_id") == lock.repo_id
        and value.get("revision") == lock.revision
        and value.get("filename") == lock.filename
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and value.get("bytes") == lock.byte_count
        and value.get("sha256") == lock.sha256
        and value.get("license") == lock.license
        and isinstance(value.get("offline_cache_only"), bool)
    )


def _preflight_payload_structure_checks(payload: Mapping[str, object]) -> dict[str, bool]:
    expected_top = {
        "schema_version",
        "protocol_id",
        "mode",
        "created_at_unix",
        "git",
        "source_hashes",
        "runtime",
        "timm_sources",
        "device",
        "weights",
        "strict_load",
        "focused_tests",
        "synthetic",
        "cuda_fit",
        "deployment",
        "offline",
        "end_rehash",
        "permissions",
        "checks",
        "passed",
    }
    git = payload.get("git")
    sources = payload.get("source_hashes")
    timm_sources = payload.get("timm_sources")
    device = payload.get("device")
    weights = payload.get("weights")
    strict_load = payload.get("strict_load")
    focused = payload.get("focused_tests")
    offline = payload.get("offline")
    end_rehash = payload.get("end_rehash")
    checks = payload.get("checks")
    return {
        "top_schema_exact": set(payload) == expected_top,
        "identity_exact": type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("mode") == "synthetic_preflight_no_dataset"
        and _positive_finite(payload.get("created_at_unix")),
        "git_exact": isinstance(git, Mapping)
        and set(git) == {
            "head",
            "branch",
            "status",
            "tracked_worktree_clean",
            "head_is_commit",
        }
        and git.get("branch") == EXPECTED_BRANCH
        and git.get("status") == ""
        and git.get("tracked_worktree_clean") is True
        and git.get("head_is_commit") is True
        and isinstance(git.get("head"), str)
        and re.fullmatch(r"[0-9a-f]{40}", str(git.get("head"))) is not None,
        "sources_exact": isinstance(sources, Mapping)
        and set(sources) == set(_source_paths())
        and all(_hex64(value) for value in sources.values())
        and sources.get("protocol") == PROTOCOL_SHA256,
        "runtime_exact": payload.get("runtime") == EXPECTED_RUNTIME,
        "timm_sources_exact": isinstance(timm_sources, Mapping)
        and set(timm_sources) == {"files", "sha256"}
        and isinstance(timm_sources.get("files"), Mapping)
        and set(timm_sources["files"]) == set(EXPECTED_TIMM_SOURCE_HASHES)
        and all(
            isinstance(path, str) and bool(path)
            for path in timm_sources["files"].values()
        )
        and timm_sources.get("sha256") == EXPECTED_TIMM_SOURCE_HASHES,
        "device_exact": isinstance(device, Mapping)
        and set(device)
        == {
            "requested",
            "resolved",
            "index",
            "name",
            "total_memory",
            "capability",
            "bf16_supported",
        }
        and device.get("requested") == "cuda"
        and device.get("resolved") == "cuda:0"
        and type(device.get("index")) is int
        and device.get("index") == 0
        and device.get("name") == EXPECTED_CUDA_NAME
        and device.get("total_memory") == EXPECTED_CUDA_MEMORY_BYTES
        and _exact_int_sequence(
            device.get("capability"), tuple(EXPECTED_CUDA_CAPABILITY)
        )
        and device.get("bf16_supported") is True,
        "weights_exact": isinstance(weights, Mapping)
        and set(weights) == {"student", "teacher"}
        and _asset_payload_contract(weights.get("student"), STUDENT_LOCK)
        and _asset_payload_contract(weights.get("teacher"), DINO_LOCK),
        "strict_load_exact": isinstance(strict_load, Mapping)
        and set(strict_load) == {"student_load", "teacher_load"}
        and isinstance(strict_load.get("student_load"), Mapping)
        and set(strict_load["student_load"])
        == {
            "runtime_class",
            "stages",
            "head",
            "head_dist",
            "parameters_5class",
            "strict_load",
            "head_reset",
        }
        and strict_load["student_load"].get("runtime_class")
        == "timm.models.swiftformer.SwiftFormer"
        and _exact_int_sequence(
            strict_load["student_load"].get("stages"), (48, 56, 112, 220)
        )
        and _exact_int_sequence(
            strict_load["student_load"].get("head"), (220, 5)
        )
        and _exact_int_sequence(
            strict_load["student_load"].get("head_dist"), (220, 5)
        )
        and strict_load["student_load"].get("parameters_5class") == STOCK_PARAMETERS
        and strict_load["student_load"].get("strict_load") is True
        and isinstance(strict_load["student_load"].get("head_reset"), Mapping)
        and set(strict_load["student_load"]["head_reset"])
        == {"seed", "initializer", "bias_zero", "state_sha256"}
        and strict_load["student_load"]["head_reset"].get("seed") == SEED
        and strict_load["student_load"]["head_reset"].get("initializer")
        == "timm.trunc_normal_std_0p02_sequential_head_then_head_dist"
        and strict_load["student_load"]["head_reset"].get("bias_zero") is True
        and _hex64(strict_load["student_load"]["head_reset"].get("state_sha256"))
        and strict_load.get("teacher_load")
        == {
            "runtime_class": "timm.models.eva.Eva",
            "features": DINO_FEATURE_DIM,
            "prefix_tokens": DINO_PREFIX_TOKENS,
            "grid": [16, 16],
            "all_requires_grad_false": True,
            "forbidden_classf_state_keys": [],
            "strict_load": True,
        }
        and _exact_int_sequence(
            strict_load["teacher_load"].get("grid"), (16, 16)
        )
        and strict_load["teacher_load"].get("all_requires_grad_false") is True
        and strict_load["teacher_load"].get("strict_load") is True,
        "focused_tests_exact": isinstance(focused, Mapping)
        and set(focused) == {"command", "returncode", "output"}
        and focused.get("command") == _focused_test_command()
        and type(focused.get("returncode")) is int
        and focused.get("returncode") == 0
        and isinstance(focused.get("output"), str)
        and "passed" in str(focused.get("output")).casefold()
        and "skipped" not in str(focused.get("output")).casefold(),
        "synthetic_exact": _synthetic_payload_contract(payload.get("synthetic")),
        "cuda_exact": _cuda_payload_contract(payload.get("cuda_fit")),
        "deployment_exact": _deployment_payload_contract(payload.get("deployment")),
        "offline_exact": isinstance(offline, Mapping)
        and set(offline)
        == {"environment", "socket_connect_denied", "network_attempts"}
        and offline.get("environment") == EXPECTED_OFFLINE_ENVIRONMENT
        and offline.get("socket_connect_denied") is True
        and type(offline.get("network_attempts")) is int
        and offline.get("network_attempts") == 0,
        "end_rehash_exact": isinstance(end_rehash, Mapping)
        and set(end_rehash)
        == {"git", "source_hashes", "timm_sources", "weights"}
        and _strict_json_equal(end_rehash.get("git"), git)
        and _strict_json_equal(end_rehash.get("source_hashes"), sources)
        and _strict_json_equal(end_rehash.get("timm_sources"), timm_sources)
        and _strict_json_equal(end_rehash.get("weights"), weights),
        "permissions_exact": isinstance(payload.get("permissions"), Mapping)
        and set(payload["permissions"]) == set(PERMISSIONS)
        and all(payload["permissions"][name] is False for name in PERMISSIONS),
        "top_checks_exact": _exact_true_checks(checks, TOP_CHECK_NAMES),
        "top_passed_exact": payload.get("passed") is True,
    }


def _validated_output_path(output_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    runs = (_repository_root() / "runs").resolve()
    if output.parent != runs or not output.name.startswith(PREFLIGHT_OUTPUT_PREFIX):
        raise ValueError(
            "B16 preflight output must be a fresh direct child of runs with the locked prefix"
        )
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Refuse to overwrite B16 preflight output: {output}")
    return output


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _atomic_bytes(path: Path, encoded: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    encoded = _canonical_json_bytes(payload)
    _atomic_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _publish_artifact_directory(
    output: Path, filename: str, payload: Mapping[str, object]
) -> dict[str, str]:
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite B16 artifact root: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Refuse to overwrite B16 partial root: {partial}")
    partial.mkdir(parents=False, exist_ok=False)
    try:
        artifact = partial / filename
        digest = _atomic_json(artifact, payload)
        sidecar = partial / f"{Path(filename).stem}.sha256"
        _atomic_bytes(sidecar, f"{digest}\n".encode("ascii"))
        observed = artifact.read_bytes()
        if (
            hashlib.sha256(observed).hexdigest() != digest
            or observed != _canonical_json_bytes(payload)
            or sidecar.read_text(encoding="ascii") != f"{digest}\n"
        ):
            raise RuntimeError("B16 artifact staging verification failed")
        partial.replace(output)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {
        "artifact": str(output / filename),
        "sha256": digest,
        "sidecar": str(output / f"{Path(filename).stem}.sha256"),
    }


def _failure_output_path(requested_output: Path) -> Path:
    requested = requested_output.expanduser().resolve()
    runs = (_repository_root() / "runs").resolve()
    if requested.parent != runs:
        raise ValueError("B16 failure artifact requires a canonical runs child")
    suffix = requested.name
    if suffix.startswith(PREFLIGHT_OUTPUT_PREFIX):
        suffix = suffix[len(PREFLIGHT_OUTPUT_PREFIX) :]
    failure = runs / f"{FAILURE_OUTPUT_PREFIX}{suffix}"
    if failure.exists() or failure.with_name(failure.name + ".partial").exists():
        failure = runs / f"{failure.name}_{uuid.uuid4().hex}"
    if failure.exists() or failure.with_name(failure.name + ".partial").exists():
        raise FileExistsError(f"Cannot allocate fresh B16 failure artifact: {failure}")
    return failure


def _write_failure_artifact(
    requested_output: Path,
    *,
    stage: str,
    error: BaseException,
    traceback_text: str,
) -> dict[str, str]:
    failure = _failure_output_path(requested_output)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "synthetic_preflight_failure_no_dataset",
        "created_at_unix": time.time(),
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback_sha256": hashlib.sha256(
            traceback_text.encode("utf-8")
        ).hexdigest(),
        "permissions": dict(PERMISSIONS),
        "passed": False,
    }
    return _publish_artifact_directory(failure, "failure.json", payload)


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"B16 artifact contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked offline label-free B16 SwiftSurface-XS preflight"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-weight", type=Path, default=None)
    parser.add_argument("--dino-weight", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def _end_rehash_contract(
    *,
    start_git: Mapping[str, object],
    start_sources: Mapping[str, str],
    start_timm_sources: Mapping[str, object],
    start_weights: Mapping[str, object],
    student_path: Path,
    dino_path: Path,
) -> dict[str, object]:
    end_git = _git_contract()
    end_sources = _source_hashes()
    end_timm_sources = _timm_source_contract()
    _, end_student = _resolve_offline_asset(STUDENT_LOCK, student_path)
    _, end_teacher = _resolve_offline_asset(DINO_LOCK, dino_path)
    if isinstance(start_weights.get("student"), Mapping):
        end_student["offline_cache_only"] = start_weights["student"].get(
            "offline_cache_only"
        )
    if isinstance(start_weights.get("teacher"), Mapping):
        end_teacher["offline_cache_only"] = start_weights["teacher"].get(
            "offline_cache_only"
        )
    end: dict[str, object] = {
        "git": end_git,
        "source_hashes": end_sources,
        "timm_sources": end_timm_sources,
        "weights": {"student": end_student, "teacher": end_teacher},
    }
    expected = {
        "git": dict(start_git),
        "source_hashes": dict(start_sources),
        "timm_sources": dict(start_timm_sources),
        "weights": dict(start_weights),
    }
    if not _strict_json_equal(end, expected):
        raise RuntimeError(f"B16 end rehash drifted from start: end={end}")
    return end


def build_preflight(args: argparse.Namespace) -> dict[str, object]:
    output = _validated_output_path(args.output_dir)
    stage = "offline_guard"
    try:
        offline = _offline_environment_contract()
        with _offline_network_guard() as network_attempts:
            stage = "start_git_sources_runtime"
            git = _git_contract()
            if not (
                git["branch"] == EXPECTED_BRANCH
                and git["tracked_worktree_clean"] is True
                and git["head_is_commit"] is True
            ):
                raise RuntimeError(
                    "B16 preflight requires a clean committed canonical branch: "
                    f"{git}"
                )
            sources = _source_hashes()
            runtime = _runtime_contract()
            timm_sources = _timm_source_contract()
            device, device_payload = _device_contract(args.device)

            stage = "offline_assets"
            student_path, student_asset = _resolve_offline_asset(
                STUDENT_LOCK, args.student_weight
            )
            dino_path, dino_asset = _resolve_offline_asset(
                DINO_LOCK, args.dino_weight
            )
            weights = {"student": student_asset, "teacher": dino_asset}

            stage = "focused_tests"
            focused = _run_focused_tests()
            stage = "strict_load"
            bundle, teacher, strict_load = _strict_load_models(
                student_path, dino_path
            )
            stage = "synthetic"
            synthetic = _synthetic_mechanism_contract(bundle)
            stage = "deployment"
            deployment = _deployment_contract(bundle)
            stage = "cuda_fit"
            cuda_fit = _cuda_fit_contract(bundle, teacher, device)

            stage = "end_rehash"
            end_rehash = _end_rehash_contract(
                start_git=git,
                start_sources=sources,
                start_timm_sources=timm_sources,
                start_weights=weights,
                student_path=student_path,
                dino_path=dino_path,
            )
            if network_attempts:
                raise RuntimeError(
                    f"B16 network guard recorded attempts: {network_attempts}"
                )
            checks = {
                "canonical_branch_clean_committed_head": True,
                "protocol_and_sources_exact_tracked": sources["protocol"]
                == PROTOCOL_SHA256,
                "runtime_and_timm_sources_exact": runtime == EXPECTED_RUNTIME
                and timm_sources["sha256"] == EXPECTED_TIMM_SOURCE_HASHES,
                "offline_assets_exact_strict": strict_load["student_load"][
                    "strict_load"
                ]
                is True
                and strict_load["teacher_load"]["strict_load"] is True,
                "focused_tests_passed_without_skip": focused["returncode"] == 0,
                "synthetic_mechanism_contract": synthetic["passed"] is True,
                "cuda_fit_contract": cuda_fit["passed"] is True,
                "deployment_contract": deployment["passed"] is True,
                "no_dataset_or_network_access": not network_attempts,
                "end_rehash_matches_start": True,
            }
            payload: dict[str, object] = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "mode": "synthetic_preflight_no_dataset",
                "created_at_unix": time.time(),
                "git": git,
                "source_hashes": sources,
                "runtime": runtime,
                "timm_sources": timm_sources,
                "device": device_payload,
                "weights": weights,
                "strict_load": strict_load,
                "focused_tests": focused,
                "synthetic": synthetic,
                "cuda_fit": cuda_fit,
                "deployment": deployment,
                "offline": offline,
                "end_rehash": end_rehash,
                "permissions": dict(PERMISSIONS),
                "checks": checks,
                "passed": all(checks.values()),
            }
            structure = _preflight_payload_structure_checks(payload)
            if not all(structure.values()):
                raise RuntimeError(
                    f"B16 preflight payload failed its own schema: {structure}"
                )

            stage = "final_rehash_before_atomic_publish"
            final_rehash = _end_rehash_contract(
                start_git=git,
                start_sources=sources,
                start_timm_sources=timm_sources,
                start_weights=weights,
                student_path=student_path,
                dino_path=dino_path,
            )
            if final_rehash != end_rehash or network_attempts:
                raise RuntimeError("B16 final rehash/network state drifted before publish")
            stage = "atomic_publish"
            _publish_artifact_directory(output, "preflight.json", payload)
            return payload
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        try:
            failure = _write_failure_artifact(
                output,
                stage=stage,
                error=error,
                traceback_text=failure_traceback,
            )
        except BaseException as failure_error:
            failure = {"failure_write_error": repr(failure_error)}
        setattr(error, "b16_failure_artifact", failure)
        raise


def validate_accepted_preflight(
    artifact: Path,
    expected_sha256: str,
    *,
    student_weight: Path | None = None,
    dino_weight: Path | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    path = artifact.expanduser().resolve(strict=True)
    runs = (_repository_root() / "runs").resolve(strict=True)
    if (
        path.name != "preflight.json"
        or path.parent.parent != runs
        or not path.parent.name.startswith(PREFLIGHT_OUTPUT_PREFIX)
    ):
        raise ValueError(
            "Accepted B16 artifact must be canonical repo/runs/<locked-prefix>/"
            "preflight.json"
        )
    if set(child.name for child in path.parent.iterdir()) != {
        "preflight.json",
        "preflight.sha256",
    }:
        raise ValueError("Accepted B16 artifact root contains unexpected files")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not _hex64(expected_sha256) or digest != expected_sha256.casefold():
        raise ValueError("Accepted B16 preflight SHA-256 mismatch")
    sidecar = path.with_name("preflight.sha256").read_bytes()
    if sidecar != f"{digest}\n".encode("ascii"):
        raise ValueError("Accepted B16 preflight SHA-256 sidecar mismatch")
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Accepted B16 preflight JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise TypeError("Accepted B16 preflight must be a mapping")
    if raw != _canonical_json_bytes(payload):
        raise ValueError("Accepted B16 preflight JSON bytes are not canonical")
    structure = _preflight_payload_structure_checks(payload)
    _, student_asset = _resolve_offline_asset(STUDENT_LOCK, student_weight)
    _, dino_asset = _resolve_offline_asset(DINO_LOCK, dino_weight)
    current_device = _device_contract(device)[1]
    checks = {
        **structure,
        "same_git": payload.get("git") == _git_contract(),
        "same_sources": payload.get("source_hashes") == _source_hashes(),
        "same_runtime": payload.get("runtime") == _runtime_contract(),
        "same_timm_sources": payload.get("timm_sources") == _timm_source_contract(),
        "same_device": payload.get("device") == current_device,
        "same_assets": payload.get("weights")
        == {"student": student_asset, "teacher": dino_asset},
    }
    if not all(checks.values()):
        raise RuntimeError(f"Accepted B16 preflight no longer applies: {checks}")
    if (
        path.read_bytes() != raw
        or path.with_name("preflight.sha256").read_bytes() != sidecar
        or set(child.name for child in path.parent.iterdir())
        != {"preflight.json", "preflight.sha256"}
    ):
        raise RuntimeError("Accepted B16 preflight changed during validation")
    return {"artifact": str(path), "sha256": digest, "payload": dict(payload)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = build_preflight(args)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "failure_artifact": getattr(
                        error, "b16_failure_artifact", None
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    artifact = args.output_dir.expanduser().resolve() / "preflight.json"
    print(
        json.dumps(
            {
                "protocol_id": payload["protocol_id"],
                "mode": payload["mode"],
                "passed": payload["passed"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "artifact": str(artifact),
                "sha256": _sha256(artifact),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

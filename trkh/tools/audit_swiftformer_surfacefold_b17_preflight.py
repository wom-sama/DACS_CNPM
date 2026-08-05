from __future__ import annotations

import argparse
import builtins
import copy
import gc
import hashlib
import importlib.metadata
import io
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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from safetensors.torch import load_file
from timm.layers import trunc_normal_
from torch import Tensor, nn

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
    dino_s3_relation_target_b17,
    surfacefold_s3_relation_loss_b17,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B17_SURFACEFOLD_XS_20260805"
PROTOCOL_SHA256 = "56ae72973b844186198dbc78c94851f48cf463b902370d2b3d2078b280169540"
MODEL_SHA256 = "7707ed6153fc4beed91312fc5d962f84a65652468e428f7da680546f899778db"
DEPLOYMENT_SHA256 = "bd662257105a1852539cc37ae10da3d82600129c7e35433b726355ef64a7cb00"
EXPECTED_BRANCH = "research/pretrained-classf-b1"
SEED = 20_260_805

STUDENT_TIMM_ID = "swiftformer_xs.dist_in1k"
DINO_TIMM_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_PREFIX_TOKENS = 5
DINO_CHANNELS = 384
DINO_IMAGE_SIZE = 256
STOCK_PARAMETERS = 3_035_570
ACTIVE_PARAMETERS = 3_061_218
STOCK_TRAINABLE = 2_813_590
ACTIVE_TRAINABLE = 2_839_238
FACTOR_PARAMETERS = 25_648
EXPECTED_HEAD_STATE_SHA256 = "565f95d6fd66341df4b0caf012dc0aae53038b0a8fc988ab247a734ba83d20a1"
EXPECTED_FOCUSED_TEST_COUNT = 82
SYNTHETIC_CE_WEIGHTS = (1.0, 1.1, 0.9, 1.2, 0.8)
MAX_ONNX_BYTES = int(13.5 * 1024**2)
MAX_ORT_BYTES = 14 * 1024**2
MAX_ORT_SOURCE_RATIO = 1.10
MAX_FORMAT_LATENCY_RATIO = 1.05
MAX_PARITY_ERROR = 1.0e-5
MAX_FOLD_ERROR = 1.0e-6
MAX_ORACLE_ERROR = 1.0e-10
MAX_FOLDED_SIZE_DELTA = 4096
MAX_CUDA_ALLOCATED_BYTES = 7 * 1024**3
OUTPUT_PREFIX = "preflight_b17_surfacefold_xs_"
FAILURE_PREFIX = "failed_b17_surfacefold_xs_"
DATASET_ROOT = Path(r"D:\DataAI\AIEx\newdataset\class_f")

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
EXPECTED_TIMM_HASHES = {
    "swiftformer.py": "e7f79c9bfb3750636ada4cd776c9ab9da41d321ba56ada28201a7dc480b654a7",
    "_factory.py": "30a6eecdaba750af470cfae3196186fd647052c72338a21c928914aac06163e4",
    "_builder.py": "424afd527cb1780d73c50f2302cd0945f3b1589a860427c45221d4f136e29708",
}
EXPECTED_CUDA = {
    "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "capability": [8, 9],
    "total_memory": 8_585_216_000,
    "bf16_supported": True,
}
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
NO_ACCESS_PERMISSIONS = {
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
SUCCESS_PERMISSIONS = {**NO_ACCESS_PERMISSIONS, "formal_train_permission": True}
AUTHORIZATION = {
    "scope": "one_train_only_five_fold_causal_screen",
    "deployment_status": "DEVICE_PENDING",
    "validation_permission": False,
    "test_permission": False,
}
SOURCE_LOCKS = {
    "protocol": PROTOCOL_SHA256,
    "model": MODEL_SHA256,
    "deployment": DEPLOYMENT_SHA256,
}
TOP_CHECKS = (
    "clean_committed_canonical_branch",
    "bound_sources_exact",
    "runtime_and_timm_exact",
    "offline_assets_strict",
    "focused_tests_no_skip",
    "stock_deployment_first",
    "mechanism_and_gradient_contract",
    "folded_topology_contract",
    "cuda_batch16_contract",
    "no_dataset_or_network_access",
    "start_end_rehash_exact",
    "train_only_authorization",
)
DEPLOYMENT_CHECKS = (
    "static_opset17_batch1_names",
    "deployment_parameter_count",
    "teacher_factor_relation_free",
    "onnx_size",
    "torch_ort_parity",
    "mobile_checkers_ran_with_logs",
    "fixed_arm_type_reduced_ort",
    "ort_reload_parity",
    "ort_size",
    "host_protocol_exact",
    "host_format_ratio",
)
MECHANISM_CHECKS = (
    "parameter_and_freeze_counts_exact",
    "matched_equal_disjoint_state",
    "mode_identity_and_cross_load_rejection",
    "initial_candidate_control_logits_bit_equal",
    "delta_off_stock_bit_exact",
    "relation_only_gradients_exact",
    "total_objective_gradients_exact",
    "float64_factor_oracle",
    "folded_feature_logit_parity",
    "immutable_w0_b0",
)
FOLDED_CHECKS = (
    "each_folded_arm_passes",
    "topology_identical_to_stock",
    "artifact_sizes_within_4kib",
    "no_extra_parameters_or_identifiers",
    "host_timing_diagnostic_not_gate",
)
CUDA_CHECKS = (
    "batch16_shared_exposure",
    "teacher_once_raw_fp32_gradient_free",
    "bf16_sequential_three_arms",
    "all_losses_finite",
    "every_trainable_gradient_finite",
    "required_gradients_nonzero",
    "adamw_state_every_trainable",
    "immutable_w0_b0",
    "peak_allocated_lte_7gib",
    "precision_determinism_exact",
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
    "student",
    "timm/swiftformer_xs.dist_in1k",
    "ac0196f198e58c82183a67f5f5c0952421b3e6ac",
    "model.safetensors",
    13_957_760,
    "c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c",
    "Apache-2.0",
)
DINO_LOCK = AssetLock(
    "teacher",
    "timm/vit_small_patch16_dinov3.lvd1689m",
    "3bf4720a82ec2066db88137180ff1f83a675cef0",
    "model.safetensors",
    86_362_376,
    "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040",
    "DINOv3-license",
)


@dataclass
class ArmBundle:
    stock: SwiftFormerSurfaceFoldB17
    control: SwiftFormerSurfaceFoldB17
    candidate: SwiftFormerSurfaceFoldB17
    rng_preserved: bool

    def items(self) -> tuple[tuple[str, SwiftFormerSurfaceFoldB17], ...]:
        return (
            ("stock", self.stock),
            ("control", self.control),
            ("candidate", self.candidate),
        )


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(root: Path, filename: str, payload: Mapping[str, object]) -> dict[str, str]:
    if root.exists() or root.with_name(root.name + ".partial").exists():
        raise FileExistsError(f"Refuse to overwrite B17 artifact root: {root}")
    partial = root.with_name(root.name + ".partial")
    partial.mkdir(parents=False)
    try:
        encoded = _canonical_bytes(payload)
        digest = hashlib.sha256(encoded).hexdigest()
        _atomic_bytes(partial / filename, encoded)
        _atomic_bytes(partial / f"{Path(filename).stem}.sha256", f"{digest}\n".encode("ascii"))
        if (partial / filename).read_bytes() != encoded:
            raise RuntimeError("B17 atomic artifact verification failed")
        partial.replace(root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {
        "artifact": str(root / filename),
        "sha256": digest,
        "sidecar": str(root / f"{Path(filename).stem}.sha256"),
    }


def _source_paths() -> dict[str, Path]:
    root = _root()
    return {
        "protocol": root / "docs" / "TRKH_PRETRAINED_CLASSF_B17_SURFACEFOLD_XS_PROTOCOL_20260805.md",
        "model": root / "trkh" / "models" / "swiftformer_surfacefold_b17.py",
        "model_test": root / "tests" / "test_swiftformer_surfacefold_b17.py",
        "deployment": root / "trkh" / "inference" / "surfacefold_deployment.py",
        "deployment_test": root / "tests" / "test_surfacefold_deployment.py",
        "preflight_runner": Path(__file__).resolve(),
        "preflight_test": root / "tests" / "test_audit_swiftformer_surfacefold_b17_preflight.py",
        "formal_runner": root / "trkh" / "tools" / "run_swiftformer_surfacefold_b17_train_oof.py",
        "formal_runner_test": root / "tests" / "test_run_swiftformer_surfacefold_b17_train_oof.py",
    }


def _source_hashes() -> dict[str, str]:
    root = _root()
    hashes: dict[str, str] = {}
    for name, path in _source_paths().items():
        if not path.is_file():
            raise FileNotFoundError(f"Bound B17 source is missing: {name}={path}")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"Bound B17 source is not tracked: {name}={path}")
        hashes[name] = _sha256(path)
    if any(hashes[name] != value for name, value in SOURCE_LOCKS.items()):
        raise RuntimeError(f"B17 locked source changed: {hashes}")
    return hashes


def _git_contract() -> dict[str, object]:
    def output(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=_root(), check=True, capture_output=True, text=True
        ).stdout.strip()

    status = output("status", "--porcelain", "--untracked-files=all")
    head = output("rev-parse", "HEAD")
    return {
        "head": head,
        "branch": output("branch", "--show-current"),
        "status": status,
        "clean": status == "",
        "head_is_commit": bool(re.fullmatch(r"[0-9a-f]{40}", head)),
    }


def _runtime_contract() -> dict[str, object]:
    ort_distribution = None
    for distribution in ("onnxruntime-gpu", "onnxruntime"):
        try:
            ort_distribution = {
                "distribution": distribution,
                "version": importlib.metadata.version(distribution),
            }
            break
        except importlib.metadata.PackageNotFoundError:
            pass
    if ort_distribution is None:
        raise importlib.metadata.PackageNotFoundError("ONNX Runtime is absent")
    observed = {
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
        raise RuntimeError(f"B17 runtime drifted: {observed}")
    return observed


def _timm_contract() -> dict[str, object]:
    import timm.models._builder as builder
    import timm.models._factory as factory
    import timm.models.swiftformer as swiftformer

    paths = {
        "swiftformer.py": Path(swiftformer.__file__).resolve(),
        "_factory.py": Path(factory.__file__).resolve(),
        "_builder.py": Path(builder.__file__).resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_TIMM_HASHES:
        raise RuntimeError(f"B17 timm source drifted: {hashes}")
    return {"paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes}


def _offline_environment() -> dict[str, object]:
    observed = {name: str(os.environ.get(name, "")) for name in OFFLINE_ENV}
    if observed != OFFLINE_ENV or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(f"B17 offline/deterministic environment drifted: {observed}")
    return {"environment": observed, "cublas_workspace_config": ":4096:8"}


def _dataset_target(value: object) -> bool:
    if isinstance(value, int):
        return False
    try:
        candidate = os.path.abspath(os.fspath(value)).replace("/", "\\").casefold()
    except TypeError:
        return False
    root = str(DATASET_ROOT).replace("/", "\\").casefold().rstrip("\\")
    return candidate == root or candidate.startswith(root + "\\")


@contextmanager
def _isolation_guard(*, deny_process: bool = False) -> Iterator[dict[str, list[str]]]:
    audit: dict[str, list[str]] = {
        "dataset_attempts": [],
        "network_attempts": [],
        "process_attempts": [],
    }

    def deny_dataset(function: Any, label: str) -> Any:
        def guarded(path: object, *args: object, **kwargs: object) -> Any:
            if _dataset_target(path):
                audit["dataset_attempts"].append(f"{label}:{path!s}")
                raise RuntimeError(f"B17 preflight blocked dataset access: {path}")
            return function(path, *args, **kwargs)

        return guarded

    def deny_network(*args: object, **kwargs: object) -> None:
        del kwargs
        target = repr(args[-1]) if args else "unknown"
        audit["network_attempts"].append(target)
        raise RuntimeError(f"B17 preflight blocked network access: {target}")

    def deny_child_process(*args: object, **kwargs: object) -> None:
        del kwargs
        target = repr(args[0]) if args else "unknown"
        audit["process_attempts"].append(target)
        raise RuntimeError(f"B17 guarded pytest blocked child process: {target}")

    with ExitStack() as stack:
        for owner, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "open"),
            (os, "listdir"),
            (os, "scandir"),
            (os, "stat"),
            (os, "lstat"),
        ):
            original = getattr(owner, name)
            stack.enter_context(mock.patch.object(owner, name, deny_dataset(original, name)))
        for name in ("connect", "connect_ex", "send", "sendall", "sendto"):
            stack.enter_context(mock.patch.object(socket.socket, name, deny_network))
        stack.enter_context(mock.patch.object(socket, "create_connection", deny_network))
        stack.enter_context(mock.patch.object(socket, "getaddrinfo", deny_network))
        if deny_process:
            for name in ("Popen", "run", "call", "check_call", "check_output"):
                stack.enter_context(
                    mock.patch.object(subprocess, name, deny_child_process)
                )
            stack.enter_context(mock.patch.object(os, "system", deny_child_process))
            if hasattr(os, "startfile"):
                stack.enter_context(mock.patch.object(os, "startfile", deny_child_process))
        yield audit
    if any(audit.values()):
        raise RuntimeError(f"B17 isolation guard observed attempts: {audit}")


def _resolve_asset(lock: AssetLock, explicit: Path | None) -> tuple[Path, dict[str, object]]:
    if explicit is None:
        cached = try_to_load_from_cache(lock.repo_id, lock.filename, revision=lock.revision)
        if not isinstance(cached, str):
            raise FileNotFoundError(f"Locked offline {lock.role} asset is absent")
        path = Path(cached).expanduser().resolve()
    else:
        path = explicit.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Locked B17 {lock.role} asset is missing: {path}")
    observed = {"bytes": int(path.stat().st_size), "sha256": _sha256(path)}
    if observed != {"bytes": lock.byte_count, "sha256": lock.sha256}:
        raise RuntimeError(f"Locked B17 {lock.role} asset changed: {observed}")
    return path, {
        "role": lock.role,
        "repo_id": lock.repo_id,
        "revision": lock.revision,
        "filename": lock.filename,
        "path": str(path),
        **observed,
        "license": lock.license,
        "offline_cache_only": explicit is None,
    }


def _device_contract(requested: str) -> tuple[torch.device, dict[str, object]]:
    if requested.casefold() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B17 formal preflight requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    observed = {
        "name": str(properties.name),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    if observed != EXPECTED_CUDA:
        raise RuntimeError(f"B17 CUDA device drifted: {observed}")
    return device, {"requested": "cuda", "resolved": str(device), **observed}


def _rng_digest() -> dict[str, object]:
    result: dict[str, object] = {
        "cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    if torch.cuda.is_initialized():
        result["cuda"] = [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ]
    return result


def _tensor_digest(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def reset_five_class_heads(model: nn.Module, *, seed: int = SEED) -> dict[str, object]:
    heads = (getattr(model, "head", None), getattr(model, "head_dist", None))
    if not all(isinstance(head, nn.Linear) and head.in_features == 220 for head in heads):
        raise TypeError("B17 requires two pretrained 220-channel linear heads")
    before = _rng_digest()
    device, dtype = heads[0].weight.device, heads[0].weight.dtype
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(int(seed))
        for name in ("head", "head_dist"):
            layer = nn.Linear(220, 5, device=device, dtype=dtype)
            trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
            setattr(model, name, layer)
    model.num_classes = 5
    state = {name: value for name, value in model.state_dict().items() if name.startswith("head")}
    result = {
        "seed": int(seed),
        "initializer": "timm_trunc_normal_0p02_head_then_head_dist",
        "bias_zero": bool(torch.count_nonzero(model.head.bias) == 0 and torch.count_nonzero(model.head_dist.bias) == 0),
        "state_sha256": _tensor_digest(state),
        "caller_rng_preserved": before == _rng_digest(),
    }
    if not result["bias_zero"] or not result["caller_rng_preserved"]:
        raise RuntimeError(f"B17 deterministic head reset failed: {result}")
    return result


def build_b17_arms(base: nn.Module) -> ArmBundle:
    before = _rng_digest()
    stock = SwiftFormerSurfaceFoldB17(copy.deepcopy(base), SURFACEFOLD_OFF_B17_MODE)
    bundle = build_active_arms_after_stock(base, stock)
    bundle.rng_preserved = bundle.rng_preserved and before == _rng_digest()
    return bundle


def build_active_arms_after_stock(
    base: nn.Module, qualified_stock: SwiftFormerSurfaceFoldB17
) -> ArmBundle:
    if qualified_stock.mode != SURFACEFOLD_OFF_B17_MODE:
        raise ValueError("B17 active arms require the already-qualified stock arm")
    before = _rng_digest()
    control = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_MEAN_CONTROL_B17_MODE
    )
    candidate = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_SPATIAL_B17_MODE
    )
    if not all(
        torch.equal(value, qualified_stock.backbone.state_dict()[name])
        for name, value in base.state_dict().items()
    ):
        raise RuntimeError("B17 active-arm base differs from qualified stock")
    return ArmBundle(qualified_stock, control, candidate, before == _rng_digest())


def _stage_channels(student: nn.Module) -> list[int]:
    feature_info = getattr(student, "feature_info", None)
    stages = feature_info.get_dicts() if hasattr(feature_info, "get_dicts") else feature_info
    if not isinstance(stages, (list, tuple)) or any(
        not isinstance(stage, Mapping) or type(stage.get("num_chs")) is not int
        for stage in stages
    ):
        raise TypeError("B17 SwiftFormer feature_info contract changed")
    return [int(stage["num_chs"]) for stage in stages]


def _strict_load_models(
    student_path: Path, dino_path: Path
) -> tuple[nn.Module, SwiftFormerSurfaceFoldB17, nn.Module, dict[str, object]]:
    before = _rng_digest()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(SEED)
        student = timm.create_model(
            STUDENT_TIMM_ID,
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        student_state = load_file(str(student_path), device="cpu")
        student.load_state_dict(student_state, strict=True)
        head_reset = reset_five_class_heads(student)
        stock = SwiftFormerSurfaceFoldB17(
            copy.deepcopy(student), SURFACEFOLD_OFF_B17_MODE
        )
        teacher = timm.create_model(
            DINO_TIMM_ID, pretrained=False, num_classes=0, img_size=DINO_IMAGE_SIZE
        )
        teacher_state = load_file(str(dino_path), device="cpu")
        teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval().requires_grad_(False)
    student_observed = {
        "runtime_class": f"{type(student).__module__}.{type(student).__name__}",
        "stage_channels": _stage_channels(student),
        "parameters": sum(parameter.numel() for parameter in student.parameters()),
        "heads": [[student.head.in_features, student.head.out_features], [student.head_dist.in_features, student.head_dist.out_features]],
        "distilled_training": bool(getattr(student, "distilled_training", False)),
        "nonzero_dropout_or_path": [
            name
            for name, module in student.named_modules()
            if (isinstance(module, nn.Dropout) and float(module.p) != 0.0)
            or (
                hasattr(module, "drop_prob")
                and getattr(module, "drop_prob") is not None
                and float(getattr(module, "drop_prob")) != 0.0
            )
        ],
    }
    teacher_observed = {
        "runtime_class": f"{type(teacher).__module__}.{type(teacher).__name__}",
        "features": int(getattr(teacher, "num_features", -1)),
        "prefix_tokens": int(getattr(teacher, "num_prefix_tokens", -1)),
        "grid": list(getattr(teacher.patch_embed, "grid_size", ())),
        "all_frozen": all(not parameter.requires_grad for parameter in teacher.parameters()),
    }
    if student_observed != {
        "runtime_class": "timm.models.swiftformer.SwiftFormer",
        "stage_channels": [48, 56, 112, 220],
        "parameters": STOCK_PARAMETERS,
        "heads": [[220, 5], [220, 5]],
        "distilled_training": False,
        "nonzero_dropout_or_path": [],
    } or teacher_observed != {
        "runtime_class": "timm.models.eva.Eva",
        "features": DINO_CHANNELS,
        "prefix_tokens": DINO_PREFIX_TOKENS,
        "grid": [16, 16],
        "all_frozen": True,
    }:
        raise RuntimeError(
            f"B17 strict architecture contract failed: {student_observed}, {teacher_observed}"
        )
    if before != _rng_digest():
        raise RuntimeError("B17 strict load/build changed caller RNG")
    return student, stock, teacher, {
        "student": {**student_observed, "strict_load": True, "head_reset": head_reset, "asset_keys": len(student_state)},
        "teacher": {**teacher_observed, "strict_load": True, "asset_keys": len(teacher_state)},
        "caller_rng_preserved": True,
    }


def _state_storage_disjoint(left: nn.Module, right: nn.Module) -> bool:
    left_ptrs = {
        value.untyped_storage().data_ptr()
        for value in left.state_dict().values()
        if value.numel()
    }
    right_ptrs = {
        value.untyped_storage().data_ptr()
        for value in right.state_dict().values()
        if value.numel()
    }
    return not bool(left_ptrs & right_ptrs)


def _common_state_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(
        name == "_surfacefold_mode_id" or torch.equal(left_state[name], right_state[name])
        for name in left_state
    )


def _surfacefold_initialization_exact(model: SwiftFormerSurfaceFoldB17) -> bool:
    projection = model.backbone.stages[3].downsample.proj
    if (
        model.factor_p is None
        or model.factor_d is None
        or projection.weight.requires_grad
        or projection.bias is None
        or projection.bias.requires_grad
        or not torch.equal(model.factor_d, torch.full_like(model.factor_d, 1.0 / 9.0))
    ):
        return False
    delta_norm = model.effective_delta_weight().double().flatten(1).norm(dim=1)
    base_norm = projection.weight.detach().double().flatten(1).norm(dim=1)
    return bool(
        torch.isfinite(model.factor_p).all()
        and torch.isfinite(model.factor_d).all()
        and torch.allclose(
            delta_norm / base_norm,
            torch.full_like(base_norm, 0.01),
            atol=2.0e-8,
            rtol=2.0e-6,
        )
    )


def _role_gradient(model: nn.Module, role: str) -> dict[str, object]:
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        lower = name.casefold()
        if role == "head":
            matched = lower == "backbone.head.weight" or lower == "backbone.head.bias"
        elif role == "head_dist":
            matched = lower == "backbone.head_dist.weight" or lower == "backbone.head_dist.bias"
        elif role == "p":
            matched = lower == "factor_p"
        elif role == "d":
            matched = lower == "factor_d"
        else:
            raise ValueError(f"Unknown B17 gradient role: {role}")
        if matched:
            selected.append((name, parameter))
    present = bool(selected)
    finite = present and all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for _, parameter in selected
    )
    nonzero = finite and any(
        int(torch.count_nonzero(parameter.grad)) > 0 for _, parameter in selected
    )
    return {
        "names": [name for name, _ in selected],
        "present": present,
        "finite": bool(finite),
        "nonzero": bool(nonzero),
    }


def _relation_only_gradients(model: SwiftFormerSurfaceFoldB17) -> dict[str, object]:
    probe = copy.deepcopy(model).train()
    generator = torch.Generator().manual_seed(SEED + 11)
    s2 = torch.randn(2, 112, 14, 14, generator=generator)
    raw_teacher = torch.randn(2, 384, 16, 16, generator=generator, requires_grad=True)
    target = dino_s3_relation_target_b17(raw_teacher)
    s3 = probe.forward_stage3_downsample(s2)
    s3.retain_grad()
    loss = surfacefold_s3_relation_loss_b17(s3, target)
    loss.backward()
    p_gradient = _role_gradient(probe, "p")
    d_gradient = _role_gradient(probe, "d")
    if probe.factor_d is None or probe.factor_d.grad is None:
        centered_nonzero = False
        spatially_equal = False
    else:
        gradient = probe.factor_d.grad.detach()
        centered = gradient - gradient.mean(dim=(1, 2), keepdim=True)
        centered_nonzero = bool(torch.count_nonzero(centered) > 0)
        spatially_equal = bool(
            torch.equal(gradient, gradient[:, :1, :1].expand_as(gradient))
        )
    return {
        "loss": float(loss.detach()),
        "p": p_gradient,
        "d": d_gradient,
        "candidate_centered_d_nonzero": centered_nonzero,
        "control_d_spatially_equal": spatially_equal,
        "s3_gradient_nonzero": bool(
            s3.grad is not None
            and torch.isfinite(s3.grad).all()
            and torch.count_nonzero(s3.grad) > 0
        ),
        "teacher_gradient_absent": raw_teacher.grad is None,
    }


def _total_objective_gradients(
    model: SwiftFormerSurfaceFoldB17, *, active: bool
) -> dict[str, object]:
    probe = copy.deepcopy(model).train()
    generator = torch.Generator().manual_seed(SEED + 12)
    images = torch.randn(2, 3, 224, 224, generator=generator)
    labels = torch.tensor([1, 2], dtype=torch.long)
    raw_teacher = torch.randn(2, 384, 16, 16, generator=generator, requires_grad=True)
    target = dino_s3_relation_target_b17(raw_teacher)
    logits, s3 = probe(images, return_s3=True)
    s3.retain_grad()
    relation = surfacefold_s3_relation_loss_b17(s3, target)
    weights = torch.tensor(SYNTHETIC_CE_WEIGHTS)
    ce = F.cross_entropy(logits.float(), labels, weight=weights, reduction="mean")
    loss = ce + 0.10 * relation
    loss.backward()
    roles = {role: _role_gradient(probe, role) for role in ("head", "head_dist")}
    if active:
        roles.update({role: _role_gradient(probe, role) for role in ("p", "d")})
    return {
        "ce": float(ce.detach()),
        "relation": float(relation.detach()),
        "total": float(loss.detach()),
        "roles": roles,
        "s3_gradient_nonzero": bool(
            s3.grad is not None
            and torch.isfinite(s3.grad).all()
            and torch.count_nonzero(s3.grad) > 0
        ),
        "teacher_gradient_absent": raw_teacher.grad is None,
    }


def _float64_factor_oracle(model: SwiftFormerSurfaceFoldB17) -> float:
    if model.factor_p is None or model.factor_d is None:
        raise TypeError("B17 factor oracle requires an active arm")
    generator = torch.Generator().manual_seed(SEED + 13)
    s2 = torch.randn(2, 112, 14, 14, generator=generator, dtype=torch.float64)
    p = model.factor_p.detach().double()
    d = model.factor_d.detach().double()
    if model.mode == SURFACEFOLD_MEAN_CONTROL_B17_MODE:
        d = d.mean(dim=(1, 2), keepdim=True).expand_as(d)
    expanded = F.conv2d(s2, p[:, :, None, None] * d[None], stride=2, padding=1)
    depthwise = F.conv2d(s2, d[:, None], stride=2, padding=1, groups=112)
    factored = F.conv2d(depthwise, p[:, :, None, None])
    return float((expanded - factored).abs().max())


def _factor_fold_parity(model: SwiftFormerSurfaceFoldB17) -> dict[str, float]:
    probe = copy.deepcopy(model).cpu().eval()
    deployed = probe.fold_to_deploy().cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 14)
    images = torch.randn(1, 3, 224, 224, generator=generator)
    with torch.inference_mode():
        stem = probe.backbone.stem(images)
        stem = probe.backbone.stages[0](stem)
        stem = probe.backbone.stages[1](stem)
        s2 = probe.backbone.stages[2](stem)
        projection = probe.backbone.stages[3].downsample.proj
        factor_pre_bn = F.conv2d(
            s2,
            probe.effective_projection_weight(),
            projection.bias,
            stride=projection.stride,
            padding=projection.padding,
        )
        folded_projection = deployed.stages[3].downsample.proj
        folded_pre_bn = folded_projection(s2)
        factor_features = probe.forward_features(images)
        folded_features = deployed.forward_features(images)
        factor_logits = probe(images)
        folded_logits = deployed(images)
    return {
        "pre_bn": float((factor_pre_bn - folded_pre_bn).abs().max()),
        "features": float((factor_features - folded_features).abs().max()),
        "logits": float((factor_logits - folded_logits).abs().max()),
    }


def _mechanism_contract(bundle: ArmBundle) -> dict[str, object]:
    models = dict(bundle.items())
    counts = {
        name: {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
        for name, model in models.items()
    }
    frozen_before = {
        name: _tensor_digest(
            {
                "weight": model.backbone.stages[3].downsample.proj.weight,
                "bias": model.backbone.stages[3].downsample.proj.bias,
            }
        )
        for name, model in models.items()
    }
    common_equal = _common_state_equal(bundle.control, bundle.candidate)
    disjoint = all(
        _state_storage_disjoint(left, right)
        for left, right in (
            (bundle.stock, bundle.control),
            (bundle.stock, bundle.candidate),
            (bundle.control, bundle.candidate),
        )
    )
    generator = torch.Generator().manual_seed(SEED + 10)
    image = torch.randn(1, 3, 224, 224, generator=generator)
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        control_logits = bundle.control(image)
        candidate_logits = bundle.candidate(image)
        stock_logits = bundle.stock(image)
        raw_stock_logits = bundle.stock.backbone(image)
    cross_mode_rejected = True
    for source, target in (
        (bundle.candidate, bundle.control),
        (bundle.control, bundle.candidate),
    ):
        try:
            copy.deepcopy(target).load_state_dict(source.state_dict(), strict=True)
            cross_mode_rejected = False
        except RuntimeError:
            pass
    relation_gradients = {
        "control": _relation_only_gradients(bundle.control),
        "candidate": _relation_only_gradients(bundle.candidate),
    }
    total_gradients = {
        name: _total_objective_gradients(model, active=name != "stock")
        for name, model in models.items()
    }
    oracle = {
        "control": _float64_factor_oracle(bundle.control),
        "candidate": _float64_factor_oracle(bundle.candidate),
    }
    fold_parity = {
        "control": _factor_fold_parity(bundle.control),
        "candidate": _factor_fold_parity(bundle.candidate),
    }
    frozen_after = {
        name: _tensor_digest(
            {
                "weight": model.backbone.stages[3].downsample.proj.weight,
                "bias": model.backbone.stages[3].downsample.proj.bias,
            }
        )
        for name, model in models.items()
    }
    relation_pass = (
        all(
            evidence["p"]["nonzero"] is True
            and evidence["d"]["nonzero"] is True
            and evidence["s3_gradient_nonzero"] is True
            and evidence["teacher_gradient_absent"] is True
            for evidence in relation_gradients.values()
        )
        and relation_gradients["candidate"]["candidate_centered_d_nonzero"] is True
        and relation_gradients["control"]["control_d_spatially_equal"] is True
    )
    total_pass = all(
        evidence["s3_gradient_nonzero"] is True
        and evidence["teacher_gradient_absent"] is True
        and math.isfinite(float(evidence["total"]))
        and all(role["nonzero"] is True for role in evidence["roles"].values())
        for evidence in total_gradients.values()
    )
    checks = {
        "parameter_and_freeze_counts_exact": counts
        == {
            "stock": {"total": STOCK_PARAMETERS, "trainable": STOCK_TRAINABLE},
            "control": {"total": ACTIVE_PARAMETERS, "trainable": ACTIVE_TRAINABLE},
            "candidate": {"total": ACTIVE_PARAMETERS, "trainable": ACTIVE_TRAINABLE},
        }
        and bundle.control.factor_p.numel() + bundle.control.factor_d.numel()
        == FACTOR_PARAMETERS
        and _surfacefold_initialization_exact(bundle.control)
        and _surfacefold_initialization_exact(bundle.candidate)
        and not bundle.stock.backbone.stages[3].downsample.proj.weight.requires_grad
        and not bundle.stock.backbone.stages[3].downsample.proj.bias.requires_grad,
        "matched_equal_disjoint_state": common_equal and disjoint and bundle.rng_preserved,
        "mode_identity_and_cross_load_rejection": len(
            {int(model._surfacefold_mode_id.item()) for model in models.values()}
        )
        == 3
        and cross_mode_rejected,
        "initial_candidate_control_logits_bit_equal": torch.equal(
            control_logits, candidate_logits
        ),
        "delta_off_stock_bit_exact": torch.equal(stock_logits, raw_stock_logits),
        "relation_only_gradients_exact": relation_pass,
        "total_objective_gradients_exact": total_pass,
        "float64_factor_oracle": all(value <= MAX_ORACLE_ERROR for value in oracle.values()),
        "folded_feature_logit_parity": all(
            value <= MAX_FOLD_ERROR
            for arm in fold_parity.values()
            for value in arm.values()
        ),
        "immutable_w0_b0": frozen_before == frozen_after,
    }
    return {
        "counts": counts,
        "relation_only_gradients": relation_gradients,
        "total_objective_gradients": total_gradients,
        "float64_oracle_max_abs": oracle,
        "fold_parity_max_abs": fold_parity,
        "w0_b0_sha256": frozen_after,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _synthetic_arrays() -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(SEED + 101)
    return tuple(
        generator.standard_normal((1, 3, 224, 224)).astype(np.float32)
        for _ in range(64)
    )


def _portable_deployment_evidence(
    model: nn.Module,
    name: str,
    root: Path,
    arrays: tuple[np.ndarray, ...],
    *,
    require_host_ratio: bool,
) -> dict[str, object]:
    model = copy.deepcopy(model).cpu().eval()
    model_terms = "\n".join(
        [module_name for module_name, _ in model.named_modules()]
        + [parameter_name for parameter_name, _ in model.named_parameters()]
    ).casefold()
    forbidden = [
        term
        for term in ("surfacefold", "factor_p", "factor_d", "teacher", "dino", "relation")
        if term in model_terms
    ]
    onnx_path = root / f"{name}.onnx"
    exported = export_fixed_opset17_onnx(
        model,
        torch.from_numpy(arrays[0].copy()),
        onnx_path,
        input_name="x",
        output_name="logits",
    )
    topology = onnx_topology_fingerprint(onnx_path)
    topology_text = json.dumps(
        topology["descriptor"], sort_keys=True, separators=(",", ":")
    ).casefold()
    forbidden.extend(
        term
        for term in ("surfacefold", "factor_p", "factor_d", "teacher", "dino", "relation")
        if term in topology_text
    )
    parity = compare_pytorch_ort(
        model,
        onnx_path,
        arrays,
        providers=("CPUExecutionProvider",),
        intra_op_threads=4,
        inter_op_threads=1,
        max_abs_error=MAX_PARITY_ERROR,
        require_argmax_match=True,
    )
    mobile = check_ort_mobile_usability(onnx_path)
    conversion = convert_fixed_ort_arm(
        onnx_path,
        root / f"{name}_ort",
        arrays,
        intra_op_threads=4,
        inter_op_threads=1,
        max_abs_error=MAX_PARITY_ERROR,
    )
    ort_path = Path(str(conversion["ort_path"]))
    latency = benchmark_onnx_vs_ort_cpu(
        onnx_path,
        ort_path,
        arrays[0],
        intra_op_threads=4,
        inter_op_threads=1,
        warmups=15,
        trials=5,
        iterations_per_trial=100,
        max_abs_error=MAX_PARITY_ERROR,
    )
    ratios = latency["ort_to_onnx_ratio"]
    checks = {
        "static_opset17_batch1_names": exported["opset"] == 17
        and exported["input_name"] == "x"
        and exported["input_shape"] == [1, 3, 224, 224]
        and exported["output_name"] == "logits"
        and exported["output_shape"] == [1, 5],
        "deployment_parameter_count": sum(parameter.numel() for parameter in model.parameters())
        == STOCK_PARAMETERS,
        "teacher_factor_relation_free": not forbidden,
        "onnx_size": int(exported["size_bytes"]) <= MAX_ONNX_BYTES,
        "torch_ort_parity": parity["max_abs_error"] <= MAX_PARITY_ERROR
        and parity["argmax_mismatches"] == 0
        and parity["samples"] == 64,
        "mobile_checkers_ran_with_logs": type(mobile["prebuilt_mobile_package_supported"])
        is bool
        and type(mobile["nnapi_or_coreml_may_help"]) is bool
        and bool(str(mobile["log"]).strip()),
        "fixed_arm_type_reduced_ort": conversion["optimization_style"] == "Fixed"
        and conversion["target_platform"] == "arm"
        and conversion["type_reduction"] is True
        and conversion["providers"] == ["CPUExecutionProvider"],
        "ort_reload_parity": conversion["parity"]["max_abs_error"]
        <= MAX_PARITY_ERROR
        and conversion["parity"]["argmax_mismatches"] == 0
        and conversion["parity"]["samples"] == 64,
        "ort_size": int(conversion["ort_size_bytes"]) <= MAX_ORT_BYTES
        and int(conversion["ort_size_bytes"])
        <= MAX_ORT_SOURCE_RATIO * int(exported["size_bytes"]),
        "host_protocol_exact": latency["provider"] == "CPUExecutionProvider"
        and latency["intra_op_threads"] == 4
        and latency["inter_op_threads"] == 1
        and latency["execution_mode"] == "ORT_SEQUENTIAL"
        and latency["warmups"] == 15
        and latency["trials"] == 5
        and latency["iterations_per_trial"] == 100
        and latency["order_rule"] == "alternate by (trial + iteration) modulo 2",
        "host_format_ratio": (
            ratios["median"] <= MAX_FORMAT_LATENCY_RATIO
            and ratios["p95"] <= MAX_FORMAT_LATENCY_RATIO
        )
        if require_host_ratio
        else True,
    }
    return {
        "arm": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "onnx": {key: value for key, value in exported.items() if key != "path"},
        "topology_sha256": topology["sha256"],
        "parity": {key: value for key, value in parity.items() if key != "onnx_path"},
        "mobile": {key: value for key, value in mobile.items() if key != "onnx_path"},
        "ort": {
            key: value
            for key, value in conversion.items()
            if key not in {"source_onnx_path", "ort_path", "config_path"}
        },
        "latency": latency,
        "forbidden_identifiers": sorted(set(forbidden)),
        "host_ratio_is_gate": require_host_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _stock_deployment_contract(stock: SwiftFormerSurfaceFoldB17) -> dict[str, object]:
    arrays = _synthetic_arrays()
    with tempfile.TemporaryDirectory(prefix="trkh_b17_stock_") as temporary:
        evidence = _portable_deployment_evidence(
            stock.fold_to_deploy(),
            "stock",
            Path(temporary),
            arrays,
            require_host_ratio=True,
        )
    if evidence["passed"] is not True:
        error = RuntimeError(
            f"B17 stock deployment gate failed: {evidence['checks']}"
        )
        setattr(
            error,
            "b17_gate_evidence",
            {"gate": "stock_deployment_first", "evidence": evidence},
        )
        raise error
    return evidence


def _folded_deployment_contract(
    bundle: ArmBundle, stock: Mapping[str, object]
) -> dict[str, object]:
    arrays = _synthetic_arrays()
    with tempfile.TemporaryDirectory(prefix="trkh_b17_folded_") as temporary:
        root = Path(temporary)
        arms = {
            "control": _portable_deployment_evidence(
                bundle.control.fold_to_deploy(),
                "control",
                root,
                arrays,
                require_host_ratio=False,
            ),
            "candidate": _portable_deployment_evidence(
                bundle.candidate.fold_to_deploy(),
                "candidate",
                root,
                arrays,
                require_host_ratio=False,
            ),
        }
    stock_onnx = int(stock["onnx"]["size_bytes"])
    stock_ort = int(stock["ort"]["ort_size_bytes"])
    checks = {
        "each_folded_arm_passes": all(arm["passed"] is True for arm in arms.values()),
        "topology_identical_to_stock": all(
            arm["topology_sha256"] == stock["topology_sha256"] for arm in arms.values()
        ),
        "artifact_sizes_within_4kib": all(
            abs(int(arm["onnx"]["size_bytes"]) - stock_onnx) <= MAX_FOLDED_SIZE_DELTA
            and abs(int(arm["ort"]["ort_size_bytes"]) - stock_ort)
            <= MAX_FOLDED_SIZE_DELTA
            for arm in arms.values()
        ),
        "no_extra_parameters_or_identifiers": all(
            arm["parameters"] == STOCK_PARAMETERS and not arm["forbidden_identifiers"]
            for arm in arms.values()
        ),
        "host_timing_diagnostic_not_gate": all(
            arm["host_ratio_is_gate"] is False for arm in arms.values()
        ),
    }
    return {
        "stock_topology_sha256": stock["topology_sha256"],
        "arms": arms,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _optimizer_groups(model: nn.Module) -> list[dict[str, object]]:
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.casefold()
        task = lower in {"factor_p", "factor_d"} or lower.startswith(
            ("backbone.head.", "backbone.head_dist.")
        )
        lr = 3.0e-4 if task else 3.0e-5
        no_decay = (
            parameter.ndim <= 1
            or lower.endswith("bias")
            or "norm" in lower
            or "layer_scale" in lower
        )
        groups.setdefault((lr, 0.0 if no_decay else 0.05), []).append(parameter)
    return [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in sorted(groups.items())
    ]


def _optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    parameters = {parameter for parameter in model.parameters() if parameter.requires_grad}
    present = set(optimizer.state)
    exact = present == parameters
    finite = exact
    shapes = exact
    steps = exact
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        exact = exact and set(state) == {"step", "exp_avg", "exp_avg_sq"}
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            finite = shapes = steps = False
            continue
        finite = finite and all(torch.isfinite(value).all() for value in state.values())
        shapes = shapes and state["exp_avg"].shape == parameter.shape and state[
            "exp_avg_sq"
        ].shape == parameter.shape
        steps = steps and float(state["step"].detach().cpu()) == 1.0
    return {
        "trainable_tensors": len(parameters),
        "state_tensors": len(present),
        "parameter_set_exact": exact,
        "finite": bool(finite),
        "shapes_exact": bool(shapes),
        "steps_one": bool(steps),
    }


def _all_trainable_gradients(model: nn.Module) -> dict[str, object]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    return {
        "tensors": len(trainable),
        "missing": missing,
        "nonfinite": nonfinite,
        "all_present_finite": not missing and not nonfinite,
    }


def _cuda_contract(
    bundle: ArmBundle, teacher: nn.Module, device: torch.device
) -> dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("B17 CUDA preflight requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    old_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    old_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    old_cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    models: dict[str, SwiftFormerSurfaceFoldB17] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    teacher_cuda: nn.Module | None = None
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        models = {name: copy.deepcopy(model).to(device).train() for name, model in bundle.items()}
        teacher_cuda = copy.deepcopy(teacher).to(device).eval().requires_grad_(False)
        optimizers = {
            name: torch.optim.AdamW(
                _optimizer_groups(model), betas=(0.9, 0.999), eps=1.0e-8
            )
            for name, model in models.items()
        }
        generator = torch.Generator(device="cpu").manual_seed(SEED + 41)
        shared_base = torch.rand(16, 3, 256, 256, generator=generator)
        shared_student = F.interpolate(
            shared_base,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        shared_student = ((shared_student - mean) / std).to(device)
        shared_teacher = ((shared_base - mean) / std).to(device)
        shared_labels = (torch.arange(16, device=device) % 5).long()
        synthetic_weights = torch.tensor(
            SYNTHETIC_CE_WEIGHTS, device=device, dtype=torch.float32
        )
        exposure_sha256 = _tensor_digest(
            {
                "student": shared_student,
                "teacher": shared_teacher,
                "labels": shared_labels,
            }
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        teacher_calls = 0
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            tokens = teacher_cuda.forward_features(shared_teacher.float())
            teacher_calls += 1
            if not isinstance(tokens, Tensor) or tuple(tokens.shape) != (16, 261, 384):
                raise RuntimeError(f"B17 DINO geometry drifted: {getattr(tokens, 'shape', None)}")
            raw_teacher = (
                tokens[:, DINO_PREFIX_TOKENS:]
                .reshape(16, 16, 16, DINO_CHANNELS)
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
            )
        raw_teacher = raw_teacher.clone()
        relation_target = dino_s3_relation_target_b17(raw_teacher)
        teacher_evidence = {
            "calls": teacher_calls,
            "tokens_shape": [16, 261, 384],
            "raw_map_shape": list(raw_teacher.shape),
            "target_shape": list(relation_target.shape),
            "target_dtype": str(relation_target.dtype),
            "target_requires_grad": bool(relation_target.requires_grad),
            "parameter_gradients_present": sum(
                parameter.grad is not None for parameter in teacher_cuda.parameters()
            ),
        }
        del tokens, raw_teacher

        losses: dict[str, dict[str, float]] = {}
        gradients: dict[str, dict[str, object]] = {}
        optimizer_states: dict[str, dict[str, object]] = {}
        w0_before: dict[str, str] = {}
        w0_after: dict[str, str] = {}
        for name in ("stock", "control", "candidate"):
            model, optimizer = models[name], optimizers[name]
            projection = model.backbone.stages[3].downsample.proj
            w0_before[name] = _tensor_digest(
                {"weight": projection.weight, "bias": projection.bias}
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, s3 = model(shared_student, return_s3=True)
                s3.retain_grad()
                relation = surfacefold_s3_relation_loss_b17(s3, relation_target)
                ce = F.cross_entropy(
                    logits.float(),
                    shared_labels,
                    weight=synthetic_weights,
                    reduction="mean",
                )
                total = ce + 0.10 * relation
            total.backward()
            all_gradients = _all_trainable_gradients(model)
            roles = {
                role: _role_gradient(model, role)
                for role in (
                    ("head", "head_dist", "p", "d")
                    if name != "stock"
                    else ("head", "head_dist")
                )
            }
            gradients[name] = {
                "all": all_gradients,
                "roles": roles,
                "s3_nonzero": bool(
                    s3.grad is not None
                    and torch.isfinite(s3.grad).all()
                    and torch.count_nonzero(s3.grad) > 0
                ),
                "logits_dtype": str(logits.dtype),
                "s3_dtype": str(s3.dtype),
            }
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            optimizer_states[name] = _optimizer_state(model, optimizer)
            w0_after[name] = _tensor_digest(
                {"weight": projection.weight, "bias": projection.bias}
            )
            losses[name] = {
                "ce": float(ce.detach()),
                "relation": float(relation.detach()),
                "total": float(total.detach()),
            }
            optimizer.zero_grad(set_to_none=True)
            del logits, s3, relation, ce, total
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        checks = {
            "batch16_shared_exposure": tuple(shared_student.shape) == (16, 3, 224, 224)
            and tuple(shared_teacher.shape) == (16, 3, 256, 256)
            and tuple(relation_target.shape) == (16, 84),
            "teacher_once_raw_fp32_gradient_free": teacher_evidence
            == {
                "calls": 1,
                "tokens_shape": [16, 261, 384],
                "raw_map_shape": [16, 384, 16, 16],
                "target_shape": [16, 84],
                "target_dtype": "torch.float32",
                "target_requires_grad": False,
                "parameter_gradients_present": 0,
            }
            and relation_target.dtype == torch.float32
            and not relation_target.requires_grad
            and all(parameter.grad is None for parameter in teacher_cuda.parameters()),
            "bf16_sequential_three_arms": list(losses) == ["stock", "control", "candidate"],
            "all_losses_finite": all(
                math.isfinite(value)
                for arm in losses.values()
                for value in arm.values()
            ),
            "every_trainable_gradient_finite": all(
                evidence["all"]["all_present_finite"] is True
                for evidence in gradients.values()
            ),
            "required_gradients_nonzero": all(
                evidence["s3_nonzero"] is True
                and all(role["nonzero"] is True for role in evidence["roles"].values())
                for evidence in gradients.values()
            ),
            "adamw_state_every_trainable": all(
                state["parameter_set_exact"] is True
                and state["finite"] is True
                and state["shapes_exact"] is True
                and state["steps_one"] is True
                for state in optimizer_states.values()
            ),
            "immutable_w0_b0": w0_before == w0_after,
            "peak_allocated_lte_7gib": peak_allocated <= MAX_CUDA_ALLOCATED_BYTES,
            "precision_determinism_exact": torch.are_deterministic_algorithms_enabled()
            and not torch.backends.cuda.matmul.allow_tf32
            and not torch.backends.cudnn.allow_tf32
            and torch.backends.cudnn.deterministic
            and not torch.backends.cudnn.benchmark,
        }
        runtime_state = {
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
        return {
            "settings": {
                "batch": 16,
                "teacher_precision": "fp32_inference_once",
                "student_precision": "bf16_autocast",
                "arm_order": ["stock", "control", "candidate"],
                "optimizer": "AdamW_one_real_step_each",
                "gradient_clip": 0.7,
                "synthetic_ce_weights": list(SYNTHETIC_CE_WEIGHTS),
                "tf32": False,
            },
            "exposure_sha256": exposure_sha256,
            "teacher": teacher_evidence,
            "runtime_state": runtime_state,
            "losses": losses,
            "gradients": gradients,
            "optimizer_states": optimizer_states,
            "w0_b0_before": w0_before,
            "w0_b0_after": w0_after,
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
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        del optimizers, models, teacher_cuda
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


FOCUSED_TEST_PATHS = (
    "tests/test_swiftformer_surfacefold_b17.py",
    "tests/test_surfacefold_deployment.py",
    "tests/test_audit_swiftformer_surfacefold_b17_preflight.py",
    "tests/test_run_swiftformer_surfacefold_b17_train_oof.py",
)


def _focused_test_bootstrap() -> str:
    return """\
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

import pytest
from trkh.tools import audit_swiftformer_surfacefold_b17_preflight as gate

handshake = Path(sys.argv[1]).resolve()
test_paths = sys.argv[2:]
gate.DATASET_ROOT = Path(os.path.abspath(os.environ["TRKH_B17_CHILD_GUARD_ROOT"]))
audit = {"dataset_attempts": [], "network_attempts": [], "process_attempts": []}
error = None
pytest_returncode = 86
class Recorder:
    def __init__(self):
        self.collected = []
        self.executed = []
    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.executed.append(report.nodeid)
recorder = Recorder()
try:
    with gate._isolation_guard(deny_process=True) as observed:
        audit = observed
        pytest_returncode = int(pytest.main(["-q", "-ra", "--noconftest", f"--rootdir={os.environ['TRKH_B17_PYTEST_ROOTDIR']}", *test_paths], plugins=[recorder]))
except BaseException as caught:
    error = {"type": type(caught).__name__, "message": str(caught)}
payload = {
    "schema_version": 1,
    "guard_root": str(gate.DATASET_ROOT),
    "test_paths": test_paths,
    "audit": audit,
    "collected_nodeids": recorder.collected,
    "executed_nodeids": recorder.executed,
    "pytest_returncode": pytest_returncode,
    "error": error,
}
encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\\n").encode("utf-8")
temporary = handshake.with_name(f".{handshake.name}.{uuid.uuid4().hex}.tmp")
temporary.write_bytes(encoded)
temporary.replace(handshake)
passed = error is None and pytest_returncode == 0 and not any(audit.values())
raise SystemExit(0 if passed else 86)
"""


def _run_guarded_pytest(
    test_paths: Sequence[str],
    *,
    guard_root: Path | None = None,
    pytest_root: Path | None = None,
) -> dict[str, object]:
    paths = [str(value) for value in test_paths]
    if not paths or any(not value for value in paths) or len(paths) != len(set(paths)):
        raise ValueError("B17 focused test paths must be non-empty and unique")
    bootstrap = _focused_test_bootstrap()
    bootstrap_sha256 = hashlib.sha256(bootstrap.encode("utf-8")).hexdigest()
    requested_root = (DATASET_ROOT if guard_root is None else guard_root).expanduser()
    root = Path(os.path.abspath(os.fspath(requested_root)))
    requested_pytest_root = (_root() if pytest_root is None else pytest_root).expanduser()
    resolved_pytest_root = Path(os.path.abspath(os.fspath(requested_pytest_root)))
    with tempfile.TemporaryDirectory(prefix="trkh_b17_pytest_guard_") as temporary:
        handshake = Path(temporary) / "child_guard.json"
        command = [sys.executable, "-c", bootstrap, str(handshake), *paths]
        environment = {
            **os.environ,
            **OFFLINE_ENV,
            "TRKH_B17_CHILD_GUARD_ROOT": str(root),
            "TRKH_B17_PYTEST_ROOTDIR": str(resolved_pytest_root),
            "PYTEST_ADDOPTS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TRKH_B17_IN_GUARDED_PYTEST": "1",
        }
        result = subprocess.run(
            command,
            cwd=_root(),
            env=environment,
            capture_output=True,
            text=True,
        )
        if not handshake.is_file():
            child = {
                "schema_version": 0,
                "guard_root": str(root),
                "test_paths": paths,
                "audit": {"dataset_attempts": ["missing_handshake"], "network_attempts": [], "process_attempts": []},
                "collected_nodeids": [],
                "executed_nodeids": [],
                "pytest_returncode": 86,
                "error": {"type": "MissingHandshake", "message": "child guard did not publish"},
            }
        else:
            child = json.loads(
                handshake.read_text(encoding="utf-8"),
                object_pairs_hook=_no_duplicate_keys,
            )
    output = result.stdout + "\n" + result.stderr
    counts = {
        name: sum(int(value) for value in re.findall(rf"\b(\d+) {name}\b", output))
        for name in ("passed", "skipped", "xfailed", "xpassed")
    }
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "test_paths": paths,
        "bootstrap_sha256": bootstrap_sha256,
        "returncode": int(result.returncode),
        "pytest_returncode": child.get("pytest_returncode"),
        "passed_count": counts["passed"],
        "skipped": counts["skipped"],
        "xfailed": counts["xfailed"],
        "xpassed": counts["xpassed"],
        "child_guard": child,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _focused_tests() -> dict[str, object]:
    evidence = _run_guarded_pytest(FOCUSED_TEST_PATHS)
    child = evidence["child_guard"]
    evidence["passed"] = bool(
        evidence["returncode"] == 0
        and evidence["pytest_returncode"] == 0
        and evidence["passed_count"] == EXPECTED_FOCUSED_TEST_COUNT
        and evidence["skipped"] == 0
        and evidence["xfailed"] == 0
        and evidence["xpassed"] == 0
        and child
        == {
            "schema_version": 1,
            "guard_root": os.path.abspath(os.fspath(DATASET_ROOT)),
            "test_paths": list(FOCUSED_TEST_PATHS),
            "audit": {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
            "collected_nodeids": child["collected_nodeids"],
            "executed_nodeids": child["executed_nodeids"],
            "pytest_returncode": 0,
            "error": None,
        }
        and len(child["collected_nodeids"]) == EXPECTED_FOCUSED_TEST_COUNT
        and child["executed_nodeids"] == child["collected_nodeids"]
    )
    if evidence["passed"] is not True:
        raise RuntimeError(f"B17 guarded focused tests failed/skipped: {evidence}")
    return evidence


def _end_rehash(
    start: Mapping[str, object], student_path: Path, teacher_path: Path
) -> dict[str, object]:
    _, student = _resolve_asset(STUDENT_LOCK, student_path)
    _, teacher = _resolve_asset(DINO_LOCK, teacher_path)
    start_weights = start["weights"]
    student["offline_cache_only"] = start_weights["student"]["offline_cache_only"]
    teacher["offline_cache_only"] = start_weights["teacher"]["offline_cache_only"]
    observed = {
        "git": _git_contract(),
        "source_hashes": _source_hashes(),
        "runtime": _runtime_contract(),
        "timm_sources": _timm_contract(),
        "weights": {"student": student, "teacher": teacher},
    }
    if observed != dict(start):
        raise RuntimeError(f"B17 start/end rehash drifted: start={start}, end={observed}")
    return observed


def _exact_keys(value: object, names: Sequence[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == set(names)


def _hex(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _true_gate(value: object, names: Sequence[str]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("passed") is True
        and _exact_keys(value.get("checks"), names)
        and all(value["checks"][name] is True for name in names)
    )


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _latency_payload_ok(value: object) -> bool:
    try:
        if not _exact_keys(
            value,
            (
                "provider",
                "intra_op_threads",
                "inter_op_threads",
                "execution_mode",
                "warmups",
                "trials",
                "iterations_per_trial",
                "order_rule",
                "pre_benchmark_parity",
                "arms",
                "ort_to_onnx_ratio",
            ),
        ):
            return False
        summaries = value["arms"]
        if set(summaries) != {"onnx", "ort"}:
            return False
        recomputed: dict[str, tuple[float, float]] = {}
        for name, summary in summaries.items():
            if not _exact_keys(summary, ("trials_ms", "median_ms", "p95_ms")):
                return False
            trials = summary["trials_ms"]
            if (
                not isinstance(trials, list)
                or len(trials) != 5
                or any(not isinstance(row, list) or len(row) != 100 for row in trials)
                or any(not _finite_number(item) for row in trials for item in row)
            ):
                return False
            array = np.asarray(trials, dtype=np.float64)
            if array.shape != (5, 100) or not np.isfinite(array).all() or np.any(array <= 0):
                return False
            recomputed[name] = (
                float(np.median(array)),
                float(np.quantile(array, 0.95, method="linear")),
            )
            if (
                not _finite_number(summary["median_ms"])
                or not _finite_number(summary["p95_ms"])
                or summary["median_ms"] != recomputed[name][0]
                or summary["p95_ms"] != recomputed[name][1]
            ):
                return False
        return bool(
            value["provider"] == "CPUExecutionProvider"
            and value["intra_op_threads"] == 4
            and value["inter_op_threads"] == 1
            and value["execution_mode"] == "ORT_SEQUENTIAL"
            and value["warmups"] == 15
            and value["trials"] == 5
            and value["iterations_per_trial"] == 100
            and value["order_rule"] == "alternate by (trial + iteration) modulo 2"
            and _ort_artifact_parity_ok(value["pre_benchmark_parity"])
            and _exact_keys(value["ort_to_onnx_ratio"], ("median", "p95"))
            and _finite_number(value["ort_to_onnx_ratio"]["median"])
            and _finite_number(value["ort_to_onnx_ratio"]["p95"])
            and value["ort_to_onnx_ratio"]["median"]
            == recomputed["ort"][0] / recomputed["onnx"][0]
            and value["ort_to_onnx_ratio"]["p95"]
            == recomputed["ort"][1] / recomputed["onnx"][1]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _torch_ort_parity_ok(value: object) -> bool:
    try:
        return bool(
            _exact_keys(
                value,
                (
                    "passed",
                    "providers",
                    "intra_op_threads",
                    "inter_op_threads",
                    "samples",
                    "max_abs_error",
                    "mean_abs_error",
                    "argmax_mismatches",
                ),
            )
            and value["passed"] is True
            and value["providers"] == ["CPUExecutionProvider"]
            and value["intra_op_threads"] == 4
            and value["inter_op_threads"] == 1
            and value["samples"] == 64
            and _finite_number(value["max_abs_error"])
            and 0 <= value["max_abs_error"] <= MAX_PARITY_ERROR
            and _finite_number(value["mean_abs_error"])
            and 0 <= value["mean_abs_error"] <= value["max_abs_error"]
            and value["argmax_mismatches"] == 0
        )
    except (KeyError, TypeError):
        return False


def _ort_artifact_parity_ok(value: object) -> bool:
    try:
        return bool(
            _exact_keys(
                value,
                ("passed", "samples", "max_abs_error", "argmax_mismatches"),
            )
            and value["passed"] is True
            and value["samples"] in (1, 64)
            and _finite_number(value["max_abs_error"])
            and 0 <= value["max_abs_error"] <= MAX_PARITY_ERROR
            and value["argmax_mismatches"] == 0
        )
    except (KeyError, TypeError):
        return False


def _deployment_payload_ok(value: object, *, stock: bool) -> bool:
    try:
        if not _true_gate(value, DEPLOYMENT_CHECKS) or not _exact_keys(
            value,
            (
                "arm",
                "parameters",
                "onnx",
                "topology_sha256",
                "parity",
                "mobile",
                "ort",
                "latency",
                "forbidden_identifiers",
                "host_ratio_is_gate",
                "checks",
                "passed",
            ),
        ):
            return False
        onnx, ort_format, latency = value["onnx"], value["ort"], value["latency"]
        ratios = latency["ort_to_onnx_ratio"]
        return bool(
            value["arm"] in {"stock", "control", "candidate"}
            and (not stock or value["arm"] == "stock")
            and value["parameters"] == STOCK_PARAMETERS
            and value["forbidden_identifiers"] == []
            and _exact_keys(
                onnx,
                (
                    "sha256",
                    "size_bytes",
                    "opset",
                    "input_name",
                    "input_shape",
                    "output_name",
                    "output_shape",
                    "topology_sha256",
                ),
            )
            and _hex(onnx["sha256"])
            and onnx["topology_sha256"] == value["topology_sha256"]
            and onnx["opset"] == 17
            and onnx["input_name"] == "x"
            and onnx["input_shape"] == [1, 3, 224, 224]
            and onnx["output_name"] == "logits"
            and onnx["output_shape"] == [1, 5]
            and type(onnx["size_bytes"]) is int
            and onnx["size_bytes"] > 0
            and onnx["size_bytes"] <= MAX_ONNX_BYTES
            and _torch_ort_parity_ok(value["parity"])
            and _exact_keys(
                value["mobile"],
                (
                    "prebuilt_mobile_package_supported",
                    "nnapi_or_coreml_may_help",
                    "checker_is_structural_not_device_certification",
                    "log",
                ),
            )
            and type(value["mobile"]["prebuilt_mobile_package_supported"]) is bool
            and type(value["mobile"]["nnapi_or_coreml_may_help"]) is bool
            and value["mobile"]["checker_is_structural_not_device_certification"] is True
            and isinstance(value["mobile"]["log"], str)
            and "ORT Mobile prebuilt-package compatibility" in value["mobile"]["log"]
            and "ORT Mobile NNAPI/CoreML usability heuristic" in value["mobile"]["log"]
            and _exact_keys(
                ort_format,
                (
                    "passed",
                    "ort_sha256",
                    "ort_size_bytes",
                    "config_sha256",
                    "optimization_style",
                    "target_platform",
                    "type_reduction",
                    "providers",
                    "parity",
                ),
            )
            and ort_format["passed"] is True
            and _hex(ort_format["ort_sha256"])
            and _hex(ort_format["config_sha256"])
            and ort_format["optimization_style"] == "Fixed"
            and ort_format["target_platform"] == "arm"
            and ort_format["type_reduction"] is True
            and ort_format["providers"] == ["CPUExecutionProvider"]
            and _ort_artifact_parity_ok(ort_format["parity"])
            and ort_format["parity"]["samples"] == 64
            and type(ort_format["ort_size_bytes"]) is int
            and ort_format["ort_size_bytes"] > 0
            and ort_format["ort_size_bytes"] <= MAX_ORT_BYTES
            and ort_format["ort_size_bytes"] <= MAX_ORT_SOURCE_RATIO * onnx["size_bytes"]
            and _latency_payload_ok(latency)
            and (not stock or ratios["median"] <= MAX_FORMAT_LATENCY_RATIO)
            and (not stock or ratios["p95"] <= MAX_FORMAT_LATENCY_RATIO)
            and value["host_ratio_is_gate"] is stock
            and _hex(value["topology_sha256"])
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _role_payload_ok(value: object, names: Sequence[str]) -> bool:
    try:
        return bool(
            _exact_keys(value, ("names", "present", "finite", "nonzero"))
            and value["names"] == list(names)
            and value["present"] is True
            and value["finite"] is True
            and value["nonzero"] is True
        )
    except (KeyError, TypeError):
        return False


def _relation_gradient_payload_ok(value: object, *, candidate: bool) -> bool:
    try:
        return bool(
            _exact_keys(
                value,
                (
                    "loss",
                    "p",
                    "d",
                    "candidate_centered_d_nonzero",
                    "control_d_spatially_equal",
                    "s3_gradient_nonzero",
                    "teacher_gradient_absent",
                ),
            )
            and _finite_number(value["loss"])
            and value["loss"] >= 0
            and _role_payload_ok(value["p"], ("factor_p",))
            and _role_payload_ok(value["d"], ("factor_d",))
            and type(value["candidate_centered_d_nonzero"]) is bool
            and type(value["control_d_spatially_equal"]) is bool
            and (
                value["candidate_centered_d_nonzero"] is True
                if candidate
                else value["control_d_spatially_equal"] is True
            )
            and value["s3_gradient_nonzero"] is True
            and value["teacher_gradient_absent"] is True
        )
    except (KeyError, TypeError):
        return False


def _total_gradient_payload_ok(value: object, *, active: bool) -> bool:
    try:
        expected_roles = {
            "head": ("backbone.head.weight", "backbone.head.bias"),
            "head_dist": ("backbone.head_dist.weight", "backbone.head_dist.bias"),
        }
        if active:
            expected_roles.update({"p": ("factor_p",), "d": ("factor_d",)})
        return bool(
            _exact_keys(
                value,
                ("ce", "relation", "total", "roles", "s3_gradient_nonzero", "teacher_gradient_absent"),
            )
            and all(_finite_number(value[name]) and value[name] >= 0 for name in ("ce", "relation", "total"))
            and math.isclose(
                value["total"], value["ce"] + 0.10 * value["relation"], rel_tol=1.0e-6, abs_tol=1.0e-6
            )
            and set(value["roles"]) == set(expected_roles)
            and all(_role_payload_ok(value["roles"][name], names) for name, names in expected_roles.items())
            and value["s3_gradient_nonzero"] is True
            and value["teacher_gradient_absent"] is True
        )
    except (KeyError, TypeError):
        return False


def _mechanism_payload_ok(value: object) -> bool:
    try:
        return bool(
            _true_gate(value, MECHANISM_CHECKS)
            and _exact_keys(
                value,
                (
                    "counts",
                    "relation_only_gradients",
                    "total_objective_gradients",
                    "float64_oracle_max_abs",
                    "fold_parity_max_abs",
                    "w0_b0_sha256",
                    "checks",
                    "passed",
                ),
            )
            and value["counts"]
            == {
                "stock": {"total": STOCK_PARAMETERS, "trainable": STOCK_TRAINABLE},
                "control": {"total": ACTIVE_PARAMETERS, "trainable": ACTIVE_TRAINABLE},
                "candidate": {"total": ACTIVE_PARAMETERS, "trainable": ACTIVE_TRAINABLE},
            }
            and set(value["float64_oracle_max_abs"]) == {"control", "candidate"}
            and all(_finite_number(error) and 0 <= error <= MAX_ORACLE_ERROR for error in value["float64_oracle_max_abs"].values())
            and set(value["fold_parity_max_abs"]) == {"control", "candidate"}
            and all(
                set(arm) == {"pre_bn", "features", "logits"}
                and all(_finite_number(error) and 0 <= error <= MAX_FOLD_ERROR for error in arm.values())
                for arm in value["fold_parity_max_abs"].values()
            )
            and set(value["relation_only_gradients"]) == {"control", "candidate"}
            and _relation_gradient_payload_ok(value["relation_only_gradients"]["control"], candidate=False)
            and _relation_gradient_payload_ok(value["relation_only_gradients"]["candidate"], candidate=True)
            and set(value["total_objective_gradients"]) == {"stock", "control", "candidate"}
            and _total_gradient_payload_ok(value["total_objective_gradients"]["stock"], active=False)
            and _total_gradient_payload_ok(value["total_objective_gradients"]["control"], active=True)
            and _total_gradient_payload_ok(value["total_objective_gradients"]["candidate"], active=True)
            and set(value["w0_b0_sha256"]) == {"stock", "control", "candidate"}
            and all(_hex(digest) for digest in value["w0_b0_sha256"].values())
        )
    except (KeyError, TypeError):
        return False


def _folded_payload_ok(value: object, stock: Mapping[str, object]) -> bool:
    try:
        arms = value["arms"]
        return bool(
            _true_gate(value, FOLDED_CHECKS)
            and _exact_keys(value, ("stock_topology_sha256", "arms", "checks", "passed"))
            and set(arms) == {"control", "candidate"}
            and value["stock_topology_sha256"] == stock["topology_sha256"]
            and all(_deployment_payload_ok(arm, stock=False) for arm in arms.values())
            and all(
                arm["topology_sha256"] == stock["topology_sha256"]
                and abs(arm["onnx"]["size_bytes"] - stock["onnx"]["size_bytes"])
                <= MAX_FOLDED_SIZE_DELTA
                and abs(arm["ort"]["ort_size_bytes"] - stock["ort"]["ort_size_bytes"])
                <= MAX_FOLDED_SIZE_DELTA
                for arm in arms.values()
            )
        )
    except (KeyError, TypeError):
        return False


def _cuda_gradient_payload_ok(value: object, *, active: bool) -> bool:
    try:
        expected_roles = {
            "head": ("backbone.head.weight", "backbone.head.bias"),
            "head_dist": ("backbone.head_dist.weight", "backbone.head_dist.bias"),
        }
        if active:
            expected_roles.update({"p": ("factor_p",), "d": ("factor_d",)})
        all_gradients = value["all"]
        return bool(
            _exact_keys(value, ("all", "roles", "s3_nonzero", "logits_dtype", "s3_dtype"))
            and _exact_keys(all_gradients, ("tensors", "missing", "nonfinite", "all_present_finite"))
            and type(all_gradients["tensors"]) is int
            and all_gradients["tensors"] > 0
            and all_gradients["missing"] == []
            and all_gradients["nonfinite"] == []
            and all_gradients["all_present_finite"] is True
            and set(value["roles"]) == set(expected_roles)
            and all(_role_payload_ok(value["roles"][name], names) for name, names in expected_roles.items())
            and value["s3_nonzero"] is True
            and value["logits_dtype"] == "torch.bfloat16"
            and value["s3_dtype"] == "torch.bfloat16"
        )
    except (KeyError, TypeError):
        return False


def _optimizer_payload_ok(value: object) -> bool:
    try:
        return bool(
            _exact_keys(
                value,
                ("trainable_tensors", "state_tensors", "parameter_set_exact", "finite", "shapes_exact", "steps_one"),
            )
            and type(value["trainable_tensors"]) is int
            and value["trainable_tensors"] > 0
            and value["state_tensors"] == value["trainable_tensors"]
            and value["parameter_set_exact"] is True
            and value["finite"] is True
            and value["shapes_exact"] is True
            and value["steps_one"] is True
        )
    except (KeyError, TypeError):
        return False


def _cuda_payload_ok(value: object) -> bool:
    try:
        return bool(
            _true_gate(value, CUDA_CHECKS)
            and _exact_keys(
                value,
                (
                    "settings",
                    "exposure_sha256",
                    "teacher",
                    "runtime_state",
                    "losses",
                    "gradients",
                    "optimizer_states",
                    "w0_b0_before",
                    "w0_b0_after",
                    "memory",
                    "checks",
                    "passed",
                ),
            )
            and value["settings"]
            == {
                "batch": 16,
                "teacher_precision": "fp32_inference_once",
                "student_precision": "bf16_autocast",
                "arm_order": ["stock", "control", "candidate"],
                "optimizer": "AdamW_one_real_step_each",
                "gradient_clip": 0.7,
                "synthetic_ce_weights": list(SYNTHETIC_CE_WEIGHTS),
                "tf32": False,
            }
            and value["teacher"]
            == {
                "calls": 1,
                "tokens_shape": [16, 261, 384],
                "raw_map_shape": [16, 384, 16, 16],
                "target_shape": [16, 84],
                "target_dtype": "torch.float32",
                "target_requires_grad": False,
                "parameter_gradients_present": 0,
            }
            and value["runtime_state"]
            == {
                "deterministic_algorithms": True,
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            }
            and set(value["losses"]) == {"stock", "control", "candidate"}
            and all(
                _exact_keys(loss, ("ce", "relation", "total"))
                and all(_finite_number(loss[name]) and loss[name] >= 0 for name in loss)
                and math.isclose(loss["total"], loss["ce"] + 0.10 * loss["relation"], rel_tol=1.0e-6, abs_tol=1.0e-6)
                for loss in value["losses"].values()
            )
            and set(value["gradients"]) == {"stock", "control", "candidate"}
            and _cuda_gradient_payload_ok(value["gradients"]["stock"], active=False)
            and _cuda_gradient_payload_ok(value["gradients"]["control"], active=True)
            and _cuda_gradient_payload_ok(value["gradients"]["candidate"], active=True)
            and set(value["optimizer_states"]) == {"stock", "control", "candidate"}
            and all(_optimizer_payload_ok(item) for item in value["optimizer_states"].values())
            and value["w0_b0_before"] == value["w0_b0_after"]
            and set(value["w0_b0_before"]) == {"stock", "control", "candidate"}
            and all(_hex(digest) for digest in value["w0_b0_before"].values())
            and _exact_keys(
                value["memory"],
                ("baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved", "limit_allocated"),
            )
            and all(type(value["memory"][name]) is int and value["memory"][name] >= 0 for name in ("baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved"))
            and value["memory"]["peak_allocated"] >= value["memory"]["baseline_allocated"]
            and value["memory"]["peak_allocated"] <= MAX_CUDA_ALLOCATED_BYTES
            and value["memory"]["limit_allocated"] == MAX_CUDA_ALLOCATED_BYTES
            and _hex(value["exposure_sha256"])
        )
    except (KeyError, TypeError):
        return False


def _strict_load_payload_ok(value: object) -> bool:
    try:
        student, teacher = value["student"], value["teacher"]
        head = student["head_reset"]
        return bool(
            _exact_keys(value, ("student", "teacher", "caller_rng_preserved"))
            and value["caller_rng_preserved"] is True
            and _exact_keys(
                student,
                (
                    "runtime_class",
                    "stage_channels",
                    "parameters",
                    "heads",
                    "distilled_training",
                    "nonzero_dropout_or_path",
                    "strict_load",
                    "head_reset",
                    "asset_keys",
                ),
            )
            and student["runtime_class"] == "timm.models.swiftformer.SwiftFormer"
            and student["stage_channels"] == [48, 56, 112, 220]
            and student["parameters"] == STOCK_PARAMETERS
            and student["heads"] == [[220, 5], [220, 5]]
            and student["distilled_training"] is False
            and student["nonzero_dropout_or_path"] == []
            and student["strict_load"] is True
            and student["asset_keys"] == 316
            and _exact_keys(
                head,
                ("seed", "initializer", "bias_zero", "state_sha256", "caller_rng_preserved"),
            )
            and head["seed"] == SEED
            and head["initializer"] == "timm_trunc_normal_0p02_head_then_head_dist"
            and head["bias_zero"] is True
            and head["state_sha256"] == EXPECTED_HEAD_STATE_SHA256
            and head["caller_rng_preserved"] is True
            and _exact_keys(
                teacher,
                ("runtime_class", "features", "prefix_tokens", "grid", "all_frozen", "strict_load", "asset_keys"),
            )
            and teacher["runtime_class"] == "timm.models.eva.Eva"
            and teacher["features"] == DINO_CHANNELS
            and teacher["prefix_tokens"] == DINO_PREFIX_TOKENS
            and teacher["grid"] == [16, 16]
            and teacher["all_frozen"] is True
            and teacher["strict_load"] is True
            and teacher["asset_keys"] == 162
        )
    except (KeyError, TypeError):
        return False


def _focused_payload_ok(value: object) -> bool:
    try:
        child = value["child_guard"]
        output = value["stdout"] + "\n" + value["stderr"]
        counts = {
            name: sum(int(item) for item in re.findall(rf"\b(\d+) {name}\b", output))
            for name in ("passed", "skipped", "xfailed", "xpassed")
        }
        expected_child = {
            "schema_version": 1,
            "guard_root": os.path.abspath(os.fspath(DATASET_ROOT)),
            "test_paths": list(FOCUSED_TEST_PATHS),
            "audit": {
                "dataset_attempts": [],
                "network_attempts": [],
                "process_attempts": [],
            },
            "collected_nodeids": child["collected_nodeids"],
            "executed_nodeids": child["executed_nodeids"],
            "pytest_returncode": 0,
            "error": None,
        }
        return bool(
            _exact_keys(
                value,
                (
                    "python_executable",
                    "test_paths",
                    "bootstrap_sha256",
                    "returncode",
                    "pytest_returncode",
                    "passed_count",
                    "skipped",
                    "xfailed",
                    "xpassed",
                    "child_guard",
                    "stdout",
                    "stderr",
                    "passed",
                ),
            )
            and value["python_executable"] == str(Path(sys.executable).resolve())
            and value["test_paths"] == list(FOCUSED_TEST_PATHS)
            and value["bootstrap_sha256"]
            == hashlib.sha256(_focused_test_bootstrap().encode("utf-8")).hexdigest()
            and value["returncode"] == 0
            and value["pytest_returncode"] == 0
            and type(value["passed_count"]) is int
            and value["passed_count"] == EXPECTED_FOCUSED_TEST_COUNT
            and value["passed_count"] == counts["passed"]
            and value["skipped"] == counts["skipped"] == 0
            and value["xfailed"] == counts["xfailed"] == 0
            and value["xpassed"] == counts["xpassed"] == 0
            and child == expected_child
            and len(child["collected_nodeids"]) == EXPECTED_FOCUSED_TEST_COUNT
            and child["executed_nodeids"] == child["collected_nodeids"]
            and value["passed"] is True
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _asset_payload_ok(value: object, lock: AssetLock) -> bool:
    try:
        return bool(
            _exact_keys(
                value,
                ("role", "repo_id", "revision", "filename", "path", "bytes", "sha256", "license", "offline_cache_only"),
            )
            and value["role"] == lock.role
            and value["repo_id"] == lock.repo_id
            and value["revision"] == lock.revision
            and value["filename"] == lock.filename
            and isinstance(value["path"], str)
            and bool(value["path"])
            and value["bytes"] == lock.byte_count
            and value["sha256"] == lock.sha256
            and value["license"] == lock.license
            and type(value["offline_cache_only"]) is bool
        )
    except (KeyError, TypeError):
        return False


def _payload_checks(payload: Mapping[str, object]) -> dict[str, bool]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
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
        "stock_deployment",
        "mechanism",
        "folded_deployment",
        "cuda",
        "isolation",
        "offline",
        "end_rehash",
        "permissions",
        "authorization",
        "checks",
        "passed",
    }
    git = payload.get("git")
    sources = payload.get("source_hashes")
    stock = payload.get("stock_deployment")
    try:
        sources_exact = bool(
            isinstance(sources, Mapping)
            and set(sources) == set(_source_paths())
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values())
            and all(sources[name] == value for name, value in SOURCE_LOCKS.items())
        )
        end_exact = payload.get("end_rehash") == {
            "git": git,
            "source_hashes": sources,
            "runtime": payload.get("runtime"),
            "timm_sources": payload.get("timm_sources"),
            "weights": payload.get("weights"),
        }
        strict_ok = _strict_load_payload_ok(payload.get("strict_load"))
    except (KeyError, TypeError):
        sources_exact = end_exact = strict_ok = False
    return {
        "top_level_exact": set(payload) == expected_keys,
        "identity_exact": payload.get("schema_version") == 1
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("protocol_sha256") == PROTOCOL_SHA256
        and payload.get("mode") == "offline_label_free_stock_first_preflight"
        and _finite_number(payload.get("created_at_unix"))
        and payload["created_at_unix"] > 0,
        "git_exact": _exact_keys(
            git, ("head", "branch", "status", "clean", "head_is_commit")
        )
        and git.get("branch") == EXPECTED_BRANCH
        and git.get("clean") is True
        and git.get("status") == ""
        and git.get("head_is_commit") is True
        and re.fullmatch(r"[0-9a-f]{40}", str(git.get("head", ""))) is not None,
        "source_hashes_exact": sources_exact,
        "runtime_exact": payload.get("runtime") == EXPECTED_RUNTIME
        and _exact_keys(payload.get("timm_sources"), ("paths", "sha256"))
        and _exact_keys(
            payload["timm_sources"].get("paths"), EXPECTED_TIMM_HASHES.keys()
        )
        and all(
            isinstance(path, str) and bool(path)
            for path in payload["timm_sources"]["paths"].values()
        )
        and payload["timm_sources"].get("sha256") == EXPECTED_TIMM_HASHES,
        "device_exact": payload.get("device")
        == {"requested": "cuda", "resolved": "cuda:0", **EXPECTED_CUDA},
        "weights_exact": _exact_keys(payload.get("weights"), ("student", "teacher"))
        and _asset_payload_ok(payload["weights"]["student"], STUDENT_LOCK)
        and _asset_payload_ok(payload["weights"]["teacher"], DINO_LOCK),
        "strict_load_exact": strict_ok,
        "focused_tests_exact": _focused_payload_ok(payload.get("focused_tests")),
        "stock_deployment_exact": _deployment_payload_ok(stock, stock=True),
        "mechanism_exact": _mechanism_payload_ok(payload.get("mechanism")),
        "folded_deployment_exact": isinstance(stock, Mapping)
        and _folded_payload_ok(payload.get("folded_deployment"), stock),
        "cuda_exact": _cuda_payload_ok(payload.get("cuda")),
        "isolation_exact": payload.get("isolation")
        == {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
        "offline_exact": payload.get("offline")
        == {"environment": OFFLINE_ENV, "cublas_workspace_config": ":4096:8"},
        "end_rehash_exact": end_exact,
        "permission_exact": payload.get("permissions") == SUCCESS_PERMISSIONS
        and payload.get("authorization") == AUTHORIZATION,
        "top_checks_exact": isinstance(payload.get("checks"), Mapping)
        and set(payload["checks"]) == set(TOP_CHECKS)
        and all(payload["checks"][name] is True for name in TOP_CHECKS),
        "passed_exact": payload.get("passed") is True,
    }


def _validated_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    runs = (_root() / "runs").resolve()
    if output.parent != runs or not output.name.startswith(OUTPUT_PREFIX):
        raise ValueError("B17 output must be a direct runs child with the locked prefix")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Refuse to overwrite B17 output: {output}")
    return output


def _failure_path(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    runs = (_root() / "runs").resolve()
    if requested.parent != runs:
        raise ValueError("B17 failure output requires a canonical runs child")
    suffix = requested.name[len(OUTPUT_PREFIX) :] if requested.name.startswith(OUTPUT_PREFIX) else requested.name
    candidate = runs / f"{FAILURE_PREFIX}{suffix}"
    if candidate.exists() or candidate.with_name(candidate.name + ".partial").exists():
        candidate = runs / f"{candidate.name}_{uuid.uuid4().hex}"
    return candidate


def _write_failure(requested: Path, stage: str, error: BaseException) -> dict[str, str]:
    trace = traceback.format_exc()
    gate_evidence = getattr(error, "b17_gate_evidence", None)
    gate_evidence_sha256 = (
        hashlib.sha256(_canonical_bytes(gate_evidence)).hexdigest()
        if isinstance(gate_evidence, Mapping)
        else None
    )
    return _publish(
        _failure_path(requested),
        "failure.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "mode": "offline_label_free_preflight_failure",
            "created_at_unix": time.time(),
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback_sha256": hashlib.sha256(trace.encode()).hexdigest(),
            "gate_evidence": gate_evidence,
            "gate_evidence_sha256": gate_evidence_sha256,
            "permissions": dict(NO_ACCESS_PERMISSIONS),
            "passed": False,
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked offline label-free B17 SurfaceFold-XS preflight"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--student-weight", type=Path, default=None)
    parser.add_argument("--dino-weight", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_preflight(args: argparse.Namespace) -> dict[str, object]:
    output = _validated_output(args.output_dir)
    stage = "offline_environment"
    try:
        offline = _offline_environment()
        with _isolation_guard() as isolation:
            stage = "start_contract"
            git = _git_contract()
            if not (
                git["branch"] == EXPECTED_BRANCH
                and git["clean"] is True
                and git["head_is_commit"] is True
            ):
                raise RuntimeError(f"B17 requires a clean committed canonical branch: {git}")
            sources = _source_hashes()
            runtime = _runtime_contract()
            timm_sources = _timm_contract()
            device, device_evidence = _device_contract(args.device)

            stage = "offline_assets"
            student_path, student_asset = _resolve_asset(STUDENT_LOCK, args.student_weight)
            teacher_path, teacher_asset = _resolve_asset(DINO_LOCK, args.dino_weight)
            weights = {"student": student_asset, "teacher": teacher_asset}
            start = {
                "git": git,
                "source_hashes": sources,
                "runtime": runtime,
                "timm_sources": timm_sources,
                "weights": weights,
            }

            stage = "focused_tests"
            focused = _focused_tests()
            stage = "strict_load"
            base_student, stock, teacher, strict_load = _strict_load_models(
                student_path, teacher_path
            )
            stage = "stock_deployment_first"
            stock_deployment = _stock_deployment_contract(stock)
            stage = "construct_active_arms_after_stock"
            bundle = build_active_arms_after_stock(base_student, stock)
            stage = "mechanism_after_stock"
            mechanism = _mechanism_contract(bundle)
            if mechanism["passed"] is not True:
                raise RuntimeError(f"B17 mechanism gate failed: {mechanism['checks']}")
            stage = "folded_deployment"
            folded_deployment = _folded_deployment_contract(bundle, stock_deployment)
            if folded_deployment["passed"] is not True:
                raise RuntimeError(
                    f"B17 folded deployment gate failed: {folded_deployment['checks']}"
                )
            stage = "cuda_batch16"
            cuda = _cuda_contract(bundle, teacher, device)
            if cuda["passed"] is not True:
                raise RuntimeError(f"B17 CUDA gate failed: {cuda['checks']}")
            stage = "end_rehash"
            end_rehash = _end_rehash(start, student_path, teacher_path)
            if any(isolation.values()):
                raise RuntimeError(f"B17 isolation attempts observed: {isolation}")
            checks = {
                "clean_committed_canonical_branch": True,
                "bound_sources_exact": all(sources[name] == value for name, value in SOURCE_LOCKS.items()),
                "runtime_and_timm_exact": runtime == EXPECTED_RUNTIME
                and timm_sources["sha256"] == EXPECTED_TIMM_HASHES,
                "offline_assets_strict": strict_load["student"]["strict_load"] is True
                and strict_load["teacher"]["strict_load"] is True,
                "focused_tests_no_skip": focused["passed"] is True,
                "stock_deployment_first": stock_deployment["passed"] is True,
                "mechanism_and_gradient_contract": mechanism["passed"] is True,
                "folded_topology_contract": folded_deployment["passed"] is True,
                "cuda_batch16_contract": cuda["passed"] is True,
                "no_dataset_or_network_access": not any(isolation.values()),
                "start_end_rehash_exact": end_rehash == start,
                "train_only_authorization": SUCCESS_PERMISSIONS["formal_train_permission"] is True
                and AUTHORIZATION["validation_permission"] is False
                and AUTHORIZATION["test_permission"] is False,
            }
            payload: dict[str, object] = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": PROTOCOL_SHA256,
                "mode": "offline_label_free_stock_first_preflight",
                "created_at_unix": time.time(),
                "git": git,
                "source_hashes": sources,
                "runtime": runtime,
                "timm_sources": timm_sources,
                "device": device_evidence,
                "weights": weights,
                "strict_load": strict_load,
                "focused_tests": focused,
                "stock_deployment": stock_deployment,
                "mechanism": mechanism,
                "folded_deployment": folded_deployment,
                "cuda": cuda,
                "isolation": isolation,
                "offline": offline,
                "end_rehash": end_rehash,
                "permissions": dict(SUCCESS_PERMISSIONS),
                "authorization": dict(AUTHORIZATION),
                "checks": checks,
                "passed": all(checks.values()),
            }
            structure = _payload_checks(payload)
            if not all(structure.values()):
                raise RuntimeError(f"B17 payload self-validation failed: {structure}")
            stage = "final_rehash"
            if _end_rehash(start, student_path, teacher_path) != end_rehash:
                raise RuntimeError("B17 final rehash changed")
            if any(isolation.values()):
                raise RuntimeError("B17 isolation state changed before publish")
            stage = "atomic_publish"
            _publish(output, "preflight.json", payload)
            return payload
    except BaseException as error:
        try:
            failure = _write_failure(output, stage, error)
        except BaseException as write_error:
            failure = {"failure_write_error": repr(write_error)}
        setattr(error, "b17_failure_artifact", failure)
        raise


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"B17 artifact contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _asset_identity(value: Mapping[str, object]) -> dict[str, object]:
    keys = ("role", "repo_id", "revision", "filename", "bytes", "sha256", "license")
    return {key: value.get(key) for key in keys}


def validate_accepted_preflight(
    artifact: Path,
    expected_sha256: str,
    *,
    student_weight: Path | None = None,
    dino_weight: Path | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    path = artifact.expanduser().resolve(strict=True)
    runs = (_root() / "runs").resolve(strict=True)
    if (
        path.name != "preflight.json"
        or path.parent.parent != runs
        or not path.parent.name.startswith(OUTPUT_PREFIX)
    ):
        raise ValueError("Accepted B17 preflight path is not canonical")
    if {child.name for child in path.parent.iterdir()} != {
        "preflight.json",
        "preflight.sha256",
    }:
        raise ValueError("Accepted B17 preflight root contains unexpected files")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None or digest != expected_sha256.casefold():
        raise ValueError("Accepted B17 preflight SHA-256 mismatch")
    sidecar = path.with_name("preflight.sha256").read_bytes()
    if sidecar != f"{digest}\n".encode("ascii"):
        raise ValueError("Accepted B17 preflight sidecar mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Accepted B17 preflight JSON is invalid") from error
    if not isinstance(payload, Mapping) or raw != _canonical_bytes(payload):
        raise ValueError("Accepted B17 preflight is not canonical JSON")
    checks = _payload_checks(payload)
    git = _git_contract()
    sources = _source_hashes()
    runtime = _runtime_contract()
    timm_sources = _timm_contract()
    current_device = _device_contract(device)[1]
    _, student = _resolve_asset(STUDENT_LOCK, student_weight)
    _, teacher = _resolve_asset(DINO_LOCK, dino_weight)
    current_checks = {
        **checks,
        "same_clean_git": payload["git"] == git,
        "same_source_hashes": payload["source_hashes"] == sources,
        "same_runtime": payload["runtime"] == runtime,
        "same_timm_sources": payload["timm_sources"] == timm_sources,
        "same_device": payload["device"] == current_device,
        "same_student_asset": _asset_identity(payload["weights"]["student"])
        == _asset_identity(student),
        "same_teacher_asset": _asset_identity(payload["weights"]["teacher"])
        == _asset_identity(teacher),
    }
    if not all(current_checks.values()):
        raise RuntimeError(f"Accepted B17 preflight no longer applies: {current_checks}")
    if path.read_bytes() != raw or path.with_name("preflight.sha256").read_bytes() != sidecar:
        raise RuntimeError("Accepted B17 preflight changed during validation")
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
                    "failure_artifact": getattr(error, "b17_failure_artifact", None),
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
                "passed": payload["passed"],
                "artifact": str(artifact),
                "sha256": _sha256(artifact),
                "formal_train_permission": payload["permissions"]["formal_train_permission"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

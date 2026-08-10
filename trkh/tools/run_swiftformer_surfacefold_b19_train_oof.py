from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# B19 deliberately reuses the immutable B17 SurfaceFold architecture; only the protocol gates change.
from trkh.models.swiftformer_surfacefold_b17 import (
    SURFACEFOLD_MEAN_CONTROL_B17_MODE,
    SURFACEFOLD_OFF_B17_MODE,
    SURFACEFOLD_SPATIAL_B17_MODE,
    SwiftFormerSurfaceFoldB17,
    dino_s3_relation_target_b17,
    surfacefold_s3_relation_loss_b17,
)

PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B19_BITEXACT_LR_SURFACEFOLD_XS_20260805"
PROTOCOL_SHA256 = "740551a99d0899af5ee37990ab8d71890cf9351118bcac57b19bcc34482e01fc"
EXPECTED_BRANCH = "research/pretrained-classf-b1"
SEED = 20260805
AUGMENTATION_SEED_BASE = 30260805
EXPECTED_DATA_YAML = Path("D:\\DataAI\\AIEx\\newdataset\\class_f\\data.yaml")
EXPECTED_DATA_SHA256 = "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8"
EXPECTED_TRAIN_ROWS = 8278
EXPECTED_CLASS_COUNTS = (1987, 497, 1326, 2080, 2388)
EXPECTED_CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
EXPECTED_TRAIN_CONTENT_SHA256 = "e9319e6fbddd03382050fadc42bcf156195c590926700b3a7287adc38d4373c8"
EXPECTED_ASSIGNMENT_CSV_SHA256 = "afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f"
EXPECTED_FOLD_VECTOR_SHA256 = "fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b"
EXPECTED_PATH_FOLD_SHA256 = "0f3856bf6020ba31d8564a48394e2a6fba7ad6f415e279e0063c5baadff20cb7"
EXPECTED_GROUP_VECTOR_SHA256 = "1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2"
STUDENT_TIMM_ID = "swiftformer_xs.dist_in1k"
STUDENT_BYTES = 13957760
STUDENT_SHA256 = "c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c"
DINO_TIMM_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_BYTES = 86362376
DINO_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DINO_PREFIX_TOKENS = 5
DINO_FEATURE_DIM = 384
FOLDS = 5
CLASSES = 5
CLASS1 = 1
RIVALS = (0, 2, 4)
ARMS = ("stock_relation", "surfacefold_mean_control", "surfacefold_spatial_candidate")
OOF_NAMES = (*ARMS, "candidate_delta_off")
ARM_MODES = {
    "stock_relation": SURFACEFOLD_OFF_B17_MODE,
    "surfacefold_mean_control": SURFACEFOLD_MEAN_CONTROL_B17_MODE,
    "surfacefold_spatial_candidate": SURFACEFOLD_SPATIAL_B17_MODE,
}
ARM_PARAMETERS = {
    "stock_relation": 3035570,
    "surfacefold_mean_control": 3061218,
    "surfacefold_spatial_candidate": 3061218,
}
ARM_TRAINABLE_PARAMETERS = {
    "stock_relation": 2813590,
    "surfacefold_mean_control": 2839238,
    "surfacefold_spatial_candidate": 2839238,
}
EPOCHS = 8
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
WORKERS = 0
BACKBONE_LR = 3e-05
TASK_LR = 0.0003
MIN_LR = 1e-06
EXPECTED_UPDATES_PER_EPOCH = (202, 206, 212, 207, 210)
WEIGHT_DECAY = 0.05
GRAD_CLIP = 0.7
RELATION_WEIGHT = 0.1
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260803
EXPECTED_BOOTSTRAP_DRAW_SHA256 = "d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469"
MAX_CUDA_ALLOCATED_BYTES = 7 * 1024**3
MAX_WALL_SECONDS = 12 * 60 * 60
MAX_RETAINED_BYTES = int(1.5 * 1024**3)
OUTPUT_PREFIX = "pretrained_surfacefold_xs_b19_train_oof_"
PREFLIGHT_OUTPUT_PREFIX = "preflight_b19_surfacefold_xs_"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METRIC_BARRIER_PHASE = "all_15_final_states_before_any_held_inference_or_metric"
ZERO_BARRIER_COUNTERS = {
    "held_image_open_count": 0,
    "held_forward_count": 0,
    "oof_buffer_count": 0,
    "logit_write_count": 0,
    "metric_call_count": 0,
}
_METRIC_BACKEND_LOADED = False
_VALID_BARRIER_TOKENS: set[str] = set()


class B19ContractError(RuntimeError):
    """Raised when the prospective B19 contract is no longer exact."""


@dataclass(frozen=True)
class CropParameters:
    top: int
    left: int
    height: int
    width: int
    horizontal_flip: bool


@dataclass(frozen=True)
class FinalStateRecord:
    state_id: str
    fold: int
    arm: str
    path: str
    sha256: str
    bytes: int
    parameter_count: int
    trainable_parameter_count: int
    mode: str
    metadata_sha256: str
    exposure_sha256: str
    loss_curve_sha256: str
    frozen_projection_sha256: str
    folded_max_abs_error: float


@dataclass(frozen=True)
class MetricBarrierToken:
    path: str
    sha256: str
    nonce: str


@dataclass(frozen=True)
class _MetricBarrierSealProof:
    path: str
    sha256: str
    nonce: str


_PENDING_BARRIER_SEALS: dict[str, _MetricBarrierSealProof] = {}
_ACTIVE_BARRIER_SEALS: dict[str, _MetricBarrierSealProof] = {}


def _barrier_seal_key(path: Path | str, sha256: str) -> str:
    return f"{Path(path).resolve()}:{sha256}"


def _metric_token_key(sha256: str, nonce: str) -> str:
    return f"{sha256}:{nonce}"


def _revoke_metric_token(key: str) -> None:
    _VALID_BARRIER_TOKENS.discard(key)
    _ACTIVE_BARRIER_SEALS.pop(key, None)


@dataclass
class TeacherCallLedger:
    calls: int = 0
    rows: int = 0


@dataclass
class ArmRngState:
    cpu: Tensor
    cuda: Tensor


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: object, length: int) -> bool:
    return bool(
        isinstance(value, str) and len(value) == length and all((char in "0123456789abcdef" for char in value))
    )


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and (not isinstance(value, bool))


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(float(value))


def _float64_bits(value: float) -> bytes:
    return struct.pack("!d", float(value))


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _atomic_bytes(path: Path, payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite B19 artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        digest = sha256_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise B19ContractError(f"Atomic B19 artifact hash drifted: {path}")
    return digest


def atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    return _atomic_bytes(path, _json_bytes(payload))


def atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite B19 artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        digest = sha256_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise B19ContractError(f"Atomic B19 NPZ hash drifted: {path}")
    return digest


def _metadata_for_state(fold: int, arm: str) -> dict[str, str]:
    return {
        "schema_version": "1",
        "protocol_id": PROTOCOL_ID,
        "research_track": "pretrained",
        "phase": "TRAIN_only_final_state",
        "state_id": f"fold_{fold}:{arm}",
        "fold": str(fold),
        "arm": arm,
        "mode": ARM_MODES[arm],
        "validation_used": "false",
        "test_used": "false",
        "teacher_included": "false",
    }


def _metadata_sha256(metadata: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_safetensors(path: Path, state: Mapping[str, Tensor], metadata: Mapping[str, str]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite B19 state: {path}")
    tensors = {str(name): tensor.detach().cpu().contiguous() for (name, tensor) in state.items()}
    if not tensors or any(
        (not bool(torch.isfinite(value).all()) for value in tensors.values() if value.is_floating_point())
    ):
        raise B19ContractError("B19 state is empty or contains non-finite tensors")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, str(temporary), metadata=dict(metadata))
        digest = sha256_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise B19ContractError(f"Atomic B19 state hash drifted: {path}")
    return digest


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _git_contract() -> dict[str, object]:
    root = _repository_root()

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    return {
        "head": head,
        "branch": branch,
        "status": status,
        "clean": status == "",
        "head_is_commit": _hex(head, 40),
    }


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = {
        "protocol": root / "docs" / "TRKH_PRETRAINED_CLASSF_B19_BITEXACT_LR_PROTOCOL_20260805.md",
        "model": root / "trkh" / "models" / "swiftformer_surfacefold_b17.py",
        "model_test": root / "tests" / "test_swiftformer_surfacefold_b17.py",
        "deployment": root / "trkh" / "inference" / "surfacefold_deployment.py",
        "deployment_test": root / "tests" / "test_surfacefold_deployment.py",
        "preflight_runner": root / "trkh" / "tools" / "audit_swiftformer_surfacefold_b19_preflight.py",
        "preflight_test": root / "tests" / "test_audit_swiftformer_surfacefold_b19_preflight.py",
        "formal_runner": Path(__file__).resolve(),
        "formal_runner_test": root / "tests" / "test_run_swiftformer_surfacefold_b19_train_oof.py",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing B19 source: {name}={path}")
        relative = path.relative_to(root)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise B19ContractError(f"B19 source is not tracked: {relative}")
    hashes = {name: sha256_file(path) for (name, path) in paths.items()}
    if hashes["protocol"] != PROTOCOL_SHA256:
        raise B19ContractError("B19 protocol hash changed")
    return hashes


def _canonical_preflight_artifact(artifact: object, expected_sha256: object) -> tuple[Path, str]:
    if not isinstance(artifact, (str, Path)) or not _hex(str(expected_sha256), 64):
        raise B19ContractError("Accepted B19 preflight path/hash schema is invalid")
    raw_path = Path(artifact)
    if not raw_path.is_absolute():
        raise B19ContractError("Accepted B19 preflight path must be absolute")
    resolved = raw_path.resolve(strict=True)
    runs = (_repository_root() / "runs").resolve()
    if (
        resolved.name != "preflight.json"
        or resolved.parent.parent != runs
        or (not resolved.parent.name.startswith(PREFLIGHT_OUTPUT_PREFIX))
        or (resolved != raw_path)
    ):
        raise B19ContractError("Accepted B19 preflight is outside its canonical runs path")
    sidecar = resolved.with_name("preflight.sha256")
    if set((child.name for child in resolved.parent.iterdir())) != {
        "preflight.json",
        "preflight.sha256",
    }:
        raise B19ContractError("Accepted B19 preflight publication schema changed")
    digest = sha256_file(resolved)
    if digest != expected_sha256 or sidecar.read_bytes() != f"{digest}\n".encode("ascii"):
        raise B19ContractError("Accepted B19 preflight hash/sidecar mismatch")
    return (resolved, digest)


def _validate_preflight_payload(
    payload: Mapping[str, object], *, git: Mapping[str, object], sources: Mapping[str, str]
) -> None:
    expected_top = {
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
    permissions = payload.get("permissions")
    expected_permissions = {
        "dataset_yaml_read": False,
        "dataset_path_enumerated": False,
        "train_image_read": False,
        "real_label_read": False,
        "prior_prediction_read": False,
        "validation_constructed": False,
        "test_constructed": False,
        "network_used": False,
        "formal_train_permission": True,
    }
    authorization = payload.get("authorization")
    expected_authorization = {
        "scope": "one_train_only_five_fold_causal_screen",
        "deployment_status": "DEVICE_PENDING",
        "validation_permission": False,
        "test_permission": False,
    }
    checks = payload.get("checks")
    expected_checks = {
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
    }
    expected_end = {
        "git": payload.get("git"),
        "source_hashes": payload.get("source_hashes"),
        "runtime": payload.get("runtime"),
        "timm_sources": payload.get("timm_sources"),
        "weights": payload.get("weights"),
    }
    if (
        set(payload) != expected_top
        or payload.get("schema_version") != 1
        or payload.get("protocol_id") != PROTOCOL_ID
        or (payload.get("protocol_sha256") != PROTOCOL_SHA256)
        or (payload.get("mode") != "offline_label_free_stock_first_preflight")
        or not _finite(payload.get("created_at_unix"))
        or (payload.get("passed") is not True)
        or (payload.get("git") != dict(git))
        or (payload.get("source_hashes") != dict(sources))
        or (permissions != expected_permissions)
        or (authorization != expected_authorization)
        or (not isinstance(checks, Mapping))
        or set(checks) != expected_checks
        or any((value is not True for value in checks.values()))
        or payload.get("isolation") != {"dataset_attempts": [], "network_attempts": [], "process_attempts": []}
        or payload.get("end_rehash") != expected_end
    ):
        raise B19ContractError("Accepted B19 preflight identity/schema/authorization changed")


def validate_accepted_preflight(
    artifact: Path,
    expected_sha256: str,
    student_weight: Path,
    dino_weight: Path,
    *,
    git: Mapping[str, object],
    sources: Mapping[str, str],
) -> dict[str, object]:
    from trkh.tools.audit_swiftformer_surfacefold_b19_preflight import (
        validate_accepted_preflight as validate_exact,
    )

    accepted = validate_exact(
        artifact,
        expected_sha256,
        student_weight=student_weight,
        dino_weight=dino_weight,
        device="cuda",
    )
    (resolved, digest) = _canonical_preflight_artifact(accepted.get("artifact"), accepted.get("sha256"))
    payload = accepted.get("payload")
    if not isinstance(payload, Mapping):
        raise B19ContractError("Accepted B19 preflight validator returned no payload")
    raw = resolved.read_bytes()
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, Mapping) or raw != _json_bytes(decoded) or dict(decoded) != dict(payload):
        raise B19ContractError("Accepted B19 preflight payload is not canonical/bound")
    _validate_preflight_payload(payload, git=git, sources=sources)
    return {"artifact": str(resolved), "sha256": digest, "payload": dict(payload)}


def _lexical_absolute(path: Path) -> str:
    """Bind a CLI path without opening, resolving, or enumerating its target."""
    return os.path.abspath(os.fspath(Path(path).expanduser()))


def _claim_payload(
    *,
    accepted_preflight: Mapping[str, object],
    git: Mapping[str, object],
    sources: Mapping[str, str],
    output: Path,
    args: argparse.Namespace,
    created_at_unix: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "authorization_scope": "one_train_only_five_fold_causal_screen",
        "consumption_policy": "exclusive_once_no_automatic_retry",
        "created_at_unix": float(created_at_unix),
        "preflight": {
            "artifact": accepted_preflight["artifact"],
            "sha256": accepted_preflight["sha256"],
        },
        "git_head": git["head"],
        "source_hashes": dict(sources),
        "output_dir": str(Path(output).resolve()),
        "arguments": {
            "data": _lexical_absolute(args.data),
            "assignment_csv": _lexical_absolute(args.assignment_csv),
            "student_weight": _lexical_absolute(args.student_weight),
            "dino_weight": _lexical_absolute(args.dino_weight),
        },
        "train": True,
        "validation": False,
        "test": False,
        "deployment_status": "DEVICE_PENDING",
    }


def claim_train_authorization_once(
    *,
    accepted_preflight: Mapping[str, object],
    git: Mapping[str, object],
    sources: Mapping[str, str],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Consume the preflight's one-screen authorization before any data read."""
    preflight_sha = accepted_preflight.get("sha256")
    if not _hex(preflight_sha, 64):
        raise B19ContractError("B19 authorization claim lacks a preflight SHA-256")
    claims_root = (_repository_root() / "runs" / "b19_train_authorization_claims").resolve()
    claims_root.mkdir(parents=True, exist_ok=True)
    path = claims_root / f"{preflight_sha}.json"
    payload = _claim_payload(
        accepted_preflight=accepted_preflight,
        git=git,
        sources=sources,
        output=output,
        args=args,
        created_at_unix=time.time(),
    )
    encoded = _json_bytes(payload)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise B19ContractError(f"B19 preflight authorization was already consumed: {path}") from exc
    digest = sha256_file(path)
    if path.read_bytes() != encoded:
        raise B19ContractError("B19 authorization claim changed during exclusive creation")
    return {"path": str(path), "sha256": digest, "payload": payload}


def validate_train_authorization_claim(
    claim: Mapping[str, object],
    *,
    accepted_preflight: Mapping[str, object],
    git: Mapping[str, object],
    sources: Mapping[str, str],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    if set(claim) != {"path", "sha256", "payload"} or not _hex(claim.get("sha256"), 64):
        raise B19ContractError("B19 authorization claim binding schema changed")
    preflight_sha = accepted_preflight.get("sha256")
    path = Path(str(claim["path"])).resolve(strict=True)
    expected = (_repository_root() / "runs" / "b19_train_authorization_claims" / f"{preflight_sha}.json").resolve()
    raw = path.read_bytes()
    decoded = json.loads(raw.decode("utf-8"))
    if (
        path != expected
        or claim["path"] != str(path)
        or sha256_file(path) != claim["sha256"]
        or not isinstance(decoded, Mapping)
        or raw != _json_bytes(decoded)
        or dict(decoded) != claim["payload"]
        or not _finite(decoded.get("created_at_unix"))
    ):
        raise B19ContractError("B19 authorization claim path/hash/payload changed")
    expected_payload = _claim_payload(
        accepted_preflight=accepted_preflight,
        git=git,
        sources=sources,
        output=output,
        args=args,
        created_at_unix=float(decoded["created_at_unix"]),
    )
    if dict(decoded) != expected_payload:
        raise B19ContractError("B19 authorization claim bindings changed")
    return {"path": str(path), "sha256": claim["sha256"], "payload": dict(decoded)}


def compute_assignment_hashes(relative_paths: Sequence[str], folds: np.ndarray, groups: np.ndarray) -> dict[str, str]:
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if len(relative_paths) != folds.size or folds.shape != groups.shape:
        raise ValueError("B19 assignment arrays are not aligned")
    path_fold = (
        "\n".join((f"{relative_paths[index]}\t{int(folds[index])}" for index in range(len(relative_paths)))) + "\n"
    )
    return {
        "fold_vector_sha256": hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest(),
        "path_fold_sha256": hashlib.sha256(path_fold.encode("utf-8")).hexdigest(),
        "group_vector_sha256": hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest(),
    }


def validate_assignment_arrays(
    relative_paths: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    *,
    enforce_locked: bool = True,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    if (
        not relative_paths
        or labels.shape != groups.shape
        or labels.shape != folds.shape
        or (labels.size != len(relative_paths))
    ):
        raise ValueError("B19 assignment is empty or misaligned")
    if bool((labels < 0).any()) or bool((labels >= CLASSES).any()) or bool((groups < 0).any()):
        raise ValueError("B19 assignment label/group domain is invalid")
    if set(np.unique(folds).tolist()) != set(range(FOLDS)):
        raise ValueError("B19 assignment must contain exactly folds 0..4")
    fold_rows: list[dict[str, object]] = []
    for fold in range(FOLDS):
        held = folds == fold
        overlap = set(groups[held].tolist()) & set(groups[~held].tolist())
        counts = np.bincount(labels[held], minlength=CLASSES).astype(int)
        if overlap or bool((counts <= 0).any()):
            raise ValueError(f"B19 fold {fold} is not component-disjoint/class-complete")
        fold_rows.append(
            {
                "fold": fold,
                "fit_samples": int((~held).sum()),
                "held_samples": int(held.sum()),
                "held_class_counts": counts.tolist(),
                "group_overlap": 0,
            }
        )
    hashes = compute_assignment_hashes(relative_paths, folds, groups)
    if enforce_locked:
        expected = {
            "fold_vector_sha256": EXPECTED_FOLD_VECTOR_SHA256,
            "path_fold_sha256": EXPECTED_PATH_FOLD_SHA256,
            "group_vector_sha256": EXPECTED_GROUP_VECTOR_SHA256,
        }
        if (
            labels.size != EXPECTED_TRAIN_ROWS
            or tuple(np.bincount(labels, minlength=CLASSES).tolist()) != EXPECTED_CLASS_COUNTS
            or hashes != expected
        ):
            raise B19ContractError("Locked B19 assignment counts/vector hashes changed")
    return {"hashes": hashes, "fold_rows": fold_rows}


def read_locked_assignment_csv(path: Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if sha256_file(resolved) != EXPECTED_ASSIGNMENT_CSV_SHA256:
        raise B19ContractError("B19 assignment CSV hash changed")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != ["relative_path", "label", "union_group", "fold"]:
            raise B19ContractError("B19 assignment CSV columns changed")
        rows = list(reader)
    relative_paths = [str(row["relative_path"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([int(row["union_group"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    return {
        "csv": str(resolved),
        "csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
        "relative_paths": relative_paths,
        "labels": labels,
        "groups": groups,
        "folds": folds,
        **validate_assignment_arrays(relative_paths, labels, groups, folds),
    }


def _read_locked_data_root(data_yaml: Path) -> Path:
    resolved = Path(data_yaml).expanduser().resolve(strict=True)
    if resolved != EXPECTED_DATA_YAML.resolve() or sha256_file(resolved) != EXPECTED_DATA_SHA256:
        raise B19ContractError("Canonical class_f YAML path/hash changed")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    data_format = str(raw.get("format", raw.get("data_format", ""))).strip().casefold()
    names = raw.get("names")
    if isinstance(names, Mapping):
        class_names = tuple((str(names[key]) for key in sorted(names, key=lambda value: int(value))))
    elif isinstance(names, Sequence) and (not isinstance(names, (str, bytes))):
        class_names = tuple((str(value) for value in names))
    else:
        class_names = ()
    configured = Path(str(raw.get("path", ".")))
    root = configured if configured.is_absolute() else resolved.parent / configured
    root = root.expanduser().resolve()
    train_value = Path(str(raw.get("train", "train")))
    train_root = train_value if train_value.is_absolute() else root / train_value
    if (
        data_format != "classification_folder"
        or class_names != EXPECTED_CLASS_NAMES
        or train_root.expanduser().resolve() != (root / "train").resolve()
    ):
        raise B19ContractError("Canonical B19 TRAIN data declaration changed")
    return root


def train_content_contract(relative_paths: Sequence[str], absolute_paths: Sequence[Path]) -> dict[str, object]:
    if not relative_paths or len(relative_paths) != len(absolute_paths):
        raise ValueError("B19 TRAIN content paths are empty or misaligned")
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, absolute in zip(relative_paths, absolute_paths):
        path = Path(absolute).resolve(strict=True)
        before = path.stat()
        file_sha = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise B19ContractError(f"TRAIN image changed while hashing: {path}")
        encoded = str(relative).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(before.st_size).to_bytes(8, "little"))
        digest.update(bytes.fromhex(file_sha))
        total_bytes += int(before.st_size)
    payload = {
        "algorithm": "ordered_path_length_file_sha256_v1",
        "files": len(relative_paths),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "train_only": True,
    }
    if len(relative_paths) == EXPECTED_TRAIN_ROWS and payload["sha256"] != EXPECTED_TRAIN_CONTENT_SHA256:
        raise B19ContractError("Canonical B19 TRAIN content hash changed")
    return payload


def sample_random_resized_crop(height: int, width: int, generator: torch.Generator) -> CropParameters:
    (height, width) = (int(height), int(width))
    if height <= 0 or width <= 0:
        raise ValueError("Image geometry must be positive")
    area = float(height * width)
    log_ratio = torch.log(torch.tensor((0.9, 1.1), dtype=torch.float64))
    for _ in range(10):
        target = area * float(torch.empty((), dtype=torch.float64).uniform_(0.8, 1.0, generator=generator))
        aspect = float(
            torch.exp(
                torch.empty((), dtype=torch.float64).uniform_(
                    float(log_ratio[0]), float(log_ratio[1]), generator=generator
                )
            )
        )
        crop_width = int(round(math.sqrt(target * aspect)))
        crop_height = int(round(math.sqrt(target / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
            left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
            flip = bool(float(torch.rand((), generator=generator)) < 0.5)
            return CropParameters(top, left, crop_height, crop_width, flip)
    input_ratio = width / height
    if input_ratio < 0.9:
        (crop_width, crop_height) = (width, int(round(width / 0.9)))
    elif input_ratio > 1.1:
        (crop_height, crop_width) = (height, int(round(height * 1.1)))
    else:
        (crop_height, crop_width) = (height, width)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    flip = bool(float(torch.rand((), generator=generator)) < 0.5)
    return CropParameters(top, left, crop_height, crop_width, flip)


def _normalize_image(image: Image.Image, size: int) -> Tensor:
    resized = TF.resize(image, [int(size), int(size)], interpolation=InterpolationMode.BICUBIC, antialias=True)
    tensor = TF.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def build_shared_train_views(image: Image.Image, parameters: CropParameters) -> tuple[Tensor, Tensor]:
    crop = TF.crop(image.convert("RGB"), parameters.top, parameters.left, parameters.height, parameters.width)
    if parameters.horizontal_flip:
        crop = TF.hflip(crop)
    return (_normalize_image(crop, 224), _normalize_image(crop, 256))


def build_shared_batch_views(
    images: Sequence[Image.Image], generator: torch.Generator
) -> tuple[Tensor, Tensor, list[CropParameters]]:
    (student, teacher, parameters) = ([], [], [])
    for image in images:
        crop = sample_random_resized_crop(image.height, image.width, generator)
        (student_view, teacher_view) = build_shared_train_views(image, crop)
        student.append(student_view)
        teacher.append(teacher_view)
        parameters.append(crop)
    return (torch.stack(student), torch.stack(teacher), parameters)


def build_eval_view(image: Image.Image) -> Tensor:
    resized = TF.resize(image.convert("RGB"), 235, interpolation=InterpolationMode.BICUBIC, antialias=True)
    cropped = TF.center_crop(resized, [224, 224])
    tensor = TF.pil_to_tensor(cropped).to(dtype=torch.float32).div_(255.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def load_raw_batch(
    paths: Sequence[Path], labels: np.ndarray, indices: Sequence[int]
) -> tuple[list[Image.Image], Tensor, Tensor]:
    images: list[Image.Image] = []
    observed_labels: list[int] = []
    observed_indices: list[int] = []
    for raw_index in indices:
        index = int(raw_index)
        with Image.open(paths[index]) as handle:
            images.append(ImageOps.exif_transpose(handle).convert("RGB").copy())
        observed_labels.append(int(labels[index]))
        observed_indices.append(index)
    return (
        images,
        torch.tensor(observed_labels, dtype=torch.long),
        torch.tensor(observed_indices, dtype=torch.long),
    )


def teacher_relation_target(teacher: nn.Module, images: Tensor, ledger: TeacherCallLedger) -> Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
        raise B19ContractError("DINO input must be [B,3,256,256]")
    with torch.inference_mode(), torch.autocast(device_type=images.device.type, enabled=False):
        tokens = teacher.forward_features(images.float())
    ledger.calls += 1
    ledger.rows += int(images.size(0))
    if not isinstance(tokens, Tensor) or tuple(tokens.shape[1:]) != (261, DINO_FEATURE_DIM):
        raise B19ContractError(f"DINO token geometry changed: {getattr(tokens, 'shape', None)}")
    patch_map = (
        tokens[:, DINO_PREFIX_TOKENS:].reshape(images.size(0), 16, 16, DINO_FEATURE_DIM).permute(0, 3, 1, 2).float()
    )
    target = dino_s3_relation_target_b17(patch_map)
    if tuple(target.shape) != (images.size(0), 84) or target.requires_grad:
        raise B19ContractError("DINO B19 relation target geometry/gradient changed")
    return target


def exact_b19_objective(
    logits: Tensor,
    student_s3: Tensor,
    labels: Tensor,
    class_weights: Tensor,
    relation_target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the one locked CE + raw-DINO relation objective used by every arm."""
    relation = surfacefold_s3_relation_loss_b17(student_s3, relation_target)
    ce = F.cross_entropy(logits.float(), labels, weight=class_weights, reduction="mean")
    total = ce + RELATION_WEIGHT * relation
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("B19 exact CE+relation objective is non-finite")
    return total, ce, relation


def _rng_snapshot(device: torch.device) -> ArmRngState:
    if device.type != "cuda":
        raise B19ContractError("B19 arm RNG streams require CUDA")
    return ArmRngState(torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())


def _rng_restore(device: torch.device, state: ArmRngState) -> None:
    torch.random.set_rng_state(state.cpu)
    torch.cuda.set_rng_state(state.cuda, device)


def make_arm_rng_states(device: torch.device, seed: int) -> dict[str, ArmRngState]:
    caller = _rng_snapshot(device)
    result: dict[str, ArmRngState] = {}
    try:
        for arm in ARMS:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            state = _rng_snapshot(device)
            result[arm] = ArmRngState(state.cpu.clone(), state.cuda.clone())
    finally:
        _rng_restore(device, caller)
    if len({id(state.cpu) for state in result.values()}) != len(ARMS):
        raise B19ContractError("B19 arms do not own disjoint RNG state tensors")
    return result


def run_with_arm_rng(
    arm: str, states: dict[str, ArmRngState], device: torch.device, function: Callable[[], Any]
) -> Any:
    if arm not in ARMS or set(states) != set(ARMS):
        raise B19ContractError("B19 arm RNG state mapping changed")
    caller = _rng_snapshot(device)
    try:
        _rng_restore(device, states[arm])
        result = function()
        captured = _rng_snapshot(device)
        states[arm] = ArmRngState(captured.cpu.clone(), captured.cuda.clone())
        return result
    finally:
        _rng_restore(device, caller)


def _preserve_rng(function: Callable[[], Any]) -> Any:
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    try:
        return function()
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def reset_locked_five_class_heads(model: nn.Module, seed: int = SEED) -> dict[str, object]:
    old_heads = (getattr(model, "head", None), getattr(model, "head_dist", None))
    if not all(isinstance(head, nn.Linear) and head.in_features == 220 for head in old_heads):
        raise B19ContractError("B19 head reset requires two pretrained 220-channel linear heads")
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        device, dtype = old_heads[0].weight.device, old_heads[0].weight.dtype
        for name in ("head", "head_dist"):
            layer = nn.Linear(220, CLASSES, device=device, dtype=dtype)
            timm.layers.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
            setattr(model, name, layer)
    after_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    if not torch.equal(before_cpu, torch.random.get_rng_state()) or (
        before_cuda is not None
        and (
            after_cuda is None
            or len(before_cuda) != len(after_cuda)
            or any((not torch.equal(left, right) for (left, right) in zip(before_cuda, after_cuda)))
        )
    ):
        raise B19ContractError("B19 five-class head reset consumed caller RNG")
    model.num_classes = CLASSES
    model.distilled_training = False
    return {
        "seed": int(seed),
        "order": ["head", "head_dist"],
        "initializer": "timm.trunc_normal_std_0.02_bias_zero",
    }


def _formal_initialized_cuda_head_reset_probe() -> dict[str, object]:
    """Repeat the B19 initialized-CUDA RNG proof before any TRAIN data access."""
    if not torch.cuda.is_available():
        raise B19ContractError("Formal B19 head-reset RNG proof requires CUDA")
    torch.cuda.init()
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise B19ContractError("Formal B19 head-reset RNG proof found no CUDA device")
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]

    def build_probe() -> nn.Module:
        probe = nn.Module()
        probe.head = nn.Linear(220, 1000, device=torch.device("cuda:0"))
        probe.head_dist = nn.Linear(220, 1000, device=torch.device("cuda:0"))
        return probe

    probe = _preserve_rng(build_probe)
    reset = reset_locked_five_class_heads(probe)
    after_cpu = torch.random.get_rng_state()
    after_cuda = torch.cuda.get_rng_state_all()
    cpu_equal = torch.equal(before_cpu, after_cpu)
    cuda_equal = len(before_cuda) == len(after_cuda) and all(
        torch.equal(left, right) for left, right in zip(before_cuda, after_cuda)
    )
    if not cpu_equal or not cuda_equal:
        raise B19ContractError("Formal B19 initialized-CUDA head reset consumed caller RNG")
    return {
        "cuda_initialized": torch.cuda.is_initialized(),
        "device_count": device_count,
        "cpu_rng_equal": cpu_equal,
        "all_cuda_rng_equal": cuda_equal,
        "reset": reset,
    }


def _tensor_sha256(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _projection_hash(model: SwiftFormerSurfaceFoldB17) -> str:
    projection = model.backbone.stages[3].downsample.proj
    if projection.bias is None:
        raise B19ContractError("B19 frozen projection unexpectedly lacks bias")
    return _tensor_sha256(projection.weight, projection.bias)


def _assert_disjoint_model_storage(models: Mapping[str, nn.Module]) -> None:
    pointers: dict[int, str] = {}
    for arm, model in models.items():
        for name, tensor in model.state_dict().items():
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in pointers:
                raise B19ContractError(f"B19 arm storage aliases: {pointers[pointer]} and {arm}:{name}")
            pointers[pointer] = f"{arm}:{name}"


def build_b19_arm_models(base_five_class: nn.Module) -> dict[str, SwiftFormerSurfaceFoldB17]:

    def build() -> dict[str, SwiftFormerSurfaceFoldB17]:
        return {arm: SwiftFormerSurfaceFoldB17(copy.deepcopy(base_five_class), ARM_MODES[arm]) for arm in ARMS}

    arms = _preserve_rng(build)
    counts = {arm: sum((parameter.numel() for parameter in model.parameters())) for (arm, model) in arms.items()}
    trainable = {
        arm: sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        for (arm, model) in arms.items()
    }
    if counts != ARM_PARAMETERS or trainable != ARM_TRAINABLE_PARAMETERS:
        raise B19ContractError(f"B19 arm parameter contracts changed: total={counts}, trainable={trainable}")
    control = arms["surfacefold_mean_control"]
    candidate = arms["surfacefold_spatial_candidate"]
    if not (torch.equal(control.factor_p, candidate.factor_p) and torch.equal(control.factor_d, candidate.factor_d)):
        raise B19ContractError("B19 candidate/control factor initial states differ")
    reference_backbone = arms[ARMS[0]].backbone.state_dict()
    for arm in ARMS[1:]:
        observed = arms[arm].backbone.state_dict()
        if observed.keys() != reference_backbone.keys() or any(
            (not torch.equal(reference_backbone[key], observed[key]) for key in reference_backbone)
        ):
            raise B19ContractError("B19 arms do not share an identical base state")
    if len({_projection_hash(model) for model in arms.values()}) != 1:
        raise B19ContractError("B19 arms do not share the immutable W0/b0 state")
    _assert_disjoint_model_storage(arms)
    for arm, model in arms.items():
        projection = model.backbone.stages[3].downsample.proj
        if projection.weight.requires_grad or projection.bias.requires_grad:
            raise B19ContractError(f"B19 {arm} did not freeze W0/b0")
        for module in model.modules():
            if isinstance(module, nn.Dropout) and float(module.p) != 0.0:
                raise B19ContractError(f"B19 {arm} contains active dropout")
            if float(getattr(module, "drop_prob", 0.0) or 0.0) != 0.0:
                raise B19ContractError(f"B19 {arm} contains stochastic depth")
    return arms


def _validate_asset(path: Path, expected_bytes: int, expected_sha: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.stat().st_size != expected_bytes or sha256_file(resolved) != expected_sha:
        raise B19ContractError(f"Locked pretrained asset changed: {resolved}")
    return resolved


def _load_templates(student_weight: Path, dino_weight: Path) -> tuple[nn.Module, nn.Module]:
    student_path = _validate_asset(student_weight, STUDENT_BYTES, STUDENT_SHA256)
    dino_path = _validate_asset(dino_weight, DINO_BYTES, DINO_SHA256)

    def construct() -> tuple[nn.Module, nn.Module]:
        student = timm.create_model(
            STUDENT_TIMM_ID, pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.0
        )
        student.load_state_dict(load_file(str(student_path), device="cpu"), strict=True)
        reset_locked_five_class_heads(student, SEED)
        teacher = timm.create_model(DINO_TIMM_ID, pretrained=False, num_classes=0, img_size=256)
        teacher_state = load_file(str(dino_path), device="cpu")
        forbidden = [
            key
            for key in teacher_state
            if any((term in key.casefold() for term in ("class_f", "b19", "classifier_5")))
        ]
        if forbidden:
            raise B19ContractError(f"Forbidden class-fitted teacher state: {forbidden}")
        teacher.load_state_dict(teacher_state, strict=True)
        return (student, teacher)

    (student, teacher) = _preserve_rng(construct)
    teacher.eval().requires_grad_(False)
    if sum((parameter.numel() for parameter in student.parameters())) != 3035570:
        raise B19ContractError("Five-class SwiftFormer parameter count changed")
    if int(getattr(teacher, "num_prefix_tokens", -1)) != 5 or tuple(teacher.patch_embed.grid_size) != (16, 16):
        raise B19ContractError("Raw DINO architecture geometry changed")
    return (student, teacher)


def _is_task_parameter(name: str) -> bool:
    return name in {"factor_p", "factor_d"} or name.startswith(("backbone.head.", "backbone.head_dist."))


def _normalization_parameter_names(model: nn.Module) -> set[str]:
    normalization_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, normalization_types):
            for parameter_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return names


def build_discriminative_adamw_groups(model: nn.Module) -> list[dict[str, object]]:
    norm_names = _normalization_parameter_names(model)
    buckets: dict[tuple[str, bool], list[tuple[str, nn.Parameter]]] = {
        (role, decay): [] for role in ("backbone", "task") for decay in (True, False)
    }
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise B19ContractError(f"Duplicate B19 trainable parameter: {name}")
        seen.add(id(parameter))
        role = "task" if _is_task_parameter(name) else "backbone"
        no_decay = name.endswith(".bias") or name in norm_names or "layer_scale" in name.casefold()
        buckets[role, not no_decay].append((name, parameter))
    if seen != expected:
        raise B19ContractError("B19 AdamW groups do not exactly cover trainables")
    result: list[dict[str, object]] = []
    for role, decay in (("backbone", True), ("backbone", False), ("task", True), ("task", False)):
        values = buckets[role, decay]
        if not values:
            continue
        names = [name for (name, _) in values]
        peak_lr = TASK_LR if role == "task" else BACKBONE_LR
        result.append(
            {
                "params": [parameter for (_, parameter) in values],
                "lr": peak_lr,
                "peak_lr": peak_lr,
                "weight_decay": WEIGHT_DECAY if decay else 0.0,
                "name": f"{role}_{('decay' if decay else 'no_decay')}",
                "parameter_names_sha256": hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest(),
                "parameter_count": sum((parameter.numel() for (_, parameter) in values)),
            }
        )
    return result


def audit_and_clip_gradients(model: nn.Module, max_norm: float = GRAD_CLIP) -> dict[str, object]:
    named = [(name, parameter) for (name, parameter) in model.named_parameters() if parameter.requires_grad]
    missing = [name for (name, parameter) in named if parameter.grad is None]
    if missing:
        raise B19ContractError(f"B19 trainables lack gradients: {missing}")
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for (_, parameter) in named], float(max_norm), error_if_nonfinite=True
        )
    except RuntimeError as exc:
        raise B19ContractError("B19 gradient tensor is non-finite") from exc
    if not bool(torch.isfinite(norm)):
        raise B19ContractError("B19 global gradient norm is non-finite")
    return {
        "parameter_tensors": len(named),
        "parameter_elements": sum((parameter.numel() for (_, parameter) in named)),
        "preclip_global_norm": float(norm),
    }


class LockedOptimizerStepper:
    def __init__(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> None:
        self.optimizer = optimizer
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_steps = EPOCHS * self.steps_per_epoch
        self.completed_steps = 0
        self.checkpoints: dict[str, dict[str, float]] = {}
        if self.steps_per_epoch < 2 or not optimizer.param_groups:
            raise ValueError("B19 scheduler needs at least two steps and parameter groups")
        names: set[str] = set()
        for index, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            peak = group.get("peak_lr")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or (not _finite(peak))
                or (float(peak) <= MIN_LR)
            ):
                raise ValueError("Invalid B19 optimizer group name/peak LR")
            names.add(name)
            group["name"] = name
            group["lr"] = 0.0

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _lr(self, peak: float, update: int) -> float:
        if update < 0 or update >= self.total_steps:
            raise B19ContractError("B19 LR requested outside locked horizon")
        if update == 0:
            return 0.0
        if update == self.steps_per_epoch - 1:
            return peak
        if update == self.total_steps - 1:
            return MIN_LR
        if update < self.steps_per_epoch:
            return float(peak * update / (self.steps_per_epoch - 1))
        progress = (update - self.steps_per_epoch + 1) / (self.total_steps - self.steps_per_epoch)
        return float(MIN_LR + (peak - MIN_LR) * (1.0 + math.cos(math.pi * progress)) / 2.0)

    def step(self) -> dict[str, float]:
        applied: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            lr = self._lr(float(group["peak_lr"]), self.completed_steps)
            group["lr"] = lr
            applied[str(group["name"])] = lr
        self.optimizer.step()
        self.completed_steps += 1
        if self.completed_steps == 1:
            self.checkpoints["first"] = dict(applied)
        if self.completed_steps == self.steps_per_epoch:
            self.checkpoints["warmup_last"] = dict(applied)
        if self.completed_steps == self.total_steps:
            self.checkpoints["final"] = dict(applied)
        return applied

    def evidence(self, *, require_complete: bool) -> dict[str, object]:
        if require_complete and (
            self.completed_steps != self.total_steps or set(self.checkpoints) != {"first", "warmup_last", "final"}
        ):
            raise B19ContractError("B19 LR horizon is incomplete")
        payload: dict[str, object] = {
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "first_applied_lrs": dict(self.checkpoints.get("first", {})),
            "warmup_last_applied_lrs": dict(self.checkpoints.get("warmup_last", {})),
            "final_applied_lrs": dict(self.checkpoints.get("final", {})),
        }
        if require_complete:
            peaks = {str(group["name"]): float(group["peak_lr"]) for group in self.optimizer.param_groups}
            first = payload["first_applied_lrs"]
            warmup_last = payload["warmup_last_applied_lrs"]
            final = payload["final_applied_lrs"]
            if (
                set(first) != set(peaks)
                or set(warmup_last) != set(peaks)
                or set(final) != set(peaks)
                or any((_float64_bits(first[name]) != _float64_bits(0.0) for name in peaks))
                or any((_float64_bits(warmup_last[name]) != _float64_bits(peaks[name]) for name in peaks))
                or any((_float64_bits(final[name]) != _float64_bits(MIN_LR) for name in peaks))
            ):
                raise B19ContractError("B19 applied LR checkpoints changed")
        return payload


def sqrt_inverse_class_weights(labels: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, Tensor]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64)[np.asarray(fit, dtype=bool)], minlength=CLASSES).astype(
        np.int64
    )
    if counts.shape != (CLASSES,) or bool((counts <= 0).any()):
        raise B19ContractError("Every B19 fit fold must contain all classes")
    weights = counts.astype(np.float64) ** (-0.5)
    weights /= weights.mean()
    return (counts, torch.tensor(weights, dtype=torch.float32))


def derive_locked_updates_per_epoch(folds: np.ndarray) -> tuple[int, ...]:
    raw = np.asarray(folds)
    if (
        raw.ndim != 1
        or raw.shape != (EXPECTED_TRAIN_ROWS,)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise B19ContractError(
            f"B19 fold vector must be a one-dimensional integer vector with exactly {EXPECTED_TRAIN_ROWS} rows"
        )
    canonical = raw.astype(np.int64, copy=False)
    if set(np.unique(canonical).tolist()) != set(range(FOLDS)):
        raise B19ContractError("B19 fold vector must contain exactly folds 0..4")
    derived = tuple(
        math.ceil(
            math.ceil(int(np.count_nonzero(canonical != fold)) / BATCH_SIZE)
            / ACCUMULATION_STEPS
        )
        for fold in range(FOLDS)
    )
    if derived != EXPECTED_UPDATES_PER_EPOCH:
        raise B19ContractError(f"B19 assignment-derived optimizer horizons changed: {derived}")
    return derived


def expected_state_ids() -> set[str]:
    return {f"fold_{fold}:{arm}" for fold in range(FOLDS) for arm in ARMS}


def _validate_preflight_binding(
    accepted: Mapping[str, object], *, git: Mapping[str, object], sources: Mapping[str, str]
) -> dict[str, object]:
    if set(accepted) != {"artifact", "sha256", "payload"}:
        raise B19ContractError("B19 accepted-preflight binding schema changed")
    (path, digest) = _canonical_preflight_artifact(accepted.get("artifact"), accepted.get("sha256"))
    raw = path.read_bytes()
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, Mapping) or raw != _json_bytes(decoded) or dict(decoded) != accepted.get("payload"):
        raise B19ContractError("B19 accepted-preflight artifact/payload changed")
    _validate_preflight_payload(decoded, git=git, sources=sources)
    return {"artifact": str(path), "sha256": digest, "payload": dict(decoded)}


def _validate_state_records(
    output: Path,
    records: Sequence[FinalStateRecord],
    strict_reload: Callable[[FinalStateRecord], None],
) -> list[dict[str, object]]:
    if (
        len(records) != FOLDS * len(ARMS)
        or any((not isinstance(record, FinalStateRecord) for record in records))
        or {record.state_id for record in records} != expected_state_ids()
    ):
        raise B19ContractError("B19 metric barrier requires exact 15 final states")
    states_root = (Path(output) / "states").resolve()
    result: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.state_id):
        expected_path = (states_root / f"fold_{record.fold}_{record.arm}.safetensors").resolve()
        path = Path(record.path).resolve(strict=True)
        metadata = _metadata_for_state(record.fold, record.arm)
        if (
            record.fold not in range(FOLDS)
            or record.arm not in ARMS
            or record.state_id != f"fold_{record.fold}:{record.arm}"
            or (path != expected_path)
            or (record.path != str(path))
            or (record.bytes != path.stat().st_size)
            or (record.sha256 != sha256_file(path))
            or (record.parameter_count != ARM_PARAMETERS[record.arm])
            or (record.trainable_parameter_count != ARM_TRAINABLE_PARAMETERS[record.arm])
            or (record.mode != ARM_MODES[record.arm])
            or (_read_safetensors_metadata(path) != metadata)
            or (record.metadata_sha256 != _metadata_sha256(metadata))
            or (
                not all(
                    (
                        _hex(value, 64)
                        for value in (
                            record.sha256,
                            record.metadata_sha256,
                            record.exposure_sha256,
                            record.loss_curve_sha256,
                            record.frozen_projection_sha256,
                        )
                    )
                )
            )
            or (not _finite(record.folded_max_abs_error))
            or (float(record.folded_max_abs_error) > 1e-06)
        ):
            raise B19ContractError(f"B19 final state contract changed: {record.state_id}")
        strict_reload(record)
        result.append(asdict(record))
    partials = [
        path
        for path in states_root.rglob("*")
        if path.is_file() and (path.suffix in {".tmp", ".partial"} or ".tmp" in path.name)
    ]
    if partials:
        raise B19ContractError(f"Partial B19 states remain: {partials}")
    return result


def seal_metric_barrier(
    output: Path,
    records: Sequence[FinalStateRecord],
    *,
    accepted_preflight: Mapping[str, object],
    authorization_claim: Mapping[str, object],
    args: argparse.Namespace,
    git: Mapping[str, object],
    sources: Mapping[str, str],
    counters: Mapping[str, int],
    strict_reload: Callable[[FinalStateRecord], None],
    cross_mode_rejection: Callable[[], None],
) -> MetricBarrierToken:
    if (
        dict(counters) != ZERO_BARRIER_COUNTERS
        or _METRIC_BACKEND_LOADED
        or set(counters) != set(ZERO_BARRIER_COUNTERS)
    ):
        raise B19ContractError("Held inference/logits/metrics occurred before B19 barrier")
    if _git_contract() != dict(git) or _source_hashes() != dict(sources):
        raise B19ContractError("B19 git/source bindings changed before metric barrier sealing")
    accepted = _validate_preflight_binding(accepted_preflight, git=git, sources=sources)
    claim = validate_train_authorization_claim(
        authorization_claim,
        accepted_preflight=accepted,
        git=git,
        sources=sources,
        output=output,
        args=args,
    )
    states = _validate_state_records(output, records, strict_reload)
    cross_mode_rejection()
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": METRIC_BARRIER_PHASE,
        "created_at_unix": time.time(),
        "accepted_preflight": {
            "artifact": accepted["artifact"],
            "sha256": accepted["sha256"],
            "formal_train_permission": True,
            "deployment_status": "DEVICE_PENDING",
        },
        "authorization_claim": {"path": claim["path"], "sha256": claim["sha256"]},
        "git_head": git["head"],
        "source_hashes": dict(sources),
        "expected_state_ids": sorted(expected_state_ids()),
        "states": states,
        "counters": dict(counters),
        "checks": {
            "exact_15_states": True,
            "hash_metadata_strict_reload": True,
            "cross_mode_rejected": True,
            "accepted_b19_preflight": True,
            "held_inference_absent": True,
            "metrics_absent": True,
        },
        "passed": True,
    }
    path = Path(output).resolve() / "metric_barrier.json"
    digest = atomic_json(path, payload)
    resolved = path.resolve(strict=True)
    proof = _MetricBarrierSealProof(path=str(resolved), sha256=digest, nonce=uuid.uuid4().hex)
    pending_key = _barrier_seal_key(resolved, digest)
    if pending_key in _PENDING_BARRIER_SEALS:
        raise B19ContractError("B19 metric barrier already has a pending private seal proof")
    _PENDING_BARRIER_SEALS[pending_key] = proof
    try:
        return validate_metric_barrier(resolved, digest)
    finally:
        _PENDING_BARRIER_SEALS.pop(pending_key, None)


def validate_metric_barrier(path: Path, expected_sha256: str) -> MetricBarrierToken:
    resolved = Path(path).resolve(strict=True)
    digest = sha256_file(resolved)
    pending_key = _barrier_seal_key(resolved, digest)
    proof = _PENDING_BARRIER_SEALS.pop(pending_key, None)
    if proof is None:
        raise B19ContractError("B19 metric capability requires a pending private seal proof")
    if proof.path != str(resolved) or proof.sha256 != digest or not proof.nonce:
        raise B19ContractError("B19 private metric-barrier seal proof changed")
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if (
        digest != expected_sha256
        or not _hex(digest, 64)
        or not isinstance(payload, Mapping)
        or raw != _json_bytes(payload)
    ):
        raise B19ContractError("B19 metric barrier identity/hash changed")
    _validate_barrier_payload(payload, resolved)
    token_key = _metric_token_key(digest, proof.nonce)
    if token_key in _VALID_BARRIER_TOKENS or token_key in _ACTIVE_BARRIER_SEALS:
        raise B19ContractError("B19 private metric-barrier seal proof was replayed")
    _VALID_BARRIER_TOKENS.add(token_key)
    _ACTIVE_BARRIER_SEALS[token_key] = proof
    return MetricBarrierToken(str(resolved), digest, proof.nonce)


def _validate_barrier_payload(payload: Mapping[str, object], resolved: Path) -> None:
    expected_top = {
        "schema_version",
        "protocol_id",
        "phase",
        "created_at_unix",
        "accepted_preflight",
        "authorization_claim",
        "git_head",
        "source_hashes",
        "expected_state_ids",
        "states",
        "counters",
        "checks",
        "passed",
    }
    expected_checks = {
        "exact_15_states": True,
        "hash_metadata_strict_reload": True,
        "cross_mode_rejected": True,
        "accepted_b19_preflight": True,
        "held_inference_absent": True,
        "metrics_absent": True,
    }
    if (
        set(payload) != expected_top
        or payload.get("schema_version") != 1
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("phase") != METRIC_BARRIER_PHASE
        or not _finite(payload.get("created_at_unix"))
        or payload.get("passed") is not True
        or payload.get("expected_state_ids") != sorted(expected_state_ids())
        or payload.get("counters") != ZERO_BARRIER_COUNTERS
        or payload.get("checks") != expected_checks
        or not _hex(payload.get("git_head"), 40)
    ):
        raise B19ContractError("B19 metric barrier schema/identity changed")
    source_hashes = payload.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any((not isinstance(key, str) or not _hex(value, 64) for key, value in source_hashes.items()))
    ):
        raise B19ContractError("B19 metric barrier source hashes changed")
    states = payload.get("states")
    if (
        not isinstance(states, list)
        or len(states) != 15
        or {row.get("state_id") for row in states if isinstance(row, Mapping)} != expected_state_ids()
    ):
        raise B19ContractError("B19 metric barrier state list changed")
    for row in states:
        if not isinstance(row, Mapping) or set(row) != set(FinalStateRecord.__dataclass_fields__):
            raise B19ContractError("B19 metric barrier state schema changed")
        fold, arm = row.get("fold"), row.get("arm")
        if (
            not _strict_int(fold)
            or fold not in range(FOLDS)
            or arm not in ARMS
            or row.get("state_id") != f"fold_{fold}:{arm}"
            or row.get("parameter_count") != ARM_PARAMETERS[arm]
            or row.get("trainable_parameter_count") != ARM_TRAINABLE_PARAMETERS[arm]
            or row.get("mode") != ARM_MODES[arm]
            or not _finite(row.get("folded_max_abs_error"))
            or float(row["folded_max_abs_error"]) > 1e-06
        ):
            raise B19ContractError("B19 metric barrier state identity changed")
        state_path = Path(str(row["path"])).resolve(strict=True)
        expected = (resolved.parent / "states" / f"fold_{fold}_{arm}.safetensors").resolve()
        exposure = resolved.parent / "folds" / f"fold_{fold}" / "exposure.npz"
        losses = resolved.parent / "folds" / f"fold_{fold}" / "loss_curves.npz"
        metadata = _metadata_for_state(fold, arm)
        if (
            state_path != expected
            or row["path"] != str(state_path)
            or not _strict_int(row.get("bytes"))
            or state_path.stat().st_size != row["bytes"]
            or sha256_file(state_path) != row.get("sha256")
            or _read_safetensors_metadata(state_path) != metadata
            or row.get("metadata_sha256") != _metadata_sha256(metadata)
            or not exposure.is_file()
            or sha256_file(exposure) != row.get("exposure_sha256")
            or not losses.is_file()
            or sha256_file(losses) != row.get("loss_curve_sha256")
            or not _hex(row.get("frozen_projection_sha256"), 64)
        ):
            raise B19ContractError("B19 barrier-bound state changed")
    accepted = payload.get("accepted_preflight")
    if not isinstance(accepted, Mapping) or accepted != {
        "artifact": accepted.get("artifact"),
        "sha256": accepted.get("sha256"),
        "formal_train_permission": True,
        "deployment_status": "DEVICE_PENDING",
    }:
        raise B19ContractError("B19 barrier preflight binding schema changed")
    _canonical_preflight_artifact(accepted["artifact"], accepted["sha256"])
    claim = payload.get("authorization_claim")
    if not isinstance(claim, Mapping) or set(claim) != {"path", "sha256"} or not _hex(claim.get("sha256"), 64):
        raise B19ContractError("B19 barrier authorization-claim binding changed")
    claim_path = Path(str(claim["path"])).resolve(strict=True)
    expected_claim = (
        _repository_root() / "runs" / "b19_train_authorization_claims" / f"{accepted['sha256']}.json"
    ).resolve()
    if claim_path != expected_claim or claim["path"] != str(claim_path) or sha256_file(claim_path) != claim["sha256"]:
        raise B19ContractError("B19 barrier authorization claim changed")


def _assert_metric_token(token: MetricBarrierToken) -> None:
    key = _metric_token_key(token.sha256, token.nonce)
    proof = _ACTIVE_BARRIER_SEALS.get(key)
    if (
        key not in _VALID_BARRIER_TOKENS
        or proof is None
        or proof.path != token.path
        or proof.sha256 != token.sha256
        or proof.nonce != token.nonce
    ):
        _revoke_metric_token(key)
        raise B19ContractError("Metrics require an unchanged validated B19 barrier")
    try:
        path = Path(token.path).resolve(strict=True)
        if str(path) != proof.path or sha256_file(path) != proof.sha256:
            raise B19ContractError("B19 active private metric-barrier seal proof changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_barrier_payload(payload, path)
        expected_git = {
            "head": payload["git_head"],
            "branch": EXPECTED_BRANCH,
            "status": "",
            "clean": True,
            "head_is_commit": True,
        }
        if _git_contract() != expected_git or _source_hashes() != payload["source_hashes"]:
            raise B19ContractError("B19 git/source bindings changed after metric barrier sealing")
    except B19ContractError:
        _revoke_metric_token(key)
        raise
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _revoke_metric_token(key)
        raise B19ContractError("Metrics require unchanged B19 barrier state/preflight artifacts") from exc


def _metric_backend(token: MetricBarrierToken) -> dict[str, Callable[..., Any]]:
    global _METRIC_BACKEND_LOADED
    _assert_metric_token(token)
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _METRIC_BACKEND_LOADED = True
    return {
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
    }


def classification_summary(
    labels: np.ndarray,
    logits: np.ndarray,
    token: MetricBarrierToken,
    folds: np.ndarray | None = None,
) -> dict[str, object]:
    backend = _metric_backend(token)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if labels.ndim != 1 or logits.shape != (labels.size, CLASSES) or (not np.isfinite(logits).all()):
        raise ValueError("B19 labels/logits are misaligned or non-finite")
    predictions = logits.argmax(axis=1).astype(np.int64)
    matrix = backend["confusion_matrix"](labels, predictions, labels=np.arange(CLASSES))
    (precision, recall, f1, support) = backend["precision_recall_fscore_support"](
        labels, predictions, labels=np.arange(CLASSES), zero_division=0
    )
    pairs: dict[str, object] = {}
    for rival in RIVALS:
        mask = (labels == CLASS1) | (labels == rival)
        binary = (labels[mask] == CLASS1).astype(np.int64)
        margin = logits[mask, CLASS1] - logits[mask, rival]
        pairs[str(rival)] = {
            "auroc": float(backend["roc_auc_score"](binary, margin)),
            "rows": int(mask.sum()),
        }
    result: dict[str, object] = {
        "accuracy": float(backend["accuracy_score"](labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "class1_precision": float(precision[CLASS1]),
        "class1_recall": float(recall[CLASS1]),
        "class1_f1": float(f1[CLASS1]),
        "class1_tp": int(matrix[1, 1]),
        "class1_fp": int(matrix[:, 1].sum() - matrix[1, 1]),
        "class1_fn": int(matrix[1, :].sum() - matrix[1, 1]),
        "transition_1_to_2": int(matrix[1, 2]),
        "transition_2_to_1": int(matrix[2, 1]),
        "restricted_0_2_4_to_1": int(sum((matrix[rival, 1] for rival in RIVALS))),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": [
            {
                "class": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(CLASSES)
        ],
        "pairs": pairs,
        "mean_pair_auroc": float(np.mean([pairs[str(rival)]["auroc"] for rival in RIVALS])),
    }
    if folds is not None:
        fold_values = np.asarray(folds, dtype=np.int64)
        if fold_values.shape != labels.shape:
            raise ValueError("B19 fold vector is misaligned")
        result["folds"] = [
            {
                "fold": fold,
                **classification_summary(labels[fold_values == fold], logits[fold_values == fold], token),
            }
            for fold in range(FOLDS)
        ]
    return result


def _fast_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    matrix = np.bincount(labels * CLASSES + predictions, minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    values = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return (float(values.mean()), float(values[CLASS1]))


def paired_component_bootstrap(
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    logits: Mapping[str, np.ndarray],
    token: MetricBarrierToken,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    enforce_locked_draw_hash: bool = True,
) -> dict[str, object]:
    backend = _metric_backend(token)
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    scores = {name: np.asarray(value, dtype=np.float64) for (name, value) in logits.items()}
    if (
        set(scores) != set(OOF_NAMES)
        or labels.shape != folds.shape
        or labels.shape != groups.shape
        or any((value.shape != (labels.size, CLASSES) or not np.isfinite(value).all() for value in scores.values()))
    ):
        raise ValueError("B19 bootstrap inputs are invalid")
    members: dict[int, np.ndarray] = {}
    fold_groups: dict[int, list[int]] = {fold: [] for fold in range(FOLDS)}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        observed = np.unique(folds[positions])
        if observed.size != 1:
            raise B19ContractError("Bootstrap union component crosses folds")
        members[int(group)] = positions
        fold_groups[int(observed[0])].append(int(group))
    comparisons = {
        "candidate_vs_control": ("surfacefold_spatial_candidate", "surfacefold_mean_control"),
        "candidate_vs_stock": ("surfacefold_spatial_candidate", "stock_relation"),
        "candidate_vs_delta_off": ("surfacefold_spatial_candidate", "candidate_delta_off"),
    }
    metric_names = (
        "macro_f1_delta",
        "class1_f1_delta",
        "pair0_auroc_delta",
        "pair2_auroc_delta",
        "pair4_auroc_delta",
        "mean_pair_auroc_delta",
    )
    values = {
        comparison: {metric: np.empty(int(replicates), dtype=np.float64) for metric in metric_names}
        for comparison in comparisons
    }
    predictions = {name: value.argmax(axis=1).astype(np.int64) for (name, value) in scores.items()}
    rng = np.random.default_rng(int(seed))
    draw_hash = hashlib.sha256()
    for replicate in range(int(replicates)):
        chunks: list[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            if not available:
                raise B19ContractError(f"Bootstrap fold {fold} has no component")
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draw_hash.update(np.asarray((replicate, fold), dtype="<i8").tobytes())
            draw_hash.update(draw.astype("<i8").tobytes())
            chunks.extend((members[available[int(position)]] for position in draw))
        selected = np.concatenate(chunks)
        selected_labels = labels[selected]
        arm_metrics: dict[str, dict[str, float]] = {}
        for arm in OOF_NAMES:
            (macro, class1) = _fast_f1(selected_labels, predictions[arm][selected])
            pair_aucs: dict[int, float] = {}
            for rival in RIVALS:
                mask = (selected_labels == CLASS1) | (selected_labels == rival)
                binary = (selected_labels[mask] == CLASS1).astype(np.int64)
                selected_scores = scores[arm][selected]
                margin = selected_scores[mask, CLASS1] - selected_scores[mask, rival]
                pair_aucs[rival] = float(backend["roc_auc_score"](binary, margin))
            arm_metrics[arm] = {
                "macro_f1": macro,
                "class1_f1": class1,
                **{f"pair{rival}_auroc": pair_aucs[rival] for rival in RIVALS},
                "mean_pair_auroc": float(np.mean(list(pair_aucs.values()))),
            }
        for comparison, (left, right) in comparisons.items():
            for metric in metric_names:
                base = metric.removesuffix("_delta")
                values[comparison][metric][replicate] = arm_metrics[left][base] - arm_metrics[right][base]
    digest = draw_hash.hexdigest()
    if enforce_locked_draw_hash and (
        int(replicates) != BOOTSTRAP_REPLICATES
        or int(seed) != BOOTSTRAP_SEED
        or digest != EXPECTED_BOOTSTRAP_DRAW_SHA256
    ):
        raise B19ContractError(f"B19 bootstrap draw contract changed: {digest}")
    if any((not np.isfinite(sample).all() for comparison in values.values() for sample in comparison.values())):
        raise B19ContractError("B19 bootstrap produced a non-finite replicate")
    intervals = {
        comparison: {
            metric: {
                "lower": float(np.quantile(sample, 0.025, method="linear")),
                "upper": float(np.quantile(sample, 0.975, method="linear")),
            }
            for (metric, sample) in metrics.items()
        }
        for (comparison, metrics) in values.items()
    }
    return {
        "method": "paired_fold_stratified_union_component_percentile_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "draws_int64_sha256": digest,
        "quantile_method": "linear",
        "values": values,
        "intervals": intervals,
    }


def assess_b19_gate(
    summaries: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, object],
    token: MetricBarrierToken,
    *,
    integrity_complete: bool,
) -> dict[str, object]:
    _assert_metric_token(token)
    if set(summaries) != set(OOF_NAMES):
        raise ValueError("B19 gate requires all four OOF summaries")
    stock = summaries["stock_relation"]
    control = summaries["surfacefold_mean_control"]
    candidate = summaries["surfacefold_spatial_candidate"]
    delta_off = summaries["candidate_delta_off"]
    intervals = bootstrap["intervals"]

    def delta(metric: str, left: Mapping[str, object], right: Mapping[str, object]) -> float:
        return float(left[metric]) - float(right[metric])

    c1_control = delta("class1_f1", candidate, control)
    c1_stock = delta("class1_f1", candidate, stock)
    c1_off = delta("class1_f1", candidate, delta_off)
    pair2 = float(candidate["pairs"]["2"]["auroc"]) - float(control["pairs"]["2"]["auroc"])
    mean_pair = delta("mean_pair_auroc", candidate, control)
    control_restricted = int(control["restricted_0_2_4_to_1"])
    control_2_to_1 = int(control["transition_2_to_1"])
    restricted_reduction = (
        (control_restricted - int(candidate["restricted_0_2_4_to_1"])) / control_restricted
        if control_restricted > 0
        else None
    )
    two_to_one_reduction = (
        (control_2_to_1 - int(candidate["transition_2_to_1"])) / control_2_to_1 if control_2_to_1 > 0 else None
    )
    candidate_folds = {int(row["fold"]): row for row in candidate["folds"]}
    control_folds = {int(row["fold"]): row for row in control["folds"]}
    stock_folds = {int(row["fold"]): row for row in stock["folds"]}
    wins_control = sum(
        (float(candidate_folds[fold]["class1_f1"]) > float(control_folds[fold]["class1_f1"]) for fold in range(FOLDS))
    )
    wins_stock = sum(
        (float(candidate_folds[fold]["class1_f1"]) > float(stock_folds[fold]["class1_f1"]) for fold in range(FOLDS))
    )
    checks = {
        "absolute_engineering_floors": float(candidate["class1_f1"]) >= 0.7 and float(candidate["macro_f1"]) >= 0.8,
        "class1_f1_vs_control": c1_control >= 0.01
        and float(intervals["candidate_vs_control"]["class1_f1_delta"]["lower"]) > 0.0,
        "class1_f1_vs_stock": c1_stock >= 0.005
        and float(intervals["candidate_vs_stock"]["class1_f1_delta"]["lower"]) > 0.0,
        "pair_and_mean_auroc_vs_control": pair2 >= 0.005
        and mean_pair >= 0.003
        and (float(intervals["candidate_vs_control"]["pair2_auroc_delta"]["lower"]) > 0.0)
        and (float(intervals["candidate_vs_control"]["mean_pair_auroc_delta"]["lower"]) > 0.0),
        "macro_lcb_noninferiority": float(intervals["candidate_vs_control"]["macro_f1_delta"]["lower"]) >= -0.002
        and float(intervals["candidate_vs_stock"]["macro_f1_delta"]["lower"]) >= -0.002,
        "recall_and_transitions": float(candidate["class1_recall"]) >= float(control["class1_recall"]) - 0.005
        and float(candidate["class1_recall"]) >= float(stock["class1_recall"]) - 0.005
        and (restricted_reduction is not None)
        and (restricted_reduction >= 0.1)
        and (two_to_one_reduction is not None)
        and (two_to_one_reduction >= 0.1)
        and (int(candidate["restricted_0_2_4_to_1"]) <= int(stock["restricted_0_2_4_to_1"]))
        and (int(candidate["transition_2_to_1"]) <= int(stock["transition_2_to_1"]))
        and (int(candidate["transition_1_to_2"]) <= int(control["transition_1_to_2"]))
        and (int(candidate["transition_1_to_2"]) <= int(stock["transition_1_to_2"])),
        "fold_wins": wins_control >= 4 and wins_stock >= 3,
        "same_checkpoint_delta_off": c1_off >= 0.003
        and float(intervals["candidate_vs_delta_off"]["class1_f1_delta"]["lower"]) > 0.0
        and (int(candidate["transition_2_to_1"]) <= int(delta_off["transition_2_to_1"])),
        "integrity_complete": bool(integrity_complete),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "validation_protocol_permission": passed,
        "validation_execution_permission": False,
        "test_permission": False,
        "deployment_status": "DEVICE_PENDING",
        "checks": checks,
        "failed_checks": [name for (name, value) in checks.items() if not value],
        "denominator_valid": {
            "restricted_reduction_vs_control": control_restricted > 0,
            "2_to_1_reduction_vs_control": control_2_to_1 > 0,
        },
        "point_deltas": {
            "class1_f1_vs_control": c1_control,
            "class1_f1_vs_stock": c1_stock,
            "class1_f1_vs_delta_off": c1_off,
            "macro_f1_vs_control": delta("macro_f1", candidate, control),
            "macro_f1_vs_stock": delta("macro_f1", candidate, stock),
            "pair2_auroc_vs_control": pair2,
            "mean_pair_auroc_vs_control": mean_pair,
            "restricted_reduction_vs_control": restricted_reduction,
            "2_to_1_reduction_vs_control": two_to_one_reduction,
            "fold_wins_vs_control": wins_control,
            "fold_wins_vs_stock": wins_stock,
        },
    }


def _configure_determinism(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available() or (not torch.cuda.is_bf16_supported()):
        raise B19ContractError("Formal B19 requires a BF16-capable CUDA GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _directory_bytes(path: Path) -> int:
    return sum((item.stat().st_size for item in Path(path).rglob("*") if item.is_file()))


def _folded_parity(model: SwiftFormerSurfaceFoldB17) -> float:
    generator = torch.Generator(device="cpu").manual_seed(17017)
    inputs = torch.randn((1, 3, 224, 224), generator=generator)
    active = copy.deepcopy(model).cpu().eval()
    folded = active.fold_to_deploy().cpu().eval()
    with torch.inference_mode():
        left = active(inputs).float()
        right = folded(inputs).float()
    maximum = float((left - right).abs().max())
    if not math.isfinite(maximum) or maximum > 1e-06:
        raise B19ContractError(f"Trained B19 folded parity failed: {maximum}")
    return maximum


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked TRAIN-only B19 SurfaceFold-XS 5-fold/3-arm OOF screen")
    parser.add_argument("--confirm-protocol-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--student-weight", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _validated_output(path: Path) -> Path:
    output = Path(path).expanduser().resolve()
    runs = (_repository_root() / "runs").resolve()
    if output.parent != runs or not output.name.startswith(OUTPUT_PREFIX):
        raise B19ContractError("B19 output must be a fresh direct runs child")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Refuse to overwrite B19 output: {output}")
    return output


def _save_failure(output: Path, exc: BaseException, started: float) -> None:
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "failed": True,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "monotonic_wall_seconds": time.monotonic() - started,
        "deployment_status": "DEVICE_PENDING",
        "validation_used": False,
        "test_used": False,
    }
    digest = atomic_json(output / "failure.json", payload)
    _atomic_bytes(output / "failure.json.sha256", f"{digest}\n".encode("ascii"))


def _build_model_for_state(base: nn.Module, arm: str) -> SwiftFormerSurfaceFoldB17:
    if arm not in ARMS:
        raise B19ContractError(f"Unknown B19 arm: {arm}")
    model = _preserve_rng(lambda: SwiftFormerSurfaceFoldB17(copy.deepcopy(base), ARM_MODES[arm]))
    if sum((parameter.numel() for parameter in model.parameters())) != ARM_PARAMETERS[arm]:
        raise B19ContractError(f"B19 reload model parameter count changed: {arm}")
    return model


def _rng_state_sha256(state: ArmRngState) -> str:
    return _tensor_sha256(state.cpu, state.cuda)


def run_formal(args: argparse.Namespace) -> dict[str, object]:
    if args.confirm_protocol_id != PROTOCOL_ID:
        raise B19ContractError("Explicit B19 protocol confirmation is missing")
    git = _git_contract()
    if git["branch"] != EXPECTED_BRANCH or git["clean"] is not True or git["head_is_commit"] is not True:
        raise B19ContractError(f"Formal B19 requires a clean committed canonical branch: {git}")
    sources = _source_hashes()
    accepted = validate_accepted_preflight(
        args.preflight_artifact,
        args.preflight_sha256,
        args.student_weight,
        args.dino_weight,
        git=git,
        sources=sources,
    )
    output = _validated_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        authorization_claim = claim_train_authorization_once(
            accepted_preflight=accepted,
            git=git,
            sources=sources,
            output=output,
            args=args,
        )
        formal_rng_probe = _formal_initialized_cuda_head_reset_probe()
        data_root = _read_locked_data_root(args.data)
        assignment = read_locked_assignment_csv(args.assignment_csv)
        relative_paths = assignment["relative_paths"]
        labels = assignment["labels"]
        groups = assignment["groups"]
        folds = assignment["folds"]
        derived_updates_per_epoch = derive_locked_updates_per_epoch(folds)
        train_root = (data_root / "train").resolve()
        absolute_paths: list[Path] = []
        for relative in relative_paths:
            path = (data_root / relative).resolve(strict=True)
            try:
                path.relative_to(train_root)
            except ValueError as exc:
                raise B19ContractError(f"Non-TRAIN path in B19 assignment: {path}") from exc
            absolute_paths.append(path)
        start_content = train_content_contract(relative_paths, absolute_paths)
        atomic_json(
            output / "contract.json",
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "git": git,
                "source_hashes": sources,
                "accepted_preflight": {
                    "artifact": accepted["artifact"],
                    "sha256": accepted["sha256"],
                },
                "authorization_claim": {
                    "path": authorization_claim["path"],
                    "sha256": authorization_claim["sha256"],
                },
                "formal_initialized_cuda_head_reset": formal_rng_probe,
                "data_yaml": str(Path(args.data).resolve()),
                "data_yaml_sha256": EXPECTED_DATA_SHA256,
                "assignment_csv": str(Path(args.assignment_csv).resolve()),
                "assignment_csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
                "assignment_derived_updates_per_epoch": list(derived_updates_per_epoch),
                "train_content_start": start_content,
                "deployment_status": "DEVICE_PENDING",
                "train": True,
                "validation": False,
                "test": False,
            },
        )
        atomic_json(output / "ledger" / "paths.json", {"paths": list(relative_paths)})
        atomic_npz(output / "ledger" / "labels_folds_groups.npz", labels=labels, folds=folds, groups=groups)
        device = torch.device("cuda")
        _configure_determinism(device)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        (base, teacher) = _load_templates(args.student_weight, args.dino_weight)
        teacher.to(device).eval()
        records: list[FinalStateRecord] = []
        peak_allocated = 0
        immutable_projection_complete = True
        folded_parity_complete = True
        for fold in range(FOLDS):
            fold_seed = SEED + fold
            fit = folds != fold
            fit_indices = np.flatnonzero(fit).astype(np.int64)
            (fit_counts, class_weights_cpu) = sqrt_inverse_class_weights(labels, fit)
            class_weights = class_weights_cpu.to(device)
            shuffle_generator = torch.Generator(device="cpu").manual_seed(fold_seed)
            augmentation_generator = torch.Generator(device="cpu").manual_seed(AUGMENTATION_SEED_BASE + fold)
            arms = {arm: model.to(device).train() for (arm, model) in build_b19_arm_models(base).items()}
            initial_projection = {arm: _projection_hash(model) for (arm, model) in arms.items()}
            batches_per_epoch = math.ceil(fit_indices.size / BATCH_SIZE)
            updates_per_epoch = math.ceil(batches_per_epoch / ACCUMULATION_STEPS)
            if updates_per_epoch != derived_updates_per_epoch[fold]:
                raise B19ContractError("B19 fold optimizer horizon changed after formal lock")
            steppers: dict[str, LockedOptimizerStepper] = {}
            optimizer_contract: dict[str, object] = {}
            gradient_audit: dict[str, dict[str, int]] = {}
            for arm, model in arms.items():
                optimizer_groups = build_discriminative_adamw_groups(model)
                optimizer_contract[arm] = [
                    {key: value for (key, value) in group.items() if key != "params"} for group in optimizer_groups
                ]
                optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999), eps=1e-08)
                steppers[arm] = LockedOptimizerStepper(optimizer, updates_per_epoch)
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                gradient_audit[arm] = {
                    "updates": 0,
                    "checked_parameter_tensors": 0,
                    "checked_parameter_elements": 0,
                    "parameter_tensors_per_update": len(trainable),
                    "parameter_elements_per_update": sum((parameter.numel() for parameter in trainable)),
                }
            arm_rng = make_arm_rng_states(device, fold_seed)
            teacher_ledger = TeacherCallLedger()
            exposure: dict[str, list[Any]] = {
                "indices": [],
                "crop": [],
                "flip": [],
                "epoch": [],
                "batch": [],
                "window": [],
                "divisor": [],
            }
            loss_rows: list[tuple[float, ...]] = []
            torch.cuda.reset_peak_memory_stats(device)
            for epoch in range(EPOCHS):
                order = fit_indices[torch.randperm(fit_indices.size, generator=shuffle_generator).numpy()]
                batches = [order[start : start + BATCH_SIZE].tolist() for start in range(0, order.size, BATCH_SIZE)]
                for window_start in range(0, len(batches), ACCUMULATION_STEPS):
                    window = batches[window_start : window_start + ACCUMULATION_STEPS]
                    divisor = len(window)
                    for stepper in steppers.values():
                        stepper.zero_grad()
                    for within, batch_indices in enumerate(window):
                        (images, batch_labels, observed) = load_raw_batch(absolute_paths, labels, batch_indices)
                        if observed.tolist() != batch_indices or any((not fit[index] for index in batch_indices)):
                            raise B19ContractError("B19 fit batch order/split exposure changed")
                        (student_cpu, teacher_cpu, crops) = build_shared_batch_views(images, augmentation_generator)
                        for image in images:
                            image.close()
                        student_images = student_cpu.to(device)
                        teacher_images = teacher_cpu.to(device)
                        batch_labels = batch_labels.to(device)
                        target = teacher_relation_target(teacher, teacher_images, teacher_ledger)
                        batch_number = window_start + within
                        for index, crop in zip(batch_indices, crops):
                            exposure["indices"].append(index)
                            exposure["crop"].append((crop.top, crop.left, crop.height, crop.width))
                            exposure["flip"].append(crop.horizontal_flip)
                            exposure["epoch"].append(epoch)
                            exposure["batch"].append(batch_number)
                            exposure["window"].append(window_start // ACCUMULATION_STEPS)
                            exposure["divisor"].append(divisor)
                        losses: dict[str, tuple[float, float, float]] = {}
                        for arm in ARMS:
                            model = arms[arm]

                            def forward_backward() -> tuple[float, float, float]:
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    (logits, s3) = model(student_images, return_s3=True)
                                total, ce, relation = exact_b19_objective(
                                    logits,
                                    s3,
                                    batch_labels,
                                    class_weights,
                                    target,
                                )
                                (total / divisor).backward()
                                return (float(ce), float(relation), float(total))

                            losses[arm] = run_with_arm_rng(arm, arm_rng, device, forward_backward)
                        loss_rows.append(
                            (
                                float(epoch),
                                float(batch_number),
                                *(value for arm in ARMS for value in losses[arm]),
                            )
                        )
                    for arm in ARMS:

                        def update_arm(
                            model: nn.Module = arms[arm],
                            stepper: LockedOptimizerStepper = steppers[arm],
                        ) -> dict[str, object]:
                            audited = audit_and_clip_gradients(model, GRAD_CLIP)
                            applied_lrs = stepper.step()
                            return {"audit": audited, "lrs": applied_lrs}

                        update = run_with_arm_rng(arm, arm_rng, device, update_arm)
                        audited = update["audit"]
                        gradient_audit[arm]["updates"] += 1
                        gradient_audit[arm]["checked_parameter_tensors"] += int(audited["parameter_tensors"])
                        gradient_audit[arm]["checked_parameter_elements"] += int(audited["parameter_elements"])
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise TimeoutError("B19 formal wall-time budget exceeded during training")
            expected_calls = EPOCHS * batches_per_epoch
            if (
                teacher_ledger.calls != expected_calls
                or teacher_ledger.rows != EPOCHS * int(fit.sum())
                or any((parameter.grad is not None for parameter in teacher.parameters()))
            ):
                raise B19ContractError("DINO was not called exactly once per fit batch or acquired gradients")
            exposure_counts = np.bincount(
                np.asarray(exposure["indices"], dtype=np.int64), minlength=EXPECTED_TRAIN_ROWS
            )
            if not (np.all(exposure_counts[fit] == EPOCHS) and np.all(exposure_counts[~fit] == 0)):
                raise B19ContractError("B19 exposure is not exact natural fit-only")
            for arm in ARMS:
                evidence = steppers[arm].evidence(require_complete=True)
                audit = gradient_audit[arm]
                if (
                    evidence["completed_steps"] != evidence["total_steps"]
                    or audit["updates"] != steppers[arm].total_steps
                    or audit["checked_parameter_tensors"] != audit["updates"] * audit["parameter_tensors_per_update"]
                    or (
                        audit["checked_parameter_elements"]
                        != audit["updates"] * audit["parameter_elements_per_update"]
                    )
                    or (_projection_hash(arms[arm]) != initial_projection[arm])
                ):
                    immutable_projection_complete = False
                    raise B19ContractError(f"B19 scheduler/gradient/W0-b0 contract failed: {arm}")
            fold_root = output / "folds" / f"fold_{fold}"
            exposure_sha = atomic_npz(
                fold_root / "exposure.npz",
                indices=np.asarray(exposure["indices"], dtype=np.int64),
                crop=np.asarray(exposure["crop"], dtype=np.int32),
                flip=np.asarray(exposure["flip"], dtype=np.bool_),
                epoch=np.asarray(exposure["epoch"], dtype=np.int16),
                batch=np.asarray(exposure["batch"], dtype=np.int32),
                accumulation_window=np.asarray(exposure["window"], dtype=np.int32),
                accumulation_divisor=np.asarray(exposure["divisor"], dtype=np.int8),
            )
            loss_sha = atomic_npz(fold_root / "loss_curves.npz", rows=np.asarray(loss_rows, dtype=np.float64))
            mechanism = {}
            for arm, model in arms.items():
                delta = model.effective_delta_weight().detach().float()
                row: dict[str, object] = {
                    "delta_norm": float(delta.norm()),
                    "projection_sha256": _projection_hash(model),
                }
                if model.factor_d is not None:
                    centered = model.factor_d.detach().float() - model.factor_d.detach().float().mean(
                        dim=(1, 2), keepdim=True
                    )
                    row["factor_p_norm"] = float(model.factor_p.detach().float().norm())
                    row["factor_d_centered_norm"] = float(centered.norm())
                if any((not _finite(value) for value in row.values() if isinstance(value, float))):
                    raise B19ContractError("B19 mechanism telemetry is non-finite")
                mechanism[arm] = row
            atomic_json(
                fold_root / "train_contract.json",
                {
                    "fold": fold,
                    "fold_seed": fold_seed,
                    "augmentation_seed": AUGMENTATION_SEED_BASE + fold,
                    "fit_samples": int(fit.sum()),
                    "fit_class_counts": fit_counts.tolist(),
                    "class_weights": class_weights_cpu.tolist(),
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "accumulation_steps": ACCUMULATION_STEPS,
                    "workers": WORKERS,
                    "batches_per_epoch": batches_per_epoch,
                    "updates_per_epoch": updates_per_epoch,
                    "teacher_calls": teacher_ledger.calls,
                    "gradient_audit": gradient_audit,
                    "optimizer_groups": optimizer_contract,
                    "scheduler": {arm: steppers[arm].evidence(require_complete=True) for arm in ARMS},
                    "arm_rng_final_sha256": {arm: _rng_state_sha256(arm_rng[arm]) for arm in ARMS},
                    "mechanism": mechanism,
                    "held_inference": False,
                },
            )
            for arm in ARMS:
                model = arms[arm].cpu().eval()
                parity = _folded_parity(model)
                folded_parity_complete &= parity <= 1e-06
                state_path = (output / "states" / f"fold_{fold}_{arm}.safetensors").resolve()
                metadata = _metadata_for_state(fold, arm)
                state_sha = atomic_safetensors(state_path, model.state_dict(), metadata)
                records.append(
                    FinalStateRecord(
                        state_id=f"fold_{fold}:{arm}",
                        fold=fold,
                        arm=arm,
                        path=str(state_path),
                        sha256=state_sha,
                        bytes=state_path.stat().st_size,
                        parameter_count=ARM_PARAMETERS[arm],
                        trainable_parameter_count=ARM_TRAINABLE_PARAMETERS[arm],
                        mode=ARM_MODES[arm],
                        metadata_sha256=_metadata_sha256(metadata),
                        exposure_sha256=exposure_sha,
                        loss_curve_sha256=loss_sha,
                        frozen_projection_sha256=_projection_hash(model),
                        folded_max_abs_error=parity,
                    )
                )
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            if peak_allocated > MAX_CUDA_ALLOCATED_BYTES:
                raise B19ContractError("B19 peak CUDA allocation exceeded 7 GiB")
            if _directory_bytes(output) > MAX_RETAINED_BYTES:
                raise B19ContractError("B19 retained artifact budget exceeded 1.5 GiB")
            del arms, steppers
            torch.cuda.empty_cache()
        teacher.cpu()
        del teacher
        torch.cuda.empty_cache()

        def strict_reload(record: FinalStateRecord) -> None:
            model = _build_model_for_state(base, record.arm)
            incompatible = model.load_state_dict(load_file(record.path, device="cpu"), strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise B19ContractError(f"Strict B19 reload failed: {record.state_id}")
            if _projection_hash(model) != record.frozen_projection_sha256 or any(
                (
                    not bool(torch.isfinite(tensor).all())
                    for tensor in model.state_dict().values()
                    if tensor.is_floating_point()
                )
            ):
                raise B19ContractError(f"Reloaded B19 state is invalid: {record.state_id}")

        def cross_mode_rejection() -> None:
            candidate_path = output / "states" / "fold_0_surfacefold_spatial_candidate.safetensors"
            control_path = output / "states" / "fold_0_surfacefold_mean_control.safetensors"
            for model_arm, state_path in (
                ("surfacefold_mean_control", candidate_path),
                ("surfacefold_spatial_candidate", control_path),
            ):
                model = _build_model_for_state(base, model_arm)
                try:
                    model.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
                except RuntimeError:
                    continue
                raise B19ContractError("B19 cross-mode strict load unexpectedly passed")

        counters = dict(ZERO_BARRIER_COUNTERS)
        token = seal_metric_barrier(
            output,
            records,
            accepted_preflight=accepted,
            authorization_claim=authorization_claim,
            args=args,
            git=git,
            sources=sources,
            counters=counters,
            strict_reload=strict_reload,
            cross_mode_rejection=cross_mode_rejection,
        )
        counters["oof_buffer_count"] += 1
        logits = {name: np.full((EXPECTED_TRAIN_ROWS, CLASSES), np.nan, dtype=np.float32) for name in OOF_NAMES}
        fill = {name: np.zeros(EXPECTED_TRAIN_ROWS, dtype=np.int8) for name in OOF_NAMES}
        active_off_l2 = np.full(EXPECTED_TRAIN_ROWS, np.nan, dtype=np.float32)
        for fold in range(FOLDS):
            held_indices = np.flatnonzero(folds == fold).astype(np.int64)
            batches = [
                held_indices[start : start + BATCH_SIZE].tolist() for start in range(0, held_indices.size, BATCH_SIZE)
            ]
            loaded: dict[str, SwiftFormerSurfaceFoldB17] = {}
            for arm in ARMS:
                model = _build_model_for_state(base, arm)
                model.load_state_dict(
                    load_file(str(output / "states" / f"fold_{fold}_{arm}.safetensors"), device="cpu"),
                    strict=True,
                )
                loaded[arm] = model.to(device).eval()
            with torch.inference_mode():
                for batch_indices in batches:
                    (images, _batch_labels, observed) = load_raw_batch(absolute_paths, labels, batch_indices)
                    counters["held_image_open_count"] += len(images)
                    if observed.tolist() != batch_indices:
                        raise B19ContractError("Held B19 row order changed")
                    inputs = torch.stack([build_eval_view(image) for image in images]).to(device)
                    for image in images:
                        image.close()
                    for arm in ARMS:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            output_logits = loaded[arm](inputs)
                        counters["held_forward_count"] += 1
                        values = output_logits.float().cpu().numpy()
                        logits[arm][batch_indices] = values
                        fill[arm][batch_indices] += 1
                        counters["logit_write_count"] += len(batch_indices)
                    candidate = loaded["surfacefold_spatial_candidate"]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        delta_off_logits = candidate.backbone(inputs)
                    counters["held_forward_count"] += 1
                    off_values = delta_off_logits.float().cpu().numpy()
                    logits["candidate_delta_off"][batch_indices] = off_values
                    fill["candidate_delta_off"][batch_indices] += 1
                    counters["logit_write_count"] += len(batch_indices)
                    active_off_l2[batch_indices] = np.linalg.norm(
                        logits["surfacefold_spatial_candidate"][batch_indices] - off_values, axis=1
                    ).astype(np.float32)
            for model in loaded.values():
                model.cpu()
            del loaded
            torch.cuda.empty_cache()
        if (
            any((not np.isfinite(value).all() for value in logits.values()))
            or any((not np.all(value == 1) for value in fill.values()))
            or (not np.isfinite(active_off_l2).all())
            or (counters["held_image_open_count"] != EXPECTED_TRAIN_ROWS)
        ):
            raise B19ContractError("B19 OOF logits/telemetry are incomplete or duplicated")
        telemetry_npz_sha = atomic_npz(
            output / "oof" / "candidate_delta_off_telemetry_f32.npz",
            active_off_logit_l2=active_off_l2,
        )
        telemetry_json_sha = atomic_json(
            output / "oof" / "candidate_delta_off_telemetry.json",
            {
                "rows": EXPECTED_TRAIN_ROWS,
                "selection_or_tuning_use": False,
                "active_off_logit_l2_p50": float(np.quantile(active_off_l2, 0.5, method="linear")),
                "active_off_logit_l2_p95": float(np.quantile(active_off_l2, 0.95, method="linear")),
                "finite": True,
                "npz_sha256": telemetry_npz_sha,
            },
        )
        oof_sha = atomic_npz(
            output / "oof" / "oof_logits_f32.npz",
            labels=labels,
            folds=folds,
            groups=groups,
            **{f"logits_{name}": value for (name, value) in logits.items()},
            **{f"fill_{name}": value for (name, value) in fill.items()},
        )
        counters["metric_call_count"] += 1
        summaries = {name: classification_summary(labels, value, token, folds) for (name, value) in logits.items()}
        metrics_sha = atomic_json(output / "metrics.json", summaries)
        bootstrap = paired_component_bootstrap(labels, folds, groups, logits, token)
        bootstrap_npz_sha = atomic_npz(
            output / "bootstrap_replicates_f64.npz",
            **{
                f"{comparison}__{metric}": sample
                for (comparison, metrics) in bootstrap["values"].items()
                for (metric, sample) in metrics.items()
            },
        )
        bootstrap_json_sha = atomic_json(
            output / "bootstrap.json",
            {key: value for (key, value) in bootstrap.items() if key != "values"},
        )
        end_content = train_content_contract(relative_paths, absolute_paths)
        end_git = _git_contract()
        end_sources = _source_hashes()
        end_preflight = validate_accepted_preflight(
            args.preflight_artifact,
            args.preflight_sha256,
            args.student_weight,
            args.dino_weight,
            git=git,
            sources=sources,
        )
        end_claim = validate_train_authorization_claim(
            authorization_claim,
            accepted_preflight=accepted,
            git=git,
            sources=sources,
            output=output,
            args=args,
        )
        end_student = _validate_asset(args.student_weight, STUDENT_BYTES, STUDENT_SHA256)
        end_dino = _validate_asset(args.dino_weight, DINO_BYTES, DINO_SHA256)
        wall = time.monotonic() - started
        base_integrity = bool(
            start_content == end_content
            and end_git == git
            and (end_sources == sources)
            and (end_preflight == accepted)
            and (end_claim == authorization_claim)
            and (sha256_file(end_student) == STUDENT_SHA256)
            and (sha256_file(end_dino) == DINO_SHA256)
            and (peak_allocated <= MAX_CUDA_ALLOCATED_BYTES)
            and (wall <= MAX_WALL_SECONDS)
            and immutable_projection_complete
            and folded_parity_complete
            and all((np.all(value == 1) for value in fill.values()))
        )
        resource_sha = atomic_json(
            output / "resources.json",
            {
                "peak_cuda_allocated_bytes": peak_allocated,
                "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED_BYTES,
                "monotonic_wall_seconds": wall,
                "max_wall_seconds": MAX_WALL_SECONDS,
                "retained_bytes_before_final_metadata": _directory_bytes(output),
                "max_retained_bytes": MAX_RETAINED_BYTES,
                "device_status": "DEVICE_PENDING",
                "passed": base_integrity,
            },
        )
        gate = assess_b19_gate(summaries, bootstrap, token, integrity_complete=base_integrity)
        gate_bytes = _json_bytes(gate)
        gate_sha = hashlib.sha256(gate_bytes).hexdigest()
        before_final = _directory_bytes(output)

        def make_summary(final_bytes: int) -> dict[str, object]:
            return {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "passed": gate["passed"],
                "validation_protocol_permission": gate["validation_protocol_permission"],
                "validation_execution_permission": False,
                "test_permission": False,
                "deployment_status": "DEVICE_PENDING",
                "train": True,
                "validation": False,
                "test": False,
                "metric_barrier": {"path": token.path, "sha256": token.sha256},
                "authorization_claim": {
                    "path": end_claim["path"],
                    "sha256": end_claim["sha256"],
                    "exclusive_once": True,
                },
                "train_content": {"start": start_content, "end": end_content},
                "end_revalidation": {
                    "git_equal": end_git == git,
                    "source_hashes_equal": end_sources == sources,
                    "preflight_equal": end_preflight == accepted,
                    "authorization_claim_equal": end_claim == authorization_claim,
                    "student_asset_sha256": STUDENT_SHA256,
                    "teacher_asset_sha256": DINO_SHA256,
                },
                "resources": {
                    "peak_cuda_allocated_bytes": peak_allocated,
                    "monotonic_wall_seconds": wall,
                    "retained_bytes_final": int(final_bytes),
                    "max_retained_bytes": MAX_RETAINED_BYTES,
                },
                "post_barrier_counters": counters,
                "artifacts": {
                    "oof_sha256": oof_sha,
                    "metrics_sha256": metrics_sha,
                    "bootstrap_npz_sha256": bootstrap_npz_sha,
                    "bootstrap_json_sha256": bootstrap_json_sha,
                    "gate_sha256": gate_sha,
                    "resource_sha256": resource_sha,
                    "delta_off_telemetry_npz_sha256": telemetry_npz_sha,
                    "delta_off_telemetry_json_sha256": telemetry_json_sha,
                },
            }

        summary = make_summary(0)
        for _ in range(8):
            predicted = before_final + len(gate_bytes) + len(_json_bytes(summary))
            if summary["resources"]["retained_bytes_final"] == predicted:
                break
            summary = make_summary(predicted)
        predicted = int(summary["resources"]["retained_bytes_final"])
        if not base_integrity or predicted > MAX_RETAINED_BYTES:
            raise B19ContractError("B19 final integrity/resource gate failed")
        if _atomic_bytes(output / "gate.json", gate_bytes) != gate_sha:
            raise B19ContractError("B19 gate digest changed during promotion")
        atomic_json(output / "summary.json", summary)
        if _directory_bytes(output) != predicted:
            raise B19ContractError("B19 final retained-byte accounting changed")
        return summary
    except BaseException as exc:
        _save_failure(output, exc, started)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_formal(_parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

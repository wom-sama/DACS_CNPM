from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# This must precede the first CUDA/cuBLAS operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

from trkh.models.swiftformer_surface_b16 import (  # noqa: E402
    SURFACE_MEAN_CONTROL_B16_MODE,
    SURFACE_OFF_B16_MODE,
    SURFACE_SPATIAL_B16_MODE,
    SwiftFormerSurfaceB16,
    cosine_neighbor_relation_field,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B16_SWIFTSURFACE_XS_20260805"
PROTOCOL_SHA256 = "4184395eb2c599d5bbd8b0cfa0a44cd20a73864b2b509f8602b0c73573c99cde"
EXPECTED_BRANCH = "research/pretrained-classf-b1"
SEED = 20260805

EXPECTED_DATA_YAML = Path(r"D:\DataAI\AIEx\newdataset\class_f\data.yaml")
EXPECTED_DATA_SHA256 = "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8"
EXPECTED_TRAIN_ROWS = 8_278
EXPECTED_CLASS_COUNTS = (1_987, 497, 1_326, 2_080, 2_388)
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
STUDENT_BYTES = 13_957_760
STUDENT_SHA256 = "c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c"
DINO_TIMM_ID = "vit_small_patch16_dinov3.lvd1689m"
DINO_BYTES = 86_362_376
DINO_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DINO_PREFIX_TOKENS = 5
DINO_FEATURE_DIM = 384

FOLDS = 5
CLASSES = 5
CLASS1 = 1
RIVALS = (0, 2, 4)
ARMS = ("stock", "globalized_control", "swiftsurface_candidate")
STOCK_PARAMETERS = 3_035_570
CANDIDATE_PARAMETERS = 3_061_662
ARM_PARAMETERS = {
    "stock": STOCK_PARAMETERS,
    "globalized_control": CANDIDATE_PARAMETERS,
    "swiftsurface_candidate": CANDIDATE_PARAMETERS,
}

EPOCHS = 8
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
WORKERS = 0
BACKBONE_LR = 3.0e-5
TASK_LR = 3.0e-4
MIN_LR = 1.0e-6
WEIGHT_DECAY = 0.05
GRAD_CLIP = 0.7
RELATION_WEIGHT = 0.10
RELATION_BETA = 0.10
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260803
EXPECTED_BOOTSTRAP_DRAW_SHA256 = "d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469"
MAX_CUDA_ALLOCATED_BYTES = 7 * 1024**3
MAX_WALL_SECONDS = 12 * 60 * 60
MAX_RETAINED_BYTES = int(1.5 * 1024**3)
OUTPUT_PREFIX = "pretrained_swiftsurface_xs_b16_train_oof_"
ACCEPTED_PREFLIGHT_OUTPUT_PREFIX = "preflight_b16_swiftsurface_xs_"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_METRIC_BACKEND_LOADED = False
_VALID_BARRIER_TOKENS: set[str] = set()
_VERIFIED_BARRIER_STATE_TOKENS: set[str] = set()

METRIC_BARRIER_PHASE = "all_15_final_states_before_any_held_inference_or_metric"
METRIC_BARRIER_COUNTERS = {
    "held_image_open_count": 0,
    "held_forward_count": 0,
    "oof_buffer_count": 0,
    "logit_write_count": 0,
    "metric_call_count": 0,
}
METRIC_BARRIER_CHECKS = {
    "exact_15_states": True,
    "state_hashes_and_strict_reload": True,
    "accepted_qdq_preflight": True,
    "held_inference_absent": True,
    "logits_absent": True,
    "metrics_absent": True,
}


class B16ContractError(RuntimeError):
    """Raised whenever the prospective B16 contract is no longer exact."""


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
    mode: str
    exposure_sha256: str
    loss_curve_sha256: str


@dataclass(frozen=True)
class MetricBarrierToken:
    path: str
    sha256: str
    nonce: str


@dataclass
class TeacherCallLedger:
    calls: int = 0
    rows: int = 0


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_preflight_runs_root() -> Path:
    return (_repository_root() / "runs").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex64(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _hex40(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _atomic_bytes(path: Path, payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite B16 artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        digest = sha256_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise B16ContractError(f"Atomic artifact hash drifted: {path}")
    return digest


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    return _atomic_bytes(path, _json_bytes(payload))


def atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite B16 artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        digest = sha256_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise B16ContractError(f"Atomic NPZ hash drifted: {path}")
    return digest


def atomic_safetensors(path: Path, state: Mapping[str, Tensor]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite B16 state: {path}")
    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
    }
    if not tensors or any(not bool(torch.isfinite(value).all()) for value in tensors.values() if value.is_floating_point()):
        raise B16ContractError("B16 state is empty or contains non-finite tensors")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, str(temporary))
        digest = sha256_file(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(path) != digest:
        raise B16ContractError(f"Atomic state hash drifted: {path}")
    return digest


def _git_contract() -> dict[str, object]:
    root = _repository_root()

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    return {
        "head": head,
        "branch": branch,
        "status": status,
        "clean": status == "",
        "head_is_commit": _hex40(head),
    }


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = {
        "runner": Path(__file__).resolve(),
        "runner_test": root / "tests" / "test_run_swiftformer_surface_b16_train_oof.py",
        "model": root / "trkh" / "models" / "swiftformer_surface_b16.py",
        "protocol": root / "docs" / "TRKH_PRETRAINED_CLASSF_B16_SWIFTSURFACE_XS_PROTOCOL_20260805.md",
        "preflight": root / "trkh" / "tools" / "audit_swiftformer_surface_b16_preflight.py",
        "preflight_test": root / "tests" / "test_audit_swiftformer_surface_b16_preflight.py",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing B16 source: {name}={path}")
        relative = path.relative_to(root)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise B16ContractError(f"B16 source is not tracked: {relative}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes["protocol"] != PROTOCOL_SHA256:
        raise B16ContractError("B16 protocol hash changed")
    return hashes


def validate_qdq_preflight_payload(payload: Mapping[str, object]) -> None:
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("passed") is not True:
        raise B16ContractError("Accepted B16 preflight identity/pass flag is invalid")
    permissions = payload.get("permissions")
    required_false = {
        "dataset_yaml_read",
        "dataset_path_enumerated",
        "train_image_read",
        "real_label_read",
        "prior_prediction_read",
        "validation_constructed",
        "test_constructed",
        "network_used",
        "formal_train_permission",
    }
    if (
        not isinstance(permissions, Mapping)
        or set(permissions) != required_false
        or any(permissions.get(key) is not False for key in required_false)
    ):
        raise B16ContractError("Accepted preflight is not label/data/network free")
    deployment = payload.get("deployment")
    if not isinstance(deployment, Mapping) or deployment.get("passed") is not True:
        raise B16ContractError("Accepted preflight deployment gate did not pass")
    int8 = deployment.get("int8")
    if not isinstance(int8, Mapping) or set(int8) != {"stock", "control", "candidate"}:
        raise B16ContractError("Accepted preflight lacks all three INT8 graphs")
    for arm in ("stock", "control", "candidate"):
        value = int8[arm]
        if not isinstance(value, Mapping):
            raise B16ContractError(f"Invalid INT8 preflight record: {arm}")
        qdq = value.get("qdq")
        coverage = value.get("coverage")
        if not (
            isinstance(qdq, Mapping)
            and qdq.get("format") == "QDQ"
            and _strict_int(qdq.get("quantize_linear"))
            and int(qdq["quantize_linear"]) > 0
            and _strict_int(qdq.get("dequantize_linear"))
            and int(qdq["dequantize_linear"]) > 0
            and isinstance(coverage, Mapping)
            and _finite_number(coverage.get("ratio"))
            and float(coverage["ratio"]) >= 0.90
        ):
            raise B16ContractError(f"Accepted preflight QDQ/coverage failed: {arm}")
    latency = deployment.get("latency")
    if not isinstance(latency, Mapping) or latency.get("passed") is not True:
        raise B16ContractError("Accepted preflight latency gate did not pass")
    cuda_fit = payload.get("cuda_fit")
    if not isinstance(cuda_fit, Mapping) or cuda_fit.get("passed") is not True:
        raise B16ContractError("Accepted preflight CUDA-fit gate did not pass")


def validate_accepted_preflight(
    artifact: Path,
    expected_sha256: str,
    student_weight: Path,
    dino_weight: Path,
) -> dict[str, object]:
    # Kept lazy so the formal runner does not import ONNX/ORT before it has an
    # explicit accepted-preflight argument.
    from trkh.tools.audit_swiftformer_surface_b16_preflight import (
        validate_accepted_preflight as validate_exact,
    )

    accepted = validate_exact(
        artifact,
        expected_sha256,
        student_weight=student_weight,
        dino_weight=dino_weight,
        device="cuda",
    )
    payload = accepted.get("payload")
    if not isinstance(payload, Mapping):
        raise B16ContractError("Accepted preflight validator returned no payload")
    validate_qdq_preflight_payload(payload)
    return dict(accepted)


def compute_assignment_hashes(
    relative_paths: Sequence[str], folds: np.ndarray, groups: np.ndarray
) -> dict[str, str]:
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if len(relative_paths) != folds.size or folds.shape != groups.shape:
        raise ValueError("Assignment arrays are not aligned")
    path_fold = "\n".join(
        f"{relative_paths[index]}\t{int(folds[index])}"
        for index in range(len(relative_paths))
    ) + "\n"
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
    if not relative_paths or labels.shape != groups.shape or labels.shape != folds.shape or labels.size != len(relative_paths):
        raise ValueError("B16 assignment is empty or misaligned")
    if bool((labels < 0).any()) or bool((labels >= CLASSES).any()) or bool((groups < 0).any()):
        raise ValueError("B16 assignment label/group domain is invalid")
    if set(np.unique(folds).tolist()) != set(range(FOLDS)):
        raise ValueError("B16 assignment must contain exactly folds 0..4")
    fold_rows: list[dict[str, object]] = []
    for fold in range(FOLDS):
        held = folds == fold
        fit = ~held
        overlap = set(groups[held].tolist()) & set(groups[fit].tolist())
        counts = np.bincount(labels[held], minlength=CLASSES).astype(int)
        if overlap or bool((counts <= 0).any()):
            raise ValueError(f"B16 fold {fold} is not component-disjoint/class-complete")
        fold_rows.append(
            {
                "fold": fold,
                "fit_samples": int(fit.sum()),
                "held_samples": int(held.sum()),
                "held_class_counts": counts.tolist(),
                "group_overlap": 0,
            }
        )
    hashes = compute_assignment_hashes(relative_paths, folds, groups)
    if enforce_locked:
        if labels.size != EXPECTED_TRAIN_ROWS:
            raise B16ContractError("B16 assignment row count changed")
        counts = tuple(np.bincount(labels, minlength=CLASSES).astype(int).tolist())
        if counts != EXPECTED_CLASS_COUNTS:
            raise B16ContractError("B16 assignment class counts changed")
        expected = {
            "fold_vector_sha256": EXPECTED_FOLD_VECTOR_SHA256,
            "path_fold_sha256": EXPECTED_PATH_FOLD_SHA256,
            "group_vector_sha256": EXPECTED_GROUP_VECTOR_SHA256,
        }
        if hashes != expected:
            raise B16ContractError(f"B16 assignment vector hashes changed: {hashes}")
    return {"hashes": hashes, "fold_rows": fold_rows}


def read_locked_assignment_csv(path: Path) -> dict[str, object]:
    path = Path(path).expanduser().resolve(strict=True)
    if sha256_file(path) != EXPECTED_ASSIGNMENT_CSV_SHA256:
        raise B16ContractError("B16 assignment CSV hash changed")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != ["relative_path", "label", "union_group", "fold"]:
            raise B16ContractError("B16 assignment CSV columns changed")
        rows = list(reader)
    relative_paths = [str(row["relative_path"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([int(row["union_group"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    contract = validate_assignment_arrays(relative_paths, labels, groups, folds)
    return {
        "csv": str(path),
        "csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
        "relative_paths": relative_paths,
        "labels": labels,
        "groups": groups,
        "folds": folds,
        **contract,
    }


def _read_locked_data_root(data_yaml: Path) -> Path:
    resolved = Path(data_yaml).expanduser().resolve(strict=True)
    if resolved != EXPECTED_DATA_YAML.resolve() or sha256_file(resolved) != EXPECTED_DATA_SHA256:
        raise B16ContractError("Canonical class_f YAML path/hash changed")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    data_format = str(raw.get("format", raw.get("data_format", ""))).strip().casefold()
    if data_format != "classification_folder":
        raise B16ContractError("B16 requires classification_folder data")
    names = raw.get("names")
    if isinstance(names, Mapping):
        class_names = tuple(str(names[key]) for key in sorted(names, key=lambda value: int(value)))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        class_names = tuple(str(value) for value in names)
    else:
        class_names = ()
    if class_names != EXPECTED_CLASS_NAMES:
        raise B16ContractError("Canonical class names/order changed")
    configured = Path(str(raw.get("path", ".")))
    root = configured if configured.is_absolute() else resolved.parent / configured
    train_value = Path(str(raw.get("train", "train")))
    train_root = train_value if train_value.is_absolute() else root / train_value
    root = root.expanduser().resolve()
    if train_root.expanduser().resolve() != (root / "train").resolve():
        raise B16ContractError("Canonical B16 TRAIN root changed")
    return root


def train_content_contract(
    relative_paths: Sequence[str], absolute_paths: Sequence[Path]
) -> dict[str, object]:
    if not relative_paths or len(relative_paths) != len(absolute_paths):
        raise ValueError("TRAIN content paths are empty or misaligned")
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, absolute in zip(relative_paths, absolute_paths):
        path = Path(absolute).resolve(strict=True)
        before = path.stat()
        file_sha = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise B16ContractError(f"TRAIN image changed while hashing: {path}")
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
        raise B16ContractError("Canonical TRAIN content hash changed")
    return payload


def sample_random_resized_crop(
    height: int,
    width: int,
    generator: torch.Generator,
) -> CropParameters:
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Image geometry must be positive")
    area = float(height * width)
    log_ratio = torch.log(torch.tensor((0.90, 1.10), dtype=torch.float64))
    for _ in range(10):
        target = area * float(torch.empty((), dtype=torch.float64).uniform_(0.80, 1.00, generator=generator))
        aspect = float(torch.exp(torch.empty((), dtype=torch.float64).uniform_(float(log_ratio[0]), float(log_ratio[1]), generator=generator)))
        crop_width = int(round(math.sqrt(target * aspect)))
        crop_height = int(round(math.sqrt(target / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
            left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
            flip = bool(float(torch.rand((), generator=generator)) < 0.5)
            return CropParameters(top, left, crop_height, crop_width, flip)
    input_ratio = width / height
    if input_ratio < 0.90:
        crop_width, crop_height = width, int(round(width / 0.90))
    elif input_ratio > 1.10:
        crop_height, crop_width = height, int(round(height * 1.10))
    else:
        crop_height, crop_width = height, width
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    flip = bool(float(torch.rand((), generator=generator)) < 0.5)
    return CropParameters(top, left, crop_height, crop_width, flip)


def _normalize_image(image: Image.Image, size: int) -> Tensor:
    resized = TF.resize(
        image,
        [int(size), int(size)],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    tensor = TF.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def build_shared_train_views(
    image: Image.Image, parameters: CropParameters
) -> tuple[Tensor, Tensor]:
    rgb = image.convert("RGB")
    crop = TF.crop(
        rgb,
        parameters.top,
        parameters.left,
        parameters.height,
        parameters.width,
    )
    if parameters.horizontal_flip:
        crop = TF.hflip(crop)
    return _normalize_image(crop, 224), _normalize_image(crop, 256)


def build_shared_batch_views(
    images: Sequence[Image.Image], generator: torch.Generator
) -> tuple[Tensor, Tensor, list[CropParameters]]:
    student: list[Tensor] = []
    teacher: list[Tensor] = []
    parameters: list[CropParameters] = []
    for image in images:
        crop = sample_random_resized_crop(image.height, image.width, generator)
        student_view, teacher_view = build_shared_train_views(image, crop)
        student.append(student_view)
        teacher.append(teacher_view)
        parameters.append(crop)
    return torch.stack(student), torch.stack(teacher), parameters


def build_eval_view(image: Image.Image) -> Tensor:
    rgb = image.convert("RGB")
    resized = TF.resize(
        rgb,
        235,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    cropped = TF.center_crop(resized, [224, 224])
    tensor = TF.pil_to_tensor(cropped).to(dtype=torch.float32).div_(255.0)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def teacher_relation_target(
    teacher: nn.Module,
    images: Tensor,
    ledger: TeacherCallLedger,
) -> Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
        raise B16ContractError("DINO input must be [B,3,256,256]")
    with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        tokens = teacher.forward_features(images.float())
    ledger.calls += 1
    ledger.rows += int(images.size(0))
    if not isinstance(tokens, Tensor) or tuple(tokens.shape[1:]) != (261, DINO_FEATURE_DIM):
        raise B16ContractError(f"DINO token geometry changed: {getattr(tokens, 'shape', None)}")
    patch_map = (
        tokens[:, DINO_PREFIX_TOKENS:]
        .reshape(images.size(0), 16, 16, DINO_FEATURE_DIM)
        .permute(0, 3, 1, 2)
        .float()
    )
    resized = F.interpolate(patch_map, size=(14, 14), mode="bilinear", align_corners=False)
    target = cosine_neighbor_relation_field(resized).detach()
    if tuple(target.shape) != (images.size(0), 364) or target.requires_grad:
        raise B16ContractError("DINO relation target geometry/gradient changed")
    return target


def relation_loss_from_target(student_s2: Tensor, target: Tensor) -> Tensor:
    relation = cosine_neighbor_relation_field(student_s2)
    if relation.shape != target.shape or target.requires_grad:
        raise B16ContractError("Shared relation target is not aligned/detached")
    return F.smooth_l1_loss(
        relation,
        target,
        beta=RELATION_BETA,
        reduction="mean",
    )


def _rng_snapshot(device: torch.device) -> tuple[Tensor, Tensor | None]:
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu, cuda


def _rng_snapshot_equal(
    left: tuple[Tensor, Tensor | None], right: tuple[Tensor, Tensor | None]
) -> bool:
    return bool(
        torch.equal(left[0], right[0])
        and (
            (left[1] is None and right[1] is None)
            or (
                left[1] is not None
                and right[1] is not None
                and torch.equal(left[1], right[1])
            )
        )
    )


def _preserve_rng(function: Callable[[], Any]) -> Any:
    cpu = torch.random.get_rng_state().clone()
    cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    try:
        return function()
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def reset_locked_five_class_heads(
    model: nn.Module, seed: int = SEED
) -> dict[str, object]:
    # Reuse the preflight's exact sequential Linear-create/trunc-normal routine.
    # Calling reset_classifier first would consume a different RNG sequence and
    # silently give formal training different heads from accepted preflight.
    from trkh.tools.audit_swiftformer_surface_b16_preflight import (
        reset_swiftformer_five_class_heads,
    )

    result = reset_swiftformer_five_class_heads(model, seed=int(seed))
    model.distilled_training = False
    return dict(result)


def build_b16_arm_models(base_five_class: nn.Module) -> dict[str, nn.Module]:
    def build() -> dict[str, nn.Module]:
        # Match the accepted preflight's exact CPU branch-template generator.
        torch.random.default_generator.manual_seed(SEED)
        stock = copy.deepcopy(base_five_class)
        candidate = SwiftFormerSurfaceB16(
            copy.deepcopy(base_five_class), SURFACE_SPATIAL_B16_MODE
        )
        control = copy.deepcopy(candidate)
        control.mode = SURFACE_MEAN_CONTROL_B16_MODE
        return {
            "stock": stock,
            "globalized_control": control,
            "swiftsurface_candidate": candidate,
        }

    arms = _preserve_rng(build)
    if isinstance(arms["stock"], SwiftFormerSurfaceB16):
        raise B16ContractError("B16 stock must be the bare timm SwiftFormer")
    counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in arms.items()}
    if counts != ARM_PARAMETERS:
        raise B16ContractError(f"B16 arm parameter counts changed: {counts}")
    left = arms["globalized_control"].state_dict()
    right = arms["swiftsurface_candidate"].state_dict()
    if left.keys() != right.keys() or any(not torch.equal(left[key], right[key]) for key in left):
        raise B16ContractError("B16 candidate/control initial states differ")
    for arm, model in arms.items():
        for module in model.modules():
            probability = getattr(module, "p", None)
            drop_probability = getattr(module, "drop_prob", None)
            if isinstance(module, nn.Dropout) and float(probability or 0.0) != 0.0:
                raise B16ContractError(f"B16 {arm} contains active dropout")
            if drop_probability is not None and float(drop_probability) != 0.0:
                raise B16ContractError(f"B16 {arm} contains active stochastic depth")
    return arms


def _is_task_parameter(name: str) -> bool:
    lower = name.casefold()
    return lower.startswith(("head.", "head_dist.", "surface_branch.", "backbone.head.", "backbone.head_dist."))


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
        ("backbone", True): [],
        ("backbone", False): [],
        ("task", True): [],
        ("task", False): [],
    }
    seen: set[int] = set()
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise B16ContractError(f"Duplicate trainable parameter: {name}")
        seen.add(id(parameter))
        task = _is_task_parameter(name)
        no_decay = name.endswith(".bias") or name in norm_names or "layer_scale" in name.casefold()
        buckets[("task" if task else "backbone", not no_decay)].append((name, parameter))
    if seen != expected:
        raise B16ContractError("AdamW groups do not cover every trainable parameter exactly once")
    result: list[dict[str, object]] = []
    for role, decay in (("backbone", True), ("backbone", False), ("task", True), ("task", False)):
        values = buckets[(role, decay)]
        if not values:
            continue
        names = [name for name, _ in values]
        peak_lr = TASK_LR if role == "task" else BACKBONE_LR
        result.append(
            {
                "params": [parameter for _, parameter in values],
                "lr": peak_lr,
                "peak_lr": peak_lr,
                "weight_decay": WEIGHT_DECAY if decay else 0.0,
                "name": f"{role}_{'decay' if decay else 'no_decay'}",
                "parameter_names_sha256": hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest(),
                "parameter_count": sum(parameter.numel() for _, parameter in values),
            }
        )
    return result


def audit_and_clip_gradients(
    model: nn.Module, max_norm: float = GRAD_CLIP
) -> dict[str, object]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in named if parameter.grad is None]
    if missing:
        raise B16ContractError(f"Trainable B16 parameters lack gradients: {missing}")
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named],
            float(max_norm),
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise B16ContractError("B16 gradient tensor is non-finite") from exc
    if not bool(torch.isfinite(norm)):
        raise B16ContractError("B16 global gradient norm is non-finite")
    return {
        "parameter_tensors": len(named),
        "parameter_elements": sum(parameter.numel() for _, parameter in named),
        "preclip_global_norm": float(norm),
    }


class LockedOptimizerStepper:
    def __init__(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> None:
        self._optimizer = optimizer
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_steps = EPOCHS * self.steps_per_epoch
        self.completed_steps = 0
        self._applied_checkpoints: dict[str, dict[str, float]] = {}
        if self.steps_per_epoch < 2:
            raise ValueError(
                "steps_per_epoch must be at least two so first=0 and epoch-1-last=peak"
            )
        if not self._optimizer.param_groups:
            raise ValueError("optimizer must contain parameter groups")
        names: set[str] = set()
        for index, group in enumerate(self._optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("optimizer group names must be unique non-empty strings")
            names.add(name)
            group["name"] = name
            peak = group.get("peak_lr")
            if not _finite_number(peak) or float(peak) <= MIN_LR:
                raise ValueError("optimizer peak_lr must be finite and greater than MIN_LR")
            group["lr"] = 0.0

    def zero_grad(self) -> None:
        self._optimizer.zero_grad(set_to_none=True)

    def _lr_for_update(self, peak: float, update_index: int) -> float:
        if update_index < 0 or update_index >= self.total_steps:
            raise B16ContractError("B16 LR requested outside the locked horizon")
        if update_index < self.steps_per_epoch:
            return float(peak * update_index / (self.steps_per_epoch - 1))
        decay_steps = self.total_steps - self.steps_per_epoch
        progress = (update_index - self.steps_per_epoch + 1) / decay_steps
        return float(
            MIN_LR
            + (peak - MIN_LR)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    def step(self) -> dict[str, float]:
        if self.completed_steps >= self.total_steps:
            raise B16ContractError("B16 scheduler stepped beyond the locked horizon")
        update_index = self.completed_steps
        applied: dict[str, float] = {}
        for group in self._optimizer.param_groups:
            peak = float(group["peak_lr"])
            lr = self._lr_for_update(peak, update_index)
            group["lr"] = lr
            applied[str(group["name"])] = lr
        # The LR snapshot and optimizer update are deliberately indivisible here:
        # callers cannot advance/report a scheduler after an already-applied update.
        self._optimizer.step()
        self.completed_steps += 1
        if self.completed_steps == 1:
            self._applied_checkpoints["first"] = dict(applied)
        if self.completed_steps == self.steps_per_epoch:
            self._applied_checkpoints["warmup_last"] = dict(applied)
        if self.completed_steps == self.total_steps:
            self._applied_checkpoints["final"] = dict(applied)
        return applied

    def evidence(self, *, require_complete: bool) -> dict[str, object]:
        expected_keys = {"first", "warmup_last", "final"}
        if require_complete and (
            self.completed_steps != self.total_steps
            or set(self._applied_checkpoints) != expected_keys
        ):
            raise B16ContractError("B16 optimizer-step LR horizon is incomplete")
        evidence: dict[str, object] = {
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "first_update": 1,
            "warmup_last_update": self.steps_per_epoch,
            "final_update": self.total_steps,
            "first_applied_lrs": dict(self._applied_checkpoints.get("first", {})),
            "warmup_last_applied_lrs": dict(
                self._applied_checkpoints.get("warmup_last", {})
            ),
            "final_applied_lrs": dict(self._applied_checkpoints.get("final", {})),
        }
        if require_complete:
            peaks = {
                str(group["name"]): float(group["peak_lr"])
                for group in self._optimizer.param_groups
            }
            if (
                any(value != 0.0 for value in evidence["first_applied_lrs"].values())
                or evidence["warmup_last_applied_lrs"] != peaks
                or any(
                    not math.isclose(value, MIN_LR, rel_tol=0.0, abs_tol=1e-15)
                    for value in evidence["final_applied_lrs"].values()
                )
            ):
                raise B16ContractError("B16 applied LR checkpoints changed")
        return evidence


def sqrt_inverse_class_weights(labels: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, Tensor]:
    labels = np.asarray(labels, dtype=np.int64)
    fit = np.asarray(fit, dtype=bool)
    counts = np.bincount(labels[fit], minlength=CLASSES).astype(np.int64)
    if counts.shape != (CLASSES,) or bool((counts <= 0).any()):
        raise B16ContractError("Every B16 fit fold must contain all classes")
    weights = counts.astype(np.float64) ** -0.5
    weights /= weights.mean()
    return counts, torch.tensor(weights, dtype=torch.float32)


def expected_state_ids() -> set[str]:
    return {f"fold_{fold}:{arm}" for fold in range(FOLDS) for arm in ARMS}


def _validated_preflight_artifact(
    artifact: object, expected_sha256: object
) -> tuple[Path, str, dict[str, object]]:
    if not isinstance(artifact, str) or not artifact or not _hex64(expected_sha256):
        raise B16ContractError("Metric barrier preflight binding schema is invalid")
    candidate = Path(artifact)
    if not candidate.is_absolute():
        raise B16ContractError("Metric barrier preflight path is not absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise B16ContractError("Metric barrier preflight artifact is missing") from exc
    if not resolved.is_file() or artifact != str(resolved):
        raise B16ContractError(
            "Metric barrier preflight artifact is not a direct canonical path"
        )
    canonical_runs = _canonical_preflight_runs_root()
    if (
        resolved.name != "preflight.json"
        or resolved.parent.parent != canonical_runs
        or not resolved.parent.name.startswith(ACCEPTED_PREFLIGHT_OUTPUT_PREFIX)
    ):
        raise B16ContractError(
            "Metric barrier preflight artifact is outside the canonical runs root"
        )
    sidecar = resolved.with_name("preflight.sha256")
    if set(child.name for child in resolved.parent.iterdir()) != {
        "preflight.json",
        "preflight.sha256",
    }:
        raise B16ContractError(
            "Metric barrier preflight artifact directory schema changed"
        )
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise B16ContractError("Metric barrier preflight SHA-256 mismatch")
    if sidecar.read_bytes() != f"{digest}\n".encode("ascii"):
        raise B16ContractError("Metric barrier preflight SHA-256 sidecar mismatch")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B16ContractError("Metric barrier preflight JSON is invalid") from exc
    if not isinstance(decoded, Mapping) or raw != _json_bytes(decoded):
        raise B16ContractError("Metric barrier preflight JSON is not canonical")
    payload = dict(decoded)
    validate_qdq_preflight_payload(payload)
    return resolved, digest, payload


def _validate_state_records(
    root: Path,
    records: Sequence[FinalStateRecord],
    strict_reload: Callable[[FinalStateRecord], None] | None = None,
) -> list[dict[str, object]]:
    if len(records) != FOLDS * len(ARMS):
        raise B16ContractError("Metric barrier requires exactly 15 final states")
    if any(not isinstance(record, FinalStateRecord) for record in records):
        raise B16ContractError("Metric barrier state record type changed")
    if {record.state_id for record in records} != expected_state_ids():
        raise B16ContractError("Metric barrier state IDs are incomplete or duplicated")
    payload: list[dict[str, object]] = []
    states_root = (Path(root) / "states").resolve()
    for record in sorted(records, key=lambda value: value.state_id):
        if (
            not _strict_int(record.fold)
            or record.fold not in range(FOLDS)
            or not isinstance(record.arm, str)
            or record.arm not in ARMS
            or not _strict_int(record.bytes)
            or record.bytes <= 0
            or not _strict_int(record.parameter_count)
        ):
            raise B16ContractError(f"Invalid B16 state identity: {record.state_id}")
        if record.state_id != f"fold_{record.fold}:{record.arm}":
            raise B16ContractError(f"B16 state ID/fold/arm mismatch: {record.state_id}")
        if record.parameter_count != ARM_PARAMETERS[record.arm]:
            raise B16ContractError(f"B16 state parameter count changed: {record.state_id}")
        if record.mode != _state_mode(record.arm):
            raise B16ContractError(f"B16 state mode changed: {record.state_id}")
        path = Path(record.path).resolve(strict=True)
        expected_path = (
            states_root / f"fold_{record.fold}_{record.arm}.safetensors"
        ).resolve()
        if record.path != str(path) or path != expected_path:
            raise B16ContractError(
                f"B16 state is not the exact current run-root filename: {record.state_id}"
            )
        if path.suffix != ".safetensors" or path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
            raise B16ContractError(f"B16 state artifact changed: {record.state_id}")
        if not (_hex64(record.exposure_sha256) and _hex64(record.loss_curve_sha256)):
            raise B16ContractError(f"B16 state evidence hashes invalid: {record.state_id}")
        if strict_reload is not None:
            strict_reload(record)
        payload.append(asdict(record))
    partials = [path for path in states_root.rglob("*") if path.is_file() and (path.suffix in {".tmp", ".partial"} or ".tmp" in path.name)]
    if partials:
        raise B16ContractError(f"Partial state artifacts remain: {partials}")
    return payload


def seal_metric_barrier(
    output_dir: Path,
    records: Sequence[FinalStateRecord],
    *,
    accepted_preflight: Mapping[str, object],
    counters: Mapping[str, int],
    strict_reload: Callable[[FinalStateRecord], None] | None = None,
) -> MetricBarrierToken:
    output = Path(output_dir).resolve()
    if (
        set(counters) != set(METRIC_BARRIER_COUNTERS)
        or any(
            not _strict_int(counters[key])
            or counters[key] != METRIC_BARRIER_COUNTERS[key]
            for key in METRIC_BARRIER_COUNTERS
        )
        or _METRIC_BACKEND_LOADED
    ):
        raise B16ContractError("Held inference/logit/metric work occurred before the 15-state barrier")
    if set(accepted_preflight) != {"artifact", "sha256", "payload"}:
        raise B16ContractError("Metric barrier accepted-preflight input schema changed")
    raw_preflight_sha = accepted_preflight.get("sha256")
    raw_preflight_artifact = accepted_preflight.get("artifact")
    preflight_payload = accepted_preflight.get("payload")
    if not isinstance(preflight_payload, Mapping):
        raise B16ContractError("Metric barrier lacks an accepted preflight identity")
    preflight_path, preflight_sha, artifact_payload = _validated_preflight_artifact(
        raw_preflight_artifact, raw_preflight_sha
    )
    if dict(preflight_payload) != artifact_payload:
        raise B16ContractError(
            "Metric barrier accepted-preflight payload differs from its artifact"
        )
    state_payload = _validate_state_records(output, records, strict_reload)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": METRIC_BARRIER_PHASE,
        "created_at_unix": time.time(),
        "accepted_preflight": {
            "artifact": str(preflight_path),
            "sha256": preflight_sha,
            "qdq_passed": True,
        },
        "expected_state_ids": sorted(expected_state_ids()),
        "states": state_payload,
        "counters": dict(counters),
        "checks": dict(METRIC_BARRIER_CHECKS),
        "passed": True,
    }
    path = output / "metric_barrier.json"
    digest = atomic_json(path, payload)
    return validate_metric_barrier(path, digest)


def validate_metric_barrier(path: Path, expected_sha256: str) -> MetricBarrierToken:
    resolved = Path(path).resolve(strict=True)
    digest = sha256_file(resolved)
    if (
        not isinstance(expected_sha256, str)
        or not _hex64(expected_sha256.casefold())
        or digest != expected_sha256.casefold()
        or not _hex64(digest)
    ):
        raise B16ContractError("Metric barrier SHA-256 mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise B16ContractError("Metric barrier must be a mapping")
    expected_top = {
        "schema_version", "protocol_id", "phase", "created_at_unix", "accepted_preflight",
        "expected_state_ids", "states", "counters", "checks", "passed",
    }
    if (
        set(payload) != expected_top
        or not _strict_int(payload.get("schema_version"))
        or payload.get("schema_version") != 1
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("phase") != METRIC_BARRIER_PHASE
        or not _finite_number(payload.get("created_at_unix"))
        or payload.get("passed") is not True
    ):
        raise B16ContractError("Metric barrier schema/identity invalid")
    accepted_preflight = payload.get("accepted_preflight")
    if (
        not isinstance(accepted_preflight, Mapping)
        or set(accepted_preflight) != {"artifact", "sha256", "qdq_passed"}
        or accepted_preflight.get("qdq_passed") is not True
    ):
        raise B16ContractError("Metric barrier accepted-preflight schema changed")
    _validated_preflight_artifact(
        accepted_preflight.get("artifact"), accepted_preflight.get("sha256")
    )
    if payload.get("expected_state_ids") != sorted(expected_state_ids()):
        raise B16ContractError("Metric barrier expected state set changed")
    states = payload.get("states")
    if not isinstance(states, list) or len(states) != 15 or {row.get("state_id") for row in states if isinstance(row, Mapping)} != expected_state_ids():
        raise B16ContractError("Metric barrier does not contain exact 15 states")
    root = resolved.parent
    for row in states:
        if not isinstance(row, Mapping):
            raise B16ContractError("Metric barrier state row invalid")
        if set(row) != set(FinalStateRecord.__dataclass_fields__):
            raise B16ContractError("Metric barrier state row schema changed")
        if not (
            _strict_int(row.get("fold"))
            and isinstance(row.get("arm"), str)
            and isinstance(row.get("state_id"), str)
            and isinstance(row.get("path"), str)
            and _hex64(row.get("sha256"))
            and _strict_int(row.get("bytes"))
            and int(row["bytes"]) > 0
            and _strict_int(row.get("parameter_count"))
            and isinstance(row.get("mode"), str)
            and _hex64(row.get("exposure_sha256"))
            and _hex64(row.get("loss_curve_sha256"))
        ):
            raise B16ContractError("Metric barrier state field types changed")
        fold = row["fold"]
        arm = row["arm"]
        if (
            fold not in range(FOLDS)
            or arm not in ARMS
            or row["state_id"] != f"fold_{fold}:{arm}"
            or int(row["parameter_count"]) != ARM_PARAMETERS[arm]
            or row["mode"] != _state_mode(arm)
        ):
            raise B16ContractError("Metric barrier state identity changed")
        state_path = Path(str(row.get("path", ""))).resolve(strict=True)
        expected_state_path = (
            root / "states" / f"fold_{fold}_{arm}.safetensors"
        ).resolve()
        if row["path"] != str(state_path) or state_path != expected_state_path:
            raise B16ContractError(
                "Metric barrier state filename/current run root changed"
            )
        if state_path.stat().st_size != int(row.get("bytes", -1)) or sha256_file(state_path) != row.get("sha256"):
            raise B16ContractError(f"Metric barrier state changed: {row.get('state_id')}")
    observed_counters = payload.get("counters")
    if (
        not isinstance(observed_counters, Mapping)
        or set(observed_counters) != set(METRIC_BARRIER_COUNTERS)
        or any(
            not _strict_int(observed_counters[key])
            or observed_counters[key] != METRIC_BARRIER_COUNTERS[key]
            for key in METRIC_BARRIER_COUNTERS
        )
    ):
        raise B16ContractError("Metric barrier premetric counters changed")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or dict(checks) != METRIC_BARRIER_CHECKS:
        raise B16ContractError("Metric barrier check schema/values changed")
    nonce = uuid.uuid4().hex
    token_key = f"{digest}:{nonce}"
    _VALID_BARRIER_TOKENS.add(token_key)
    return MetricBarrierToken(str(resolved), digest, nonce)


def _assert_metric_token(token: MetricBarrierToken) -> None:
    key = f"{token.sha256}:{token.nonce}"
    if key not in _VALID_BARRIER_TOKENS:
        raise B16ContractError("Metric code requires a validated 15-state barrier token")
    if sha256_file(Path(token.path)) != token.sha256:
        _VALID_BARRIER_TOKENS.discard(key)
        raise B16ContractError("Metric barrier changed after validation")
    payload = json.loads(Path(token.path).read_text(encoding="utf-8"))
    accepted = payload["accepted_preflight"]
    _validated_preflight_artifact(accepted["artifact"], accepted["sha256"])
    if key not in _VERIFIED_BARRIER_STATE_TOKENS:
        root = Path(token.path).resolve().parent
        for row in payload["states"]:
            try:
                state_path = Path(row["path"]).resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                _VALID_BARRIER_TOKENS.discard(key)
                raise B16ContractError(
                    "Metric barrier state changed after validation"
                ) from exc
            expected_state_path = (
                root
                / "states"
                / f"fold_{row['fold']}_{row['arm']}.safetensors"
            ).resolve()
            if (
                row["path"] != str(state_path)
                or state_path != expected_state_path
                or state_path.stat().st_size != int(row["bytes"])
                or sha256_file(state_path) != row["sha256"]
            ):
                _VALID_BARRIER_TOKENS.discard(key)
                raise B16ContractError(
                    "Metric barrier state changed after validation"
                )
        _VERIFIED_BARRIER_STATE_TOKENS.add(key)


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


def metric_backend_loaded() -> bool:
    return _METRIC_BACKEND_LOADED


def classification_summary(
    labels: np.ndarray,
    logits: np.ndarray,
    token: MetricBarrierToken,
    folds: np.ndarray | None = None,
) -> dict[str, object]:
    backend = _metric_backend(token)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if labels.ndim != 1 or logits.shape != (labels.size, CLASSES) or not np.isfinite(logits).all():
        raise ValueError("B16 labels/logits are misaligned or non-finite")
    predictions = logits.argmax(axis=1).astype(np.int64)
    matrix = backend["confusion_matrix"](labels, predictions, labels=np.arange(CLASSES))
    precision, recall, f1, support = backend["precision_recall_fscore_support"](
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
    payload: dict[str, object] = {
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
        "restricted_0_2_4_to_1": int(sum(matrix[rival, 1] for rival in RIVALS)),
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
            raise ValueError("B16 fold vector is misaligned")
        payload["folds"] = [
            {"fold": fold, **classification_summary(labels[fold_values == fold], logits[fold_values == fold], token)}
            for fold in range(FOLDS)
        ]
    return payload


def _fast_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    matrix = np.bincount(
        labels.astype(np.int64) * CLASSES + predictions.astype(np.int64),
        minlength=CLASSES * CLASSES,
    ).reshape(CLASSES, CLASSES)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1.mean()), float(f1[CLASS1])


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
    required = {"stock", "globalized_control", "swiftsurface_candidate", "candidate_r0"}
    if set(logits) != required or labels.shape != folds.shape or labels.shape != groups.shape:
        raise ValueError("B16 bootstrap inputs are incomplete or misaligned")
    scores = {name: np.asarray(value, dtype=np.float64) for name, value in logits.items()}
    if any(value.shape != (labels.size, CLASSES) or not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("B16 bootstrap logits are invalid")
    members: dict[int, np.ndarray] = {}
    fold_groups: dict[int, list[int]] = {fold: [] for fold in range(FOLDS)}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        observed = np.unique(folds[positions])
        if observed.size != 1:
            raise B16ContractError("Bootstrap union component crosses folds")
        members[int(group)] = positions
        fold_groups[int(observed[0])].append(int(group))
    comparisons = {
        "candidate_vs_control": ("swiftsurface_candidate", "globalized_control"),
        "candidate_vs_stock": ("swiftsurface_candidate", "stock"),
        "candidate_vs_ablation": ("swiftsurface_candidate", "candidate_r0"),
    }
    metric_names = (
        "macro_f1_delta", "class1_f1_delta", "pair0_auroc_delta",
        "pair2_auroc_delta", "pair4_auroc_delta", "mean_pair_auroc_delta",
    )
    values = {
        comparison: {name: np.empty(int(replicates), dtype=np.float64) for name in metric_names}
        for comparison in comparisons
    }
    predictions = {name: value.argmax(axis=1).astype(np.int64) for name, value in scores.items()}
    rng = np.random.default_rng(int(seed))
    draw_hash = hashlib.sha256()
    for replicate in range(int(replicates)):
        chunks: list[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            if not available:
                raise B16ContractError(f"Bootstrap fold {fold} has no component")
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draw_hash.update(np.asarray((replicate, fold), dtype="<i8").tobytes())
            draw_hash.update(draw.astype("<i8").tobytes())
            chunks.extend(members[available[int(position)]] for position in draw)
        selected = np.concatenate(chunks)
        selected_labels = labels[selected]
        arm_metrics: dict[str, dict[str, float]] = {}
        for arm in required:
            macro, class1 = _fast_f1(selected_labels, predictions[arm][selected])
            pair_aucs: dict[int, float] = {}
            for rival in RIVALS:
                mask = (selected_labels == CLASS1) | (selected_labels == rival)
                binary = (selected_labels[mask] == CLASS1).astype(np.int64)
                margin = scores[arm][selected][mask, CLASS1] - scores[arm][selected][mask, rival]
                pair_aucs[rival] = float(backend["roc_auc_score"](binary, margin))
            arm_metrics[arm] = {
                "macro_f1": macro,
                "class1_f1": class1,
                **{f"pair{rival}_auroc": pair_aucs[rival] for rival in RIVALS},
                "mean_pair_auroc": float(np.mean(list(pair_aucs.values()))),
            }
        for comparison, (left, right) in comparisons.items():
            for metric_name in metric_names:
                base_name = metric_name.removesuffix("_delta")
                values[comparison][metric_name][replicate] = arm_metrics[left][base_name] - arm_metrics[right][base_name]
    digest = draw_hash.hexdigest()
    if enforce_locked_draw_hash and (
        int(replicates) != BOOTSTRAP_REPLICATES
        or int(seed) != BOOTSTRAP_SEED
        or digest != EXPECTED_BOOTSTRAP_DRAW_SHA256
    ):
        raise B16ContractError(f"B16 bootstrap draw contract changed: {digest}")
    for comparison in values.values():
        if any(not np.isfinite(sample).all() for sample in comparison.values()):
            raise B16ContractError("B16 bootstrap produced a non-finite replicate")
    intervals = {
        comparison: {
            metric: {
                "lower": float(np.quantile(sample, 0.025, method="linear")),
                "upper": float(np.quantile(sample, 0.975, method="linear")),
            }
            for metric, sample in metrics.items()
        }
        for comparison, metrics in values.items()
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


def assess_b16_gate(
    summaries: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, object],
    token: MetricBarrierToken,
    *,
    integrity_complete: bool,
) -> dict[str, object]:
    _assert_metric_token(token)
    required = {"stock", "globalized_control", "swiftsurface_candidate", "candidate_r0"}
    if set(summaries) != required:
        raise ValueError("B16 gate requires all four OOF summaries")
    stock = summaries["stock"]
    control = summaries["globalized_control"]
    candidate = summaries["swiftsurface_candidate"]
    ablation = summaries["candidate_r0"]
    intervals = bootstrap["intervals"]

    def delta(metric: str, left: Mapping[str, object], right: Mapping[str, object]) -> float:
        return float(left[metric]) - float(right[metric])

    c1_control = delta("class1_f1", candidate, control)
    c1_stock = delta("class1_f1", candidate, stock)
    c1_ablation = delta("class1_f1", candidate, ablation)
    macro_control = delta("macro_f1", candidate, control)
    macro_stock = delta("macro_f1", candidate, stock)
    pair2 = float(candidate["pairs"]["2"]["auroc"]) - float(control["pairs"]["2"]["auroc"])
    mean_pair = delta("mean_pair_auroc", candidate, control)
    control_restricted = int(control["restricted_0_2_4_to_1"])
    control_2_to_1 = int(control["transition_2_to_1"])
    restricted_denominator_valid = control_restricted > 0
    transition_2_to_1_denominator_valid = control_2_to_1 > 0
    reduction_restricted = (
        (control_restricted - int(candidate["restricted_0_2_4_to_1"])) / control_restricted
        if restricted_denominator_valid
        else None
    )
    reduction_2_to_1 = (
        (control_2_to_1 - int(candidate["transition_2_to_1"])) / control_2_to_1
        if transition_2_to_1_denominator_valid
        else None
    )
    candidate_folds = {int(row["fold"]): row for row in candidate["folds"]}
    control_folds = {int(row["fold"]): row for row in control["folds"]}
    stock_folds = {int(row["fold"]): row for row in stock["folds"]}
    control_wins = sum(float(candidate_folds[f]["class1_f1"]) > float(control_folds[f]["class1_f1"]) for f in range(FOLDS))
    stock_wins = sum(float(candidate_folds[f]["class1_f1"]) > float(stock_folds[f]["class1_f1"]) for f in range(FOLDS))
    checks = {
        "absolute_engineering_floors": float(candidate["class1_f1"]) >= 0.700 and float(candidate["macro_f1"]) >= 0.800,
        "class1_f1_vs_control": c1_control >= 0.010 and float(intervals["candidate_vs_control"]["class1_f1_delta"]["lower"]) > 0.0,
        "pair_and_mean_auroc_vs_control": pair2 >= 0.005 and mean_pair >= 0.003 and float(intervals["candidate_vs_control"]["pair2_auroc_delta"]["lower"]) > 0.0 and float(intervals["candidate_vs_control"]["mean_pair_auroc_delta"]["lower"]) > 0.0,
        "class1_f1_vs_stock": c1_stock >= 0.005 and float(intervals["candidate_vs_stock"]["class1_f1_delta"]["lower"]) > 0.0,
        "macro_lcb_noninferiority": float(intervals["candidate_vs_control"]["macro_f1_delta"]["lower"]) >= -0.002 and float(intervals["candidate_vs_stock"]["macro_f1_delta"]["lower"]) >= -0.002,
        "recall_and_transition_control": float(candidate["class1_recall"]) >= float(control["class1_recall"]) - 0.005 and float(candidate["class1_recall"]) >= float(stock["class1_recall"]) - 0.005 and restricted_denominator_valid and reduction_restricted is not None and reduction_restricted >= 0.10 and transition_2_to_1_denominator_valid and reduction_2_to_1 is not None and reduction_2_to_1 >= 0.10 and int(candidate["restricted_0_2_4_to_1"]) <= int(stock["restricted_0_2_4_to_1"]) and int(candidate["transition_2_to_1"]) <= int(stock["transition_2_to_1"]) and int(candidate["transition_1_to_2"]) <= int(control["transition_1_to_2"]) and int(candidate["transition_1_to_2"]) <= int(stock["transition_1_to_2"]),
        "fold_wins": control_wins >= 4 and stock_wins >= 3,
        "active_branch_ablation": c1_ablation >= 0.003 and float(intervals["candidate_vs_ablation"]["class1_f1_delta"]["lower"]) > 0.0 and int(candidate["transition_2_to_1"]) <= int(ablation["transition_2_to_1"]),
        "integrity_complete": bool(integrity_complete),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "validation_protocol_permission": passed,
        "validation_execution_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "denominator_valid": {
            "restricted_reduction_vs_control": restricted_denominator_valid,
            "2_to_1_reduction_vs_control": transition_2_to_1_denominator_valid,
        },
        "point_deltas": {
            "class1_f1_vs_control": c1_control,
            "class1_f1_vs_stock": c1_stock,
            "class1_f1_vs_ablation": c1_ablation,
            "macro_f1_vs_control": macro_control,
            "macro_f1_vs_stock": macro_stock,
            "pair2_auroc_vs_control": pair2,
            "mean_pair_auroc_vs_control": mean_pair,
            "restricted_reduction_vs_control": reduction_restricted,
            "2_to_1_reduction_vs_control": reduction_2_to_1,
            "fold_wins_vs_control": control_wins,
            "fold_wins_vs_stock": stock_wins,
        },
    }


class _RawLedgerDataset(Dataset[tuple[Image.Image, int, int]]):
    def __init__(self, paths: Sequence[Path], labels: np.ndarray) -> None:
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.open_count = 0
        if len(self.paths) != self.labels.size:
            raise ValueError("Raw B16 ledger paths/labels are misaligned")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, int]:
        with Image.open(self.paths[int(index)]) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB").copy()
        self.open_count += 1
        return image, int(self.labels[int(index)]), int(index)


def _raw_collate(batch: Sequence[tuple[Image.Image, int, int]]) -> tuple[list[Image.Image], Tensor, Tensor]:
    images, labels, indices = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), torch.tensor(indices, dtype=torch.long)


def _load_raw_indices(
    dataset: _RawLedgerDataset, indices: Sequence[int]
) -> tuple[list[Image.Image], Tensor, Tensor]:
    # The protocol locks workers=0. Direct collation avoids constructing 16,576
    # one-batch DataLoader objects while preserving exact sequential exposure.
    return _raw_collate([dataset[int(index)] for index in indices])


def _configure_determinism(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise B16ContractError("Formal B16 requires a BF16-capable CUDA GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _validate_asset(path: Path, expected_bytes: int, expected_sha: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.stat().st_size != expected_bytes or sha256_file(resolved) != expected_sha:
        raise B16ContractError(f"Locked pretrained asset changed: {resolved}")
    return resolved


def _load_templates(student_weight: Path, dino_weight: Path) -> tuple[nn.Module, nn.Module]:
    student_path = _validate_asset(student_weight, STUDENT_BYTES, STUDENT_SHA256)
    dino_path = _validate_asset(dino_weight, DINO_BYTES, DINO_SHA256)

    def construct() -> tuple[nn.Module, nn.Module]:
        torch.manual_seed(SEED)
        student = timm.create_model(
            STUDENT_TIMM_ID,
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        student.load_state_dict(load_file(str(student_path), device="cpu"), strict=True)
        reset_locked_five_class_heads(student, SEED)
        teacher = timm.create_model(
            DINO_TIMM_ID,
            pretrained=False,
            num_classes=0,
            img_size=256,
        )
        teacher_state = load_file(str(dino_path), device="cpu")
        forbidden = [key for key in teacher_state if any(term in key.casefold() for term in ("class_f", "b9", "classifier_5"))]
        if forbidden:
            raise B16ContractError(f"Forbidden class-fitted teacher state: {forbidden}")
        teacher.load_state_dict(teacher_state, strict=True)
        return student, teacher

    student, teacher = _preserve_rng(construct)
    teacher.eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in student.parameters()) != STOCK_PARAMETERS:
        raise B16ContractError("Five-class SwiftFormer parameter count changed")
    if int(getattr(teacher, "num_prefix_tokens", -1)) != 5 or tuple(teacher.patch_embed.grid_size) != (16, 16):
        raise B16ContractError("Raw DINO architecture geometry changed")
    return student, teacher


def _state_mode(arm: str) -> str:
    return {
        "stock": "bare_timm_stock",
        "globalized_control": SURFACE_MEAN_CONTROL_B16_MODE,
        "swiftsurface_candidate": SURFACE_SPATIAL_B16_MODE,
    }[arm]


def _build_model_for_state(base: nn.Module, arm: str) -> nn.Module:
    models = build_b16_arm_models(base)
    return models[arm]


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def _candidate_forward_with_residual_telemetry(
    model: SwiftFormerSurfaceB16, inputs: Tensor
) -> tuple[Tensor, Tensor]:
    captured: dict[str, Tensor] = {}

    def capture_base(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        captured["base"] = output.detach()

    def capture_residual(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        captured["residual"] = output.detach()

    base_handle = model.backbone.stages[3].downsample.register_forward_hook(capture_base)
    residual_handle = model.surface_branch.register_forward_hook(capture_residual)
    try:
        logits = model(inputs)
    finally:
        base_handle.remove()
        residual_handle.remove()
    if set(captured) != {"base", "residual"}:
        raise B16ContractError("B16 candidate telemetry hooks did not observe base/residual")
    base_norm = captured["base"].float().flatten(1).norm(dim=1)
    residual_norm = captured["residual"].float().flatten(1).norm(dim=1)
    ratio = residual_norm / base_norm.clamp_min(1.0e-12)
    if not bool(torch.isfinite(ratio).all()):
        raise B16ContractError("B16 residual/base telemetry is non-finite")
    return logits, ratio


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked TRAIN-only B16 SwiftSurface-XS three-arm OOF screen")
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
        raise B16ContractError("B16 output must be a fresh direct runs child with the locked prefix")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"Refuse to overwrite B16 output: {output}")
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
        "validation_used": False,
        "test_used": False,
    }
    path = output / "failure.json"
    digest = atomic_json(path, payload)
    _atomic_bytes(output / "failure.json.sha256", (digest + "\n").encode("ascii"))


def run_formal(args: argparse.Namespace) -> dict[str, object]:
    if args.confirm_protocol_id != PROTOCOL_ID:
        raise B16ContractError("Explicit B16 protocol confirmation is missing")
    git = _git_contract()
    if git["branch"] != EXPECTED_BRANCH or git["clean"] is not True or git["head_is_commit"] is not True:
        raise B16ContractError(f"Formal B16 requires a clean committed canonical branch: {git}")
    sources = _source_hashes()
    accepted = validate_accepted_preflight(
        args.preflight_artifact,
        args.preflight_sha256,
        args.student_weight,
        args.dino_weight,
    )
    output = _validated_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        data_root = _read_locked_data_root(args.data)
        assignment = read_locked_assignment_csv(args.assignment_csv)
        relative_paths = assignment["relative_paths"]
        labels = assignment["labels"]
        groups = assignment["groups"]
        folds = assignment["folds"]
        absolute_paths: list[Path] = []
        train_root = (data_root / "train").resolve()
        for relative in relative_paths:
            path = (data_root / relative).resolve()
            try:
                path.relative_to(train_root)
            except ValueError as exc:
                raise B16ContractError(f"Non-TRAIN path in B16 assignment: {path}") from exc
            absolute_paths.append(path)
        start_content = train_content_contract(relative_paths, absolute_paths)
        atomic_json(
            output / "contract.json",
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "git": git,
                "source_hashes": sources,
                "accepted_preflight": {"artifact": accepted["artifact"], "sha256": accepted["sha256"]},
                "data_yaml": str(Path(args.data).resolve()),
                "data_yaml_sha256": EXPECTED_DATA_SHA256,
                "assignment_csv": str(Path(args.assignment_csv).resolve()),
                "assignment_csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
                "train_content_start": start_content,
                "train": True,
                "validation": False,
                "test": False,
            },
        )
        atomic_json(output / "ledger" / "paths.json", {"paths": list(relative_paths)})
        atomic_npz(
            output / "ledger" / "labels_folds_groups.npz",
            labels=labels,
            folds=folds,
            groups=groups,
        )
        device = torch.device("cuda")
        _configure_determinism(device)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        base, teacher = _load_templates(args.student_weight, args.dino_weight)
        teacher.to(device).eval()
        dataset = _RawLedgerDataset(absolute_paths, labels)
        records: list[FinalStateRecord] = []
        peak_allocated = 0
        for fold in range(FOLDS):
            fold_seed = SEED + fold
            generator = torch.Generator().manual_seed(fold_seed)
            fit = folds != fold
            fit_indices = np.flatnonzero(fit).astype(np.int64)
            fit_counts, class_weights = sqrt_inverse_class_weights(labels, fit)
            class_weights = class_weights.to(device)
            arms = {name: model.to(device).train() for name, model in build_b16_arm_models(base).items()}
            steppers: dict[str, LockedOptimizerStepper] = {}
            batches_per_epoch = math.ceil(fit_indices.size / BATCH_SIZE)
            update_steps_per_epoch = math.ceil(batches_per_epoch / ACCUMULATION_STEPS)
            optimizer_contract: dict[str, object] = {}
            gradient_audit: dict[str, dict[str, int]] = {}
            for name, model in arms.items():
                groups_for_optimizer = build_discriminative_adamw_groups(model)
                optimizer_contract[name] = [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in groups_for_optimizer
                ]
                optimizer = torch.optim.AdamW(groups_for_optimizer, betas=(0.9, 0.999), eps=1e-8)
                steppers[name] = LockedOptimizerStepper(
                    optimizer, update_steps_per_epoch
                )
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                gradient_audit[name] = {
                    "updates": 0,
                    "checked_parameter_tensors": 0,
                    "checked_parameter_elements": 0,
                    "parameter_tensors_per_update": len(trainable),
                    "parameter_elements_per_update": sum(parameter.numel() for parameter in trainable),
                }
            teacher_ledger = TeacherCallLedger()
            exposure_indices: list[int] = []
            exposure_crop: list[tuple[int, int, int, int]] = []
            exposure_flip: list[bool] = []
            exposure_epoch: list[int] = []
            exposure_batch: list[int] = []
            exposure_window: list[int] = []
            exposure_divisor: list[int] = []
            loss_rows: list[tuple[int, int, float, float, float, float, float]] = []
            rng_forward_checks = {arm: 0 for arm in ARMS}
            torch.cuda.reset_peak_memory_stats(device)
            for epoch in range(EPOCHS):
                order = fit_indices[torch.randperm(fit_indices.size, generator=generator).numpy()]
                batches = [order[start : start + BATCH_SIZE].tolist() for start in range(0, order.size, BATCH_SIZE)]
                for window_start in range(0, len(batches), ACCUMULATION_STEPS):
                    window = batches[window_start : window_start + ACCUMULATION_STEPS]
                    divisor = len(window)
                    for stepper in steppers.values():
                        stepper.zero_grad()
                    for within, batch_indices in enumerate(window):
                        images, batch_labels, observed_indices = _load_raw_indices(
                            dataset, batch_indices
                        )
                        if observed_indices.tolist() != batch_indices or any(not fit[index] for index in batch_indices):
                            raise B16ContractError("B16 batch order/split exposure changed")
                        student_cpu, teacher_cpu, crops = build_shared_batch_views(images, generator)
                        for image in images:
                            image.close()
                        student_images = student_cpu.to(device, non_blocking=False)
                        teacher_images = teacher_cpu.to(device, non_blocking=False)
                        batch_labels = batch_labels.to(device)
                        target = teacher_relation_target(teacher, teacher_images, teacher_ledger)
                        batch_number = window_start + within
                        for index, crop in zip(batch_indices, crops):
                            exposure_indices.append(index)
                            exposure_crop.append((crop.top, crop.left, crop.height, crop.width))
                            exposure_flip.append(crop.horizontal_flip)
                            exposure_epoch.append(epoch)
                            exposure_batch.append(batch_number)
                            exposure_window.append(window_start // ACCUMULATION_STEPS)
                            exposure_divisor.append(divisor)
                        losses: dict[str, tuple[Tensor, Tensor]] = {}
                        for arm in ARMS:
                            model = arms[arm]
                            audit_rng = rng_forward_checks[arm] == 0
                            rng_before = _rng_snapshot(device) if audit_rng else None
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                if arm == "stock":
                                    logits = model(student_images)
                                    relation = torch.zeros((), device=device, dtype=torch.float32)
                                else:
                                    logits, s2 = model(student_images, return_s2=True)
                                    relation = relation_loss_from_target(s2, target)
                            if audit_rng:
                                if not _rng_snapshot_equal(rng_before, _rng_snapshot(device)):
                                    raise B16ContractError(
                                        f"B16 {arm} forward consumed RNG despite zero stochasticity"
                                    )
                                rng_forward_checks[arm] += 1
                            ce = F.cross_entropy(logits.float(), batch_labels, weight=class_weights, reduction="mean")
                            total = ce if arm == "stock" else ce + RELATION_WEIGHT * relation
                            if not bool(torch.isfinite(total)):
                                raise FloatingPointError(f"Non-finite B16 loss: fold={fold}, arm={arm}")
                            (total / divisor).backward()
                            losses[arm] = (ce.detach(), relation.detach())
                        loss_rows.append(
                            (
                                epoch,
                                batch_number,
                                float(losses["stock"][0]),
                                float(losses["globalized_control"][0]),
                                float(losses["globalized_control"][1]),
                                float(losses["swiftsurface_candidate"][0]),
                                float(losses["swiftsurface_candidate"][1]),
                            )
                        )
                    for arm in ARMS:
                        audited = audit_and_clip_gradients(arms[arm], GRAD_CLIP)
                        gradient_audit[arm]["updates"] += 1
                        gradient_audit[arm]["checked_parameter_tensors"] += int(audited["parameter_tensors"])
                        gradient_audit[arm]["checked_parameter_elements"] += int(audited["parameter_elements"])
                        steppers[arm].step()
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise TimeoutError("B16 formal wall-time budget exceeded during training")
            expected_calls = EPOCHS * batches_per_epoch
            if teacher_ledger.calls != expected_calls or teacher_ledger.rows != EPOCHS * int(fit.sum()):
                raise B16ContractError("DINO was not called exactly once per fit batch")
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise B16ContractError("Raw DINO teacher acquired gradients")
            for arm in ARMS:
                stepper = steppers[arm]
                scheduler_evidence = stepper.evidence(require_complete=True)
                if (
                    stepper.completed_steps != stepper.total_steps
                    or stepper.total_steps != EPOCHS * update_steps_per_epoch
                    or scheduler_evidence["completed_steps"]
                    != scheduler_evidence["total_steps"]
                ):
                    raise B16ContractError(f"B16 scheduler horizon/final LR changed: {arm}")
                audit = gradient_audit[arm]
                if (
                    audit["updates"] != stepper.total_steps
                    or audit["checked_parameter_tensors"]
                    != audit["updates"] * audit["parameter_tensors_per_update"]
                    or audit["checked_parameter_elements"]
                    != audit["updates"] * audit["parameter_elements_per_update"]
                ):
                    raise B16ContractError(f"B16 per-update gradient audit is incomplete: {arm}")
            exposure_sha = atomic_npz(
                output / "folds" / f"fold_{fold}" / "exposure.npz",
                indices=np.asarray(exposure_indices, dtype=np.int64),
                crop=np.asarray(exposure_crop, dtype=np.int32),
                flip=np.asarray(exposure_flip, dtype=np.bool_),
                epoch=np.asarray(exposure_epoch, dtype=np.int16),
                batch=np.asarray(exposure_batch, dtype=np.int32),
                accumulation_window=np.asarray(exposure_window, dtype=np.int32),
                accumulation_divisor=np.asarray(exposure_divisor, dtype=np.int8),
            )
            loss_array = np.asarray(loss_rows, dtype=np.float64)
            loss_sha = atomic_npz(output / "folds" / f"fold_{fold}" / "loss_curves.npz", rows=loss_array)
            atomic_json(
                output / "folds" / f"fold_{fold}" / "train_contract.json",
                {
                    "fold": fold,
                    "seed": fold_seed,
                    "fit_samples": int(fit.sum()),
                    "fit_class_counts": fit_counts.tolist(),
                    "class_weights": class_weights.cpu().tolist(),
                    "epochs": EPOCHS,
                    "workers": WORKERS,
                    "batches_per_epoch": batches_per_epoch,
                    "updates_per_epoch": update_steps_per_epoch,
                    "teacher_calls": teacher_ledger.calls,
                    "rng_forward_checks": rng_forward_checks,
                    "gradient_audit": gradient_audit,
                    "optimizer_groups": optimizer_contract,
                    "scheduler": {
                        arm: steppers[arm].evidence(require_complete=True)
                        for arm in ARMS
                    },
                    "held_inference": False,
                },
            )
            for arm in ARMS:
                model = arms[arm].cpu().eval()
                state_path = (output / "states" / f"fold_{fold}_{arm}.safetensors").resolve()
                state_sha = atomic_safetensors(state_path, model.state_dict())
                records.append(
                    FinalStateRecord(
                        state_id=f"fold_{fold}:{arm}",
                        fold=fold,
                        arm=arm,
                        path=str(state_path),
                        sha256=state_sha,
                        bytes=state_path.stat().st_size,
                        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                        mode=_state_mode(arm),
                        exposure_sha256=exposure_sha,
                        loss_curve_sha256=loss_sha,
                    )
                )
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            if peak_allocated > MAX_CUDA_ALLOCATED_BYTES:
                raise B16ContractError("B16 peak CUDA allocation exceeded 7 GiB")
            if _directory_bytes(output) > MAX_RETAINED_BYTES:
                raise B16ContractError("B16 retained artifact budget exceeded 1.5 GiB")
            del arms, steppers
            torch.cuda.empty_cache()
        teacher.cpu()
        del teacher
        torch.cuda.empty_cache()

        def strict_reload(record: FinalStateRecord) -> None:
            model = _build_model_for_state(base, record.arm)
            incompatible = model.load_state_dict(load_file(record.path, device="cpu"), strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise B16ContractError(f"Strict final-state reload failed: {record.state_id}")
            if any(not bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values() if tensor.is_floating_point()):
                raise B16ContractError(f"Non-finite reloaded state: {record.state_id}")

        counters = {
            "held_image_open_count": 0,
            "held_forward_count": 0,
            "oof_buffer_count": 0,
            "logit_write_count": 0,
            "metric_call_count": 0,
        }
        token = seal_metric_barrier(
            output,
            records,
            accepted_preflight=accepted,
            counters=counters,
            strict_reload=strict_reload,
        )
        logits = {
            name: np.full((EXPECTED_TRAIN_ROWS, CLASSES), np.nan, dtype=np.float32)
            for name in ("stock", "globalized_control", "swiftsurface_candidate", "candidate_r0")
        }
        fill = {name: np.zeros(EXPECTED_TRAIN_ROWS, dtype=np.int8) for name in logits}
        residual_base_ratio = np.full(EXPECTED_TRAIN_ROWS, np.nan, dtype=np.float32)
        active_off_logit_l2 = np.full(EXPECTED_TRAIN_ROWS, np.nan, dtype=np.float32)
        for fold in range(FOLDS):
            held_indices = np.flatnonzero(folds == fold).astype(np.int64)
            batches = [held_indices[start : start + BATCH_SIZE].tolist() for start in range(0, held_indices.size, BATCH_SIZE)]
            loaded: dict[str, nn.Module] = {}
            for arm in ARMS:
                model = _build_model_for_state(base, arm)
                state_path = output / "states" / f"fold_{fold}_{arm}.safetensors"
                model.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
                loaded[arm] = model.to(device).eval()
            with torch.inference_mode():
                for batch_indices in batches:
                    images, _batch_labels, observed = _load_raw_indices(
                        dataset, batch_indices
                    )
                    if observed.tolist() != batch_indices:
                        raise B16ContractError("Held B16 row order changed")
                    inputs = torch.stack([build_eval_view(image) for image in images]).to(device)
                    for image in images:
                        image.close()
                    candidate_active_batch: np.ndarray | None = None
                    for arm in ARMS:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            if arm == "swiftsurface_candidate":
                                output_logits, ratio = _candidate_forward_with_residual_telemetry(
                                    loaded[arm], inputs
                                )
                                residual_base_ratio[batch_indices] = ratio.float().cpu().numpy()
                            else:
                                output_logits = loaded[arm](inputs)
                        values = output_logits.float().cpu().numpy()
                        if arm == "swiftsurface_candidate":
                            candidate_active_batch = values
                        logits[arm][batch_indices] = values
                        fill[arm][batch_indices] += 1
                    candidate = loaded["swiftsurface_candidate"]
                    candidate.mode = SURFACE_OFF_B16_MODE
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        off_logits = candidate(inputs)
                    candidate.mode = SURFACE_SPATIAL_B16_MODE
                    off_values = off_logits.float().cpu().numpy()
                    if candidate_active_batch is None:
                        raise B16ContractError("Active candidate logits missing before ablation")
                    logits["candidate_r0"][batch_indices] = off_values
                    active_off_logit_l2[batch_indices] = np.linalg.norm(
                        candidate_active_batch - off_values, axis=1
                    ).astype(np.float32)
                    fill["candidate_r0"][batch_indices] += 1
            for model in loaded.values():
                model.cpu()
            del loaded
            torch.cuda.empty_cache()
        if any(not np.isfinite(value).all() for value in logits.values()) or any(not np.all(value == 1) for value in fill.values()):
            raise B16ContractError("B16 OOF logits are incomplete/non-finite/non-unique")
        if not np.isfinite(residual_base_ratio).all() or not np.isfinite(active_off_logit_l2).all():
            raise B16ContractError("B16 branch/ablation OOF telemetry is incomplete/non-finite")
        telemetry_npz_sha = atomic_npz(
            output / "oof" / "branch_ablation_telemetry_f32.npz",
            residual_base_norm_ratio=residual_base_ratio,
            active_off_logit_l2=active_off_logit_l2,
        )
        telemetry_summary = {
            "rows": EXPECTED_TRAIN_ROWS,
            "selection_or_tuning_use": False,
            "residual_base_norm_ratio": {
                "p50": float(np.quantile(residual_base_ratio, 0.50, method="linear")),
                "p95": float(np.quantile(residual_base_ratio, 0.95, method="linear")),
            },
            "active_off_logit_l2": {
                "p50": float(np.quantile(active_off_logit_l2, 0.50, method="linear")),
                "p95": float(np.quantile(active_off_logit_l2, 0.95, method="linear")),
            },
            "finite": True,
            "npz_sha256": telemetry_npz_sha,
        }
        telemetry_json_sha = atomic_json(
            output / "oof" / "branch_ablation_telemetry.json",
            telemetry_summary,
        )
        oof_sha = atomic_npz(
            output / "oof" / "oof_logits_f32.npz",
            labels=labels,
            folds=folds,
            groups=groups,
            **{f"logits_{name}": value for name, value in logits.items()},
            **{f"fill_{name}": value for name, value in fill.items()},
        )
        summaries = {name: classification_summary(labels, value, token, folds) for name, value in logits.items()}
        metrics_sha = atomic_json(output / "metrics.json", summaries)
        bootstrap = paired_component_bootstrap(labels, folds, groups, logits, token)
        bootstrap_arrays = {
            f"{comparison}__{metric}": sample
            for comparison, metrics in bootstrap["values"].items()
            for metric, sample in metrics.items()
        }
        bootstrap_npz_sha = atomic_npz(output / "bootstrap_replicates_f64.npz", **bootstrap_arrays)
        bootstrap_json = {key: value for key, value in bootstrap.items() if key != "values"}
        bootstrap_json_sha = atomic_json(output / "bootstrap.json", bootstrap_json)
        end_content = train_content_contract(relative_paths, absolute_paths)
        end_git = _git_contract()
        end_sources = _source_hashes()
        end_student = _validate_asset(args.student_weight, STUDENT_BYTES, STUDENT_SHA256)
        end_dino = _validate_asset(args.dino_weight, DINO_BYTES, DINO_SHA256)
        if end_git != git or end_sources != sources:
            raise B16ContractError("Git HEAD/status or bound source hashes changed during B16")
        if (
            sha256_file(end_student) != STUDENT_SHA256
            or sha256_file(end_dino) != DINO_SHA256
        ):
            raise B16ContractError("Bound pretrained assets changed during B16")
        wall = time.monotonic() - started
        base_integrity = bool(
            start_content == end_content
            and peak_allocated <= MAX_CUDA_ALLOCATED_BYTES
            and wall <= MAX_WALL_SECONDS
            and all(np.all(value == 1) for value in fill.values())
            and telemetry_summary["finite"] is True
        )
        before_final_metadata = _directory_bytes(output)
        gate = assess_b16_gate(
            summaries, bootstrap, token, integrity_complete=base_integrity
        )
        gate_bytes = _json_bytes(gate)
        gate_sha = hashlib.sha256(gate_bytes).hexdigest()

        def make_summary(retained_bytes_final: int) -> dict[str, object]:
            return {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "passed": gate["passed"],
                "validation_protocol_permission": gate["validation_protocol_permission"],
                "validation_execution_permission": False,
                "test_permission": False,
                "train": True,
                "validation": False,
                "test": False,
                "metric_barrier": {"path": token.path, "sha256": token.sha256},
                "train_content": {"start": start_content, "end": end_content},
                "end_revalidation": {
                    "git_equal": end_git == git,
                    "source_hashes_equal": end_sources == sources,
                    "student_asset_sha256": STUDENT_SHA256,
                    "teacher_asset_sha256": DINO_SHA256,
                },
                "resources": {
                    "peak_cuda_allocated_bytes": peak_allocated,
                    "max_cuda_allocated_bytes": MAX_CUDA_ALLOCATED_BYTES,
                    "monotonic_wall_seconds": wall,
                    "max_wall_seconds": MAX_WALL_SECONDS,
                    "retained_bytes_final": int(retained_bytes_final),
                    "max_retained_bytes": MAX_RETAINED_BYTES,
                },
                "artifacts": {
                    "oof_sha256": oof_sha,
                    "metrics_sha256": metrics_sha,
                    "bootstrap_npz_sha256": bootstrap_npz_sha,
                    "bootstrap_json_sha256": bootstrap_json_sha,
                    "gate_sha256": gate_sha,
                    "branch_ablation_telemetry_npz_sha256": telemetry_npz_sha,
                    "branch_ablation_telemetry_json_sha256": telemetry_json_sha,
                },
            }

        summary = make_summary(0)
        for _ in range(8):
            expected_final = (
                before_final_metadata + len(gate_bytes) + len(_json_bytes(summary))
            )
            if summary["resources"]["retained_bytes_final"] == expected_final:
                break
            summary = make_summary(expected_final)
        predicted_final = int(summary["resources"]["retained_bytes_final"])
        if not base_integrity or predicted_final > MAX_RETAINED_BYTES:
            raise B16ContractError(
                "B16 final integrity/resource gate failed before gate/summary promotion"
            )
        if _atomic_bytes(output / "gate.json", gate_bytes) != gate_sha:
            raise B16ContractError("B16 gate digest changed during promotion")
        atomic_json(output / "summary.json", summary)
        observed_final = _directory_bytes(output)
        if (
            observed_final != int(summary["resources"]["retained_bytes_final"])
            or observed_final > MAX_RETAINED_BYTES
        ):
            raise B16ContractError("B16 final retained bytes exceeded or escaped accounting")
        return summary
    except BaseException as exc:
        _save_failure(output, exc, started)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_formal(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

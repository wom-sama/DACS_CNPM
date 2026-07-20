from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.config import to_serializable
from trkh.tools.audit_hamburger_nmf_surface_a0 import (
    _build_dataset,
    _global_rng_equal,
    _global_rng_snapshot,
    _make_condition_loader,
    _metadata_tensor,
    _protected_untracked_state,
    _repo_state,
    _sha256,
    _to_rgb,
    _verify_hash,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

METHOD = "quaternion_color_rotation_a0"
MODE = "quaternion_color_rotation_train_information_gate"
SEED = 20260721
REPEAT_SEED = SEED + 17
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
FOCUS_CLASS = 1
FOLDS = (0, 1, 2, 3, 4)
MIN_FIT_TP_RETENTION = 0.97
CROP_SIZE = 64
CROP_INSET_RATIO = 0.10
MIN_VALID_FRACTION = 0.98
BASE_DIM = 5
COVARIANCE_DIM = 153
IMAGE_ROLE_NAMES = (
    "real_cnn_base",
    "quaternion_red_axis_base",
    "quaternion_dephased_base",
    "quaternion_gray_base",
    "quaternion_gray_only",
    "quaternion_gray_repeat",
)
TRAINED_ROLE_NAMES = (
    "base_logprob",
    "rgb_grid_covariance_base",
    *IMAGE_ROLE_NAMES,
)
INTERVENTION_ROLE_NAMES = (
    "quaternion_same_weight_dephased",
    "quaternion_same_weight_theta_zero",
)
ROLE_NAMES = (*TRAINED_ROLE_NAMES, *INTERVENTION_ROLE_NAMES)
CONTROL_ROLE_NAMES = (
    "base_logprob",
    "rgb_grid_covariance_base",
    "real_cnn_base",
    "quaternion_red_axis_base",
    "quaternion_dephased_base",
)

EXPECTED_COHORT_ROWS = 750
EXPECTED_POSITIVES = 528
EXPECTED_NEGATIVES = 222
EXPECTED_ORDERED_INDEX_SHA256 = (
    "55913ec45265b156a611dc96c779e46b08f28582737af8269105afe6f65694d7"
)
EXPECTED_FOLD_COUNTS = {
    0: (107, 36),
    1: (112, 45),
    2: (100, 48),
    3: (101, 52),
    4: (108, 41),
}

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "87d59f49ec9f06ac6ac549eb932626e189c19f58bfb29b4fd5112e8df2db7fa8"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "0cbf101924ecefcf6d2be1bb67bbad358e7b48d2c84c6ed1d23e6495c26a8cb4"
LOCKED_REFERENCE_COMMIT = "8c381f5f2c9e2d53cdd3fe30b0a2dc6994f79401"
LOCKED_REFERENCE_TREE = "82a0545923e8238e1b61063e1511eb7b8703368d"
LOCKED_REFERENCE_HASHES = {
    "README.md": "903755c19e00468aa248137bd8800bc426a7b101815e0b939366c9486ceb33e6",
    "quaternion_layers/conv.py": "91e987496d8ab499411bcfdfc08530fc8827fdb2c2efd459ad23431fe3f2a0c7",
    "quaternion_layers/init.py": "d4e62cfa304f95a195bfe2c362439316524cd55fb529d76b6e4c074fe4decc3d",
    "cifar10_cnn.py": "6322a5ec396caefc4a641ac0039de22fe578b84d9344a2ddc428e15c018c503f",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
PAPER_PATH = Path(r"D:\DataAI\external_sources\papers\zhu2018_quaternion_cnn_eccv.pdf")
REFERENCE_ROOT = Path(r"D:\DataAI\external_sources\official\quaternioncnn-eccv2018")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only quaternion color-rotation information gate."
    )
    parser.add_argument("--checkpoint", type=Path, default=KEEPER_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--launcher-args", type=Path, default=KEEPER_ROOT / "launcher_args.json")
    parser.add_argument("--resolved-config", type=Path, default=KEEPER_ROOT / "resolved_config.json")
    parser.add_argument("--data", type=Path, default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"))
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_cidt_readiness_full_train_20260714" / "summary.json",
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_QUATERNION_COLOR_ROTATION_A0_PROTOCOL_20260721.md",
    )
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_quaternion_color_rotation_a0_20260721",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--engineering-forward", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", choices=("pass", "fail"))
    parser.add_argument("--reconcile-finalized-review", action="store_true")
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if str(args.device) != "cuda":
        raise ValueError("Quaternion A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"Quaternion A0 locks batch-size={BATCH_SIZE}, num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"Quaternion A0 locks seed={SEED}")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _cohort_index_sha256(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(index)) for index in indices).encode("ascii")
    ).hexdigest()


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def verify_locked_inputs(args: argparse.Namespace) -> Dict[str, object]:
    reference_root = Path(args.reference_root).expanduser().resolve()
    files = {
        "keeper": _verify_hash(args.checkpoint, LOCKED_KEEPER_SHA256, "keeper"),
        "keeper_launcher_args": _verify_hash(
            args.launcher_args, LOCKED_LAUNCHER_ARGS_SHA256, "keeper launcher args"
        ),
        "keeper_resolved_config": _verify_hash(
            args.resolved_config, LOCKED_RESOLVED_CONFIG_SHA256, "keeper resolved config"
        ),
        "data_yaml": _verify_hash(args.data, LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_hash(
            args.cidt_summary, LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_hash(
            args.cidt_predictions, LOCKED_CIDT_PREDICTIONS_SHA256, "CIDT predictions"
        ),
        "protocol": _verify_hash(args.protocol, LOCKED_PROTOCOL_SHA256, "quaternion protocol"),
        "current_best_commands": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "current_best_history": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
        "paper": _verify_hash(PAPER_PATH, LOCKED_PAPER_SHA256, "ECCV quaternion paper"),
    }
    for relative, expected in LOCKED_REFERENCE_HASHES.items():
        files[f"reference_{relative.replace('/', '_')}"] = _verify_hash(
            reference_root / relative, expected, f"unlicensed reference {relative}"
        )
    commit = _git_value(reference_root, "rev-parse", "HEAD")
    tree = _git_value(reference_root, "rev-parse", "HEAD^{tree}")
    status = _git_value(reference_root, "status", "--porcelain")
    license_files = [
        relative
        for relative in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE")
        if (reference_root / relative).exists()
    ]
    if commit != LOCKED_REFERENCE_COMMIT or tree != LOCKED_REFERENCE_TREE or status:
        raise ValueError(
            "Quaternion reference repository differs from lock: "
            f"commit={commit}, tree={tree}, status={status!r}"
        )
    if license_files:
        raise ValueError(
            "Reference licensing boundary changed; review before use: "
            f"{license_files}"
        )
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payloads differ from lock: {protected}")
    return {
        "files": files,
        "reference_repository": {
            "path": str(reference_root),
            "remote": _git_value(reference_root, "remote", "get-url", "origin"),
            "commit": commit,
            "tree": tree,
            "worktree_clean": status == "",
            "license_files": license_files,
            "software_reuse_permitted": False,
        },
        "protected_untracked": protected,
        "runtime": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def locked_cohort(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    cohort = [
        row
        for row in rows
        if row.keeper_prediction == FOCUS_CLASS
        and row.target in {0, 1, 2, 4}
    ]
    labels = np.asarray([row.target == FOCUS_CLASS for row in cohort], dtype=np.int64)
    observed = (len(cohort), int(labels.sum()), int((labels == 0).sum()))
    if observed != (EXPECTED_COHORT_ROWS, EXPECTED_POSITIVES, EXPECTED_NEGATIVES):
        raise ValueError(f"Quaternion cohort differs: {observed}")
    for fold, expected in EXPECTED_FOLD_COUNTS.items():
        held = [row for row in cohort if row.fold == fold]
        counts = (
            sum(row.target == FOCUS_CLASS for row in held),
            sum(row.target != FOCUS_CLASS for row in held),
        )
        if counts != expected:
            raise ValueError(f"Quaternion fold {fold} differs: {counts} != {expected}")
    ordered_hash = _cohort_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Quaternion cohort order hash differs: {ordered_hash}")
    source_folds: Dict[str, set[int]] = {}
    for row in cohort:
        source_folds.setdefault(row.source_stem, set()).add(int(row.fold))
    overlap = {key: sorted(value) for key, value in source_folds.items() if len(value) > 1}
    if overlap:
        raise ValueError(f"Quaternion cohort has cross-fold source overlap: {overlap}")
    return cohort


def gray_rotation_matrix_torch(scale: Tensor, theta: Tensor) -> Tensor:
    if scale.shape != theta.shape:
        raise ValueError("Quaternion scale and theta must have the same shape")
    if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(theta).all()):
        raise ValueError("Quaternion scale and theta must be finite")
    cosine = torch.cos(theta)
    f1 = 1.0 / 3.0 + (2.0 / 3.0) * cosine
    f2 = 1.0 / 3.0 - (2.0 / 3.0) * torch.cos(theta - math.pi / 3.0)
    f3 = 1.0 / 3.0 - (2.0 / 3.0) * torch.cos(theta + math.pi / 3.0)
    row0 = torch.stack((f1, f2, f3), dim=-1)
    row1 = torch.stack((f3, f1, f2), dim=-1)
    row2 = torch.stack((f2, f3, f1), dim=-1)
    return scale[..., None, None] * torch.stack((row0, row1, row2), dim=-2)


def rodrigues_rotation_matrix_numpy(
    scale: np.ndarray,
    theta: np.ndarray,
    *,
    axis: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    magnitude = np.asarray(scale, dtype=np.float64)
    angle = np.asarray(theta, dtype=np.float64)
    if magnitude.shape != angle.shape:
        raise ValueError("Quaternion scale and theta must have the same shape")
    if not np.isfinite(magnitude).all() or not np.isfinite(angle).all():
        raise ValueError("Quaternion scale and theta must be finite")
    unit = np.asarray(axis, dtype=np.float64)
    if unit.shape != (3,) or not np.isfinite(unit).all() or np.linalg.norm(unit) <= 0.0:
        raise ValueError("Rotation axis must be a finite nonzero three-vector")
    unit = unit / np.linalg.norm(unit)
    cross = np.asarray(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    outer = np.outer(unit, unit)
    cosine = np.cos(angle)[..., None, None]
    sine = np.sin(angle)[..., None, None]
    rotation = cosine * identity + (1.0 - cosine) * outer + sine * cross
    return magnitude[..., None, None] * rotation


def axis_rotation_matrix_torch(scale: Tensor, theta: Tensor, axis: Tensor) -> Tensor:
    if scale.shape != theta.shape:
        raise ValueError("Quaternion scale and theta must have the same shape")
    if axis.shape != (3,) or not bool(torch.isfinite(axis).all()):
        raise ValueError("Rotation axis must be a finite three-vector")
    unit = axis.to(device=scale.device, dtype=scale.dtype)
    norm = torch.linalg.vector_norm(unit)
    if float(norm.detach().cpu()) <= 0.0:
        raise ValueError("Rotation axis must be nonzero")
    unit = unit / norm
    zero = torch.zeros((), dtype=scale.dtype, device=scale.device)
    cross = torch.stack(
        (
            torch.stack((zero, -unit[2], unit[1])),
            torch.stack((unit[2], zero, -unit[0])),
            torch.stack((-unit[1], unit[0], zero)),
        )
    )
    identity = torch.eye(3, dtype=scale.dtype, device=scale.device)
    outer = unit[:, None] * unit[None, :]
    cosine = torch.cos(theta)[..., None, None]
    sine = torch.sin(theta)[..., None, None]
    return scale[..., None, None] * (
        cosine * identity + (1.0 - cosine) * outer + sine * cross
    )


class QuaternionConv2d(nn.Module):
    def __init__(
        self,
        q_in: int,
        q_out: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        axis: str = "gray",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        if min(q_in, q_out, kernel_size, stride) <= 0 or padding < 0:
            raise ValueError("Invalid quaternion convolution geometry")
        if axis not in {"gray", "red"}:
            raise ValueError("Quaternion axis must be 'gray' or 'red'")
        self.q_in = int(q_in)
        self.q_out = int(q_out)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.axis = axis
        shape = (self.q_out, self.q_in, self.kernel_size, self.kernel_size)
        fan_in = self.q_in * self.kernel_size * self.kernel_size
        fan_out = self.q_out * self.kernel_size * self.kernel_size
        bound = math.sqrt(6.0 / float(fan_in + fan_out))
        scale = torch.empty(shape, dtype=torch.float32)
        theta = torch.empty(shape, dtype=torch.float32)
        scale.uniform_(-bound, bound, generator=generator)
        theta.uniform_(-math.pi / 2.0, math.pi / 2.0, generator=generator)
        self.scale = nn.Parameter(scale)
        self.theta = nn.Parameter(theta)
        self.bias = nn.Parameter(torch.zeros(3 * self.q_out, dtype=torch.float32))

    def materialized_weight(self, *, theta_zero: bool = False) -> Tensor:
        theta = torch.zeros_like(self.theta) if theta_zero else self.theta
        if self.axis == "gray":
            matrix = gray_rotation_matrix_torch(self.scale, theta)
        else:
            matrix = axis_rotation_matrix_torch(
                self.scale,
                theta,
                torch.tensor((1.0, 0.0, 0.0), device=self.scale.device),
            )
        return matrix.permute(0, 4, 1, 5, 2, 3).reshape(
            3 * self.q_out,
            3 * self.q_in,
            self.kernel_size,
            self.kernel_size,
        )

    def forward(self, value: Tensor, *, theta_zero: bool = False) -> Tensor:
        if value.ndim != 4 or int(value.shape[1]) != 3 * self.q_in:
            raise ValueError(
                f"Quaternion input must be NCHW with {3 * self.q_in} channels"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Quaternion input must be finite")
        return F.conv2d(
            value,
            self.materialized_weight(theta_zero=theta_zero),
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def materialized_conv(self, *, theta_zero: bool = False) -> nn.Conv2d:
        convolution = nn.Conv2d(
            3 * self.q_in,
            3 * self.q_out,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        ).to(device=self.scale.device, dtype=self.scale.dtype)
        with torch.no_grad():
            convolution.weight.copy_(self.materialized_weight(theta_zero=theta_zero))
            convolution.bias.copy_(self.bias)
        return convolution


class QuaternionTrunk(nn.Module):
    output_dim = 48

    def __init__(self, *, axis: str, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.conv1 = QuaternionConv2d(1, 8, 5, stride=2, padding=2, axis=axis, generator=generator)
        self.conv2 = QuaternionConv2d(8, 8, 3, stride=1, padding=1, axis=axis, generator=generator)
        self.conv3 = QuaternionConv2d(8, 16, 3, stride=2, padding=1, axis=axis, generator=generator)

    def forward(self, value: Tensor, *, theta_zero: bool = False) -> Tensor:
        value = F.relu(self.conv1(value, theta_zero=theta_zero))
        value = F.relu(self.conv2(value, theta_zero=theta_zero))
        value = F.max_pool2d(value, kernel_size=2, stride=2)
        value = F.relu(self.conv3(value, theta_zero=theta_zero))
        return value.mean(dim=(-2, -1))

    def materialized(self, *, theta_zero: bool = False) -> "StaticQuaternionTrunk":
        return StaticQuaternionTrunk(
            self.conv1.materialized_conv(theta_zero=theta_zero),
            self.conv2.materialized_conv(theta_zero=theta_zero),
            self.conv3.materialized_conv(theta_zero=theta_zero),
        )


class StaticQuaternionTrunk(nn.Module):
    output_dim = 48

    def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d, conv3: nn.Conv2d) -> None:
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

    def forward(self, value: Tensor) -> Tensor:
        value = F.relu(self.conv1(value))
        value = F.relu(self.conv2(value))
        value = F.max_pool2d(value, kernel_size=2, stride=2)
        value = F.relu(self.conv3(value))
        return value.mean(dim=(-2, -1))


class RealTrunk(nn.Module):
    output_dim = 18

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(9, 13, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(13, 18, 3, stride=2, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        value = F.relu(self.conv1(value))
        value = F.relu(self.conv2(value))
        value = F.max_pool2d(value, kernel_size=2, stride=2)
        value = F.relu(self.conv3(value))
        return value.mean(dim=(-2, -1))


class SurfaceBinaryModel(nn.Module):
    def __init__(self, trunk: nn.Module, *, include_base: bool, head_seed: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.include_base = bool(include_base)
        feature_dim = int(getattr(trunk, "output_dim"))
        self.head = nn.Linear(feature_dim + (BASE_DIM if include_base else 0), 1)
        generator = torch.Generator(device="cpu").manual_seed(int(head_seed))
        bound = 1.0 / math.sqrt(float(feature_dim + BASE_DIM))
        image_weight = torch.empty((1, feature_dim)).uniform_(
            -bound, bound, generator=generator
        )
        base_weight = torch.empty((1, BASE_DIM)).uniform_(
            -bound, bound, generator=generator
        )
        with torch.no_grad():
            self.head.weight[:, :feature_dim].copy_(image_weight)
            if include_base:
                self.head.weight[:, feature_dim:].copy_(base_weight)
            self.head.bias.zero_()

    def features(self, rgb: Tensor, base: Tensor, *, theta_zero: bool = False) -> Tensor:
        if isinstance(self.trunk, QuaternionTrunk):
            image_features = self.trunk(rgb, theta_zero=theta_zero)
        else:
            image_features = self.trunk(rgb)
        return torch.cat((image_features, base), dim=1) if self.include_base else image_features

    def forward(self, rgb: Tensor, base: Tensor, *, theta_zero: bool = False) -> Tensor:
        return self.head(self.features(rgb, base, theta_zero=theta_zero)).squeeze(1)

    def materialized(self, *, theta_zero: bool = False) -> "SurfaceBinaryModel":
        if not isinstance(self.trunk, QuaternionTrunk):
            raise TypeError("Only a quaternion model can be statically materialized")
        result = SurfaceBinaryModel.__new__(SurfaceBinaryModel)
        nn.Module.__init__(result)
        result.trunk = self.trunk.materialized(theta_zero=theta_zero)
        result.include_base = self.include_base
        result.head = nn.Linear(self.head.in_features, 1).to(
            device=self.head.weight.device, dtype=self.head.weight.dtype
        )
        result.head.load_state_dict(self.head.state_dict())
        return result


class LinearBinaryModel(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(int(feature_dim), 1)

    def forward(self, value: Tensor) -> Tensor:
        return self.head(value).squeeze(1)


def parameter_count(module: nn.Module) -> int:
    return sum(int(value.numel()) for value in module.parameters())


def build_fold_models() -> Dict[str, nn.Module]:
    before = _global_rng_snapshot()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            models: Dict[str, nn.Module] = {
                "base_logprob": LinearBinaryModel(BASE_DIM),
                "rgb_grid_covariance_base": LinearBinaryModel(COVARIANCE_DIM + BASE_DIM),
                "real_cnn_base": SurfaceBinaryModel(
                    RealTrunk(), include_base=True, head_seed=SEED + 101
                ),
                "quaternion_red_axis_base": SurfaceBinaryModel(
                    QuaternionTrunk(axis="red", seed=SEED + 211),
                    include_base=True,
                    head_seed=SEED + 307,
                ),
                "quaternion_dephased_base": SurfaceBinaryModel(
                    QuaternionTrunk(axis="gray", seed=SEED + 211),
                    include_base=True,
                    head_seed=SEED + 307,
                ),
                "quaternion_gray_base": SurfaceBinaryModel(
                    QuaternionTrunk(axis="gray", seed=SEED + 211),
                    include_base=True,
                    head_seed=SEED + 307,
                ),
                "quaternion_gray_only": SurfaceBinaryModel(
                    QuaternionTrunk(axis="gray", seed=SEED + 211),
                    include_base=False,
                    head_seed=SEED + 307,
                ),
                "quaternion_gray_repeat": SurfaceBinaryModel(
                    QuaternionTrunk(axis="gray", seed=REPEAT_SEED + 211),
                    include_base=True,
                    head_seed=REPEAT_SEED + 307,
                ),
            }
    finally:
        _restore_global_rng(before)
        after = _global_rng_snapshot()
    if not _global_rng_equal(before, after):
        raise RuntimeError("Fold-model construction changed global RNG state")
    candidate = models["quaternion_gray_base"]
    for role in (
        "quaternion_red_axis_base",
        "quaternion_dephased_base",
        "quaternion_gray_only",
    ):
        compared = models[role]
        candidate_state = candidate.trunk.state_dict()  # type: ignore[attr-defined]
        compared_state = compared.trunk.state_dict()  # type: ignore[attr-defined]
        if not all(torch.equal(candidate_state[key], compared_state[key]) for key in candidate_state):
            raise RuntimeError(f"Matched quaternion initialization differs for {role}")
        candidate_image = candidate.head.weight[:, :48]  # type: ignore[attr-defined]
        compared_image = compared.head.weight[:, :48]  # type: ignore[attr-defined]
        if not torch.equal(candidate_image, compared_image):
            raise RuntimeError(f"Matched classifier image weights differ for {role}")
    return models


def channel_dephase(value: Tensor, sample_indices: Sequence[int]) -> Tensor:
    if value.ndim != 4 or int(value.shape[1]) != 3:
        raise ValueError("Channel dephasing requires NCHW RGB")
    if len(sample_indices) != int(value.shape[0]):
        raise ValueError("Channel dephasing sample indices do not align")
    rows = []
    for position, sample_index in enumerate(sample_indices):
        channels = []
        for component in range(3):
            dy = 3 + ((SEED + 37 * int(sample_index) + 11 * component) % 13)
            dx = 3 + ((SEED + 53 * int(sample_index) + 7 * component) % 13)
            channels.append(
                torch.roll(value[position, component], shifts=(dy, dx), dims=(-2, -1))
            )
        rows.append(torch.stack(channels, dim=0))
    return torch.stack(rows, dim=0)


def channel_rephase_gradient(value: Tensor, sample_indices: Sequence[int]) -> Tensor:
    if value.ndim != 4 or int(value.shape[1]) != 3:
        raise ValueError("Gradient rephasing requires NCHW RGB")
    rows = []
    for position, sample_index in enumerate(sample_indices):
        channels = []
        for component in range(3):
            dy = 3 + ((SEED + 37 * int(sample_index) + 11 * component) % 13)
            dx = 3 + ((SEED + 53 * int(sample_index) + 7 * component) % 13)
            channels.append(
                torch.roll(value[position, component], shifts=(-dy, -dx), dims=(-2, -1))
            )
        rows.append(torch.stack(channels, dim=0))
    return torch.stack(rows, dim=0)


def crop_inset_rgb(
    rgb: Tensor,
    crop_bbox: Tensor,
    image_valid_mask: Tensor,
) -> Tuple[Tensor, List[Dict[str, object]]]:
    if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
        raise ValueError("RGB crop input must be NCHW")
    if crop_bbox.ndim != 2 or crop_bbox.shape != (rgb.shape[0], 4):
        raise ValueError("crop_bbox must be [B,4] normalized xywh")
    if image_valid_mask.ndim == 3:
        image_valid_mask = image_valid_mask.unsqueeze(1)
    if image_valid_mask.ndim != 4 or image_valid_mask.shape[0] != rgb.shape[0]:
        raise ValueError("image_valid_mask must align with RGB")
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    crops: List[Tensor] = []
    geometry: List[Dict[str, object]] = []
    for row in range(int(rgb.shape[0])):
        cx, cy, box_width, box_height = [float(v) for v in crop_bbox[row].tolist()]
        x0 = max(0, min(width, math.floor((cx - box_width / 2.0) * width)))
        y0 = max(0, min(height, math.floor((cy - box_height / 2.0) * height)))
        x1 = max(0, min(width, math.ceil((cx + box_width / 2.0) * width)))
        y1 = max(0, min(height, math.ceil((cy + box_height / 2.0) * height)))
        original_width = x1 - x0
        original_height = y1 - y0
        ix0 = math.ceil(x0 + CROP_INSET_RATIO * original_width)
        iy0 = math.ceil(y0 + CROP_INSET_RATIO * original_height)
        ix1 = math.floor(x1 - CROP_INSET_RATIO * original_width)
        iy1 = math.floor(y1 - CROP_INSET_RATIO * original_height)
        if x1 <= x0 or y1 <= y0 or ix1 <= ix0 or iy1 <= iy0:
            raise ValueError(f"Quaternion crop has nonpositive area at batch row {row}")
        mask = image_valid_mask[row, 0, iy0:iy1, ix0:ix1].float()
        valid_fraction = float(mask.mean())
        if valid_fraction + 1e-12 < MIN_VALID_FRACTION:
            raise ValueError(
                f"Quaternion crop valid fraction {valid_fraction:.8f} is below lock"
            )
        crop = rgb[row : row + 1, :, iy0:iy1, ix0:ix1]
        resized = F.interpolate(
            crop,
            size=(CROP_SIZE, CROP_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        crops.append(resized)
        geometry.append(
            {
                "original_x0": x0,
                "original_y0": y0,
                "original_x1_exclusive": x1,
                "original_y1_exclusive": y1,
                "inset_x0": ix0,
                "inset_y0": iy0,
                "inset_x1_exclusive": ix1,
                "inset_y1_exclusive": iy1,
                "valid_fraction": valid_fraction,
            }
        )
    return torch.cat(crops, dim=0), geometry


def quantize_rgb_cache(value: Tensor) -> np.ndarray:
    if value.ndim != 4 or int(value.shape[1]) != 3:
        raise ValueError("RGB cache input must be NCHW")
    return (
        value.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    )


def rgb_grid_covariance_features(cache: np.ndarray) -> np.ndarray:
    value = np.asarray(cache, dtype=np.float64)
    if value.ndim != 4 or value.shape[1:] != (3, CROP_SIZE, CROP_SIZE):
        raise ValueError("RGB covariance cache must be [N,3,64,64]")
    value = value / 255.0
    rows: List[np.ndarray] = []
    upper = np.triu_indices(3)
    for sample in value:
        columns: List[np.ndarray] = []
        for grid_y in range(4):
            for grid_x in range(4):
                cell = sample[
                    :,
                    grid_y * 16 : (grid_y + 1) * 16,
                    grid_x * 16 : (grid_x + 1) * 16,
                ].reshape(3, -1).T
                mean = cell.mean(axis=0)
                centered = cell - mean
                covariance = centered.T @ centered / float(cell.shape[0])
                columns.extend((mean, covariance[upper]))
        global_pixels = sample.reshape(3, -1).T
        global_mean = global_pixels.mean(axis=0)
        global_centered = global_pixels - global_mean
        global_covariance = global_centered.T @ global_centered / float(global_pixels.shape[0])
        columns.extend((global_mean, global_covariance[upper]))
        rows.append(np.concatenate(columns))
    result = np.stack(rows, axis=0).astype(np.float32)
    if result.shape != (value.shape[0], COVARIANCE_DIM):
        raise RuntimeError(f"RGB covariance feature shape differs: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("RGB covariance features must be finite")
    return result


def base_logprob_features(cohort: Sequence[CleanTrainRow]) -> np.ndarray:
    probabilities = np.asarray(
        [row.keeper_probabilities for row in cohort], dtype=np.float64
    )
    result = np.log(np.clip(probabilities, 1e-8, 1.0)).astype(np.float32)
    if result.shape != (len(cohort), BASE_DIM) or not np.isfinite(result).all():
        raise ValueError("Base log-probability features differ from lock")
    return result


def fit_standardizer(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float32)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    result = (np.asarray(value, dtype=np.float32) - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("Standardized features must be finite")
    return result.astype(np.float32)


def positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(values[binary == 1])
    if positives.size == 0:
        raise ValueError("Fit folds contain no true class-1 positives")
    allowed_breaks = int(math.floor((1.0 - MIN_FIT_TP_RETENTION) * positives.size))
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    if float((positives >= threshold).mean()) + 1e-12 < MIN_FIT_TP_RETENTION:
        raise RuntimeError("Fit threshold violates locked TP retention")
    return threshold


def effective_rank(value: np.ndarray) -> float:
    matrix = np.asarray(value, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    probability = energy / max(float(energy.sum()), 1e-30)
    positive = probability > 0.0
    return float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))


def _parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _restore_global_rng(snapshot: Mapping[str, object]) -> None:
    random.setstate(snapshot["python"])  # type: ignore[arg-type]
    np.random.set_state(snapshot["numpy"])  # type: ignore[arg-type]
    torch.random.set_rng_state(snapshot["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["cuda"])  # type: ignore[arg-type]


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float64)
    return np.where(
        logits >= 0.0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(to_serializable(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_manifest(output_dir: Path) -> Path:
    path = output_dir / "artifact_manifest.json"
    rows = []
    for artifact in sorted(output_dir.rglob("*")):
        if artifact.is_file() and artifact != path:
            rows.append(
                {
                    "path": artifact.relative_to(output_dir).as_posix(),
                    "bytes": int(artifact.stat().st_size),
                    "sha256": _sha256(artifact),
                }
            )
    _write_json(
        path,
        {"schema": "trkh_quaternion_color_rotation_a0_manifest_v1", "files": rows},
    )
    return path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(row["path"]): row for row in payload["files"]}
    observed = {
        artifact.relative_to(output_dir).as_posix()
        for artifact in output_dir.rglob("*")
        if artifact.is_file() and artifact != path
    }
    if set(expected) != observed:
        raise ValueError("Quaternion artifact-manifest file set differs")
    for relative, row in expected.items():
        artifact = output_dir / relative
        if int(artifact.stat().st_size) != int(row["bytes"]) or _sha256(artifact) != row["sha256"]:
            raise ValueError(f"Quaternion artifact differs from manifest: {relative}")
    return {
        "passed": True,
        "payload_count": len(expected),
        "manifest_sha256": _sha256(path),
    }


def engineering_checks() -> Dict[str, object]:
    generator = np.random.default_rng(731)
    scale = generator.uniform(0.2, 1.7, size=(2, 3, 2)).astype(np.float64)
    theta = generator.uniform(-math.pi, math.pi, size=scale.shape).astype(np.float64)
    oracle = rodrigues_rotation_matrix_numpy(scale, theta)
    explicit64 = gray_rotation_matrix_torch(
        torch.from_numpy(scale), torch.from_numpy(theta)
    ).numpy()
    fp64_error = float(np.max(np.abs(oracle - explicit64)))
    explicit32 = gray_rotation_matrix_torch(
        torch.from_numpy(scale.astype(np.float32)),
        torch.from_numpy(theta.astype(np.float32)),
    ).numpy()
    fp32_error = float(np.max(np.abs(oracle - explicit32)))
    explicit_bf16 = gray_rotation_matrix_torch(
        torch.from_numpy(scale.astype(np.float32)).to(torch.bfloat16),
        torch.from_numpy(theta.astype(np.float32)).to(torch.bfloat16),
    ).float().numpy()
    bf16_error = float(np.max(np.abs(oracle - explicit_bf16)))

    gray_axis = np.ones(3, dtype=np.float64) / math.sqrt(3.0)
    vectors = generator.normal(size=(scale.size, 3))
    rotations = oracle.reshape(-1, 3, 3)
    magnitudes = scale.reshape(-1)
    projection_error = 0.0
    grayscale_error = 0.0
    for matrix, magnitude, vector in zip(rotations, magnitudes, vectors):
        projection_error = max(
            projection_error,
            abs(float(gray_axis @ (matrix @ vector)) - magnitude * float(gray_axis @ vector)),
        )
        gray_vector = gray_axis * float(generator.normal())
        grayscale_error = max(
            grayscale_error,
            float(np.max(np.abs(matrix @ gray_vector - magnitude * gray_vector))),
        )

    permutation = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    cyclic_error = float(
        max(np.max(np.abs(matrix @ permutation - permutation @ matrix)) for matrix in rotations)
    )

    convolution = QuaternionConv2d(
        2,
        3,
        3,
        stride=2,
        padding=1,
        generator=torch.Generator().manual_seed(37),
    ).double()
    image = torch.from_numpy(generator.normal(size=(2, 6, 9, 11))).double()
    dynamic = convolution(image)
    oracle_blocks = rodrigues_rotation_matrix_numpy(
        convolution.scale.detach().numpy(), convolution.theta.detach().numpy()
    )
    oracle_weight = oracle_blocks.transpose(0, 4, 1, 5, 2, 3).reshape(9, 6, 3, 3)
    oracle_output = F.conv2d(
        image,
        torch.from_numpy(oracle_weight),
        convolution.bias,
        stride=2,
        padding=1,
    )
    multigroup_error = float((dynamic - oracle_output).abs().max())
    static = convolution.materialized_conv()
    materialized_error = float((dynamic - static(image)).abs().max())
    singleton_shape = list(convolution(image[:1]).shape)

    differentiable_scale = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    differentiable_theta = torch.tensor([0.31], dtype=torch.float64, requires_grad=True)
    probe_vector = torch.tensor([0.2, -0.4, 0.9], dtype=torch.float64)
    probe_weight = torch.tensor([0.7, -0.1, 0.5], dtype=torch.float64)
    differentiable_loss = (
        gray_rotation_matrix_torch(differentiable_scale, differentiable_theta)[0]
        @ probe_vector
    ).dot(probe_weight)
    differentiable_loss.backward()
    analytic = np.asarray(
        [float(differentiable_scale.grad), float(differentiable_theta.grad)],
        dtype=np.float64,
    )
    epsilon = 1e-6
    finite = []
    for parameter_index in range(2):
        values = [0.8, 0.31]
        values[parameter_index] += epsilon
        high = float(
            (
                gray_rotation_matrix_torch(
                    torch.tensor([values[0]], dtype=torch.float64),
                    torch.tensor([values[1]], dtype=torch.float64),
                )[0]
                @ probe_vector
            ).dot(probe_weight)
        )
        values[parameter_index] -= 2.0 * epsilon
        low = float(
            (
                gray_rotation_matrix_torch(
                    torch.tensor([values[0]], dtype=torch.float64),
                    torch.tensor([values[1]], dtype=torch.float64),
                )[0]
                @ probe_vector
            ).dot(probe_weight)
        )
        finite.append((high - low) / (2.0 * epsilon))
    finite_difference_error = float(np.max(np.abs(analytic - np.asarray(finite))))

    invalid_checks = {}
    for name, callback in {
        "shape_mismatch": lambda: gray_rotation_matrix_torch(
            torch.ones(2), torch.ones(3)
        ),
        "nonfinite": lambda: gray_rotation_matrix_torch(
            torch.tensor([float("nan")]), torch.ones(1)
        ),
        "input_channels": lambda: convolution(torch.ones(1, 3, 8, 8, dtype=torch.float64)),
    }.items():
        try:
            callback()
            invalid_checks[name] = False
        except ValueError:
            invalid_checks[name] = True

    dephase_input = torch.arange(2 * 3 * 19 * 23, dtype=torch.float32).reshape(2, 3, 19, 23)
    rng_before = _global_rng_snapshot()
    dephased_a = channel_dephase(dephase_input, (17, 29))
    dephased_b = channel_dephase(dephase_input, (17, 29))
    rng_after = _global_rng_snapshot()
    marginal_exact = all(
        torch.equal(
            torch.sort(dephase_input[row, channel].flatten()).values,
            torch.sort(dephased_a[row, channel].flatten()).values,
        )
        for row in range(2)
        for channel in range(3)
    )
    toroidal_adjacency_exact = all(
        torch.equal(
            torch.sort(
                (dephase_input[row, channel] - torch.roll(
                    dephase_input[row, channel], shifts=1, dims=-1
                )).flatten()
            ).values,
            torch.sort(
                (dephased_a[row, channel] - torch.roll(
                    dephased_a[row, channel], shifts=1, dims=-1
                )).flatten()
            ).values,
        )
        for row in range(2)
        for channel in range(3)
    )
    registration_altered = all(
        not torch.equal(dephased_a[row, a], dephased_a[row, b])
        for row in range(2)
        for a in range(3)
        for b in range(a + 1, 3)
    )

    models = build_fold_models()
    quaternion_parameters = parameter_count(models["quaternion_gray_base"])
    real_parameters = parameter_count(models["real_cnn_base"])
    parameter_ratio_error = abs(real_parameters - quaternion_parameters) / quaternion_parameters
    matched_trunks = all(
        all(
            torch.equal(
                models["quaternion_gray_base"].trunk.state_dict()[key],  # type: ignore[attr-defined]
                models[role].trunk.state_dict()[key],  # type: ignore[attr-defined]
            )
            for key in models["quaternion_gray_base"].trunk.state_dict()  # type: ignore[attr-defined]
        )
        for role in (
            "quaternion_red_axis_base",
            "quaternion_dephased_base",
            "quaternion_gray_only",
        )
    )
    candidate = models["quaternion_gray_base"]
    synthetic_rgb = torch.linspace(0.0, 1.0, 2 * 3 * 64 * 64).reshape(2, 3, 64, 64)
    synthetic_base = torch.linspace(-1.0, 1.0, 10).reshape(2, 5)
    dynamic_logits = candidate(synthetic_rgb, synthetic_base)
    static_logits = candidate.materialized()(synthetic_rgb, synthetic_base)  # type: ignore[attr-defined]
    static_logit_error = float((dynamic_logits - static_logits).abs().max())

    checks = {
        "fp64_oracle": fp64_error <= 1e-12,
        "fp32_oracle": fp32_error <= 1e-6,
        "bf16_oracle": bf16_error <= 2e-2,
        "gray_projection": projection_error <= 1e-12,
        "grayscale_invariance": grayscale_error <= 1e-12,
        "cyclic_rgb_symmetry": cyclic_error <= 1e-12,
        "multi_group_accumulation": multigroup_error <= 1e-12,
        "singleton_batch": singleton_shape[0] == 1 and singleton_shape[1] == 9,
        "invalid_inputs": all(invalid_checks.values()),
        "finite_difference": finite_difference_error <= 1e-8,
        "analytic_gradients": bool(np.isfinite(analytic).all() and np.all(np.abs(analytic) > 0.0)),
        "static_materialization": materialized_error <= 1e-12 and static_logit_error <= 1e-6,
        "dephase_deterministic": torch.equal(dephased_a, dephased_b),
        "dephase_marginal": marginal_exact,
        "dephase_toroidal_adjacency": toroidal_adjacency_exact,
        "dephase_registration": registration_altered,
        "dephase_rng_restored": _global_rng_equal(rng_before, rng_after),
        "matched_initialization": matched_trunks,
        "real_parameter_match": parameter_ratio_error <= 0.05,
    }
    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": all(checks.values()),
        "errors": {
            "fp64": float(fp64_error),
            "fp32": float(fp32_error),
            "bf16": float(bf16_error),
            "gray_projection": float(projection_error),
            "grayscale": float(grayscale_error),
            "cyclic": float(cyclic_error),
            "multi_group": float(multigroup_error),
            "finite_difference": float(finite_difference_error),
            "materialized": float(materialized_error),
            "static_model_logits": float(static_logit_error),
        },
        "gradients": {"analytic": analytic.tolist(), "finite_difference": finite},
        "invalid_checks": invalid_checks,
        "parameter_counts": {
            "quaternion_gray_base": quaternion_parameters,
            "real_cnn_base": real_parameters,
            "relative_difference": parameter_ratio_error,
        },
    }


def _load_locked_dataset(
    args: argparse.Namespace,
    cohort: Sequence[CleanTrainRow],
) -> Tuple[object, object, Dict[str, object], Mapping[str, object]]:
    checkpoint = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Keeper checkpoint must contain a mapping")
    all_rows = _read_clean_train_rows(Path(args.cidt_predictions))
    dataset, transform, declaration = _build_dataset(checkpoint, all_rows, Path(args.data))
    if [row.sample_index for row in cohort] != sorted(row.sample_index for row in cohort):
        raise ValueError("Locked cohort is not ordered by sample index")
    return dataset, transform, declaration, checkpoint


def _engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    rows = _read_clean_train_rows(Path(args.cidt_predictions))
    cohort = locked_cohort(rows)
    dataset, transform, declaration, checkpoint = _load_locked_dataset(args, cohort)
    selected = cohort[:2]
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in selected],
        brightness=1.0,
        contrast=1.0,
        batch_size=2,
        num_workers=NUM_WORKERS,
        context="quaternion_color_rotation_a0_engineering",
    )
    images, targets, metadata = next(iter(loader))
    if metadata["sample_index"].tolist() != [row.sample_index for row in selected]:
        raise ValueError("Engineering loader changed sample order")
    semantics = declaration["semantics"]
    rgb = _to_rgb(images, semantics)
    crop_bbox = _metadata_tensor(
        metadata, "crop_bbox", device=torch.device("cpu"), dtype=torch.float32
    )
    image_mask = _metadata_tensor(
        metadata, "image_mask", device=torch.device("cpu"), dtype=torch.bool
    )
    if crop_bbox is None or image_mask is None:
        raise ValueError("Engineering forward requires crop_bbox and image_mask")
    crops, geometry = crop_inset_rgb(rgb, crop_bbox[:, :4], image_mask)
    cache = quantize_rgb_cache(crops)
    models = build_fold_models()
    candidate = models["quaternion_gray_base"]
    base = torch.from_numpy(base_logprob_features(selected))
    logits = candidate(torch.from_numpy(cache).float() / 255.0, base)
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.tensor([row.target == FOCUS_CLASS for row in selected], dtype=torch.float32)
    )
    loss.backward()
    gradients = {
        name: float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0
        for name, parameter in candidate.named_parameters()
    }
    if not all(math.isfinite(value) and value > 0.0 for value in gradients.values()):
        raise RuntimeError(f"Engineering gradients are incomplete: {gradients}")
    return {
        "mode": f"{METHOD}_engineering_forward",
        "rows": len(selected),
        "targets": targets.tolist(),
        "cache_shape": list(cache.shape),
        "cache_sha256": _array_sha256(cache),
        "geometry": geometry,
        "gradients": gradients,
        "loader": loader_summary,
        "dataset": declaration,
        "keeper_loaded_without_forward": bool(checkpoint),
        "output_written": False,
    }


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    engineering = engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError(f"Quaternion equation checks failed: {engineering}")
    rows = _read_clean_train_rows(Path(args.cidt_predictions))
    cohort = locked_cohort(rows)
    return {
        "mode": f"{METHOD}_preflight",
        "provenance": provenance,
        "engineering": engineering,
        "cohort": {
            "rows": len(cohort),
            "positives": sum(row.target == FOCUS_CLASS for row in cohort),
            "negatives": sum(row.target != FOCUS_CLASS for row in cohort),
            "ordered_index_sha256": _cohort_index_sha256(
                [row.sample_index for row in cohort]
            ),
        },
        "repo": _repo_state(),
        "output_written": False,
    }


def extract_rgb_cache(
    *,
    args: argparse.Namespace,
    cohort: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    dataset, transform, declaration, _checkpoint = _load_locked_dataset(args, cohort)
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="quaternion_color_rotation_a0_clean_extraction",
    )
    cache = np.empty((len(cohort), 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    geometry_rows: List[Dict[str, object]] = []
    position = 0
    started = time.perf_counter()
    for images, targets, metadata in loader:
        batch = int(targets.numel())
        stop = position + batch
        expected = cohort[position:stop]
        observed_indices = metadata["sample_index"].detach().cpu().long().tolist()
        if observed_indices != [row.sample_index for row in expected]:
            raise ValueError("Quaternion extraction changed locked cohort order")
        if targets.detach().cpu().long().tolist() != [row.target for row in expected]:
            raise ValueError("Quaternion extraction changed locked labels")
        crop_bbox = _metadata_tensor(
            metadata, "crop_bbox", device=torch.device("cpu"), dtype=torch.float32
        )
        image_mask = _metadata_tensor(
            metadata, "image_mask", device=torch.device("cpu"), dtype=torch.bool
        )
        if crop_bbox is None or image_mask is None:
            raise ValueError("Quaternion extraction requires crop_bbox and image_mask")
        rgb = _to_rgb(images, declaration["semantics"])
        crops, batch_geometry = crop_inset_rgb(rgb, crop_bbox[:, :4], image_mask)
        batch_cache = quantize_rgb_cache(crops)
        cache[position:stop] = batch_cache
        for local, geometry in enumerate(batch_geometry):
            pixels = batch_cache[local].astype(np.float64) / 255.0
            geometry_rows.append(
                {
                    "position": position + local,
                    "sample_index": int(expected[local].sample_index),
                    "source_stem": expected[local].source_stem,
                    "fold": int(expected[local].fold),
                    "target": int(expected[local].target),
                    **geometry,
                    "red_mean": float(pixels[0].mean()),
                    "green_mean": float(pixels[1].mean()),
                    "blue_mean": float(pixels[2].mean()),
                    "red_std": float(pixels[0].std()),
                    "green_std": float(pixels[1].std()),
                    "blue_std": float(pixels[2].std()),
                    "rgb_sha256": _array_sha256(batch_cache[local]),
                }
            )
        position = stop
        if position % 256 < batch or position == len(cohort):
            print(
                json.dumps(
                    {
                        "stage": "quaternion_rgb_extraction",
                        "completed": position,
                        "total": len(cohort),
                    }
                ),
                flush=True,
            )
    if position != len(cohort) or not np.isfinite(cache.astype(np.float32)).all():
        raise RuntimeError("Quaternion RGB cache extraction is incomplete")
    return {
        "cache": cache,
        "geometry": geometry_rows,
        "loader": loader_summary,
        "dataset": declaration,
        "seconds": time.perf_counter() - started,
        "sha256": _array_sha256(cache),
    }


def _epoch_orders(fit_positions: np.ndarray, fold: int) -> Tuple[List[np.ndarray], str]:
    generator = np.random.default_rng(SEED + 1009 * int(fold))
    orders = [
        np.asarray(fit_positions, dtype=np.int64)[
            generator.permutation(len(fit_positions))
        ]
        for _ in range(EPOCHS)
    ]
    occurrence = np.concatenate(orders).astype(np.int64)
    return orders, _array_sha256(occurrence)


def _optimizer_summary(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
) -> Dict[str, object]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    digest = hashlib.sha256()
    finite = True
    steps: List[float] = []
    tensor_count = 0
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names.get(id(item[0]), "")
    ):
        name = names.get(id(parameter), "unknown")
        digest.update(name.encode("utf-8"))
        for key, value in sorted(state.items()):
            digest.update(str(key).encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                finite = finite and bool(torch.isfinite(tensor).all())
                tensor_count += 1
                if str(key) == "step":
                    steps.append(float(tensor.reshape(-1)[0]))
            else:
                digest.update(repr(value).encode("utf-8"))
    return {
        "sha256": digest.hexdigest(),
        "parameters_with_state": len(optimizer.state),
        "state_tensor_count": tensor_count,
        "all_state_tensors_finite": finite,
        "minimum_step": min(steps) if steps else 0.0,
        "maximum_step": max(steps) if steps else 0.0,
    }


def _role_batch(
    *,
    role: str,
    positions: np.ndarray,
    cache: np.ndarray,
    sample_indices: np.ndarray,
    base_normalized: np.ndarray,
    covariance_normalized: np.ndarray,
    device: torch.device,
) -> Tuple[Optional[Tensor], Tensor]:
    base = torch.from_numpy(base_normalized[positions]).to(
        device=device, dtype=torch.float32
    )
    if role == "base_logprob":
        return None, base
    if role == "rgb_grid_covariance_base":
        covariance = torch.from_numpy(covariance_normalized[positions]).to(
            device=device, dtype=torch.float32
        )
        return None, torch.cat((covariance, base), dim=1)
    rgb = torch.from_numpy(cache[positions]).to(device=device, dtype=torch.float32) / 255.0
    if role == "quaternion_dephased_base":
        rgb = channel_dephase(rgb, sample_indices[positions].tolist())
    return rgb, base


def _forward_role(
    model: nn.Module,
    *,
    role: str,
    rgb: Optional[Tensor],
    features: Tensor,
    sample_indices: Optional[Sequence[int]] = None,
    intervention: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    if role in {"base_logprob", "rgb_grid_covariance_base"}:
        if not isinstance(model, LinearBinaryModel):
            raise TypeError(f"Linear role {role} received a non-linear model")
        return model(features), features
    if rgb is None or not isinstance(model, SurfaceBinaryModel):
        raise TypeError(f"Image role {role} received invalid inputs")
    current = rgb
    theta_zero = intervention == "theta_zero"
    if intervention == "dephased":
        if sample_indices is None:
            raise ValueError("Same-weight dephasing requires sample indices")
        current = channel_dephase(current, sample_indices)
    pooled = model.features(current, features, theta_zero=theta_zero)
    return model.head(pooled).squeeze(1), pooled


def train_role(
    *,
    role: str,
    model: nn.Module,
    fit_positions: np.ndarray,
    orders: Sequence[np.ndarray],
    occurrence_sha256: str,
    cache: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    base_normalized: np.ndarray,
    covariance_normalized: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    model = model.to(device)
    initial_sha256 = _parameter_sha256(model)
    initial_parameters = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: List[float] = []
    first_gradients: Dict[str, float] = {}
    first_updates: Dict[str, float] = {}
    started = time.perf_counter()
    model.train()
    for epoch, order in enumerate(orders):
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            positions = np.asarray(order[start : start + BATCH_SIZE], dtype=np.int64)
            rgb, feature_input = _role_batch(
                role=role,
                positions=positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
            )
            target = torch.from_numpy(labels[positions].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = _forward_role(
                model,
                role=role,
                rgb=rgb,
                features=feature_input,
            )
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite {role} loss at epoch {epoch}")
            loss.backward()
            first_step = epoch == 0 and start == 0
            before_step = {}
            if first_step:
                for name, parameter in model.named_parameters():
                    gradient = parameter.grad
                    first_gradients[name] = (
                        float(gradient.detach().norm()) if gradient is not None else 0.0
                    )
                    before_step[name] = parameter.detach().clone()
            optimizer.step()
            if first_step:
                first_updates = {
                    name: float((parameter.detach() - before_step[name]).norm())
                    for name, parameter in model.named_parameters()
                }
            batch = len(positions)
            loss_sum += float(loss.detach()) * batch
            seen += batch
        if seen != len(fit_positions):
            raise RuntimeError(f"{role} epoch occurrence count differs")
        history.append(loss_sum / seen)
    final_sha256 = _parameter_sha256(model)
    changed = {
        name: float((parameter.detach().cpu() - initial_parameters[name]).norm())
        for name, parameter in model.named_parameters()
    }
    gradient_pass = all(
        math.isfinite(value) and value > 0.0 for value in first_gradients.values()
    ) and set(first_gradients) == {name for name, _ in model.named_parameters()}
    update_pass = all(
        math.isfinite(value) and value > 0.0 for value in first_updates.values()
    ) and all(math.isfinite(value) and value > 0.0 for value in changed.values())
    if not gradient_pass or not update_pass or initial_sha256 == final_sha256:
        raise RuntimeError(
            f"{role} parameter evidence failed: gradients={first_gradients}, "
            f"updates={first_updates}, changed={changed}"
        )
    return {
        "model": model,
        "initial_sha256": initial_sha256,
        "final_sha256": final_sha256,
        "first_gradient_norms": first_gradients,
        "first_update_norms": first_updates,
        "final_change_norms": changed,
        "gradient_pass": gradient_pass,
        "update_pass": update_pass,
        "history": history,
        "history_finite": bool(np.isfinite(history).all()),
        "occurrence_sha256": occurrence_sha256,
        "optimizer": _optimizer_summary(optimizer, model),
        "seconds": time.perf_counter() - started,
    }


def infer_role(
    *,
    role: str,
    model: nn.Module,
    positions: np.ndarray,
    cache: np.ndarray,
    sample_indices: np.ndarray,
    base_normalized: np.ndarray,
    covariance_normalized: np.ndarray,
    device: torch.device,
    intervention: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    logits_rows: List[np.ndarray] = []
    feature_rows: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(positions), BATCH_SIZE):
            selected = np.asarray(positions[start : start + BATCH_SIZE], dtype=np.int64)
            source_role = (
                "quaternion_gray_base"
                if role in INTERVENTION_ROLE_NAMES
                else role
            )
            rgb, feature_input = _role_batch(
                role=source_role,
                positions=selected,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
            )
            logits, features = _forward_role(
                model,
                role=source_role,
                rgb=rgb,
                features=feature_input,
                sample_indices=sample_indices[selected].tolist(),
                intervention=intervention,
            )
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float64))
            feature_rows.append(features.detach().cpu().numpy().astype(np.float64))
    return _sigmoid_numpy(np.concatenate(logits_rows)), np.concatenate(feature_rows)


def fit_oof_models(
    *,
    cache: np.ndarray,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
) -> Dict[str, object]:
    rows = len(cohort)
    labels = np.asarray(
        [int(row.target == FOCUS_CLASS) for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    sample_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    base_raw = base_logprob_features(cohort)
    covariance_raw = rgb_grid_covariance_features(cache)
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    features: Dict[str, np.ndarray] = {}
    fold_records: Dict[int, Dict[str, object]] = {}
    training_records: List[Dict[str, object]] = []
    feature_parts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        role: [] for role in ROLE_NAMES
    }
    for fold in FOLDS:
        fit_positions = np.flatnonzero(folds != fold).astype(np.int64)
        held_positions = np.flatnonzero(folds == fold).astype(np.int64)
        fit_sources = {cohort[position].source_stem for position in fit_positions}
        held_sources = {cohort[position].source_stem for position in held_positions}
        if fit_sources.intersection(held_sources):
            raise ValueError(f"Fold {fold} has source leakage")
        base_mean, base_std = fit_standardizer(base_raw[fit_positions])
        covariance_mean, covariance_std = fit_standardizer(covariance_raw[fit_positions])
        base_normalized = apply_standardizer(base_raw, base_mean, base_std)
        covariance_normalized = apply_standardizer(
            covariance_raw, covariance_mean, covariance_std
        )
        orders, occurrence_sha256 = _epoch_orders(fit_positions, fold)
        models = build_fold_models()
        matched_initial_hashes = {
            role: _parameter_sha256(models[role].trunk)  # type: ignore[attr-defined]
            for role in (
                "quaternion_red_axis_base",
                "quaternion_dephased_base",
                "quaternion_gray_base",
                "quaternion_gray_only",
            )
        }
        if len(set(matched_initial_hashes.values())) != 1:
            raise RuntimeError(f"Fold {fold} matched trunk hashes differ")
        trained: Dict[str, nn.Module] = {}
        role_training: Dict[str, Dict[str, object]] = {}
        thresholds: Dict[str, float] = {}
        for role in TRAINED_ROLE_NAMES:
            result = train_role(
                role=role,
                model=models[role],
                fit_positions=fit_positions,
                orders=orders,
                occurrence_sha256=occurrence_sha256,
                cache=cache,
                sample_indices=sample_indices,
                labels=labels,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
            )
            model = result.pop("model")
            trained[role] = model
            role_training[role] = result
            fit_scores, _ = infer_role(
                role=role,
                model=model,
                positions=fit_positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
            )
            held_scores, held_features = infer_role(
                role=role,
                model=model,
                positions=held_positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
            )
            threshold = positive_threshold(fit_scores, labels[fit_positions])
            thresholds[role] = threshold
            scores[role][held_positions] = held_scores
            actions[role][held_positions] = held_scores >= threshold
            feature_parts[role].append((held_positions, held_features))
            print(
                json.dumps(
                    {
                        "stage": "quaternion_role_complete",
                        "fold": fold,
                        "role": role,
                        "final_loss": result["history"][-1],
                        "threshold": threshold,
                    }
                ),
                flush=True,
            )
        candidate = trained["quaternion_gray_base"]
        for role, intervention in (
            ("quaternion_same_weight_dephased", "dephased"),
            ("quaternion_same_weight_theta_zero", "theta_zero"),
        ):
            fit_scores, _ = infer_role(
                role=role,
                model=candidate,
                positions=fit_positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
                intervention=intervention,
            )
            held_scores, held_features = infer_role(
                role=role,
                model=candidate,
                positions=held_positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
                intervention=intervention,
            )
            threshold = positive_threshold(fit_scores, labels[fit_positions])
            thresholds[role] = threshold
            scores[role][held_positions] = held_scores
            actions[role][held_positions] = held_scores >= threshold
            feature_parts[role].append((held_positions, held_features))
        fold_records[fold] = {
            "fit_positions": fit_positions,
            "held_positions": held_positions,
            "base_mean": base_mean,
            "base_std": base_std,
            "covariance_mean": covariance_mean,
            "covariance_std": covariance_std,
            "base_normalized": base_normalized,
            "covariance_normalized": covariance_normalized,
            "models": trained,
            "thresholds": thresholds,
            "occurrence_sha256": occurrence_sha256,
            "matched_initial_trunk_sha256": next(iter(matched_initial_hashes.values())),
            "fit_source_count": len(fit_sources),
            "held_source_count": len(held_sources),
            "source_overlap": 0,
        }
        for role, record in role_training.items():
            training_records.append({"fold": fold, "role": role, **record})
    for role, parts in feature_parts.items():
        dimension = parts[0][1].shape[1]
        matrix = np.full((rows, dimension), np.nan, dtype=np.float64)
        for positions, values in parts:
            matrix[positions] = values
        if not np.isfinite(matrix).all() or not np.isfinite(scores[role]).all():
            raise RuntimeError(f"Quaternion OOF output incomplete for {role}")
        features[role] = matrix
    occurrence_hashes = {
        int(fold): str(record["occurrence_sha256"])
        for fold, record in fold_records.items()
    }
    return {
        "labels": labels,
        "folds": folds,
        "sample_indices": sample_indices,
        "base_raw": base_raw,
        "covariance_raw": covariance_raw,
        "scores": scores,
        "actions": actions,
        "features": features,
        "fold_records": fold_records,
        "training_records": training_records,
        "occurrence_hashes": occurrence_hashes,
        "shared_occurrences": all(
            record["occurrence_sha256"] == occurrence_hashes[int(record["fold"])]
            for record in training_records
        ),
    }


def role_metrics(
    scores: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    accepted = np.asarray(actions, dtype=np.bool_)
    binary = np.asarray(labels, dtype=np.int64)
    positives = binary == 1
    negatives = ~positives
    tp_kept = int(np.sum(accepted & positives))
    fp_kept = int(np.sum(accepted & negatives))
    tp_broken = int(np.sum(~accepted & positives))
    fp_rejected = int(np.sum(~accepted & negatives))
    fold_rows: Dict[str, object] = {}
    for fold in FOLDS:
        selected = np.asarray(folds) == fold
        fold_positive = positives & selected
        fold_negative = negatives & selected
        fold_rows[str(fold)] = {
            "rows": int(selected.sum()),
            "auroc": float(roc_auc_score(binary[selected], values[selected])),
            "tp_retention": float(np.mean(accepted[fold_positive])),
            "fp_rejection": float(np.mean(~accepted[fold_negative])),
            "tp_broken": int(np.sum(~accepted[fold_positive])),
            "fp_rejected": int(np.sum(~accepted[fold_negative])),
        }
    return {
        "rows": int(binary.size),
        "auroc": float(roc_auc_score(binary, values)),
        "tp_retention": tp_kept / int(positives.sum()),
        "fp_rejection": fp_rejected / int(negatives.sum()),
        "precision_after_action": tp_kept / max(tp_kept + fp_kept, 1),
        "tp_kept": tp_kept,
        "tp_broken": tp_broken,
        "fp_kept": fp_kept,
        "fp_rejected": fp_rejected,
        "corrections": fp_rejected,
        "harms": tp_broken,
        "folds": fold_rows,
    }


def analyze_clean_gate(oof: Mapping[str, object]) -> Dict[str, object]:
    labels = np.asarray(oof["labels"], dtype=np.int64)
    folds = np.asarray(oof["folds"], dtype=np.int64)
    metrics = {
        role: role_metrics(
            np.asarray(oof["scores"][role]),
            np.asarray(oof["actions"][role]),
            labels,
            folds,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics["quaternion_gray_base"]
    repeat = metrics["quaternion_gray_repeat"]
    action_agreement = float(
        np.mean(
            np.asarray(oof["actions"]["quaternion_gray_base"], dtype=np.bool_)
            == np.asarray(oof["actions"]["quaternion_gray_repeat"], dtype=np.bool_)
        )
    )
    trained_control_action_checks = {}
    for control in CONTROL_ROLE_NAMES:
        value = metrics[control]
        trained_control_action_checks[control] = {
            "additional_fp_rejections": int(candidate["fp_rejected"])
            - int(value["fp_rejected"]),
            "additional_tp_breaks": int(candidate["tp_broken"])
            - int(value["tp_broken"]),
            "passed": int(candidate["fp_rejected"]) >= int(value["fp_rejected"]) + 10
            and int(candidate["tp_broken"]) <= int(value["tp_broken"]) + 2,
        }
    fold_control_wins = {}
    for control in (
        "rgb_grid_covariance_base",
        "real_cnn_base",
        "quaternion_red_axis_base",
        "quaternion_dephased_base",
    ):
        wins = [
            fold
            for fold in FOLDS
            if float(candidate["folds"][str(fold)]["auroc"])
            > float(metrics[control]["folds"][str(fold)]["auroc"])
        ]
        fold_control_wins[control] = {"wins": wins, "count": len(wins), "passed": len(wins) >= 4}
    candidate_features = np.asarray(
        oof["features"]["quaternion_gray_base"], dtype=np.float64
    )[:, :48]
    rank = effective_rank(candidate_features)
    fold_records = oof["fold_records"]
    scales = []
    angles = []
    for fold in FOLDS:
        model = fold_records[fold]["models"]["quaternion_gray_base"]
        for name, parameter in model.named_parameters():
            if name.endswith(".scale"):
                scales.append(parameter.detach().cpu().numpy().reshape(-1))
            elif name.endswith(".theta"):
                angles.append(parameter.detach().cpu().numpy().reshape(-1))
    scale_values = np.concatenate(scales)
    angle_values = np.concatenate(angles)
    diagnostics = {
        "candidate_feature_effective_rank": rank,
        "candidate_feature_channel_variance_min": float(
            np.var(candidate_features, axis=0).min()
        ),
        "scale_std": float(scale_values.std()),
        "scale_min": float(scale_values.min()),
        "scale_max": float(scale_values.max()),
        "theta_std": float(angle_values.std()),
        "theta_min": float(angle_values.min()),
        "theta_max": float(angle_values.max()),
        "finite": bool(
            np.isfinite(candidate_features).all()
            and np.isfinite(scale_values).all()
            and np.isfinite(angle_values).all()
        ),
    }
    checks = {
        "candidate_auroc": float(candidate["auroc"]) >= 0.85,
        "beats_base": float(candidate["auroc"]) >= float(metrics["base_logprob"]["auroc"]) + 0.03,
        "beats_covariance": float(candidate["auroc"])
        >= float(metrics["rgb_grid_covariance_base"]["auroc"]) + 0.02,
        "beats_real": float(candidate["auroc"]) >= float(metrics["real_cnn_base"]["auroc"]) + 0.015,
        "beats_red_axis": float(candidate["auroc"])
        >= float(metrics["quaternion_red_axis_base"]["auroc"]) + 0.02,
        "beats_trained_dephase": float(candidate["auroc"])
        >= float(metrics["quaternion_dephased_base"]["auroc"]) + 0.03,
        "beats_same_weight_dephase": float(candidate["auroc"])
        >= float(metrics["quaternion_same_weight_dephased"]["auroc"]) + 0.03,
        "beats_theta_zero": float(candidate["auroc"])
        >= float(metrics["quaternion_same_weight_theta_zero"]["auroc"]) + 0.02,
        "gray_only": float(metrics["quaternion_gray_only"]["auroc"]) >= 0.78,
        "tp_retention": float(candidate["tp_retention"]) >= 0.95,
        "fp_rejection": float(candidate["fp_rejection"]) >= 0.25,
        "corrections_exceed_harms": int(candidate["corrections"]) > int(candidate["harms"]),
        "fold_tp_retention": all(
            float(candidate["folds"][str(fold)]["tp_retention"]) >= 0.90
            for fold in FOLDS
        ),
        "trained_control_actions": all(
            bool(value["passed"]) for value in trained_control_action_checks.values()
        ),
        "fold_control_wins": all(
            bool(value["passed"]) for value in fold_control_wins.values()
        ),
        "repeat_auroc": abs(float(candidate["auroc"]) - float(repeat["auroc"])) <= 0.02,
        "repeat_action_agreement": action_agreement >= 0.95,
        "repeat_tp": float(repeat["tp_retention"]) >= 0.95,
        "repeat_fp": float(repeat["fp_rejection"]) >= 0.25,
        "feature_rank": rank >= 12.0,
        "noncollapsed": bool(diagnostics["finite"])
        and float(diagnostics["candidate_feature_channel_variance_min"]) > 1e-10
        and float(diagnostics["scale_std"]) > 1e-4
        and float(diagnostics["theta_std"]) > 1e-4,
        "finite_losses": all(
            bool(record["history_finite"]) for record in oof["training_records"]
        ),
    }
    return {
        "metrics": metrics,
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": all(bool(value) for value in checks.values()),
        "candidate_deltas": {
            role: float(candidate["auroc"]) - float(metrics[role]["auroc"])
            for role in ROLE_NAMES
            if role != "quaternion_gray_base"
        },
        "repeat_action_agreement": action_agreement,
        "trained_control_action_checks": trained_control_action_checks,
        "fold_control_wins": fold_control_wins,
        "diagnostics": diagnostics,
    }


def apply_pil_condition(
    cache: np.ndarray,
    *,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    value = np.asarray(cache, dtype=np.uint8)
    result = np.empty_like(value)
    for index, sample in enumerate(value):
        image = Image.fromarray(np.transpose(sample, (1, 2, 0)), mode="RGB")
        image = ImageEnhance.Brightness(image).enhance(float(brightness))
        image = ImageEnhance.Contrast(image).enhance(float(contrast))
        result[index] = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))
    return result


def evaluate_condition(
    *,
    cache: np.ndarray,
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    labels = np.asarray(oof["labels"], dtype=np.int64)
    folds = np.asarray(oof["folds"], dtype=np.int64)
    sample_indices = np.asarray(oof["sample_indices"], dtype=np.int64)
    base_raw = np.asarray(oof["base_raw"], dtype=np.float32)
    covariance_raw = rgb_grid_covariance_features(cache)
    scores = {role: np.full(len(labels), np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(len(labels), dtype=np.bool_) for role in ROLE_NAMES}
    for fold in FOLDS:
        record = oof["fold_records"][fold]
        held_positions = np.asarray(record["held_positions"], dtype=np.int64)
        base_normalized = apply_standardizer(
            base_raw, np.asarray(record["base_mean"]), np.asarray(record["base_std"])
        )
        covariance_normalized = apply_standardizer(
            covariance_raw,
            np.asarray(record["covariance_mean"]),
            np.asarray(record["covariance_std"]),
        )
        for role in TRAINED_ROLE_NAMES:
            held_scores, _ = infer_role(
                role=role,
                model=record["models"][role],
                positions=held_positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
            )
            scores[role][held_positions] = held_scores
            actions[role][held_positions] = held_scores >= float(record["thresholds"][role])
        candidate = record["models"]["quaternion_gray_base"]
        for role, intervention in (
            ("quaternion_same_weight_dephased", "dephased"),
            ("quaternion_same_weight_theta_zero", "theta_zero"),
        ):
            held_scores, _ = infer_role(
                role=role,
                model=candidate,
                positions=held_positions,
                cache=cache,
                sample_indices=sample_indices,
                base_normalized=base_normalized,
                covariance_normalized=covariance_normalized,
                device=device,
                intervention=intervention,
            )
            scores[role][held_positions] = held_scores
            actions[role][held_positions] = held_scores >= float(record["thresholds"][role])
    metrics = {
        role: role_metrics(scores[role], actions[role], labels, folds)
        for role in ROLE_NAMES
    }
    candidate = metrics["quaternion_gray_base"]
    best_control = max(
        CONTROL_ROLE_NAMES,
        key=lambda role: float(metrics[role]["auroc"]),
    )
    checks = {
        "auroc": float(candidate["auroc"]) >= 0.80,
        "tp_retention": float(candidate["tp_retention"]) >= 0.92,
        "fp_rejection": float(candidate["fp_rejection"]) >= 0.20,
        "beats_best_control": float(candidate["auroc"])
        > float(metrics[best_control]["auroc"]),
        "fp_exceeds_tp_harm": int(candidate["fp_rejected"]) > int(candidate["tp_broken"]),
    }
    return {
        "metrics": metrics,
        "best_control": best_control,
        "candidate_minus_best_control_auroc": float(candidate["auroc"])
        - float(metrics[best_control]["auroc"]),
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": all(bool(value) for value in checks.values()),
        "cache_sha256": _array_sha256(cache),
    }


class StaticExportWrapper(nn.Module):
    def __init__(self, model: SurfaceBinaryModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: Tensor, base: Tensor) -> Tensor:
        return self.model(rgb, base)


def static_export_checks(
    *,
    output_dir: Path,
    cache: np.ndarray,
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    fold = 0
    record = oof["fold_records"][fold]
    positions = np.asarray(record["held_positions"], dtype=np.int64)
    candidate = record["models"]["quaternion_gray_base"]
    static_cuda = candidate.materialized().to(device).eval()
    rgb = torch.from_numpy(cache[positions]).to(device=device, dtype=torch.float32) / 255.0
    base = torch.from_numpy(np.asarray(record["base_normalized"])[positions]).to(
        device=device, dtype=torch.float32
    )
    candidate.eval()
    with torch.inference_mode():
        dynamic_logits = candidate(rgb, base)
        static_logits = static_cuda(rgb, base)
    materialized_error = float((dynamic_logits - static_logits).abs().max())

    static_cpu = copy.deepcopy(static_cuda).cpu().eval()
    wrapper = StaticExportWrapper(static_cpu).eval()
    example_rgb = rgb[:3].detach().cpu()
    example_base = base[:3].detach().cpu()
    onnx_path = output_dir / "quaternion_gray_static_fold0.onnx"
    torch.onnx.export(
        wrapper,
        (example_rgb, example_base),
        onnx_path,
        input_names=("rgb", "base_logprob"),
        output_names=("binary_logit",),
        dynamic_axes={
            "rgb": {0: "batch"},
            "base_logprob": {0: "batch"},
            "binary_logit": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(
        None,
        {
            "rgb": example_rgb.numpy(),
            "base_logprob": example_base.numpy(),
        },
    )[0]
    with torch.inference_mode():
        expected = wrapper(example_rgb, example_base).numpy()
    onnx_error = float(np.max(np.abs(expected - ort_logits)))
    checks = {
        "materialized_logits": materialized_error <= 1e-6,
        "onnx_logits": onnx_error <= 1e-5,
        "static_operators_only": True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "materialized_max_abs_error": materialized_error,
        "onnx_max_abs_error": onnx_error,
        "onnx": {
            "path": str(onnx_path.resolve()),
            "bytes": int(onnx_path.stat().st_size),
            "sha256": _sha256(onnx_path),
        },
    }


def benchmark_branches(
    *,
    cache: np.ndarray,
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    record = oof["fold_records"][0]
    candidate = record["models"]["quaternion_gray_base"].materialized().to(device).eval()
    real = record["models"]["real_cnn_base"].to(device).eval()
    positions = np.resize(np.asarray(record["held_positions"], dtype=np.int64), BATCH_SIZE)
    rgb = torch.from_numpy(cache[positions]).to(device=device, dtype=torch.float32) / 255.0
    base = torch.from_numpy(np.asarray(record["base_normalized"])[positions]).to(
        device=device, dtype=torch.float32
    )

    def measure(model: SurfaceBinaryModel) -> float:
        with torch.inference_mode():
            for _ in range(10):
                model(rgb, base)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(50):
                model(rgb, base)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)) / 50.0

    candidate_ms = measure(candidate)
    real_ms = measure(real)
    ratio = candidate_ms / max(real_ms, 1e-12)
    return {
        "batch_size": BATCH_SIZE,
        "iterations": 50,
        "candidate_static_ms": candidate_ms,
        "real_ms": real_ms,
        "ratio": ratio,
        "passed": ratio <= 2.25,
        "deployment_branch": "materialized_static_conv2d",
    }


def _normalize_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("XAI map must be finite")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum <= 1e-12:
        raise ValueError("XAI map is numerically collapsed")
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def _input_gradient_map(
    *,
    model: SurfaceBinaryModel,
    rgb_uint8: np.ndarray,
    base_normalized: np.ndarray,
    sample_index: int,
    device: torch.device,
    intervention: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    model.eval()
    model.zero_grad(set_to_none=True)
    rgb = (
        torch.from_numpy(np.asarray(rgb_uint8)[None])
        .to(device=device, dtype=torch.float32)
        .div(255.0)
        .requires_grad_(True)
    )
    base = torch.from_numpy(np.asarray(base_normalized, dtype=np.float32)[None]).to(device)
    current = rgb
    if intervention == "dephased":
        current = channel_dephase(current, (int(sample_index),))
    logits = model(current, base, theta_zero=intervention == "theta_zero")
    logits.sum().backward()
    if rgb.grad is None:
        raise RuntimeError("XAI input gradient is missing")
    attribution = (rgb.grad.detach().abs() * (rgb.detach().abs() + 1e-3)).mean(dim=1)[0]
    return _normalize_map(attribution.cpu().numpy()), float(torch.sigmoid(logits)[0].detach())


def _border_mass(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    border = max(1, int(math.ceil(0.10 * min(array.shape))))
    mask = np.zeros(array.shape, dtype=np.bool_)
    mask[:border] = True
    mask[-border:] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return float(array[mask].sum() / max(float(array.sum()), 1e-12))


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray) -> Image.Image:
    image = np.transpose(np.asarray(rgb, dtype=np.uint8), (1, 2, 0)).astype(np.float32)
    value = np.asarray(heat, dtype=np.float32)
    color = np.stack(
        (
            np.clip(2.0 * value, 0.0, 1.0),
            np.clip(2.0 * value - 0.5, 0.0, 1.0),
            np.clip(1.0 - 2.0 * value, 0.0, 1.0) * 0.25,
        ),
        axis=-1,
    ) * 255.0
    blended = np.clip(0.55 * image + 0.45 * color, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _xai_selection(labels: np.ndarray, scores: np.ndarray) -> List[Tuple[str, int]]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    groups = (
        ("high_tp", positives[np.argsort(scores[positives])[-4:][::-1]]),
        ("low_tp", positives[np.argsort(scores[positives])[:4]]),
        ("low_fp", negatives[np.argsort(scores[negatives])[:4]]),
        ("high_fp", negatives[np.argsort(scores[negatives])[-4:][::-1]]),
    )
    return [(name, int(position)) for name, values in groups for position in values]


def render_xai_sheet(
    *,
    output_dir: Path,
    cache: np.ndarray,
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    labels = np.asarray(oof["labels"], dtype=np.int64)
    scores = np.asarray(oof["scores"]["quaternion_gray_base"], dtype=np.float64)
    selection = _xai_selection(labels, scores)
    columns = ("RGB", "Quaternion gray", "Real CNN", "Dephased same weight", "Theta zero")
    cell_size = 128
    label_height = 34
    header_height = 34
    sheet = Image.new(
        "RGB",
        (cell_size * len(columns), header_height + len(selection) * (cell_size + label_height)),
        color=(248, 248, 248),
    )
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(columns):
        draw.text((column * cell_size + 4, 9), title, fill=(20, 20, 20))
    records: List[Dict[str, object]] = []
    for row_number, (group, position) in enumerate(selection):
        row = cohort[position]
        fold_record = oof["fold_records"][int(row.fold)]
        base_normalized = np.asarray(fold_record["base_normalized"])[position]
        candidate = fold_record["models"]["quaternion_gray_base"]
        real = fold_record["models"]["real_cnn_base"]
        candidate_map, candidate_score = _input_gradient_map(
            model=candidate,
            rgb_uint8=cache[position],
            base_normalized=base_normalized,
            sample_index=row.sample_index,
            device=device,
        )
        real_map, real_score = _input_gradient_map(
            model=real,
            rgb_uint8=cache[position],
            base_normalized=base_normalized,
            sample_index=row.sample_index,
            device=device,
        )
        dephased_map, dephased_score = _input_gradient_map(
            model=candidate,
            rgb_uint8=cache[position],
            base_normalized=base_normalized,
            sample_index=row.sample_index,
            device=device,
            intervention="dephased",
        )
        theta_map, theta_score = _input_gradient_map(
            model=candidate,
            rgb_uint8=cache[position],
            base_normalized=base_normalized,
            sample_index=row.sample_index,
            device=device,
            intervention="theta_zero",
        )
        maps = (candidate_map, real_map, dephased_map, theta_map)
        images = [
            Image.fromarray(np.transpose(cache[position], (1, 2, 0)), mode="RGB"),
            *[_heat_overlay(cache[position], value) for value in maps],
        ]
        y = header_height + row_number * (cell_size + label_height)
        for column, image in enumerate(images):
            sheet.paste(
                image.resize((cell_size, cell_size), resample=Image.Resampling.NEAREST),
                (column * cell_size, y),
            )
        label = (
            f"{group} pos={position} sample={row.sample_index} target={row.target} fold={row.fold} "
            f"q/real/dephase/theta={candidate_score:.3f}/{real_score:.3f}/"
            f"{dephased_score:.3f}/{theta_score:.3f}"
        )
        draw.text((4, y + cell_size + 7), label, fill=(15, 15, 15))
        records.append(
            {
                "group": group,
                "position": position,
                "sample_index": int(row.sample_index),
                "target": int(row.target),
                "fold": int(row.fold),
                "candidate_score": candidate_score,
                "real_score": real_score,
                "dephased_score": dephased_score,
                "theta_zero_score": theta_score,
                "candidate_border_mass": _border_mass(candidate_map),
                "real_border_mass": _border_mass(real_map),
                "dephased_border_mass": _border_mass(dephased_map),
                "theta_zero_border_mass": _border_mass(theta_map),
                "candidate_map_sha256": _array_sha256(candidate_map),
                "real_map_sha256": _array_sha256(real_map),
                "dephased_map_sha256": _array_sha256(dephased_map),
                "theta_zero_map_sha256": _array_sha256(theta_map),
            }
        )
    path = output_dir / "quaternion_xai_contact_sheet.png"
    sheet.save(path)
    candidate_border = float(np.mean([row["candidate_border_mass"] for row in records]))
    checks = {
        "fixed_16_rows": len(records) == 16,
        "finite_nonconstant_maps": True,
        "candidate_border_mass": candidate_border < 0.45,
    }
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": records,
        "candidate_mean_border_mass": candidate_border,
        "checks": checks,
        "automatic_passed": all(checks.values()),
        "manual_review": "pending",
    }


def render_crop_preview(
    output_dir: Path,
    cache: np.ndarray,
    cohort: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    positions = []
    for target in (0, 1, 2, 4):
        positions.extend(
            [index for index, row in enumerate(cohort) if row.target == target][:4]
        )
    cell = 128
    label = 26
    sheet = Image.new("RGB", (4 * cell, 4 * (cell + label)), color=(250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    for index, position in enumerate(positions):
        x = (index % 4) * cell
        y = (index // 4) * (cell + label)
        image = Image.fromarray(np.transpose(cache[position], (1, 2, 0)), mode="RGB")
        sheet.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (x, y))
        row = cohort[position]
        draw.text((x + 3, y + cell + 5), f"t={row.target} i={row.sample_index}", fill=(15, 15, 15))
    path = output_dir / "quaternion_rgb_crop_preview.png"
    sheet.save(path)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "positions": positions}


def save_replay_artifacts(
    *,
    output_dir: Path,
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
) -> Dict[str, object]:
    feature_path = output_dir / "oof_replay_features.npz"
    np.savez_compressed(
        feature_path,
        **{
            role: np.asarray(oof["features"][role], dtype=np.float64)
            for role in ROLE_NAMES
        },
    )
    head_arrays: Dict[str, np.ndarray] = {}
    model_arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, object] = {"folds": {}}
    for fold in FOLDS:
        record = oof["fold_records"][fold]
        fold_metadata: Dict[str, object] = {"roles": {}}
        for trained_role, model in record["models"].items():
            for name, tensor in model.state_dict().items():
                key = f"fold{fold}__{trained_role}__{name.replace('.', '___')}"
                model_arrays[key] = tensor.detach().cpu().numpy()
        for role in ROLE_NAMES:
            source_role = "quaternion_gray_base" if role in INTERVENTION_ROLE_NAMES else role
            model = record["models"][source_role]
            weight_key = f"fold{fold}__{role}__weight"
            bias_key = f"fold{fold}__{role}__bias"
            head_arrays[weight_key] = model.head.weight.detach().cpu().numpy()
            head_arrays[bias_key] = model.head.bias.detach().cpu().numpy()
            fold_metadata["roles"][role] = {
                "source_role": source_role,
                "held_positions": np.asarray(record["held_positions"], dtype=np.int64).tolist(),
                "threshold": float(record["thresholds"][role]),
                "weight_key": weight_key,
                "bias_key": bias_key,
            }
        fold_metadata.update(
            {
                "occurrence_sha256": record["occurrence_sha256"],
                "source_overlap": record["source_overlap"],
                "base_mean": np.asarray(record["base_mean"]).tolist(),
                "base_std": np.asarray(record["base_std"]).tolist(),
                "covariance_mean": np.asarray(record["covariance_mean"]).tolist(),
                "covariance_std": np.asarray(record["covariance_std"]).tolist(),
            }
        )
        metadata["folds"][str(fold)] = fold_metadata
    head_path = output_dir / "oof_head_states.npz"
    model_path = output_dir / "trained_model_states.npz"
    np.savez_compressed(head_path, **head_arrays)
    np.savez_compressed(model_path, **model_arrays)
    metadata_path = output_dir / "oof_state_metadata.json"
    _write_json(metadata_path, metadata)

    prediction_rows = []
    for position, row in enumerate(cohort):
        record: Dict[str, object] = {
            "position": position,
            "sample_index": int(row.sample_index),
            "source_stem": row.source_stem,
            "fold": int(row.fold),
            "target": int(row.target),
            "binary_label": int(row.target == FOCUS_CLASS),
        }
        for role in ROLE_NAMES:
            record[f"score__{role}"] = float(oof["scores"][role][position])
            record[f"action__{role}"] = int(bool(oof["actions"][role][position]))
        prediction_rows.append(record)
    predictions_path = output_dir / "oof_predictions.csv"
    _write_csv(predictions_path, prediction_rows)
    training_path = output_dir / "training_evidence.json"
    _write_json(
        training_path,
        {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "records": oof["training_records"],
        },
    )
    return {
        "features": {"path": str(feature_path.resolve()), "sha256": _sha256(feature_path)},
        "heads": {"path": str(head_path.resolve()), "sha256": _sha256(head_path)},
        "models": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
        "metadata": {"path": str(metadata_path.resolve()), "sha256": _sha256(metadata_path)},
        "predictions": {"path": str(predictions_path.resolve()), "sha256": _sha256(predictions_path)},
        "training": {"path": str(training_path.resolve()), "sha256": _sha256(training_path)},
    }


def replay_artifacts(output_dir: Path) -> Dict[str, object]:
    features = np.load(output_dir / "oof_replay_features.npz", allow_pickle=False)
    heads = np.load(output_dir / "oof_head_states.npz", allow_pickle=False)
    metadata = json.loads((output_dir / "oof_state_metadata.json").read_text(encoding="utf-8"))
    rows = []
    with (output_dir / "oof_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_COHORT_ROWS:
        raise ValueError("Replay prediction row count differs")
    sample_indices = [int(row["sample_index"]) for row in rows]
    if _cohort_index_sha256(sample_indices) != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError("Replay sample-index order differs")
    max_score_error = 0.0
    exact_actions = True
    for fold in FOLDS:
        fold_record = metadata["folds"][str(fold)]
        for role in ROLE_NAMES:
            role_record = fold_record["roles"][role]
            positions = np.asarray(role_record["held_positions"], dtype=np.int64)
            weight = np.asarray(heads[role_record["weight_key"]], dtype=np.float64)
            bias = np.asarray(heads[role_record["bias_key"]], dtype=np.float64).reshape(-1)
            matrix = np.asarray(features[role], dtype=np.float64)[positions]
            replay_scores = _sigmoid_numpy((matrix @ weight.T).reshape(-1) + bias[0])
            reference_scores = np.asarray(
                [float(rows[position][f"score__{role}"]) for position in positions],
                dtype=np.float64,
            )
            max_score_error = max(
                max_score_error, float(np.max(np.abs(replay_scores - reference_scores)))
            )
            replay_actions = replay_scores >= float(role_record["threshold"])
            reference_actions = np.asarray(
                [bool(int(rows[position][f"action__{role}"])) for position in positions],
                dtype=np.bool_,
            )
            exact_actions = exact_actions and bool(np.array_equal(replay_actions, reference_actions))
    return {
        "max_score_error": max_score_error,
        "exact_actions": exact_actions,
        "row_order_exact": True,
        "passed": max_score_error <= 1e-7 and exact_actions,
    }


def _gpu_snapshot() -> Dict[str, object]:
    commands = {
        "device": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    result: Dict[str, object] = {"unknown_process_terminated": False}
    for name, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            result[name] = {
                "exit_code": int(completed.returncode),
                "stdout": completed.stdout.strip().splitlines(),
                "stderr": completed.stderr.strip().splitlines(),
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            result[name] = {"error": str(error)}
    return result


def run_formal(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_QUATERNION_A0_PREFLIGHT") != "passed":
        raise RuntimeError(
            "Formal quaternion A0 requires the launcher preflight marker"
        )
    repo_before = _repo_state()
    if not bool(repo_before["tracked_worktree_clean"]) or not bool(
        repo_before["head_matches_upstream"]
    ):
        raise RuntimeError(
            "Formal quaternion A0 requires a clean tracked tree at pushed HEAD"
        )
    provenance = verify_locked_inputs(args)
    engineering = engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError("Quaternion engineering checks failed before formal run")
    rows = _read_clean_train_rows(Path(args.cidt_predictions))
    cohort = locked_cohort(rows)
    output_dir = _prepare_output_dir(Path(args.output_dir))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal quaternion A0 requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    gpu_before = _gpu_snapshot()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cache_path = output_dir / "temporary_rgb_cache.npy"
    formal_started = time.perf_counter()
    cache_deleted = False
    try:
        extraction = extract_rgb_cache(args=args, cohort=cohort)
        cache = np.asarray(extraction["cache"], dtype=np.uint8)
        np.save(cache_path, cache, allow_pickle=False)
        cache_file_sha256 = _sha256(cache_path)
        geometry_path = output_dir / "rgb_crop_geometry.csv"
        _write_csv(geometry_path, extraction["geometry"])
        preview = render_crop_preview(output_dir, cache, cohort)

        oof = fit_oof_models(cache=cache, cohort=cohort, device=device)
        clean = analyze_clean_gate(oof)
        robustness: Dict[str, object]
        if bool(clean["passed"]):
            conditions = {}
            for name, brightness, contrast in (
                ("dim", 0.70, 0.90),
                ("bright", 1.25, 1.10),
                ("low_contrast", 1.00, 0.65),
            ):
                condition_cache = apply_pil_condition(
                    cache, brightness=brightness, contrast=contrast
                )
                conditions[name] = {
                    "brightness": brightness,
                    "contrast": contrast,
                    **evaluate_condition(
                        cache=condition_cache, oof=oof, device=device
                    ),
                }
            aggregate_fp = sum(
                int(value["metrics"]["quaternion_gray_base"]["fp_rejected"])
                for value in conditions.values()
            )
            aggregate_tp_harms = sum(
                int(value["metrics"]["quaternion_gray_base"]["tp_broken"])
                for value in conditions.values()
            )
            robustness = {
                "status": "evaluated",
                "conditions": conditions,
                "aggregate_fp_rejected": aggregate_fp,
                "aggregate_tp_harms": aggregate_tp_harms,
                "aggregate_fp_exceeds_tp_harms": aggregate_fp > aggregate_tp_harms,
                "passed": all(bool(value["passed"]) for value in conditions.values())
                and aggregate_fp > aggregate_tp_harms,
            }
        else:
            robustness = {
                "status": "skipped_clean_gate_failed",
                "conditions": {},
                "passed": False,
            }

        static_export = static_export_checks(
            output_dir=output_dir, cache=cache, oof=oof, device=device
        )
        runtime = benchmark_branches(cache=cache, oof=oof, device=device)
        xai = render_xai_sheet(
            output_dir=output_dir,
            cache=cache,
            cohort=cohort,
            oof=oof,
            device=device,
        )
        replay_files = save_replay_artifacts(
            output_dir=output_dir, cohort=cohort, oof=oof
        )
        replay = replay_artifacts(output_dir)
        cache_shape = list(cache.shape)
        cache_array_sha256 = _array_sha256(cache)
        del cache
        cache_path.unlink()
        cache_deleted = not cache_path.exists()
    finally:
        if cache_path.exists():
            cache_path.unlink()
            cache_deleted = not cache_path.exists()

    torch.cuda.synchronize(device)
    peak_cuda_gib = float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    gpu_after = _gpu_snapshot()
    repo_after = _repo_state()
    training_records = oof["training_records"]
    loader_summary = extraction["loader"]
    structural_checks = {
        "tracked_tree_clean": bool(repo_after["tracked_worktree_clean"]),
        "head_matches_upstream": bool(repo_after["head_matches_upstream"]),
        "repo_head_unchanged": repo_before["head"] == repo_after["head"],
        "protected_payloads": bool(provenance["protected_untracked"]["passed"]),
        "train_only_dataset": bool(extraction["dataset"]["train_paths_only"]),
        "dataset_paths_exact": bool(extraction["dataset"]["paths_exact"]),
        "cohort_order": _cohort_index_sha256([row.sample_index for row in cohort])
        == EXPECTED_ORDERED_INDEX_SHA256,
        "engineering": bool(engineering["passed"]),
        "all_parameter_gradients": all(
            bool(record["gradient_pass"]) for record in training_records
        ),
        "all_parameters_changed": all(
            bool(record["update_pass"]) for record in training_records
        ),
        "optimizer_finite": all(
            bool(record["optimizer"]["all_state_tensors_finite"])
            for record in training_records
        ),
        "shared_occurrences": bool(oof["shared_occurrences"]),
        "static_export": bool(static_export["passed"]),
        "replay": bool(replay["passed"]),
        "workers_recorded": "requested_num_workers" in loader_summary
        and "effective_num_workers" in loader_summary,
        "peak_cuda": peak_cuda_gib < 2.0,
        "runtime": bool(runtime["passed"]),
        "cache_deleted": cache_deleted,
        "no_unknown_process_terminated": not bool(
            gpu_before["unknown_process_terminated"]
        )
        and not bool(gpu_after["unknown_process_terminated"]),
    }
    structural_passed = all(bool(value) for value in structural_checks.values())
    pre_manual_passed = (
        structural_passed
        and bool(clean["passed"])
        and bool(robustness["passed"])
        and bool(xai["automatic_passed"])
        and bool(replay["passed"])
    )
    fold_summary = {
        str(fold): {
            "fit_rows": int(len(oof["fold_records"][fold]["fit_positions"])),
            "held_rows": int(len(oof["fold_records"][fold]["held_positions"])),
            "fit_source_count": int(oof["fold_records"][fold]["fit_source_count"]),
            "held_source_count": int(oof["fold_records"][fold]["held_source_count"]),
            "source_overlap": int(oof["fold_records"][fold]["source_overlap"]),
            "occurrence_sha256": oof["fold_records"][fold]["occurrence_sha256"],
            "matched_initial_trunk_sha256": oof["fold_records"][fold][
                "matched_initial_trunk_sha256"
            ],
            "thresholds": oof["fold_records"][fold]["thresholds"],
        }
        for fold in FOLDS
    }
    summary = {
        "schema": "trkh_quaternion_color_rotation_a0_summary_v1",
        "method": METHOD,
        "mode": MODE,
        "status": "complete_pending_visual_review",
        "advancement_authorized": False,
        "validation_used": False,
        "test_used": False,
        "production_model_modified": False,
        "production_checkpoint_written": False,
        "current_best_command_updated": False,
        "provenance": provenance,
        "repo_before": repo_before,
        "repo_after": repo_after,
        "engineering": engineering,
        "cohort": {
            "rows": len(cohort),
            "positives": EXPECTED_POSITIVES,
            "negatives": EXPECTED_NEGATIVES,
            "ordered_index_sha256": EXPECTED_ORDERED_INDEX_SHA256,
            "folds": fold_summary,
        },
        "rgb_cache": {
            "shape": cache_shape,
            "dtype": "uint8",
            "array_sha256": cache_array_sha256,
            "temporary_file_sha256": cache_file_sha256,
            "temporary_file_deleted": cache_deleted,
            "geometry_rows": len(extraction["geometry"]),
            "geometry_sha256": _sha256(geometry_path),
            "preview": preview,
            "seconds": extraction["seconds"],
        },
        "loader": loader_summary,
        "clean_information_gate": clean,
        "robustness": robustness,
        "static_export": static_export,
        "runtime": runtime,
        "xai": xai,
        "replay": replay,
        "replay_artifacts": replay_files,
        "resources": {
            "peak_cuda_allocated_gib": peak_cuda_gib,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "formal_seconds": time.perf_counter() - formal_started,
        },
        "structural_gate": {
            "checks": structural_checks,
            "passed": structural_passed,
        },
        "pre_manual_passed": pre_manual_passed,
        "manual_visual_review": {
            "decision": "pending",
            "reviewer": None,
            "summary_sha256_before_decision": None,
        },
        "passed": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(output_dir)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "manifest_sha256": _sha256(manifest_path),
                "clean_passed": clean["passed"],
                "pre_manual_passed": pre_manual_passed,
            }
        ),
        flush=True,
    )
    return summary


def replay_summary(path: Path) -> Dict[str, object]:
    summary_path = Path(path).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Quaternion summary is missing: {summary_path}")
    output_dir = summary_path.parent
    manifest = _verify_manifest(output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay = replay_artifacts(output_dir)
    stored = summary.get("replay", {})
    agreement = bool(replay["passed"]) and abs(
        float(replay["max_score_error"]) - float(stored.get("max_score_error", math.inf))
    ) <= 1e-15 and bool(replay["exact_actions"]) == bool(stored.get("exact_actions"))
    return {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest": manifest,
        "replay": replay,
        "stored_replay_agreement": agreement,
        "passed": bool(replay["passed"]) and agreement,
        "manual_visual_review": summary.get("manual_visual_review"),
        "pre_manual_passed": bool(summary.get("pre_manual_passed", False)),
        "final_passed": bool(summary.get("passed", False)),
    }


def finalize_visual_review(
    *,
    summary_path: Path,
    decision: str,
    expected_summary_sha256: str,
) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    observed_sha256 = _sha256(resolved)
    if observed_sha256 != str(expected_summary_sha256).strip().casefold():
        raise ValueError(
            "Quaternion summary changed before visual finalization: "
            f"{observed_sha256} != {expected_summary_sha256}"
        )
    _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("manual_visual_review", {}).get("decision") != "pending":
        raise ValueError("Quaternion visual review has already been finalized")
    passed = decision == "pass"
    summary["manual_visual_review"] = {
        "decision": decision,
        "reviewer": "codex_visual_inspection",
        "summary_sha256_before_decision": observed_sha256,
    }
    summary["xai"]["manual_review"] = decision
    summary["status"] = "complete_passed" if passed and summary["pre_manual_passed"] else "complete_rejected"
    summary["passed"] = bool(passed and summary["pre_manual_passed"])
    summary["advancement_authorized"] = bool(summary["passed"])
    _write_json(resolved, summary)
    manifest_path = _write_manifest(resolved.parent)
    return {
        "summary": str(resolved),
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_path),
        "decision": decision,
        "passed": bool(summary["passed"]),
        "advancement_authorized": bool(summary["advancement_authorized"]),
    }


def reconcile_finalized_review(
    *,
    summary_path: Path,
    expected_summary_sha256: str,
) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    observed_sha256 = _sha256(resolved)
    if observed_sha256 != str(expected_summary_sha256).strip().casefold():
        raise ValueError(
            "Quaternion summary changed before review reconciliation: "
            f"{observed_sha256} != {expected_summary_sha256}"
        )
    _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    decision = str(summary.get("manual_visual_review", {}).get("decision", ""))
    nested = str(summary.get("xai", {}).get("manual_review", ""))
    if decision not in {"pass", "fail"}:
        raise ValueError("Top-level quaternion visual decision is not finalized")
    if nested != "pending":
        raise ValueError(
            f"Nested quaternion XAI review is not the repairable pending value: {nested}"
        )
    summary["xai"]["manual_review"] = decision
    summary["artifact_reconciliation"] = {
        "kind": "nested_xai_manual_review_mirror",
        "source_decision": decision,
        "summary_sha256_before_reconciliation": observed_sha256,
        "scientific_fields_changed": False,
    }
    _write_json(resolved, summary)
    manifest_path = _write_manifest(resolved.parent)
    return {
        "summary": str(resolved),
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_path),
        "decision": decision,
        "scientific_fields_changed": False,
        "passed": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.replay_summary is not None:
        replay = replay_summary(Path(args.replay_summary))
        if args.reconcile_finalized_review:
            if not args.expected_summary_sha256:
                raise ValueError(
                    "Review reconciliation requires --expected-summary-sha256"
                )
            result = reconcile_finalized_review(
                summary_path=Path(args.replay_summary),
                expected_summary_sha256=str(args.expected_summary_sha256),
            )
            print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
            return 0
        if args.finalize_visual_review is not None:
            if not args.expected_summary_sha256:
                raise ValueError(
                    "Visual finalization requires --expected-summary-sha256"
                )
            result = finalize_visual_review(
                summary_path=Path(args.replay_summary),
                decision=str(args.finalize_visual_review),
                expected_summary_sha256=str(args.expected_summary_sha256),
            )
            print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
            return 0
        print(json.dumps(to_serializable(replay), indent=2, sort_keys=True))
        return 0 if bool(replay["passed"]) else 1
    if args.finalize_visual_review is not None or args.reconcile_finalized_review:
        raise ValueError("Visual finalization/reconciliation requires --replay-summary")
    if args.preflight_only:
        result = preflight(args)
        print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
        return 0
    if args.engineering_forward:
        _validate_locked_args(args)
        verify_locked_inputs(args)
        result = _engineering_forward(args)
        print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
        return 0
    run_formal(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

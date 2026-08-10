from __future__ import annotations

"""Prospectively locked train-only DeepBDC stem information audit.

The Brownian distance covariance implementation in this module is derived
independently from Xie et al., CVPR 2022, equations (4)-(6).  It does not copy
or import the authors' CC BY-NC reference implementation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.config import to_serializable
from trkh.core.utils import set_seed
from trkh.models.model import classification_logits_from_features
from trkh.tools.audit_hamburger_nmf_surface_a0 import (
    _build_dataset,
    _cohort_from_rows,
    _global_rng_equal,
    _global_rng_snapshot,
    _make_condition_loader,
    _metadata_tensor,
    _model_state_sha256,
    _prepare_output_dir,
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
from trkh.inference.inference import load_model


METHOD = "deepbdc_stem_joint_dependence_a0"
MODE = "deepbdc_stem_joint_dependence_a0_train_information_gate"
SEED = 20260721
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FOCUS_CLASS = 1
FOLDS = (0, 1, 2, 3, 4)
TRAINED_ROLES = (
    "base_logprob",
    "mean32_base",
    "cov32_base",
    "bdc32_dephased_base",
    "bdc32_aligned_base",
    "bdc32_aligned_only",
)
SAME_WEIGHT_ROLE = "bdc32_same_weight_dephased"
ROLE_NAMES = TRAINED_ROLES + (SAME_WEIGHT_ROLE,)
MATCHED_CONTROL_ROLES = (
    "base_logprob",
    "mean32_base",
    "cov32_base",
    "bdc32_dephased_base",
    SAME_WEIGHT_ROLE,
)
PROJECTION_INPUT_DIM = 256
PROJECTION_DIM = 32
SPATIAL_SIZE = 16
SPATIAL_TOKENS = SPATIAL_SIZE * SPATIAL_SIZE
TRIANGLE_DIM = PROJECTION_DIM * (PROJECTION_DIM + 1) // 2
LOG_PROBABILITY_CLIP = 1e-8
BDC_EPSILON = 1e-5
MIN_FIT_TP_RETENTION = 0.97
MAX_CIDT_PROBABILITY_ERROR = 3e-5
MAX_CAPTURE_PROBABILITY_ERROR = 1e-6
MAX_REPLAY_SCORE_ERROR = 1e-7
MAX_PEAK_CUDA_GIB = 3.5

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "4148b61ecf40168a82796030dcb9fd4648e5312fe60e6ea21c3e27c47837f93d"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "cdb4a73cf396aaa4be0eeb5414ce5f2971512278eb43a86b2ae9fe3243bc7654"
LOCKED_OFFICIAL_COMMIT = "adfab39f121ec1c17502306eca9e134244f7a83a"
LOCKED_OFFICIAL_TREE = "058e8d08cc7cc1c3af8805be061d15c202882cec"
LOCKED_OFFICIAL_LICENSE_SHA256 = "2dd05518b1e6ee64b8f319a0d8ecfa55e02b6261dcabaea2c6ccf979b4bf7ef5"
LOCKED_OFFICIAL_BDC_SHA256 = "27ef7cb7752ff085b3858945c89818383f12a4ce25cbe0e208f38891fab30bf6"
LOCKED_OFFICIAL_STL_SHA256 = "258602a39f50e94518eee8b8e623d09666322d43e92485065310b42d9f12b37b"
LOCKED_OFFICIAL_README_SHA256 = "ed4025410d52ca745abb6393fe4153349963e9a2faeedf45a071fc5bd5d0e38d"

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\deepbdc-cvpr2022")
PAPER_PATH = OFFICIAL_ROOT / "Xie_DeepBDC_CVPR_2022.pdf"
CONDITIONS = (
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only DeepBDC stem joint-dependence gate."
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
        / "TRKH_5CLASS_DEEPBDC_STEM_JOINT_DEPENDENCE_A0_PROTOCOL_20260721.md",
    )
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_deepbdc_stem_joint_dependence_a0_20260721",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--engineering-forward", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", choices=("pass", "fail"))
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if str(args.device) != "cuda":
        raise ValueError("DeepBDC A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"DeepBDC A0 locks batch-size={BATCH_SIZE}, num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"DeepBDC A0 locks seed={SEED}")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def verify_locked_inputs(args: argparse.Namespace) -> Dict[str, object]:
    official_root = Path(args.official_root).expanduser().resolve()
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
        "protocol": _verify_hash(args.protocol, LOCKED_PROTOCOL_SHA256, "DeepBDC protocol"),
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
        "paper": _verify_hash(PAPER_PATH, LOCKED_PAPER_SHA256, "DeepBDC paper"),
        "official_license": _verify_hash(
            official_root / "LICENSE.txt",
            LOCKED_OFFICIAL_LICENSE_SHA256,
            "official DeepBDC license",
        ),
        "official_bdc_source": _verify_hash(
            official_root / "methods" / "bdc_module.py",
            LOCKED_OFFICIAL_BDC_SHA256,
            "official DeepBDC module",
        ),
        "official_stl_source": _verify_hash(
            official_root / "methods" / "stl_deepbdc.py",
            LOCKED_OFFICIAL_STL_SHA256,
            "official DeepBDC supervised head",
        ),
        "official_readme": _verify_hash(
            official_root / "README.md",
            LOCKED_OFFICIAL_README_SHA256,
            "official DeepBDC README",
        ),
    }
    commit = _git_value(official_root, "rev-parse", "HEAD")
    tree = _git_value(official_root, "rev-parse", "HEAD^{tree}")
    tracked_status = _git_value(
        official_root, "status", "--porcelain", "--untracked-files=no"
    )
    if (
        commit != LOCKED_OFFICIAL_COMMIT
        or tree != LOCKED_OFFICIAL_TREE
        or tracked_status
    ):
        raise ValueError(
            "Official DeepBDC repository differs from lock: "
            f"commit={commit}, tree={tree}, tracked_status={tracked_status!r}"
        )
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payloads differ from lock: {protected}")
    return {
        "files": files,
        "official_repository": {
            "path": str(official_root),
            "remote": _git_value(official_root, "remote", "get-url", "origin"),
            "commit": commit,
            "tree": tree,
            "tracked_worktree_clean": tracked_status == "",
            "paper_is_intentionally_untracked": True,
        },
        "installed_runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "protected_untracked": protected,
    }


def bdc_matrix_torch(features: Tensor, tau: Tensor | float) -> Tensor:
    """Return the double-centered channel-distance matrix from paper Eq. 4-6."""
    if features.ndim != 3:
        raise ValueError("BDC features must have shape [B,d,n]")
    if not features.is_floating_point() or features.shape[1] < 2 or features.shape[2] < 2:
        raise ValueError("BDC features must be floating [B,d>=2,n>=2]")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("BDC features must be finite")
    scale = torch.exp(torch.as_tensor(tau, device=features.device, dtype=features.dtype))
    gram = torch.bmm(features, features.transpose(1, 2))
    diagonal = torch.diagonal(gram, dim1=1, dim2=2)
    squared = (
        diagonal.unsqueeze(2) + diagonal.unsqueeze(1) - 2.0 * gram
    ).clamp_min(0.0)
    distance = torch.sqrt(scale * squared + BDC_EPSILON)
    return (
        distance
        - distance.mean(dim=2, keepdim=True)
        - distance.mean(dim=1, keepdim=True)
        + distance.mean(dim=(1, 2), keepdim=True)
    )


def bdc_descriptor_torch(features: Tensor, tau: Tensor | float) -> Tensor:
    matrix = bdc_matrix_torch(features, tau)
    indices = torch.triu_indices(
        matrix.shape[1], matrix.shape[2], device=matrix.device
    )
    descriptor = matrix[:, indices[0], indices[1]]
    return F.normalize(descriptor, p=2.0, dim=1, eps=1e-12)


def bdc_matrix_numpy(features: np.ndarray, tau: float) -> np.ndarray:
    """Independent FP64 oracle using explicit pairwise channel differences."""
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] < 2 or value.shape[2] < 2:
        raise ValueError("BDC oracle features must have shape [B,d>=2,n>=2]")
    if not np.isfinite(value).all():
        raise ValueError("BDC oracle features must be finite")
    difference = value[:, :, None, :] - value[:, None, :, :]
    squared = np.maximum(np.sum(np.square(difference), axis=-1), 0.0)
    distance = np.sqrt(math.exp(float(tau)) * squared + BDC_EPSILON)
    return (
        distance
        - distance.mean(axis=2, keepdims=True)
        - distance.mean(axis=1, keepdims=True)
        + distance.mean(axis=(1, 2), keepdims=True)
    )


def bdc_descriptor_numpy(features: np.ndarray, tau: float) -> np.ndarray:
    matrix = bdc_matrix_numpy(features, tau)
    indices = np.triu_indices(matrix.shape[1])
    descriptor = matrix[:, indices[0], indices[1]]
    norm = np.linalg.norm(descriptor, axis=1, keepdims=True)
    return descriptor / np.maximum(norm, 1e-12)


def covariance_descriptor_torch(features: Tensor) -> Tensor:
    if features.ndim != 3 or features.shape[1] != PROJECTION_DIM:
        raise ValueError("Covariance features must have shape [B,32,n]")
    centered = features - features.mean(dim=2, keepdim=True)
    covariance = torch.bmm(centered, centered.transpose(1, 2)) / max(
        int(features.shape[2]) - 1, 1
    )
    indices = torch.triu_indices(
        PROJECTION_DIM, PROJECTION_DIM, device=features.device
    )
    descriptor = covariance[:, indices[0], indices[1]]
    descriptor = torch.sign(descriptor) * torch.sqrt(descriptor.abs() + 1e-12)
    return F.normalize(descriptor, p=2.0, dim=1, eps=1e-12)


def dephase_permutations(
    sample_indices: Tensor | np.ndarray | Sequence[int],
    *,
    channels: int = PROJECTION_DIM,
    tokens: int = SPATIAL_TOKENS,
) -> Tensor:
    indices = [int(value) for value in torch.as_tensor(sample_indices).reshape(-1).tolist()]
    output = torch.empty((len(indices), int(channels), int(tokens)), dtype=torch.long)
    for row, sample_index in enumerate(indices):
        for channel in range(int(channels)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(SEED + 1_000_003 * sample_index + 1009 * channel)
            output[row, channel] = torch.randperm(int(tokens), generator=generator)
    return output


def dephase_projected(features: Tensor, sample_indices: Tensor) -> Tensor:
    if features.ndim != 3 or features.shape[1:] != (
        PROJECTION_DIM,
        SPATIAL_TOKENS,
    ):
        raise ValueError("Dephasing expects projected features [B,32,256]")
    permutations = dephase_permutations(sample_indices).to(features.device)
    return torch.gather(features, 2, permutations)


def _descriptor_dimension(role: str) -> int:
    if role == "base_logprob":
        return 0
    if role == "mean32_base":
        return PROJECTION_DIM
    if role in {
        "cov32_base",
        "bdc32_dephased_base",
        "bdc32_aligned_base",
        "bdc32_aligned_only",
    }:
        return TRIANGLE_DIM
    raise ValueError(f"Unknown DeepBDC role: {role}")


class StemJointHead(nn.Module):
    def __init__(
        self,
        role: str,
        *,
        base_mean: np.ndarray,
        base_std: np.ndarray,
    ) -> None:
        super().__init__()
        if role not in TRAINED_ROLES:
            raise ValueError(f"Unknown trained role: {role}")
        self.role = str(role)
        self.register_buffer(
            "base_mean",
            torch.as_tensor(base_mean, dtype=torch.float32).reshape(1, 5),
        )
        self.register_buffer(
            "base_std",
            torch.as_tensor(base_std, dtype=torch.float32).reshape(1, 5),
        )
        if role != "base_logprob":
            self.projection = nn.Sequential(
                nn.Conv2d(PROJECTION_INPUT_DIM, PROJECTION_DIM, 1, bias=False),
                nn.BatchNorm2d(PROJECTION_DIM),
                nn.ReLU(inplace=False),
            )
        else:
            self.projection = None
        if role.startswith("bdc32_"):
            self.tau = nn.Parameter(
                torch.tensor(math.log(1.0 / (2.0 * SPATIAL_TOKENS)), dtype=torch.float32)
            )
        else:
            self.register_parameter("tau", None)
        base_dim = 0 if role == "bdc32_aligned_only" else 5
        self.classifier = nn.Linear(_descriptor_dimension(role) + base_dim, 1)

    def standardized_log_probabilities(self, probabilities: Tensor) -> Tensor:
        clipped = probabilities.clamp(LOG_PROBABILITY_CLIP, 1.0)
        return (clipped.log() - self.base_mean) / self.base_std

    def feature_vector(
        self,
        stem_maps: Tensor,
        probabilities: Tensor,
        sample_indices: Tensor,
        *,
        force_dephase: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        base = self.standardized_log_probabilities(probabilities)
        if self.projection is None:
            return base, None
        projected_map = self.projection(stem_maps)
        projected = projected_map.flatten(2)
        dephased = force_dephase or self.role == "bdc32_dephased_base"
        if dephased:
            projected = dephase_projected(projected, sample_indices)
        if self.role == "mean32_base":
            descriptor = projected.mean(dim=2)
        elif self.role == "cov32_base":
            descriptor = covariance_descriptor_torch(projected)
        else:
            if self.tau is None:
                raise RuntimeError("BDC role lacks trainable tau")
            descriptor = bdc_descriptor_torch(projected, self.tau)
        if self.role == "bdc32_aligned_only":
            return descriptor, projected_map
        return torch.cat((descriptor, base), dim=1), projected_map

    def forward(
        self,
        stem_maps: Tensor,
        probabilities: Tensor,
        sample_indices: Tensor,
        *,
        force_dephase: bool = False,
    ) -> Tensor:
        features, _ = self.feature_vector(
            stem_maps,
            probabilities,
            sample_indices,
            force_dephase=force_dephase,
        )
        return self.classifier(features).squeeze(1)


def make_matched_heads(
    *,
    fold: int,
    base_mean: np.ndarray,
    base_std: np.ndarray,
) -> Dict[str, StemJointHead]:
    heads: Dict[str, StemJointHead] = {}
    before = _global_rng_snapshot()
    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for role in TRAINED_ROLES:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(SEED + 104729 * int(fold))
            heads[role] = StemJointHead(
                role, base_mean=base_mean, base_std=base_std
            )
    after = _global_rng_snapshot()
    if not _global_rng_equal(before, after):
        raise RuntimeError("Matched head construction changed global RNG state")
    projection_states = [
        head.projection.state_dict()
        for head in heads.values()
        if head.projection is not None
    ]
    reference = projection_states[0]
    if not all(
        set(state) == set(reference)
        and all(torch.equal(state[key], reference[key]) for key in reference)
        for state in projection_states[1:]
    ):
        raise RuntimeError("Matched DeepBDC roles do not share exact projection initialization")
    return heads


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
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


def _effective_rank(value: np.ndarray) -> float:
    matrix = np.asarray(value, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    distribution = energy / max(float(energy.sum()), 1e-30)
    positive = distribution > 0.0
    return float(np.exp(-np.sum(distribution[positive] * np.log(distribution[positive]))))


def engineering_checks() -> Dict[str, object]:
    generator = np.random.default_rng(73)
    value = generator.normal(size=(2, 7, 11)).astype(np.float64)
    tau = math.log(1.0 / 22.0)
    oracle = bdc_matrix_numpy(value, tau)
    torch64 = bdc_matrix_torch(torch.from_numpy(value), tau).numpy()
    descriptor_oracle = bdc_descriptor_numpy(value, tau)
    descriptor64 = bdc_descriptor_torch(torch.from_numpy(value), tau).numpy()
    matrix_error = float(np.max(np.abs(oracle - torch64)))
    descriptor_error = float(np.max(np.abs(descriptor_oracle - descriptor64)))

    torch32 = bdc_descriptor_torch(torch.from_numpy(value.astype(np.float32)), tau)
    fp32_error = float(np.max(np.abs(descriptor_oracle - torch32.numpy())))
    bf16 = bdc_descriptor_torch(
        torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16), tau
    ).float()
    bf16_error = float(np.max(np.abs(descriptor_oracle - bf16.numpy())))

    translation = generator.normal(size=(2, 1, 11))
    translated = bdc_descriptor_numpy(value + translation, tau)
    translation_error = float(np.max(np.abs(descriptor_oracle - translated)))
    q, _ = np.linalg.qr(generator.normal(size=(11, 11)))
    rotated = bdc_descriptor_numpy(value @ q, tau)
    orthonormal_error = float(np.max(np.abs(descriptor_oracle - rotated)))
    scaled = bdc_descriptor_numpy(value * 2.5, tau)
    scale_error = float(np.max(np.abs(descriptor_oracle - scaled)))
    symmetry_error = float(np.max(np.abs(torch64 - np.swapaxes(torch64, 1, 2))))

    grad_value = torch.from_numpy(value[:1, :4, :6]).clone().requires_grad_(True)
    grad_tau = torch.tensor(math.log(1.0 / 12.0), dtype=torch.float64, requires_grad=True)
    weights = torch.linspace(0.1, 1.0, 10, dtype=torch.float64).reshape(1, -1)
    objective = (bdc_descriptor_torch(grad_value, grad_tau) * weights).sum()
    gradient = torch.autograd.grad(objective, (grad_value, grad_tau), create_graph=False)
    epsilon = 1e-6
    plus = grad_value.detach().clone()
    minus = grad_value.detach().clone()
    plus[0, 1, 2] += epsilon
    minus[0, 1, 2] -= epsilon
    finite_difference = float(
        (
            (bdc_descriptor_torch(plus, grad_tau.detach()) * weights).sum()
            - (bdc_descriptor_torch(minus, grad_tau.detach()) * weights).sum()
        )
        / (2.0 * epsilon)
    )
    gradient_error = abs(float(gradient[0][0, 1, 2]) - finite_difference)

    singleton = bdc_descriptor_torch(torch.from_numpy(value[:1]), tau)
    rng_before = _global_rng_snapshot()
    permutations_a = dephase_permutations([7, 11], channels=4, tokens=9)
    permutations_b = dephase_permutations([7, 11], channels=4, tokens=9)
    rng_after = _global_rng_snapshot()
    invalid_rejected = False
    try:
        bdc_matrix_torch(torch.ones(2, 7), tau)
    except ValueError:
        invalid_rejected = True
    heads = make_matched_heads(
        fold=0,
        base_mean=np.zeros(5, dtype=np.float32),
        base_std=np.ones(5, dtype=np.float32),
    )
    projection_hashes = {
        role: _parameter_sha256(head.projection)
        for role, head in heads.items()
        if head.projection is not None
    }
    checks = {
        "numpy_fp64_matrix_error_le_1e_11": matrix_error <= 1e-11,
        "numpy_fp64_descriptor_error_le_1e_11": descriptor_error <= 1e-11,
        "fp32_descriptor_error_le_2e_5": fp32_error <= 2e-5,
        "bf16_descriptor_error_le_2e_2": bf16_error <= 2e-2,
        "symmetric_le_1e_12": symmetry_error <= 1e-12,
        "translation_invariant_le_1e_11": translation_error <= 1e-11,
        "orthonormal_invariant_le_1e_11": orthonormal_error <= 1e-11,
        # The fixed epsilon deliberately prevents exact scale invariance near
        # zero channel distances; this bound verifies the normalized limit.
        "positive_scale_invariant_le_5e_4": scale_error <= 5e-4,
        "finite_difference_gradient_le_2e_5": gradient_error <= 2e-5,
        "all_gradients_finite_nonzero": bool(
            torch.isfinite(gradient[0]).all()
            and torch.isfinite(gradient[1])
            and float(gradient[0].abs().max()) > 0.0
            and float(gradient[1].abs()) > 0.0
        ),
        "singleton_batch_preserved": tuple(singleton.shape) == (1, 28),
        "invalid_input_rejected": invalid_rejected,
        "dephasing_deterministic": torch.equal(permutations_a, permutations_b),
        "dephasing_local_rng_only": _global_rng_equal(rng_before, rng_after),
        "projection_initialization_exact": len(set(projection_hashes.values())) == 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": {
            "matrix_fp64": matrix_error,
            "descriptor_fp64": descriptor_error,
            "descriptor_fp32": fp32_error,
            "descriptor_bf16": bf16_error,
            "symmetry": symmetry_error,
            "translation": translation_error,
            "orthonormal": orthonormal_error,
            "scale": scale_error,
            "finite_difference_gradient": gradient_error,
        },
        "projection_hashes": projection_hashes,
    }


def crop_valid_stem_maps(
    stem_maps: Tensor,
    image_valid_mask: Tensor,
) -> Tuple[Tensor, List[Dict[str, object]]]:
    if stem_maps.ndim != 4 or tuple(stem_maps.shape[1:]) != (256, 32, 32):
        raise ValueError("Captured keeper stem must have shape [B,256,32,32]")
    mask = image_valid_mask
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[0] != stem_maps.shape[0]:
        raise ValueError("Image-valid mask does not align with captured stem")
    downsampled = F.interpolate(
        mask.to(device=stem_maps.device, dtype=torch.float32),
        size=(32, 32),
        mode="area",
    )
    crops: List[Tensor] = []
    geometry: List[Dict[str, object]] = []
    for row in range(int(stem_maps.shape[0])):
        valid = downsampled[row, 0] >= 0.5
        coordinates = torch.nonzero(valid, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("Image-valid mask has no valid stem location")
        y0 = int(coordinates[:, 0].min())
        y1 = int(coordinates[:, 0].max()) + 1
        x0 = int(coordinates[:, 1].min())
        x1 = int(coordinates[:, 1].max()) + 1
        crop = stem_maps[row : row + 1, :, y0:y1, x0:x1].float()
        resized = F.interpolate(
            crop,
            size=(SPATIAL_SIZE, SPATIAL_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        crops.append(resized)
        geometry.append(
            {
                "stem_x0": x0,
                "stem_y0": y0,
                "stem_x1_exclusive": x1,
                "stem_y1_exclusive": y1,
                "stem_width": x1 - x0,
                "stem_height": y1 - y0,
                "valid_fraction": float(valid.float().mean()),
                "rectangle_valid_fraction": float(valid[y0:y1, x0:x1].float().mean()),
            }
        )
    return torch.cat(crops, dim=0), geometry


def _captured_keeper_forward(
    model: nn.Module,
    images: Tensor,
    image_mask: Optional[Tensor],
    bbox: Tensor,
) -> Tuple[Tensor, Tensor]:
    captured: List[Tensor] = []

    def capture(_module: nn.Module, _inputs: Tuple[Tensor, ...], output: object) -> None:
        if not torch.is_tensor(output):
            raise TypeError("HybridConvStem hook emitted a non-tensor output")
        captured.append(output)

    hook_count_before = len(model.stem._forward_hooks)  # type: ignore[attr-defined]
    handle = model.stem.register_forward_hook(capture)  # type: ignore[attr-defined]
    try:
        features = model.forward_features(  # type: ignore[attr-defined]
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=bbox,
        )
        features["bbox"] = bbox[:, :4]
        logits = classification_logits_from_features(model, features).float()
    finally:
        handle.remove()
    hook_count_after = len(model.stem._forward_hooks)  # type: ignore[attr-defined]
    if hook_count_after != hook_count_before or len(captured) != 1:
        raise RuntimeError("Keeper stem hook was not invoked and removed exactly once")
    stem = captured[0]
    if stem.shape != (images.shape[0], 256, 32, 32):
        raise ValueError(f"Keeper stem shape differs from lock: {tuple(stem.shape)}")
    return logits, stem


def _ordinary_keeper_probabilities(
    model: nn.Module,
    images: Tensor,
    image_mask: Optional[Tensor],
    bbox: Tensor,
) -> Tensor:
    features = model.forward_features(  # type: ignore[attr-defined]
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
    )
    features["bbox"] = bbox[:, :4]
    return classification_logits_from_features(model, features).float().softmax(dim=1)


def extract_condition(
    *,
    model: nn.Module,
    loader: DataLoader,
    loader_summary: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    semantics: Mapping[str, object],
    condition: str,
    benchmark_ordinary: bool,
) -> Dict[str, object]:
    rows = len(cohort)
    maps = np.empty(
        (rows, PROJECTION_INPUT_DIM, SPATIAL_SIZE, SPATIAL_SIZE),
        dtype=np.float16,
    )
    probabilities = np.empty((rows, 5), dtype=np.float32)
    targets = np.empty(rows, dtype=np.int64)
    sample_indices = np.empty(rows, dtype=np.int64)
    rgb_bank = np.empty((rows, 256, 256, 3), dtype=np.uint8)
    geometry_rows: List[Dict[str, object]] = []
    model.eval()
    state_before = _model_state_sha256(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    ordinary_seconds = 0.0
    captured_seconds = 0.0
    crop_seconds = 0.0
    capture_probability_error = 0.0
    cidt_probability_error = 0.0
    cidt_argmax_exact = True
    position = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, batch_targets, metadata in loader:
            if not isinstance(metadata, Mapping):
                raise ValueError("DeepBDC extraction requires tensor metadata")
            batch = int(batch_targets.numel())
            stop = position + batch
            observed_indices = metadata["sample_index"].detach().cpu().long()
            expected_rows = cohort[position:stop]
            if observed_indices.tolist() != [row.sample_index for row in expected_rows]:
                raise ValueError("DeepBDC loader changed the locked cohort order")
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            if image_mask is None:
                raise ValueError("DeepBDC extraction requires image_valid_mask")
            if image_mask.ndim == 3:
                image_mask = image_mask.unsqueeze(1)
            bbox = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)
            if bbox is None or bbox.ndim != 2 or bbox.shape[1] < 4:
                raise ValueError("DeepBDC extraction requires object bbox metadata")
            images_device = images.to(device=device, dtype=torch.float32, non_blocking=True)
            ordinary = None
            if benchmark_ordinary:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ordinary_started = time.perf_counter()
                ordinary = _ordinary_keeper_probabilities(
                    model, images_device, image_mask, bbox
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ordinary_seconds += time.perf_counter() - ordinary_started
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            capture_started = time.perf_counter()
            logits, stem = _captured_keeper_forward(
                model, images_device, image_mask, bbox
            )
            current_probabilities = logits.softmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            captured_seconds += time.perf_counter() - capture_started
            if ordinary is not None:
                capture_probability_error = max(
                    capture_probability_error,
                    float((ordinary - current_probabilities).abs().max()),
                )
            crop_started = time.perf_counter()
            cropped, batch_geometry = crop_valid_stem_maps(stem, image_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            crop_seconds += time.perf_counter() - crop_started
            maps[position:stop] = cropped.detach().cpu().to(torch.float16).numpy()
            probabilities[position:stop] = current_probabilities.detach().cpu().numpy()
            targets[position:stop] = batch_targets.detach().cpu().numpy()
            sample_indices[position:stop] = observed_indices.numpy()
            rgb = _to_rgb(images, semantics)
            rgb_bank[position:stop] = (
                rgb.permute(0, 2, 3, 1).numpy().clip(0.0, 1.0) * 255.0
            ).round().astype(np.uint8)
            expected_probabilities = np.asarray(
                [row.keeper_probabilities for row in expected_rows], dtype=np.float32
            )
            cidt_probability_error = max(
                cidt_probability_error,
                float(np.max(np.abs(probabilities[position:stop] - expected_probabilities))),
            )
            if condition in {"clean", "engineering_clean"}:
                cidt_argmax_exact = cidt_argmax_exact and bool(
                    np.array_equal(
                        probabilities[position:stop].argmax(axis=1),
                        np.asarray(
                            [row.keeper_prediction for row in expected_rows],
                            dtype=np.int64,
                        ),
                    )
                )
            for local, geometry in enumerate(batch_geometry):
                feature_map = maps[position + local]
                geometry_rows.append(
                    {
                        "position": position + local,
                        "sample_index": int(observed_indices[local]),
                        "target": int(batch_targets[local]),
                        "condition": condition,
                        **geometry,
                        "feature_sha256": _array_sha256(feature_map),
                    }
                )
            position = stop
            if position % 256 < batch or position == rows:
                print(
                    json.dumps(
                        {
                            "stage": "deepbdc_stem_extraction",
                            "condition": condition,
                            "processed": position,
                            "rows": rows,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if position != rows:
        raise RuntimeError("DeepBDC extraction did not cover the locked cohort")
    state_after = _model_state_sha256(model)
    expected_targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    expected_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    checks = {
        "sample_indices_exact": np.array_equal(sample_indices, expected_indices),
        "targets_exact": np.array_equal(targets, expected_targets),
        "capture_probability_error_le_1e_6": capture_probability_error
        <= MAX_CAPTURE_PROBABILITY_ERROR,
        "cidt_probability_error_le_3e_5": (
            cidt_probability_error <= MAX_CIDT_PROBABILITY_ERROR
            if condition in {"clean", "engineering_clean"}
            else True
        ),
        "cidt_argmax_exact": cidt_argmax_exact,
        "keeper_state_exact": state_before == state_after,
        "all_features_finite": bool(np.isfinite(maps).all()),
        "all_probabilities_finite": bool(np.isfinite(probabilities).all()),
    }
    return {
        "condition": condition,
        "maps": maps,
        "probabilities": probabilities,
        "targets": targets,
        "sample_indices": sample_indices,
        "rgb": rgb_bank,
        "geometry": geometry_rows,
        "runtime": {
            "loader": dict(loader_summary),
            "elapsed_seconds": float(time.perf_counter() - started),
            "ordinary_forward_seconds": ordinary_seconds,
            "captured_forward_seconds": captured_seconds,
            "valid_crop_seconds": crop_seconds,
            "peak_cuda_bytes": int(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
        },
        "checks": checks,
        "capture_probability_max_abs_error": capture_probability_error,
        "cidt_probability_max_abs_error": cidt_probability_error,
        "feature_cache": {
            "shape": list(maps.shape),
            "dtype": str(maps.dtype),
            "sha256": _array_sha256(maps),
            "temporary": True,
        },
    }


def _protocol_extraction_checks_passed(checks: Mapping[str, object]) -> bool:
    # CIDT was serialized under a different BF16 batch shape. Its probability
    # delta is retained as telemetry; the prospective protocol locks ordinary
    # versus hooked forward, argmax, state, geometry, and finite values.
    return all(
        bool(value)
        for key, value in checks.items()
        if key != "cidt_probability_error_le_3e_5"
    )


def _log_probability_statistics(
    probabilities: np.ndarray,
    fit_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.log(
        np.clip(np.asarray(probabilities, dtype=np.float64), LOG_PROBABILITY_CLIP, 1.0)
    )
    mean = values[fit_mask].mean(axis=0)
    std = values[fit_mask].std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError("Fit-only keeper log-probability standardizer is invalid")
    return mean.astype(np.float32), std.astype(np.float32)


def _shared_epoch_orders(
    fit_positions: np.ndarray,
    sample_indices: np.ndarray,
    *,
    held_fold: int,
    epochs: int = EPOCHS,
) -> Tuple[List[np.ndarray], str]:
    generator = np.random.default_rng(SEED + 65537 * int(held_fold))
    orders = [generator.permutation(fit_positions) for _ in range(int(epochs))]
    occurrence = np.concatenate([sample_indices[order] for order in orders]).astype(np.int64)
    return orders, _array_sha256(occurrence)


def _feature_scores(
    head: StemJointHead,
    maps: np.ndarray,
    probabilities: np.ndarray,
    sample_indices: np.ndarray,
    positions: np.ndarray,
    *,
    device: torch.device,
    force_dephase: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], float]:
    head.eval()
    feature_rows: List[np.ndarray] = []
    projected_sum = np.zeros(PROJECTION_DIM, dtype=np.float64)
    projected_square_sum = np.zeros(PROJECTION_DIM, dtype=np.float64)
    projected_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, int(positions.size), BATCH_SIZE):
            selected = positions[start : start + BATCH_SIZE]
            map_tensor = torch.from_numpy(maps[selected].astype(np.float32)).to(device)
            probability_tensor = torch.from_numpy(probabilities[selected]).to(device)
            index_tensor = torch.from_numpy(sample_indices[selected]).to(device)
            features, projected = head.feature_vector(
                map_tensor,
                probability_tensor,
                index_tensor,
                force_dephase=force_dephase,
            )
            feature_rows.append(features.detach().cpu().double().numpy())
            if projected is not None:
                projected_value = projected.detach().cpu().double().numpy()
                projected_sum += projected_value.sum(axis=(0, 2, 3))
                projected_square_sum += np.square(projected_value).sum(axis=(0, 2, 3))
                projected_count += int(projected_value.shape[0] * SPATIAL_TOKENS)
    elapsed = time.perf_counter() - started
    feature_matrix = np.concatenate(feature_rows, axis=0)
    weight = head.classifier.weight.detach().cpu().double().numpy().reshape(-1)
    bias = float(head.classifier.bias.detach().cpu().double().item())
    logits = feature_matrix @ weight + bias
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    statistics: Dict[str, np.ndarray] = {}
    if projected_count:
        mean = projected_sum / projected_count
        variance = projected_square_sum / projected_count - np.square(mean)
        statistics = {"mean": mean, "variance": variance}
    return scores, feature_matrix, statistics, elapsed


def _train_role(
    head: StemJointHead,
    *,
    maps: np.ndarray,
    probabilities: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    orders: Sequence[np.ndarray],
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    head.to(device)
    head.train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in head.named_parameters()
    }
    gradient_max = {name: 0.0 for name in initial_parameters}
    gradient_finite = {name: True for name in initial_parameters}
    history: List[Dict[str, object]] = []
    for epoch, order in enumerate(orders, start=1):
        losses: List[float] = []
        for start in range(0, int(order.size), BATCH_SIZE):
            selected = order[start : start + BATCH_SIZE]
            map_tensor = torch.from_numpy(maps[selected].astype(np.float32)).to(device)
            probability_tensor = torch.from_numpy(probabilities[selected]).to(device)
            index_tensor = torch.from_numpy(sample_indices[selected]).to(device)
            label_tensor = torch.from_numpy(labels[selected].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(map_tensor, probability_tensor, index_tensor)
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite DeepBDC loss for role {head.role}")
            loss.backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is None:
                    gradient_finite[name] = False
                    continue
                gradient_finite[name] = gradient_finite[name] and bool(
                    torch.isfinite(parameter.grad).all()
                )
                gradient_max[name] = max(
                    gradient_max[name], float(parameter.grad.detach().abs().max())
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "batches": len(losses),
            }
        )
    changed = {
        name: not torch.equal(initial_parameters[name], parameter.detach().cpu())
        for name, parameter in head.named_parameters()
    }
    optimizer_state_sha256, optimizer_state_summary = _optimizer_state_summary(
        optimizer, head
    )
    diagnostics = {
        "initial_parameter_sha256": _parameter_dict_sha256(initial_parameters),
        "final_parameter_sha256": _parameter_sha256(head),
        "gradient_max_abs": gradient_max,
        "gradient_finite": gradient_finite,
        "parameter_changed": changed,
        "all_parameters_receive_finite_nonzero_gradient": all(gradient_finite.values())
        and all(value > 0.0 for value in gradient_max.values()),
        "all_parameters_changed": all(changed.values()),
        "optimizer": "AdamW",
        "optimizer_state_sha256": optimizer_state_sha256,
        "optimizer_state": optimizer_state_summary,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": len(orders),
        "last_loss": history[-1]["loss"],
        "loss_finite": bool(np.isfinite([row["loss"] for row in history]).all()),
    }
    return history, diagnostics


def _parameter_dict_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_state_summary(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
) -> Tuple[str, Dict[str, object]]:
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    digest = hashlib.sha256()
    tensor_count = 0
    finite = True
    steps: List[float] = []
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names.get(id(item[0]), "")
    ):
        name = names.get(id(parameter))
        if name is None:
            raise RuntimeError("Optimizer state contains an unknown parameter")
        digest.update(name.encode("utf-8"))
        for key, value in sorted(state.items()):
            digest.update(str(key).encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                tensor_count += 1
                finite = finite and bool(torch.isfinite(tensor).all())
                if str(key) == "step":
                    steps.append(float(tensor.reshape(-1)[0]))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest(), {
        "parameters_with_state": len(optimizer.state),
        "state_tensor_count": tensor_count,
        "all_state_tensors_finite": finite,
        "minimum_step": min(steps) if steps else 0.0,
        "maximum_step": max(steps) if steps else 0.0,
    }


def _role_feature_dim(role: str) -> int:
    if role == SAME_WEIGHT_ROLE:
        role = "bdc32_aligned_base"
    base_dim = 0 if role == "bdc32_aligned_only" else 5
    return _descriptor_dimension(role) + base_dim


def fit_clean_oof_heads(
    *,
    maps: np.ndarray,
    probabilities: np.ndarray,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    epochs: int = EPOCHS,
) -> Dict[str, object]:
    rows = len(cohort)
    labels = np.asarray(
        [int(row.target == FOCUS_CLASS) for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    sample_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    sources = np.asarray([row.source_stem for row in cohort], dtype=str)
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    features = {
        role: np.full((rows, _role_feature_dim(role)), np.nan, dtype=np.float64)
        for role in ROLE_NAMES
    }
    fold_heads: Dict[int, Dict[str, StemJointHead]] = {}
    thresholds: Dict[int, Dict[str, float]] = {}
    histories: List[Dict[str, object]] = []
    diagnostics: Dict[int, Dict[str, object]] = {}
    occurrence_hashes: Dict[int, str] = {}
    candidate_head_seconds = 0.0
    projected_sum = np.zeros(PROJECTION_DIM, dtype=np.float64)
    projected_square_sum = np.zeros(PROJECTION_DIM, dtype=np.float64)
    projected_count = 0
    for held_fold in FOLDS:
        held = folds == int(held_fold)
        fit = ~held
        fit_positions = np.flatnonzero(fit)
        held_positions = np.flatnonzero(held)
        if set(sources[fit]).intersection(set(sources[held])):
            raise ValueError(f"DeepBDC fold {held_fold} has source leakage")
        if np.unique(labels[fit]).size != 2 or np.unique(labels[held]).size != 2:
            raise ValueError(f"DeepBDC fold {held_fold} lacks binary support")
        base_mean, base_std = _log_probability_statistics(probabilities, fit)
        heads = make_matched_heads(
            fold=int(held_fold), base_mean=base_mean, base_std=base_std
        )
        projection_hashes = {
            role: _parameter_sha256(head.projection)
            for role, head in heads.items()
            if head.projection is not None
        }
        orders, occurrence_hash = _shared_epoch_orders(
            fit_positions, sample_indices, held_fold=int(held_fold), epochs=int(epochs)
        )
        occurrence_hashes[int(held_fold)] = occurrence_hash
        thresholds[int(held_fold)] = {}
        fold_diagnostics: Dict[str, object] = {
            "fit_rows": int(fit.sum()),
            "held_rows": int(held.sum()),
            "fit_tp": int(labels[fit].sum()),
            "fit_fp": int(fit.sum() - labels[fit].sum()),
            "held_tp": int(labels[held].sum()),
            "held_fp": int(held.sum() - labels[held].sum()),
            "source_overlap": [],
            "occurrence_sha256": occurrence_hash,
            "projection_initial_hashes": projection_hashes,
            "projection_initialization_exact": len(set(projection_hashes.values())) == 1,
            "roles": {},
        }
        for role in TRAINED_ROLES:
            head = heads[role]
            role_history, role_diagnostics = _train_role(
                head,
                maps=maps,
                probabilities=probabilities,
                sample_indices=sample_indices,
                labels=labels,
                orders=orders,
                device=device,
            )
            fit_scores, _, _, _ = _feature_scores(
                head,
                maps,
                probabilities,
                sample_indices,
                fit_positions,
                device=device,
            )
            held_scores, held_features, projected_statistics, elapsed = _feature_scores(
                head,
                maps,
                probabilities,
                sample_indices,
                held_positions,
                device=device,
            )
            threshold = _positive_threshold(fit_scores, labels[fit])
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= threshold
            features[role][held] = held_features
            thresholds[int(held_fold)][role] = threshold
            for row in role_history:
                histories.append(
                    {"fold": int(held_fold), "role": role, **row}
                )
            role_diagnostics["threshold"] = threshold
            role_diagnostics["fit_tp_retention"] = float(
                (fit_scores[labels[fit] == 1] >= threshold).mean()
            )
            fold_diagnostics["roles"][role] = role_diagnostics
            if role == "bdc32_aligned_base":
                candidate_head_seconds += elapsed
                if projected_statistics:
                    count = int(held.sum() * SPATIAL_TOKENS)
                    mean = projected_statistics["mean"]
                    variance = projected_statistics["variance"]
                    projected_sum += mean * count
                    projected_square_sum += (variance + np.square(mean)) * count
                    projected_count += count
        candidate = heads["bdc32_aligned_base"]
        fit_scores, _, _, _ = _feature_scores(
            candidate,
            maps,
            probabilities,
            sample_indices,
            fit_positions,
            device=device,
            force_dephase=True,
        )
        held_scores, held_features, _, _ = _feature_scores(
            candidate,
            maps,
            probabilities,
            sample_indices,
            held_positions,
            device=device,
            force_dephase=True,
        )
        threshold = _positive_threshold(fit_scores, labels[fit])
        scores[SAME_WEIGHT_ROLE][held] = held_scores
        actions[SAME_WEIGHT_ROLE][held] = held_scores >= threshold
        features[SAME_WEIGHT_ROLE][held] = held_features
        thresholds[int(held_fold)][SAME_WEIGHT_ROLE] = threshold
        fold_diagnostics["roles"][SAME_WEIGHT_ROLE] = {
            "trained": False,
            "source_head": "bdc32_aligned_base",
            "force_dephase": True,
            "threshold": threshold,
            "fit_tp_retention": float(
                (fit_scores[labels[fit] == 1] >= threshold).mean()
            ),
        }
        for head in heads.values():
            head.to("cpu")
        fold_heads[int(held_fold)] = heads
        diagnostics[int(held_fold)] = fold_diagnostics
        print(
            json.dumps(
                {
                    "stage": "deepbdc_oof_training",
                    "held_fold": int(held_fold),
                    "roles_completed": len(TRAINED_ROLES),
                    "epochs": int(epochs),
                }
            ),
            flush=True,
        )
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("DeepBDC OOF scoring left non-finite rows")
    if not all(np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("DeepBDC OOF replay features left non-finite rows")
    if projected_count <= 0:
        raise RuntimeError("DeepBDC candidate projection statistics are empty")
    projected_mean = projected_sum / projected_count
    projected_variance = projected_square_sum / projected_count - np.square(projected_mean)
    return {
        "scores": scores,
        "actions": actions,
        "features": features,
        "heads": fold_heads,
        "thresholds": thresholds,
        "histories": histories,
        "diagnostics": diagnostics,
        "occurrence_hashes": occurrence_hashes,
        "candidate_head_seconds": candidate_head_seconds,
        "candidate_projected_channel_variance": projected_variance,
        "labels": labels,
        "folds": folds,
        "sample_indices": sample_indices,
    }


def _binary_role_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, object]:
    binary = np.asarray(labels, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    accepted = np.asarray(actions, dtype=np.bool_)
    positives = binary == 1
    negatives = ~positives
    tp_retained = int((accepted & positives).sum())
    tp_broken = int((~accepted & positives).sum())
    fp_retained = int((accepted & negatives).sum())
    fp_rejected = int((~accepted & negatives).sum())
    return {
        "rows": int(binary.size),
        "positives": int(positives.sum()),
        "negatives": int(negatives.sum()),
        "auroc": float(roc_auc_score(binary, score)),
        "tp_retained": tp_retained,
        "tp_broken": tp_broken,
        "tp_retention": float(tp_retained / max(int(positives.sum()), 1)),
        "fp_retained": fp_retained,
        "fp_rejected": fp_rejected,
        "fp_rejection": float(fp_rejected / max(int(negatives.sum()), 1)),
        "corrections": fp_rejected,
        "harms": tp_broken,
        "score_mean_positive": float(score[positives].mean()),
        "score_mean_negative": float(score[negatives].mean()),
    }


def build_clean_analysis(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    candidate_features: np.ndarray,
    projected_variance: np.ndarray,
    training_diagnostics: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    aggregate = {
        role: _binary_role_metrics(labels, scores[role], actions[role])
        for role in ROLE_NAMES
    }
    fold_metrics: Dict[str, Dict[str, object]] = {}
    for fold in FOLDS:
        held = np.asarray(folds) == int(fold)
        fold_metrics[str(fold)] = {
            role: _binary_role_metrics(
                labels[held], scores[role][held], actions[role][held]
            )
            for role in ROLE_NAMES
        }
    candidate = aggregate["bdc32_aligned_base"]
    fold_wins = {
        role: sum(
            fold_metrics[str(fold)]["bdc32_aligned_base"]["auroc"]
            > fold_metrics[str(fold)][role]["auroc"]
            for fold in FOLDS
        )
        for role in MATCHED_CONTROL_ROLES
    }
    descriptor = np.asarray(candidate_features, dtype=np.float64)[:, :TRIANGLE_DIM]
    effective_rank = _effective_rank(descriptor)
    all_training_gradients = all(
        bool(fold_record["roles"][role]["all_parameters_receive_finite_nonzero_gradient"])
        and bool(fold_record["roles"][role]["all_parameters_changed"])
        and bool(fold_record["roles"][role]["loss_finite"])
        for fold_record in training_diagnostics.values()
        for role in TRAINED_ROLES
    )
    checks = {
        "candidate_auroc_ge_0_85": candidate["auroc"] >= 0.85,
        "candidate_minus_base_ge_0_02": candidate["auroc"]
        - aggregate["base_logprob"]["auroc"]
        >= 0.02,
        "candidate_minus_covariance_ge_0_02": candidate["auroc"]
        - aggregate["cov32_base"]["auroc"]
        >= 0.02,
        "candidate_minus_mean_ge_0_015": candidate["auroc"]
        - aggregate["mean32_base"]["auroc"]
        >= 0.015,
        "candidate_minus_trained_dephased_ge_0_03": candidate["auroc"]
        - aggregate["bdc32_dephased_base"]["auroc"]
        >= 0.03,
        "candidate_minus_same_weight_dephased_ge_0_03": candidate["auroc"]
        - aggregate[SAME_WEIGHT_ROLE]["auroc"]
        >= 0.03,
        "aligned_only_auroc_ge_0_65": aggregate["bdc32_aligned_only"]["auroc"]
        >= 0.65,
        "candidate_tp_retention_ge_0_95": candidate["tp_retention"] >= 0.95,
        "candidate_fp_rejection_ge_0_25": candidate["fp_rejection"] >= 0.25,
        "candidate_corrections_gt_harms": candidate["corrections"]
        > candidate["harms"],
        "every_fold_tp_retention_ge_0_90": all(
            fold_metrics[str(fold)]["bdc32_aligned_base"]["tp_retention"] >= 0.90
            for fold in FOLDS
        ),
        "candidate_rejects_ten_more_fp_than_each_control": all(
            candidate["fp_rejected"] >= aggregate[role]["fp_rejected"] + 10
            for role in MATCHED_CONTROL_ROLES
        ),
        "candidate_breaks_at_most_two_more_tp_than_each_control": all(
            candidate["tp_broken"] <= aggregate[role]["tp_broken"] + 2
            for role in MATCHED_CONTROL_ROLES
        ),
        "candidate_beats_each_control_in_four_of_five_folds": all(
            value >= 4 for value in fold_wins.values()
        ),
        "candidate_effective_rank_ge_16": effective_rank >= 16.0,
        "projected_channel_variance_finite_nonzero": bool(
            np.isfinite(projected_variance).all()
            and float(np.min(projected_variance)) > 0.0
        ),
        "all_trainable_parameters_active_and_changed": all_training_gradients,
    }
    return {
        "roles": aggregate,
        "folds": fold_metrics,
        "candidate_minus_controls_auroc": {
            role: float(candidate["auroc"] - aggregate[role]["auroc"])
            for role in MATCHED_CONTROL_ROLES
        },
        "candidate_fold_auroc_wins": fold_wins,
        "candidate_descriptor_effective_rank": effective_rank,
        "candidate_projected_channel_variance": {
            "minimum": float(np.min(projected_variance)),
            "maximum": float(np.max(projected_variance)),
            "mean": float(np.mean(projected_variance)),
        },
        "mechanism_gates": checks,
        "mechanism_gates_passed": all(checks.values()),
    }


def _prediction_rows(
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for position, source in enumerate(cohort):
        row: Dict[str, object] = {
            "position": position,
            "sample_index": int(source.sample_index),
            "source_stem": source.source_stem,
            "fold": int(source.fold),
            "target": int(source.target),
            "binary_target": int(source.target == FOCUS_CLASS),
        }
        for role in ROLE_NAMES:
            row[f"{role}_score"] = float(scores[role][position])
            row[f"{role}_accept"] = int(actions[role][position])
        output.append(row)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(to_serializable(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
        {
            "schema": "trkh_deepbdc_stem_joint_dependence_a0_manifest_v1",
            "files": rows,
        },
    )
    return path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in payload["files"]}
    observed = {
        artifact.relative_to(output_dir).as_posix()
        for artifact in output_dir.rglob("*")
        if artifact.is_file() and artifact != path
    }
    if set(expected) != observed:
        raise ValueError("DeepBDC artifact-manifest file set differs")
    for relative, row in expected.items():
        artifact = output_dir / relative
        if int(artifact.stat().st_size) != int(row["bytes"]) or _sha256(artifact) != row["sha256"]:
            raise ValueError(f"DeepBDC artifact differs from manifest: {relative}")
    return {
        "passed": True,
        "payload_count": len(expected),
        "manifest_sha256": _sha256(path),
    }


def save_replay_state(
    output_dir: Path,
    *,
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
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, object] = {"folds": {}}
    heads: Mapping[int, Mapping[str, StemJointHead]] = oof["heads"]
    thresholds: Mapping[int, Mapping[str, float]] = oof["thresholds"]
    for fold in FOLDS:
        fold_record: Dict[str, object] = {"roles": {}}
        for role in ROLE_NAMES:
            source_role = (
                "bdc32_aligned_base" if role == SAME_WEIGHT_ROLE else role
            )
            head = heads[int(fold)][source_role]
            state_keys: Dict[str, str] = {}
            for name, tensor in head.state_dict().items():
                key = f"fold{fold}__{role}__{name.replace('.', '___')}"
                arrays[key] = tensor.detach().cpu().numpy()
                state_keys[name] = key
            fold_record["roles"][role] = {
                "source_role": source_role,
                "force_dephase": role == SAME_WEIGHT_ROLE,
                "threshold": float(thresholds[int(fold)][role]),
                "state_arrays": state_keys,
            }
        metadata["folds"][str(fold)] = fold_record
    state_path = output_dir / "head_states.npz"
    np.savez_compressed(state_path, **arrays)
    metadata_path = output_dir / "head_state_metadata.json"
    _write_json(metadata_path, metadata)
    return {
        "features": {
            "path": str(feature_path.resolve()),
            "sha256": _sha256(feature_path),
        },
        "head_states": {
            "path": str(state_path.resolve()),
            "sha256": _sha256(state_path),
        },
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": _sha256(metadata_path),
        },
    }


def apply_frozen_heads(
    *,
    extraction: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    labels = np.asarray(
        [int(row.target == FOCUS_CLASS) for row in cohort], dtype=np.int64
    )
    maps = np.asarray(extraction["maps"], dtype=np.float16)
    probabilities = np.asarray(extraction["probabilities"], dtype=np.float32)
    sample_indices = np.asarray(extraction["sample_indices"], dtype=np.int64)
    scores = {
        role: np.full(len(cohort), np.nan, dtype=np.float64) for role in ROLE_NAMES
    }
    actions = {
        role: np.zeros(len(cohort), dtype=np.bool_) for role in ROLE_NAMES
    }
    heads: Mapping[int, Mapping[str, StemJointHead]] = oof["heads"]
    thresholds: Mapping[int, Mapping[str, float]] = oof["thresholds"]
    for fold in FOLDS:
        held = folds == int(fold)
        positions = np.flatnonzero(held)
        for role in ROLE_NAMES:
            source_role = (
                "bdc32_aligned_base" if role == SAME_WEIGHT_ROLE else role
            )
            head = heads[int(fold)][source_role].to(device)
            held_scores, _, _, _ = _feature_scores(
                head,
                maps,
                probabilities,
                sample_indices,
                positions,
                device=device,
                force_dephase=role == SAME_WEIGHT_ROLE,
            )
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= float(
                thresholds[int(fold)][role]
            )
            head.to("cpu")
    role_metrics = {
        role: _binary_role_metrics(labels, scores[role], actions[role])
        for role in ROLE_NAMES
    }
    candidate = role_metrics["bdc32_aligned_base"]
    best_control = max(
        role_metrics[role]["auroc"] for role in MATCHED_CONTROL_ROLES
    )
    return {
        "roles": role_metrics,
        "candidate_minus_best_control_auroc": float(
            candidate["auroc"] - best_control
        ),
        "condition_gates": {
            "candidate_auroc_ge_0_80": candidate["auroc"] >= 0.80,
            "candidate_tp_retention_ge_0_92": candidate["tp_retention"] >= 0.92,
            "candidate_minus_best_control_positive": candidate["auroc"]
            > best_control,
        },
        "scores": scores,
        "actions": actions,
    }


def _review_positions(
    labels: np.ndarray,
    actions: np.ndarray,
    scores: np.ndarray,
) -> List[int]:
    binary = np.asarray(labels, dtype=np.int64)
    accepted = np.asarray(actions, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    categories = (
        ("tp_accept", (binary == 1) & accepted, False),
        ("tp_reject", (binary == 1) & ~accepted, True),
        ("fp_accept", (binary == 0) & accepted, True),
        ("fp_reject", (binary == 0) & ~accepted, False),
    )
    selected: List[int] = []
    for category_index, (_name, mask, ascending) in enumerate(categories):
        target_size = (category_index + 1) * 4
        positions = np.flatnonzero(mask)
        if positions.size:
            order = np.argsort(values[positions])
            if not ascending:
                order = order[::-1]
            for position in positions[order]:
                if int(position) not in selected:
                    selected.append(int(position))
                if len(selected) == target_size:
                    break
        while len(selected) < target_size:
            fallback = [
                int(position)
                for position in np.argsort(np.abs(values - 0.5))
                if int(position) not in selected
            ]
            if not fallback:
                raise RuntimeError("DeepBDC contact sheet cannot select 16 unique rows")
            selected.append(fallback[0])
    if len(selected) != 16 or len(set(selected)) != 16:
        raise RuntimeError("DeepBDC contact-sheet selection differs from 16-row lock")
    return selected


def _normalized_heatmap(value: Tensor) -> np.ndarray:
    array = value.detach().cpu().float().numpy()
    array = array - float(array.min())
    maximum = float(array.max())
    if maximum > 0.0:
        array = array / maximum
    return array.astype(np.float32)


def _head_gradient_map(
    head: StemJointHead,
    *,
    stem_map: np.ndarray,
    probabilities: np.ndarray,
    sample_index: int,
    device: torch.device,
    force_dephase: bool,
) -> Tuple[np.ndarray, float]:
    head = head.to(device).eval()
    head.zero_grad(set_to_none=True)
    map_tensor = torch.from_numpy(stem_map.astype(np.float32)).unsqueeze(0).to(device)
    map_tensor.requires_grad_(True)
    probability_tensor = torch.from_numpy(probabilities.astype(np.float32)).unsqueeze(0).to(device)
    index_tensor = torch.tensor([int(sample_index)], device=device, dtype=torch.long)
    score = torch.sigmoid(
        head(
            map_tensor,
            probability_tensor,
            index_tensor,
            force_dephase=force_dephase,
        )
    )[0]
    score.backward()
    if map_tensor.grad is None or not bool(torch.isfinite(map_tensor.grad).all()):
        raise RuntimeError("DeepBDC feature-gradient map is missing or non-finite")
    saliency = (map_tensor.grad * map_tensor).abs().mean(dim=1)[0]
    result = _normalized_heatmap(saliency)
    score_value = float(score.detach().cpu())
    head.to("cpu")
    return result, score_value


def _heatmap_image(heatmap: np.ndarray, *, size: int = 256) -> Image.Image:
    value = np.asarray(heatmap, dtype=np.float32).clip(0.0, 1.0)
    red = value
    green = np.clip(1.5 - np.abs(value - 0.5) * 3.0, 0.0, 1.0)
    blue = 1.0 - value
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _annotated_input(
    rgb: np.ndarray,
    geometry: Mapping[str, object],
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x = image.width / 32.0
    scale_y = image.height / 32.0
    rectangle = (
        int(float(geometry["stem_x0"]) * scale_x),
        int(float(geometry["stem_y0"]) * scale_y),
        int(float(geometry["stem_x1_exclusive"]) * scale_x) - 1,
        int(float(geometry["stem_y1_exclusive"]) * scale_y) - 1,
    )
    draw.rectangle(rectangle, outline=(255, 235, 40), width=3)
    return image


def render_contact_sheet(
    path: Path,
    *,
    extraction: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    labels = np.asarray(oof["labels"], dtype=np.int64)
    candidate_actions = np.asarray(
        oof["actions"]["bdc32_aligned_base"], dtype=np.bool_
    )
    candidate_scores = np.asarray(
        oof["scores"]["bdc32_aligned_base"], dtype=np.float64
    )
    positions = _review_positions(labels, candidate_actions, candidate_scores)
    tile = 256
    title_height = 38
    row_height = tile + title_height
    columns = (
        "input_valid_rect",
        "aligned_bdc_gradient",
        "covariance_gradient",
        "same_weight_dephased_gradient",
    )
    canvas = Image.new("RGB", (tile * len(columns), row_height * len(positions)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    manifest: List[Dict[str, object]] = []
    maps = np.asarray(extraction["maps"], dtype=np.float16)
    probabilities = np.asarray(extraction["probabilities"], dtype=np.float32)
    sample_indices = np.asarray(extraction["sample_indices"], dtype=np.int64)
    geometry = extraction["geometry"]
    rgb = np.asarray(extraction["rgb"], dtype=np.uint8)
    heads: Mapping[int, Mapping[str, StemJointHead]] = oof["heads"]
    for row_number, position in enumerate(positions):
        source = cohort[position]
        fold_heads = heads[int(source.fold)]
        candidate_map, candidate_score = _head_gradient_map(
            fold_heads["bdc32_aligned_base"],
            stem_map=maps[position],
            probabilities=probabilities[position],
            sample_index=int(sample_indices[position]),
            device=device,
            force_dephase=False,
        )
        covariance_map, covariance_score = _head_gradient_map(
            fold_heads["cov32_base"],
            stem_map=maps[position],
            probabilities=probabilities[position],
            sample_index=int(sample_indices[position]),
            device=device,
            force_dephase=False,
        )
        dephased_map, dephased_score = _head_gradient_map(
            fold_heads["bdc32_aligned_base"],
            stem_map=maps[position],
            probabilities=probabilities[position],
            sample_index=int(sample_indices[position]),
            device=device,
            force_dephase=True,
        )
        images = (
            _annotated_input(rgb[position], geometry[position]),
            _heatmap_image(candidate_map),
            _heatmap_image(covariance_map),
            _heatmap_image(dephased_map),
        )
        y = row_number * row_height
        action = "accept" if candidate_actions[position] else "reject"
        label = "TP" if labels[position] else "FP"
        titles = (
            f"idx={source.sample_index} target={source.target} {label}/{action}",
            f"aligned score={candidate_score:.4f}",
            f"cov score={covariance_score:.4f}",
            f"dephased score={dephased_score:.4f}",
        )
        for column, (image, title) in enumerate(zip(images, titles)):
            x = column * tile
            draw.text((x + 4, y + 4), title, fill="black", font=font)
            draw.text((x + 4, y + 19), columns[column], fill=(70, 70, 70), font=font)
            canvas.paste(image, (x, y + title_height))
        manifest.append(
            {
                "row": row_number,
                "position": position,
                "sample_index": int(source.sample_index),
                "target": int(source.target),
                "fold": int(source.fold),
                "binary_label": int(labels[position]),
                "candidate_action": int(candidate_actions[position]),
                "candidate_score": candidate_score,
                "covariance_score": covariance_score,
                "same_weight_dephased_score": dephased_score,
                "candidate_map_sha256": _array_sha256(candidate_map),
                "covariance_map_sha256": _array_sha256(covariance_map),
                "dephased_map_sha256": _array_sha256(dephased_map),
            }
        )
    canvas.save(path)
    manifest_path = path.with_suffix(".json")
    _write_json(manifest_path, {"columns": list(columns), "rows": manifest})
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(positions),
        "columns": list(columns),
    }


def render_geometry_preview(
    path: Path,
    *,
    extraction: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    targets = (1, 0, 2, 4)
    positions = [
        next(position for position, row in enumerate(cohort) if row.target == target)
        for target in targets
    ]
    canvas = Image.new("RGB", (256 * len(positions), 286), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    rgb = np.asarray(extraction["rgb"], dtype=np.uint8)
    geometry = extraction["geometry"]
    for column, position in enumerate(positions):
        row = cohort[position]
        draw.text(
            (column * 256 + 4, 5),
            f"sample={row.sample_index} target={row.target} fold={row.fold}",
            fill="black",
            font=font,
        )
        canvas.paste(
            _annotated_input(rgb[position], geometry[position]),
            (column * 256, 30),
        )
    canvas.save(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "positions": positions,
        "sample_indices": [int(cohort[position].sample_index) for position in positions],
        "targets": list(targets),
    }


def _score_from_features(
    feature_matrix: np.ndarray,
    weight: np.ndarray,
    bias: float,
) -> np.ndarray:
    logits = (
        np.asarray(feature_matrix, dtype=np.float64)
        @ np.asarray(weight, dtype=np.float64).reshape(-1)
        + float(bias)
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def replay_oof_from_saved_state(
    *,
    feature_path: Path,
    state_path: Path,
    metadata_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with np.load(feature_path, allow_pickle=False) as feature_archive, np.load(
        state_path, allow_pickle=False
    ) as state_archive:
        folds = np.asarray(feature_archive["folds"], dtype=np.int64)
        rows = int(folds.size)
        scores = {
            role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES
        }
        actions = {
            role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES
        }
        for fold in FOLDS:
            held = folds == int(fold)
            fold_record = metadata["folds"][str(fold)]["roles"]
            for role in ROLE_NAMES:
                role_record = fold_record[role]
                state_keys = role_record["state_arrays"]
                weight = np.asarray(
                    state_archive[state_keys["classifier.weight"]], dtype=np.float64
                )
                bias = float(
                    np.asarray(state_archive[state_keys["classifier.bias"]]).reshape(-1)[0]
                )
                role_scores = _score_from_features(
                    np.asarray(feature_archive[role])[held], weight, bias
                )
                scores[role][held] = role_scores
                actions[role][held] = role_scores >= float(role_record["threshold"])
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("Saved DeepBDC replay produced non-finite scores")
    return scores, actions


def _maximum_score_difference(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> float:
    return max(
        float(np.max(np.abs(np.asarray(left[role]) - np.asarray(right[role]))))
        for role in ROLE_NAMES
    )


def _save_replay_state_with_axes(
    output_dir: Path,
    *,
    oof: Mapping[str, object],
) -> Dict[str, object]:
    feature_path = output_dir / "oof_replay_features.npz"
    payload = {
        role: np.asarray(oof["features"][role], dtype=np.float64)
        for role in ROLE_NAMES
    }
    payload.update(
        {
            "labels": np.asarray(oof["labels"], dtype=np.int64),
            "folds": np.asarray(oof["folds"], dtype=np.int64),
            "sample_indices": np.asarray(oof["sample_indices"], dtype=np.int64),
            "candidate_projected_variance": np.asarray(
                oof["candidate_projected_channel_variance"], dtype=np.float64
            ),
        }
    )
    np.savez_compressed(feature_path, **payload)
    state_artifacts = save_replay_state(output_dir, oof=oof)
    # save_replay_state writes the same feature path without axes; restore the
    # complete payload after it serializes the head state.
    np.savez_compressed(feature_path, **payload)
    state_artifacts["features"] = {
        "path": str(feature_path.resolve()),
        "sha256": _sha256(feature_path),
    }
    return state_artifacts


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
    output: Dict[str, object] = {"unknown_process_terminated": False}
    for name, command in commands.items():
        try:
            result = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=10,
                check=False,
            )
            output[name] = {
                "exit_code": int(result.returncode),
                "stdout": result.stdout.strip().splitlines(),
                "stderr": result.stderr.strip().splitlines(),
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            output[name] = {"error": str(error)}
    try:
        script = (
            "$samples=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' "
            "-SampleInterval 1 -MaxSamples 2).CounterSamples | "
            "Where-Object {$_.CookedValue -ge 0.5} | "
            "Select-Object InstanceName,CookedValue; "
            "$samples | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
        output["windows_gpu_engine"] = {
            "exit_code": int(result.returncode),
            "active_samples_json": result.stdout.strip(),
            "stderr": result.stderr.strip().splitlines(),
            "minimum_reported_percent": 0.5,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        output["windows_gpu_engine"] = {"error": str(error)}
    return output


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    engineering = engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError(f"DeepBDC equation checks failed: {engineering}")
    return {
        "mode": "deepbdc_stem_joint_dependence_a0_preflight",
        "passed": True,
        "output_created": False,
        "model_loaded": False,
        "dataset_pixels_loaded": False,
        "validation_data_used": False,
        "test_data_used": False,
        "locked_inputs_verified": True,
        "engineering": engineering,
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "provenance": provenance,
    }


def engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for DeepBDC engineering forward")
    provenance = verify_locked_inputs(args)
    device = torch.device("cuda")
    model, checkpoint, _ = load_model(Path(provenance["files"]["keeper"]["path"]), device)
    all_rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    full_cohort = _cohort_from_rows(all_rows)
    positive = next(row for row in full_cohort if row.target == FOCUS_CLASS)
    negative = next(row for row in full_cohort if row.target != FOCUS_CLASS)
    cohort = [positive, negative]
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint, all_rows, Path(provenance["files"]["data_yaml"]["path"])
    )
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="deepbdc_stem_joint_dependence_a0_engineering_forward",
    )
    extraction = extract_condition(
        model=model,
        loader=loader,
        loader_summary=loader_summary,
        cohort=cohort,
        device=device,
        semantics=dataset_summary["semantics"],
        condition="engineering_clean",
        benchmark_ordinary=True,
    )
    labels = np.asarray([1, 0], dtype=np.int64)
    sample_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    heads = make_matched_heads(
        fold=0,
        base_mean=np.zeros(5, dtype=np.float32),
        base_std=np.ones(5, dtype=np.float32),
    )
    role_checks: Dict[str, object] = {}
    for role, head in heads.items():
        _, diagnostics = _train_role(
            head,
            maps=np.asarray(extraction["maps"], dtype=np.float16),
            probabilities=np.asarray(extraction["probabilities"], dtype=np.float32),
            sample_indices=sample_indices,
            labels=labels,
            orders=[np.asarray([0, 1], dtype=np.int64)],
            device=device,
        )
        role_checks[role] = diagnostics
        head.to("cpu")
    passed = bool(
        _protocol_extraction_checks_passed(extraction["checks"])
        and all(
            record["all_parameters_receive_finite_nonzero_gradient"]
            and record["all_parameters_changed"]
            for record in role_checks.values()
        )
    )
    return {
        "mode": "deepbdc_stem_joint_dependence_a0_engineering_forward",
        "passed": passed,
        "rows": len(cohort),
        "extraction": {
            "checks": extraction["checks"],
            "runtime": extraction["runtime"],
            "feature_cache": extraction["feature_cache"],
            "capture_probability_max_abs_error": extraction[
                "capture_probability_max_abs_error"
            ],
            "cidt_probability_max_abs_error": extraction[
                "cidt_probability_max_abs_error"
            ],
        },
        "one_step_roles": role_checks,
        "gpu": _gpu_snapshot(),
        "output_created": False,
        "validation_data_used": False,
        "test_data_used": False,
    }


def _condition_summary(
    *,
    name: str,
    analysis: Mapping[str, object],
    extraction: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "condition": name,
        "roles": analysis["roles"],
        "candidate_minus_best_control_auroc": analysis[
            "candidate_minus_best_control_auroc"
        ],
        "condition_gates": analysis["condition_gates"],
        "condition_gates_passed": all(analysis["condition_gates"].values()),
        "extraction_checks": extraction["checks"],
        "runtime": extraction["runtime"],
        "temporary_feature_cache_sha256": extraction["feature_cache"]["sha256"],
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_DEEPBDC_A0_PREFLIGHT") != "passed":
        raise RuntimeError("Formal DeepBDC A0 must use the locked PowerShell preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal DeepBDC A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            f"Formal DeepBDC A0 requires clean pushed tracked state: {repo_state}"
        )
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    equation_checks = engineering_checks()
    if not bool(equation_checks["passed"]):
        raise RuntimeError("DeepBDC engineering checks changed before formal run")
    device = torch.device("cuda")
    gpu_before = _gpu_snapshot()
    model, checkpoint, class_names = load_model(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    all_rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    cohort = _cohort_from_rows(all_rows)
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint, all_rows, Path(provenance["files"]["data_yaml"]["path"])
    )
    clean_loader, clean_loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="deepbdc_stem_joint_dependence_a0_clean",
    )
    clean = extract_condition(
        model=model,
        loader=clean_loader,
        loader_summary=clean_loader_summary,
        cohort=cohort,
        device=device,
        semantics=dataset_summary["semantics"],
        condition="clean",
        benchmark_ordinary=True,
    )
    if not _protocol_extraction_checks_passed(clean["checks"]):
        raise RuntimeError(f"DeepBDC clean extraction failed: {clean['checks']}")
    oof = fit_clean_oof_heads(
        maps=np.asarray(clean["maps"], dtype=np.float16),
        probabilities=np.asarray(clean["probabilities"], dtype=np.float32),
        cohort=cohort,
        device=device,
    )
    clean_analysis = build_clean_analysis(
        labels=np.asarray(oof["labels"]),
        folds=np.asarray(oof["folds"]),
        scores=oof["scores"],
        actions=oof["actions"],
        candidate_features=np.asarray(oof["features"]["bdc32_aligned_base"]),
        projected_variance=np.asarray(oof["candidate_projected_channel_variance"]),
        training_diagnostics=oof["diagnostics"],
    )
    state_artifacts = _save_replay_state_with_axes(output_dir, oof=oof)
    replay_scores, replay_actions = replay_oof_from_saved_state(
        feature_path=Path(state_artifacts["features"]["path"]),
        state_path=Path(state_artifacts["head_states"]["path"]),
        metadata_path=Path(state_artifacts["metadata"]["path"]),
    )
    replay_error = _maximum_score_difference(oof["scores"], replay_scores)
    replay_actions_exact = all(
        np.array_equal(oof["actions"][role], replay_actions[role])
        for role in ROLE_NAMES
    )
    ordinary_seconds = float(clean["runtime"]["ordinary_forward_seconds"])
    candidate_path_seconds = (
        float(clean["runtime"]["captured_forward_seconds"])
        + float(clean["runtime"]["valid_crop_seconds"])
        + float(oof["candidate_head_seconds"])
    )
    runtime_ratio = candidate_path_seconds / max(ordinary_seconds, 1e-12)
    peak_cuda_bytes = int(clean["runtime"]["peak_cuda_bytes"])
    structural_gates = {
        "locked_extraction_checks_passed": _protocol_extraction_checks_passed(
            clean["checks"]
        ),
        "equation_checks_passed": bool(equation_checks["passed"]),
        "all_fold_projection_initializations_exact": all(
            bool(record["projection_initialization_exact"])
            for record in oof["diagnostics"].values()
        ),
        "all_occurrence_hashes_present": set(oof["occurrence_hashes"])
        == set(FOLDS),
        "all_parameters_receive_gradient_and_change": clean_analysis[
            "mechanism_gates"
        ]["all_trainable_parameters_active_and_changed"],
        "in_process_score_replay_le_1e_7": replay_error
        <= MAX_REPLAY_SCORE_ERROR,
        "in_process_actions_exact": replay_actions_exact,
        "runtime_ratio_le_1_15": runtime_ratio <= 1.15,
        "peak_cuda_below_3_5_gib": peak_cuda_bytes
        < int(MAX_PEAK_CUDA_GIB * 1024**3),
        "temporary_cache_never_persisted": True,
        "unknown_process_not_terminated": True,
    }
    clean_gate_passed = bool(
        all(structural_gates.values()) and clean_analysis["mechanism_gates_passed"]
    )

    robustness_results: Dict[str, object] = {}
    robustness_passed = False
    if clean_gate_passed:
        aggregate_removals = 0
        aggregate_harms = 0
        for name, brightness, contrast in CONDITIONS:
            condition_loader, condition_loader_summary = _make_condition_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=[row.sample_index for row in cohort],
                brightness=brightness,
                contrast=contrast,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                context=f"deepbdc_stem_joint_dependence_a0_{name}",
            )
            condition_extraction = extract_condition(
                model=model,
                loader=condition_loader,
                loader_summary=condition_loader_summary,
                cohort=cohort,
                device=device,
                semantics=dataset_summary["semantics"],
                condition=name,
                benchmark_ordinary=False,
            )
            condition_analysis = apply_frozen_heads(
                extraction=condition_extraction,
                cohort=cohort,
                oof=oof,
                device=device,
            )
            robustness_results[name] = _condition_summary(
                name=name,
                analysis=condition_analysis,
                extraction=condition_extraction,
            )
            candidate_metrics = condition_analysis["roles"]["bdc32_aligned_base"]
            aggregate_removals += int(candidate_metrics["fp_rejected"])
            aggregate_harms += int(candidate_metrics["tp_broken"])
            del condition_extraction
        robustness_passed = bool(
            all(
                record["condition_gates_passed"]
                and _protocol_extraction_checks_passed(record["extraction_checks"])
                for record in robustness_results.values()
            )
            and aggregate_removals > aggregate_harms
        )
        robustness_aggregate = {
            "fp_removals": aggregate_removals,
            "tp_harms": aggregate_harms,
            "removals_exceed_harms": aggregate_removals > aggregate_harms,
        }
    else:
        robustness_aggregate = {
            "skipped": True,
            "reason": "clean conjunctive gate failed",
        }

    prediction_path = output_dir / "predictions_all_roles.csv"
    history_path = output_dir / "training_history.csv"
    geometry_path = output_dir / "stem_valid_geometry.csv"
    _write_csv(
        prediction_path,
        _prediction_rows(cohort, oof["scores"], oof["actions"]),
    )
    _write_csv(history_path, oof["histories"])
    _write_csv(geometry_path, clean["geometry"])
    contact_sheet = render_contact_sheet(
        output_dir / "deepbdc_clean_xai_contact_sheet.png",
        extraction=clean,
        cohort=cohort,
        oof=oof,
        device=device,
    )
    geometry_preview = render_geometry_preview(
        output_dir / "deepbdc_valid_geometry_preview.png",
        extraction=clean,
        cohort=cohort,
    )
    fold_metric_rows = []
    for fold in FOLDS:
        for role in ROLE_NAMES:
            fold_metric_rows.append(
                {
                    "fold": fold,
                    "role": role,
                    **clean_analysis["folds"][str(fold)][role],
                }
            )
    fold_metrics_path = output_dir / "fold_metrics.csv"
    _write_csv(fold_metrics_path, fold_metric_rows)
    gpu_after = _gpu_snapshot()
    summary_path = output_dir / "summary.json"
    automated_pre_replay = bool(
        clean_gate_passed and robustness_passed and contact_sheet["rows"] == 16
    )
    summary: Dict[str, object] = {
        "mode": MODE,
        "method": METHOD,
        "status": "awaiting_external_replay",
        "seed": SEED,
        "class_names": list(class_names),
        "validation_data_used": False,
        "test_data_used": False,
        "raw_dataset_modified": False,
        "image_model_modified": False,
        "image_model_checkpoint_written": False,
        "repo_state": repo_state,
        "provenance": provenance,
        "dataset": dataset_summary,
        "cohort": {
            "rows": len(cohort),
            "positives": int(np.asarray(oof["labels"]).sum()),
            "negatives": int(len(cohort) - np.asarray(oof["labels"]).sum()),
            "sample_indices_sha256": _array_sha256(
                np.asarray(oof["sample_indices"], dtype=np.int64)
            ),
        },
        "equation_checks": equation_checks,
        "clean_extraction": {
            "checks": clean["checks"],
            "runtime": clean["runtime"],
            "capture_probability_max_abs_error": clean[
                "capture_probability_max_abs_error"
            ],
            "cidt_probability_max_abs_error": clean[
                "cidt_probability_max_abs_error"
            ],
            "temporary_feature_cache": {
                **clean["feature_cache"],
                "persisted": False,
                "deleted_after_artifact_finalization": True,
            },
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "amp": False,
            "scheduler": None,
            "early_stopping": None,
            "occurrence_hashes": oof["occurrence_hashes"],
            "fold_diagnostics": oof["diagnostics"],
        },
        "clean_analysis": clean_analysis,
        "in_process_replay": {
            "score_max_abs_difference": replay_error,
            "actions_exact": replay_actions_exact,
            "passed": replay_error <= MAX_REPLAY_SCORE_ERROR
            and replay_actions_exact,
        },
        "resource": {
            "ordinary_keeper_seconds": ordinary_seconds,
            "captured_crop_candidate_seconds": candidate_path_seconds,
            "runtime_ratio": runtime_ratio,
            "peak_cuda_bytes": peak_cuda_bytes,
            "max_peak_cuda_bytes": int(MAX_PEAK_CUDA_GIB * 1024**3),
            "requested_workers": NUM_WORKERS,
            "effective_loader": clean["runtime"]["loader"],
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "unknown_process_terminated": False,
        },
        "structural_gates": structural_gates,
        "structural_gates_passed": all(structural_gates.values()),
        "clean_gate_passed": clean_gate_passed,
        "robustness_authorized_by_clean_gate": clean_gate_passed,
        "robustness_results": robustness_results,
        "robustness_aggregate": robustness_aggregate,
        "robustness_passed": robustness_passed,
        "contact_sheet": contact_sheet,
        "geometry_preview": geometry_preview,
        "artifacts": {
            "predictions": str(prediction_path.resolve()),
            "training_history": str(history_path.resolve()),
            "fold_metrics": str(fold_metrics_path.resolve()),
            "geometry": str(geometry_path.resolve()),
            "replay_state": state_artifacts,
        },
        "automated_gate_pre_external_replay": automated_pre_replay,
        "external_replay": {
            "required": True,
            "completed": False,
            "passed": False,
        },
        "visual_review": {
            "required": True,
            "completed": False,
            "passed": False,
        },
        "automated_gate_passed": False,
        "a0_passed": False,
        "trainer_integration_authorized": False,
        "matched_short_smoke_authorized": False,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(output_dir)
    del clean["maps"]
    del clean["rgb"]
    torch.cuda.empty_cache()
    return {
        "mode": MODE,
        "status": summary["status"],
        "clean_gate_passed": clean_gate_passed,
        "robustness_passed": robustness_passed,
        "automated_gate_pre_external_replay": automated_pre_replay,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }


def _load_prediction_scores(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("DeepBDC prediction CSV is empty")
    labels = np.asarray([int(row["binary_target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    scores = {
        role: np.asarray([float(row[f"{role}_score"]) for row in rows], dtype=np.float64)
        for role in ROLE_NAMES
    }
    actions = {
        role: np.asarray([bool(int(row[f"{role}_accept"])) for row in rows], dtype=np.bool_)
        for role in ROLE_NAMES
    }
    return labels, folds, sample_indices, scores, actions


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_recursive_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_recursive_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        a = float(left)
        b = float(right)
        if math.isnan(a) and math.isnan(b):
            return 0.0
        return abs(a - b)
    return 0.0 if left == right else float("inf")


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Replay target is not DeepBDC stem A0")
    artifacts = summary["artifacts"]
    state_artifacts = artifacts["replay_state"]
    feature_path = Path(state_artifacts["features"]["path"])
    state_path = Path(state_artifacts["head_states"]["path"])
    metadata_path = Path(state_artifacts["metadata"]["path"])
    for record in state_artifacts.values():
        artifact = Path(record["path"])
        if _sha256(artifact) != record["sha256"]:
            raise ValueError(f"Replay artifact hash differs: {artifact}")
    labels, folds, sample_indices, persisted_scores, persisted_actions = (
        _load_prediction_scores(Path(artifacts["predictions"]))
    )
    replay_scores, replay_actions = replay_oof_from_saved_state(
        feature_path=feature_path,
        state_path=state_path,
        metadata_path=metadata_path,
    )
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_labels = np.asarray(archive["labels"], dtype=np.int64)
        feature_folds = np.asarray(archive["folds"], dtype=np.int64)
        feature_indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        candidate_features = np.asarray(
            archive["bdc32_aligned_base"], dtype=np.float64
        )
        projected_variance = np.asarray(
            archive["candidate_projected_variance"], dtype=np.float64
        )
    score_error = _maximum_score_difference(persisted_scores, replay_scores)
    actions_exact = all(
        np.array_equal(persisted_actions[role], replay_actions[role])
        for role in ROLE_NAMES
    )
    axes_exact = bool(
        np.array_equal(labels, feature_labels)
        and np.array_equal(folds, feature_folds)
        and np.array_equal(sample_indices, feature_indices)
    )
    replay_analysis = build_clean_analysis(
        labels=feature_labels,
        folds=feature_folds,
        scores=replay_scores,
        actions=replay_actions,
        candidate_features=candidate_features,
        projected_variance=projected_variance,
        training_diagnostics=summary["training"]["fold_diagnostics"],
    )
    analysis_error = _recursive_numeric_difference(
        summary["clean_analysis"], replay_analysis
    )
    passed = bool(
        score_error <= MAX_REPLAY_SCORE_ERROR
        and actions_exact
        and axes_exact
        and analysis_error <= MAX_REPLAY_SCORE_ERROR
    )
    summary["external_replay"] = {
        "required": True,
        "completed": True,
        "second_process": True,
        "passed": passed,
        "score_max_abs_difference": score_error,
        "actions_exact": actions_exact,
        "sample_label_fold_axes_exact": axes_exact,
        "analysis_maximum_numeric_difference": analysis_error,
        "manifest_sha256_before_replay": manifest_before["manifest_sha256"],
    }
    automated = bool(summary["automated_gate_pre_external_replay"] and passed)
    summary["automated_gate_passed"] = automated
    summary["status"] = (
        "awaiting_visual_review"
        if automated
        else "rejected_clean_automated_gate"
        if not summary["clean_gate_passed"]
        else "rejected_robustness_or_replay_gate"
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    return {
        "mode": "deepbdc_stem_joint_dependence_a0_external_replay",
        "passed": passed,
        "automated_gate_passed": automated,
        "status": summary["status"],
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_after),
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }


def finalize_visual_review(
    summary_path: Path,
    *,
    result: str,
    expected_summary_sha256: str,
) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    observed_sha = _sha256(resolved)
    if observed_sha != str(expected_summary_sha256).strip().casefold():
        raise ValueError(
            "Visual-review summary SHA differs: "
            f"expected={expected_summary_sha256}, observed={observed_sha}"
        )
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Visual-review target is not DeepBDC stem A0")
    if not bool(summary.get("external_replay", {}).get("completed", False)):
        raise ValueError("DeepBDC visual review requires completed external replay")
    contact_path = Path(summary["contact_sheet"]["path"])
    contact_sha = _sha256(contact_path)
    if contact_sha != summary["contact_sheet"]["sha256"]:
        raise ValueError("DeepBDC contact sheet differs from its summary lock")
    visual_passed = str(result) == "pass"
    automated_passed = bool(summary["automated_gate_passed"])
    a0_passed = bool(visual_passed and automated_passed)
    summary["visual_review"] = {
        "required": True,
        "completed": True,
        "passed": visual_passed,
        "decision": str(result),
        "reviewed_summary_sha256": observed_sha,
        "contact_sheet_sha256": contact_sha,
        "manifest_sha256_before_review": manifest_before["manifest_sha256"],
        "cannot_rescue_automated_failure": True,
    }
    summary["a0_passed"] = a0_passed
    summary["trainer_integration_authorized"] = a0_passed
    summary["matched_short_smoke_authorized"] = a0_passed
    summary["full_train_authorized"] = False
    summary["validation_access_authorized"] = False
    summary["test_access_authorized"] = False
    summary["current_command_update_authorized"] = False
    summary["status"] = (
        "passed_a0_default_off_integration_and_matched_smoke_authorized"
        if a0_passed
        else "rejected_visual_gate"
        if automated_passed and not visual_passed
        else "rejected_automated_gate_visual_review_recorded"
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    return {
        "mode": "deepbdc_stem_joint_dependence_a0_visual_review",
        "decision": str(result),
        "automated_gate_passed": automated_passed,
        "a0_passed": a0_passed,
        "trainer_integration_authorized": a0_passed,
        "matched_short_smoke_authorized": a0_passed,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_after),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.finalize_visual_review:
        if args.replay_summary is None or not args.expected_summary_sha256:
            raise ValueError(
                "Visual finalization requires --replay-summary and --expected-summary-sha256"
            )
        result = finalize_visual_review(
            args.replay_summary,
            result=args.finalize_visual_review,
            expected_summary_sha256=args.expected_summary_sha256,
        )
    elif args.replay_summary is not None:
        result = replay_summary(args.replay_summary)
    elif args.preflight_only:
        result = preflight(args)
    elif args.engineering_forward:
        result = engineering_forward(args)
    else:
        result = run_audit(args)
    print(json.dumps(to_serializable(result), indent=2, sort_keys=True), flush=True)
    if (args.preflight_only or args.engineering_forward) and not bool(
        result.get("passed", False)
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

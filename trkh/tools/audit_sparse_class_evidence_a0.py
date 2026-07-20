from __future__ import annotations

"""Prospectively locked train-only sparse class-evidence audit.

The class-evidence-map and map-L1 equations are implemented independently from
Djoumessi et al., MIDL 2023. The authors' repository is provenance only and is
never imported by this module.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.config import to_serializable
from trkh.core.utils import set_seed
from trkh.inference.inference import load_model
from trkh.models.model import classification_logits_from_features
from trkh.tools.audit_hamburger_nmf_surface_a0 import (
    _build_dataset,
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


METHOD = "sparse_class_evidence_a0"
MODE = "sparse_class_evidence_a0_train_information_gate"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
FOCUS_CLASS = 1
FOLDS = (0, 1, 2, 3, 4)
FEATURE_CHANNELS = 64
FEATURE_SIZE = 48
EVIDENCE_CHANNELS = 2
EVIDENCE_SIZE = 24
MIN_FIT_TP_RETENTION = 0.97
REGULARIZER_CE_RATIO = 0.05
MAX_CAPTURE_PROBABILITY_ERROR = 1e-6
MAX_CIDT_PROBABILITY_ERROR = 3e-5
MAX_REPLAY_ERROR = 1e-7

TRAINED_ROLES = (
    "dense_aligned",
    "sparse_aligned",
    "logit_l1_aligned",
    "sparse_channel_dephased",
)
SAME_WEIGHT_ROLE = "sparse_same_weight_dephased"
ROLE_NAMES = TRAINED_ROLES + (SAME_WEIGHT_ROLE,)
CANDIDATE_ROLE = "sparse_aligned"
DENSE_ROLE = "dense_aligned"
LOGIT_L1_ROLE = "logit_l1_aligned"
TRAINED_DEPHASE_ROLE = "sparse_channel_dephased"

EXPECTED_COHORT_ROWS = 763
EXPECTED_KEEPER_TP = 528
EXPECTED_KEEPER_FN = 13
EXPECTED_RESTRICTED_FP = 222
EXPECTED_RESTRICTED_TARGET_COUNTS = {0: 158, 2: 54, 4: 10}
EXPECTED_FOLD_COUNTS = {
    0: {"tp": 107, "fn": 2, "fp": 36},
    1: {"tp": 112, "fn": 3, "fp": 45},
    2: {"tp": 100, "fn": 4, "fp": 48},
    3: {"tp": 101, "fn": 2, "fp": 52},
    4: {"tp": 108, "fn": 2, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2"
)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_FOLD_SUMMARY_SHA256 = "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
LOCKED_PROTOCOL_SHA256 = "c046891266bd07b59723b8f4d1a8f5db52e070404f0f3c8d71db1c415f286a7b"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "0fe7a832e20d4fb3438410664a2fc7ebe823189ec3a7a80f8989d2ee1e3a10ca"
LOCKED_OFFICIAL_COMMIT = "2b3731fdbec09e10cd2aa25b0742bc9164f9a60c"
LOCKED_OFFICIAL_TREE = "d9d54544af7392ebe6690fb412a86087540229a5"
LOCKED_OFFICIAL_LICENSE_SHA256 = "05e28d9ece454402704f2d6c8a41a89b7a4dd89214aa4edc8977fca32c5fe4e7"
LOCKED_OFFICIAL_BUILDER_SHA256 = "0b673cc8136d39b60e311ad9f6df5a8de8b28ec1105db8e8cd2a4c6620ec1480"
LOCKED_OFFICIAL_TRAIN_SHA256 = "33dacba60e37c884a4f91bb80d74d43bcad21b09e1d0f7340991fe7d7b2be9a1"
LOCKED_OFFICIAL_README_SHA256 = "6afef729a89c796e1162c2fd70cd6fd65e0f19fbec0b95c128871ec43875e19e"

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\sparse-activation-midl2023")
PAPER_PATH = Path(
    r"D:\DataAI\external_sources\papers\Djoumessi_Sparse_Activations_MIDL_2023.pdf"
)
CONDITIONS = (
    ("dim", 0.70, 0.90),
    ("bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
IMPLEMENTATION_PATHS = (
    "trkh/tools/audit_sparse_class_evidence_a0.py",
    "tests/test_audit_sparse_class_evidence_a0.py",
    "scripts/run_trkh_sparse_class_evidence_a0.ps1",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only sparse class-evidence A0 gate."
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
        "--fold-summary",
        type=Path,
        default=REPO_ROOT / "runs" / "yolof_cidt_fold0_trainonly_20260716" / "summary.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_SPARSE_CLASS_EVIDENCE_A0_PROTOCOL_20260721.md",
    )
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_sparse_class_evidence_a0_20260721",
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
        raise ValueError("Sparse class-evidence A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"Sparse class-evidence A0 locks batch-size={BATCH_SIZE}, "
            f"num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"Sparse class-evidence A0 locks seed={SEED}")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _cohort_index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def implementation_tracking_state() -> Dict[str, object]:
    records: Dict[str, object] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        tracked = False
        head_blob = ""
        worktree_blob = ""
        try:
            _git_value(REPO_ROOT, "ls-files", "--error-unmatch", relative)
            tracked = True
            head_blob = _git_value(REPO_ROOT, "rev-parse", f"HEAD:{relative}")
            worktree_blob = _git_value(REPO_ROOT, "hash-object", str(path))
        except subprocess.CalledProcessError:
            tracked = False
        records[relative] = {
            "tracked": tracked,
            "head_blob": head_blob,
            "worktree_blob": worktree_blob,
            "matches_head": bool(tracked and head_blob == worktree_blob),
        }
    return {
        "files": records,
        "passed": all(
            bool(record["tracked"] and record["matches_head"])
            for record in records.values()
        ),
    }


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
        "fold_summary": _verify_hash(
            args.fold_summary, LOCKED_FOLD_SUMMARY_SHA256, "CIDT fold summary"
        ),
        "protocol": _verify_hash(
            args.protocol, LOCKED_PROTOCOL_SHA256, "sparse class-evidence protocol"
        ),
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
        "paper": _verify_hash(PAPER_PATH, LOCKED_PAPER_SHA256, "accepted MIDL paper"),
        "official_license": _verify_hash(
            official_root / "LICENSE",
            LOCKED_OFFICIAL_LICENSE_SHA256,
            "official sparse-activation license",
        ),
        "official_builder": _verify_hash(
            official_root / "modules" / "builder.py",
            LOCKED_OFFICIAL_BUILDER_SHA256,
            "official sparse-activation builder",
        ),
        "official_train": _verify_hash(
            official_root / "train.py",
            LOCKED_OFFICIAL_TRAIN_SHA256,
            "official sparse-activation train",
        ),
        "official_readme": _verify_hash(
            official_root / "README.md",
            LOCKED_OFFICIAL_README_SHA256,
            "official sparse-activation README",
        ),
    }
    commit = _git_value(official_root, "rev-parse", "HEAD")
    tree = _git_value(official_root, "rev-parse", "HEAD^{tree}")
    status = _git_value(
        official_root, "status", "--porcelain", "--untracked-files=no"
    )
    if commit != LOCKED_OFFICIAL_COMMIT or tree != LOCKED_OFFICIAL_TREE or status:
        raise ValueError(
            "Official sparse-activation repository differs from lock: "
            f"commit={commit}, tree={tree}, status={status!r}"
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
            "tracked_worktree_clean": status == "",
            "license": "MIT",
            "software_reuse_permitted_but_not_used": True,
        },
        "protected_untracked": protected,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
    }


def evidence_logits_torch(evidence_maps: Tensor) -> Tensor:
    if evidence_maps.ndim != 4 or int(evidence_maps.shape[1]) != EVIDENCE_CHANNELS:
        raise ValueError("Evidence maps must have shape [B,2,H,W]")
    if not evidence_maps.is_floating_point() or not bool(torch.isfinite(evidence_maps).all()):
        raise ValueError("Evidence maps must be finite floating tensors")
    return evidence_maps.mean(dim=(2, 3))


def evidence_logits_numpy(evidence_maps: np.ndarray) -> np.ndarray:
    value = np.asarray(evidence_maps, dtype=np.float64)
    if value.ndim != 4 or value.shape[1] != EVIDENCE_CHANNELS:
        raise ValueError("Evidence oracle maps must have shape [B,2,H,W]")
    if not np.isfinite(value).all():
        raise ValueError("Evidence oracle maps must be finite")
    return value.mean(axis=(2, 3))


def evidence_objective_torch(
    evidence_maps: Tensor,
    targets: Tensor,
    *,
    regularizer: str,
    regularizer_lambda: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    logits = evidence_logits_torch(evidence_maps)
    labels = targets.reshape(-1).long()
    if int(labels.numel()) != int(logits.shape[0]):
        raise ValueError("Evidence targets do not align with maps")
    ce = F.cross_entropy(logits, labels)
    if regularizer == "none":
        penalty = logits.new_zeros(())
    elif regularizer == "map_l1":
        penalty = evidence_maps.abs().mean()
    elif regularizer == "logit_l1":
        penalty = logits.abs().mean()
    else:
        raise ValueError(f"Unknown evidence regularizer: {regularizer}")
    return ce + float(regularizer_lambda) * penalty, ce, penalty


def evidence_objective_numpy(
    evidence_maps: np.ndarray,
    targets: np.ndarray,
    *,
    regularizer: str,
    regularizer_lambda: float,
) -> Tuple[float, float, float]:
    maps = np.asarray(evidence_maps, dtype=np.float64)
    logits = evidence_logits_numpy(maps)
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    ce = float(np.mean(logsumexp - shifted[np.arange(labels.size), labels]))
    if regularizer == "none":
        penalty = 0.0
    elif regularizer == "map_l1":
        penalty = float(np.mean(np.abs(maps)))
    elif regularizer == "logit_l1":
        penalty = float(np.mean(np.abs(logits)))
    else:
        raise ValueError(f"Unknown evidence oracle regularizer: {regularizer}")
    return ce + float(regularizer_lambda) * penalty, ce, penalty


def _dephase_offsets(
    sample_indices: Sequence[int], *, fold: int, channels: int = FEATURE_CHANNELS
) -> Tuple[np.ndarray, np.ndarray]:
    vertical = np.empty((len(sample_indices), int(channels)), dtype=np.int64)
    horizontal = np.empty_like(vertical)
    for row, sample_index in enumerate(sample_indices):
        for channel in range(int(channels)):
            digest = hashlib.sha256(
                f"{SEED}:{int(fold)}:{int(sample_index)}:{channel}".encode("ascii")
            ).digest()
            dy = int.from_bytes(digest[:8], "little") % FEATURE_SIZE
            dx = int.from_bytes(digest[8:16], "little") % FEATURE_SIZE
            if dy == 0 and dx == 0:
                dx = 1
            vertical[row, channel] = dy
            horizontal[row, channel] = dx
    return vertical, horizontal


def channel_dephase(
    value: Tensor,
    sample_indices: Sequence[int],
    *,
    fold: int,
) -> Tensor:
    if value.ndim != 4 or tuple(value.shape[1:]) != (
        FEATURE_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError("Channel dephasing expects [B,64,48,48]")
    if len(sample_indices) != int(value.shape[0]):
        raise ValueError("Channel dephasing sample indices do not align")
    dy_numpy, dx_numpy = _dephase_offsets(sample_indices, fold=fold)
    dy = torch.from_numpy(dy_numpy).to(device=value.device)
    dx = torch.from_numpy(dx_numpy).to(device=value.device)
    rows = torch.arange(FEATURE_SIZE, device=value.device).view(1, 1, -1, 1)
    columns = torch.arange(FEATURE_SIZE, device=value.device).view(1, 1, 1, -1)
    source_y = (rows - dy.unsqueeze(-1).unsqueeze(-1)) % FEATURE_SIZE
    source_x = (columns - dx.unsqueeze(-1).unsqueeze(-1)) % FEATURE_SIZE
    gather_index = source_y * FEATURE_SIZE + source_x
    return torch.gather(value.flatten(2), 2, gather_index.flatten(2)).reshape_as(value)


class SparseEvidenceBranch(nn.Module):
    def __init__(self, local_block: nn.Module) -> None:
        super().__init__()
        self.local_block = copy.deepcopy(local_block)
        self.evidence = nn.Conv2d(256, EVIDENCE_CHANNELS, kernel_size=1, bias=True)

    def evidence_maps(
        self,
        features: Tensor,
        *,
        sample_indices: Optional[Sequence[int]] = None,
        fold: Optional[int] = None,
        dephase: bool = False,
    ) -> Tensor:
        value = features
        if dephase:
            if sample_indices is None or fold is None:
                raise ValueError("Dephased branch requires sample indices and fold")
            value = channel_dephase(value, sample_indices, fold=int(fold))
        result = self.evidence(self.local_block(value))
        if tuple(result.shape[1:]) != (
            EVIDENCE_CHANNELS,
            EVIDENCE_SIZE,
            EVIDENCE_SIZE,
        ):
            raise ValueError(f"Evidence map shape differs from lock: {tuple(result.shape)}")
        return result

    def forward(
        self,
        features: Tensor,
        *,
        sample_indices: Optional[Sequence[int]] = None,
        fold: Optional[int] = None,
        dephase: bool = False,
    ) -> Tensor:
        maps = self.evidence_maps(
            features,
            sample_indices=sample_indices,
            fold=fold,
            dephase=dephase,
        )
        return evidence_logits_torch(maps)


def _parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_matched_branches(local_block: nn.Module, *, fold: int) -> Dict[str, SparseEvidenceBranch]:
    before = _global_rng_snapshot()
    fork_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(SEED + int(fold))
        base = SparseEvidenceBranch(copy.deepcopy(local_block).cpu())
    branches = {role: copy.deepcopy(base) for role in TRAINED_ROLES}
    after = _global_rng_snapshot()
    if not _global_rng_equal(before, after):
        raise RuntimeError("Matched branch construction changed global RNG state")
    hashes = {_parameter_sha256(branch) for branch in branches.values()}
    if len(hashes) != 1:
        raise RuntimeError("Matched branch initial states differ")
    return branches


def engineering_checks() -> Dict[str, object]:
    generator = np.random.default_rng(1203)
    maps = generator.normal(size=(4, 2, 5, 7)).astype(np.float64)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    tensor = torch.from_numpy(maps).requires_grad_(True)
    logit_error = float(
        np.max(np.abs(evidence_logits_torch(tensor).detach().numpy() - evidence_logits_numpy(maps)))
    )
    objective_errors: Dict[str, float] = {}
    all_sample_gradients = True
    for regularizer in ("none", "map_l1", "logit_l1"):
        tensor.grad = None
        objective, _, _ = evidence_objective_torch(
            tensor,
            torch.from_numpy(labels),
            regularizer=regularizer,
            regularizer_lambda=0.173,
        )
        oracle, _, _ = evidence_objective_numpy(
            maps,
            labels,
            regularizer=regularizer,
            regularizer_lambda=0.173,
        )
        objective_errors[regularizer] = abs(float(objective.detach()) - oracle)
        objective.backward(retain_graph=True)
        all_sample_gradients = all_sample_gradients and bool(
            torch.isfinite(tensor.grad).all()
            and torch.all(tensor.grad.abs().flatten(1).sum(dim=1) > 0.0)
        )
    permutation = generator.permutation(maps.shape[2] * maps.shape[3])
    permuted = maps.reshape(4, 2, -1)[:, :, permutation].reshape(maps.shape)
    permutation_logit_error = float(
        np.max(np.abs(evidence_logits_numpy(permuted) - evidence_logits_numpy(maps)))
    )
    permutation_l1_error = abs(float(np.mean(np.abs(permuted))) - float(np.mean(np.abs(maps))))

    feature = torch.arange(
        2 * FEATURE_CHANNELS * FEATURE_SIZE * FEATURE_SIZE,
        dtype=torch.float32,
    ).reshape(2, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    indices = [19, 71]
    rng_before = _global_rng_snapshot()
    first = channel_dephase(feature, indices, fold=3)
    second = channel_dephase(feature, indices, fold=3)
    rng_after = _global_rng_snapshot()
    marginal_error = float(
        (
            torch.sort(first.flatten(2), dim=2).values
            - torch.sort(feature.flatten(2), dim=2).values
        )
        .abs()
        .max()
    )
    dephase_changed = not torch.equal(first, feature)
    checks = {
        "fp64_logits_match_numpy_le_1e_12": logit_error <= 1e-12,
        "fp64_objectives_match_numpy_le_1e_12": max(objective_errors.values()) <= 1e-12,
        "all_batch_items_regularized": all_sample_gradients,
        "common_spatial_permutation_preserves_logits": permutation_logit_error <= 1e-12,
        "common_spatial_permutation_preserves_map_l1": permutation_l1_error <= 1e-12,
        "dephasing_deterministic": torch.equal(first, second),
        "dephasing_marginal_preserving": marginal_error == 0.0,
        "dephasing_changes_alignment": dephase_changed,
        "dephasing_rng_local": _global_rng_equal(rng_before, rng_after),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": {
            "logits": logit_error,
            "objectives": objective_errors,
            "permutation_logits": permutation_logit_error,
            "permutation_map_l1": permutation_l1_error,
            "dephasing_marginal": marginal_error,
        },
    }


def locked_cohort(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    cohort = [
        row
        for row in rows
        if row.target == FOCUS_CLASS
        or (
            row.keeper_prediction == FOCUS_CLASS
            and row.target in EXPECTED_RESTRICTED_TARGET_COUNTS
        )
    ]
    keeper_tp = sum(
        row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
        for row in cohort
    )
    keeper_fn = sum(
        row.target == FOCUS_CLASS and row.keeper_prediction != FOCUS_CLASS
        for row in cohort
    )
    restricted_fp = sum(row.target != FOCUS_CLASS for row in cohort)
    observed = (len(cohort), keeper_tp, keeper_fn, restricted_fp)
    expected = (
        EXPECTED_COHORT_ROWS,
        EXPECTED_KEEPER_TP,
        EXPECTED_KEEPER_FN,
        EXPECTED_RESTRICTED_FP,
    )
    if observed != expected:
        raise ValueError(f"Sparse evidence cohort differs: {observed} != {expected}")
    target_counts = {
        target: sum(row.target == target for row in cohort)
        for target in EXPECTED_RESTRICTED_TARGET_COUNTS
    }
    if target_counts != EXPECTED_RESTRICTED_TARGET_COUNTS:
        raise ValueError(
            "Sparse evidence restricted target counts differ: "
            f"{target_counts} != {EXPECTED_RESTRICTED_TARGET_COUNTS}"
        )
    fold_counts: Dict[int, Dict[str, int]] = {}
    for fold in FOLDS:
        selected = [row for row in cohort if row.fold == fold]
        fold_counts[fold] = {
            "tp": sum(
                row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
                for row in selected
            ),
            "fn": sum(
                row.target == FOCUS_CLASS and row.keeper_prediction != FOCUS_CLASS
                for row in selected
            ),
            "fp": sum(row.target != FOCUS_CLASS for row in selected),
        }
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(
            f"Sparse evidence fold counts differ: {fold_counts} != {EXPECTED_FOLD_COUNTS}"
        )
    ordered_hash = _cohort_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Sparse evidence cohort order hash differs: {ordered_hash}")
    source_folds: Dict[str, set[int]] = {}
    for row in cohort:
        source_folds.setdefault(row.source_stem, set()).add(int(row.fold))
    overlap = {
        source: sorted(folds)
        for source, folds in source_folds.items()
        if len(folds) > 1
    }
    if overlap:
        raise ValueError(f"Sparse evidence cohort has cross-fold source overlap: {overlap}")
    return cohort


def _captured_keeper_forward(
    model: nn.Module,
    images: Tensor,
    image_mask: Tensor,
    bbox: Tensor,
) -> Tuple[Tensor, Tensor]:
    if not hasattr(model, "stem") or not hasattr(model.stem, "blocks"):  # type: ignore[attr-defined]
        raise TypeError("Keeper does not expose the locked HybridConvStem blocks")
    block = model.stem.blocks[1]  # type: ignore[attr-defined]
    captured: List[Tensor] = []

    def capture(_module: nn.Module, _inputs: Tuple[Tensor, ...], output: object) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Keeper block-2 hook emitted a non-tensor output")
        captured.append(output)

    hooks_before = len(block._forward_hooks)
    handle = block.register_forward_hook(capture)
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
    hooks_after = len(block._forward_hooks)
    if hooks_after != hooks_before or len(captured) != 1:
        raise RuntimeError("Keeper block-2 hook was not invoked and removed exactly once")
    block2 = captured[0]
    expected = (int(images.shape[0]), FEATURE_CHANNELS, 64, 64)
    if tuple(block2.shape) != expected:
        raise ValueError(
            f"Captured keeper block-2 shape differs: {tuple(block2.shape)} != {expected}"
        )
    return logits, block2


def _ordinary_keeper_probabilities(
    model: nn.Module,
    images: Tensor,
    image_mask: Tensor,
    bbox: Tensor,
) -> Tensor:
    features = model.forward_features(  # type: ignore[attr-defined]
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
    )
    features["bbox"] = bbox[:, :4]
    return classification_logits_from_features(model, features).float().softmax(dim=1)


def _normalized_bbox_mask(box: Tensor, *, size: int) -> Tensor:
    value = box[:4].float().clamp(0.0, 1.0)
    center_x, center_y, width, height = value.unbind(dim=0)
    yy = (torch.arange(size, device=box.device, dtype=torch.float32) + 0.5) / float(size)
    xx = (torch.arange(size, device=box.device, dtype=torch.float32) + 0.5) / float(size)
    return (
        (xx.view(1, -1) >= center_x - 0.5 * width)
        & (xx.view(1, -1) <= center_x + 0.5 * width)
        & (yy.view(-1, 1) >= center_y - 0.5 * height)
        & (yy.view(-1, 1) <= center_y + 0.5 * height)
    )


def crop_valid_block2_maps(
    block2: Tensor,
    image_valid_mask: Tensor,
    crop_bbox: Tensor,
    rgb: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, np.ndarray, List[Dict[str, object]]]:
    if block2.ndim != 4 or tuple(block2.shape[1:]) != (
        FEATURE_CHANNELS,
        64,
        64,
    ):
        raise ValueError("Captured block-2 features must have shape [B,64,64,64]")
    mask = image_valid_mask
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or int(mask.shape[0]) != int(block2.shape[0]):
        raise ValueError("Image-valid mask does not align with block-2 features")
    if crop_bbox.ndim != 2 or tuple(crop_bbox.shape) != (int(block2.shape[0]), 4):
        raise ValueError("Crop bbox does not align with block-2 features")
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 256, 256):
        raise ValueError("RGB tensor must have shape [B,3,256,256]")
    downsampled = F.interpolate(
        mask.to(device=block2.device, dtype=torch.float32),
        size=(64, 64),
        mode="area",
    )
    feature_crops: List[Tensor] = []
    valid_masks: List[Tensor] = []
    foreground_masks: List[Tensor] = []
    rgb_crops: List[np.ndarray] = []
    geometry: List[Dict[str, object]] = []
    for row in range(int(block2.shape[0])):
        valid = downsampled[row, 0] >= 0.5
        coordinates = torch.nonzero(valid, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("Image-valid mask has no valid block-2 location")
        y0 = int(coordinates[:, 0].min())
        y1 = int(coordinates[:, 0].max()) + 1
        x0 = int(coordinates[:, 1].min())
        x1 = int(coordinates[:, 1].max()) + 1
        feature_crop = block2[row : row + 1, :, y0:y1, x0:x1].float()
        feature_crops.append(
            F.interpolate(
                feature_crop,
                size=(FEATURE_SIZE, FEATURE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        )
        valid_crop = valid[y0:y1, x0:x1].float().view(1, 1, y1 - y0, x1 - x0)
        valid_masks.append(
            F.interpolate(valid_crop, size=(EVIDENCE_SIZE, EVIDENCE_SIZE), mode="nearest")
            .view(EVIDENCE_SIZE, EVIDENCE_SIZE)
            .bool()
        )
        bbox_mask = _normalized_bbox_mask(crop_bbox[row], size=64) & valid
        bbox_crop = bbox_mask[y0:y1, x0:x1].float().view(1, 1, y1 - y0, x1 - x0)
        foreground = (
            F.interpolate(
                bbox_crop,
                size=(EVIDENCE_SIZE, EVIDENCE_SIZE),
                mode="nearest",
            )
            .view(EVIDENCE_SIZE, EVIDENCE_SIZE)
            .bool()
        )
        if not bool(foreground.any()):
            raise ValueError("Object bbox has no evidence-grid foreground cell")
        foreground_masks.append(foreground)
        pixel_y0, pixel_y1 = y0 * 4, y1 * 4
        pixel_x0, pixel_x1 = x0 * 4, x1 * 4
        rgb_crop = F.interpolate(
            rgb[row : row + 1, :, pixel_y0:pixel_y1, pixel_x0:pixel_x1],
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )[0]
        rgb_crops.append(
            (rgb_crop.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        geometry.append(
            {
                "block2_x0": x0,
                "block2_y0": y0,
                "block2_x1_exclusive": x1,
                "block2_y1_exclusive": y1,
                "block2_width": x1 - x0,
                "block2_height": y1 - y0,
                "source_aspect_ratio": float((x1 - x0) / max(y1 - y0, 1)),
                "valid_fraction": float(valid.float().mean()),
                "rectangle_valid_fraction": float(valid[y0:y1, x0:x1].float().mean()),
                "foreground_fraction_evidence_grid": float(foreground.float().mean()),
            }
        )
    return (
        torch.cat(feature_crops, dim=0),
        torch.stack(valid_masks),
        torch.stack(foreground_masks),
        np.stack(rgb_crops),
        geometry,
    )


def extract_condition(
    *,
    model: nn.Module,
    loader: DataLoader,
    loader_summary: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    semantics: Mapping[str, object],
    condition: str,
    cache_path: Path,
    benchmark_ordinary: bool,
) -> Dict[str, object]:
    rows = len(cohort)
    resolved_cache = Path(cache_path).resolve()
    if resolved_cache.exists():
        raise FileExistsError(f"Temporary feature cache already exists: {resolved_cache}")
    resolved_cache.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        resolved_cache,
        mode="w+",
        dtype=np.float16,
        shape=(rows, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    )
    probabilities = np.empty((rows, 5), dtype=np.float32)
    targets = np.empty(rows, dtype=np.int64)
    sample_indices = np.empty(rows, dtype=np.int64)
    rgb_crops = np.empty((rows, 256, 256, 3), dtype=np.uint8)
    valid_masks = np.empty((rows, EVIDENCE_SIZE, EVIDENCE_SIZE), dtype=np.bool_)
    foreground_masks = np.empty_like(valid_masks)
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
                raise ValueError("Sparse evidence extraction requires tensor metadata")
            batch = int(batch_targets.numel())
            stop = position + batch
            expected_rows = cohort[position:stop]
            observed_indices = metadata["sample_index"].detach().cpu().long()
            if observed_indices.tolist() != [row.sample_index for row in expected_rows]:
                raise ValueError("Sparse evidence loader changed locked cohort order")
            if batch_targets.detach().cpu().long().tolist() != [
                row.target for row in expected_rows
            ]:
                raise ValueError("Sparse evidence loader changed locked targets")
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            bbox = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _metadata_tensor(
                metadata, "crop_bbox", device=device, dtype=torch.float32
            )
            if image_mask is None or bbox is None or crop_bbox is None:
                raise ValueError("Sparse evidence extraction requires image_mask/bbox/crop_bbox")
            if image_mask.ndim == 3:
                image_mask = image_mask.unsqueeze(1)
            bbox = bbox[:, :4]
            crop_bbox = crop_bbox[:, :4]
            images_device = images.to(device=device, dtype=torch.float32, non_blocking=True)
            ordinary = None
            if benchmark_ordinary:
                torch.cuda.synchronize(device)
                ordinary_started = time.perf_counter()
                ordinary = _ordinary_keeper_probabilities(
                    model, images_device, image_mask, bbox
                )
                torch.cuda.synchronize(device)
                ordinary_seconds += time.perf_counter() - ordinary_started
            torch.cuda.synchronize(device)
            captured_started = time.perf_counter()
            logits, block2 = _captured_keeper_forward(
                model, images_device, image_mask, bbox
            )
            current_probabilities = logits.softmax(dim=1)
            torch.cuda.synchronize(device)
            captured_seconds += time.perf_counter() - captured_started
            if ordinary is not None:
                capture_probability_error = max(
                    capture_probability_error,
                    float((ordinary - current_probabilities).abs().max()),
                )
            crop_started = time.perf_counter()
            rgb = _to_rgb(images, semantics).to(device)
            cropped, batch_valid, batch_foreground, batch_rgb, batch_geometry = (
                crop_valid_block2_maps(
                    block2,
                    image_mask,
                    crop_bbox,
                    rgb,
                )
            )
            torch.cuda.synchronize(device)
            crop_seconds += time.perf_counter() - crop_started
            batch_cache = cropped.detach().cpu().to(torch.float16).numpy()
            cache[position:stop] = batch_cache
            probabilities[position:stop] = current_probabilities.detach().cpu().numpy()
            targets[position:stop] = batch_targets.detach().cpu().numpy()
            sample_indices[position:stop] = observed_indices.numpy()
            rgb_crops[position:stop] = batch_rgb
            valid_masks[position:stop] = batch_valid.cpu().numpy()
            foreground_masks[position:stop] = batch_foreground.cpu().numpy()
            expected_probabilities = np.asarray(
                [row.keeper_probabilities for row in expected_rows], dtype=np.float32
            )
            cidt_probability_error = max(
                cidt_probability_error,
                float(
                    np.max(
                        np.abs(
                            probabilities[position:stop] - expected_probabilities
                        )
                    )
                ),
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
                geometry_rows.append(
                    {
                        "position": position + local,
                        "sample_index": int(observed_indices[local]),
                        "source_stem": expected_rows[local].source_stem,
                        "fold": int(expected_rows[local].fold),
                        "target": int(batch_targets[local]),
                        "condition": condition,
                        **geometry,
                        "feature_sha256": _array_sha256(batch_cache[local]),
                    }
                )
            position = stop
            if position % 256 < batch or position == rows:
                print(
                    json.dumps(
                        {
                            "stage": "sparse_evidence_block2_extraction",
                            "condition": condition,
                            "processed": position,
                            "rows": rows,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    cache.flush()
    if position != rows:
        raise RuntimeError("Sparse evidence extraction did not cover locked cohort")
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
        "all_features_finite": bool(np.isfinite(cache).all()),
        "all_probabilities_finite": bool(np.isfinite(probabilities).all()),
        "all_valid_masks_nonempty": bool(valid_masks.reshape(rows, -1).any(axis=1).all()),
        "all_foreground_masks_nonempty": bool(
            foreground_masks.reshape(rows, -1).any(axis=1).all()
        ),
    }
    elapsed = time.perf_counter() - started
    return {
        "condition": condition,
        "features": cache,
        "probabilities": probabilities,
        "targets": targets,
        "sample_indices": sample_indices,
        "rgb_crops": rgb_crops,
        "valid_masks": valid_masks,
        "foreground_masks": foreground_masks,
        "geometry": geometry_rows,
        "runtime": {
            "loader": dict(loader_summary),
            "elapsed_seconds": elapsed,
            "throughput_images_per_second": rows / max(elapsed, 1e-12),
            "ordinary_forward_seconds": ordinary_seconds,
            "captured_forward_seconds": captured_seconds,
            "valid_crop_seconds": crop_seconds,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
        },
        "checks": checks,
        "capture_probability_max_abs_error": capture_probability_error,
        "cidt_probability_max_abs_error": cidt_probability_error,
        "feature_cache": {
            "path": str(resolved_cache),
            "shape": list(cache.shape),
            "dtype": str(cache.dtype),
            "sha256": _array_sha256(cache),
            "temporary": True,
        },
    }


def _protocol_extraction_checks_passed(checks: Mapping[str, object]) -> bool:
    return all(
        bool(value)
        for key, value in checks.items()
        if key != "cidt_probability_error_le_3e_5"
    )


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(values[binary == 1])
    if positives.size == 0:
        raise ValueError("Fit folds contain no class-1 positive")
    allowed_breaks = int(
        math.floor((1.0 - MIN_FIT_TP_RETENTION) * positives.size)
    )
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    if float(np.mean(positives >= threshold)) + 1e-12 < MIN_FIT_TP_RETENTION:
        raise RuntimeError("Fit threshold violates locked class-1 retention")
    return threshold


def _epoch_orders(fit_positions: np.ndarray, *, fold: int) -> Tuple[List[np.ndarray], str]:
    generator = np.random.default_rng(SEED + int(fold))
    orders = [
        np.asarray(fit_positions, dtype=np.int64)[
            generator.permutation(len(fit_positions))
        ]
        for _ in range(EPOCHS)
    ]
    occurrence = np.concatenate(
        [np.asarray(order, dtype=np.int64) for order in orders]
    )
    return orders, _array_sha256(occurrence)


def _optimizer_summary(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> Dict[str, object]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    digest = hashlib.sha256()
    finite = True
    steps: List[float] = []
    tensor_count = 0
    tensor_dtypes: set[str] = set()
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names.get(id(item[0]), "")
    ):
        digest.update(names.get(id(parameter), "unknown").encode("utf-8"))
        for key, value in sorted(state.items()):
            digest.update(str(key).encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                finite = finite and bool(torch.isfinite(tensor).all())
                tensor_count += 1
                tensor_dtypes.add(str(tensor.dtype))
                if str(key) == "step":
                    steps.append(float(tensor.reshape(-1)[0]))
            else:
                digest.update(repr(value).encode("utf-8"))
    return {
        "sha256": digest.hexdigest(),
        "parameters_with_state": len(optimizer.state),
        "state_tensor_count": tensor_count,
        "state_tensor_dtypes": sorted(tensor_dtypes),
        "all_state_tensors_finite": finite,
        "minimum_step": min(steps) if steps else 0.0,
        "maximum_step": max(steps) if steps else 0.0,
    }


def _role_regularizer(role: str) -> str:
    if role == DENSE_ROLE:
        return "none"
    if role == LOGIT_L1_ROLE:
        return "logit_l1"
    if role in {CANDIDATE_ROLE, TRAINED_DEPHASE_ROLE}:
        return "map_l1"
    raise ValueError(f"Unknown trained sparse-evidence role: {role}")


def _probabilities_from_maps(maps: np.ndarray) -> np.ndarray:
    logits = evidence_logits_numpy(np.asarray(maps, dtype=np.float64))
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential[:, 1] / exponential.sum(axis=1)


def train_role(
    *,
    role: str,
    branch: SparseEvidenceBranch,
    features: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    fit_positions: np.ndarray,
    orders: Sequence[np.ndarray],
    occurrence_sha256: str,
    fold: int,
    device: torch.device,
) -> Dict[str, object]:
    branch = branch.to(device)
    initial_sha256 = _parameter_sha256(branch)
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in branch.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        branch.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
    )
    regularizer = _role_regularizer(role)
    regularizer_lambda = 0.0 if regularizer == "none" else None
    calibration: Dict[str, object] = {
        "regularizer": regularizer,
        "target_ratio": 0.0 if regularizer == "none" else REGULARIZER_CE_RATIO,
    }
    first_gradients: Dict[str, float] = {}
    first_updates: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    branch.train()
    for epoch, order in enumerate(orders, start=1):
        sums = {"objective": 0.0, "ce": 0.0, "regularizer": 0.0}
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            positions = np.asarray(order[start : start + BATCH_SIZE], dtype=np.int64)
            feature_tensor = torch.from_numpy(
                np.asarray(features[positions], dtype=np.float32)
            ).to(device=device, non_blocking=True)
            target_tensor = torch.from_numpy(labels[positions].astype(np.int64)).to(device)
            indices = sample_indices[positions].tolist()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                evidence_maps = branch.evidence_maps(
                    feature_tensor,
                    sample_indices=indices,
                    fold=fold,
                    dephase=role == TRAINED_DEPHASE_ROLE,
                )
            _, ce, penalty = evidence_objective_torch(
                evidence_maps.float(),
                target_tensor,
                regularizer=regularizer,
                regularizer_lambda=0.0,
            )
            if regularizer_lambda is None:
                penalty_value = float(penalty.detach())
                if not math.isfinite(penalty_value) or penalty_value <= 0.0:
                    raise RuntimeError(f"{role} first-batch regularizer is invalid")
                regularizer_lambda = (
                    REGULARIZER_CE_RATIO * float(ce.detach()) / penalty_value
                )
                calibration.update(
                    {
                        "ce": float(ce.detach()),
                        "regularizer_value": penalty_value,
                        "lambda": regularizer_lambda,
                        "observed_ratio": regularizer_lambda
                        * penalty_value
                        / float(ce.detach()),
                        "calibrated_before_first_update": True,
                        "first_batch_positions_sha256": _array_sha256(positions),
                    }
                )
            objective = ce + float(regularizer_lambda) * penalty
            if not bool(torch.isfinite(objective)):
                raise RuntimeError(f"Nonfinite {role} objective at epoch {epoch}")
            objective.backward()
            first_step = epoch == 1 and start == 0
            before_step: Dict[str, Tensor] = {}
            if first_step:
                for name, parameter in branch.named_parameters():
                    gradient = parameter.grad
                    first_gradients[name] = (
                        float(gradient.detach().norm()) if gradient is not None else 0.0
                    )
                    before_step[name] = parameter.detach().clone()
            optimizer.step()
            if first_step:
                first_updates = {
                    name: float((parameter.detach() - before_step[name]).norm())
                    for name, parameter in branch.named_parameters()
                }
            batch = len(positions)
            sums["objective"] += float(objective.detach()) * batch
            sums["ce"] += float(ce.detach()) * batch
            sums["regularizer"] += float(penalty.detach()) * batch
            seen += batch
        if seen != len(fit_positions):
            raise RuntimeError(f"{role} epoch occurrence count differs")
        history.append(
            {
                "epoch": float(epoch),
                "objective": sums["objective"] / seen,
                "ce": sums["ce"] / seen,
                "regularizer": sums["regularizer"] / seen,
            }
        )
    if regularizer_lambda is None:
        raise RuntimeError(f"{role} lambda was not calibrated")
    final_sha256 = _parameter_sha256(branch)
    final_changes = {
        name: float((parameter.detach().cpu() - initial_parameters[name]).norm())
        for name, parameter in branch.named_parameters()
    }
    parameter_names = {name for name, _ in branch.named_parameters()}
    gradient_pass = set(first_gradients) == parameter_names and all(
        math.isfinite(value) and value > 0.0 for value in first_gradients.values()
    )
    update_pass = (
        set(first_updates) == parameter_names
        and all(math.isfinite(value) and value > 0.0 for value in first_updates.values())
        and all(math.isfinite(value) and value > 0.0 for value in final_changes.values())
        and initial_sha256 != final_sha256
    )
    calibration_pass = regularizer == "none" or (
        bool(calibration.get("calibrated_before_first_update"))
        and abs(float(calibration["observed_ratio"]) - REGULARIZER_CE_RATIO)
        <= 1e-7
    )
    if not gradient_pass or not update_pass or not calibration_pass:
        raise RuntimeError(
            f"{role} training evidence failed: gradients={first_gradients}, "
            f"updates={first_updates}, calibration={calibration}"
        )
    return {
        "model": branch,
        "role": role,
        "fold": int(fold),
        "initial_sha256": initial_sha256,
        "final_sha256": final_sha256,
        "first_gradient_norms": first_gradients,
        "first_update_norms": first_updates,
        "final_change_norms": final_changes,
        "gradient_pass": gradient_pass,
        "update_pass": update_pass,
        "calibration": calibration,
        "calibration_pass": calibration_pass,
        "history": history,
        "history_finite": all(
            math.isfinite(value)
            for record in history
            for value in record.values()
        ),
        "occurrence_sha256": occurrence_sha256,
        "updates": int(sum(math.ceil(len(order) / BATCH_SIZE) for order in orders)),
        "optimizer": _optimizer_summary(optimizer, branch),
        "parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in branch.parameters()}
        ),
        "seconds": time.perf_counter() - started,
    }


def infer_role(
    *,
    branch: SparseEvidenceBranch,
    features: np.ndarray,
    sample_indices: np.ndarray,
    positions: np.ndarray,
    fold: int,
    device: torch.device,
    dephase: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    branch = branch.to(device).eval()
    map_rows: List[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(positions), BATCH_SIZE):
            selected = np.asarray(
                positions[start : start + BATCH_SIZE], dtype=np.int64
            )
            feature_tensor = torch.from_numpy(
                np.asarray(features[selected], dtype=np.float32)
            ).to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                maps = branch.evidence_maps(
                    feature_tensor,
                    sample_indices=sample_indices[selected].tolist(),
                    fold=fold,
                    dephase=dephase,
                )
            map_rows.append(maps.detach().cpu().float().numpy())
    evidence_maps = np.concatenate(map_rows, axis=0)
    scores = _probabilities_from_maps(evidence_maps)
    return scores, evidence_maps, time.perf_counter() - started


def fit_oof_branches(
    *,
    features: np.ndarray,
    cohort: Sequence[CleanTrainRow],
    local_block: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    rows = len(cohort)
    labels = np.asarray(
        [int(row.target == FOCUS_CLASS) for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    sample_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    evidence_maps = {
        role: np.full(
            (rows, EVIDENCE_CHANNELS, EVIDENCE_SIZE, EVIDENCE_SIZE),
            np.nan,
            dtype=np.float32,
        )
        for role in ROLE_NAMES
    }
    fold_records: Dict[int, Dict[str, object]] = {}
    training_records: List[Dict[str, object]] = []
    fit_score_records: Dict[str, np.ndarray] = {}
    occurrence_hashes: Dict[int, str] = {}
    candidate_inference_seconds = 0.0
    for fold in FOLDS:
        fit_positions = np.flatnonzero(folds != fold).astype(np.int64)
        held_positions = np.flatnonzero(folds == fold).astype(np.int64)
        fit_sources = {cohort[position].source_stem for position in fit_positions}
        held_sources = {cohort[position].source_stem for position in held_positions}
        source_overlap = sorted(fit_sources.intersection(held_sources))
        if source_overlap:
            raise ValueError(f"Sparse evidence fold {fold} has source leakage")
        orders, occurrence_sha256 = _epoch_orders(fit_positions, fold=fold)
        occurrence_hashes[fold] = occurrence_sha256
        branches = build_matched_branches(local_block, fold=fold)
        initial_hashes = {
            role: _parameter_sha256(branch) for role, branch in branches.items()
        }
        parameter_counts = {
            role: sum(int(parameter.numel()) for parameter in branch.parameters())
            for role, branch in branches.items()
        }
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError(f"Sparse evidence fold {fold} initial states differ")
        if len(set(parameter_counts.values())) != 1:
            raise RuntimeError(f"Sparse evidence fold {fold} parameter counts differ")
        trained: Dict[str, SparseEvidenceBranch] = {}
        thresholds: Dict[str, float] = {}
        fold_training: Dict[str, Dict[str, object]] = {}
        for role in TRAINED_ROLES:
            result = train_role(
                role=role,
                branch=branches[role],
                features=features,
                sample_indices=sample_indices,
                labels=labels,
                fit_positions=fit_positions,
                orders=orders,
                occurrence_sha256=occurrence_sha256,
                fold=fold,
                device=device,
            )
            branch = result.pop("model")
            trained[role] = branch
            fold_training[role] = result
            training_records.append(result)
            dephase = role == TRAINED_DEPHASE_ROLE
            fit_scores, _, _ = infer_role(
                branch=branch,
                features=features,
                sample_indices=sample_indices,
                positions=fit_positions,
                fold=fold,
                device=device,
                dephase=dephase,
            )
            held_scores, held_maps, elapsed = infer_role(
                branch=branch,
                features=features,
                sample_indices=sample_indices,
                positions=held_positions,
                fold=fold,
                device=device,
                dephase=dephase,
            )
            threshold = _positive_threshold(fit_scores, labels[fit_positions])
            thresholds[role] = threshold
            fit_score_records[f"fold{fold}__{role}"] = fit_scores
            scores[role][held_positions] = held_scores
            actions[role][held_positions] = held_scores >= threshold
            evidence_maps[role][held_positions] = held_maps
            if role == CANDIDATE_ROLE:
                candidate_inference_seconds += elapsed
            print(
                json.dumps(
                    {
                        "stage": "sparse_evidence_role_complete",
                        "fold": fold,
                        "role": role,
                        "lambda": result["calibration"].get("lambda", 0.0),
                        "final_objective": result["history"][-1]["objective"],
                        "threshold": threshold,
                    }
                ),
                flush=True,
            )
        candidate = trained[CANDIDATE_ROLE]
        same_fit_scores, _, _ = infer_role(
            branch=candidate,
            features=features,
            sample_indices=sample_indices,
            positions=fit_positions,
            fold=fold,
            device=device,
            dephase=True,
        )
        same_scores, same_maps, elapsed = infer_role(
            branch=candidate,
            features=features,
            sample_indices=sample_indices,
            positions=held_positions,
            fold=fold,
            device=device,
            dephase=True,
        )
        thresholds[SAME_WEIGHT_ROLE] = thresholds[CANDIDATE_ROLE]
        fit_score_records[f"fold{fold}__{SAME_WEIGHT_ROLE}"] = same_fit_scores
        scores[SAME_WEIGHT_ROLE][held_positions] = same_scores
        actions[SAME_WEIGHT_ROLE][held_positions] = (
            same_scores >= thresholds[CANDIDATE_ROLE]
        )
        evidence_maps[SAME_WEIGHT_ROLE][held_positions] = same_maps
        candidate_inference_seconds += elapsed
        for branch in trained.values():
            branch.to("cpu")
        fold_records[fold] = {
            "fit_positions": fit_positions,
            "held_positions": held_positions,
            "source_overlap": source_overlap,
            "occurrence_sha256": occurrence_sha256,
            "initial_hashes": initial_hashes,
            "parameter_counts": parameter_counts,
            "thresholds": thresholds,
            "models": trained,
            "training": fold_training,
        }
        torch.cuda.empty_cache()
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("Sparse evidence OOF scores are incomplete")
    if not all(np.isfinite(value).all() for value in evidence_maps.values()):
        raise RuntimeError("Sparse evidence OOF maps are incomplete")
    return {
        "labels": labels,
        "folds": folds,
        "sample_indices": sample_indices,
        "scores": scores,
        "actions": actions,
        "evidence_maps": evidence_maps,
        "fold_records": fold_records,
        "training_records": training_records,
        "fit_scores": fit_score_records,
        "occurrence_hashes": occurrence_hashes,
        "shared_occurrences": all(
            record["occurrence_sha256"]
            == occurrence_hashes[int(record["fold"])]
            for record in training_records
        ),
        "candidate_inference_seconds": candidate_inference_seconds,
    }


def role_metrics(
    *,
    scores: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    keeper_predictions: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    accepted = np.asarray(actions, dtype=np.bool_)
    target_values = np.asarray(targets, dtype=np.int64)
    keeper = np.asarray(keeper_predictions, dtype=np.int64)
    binary = (target_values == FOCUS_CLASS).astype(np.int64)
    positives = binary == 1
    negatives = ~positives
    keeper_tp = positives & (keeper == FOCUS_CLASS)
    keeper_fn = positives & (keeper != FOCUS_CLASS)
    tp_kept = int(np.sum(accepted & keeper_tp))
    tp_broken = int(np.sum(~accepted & keeper_tp))
    fn_supported = int(np.sum(accepted & keeper_fn))
    fp_rejected = int(np.sum(~accepted & negatives))
    fp_kept = int(np.sum(accepted & negatives))
    fold_rows: Dict[str, object] = {}
    for fold in FOLDS:
        selected = np.asarray(folds, dtype=np.int64) == fold
        fold_positive = positives & selected
        fold_negative = negatives & selected
        fold_keeper_tp = keeper_tp & selected
        fold_keeper_fn = keeper_fn & selected
        fold_rows[str(fold)] = {
            "rows": int(selected.sum()),
            "auroc": float(roc_auc_score(binary[selected], values[selected])),
            "auprc": float(average_precision_score(binary[selected], values[selected])),
            "all_class1_retention": float(np.mean(accepted[fold_positive])),
            "keeper_tp_retention": float(np.mean(accepted[fold_keeper_tp])),
            "restricted_fp_rejection": float(np.mean(~accepted[fold_negative])),
            "keeper_tp_broken": int(np.sum(~accepted[fold_keeper_tp])),
            "keeper_fn_supported": int(np.sum(accepted[fold_keeper_fn])),
            "restricted_fp_rejected": int(np.sum(~accepted[fold_negative])),
        }
    transition_rows: Dict[str, object] = {}
    for target in sorted(set(target_values.tolist())):
        selected = target_values == int(target)
        transition_rows[f"target_{target}"] = {
            "rows": int(selected.sum()),
            "accepted": int(np.sum(accepted[selected])),
            "rejected": int(np.sum(~accepted[selected])),
            "acceptance_rate": float(np.mean(accepted[selected])),
        }
    return {
        "rows": int(binary.size),
        "auroc": float(roc_auc_score(binary, values)),
        "auprc": float(average_precision_score(binary, values)),
        "all_class1_retention": float(np.mean(accepted[positives])),
        "keeper_tp_retention": float(np.mean(accepted[keeper_tp])),
        "keeper_fn_support_rate": float(np.mean(accepted[keeper_fn])),
        "restricted_fp_rejection": float(np.mean(~accepted[negatives])),
        "precision_after_action": float(
            np.sum(accepted & positives) / max(int(np.sum(accepted)), 1)
        ),
        "all_class1_kept": int(np.sum(accepted & positives)),
        "keeper_tp_kept": tp_kept,
        "keeper_tp_broken": tp_broken,
        "keeper_fn_supported": fn_supported,
        "restricted_fp_kept": fp_kept,
        "restricted_fp_rejected": fp_rejected,
        "corrections_plus_fn_supports": fp_rejected + fn_supported,
        "folds": fold_rows,
        "target_transitions": transition_rows,
    }


def _hoyer_sparsity(values: np.ndarray) -> np.ndarray:
    matrix = np.abs(np.asarray(values, dtype=np.float64).reshape(len(values), -1))
    count = matrix.shape[1]
    l1 = matrix.sum(axis=1)
    l2 = np.sqrt(np.square(matrix).sum(axis=1))
    return (
        math.sqrt(count) - l1 / np.maximum(l2, 1e-30)
    ) / (math.sqrt(count) - 1.0)


def _top_mass_share(values: np.ndarray, *, fraction: float = 0.10) -> np.ndarray:
    matrix = np.abs(np.asarray(values, dtype=np.float64).reshape(len(values), -1))
    count = max(1, int(math.ceil(float(fraction) * matrix.shape[1])))
    partitioned = np.partition(matrix, matrix.shape[1] - count, axis=1)
    top = partitioned[:, -count:].sum(axis=1)
    return top / np.maximum(matrix.sum(axis=1), 1e-30)


def map_statistics(
    evidence_maps: np.ndarray,
    *,
    valid_masks: np.ndarray,
    foreground_masks: np.ndarray,
) -> Dict[str, object]:
    maps = np.asarray(evidence_maps, dtype=np.float64)
    if maps.ndim != 4 or tuple(maps.shape[1:]) != (
        EVIDENCE_CHANNELS,
        EVIDENCE_SIZE,
        EVIDENCE_SIZE,
    ):
        raise ValueError("Evidence map statistics received an invalid shape")
    valid = np.asarray(valid_masks, dtype=np.bool_)
    foreground = np.asarray(foreground_masks, dtype=np.bool_) & valid
    if tuple(valid.shape) != tuple(maps.shape[0:1] + maps.shape[2:]):
        raise ValueError("Valid masks do not align with evidence maps")
    margin = maps[:, 1] - maps[:, 0]
    absolute = np.abs(margin)
    total = absolute.reshape(len(maps), -1).sum(axis=1)
    foreground_mass = (
        (absolute * foreground).reshape(len(maps), -1).sum(axis=1)
        / np.maximum(total, 1e-30)
    )
    padding_mass = (
        (absolute * ~valid).reshape(len(maps), -1).sum(axis=1)
        / np.maximum(total, 1e-30)
    )
    border = np.zeros_like(valid)
    border_width = max(1, int(round(EVIDENCE_SIZE * 0.08)))
    border[:, :border_width, :] = True
    border[:, -border_width:, :] = True
    border[:, :, :border_width] = True
    border[:, :, -border_width:] = True
    border &= valid
    border_mass = (
        (absolute * border).reshape(len(maps), -1).sum(axis=1)
        / np.maximum(total, 1e-30)
    )
    hoyer = _hoyer_sparsity(margin)
    top_mass = _top_mass_share(margin)
    standard_deviation = margin.reshape(len(maps), -1).std(axis=1)
    logits = evidence_logits_numpy(maps)
    margin_reconstruction = np.abs(
        margin.mean(axis=(1, 2)) - (logits[:, 1] - logits[:, 0])
    )
    return {
        "hoyer_mean": float(hoyer.mean()),
        "hoyer_minimum": float(hoyer.min()),
        "top10_absolute_mass_share_mean": float(top_mass.mean()),
        "top10_absolute_mass_share_minimum": float(top_mass.min()),
        "absolute_mass_minimum": float(total.min()),
        "map_standard_deviation_minimum": float(standard_deviation.min()),
        "zero_map_count": int(np.sum(total <= 1e-12)),
        "constant_map_count": int(np.sum(standard_deviation <= 1e-8)),
        "foreground_mass_mean": float(foreground_mass.mean()),
        "foreground_mass_minimum": float(foreground_mass.min()),
        "border_mass_mean": float(border_mass.mean()),
        "border_mass_maximum": float(border_mass.max()),
        "padding_mass_mean": float(padding_mass.mean()),
        "padding_mass_maximum": float(padding_mass.max()),
        "margin_reconstruction_max_abs_error": float(margin_reconstruction.max()),
        "all_finite": bool(
            np.isfinite(maps).all()
            and np.isfinite(hoyer).all()
            and np.isfinite(top_mass).all()
        ),
    }


def build_clean_analysis(
    *,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    evidence_maps: Mapping[str, np.ndarray],
    targets: np.ndarray,
    keeper_predictions: np.ndarray,
    folds: np.ndarray,
    valid_masks: np.ndarray,
    foreground_masks: np.ndarray,
    training_records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    metrics = {
        role: role_metrics(
            scores=np.asarray(scores[role]),
            actions=np.asarray(actions[role]),
            targets=targets,
            keeper_predictions=keeper_predictions,
            folds=folds,
        )
        for role in ROLE_NAMES
    }
    map_stats = {
        role: map_statistics(
            np.asarray(evidence_maps[role]),
            valid_masks=valid_masks,
            foreground_masks=foreground_masks,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics[CANDIDATE_ROLE]
    dense = metrics[DENSE_ROLE]
    logit_l1 = metrics[LOGIT_L1_ROLE]
    trained_dephase = metrics[TRAINED_DEPHASE_ROLE]
    same_weight = metrics[SAME_WEIGHT_ROLE]
    candidate_maps = map_stats[CANDIDATE_ROLE]
    dense_maps = map_stats[DENSE_ROLE]
    fold_wins = [
        fold
        for fold in FOLDS
        if float(candidate["folds"][str(fold)]["auroc"])
        > float(trained_dephase["folds"][str(fold)]["auroc"])
    ]
    all_parameter_evidence = all(
        bool(record["gradient_pass"])
        and bool(record["update_pass"])
        and bool(record["calibration_pass"])
        and bool(record["history_finite"])
        for record in training_records
    )
    gates = {
        "candidate_auroc_ge_0_85": float(candidate["auroc"]) >= 0.85,
        "candidate_auprc_ge_0_90": float(candidate["auprc"]) >= 0.90,
        "candidate_all_class1_retention_ge_0_95": float(
            candidate["all_class1_retention"]
        )
        >= 0.95,
        "candidate_keeper_tp_retention_ge_0_95": float(
            candidate["keeper_tp_retention"]
        )
        >= 0.95,
        "candidate_keeper_fn_support_ge_8": int(candidate["keeper_fn_supported"])
        >= 8,
        "candidate_restricted_fp_rejection_ge_0_25": float(
            candidate["restricted_fp_rejection"]
        )
        >= 0.25,
        "corrections_plus_fn_supports_ge_tp_breaks": int(
            candidate["corrections_plus_fn_supports"]
        )
        >= int(candidate["keeper_tp_broken"]),
        "candidate_auroc_within_0_005_dense": float(candidate["auroc"])
        >= float(dense["auroc"]) - 0.005,
        "candidate_auprc_within_0_005_dense": float(candidate["auprc"])
        >= float(dense["auprc"]) - 0.005,
        "candidate_fp_rejection_ge_dense_plus_0_03": float(
            candidate["restricted_fp_rejection"]
        )
        >= float(dense["restricted_fp_rejection"]) + 0.03,
        "candidate_auroc_ge_logit_l1_plus_0_01": float(candidate["auroc"])
        >= float(logit_l1["auroc"]) + 0.01,
        "candidate_fp_rejection_ge_logit_l1_plus_0_03": float(
            candidate["restricted_fp_rejection"]
        )
        >= float(logit_l1["restricted_fp_rejection"]) + 0.03,
        "candidate_auroc_ge_trained_dephase_plus_0_03": float(candidate["auroc"])
        >= float(trained_dephase["auroc"]) + 0.03,
        "candidate_fp_rejection_ge_trained_dephase_plus_0_05": float(
            candidate["restricted_fp_rejection"]
        )
        >= float(trained_dephase["restricted_fp_rejection"]) + 0.05,
        "candidate_beats_trained_dephase_in_four_folds": len(fold_wins) >= 4,
        "same_weight_dephasing_lowers_auroc_ge_0_02": float(candidate["auroc"])
        >= float(same_weight["auroc"]) + 0.02,
        "candidate_hoyer_ge_dense_plus_0_10": float(candidate_maps["hoyer_mean"])
        >= float(dense_maps["hoyer_mean"]) + 0.10,
        "candidate_top10_mass_ge_dense_plus_0_10": float(
            candidate_maps["top10_absolute_mass_share_mean"]
        )
        >= float(dense_maps["top10_absolute_mass_share_mean"]) + 0.10,
        "candidate_maps_nonzero_nonconstant": int(candidate_maps["zero_map_count"])
        == 0
        and int(candidate_maps["constant_map_count"]) == 0,
        "candidate_margin_reconstruction_exact": float(
            candidate_maps["margin_reconstruction_max_abs_error"]
        )
        <= 1e-12,
        "candidate_foreground_mass_ge_0_60": float(
            candidate_maps["foreground_mass_mean"]
        )
        >= 0.60,
        "candidate_border_mass_le_0_30": float(candidate_maps["border_mass_mean"])
        <= 0.30,
        "candidate_has_no_padding_leak": float(
            candidate_maps["padding_mass_maximum"]
        )
        <= 1e-12,
        "all_trainable_parameters_active_changed_and_finite": all_parameter_evidence,
    }
    return {
        "roles": metrics,
        "map_statistics": map_stats,
        "candidate_deltas": {
            role: {
                "auroc": float(candidate["auroc"]) - float(metrics[role]["auroc"]),
                "auprc": float(candidate["auprc"]) - float(metrics[role]["auprc"]),
                "restricted_fp_rejection": float(
                    candidate["restricted_fp_rejection"]
                )
                - float(metrics[role]["restricted_fp_rejection"]),
            }
            for role in ROLE_NAMES
            if role != CANDIDATE_ROLE
        },
        "trained_dephase_fold_wins": fold_wins,
        "mechanism_gates": {key: bool(value) for key, value in gates.items()},
        "mechanism_gates_passed": all(bool(value) for value in gates.values()),
    }


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


def _write_manifest(output_dir: Path) -> Path:
    path = output_dir / "artifact_manifest.json"
    rows = []
    for artifact in sorted(output_dir.rglob("*")):
        if not artifact.is_file() or artifact == path:
            continue
        rows.append(
            {
                "path": artifact.relative_to(output_dir).as_posix(),
                "bytes": int(artifact.stat().st_size),
                "sha256": _sha256(artifact),
            }
        )
    _write_json(
        path,
        {"schema": "trkh_sparse_class_evidence_a0_manifest_v1", "files": rows},
    )
    return path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in payload["files"]}
    observed = {
        artifact.relative_to(output_dir).as_posix(): artifact
        for artifact in output_dir.rglob("*")
        if artifact.is_file() and artifact != path
    }
    if set(expected) != set(observed):
        raise ValueError("Sparse evidence artifact-manifest file set differs")
    for relative, artifact in observed.items():
        row = expected[relative]
        if int(artifact.stat().st_size) != int(row["bytes"]) or _sha256(artifact) != row["sha256"]:
            raise ValueError(f"Sparse evidence artifact differs from manifest: {relative}")
    return {
        "files": len(expected),
        "manifest_sha256": _sha256(path),
        "passed": True,
    }


def save_replay_artifacts(
    *,
    output_dir: Path,
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    valid_masks: np.ndarray,
    foreground_masks: np.ndarray,
    geometry: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    maps_path = output_dir / "oof_evidence_maps.npz"
    np.savez_compressed(
        maps_path,
        **{
            role: np.asarray(oof["evidence_maps"][role], dtype=np.float32)
            for role in ROLE_NAMES
        },
    )
    masks_path = output_dir / "evidence_geometry_masks.npz"
    np.savez_compressed(
        masks_path,
        valid=np.asarray(valid_masks, dtype=np.bool_),
        foreground=np.asarray(foreground_masks, dtype=np.bool_),
    )
    fit_scores_path = output_dir / "fit_scores.npz"
    np.savez_compressed(
        fit_scores_path,
        **{
            key: np.asarray(value, dtype=np.float64)
            for key, value in oof["fit_scores"].items()
        },
    )
    state_arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, object] = {"folds": {}}
    for fold in FOLDS:
        record = oof["fold_records"][fold]
        fold_metadata: Dict[str, object] = {
            "fit_positions": np.asarray(record["fit_positions"], dtype=np.int64).tolist(),
            "held_positions": np.asarray(record["held_positions"], dtype=np.int64).tolist(),
            "source_overlap": list(record["source_overlap"]),
            "occurrence_sha256": record["occurrence_sha256"],
            "thresholds": {
                role: float(record["thresholds"][role]) for role in ROLE_NAMES
            },
            "roles": {},
        }
        for role, model in record["models"].items():
            state_keys: Dict[str, str] = {}
            for name, tensor in model.state_dict().items():
                key = f"fold{fold}__{role}__{name.replace('.', '___')}"
                state_arrays[key] = tensor.detach().cpu().numpy()
                state_keys[name] = key
            fold_metadata["roles"][role] = {"state_arrays": state_keys}
        metadata["folds"][str(fold)] = fold_metadata
    states_path = output_dir / "trained_branch_states.npz"
    np.savez_compressed(states_path, **state_arrays)
    metadata_path = output_dir / "replay_metadata.json"
    _write_json(metadata_path, metadata)

    prediction_rows: List[Dict[str, object]] = []
    for position, row in enumerate(cohort):
        record: Dict[str, object] = {
            "position": position,
            "sample_index": int(row.sample_index),
            "source_stem": row.source_stem,
            "fold": int(row.fold),
            "target": int(row.target),
            "keeper_prediction": int(row.keeper_prediction),
            "binary_target": int(row.target == FOCUS_CLASS),
        }
        for role in ROLE_NAMES:
            record[f"score__{role}"] = float(oof["scores"][role][position])
            record[f"action__{role}"] = int(bool(oof["actions"][role][position]))
        prediction_rows.append(record)
    predictions_path = output_dir / "oof_predictions.csv"
    _write_csv(predictions_path, prediction_rows)
    geometry_path = output_dir / "block2_valid_geometry.csv"
    _write_csv(geometry_path, geometry)
    training_path = output_dir / "training_evidence.json"
    _write_json(
        training_path,
        {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "records": oof["training_records"],
        },
    )
    paths = {
        "evidence_maps": maps_path,
        "masks": masks_path,
        "fit_scores": fit_scores_path,
        "states": states_path,
        "metadata": metadata_path,
        "predictions": predictions_path,
        "geometry": geometry_path,
        "training": training_path,
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


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
        a, b = float(left), float(right)
        if math.isnan(a) and math.isnan(b):
            return 0.0
        return abs(a - b)
    return 0.0 if left == right else float("inf")


def replay_artifacts(
    output_dir: Path,
    *,
    expected_analysis: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    with (root / "oof_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_COHORT_ROWS:
        raise ValueError("Sparse evidence replay prediction row count differs")
    sample_indices = [int(row["sample_index"]) for row in rows]
    if _cohort_index_sha256(sample_indices) != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError("Sparse evidence replay sample order differs")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    keeper = np.asarray(
        [int(row["keeper_prediction"]) for row in rows], dtype=np.int64
    )
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    metadata = json.loads((root / "replay_metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "oof_evidence_maps.npz", allow_pickle=False) as archive:
        evidence_maps = {
            role: np.asarray(archive[role], dtype=np.float32) for role in ROLE_NAMES
        }
    with np.load(root / "evidence_geometry_masks.npz", allow_pickle=False) as archive:
        valid_masks = np.asarray(archive["valid"], dtype=np.bool_)
        foreground_masks = np.asarray(archive["foreground"], dtype=np.bool_)
    scores = {
        role: _probabilities_from_maps(evidence_maps[role]) for role in ROLE_NAMES
    }
    actions = {role: np.zeros(len(rows), dtype=np.bool_) for role in ROLE_NAMES}
    threshold_error = 0.0
    with np.load(root / "fit_scores.npz", allow_pickle=False) as fit_archive:
        for fold in FOLDS:
            fold_record = metadata["folds"][str(fold)]
            fit_positions = np.asarray(fold_record["fit_positions"], dtype=np.int64)
            held_positions = np.asarray(fold_record["held_positions"], dtype=np.int64)
            for role in TRAINED_ROLES:
                fit_scores = np.asarray(
                    fit_archive[f"fold{fold}__{role}"], dtype=np.float64
                )
                threshold = _positive_threshold(fit_scores, labels[fit_positions])
                stored = float(fold_record["thresholds"][role])
                threshold_error = max(threshold_error, abs(threshold - stored))
                actions[role][held_positions] = scores[role][held_positions] >= threshold
            candidate_threshold = float(
                fold_record["thresholds"][CANDIDATE_ROLE]
            )
            same_threshold = float(fold_record["thresholds"][SAME_WEIGHT_ROLE])
            threshold_error = max(
                threshold_error, abs(candidate_threshold - same_threshold)
            )
            actions[SAME_WEIGHT_ROLE][held_positions] = (
                scores[SAME_WEIGHT_ROLE][held_positions] >= candidate_threshold
            )
    persisted_scores = {
        role: np.asarray(
            [float(row[f"score__{role}"]) for row in rows], dtype=np.float64
        )
        for role in ROLE_NAMES
    }
    persisted_actions = {
        role: np.asarray(
            [bool(int(row[f"action__{role}"])) for row in rows], dtype=np.bool_
        )
        for role in ROLE_NAMES
    }
    score_error = max(
        float(np.max(np.abs(scores[role] - persisted_scores[role])))
        for role in ROLE_NAMES
    )
    actions_exact = all(
        np.array_equal(actions[role], persisted_actions[role]) for role in ROLE_NAMES
    )
    training_payload = json.loads(
        (root / "training_evidence.json").read_text(encoding="utf-8")
    )
    analysis = build_clean_analysis(
        scores=scores,
        actions=actions,
        evidence_maps=evidence_maps,
        targets=targets,
        keeper_predictions=keeper,
        folds=folds,
        valid_masks=valid_masks,
        foreground_masks=foreground_masks,
        training_records=training_payload["records"],
    )
    analysis_error = (
        _recursive_numeric_difference(expected_analysis, analysis)
        if expected_analysis is not None
        else 0.0
    )
    passed = bool(
        threshold_error <= 1e-15
        and score_error <= MAX_REPLAY_ERROR
        and actions_exact
        and analysis_error <= MAX_REPLAY_ERROR
    )
    return {
        "passed": passed,
        "rows": len(rows),
        "threshold_max_abs_error": threshold_error,
        "score_max_abs_error": score_error,
        "actions_exact": actions_exact,
        "analysis_maximum_numeric_difference": analysis_error,
        "analysis": analysis,
    }


def fixed_xai_positions(cohort: Sequence[CleanTrainRow]) -> List[int]:
    positions: List[int] = []
    for fold in FOLDS:
        categories = (
            (
                "keeper_tp",
                lambda row: row.target == FOCUS_CLASS
                and row.keeper_prediction == FOCUS_CLASS,
            ),
            (
                "keeper_fn",
                lambda row: row.target == FOCUS_CLASS
                and row.keeper_prediction != FOCUS_CLASS,
            ),
            (
                "restricted_fp",
                lambda row: row.target != FOCUS_CLASS
                and row.keeper_prediction == FOCUS_CLASS,
            ),
        )
        for _name, predicate in categories:
            candidates = [
                position
                for position, row in enumerate(cohort)
                if row.fold == fold and predicate(row)
            ]
            if not candidates:
                raise RuntimeError(f"Fixed XAI category is empty in fold {fold}")
            positions.append(
                min(candidates, key=lambda position: cohort[position].sample_index)
            )
    if len(positions) != 15 or len(set(positions)) != 15:
        raise RuntimeError("Fixed sparse-evidence XAI selection differs from 15 rows")
    return positions


def _signed_heatmap(value: np.ndarray, *, scale: float, size: int = 256) -> Image.Image:
    normalized = np.asarray(value, dtype=np.float64) / max(float(scale), 1e-12)
    normalized = np.clip(normalized, -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    red = 1.0 - negative
    green = 1.0 - np.maximum(positive, negative)
    blue = 1.0 - positive
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _positive_heatmap(value: np.ndarray, *, scale: float, size: int = 256) -> Image.Image:
    normalized = np.asarray(value, dtype=np.float64) / max(float(scale), 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)
    red = normalized
    green = np.clip(1.5 - np.abs(normalized - 0.5) * 3.0, 0.0, 1.0)
    blue = 1.0 - normalized
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def render_evidence_contact_sheet(
    path: Path,
    *,
    cohort: Sequence[CleanTrainRow],
    rgb_crops: np.ndarray,
    oof: Mapping[str, object],
) -> Dict[str, object]:
    positions = fixed_xai_positions(cohort)
    margins = {
        role: np.asarray(oof["evidence_maps"][role], dtype=np.float32)[:, 1]
        - np.asarray(oof["evidence_maps"][role], dtype=np.float32)[:, 0]
        for role in ROLE_NAMES
    }
    common_scale = max(
        float(np.max(np.abs(margins[role][positions]))) for role in ROLE_NAMES
    )
    columns = ("rgb_crop",) + ROLE_NAMES
    tile = 256
    title_height = 40
    row_height = tile + title_height
    canvas = Image.new(
        "RGB", (tile * len(columns), row_height * len(positions)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    manifest_rows: List[Dict[str, object]] = []
    for row_number, position in enumerate(positions):
        source = cohort[position]
        category = (
            "keeper_tp"
            if source.target == FOCUS_CLASS
            and source.keeper_prediction == FOCUS_CLASS
            else "keeper_fn"
            if source.target == FOCUS_CLASS
            else "restricted_fp"
        )
        images = [Image.fromarray(np.asarray(rgb_crops[position], dtype=np.uint8))]
        images.extend(
            _signed_heatmap(margins[role][position], scale=common_scale)
            for role in ROLE_NAMES
        )
        titles = [
            f"idx={source.sample_index} fold={source.fold} {category}",
            *[
                f"{role} p={float(oof['scores'][role][position]):.4f}"
                for role in ROLE_NAMES
            ],
        ]
        y = row_number * row_height
        for column, (image, title) in enumerate(zip(images, titles)):
            x = column * tile
            draw.text((x + 4, y + 4), title, fill="black", font=font)
            draw.text(
                (x + 4, y + 21), columns[column], fill=(70, 70, 70), font=font
            )
            canvas.paste(image.resize((tile, tile)), (x, y + title_height))
        manifest_rows.append(
            {
                "row": row_number,
                "position": position,
                "sample_index": int(source.sample_index),
                "fold": int(source.fold),
                "target": int(source.target),
                "keeper_prediction": int(source.keeper_prediction),
                "category": category,
                **{
                    f"{role}_margin_map_sha256": _array_sha256(
                        margins[role][position]
                    )
                    for role in ROLE_NAMES
                },
            }
        )
    canvas.save(path)
    manifest_path = path.with_suffix(".json")
    _write_json(
        manifest_path,
        {
            "columns": list(columns),
            "common_signed_scale": common_scale,
            "rows": manifest_rows,
        },
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(positions),
        "positions": positions,
        "columns": list(columns),
        "common_signed_scale": common_scale,
    }


def _feature_input_gradient(
    branch: SparseEvidenceBranch,
    *,
    feature: np.ndarray,
    sample_index: int,
    fold: int,
    dephase: bool,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    branch = branch.to(device).eval()
    branch.zero_grad(set_to_none=True)
    value = torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0).to(device)
    value.requires_grad_(True)
    maps = branch.evidence_maps(
        value,
        sample_indices=[int(sample_index)],
        fold=fold,
        dephase=dephase,
    )
    logits = evidence_logits_torch(maps)
    margin = logits[0, 1] - logits[0, 0]
    margin.backward()
    if value.grad is None or not bool(torch.isfinite(value.grad).all()):
        raise RuntimeError("Sparse evidence feature-input gradient is missing/nonfinite")
    saliency = (value.grad * value).abs().mean(dim=1)[0].detach().cpu().numpy()
    branch.to("cpu")
    return saliency.astype(np.float32), float(margin.detach().cpu())


def render_gradient_contact_sheet(
    path: Path,
    *,
    cohort: Sequence[CleanTrainRow],
    features: np.ndarray,
    rgb_crops: np.ndarray,
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    positions = fixed_xai_positions(cohort)
    gradients: Dict[str, List[np.ndarray]] = {role: [] for role in ROLE_NAMES}
    margins: Dict[str, List[float]] = {role: [] for role in ROLE_NAMES}
    for position in positions:
        fold = int(cohort[position].fold)
        models = oof["fold_records"][fold]["models"]
        for role in ROLE_NAMES:
            source_role = CANDIDATE_ROLE if role == SAME_WEIGHT_ROLE else role
            gradient, margin = _feature_input_gradient(
                models[source_role],
                feature=np.asarray(features[position]),
                sample_index=int(cohort[position].sample_index),
                fold=fold,
                dephase=role in {TRAINED_DEPHASE_ROLE, SAME_WEIGHT_ROLE},
                device=device,
            )
            gradients[role].append(gradient)
            margins[role].append(margin)
    common_scale = max(
        float(np.max(np.asarray(gradients[role]))) for role in ROLE_NAMES
    )
    columns = ("rgb_crop",) + ROLE_NAMES
    tile = 256
    title_height = 40
    row_height = tile + title_height
    canvas = Image.new(
        "RGB", (tile * len(columns), row_height * len(positions)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    manifest_rows: List[Dict[str, object]] = []
    gradient_arrays: Dict[str, np.ndarray] = {
        role: np.stack(values) for role, values in gradients.items()
    }
    for row_number, position in enumerate(positions):
        source = cohort[position]
        images = [Image.fromarray(np.asarray(rgb_crops[position], dtype=np.uint8))]
        images.extend(
            _positive_heatmap(
                gradient_arrays[role][row_number], scale=common_scale
            )
            for role in ROLE_NAMES
        )
        titles = [
            f"idx={source.sample_index} fold={source.fold} target={source.target}",
            *[
                f"{role} margin={margins[role][row_number]:.4f}"
                for role in ROLE_NAMES
            ],
        ]
        y = row_number * row_height
        for column, (image, title) in enumerate(zip(images, titles)):
            x = column * tile
            draw.text((x + 4, y + 4), title, fill="black", font=font)
            draw.text(
                (x + 4, y + 21), columns[column], fill=(70, 70, 70), font=font
            )
            canvas.paste(image.resize((tile, tile)), (x, y + title_height))
        manifest_rows.append(
            {
                "row": row_number,
                "position": position,
                "sample_index": int(source.sample_index),
                **{
                    f"{role}_gradient_sha256": _array_sha256(
                        gradient_arrays[role][row_number]
                    )
                    for role in ROLE_NAMES
                },
            }
        )
    canvas.save(path)
    arrays_path = path.with_suffix(".npz")
    np.savez_compressed(arrays_path, **gradient_arrays)
    manifest_path = path.with_suffix(".json")
    _write_json(
        manifest_path,
        {
            "description": "gradient of candidate margin with respect to captured block-2 feature input",
            "not_raw_image_saliency": True,
            "columns": list(columns),
            "common_positive_scale": common_scale,
            "rows": manifest_rows,
        },
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": _sha256(arrays_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(positions),
        "positions": positions,
        "columns": list(columns),
        "all_finite": all(
            np.isfinite(value).all() for value in gradient_arrays.values()
        ),
    }


class StaticEvidenceExport(nn.Module):
    def __init__(self, branch: SparseEvidenceBranch) -> None:
        super().__init__()
        self.local_block = copy.deepcopy(branch.local_block)
        self.evidence = copy.deepcopy(branch.evidence)

    def forward(self, features: Tensor) -> Tensor:
        maps = self.evidence(self.local_block(features))
        return torch.softmax(maps.mean(dim=(2, 3)), dim=1)


def static_export_checks(
    *,
    output_dir: Path,
    features: np.ndarray,
    oof: Mapping[str, object],
) -> Dict[str, object]:
    fold = 0
    record = oof["fold_records"][fold]
    positions = np.asarray(record["held_positions"], dtype=np.int64)
    wrapper = StaticEvidenceExport(record["models"][CANDIDATE_ROLE]).cpu().eval()
    input_tensor = torch.from_numpy(
        np.asarray(features[positions], dtype=np.float32)
    )
    with torch.inference_mode():
        expected = wrapper(input_tensor).numpy()
    onnx_path = output_dir / "sparse_class_evidence_fold0.onnx"
    torch.onnx.export(
        wrapper,
        input_tensor[:3],
        onnx_path,
        input_names=("block2_features",),
        output_names=("binary_probabilities",),
        dynamic_axes={
            "block2_features": {0: "batch"},
            "binary_probabilities": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(onnx_path))
    domains = sorted({str(node.domain) for node in graph.graph.node})
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    observed = session.run(None, {"block2_features": input_tensor.numpy()})[0]
    probability_error = float(np.max(np.abs(expected - observed)))
    threshold = float(record["thresholds"][CANDIDATE_ROLE])
    expected_actions = expected[:, 1] >= threshold
    observed_actions = observed[:, 1] >= threshold
    actions_exact = bool(np.array_equal(expected_actions, observed_actions))
    checks = {
        "onnx_probability_error_le_1e_5": probability_error <= 1e-5,
        "onnx_actions_exact_at_stored_threshold": actions_exact,
        "onnx_has_no_custom_operator_domain": all(
            domain in {"", "ai.onnx"} for domain in domains
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "probability_max_abs_error": probability_error,
        "actions_exact": actions_exact,
        "operator_domains": domains,
        "path": str(onnx_path.resolve()),
        "bytes": int(onnx_path.stat().st_size),
        "sha256": _sha256(onnx_path),
    }


def numeric_precision_checks(
    *,
    features: np.ndarray,
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    record = oof["fold_records"][0]
    positions = np.asarray(record["held_positions"], dtype=np.int64)[:4]
    value = torch.from_numpy(np.asarray(features[positions], dtype=np.float32))
    source = record["models"][CANDIDATE_ROLE]
    fp32_model = copy.deepcopy(source).cpu().float().eval()
    fp64_model = copy.deepcopy(source).cpu().double().eval()
    with torch.inference_mode():
        fp32_maps = fp32_model.evidence_maps(value.float())
        fp32 = torch.softmax(evidence_logits_torch(fp32_maps), dim=1).numpy()
        fp64_maps = fp64_model.evidence_maps(value.double())
        fp64 = torch.softmax(evidence_logits_torch(fp64_maps), dim=1).numpy()
    bf16_model = copy.deepcopy(source).to(device).float().eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        bf16_maps = bf16_model.evidence_maps(value.to(device))
    bf16 = torch.softmax(
        evidence_logits_torch(bf16_maps.float()), dim=1
    ).cpu().numpy()
    fp32_fp64_error = float(np.max(np.abs(fp32.astype(np.float64) - fp64)))
    bf16_fp32_error = float(np.max(np.abs(bf16 - fp32)))
    gap = np.abs(fp32[:, 1] - fp32[:, 0])
    non_near_tie = gap > 1e-3
    argmax_exact = bool(
        np.array_equal(
            bf16[non_near_tie].argmax(axis=1),
            fp32[non_near_tie].argmax(axis=1),
        )
    )
    checks = {
        "fp32_vs_fp64_probability_error_le_1e_5": fp32_fp64_error <= 1e-5,
        "bf16_vs_fp32_probability_error_le_0_01": bf16_fp32_error <= 0.01,
        "bf16_non_near_tie_argmax_exact": argmax_exact,
    }
    bf16_model.cpu()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rows": len(positions),
        "non_near_tie_rows": int(non_near_tie.sum()),
        "fp32_vs_fp64_probability_max_abs_error": fp32_fp64_error,
        "bf16_vs_fp32_probability_max_abs_error": bf16_fp32_error,
    }


def benchmark_branches(
    *,
    features: np.ndarray,
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    record = oof["fold_records"][0]
    held = np.asarray(record["held_positions"], dtype=np.int64)
    positions = np.resize(held, BATCH_SIZE)
    value = torch.from_numpy(np.asarray(features[positions], dtype=np.float32)).to(device)
    candidate = record["models"][CANDIDATE_ROLE].to(device).eval()
    dense = record["models"][DENSE_ROLE].to(device).eval()

    def measure(model: SparseEvidenceBranch) -> float:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for _ in range(20):
                model(value)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                for _ in range(100):
                    model(value)
            end.record()
            torch.cuda.synchronize(device)
            samples.append(float(start.elapsed_time(end)) / 100.0)
        return float(np.median(samples))

    def peak_extra(model: SparseEvidenceBranch) -> int:
        torch.cuda.empty_cache()
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            model(value)
        torch.cuda.synchronize(device)
        return max(0, int(torch.cuda.max_memory_allocated(device)) - baseline)

    candidate_ms = measure(candidate)
    dense_ms = measure(dense)
    candidate_peak = peak_extra(candidate)
    dense_peak = peak_extra(dense)
    runtime_ratio = candidate_ms / max(dense_ms, 1e-12)
    extra_peak = max(0, candidate_peak - dense_peak)
    candidate.cpu()
    dense.cpu()
    return {
        "batch_size": BATCH_SIZE,
        "iterations_per_sample": 100,
        "samples": 3,
        "candidate_ms": candidate_ms,
        "dense_ms": dense_ms,
        "runtime_ratio": runtime_ratio,
        "candidate_peak_activation_bytes": candidate_peak,
        "dense_peak_activation_bytes": dense_peak,
        "candidate_extra_peak_bytes": extra_peak,
        "checks": {
            "candidate_runtime_ratio_le_1_10": runtime_ratio <= 1.10,
            "candidate_extra_peak_le_0_25_gib": extra_peak <= int(0.25 * 1024**3),
        },
        "passed": runtime_ratio <= 1.10 and extra_peak <= int(0.25 * 1024**3),
    }


def apply_frozen_branches(
    *,
    features: np.ndarray,
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    rows = len(cohort)
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    keeper = np.asarray([row.keeper_prediction for row in cohort], dtype=np.int64)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    sample_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    for fold in FOLDS:
        record = oof["fold_records"][fold]
        held = np.asarray(record["held_positions"], dtype=np.int64)
        for role in ROLE_NAMES:
            source_role = CANDIDATE_ROLE if role == SAME_WEIGHT_ROLE else role
            branch = record["models"][source_role]
            role_scores, _, _ = infer_role(
                branch=branch,
                features=features,
                sample_indices=sample_indices,
                positions=held,
                fold=fold,
                device=device,
                dephase=role in {TRAINED_DEPHASE_ROLE, SAME_WEIGHT_ROLE},
            )
            threshold = float(record["thresholds"][role])
            scores[role][held] = role_scores
            actions[role][held] = role_scores >= threshold
            branch.cpu()
    metrics = {
        role: role_metrics(
            scores=scores[role],
            actions=actions[role],
            targets=targets,
            keeper_predictions=keeper,
            folds=folds,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics[CANDIDATE_ROLE]
    trained_dephase = metrics[TRAINED_DEPHASE_ROLE]
    gates = {
        "candidate_auroc_ge_0_80": float(candidate["auroc"]) >= 0.80,
        "candidate_all_class1_retention_ge_0_92": float(
            candidate["all_class1_retention"]
        )
        >= 0.92,
        "candidate_keeper_tp_retention_ge_0_92": float(
            candidate["keeper_tp_retention"]
        )
        >= 0.92,
        "candidate_restricted_fp_rejection_ge_0_20": float(
            candidate["restricted_fp_rejection"]
        )
        >= 0.20,
        "candidate_auroc_ge_trained_dephase_plus_0_02": float(candidate["auroc"])
        >= float(trained_dephase["auroc"]) + 0.02,
    }
    return {
        "roles": metrics,
        "gates": {key: bool(value) for key, value in gates.items()},
        "passed": all(bool(value) for value in gates.values()),
    }


def _close_delete_feature_cache(extraction: Dict[str, object]) -> Dict[str, object]:
    cache_path = Path(extraction["feature_cache"]["path"])
    cache = extraction.pop("features", None)
    if isinstance(cache, np.memmap) and cache._mmap is not None:
        cache.flush()
        cache._mmap.close()
    del cache
    gc.collect()
    if cache_path.exists():
        cache_path.unlink()
    return {
        "path": str(cache_path),
        "deleted": not cache_path.exists(),
        "sha256_before_deletion": extraction["feature_cache"]["sha256"],
    }


def _gpu_snapshot() -> Dict[str, object]:
    result: Dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "name": torch.cuda.get_device_name(0),
                "allocated_bytes": int(torch.cuda.memory_allocated(0)),
                "reserved_bytes": int(torch.cuda.memory_reserved(0)),
            }
        )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
        result["compute_processes"] = output.splitlines() if output else []
    except (OSError, subprocess.CalledProcessError) as error:
        result["compute_process_query_error"] = str(error)
    return result


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    equations = engineering_checks()
    if not bool(equations["passed"]):
        raise RuntimeError(f"Sparse evidence equation checks failed: {equations}")
    rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    cohort = locked_cohort(rows)
    return {
        "mode": f"{METHOD}_preflight",
        "passed": True,
        "output_created": False,
        "model_loaded": False,
        "dataset_pixels_loaded": False,
        "validation_data_used": False,
        "test_data_used": False,
        "provenance": provenance,
        "equations": equations,
        "cohort": {
            "rows": len(cohort),
            "keeper_tp": EXPECTED_KEEPER_TP,
            "keeper_fn": EXPECTED_KEEPER_FN,
            "restricted_fp": EXPECTED_RESTRICTED_FP,
            "ordered_index_sha256": _cohort_index_sha256(
                [row.sample_index for row in cohort]
            ),
        },
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "implementation_tracking_observed_not_gated_until_formal": (
            implementation_tracking_state()
        ),
    }


def _actual_branch_precision_checks(
    branch: SparseEvidenceBranch,
    *,
    features: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    value = torch.from_numpy(np.asarray(features, dtype=np.float32))
    fp32_model = copy.deepcopy(branch).cpu().float().eval()
    fp64_model = copy.deepcopy(branch).cpu().double().eval()
    with torch.inference_mode():
        fp32 = torch.softmax(fp32_model(value.float()), dim=1).numpy()
        fp64 = torch.softmax(fp64_model(value.double()), dim=1).numpy()
    bf16_model = copy.deepcopy(branch).to(device).float().eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        bf16_maps = bf16_model.evidence_maps(value.to(device))
    bf16 = torch.softmax(
        evidence_logits_torch(bf16_maps.float()), dim=1
    ).cpu().numpy()
    fp32_fp64_error = float(np.max(np.abs(fp32.astype(np.float64) - fp64)))
    bf16_fp32_error = float(np.max(np.abs(bf16 - fp32)))
    non_near_tie = np.abs(fp32[:, 1] - fp32[:, 0]) > 1e-3
    argmax_exact = bool(
        np.array_equal(
            fp32[non_near_tie].argmax(axis=1),
            bf16[non_near_tie].argmax(axis=1),
        )
    )
    bf16_model.cpu()
    checks = {
        "fp32_vs_fp64_probability_error_le_1e_5": fp32_fp64_error <= 1e-5,
        "bf16_vs_fp32_probability_error_le_0_01": bf16_fp32_error <= 0.01,
        "bf16_non_near_tie_argmax_exact": argmax_exact,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fp32_vs_fp64_probability_max_abs_error": fp32_fp64_error,
        "bf16_vs_fp32_probability_max_abs_error": bf16_fp32_error,
        "non_near_tie_rows": int(non_near_tie.sum()),
    }


def engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for sparse evidence engineering")
    provenance = verify_locked_inputs(args)
    device = torch.device("cuda")
    model, checkpoint, _ = load_model(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    full_cohort = locked_cohort(rows)
    positive = next(
        row
        for row in full_cohort
        if row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
    )
    negative = next(row for row in full_cohort if row.target != FOCUS_CLASS)
    cohort = [positive, negative]
    dataset, transform, declaration = _build_dataset(
        checkpoint,
        rows,
        Path(provenance["files"]["data_yaml"]["path"]),
    )
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=2,
        num_workers=NUM_WORKERS,
        context="sparse_class_evidence_a0_engineering",
    )
    with tempfile.TemporaryDirectory(prefix="trkh_sparse_evidence_engineering_") as temporary:
        cache_path = Path(temporary) / "block2_features.npy"
        extraction = extract_condition(
            model=model,
            loader=loader,
            loader_summary=loader_summary,
            cohort=cohort,
            device=device,
            semantics=declaration["semantics"],
            condition="engineering_clean",
            cache_path=cache_path,
            benchmark_ordinary=True,
        )
        local_block = copy.deepcopy(model.stem.blocks[2]).cpu()  # type: ignore[attr-defined]
        branches = build_matched_branches(local_block, fold=0)
        initial_hashes = {
            role: _parameter_sha256(branch) for role, branch in branches.items()
        }
        labels = np.asarray([1, 0], dtype=np.int64)
        indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
        role_evidence: Dict[str, object] = {}
        for role in TRAINED_ROLES:
            result = train_role(
                role=role,
                branch=branches[role],
                features=np.asarray(extraction["features"]),
                sample_indices=indices,
                labels=labels,
                fit_positions=np.asarray([0, 1], dtype=np.int64),
                orders=[np.asarray([0, 1], dtype=np.int64)],
                occurrence_sha256=_array_sha256(
                    np.asarray([0, 1], dtype=np.int64)
                ),
                fold=0,
                device=device,
            )
            branch = result.pop("model")
            role_evidence[role] = result
            branches[role] = branch.cpu()
        precision = _actual_branch_precision_checks(
            branches[CANDIDATE_ROLE],
            features=np.asarray(extraction["features"]),
            device=device,
        )
        extraction_summary = {
            "checks": extraction["checks"],
            "runtime": extraction["runtime"],
            "feature_cache": extraction["feature_cache"],
            "capture_probability_max_abs_error": extraction[
                "capture_probability_max_abs_error"
            ],
            "cidt_probability_max_abs_error": extraction[
                "cidt_probability_max_abs_error"
            ],
        }
        cleanup = _close_delete_feature_cache(extraction)
    passed = bool(
        _protocol_extraction_checks_passed(extraction_summary["checks"])
        and len(set(initial_hashes.values())) == 1
        and all(
            record["gradient_pass"]
            and record["update_pass"]
            and record["calibration_pass"]
            for record in role_evidence.values()
        )
        and precision["passed"]
        and cleanup["deleted"]
    )
    return {
        "mode": f"{METHOD}_engineering_forward",
        "passed": passed,
        "rows": len(cohort),
        "extraction": extraction_summary,
        "matched_initial_state_exact": len(set(initial_hashes.values())) == 1,
        "initial_hashes": initial_hashes,
        "one_step_roles": role_evidence,
        "precision": precision,
        "temporary_cache_cleanup": cleanup,
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
    cleanup: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "condition": name,
        "analysis": analysis,
        "extraction_checks": extraction["checks"],
        "runtime": extraction["runtime"],
        "temporary_cache": {
            **dict(extraction["feature_cache"]),
            **dict(cleanup),
        },
        "passed": bool(
            analysis["passed"]
            and _protocol_extraction_checks_passed(extraction["checks"])
            and cleanup["deleted"]
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_SPARSE_EVIDENCE_A0_PREFLIGHT") != "passed":
        raise RuntimeError(
            "Formal sparse class-evidence A0 must use the locked PowerShell preflight"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal sparse evidence A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    implementation_state = implementation_tracking_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            "Formal sparse evidence A0 requires a clean pushed tracked state: "
            f"{repo_state}"
        )
    if not bool(implementation_state["passed"]):
        raise ValueError(
            "Formal sparse evidence A0 requires committed implementation blobs: "
            f"{implementation_state}"
        )
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    equation_checks = engineering_checks()
    if not bool(equation_checks["passed"]):
        raise RuntimeError("Sparse evidence equation checks changed before formal run")
    device = torch.device("cuda")
    gpu_before = _gpu_snapshot()
    model, checkpoint, class_names = load_model(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    cohort = locked_cohort(rows)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint,
        rows,
        Path(provenance["files"]["data_yaml"]["path"]),
    )
    clean_loader, clean_loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="sparse_class_evidence_a0_clean",
    )
    clean: Optional[Dict[str, object]] = None
    clean_cleanup: Dict[str, object] = {"deleted": False}
    robustness_results: Dict[str, object] = {}
    try:
        clean = extract_condition(
            model=model,
            loader=clean_loader,
            loader_summary=clean_loader_summary,
            cohort=cohort,
            device=device,
            semantics=dataset_summary["semantics"],
            condition="clean",
            cache_path=output_dir / "_temporary_clean_block2_features.npy",
            benchmark_ordinary=True,
        )
        if not _protocol_extraction_checks_passed(clean["checks"]):
            raise RuntimeError(
                f"Sparse evidence clean extraction failed: {clean['checks']}"
            )
        local_block = copy.deepcopy(model.stem.blocks[2]).cpu()  # type: ignore[attr-defined]
        oof = fit_oof_branches(
            features=np.asarray(clean["features"]),
            cohort=cohort,
            local_block=local_block,
            device=device,
        )
        targets = np.asarray([row.target for row in cohort], dtype=np.int64)
        keeper_predictions = np.asarray(
            [row.keeper_prediction for row in cohort], dtype=np.int64
        )
        clean_analysis = build_clean_analysis(
            scores=oof["scores"],
            actions=oof["actions"],
            evidence_maps=oof["evidence_maps"],
            targets=targets,
            keeper_predictions=keeper_predictions,
            folds=np.asarray(oof["folds"]),
            valid_masks=np.asarray(clean["valid_masks"]),
            foreground_masks=np.asarray(clean["foreground_masks"]),
            training_records=oof["training_records"],
        )
        direct_xai = render_evidence_contact_sheet(
            output_dir / "sparse_evidence_maps_contact_sheet.png",
            cohort=cohort,
            rgb_crops=np.asarray(clean["rgb_crops"]),
            oof=oof,
        )
        gradient_xai = render_gradient_contact_sheet(
            output_dir / "sparse_evidence_feature_gradients_contact_sheet.png",
            cohort=cohort,
            features=np.asarray(clean["features"]),
            rgb_crops=np.asarray(clean["rgb_crops"]),
            oof=oof,
            device=device,
        )
        export = static_export_checks(
            output_dir=output_dir,
            features=np.asarray(clean["features"]),
            oof=oof,
        )
        numeric = numeric_precision_checks(
            features=np.asarray(clean["features"]), oof=oof, device=device
        )
        resource = benchmark_branches(
            features=np.asarray(clean["features"]), oof=oof, device=device
        )
        artifacts = save_replay_artifacts(
            output_dir=output_dir,
            cohort=cohort,
            oof=oof,
            valid_masks=np.asarray(clean["valid_masks"]),
            foreground_masks=np.asarray(clean["foreground_masks"]),
            geometry=clean["geometry"],
        )
        in_process_replay = replay_artifacts(
            output_dir, expected_analysis=clean_analysis
        )
        loader_state = clean["runtime"]["loader"]
        structural_gates = {
            "locked_extraction_checks_passed": _protocol_extraction_checks_passed(
                clean["checks"]
            ),
            "equation_checks_passed": bool(equation_checks["passed"]),
            "matched_initial_states_exact_every_fold": all(
                len(set(record["initial_hashes"].values())) == 1
                for record in oof["fold_records"].values()
            ),
            "matched_parameter_counts_every_fold": all(
                len(set(record["parameter_counts"].values())) == 1
                for record in oof["fold_records"].values()
            ),
            "shared_occurrence_order_across_roles": bool(
                oof["shared_occurrences"]
            ),
            "matched_update_counts_across_roles": all(
                len(
                    {
                        int(record["training"][role]["updates"])
                        for role in TRAINED_ROLES
                    }
                )
                == 1
                for record in oof["fold_records"].values()
            ),
            "all_parameters_receive_gradient_and_change": all(
                record["gradient_pass"]
                and record["update_pass"]
                and record["calibration_pass"]
                for record in oof["training_records"]
            ),
            "all_optimizer_states_finite": all(
                record["optimizer"]["all_state_tensors_finite"]
                for record in oof["training_records"]
            ),
            "optimizer_and_parameters_remain_fp32": all(
                record["optimizer"]["state_tensor_dtypes"] == ["torch.float32"]
                and record["parameter_dtypes"] == ["torch.float32"]
                for record in oof["training_records"]
            ),
            "requested_and_effective_workers_4_4": int(
                loader_state["requested_num_workers"]
            )
            == NUM_WORKERS
            and int(loader_state["effective_num_workers"]) == NUM_WORKERS,
            "direct_xai_exact_15_rows": int(direct_xai["rows"]) == 15,
            "feature_gradient_xai_exact_15_rows_and_finite": int(
                gradient_xai["rows"]
            )
            == 15
            and bool(gradient_xai["all_finite"]),
            "static_export_passed": bool(export["passed"]),
            "numeric_precision_passed": bool(numeric["passed"]),
            "resource_gates_passed": bool(resource["passed"]),
            "in_process_exact_replay_passed": bool(in_process_replay["passed"]),
            "temporary_cache_is_hash_attested": bool(
                clean["feature_cache"]["sha256"]
            ),
            "unknown_process_not_terminated": True,
        }
        clean_gate_passed = bool(
            all(structural_gates.values())
            and clean_analysis["mechanism_gates_passed"]
        )
        robustness_passed = False
        if clean_gate_passed:
            for name, brightness, contrast in CONDITIONS:
                condition_loader, condition_loader_summary = _make_condition_loader(
                    base_dataset=dataset,
                    transform=transform,
                    indices=[row.sample_index for row in cohort],
                    brightness=brightness,
                    contrast=contrast,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    context=f"sparse_class_evidence_a0_{name}",
                )
                condition_extraction: Optional[Dict[str, object]] = None
                condition_cleanup: Dict[str, object] = {"deleted": False}
                try:
                    condition_extraction = extract_condition(
                        model=model,
                        loader=condition_loader,
                        loader_summary=condition_loader_summary,
                        cohort=cohort,
                        device=device,
                        semantics=dataset_summary["semantics"],
                        condition=name,
                        cache_path=output_dir
                        / f"_temporary_{name}_block2_features.npy",
                        benchmark_ordinary=False,
                    )
                    condition_analysis = apply_frozen_branches(
                        features=np.asarray(condition_extraction["features"]),
                        cohort=cohort,
                        oof=oof,
                        device=device,
                    )
                finally:
                    if condition_extraction is not None:
                        condition_cleanup = _close_delete_feature_cache(
                            condition_extraction
                        )
                if condition_extraction is None:
                    raise RuntimeError(f"Condition {name} extraction did not complete")
                robustness_results[name] = _condition_summary(
                    name=name,
                    analysis=condition_analysis,
                    extraction=condition_extraction,
                    cleanup=condition_cleanup,
                )
            robustness_passed = all(
                bool(record["passed"]) for record in robustness_results.values()
            )
        else:
            robustness_results = {
                "skipped": True,
                "reason": "clean automatic conjunctive gate failed",
            }
        automated_pre_replay = bool(
            clean_gate_passed
            and robustness_passed
            and direct_xai["rows"] == 15
            and gradient_xai["rows"] == 15
        )
        clean_runtime = dict(clean["runtime"])
        clean_checks = dict(clean["checks"])
        clean_feature_cache = dict(clean["feature_cache"])
        capture_error = float(clean["capture_probability_max_abs_error"])
        cidt_error = float(clean["cidt_probability_max_abs_error"])
    finally:
        if clean is not None and "features" in clean:
            clean_cleanup = _close_delete_feature_cache(clean)
    if not clean_cleanup.get("deleted", False):
        raise RuntimeError(f"Temporary clean cache cleanup failed: {clean_cleanup}")
    gpu_after = _gpu_snapshot()
    summary_path = output_dir / "summary.json"
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
        "implementation_tracking": implementation_state,
        "provenance": provenance,
        "dataset": dataset_summary,
        "cohort": {
            "rows": len(cohort),
            "keeper_tp": EXPECTED_KEEPER_TP,
            "keeper_fn": EXPECTED_KEEPER_FN,
            "restricted_fp": EXPECTED_RESTRICTED_FP,
            "ordered_index_sha256": EXPECTED_ORDERED_INDEX_SHA256,
        },
        "equation_checks": equation_checks,
        "clean_extraction": {
            "checks": clean_checks,
            "runtime": clean_runtime,
            "capture_probability_max_abs_error": capture_error,
            "cidt_probability_max_abs_error": cidt_error,
            "temporary_feature_cache": {
                **clean_feature_cache,
                **clean_cleanup,
                "persisted": False,
            },
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "amp_dtype": "bfloat16",
            "optimizer_state_dtype": "float32",
            "scheduler": None,
            "early_stopping": None,
            "occurrence_hashes": oof["occurrence_hashes"],
            "records": oof["training_records"],
        },
        "clean_analysis": clean_analysis,
        "structural_gates": structural_gates,
        "structural_gates_passed": all(structural_gates.values()),
        "clean_gate_passed": clean_gate_passed,
        "in_process_replay": {
            key: value
            for key, value in in_process_replay.items()
            if key != "analysis"
        },
        "numeric_precision": numeric,
        "static_export": export,
        "resource": {
            **resource,
            "requested_workers": NUM_WORKERS,
            "effective_loader": clean_runtime["loader"],
            "extraction_throughput_images_per_second": clean_runtime[
                "throughput_images_per_second"
            ],
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "unknown_process_terminated": False,
        },
        "robustness_authorized_by_clean_gate": clean_gate_passed,
        "robustness_results": robustness_results,
        "robustness_passed": robustness_passed,
        "xai": {
            "direct_evidence_maps": direct_xai,
            "feature_input_gradients": gradient_xai,
            "automatic_statistics": clean_analysis["map_statistics"],
            "manual_review": {
                "required": True,
                "completed": False,
                "passed": False,
            },
        },
        "artifacts": artifacts,
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


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Replay target is not sparse class-evidence A0")
    replay = replay_artifacts(
        resolved.parent, expected_analysis=summary["clean_analysis"]
    )
    passed = bool(replay["passed"])
    summary["external_replay"] = {
        "required": True,
        "completed": True,
        "second_process": True,
        "passed": passed,
        "rows": replay["rows"],
        "threshold_max_abs_error": replay["threshold_max_abs_error"],
        "score_max_abs_error": replay["score_max_abs_error"],
        "actions_exact": replay["actions_exact"],
        "analysis_maximum_numeric_difference": replay[
            "analysis_maximum_numeric_difference"
        ],
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
        "mode": f"{METHOD}_external_replay",
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
        raise ValueError("Visual-review target is not sparse class-evidence A0")
    if not bool(summary.get("external_replay", {}).get("completed", False)):
        raise ValueError("Sparse evidence visual review requires external replay")
    direct = Path(summary["xai"]["direct_evidence_maps"]["path"])
    gradient = Path(summary["xai"]["feature_input_gradients"]["path"])
    direct_sha = _sha256(direct)
    gradient_sha = _sha256(gradient)
    if direct_sha != summary["xai"]["direct_evidence_maps"]["sha256"]:
        raise ValueError("Direct sparse-evidence XAI sheet differs from lock")
    if gradient_sha != summary["xai"]["feature_input_gradients"]["sha256"]:
        raise ValueError("Feature-gradient XAI sheet differs from lock")
    visual_passed = str(result) == "pass"
    automated_passed = bool(summary["automated_gate_passed"])
    a0_passed = bool(visual_passed and automated_passed)
    review = {
        "required": True,
        "completed": True,
        "passed": visual_passed,
        "decision": str(result),
        "reviewed_summary_sha256": observed_sha,
        "direct_sheet_sha256": direct_sha,
        "gradient_sheet_sha256": gradient_sha,
        "manifest_sha256_before_review": manifest_before["manifest_sha256"],
        "cannot_rescue_automated_failure": True,
    }
    summary["visual_review"] = review
    summary["xai"]["manual_review"] = dict(review)
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
        "mode": f"{METHOD}_visual_review",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.finalize_visual_review:
        if args.replay_summary is None or not args.expected_summary_sha256:
            raise ValueError(
                "Visual finalization requires --replay-summary and "
                "--expected-summary-sha256"
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
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

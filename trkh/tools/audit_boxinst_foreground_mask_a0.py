from __future__ import annotations

"""Prospectively locked train-only BoxInst foreground-mask information gate.

The projection and pairwise equations are implemented independently from
Tian et al., CVPR 2021. AdelaiDet is used only as pinned provenance and is
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
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
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
    _verify_hash,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)


METHOD = "boxinst_foreground_mask_a0"
MODE = "boxinst_foreground_mask_a0_train_information_gate"
SEED = 42
FOCUS_CLASS = 1
FOLDS = (0, 1, 2, 3, 4)
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
FEATURE_CHANNELS = 64
FEATURE_SIZE = 64
MASK_WIDTH = 16
PROJECTION_DIM = 24
DESCRIPTOR_DIM = 105
DESCRIPTOR_WITHOUT_KEEPER_DIM = 100
PAIRWISE_SIZE = 3
PAIRWISE_DILATION = 2
PAIRWISE_COLOR_THRESHOLD = 0.3
DICE_EPSILON = 1e-5
READOUT_C = 0.1
READOUT_MAX_ITER = 2000
MIN_FIT_POSITIVE_RETENTION = 0.97
MAX_REPLAY_ERROR = 1e-7
MAX_CAPTURE_PROBABILITY_ERROR = 1e-6
MAX_CIDT_PROBABILITY_ERROR = 3e-5

TRAINED_ROLES = (
    "projection_aligned",
    "boxinst_aligned",
    "boxinst_affinity_dephased",
)
CANDIDATE_ROLE = "boxinst_aligned"
PROJECTION_ROLE = "projection_aligned"
DEPHASED_ROLE = "boxinst_affinity_dephased"
VALID_ROLE = "valid_uniform"
BBOX_ROLE = "bbox_rectangle"
ROLLED_ROLE = "boxinst_same_weight_mask_rolled"
KEEPER_ROLE = "keeper_logprob"
WITHOUT_KEEPER_ROLE = "boxinst_aligned_without_keeper"
MASK_ROLES = TRAINED_ROLES + (VALID_ROLE, BBOX_ROLE, ROLLED_ROLE)
ROLE_NAMES = MASK_ROLES + (KEEPER_ROLE, WITHOUT_KEEPER_ROLE)

EXPECTED_TRAIN_ROWS = 9215
EXPECTED_TRAIN_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_HOLDOUT_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_TRAIN_INDEX_SHA256 = (
    "d41d14b04dfb70b60b5058f30d1c4cfe0189aef3b0efc0a22741f5672c1473c8"
)
EXPECTED_COHORT_ROWS = 763
EXPECTED_KEEPER_TP = 528
EXPECTED_KEEPER_FN = 13
EXPECTED_RESTRICTED_FP = 222
EXPECTED_RESTRICTED_TARGET_COUNTS = {0: 158, 2: 54, 4: 10}
EXPECTED_COHORT_FOLD_COUNTS = {
    0: {"tp": 107, "fn": 2, "fp": 36},
    1: {"tp": 112, "fn": 3, "fp": 45},
    2: {"tp": 100, "fn": 4, "fp": 48},
    3: {"tp": 101, "fn": 2, "fp": 52},
    4: {"tp": 108, "fn": 2, "fp": 41},
}
EXPECTED_COHORT_INDEX_SHA256 = (
    "59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2"
)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "cb6335e787df64eb624db4abe1c677e16eaf248d8ab74d3b28a5db6ac3643425"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "95b01b2bfaa56f522aa84e7817cbf39ccaa72f01f6572723655a7989552f66e0"
LOCKED_OFFICIAL_COMMIT = "5e19cb172b8363820b409ed1a2754fb19ad3acb8"
LOCKED_OFFICIAL_TREE = "bd7918078c2b7145ca06809b9e5f976f1286a702"
LOCKED_OFFICIAL_LICENSE_SHA256 = "9515f73d1af71fabae85d2992c4c002d7ac8d4240e1ec02afbeeca13f27aee42"
LOCKED_OFFICIAL_MASK_HEAD_SHA256 = "96bc6c06bdfddac254a289f3d93dc6c425aa5e916cf30fe1ea2fdc29c2d962bf"
LOCKED_OFFICIAL_CONDINST_SHA256 = "91559ade3036373f907960ac37a8681afb82c78b6a9805879b4f94ffd68fa2d6"
LOCKED_OFFICIAL_CONFIG_SHA256 = "e64a417214d62e59b7c787044f500206fce8fab618810137f96c5180c306e3be"
LOCKED_OFFICIAL_DEFAULTS_SHA256 = "df0cd255c616a25d4bf6535258067bbb6ea2b1870ec8d014562b9ec435a048f9"
LOCKED_OFFICIAL_README_SHA256 = "86a1f7582f1c62ac8b5f3afcf5c6f1986ea23d561710622c182705362b76c9a4"

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\AdelaiDet")
PAPER_PATH = Path(r"D:\DataAI\external_sources\papers\Tian_BoxInst_CVPR_2021.pdf")
CONDITIONS = (
    ("dim", 0.70, 0.90),
    ("bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
IMPLEMENTATION_PATHS = (
    "trkh/tools/audit_boxinst_foreground_mask_a0.py",
    "tests/test_audit_boxinst_foreground_mask_a0.py",
    "scripts/run_trkh_boxinst_foreground_mask_a0.ps1",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only BoxInst foreground-mask A0 gate."
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
        default=REPO_ROOT / "docs" / "TRKH_5CLASS_BOXINST_FOREGROUND_MASK_A0_PROTOCOL_20260721.md",
    )
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_boxinst_foreground_mask_a0_20260721",
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
        raise ValueError("BoxInst foreground-mask A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"BoxInst foreground-mask A0 locks batch-size={BATCH_SIZE}, "
            f"num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"BoxInst foreground-mask A0 locks seed={SEED}")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
    ).strip()


def implementation_tracking_state() -> Dict[str, object]:
    records: Dict[str, object] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        try:
            _git_value(REPO_ROOT, "ls-files", "--error-unmatch", relative)
            head_blob = _git_value(REPO_ROOT, "rev-parse", f"HEAD:{relative}")
            worktree_blob = _git_value(REPO_ROOT, "hash-object", str(path))
            tracked = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            tracked = False
            head_blob = ""
            worktree_blob = ""
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
    official = Path(args.official_root).expanduser().resolve()
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
        "protocol": _verify_hash(
            args.protocol, LOCKED_PROTOCOL_SHA256, "BoxInst A0 protocol"
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
        "paper": _verify_hash(PAPER_PATH, LOCKED_PAPER_SHA256, "BoxInst paper"),
        "official_license": _verify_hash(
            official / "LICENSE", LOCKED_OFFICIAL_LICENSE_SHA256, "AdelaiDet license"
        ),
        "official_mask_head": _verify_hash(
            official / "adet" / "modeling" / "condinst" / "dynamic_mask_head.py",
            LOCKED_OFFICIAL_MASK_HEAD_SHA256,
            "AdelaiDet dynamic mask head",
        ),
        "official_condinst": _verify_hash(
            official / "adet" / "modeling" / "condinst" / "condinst.py",
            LOCKED_OFFICIAL_CONDINST_SHA256,
            "AdelaiDet CondInst",
        ),
        "official_config": _verify_hash(
            official / "configs" / "BoxInst" / "Base-BoxInst.yaml",
            LOCKED_OFFICIAL_CONFIG_SHA256,
            "AdelaiDet BoxInst config",
        ),
        "official_defaults": _verify_hash(
            official / "adet" / "config" / "defaults.py",
            LOCKED_OFFICIAL_DEFAULTS_SHA256,
            "AdelaiDet defaults",
        ),
        "official_readme": _verify_hash(
            official / "README.md", LOCKED_OFFICIAL_README_SHA256, "AdelaiDet README"
        ),
    }
    commit = _git_value(official, "rev-parse", "HEAD")
    tree = _git_value(official, "rev-parse", "HEAD^{tree}")
    status = _git_value(official, "status", "--porcelain", "--untracked-files=no")
    if commit != LOCKED_OFFICIAL_COMMIT or tree != LOCKED_OFFICIAL_TREE or status:
        raise ValueError(
            "Pinned AdelaiDet repository differs: "
            f"commit={commit}, tree={tree}, status={status!r}"
        )
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payloads differ from lock: {protected}")
    return {
        "files": files,
        "official_repository": {
            "path": str(official),
            "remote": _git_value(official, "remote", "get-url", "origin"),
            "commit": commit,
            "tree": tree,
            "tracked_worktree_clean": status == "",
            "source_imported": False,
        },
        "protected_untracked": protected,
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def locked_training_rows(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    selected = list(rows)
    if len(selected) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"BoxInst train rows differ: {len(selected)}")
    counts = tuple(sum(row.target == index for row in selected) for index in range(5))
    if counts != EXPECTED_TRAIN_CLASS_COUNTS:
        raise ValueError(f"BoxInst train class counts differ: {counts}")
    holdouts = tuple(sum(row.fold == fold for row in selected) for fold in FOLDS)
    if holdouts != EXPECTED_HOLDOUT_COUNTS:
        raise ValueError(f"BoxInst holdout counts differ: {holdouts}")
    index_hash = _index_sha256([row.sample_index for row in selected])
    if index_hash != EXPECTED_TRAIN_INDEX_SHA256:
        raise ValueError(f"BoxInst train index order differs: {index_hash}")
    source_folds: Dict[str, set[int]] = {}
    for row in selected:
        source_folds.setdefault(row.source_stem, set()).add(int(row.fold))
    overlap = {key: value for key, value in source_folds.items() if len(value) > 1}
    if overlap:
        raise ValueError(f"BoxInst train source overlap: {len(overlap)} sources")
    return selected


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
    observed = (
        len(cohort),
        sum(row.target == 1 and row.keeper_prediction == 1 for row in cohort),
        sum(row.target == 1 and row.keeper_prediction != 1 for row in cohort),
        sum(row.target != 1 for row in cohort),
    )
    expected = (
        EXPECTED_COHORT_ROWS,
        EXPECTED_KEEPER_TP,
        EXPECTED_KEEPER_FN,
        EXPECTED_RESTRICTED_FP,
    )
    if observed != expected:
        raise ValueError(f"BoxInst cohort differs: {observed} != {expected}")
    target_counts = {
        target: sum(row.target == target for row in cohort)
        for target in EXPECTED_RESTRICTED_TARGET_COUNTS
    }
    if target_counts != EXPECTED_RESTRICTED_TARGET_COUNTS:
        raise ValueError(f"BoxInst restricted target counts differ: {target_counts}")
    fold_counts: Dict[int, Dict[str, int]] = {}
    for fold in FOLDS:
        fold_rows = [row for row in cohort if row.fold == fold]
        fold_counts[fold] = {
            "tp": sum(row.target == 1 and row.keeper_prediction == 1 for row in fold_rows),
            "fn": sum(row.target == 1 and row.keeper_prediction != 1 for row in fold_rows),
            "fp": sum(row.target != 1 for row in fold_rows),
        }
    if fold_counts != EXPECTED_COHORT_FOLD_COUNTS:
        raise ValueError(f"BoxInst cohort fold counts differ: {fold_counts}")
    if _index_sha256([row.sample_index for row in cohort]) != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError("BoxInst cohort order hash differs")
    return cohort


def dice_loss_torch(prediction: Tensor, target: Tensor, *, epsilon: float = DICE_EPSILON) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("Dice tensors must have equal [B,...] shapes")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Dice tensors must be floating point")
    pred = prediction.reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    intersection = (pred * truth).sum(dim=1)
    denominator = pred.square().sum(dim=1) + truth.square().sum(dim=1) + float(epsilon)
    return 1.0 - 2.0 * intersection / denominator


def dice_loss_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float = DICE_EPSILON,
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim < 2:
        raise ValueError("Dice oracle arrays must have equal [B,...] shapes")
    pred = pred.reshape(pred.shape[0], -1)
    truth = truth.reshape(truth.shape[0], -1)
    intersection = np.sum(pred * truth, axis=1)
    denominator = np.sum(pred * pred, axis=1) + np.sum(truth * truth, axis=1) + float(epsilon)
    return 1.0 - 2.0 * intersection / denominator


def projection_loss_torch(mask: Tensor, box_mask: Tensor) -> Tensor:
    value = _mask_3d(mask, "prediction")
    target = _mask_3d(box_mask, "box")
    if value.shape != target.shape:
        raise ValueError("Projection prediction and box masks must align")
    loss_x = dice_loss_torch(value.amax(dim=1), target.amax(dim=1))
    loss_y = dice_loss_torch(value.amax(dim=2), target.amax(dim=2))
    return loss_x + loss_y


def projection_loss_numpy(mask: np.ndarray, box_mask: np.ndarray) -> np.ndarray:
    value = _mask_3d_numpy(mask, "prediction")
    target = _mask_3d_numpy(box_mask, "box")
    if value.shape != target.shape:
        raise ValueError("Projection oracle prediction and box masks must align")
    loss_x = dice_loss_numpy(np.max(value, axis=1), np.max(target, axis=1))
    loss_y = dice_loss_numpy(np.max(value, axis=2), np.max(target, axis=2))
    return loss_x + loss_y


def pairwise_same_probability_torch(center: Tensor, neighbor: Tensor) -> Tensor:
    if center.shape != neighbor.shape:
        raise ValueError("Pairwise tensors must align")
    return center * neighbor + (1.0 - center) * (1.0 - neighbor)


def pairwise_same_probability_numpy(center: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
    left = np.asarray(center, dtype=np.float64)
    right = np.asarray(neighbor, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Pairwise oracle arrays must align")
    return left * right + (1.0 - left) * (1.0 - right)


def _mask_3d(value: Tensor, label: str) -> Tensor:
    result = value[:, 0] if value.ndim == 4 and value.shape[1] == 1 else value
    if result.ndim != 3:
        raise ValueError(f"{label} mask must have shape [B,H,W] or [B,1,H,W]")
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} mask must be a finite floating tensor")
    return result


def _mask_3d_numpy(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 4 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError(f"{label} oracle mask must be finite [B,H,W]")
    return result


def _neighbor_stack_torch(value: Tensor, *, dilation: int = PAIRWISE_DILATION) -> Tensor:
    if value.ndim != 4:
        raise ValueError("Neighbor extraction requires [B,C,H,W]")
    kernel = 3
    unfolded = F.unfold(value, kernel_size=kernel, dilation=int(dilation), padding=int(dilation))
    batch, channels, height, width = value.shape
    unfolded = unfolded.view(batch, channels, kernel * kernel, height, width)
    return torch.cat((unfolded[:, :, :4], unfolded[:, :, 5:]), dim=2)


def pairwise_affinity_torch(
    mask: Tensor,
    lab: Tensor,
    valid_mask: Tensor,
    box_mask: Tensor,
    *,
    color_threshold: float = PAIRWISE_COLOR_THRESHOLD,
    dilation: int = PAIRWISE_DILATION,
) -> Tuple[Tensor, Tensor, Tensor]:
    value = _mask_3d(mask, "prediction")
    if lab.ndim != 4 or lab.shape[1] != 3 or tuple(lab.shape[2:]) != tuple(value.shape[1:]):
        raise ValueError("LAB maps must align as [B,3,H,W]")
    valid = _mask_3d(valid_mask.float(), "valid") > 0.5
    box = _mask_3d(box_mask.float(), "box") > 0.5
    if valid.shape != value.shape or box.shape != value.shape:
        raise ValueError("Pairwise geometry masks must align")
    neighbors = _neighbor_stack_torch(value[:, None], dilation=dilation)[:, 0]
    neighbor_lab = _neighbor_stack_torch(lab, dilation=dilation)
    center_lab = lab[:, :, None]
    similarity = torch.exp(-torch.linalg.vector_norm(center_lab - neighbor_lab, dim=1) / 2.0)
    geometric_neighbor = (
        _neighbor_stack_torch(torch.ones_like(value[:, None]), dilation=dilation)[:, 0]
        > 0.5
    )
    similarity = similarity * geometric_neighbor.to(similarity.dtype)
    neighbor_valid = _neighbor_stack_torch(valid[:, None].float(), dilation=dilation)[:, 0] > 0.5
    weights = (
        (similarity >= float(color_threshold))
        & valid[:, None]
        & neighbor_valid
        & box[:, None]
    )
    centers = value[:, None].expand_as(neighbors)
    same = pairwise_same_probability_torch(centers, neighbors).clamp_min(1e-12)
    losses = -torch.log(same)
    denominator = weights.sum(dim=(1, 2, 3)).clamp_min(1)
    per_sample = (losses * weights).sum(dim=(1, 2, 3)) / denominator
    return per_sample, similarity, weights


def _neighbor_offsets(dilation: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (dy * int(dilation), dx * int(dilation))
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    )


def pairwise_affinity_numpy(
    mask: np.ndarray,
    lab: np.ndarray,
    valid_mask: np.ndarray,
    box_mask: np.ndarray,
    *,
    color_threshold: float = PAIRWISE_COLOR_THRESHOLD,
    dilation: int = PAIRWISE_DILATION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = _mask_3d_numpy(mask, "prediction")
    colors = np.asarray(lab, dtype=np.float64)
    valid = _mask_3d_numpy(valid_mask, "valid") > 0.5
    box = _mask_3d_numpy(box_mask, "box") > 0.5
    if colors.shape != (value.shape[0], 3, value.shape[1], value.shape[2]):
        raise ValueError("LAB oracle arrays must align")
    batch, height, width = value.shape
    similarities = np.zeros((batch, 8, height, width), dtype=np.float64)
    weights = np.zeros_like(similarities, dtype=np.bool_)
    losses = np.zeros_like(similarities)
    for slot, (dy, dx) in enumerate(_neighbor_offsets(dilation)):
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        ny0, ny1 = y0 + dy, y1 + dy
        nx0, nx1 = x0 + dx, x1 + dx
        center = value[:, y0:y1, x0:x1]
        neighbor = value[:, ny0:ny1, nx0:nx1]
        delta = colors[:, :, y0:y1, x0:x1] - colors[:, :, ny0:ny1, nx0:nx1]
        similarity = np.exp(-np.linalg.norm(delta, axis=1) / 2.0)
        edge = (
            (similarity >= float(color_threshold))
            & valid[:, y0:y1, x0:x1]
            & valid[:, ny0:ny1, nx0:nx1]
            & box[:, y0:y1, x0:x1]
        )
        same = np.clip(pairwise_same_probability_numpy(center, neighbor), 1e-12, None)
        similarities[:, slot, y0:y1, x0:x1] = similarity
        weights[:, slot, y0:y1, x0:x1] = edge
        losses[:, slot, y0:y1, x0:x1] = -np.log(same)
    denominator = np.maximum(weights.sum(axis=(1, 2, 3)), 1)
    per_sample = (losses * weights).sum(axis=(1, 2, 3)) / denominator
    return per_sample, similarities, weights


def srgb_to_lab_torch(rgb: Tensor) -> Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("sRGB input must have shape [B,3,H,W]")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()):
        raise ValueError("sRGB input must be finite floating point")
    value = rgb.clamp(0.0, 1.0)
    linear = torch.where(
        value > 0.04045,
        ((value + 0.055) / 1.055).pow(2.4),
        value / 12.92,
    )
    matrix = linear.new_tensor(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    xyz = xyz / xyz.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    delta = 6.0 / 29.0
    transformed = torch.where(
        xyz > delta**3,
        torch.pow(xyz.clamp_min(0.0), 1.0 / 3.0),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    x, y, z = transformed.unbind(dim=1)
    return torch.stack((116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)), dim=1)


def srgb_to_lab_numpy(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    if value.ndim != 4 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("sRGB oracle input must be finite [B,3,H,W]")
    value = np.clip(value, 0.0, 1.0)
    linear = np.where(
        value > 0.04045,
        ((value + 0.055) / 1.055) ** 2.4,
        value / 12.92,
    )
    matrix = np.asarray(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        dtype=np.float64,
    )
    xyz = np.einsum("ij,bjhw->bihw", matrix, linear)
    xyz = xyz / np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)[None, :, None, None]
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(np.maximum(xyz, 0.0)),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]
    return np.stack((116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)), axis=1)


def bbox_coordinate_channels(boxes: Tensor, *, size: int = FEATURE_SIZE) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError("Bboxes must have shape [B,4+]")
    value = boxes[:, :4].float().clamp(0.0, 1.0)
    cx, cy, width, height = value.unbind(dim=1)
    yy = (torch.arange(size, device=boxes.device, dtype=torch.float32) + 0.5) / float(size)
    xx = (torch.arange(size, device=boxes.device, dtype=torch.float32) + 0.5) / float(size)
    dx = (xx.view(1, 1, size) - cx.view(-1, 1, 1)) / (0.5 * width.clamp_min(1e-6)).view(-1, 1, 1)
    dy = (yy.view(1, size, 1) - cy.view(-1, 1, 1)) / (0.5 * height.clamp_min(1e-6)).view(-1, 1, 1)
    return torch.stack((dx.expand(-1, size, -1), dy.expand(-1, -1, size)), dim=1).clamp(-2.0, 2.0)


def normalized_bbox_masks(boxes: Tensor, *, size: int = FEATURE_SIZE) -> Tensor:
    coordinates = bbox_coordinate_channels(boxes, size=size)
    return (coordinates[:, 0].abs() <= 1.0) & (coordinates[:, 1].abs() <= 1.0)


def _roll_offsets(
    sample_indices: Sequence[int],
    *,
    fold: int,
    purpose: str,
    size: int = FEATURE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    vertical = np.empty(len(sample_indices), dtype=np.int64)
    horizontal = np.empty_like(vertical)
    for row, sample_index in enumerate(sample_indices):
        digest = hashlib.sha256(
            f"{METHOD}:{purpose}:{SEED}:{int(fold)}:{int(sample_index)}".encode("ascii")
        ).digest()
        dy = int.from_bytes(digest[:8], "little") % int(size)
        dx = int.from_bytes(digest[8:16], "little") % int(size)
        if dy == 0 and dx == 0:
            dx = 1
        vertical[row] = dy
        horizontal[row] = dx
    return vertical, horizontal


def common_toroidal_roll(
    value: Tensor,
    sample_indices: Sequence[int],
    *,
    fold: int,
    purpose: str,
) -> Tensor:
    if value.ndim not in (3, 4) or tuple(value.shape[-2:]) != (FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError("Toroidal roll expects [B,H,W] or [B,C,H,W] at 64x64")
    if len(sample_indices) != value.shape[0]:
        raise ValueError("Toroidal roll indices do not align")
    had_channel = value.ndim == 4
    source = value if had_channel else value[:, None]
    dy_np, dx_np = _roll_offsets(sample_indices, fold=fold, purpose=purpose)
    dy = torch.from_numpy(dy_np).to(value.device)
    dx = torch.from_numpy(dx_np).to(value.device)
    yy = torch.arange(FEATURE_SIZE, device=value.device).view(1, -1, 1)
    xx = torch.arange(FEATURE_SIZE, device=value.device).view(1, 1, -1)
    source_y = (yy - dy.view(-1, 1, 1)) % FEATURE_SIZE
    source_x = (xx - dx.view(-1, 1, 1)) % FEATURE_SIZE
    gather = (source_y * FEATURE_SIZE + source_x).view(value.shape[0], 1, -1)
    gather = gather.expand(-1, source.shape[1], -1)
    result = torch.gather(source.flatten(2), 2, gather).reshape_as(source)
    return result if had_channel else result[:, 0]


class BoxInstMaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(FEATURE_CHANNELS + 2, MASK_WIDTH, kernel_size=1)
        self.norm = nn.GroupNorm(4, MASK_WIDTH)
        self.conv2 = nn.Conv2d(MASK_WIDTH, MASK_WIDTH, kernel_size=1)
        self.output = nn.Conv2d(MASK_WIDTH, 1, kernel_size=1)

    def logits(self, features: Tensor, boxes: Tensor) -> Tensor:
        if features.ndim != 4 or tuple(features.shape[1:]) != (
            FEATURE_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ):
            raise ValueError("BoxInst mask head expects [B,64,64,64]")
        coordinates = bbox_coordinate_channels(boxes, size=FEATURE_SIZE).to(features.dtype)
        value = torch.cat((features, coordinates), dim=1)
        value = F.gelu(self.norm(self.conv1(value)))
        value = F.gelu(self.conv2(value))
        return self.output(value)

    def forward(self, features: Tensor, boxes: Tensor) -> Tensor:
        return self.logits(features, boxes).sigmoid()


def _parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_matched_heads(*, fold: int) -> Dict[str, BoxInstMaskHead]:
    before = _global_rng_snapshot()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED + int(fold))
        base = BoxInstMaskHead().cpu()
    heads = {role: copy.deepcopy(base) for role in TRAINED_ROLES}
    after = _global_rng_snapshot()
    if not _global_rng_equal(before, after):
        raise RuntimeError("Matched BoxInst head construction changed global RNG")
    if len({_parameter_sha256(head) for head in heads.values()}) != 1:
        raise RuntimeError("Matched BoxInst head initial states differ")
    return heads


def fixed_projection() -> np.ndarray:
    generator = np.random.default_rng(SEED)
    raw = generator.standard_normal((FEATURE_CHANNELS, PROJECTION_DIM), dtype=np.float64)
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return (q * signs[None, :]).astype(np.float64)


def build_descriptors(
    projected_features: np.ndarray,
    masks: Mapping[str, np.ndarray],
    valid_masks: np.ndarray,
    bbox_masks: np.ndarray,
    keeper_probabilities: np.ndarray,
) -> Dict[str, np.ndarray]:
    features = np.asarray(projected_features, dtype=np.float64)
    valid = np.asarray(valid_masks, dtype=np.bool_)
    bbox = np.asarray(bbox_masks, dtype=np.bool_)
    keeper = np.asarray(keeper_probabilities, dtype=np.float64)
    rows = features.shape[0]
    if features.shape != (rows, PROJECTION_DIM, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError("Projected features must be [N,24,64,64]")
    if valid.shape != (rows, FEATURE_SIZE, FEATURE_SIZE) or bbox.shape != valid.shape:
        raise ValueError("Descriptor geometry masks do not align")
    if keeper.shape != (rows, 5):
        raise ValueError("Keeper probabilities must be [N,5]")
    valid_weight = valid.astype(np.float64)
    valid_denominator = valid_weight.sum(axis=(1, 2)).clip(min=1.0)
    valid_mean = np.einsum("nchw,nhw->nc", features, valid_weight) / valid_denominator[:, None]
    valid_variance = (
        np.einsum(
            "nchw,nhw->nc",
            (features - valid_mean[:, :, None, None]) ** 2,
            valid_weight,
        )
        / valid_denominator[:, None]
    )
    valid_std = np.sqrt(np.maximum(valid_variance, 0.0))
    log_keeper = np.log(np.clip(keeper, 1e-7, 1.0))
    outputs: Dict[str, np.ndarray] = {}
    for role in MASK_ROLES:
        mask = np.asarray(masks[role], dtype=np.float64)
        if mask.shape != valid.shape or not np.isfinite(mask).all():
            raise ValueError(f"Mask role {role} does not align")
        mask = np.clip(mask, 0.0, 1.0) * valid_weight
        denominator = mask.sum(axis=(1, 2)).clip(min=1e-12)
        mean = np.einsum("nchw,nhw->nc", features, mask) / denominator[:, None]
        variance = (
            np.einsum(
                "nchw,nhw->nc",
                (features - mean[:, :, None, None]) ** 2,
                mask,
            )
            / denominator[:, None]
        )
        std = np.sqrt(np.maximum(variance, 0.0))
        bbox_area = bbox.sum(axis=(1, 2)).clip(min=1)
        probability = np.clip(mask, 1e-7, 1.0 - 1e-7)
        entropy = -(
            probability * np.log(probability)
            + (1.0 - probability) * np.log(1.0 - probability)
        )
        stats = np.stack(
            (
                denominator / valid_denominator,
                denominator / bbox_area,
                (entropy * valid_weight).sum(axis=(1, 2)) / valid_denominator,
                (mask * (~bbox)).sum(axis=(1, 2)) / denominator,
            ),
            axis=1,
        )
        base = np.concatenate((mean, std, valid_mean, valid_std, stats), axis=1)
        descriptor = np.concatenate((base, log_keeper), axis=1)
        if descriptor.shape != (rows, DESCRIPTOR_DIM):
            raise RuntimeError(f"Descriptor role {role} has wrong shape: {descriptor.shape}")
        outputs[role] = descriptor
        if role == CANDIDATE_ROLE:
            outputs[WITHOUT_KEEPER_ROLE] = base
    outputs[KEEPER_ROLE] = log_keeper
    if set(outputs) != set(ROLE_NAMES):
        raise RuntimeError("Descriptor roles differ from protocol")
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise RuntimeError("Descriptor bank contains non-finite values")
    return outputs


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(values[binary == 1])
    if positives.size == 0:
        raise ValueError("Readout fit fold lacks positives")
    allowed_breaks = int(math.floor((1.0 - MIN_FIT_POSITIVE_RETENTION) * positives.size))
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    if float((positives >= threshold).mean()) + 1e-12 < MIN_FIT_POSITIVE_RETENTION:
        raise RuntimeError("Selected readout threshold violates positive retention")
    return threshold


def _positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_, dtype=np.int64)
    position = np.flatnonzero(classes == 1)
    if position.size != 1:
        raise ValueError("Binary readout lacks one positive class")
    return np.asarray(model.predict_proba(features)[:, int(position[0])], dtype=np.float64)


def fit_oof_readouts(
    descriptors: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    source_stems: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    if set(descriptors) != set(ROLE_NAMES):
        raise ValueError("OOF descriptor roles differ from protocol")
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    sources = np.asarray([str(value) for value in source_stems], dtype=str)
    rows = binary.size
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    states: Dict[str, object] = {role: {"folds": []} for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        values = np.asarray(descriptors[role], dtype=np.float64)
        for held_fold in FOLDS:
            held = fold_values == int(held_fold)
            fit = ~held
            overlap = set(sources[held]).intersection(set(sources[fit]))
            if overlap:
                raise ValueError(f"Readout fold {held_fold} has source overlap")
            scaler = StandardScaler().fit(values[fit])
            fit_values = scaler.transform(values[fit])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(
                    C=READOUT_C,
                    penalty="l2",
                    class_weight=None,
                    max_iter=READOUT_MAX_ITER,
                    tol=1e-8,
                    random_state=SEED,
                    solver="lbfgs",
                )
                model.fit(fit_values, binary[fit])
            fit_scores = _positive_probability(model, fit_values)
            threshold = _positive_threshold(fit_scores, binary[fit])
            held_scores = _positive_probability(model, scaler.transform(values[held]))
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= threshold
            convergence = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            states[role]["folds"].append(
                {
                    "held_fold": int(held_fold),
                    "fit_rows": int(fit.sum()),
                    "held_rows": int(held.sum()),
                    "source_overlap": 0,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "coefficient": model.coef_.tolist(),
                    "intercept": model.intercept_.tolist(),
                    "classes": model.classes_.astype(int).tolist(),
                    "threshold": threshold,
                    "iterations": model.n_iter_.astype(int).tolist(),
                    "convergence_warnings": convergence,
                    "converged": not convergence and int(np.max(model.n_iter_)) < READOUT_MAX_ITER,
                    "fit_positive_retention": float(
                        (fit_scores[binary[fit] == 1] >= threshold).mean()
                    ),
                }
            )
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("OOF readout left non-finite scores")
    return scores, actions, states


def apply_readout_states(
    descriptors: Mapping[str, np.ndarray],
    folds: np.ndarray,
    states: Mapping[str, object],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    rows = fold_values.size
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        values = np.asarray(descriptors[role], dtype=np.float64)
        for record in states[role]["folds"]:
            held = fold_values == int(record["held_fold"])
            mean = np.asarray(record["scaler_mean"], dtype=np.float64)
            scale = np.asarray(record["scaler_scale"], dtype=np.float64)
            coefficient = np.asarray(record["coefficient"], dtype=np.float64)
            intercept = np.asarray(record["intercept"], dtype=np.float64)
            transformed = (values[held] - mean) / scale
            decision = transformed @ coefficient.T + intercept
            held_scores = 1.0 / (1.0 + np.exp(-decision[:, 0]))
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= float(record["threshold"])
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("Persisted readout states left non-finite scores")
    return scores, actions


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
    targets = np.asarray(targets, dtype=np.int64)
    keeper = np.asarray(keeper_predictions, dtype=np.int64)
    fold_values = np.asarray(folds, dtype=np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    positive = labels == 1
    negative = ~positive
    keeper_tp = positive & (keeper == FOCUS_CLASS)
    keeper_fn = positive & (keeper != FOCUS_CLASS)
    true_accepts = int(np.sum(accepted & positive))
    accepted_count = int(accepted.sum())
    fold_metrics = {}
    for fold in FOLDS:
        selected = fold_values == fold
        fold_metrics[str(fold)] = {
            "auroc": float(roc_auc_score(labels[selected], values[selected])),
            "auprc": float(average_precision_score(labels[selected], values[selected])),
            "positive_retention": float(accepted[selected & positive].mean()),
            "restricted_fp_rejection": float((~accepted[selected & negative]).mean()),
        }
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
        "all_positive_retention": float(accepted[positive].mean()),
        "keeper_tp_retention": float(accepted[keeper_tp].mean()),
        "keeper_tp_broken": int(np.sum(~accepted & keeper_tp)),
        "keeper_fn_supported": int(np.sum(accepted & keeper_fn)),
        "restricted_fp_rejected": int(np.sum(~accepted & negative)),
        "restricted_fp_rejection": float((~accepted[negative]).mean()),
        "precision": float(true_accepts / max(accepted_count, 1)),
        "accepted": accepted_count,
        "folds": fold_metrics,
    }


def _periodic_affinity_multiset(lab: Tensor, *, dilation: int = PAIRWISE_DILATION) -> Tensor:
    values = []
    for dy, dx in _neighbor_offsets(dilation):
        neighbor = torch.roll(lab, shifts=(-dy, -dx), dims=(-2, -1))
        values.append(torch.exp(-torch.linalg.vector_norm(lab - neighbor, dim=1) / 2.0))
    return torch.stack(values, dim=1)


def engineering_checks() -> Dict[str, object]:
    from skimage.color import rgb2lab

    generator = np.random.default_rng(1209)
    mask = generator.uniform(0.01, 0.99, size=(3, 7, 9)).astype(np.float64)
    box = np.zeros_like(mask)
    box[:, 1:6, 2:8] = 1.0
    valid = np.ones_like(mask)
    valid[:, :1, :] = 0.0
    lab = generator.normal(size=(3, 3, 7, 9)).astype(np.float64)
    torch_projection = projection_loss_torch(
        torch.from_numpy(mask), torch.from_numpy(box)
    ).numpy()
    numpy_projection = projection_loss_numpy(mask, box)
    projection_error = float(np.max(np.abs(torch_projection - numpy_projection)))
    torch_pair, torch_similarity, torch_weights = pairwise_affinity_torch(
        torch.from_numpy(mask),
        torch.from_numpy(lab),
        torch.from_numpy(valid),
        torch.from_numpy(box),
    )
    numpy_pair, numpy_similarity, numpy_weights = pairwise_affinity_numpy(mask, lab, valid, box)
    pairwise_error = float(np.max(np.abs(torch_pair.numpy() - numpy_pair)))
    similarity_error = float(np.max(np.abs(torch_similarity.numpy() - numpy_similarity)))
    weights_exact = bool(np.array_equal(torch_weights.numpy(), numpy_weights))

    colors = np.concatenate(
        (
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            generator.uniform(size=(32, 3)),
        ),
        axis=0,
    )
    rgb = colors.T.reshape(1, 3, 1, colors.shape[0])
    observed_lab = srgb_to_lab_torch(torch.from_numpy(rgb)).numpy()
    oracle_lab = rgb2lab(colors.reshape(1, colors.shape[0], 3)).transpose(2, 0, 1)[None]
    lab_error = float(np.max(np.abs(observed_lab - oracle_lab)))
    numpy_lab_error = float(np.max(np.abs(srgb_to_lab_numpy(rgb) - oracle_lab)))

    complement = 1.0 - mask
    same = pairwise_same_probability_numpy(mask, np.roll(mask, 1, axis=2))
    complement_same = pairwise_same_probability_numpy(
        complement, np.roll(complement, 1, axis=2)
    )
    complement_error = float(np.max(np.abs(same - complement_same)))
    symmetry_error = float(
        np.max(
            np.abs(
                dice_loss_numpy(mask, box)
                - dice_loss_numpy(box, mask)
            )
        )
    )

    feature = torch.randn(2, 3, FEATURE_SIZE, FEATURE_SIZE, generator=torch.Generator().manual_seed(8))
    indices = [17, 91]
    rng_before = _global_rng_snapshot()
    rolled_a = common_toroidal_roll(feature, indices, fold=2, purpose="affinity")
    rolled_b = common_toroidal_roll(feature, indices, fold=2, purpose="affinity")
    rng_after = _global_rng_snapshot()
    marginal_error = float(
        (
            torch.sort(feature.flatten(2), dim=2).values
            - torch.sort(rolled_a.flatten(2), dim=2).values
        ).abs().max()
    )
    affinity_before = torch.sort(_periodic_affinity_multiset(feature).flatten(1), dim=1).values
    affinity_after = torch.sort(_periodic_affinity_multiset(rolled_a).flatten(1), dim=1).values
    affinity_multiset_error = float((affinity_before - affinity_after).abs().max())

    heads = build_matched_heads(fold=1)
    initial_hashes = {_parameter_sha256(head) for head in heads.values()}
    sample_features = torch.randn(2, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    sample_boxes = torch.tensor([[0.5, 0.5, 0.5, 0.6], [0.4, 0.6, 0.7, 0.3]])
    sample_valid = torch.ones(2, FEATURE_SIZE, FEATURE_SIZE)
    sample_box = normalized_bbox_masks(sample_boxes).float()
    sample_lab = torch.randn(2, 3, FEATURE_SIZE, FEATURE_SIZE)
    gradient_checks: Dict[str, bool] = {}
    update_checks: Dict[str, bool] = {}
    for role, head in heads.items():
        before = _parameter_sha256(head)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY
        )
        prediction = head(sample_features, sample_boxes)
        projection = projection_loss_torch(prediction, sample_box).mean()
        if role == PROJECTION_ROLE:
            loss = projection
        else:
            pair, _, _ = pairwise_affinity_torch(
                prediction, sample_lab, sample_valid, sample_box
            )
            loss = projection + pair.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_checks[role] = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in head.parameters()
        )
        optimizer.step()
        update_checks[role] = _parameter_sha256(head) != before

    checks = {
        "projection_matches_numpy_le_1e_10": projection_error <= 1e-10,
        "pairwise_matches_numpy_le_1e_10": pairwise_error <= 1e-10,
        "similarity_matches_numpy_le_1e_10": similarity_error <= 1e-10,
        "pairwise_valid_weights_exact": weights_exact,
        "lab_matches_skimage_le_5e_4": max(lab_error, numpy_lab_error) <= 5e-4,
        "lab_finite": bool(np.isfinite(observed_lab).all()),
        "dice_symmetric": symmetry_error <= 1e-12,
        "pairwise_complement_invariant": complement_error <= 1e-12,
        "dephasing_deterministic": torch.equal(rolled_a, rolled_b),
        "dephasing_nonzero": not torch.equal(feature, rolled_a),
        "dephasing_value_multiset_exact": marginal_error == 0.0,
        "dephasing_periodic_affinity_multiset_le_1e_6": affinity_multiset_error <= 1e-6,
        "dephasing_rng_local": _global_rng_equal(rng_before, rng_after),
        "matched_initial_states_exact": len(initial_hashes) == 1,
        "all_parameters_receive_nonzero_finite_gradient": all(gradient_checks.values()),
        "all_heads_change_on_first_step": all(update_checks.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": {
            "projection": projection_error,
            "pairwise": pairwise_error,
            "similarity": similarity_error,
            "lab_torch": lab_error,
            "lab_numpy": numpy_lab_error,
            "dice_symmetry": symmetry_error,
            "pairwise_complement": complement_error,
            "dephasing_value_multiset": marginal_error,
            "dephasing_periodic_affinity_multiset": affinity_multiset_error,
        },
        "gradient_checks": gradient_checks,
        "update_checks": update_checks,
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
        if artifact.is_file() and artifact != path:
            rows.append(
                {
                    "path": artifact.relative_to(output_dir).as_posix(),
                    "bytes": int(artifact.stat().st_size),
                    "sha256": _sha256(artifact),
                }
            )
    _write_json(path, {"schema": "trkh_boxinst_foreground_mask_a0_manifest_v1", "files": rows})
    return path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    path = Path(output_dir) / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in payload["files"]}
    observed = {
        artifact.relative_to(output_dir).as_posix(): artifact
        for artifact in Path(output_dir).rglob("*")
        if artifact.is_file() and artifact != path
    }
    if set(expected) != set(observed):
        raise ValueError("BoxInst artifact-manifest file set differs")
    for relative, artifact in observed.items():
        record = expected[relative]
        if int(artifact.stat().st_size) != int(record["bytes"]) or _sha256(artifact) != record["sha256"]:
            raise ValueError(f"BoxInst artifact differs from manifest: {relative}")
    return {"passed": True, "files": len(expected), "manifest_sha256": _sha256(path)}


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
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        a, b = float(left), float(right)
        if math.isnan(a) and math.isnan(b):
            return 0.0
        return abs(a - b)
    return 0.0 if left == right else float("inf")


def _close_delete_caches(extraction: Dict[str, object]) -> Dict[str, object]:
    paths = []
    for key in ("features", "lab"):
        value = extraction.pop(key, None)
        if isinstance(value, np.memmap):
            value.flush()
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        record = extraction.get(f"{key}_cache")
        if isinstance(record, Mapping):
            paths.append(Path(str(record["path"])))
    gc.collect()
    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
        removed.append({"path": str(path), "absent": not path.exists()})
    return {"deleted": all(item["absent"] for item in removed), "files": removed}


def _captured_keeper_forward(
    model: nn.Module,
    images: Tensor,
    image_mask: Tensor,
    bbox: Tensor,
) -> Tuple[Tensor, Tensor]:
    if not hasattr(model, "stem") or not hasattr(model.stem, "blocks"):  # type: ignore[attr-defined]
        raise TypeError("Keeper does not expose HybridConvStem blocks")
    block = model.stem.blocks[1]  # type: ignore[attr-defined]
    captured: List[Tensor] = []

    def capture(_module: nn.Module, _inputs: Tuple[Tensor, ...], output: object) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Keeper block-2 hook emitted a non-tensor")
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
    if len(block._forward_hooks) != hooks_before or len(captured) != 1:
        raise RuntimeError("Keeper block-2 hook lifecycle differs from lock")
    block2 = captured[0]
    expected = (images.shape[0], FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    if tuple(block2.shape) != expected:
        raise ValueError(f"Keeper block-2 shape differs: {tuple(block2.shape)} != {expected}")
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


def _geometry_from_metadata(
    image_mask: Tensor,
    crop_bbox: Tensor,
) -> Tuple[Tensor, Tensor, List[Dict[str, object]]]:
    mask = image_mask
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Image-valid mask must be [B,1,H,W]")
    valid = F.interpolate(
        mask.float(), size=(FEATURE_SIZE, FEATURE_SIZE), mode="area"
    )[:, 0] >= 0.5
    raw_bbox = normalized_bbox_masks(crop_bbox[:, :4], size=FEATURE_SIZE)
    outside = raw_bbox & ~valid
    if bool(outside.any()):
        rows = torch.nonzero(outside.flatten(1).any(dim=1), as_tuple=False).flatten().tolist()
        raise ValueError(f"Transformed bbox leaves valid image support: rows={rows[:8]}")
    bbox = raw_bbox & valid
    if not bool(bbox.flatten(1).any(dim=1).all()):
        raise ValueError("Transformed bbox has empty 64x64 support")
    records: List[Dict[str, object]] = []
    for row in range(valid.shape[0]):
        valid_coordinates = torch.nonzero(valid[row], as_tuple=False)
        bbox_coordinates = torch.nonzero(bbox[row], as_tuple=False)
        vy0 = int(valid_coordinates[:, 0].min())
        vy1 = int(valid_coordinates[:, 0].max()) + 1
        vx0 = int(valid_coordinates[:, 1].min())
        vx1 = int(valid_coordinates[:, 1].max()) + 1
        by0 = int(bbox_coordinates[:, 0].min())
        by1 = int(bbox_coordinates[:, 0].max()) + 1
        bx0 = int(bbox_coordinates[:, 1].min())
        bx1 = int(bbox_coordinates[:, 1].max()) + 1
        records.append(
            {
                "valid_x0": vx0,
                "valid_y0": vy0,
                "valid_x1_exclusive": vx1,
                "valid_y1_exclusive": vy1,
                "bbox_x0": bx0,
                "bbox_y0": by0,
                "bbox_x1_exclusive": bx1,
                "bbox_y1_exclusive": by1,
                "valid_pixels": int(valid[row].sum()),
                "bbox_pixels": int(bbox[row].sum()),
                "bbox_outside_valid_pixels": int(outside[row].sum()),
                "valid_fraction": float(valid[row].float().mean()),
                "bbox_fraction": float(bbox[row].float().mean()),
                "valid_aspect_ratio": float((vx1 - vx0) / max(vy1 - vy0, 1)),
                "bbox_aspect_ratio": float((bx1 - bx0) / max(by1 - by0, 1)),
            }
        )
    return valid, bbox, records


def extract_condition(
    *,
    model: nn.Module,
    loader: DataLoader,
    loader_summary: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    device: torch.device,
    semantics: Mapping[str, object],
    condition: str,
    feature_cache_path: Path,
    lab_cache_path: Path,
    benchmark_ordinary: bool,
) -> Dict[str, object]:
    count = len(rows)
    feature_path = Path(feature_cache_path).resolve()
    lab_path = Path(lab_cache_path).resolve()
    if feature_path.exists() or lab_path.exists():
        raise FileExistsError("Temporary BoxInst cache already exists")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features_cache = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    )
    lab_cache = np.lib.format.open_memmap(
        lab_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, 3, FEATURE_SIZE, FEATURE_SIZE),
    )
    probabilities = np.empty((count, 5), dtype=np.float32)
    targets = np.empty(count, dtype=np.int64)
    sample_indices = np.empty(count, dtype=np.int64)
    model_boxes = np.empty((count, 4), dtype=np.float32)
    crop_boxes = np.empty((count, 4), dtype=np.float32)
    valid_masks = np.empty((count, FEATURE_SIZE, FEATURE_SIZE), dtype=np.bool_)
    bbox_masks = np.empty_like(valid_masks)
    rgb64 = np.empty((count, FEATURE_SIZE, FEATURE_SIZE, 3), dtype=np.uint8)
    geometry: List[Dict[str, object]] = []
    model.eval()
    state_before = _model_state_sha256(model)
    mean = torch.tensor(
        semantics["input_mean"], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        semantics["input_std"], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    position = 0
    capture_error = 0.0
    cidt_error = 0.0
    cidt_argmax_exact = True
    ordinary_seconds = 0.0
    captured_seconds = 0.0
    preprocessing_seconds = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, batch_targets, metadata in loader:
            if not isinstance(metadata, Mapping):
                raise ValueError("BoxInst extraction requires tensor metadata")
            batch = int(batch_targets.numel())
            stop = position + batch
            expected = rows[position:stop]
            observed_indices = metadata["sample_index"].detach().cpu().long().tolist()
            if observed_indices != [row.sample_index for row in expected]:
                raise ValueError("BoxInst loader changed locked sample order")
            observed_targets = batch_targets.detach().cpu().long().tolist()
            if observed_targets != [row.target for row in expected]:
                raise ValueError("BoxInst loader changed locked targets")
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            bbox = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _metadata_tensor(
                metadata, "crop_bbox", device=device, dtype=torch.float32
            )
            if image_mask is None or bbox is None or crop_bbox is None:
                raise ValueError("BoxInst extraction requires image_mask/bbox/crop_bbox")
            if image_mask.ndim == 3:
                image_mask = image_mask[:, None]
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
                capture_error = max(
                    capture_error,
                    float((ordinary - current_probabilities).abs().max()),
                )
                if not torch.equal(ordinary.argmax(dim=1), current_probabilities.argmax(dim=1)):
                    raise RuntimeError("Hooked keeper changed argmax")

            preprocess_started = time.perf_counter()
            valid, bbox_mask, batch_geometry = _geometry_from_metadata(
                image_mask, crop_bbox
            )
            rgb = (images_device * std + mean).clamp(0.0, 1.0)
            rgb_small = F.interpolate(
                rgb, size=(FEATURE_SIZE, FEATURE_SIZE), mode="area"
            )
            lab_small = srgb_to_lab_torch(rgb_small)
            torch.cuda.synchronize(device)
            preprocessing_seconds += time.perf_counter() - preprocess_started

            features_cache[position:stop] = block2.detach().cpu().numpy().astype(np.float16)
            lab_cache[position:stop] = lab_small.detach().cpu().numpy().astype(np.float16)
            current_numpy = current_probabilities.detach().cpu().numpy().astype(np.float32)
            probabilities[position:stop] = current_numpy
            targets[position:stop] = np.asarray(observed_targets, dtype=np.int64)
            sample_indices[position:stop] = np.asarray(observed_indices, dtype=np.int64)
            model_boxes[position:stop] = bbox.detach().cpu().numpy().astype(np.float32)
            crop_boxes[position:stop] = crop_bbox.detach().cpu().numpy().astype(np.float32)
            valid_masks[position:stop] = valid.detach().cpu().numpy()
            bbox_masks[position:stop] = bbox_mask.detach().cpu().numpy()
            rgb64[position:stop] = (
                rgb_small.permute(0, 2, 3, 1).detach().cpu().numpy().clip(0.0, 1.0)
                * 255.0
            ).round().astype(np.uint8)
            for offset, record in enumerate(batch_geometry):
                expected_row = expected[offset]
                geometry.append(
                    {
                        "position": position + offset,
                        "sample_index": int(expected_row.sample_index),
                        "source_stem": expected_row.source_stem,
                        "fold": int(expected_row.fold),
                        **record,
                    }
                )
            historical = np.asarray(
                [row.keeper_probabilities for row in expected], dtype=np.float32
            )
            cidt_error = max(cidt_error, float(np.max(np.abs(current_numpy - historical))))
            cidt_argmax_exact = cidt_argmax_exact and bool(
                np.array_equal(current_numpy.argmax(axis=1), historical.argmax(axis=1))
            )
            position = stop
    if position != count:
        raise ValueError(f"BoxInst extraction row count differs: {position} != {count}")
    features_cache.flush()
    lab_cache.flush()
    elapsed = time.perf_counter() - started
    state_after = _model_state_sha256(model)
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    geometry_reconstructed = all(
        int(record["bbox_pixels"])
        == int(
            bbox_masks[int(record["position"])][
                int(record["bbox_y0"]):int(record["bbox_y1_exclusive"]),
                int(record["bbox_x0"]):int(record["bbox_x1_exclusive"]),
            ].sum()
        )
        for record in geometry
    )
    checks = {
        "rows_exact": position == count,
        "sample_order_exact": sample_indices.tolist() == [row.sample_index for row in rows],
        "targets_exact": targets.tolist() == [row.target for row in rows],
        "hook_probability_error_le_1e_6": capture_error <= MAX_CAPTURE_PROBABILITY_ERROR,
        "cidt_argmax_exact": cidt_argmax_exact,
        "keeper_state_exact": state_before == state_after,
        "feature_shape_exact": features_cache.shape
        == (count, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        "lab_shape_exact": lab_cache.shape == (count, 3, FEATURE_SIZE, FEATURE_SIZE),
        "all_geometry_nonempty": bool(
            valid_masks.reshape(count, -1).any(axis=1).all()
            and bbox_masks.reshape(count, -1).any(axis=1).all()
        ),
        "bbox_inside_valid": not bool((bbox_masks & ~valid_masks).any()),
        "geometry_reconstructs": geometry_reconstructed,
        "features_finite": bool(np.isfinite(features_cache).all()),
        "lab_finite": bool(np.isfinite(lab_cache).all()),
    }
    return {
        "features": features_cache,
        "lab": lab_cache,
        "probabilities": probabilities,
        "targets": targets,
        "sample_indices": sample_indices,
        "model_boxes": model_boxes,
        "crop_boxes": crop_boxes,
        "valid_masks": valid_masks,
        "bbox_masks": bbox_masks,
        "rgb64": rgb64,
        "geometry": geometry,
        "checks": checks,
        "capture_probability_max_abs_error": capture_error,
        "cidt_probability_max_abs_error": cidt_error,
        "cidt_argmax_exact": cidt_argmax_exact,
        "features_cache": {
            "path": str(feature_path),
            "bytes": int(feature_path.stat().st_size),
            "sha256": _array_sha256(features_cache),
        },
        "lab_cache": {
            "path": str(lab_path),
            "bytes": int(lab_path.stat().st_size),
            "sha256": _array_sha256(lab_cache),
        },
        "runtime": {
            "condition": condition,
            "seconds": elapsed,
            "ordinary_forward_seconds": ordinary_seconds,
            "captured_forward_seconds": captured_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "throughput_images_per_second": count / max(elapsed, 1e-12),
            "peak_cuda_bytes": peak_bytes,
            "loader": dict(loader_summary),
        },
    }


def _epoch_orders(fit_positions: np.ndarray, *, fold: int) -> Tuple[List[np.ndarray], str]:
    positions = np.asarray(fit_positions, dtype=np.int64)
    orders: List[np.ndarray] = []
    digest = hashlib.sha256()
    for epoch in range(EPOCHS):
        generator = np.random.default_rng(SEED + int(fold) * 1009 + epoch)
        order = positions[generator.permutation(positions.size)]
        orders.append(order)
        digest.update(order.tobytes())
    return orders, digest.hexdigest()


def _optimizer_summary(optimizer: torch.optim.Optimizer) -> Dict[str, object]:
    dtypes = set()
    finite = True
    tensors = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensors += 1
                dtypes.add(str(value.dtype))
                finite = finite and bool(torch.isfinite(value).all())
    return {
        "state_tensors": tensors,
        "state_tensor_dtypes": sorted(dtypes),
        "all_state_tensors_finite": finite,
    }


def _changed_parameters(
    initial: Mapping[str, Tensor],
    module: nn.Module,
) -> Dict[str, bool]:
    return {
        name: not torch.equal(initial[name], value.detach().cpu())
        for name, value in module.named_parameters()
    }


def _batch_from_cache(
    extraction: Mapping[str, object],
    positions: np.ndarray,
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[int]]:
    selected = np.asarray(positions, dtype=np.int64)
    features = torch.from_numpy(
        np.asarray(extraction["features"][selected], dtype=np.float32)
    ).to(device, non_blocking=True)
    lab = torch.from_numpy(
        np.asarray(extraction["lab"][selected], dtype=np.float32)
    ).to(device, non_blocking=True)
    valid = torch.from_numpy(
        np.asarray(extraction["valid_masks"][selected], dtype=np.bool_)
    ).to(device, non_blocking=True)
    bbox_mask = torch.from_numpy(
        np.asarray(extraction["bbox_masks"][selected], dtype=np.bool_)
    ).to(device, non_blocking=True)
    boxes = torch.from_numpy(
        np.asarray(extraction["crop_boxes"][selected], dtype=np.float32)
    ).to(device, non_blocking=True)
    indices = np.asarray(extraction["sample_indices"])[selected].astype(int).tolist()
    return features, lab, valid, bbox_mask, boxes, indices


def train_oof_mask_heads(
    *,
    extraction: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    device: torch.device,
) -> Dict[str, object]:
    row_folds = np.asarray([row.fold for row in rows], dtype=np.int64)
    count = len(rows)
    masks = {
        role: np.empty((count, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float16)
        for role in TRAINED_ROLES
    }
    fold_records: Dict[int, Dict[str, object]] = {}
    training_records: List[Dict[str, object]] = []
    total_started = time.perf_counter()
    for fold in FOLDS:
        fit_positions = np.flatnonzero(row_folds != fold)
        held_positions = np.flatnonzero(row_folds == fold)
        fit_sources = {rows[position].source_stem for position in fit_positions}
        held_sources = {rows[position].source_stem for position in held_positions}
        overlap = sorted(fit_sources.intersection(held_sources))
        if overlap:
            raise ValueError(f"Mask fold {fold} has source overlap")
        orders, occurrence_sha = _epoch_orders(fit_positions, fold=fold)
        heads = {role: head.to(device).float().train() for role, head in build_matched_heads(fold=fold).items()}
        initial_hashes = {role: _parameter_sha256(head) for role, head in heads.items()}
        initial_parameters = {
            role: {
                name: value.detach().cpu().clone()
                for name, value in head.named_parameters()
            }
            for role, head in heads.items()
        }
        optimizers = {
            role: torch.optim.AdamW(
                head.parameters(),
                lr=LEARNING_RATE,
                betas=BETAS,
                weight_decay=WEIGHT_DECAY,
            )
            for role, head in heads.items()
        }
        updates_per_epoch = int(math.ceil(fit_positions.size / BATCH_SIZE))
        total_updates = updates_per_epoch * EPOCHS
        warmup_updates = int(math.ceil(total_updates / 9.0))
        update = 0
        first_gradients: Dict[str, Dict[str, bool]] = {}
        epoch_records: List[Dict[str, object]] = []
        torch.cuda.reset_peak_memory_stats(device)
        fold_started = time.perf_counter()
        for epoch, order in enumerate(orders):
            sums = {
                role: {"loss": 0.0, "projection": 0.0, "pairwise": 0.0, "rows": 0}
                for role in TRAINED_ROLES
            }
            for start in range(0, order.size, BATCH_SIZE):
                batch_positions = order[start:start + BATCH_SIZE]
                features, lab, valid, bbox_mask, boxes, indices = _batch_from_cache(
                    extraction, batch_positions, device=device
                )
                dephased_lab = common_toroidal_roll(
                    lab, indices, fold=fold, purpose="affinity"
                )
                warmup = min(float(update + 1) / float(warmup_updates), 1.0)
                for role in TRAINED_ROLES:
                    optimizer = optimizers[role]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = heads[role](features, boxes)[:, 0]
                    masked_prediction = prediction * valid.float()
                    projection = projection_loss_torch(
                        masked_prediction, bbox_mask.float()
                    ).mean()
                    pairwise = projection.new_zeros(())
                    if role != PROJECTION_ROLE:
                        pair_lab = lab if role == CANDIDATE_ROLE else dephased_lab
                        pairwise_values, _, _ = pairwise_affinity_torch(
                            masked_prediction,
                            pair_lab,
                            valid,
                            bbox_mask,
                        )
                        pairwise = pairwise_values.mean()
                    loss = projection + warmup * pairwise
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"Non-finite BoxInst loss in fold {fold}, role {role}")
                    loss.backward()
                    if update == 0:
                        first_gradients[role] = {
                            name: bool(
                                parameter.grad is not None
                                and torch.isfinite(parameter.grad).all()
                                and float(parameter.grad.abs().sum()) > 0.0
                            )
                            for name, parameter in heads[role].named_parameters()
                        }
                    optimizer.step()
                    rows_in_batch = int(batch_positions.size)
                    sums[role]["loss"] += float(loss.detach()) * rows_in_batch
                    sums[role]["projection"] += float(projection.detach()) * rows_in_batch
                    sums[role]["pairwise"] += float(pairwise.detach()) * rows_in_batch
                    sums[role]["rows"] += rows_in_batch
                update += 1
                del features, lab, dephased_lab, valid, bbox_mask, boxes
            epoch_records.append(
                {
                    "epoch": epoch + 1,
                    "updates_completed": update,
                    "warmup_factor": min(float(update) / float(warmup_updates), 1.0),
                    "roles": {
                        role: {
                            key: float(value / max(sums[role]["rows"], 1))
                            for key, value in sums[role].items()
                            if key != "rows"
                        }
                        for role in TRAINED_ROLES
                    },
                }
            )
        torch.cuda.synchronize(device)
        fold_training_seconds = time.perf_counter() - fold_started
        if update != total_updates:
            raise RuntimeError(f"Mask fold {fold} update count differs: {update} != {total_updates}")
        parameter_changes = {
            role: _changed_parameters(initial_parameters[role], heads[role])
            for role in TRAINED_ROLES
        }
        optimizer_summaries = {
            role: _optimizer_summary(optimizers[role]) for role in TRAINED_ROLES
        }
        parameter_dtypes = {
            role: sorted({str(value.dtype) for value in head.parameters()})
            for role, head in heads.items()
        }
        final_hashes = {role: _parameter_sha256(head) for role, head in heads.items()}

        for start in range(0, held_positions.size, BATCH_SIZE):
            positions = held_positions[start:start + BATCH_SIZE]
            features, _lab, valid, _bbox_mask, boxes, _indices = _batch_from_cache(
                extraction, positions, device=device
            )
            with torch.inference_mode():
                for role in TRAINED_ROLES:
                    heads[role].eval()
                    prediction = heads[role](features, boxes)[:, 0] * valid.float()
                    masks[role][positions] = prediction.cpu().numpy().astype(np.float16)
            del features, _lab, valid, _bbox_mask, boxes

        cpu_heads = {role: head.cpu().eval() for role, head in heads.items()}
        fold_record = {
            "held_fold": fold,
            "fit_positions": fit_positions,
            "held_positions": held_positions,
            "fit_rows": int(fit_positions.size),
            "held_rows": int(held_positions.size),
            "fit_sources": len(fit_sources),
            "held_sources": len(held_sources),
            "source_overlap": overlap,
            "occurrence_sha256": occurrence_sha,
            "updates": total_updates,
            "warmup_updates": warmup_updates,
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
            "parameter_changes": parameter_changes,
            "first_gradients": first_gradients,
            "optimizer": optimizer_summaries,
            "parameter_dtypes": parameter_dtypes,
            "epochs": epoch_records,
            "training_seconds": fold_training_seconds,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "models": cpu_heads,
        }
        fold_records[fold] = fold_record
        for role in TRAINED_ROLES:
            training_records.append(
                {
                    "fold": fold,
                    "role": role,
                    "updates": total_updates,
                    "warmup_updates": warmup_updates,
                    "occurrence_sha256": occurrence_sha,
                    "initial_sha256": initial_hashes[role],
                    "final_sha256": final_hashes[role],
                    "gradient_pass": all(first_gradients[role].values()),
                    "update_pass": all(parameter_changes[role].values()),
                    "parameter_dtypes": parameter_dtypes[role],
                    "optimizer": optimizer_summaries[role],
                    "final_epoch": epoch_records[-1]["roles"][role],
                }
            )
        torch.cuda.empty_cache()
    if any(not np.isfinite(mask).all() for mask in masks.values()):
        raise RuntimeError("OOF BoxInst masks contain non-finite values")
    return {
        "masks": masks,
        "fold_records": fold_records,
        "training_records": training_records,
        "occurrence_hashes": {
            str(fold): fold_records[fold]["occurrence_sha256"] for fold in FOLDS
        },
        "shared_occurrence_order": all(
            len({record["occurrence_sha256"] for record in training_records if record["fold"] == fold}) == 1
            for fold in FOLDS
        ),
        "seconds": time.perf_counter() - total_started,
    }


def _cohort_positions(
    rows: Sequence[CleanTrainRow], cohort: Sequence[CleanTrainRow]
) -> np.ndarray:
    mapping = {row.sample_index: position for position, row in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError("Training sample indices are not unique")
    try:
        positions = np.asarray([mapping[row.sample_index] for row in cohort], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Cohort sample is absent from training rows: {error}") from error
    if np.asarray([rows[position].sample_index for position in positions]).tolist() != [
        row.sample_index for row in cohort
    ]:
        raise RuntimeError("Cohort position mapping changed order")
    return positions


def _rolled_candidate_masks(
    candidate: np.ndarray,
    sample_indices: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    source = np.asarray(candidate, dtype=np.float32)
    output = np.empty_like(source)
    for fold in FOLDS:
        selected = np.flatnonzero(folds == fold)
        tensor = torch.from_numpy(source[selected])
        output[selected] = common_toroidal_roll(
            tensor,
            sample_indices[selected].astype(int).tolist(),
            fold=fold,
            purpose="mask_roll",
        ).numpy()
    return output


def build_cohort_artifacts(
    *,
    output_dir: Path,
    extraction: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    positions = _cohort_positions(rows, cohort)
    count = len(cohort)
    feature_path = output_dir / "cohort_block2_features_fp16.npy"
    lab_path = output_dir / "cohort_lab_fp16.npy"
    cohort_features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    )
    cohort_lab = np.lib.format.open_memmap(
        lab_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, 3, FEATURE_SIZE, FEATURE_SIZE),
    )
    valid = np.asarray(extraction["valid_masks"])[positions].astype(np.bool_)
    bbox = np.asarray(extraction["bbox_masks"])[positions].astype(np.bool_)
    keeper = np.asarray(extraction["probabilities"])[positions].astype(np.float64)
    crop_boxes = np.asarray(extraction["crop_boxes"])[positions].astype(np.float32)
    sample_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    keeper_predictions = np.asarray(
        [row.keeper_prediction for row in cohort], dtype=np.int64
    )
    rgb64 = np.asarray(extraction["rgb64"])[positions].astype(np.uint8)
    masks = {
        role: np.asarray(oof["masks"][role])[positions].astype(np.float32)
        for role in TRAINED_ROLES
    }
    masks[VALID_ROLE] = valid.astype(np.float32)
    masks[BBOX_ROLE] = bbox.astype(np.float32)
    masks[ROLLED_ROLE] = _rolled_candidate_masks(
        masks[CANDIDATE_ROLE], sample_indices, folds
    ) * valid.astype(np.float32)
    projection = torch.from_numpy(fixed_projection().astype(np.float32)).to(device)
    descriptor_chunks: Dict[str, List[np.ndarray]] = {role: [] for role in ROLE_NAMES}
    for start in range(0, count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, count)
        source_positions = positions[start:stop]
        feature_chunk = np.asarray(
            extraction["features"][source_positions], dtype=np.float16
        )
        lab_chunk = np.asarray(extraction["lab"][source_positions], dtype=np.float16)
        cohort_features[start:stop] = feature_chunk
        cohort_lab[start:stop] = lab_chunk
        feature_tensor = torch.from_numpy(feature_chunk.astype(np.float32)).to(device)
        with torch.inference_mode():
            projected = torch.einsum(
                "bchw,cd->bdhw", feature_tensor, projection
            ).cpu().numpy().astype(np.float32)
        chunk_masks = {role: value[start:stop] for role, value in masks.items()}
        built = build_descriptors(
            projected,
            chunk_masks,
            valid[start:stop],
            bbox[start:stop],
            keeper[start:stop],
        )
        for role in ROLE_NAMES:
            descriptor_chunks[role].append(built[role])
    cohort_features.flush()
    cohort_lab.flush()
    descriptors = {
        role: np.concatenate(chunks, axis=0).astype(np.float64)
        for role, chunks in descriptor_chunks.items()
    }
    if any(value.shape[0] != count for value in descriptors.values()):
        raise RuntimeError("Cohort descriptor row count differs")
    inputs_path = output_dir / "cohort_inputs.npz"
    np.savez_compressed(
        inputs_path,
        positions=positions,
        sample_indices=sample_indices,
        folds=folds,
        targets=targets,
        keeper_predictions=keeper_predictions,
        keeper_probabilities=keeper.astype(np.float32),
        crop_boxes=crop_boxes,
        valid_masks=valid,
        bbox_masks=bbox,
        rgb64=rgb64,
        **{f"mask__{role}": value.astype(np.float16) for role, value in masks.items()},
    )
    descriptors_path = output_dir / "cohort_descriptors.npz"
    np.savez_compressed(
        descriptors_path,
        **{role: value for role, value in descriptors.items()},
    )
    return {
        "positions": positions,
        "sample_indices": sample_indices,
        "folds": folds,
        "targets": targets,
        "keeper_predictions": keeper_predictions,
        "keeper_probabilities": keeper,
        "crop_boxes": crop_boxes,
        "valid_masks": valid,
        "bbox_masks": bbox,
        "rgb64": rgb64,
        "masks": masks,
        "descriptors": descriptors,
        "features": cohort_features,
        "lab": cohort_lab,
        "artifacts": {
            "features": {
                "path": str(feature_path.resolve()),
                "sha256": _array_sha256(cohort_features),
                "bytes": int(feature_path.stat().st_size),
            },
            "lab": {
                "path": str(lab_path.resolve()),
                "sha256": _array_sha256(cohort_lab),
                "bytes": int(lab_path.stat().st_size),
            },
            "inputs": {
                "path": str(inputs_path.resolve()),
                "sha256": _sha256(inputs_path),
            },
            "descriptors": {
                "path": str(descriptors_path.resolve()),
                "sha256": _sha256(descriptors_path),
            },
        },
    }


def mask_statistics(
    *,
    masks: Mapping[str, np.ndarray],
    extraction: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    count = int(np.asarray(extraction["sample_indices"]).size)
    per_role: Dict[str, Dict[str, np.ndarray]] = {}
    for role in TRAINED_ROLES:
        per_role[role] = {
            "projection": np.empty(count, dtype=np.float64),
            "pairwise_aligned": np.empty(count, dtype=np.float64),
            "area_valid": np.empty(count, dtype=np.float64),
            "area_bbox": np.empty(count, dtype=np.float64),
            "outside_bbox": np.empty(count, dtype=np.float64),
            "entropy": np.empty(count, dtype=np.float64),
            "saturated": np.empty(count, dtype=np.float64),
            "standard_deviation": np.empty(count, dtype=np.float64),
        }
    for start in range(0, count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, count)
        positions = np.arange(start, stop, dtype=np.int64)
        _features, lab, valid, bbox, _boxes, _indices = _batch_from_cache(
            extraction, positions, device=device
        )
        valid_float = valid.float()
        bbox_float = bbox.float()
        valid_area = valid_float.sum(dim=(1, 2)).clamp_min(1.0)
        bbox_area = bbox_float.sum(dim=(1, 2)).clamp_min(1.0)
        for role in TRAINED_ROLES:
            mask = torch.from_numpy(
                np.asarray(masks[role][start:stop], dtype=np.float32)
            ).to(device)
            mask = mask * valid_float
            mass = mask.sum(dim=(1, 2)).clamp_min(1e-12)
            projection = projection_loss_torch(mask, bbox_float)
            pairwise, _, _ = pairwise_affinity_torch(mask, lab, valid, bbox)
            probability = mask.clamp(1e-7, 1.0 - 1e-7)
            entropy = -(
                probability * torch.log(probability)
                + (1.0 - probability) * torch.log(1.0 - probability)
            )
            values = {
                "projection": projection,
                "pairwise_aligned": pairwise,
                "area_valid": mass / valid_area,
                "area_bbox": mass / bbox_area,
                "outside_bbox": (mask * (~bbox).float()).sum(dim=(1, 2)) / mass,
                "entropy": (entropy * valid_float).sum(dim=(1, 2)) / valid_area,
                "saturated": (
                    (((mask <= 0.005) | (mask >= 0.995)) & valid).float().sum(dim=(1, 2))
                    / valid_area
                ),
                "standard_deviation": torch.sqrt(
                    (
                        (mask - (mask.sum(dim=(1, 2)) / valid_area)[:, None, None]).square()
                        * valid_float
                    ).sum(dim=(1, 2))
                    / valid_area
                ),
            }
            for key, value in values.items():
                per_role[role][key][start:stop] = value.detach().cpu().numpy()
        del _features, lab, valid, bbox, _boxes
    summaries: Dict[str, object] = {}
    for role, values in per_role.items():
        role_mask = np.asarray(masks[role], dtype=np.float32)
        valid = np.asarray(extraction["valid_masks"], dtype=np.bool_)
        valid_values = np.where(valid, role_mask, np.nan)
        row_min = np.nanmin(valid_values, axis=(1, 2))
        row_max = np.nanmax(valid_values, axis=(1, 2))
        summaries[role] = {
            **{f"{key}_mean": float(value.mean()) for key, value in values.items()},
            "area_bbox_standard_deviation": float(values["area_bbox"].std()),
            "nonempty_rows": int(np.sum(values["area_valid"] > 0.0)),
            "all_zero_rows": int(np.sum(row_max <= 0.0)),
            "all_one_rows": int(np.sum(row_min >= 1.0)),
            "constant_rows": int(np.sum(values["standard_deviation"] <= 1e-8)),
            "more_than_99_5_percent_saturated_rows": int(
                np.sum(values["saturated"] > 0.995)
            ),
            "arrays": values,
        }
    return summaries


def _clean_mask_statistics_payload(statistics: Mapping[str, object]) -> Dict[str, object]:
    return {
        role: {
            key: value
            for key, value in record.items()
            if key != "arrays"
        }
        for role, record in statistics.items()
    }


def build_clean_analysis(
    *,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    keeper_predictions: np.ndarray,
    folds: np.ndarray,
    statistics: Mapping[str, object],
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
    candidate = metrics[CANDIDATE_ROLE]
    candidate_stats = statistics[CANDIDATE_ROLE]
    dephased_stats = statistics[DEPHASED_ROLE]
    controls = (BBOX_ROLE, PROJECTION_ROLE, DEPHASED_ROLE, ROLLED_ROLE)
    fold_wins = {
        role: sum(
            candidate["folds"][str(fold)]["auroc"]
            > metrics[role]["folds"][str(fold)]["auroc"]
            for fold in FOLDS
        )
        for role in controls
    }
    candidate_balance = (
        candidate["restricted_fp_rejected"]
        + candidate["keeper_fn_supported"]
        - candidate["keeper_tp_broken"]
    )
    gates = {
        "candidate_auroc_ge_0_85": candidate["auroc"] >= 0.85,
        "candidate_auprc_ge_0_90": candidate["auprc"] >= 0.90,
        "candidate_positive_and_tp_retention_ge_0_95": min(
            candidate["all_positive_retention"], candidate["keeper_tp_retention"]
        )
        >= 0.95,
        "candidate_supports_at_least_8_keeper_fn": candidate["keeper_fn_supported"] >= 8,
        "candidate_restricted_fp_rejection_ge_0_25": candidate["restricted_fp_rejection"] >= 0.25,
        "candidate_precision_ge_0_75": candidate["precision"] >= 0.75,
        "candidate_corrections_plus_fn_support_ge_tp_breaks": (
            candidate["restricted_fp_rejected"] + candidate["keeper_fn_supported"]
            >= candidate["keeper_tp_broken"]
        ),
        "candidate_auroc_margin_ge_0_02_over_keeper_valid_bbox": all(
            candidate["auroc"] - metrics[role]["auroc"] >= 0.02
            for role in (KEEPER_ROLE, VALID_ROLE, BBOX_ROLE)
        ),
        "candidate_auroc_margin_ge_0_01_over_projection": (
            candidate["auroc"] - metrics[PROJECTION_ROLE]["auroc"] >= 0.01
        ),
        "candidate_auroc_margin_ge_0_02_over_dephased_rolled": all(
            candidate["auroc"] - metrics[role]["auroc"] >= 0.02
            for role in (DEPHASED_ROLE, ROLLED_ROLE)
        ),
        "candidate_auprc_margin_ge_0_01_over_mask_controls": all(
            candidate["auprc"] - metrics[role]["auprc"] >= 0.01
            for role in controls
        ),
        "candidate_fp_rejection_margins_pass": all(
            candidate["restricted_fp_rejection"]
            - metrics[role]["restricted_fp_rejection"]
            >= (0.03 if role in (BBOX_ROLE, PROJECTION_ROLE) else 0.05)
            for role in controls
        ),
        "candidate_wins_at_least_4_of_5_folds_against_controls": all(
            wins >= 4 for wins in fold_wins.values()
        ),
        "candidate_without_keeper_auroc_ge_0_70": metrics[WITHOUT_KEEPER_ROLE]["auroc"] >= 0.70,
        "candidate_projection_outside_and_support_pass": (
            candidate_stats["projection_mean"] <= 0.10
            and candidate_stats["outside_bbox_mean"] <= 0.05
            and candidate_stats["nonempty_rows"] == EXPECTED_TRAIN_ROWS
        ),
        "candidate_area_variation_and_nondegenerate_pass": (
            0.35 <= candidate_stats["area_bbox_mean"] <= 0.90
            and candidate_stats["area_bbox_standard_deviation"] >= 0.02
            and candidate_stats["all_zero_rows"] == 0
            and candidate_stats["all_one_rows"] == 0
            and candidate_stats["constant_rows"] == 0
            and candidate_stats["more_than_99_5_percent_saturated_rows"] == 0
        ),
        "candidate_pairwise_aligned_loss_5pct_below_dephased": (
            candidate_stats["pairwise_aligned_mean"]
            <= 0.95 * dephased_stats["pairwise_aligned_mean"]
        ),
        "causal_controls_reduce_auroc_without_better_balance": (
            all(candidate["auroc"] - metrics[role]["auroc"] >= 0.02 for role in (DEPHASED_ROLE, ROLLED_ROLE))
            and all(
                candidate_balance
                >= metrics[role]["restricted_fp_rejected"]
                + metrics[role]["keeper_fn_supported"]
                - metrics[role]["keeper_tp_broken"]
                for role in (DEPHASED_ROLE, ROLLED_ROLE)
            )
        ),
    }
    return {
        "roles": metrics,
        "mask_statistics": _clean_mask_statistics_payload(statistics),
        "candidate_fold_wins": fold_wins,
        "candidate_balance": candidate_balance,
        "mechanism_gates": {key: bool(value) for key, value in gates.items()},
        "mechanism_gates_passed": all(bool(value) for value in gates.values()),
    }


def _statistics_from_arrays(
    *,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    masks: Mapping[str, np.ndarray],
    valid_masks: np.ndarray,
) -> Dict[str, object]:
    valid = np.asarray(valid_masks, dtype=np.bool_)
    summaries: Dict[str, object] = {}
    for role in TRAINED_ROLES:
        values = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in arrays[role].items()
        }
        role_mask = np.asarray(masks[role], dtype=np.float32)
        valid_values = np.where(valid, role_mask, np.nan)
        row_min = np.nanmin(valid_values, axis=(1, 2))
        row_max = np.nanmax(valid_values, axis=(1, 2))
        summaries[role] = {
            **{f"{key}_mean": float(value.mean()) for key, value in values.items()},
            "area_bbox_standard_deviation": float(values["area_bbox"].std()),
            "nonempty_rows": int(np.sum(values["area_valid"] > 0.0)),
            "all_zero_rows": int(np.sum(row_max <= 0.0)),
            "all_one_rows": int(np.sum(row_min >= 1.0)),
            "constant_rows": int(np.sum(values["standard_deviation"] <= 1e-8)),
            "more_than_99_5_percent_saturated_rows": int(
                np.sum(values["saturated"] > 0.995)
            ),
            "arrays": values,
        }
    return summaries


def save_replay_artifacts(
    *,
    output_dir: Path,
    rows: Sequence[CleanTrainRow],
    cohort: Sequence[CleanTrainRow],
    extraction: Mapping[str, object],
    oof: Mapping[str, object],
    cohort_artifacts: Mapping[str, object],
    readout_states: Mapping[str, object],
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    statistics: Mapping[str, object],
) -> Dict[str, object]:
    masks_path = output_dir / "oof_train_masks.npz"
    np.savez_compressed(
        masks_path,
        **{
            role: np.asarray(oof["masks"][role], dtype=np.float16)
            for role in TRAINED_ROLES
        },
    )
    geometry_path = output_dir / "train_geometry_masks.npz"
    np.savez_compressed(
        geometry_path,
        sample_indices=np.asarray(extraction["sample_indices"], dtype=np.int64),
        folds=np.asarray([row.fold for row in rows], dtype=np.int64),
        crop_boxes=np.asarray(extraction["crop_boxes"], dtype=np.float32),
        valid_masks=np.asarray(extraction["valid_masks"], dtype=np.bool_),
        bbox_masks=np.asarray(extraction["bbox_masks"], dtype=np.bool_),
    )
    geometry_csv_path = output_dir / "block2_full_geometry.csv"
    _write_csv(geometry_csv_path, extraction["geometry"])

    statistic_arrays = {}
    for role in TRAINED_ROLES:
        for key, value in statistics[role]["arrays"].items():
            statistic_arrays[f"{role}__{key}"] = np.asarray(value, dtype=np.float64)
    statistics_path = output_dir / "mask_statistics_arrays.npz"
    np.savez_compressed(statistics_path, **statistic_arrays)

    state_arrays: Dict[str, np.ndarray] = {}
    mask_metadata: Dict[str, object] = {"folds": {}}
    for fold in FOLDS:
        record = oof["fold_records"][fold]
        role_records = {}
        for role, model in record["models"].items():
            state_keys = {}
            for name, tensor in model.state_dict().items():
                key = f"fold{fold}__{role}__{name.replace('.', '___')}"
                state_arrays[key] = tensor.detach().cpu().numpy()
                state_keys[name] = key
            role_records[role] = {
                "state_arrays": state_keys,
                "initial_sha256": record["initial_hashes"][role],
                "final_sha256": record["final_hashes"][role],
            }
        mask_metadata["folds"][str(fold)] = {
            "held_fold": fold,
            "fit_positions": np.asarray(record["fit_positions"], dtype=np.int64).tolist(),
            "held_positions": np.asarray(record["held_positions"], dtype=np.int64).tolist(),
            "source_overlap": list(record["source_overlap"]),
            "occurrence_sha256": record["occurrence_sha256"],
            "updates": int(record["updates"]),
            "warmup_updates": int(record["warmup_updates"]),
            "roles": role_records,
        }
    states_path = output_dir / "mask_head_states.npz"
    np.savez_compressed(states_path, **state_arrays)
    metadata_path = output_dir / "mask_training_replay_metadata.json"
    _write_json(metadata_path, mask_metadata)
    readout_path = output_dir / "readout_states.json"
    _write_json(readout_path, readout_states)
    training_path = output_dir / "mask_training_evidence.json"
    _write_json(
        training_path,
        {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "pairwise_size": PAIRWISE_SIZE,
            "pairwise_dilation": PAIRWISE_DILATION,
            "pairwise_color_threshold": PAIRWISE_COLOR_THRESHOLD,
            "records": oof["training_records"],
        },
    )
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
            record[f"score__{role}"] = float(scores[role][position])
            record[f"action__{role}"] = int(bool(actions[role][position]))
        prediction_rows.append(record)
    predictions_path = output_dir / "oof_predictions.csv"
    _write_csv(predictions_path, prediction_rows)
    paths = {
        "oof_train_masks": masks_path,
        "train_geometry": geometry_path,
        "geometry_csv": geometry_csv_path,
        "mask_statistics": statistics_path,
        "mask_head_states": states_path,
        "mask_training_metadata": metadata_path,
        "readout_states": readout_path,
        "training_evidence": training_path,
        "predictions": predictions_path,
    }
    result = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    result.update(cohort_artifacts["artifacts"])
    return result


def _load_mask_heads(
    root: Path,
    *,
    device: torch.device,
) -> Dict[int, Dict[str, BoxInstMaskHead]]:
    metadata = json.loads(
        (root / "mask_training_replay_metadata.json").read_text(encoding="utf-8")
    )
    output: Dict[int, Dict[str, BoxInstMaskHead]] = {}
    with np.load(root / "mask_head_states.npz", allow_pickle=False) as archive:
        for fold in FOLDS:
            output[fold] = {}
            for role in TRAINED_ROLES:
                record = metadata["folds"][str(fold)]["roles"][role]
                state = {
                    name: torch.from_numpy(np.asarray(archive[key])).clone()
                    for name, key in record["state_arrays"].items()
                }
                head = BoxInstMaskHead()
                head.load_state_dict(state, strict=True)
                head = head.to(device).float().eval()
                if _parameter_sha256(head) != record["final_sha256"]:
                    raise ValueError(f"Replayed mask head state differs: fold={fold}, role={role}")
                output[fold][role] = head
    return output


def _reconstruct_cohort_masks(
    root: Path,
    *,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    features = np.load(root / "cohort_block2_features_fp16.npy", mmap_mode="r")
    with np.load(root / "cohort_inputs.npz", allow_pickle=False) as inputs:
        sample_indices = np.asarray(inputs["sample_indices"], dtype=np.int64)
        folds = np.asarray(inputs["folds"], dtype=np.int64)
        boxes = np.asarray(inputs["crop_boxes"], dtype=np.float32)
        valid = np.asarray(inputs["valid_masks"], dtype=np.bool_)
        persisted = {
            role: np.asarray(inputs[f"mask__{role}"], dtype=np.float16)
            for role in MASK_ROLES
        }
    heads = _load_mask_heads(root, device=device)
    reconstructed = {
        role: np.empty_like(persisted[role], dtype=np.float16)
        for role in TRAINED_ROLES
    }
    for fold in FOLDS:
        selected = np.flatnonzero(folds == fold)
        for start in range(0, selected.size, BATCH_SIZE):
            positions = selected[start:start + BATCH_SIZE]
            feature = torch.from_numpy(
                np.asarray(features[positions], dtype=np.float32)
            ).to(device)
            box = torch.from_numpy(boxes[positions]).to(device)
            valid_tensor = torch.from_numpy(valid[positions]).to(device)
            with torch.inference_mode():
                for role in TRAINED_ROLES:
                    mask = heads[fold][role](feature, box)[:, 0] * valid_tensor.float()
                    reconstructed[role][positions] = mask.cpu().numpy().astype(np.float16)
    reconstructed[VALID_ROLE] = valid.astype(np.float16)
    with np.load(root / "cohort_inputs.npz", allow_pickle=False) as inputs:
        reconstructed[BBOX_ROLE] = np.asarray(inputs["bbox_masks"], dtype=np.float16)
    reconstructed[ROLLED_ROLE] = _rolled_candidate_masks(
        reconstructed[CANDIDATE_ROLE].astype(np.float32), sample_indices, folds
    ).astype(np.float16) * valid.astype(np.float16)
    return reconstructed, persisted


def _rebuild_cohort_descriptors(
    root: Path,
    *,
    masks: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    features = np.load(root / "cohort_block2_features_fp16.npy", mmap_mode="r")
    with np.load(root / "cohort_inputs.npz", allow_pickle=False) as inputs:
        valid = np.asarray(inputs["valid_masks"], dtype=np.bool_)
        bbox = np.asarray(inputs["bbox_masks"], dtype=np.bool_)
        keeper = np.asarray(inputs["keeper_probabilities"], dtype=np.float64)
    projection = torch.from_numpy(fixed_projection().astype(np.float32)).to(device)
    chunks = {role: [] for role in ROLE_NAMES}
    for start in range(0, features.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, features.shape[0])
        tensor = torch.from_numpy(
            np.asarray(features[start:stop], dtype=np.float32)
        ).to(device)
        with torch.inference_mode():
            projected = torch.einsum("bchw,cd->bdhw", tensor, projection).cpu().numpy()
        built = build_descriptors(
            projected,
            {role: np.asarray(masks[role][start:stop], dtype=np.float32) for role in MASK_ROLES},
            valid[start:stop],
            bbox[start:stop],
            keeper[start:stop],
        )
        for role in ROLE_NAMES:
            chunks[role].append(built[role])
    return {role: np.concatenate(value, axis=0) for role, value in chunks.items()}


def _read_prediction_artifact(
    path: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {
        role: np.asarray([float(row[f"score__{role}"]) for row in rows], dtype=np.float64)
        for role in ROLE_NAMES
    }
    actions = {
        role: np.asarray([bool(int(row[f"action__{role}"])) for row in rows], dtype=np.bool_)
        for role in ROLE_NAMES
    }
    return rows, scores, actions


def _read_statistic_arrays(root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    output = {role: {} for role in TRAINED_ROLES}
    with np.load(root / "mask_statistics_arrays.npz", allow_pickle=False) as archive:
        for key in archive.files:
            role, name = key.split("__", 1)
            output[role][name] = np.asarray(archive[key], dtype=np.float64)
    return output


def _verify_readout_thresholds(
    descriptors: Mapping[str, np.ndarray],
    folds: np.ndarray,
    labels: np.ndarray,
    states: Mapping[str, object],
) -> float:
    maximum = 0.0
    fold_values = np.asarray(folds, dtype=np.int64)
    for role in ROLE_NAMES:
        values = np.asarray(descriptors[role], dtype=np.float64)
        for record in states[role]["folds"]:
            held_fold = int(record["held_fold"])
            fit = fold_values != held_fold
            mean = np.asarray(record["scaler_mean"], dtype=np.float64)
            scale = np.asarray(record["scaler_scale"], dtype=np.float64)
            coefficient = np.asarray(record["coefficient"], dtype=np.float64)
            intercept = np.asarray(record["intercept"], dtype=np.float64)
            decision = ((values[fit] - mean) / scale) @ coefficient.T + intercept
            fit_scores = 1.0 / (1.0 + np.exp(-decision[:, 0]))
            threshold = _positive_threshold(fit_scores, labels[fit])
            maximum = max(maximum, abs(threshold - float(record["threshold"])))
    return maximum


def replay_artifacts(
    output_dir: Path,
    *,
    expected_analysis: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, persisted_scores, persisted_actions = _read_prediction_artifact(
        root / "oof_predictions.csv"
    )
    if len(prediction_rows) != EXPECTED_COHORT_ROWS:
        raise ValueError("BoxInst replay cohort row count differs")
    sample_indices = np.asarray(
        [int(row["sample_index"]) for row in prediction_rows], dtype=np.int64
    )
    if _index_sha256(sample_indices.tolist()) != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError("BoxInst replay cohort order differs")
    targets = np.asarray([int(row["target"]) for row in prediction_rows], dtype=np.int64)
    keeper = np.asarray(
        [int(row["keeper_prediction"]) for row in prediction_rows], dtype=np.int64
    )
    folds = np.asarray([int(row["fold"]) for row in prediction_rows], dtype=np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)

    reconstructed_masks, persisted_masks = _reconstruct_cohort_masks(root, device=device)
    mask_error = max(
        float(
            np.max(
                np.abs(
                    reconstructed_masks[role].astype(np.float32)
                    - persisted_masks[role].astype(np.float32)
                )
            )
        )
        for role in MASK_ROLES
    )
    descriptors = _rebuild_cohort_descriptors(
        root, masks=reconstructed_masks, device=device
    )
    with np.load(root / "cohort_descriptors.npz", allow_pickle=False) as archive:
        persisted_descriptors = {
            role: np.asarray(archive[role], dtype=np.float64) for role in ROLE_NAMES
        }
    descriptor_error = max(
        float(np.max(np.abs(descriptors[role] - persisted_descriptors[role])))
        for role in ROLE_NAMES
    )
    states = json.loads((root / "readout_states.json").read_text(encoding="utf-8"))
    scores, actions = apply_readout_states(descriptors, folds, states)
    score_error = max(
        float(np.max(np.abs(scores[role] - persisted_scores[role])))
        for role in ROLE_NAMES
    )
    actions_exact = all(
        np.array_equal(actions[role], persisted_actions[role]) for role in ROLE_NAMES
    )
    threshold_error = _verify_readout_thresholds(descriptors, folds, labels, states)
    with np.load(root / "oof_train_masks.npz", allow_pickle=False) as mask_archive:
        full_masks = {
            role: np.asarray(mask_archive[role], dtype=np.float16)
            for role in TRAINED_ROLES
        }
    with np.load(root / "train_geometry_masks.npz", allow_pickle=False) as geometry:
        full_valid = np.asarray(geometry["valid_masks"], dtype=np.bool_)
    statistics = _statistics_from_arrays(
        arrays=_read_statistic_arrays(root),
        masks=full_masks,
        valid_masks=full_valid,
    )
    analysis = build_clean_analysis(
        scores=scores,
        actions=actions,
        targets=targets,
        keeper_predictions=keeper,
        folds=folds,
        statistics=statistics,
    )
    analysis_error = (
        _recursive_numeric_difference(expected_analysis, analysis)
        if expected_analysis is not None
        else 0.0
    )
    passed = bool(
        mask_error <= MAX_REPLAY_ERROR
        and descriptor_error <= MAX_REPLAY_ERROR
        and score_error <= MAX_REPLAY_ERROR
        and threshold_error <= MAX_REPLAY_ERROR
        and actions_exact
        and analysis_error <= MAX_REPLAY_ERROR
    )
    return {
        "passed": passed,
        "rows": len(prediction_rows),
        "mask_max_abs_error": mask_error,
        "descriptor_max_abs_error": descriptor_error,
        "score_max_abs_error": score_error,
        "threshold_max_abs_error": threshold_error,
        "actions_exact": actions_exact,
        "analysis_maximum_numeric_difference": analysis_error,
        "analysis": analysis,
    }


def fixed_xai_positions(cohort: Sequence[CleanTrainRow]) -> List[int]:
    positions: List[int] = []
    for fold in FOLDS:
        categories = (
            lambda row: row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS,
            lambda row: row.target == FOCUS_CLASS and row.keeper_prediction != FOCUS_CLASS,
            lambda row: row.target != FOCUS_CLASS,
        )
        for predicate in categories:
            candidates = [
                (row.sample_index, position)
                for position, row in enumerate(cohort)
                if row.fold == fold and predicate(row)
            ]
            if not candidates:
                raise ValueError(f"XAI selection lacks fold {fold} category")
            positions.append(min(candidates)[1])
    return positions


def _positive_heatmap(value: np.ndarray, *, size: int = 192) -> Image.Image:
    array = np.asarray(value, dtype=np.float64)
    finite = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(finite.min())
    maximum = float(finite.max())
    normalized = (finite - minimum) / max(maximum - minimum, 1e-12)
    red = (255.0 * normalized).astype(np.uint8)
    green = (255.0 * np.sqrt(normalized)).astype(np.uint8)
    blue = (80.0 * (1.0 - normalized)).astype(np.uint8)
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)


def render_mask_contact_sheet(
    path: Path,
    *,
    cohort: Sequence[CleanTrainRow],
    cohort_artifacts: Mapping[str, object],
) -> Dict[str, object]:
    positions = fixed_xai_positions(cohort)
    roles = (VALID_ROLE, BBOX_ROLE, PROJECTION_ROLE, CANDIDATE_ROLE, DEPHASED_ROLE, ROLLED_ROLE)
    cell = 192
    header = 34
    label = 30
    columns = 1 + len(roles)
    sheet = Image.new("RGB", (columns * cell, header + len(positions) * (cell + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    titles = ("RGB + bbox",) + roles
    for column, title in enumerate(titles):
        draw.text((column * cell + 4, 10), title, fill="black", font=font)
    arrays = {"positions": np.asarray(positions, dtype=np.int64)}
    padding_leakage = 0.0
    all_finite = True
    for row_number, position in enumerate(positions):
        y = header + row_number * (cell + label)
        rgb = Image.fromarray(np.asarray(cohort_artifacts["rgb64"])[position]).resize(
            (cell, cell), Image.Resampling.BILINEAR
        )
        bbox = np.asarray(cohort_artifacts["bbox_masks"])[position]
        coordinates = np.argwhere(bbox)
        rgb_draw = ImageDraw.Draw(rgb)
        if coordinates.size:
            y0, x0 = coordinates.min(axis=0)
            y1, x1 = coordinates.max(axis=0)
            scale = cell / FEATURE_SIZE
            rgb_draw.rectangle(
                (int(x0 * scale), int(y0 * scale), int((x1 + 1) * scale), int((y1 + 1) * scale)),
                outline=(0, 255, 0),
                width=2,
            )
        sheet.paste(rgb, (0, y))
        valid = np.asarray(cohort_artifacts["valid_masks"])[position]
        for column, role in enumerate(roles, start=1):
            mask = np.asarray(cohort_artifacts["masks"][role][position], dtype=np.float32)
            all_finite = all_finite and bool(np.isfinite(mask).all())
            padding_leakage = max(
                padding_leakage,
                float(np.max(np.abs(mask[~valid]))) if bool((~valid).any()) else 0.0,
            )
            sheet.paste(_positive_heatmap(mask, size=cell), (column * cell, y))
            arrays[f"row{row_number}__{role}"] = mask.astype(np.float32)
        row = cohort[position]
        draw.text(
            (4, y + cell + 7),
            f"idx={row.sample_index} fold={row.fold} y={row.target} keeper={row.keeper_prediction}",
            fill="black",
            font=font,
        )
    sheet.save(path)
    arrays_path = path.with_name("boxinst_mask_xai_arrays.npz")
    np.savez_compressed(arrays_path, **arrays)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": _sha256(arrays_path),
        "rows": len(positions),
        "columns": columns,
        "roles": list(roles),
        "all_finite": all_finite,
        "padding_leakage_maximum": padding_leakage,
        "automatic_passed": bool(
            len(positions) == 15 and all_finite and padding_leakage == 0.0
        ),
    }


def _torch_descriptor_from_fixed_mask(
    feature: Tensor,
    mask: Tensor,
    valid: Tensor,
    bbox: Tensor,
    keeper: Tensor,
    *,
    include_keeper: bool,
) -> Tensor:
    projection = feature.new_tensor(fixed_projection().astype(np.float32))
    projected = torch.einsum("chw,cd->dhw", feature, projection).double()
    mask64 = (mask.double() * valid.double()).clamp(0.0, 1.0)
    valid64 = valid.double()
    bbox_bool = bbox.bool()
    denominator = mask64.sum().clamp_min(1e-12)
    valid_denominator = valid64.sum().clamp_min(1.0)
    mean = (projected * mask64[None]).sum(dim=(1, 2)) / denominator
    std = torch.sqrt(
        (((projected - mean[:, None, None]) ** 2) * mask64[None]).sum(dim=(1, 2))
        / denominator
    )
    valid_mean = (projected * valid64[None]).sum(dim=(1, 2)) / valid_denominator
    valid_std = torch.sqrt(
        (((projected - valid_mean[:, None, None]) ** 2) * valid64[None]).sum(dim=(1, 2))
        / valid_denominator
    )
    probability = mask64.clamp(1e-7, 1.0 - 1e-7)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    stats = torch.stack(
        (
            denominator / valid_denominator,
            denominator / bbox.double().sum().clamp_min(1.0),
            (entropy * valid64).sum() / valid_denominator,
            (mask64 * (~bbox_bool).double()).sum() / denominator,
        )
    )
    base = torch.cat((mean, std, valid_mean, valid_std, stats))
    if include_keeper:
        return torch.cat((base, torch.log(keeper.double().clamp(1e-7, 1.0))))
    return base


def render_gradient_contact_sheet(
    path: Path,
    *,
    cohort: Sequence[CleanTrainRow],
    cohort_artifacts: Mapping[str, object],
    readout_states: Mapping[str, object],
    persisted_scores: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    positions = fixed_xai_positions(cohort)
    roles = MASK_ROLES + (WITHOUT_KEEPER_ROLE,)
    cell = 192
    header = 34
    label = 30
    sheet = Image.new("RGB", ((1 + len(roles)) * cell, header + len(positions) * (cell + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, title in enumerate(("RGB",) + roles):
        draw.text((column * cell + 4, 10), title, fill="black", font=font)
    arrays: Dict[str, np.ndarray] = {"positions": np.asarray(positions, dtype=np.int64)}
    score_error = 0.0
    padding_leakage = 0.0
    all_finite = True
    for row_number, position in enumerate(positions):
        y = header + row_number * (cell + label)
        rgb = Image.fromarray(np.asarray(cohort_artifacts["rgb64"])[position]).resize(
            (cell, cell), Image.Resampling.BILINEAR
        )
        sheet.paste(rgb, (0, y))
        row = cohort[position]
        valid = torch.from_numpy(np.asarray(cohort_artifacts["valid_masks"])[position])
        bbox = torch.from_numpy(np.asarray(cohort_artifacts["bbox_masks"])[position])
        keeper = torch.from_numpy(
            np.asarray(cohort_artifacts["keeper_probabilities"])[position].astype(np.float32)
        )
        for column, role in enumerate(roles, start=1):
            feature = torch.from_numpy(
                np.asarray(cohort_artifacts["features"][position], dtype=np.float32)
            ).requires_grad_(True)
            mask_role = CANDIDATE_ROLE if role == WITHOUT_KEEPER_ROLE else role
            mask = torch.from_numpy(
                np.asarray(cohort_artifacts["masks"][mask_role][position], dtype=np.float32)
            )
            descriptor = _torch_descriptor_from_fixed_mask(
                feature,
                mask,
                valid,
                bbox,
                keeper,
                include_keeper=role != WITHOUT_KEEPER_ROLE,
            )
            state = next(
                record
                for record in readout_states[role]["folds"]
                if int(record["held_fold"]) == int(row.fold)
            )
            mean = descriptor.new_tensor(state["scaler_mean"])
            scale = descriptor.new_tensor(state["scaler_scale"])
            coefficient = descriptor.new_tensor(state["coefficient"])[0]
            intercept = descriptor.new_tensor(state["intercept"])[0]
            score = torch.sigmoid(((descriptor - mean) / scale * coefficient).sum() + intercept)
            score.backward()
            gradient = torch.linalg.vector_norm(feature.grad, dim=0).detach().numpy()
            all_finite = all_finite and bool(np.isfinite(gradient).all())
            padding_leakage = max(
                padding_leakage,
                float(np.max(np.abs(gradient[~valid.numpy()])))
                if bool((~valid).any())
                else 0.0,
            )
            score_error = max(
                score_error,
                abs(float(score.detach()) - float(persisted_scores[role][position])),
            )
            arrays[f"row{row_number}__{role}"] = gradient.astype(np.float32)
            sheet.paste(_positive_heatmap(gradient, size=cell), (column * cell, y))
        draw.text(
            (4, y + cell + 7),
            f"idx={row.sample_index} fold={row.fold} y={row.target} keeper={row.keeper_prediction}",
            fill="black",
            font=font,
        )
    sheet.save(path)
    arrays_path = path.with_name("boxinst_gradient_xai_arrays.npz")
    np.savez_compressed(arrays_path, **arrays)
    automatic = bool(
        len(positions) == 15
        and all_finite
        and padding_leakage <= 1e-12
        and score_error <= 1e-5
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": _sha256(arrays_path),
        "rows": len(positions),
        "columns": 1 + len(roles),
        "roles": list(roles),
        "all_finite": all_finite,
        "padding_leakage_maximum": padding_leakage,
        "score_reconstruction_max_abs_error": score_error,
        "automatic_passed": automatic,
    }


def _torch_descriptor_batch(
    features: Tensor,
    masks: Tensor,
    valid: Tensor,
    bbox: Tensor,
    keeper: Tensor,
    projection: Tensor,
) -> Tensor:
    projected = torch.einsum("bchw,cd->bdhw", features, projection)
    valid_float = valid.to(projected.dtype)
    bbox_float = bbox.to(projected.dtype)
    mask = masks.to(projected.dtype).clamp(0.0, 1.0) * valid_float
    denominator = mask.sum(dim=(1, 2)).clamp_min(1e-12)
    valid_denominator = valid_float.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (projected * mask[:, None]).sum(dim=(2, 3)) / denominator[:, None]
    variance = (
        ((projected - mean[:, :, None, None]) ** 2 * mask[:, None]).sum(dim=(2, 3))
        / denominator[:, None]
    )
    std = torch.sqrt(torch.clamp_min(variance, 0.0))
    valid_mean = (
        (projected * valid_float[:, None]).sum(dim=(2, 3))
        / valid_denominator[:, None]
    )
    valid_variance = (
        (
            (projected - valid_mean[:, :, None, None]) ** 2
            * valid_float[:, None]
        ).sum(dim=(2, 3))
        / valid_denominator[:, None]
    )
    valid_std = torch.sqrt(torch.clamp_min(valid_variance, 0.0))
    probability = mask.clamp(1e-7, 1.0 - 1e-7)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    stats = torch.stack(
        (
            denominator / valid_denominator,
            denominator / bbox_float.sum(dim=(1, 2)).clamp_min(1.0),
            (entropy * valid_float).sum(dim=(1, 2)) / valid_denominator,
            (mask * (1.0 - bbox_float)).sum(dim=(1, 2)) / denominator,
        ),
        dim=1,
    )
    return torch.cat(
        (
            mean,
            std,
            valid_mean,
            valid_std,
            stats,
            torch.log(keeper.clamp(1e-7, 1.0)),
        ),
        dim=1,
    )


class StaticMaskDescriptorExport(nn.Module):
    def __init__(
        self,
        head: BoxInstMaskHead,
        state: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.head = copy.deepcopy(head).float().eval()
        self.register_buffer(
            "projection", torch.from_numpy(fixed_projection().astype(np.float32))
        )
        self.register_buffer(
            "readout_mean", torch.tensor(state["scaler_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "readout_scale", torch.tensor(state["scaler_scale"], dtype=torch.float32)
        )
        self.register_buffer(
            "readout_coefficient",
            torch.tensor(state["coefficient"], dtype=torch.float32)[0],
        )
        self.register_buffer(
            "readout_intercept", torch.tensor(state["intercept"], dtype=torch.float32)[0]
        )

    def forward(
        self,
        features: Tensor,
        boxes: Tensor,
        valid: Tensor,
        bbox: Tensor,
        keeper: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mask = self.head(features, boxes)[:, 0] * valid
        descriptor = _torch_descriptor_batch(
            features,
            mask,
            valid,
            bbox,
            keeper,
            self.projection,
        )
        decision = (
            (descriptor - self.readout_mean)
            / self.readout_scale
            * self.readout_coefficient
        ).sum(dim=1) + self.readout_intercept
        return mask, descriptor, torch.sigmoid(decision)


class StaticBBoxDescriptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "projection", torch.from_numpy(fixed_projection().astype(np.float32))
        )

    def forward(
        self,
        features: Tensor,
        boxes: Tensor,
        valid: Tensor,
        bbox: Tensor,
        keeper: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        del boxes
        descriptor = _torch_descriptor_batch(
            features,
            bbox,
            valid,
            bbox,
            keeper,
            self.projection,
        )
        return bbox, descriptor


def _candidate_fold_state(
    readout_states: Mapping[str, object], fold: int
) -> Mapping[str, object]:
    return next(
        record
        for record in readout_states[CANDIDATE_ROLE]["folds"]
        if int(record["held_fold"]) == int(fold)
    )


def static_export_checks(
    *,
    output_dir: Path,
    cohort_artifacts: Mapping[str, object],
    oof: Mapping[str, object],
    readout_states: Mapping[str, object],
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    fold = 0
    selected = np.flatnonzero(np.asarray(cohort_artifacts["folds"]) == fold)[:2]
    head = oof["fold_records"][fold]["models"][CANDIDATE_ROLE]
    wrapper = StaticMaskDescriptorExport(
        head, _candidate_fold_state(readout_states, fold)
    ).cpu().eval()
    features = torch.from_numpy(
        np.asarray(cohort_artifacts["features"][selected], dtype=np.float32)
    )
    boxes = torch.from_numpy(
        np.asarray(cohort_artifacts["crop_boxes"])[selected].astype(np.float32)
    )
    valid = torch.from_numpy(
        np.asarray(cohort_artifacts["valid_masks"])[selected].astype(np.float32)
    )
    bbox = torch.from_numpy(
        np.asarray(cohort_artifacts["bbox_masks"])[selected].astype(np.float32)
    )
    keeper = torch.from_numpy(
        np.asarray(cohort_artifacts["keeper_probabilities"])[selected].astype(np.float32)
    )
    with torch.inference_mode():
        expected = wrapper(features, boxes, valid, bbox, keeper)
    path = output_dir / "boxinst_mask_descriptor_fold0.onnx"
    torch.onnx.export(
        wrapper,
        (features, boxes, valid, bbox, keeper),
        path,
        input_names=("features", "boxes", "valid", "bbox", "keeper"),
        output_names=("mask", "descriptor", "score"),
        dynamic_axes={
            name: {0: "batch"}
            for name in ("features", "boxes", "valid", "bbox", "keeper", "mask", "descriptor", "score")
        },
        opset_version=17,
        do_constant_folding=True,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    observed = session.run(
        None,
        {
            "features": features.numpy(),
            "boxes": boxes.numpy(),
            "valid": valid.numpy(),
            "bbox": bbox.numpy(),
            "keeper": keeper.numpy(),
        },
    )
    errors = [
        float(np.max(np.abs(value.detach().numpy() - actual)))
        for value, actual in zip(expected, observed)
    ]
    threshold = float(_candidate_fold_state(readout_states, fold)["threshold"])
    actions_exact = bool(
        np.array_equal(
            expected[2].detach().numpy() >= threshold,
            np.asarray(observed[2]) >= threshold,
        )
    )
    graph = onnx.load(str(path))
    custom_domains = sorted({node.domain for node in graph.graph.node if node.domain})
    checks = {
        "mask_error_le_1e_5": errors[0] <= 1e-5,
        "descriptor_error_le_1e_5": errors[1] <= 1e-5,
        "score_error_le_1e_5": errors[2] <= 1e-5,
        "stored_threshold_actions_exact": actions_exact,
        "no_custom_operator_domain": not custom_domains,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "mask_max_abs_error": errors[0],
        "descriptor_max_abs_error": errors[1],
        "score_max_abs_error": errors[2],
        "actions_exact": actions_exact,
        "custom_domains": custom_domains,
    }


def _candidate_descriptors_from_masks(
    *,
    features: np.ndarray,
    masks: np.ndarray,
    valid: np.ndarray,
    bbox: np.ndarray,
    keeper: np.ndarray,
) -> np.ndarray:
    projection = fixed_projection()
    projected = np.einsum(
        "bchw,cd->bdhw",
        np.asarray(features, dtype=np.float64),
        projection,
    )
    mask_bank = {role: np.asarray(masks, dtype=np.float64) for role in MASK_ROLES}
    return build_descriptors(projected, mask_bank, valid, bbox, keeper)[CANDIDATE_ROLE]


def _readout_probabilities_from_state(
    descriptors: np.ndarray,
    state: Mapping[str, object],
) -> np.ndarray:
    mean = np.asarray(state["scaler_mean"], dtype=np.float64)
    scale = np.asarray(state["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(state["coefficient"], dtype=np.float64)[0]
    intercept = float(np.asarray(state["intercept"], dtype=np.float64)[0])
    decision = ((descriptors - mean) / scale * coefficient).sum(axis=1) + intercept
    return 1.0 / (1.0 + np.exp(-decision))


def numeric_precision_checks(
    *,
    cohort_artifacts: Mapping[str, object],
    oof: Mapping[str, object],
    readout_states: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    fold = 0
    selected = np.flatnonzero(np.asarray(cohort_artifacts["folds"]) == fold)[:8]
    feature_numpy = np.asarray(
        cohort_artifacts["features"][selected], dtype=np.float32
    )
    boxes_numpy = np.asarray(cohort_artifacts["crop_boxes"])[selected].astype(np.float32)
    valid = np.asarray(cohort_artifacts["valid_masks"])[selected].astype(np.bool_)
    bbox = np.asarray(cohort_artifacts["bbox_masks"])[selected].astype(np.bool_)
    keeper = np.asarray(cohort_artifacts["keeper_probabilities"])[selected].astype(np.float64)
    state = _candidate_fold_state(readout_states, fold)
    head = oof["fold_records"][fold]["models"][CANDIDATE_ROLE]
    fp32_head = copy.deepcopy(head).cpu().float().eval()
    fp64_head = copy.deepcopy(head).cpu().double().eval()
    feature = torch.from_numpy(feature_numpy)
    boxes = torch.from_numpy(boxes_numpy)
    with torch.inference_mode():
        fp32_mask = fp32_head(feature, boxes)[:, 0].numpy() * valid
        fp64_mask = fp64_head(feature.double(), boxes.double())[:, 0].numpy() * valid
    bf16_head = copy.deepcopy(head).to(device).float().eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bf16_mask = bf16_head(feature.to(device), boxes.to(device))[:, 0]
    bf16_mask_numpy = bf16_mask.float().cpu().numpy() * valid
    fp32_descriptor = _candidate_descriptors_from_masks(
        features=feature_numpy,
        masks=fp32_mask,
        valid=valid,
        bbox=bbox,
        keeper=keeper,
    )
    fp64_descriptor = _candidate_descriptors_from_masks(
        features=feature_numpy.astype(np.float64),
        masks=fp64_mask,
        valid=valid,
        bbox=bbox,
        keeper=keeper,
    )
    bf16_descriptor = _candidate_descriptors_from_masks(
        features=feature_numpy,
        masks=bf16_mask_numpy,
        valid=valid,
        bbox=bbox,
        keeper=keeper,
    )
    fp32_scores = _readout_probabilities_from_state(fp32_descriptor, state)
    fp64_scores = _readout_probabilities_from_state(fp64_descriptor, state)
    bf16_scores = _readout_probabilities_from_state(bf16_descriptor, state)
    fp32_fp64_mask_error = float(np.max(np.abs(fp32_mask - fp64_mask)))
    fp32_fp64_descriptor_error = float(
        np.max(np.abs(fp32_descriptor - fp64_descriptor))
    )
    fp32_fp64_score_error = float(np.max(np.abs(fp32_scores - fp64_scores)))
    bf16_fp32_mask_error = float(np.max(np.abs(bf16_mask_numpy - fp32_mask)))
    bf16_fp32_descriptor_error = float(
        np.max(np.abs(bf16_descriptor - fp32_descriptor))
    )
    bf16_fp32_score_error = float(np.max(np.abs(bf16_scores - fp32_scores)))
    threshold = float(state["threshold"])
    non_near_tie = np.abs(fp32_scores - threshold) > 1e-3
    action_exact = bool(
        np.array_equal(
            fp32_scores[non_near_tie] >= threshold,
            bf16_scores[non_near_tie] >= threshold,
        )
    )
    checks = {
        "fp32_fp64_mask_error_le_1e_5": fp32_fp64_mask_error <= 1e-5,
        "fp32_fp64_descriptor_error_le_1e_5": fp32_fp64_descriptor_error <= 1e-5,
        "fp32_fp64_score_error_le_1e_5": fp32_fp64_score_error <= 1e-5,
        "bf16_fp32_mask_error_le_0_01": bf16_fp32_mask_error <= 0.01,
        "bf16_fp32_descriptor_error_le_0_01": bf16_fp32_descriptor_error <= 0.01,
        "bf16_fp32_score_error_le_0_01": bf16_fp32_score_error <= 0.01,
        "bf16_non_near_tie_actions_exact": action_exact,
    }
    bf16_head.cpu()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fp32_fp64_mask_max_abs_error": fp32_fp64_mask_error,
        "fp32_fp64_descriptor_max_abs_error": fp32_fp64_descriptor_error,
        "fp32_fp64_score_max_abs_error": fp32_fp64_score_error,
        "bf16_fp32_mask_max_abs_error": bf16_fp32_mask_error,
        "bf16_fp32_descriptor_max_abs_error": bf16_fp32_descriptor_error,
        "bf16_fp32_score_max_abs_error": bf16_fp32_score_error,
        "non_near_tie_rows": int(non_near_tie.sum()),
    }


def benchmark_mask_path(
    *,
    cohort_artifacts: Mapping[str, object],
    oof: Mapping[str, object],
    readout_states: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    fold = 0
    selected = np.flatnonzero(np.asarray(cohort_artifacts["folds"]) == fold)[:BATCH_SIZE]
    features = torch.from_numpy(
        np.asarray(cohort_artifacts["features"][selected], dtype=np.float32)
    ).to(device)
    boxes = torch.from_numpy(
        np.asarray(cohort_artifacts["crop_boxes"])[selected].astype(np.float32)
    ).to(device)
    valid = torch.from_numpy(
        np.asarray(cohort_artifacts["valid_masks"])[selected].astype(np.float32)
    ).to(device)
    bbox = torch.from_numpy(
        np.asarray(cohort_artifacts["bbox_masks"])[selected].astype(np.float32)
    ).to(device)
    keeper = torch.from_numpy(
        np.asarray(cohort_artifacts["keeper_probabilities"])[selected].astype(np.float32)
    ).to(device)
    candidate = StaticMaskDescriptorExport(
        oof["fold_records"][fold]["models"][CANDIDATE_ROLE],
        _candidate_fold_state(readout_states, fold),
    ).to(device).eval()
    control = StaticBBoxDescriptor().to(device).eval()

    def measure(module: nn.Module, repeats: int = 30) -> Tuple[float, int]:
        with torch.inference_mode():
            for _ in range(10):
                module(features, boxes, valid, bbox, keeper)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        durations = []
        with torch.inference_mode():
            for _ in range(repeats):
                started = time.perf_counter()
                module(features, boxes, valid, bbox, keeper)
                torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - started)
        return float(np.median(durations)), int(torch.cuda.max_memory_allocated(device))

    control_seconds, control_peak = measure(control)
    candidate_seconds, candidate_peak = measure(candidate)
    ratio = candidate_seconds / max(control_seconds, 1e-12)
    extra_bytes = max(candidate_peak - control_peak, 0)
    checks = {
        "runtime_ratio_le_1_15": ratio <= 1.15,
        "extra_peak_cuda_le_0_50_gib": extra_bytes <= int(0.50 * 1024**3),
        "total_peak_cuda_le_3_5_gib": candidate_peak <= int(3.5 * 1024**3),
    }
    candidate.cpu()
    control.cpu()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "batch_size": int(selected.size),
        "repeats": 30,
        "bbox_descriptor_median_seconds": control_seconds,
        "candidate_median_seconds": candidate_seconds,
        "candidate_runtime_ratio": ratio,
        "bbox_peak_cuda_bytes": control_peak,
        "candidate_peak_cuda_bytes": candidate_peak,
        "extra_peak_cuda_bytes": extra_bytes,
    }


def _gpu_snapshot() -> Dict[str, object]:
    result: Dict[str, object] = {}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ).strip()
        result["gpu"] = output
    except (OSError, subprocess.CalledProcessError) as error:
        result["gpu_query_error"] = str(error)
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


def apply_frozen_condition(
    *,
    name: str,
    extraction: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    oof: Mapping[str, object],
    readout_states: Mapping[str, object],
    clean_candidate_auroc: float,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    count = len(cohort)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    masks = {
        role: np.empty((count, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float32)
        for role in TRAINED_ROLES
    }
    for fold in FOLDS:
        selected = np.flatnonzero(folds == fold)
        heads = oof["fold_records"][fold]["models"]
        for start in range(0, selected.size, BATCH_SIZE):
            positions = selected[start:start + BATCH_SIZE]
            features, _lab, valid, _bbox, boxes, _indices = _batch_from_cache(
                extraction, positions, device=device
            )
            with torch.inference_mode():
                for role in TRAINED_ROLES:
                    head = heads[role].to(device).float().eval()
                    value = head(features, boxes)[:, 0] * valid.float()
                    masks[role][positions] = value.cpu().numpy()
                    head.cpu()
            del features, _lab, valid, _bbox, boxes
    valid = np.asarray(extraction["valid_masks"], dtype=np.bool_)
    bbox = np.asarray(extraction["bbox_masks"], dtype=np.bool_)
    masks[VALID_ROLE] = valid.astype(np.float32)
    masks[BBOX_ROLE] = bbox.astype(np.float32)
    masks[ROLLED_ROLE] = _rolled_candidate_masks(
        masks[CANDIDATE_ROLE], indices, folds
    ) * valid.astype(np.float32)
    keeper_probabilities = np.asarray(extraction["probabilities"], dtype=np.float64)
    projection = torch.from_numpy(fixed_projection().astype(np.float32)).to(device)
    chunks = {role: [] for role in ROLE_NAMES}
    for start in range(0, count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, count)
        feature = torch.from_numpy(
            np.asarray(extraction["features"][start:stop], dtype=np.float32)
        ).to(device)
        with torch.inference_mode():
            projected = torch.einsum("bchw,cd->bdhw", feature, projection).cpu().numpy()
        built = build_descriptors(
            projected,
            {role: masks[role][start:stop] for role in MASK_ROLES},
            valid[start:stop],
            bbox[start:stop],
            keeper_probabilities[start:stop],
        )
        for role in ROLE_NAMES:
            chunks[role].append(built[role])
    descriptors = {role: np.concatenate(value, axis=0) for role, value in chunks.items()}
    scores, actions = apply_readout_states(descriptors, folds, readout_states)
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    keeper = np.asarray([row.keeper_prediction for row in cohort], dtype=np.int64)
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
    gates = {
        "candidate_auroc_ge_0_80": candidate["auroc"] >= 0.80,
        "candidate_tp_retention_ge_0_92": candidate["keeper_tp_retention"] >= 0.92,
        "candidate_fp_rejection_ge_0_15": candidate["restricted_fp_rejection"] >= 0.15,
        "candidate_auroc_margin_ge_0_01_over_bbox": (
            candidate["auroc"] - metrics[BBOX_ROLE]["auroc"] >= 0.01
        ),
        "candidate_auroc_margin_ge_0_01_over_dephased": (
            candidate["auroc"] - metrics[DEPHASED_ROLE]["auroc"] >= 0.01
        ),
        "candidate_auroc_drop_le_0_05": clean_candidate_auroc - candidate["auroc"] <= 0.05,
    }
    path = output_dir / f"robustness_{name}.npz"
    np.savez_compressed(
        path,
        sample_indices=indices,
        folds=folds,
        **{f"mask__{role}": masks[role].astype(np.float16) for role in MASK_ROLES},
        **{f"descriptor__{role}": descriptors[role] for role in ROLE_NAMES},
        **{f"score__{role}": scores[role] for role in ROLE_NAMES},
        **{f"action__{role}": actions[role] for role in ROLE_NAMES},
    )
    replay_scores, replay_actions = apply_readout_states(descriptors, folds, readout_states)
    replay_error = max(
        float(np.max(np.abs(replay_scores[role] - scores[role]))) for role in ROLE_NAMES
    )
    replay_actions_exact = all(
        np.array_equal(replay_actions[role], actions[role]) for role in ROLE_NAMES
    )
    return {
        "condition": name,
        "metrics": metrics,
        "gates": {key: bool(value) for key, value in gates.items()},
        "replay_score_max_abs_error": replay_error,
        "replay_actions_exact": replay_actions_exact,
        "artifact": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "passed": bool(all(gates.values()) and replay_error <= MAX_REPLAY_ERROR and replay_actions_exact),
    }


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    equations = engineering_checks()
    if not bool(equations["passed"]):
        raise RuntimeError(f"BoxInst equation checks failed: {equations}")
    rows = locked_training_rows(
        _read_clean_train_rows(Path(provenance["files"]["cidt_predictions"]["path"]))
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
        "training_population": {
            "rows": len(rows),
            "holdout_counts": list(EXPECTED_HOLDOUT_COUNTS),
            "ordered_index_sha256": EXPECTED_TRAIN_INDEX_SHA256,
        },
        "cohort": {
            "rows": len(cohort),
            "keeper_tp": EXPECTED_KEEPER_TP,
            "keeper_fn": EXPECTED_KEEPER_FN,
            "restricted_fp": EXPECTED_RESTRICTED_FP,
            "ordered_index_sha256": EXPECTED_COHORT_INDEX_SHA256,
        },
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "implementation_tracking_observed_not_gated_until_formal": implementation_tracking_state(),
    }


def engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for BoxInst engineering forward")
    provenance = verify_locked_inputs(args)
    device = torch.device("cuda")
    model, checkpoint, _ = load_model(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    rows = locked_training_rows(
        _read_clean_train_rows(Path(provenance["files"]["cidt_predictions"]["path"]))
    )
    full_cohort = locked_cohort(rows)
    positive = next(
        row
        for row in full_cohort
        if row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
    )
    negative = next(row for row in full_cohort if row.target != FOCUS_CLASS)
    selected = [positive, negative]
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, Path(provenance["files"]["data_yaml"]["path"])
    )
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in selected],
        brightness=1.0,
        contrast=1.0,
        batch_size=2,
        num_workers=0,
        context="boxinst_foreground_mask_a0_engineering",
    )
    cleanup = {"deleted": False}
    with tempfile.TemporaryDirectory(prefix="trkh_boxinst_engineering_") as temporary:
        extraction = extract_condition(
            model=model,
            loader=loader,
            loader_summary=loader_summary,
            rows=selected,
            device=device,
            semantics=dataset_summary["semantics"],
            condition="engineering",
            feature_cache_path=Path(temporary) / "features.npy",
            lab_cache_path=Path(temporary) / "lab.npy",
            benchmark_ordinary=True,
        )
        features, lab, valid, bbox, boxes, _indices = _batch_from_cache(
            extraction, np.asarray([0, 1]), device=device
        )
        heads = build_matched_heads(fold=0)
        actual_checks = {}
        for role, head in heads.items():
            head = head.to(device).float().train()
            prediction = head(features, boxes)[:, 0] * valid.float()
            projection = projection_loss_torch(prediction, bbox.float()).mean()
            pairwise = projection.new_zeros(())
            if role != PROJECTION_ROLE:
                pairwise = pairwise_affinity_torch(
                    prediction, lab, valid, bbox
                )[0].mean()
            loss = projection + pairwise
            loss.backward()
            actual_checks[role] = {
                "loss_finite": bool(torch.isfinite(loss)),
                "all_gradients_nonzero_finite": all(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and float(parameter.grad.abs().sum()) > 0.0
                    for parameter in head.parameters()
                ),
                "mask_nonconstant": bool(
                    torch.all(prediction.flatten(1).std(dim=1) > 0.0)
                ),
            }
        cleanup = _close_delete_caches(extraction)
    passed = bool(
        all(extraction["checks"].values())
        and all(all(record.values()) for record in actual_checks.values())
        and cleanup["deleted"]
    )
    return {
        "mode": f"{METHOD}_engineering_forward",
        "passed": passed,
        "output_created": False,
        "rows": len(selected),
        "validation_data_used": False,
        "test_data_used": False,
        "extraction_checks": extraction["checks"],
        "actual_head_checks": actual_checks,
        "capture_probability_max_abs_error": extraction["capture_probability_max_abs_error"],
        "cidt_probability_max_abs_error": extraction["cidt_probability_max_abs_error"],
        "temporary_cache_cleanup": cleanup,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_BOXINST_MASK_A0_PREFLIGHT") != "passed":
        raise RuntimeError("Formal BoxInst A0 must use the locked PowerShell preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal BoxInst A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    implementation_state = implementation_tracking_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(f"Formal BoxInst A0 requires clean pushed tracked state: {repo_state}")
    if not bool(implementation_state["passed"]):
        raise ValueError(
            "Formal BoxInst A0 requires committed implementation blobs: "
            f"{implementation_state}"
        )
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    equations = engineering_checks()
    if not bool(equations["passed"]):
        raise RuntimeError("BoxInst equations changed before formal execution")
    device = torch.device("cuda")
    gpu_before = _gpu_snapshot()
    model, checkpoint, class_names = load_model(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    rows = locked_training_rows(
        _read_clean_train_rows(Path(provenance["files"]["cidt_predictions"]["path"]))
    )
    cohort = locked_cohort(rows)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, Path(provenance["files"]["data_yaml"]["path"])
    )
    clean_loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in rows],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="boxinst_foreground_mask_a0_clean",
    )
    extraction: Optional[Dict[str, object]] = None
    cleanup: Dict[str, object] = {"deleted": False}
    robustness_results: Dict[str, object] = {}
    try:
        extraction = extract_condition(
            model=model,
            loader=clean_loader,
            loader_summary=loader_summary,
            rows=rows,
            device=device,
            semantics=dataset_summary["semantics"],
            condition="clean",
            feature_cache_path=output_dir / "_temporary_train_block2_features.npy",
            lab_cache_path=output_dir / "_temporary_train_lab.npy",
            benchmark_ordinary=True,
        )
        if not all(extraction["checks"].values()):
            raise RuntimeError(f"BoxInst clean extraction failed: {extraction['checks']}")
        oof = train_oof_mask_heads(
            extraction=extraction,
            rows=rows,
            device=device,
        )
        statistics = mask_statistics(
            masks=oof["masks"], extraction=extraction, device=device
        )
        cohort_artifacts = build_cohort_artifacts(
            output_dir=output_dir,
            extraction=extraction,
            rows=rows,
            cohort=cohort,
            oof=oof,
            device=device,
        )
        labels = (cohort_artifacts["targets"] == FOCUS_CLASS).astype(np.int64)
        scores, actions, readout_states = fit_oof_readouts(
            cohort_artifacts["descriptors"],
            labels,
            cohort_artifacts["folds"],
            [row.source_stem for row in cohort],
        )
        clean_analysis = build_clean_analysis(
            scores=scores,
            actions=actions,
            targets=cohort_artifacts["targets"],
            keeper_predictions=cohort_artifacts["keeper_predictions"],
            folds=cohort_artifacts["folds"],
            statistics=statistics,
        )
        direct_xai = render_mask_contact_sheet(
            output_dir / "boxinst_foreground_masks_contact_sheet.png",
            cohort=cohort,
            cohort_artifacts=cohort_artifacts,
        )
        gradient_xai = render_gradient_contact_sheet(
            output_dir / "boxinst_feature_gradients_contact_sheet.png",
            cohort=cohort,
            cohort_artifacts=cohort_artifacts,
            readout_states=readout_states,
            persisted_scores=scores,
        )
        export = static_export_checks(
            output_dir=output_dir,
            cohort_artifacts=cohort_artifacts,
            oof=oof,
            readout_states=readout_states,
        )
        numeric = numeric_precision_checks(
            cohort_artifacts=cohort_artifacts,
            oof=oof,
            readout_states=readout_states,
            device=device,
        )
        resource = benchmark_mask_path(
            cohort_artifacts=cohort_artifacts,
            oof=oof,
            readout_states=readout_states,
            device=device,
        )
        artifacts = save_replay_artifacts(
            output_dir=output_dir,
            rows=rows,
            cohort=cohort,
            extraction=extraction,
            oof=oof,
            cohort_artifacts=cohort_artifacts,
            readout_states=readout_states,
            scores=scores,
            actions=actions,
            statistics=statistics,
        )
        in_process_replay = replay_artifacts(
            output_dir, expected_analysis=clean_analysis
        )
        loader = extraction["runtime"]["loader"]
        structural_gates = {
            "equation_checks_passed": bool(equations["passed"]),
            "locked_extraction_checks_passed": all(extraction["checks"].values()),
            "source_disjoint_mask_folds": all(
                not oof["fold_records"][fold]["source_overlap"] for fold in FOLDS
            ),
            "matched_initial_states_exact": all(
                len(set(oof["fold_records"][fold]["initial_hashes"].values())) == 1
                for fold in FOLDS
            ),
            "shared_occurrence_order_across_roles": bool(oof["shared_occurrence_order"]),
            "matched_update_counts": all(
                len({record["updates"] for record in oof["training_records"] if record["fold"] == fold}) == 1
                for fold in FOLDS
            ),
            "all_parameters_receive_gradient_and_change": all(
                record["gradient_pass"] and record["update_pass"]
                for record in oof["training_records"]
            ),
            "optimizer_and_parameters_fp32_finite": all(
                record["parameter_dtypes"] == ["torch.float32"]
                and record["optimizer"]["state_tensor_dtypes"] == ["torch.float32"]
                and record["optimizer"]["all_state_tensors_finite"]
                for record in oof["training_records"]
            ),
            "requested_effective_workers_pin_persistent_exact": (
                int(loader["requested_num_workers"]) == NUM_WORKERS
                and int(loader["effective_num_workers"]) == NUM_WORKERS
                and bool(loader["effective_pin_memory"])
                and bool(loader["persistent_workers"])
            ),
            "direct_xai_automatic_passed": bool(direct_xai["automatic_passed"]),
            "gradient_xai_automatic_passed": bool(gradient_xai["automatic_passed"]),
            "static_export_passed": bool(export["passed"]),
            "numeric_precision_passed": bool(numeric["passed"]),
            "resource_gates_passed": bool(resource["passed"]),
            "in_process_replay_passed": bool(in_process_replay["passed"]),
            "temporary_caches_hash_attested": bool(
                extraction["features_cache"]["sha256"]
                and extraction["lab_cache"]["sha256"]
            ),
            "unknown_process_not_terminated": True,
        }
        clean_gate_passed = bool(
            all(structural_gates.values())
            and clean_analysis["mechanism_gates_passed"]
        )
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
                    context=f"boxinst_foreground_mask_a0_{name}",
                )
                condition_extraction: Optional[Dict[str, object]] = None
                condition_cleanup: Dict[str, object] = {"deleted": False}
                try:
                    condition_extraction = extract_condition(
                        model=model,
                        loader=condition_loader,
                        loader_summary=condition_loader_summary,
                        rows=cohort,
                        device=device,
                        semantics=dataset_summary["semantics"],
                        condition=name,
                        feature_cache_path=output_dir / f"_temporary_{name}_features.npy",
                        lab_cache_path=output_dir / f"_temporary_{name}_lab.npy",
                        benchmark_ordinary=False,
                    )
                    condition_result = apply_frozen_condition(
                        name=name,
                        extraction=condition_extraction,
                        cohort=cohort,
                        oof=oof,
                        readout_states=readout_states,
                        clean_candidate_auroc=float(
                            clean_analysis["roles"][CANDIDATE_ROLE]["auroc"]
                        ),
                        device=device,
                        output_dir=output_dir,
                    )
                finally:
                    if condition_extraction is not None:
                        condition_cleanup = _close_delete_caches(condition_extraction)
                condition_result["temporary_cache_cleanup"] = condition_cleanup
                condition_result["passed"] = bool(
                    condition_result["passed"] and condition_cleanup["deleted"]
                )
                robustness_results[name] = condition_result
            robustness_passed = all(
                bool(value["passed"]) for value in robustness_results.values()
            )
        else:
            robustness_results = {
                "skipped": True,
                "reason": "clean automatic conjunctive gate failed",
            }
            robustness_passed = False
        automated_pre_replay = bool(
            clean_gate_passed and robustness_passed
        )
        extraction_summary = {
            "checks": dict(extraction["checks"]),
            "runtime": dict(extraction["runtime"]),
            "capture_probability_max_abs_error": extraction["capture_probability_max_abs_error"],
            "cidt_probability_max_abs_error": extraction["cidt_probability_max_abs_error"],
            "features_cache": dict(extraction["features_cache"]),
            "lab_cache": dict(extraction["lab_cache"]),
        }
    finally:
        if extraction is not None and ("features" in extraction or "lab" in extraction):
            cleanup = _close_delete_caches(extraction)
    if not cleanup.get("deleted", False):
        raise RuntimeError(f"Temporary BoxInst cache cleanup failed: {cleanup}")
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
        "training_population": {
            "rows": len(rows),
            "class_counts": list(EXPECTED_TRAIN_CLASS_COUNTS),
            "holdout_counts": list(EXPECTED_HOLDOUT_COUNTS),
            "ordered_index_sha256": EXPECTED_TRAIN_INDEX_SHA256,
        },
        "cohort": {
            "rows": len(cohort),
            "keeper_tp": EXPECTED_KEEPER_TP,
            "keeper_fn": EXPECTED_KEEPER_FN,
            "restricted_fp": EXPECTED_RESTRICTED_FP,
            "ordered_index_sha256": EXPECTED_COHORT_INDEX_SHA256,
        },
        "equation_checks": equations,
        "clean_extraction": {
            **extraction_summary,
            "temporary_cache_cleanup": cleanup,
            "persisted": False,
        },
        "training": {
            "roles": list(TRAINED_ROLES),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "pairwise_size": PAIRWISE_SIZE,
            "pairwise_dilation": PAIRWISE_DILATION,
            "pairwise_color_threshold": PAIRWISE_COLOR_THRESHOLD,
            "amp": None,
            "scheduler": None,
            "early_stopping": None,
            "occurrence_hashes": oof["occurrence_hashes"],
            "seconds": oof["seconds"],
            "records": oof["training_records"],
        },
        "clean_analysis": clean_analysis,
        "structural_gates": structural_gates,
        "structural_gates_passed": all(structural_gates.values()),
        "clean_gate_passed": clean_gate_passed,
        "in_process_replay": {
            key: value for key, value in in_process_replay.items() if key != "analysis"
        },
        "numeric_precision": numeric,
        "static_export": export,
        "resource": {
            **resource,
            "requested_workers": NUM_WORKERS,
            "effective_loader": extraction_summary["runtime"]["loader"],
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "unknown_process_terminated": False,
        },
        "robustness_authorized_by_clean_gate": clean_gate_passed,
        "robustness_results": robustness_results,
        "robustness_passed": robustness_passed,
        "xai": {
            "direct_masks": direct_xai,
            "feature_gradients": gradient_xai,
            "manual_review": {"required": True, "completed": False, "passed": False},
        },
        "artifacts": artifacts,
        "automated_gate_pre_external_replay": automated_pre_replay,
        "external_replay": {"required": True, "completed": False, "passed": False},
        "visual_review": {"required": True, "completed": False, "passed": False},
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
        raise ValueError("Replay target is not BoxInst foreground-mask A0")
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
        "mask_max_abs_error": replay["mask_max_abs_error"],
        "descriptor_max_abs_error": replay["descriptor_max_abs_error"],
        "score_max_abs_error": replay["score_max_abs_error"],
        "threshold_max_abs_error": replay["threshold_max_abs_error"],
        "actions_exact": replay["actions_exact"],
        "analysis_maximum_numeric_difference": replay["analysis_maximum_numeric_difference"],
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
            f"Visual-review summary SHA differs: expected={expected_summary_sha256}, observed={observed_sha}"
        )
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Visual-review target is not BoxInst foreground-mask A0")
    if not bool(summary.get("external_replay", {}).get("completed", False)):
        raise ValueError("BoxInst visual review requires external replay")
    for key in ("direct_masks", "feature_gradients"):
        record = summary["xai"][key]
        if _sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"BoxInst XAI artifact differs before visual review: {key}")
    visual_passed = str(result) == "pass"
    automated_passed = bool(summary["automated_gate_passed"])
    a0_passed = bool(visual_passed and automated_passed)
    review = {
        "required": True,
        "completed": True,
        "passed": visual_passed,
        "decision": str(result),
        "reviewed_summary_sha256": observed_sha,
        "manifest_sha256_before_review": manifest_before["manifest_sha256"],
    }
    summary["xai"]["manual_review"] = review
    summary["visual_review"] = review
    summary["a0_passed"] = a0_passed
    summary["trainer_integration_authorized"] = a0_passed
    summary["matched_short_smoke_authorized"] = a0_passed
    summary["full_train_authorized"] = False
    summary["current_command_update_authorized"] = False
    summary["status"] = (
        "passed_a0_integration_smoke_only"
        if a0_passed
        else "rejected_manual_visual_gate"
        if automated_passed
        else "rejected_automated_gate_visual_review_recorded"
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    return {
        "mode": f"{METHOD}_visual_review",
        "status": summary["status"],
        "a0_passed": a0_passed,
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_after),
        "trainer_integration_authorized": a0_passed,
        "matched_short_smoke_authorized": a0_passed,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.replay_summary is not None:
        if args.finalize_visual_review is not None:
            if not args.expected_summary_sha256:
                raise ValueError("Visual review requires --expected-summary-sha256")
            result = finalize_visual_review(
                args.replay_summary,
                result=args.finalize_visual_review,
                expected_summary_sha256=args.expected_summary_sha256,
            )
        else:
            result = replay_summary(args.replay_summary)
    elif args.finalize_visual_review is not None:
        raise ValueError("Visual review requires --replay-summary")
    elif args.preflight_only:
        result = preflight(args)
    elif args.engineering_forward:
        result = engineering_forward(args)
    else:
        result = run_audit(args)
    print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

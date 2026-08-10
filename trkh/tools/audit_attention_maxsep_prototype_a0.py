from __future__ import annotations

"""Locked train-only attention-aware maximal-separation prototype A0.

The implementation follows the equations in Nakka and Salzmann, ACCV 2020,
and is independent from the authors' unlicensed source tree. ProtoPNet is used
only as pinned, MIT-licensed architectural provenance; no external module is
imported here.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import subprocess
import tempfile
import time
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
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
from trkh.tools.audit_boxinst_foreground_mask_a0 import (
    locked_cohort as _boxinst_locked_cohort,
    locked_training_rows as _boxinst_locked_training_rows,
)
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


METHOD = "attention_maxsep_prototype_a0"
MODE = "attention_maxsep_prototype_a0_train_information_gate"
SEED = 42
FOCUS_CLASS = 1
NUM_CLASSES = 5
FOLDS = (0, 1, 2, 3, 4)
FEATURE_CHANNELS = 256
FEATURE_SIZE = 16
PREFIX_TOKENS = 7
BLOCK_INDEX = 1
PROJECTION_HIDDEN = 64
PROJECTION_DIM = 32
PROTOTYPES_PER_CLASS = 5
NUM_PROTOTYPES = NUM_CLASSES * PROTOTYPES_PER_CLASS
PROTOTYPE_EPSILON = 1e-5
ATTENTION_EPSILON = 1e-6
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
WARM_EPOCHS = 5
JOINT_DECAY_EPOCH = 15
WARM_LEARNING_RATE = 3e-4
JOINT_LEARNING_RATE = 3e-3
JOINT_DECAYED_LEARNING_RATE = 3e-4
CLUSTER_WEIGHT = 100.0
SEPARATION_WEIGHT = 0.08
READOUT_C = 0.1
READOUT_MAX_ITER = 2000
MIN_FIT_POSITIVE_RETENTION = 0.97
ATTENTION_ROLL = (3, 5)
MAX_REPLAY_ERROR = 1e-7
MAX_CAPTURE_PROBABILITY_ERROR = 1e-6
MAX_CIDT_PROBABILITY_ERROR = 3e-5

TRAINED_ROLES = (
    "cross_sample_maxsep",
    "self_attention_maxsep",
    "cross_sample_cluster_only",
    "cross_sample_channel_dephased",
)
CANDIDATE_ROLE = TRAINED_ROLES[0]
SELF_ROLE = TRAINED_ROLES[1]
CLUSTER_ROLE = TRAINED_ROLES[2]
DEPHASED_ROLE = TRAINED_ROLES[3]
ROLLED_ROLE = "cross_sample_same_weight_attention_rolled"
CYCLED_ROLE = "cross_sample_same_weight_class_cycled"
KEEPER_ROLE = "keeper_logprob"
BBOX_KEEPER_ROLE = "bbox_geometry_plus_keeper"
BBOX_ONLY_ROLE = "bbox_geometry_only"
GAP_KEEPER_ROLE = "block2_gap_plus_keeper"
WITHOUT_KEEPER_ROLE = "cross_sample_maxsep_without_keeper"
LEARNED_DESCRIPTOR_ROLES = TRAINED_ROLES + (ROLLED_ROLE, CYCLED_ROLE)
ROLE_NAMES = LEARNED_DESCRIPTOR_ROLES + (
    KEEPER_ROLE,
    BBOX_KEEPER_ROLE,
    BBOX_ONLY_ROLE,
    GAP_KEEPER_ROLE,
    WITHOUT_KEEPER_ROLE,
)

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
LOCKED_PROTOCOL_SHA256 = "43a965074e1a9dc2f9074598fdedd0810ed3977a3b60df19bc4a88a0f3b5907b"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_MAXSEP_PAPER_SHA256 = "68a9c61c166762e42042968673de75fbfd0b2e930a4f492818a8fc199c96b4ff"
LOCKED_MAXSEP_SUPPLEMENT_SHA256 = "4e360302ed62a6f90b70685a464ac4c393bfa7e97a36db5fbf8590996d5f8f75"
LOCKED_MAXSEP_COMMIT = "20939384e02e7941790e7a8fa4498ae9fd0fe431"
LOCKED_MAXSEP_TREE = "5d26bc4ea1b1a40dc288a841fedadde2093e949b"
LOCKED_MAXSEP_README_SHA256 = "77a106d52aaa64b01464201fcb9cd4c897e2c65ed7f3017767d1cd9598ff7ce8"
LOCKED_MAXSEP_MODEL_SHA256 = "c69df11d38d654a86f8284c876d3c1e3d2823e7b48bd2fc2c9c6e60f3cde6fa8"
LOCKED_MAXSEP_POOL_SHA256 = "b2430fbb99bd98c1db97533140122529c13ac9bc4e730dafd19cc9d849fdd6ab"
LOCKED_MAXSEP_TRAIN_SHA256 = "f90995c5f12b087ea2576f20c6579a70c9e4c8d1d77548cdcbdc415e6ad9cd8d"
LOCKED_MAXSEP_CONFIG_SHA256 = "fe92bfb24e573fcf3b3f36c95a177f27c98c590aa0ac50218c897005920741ff"
LOCKED_PROTOPNET_PAPER_SHA256 = "d0ba7d150cfd179d2c7d1ab5a7f0e9b100bc4c70f9473d7b272ce757bb3b7ee3"
LOCKED_PROTOPNET_COMMIT = "81bf2b70cb60e4f36e25e8be386eb616b7459321"
LOCKED_PROTOPNET_TREE = "b6275c576911838e5dac07a6d481c1e79872ae01"
LOCKED_PROTOPNET_LICENSE_SHA256 = "f57a109eaff765d75409315952276a75b0d5c82aa6c7a19e190761193b92aaa3"
LOCKED_PROTOPNET_MODEL_SHA256 = "448b24ba025487f9c9ea7a1c9b8f17d410b88d1f507b1c56eb7b7b9ef15eb23b"
LOCKED_PROTOPNET_TRAIN_SHA256 = "6b183629e40c3d15b291c39820899ec066d42f18350f503392d35fcdd4b73fe3"

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
MAXSEP_ROOT = Path(r"D:\DataAI\external_sources\official\RobustFineGrained")
PROTOPNET_ROOT = Path(r"D:\DataAI\external_sources\official\ProtoPNet")
MAXSEP_PAPER = Path(r"D:\DataAI\external_sources\papers\Nakka_Maximal_Separation_ACCV_2020.pdf")
MAXSEP_SUPPLEMENT = Path(r"D:\DataAI\external_sources\papers\Nakka_Maximal_Separation_ACCV_2020_supp.pdf")
PROTOPNET_PAPER = Path(r"D:\DataAI\external_sources\papers\Chen_ProtoPNet_NeurIPS_2019.pdf")
IMPLEMENTATION_PATHS = (
    "trkh/tools/audit_attention_maxsep_prototype_a0.py",
    "tests/test_audit_attention_maxsep_prototype_a0.py",
    "scripts/run_trkh_attention_maxsep_prototype_a0.ps1",
)
CONDITIONS = (
    ("dim", 0.70, 0.90),
    ("bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only Attention-MaxSep Prototype A0 gate."
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
        / "TRKH_5CLASS_ATTENTION_MAXSEP_PROTOTYPE_A0_PROTOCOL_20260721.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_attention_maxsep_prototype_a0_20260721",
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
        raise ValueError("Attention-MaxSep A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError("Batch size or workers differ from prospective lock")
    if int(args.seed) != SEED:
        raise ValueError("Seed differs from prospective lock")


def _index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def implementation_tracking_state() -> Dict[str, object]:
    tracked = set(
        line.strip().replace("\\", "/")
        for line in _git_value(REPO_ROOT, "ls-files").splitlines()
        if line.strip()
    )
    status = _git_value(REPO_ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    dirty_paths = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"').replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty_paths.append(path)
    return {
        "paths": list(IMPLEMENTATION_PATHS),
        "all_tracked": all(path in tracked for path in IMPLEMENTATION_PATHS),
        "dirty_implementation_paths": sorted(
            path for path in dirty_paths if path in IMPLEMENTATION_PATHS
        ),
        "head": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "passed": all(path in tracked for path in IMPLEMENTATION_PATHS)
        and not any(path in IMPLEMENTATION_PATHS for path in dirty_paths),
    }


def _verify_external_repo(
    root: Path,
    *,
    commit: str,
    tree: str,
    label: str,
) -> Dict[str, object]:
    observed_commit = _git_value(root, "rev-parse", "HEAD")
    observed_tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    status = _git_value(root, "status", "--porcelain=v1")
    if observed_commit != commit or observed_tree != tree or status:
        raise ValueError(f"Pinned {label} source state differs")
    return {
        "path": str(root.resolve()),
        "commit": observed_commit,
        "tree": observed_tree,
        "clean": not bool(status),
    }


def verify_locked_inputs(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    hashes = {
        "keeper": _verify_hash(args.checkpoint, LOCKED_KEEPER_SHA256, "keeper"),
        "launcher_args": _verify_hash(
            args.launcher_args, LOCKED_LAUNCHER_ARGS_SHA256, "keeper launcher args"
        ),
        "resolved_config": _verify_hash(
            args.resolved_config, LOCKED_RESOLVED_CONFIG_SHA256, "keeper resolved config"
        ),
        "data_yaml": _verify_hash(args.data, LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_hash(
            args.cidt_summary, LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_hash(
            args.cidt_predictions, LOCKED_CIDT_PREDICTIONS_SHA256, "CIDT predictions"
        ),
        "protocol": _verify_hash(args.protocol, LOCKED_PROTOCOL_SHA256, "protocol"),
        "current_commands": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
            LOCKED_CURRENT_COMMAND_SHA256,
            "current commands",
        ),
        "command_history": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
            LOCKED_COMMAND_HISTORY_SHA256,
            "command history",
        ),
        "maxsep_paper": _verify_hash(
            MAXSEP_PAPER, LOCKED_MAXSEP_PAPER_SHA256, "MaxSep paper"
        ),
        "maxsep_supplement": _verify_hash(
            MAXSEP_SUPPLEMENT, LOCKED_MAXSEP_SUPPLEMENT_SHA256, "MaxSep supplement"
        ),
        "maxsep_readme": _verify_hash(
            MAXSEP_ROOT / "README.md", LOCKED_MAXSEP_README_SHA256, "MaxSep README"
        ),
        "maxsep_model": _verify_hash(
            MAXSEP_ROOT / "models" / "model_AttProto.py",
            LOCKED_MAXSEP_MODEL_SHA256,
            "MaxSep model source",
        ),
        "maxsep_pool": _verify_hash(
            MAXSEP_ROOT / "models" / "pooling.py",
            LOCKED_MAXSEP_POOL_SHA256,
            "MaxSep pooling source",
        ),
        "maxsep_train": _verify_hash(
            MAXSEP_ROOT / "train_and_test.py",
            LOCKED_MAXSEP_TRAIN_SHA256,
            "MaxSep training source",
        ),
        "maxsep_config": _verify_hash(
            MAXSEP_ROOT / "configs" / "AttProto" / "cub200_vgg16_normal.yaml",
            LOCKED_MAXSEP_CONFIG_SHA256,
            "MaxSep config",
        ),
        "protopnet_paper": _verify_hash(
            PROTOPNET_PAPER, LOCKED_PROTOPNET_PAPER_SHA256, "ProtoPNet paper"
        ),
        "protopnet_license": _verify_hash(
            PROTOPNET_ROOT / "LICENSE",
            LOCKED_PROTOPNET_LICENSE_SHA256,
            "ProtoPNet license",
        ),
        "protopnet_model": _verify_hash(
            PROTOPNET_ROOT / "model.py",
            LOCKED_PROTOPNET_MODEL_SHA256,
            "ProtoPNet model source",
        ),
        "protopnet_train": _verify_hash(
            PROTOPNET_ROOT / "train_and_test.py",
            LOCKED_PROTOPNET_TRAIN_SHA256,
            "ProtoPNet training source",
        ),
    }
    if (MAXSEP_ROOT / "LICENSE").exists():
        raise ValueError("MaxSep source unexpectedly gained a license; refresh provenance lock")
    return {
        "hashes": hashes,
        "maxsep_repo": _verify_external_repo(
            MAXSEP_ROOT,
            commit=LOCKED_MAXSEP_COMMIT,
            tree=LOCKED_MAXSEP_TREE,
            label="MaxSep",
        ),
        "protopnet_repo": _verify_external_repo(
            PROTOPNET_ROOT,
            commit=LOCKED_PROTOPNET_COMMIT,
            tree=LOCKED_PROTOPNET_TREE,
            label="ProtoPNet",
        ),
        "maxsep_license_absent": not (MAXSEP_ROOT / "LICENSE").exists(),
        "external_imported": False,
        "implementation": implementation_tracking_state(),
        "protected_untracked": _protected_untracked_state(),
    }


def locked_training_rows(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    selected = _boxinst_locked_training_rows(rows)
    if len(selected) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Attention-MaxSep train row count differs")
    if _index_sha256([row.sample_index for row in selected]) != EXPECTED_TRAIN_INDEX_SHA256:
        raise ValueError("Attention-MaxSep train index SHA differs")
    return selected


def locked_cohort(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    cohort = _boxinst_locked_cohort(rows)
    if len(cohort) != EXPECTED_COHORT_ROWS:
        raise ValueError("Attention-MaxSep cohort row count differs")
    if _index_sha256([row.sample_index for row in cohort]) != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError("Attention-MaxSep cohort index SHA differs")
    return cohort


def _class_identity(device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
    identity = torch.zeros(NUM_PROTOTYPES, NUM_CLASSES, device=device, dtype=dtype)
    for prototype in range(NUM_PROTOTYPES):
        identity[prototype, prototype // PROTOTYPES_PER_CLASS] = 1.0
    return identity


def _classifier_weights(
    identity: Tensor,
    *,
    own_weight: float = 1.0,
    rival_weight: float = -0.5,
) -> Tensor:
    return own_weight * identity.T + rival_weight * (1.0 - identity.T)


class AttentionMaxSepPrototypeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, PROJECTION_HIDDEN, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(PROJECTION_HIDDEN, PROJECTION_DIM, kernel_size=1),
            nn.Sigmoid(),
        )
        self.class_agnostic = nn.Conv2d(PROJECTION_DIM, 1, kernel_size=1)
        self.class_specific = nn.Conv2d(PROJECTION_DIM, NUM_CLASSES, kernel_size=1)
        self.prototypes = nn.Parameter(
            torch.rand(NUM_PROTOTYPES, PROJECTION_DIM, 1, 1)
        )
        identity = _class_identity()
        self.register_buffer("prototype_class_identity", identity)
        self.register_buffer("classifier_weight", _classifier_weights(identity))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _distances(self, projected: Tensor) -> Tensor:
        x2 = projected.square().sum(dim=1, keepdim=True)
        p2 = self.prototypes.square().sum(dim=(1, 2, 3)).view(1, -1, 1, 1)
        xp = F.conv2d(projected, self.prototypes)
        return (x2 - 2.0 * xp + p2).clamp_min(0.0)

    @staticmethod
    def _normalize_valid(valid: Tensor, dtype: torch.dtype) -> Tensor:
        mask = valid
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("Valid mask must be [B,1,H,W] or [B,H,W]")
        if not bool(mask.flatten(1).any(dim=1).all()):
            raise ValueError("Every row needs nonempty valid support")
        return mask.to(dtype=dtype)

    def forward(
        self,
        features: Tensor,
        valid: Tensor,
        *,
        attention_roll: Optional[Tuple[int, int]] = None,
        class_cycle: bool = False,
    ) -> Dict[str, Tensor]:
        if features.ndim != 4 or tuple(features.shape[1:]) != (
            FEATURE_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ):
            raise ValueError("MaxSep features must be [B,256,16,16]")
        valid_float = self._normalize_valid(valid, features.dtype)
        projected = self.projector(features)
        class_agnostic = self.class_agnostic(projected)
        class_specific = self.class_specific(projected)
        attention_product = class_agnostic * class_specific
        valid_count = valid_float.sum(dim=(2, 3)).clamp_min(1.0)
        attention_logits = (attention_product * valid_float).sum(dim=(2, 3)) / valid_count
        attention = attention_product.relu().amax(dim=1, keepdim=True) * valid_float
        attention_peak = attention.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        attention = attention / (attention_peak + ATTENTION_EPSILON)

        distances = self._distances(projected)
        similarities = torch.log(
            (distances + 1.0) / (distances + PROTOTYPE_EPSILON)
        )
        evidence_attention = attention
        if attention_roll is not None:
            evidence_attention = torch.roll(
                evidence_attention,
                shifts=(int(attention_roll[0]), int(attention_roll[1])),
                dims=(-2, -1),
            )
            evidence_attention = evidence_attention * valid_float
        evidence_maps = similarities * evidence_attention
        evidence_maps = evidence_maps.masked_fill(valid_float == 0, -torch.inf)
        prototype_evidence = evidence_maps.flatten(2).amax(dim=2)
        if not bool(torch.isfinite(prototype_evidence).all()):
            raise ValueError("Prototype evidence is non-finite")

        identity = self.prototype_class_identity
        if class_cycle:
            identity = torch.roll(identity, shifts=1, dims=1)
        classifier = _classifier_weights(identity).to(
            device=prototype_evidence.device, dtype=prototype_evidence.dtype
        )
        prototype_logits = prototype_evidence @ classifier.T
        grouped_evidence = prototype_evidence.view(
            prototype_evidence.shape[0], NUM_CLASSES, PROTOTYPES_PER_CLASS
        )
        if class_cycle:
            grouped_evidence = torch.roll(grouped_evidence, shifts=1, dims=1)
        class_evidence_max = grouped_evidence.amax(dim=2)
        class_evidence_mean = grouped_evidence.mean(dim=2)

        spatial_min = distances.masked_fill(valid_float == 0, torch.inf).flatten(2).amin(dim=2)
        grouped_distance = spatial_min.view(
            spatial_min.shape[0], NUM_CLASSES, PROTOTYPES_PER_CLASS
        )
        if class_cycle:
            grouped_distance = torch.roll(grouped_distance, shifts=1, dims=1)
        class_min_distance = grouped_distance.amin(dim=2)
        return {
            "projected": projected,
            "class_agnostic": class_agnostic,
            "class_specific": class_specific,
            "attention_product": attention_product,
            "attention": attention,
            "evidence_attention": evidence_attention,
            "attention_logits": attention_logits,
            "distances": distances,
            "similarities": similarities,
            "evidence_maps": evidence_maps,
            "prototype_evidence": prototype_evidence,
            "prototype_logits": prototype_logits,
            "class_evidence_max": class_evidence_max,
            "class_evidence_mean": class_evidence_mean,
            "class_min_distance": class_min_distance,
            "valid": valid_float,
        }


def maxsep_regularization(
    outputs: Mapping[str, Tensor],
    targets: Tensor,
    *,
    cross_sample: bool,
    include_separation: bool,
) -> Dict[str, Tensor]:
    distances = outputs["distances"]
    attention = outputs["attention"]
    valid = outputs["valid"]
    identity = _class_identity(device=distances.device, dtype=distances.dtype)
    own = identity[:, targets].T[:, :, None, None]
    wrong = 1.0 - own
    nearest_own = distances.masked_fill(own == 0, torch.inf).amin(dim=1, keepdim=True)
    nearest_wrong = distances.masked_fill(wrong == 0, torch.inf).amin(dim=1, keepdim=True)
    if cross_sample:
        coordinate_count = valid.sum(dim=0, keepdim=True).clamp_min(1.0)
        weight = attention.sum(dim=0, keepdim=True) / coordinate_count
        weight = weight.expand_as(attention) * valid
    else:
        weight = attention * valid
    denominator = weight.sum().clamp_min(ATTENTION_EPSILON)
    cluster = (weight * nearest_own).sum() / denominator
    separation_distance = (weight * nearest_wrong).sum() / denominator
    regularization = CLUSTER_WEIGHT * cluster
    if include_separation:
        regularization = regularization - SEPARATION_WEIGHT * separation_distance
    return {
        "cluster": cluster,
        "separation_distance": separation_distance,
        "regularization": regularization,
        "weight": weight,
        "nearest_own": nearest_own,
        "nearest_wrong": nearest_wrong,
    }


def maxsep_total_loss(
    head: AttentionMaxSepPrototypeHead,
    features: Tensor,
    valid: Tensor,
    targets: Tensor,
    *,
    cross_sample: bool,
    include_separation: bool,
    include_regularization: bool,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    outputs = head(features, valid)
    prototype_ce = F.cross_entropy(outputs["prototype_logits"], targets)
    attention_ce = F.cross_entropy(outputs["attention_logits"], targets)
    regularization = maxsep_regularization(
        outputs,
        targets,
        cross_sample=cross_sample,
        include_separation=include_separation,
    )
    total = prototype_ce + attention_ce
    if include_regularization:
        total = total + regularization["regularization"]
    return total, {
        **outputs,
        **regularization,
        "prototype_ce": prototype_ce,
        "attention_ce": attention_ce,
        "total_loss": total,
    }


def _numpy_conv1x1(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.einsum("bchw,oc->bohw", value, weight[:, :, 0, 0]) + bias[None, :, None, None]


def maxsep_numpy(
    features: np.ndarray,
    valid: np.ndarray,
    state: Mapping[str, np.ndarray],
    targets: np.ndarray,
    *,
    cross_sample: bool = True,
    include_separation: bool = True,
) -> Dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.float64)
    if mask.ndim == 3:
        mask = mask[:, None]
    first = _numpy_conv1x1(
        x,
        np.asarray(state["projector.0.weight"], dtype=np.float64),
        np.asarray(state["projector.0.bias"], dtype=np.float64),
    )
    first = np.maximum(first, 0.0)
    second = _numpy_conv1x1(
        first,
        np.asarray(state["projector.2.weight"], dtype=np.float64),
        np.asarray(state["projector.2.bias"], dtype=np.float64),
    )
    projected = 1.0 / (1.0 + np.exp(-second))
    ca = _numpy_conv1x1(
        projected,
        np.asarray(state["class_agnostic.weight"], dtype=np.float64),
        np.asarray(state["class_agnostic.bias"], dtype=np.float64),
    )
    cs = _numpy_conv1x1(
        projected,
        np.asarray(state["class_specific.weight"], dtype=np.float64),
        np.asarray(state["class_specific.bias"], dtype=np.float64),
    )
    product = ca * cs
    attention_logits = (product * mask).sum(axis=(2, 3)) / np.maximum(
        mask.sum(axis=(2, 3)), 1.0
    )
    attention = np.maximum(product, 0.0).max(axis=1, keepdims=True) * mask
    peak = attention.reshape(attention.shape[0], -1).max(axis=1)[:, None, None, None]
    attention = attention / (peak + ATTENTION_EPSILON)
    prototypes = np.asarray(state["prototypes"], dtype=np.float64)[:, :, 0, 0]
    local = projected.transpose(0, 2, 3, 1)
    distances = np.square(local[:, None] - prototypes[None, :, None, None]).sum(axis=-1)
    similarities = np.log((distances + 1.0) / (distances + PROTOTYPE_EPSILON))
    evidence = similarities * attention
    evidence = np.where(mask > 0, evidence, -np.inf)
    prototype_evidence = evidence.reshape(evidence.shape[0], evidence.shape[1], -1).max(axis=2)
    identity = np.eye(NUM_CLASSES, dtype=np.float64).repeat(PROTOTYPES_PER_CLASS, axis=0)
    classifier = identity.T - 0.5 * (1.0 - identity.T)
    prototype_logits = prototype_evidence @ classifier.T
    target_identity = identity[:, np.asarray(targets, dtype=np.int64)].T[:, :, None, None]
    nearest_own = np.where(target_identity > 0, distances, np.inf).min(axis=1, keepdims=True)
    nearest_wrong = np.where(target_identity == 0, distances, np.inf).min(axis=1, keepdims=True)
    if cross_sample:
        weight = attention.sum(axis=0, keepdims=True) / np.maximum(
            mask.sum(axis=0, keepdims=True), 1.0
        )
        weight = np.broadcast_to(weight, attention.shape) * mask
    else:
        weight = attention * mask
    denominator = max(float(weight.sum()), ATTENTION_EPSILON)
    cluster = float((weight * nearest_own).sum() / denominator)
    separation_distance = float((weight * nearest_wrong).sum() / denominator)
    regularization = CLUSTER_WEIGHT * cluster
    if include_separation:
        regularization -= SEPARATION_WEIGHT * separation_distance
    return {
        "projected": projected,
        "attention_logits": attention_logits,
        "attention": attention,
        "distances": distances,
        "similarities": similarities,
        "prototype_evidence": prototype_evidence,
        "prototype_logits": prototype_logits,
        "cluster": cluster,
        "separation_distance": separation_distance,
        "regularization": regularization,
        "weight": weight,
    }


def _state_numpy(head: nn.Module) -> Dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in head.state_dict().items()
        if name
        in {
            "projector.0.weight",
            "projector.0.bias",
            "projector.2.weight",
            "projector.2.bias",
            "class_agnostic.weight",
            "class_agnostic.bias",
            "class_specific.weight",
            "class_specific.bias",
            "prototypes",
        }
    }


def _state_mapping_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dephase_gather_indices(fold: int, *, device: Optional[torch.device] = None) -> Tensor:
    indices = np.empty((FEATURE_CHANNELS, FEATURE_SIZE * FEATURE_SIZE), dtype=np.int64)
    rows = np.arange(FEATURE_SIZE, dtype=np.int64)[:, None]
    columns = np.arange(FEATURE_SIZE, dtype=np.int64)[None, :]
    for channel in range(FEATURE_CHANNELS):
        digest = hashlib.sha256(
            f"maxsep-dephase:{SEED}:{int(fold)}:{channel}".encode("ascii")
        ).digest()
        dy = int.from_bytes(digest[:8], "little") % FEATURE_SIZE
        dx = int.from_bytes(digest[8:16], "little") % FEATURE_SIZE
        source_y = (rows - dy) % FEATURE_SIZE
        source_x = (columns - dx) % FEATURE_SIZE
        indices[channel] = (source_y * FEATURE_SIZE + source_x).reshape(-1)
    return torch.from_numpy(indices).to(device=device)


def channel_dephase(features: Tensor, indices: Tensor) -> Tensor:
    if tuple(indices.shape) != (FEATURE_CHANNELS, FEATURE_SIZE * FEATURE_SIZE):
        raise ValueError("Dephase index shape differs from lock")
    flat = features.flatten(2)
    gather = indices[None].expand(features.shape[0], -1, -1)
    return torch.gather(flat, dim=2, index=gather).reshape_as(features)


def build_matched_heads(*, fold: int) -> Dict[str, AttentionMaxSepPrototypeHead]:
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(SEED + 1009 * int(fold))
        prototype = AttentionMaxSepPrototypeHead()
    rng_after = torch.get_rng_state().clone()
    if not torch.equal(rng_before, rng_after):
        raise RuntimeError("Matched head construction changed global RNG")
    heads = {role: copy.deepcopy(prototype) for role in TRAINED_ROLES}
    hashes = {_model_state_sha256(head) for head in heads.values()}
    if len(hashes) != 1:
        raise RuntimeError("Matched roles did not start byte-identically")
    if prototype.trainable_parameter_count > 25_000:
        raise ValueError("Candidate parameter count exceeds prospective lock")
    return heads


def _log_keeper(probabilities: np.ndarray) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != NUM_CLASSES:
        raise ValueError("Keeper probabilities must be [N,5]")
    return np.log(np.clip(value, 1e-7, 1.0))


def _rival_margin(values: np.ndarray, *, distance: bool = False) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    rivals = np.delete(matrix, FOCUS_CLASS, axis=1)
    if distance:
        return rivals.min(axis=1) - matrix[:, FOCUS_CLASS]
    return matrix[:, FOCUS_CLASS] - rivals.max(axis=1)


def learned_descriptor(
    outputs: Mapping[str, np.ndarray],
    valid_masks: np.ndarray,
    bbox_masks: np.ndarray,
    keeper_probabilities: np.ndarray,
    *,
    include_keeper: bool,
) -> np.ndarray:
    attention = np.asarray(outputs["attention"], dtype=np.float64)
    if attention.ndim == 4:
        attention = attention[:, 0]
    valid = np.asarray(valid_masks, dtype=np.bool_)
    bbox = np.asarray(bbox_masks, dtype=np.bool_)
    if attention.shape != valid.shape or bbox.shape != valid.shape:
        raise ValueError("Attention descriptor masks do not align")
    valid_weight = valid.astype(np.float64)
    attention = np.clip(attention, 0.0, None) * valid_weight
    valid_count = valid_weight.sum(axis=(1, 2)).clip(min=1.0)
    attention_total = attention.sum(axis=(1, 2)).clip(min=1e-12)
    probability = attention / attention_total[:, None, None]
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=(1, 2))
    entropy /= np.log(np.maximum(valid_count, 2.0))
    stats = np.stack(
        (
            attention_total / valid_count,
            entropy,
            attention.reshape(attention.shape[0], -1).max(axis=1),
            (attention * bbox).sum(axis=(1, 2)) / attention_total,
            (attention * (~bbox)).sum(axis=(1, 2)) / attention_total,
        ),
        axis=1,
    )
    attention_logits = np.asarray(outputs["attention_logits"], dtype=np.float64)
    prototype_logits = np.asarray(outputs["prototype_logits"], dtype=np.float64)
    class_max = np.asarray(outputs["class_evidence_max"], dtype=np.float64)
    class_mean = np.asarray(outputs["class_evidence_mean"], dtype=np.float64)
    class_distance = np.asarray(outputs["class_min_distance"], dtype=np.float64)
    margins = np.stack(
        (
            _rival_margin(attention_logits),
            _rival_margin(prototype_logits),
            _rival_margin(class_distance, distance=True),
        ),
        axis=1,
    )
    columns = (
        attention_logits,
        prototype_logits,
        class_max,
        class_mean,
        class_distance,
        stats,
        margins,
    )
    descriptor = np.concatenate(columns, axis=1)
    if include_keeper:
        descriptor = np.concatenate((descriptor, _log_keeper(keeper_probabilities)), axis=1)
    expected = 38 if include_keeper else 33
    if descriptor.shape != (attention.shape[0], expected):
        raise RuntimeError(f"Learned descriptor shape differs: {descriptor.shape}")
    if not np.isfinite(descriptor).all():
        raise ValueError("Learned descriptor is non-finite")
    return descriptor


def bbox_geometry_descriptor(valid_masks: np.ndarray, bbox_masks: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_masks, dtype=np.bool_)
    bbox = np.asarray(bbox_masks, dtype=np.bool_)
    if valid.shape != bbox.shape or valid.ndim != 3:
        raise ValueError("BBox descriptor masks must be aligned [N,H,W]")
    rows: List[List[float]] = []
    for valid_row, bbox_row in zip(valid, bbox):
        valid_points = np.argwhere(valid_row)
        box_points = np.argwhere(bbox_row)
        if valid_points.size == 0 or box_points.size == 0:
            raise ValueError("BBox descriptor received empty support")
        vy0, vx0 = valid_points.min(axis=0)
        vy1, vx1 = valid_points.max(axis=0) + 1
        by0, bx0 = box_points.min(axis=0)
        by1, bx1 = box_points.max(axis=0) + 1
        vw, vh = max(vx1 - vx0, 1), max(vy1 - vy0, 1)
        bw, bh = max(bx1 - bx0, 1), max(by1 - by0, 1)
        rows.append(
            [
                float(valid_row.mean()),
                float(bbox_row.mean()),
                float(bbox_row.sum() / max(valid_row.sum(), 1)),
                float(bw / bh),
                float(((bx0 + bx1) * 0.5 - vx0) / vw),
                float(((by0 + by1) * 0.5 - vy0) / vh),
                float(bw / vw),
                float(bh / vh),
                float((bx0 - vx0) / vw),
                float((by0 - vy0) / vh),
            ]
        )
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (valid.shape[0], 10) or not np.isfinite(result).all():
        raise RuntimeError("BBox geometry descriptor differs from lock")
    return result


def gap_descriptor(features: np.ndarray, valid_masks: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    valid = np.asarray(valid_masks, dtype=np.float64)
    denominator = valid.sum(axis=(1, 2)).clip(min=1.0)
    mean = np.einsum("nchw,nhw->nc", value, valid) / denominator[:, None]
    variance = np.einsum(
        "nchw,nhw->nc", (value - mean[:, :, None, None]) ** 2, valid
    ) / denominator[:, None]
    result = np.concatenate((mean, np.sqrt(np.maximum(variance, 0.0))), axis=1)
    if result.shape != (value.shape[0], 2 * FEATURE_CHANNELS):
        raise RuntimeError("GAP descriptor shape differs")
    return result


def build_role_descriptors(
    learned_outputs: Mapping[str, Mapping[str, np.ndarray]],
    features: np.ndarray,
    valid_masks: np.ndarray,
    bbox_masks: np.ndarray,
    keeper_probabilities: np.ndarray,
) -> Dict[str, np.ndarray]:
    descriptors = {
        role: learned_descriptor(
            learned_outputs[role],
            valid_masks,
            bbox_masks,
            keeper_probabilities,
            include_keeper=True,
        )
        for role in LEARNED_DESCRIPTOR_ROLES
    }
    geometry = bbox_geometry_descriptor(valid_masks, bbox_masks)
    log_keeper = _log_keeper(keeper_probabilities)
    descriptors[KEEPER_ROLE] = log_keeper
    descriptors[BBOX_KEEPER_ROLE] = np.concatenate((geometry, log_keeper), axis=1)
    descriptors[BBOX_ONLY_ROLE] = geometry
    descriptors[GAP_KEEPER_ROLE] = np.concatenate(
        (gap_descriptor(features, valid_masks), log_keeper), axis=1
    )
    descriptors[WITHOUT_KEEPER_ROLE] = learned_descriptor(
        learned_outputs[CANDIDATE_ROLE],
        valid_masks,
        bbox_masks,
        keeper_probabilities,
        include_keeper=False,
    )
    if set(descriptors) != set(ROLE_NAMES):
        raise RuntimeError("Descriptor roles differ from prospective lock")
    return descriptors


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(values[binary == 1])
    if positives.size == 0:
        raise ValueError("Readout fit fold lacks positives")
    allowed_breaks = int(math.floor((1.0 - MIN_FIT_POSITIVE_RETENTION) * positives.size))
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    if float((positives >= threshold).mean()) + 1e-12 < MIN_FIT_POSITIVE_RETENTION:
        raise RuntimeError("Readout threshold violates positive retention")
    return threshold


def _positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_, dtype=np.int64)
    position = np.flatnonzero(classes == 1)
    if position.size != 1:
        raise ValueError("Binary readout lacks positive class")
    return np.asarray(model.predict_proba(features)[:, int(position[0])], dtype=np.float64)


def fit_oof_readouts(
    descriptors: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    source_stems: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    if set(descriptors) != set(ROLE_NAMES):
        raise ValueError("Readout descriptor roles differ from lock")
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    sources = np.asarray([str(value) for value in source_stems], dtype=str)
    rows = binary.size
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    states: Dict[str, object] = {role: {"folds": []} for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        values = np.asarray(descriptors[role], dtype=np.float64)
        if values.shape[0] != rows or not np.isfinite(values).all():
            raise ValueError(f"Readout descriptor {role} is invalid")
        for held_fold in FOLDS:
            held = fold_values == held_fold
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
                    "converged": not convergence
                    and int(np.max(model.n_iter_)) < READOUT_MAX_ITER,
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
            "keeper_tp_retention": float(accepted[selected & keeper_tp].mean()),
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


def _numpy_prototype_regularization_gradient(
    result: Mapping[str, np.ndarray | float],
    state: Mapping[str, np.ndarray],
    targets: np.ndarray,
    *,
    include_separation: bool,
) -> np.ndarray:
    projected = np.asarray(result["projected"], dtype=np.float64)
    distances = np.asarray(result["distances"], dtype=np.float64)
    weight = np.asarray(result["weight"], dtype=np.float64)
    prototypes = np.asarray(state["prototypes"], dtype=np.float64)[:, :, 0, 0]
    target_values = np.asarray(targets, dtype=np.int64)
    denominator = max(float(weight.sum()), ATTENTION_EPSILON)
    gradient = np.zeros_like(prototypes)
    for batch in range(projected.shape[0]):
        target = int(target_values[batch])
        own_indices = np.arange(
            target * PROTOTYPES_PER_CLASS,
            (target + 1) * PROTOTYPES_PER_CLASS,
            dtype=np.int64,
        )
        wrong_indices = np.asarray(
            [index for index in range(NUM_PROTOTYPES) if index not in own_indices],
            dtype=np.int64,
        )
        own_choice = own_indices[np.argmin(distances[batch, own_indices], axis=0)]
        wrong_choice = wrong_indices[np.argmin(distances[batch, wrong_indices], axis=0)]
        local = projected[batch].transpose(1, 2, 0)
        for y in range(FEATURE_SIZE):
            for x in range(FEATURE_SIZE):
                coefficient = float(weight[batch, 0, y, x]) / denominator
                if coefficient == 0.0:
                    continue
                own = int(own_choice[y, x])
                gradient[own] += (
                    CLUSTER_WEIGHT
                    * coefficient
                    * 2.0
                    * (prototypes[own] - local[y, x])
                )
                if include_separation:
                    wrong = int(wrong_choice[y, x])
                    gradient[wrong] += (
                        -SEPARATION_WEIGHT
                        * coefficient
                        * 2.0
                        * (prototypes[wrong] - local[y, x])
                    )
    return gradient


def _effective_rank(value: np.ndarray) -> float:
    singular = np.linalg.svd(
        np.asarray(value, dtype=np.float64), full_matrices=False, compute_uv=False
    )
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= 1e-30:
        return 0.0
    probability = energy / total
    positive = probability > 0.0
    return float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))


def _parameter_family(name: str) -> str:
    if name.startswith("projector.0"):
        return "projector_first"
    if name.startswith("projector.2"):
        return "projector_second"
    if name.startswith("class_agnostic"):
        return "class_agnostic"
    if name.startswith("class_specific"):
        return "class_specific"
    if name == "prototypes":
        return "prototypes"
    return name


def _gradient_summary(head: nn.Module) -> Dict[str, object]:
    families: Dict[str, List[Tensor]] = {}
    for name, parameter in head.named_parameters():
        if parameter.grad is None:
            continue
        families.setdefault(_parameter_family(name), []).append(parameter.grad.detach())
    summary = {}
    for family, values in families.items():
        flat = torch.cat([value.reshape(-1).float().cpu() for value in values])
        summary[family] = {
            "finite": bool(torch.isfinite(flat).all()),
            "nonzero": bool((flat.abs() > 0).any()),
            "maximum_absolute": float(flat.abs().max()),
            "l2": float(torch.linalg.vector_norm(flat)),
        }
    return summary


def _changed_parameter_families(
    initial: Mapping[str, Tensor], head: nn.Module
) -> Dict[str, object]:
    changes: Dict[str, float] = {}
    for name, value in head.state_dict().items():
        if name not in initial or name in {
            "prototype_class_identity",
            "classifier_weight",
        }:
            continue
        delta = float((value.detach().cpu() - initial[name]).abs().max())
        family = _parameter_family(name)
        changes[family] = max(changes.get(family, 0.0), delta)
    return {
        family: {"maximum_absolute_change": value, "changed": value > 0.0}
        for family, value in sorted(changes.items())
    }


def engineering_checks() -> Dict[str, object]:
    rng_before = _global_rng_snapshot()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(SEED + 17)
        head = AttentionMaxSepPrototypeHead().double()
    generator = np.random.default_rng(SEED + 29)
    features = generator.normal(
        0.0,
        0.35,
        size=(3, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    ).astype(np.float64)
    valid = np.ones((3, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64)
    valid[0, :2] = 0.0
    valid[1, :, :1] = 0.0
    targets = np.asarray([0, 1, 4], dtype=np.int64)
    tensor_features = torch.from_numpy(features).requires_grad_(True)
    tensor_valid = torch.from_numpy(valid)
    tensor_targets = torch.from_numpy(targets)
    outputs = head(tensor_features, tensor_valid)
    regularization = maxsep_regularization(
        outputs,
        tensor_targets,
        cross_sample=True,
        include_separation=True,
    )
    state = _state_numpy(head)
    oracle = maxsep_numpy(
        features,
        valid,
        state,
        targets,
        cross_sample=True,
        include_separation=True,
    )
    torch_values = {
        "projected": outputs["projected"].detach().numpy(),
        "attention_logits": outputs["attention_logits"].detach().numpy(),
        "attention": outputs["attention"].detach().numpy(),
        "distances": outputs["distances"].detach().numpy(),
        "similarities": outputs["similarities"].detach().numpy(),
        "prototype_evidence": outputs["prototype_evidence"].detach().numpy(),
        "prototype_logits": outputs["prototype_logits"].detach().numpy(),
    }
    equation_errors = {
        key: float(np.max(np.abs(value - np.asarray(oracle[key]))))
        for key, value in torch_values.items()
    }
    equation_errors.update(
        {
            "cluster": abs(float(regularization["cluster"]) - float(oracle["cluster"])),
            "separation_distance": abs(
                float(regularization["separation_distance"])
                - float(oracle["separation_distance"])
            ),
            "regularization": abs(
                float(regularization["regularization"])
                - float(oracle["regularization"])
            ),
        }
    )
    regularization["regularization"].backward()
    torch_gradient = head.prototypes.grad.detach().numpy()[:, :, 0, 0]
    numpy_gradient = _numpy_prototype_regularization_gradient(
        oracle, state, targets, include_separation=True
    )
    gradient_error = float(np.max(np.abs(torch_gradient - numpy_gradient)))

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(SEED + 31)
        movement_head = AttentionMaxSepPrototypeHead().float()
    initial = {
        name: value.detach().cpu().clone()
        for name, value in movement_head.state_dict().items()
    }
    optimizer = torch.optim.Adam(movement_head.parameters(), lr=3e-4)
    movement_loss, _ = maxsep_total_loss(
        movement_head,
        torch.from_numpy(features[:2]).float(),
        torch.from_numpy(valid[:2]).bool(),
        torch.from_numpy(targets[:2]),
        cross_sample=True,
        include_separation=True,
        include_regularization=True,
    )
    optimizer.zero_grad(set_to_none=True)
    movement_loss.backward()
    gradients = _gradient_summary(movement_head)
    optimizer.step()
    changes = _changed_parameter_families(initial, movement_head)

    indices = dephase_gather_indices(0)
    original = torch.from_numpy(features[:2]).float()
    dephased = channel_dephase(original, indices)
    marginal_error = float(
        (
            original.flatten(2).sort(dim=2).values
            - dephased.flatten(2).sort(dim=2).values
        )
        .abs()
        .max()
    )
    aligned_output = movement_head(original, torch.from_numpy(valid[:2]).bool())
    rolled_output = movement_head(
        original,
        torch.from_numpy(valid[:2]).bool(),
        attention_roll=ATTENTION_ROLL,
    )
    cycled_output = movement_head(
        original,
        torch.from_numpy(valid[:2]).bool(),
        class_cycle=True,
    )
    matched = build_matched_heads(fold=0)
    matched_hashes = {_model_state_sha256(value) for value in matched.values()}
    rng_after = _global_rng_snapshot()
    expected_families = {
        "projector_first",
        "projector_second",
        "class_agnostic",
        "class_specific",
        "prototypes",
    }
    checks = {
        "all_equation_errors_le_1e_10": max(equation_errors.values()) <= 1e-10,
        "prototype_gradient_error_le_1e_10": gradient_error <= 1e-10,
        "all_gradient_families_live": set(gradients) == expected_families
        and all(item["finite"] and item["nonzero"] for item in gradients.values()),
        "all_parameter_families_changed": set(changes) == expected_families
        and all(item["changed"] for item in changes.values()),
        "dephase_channel_marginals_exact": marginal_error == 0.0,
        "attention_roll_active": not torch.equal(
            aligned_output["prototype_logits"], rolled_output["prototype_logits"]
        ),
        "class_cycle_active": not torch.equal(
            aligned_output["prototype_logits"], cycled_output["prototype_logits"]
        ),
        "matched_roles_byte_identical": len(matched_hashes) == 1,
        "parameter_count_le_25000": movement_head.trainable_parameter_count <= 25_000,
        "fixed_classifier_exact": bool(
            torch.equal(
                movement_head.classifier_weight,
                _classifier_weights(movement_head.prototype_class_identity),
            )
        ),
        "padding_attention_zero": bool(
            (outputs["attention"] * (1.0 - outputs["valid"])).abs().max() == 0
        ),
        "global_rng_restored": _global_rng_equal(rng_before, rng_after),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "equation_errors": equation_errors,
        "prototype_gradient_max_abs_error": gradient_error,
        "gradient_families": gradients,
        "parameter_changes": changes,
        "dephase_marginal_max_abs_error": marginal_error,
        "trainable_parameters": movement_head.trainable_parameter_count,
        "matched_state_sha256": next(iter(matched_hashes)),
    }


class DenseBlock2Capture:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.value: Optional[Tensor] = None
        self.call_count = 0
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.handle = None

    def _hook(self, _module: nn.Module, _inputs: Tuple[Tensor, ...], output: object) -> None:
        tokens = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(tokens) or tokens.ndim != 3:
            raise TypeError("Block-2 hook did not receive token output")
        prefix = int(getattr(self.model, "num_prefix_tokens", -1))
        patches = tokens[:, prefix:]
        if prefix != PREFIX_TOKENS or tuple(patches.shape[1:]) != (
            FEATURE_SIZE * FEATURE_SIZE,
            FEATURE_CHANNELS,
        ):
            raise ValueError(
                f"Block-2 contract differs: prefix={prefix}, patches={tuple(patches.shape)}"
            )
        self.output_shape = tuple(int(value) for value in tokens.shape)
        self.value = patches.transpose(1, 2).reshape(
            patches.shape[0], FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE
        )
        self.call_count += 1

    def take(self) -> Tensor:
        if self.value is None:
            raise RuntimeError("Block-2 hook has no captured value")
        value = self.value
        self.value = None
        return value

    def __enter__(self) -> "DenseBlock2Capture":
        blocks = getattr(self.model, "blocks", None)
        if blocks is None or len(blocks) <= BLOCK_INDEX:
            raise TypeError("Keeper does not expose locked transformer blocks")
        self.handle = blocks[BLOCK_INDEX].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _captured_keeper_forward(
    model: nn.Module,
    images: Tensor,
    image_mask: Tensor,
    bbox: Tensor,
) -> Tuple[Tensor, Tensor, Tuple[int, ...]]:
    with DenseBlock2Capture(model) as capture:
        features = model.forward_features(  # type: ignore[attr-defined]
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=bbox,
        )
        features["bbox"] = bbox[:, :4]
        logits = classification_logits_from_features(model, features).float()
        block2 = capture.take()
        if capture.call_count != 1 or capture.output_shape is None:
            raise RuntimeError("Block-2 hook call count differs")
        shape = capture.output_shape
    return logits, block2, shape


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


def _feature_geometry(
    image_mask: Tensor,
    crop_bbox: Tensor,
) -> Tuple[Tensor, Tensor]:
    mask = image_mask
    if mask.ndim == 3:
        mask = mask[:, None]
    valid = F.interpolate(mask.float(), size=(FEATURE_SIZE, FEATURE_SIZE), mode="area")[:, 0] >= 0.5
    from trkh.tools.audit_boxinst_foreground_mask_a0 import normalized_bbox_masks

    bbox = normalized_bbox_masks(crop_bbox[:, :4], size=FEATURE_SIZE) & valid
    if not bool(valid.flatten(1).any(dim=1).all()):
        raise ValueError("Feature-valid support is empty")
    if not bool(bbox.flatten(1).any(dim=1).all()):
        raise ValueError("Feature bbox support is empty")
    if bool((bbox & ~valid).any()):
        raise ValueError("Feature bbox leaves valid support")
    return valid, bbox


def extract_condition(
    *,
    model: nn.Module,
    loader: DataLoader,
    loader_summary: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    device: torch.device,
    condition: str,
    feature_cache_path: Path,
    benchmark_ordinary: bool,
) -> Dict[str, object]:
    count = len(rows)
    cache_path = Path(feature_cache_path).resolve()
    if cache_path.exists():
        raise FileExistsError(f"Temporary feature cache already exists: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    )
    probabilities = np.empty((count, NUM_CLASSES), dtype=np.float32)
    targets = np.empty(count, dtype=np.int64)
    sample_indices = np.empty(count, dtype=np.int64)
    valid_masks = np.empty((count, FEATURE_SIZE, FEATURE_SIZE), dtype=np.bool_)
    bbox_masks = np.empty_like(valid_masks)
    model_boxes = np.empty((count, 4), dtype=np.float32)
    crop_boxes = np.empty((count, 4), dtype=np.float32)
    position = 0
    capture_error = 0.0
    cidt_error = 0.0
    cidt_argmax_exact = True
    ordinary_seconds = 0.0
    captured_seconds = 0.0
    output_shapes: set[Tuple[int, ...]] = set()
    model.eval()
    state_before = _model_state_sha256(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, batch_targets, metadata in loader:
            if not isinstance(metadata, Mapping):
                raise ValueError("Feature extraction requires tensor metadata")
            batch = int(batch_targets.numel())
            stop = position + batch
            expected = rows[position:stop]
            observed_indices = metadata["sample_index"].detach().cpu().long().tolist()
            observed_targets = batch_targets.detach().cpu().long().tolist()
            if observed_indices != [row.sample_index for row in expected]:
                raise ValueError("Feature loader changed locked sample order")
            if observed_targets != [row.target for row in expected]:
                raise ValueError("Feature loader changed locked targets")
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            bbox = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _metadata_tensor(
                metadata, "crop_bbox", device=device, dtype=torch.float32
            )
            if image_mask is None or bbox is None or crop_bbox is None:
                raise ValueError("Extraction requires image_mask/bbox/crop_bbox")
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
            logits, block2, output_shape = _captured_keeper_forward(
                model, images_device, image_mask, bbox
            )
            current_probabilities = logits.softmax(dim=1)
            torch.cuda.synchronize(device)
            captured_seconds += time.perf_counter() - captured_started
            output_shapes.add(tuple(output_shape[1:]))
            if ordinary is not None:
                capture_error = max(
                    capture_error,
                    float((ordinary - current_probabilities).abs().max()),
                )
                if not torch.equal(ordinary.argmax(dim=1), current_probabilities.argmax(dim=1)):
                    raise RuntimeError("Block-2 hook changed keeper argmax")
            valid, box_mask = _feature_geometry(image_mask, crop_bbox)
            current_numpy = current_probabilities.detach().cpu().numpy().astype(np.float32)
            cache[position:stop] = block2.detach().cpu().numpy().astype(np.float16)
            probabilities[position:stop] = current_numpy
            targets[position:stop] = np.asarray(observed_targets, dtype=np.int64)
            sample_indices[position:stop] = np.asarray(observed_indices, dtype=np.int64)
            valid_masks[position:stop] = valid.detach().cpu().numpy()
            bbox_masks[position:stop] = box_mask.detach().cpu().numpy()
            model_boxes[position:stop] = bbox.detach().cpu().numpy().astype(np.float32)
            crop_boxes[position:stop] = crop_bbox.detach().cpu().numpy().astype(np.float32)
            historical = np.asarray(
                [row.keeper_probabilities for row in expected], dtype=np.float32
            )
            cidt_error = max(cidt_error, float(np.max(np.abs(current_numpy - historical))))
            cidt_argmax_exact = cidt_argmax_exact and bool(
                np.array_equal(current_numpy.argmax(axis=1), historical.argmax(axis=1))
            )
            position = stop
    cache.flush()
    elapsed = time.perf_counter() - started
    state_after = _model_state_sha256(model)
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    checks = {
        "rows_exact": position == count,
        "sample_order_exact": sample_indices.tolist() == [row.sample_index for row in rows],
        "targets_exact": targets.tolist() == [row.target for row in rows],
        "hook_probability_error_le_1e_6": capture_error <= MAX_CAPTURE_PROBABILITY_ERROR,
        "cidt_argmax_exact_or_shifted_condition": cidt_argmax_exact
        if condition in {"clean", "engineering"}
        else True,
        "cidt_probability_telemetry_finite": math.isfinite(cidt_error),
        "keeper_state_exact": state_before == state_after,
        "feature_shape_exact": cache.shape
        == (count, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        "block_output_shape_single": len(output_shapes) == 1,
        "all_geometry_nonempty": bool(
            valid_masks.reshape(count, -1).any(axis=1).all()
            and bbox_masks.reshape(count, -1).any(axis=1).all()
        ),
        "bbox_inside_valid": not bool((bbox_masks & ~valid_masks).any()),
        "features_finite": bool(np.isfinite(cache).all()),
    }
    if not all(checks.values()):
        cache.flush()
        mmap = getattr(cache, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del cache
        gc.collect()
        if cache_path.exists():
            cache_path.unlink()
        raise RuntimeError(
            "Feature extraction structural checks failed: "
            f"checks={checks}, capture_error={capture_error:.9g}, "
            f"cidt_error={cidt_error:.9g}, cache_absent={not cache_path.exists()}"
        )
    return {
        "features": cache,
        "probabilities": probabilities,
        "targets": targets,
        "sample_indices": sample_indices,
        "valid_masks": valid_masks,
        "bbox_masks": bbox_masks,
        "model_boxes": model_boxes,
        "crop_boxes": crop_boxes,
        "checks": checks,
        "capture_probability_max_abs_error": capture_error,
        "cidt_probability_max_abs_error": cidt_error,
        "cidt_probability_reference_tolerance": MAX_CIDT_PROBABILITY_ERROR,
        "cidt_probability_reference_tolerance_passed": cidt_error
        <= MAX_CIDT_PROBABILITY_ERROR,
        "cidt_argmax_exact": cidt_argmax_exact,
        "block_output_shapes": [list(value) for value in sorted(output_shapes)],
        "cache": {
            "path": str(cache_path),
            "bytes": int(cache_path.stat().st_size),
            "sha256": _array_sha256(cache),
        },
        "runtime": {
            "condition": condition,
            "seconds": elapsed,
            "ordinary_forward_seconds": ordinary_seconds,
            "captured_forward_seconds": captured_seconds,
            "throughput_images_per_second": count / max(elapsed, 1e-12),
            "peak_cuda_bytes": peak_bytes,
            "loader": dict(loader_summary),
        },
    }


OUTPUT_SPECS: Mapping[str, Tuple[int, ...]] = {
    "attention": (FEATURE_SIZE, FEATURE_SIZE),
    "attention_logits": (NUM_CLASSES,),
    "prototype_logits": (NUM_CLASSES,),
    "class_evidence_max": (NUM_CLASSES,),
    "class_evidence_mean": (NUM_CLASSES,),
    "class_min_distance": (NUM_CLASSES,),
    "prototype_evidence": (NUM_PROTOTYPES,),
    "projected_mean": (PROJECTION_DIM,),
}


def _empty_output_bank(rows: int) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        role: {
            key: np.full((rows, *shape), np.nan, dtype=np.float32)
            for key, shape in OUTPUT_SPECS.items()
        }
        for role in LEARNED_DESCRIPTOR_ROLES
    }


def _write_output_batch(
    bank: MutableMapping[str, np.ndarray],
    positions: np.ndarray,
    outputs: Mapping[str, Tensor],
) -> None:
    valid = outputs["valid"]
    projected = outputs["projected"]
    denominator = valid.sum(dim=(2, 3)).clamp_min(1.0)
    projected_mean = (projected * valid).sum(dim=(2, 3)) / denominator
    tensors = {
        "attention": outputs.get("evidence_attention", outputs["attention"])[:, 0],
        "attention_logits": outputs["attention_logits"],
        "prototype_logits": outputs["prototype_logits"],
        "class_evidence_max": outputs["class_evidence_max"],
        "class_evidence_mean": outputs["class_evidence_mean"],
        "class_min_distance": outputs["class_min_distance"],
        "prototype_evidence": outputs["prototype_evidence"],
        "projected_mean": projected_mean,
    }
    for key, tensor in tensors.items():
        bank[key][positions] = tensor.detach().cpu().float().numpy()


def _epoch_orders(fit_positions: np.ndarray, *, fold: int) -> Tuple[List[np.ndarray], str]:
    positions = np.asarray(fit_positions, dtype=np.int64)
    orders: List[np.ndarray] = []
    digest = hashlib.sha256()
    for epoch in range(EPOCHS):
        generator = np.random.default_rng(SEED + 100_003 * int(fold) + 997 * epoch)
        order = positions[generator.permutation(positions.size)]
        orders.append(order)
        digest.update(order.astype("<i8", copy=False).tobytes())
    return orders, digest.hexdigest()


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, epoch: int) -> float:
    if epoch < WARM_EPOCHS:
        learning_rate = WARM_LEARNING_RATE
    elif epoch < JOINT_DECAY_EPOCH:
        learning_rate = JOINT_LEARNING_RATE
    else:
        learning_rate = JOINT_DECAYED_LEARNING_RATE
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _role_loss_settings(role: str) -> Tuple[bool, bool]:
    if role == SELF_ROLE:
        return False, True
    if role == CLUSTER_ROLE:
        return True, False
    if role in {CANDIDATE_ROLE, DEPHASED_ROLE}:
        return True, True
    raise ValueError(f"Unknown trained role: {role}")


def train_oof_heads(
    *,
    features: np.ndarray,
    valid_masks: np.ndarray,
    targets: np.ndarray,
    sample_indices: np.ndarray,
    row_folds: np.ndarray,
    source_stems: Sequence[str],
    device: torch.device,
) -> Dict[str, object]:
    rows = int(targets.size)
    if not np.array_equal(
        np.asarray(sample_indices, dtype=np.int64), np.arange(rows, dtype=np.int64)
    ):
        raise ValueError("Training sample indices differ from locked 0..N-1 order")
    if tuple(features.shape) != (
        rows,
        FEATURE_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError("Training feature cache shape differs")
    bank = _empty_output_bank(rows)
    state_payload: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
    fold_records: Dict[int, Dict[str, object]] = {}
    training_started = time.perf_counter()
    for fold in FOLDS:
        fit_positions = np.flatnonzero(row_folds != fold)
        held_positions = np.flatnonzero(row_folds == fold)
        fit_sources = {source_stems[position] for position in fit_positions}
        held_sources = {source_stems[position] for position in held_positions}
        overlap = sorted(fit_sources.intersection(held_sources))
        if overlap:
            raise ValueError(f"Head fold {fold} has source overlap")
        heads = {role: head.to(device) for role, head in build_matched_heads(fold=fold).items()}
        initial_states = {
            role: {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            for role, head in heads.items()
        }
        optimizers = {
            role: torch.optim.Adam(head.parameters(), lr=WARM_LEARNING_RATE)
            for role, head in heads.items()
        }
        orders, occurrence_sha = _epoch_orders(fit_positions, fold=fold)
        dephase_indices = dephase_gather_indices(fold, device=device)
        epoch_records: List[Dict[str, object]] = []
        first_joint_gradients: Dict[str, object] = {}
        for epoch, order in enumerate(orders):
            learning_rate = _set_optimizer_lr(optimizers[CANDIDATE_ROLE], epoch)
            for role in TRAINED_ROLES[1:]:
                observed = _set_optimizer_lr(optimizers[role], epoch)
                if observed != learning_rate:
                    raise RuntimeError("Matched role learning rates differ")
            aggregates = {
                role: {
                    "loss": 0.0,
                    "prototype_ce": 0.0,
                    "attention_ce": 0.0,
                    "cluster": 0.0,
                    "separation_distance": 0.0,
                    "batches": 0,
                }
                for role in TRAINED_ROLES
            }
            for start in range(0, order.size, BATCH_SIZE):
                positions = order[start : start + BATCH_SIZE]
                batch_features = torch.from_numpy(
                    np.asarray(features[positions], dtype=np.float32)
                ).to(device=device, non_blocking=True)
                batch_valid = torch.from_numpy(valid_masks[positions]).to(
                    device=device, non_blocking=True
                )
                batch_targets = torch.from_numpy(targets[positions]).to(
                    device=device, non_blocking=True
                )
                dephased = channel_dephase(batch_features, dephase_indices)
                for role in TRAINED_ROLES:
                    cross_sample, include_separation = _role_loss_settings(role)
                    role_features = dephased if role == DEPHASED_ROLE else batch_features
                    head = heads[role]
                    optimizer = optimizers[role]
                    optimizer.zero_grad(set_to_none=True)
                    total, outputs = maxsep_total_loss(
                        head,
                        role_features,
                        batch_valid,
                        batch_targets,
                        cross_sample=cross_sample,
                        include_separation=include_separation,
                        include_regularization=epoch >= WARM_EPOCHS,
                    )
                    total.backward()
                    if epoch == WARM_EPOCHS and role not in first_joint_gradients:
                        first_joint_gradients[role] = _gradient_summary(head)
                    optimizer.step()
                    aggregate = aggregates[role]
                    aggregate["loss"] += float(total.detach())
                    for key in (
                        "prototype_ce",
                        "attention_ce",
                        "cluster",
                        "separation_distance",
                    ):
                        aggregate[key] += float(outputs[key].detach())
                    aggregate["batches"] += 1
            epoch_records.append(
                {
                    "epoch": epoch + 1,
                    "learning_rate": learning_rate,
                    "regularization_active": epoch >= WARM_EPOCHS,
                    "roles": {
                        role: {
                            key: (
                                float(value / max(aggregates[role]["batches"], 1))
                                if key != "batches"
                                else int(value)
                            )
                            for key, value in aggregates[role].items()
                        }
                        for role in TRAINED_ROLES
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "attention_maxsep_epoch",
                        "fold": fold,
                        "epoch": epoch + 1,
                        "learning_rate": learning_rate,
                        "candidate_loss": epoch_records[-1]["roles"][
                            CANDIDATE_ROLE
                        ]["loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        for head in heads.values():
            head.eval()
        with torch.inference_mode():
            for start in range(0, held_positions.size, BATCH_SIZE):
                positions = held_positions[start : start + BATCH_SIZE]
                batch_features = torch.from_numpy(
                    np.asarray(features[positions], dtype=np.float32)
                ).to(device=device, non_blocking=True)
                batch_valid = torch.from_numpy(valid_masks[positions]).to(
                    device=device, non_blocking=True
                )
                dephased = channel_dephase(batch_features, dephase_indices)
                for role in TRAINED_ROLES:
                    role_features = dephased if role == DEPHASED_ROLE else batch_features
                    outputs = heads[role](role_features, batch_valid)
                    _write_output_batch(bank[role], positions, outputs)
                rolled = heads[CANDIDATE_ROLE](
                    batch_features, batch_valid, attention_roll=ATTENTION_ROLL
                )
                cycled = heads[CANDIDATE_ROLE](
                    batch_features, batch_valid, class_cycle=True
                )
                _write_output_batch(bank[ROLLED_ROLE], positions, rolled)
                _write_output_batch(bank[CYCLED_ROLE], positions, cycled)
        family_changes = {
            role: _changed_parameter_families(initial_states[role], heads[role])
            for role in TRAINED_ROLES
        }
        optimizer_checks = {}
        for role in TRAINED_ROLES:
            state_tensors = [
                value
                for parameter_state in optimizers[role].state.values()
                for value in parameter_state.values()
                if torch.is_tensor(value)
            ]
            parameter_tensors = list(heads[role].parameters())
            optimizer_checks[role] = {
                "updates": EPOCHS * int(math.ceil(fit_positions.size / BATCH_SIZE)),
                "parameter_dtypes": sorted(
                    {str(value.dtype) for value in parameter_tensors}
                ),
                "optimizer_state_dtypes": sorted(
                    {str(value.dtype) for value in state_tensors}
                ),
                "parameters_finite": all(
                    bool(torch.isfinite(value).all()) for value in parameter_tensors
                ),
                "optimizer_state_finite": bool(state_tensors)
                and all(bool(torch.isfinite(value).all()) for value in state_tensors),
                "fixed_classifier_exact": bool(
                    torch.equal(
                        heads[role].classifier_weight,
                        _classifier_weights(
                            heads[role].prototype_class_identity
                        ).to(
                            device=heads[role].classifier_weight.device,
                            dtype=heads[role].classifier_weight.dtype,
                        ),
                    )
                ),
            }
        state_payload[str(fold)] = {
            role: {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            for role, head in heads.items()
        }
        fold_records[fold] = {
            "fit_rows": int(fit_positions.size),
            "held_rows": int(held_positions.size),
            "fit_sources": len(fit_sources),
            "held_sources": len(held_sources),
            "source_overlap": overlap,
            "occurrence_sha256": occurrence_sha,
            "epochs": epoch_records,
            "first_joint_gradients": first_joint_gradients,
            "parameter_changes": family_changes,
            "optimizer_checks": optimizer_checks,
            "initial_state_sha256": {
                role: _state_mapping_sha256(initial_states[role])
                for role in heads
            },
            "final_state_sha256": {
                role: _model_state_sha256(head) for role, head in heads.items()
            },
        }
        del heads, optimizers
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if any(
        not np.isfinite(value).all()
        for role in bank.values()
        for value in role.values()
    ):
        raise RuntimeError("OOF head output bank contains non-finite values")
    occurrence_hashes = {
        fold_records[fold]["occurrence_sha256"] for fold in FOLDS
    }
    return {
        "outputs": bank,
        "states": state_payload,
        "fold_records": fold_records,
        "training_seconds": time.perf_counter() - training_started,
        "fold_occurrence_hashes_unique": len(occurrence_hashes) == len(FOLDS),
        "shared_occurrence_order_across_roles": True,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(to_serializable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _write_manifest(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    path = root / "artifact_manifest.json"
    files = []
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and artifact != path:
            files.append(
                {
                    "path": artifact.relative_to(root).as_posix(),
                    "bytes": int(artifact.stat().st_size),
                    "sha256": _sha256(artifact),
                }
            )
    _write_json(
        path,
        {
            "schema": "trkh_attention_maxsep_prototype_a0_manifest_v1",
            "files": files,
        },
    )
    return path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    path = root / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(row["path"]): row for row in payload["files"]}
    observed = {
        artifact.relative_to(root).as_posix(): artifact
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact != path
    }
    if set(expected) != set(observed):
        raise ValueError("Attention-MaxSep artifact-manifest file set differs")
    for relative, artifact in observed.items():
        record = expected[relative]
        if (
            int(artifact.stat().st_size) != int(record["bytes"])
            or _sha256(artifact) != str(record["sha256"])
        ):
            raise ValueError(f"Attention-MaxSep artifact differs: {relative}")
    return {
        "passed": True,
        "files": len(expected),
        "manifest_sha256": _sha256(path),
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
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else float("inf")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _close_delete_feature_cache(extraction: MutableMapping[str, object]) -> Dict[str, object]:
    value = extraction.pop("features", None)
    if isinstance(value, np.memmap):
        value.flush()
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    record = extraction.get("cache")
    paths = []
    if isinstance(record, Mapping):
        paths.append(Path(str(record["path"])))
    gc.collect()
    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
        removed.append({"path": str(path), "absent": not path.exists()})
    return {
        "deleted": bool(removed) and all(item["absent"] for item in removed),
        "files": removed,
    }


def fixed_xai_positions(cohort: Sequence[CleanTrainRow]) -> List[int]:
    positions: List[int] = []
    for fold in FOLDS:
        predicates = (
            lambda row: row.target == FOCUS_CLASS
            and row.keeper_prediction == FOCUS_CLASS,
            lambda row: row.target == FOCUS_CLASS
            and row.keeper_prediction != FOCUS_CLASS,
            lambda row: row.target != FOCUS_CLASS,
        )
        for predicate in predicates:
            candidates = [
                (row.sample_index, position)
                for position, row in enumerate(cohort)
                if row.fold == fold and predicate(row)
            ]
            if not candidates:
                raise ValueError(f"XAI selection lacks fold {fold} category")
            positions.append(min(candidates)[1])
    if len(positions) != 15 or len(set(positions)) != 15:
        raise RuntimeError("Fixed XAI selection differs from 15 unique rows")
    return positions


def _load_head_state(
    state: Mapping[str, Tensor], *, device: torch.device, dtype: torch.dtype = torch.float32
) -> AttentionMaxSepPrototypeHead:
    head = AttentionMaxSepPrototypeHead()
    head.load_state_dict(state, strict=True)
    return head.to(device=device, dtype=dtype).eval()


def evaluate_oof_heads(
    *,
    features: np.ndarray,
    valid_masks: np.ndarray,
    folds: np.ndarray,
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    rows = int(len(folds))
    bank = _empty_output_bank(rows)
    with torch.inference_mode():
        for fold in FOLDS:
            positions = np.flatnonzero(np.asarray(folds, dtype=np.int64) == fold)
            heads = {
                role: _load_head_state(states[str(fold)][role], device=device)
                for role in TRAINED_ROLES
            }
            dephase_indices = dephase_gather_indices(fold, device=device)
            for start in range(0, positions.size, BATCH_SIZE):
                selected = positions[start : start + BATCH_SIZE]
                batch_features = torch.from_numpy(
                    np.asarray(features[selected], dtype=np.float32)
                ).to(device=device, non_blocking=True)
                batch_valid = torch.from_numpy(valid_masks[selected]).to(
                    device=device, non_blocking=True
                )
                dephased = channel_dephase(batch_features, dephase_indices)
                for role in TRAINED_ROLES:
                    role_features = dephased if role == DEPHASED_ROLE else batch_features
                    _write_output_batch(
                        bank[role], selected, heads[role](role_features, batch_valid)
                    )
                candidate = heads[CANDIDATE_ROLE]
                _write_output_batch(
                    bank[ROLLED_ROLE],
                    selected,
                    candidate(
                        batch_features,
                        batch_valid,
                        attention_roll=ATTENTION_ROLL,
                    ),
                )
                _write_output_batch(
                    bank[CYCLED_ROLE],
                    selected,
                    candidate(batch_features, batch_valid, class_cycle=True),
                )
            del heads
    if any(
        not np.isfinite(value).all()
        for output in bank.values()
        for value in output.values()
    ):
        raise RuntimeError("Reconstructed OOF outputs contain non-finite values")
    return bank


def _effective_rank_from_tokens(tokens: np.ndarray) -> float:
    value = np.asarray(tokens, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != PROJECTION_DIM:
        raise ValueError("Projected tokens must be [N,32]")
    centered = value - value.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(value.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-30:
        return 0.0
    probability = eigenvalues / total
    positive = probability > 0.0
    return float(
        np.exp(-np.sum(probability[positive] * np.log(probability[positive])))
    )


def prototype_diagnostics(
    *,
    features: np.ndarray,
    valid_masks: np.ndarray,
    bbox_masks: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    device: torch.device,
) -> Dict[str, object]:
    token_rows: List[np.ndarray] = []
    prototype_use = np.zeros(NUM_PROTOTYPES, dtype=np.int64)
    class1_margins = np.empty(len(targets), dtype=np.float64)
    nearest_correct = 0
    nearest_total = 0
    all_zero = 0
    low_peak = 0
    padding_leakage = 0.0
    cross_class_distances: List[float] = []
    distinct_prototypes = True
    with torch.inference_mode():
        for fold in FOLDS:
            positions = np.flatnonzero(np.asarray(folds, dtype=np.int64) == fold)
            head = _load_head_state(
                states[str(fold)][CANDIDATE_ROLE], device=device
            )
            prototypes = head.prototypes.detach().cpu().numpy()[:, :, 0, 0]
            pair_distance = np.sqrt(
                np.maximum(
                    np.square(prototypes[:, None] - prototypes[None]).sum(axis=2),
                    0.0,
                )
            )
            identity = np.arange(NUM_PROTOTYPES) // PROTOTYPES_PER_CLASS
            cross = identity[:, None] != identity[None]
            nearest_cross = np.where(cross, pair_distance, np.inf).min(axis=1)
            cross_class_distances.extend(nearest_cross.tolist())
            distinct_prototypes = distinct_prototypes and bool(
                np.min(pair_distance[np.triu_indices(NUM_PROTOTYPES, 1)]) > 0.0
            )
            for start in range(0, positions.size, BATCH_SIZE):
                selected = positions[start : start + BATCH_SIZE]
                batch_features = torch.from_numpy(
                    np.asarray(features[selected], dtype=np.float32)
                ).to(device=device, non_blocking=True)
                batch_valid = torch.from_numpy(valid_masks[selected]).to(
                    device=device, non_blocking=True
                )
                outputs = head(batch_features, batch_valid)
                projected = outputs["projected"].detach().cpu().numpy()
                valid = np.asarray(valid_masks[selected], dtype=np.bool_)
                for local, mask in zip(projected, valid):
                    token_rows.append(local[:, mask].T.astype(np.float32, copy=False))
                evidence = outputs["prototype_evidence"].detach().cpu().numpy()
                selected_targets = np.asarray(targets[selected], dtype=np.int64)
                for row_position, target in enumerate(selected_targets):
                    start_proto = int(target) * PROTOTYPES_PER_CLASS
                    choice = start_proto + int(
                        np.argmax(
                            evidence[
                                row_position,
                                start_proto : start_proto + PROTOTYPES_PER_CLASS,
                            ]
                        )
                    )
                    prototype_use[choice] += 1
                class_distance = outputs["class_min_distance"].detach().cpu().numpy()
                class1_margins[selected] = _rival_margin(
                    class_distance, distance=True
                )
                distances = outputs["distances"].detach().cpu().numpy()
                closest = distances.argmin(axis=1)
                predicted_patch_class = closest // PROTOTYPES_PER_CLASS
                target_map = selected_targets[:, None, None]
                nearest_correct += int(
                    ((predicted_patch_class == target_map) & valid).sum()
                )
                nearest_total += int(valid.sum())
                attention = outputs["attention"].detach().cpu().numpy()[:, 0]
                peaks = attention.reshape(attention.shape[0], -1).max(axis=1)
                all_zero += int(np.sum(peaks == 0.0))
                low_peak += int(np.sum(peaks < 1e-4))
                padding_leakage = max(
                    padding_leakage,
                    float(np.max(np.abs(attention * (~valid)))),
                )
            del head
    tokens = np.concatenate(token_rows, axis=0)
    labels = (np.asarray(targets) == FOCUS_CLASS).astype(np.int64)
    class1_use = prototype_use[
        FOCUS_CLASS * PROTOTYPES_PER_CLASS : (FOCUS_CLASS + 1) * PROTOTYPES_PER_CLASS
    ].astype(np.float64)
    class1_probability = class1_use / max(float(class1_use.sum()), 1.0)
    positive = class1_probability > 0.0
    class1_entropy = float(
        -np.sum(class1_probability[positive] * np.log(class1_probability[positive]))
        / math.log(PROTOTYPES_PER_CLASS)
    )
    used_per_class = [
        int(
            np.sum(
                prototype_use[
                    class_index
                    * PROTOTYPES_PER_CLASS : (class_index + 1)
                    * PROTOTYPES_PER_CLASS
                ]
                > 0
            )
        )
        for class_index in range(NUM_CLASSES)
    ]
    return {
        "projected_feature_effective_rank": _effective_rank_from_tokens(tokens),
        "projected_token_rows": int(tokens.shape[0]),
        "prototype_use_counts": prototype_use.tolist(),
        "prototypes_used": int(np.sum(prototype_use > 0)),
        "prototypes_used_per_class": used_per_class,
        "class1_prototype_use_entropy_normalized": class1_entropy,
        "mean_nearest_cross_class_prototype_distance": float(
            np.mean(cross_class_distances)
        ),
        "all_prototype_vectors_distinct": distinct_prototypes,
        "nearest_held_patch_class_purity": float(
            nearest_correct / max(nearest_total, 1)
        ),
        "class1_correct_vs_rival_margin_auroc": float(
            roc_auc_score(labels, class1_margins)
        ),
        "class1_margin_mean_positive": float(class1_margins[labels == 1].mean()),
        "class1_margin_mean_restricted_fp": float(class1_margins[labels == 0].mean()),
        "attention_all_zero_rows": all_zero,
        "attention_low_peak_rows": low_peak,
        "attention_low_peak_fraction": float(low_peak / max(len(targets), 1)),
        "attention_padding_max_abs": padding_leakage,
        "nearest_patch_rows": nearest_total,
    }


def build_clean_analysis(
    *,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    keeper_predictions: np.ndarray,
    folds: np.ndarray,
    diagnostics: Mapping[str, object],
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
    trained_controls = (SELF_ROLE, CLUSTER_ROLE, DEPHASED_ROLE)
    fold_wins = {
        role: sum(
            candidate["folds"][str(fold)]["auroc"]
            > metrics[role]["folds"][str(fold)]["auroc"]
            and candidate["folds"][str(fold)]["restricted_fp_rejection"]
            > metrics[role]["folds"][str(fold)]["restricted_fp_rejection"]
            for fold in FOLDS
        )
        for role in (KEEPER_ROLE, *trained_controls)
    }
    fold_direction_checks = {}
    for fold in FOLDS:
        candidate_fold = candidate["folds"][str(fold)]
        fold_direction_checks[str(fold)] = {
            "auroc_ge_0_50": candidate_fold["auroc"] >= 0.50,
            "keeper_tp_retention_ge_0_95": candidate_fold["keeper_tp_retention"]
            >= 0.95,
            "restricted_fp_rejection_ge_0_25": candidate_fold[
                "restricted_fp_rejection"
            ]
            >= 0.25,
            "control_margins": (
                candidate_fold["auroc"]
                - metrics[SELF_ROLE]["folds"][str(fold)]["auroc"]
                >= 0.01
                and candidate_fold["auroc"]
                - metrics[CLUSTER_ROLE]["folds"][str(fold)]["auroc"]
                >= 0.01
                and candidate_fold["auroc"]
                - metrics[DEPHASED_ROLE]["folds"][str(fold)]["auroc"]
                >= 0.02
            ),
        }
    gates = {
        "candidate_auroc_ge_0_85": candidate["auroc"] >= 0.85,
        "candidate_auprc_ge_0_90": candidate["auprc"] >= 0.90,
        "candidate_keeper_tp_retention_ge_0_95": candidate[
            "keeper_tp_retention"
        ]
        >= 0.95,
        "candidate_keeper_tp_broken_le_26": candidate["keeper_tp_broken"] <= 26,
        "candidate_supports_at_least_8_keeper_fn": candidate[
            "keeper_fn_supported"
        ]
        >= 8,
        "candidate_restricted_fp_rejection_ge_0_25": candidate[
            "restricted_fp_rejection"
        ]
        >= 0.25,
        "candidate_precision_ge_0_75": candidate["precision"] >= 0.75,
        "candidate_auroc_margin_over_keeper_ge_0_02": (
            candidate["auroc"] - metrics[KEEPER_ROLE]["auroc"] >= 0.02
        ),
        "candidate_auroc_margin_over_bbox_keeper_ge_0_015": (
            candidate["auroc"] - metrics[BBOX_KEEPER_ROLE]["auroc"] >= 0.015
        ),
        "candidate_auroc_margin_over_gap_keeper_ge_0_01": (
            candidate["auroc"] - metrics[GAP_KEEPER_ROLE]["auroc"] >= 0.01
        ),
        "candidate_auroc_margin_over_self_ge_0_01": (
            candidate["auroc"] - metrics[SELF_ROLE]["auroc"] >= 0.01
        ),
        "candidate_auroc_margin_over_cluster_only_ge_0_01": (
            candidate["auroc"] - metrics[CLUSTER_ROLE]["auroc"] >= 0.01
        ),
        "candidate_auroc_margin_over_dephased_ge_0_02": (
            candidate["auroc"] - metrics[DEPHASED_ROLE]["auroc"] >= 0.02
        ),
        "attention_roll_reduces_auroc_ge_0_015": (
            candidate["auroc"] - metrics[ROLLED_ROLE]["auroc"] >= 0.015
        ),
        "class_cycle_reduces_auroc_ge_0_025": (
            candidate["auroc"] - metrics[CYCLED_ROLE]["auroc"] >= 0.025
        ),
        "candidate_precision_and_fp_control_beat_trained_controls": all(
            candidate["precision"] - metrics[role]["precision"] >= 0.01
            and candidate["restricted_fp_rejected"]
            - metrics[role]["restricted_fp_rejected"]
            >= 10
            and candidate["keeper_tp_broken"]
            <= metrics[role]["keeper_tp_broken"]
            for role in trained_controls
        ),
        "candidate_wins_auroc_and_fp_rejection_in_4_of_5_folds": all(
            value >= 4 for value in fold_wins.values()
        ),
        "candidate_without_keeper_auroc_ge_0_72": metrics[WITHOUT_KEEPER_ROLE][
            "auroc"
        ]
        >= 0.72,
        "candidate_without_keeper_margin_over_bbox_only_ge_0_02": (
            metrics[WITHOUT_KEEPER_ROLE]["auroc"]
            - metrics[BBOX_ONLY_ROLE]["auroc"]
            >= 0.02
        ),
        "class1_margin_auroc_ge_0_75": float(
            diagnostics["class1_correct_vs_rival_margin_auroc"]
        )
        >= 0.75,
        "projected_feature_effective_rank_ge_16": float(
            diagnostics["projected_feature_effective_rank"]
        )
        >= 16.0,
        "prototype_use_nondegenerate": (
            int(diagnostics["prototypes_used"]) >= 20
            and min(int(value) for value in diagnostics["prototypes_used_per_class"])
            >= 3
            and float(diagnostics["class1_prototype_use_entropy_normalized"])
            >= 0.70
        ),
        "prototype_separation_and_patch_purity_pass": (
            float(diagnostics["mean_nearest_cross_class_prototype_distance"])
            >= 0.10
            and bool(diagnostics["all_prototype_vectors_distinct"])
            and float(diagnostics["nearest_held_patch_class_purity"]) >= 0.60
        ),
        "attention_maps_nondegenerate_and_padding_clean": (
            int(diagnostics["attention_all_zero_rows"]) == 0
            and float(diagnostics["attention_low_peak_fraction"]) < 0.01
            and float(diagnostics["attention_padding_max_abs"]) == 0.0
        ),
        "all_fold_direction_checks_pass": all(
            all(bool(value) for value in record.values())
            for record in fold_direction_checks.values()
        ),
    }
    return {
        "roles": metrics,
        "prototype_diagnostics": dict(diagnostics),
        "candidate_fold_wins": fold_wins,
        "fold_direction_checks": fold_direction_checks,
        "mechanism_gates": {key: bool(value) for key, value in gates.items()},
        "mechanism_gates_passed": all(bool(value) for value in gates.values()),
    }


def build_cohort_artifacts(
    *,
    extraction: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    learned_outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    positions = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    observed_indices = np.asarray(extraction["sample_indices"], dtype=np.int64)[positions]
    if not np.array_equal(observed_indices, positions):
        raise ValueError("Cohort positions do not match sample indices")
    features = np.asarray(extraction["features"][positions], dtype=np.float16)
    valid_masks = np.asarray(extraction["valid_masks"], dtype=np.bool_)[positions]
    bbox_masks = np.asarray(extraction["bbox_masks"], dtype=np.bool_)[positions]
    keeper_probabilities = np.asarray(
        extraction["probabilities"], dtype=np.float32
    )[positions]
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    outputs = {
        role: {
            key: np.asarray(value[positions], dtype=np.float32)
            for key, value in learned_outputs[role].items()
        }
        for role in LEARNED_DESCRIPTOR_ROLES
    }
    descriptors = build_role_descriptors(
        outputs,
        features,
        valid_masks,
        bbox_masks,
        keeper_probabilities,
    )
    return {
        "positions": positions,
        "features": features,
        "valid_masks": valid_masks,
        "bbox_masks": bbox_masks,
        "keeper_probabilities": keeper_probabilities,
        "keeper_predictions": keeper_probabilities.argmax(axis=1).astype(np.int64),
        "targets": targets,
        "folds": folds,
        "model_boxes": np.asarray(extraction["model_boxes"], dtype=np.float32)[positions],
        "crop_boxes": np.asarray(extraction["crop_boxes"], dtype=np.float32)[positions],
        "outputs": outputs,
        "descriptors": descriptors,
    }


def _state_hashes(
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]]
) -> Dict[str, Dict[str, str]]:
    return {
        str(fold): {
            role: _state_mapping_sha256(state)
            for role, state in states[str(fold)].items()
        }
        for fold in FOLDS
    }


def save_replay_artifacts(
    *,
    output_dir: Path,
    cohort: Sequence[CleanTrainRow],
    cohort_artifacts: Mapping[str, object],
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    readout_states: Mapping[str, object],
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    feature_path = root / "cohort_block2_features.npy"
    np.save(
        feature_path,
        np.asarray(cohort_artifacts["features"], dtype=np.float16),
        allow_pickle=False,
    )
    geometry_path = root / "cohort_geometry.npz"
    np.savez_compressed(
        geometry_path,
        sample_indices=np.asarray(cohort_artifacts["positions"], dtype=np.int64),
        targets=np.asarray(cohort_artifacts["targets"], dtype=np.int64),
        folds=np.asarray(cohort_artifacts["folds"], dtype=np.int64),
        source_stems=np.asarray([row.source_stem for row in cohort], dtype=str),
        keeper_probabilities=np.asarray(
            cohort_artifacts["keeper_probabilities"], dtype=np.float32
        ),
        valid_masks=np.asarray(cohort_artifacts["valid_masks"], dtype=np.bool_),
        bbox_masks=np.asarray(cohort_artifacts["bbox_masks"], dtype=np.bool_),
        model_boxes=np.asarray(cohort_artifacts["model_boxes"], dtype=np.float32),
        crop_boxes=np.asarray(cohort_artifacts["crop_boxes"], dtype=np.float32),
    )
    state_path = root / "oof_head_states.pt"
    torch.save(states, state_path)
    state_hash_path = root / "oof_head_state_hashes.json"
    state_hashes = _state_hashes(states)
    _write_json(state_hash_path, state_hashes)
    descriptor_path = root / "cohort_descriptors.npz"
    np.savez_compressed(
        descriptor_path,
        **{
            role: np.asarray(cohort_artifacts["descriptors"][role], dtype=np.float64)
            for role in ROLE_NAMES
        },
    )
    output_path = root / "cohort_learned_outputs.npz"
    np.savez_compressed(
        output_path,
        **{
            f"{role}__{key}": np.asarray(value, dtype=np.float32)
            for role, role_outputs in cohort_artifacts["outputs"].items()
            for key, value in role_outputs.items()
        },
    )
    readout_path = root / "readout_states.json"
    _write_json(readout_path, readout_states)
    prediction_path = root / "oof_predictions.csv"
    prediction_rows = []
    for position, row in enumerate(cohort):
        record: Dict[str, object] = {
            "sample_index": row.sample_index,
            "source_stem": row.source_stem,
            "fold": row.fold,
            "target": row.target,
            "keeper_prediction": row.keeper_prediction,
        }
        for role in ROLE_NAMES:
            record[f"score__{role}"] = format(float(scores[role][position]), ".17g")
            record[f"action__{role}"] = int(bool(actions[role][position]))
        prediction_rows.append(record)
    _write_csv(prediction_path, prediction_rows)
    paths = (
        feature_path,
        geometry_path,
        state_path,
        state_hash_path,
        descriptor_path,
        output_path,
        readout_path,
        prediction_path,
    )
    return {
        path.name: {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in paths
    } | {"state_hashes": state_hashes}


def _read_prediction_artifact(
    path: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {
        role: np.asarray([float(row[f"score__{role}"]) for row in rows])
        for role in ROLE_NAMES
    }
    actions = {
        role: np.asarray(
            [bool(int(row[f"action__{role}"])) for row in rows], dtype=np.bool_
        )
        for role in ROLE_NAMES
    }
    return rows, scores, actions


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
            fit = fold_values != int(record["held_fold"])
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
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(output_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, persisted_scores, persisted_actions = _read_prediction_artifact(
        root / "oof_predictions.csv"
    )
    if len(prediction_rows) != EXPECTED_COHORT_ROWS:
        raise ValueError("Attention-MaxSep replay cohort row count differs")
    sample_indices = np.asarray(
        [int(row["sample_index"]) for row in prediction_rows], dtype=np.int64
    )
    if _index_sha256(sample_indices.tolist()) != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError("Attention-MaxSep replay cohort order differs")
    with np.load(root / "cohort_geometry.npz", allow_pickle=False) as archive:
        geometry = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(sample_indices, geometry["sample_indices"]):
        raise ValueError("Replay prediction and geometry row orders differ")
    features = np.load(root / "cohort_block2_features.npy", allow_pickle=False)
    states = torch.load(
        root / "oof_head_states.pt", map_location="cpu", weights_only=True
    )
    observed_state_hashes = _state_hashes(states)
    expected_state_hashes = json.loads(
        (root / "oof_head_state_hashes.json").read_text(encoding="utf-8")
    )
    states_exact = observed_state_hashes == expected_state_hashes
    outputs = evaluate_oof_heads(
        features=features,
        valid_masks=geometry["valid_masks"].astype(np.bool_),
        folds=geometry["folds"].astype(np.int64),
        states=states,
        device=device,
    )
    with np.load(root / "cohort_learned_outputs.npz", allow_pickle=False) as archive:
        persisted_outputs = {key: np.asarray(archive[key]) for key in archive.files}
    output_error = max(
        float(np.max(np.abs(value - persisted_outputs[f"{role}__{key}"])))
        for role, role_outputs in outputs.items()
        for key, value in role_outputs.items()
    )
    descriptors = build_role_descriptors(
        outputs,
        features,
        geometry["valid_masks"].astype(np.bool_),
        geometry["bbox_masks"].astype(np.bool_),
        geometry["keeper_probabilities"].astype(np.float32),
    )
    with np.load(root / "cohort_descriptors.npz", allow_pickle=False) as archive:
        persisted_descriptors = {
            role: np.asarray(archive[role], dtype=np.float64) for role in ROLE_NAMES
        }
    descriptor_error = max(
        float(np.max(np.abs(descriptors[role] - persisted_descriptors[role])))
        for role in ROLE_NAMES
    )
    readout_states = json.loads(
        (root / "readout_states.json").read_text(encoding="utf-8")
    )
    folds = geometry["folds"].astype(np.int64)
    scores, actions = apply_readout_states(descriptors, folds, readout_states)
    score_error = max(
        float(np.max(np.abs(scores[role] - persisted_scores[role])))
        for role in ROLE_NAMES
    )
    actions_exact = all(
        np.array_equal(actions[role], persisted_actions[role]) for role in ROLE_NAMES
    )
    targets = geometry["targets"].astype(np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    threshold_error = _verify_readout_thresholds(
        descriptors, folds, labels, readout_states
    )
    diagnostics = prototype_diagnostics(
        features=features,
        valid_masks=geometry["valid_masks"].astype(np.bool_),
        bbox_masks=geometry["bbox_masks"].astype(np.bool_),
        targets=targets,
        folds=folds,
        states=states,
        device=device,
    )
    analysis = build_clean_analysis(
        scores=scores,
        actions=actions,
        targets=targets,
        keeper_predictions=geometry["keeper_probabilities"].argmax(axis=1),
        folds=folds,
        diagnostics=diagnostics,
    )
    analysis_error = (
        _recursive_numeric_difference(expected_analysis, analysis)
        if expected_analysis is not None
        else 0.0
    )
    passed = bool(
        states_exact
        and output_error <= MAX_REPLAY_ERROR
        and descriptor_error <= MAX_REPLAY_ERROR
        and score_error <= MAX_REPLAY_ERROR
        and threshold_error <= MAX_REPLAY_ERROR
        and actions_exact
        and analysis_error <= MAX_REPLAY_ERROR
    )
    return {
        "passed": passed,
        "rows": len(prediction_rows),
        "states_exact": states_exact,
        "output_max_abs_error": output_error,
        "descriptor_max_abs_error": descriptor_error,
        "score_max_abs_error": score_error,
        "threshold_max_abs_error": threshold_error,
        "actions_exact": actions_exact,
        "analysis_maximum_numeric_difference": analysis_error,
        "analysis": analysis,
    }


def learned_descriptor_torch(
    outputs: Mapping[str, Tensor],
    valid_masks: Tensor,
    bbox_masks: Tensor,
    keeper_probabilities: Tensor,
    *,
    include_keeper: bool,
) -> Tensor:
    attention = outputs.get("evidence_attention", outputs["attention"])
    if attention.ndim == 4:
        attention = attention[:, 0]
    valid = valid_masks.bool()
    bbox = bbox_masks.bool()
    valid_float = valid.to(dtype=attention.dtype)
    attention = attention.clamp_min(0.0) * valid_float
    valid_count = valid_float.sum(dim=(1, 2)).clamp_min(1.0)
    attention_total = attention.sum(dim=(1, 2)).clamp_min(1e-12)
    probability = attention / attention_total[:, None, None]
    entropy = -(
        probability * probability.clamp_min(1e-12).log()
    ).sum(dim=(1, 2))
    entropy = entropy / torch.maximum(
        valid_count, valid_count.new_full(valid_count.shape, 2.0)
    ).log()
    stats = torch.stack(
        (
            attention_total / valid_count,
            entropy,
            attention.flatten(1).amax(dim=1),
            (attention * bbox.to(attention.dtype)).sum(dim=(1, 2))
            / attention_total,
            (attention * (~bbox).to(attention.dtype)).sum(dim=(1, 2))
            / attention_total,
        ),
        dim=1,
    )

    def rival_margin(values: Tensor, *, distance: bool = False) -> Tensor:
        rivals = torch.cat(
            (values[:, :FOCUS_CLASS], values[:, FOCUS_CLASS + 1 :]), dim=1
        )
        if distance:
            return rivals.amin(dim=1) - values[:, FOCUS_CLASS]
        return values[:, FOCUS_CLASS] - rivals.amax(dim=1)

    margins = torch.stack(
        (
            rival_margin(outputs["attention_logits"]),
            rival_margin(outputs["prototype_logits"]),
            rival_margin(outputs["class_min_distance"], distance=True),
        ),
        dim=1,
    )
    descriptor = torch.cat(
        (
            outputs["attention_logits"],
            outputs["prototype_logits"],
            outputs["class_evidence_max"],
            outputs["class_evidence_mean"],
            outputs["class_min_distance"],
            stats,
            margins,
        ),
        dim=1,
    )
    if include_keeper:
        descriptor = torch.cat(
            (descriptor, keeper_probabilities.clamp(1e-7, 1.0).log()), dim=1
        )
    expected = 38 if include_keeper else 33
    if descriptor.shape[1] != expected:
        raise RuntimeError("Torch learned descriptor shape differs")
    return descriptor


def _readout_fold_state(states: Mapping[str, object], role: str, fold: int) -> Mapping[str, object]:
    records = [
        record
        for record in states[role]["folds"]
        if int(record["held_fold"]) == int(fold)
    ]
    if len(records) != 1:
        raise ValueError(f"Readout state lacks unique {role} fold {fold}")
    return records[0]


def readout_score_torch(descriptor: Tensor, state: Mapping[str, object]) -> Tensor:
    mean = descriptor.new_tensor(state["scaler_mean"])
    scale = descriptor.new_tensor(state["scaler_scale"])
    coefficient = descriptor.new_tensor(state["coefficient"])[0]
    intercept = descriptor.new_tensor(state["intercept"])[0]
    decision = ((descriptor - mean) / scale * coefficient).sum(dim=1) + intercept
    return decision.sigmoid()


class StaticMaxSepReadout(nn.Module):
    def __init__(
        self,
        head: AttentionMaxSepPrototypeHead,
        state: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.head = copy.deepcopy(head).float().eval()
        self.register_buffer("mean", torch.tensor(state["scaler_mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(state["scaler_scale"], dtype=torch.float32))
        self.register_buffer(
            "coefficient",
            torch.tensor(state["coefficient"], dtype=torch.float32)[0],
        )
        self.register_buffer(
            "intercept", torch.tensor(state["intercept"], dtype=torch.float32)[0]
        )

    def forward(
        self,
        features: Tensor,
        valid_masks: Tensor,
        bbox_masks: Tensor,
        keeper_probabilities: Tensor,
    ) -> Tensor:
        outputs = self.head(features, valid_masks)
        descriptor = learned_descriptor_torch(
            outputs,
            valid_masks,
            bbox_masks,
            keeper_probabilities,
            include_keeper=True,
        )
        decision = (
            (descriptor - self.mean) / self.scale * self.coefficient
        ).sum(dim=1) + self.intercept
        return decision.sigmoid()


def numeric_precision_checks(
    *,
    cohort_artifacts: Mapping[str, object],
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    readout_states: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    fold = 0
    selected = np.flatnonzero(np.asarray(cohort_artifacts["folds"]) == fold)[:8]
    features = np.asarray(cohort_artifacts["features"])[selected].astype(np.float32)
    valid = np.asarray(cohort_artifacts["valid_masks"])[selected].astype(np.bool_)
    bbox = np.asarray(cohort_artifacts["bbox_masks"])[selected].astype(np.bool_)
    keeper = np.asarray(cohort_artifacts["keeper_probabilities"])[selected].astype(
        np.float32
    )
    targets = np.asarray(cohort_artifacts["targets"])[selected].astype(np.int64)
    state = _readout_fold_state(readout_states, CANDIDATE_ROLE, fold)
    fp32_head = _load_head_state(
        states[str(fold)][CANDIDATE_ROLE], device=torch.device("cpu")
    )
    fp64_head = _load_head_state(
        states[str(fold)][CANDIDATE_ROLE],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    feature_tensor = torch.from_numpy(features)
    valid_tensor = torch.from_numpy(valid)
    bbox_tensor = torch.from_numpy(bbox)
    keeper_tensor = torch.from_numpy(keeper)
    with torch.inference_mode():
        fp32_outputs = fp32_head(feature_tensor, valid_tensor)
        fp64_outputs = fp64_head(feature_tensor.double(), valid_tensor)
        fp32_descriptor = learned_descriptor_torch(
            fp32_outputs,
            valid_tensor,
            bbox_tensor,
            keeper_tensor,
            include_keeper=True,
        )
        fp64_descriptor = learned_descriptor_torch(
            fp64_outputs,
            valid_tensor,
            bbox_tensor,
            keeper_tensor.double(),
            include_keeper=True,
        )
        fp32_scores = readout_score_torch(fp32_descriptor, state)
        fp64_scores = readout_score_torch(fp64_descriptor, state)
    bf16_head = _load_head_state(states[str(fold)][CANDIDATE_ROLE], device=device)
    bf16_features = torch.from_numpy(features).to(device).requires_grad_(True)
    bf16_valid = torch.from_numpy(valid).to(device)
    bf16_bbox = torch.from_numpy(bbox).to(device)
    bf16_keeper = torch.from_numpy(keeper).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bf16_outputs = bf16_head(bf16_features, bf16_valid)
        bf16_descriptor = learned_descriptor_torch(
            bf16_outputs,
            bf16_valid,
            bf16_bbox,
            bf16_keeper,
            include_keeper=True,
        )
        bf16_scores = readout_score_torch(bf16_descriptor, state)
        bf16_loss = (
            F.cross_entropy(
                bf16_outputs["prototype_logits"],
                torch.from_numpy(targets).to(device),
            )
            + F.cross_entropy(
                bf16_outputs["attention_logits"],
                torch.from_numpy(targets).to(device),
            )
        )
    bf16_loss.backward()
    bf16_gradient_finite = bool(torch.isfinite(bf16_features.grad).all()) and all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in bf16_head.parameters()
    )
    fp32_numpy = fp32_scores.detach().cpu().numpy().astype(np.float64)
    fp64_numpy = fp64_scores.detach().cpu().numpy().astype(np.float64)
    bf16_numpy = bf16_scores.detach().float().cpu().numpy().astype(np.float64)
    descriptor_numpy = fp64_descriptor.detach().cpu().numpy().astype(np.float64)
    numpy_reference = learned_descriptor(
        {
            key: value.detach().cpu().numpy()
            for key, value in fp64_outputs.items()
            if key
            in {
                "attention",
                "attention_logits",
                "prototype_logits",
                "class_evidence_max",
                "class_evidence_mean",
                "class_min_distance",
            }
        },
        valid,
        bbox,
        keeper.astype(np.float64),
        include_keeper=True,
    )
    descriptor_oracle_error = float(
        np.max(np.abs(descriptor_numpy - numpy_reference))
    )
    fp32_fp64_score_error = float(np.max(np.abs(fp32_numpy - fp64_numpy)))
    bf16_fp32_score_error = float(np.max(np.abs(bf16_numpy - fp32_numpy)))
    threshold = float(state["threshold"])
    non_near_tie = np.abs(fp32_numpy - threshold) > 0.02
    actions_exact = bool(
        np.array_equal(
            fp32_numpy[non_near_tie] >= threshold,
            bf16_numpy[non_near_tie] >= threshold,
        )
    )
    checks = {
        "torch_numpy_descriptor_error_le_1e_10": descriptor_oracle_error <= 1e-10,
        "fp32_fp64_score_error_le_1e_5": fp32_fp64_score_error <= 1e-5,
        "bf16_fp32_score_error_le_0_02": bf16_fp32_score_error <= 0.02,
        "bf16_outputs_finite": bool(torch.isfinite(bf16_scores).all()),
        "bf16_gradients_finite": bf16_gradient_finite,
        "bf16_non_near_tie_actions_exact": actions_exact,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "torch_numpy_descriptor_max_abs_error": descriptor_oracle_error,
        "fp32_fp64_score_max_abs_error": fp32_fp64_score_error,
        "bf16_fp32_score_max_abs_error": bf16_fp32_score_error,
        "non_near_tie_rows": int(non_near_tie.sum()),
    }


def static_export_checks(
    *,
    output_dir: Path,
    cohort_artifacts: Mapping[str, object],
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    readout_states: Mapping[str, object],
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    fold = 0
    selected = np.flatnonzero(np.asarray(cohort_artifacts["folds"]) == fold)[:8]
    features = torch.from_numpy(
        np.asarray(cohort_artifacts["features"])[selected].astype(np.float32)
    )
    valid = torch.from_numpy(
        np.asarray(cohort_artifacts["valid_masks"])[selected].astype(np.bool_)
    )
    bbox = torch.from_numpy(
        np.asarray(cohort_artifacts["bbox_masks"])[selected].astype(np.bool_)
    )
    keeper = torch.from_numpy(
        np.asarray(cohort_artifacts["keeper_probabilities"])[selected].astype(
            np.float32
        )
    )
    head = _load_head_state(
        states[str(fold)][CANDIDATE_ROLE], device=torch.device("cpu")
    )
    state = _readout_fold_state(readout_states, CANDIDATE_ROLE, fold)
    wrapper = StaticMaxSepReadout(head, state).eval()
    path = Path(output_dir) / "attention_maxsep_static_readout.onnx"
    with torch.inference_mode():
        reference = wrapper(features, valid, bbox, keeper).numpy()
    torch.onnx.export(
        wrapper,
        (features, valid, bbox, keeper),
        str(path),
        input_names=("features", "valid_masks", "bbox_masks", "keeper_probabilities"),
        output_names=("class1_support_score",),
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    custom_domains = sorted(
        {node.domain for node in graph.graph.node if str(node.domain).strip()}
    )
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    observed = session.run(
        None,
        {
            "features": features.numpy(),
            "valid_masks": valid.numpy(),
            "bbox_masks": bbox.numpy(),
            "keeper_probabilities": keeper.numpy(),
        },
    )[0]
    maximum_error = float(np.max(np.abs(reference - observed)))
    threshold = float(state["threshold"])
    non_near_tie = np.abs(reference - threshold) > 1e-3
    actions_exact = bool(
        np.array_equal(
            reference[non_near_tie] >= threshold,
            observed[non_near_tie] >= threshold,
        )
    )
    checks = {
        "onnx_no_custom_domain": not custom_domains,
        "onnx_output_error_le_1e_4": maximum_error <= 1e-4,
        "onnx_non_near_tie_actions_exact": actions_exact,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
        "custom_domains": custom_domains,
        "maximum_output_error": maximum_error,
        "non_near_tie_rows": int(non_near_tie.sum()),
    }


def benchmark_candidate_head(
    *,
    cohort_artifacts: Mapping[str, object],
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    device: torch.device,
) -> Dict[str, object]:
    fold = 0
    selected = np.flatnonzero(np.asarray(cohort_artifacts["folds"]) == fold)[:32]
    features = torch.from_numpy(
        np.asarray(cohort_artifacts["features"])[selected].astype(np.float32)
    ).to(device)
    valid = torch.from_numpy(
        np.asarray(cohort_artifacts["valid_masks"])[selected].astype(np.bool_)
    ).to(device)
    head = _load_head_state(states[str(fold)][CANDIDATE_ROLE], device=device)

    def measure_candidate(repeats: int = 50) -> Tuple[float, int]:
        with torch.inference_mode():
            for _ in range(10):
                head(features, valid)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = int(torch.cuda.memory_allocated(device))
        durations = []
        with torch.inference_mode():
            for _ in range(repeats):
                started = time.perf_counter()
                head(features, valid)
                torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - started)
        peak = int(torch.cuda.max_memory_allocated(device))
        return float(np.median(durations)), max(peak - baseline, 0)

    latency, extra_bytes = measure_candidate()
    checks = {
        "batch_size_exact_32": int(selected.size) == 32,
        "median_latency_le_3_ms": latency <= 0.003,
        "extra_peak_cuda_le_0_25_gib": extra_bytes <= int(0.25 * 1024**3),
        "trainable_parameters_le_25000": head.trainable_parameter_count <= 25_000,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "median_latency_seconds": latency,
        "median_latency_ms": latency * 1000.0,
        "extra_peak_cuda_bytes": extra_bytes,
        "trainable_parameters": head.trainable_parameter_count,
    }


def _normalized_rgb(image: Tensor, semantics: Mapping[str, object]) -> np.ndarray:
    mean = torch.tensor(semantics["input_mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(semantics["input_std"], dtype=torch.float32).view(3, 1, 1)
    value = (image.detach().cpu().float() * std + mean).clamp(0.0, 1.0)
    return value.permute(1, 2, 0).numpy()


def _color_heatmap(value: np.ndarray, *, signed: bool = False) -> Image.Image:
    array = np.nan_to_num(
        np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if signed:
        scale = max(float(np.max(np.abs(array))), 1e-12)
        normalized = np.clip(array / scale, -1.0, 1.0)
        red = (255.0 * np.clip(normalized, 0.0, 1.0)).astype(np.uint8)
        blue = (255.0 * np.clip(-normalized, 0.0, 1.0)).astype(np.uint8)
        green = (64.0 * (1.0 - np.abs(normalized))).astype(np.uint8)
    else:
        minimum = float(array.min())
        maximum = float(array.max())
        normalized = (array - minimum) / max(maximum - minimum, 1e-12)
        red = (255.0 * normalized).astype(np.uint8)
        green = (255.0 * np.sqrt(normalized)).astype(np.uint8)
        blue = (80.0 * (1.0 - normalized)).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=2), mode="RGB")


def _overlay_map(
    rgb: np.ndarray,
    value: np.ndarray,
    *,
    size: int,
    signed: bool = False,
) -> Image.Image:
    base = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    base = base.resize((size, size), Image.Resampling.BILINEAR)
    heatmap = _color_heatmap(value, signed=signed).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return Image.blend(base, heatmap, 0.52)


def _draw_bbox_from_mask(image: Image.Image, bbox_mask: np.ndarray) -> Image.Image:
    result = image.copy()
    points = np.argwhere(np.asarray(bbox_mask, dtype=np.bool_))
    if points.size == 0:
        return result
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    scale_x = result.width / FEATURE_SIZE
    scale_y = result.height / FEATURE_SIZE
    draw = ImageDraw.Draw(result)
    draw.rectangle(
        (
            int(x0 * scale_x),
            int(y0 * scale_y),
            int(x1 * scale_x) - 1,
            int(y1 * scale_y) - 1,
        ),
        outline=(255, 255, 255),
        width=3,
    )
    draw.rectangle(
        (
            int(x0 * scale_x) + 3,
            int(y0 * scale_y) + 3,
            int(x1 * scale_x) - 4,
            int(y1 * scale_y) - 4,
        ),
        outline=(20, 20, 20),
        width=1,
    )
    return result


def render_xai_contact_sheet(
    *,
    path: Path,
    model: nn.Module,
    dataset,
    transform,
    semantics: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    readout_states: Mapping[str, object],
    persisted_scores: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, object]:
    positions = fixed_xai_positions(cohort)
    selected_rows = [cohort[position] for position in positions]
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in selected_rows],
        brightness=1.0,
        contrast=1.0,
        batch_size=len(selected_rows),
        num_workers=0,
        context="attention_maxsep_prototype_a0_xai",
    )
    batches = list(loader)
    if len(batches) != 1:
        raise RuntimeError("Fixed XAI loader must emit exactly one batch")
    images, batch_targets, metadata = batches[0]
    if not isinstance(metadata, Mapping):
        raise ValueError("XAI loader requires tensor metadata")
    observed_indices = metadata["sample_index"].detach().cpu().long().tolist()
    if observed_indices != [row.sample_index for row in selected_rows]:
        raise ValueError("XAI loader changed fixed sample order")
    if batch_targets.detach().cpu().long().tolist() != [
        row.target for row in selected_rows
    ]:
        raise ValueError("XAI loader changed fixed targets")
    image_masks = _metadata_tensor(
        metadata, "image_mask", device=device, dtype=torch.bool
    )
    model_boxes = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)
    crop_boxes = _metadata_tensor(
        metadata, "crop_bbox", device=device, dtype=torch.float32
    )
    if image_masks is None or model_boxes is None or crop_boxes is None:
        raise ValueError("XAI metadata lacks image mask or bbox")
    if image_masks.ndim == 3:
        image_masks = image_masks[:, None]
    images_device = images.to(device=device, dtype=torch.float32)
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    state_before = _model_state_sha256(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    titles = (
        "RGB + bbox",
        "candidate attention",
        "own prototype",
        "rival prototype",
        "own - rival",
        "self attention",
        "cluster-only",
        "dephased",
        "rolled attention",
        "input gradient",
    )
    maps: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "candidate_attention",
            "own_similarity",
            "rival_similarity",
            "signed_margin",
            "self_attention",
            "cluster_attention",
            "dephased_attention",
            "rolled_attention",
            "input_gradient",
        )
    }
    reconstructed_scores = []
    gradient_repeat_error = 0.0
    padding_leakage = 0.0
    finite = True
    cell = 160
    header = 34
    label_height = 26
    sheet = Image.new(
        "RGB",
        (len(titles) * cell, header + len(selected_rows) * (cell + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, title in enumerate(titles):
        draw.text((column * cell + 4, 11), title, fill="black", font=font)
    try:
        for local_position, (cohort_position, row) in enumerate(
            zip(positions, selected_rows)
        ):
            fold = int(row.fold)
            heads = {
                role: _load_head_state(states[str(fold)][role], device=device)
                for role in TRAINED_ROLES
            }
            for head in heads.values():
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
            readout_state = _readout_fold_state(
                readout_states, CANDIDATE_ROLE, fold
            )
            candidate_head = heads[CANDIDATE_ROLE]
            image = images_device[local_position : local_position + 1]
            image_mask = image_masks[local_position : local_position + 1]
            model_box = model_boxes[local_position : local_position + 1, :4]
            crop_box = crop_boxes[local_position : local_position + 1, :4]

            def gradient_pass() -> Tuple[float, np.ndarray, Dict[str, Tensor], Tensor, Tensor]:
                leaf = image.detach().clone().requires_grad_(True)
                mask_float = image_mask.to(dtype=leaf.dtype)
                masked_image = leaf * mask_float + image.detach() * (1.0 - mask_float)
                logits, block2, _ = _captured_keeper_forward(
                    model, masked_image, image_mask, model_box
                )
                valid, bbox = _feature_geometry(image_mask, crop_box)
                outputs = candidate_head(block2, valid)
                descriptor = learned_descriptor_torch(
                    outputs,
                    valid,
                    bbox,
                    logits.softmax(dim=1),
                    include_keeper=True,
                )
                score = readout_score_torch(descriptor, readout_state)[0]
                score.backward()
                gradient = (
                    leaf.grad.detach().abs().sum(dim=1)[0]
                    * image_mask.detach()[0, 0].to(dtype=leaf.dtype)
                )
                return (
                    float(score.detach()),
                    gradient.cpu().numpy(),
                    {key: value.detach() for key, value in outputs.items()},
                    block2.detach(),
                    valid.detach(),
                )

            first_score, first_gradient, candidate_outputs, block2, valid = gradient_pass()
            second_score, second_gradient, _, _, _ = gradient_pass()
            gradient_repeat_error = max(
                gradient_repeat_error,
                abs(first_score - second_score),
                float(np.max(np.abs(first_gradient - second_gradient))),
            )
            reconstructed_scores.append(first_score)
            with torch.inference_mode():
                bbox = _feature_geometry(image_mask, crop_box)[1]
                self_outputs = heads[SELF_ROLE](block2, valid)
                cluster_outputs = heads[CLUSTER_ROLE](block2, valid)
                dephased_features = channel_dephase(
                    block2, dephase_gather_indices(fold, device=device)
                )
                dephased_outputs = heads[DEPHASED_ROLE](dephased_features, valid)
                rolled_outputs = heads[CANDIDATE_ROLE](
                    block2, valid, attention_roll=ATTENTION_ROLL
                )
            target = int(row.target)
            evidence = candidate_outputs["prototype_evidence"][0]
            own_start = target * PROTOTYPES_PER_CLASS
            own_indices = torch.arange(
                own_start,
                own_start + PROTOTYPES_PER_CLASS,
                device=device,
            )
            all_indices = torch.arange(NUM_PROTOTYPES, device=device)
            rival_indices = all_indices[
                (all_indices < own_start)
                | (all_indices >= own_start + PROTOTYPES_PER_CLASS)
            ]
            own_index = int(own_indices[evidence[own_indices].argmax()])
            rival_index = int(rival_indices[evidence[rival_indices].argmax()])
            valid_numpy = valid[0].cpu().numpy().astype(np.bool_)
            bbox_numpy = bbox[0].cpu().numpy().astype(np.bool_)
            row_maps = {
                "candidate_attention": candidate_outputs["attention"][0, 0]
                .cpu()
                .numpy(),
                "own_similarity": candidate_outputs["similarities"][0, own_index]
                .cpu()
                .numpy()
                * valid_numpy,
                "rival_similarity": candidate_outputs["similarities"][0, rival_index]
                .cpu()
                .numpy()
                * valid_numpy,
                "self_attention": self_outputs["attention"][0, 0].cpu().numpy(),
                "cluster_attention": cluster_outputs["attention"][0, 0]
                .cpu()
                .numpy(),
                "dephased_attention": dephased_outputs["attention"][0, 0]
                .cpu()
                .numpy(),
                "rolled_attention": rolled_outputs["evidence_attention"][0, 0]
                .cpu()
                .numpy(),
                "input_gradient": first_gradient,
            }
            row_maps["signed_margin"] = (
                row_maps["own_similarity"] - row_maps["rival_similarity"]
            )
            for key, value in row_maps.items():
                finite = finite and bool(np.isfinite(value).all())
                maps[key].append(np.asarray(value, dtype=np.float32))
            for key in (
                "candidate_attention",
                "self_attention",
                "cluster_attention",
                "dephased_attention",
                "rolled_attention",
            ):
                padding_leakage = max(
                    padding_leakage,
                    float(np.max(np.abs(row_maps[key] * (~valid_numpy)))),
                )
            image_padding = (~image_mask[0, 0].detach().cpu().numpy().astype(np.bool_))
            padding_leakage = max(
                padding_leakage,
                float(np.max(np.abs(first_gradient * image_padding))),
            )
            rgb = _normalized_rgb(images[local_position], semantics)
            input_cell = _draw_bbox_from_mask(
                Image.fromarray((rgb * 255.0).astype(np.uint8)).resize(
                    (cell, cell), Image.Resampling.BILINEAR
                ),
                bbox_numpy,
            )
            visual_cells = [
                input_cell,
                _overlay_map(rgb, row_maps["candidate_attention"], size=cell),
                _overlay_map(rgb, row_maps["own_similarity"], size=cell),
                _overlay_map(rgb, row_maps["rival_similarity"], size=cell),
                _overlay_map(
                    rgb, row_maps["signed_margin"], size=cell, signed=True
                ),
                _overlay_map(rgb, row_maps["self_attention"], size=cell),
                _overlay_map(rgb, row_maps["cluster_attention"], size=cell),
                _overlay_map(rgb, row_maps["dephased_attention"], size=cell),
                _overlay_map(rgb, row_maps["rolled_attention"], size=cell),
                _overlay_map(rgb, row_maps["input_gradient"], size=cell),
            ]
            top = header + local_position * (cell + label_height)
            for column, visual in enumerate(visual_cells):
                sheet.paste(visual, (column * cell, top))
            category = (
                "TP"
                if row.target == FOCUS_CLASS
                and row.keeper_prediction == FOCUS_CLASS
                else "FN"
                if row.target == FOCUS_CLASS
                else "FP"
            )
            draw.text(
                (4, top + cell + 7),
                f"fold={fold} {category} idx={row.sample_index} y={row.target} "
                f"score={first_score:.4f}",
                fill="black",
                font=font,
            )
            del heads
    finally:
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
    state_after = _model_state_sha256(model)
    persisted = np.asarray(persisted_scores[CANDIDATE_ROLE], dtype=np.float64)[
        positions
    ]
    score_error = float(
        np.max(np.abs(np.asarray(reconstructed_scores, dtype=np.float64) - persisted))
    )
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    arrays_path = path.with_name("attention_maxsep_xai_arrays.npz")
    np.savez_compressed(
        arrays_path,
        sample_indices=np.asarray([row.sample_index for row in selected_rows]),
        cohort_positions=np.asarray(positions, dtype=np.int64),
        reconstructed_scores=np.asarray(reconstructed_scores, dtype=np.float64),
        **{key: np.stack(value) for key, value in maps.items()},
    )
    checks = {
        "fixed_15_rows_exact": len(selected_rows) == 15,
        "all_arrays_finite": finite,
        "two_pass_deterministic_le_1e_7": gradient_repeat_error <= 1e-7,
        "readout_score_reconstruction_le_1e_5": score_error <= 1e-5,
        "padding_leakage_exact_zero": padding_leakage == 0.0,
        "keeper_state_exact": state_before == state_after,
    }
    return {
        "automatic_passed": all(checks.values()),
        "checks": checks,
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": _sha256(arrays_path),
        "rows": len(selected_rows),
        "sample_indices": [row.sample_index for row in selected_rows],
        "gradient_repeat_max_abs_error": gradient_repeat_error,
        "score_reconstruction_max_abs_error": score_error,
        "padding_max_abs": padding_leakage,
        "loader": dict(loader_summary),
    }


def _gpu_snapshot() -> Dict[str, object]:
    result: Dict[str, object] = {}
    queries = {
        "gpu": "name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "compute_processes": "pid,process_name,used_memory",
    }
    for key, query in queries.items():
        arguments = [
            "nvidia-smi",
            f"--query-{'gpu' if key == 'gpu' else 'compute-apps'}={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                arguments,
                text=True,
                encoding="utf-8",
                stderr=subprocess.STDOUT,
            ).strip()
            result[key] = output if key == "gpu" else output.splitlines() if output else []
        except (OSError, subprocess.CalledProcessError) as error:
            result[f"{key}_query_error"] = str(error)
    return result


def _dataset_stat_snapshot(root: Path) -> Dict[str, object]:
    resolved = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = path.relative_to(resolved).as_posix()
        digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
        count += 1
        total_bytes += int(stat.st_size)
    return {
        "root": str(resolved),
        "files": count,
        "bytes": total_bytes,
        "path_size_mtime_sha256": digest.hexdigest(),
    }


def apply_frozen_condition(
    *,
    name: str,
    extraction: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    states: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    readout_states: Mapping[str, object],
    clean_analysis: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    outputs = evaluate_oof_heads(
        features=np.asarray(extraction["features"]),
        valid_masks=np.asarray(extraction["valid_masks"], dtype=np.bool_),
        folds=folds,
        states=states,
        device=device,
    )
    descriptors = build_role_descriptors(
        outputs,
        np.asarray(extraction["features"]),
        np.asarray(extraction["valid_masks"], dtype=np.bool_),
        np.asarray(extraction["bbox_masks"], dtype=np.bool_),
        np.asarray(extraction["probabilities"], dtype=np.float32),
    )
    scores, actions = apply_readout_states(descriptors, folds, readout_states)
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    keeper_predictions = np.asarray(extraction["probabilities"]).argmax(axis=1)
    metrics = {
        role: role_metrics(
            scores=scores[role],
            actions=actions[role],
            targets=targets,
            keeper_predictions=keeper_predictions,
            folds=folds,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics[CANDIDATE_ROLE]
    clean_candidate = clean_analysis["roles"][CANDIDATE_ROLE]
    controls = (*TRAINED_ROLES[1:], ROLLED_ROLE, CYCLED_ROLE)
    checks = {
        "candidate_auroc_ge_0_80": candidate["auroc"] >= 0.80,
        "keeper_tp_retention_ge_0_90": candidate["keeper_tp_retention"] >= 0.90,
        "restricted_fp_rejection_ge_0_18": candidate[
            "restricted_fp_rejection"
        ]
        >= 0.18,
        "accepted_precision_ge_0_70": candidate["precision"] >= 0.70,
        "clean_precision_direction_not_reversed": all(
            candidate["precision"] >= metrics[role]["precision"]
            for role in TRAINED_ROLES[1:]
        )
        and candidate["precision"] - max(
            metrics[role]["precision"] for role in TRAINED_ROLES[1:]
        )
        >= min(
            0.0,
            float(clean_candidate["precision"])
            - max(
                float(clean_analysis["roles"][role]["precision"])
                for role in TRAINED_ROLES[1:]
            ),
        ),
    }
    return {
        "condition": name,
        "passed_base_gates": all(checks.values()),
        "checks": checks,
        "roles": metrics,
        "candidate_beats_controls": {
            role: candidate["auroc"] > metrics[role]["auroc"] for role in controls
        },
    }


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    equations = engineering_checks()
    if not bool(equations["passed"]):
        raise RuntimeError(f"Attention-MaxSep equation checks failed: {equations}")
    rows = locked_training_rows(_read_clean_train_rows(args.cidt_predictions))
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
            "class_counts": list(EXPECTED_TRAIN_CLASS_COUNTS),
            "holdout_counts": list(EXPECTED_HOLDOUT_COUNTS),
            "ordered_index_sha256": EXPECTED_TRAIN_INDEX_SHA256,
        },
        "cohort": {
            "rows": len(cohort),
            "ordered_index_sha256": EXPECTED_COHORT_INDEX_SHA256,
        },
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "implementation_observed_not_gated_until_formal": implementation_tracking_state(),
    }


def engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for Attention-MaxSep engineering forward")
    provenance = verify_locked_inputs(args)
    device = torch.device("cuda")
    model, checkpoint, _ = load_model(args.checkpoint, device)
    rows = locked_training_rows(_read_clean_train_rows(args.cidt_predictions))
    cohort = locked_cohort(rows)
    selected = [
        next(
            row
            for row in cohort
            if row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
        ),
        next(row for row in cohort if row.target != FOCUS_CLASS),
    ]
    dataset, transform, dataset_summary = _build_dataset(checkpoint, rows, args.data)
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in selected],
        brightness=1.0,
        contrast=1.0,
        batch_size=2,
        num_workers=0,
        context="attention_maxsep_prototype_a0_engineering",
    )
    cleanup: Dict[str, object] = {"deleted": False}
    with tempfile.TemporaryDirectory(prefix="trkh_attention_maxsep_engineering_") as temporary:
        extraction = extract_condition(
            model=model,
            loader=loader,
            loader_summary=loader_summary,
            rows=selected,
            device=device,
            condition="engineering",
            feature_cache_path=Path(temporary) / "features.npy",
            benchmark_ordinary=True,
        )
        features = torch.from_numpy(
            np.asarray(extraction["features"], dtype=np.float32)
        ).to(device)
        valid = torch.from_numpy(
            np.asarray(extraction["valid_masks"], dtype=np.bool_)
        ).to(device)
        targets = torch.tensor([row.target for row in selected], device=device)
        heads = {role: head.to(device) for role, head in build_matched_heads(fold=0).items()}
        dephased = channel_dephase(features, dephase_gather_indices(0, device=device))
        actual_checks = {}
        for role, head in heads.items():
            cross_sample, include_separation = _role_loss_settings(role)
            role_features = dephased if role == DEPHASED_ROLE else features
            total, outputs = maxsep_total_loss(
                head,
                role_features,
                valid,
                targets,
                cross_sample=cross_sample,
                include_separation=include_separation,
                include_regularization=True,
            )
            total.backward()
            gradients = _gradient_summary(head)
            actual_checks[role] = {
                "loss_finite": bool(torch.isfinite(total)),
                "outputs_finite": all(
                    bool(torch.isfinite(outputs[key]).all())
                    for key in (
                        "attention_logits",
                        "prototype_logits",
                        "distances",
                        "similarities",
                    )
                ),
                "all_gradient_families_live": len(gradients) == 5
                and all(
                    value["finite"] and value["nonzero"]
                    for value in gradients.values()
                ),
            }
        cleanup = _close_delete_feature_cache(extraction)
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
        "provenance_verified": bool(provenance),
        "dataset": dataset_summary,
        "extraction_checks": extraction["checks"],
        "actual_head_checks": actual_checks,
        "capture_probability_max_abs_error": extraction[
            "capture_probability_max_abs_error"
        ],
        "cidt_probability_max_abs_error": extraction[
            "cidt_probability_max_abs_error"
        ],
        "temporary_cache_cleanup": cleanup,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_ATTENTION_MAXSEP_A0_PREFLIGHT") != "passed":
        raise RuntimeError(
            "Formal Attention-MaxSep A0 must use the locked PowerShell preflight"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal Attention-MaxSep A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    implementation = implementation_tracking_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            f"Formal Attention-MaxSep A0 requires clean pushed tracked state: {repo_state}"
        )
    if not bool(implementation["passed"]):
        raise ValueError(
            "Formal Attention-MaxSep A0 requires committed implementation: "
            f"{implementation}"
        )
    if not bool(provenance["protected_untracked"]["passed"]):
        raise ValueError("Protected user payload state differs before formal audit")
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    equations = engineering_checks()
    if not bool(equations["passed"]):
        raise RuntimeError("Attention-MaxSep equations changed before formal execution")
    device = torch.device("cuda")
    gpu_before = _gpu_snapshot()
    dataset_before = _dataset_stat_snapshot(args.data.parent)
    model, checkpoint, class_names = load_model(args.checkpoint, device)
    rows = locked_training_rows(_read_clean_train_rows(args.cidt_predictions))
    cohort = locked_cohort(rows)
    dataset, transform, dataset_summary = _build_dataset(checkpoint, rows, args.data)
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in rows],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="attention_maxsep_prototype_a0_clean",
    )
    extraction: Optional[Dict[str, object]] = None
    cleanup: Dict[str, object] = {"deleted": False}
    robustness_results: Dict[str, object] = {}
    robustness_passed = False
    try:
        extraction = extract_condition(
            model=model,
            loader=loader,
            loader_summary=loader_summary,
            rows=rows,
            device=device,
            condition="clean",
            feature_cache_path=output_dir / "_temporary_train_block2_features.npy",
            benchmark_ordinary=True,
        )
        oof = train_oof_heads(
            features=np.asarray(extraction["features"]),
            valid_masks=np.asarray(extraction["valid_masks"], dtype=np.bool_),
            targets=np.asarray(extraction["targets"], dtype=np.int64),
            sample_indices=np.asarray(extraction["sample_indices"], dtype=np.int64),
            row_folds=np.asarray([row.fold for row in rows], dtype=np.int64),
            source_stems=[row.source_stem for row in rows],
            device=device,
        )
        cohort_artifacts = build_cohort_artifacts(
            extraction=extraction,
            cohort=cohort,
            learned_outputs=oof["outputs"],
        )
        labels = (
            np.asarray(cohort_artifacts["targets"]) == FOCUS_CLASS
        ).astype(np.int64)
        scores, actions, readout_states = fit_oof_readouts(
            cohort_artifacts["descriptors"],
            labels,
            np.asarray(cohort_artifacts["folds"], dtype=np.int64),
            [row.source_stem for row in cohort],
        )
        diagnostics = prototype_diagnostics(
            features=np.asarray(cohort_artifacts["features"]),
            valid_masks=np.asarray(cohort_artifacts["valid_masks"], dtype=np.bool_),
            bbox_masks=np.asarray(cohort_artifacts["bbox_masks"], dtype=np.bool_),
            targets=np.asarray(cohort_artifacts["targets"], dtype=np.int64),
            folds=np.asarray(cohort_artifacts["folds"], dtype=np.int64),
            states=oof["states"],
            device=device,
        )
        clean_analysis = build_clean_analysis(
            scores=scores,
            actions=actions,
            targets=np.asarray(cohort_artifacts["targets"], dtype=np.int64),
            keeper_predictions=np.asarray(
                cohort_artifacts["keeper_predictions"], dtype=np.int64
            ),
            folds=np.asarray(cohort_artifacts["folds"], dtype=np.int64),
            diagnostics=diagnostics,
        )
        xai = render_xai_contact_sheet(
            path=output_dir / "attention_maxsep_xai_contact_sheet.png",
            model=model,
            dataset=dataset,
            transform=transform,
            semantics=dataset_summary["semantics"],
            cohort=cohort,
            states=oof["states"],
            readout_states=readout_states,
            persisted_scores=scores,
            device=device,
        )
        numeric = numeric_precision_checks(
            cohort_artifacts=cohort_artifacts,
            states=oof["states"],
            readout_states=readout_states,
            device=device,
        )
        export = static_export_checks(
            output_dir=output_dir,
            cohort_artifacts=cohort_artifacts,
            states=oof["states"],
            readout_states=readout_states,
        )
        resource = benchmark_candidate_head(
            cohort_artifacts=cohort_artifacts,
            states=oof["states"],
            device=device,
        )
        artifacts = save_replay_artifacts(
            output_dir=output_dir,
            cohort=cohort,
            cohort_artifacts=cohort_artifacts,
            states=oof["states"],
            readout_states=readout_states,
            scores=scores,
            actions=actions,
        )
        in_process_replay = replay_artifacts(
            output_dir, expected_analysis=clean_analysis
        )
        cleanup = _close_delete_feature_cache(extraction)
        loader_record = extraction["runtime"]["loader"]
        expected_families = {
            "projector_first",
            "projector_second",
            "class_agnostic",
            "class_specific",
            "prototypes",
        }
        structural_gates = {
            "equation_checks_passed": bool(equations["passed"]),
            "locked_extraction_checks_passed": all(
                extraction["checks"].values()
            ),
            "source_disjoint_folds": all(
                not oof["fold_records"][fold]["source_overlap"] for fold in FOLDS
            ),
            "matched_initial_states_exact": all(
                len(
                    set(
                        oof["fold_records"][fold]["initial_state_sha256"].values()
                    )
                )
                == 1
                for fold in FOLDS
            ),
            "shared_occurrence_order_across_roles": bool(
                oof["shared_occurrence_order_across_roles"]
            ),
            "all_joint_gradients_live": all(
                set(
                    oof["fold_records"][fold]["first_joint_gradients"][role]
                )
                == expected_families
                and all(
                    value["finite"] and value["nonzero"]
                    for value in oof["fold_records"][fold][
                        "first_joint_gradients"
                    ][role].values()
                )
                for fold in FOLDS
                for role in TRAINED_ROLES
            ),
            "all_parameter_families_changed": all(
                set(oof["fold_records"][fold]["parameter_changes"][role])
                == expected_families
                and all(
                    value["changed"]
                    for value in oof["fold_records"][fold]["parameter_changes"][
                        role
                    ].values()
                )
                for fold in FOLDS
                for role in TRAINED_ROLES
            ),
            "optimizer_and_parameters_fp32_finite": all(
                oof["fold_records"][fold]["optimizer_checks"][role][
                    "parameter_dtypes"
                ]
                == ["torch.float32"]
                and oof["fold_records"][fold]["optimizer_checks"][role][
                    "optimizer_state_dtypes"
                ]
                == ["torch.float32"]
                and bool(
                    oof["fold_records"][fold]["optimizer_checks"][role][
                        "parameters_finite"
                    ]
                )
                and bool(
                    oof["fold_records"][fold]["optimizer_checks"][role][
                        "optimizer_state_finite"
                    ]
                )
                and bool(
                    oof["fold_records"][fold]["optimizer_checks"][role][
                        "fixed_classifier_exact"
                    ]
                )
                for fold in FOLDS
                for role in TRAINED_ROLES
            ),
            "readouts_converged": all(
                bool(record["converged"])
                for role in ROLE_NAMES
                for record in readout_states[role]["folds"]
            ),
            "requested_effective_workers_exact_4": int(
                loader_record["requested_num_workers"]
            )
            == NUM_WORKERS
            and int(loader_record["effective_num_workers"]) == NUM_WORKERS,
            "numeric_precision_passed": bool(numeric["passed"]),
            "static_export_passed": bool(export["passed"]),
            "resource_gates_passed": bool(resource["passed"]),
            "xai_automatic_passed": bool(xai["automatic_passed"]),
            "in_process_replay_passed": bool(in_process_replay["passed"]),
            "temporary_full_cache_deleted": bool(cleanup["deleted"]),
            "protected_user_payloads_exact": bool(
                provenance["protected_untracked"]["passed"]
            ),
            "implementation_committed_and_clean": bool(implementation["passed"]),
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
                    context=f"attention_maxsep_prototype_a0_{name}",
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
                        condition=name,
                        feature_cache_path=output_dir
                        / f"_temporary_{name}_features.npy",
                        benchmark_ordinary=False,
                    )
                    condition_result = apply_frozen_condition(
                        name=name,
                        extraction=condition_extraction,
                        cohort=cohort,
                        states=oof["states"],
                        readout_states=readout_states,
                        clean_analysis=clean_analysis,
                        device=device,
                    )
                finally:
                    if condition_extraction is not None:
                        condition_cleanup = _close_delete_feature_cache(
                            condition_extraction
                        )
                condition_result["temporary_cache_cleanup"] = condition_cleanup
                condition_result["passed_base_gates"] = bool(
                    condition_result["passed_base_gates"]
                    and condition_cleanup["deleted"]
                )
                robustness_results[name] = condition_result
            control_wins = {
                role: sum(
                    bool(result["candidate_beats_controls"][role])
                    for result in robustness_results.values()
                )
                for role in (*TRAINED_ROLES[1:], ROLLED_ROLE, CYCLED_ROLE)
            }
            robustness_passed = bool(
                all(result["passed_base_gates"] for result in robustness_results.values())
                and all(value >= 2 for value in control_wins.values())
            )
            robustness_results["aggregate"] = {
                "candidate_control_condition_wins": control_wins,
                "all_controls_beaten_in_at_least_2_of_3": all(
                    value >= 2 for value in control_wins.values()
                ),
            }
        else:
            robustness_results = {
                "skipped": True,
                "reason": "clean conjunctive gate failed",
            }
        extraction_summary = {
            "checks": dict(extraction["checks"]),
            "runtime": dict(extraction["runtime"]),
            "capture_probability_max_abs_error": extraction[
                "capture_probability_max_abs_error"
            ],
            "cidt_probability_max_abs_error": extraction[
                "cidt_probability_max_abs_error"
            ],
            "cache": dict(extraction["cache"]),
            "temporary_cache_cleanup": cleanup,
            "persisted": False,
        }
    finally:
        if extraction is not None and "features" in extraction:
            cleanup = _close_delete_feature_cache(extraction)
    if not cleanup.get("deleted", False):
        raise RuntimeError(f"Temporary Attention-MaxSep cache cleanup failed: {cleanup}")
    dataset_after = _dataset_stat_snapshot(args.data.parent)
    dataset_unchanged = dataset_before == dataset_after
    if not dataset_unchanged:
        raise RuntimeError("Raw dataset stat snapshot changed during Attention-MaxSep A0")
    gpu_after = _gpu_snapshot()
    automated_pre_replay = bool(clean_gate_passed and robustness_passed)
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
        "implementation_tracking": implementation,
        "provenance": provenance,
        "dataset": {
            **dataset_summary,
            "stat_before": dataset_before,
            "stat_after": dataset_after,
            "unchanged": dataset_unchanged,
        },
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
        "clean_extraction": extraction_summary,
        "training": {
            "roles": list(TRAINED_ROLES),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "warm_learning_rate": WARM_LEARNING_RATE,
            "joint_learning_rate": JOINT_LEARNING_RATE,
            "joint_decayed_learning_rate": JOINT_DECAYED_LEARNING_RATE,
            "cluster_weight": CLUSTER_WEIGHT,
            "separation_weight": SEPARATION_WEIGHT,
            "seconds": oof["training_seconds"],
            "fold_records": oof["fold_records"],
        },
        "clean_analysis": clean_analysis,
        "structural_gates": structural_gates,
        "structural_gates_passed": all(structural_gates.values()),
        "clean_gate_passed": clean_gate_passed,
        "numeric_precision": numeric,
        "static_export": export,
        "resource": {
            **resource,
            "requested_workers": NUM_WORKERS,
            "effective_loader": loader_record,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "unknown_process_terminated": False,
        },
        "in_process_replay": {
            key: value for key, value in in_process_replay.items() if key != "analysis"
        },
        "robustness_authorized_by_clean_gate": clean_gate_passed,
        "robustness_results": robustness_results,
        "robustness_passed": robustness_passed,
        "xai": {
            "automatic": xai,
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
    summary_path = output_dir / "summary.json"
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
        raise ValueError("Replay target is not Attention-MaxSep Prototype A0")
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
        "states_exact": replay["states_exact"],
        "output_max_abs_error": replay["output_max_abs_error"],
        "descriptor_max_abs_error": replay["descriptor_max_abs_error"],
        "score_max_abs_error": replay["score_max_abs_error"],
        "threshold_max_abs_error": replay["threshold_max_abs_error"],
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
            f"Visual-review summary SHA differs: {observed_sha} != {expected_summary_sha256}"
        )
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Visual-review target is not Attention-MaxSep Prototype A0")
    if not bool(summary.get("external_replay", {}).get("completed", False)):
        raise ValueError("Visual review requires completed external replay")
    xai = summary["xai"]["automatic"]
    if _sha256(Path(xai["path"])) != xai["sha256"]:
        raise ValueError("Attention-MaxSep XAI sheet changed before visual review")
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

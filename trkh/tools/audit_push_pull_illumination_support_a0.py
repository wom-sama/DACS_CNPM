from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Dict, Mapping, Optional, Sequence
import warnings

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import ConvStemBlock, HybridConvStem
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_pixel_difference_stem_signal import (
    _build_dataset,
    _canonical_equal,
    _full_worktree_clean,
    _git_value,
    _load_keeper_model,
    _make_condition_loader,
    _prepare_output_dir,
    _sha256,
    _tracked_worktree_clean,
    _verify_sha256,
    region_masks,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "push_pull_illumination_support_a0"
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42
BENCHMARK_BATCH_SIZE = 32
BENCHMARK_WARMUPS = 5
BENCHMARK_REPEATS = 3
PUSH_SCALE = 2.0
INHIBITION_STRENGTH = 1.0
MAX_RUNTIME_RATIO = 1.20
MAX_MEMORY_RATIO = 1.10
MAX_ONNX_ERROR = 1e-5
MAX_EQUATION_ERROR = 1e-6
MAX_FINITE_DIFFERENCE_ERROR = 1e-4
MAX_BF16_ERROR = 0.02
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FIT_FOLDS = (1, 2, 3, 4)
EXPECTED_COHORT_ROWS = 607
EXPECTED_POSITIVES = 421
EXPECTED_NEGATIVES = 186
EXPECTED_FOLD_COUNTS = {
    1: {"tp": 112, "fp": 45},
    2: {"tp": 100, "fp": 48},
    3: {"tp": 101, "fp": 52},
    4: {"tp": 108, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd"
)
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
ROLES = ("native_identity_h1", "push_pull_h2")
STAGE_NAMES = ("operator", "block")
REGION_NAMES = ("core", "boundary", "outside")
STAGE_FEATURE_SUFFIXES = (
    "core_signed_mean",
    "core_mean_abs",
    "core_rms",
    "boundary_signed_mean",
    "boundary_mean_abs",
    "boundary_rms",
    "outside_signed_mean",
    "outside_mean_abs",
    "outside_rms",
    "boundary_core_energy_contrast",
    "core_outside_energy_contrast",
)
FEATURE_NAMES = tuple(
    f"{stage}_{suffix}"
    for stage in STAGE_NAMES
    for suffix in STAGE_FEATURE_SUFFIXES
)
OBJECT_FEATURE_INDICES = tuple(
    stage * len(STAGE_FEATURE_SUFFIXES) + offset
    for stage in range(len(STAGE_NAMES))
    for offset in (0, 1, 2, 3, 4, 5, 9)
)
MECHANISM_NAMES = (
    "push_object_rms",
    "pull_object_rms",
    "inhibited_object_rms",
    "push_boundary_rms",
    "pull_boundary_rms",
    "inhibited_boundary_rms",
    "push_outside_rms",
    "pull_outside_rms",
    "inhibited_outside_rms",
    "object_pull_to_push_ratio",
)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "111e648ad6c4a37e8839794a61f18055890ca512bd2ec42d5813a9d87ed8c7ae"
LOCKED_PAPER_SHA256 = "04cf18f3857a7ddede02cc83851b5e99b0f25c5e9c4e36793bbe34bd6ebb1580"
LOCKED_OFFICIAL_COMMIT = "c340f329368e60c888ebdca8e30c373cdad9d29f"
LOCKED_OFFICIAL_TREE = "e9c984b6a71dd2f67aeb56388b9e8e84f1d5e5b7"
LOCKED_OFFICIAL_HASHES = {
    "official_module": "1de24434c557967f8c2a58a905e199f65757152ba23483481d60483000235b6a",
    "official_readme": "f7e394b176f16fb7a759ad4d96f960b30919d9a1c0bfc70832623b61e93ea4b7",
    "official_license": "0d4cced2ae017bd8484f69877b72d45114c2c4990689458dc21d1087e31c0564",
}
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked fit-only push-pull illumination-support gate. "
            "Holdout, validation, test, and training are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_PUSH_PULL_ILLUMINATION_SUPPORT_A0_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\push-pull-cnn-layer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_push_pull_illumination_support_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--benchmark-repeats", type=int, default=BENCHMARK_REPEATS
    )
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
        and int(args.seed) == SEED
        and int(args.benchmark_repeats) == BENCHMARK_REPEATS
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "official_root": official,
        "paper": official
        / "papers"
        / "Strisciuglio2020_Article_EnhancedRobustnessOfConvolutio.pdf",
        "official_module": official / "pushpull" / "PPmodule2d.py",
        "official_readme": official / "README.md",
        "official_license": official / "LICENSE",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _cohort_label(row: CleanTrainRow) -> Optional[str]:
    if row.fold not in FIT_FOLDS or row.keeper_prediction != FOCUS_CLASS:
        return None
    if row.target == FOCUS_CLASS:
        return "tp"
    if row.target in RESTRICTED_NEGATIVE_CLASSES:
        return "fp"
    return None


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], list[CleanTrainRow]]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted push-pull A0 protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"
        ),
        "launcher_args": _verify_sha256(
            paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(
            paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "push-pull protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "push-pull paper"
        ),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
    }
    for name, expected in LOCKED_OFFICIAL_HASHES.items():
        hashes[name] = _verify_sha256(
            paths[name], expected, f"official push-pull {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(paths["official_root"], "rev-parse", "HEAD^{tree}")
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official push-pull commit differs: {official_commit} != {LOCKED_OFFICIAL_COMMIT}"
        )
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(
            f"Official push-pull tree differs: {official_tree} != {LOCKED_OFFICIAL_TREE}"
        )
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official push-pull worktree must be clean.")

    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test-data use.")
    if bool(
        cidt_summary.get(
            "validation_predictions_used",
            cidt_summary.get("validation_data_used", True),
        )
    ):
        raise ValueError("CIDT provenance indicates validation-data use.")

    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohort = [row for row in rows if _cohort_label(row) is not None]
    if len(cohort) != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"Locked push-pull cohort differs: {len(cohort)} != {EXPECTED_COHORT_ROWS}"
        )
    positives = sum(_cohort_label(row) == "tp" for row in cohort)
    negatives = sum(_cohort_label(row) == "fp" for row in cohort)
    if (positives, negatives) != (EXPECTED_POSITIVES, EXPECTED_NEGATIVES):
        raise ValueError(
            "Locked push-pull class counts differ: "
            f"{positives}/{negatives} != {EXPECTED_POSITIVES}/{EXPECTED_NEGATIVES}"
        )
    fold_counts: Dict[int, Dict[str, int]] = {
        fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS
    }
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked push-pull fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for formal push-pull A0.")
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal push-pull A0 requires the exact pushed commit.")

    return (
        {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "official_worktree_clean": True,
            "repository_commit": repository_commit,
            "upstream_commit": upstream_commit,
            "tracked_worktree_clean": True,
            "ordered_cohort_index_sha256": ordered_hash,
            "cohort_rows": len(cohort),
            "positive_rows": positives,
            "negative_rows": negatives,
            "fold_counts": fold_counts,
            "validation_predictions_used": False,
            "holdout_data_used": False,
            "test_data_used": False,
            "training_used": False,
        },
        rows,
        cohort,
    )


def pull_kernel_size(push_size: int, scale: float) -> int:
    if int(push_size) <= 0 or float(scale) < 1.0:
        raise ValueError("Push size must be positive and scale must be at least one.")
    scaled = int(math.floor(int(push_size) * float(scale)))
    return scaled + 1 - (scaled % 2)


def upsample_pull_weight(weight: Tensor, scale: float) -> Tensor:
    if weight.ndim != 4 or weight.size(-1) != weight.size(-2):
        raise ValueError("Push weight must be a square OIHW tensor.")
    push_size = int(weight.size(-1))
    pull_size = pull_kernel_size(push_size, scale)
    if pull_size == push_size:
        return weight
    return F.interpolate(
        weight,
        size=(pull_size, pull_size),
        mode="bilinear",
        align_corners=True,
    )


def independent_push_pull_oracle(
    images: Tensor,
    weight: Tensor,
    *,
    scale: float = PUSH_SCALE,
    alpha: float = INHIBITION_STRENGTH,
    stride: int = 1,
    padding: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    pull_weight = upsample_pull_weight(weight, scale)
    push = torch.clamp_min(
        F.conv2d(images, weight, stride=stride, padding=padding), 0.0
    )
    pull_padding = int(pull_weight.size(-1) // 2 - weight.size(-1) // 2 + padding)
    pull = torch.clamp_min(
        F.conv2d(images, -pull_weight, stride=stride, padding=pull_padding), 0.0
    )
    return push, pull, push - float(alpha) * pull


class PushPullConv2d(nn.Module):
    def __init__(
        self,
        source: nn.Conv2d,
        *,
        scale: float = PUSH_SCALE,
        alpha: float = INHIBITION_STRENGTH,
    ) -> None:
        super().__init__()
        if source.bias is not None:
            raise ValueError("Locked TRKH push-pull adaptation requires bias=False.")
        if source.padding_mode != "zeros":
            raise ValueError("Locked push-pull adaptation requires zero padding.")
        if source.kernel_size[0] != source.kernel_size[1]:
            raise ValueError("Locked push-pull adaptation requires square kernels.")
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.stride = tuple(int(value) for value in source.stride)
        self.padding = tuple(int(value) for value in source.padding)
        self.dilation = tuple(int(value) for value in source.dilation)
        self.groups = int(source.groups)
        self.scale = float(scale)
        self.alpha = float(alpha)
        self.push_size = int(source.kernel_size[0])
        self.pull_size = pull_kernel_size(self.push_size, self.scale)
        self.pull_padding = tuple(
            int(self.pull_size // 2 - self.push_size // 2 + value)
            for value in self.padding
        )

    def components(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pull_weight = upsample_pull_weight(self.weight, self.scale)
        linear_push = F.conv2d(
            images,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        push = F.relu(
            linear_push
        )
        pull = F.relu(
            F.conv2d(
                images,
                -pull_weight,
                None,
                self.stride,
                self.pull_padding,
                self.dilation,
                self.groups,
            )
        )
        inhibited = (
            linear_push
            if self.pull_size == self.push_size and self.alpha == 1.0
            else push - self.alpha * pull
        )
        return push, pull, inhibited

    def forward(self, images: Tensor) -> Tensor:
        return self.components(images)[2]


def _first_block(model: nn.Module) -> ConvStemBlock:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, HybridConvStem):
        raise TypeError("Keeper does not expose the locked HybridConvStem.")
    block = stem.blocks[0]
    if not isinstance(block, ConvStemBlock):
        raise TypeError("Keeper first stem block is not ConvStemBlock.")
    convolution = block.block.conv
    if not isinstance(convolution, nn.Conv2d):
        raise TypeError("Keeper first stem operation is not Conv2d.")
    if (
        convolution.in_channels != 3
        or convolution.out_channels != 32
        or convolution.kernel_size != (3, 3)
        or convolution.stride != (1, 1)
        or convolution.padding != (1, 1)
        or convolution.bias is not None
    ):
        raise ValueError("Keeper first convolution differs from the locked RGB 3x3 case.")
    return block


class FrozenFirstBlockTrace(nn.Module):
    def __init__(self, source: ConvStemBlock, *, push_pull: bool) -> None:
        super().__init__()
        convolution = source.block.conv
        if not isinstance(convolution, nn.Conv2d):
            raise TypeError("First block source convolution is invalid.")
        self.operator: nn.Module = (
            PushPullConv2d(convolution, scale=PUSH_SCALE, alpha=INHIBITION_STRENGTH)
            if push_pull
            else copy.deepcopy(convolution)
        )
        self.norm = copy.deepcopy(source.block.norm)
        self.act = copy.deepcopy(source.block.act)
        self.pool = copy.deepcopy(source.block.pool)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward_components(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if isinstance(self.operator, PushPullConv2d):
            push, pull, pre_norm = self.operator.components(images)
        else:
            pre_norm = self.operator(images)
            push = None
            pull = None
        block = self.pool(self.act(self.norm(pre_norm)))
        return pre_norm, block, push, pull

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_components(images)[1]


def _replace_first_conv(model: nn.Module, *, scale: float) -> nn.Module:
    candidate = copy.deepcopy(model)
    block = _first_block(candidate)
    source = block.block.conv
    if not isinstance(source, nn.Conv2d):
        raise TypeError("Candidate source convolution is invalid.")
    block.block.conv = PushPullConv2d(
        source,
        scale=scale,
        alpha=INHIBITION_STRENGTH,
    )
    return candidate.eval()


def _load_official_class(path: Path):
    spec = importlib.util.spec_from_file_location("locked_push_pull_official", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official push-pull module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PPmodule2d


def _module_from_weight(weight: Tensor, *, scale: float) -> PushPullConv2d:
    source = nn.Conv2d(
        int(weight.size(1)),
        int(weight.size(0)),
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    ).to(dtype=weight.dtype, device=weight.device)
    with torch.no_grad():
        source.weight.copy_(weight)
    return PushPullConv2d(source, scale=scale, alpha=INHIBITION_STRENGTH)


def _equation_case(weight: Tensor, official_class) -> Dict[str, object]:
    weight = weight.detach().cpu().float()[:4].contiguous()
    generator = torch.Generator().manual_seed(SEED + int(weight.numel()))
    images = torch.randn(
        2, int(weight.size(1)), 17, 19, generator=generator, dtype=torch.float32
    )
    local = _module_from_weight(weight, scale=PUSH_SCALE).eval()
    official = official_class(
        int(weight.size(1)),
        int(weight.size(0)),
        3,
        stride=1,
        padding=1,
        bias=False,
        alpha=INHIBITION_STRENGTH,
        scale=PUSH_SCALE,
        train_alpha=False,
    ).eval()
    with torch.no_grad():
        official.push.weight.copy_(weight)
        local_output = local(images)
        official_output = official(images)
        _, _, oracle_output = independent_push_pull_oracle(images, weight)

    local_images = images.double().requires_grad_(True)
    local_direct = _module_from_weight(weight.double(), scale=PUSH_SCALE).double()
    local_value = local_direct(local_images)
    local_loss = local_value.square().mean()
    local_input_gradient, local_weight_gradient = torch.autograd.grad(
        local_loss, (local_images, local_direct.weight)
    )

    oracle_images = images.double().requires_grad_(True)
    oracle_weight = weight.double().requires_grad_(True)
    _, _, oracle_value = independent_push_pull_oracle(
        oracle_images, oracle_weight, scale=PUSH_SCALE
    )
    oracle_loss = oracle_value.square().mean()
    oracle_input_gradient, oracle_weight_gradient = torch.autograd.grad(
        oracle_loss, (oracle_images, oracle_weight)
    )

    identity_module = _module_from_weight(weight.double(), scale=1.0).double()
    identity_images = images.double().requires_grad_(True)
    identity_value = identity_module(identity_images)
    native_value = F.conv2d(identity_images, identity_module.weight, padding=1)
    identity_input_gradient, identity_weight_gradient = torch.autograd.grad(
        identity_value.square().mean(),
        (identity_images, identity_module.weight),
        retain_graph=True,
    )
    native_input_gradient, native_weight_gradient = torch.autograd.grad(
        native_value.square().mean(), (identity_images, identity_module.weight)
    )
    return {
        "official_output_max_abs_error": float(
            (local_output - official_output).abs().max()
        ),
        "oracle_output_max_abs_error": float(
            (local_output - oracle_output).abs().max()
        ),
        "oracle_input_gradient_max_abs_error": float(
            (local_input_gradient - oracle_input_gradient).abs().max()
        ),
        "oracle_weight_gradient_max_abs_error": float(
            (local_weight_gradient - oracle_weight_gradient).abs().max()
        ),
        "identity_output_max_abs_error": float(
            (identity_value - native_value).abs().max()
        ),
        "identity_input_gradient_max_abs_error": float(
            (identity_input_gradient - native_input_gradient).abs().max()
        ),
        "identity_weight_gradient_max_abs_error": float(
            (identity_weight_gradient - native_weight_gradient).abs().max()
        ),
        "candidate_identity_max_abs_delta": float(
            (local_output.double() - F.conv2d(images.double(), weight.double(), padding=1))
            .abs()
            .max()
        ),
        "candidate_output_rms": float(local_output.square().mean().sqrt()),
        "all_finite": bool(
            torch.isfinite(local_output).all()
            and torch.isfinite(local_input_gradient).all()
            and torch.isfinite(local_weight_gradient).all()
        ),
    }


def _finite_difference_error() -> float:
    generator = torch.Generator().manual_seed(SEED + 1103)
    images = torch.randn(1, 3, 9, 11, generator=generator, dtype=torch.float64)
    weight = torch.randn(1, 3, 3, 3, generator=generator, dtype=torch.float64)
    module = _module_from_weight(weight, scale=PUSH_SCALE).double()
    loss = module(images).square().mean()
    analytic = torch.autograd.grad(loss, module.weight)[0]
    index = (0, 1, 1, 2)
    epsilon = 1e-5
    values = []
    for direction in (-1.0, 1.0):
        changed = weight.detach().clone()
        changed[index] += direction * epsilon
        changed_module = _module_from_weight(changed, scale=PUSH_SCALE).double()
        values.append(float(changed_module(images).square().mean()))
    numerical = (values[1] - values[0]) / (2.0 * epsilon)
    return abs(float(analytic[index]) - numerical)


def _bf16_diagnostics(weight: Tensor, device: torch.device) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(SEED + 1409)
    images = torch.randn(2, 3, 33, 35, generator=generator, dtype=torch.float32)
    reference_module = _module_from_weight(weight.detach().cpu().float()[:4], scale=PUSH_SCALE)
    with torch.inference_mode():
        reference = reference_module(images).float()
    module = copy.deepcopy(reference_module).to(device=device, dtype=torch.bfloat16)
    candidate_images = images.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    output = module(candidate_images)
    loss = output.float().square().mean()
    input_gradient, weight_gradient = torch.autograd.grad(
        loss, (candidate_images, module.weight)
    )
    observed = output.detach().float().cpu()
    result = {
        "supported": True,
        "output_max_abs_error": float((observed - reference).abs().max()),
        "output_finite": bool(torch.isfinite(output).all()),
        "input_gradient_finite": bool(torch.isfinite(input_gradient).all()),
        "weight_gradient_finite": bool(torch.isfinite(weight_gradient).all()),
        "weight_gradient_nonzero": bool(torch.count_nonzero(weight_gradient)),
    }
    module.cpu()
    del module, candidate_images, output, loss, input_gradient, weight_gradient
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _equation_diagnostics(
    *,
    official_path: Path,
    keeper_weight: Tensor,
    device: torch.device,
) -> Dict[str, object]:
    official_class = _load_official_class(official_path)
    generator = torch.Generator().manual_seed(SEED + 701)
    random_weight = torch.randn(4, 3, 3, 3, generator=generator) * 0.2
    cases = {
        "deterministic_random": _equation_case(random_weight, official_class),
        "real_keeper": _equation_case(keeper_weight, official_class),
    }
    bf16 = _bf16_diagnostics(keeper_weight, device)
    return {
        "cases": cases,
        "finite_difference_error": _finite_difference_error(),
        "bf16": bf16,
        "max_official_output_error": max(
            float(row["official_output_max_abs_error"]) for row in cases.values()
        ),
        "max_oracle_output_error": max(
            float(row["oracle_output_max_abs_error"]) for row in cases.values()
        ),
        "max_input_gradient_error": max(
            float(row["oracle_input_gradient_max_abs_error"])
            for row in cases.values()
        ),
        "max_weight_gradient_error": max(
            float(row["oracle_weight_gradient_max_abs_error"])
            for row in cases.values()
        ),
        "max_identity_output_error": max(
            float(row["identity_output_max_abs_error"]) for row in cases.values()
        ),
        "max_identity_input_gradient_error": max(
            float(row["identity_input_gradient_max_abs_error"])
            for row in cases.values()
        ),
        "max_identity_weight_gradient_error": max(
            float(row["identity_weight_gradient_max_abs_error"])
            for row in cases.values()
        ),
        "all_cases_finite": all(bool(row["all_finite"]) for row in cases.values()),
        "all_cases_nondegenerate": all(
            float(row["candidate_identity_max_abs_delta"]) > 1e-4
            and float(row["candidate_output_rms"]) > 1e-6
            for row in cases.values()
        ),
    }


def _gpu_snapshot() -> Dict[str, object]:
    queries = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,"
            "memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "compute_apps": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    }
    output: Dict[str, object] = {"available": True}
    for name, command in queries.items():
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output[name] = result.stdout.strip()
        except Exception as exc:  # pragma: no cover - environment dependent
            output["available"] = False
            output[f"{name}_error"] = repr(exc)
    return output


class _FullModelBenchmarkWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, image_mask: Tensor, bbox: Tensor) -> Tensor:
        logits, _ = _forward_classification_with_metadata(
            self.model,
            images,
            {"image_mask": image_mask, "bbox": bbox},
            device=images.device,
        )
        if not torch.is_tensor(logits) or logits.ndim != 2:
            raise TypeError("Push-pull benchmark requires classification logits [B,C].")
        return logits


def _benchmark_model(
    model: nn.Module, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    wrapper = _FullModelBenchmarkWrapper(copy.deepcopy(model)).to(device).eval()
    generator = torch.Generator().manual_seed(SEED + 1709)
    images = torch.randn(
        BENCHMARK_BATCH_SIZE,
        3,
        256,
        256,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    image_mask = torch.ones(
        BENCHMARK_BATCH_SIZE, 256, 256, device=device, dtype=torch.bool
    )
    bbox = torch.tensor(
        [0.5, 0.5, 0.70, 0.80], device=device, dtype=torch.float32
    ).view(1, 4).expand(BENCHMARK_BATCH_SIZE, -1).contiguous()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    last_logits: Optional[Tensor] = None
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        for _ in range(BENCHMARK_WARMUPS):
            last_logits = wrapper(images, image_mask, bbox)
        torch.cuda.synchronize(device)
        for _ in range(int(repeats)):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            last_logits = wrapper(images, image_mask, bbox)
            finished.record()
            torch.cuda.synchronize(device)
            elapsed.append(float(started.elapsed_time(finished)))
    if last_logits is None:
        raise RuntimeError("Push-pull benchmark produced no logits.")
    peak = int(torch.cuda.max_memory_allocated(device))
    logits = last_logits.detach().float().cpu()
    wrapper.cpu()
    del wrapper, images, image_mask, bbox, last_logits
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "elapsed_ms": elapsed,
        "median_ms": float(statistics.median(elapsed)),
        "peak_allocated_bytes": peak,
        "batch_size": BENCHMARK_BATCH_SIZE,
        "warmups": BENCHMARK_WARMUPS,
        "repeats": int(repeats),
        "dtype": "bfloat16_autocast",
        "mode": "normal_metadata_aware_evaluator",
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
    }


def _resource_audit(
    *,
    native: nn.Module,
    candidate: nn.Module,
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    before = _gpu_snapshot()
    native_result = _benchmark_model(native, device=device, repeats=repeats)
    middle = _gpu_snapshot()
    candidate_result = _benchmark_model(candidate, device=device, repeats=repeats)
    after = _gpu_snapshot()
    return {
        "gpu_before": before,
        "gpu_between_roles": middle,
        "gpu_after": after,
        "native": native_result,
        "push_pull_h2": candidate_result,
        "runtime_ratio": float(candidate_result["median_ms"])
        / float(native_result["median_ms"]),
        "peak_memory_ratio": float(candidate_result["peak_allocated_bytes"])
        / float(native_result["peak_allocated_bytes"]),
    }


def _onnx_audit(first_block: ConvStemBlock) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    model = FrozenFirstBlockTrace(first_block, push_pull=True).cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 1901)
    example = torch.randn(1, 3, 256, 256, generator=generator)
    with torch.inference_mode():
        reference = model(example).numpy()
    with tempfile.TemporaryDirectory(prefix="trkh_push_pull_onnx_") as directory:
        path = Path(directory) / "push_pull_first_block.onnx"
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=["images"],
            output_names=["first_block"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        observed = session.run(None, {"images": example.numpy()})[0]
        domains = sorted({str(node.domain) for node in graph.graph.node})
        operators = sorted({str(node.op_type) for node in graph.graph.node})
        payload_sha = _sha256(path)
        payload_bytes = int(path.stat().st_size)
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            parse_ok = bool(parser.parse(path.read_bytes()))
            parser_errors = [
                str(parser.get_error(index)) for index in range(parser.num_errors)
            ]
            engine_ok = False
            engine_bytes = 0
            if parse_ok:
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                engine = builder.build_serialized_network(network, config)
                engine_ok = engine is not None
                if engine is not None:
                    engine_bytes = len(bytes(engine))
        except Exception as exc:  # pragma: no cover - environment dependent
            parse_ok = False
            engine_ok = False
            engine_bytes = 0
            parser_errors = [repr(exc)]
    return {
        "opset": 17,
        "domains": domains,
        "operators": operators,
        "standard_domains_only": all(value in {"", "ai.onnx"} for value in domains),
        "output_shape_match": list(reference.shape) == list(observed.shape),
        "max_abs_error": float(np.max(np.abs(reference - observed))),
        "onnx_sha256_ephemeral": payload_sha,
        "onnx_bytes_ephemeral": payload_bytes,
        "onnx_retained": False,
        "tensorrt_parse": parse_ok,
        "tensorrt_errors": parser_errors,
        "tensorrt_engine_build": engine_ok,
        "tensorrt_engine_bytes_ephemeral": engine_bytes,
        "tensorrt_engine_retained": False,
    }


def _structure_audit(
    model: nn.Module,
) -> tuple[ConvStemBlock, nn.Module, nn.Module, Dict[str, object]]:
    first_block = _first_block(model)
    native_parameters = sum(int(value.numel()) for value in model.parameters())
    identity_model = _replace_first_conv(model, scale=1.0)
    candidate_model = _replace_first_conv(model, scale=PUSH_SCALE)
    identity_parameters = sum(
        int(value.numel()) for value in identity_model.parameters()
    )
    candidate_parameters = sum(
        int(value.numel()) for value in candidate_model.parameters()
    )
    candidate_operator = _first_block(candidate_model).block.conv
    if not isinstance(candidate_operator, PushPullConv2d):
        raise TypeError("Candidate first operator is not PushPullConv2d.")
    return first_block, identity_model, candidate_model, {
        "source_model_state_sha256": _state_sha256(model),
        "identity_model_state_sha256": _state_sha256(identity_model),
        "candidate_model_state_sha256": _state_sha256(candidate_model),
        "source_parameter_count": native_parameters,
        "identity_parameter_count": identity_parameters,
        "candidate_parameter_count": candidate_parameters,
        "parameter_counts_equal": native_parameters
        == identity_parameters
        == candidate_parameters,
        "source_kernel_shape": list(first_block.block.conv.weight.shape),
        "candidate_push_kernel_shape": list(candidate_operator.weight.shape),
        "candidate_pull_size": candidate_operator.pull_size,
        "candidate_pull_padding": list(candidate_operator.pull_padding),
        "candidate_scale": candidate_operator.scale,
        "candidate_alpha": candidate_operator.alpha,
        "production_model_mutated": False,
    }


def _collect_logits(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int]]:
    model = model.to(device).eval()
    batches: list[np.ndarray] = []
    indices: list[int] = []
    targets_out: list[int] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                logits, _ = _forward_classification_with_metadata(
                    model, images, metadata, device=device
                )
            if not torch.is_tensor(logits) or logits.ndim != 2 or logits.size(1) != 5:
                raise ValueError("Classification evaluator did not return [B,5] logits.")
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Classification metadata lacks sample_index.")
            batches.append(logits.detach().float().cpu().numpy())
            indices.extend(int(value) for value in sample_indices.tolist())
            targets_out.extend(int(value) for value in targets.tolist())
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0), indices, targets_out


def _full_model_identity_audit(
    *,
    native: nn.Module,
    identity: nn.Module,
    candidate: nn.Module,
    loader,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
) -> Dict[str, object]:
    expected_indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    native_logits, indices, targets = _collect_logits(
        model=native, loader=loader, device=device
    )
    identity_logits, identity_indices, identity_targets = _collect_logits(
        model=identity, loader=loader, device=device
    )
    candidate_logits, candidate_indices, candidate_targets = _collect_logits(
        model=candidate, loader=loader, device=device
    )
    mapping_exact = bool(
        indices
        == identity_indices
        == candidate_indices
        == expected_indices
        and targets
        == identity_targets
        == candidate_targets
        == expected_targets
    )
    native_predictions = native_logits.argmax(axis=1)
    identity_predictions = identity_logits.argmax(axis=1)
    candidate_predictions = candidate_logits.argmax(axis=1)
    declarations_exact = all(
        int(prediction) == row.keeper_prediction
        for prediction, row in zip(native_predictions, cohort)
    )
    unique, counts = np.unique(candidate_predictions, return_counts=True)
    distribution = {str(int(key)): int(value) for key, value in zip(unique, counts)}
    return {
        "rows": len(indices),
        "mapping_exact": mapping_exact,
        "native_declarations_exact": declarations_exact,
        "identity_logit_max_abs_error": float(
            np.max(np.abs(identity_logits - native_logits))
        ),
        "identity_prediction_mismatches": int(
            np.count_nonzero(identity_predictions != native_predictions)
        ),
        "candidate_logits_finite": bool(np.isfinite(candidate_logits).all()),
        "candidate_prediction_distribution": distribution,
        "candidate_unique_predictions": len(distribution),
        "candidate_max_class_share": float(counts.max() / counts.sum()),
        "candidate_prediction_mismatches_vs_native": int(
            np.count_nonzero(candidate_predictions != native_predictions)
        ),
        "candidate_logit_max_abs_delta_vs_native": float(
            np.max(np.abs(candidate_logits - native_logits))
        ),
    }


def _masked_moments(
    activation: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    expanded = mask[:, None].to(dtype=activation.dtype)
    denominator = expanded.sum(dim=(-2, -1)).squeeze(1) * float(activation.size(1))
    signed = (activation * expanded).sum(dim=(1, 2, 3)) / denominator
    mean_abs = (activation.abs() * expanded).sum(dim=(1, 2, 3)) / denominator
    rms = (
        (activation.square() * expanded).sum(dim=(1, 2, 3)) / denominator
    ).clamp_min(0.0).sqrt()
    return signed, mean_abs, rms


def activation_descriptors(activation: Tensor, bboxes: Tensor) -> Tensor:
    core, boundary, outside = region_masks(
        bboxes, int(activation.size(-2)), int(activation.size(-1))
    )
    core_signed, core_abs, core_rms = _masked_moments(activation, core)
    boundary_signed, boundary_abs, boundary_rms = _masked_moments(
        activation, boundary
    )
    outside_signed, outside_abs, outside_rms = _masked_moments(activation, outside)
    epsilon = torch.finfo(activation.dtype).eps
    boundary_core = (boundary_abs - core_abs) / (
        boundary_abs + core_abs + epsilon
    )
    core_outside = (core_abs - outside_abs) / (
        core_abs + outside_abs + epsilon
    )
    return torch.stack(
        (
            core_signed,
            core_abs,
            core_rms,
            boundary_signed,
            boundary_abs,
            boundary_rms,
            outside_signed,
            outside_abs,
            outside_rms,
            boundary_core,
            core_outside,
        ),
        dim=1,
    )


def _masked_rms(activation: Tensor, mask: Tensor) -> Tensor:
    return _masked_moments(activation, mask)[2]


def mechanism_descriptors(
    *,
    push: Tensor,
    pull: Tensor,
    inhibited: Tensor,
    bboxes: Tensor,
) -> Tensor:
    core, boundary, outside = region_masks(
        bboxes, int(inhibited.size(-2)), int(inhibited.size(-1))
    )
    object_mask = core | boundary
    push_object = _masked_rms(push, object_mask)
    pull_object = _masked_rms(pull, object_mask)
    inhibited_object = _masked_rms(inhibited, object_mask)
    push_boundary = _masked_rms(push, boundary)
    pull_boundary = _masked_rms(pull, boundary)
    inhibited_boundary = _masked_rms(inhibited, boundary)
    push_outside = _masked_rms(push, outside)
    pull_outside = _masked_rms(pull, outside)
    inhibited_outside = _masked_rms(inhibited, outside)
    ratio = pull_object / (push_object + pull_object + torch.finfo(push.dtype).eps)
    return torch.stack(
        (
            push_object,
            pull_object,
            inhibited_object,
            push_boundary,
            pull_boundary,
            inhibited_boundary,
            push_outside,
            pull_outside,
            inhibited_outside,
            ratio,
        ),
        dim=1,
    )


def _extract_features(
    *,
    first_block: ConvStemBlock,
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, object],
]:
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    native = FrozenFirstBlockTrace(first_block, push_pull=False).to(device).eval()
    candidate = FrozenFirstBlockTrace(first_block, push_pull=True).to(device).eval()
    features: Dict[str, Dict[str, np.ndarray]] = {}
    mechanism: Dict[str, np.ndarray] = {}
    loader_summaries: Dict[str, object] = {}
    with torch.inference_mode():
        for condition, brightness, contrast in CONDITIONS:
            loader, loader_summary = _make_condition_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=indices,
                brightness=brightness,
                contrast=contrast,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context=f"push_pull_a0_{condition}",
            )
            loader_summaries[condition] = loader_summary
            role_batches: Dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
            mechanism_batches: list[np.ndarray] = []
            observed_indices: list[int] = []
            observed_targets: list[int] = []
            for images, targets, metadata in loader:
                sample_indices = metadata.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bboxes):
                    raise ValueError("Feature loader lacks sample_index/crop_bbox metadata.")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                crop_bboxes = crop_bboxes.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                region_masks(crop_bboxes, int(images.size(-2)), int(images.size(-1)))
                region_masks(
                    crop_bboxes, int(images.size(-2) // 2), int(images.size(-1) // 2)
                )

                native_pre, native_block, _, _ = native.forward_components(images)
                native_descriptor = torch.cat(
                    (
                        activation_descriptors(native_pre.float(), crop_bboxes),
                        activation_descriptors(native_block.float(), crop_bboxes),
                    ),
                    dim=1,
                )
                role_batches["native_identity_h1"].append(
                    native_descriptor.cpu().numpy()
                )
                del native_pre, native_block, native_descriptor

                candidate_pre, candidate_block, push, pull = (
                    candidate.forward_components(images)
                )
                if push is None or pull is None:
                    raise RuntimeError("Push-pull candidate did not expose components.")
                candidate_descriptor = torch.cat(
                    (
                        activation_descriptors(candidate_pre.float(), crop_bboxes),
                        activation_descriptors(candidate_block.float(), crop_bboxes),
                    ),
                    dim=1,
                )
                role_batches["push_pull_h2"].append(
                    candidate_descriptor.cpu().numpy()
                )
                mechanism_batches.append(
                    mechanism_descriptors(
                        push=push.float(),
                        pull=pull.float(),
                        inhibited=candidate_pre.float(),
                        bboxes=crop_bboxes,
                    )
                    .cpu()
                    .numpy()
                )
                del candidate_pre, candidate_block, candidate_descriptor, push, pull
                observed_indices.extend(int(value) for value in sample_indices.tolist())
                observed_targets.extend(int(value) for value in targets.tolist())
            if observed_indices != indices or observed_targets != expected_targets:
                raise ValueError(f"{condition} feature mapping differs from cohort.")
            features[condition] = {
                role: np.concatenate(values, axis=0).astype(np.float64)
                for role, values in role_batches.items()
            }
            mechanism[condition] = np.concatenate(
                mechanism_batches, axis=0
            ).astype(np.float64)
            for role in ROLES:
                matrix = features[condition][role]
                if matrix.shape != (EXPECTED_COHORT_ROWS, len(FEATURE_NAMES)):
                    raise ValueError(
                        f"Unexpected {condition}/{role} feature shape: {matrix.shape}"
                    )
                if not bool(np.isfinite(matrix).all()):
                    raise ValueError(f"Non-finite {condition}/{role} descriptors.")
            if mechanism[condition].shape != (
                EXPECTED_COHORT_ROWS,
                len(MECHANISM_NAMES),
            ) or not bool(np.isfinite(mechanism[condition]).all()):
                raise ValueError(f"Invalid {condition} mechanism descriptors.")
    native.cpu()
    candidate.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return features, mechanism, loader_summaries


def _view_indices(view: str) -> tuple[int, ...]:
    if view == "object":
        return OBJECT_FEATURE_INDICES
    if view == "context":
        return tuple(range(len(FEATURE_NAMES)))
    raise ValueError(f"Unknown descriptor view: {view}")


def _fit_oof_scores(
    *,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    view: str,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    columns = _view_indices(view)
    scores = {
        condition: np.full(labels.size, np.nan, dtype=np.float64)
        for condition, _, _ in CONDITIONS
    }
    diagnostics: list[Dict[str, object]] = []
    for fold in FIT_FOLDS:
        fit_mask = folds != fold
        holdout_mask = folds == fold
        if set(labels[fit_mask].tolist()) != {0, 1} or set(
            labels[holdout_mask].tolist()
        ) != {0, 1}:
            raise ValueError(f"Fold {fold} does not contain both binary classes.")
        scaler = StandardScaler()
        clean_fit = scaler.fit_transform(features["clean"][fit_mask][:, columns])
        classifier = LogisticRegression(
            solver="liblinear",
            C=0.1,
            class_weight="balanced",
            fit_intercept=True,
            max_iter=1000,
            tol=1e-6,
            random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(clean_fit, labels[fit_mask])
        convergence_warnings = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        converged = bool(
            not convergence_warnings and int(classifier.n_iter_[0]) < 1000
        )
        if not converged:
            raise ValueError(f"Locked logistic readout did not converge in fold {fold}.")
        for condition, _, _ in CONDITIONS:
            transformed = scaler.transform(
                features[condition][holdout_mask][:, columns]
            )
            scores[condition][holdout_mask] = classifier.predict_proba(transformed)[
                :, 1
            ]
        diagnostics.append(
            {
                "fold": fold,
                "fit_rows": int(fit_mask.sum()),
                "holdout_rows": int(holdout_mask.sum()),
                "iterations": int(classifier.n_iter_[0]),
                "converged": converged,
                "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
                "intercept": float(classifier.intercept_[0]),
            }
        )
    if not all(bool(np.isfinite(values).all()) for values in scores.values()):
        raise ValueError(f"OOF scores are incomplete for {view} view.")
    return scores, {
        "view": view,
        "feature_count": len(columns),
        "feature_names": [FEATURE_NAMES[index] for index in columns],
        "folds": diagnostics,
    }


def _metrics_from_scores(
    scores: Mapping[str, np.ndarray], labels: np.ndarray
) -> Dict[str, object]:
    clean_tp_scores = np.asarray(scores["clean"])[labels == 1]
    threshold = float(np.quantile(clean_tp_scores, 0.03, method="lower"))
    conditions: Dict[str, object] = {}
    for condition, _, _ in CONDITIONS:
        values = np.asarray(scores[condition], dtype=np.float64)
        tp_values = values[labels == 1]
        fp_values = values[labels == 0]
        conditions[condition] = {
            "auroc": float(roc_auc_score(labels, values)),
            "tp_retention": float(np.mean(tp_values >= threshold)),
            "fp_rejection": float(np.mean(fp_values < threshold)),
            "tp_median_score": float(np.median(tp_values)),
            "fp_median_score": float(np.median(fp_values)),
            "tp_min_score": float(np.min(tp_values)),
            "tp_max_score": float(np.max(tp_values)),
            "fp_min_score": float(np.min(fp_values)),
            "fp_max_score": float(np.max(fp_values)),
        }
    return {"threshold": threshold, "conditions": conditions}


def _run_readouts(
    *,
    features: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    folds: np.ndarray,
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, object],
    Dict[str, object],
]:
    all_scores: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    metrics: Dict[str, object] = {}
    diagnostics: Dict[str, object] = {}
    for role in ROLES:
        role_features = {
            condition: features[condition][role] for condition, _, _ in CONDITIONS
        }
        all_scores[role] = {}
        metrics[role] = {}
        diagnostics[role] = {}
        for view in ("object", "context"):
            scores, readout_diagnostics = _fit_oof_scores(
                features=role_features,
                labels=labels,
                folds=folds,
                view=view,
            )
            all_scores[role][view] = scores
            metrics[role][view] = _metrics_from_scores(scores, labels)
            diagnostics[role][view] = readout_diagnostics
    return all_scores, metrics, diagnostics


def _mechanism_summary(
    mechanism: Mapping[str, np.ndarray], labels: np.ndarray
) -> Dict[str, object]:
    indices = {name: index for index, name in enumerate(MECHANISM_NAMES)}
    conditions: Dict[str, object] = {}
    for condition, _, _ in CONDITIONS:
        matrix = np.asarray(mechanism[condition], dtype=np.float64)
        cohorts: Dict[str, object] = {}
        for name, label in (("tp", 1), ("fp", 0)):
            selected = matrix[labels == label]
            object_energy = float(
                np.median(selected[:, indices["inhibited_object_rms"]])
            )
            outside_energy = float(
                np.median(selected[:, indices["inhibited_outside_rms"]])
            )
            pull_ratio = float(
                np.median(selected[:, indices["object_pull_to_push_ratio"]])
            )
            cohorts[name] = {
                "inhibited_object_rms_median": object_energy,
                "inhibited_outside_rms_median": outside_energy,
                "object_gt_outside": object_energy > outside_energy,
                "object_pull_to_push_ratio_median": pull_ratio,
            }
        conditions[condition] = {
            "cohorts": cohorts,
            "both_cohorts_object_gt_outside": bool(
                cohorts["tp"]["object_gt_outside"]
                and cohorts["fp"]["object_gt_outside"]
            ),
            "fp_minus_tp_pull_ratio": float(
                cohorts["fp"]["object_pull_to_push_ratio_median"]
                - cohorts["tp"]["object_pull_to_push_ratio_median"]
            ),
        }
    shifted = [condition for condition, _, _ in CONDITIONS if condition != "clean"]
    return {
        "conditions": conditions,
        "shifted_object_gt_outside_count": sum(
            bool(conditions[condition]["both_cohorts_object_gt_outside"])
            for condition in shifted
        ),
        "shifted_fp_inhibition_gap_gte_0p01_count": sum(
            float(conditions[condition]["fp_minus_tp_pull_ratio"]) >= 0.01
            for condition in shifted
        ),
    }


def assess_information_gate(
    *,
    metrics: Mapping[str, object],
    mechanism_summary: Mapping[str, object],
) -> Dict[str, object]:
    native = metrics["native_identity_h1"]["object"]["conditions"]
    candidate = metrics["push_pull_h2"]["object"]["conditions"]
    candidate_context = metrics["push_pull_h2"]["context"]["conditions"]
    shifts = [condition for condition, _, _ in CONDITIONS if condition != "clean"]
    clean_delta = float(candidate["clean"]["auroc"]) - float(
        native["clean"]["auroc"]
    )
    shifted_deltas = {
        condition: float(candidate[condition]["auroc"])
        - float(native[condition]["auroc"])
        for condition in shifts
    }
    clean_fp_delta = float(candidate["clean"]["fp_rejection"]) - float(
        native["clean"]["fp_rejection"]
    )
    mechanism_conditions = mechanism_summary["conditions"]
    checks = {
        "clean_object_auroc_gte_0p64": float(candidate["clean"]["auroc"])
        >= 0.64,
        "clean_object_auroc_delta_gte_0p025": clean_delta >= 0.025,
        "minimum_shifted_object_auroc_gte_0p58": min(
            float(candidate[condition]["auroc"]) for condition in shifts
        )
        >= 0.58,
        "candidate_no_worse_than_native_in_two_shifts": sum(
            delta >= 0.0 for delta in shifted_deltas.values()
        )
        >= 2,
        "worst_shifted_auroc_delta_gte_minus_0p01": min(shifted_deltas.values())
        >= -0.01,
        "clean_tp_retention_gte_0p97": float(
            candidate["clean"]["tp_retention"]
        )
        >= 0.97,
        "clean_fp_rejection_gte_0p20": float(
            candidate["clean"]["fp_rejection"]
        )
        >= 0.20,
        "clean_fp_rejection_delta_gte_0p08": clean_fp_delta >= 0.08,
        "all_shifted_tp_retention_gte_0p93": all(
            float(candidate[condition]["tp_retention"]) >= 0.93
            for condition in shifts
        ),
        "all_shifted_tp_retention_within_native_minus_0p01": all(
            float(candidate[condition]["tp_retention"])
            >= float(native[condition]["tp_retention"]) - 0.01
            for condition in shifts
        ),
        "all_shifted_fp_rejection_gte_0p10": all(
            float(candidate[condition]["fp_rejection"]) >= 0.10
            for condition in shifts
        ),
        "aggregate_shifted_fp_rejection_beats_native": sum(
            float(candidate[condition]["fp_rejection"]) for condition in shifts
        )
        > sum(float(native[condition]["fp_rejection"]) for condition in shifts),
        "tp_median_gt_fp_every_condition": all(
            float(candidate[condition]["tp_median_score"])
            > float(candidate[condition]["fp_median_score"])
            for condition, _, _ in CONDITIONS
        ),
        "clean_object_within_0p02_of_context": float(
            candidate["clean"]["auroc"]
        )
        >= float(candidate_context["clean"]["auroc"]) - 0.02,
        "object_candidate_still_beats_native": clean_delta >= 0.025,
        "clean_both_cohorts_object_gt_outside": bool(
            mechanism_conditions["clean"]["both_cohorts_object_gt_outside"]
        ),
        "object_gt_outside_in_two_shifts": int(
            mechanism_summary["shifted_object_gt_outside_count"]
        )
        >= 2,
        "clean_fp_pull_ratio_gap_gte_0p01": float(
            mechanism_conditions["clean"]["fp_minus_tp_pull_ratio"]
        )
        >= 0.01,
        "fp_pull_ratio_gap_gte_0p01_in_two_shifts": int(
            mechanism_summary["shifted_fp_inhibition_gap_gte_0p01_count"]
        )
        >= 2,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "clean_object_auroc_delta": clean_delta,
        "clean_fp_rejection_delta": clean_fp_delta,
        "shifted_object_auroc_deltas": shifted_deltas,
    }


def _write_cohort_csv(
    *,
    path: Path,
    cohort: Sequence[CleanTrainRow],
    features: Mapping[str, Mapping[str, np.ndarray]],
    mechanism: Mapping[str, np.ndarray],
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    fieldnames = [
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        "cohort",
        "condition",
        "role",
        *FEATURE_NAMES,
        *MECHANISM_NAMES,
        "object_score",
        "context_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for condition, _, _ in CONDITIONS:
            for role in ROLES:
                matrix = features[condition][role]
                mechanism_matrix = mechanism[condition]
                for position, source in enumerate(cohort):
                    row: Dict[str, object] = {
                        "sample_index": source.sample_index,
                        "source_stem": source.source_stem,
                        "image_path": str(source.image_path),
                        "fold": source.fold,
                        "target": source.target,
                        "keeper_prediction": source.keeper_prediction,
                        "cohort": str(_cohort_label(source)),
                        "condition": condition,
                        "role": role,
                        "object_score": float(
                            scores[role]["object"][condition][position]
                        ),
                        "context_score": float(
                            scores[role]["context"][condition][position]
                        ),
                    }
                    row.update(
                        {
                            name: float(matrix[position, index])
                            for index, name in enumerate(FEATURE_NAMES)
                        }
                    )
                    row.update(
                        {
                            name: float(mechanism_matrix[position, index])
                            for index, name in enumerate(MECHANISM_NAMES)
                        }
                    )
                    writer.writerow(row)


def _replay_csv(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_index",
            "fold",
            "cohort",
            "condition",
            "role",
            "object_score",
            "context_score",
            *FEATURE_NAMES,
            *MECHANISM_NAMES,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Push-pull replay CSV misses columns: {sorted(missing)}")
        records = [dict(row) for row in reader]
    expected_rows = EXPECTED_COHORT_ROWS * len(CONDITIONS) * len(ROLES)
    if len(records) != expected_rows:
        raise ValueError(
            f"Push-pull replay rows differ: {len(records)} != {expected_rows}"
        )
    clean_native = [
        row
        for row in records
        if row["condition"] == "clean" and row["role"] == ROLES[0]
    ]
    ordered_indices = [int(row["sample_index"]) for row in clean_native]
    if _ordered_index_sha256(ordered_indices) != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError("Push-pull replay ordered cohort hash differs.")
    labels = np.asarray(
        [1 if row["cohort"] == "tp" else 0 for row in clean_native],
        dtype=np.int64,
    )
    metrics: Dict[str, object] = {}
    for role in ROLES:
        metrics[role] = {}
        for view in ("object", "context"):
            score_key = f"{view}_score"
            score_table: Dict[str, np.ndarray] = {}
            for condition, _, _ in CONDITIONS:
                selected = [
                    row
                    for row in records
                    if row["role"] == role and row["condition"] == condition
                ]
                if [int(row["sample_index"]) for row in selected] != ordered_indices:
                    raise ValueError(f"Replay order differs for {role}/{condition}.")
                score_table[condition] = np.asarray(
                    [float(row[score_key]) for row in selected], dtype=np.float64
                )
            metrics[role][view] = _metrics_from_scores(score_table, labels)
    mechanism: Dict[str, np.ndarray] = {}
    for condition, _, _ in CONDITIONS:
        selected = [
            row
            for row in records
            if row["role"] == "push_pull_h2" and row["condition"] == condition
        ]
        mechanism[condition] = np.asarray(
            [[float(row[name]) for name in MECHANISM_NAMES] for row in selected],
            dtype=np.float64,
        )
    mechanism_summary = _mechanism_summary(mechanism, labels)
    information_gate = assess_information_gate(
        metrics=metrics, mechanism_summary=mechanism_summary
    )
    return {
        "rows": len(records),
        "metrics": metrics,
        "mechanism_summary": mechanism_summary,
        "information_gate": information_gate,
    }


def _tensor_to_rgb(image: Tensor, semantics: Mapping[str, object]) -> Tensor:
    mean = torch.tensor(
        semantics["input_mean"], device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        semantics["input_std"], device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def _pil_from_rgb_tensor(image: Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0.0, 255.0)
        .to(dtype=torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _bbox_core_overlay(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    cx, cy, width, height = [float(value) for value in bbox[:4]]
    x1 = max(0.0, cx - width / 2.0) * output.width
    x2 = min(1.0, cx + width / 2.0) * output.width
    y1 = max(0.0, cy - height / 2.0) * output.height
    y2 = min(1.0, cy + height / 2.0) * output.height
    draw.rectangle((x1, y1, x2, y2), outline=(235, 40, 40), width=3)
    core = (
        x1 + 0.20 * (x2 - x1),
        y1 + 0.20 * (y2 - y1),
        x2 - 0.20 * (x2 - x1),
        y2 - 0.20 * (y2 - y1),
    )
    draw.rectangle(core, outline=(30, 220, 80), width=3)
    return output


def _energy_image(energy: Tensor, *, upper: float) -> Image.Image:
    values = energy.detach().float().cpu().numpy()
    normalized = np.clip(values / max(float(upper), 1e-8), 0.0, 1.0)
    red = np.clip(1.7 * normalized, 0.0, 1.0)
    green = np.clip(1.6 - np.abs(normalized - 0.55) * 3.0, 0.0, 1.0)
    blue = np.clip(1.25 * (1.0 - normalized), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(np.rint(rgb * 255.0)), mode="RGB")


def _select_visual_positions(
    *,
    condition: str,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> tuple[list[int], list[str]]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    candidate_scores = scores["push_pull_h2"]["object"][condition]
    positions_out: list[int] = []
    names: list[str] = []
    for cohort_name, label in (("tp", 1), ("fp", 0)):
        positions = np.flatnonzero(labels == label)
        ordered = positions[
            np.argsort(candidate_scores[positions], kind="mergesort")
        ]
        for rank_name, selected in (("lowest", ordered[:4]), ("highest", ordered[-4:])):
            positions_out.extend(int(value) for value in selected)
            names.extend(
                f"{cohort_name}_{rank_name}_{rank + 1}"
                for rank in range(len(selected))
            )
    return positions_out, names


def _render_contact_sheets(
    *,
    output_dir: Path,
    first_block: ConvStemBlock,
    checkpoint: Mapping[str, object],
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    semantics = _eval_semantics(checkpoint)
    candidate = FrozenFirstBlockTrace(first_block, push_pull=True).to(device).eval()
    sheets: list[Dict[str, object]] = []
    font = ImageFont.load_default()
    for condition, brightness, contrast in CONDITIONS:
        positions, selection_names = _select_visual_positions(
            condition=condition, cohort=cohort, scores=scores
        )
        selected_rows = [cohort[position] for position in positions]
        loader, _ = _make_condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=[row.sample_index for row in selected_rows],
            brightness=brightness,
            contrast=contrast,
            batch_size=16,
            num_workers=int(args.num_workers),
            context=f"push_pull_a0_visual_{condition}",
        )
        image_batches: list[Tensor] = []
        bbox_batches: list[Tensor] = []
        push_batches: list[Tensor] = []
        pull_batches: list[Tensor] = []
        inhibited_batches: list[Tensor] = []
        observed_indices: list[int] = []
        with torch.inference_mode():
            for images, _, metadata in loader:
                crop_bboxes = metadata.get("crop_bbox")
                sample_indices = metadata.get("sample_index")
                if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(
                    sample_indices
                ):
                    raise ValueError("Visual loader lacks crop_bbox/sample_index.")
                images_device = images.to(device=device, dtype=torch.float32)
                pre_norm, _, push, pull = candidate.forward_components(images_device)
                if push is None or pull is None:
                    raise RuntimeError("Visual candidate components are missing.")
                image_batches.append(images.float())
                bbox_batches.append(crop_bboxes[:, :4].float())
                push_batches.append(push.square().mean(dim=1).sqrt().cpu())
                pull_batches.append(pull.square().mean(dim=1).sqrt().cpu())
                inhibited_batches.append(pre_norm.square().mean(dim=1).sqrt().cpu())
                observed_indices.extend(int(value) for value in sample_indices.tolist())
        expected_indices = [row.sample_index for row in selected_rows]
        if observed_indices != expected_indices:
            raise ValueError(f"Visual order differs for {condition}.")
        images = torch.cat(image_batches, dim=0)
        bboxes = torch.cat(bbox_batches, dim=0)
        push_energy = torch.cat(push_batches, dim=0)
        pull_energy = torch.cat(pull_batches, dim=0)
        inhibited_energy = torch.cat(inhibited_batches, dim=0)
        rgb = _tensor_to_rgb(images, semantics)
        canvas = Image.new("RGB", (1600, 16 * 196 + 44), color=(248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10),
            f"{condition} | red=bbox green=core | push / pull / inhibited",
            fill=(0, 0, 0),
            font=font,
        )
        for row_index, (position, source, selection) in enumerate(
            zip(positions, selected_rows, selection_names)
        ):
            top = 38 + row_index * 196
            source_image = _bbox_core_overlay(
                _pil_from_rgb_tensor(rgb[row_index]), bboxes[row_index].tolist()
            ).resize((180, 180), Image.Resampling.BILINEAR)
            energy_values = torch.cat(
                (
                    push_energy[row_index].flatten(),
                    pull_energy[row_index].flatten(),
                    inhibited_energy[row_index].flatten(),
                )
            )
            upper = float(torch.quantile(energy_values, 0.98))
            visual_maps = [
                _energy_image(value, upper=upper).resize(
                    (180, 180), Image.Resampling.BILINEAR
                )
                for value in (
                    push_energy[row_index],
                    pull_energy[row_index],
                    inhibited_energy[row_index],
                )
            ]
            canvas.paste(source_image, (12, top))
            for map_index, visual in enumerate(visual_maps):
                canvas.paste(visual, (204 + map_index * 192, top))
            native_score = float(
                scores["native_identity_h1"]["object"][condition][position]
            )
            candidate_score = float(
                scores["push_pull_h2"]["object"][condition][position]
            )
            lines = [
                f"{selection} | sample={source.sample_index} fold={source.fold}",
                f"target={source.target} keeper={source.keeper_prediction} source={source.source_stem}",
                f"native={native_score:.6f} push_pull={candidate_score:.6f}",
                f"shared map upper={upper:.6f}",
                str(source.image_path),
            ]
            draw.multiline_text(
                (790, top + 18),
                "\n".join(lines),
                fill=(10, 10, 10),
                font=font,
                spacing=9,
            )
        path = output_dir / f"push_pull_contact_sheet_{condition}.png"
        canvas.save(path)
        sheets.append(
            {
                "condition": condition,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "selected_rows": len(positions),
                "sample_indices": expected_indices,
                "selections": selection_names,
            }
        )
    candidate.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "sheet_count": len(sheets),
        "selected_rows_per_sheet": [int(row["selected_rows"]) for row in sheets],
        "manual_review_status": "pending",
        "sheets": sheets,
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    metrics = summary.get("readout_metrics")
    lines = [
        "# Push-Pull Illumination Support A0 Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Auto gate eligible: `{summary.get('auto_gate_eligible', False)}`",
        f"- Matched 5e pair authorized: `{summary.get('matched_5e_pair_authorized', False)}`",
        f"- Structural failed checks: `{summary['structural_gate']['failed_checks']}`",
        f"- Information failed checks: `{summary.get('information_gate', {}).get('failed_checks', [])}`",
        "",
    ]
    if isinstance(metrics, Mapping):
        native = metrics["native_identity_h1"]["object"]["conditions"]
        candidate = metrics["push_pull_h2"]["object"]["conditions"]
        lines.extend(
            [
                "## Object-Only OOF Metrics",
                "",
                "| Condition | Native AUROC | Push-pull AUROC | Native FP reject | Push-pull FP reject | Push-pull TP retain |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, _, _ in CONDITIONS:
            lines.append(
                "| {condition} | {native_auc:.6f} | {candidate_auc:.6f} | "
                "{native_fp:.6f} | {candidate_fp:.6f} | {candidate_tp:.6f} |".format(
                    condition=condition,
                    native_auc=float(native[condition]["auroc"]),
                    candidate_auc=float(candidate[condition]["auroc"]),
                    native_fp=float(native[condition]["fp_rejection"]),
                    candidate_fp=float(candidate[condition]["fp_rejection"]),
                    candidate_tp=float(candidate[condition]["tp_retention"]),
                )
            )
        lines.append("")
    resource = summary.get("resource_audit", {})
    onnx = summary.get("onnx_audit", {})
    lines.extend(
        [
            "## Deployment",
            "",
            f"- Runtime ratio: `{resource.get('runtime_ratio')}`",
            f"- Peak-memory ratio: `{resource.get('peak_memory_ratio')}`",
            f"- ONNX max error: `{onnx.get('max_abs_error')}`",
            f"- TensorRT parse/build: `{onnx.get('tensorrt_parse')}/{onnx.get('tensorrt_engine_build')}`",
            "",
            "No holdout, validation, test, image training, or current-command update was used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    artifacts = []
    forbidden = {".pt", ".pth", ".ckpt", ".engine", ".onnx", ".npy", ".npz"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden model artifact in push-pull A0 output: {path}")
        artifacts.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload = {
        "method": METHOD,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        **payload,
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    provenance, rows, cohort = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "preflight_created_output": False,
            "provenance": provenance,
        }

    output_dir = _prepare_output_dir(Path(args.output_dir))
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("Push-pull A0 requires CUDA.")
    set_seed(int(args.seed), deterministic=True)
    checkpoint = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    model = _load_keeper_model(checkpoint).eval()
    first_block, identity_model, candidate_model, structure = _structure_audit(model)
    equation = _equation_diagnostics(
        official_path=Path(provenance["paths"]["official_module"]),
        keeper_weight=first_block.block.conv.weight,
        device=device,
    )
    resource = _resource_audit(
        native=model,
        candidate=candidate_model,
        device=device,
        repeats=int(args.benchmark_repeats),
    )
    onnx = _onnx_audit(first_block)

    base_dataset, transform, dataset_mapping = _build_dataset(
        checkpoint, rows, Path(args.data)
    )
    indices = [row.sample_index for row in cohort]
    declaration_loader, declaration_loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=int(args.num_workers),
        context="push_pull_a0_declaration_identity",
    )
    full_model_identity = _full_model_identity_audit(
        native=model,
        identity=identity_model,
        candidate=candidate_model,
        loader=declaration_loader,
        cohort=cohort,
        device=device,
    )

    features, mechanism, feature_loaders = _extract_features(
        first_block=first_block,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        args=args,
        device=device,
    )
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    scores, readout_metrics, readout_diagnostics = _run_readouts(
        features=features, labels=labels, folds=folds
    )
    mechanism_summary = _mechanism_summary(mechanism, labels)
    information_gate = assess_information_gate(
        metrics=readout_metrics, mechanism_summary=mechanism_summary
    )

    cohort_csv = output_dir / "cohort_scores_features_and_mechanism.csv"
    _write_cohort_csv(
        path=cohort_csv,
        cohort=cohort,
        features=features,
        mechanism=mechanism,
        scores=scores,
    )
    replay = _replay_csv(cohort_csv)
    replay_metrics_exact = _canonical_equal(readout_metrics, replay["metrics"])
    replay_mechanism_exact = _canonical_equal(
        mechanism_summary, replay["mechanism_summary"]
    )
    replay_gate_exact = _canonical_equal(
        information_gate, replay["information_gate"]
    )
    contact_sheets = _render_contact_sheets(
        output_dir=output_dir,
        first_block=first_block,
        checkpoint=checkpoint,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        scores=scores,
        args=args,
        device=device,
    )

    structural_checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_commit_tree_hashes_exact": True,
        "official_and_trkh_worktrees_clean": bool(
            provenance["official_worktree_clean"]
            and provenance["tracked_worktree_clean"]
        ),
        "repository_commit_pushed": provenance["repository_commit"]
        == provenance["upstream_commit"],
        "cohort_count_fold_order_exact": bool(
            provenance["cohort_rows"] == EXPECTED_COHORT_ROWS
            and provenance["positive_rows"] == EXPECTED_POSITIVES
            and provenance["negative_rows"] == EXPECTED_NEGATIVES
            and provenance["ordered_cohort_index_sha256"]
            == EXPECTED_ORDERED_INDEX_SHA256
        ),
        "dataset_mapping_train_only_exact": bool(
            dataset_mapping["paths_exact"] and dataset_mapping["train_paths_only"]
        ),
        "full_model_mapping_and_declarations_exact": bool(
            full_model_identity["mapping_exact"]
            and full_model_identity["native_declarations_exact"]
        ),
        "h1_full_model_identity_exact": bool(
            float(full_model_identity["identity_logit_max_abs_error"])
            <= MAX_EQUATION_ERROR
            and int(full_model_identity["identity_prediction_mismatches"]) == 0
        ),
        "official_oracle_output_error_lte_1e6": max(
            float(equation["max_official_output_error"]),
            float(equation["max_oracle_output_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "oracle_gradient_errors_lte_1e6": max(
            float(equation["max_input_gradient_error"]),
            float(equation["max_weight_gradient_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "h1_output_gradient_identity_lte_1e6": max(
            float(equation["max_identity_output_error"]),
            float(equation["max_identity_input_gradient_error"]),
            float(equation["max_identity_weight_gradient_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "finite_difference_error_lte_1e4": float(
            equation["finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "equations_finite_and_nondegenerate": bool(
            equation["all_cases_finite"] and equation["all_cases_nondegenerate"]
        ),
        "bf16_error_lte_0p02_and_gradients_finite": bool(
            equation["bf16"]["supported"]
            and equation["bf16"]["output_finite"]
            and equation["bf16"]["input_gradient_finite"]
            and equation["bf16"]["weight_gradient_finite"]
            and equation["bf16"]["weight_gradient_nonzero"]
            and float(equation["bf16"]["output_max_abs_error"]) <= MAX_BF16_ERROR
        ),
        "locked_structure_and_parameter_count_exact": bool(
            structure["parameter_counts_equal"]
            and structure["source_kernel_shape"] == [32, 3, 3, 3]
            and structure["candidate_push_kernel_shape"] == [32, 3, 3, 3]
            and structure["candidate_pull_size"] == 7
            and structure["candidate_pull_padding"] == [3, 3]
            and math.isclose(float(structure["candidate_scale"]), PUSH_SCALE)
            and math.isclose(
                float(structure["candidate_alpha"]), INHIBITION_STRENGTH
            )
            and not structure["production_model_mutated"]
        ),
        "onnx_standard_shape_error_and_tensorrt_pass": bool(
            onnx["standard_domains_only"]
            and onnx["output_shape_match"]
            and float(onnx["max_abs_error"]) <= MAX_ONNX_ERROR
            and onnx["tensorrt_parse"]
            and onnx["tensorrt_engine_build"]
            and not onnx["onnx_retained"]
            and not onnx["tensorrt_engine_retained"]
        ),
        "runtime_ratio_lte_1p20": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_lte_1p10": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "descriptor_mechanism_and_gate_replay_exact": bool(
            replay_metrics_exact and replay_mechanism_exact and replay_gate_exact
        ),
        "four_contact_sheets_complete": bool(
            int(contact_sheets["sheet_count"]) == 4
            and contact_sheets["selected_rows_per_sheet"] == [16, 16, 16, 16]
        ),
        "no_holdout_validation_test_or_training": bool(
            not provenance["holdout_data_used"]
            and not provenance["validation_predictions_used"]
            and not provenance["test_data_used"]
            and not provenance["training_used"]
        ),
    }
    structural_gate = {
        "checks": structural_checks,
        "failed_checks": [
            name for name, passed in structural_checks.items() if not passed
        ],
        "passed": all(structural_checks.values()),
    }
    auto_eligible = bool(structural_gate["passed"] and information_gate["passed"])
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": (
            "pending_manual_visual_review" if auto_eligible else "rejected_at_a0"
        ),
        "auto_gate_eligible": auto_eligible,
        "matched_5e_pair_authorized": False,
        "manual_visual_review_required_for_authorization": True,
        "provenance": provenance,
        "structure_audit": structure,
        "equation_audit": equation,
        "resource_audit": resource,
        "onnx_audit": onnx,
        "dataset_mapping": dataset_mapping,
        "full_model_identity_and_frozen_candidate": full_model_identity,
        "loader_audit": {
            "declaration": declaration_loader_summary,
            "features": feature_loaders,
        },
        "readout_diagnostics": readout_diagnostics,
        "readout_metrics": readout_metrics,
        "mechanism_summary": mechanism_summary,
        "information_gate": information_gate,
        "replay_audit": {
            "rows": replay["rows"],
            "metrics_exact": replay_metrics_exact,
            "mechanism_exact": replay_mechanism_exact,
            "gate_exact": replay_gate_exact,
        },
        "contact_sheet_audit": contact_sheets,
        "structural_gate": structural_gate,
        "validation_predictions_used": False,
        "holdout_data_used": False,
        "test_data_used": False,
        "training_used": False,
        "current_best_commands_updated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "cohort_csv": str(cohort_csv.resolve()),
            "report": str(report_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    manifest = _write_manifest(output_dir)
    summary["artifact_manifest"] = manifest
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    if bool(args.preflight_only):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return
    print(
        json.dumps(
            {
                "status": result["status"],
                "auto_gate_eligible": result["auto_gate_eligible"],
                "matched_5e_pair_authorized": result["matched_5e_pair_authorized"],
                "structural_failed_checks": result["structural_gate"][
                    "failed_checks"
                ],
                "information_failed_checks": result["information_gate"][
                    "failed_checks"
                ],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

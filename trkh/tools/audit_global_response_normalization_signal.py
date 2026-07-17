from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import types
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_pixel_difference_stem_signal import (
    _build_dataset,
    _load_keeper_model,
    _make_condition_loader,
)


METHOD = "global_response_normalization_signal_a0"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 12
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
MIN_FIT_TP_RETENTION = 0.95
BENCHMARK_REPEATS = 3
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
LOCKED_DECLARATION_EXCEPTION = {
    "sample_index": 3657,
    "target": 2,
    "expected_prediction": 1,
    "observed_prediction": 2,
}
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
ROLES = (
    "identity",
    "grn_l2",
    "grn_object_response",
    "grn_outside_response",
)
FEATURE_WIDTH = 1024
TOKEN_WIDTH = 256
GRID_HEIGHT = 16
GRID_WIDTH = 16
PATCH_TOKENS = GRID_HEIGHT * GRID_WIDTH
PREFIX_TOKENS = 7
TOKENS = PREFIX_TOKENS + PATCH_TOKENS
MAX_EQUATION_ERROR = 1e-10
MAX_GRADIENT_ERROR = 1e-9
MAX_FINITE_DIFFERENCE_ERROR = 1e-5
MAX_BF16_ERROR = 0.01
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.15
MAX_MEMORY_RATIO = 1.10

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "912a8f101dd41545439a8a235690c3eca29d0d89c5ea7a8a6c65bf92141d86cf"
LOCKED_PAPER_SHA256 = "1c2de7ea1d4f811dcbbafb8fcfac4b9ac506c27fe82cfdb720dec0282890b6a5"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "2553895753323c6fe0b2bf390683f5ea358a42b9"
LOCKED_OFFICIAL_TREE = "0b23579ac3ded0c671f592def6f7b66300a80799"
LOCKED_OFFICIAL_HASHES = {
    "utils_source": "0dabefc9489ed74d0fbfd614527edf83d4172fe905297a17697f15b76c268bb5",
    "convnext_source": "c6a65592720aa7ee57ad92f143ee392e29e18249793ee91259642b401e0f5855",
    "license": "a6a3de076198f9cabf4d2c59429fb6cf74f65d0bde2ceb6f0d9ff9dbc257f0eb",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only ConvNeXt-V2 GRN signal gate. Validation, test, "
            "trainer integration, and raw-data edits are forbidden."
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
            "docs/TRKH_5CLASS_GLOBAL_RESPONSE_NORMALIZATION_SIGNAL_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\convnext-v2-cvpr2023-paper.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\convnext-v2-cvpr2023"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_global_response_normalization_signal_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--finalize-visual-review", action="store_true", default=False)
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
        and int(args.epochs) == EPOCHS
        and math.isclose(float(args.learning_rate), LEARNING_RATE)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY)
        and int(args.benchmark_repeats) == BENCHMARK_REPEATS
        and int(args.seed) == SEED
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    observed = _sha256(resolved)
    if observed != str(expected).strip().casefold():
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tracked_worktree_clean(root: Path) -> bool:
    return not _git_value(root, "status", "--porcelain", "--untracked-files=no")


def _full_worktree_clean(root: Path) -> bool:
    return not _git_value(root, "status", "--porcelain")


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "utils_source": official / "models" / "utils.py",
        "convnext_source": official / "models" / "convnextv2.py",
        "license": official / "LICENSE",
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
        raise ValueError("Arguments differ from the locked GRN A0 protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "GRN protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "accepted ConvNeXt-V2 paper"
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
            paths[name], expected, f"official ConvNeXt-V2 {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(
        paths["official_root"], "rev-parse", "HEAD^{tree}"
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official ConvNeXt-V2 commit differs: {official_commit}"
        )
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official ConvNeXt-V2 tree differs: {official_tree}")
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official ConvNeXt-V2 worktree must be clean.")

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
    positives = sum(_cohort_label(row) == "tp" for row in cohort)
    negatives = sum(_cohort_label(row) == "fp" for row in cohort)
    if (len(cohort), positives, negatives) != (
        EXPECTED_COHORT_ROWS,
        EXPECTED_POSITIVES,
        EXPECTED_NEGATIVES,
    ):
        raise ValueError(
            "Locked GRN cohort differs: "
            f"{len(cohort)}/{positives}/{negatives}"
        )
    fold_counts = {fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS}
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked GRN fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for formal GRN A0.")
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal GRN A0 requires the pushed repository commit.")

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
            "validation_data_used": False,
            "test_data_used": False,
            "image_model_training_used": False,
            "readout_training_used": True,
        },
        rows,
        cohort,
    )


class GlobalResponseNormalization(nn.Module):
    """Channels-last ConvNeXt-V2 GRN equation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))

    def forward(self, x: Tensor) -> Tensor:
        response = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * normalized) + self.beta + x


def normalized_l2_response(
    patches: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    if patches.ndim != 3:
        raise ValueError("GRN patch response requires [B,P,C].")
    if mask is None:
        response = torch.norm(patches, p=2, dim=1, keepdim=True)
    else:
        if mask.ndim != 2 or tuple(mask.shape) != tuple(patches.shape[:2]):
            raise ValueError("GRN response mask must be [B,P].")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every GRN response mask must contain a patch.")
        response = torch.sqrt(
            (patches.square() * mask[:, :, None].to(patches.dtype)).sum(
                dim=1, keepdim=True
            )
        )
    return response / (response.mean(dim=-1, keepdim=True) + 1e-6)


class PatchGlobalResponseNormalization(nn.Module):
    def __init__(self, dim: int, prefix_count: int = PREFIX_TOKENS) -> None:
        super().__init__()
        self.prefix_count = int(prefix_count)
        self.gamma = nn.Parameter(torch.zeros(1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, int(dim)))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or int(tokens.size(1)) <= self.prefix_count:
            raise ValueError("Patch GRN requires [B,N,C] with patch tokens.")
        prefixes = tokens[:, : self.prefix_count]
        patches = tokens[:, self.prefix_count :]
        response = normalized_l2_response(patches)
        if not self.training and not torch.is_grad_enabled():
            patches.addcmul_(patches, self.gamma * response).add_(self.beta)
            return tokens
        patches = patches + self.gamma * (patches * response) + self.beta
        return torch.cat((prefixes, patches), dim=1)


class GlobalResponseReadout(nn.Module):
    def __init__(self, role: str, width: int = FEATURE_WIDTH) -> None:
        super().__init__()
        normalized = str(role).strip().casefold()
        if normalized not in {"identity", "grn_l2"}:
            raise ValueError(f"Unknown GRN readout role: {role}")
        self.role = normalized
        if normalized == "grn_l2":
            self.gamma = nn.Parameter(torch.zeros(1, int(width)))
            self.beta = nn.Parameter(torch.zeros(1, int(width)))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self.head = nn.Linear(int(width), 1)

    def transform(self, object_mean: Tensor, response: Tensor) -> Tensor:
        if self.role == "identity":
            return object_mean
        assert self.gamma is not None and self.beta is not None
        if response.ndim == 3:
            response = response.squeeze(1)
        if response.shape != object_mean.shape:
            raise ValueError("GRN readout response does not match object mean.")
        return object_mean + self.gamma * (object_mean * response) + self.beta

    def forward(
        self, object_mean: Tensor, response: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        transformed = self.transform(object_mean, response)
        logits = self.head(transformed).squeeze(1)
        delta = (transformed.float() - object_mean.float()).square().mean(1).sqrt()
        return logits, transformed, delta


def _object_patch_intersections(
    bboxes: Tensor, height: int = GRID_HEIGHT, width: int = GRID_WIDTH
) -> tuple[Tensor, Tensor]:
    if bboxes.ndim != 2 or int(bboxes.size(1)) != 4:
        raise ValueError("Bboxes must be normalized xywh tensors [B,4].")
    boxes = bboxes.to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, box_width, box_height = boxes.unbind(dim=1)
    x1 = (cx - box_width / 2.0).clamp(0.0, 1.0)
    x2 = (cx + box_width / 2.0).clamp(0.0, 1.0)
    y1 = (cy - box_height / 2.0).clamp(0.0, 1.0)
    y2 = (cy + box_height / 2.0).clamp(0.0, 1.0)
    cell_x1 = (
        torch.arange(width, device=boxes.device, dtype=torch.float32) / float(width)
    ).view(1, 1, width)
    cell_x2 = cell_x1 + 1.0 / float(width)
    cell_y1 = (
        torch.arange(height, device=boxes.device, dtype=torch.float32) / float(height)
    ).view(1, height, 1)
    cell_y2 = cell_y1 + 1.0 / float(height)
    intersects = (
        (cell_x2 > x1[:, None, None])
        & (cell_x1 < x2[:, None, None])
        & (cell_y2 > y1[:, None, None])
        & (cell_y1 < y2[:, None, None])
    )
    outside = ~intersects
    return intersects.flatten(1), outside.flatten(1)


def object_patch_mask(
    bboxes: Tensor, height: int = GRID_HEIGHT, width: int = GRID_WIDTH
) -> tuple[Tensor, Tensor]:
    """Return cells whose area intersects normalized xywh boxes and the complement."""
    intersects, outside = _object_patch_intersections(bboxes, height, width)
    if not bool(intersects.flatten(1).any(1).all()):
        raise ValueError("Every bbox must intersect at least one patch cell.")
    if not bool(outside.flatten(1).any(1).all()):
        raise ValueError("Every bbox must leave at least one outside patch cell.")
    return intersects, outside


def _load_official_grn_class(path: Path):
    """Compile only imports required by the official GRN AST."""
    resolved = Path(path).resolve()
    tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
    class_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GRN"
    ]
    if len(class_nodes) != 1:
        raise ValueError("Official ConvNeXt-V2 source must contain exactly one GRN.")
    module_tree = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="torch")]),
            ast.Import(names=[ast.alias(name="torch.nn", asname="nn")]),
            class_nodes[0],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module_tree)
    module = types.ModuleType("locked_official_convnextv2_grn")
    module.__file__ = str(resolved)
    exec(compile(module_tree, str(resolved), "exec"), module.__dict__)
    return module.GRN


def _independent_grn(
    x: Tensor, gamma: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    response = torch.sqrt(torch.sum(x.square(), dim=(1, 2), keepdim=True))
    normalized = response / (torch.mean(response, dim=-1, keepdim=True) + 1e-6)
    return x + gamma * x * normalized + beta, normalized


def _equation_diagnostics(
    official_source: Path, device: torch.device
) -> Dict[str, object]:
    official_class = _load_official_grn_class(official_source)
    generator = torch.Generator().manual_seed(SEED + 101)
    local = GlobalResponseNormalization(16).double().eval()
    official = official_class(16).double().eval()
    with torch.no_grad():
        local.gamma.copy_(
            torch.randn(local.gamma.shape, generator=generator, dtype=torch.float64)
            * 0.1
        )
        local.beta.copy_(
            torch.randn(local.beta.shape, generator=generator, dtype=torch.float64)
            * 0.1
        )
        official.load_state_dict(local.state_dict(), strict=True)
    x_local = torch.randn(3, 5, 7, 16, generator=generator, dtype=torch.float64)
    x_official = x_local.detach().clone()
    x_oracle = x_local.detach().clone()
    x_local.requires_grad_(True)
    x_official.requires_grad_(True)
    x_oracle.requires_grad_(True)
    local_output = local(x_local)
    official_output = official(x_official)
    oracle_output, oracle_response = _independent_grn(
        x_oracle, local.gamma, local.beta
    )
    probe = torch.randn(local_output.shape, generator=generator, dtype=torch.float64)
    local_gradients = torch.autograd.grad(
        (local_output * probe).sum(),
        [x_local, local.gamma, local.beta],
        retain_graph=True,
    )
    official_gradients = torch.autograd.grad(
        (official_output * probe).sum(),
        [x_official, official.gamma, official.beta],
    )
    oracle_gradients = torch.autograd.grad(
        (oracle_output * probe).sum(),
        [x_oracle, local.gamma, local.beta],
    )
    gradient_errors = {
        name: max(
            float((local_value - official_value).abs().max()),
            float((local_value - oracle_value).abs().max()),
        )
        for name, local_value, official_value, oracle_value in zip(
            ("input", "gamma", "beta"),
            local_gradients,
            official_gradients,
            oracle_gradients,
        )
    }

    finite = copy.deepcopy(local)
    coordinate = (0, 0, 0, 3)
    epsilon = 1e-5
    with torch.no_grad():
        original = float(finite.gamma[coordinate])
        finite.gamma[coordinate] = original + epsilon
        plus = float((finite(x_local.detach()) * probe).sum())
        finite.gamma[coordinate] = original - epsilon
        minus = float((finite(x_local.detach()) * probe).sum())
        finite.gamma[coordinate] = original
    numerical = (plus - minus) / (2.0 * epsilon)
    analytic = float(local_gradients[1][coordinate])

    zero = GlobalResponseNormalization(16).double().eval()
    zero_input = torch.randn(2, 4, 6, 16, generator=generator, dtype=torch.float64)
    zero_error = float((zero(zero_input) - zero_input).abs().max())

    fp32 = GlobalResponseNormalization(16).to(device).eval()
    with torch.no_grad():
        fp32.gamma.uniform_(-0.1, 0.1)
        fp32.beta.uniform_(-0.1, 0.1)
    bf16 = copy.deepcopy(fp32).to(dtype=torch.bfloat16)
    fp32_input = (
        0.25 * torch.randn(2, 5, 7, 16, device=device, dtype=torch.float32)
    ).requires_grad_(True)
    bf16_input = fp32_input.detach().to(torch.bfloat16).requires_grad_(True)
    fp32_output = fp32(fp32_input)
    bf16_output = bf16(bf16_input)
    fp32_gradients = torch.autograd.grad(
        fp32_output.float().square().mean(),
        [fp32_input, fp32.gamma, fp32.beta],
    )
    bf16_gradients = torch.autograd.grad(
        bf16_output.float().square().mean(),
        [bf16_input, bf16.gamma, bf16.beta],
    )
    bf16_error = float(
        (fp32_output.detach() - bf16_output.detach().float()).abs().max()
    )
    fp32_finite = all(bool(torch.isfinite(value).all()) for value in fp32_gradients)
    bf16_finite = all(bool(torch.isfinite(value).all()) for value in bf16_gradients)
    fp32_nonzero = all(int(torch.count_nonzero(value)) > 0 for value in fp32_gradients)
    bf16_nonzero = all(int(torch.count_nonzero(value)) > 0 for value in bf16_gradients)
    fp32.cpu()
    bf16.cpu()
    torch.cuda.empty_cache()

    return {
        "official_class_loaded_from_ast_without_bytecode": True,
        "official_output_max_abs_error": float(
            (local_output.detach() - official_output.detach()).abs().max()
        ),
        "oracle_output_max_abs_error": float(
            (local_output.detach() - oracle_output.detach()).abs().max()
        ),
        "oracle_response_finite": bool(torch.isfinite(oracle_response).all()),
        "gradient_max_abs_errors": gradient_errors,
        "maximum_gradient_error": max(gradient_errors.values()),
        "finite_difference_error": abs(numerical - analytic),
        "finite_difference_analytic": analytic,
        "finite_difference_numerical": numerical,
        "zero_initialization_identity_error": zero_error,
        "bf16_max_abs_error": bf16_error,
        "fp32_gradients_finite": fp32_finite,
        "fp32_gradient_families_nonzero": fp32_nonzero,
        "bf16_gradients_finite": bf16_finite,
        "bf16_gradient_families_nonzero": bf16_nonzero,
        "all_outputs_finite": bool(
            torch.isfinite(local_output).all()
            and torch.isfinite(official_output).all()
            and torch.isfinite(oracle_output).all()
            and torch.isfinite(fp32_output).all()
            and torch.isfinite(bf16_output).all()
        ),
    }


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _block2_hook_module(model: nn.Module) -> nn.Module:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < 2:
        raise TypeError("Keeper does not expose the locked Transformer block list.")
    mlp = getattr(blocks[1], "mlp", None)
    net = getattr(mlp, "net", None)
    if not isinstance(net, nn.Sequential) or len(net) < 4:
        raise TypeError("Keeper block-2 FFN does not expose the locked Sequential.")
    if not isinstance(net[0], nn.Linear) or not isinstance(net[1], nn.GELU):
        raise TypeError("Keeper block-2 hook site is not Linear+GELU.")
    if (int(net[0].in_features), int(net[0].out_features)) != (
        TOKEN_WIDTH,
        FEATURE_WIDTH,
    ):
        raise ValueError("Keeper block-2 FFN dimensions differ from the protocol.")
    return net[1]


def assess_declaration_replay(
    observed: Sequence[Mapping[str, object]], cohort: Sequence[CleanTrainRow]
) -> Dict[str, object]:
    if len(observed) != len(cohort):
        raise ValueError("GRN declaration replay row count differs.")
    mismatches: list[Dict[str, int]] = []
    structural_errors: list[Dict[str, int]] = []
    for source, result in zip(cohort, observed):
        sample_index = int(result["sample_index"])
        target = int(result["target"])
        prediction = int(result["prediction"])
        if sample_index != source.sample_index or target != source.target:
            structural_errors.append(
                {
                    "expected_sample_index": source.sample_index,
                    "observed_sample_index": sample_index,
                    "expected_target": source.target,
                    "observed_target": target,
                }
            )
            continue
        if prediction != source.keeper_prediction:
            mismatches.append(
                {
                    "sample_index": sample_index,
                    "target": target,
                    "expected_prediction": source.keeper_prediction,
                    "observed_prediction": prediction,
                }
            )
    locked_exception_exact = mismatches == [LOCKED_DECLARATION_EXCEPTION]
    return {
        "rows": len(observed),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "structural_error_count": len(structural_errors),
        "structural_errors": structural_errors[:20],
        "locked_exception_exact": locked_exception_exact,
        "new_mismatch_count": (
            0 if locked_exception_exact else len(mismatches)
        ),
        "passed": bool(not structural_errors and locked_exception_exact),
        "batch_size": BATCH_SIZE,
    }


def _audit_mask_geometry(
    *,
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
) -> Dict[str, object]:
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    by_index = {row.sample_index: row for row in cohort}
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="grn_a0_mask_geometry_pre_oof",
    )
    observed_indices: list[int] = []
    observed_targets: list[int] = []
    object_cell_counts: list[int] = []
    invalid_rows: list[Dict[str, object]] = []
    for _images, targets, metadata in loader:
        sample_indices = metadata.get("sample_index")
        crop_bboxes = metadata.get("crop_bbox")
        original_bboxes = metadata.get("bbox")
        if not torch.is_tensor(sample_indices):
            raise ValueError("GRN mask pre-scan lacks sample_index metadata.")
        if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(original_bboxes):
            raise ValueError("GRN mask pre-scan lacks bbox metadata.")
        object_mask, outside_mask = _object_patch_intersections(crop_bboxes)
        counts = object_mask.sum(1)
        for local_index, sample_index_value in enumerate(sample_indices.tolist()):
            sample_index = int(sample_index_value)
            target = int(targets[local_index])
            observed_indices.append(sample_index)
            observed_targets.append(target)
            object_cells = int(counts[local_index])
            object_cell_counts.append(object_cells)
            if not bool(outside_mask[local_index].any()):
                row = by_index[sample_index]
                invalid_rows.append(
                    {
                        "sample_index": sample_index,
                        "fold": int(row.fold),
                        "target": int(row.target),
                        "cohort": str(_cohort_label(row)),
                        "source_stem": row.source_stem,
                        "image_path": str(row.image_path),
                        "crop_bbox_xywh": [
                            float(value) for value in crop_bboxes[local_index].tolist()
                        ],
                        "original_bbox_xywh": [
                            float(value)
                            for value in original_bboxes[local_index].tolist()
                        ],
                        "object_patch_cells": object_cells,
                        "outside_patch_cells": int(
                            outside_mask[local_index].sum().item()
                        ),
                    }
                )
    if observed_indices != indices or observed_targets != expected_targets:
        raise ValueError("GRN mask pre-scan cohort order differs.")
    invalid_indices = [int(row["sample_index"]) for row in invalid_rows]
    return {
        "rows": len(object_cell_counts),
        "condition": "clean",
        "lighting_conditions_share_identical_geometry": True,
        "grid": [GRID_HEIGHT, GRID_WIDTH],
        "mask_definition": "every_patch_cell_intersecting_transformed_bbox",
        "minimum_object_patch_cells": min(object_cell_counts),
        "maximum_object_patch_cells": max(object_cell_counts),
        "rows_with_at_least_240_object_cells": sum(
            value >= 240 for value in object_cell_counts
        ),
        "rows_without_outside_patch": len(invalid_rows),
        "invalid_tp_rows": sum(row["cohort"] == "tp" for row in invalid_rows),
        "invalid_fp_rows": sum(row["cohort"] == "fp" for row in invalid_rows),
        "invalid_fold_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in invalid_rows)
            for fold in FIT_FOLDS
        },
        "invalid_ordered_index_sha256": (
            _ordered_index_sha256(invalid_indices) if invalid_indices else None
        ),
        "invalid_rows": invalid_rows,
        "object_nonempty_all": min(object_cell_counts) > 0,
        "outside_nonempty_all": not invalid_rows,
        "passed": bool(min(object_cell_counts) > 0 and not invalid_rows),
        "loader": loader_summary,
        "raw_dataset_touched": False,
    }


def _object_descriptors(
    patches: Tensor, bboxes: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    object_mask, outside_mask = object_patch_mask(bboxes)
    object_weights = object_mask[:, :, None].to(patches.dtype)
    object_mean = (patches * object_weights).sum(1) / object_weights.sum(1)
    full_response = normalized_l2_response(patches).squeeze(1)
    object_response = normalized_l2_response(patches, object_mask).squeeze(1)
    outside_response = normalized_l2_response(patches, outside_mask).squeeze(1)
    return (
        object_mean,
        full_response,
        object_response,
        outside_response,
        object_mask,
        outside_mask,
    )


def _extract_descriptors(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[int, tuple[float, float, float, float]],
    Dict[str, object],
    Dict[str, object],
]:
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    model = model.to(device).eval()
    hook_module = _block2_hook_module(model)
    features: Dict[str, Dict[str, np.ndarray]] = {}
    original_bboxes: Dict[int, tuple[float, float, float, float]] = {}
    loader_summaries: Dict[str, object] = {}
    observed_declarations: list[Dict[str, object]] = []
    captured_shapes: set[tuple[int, int]] = set()
    descriptor_shapes: set[tuple[int, int]] = set()
    hook_parity_error: Optional[float] = None
    hook_parity_prediction_exact = False
    amp_dtype = _amp_dtype(device)

    for condition, brightness, contrast in CONDITIONS:
        loader, loader_summary = _make_condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=indices,
            brightness=brightness,
            contrast=contrast,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"grn_a0_{condition}",
        )
        loader_summaries[condition] = loader_summary
        batches: Dict[str, list[np.ndarray]] = {
            "object_mean": [],
            "full_response": [],
            "object_response": [],
            "outside_response": [],
        }
        observed_indices: list[int] = []
        observed_targets: list[int] = []
        for batch_number, (images, targets, metadata) in enumerate(loader):
            sample_indices = metadata.get("sample_index")
            crop_bboxes = metadata.get("crop_bbox")
            bboxes = metadata.get("bbox")
            if not torch.is_tensor(sample_indices):
                raise ValueError("GRN feature loader lacks sample_index metadata.")
            if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(bboxes):
                raise ValueError("GRN feature loader lacks bbox metadata.")
            images = images.to(device=device, non_blocking=True)
            targets_device = targets.to(device=device, dtype=torch.long)
            if condition == "clean" and batch_number == 0:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    baseline_logits, _ = _forward_classification_with_metadata(
                        model, images, metadata, device=device
                    )
                baseline_logits = baseline_logits.detach().float()
            else:
                baseline_logits = None

            captured: list[Tensor] = []

            def capture_hidden(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError("GRN hook output is not a tensor.")
                captured.append(output.detach())

            handle = hook_module.register_forward_hook(capture_hidden)
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    logits, _ = _forward_classification_with_metadata(
                        model, images, metadata, device=device
                    )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError(f"GRN hook fired {len(captured)} times.")
            hidden = captured[0]
            if tuple(hidden.shape[1:]) != (TOKENS, FEATURE_WIDTH):
                raise ValueError(f"Locked GRN hook shape differs: {tuple(hidden.shape)}")
            captured_shapes.add((int(hidden.size(1)), int(hidden.size(2))))
            if baseline_logits is not None:
                hook_parity_error = float(
                    (baseline_logits - logits.detach().float()).abs().max()
                )
                hook_parity_prediction_exact = bool(
                    torch.equal(
                        baseline_logits.argmax(1), logits.detach().float().argmax(1)
                    )
                )

            patches = hidden[:, PREFIX_TOKENS:].float()
            crop_bboxes_device = crop_bboxes.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            (
                object_mean,
                full_response,
                object_response,
                outside_response,
                object_mask,
                outside_mask,
            ) = _object_descriptors(patches, crop_bboxes_device)
            if not bool(
                object_mask.any(1).all()
                and outside_mask.any(1).all()
                and torch.isfinite(object_mean).all()
                and torch.isfinite(full_response).all()
                and torch.isfinite(object_response).all()
                and torch.isfinite(outside_response).all()
            ):
                raise ValueError("GRN bbox mask or descriptor is invalid.")
            for name, value in (
                ("object_mean", object_mean),
                ("full_response", full_response),
                ("object_response", object_response),
                ("outside_response", outside_response),
            ):
                descriptor_shapes.add((int(value.size(0)), int(value.size(1))))
                batches[name].append(value.cpu().numpy().astype(np.float32))

            probabilities = F.softmax(logits.detach().float(), dim=1)
            batch_indices = [int(value) for value in sample_indices.tolist()]
            batch_targets = [int(value) for value in targets_device.tolist()]
            observed_indices.extend(batch_indices)
            observed_targets.extend(batch_targets)
            if condition == "clean":
                for local_index, sample_index in enumerate(batch_indices):
                    observed_declarations.append(
                        {
                            "sample_index": sample_index,
                            "target": batch_targets[local_index],
                            "prediction": int(
                                probabilities[local_index].argmax().item()
                            ),
                        }
                    )
                    original_bboxes[sample_index] = tuple(
                        float(value) for value in bboxes[local_index].tolist()
                    )
            del hidden, patches, logits

        if observed_indices != indices or observed_targets != expected_targets:
            raise ValueError(f"{condition} GRN descriptor order differs.")
        features[condition] = {
            name: np.concatenate(values, axis=0).astype(np.float32)
            for name, values in batches.items()
        }
        for name, matrix in features[condition].items():
            if matrix.shape != (EXPECTED_COHORT_ROWS, FEATURE_WIDTH):
                raise ValueError(
                    f"Unexpected {condition}/{name} descriptor shape: {matrix.shape}"
                )
            if not bool(np.isfinite(matrix).all()):
                raise ValueError(f"Non-finite {condition}/{name} descriptors.")

    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    if len(original_bboxes) != EXPECTED_COHORT_ROWS:
        raise ValueError("Clean GRN bbox coverage is incomplete.")
    declaration = assess_declaration_replay(observed_declarations, cohort)
    extraction = {
        "conditions": loader_summaries,
        "captured_token_channel_shapes": [list(value) for value in sorted(captured_shapes)],
        "descriptor_batch_shapes": [list(value) for value in sorted(descriptor_shapes)],
        "hook_site": "blocks[1].mlp.net[1]",
        "hook_before_first_pruning": True,
        "prefix_tokens_excluded": PREFIX_TOKENS,
        "patch_tokens": PATCH_TOKENS,
        "hook_parity_max_abs_logit_error": hook_parity_error,
        "hook_parity_prediction_exact": hook_parity_prediction_exact,
        "hidden_token_map_written_to_disk": False,
        "descriptor_cache_written_to_disk": False,
        "all_masks_nonempty": True,
        "all_descriptors_finite": True,
    }
    return features, original_bboxes, declaration, extraction


def _head_state(seed: int) -> Dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = nn.Linear(FEATURE_WIDTH, 1)
    return {
        name: value.detach().clone() for name, value in head.state_dict().items()
    }


def _build_readout(
    role: str, head_state: Mapping[str, Tensor], seed: int
) -> GlobalResponseReadout:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = GlobalResponseReadout(role, FEATURE_WIDTH)
    model.head.load_state_dict(head_state, strict=True)
    return model


def select_recall_constrained_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    min_tp_retention: float = MIN_FIT_TP_RETENTION,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.size != labels.size or not scores.size:
        raise ValueError("Threshold scores/labels do not align.")
    positives = scores[labels == 1]
    if positives.size == 0:
        raise ValueError("Threshold fit contains no positives.")
    candidates = np.unique(scores)
    valid = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(positives >= threshold)) >= float(min_tp_retention)
    ]
    if not valid:
        raise RuntimeError("No threshold satisfies the fit TP-retention constraint.")
    return max(valid)


def _predict_readout(
    *,
    model: GlobalResponseReadout,
    object_mean: np.ndarray,
    response: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(indices.size), int(batch_size)):
            selected = indices[start : start + int(batch_size)]
            means = torch.from_numpy(object_mean[selected]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            responses = torch.from_numpy(response[selected]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logits, _, delta = model(means, responses)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            deltas.append(delta.cpu().numpy())
    return (
        np.concatenate(scores).astype(np.float64),
        np.concatenate(deltas).astype(np.float64),
    )


def _train_readout(
    *,
    role: str,
    head_state: Mapping[str, Tensor],
    clean: Mapping[str, np.ndarray],
    labels: np.ndarray,
    fit_indices: np.ndarray,
    permutations: Sequence[np.ndarray],
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[GlobalResponseReadout, Dict[str, object]]:
    model = _build_readout(
        role, head_state, seed=int(args.seed) + int(fold) * 1000 + 17
    ).to(device)
    initial_head = {
        name: value.detach().cpu().clone()
        for name, value in model.head.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    curve: list[Dict[str, float]] = []
    for epoch, permutation in enumerate(permutations, start=1):
        model.train()
        losses: list[float] = []
        for start in range(0, int(permutation.size), int(args.batch_size)):
            local = permutation[start : start + int(args.batch_size)]
            selected = fit_indices[local]
            means = torch.from_numpy(clean["object_mean"][selected]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            responses = torch.from_numpy(clean["full_response"][selected]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = torch.as_tensor(
                labels[selected], device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(means, responses)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {role} loss in fold {fold}.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append(
            {"epoch": float(epoch), "mean_loss": float(statistics.mean(losses))}
        )
    return model, {
        "role": role,
        "fold": int(fold),
        "fit_rows": int(fit_indices.size),
        "epochs": int(args.epochs),
        "head_initialization_exact": all(
            torch.equal(initial_head[name], head_state[name]) for name in head_state
        ),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "train_curve": curve,
    }


def _binary_metrics(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> Dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if scores.shape != labels.shape or scores.shape != thresholds.shape:
        raise ValueError("OOF score/label/threshold arrays do not align.")
    if not bool(np.isfinite(scores).all() and np.isfinite(thresholds).all()):
        raise ValueError("OOF scores or thresholds are non-finite.")
    predicted = scores >= thresholds
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tp_retention": float(tp / max(1, tp + fn)),
        "fp_rejection": float(tn / max(1, fp + tn)),
        "precision": float(tp / max(1, tp + fp)),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "tp_median_score": float(np.median(scores[positive])),
        "fp_median_score": float(np.median(scores[negative])),
        "score_mean": float(np.mean(scores)),
        "threshold_mean": float(np.mean(thresholds)),
    }


def _metrics_and_fold_deltas(
    *,
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    include: Optional[np.ndarray] = None,
) -> tuple[Dict[str, Dict[str, Dict[str, object]]], list[float]]:
    mask = (
        np.ones(labels.size, dtype=bool)
        if include is None
        else np.asarray(include, dtype=bool)
    )
    metrics = {
        role: {
            condition: _binary_metrics(
                values[mask], labels[mask], thresholds[role][mask]
            )
            for condition, values in conditions.items()
        }
        for role, conditions in scores.items()
    }
    fold_deltas: list[float] = []
    for fold in FIT_FOLDS:
        selected = mask & (folds == int(fold))
        candidate = _binary_metrics(
            scores["grn_l2"]["clean"][selected],
            labels[selected],
            thresholds["grn_l2"][selected],
        )
        identity = _binary_metrics(
            scores["identity"]["clean"][selected],
            labels[selected],
            thresholds["identity"][selected],
        )
        fold_deltas.append(float(candidate["auroc"] - identity["auroc"]))
    return metrics, fold_deltas


def _run_oof(
    *,
    features: Mapping[str, Mapping[str, np.ndarray]],
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, object],
    Dict[str, object],
    Dict[int, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    scores = {
        role: {
            condition: np.full(labels.size, np.nan, dtype=np.float64)
            for condition, _, _ in CONDITIONS
        }
        for role in ROLES
    }
    thresholds = {
        role: np.full(labels.size, np.nan, dtype=np.float64) for role in ROLES
    }
    clean_delta_rows = np.full(labels.size, np.nan, dtype=np.float64)
    fold_states: Dict[int, Dict[str, np.ndarray]] = {}
    fold_diagnostics: list[Dict[str, object]] = []
    gamma_rms_rows: list[float] = []

    for fold in FIT_FOLDS:
        fit_indices = np.flatnonzero(folds != int(fold)).astype(np.int64)
        hold_indices = np.flatnonzero(folds == int(fold)).astype(np.int64)
        fit_sources = {cohort[index].source_stem for index in fit_indices}
        hold_sources = {cohort[index].source_stem for index in hold_indices}
        overlap = fit_sources.intersection(hold_sources)
        if overlap:
            raise ValueError(f"Source overlap in GRN fold {fold}: {sorted(overlap)[:5]}")
        if set(labels[fit_indices].tolist()) != {0, 1} or set(
            labels[hold_indices].tolist()
        ) != {0, 1}:
            raise ValueError(f"GRN fold {fold} does not contain both classes.")

        rng = np.random.default_rng(int(args.seed) + int(fold) * 100)
        permutations = [
            rng.permutation(fit_indices.size) for _ in range(int(args.epochs))
        ]
        shared_head = _head_state(int(args.seed) + int(fold) * 1000 + 11)
        fold_row: Dict[str, object] = {
            "fold": int(fold),
            "fit_rows": int(fit_indices.size),
            "holdout_rows": int(hold_indices.size),
            "fit_tp": int(np.sum(labels[fit_indices] == 1)),
            "fit_fp": int(np.sum(labels[fit_indices] == 0)),
            "holdout_tp": int(np.sum(labels[hold_indices] == 1)),
            "holdout_fp": int(np.sum(labels[hold_indices] == 0)),
            "fit_sources": len(fit_sources),
            "holdout_sources": len(hold_sources),
            "source_overlap": 0,
            "roles": {},
        }

        for role in ("identity", "grn_l2"):
            readout, diagnostic = _train_readout(
                role=role,
                head_state=shared_head,
                clean=features["clean"],
                labels=labels,
                fit_indices=fit_indices,
                permutations=permutations,
                fold=fold,
                args=args,
                device=device,
            )
            fit_scores, _ = _predict_readout(
                model=readout,
                object_mean=features["clean"]["object_mean"],
                response=features["clean"]["full_response"],
                indices=fit_indices,
                device=device,
                batch_size=int(args.batch_size),
            )
            threshold = select_recall_constrained_threshold(
                fit_scores, labels[fit_indices]
            )
            thresholds[role][hold_indices] = threshold
            if role == "grn_l2":
                thresholds["grn_object_response"][hold_indices] = threshold
                thresholds["grn_outside_response"][hold_indices] = threshold

            condition_metrics: Dict[str, object] = {}
            for condition, _, _ in CONDITIONS:
                hold_scores, hold_delta = _predict_readout(
                    model=readout,
                    object_mean=features[condition]["object_mean"],
                    response=features[condition]["full_response"],
                    indices=hold_indices,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                scores[role][condition][hold_indices] = hold_scores
                condition_metrics[condition] = _binary_metrics(
                    hold_scores,
                    labels[hold_indices],
                    np.full(hold_indices.size, threshold, dtype=np.float64),
                )
                if role == "grn_l2" and condition == "clean":
                    clean_delta_rows[hold_indices] = hold_delta

            if role == "grn_l2":
                for control_role, response_name in (
                    ("grn_object_response", "object_response"),
                    ("grn_outside_response", "outside_response"),
                ):
                    for condition, _, _ in CONDITIONS:
                        control_scores, _ = _predict_readout(
                            model=readout,
                            object_mean=features[condition]["object_mean"],
                            response=features[condition][response_name],
                            indices=hold_indices,
                            device=device,
                            batch_size=int(args.batch_size),
                        )
                        scores[control_role][condition][hold_indices] = control_scores
                assert readout.gamma is not None and readout.beta is not None
                gamma = readout.gamma.detach().cpu().numpy().reshape(-1).astype(np.float64)
                beta = readout.beta.detach().cpu().numpy().reshape(-1).astype(np.float64)
                fold_states[int(fold)] = {"gamma": gamma, "beta": beta}
                gamma_rms = float(np.sqrt(np.mean(gamma**2)))
                beta_rms = float(np.sqrt(np.mean(beta**2)))
                gamma_rms_rows.append(gamma_rms)
                diagnostic["gamma_rms"] = gamma_rms
                diagnostic["beta_rms"] = beta_rms
                diagnostic["added_parameter_count_vs_identity"] = FEATURE_WIDTH * 2

            diagnostic["fit_threshold"] = threshold
            diagnostic["conditions"] = condition_metrics
            fold_row["roles"][role] = diagnostic
            readout.cpu()
            del readout
            gc.collect()
            torch.cuda.empty_cache()

        fold_row["clean_auroc_delta_grn_vs_identity"] = float(
            fold_row["roles"]["grn_l2"]["conditions"]["clean"]["auroc"]
            - fold_row["roles"]["identity"]["conditions"]["clean"]["auroc"]
        )
        fold_diagnostics.append(fold_row)

    for role in ROLES:
        if not bool(np.isfinite(thresholds[role]).all()):
            raise ValueError(f"Incomplete OOF thresholds for {role}.")
        for condition, _, _ in CONDITIONS:
            if not bool(np.isfinite(scores[role][condition]).all()):
                raise ValueError(f"Incomplete OOF scores for {role}/{condition}.")
    if not bool(np.isfinite(clean_delta_rows).all()):
        raise ValueError("GRN activity delta telemetry is incomplete.")

    metrics, fold_deltas = _metrics_and_fold_deltas(
        scores=scores,
        thresholds=thresholds,
        labels=labels,
        folds=folds,
    )
    full_response = features["clean"]["full_response"].astype(np.float64)
    centered_response = full_response - np.mean(full_response, axis=0, keepdims=True)
    activity = {
        "mean_oof_gamma_rms": float(np.mean(gamma_rms_rows)),
        "fold_gamma_rms": gamma_rms_rows,
        "mean_object_vector_delta_rms": float(np.mean(clean_delta_rows)),
        "median_object_vector_delta_rms": float(np.median(clean_delta_rows)),
        "between_sample_full_response_rms": float(
            np.sqrt(np.mean(centered_response**2))
        ),
        "all_finite": True,
    }
    diagnostics = {
        "folds": fold_diagnostics,
        "fold_deltas_grn_vs_identity": fold_deltas,
        "activity": activity,
        "head_initialization_shared_every_role": all(
            bool(role["head_initialization_exact"])
            for fold in fold_diagnostics
            for role in fold["roles"].values()
        ),
        "source_overlap_zero": all(
            int(fold["source_overlap"]) == 0 for fold in fold_diagnostics
        ),
        "natural_row_counts_exact": all(
            int(fold["holdout_tp"]) == EXPECTED_FOLD_COUNTS[int(fold["fold"])]["tp"]
            and int(fold["holdout_fp"])
            == EXPECTED_FOLD_COUNTS[int(fold["fold"])]["fp"]
            for fold in fold_diagnostics
        ),
    }
    activity_rows = {
        "clean_delta_rms": clean_delta_rows,
        "full_response": full_response,
    }
    return scores, thresholds, metrics, diagnostics, fold_states, activity_rows


def assess_information_gate(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    candidate = metrics["grn_l2"]
    identity = metrics["identity"]
    object_response = metrics["grn_object_response"]
    outside_response = metrics["grn_outside_response"]
    fold_deltas = [
        float(value) for value in diagnostics["fold_deltas_grn_vs_identity"]
    ]
    activity = diagnostics["activity"]
    shifts = [name for name, _, _ in CONDITIONS if name != "clean"]
    clean_auroc = float(candidate["clean"]["auroc"])
    checks = {
        "clean_grn_auroc_gte_0p65": clean_auroc >= 0.65,
        "clean_delta_vs_identity_gte_0p03": clean_auroc
        - float(identity["clean"]["auroc"])
        >= 0.03,
        "three_of_four_fold_deltas_positive": sum(value > 0.0 for value in fold_deltas)
        >= 3,
        "no_fold_delta_below_minus_0p02": min(fold_deltas) >= -0.02,
        "all_shift_grn_auroc_gte_0p60": all(
            float(candidate[name]["auroc"]) >= 0.60 for name in shifts
        ),
        "all_shift_grn_not_below_identity": all(
            float(candidate[name]["auroc"]) >= float(identity[name]["auroc"])
            for name in shifts
        ),
        "worst_clean_to_shift_drop_lte_0p05": max(
            clean_auroc - float(candidate[name]["auroc"]) for name in shifts
        )
        <= 0.05,
        "clean_tp_retention_gte_0p95": float(
            candidate["clean"]["tp_retention"]
        )
        >= 0.95,
        "clean_fp_rejection_gte_0p20": float(candidate["clean"]["fp_rejection"])
        >= 0.20,
        "all_shift_tp_retention_gte_0p90": all(
            float(candidate[name]["tp_retention"]) >= 0.90 for name in shifts
        ),
        "all_shift_fp_rejection_gte_0p10": all(
            float(candidate[name]["fp_rejection"]) >= 0.10 for name in shifts
        ),
        "tp_median_gt_fp_every_condition": all(
            float(candidate[name]["tp_median_score"])
            > float(candidate[name]["fp_median_score"])
            for name, _, _ in CONDITIONS
        ),
        "object_response_within_0p02_of_full": float(
            object_response["clean"]["auroc"]
        )
        >= clean_auroc - 0.02,
        "object_response_delta_vs_identity_gte_0p025": float(
            object_response["clean"]["auroc"]
        )
        - float(identity["clean"]["auroc"])
        >= 0.025,
        "object_response_not_below_outside": float(
            object_response["clean"]["auroc"]
        )
        >= float(outside_response["clean"]["auroc"]),
        "object_response_clean_tp_gte_0p93": float(
            object_response["clean"]["tp_retention"]
        )
        >= 0.93,
        "object_response_clean_fp_reject_gte_0p18": float(
            object_response["clean"]["fp_rejection"]
        )
        >= 0.18,
        "mean_gamma_rms_gte_0p005": float(activity["mean_oof_gamma_rms"])
        >= 0.005,
        "mean_object_delta_rms_gte_0p005": float(
            activity["mean_object_vector_delta_rms"]
        )
        >= 0.005,
        "between_sample_response_rms_gte_0p01": float(
            activity["between_sample_full_response_rms"]
        )
        >= 0.01,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "clean_grn_auroc": clean_auroc,
        "clean_delta_vs_identity": clean_auroc
        - float(identity["clean"]["auroc"]),
        "fold_deltas_grn_vs_identity": fold_deltas,
    }


def _no_refit_exception_sensitivity(
    *,
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
    cohort: Sequence[CleanTrainRow],
    diagnostics: Mapping[str, object],
    activity_rows: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    include = np.asarray(
        [row.sample_index != LOCKED_DECLARATION_EXCEPTION["sample_index"] for row in cohort],
        dtype=bool,
    )
    if int(np.sum(~include)) != 1:
        raise ValueError("Locked sample-3657 sensitivity row is missing or duplicated.")
    metrics, fold_deltas = _metrics_and_fold_deltas(
        scores=scores,
        thresholds=thresholds,
        labels=labels,
        folds=folds,
        include=include,
    )
    response = activity_rows["full_response"][include]
    centered = response - np.mean(response, axis=0, keepdims=True)
    sensitivity_diagnostics = {
        "fold_deltas_grn_vs_identity": fold_deltas,
        "activity": {
            "mean_oof_gamma_rms": diagnostics["activity"]["mean_oof_gamma_rms"],
            "mean_object_vector_delta_rms": float(
                np.mean(activity_rows["clean_delta_rms"][include])
            ),
            "between_sample_full_response_rms": float(
                np.sqrt(np.mean(centered**2))
            ),
        },
    }
    return {
        "excluded_sample_index": LOCKED_DECLARATION_EXCEPTION["sample_index"],
        "rows": int(np.sum(include)),
        "refit_performed": False,
        "thresholds_changed": False,
        "metrics": metrics,
        "information_gate": assess_information_gate(
            metrics, sensitivity_diagnostics
        ),
        "cannot_rescue_full_cohort_failure": True,
    }


class MatchedBlock2FFN(nn.Module):
    def __init__(self, source: nn.Sequential, *, use_grn: bool) -> None:
        super().__init__()
        if len(source) < 5:
            raise ValueError("Matched FFN source must contain five modules.")
        self.fc1 = copy.deepcopy(source[0])
        self.activation = copy.deepcopy(source[1])
        self.dropout1 = copy.deepcopy(source[2])
        self.fc2 = copy.deepcopy(source[3])
        self.dropout2 = copy.deepcopy(source[4])
        self.grn = (
            PatchGlobalResponseNormalization(FEATURE_WIDTH, PREFIX_TOKENS)
            if use_grn
            else None
        )

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.activation(self.fc1(tokens))
        if self.grn is not None:
            hidden = self.grn(hidden)
        return self.dropout2(self.fc2(self.dropout1(hidden)))


def _matched_ffns(model: nn.Module) -> tuple[MatchedBlock2FFN, MatchedBlock2FFN]:
    block = getattr(model, "blocks")[1]
    source = getattr(getattr(block, "mlp"), "net")
    native = MatchedBlock2FFN(source, use_grn=False).eval()
    candidate = MatchedBlock2FFN(source, use_grn=True).eval()
    candidate.fc1.load_state_dict(native.fc1.state_dict(), strict=True)
    candidate.fc2.load_state_dict(native.fc2.state_dict(), strict=True)
    return native, candidate


def _linear_states_equal(
    native: MatchedBlock2FFN, candidate: MatchedBlock2FFN
) -> bool:
    return all(
        torch.equal(left, right)
        for left, right in zip(
            (
                native.fc1.weight,
                native.fc1.bias,
                native.fc2.weight,
                native.fc2.bias,
            ),
            (
                candidate.fc1.weight,
                candidate.fc1.bias,
                candidate.fc2.weight,
                candidate.fc2.bias,
            ),
        )
    )


def _activate_ephemeral_grn(candidate: MatchedBlock2FFN) -> None:
    if candidate.grn is None:
        raise ValueError("Candidate FFN lacks GRN.")
    generator = torch.Generator().manual_seed(SEED + 401)
    with torch.no_grad():
        candidate.grn.gamma.copy_(
            torch.randn(
                candidate.grn.gamma.shape, generator=generator, dtype=torch.float32
            )
            * 0.025
        )
        candidate.grn.beta.copy_(
            torch.randn(
                candidate.grn.beta.shape, generator=generator, dtype=torch.float32
            )
            * 0.01
        )


def _benchmark_one(
    model: nn.Module, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    model = copy.deepcopy(model).to(device=device, dtype=torch.float16).eval()
    generator = torch.Generator().manual_seed(SEED + 501)
    tokens = torch.randn(
        BATCH_SIZE, TOKENS, TOKEN_WIDTH, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.float16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    with torch.inference_mode():
        for _ in range(2):
            model(tokens)
        torch.cuda.synchronize(device)
        for _ in range(int(repeats)):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record()
            model(tokens)
            finish.record()
            torch.cuda.synchronize(device)
            elapsed.append(float(start.elapsed_time(finish)))
    peak = int(torch.cuda.max_memory_allocated(device))
    model.cpu()
    del model, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "elapsed_ms": elapsed,
        "median_ms": float(statistics.median(elapsed)),
        "peak_allocated_bytes": peak,
        "batch_size": BATCH_SIZE,
        "tokens": TOKENS,
        "token_width": TOKEN_WIDTH,
        "dtype": "float16",
        "mode": "inference",
    }


def _resource_audit(
    model: nn.Module, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    native, candidate = _matched_ffns(model)
    generator = torch.Generator().manual_seed(SEED + 402)
    probe = torch.randn(2, TOKENS, TOKEN_WIDTH, generator=generator)
    with torch.inference_mode():
        zero_init_error = float((native(probe) - candidate(probe)).abs().max())
    common_state = _linear_states_equal(native, candidate)
    added_parameters = int(
        sum(value.numel() for value in candidate.grn.parameters())
        if candidate.grn is not None
        else 0
    )
    _activate_ephemeral_grn(candidate)
    native_result = _benchmark_one(native, device=device, repeats=repeats)
    candidate_result = _benchmark_one(candidate, device=device, repeats=repeats)
    return {
        "native": native_result,
        "grn": candidate_result,
        "runtime_ratio": float(candidate_result["median_ms"])
        / float(native_result["median_ms"]),
        "peak_memory_ratio": float(candidate_result["peak_allocated_bytes"])
        / float(native_result["peak_allocated_bytes"]),
        "added_parameters": added_parameters,
        "expected_added_parameters": FEATURE_WIDTH * 2,
        "linear_state_common_exact": common_state,
        "zero_init_ffn_output_max_abs_error": zero_init_error,
        "active_nonzero_grn_benchmark": True,
    }


def _onnx_audit(model: nn.Module) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    _, candidate = _matched_ffns(model)
    _activate_ephemeral_grn(candidate)
    candidate = candidate.cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 601)
    example = torch.randn(1, TOKENS, TOKEN_WIDTH, generator=generator)
    with torch.inference_mode():
        reference = candidate(example).numpy()
    with tempfile.TemporaryDirectory(prefix="trkh_grn_onnx_") as directory:
        path = Path(directory) / "block2_ffn_grn.onnx"
        torch.onnx.export(
            candidate,
            example,
            str(path),
            input_names=["tokens"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        observed = session.run(None, {"tokens": example.numpy()})[0]
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
        "input_shape": list(example.shape),
        "output_shape": list(observed.shape),
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


def _write_oof_csv(
    *,
    path: Path,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
) -> None:
    fields = [
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        "cohort",
        "condition",
        *[f"score_{role}" for role in ROLES],
        *[f"threshold_{role}" for role in ROLES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, row in enumerate(cohort):
            for condition, _, _ in CONDITIONS:
                payload: Dict[str, object] = {
                    "sample_index": row.sample_index,
                    "source_stem": row.source_stem,
                    "image_path": str(row.image_path),
                    "fold": row.fold,
                    "target": row.target,
                    "keeper_prediction": row.keeper_prediction,
                    "cohort": _cohort_label(row),
                    "condition": condition,
                }
                for role in ROLES:
                    payload[f"score_{role}"] = float(
                        scores[role][condition][position]
                    )
                    payload[f"threshold_{role}"] = float(
                        thresholds[role][position]
                    )
                writer.writerow(payload)


def _replay_oof_csv(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_index",
            "fold",
            "cohort",
            "condition",
            *{f"score_{role}" for role in ROLES},
            *{f"threshold_{role}" for role in ROLES},
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"GRN replay CSV misses columns: {sorted(missing)}")
        records = [dict(row) for row in reader]
    expected_rows = EXPECTED_COHORT_ROWS * len(CONDITIONS)
    if len(records) != expected_rows:
        raise ValueError(f"GRN replay rows differ: {len(records)} != {expected_rows}")
    clean = [row for row in records if row["condition"] == "clean"]
    ordered_indices = [int(row["sample_index"]) for row in clean]
    if _ordered_index_sha256(ordered_indices) != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError("GRN replay ordered cohort hash differs.")
    labels = np.asarray(
        [1 if row["cohort"] == "tp" else 0 for row in clean], dtype=np.int64
    )
    folds = np.asarray([int(row["fold"]) for row in clean], dtype=np.int64)
    scores = {
        role: {
            condition: np.asarray(
                [
                    float(row[f"score_{role}"])
                    for row in records
                    if row["condition"] == condition
                ],
                dtype=np.float64,
            )
            for condition, _, _ in CONDITIONS
        }
        for role in ROLES
    }
    thresholds = {
        role: np.asarray(
            [float(row[f"threshold_{role}"]) for row in clean], dtype=np.float64
        )
        for role in ROLES
    }
    for role in ROLES:
        for condition, _, _ in CONDITIONS:
            selected = [
                int(row["sample_index"])
                for row in records
                if row["condition"] == condition
            ]
            if selected != ordered_indices:
                raise ValueError(f"GRN replay order differs for {role}/{condition}.")
    metrics, fold_deltas = _metrics_and_fold_deltas(
        scores=scores,
        thresholds=thresholds,
        labels=labels,
        folds=folds,
    )
    return {
        "rows": len(records),
        "ordered_cohort_index_sha256": _ordered_index_sha256(ordered_indices),
        "metrics": metrics,
        "fold_deltas_grn_vs_identity": fold_deltas,
    }


def _canonical_equal(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _select_visual_positions(
    cohort: Sequence[CleanTrainRow], clean_scores: np.ndarray
) -> tuple[list[int], list[str]]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    positions: list[int] = []
    names: list[str] = []
    for cohort_name, label in (("tp", 1), ("fp", 0)):
        available = np.flatnonzero(labels == label)
        ordered = available[np.argsort(clean_scores[available], kind="mergesort")]
        for rank_name, selected in (("lowest", ordered[:4]), ("highest", ordered[-4:])):
            positions.extend(int(value) for value in selected)
            names.extend(
                f"{cohort_name}_{rank_name}_{rank + 1}"
                for rank in range(len(selected))
            )
    if len(positions) != 16 or len(set(positions)) != 16:
        raise ValueError("GRN visual selection must contain 16 unique rows.")
    return positions, names


def _extract_selected_visual_maps(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    selected_positions: Sequence[int],
    fold_states: Mapping[int, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[int, Dict[str, np.ndarray]]:
    selected_rows = [cohort[position] for position in selected_positions]
    loader, _ = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in selected_rows],
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="grn_a0_visual_maps",
    )
    model = model.to(device).eval()
    hook_module = _block2_hook_module(model)
    amp_dtype = _amp_dtype(device)
    output: Dict[int, Dict[str, np.ndarray]] = {}
    observed_indices: list[int] = []
    for images, _targets, metadata in loader:
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("GRN visual loader lacks sample_index metadata.")
        images = images.to(device=device, non_blocking=True)
        captured: list[Tensor] = []

        def capture_hidden(_module, _inputs, value):
            if torch.is_tensor(value):
                captured.append(value.detach())

        handle = hook_module.register_forward_hook(capture_hidden)
        try:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                _forward_classification_with_metadata(
                    model, images, metadata, device=device
                )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("GRN visual hook capture count differs.")
        patches = captured[0][:, PREFIX_TOKENS:].float()
        response = normalized_l2_response(patches)
        indices = [int(value) for value in sample_indices.tolist()]
        observed_indices.extend(indices)
        ordered_gamma = torch.stack(
            [
                torch.from_numpy(
                    fold_states[
                        next(
                            row.fold
                            for row in selected_rows
                            if row.sample_index == sample_index
                        )
                    ]["gamma"]
                )
                for sample_index in indices
            ],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        hidden_rms = patches.square().mean(2).sqrt().reshape(-1, GRID_HEIGHT, GRID_WIDTH)
        modulation = ordered_gamma[:, None, :] * patches * response
        modulation_rms = (
            modulation.square().mean(2).sqrt().reshape(-1, GRID_HEIGHT, GRID_WIDTH)
        )
        for local_index, sample_index in enumerate(indices):
            output[sample_index] = {
                "hidden_rms": hidden_rms[local_index].cpu().numpy().astype(np.float32),
                "modulation_rms": modulation_rms[local_index]
                .cpu()
                .numpy()
                .astype(np.float32),
            }
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    expected_indices = [row.sample_index for row in selected_rows]
    if observed_indices != expected_indices or set(output) != set(expected_indices):
        raise ValueError("GRN visual map coverage/order differs.")
    return output


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas.paste(
        fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    )
    return canvas


def _bbox_overlay(
    image: Image.Image, bbox: tuple[float, float, float, float]
) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    cx, cy, width, height = bbox
    draw.rectangle(
        (
            (cx - width / 2.0) * output.width,
            (cy - height / 2.0) * output.height,
            (cx + width / 2.0) * output.width,
            (cy + height / 2.0) * output.height,
        ),
        outline=(220, 45, 45),
        width=max(2, output.width // 160),
    )
    return output


def _energy_image(energy: np.ndarray) -> Image.Image:
    values = np.asarray(energy, dtype=np.float32)
    lower, upper = np.percentile(values, (2.0, 98.0))
    normalized = np.clip(
        (values - lower) / max(float(upper - lower), 1e-8), 0.0, 1.0
    )
    red = np.clip(1.8 * normalized, 0.0, 1.0)
    green = np.clip(1.8 - np.abs(normalized - 0.55) * 3.2, 0.0, 1.0)
    blue = np.clip(1.4 * (1.0 - normalized), 0.0, 1.0)
    return Image.fromarray(
        np.uint8(np.round(np.stack((red, green, blue), axis=-1) * 255.0))
    )


def _render_contact_sheets(
    *,
    output_dir: Path,
    cohort: Sequence[CleanTrainRow],
    selected_positions: Sequence[int],
    selection_names: Sequence[str],
    bboxes: Mapping[int, tuple[float, float, float, float]],
    maps: Mapping[int, Mapping[str, np.ndarray]],
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    pages: list[Dict[str, object]] = []
    font = ImageFont.load_default()
    for page_number, start in enumerate(range(0, len(selected_positions), 4), start=1):
        page_positions = selected_positions[start : start + 4]
        page_names = selection_names[start : start + 4]
        page = Image.new("RGB", (1760, 1120), (245, 245, 245))
        draw = ImageDraw.Draw(page)
        for panel, (position, selection) in enumerate(zip(page_positions, page_names)):
            row = cohort[position]
            top = panel * 280
            with Image.open(row.image_path) as handle:
                source_image = ImageOps.exif_transpose(handle).convert("RGB")
            overlay = _fit_image(
                _bbox_overlay(source_image, bboxes[row.sample_index]), (300, 250)
            )
            hidden = _fit_image(
                _energy_image(maps[row.sample_index]["hidden_rms"]), (260, 250)
            )
            modulation = _fit_image(
                _energy_image(maps[row.sample_index]["modulation_rms"]), (260, 250)
            )
            page.paste(overlay, (10, top + 20))
            page.paste(hidden, (320, top + 20))
            page.paste(modulation, (590, top + 20))
            lines = [
                f"{selection} | sample={row.sample_index} fold={row.fold}",
                f"target={row.target} keeper={row.keeper_prediction} cohort={_cohort_label(row)}",
                "left=bbox | middle=hidden RMS | right=fitted GRN modulation RMS",
            ]
            for condition, _, _ in CONDITIONS:
                lines.append(
                    f"{condition}: grn={scores['grn_l2'][condition][position]:.6f} "
                    f"identity={scores['identity'][condition][position]:.6f}"
                )
            lines.extend((f"source={row.source_stem}", str(row.image_path)))
            draw.multiline_text(
                (870, top + 28),
                "\n".join(lines),
                fill=(20, 20, 20),
                font=font,
                spacing=7,
            )
        path = output_dir / f"grn_signal_contact_sheet_{page_number:02d}.png"
        page.save(path)
        pages.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "selections": list(page_names),
                "sample_indices": [cohort[value].sample_index for value in page_positions],
            }
        )
    return {
        "page_count": len(pages),
        "selected_rows": len(selected_positions),
        "coverage_complete": len(pages) == 4 and len(selected_positions) == 16,
        "pages": pages,
    }


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _compact_output_files_only(output_dir: Path) -> bool:
    allowed = {".csv", ".json", ".md", ".png"}
    return all(
        path.suffix.casefold() in allowed
        for path in output_dir.rglob("*")
        if path.is_file()
    )


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    forbidden = {".pt", ".pth", ".ckpt", ".engine", ".onnx", ".npy", ".npz"}
    rows: list[Dict[str, object]] = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden retained GRN artifact: {path}")
        rows.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload = {
        "method": METHOD,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
        "forbidden_binary_count": 0,
    }
    path = output_dir / "artifact_manifest.json"
    _write_json(path, payload)
    return {**payload, "path": str(path.resolve()), "sha256": _sha256(path)}


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    metrics = summary["oof_metrics"]
    candidate = metrics["grn_l2"]
    lines = [
        "# Global Response Normalization Signal A0 Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Matched 5e pair authorized: `{str(summary['matched_5e_pair_authorized']).lower()}`",
        f"- Structural failures: `{', '.join(summary['structural_gate']['failed_checks']) or 'none'}`",
        f"- Information failures: `{', '.join(summary['information_gate']['failed_checks']) or 'none'}`",
        "",
        "| Condition | Identity AUROC | GRN AUROC | GRN TP retention | GRN FP rejection | Object-response AUROC | Outside-response AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, _, _ in CONDITIONS:
        lines.append(
            "| {condition} | {identity:.6f} | {grn:.6f} | {tp:.6f} | "
            "{fp:.6f} | {obj:.6f} | {out:.6f} |".format(
                condition=condition,
                identity=float(metrics["identity"][condition]["auroc"]),
                grn=float(candidate[condition]["auroc"]),
                tp=float(candidate[condition]["tp_retention"]),
                fp=float(candidate[condition]["fp_rejection"]),
                obj=float(metrics["grn_object_response"][condition]["auroc"]),
                out=float(metrics["grn_outside_response"][condition]["auroc"]),
            )
        )
    activity = summary["oof_diagnostics"]["activity"]
    lines.extend(
        [
            "",
            "## Activity And Deployment",
            "",
            f"- Mean OOF gamma RMS: `{activity['mean_oof_gamma_rms']:.6f}`",
            f"- Mean object-vector delta RMS: `{activity['mean_object_vector_delta_rms']:.6f}`",
            f"- Between-sample response RMS: `{activity['between_sample_full_response_rms']:.6f}`",
            f"- Runtime ratio: `{summary['resource_audit']['runtime_ratio']:.6f}`",
            f"- Peak-memory ratio: `{summary['resource_audit']['peak_memory_ratio']:.6f}`",
            f"- ONNX Runtime max error: `{summary['onnx_audit']['max_abs_error']:.9g}`",
            f"- TensorRT parse/build: `{summary['onnx_audit']['tensorrt_parse']}/{summary['onnx_audit']['tensorrt_engine_build']}`",
            f"- Visual review: `{summary['visual_review']['result']}`",
            "",
            "No validation/test data, image-model training, raw-data edit, checkpoint, engine, or reusable feature cache was used or retained.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _structural_gate(
    *,
    args: argparse.Namespace,
    provenance: Mapping[str, object],
    equation: Mapping[str, object],
    dataset_mapping: Mapping[str, object],
    declaration: Mapping[str, object],
    extraction: Mapping[str, object],
    diagnostics: Mapping[str, object],
    resource: Mapping[str, object],
    onnx_audit: Mapping[str, object],
    replay_exact: bool,
    contact_sheets: Mapping[str, object],
    compact_output: bool,
    visual_review_completed: bool,
    visual_review_passed: bool,
) -> Dict[str, object]:
    checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_commit_tree_hashes_and_license_exact": True,
        "official_and_trkh_worktrees_clean": bool(
            provenance["official_worktree_clean"]
            and provenance["tracked_worktree_clean"]
        ),
        "repository_commit_pushed": provenance["repository_commit"]
        == provenance["upstream_commit"],
        "cohort_count_fold_and_order_exact": bool(
            provenance["cohort_rows"] == EXPECTED_COHORT_ROWS
            and provenance["positive_rows"] == EXPECTED_POSITIVES
            and provenance["negative_rows"] == EXPECTED_NEGATIVES
            and provenance["ordered_cohort_index_sha256"]
            == EXPECTED_ORDERED_INDEX_SHA256
        ),
        "dataset_mapping_train_only_exact": bool(
            dataset_mapping["paths_exact"] and dataset_mapping["train_paths_only"]
        ),
        "keeper_locked_exception_only": bool(declaration["passed"]),
        "official_and_oracle_output_error_lte_1e10": max(
            float(equation["official_output_max_abs_error"]),
            float(equation["oracle_output_max_abs_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "gradient_error_lte_1e9": float(equation["maximum_gradient_error"])
        <= MAX_GRADIENT_ERROR,
        "finite_difference_error_lte_1e5": float(
            equation["finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "zero_initialization_exact_identity": float(
            equation["zero_initialization_identity_error"]
        )
        == 0.0,
        "fp32_bf16_outputs_and_gradients_pass": bool(
            float(equation["bf16_max_abs_error"]) <= MAX_BF16_ERROR
            and equation["fp32_gradients_finite"]
            and equation["fp32_gradient_families_nonzero"]
            and equation["bf16_gradients_finite"]
            and equation["bf16_gradient_families_nonzero"]
            and equation["all_outputs_finite"]
        ),
        "deployment_hook_shape_prefix_and_parity_exact": bool(
            extraction["captured_token_channel_shapes"] == [[TOKENS, FEATURE_WIDTH]]
            and extraction["prefix_tokens_excluded"] == PREFIX_TOKENS
            and extraction["patch_tokens"] == PATCH_TOKENS
            and float(extraction["hook_parity_max_abs_logit_error"]) == 0.0
            and extraction["hook_parity_prediction_exact"]
        ),
        "bbox_masks_and_descriptors_valid": bool(
            extraction["all_masks_nonempty"]
            and extraction["all_descriptors_finite"]
        ),
        "oof_sources_heads_classes_and_counts_exact": bool(
            diagnostics["source_overlap_zero"]
            and diagnostics["head_initialization_shared_every_role"]
            and diagnostics["natural_row_counts_exact"]
        ),
        "candidate_activity_pass": bool(
            float(diagnostics["activity"]["mean_oof_gamma_rms"]) >= 0.005
            and float(diagnostics["activity"]["mean_object_vector_delta_rms"])
            >= 0.005
            and float(
                diagnostics["activity"]["between_sample_full_response_rms"]
            )
            >= 0.01
        ),
        "matched_ffn_common_state_parameter_and_identity_exact": bool(
            resource["linear_state_common_exact"]
            and int(resource["added_parameters"])
            == int(resource["expected_added_parameters"])
            and float(resource["zero_init_ffn_output_max_abs_error"]) == 0.0
        ),
        "runtime_ratio_lte_1p15": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_lte_1p10": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "onnx_standard_shape_error_pass": bool(
            onnx_audit["standard_domains_only"]
            and onnx_audit["output_shape_match"]
            and float(onnx_audit["max_abs_error"]) <= MAX_ONNX_ERROR
        ),
        "tensorrt_parse_and_build_pass": bool(
            onnx_audit["tensorrt_parse"]
            and onnx_audit["tensorrt_engine_build"]
        ),
        "independent_csv_replay_exact": bool(replay_exact),
        "fixed_contact_sheet_complete": bool(
            contact_sheets["coverage_complete"]
            and int(contact_sheets["selected_rows"]) == 16
            and int(contact_sheets["page_count"]) == 4
        ),
        "formal_output_compact_only": bool(compact_output),
        "ephemeral_features_and_binaries_not_retained": bool(
            not extraction["hidden_token_map_written_to_disk"]
            and not extraction["descriptor_cache_written_to_disk"]
            and not onnx_audit["onnx_retained"]
            and not onnx_audit["tensorrt_engine_retained"]
        ),
        "no_validation_test_or_image_model_training": bool(
            not provenance["validation_data_used"]
            and not provenance["test_data_used"]
            and not provenance["image_model_training_used"]
        ),
        "visual_review_completed": bool(visual_review_completed),
        "visual_review_passed": bool(visual_review_passed),
    }
    automated_checks = {
        name: passed for name, passed in checks.items() if not name.startswith("visual_")
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "automated_checks_passed": all(automated_checks.values()),
    }


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"GRN summary not found: {summary_path}")
    expected = str(args.expected_summary_sha256).strip().casefold()
    if len(expected) != 64 or _sha256(summary_path) != expected:
        raise ValueError("GRN visual finalization summary SHA-256 mismatch.")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual review result and nonempty note are required.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("visual_review", {}).get("result") != "pending":
        raise ValueError("GRN visual review is already finalized.")
    visual = summary["contact_sheet_audit"]
    for page in visual["pages"]:
        if _sha256(Path(page["path"])) != page["sha256"]:
            raise ValueError(f"GRN contact-sheet hash changed: {page['path']}")
    passed = args.visual_review_result == "pass"
    review = {
        "result": str(args.visual_review_result),
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "reviewed_page_count": int(visual["page_count"]),
        "reviewed_page_sha256": [page["sha256"] for page in visual["pages"]],
        "pre_review_summary_sha256": expected,
    }
    _write_json(output_dir / "visual_review.json", review)
    gate = summary["structural_gate"]
    gate["checks"]["visual_review_completed"] = True
    gate["checks"]["visual_review_passed"] = passed
    gate["failed_checks"] = [
        name for name, value in gate["checks"].items() if not value
    ]
    gate["passed"] = all(gate["checks"].values())
    authorized = bool(gate["passed"] and summary["information_gate"]["passed"])
    summary["visual_review"] = review
    summary["matched_5e_pair_authorized"] = authorized
    summary["status"] = (
        "authorized_for_matched_5e_pair" if authorized else "rejected_at_a0"
    )
    _write_json(summary_path, summary)
    _write_report(output_dir / "report.md", summary)
    _write_json(
        output_dir / "visual_review_required.json",
        {
            "completed": True,
            "result": str(args.visual_review_result),
            "pre_review_summary_sha256": expected,
            "final_summary_sha256": _sha256(summary_path),
        },
    )
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def _write_pre_oof_mask_rejection(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, object],
    equation: Mapping[str, object],
    dataset_mapping: Mapping[str, object],
    mask_geometry: Mapping[str, object],
    resource: Mapping[str, object],
    onnx_audit: Mapping[str, object],
    started: float,
) -> Dict[str, object]:
    checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_commit_tree_hashes_and_license_exact": True,
        "official_and_trkh_worktrees_clean": bool(
            provenance["official_worktree_clean"]
            and provenance["tracked_worktree_clean"]
        ),
        "repository_commit_pushed": provenance["repository_commit"]
        == provenance["upstream_commit"],
        "cohort_count_fold_and_order_exact": bool(
            provenance["cohort_rows"] == EXPECTED_COHORT_ROWS
            and provenance["ordered_cohort_index_sha256"]
            == EXPECTED_ORDERED_INDEX_SHA256
        ),
        "dataset_mapping_train_only_exact": bool(
            dataset_mapping["paths_exact"] and dataset_mapping["train_paths_only"]
        ),
        "official_and_oracle_equation_pass": bool(
            max(
                float(equation["official_output_max_abs_error"]),
                float(equation["oracle_output_max_abs_error"]),
            )
            <= MAX_EQUATION_ERROR
            and float(equation["maximum_gradient_error"]) <= MAX_GRADIENT_ERROR
            and float(equation["finite_difference_error"])
            <= MAX_FINITE_DIFFERENCE_ERROR
            and float(equation["bf16_max_abs_error"]) <= MAX_BF16_ERROR
        ),
        "object_patch_nonempty_all_rows": bool(mask_geometry["object_nonempty_all"]),
        "outside_patch_nonempty_all_rows": bool(mask_geometry["outside_nonempty_all"]),
        "runtime_ratio_lte_1p15": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_lte_1p10": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "onnx_standard_shape_error_pass": bool(
            onnx_audit["standard_domains_only"]
            and onnx_audit["output_shape_match"]
            and float(onnx_audit["max_abs_error"]) <= MAX_ONNX_ERROR
        ),
        "tensorrt_parse_and_build_pass": bool(
            onnx_audit["tensorrt_parse"]
            and onnx_audit["tensorrt_engine_build"]
        ),
        "no_validation_test_or_image_model_training": bool(
            not provenance["validation_data_used"]
            and not provenance["test_data_used"]
            and not provenance["image_model_training_used"]
        ),
    }
    gate = {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }
    prior_path = Path(
        "runs/audit_global_response_normalization_signal_a0_20260717"
    ).resolve()
    prior_files = (
        [path for path in prior_path.rglob("*") if path.is_file()]
        if prior_path.exists() and prior_path != output_dir
        else []
    )
    prior_attempt = {
        "path": str(prior_path),
        "exists": prior_path.exists(),
        "file_count": len(prior_files),
        "empty_output_directory": bool(prior_path.exists() and not prior_files),
        "implementation_commit": "ed5c15e512b8af3231f2b160f7471e47a8df05e4",
        "failure": (
            "ValueError: Every bbox must leave at least one outside patch cell."
        ),
        "failure_stage": "first clean descriptor batch before any artifact write",
        "overwritten": False,
    }
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": "rejected_pre_oof_mask_geometry",
        "matched_5e_pair_authorized": False,
        "provenance": provenance,
        "equation_audit": equation,
        "dataset_mapping": dataset_mapping,
        "mask_geometry_audit": mask_geometry,
        "resource_audit": resource,
        "onnx_audit": onnx_audit,
        "structural_gate": gate,
        "information_gate": {
            "status": "not_run",
            "reason": "outside-response control undefined for 13 locked cohort rows",
            "passed": False,
            "cannot_rescue_structural_failure": True,
        },
        "contact_sheet_audit": {
            "status": "not_run",
            "reason": "protocol stops before OOF/XAI after material structural failure",
        },
        "prior_interrupted_attempt": prior_attempt,
        "recovery_replay": True,
        "mask_definition_changed": False,
        "rows_excluded": 0,
        "threshold_or_hyperparameter_sweep": False,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "readout_training_used": False,
        "raw_dataset_touched": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "mask_geometry.json", mask_geometry)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    report_lines = [
        "# Global Response Normalization A0 Pre-OOF Rejection",
        "",
        "- Status: `rejected_pre_oof_mask_geometry`",
        "- Matched 5e pair authorized: `false`",
        f"- Locked cohort rows: `{mask_geometry['rows']}`",
        f"- Rows with no outside patch: `{mask_geometry['rows_without_outside_patch']}`",
        f"- Invalid TP/FP rows: `{mask_geometry['invalid_tp_rows']}/{mask_geometry['invalid_fp_rows']}`",
        f"- Object-cell range: `{mask_geometry['minimum_object_patch_cells']}..{mask_geometry['maximum_object_patch_cells']}` of 256",
        f"- Runtime/memory ratio: `{resource['runtime_ratio']:.6f}/{resource['peak_memory_ratio']:.6f}`",
        f"- Failed structural checks: `{', '.join(gate['failed_checks'])}`",
        "",
        "The locked cell-intersection mask, cohort, and thresholds were not changed. OOF readouts, contact sheets, validation, test, and image-model training were not run. The stop rule closes this route before trainer integration.",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if bool(args.finalize_visual_review):
        if bool(args.preflight_only):
            raise ValueError("Preflight and visual finalization are mutually exclusive.")
        return _finalize_visual_review(args)

    started = time.perf_counter()
    provenance, rows, cohort = _load_locked_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Locked GRN A0 requires CUDA.")
    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    equation = _equation_diagnostics(
        Path(provenance["paths"]["utils_source"]), device
    )
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "preflight_created_output": False,
            "provenance": provenance,
            "equation_audit": equation,
        }

    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint = load_checkpoint(
        Path(provenance["paths"]["checkpoint"]), map_location="cpu"
    )
    model = _load_keeper_model(checkpoint)
    resource = _resource_audit(
        model, device=device, repeats=int(args.benchmark_repeats)
    )
    onnx_audit = _onnx_audit(model)
    base_dataset, transform, dataset_mapping = _build_dataset(
        checkpoint, rows, Path(provenance["paths"]["data"])
    )
    mask_geometry = _audit_mask_geometry(
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        args=args,
    )
    if not bool(mask_geometry["passed"]):
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return _write_pre_oof_mask_rejection(
            output_dir=output_dir,
            args=args,
            provenance=provenance,
            equation=equation,
            dataset_mapping=dataset_mapping,
            mask_geometry=mask_geometry,
            resource=resource,
            onnx_audit=onnx_audit,
            started=started,
        )
    features, bboxes, declaration, extraction = _extract_descriptors(
        model=model,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        args=args,
        device=device,
    )
    (
        scores,
        thresholds,
        metrics,
        diagnostics,
        fold_states,
        activity_rows,
    ) = _run_oof(
        features=features,
        cohort=cohort,
        args=args,
        device=device,
    )
    information_gate = assess_information_gate(metrics, diagnostics)
    sensitivity = _no_refit_exception_sensitivity(
        scores=scores,
        thresholds=thresholds,
        cohort=cohort,
        diagnostics=diagnostics,
        activity_rows=activity_rows,
    )

    oof_path = output_dir / "oof_scores.csv"
    _write_oof_csv(
        path=oof_path, cohort=cohort, scores=scores, thresholds=thresholds
    )
    replay = _replay_oof_csv(oof_path)
    replay_exact = bool(
        _canonical_equal(replay["metrics"], metrics)
        and _canonical_equal(
            replay["fold_deltas_grn_vs_identity"],
            diagnostics["fold_deltas_grn_vs_identity"],
        )
    )
    replay_payload = {**replay, "comparisons_exact": replay_exact}
    _write_json(output_dir / "independent_replay.json", replay_payload)
    _write_json(output_dir / "fold_diagnostics.json", diagnostics)

    selected_positions, selection_names = _select_visual_positions(
        cohort, scores["grn_l2"]["clean"]
    )
    visual_maps = _extract_selected_visual_maps(
        model=model,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        selected_positions=selected_positions,
        fold_states=fold_states,
        args=args,
        device=device,
    )
    contact_sheets = _render_contact_sheets(
        output_dir=output_dir,
        cohort=cohort,
        selected_positions=selected_positions,
        selection_names=selection_names,
        bboxes=bboxes,
        maps=visual_maps,
        scores=scores,
    )
    structural_gate = _structural_gate(
        args=args,
        provenance=provenance,
        equation=equation,
        dataset_mapping=dataset_mapping,
        declaration=declaration,
        extraction=extraction,
        diagnostics=diagnostics,
        resource=resource,
        onnx_audit=onnx_audit,
        replay_exact=replay_exact,
        contact_sheets=contact_sheets,
        compact_output=_compact_output_files_only(output_dir),
        visual_review_completed=False,
        visual_review_passed=False,
    )
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": "awaiting_visual_review",
        "matched_5e_pair_authorized": False,
        "provenance": provenance,
        "equation_audit": equation,
        "dataset_mapping": dataset_mapping,
        "mask_geometry_audit": mask_geometry,
        "declaration_replay": declaration,
        "extraction_audit": extraction,
        "oof_metrics": metrics,
        "oof_diagnostics": diagnostics,
        "information_gate": information_gate,
        "sample_3657_no_refit_sensitivity": sensitivity,
        "resource_audit": resource,
        "onnx_audit": onnx_audit,
        "independent_replay": replay_payload,
        "contact_sheet_audit": contact_sheets,
        "visual_review": {"result": "pending", "passed": False},
        "structural_gate": structural_gate,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "readout_training_used": True,
        "raw_dataset_touched": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_report(output_dir / "report.md", summary)
    summary_sha = _sha256(summary_path)
    _write_json(
        output_dir / "visual_review_required.json",
        {
            "completed": False,
            "page_count": int(contact_sheets["page_count"]),
            "summary_sha256": summary_sha,
            "finalize_arguments": (
                f'--output-dir "{output_dir}" --finalize-visual-review '
                "--visual-review-result pass --visual-review-note <note> "
                f"--expected-summary-sha256 {summary_sha}"
            ),
        },
    )
    del features, activity_rows, fold_states, visual_maps, model
    gc.collect()
    torch.cuda.empty_cache()
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": summary_sha,
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_audit(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    ConvStemBlock,
    HybridConvStem,
    create_model,
)
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
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import _amp_dtype, _forward_logits


METHOD = "learnable_polyphase_downsampling_a0"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
BENCHMARK_BATCH_SIZE = 32
BENCHMARK_REPEATS = 7
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FIT_FOLDS = (1, 2, 3, 4)
SHIFT_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
HIDDEN_CHANNELS = (8, 16, 32)
EXPECTED_COHORT_ROWS = 607
EXPECTED_POSITIVES = 421
EXPECTED_NEGATIVES = 186
EXPECTED_REPLAY_STABLE_ROWS = 606
EXPECTED_REPLAY_STABLE_NEGATIVES = 185
EXPECTED_FOLD_COUNTS = {
    1: {"tp": 112, "fp": 45},
    2: {"tp": 100, "fp": 48},
    3: {"tp": 101, "fp": 52},
    4: {"tp": 108, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd"
)
EXPECTED_DECLARATION_EXCEPTION = {
    "sample_index": 3657,
    "target": 2,
    "expected_prediction": 1,
    "observed_prediction": 2,
}
LOCKED_BBOX_BYTES_SHA256 = (
    "e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b"
)

MAX_OFFICIAL_ERROR = 1e-6
MAX_ORACLE_ERROR = 1e-12
MAX_GRADIENT_ERROR = 1e-7
MAX_FINITE_DIFFERENCE_ERROR = 1e-4
MAX_BF16_ERROR = 0.02
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.15
MAX_MEMORY_RATIO = 1.10
MAX_PARAMETER_RATIO = 1.025

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "6ba9f452dcf75b3c2815adf0055651769a5e362cec8d0c3b5e033e17bb3b8485"
LOCKED_PAPER_SHA256 = "cd13170aa45bc0b2f7afdd62e723bb2dc1674345012f26d45994137c4e3689dc"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "ef28ff29aa058c5f3fdf05b9a096f196ee1d0f30"
LOCKED_OFFICIAL_TREE = "74cda577ec36cc9288a0f2cded289b9b3b3b2269"
LOCKED_OFFICIAL_HASHES = {
    "lps_utils": "f8e28327ad0df66afcf9e5166624c06d59ef8c0555c3f16c8d0564cf05bd3f5e",
    "lps_logit_layers": "003a20c75ccbaa4c704f9ec0f63a2dc7a520f458650f27af7ab339fc91cc354a",
    "polydown": "21e32cac7226b4bd5bdaf33dd45d916e8415a6a3b66ba4193b373b55e1b0b898",
    "logits_channels": "426163ab52ea97c2dc10dd6ca3334f9634d9b0b9d3bb3666a36251b731790113",
    "license": "f5a41d36e1883bc5f0eb5eb520dd2d11c4b3cc225a60c9dfb2bf25ffafa31842",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only no-epoch learnable-polyphase downsampling gate."
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
            "docs/TRKH_5CLASS_LEARNABLE_POLYPHASE_DOWNSAMPLING_A0_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\RojasGomez_Learnable_Polyphase_Sampling_NeurIPS2022.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\learnable_polyphase_sampling"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/audit_learnable_polyphase_downsampling_a0_20260717"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--finalize-visual-review", action="store_true", default=False)
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--benchmark-repeats", type=int, default=BENCHMARK_REPEATS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
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
    layers = official / "learn_poly_sampling" / "layers"
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "lps_utils": layers / "lps_utils.py",
        "lps_logit_layers": layers / "lps_logit_layers.py",
        "polydown": layers / "polydown.py",
        "logits_channels": official
        / "learn_poly_sampling"
        / "configs"
        / "logits_channels"
        / "resnet18_imagenet.json",
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
        raise ValueError("Arguments differ from the locked LPD A0 protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"
        ),
        "launcher_args": _verify_sha256(
            paths["launcher_args"],
            LOCKED_LAUNCHER_ARGS_SHA256,
            "keeper launcher args",
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "LPD A0 protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "accepted LPS paper"
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
            paths[name], expected, f"official LPS {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(
        paths["official_root"], "rev-parse", "HEAD^{tree}"
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(f"Official LPS commit differs: {official_commit}")
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official LPS tree differs: {official_tree}")
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official LPS worktree must be clean.")

    hidden_config = json.loads(paths["logits_channels"].read_text(encoding="utf-8"))
    if hidden_config != {
        "maxpool": 8,
        "layer2": 16,
        "layer3": 32,
        "layer4": 64,
    }:
        raise ValueError(f"Official LPS hidden-channel declaration differs: {hidden_config}")

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
            "Locked LPD cohort differs: "
            f"{len(cohort)}/{positives}/{negatives}"
        )
    fold_counts = {fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS}
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked LPD fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    exception = next(
        (row for row in cohort if row.sample_index == EXPECTED_DECLARATION_EXCEPTION["sample_index"]),
        None,
    )
    if exception is None or exception.target != 2 or exception.keeper_prediction != 1:
        raise ValueError("Locked sample-3657 declaration exception differs.")

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for formal LPD A0.")
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal LPD A0 requires the exact pushed repository commit.")

    return (
        {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "official_worktree_clean": True,
            "official_hidden_channels": hidden_config,
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
        },
        rows,
        cohort,
    )


def split_polyphase_v2(x: Tensor) -> Tensor:
    """Official V2 phase order [00, 01, 10, 11]."""
    if x.ndim != 4:
        raise ValueError("LPD expects a [B,C,H,W] tensor.")
    if int(x.size(-2)) % 2 or int(x.size(-1)) % 2:
        raise ValueError("Locked LPD A0 requires even feature-map dimensions.")
    return torch.stack(
        (
            x[:, :, 0::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 1::2, 1::2],
        ),
        dim=0,
    )


# Adapted for this isolated audit from raymondyeh07/learnable_polyphase_sampling
# at locked commit ef28ff2 under its MIT license.
class LearnablePolyphaseSelector(nn.Module):
    """MIT-source-grounded adaptation of official LPSLogitLayersV2."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            int(in_channels),
            int(hidden_channels),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            int(hidden_channels),
            int(hidden_channels),
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.relu = nn.ReLU()

    def forward(self, components: Tensor) -> Tensor:
        if components.ndim != 5 or int(components.size(0)) != 4:
            raise ValueError("Selector expects [4,B,C,H,W] components.")
        phases, batch = int(components.size(0)), int(components.size(1))
        combined = components.flatten(start_dim=0, end_dim=1)
        values = self.conv2(self.relu(self.conv1(combined)))
        values = values.mean(dim=(-1, -2, -3))
        return values.reshape(phases, batch).transpose(0, 1)


def combine_polyphase(
    components: Tensor,
    logits: Tensor,
    *,
    training: bool,
    tau: float = 1.0,
    convex: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    if training:
        probabilities = (
            F.softmax(logits / float(tau), dim=-1)
            if convex
            else F.gumbel_softmax(logits, tau=float(tau), hard=False, dim=-1)
        )
    else:
        phase = logits.argmax(dim=-1)
        probabilities = F.one_hot(phase, num_classes=4).to(dtype=components.dtype)
    phase = probabilities.argmax(dim=-1)
    stacked = components.permute(1, 2, 3, 4, 0)
    output = (stacked * probabilities[:, None, None, None, :]).sum(dim=-1)
    return output, probabilities, phase


class LearnablePolyphaseMaxPool2d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.selector = LearnablePolyphaseSelector(in_channels, hidden_channels)
        self.register_buffer("gumbel_tau", torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def dense_max(x: Tensor) -> Tensor:
        padded = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return F.max_pool2d(padded, kernel_size=2, stride=1)

    def forward_with_details(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        components = split_polyphase_v2(self.dense_max(x))
        logits = self.selector(components)
        return combine_polyphase(
            components,
            logits,
            training=bool(self.training),
            tau=float(self.gumbel_tau.item()) if self.training else 1.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        output, _, _ = self.forward_with_details(x)
        return output


class LPDStemTrace(nn.Module):
    def __init__(
        self,
        source: HybridConvStem,
        hidden_channels: Sequence[int] = HIDDEN_CHANNELS,
    ) -> None:
        super().__init__()
        if len(source.blocks) != 3 or len(hidden_channels) != 3:
            raise ValueError("LPD A0 requires exactly three stem blocks.")
        blocks: list[ConvStemBlock] = []
        for source_block, hidden in zip(source.blocks, hidden_channels):
            if not isinstance(source_block, ConvStemBlock):
                raise TypeError("LPD A0 requires native ConvStemBlock instances.")
            block = copy.deepcopy(source_block)
            convolution = block.block.conv
            if not isinstance(convolution, nn.Conv2d):
                raise TypeError("LPD stem convolution is not Conv2d.")
            block.block.pool = LearnablePolyphaseMaxPool2d(
                int(convolution.out_channels), int(hidden)
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.downsample_factor = 8
        self.out_channels = int(blocks[-1].block.conv.out_channels)

    def forward_with_phases(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        phases: list[Tensor] = []
        hidden = images
        for block in self.blocks:
            hidden = block.block.conv(hidden)
            hidden = block.block.norm(hidden)
            hidden = block.block.act(hidden)
            pool = block.block.pool
            if not isinstance(pool, LearnablePolyphaseMaxPool2d):
                raise TypeError("LPD stem pool was replaced unexpectedly.")
            hidden, _, phase = pool.forward_with_details(hidden)
            phases.append(phase)
        return hidden, phases[0], phases[1], phases[2]

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_phases(images)[0]


class _LPDStemExport(nn.Module):
    def __init__(self, stem: LPDStemTrace) -> None:
        super().__init__()
        self.stem = stem

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features, phase0, phase1, phase2 = self.stem.forward_with_phases(images)
        return (
            features,
            phase0.to(dtype=torch.int32),
            phase1.to(dtype=torch.int32),
            phase2.to(dtype=torch.int32),
        )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load locked official module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_split(path: Path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "split_polyV2"
    ]
    if len(selected) != 1:
        raise ValueError("Official polydown.py lacks one split_polyV2 function.")
    namespace = {"nn": nn}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["split_polyV2"]


def _copy_selector_state(
    local: LearnablePolyphaseSelector, official: nn.Module
) -> None:
    local.load_state_dict(official.state_dict(), strict=True)


def _phase0_legacy_error(shape: Sequence[int]) -> tuple[float, bool]:
    generator = torch.Generator().manual_seed(SEED + int(sum(shape)))
    source = torch.randn(tuple(int(value) for value in shape), generator=generator)
    dense = LearnablePolyphaseMaxPool2d.dense_max(source)
    phase0 = split_polyphase_v2(dense)[0]
    legacy = F.max_pool2d(source, kernel_size=2, stride=2)
    return float((phase0 - legacy).abs().max()), bool(torch.equal(phase0, legacy))


def _equation_diagnostics(paths: Mapping[str, Path], device: torch.device) -> Dict[str, object]:
    official_logits = _load_module(
        "_trkh_locked_lps_logit_layers", paths["lps_logit_layers"]
    )
    official_utils = _load_module(
        "_trkh_locked_lps_utils", paths["lps_utils"]
    )
    official_split = _load_official_split(paths["polydown"])

    torch.manual_seed(SEED + 701)
    source = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    local_components = split_polyphase_v2(source)
    official_components = official_split(
        x=source,
        stride=2,
        in_channels=3,
        num_components=4,
    )
    oracle_components = torch.stack(
        (
            source[:, :, 0::2, 0::2],
            source[:, :, 0::2, 1::2],
            source[:, :, 1::2, 0::2],
            source[:, :, 1::2, 1::2],
        ),
        dim=0,
    )
    split_official_error = float(
        (local_components - official_components).abs().max()
    )
    split_oracle_error = float((local_components - oracle_components).abs().max())

    torch.manual_seed(SEED + 702)
    official_selector = official_logits.LPSLogitLayersV2(
        in_channels=3,
        hid_channels=5,
        padding_mode="circular",
    ).double()
    local_selector = LearnablePolyphaseSelector(3, 5).double()
    _copy_selector_state(local_selector, official_selector)
    official_phase_logits = official_selector(official_components)
    local_phase_logits = local_selector(local_components)
    logit_error = float((local_phase_logits - official_phase_logits).abs().max())

    official_output, official_probability = official_utils.lps_downsampleV2(
        official_components,
        stride=2,
        polyphase_logits=official_phase_logits,
        mode="test",
        tau=1.0,
        hard=False,
    )
    local_output, _, local_phase = combine_polyphase(
        local_components,
        local_phase_logits,
        training=False,
    )
    output_error = float((local_output - official_output).abs().max())
    official_phase = official_probability.argmax(dim=-1)
    phase_exact = bool(torch.equal(local_phase, official_phase))

    source_official = source.clone().requires_grad_(True)
    source_local = source.clone().requires_grad_(True)
    official_for_grad = official_logits.LPSLogitLayersV2(
        in_channels=3,
        hid_channels=5,
        padding_mode="circular",
    ).double()
    local_for_grad = LearnablePolyphaseSelector(3, 5).double()
    official_for_grad.load_state_dict(official_selector.state_dict(), strict=True)
    _copy_selector_state(local_for_grad, official_for_grad)
    official_grad_components = official_split(
        x=source_official,
        stride=2,
        in_channels=3,
        num_components=4,
    )
    local_grad_components = split_polyphase_v2(source_local)
    official_grad_logits = official_for_grad(official_grad_components)
    local_grad_logits = local_for_grad(local_grad_components)
    official_convex, _ = official_utils.lps_downsampleV2(
        official_grad_components,
        stride=2,
        polyphase_logits=official_grad_logits,
        mode="train_convex",
        tau=1.0,
        hard=False,
    )
    local_convex, _, _ = combine_polyphase(
        local_grad_components,
        local_grad_logits,
        training=True,
        tau=1.0,
        convex=True,
    )
    upstream = torch.linspace(
        -0.2, 0.3, steps=local_convex.numel(), dtype=torch.float64
    ).reshape_as(local_convex)
    official_gradients = torch.autograd.grad(
        (official_convex * upstream).sum(),
        (source_official, *tuple(official_for_grad.parameters())),
    )
    local_gradients = torch.autograd.grad(
        (local_convex * upstream).sum(),
        (source_local, *tuple(local_for_grad.parameters())),
    )
    gradient_errors = [
        float((left - right).abs().max())
        for left, right in zip(local_gradients, official_gradients)
    ]

    parameter = local_for_grad.conv1.weight
    flat_index = 7
    epsilon = 1e-5
    original = float(parameter.detach().reshape(-1)[flat_index])

    def objective(value: float) -> float:
        with torch.no_grad():
            parameter.reshape(-1)[flat_index] = value
        components = split_polyphase_v2(source)
        logits = local_for_grad(components)
        output, _, _ = combine_polyphase(
            components, logits, training=True, tau=1.0, convex=True
        )
        return float((output * upstream).sum())

    plus = objective(original + epsilon)
    minus = objective(original - epsilon)
    with torch.no_grad():
        parameter.reshape(-1)[flat_index] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytic = float(local_gradients[1].reshape(-1)[flat_index])
    finite_difference_error = abs(finite_difference - analytic)

    shifted = torch.roll(source, shifts=(-1, -1), dims=(-2, -1))
    shifted_logits = local_selector(split_polyphase_v2(shifted))
    permutation = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    permutation_error = float(
        (local_phase_logits - shifted_logits.index_select(1, permutation))
        .abs()
        .max()
    )
    shifted_output, _, _ = combine_polyphase(
        split_polyphase_v2(shifted), shifted_logits, training=False
    )
    global_mean_invariance_error = float(
        (
            local_output.mean(dim=(-1, -2))
            - shifted_output.mean(dim=(-1, -2))
        )
        .abs()
        .max()
    )

    legacy_rows = []
    for shape in (
        (1, 32, 256, 256),
        (1, 64, 128, 128),
        (1, 256, 64, 64),
    ):
        error, exact = _phase0_legacy_error(shape)
        legacy_rows.append({"shape": list(shape), "max_abs_error": error, "bit_exact": exact})

    torch.manual_seed(SEED + 703)
    training_pool = LearnablePolyphaseMaxPool2d(4, 6).train()
    optimizer = torch.optim.SGD(training_pool.parameters(), lr=1e-2)
    training_input = torch.randn(3, 4, 16, 16, requires_grad=True)
    before = {
        name: value.detach().clone()
        for name, value in training_pool.named_parameters()
    }
    optimizer.zero_grad(set_to_none=True)
    training_output, _, _ = training_pool.forward_with_details(training_input)
    training_output.square().mean().backward()
    gradient_rows = {
        name: {
            "finite": bool(value.grad is not None and torch.isfinite(value.grad).all()),
            "nonzero": bool(value.grad is not None and torch.count_nonzero(value.grad) > 0),
        }
        for name, value in training_pool.named_parameters()
    }
    optimizer.step()
    movement_rows = {
        name: float((value.detach() - before[name]).abs().max())
        for name, value in training_pool.named_parameters()
    }

    torch.manual_seed(SEED + 704)
    precision_pool = LearnablePolyphaseMaxPool2d(8, 8).to(device).eval()
    precision_input = torch.randn(2, 8, 32, 32, device=device)
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Locked LPD precision gate requires native CUDA BF16.")
    bf16_pool = copy.deepcopy(precision_pool).to(dtype=torch.bfloat16).eval()
    bf16_input = precision_input.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        fp32_output, _, fp32_phase = precision_pool.forward_with_details(
            precision_input
        )
        bf16_output, _, bf16_phase = bf16_pool.forward_with_details(bf16_input)
    bf16_error = float((fp32_output.float() - bf16_output.float()).abs().max())
    bf16_phase_mismatches = int((fp32_phase != bf16_phase).sum())

    bf16_pool.train()
    precision_grad_input = bf16_input.detach().clone().requires_grad_(True)
    bf16_pool.zero_grad(set_to_none=True)
    precision_grad_output, _, _ = bf16_pool.forward_with_details(
        precision_grad_input
    )
    precision_loss = precision_grad_output.float().square().mean()
    precision_loss.backward()
    bf16_gradients_finite = bool(
        precision_grad_input.grad is not None
        and torch.isfinite(precision_grad_input.grad).all()
        and all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in bf16_pool.parameters()
        )
    )
    precision_pool.eval()
    with torch.inference_mode():
        repeats = [
            precision_pool.forward_with_details(precision_input)
            for _ in range(3)
        ]
    deterministic_eval = all(
        torch.equal(repeats[0][0], value[0])
        and torch.equal(repeats[0][2], value[2])
        for value in repeats[1:]
    )

    result = {
        "phase_order": ["00", "01", "10", "11"],
        "split_official_max_abs_error": split_official_error,
        "split_oracle_max_abs_error": split_oracle_error,
        "selector_logit_max_abs_error": logit_error,
        "downsample_output_max_abs_error": output_error,
        "hard_phase_exact": phase_exact,
        "gradient_max_abs_errors": gradient_errors,
        "max_gradient_error": max(gradient_errors),
        "finite_difference_error": finite_difference_error,
        "selector_permutation_max_abs_error": permutation_error,
        "hard_global_mean_invariance_max_abs_error": global_mean_invariance_error,
        "legacy_phase0": legacy_rows,
        "gumbel_gradient_rows": gradient_rows,
        "gumbel_parameter_movement": movement_rows,
        "gumbel_all_gradients_finite": all(
            row["finite"] for row in gradient_rows.values()
        ),
        "gumbel_both_convolution_weights_have_nonzero_gradients": all(
            gradient_rows[name]["nonzero"]
            for name in ("selector.conv1.weight", "selector.conv2.weight")
        ),
        "gumbel_both_convolution_weights_moved": all(
            movement_rows[name] > 0.0
            for name in ("selector.conv1.weight", "selector.conv2.weight")
        ),
        "synthetic_selector_logit_std": float(local_phase_logits.std()),
        "bf16_max_abs_error": bf16_error,
        "bf16_phase_mismatches": bf16_phase_mismatches,
        "bf16_gradients_finite": bf16_gradients_finite,
        "deterministic_eval": deterministic_eval,
        "all_outputs_finite": bool(
            torch.isfinite(local_output).all()
            and torch.isfinite(local_phase_logits).all()
            and torch.isfinite(bf16_output).all()
        ),
    }
    precision_pool.cpu()
    bf16_pool.cpu()
    del precision_input, bf16_input, precision_grad_input, precision_grad_output, bf16_output, fp32_output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _random_model_pair(
    checkpoint: Mapping[str, object],
) -> tuple[nn.Module, nn.Module, Dict[str, object]]:
    model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping):
        raise ValueError("Keeper checkpoint lacks a model config.")
    if not isinstance(class_names, list) or len(class_names) != 5:
        raise ValueError("Keeper checkpoint class declaration is invalid.")
    set_seed(SEED + 801, deterministic=True)
    native = create_model(num_classes=5, model_config=model_config).cpu().eval()
    candidate = copy.deepcopy(native).cpu().eval()
    source_stem = getattr(candidate, "stem", None)
    if not isinstance(source_stem, HybridConvStem):
        raise TypeError("Keeper architecture does not expose HybridConvStem.")
    set_seed(SEED + 802, deterministic=True)
    candidate.stem = LPDStemTrace(source_stem).eval()
    native_count = sum(int(value.numel()) for value in native.parameters())
    candidate_count = sum(int(value.numel()) for value in candidate.parameters())
    native_stem_count = sum(
        int(value.numel()) for value in getattr(native, "stem").parameters()
    )
    candidate_stem_count = sum(
        int(value.numel()) for value in getattr(candidate, "stem").parameters()
    )
    return native, candidate, {
        "native_state_sha256": _state_sha256(native),
        "candidate_state_sha256": _state_sha256(candidate),
        "native_parameter_count": native_count,
        "candidate_parameter_count": candidate_count,
        "added_parameters": candidate_count - native_count,
        "parameter_ratio": float(candidate_count) / float(native_count),
        "native_stem_parameter_count": native_stem_count,
        "candidate_stem_parameter_count": candidate_stem_count,
        "hidden_channels": list(HIDDEN_CHANNELS),
        "no_pretrained_weights": True,
    }


def _benchmark_once(
    *,
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    model = model.to(device).eval()
    images_device = images.to(device=device, non_blocking=True)
    bbox = metadata["bbox"].to(device=device, dtype=torch.float32)
    image_mask = metadata["image_mask"].to(device=device, dtype=torch.bool)
    local_metadata = {"bbox": bbox, "image_mask": image_mask}
    amp_dtype = _amp_dtype(device)
    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                _forward_logits(model, images_device, local_metadata, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms: list[float] = []
        final_logits: Optional[Tensor] = None
        for _ in range(int(repeats)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                final_logits = _forward_logits(
                    model, images_device, local_metadata, device=device
                )
            end.record()
            end.synchronize()
            elapsed_ms.append(float(start.elapsed_time(end)))
        peak = int(torch.cuda.max_memory_allocated(device))
    if final_logits is None:
        raise RuntimeError("LPD benchmark did not execute a forward.")
    result = {
        "median_ms": float(statistics.median(elapsed_ms)),
        "samples_per_second": float(images.size(0))
        / (float(statistics.median(elapsed_ms)) / 1000.0),
        "times_ms": elapsed_ms,
        "peak_allocated_bytes": peak,
        "output_shape": list(final_logits.shape),
        "all_outputs_finite": bool(torch.isfinite(final_logits).all()),
    }
    model.cpu().eval()
    del images_device, bbox, image_mask, local_metadata, final_logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _resource_audit(
    *,
    native: nn.Module,
    candidate: nn.Module,
    model_declaration: Mapping[str, object],
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(SEED + 803)
    images = torch.randn(
        BENCHMARK_BATCH_SIZE,
        3,
        256,
        256,
        generator=generator,
        dtype=torch.float32,
    )
    bbox = torch.tensor(
        [[0.5, 0.5, 0.65, 0.70]], dtype=torch.float32
    ).repeat(BENCHMARK_BATCH_SIZE, 1)
    image_mask = torch.ones(
        BENCHMARK_BATCH_SIZE, 256, 256, dtype=torch.bool
    )
    metadata = {"bbox": bbox, "image_mask": image_mask}

    native_rounds = [
        _benchmark_once(
            model=native,
            images=images,
            metadata=metadata,
            device=device,
            repeats=repeats,
        )
    ]
    candidate_rounds = [
        _benchmark_once(
            model=candidate,
            images=images,
            metadata=metadata,
            device=device,
            repeats=repeats,
        ),
        _benchmark_once(
            model=candidate,
            images=images,
            metadata=metadata,
            device=device,
            repeats=repeats,
        ),
    ]
    native_rounds.append(
        _benchmark_once(
            model=native,
            images=images,
            metadata=metadata,
            device=device,
            repeats=repeats,
        )
    )
    native_ms = float(
        statistics.median(float(row["median_ms"]) for row in native_rounds)
    )
    candidate_ms = float(
        statistics.median(float(row["median_ms"]) for row in candidate_rounds)
    )
    native_memory = float(
        statistics.median(
            float(row["peak_allocated_bytes"]) for row in native_rounds
        )
    )
    candidate_memory = float(
        statistics.median(
            float(row["peak_allocated_bytes"]) for row in candidate_rounds
        )
    )

    native_stem = getattr(native, "stem")
    candidate_stem = getattr(candidate, "stem")
    native_stem.eval()
    candidate_stem.eval()
    with torch.inference_mode():
        small = images[:2]
        native_stem_output = native_stem(small)
        candidate_stem_output, phase0, phase1, phase2 = (
            candidate_stem.forward_with_phases(small)
        )
    return {
        "model_declaration": dict(model_declaration),
        "native_rounds": native_rounds,
        "candidate_rounds": candidate_rounds,
        "native_median_ms": native_ms,
        "candidate_median_ms": candidate_ms,
        "runtime_ratio": candidate_ms / max(native_ms, 1e-12),
        "native_peak_allocated_bytes": int(native_memory),
        "candidate_peak_allocated_bytes": int(candidate_memory),
        "peak_memory_ratio": candidate_memory / max(native_memory, 1.0),
        "native_stem_output_shape": list(native_stem_output.shape),
        "candidate_stem_output_shape": list(candidate_stem_output.shape),
        "phase_shapes": [list(phase0.shape), list(phase1.shape), list(phase2.shape)],
        "phase_values_in_range": bool(
            all(
                torch.all((phase >= 0) & (phase < 4))
                for phase in (phase0, phase1, phase2)
            )
        ),
        "all_outputs_finite": all(
            bool(row["all_outputs_finite"])
            for row in [*native_rounds, *candidate_rounds]
        )
        and bool(torch.isfinite(candidate_stem_output).all()),
    }


def _onnx_audit(candidate_stem: LPDStemTrace) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    stem = _LPDStemExport(copy.deepcopy(candidate_stem).cpu().eval()).eval()
    generator = torch.Generator().manual_seed(SEED + 804)
    images = torch.randn(
        BENCHMARK_BATCH_SIZE,
        3,
        256,
        256,
        generator=generator,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        reference = stem(images)
        reference_arrays = [value.detach().cpu().numpy() for value in reference]
    with tempfile.TemporaryDirectory(prefix="trkh_lpd_onnx_") as directory:
        path = Path(directory) / "lpd_stem_batch32.onnx"
        torch.onnx.export(
            stem,
            (images,),
            str(path),
            input_names=["images"],
            output_names=["stem_features", "phase_0", "phase_1", "phase_2"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        observed = session.run(None, {"images": images.numpy()})
        domains = sorted({str(node.domain) for node in graph.graph.node})
        operators = sorted({str(node.op_type) for node in graph.graph.node})
        payload_sha = _sha256(path)
        payload_bytes = int(path.stat().st_size)
        feature_error = float(
            np.max(np.abs(reference_arrays[0] - observed[0]))
        )
        phase_mismatches = [
            int(np.count_nonzero(reference_arrays[index] != observed[index]))
            for index in range(1, 4)
        ]
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
        "input_shape": list(images.shape),
        "output_shapes": [list(value.shape) for value in observed],
        "domains": domains,
        "operators": operators,
        "standard_domains_only": all(value in {"", "ai.onnx"} for value in domains),
        "feature_max_abs_error": feature_error,
        "phase_mismatches": phase_mismatches,
        "all_outputs_finite": all(np.isfinite(value).all() for value in observed),
        "onnx_sha256_ephemeral": payload_sha,
        "onnx_bytes_ephemeral": payload_bytes,
        "onnx_retained": False,
        "tensorrt_parse": parse_ok,
        "tensorrt_errors": parser_errors,
        "tensorrt_engine_build": engine_ok,
        "tensorrt_engine_bytes_ephemeral": engine_bytes,
        "tensorrt_engine_retained": False,
    }


def _process_snapshot() -> Dict[str, object]:
    current_pid = os.getpid()
    auditor_chain = {current_pid}
    try:
        auditor_chain.update(
            parent.pid for parent in psutil.Process(current_pid).parents()
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    rows = []
    for process in psutil.process_iter(attrs=("pid", "name", "create_time", "cmdline")):
        try:
            name = str(process.info.get("name") or "").casefold()
            if name not in {"python.exe", "trtexec.exe"}:
                continue
            pid = int(process.info["pid"])
            rows.append(
                {
                    "pid": pid,
                    "name": name,
                    "create_time": float(process.info.get("create_time") or 0.0),
                    "command_line": " ".join(process.info.get("cmdline") or []),
                    "current_auditor": pid == current_pid,
                    "current_auditor_chain": pid in auditor_chain,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "current_pid": current_pid,
        "auditor_chain_pids": sorted(auditor_chain),
        "processes": rows,
        "unexpected_processes": [
            row for row in rows if not bool(row["current_auditor_chain"])
        ],
    }


def _gpu_snapshot() -> Dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"Cannot parse nvidia-smi row: {line}")
        rows.append(
            {
                "name": fields[0],
                "utilization_percent": int(fields[1]),
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "temperature_c": int(fields[4]),
                "power_w": float(fields[5]),
            }
        )
    return {"gpus": rows}


def assess_equation_gate(equation: Mapping[str, object]) -> Dict[str, object]:
    checks = {
        "official_phase_split_within_limit": float(
            equation["split_official_max_abs_error"]
        )
        <= MAX_ORACLE_ERROR,
        "independent_phase_split_within_limit": float(
            equation["split_oracle_max_abs_error"]
        )
        <= MAX_ORACLE_ERROR,
        "official_selector_logits_within_limit": float(
            equation["selector_logit_max_abs_error"]
        )
        <= MAX_OFFICIAL_ERROR,
        "official_downsample_output_within_limit": float(
            equation["downsample_output_max_abs_error"]
        )
        <= MAX_OFFICIAL_ERROR,
        "hard_phase_exact": bool(equation["hard_phase_exact"]),
        "official_gradients_within_limit": float(equation["max_gradient_error"])
        <= MAX_GRADIENT_ERROR,
        "finite_difference_within_limit": float(
            equation["finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "selector_logits_permute_under_circular_shift": float(
            equation["selector_permutation_max_abs_error"]
        )
        <= MAX_OFFICIAL_ERROR,
        "hard_global_mean_is_circular_shift_invariant": float(
            equation["hard_global_mean_invariance_max_abs_error"]
        )
        <= MAX_OFFICIAL_ERROR,
        "phase0_is_bit_exact_legacy_maxpool": all(
            bool(row["bit_exact"])
            and float(row["max_abs_error"]) == 0.0
            for row in equation["legacy_phase0"]
        ),
        "gumbel_all_gradients_finite": bool(
            equation["gumbel_all_gradients_finite"]
        ),
        "gumbel_both_convolution_weights_receive_gradient": bool(
            equation["gumbel_both_convolution_weights_have_nonzero_gradients"]
        ),
        "gumbel_step_moves_both_convolution_weights": bool(
            equation["gumbel_both_convolution_weights_moved"]
        ),
        "bf16_output_within_limit": float(equation["bf16_max_abs_error"])
        <= MAX_BF16_ERROR,
        "bf16_gradients_finite": bool(equation["bf16_gradients_finite"]),
        "deterministic_eval": bool(equation["deterministic_eval"]),
        "synthetic_selector_is_nonconstant": float(
            equation["synthetic_selector_logit_std"]
        )
        > 1e-8,
        "all_outputs_finite": bool(equation["all_outputs_finite"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def _translate_tensor(
    tensor: Tensor, *, dx: int, dy: int, replicate: bool
) -> Tensor:
    if tensor.ndim not in {3, 4}:
        raise ValueError("Translation expects [B,H,W] or [B,C,H,W].")
    if abs(int(dx)) != 1 and int(dx) != 0:
        raise ValueError("Locked translation dx must be -1, 0, or 1.")
    if abs(int(dy)) != 1 and int(dy) != 0:
        raise ValueError("Locked translation dy must be -1, 0, or 1.")
    original_height, original_width = int(tensor.size(-2)), int(tensor.size(-1))
    if replicate:
        padded = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
    else:
        padded = F.pad(tensor, (1, 1, 1, 1), mode="constant", value=0)
    y_start = 1 - int(dy)
    x_start = 1 - int(dx)
    return padded[
        ...,
        y_start : y_start + original_height,
        x_start : x_start + original_width,
    ].contiguous()


def _translate_bbox(boxes: Tensor, *, dx: int, dy: int) -> Tensor:
    if boxes.ndim != 2 or int(boxes.size(1)) < 4:
        raise ValueError("Translation bbox must be [B,4+].")
    translated = boxes.to(dtype=torch.float32).clone()
    translated[:, 0] = (translated[:, 0] + float(dx) / 256.0).clamp(0.0, 1.0)
    translated[:, 1] = (translated[:, 1] + float(dy) / 256.0).clamp(0.0, 1.0)
    return translated


def _update_tensor_hash(digest: "hashlib._Hash", name: str, tensor: Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    digest.update(str(name).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))


def _condition_name(dx: int, dy: int) -> str:
    def component(value: int) -> str:
        if value < 0:
            return "m1"
        if value > 0:
            return "p1"
        return "0"

    return f"shift_dx_{component(dx)}_dy_{component(dy)}"


def _metadata_for_shift(
    metadata: Mapping[str, object], *, dx: int, dy: int
) -> Dict[str, Tensor]:
    bbox = metadata.get("bbox")
    crop_bbox = metadata.get("crop_bbox")
    image_mask = metadata.get("image_mask")
    sample_index = metadata.get("sample_index")
    if not all(
        torch.is_tensor(value)
        for value in (bbox, crop_bbox, image_mask, sample_index)
    ):
        raise ValueError("Shift loader lacks bbox/crop_bbox/image_mask/sample_index.")
    assert isinstance(bbox, Tensor)
    assert isinstance(crop_bbox, Tensor)
    assert isinstance(image_mask, Tensor)
    assert isinstance(sample_index, Tensor)
    return {
        "bbox": _translate_bbox(bbox, dx=dx, dy=dy),
        "crop_bbox": _translate_bbox(crop_bbox, dx=dx, dy=dy),
        "image_mask": _translate_tensor(
            image_mask.to(dtype=torch.uint8), dx=dx, dy=dy, replicate=False
        ).to(dtype=torch.bool),
        "sample_index": sample_index.clone(),
    }


def _real_selector_diagnostics(
    *,
    keeper: nn.Module,
    images: Tensor,
    device: torch.device,
) -> Dict[str, object]:
    source_stem = getattr(keeper, "stem", None)
    if not isinstance(source_stem, HybridConvStem):
        raise TypeError("Keeper does not expose the locked HybridConvStem.")
    set_seed(SEED + 901, deterministic=True)
    stem = LPDStemTrace(source_stem).to(device).eval()
    hidden = images.to(device=device, non_blocking=True)
    rows = []
    with torch.inference_mode():
        for layer_index, block in enumerate(stem.blocks):
            hidden = block.block.conv(hidden)
            hidden = block.block.norm(hidden)
            hidden = block.block.act(hidden)
            pool = block.block.pool
            if not isinstance(pool, LearnablePolyphaseMaxPool2d):
                raise TypeError("Real-feature LPD pool differs.")
            components = split_polyphase_v2(pool.dense_max(hidden))
            logits = pool.selector(components)
            output, probabilities, phases = combine_polyphase(
                components, logits, training=False
            )
            counts = torch.bincount(phases, minlength=4)
            rows.append(
                {
                    "layer": layer_index,
                    "input_shape": list(hidden.shape),
                    "output_shape": list(output.shape),
                    "logit_std": float(logits.float().std().item()),
                    "logit_range": float(
                        (logits.float().max() - logits.float().min()).item()
                    ),
                    "phase_counts": [int(value) for value in counts.cpu().tolist()],
                    "probability_shape": list(probabilities.shape),
                    "finite": bool(
                        torch.isfinite(logits).all() and torch.isfinite(output).all()
                    ),
                }
            )
            hidden = output
    stem.cpu()
    del hidden, stem
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "rows": rows,
        "all_finite": all(bool(row["finite"]) for row in rows),
        "all_nonconstant": all(float(row["logit_std"]) > 1e-8 for row in rows),
    }


def _forward_probability_rows(
    *,
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, Tensor],
    condition: str,
    dx: int,
    dy: int,
    cohort_lookup: Mapping[int, CleanTrainRow],
    device: torch.device,
) -> list[Dict[str, object]]:
    images_device = images.to(device=device, non_blocking=True)
    targets_device = targets.to(device=device, dtype=torch.long, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=_amp_dtype(device),
        enabled=device.type == "cuda",
    ):
        logits = _forward_logits(model, images_device, metadata, device=device)
    probabilities = F.softmax(logits.float(), dim=1).cpu()
    sample_indices = metadata["sample_index"].cpu()
    rows = []
    for position in range(int(targets_device.numel())):
        sample_index = int(sample_indices[position].item())
        source = cohort_lookup[sample_index]
        probability = probabilities[position]
        other = torch.cat((probability[:FOCUS_CLASS], probability[FOCUS_CLASS + 1 :]))
        row: Dict[str, object] = {
            "condition": condition,
            "dx": int(dx),
            "dy": int(dy),
            "sample_index": sample_index,
            "fold": int(source.fold),
            "target": int(targets_device[position].item()),
            "category": str(_cohort_label(source)),
            "prediction": int(probability.argmax().item()),
            "focus_probability": float(probability[FOCUS_CLASS].item()),
            "target_probability": float(probability[int(source.target)].item()),
            "focus_margin": float(
                probability[FOCUS_CLASS].item() - other.max().item()
            ),
        }
        for class_index in range(5):
            row[f"prob_{class_index}"] = float(probability[class_index].item())
        rows.append(row)
    return rows


def _shift_inference(
    *,
    model: nn.Module,
    loader,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
) -> tuple[list[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    model = model.to(device).eval()
    cohort_lookup = {row.sample_index: row for row in cohort}
    all_rows: list[Dict[str, object]] = []
    hashes = {
        "clean": hashlib.sha256(),
        **{
            _condition_name(dx, dy): hashlib.sha256()
            for dx, dy in SHIFT_DIRECTIONS
        },
    }
    bbox_batches: list[Tensor] = []
    real_selector: Optional[Dict[str, object]] = None
    for batch_index, (images, targets, metadata_raw) in enumerate(loader):
        if not isinstance(metadata_raw, Mapping):
            raise ValueError("LPD shift loader lacks metadata.")
        required = {
            name: metadata_raw.get(name)
            for name in ("bbox", "crop_bbox", "image_mask", "sample_index")
        }
        if not all(torch.is_tensor(value) for value in required.values()):
            raise ValueError("LPD shift loader metadata is incomplete.")
        clean_metadata = {
            name: value.clone()  # type: ignore[union-attr]
            for name, value in required.items()
        }
        bbox_batches.append(clean_metadata["crop_bbox"][:, :4].float())
        if batch_index == 0:
            real_selector = _real_selector_diagnostics(
                keeper=model,
                images=images,
                device=device,
            )

        for name, tensor in (
            ("images", images),
            ("targets", targets),
            ("bbox", clean_metadata["bbox"]),
            ("crop_bbox", clean_metadata["crop_bbox"]),
            ("image_mask", clean_metadata["image_mask"]),
            ("sample_index", clean_metadata["sample_index"]),
        ):
            _update_tensor_hash(hashes["clean"], name, tensor)
        all_rows.extend(
            _forward_probability_rows(
                model=model,
                images=images,
                targets=targets,
                metadata=clean_metadata,
                condition="clean",
                dx=0,
                dy=0,
                cohort_lookup=cohort_lookup,
                device=device,
            )
        )

        for dx, dy in SHIFT_DIRECTIONS:
            condition = _condition_name(dx, dy)
            shifted_images = _translate_tensor(
                images, dx=dx, dy=dy, replicate=True
            )
            shifted_metadata = _metadata_for_shift(metadata_raw, dx=dx, dy=dy)
            for name, tensor in (
                ("images", shifted_images),
                ("targets", targets),
                ("bbox", shifted_metadata["bbox"]),
                ("crop_bbox", shifted_metadata["crop_bbox"]),
                ("image_mask", shifted_metadata["image_mask"]),
                ("sample_index", shifted_metadata["sample_index"]),
            ):
                _update_tensor_hash(hashes[condition], name, tensor)
            all_rows.extend(
                _forward_probability_rows(
                    model=model,
                    images=shifted_images,
                    targets=targets,
                    metadata=shifted_metadata,
                    condition=condition,
                    dx=dx,
                    dy=dy,
                    cohort_lookup=cohort_lookup,
                    device=device,
                )
            )

    bboxes = torch.cat(bbox_batches, dim=0).contiguous()
    bbox_hash = hashlib.sha256(bboxes.numpy().tobytes(order="C")).hexdigest()
    if tuple(bboxes.shape) != (EXPECTED_COHORT_ROWS, 4):
        raise ValueError(f"Locked LPD bbox shape differs: {tuple(bboxes.shape)}")
    if bbox_hash != LOCKED_BBOX_BYTES_SHA256:
        raise ValueError(f"Locked LPD bbox byte hash differs: {bbox_hash}")
    model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    if real_selector is None:
        raise RuntimeError("LPD real-selector diagnostics did not run.")
    return (
        all_rows,
        {
            "condition_tensor_sha256": {
                name: digest.hexdigest() for name, digest in hashes.items()
            },
            "bbox_shape": list(bboxes.shape),
            "bbox_bytes_sha256": bbox_hash,
            "rows": len(all_rows),
            "expected_rows": EXPECTED_COHORT_ROWS * (1 + len(SHIFT_DIRECTIONS)),
            "raw_data_modified": False,
        },
        real_selector,
    )


def summarize_shift_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    grouped: Dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_index"]), []).append(row)
    if len(grouped) != EXPECTED_COHORT_ROWS:
        raise ValueError(f"Shift replay sample count differs: {len(grouped)}")

    samples: list[Dict[str, object]] = []
    direction_events = {
        _condition_name(dx, dy): 0 for dx, dy in SHIFT_DIRECTIONS
    }
    declaration_mismatches: list[Dict[str, int]] = []
    for sample_index in sorted(grouped):
        sample_rows = grouped[sample_index]
        if len(sample_rows) != 1 + len(SHIFT_DIRECTIONS):
            raise ValueError(f"Sample {sample_index} has {len(sample_rows)} conditions.")
        by_condition = {str(row["condition"]): row for row in sample_rows}
        clean = by_condition.get("clean")
        if clean is None:
            raise ValueError(f"Sample {sample_index} lacks clean inference.")
        expected_prediction = FOCUS_CLASS
        observed_prediction = int(clean["prediction"])
        if observed_prediction != expected_prediction:
            declaration_mismatches.append(
                {
                    "sample_index": sample_index,
                    "target": int(clean["target"]),
                    "expected_prediction": expected_prediction,
                    "observed_prediction": observed_prediction,
                }
            )
        ordered = [clean] + [
            by_condition[_condition_name(dx, dy)] for dx, dy in SHIFT_DIRECTIONS
        ]
        focus_values = [float(row["focus_probability"]) for row in ordered]
        target_values = [float(row["target_probability"]) for row in ordered]
        margin_values = [float(row["focus_margin"]) for row in ordered]
        replay_stable = sample_index != EXPECTED_DECLARATION_EXCEPTION["sample_index"]
        exits = [
            row
            for row in ordered[1:]
            if replay_stable and int(row["prediction"]) != FOCUS_CLASS
        ]
        for row in exits:
            direction_events[str(row["condition"])] += 1
        adverse = min(
            ordered[1:], key=lambda row: (float(row["focus_probability"]), str(row["condition"]))
        )
        samples.append(
            {
                "sample_index": sample_index,
                "fold": int(clean["fold"]),
                "target": int(clean["target"]),
                "category": str(clean["category"]),
                "replay_stable": replay_stable,
                "clean_prediction": observed_prediction,
                "any_class1_exit": bool(exits),
                "exit_conditions": [str(row["condition"]) for row in exits],
                "focus_probability_span": max(focus_values) - min(focus_values),
                "target_probability_span": max(target_values) - min(target_values),
                "focus_margin_span": max(margin_values) - min(margin_values),
                "adverse_condition": str(adverse["condition"]),
                "adverse_dx": int(adverse["dx"]),
                "adverse_dy": int(adverse["dy"]),
                "adverse_prediction": int(adverse["prediction"]),
                "adverse_focus_probability": float(adverse["focus_probability"]),
                "clean_focus_probability": float(clean["focus_probability"]),
                "clean_focus_margin": float(clean["focus_margin"]),
                "adverse_focus_margin": float(adverse["focus_margin"]),
            }
        )

    stable = [row for row in samples if bool(row["replay_stable"])]
    stable_tp = [row for row in stable if row["category"] == "tp"]
    stable_fp = [row for row in stable if row["category"] == "fp"]
    exits = [row for row in stable if bool(row["any_class1_exit"])]
    tp_exits = [row for row in stable_tp if bool(row["any_class1_exit"])]
    fp_exits = [row for row in stable_fp if bool(row["any_class1_exit"])]
    fold_events = {
        str(fold): sum(
            bool(row["any_class1_exit"]) and int(row["fold"]) == fold
            for row in stable
        )
        for fold in FIT_FOLDS
    }
    spans = np.asarray(
        [float(row["focus_probability_span"]) for row in samples],
        dtype=np.float64,
    )
    cardinal_names = {
        _condition_name(dx, dy) for dx, dy in SHIFT_DIRECTIONS if dx == 0 or dy == 0
    }
    diagonal_names = set(direction_events) - cardinal_names
    summary = {
        "rows": len(rows),
        "samples": len(samples),
        "replay_stable_samples": len(stable),
        "replay_stable_tp": len(stable_tp),
        "replay_stable_fp": len(stable_fp),
        "any_class1_exit_count": len(exits),
        "tp_exit_count": len(tp_exits),
        "restricted_fp_exit_count": len(fp_exits),
        "fold_exit_counts": fold_events,
        "direction_exit_counts": direction_events,
        "cardinal_directions_with_exit": sum(
            int(direction_events[name] > 0) for name in cardinal_names
        ),
        "diagonal_directions_with_exit": sum(
            int(direction_events[name] > 0) for name in diagonal_names
        ),
        "focus_probability_span_median": float(np.quantile(spans, 0.50)),
        "focus_probability_span_p90": float(np.quantile(spans, 0.90)),
        "focus_probability_span_max": float(spans.max()),
        "declaration_mismatches": declaration_mismatches,
        "declaration_mismatch_count": len(declaration_mismatches),
        "all_numeric_finite": bool(np.isfinite(spans).all())
        and all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "focus_probability",
                "target_probability",
                "focus_margin",
                "prob_0",
                "prob_1",
                "prob_2",
                "prob_3",
                "prob_4",
            )
        ),
    }
    return summary, samples


def assess_shift_signal(summary: Mapping[str, object]) -> Dict[str, object]:
    expected_exception = [EXPECTED_DECLARATION_EXCEPTION]
    checks = {
        "exact_condition_row_count": int(summary["rows"])
        == EXPECTED_COHORT_ROWS * (1 + len(SHIFT_DIRECTIONS)),
        "exact_sample_count": int(summary["samples"]) == EXPECTED_COHORT_ROWS,
        "exact_replay_stable_count": int(summary["replay_stable_samples"])
        == EXPECTED_REPLAY_STABLE_ROWS,
        "exact_replay_stable_tp_count": int(summary["replay_stable_tp"])
        == EXPECTED_POSITIVES,
        "exact_replay_stable_fp_count": int(summary["replay_stable_fp"])
        == EXPECTED_REPLAY_STABLE_NEGATIVES,
        "sample3657_is_only_declaration_exception": list(
            summary["declaration_mismatches"]
        )
        == expected_exception,
        "at_least_18_class1_exit_samples": int(summary["any_class1_exit_count"])
        >= 18,
        "at_least_4_tp_exits": int(summary["tp_exit_count"]) >= 4,
        "at_least_4_restricted_fp_exits": int(
            summary["restricted_fp_exit_count"]
        )
        >= 4,
        "at_least_3_folds_have_2_exits": sum(
            int(value) >= 2 for value in summary["fold_exit_counts"].values()
        )
        >= 3,
        "median_focus_span_at_least_0p010": float(
            summary["focus_probability_span_median"]
        )
        >= 0.010,
        "p90_focus_span_at_least_0p030": float(
            summary["focus_probability_span_p90"]
        )
        >= 0.030,
        "at_least_2_cardinal_directions_have_exit": int(
            summary["cardinal_directions_with_exit"]
        )
        >= 2,
        "at_least_2_diagonal_directions_have_exit": int(
            summary["diagonal_directions_with_exit"]
        )
        >= 2,
        "all_numeric_values_finite": bool(summary["all_numeric_finite"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def assess_structural_gate(
    *,
    equation_gate: Mapping[str, object],
    resource: Mapping[str, object],
    onnx: Mapping[str, object],
    transform_audit: Mapping[str, object],
    real_selector: Mapping[str, object],
    process_before: Mapping[str, object],
    gpu_before: Mapping[str, object],
) -> Dict[str, object]:
    declaration = resource["model_declaration"]
    gpu_rows = gpu_before.get("gpus", [])
    isolated_gpu = bool(gpu_rows) and all(
        int(row["utilization_percent"]) <= 15
        and int(row["memory_used_mib"]) <= 2200
        for row in gpu_rows
    )
    checks = {
        "equation_gate_passed": bool(equation_gate["passed"]),
        "no_unexpected_python_or_tensorrt_process": not bool(
            process_before["unexpected_processes"]
        ),
        "gpu_isolated_before_measurement": isolated_gpu,
        "locked_bbox_hash_exact": str(transform_audit["bbox_bytes_sha256"])
        == LOCKED_BBOX_BYTES_SHA256,
        "all_condition_tensor_rows_present": int(transform_audit["rows"])
        == int(transform_audit["expected_rows"]),
        "raw_data_not_modified": not bool(transform_audit["raw_data_modified"]),
        "real_selector_outputs_finite": bool(real_selector["all_finite"]),
        "real_selector_logits_nonconstant": bool(real_selector["all_nonconstant"]),
        "parameter_increase_within_2p5_percent": float(
            declaration["parameter_ratio"]
        )
        <= MAX_PARAMETER_RATIO,
        "added_parameters_positive": int(declaration["added_parameters"]) > 0,
        "no_pretrained_weights": bool(declaration["no_pretrained_weights"]),
        "stem_output_shape_unchanged": list(resource["native_stem_output_shape"])
        == list(resource["candidate_stem_output_shape"]),
        "phase_shapes_match_batch": all(
            list(shape) == [2] for shape in resource["phase_shapes"]
        ),
        "phase_values_in_range": bool(resource["phase_values_in_range"]),
        "all_resource_outputs_finite": bool(resource["all_outputs_finite"]),
        "runtime_ratio_within_limit": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_within_limit": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "onnx_standard_domains_only": bool(onnx["standard_domains_only"]),
        "onnx_output_error_within_limit": float(onnx["feature_max_abs_error"])
        <= MAX_ONNX_ERROR,
        "onnx_phase_outputs_exact": all(
            int(value) == 0 for value in onnx["phase_mismatches"]
        ),
        "onnx_outputs_finite": bool(onnx["all_outputs_finite"]),
        "onnx_payload_ephemeral": not bool(onnx["onnx_retained"]),
        "tensorrt_parse_passed": bool(onnx["tensorrt_parse"]),
        "tensorrt_engine_build_passed": bool(onnx["tensorrt_engine_build"]),
        "tensorrt_payload_ephemeral": not bool(
            onnx["tensorrt_engine_retained"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


SHIFT_CSV_FIELDS = (
    "condition",
    "dx",
    "dy",
    "sample_index",
    "fold",
    "target",
    "category",
    "prediction",
    "focus_probability",
    "target_probability",
    "focus_margin",
    "prob_0",
    "prob_1",
    "prob_2",
    "prob_3",
    "prob_4",
)


def _write_shift_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHIFT_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            payload = {}
            for name in SHIFT_CSV_FIELDS:
                value = row[name]
                payload[name] = (
                    format(float(value), ".17g")
                    if name
                    in {
                        "focus_probability",
                        "target_probability",
                        "focus_margin",
                        "prob_0",
                        "prob_1",
                        "prob_2",
                        "prob_3",
                        "prob_4",
                    }
                    else value
                )
            writer.writerow(payload)


def _read_shift_csv(path: Path) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, object] = {
                "condition": str(raw["condition"]),
                "dx": int(raw["dx"]),
                "dy": int(raw["dy"]),
                "sample_index": int(raw["sample_index"]),
                "fold": int(raw["fold"]),
                "target": int(raw["target"]),
                "category": str(raw["category"]),
                "prediction": int(raw["prediction"]),
            }
            for name in (
                "focus_probability",
                "target_probability",
                "focus_margin",
                "prob_0",
                "prob_1",
                "prob_2",
                "prob_3",
                "prob_4",
            ):
                row[name] = float(raw[name])
            rows.append(row)
    return rows


SAMPLE_CSV_FIELDS = (
    "sample_index",
    "fold",
    "target",
    "category",
    "replay_stable",
    "clean_prediction",
    "any_class1_exit",
    "exit_conditions",
    "focus_probability_span",
    "target_probability_span",
    "focus_margin_span",
    "adverse_condition",
    "adverse_dx",
    "adverse_dy",
    "adverse_prediction",
    "adverse_focus_probability",
    "clean_focus_probability",
    "clean_focus_margin",
    "adverse_focus_margin",
)


def _write_sample_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            payload = {}
            for name in SAMPLE_CSV_FIELDS:
                value = row[name]
                if name == "exit_conditions":
                    payload[name] = ";".join(str(item) for item in value)
                elif isinstance(value, float):
                    payload[name] = format(value, ".17g")
                else:
                    payload[name] = value
            writer.writerow(payload)


def _numeric_differences(
    expected: object,
    observed: object,
    *,
    path: str = "root",
    tolerance: float = 1e-12,
) -> list[Dict[str, object]]:
    differences: list[Dict[str, object]] = []
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        keys = sorted(set(expected) | set(observed))
        for key in keys:
            if key not in expected or key not in observed:
                differences.append(
                    {"path": f"{path}.{key}", "expected": expected.get(key), "observed": observed.get(key)}
                )
            else:
                differences.extend(
                    _numeric_differences(
                        expected[key],
                        observed[key],
                        path=f"{path}.{key}",
                        tolerance=tolerance,
                    )
                )
        return differences
    if isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
        if len(expected) != len(observed):
            return [{"path": path, "expected_length": len(expected), "observed_length": len(observed)}]
        for index, (left, right) in enumerate(zip(expected, observed)):
            differences.extend(
                _numeric_differences(
                    left,
                    right,
                    path=f"{path}[{index}]",
                    tolerance=tolerance,
                )
            )
        return differences
    if (
        isinstance(expected, (int, float, np.integer, np.floating))
        and not isinstance(expected, bool)
        and isinstance(observed, (int, float, np.integer, np.floating))
        and not isinstance(observed, bool)
    ):
        delta = abs(float(expected) - float(observed))
        if not math.isfinite(delta) or delta > float(tolerance):
            differences.append(
                {
                    "path": path,
                    "expected": float(expected),
                    "observed": float(observed),
                    "abs_delta": delta,
                }
            )
        return differences
    if expected != observed:
        differences.append({"path": path, "expected": expected, "observed": observed})
    return differences


def _tensor_to_pil(image: Tensor, semantics: Mapping[str, object]) -> Image.Image:
    mean = torch.tensor(
        semantics["input_mean"], dtype=torch.float32
    ).view(3, 1, 1)
    std = torch.tensor(
        semantics["input_std"], dtype=torch.float32
    ).view(3, 1, 1)
    rgb = (image.detach().cpu().float() * std + mean).clamp(0.0, 1.0)
    array = (
        rgb.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _overlay_bbox(image: Image.Image, bbox: Sequence[float], color: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    cx, cy, width, height = [float(value) for value in bbox[:4]]
    x1 = int(round((cx - width / 2.0) * output.width))
    x2 = int(round((cx + width / 2.0) * output.width))
    y1 = int(round((cy - height / 2.0) * output.height))
    y2 = int(round((cy + height / 2.0) * output.height))
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    return output


def _render_visuals(
    *,
    output_dir: Path,
    base_dataset,
    transform,
    semantics: Mapping[str, object],
    sample_rows: Sequence[Mapping[str, object]],
    shift_rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    by_category = {
        "tp": [row for row in sample_rows if row["category"] == "tp"],
        "fp": [row for row in sample_rows if row["category"] == "fp"],
    }
    selections = {
        "largest_span_tp": [
            int(row["sample_index"])
            for row in sorted(
                by_category["tp"],
                key=lambda value: (-float(value["focus_probability_span"]), int(value["sample_index"])),
            )[:8]
        ],
        "largest_span_fp": [
            int(row["sample_index"])
            for row in sorted(
                by_category["fp"],
                key=lambda value: (-float(value["focus_probability_span"]), int(value["sample_index"])),
            )[:8]
        ],
        "decision_exit_tp": [
            int(row["sample_index"])
            for row in sorted(
                [value for value in by_category["tp"] if bool(value["any_class1_exit"])],
                key=lambda value: (-float(value["focus_probability_span"]), int(value["sample_index"])),
            )[:32]
        ],
        "decision_exit_fp": [
            int(row["sample_index"])
            for row in sorted(
                [value for value in by_category["fp"] if bool(value["any_class1_exit"])],
                key=lambda value: (-float(value["focus_probability_span"]), int(value["sample_index"])),
            )[:32]
        ],
    }
    selected_indices = sorted({index for values in selections.values() for index in values})
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=selected_indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=int(args.num_workers),
        context="lpd_visuals",
    )
    clean_images: Dict[int, Tensor] = {}
    clean_bboxes: Dict[int, Tensor] = {}
    for images, _, metadata in loader:
        sample_indices = metadata.get("sample_index")
        crop_bbox = metadata.get("crop_bbox")
        if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bbox):
            raise ValueError("LPD visual loader lacks sample_index/crop_bbox.")
        for position, value in enumerate(sample_indices.tolist()):
            clean_images[int(value)] = images[position].clone()
            clean_bboxes[int(value)] = crop_bbox[position, :4].float().clone()

    sample_lookup = {int(row["sample_index"]): row for row in sample_rows}
    shift_lookup = {
        (int(row["sample_index"]), str(row["condition"])): row
        for row in shift_rows
    }
    page_dir = output_dir / "visuals"
    page_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    panel_width, panel_height = 360, 305
    columns = 4
    font = ImageFont.load_default()
    for category, indices in selections.items():
        rows_count = max(1, math.ceil(len(indices) / columns))
        canvas = Image.new(
            "RGB",
            (columns * panel_width, rows_count * panel_height + 36),
            "white",
        )
        title_draw = ImageDraw.Draw(canvas)
        title_draw.text((8, 10), category, fill="black", font=font)
        for panel_index, sample_index in enumerate(indices):
            summary = sample_lookup[sample_index]
            condition = str(summary["adverse_condition"])
            dx, dy = int(summary["adverse_dx"]), int(summary["adverse_dy"])
            clean_tensor = clean_images[sample_index]
            shifted_tensor = _translate_tensor(
                clean_tensor.unsqueeze(0), dx=dx, dy=dy, replicate=True
            )[0]
            clean_bbox = clean_bboxes[sample_index]
            shifted_bbox = _translate_bbox(
                clean_bbox.unsqueeze(0), dx=dx, dy=dy
            )[0]
            clean_pil = _overlay_bbox(
                _tensor_to_pil(clean_tensor, semantics), clean_bbox.tolist(), "lime"
            ).resize((160, 160), Image.Resampling.BILINEAR)
            shifted_pil = _overlay_bbox(
                _tensor_to_pil(shifted_tensor, semantics), shifted_bbox.tolist(), "red"
            ).resize((160, 160), Image.Resampling.BILINEAR)
            column = panel_index % columns
            row_index = panel_index // columns
            x0 = column * panel_width + 10
            y0 = row_index * panel_height + 42
            canvas.paste(clean_pil, (x0, y0))
            canvas.paste(shifted_pil, (x0 + 174, y0))
            clean_record = shift_lookup[(sample_index, "clean")]
            shifted_record = shift_lookup[(sample_index, condition)]
            lines = [
                f"idx={sample_index} fold={summary['fold']} target={summary['target']} {summary['category']}",
                f"shift=({dx},{dy}) exit={summary['any_class1_exit']}",
                f"pred {clean_record['prediction']}->{shifted_record['prediction']}",
                f"p1 {float(clean_record['focus_probability']):.5f}->{float(shifted_record['focus_probability']):.5f}",
                f"margin {float(clean_record['focus_margin']):.5f}->{float(shifted_record['focus_margin']):.5f}",
                f"span={float(summary['focus_probability_span']):.5f}",
            ]
            draw = ImageDraw.Draw(canvas)
            for line_index, line in enumerate(lines):
                draw.text(
                    (x0, y0 + 168 + line_index * 18),
                    line,
                    fill="black",
                    font=font,
                )
        page_path = page_dir / f"{category}.png"
        canvas.save(page_path, format="PNG", optimize=False)
        pages.append(
            {
                "category": category,
                "path": str(page_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": _sha256(page_path),
                "bytes": int(page_path.stat().st_size),
                "rows": len(indices),
            }
        )

    selection_path = page_dir / "selection.json"
    _write_json(
        selection_path,
        {
            "method": METHOD,
            "selection_rule": "largest focus-probability span; exits first for exit pages",
            "categories": selections,
            "loader": loader_summary,
        },
    )
    expected_exit_tp = min(
        32,
        sum(
            row["category"] == "tp" and bool(row["any_class1_exit"])
            for row in sample_rows
        ),
    )
    expected_exit_fp = min(
        32,
        sum(
            row["category"] == "fp" and bool(row["any_class1_exit"])
            for row in sample_rows
        ),
    )
    complete = bool(
        len(pages) == 4
        and len(selections["largest_span_tp"]) == 8
        and len(selections["largest_span_fp"]) == 8
        and len(selections["decision_exit_tp"]) == expected_exit_tp
        and len(selections["decision_exit_fp"]) == expected_exit_fp
        and all(page["bytes"] > 0 for page in pages)
    )
    return {
        "status": "pending_manual_review",
        "complete": complete,
        "pages": pages,
        "selection_path": str(selection_path.relative_to(output_dir)).replace("\\", "/"),
        "selection_sha256": _sha256(selection_path),
        "selected_unique_rows": len(selected_indices),
        "manual_result": None,
        "manual_note": "",
    }


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    files = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload = {
        "method": METHOD,
        "files": files,
        "file_count": len(files),
        "payload_bytes": sum(int(value["bytes"]) for value in files),
        "binary_model_payloads_retained": False,
        "onnx_payload_retained": False,
        "tensorrt_engine_retained": False,
        "raw_data_modified": False,
    }
    _write_json(manifest_path, payload)
    return {**payload, "manifest_sha256": _sha256(manifest_path)}


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    equation_gate = summary.get("equation_gate", {})
    structural = summary.get("structural_gate", {})
    signal = summary.get("shift_signal_gate", {})
    shift = summary.get("shift_summary", {})
    resource = summary.get("resource", {})
    lines = [
        "# Learnable Polyphase Downsampling A0 Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Equation gate: `{equation_gate.get('passed')}`",
        f"- Structural/deployment gate: `{structural.get('passed')}`",
        f"- Shift-signal gate: `{signal.get('passed')}`",
        f"- Automated gate: `{summary.get('automated_gate_passed')}`",
        f"- Validation used: `{summary.get('validation_data_used')}`",
        f"- Test used: `{summary.get('test_data_used')}`",
        f"- Image epoch used: `{summary.get('image_model_training_used')}`",
        "",
        "## Shift Evidence",
        "",
        f"- Class-1 exits: `{shift.get('any_class1_exit_count')}`",
        f"- TP exits: `{shift.get('tp_exit_count')}`",
        f"- Restricted-FP exits: `{shift.get('restricted_fp_exit_count')}`",
        f"- Median class-1 probability span: `{shift.get('focus_probability_span_median')}`",
        f"- P90 class-1 probability span: `{shift.get('focus_probability_span_p90')}`",
        "",
        "## Engineering",
        "",
        f"- Runtime ratio: `{resource.get('runtime_ratio')}`",
        f"- Peak-memory ratio: `{resource.get('peak_memory_ratio')}`",
        f"- Parameter ratio: `{resource.get('model_declaration', {}).get('parameter_ratio') if isinstance(resource, Mapping) else None}`",
        "",
        "## Decision",
        "",
        (
            "A0 awaits hash-locked visual review. Even a pass authorizes only a separate matched scratch protocol."
            if summary.get("status") == "awaiting_visual_review"
            else "The fail-closed status above controls this route."
        ),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_equation_rejection(
    *,
    output_dir: Path,
    provenance: Mapping[str, object],
    equation: Mapping[str, object],
    equation_gate: Mapping[str, object],
    process_before: Mapping[str, object],
    gpu_before: Mapping[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    summary = {
        "method": METHOD,
        "status": "rejected_equation_gate_before_keeper_shift",
        "provenance": provenance,
        "process_before": process_before,
        "gpu_before": gpu_before,
        "equation": equation,
        "equation_gate": equation_gate,
        "resource": {},
        "onnx": {},
        "structural_gate": {"passed": False, "not_run": True},
        "shift_summary": {},
        "shift_signal_gate": {"passed": False, "not_run": True},
        "independent_csv_replay": {"passed": False, "not_run": True},
        "visual_review": {"complete": False, "not_run": True},
        "automated_gate_passed": False,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "matched_scratch_protocol_authorized": False,
        "production_integration_authorized": False,
        "current_command_update_authorized": False,
        "raw_data_modified": False,
        "elapsed_seconds": float(elapsed_seconds),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "formal_summary_sha256": _sha256(output_dir / "summary.json"),
        "artifact_manifest": manifest,
    }


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Formal LPD summary not found: {summary_path}")
    expected = str(args.expected_summary_sha256).strip().casefold()
    if not expected:
        raise ValueError("--expected-summary-sha256 is required for finalization.")
    observed = _sha256(summary_path)
    if observed != expected:
        raise ValueError(f"Formal summary SHA differs: {observed} != {expected}")
    if args.visual_review_result not in {"pass", "fail"}:
        raise ValueError("--visual-review-result pass|fail is required.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError(f"Summary is not awaiting visual review: {summary.get('status')}")
    visual = summary.get("visual_review")
    if not isinstance(visual, Mapping) or not bool(visual.get("complete")):
        raise ValueError("Visual artifact is incomplete and cannot be finalized.")
    pages = visual.get("pages")
    if not isinstance(pages, list) or len(pages) != 4:
        raise ValueError("LPD visual page declaration differs from four pages.")
    for page in pages:
        page_path = output_dir / str(page["path"])
        if _sha256(page_path) != str(page["sha256"]):
            raise ValueError(f"LPD visual page hash differs: {page_path}")
    updated_visual = dict(visual)
    updated_visual.update(
        {
            "status": "reviewed",
            "manual_result": str(args.visual_review_result),
            "manual_note": str(args.visual_review_note),
            "reviewed_summary_sha256": observed,
        }
    )
    summary["visual_review"] = updated_visual
    final_pass = bool(summary.get("automated_gate_passed")) and (
        args.visual_review_result == "pass"
    )
    summary["status"] = "passed" if final_pass else "rejected"
    summary["matched_scratch_protocol_authorized"] = final_pass
    summary["production_integration_authorized"] = False
    summary["validation_data_used"] = False
    summary["test_data_used"] = False
    summary["image_model_training_used"] = False
    summary["current_command_update_authorized"] = False
    _write_json(summary_path, summary)
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir)
    return {
        "method": METHOD,
        "status": summary["status"],
        "summary_sha256": _sha256(summary_path),
        "report_sha256": _sha256(output_dir / "report.md"),
        "artifact_manifest": manifest,
        "matched_scratch_protocol_authorized": final_pass,
        "production_integration_authorized": False,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if bool(args.finalize_visual_review):
        return _finalize_visual_review(args)
    started = time.perf_counter()
    provenance, rows, cohort = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "preflight_created_output": False,
            "provenance": provenance,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Locked LPD A0 requires CUDA.")
    process_before = _process_snapshot()
    gpu_before = _gpu_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "LPD A0 found another python.exe/trtexec.exe; no process was terminated."
        )
    if not gpu_before["gpus"] or any(
        int(row["utilization_percent"]) > 15
        or int(row["memory_used_mib"]) > 2200
        for row in gpu_before["gpus"]
    ):
        raise RuntimeError("LPD A0 GPU is not isolated; no timing decision was made.")

    output_dir = _prepare_output_dir(args.output_dir)
    device = torch.device("cuda")
    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    paths = {name: Path(value) for name, value in provenance["paths"].items()}
    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    base_dataset, transform, dataset_audit = _build_dataset(
        checkpoint,
        rows,
        paths["data"],
    )
    cohort_indices = [row.sample_index for row in cohort]
    loader, loader_audit = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=cohort_indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="lpd_shift_cohort",
    )

    equation = _equation_diagnostics(paths, device)
    equation_gate = assess_equation_gate(equation)
    if not bool(equation_gate["passed"]):
        return _write_equation_rejection(
            output_dir=output_dir,
            provenance=provenance,
            equation=equation,
            equation_gate=equation_gate,
            process_before=process_before,
            gpu_before=gpu_before,
            elapsed_seconds=time.perf_counter() - started,
        )

    native, candidate, model_declaration = _random_model_pair(checkpoint)
    resource = _resource_audit(
        native=native,
        candidate=candidate,
        model_declaration=model_declaration,
        device=device,
        repeats=int(args.benchmark_repeats),
    )
    candidate_stem = getattr(candidate, "stem", None)
    if not isinstance(candidate_stem, LPDStemTrace):
        raise TypeError("Resource candidate lost the isolated LPD stem.")
    onnx = _onnx_audit(candidate_stem)
    del native, candidate
    gc.collect()
    torch.cuda.empty_cache()

    keeper = _load_keeper_model(checkpoint)
    shift_rows, transform_audit, real_selector = _shift_inference(
        model=keeper,
        loader=loader,
        cohort=cohort,
        device=device,
    )
    shift_summary, sample_rows = summarize_shift_rows(shift_rows)
    shift_signal_gate = assess_shift_signal(shift_summary)
    shift_csv = output_dir / "shift_predictions.csv"
    sample_csv = output_dir / "shift_sample_summary.csv"
    _write_shift_csv(shift_csv, shift_rows)
    _write_sample_csv(sample_csv, sample_rows)
    replay_rows = _read_shift_csv(shift_csv)
    replay_summary, replay_samples = summarize_shift_rows(replay_rows)
    replay_differences = _numeric_differences(
        shift_summary, replay_summary, path="shift_summary"
    )
    sample_differences = _numeric_differences(
        sample_rows, replay_samples, path="sample_rows"
    )
    independent_replay = {
        "rows": len(replay_rows),
        "summary_difference_count": len(replay_differences),
        "sample_difference_count": len(sample_differences),
        "differences": [*replay_differences[:10], *sample_differences[:10]],
        "max_abs_numeric_difference": max(
            [
                float(value.get("abs_delta", 0.0))
                for value in [*replay_differences, *sample_differences]
            ]
            or [0.0]
        ),
        "passed": not replay_differences and not sample_differences,
    }

    process_after = _process_snapshot()
    gpu_after = _gpu_snapshot()
    structural_gate = assess_structural_gate(
        equation_gate=equation_gate,
        resource=resource,
        onnx=onnx,
        transform_audit=transform_audit,
        real_selector=real_selector,
        process_before=process_before,
        gpu_before=gpu_before,
    )
    visual_review = _render_visuals(
        output_dir=output_dir,
        base_dataset=base_dataset,
        transform=transform,
        semantics=dataset_audit["semantics"],
        sample_rows=sample_rows,
        shift_rows=shift_rows,
        args=args,
    )
    automated_gate = bool(
        equation_gate["passed"]
        and structural_gate["passed"]
        and shift_signal_gate["passed"]
        and independent_replay["passed"]
        and visual_review["complete"]
        and not process_after["unexpected_processes"]
    )
    summary = {
        "method": METHOD,
        "status": "awaiting_visual_review",
        "provenance": provenance,
        "dataset_audit": dataset_audit,
        "loader_audit": loader_audit,
        "process_before": process_before,
        "gpu_before": gpu_before,
        "process_after": process_after,
        "gpu_after": gpu_after,
        "equation": equation,
        "equation_gate": equation_gate,
        "resource": resource,
        "onnx": onnx,
        "transform_audit": transform_audit,
        "real_selector_diagnostics": real_selector,
        "structural_gate": structural_gate,
        "shift_summary": shift_summary,
        "shift_signal_gate": shift_signal_gate,
        "independent_csv_replay": independent_replay,
        "visual_review": visual_review,
        "automated_gate_passed": automated_gate,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "matched_scratch_protocol_authorized": False,
        "production_integration_authorized": False,
        "current_command_update_authorized": False,
        "raw_data_modified": False,
        "binary_model_payload_retained": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "formal_summary_sha256": _sha256(output_dir / "summary.json"),
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(
        json.dumps(
            {
                "method": result.get("method"),
                "status": result.get("status"),
                "formal_summary_sha256": result.get(
                    "formal_summary_sha256", result.get("summary_sha256")
                ),
                "automated_gate_passed": result.get("automated_gate_passed"),
                "matched_scratch_protocol_authorized": result.get(
                    "matched_scratch_protocol_authorized", False
                ),
                "production_integration_authorized": result.get(
                    "production_integration_authorized", False
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from typing import Dict, Mapping, Optional, Sequence
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import BBoxSpatialPriorFusion, HybridConvStem
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
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "fcanet_object_frequency_signal_a0"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
BENCHMARK_REPEATS = 7
READOUT_C = 0.05
READOUT_MAX_ITER = 2000
READOUT_TOLERANCE = 1e-9
MIN_FIT_TP_RETENTION = 0.90
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
LOCKED_BBOX_BYTES_SHA256 = (
    "e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b"
)
LOCKED_GEOMETRY = {
    16: {
        "object_min": 28,
        "object_max": 210,
        "context_min": 46,
        "context_max": 228,
    },
    32: {
        "object_min": 116,
        "object_max": 900,
        "context_min": 124,
        "context_max": 908,
    },
}
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
ROLES = (
    "full_gap",
    "object_gap",
    "full_top16",
    "object_top16",
    "bbox_geometry",
)
GRID_SIZE = 16
CHANNELS = 256
PATCH_TOKENS = GRID_SIZE * GRID_SIZE
FREQUENCY_GROUPS = 16
CHANNELS_PER_FREQUENCY = CHANNELS // FREQUENCY_GROUPS
MAX_OFFICIAL_ERROR = 1e-6
MAX_ORACLE_ERROR = 1e-12
MAX_GRADIENT_ERROR = 1e-8
MAX_FINITE_DIFFERENCE_ERROR = 1e-5
MAX_BF16_ERROR = 0.05
MAX_GAP_ERROR = 1e-6
MAX_BBOX_GEOMETRY_ERROR = 1e-7
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.15
MAX_MEMORY_RATIO = 1.10

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "b28f4330c3cbaddf82cca9c0da222103a4199b7083b0cfde914b4607cc726283"
LOCKED_PAPER_SHA256 = "13c707b575722cdb81003ac2e1bcd6650eb2ef5984265666d495662766f69d55"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "aa5fb63505575bb4e4e094613565379c3f6ada33"
LOCKED_OFFICIAL_TREE = "85aa7989ca7d957e1ab87991c65eefdb7fb53a8d"
LOCKED_OFFICIAL_HASHES = {
    "layer_source": "9b4337558604958dca1257f1b05e92fbee8f546e536604de5cff83f2a3d0200e",
    "license": "31e61e165ea1409c53836aadf965a601a0c225aa7c1aaa4eb0c01e18559e4cf1",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked fit-only FcaNet object-frequency signal gate. Validation, "
            "test, trainer integration, and raw-data edits are forbidden."
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
            "docs/TRKH_5CLASS_FCANET_OBJECT_FREQUENCY_SIGNAL_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\fcanet_iccv2021.pdf"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\fcanet-iccv2021"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_fcanet_object_frequency_signal_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--finalize-visual-review", action="store_true", default=False)
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
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
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "layer_source": official / "model" / "layer.py",
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
        raise ValueError("Arguments differ from the locked FcaNet A0 protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "FcaNet protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "accepted FcaNet paper"
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
            paths[name], expected, f"official FcaNet {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(
        paths["official_root"], "rev-parse", "HEAD^{tree}"
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(f"Official FcaNet commit differs: {official_commit}")
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official FcaNet tree differs: {official_tree}")
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official FcaNet worktree must be clean.")

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
            "Locked FcaNet cohort differs: "
            f"{len(cohort)}/{positives}/{negatives}"
        )
    fold_counts = {fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS}
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked FcaNet fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for formal FcaNet A0.")
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal FcaNet A0 requires the pushed repository commit.")

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


def get_frequency_indices(method: str) -> tuple[list[int], list[int]]:
    if method not in {"top1", "top16"}:
        raise ValueError(f"Unsupported locked frequency method: {method}")
    all_top_x = [
        0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2,
        4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1,
    ]
    all_top_y = [
        0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2,
        6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3,
    ]
    count = int(method[3:])
    return all_top_x[:count], all_top_y[:count]


def _build_filter(position: int, frequency: int, size: int) -> float:
    result = math.cos(math.pi * frequency * (position + 0.5) / size)
    result /= math.sqrt(size)
    return result if frequency == 0 else result * math.sqrt(2.0)


def fixed_dct_filter(
    height: int,
    width: int,
    mapper_x: Sequence[int],
    mapper_y: Sequence[int],
    channels: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if len(mapper_x) != len(mapper_y) or not mapper_x:
        raise ValueError("DCT mapper axes must have the same nonzero length.")
    if int(channels) % len(mapper_x) != 0:
        raise ValueError("Channels must divide evenly across DCT frequencies.")
    result = torch.zeros(int(channels), int(height), int(width), dtype=dtype)
    channels_per_frequency = int(channels) // len(mapper_x)
    for group, (frequency_x, frequency_y) in enumerate(zip(mapper_x, mapper_y)):
        start = group * channels_per_frequency
        stop = start + channels_per_frequency
        for row in range(int(height)):
            for column in range(int(width)):
                result[start:stop, row, column] = _build_filter(
                    row, int(frequency_x), int(height)
                ) * _build_filter(column, int(frequency_y), int(width))
    return result


class MultiSpectralDCTLayer(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        mapper_x: Sequence[int],
        mapper_y: Sequence[int],
        channels: int,
    ) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.register_buffer(
            "weight",
            fixed_dct_filter(
                self.height,
                self.width,
                mapper_x,
                mapper_y,
                self.channels,
            ),
        )

    def forward(self, features: Tensor) -> Tensor:
        if tuple(features.shape[1:]) != (
            self.channels,
            self.height,
            self.width,
        ):
            raise ValueError(
                "DCT feature shape differs from the locked map: "
                f"{tuple(features.shape)}"
            )
        if features.dtype in {torch.float16, torch.bfloat16}:
            return (features.float() * self.weight.float()).sum(dim=(2, 3))
        return (features * self.weight.to(dtype=features.dtype)).sum(dim=(2, 3))


def object_sampling_grid(
    bboxes: Tensor,
    *,
    output_height: int = GRID_SIZE,
    output_width: int = GRID_SIZE,
) -> Tensor:
    if bboxes.ndim != 2 or int(bboxes.size(1)) < 4:
        raise ValueError("Object alignment requires normalized xywh bboxes [B,4+].")
    boxes = bboxes[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = boxes.unbind(dim=1)
    if bool((width <= 0.0).any() or (height <= 0.0).any()):
        raise ValueError("Object alignment bbox has nonpositive width or height.")
    left = (cx - width / 2.0).clamp(0.0, 1.0)
    right = (cx + width / 2.0).clamp(0.0, 1.0)
    top = (cy - height / 2.0).clamp(0.0, 1.0)
    bottom = (cy + height / 2.0).clamp(0.0, 1.0)
    x_fraction = (
        (torch.arange(output_width, device=boxes.device, dtype=torch.float32) + 0.5)
        / float(output_width)
    )
    y_fraction = (
        (torch.arange(output_height, device=boxes.device, dtype=torch.float32) + 0.5)
        / float(output_height)
    )
    x = left[:, None] + (right - left)[:, None] * x_fraction[None]
    y = top[:, None] + (bottom - top)[:, None] * y_fraction[None]
    grid_x = x[:, None, :].expand(-1, output_height, -1)
    grid_y = y[:, :, None].expand(-1, -1, output_width)
    return torch.stack((2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1)


def align_object_feature_map(
    features: Tensor,
    bboxes: Tensor,
    *,
    output_height: int = GRID_SIZE,
    output_width: int = GRID_SIZE,
) -> Tensor:
    compute_features = (
        features.float()
        if features.dtype in {torch.float16, torch.bfloat16}
        else features
    )
    grid = object_sampling_grid(
        bboxes,
        output_height=output_height,
        output_width=output_width,
    ).to(device=compute_features.device, dtype=compute_features.dtype)
    return F.grid_sample(
        compute_features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class ObjectAlignedDCTExtractor(nn.Module):
    def __init__(self, method: str = "top16") -> None:
        super().__init__()
        mapper_x, mapper_y = get_frequency_indices(method)
        scale = GRID_SIZE // 7
        scaled_x = [value * scale for value in mapper_x]
        scaled_y = [value * scale for value in mapper_y]
        self.method = str(method)
        self.dct = MultiSpectralDCTLayer(
            GRID_SIZE,
            GRID_SIZE,
            scaled_x,
            scaled_y,
            CHANNELS,
        )

    def forward(self, features: Tensor, bboxes: Tensor) -> Tensor:
        aligned = align_object_feature_map(features, bboxes)
        return self.dct(aligned)


def _center_mask_support(bboxes: Tensor, grid_size: int) -> tuple[Tensor, Tensor]:
    boxes = bboxes[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = boxes.unbind(dim=1)
    left = (cx - width / 2.0).clamp(0.0, 1.0)
    right = (cx + width / 2.0).clamp(0.0, 1.0)
    top = (cy - height / 2.0).clamp(0.0, 1.0)
    bottom = (cy + height / 2.0).clamp(0.0, 1.0)
    coordinates = (
        torch.arange(grid_size, dtype=torch.float32) + 0.5
    ) / float(grid_size)
    x = coordinates.view(1, 1, grid_size)
    y = coordinates.view(1, grid_size, 1)
    object_mask = (
        (x >= left[:, None, None])
        & (x < right[:, None, None])
        & (y >= top[:, None, None])
        & (y < bottom[:, None, None])
    )
    object_count = object_mask.flatten(1).sum(dim=1)
    context_count = (~object_mask).flatten(1).sum(dim=1)
    return object_count, context_count


def _geometry_preflight(
    *,
    checkpoint: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    cohort: Sequence[CleanTrainRow],
    data_path: Path,
    args: argparse.Namespace,
) -> tuple[object, object, Tensor, Dict[str, object]]:
    dataset, transform, dataset_summary = _build_dataset(checkpoint, rows, data_path)
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="fcanet_a0_geometry_cpu",
    )
    bbox_batches: list[Tensor] = []
    observed_indices: list[int] = []
    observed_targets: list[int] = []
    for _, targets, metadata in loader:
        sample_indices = metadata.get("sample_index")
        crop_bboxes = metadata.get("crop_bbox")
        if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bboxes):
            raise ValueError("FcaNet geometry loader lacks sample_index/crop_bbox.")
        observed_indices.extend(int(value) for value in sample_indices.tolist())
        observed_targets.extend(int(value) for value in targets.tolist())
        bbox_batches.append(crop_bboxes[:, :4].to(dtype=torch.float32).cpu())
    if observed_indices != indices or observed_targets != expected_targets:
        raise ValueError("FcaNet geometry loader order/targets differ.")
    bboxes = torch.cat(bbox_batches, dim=0).contiguous()
    if tuple(bboxes.shape) != (EXPECTED_COHORT_ROWS, 4):
        raise ValueError(f"Unexpected locked bbox shape: {tuple(bboxes.shape)}")
    if not bool(torch.isfinite(bboxes).all()):
        raise ValueError("Locked FcaNet bbox geometry is non-finite.")
    bbox_hash = hashlib.sha256(bboxes.numpy().tobytes(order="C")).hexdigest()
    if bbox_hash != LOCKED_BBOX_BYTES_SHA256:
        raise ValueError(f"Locked bbox byte hash differs: {bbox_hash}")

    grids: Dict[str, object] = {}
    for grid_size, expected in LOCKED_GEOMETRY.items():
        object_count, context_count = _center_mask_support(bboxes, grid_size)
        observed = {
            "object_min": int(object_count.min()),
            "object_max": int(object_count.max()),
            "context_min": int(context_count.min()),
            "context_max": int(context_count.max()),
        }
        if observed != expected:
            raise ValueError(
                f"Locked {grid_size}x{grid_size} support differs: {observed}"
            )
        if bool((object_count == 0).any() or (context_count == 0).any()):
            raise ValueError(f"Empty object/context support at {grid_size}x{grid_size}.")
        grids[str(grid_size)] = {
            **observed,
            "object_empty": int((object_count == 0).sum()),
            "context_empty": int((context_count == 0).sum()),
        }
    return dataset, transform, bboxes, {
        "status": "passed",
        "model_inference_used": False,
        "rows": len(cohort),
        "bbox_shape": list(bboxes.shape),
        "bbox_bytes_sha256": bbox_hash,
        "bbox_width_min": float(bboxes[:, 2].min()),
        "bbox_width_max": float(bboxes[:, 2].max()),
        "bbox_height_min": float(bboxes[:, 3].min()),
        "bbox_height_max": float(bboxes[:, 3].max()),
        "grids": grids,
        "dataset": dataset_summary,
        "loader": loader_summary,
    }


def _load_official_dct_class(source_path: Path):
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef) and node.name == "get_freq_indices"
        )
        or (
            isinstance(node, ast.ClassDef) and node.name == "MultiSpectralDCTLayer"
        )
    ]
    if len(selected) != 2:
        raise ValueError("Official FcaNet source lacks locked DCT definitions.")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"math": math, "torch": torch, "nn": nn}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["get_freq_indices"], namespace["MultiSpectralDCTLayer"]


def _independent_dct(features: Tensor, weight: Tensor) -> Tensor:
    rows = []
    for batch_index in range(int(features.size(0))):
        channels = []
        for channel_index in range(int(features.size(1))):
            channels.append(
                torch.dot(
                    features[batch_index, channel_index].reshape(-1),
                    weight[channel_index].reshape(-1),
                )
            )
        rows.append(torch.stack(channels))
    return torch.stack(rows)


def _equation_diagnostics(source_path: Path, device: torch.device) -> Dict[str, object]:
    official_get_indices, official_dct_class = _load_official_dct_class(source_path)
    official_x, official_y = official_get_indices("top16")
    local_x, local_y = get_frequency_indices("top16")
    index_exact = list(official_x) == local_x and list(official_y) == local_y
    scale = GRID_SIZE // 7
    scaled_x = [value * scale for value in local_x]
    scaled_y = [value * scale for value in local_y]

    local = MultiSpectralDCTLayer(
        GRID_SIZE, GRID_SIZE, scaled_x, scaled_y, CHANNELS
    ).to(dtype=torch.float64)
    official = official_dct_class(
        GRID_SIZE, GRID_SIZE, scaled_x, scaled_y, CHANNELS
    ).to(dtype=torch.float64)
    generator = torch.Generator().manual_seed(SEED + 101)
    local_input = torch.randn(
        3,
        CHANNELS,
        GRID_SIZE,
        GRID_SIZE,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    official_input = local_input.detach().clone().requires_grad_(True)
    local_output = local(local_input)
    official_output = official(official_input)
    oracle_output = _independent_dct(local_input, local.weight)
    probe = torch.randn(local_output.shape, generator=generator, dtype=torch.float64)
    local_gradient = torch.autograd.grad((local_output * probe).sum(), local_input)[0]
    official_gradient = torch.autograd.grad(
        (official_output * probe).sum(), official_input
    )[0]

    coordinate = (0, 0, 0, 0)
    epsilon = 1e-5
    finite_input = local_input.detach().clone()
    original = float(finite_input[coordinate])
    finite_input[coordinate] = original + epsilon
    plus = float((local(finite_input) * probe).sum())
    finite_input[coordinate] = original - epsilon
    minus = float((local(finite_input) * probe).sum())
    numerical = (plus - minus) / (2.0 * epsilon)
    analytic = float(local_gradient[coordinate])

    gap_x, gap_y = get_frequency_indices("top1")
    gap = MultiSpectralDCTLayer(
        GRID_SIZE,
        GRID_SIZE,
        gap_x,
        gap_y,
        CHANNELS,
    ).to(dtype=torch.float64)
    gap_descriptor = gap(local_input.detach())
    direct_gap = local_input.detach().mean(dim=(2, 3))
    gap_scale = math.sqrt(float(GRID_SIZE * GRID_SIZE))
    gap_error = float((gap_descriptor - direct_gap * gap_scale).abs().max())

    bf16_features = torch.randn(
        4,
        CHANNELS,
        GRID_SIZE,
        GRID_SIZE,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    bf16_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.70, 0.80],
            [0.42, 0.55, 0.45, 0.35],
            [0.60, 0.40, 0.85, 0.60],
            [0.35, 0.65, 0.30, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )
    fp32_extractor = ObjectAlignedDCTExtractor("top16").to(device).eval()
    fp32_input = bf16_features.detach().clone().requires_grad_(True)
    fp32_output = fp32_extractor(fp32_input, bf16_boxes)
    fp32_gradient = torch.autograd.grad(fp32_output.square().mean(), fp32_input)[0]
    bf16_extractor = copy.deepcopy(fp32_extractor).to(dtype=torch.bfloat16)
    bf16_input = bf16_features.to(dtype=torch.bfloat16).requires_grad_(True)
    bf16_output = bf16_extractor(bf16_input, bf16_boxes)
    bf16_gradient = torch.autograd.grad(
        bf16_output.float().square().mean(), bf16_input
    )[0]
    canonical_grid = object_sampling_grid(bf16_boxes)
    repeated_grid = object_sampling_grid(bf16_boxes)
    bf16_bbox_dtype = str(bf16_boxes.dtype)
    bbox_control = BBoxSpatialPriorFusion(num_classes=5).to(device).eval()
    bbox_control_descriptor = bbox_control.extract_stats(bf16_boxes)
    bbox_local_descriptor = bbox_geometry_descriptor(bf16_boxes)
    bbox_geometry_error = float(
        (bbox_control_descriptor - bbox_local_descriptor).abs().max()
    )
    bf16_error = float(
        (fp32_output.detach() - bf16_output.detach().float()).abs().max()
    )
    fp32_extractor.cpu()
    bf16_extractor.cpu()
    bbox_control.cpu()
    del bf16_features, bf16_boxes, fp32_input, bf16_input, bbox_control
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "official_frequency_indices_exact": index_exact,
        "official_output_max_abs_error": float(
            (local_output.detach() - official_output.detach()).abs().max()
        ),
        "oracle_output_max_abs_error": float(
            (local_output.detach() - oracle_output.detach()).abs().max()
        ),
        "official_weight_max_abs_error": float(
            (local.weight.detach() - official.weight.detach()).abs().max()
        ),
        "gradient_max_abs_error": float(
            (local_gradient.detach() - official_gradient.detach()).abs().max()
        ),
        "finite_difference_error": abs(numerical - analytic),
        "finite_difference_analytic": analytic,
        "finite_difference_numerical": numerical,
        "gap_proportionality_max_abs_error": gap_error,
        "gap_scale": gap_scale,
        "bbox_geometry_control_max_abs_error": bbox_geometry_error,
        "bf16_max_abs_error": bf16_error,
        "bf16_bbox_dtype": bf16_bbox_dtype,
        "fp32_gradient_finite": bool(torch.isfinite(fp32_gradient).all()),
        "fp32_gradient_nonzero": int(torch.count_nonzero(fp32_gradient)) > 0,
        "bf16_gradient_finite": bool(torch.isfinite(bf16_gradient).all()),
        "bf16_gradient_nonzero": int(torch.count_nonzero(bf16_gradient)) > 0,
        "object_grid_repeat_exact": bool(torch.equal(canonical_grid, repeated_grid)),
        "all_outputs_finite": bool(
            torch.isfinite(local_output).all()
            and torch.isfinite(official_output).all()
            and torch.isfinite(oracle_output).all()
            and torch.isfinite(fp32_output).all()
            and torch.isfinite(bf16_output).all()
        ),
        "scaled_frequency_x": scaled_x,
        "scaled_frequency_y": scaled_y,
    }


def bbox_geometry_descriptor(bboxes: Tensor) -> Tensor:
    if bboxes.ndim != 2 or int(bboxes.size(1)) < 4:
        raise ValueError("BBox geometry descriptor requires [B,4+].")
    bbox = bboxes[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
    x_center, y_center, box_width, box_height = bbox.unbind(dim=1)
    half_width = box_width * 0.5
    half_height = box_height * 0.5
    left = (x_center - half_width).clamp(0.0, 1.0)
    top = (y_center - half_height).clamp(0.0, 1.0)
    right = (x_center + half_width).clamp(0.0, 1.0)
    bottom = (y_center + half_height).clamp(0.0, 1.0)
    area = (box_width * box_height).clamp(0.0, 1.0)
    sqrt_area = torch.sqrt(area.clamp(min=1e-8))
    aspect_log = torch.log((box_width + 1e-6) / (box_height + 1e-6))
    aspect_log = aspect_log.clamp(-4.0, 4.0) / 4.0
    x_offset = x_center - 0.5
    y_offset = y_center - 0.5
    center_distance = torch.sqrt(
        (x_offset * x_offset + y_offset * y_offset).clamp(min=0.0)
    ) / math.sqrt(0.5)
    border_values = torch.stack((left, top, 1.0 - right, 1.0 - bottom), dim=1)
    border_min = border_values.amin(dim=1)
    border_max = border_values.amax(dim=1)
    angle = 2.0 * math.pi
    return torch.stack(
        (
            x_center,
            y_center,
            box_width,
            box_height,
            left,
            top,
            right,
            bottom,
            area,
            sqrt_area,
            aspect_log,
            x_offset,
            y_offset,
            center_distance.clamp(0.0, 1.0),
            border_min.clamp(0.0, 1.0),
            border_max.clamp(0.0, 1.0),
            torch.sin(x_center * angle),
            torch.cos(y_center * angle),
        ),
        dim=1,
    )


def _declaration_replay(
    *,
    model: nn.Module,
    dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="fcanet_a0_declaration_replay",
    )
    model = model.to(device).eval()
    observed_rows: list[Dict[str, object]] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Declaration replay lacks sample_index metadata.")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                logits, _ = _forward_classification_with_metadata(
                    model,
                    images,
                    metadata,
                    device=device,
                )
            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)
            for batch_index in range(int(images.size(0))):
                probability = probabilities[batch_index]
                top_values, top_indices = probability.topk(k=2)
                observed_rows.append(
                    {
                        "sample_index": int(sample_indices[batch_index].item()),
                        "target": int(targets[batch_index].item()),
                        "prediction": int(predictions[batch_index].item()),
                        "top1": int(top_indices[0].item()),
                        "top2": int(top_indices[1].item()),
                        "top1_probability": float(top_values[0].item()),
                        "top2_probability": float(top_values[1].item()),
                        "top1_top2_margin": float(
                            (top_values[0] - top_values[1]).item()
                        ),
                    }
                )
    if len(observed_rows) != len(cohort):
        raise ValueError("Declaration replay row count differs.")
    mismatches: list[Dict[str, object]] = []
    order_exact = True
    for expected, observed in zip(cohort, observed_rows):
        if (
            int(observed["sample_index"]) != expected.sample_index
            or int(observed["target"]) != expected.target
        ):
            order_exact = False
        if int(observed["prediction"]) != expected.keeper_prediction:
            mismatches.append(
                {
                    **observed,
                    "expected_prediction": expected.keeper_prediction,
                    "observed_prediction": observed["prediction"],
                }
            )
    exception_exact = bool(
        len(mismatches) == 1
        and all(
            int(mismatches[0][key]) == int(value)
            for key, value in LOCKED_DECLARATION_EXCEPTION.items()
        )
    )
    return {
        "rows": len(observed_rows),
        "batch_size": int(args.batch_size),
        "dtype": "bfloat16_autocast",
        "normal_evaluator_forward": True,
        "order_and_targets_exact": order_exact,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "locked_exception_exact": exception_exact,
        "passed": bool(order_exact and exception_exact),
        "loader": loader_summary,
    }


def _patch_map(model: nn.Module, images: Tensor) -> Tensor:
    stem = getattr(model, "stem", None)
    patch_embed = getattr(model, "patch_embed", None)
    if not isinstance(stem, HybridConvStem) or patch_embed is None:
        raise TypeError("FcaNet A0 requires the locked hybrid stem and patch embed.")
    stem_features = stem(images)
    patch_tokens = patch_embed(stem_features)
    if tuple(patch_tokens.shape[1:]) != (PATCH_TOKENS, CHANNELS):
        raise ValueError(
            f"Locked patch embedding shape differs: {tuple(patch_tokens.shape)}"
        )
    return (
        patch_tokens.reshape(
            int(patch_tokens.size(0)), GRID_SIZE, GRID_SIZE, CHANNELS
        )
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _extract_descriptors(
    *,
    model: nn.Module,
    dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    locked_bboxes: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, object]]:
    model = model.to(device).eval()
    gap_x, gap_y = get_frequency_indices("top1")
    top_x, top_y = get_frequency_indices("top16")
    scale = GRID_SIZE // 7
    gap_dct = MultiSpectralDCTLayer(
        GRID_SIZE, GRID_SIZE, gap_x, gap_y, CHANNELS
    ).to(device).eval()
    top_dct = MultiSpectralDCTLayer(
        GRID_SIZE,
        GRID_SIZE,
        [value * scale for value in top_x],
        [value * scale for value in top_y],
        CHANNELS,
    ).to(device).eval()
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    descriptors: Dict[str, Dict[str, np.ndarray]] = {}
    loader_summaries: Dict[str, object] = {}

    with torch.inference_mode():
        for condition, brightness, contrast in CONDITIONS:
            loader, loader_summary = _make_condition_loader(
                base_dataset=dataset,
                transform=transform,
                indices=indices,
                brightness=brightness,
                contrast=contrast,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context=f"fcanet_a0_descriptor_{condition}",
            )
            loader_summaries[condition] = loader_summary
            role_batches: Dict[str, list[Tensor]] = {role: [] for role in ROLES}
            bbox_batches: list[Tensor] = []
            observed_indices: list[int] = []
            observed_targets: list[int] = []
            for images, targets, metadata in loader:
                sample_indices = metadata.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bboxes):
                    raise ValueError(
                        f"{condition} descriptor loader lacks sample_index/crop_bbox."
                    )
                images = images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                boxes = crop_bboxes[:, :4].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                full_map = _patch_map(model, images)
                object_map = align_object_feature_map(full_map, boxes)
                role_batches["full_gap"].append(gap_dct(full_map).cpu())
                role_batches["object_gap"].append(gap_dct(object_map).cpu())
                role_batches["full_top16"].append(top_dct(full_map).cpu())
                role_batches["object_top16"].append(top_dct(object_map).cpu())
                role_batches["bbox_geometry"].append(
                    bbox_geometry_descriptor(boxes).cpu()
                )
                bbox_batches.append(boxes.cpu())
                observed_indices.extend(int(value) for value in sample_indices.tolist())
                observed_targets.extend(int(value) for value in targets.tolist())
            if observed_indices != indices or observed_targets != expected_targets:
                raise ValueError(f"{condition} descriptor order/targets differ.")
            observed_bboxes = torch.cat(bbox_batches, dim=0).contiguous()
            if not torch.equal(observed_bboxes, locked_bboxes):
                raise ValueError(f"{condition} transformed bbox differs from CPU lock.")
            condition_roles: Dict[str, np.ndarray] = {}
            for role in ROLES:
                values = torch.cat(role_batches[role], dim=0).float().numpy()
                expected_width = 18 if role == "bbox_geometry" else CHANNELS
                if values.shape != (EXPECTED_COHORT_ROWS, expected_width):
                    raise ValueError(
                        f"Unexpected {condition}/{role} descriptor shape: {values.shape}"
                    )
                if not bool(np.isfinite(values).all()):
                    raise ValueError(f"Non-finite {condition}/{role} descriptor.")
                condition_roles[role] = values.astype(np.float64)
            descriptors[condition] = condition_roles

    clean_top = descriptors["clean"]["object_top16"]
    clean_gap = descriptors["clean"]["object_gap"]
    group_variances = []
    for group in range(FREQUENCY_GROUPS):
        start = group * CHANNELS_PER_FREQUENCY
        stop = start + CHANNELS_PER_FREQUENCY
        group_variances.append(float(np.var(clean_top[:, start:stop])))
    telemetry = {
        "conditions": loader_summaries,
        "roles": list(ROLES),
        "descriptor_shapes": {
            condition: {
                role: list(values.shape) for role, values in role_values.items()
            }
            for condition, role_values in descriptors.items()
        },
        "all_finite": all(
            bool(np.isfinite(values).all())
            for role_values in descriptors.values()
            for values in role_values.values()
        ),
        "object_top16_gap_max_abs_delta": float(
            np.max(np.abs(clean_top - clean_gap))
        ),
        "frequency_group_variances": group_variances,
        "all_frequency_groups_nonzero_variance": all(
            value > 1e-12 for value in group_variances
        ),
        "feature_cache_written_to_disk": False,
    }
    model.cpu()
    gap_dct.cpu()
    top_dct.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return descriptors, telemetry


def select_recall_constrained_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    min_tp_retention: float = MIN_FIT_TP_RETENTION,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.size != labels.size or scores.size == 0:
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


def _standardize_fit(
    fit_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(fit_values, axis=0, dtype=np.float64)
    std = np.std(fit_values, axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return (fit_values - mean) / std, mean, std


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


def _run_oof(
    *,
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    cohort: Sequence[CleanTrainRow],
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, object],
]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort],
        dtype=np.int64,
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    scores = {
        condition: {
            role: np.full(EXPECTED_COHORT_ROWS, np.nan, dtype=np.float64)
            for role in ROLES
        }
        for condition, _, _ in CONDITIONS
    }
    thresholds = {
        role: np.full(EXPECTED_COHORT_ROWS, np.nan, dtype=np.float64)
        for role in ROLES
    }
    fit_records: list[Dict[str, object]] = []
    convergence_failures: list[Dict[str, object]] = []

    for hold_fold in FIT_FOLDS:
        fit_mask = folds != int(hold_fold)
        hold_mask = folds == int(hold_fold)
        for role in ROLES:
            clean_values = np.asarray(descriptors["clean"][role], dtype=np.float64)
            fit_values, mean, std = _standardize_fit(clean_values[fit_mask])
            readout = LogisticRegression(
                penalty="l2",
                C=READOUT_C,
                solver="lbfgs",
                fit_intercept=True,
                max_iter=READOUT_MAX_ITER,
                tol=READOUT_TOLERANCE,
                class_weight=None,
                random_state=SEED,
            )
            caught: list[warnings.WarningMessage]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                readout.fit(fit_values, labels[fit_mask])
            convergence = [
                str(value.message)
                for value in caught
                if issubclass(value.category, ConvergenceWarning)
            ]
            if convergence or int(readout.n_iter_[0]) >= READOUT_MAX_ITER:
                convergence_failures.append(
                    {
                        "fold": int(hold_fold),
                        "role": role,
                        "n_iter": int(readout.n_iter_[0]),
                        "warnings": convergence,
                    }
                )
            fit_scores = readout.predict_proba(fit_values)[:, 1]
            threshold = select_recall_constrained_threshold(
                fit_scores, labels[fit_mask]
            )
            thresholds[role][hold_mask] = threshold
            condition_iterations: Dict[str, int] = {}
            for condition, _, _ in CONDITIONS:
                hold_values = (
                    np.asarray(descriptors[condition][role], dtype=np.float64)[hold_mask]
                    - mean
                ) / std
                hold_scores = readout.predict_proba(hold_values)[:, 1]
                scores[condition][role][hold_mask] = hold_scores
                condition_iterations[condition] = int(hold_scores.size)
            fit_records.append(
                {
                    "fold": int(hold_fold),
                    "role": role,
                    "fit_rows": int(np.sum(fit_mask)),
                    "hold_rows": int(np.sum(hold_mask)),
                    "threshold": float(threshold),
                    "fit_tp_retention": float(
                        np.mean(fit_scores[labels[fit_mask] == 1] >= threshold)
                    ),
                    "n_iter": int(readout.n_iter_[0]),
                    "coefficient_l2": float(np.linalg.norm(readout.coef_)),
                    "intercept": float(readout.intercept_[0]),
                    "condition_hold_rows": condition_iterations,
                }
            )

    if convergence_failures:
        raise RuntimeError(
            f"Locked logistic readout did not converge: {convergence_failures}"
        )
    if not all(
        bool(np.isfinite(values).all())
        for condition_scores in scores.values()
        for values in condition_scores.values()
    ):
        raise ValueError("OOF score matrix is incomplete/non-finite.")
    if not all(bool(np.isfinite(values).all()) for values in thresholds.values()):
        raise ValueError("OOF threshold matrix is incomplete/non-finite.")

    metrics: Dict[str, object] = {
        "conditions": {},
        "fit_records": fit_records,
        "convergence_failures": convergence_failures,
        "readout": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": READOUT_C,
            "penalty": "l2",
            "fit_intercept": True,
            "max_iter": READOUT_MAX_ITER,
            "tol": READOUT_TOLERANCE,
            "class_weight": None,
            "random_state": SEED,
        },
    }
    for condition, _, _ in CONDITIONS:
        condition_metrics: Dict[str, object] = {}
        for role in ROLES:
            role_metrics = _binary_metrics(
                scores[condition][role], labels, thresholds[role]
            )
            role_metrics["fold_auroc"] = {
                str(fold): float(
                    roc_auc_score(
                        labels[folds == fold], scores[condition][role][folds == fold]
                    )
                )
                for fold in FIT_FOLDS
            }
            condition_metrics[role] = role_metrics
        metrics["conditions"][condition] = condition_metrics
    return scores, thresholds, metrics


def assess_signal_gate(
    metrics: Mapping[str, object], telemetry: Mapping[str, object]
) -> Dict[str, object]:
    conditions = metrics["conditions"]
    clean = conditions["clean"]
    candidate = clean["object_top16"]
    object_gap = clean["object_gap"]
    full_top16 = clean["full_top16"]
    geometry = clean["bbox_geometry"]
    fold_deltas = {
        str(fold): float(candidate["fold_auroc"][str(fold)])
        - float(object_gap["fold_auroc"][str(fold)])
        for fold in FIT_FOLDS
    }
    shifted_deltas = {
        condition: float(conditions[condition]["object_top16"]["auroc"])
        - float(conditions[condition]["object_gap"]["auroc"])
        for condition, _, _ in CONDITIONS
        if condition != "clean"
    }
    shifted_tp = {
        condition: float(
            conditions[condition]["object_top16"]["tp_retention"]
        )
        for condition, _, _ in CONDITIONS
        if condition != "clean"
    }
    checks = {
        "clean_auroc_at_least_065": float(candidate["auroc"]) >= 0.65,
        "clean_gain_over_object_gap_at_least_0025": (
            float(candidate["auroc"]) - float(object_gap["auroc"]) >= 0.025
        ),
        "clean_gain_over_full_top16_at_least_0015": (
            float(candidate["auroc"]) - float(full_top16["auroc"]) >= 0.015
        ),
        "clean_gain_over_bbox_geometry_at_least_005": (
            float(candidate["auroc"]) - float(geometry["auroc"]) >= 0.05
        ),
        "at_least_three_positive_fold_deltas": sum(
            value > 0.0 for value in fold_deltas.values()
        )
        >= 3,
        "no_fold_delta_below_minus002": min(fold_deltas.values()) >= -0.02,
        "mean_shifted_gain_at_least_002": statistics.mean(
            shifted_deltas.values()
        )
        >= 0.02,
        "no_shifted_delta_below_minus0015": min(shifted_deltas.values())
        >= -0.015,
        "clean_tp_retention_at_least_090": float(candidate["tp_retention"])
        >= 0.90,
        "clean_fp_rejection_at_least_020": float(candidate["fp_rejection"])
        >= 0.20,
        "clean_fp_rejection_gain_at_least_005": (
            float(candidate["fp_rejection"])
            - float(object_gap["fp_rejection"])
            >= 0.05
        ),
        "all_shifted_tp_retention_at_least_088": min(shifted_tp.values())
        >= 0.88,
        "all_descriptors_finite": bool(telemetry["all_finite"]),
        "top16_differs_from_top1": float(
            telemetry["object_top16_gap_max_abs_delta"]
        )
        > 1e-6,
        "all_frequency_groups_nonzero_variance": bool(
            telemetry["all_frequency_groups_nonzero_variance"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "clean_auroc": {
            role: float(clean[role]["auroc"]) for role in ROLES
        },
        "clean_candidate_gains": {
            "over_object_gap": float(candidate["auroc"])
            - float(object_gap["auroc"]),
            "over_full_top16": float(candidate["auroc"])
            - float(full_top16["auroc"]),
            "over_bbox_geometry": float(candidate["auroc"])
            - float(geometry["auroc"]),
            "fp_rejection_over_object_gap": float(candidate["fp_rejection"])
            - float(object_gap["fp_rejection"]),
        },
        "fold_auroc_deltas_vs_object_gap": fold_deltas,
        "shifted_auroc_deltas_vs_object_gap": shifted_deltas,
        "shifted_candidate_tp_retention": shifted_tp,
    }


class _PatchEmbeddingWithDescriptor(nn.Module):
    def __init__(self, patch_embed: nn.Module) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.extractor = ObjectAlignedDCTExtractor("top16")
        self.current_bboxes: Optional[Tensor] = None
        self.last_descriptor: Optional[Tensor] = None
        for name in ("num_patches", "patch_size", "base_grid_size"):
            if hasattr(patch_embed, name):
                setattr(self, name, getattr(patch_embed, name))

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.patch_embed(features)
        if self.current_bboxes is None:
            raise RuntimeError("Descriptor benchmark bbox was not set.")
        if tuple(tokens.shape[1:]) != (PATCH_TOKENS, CHANNELS):
            raise ValueError(f"Benchmark patch shape differs: {tuple(tokens.shape)}")
        patch_map = (
            tokens.reshape(int(tokens.size(0)), GRID_SIZE, GRID_SIZE, CHANNELS)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.last_descriptor = self.extractor(patch_map, self.current_bboxes)
        return tokens


class _FullModelBenchmarkWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, with_descriptor: bool) -> None:
        super().__init__()
        self.model = model
        self.with_descriptor = bool(with_descriptor)
        self.side_patch_embed: Optional[_PatchEmbeddingWithDescriptor] = None
        if self.with_descriptor:
            side = _PatchEmbeddingWithDescriptor(self.model.patch_embed)
            self.model.patch_embed = side
            self.side_patch_embed = side

    def forward(
        self,
        images: Tensor,
        image_mask: Tensor,
        bbox: Tensor,
        crop_bbox: Tensor,
    ) -> Tensor:
        if self.side_patch_embed is not None:
            self.side_patch_embed.current_bboxes = crop_bbox
        logits, _ = _forward_classification_with_metadata(
            self.model,
            images,
            {"image_mask": image_mask, "bbox": bbox},
            device=images.device,
        )
        if not torch.is_tensor(logits):
            raise TypeError("Benchmark model output lacks logits.")
        return logits


def _benchmark_model_role(
    model: nn.Module,
    *,
    with_descriptor: bool,
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    wrapper = _FullModelBenchmarkWrapper(
        copy.deepcopy(model), with_descriptor=with_descriptor
    ).to(device=device).eval()
    generator = torch.Generator().manual_seed(SEED + 501)
    images = torch.randn(
        32, 3, 256, 256, generator=generator, dtype=torch.float32
    ).to(device=device)
    image_mask = torch.ones(32, 256, 256, device=device, dtype=torch.bool)
    bbox = torch.tensor(
        [0.5, 0.5, 0.70, 0.80], device=device, dtype=torch.float32
    ).view(1, 4).expand(32, -1).contiguous()
    crop_bbox = torch.tensor(
        [0.5, 0.5, 0.62, 0.72], device=device, dtype=torch.float32
    ).view(1, 4).expand(32, -1).contiguous()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    last_logits: Optional[Tensor] = None
    descriptor_shape: Optional[list[int]] = None
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        for _ in range(5):
            last_logits = wrapper(images, image_mask, bbox, crop_bbox)
        torch.cuda.synchronize(device)
        for _ in range(int(repeats)):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record()
            last_logits = wrapper(images, image_mask, bbox, crop_bbox)
            finish.record()
            torch.cuda.synchronize(device)
            elapsed.append(float(start.elapsed_time(finish)))
        if wrapper.side_patch_embed is not None:
            descriptor = wrapper.side_patch_embed.last_descriptor
            if not torch.is_tensor(descriptor):
                raise RuntimeError("Candidate benchmark did not execute descriptor.")
            descriptor_shape = list(descriptor.shape)
    if last_logits is None:
        raise RuntimeError("Benchmark produced no logits.")
    peak = int(torch.cuda.max_memory_allocated(device))
    logits_cpu = last_logits.detach().float().cpu()
    wrapper.cpu()
    del wrapper, images, image_mask, bbox, crop_bbox, last_logits
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "elapsed_ms": elapsed,
        "median_ms": float(statistics.median(elapsed)),
        "peak_allocated_bytes": peak,
        "batch_size": 32,
        "dtype": "bfloat16_autocast",
        "mode": "normal_evaluator_forward",
        "descriptor_shape": descriptor_shape,
        "logits": logits_cpu,
    }


def _resource_audit(
    model: nn.Module, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    native = _benchmark_model_role(
        model, with_descriptor=False, device=device, repeats=repeats
    )
    candidate = _benchmark_model_role(
        model, with_descriptor=True, device=device, repeats=repeats
    )
    native_logits = native.pop("logits")
    candidate_logits = candidate.pop("logits")
    logit_error = float((native_logits - candidate_logits).abs().max())
    prediction_mismatches = int(
        (native_logits.argmax(dim=1) != candidate_logits.argmax(dim=1)).sum()
    )
    return {
        "native": native,
        "object_top16_side_descriptor": candidate,
        "runtime_ratio": float(candidate["median_ms"])
        / float(native["median_ms"]),
        "peak_memory_ratio": float(candidate["peak_allocated_bytes"])
        / float(native["peak_allocated_bytes"]),
        "normal_forward_logit_max_abs_error": logit_error,
        "normal_forward_prediction_mismatches": prediction_mismatches,
        "trace_path_used": False,
        "descriptor_executed": candidate["descriptor_shape"] == [32, CHANNELS],
        "added_trainable_parameters": 0,
        "fixed_dct_buffer_values": CHANNELS * GRID_SIZE * GRID_SIZE,
    }


def _onnx_audit() -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    extractor = ObjectAlignedDCTExtractor("top16").cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 601)
    features = torch.randn(
        1,
        CHANNELS,
        GRID_SIZE,
        GRID_SIZE,
        generator=generator,
        dtype=torch.float32,
    )
    bboxes = torch.tensor([[0.5, 0.5, 0.70, 0.80]], dtype=torch.float32)
    with torch.inference_mode():
        reference = extractor(features, bboxes).numpy()
    with tempfile.TemporaryDirectory(prefix="trkh_fcanet_onnx_") as directory:
        path = Path(directory) / "object_top16_extractor.onnx"
        torch.onnx.export(
            extractor,
            (features, bboxes),
            str(path),
            input_names=["patch_map", "crop_bbox"],
            output_names=["descriptor"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        observed = session.run(
            None,
            {"patch_map": features.numpy(), "crop_bbox": bboxes.numpy()},
        )[0]
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
        "input_shapes": {
            "patch_map": list(features.shape),
            "crop_bbox": list(bboxes.shape),
        },
        "output_shape": list(observed.shape),
        "domains": domains,
        "operators": operators,
        "standard_domains_only": all(value in {"", "ai.onnx"} for value in domains),
        "output_shape_match": list(reference.shape) == list(observed.shape),
        "max_abs_error": float(np.max(np.abs(reference - observed))),
        "all_outputs_finite": bool(np.isfinite(observed).all()),
        "onnx_sha256_ephemeral": payload_sha,
        "onnx_bytes_ephemeral": payload_bytes,
        "onnx_retained": False,
        "tensorrt_parse": parse_ok,
        "tensorrt_errors": parser_errors,
        "tensorrt_engine_build": engine_ok,
        "tensorrt_engine_bytes_ephemeral": engine_bytes,
        "tensorrt_engine_retained": False,
    }


def assess_structural_gate(
    *,
    geometry: Mapping[str, object],
    equation: Mapping[str, object],
    declaration: Mapping[str, object],
    resource: Mapping[str, object],
    onnx: Mapping[str, object],
) -> Dict[str, object]:
    checks = {
        "geometry_preflight_passed": geometry.get("status") == "passed",
        "geometry_before_model_inference": not bool(
            geometry.get("model_inference_used", True)
        ),
        "official_frequency_indices_exact": bool(
            equation["official_frequency_indices_exact"]
        ),
        "official_output_error_within_limit": float(
            equation["official_output_max_abs_error"]
        )
        <= MAX_OFFICIAL_ERROR,
        "official_weight_error_within_limit": float(
            equation["official_weight_max_abs_error"]
        )
        <= MAX_ORACLE_ERROR,
        "oracle_output_error_within_limit": float(
            equation["oracle_output_max_abs_error"]
        )
        <= MAX_ORACLE_ERROR,
        "gradient_error_within_limit": float(equation["gradient_max_abs_error"])
        <= MAX_GRADIENT_ERROR,
        "finite_difference_within_limit": float(
            equation["finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "gap_proportionality_within_limit": float(
            equation["gap_proportionality_max_abs_error"]
        )
        <= MAX_GAP_ERROR,
        "bbox_geometry_control_within_limit": float(
            equation["bbox_geometry_control_max_abs_error"]
        )
        <= MAX_BBOX_GEOMETRY_ERROR,
        "bf16_error_within_limit": float(equation["bf16_max_abs_error"])
        <= MAX_BF16_ERROR,
        "bf16_bbox_coordinates_remain_fp32": str(equation["bf16_bbox_dtype"])
        == "torch.float32",
        "fp32_gradient_finite_nonzero": bool(
            equation["fp32_gradient_finite"] and equation["fp32_gradient_nonzero"]
        ),
        "bf16_gradient_finite_nonzero": bool(
            equation["bf16_gradient_finite"] and equation["bf16_gradient_nonzero"]
        ),
        "object_grid_repeat_exact": bool(equation["object_grid_repeat_exact"]),
        "all_equation_outputs_finite": bool(equation["all_outputs_finite"]),
        "locked_declaration_exception_exact": bool(declaration["passed"]),
        "onnx_standard_domains_only": bool(onnx["standard_domains_only"]),
        "onnx_shape_match": bool(onnx["output_shape_match"]),
        "onnx_error_within_limit": float(onnx["max_abs_error"])
        <= MAX_ONNX_ERROR,
        "onnx_outputs_finite": bool(onnx["all_outputs_finite"]),
        "tensorrt_parse_passed": bool(onnx["tensorrt_parse"]),
        "tensorrt_engine_build_passed": bool(onnx["tensorrt_engine_build"]),
        "runtime_ratio_within_limit": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_within_limit": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "normal_forward_logits_exact": float(
            resource["normal_forward_logit_max_abs_error"]
        )
        == 0.0,
        "normal_forward_predictions_exact": int(
            resource["normal_forward_prediction_mismatches"]
        )
        == 0,
        "descriptor_executed": bool(resource["descriptor_executed"]),
        "trace_path_not_used": not bool(resource["trace_path_used"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


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
                    payload[f"score_{role}"] = repr(
                        float(scores[condition][role][position])
                    )
                    payload[f"threshold_{role}"] = repr(
                        float(thresholds[role][position])
                    )
                writer.writerow(payload)


def _replay_oof_csv(path: Path) -> Dict[str, object]:
    grouped: Dict[str, list[Dict[str, str]]] = {
        condition: [] for condition, _, _ in CONDITIONS
    }
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_index",
            "fold",
            "target",
            "keeper_prediction",
            "cohort",
            "condition",
            *{f"score_{role}" for role in ROLES},
            *{f"threshold_{role}" for role in ROLES},
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"OOF CSV misses columns: {sorted(missing)}")
        for row in reader:
            condition = str(row["condition"])
            if condition not in grouped:
                raise ValueError(f"Unexpected OOF condition: {condition}")
            grouped[condition].append(dict(row))
    replay: Dict[str, object] = {"conditions": {}}
    expected_rows = EXPECTED_COHORT_ROWS
    for condition, rows in grouped.items():
        if len(rows) != expected_rows:
            raise ValueError(
                f"OOF CSV {condition} rows differ: {len(rows)} != {expected_rows}"
            )
        sample_indices = [int(row["sample_index"]) for row in rows]
        if len(set(sample_indices)) != expected_rows:
            raise ValueError(f"OOF CSV {condition} has duplicate sample indices.")
        ordered_hash = _ordered_index_sha256(sample_indices)
        if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
            raise ValueError(
                f"OOF CSV {condition} ordered cohort hash differs: {ordered_hash}"
            )
        if any(int(row["fold"]) not in FIT_FOLDS for row in rows):
            raise ValueError(f"OOF CSV {condition} contains an unlocked fold.")
        if any(row["cohort"] not in {"tp", "fp"} for row in rows):
            raise ValueError(f"OOF CSV {condition} contains an unlocked cohort label.")
        if any(
            (row["cohort"] == "tp") != (int(row["target"]) == FOCUS_CLASS)
            or int(row["keeper_prediction"]) != FOCUS_CLASS
            for row in rows
        ):
            raise ValueError(f"OOF CSV {condition} target/prediction declaration differs.")
        labels = np.asarray(
            [1 if row["cohort"] == "tp" else 0 for row in rows], dtype=np.int64
        )
        folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
        condition_metrics: Dict[str, object] = {}
        for role in ROLES:
            role_scores = np.asarray(
                [float(row[f"score_{role}"]) for row in rows], dtype=np.float64
            )
            role_thresholds = np.asarray(
                [float(row[f"threshold_{role}"]) for row in rows], dtype=np.float64
            )
            role_metrics = _binary_metrics(role_scores, labels, role_thresholds)
            role_metrics["fold_auroc"] = {
                str(fold): float(
                    roc_auc_score(
                        labels[folds == fold], role_scores[folds == fold]
                    )
                )
                for fold in FIT_FOLDS
            }
            condition_metrics[role] = role_metrics
        replay["conditions"][condition] = condition_metrics
    replay["rows"] = sum(len(rows) for rows in grouped.values())
    replay["expected_rows"] = EXPECTED_COHORT_ROWS * len(CONDITIONS)
    return replay


def _numeric_differences(
    left: object, right: object, *, path: str = "root"
) -> list[Dict[str, object]]:
    differences: list[Dict[str, object]] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                differences.append(
                    {"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key)}
                )
            else:
                differences.extend(
                    _numeric_differences(left[key], right[key], path=f"{path}.{key}")
                )
        return differences
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = abs(float(left) - float(right))
        if delta > 1e-12:
            differences.append(
                {
                    "path": path,
                    "left": float(left),
                    "right": float(right),
                    "abs_delta": delta,
                }
            )
    elif left != right:
        differences.append({"path": path, "left": left, "right": right})
    return differences


def _select_visual_positions(
    *,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, list[int]]:
    candidate = np.asarray(scores["clean"]["object_top16"], dtype=np.float64)
    control = np.asarray(scores["clean"]["object_gap"], dtype=np.float64)
    candidate_z = (candidate - candidate.mean()) / max(candidate.std(), 1e-12)
    control_z = (control - control.mean()) / max(control.std(), 1e-12)
    advantage = candidate_z - control_z
    tp_positions = np.asarray(
        [index for index, row in enumerate(cohort) if _cohort_label(row) == "tp"],
        dtype=np.int64,
    )
    fp_positions = np.asarray(
        [index for index, row in enumerate(cohort) if _cohort_label(row) == "fp"],
        dtype=np.int64,
    )

    def top_and_bottom(positions: np.ndarray) -> tuple[list[int], list[int]]:
        ordered = positions[np.argsort(advantage[positions], kind="mergesort")]
        bottom = [int(value) for value in ordered[:8]]
        top = [int(value) for value in ordered[-8:][::-1]]
        return top, bottom

    tp_top, tp_bottom = top_and_bottom(tp_positions)
    fp_top, fp_bottom = top_and_bottom(fp_positions)
    return {
        "positive_advantage_tp": tp_top,
        "positive_advantage_fp": fp_top,
        "negative_advantage_tp": tp_bottom,
        "negative_advantage_fp": fp_bottom,
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


def _bbox_overlay(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    cx, cy, width, height = [float(value) for value in bbox[:4]]
    left = max(0.0, cx - width / 2.0) * result.width
    right = min(1.0, cx + width / 2.0) * result.width
    top = max(0.0, cy - height / 2.0) * result.height
    bottom = min(1.0, cy + height / 2.0) * result.height
    draw.rectangle((left, top, right, bottom), outline=(255, 40, 40), width=3)
    return result


def _render_visuals(
    *,
    output_dir: Path,
    model_checkpoint: Mapping[str, object],
    dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, object]:
    selections = _select_visual_positions(cohort=cohort, scores=scores)
    semantics = _eval_semantics(model_checkpoint)
    page_dir = output_dir / "visual_review"
    page_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    top_x, top_y = get_frequency_indices("top16")
    scale = GRID_SIZE // 7
    frequencies = [
        (int(x * scale), int(y * scale)) for x, y in zip(top_x, top_y)
    ]
    page_records: list[Dict[str, object]] = []
    selection_rows: list[Dict[str, object]] = []

    for category, positions in selections.items():
        selected_rows = [cohort[position] for position in positions]
        loader, _ = _make_condition_loader(
            base_dataset=dataset,
            transform=transform,
            indices=[row.sample_index for row in selected_rows],
            brightness=1.0,
            contrast=1.0,
            batch_size=8,
            num_workers=int(args.num_workers),
            context=f"fcanet_a0_visual_{category}",
        )
        image_batches: list[Tensor] = []
        bbox_batches: list[Tensor] = []
        observed_indices: list[int] = []
        for images, _, metadata in loader:
            crop_bboxes = metadata.get("crop_bbox")
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(sample_indices):
                raise ValueError("Visual loader lacks crop_bbox/sample_index.")
            image_batches.append(images.float())
            bbox_batches.append(crop_bboxes[:, :4].float())
            observed_indices.extend(int(value) for value in sample_indices.tolist())
        if observed_indices != [row.sample_index for row in selected_rows]:
            raise ValueError(f"Visual {category} order differs.")
        images = torch.cat(image_batches, dim=0)
        bboxes = torch.cat(bbox_batches, dim=0)
        rgb = _tensor_to_rgb(images, semantics)
        object_rgb = align_object_feature_map(
            rgb,
            bboxes,
            output_height=160,
            output_width=160,
        )

        canvas = Image.new("RGB", (1120, 8 * 184 + 40), color=(250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), category, fill=(0, 0, 0), font=font)
        for row_index, (position, row) in enumerate(zip(positions, selected_rows)):
            y = 36 + row_index * 184
            source_image = _pil_from_rgb_tensor(rgb[row_index])
            source_image = _bbox_overlay(source_image, bboxes[row_index].tolist())
            source_image = source_image.resize((160, 160), Image.Resampling.BILINEAR)
            crop_image = _pil_from_rgb_tensor(object_rgb[row_index])
            canvas.paste(source_image, (12, y))
            canvas.paste(crop_image, (184, y))
            candidate_score = float(scores["clean"]["object_top16"][position])
            control_score = float(scores["clean"]["object_gap"][position])
            candidate_threshold = float(thresholds["object_top16"][position])
            control_threshold = float(thresholds["object_gap"][position])
            candidate_values = descriptors["clean"]["object_top16"][position]
            group_strength = [
                float(
                    np.sum(
                        np.abs(
                            candidate_values[
                                group * CHANNELS_PER_FREQUENCY :
                                (group + 1) * CHANNELS_PER_FREQUENCY
                            ]
                        )
                    )
                )
                for group in range(FREQUENCY_GROUPS)
            ]
            strongest_group = int(np.argmax(group_strength))
            frequency = frequencies[strongest_group]
            lines = [
                f"sample={row.sample_index} fold={row.fold} target={row.target} cohort={_cohort_label(row)}",
                f"keeper_p1={row.keeper_probabilities[1]:.6f}",
                f"object_top16={candidate_score:.6f} threshold={candidate_threshold:.6f}",
                f"object_gap={control_score:.6f} threshold={control_threshold:.6f}",
                f"strongest_group={strongest_group} frequency={frequency}",
                f"bbox={[round(float(value), 5) for value in bboxes[row_index].tolist()]}",
            ]
            for line_index, line in enumerate(lines):
                draw.text(
                    (360, y + 8 + line_index * 22),
                    line,
                    fill=(0, 0, 0),
                    font=font,
                )
            selection_rows.append(
                {
                    "category": category,
                    "position": int(position),
                    "sample_index": row.sample_index,
                    "fold": row.fold,
                    "target": row.target,
                    "cohort": _cohort_label(row),
                    "object_top16_score": candidate_score,
                    "object_gap_score": control_score,
                    "object_top16_threshold": candidate_threshold,
                    "object_gap_threshold": control_threshold,
                    "strongest_frequency_group": strongest_group,
                    "strongest_frequency": list(frequency),
                    "bbox": [float(value) for value in bboxes[row_index].tolist()],
                }
            )
        page_path = page_dir / f"{category}.png"
        canvas.save(page_path, format="PNG", optimize=False)
        page_records.append(
            {
                "category": category,
                "path": str(page_path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": _sha256(page_path),
                "bytes": int(page_path.stat().st_size),
                "rows": len(positions),
            }
        )

    selection_path = page_dir / "selection.json"
    _write_json(
        selection_path,
        {
            "method": METHOD,
            "selection_rule": "zscore(object_top16)-zscore(object_gap)",
            "categories": selections,
            "rows": selection_rows,
        },
    )
    expected_categories = {
        "positive_advantage_tp",
        "positive_advantage_fp",
        "negative_advantage_tp",
        "negative_advantage_fp",
    }
    complete = bool(
        set(selections) == expected_categories
        and all(len(values) == 8 for values in selections.values())
        and len(page_records) == 4
        and all(record["bytes"] > 0 and record["rows"] == 8 for record in page_records)
    )
    return {
        "status": "pending_manual_review",
        "complete": complete,
        "pages": page_records,
        "selection_path": str(selection_path.relative_to(output_dir)).replace("\\", "/"),
        "selection_sha256": _sha256(selection_path),
        "selected_rows": len(selection_rows),
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
        "raw_data_modified": False,
    }
    _write_json(manifest_path, payload)
    return {**payload, "manifest_sha256": _sha256(manifest_path)}


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    structural = summary.get("structural_gate", {})
    signal = summary.get("signal_gate", {})
    metrics = summary.get("oof_metrics", {}).get("conditions", {})
    clean = metrics.get("clean", {}) if isinstance(metrics, Mapping) else {}
    candidate = clean.get("object_top16", {}) if isinstance(clean, Mapping) else {}
    control = clean.get("object_gap", {}) if isinstance(clean, Mapping) else {}
    lines = [
        "# FcaNet Object-Frequency Signal A0 Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Structural gate: `{structural.get('passed')}`",
        f"- Signal gate: `{signal.get('passed')}`",
        f"- Automated gate: `{summary.get('automated_gate_passed')}`",
        f"- Validation used: `{summary.get('validation_data_used')}`",
        f"- Test used: `{summary.get('test_data_used')}`",
        "",
        "## Clean OOF",
        "",
        f"- object_top16 AUROC: `{candidate.get('auroc')}`",
        f"- object_gap AUROC: `{control.get('auroc')}`",
        f"- object_top16 TP retention: `{candidate.get('tp_retention')}`",
        f"- object_top16 restricted-FP rejection: `{candidate.get('fp_rejection')}`",
        "",
        "## Decision",
        "",
        (
            "A0 is awaiting hash-locked manual visual review. It cannot authorize "
            "trainer integration until finalization."
            if summary.get("status") == "awaiting_visual_review"
            else "The fail-closed status above controls the route."
        ),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_pre_oof_rejection(
    *,
    output_dir: Path,
    provenance: Mapping[str, object],
    geometry: Mapping[str, object],
    equation: Mapping[str, object],
    declaration: Mapping[str, object],
    resource: Mapping[str, object],
    onnx: Mapping[str, object],
    structural_gate: Mapping[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    summary = {
        "method": METHOD,
        "status": "rejected_pre_oof_structural_gate",
        "provenance": provenance,
        "geometry_preflight": geometry,
        "equation": equation,
        "declaration_replay": declaration,
        "resource": resource,
        "onnx": onnx,
        "structural_gate": structural_gate,
        "signal_gate": {"passed": False, "not_run": True},
        "oof_readout_used": False,
        "visual_review": {"complete": False, "not_run": True},
        "automated_gate_passed": False,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "trainer_integration_authorized": False,
        "short_pair_authorized": False,
        "current_command_update_authorized": False,
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
        raise FileNotFoundError(f"Formal summary not found: {summary_path}")
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
        raise ValueError(f"Summary is not awaiting review: {summary.get('status')}")
    visual = summary.get("visual_review")
    if not isinstance(visual, Mapping) or not bool(visual.get("complete")):
        raise ValueError("Visual artifact is incomplete and cannot be finalized.")
    pages = visual.get("pages")
    if not isinstance(pages, list) or len(pages) != 4:
        raise ValueError("Visual page manifest differs from the locked four pages.")
    for page in pages:
        page_path = output_dir / str(page["path"])
        if _sha256(page_path) != str(page["sha256"]):
            raise ValueError(f"Visual page hash differs: {page_path}")
    visual = dict(visual)
    visual.update(
        {
            "status": "reviewed",
            "manual_result": args.visual_review_result,
            "manual_note": str(args.visual_review_note),
            "reviewed_summary_sha256": observed,
        }
    )
    summary["visual_review"] = visual
    final_pass = bool(summary.get("automated_gate_passed")) and (
        args.visual_review_result == "pass"
    )
    summary["status"] = "passed" if final_pass else "rejected"
    summary["trainer_integration_authorized"] = final_pass
    summary["short_pair_authorized"] = final_pass
    summary["validation_data_used"] = False
    summary["test_data_used"] = False
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
        "trainer_integration_authorized": final_pass,
        "short_pair_authorized": final_pass,
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

    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint = load_checkpoint(
        Path(provenance["paths"]["checkpoint"]), map_location="cpu"
    )
    dataset, transform, locked_bboxes, geometry = _geometry_preflight(
        checkpoint=checkpoint,
        rows=rows,
        cohort=cohort,
        data_path=Path(provenance["paths"]["data"]),
        args=args,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Locked FcaNet A0 requires CUDA.")
    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    model = _load_keeper_model(checkpoint)
    equation = _equation_diagnostics(
        Path(provenance["paths"]["layer_source"]), device
    )
    declaration = _declaration_replay(
        model=model,
        dataset=dataset,
        transform=transform,
        cohort=cohort,
        args=args,
        device=device,
    )
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    resource = _resource_audit(
        model, device=device, repeats=int(args.benchmark_repeats)
    )
    onnx = _onnx_audit()
    structural_gate = assess_structural_gate(
        geometry=geometry,
        equation=equation,
        declaration=declaration,
        resource=resource,
        onnx=onnx,
    )
    if not bool(structural_gate["passed"]):
        return _write_pre_oof_rejection(
            output_dir=output_dir,
            provenance=provenance,
            geometry=geometry,
            equation=equation,
            declaration=declaration,
            resource=resource,
            onnx=onnx,
            structural_gate=structural_gate,
            elapsed_seconds=time.perf_counter() - started,
        )

    descriptors, descriptor_telemetry = _extract_descriptors(
        model=model,
        dataset=dataset,
        transform=transform,
        cohort=cohort,
        locked_bboxes=locked_bboxes,
        args=args,
        device=device,
    )
    scores, thresholds, oof_metrics = _run_oof(
        descriptors=descriptors,
        cohort=cohort,
    )
    signal_gate = assess_signal_gate(oof_metrics, descriptor_telemetry)
    oof_path = output_dir / "oof_scores.csv"
    _write_oof_csv(
        path=oof_path,
        cohort=cohort,
        scores=scores,
        thresholds=thresholds,
    )
    replay = _replay_oof_csv(oof_path)
    replay_differences = _numeric_differences(
        oof_metrics["conditions"], replay["conditions"], path="conditions"
    )
    replay_summary = {
        "rows": replay["rows"],
        "expected_rows": replay["expected_rows"],
        "differences": replay_differences[:20],
        "difference_count": len(replay_differences),
        "max_abs_numeric_difference": max(
            [float(value.get("abs_delta", 0.0)) for value in replay_differences]
            or [0.0]
        ),
        "passed": not replay_differences
        and int(replay["rows"]) == int(replay["expected_rows"]),
    }
    visual_review = _render_visuals(
        output_dir=output_dir,
        model_checkpoint=checkpoint,
        dataset=dataset,
        transform=transform,
        cohort=cohort,
        descriptors=descriptors,
        scores=scores,
        thresholds=thresholds,
        args=args,
    )
    automated_gate = bool(
        structural_gate["passed"]
        and signal_gate["passed"]
        and replay_summary["passed"]
        and visual_review["complete"]
    )
    summary = {
        "method": METHOD,
        "status": "awaiting_visual_review",
        "provenance": provenance,
        "geometry_preflight": geometry,
        "equation": equation,
        "declaration_replay": declaration,
        "resource": resource,
        "onnx": onnx,
        "structural_gate": structural_gate,
        "descriptor_telemetry": descriptor_telemetry,
        "oof_metrics": oof_metrics,
        "signal_gate": signal_gate,
        "independent_csv_replay": replay_summary,
        "visual_review": visual_review,
        "automated_gate_passed": automated_gate,
        "oof_readout_used": True,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "trainer_integration_authorized": False,
        "short_pair_authorized": False,
        "current_command_update_authorized": False,
        "raw_data_modified": False,
        "feature_cache_written_to_disk": False,
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
                "trainer_integration_authorized": result.get(
                    "trainer_integration_authorized", False
                ),
                "short_pair_authorized": result.get("short_pair_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

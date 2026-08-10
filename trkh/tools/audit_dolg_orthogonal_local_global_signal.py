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
import weakref

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
from trkh.tools.audit_fcanet_object_frequency_signal import (
    bbox_geometry_descriptor,
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
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "dolg_orthogonal_local_global_signal_a0"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
BENCHMARK_REPEATS = 5
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
    "object_min": 28,
    "object_max": 210,
    "context_min": 46,
    "context_max": 228,
}
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
ROLES = (
    "global",
    "full_local",
    "object_local",
    "global_object_concat",
    "global_full_orth",
    "global_object_orth",
    "bbox_geometry",
)
ROLE_DIMS = {
    "global": 256,
    "full_local": 256,
    "object_local": 256,
    "global_object_concat": 512,
    "global_full_orth": 512,
    "global_object_orth": 512,
    "bbox_geometry": 18,
}
MECHANISM_FIELDS = (
    "global_norm",
    "object_local_norm",
    "object_orth_norm",
    "full_orth_norm",
    "object_global_cosine",
    "object_parallel_energy_ratio",
    "full_parallel_energy_ratio",
)
GRID_SIZE = 16
PATCH_TOKENS = GRID_SIZE * GRID_SIZE
CHANNELS = 256
MAX_OFFICIAL_ERROR = 1e-12
MAX_ORACLE_ERROR = 1e-12
MAX_ORTHOGONALITY_ERROR = 1e-10
MAX_POOLING_EQUIVALENCE_ERROR = 1e-12
MAX_GRADIENT_ERROR = 1e-10
MAX_FINITE_DIFFERENCE_ERROR = 1e-5
MAX_BF16_ERROR = 0.05
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.10
MAX_MEMORY_RATIO = 1.10

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "51b52aaeb2810ea4172ccd93c6238add378fd845149808fa4dea1cfc694e150d"
LOCKED_PAPER_SHA256 = "3ca7070e9d749fa84c9606b43dabd159242a518bd902817565b27b2fc22fb679"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "63b117d76f660db9617c2081ec771e3cbdf4bcc8"
LOCKED_OFFICIAL_TREE = "c7639a440fbd5b25b911737dae392ddfd1d346a6"
LOCKED_OFFICIAL_HASHES = {
    "model_source": "83b1984de8ca0be585975622b1dba6e85351b1cd6d17b71fb06c3ce17ab6dd54",
    "license": "4a9589726b97c388428d37e0526743b77f189f531aef987017af06abb1b450d4",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked fit-only DOLG orthogonal local-global signal gate. "
            "Validation, test, trainer integration, and raw-data edits are forbidden."
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
            "docs/TRKH_5CLASS_DOLG_ORTHOGONAL_LOCAL_GLOBAL_SIGNAL_"
            "PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\Yang_DOLG_ICCV2021_accepted.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\dolg-iccv2021"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_dolg_orthogonal_local_global_signal_a0_20260717"),
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


def _verify_committed_implementation(root: Path) -> Dict[str, object]:
    root = Path(root).resolve()
    paths = {
        "auditor": Path(__file__).resolve(),
        "focused_test": root
        / "tests"
        / "test_audit_dolg_orthogonal_local_global_signal.py",
        "launcher": root
        / "scripts"
        / "run_trkh_dolg_orthogonal_local_global_signal_a0.ps1",
    }
    relative_paths: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"DOLG {name} is missing: {path}")
        relative = path.relative_to(root).as_posix()
        _git_value(root, "ls-files", "--error-unmatch", relative)
        relative_paths[name] = relative
    status = _git_value(
        root,
        "status",
        "--porcelain",
        "--",
        *relative_paths.values(),
    )
    if status:
        raise ValueError(f"DOLG implementation is not commit-clean: {status}")
    return {
        name: {
            "path": relative_paths[name],
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


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
        "model_source": official / "core" / "model" / "dolg_model.py",
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
        raise ValueError("Arguments differ from the locked DOLG A0 protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "DOLG protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "accepted DOLG paper"
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
            paths[name], expected, f"official DOLG {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(
        paths["official_root"], "rev-parse", "HEAD^{tree}"
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(f"Official DOLG commit differs: {official_commit}")
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official DOLG tree differs: {official_tree}")
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official DOLG worktree must be clean.")

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
            "Locked DOLG cohort differs: "
            f"{len(cohort)}/{positives}/{negatives}"
        )
    fold_counts = {fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS}
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked DOLG fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for formal DOLG A0.")
    implementation = _verify_committed_implementation(repo_root)
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal DOLG A0 requires the pushed repository commit.")

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
            "implementation_committed_clean": True,
            "implementation": implementation,
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


def object_center_mask(bboxes: Tensor, grid_size: int = GRID_SIZE) -> Tensor:
    if bboxes.ndim != 2 or int(bboxes.size(1)) < 4:
        raise ValueError("Bboxes must have shape [B,4+] in normalized xywh form.")
    boxes = bboxes[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = boxes.unbind(dim=1)
    left = (cx - width / 2.0).clamp(0.0, 1.0)
    right = (cx + width / 2.0).clamp(0.0, 1.0)
    top = (cy - height / 2.0).clamp(0.0, 1.0)
    bottom = (cy + height / 2.0).clamp(0.0, 1.0)
    coordinates = (
        torch.arange(grid_size, device=boxes.device, dtype=torch.float32) + 0.5
    ) / float(grid_size)
    x = coordinates.view(1, 1, grid_size)
    y = coordinates.view(1, grid_size, 1)
    mask = (
        (x >= left[:, None, None])
        & (x < right[:, None, None])
        & (y >= top[:, None, None])
        & (y < bottom[:, None, None])
    )
    return mask.flatten(1)


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
        context="dolg_a0_geometry_cpu",
    )
    bbox_batches: list[Tensor] = []
    observed_indices: list[int] = []
    observed_targets: list[int] = []
    for _, targets, metadata in loader:
        sample_indices = metadata.get("sample_index")
        crop_bboxes = metadata.get("crop_bbox")
        if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bboxes):
            raise ValueError("DOLG geometry loader lacks sample_index/crop_bbox.")
        observed_indices.extend(int(value) for value in sample_indices.tolist())
        observed_targets.extend(int(value) for value in targets.tolist())
        bbox_batches.append(crop_bboxes[:, :4].to(dtype=torch.float32).cpu())
    if observed_indices != indices or observed_targets != expected_targets:
        raise ValueError("DOLG geometry loader order/targets differ.")
    bboxes = torch.cat(bbox_batches, dim=0).contiguous()
    if tuple(bboxes.shape) != (EXPECTED_COHORT_ROWS, 4):
        raise ValueError(f"Unexpected locked bbox shape: {tuple(bboxes.shape)}")
    if not bool(torch.isfinite(bboxes).all()):
        raise ValueError("Locked DOLG bbox geometry is non-finite.")
    bbox_hash = hashlib.sha256(bboxes.numpy().tobytes(order="C")).hexdigest()
    if bbox_hash != LOCKED_BBOX_BYTES_SHA256:
        raise ValueError(f"Locked bbox byte hash differs: {bbox_hash}")
    mask = object_center_mask(bboxes)
    object_count = mask.sum(dim=1)
    context_count = (~mask).sum(dim=1)
    observed = {
        "object_min": int(object_count.min()),
        "object_max": int(object_count.max()),
        "context_min": int(context_count.min()),
        "context_max": int(context_count.max()),
    }
    if observed != LOCKED_GEOMETRY:
        raise ValueError(f"Locked 16x16 DOLG support differs: {observed}")
    if bool((object_count == 0).any() or (context_count == 0).any()):
        raise ValueError("Empty DOLG object/context support at 16x16.")
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
        "grid": {
            **observed,
            "object_empty": int((object_count == 0).sum()),
            "context_empty": int((context_count == 0).sum()),
        },
        "dataset": dataset_summary,
        "loader": loader_summary,
    }


def orthogonal_component(local: Tensor, global_feature: Tensor) -> Tensor:
    if local.ndim < 2 or global_feature.ndim != 2:
        raise ValueError("Local/global tensors must be [B,...,C] and [B,C].")
    if int(local.size(0)) != int(global_feature.size(0)):
        raise ValueError("Local/global batch dimensions differ.")
    if int(local.size(-1)) != int(global_feature.size(-1)):
        raise ValueError("Local/global channel dimensions differ.")
    batch_size = int(local.size(0))
    channels = int(local.size(-1))
    global_view = global_feature.reshape(
        batch_size, *([1] * (local.ndim - 2)), channels
    )
    denominator = global_feature.square().sum(dim=-1)
    if bool((denominator <= 0).any()):
        raise ValueError("DOLG global feature has zero norm.")
    denominator = denominator.reshape(batch_size, *([1] * (local.ndim - 1)))
    coefficient = (local * global_view).sum(dim=-1, keepdim=True) / denominator
    return local - coefficient * global_view


class DOLGObjectOrthogonalSidecar(nn.Module):
    def forward(
        self,
        local_tokens: Tensor,
        global_feature: Tensor,
        object_weights: Tensor,
    ) -> Tensor:
        local = F.normalize(local_tokens.float(), p=2, dim=-1, eps=1e-12)
        global_float = global_feature.float()
        weights = object_weights.to(dtype=torch.float32).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        object_local = (local * weights).sum(dim=1) / denominator
        object_orth = orthogonal_component(object_local, global_float)
        return torch.cat((global_float, object_orth), dim=-1)


def descriptor_roles(
    local_tokens: Tensor,
    global_feature: Tensor,
    bboxes: Tensor,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    if tuple(local_tokens.shape[1:]) != (PATCH_TOKENS, CHANNELS):
        raise ValueError(f"Locked local token shape differs: {tuple(local_tokens.shape)}")
    if tuple(global_feature.shape[1:]) != (CHANNELS,):
        raise ValueError(f"Locked global feature shape differs: {tuple(global_feature.shape)}")
    local = F.normalize(local_tokens.float(), p=2, dim=-1, eps=1e-12)
    global_float = global_feature.float()
    mask = object_center_mask(bboxes).to(device=local.device)
    weights = mask.to(dtype=local.dtype).unsqueeze(-1)
    object_count = weights.sum(dim=1)
    if bool((object_count <= 0).any()):
        raise ValueError("Object-local descriptor has empty support.")
    full_local = local.mean(dim=1)
    object_local = (local * weights).sum(dim=1) / object_count
    full_orth = orthogonal_component(full_local, global_float)
    object_orth = orthogonal_component(object_local, global_float)
    cell_orth = orthogonal_component(local, global_float)
    object_projection = object_local - object_orth
    full_projection = full_local - full_orth
    object_energy = object_local.square().sum(dim=1).clamp_min(1e-12)
    full_energy = full_local.square().sum(dim=1).clamp_min(1e-12)
    roles = {
        "global": global_float,
        "full_local": full_local,
        "object_local": object_local,
        "global_object_concat": torch.cat((global_float, object_local), dim=1),
        "global_full_orth": torch.cat((global_float, full_orth), dim=1),
        "global_object_orth": torch.cat((global_float, object_orth), dim=1),
        "bbox_geometry": bbox_geometry_descriptor(bboxes).to(device=local.device),
    }
    mechanism = {
        "global_norm": global_float.norm(dim=1),
        "object_local_norm": object_local.norm(dim=1),
        "object_orth_norm": object_orth.norm(dim=1),
        "full_orth_norm": full_orth.norm(dim=1),
        "object_global_cosine": F.cosine_similarity(
            object_local, global_float, dim=1, eps=1e-12
        ),
        "object_parallel_energy_ratio": (
            object_projection.square().sum(dim=1) / object_energy
        ),
        "full_parallel_energy_ratio": (
            full_projection.square().sum(dim=1) / full_energy
        ),
        "cell_orth_energy": cell_orth.norm(dim=-1),
        "object_mask": mask,
    }
    return roles, mechanism


def _assignment_target_name(statement: ast.stmt) -> Optional[str]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _load_official_orthogonal_function(source_path: Path):
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    dolg_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DOLG"
        ),
        None,
    )
    if dolg_class is None:
        raise ValueError("Official DOLG class was not found.")
    forward = next(
        (
            node
            for node in dolg_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ),
        None,
    )
    if forward is None:
        raise ValueError("Official DOLG forward method was not found.")
    expected = ["fg_norm", "proj", "proj", "proj", "orth_comp"]
    selected: list[ast.stmt] = []
    observed: list[str] = []
    collecting = False
    for statement in forward.body:
        name = _assignment_target_name(statement)
        if name == "fg_norm":
            collecting = True
        if collecting and name in {"fg_norm", "proj", "orth_comp"}:
            selected.append(statement)
            observed.append(str(name))
            if name == "orth_comp":
                break
    if observed != expected:
        raise ValueError(f"Official DOLG equation assignments differ: {observed}")
    segments = [ast.get_source_segment(source, statement) for statement in selected]
    if any(segment is None for segment in segments):
        raise ValueError("Official DOLG equation source extraction failed.")
    body = "\n".join(
        "\n".join(f"    {line}" for line in str(segment).splitlines())
        for segment in segments
    )
    code = f"def official_orthogonal(fg, fl):\n{body}\n    return orth_comp\n"
    namespace = {"torch": torch}
    exec(compile(code, str(source_path), "exec"), namespace)
    return namespace["official_orthogonal"], observed


def _independent_orthogonal(local: np.ndarray, global_feature: np.ndarray) -> np.ndarray:
    local = np.asarray(local, dtype=np.float64)
    global_feature = np.asarray(global_feature, dtype=np.float64)
    output = np.empty_like(local)
    for batch_index in range(local.shape[0]):
        global_vector = global_feature[batch_index]
        denominator = float(np.dot(global_vector, global_vector))
        if denominator <= 0.0:
            raise ValueError("Independent DOLG oracle received a zero global vector.")
        flattened = local[batch_index].reshape(-1, local.shape[-1])
        projected = np.empty_like(flattened)
        for row_index, local_vector in enumerate(flattened):
            coefficient = float(np.dot(local_vector, global_vector)) / denominator
            projected[row_index] = local_vector - coefficient * global_vector
        output[batch_index] = projected.reshape(local[batch_index].shape)
    return output


def _finite_difference_error() -> float:
    generator = torch.Generator().manual_seed(SEED + 301)
    local = torch.randn(1, 3, 5, generator=generator, dtype=torch.float64)
    global_feature = torch.randn(1, 5, generator=generator, dtype=torch.float64)
    upstream = torch.linspace(-0.2, 0.3, steps=15, dtype=torch.float64).reshape(1, 3, 5)
    local_grad = local.clone().requires_grad_(True)
    global_grad = global_feature.clone().requires_grad_(True)
    value = (orthogonal_component(local_grad, global_grad) * upstream).sum()
    analytic_local, analytic_global = torch.autograd.grad(
        value, (local_grad, global_grad)
    )
    step = 1e-6
    errors = []
    for tensor_name, index, analytic in (
        ("local", (0, 1, 2), analytic_local[0, 1, 2]),
        ("global", (0, 3), analytic_global[0, 3]),
    ):
        plus_local = local.clone()
        minus_local = local.clone()
        plus_global = global_feature.clone()
        minus_global = global_feature.clone()
        if tensor_name == "local":
            plus_local[index] += step
            minus_local[index] -= step
        else:
            plus_global[index] += step
            minus_global[index] -= step
        plus = (orthogonal_component(plus_local, plus_global) * upstream).sum()
        minus = (orthogonal_component(minus_local, minus_global) * upstream).sum()
        numeric = (plus - minus) / (2.0 * step)
        errors.append(float((numeric - analytic).abs()))
    return max(errors)


def _equation_diagnostics(source_path: Path, device: torch.device) -> Dict[str, object]:
    official, assignment_names = _load_official_orthogonal_function(source_path)
    generator = torch.Generator().manual_seed(SEED + 201)
    global_feature = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    local_map = torch.randn(3, 7, 4, 5, generator=generator, dtype=torch.float64)
    local_tokens = local_map.permute(0, 2, 3, 1).reshape(3, 20, 7).contiguous()
    official_output = official(global_feature, local_map)
    local_output = orthogonal_component(local_tokens, global_feature)
    local_output_map = local_output.reshape(3, 4, 5, 7).permute(0, 3, 1, 2)
    oracle = _independent_orthogonal(
        local_tokens.numpy(), global_feature.numpy()
    )

    official_local = local_map.clone().requires_grad_(True)
    official_global = global_feature.clone().requires_grad_(True)
    local_local = local_tokens.clone().requires_grad_(True)
    local_global = global_feature.clone().requires_grad_(True)
    upstream_map = torch.linspace(
        -0.4, 0.5, steps=official_output.numel(), dtype=torch.float64
    ).reshape_as(official_output)
    upstream_tokens = upstream_map.permute(0, 2, 3, 1).reshape_as(local_tokens)
    official_value = (official(official_global, official_local) * upstream_map).sum()
    local_value = (
        orthogonal_component(local_local, local_global) * upstream_tokens
    ).sum()
    official_local_grad, official_global_grad = torch.autograd.grad(
        official_value, (official_local, official_global)
    )
    local_local_grad, local_global_grad = torch.autograd.grad(
        local_value, (local_local, local_global)
    )
    local_local_grad_map = local_local_grad.reshape(3, 4, 5, 7).permute(0, 3, 1, 2)

    pooled_then_project = orthogonal_component(
        local_tokens.mean(dim=1), global_feature
    )
    projected_then_pool = local_output.mean(dim=1)
    dot = (local_output * global_feature[:, None, :]).sum(dim=-1)
    scale = (
        local_output.norm(dim=-1) * global_feature.norm(dim=-1)[:, None]
    ).clamp_min(1e-12)

    bf16_supported = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    bf16_max_error = float("inf")
    bf16_gradient_finite = False
    bf16_gradient_nonzero = False
    object_mask_repeat_exact = False
    if bf16_supported:
        sidecar = DOLGObjectOrthogonalSidecar().to(device).eval()
        token_values = torch.randn(
            2, PATCH_TOKENS, CHANNELS, generator=generator, dtype=torch.float32
        )
        global_values = torch.randn(
            2, CHANNELS, generator=generator, dtype=torch.float32
        )
        boxes = torch.tensor(
            [[0.5, 0.5, 0.70, 0.80], [0.45, 0.55, 0.62, 0.72]],
            dtype=torch.float32,
        )
        object_weights = object_center_mask(boxes).to(dtype=torch.float32)
        reference = sidecar(
            token_values.to(device),
            global_values.to(device),
            object_weights.to(device),
        )
        bf16_tokens = token_values.to(device=device, dtype=torch.bfloat16).requires_grad_(
            True
        )
        bf16_global = global_values.to(
            device=device, dtype=torch.bfloat16
        ).requires_grad_(True)
        observed = sidecar(
            bf16_tokens, bf16_global, object_weights.to(device)
        )
        loss = observed.square().mean()
        token_gradient, global_gradient = torch.autograd.grad(
            loss, (bf16_tokens, bf16_global)
        )
        bf16_max_error = float((reference - observed).abs().max())
        bf16_gradient_finite = bool(
            torch.isfinite(token_gradient).all()
            and torch.isfinite(global_gradient).all()
        )
        bf16_gradient_nonzero = bool(
            torch.count_nonzero(token_gradient) > 0
            and torch.count_nonzero(global_gradient) > 0
        )
        object_mask_repeat_exact = torch.equal(
            object_weights, object_center_mask(boxes).to(dtype=torch.float32)
        )
        sidecar.cpu()
        del sidecar, reference, observed, bf16_tokens, bf16_global

    return {
        "official_assignment_names": assignment_names,
        "official_assignments_exact": assignment_names
        == ["fg_norm", "proj", "proj", "proj", "orth_comp"],
        "official_output_max_abs_error": float(
            (official_output - local_output_map).abs().max()
        ),
        "oracle_output_max_abs_error": float(
            np.max(np.abs(oracle - local_output.detach().numpy()))
        ),
        "local_gradient_max_abs_error": float(
            (official_local_grad - local_local_grad_map).abs().max()
        ),
        "global_gradient_max_abs_error": float(
            (official_global_grad - local_global_grad).abs().max()
        ),
        "pooling_equivalence_max_abs_error": float(
            (pooled_then_project - projected_then_pool).abs().max()
        ),
        "orthogonality_relative_error": float((dot.abs() / scale).max()),
        "finite_difference_error": _finite_difference_error(),
        "bf16_supported": bf16_supported,
        "bf16_max_abs_error": bf16_max_error,
        "bf16_gradient_finite": bf16_gradient_finite,
        "bf16_gradient_nonzero": bf16_gradient_nonzero,
        "object_mask_repeat_exact": object_mask_repeat_exact,
        "all_outputs_finite": bool(
            torch.isfinite(official_output).all()
            and torch.isfinite(local_output).all()
            and np.isfinite(oracle).all()
        ),
    }


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
        context="dolg_a0_declaration_replay",
    )
    model = model.to(device).eval()
    observed_rows: list[Dict[str, object]] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("DOLG declaration replay lacks sample_index metadata.")
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
        raise ValueError("DOLG declaration replay row count differs.")
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


class StandardForwardFeatureCapture:
    def __init__(self, model: nn.Module) -> None:
        patch_embed = getattr(model, "patch_embed", None)
        head = getattr(model, "head", None)
        if not isinstance(patch_embed, nn.Module) or not isinstance(head, nn.Module):
            raise TypeError("DOLG A0 requires patch_embed and main head modules.")
        self.patch_embed = patch_embed
        self.head = head
        self.local_values: list[Tensor] = []
        self.global_values: list[Tensor] = []
        self._handles: list[object] = []

    def _patch_hook(self, _module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Patch embedding hook did not receive a tensor.")
        self.local_values.append(output)

    def _head_pre_hook(self, _module, inputs) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise TypeError("Main-head hook did not receive a tensor input.")
        self.global_values.append(inputs[0])

    def __enter__(self):
        self._handles = [
            self.patch_embed.register_forward_hook(self._patch_hook),
            self.head.register_forward_pre_hook(self._head_pre_hook),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.local_values.clear()
        self.global_values.clear()

    def pop(self) -> tuple[Tensor, Tensor, Dict[str, int]]:
        counts = {
            "patch_embed": len(self.local_values),
            "main_head": len(self.global_values),
        }
        if counts != {"patch_embed": 1, "main_head": 1}:
            raise RuntimeError(f"Locked DOLG hook counts differ: {counts}")
        local = self.local_values.pop()
        global_feature = self.global_values.pop()
        return local, global_feature, counts


def _extract_descriptors(
    *,
    model: nn.Module,
    dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    locked_bboxes: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, object],
]:
    model = model.to(device).eval()
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    descriptors: Dict[str, Dict[str, np.ndarray]] = {}
    mechanism_values: Dict[str, Dict[str, np.ndarray]] = {}
    loader_summaries: Dict[str, object] = {}
    capture_count_records: list[Dict[str, int]] = []
    clean_cell_energy: Optional[np.ndarray] = None
    clean_keeper_p1: Optional[np.ndarray] = None

    with StandardForwardFeatureCapture(model) as capture, torch.inference_mode():
        for condition, brightness, contrast in CONDITIONS:
            loader, loader_summary = _make_condition_loader(
                base_dataset=dataset,
                transform=transform,
                indices=indices,
                brightness=brightness,
                contrast=contrast,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context=f"dolg_a0_descriptor_{condition}",
            )
            loader_summaries[condition] = loader_summary
            role_batches: Dict[str, list[Tensor]] = {role: [] for role in ROLES}
            mechanism_batches: Dict[str, list[Tensor]] = {
                field: [] for field in MECHANISM_FIELDS
            }
            cell_energy_batches: list[Tensor] = []
            keeper_p1_batches: list[Tensor] = []
            bbox_batches: list[Tensor] = []
            observed_indices: list[int] = []
            observed_targets: list[int] = []
            for images, targets, metadata in loader:
                sample_indices = metadata.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(
                    crop_bboxes
                ):
                    raise ValueError(
                        f"{condition} DOLG loader lacks sample_index/crop_bbox."
                    )
                images = images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                boxes = crop_bboxes[:, :4].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                capture.reset()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    logits, _ = _forward_classification_with_metadata(
                        model,
                        images,
                        metadata,
                        device=device,
                    )
                local_tokens, global_feature, counts = capture.pop()
                capture_count_records.append(counts)
                roles, mechanism = descriptor_roles(
                    local_tokens,
                    global_feature,
                    boxes,
                )
                for role in ROLES:
                    role_batches[role].append(roles[role].detach().cpu())
                for field in MECHANISM_FIELDS:
                    mechanism_batches[field].append(
                        mechanism[field].detach().float().cpu()
                    )
                cell_energy_batches.append(
                    mechanism["cell_orth_energy"].detach().float().cpu()
                )
                keeper_p1_batches.append(
                    torch.softmax(logits.float(), dim=1)[:, FOCUS_CLASS].cpu()
                )
                bbox_batches.append(boxes.cpu())
                observed_indices.extend(int(value) for value in sample_indices.tolist())
                observed_targets.extend(int(value) for value in targets.tolist())
            if observed_indices != indices or observed_targets != expected_targets:
                raise ValueError(f"{condition} DOLG descriptor order/targets differ.")
            observed_bboxes = torch.cat(bbox_batches, dim=0).contiguous()
            if not torch.equal(observed_bboxes, locked_bboxes):
                raise ValueError(f"{condition} transformed bbox differs from CPU lock.")
            condition_roles: Dict[str, np.ndarray] = {}
            for role in ROLES:
                values = torch.cat(role_batches[role], dim=0).float().numpy()
                expected_shape = (EXPECTED_COHORT_ROWS, ROLE_DIMS[role])
                if values.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected {condition}/{role} shape: {values.shape}"
                    )
                if not bool(np.isfinite(values).all()):
                    raise ValueError(f"Non-finite {condition}/{role} descriptor.")
                condition_roles[role] = values.astype(np.float64)
            descriptors[condition] = condition_roles
            condition_mechanism = {
                field: torch.cat(mechanism_batches[field], dim=0).numpy().astype(
                    np.float64
                )
                for field in MECHANISM_FIELDS
            }
            mechanism_values[condition] = condition_mechanism
            if condition == "clean":
                clean_cell_energy = (
                    torch.cat(cell_energy_batches, dim=0).numpy().astype(np.float32)
                )
                clean_keeper_p1 = (
                    torch.cat(keeper_p1_batches, dim=0).numpy().astype(np.float64)
                )

    if clean_cell_energy is None or clean_keeper_p1 is None:
        raise RuntimeError("Clean DOLG mechanism cache was not created.")
    mechanism_values["clean"]["cell_orth_energy"] = clean_cell_energy
    mechanism_values["clean"]["keeper_p1"] = clean_keeper_p1
    candidate = descriptors["clean"]["global_object_orth"]
    concat = descriptors["clean"]["global_object_concat"]
    feature_variance = np.var(candidate, axis=0, dtype=np.float64)
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
        "global_norm_min": float(
            min(values["global_norm"].min() for values in mechanism_values.values())
        ),
        "candidate_concat_max_abs_delta": float(
            np.max(np.abs(candidate - concat))
        ),
        "candidate_feature_variance_min": float(feature_variance.min()),
        "candidate_features_all_nonzero_variance": bool(
            np.all(feature_variance > 1e-12)
        ),
        "hook_calls_exact": all(
            record == {"patch_embed": 1, "main_head": 1}
            for record in capture_count_records
        ),
        "hook_call_records": len(capture_count_records),
        "feature_cache_written_to_disk": False,
        "cell_energy_cache_shape": list(clean_cell_energy.shape),
    }
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return descriptors, mechanism_values, telemetry


def select_recall_constrained_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_tp_retention: float = MIN_FIT_TP_RETENTION,
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = scores[labels == 1]
    if positives.size == 0:
        raise ValueError("Recall-constrained threshold has no positive rows.")
    candidates = sorted(set(float(value) for value in scores), reverse=True)
    valid = [
        threshold
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
            condition_rows: Dict[str, int] = {}
            for condition, _, _ in CONDITIONS:
                hold_values = (
                    np.asarray(descriptors[condition][role], dtype=np.float64)[
                        hold_mask
                    ]
                    - mean
                ) / std
                hold_scores = readout.predict_proba(hold_values)[:, 1]
                scores[condition][role][hold_mask] = hold_scores
                condition_rows[condition] = int(hold_scores.size)
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
                    "condition_hold_rows": condition_rows,
                }
            )
    if convergence_failures:
        raise RuntimeError(
            f"Locked DOLG logistic readout did not converge: {convergence_failures}"
        )
    if not all(
        bool(np.isfinite(values).all())
        for condition_scores in scores.values()
        for values in condition_scores.values()
    ):
        raise ValueError("DOLG OOF score matrix is incomplete/non-finite.")
    if not all(bool(np.isfinite(values).all()) for values in thresholds.values()):
        raise ValueError("DOLG OOF thresholds are incomplete/non-finite.")
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
    candidate = clean["global_object_orth"]
    global_control = clean["global"]
    concat = clean["global_object_concat"]
    full_orth = clean["global_full_orth"]
    geometry = clean["bbox_geometry"]
    fold_deltas = {
        str(fold): float(candidate["fold_auroc"][str(fold)])
        - float(concat["fold_auroc"][str(fold)])
        for fold in FIT_FOLDS
    }
    shifted_deltas = {
        condition: float(conditions[condition]["global_object_orth"]["auroc"])
        - float(conditions[condition]["global_object_concat"]["auroc"])
        for condition, _, _ in CONDITIONS
        if condition != "clean"
    }
    shifted_tp = {
        condition: float(
            conditions[condition]["global_object_orth"]["tp_retention"]
        )
        for condition, _, _ in CONDITIONS
        if condition != "clean"
    }
    shifted_fp = {
        condition: float(
            conditions[condition]["global_object_orth"]["fp_rejection"]
        )
        for condition, _, _ in CONDITIONS
        if condition != "clean"
    }
    checks = {
        "clean_auroc_at_least_067": float(candidate["auroc"]) >= 0.67,
        "clean_gain_over_global_at_least_0025": (
            float(candidate["auroc"]) - float(global_control["auroc"]) >= 0.025
        ),
        "clean_gain_over_concat_at_least_0015": (
            float(candidate["auroc"]) - float(concat["auroc"]) >= 0.015
        ),
        "clean_gain_over_full_orth_at_least_0015": (
            float(candidate["auroc"]) - float(full_orth["auroc"]) >= 0.015
        ),
        "clean_gain_over_bbox_geometry_at_least_005": (
            float(candidate["auroc"]) - float(geometry["auroc"]) >= 0.05
        ),
        "at_least_three_positive_fold_deltas_vs_concat": sum(
            value > 0.0 for value in fold_deltas.values()
        )
        >= 3,
        "no_fold_delta_below_minus002": min(fold_deltas.values()) >= -0.02,
        "mean_shifted_gain_vs_concat_at_least_001": statistics.mean(
            shifted_deltas.values()
        )
        >= 0.01,
        "no_shifted_delta_below_minus0015": min(shifted_deltas.values())
        >= -0.015,
        "clean_tp_retention_at_least_090": float(candidate["tp_retention"])
        >= 0.90,
        "clean_fp_rejection_at_least_025": float(candidate["fp_rejection"])
        >= 0.25,
        "clean_fp_rejection_gain_vs_concat_at_least_005": (
            float(candidate["fp_rejection"]) - float(concat["fp_rejection"])
            >= 0.05
        ),
        "clean_precision_at_least_074": float(candidate["precision"]) >= 0.74,
        "clean_precision_gain_vs_concat_at_least_001": (
            float(candidate["precision"]) - float(concat["precision"]) >= 0.01
        ),
        "clean_f1_not_below_concat": float(candidate["f1"])
        >= float(concat["f1"]),
        "all_shifted_tp_retention_at_least_088": min(shifted_tp.values())
        >= 0.88,
        "all_shifted_fp_rejection_at_least_018": min(shifted_fp.values())
        >= 0.18,
        "all_descriptors_finite": bool(telemetry["all_finite"]),
        "all_global_norms_nonzero": float(telemetry["global_norm_min"]) > 0.0,
        "candidate_differs_from_concat": float(
            telemetry["candidate_concat_max_abs_delta"]
        )
        > 1e-6,
        "candidate_features_nonzero_variance": bool(
            telemetry["candidate_features_all_nonzero_variance"]
        ),
        "standard_forward_hook_calls_exact": bool(telemetry["hook_calls_exact"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "fold_auroc_deltas_vs_concat": fold_deltas,
        "shifted_auroc_deltas_vs_concat": shifted_deltas,
        "shifted_tp_retention": shifted_tp,
        "shifted_fp_rejection": shifted_fp,
    }


class _PatchEmbeddingCaptureModule(nn.Module):
    def __init__(self, patch_embed: nn.Module) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.last_tokens: Optional[Tensor] = None
        self.call_count = 0
        for name in ("num_patches", "patch_size", "base_grid_size"):
            if hasattr(patch_embed, name):
                setattr(self, name, getattr(patch_embed, name))

    def reset_capture(self) -> None:
        self.last_tokens = None
        self.call_count = 0

    def forward(self, features: Tensor) -> Tensor:
        tokens = self.patch_embed(features)
        self.last_tokens = tokens
        self.call_count += 1
        return tokens


class _HeadWithDOLGSidecar(nn.Module):
    def __init__(self, head: nn.Module, patch_capture: _PatchEmbeddingCaptureModule) -> None:
        super().__init__()
        self.head = head
        self.sidecar = DOLGObjectOrthogonalSidecar()
        self._patch_capture = weakref.ref(patch_capture)
        self.current_object_weights: Optional[Tensor] = None
        self.last_descriptor: Optional[Tensor] = None
        self.call_count = 0

    def reset_capture(self) -> None:
        self.last_descriptor = None
        self.call_count = 0

    def forward(self, global_feature: Tensor) -> Tensor:
        patch_capture = self._patch_capture()
        if patch_capture is None or not torch.is_tensor(patch_capture.last_tokens):
            raise RuntimeError("DOLG benchmark patch capture is missing.")
        if self.current_object_weights is None:
            raise RuntimeError("DOLG benchmark object weights were not set.")
        self.last_descriptor = self.sidecar(
            patch_capture.last_tokens,
            global_feature,
            self.current_object_weights,
        )
        self.call_count += 1
        return self.head(global_feature)


class _FullModelBenchmarkWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, with_descriptor: bool) -> None:
        super().__init__()
        self.model = model
        self.with_descriptor = bool(with_descriptor)
        self.patch_capture: Optional[_PatchEmbeddingCaptureModule] = None
        self.head_sidecar: Optional[_HeadWithDOLGSidecar] = None
        if self.with_descriptor:
            patch_capture = _PatchEmbeddingCaptureModule(self.model.patch_embed)
            head_sidecar = _HeadWithDOLGSidecar(self.model.head, patch_capture)
            self.model.patch_embed = patch_capture
            self.model.head = head_sidecar
            self.patch_capture = patch_capture
            self.head_sidecar = head_sidecar

    def forward(
        self,
        images: Tensor,
        image_mask: Tensor,
        bbox: Tensor,
        crop_bbox: Tensor,
    ) -> Tensor:
        if self.patch_capture is not None and self.head_sidecar is not None:
            self.patch_capture.reset_capture()
            self.head_sidecar.reset_capture()
            self.head_sidecar.current_object_weights = object_center_mask(
                crop_bbox
            ).to(dtype=torch.float32)
        logits, _ = _forward_classification_with_metadata(
            self.model,
            images,
            {"image_mask": image_mask, "bbox": bbox},
            device=images.device,
        )
        if not torch.is_tensor(logits):
            raise TypeError("DOLG benchmark model output lacks logits.")
        if self.patch_capture is not None and self.head_sidecar is not None:
            if self.patch_capture.call_count != 1 or self.head_sidecar.call_count != 1:
                raise RuntimeError(
                    "DOLG benchmark capture count differs: "
                    f"patch={self.patch_capture.call_count}, "
                    f"head={self.head_sidecar.call_count}"
                )
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
    capture_counts: Optional[Dict[str, int]] = None
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
        if wrapper.head_sidecar is not None and wrapper.patch_capture is not None:
            descriptor = wrapper.head_sidecar.last_descriptor
            if not torch.is_tensor(descriptor):
                raise RuntimeError("DOLG benchmark did not execute the sidecar.")
            descriptor_shape = list(descriptor.shape)
            capture_counts = {
                "patch_embed": int(wrapper.patch_capture.call_count),
                "main_head": int(wrapper.head_sidecar.call_count),
            }
    if last_logits is None:
        raise RuntimeError("DOLG benchmark produced no logits.")
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
        "capture_counts": capture_counts,
        "logits": logits_cpu,
    }


def _gpu_snapshot() -> Dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,"
        "memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"available": True, "raw": result.stdout.strip()}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": repr(exc)}


def _resource_audit(
    model: nn.Module, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    before = _gpu_snapshot()
    native = _benchmark_model_role(
        model, with_descriptor=False, device=device, repeats=repeats
    )
    candidate = _benchmark_model_role(
        model, with_descriptor=True, device=device, repeats=repeats
    )
    after = _gpu_snapshot()
    native_logits = native.pop("logits")
    candidate_logits = candidate.pop("logits")
    logit_error = float((native_logits - candidate_logits).abs().max())
    prediction_mismatches = int(
        (native_logits.argmax(dim=1) != candidate_logits.argmax(dim=1)).sum()
    )
    return {
        "gpu_before": before,
        "gpu_after": after,
        "native": native,
        "global_object_orth_sidecar": candidate,
        "runtime_ratio": float(candidate["median_ms"])
        / float(native["median_ms"]),
        "peak_memory_ratio": float(candidate["peak_allocated_bytes"])
        / float(native["peak_allocated_bytes"]),
        "normal_forward_logit_max_abs_error": logit_error,
        "normal_forward_prediction_mismatches": prediction_mismatches,
        "trace_path_used": False,
        "descriptor_executed": candidate["descriptor_shape"] == [32, 512],
        "capture_counts_exact": candidate["capture_counts"]
        == {"patch_embed": 1, "main_head": 1},
        "added_trainable_parameters": 0,
    }


def _onnx_audit() -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    sidecar = DOLGObjectOrthogonalSidecar().cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 601)
    local_tokens = torch.randn(
        1, PATCH_TOKENS, CHANNELS, generator=generator, dtype=torch.float32
    )
    global_feature = torch.randn(
        1, CHANNELS, generator=generator, dtype=torch.float32
    )
    object_weights = object_center_mask(
        torch.tensor([[0.5, 0.5, 0.70, 0.80]], dtype=torch.float32)
    ).to(dtype=torch.float32)
    with torch.inference_mode():
        reference = sidecar(local_tokens, global_feature, object_weights).numpy()
    with tempfile.TemporaryDirectory(prefix="trkh_dolg_onnx_") as directory:
        path = Path(directory) / "dolg_object_orthogonal_sidecar.onnx"
        torch.onnx.export(
            sidecar,
            (local_tokens, global_feature, object_weights),
            str(path),
            input_names=["local_tokens", "global_feature", "object_weights"],
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
            {
                "local_tokens": local_tokens.numpy(),
                "global_feature": global_feature.numpy(),
                "object_weights": object_weights.numpy(),
            },
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
            "local_tokens": list(local_tokens.shape),
            "global_feature": list(global_feature.shape),
            "object_weights": list(object_weights.shape),
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
        "official_assignments_exact": bool(equation["official_assignments_exact"]),
        "official_output_error_within_limit": float(
            equation["official_output_max_abs_error"]
        )
        <= MAX_OFFICIAL_ERROR,
        "oracle_output_error_within_limit": float(
            equation["oracle_output_max_abs_error"]
        )
        <= MAX_ORACLE_ERROR,
        "local_gradient_error_within_limit": float(
            equation["local_gradient_max_abs_error"]
        )
        <= MAX_GRADIENT_ERROR,
        "global_gradient_error_within_limit": float(
            equation["global_gradient_max_abs_error"]
        )
        <= MAX_GRADIENT_ERROR,
        "pooling_equivalence_within_limit": float(
            equation["pooling_equivalence_max_abs_error"]
        )
        <= MAX_POOLING_EQUIVALENCE_ERROR,
        "orthogonality_within_limit": float(
            equation["orthogonality_relative_error"]
        )
        <= MAX_ORTHOGONALITY_ERROR,
        "finite_difference_within_limit": float(
            equation["finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "bf16_supported": bool(equation["bf16_supported"]),
        "bf16_error_within_limit": float(equation["bf16_max_abs_error"])
        <= MAX_BF16_ERROR,
        "bf16_gradient_finite_nonzero": bool(
            equation["bf16_gradient_finite"]
            and equation["bf16_gradient_nonzero"]
        ),
        "object_mask_repeat_exact": bool(equation["object_mask_repeat_exact"]),
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
        "capture_counts_exact": bool(resource["capture_counts_exact"]),
        "trace_path_not_used": not bool(resource["trace_path_used"]),
        "no_added_trainable_parameters": int(resource["added_trainable_parameters"])
        == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _write_oof_csv(
    *,
    path: Path,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
    mechanism: Mapping[str, Mapping[str, np.ndarray]],
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
        "keeper_p1_clean",
        *[f"score_{role}" for role in ROLES],
        *[f"threshold_{role}" for role in ROLES],
        *MECHANISM_FIELDS,
    ]
    keeper_p1 = np.asarray(mechanism["clean"]["keeper_p1"], dtype=np.float64)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, _, _ in CONDITIONS:
            condition_mechanism = mechanism[condition]
            for position, row in enumerate(cohort):
                payload: Dict[str, object] = {
                    "sample_index": row.sample_index,
                    "source_stem": row.source_stem,
                    "image_path": str(row.image_path),
                    "fold": row.fold,
                    "target": row.target,
                    "keeper_prediction": row.keeper_prediction,
                    "cohort": _cohort_label(row),
                    "condition": condition,
                    "keeper_p1_clean": format(float(keeper_p1[position]), ".17g"),
                }
                for role in ROLES:
                    payload[f"score_{role}"] = format(
                        float(scores[condition][role][position]), ".17g"
                    )
                    payload[f"threshold_{role}"] = format(
                        float(thresholds[role][position]), ".17g"
                    )
                for field in MECHANISM_FIELDS:
                    payload[field] = format(
                        float(condition_mechanism[field][position]), ".17g"
                    )
                writer.writerow(payload)


def _replay_oof_csv(path: Path) -> Dict[str, object]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = EXPECTED_COHORT_ROWS * len(CONDITIONS)
    if len(rows) != expected_rows:
        raise ValueError(f"DOLG OOF CSV row count differs: {len(rows)}")
    condition_names = [name for name, _, _ in CONDITIONS]
    metrics: Dict[str, object] = {"conditions": {}}
    ordered_hashes: Dict[str, str] = {}
    for condition in condition_names:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if len(condition_rows) != EXPECTED_COHORT_ROWS:
            raise ValueError(f"DOLG OOF CSV {condition} count differs.")
        sample_indices = [int(row["sample_index"]) for row in condition_rows]
        ordered_hash = _ordered_index_sha256(sample_indices)
        if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
            raise ValueError(f"DOLG OOF CSV {condition} order differs.")
        ordered_hashes[condition] = ordered_hash
        labels = np.asarray(
            [1 if row["cohort"] == "tp" else 0 for row in condition_rows],
            dtype=np.int64,
        )
        folds = np.asarray([int(row["fold"]) for row in condition_rows])
        condition_metrics: Dict[str, object] = {}
        for role in ROLES:
            role_scores = np.asarray(
                [float(row[f"score_{role}"]) for row in condition_rows],
                dtype=np.float64,
            )
            role_thresholds = np.asarray(
                [float(row[f"threshold_{role}"]) for row in condition_rows],
                dtype=np.float64,
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
        metrics["conditions"][condition] = condition_metrics
    return {
        "rows": len(rows),
        "ordered_index_sha256": ordered_hashes,
        "metrics": metrics,
        "all_values_finite": all(
            math.isfinite(float(row[f"score_{role}"]))
            and math.isfinite(float(row[f"threshold_{role}"]))
            for row in rows
            for role in ROLES
        ),
    }


def _numeric_differences(left: object, right: object, prefix: str = "") -> list[float]:
    differences: list[float] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) & set(right)):
            differences.extend(
                _numeric_differences(left[key], right[key], f"{prefix}.{key}")
            )
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        differences.append(abs(float(left) - float(right)))
    return differences


def _select_visual_positions(
    *,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, list[int]]:
    candidate = np.asarray(scores["clean"]["global_object_orth"], dtype=np.float64)
    concat = np.asarray(scores["clean"]["global_object_concat"], dtype=np.float64)
    advantage = candidate - concat
    labels = np.asarray([_cohort_label(row) for row in cohort], dtype=object)
    selections: Dict[str, list[int]] = {}
    for cohort_name in ("tp", "fp"):
        positions = np.flatnonzero(labels == cohort_name)
        if positions.size < 16:
            raise ValueError(f"Insufficient DOLG visual rows for {cohort_name}.")
        order = positions[np.argsort(advantage[positions], kind="mergesort")]
        selections[f"positive_advantage_{cohort_name}"] = [
            int(value) for value in order[-8:][::-1]
        ]
        selections[f"negative_advantage_{cohort_name}"] = [
            int(value) for value in order[:8]
        ]
    return selections


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


def _energy_overlay(
    image: Image.Image,
    energy: np.ndarray,
    bbox: Sequence[float],
) -> Image.Image:
    values = np.asarray(energy, dtype=np.float64).reshape(GRID_SIZE, GRID_SIZE)
    minimum = float(values.min())
    maximum = float(values.max())
    normalized = (values - minimum) / max(1e-12, maximum - minimum)
    red = np.rint(255.0 * normalized).astype(np.uint8)
    green = np.rint(210.0 * np.sqrt(normalized)).astype(np.uint8)
    blue = np.rint(255.0 * (1.0 - normalized)).astype(np.uint8)
    heatmap = Image.fromarray(
        np.stack((red, green, blue), axis=-1), mode="RGB"
    ).resize(image.size, Image.Resampling.NEAREST)
    blended = Image.blend(image, heatmap, alpha=0.48)
    return _bbox_overlay(blended, bbox)


def _render_visuals(
    *,
    output_dir: Path,
    model_checkpoint: Mapping[str, object],
    dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
    mechanism: Mapping[str, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    selections = _select_visual_positions(cohort=cohort, scores=scores)
    semantics = _eval_semantics(model_checkpoint)
    page_dir = output_dir / "visual_review"
    page_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    page_records: list[Dict[str, object]] = []
    selection_rows: list[Dict[str, object]] = []
    clean_mechanism = mechanism["clean"]
    cell_energy = np.asarray(clean_mechanism["cell_orth_energy"])

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
            context=f"dolg_a0_visual_{category}",
        )
        image_batches: list[Tensor] = []
        bbox_batches: list[Tensor] = []
        observed_indices: list[int] = []
        for images, _, metadata in loader:
            crop_bboxes = metadata.get("crop_bbox")
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(sample_indices):
                raise ValueError("DOLG visual loader lacks crop_bbox/sample_index.")
            image_batches.append(images.float())
            bbox_batches.append(crop_bboxes[:, :4].float())
            observed_indices.extend(int(value) for value in sample_indices.tolist())
        if observed_indices != [row.sample_index for row in selected_rows]:
            raise ValueError(f"DOLG visual {category} order differs.")
        images = torch.cat(image_batches, dim=0)
        bboxes = torch.cat(bbox_batches, dim=0)
        rgb = _tensor_to_rgb(images, semantics)

        canvas = Image.new("RGB", (1180, 8 * 204 + 42), color=(250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), category, fill=(0, 0, 0), font=font)
        for row_index, (position, row) in enumerate(zip(positions, selected_rows)):
            y = 36 + row_index * 204
            source_image = _pil_from_rgb_tensor(rgb[row_index])
            source_image = _bbox_overlay(source_image, bboxes[row_index].tolist())
            source_image = source_image.resize((180, 180), Image.Resampling.BILINEAR)
            heatmap_image = _energy_overlay(
                _pil_from_rgb_tensor(rgb[row_index]),
                cell_energy[position],
                bboxes[row_index].tolist(),
            ).resize((180, 180), Image.Resampling.BILINEAR)
            canvas.paste(source_image, (12, y))
            canvas.paste(heatmap_image, (204, y))
            candidate_score = float(scores["clean"]["global_object_orth"][position])
            concat_score = float(scores["clean"]["global_object_concat"][position])
            candidate_threshold = float(thresholds["global_object_orth"][position])
            concat_threshold = float(thresholds["global_object_concat"][position])
            lines = [
                f"sample={row.sample_index} fold={row.fold} target={row.target} cohort={_cohort_label(row)}",
                f"keeper_p1={float(clean_mechanism['keeper_p1'][position]):.6f}",
                f"object_orth={candidate_score:.6f} threshold={candidate_threshold:.6f}",
                f"direct_concat={concat_score:.6f} threshold={concat_threshold:.6f}",
                f"advantage={candidate_score - concat_score:+.6f}",
                f"object_global_cosine={float(clean_mechanism['object_global_cosine'][position]):+.6f}",
                f"removed_parallel_energy={float(clean_mechanism['object_parallel_energy_ratio'][position]):.6f}",
                f"bbox={[round(float(value), 5) for value in bboxes[row_index].tolist()]}",
            ]
            for line_index, line in enumerate(lines):
                draw.text(
                    (400, y + 5 + line_index * 21),
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
                    "global_object_orth_score": candidate_score,
                    "global_object_concat_score": concat_score,
                    "global_object_orth_threshold": candidate_threshold,
                    "global_object_concat_threshold": concat_threshold,
                    "object_global_cosine": float(
                        clean_mechanism["object_global_cosine"][position]
                    ),
                    "object_parallel_energy_ratio": float(
                        clean_mechanism["object_parallel_energy_ratio"][position]
                    ),
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
            "selection_rule": "oof(global_object_orth)-oof(global_object_concat)",
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
        "selection_path": str(selection_path.relative_to(output_dir)).replace(
            "\\", "/"
        ),
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
    candidate = (
        clean.get("global_object_orth", {}) if isinstance(clean, Mapping) else {}
    )
    concat = (
        clean.get("global_object_concat", {}) if isinstance(clean, Mapping) else {}
    )
    lines = [
        "# DOLG Orthogonal Local-Global Signal A0 Report",
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
        f"- object orthogonal AUROC: `{candidate.get('auroc')}`",
        f"- direct concat AUROC: `{concat.get('auroc')}`",
        f"- object orthogonal precision: `{candidate.get('precision')}`",
        f"- object orthogonal TP retention: `{candidate.get('tp_retention')}`",
        f"- object orthogonal restricted-FP rejection: `{candidate.get('fp_rejection')}`",
        "",
        "## Decision",
        "",
        (
            "A0 is awaiting hash-locked review of all four contact sheets. It "
            "cannot authorize trainer integration until finalization."
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
        raise FileNotFoundError(f"Formal DOLG summary not found: {summary_path}")
    expected = str(args.expected_summary_sha256).strip().casefold()
    if not expected:
        raise ValueError("--expected-summary-sha256 is required for finalization.")
    observed = _sha256(summary_path)
    if observed != expected:
        raise ValueError(f"Formal DOLG summary SHA differs: {observed} != {expected}")
    if args.visual_review_result not in {"pass", "fail"}:
        raise ValueError("--visual-review-result pass|fail is required.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError(f"DOLG summary is not awaiting review: {summary.get('status')}")
    visual = summary.get("visual_review")
    if not isinstance(visual, Mapping) or not bool(visual.get("complete")):
        raise ValueError("DOLG visual artifact is incomplete and cannot be finalized.")
    pages = visual.get("pages")
    if not isinstance(pages, list) or len(pages) != 4:
        raise ValueError("DOLG visual page manifest differs from four pages.")
    for page in pages:
        page_path = output_dir / str(page["path"])
        if _sha256(page_path) != str(page["sha256"]):
            raise ValueError(f"DOLG visual page hash differs: {page_path}")
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
    device = torch.device(args.device)
    set_seed(int(args.seed), deterministic=True)
    model = _load_keeper_model(checkpoint)
    equation = _equation_diagnostics(
        Path(provenance["paths"]["model_source"]), device
    )
    declaration = _declaration_replay(
        model=model,
        dataset=dataset,
        transform=transform,
        cohort=cohort,
        args=args,
        device=device,
    )
    resource = _resource_audit(
        model,
        device=device,
        repeats=int(args.benchmark_repeats),
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
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
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

    descriptors, mechanism, telemetry = _extract_descriptors(
        model=model,
        dataset=dataset,
        transform=transform,
        cohort=cohort,
        locked_bboxes=locked_bboxes,
        args=args,
        device=device,
    )
    scores, thresholds, metrics = _run_oof(
        descriptors=descriptors,
        cohort=cohort,
    )
    signal_gate = assess_signal_gate(metrics, telemetry)
    oof_path = output_dir / "oof_scores_all_conditions.csv"
    _write_oof_csv(
        path=oof_path,
        cohort=cohort,
        scores=scores,
        thresholds=thresholds,
        mechanism=mechanism,
    )
    replay = _replay_oof_csv(oof_path)
    differences = _numeric_differences(
        metrics["conditions"], replay["metrics"]["conditions"]
    )
    replay_max_error = max(differences) if differences else float("inf")
    independent_replay = {
        **replay,
        "max_abs_metric_error": replay_max_error,
        "exact_within_1e12": replay_max_error <= 1e-12,
        "csv_sha256": _sha256(oof_path),
    }
    visual_review = _render_visuals(
        output_dir=output_dir,
        model_checkpoint=checkpoint,
        dataset=dataset,
        transform=transform,
        cohort=cohort,
        scores=scores,
        thresholds=thresholds,
        mechanism=mechanism,
        args=args,
    )
    automated_gate_passed = bool(
        structural_gate["passed"]
        and signal_gate["passed"]
        and independent_replay["exact_within_1e12"]
        and independent_replay["all_values_finite"]
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
        "descriptor_telemetry": telemetry,
        "oof_metrics": metrics,
        "signal_gate": signal_gate,
        "independent_replay": independent_replay,
        "visual_review": visual_review,
        "automated_gate_passed": automated_gate_passed,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "trainer_integration_authorized": False,
        "short_pair_authorized": False,
        "current_command_update_authorized": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir)
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **summary,
        "formal_summary_sha256": _sha256(output_dir / "summary.json"),
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

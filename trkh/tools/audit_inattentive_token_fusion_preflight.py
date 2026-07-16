from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.models.model import (
    VisionTransformerWithRegisters,
    create_model,
    load_model_state,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _failed_export,
    _onnx_compare,
    _rgb_from_tensor,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _classification_metrics,
)
from trkh.tools.audit_deformable_spatial_attention_preflight import (
    _inference_benchmark,
)
from trkh.tools.audit_foveal_aggregated_attention_pair import (
    CONDITIONS,
    _build_holdout_dataset,
    _condition_corruption,
    _make_loader,
    _metadata_to_device,
)
from trkh.tools.audit_foveal_aggregated_attention_preflight import (
    _FullExport,
    _benchmark,
    _build_dataset,
    _forward,
    _git_value,
    _load_json,
    _prepare_output,
    _sha256,
    _tracked_worktree_clean,
    _unpack_batch,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)


METHOD = "inattentive_token_fusion_a0_preflight"
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLD = 0
GRID_SIZE = 16
ORIGINAL_PATCH_COUNT = GRID_SIZE * GRID_SIZE
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_SOURCES = 6_452
EXPECTED_HOLDOUT_SOURCES = 1_612
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
EXPECTED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_OFFICIAL_COMMIT = "97e58f610c51d4b74a070341739e41647dced32c"
LOCKED_OFFICIAL_TREE = "7d907e81f4a764972baa96a639a16164af6f08a6"
LOCKED_HASHES = {
    "protocol": "d3f8de6fbef7ad5d88ca33e5db71f9c5ca587f32c1ad5ee4797fd400aaf3ee4b",
    "raw_data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "declaration": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "declaration_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "resolved_config": "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97",
    "scratch_launcher": "dac9977b13249ffb71ffd74339d3c8980e504f2652788c45e677c5aeeca587fe",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "fold_data": "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e",
    "fold_summary": "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763",
    "bra_closure": "ab74dc74d7d25b0d1062db14d12654c11e3febfd498a1a473d80d3a4625766e4",
    "official_evit": "84b164a630d3c8fad2925813bb894e2a796608f029540f57b619a08f7b388b45",
    "official_helpers": "f7298ec0744ec315852627391f32c7d1270ad49c8a6a0645214026bb45e528e4",
    "official_license": "84f2795e9d1a3be7f3672fad8ae5bc391393ff9904d56e80667c2f90a320e860",
    "paper": "c6d98d2ac474608218014af73ef9a1e4f4eae5db496e6d038c5d8e451959f1a7",
    "current_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
MAX_INFERENCE_RUNTIME_RATIO = 1.05
MAX_TRAIN_RUNTIME_RATIO = 1.10
MAX_VRAM_GIB = 7.5
MAX_VRAM_RATIO = 1.10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only EViT-style inattentive-token fusion A0 preflight. "
            "Official validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--scratch-launcher",
        type=Path,
        default=Path("scripts/run_trkh_5class_surface_detail_v9.ps1"),
    )
    parser.add_argument(
        "--keeper-checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--fold-data",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/data.yaml"),
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--declaration-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_INATTENTIVE_TOKEN_FUSION_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--bra-closure",
        type=Path,
        default=Path("docs/TRKH_5CLASS_BILEVEL_ROUTING_ATTENTION_CLOSURE_20260716.md"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\evit-iclr2022"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--audit-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=5)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    return parser.parse_args(argv)


def _fusion_config(
    source: Mapping[str, object], *, enabled: bool
) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "pretrained": False,
            "early_token_mask_keep_rate": 1.0,
            "inattentive_token_fusion": bool(enabled),
        }
    )
    return config


def _rng_sha256(state: Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _rng_snapshot() -> Dict[str, object]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _rng_summary(state: Mapping[str, object]) -> Dict[str, object]:
    return {
        "cpu_sha256": _rng_sha256(state["cpu"]),
        "cuda_sha256": [_rng_sha256(value) for value in state["cuda"]],
    }


def _rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_cuda = left["cuda"]
    right_cuda = right["cuda"]
    return bool(
        torch.equal(left["cpu"], right["cpu"])
        and len(left_cuda) == len(right_cuda)
        and all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))
    )


def _repository_state(root: Path) -> Dict[str, object]:
    def value(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "commit": value("rev-parse", "HEAD"),
        "tree": value("rev-parse", "HEAD^{tree}"),
        "tracked_clean": not bool(
            value("status", "--short", "--untracked-files=no").strip()
        ),
    }


def _state_summary(control: nn.Module, candidate: nn.Module) -> Dict[str, object]:
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    control_keys = list(control_state)
    candidate_keys = list(candidate_state)
    common = sorted(set(control_state).intersection(candidate_state))
    differing = [
        key
        for key in common
        if not torch.equal(control_state[key].cpu(), candidate_state[key].cpu())
    ]
    return {
        "inventory_equal": control_keys == candidate_keys,
        "control_only": sorted(set(control_state).difference(candidate_state)),
        "candidate_only": sorted(set(candidate_state).difference(control_state)),
        "common_keys": len(common),
        "differing_common_keys": differing,
        "all_common_bit_exact": not differing,
    }


def _tensor_tree_comparison(left: object, right: object) -> Dict[str, object]:
    maximum_error = 0.0
    mismatches: list[str] = []

    def visit(a: object, b: object, path: str) -> None:
        nonlocal maximum_error
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not torch.is_tensor(a) or not torch.is_tensor(b):
                mismatches.append(path)
                return
            if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
                mismatches.append(path)
                return
            if a.is_floating_point() or a.is_complex():
                error = float((a.detach().float() - b.detach().float()).abs().amax().item())
                maximum_error = max(maximum_error, error)
                if error != 0.0:
                    mismatches.append(path)
            elif not torch.equal(a.detach().cpu(), b.detach().cpu()):
                mismatches.append(path)
            return
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                mismatches.append(path)
                return
            if set(a) != set(b):
                mismatches.append(path)
                return
            for key in sorted(a):
                visit(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
                mismatches.append(path)
                return
            if len(a) != len(b):
                mismatches.append(path)
                return
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{path}[{index}]")
            return
        if a != b:
            mismatches.append(path)

    visit(left, right, "root")
    return {
        "maximum_absolute_error": maximum_error,
        "mismatch_count": len(mismatches),
        "mismatch_paths": mismatches[:100],
        "exact": not mismatches,
    }


def _set_jaccard(left: Tensor, right: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.size(0) != right.size(0):
        raise ValueError("Set Jaccard expects aligned [B,K] tensors.")
    equality = left[:, :, None].eq(right[:, None, :])
    intersection = equality.any(dim=2).sum(dim=1).float()
    union = float(left.size(1) + right.size(1)) - intersection
    return intersection / union.clamp_min(1.0)


def _partition_valid(
    previous: Tensor, kept: Tensor, dropped: Tensor
) -> Tensor:
    if previous.ndim != 2 or kept.ndim != 2 or dropped.ndim != 2:
        raise ValueError("Partition tensors must have shape [B,N].")
    rows = []
    for row in range(previous.size(0)):
        previous_set = set(int(value) for value in previous[row].tolist())
        kept_set = set(int(value) for value in kept[row].tolist())
        dropped_set = set(int(value) for value in dropped[row].tolist())
        rows.append(
            kept_set.isdisjoint(dropped_set)
            and kept_set.union(dropped_set) == previous_set
            and len(kept_set) == int(kept.size(1))
            and len(dropped_set) == int(dropped.size(1))
        )
    return torch.tensor(rows, dtype=torch.bool)


def _bbox_geometry(bbox: Tensor) -> Dict[str, float]:
    values = bbox.detach().float().cpu().flatten()
    if values.numel() < 4 or not bool(torch.isfinite(values[:4]).all()):
        raise ValueError("Fusion audit requires a finite normalized bbox.")
    center_x, center_y, width, height = [float(value) for value in values[:4]]
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Fusion audit requires a positive bbox width and height.")
    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "area": width * height,
        "edge_gap": min(
            center_x - width / 2.0,
            center_y - height / 2.0,
            1.0 - center_x - width / 2.0,
            1.0 - center_y - height / 2.0,
        ),
    }


def _bbox_patch_overlap(bbox: Tensor, *, grid_size: int = GRID_SIZE) -> Tensor:
    geometry = _bbox_geometry(bbox)
    x1 = max(0.0, geometry["center_x"] - geometry["width"] / 2.0)
    y1 = max(0.0, geometry["center_y"] - geometry["height"] / 2.0)
    x2 = min(1.0, geometry["center_x"] + geometry["width"] / 2.0)
    y2 = min(1.0, geometry["center_y"] + geometry["height"] / 2.0)
    lower = torch.arange(grid_size, dtype=torch.float64) / float(grid_size)
    upper = lower + 1.0 / float(grid_size)
    overlap_x = (torch.minimum(upper, torch.tensor(x2)) - torch.maximum(lower, torch.tensor(x1))).clamp_min(0.0)
    overlap_y = (torch.minimum(upper, torch.tensor(y2)) - torch.maximum(lower, torch.tensor(y1))).clamp_min(0.0)
    overlap = overlap_y[:, None] * overlap_x[None, :]
    return (overlap * float(grid_size * grid_size)).clamp(0.0, 1.0).float().flatten()


def _resolve_tiny_edge_cohort(
    geometries: Sequence[Mapping[str, object]],
    holdout_rows: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    if len(geometries) != EXPECTED_HOLDOUT_ROWS or len(holdout_rows) != len(geometries):
        raise ValueError("Tiny/edge cohort requires every locked holdout row.")
    areas = np.asarray([float(row["bbox_area"]) for row in geometries], dtype=np.float64)
    gaps = np.asarray([float(row["bbox_edge_gap"]) for row in geometries], dtype=np.float64)
    if not np.isfinite(areas).all() or not np.isfinite(gaps).all() or bool((areas <= 0).any()):
        raise ValueError("Tiny/edge cohort contains invalid bbox geometry.")
    threshold = float(np.quantile(areas, 0.25, method="linear"))
    tiny = areas <= threshold
    edge = gaps <= (1.0 / 16.0)
    union = tiny | edge
    local_indices = np.flatnonzero(union).astype(np.int64).tolist()
    sample_indices = [int(holdout_rows[index].sample_index) for index in local_indices]
    return {
        "definition": {
            "tiny": "bbox_area <= linear_Q25(clean_transformed_holdout)",
            "edge": "bbox_edge_gap <= 1/16",
            "union": "tiny OR edge",
        },
        "bbox_area_q25": threshold,
        "edge_gap_threshold": 1.0 / 16.0,
        "tiny_count": int(tiny.sum()),
        "edge_count": int(edge.sum()),
        "intersection_count": int((tiny & edge).sum()),
        "cohort_count": int(union.sum()),
        "local_holdout_indices": local_indices,
        "sample_indices": sample_indices,
        "ordered_sample_index_sha256": _ordered_index_sha256(sample_indices),
    }


def _serialize_numbers(values: Tensor, *, precision: int = 10) -> str:
    flattened = values.detach().cpu().flatten().tolist()
    if values.is_floating_point():
        return ";".join(f"{float(value):.{precision}g}" for value in flattened)
    return ";".join(str(int(value)) for value in flattened)


def _parse_numbers(value: object, *, integer: bool = False) -> list:
    text = str(value).strip()
    if not text:
        return []
    if integer:
        return [int(item) for item in text.split(";")]
    return [float(item) for item in text.split(";")]


def _fusion_lineage(
    pruning: Sequence[Mapping[str, Tensor]],
    row: int,
    overlap: Tensor,
) -> Dict[str, object]:
    if len(pruning) != 2 or overlap.numel() != ORIGINAL_PATCH_COUNT:
        raise ValueError("Locked fusion lineage expects two stages and a 16x16 grid.")
    lineage = torch.zeros(ORIGINAL_PATCH_COUNT, dtype=torch.float64)
    support = torch.zeros(ORIGINAL_PATCH_COUNT, dtype=torch.bool)
    previous = torch.arange(ORIGINAL_PATCH_COUNT, dtype=torch.long)
    stage_rows: list[Dict[str, object]] = []
    all_partitions_valid = True
    for stage_index, stage in enumerate(pruning):
        kept = stage["kept_indices"][row].detach().long().cpu()
        dropped = stage["dropped_indices"][row].detach().long().cpu()
        weights = stage["fusion_weights"][row].detach().double().cpu()
        reported_mass = float(stage["context_attention_mass"][row].detach().float().item())
        previous_mass = float(
            stage["previous_context_attention_mass"][row].detach().float().item()
        )
        partition = bool(
            _partition_valid(
                previous.unsqueeze(0), kept.unsqueeze(0), dropped.unsqueeze(0)
            )[0]
        )
        all_partitions_valid = all_partitions_valid and partition
        if stage_index > 0:
            lineage = lineage * previous_mass
        lineage[dropped] = lineage[dropped] + weights
        support[dropped] = True
        independent_reported_mass = float(weights.sum().item() + previous_mass)
        stage_rows.append(
            {
                "kept": kept,
                "dropped": dropped,
                "weights": weights,
                "reported_mass": reported_mass,
                "previous_mass": previous_mass,
                "mass_replay_error": abs(reported_mass - independent_reported_mass),
                "finite_nonnegative": bool(
                    torch.isfinite(weights).all()
                    and bool((weights >= 0).all())
                    and math.isfinite(reported_mass)
                    and math.isfinite(previous_mass)
                    and reported_mass >= 0.0
                    and previous_mass >= 0.0
                ),
                "partition_valid": partition,
            }
        )
        previous = kept
    lineage_total = float(lineage.sum().item())
    overlap64 = overlap.detach().double().cpu()
    object_mass = float((lineage * overlap64).sum().item())
    weighted_object_fraction = object_mass / max(lineage_total, 1e-15)
    outside_fraction = float((lineage * (1.0 - overlap64)).sum().item()) / max(
        lineage_total, 1e-15
    )
    raw_object_fraction = float(overlap64[support].mean().item()) if bool(support.any()) else math.nan
    return {
        "stages": stage_rows,
        "lineage": lineage,
        "support": support,
        "lineage_total": lineage_total,
        "object_mass": object_mass,
        "weighted_object_fraction": weighted_object_fraction,
        "outside_fraction": outside_fraction,
        "raw_object_fraction": raw_object_fraction,
        "partitions_valid": all_partitions_valid,
    }


def _official_equation_replay() -> Dict[str, object]:
    torch.manual_seed(1902)
    tokens_native = torch.randn(3, 11, 7, requires_grad=True)
    attention_native = torch.rand(3, 11, requires_grad=True)
    selected = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10],
            [1, 2, 3, 7, 8, 9],
            [0, 1, 5, 6, 9, 10],
        ],
        dtype=torch.long,
    )
    tokens_replay = tokens_native.detach().clone().requires_grad_(True)
    attention_replay = attention_native.detach().clone().requires_grad_(True)
    native = VisionTransformerWithRegisters._fuse_inattentive_context(
        patch_tokens=tokens_native,
        cls_attention=attention_native,
        selected_local=selected,
    )
    selected_mask = torch.zeros_like(attention_replay).scatter(
        1, selected, torch.ones_like(selected, dtype=attention_replay.dtype)
    )
    replay_weight_map = attention_replay * (1.0 - selected_mask)
    replay_context = (tokens_replay * replay_weight_map.unsqueeze(-1)).sum(
        dim=1, keepdim=True
    )
    native_weight_map = torch.zeros_like(attention_native).scatter(
        1, native[1], native[2]
    )
    probe = torch.randn_like(replay_context)
    native_objective = (native[0] * probe).sum() + native[3].sum() * 0.13
    replay_objective = (replay_context * probe).sum() + replay_weight_map.sum() * 0.13
    native_objective.backward()
    replay_objective.backward()

    torch.manual_seed(2202)
    later_tokens_native = torch.randn(2, 9, 5, requires_grad=True)
    later_attention_native = torch.rand(2, 9, requires_grad=True)
    previous_native = torch.randn(2, 1, 5, requires_grad=True)
    previous_attention_native = torch.rand(2, 1, requires_grad=True)
    later_selected = torch.tensor(
        [[0, 2, 3, 5, 7, 8], [1, 2, 4, 5, 6, 8]], dtype=torch.long
    )
    later_tokens_replay = later_tokens_native.detach().clone().requires_grad_(True)
    later_attention_replay = later_attention_native.detach().clone().requires_grad_(True)
    previous_replay = previous_native.detach().clone().requires_grad_(True)
    previous_attention_replay = (
        previous_attention_native.detach().clone().requires_grad_(True)
    )
    native_later = VisionTransformerWithRegisters._fuse_inattentive_context(
        patch_tokens=later_tokens_native,
        cls_attention=later_attention_native,
        selected_local=later_selected,
        previous_context=previous_native,
        previous_context_attention=previous_attention_native,
    )
    later_selected_mask = torch.zeros_like(later_attention_replay).scatter(
        1,
        later_selected,
        torch.ones_like(later_selected, dtype=later_attention_replay.dtype),
    )
    later_weight_map = later_attention_replay * (1.0 - later_selected_mask)
    replay_later = (
        (later_tokens_replay * later_weight_map.unsqueeze(-1)).sum(dim=1, keepdim=True)
        + previous_replay * previous_attention_replay.unsqueeze(-1)
    )
    native_later_weight_map = torch.zeros_like(later_attention_native).scatter(
        1, native_later[1], native_later[2]
    )
    later_probe = torch.randn_like(replay_later)
    native_later_objective = (native_later[0] * later_probe).sum() + native_later[3].sum() * 0.07
    replay_later_objective = (replay_later * later_probe).sum() + (
        later_weight_map.sum() + previous_attention_replay.sum()
    ) * 0.07
    native_later_objective.backward()
    replay_later_objective.backward()

    errors = {
        "first_context": float((native[0] - replay_context).abs().amax().item()),
        "first_weight_map": float(
            (native_weight_map - replay_weight_map).abs().amax().item()
        ),
        "first_mass": float(
            (native[3] - replay_weight_map.sum(dim=1, keepdim=True)).abs().amax().item()
        ),
        "first_token_gradient": float(
            (tokens_native.grad - tokens_replay.grad).abs().amax().item()
        ),
        "first_attention_gradient": float(
            (attention_native.grad - attention_replay.grad).abs().amax().item()
        ),
        "later_context": float((native_later[0] - replay_later).abs().amax().item()),
        "later_weight_map": float(
            (native_later_weight_map - later_weight_map).abs().amax().item()
        ),
        "later_mass": float(
            (
                native_later[3]
                - later_weight_map.sum(dim=1, keepdim=True)
                - previous_attention_replay
            )
            .abs()
            .amax()
            .item()
        ),
        "later_token_gradient": float(
            (later_tokens_native.grad - later_tokens_replay.grad).abs().amax().item()
        ),
        "later_attention_gradient": float(
            (later_attention_native.grad - later_attention_replay.grad)
            .abs()
            .amax()
            .item()
        ),
        "previous_context_gradient": float(
            (previous_native.grad - previous_replay.grad).abs().amax().item()
        ),
        "previous_attention_gradient": float(
            (
                previous_attention_native.grad
                - previous_attention_replay.grad
            )
            .abs()
            .amax()
            .item()
        ),
    }
    maximum_error = max(errors.values())
    return {
        "source_equation": "sum(dropped_cls_attention * dropped_post_block_tokens)",
        "errors": errors,
        "maximum_absolute_error": maximum_error,
        "first_dropped_partition_valid": bool(
            _partition_valid(
                torch.arange(11).view(1, -1).expand(3, -1), selected, native[1]
            ).all()
        ),
        "later_dropped_partition_valid": bool(
            _partition_valid(
                torch.arange(9).view(1, -1).expand(2, -1),
                later_selected,
                native_later[1],
            ).all()
        ),
        "passed": bool(maximum_error <= 1e-6),
    }


def _gradient_summary(model: nn.Module) -> Dict[str, object]:
    families = {
        "stem": ("stem.",),
        "first_prune_qkv": ("blocks.1.attn.qkv",),
        "second_prune_qkv": ("blocks.4.attn.qkv",),
        "classifier": ("head.",),
    }
    parameters = dict(model.named_parameters())
    result: Dict[str, object] = {}
    for family, markers in families.items():
        selected = {
            name: parameter
            for name, parameter in parameters.items()
            if any(marker in name for marker in markers)
        }
        gradients = [parameter.grad for parameter in selected.values()]
        present = [gradient for gradient in gradients if gradient is not None]
        finite = bool(present) and all(
            bool(torch.isfinite(gradient).all()) for gradient in present
        )
        nonzero = sum(int(torch.count_nonzero(gradient).item()) for gradient in present)
        result[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(present),
            "finite": finite,
            "nonzero_elements": nonzero,
            "passed": bool(selected and present and finite and nonzero > 0),
        }
    return result


def _full_backward_check(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    bf16: bool,
) -> Dict[str, object]:
    model.train()
    model.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with context:
        logits, features = _forward(
            model, images, metadata, return_attention=False, return_trace=True
        )
        pruning = features["trace"]["pruning"]
        stage_contexts = [stage["context_token"] for stage in pruning]
        for token in stage_contexts:
            token.retain_grad()
        loss = F.cross_entropy(logits.float(), labels)
    loss.backward()
    stage_gradients = []
    for stage, token in zip(pruning, stage_contexts):
        gradient = token.grad
        weights = stage["fusion_weights"]
        context_nonzero = bool(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0
        )
        positive_weight = bool(torch.isfinite(weights).all() and (weights > 0).any())
        stage_gradients.append(
            {
                "context_gradient_finite_nonzero": context_nonzero,
                "positive_fusion_weight": positive_weight,
                "inferred_dropped_token_gradient_nonzero": bool(
                    context_nonzero and positive_weight
                ),
                "context_gradient_norm": (
                    float(gradient.detach().float().norm().item())
                    if gradient is not None
                    else None
                ),
            }
        )
    return {
        "dtype": "cuda_bf16" if bf16 else "cuda_fp32",
        "batch_size": int(images.size(0)),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss)),
        "stage_context_gradients": stage_gradients,
        "parameter_gradients": _gradient_summary(model),
        "passed": bool(
            torch.isfinite(logits).all()
            and torch.isfinite(loss)
            and all(
                bool(row["inferred_dropped_token_gradient_nonzero"])
                for row in stage_gradients
            )
            and all(
                bool(row["passed"])
                for row in _gradient_summary(model).values()
            )
        ),
    }


def _capture_standard_pruning(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Tuple[Tensor, Mapping[str, object], Sequence[Mapping[str, Tensor]]]:
    original = model._prune_patch_tokens
    captured: list[Dict[str, Tensor]] = []

    def capture(**kwargs):
        output = original(**kwargs)
        captured.append(
            {
                key: value.detach().clone()
                for key, value in output[2].items()
                if torch.is_tensor(value)
            }
        )
        return output

    object.__setattr__(model, "_prune_patch_tokens", capture)
    try:
        logits, features = _forward(model, images, metadata)
    finally:
        object.__setattr__(model, "_prune_patch_tokens", original)
    return logits, features, captured


def _trace_parity(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    bf16: bool,
) -> Dict[str, object]:
    model.eval()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with torch.inference_mode(), context:
        standard_logits, standard_features, captured = _capture_standard_pruning(
            model, images, metadata
        )
        trace_logits, trace_features = _forward(
            model, images, metadata, return_attention=False, return_trace=True
        )
    trace_pruning = trace_features["trace"]["pruning"]
    stage_rows = []
    for standard_stage, trace_stage in zip(captured, trace_pruning):
        stage_rows.append(
            {
                "kept_indices_exact": bool(
                    torch.equal(
                        standard_stage["kept_indices"], trace_stage["kept_indices"]
                    )
                ),
                "dropped_indices_exact": bool(
                    torch.equal(
                        standard_stage["dropped_indices"],
                        trace_stage["dropped_indices"],
                    )
                ),
                "context_maximum_absolute_error": float(
                    (
                        standard_stage["context_token"].float()
                        - trace_stage["context_token"].float()
                    )
                    .abs()
                    .amax()
                    .item()
                ),
            }
        )
    logit_error = float(
        (standard_logits.float() - trace_logits.float()).abs().amax().item()
    )
    return {
        "dtype": "cuda_bf16" if bf16 else "cuda_fp32",
        "logit_maximum_absolute_error": logit_error,
        "argmax_match": bool(
            torch.equal(standard_logits.argmax(dim=1), trace_logits.argmax(dim=1))
        ),
        "final_patch_indices_exact": bool(
            torch.equal(
                standard_features["patch_indices"], trace_features["patch_indices"]
            )
        ),
        "stage_count": len(stage_rows),
        "stages": stage_rows,
        "passed": bool(
            logit_error <= 1e-6
            and torch.equal(
                standard_logits.argmax(dim=1), trace_logits.argmax(dim=1)
            )
            and torch.equal(
                standard_features["patch_indices"], trace_features["patch_indices"]
            )
            and len(stage_rows) == 2
            and all(
                bool(row["kept_indices_exact"])
                and bool(row["dropped_indices_exact"])
                and float(row["context_maximum_absolute_error"]) <= 1e-6
                for row in stage_rows
            )
        ),
    }


def _pruning_comparison(
    control_pruning: Sequence[Mapping[str, Tensor]],
    candidate_pruning: Sequence[Mapping[str, Tensor]],
) -> Dict[str, object]:
    if len(control_pruning) != 2 or len(candidate_pruning) != 2:
        raise ValueError("Locked pruning comparison expects two stages.")
    rows = []
    previous = torch.arange(ORIGINAL_PATCH_COUNT).view(1, -1).expand(
        candidate_pruning[0]["kept_indices"].size(0), -1
    )
    for control_stage, candidate_stage in zip(control_pruning, candidate_pruning):
        control_kept = control_stage["kept_indices"].detach().cpu()
        candidate_kept = candidate_stage["kept_indices"].detach().cpu()
        candidate_dropped = candidate_stage["dropped_indices"].detach().cpu()
        jaccard = _set_jaccard(control_kept, candidate_kept)
        partition = _partition_valid(previous, candidate_kept, candidate_dropped)
        rows.append(
            {
                "rows": int(jaccard.numel()),
                "mean_jaccard": float(jaccard.mean().item()),
                "minimum_jaccard": float(jaccard.min().item()),
                "exact_fraction": float(control_kept.eq(candidate_kept).all(dim=1).float().mean().item()),
                "changed_rows": int((jaccard < 1.0).sum().item()),
                "partition_valid_fraction": float(partition.float().mean().item()),
            }
        )
        previous = candidate_kept
    return {
        "stages": rows,
        "passed": bool(
            rows[0]["exact_fraction"] == 1.0
            and rows[0]["partition_valid_fraction"] == 1.0
            and rows[1]["mean_jaccard"] >= 0.98
            and rows[1]["partition_valid_fraction"] == 1.0
        ),
    }


def _construct_keeper_models(
    checkpoint: Mapping[str, object], *, seed: int
) -> Tuple[nn.Module, nn.Module, Dict[str, object]]:
    source_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state")
    if not isinstance(source_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Keeper checkpoint lacks model_config/class_names.")
    if not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint lacks model_state.")
    set_seed(seed, deterministic=True)
    control = create_model(
        num_classes=len(class_names),
        model_config=_fusion_config(source_config, enabled=False),
    )
    control_rng = _rng_snapshot()
    set_seed(seed, deterministic=True)
    candidate = create_model(
        num_classes=len(class_names),
        model_config=_fusion_config(source_config, enabled=True),
    )
    candidate_rng = _rng_snapshot()
    constructor_state = _state_summary(control, candidate)
    control_missing, control_unexpected = load_model_state(
        control, dict(state), strict=True
    )
    candidate_missing, candidate_unexpected = load_model_state(
        candidate, dict(state), strict=True
    )
    loaded_state = _state_summary(control, candidate)
    return control.eval(), candidate.eval(), {
        "control_constructor_rng": _rng_summary(control_rng),
        "candidate_constructor_rng": _rng_summary(candidate_rng),
        "constructor_rng_equal": _rng_equal(control_rng, candidate_rng),
        "constructor_state": constructor_state,
        "loaded_state": loaded_state,
        "control_missing_keys": sorted(control_missing),
        "candidate_missing_keys": sorted(candidate_missing),
        "control_unexpected_keys": sorted(control_unexpected),
        "candidate_unexpected_keys": sorted(candidate_unexpected),
        "parameter_count": sum(int(value.numel()) for value in control.parameters()),
    }


def _precompute_cohort(
    *,
    base_dataset,
    transform,
    holdout_rows: Sequence[CleanTrainRow],
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, object], Dict[int, Dict[str, float]]]:
    dataset = _SelectedConditionDataset(
        base_dataset,
        list(range(len(base_dataset))),
        corruption=_condition_corruption("clean", 1.0, 1.0),
        transform=transform,
    )
    loader, loader_summary = _make_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        context="inattentive_fusion_cohort_metadata_only",
        seed=seed + 17,
    )
    geometries: Dict[int, Dict[str, float]] = {}
    for _images, _targets, metadata in loader:
        sample_indices = metadata.get("sample_index")
        bboxes = metadata.get("bbox")
        if not torch.is_tensor(sample_indices) or not torch.is_tensor(bboxes):
            raise ValueError("Cohort metadata lacks sample_index/bbox tensors.")
        for position, local_value in enumerate(sample_indices.tolist()):
            local_index = int(local_value)
            geometry = _bbox_geometry(bboxes[position])
            geometries[local_index] = {
                "bbox_area": float(geometry["area"]),
                "bbox_edge_gap": float(geometry["edge_gap"]),
            }
    if sorted(geometries) != list(range(EXPECTED_HOLDOUT_ROWS)):
        raise ValueError("Metadata-only cohort scan did not cover ordered holdout rows.")
    ordered = [geometries[index] for index in range(EXPECTED_HOLDOUT_ROWS)]
    cohort = _resolve_tiny_edge_cohort(ordered, holdout_rows)
    cohort["loader"] = loader_summary
    cohort["computed_before_model_scoring"] = True
    cohort_path = output_dir / "tiny_edge_cohort.json"
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    cohort["path"] = str(cohort_path.resolve())
    cohort["sha256"] = _sha256(cohort_path)
    return cohort, geometries


def _condition_behavior_summary(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not rows:
        raise ValueError("Cannot summarize an empty fusion condition.")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    control_predictions = np.asarray(
        [int(row["control_prediction"]) for row in rows], dtype=np.int64
    )
    candidate_predictions = np.asarray(
        [int(row["candidate_prediction"]) for row in rows], dtype=np.int64
    )
    control_metrics = _classification_metrics(
        targets, control_predictions, num_classes=5
    )
    candidate_metrics = _classification_metrics(
        targets, candidate_predictions, num_classes=5
    )

    def mean(name: str, *, selected: Optional[np.ndarray] = None) -> float:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if selected is not None:
            values = values[selected]
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else math.nan

    control_tp = int(((targets == FOCUS_CLASS) & (control_predictions == FOCUS_CLASS)).sum())
    candidate_tp = int(
        ((targets == FOCUS_CLASS) & (candidate_predictions == FOCUS_CLASS)).sum()
    )
    restricted = np.isin(targets, np.asarray(RESTRICTED_NEGATIVE_CLASSES))
    control_restricted_fp = int(
        (restricted & (control_predictions == FOCUS_CLASS)).sum()
    )
    candidate_restricted_fp = int(
        (restricted & (candidate_predictions == FOCUS_CLASS)).sum()
    )
    cohort_mask = np.asarray(
        [bool(row["tiny_edge_cohort"]) for row in rows], dtype=bool
    )
    cohort_nonzero = np.asarray(
        [bool(row["dropped_object_mass_nonzero"]) for row in rows], dtype=bool
    )
    cosines = np.asarray(
        [
            float(row["clean_context_cosine"])
            for row in rows
            if row["clean_context_cosine"] not in (None, "")
        ],
        dtype=np.float64,
    )
    control_class1_f1 = float(control_metrics["per_class_f1"][FOCUS_CLASS])
    candidate_class1_f1 = float(candidate_metrics["per_class_f1"][FOCUS_CLASS])
    control_class1_precision = float(
        control_metrics["per_class_precision"][FOCUS_CLASS]
    )
    candidate_class1_precision = float(
        candidate_metrics["per_class_precision"][FOCUS_CLASS]
    )
    return {
        "rows": len(rows),
        "control_metrics": control_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": {
            "macro_f1": float(candidate_metrics["macro_f1"])
            - float(control_metrics["macro_f1"]),
            "class1_f1": candidate_class1_f1 - control_class1_f1,
            "class1_precision": candidate_class1_precision
            - control_class1_precision,
        },
        "control_class1_tp": control_tp,
        "candidate_class1_tp": candidate_tp,
        "class1_tp_retention": candidate_tp / max(1, control_tp),
        "control_restricted_fp": control_restricted_fp,
        "candidate_restricted_fp": candidate_restricted_fp,
        "restricted_fp_delta": candidate_restricted_fp - control_restricted_fp,
        "candidate_corrections": sum(
            int(bool(row["candidate_correction"])) for row in rows
        ),
        "candidate_harms": sum(int(bool(row["candidate_harm"])) for row in rows),
        "probability_mae_mean": mean("probability_mae"),
        "first_prune_exact_fraction": mean("first_prune_exact"),
        "second_prune_jaccard_mean": mean("second_prune_jaccard"),
        "second_prune_jaccard_minimum": float(
            min(float(row["second_prune_jaccard"]) for row in rows)
        ),
        "second_prune_changed_rows": sum(
            int(bool(row["second_prune_changed"])) for row in rows
        ),
        "partition_valid_fraction": mean("partitions_valid"),
        "stage1_positive_finite_mass_fraction": mean(
            "stage1_positive_finite_mass"
        ),
        "stage2_positive_finite_mass_fraction": mean(
            "stage2_positive_finite_mass"
        ),
        "maximum_mass_replay_error": float(
            max(
                max(
                    float(row["stage1_mass_replay_error"]),
                    float(row["stage2_mass_replay_error"]),
                )
                for row in rows
            )
        ),
        "weighted_object_fraction_mean": mean("weighted_object_fraction"),
        "raw_dropped_object_fraction_mean": mean("raw_dropped_object_fraction"),
        "weighted_object_gain_mean": mean("weighted_object_gain"),
        "outside_context_fraction_mean": mean("outside_context_fraction"),
        "tiny_edge_cohort_rows": int(cohort_mask.sum()),
        "tiny_edge_nonzero_object_mass_fraction": (
            float(cohort_nonzero[cohort_mask].mean())
            if bool(cohort_mask.any())
            else math.nan
        ),
        "clean_context_cosine_mean": (
            float(cosines.mean()) if cosines.size else None
        ),
        "clean_context_cosine_minimum": (
            float(cosines.min()) if cosines.size else None
        ),
        "standard_trace_control_maximum_error": float(
            max(float(row["control_standard_trace_error"]) for row in rows)
        ),
        "standard_trace_candidate_maximum_error": float(
            max(float(row["candidate_standard_trace_error"]) for row in rows)
        ),
        "finite_rows": sum(int(bool(row["finite_row"])) for row in rows),
    }


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    template: Optional[Mapping[str, object]] = None,
) -> None:
    if not rows and template is None:
        raise ValueError("CSV output requires rows or an explicit template.")
    fieldnames = list((rows[0] if rows else template).keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _behavior_gate_checks(
    condition_summaries: Mapping[str, Mapping[str, object]],
    *,
    row_count: int,
) -> Dict[str, bool]:
    clean = condition_summaries["clean"]
    shifted = [name for name, _, _ in CONDITIONS if name != "clean"]
    checks: Dict[str, bool] = {
        "all_holdout_rows_all_conditions": row_count
        == EXPECTED_HOLDOUT_ROWS * len(CONDITIONS),
        "clean_probability_change_active_bounded": 1e-5
        <= float(clean["probability_mae_mean"])
        <= 0.05,
        "clean_macro_f1_safe": float(clean["metric_deltas"]["macro_f1"])
        >= -0.0025,
        "clean_class1_f1_safe": float(clean["metric_deltas"]["class1_f1"])
        >= -0.0050,
        "clean_class1_precision_safe": float(
            clean["metric_deltas"]["class1_precision"]
        )
        >= -0.0050,
        "clean_class1_tp_retention": float(clean["class1_tp_retention"]) >= 0.98,
        "clean_tiny_edge_object_mass": float(
            clean["tiny_edge_nonzero_object_mass_fraction"]
        )
        >= 0.20,
        "clean_weighted_object_gain": float(clean["weighted_object_gain_mean"])
        >= 0.010,
        "clean_outside_context_mass": float(
            clean["outside_context_fraction_mean"]
        )
        >= 0.20,
    }
    for name, summary in condition_summaries.items():
        checks[f"{name}_first_prune_exact"] = (
            float(summary["first_prune_exact_fraction"]) == 1.0
        )
        checks[f"{name}_second_prune_jaccard"] = (
            float(summary["second_prune_jaccard_mean"]) >= 0.98
        )
        checks[f"{name}_partitions_valid"] = (
            float(summary["partition_valid_fraction"]) == 1.0
        )
        checks[f"{name}_stage1_mass_coverage"] = (
            float(summary["stage1_positive_finite_mass_fraction"]) >= 0.95
        )
        checks[f"{name}_stage2_mass_coverage"] = (
            float(summary["stage2_positive_finite_mass_fraction"]) >= 0.95
        )
        checks[f"{name}_mass_replay"] = (
            float(summary["maximum_mass_replay_error"]) <= 1e-6
        )
        checks[f"{name}_restricted_fp_safe"] = (
            int(summary["restricted_fp_delta"]) <= 2
        )
        checks[f"{name}_all_rows_finite"] = (
            int(summary["finite_rows"]) == EXPECTED_HOLDOUT_ROWS
        )
        checks[f"{name}_standard_trace_parity"] = max(
            float(summary["standard_trace_control_maximum_error"]),
            float(summary["standard_trace_candidate_maximum_error"]),
        ) <= 1e-6
    for name in shifted:
        summary = condition_summaries[name]
        checks[f"{name}_class1_f1_safe"] = (
            float(summary["metric_deltas"]["class1_f1"]) >= -0.010
        )
        checks[f"{name}_class1_precision_safe"] = (
            float(summary["metric_deltas"]["class1_precision"]) >= -0.010
        )
        checks[f"{name}_context_stability"] = (
            summary["clean_context_cosine_mean"] is not None
            and float(summary["clean_context_cosine_mean"]) >= 0.65
        )
    return checks


def _behavior_audit(
    *,
    control: nn.Module,
    candidate: nn.Module,
    base_dataset,
    transform,
    dataset_summary: Mapping[str, object],
    holdout_rows: Sequence[CleanTrainRow],
    cohort: Mapping[str, object],
    geometries: Mapping[int, Mapping[str, float]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, object], Sequence[Mapping[str, object]]]:
    control = control.to(device).eval()
    candidate = candidate.to(device).eval()
    cohort_indices = set(int(value) for value in cohort["local_holdout_indices"])
    tiny_threshold = float(cohort["bbox_area_q25"])
    edge_threshold = float(cohort["edge_gap_threshold"])
    clean_context = torch.empty(EXPECTED_HOLDOUT_ROWS, 256, dtype=torch.float32)
    clean_context_seen = torch.zeros(EXPECTED_HOLDOUT_ROWS, dtype=torch.bool)
    all_rows: list[Dict[str, object]] = []
    condition_summaries: Dict[str, object] = {}
    loader_summaries: Dict[str, object] = {}
    for condition_index, (condition, brightness, contrast) in enumerate(CONDITIONS):
        condition_dataset = _SelectedConditionDataset(
            base_dataset,
            list(range(len(base_dataset))),
            corruption=_condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = _make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"inattentive_fusion_{condition}_holdout",
            seed=seed + 100 + condition_index,
        )
        loader_summaries[condition] = loader_summary
        condition_rows: list[Dict[str, object]] = []
        processed = 0
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                metadata = _metadata_to_device(metadata_cpu, device)
                control_logits, _ = _forward(control, images, metadata)
                candidate_logits, _ = _forward(candidate, images, metadata)
                control_trace_logits, control_features = _forward(
                    control,
                    images,
                    metadata,
                    return_attention=False,
                    return_trace=True,
                )
                candidate_trace_logits, candidate_features = _forward(
                    candidate,
                    images,
                    metadata,
                    return_attention=False,
                    return_trace=True,
                )
                control_probabilities = control_logits.float().softmax(dim=1).cpu()
                candidate_probabilities = candidate_logits.float().softmax(dim=1).cpu()
                control_predictions = control_probabilities.argmax(dim=1)
                candidate_predictions = candidate_probabilities.argmax(dim=1)
                control_pruning = control_features["trace"]["pruning"]
                candidate_pruning = candidate_features["trace"]["pruning"]
                contexts = candidate_features["inattentive_context"].detach().float().cpu()
                sample_indices = metadata_cpu.get("sample_index")
                bboxes = metadata_cpu.get("bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(bboxes):
                    raise ValueError("Behavior metadata lacks sample_index/bbox tensors.")
                if tuple(contexts.shape[1:]) != (1, 256):
                    raise ValueError("Candidate context does not have locked [B,1,256] shape.")
                for position, local_value in enumerate(sample_indices.tolist()):
                    local_index = int(local_value)
                    source = holdout_rows[local_index]
                    target = int(targets_cpu[position].item())
                    if target != int(source.target):
                        raise ValueError("Behavior target order differs from locked declaration.")
                    overlap = _bbox_patch_overlap(bboxes[position])
                    lineage = _fusion_lineage(
                        candidate_pruning, position, overlap
                    )
                    first_control = control_pruning[0]["kept_indices"][position].detach().cpu()
                    first_candidate = candidate_pruning[0]["kept_indices"][position].detach().cpu()
                    second_control = control_pruning[1]["kept_indices"][position].detach().cpu()
                    second_candidate = candidate_pruning[1]["kept_indices"][position].detach().cpu()
                    first_exact = bool(torch.equal(first_control, first_candidate))
                    second_jaccard = float(
                        _set_jaccard(
                            second_control.unsqueeze(0), second_candidate.unsqueeze(0)
                        )[0].item()
                    )
                    context_vector = contexts[position, 0]
                    if condition == "clean":
                        clean_context[local_index] = context_vector
                        clean_context_seen[local_index] = True
                        cosine: object = ""
                    else:
                        if not bool(clean_context_seen[local_index]):
                            raise RuntimeError("Shifted context was scored before clean context.")
                        cosine = float(
                            F.cosine_similarity(
                                clean_context[local_index].unsqueeze(0),
                                context_vector.unsqueeze(0),
                                dim=1,
                            ).item()
                        )
                    control_probability = control_probabilities[position]
                    candidate_probability = candidate_probabilities[position]
                    control_prediction = int(control_predictions[position].item())
                    candidate_prediction = int(candidate_predictions[position].item())
                    geometry = geometries[local_index]
                    stage1 = lineage["stages"][0]
                    stage2 = lineage["stages"][1]
                    finite_row = bool(
                        torch.isfinite(control_probability).all()
                        and torch.isfinite(candidate_probability).all()
                        and torch.isfinite(context_vector).all()
                        and stage1["finite_nonnegative"]
                        and stage2["finite_nonnegative"]
                        and math.isfinite(float(lineage["weighted_object_fraction"]))
                        and math.isfinite(float(lineage["raw_object_fraction"]))
                        and math.isfinite(float(lineage["outside_fraction"]))
                    )
                    row: Dict[str, object] = {
                        "condition": condition,
                        "sample_index": int(source.sample_index),
                        "local_holdout_index": local_index,
                        "source_stem": source.source_stem,
                        "target": target,
                        "declaration_keeper_prediction": int(source.keeper_prediction),
                        "control_prediction": control_prediction,
                        "candidate_prediction": candidate_prediction,
                        "candidate_correction": bool(
                            control_prediction != target and candidate_prediction == target
                        ),
                        "candidate_harm": bool(
                            control_prediction == target and candidate_prediction != target
                        ),
                        "bbox_area": float(geometry["bbox_area"]),
                        "bbox_edge_gap": float(geometry["bbox_edge_gap"]),
                        "tiny": bool(float(geometry["bbox_area"]) <= tiny_threshold),
                        "edge": bool(float(geometry["bbox_edge_gap"]) <= edge_threshold),
                        "tiny_edge_cohort": local_index in cohort_indices,
                        "probability_mae": float(
                            (control_probability - candidate_probability).abs().mean().item()
                        ),
                        "first_prune_exact": first_exact,
                        "second_prune_jaccard": second_jaccard,
                        "second_prune_changed": second_jaccard < 1.0,
                        "partitions_valid": bool(lineage["partitions_valid"]),
                        "stage1_context_mass": float(stage1["reported_mass"]),
                        "stage2_context_mass": float(stage2["reported_mass"]),
                        "stage1_positive_finite_mass": bool(
                            stage1["finite_nonnegative"]
                            and float(stage1["reported_mass"]) > 0.0
                        ),
                        "stage2_positive_finite_mass": bool(
                            stage2["finite_nonnegative"]
                            and float(stage2["reported_mass"]) > 0.0
                        ),
                        "stage1_mass_replay_error": float(stage1["mass_replay_error"]),
                        "stage2_mass_replay_error": float(stage2["mass_replay_error"]),
                        "effective_context_lineage_mass": float(lineage["lineage_total"]),
                        "dropped_object_attention_mass": float(lineage["object_mass"]),
                        "dropped_object_mass_nonzero": bool(
                            float(lineage["object_mass"]) > 1e-12
                        ),
                        "weighted_object_fraction": float(
                            lineage["weighted_object_fraction"]
                        ),
                        "raw_dropped_object_fraction": float(
                            lineage["raw_object_fraction"]
                        ),
                        "weighted_object_gain": float(
                            lineage["weighted_object_fraction"]
                            - lineage["raw_object_fraction"]
                        ),
                        "outside_context_fraction": float(lineage["outside_fraction"]),
                        "clean_context_cosine": cosine,
                        "control_standard_trace_error": float(
                            (
                                control_logits[position].float()
                                - control_trace_logits[position].float()
                            )
                            .abs()
                            .amax()
                            .item()
                        ),
                        "candidate_standard_trace_error": float(
                            (
                                candidate_logits[position].float()
                                - candidate_trace_logits[position].float()
                            )
                            .abs()
                            .amax()
                            .item()
                        ),
                        "finite_row": finite_row,
                        "stage1_kept_indices": _serialize_numbers(stage1["kept"]),
                        "stage1_dropped_indices": _serialize_numbers(stage1["dropped"]),
                        "stage1_fusion_weights": _serialize_numbers(stage1["weights"]),
                        "stage2_kept_indices": _serialize_numbers(stage2["kept"]),
                        "stage2_dropped_indices": _serialize_numbers(stage2["dropped"]),
                        "stage2_fusion_weights": _serialize_numbers(stage2["weights"]),
                        "stage2_previous_context_attention": float(
                            stage2["previous_mass"]
                        ),
                    }
                    for class_index in range(5):
                        row[f"control_prob_{class_index}"] = float(
                            control_probability[class_index].item()
                        )
                        row[f"candidate_prob_{class_index}"] = float(
                            candidate_probability[class_index].item()
                        )
                    condition_rows.append(row)
                    all_rows.append(row)
                processed += int(images.size(0))
                if processed % 512 < int(images.size(0)) or processed == len(condition_dataset):
                    print(
                        json.dumps(
                            {
                                "method": METHOD,
                                "condition": condition,
                                "processed": processed,
                                "rows": len(condition_dataset),
                            }
                        ),
                        flush=True,
                    )
        condition_summaries[condition] = _condition_behavior_summary(condition_rows)
    if not bool(clean_context_seen.all()):
        raise RuntimeError("Clean context cache is incomplete.")
    rows_path = output_dir / "behavior_rows_all_conditions.csv"
    _write_rows_csv(rows_path, all_rows)
    changed_rows = [row for row in all_rows if bool(row["second_prune_changed"])]
    changed_path = output_dir / "second_prune_changed_rows.csv"
    _write_rows_csv(changed_path, changed_rows, template=all_rows[0])
    checks = _behavior_gate_checks(condition_summaries, row_count=len(all_rows))
    return {
        "dataset": dict(dataset_summary),
        "cohort": dict(cohort),
        "conditions": condition_summaries,
        "loader": loader_summaries,
        "rows": len(all_rows),
        "changed_second_prune_rows": len(changed_rows),
        "rows_csv": str(rows_path.resolve()),
        "rows_csv_sha256": _sha256(rows_path),
        "changed_rows_csv": str(changed_path.resolve()),
        "changed_rows_csv_sha256": _sha256(changed_path),
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }, all_rows


def _event_category(source: CleanTrainRow) -> Optional[str]:
    if source.target == FOCUS_CLASS and source.keeper_prediction == FOCUS_CLASS:
        return "class1_keeper_tp"
    if source.target == FOCUS_CLASS and source.keeper_prediction != FOCUS_CLASS:
        return "class1_keeper_fn"
    if (
        source.target in RESTRICTED_NEGATIVE_CLASSES
        and source.keeper_prediction == FOCUS_CLASS
    ):
        return "restricted_keeper_fp"
    return None


def _select_visual_rows(
    clean_rows: Sequence[Mapping[str, object]],
    holdout_rows: Sequence[CleanTrainRow],
) -> Sequence[Dict[str, object]]:
    indexed = {int(row["local_holdout_index"]): row for row in clean_rows}
    if len(indexed) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("Visual selection requires every clean holdout row.")
    selected: Dict[int, set[str]] = {}

    def add(local_index: int, tag: str) -> None:
        selected.setdefault(int(local_index), set()).add(tag)

    for category in (
        "class1_keeper_tp",
        "class1_keeper_fn",
        "restricted_keeper_fp",
    ):
        candidates = [
            local_index
            for local_index, source in enumerate(holdout_rows)
            if _event_category(source) == category
        ]
        if candidates:
            add(
                max(
                    candidates,
                    key=lambda index: float(indexed[index]["weighted_object_gain"]),
                ),
                category,
            )
    valid = sorted(indexed)
    add(max(valid, key=lambda index: float(indexed[index]["bbox_area"])), "close")
    add(min(valid, key=lambda index: float(indexed[index]["bbox_area"])), "wide")
    tiny = [index for index in valid if bool(indexed[index]["tiny"])]
    edge = [index for index in valid if bool(indexed[index]["edge"])]
    add(
        min(tiny or valid, key=lambda index: float(indexed[index]["bbox_area"])),
        "tiny",
    )
    add(
        min(edge or valid, key=lambda index: float(indexed[index]["bbox_edge_gap"])),
        "edge",
    )
    partial = [
        index for index in valid if float(indexed[index]["bbox_edge_gap"]) <= 0.0
    ]
    add(
        min(partial or valid, key=lambda index: float(indexed[index]["bbox_edge_gap"])),
        "partial",
    )
    result = []
    for local_index in sorted(selected):
        source = holdout_rows[local_index]
        result.append(
            {
                "local_holdout_index": local_index,
                "sample_index": int(source.sample_index),
                "source_stem": source.source_stem,
                "target": int(source.target),
                "keeper_prediction": int(source.keeper_prediction),
                "event_category": _event_category(source)
                or "geometry_representative",
                "tags": sorted(selected[local_index]),
            }
        )
    return result


def _draw_fusion_overlay(
    image_tensor: Tensor,
    bbox: Tensor,
    row: Mapping[str, object],
    *,
    mean: Sequence[float],
    std: Sequence[float],
    title: str,
) -> Image.Image:
    rgb = _rgb_from_tensor(image_tensor, mean=mean, std=std)
    base = Image.fromarray(rgb).convert("RGBA")
    width, height = base.size
    stage1_dropped = _parse_numbers(row["stage1_dropped_indices"], integer=True)
    stage1_weights = _parse_numbers(row["stage1_fusion_weights"])
    stage2_dropped = _parse_numbers(row["stage2_dropped_indices"], integer=True)
    stage2_weights = _parse_numbers(row["stage2_fusion_weights"])
    final_kept = set(_parse_numbers(row["stage2_kept_indices"], integer=True))
    if len(stage1_dropped) != len(stage1_weights) or len(stage2_dropped) != len(stage2_weights):
        raise ValueError("Visual fusion row has misaligned dropped indices/weights.")
    lineage = np.zeros(ORIGINAL_PATCH_COUNT, dtype=np.float64)
    for index, weight in zip(stage1_dropped, stage1_weights):
        lineage[int(index)] += float(weight)
    lineage *= float(row["stage2_previous_context_attention"])
    for index, weight in zip(stage2_dropped, stage2_weights):
        lineage[int(index)] += float(weight)
    maximum = float(lineage.max())
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    stage1_set = set(int(value) for value in stage1_dropped)
    stage2_set = set(int(value) for value in stage2_dropped)
    patch_width = width / float(GRID_SIZE)
    patch_height = height / float(GRID_SIZE)
    for index in range(ORIGINAL_PATCH_COUNT):
        patch_row, patch_column = divmod(index, GRID_SIZE)
        rectangle = (
            int(round(patch_column * patch_width)),
            int(round(patch_row * patch_height)),
            int(round((patch_column + 1) * patch_width)),
            int(round((patch_row + 1) * patch_height)),
        )
        if lineage[index] > 0.0 and maximum > 0.0:
            normalized = min(1.0, float(lineage[index] / maximum))
            alpha = int(round(28 + 170 * math.sqrt(normalized)))
            draw_overlay.rectangle(rectangle, fill=(235, 90, 25, alpha))
        if index in stage1_set:
            draw_overlay.rectangle(rectangle, outline=(45, 105, 220, 150), width=1)
        if index in stage2_set:
            draw_overlay.rectangle(rectangle, outline=(230, 55, 25, 190), width=2)
        if index in final_kept:
            draw_overlay.rectangle(rectangle, outline=(40, 190, 75, 130), width=1)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    values = bbox.detach().float().cpu().flatten()
    center_x, center_y, bbox_width, bbox_height = [float(value) for value in values[:4]]
    bbox_rectangle = (
        int(round((center_x - bbox_width / 2.0) * width)),
        int(round((center_y - bbox_height / 2.0) * height)),
        int(round((center_x + bbox_width / 2.0) * width)),
        int(round((center_y + bbox_height / 2.0) * height)),
    )
    draw.rectangle(bbox_rectangle, outline=(20, 225, 230), width=4)
    for index in range(1, GRID_SIZE):
        x = int(round(index * patch_width))
        y = int(round(index * patch_height))
        draw.line((x, 0, x, height), fill=(255, 255, 255), width=1)
        draw.line((0, y, width, y), fill=(255, 255, 255), width=1)
    header_height = 58
    canvas = Image.new("RGB", (width, height + header_height), (248, 248, 248))
    canvas.paste(composed, (0, header_height))
    header = ImageDraw.Draw(canvas)
    header.text((6, 5), title[:105], fill=(10, 10, 10))
    header.text(
        (6, 25),
        (
            f"obj_w={float(row['weighted_object_fraction']):.3f} "
            f"raw={float(row['raw_dropped_object_fraction']):.3f} "
            f"out={float(row['outside_context_fraction']):.3f} "
            f"J2={float(row['second_prune_jaccard']):.3f}"
        ),
        fill=(25, 25, 25),
    )
    header.text(
        (6, 42),
        "cyan=bbox orange=effective context blue=drop1 red=drop2 green=kept",
        fill=(45, 45, 45),
    )
    return canvas


def _render_visual_overlays(
    *,
    base_dataset,
    transform,
    selected_rows: Sequence[Mapping[str, object]],
    behavior_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    row_map = {
        (str(row["condition"]), int(row["local_holdout_index"])): row
        for row in behavior_rows
    }
    cells: list[Tuple[int, str, Image.Image]] = []
    for row_index, selected in enumerate(selected_rows):
        local_index = int(selected["local_holdout_index"])
        for condition, brightness, contrast in CONDITIONS:
            dataset = _SelectedConditionDataset(
                base_dataset,
                [local_index],
                corruption=_condition_corruption(condition, brightness, contrast),
                transform=transform,
            )
            image_tensor, _label, metadata = dataset[0]
            behavior = row_map[(condition, local_index)]
            title = (
                f"{condition} idx={selected['sample_index']} y={selected['target']} "
                f"c={behavior['control_prediction']} f={behavior['candidate_prediction']} "
                f"{','.join(selected['tags'])}"
            )
            cells.append(
                (
                    row_index,
                    condition,
                    _draw_fusion_overlay(
                        image_tensor,
                        metadata["bbox"],
                        behavior,
                        mean=mean,
                        std=std,
                        title=title,
                    ),
                )
            )
    if not cells:
        raise ValueError("No fusion visual-review cells were rendered.")
    rows_per_page = 3
    page_paths: list[str] = []
    page_hashes: Dict[str, str] = {}
    for page_index, row_start in enumerate(
        range(0, len(selected_rows), rows_per_page), start=1
    ):
        page_rows = min(rows_per_page, len(selected_rows) - row_start)
        cell_width, cell_height = cells[0][2].size
        page = Image.new(
            "RGB",
            (cell_width * len(CONDITIONS), cell_height * page_rows),
            (235, 235, 235),
        )
        for row_offset in range(page_rows):
            global_row = row_start + row_offset
            row_cells = [cell for index, _condition, cell in cells if index == global_row]
            if len(row_cells) != len(CONDITIONS):
                raise RuntimeError("Fusion visual page has incomplete condition coverage.")
            for column, cell in enumerate(row_cells):
                page.paste(cell, (column * cell_width, row_offset * cell_height))
        path = output_dir / f"fusion_overlay_page_{page_index:02d}.png"
        page.save(path, format="PNG", optimize=True)
        resolved = str(path.resolve())
        page_paths.append(resolved)
        page_hashes[resolved] = _sha256(path)
    represented_tags = {
        str(tag) for selected in selected_rows for tag in selected["tags"]
    }
    required_tags = {
        "class1_keeper_tp",
        "class1_keeper_fn",
        "restricted_keeper_fp",
        "close",
        "wide",
        "partial",
        "tiny",
        "edge",
    }
    return {
        "selected_rows": list(selected_rows),
        "selected_row_count": len(selected_rows),
        "conditions": [name for name, _, _ in CONDITIONS],
        "represented_tags": sorted(represented_tags),
        "required_tags": sorted(required_tags),
        "coverage_complete": required_tags.issubset(represented_tags),
        "pages": page_paths,
        "page_sha256": page_hashes,
        "page_count": len(page_paths),
    }


def _write_manifest(output_dir: Path) -> Path:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": METHOD,
                "artifacts": artifacts,
                "raw_data_modified": False,
                "validation_used": False,
                "test_used": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Fusion preflight summary does not exist: {summary_path}")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual finalization requires a result and nonempty note.")
    expected_summary = str(args.expected_summary_sha256).strip().lower()
    if len(expected_summary) != 64:
        raise ValueError("Visual finalization requires --expected-summary-sha256.")
    observed_summary = _sha256(summary_path)
    if observed_summary != expected_summary:
        raise ValueError(
            "Fusion summary changed before visual review: "
            f"observed={observed_summary} expected={expected_summary}"
        )
    review_path = output_dir / "visual_review.json"
    if review_path.exists():
        raise FileExistsError(f"Fusion visual review is already finalized: {review_path}")
    summary = _load_json(summary_path)
    visual = summary.get("visual_review_artifacts")
    if not isinstance(visual, Mapping):
        raise ValueError("Fusion summary lacks visual-review artifacts.")
    pages = visual.get("pages")
    page_hashes = visual.get("page_sha256")
    if not isinstance(pages, list) or not isinstance(page_hashes, Mapping) or not pages:
        raise ValueError("Fusion visual-review pages are incomplete.")
    observed_hashes: Dict[str, str] = {}
    for raw_path in pages:
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fusion visual-review page is missing: {path}")
        observed_hashes[str(path)] = _sha256(path)
    expected_hashes = {str(key): str(value) for key, value in page_hashes.items()}
    if observed_hashes != expected_hashes:
        raise ValueError("Fusion visual-review page hashes changed after automated audit.")
    passed = args.visual_review_result == "pass"
    review = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": str(args.visual_review_result),
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "pre_review_summary_sha256": observed_summary,
        "page_count": len(pages),
        "pages_sha256": observed_hashes,
    }
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    gate = summary["gate"]
    gate["visual_review_completed"] = True
    gate["visual_review_passed"] = passed
    gate["formal_pair_permission"] = bool(gate["automated_pass"] and passed)
    summary["visual_review"] = review
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    return summary


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Inattentive-token fusion preflight requires CUDA.")
    if (
        int(args.seed) != 42
        or int(args.batch_size) != 32
        or int(args.fp32_batch_size) != 2
        or int(args.audit_batch_size) != 32
    ):
        raise ValueError(
            "Fusion preflight is locked to seed42, batch32, FP32 batch2, audit batch32."
        )
    output_dir = _prepare_output(Path(args.output_dir))
    os.environ.setdefault("TRKH_ALLOW_WINDOWS_MULTIPROCESSING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    official_root = Path(args.official_root).resolve()
    paths = {
        "resolved_config": Path(args.resolved_config).resolve(),
        "scratch_launcher": Path(args.scratch_launcher).resolve(),
        "keeper_checkpoint": Path(args.keeper_checkpoint).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "declaration_summary": Path(args.declaration_summary).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "bra_closure": Path(args.bra_closure).resolve(),
        "official_evit": official_root / "evit.py",
        "official_helpers": official_root / "helpers.py",
        "official_license": official_root / "LICENSE",
        "paper": official_root / "paper_arxiv_2202.07800v2.pdf",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Locked fusion inputs are missing: {missing}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    resolved_config = _load_json(paths["resolved_config"])
    source_model_config = resolved_config.get("model_config")
    if not isinstance(source_model_config, Mapping):
        raise ValueError("Resolved config lacks model_config.")
    fold_summary = _load_json(paths["fold_summary"])
    checkpoint = torch.load(
        paths["keeper_checkpoint"], map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Keeper checkpoint must be a mapping.")
    checkpoint_model_config = checkpoint.get("model_config")
    checkpoint_state = checkpoint.get("model_state")
    if not isinstance(checkpoint_model_config, Mapping) or not isinstance(
        checkpoint_state, Mapping
    ):
        raise ValueError("Keeper checkpoint lacks model_config/model_state.")
    declaration_rows = _read_clean_train_rows(paths["declaration"])
    fit_rows = [row for row in declaration_rows if row.fold != FOLD]
    holdout_rows = [row for row in declaration_rows if row.fold == FOLD]

    control_config = _fusion_config(source_model_config, enabled=False)
    candidate_config = _fusion_config(source_model_config, enabled=True)
    fit_dataset = _build_dataset(
        fold_data=paths["fold_data"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    loader_kwargs, fit_loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="inattentive_fusion_train_only_preflight",
        persistent_workers=False,
    )
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    images_cpu, labels_cpu, metadata_cpu = _unpack_batch(next(iter(fit_loader)))
    base_dataset, eval_transform, holdout_dataset_summary = _build_holdout_dataset(
        checkpoint, fold_data=paths["fold_data"], holdout_rows=holdout_rows
    )
    cohort, geometries = _precompute_cohort(
        base_dataset=base_dataset,
        transform=eval_transform,
        holdout_rows=holdout_rows,
        output_dir=output_dir,
        batch_size=int(args.audit_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    device = torch.device("cuda")
    images = images_cpu.to(device=device, non_blocking=True)
    labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    metadata = {
        key: value.to(device=device, non_blocking=True)
        for key, value in metadata_cpu.items()
    }
    set_seed(int(args.seed), deterministic=True)
    scratch_control = create_model(num_classes=5, model_config=control_config)
    control_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    scratch_candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_rng = _rng_snapshot()
    constructor_state = _state_summary(scratch_control, scratch_candidate)
    scratch_candidate.load_state_dict(scratch_control.state_dict(), strict=True)
    matched_state = _state_summary(scratch_control, scratch_candidate)
    control_parameters = sum(
        int(parameter.numel()) for parameter in scratch_control.parameters()
    )
    candidate_parameters = sum(
        int(parameter.numel()) for parameter in scratch_candidate.parameters()
    )
    scratch_control = scratch_control.to(device)
    scratch_candidate = scratch_candidate.to(device)

    fp32_count = int(args.fp32_batch_size)
    fp32_backward = _full_backward_check(
        scratch_candidate,
        images[:fp32_count],
        labels[:fp32_count],
        {key: value[:fp32_count] for key, value in metadata.items()},
        bf16=False,
    )
    bf16_backward = _full_backward_check(
        scratch_candidate, images, labels, metadata, bf16=True
    )
    scratch_control.eval()
    scratch_candidate.eval()
    with torch.inference_mode():
        _, control_fp32_features = _forward(
            scratch_control, images, metadata, return_trace=True
        )
        _, candidate_fp32_features = _forward(
            scratch_candidate, images, metadata, return_trace=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, candidate_bf16_features = _forward(
                scratch_candidate, images, metadata, return_trace=True
            )
    engineering_pruning = _pruning_comparison(
        control_fp32_features["trace"]["pruning"],
        candidate_fp32_features["trace"]["pruning"],
    )
    precision_stages = []
    for fp32_stage, bf16_stage in zip(
        candidate_fp32_features["trace"]["pruning"],
        candidate_bf16_features["trace"]["pruning"],
    ):
        jaccard = _set_jaccard(
            fp32_stage["kept_indices"].detach().cpu(),
            bf16_stage["kept_indices"].detach().cpu(),
        )
        precision_stages.append(
            {
                "mean_jaccard": float(jaccard.mean().item()),
                "minimum_jaccard": float(jaccard.min().item()),
                "exact_fraction": float((jaccard == 1.0).float().mean().item()),
            }
        )
    precision_stability = {
        "stages": precision_stages,
        "passed": all(float(row["mean_jaccard"]) >= 0.98 for row in precision_stages),
    }
    trace_parity_fp32 = _trace_parity(
        scratch_candidate,
        images[:fp32_count],
        {key: value[:fp32_count] for key, value in metadata.items()},
        bf16=False,
    )
    trace_parity_bf16 = _trace_parity(
        scratch_candidate, images, metadata, bf16=True
    )
    candidate_context = candidate_fp32_features["inattentive_context"]
    public_layout = {
        "base_prefix_count": int(scratch_candidate.num_prefix_tokens),
        "context_shape": [int(value) for value in candidate_context.shape],
        "public_token_shape": [
            int(value) for value in candidate_fp32_features["tokens"].shape
        ],
        "patch_shape": [
            int(value) for value in candidate_fp32_features["patches"].shape
        ],
        "patch_index_shape": [
            int(value) for value in candidate_fp32_features["patch_indices"].shape
        ],
        "active_prefix_count": int(
            candidate_fp32_features["trace"]["active_prefix_count"].item()
        ),
        "indices_nonnegative": bool(
            (candidate_fp32_features["patch_indices"] >= 0).all()
        ),
        "indices_bounded": bool(
            (candidate_fp32_features["patch_indices"] < ORIGINAL_PATCH_COUNT).all()
        ),
    }
    public_layout["passed"] = bool(
        public_layout["base_prefix_count"] == 7
        and public_layout["context_shape"] == [int(images.size(0)), 1, 256]
        and public_layout["patch_shape"][:2]
        == public_layout["patch_index_shape"]
        and public_layout["public_token_shape"][1]
        == public_layout["base_prefix_count"] + public_layout["patch_shape"][1]
        and public_layout["active_prefix_count"] == 8
        and public_layout["indices_nonnegative"]
        and public_layout["indices_bounded"]
    )

    legacy_config = dict(checkpoint_model_config)
    legacy_config.pop("inattentive_token_fusion", None)
    legacy_config["pretrained"] = False
    set_seed(int(args.seed), deterministic=True)
    legacy = create_model(num_classes=5, model_config=legacy_config)
    load_model_state(legacy, dict(checkpoint_state), strict=True)
    keeper_control, keeper_candidate, keeper_model_summary = _construct_keeper_models(
        checkpoint, seed=int(args.seed)
    )
    legacy = legacy.to(device).eval()
    keeper_control = keeper_control.to(device).eval()
    with torch.inference_mode():
        legacy_logits, legacy_features = _forward(
            legacy,
            images[:fp32_count],
            {key: value[:fp32_count] for key, value in metadata.items()},
            return_trace=True,
        )
        explicit_logits, explicit_features = _forward(
            keeper_control,
            images[:fp32_count],
            {key: value[:fp32_count] for key, value in metadata.items()},
            return_trace=True,
        )
    default_off_trace = _tensor_tree_comparison(
        legacy_features, explicit_features
    )
    default_off = {
        "logit_maximum_absolute_error": float(
            (legacy_logits - explicit_logits).abs().amax().item()
        ),
        "argmax_match": bool(
            torch.equal(legacy_logits.argmax(dim=1), explicit_logits.argmax(dim=1))
        ),
        "feature_trace": default_off_trace,
        "passed": bool(
            torch.equal(legacy_logits, explicit_logits)
            and default_off_trace["exact"]
        ),
    }
    del legacy
    gc.collect()
    torch.cuda.empty_cache()

    del scratch_control, scratch_candidate
    gc.collect()
    torch.cuda.empty_cache()
    control_train_resource = _benchmark(
        config=control_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    candidate_train_resource = _benchmark(
        config=candidate_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    control_inference = _inference_benchmark(
        config=control_config,
        images=images,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    candidate_inference = _inference_benchmark(
        config=candidate_config,
        images=images,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    train_runtime_ratio = float(candidate_train_resource["median_seconds"]) / max(
        float(control_train_resource["median_seconds"]), 1e-12
    )
    inference_runtime_ratio = float(candidate_inference["median_seconds"]) / max(
        float(control_inference["median_seconds"]), 1e-12
    )
    train_vram_ratio = float(candidate_train_resource["peak_vram_gib"]) / max(
        float(control_train_resource["peak_vram_gib"]), 1e-12
    )

    set_seed(int(args.seed), deterministic=True)
    export_candidate = create_model(
        num_classes=5, model_config=candidate_config
    ).eval()
    bbox = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(1, 4)
    image_mask = metadata_cpu.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            1, images_cpu.size(-2), images_cpu.size(-1), dtype=torch.bool
        )
    onnx_path = output_dir / "inattentive_fusion_candidate_static_batch1.onnx"
    try:
        onnx = _onnx_compare(
            wrapper=_FullExport(export_candidate),
            inputs=(
                images_cpu[:1].float(),
                bbox[:1].float(),
                image_mask[:1].bool(),
            ),
            input_names=("images", "bbox", "image_mask"),
            path=onnx_path,
        )
        import onnx as onnx_package

        graph = onnx_package.load(str(onnx_path)).graph
        operation_counts: Dict[str, int] = {}
        for node in graph.node:
            operation_counts[node.op_type] = operation_counts.get(node.op_type, 0) + 1
        onnx["operation_counts"] = operation_counts
        onnx["contains_topk"] = int(operation_counts.get("TopK", 0)) > 0
        onnx["contains_gather"] = any(
            int(operation_counts.get(name, 0)) > 0
            for name in ("Gather", "GatherElements", "GatherND")
        )
        onnx["contains_scatter"] = any(
            int(operation_counts.get(name, 0)) > 0
            for name in ("Scatter", "ScatterElements", "ScatterND")
        )
        onnx["contains_reduce_sum"] = int(operation_counts.get("ReduceSum", 0)) > 0
    except Exception as error:
        onnx = _failed_export(onnx_path, error)
        onnx["operation_counts"] = {}
        onnx["contains_topk"] = False
        onnx["contains_gather"] = False
        onnx["contains_scatter"] = False
        onnx["contains_reduce_sum"] = False

    official_equation = _official_equation_replay()
    behavior, behavior_rows = _behavior_audit(
        control=keeper_control,
        candidate=keeper_candidate,
        base_dataset=base_dataset,
        transform=eval_transform,
        dataset_summary=holdout_dataset_summary,
        holdout_rows=holdout_rows,
        cohort=cohort,
        geometries=geometries,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.audit_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    clean_rows = [row for row in behavior_rows if row["condition"] == "clean"]
    selected_visual_rows = _select_visual_rows(clean_rows, holdout_rows)
    semantics = holdout_dataset_summary["semantics"]
    visual_review_artifacts = _render_visual_overlays(
        base_dataset=base_dataset,
        transform=eval_transform,
        selected_rows=selected_visual_rows,
        behavior_rows=behavior_rows,
        output_dir=output_dir,
        mean=semantics["input_mean"],
        std=semantics["input_std"],
    )
    del keeper_control, keeper_candidate
    gc.collect()
    torch.cuda.empty_cache()

    official_repository = _repository_state(official_root)
    source_hash_checks = {
        f"{name}_hash": hashes[name] == expected
        for name, expected in LOCKED_HASHES.items()
    }
    schedule = getattr(
        create_model(num_classes=5, model_config=candidate_config),
        "token_prune_schedule",
        {},
    )
    checks: Dict[str, bool] = {
        "tracked_worktree_clean": _tracked_worktree_clean(),
        "head_pushed": _git_value("rev-parse", "HEAD")
        == _git_value("rev-parse", "@{upstream}"),
        **source_hash_checks,
        "official_commit": official_repository["commit"] == LOCKED_OFFICIAL_COMMIT,
        "official_tree": official_repository["tree"] == LOCKED_OFFICIAL_TREE,
        "official_tracked_clean": bool(official_repository["tracked_clean"]),
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_sources": int(fold_summary.get("fit_sources", -1))
        == EXPECTED_FIT_SOURCES,
        "fold_holdout_sources": int(fold_summary.get("holdout_sources", -1))
        == EXPECTED_HOLDOUT_SOURCES,
        "fold_source_disjoint": int(fold_summary.get("source_overlap", -1)) == 0,
        "fold_fit_counts": fold_summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fit_index_hash": _ordered_index_sha256(
            [row.sample_index for row in fit_rows]
        )
        == EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_hash": _ordered_index_sha256(
            [row.sample_index for row in holdout_rows]
        )
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "fit_dataset_rows": len(fit_dataset) == EXPECTED_FIT_ROWS,
        "cohort_complete_before_scoring": bool(
            cohort["computed_before_model_scoring"]
            and int(cohort["cohort_count"]) > 0
            and len(cohort["sample_indices"]) == int(cohort["cohort_count"])
        ),
        "locked_architecture": bool(
            int(candidate_config.get("image_size", -1)) == 256
            and int(candidate_config.get("embed_dim", -1)) == 256
            and int(candidate_config.get("depth", -1)) == 8
            and int(candidate_config.get("num_heads", -1)) == 8
            and bool(candidate_config["inattentive_token_fusion"])
            and schedule == {1: 0.85, 4: 0.65}
        ),
        "parameter_free_inventory": bool(
            control_parameters == candidate_parameters
            and constructor_state["inventory_equal"]
            and constructor_state["all_common_bit_exact"]
            and matched_state["inventory_equal"]
            and matched_state["all_common_bit_exact"]
        ),
        "constructor_rng_equal": _rng_equal(control_rng, candidate_rng),
        "keeper_state_exact": bool(
            keeper_model_summary["constructor_rng_equal"]
            and keeper_model_summary["constructor_state"]["all_common_bit_exact"]
            and keeper_model_summary["loaded_state"]["all_common_bit_exact"]
            and not keeper_model_summary["control_missing_keys"]
            and not keeper_model_summary["candidate_missing_keys"]
            and not keeper_model_summary["control_unexpected_keys"]
            and not keeper_model_summary["candidate_unexpected_keys"]
        ),
        "official_equation_replay": bool(official_equation["passed"]),
        "default_off_exact": bool(default_off["passed"]),
        "fp32_backward": bool(fp32_backward["passed"]),
        "bf16_backward": bool(bf16_backward["passed"]),
        "engineering_pruning": bool(engineering_pruning["passed"]),
        "precision_stability": bool(precision_stability["passed"]),
        "trace_parity_fp32": bool(trace_parity_fp32["passed"]),
        "trace_parity_bf16": bool(trace_parity_bf16["passed"]),
        "public_layout": bool(public_layout["passed"]),
        "onnx_succeeded": bool(onnx.get("succeeded", False)),
        "onnx_finite": bool(onnx.get("finite", False)),
        "onnx_error": float(onnx.get("maximum_absolute_error", 1e9)) <= 1e-4,
        "onnx_argmax": bool(onnx.get("argmax_match", False)),
        "onnx_dynamic_fusion_ops": bool(
            onnx.get("contains_topk", False)
            and onnx.get("contains_gather", False)
            and onnx.get("contains_scatter", False)
            and onnx.get("contains_reduce_sum", False)
        ),
        "inference_runtime_ratio": math.isfinite(inference_runtime_ratio)
        and inference_runtime_ratio <= MAX_INFERENCE_RUNTIME_RATIO,
        "train_runtime_ratio": math.isfinite(train_runtime_ratio)
        and train_runtime_ratio <= MAX_TRAIN_RUNTIME_RATIO,
        "candidate_vram_budget": float(candidate_train_resource["peak_vram_gib"])
        <= MAX_VRAM_GIB,
        "candidate_vram_ratio": math.isfinite(train_vram_ratio)
        and train_vram_ratio <= MAX_VRAM_RATIO,
        "resource_finite": all(
            bool(value["finite"])
            for value in (
                control_train_resource,
                candidate_train_resource,
                control_inference,
                candidate_inference,
            )
        ),
        **{
            f"behavior_{name}": bool(value)
            for name, value in behavior["checks"].items()
        },
        "visual_artifact_coverage": bool(
            visual_review_artifacts["coverage_complete"]
            and visual_review_artifacts["page_count"] > 0
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "upstream": _git_value("rev-parse", "@{upstream}"),
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "official_repository": official_repository,
        "fold": fold_summary,
        "data": {
            "fit_rows": len(fit_dataset),
            "holdout_rows": len(holdout_rows),
            "fit_batch_size": int(images.size(0)),
            "fit_batch_labels": [int(value) for value in labels.cpu().tolist()],
            "fit_loader": fit_loader_summary,
            "cohort": cohort,
        },
        "model": {
            "control_parameters": control_parameters,
            "candidate_parameters": candidate_parameters,
            "constructor_state": constructor_state,
            "matched_state": matched_state,
            "rng": {
                "control": _rng_summary(control_rng),
                "candidate": _rng_summary(candidate_rng),
            },
            "keeper_models": keeper_model_summary,
            "token_prune_schedule": schedule,
            "public_layout": public_layout,
        },
        "official_equation": official_equation,
        "default_off": default_off,
        "fp32_backward": fp32_backward,
        "bf16_backward": bf16_backward,
        "engineering_pruning": engineering_pruning,
        "precision_stability": precision_stability,
        "trace_parity": {
            "fp32": trace_parity_fp32,
            "bf16": trace_parity_bf16,
        },
        "resource": {
            "control_train_step": control_train_resource,
            "candidate_train_step": candidate_train_resource,
            "control_inference": control_inference,
            "candidate_inference": candidate_inference,
            "train_runtime_ratio": train_runtime_ratio,
            "inference_runtime_ratio": inference_runtime_ratio,
            "train_peak_vram_ratio": train_vram_ratio,
        },
        "onnx": onnx,
        "behavior": behavior,
        "visual_review_artifacts": visual_review_artifacts,
        "validation_used": False,
        "test_used": False,
        "gate": {
            "checks": checks,
            "failed_checks": failed,
            "automated_pass": not failed,
            "visual_review_completed": False,
            "visual_review_passed": False,
            "formal_pair_permission": False,
            "validation_permission": False,
            "test_permission": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "visual_review_required.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "required": True,
                "page_count": visual_review_artifacts["page_count"],
                "summary_sha256": _sha256(summary_path),
                "finalize_command": (
                    "python -m trkh.tools.audit_inattentive_token_fusion_preflight "
                    f"--output-dir \"{output_dir}\" --finalize-visual-review "
                    f"--expected-summary-sha256 {_sha256(summary_path)} "
                    "--visual-review-result pass|fail --visual-review-note \"...\""
                ),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = (
        _finalize_visual_review(args)
        if bool(args.finalize_visual_review)
        else run_preflight(args)
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True), flush=True)
    if bool(args.finalize_visual_review):
        if not bool(summary["gate"]["formal_pair_permission"]):
            raise SystemExit(2)
    elif not bool(summary["gate"]["automated_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

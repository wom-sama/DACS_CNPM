from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import random
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import psutil

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset, build_train_transform
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_class1_protected_rsc_readiness import (
    _features_from_batch,
    _parameter_group,
    _parameter_group_hashes,
    _predict_fp32,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import (
    _failed_export,
    _make_lighting_loader,
    _onnx_compare,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _comparison,
    _make_loader,
    _prepare_output_dir,
    _sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
FOCUS_CLASS = 1
FIT_FOLDS = (1, 2, 3, 4)
ROLE_SPECS = (
    ("context_geometry", 1),
    ("illumination_surface", 2),
    ("occlusion_parts", 3),
    ("keeper_mixed", 4),
)
UPDATES_PER_ROLE = 8
CYCLE_UPDATES = 4
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_HOLDOUT_ROWS = 1843
EXPECTED_BRANCH_ROWS = 256
EXPECTED_PROBE_ROWS_PER_FOLD = 64
EXPECTED_BRANCH_CLASS_COUNTS = {
    1: (54, 16, 51, 71, 64),
    2: (55, 15, 54, 69, 63),
    3: (56, 14, 53, 69, 64),
    4: (52, 15, 54, 71, 64),
}
EXPECTED_BRANCH_SOURCE_COUNTS = {1: 233, 2: 237, 3: 227, 4: 235}
EXPECTED_BRANCH_INDEX_SHA256 = {
    1: "dde3e3836cfb1777cfb40acdaf223e53c6a7d73550f539bd791be1799e48cca2",
    2: "6dd694b539c77bbba50caa192b2b3be1023833106bea79be973faa5434b8b41f",
    3: "6a53d03930a94fca27729160802e548a6e6413a113f72370b5c7fcb8ec63f83f",
    4: "ee75abd58cb502434612bd3186ee538f770627d94a31ad08a3f6a4ee04bc23d1",
}
EXPECTED_UPDATE_INDEX_SHA256 = (
    "2279a58ace27fbb5840c31a037a3e9871d6173799279327c8729f9298b572bb7"
)
EXPECTED_PROBE_CLASS_COUNTS = (54, 16, 52, 70, 64)
EXPECTED_PROBE_SOURCE_COUNTS = {1: 56, 2: 58, 3: 60, 4: 54}
EXPECTED_PROBE_INDEX_SHA256 = {
    1: "ec30267214ba2a71bdae2c4d2e4654d503371357f7b18cc9a234418a073ff617",
    2: "facac439f85707d92a20f82e60748a33e12dd53b9af5ffa68ffb263f77b8bb46",
    3: "c52a6c3294b17b24fc086438fae93e0d9d05d4e550d82f65205398c4e5c397a5",
    4: "0b5c7f8a76827e2f3c34772bfb7b3857adaad1abc6594e7dd0ee2b2be8dec307",
}
EXPECTED_PROBE_TOTAL_INDEX_SHA256 = (
    "6490523a2de91848dc06e3ace3f78be5cfb1cb1670a28e05468527eff050c5bd"
)
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "a9b2bd96a257bd00aa2a228d22b3072c22a044225c29b6adbc44e5c663a11e55"
LOCKED_PAPER_SHA256 = "6d941daf99495df5c1e88a6a4d53054c491ef14f34e259c443ac8db0c681b06d"
LOCKED_OFFICIAL_COMMIT = "62274e83d2f08eb416db61d0957476c53fde9361"
LOCKED_OFFICIAL_TREE = "83d56bedfc1159eb16ecbf8b4363e014d15c6f75"
LOCKED_OFFICIAL_HASHES = {
    "trainer": "02cfea225325f94d6d68671107f735b708065a57aa9a6ced316be080a8917aa6",
    "license": "3dbd7ffa11dde98c571ecf13468792d03d0185936f225de9cc99240b01dc3fcf",
    "readme": "876b96849ba23436865b6bc344fb2798d6f65c539d5bca841923e29cff3aa763",
}
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
MAX_BARRIER = 0.02
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.02
MAX_INFERENCE_MEMORY_RATIO = 1.02
MAX_TRAINING_VRAM_GIB = 4.5
BENCHMARK_REPEATS = 40
REQUIRED_PARAMETER_GROUPS = {
    "stem",
    "input_tokens",
    "transformer",
    "normalization",
    "fine_grained_pool",
    "classifier",
    "cnn_fusion",
    "other",
}


@dataclass(frozen=True)
class CachedBatch:
    images: Tensor
    targets: Tensor
    bbox: Tensor
    image_mask: Tensor
    sample_indices: Tensor

    def metadata(self) -> Dict[str, Tensor]:
        return {
            "bbox": self.bbox,
            "image_mask": self.image_mask,
            "sample_index": self.sample_indices,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only DART recurrent-aggregation A0. Validation and "
            "test construction are forbidden."
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
            "docs/TRKH_5CLASS_DART_RECURRENT_AGGREGATION_A0_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\Jain_DART_CVPR2023_accepted.pdf"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\dart-cvpr2023"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_dart_recurrent_aggregation_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
        and int(args.seed) == SEED
        and math.isclose(float(args.learning_rate), 1e-5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.weight_decay), 0.05, rel_tol=0.0, abs_tol=1e-12)
        and int(args.benchmark_repeats) == BENCHMARK_REPEATS
    )


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
        / "test_audit_dart_recurrent_aggregation_readiness.py",
        "launcher": root
        / "scripts"
        / "run_trkh_dart_recurrent_aggregation_a0.ps1",
    }
    relative_paths: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"DART {name} is missing: {path}")
        relative = path.relative_to(root).as_posix()
        _git_value(root, "ls-files", "--error-unmatch", relative)
        relative_paths[name] = relative
    status = _git_value(root, "status", "--porcelain", "--", *relative_paths.values())
    if status:
        raise ValueError(f"DART implementation is not commit-clean: {status}")
    return {
        name: {"path": relative_paths[name], "sha256": _sha256(path)}
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
        "trainer": official / "domainbed" / "trainer.py",
        "license": official / "LICENSE",
        "readme": official / "README.md",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _largest_remainder_quotas(
    class_counts: Mapping[int, int],
    limit: int,
    *,
    num_classes: int = 5,
) -> tuple[int, ...]:
    counts = [int(class_counts.get(index, 0)) for index in range(num_classes)]
    population = sum(counts)
    if population <= 0 or int(limit) <= 0 or int(limit) > population:
        raise ValueError("Invalid largest-remainder population or limit.")
    quotas = [(int(limit) * count) // population for count in counts]
    missing = int(limit) - sum(quotas)
    remainder_order = sorted(
        range(num_classes),
        key=lambda index: (-((int(limit) * counts[index]) % population), index),
    )
    for index in remainder_order[:missing]:
        quotas[index] += 1
    return tuple(quotas)


def _protocol_index_sha256(indices: Sequence[int]) -> str:
    payload = "\n".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _select_grouped_rows(
    rows: Sequence[CleanTrainRow],
    *,
    fold: int,
    limit: int,
    namespace: str,
) -> tuple[list[CleanTrainRow], Dict[str, object]]:
    candidates = [row for row in rows if int(row.fold) == int(fold)]
    quotas = _largest_remainder_quotas(
        Counter(int(row.target) for row in candidates),
        int(limit),
    )
    remaining = list(quotas)
    groups: Dict[str, list[CleanTrainRow]] = defaultdict(list)
    for row in candidates:
        groups[str(row.source_stem)].append(row)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            hashlib.sha256(
                f"{SEED}|{namespace}|fold={int(fold)}|{item[0]}".encode("utf-8")
            ).digest(),
            item[0],
        ),
    )
    selected: list[CleanTrainRow] = []
    selected_sources: list[str] = []
    for source_stem, group in ordered_groups:
        ordered = sorted(group, key=lambda row: int(row.sample_index))
        group_counts = Counter(int(row.target) for row in ordered)
        if all(group_counts[index] <= remaining[index] for index in range(5)):
            selected.extend(ordered)
            selected_sources.append(source_stem)
            for index in range(5):
                remaining[index] -= group_counts[index]
            if len(selected) == int(limit):
                break
    if len(selected) != int(limit) or any(remaining):
        raise ValueError(
            f"Could not select exact grouped quota for {namespace}/fold={fold}: "
            f"rows={len(selected)}, remaining={remaining}"
        )
    class_counts = tuple(
        Counter(int(row.target) for row in selected).get(index, 0)
        for index in range(5)
    )
    return selected, {
        "fold": int(fold),
        "namespace": str(namespace),
        "rows": len(selected),
        "source_groups": len(selected_sources),
        "class_quotas": list(quotas),
        "class_counts": list(class_counts),
        "ordered_sample_index_sha256": _protocol_index_sha256(
            [row.sample_index for row in selected]
        ),
    }


def _build_locked_cohorts(
    rows: Sequence[CleanTrainRow],
) -> tuple[
    Dict[str, list[CleanTrainRow]],
    list[CleanTrainRow],
    list[CleanTrainRow],
    Dict[str, object],
]:
    source_folds: Dict[str, set[int]] = defaultdict(set)
    for row in rows:
        source_folds[row.source_stem].add(int(row.fold))
    cross_fold_sources = {
        source: sorted(folds) for source, folds in source_folds.items() if len(folds) != 1
    }
    if cross_fold_sources:
        raise ValueError(f"Source stems cross CIDT folds: {len(cross_fold_sources)}")

    branches: Dict[str, list[CleanTrainRow]] = {}
    branch_summaries: Dict[str, object] = {}
    update_sources: set[str] = set()
    all_updates: list[CleanTrainRow] = []
    for role, fold in ROLE_SPECS:
        selected, summary = _select_grouped_rows(
            rows,
            fold=fold,
            limit=EXPECTED_BRANCH_ROWS,
            namespace="update",
        )
        if tuple(summary["class_counts"]) != EXPECTED_BRANCH_CLASS_COUNTS[fold]:
            raise ValueError(f"Locked branch class counts differ for fold {fold}.")
        if int(summary["source_groups"]) != EXPECTED_BRANCH_SOURCE_COUNTS[fold]:
            raise ValueError(f"Locked branch source count differs for fold {fold}.")
        if summary["ordered_sample_index_sha256"] != EXPECTED_BRANCH_INDEX_SHA256[fold]:
            raise ValueError(f"Locked branch index hash differs for fold {fold}.")
        branches[role] = selected
        branch_summaries[role] = summary
        all_updates.extend(selected)
        update_sources.update(row.source_stem for row in selected)
    update_hash = _protocol_index_sha256([row.sample_index for row in all_updates])
    if update_hash != EXPECTED_UPDATE_INDEX_SHA256:
        raise ValueError(f"Combined update index hash differs: {update_hash}")

    probe: list[CleanTrainRow] = []
    probe_summaries: Dict[str, object] = {}
    probe_sources: set[str] = set()
    remaining_rows = [row for row in rows if row.source_stem not in update_sources]
    for fold in FIT_FOLDS:
        selected, summary = _select_grouped_rows(
            remaining_rows,
            fold=fold,
            limit=EXPECTED_PROBE_ROWS_PER_FOLD,
            namespace="probe",
        )
        if int(summary["source_groups"]) != EXPECTED_PROBE_SOURCE_COUNTS[fold]:
            raise ValueError(f"Locked probe source count differs for fold {fold}.")
        if summary["ordered_sample_index_sha256"] != EXPECTED_PROBE_INDEX_SHA256[fold]:
            raise ValueError(f"Locked probe index hash differs for fold {fold}.")
        probe.extend(selected)
        probe_sources.update(row.source_stem for row in selected)
        probe_summaries[str(fold)] = summary
    probe_counts = tuple(
        Counter(row.target for row in probe).get(index, 0) for index in range(5)
    )
    probe_hash = _protocol_index_sha256([row.sample_index for row in probe])
    if probe_counts != EXPECTED_PROBE_CLASS_COUNTS:
        raise ValueError(f"Locked fit-probe class counts differ: {probe_counts}")
    if probe_hash != EXPECTED_PROBE_TOTAL_INDEX_SHA256:
        raise ValueError(f"Locked fit-probe index hash differs: {probe_hash}")

    holdout = [row for row in rows if int(row.fold) == 0]
    holdout_sources = {row.source_stem for row in holdout}
    overlap = {
        "update_probe": len(update_sources.intersection(probe_sources)),
        "update_holdout": len(update_sources.intersection(holdout_sources)),
        "probe_holdout": len(probe_sources.intersection(holdout_sources)),
    }
    if len(holdout) != EXPECTED_HOLDOUT_ROWS or any(overlap.values()):
        raise ValueError(
            f"Locked holdout/source-disjoint contract differs: {len(holdout)}, {overlap}"
        )
    return branches, probe, holdout, {
        "branch_summaries": branch_summaries,
        "probe_summaries": probe_summaries,
        "update_rows": len(all_updates),
        "update_source_groups": len(update_sources),
        "update_class_counts": [
            Counter(row.target for row in all_updates).get(index, 0)
            for index in range(5)
        ],
        "update_ordered_sample_index_sha256": update_hash,
        "probe_rows": len(probe),
        "probe_source_groups": len(probe_sources),
        "probe_class_counts": list(probe_counts),
        "probe_ordered_sample_index_sha256": probe_hash,
        "holdout_rows": len(holdout),
        "holdout_ordered_sample_index_sha256": _ordered_index_sha256(
            [row.sample_index for row in holdout]
        ),
        "source_overlap": overlap,
        "cross_fold_source_leaks": 0,
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[
    Dict[str, object],
    list[CleanTrainRow],
    Dict[str, list[CleanTrainRow]],
    list[CleanTrainRow],
    list[CleanTrainRow],
    Dict[str, object],
]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the locked DART A0 protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "DART protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "DART paper"),
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
        hashes[name] = _verify_sha256(paths[name], expected, f"official DART {name}")

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(paths["official_root"], "rev-parse", "HEAD^{tree}")
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(f"Official DART commit differs: {official_commit}")
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official DART tree differs: {official_tree}")
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official DART worktree must be clean.")
    trainer_text = paths["trainer"].read_text(encoding="utf-8")
    official_equation_present = bool(
        "def interpolate_algos(sd1, sd2, sd3, sd4):" in trainer_text
        and "/4 for key in sd1.keys()" in trainer_text
        and "step%args.inter_freq==0" in trainer_text
        and "algorithmCE4.featurizer.load_state_dict(inter_state_dict)" in trainer_text
    )
    if not official_equation_present:
        raise ValueError("Locked official DART averaging/broadcast source differs.")

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
    rows = _read_clean_train_rows(
        paths["cidt_predictions"],
        expected_rows=EXPECTED_TRAIN_ROWS,
        expected_class_counts=EXPECTED_CLASS_COUNTS,
        expected_fold_counts=EXPECTED_FOLD_COUNTS,
    )
    branches, probe, holdout, cohort_summary = _build_locked_cohorts(rows)

    repo_root = Path(__file__).resolve().parents[2]
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for DART A0.")
    implementation = _verify_committed_implementation(repo_root)
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal DART A0 requires the pushed repository commit.")
    provenance = {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "official_commit": official_commit,
        "official_tree": official_tree,
        "official_worktree_clean": True,
        "official_equation_present": official_equation_present,
        "repository_commit": repository_commit,
        "upstream_commit": upstream_commit,
        "tracked_worktree_clean": True,
        "implementation_committed_clean": True,
        "implementation": implementation,
        "validation_data_used": False,
        "test_data_used": False,
    }
    return provenance, rows, branches, probe, holdout, cohort_summary


def _role_transform_config(role: str) -> Dict[str, object]:
    common: Dict[str, object] = {
        "image_size": 256,
        "resize_mode": "pad",
        "scale_min": 1.0,
        "scale_crop_probability": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
        "random_erasing_probability": 0.0,
        "random_affine_degrees": 0.0,
        "random_affine_translate": 0.0,
        "random_affine_scale_min": 1.0,
        "horizontal_flip_probability": 0.0,
        "vertical_flip_probability": 0.0,
        "rotate90_probability": 0.0,
        "lighting_probability": 0.0,
        "randaugment_num_ops": 0,
        "randaugment_magnitude": 0,
        "illumination_normalization": True,
        "illumination_normalization_strength": 0.35,
        "foreground_crop_mode": "none",
        "foreground_crop_probability": 0.0,
        "background_suppression_mode": "none",
        "background_suppression_probability": 0.0,
        "background_suppression_margin": 0.08,
        "background_suppression_blur_radius": 7.0,
        "surface_detail_amplification_mode": "none",
        "surface_detail_amplification_probability": 0.0,
        "surface_detail_amplification_strength": 0.0,
        "local_exposure_probability": 0.0,
        "local_exposure_strength": 0.25,
        "obstacle_probability": 0.0,
        "obstacle_max_area": 0.08,
        "scale_photometric_with_augmentation": False,
    }
    if role == "context_geometry":
        common.update(
            scale_min=0.88,
            scale_crop_probability=0.35,
            random_affine_degrees=3.0,
            random_affine_translate=0.02,
            random_affine_scale_min=0.96,
            horizontal_flip_probability=0.5,
            rotate90_probability=0.03,
        )
    elif role == "illumination_surface":
        common.update(
            brightness=0.08,
            contrast=0.08,
            saturation=0.04,
            hue=0.01,
            lighting_probability=0.20,
            local_exposure_probability=0.30,
            local_exposure_strength=0.25,
            background_suppression_mode="desaturate_blur",
            background_suppression_probability=0.80,
        )
    elif role == "occlusion_parts":
        common.update(
            scale_min=0.88,
            scale_crop_probability=0.35,
            random_erasing_probability=0.08,
            random_affine_degrees=3.0,
            random_affine_translate=0.02,
            random_affine_scale_min=0.96,
            horizontal_flip_probability=0.5,
            rotate90_probability=0.03,
            obstacle_probability=0.12,
            obstacle_max_area=0.08,
            background_suppression_mode="desaturate_blur",
            background_suppression_probability=0.80,
        )
    elif role == "keeper_mixed":
        common.update(
            scale_min=0.88,
            scale_crop_probability=0.35,
            brightness=0.04,
            contrast=0.04,
            saturation=0.02,
            hue=0.01,
            random_affine_degrees=3.0,
            random_affine_translate=0.02,
            random_affine_scale_min=0.96,
            horizontal_flip_probability=0.5,
            rotate90_probability=0.03,
            local_exposure_probability=0.15,
            local_exposure_strength=0.25,
            obstacle_probability=0.04,
            obstacle_max_area=0.08,
            background_suppression_mode="desaturate_blur",
            background_suppression_probability=0.80,
        )
    else:
        raise ValueError(f"Unknown locked DART role: {role}")
    return common


def _build_role_transforms(
    semantics: Mapping[str, object],
) -> tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    mean = tuple(float(value) for value in semantics["input_mean"])
    std = tuple(float(value) for value in semantics["input_std"])
    transforms: Dict[str, object] = {}
    declarations: Dict[str, Dict[str, object]] = {}
    for role, _ in ROLE_SPECS:
        config = _role_transform_config(role)
        config["mean"] = mean
        config["std"] = std
        transforms[role] = build_train_transform(**config)
        declarations[role] = dict(config)
    return transforms, declarations


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}\n".encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cached_batch_manifest(batch: CachedBatch) -> Dict[str, object]:
    field_hashes = {
        "images": _tensor_sha256(batch.images),
        "targets": _tensor_sha256(batch.targets),
        "bbox": _tensor_sha256(batch.bbox),
        "image_mask": _tensor_sha256(batch.image_mask),
        "sample_indices": _tensor_sha256(batch.sample_indices),
    }
    digest = hashlib.sha256()
    for name, value in field_hashes.items():
        digest.update(f"{name}:{value}\n".encode("ascii"))
    return {
        "rows": int(batch.targets.numel()),
        "sample_indices": [int(value) for value in batch.sample_indices.tolist()],
        "field_sha256": field_hashes,
        "combined_sha256": digest.hexdigest(),
    }


def _materialization_seed(role: str) -> int:
    digest = hashlib.sha256(f"DART|{SEED}|materialize|{role}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _logical_seed(role: str, role_step: int) -> int:
    digest = hashlib.sha256(
        f"DART|{SEED}|update|{role}|{int(role_step)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _set_logical_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _materialize_role_batches(
    *,
    base_dataset: MangoYOLOCropDataset,
    transforms: Mapping[str, object],
    branches: Mapping[str, Sequence[CleanTrainRow]],
    args: argparse.Namespace,
) -> tuple[Dict[str, list[CachedBatch]], Dict[str, object]]:
    cache: Dict[str, list[CachedBatch]] = {}
    summaries: Dict[str, object] = {}
    for role, _ in ROLE_SPECS:
        indices = [row.sample_index for row in branches[role]]
        materialization_seed = _materialization_seed(role)
        set_seed(materialization_seed, deterministic=True)
        loader, loader_summary = _make_loader(
            base_dataset=base_dataset,
            transform=transforms[role],
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"dart_a0_{role}_materialize",
            seed=materialization_seed,
        )
        batches: list[CachedBatch] = []
        observed_indices: list[int] = []
        manifests: list[Dict[str, object]] = []
        for images, targets, metadata in loader:
            bbox = metadata.get("bbox")
            image_mask = metadata.get("image_mask")
            sample_indices = metadata.get("sample_index")
            if not all(torch.is_tensor(value) for value in (bbox, image_mask, sample_indices)):
                raise ValueError(f"DART role {role} is missing bbox/mask/sample metadata.")
            batch = CachedBatch(
                images=images.detach().float().cpu().contiguous().clone(),
                targets=targets.detach().long().cpu().contiguous().clone(),
                bbox=bbox.detach().float().cpu().contiguous().clone(),
                image_mask=image_mask.detach().bool().cpu().contiguous().clone(),
                sample_indices=sample_indices.detach().long().cpu().contiguous().clone(),
            )
            observed_indices.extend(int(value) for value in batch.sample_indices.tolist())
            batches.append(batch)
            manifests.append(_cached_batch_manifest(batch))
        if len(batches) != UPDATES_PER_ROLE or observed_indices != indices:
            raise ValueError(
                f"Locked DART materialization differs for {role}: "
                f"batches={len(batches)}, order_match={observed_indices == indices}"
            )
        cache[role] = batches
        summaries[role] = {
            "materialization_seed": materialization_seed,
            "loader": loader_summary,
            "ordered_sample_index_sha256": _ordered_index_sha256(observed_indices),
            "batches": manifests,
        }
    return cache, summaries


def _materialize_eval_batches(
    loader: DataLoader,
    *,
    expected_indices: Sequence[int],
) -> tuple[list[CachedBatch], Dict[str, object]]:
    batches: list[CachedBatch] = []
    manifests: list[Dict[str, object]] = []
    observed_indices: list[int] = []
    for images, targets, metadata in loader:
        bbox = metadata.get("bbox")
        image_mask = metadata.get("image_mask")
        sample_indices = metadata.get("sample_index")
        if not all(torch.is_tensor(value) for value in (bbox, image_mask, sample_indices)):
            raise ValueError("DART eval cache is missing bbox/mask/sample metadata.")
        batch = CachedBatch(
            images=images.detach().float().cpu().contiguous().clone(),
            targets=targets.detach().long().cpu().contiguous().clone(),
            bbox=bbox.detach().float().cpu().contiguous().clone(),
            image_mask=image_mask.detach().bool().cpu().contiguous().clone(),
            sample_indices=sample_indices.detach().long().cpu().contiguous().clone(),
        )
        observed_indices.extend(int(value) for value in batch.sample_indices.tolist())
        batches.append(batch)
        manifests.append(_cached_batch_manifest(batch))
    if observed_indices != [int(value) for value in expected_indices]:
        raise ValueError("DART eval-cache sample order differs from its locked cohort.")
    return batches, {
        "rows": len(observed_indices),
        "batches": len(batches),
        "ordered_sample_index_sha256": _protocol_index_sha256(observed_indices),
        "batch_manifests": manifests,
    }


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _optimizer_state_sha256(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> str:
    digest = hashlib.sha256()
    names = {parameter: name for name, parameter in model.named_parameters()}
    for parameter, state in sorted(
        optimizer.state.items(), key=lambda item: names.get(item[0], "")
    ):
        name = names.get(parameter)
        if name is None:
            raise ValueError("Optimizer contains an unknown model parameter.")
        digest.update(f"parameter:{name}\n".encode("utf-8"))
        for key, value in sorted(state.items(), key=lambda item: str(item[0])):
            digest.update(f"state:{key}\n".encode("utf-8"))
            if torch.is_tensor(value):
                digest.update(_tensor_sha256(value).encode("ascii"))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> list[float]:
    values = []
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step):
            values.append(float(step.detach().cpu().item()))
        elif step is not None:
            values.append(float(step))
    return sorted(set(values))


def _new_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
    )


def _new_tracker(model: nn.Module) -> Dict[str, object]:
    groups = sorted({_parameter_group(name) for name, _ in model.named_parameters()})
    return {
        "initial_group_hashes": _parameter_group_hashes(model),
        "gradient_seen": {group: False for group in groups},
        "all_gradients_finite": True,
        "history": [],
        "peak_vram_gib": 0.0,
    }


def _train_sequence(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sequence: Sequence[tuple[str, int, CachedBatch]],
    tracker: Dict[str, object],
    device: torch.device,
    segment: str,
) -> None:
    model.to(device).train()
    _optimizer_to(optimizer, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    history = tracker["history"]
    gradient_seen = tracker["gradient_seen"]
    assert isinstance(history, list) and isinstance(gradient_seen, dict)
    for role, role_step, batch in sequence:
        logical_seed = _logical_seed(role, role_step)
        _set_logical_rng(logical_seed)
        images = batch.images.to(device=device)
        targets = batch.targets.to(device=device)
        metadata = {
            "bbox": batch.bbox.to(device=device),
            "image_mask": batch.image_mask.to(device=device),
            "sample_index": batch.sample_indices,
        }
        optimizer.zero_grad(set_to_none=True)
        features = _features_from_batch(model, images, metadata, device=device)
        logits = classification_logits_from_features(model, features)
        loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"Non-finite DART loss in {segment}/{role}/{role_step}.")
        loss.backward()
        group_norm_squared: Dict[str, float] = defaultdict(float)
        for parameter_name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            group = _parameter_group(parameter_name)
            finite = bool(torch.isfinite(gradient).all().item())
            nonzero = bool(torch.count_nonzero(gradient).item())
            tracker["all_gradients_finite"] = bool(
                tracker["all_gradients_finite"] and finite
            )
            gradient_seen[group] = bool(gradient_seen.get(group, False) or nonzero)
            group_norm_squared[group] += float(
                gradient.detach().float().square().sum().item()
            )
        optimizer.step()
        history.append(
            {
                "segment": segment,
                "role": role,
                "role_step": int(role_step),
                "logical_seed": logical_seed,
                "batch_sha256": _cached_batch_manifest(batch)["combined_sha256"],
                "loss": float(loss.detach().item()),
                "gradient_norms": {
                    key: math.sqrt(value) for key, value in sorted(group_norm_squared.items())
                },
            }
        )
        del images, targets, metadata, features, logits, loss
    torch.cuda.synchronize(device)
    tracker["peak_vram_gib"] = max(
        float(tracker["peak_vram_gib"]),
        float(torch.cuda.max_memory_allocated(device) / (1024.0**3)),
    )
    model.cpu()
    _optimizer_to(optimizer, torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()


def _finalize_tracker(model: nn.Module, tracker: Mapping[str, object]) -> Dict[str, object]:
    before = tracker["initial_group_hashes"]
    assert isinstance(before, Mapping)
    after = _parameter_group_hashes(model)
    return {
        "updates": len(tracker["history"]),
        "state_sha256": _state_sha256(model),
        "parameter_group_hashes_before": dict(before),
        "parameter_group_hashes_after": after,
        "parameter_group_movement": {
            key: before.get(key) != after.get(key)
            for key in sorted(set(before).union(after))
        },
        "gradient_seen": dict(tracker["gradient_seen"]),
        "all_gradients_finite": bool(tracker["all_gradients_finite"]),
        "peak_vram_gib": float(tracker["peak_vram_gib"]),
    }


def _clone_parameter_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().float().cpu().contiguous().clone()
        for name, value in model.named_parameters()
    }


def _pairwise_delta_diagnostics(
    models: Sequence[nn.Module],
    base_parameters: Mapping[str, Tensor],
) -> list[Dict[str, object]]:
    named = [dict(model.named_parameters()) for model in models]
    results: list[Dict[str, object]] = []
    for left_index, right_index in combinations(range(len(models)), 2):
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        distance = 0.0
        for name in sorted(base_parameters):
            base = base_parameters[name].double()
            left = named[left_index][name].detach().cpu().double() - base
            right = named[right_index][name].detach().cpu().double() - base
            dot += float((left * right).sum().item())
            left_norm += float(left.square().sum().item())
            right_norm += float(right.square().sum().item())
            distance += float((left - right).square().sum().item())
        denominator = math.sqrt(left_norm * right_norm)
        cosine = dot / denominator if denominator > 0.0 else float("nan")
        results.append(
            {
                "left": left_index,
                "right": right_index,
                "delta_cosine": cosine,
                "endpoint_distance": math.sqrt(distance),
                "left_delta_norm": math.sqrt(left_norm),
                "right_delta_norm": math.sqrt(right_norm),
            }
        )
    return results


def _average_state_dicts_strict(
    states: Sequence[Mapping[str, Tensor]],
) -> tuple[Dict[str, Tensor], Dict[str, object]]:
    if len(states) != 4:
        raise ValueError("Locked DART averaging requires exactly four states.")
    keys = list(states[0].keys())
    if any(list(state.keys()) != keys for state in states[1:]):
        raise ValueError("DART branch state schemas differ.")
    output: Dict[str, Tensor] = {}
    nonfloating_keys: list[str] = []
    for key in keys:
        values = [state[key].detach().cpu().contiguous() for state in states]
        if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError(f"DART branch state tensor schema differs: {key}")
        if values[0].is_floating_point() or values[0].is_complex():
            if not all(bool(torch.isfinite(value).all().item()) for value in values):
                raise ValueError(f"Non-finite DART branch state: {key}")
            average = torch.stack([value.float() for value in values], dim=0).mean(dim=0)
            output[key] = average.to(dtype=values[0].dtype)
        else:
            nonfloating_keys.append(key)
            if not all(torch.equal(values[0], value) for value in values[1:]):
                raise ValueError(f"DART nonfloating buffer differs: {key}")
            output[key] = values[0].clone()
    return output, {
        "state_keys": len(keys),
        "nonfloating_keys": nonfloating_keys,
        "nonfloating_bit_exact": True,
        "floating_average_dtype": "float32",
    }


def _model_ce(
    model: nn.Module,
    batches: Sequence[CachedBatch],
    device: torch.device,
) -> float:
    model.to(device).eval()
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for batch in batches:
            images = batch.images.to(device=device)
            targets = batch.targets.to(device=device, dtype=torch.long)
            metadata = {
                "bbox": batch.bbox,
                "image_mask": batch.image_mask,
                "sample_index": batch.sample_indices,
            }
            features = _features_from_batch(model, images, metadata, device=device)
            logits = classification_logits_from_features(model, features)
            total_loss += float(F.cross_entropy(logits.float(), targets, reduction="sum").item())
            total_rows += int(targets.numel())
    model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    if total_rows <= 0:
        raise ValueError("Empty DART fit-probe loader.")
    return total_loss / total_rows


def _aggregation_event(
    *,
    label: str,
    models: Sequence[nn.Module],
    optimizers: Sequence[torch.optim.Optimizer],
    base_parameters: Mapping[str, Tensor],
    fit_probe_batches: Sequence[CachedBatch],
    device: torch.device,
    broadcast: bool,
) -> tuple[nn.Module, Dict[str, object]]:
    branch_state_hashes = [_state_sha256(model) for model in models]
    optimizer_hashes_before = [
        _optimizer_state_sha256(model, optimizer)
        for model, optimizer in zip(models, optimizers)
    ]
    optimizer_steps = [_optimizer_steps(optimizer) for optimizer in optimizers]
    pairwise = _pairwise_delta_diagnostics(models, base_parameters)
    branch_ce = [_model_ce(model, fit_probe_batches, device) for model in models]
    averaged_state, averaging = _average_state_dicts_strict(
        [model.state_dict() for model in models]
    )
    aggregate = copy.deepcopy(models[0]).cpu().eval()
    aggregate.load_state_dict(averaged_state, strict=True)
    aggregate_hash = _state_sha256(aggregate)
    aggregate_ce = _model_ce(aggregate, fit_probe_batches, device)
    barrier = float(aggregate_ce - float(np.mean(branch_ce)))
    broadcast_hashes: list[str] = []
    if broadcast:
        for model in models:
            model.load_state_dict(averaged_state, strict=True)
            broadcast_hashes.append(_state_sha256(model))
    optimizer_hashes_after = [
        _optimizer_state_sha256(model, optimizer)
        for model, optimizer in zip(models, optimizers)
    ]
    event = {
        "label": label,
        "branch_state_sha256": branch_state_hashes,
        "branch_fit_probe_ce": branch_ce,
        "mean_branch_fit_probe_ce": float(np.mean(branch_ce)),
        "aggregate_fit_probe_ce": aggregate_ce,
        "barrier": barrier,
        "pairwise_delta": pairwise,
        "averaging": averaging,
        "aggregate_state_sha256": aggregate_hash,
        "broadcast": bool(broadcast),
        "broadcast_state_sha256": broadcast_hashes,
        "broadcast_hashes_exact": bool(
            not broadcast or all(value == aggregate_hash for value in broadcast_hashes)
        ),
        "optimizer_state_sha256_before": optimizer_hashes_before,
        "optimizer_state_sha256_after": optimizer_hashes_after,
        "optimizer_state_preserved": optimizer_hashes_before == optimizer_hashes_after,
        "optimizer_steps": optimizer_steps,
        "optimizer_states_distinct": len(set(optimizer_hashes_before)) > 1,
    }
    return aggregate.cpu().eval(), event


def _run_mixed_control(
    *,
    prototype: nn.Module,
    batch_cache: Mapping[str, Sequence[CachedBatch]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, object], list[Dict[str, object]]]:
    model = copy.deepcopy(prototype).cpu().train()
    initial_state_sha256 = _state_sha256(model)
    optimizer = _new_optimizer(model, args)
    tracker = _new_tracker(model)
    sequence = [
        (role, role_step, batch_cache[role][role_step])
        for role_step in range(UPDATES_PER_ROLE)
        for role, _ in ROLE_SPECS
    ]
    _train_sequence(
        model=model,
        optimizer=optimizer,
        sequence=sequence,
        tracker=tracker,
        device=device,
        segment="mixed_control",
    )
    summary = _finalize_tracker(model, tracker)
    summary["initial_state_sha256"] = initial_state_sha256
    summary["optimizer_state_sha256"] = _optimizer_state_sha256(model, optimizer)
    summary["optimizer_steps"] = _optimizer_steps(optimizer)
    history = list(tracker["history"])
    del optimizer
    return model.cpu().eval(), summary, history


def _run_single_final_average(
    *,
    prototype: nn.Module,
    batch_cache: Mapping[str, Sequence[CachedBatch]],
    fit_probe_batches: Sequence[CachedBatch],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, object], list[Dict[str, object]], Dict[str, object]]:
    models = [copy.deepcopy(prototype).cpu().train() for _ in ROLE_SPECS]
    optimizers = [_new_optimizer(model, args) for model in models]
    trackers = [_new_tracker(model) for model in models]
    initial_hashes = [_state_sha256(model) for model in models]
    for index, (role, _) in enumerate(ROLE_SPECS):
        _train_sequence(
            model=models[index],
            optimizer=optimizers[index],
            sequence=[(role, step, batch_cache[role][step]) for step in range(CYCLE_UPDATES)],
            tracker=trackers[index],
            device=device,
            segment="single_final_phase1",
        )
    phase1 = {
        "state_sha256": [_state_sha256(model) for model in models],
        "optimizer_sha256": [
            _optimizer_state_sha256(model, optimizer)
            for model, optimizer in zip(models, optimizers)
        ],
    }
    for index, (role, _) in enumerate(ROLE_SPECS):
        _train_sequence(
            model=models[index],
            optimizer=optimizers[index],
            sequence=[
                (role, step, batch_cache[role][step])
                for step in range(CYCLE_UPDATES, UPDATES_PER_ROLE)
            ],
            tracker=trackers[index],
            device=device,
            segment="single_final_phase2",
        )
    aggregate, event = _aggregation_event(
        label="single_final_average",
        models=models,
        optimizers=optimizers,
        base_parameters=_clone_parameter_state(prototype),
        fit_probe_batches=fit_probe_batches,
        device=device,
        broadcast=False,
    )
    branch_summaries = [
        _finalize_tracker(model, tracker) for model, tracker in zip(models, trackers)
    ]
    history = [row for tracker in trackers for row in tracker["history"]]
    summary = {
        "initial_state_sha256": initial_hashes,
        "branches": branch_summaries,
        "updates": sum(int(value["updates"]) for value in branch_summaries),
        "aggregation": event,
        "aggregate_state_sha256": _state_sha256(aggregate),
        "peak_vram_gib": max(float(value["peak_vram_gib"]) for value in branch_summaries),
    }
    del models, optimizers, trackers
    gc.collect()
    return aggregate, summary, history, phase1


def _run_dart_recurrent(
    *,
    prototype: nn.Module,
    batch_cache: Mapping[str, Sequence[CachedBatch]],
    fit_probe_batches: Sequence[CachedBatch],
    final_only_phase1: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, object], list[Dict[str, object]]]:
    models = [copy.deepcopy(prototype).cpu().train() for _ in ROLE_SPECS]
    optimizers = [_new_optimizer(model, args) for model in models]
    trackers = [_new_tracker(model) for model in models]
    initial_hashes = [_state_sha256(model) for model in models]
    for index, (role, _) in enumerate(ROLE_SPECS):
        _train_sequence(
            model=models[index],
            optimizer=optimizers[index],
            sequence=[(role, step, batch_cache[role][step]) for step in range(CYCLE_UPDATES)],
            tracker=trackers[index],
            device=device,
            segment="dart_cycle1",
        )
    phase1_states = [_state_sha256(model) for model in models]
    phase1_optimizers = [
        _optimizer_state_sha256(model, optimizer)
        for model, optimizer in zip(models, optimizers)
    ]
    phase1_exact = bool(
        phase1_states == list(final_only_phase1["state_sha256"])
        and phase1_optimizers == list(final_only_phase1["optimizer_sha256"])
    )
    first_average, cycle1 = _aggregation_event(
        label="dart_cycle1",
        models=models,
        optimizers=optimizers,
        base_parameters=_clone_parameter_state(prototype),
        fit_probe_batches=fit_probe_batches,
        device=device,
        broadcast=True,
    )
    cycle2_base = _clone_parameter_state(first_average)
    del first_average
    for index, (role, _) in enumerate(ROLE_SPECS):
        _train_sequence(
            model=models[index],
            optimizer=optimizers[index],
            sequence=[
                (role, step, batch_cache[role][step])
                for step in range(CYCLE_UPDATES, UPDATES_PER_ROLE)
            ],
            tracker=trackers[index],
            device=device,
            segment="dart_cycle2",
        )
    final_average, cycle2 = _aggregation_event(
        label="dart_cycle2",
        models=models,
        optimizers=optimizers,
        base_parameters=cycle2_base,
        fit_probe_batches=fit_probe_batches,
        device=device,
        broadcast=True,
    )
    branch_summaries = [
        _finalize_tracker(model, tracker) for model, tracker in zip(models, trackers)
    ]
    history = [row for tracker in trackers for row in tracker["history"]]
    summary = {
        "initial_state_sha256": initial_hashes,
        "phase1_matches_final_only_exactly": phase1_exact,
        "phase1_state_sha256": phase1_states,
        "phase1_optimizer_sha256": phase1_optimizers,
        "branches": branch_summaries,
        "updates": sum(int(value["updates"]) for value in branch_summaries),
        "cycles": [cycle1, cycle2],
        "aggregate_state_sha256": _state_sha256(final_average),
        "peak_vram_gib": max(float(value["peak_vram_gib"]) for value in branch_summaries),
    }
    del models, optimizers, trackers
    gc.collect()
    return final_average, summary, history


def _logical_batch_map(history: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for row in history:
        key = f"{row['role']}:{int(row['role_step'])}"
        if key in output:
            raise ValueError(f"Duplicate DART logical batch key: {key}")
        output[key] = {
            "batch_sha256": str(row["batch_sha256"]),
            "logical_seed": int(row["logical_seed"]),
        }
    return output


def _predict_cpu_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[Dict[str, object]]:
    model.to(device).eval()
    rows = _predict_fp32(model=model, loader=loader, device=device)
    model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _write_prediction_csv(
    path: Path,
    *,
    source_rows: Sequence[CleanTrainRow],
    condition_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    source = {row.sample_index: row for row in source_rows}
    variants = ("mixed_control", "single_final_average", "dart_recurrent")
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for variant in variants:
        fields.append(f"{variant}_prediction")
        fields.extend(f"{variant}_prob_{index}" for index in range(5))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, rows_by_variant in condition_rows.items():
            indexed = {
                variant: {
                    int(row["sample_index"]): row for row in rows_by_variant[variant]
                }
                for variant in variants
            }
            ordered_indices = [
                int(row["sample_index"]) for row in rows_by_variant[variants[0]]
            ]
            for sample_index in ordered_indices:
                source_row = source[sample_index]
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source_row.source_stem,
                    "image_path": str(source_row.image_path),
                    "fold": source_row.fold,
                    "target": source_row.target,
                }
                for variant in variants:
                    value = indexed[variant][sample_index]
                    if int(value["target"]) != source_row.target:
                        raise ValueError("DART prediction target differs from CIDT target.")
                    output[f"{variant}_prediction"] = int(value["prediction"])
                    for class_index in range(5):
                        output[f"{variant}_prob_{class_index}"] = float(
                            value[f"prob_{class_index}"]
                        )
                writer.writerow(output)


def _replay_prediction_csv(path: Path) -> Dict[str, object]:
    variants = ("mixed_control", "single_final_average", "dart_recurrent")
    grouped: Dict[str, Dict[str, list[Dict[str, object]]]] = defaultdict(
        lambda: {variant: [] for variant in variants}
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            condition = str(raw["condition"])
            for variant in variants:
                row: Dict[str, object] = {
                    "sample_index": int(raw["sample_index"]),
                    "target": int(raw["target"]),
                    "prediction": int(raw[f"{variant}_prediction"]),
                }
                for class_index in range(5):
                    row[f"prob_{class_index}"] = float(raw[f"{variant}_prob_{class_index}"])
                grouped[condition][variant].append(row)
    expected_conditions = {"clean", *(name for name, _, _ in LIGHTING_CONDITIONS)}
    if set(grouped) != expected_conditions:
        raise ValueError(f"DART replay conditions differ: {sorted(grouped)}")
    output: Dict[str, object] = {"clean": {}, "illumination": {}}
    for condition, rows in grouped.items():
        comparisons = {
            "mixed_vs_recurrent": _comparison(
                control_rows=rows["mixed_control"],
                candidate_rows=rows["dart_recurrent"],
                num_classes=5,
                focus_class=FOCUS_CLASS,
            ),
            "final_vs_recurrent": _comparison(
                control_rows=rows["single_final_average"],
                candidate_rows=rows["dart_recurrent"],
                num_classes=5,
                focus_class=FOCUS_CLASS,
            ),
        }
        if condition == "clean":
            output["clean"] = comparisons
        else:
            output["illumination"][condition] = comparisons
    return output


def _maximum_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


class _FullExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _export_candidate(
    *,
    candidate: nn.Module,
    batch: CachedBatch,
    output_dir: Path,
) -> Dict[str, object]:
    path = output_dir / "dart_recurrent_candidate.onnx"
    try:
        result = _onnx_compare(
            wrapper=_FullExportWrapper(candidate.cpu().eval()),
            inputs=(batch.images[:1], batch.bbox[:1], batch.image_mask[:1]),
            input_names=("images", "bbox", "image_mask"),
            path=path,
        )
    except Exception as error:
        result = _failed_export(path, error)
    if path.is_file():
        result["deleted_after_verification"] = True
        result["deleted_size_bytes"] = int(path.stat().st_size)
        path.unlink()
    else:
        result["deleted_after_verification"] = False
        result["deleted_size_bytes"] = 0
    return result


def _benchmark_once(
    *,
    model: nn.Module,
    batch: CachedBatch,
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    model.to(device).eval()
    images = batch.images.to(device=device)
    bbox = batch.bbox.to(device=device)
    image_mask = batch.image_mask.to(device=device)
    metadata = {"bbox": bbox, "image_mask": image_mask}
    with torch.inference_mode():
        for _ in range(6):
            features = _features_from_batch(model, images, metadata, device=device)
            classification_logits_from_features(model, features)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        final_logits = None
        for _ in range(int(repeats)):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            features = _features_from_batch(model, images, metadata, device=device)
            final_logits = classification_logits_from_features(model, features)
            torch.cuda.synchronize(device)
            timings.append(float(time.perf_counter() - started))
        peak = float(torch.cuda.max_memory_allocated(device) / (1024.0**3))
    assert final_logits is not None
    output = {
        "median_seconds": float(np.median(timings)),
        "peak_vram_gib": peak,
        "finite": bool(torch.isfinite(final_logits).all().item()),
        "argmax": [int(value) for value in final_logits.argmax(dim=1).cpu().tolist()],
    }
    model.cpu().eval()
    del images, bbox, image_mask, metadata, final_logits
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _benchmark_pair(
    *,
    native: nn.Module,
    candidate: nn.Module,
    batch: CachedBatch,
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    native_rows = []
    candidate_rows = []
    native_rows.append(
        _benchmark_once(model=native, batch=batch, device=device, repeats=repeats)
    )
    candidate_rows.append(
        _benchmark_once(model=candidate, batch=batch, device=device, repeats=repeats)
    )
    candidate_rows.append(
        _benchmark_once(model=candidate, batch=batch, device=device, repeats=repeats)
    )
    native_rows.append(
        _benchmark_once(model=native, batch=batch, device=device, repeats=repeats)
    )
    native_time = float(np.median([row["median_seconds"] for row in native_rows]))
    candidate_time = float(
        np.median([row["median_seconds"] for row in candidate_rows])
    )
    native_memory = float(np.median([row["peak_vram_gib"] for row in native_rows]))
    candidate_memory = float(
        np.median([row["peak_vram_gib"] for row in candidate_rows])
    )
    return {
        "native_rounds": native_rows,
        "candidate_rounds": candidate_rows,
        "native_median_seconds": native_time,
        "candidate_median_seconds": candidate_time,
        "runtime_ratio": candidate_time / max(native_time, 1e-12),
        "native_peak_vram_gib": native_memory,
        "candidate_peak_vram_gib": candidate_memory,
        "memory_ratio": candidate_memory / max(native_memory, 1e-12),
        "all_outputs_finite": all(
            bool(row["finite"]) for row in [*native_rows, *candidate_rows]
        ),
    }


def _state_schema(model: nn.Module) -> list[tuple[str, tuple[int, ...], str]]:
    return [
        (name, tuple(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    ]


def _process_snapshot() -> Dict[str, object]:
    current_pid = os.getpid()
    rows = []
    for process in psutil.process_iter(
        attrs=("pid", "name", "create_time", "cmdline")
    ):
        try:
            name = str(process.info.get("name") or "").casefold()
            if name not in {"python.exe", "trtexec.exe"}:
                continue
            rows.append(
                {
                    "pid": int(process.info["pid"]),
                    "name": name,
                    "create_time": float(process.info.get("create_time") or 0.0),
                    "command_line": " ".join(process.info.get("cmdline") or []),
                    "current_auditor": int(process.info["pid"]) == current_pid,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "current_pid": current_pid,
        "processes": rows,
        "unexpected_processes": [row for row in rows if not row["current_auditor"]],
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


def _event_diversity_passed(event: Mapping[str, object]) -> bool:
    rows = event["pairwise_delta"]
    assert isinstance(rows, list)
    cosines = [float(row["delta_cosine"]) for row in rows]
    return bool(
        cosines
        and all(math.isfinite(value) and value > -0.50 for value in cosines)
        and min(cosines) < 0.995
        and all(float(row["endpoint_distance"]) > 0.0 for row in rows)
    )


def assess_dart_a0(
    *,
    structural_checks: Mapping[str, bool],
    clean_mixed: Mapping[str, object],
    clean_final: Mapping[str, object],
    illumination_mixed: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    mixed_delta = clean_mixed["delta"]
    mixed_transitions = clean_mixed["transitions"]
    mixed_support = int(clean_mixed["control"]["predicted_support"][FOCUS_CLASS])
    recurrent_support = int(clean_mixed["candidate"]["predicted_support"][FOCUS_CLASS])
    clean_checks = {
        "mixed_macro_f1_delta_nonnegative": float(mixed_delta["macro_f1"]) >= 0.0,
        "mixed_class1_f1_delta_gte_0p005": float(mixed_delta["class1_f1"]) >= 0.005,
        "mixed_class1_precision_delta_gte_0p005": float(
            mixed_delta["class1_precision"]
        )
        >= 0.005,
        "mixed_class1_recall_delta_gte_minus_0p005": float(
            mixed_delta["class1_recall"]
        )
        >= -0.005,
        "mixed_restricted_fp_reduction_gte_2": int(
            mixed_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "mixed_fn_rescues_gte_tp_breaks": int(mixed_transitions["focus_fn_rescue"])
        >= int(mixed_transitions["focus_tp_break"]),
        "mixed_corrections_gt_harms": int(mixed_transitions["candidate_correction"])
        > int(mixed_transitions["candidate_harm"]),
        "mixed_max_nonfocus_f1_drop_lte_0p010": float(
            clean_mixed["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "mixed_class1_support_at_least_95pct": recurrent_support
        >= int(math.ceil(0.95 * mixed_support)),
    }
    final_delta = clean_final["delta"]
    final_transitions = clean_final["transitions"]
    recurrence_checks = {
        "final_class1_f1_delta_gte_0p002": float(final_delta["class1_f1"]) >= 0.002,
        "final_class1_precision_delta_nonnegative": float(
            final_delta["class1_precision"]
        )
        >= 0.0,
        "final_class1_recall_delta_nonnegative": float(final_delta["class1_recall"])
        >= 0.0,
        "final_restricted_fp_not_increased": int(
            final_transitions["restricted_focus_fp_reduction"]
        )
        >= 0,
        "final_corrections_gte_harms": int(final_transitions["candidate_correction"])
        >= int(final_transitions["candidate_harm"]),
    }
    precision_nonnegative = sum(
        float(row["delta"]["class1_precision"]) >= 0.0 for row in illumination_mixed
    )
    illumination_checks = {
        "all_lighting_class1_f1_deltas_gte_minus_0p010": all(
            float(row["delta"]["class1_f1"]) >= -0.010 for row in illumination_mixed
        ),
        "all_lighting_class1_recall_deltas_gte_minus_0p015": all(
            float(row["delta"]["class1_recall"]) >= -0.015
            for row in illumination_mixed
        ),
        "lighting_precision_nonnegative_in_at_least_two": precision_nonnegative >= 2,
        "lighting_no_restricted_fp_increase": all(
            int(row["transitions"]["restricted_focus_fp_reduction"]) >= 0
            for row in illumination_mixed
        ),
        "lighting_aggregate_rescues_gte_tp_breaks": sum(
            int(row["transitions"]["focus_fn_rescue"]) for row in illumination_mixed
        )
        >= sum(
            int(row["transitions"]["focus_tp_break"]) for row in illumination_mixed
        ),
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **clean_checks,
        **recurrence_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "recurrence_checks": recurrence_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_nonvisual_gates_passed": not failed,
        "xai_protocol_authorized": not failed,
        "validation_smoke_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "current_best_update_authorized": False,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    (
        provenance,
        rows,
        branches,
        fit_probe,
        holdout,
        cohort_summary,
    ) = _load_locked_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Locked DART A0 requires CUDA.")
    process_before = _process_snapshot()
    gpu_before = _gpu_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "DART A0 found another python.exe/trtexec.exe; no process was terminated."
        )
    output_dir = _prepare_output_dir(args.output_dir)
    paths = _source_paths(args)
    device = torch.device("cuda")
    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    source_config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != 5 or not isinstance(source_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    prototype = create_model(num_classes=5, model_config=source_config)
    load_model_state(prototype, dict(state), strict=True)
    prototype.cpu().eval()
    prototype_hash = _state_sha256(prototype)
    parameter_count = sum(int(value.numel()) for value in prototype.parameters())
    semantics = _eval_semantics(checkpoint)

    data_spec = load_data_spec(paths["data"], class_name_mode="raw", expected_num_classes=5)
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper.")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    sample_paths = [Path(value).resolve() for value in base_dataset.sample_paths()]
    if len(sample_paths) != len(rows):
        raise ValueError("Dataset/CIDT row count mismatch.")
    for row, path in zip(rows, sample_paths):
        if row.image_path != path:
            raise ValueError(f"Dataset/CIDT path mismatch at {row.sample_index}.")

    role_transforms, transform_declarations = _build_role_transforms(semantics)
    batch_cache, batch_materialization = _materialize_role_batches(
        base_dataset=base_dataset,
        transforms=role_transforms,
        branches=branches,
        args=args,
    )
    eval_transform = _build_eval_transform(semantics)
    fit_probe_indices = [row.sample_index for row in fit_probe]
    fit_probe_loader, fit_probe_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=eval_transform,
        indices=fit_probe_indices,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="dart_a0_fit_probe",
        seed=SEED + 100,
    )
    fit_probe_batches, fit_probe_cache_summary = _materialize_eval_batches(
        fit_probe_loader,
        expected_indices=fit_probe_indices,
    )
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=eval_transform,
        indices=[row.sample_index for row in holdout],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="dart_a0_holdout",
        seed=SEED + 200,
    )
    resource_images, resource_targets, resource_metadata = next(iter(holdout_loader))
    resource_batch = CachedBatch(
        images=resource_images.detach().float().cpu().contiguous().clone(),
        targets=resource_targets.detach().long().cpu().contiguous().clone(),
        bbox=resource_metadata["bbox"].detach().float().cpu().contiguous().clone(),
        image_mask=resource_metadata["image_mask"].detach().bool().cpu().contiguous().clone(),
        sample_indices=resource_metadata["sample_index"].detach().long().cpu().contiguous().clone(),
    )

    mixed_model, mixed_summary, mixed_history = _run_mixed_control(
        prototype=prototype,
        batch_cache=batch_cache,
        args=args,
        device=device,
    )
    final_model, final_summary, final_history, final_phase1 = _run_single_final_average(
        prototype=prototype,
        batch_cache=batch_cache,
        fit_probe_batches=fit_probe_batches,
        args=args,
        device=device,
    )
    recurrent_model, recurrent_summary, recurrent_history = _run_dart_recurrent(
        prototype=prototype,
        batch_cache=batch_cache,
        fit_probe_batches=fit_probe_batches,
        final_only_phase1=final_phase1,
        args=args,
        device=device,
    )
    batch_maps = {
        "mixed_control": _logical_batch_map(mixed_history),
        "single_final_average": _logical_batch_map(final_history),
        "dart_recurrent": _logical_batch_map(recurrent_history),
    }
    batch_replay_exact = bool(
        batch_maps["mixed_control"]
        == batch_maps["single_final_average"]
        == batch_maps["dart_recurrent"]
    )
    del batch_cache, fit_probe_batches
    gc.collect()
    base_dataset.enable_image_cache(max_megabytes=2304, max_items=2000)
    image_cache_before_evaluation = base_dataset.image_cache_stats()

    clean_rows = {
        "mixed_control": _predict_cpu_model(mixed_model, holdout_loader, device),
        "single_final_average": _predict_cpu_model(final_model, holdout_loader, device),
        "dart_recurrent": _predict_cpu_model(recurrent_model, holdout_loader, device),
    }
    clean_mixed = _comparison(
        control_rows=clean_rows["mixed_control"],
        candidate_rows=clean_rows["dart_recurrent"],
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    clean_final = _comparison(
        control_rows=clean_rows["single_final_average"],
        candidate_rows=clean_rows["dart_recurrent"],
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    condition_rows: Dict[str, Dict[str, Sequence[Mapping[str, object]]]] = {
        "clean": clean_rows
    }
    illumination_live: Dict[str, Dict[str, object]] = {}
    illumination_mixed = []
    illumination_loader_summaries: Dict[str, object] = {}
    for condition_index, (condition, brightness, contrast) in enumerate(LIGHTING_CONDITIONS):
        loader, loader_summary = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=eval_transform,
            indices=[row.sample_index for row in holdout],
            brightness=brightness,
            contrast=contrast,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            context=f"dart_a0_{condition}",
            seed=SEED + 300 + condition_index,
        )
        rows_by_variant = {
            "mixed_control": _predict_cpu_model(mixed_model, loader, device),
            "single_final_average": _predict_cpu_model(final_model, loader, device),
            "dart_recurrent": _predict_cpu_model(recurrent_model, loader, device),
        }
        condition_rows[condition] = rows_by_variant
        mixed_comparison = _comparison(
            control_rows=rows_by_variant["mixed_control"],
            candidate_rows=rows_by_variant["dart_recurrent"],
            num_classes=5,
            focus_class=FOCUS_CLASS,
        )
        final_comparison = _comparison(
            control_rows=rows_by_variant["single_final_average"],
            candidate_rows=rows_by_variant["dart_recurrent"],
            num_classes=5,
            focus_class=FOCUS_CLASS,
        )
        illumination_mixed.append(mixed_comparison)
        illumination_live[condition] = {
            "mixed_vs_recurrent": mixed_comparison,
            "final_vs_recurrent": final_comparison,
        }
        illumination_loader_summaries[condition] = loader_summary

    prediction_path = output_dir / "predictions_all_conditions.csv"
    _write_prediction_csv(
        prediction_path,
        source_rows=rows,
        condition_rows=condition_rows,
    )
    live_replay_payload = {
        "clean": {
            "mixed_vs_recurrent": clean_mixed,
            "final_vs_recurrent": clean_final,
        },
        "illumination": illumination_live,
    }
    independent_replay = _replay_prediction_csv(prediction_path)
    replay_error = _maximum_numeric_difference(live_replay_payload, independent_replay)
    image_cache_after_evaluation = base_dataset.image_cache_stats()

    runtime = _benchmark_pair(
        native=prototype,
        candidate=recurrent_model,
        batch=resource_batch,
        device=device,
        repeats=int(args.benchmark_repeats),
    )
    export = _export_candidate(
        candidate=recurrent_model,
        batch=resource_batch,
        output_dir=output_dir,
    )

    all_branch_summaries = [
        *final_summary["branches"],
        *recurrent_summary["branches"],
    ]
    aggregation_events = [
        final_summary["aggregation"],
        *recurrent_summary["cycles"],
    ]
    barriers = [float(event["barrier"]) for event in aggregation_events]
    optimizer_steps_exact = bool(
        all(
            event["optimizer_steps"] == [[8.0], [8.0], [8.0], [8.0]]
            for event in (final_summary["aggregation"], recurrent_summary["cycles"][1])
        )
        and recurrent_summary["cycles"][0]["optimizer_steps"]
        == [[4.0], [4.0], [4.0], [4.0]]
        and mixed_summary["optimizer_steps"] == [32.0]
    )
    all_required_gradients = bool(
        all(bool(mixed_summary["gradient_seen"].get(group, False)) for group in REQUIRED_PARAMETER_GROUPS)
        and all(
            all(bool(branch["gradient_seen"].get(group, False)) for group in REQUIRED_PARAMETER_GROUPS)
            for branch in all_branch_summaries
        )
    )
    all_required_movement = bool(
        all(
            bool(mixed_summary["parameter_group_movement"].get(group, False))
            for group in REQUIRED_PARAMETER_GROUPS
        )
        and all(
            all(
                bool(branch["parameter_group_movement"].get(group, False))
                for group in REQUIRED_PARAMETER_GROUPS
            )
            for branch in all_branch_summaries
        )
    )
    maximum_training_vram = max(
        float(mixed_summary["peak_vram_gib"]),
        float(final_summary["peak_vram_gib"]),
        float(recurrent_summary["peak_vram_gib"]),
    )
    structural_checks = {
        "locked_sources_verified": True,
        "train_only_provenance": not provenance["validation_data_used"]
        and not provenance["test_data_used"],
        "source_cohorts_and_hashes_exact": not any(
            cohort_summary["source_overlap"].values()
        )
        and cohort_summary["update_rows"] == 1024
        and cohort_summary["probe_rows"] == 256,
        "only_current_auditor_process_active": not process_before["unexpected_processes"],
        "all_initial_states_exact": bool(
            mixed_summary["initial_state_sha256"] == prototype_hash
            and all(value == prototype_hash for value in final_summary["initial_state_sha256"])
            and all(value == prototype_hash for value in recurrent_summary["initial_state_sha256"])
        ),
        "equal_update_counts_32": int(mixed_summary["updates"])
        == int(final_summary["updates"])
        == int(recurrent_summary["updates"])
        == 32,
        "batch_bytes_and_logical_rng_exact": batch_replay_exact,
        "final_and_recurrent_phase1_exact": bool(
            recurrent_summary["phase1_matches_final_only_exactly"]
        ),
        "optimizer_steps_exact": optimizer_steps_exact,
        "optimizer_states_preserved_across_broadcast": all(
            bool(event["optimizer_state_preserved"])
            for event in recurrent_summary["cycles"]
        ),
        "optimizer_states_remain_branch_distinct": all(
            bool(event["optimizer_states_distinct"])
            for event in recurrent_summary["cycles"]
        ),
        "nonfloating_buffers_exact": all(
            bool(event["averaging"]["nonfloating_bit_exact"])
            for event in aggregation_events
        ),
        "broadcast_hashes_exact": all(
            bool(event["broadcast_hashes_exact"])
            for event in recurrent_summary["cycles"]
        ),
        "all_barriers_finite_lte_0p02": all(
            math.isfinite(value) and value <= MAX_BARRIER for value in barriers
        ),
        "recurrent_final_barrier_lte_final_only": float(
            recurrent_summary["cycles"][1]["barrier"]
        )
        <= float(final_summary["aggregation"]["barrier"]),
        "all_aggregation_events_have_useful_diversity": all(
            _event_diversity_passed(event) for event in aggregation_events
        ),
        "all_gradients_finite": bool(mixed_summary["all_gradients_finite"])
        and all(bool(branch["all_gradients_finite"]) for branch in all_branch_summaries),
        "all_required_gradient_groups_seen": all_required_gradients,
        "all_required_parameter_groups_moved": all_required_movement,
        "model_schema_unchanged": _state_schema(prototype)
        == _state_schema(mixed_model)
        == _state_schema(final_model)
        == _state_schema(recurrent_model),
        "parameter_count_unchanged": all(
            sum(int(value.numel()) for value in model.parameters()) == parameter_count
            for model in (mixed_model, final_model, recurrent_model)
        ),
        "independent_csv_replay_exact": math.isfinite(replay_error)
        and replay_error <= 1e-12,
        "static_onnx_succeeded": bool(export.get("succeeded", False)),
        "static_onnx_finite": bool(export.get("finite", False)),
        "static_onnx_error_lte_1e5": float(export.get("maximum_absolute_error", 1e9))
        <= MAX_ONNX_ERROR,
        "static_onnx_argmax_match": bool(export.get("argmax_match", False)),
        "inference_outputs_finite": bool(runtime["all_outputs_finite"]),
        "inference_runtime_ratio_lte_1p02": float(runtime["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "inference_memory_ratio_lte_1p02": float(runtime["memory_ratio"])
        <= MAX_INFERENCE_MEMORY_RATIO,
        "sequential_training_peak_lte_4p5_gib": maximum_training_vram
        <= MAX_TRAINING_VRAM_GIB,
    }
    decision = assess_dart_a0(
        structural_checks=structural_checks,
        clean_mixed=clean_mixed,
        clean_final=clean_final,
        illumination_mixed=illumination_mixed,
    )

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "mixed_control": mixed_history,
                "single_final_average": final_history,
                "dart_recurrent": recurrent_history,
                "single_final_aggregation": final_summary["aggregation"],
                "dart_cycles": recurrent_summary["cycles"],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    batch_path = output_dir / "batch_manifest.json"
    batch_path.write_text(
        json.dumps(
            {
                "cohorts": cohort_summary,
                "transform_declarations": transform_declarations,
                "materialization": batch_materialization,
                "logical_replay_maps": batch_maps,
                "logical_replay_exact": batch_replay_exact,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    checkpoint_artifact = None
    if bool(decision["xai_protocol_authorized"]):
        checkpoint_path = output_dir / "dart_recurrent_candidate.pt"
        payload = {
            key: checkpoint[key]
            for key in (
                "model_config",
                "class_names",
                "augmentation_config",
                "data_summary",
                "train_config",
            )
            if key in checkpoint
        }
        payload.update(
            {
                "model_state": recurrent_model.state_dict(),
                "stage": "dart_recurrent_aggregation_a0",
                "validation_data_used": False,
                "test_data_used": False,
            }
        )
        torch.save(payload, checkpoint_path)
        checkpoint_artifact = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": int(checkpoint_path.stat().st_size),
        }

    process_after = _process_snapshot()
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = _gpu_snapshot()
    artifacts: Dict[str, Dict[str, object]] = {
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": _sha256(prediction_path),
            "size_bytes": int(prediction_path.stat().st_size),
        },
        "training_history": {
            "path": str(history_path.resolve()),
            "sha256": _sha256(history_path),
            "size_bytes": int(history_path.stat().st_size),
        },
        "batch_manifest": {
            "path": str(batch_path.resolve()),
            "sha256": _sha256(batch_path),
            "size_bytes": int(batch_path.stat().st_size),
        },
    }
    if checkpoint_artifact is not None:
        artifacts["candidate_checkpoint"] = checkpoint_artifact
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_sha256": LOCKED_PROTOCOL_SHA256,
                "status": (
                    "passed_nonvisual_xai_protocol_required"
                    if decision["xai_protocol_authorized"]
                    else "rejected"
                ),
                "artifacts": artifacts,
                "onnx_retained": False,
                "raw_dataset_files_written": 0,
                "validation_data_used": False,
                "test_data_used": False,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    artifacts["artifact_manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
        "size_bytes": int(manifest_path.stat().st_size),
    }
    summary = {
        "protocol": "dart_recurrent_aggregation_train_only_a0_v1",
        "status": (
            "passed_nonvisual_xai_protocol_required"
            if decision["xai_protocol_authorized"]
            else "rejected"
        ),
        "provenance": provenance,
        "process_gpu": {
            "before": {"process": process_before, "gpu": gpu_before},
            "after": {"process": process_after, "gpu": gpu_after},
        },
        "config": to_serializable(vars(args)),
        "dataset": {
            **cohort_summary,
            "train_rows": len(rows),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fit_probe_loader": fit_probe_loader_summary,
            "fit_probe_cache": fit_probe_cache_summary,
            "holdout_loader": holdout_loader_summary,
            "illumination_loaders": illumination_loader_summaries,
            "image_cache_before_evaluation": image_cache_before_evaluation,
            "image_cache_after_evaluation": image_cache_after_evaluation,
            "validation_data_used": False,
            "test_data_used": False,
        },
        "model": {
            "class_names": class_names,
            "parameter_count": parameter_count,
            "prototype_state_sha256": prototype_hash,
            "mixed_state_sha256": _state_sha256(mixed_model),
            "single_final_state_sha256": _state_sha256(final_model),
            "dart_recurrent_state_sha256": _state_sha256(recurrent_model),
        },
        "training": {
            "mixed_control": mixed_summary,
            "single_final_average": final_summary,
            "dart_recurrent": recurrent_summary,
            "batch_replay_exact": batch_replay_exact,
            "maximum_sequential_peak_vram_gib": maximum_training_vram,
        },
        "clean": live_replay_payload["clean"],
        "illumination": illumination_live,
        "independent_replay": {
            "maximum_absolute_difference": replay_error,
            "exact_within_1e12": replay_error <= 1e-12,
        },
        "runtime": runtime,
        "export": export,
        "decision": decision,
        "candidate_checkpoint": checkpoint_artifact,
        "xai": {
            "ran": False,
            "reason": (
                "separate_hash_locked_xai_protocol_required"
                if decision["xai_protocol_authorized"]
                else "nonvisual_gate_failed"
            ),
        },
        "artifacts": artifacts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.resolve())
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        provenance, rows, branches, probe, holdout, cohort = _load_locked_inputs(args)
        result = {
            "protocol": "dart_recurrent_aggregation_train_only_a0_v1",
            "preflight_only": True,
            "output_dir_created": False,
            "provenance": provenance,
            "dataset": cohort,
            "rows": len(rows),
            "branch_rows": {role: len(value) for role, value in branches.items()},
            "probe_rows": len(probe),
            "holdout_rows": len(holdout),
        }
        print(json.dumps(to_serializable(result), indent=2, ensure_ascii=True))
        return
    result = run_audit(args)
    print(json.dumps(to_serializable(result), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

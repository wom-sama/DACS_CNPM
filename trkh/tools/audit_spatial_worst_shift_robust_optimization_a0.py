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
from pathlib import Path
import random
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import create_model, load_model_state
from trkh.tools.audit_class1_protected_rsc_readiness import (
    _parameter_group,
    _parameter_group_hashes,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
    _classification_metrics,
)
from trkh.tools.audit_dart_recurrent_aggregation_readiness import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_FOLD_COUNTS,
    EXPECTED_PROBE_TOTAL_INDEX_SHA256,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_UPDATE_INDEX_SHA256,
    ROLE_SPECS,
    _benchmark_pair,
    _build_locked_cohorts,
    _export_candidate,
    _maximum_numeric_difference,
    _optimizer_state_sha256,
    _optimizer_steps,
    _optimizer_to,
    _state_schema,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_learnable_polyphase_downsampling_a0 import (
    SHIFT_DIRECTIONS,
    _condition_name,
    _metadata_for_shift,
    _translate_tensor,
    _update_tensor_hash,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _comparison,
    _make_loader,
    _prepare_output_dir,
    _sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
UPDATES = 32
FOCUS_CLASS = 1
RESTRICTED_CLASSES = (0, 2, 4)
TRANSFORMS = ((0, 0), *SHIFT_DIRECTIONS)
EXPECTED_UPDATE_ROWS = 1024
EXPECTED_PROBE_ROWS = 256
EXPECTED_HOLDOUT_ROWS = 1843
EXPECTED_UPDATE_CLASS_COUNTS = (217, 60, 212, 280, 255)
EXPECTED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)

LOCKED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
LOCKED_LAUNCHER_ARGS_SHA256 = (
    "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
)
LOCKED_DATA_SHA256 = (
    "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
)
LOCKED_CIDT_SUMMARY_SHA256 = (
    "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
)
LOCKED_CIDT_PREDICTIONS_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_PROTOCOL_SHA256 = (
    "eec9e20c1dd11100504a647bdb90e09b24f9f553fb5bf3c7319e1d8f93ca0a54"
)
LOCKED_PAPER_SHA256 = (
    "220d573aef26395e8414c9c59d5f91bb97d4be8fc51ef02a031fbf8894f31019"
)
LOCKED_OFFICIAL_COMMIT = "a1c9e364d4179d410209ba3a3b06fdcc73ad2ff9"
LOCKED_OFFICIAL_TREE = "60fe12b9c388b00d00068e07c2e93d134f126cf3"
LOCKED_OFFICIAL_HASHES = {
    "readme": "7d1098457088ab740f4f46c29e096bb84ca2df9fc5056ff0c7f53814a83ba256",
    "spatial_attack": "f663f8628e50b2c89b68b90284559c46e830908adf12410dd4c34d0b831b7b6e",
    "trainer": "9c27b7cfb9bbfb7472fb3b8d03e217a8d9d480c90c42f44bd3007cb38aa7aeac",
    "license": "9787e942613613893ea455e11094c625e902a4a31dd549a408b57158684b0925",
}
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)

MAX_TRAINING_VRAM_GIB = 4.5
MAX_TRAINING_WALL_RATIO = 1.10
MAX_RUNTIME_RATIO = 1.02
MAX_MEMORY_RATIO = 1.02
MAX_ONNX_ERROR = 1e-5
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
METHOD = "spatial_worst_shift_robust_optimization_a0"
VARIANTS = ("keeper", "random_control", "worst_shift")
PHOTOMETRIC_CONDITIONS = ("clean", *(row[0] for row in LIGHTING_CONDITIONS))


def _progress(stage: str, **details: object) -> None:
    print(
        json.dumps(
            {"spatial_a0_stage": stage, **details},
            sort_keys=True,
            ensure_ascii=True,
        ),
        flush=True,
    )


@dataclass(frozen=True)
class SpatialBatch:
    images: Tensor
    targets: Tensor
    bbox: Tensor
    crop_bbox: Tensor
    image_mask: Tensor
    sample_indices: Tensor

    def metadata(self) -> Dict[str, Tensor]:
        return {
            "bbox": self.bbox,
            "crop_bbox": self.crop_bbox,
            "image_mask": self.image_mask,
            "sample_index": self.sample_indices,
        }


@dataclass(frozen=True)
class GridOutput:
    losses: Tensor
    predictions: Tensor
    probabilities: Tensor


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only spatial worst-shift robust-optimization A0. "
            "Validation and test construction are forbidden."
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
            "docs/TRKH_5CLASS_SPATIAL_WORST_SHIFT_ROBUST_OPTIMIZATION_"
            "A0_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\Engstrom_Exploring_Spatial_Robustness_ICML2019.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\adversarial_spatial"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/audit_spatial_worst_shift_robust_optimization_a0_20260717"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--updates", type=int, default=UPDATES)
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
        and int(args.updates) == UPDATES
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
        / "test_audit_spatial_worst_shift_robust_optimization_a0.py",
        "launcher": root
        / "scripts"
        / "run_trkh_spatial_worst_shift_robust_optimization_a0.ps1",
        "protocol": root
        / "docs"
        / "TRKH_5CLASS_SPATIAL_WORST_SHIFT_ROBUST_OPTIMIZATION_"
        "A0_PROTOCOL_20260717.md",
    }
    relative_paths: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Spatial A0 {name} is missing: {path}")
        relative = path.relative_to(root).as_posix()
        _git_value(root, "ls-files", "--error-unmatch", relative)
        relative_paths[name] = relative
    status = _git_value(root, "status", "--porcelain", "--", *relative_paths.values())
    if status:
        raise ValueError(f"Spatial A0 implementation is not commit-clean: {status}")
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
        "readme": official / "README.md",
        "spatial_attack": official / "spatial_attack.py",
        "trainer": official / "train.py",
        "license": official / "LICENSE",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _flatten_update_rows(
    branches: Mapping[str, Sequence[CleanTrainRow]],
) -> list[CleanTrainRow]:
    rows: list[CleanTrainRow] = []
    for role, _ in ROLE_SPECS:
        rows.extend(branches[role])
    return rows


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[
    Dict[str, object],
    list[CleanTrainRow],
    list[CleanTrainRow],
    list[CleanTrainRow],
    list[CleanTrainRow],
    Dict[str, object],
]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the locked spatial A0 protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "spatial A0 protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "ICML spatial paper"
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
            paths[name], expected, f"official spatial source {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(paths["official_root"], "rev-parse", "HEAD^{tree}")
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(f"Official spatial commit differs: {official_commit}")
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official spatial tree differs: {official_tree}")
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official spatial source worktree must be clean.")
    source_text = paths["spatial_attack"].read_text(encoding="utf-8")
    trainer_text = paths["trainer"].read_text(encoding="utf-8")
    official_rule_present = bool(
        "idx = (cur_xent > max_xent) & (cur_correct == all_correct)" in source_text
        and "idx = idx | (cur_correct < all_correct)" in source_text
        and "worst_t = np.where(idx, t, worst_t)" in source_text
        and "x_batch_adv, adv_trans = attack.perturb" in trainer_text
        and "sess.run(train_step, feed_dict=adv_dict)" in trainer_text
    )
    if not official_rule_present:
        raise ValueError("Locked official spatial selection/training source differs.")

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
    update = _flatten_update_rows(branches)
    if len(update) != EXPECTED_UPDATE_ROWS:
        raise ValueError(f"Locked update-row count differs: {len(update)}")
    if tuple(Counter(row.target for row in update).get(i, 0) for i in range(5)) != (
        EXPECTED_UPDATE_CLASS_COUNTS
    ):
        raise ValueError("Locked update class counts differ.")
    if len(probe) != EXPECTED_PROBE_ROWS or len(holdout) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("Locked probe/holdout count differs.")
    if cohort_summary["update_ordered_sample_index_sha256"] != EXPECTED_UPDATE_INDEX_SHA256:
        raise ValueError("Locked update index hash differs.")
    if cohort_summary["probe_ordered_sample_index_sha256"] != EXPECTED_PROBE_TOTAL_INDEX_SHA256:
        raise ValueError("Locked probe index hash differs.")
    if cohort_summary["holdout_ordered_sample_index_sha256"] != EXPECTED_HOLDOUT_INDEX_SHA256:
        raise ValueError("Locked holdout index hash differs.")
    if any(cohort_summary["source_overlap"].values()):
        raise ValueError("Locked spatial cohorts have source overlap.")

    repo_root = Path(__file__).resolve().parents[2]
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for spatial A0.")
    implementation = _verify_committed_implementation(repo_root)
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal spatial A0 requires the pushed repository commit.")
    provenance = {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "official_commit": official_commit,
        "official_tree": official_tree,
        "official_worktree_clean": True,
        "official_rule_present": official_rule_present,
        "repository_commit": repository_commit,
        "upstream_commit": upstream_commit,
        "tracked_worktree_clean": True,
        "implementation_committed_clean": True,
        "implementation": implementation,
        "validation_data_used": False,
        "test_data_used": False,
    }
    return provenance, rows, update, probe, holdout, cohort_summary


def _batch_hash(batch: SpatialBatch) -> Dict[str, object]:
    digest = hashlib.sha256()
    component_hashes: Dict[str, str] = {}
    for name, value in (
        ("images", batch.images),
        ("targets", batch.targets),
        ("bbox", batch.bbox),
        ("crop_bbox", batch.crop_bbox),
        ("image_mask", batch.image_mask),
        ("sample_indices", batch.sample_indices),
    ):
        component = hashlib.sha256()
        _update_tensor_hash(component, name, value)
        component_hashes[name] = component.hexdigest()
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(component_hashes[name]))
    return {
        "rows": int(batch.targets.numel()),
        "sample_indices": [int(value) for value in batch.sample_indices.tolist()],
        "components": component_hashes,
        "combined_sha256": digest.hexdigest(),
    }


def _materialize_batches(
    loader: DataLoader,
    *,
    expected_indices: Sequence[int],
) -> tuple[list[SpatialBatch], Dict[str, object]]:
    batches: list[SpatialBatch] = []
    observed: list[int] = []
    manifests = []
    for images, targets, metadata in loader:
        if not isinstance(metadata, Mapping):
            raise ValueError("Spatial loader is missing metadata.")
        required = {
            name: metadata.get(name)
            for name in ("bbox", "crop_bbox", "image_mask", "sample_index")
        }
        if not all(torch.is_tensor(value) for value in required.values()):
            raise ValueError("Spatial loader metadata is incomplete.")
        batch = SpatialBatch(
            images=images.detach().float().cpu().contiguous().clone(),
            targets=targets.detach().long().cpu().contiguous().clone(),
            bbox=required["bbox"].detach().float().cpu().contiguous().clone(),
            crop_bbox=required["crop_bbox"].detach().float().cpu().contiguous().clone(),
            image_mask=required["image_mask"].detach().bool().cpu().contiguous().clone(),
            sample_indices=required["sample_index"]
            .detach()
            .long()
            .cpu()
            .contiguous()
            .clone(),
        )
        manifests.append(_batch_hash(batch))
        observed.extend(int(value) for value in batch.sample_indices.tolist())
        batches.append(batch)
    expected = [int(value) for value in expected_indices]
    if observed != expected:
        raise ValueError("Spatial materialized sample order differs from locked cohort.")
    combined = hashlib.sha256(
        "\n".join(str(row["combined_sha256"]) for row in manifests).encode("ascii")
    ).hexdigest()
    return batches, {
        "rows": len(observed),
        "batches": len(batches),
        "ordered_sample_index_sha256": hashlib.sha256(
            "\n".join(str(value) for value in observed).encode("ascii")
        ).hexdigest(),
        "batch_manifest_sha256": combined,
        "batch_manifests": manifests,
    }


def _shift_batch(batch: SpatialBatch, condition_index: int) -> SpatialBatch:
    if not 0 <= int(condition_index) < len(TRANSFORMS):
        raise ValueError(f"Invalid spatial condition index: {condition_index}")
    dx, dy = TRANSFORMS[int(condition_index)]
    if dx == 0 and dy == 0:
        return batch
    metadata = _metadata_for_shift(batch.metadata(), dx=dx, dy=dy)
    return SpatialBatch(
        images=_translate_tensor(batch.images, dx=dx, dy=dy, replicate=True),
        targets=batch.targets,
        bbox=metadata["bbox"],
        crop_bbox=metadata["crop_bbox"],
        image_mask=metadata["image_mask"],
        sample_indices=metadata["sample_index"],
    )


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> Tensor:
    logits, _ = _forward_classification_with_metadata(
        model,
        images,
        metadata,
        device=device,
    )
    return logits


def _grid_forward(
    model: nn.Module,
    batch: SpatialBatch,
    *,
    device: torch.device,
) -> GridOutput:
    model.eval()
    losses = []
    predictions = []
    probabilities = []
    targets = batch.targets.to(device=device, dtype=torch.long, non_blocking=True)
    with torch.inference_mode():
        for condition_index in range(len(TRANSFORMS)):
            shifted = _shift_batch(batch, condition_index)
            images = shifted.images.to(device=device, non_blocking=True)
            logits = _forward_logits(model, images, shifted.metadata(), device=device).float()
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Non-finite spatial grid logits.")
            probs = logits.softmax(dim=1)
            losses.append(F.cross_entropy(logits, targets, reduction="none").cpu())
            predictions.append(probs.argmax(dim=1).cpu())
            probabilities.append(probs.cpu())
            del shifted, images, logits, probs
    return GridOutput(
        losses=torch.stack(losses, dim=0).contiguous(),
        predictions=torch.stack(predictions, dim=0).contiguous(),
        probabilities=torch.stack(probabilities, dim=0).contiguous(),
    )


def official_worst_indices(
    losses: Tensor,
    predictions: Tensor,
    targets: Tensor,
) -> Tensor:
    if losses.ndim != 2 or tuple(predictions.shape) != tuple(losses.shape):
        raise ValueError("Worst-shift selection expects matching [T,B] tensors.")
    if int(losses.size(0)) != len(TRANSFORMS) or targets.ndim != 1:
        raise ValueError("Worst-shift selection shape differs from protocol.")
    if int(losses.size(1)) != int(targets.numel()):
        raise ValueError("Worst-shift batch size differs from targets.")
    if not bool(torch.isfinite(losses).all().item()):
        raise ValueError("Worst-shift losses must be finite.")
    correct = predictions.eq(targets.view(1, -1))
    any_wrong = (~correct).any(dim=0)
    candidate_losses = losses.clone()
    candidate_losses[:, any_wrong] = candidate_losses[:, any_wrong].masked_fill(
        correct[:, any_wrong], -torch.inf
    )
    return candidate_losses.argmax(dim=0)


def official_worst_indices_reference(
    losses: Tensor,
    predictions: Tensor,
    targets: Tensor,
) -> Tensor:
    output = []
    for column in range(int(targets.numel())):
        target = int(targets[column].item())
        wrong = [
            index
            for index in range(int(losses.size(0)))
            if int(predictions[index, column].item()) != target
        ]
        candidates = wrong if wrong else list(range(int(losses.size(0))))
        selected = max(candidates, key=lambda index: float(losses[index, column].item()))
        output.append(selected)
    return torch.tensor(output, dtype=torch.long)


def random_shift_indices(
    sample_indices: Tensor,
    *,
    step: int,
    seed: int = SEED,
) -> Tensor:
    output = []
    for sample_index in sample_indices.tolist():
        digest = hashlib.sha256(
            f"spatial-a0|{int(seed)}|step={int(step)}|sample={int(sample_index)}".encode(
                "ascii"
            )
        ).digest()
        output.append(int.from_bytes(digest[:8], "big") % len(TRANSFORMS))
    return torch.tensor(output, dtype=torch.long)


def _gather_selected_batch(batch: SpatialBatch, indices: Tensor) -> SpatialBatch:
    if indices.ndim != 1 or int(indices.numel()) != int(batch.targets.numel()):
        raise ValueError("Selected shift indices differ from batch size.")
    selected = SpatialBatch(
        images=batch.images.clone(),
        targets=batch.targets.clone(),
        bbox=batch.bbox.clone(),
        crop_bbox=batch.crop_bbox.clone(),
        image_mask=batch.image_mask.clone(),
        sample_indices=batch.sample_indices.clone(),
    )
    images = selected.images
    bbox = selected.bbox
    crop_bbox = selected.crop_bbox
    image_mask = selected.image_mask
    for condition_index in range(1, len(TRANSFORMS)):
        mask = indices.eq(condition_index)
        if not bool(mask.any().item()):
            continue
        shifted = _shift_batch(batch, condition_index)
        images[mask] = shifted.images[mask]
        bbox[mask] = shifted.bbox[mask]
        crop_bbox[mask] = shifted.crop_bbox[mask]
        image_mask[mask] = shifted.image_mask[mask]
    return SpatialBatch(
        images=images.contiguous(),
        targets=selected.targets,
        bbox=bbox.contiguous(),
        crop_bbox=crop_bbox.contiguous(),
        image_mask=image_mask.contiguous(),
        sample_indices=selected.sample_indices,
    )


def _selection_rows(
    *,
    batch: SpatialBatch,
    grid: GridOutput,
    official_indices: Tensor,
    random_indices: Tensor,
    selected_indices: Tensor,
    variant: str,
    step: int,
) -> list[Dict[str, object]]:
    rows = []
    for column in range(int(batch.targets.numel())):
        official = int(official_indices[column].item())
        random_index = int(random_indices[column].item())
        selected = int(selected_indices[column].item())
        dx, dy = TRANSFORMS[selected]
        row: Dict[str, object] = {
            "variant": variant,
            "step": int(step),
            "sample_index": int(batch.sample_indices[column].item()),
            "target": int(batch.targets[column].item()),
            "official_index": official,
            "random_index": random_index,
            "selected_index": selected,
            "selected_dx": int(dx),
            "selected_dy": int(dy),
            "selected_loss": float(grid.losses[selected, column].item()),
            "selected_prediction": int(grid.predictions[selected, column].item()),
            "any_misclassification": bool(
                grid.predictions[:, column].ne(batch.targets[column]).any().item()
            ),
            "target_loss_span": float(
                grid.losses[:, column].max().item()
                - grid.losses[:, column].min().item()
            ),
        }
        for condition_index in range(len(TRANSFORMS)):
            row[f"loss_{condition_index}"] = float(
                grid.losses[condition_index, column].item()
            )
            row[f"pred_{condition_index}"] = int(
                grid.predictions[condition_index, column].item()
            )
        rows.append(row)
    return rows


def assess_probe_signal(
    rows: Sequence[Mapping[str, object]],
    *,
    output_replay_exact: bool = True,
    transform_replay_exact: bool = True,
) -> Dict[str, object]:
    if len(rows) != EXPECTED_PROBE_ROWS:
        raise ValueError(f"Probe selection row count differs: {len(rows)}")
    selected = Counter(int(row["official_index"]) for row in rows)
    nonclean = sum(int(row["official_index"]) != 0 for row in rows)
    class1_tp_signal = sum(
        int(row["target"]) == FOCUS_CLASS
        and int(row["pred_0"]) == FOCUS_CLASS
        and float(row["target_loss_span"]) > 1e-6
        for row in rows
    )
    restricted_fp_signal = sum(
        int(row["target"]) in RESTRICTED_CLASSES
        and int(row["pred_0"]) == FOCUS_CLASS
        and float(row["target_loss_span"]) > 1e-6
        for row in rows
    )
    checks = {
        "all_probe_rows_present": len(rows) == EXPECTED_PROBE_ROWS,
        "all_outputs_finite": all(
            math.isfinite(float(row[f"loss_{index}"]))
            for row in rows
            for index in range(len(TRANSFORMS))
        ),
        "output_replay_exact": bool(output_replay_exact),
        "transform_replay_exact": bool(transform_replay_exact),
        "at_least_six_conditions_selected": len(selected) >= 6,
        "nonclean_selection_fraction_gte_0p05": nonclean / len(rows) >= 0.05,
        "class1_tp_loss_span_present": class1_tp_signal >= 1,
        "restricted_fp_loss_span_present": restricted_fp_signal >= 1,
        "official_oracle_parity_exact": all(
            int(row["official_index"])
            == int(
                official_worst_indices_reference(
                    torch.tensor(
                        [[float(row[f"loss_{index}"])] for index in range(len(TRANSFORMS))]
                    ),
                    torch.tensor(
                        [[int(row[f"pred_{index}"])] for index in range(len(TRANSFORMS))]
                    ),
                    torch.tensor([int(row["target"])]),
                )[0].item()
            )
            for row in rows
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "selected_condition_counts": {
            str(index): int(selected.get(index, 0)) for index in range(len(TRANSFORMS))
        },
        "nonclean_selected": int(nonclean),
        "nonclean_selection_fraction": float(nonclean / len(rows)),
        "class1_tp_loss_span_rows": int(class1_tp_signal),
        "restricted_fp_loss_span_rows": int(restricted_fp_signal),
    }


def _logical_seed(step: int) -> int:
    digest = hashlib.sha256(
        f"spatial-a0|{SEED}|gradient-step={int(step)}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _set_logical_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


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
        "initial_state_sha256": _state_sha256(model),
        "initial_group_hashes": _parameter_group_hashes(model),
        "gradient_seen": {group: False for group in groups},
        "all_gradients_finite": True,
        "peak_vram_gib": 0.0,
    }


def _finalize_tracker(model: nn.Module, tracker: Mapping[str, object]) -> Dict[str, object]:
    before = tracker["initial_group_hashes"]
    assert isinstance(before, Mapping)
    after = _parameter_group_hashes(model)
    return {
        **tracker,
        "final_state_sha256": _state_sha256(model),
        "final_group_hashes": after,
        "parameter_group_movement": {
            name: str(before.get(name)) != str(after.get(name))
            for name in sorted(set(before).union(after))
        },
    }


def _train_variant(
    *,
    prototype: nn.Module,
    batches: Sequence[SpatialBatch],
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    if variant not in {"random_control", "worst_shift"}:
        raise ValueError(f"Unknown spatial training variant: {variant}")
    if len(batches) != UPDATES:
        raise ValueError(f"Spatial training requires {UPDATES} batches.")
    model = copy.deepcopy(prototype).to(device).train()
    optimizer = _new_optimizer(model, args)
    _optimizer_to(optimizer, device)
    tracker = _new_tracker(model)
    history: list[Dict[str, object]] = []
    selection_rows: list[Dict[str, object]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for step, batch in enumerate(batches):
        grid = _grid_forward(model, batch, device=device)
        official = official_worst_indices(
            grid.losses, grid.predictions, batch.targets
        )
        reference = official_worst_indices_reference(
            grid.losses, grid.predictions, batch.targets
        )
        if not torch.equal(official, reference):
            raise ValueError(f"Official spatial selector mismatch at step {step}.")
        random_indices = random_shift_indices(batch.sample_indices, step=step)
        selected_indices = official if variant == "worst_shift" else random_indices
        selected_batch = _gather_selected_batch(batch, selected_indices)
        selection_rows.extend(
            _selection_rows(
                batch=batch,
                grid=grid,
                official_indices=official,
                random_indices=random_indices,
                selected_indices=selected_indices,
                variant=variant,
                step=step,
            )
        )

        logical_seed = _logical_seed(step)
        _set_logical_rng(logical_seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images = selected_batch.images.to(device=device, non_blocking=True)
        targets = selected_batch.targets.to(
            device=device, dtype=torch.long, non_blocking=True
        )
        logits = _forward_logits(
            model, images, selected_batch.metadata(), device=device
        ).float()
        loss = F.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"Non-finite spatial loss in {variant}/{step}.")
        loss.backward()
        group_norm_squared: Dict[str, float] = defaultdict(float)
        gradient_seen = tracker["gradient_seen"]
        assert isinstance(gradient_seen, dict)
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
                "variant": variant,
                "step": int(step),
                "logical_seed": int(logical_seed),
                "batch_sha256": _batch_hash(batch)["combined_sha256"],
                "selected_batch_sha256": _batch_hash(selected_batch)[
                    "combined_sha256"
                ],
                "selection_forwards": len(TRANSFORMS),
                "gradient_forwards": 1,
                "loss": float(loss.detach().item()),
                "official_nonclean": int(official.ne(0).sum().item()),
                "selected_nonclean": int(selected_indices.ne(0).sum().item()),
                "official_random_differences": int(
                    official.ne(random_indices).sum().item()
                ),
                "gradient_norms": {
                    name: math.sqrt(value)
                    for name, value in sorted(group_norm_squared.items())
                },
            }
        )
        if (step + 1) % 8 == 0:
            _progress(
                "training",
                variant=variant,
                updates_completed=step + 1,
                updates_total=UPDATES,
                loss=history[-1]["loss"],
                peak_vram_gib=float(
                    torch.cuda.max_memory_allocated(device) / (1024.0**3)
                ),
            )
        del grid, official, reference, random_indices, selected_indices
        del selected_batch, images, targets, logits, loss

    torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    tracker["peak_vram_gib"] = float(
        torch.cuda.max_memory_allocated(device) / (1024.0**3)
    )
    tracker["updates"] = UPDATES
    tracker["selection_forwards"] = UPDATES * len(TRANSFORMS)
    tracker["gradient_forwards"] = UPDATES
    tracker["total_forwards"] = UPDATES * (len(TRANSFORMS) + 1)
    tracker["wall_seconds"] = elapsed
    tracker["optimizer_steps"] = _optimizer_steps(optimizer)
    tracker["optimizer_state_sha256"] = _optimizer_state_sha256(model, optimizer)
    model.cpu().eval()
    _optimizer_to(optimizer, torch.device("cpu"))
    summary = _finalize_tracker(model, tracker)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return model, summary, history, selection_rows


def equation_diagnostics() -> Dict[str, object]:
    image = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    batch = SpatialBatch(
        images=image,
        targets=torch.tensor([3], dtype=torch.long),
        bbox=torch.tensor([[0.5, 0.5, 0.4, 0.6]], dtype=torch.float32),
        crop_bbox=torch.tensor([[0.4, 0.6, 0.3, 0.5]], dtype=torch.float32),
        image_mask=mask,
        sample_indices=torch.tensor([17], dtype=torch.long),
    )
    identity = _shift_batch(batch, 0)
    right = _shift_batch(batch, TRANSFORMS.index((1, 0)))
    expected_right = torch.tensor(
        [[[[0.0, 0.0, 1.0], [3.0, 3.0, 4.0], [6.0, 6.0, 7.0]]]]
    )
    expected_mask = torch.tensor(
        [[[False, True, True], [False, True, True], [False, True, True]]]
    )

    losses = torch.tensor(
        [
            [0.8, 0.4, 1.0],
            [2.0, 0.4, 1.0],
            [1.5, 0.4, 0.5],
            [1.0, 0.4, 0.5],
            [0.9, 0.4, 0.5],
            [0.7, 0.4, 0.5],
            [0.6, 0.4, 0.5],
            [0.5, 0.4, 0.5],
            [0.4, 0.4, 0.5],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([1, 2, 3], dtype=torch.long)
    predictions = targets.view(1, -1).repeat(len(TRANSFORMS), 1)
    predictions[2, 0] = 4
    predictions[3, 0] = 0
    observed = official_worst_indices(losses, predictions, targets)
    reference = official_worst_indices_reference(losses, predictions, targets)
    random_source = torch.tensor([11, 12, 13, 14], dtype=torch.long)
    random_forward = random_shift_indices(random_source, step=7)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    random_permuted = random_shift_indices(random_source[permutation], step=7)
    checks = {
        "identity_images_bit_exact": torch.equal(identity.images, batch.images),
        "identity_metadata_bit_exact": bool(
            torch.equal(identity.bbox, batch.bbox)
            and torch.equal(identity.crop_bbox, batch.crop_bbox)
            and torch.equal(identity.image_mask, batch.image_mask)
        ),
        "right_translation_direction_exact": torch.equal(
            right.images, expected_right
        ),
        "right_mask_false_padding_exact": torch.equal(
            right.image_mask, expected_mask
        ),
        "bbox_direction_exact": bool(
            torch.allclose(
                right.bbox[:, :2],
                batch.bbox[:, :2] + torch.tensor([[1.0 / 256.0, 0.0]]),
                atol=0.0,
                rtol=0.0,
            )
            and torch.equal(right.bbox[:, 2:], batch.bbox[:, 2:])
            and torch.allclose(
                right.crop_bbox[:, :2],
                batch.crop_bbox[:, :2] + torch.tensor([[1.0 / 256.0, 0.0]]),
                atol=0.0,
                rtol=0.0,
            )
        ),
        "targets_and_indices_invariant": bool(
            torch.equal(right.targets, batch.targets)
            and torch.equal(right.sample_indices, batch.sample_indices)
        ),
        "official_reference_exact": torch.equal(observed, reference),
        "misclassification_priority_exact": int(observed[0].item()) == 2,
        "all_correct_tie_keeps_first": int(observed[1].item()) == 0,
        "all_correct_largest_ce_exact": int(observed[2].item()) == 0,
        "random_selection_repeat_exact": torch.equal(
            random_forward, random_shift_indices(random_source, step=7)
        ),
        "random_selection_order_independent": torch.equal(
            random_forward[permutation], random_permuted
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "official_indices": observed.tolist(),
        "random_indices": random_forward.tolist(),
    }


def _process_snapshot() -> Dict[str, object]:
    current_pid = os.getpid()
    owned = {current_pid}
    try:
        current = psutil.Process(current_pid)
        owned.update(parent.pid for parent in current.parents())
        owned.update(child.pid for child in current.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    rows = []
    for process in psutil.process_iter(
        attrs=("pid", "name", "create_time", "cmdline")
    ):
        try:
            name = str(process.info.get("name") or "").casefold()
            if name not in {"python.exe", "pythonw.exe", "trtexec.exe"}:
                continue
            pid = int(process.info["pid"])
            rows.append(
                {
                    "pid": pid,
                    "name": name,
                    "create_time": float(process.info.get("create_time") or 0.0),
                    "command_line": " ".join(process.info.get("cmdline") or []),
                    "current_auditor": pid == current_pid,
                    "current_auditor_owned": pid in owned,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "current_pid": current_pid,
        "owned_pids": sorted(owned),
        "processes": rows,
        "unexpected_processes": [
            row for row in rows if not row["current_auditor_owned"]
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


def _metrics_with_focus(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    metrics = _classification_metrics(targets, predictions, num_classes=5)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    true_positive = int(confusion[FOCUS_CLASS, FOCUS_CLASS])
    false_positive = int(confusion[:, FOCUS_CLASS].sum() - true_positive)
    false_negative = int(confusion[FOCUS_CLASS, :].sum() - true_positive)
    restricted_fp = int(confusion[list(RESTRICTED_CLASSES), FOCUS_CLASS].sum())
    return {
        **metrics,
        "class1": {
            "precision": float(metrics["per_class_precision"][FOCUS_CLASS]),
            "recall": float(metrics["per_class_recall"][FOCUS_CLASS]),
            "f1": float(metrics["per_class_f1"][FOCUS_CLASS]),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "restricted_fp": restricted_fp,
        },
    }


def _evaluate_grid_model(
    model: nn.Module,
    batches: Sequence[SpatialBatch],
    *,
    expected_indices: Sequence[int],
    device: torch.device,
) -> Dict[str, Tensor]:
    model.to(device).eval()
    losses = []
    predictions = []
    probabilities = []
    targets = []
    indices = []
    for batch in batches:
        grid = _grid_forward(model, batch, device=device)
        losses.append(grid.losses)
        predictions.append(grid.predictions)
        probabilities.append(grid.probabilities)
        targets.append(batch.targets)
        indices.append(batch.sample_indices)
    output = {
        "losses": torch.cat(losses, dim=1).contiguous(),
        "predictions": torch.cat(predictions, dim=1).contiguous(),
        "probabilities": torch.cat(probabilities, dim=1).contiguous(),
        "targets": torch.cat(targets, dim=0).contiguous(),
        "sample_indices": torch.cat(indices, dim=0).contiguous(),
    }
    observed = [int(value) for value in output["sample_indices"].tolist()]
    if observed != [int(value) for value in expected_indices]:
        raise ValueError("Spatial evaluation sample order differs from holdout.")
    output["official_indices"] = official_worst_indices(
        output["losses"], output["predictions"], output["targets"]
    )
    output["mean_predictions"] = output["probabilities"].mean(dim=0).argmax(dim=1)
    model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _mode_predictions(evaluation: Mapping[str, Tensor], mode: str) -> Tensor:
    predictions = evaluation["predictions"]
    if mode == "clean":
        return predictions[0]
    if mode == "official_worst":
        columns = torch.arange(int(predictions.size(1)), dtype=torch.long)
        return predictions[evaluation["official_indices"], columns]
    if mode == "mean_probability_diagnostic":
        return evaluation["mean_predictions"]
    raise ValueError(f"Unknown spatial evaluation mode: {mode}")


def _summarize_evaluation(evaluation: Mapping[str, Tensor]) -> Dict[str, object]:
    targets = evaluation["targets"].numpy()
    predictions = evaluation["predictions"]
    official = evaluation["official_indices"]
    clean = predictions[0]
    all_equal = predictions.eq(clean.unsqueeze(0)).all(dim=0)
    pairwise_clean_agreement = predictions.eq(clean.unsqueeze(0)).float().mean()
    clean_class1 = clean.eq(FOCUS_CLASS)
    any_class1 = predictions.eq(FOCUS_CLASS).any(dim=0)
    any_nonclass1 = predictions.ne(FOCUS_CLASS).any(dim=0)
    target_tensor = evaluation["targets"]
    restricted = torch.zeros_like(target_tensor, dtype=torch.bool)
    for class_index in RESTRICTED_CLASSES:
        restricted |= target_tensor.eq(class_index)
    by_shift = {}
    for index, (dx, dy) in enumerate(TRANSFORMS):
        by_shift[_condition_name(dx, dy)] = {
            "index": index,
            "dx": dx,
            "dy": dy,
            "metrics": _metrics_with_focus(targets, predictions[index].numpy()),
        }
    return {
        "rows": int(target_tensor.numel()),
        "clean": _metrics_with_focus(targets, clean.numpy()),
        "official_worst": _metrics_with_focus(
            targets, _mode_predictions(evaluation, "official_worst").numpy()
        ),
        "mean_probability_diagnostic": _metrics_with_focus(
            targets,
            _mode_predictions(evaluation, "mean_probability_diagnostic").numpy(),
        ),
        "by_shift": by_shift,
        "official_selection_counts": {
            str(index): int(official.eq(index).sum().item())
            for index in range(len(TRANSFORMS))
        },
        "shift_consistency": {
            "all_nine_same_rate": float(all_equal.float().mean().item()),
            "clean_pairwise_agreement_rate": float(pairwise_clean_agreement.item()),
            "class1_entries": int((~clean_class1 & any_class1).sum().item()),
            "class1_exits": int((clean_class1 & any_nonclass1).sum().item()),
            "class1_tp_exits": int(
                (target_tensor.eq(FOCUS_CLASS) & clean_class1 & any_nonclass1)
                .sum()
                .item()
            ),
            "restricted_fp_entries": int(
                (restricted & ~clean_class1 & any_class1).sum().item()
            ),
            "restricted_fp_exits": int(
                (restricted & clean_class1 & any_nonclass1).sum().item()
            ),
        },
        "all_numeric_finite": bool(
            torch.isfinite(evaluation["losses"]).all().item()
            and torch.isfinite(evaluation["probabilities"]).all().item()
        ),
    }


def _comparison_for_mode(
    control: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    *,
    mode: str,
) -> Dict[str, object]:
    if not torch.equal(control["targets"], candidate["targets"]):
        raise ValueError("Spatial comparison targets differ.")
    if not torch.equal(control["sample_indices"], candidate["sample_indices"]):
        raise ValueError("Spatial comparison sample order differs.")
    targets = control["targets"].numpy()
    control_predictions = _mode_predictions(control, mode).numpy()
    candidate_predictions = _mode_predictions(candidate, mode).numpy()
    control_rows = [
        {"target": int(target), "prediction": int(prediction)}
        for target, prediction in zip(targets, control_predictions)
    ]
    candidate_rows = [
        {"target": int(target), "prediction": int(prediction)}
        for target, prediction in zip(targets, candidate_predictions)
    ]
    base = _comparison(
        control_rows=control_rows,
        candidate_rows=candidate_rows,
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    restricted = np.isin(targets, np.asarray(RESTRICTED_CLASSES, dtype=np.int64))
    return {
        **base,
        "restricted_fp_removed": int(
            (restricted & (control_predictions == FOCUS_CLASS)
             & (candidate_predictions != FOCUS_CLASS)).sum()
        ),
        "restricted_fp_corrected_to_target": int(
            (restricted & (control_predictions == FOCUS_CLASS)
             & (candidate_predictions == targets)).sum()
        ),
        "broken_control_class1_tp": int(
            ((targets == FOCUS_CLASS) & (control_predictions == FOCUS_CLASS)
             & (candidate_predictions != FOCUS_CLASS)).sum()
        ),
    }


def _build_evidence(
    evaluations: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
) -> Dict[str, object]:
    evidence: Dict[str, object] = {}
    for photometric in PHOTOMETRIC_CONDITIONS:
        by_variant = evaluations[photometric]
        evidence[photometric] = {}
        for mode in ("clean", "official_worst"):
            evidence[photometric][mode] = {
                "keeper_vs_candidate": _comparison_for_mode(
                    by_variant["keeper"], by_variant["worst_shift"], mode=mode
                ),
                "random_vs_candidate": _comparison_for_mode(
                    by_variant["random_control"],
                    by_variant["worst_shift"],
                    mode=mode,
                ),
            }
    return evidence


def assess_spatial_a0(
    *,
    structural_checks: Mapping[str, bool],
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
    evidence: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
) -> Dict[str, object]:
    def metrics(photo: str, variant: str, mode: str) -> Mapping[str, object]:
        value = summaries[photo][variant][mode]
        assert isinstance(value, Mapping)
        return value

    clean_refs = [
        metrics("clean", variant, "clean")
        for variant in ("keeper", "random_control")
    ]
    clean_candidate = metrics("clean", "worst_shift", "clean")
    clean_pair = evidence["clean"]["clean"]["keeper_vs_candidate"]
    clean_checks = {
        "macro_f1_gte_best_reference_plus_0p001": float(
            clean_candidate["macro_f1"]
        )
        >= max(float(value["macro_f1"]) for value in clean_refs) + 0.001,
        "class1_f1_gte_best_reference_plus_0p005": float(
            clean_candidate["class1"]["f1"]
        )
        >= max(float(value["class1"]["f1"]) for value in clean_refs) + 0.005,
        "class1_precision_gte_best_reference_plus_0p010": float(
            clean_candidate["class1"]["precision"]
        )
        >= max(float(value["class1"]["precision"]) for value in clean_refs)
        + 0.010,
        "restricted_fp_at_least_2_below_both_references": int(
            clean_candidate["class1"]["restricted_fp"]
        )
        <= min(int(value["class1"]["restricted_fp"]) for value in clean_refs) - 2,
        "class1_tp_within_1_of_larger_reference": int(
            clean_candidate["class1"]["tp"]
        )
        >= max(int(value["class1"]["tp"]) for value in clean_refs) - 1,
        "corrections_exceed_harms": int(
            clean_pair["transitions"]["candidate_correction"]
        )
        > int(clean_pair["transitions"]["candidate_harm"]),
        "restricted_fp_corrections_exceed_broken_keeper_tp": int(
            clean_pair["restricted_fp_corrected_to_target"]
        )
        > int(clean_pair["broken_control_class1_tp"]),
    }

    worst_refs = [
        metrics("clean", variant, "official_worst")
        for variant in ("keeper", "random_control")
    ]
    worst_candidate = metrics("clean", "worst_shift", "official_worst")
    candidate_support = [int(value) for value in worst_candidate["predicted_support"]]
    candidate_consistency = float(
        summaries["clean"]["worst_shift"]["shift_consistency"][
            "all_nine_same_rate"
        ]
    )
    reference_consistency = [
        float(
            summaries["clean"][variant]["shift_consistency"][
                "all_nine_same_rate"
            ]
        )
        for variant in ("keeper", "random_control")
    ]
    worst_checks = {
        "macro_f1_exceeds_both_references": float(worst_candidate["macro_f1"])
        > max(float(value["macro_f1"]) for value in worst_refs),
        "class1_f1_gte_best_reference_plus_0p005": float(
            worst_candidate["class1"]["f1"]
        )
        >= max(float(value["class1"]["f1"]) for value in worst_refs) + 0.005,
        "class1_precision_gte_best_reference_plus_0p010": float(
            worst_candidate["class1"]["precision"]
        )
        >= max(float(value["class1"]["precision"]) for value in worst_refs)
        + 0.010,
        "restricted_fp_at_least_3_below_both_references": int(
            worst_candidate["class1"]["restricted_fp"]
        )
        <= min(int(value["class1"]["restricted_fp"]) for value in worst_refs) - 3,
        "class1_tp_within_2_of_larger_reference": int(
            worst_candidate["class1"]["tp"]
        )
        >= max(int(value["class1"]["tp"]) for value in worst_refs) - 2,
        "shift_consistency_exceeds_both_references": candidate_consistency
        > max(reference_consistency),
        "no_constant_or_majority_class_collapse": bool(
            all(value > 0 for value in candidate_support)
            and max(candidate_support) / max(1, sum(candidate_support)) < 0.80
        ),
    }

    illumination_checks: Dict[str, bool] = {}
    strict_improvement_photos = 0
    pooled_candidate_fp = 0
    pooled_reference_fp = {"keeper": 0, "random_control": 0}
    for photo in PHOTOMETRIC_CONDITIONS[1:]:
        photo_strict = True
        for mode in ("clean", "official_worst"):
            refs = [
                metrics(photo, variant, mode)
                for variant in ("keeper", "random_control")
            ]
            candidate = metrics(photo, "worst_shift", mode)
            prefix = f"{photo}_{mode}"
            illumination_checks[f"{prefix}_class1_f1_safety"] = float(
                candidate["class1"]["f1"]
            ) >= max(float(value["class1"]["f1"]) for value in refs) - 0.005
            illumination_checks[f"{prefix}_class1_precision_safety"] = float(
                candidate["class1"]["precision"]
            ) >= max(
                float(value["class1"]["precision"]) for value in refs
            ) - 0.005
            illumination_checks[f"{prefix}_class1_tp_safety"] = int(
                candidate["class1"]["tp"]
            ) >= max(int(value["class1"]["tp"]) for value in refs) - 2
            illumination_checks[f"{prefix}_restricted_fp_not_increased"] = int(
                candidate["class1"]["restricted_fp"]
            ) <= min(int(value["class1"]["restricted_fp"]) for value in refs)
            photo_strict = bool(
                photo_strict
                and float(candidate["class1"]["f1"])
                > max(float(value["class1"]["f1"]) for value in refs)
                and float(candidate["class1"]["precision"])
                > max(float(value["class1"]["precision"]) for value in refs)
            )
            pooled_candidate_fp += int(candidate["class1"]["restricted_fp"])
            for variant, value in zip(("keeper", "random_control"), refs):
                pooled_reference_fp[variant] += int(
                    value["class1"]["restricted_fp"]
                )
        strict_improvement_photos += int(photo_strict)
    illumination_checks["strict_f1_precision_improvement_in_at_least_2_photos"] = (
        strict_improvement_photos >= 2
    )
    illumination_checks["pooled_restricted_fp_below_both_references"] = (
        pooled_candidate_fp < min(pooled_reference_fp.values())
    )

    all_checks = {
        **{f"structural::{name}": bool(value) for name, value in structural_checks.items()},
        **{f"clean::{name}": bool(value) for name, value in clean_checks.items()},
        **{f"worst::{name}": bool(value) for name, value in worst_checks.items()},
        **{
            f"illumination::{name}": bool(value)
            for name, value in illumination_checks.items()
        },
    }
    failed = [name for name, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "worst_shift_checks": worst_checks,
        "illumination_checks": illumination_checks,
        "strict_improvement_photometric_conditions": strict_improvement_photos,
        "pooled_restricted_fp": {
            "candidate": pooled_candidate_fp,
            **pooled_reference_fp,
        },
        "failed_checks": failed,
        "all_nonvisual_gates_passed": not failed,
        "xai_protocol_authorized": not failed,
        "validation_smoke_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "current_best_update_authorized": False,
    }


PREDICTION_FIELDS = (
    "photometric_condition",
    "variant",
    "shift_index",
    "shift_condition",
    "dx",
    "dy",
    "sample_index",
    "source_stem",
    "fold",
    "target",
    "prediction",
    "target_loss",
    "official_index",
    "is_official",
    "mean_prediction",
    "prob_0",
    "prob_1",
    "prob_2",
    "prob_3",
    "prob_4",
)


def _write_prediction_csv(
    path: Path,
    *,
    evaluations: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
    holdout: Sequence[CleanTrainRow],
) -> int:
    source = {int(row.sample_index): row for row in holdout}
    written = 0
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_FIELDS))
        writer.writeheader()
        for photo in PHOTOMETRIC_CONDITIONS:
            for variant in VARIANTS:
                evaluation = evaluations[photo][variant]
                columns = int(evaluation["targets"].numel())
                for shift_index, (dx, dy) in enumerate(TRANSFORMS):
                    for column in range(columns):
                        sample_index = int(
                            evaluation["sample_indices"][column].item()
                        )
                        source_row = source[sample_index]
                        payload: Dict[str, object] = {
                            "photometric_condition": photo,
                            "variant": variant,
                            "shift_index": shift_index,
                            "shift_condition": _condition_name(dx, dy),
                            "dx": dx,
                            "dy": dy,
                            "sample_index": sample_index,
                            "source_stem": source_row.source_stem,
                            "fold": source_row.fold,
                            "target": int(evaluation["targets"][column].item()),
                            "prediction": int(
                                evaluation["predictions"][shift_index, column].item()
                            ),
                            "target_loss": format(
                                float(
                                    evaluation["losses"][shift_index, column].item()
                                ),
                                ".17g",
                            ),
                            "official_index": int(
                                evaluation["official_indices"][column].item()
                            ),
                            "is_official": int(
                                int(evaluation["official_indices"][column].item())
                                == shift_index
                            ),
                            "mean_prediction": int(
                                evaluation["mean_predictions"][column].item()
                            ),
                        }
                        for class_index in range(5):
                            payload[f"prob_{class_index}"] = format(
                                float(
                                    evaluation["probabilities"][
                                        shift_index, column, class_index
                                    ].item()
                                ),
                                ".17g",
                            )
                        writer.writerow(payload)
                        written += 1
    return written


def _read_prediction_csv(
    path: Path,
) -> tuple[Dict[str, Dict[str, Dict[str, Tensor]]], Dict[str, object]]:
    grouped: Dict[tuple[str, str], Dict[int, list[Dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            photo = str(raw["photometric_condition"])
            variant = str(raw["variant"])
            shift_index = int(raw["shift_index"])
            row: Dict[str, object] = {
                "sample_index": int(raw["sample_index"]),
                "target": int(raw["target"]),
                "prediction": int(raw["prediction"]),
                "target_loss": float(raw["target_loss"]),
                "official_index": int(raw["official_index"]),
                "mean_prediction": int(raw["mean_prediction"]),
            }
            row["probabilities"] = [
                float(raw[f"prob_{class_index}"]) for class_index in range(5)
            ]
            grouped[(photo, variant)][shift_index].append(row)
            rows += 1
    expected_groups = {
        (photo, variant)
        for photo in PHOTOMETRIC_CONDITIONS
        for variant in VARIANTS
    }
    if set(grouped) != expected_groups:
        raise ValueError("Spatial replay photometric/variant groups differ.")
    evaluations: Dict[str, Dict[str, Dict[str, Tensor]]] = {
        photo: {} for photo in PHOTOMETRIC_CONDITIONS
    }
    declared_indices_exact = True
    declared_mean_exact = True
    for photo, variant in sorted(expected_groups):
        conditions = grouped[(photo, variant)]
        if set(conditions) != set(range(len(TRANSFORMS))):
            raise ValueError("Spatial replay shift-condition set differs.")
        first = conditions[0]
        sample_indices = [int(row["sample_index"]) for row in first]
        targets = [int(row["target"]) for row in first]
        for shift_index in range(len(TRANSFORMS)):
            current = conditions[shift_index]
            if [int(row["sample_index"]) for row in current] != sample_indices:
                raise ValueError("Spatial replay sample order differs by shift.")
            if [int(row["target"]) for row in current] != targets:
                raise ValueError("Spatial replay targets differ by shift.")
        evaluation = {
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "losses": torch.tensor(
                [
                    [float(row["target_loss"]) for row in conditions[index]]
                    for index in range(len(TRANSFORMS))
                ],
                dtype=torch.float32,
            ),
            "predictions": torch.tensor(
                [
                    [int(row["prediction"]) for row in conditions[index]]
                    for index in range(len(TRANSFORMS))
                ],
                dtype=torch.long,
            ),
            "probabilities": torch.tensor(
                [
                    [row["probabilities"] for row in conditions[index]]
                    for index in range(len(TRANSFORMS))
                ],
                dtype=torch.float32,
            ),
        }
        official = official_worst_indices(
            evaluation["losses"], evaluation["predictions"], evaluation["targets"]
        )
        mean_predictions = evaluation["probabilities"].mean(dim=0).argmax(dim=1)
        declared_official = torch.tensor(
            [int(row["official_index"]) for row in first], dtype=torch.long
        )
        declared_mean = torch.tensor(
            [int(row["mean_prediction"]) for row in first], dtype=torch.long
        )
        declared_indices_exact = bool(
            declared_indices_exact and torch.equal(official, declared_official)
        )
        declared_mean_exact = bool(
            declared_mean_exact and torch.equal(mean_predictions, declared_mean)
        )
        evaluation["official_indices"] = official
        evaluation["mean_predictions"] = mean_predictions
        evaluations[photo][variant] = evaluation
    expected_rows = (
        len(PHOTOMETRIC_CONDITIONS)
        * len(VARIANTS)
        * len(TRANSFORMS)
        * EXPECTED_HOLDOUT_ROWS
    )
    return evaluations, {
        "rows": rows,
        "expected_rows": expected_rows,
        "row_count_exact": rows == expected_rows,
        "official_indices_recomputed_exact": declared_indices_exact,
        "mean_predictions_recomputed_exact": declared_mean_exact,
    }


def _probe_signal(
    model: nn.Module,
    batches: Sequence[SpatialBatch],
    *,
    device: torch.device,
) -> tuple[list[Dict[str, object]], Dict[str, object]]:
    model.to(device).eval()
    rows: list[Dict[str, object]] = []
    output_replay_exact = True
    transform_replay_exact = True
    for step, batch in enumerate(batches):
        first = _grid_forward(model, batch, device=device)
        second = _grid_forward(model, batch, device=device)
        output_replay_exact = bool(
            output_replay_exact
            and torch.equal(first.losses, second.losses)
            and torch.equal(first.predictions, second.predictions)
            and torch.equal(first.probabilities, second.probabilities)
        )
        for condition_index in range(len(TRANSFORMS)):
            transform_replay_exact = bool(
                transform_replay_exact
                and _batch_hash(_shift_batch(batch, condition_index))
                == _batch_hash(_shift_batch(batch, condition_index))
            )
        official = official_worst_indices(
            first.losses, first.predictions, batch.targets
        )
        reference = official_worst_indices_reference(
            first.losses, first.predictions, batch.targets
        )
        if not torch.equal(official, reference):
            raise ValueError("Probe official spatial selector differs from oracle.")
        random_indices = random_shift_indices(batch.sample_indices, step=step)
        rows.extend(
            _selection_rows(
                batch=batch,
                grid=first,
                official_indices=official,
                random_indices=random_indices,
                selected_indices=official,
                variant="probe",
                step=step,
            )
        )
    model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    assessment = assess_probe_signal(
        rows,
        output_replay_exact=output_replay_exact,
        transform_replay_exact=transform_replay_exact,
    )
    return rows, {
        **assessment,
        "evaluation_forwards": len(batches) * len(TRANSFORMS) * 2,
        "optimizer_updates": 0,
    }


def _selection_audit(
    random_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if len(random_rows) != EXPECTED_UPDATE_ROWS or len(candidate_rows) != EXPECTED_UPDATE_ROWS:
        raise ValueError("Spatial training selection-row count differs.")
    random_by_key = {
        (int(row["step"]), int(row["sample_index"])): row for row in random_rows
    }
    candidate_by_key = {
        (int(row["step"]), int(row["sample_index"])): row
        for row in candidate_rows
    }
    if set(random_by_key) != set(candidate_by_key):
        raise ValueError("Spatial candidate/control selection keys differ.")

    random_replay_exact = True
    oracle_parity_exact = True
    candidate_uses_official_exact = True
    all_finite = True
    differences = 0
    for key in sorted(random_by_key):
        random_row = random_by_key[key]
        candidate_row = candidate_by_key[key]
        step, sample_index = key
        expected_random = int(
            random_shift_indices(torch.tensor([sample_index]), step=step)[0].item()
        )
        random_replay_exact = bool(
            random_replay_exact
            and int(random_row["random_index"]) == expected_random
            and int(random_row["selected_index"]) == expected_random
        )
        candidate_uses_official_exact = bool(
            candidate_uses_official_exact
            and int(candidate_row["selected_index"])
            == int(candidate_row["official_index"])
        )
        for row in (random_row, candidate_row):
            losses = torch.tensor(
                [[float(row[f"loss_{index}"])] for index in range(len(TRANSFORMS))]
            )
            predictions = torch.tensor(
                [[int(row[f"pred_{index}"])] for index in range(len(TRANSFORMS))]
            )
            target = torch.tensor([int(row["target"])])
            expected_official = int(
                official_worst_indices_reference(losses, predictions, target)[0].item()
            )
            oracle_parity_exact = bool(
                oracle_parity_exact
                and int(row["official_index"]) == expected_official
            )
            all_finite = bool(
                all_finite
                and all(
                    math.isfinite(float(row[f"loss_{index}"]))
                    for index in range(len(TRANSFORMS))
                )
                and math.isfinite(float(row["selected_loss"]))
            )
        differences += int(
            int(candidate_row["selected_index"])
            != int(random_row["selected_index"])
        )
    return {
        "rows_per_variant": EXPECTED_UPDATE_ROWS,
        "random_selection_replay_exact": random_replay_exact,
        "official_oracle_parity_exact": oracle_parity_exact,
        "candidate_uses_official_exact": candidate_uses_official_exact,
        "all_selected_rows_finite": all_finite,
        "candidate_random_selection_differences": differences,
        "candidate_random_selection_difference_fraction": float(
            differences / EXPECTED_UPDATE_ROWS
        ),
    }


SELECTION_FIELDS = (
    "variant",
    "step",
    "sample_index",
    "target",
    "official_index",
    "random_index",
    "selected_index",
    "selected_dx",
    "selected_dy",
    "selected_loss",
    "selected_prediction",
    "any_misclassification",
    "target_loss_span",
    *(f"loss_{index}" for index in range(len(TRANSFORMS))),
    *(f"pred_{index}" for index in range(len(TRANSFORMS))),
)


def _write_selection_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SELECTION_FIELDS))
        writer.writeheader()
        for row in rows:
            payload = {}
            for name in SELECTION_FIELDS:
                value = row[name]
                payload[name] = format(value, ".17g") if isinstance(value, float) else value
            writer.writerow(payload)


def _summarize_all(
    evaluations: Mapping[str, Mapping[str, Mapping[str, Tensor]]],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    return {
        photo: {
            variant: _summarize_evaluation(evaluations[photo][variant])
            for variant in VARIANTS
        }
        for photo in PHOTOMETRIC_CONDITIONS
    }


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    decision = summary.get("decision", {})
    summaries = summary.get("evaluation", {})
    clean_candidate = {}
    if isinstance(summaries, Mapping):
        clean = summaries.get("clean", {})
        if isinstance(clean, Mapping):
            candidate = clean.get("worst_shift", {})
            if isinstance(candidate, Mapping):
                clean_candidate = candidate.get("clean", {})
    class1 = clean_candidate.get("class1", {}) if isinstance(clean_candidate, Mapping) else {}
    failed = decision.get("failed_checks", []) if isinstance(decision, Mapping) else []
    probe = summary.get("probe", {})
    probe_passed = probe.get("passed") if isinstance(probe, Mapping) else None
    nonvisual_passed = (
        decision.get("all_nonvisual_gates_passed")
        if isinstance(decision, Mapping)
        else None
    )
    clean_macro = (
        clean_candidate.get("macro_f1")
        if isinstance(clean_candidate, Mapping)
        else None
    )
    class1_precision = (
        class1.get("precision") if isinstance(class1, Mapping) else None
    )
    class1_recall = class1.get("recall") if isinstance(class1, Mapping) else None
    class1_f1 = class1.get("f1") if isinstance(class1, Mapping) else None
    class1_tp = class1.get("tp") if isinstance(class1, Mapping) else None
    class1_fp = class1.get("fp") if isinstance(class1, Mapping) else None
    restricted_fp = (
        class1.get("restricted_fp") if isinstance(class1, Mapping) else None
    )
    lines = [
        "# Spatial Worst-Shift Robust Optimization A0 Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Probe passed: `{probe_passed}`",
        f"- Nonvisual gates passed: `{nonvisual_passed}`",
        f"- Validation used: `{summary.get('validation_data_used')}`",
        f"- Test used: `{summary.get('test_data_used')}`",
        f"- Raw data modified: `{summary.get('raw_data_modified')}`",
        "",
        "## Clean Candidate",
        "",
        f"- Macro F1: `{clean_macro}`",
        f"- Class-1 precision: `{class1_precision}`",
        f"- Class-1 recall: `{class1_recall}`",
        f"- Class-1 F1: `{class1_f1}`",
        (
            "- Class-1 TP / FP / restricted FP: "
            f"`{class1_tp}` / `{class1_fp}` / `{restricted_fp}`"
        ),
        "",
        "## Failed Gates",
        "",
    ]
    lines.extend(f"- `{value}`" for value in failed)
    if not failed:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Authority",
            "",
            (
                "A0 can authorize only a separately locked XAI protocol. It cannot "
                "authorize validation, test, full training, deployment, or a "
                "current-best command update."
            ),
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    output_dir: Path, *, binary_model_payload_retained: bool
) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    files = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload = {
        "method": METHOD,
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "files": files,
        "file_count": len(files),
        "payload_bytes": sum(int(value["bytes"]) for value in files),
        "binary_model_payload_retained": bool(binary_model_payload_retained),
        "onnx_payload_retained": False,
        "tensorrt_engine_retained": False,
        "raw_data_modified": False,
        "validation_data_used": False,
        "test_data_used": False,
    }
    _write_json(manifest_path, payload)
    return {**payload, "manifest_sha256": _sha256(manifest_path)}


def _load_prototype_and_dataset(
    *,
    paths: Mapping[str, Path],
    rows: Sequence[CleanTrainRow],
) -> tuple[
    Mapping[str, object],
    nn.Module,
    MangoYOLOCropDataset,
    object,
    Dict[str, object],
]:
    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    source_config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if (
        len(class_names) != 5
        or not isinstance(source_config, Mapping)
        or not isinstance(state, Mapping)
    ):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    prototype = create_model(num_classes=5, model_config=source_config)
    load_model_state(prototype, dict(state), strict=True)
    prototype.cpu().eval()
    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(
        paths["data"], class_name_mode="raw", expected_num_classes=5
    )
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
    for row, sample_path in zip(rows, sample_paths):
        if row.image_path != sample_path:
            raise ValueError(
                f"Dataset/CIDT path mismatch at sample {row.sample_index}."
            )
    transform = _build_eval_transform(semantics)
    declaration = {
        "class_names": class_names,
        "semantics": semantics,
        "parameter_count": sum(int(value.numel()) for value in prototype.parameters()),
        "prototype_state_sha256": _state_sha256(prototype),
        "model_state_schema": _state_schema(prototype),
    }
    return checkpoint, prototype, base_dataset, transform, declaration


def _condition_loader(
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    photo_index: int,
    args: argparse.Namespace,
) -> tuple[DataLoader, Dict[str, object]]:
    if photo_index == 0:
        return _make_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context="spatial_a0_holdout_clean",
            seed=SEED + 300,
        )
    condition, brightness, contrast = LIGHTING_CONDITIONS[photo_index - 1]
    return _make_lighting_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        brightness=brightness,
        contrast=contrast,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"spatial_a0_holdout_{condition}",
        seed=SEED + 300 + photo_index,
    )


def _finalize_summary(
    *,
    output_dir: Path,
    summary: Dict[str, object],
    binary_model_payload_retained: bool,
) -> Dict[str, object]:
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_json(summary_path, summary)
    _write_report(report_path, summary)
    manifest = _write_manifest(
        output_dir,
        binary_model_payload_retained=binary_model_payload_retained,
    )
    return {
        **summary,
        "formal_summary_sha256": _sha256(summary_path),
        "report_sha256": _sha256(report_path),
        "artifact_manifest": manifest,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    provenance, rows, update, probe, holdout, cohort_summary = _load_locked_inputs(
        args
    )
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "preflight_created_output": False,
            "provenance": provenance,
            "cohorts": cohort_summary,
            "rows": {
                "train": len(rows),
                "update": len(update),
                "probe": len(probe),
                "holdout": len(holdout),
            },
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Locked spatial A0 requires CUDA.")
    process_before = _process_snapshot()
    gpu_before = _gpu_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "Spatial A0 found another Python/TensorRT process; none was terminated."
        )
    if not gpu_before["gpus"] or any(
        int(row["utilization_percent"]) > 15
        or int(row["memory_used_mib"]) > 2200
        for row in gpu_before["gpus"]
    ):
        raise RuntimeError(
            "Spatial A0 GPU is not isolated; no output or timing decision was made."
        )
    output_dir = _prepare_output_dir(args.output_dir)
    _progress("formal_started", output_dir=str(output_dir))
    paths = {name: Path(value) for name, value in provenance["paths"].items()}
    device = torch.device("cuda")
    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    equation = equation_diagnostics()
    if not bool(equation["passed"]):
        raise ValueError(f"Spatial equation diagnostics failed: {equation['failed_checks']}")
    (
        checkpoint,
        prototype,
        base_dataset,
        transform,
        model_declaration,
    ) = _load_prototype_and_dataset(paths=paths, rows=rows)
    prototype_hash = str(model_declaration["prototype_state_sha256"])
    parameter_count = int(model_declaration["parameter_count"])

    probe_indices = [row.sample_index for row in probe]
    probe_loader, probe_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=probe_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="spatial_a0_probe",
        seed=SEED + 100,
    )
    probe_batches, probe_materialization = _materialize_batches(
        probe_loader, expected_indices=probe_indices
    )
    del probe_loader
    gc.collect()
    probe_rows, probe_assessment = _probe_signal(
        prototype, probe_batches, device=device
    )
    _progress(
        "probe_completed",
        passed=probe_assessment["passed"],
        failed_checks=probe_assessment["failed_checks"],
    )
    selection_path = output_dir / "selection_rows.csv"
    _write_selection_csv(selection_path, probe_rows)
    if not bool(probe_assessment["passed"]):
        process_after = _process_snapshot()
        gpu_after = _gpu_snapshot()
        batch_path = output_dir / "batch_manifest.json"
        _write_json(
            batch_path,
            {
                "cohorts": cohort_summary,
                "probe_materialization": probe_materialization,
                "update_materialization": None,
            },
        )
        summary = {
            "method": METHOD,
            "status": "rejected_probe_signal",
            "provenance": provenance,
            "config": vars(args),
            "equation": equation,
            "probe": probe_assessment,
            "dataset": {
                "cohorts": cohort_summary,
                "probe_loader": probe_loader_summary,
            },
            "process_gpu": {
                "before": {"process": process_before, "gpu": gpu_before},
                "after": {"process": process_after, "gpu": gpu_after},
            },
            "training": {"ran": False, "optimizer_updates": 0},
            "evaluation": {},
            "decision": {
                "all_nonvisual_gates_passed": False,
                "failed_checks": [
                    f"probe::{name}" for name in probe_assessment["failed_checks"]
                ],
                "xai_protocol_authorized": False,
                "validation_smoke_authorized": False,
                "test_authorized": False,
                "full_train_authorized": False,
                "current_best_update_authorized": False,
            },
            "candidate_checkpoint": None,
            "validation_data_used": False,
            "test_data_used": False,
            "image_model_training_used": False,
            "raw_data_modified": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        return _finalize_summary(
            output_dir=output_dir,
            summary=summary,
            binary_model_payload_retained=False,
        )

    del probe_batches
    gc.collect()
    update_indices = [row.sample_index for row in update]
    update_loader, update_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=update_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="spatial_a0_update",
        seed=SEED + 200,
    )
    update_batches, update_materialization = _materialize_batches(
        update_loader, expected_indices=update_indices
    )
    del update_loader
    gc.collect()
    random_model, random_summary, random_history, random_rows = _train_variant(
        prototype=prototype,
        batches=update_batches,
        variant="random_control",
        args=args,
        device=device,
    )
    _progress("random_control_completed", wall_seconds=random_summary["wall_seconds"])
    candidate_model, candidate_summary, candidate_history, candidate_rows = (
        _train_variant(
            prototype=prototype,
            batches=update_batches,
            variant="worst_shift",
            args=args,
            device=device,
        )
    )
    _progress(
        "worst_shift_completed", wall_seconds=candidate_summary["wall_seconds"]
    )
    selection_audit = _selection_audit(random_rows, candidate_rows)
    _write_selection_csv(
        selection_path, [*probe_rows, *random_rows, *candidate_rows]
    )
    batch_logical_replay_exact = [
        (row["batch_sha256"], row["logical_seed"]) for row in random_history
    ] == [
        (row["batch_sha256"], row["logical_seed"]) for row in candidate_history
    ]
    del update_batches
    gc.collect()

    base_dataset.enable_image_cache(max_megabytes=2304, max_items=2000)
    image_cache_before = base_dataset.image_cache_stats()
    holdout_indices = [row.sample_index for row in holdout]
    evaluations: Dict[str, Dict[str, Dict[str, Tensor]]] = {}
    loader_summaries: Dict[str, object] = {}
    condition_materialization: Dict[str, object] = {}
    resource_batch: Optional[SpatialBatch] = None
    for photo_index, photo in enumerate(PHOTOMETRIC_CONDITIONS):
        loader, loader_summary = _condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            photo_index=photo_index,
            args=args,
        )
        condition_batches, materialization = _materialize_batches(
            loader, expected_indices=holdout_indices
        )
        del loader
        gc.collect()
        if resource_batch is None:
            resource_batch = condition_batches[0]
        evaluations[photo] = {
            "keeper": _evaluate_grid_model(
                prototype,
                condition_batches,
                expected_indices=holdout_indices,
                device=device,
            ),
            "random_control": _evaluate_grid_model(
                random_model,
                condition_batches,
                expected_indices=holdout_indices,
                device=device,
            ),
            "worst_shift": _evaluate_grid_model(
                candidate_model,
                condition_batches,
                expected_indices=holdout_indices,
                device=device,
            ),
        }
        _progress(
            "holdout_condition_completed",
            photometric_condition=photo,
            rows=len(holdout_indices),
        )
        loader_summaries[photo] = loader_summary
        condition_materialization[photo] = materialization
        del condition_batches
        gc.collect()
    if resource_batch is None:
        raise ValueError("Spatial resource batch was not materialized.")
    image_cache_after = base_dataset.image_cache_stats()
    live_summaries = _summarize_all(evaluations)
    live_evidence = _build_evidence(evaluations)
    prediction_path = output_dir / "predictions_all_conditions.csv"
    written_prediction_rows = _write_prediction_csv(
        prediction_path,
        evaluations=evaluations,
        holdout=holdout,
    )
    _progress("prediction_csv_written", rows=written_prediction_rows)
    replay_evaluations, replay_audit = _read_prediction_csv(prediction_path)
    replay_summaries = _summarize_all(replay_evaluations)
    replay_evidence = _build_evidence(replay_evaluations)
    replay_error = _maximum_numeric_difference(
        {"summaries": live_summaries, "evidence": live_evidence},
        {"summaries": replay_summaries, "evidence": replay_evidence},
    )
    independent_replay = {
        **replay_audit,
        "writer_rows": written_prediction_rows,
        "maximum_absolute_difference": replay_error,
        "exact_within_1e12": bool(
            replay_audit["row_count_exact"]
            and replay_audit["official_indices_recomputed_exact"]
            and replay_audit["mean_predictions_recomputed_exact"]
            and math.isfinite(replay_error)
            and replay_error <= 1e-12
        ),
    }
    del replay_evaluations
    gc.collect()

    runtime = _benchmark_pair(
        native=prototype,
        candidate=candidate_model,
        batch=resource_batch,
        device=device,
        repeats=int(args.benchmark_repeats),
    )
    _progress(
        "runtime_benchmark_completed",
        runtime_ratio=runtime["runtime_ratio"],
        memory_ratio=runtime["memory_ratio"],
    )
    export = _export_candidate(
        candidate=candidate_model,
        batch=resource_batch,
        output_dir=output_dir,
    )
    process_after = _process_snapshot()
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = _gpu_snapshot()

    all_required_gradients = all(
        bool(summary["gradient_seen"].get(group, False))
        for summary in (random_summary, candidate_summary)
        for group in REQUIRED_PARAMETER_GROUPS
    )
    all_required_movement = all(
        bool(summary["parameter_group_movement"].get(group, False))
        for summary in (random_summary, candidate_summary)
        for group in REQUIRED_PARAMETER_GROUPS
    )
    repository_root = Path(__file__).resolve().parents[2]
    sources_unchanged_after = bool(
        _sha256(paths["checkpoint"]) == LOCKED_KEEPER_SHA256
        and _sha256(paths["data"]) == LOCKED_DATA_SHA256
        and _sha256(paths["current_commands"]) == LOCKED_CURRENT_COMMAND_SHA256
        and _sha256(paths["command_history"]) == LOCKED_COMMAND_HISTORY_SHA256
        and _tracked_worktree_clean(repository_root)
        and _git_value(repository_root, "rev-parse", "HEAD")
        == _git_value(
            repository_root,
            "rev-parse",
            "origin/classification-only-research",
        )
    )
    structural_checks = {
        "equation_diagnostics_passed": bool(equation["passed"]),
        "probe_signal_passed": bool(probe_assessment["passed"]),
        "locked_sources_and_train_only_provenance_exact": bool(
            not provenance["validation_data_used"]
            and not provenance["test_data_used"]
            and sources_unchanged_after
        ),
        "source_disjoint_cohorts_exact": bool(
            not any(cohort_summary["source_overlap"].values())
            and len(update) == EXPECTED_UPDATE_ROWS
            and len(probe) == EXPECTED_PROBE_ROWS
            and len(holdout) == EXPECTED_HOLDOUT_ROWS
        ),
        "only_current_auditor_process_before_and_after": bool(
            not process_before["unexpected_processes"]
            and not process_after["unexpected_processes"]
        ),
        "gpu_isolated_before_formal": all(
            int(row["utilization_percent"]) <= 15
            and int(row["memory_used_mib"]) <= 2200
            for row in gpu_before["gpus"]
        ),
        "initial_model_states_exact": bool(
            random_summary["initial_state_sha256"] == prototype_hash
            and candidate_summary["initial_state_sha256"] == prototype_hash
        ),
        "batch_bytes_and_logical_rng_exact": batch_logical_replay_exact,
        "both_variants_have_32_updates": bool(
            int(random_summary["updates"])
            == int(candidate_summary["updates"])
            == UPDATES
        ),
        "both_variants_have_320_forwards": bool(
            int(random_summary["total_forwards"])
            == int(candidate_summary["total_forwards"])
            == UPDATES * (len(TRANSFORMS) + 1)
        ),
        "optimizer_steps_exact": bool(
            random_summary["optimizer_steps"] == [float(UPDATES)]
            and candidate_summary["optimizer_steps"] == [float(UPDATES)]
        ),
        "random_selection_replay_exact": bool(
            selection_audit["random_selection_replay_exact"]
        ),
        "official_oracle_selection_parity_exact": bool(
            selection_audit["official_oracle_parity_exact"]
        ),
        "candidate_uses_official_selection_exact": bool(
            selection_audit["candidate_uses_official_exact"]
        ),
        "all_selected_rows_finite": bool(
            selection_audit["all_selected_rows_finite"]
        ),
        "candidate_random_selection_difference_gte_25pct": float(
            selection_audit["candidate_random_selection_difference_fraction"]
        )
        >= 0.25,
        "all_gradients_finite": bool(
            random_summary["all_gradients_finite"]
            and candidate_summary["all_gradients_finite"]
        ),
        "all_required_gradient_groups_seen": all_required_gradients,
        "all_required_parameter_groups_moved": all_required_movement,
        "model_state_schema_unchanged": bool(
            _state_schema(prototype)
            == _state_schema(random_model)
            == _state_schema(candidate_model)
        ),
        "parameter_count_unchanged": all(
            sum(int(value.numel()) for value in model.parameters()) == parameter_count
            for model in (prototype, random_model, candidate_model)
        ),
        "training_peak_each_lte_4p5_gib": max(
            float(random_summary["peak_vram_gib"]),
            float(candidate_summary["peak_vram_gib"]),
        )
        <= MAX_TRAINING_VRAM_GIB,
        "candidate_control_training_wall_ratio_lte_1p10": float(
            candidate_summary["wall_seconds"]
        )
        / max(float(random_summary["wall_seconds"]), 1e-12)
        <= MAX_TRAINING_WALL_RATIO,
        "independent_csv_replay_exact": bool(
            independent_replay["exact_within_1e12"]
        ),
        "all_evaluation_outputs_finite": all(
            bool(live_summaries[photo][variant]["all_numeric_finite"])
            for photo in PHOTOMETRIC_CONDITIONS
            for variant in VARIANTS
        ),
        "inference_outputs_finite": bool(runtime["all_outputs_finite"]),
        "inference_runtime_ratio_lte_1p02": float(runtime["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "inference_memory_ratio_lte_1p02": float(runtime["memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "static_onnx_succeeded": bool(export.get("succeeded", False)),
        "static_onnx_finite": bool(export.get("finite", False)),
        "static_onnx_error_lte_1e5": float(
            export.get("maximum_absolute_error", math.inf)
        )
        <= MAX_ONNX_ERROR,
        "static_onnx_argmax_exact": bool(export.get("argmax_match", False)),
    }
    decision = assess_spatial_a0(
        structural_checks=structural_checks,
        summaries=live_summaries,
        evidence=live_evidence,
    )

    history_path = output_dir / "training_history.json"
    _write_json(
        history_path,
        {
            "random_control": random_history,
            "worst_shift": candidate_history,
            "batch_logical_replay_exact": batch_logical_replay_exact,
        },
    )
    batch_path = output_dir / "batch_manifest.json"
    _write_json(
        batch_path,
        {
            "cohorts": cohort_summary,
            "probe_materialization": probe_materialization,
            "update_materialization": update_materialization,
            "holdout_materialization": condition_materialization,
            "probe_loader": probe_loader_summary,
            "update_loader": update_loader_summary,
            "holdout_loaders": loader_summaries,
        },
    )
    checkpoint_artifact = None
    if bool(decision["xai_protocol_authorized"]):
        checkpoint_path = output_dir / "spatial_worst_shift_candidate.pt"
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
                "model_state": candidate_model.state_dict(),
                "stage": METHOD,
                "validation_data_used": False,
                "test_data_used": False,
                "protocol_sha256": LOCKED_PROTOCOL_SHA256,
            }
        )
        torch.save(payload, checkpoint_path)
        checkpoint_artifact = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": int(checkpoint_path.stat().st_size),
        }

    summary = {
        "method": METHOD,
        "status": (
            "passed_nonvisual_xai_required"
            if decision["xai_protocol_authorized"]
            else "rejected"
        ),
        "provenance": provenance,
        "config": vars(args),
        "equation": equation,
        "probe": probe_assessment,
        "process_gpu": {
            "before": {"process": process_before, "gpu": gpu_before},
            "after": {"process": process_after, "gpu": gpu_after},
        },
        "dataset": {
            "cohorts": cohort_summary,
            "train_rows": len(rows),
            "update_rows": len(update),
            "probe_rows": len(probe),
            "holdout_rows": len(holdout),
            "image_cache_before_evaluation": image_cache_before,
            "image_cache_after_evaluation": image_cache_after,
            "validation_data_used": False,
            "test_data_used": False,
        },
        "model": {
            **model_declaration,
            "random_state_sha256": _state_sha256(random_model),
            "candidate_state_sha256": _state_sha256(candidate_model),
        },
        "training": {
            "random_control": random_summary,
            "worst_shift": candidate_summary,
            "selection": selection_audit,
            "batch_logical_replay_exact": batch_logical_replay_exact,
            "candidate_control_wall_ratio": float(
                candidate_summary["wall_seconds"]
            )
            / max(float(random_summary["wall_seconds"]), 1e-12),
        },
        "evaluation": live_summaries,
        "evidence": live_evidence,
        "independent_csv_replay": independent_replay,
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
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": True,
        "raw_data_modified": False,
        "production_integration_authorized": False,
        "current_command_update_authorized": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return _finalize_summary(
        output_dir=output_dir,
        summary=summary,
        binary_model_payload_retained=checkpoint_artifact is not None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(
        json.dumps(
            {
                "method": result.get("method"),
                "status": result.get("status"),
                "formal_summary_sha256": result.get("formal_summary_sha256"),
                "nonvisual_gate_passed": result.get("decision", {}).get(
                    "all_nonvisual_gates_passed"
                )
                if isinstance(result.get("decision"), Mapping)
                else None,
                "xai_protocol_authorized": result.get("decision", {}).get(
                    "xai_protocol_authorized", False
                )
                if isinstance(result.get("decision"), Mapping)
                else False,
                "current_best_update_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

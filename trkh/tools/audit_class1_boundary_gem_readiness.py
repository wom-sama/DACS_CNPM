from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import create_model, load_model_state
from trkh.tools.audit_class1_reference_agem_readiness import (
    EXPECTED_HARD_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
    STEP_RATIO,
    _apply_normalized_parameter_step,
    _cohort_serializable,
    _evaluate_condition,
    _git_worktree_clean,
    _is_relative_to,
    _locked_cohort_summary,
    _parameter_group,
    _parameter_schema,
    _write_manifest,
    project_agem_gradient,
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
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _comparison,
    _git_commit,
    _make_loader,
    _prepare_output_dir,
    _sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


Gradient = list[Tensor]

STRATUM_COUNT = 4
STRATUM_ROWS = 108
EXPECTED_REFERENCE_ROWS = STRATUM_COUNT * STRATUM_ROWS
EXPECTED_STRATUM_INDEX_HASHES = (
    "8c1e751941bca6e4bcba7420aaa7edeb20cb4d939b0ef22676b4198299ee1c1f",
    "265d7a45af94fe7509f73500e9bb2afb7f4d24f3b70e0cec39cbd42c464b34d9",
    "2b9e364969b21cf9b4c303035c768eb22818761fd236d804002ac4eef8526ac3",
    "cccd701d5e54a4491066f7f0748f39978cb647747714a652aee81095197b7868",
)
EXPECTED_STRATUM_PREDICTION_COUNTS = (
    {0: 10, 1: 97, 4: 1},
    {1: 108},
    {1: 108},
    {1: 108},
)
EXPECTED_STRATUM_FOLD_COUNTS = (
    {1: 30, 2: 24, 3: 26, 4: 28},
    {1: 31, 2: 25, 3: 26, 4: 26},
    {1: 21, 2: 32, 3: 23, 4: 32},
    {1: 33, 2: 23, 3: 28, 4: 24},
)
CONDITIONS = ("clean", *(value[0] for value in LIGHTING_CONDITIONS))

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "ed44d3f397cb6e2127edf33d211c50b73f413e594a3ccce1c9c33ca8448962c9"
LOCKED_PRIOR_AGEM_SUMMARY_SHA256 = "a74deb9b4d235ced3c860ec00d21e65baa5efdecbdf6841b67552c0ecfae1732"
LOCKED_PRIOR_AGEM_PREDICTIONS_SHA256 = "bc31de906c4f179487c5104a7d42710bda2a91be88e137aaef53f86a4af1d1d2"
LOCKED_PRIOR_AGEM_MANIFEST_SHA256 = "417c3585604ed65f747e0c2265ef28acdb5b1a26810844f367780219f8feee75"
LOCKED_GEM_PAPER_SHA256 = "9aa6a5f73220449cbe8b2c9bd4bdc3f5407ac49295c4ca1f242a36e79e34b763"
LOCKED_GEM_COMMIT = "34c6b8e9a0607db7567301c48b727430d20bee7e"
LOCKED_GEM_SOURCE_SHA256 = "28e7d895165fd4c12044505309845db0c48bdf123acfff11a35d598295c1f123"
LOCKED_GEM_LICENSE_SHA256 = "864df6b58bf3660c2944072253e854a0f51f47e26e251e1cab6e911e1e8b057e"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only class-1 boundary-stratified GEM information gate. "
            "Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/launcher_args.json"
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
            "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--prior-agem-summary",
        type=Path,
        default=Path("runs/audit_class1_reference_agem_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--prior-agem-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_reference_agem_readiness_20260715/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--prior-agem-manifest",
        type=Path,
        default=Path(
            "runs/audit_class1_reference_agem_readiness_20260715/artifact_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CLASS1_BOUNDARY_GEM_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--official-gem-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\gem"),
    )
    parser.add_argument(
        "--gem-paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\gem_neurips2017.pdf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_class1_boundary_gem_readiness_20260715"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--focus-class", type=int, default=FOCUS_CLASS)
    parser.add_argument("--strata", type=int, default=STRATUM_COUNT)
    parser.add_argument("--step-ratio", type=float, default=STEP_RATIO)
    parser.add_argument("--gem-gamma", type=float, default=0.0)
    parser.add_argument("--gem-ridge", type=float, default=0.0)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == 32
        and int(args.num_workers) == 4
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.focus_class) == FOCUS_CLASS
        and int(args.strata) == STRATUM_COUNT
        and math.isclose(float(args.step_ratio), STEP_RATIO, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(float(args.gem_gamma), 0.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(args.gem_ridge), 0.0, rel_tol=0.0, abs_tol=0.0)
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    gem_root = Path(args.official_gem_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "prior_agem_summary": Path(args.prior_agem_summary).resolve(),
        "prior_agem_predictions": Path(args.prior_agem_predictions).resolve(),
        "prior_agem_manifest": Path(args.prior_agem_manifest).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "gem_paper": Path(args.gem_paper).resolve(),
        "gem_source": gem_root / "model" / "gem.py",
        "gem_license": gem_root / "LICENSE",
    }


def _rank_boundary_strata(
    rows: Sequence[CleanTrainRow],
    *,
    fit_fold: int,
    focus_class: int,
    strata: int,
) -> list[list[CleanTrainRow]]:
    eligible = [
        row
        for row in rows
        if row.fold != int(fit_fold) and row.target == int(focus_class)
    ]
    if not eligible or len(eligible) % int(strata) != 0:
        raise ValueError("Class-1 boundary rows cannot be split into equal strata.")

    def decision_margin(row: CleanTrainRow) -> float:
        focus_probability = max(row.keeper_probabilities[int(focus_class)], 1e-30)
        nonfocus_probability = max(
            row.keeper_probabilities[index]
            for index in range(len(row.keeper_probabilities))
            if index != int(focus_class)
        )
        return math.log(focus_probability) - math.log(max(nonfocus_probability, 1e-30))

    ordered = sorted(eligible, key=lambda row: (decision_margin(row), row.sample_index))
    width = len(ordered) // int(strata)
    return [ordered[index * width : (index + 1) * width] for index in range(int(strata))]


def _locked_strata_summary(
    rows: Sequence[CleanTrainRow],
    *,
    fold: int,
) -> tuple[list[list[int]], list[Dict[str, object]]]:
    groups = _rank_boundary_strata(
        rows,
        fit_fold=fold,
        focus_class=FOCUS_CLASS,
        strata=STRATUM_COUNT,
    )
    indices: list[list[int]] = []
    summaries: list[Dict[str, object]] = []
    for stratum_index, group in enumerate(groups):
        group_indices = [row.sample_index for row in group]
        index_sha256 = _ordered_index_sha256(group_indices)
        prediction_counts = {
            class_index: sum(row.keeper_prediction == class_index for row in group)
            for class_index in range(5)
            if any(row.keeper_prediction == class_index for row in group)
        }
        fold_counts = {
            candidate_fold: sum(row.fold == candidate_fold for row in group)
            for candidate_fold in range(5)
            if candidate_fold != int(fold)
        }
        margins = []
        for row in group:
            focus_probability = max(row.keeper_probabilities[FOCUS_CLASS], 1e-30)
            nonfocus_probability = max(
                row.keeper_probabilities[index]
                for index in range(5)
                if index != FOCUS_CLASS
            )
            margins.append(math.log(focus_probability) - math.log(nonfocus_probability))
        checks = {
            "rows_exact": len(group) == STRATUM_ROWS,
            "index_hash_exact": index_sha256
            == EXPECTED_STRATUM_INDEX_HASHES[stratum_index],
            "prediction_counts_exact": prediction_counts
            == EXPECTED_STRATUM_PREDICTION_COUNTS[stratum_index],
            "fold_counts_exact": fold_counts == EXPECTED_STRATUM_FOLD_COUNTS[stratum_index],
        }
        failed = [key for key, value in checks.items() if not value]
        if failed:
            raise ValueError(f"Boundary stratum {stratum_index} differs: {failed}")
        indices.append(group_indices)
        summaries.append(
            {
                "stratum": stratum_index,
                "rows": len(group),
                "ordered_index_sha256": index_sha256,
                "prediction_counts": prediction_counts,
                "fold_counts": fold_counts,
                "margin_min": min(margins),
                "margin_max": max(margins),
                "margin_mean": sum(margins) / len(margins),
                "checks": checks,
            }
        )
    return indices, summaries


def _read_prior_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
) -> Dict[str, Dict[str, list[Dict[str, object]]]]:
    result = {
        condition: {name: [] for name in ("raw", "ce_control", "ce_agem")}
        for condition in CONDITIONS
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "fold",
            "target",
        }
        for source_name in ("raw", "control", "candidate"):
            required.add(f"{source_name}_prediction")
            required.update(f"{source_name}_prob_{index}" for index in range(5))
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prior A-GEM prediction CSV is missing: {sorted(missing)}")
        name_map = {"raw": "raw", "control": "ce_control", "candidate": "ce_agem"}
        for raw in reader:
            condition = str(raw["condition"]).strip()
            if condition not in result:
                raise ValueError(f"Unexpected prior A-GEM condition: {condition}")
            sample_index = int(raw["sample_index"])
            if not 0 <= sample_index < len(rows):
                raise ValueError("Prior A-GEM sample index is out of range.")
            source = rows[sample_index]
            if (
                source.source_stem != str(raw["source_stem"]).strip().casefold()
                or source.image_path != Path(str(raw["image_path"])).resolve()
                or source.fold != int(raw["fold"])
                or source.target != int(raw["target"])
            ):
                raise ValueError(f"Prior A-GEM source row mismatch at {sample_index}.")
            for source_name, output_name in name_map.items():
                probabilities = [
                    float(raw[f"{source_name}_prob_{index}"]) for index in range(5)
                ]
                if not all(math.isfinite(value) and value >= 0.0 for value in probabilities):
                    raise ValueError("Prior A-GEM probabilities are invalid.")
                if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
                    raise ValueError("Prior A-GEM probabilities are not normalized.")
                row: Dict[str, object] = {
                    "sample_index": sample_index,
                    "target": source.target,
                    "prediction": int(raw[f"{source_name}_prediction"]),
                }
                for class_index, value in enumerate(probabilities):
                    row[f"prob_{class_index}"] = value
                result[condition][output_name].append(row)
    for condition, variants in result.items():
        expected = None
        for name, values in variants.items():
            values.sort(key=lambda row: int(row["sample_index"]))
            indices = [int(row["sample_index"]) for row in values]
            if len(indices) != EXPECTED_HOLDOUT_ROWS or len(set(indices)) != len(indices):
                raise ValueError(f"Prior {condition}/{name} row contract differs.")
            if expected is None:
                expected = indices
            elif indices != expected:
                raise ValueError(f"Prior {condition} variant indices differ.")
    return result


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted boundary-GEM protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"),
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
        "prior_agem_summary": _verify_sha256(
            paths["prior_agem_summary"],
            LOCKED_PRIOR_AGEM_SUMMARY_SHA256,
            "prior A-GEM summary",
        ),
        "prior_agem_predictions": _verify_sha256(
            paths["prior_agem_predictions"],
            LOCKED_PRIOR_AGEM_PREDICTIONS_SHA256,
            "prior A-GEM predictions",
        ),
        "prior_agem_manifest": _verify_sha256(
            paths["prior_agem_manifest"],
            LOCKED_PRIOR_AGEM_MANIFEST_SHA256,
            "prior A-GEM manifest",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "boundary-GEM protocol"
        ),
        "gem_paper": _verify_sha256(
            paths["gem_paper"], LOCKED_GEM_PAPER_SHA256, "GEM paper"
        ),
        "gem_source": _verify_sha256(
            paths["gem_source"], LOCKED_GEM_SOURCE_SHA256, "official GEM source"
        ),
        "gem_license": _verify_sha256(
            paths["gem_license"], LOCKED_GEM_LICENSE_SHA256, "official GEM license"
        ),
    }
    gem_root = Path(args.official_gem_root).resolve()
    gem_commit = _git_commit(gem_root)
    gem_clean = _git_worktree_clean(gem_root)
    if gem_commit != LOCKED_GEM_COMMIT:
        raise ValueError(f"Official GEM commit differs: {gem_commit} != {LOCKED_GEM_COMMIT}")
    if not gem_clean:
        raise ValueError("Official GEM worktree is not clean.")
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    prior_summary = json.loads(paths["prior_agem_summary"].read_text(encoding="utf-8"))
    for name, payload in (("CIDT", cidt_summary), ("prior A-GEM", prior_summary)):
        if bool(payload.get("test_data_used", True)):
            raise ValueError(f"{name} provenance indicates test data use.")
        if bool(payload.get("validation_predictions_used", True)):
            raise ValueError(f"{name} provenance indicates validation prediction use.")
    if bool(prior_summary.get("binary_model_artifacts_written", True)):
        raise ValueError("Prior A-GEM summary indicates a binary model artifact.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    _, strata_summary = _locked_strata_summary(rows, fold=int(args.fold))
    prior_predictions = _read_prior_predictions(paths["prior_agem_predictions"], rows=rows)
    clean_raw = prior_predictions["clean"]["raw"]
    prior_indices = [int(row["sample_index"]) for row in clean_raw]
    if prior_indices != list(cohorts["holdout_indices"]):
        raise ValueError("Prior A-GEM rows differ from the locked fold-0 holdout.")
    raw_cidt_mismatches = sum(
        int(row["prediction"]) != rows[int(row["sample_index"])].keeper_prediction
        for row in clean_raw
    )
    if raw_cidt_mismatches:
        raise ValueError("Prior raw A-GEM predictions differ from CIDT.")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "official_gem_commit": gem_commit,
        "official_gem_worktree_clean": gem_clean,
        "cohort": _cohort_serializable(cohorts),
        "strata": strata_summary,
        "prior_prediction_conditions": {
            condition: len(values["raw"])
            for condition, values in prior_predictions.items()
        },
        "prior_raw_cidt_mismatches": raw_cidt_mismatches,
        "validation_predictions_used": False,
        "test_data_used": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _class1_margin_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 2 or int(logits.size(1)) != 5:
        raise ValueError("Class-1 margin loss requires logits [N,5].")
    if not bool(targets.eq(FOCUS_CLASS).all().item()):
        raise ValueError("Class-1 margin memory contains a non-class1 target.")
    nonfocus = logits[:, [0, 2, 3, 4]].amax(dim=1)
    margin = logits[:, FOCUS_CLASS] - nonfocus
    return F.softplus(-margin).sum()


def _mean_gradient(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_rows: int,
    cohort_name: str,
    loss_kind: str,
) -> tuple[Gradient, Dict[str, object]]:
    model.eval()
    parameters = [value for value in model.parameters() if value.requires_grad]
    accumulators = [
        torch.zeros(tuple(value.shape), dtype=torch.float64, device="cpu")
        for value in parameters
    ]
    rows = 0
    batches = 0
    loss_total = 0.0
    missing_counts = [0 for _ in parameters]
    for images, targets, metadata in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits, _ = _forward_classification_with_metadata(
            model, images, metadata, device=device
        )
        logits = logits.float()
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError(f"{cohort_name} logits are nonfinite.")
        if loss_kind == "multiclass_ce":
            loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        elif loss_kind == "class1_decision_margin":
            loss_sum = _class1_margin_loss(logits, targets)
        else:
            raise ValueError(f"Unsupported gradient loss: {loss_kind}")
        if not bool(torch.isfinite(loss_sum).item()):
            raise ValueError(f"{cohort_name} loss is nonfinite.")
        gradients = torch.autograd.grad(
            loss_sum,
            parameters,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        for index, gradient in enumerate(gradients):
            if gradient is None:
                missing_counts[index] += 1
                continue
            if not bool(torch.isfinite(gradient).all().item()):
                raise ValueError(f"{cohort_name} gradient is nonfinite.")
            accumulators[index].add_(gradient.detach().cpu().double())
        rows += int(targets.numel())
        batches += 1
        loss_total += float(loss_sum.detach().cpu().item())
    if rows != int(expected_rows):
        raise ValueError(f"{cohort_name} rows differ: {rows} != {expected_rows}")
    gradients = [value.div(float(rows)).float() for value in accumulators]
    norm_squared = sum(
        float(value.detach().double().square().sum().item()) for value in gradients
    )
    finite = all(bool(torch.isfinite(value).all().item()) for value in gradients)
    return gradients, {
        "cohort": cohort_name,
        "rows": rows,
        "batches": batches,
        "loss_kind": loss_kind,
        "row_mean_loss": loss_total / float(rows),
        "gradient_norm": math.sqrt(norm_squared),
        "gradient_finite": finite,
        "parameters": len(parameters),
        "parameters_missing_every_batch": sum(count == batches for count in missing_counts),
        "parameters_missing_some_batches": sum(count > 0 for count in missing_counts),
    }


def _gradient_dot(left: Sequence[Tensor], right: Sequence[Tensor]) -> float:
    if len(left) != len(right):
        raise ValueError("Gradient list lengths differ.")
    value = torch.zeros((), dtype=torch.float64)
    for left_value, right_value in zip(left, right):
        if left_value.shape != right_value.shape:
            raise ValueError("Gradient tensor shapes differ.")
        value += (left_value.detach().cpu().double() * right_value.detach().cpu().double()).sum()
    return float(value.item())


def _gradient_norm(values: Sequence[Tensor]) -> float:
    return math.sqrt(max(0.0, _gradient_dot(values, values)))


def _average_gradients(memories: Sequence[Sequence[Tensor]]) -> Gradient:
    if not memories:
        raise ValueError("Cannot average an empty gradient memory.")
    count = len(memories)
    result = []
    for tensors in zip(*memories):
        shape = tensors[0].shape
        if any(value.shape != shape for value in tensors):
            raise ValueError("Memory gradient schemas differ.")
        accumulated = torch.zeros(shape, dtype=torch.float64)
        for value in tensors:
            accumulated.add_(value.detach().cpu().double())
        result.append(accumulated.div(float(count)).float())
    return result


def solve_nonnegative_dual_active_set(
    gram: np.ndarray,
    linear: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> Dict[str, object]:
    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(linear, dtype=np.float64).reshape(-1)
    if matrix.shape != (vector.size, vector.size) or vector.size == 0:
        raise ValueError("GEM Gram/linear shapes are invalid.")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("GEM QP inputs must be finite.")
    matrix = 0.5 * (matrix + matrix.T)
    candidates = []
    dimensions = int(vector.size)
    for mask in range(1 << dimensions):
        active = [index for index in range(dimensions) if mask & (1 << index)]
        dual = np.zeros(dimensions, dtype=np.float64)
        if active:
            active_matrix = matrix[np.ix_(active, active)]
            active_linear = vector[active]
            solution, _, _, _ = np.linalg.lstsq(
                active_matrix,
                -active_linear,
                rcond=None,
            )
            if np.max(np.abs(active_matrix @ solution + active_linear)) > tolerance:
                continue
            if np.min(solution) < -tolerance:
                continue
            dual[active] = np.maximum(solution, 0.0)
        dual_gradient = matrix @ dual + vector
        inactive = [index for index in range(dimensions) if index not in active]
        active_residual = (
            float(np.max(np.abs(dual_gradient[active]))) if active else 0.0
        )
        inactive_min = (
            float(np.min(dual_gradient[inactive])) if inactive else math.inf
        )
        if active_residual > tolerance or inactive_min < -tolerance:
            continue
        objective = float(0.5 * dual @ matrix @ dual + vector @ dual)
        candidates.append(
            {
                "mask": mask,
                "active": active,
                "dual": dual,
                "dual_gradient": dual_gradient,
                "objective": objective,
                "active_stationarity_max_abs": active_residual,
                "inactive_gradient_min": inactive_min,
            }
        )
    if not candidates:
        raise RuntimeError("No KKT-feasible active set found for GEM QP.")
    candidates.sort(key=lambda row: (float(row["objective"]), int(row["mask"])))
    selected = candidates[0]
    return {
        **selected,
        "candidate_count": len(candidates),
        "tolerance": tolerance,
    }


def _scipy_dual_reference(gram: np.ndarray, linear: np.ndarray) -> Dict[str, object]:
    matrix = np.asarray(gram, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    vector = np.asarray(linear, dtype=np.float64).reshape(-1)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, 0.0, None)
    factor = np.diag(np.sqrt(clipped)) @ eigenvectors.T
    offset, _, _, _ = np.linalg.lstsq(factor.T, vector, rcond=None)
    dual, residual_norm = nnls(factor, -offset, maxiter=100 * int(vector.size))
    dual_gradient = matrix @ dual + vector
    return {
        "dual": dual,
        "residual_norm": float(residual_norm),
        "dual_gradient": dual_gradient,
        "objective": float(0.5 * dual @ matrix @ dual + vector @ dual),
    }


def project_gem_gradient(
    current: Sequence[Tensor],
    memories: Sequence[Sequence[Tensor]],
) -> tuple[Gradient, Dict[str, object]]:
    if not current or len(memories) != STRATUM_COUNT:
        raise ValueError("Boundary GEM requires one current and four memory gradients.")
    if any(len(memory) != len(current) for memory in memories):
        raise ValueError("GEM current/memory gradient schemas differ.")
    current_norm = _gradient_norm(current)
    memory_norms = [_gradient_norm(memory) for memory in memories]
    if current_norm <= 0.0 or any(value <= 0.0 for value in memory_norms):
        raise ValueError("GEM gradients must have nonzero norm.")
    gram = np.asarray(
        [[_gradient_dot(left, right) for right in memories] for left in memories],
        dtype=np.float64,
    )
    linear = np.asarray([_gradient_dot(current, memory) for memory in memories])
    solution = solve_nonnegative_dual_active_set(gram, linear)
    scipy_reference = _scipy_dual_reference(gram, linear)
    dual = np.asarray(solution["dual"], dtype=np.float64)

    projected: Gradient = []
    for parameter_index, current_value in enumerate(current):
        value = current_value.detach().cpu().double().clone()
        for memory_index, coefficient in enumerate(dual):
            value.add_(
                memories[memory_index][parameter_index].detach().cpu().double(),
                alpha=float(coefficient),
            )
        projected.append(value.to(dtype=current_value.dtype))

    materialized_dots = np.asarray(
        [_gradient_dot(projected, memory) for memory in memories], dtype=np.float64
    )
    projected_norm = _gradient_norm(projected)
    projected_cosines = np.zeros_like(materialized_dots)
    if projected_norm > 0.0:
        projected_cosines = materialized_dots / (
            np.asarray(memory_norms, dtype=np.float64) * projected_norm
        )
    flat_direct = torch.cat([value.detach().cpu().double().reshape(-1) for value in current])
    for memory_index, coefficient in enumerate(dual):
        memory_flat = torch.cat(
            [value.detach().cpu().double().reshape(-1) for value in memories[memory_index]]
        )
        flat_direct.add_(memory_flat, alpha=float(coefficient))
    flat_projected = torch.cat(
        [value.detach().cpu().double().reshape(-1) for value in projected]
    )
    flat_error = float((flat_projected - flat_direct).abs().amax().item())
    scipy_dual = np.asarray(scipy_reference["dual"], dtype=np.float64)
    dual_delta = dual - scipy_dual
    direction_difference_squared = float(dual_delta @ gram @ dual_delta)
    scipy_direction_relative_error = math.sqrt(max(0.0, direction_difference_squared)) / max(
        1e-12, projected_norm
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    return projected, {
        "current_norm": current_norm,
        "memory_norms": memory_norms,
        "gram": gram.tolist(),
        "gram_symmetry_max_abs": float(np.max(np.abs(gram - gram.T))),
        "gram_min_eigenvalue": float(np.min(eigenvalues)),
        "gram_max_eigenvalue": float(np.max(eigenvalues)),
        "raw_dots": linear.tolist(),
        "raw_cosines": (
            linear / (np.asarray(memory_norms, dtype=np.float64) * current_norm)
        ).tolist(),
        "raw_violation_count": int(np.sum(linear < 0.0)),
        "dual": dual.tolist(),
        "active_set": list(solution["active"]),
        "active_set_mask": int(solution["mask"]),
        "active_set_candidate_count": int(solution["candidate_count"]),
        "dual_objective": float(solution["objective"]),
        "active_stationarity_max_abs": float(
            solution["active_stationarity_max_abs"]
        ),
        "inactive_gradient_min": float(solution["inactive_gradient_min"]),
        "projected_norm": projected_norm,
        "projected_retained_norm_ratio": projected_norm / current_norm,
        "projected_dots": materialized_dots.tolist(),
        "projected_cosines": projected_cosines.tolist(),
        "direct_flat_max_abs_error": flat_error,
        "scipy_dual": scipy_dual.tolist(),
        "scipy_dual_objective": float(scipy_reference["objective"]),
        "scipy_direction_relative_error": scipy_direction_relative_error,
        "first_order_reference_loss_changes": (-materialized_dots).tolist(),
        "finite": bool(
            np.isfinite(gram).all()
            and np.isfinite(linear).all()
            and np.isfinite(dual).all()
            and all(bool(torch.isfinite(value).all().item()) for value in projected)
        ),
    }


def _gradient_relative_difference(
    left: Sequence[Tensor], right: Sequence[Tensor]
) -> float:
    difference_squared = 0.0
    for left_value, right_value in zip(left, right):
        difference_squared += float(
            (left_value.detach().cpu().double() - right_value.detach().cpu().double())
            .square()
            .sum()
            .item()
        )
    return math.sqrt(difference_squared) / max(1e-12, _gradient_norm(right))


def _gradient_group_rows(
    names: Sequence[str],
    hard: Sequence[Tensor],
    memories: Sequence[Sequence[Tensor]],
    control: Sequence[Tensor],
    candidate: Sequence[Tensor],
) -> list[Dict[str, object]]:
    totals: Dict[str, Dict[str, float]] = {}
    for parameter_index, name in enumerate(names):
        group = _parameter_group(name)
        row = totals.setdefault(
            group,
            {
                "parameters": 0.0,
                "elements": 0.0,
                "hard_squared": 0.0,
                "control_squared": 0.0,
                "candidate_squared": 0.0,
                **{f"memory_{index}_squared": 0.0 for index in range(STRATUM_COUNT)},
                **{f"candidate_memory_{index}_dot": 0.0 for index in range(STRATUM_COUNT)},
            },
        )
        hard_value = hard[parameter_index].double()
        control_value = control[parameter_index].double()
        candidate_value = candidate[parameter_index].double()
        row["parameters"] += 1
        row["elements"] += int(hard_value.numel())
        row["hard_squared"] += float(hard_value.square().sum().item())
        row["control_squared"] += float(control_value.square().sum().item())
        row["candidate_squared"] += float(candidate_value.square().sum().item())
        for memory_index, memory in enumerate(memories):
            memory_value = memory[parameter_index].double()
            row[f"memory_{memory_index}_squared"] += float(
                memory_value.square().sum().item()
            )
            row[f"candidate_memory_{memory_index}_dot"] += float(
                (candidate_value * memory_value).sum().item()
            )
    output = []
    for group, values in sorted(totals.items()):
        row: Dict[str, object] = {
            "group": group,
            "parameters": int(values["parameters"]),
            "elements": int(values["elements"]),
            "hard_norm": math.sqrt(values["hard_squared"]),
            "control_norm": math.sqrt(values["control_squared"]),
            "candidate_norm": math.sqrt(values["candidate_squared"]),
        }
        for memory_index in range(STRATUM_COUNT):
            row[f"memory_{memory_index}_norm"] = math.sqrt(
                values[f"memory_{memory_index}_squared"]
            )
            row[f"candidate_memory_{memory_index}_dot"] = values[
                f"candidate_memory_{memory_index}_dot"
            ]
        output.append(row)
    return output


def _aligned_comparison(
    *,
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    control_indices = [int(row["sample_index"]) for row in control_rows]
    candidate_indices = [int(row["sample_index"]) for row in candidate_rows]
    if control_indices != candidate_indices:
        raise ValueError("Comparison sample-index order differs.")
    return _comparison(
        control_rows=control_rows,
        candidate_rows=candidate_rows,
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    clean_raw_candidate: Mapping[str, object],
    clean_raw_control: Mapping[str, object],
    clean_control_candidate: Mapping[str, object],
    clean_prior_agem_candidate: Mapping[str, object],
    illumination_raw_candidate: Sequence[Mapping[str, object]],
    illumination_prior_agem_candidate: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    delta = clean_raw_candidate["delta"]
    transitions = clean_raw_candidate["transitions"]
    raw_support = int(clean_raw_candidate["control"]["predicted_support"][FOCUS_CLASS])
    candidate_support = int(
        clean_raw_candidate["candidate"]["predicted_support"][FOCUS_CLASS]
    )
    clean_checks = {
        "macro_f1_delta_nonnegative": float(delta["macro_f1"]) >= 0.0,
        "class1_f1_delta_gte_0p005": float(delta["class1_f1"]) >= 0.005,
        "class1_precision_delta_gte_0p005": float(delta["class1_precision"]) >= 0.005,
        "class1_recall_delta_gte_minus_0p005": float(delta["class1_recall"]) >= -0.005,
        "restricted_focus_fp_reduction_gte_4": int(
            transitions["restricted_focus_fp_reduction"]
        )
        >= 4,
        "focus_rescues_gte_tp_breaks": int(transitions["focus_fn_rescue"])
        >= int(transitions["focus_tp_break"]),
        "corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "max_nonfocus_f1_drop_lte_0p010": float(
            clean_raw_candidate["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "class1_support_at_least_95pct_raw": candidate_support
        >= int(math.ceil(0.95 * raw_support)),
        "candidate_f1_exceeds_aggregate_control": float(
            clean_control_candidate["delta"]["class1_f1"]
        )
        > 0.0,
        "candidate_recall_not_lower_than_aggregate_control": float(
            clean_control_candidate["delta"]["class1_recall"]
        )
        >= 0.0,
        "candidate_tp_breaks_no_greater_than_aggregate_control": int(
            transitions["focus_tp_break"]
        )
        <= int(clean_raw_control["transitions"]["focus_tp_break"]),
        "candidate_f1_gain_vs_prior_ce_agem_gte_0p005": float(
            clean_prior_agem_candidate["delta"]["class1_f1"]
        )
        >= 0.005,
    }
    precision_nonnegative = sum(
        float(row["delta"]["class1_precision"]) >= 0.0
        for row in illumination_raw_candidate
    )
    prior_by_condition = {
        str(row["condition"]): row for row in illumination_prior_agem_candidate
    }
    low_contrast_gain = float(
        prior_by_condition["low_contrast"]["delta"]["class1_f1"]
    )
    illumination_checks = {
        "all_class1_f1_deltas_gte_minus_0p010": all(
            float(row["delta"]["class1_f1"]) >= -0.010
            for row in illumination_raw_candidate
        ),
        "all_class1_recall_deltas_gte_minus_0p015": all(
            float(row["delta"]["class1_recall"]) >= -0.015
            for row in illumination_raw_candidate
        ),
        "precision_nonnegative_in_at_least_two_conditions": precision_nonnegative >= 2,
        "aggregate_rescues_gte_tp_breaks": sum(
            int(row["transitions"]["focus_fn_rescue"])
            for row in illumination_raw_candidate
        )
        >= sum(
            int(row["transitions"]["focus_tp_break"])
            for row in illumination_raw_candidate
        ),
        "no_condition_increases_restricted_focus_fp": all(
            int(row["transitions"]["restricted_focus_fp_reduction"]) >= 0
            for row in illumination_raw_candidate
        ),
        "low_contrast_f1_gain_vs_prior_ce_agem_gte_0p020": low_contrast_gain
        >= 0.020,
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **clean_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "validation_authorized": False,
        "full_train_authorized": False,
    }


def _write_predictions(
    path: Path,
    *,
    source_rows: Sequence[CleanTrainRow],
    prior: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    current: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    variants = ("raw", "prior_ce_control", "prior_ce_agem", "margin_agem", "gem")
    fieldnames = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for variant in variants:
        fieldnames.append(f"{variant}_prediction")
        fieldnames.extend(f"{variant}_prob_{index}" for index in range(5))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITIONS:
            combined = {
                "raw": prior[condition]["raw"],
                "prior_ce_control": prior[condition]["ce_control"],
                "prior_ce_agem": prior[condition]["ce_agem"],
                "margin_agem": current[condition]["margin_agem"],
                "gem": current[condition]["gem"],
            }
            indexed = {
                name: {int(row["sample_index"]): row for row in values}
                for name, values in combined.items()
            }
            expected = set(indexed["raw"])
            if any(set(values) != expected for values in indexed.values()):
                raise ValueError(f"Prediction index mismatch for {condition}.")
            for sample_index in sorted(expected):
                source = source_rows[sample_index]
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source.source_stem,
                    "image_path": str(source.image_path),
                    "fold": source.fold,
                    "target": source.target,
                }
                for variant, values in indexed.items():
                    value = values[sample_index]
                    output[f"{variant}_prediction"] = int(value["prediction"])
                    for class_index in range(5):
                        output[f"{variant}_prob_{class_index}"] = value[
                            f"prob_{class_index}"
                        ]
                writer.writerow(output)


def _write_gradient_groups(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Gradient group table is empty.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    gate = summary["gate"]
    clean = summary["comparisons"]["clean_raw_gem"]
    geometry = summary["gem_geometry"]
    lines = [
        "# Class-1 Boundary-Stratified GEM Result",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Raw violations: `{geometry['raw_violation_count']}/4`",
        f"- GEM active set: `{geometry['active_set']}`",
        f"- GEM retained norm: `{geometry['projected_retained_norm_ratio']:.6f}`",
        f"- Clean macro F1 delta: `{clean['delta']['macro_f1']:+.6f}`",
        f"- Clean class-1 F1 delta: `{clean['delta']['class1_f1']:+.6f}`",
        f"- Clean class-1 precision delta: `{clean['delta']['class1_precision']:+.6f}`",
        f"- Clean class-1 recall delta: `{clean['delta']['class1_recall']:+.6f}`",
        f"- Restricted FP reduction: `{clean['transitions']['restricted_focus_fp_reduction']}`",
        "",
        "Only source-disjoint `yolo_f/train` rows were used; validation and test were not accessed.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked boundary-GEM A0 requires CUDA.")
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    if _is_relative_to(output_path, paths["data"].parent):
        raise ValueError("Audit output cannot be written inside the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    stratum_indices, strata_summary = _locked_strata_summary(rows, fold=int(args.fold))
    prior_predictions = _read_prior_predictions(paths["prior_agem_predictions"], rows=rows)

    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    if len(class_names) != 5 or not isinstance(model_config, Mapping) or not isinstance(
        model_state, Mapping
    ):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    set_seed(int(args.seed), deterministic=True)
    prototype = create_model(num_classes=5, model_config=model_config).eval()
    load_model_state(prototype, dict(model_state), strict=True)
    initial_state_sha256 = _state_sha256(prototype)
    parameter_names, parameter_schema_sha256 = _parameter_schema(prototype)
    trainable_parameter_count = sum(
        int(value.numel()) for value in prototype.parameters() if value.requires_grad
    )

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
    sample_paths_exact = len(sample_paths) == len(rows) and all(
        source.image_path == observed for source, observed in zip(rows, sample_paths)
    )
    train_paths_only = all(
        "train" in {part.casefold() for part in path.parts}
        and "val" not in {part.casefold() for part in path.parts}
        and "test" not in {part.casefold() for part in path.parts}
        for path in sample_paths
    )
    if not sample_paths_exact or not train_paths_only:
        raise ValueError("Dataset paths violate the locked train-only order.")
    transform = _build_eval_transform(semantics)

    gradient_model = copy.deepcopy(prototype).to(device).eval()
    hard_loader, hard_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=cohorts["hard_indices"],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="class1_boundary_gem_hard_gradient",
        seed=int(args.seed) + 10,
    )
    hard_gradient, hard_summary = _mean_gradient(
        model=gradient_model,
        loader=hard_loader,
        device=device,
        expected_rows=EXPECTED_HARD_ROWS,
        cohort_name="restricted_hard_negatives",
        loss_kind="multiclass_ce",
    )
    del hard_loader
    memory_gradients: list[Gradient] = []
    memory_summaries = []
    memory_loader_summaries = []
    for stratum_index, indices in enumerate(stratum_indices):
        loader, loader_summary = _make_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"class1_boundary_gem_memory_{stratum_index}",
            seed=int(args.seed) + 20 + stratum_index,
        )
        gradient, gradient_summary = _mean_gradient(
            model=gradient_model,
            loader=loader,
            device=device,
            expected_rows=STRATUM_ROWS,
            cohort_name=f"class1_boundary_stratum_{stratum_index}",
            loss_kind="class1_decision_margin",
        )
        memory_gradients.append(gradient)
        memory_summaries.append(gradient_summary)
        memory_loader_summaries.append(loader_summary)
        del loader
    del gradient_model
    gc.collect()
    torch.cuda.empty_cache()

    aggregate_memory = _average_gradients(memory_gradients)
    margin_agem_direction, margin_agem_geometry = project_agem_gradient(
        hard_gradient, aggregate_memory
    )
    gem_direction, gem_geometry = project_gem_gradient(
        hard_gradient, memory_gradients
    )
    direction_relative_difference = _gradient_relative_difference(
        gem_direction, margin_agem_direction
    )
    gradient_group_rows = _gradient_group_rows(
        parameter_names,
        hard_gradient,
        memory_gradients,
        margin_agem_direction,
        gem_direction,
    )

    margin_agem_model = copy.deepcopy(prototype).eval()
    gem_model = copy.deepcopy(prototype).eval()
    margin_agem_initial_sha256 = _state_sha256(margin_agem_model)
    gem_initial_sha256 = _state_sha256(gem_model)
    _, margin_agem_schema_sha256 = _parameter_schema(margin_agem_model)
    _, gem_schema_sha256 = _parameter_schema(gem_model)
    margin_agem_step = _apply_normalized_parameter_step(
        margin_agem_model,
        margin_agem_direction,
        parameter_ratio=float(args.step_ratio),
    )
    gem_step = _apply_normalized_parameter_step(
        gem_model,
        gem_direction,
        parameter_ratio=float(args.step_ratio),
    )

    models = {"margin_agem": margin_agem_model, "gem": gem_model}
    current_predictions: Dict[str, Dict[str, list[Dict[str, object]]]] = {}
    loader_summaries: Dict[str, object] = {}
    conditions = [("clean", None, None), *LIGHTING_CONDITIONS]
    for condition_index, (condition, brightness, contrast) in enumerate(conditions):
        predictions, loader_summary = _evaluate_condition(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            models=models,
            base_dataset=base_dataset,
            transform=transform,
            holdout_indices=cohorts["holdout_indices"],
            args=args,
            device=device,
            seed_offset=100 + condition_index,
        )
        current_predictions[condition] = predictions
        loader_summaries[condition] = loader_summary

    clean_prior = prior_predictions["clean"]
    clean_current = current_predictions["clean"]
    clean_raw_gem = _aligned_comparison(
        control_rows=clean_prior["raw"],
        candidate_rows=clean_current["gem"],
    )
    clean_raw_control = _aligned_comparison(
        control_rows=clean_prior["raw"],
        candidate_rows=clean_current["margin_agem"],
    )
    clean_control_gem = _aligned_comparison(
        control_rows=clean_current["margin_agem"],
        candidate_rows=clean_current["gem"],
    )
    clean_prior_agem_gem = _aligned_comparison(
        control_rows=clean_prior["ce_agem"],
        candidate_rows=clean_current["gem"],
    )
    illumination_raw_gem = []
    illumination_raw_control = []
    illumination_prior_agem_gem = []
    for condition, _, _ in LIGHTING_CONDITIONS:
        raw_gem = _aligned_comparison(
            control_rows=prior_predictions[condition]["raw"],
            candidate_rows=current_predictions[condition]["gem"],
        )
        raw_gem["condition"] = condition
        illumination_raw_gem.append(raw_gem)
        raw_control = _aligned_comparison(
            control_rows=prior_predictions[condition]["raw"],
            candidate_rows=current_predictions[condition]["margin_agem"],
        )
        raw_control["condition"] = condition
        illumination_raw_control.append(raw_control)
        prior_agem_gem = _aligned_comparison(
            control_rows=prior_predictions[condition]["ce_agem"],
            candidate_rows=current_predictions[condition]["gem"],
        )
        prior_agem_gem["condition"] = condition
        illumination_prior_agem_gem.append(prior_agem_gem)

    raw_cidt_mismatches = sum(
        int(row["prediction"]) != rows[int(row["sample_index"])].keeper_prediction
        for row in clean_prior["raw"]
    )
    step_norm_mismatch = abs(
        margin_agem_step["actual_step_norm"] - gem_step["actual_step_norm"]
    ) / max(1e-12, margin_agem_step["actual_step_norm"])
    raw_dots = [float(value) for value in gem_geometry["raw_dots"]]
    projected_dots = [float(value) for value in gem_geometry["projected_dots"]]
    all_gradient_summaries = [hard_summary, *memory_summaries]
    structural_checks = {
        "locked_sources_verified": True,
        "official_worktree_clean": bool(provenance["official_gem_worktree_clean"]),
        "train_only_provenance": not provenance["validation_predictions_used"]
        and not provenance["test_data_used"],
        "source_groups_disjoint": not cohorts["source_overlap"],
        "cohort_contract_exact": all(cohorts["checks"].values()),
        "strata_contract_exact": all(
            all(summary["checks"].values()) for summary in strata_summary
        ),
        "sample_paths_exact": sample_paths_exact,
        "train_paths_only": train_paths_only,
        "prior_raw_predictions_match_cidt_1843_of_1843": raw_cidt_mismatches == 0,
        "all_gradients_finite": all(
            bool(summary["gradient_finite"]) for summary in all_gradient_summaries
        )
        and bool(gem_geometry["finite"]),
        "all_gradients_nonzero": all(
            float(summary["gradient_norm"]) > 0.0 for summary in all_gradient_summaries
        ),
        "gram_symmetric": float(gem_geometry["gram_symmetry_max_abs"]) <= 1e-12,
        "gram_psd_within_1e_minus_8": float(gem_geometry["gram_min_eigenvalue"])
        >= -1e-8,
        "at_least_two_raw_constraint_violations": int(
            gem_geometry["raw_violation_count"]
        )
        >= 2,
        "stratum0_raw_constraint_violated": raw_dots[0] < 0.0,
        "dual_nonnegative": min(float(value) for value in gem_geometry["dual"])
        >= -1e-10,
        "active_stationarity_lte_1e_minus_8": float(
            gem_geometry["active_stationarity_max_abs"]
        )
        <= 1e-8,
        "inactive_gradient_gte_minus_1e_minus_8": math.isinf(
            float(gem_geometry["inactive_gradient_min"])
        )
        or float(gem_geometry["inactive_gradient_min"]) >= -1e-8,
        "all_projected_dots_gte_minus_1e_minus_7": min(projected_dots) >= -1e-7,
        "flattened_equation_error_lte_1e_minus_7": float(
            gem_geometry["direct_flat_max_abs_error"]
        )
        <= 1e-7,
        "active_set_scipy_direction_error_lte_1e_minus_7": float(
            gem_geometry["scipy_direction_relative_error"]
        )
        <= 1e-7,
        "gem_retained_norm_gte_0p20": float(
            gem_geometry["projected_retained_norm_ratio"]
        )
        >= 0.20,
        "gem_differs_from_aggregate_agem": direction_relative_difference >= 1e-6,
        "initial_states_identical": initial_state_sha256
        == margin_agem_initial_sha256
        == gem_initial_sha256,
        "parameter_schemas_identical": parameter_schema_sha256
        == margin_agem_schema_sha256
        == gem_schema_sha256,
        "all_trainable_parameters_included": len(parameter_names)
        == len(hard_gradient)
        == len(gem_direction)
        == len(margin_agem_direction),
        "control_update_ratio_exact": abs(
            margin_agem_step["actual_ratio"] - STEP_RATIO
        )
        <= 1e-8,
        "candidate_update_ratio_exact": abs(gem_step["actual_ratio"] - STEP_RATIO)
        <= 1e-8,
        "matched_update_norms": step_norm_mismatch <= 1e-7,
        "all_first_order_memory_losses_nonincreasing": all(
            float(value) <= 1e-7
            for value in gem_geometry["first_order_reference_loss_changes"]
        ),
        "no_validation_or_test_access": True,
        "no_raw_data_write": not _is_relative_to(output_dir, paths["data"].parent),
        "no_binary_model_artifact": True,
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        clean_raw_candidate=clean_raw_gem,
        clean_raw_control=clean_raw_control,
        clean_control_candidate=clean_control_gem,
        clean_prior_agem_candidate=clean_prior_agem_gem,
        illumination_raw_candidate=illumination_raw_gem,
        illumination_prior_agem_candidate=illumination_prior_agem_gem,
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    gradient_groups_path = output_dir / "gradient_groups.csv"
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "artifact_manifest.json"
    _write_predictions(
        predictions_path,
        source_rows=rows,
        prior=prior_predictions,
        current=current_predictions,
    )
    _write_gradient_groups(gradient_groups_path, gradient_group_rows)
    summary: Dict[str, object] = {
        "protocol": "class1_boundary_stratified_margin_gem_a0",
        "status": "passed" if gate["all_gates_passed"] else "closed",
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "strata": strata_summary,
        "model": {
            "class_names": class_names,
            "trainable_parameter_tensors": len(parameter_names),
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_schema_sha256": parameter_schema_sha256,
            "initial_state_sha256": initial_state_sha256,
            "margin_agem_state_sha256": _state_sha256(margin_agem_model),
            "gem_state_sha256": _state_sha256(gem_model),
        },
        "gradient_computation": {
            "hard": hard_summary,
            "memories": memory_summaries,
            "hard_loader": hard_loader_summary,
            "memory_loaders": memory_loader_summaries,
            "accumulation_dtype": "float64",
            "materialized_dtype": "float32",
            "model_mode": "eval",
        },
        "aggregate_margin_agem_geometry": margin_agem_geometry,
        "gem_geometry": gem_geometry,
        "gem_vs_aggregate_agem_direction_relative_l2": direction_relative_difference,
        "steps": {
            "margin_agem": margin_agem_step,
            "gem": gem_step,
            "update_norm_mismatch_relative": step_norm_mismatch,
        },
        "comparisons": {
            "clean_raw_gem": clean_raw_gem,
            "clean_raw_margin_agem": clean_raw_control,
            "clean_margin_agem_gem": clean_control_gem,
            "clean_prior_ce_agem_gem": clean_prior_agem_gem,
            "illumination_raw_gem": illumination_raw_gem,
            "illumination_raw_margin_agem": illumination_raw_control,
            "illumination_prior_ce_agem_gem": illumination_prior_agem_gem,
        },
        "prior_raw_cidt_mismatches": raw_cidt_mismatches,
        "loader_summaries": loader_summaries,
        "gate": gate,
        "validation_predictions_used": False,
        "test_data_used": False,
        "binary_model_artifacts_written": False,
        "artifacts": {
            "predictions": str(predictions_path.resolve()),
            "gradient_groups": str(gradient_groups_path.resolve()),
            "report": str(report_path.resolve()),
            "summary": str(summary_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    _write_report(report_path, summary)
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(
        manifest_path,
        [predictions_path, gradient_groups_path, report_path, summary_path],
    )
    summary["artifact_manifest"] = manifest
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        result = _verify_sources(args)
        result["preflight_only"] = True
        result["output_dir_created"] = False
        print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
        return
    result = run_audit(args)
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "status": result["status"],
                "all_gates_passed": result["gate"]["all_gates_passed"],
                "stage_b_authorized": result["gate"]["stage_b_authorized"],
                "failed_checks": result["gate"]["failed_checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

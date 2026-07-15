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
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
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

EXPECTED_TRAIN_ROWS = 9215
EXPECTED_SOURCE_GROUPS = 8064
EXPECTED_FIT_ROWS = 7372
EXPECTED_HOLDOUT_ROWS = 1843
EXPECTED_REFERENCE_ROWS = 432
EXPECTED_HARD_ROWS = 186
EXPECTED_HARD_CLASS_COUNTS = {0: 135, 2: 42, 4: 9}
EXPECTED_REFERENCE_FOLD_COUNTS = {1: 115, 2: 104, 3: 103, 4: 110}
EXPECTED_HARD_FOLD_COUNTS = {1: 45, 2: 48, 3: 52, 4: 41}
EXPECTED_HOLDOUT_RESTRICTED_FP = 36
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
STEP_RATIO = 1e-4

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "4bec6a835ce8a5bfb7ebe6a31ca533cf596906fe60279903e05198d0c9010968"
LOCKED_AGEM_PAPER_SHA256 = "79372cc7f08e80acf257e7d60e0357b883a5927f6fb94e260bd32017e270c1a9"
LOCKED_AGEM_COMMIT = "45421499483b28935491251e9e821c55e8b3c089"
LOCKED_AGEM_SOURCE_SHA256 = "60ff8ad5e3b35db1700a803c0701269fad12f1e0364a5d2a517e14197391f2c8"
LOCKED_AGEM_LICENSE_SHA256 = "5d6579f2902a45aaeab167ee9e5b16d1a177c4135b833fc3be17545b35d5775f"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only class-1-reference A-GEM information gate. "
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
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CLASS1_REFERENCE_AGEM_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--official-agem-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\agem"),
    )
    parser.add_argument(
        "--agem-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\agem_iclr2019_arxiv1812.00420.pdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_class1_reference_agem_readiness_20260715"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--focus-class", type=int, default=FOCUS_CLASS)
    parser.add_argument("--step-ratio", type=float, default=STEP_RATIO)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == 32
        and int(args.num_workers) == 4
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.focus_class) == FOCUS_CLASS
        and math.isclose(
            float(args.step_ratio), STEP_RATIO, rel_tol=0.0, abs_tol=1e-15
        )
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official_root = Path(args.official_agem_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "agem_paper": Path(args.agem_paper).resolve(),
        "agem_source": official_root / "model" / "model.py",
        "agem_license": official_root / "LICENSE",
    }


def _git_worktree_clean(root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(Path(root).resolve()),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _select_cohorts(
    rows: Sequence[CleanTrainRow],
    *,
    fold: int,
    focus_class: int,
) -> Dict[str, object]:
    fit_indices = [row.sample_index for row in rows if row.fold != int(fold)]
    holdout_indices = [row.sample_index for row in rows if row.fold == int(fold)]
    reference_indices = [
        row.sample_index
        for row in rows
        if row.fold != int(fold) and row.target == int(focus_class)
    ]
    hard_indices = [
        row.sample_index
        for row in rows
        if row.fold != int(fold)
        and row.target in RESTRICTED_NEGATIVE_CLASSES
        and row.keeper_prediction == int(focus_class)
    ]
    fit_sources = {rows[index].source_stem for index in fit_indices}
    holdout_sources = {rows[index].source_stem for index in holdout_indices}
    hard_class_counts = {
        class_index: sum(rows[index].target == class_index for index in hard_indices)
        for class_index in RESTRICTED_NEGATIVE_CLASSES
    }
    reference_fold_counts = {
        candidate_fold: sum(
            rows[index].fold == candidate_fold for index in reference_indices
        )
        for candidate_fold in range(5)
        if candidate_fold != int(fold)
    }
    hard_fold_counts = {
        candidate_fold: sum(rows[index].fold == candidate_fold for index in hard_indices)
        for candidate_fold in range(5)
        if candidate_fold != int(fold)
    }
    holdout_restricted_fp = sum(
        rows[index].target in RESTRICTED_NEGATIVE_CLASSES
        and rows[index].keeper_prediction == int(focus_class)
        for index in holdout_indices
    )
    return {
        "fit_indices": fit_indices,
        "holdout_indices": holdout_indices,
        "reference_indices": reference_indices,
        "hard_indices": hard_indices,
        "fit_source_count": len(fit_sources),
        "holdout_source_count": len(holdout_sources),
        "source_overlap": sorted(fit_sources.intersection(holdout_sources)),
        "hard_class_counts": hard_class_counts,
        "reference_fold_counts": reference_fold_counts,
        "hard_fold_counts": hard_fold_counts,
        "holdout_restricted_fp": int(holdout_restricted_fp),
    }


def _locked_cohort_summary(rows: Sequence[CleanTrainRow], *, fold: int) -> Dict[str, object]:
    cohorts = _select_cohorts(rows, fold=fold, focus_class=FOCUS_CLASS)
    unique_sources = len({row.source_stem for row in rows})
    checks = {
        "train_rows_exact": len(rows) == EXPECTED_TRAIN_ROWS,
        "source_groups_exact": unique_sources == EXPECTED_SOURCE_GROUPS,
        "fit_rows_exact": len(cohorts["fit_indices"]) == EXPECTED_FIT_ROWS,
        "holdout_rows_exact": len(cohorts["holdout_indices"])
        == EXPECTED_HOLDOUT_ROWS,
        "reference_rows_exact": len(cohorts["reference_indices"])
        == EXPECTED_REFERENCE_ROWS,
        "hard_rows_exact": len(cohorts["hard_indices"]) == EXPECTED_HARD_ROWS,
        "hard_class_counts_exact": cohorts["hard_class_counts"]
        == EXPECTED_HARD_CLASS_COUNTS,
        "reference_fold_counts_exact": cohorts["reference_fold_counts"]
        == EXPECTED_REFERENCE_FOLD_COUNTS,
        "hard_fold_counts_exact": cohorts["hard_fold_counts"]
        == EXPECTED_HARD_FOLD_COUNTS,
        "holdout_restricted_fp_exact": int(cohorts["holdout_restricted_fp"])
        == EXPECTED_HOLDOUT_RESTRICTED_FP,
        "source_groups_disjoint": not cohorts["source_overlap"],
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Locked A-GEM cohort contract differs: {failed}")
    return {
        **cohorts,
        "unique_source_groups": unique_sources,
        "fit_index_sha256": _ordered_index_sha256(cohorts["fit_indices"]),
        "holdout_index_sha256": _ordered_index_sha256(cohorts["holdout_indices"]),
        "reference_index_sha256": _ordered_index_sha256(
            cohorts["reference_indices"]
        ),
        "hard_index_sha256": _ordered_index_sha256(cohorts["hard_indices"]),
        "checks": checks,
    }


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted A-GEM protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "A-GEM protocol"
        ),
        "agem_paper": _verify_sha256(
            paths["agem_paper"], LOCKED_AGEM_PAPER_SHA256, "A-GEM paper"
        ),
        "agem_source": _verify_sha256(
            paths["agem_source"], LOCKED_AGEM_SOURCE_SHA256, "official A-GEM source"
        ),
        "agem_license": _verify_sha256(
            paths["agem_license"],
            LOCKED_AGEM_LICENSE_SHA256,
            "official A-GEM license",
        ),
    }
    official_root = Path(args.official_agem_root).resolve()
    official_commit = _git_commit(official_root)
    official_clean = _git_worktree_clean(official_root)
    if official_commit != LOCKED_AGEM_COMMIT:
        raise ValueError(
            f"Official A-GEM commit differs: {official_commit} != {LOCKED_AGEM_COMMIT}"
        )
    if not official_clean:
        raise ValueError("Official A-GEM worktree is not clean.")
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use.")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "official_agem_commit": official_commit,
        "official_agem_worktree_clean": official_clean,
        "cohort": _cohort_serializable(cohorts),
        "validation_predictions_used": False,
        "test_data_used": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _cohort_serializable(cohorts: Mapping[str, object]) -> Dict[str, object]:
    excluded = {"fit_indices", "holdout_indices", "reference_indices", "hard_indices"}
    result = {key: value for key, value in cohorts.items() if key not in excluded}
    result.update(
        {
            "fit_rows": len(cohorts["fit_indices"]),
            "holdout_rows": len(cohorts["holdout_indices"]),
            "reference_rows": len(cohorts["reference_indices"]),
            "hard_rows": len(cohorts["hard_indices"]),
        }
    )
    return result


def _parameter_schema(model: nn.Module) -> tuple[list[str], str]:
    names = []
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        digest.update(
            f"{name}:{tuple(parameter.shape)}:{parameter.dtype}\n".encode("utf-8")
        )
    return names, digest.hexdigest()


def _gradient_stats(values: Sequence[Tensor]) -> Dict[str, float | bool]:
    squared = torch.zeros((), dtype=torch.float64)
    finite = True
    for value in values:
        detached = value.detach().cpu()
        finite = finite and bool(torch.isfinite(detached).all().item())
        squared += detached.double().square().sum()
    return {"norm": float(squared.sqrt().item()), "finite": bool(finite)}


def project_agem_gradient(
    current: Sequence[Tensor],
    reference: Sequence[Tensor],
) -> tuple[Gradient, Dict[str, object]]:
    if len(current) != len(reference) or not current:
        raise ValueError("Current/reference gradient schemas differ or are empty.")
    current64: list[Tensor] = []
    reference64: list[Tensor] = []
    dot = torch.zeros((), dtype=torch.float64)
    current_squared = torch.zeros((), dtype=torch.float64)
    reference_squared = torch.zeros((), dtype=torch.float64)
    for index, (current_value, reference_value) in enumerate(
        zip(current, reference)
    ):
        if current_value.shape != reference_value.shape:
            raise ValueError(f"Gradient shape mismatch at parameter {index}.")
        current_value64 = current_value.detach().cpu().double()
        reference_value64 = reference_value.detach().cpu().double()
        if not bool(torch.isfinite(current_value64).all().item()) or not bool(
            torch.isfinite(reference_value64).all().item()
        ):
            raise ValueError("A-GEM gradients must be finite.")
        current64.append(current_value64)
        reference64.append(reference_value64)
        dot += (current_value64 * reference_value64).sum()
        current_squared += current_value64.square().sum()
        reference_squared += reference_value64.square().sum()
    if float(reference_squared.item()) <= 0.0:
        raise ValueError("A-GEM reference gradient has zero norm.")
    if float(current_squared.item()) <= 0.0:
        raise ValueError("A-GEM current gradient has zero norm.")

    violated = float(dot.item()) < 0.0
    coefficient = dot / reference_squared if violated else torch.zeros_like(dot)
    projected64 = [
        current_value - coefficient * reference_value
        for current_value, reference_value in zip(current64, reference64)
    ]
    projected = [
        value.to(dtype=current_value.dtype)
        for value, current_value in zip(projected64, current)
    ]

    flat_current = torch.cat([value.reshape(-1) for value in current64])
    flat_reference = torch.cat([value.reshape(-1) for value in reference64])
    flat_direct = (
        flat_current - (dot / reference_squared) * flat_reference
        if violated
        else flat_current.clone()
    )
    offset = 0
    equation_error = 0.0
    for tensor_value in projected:
        count = int(tensor_value.numel())
        direct_chunk = flat_direct[offset : offset + count].reshape(tensor_value.shape)
        equation_error = max(
            equation_error,
            float((tensor_value.double() - direct_chunk).abs().amax().item()),
        )
        offset += count
    if offset != int(flat_direct.numel()):
        raise RuntimeError("Flattened A-GEM equation length mismatch.")

    projected_dot = torch.zeros((), dtype=torch.float64)
    projected_squared = torch.zeros((), dtype=torch.float64)
    for projected_value, reference_value in zip(projected, reference64):
        projected_value64 = projected_value.detach().cpu().double()
        projected_dot += (projected_value64 * reference_value).sum()
        projected_squared += projected_value64.square().sum()
    current_norm = current_squared.sqrt()
    reference_norm = reference_squared.sqrt()
    projected_norm = projected_squared.sqrt()
    raw_cosine = dot / (current_norm * reference_norm)
    projected_cosine = projected_dot / (projected_norm * reference_norm)
    return projected, {
        "current_norm": float(current_norm.item()),
        "reference_norm": float(reference_norm.item()),
        "raw_dot": float(dot.item()),
        "raw_cosine": float(raw_cosine.item()),
        "violation_active": bool(violated),
        "projection_coefficient": float(coefficient.item()),
        "projected_norm": float(projected_norm.item()),
        "projected_retained_norm_ratio": float(
            (projected_norm / current_norm).item()
        ),
        "projected_dot": float(projected_dot.item()),
        "projected_cosine": float(projected_cosine.item()),
        "direct_flat_max_abs_error": float(equation_error),
        "control_first_order_reference_loss_change": float(-dot.item()),
        "candidate_first_order_reference_loss_change": float(
            -projected_dot.item()
        ),
        "finite": bool(
            all(bool(torch.isfinite(value).all().item()) for value in projected)
        ),
    }


def _apply_normalized_parameter_step(
    model: nn.Module,
    direction: Sequence[Tensor],
    *,
    parameter_ratio: float,
) -> Dict[str, float]:
    parameters = [value for value in model.parameters() if value.requires_grad]
    if len(parameters) != len(direction) or not parameters:
        raise ValueError("Model/direction parameter schemas differ or are empty.")
    parameter_squared = torch.zeros((), dtype=torch.float64)
    direction_squared = torch.zeros((), dtype=torch.float64)
    originals: list[Tensor] = []
    for index, (parameter, gradient) in enumerate(zip(parameters, direction)):
        if parameter.shape != gradient.shape:
            raise ValueError(f"Step gradient shape mismatch at parameter {index}.")
        original = parameter.detach().cpu().clone()
        gradient_cpu = gradient.detach().cpu()
        if not bool(torch.isfinite(gradient_cpu).all().item()):
            raise ValueError("Normalized parameter step received a nonfinite gradient.")
        originals.append(original)
        parameter_squared += original.double().square().sum()
        direction_squared += gradient_cpu.double().square().sum()
    parameter_norm = parameter_squared.sqrt()
    direction_norm = direction_squared.sqrt()
    if float(parameter_norm.item()) <= 0.0 or float(direction_norm.item()) <= 0.0:
        raise ValueError("Normalized parameter step requires nonzero norms.")
    requested_step_norm = float(parameter_ratio) * parameter_norm
    scale = requested_step_norm / direction_norm
    with torch.no_grad():
        for parameter, gradient in zip(parameters, direction):
            parameter.add_(
                gradient.to(device=parameter.device, dtype=parameter.dtype),
                alpha=-float(scale.item()),
            )
    actual_squared = torch.zeros((), dtype=torch.float64)
    for parameter, original in zip(parameters, originals):
        actual_squared += (
            parameter.detach().cpu().double() - original.double()
        ).square().sum()
    actual_step_norm = actual_squared.sqrt()
    return {
        "parameter_norm": float(parameter_norm.item()),
        "direction_norm": float(direction_norm.item()),
        "scale": float(scale.item()),
        "requested_step_norm": float(requested_step_norm.item()),
        "actual_step_norm": float(actual_step_norm.item()),
        "requested_ratio": float(parameter_ratio),
        "actual_ratio": float((actual_step_norm / parameter_norm).item()),
    }


def _mean_cross_entropy_gradient(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_rows: int,
    cohort_name: str,
) -> tuple[Gradient, Dict[str, object]]:
    model.eval()
    parameters = [value for value in model.parameters() if value.requires_grad]
    accumulators = [
        torch.zeros(tuple(value.shape), dtype=torch.float64, device="cpu")
        for value in parameters
    ]
    rows = 0
    batches = 0
    loss_sum_total = 0.0
    missing_counts = [0 for _ in parameters]
    start = time.perf_counter()
    for images, targets, metadata in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits, _ = _forward_classification_with_metadata(
            model,
            images,
            metadata,
            device=device,
        )
        logits = logits.float()
        if logits.ndim != 2 or not bool(torch.isfinite(logits).all().item()):
            raise ValueError(f"{cohort_name} logits are invalid or nonfinite.")
        loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        if not bool(torch.isfinite(loss_sum).item()):
            raise ValueError(f"{cohort_name} CE is nonfinite.")
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
        loss_sum_total += float(loss_sum.detach().cpu().item())
    if rows != int(expected_rows):
        raise ValueError(f"{cohort_name} rows differ: {rows} != {expected_rows}")
    mean_gradient = [value.div(float(rows)).float() for value in accumulators]
    stats = _gradient_stats(mean_gradient)
    return mean_gradient, {
        "cohort": cohort_name,
        "rows": rows,
        "batches": batches,
        "row_mean_cross_entropy": loss_sum_total / float(rows),
        "elapsed_seconds": time.perf_counter() - start,
        "gradient_norm": stats["norm"],
        "gradient_finite": stats["finite"],
        "parameters": len(parameters),
        "parameters_missing_every_batch": sum(
            count == batches for count in missing_counts
        ),
        "parameters_missing_some_batches": sum(count > 0 for count in missing_counts),
    }


def _predict_fp32(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[Dict[str, object]]:
    model.eval()
    rows: list[Dict[str, object]] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            logits, _ = _forward_classification_with_metadata(
                model,
                images,
                metadata,
                device=device,
            )
            probabilities = logits.float().softmax(dim=1)
            if not bool(torch.isfinite(probabilities).all().item()):
                raise ValueError("Evaluation probabilities are nonfinite.")
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Evaluation metadata is missing sample_index.")
            for row_index in range(int(targets.numel())):
                row: Dict[str, object] = {
                    "sample_index": int(sample_indices[row_index].item()),
                    "target": int(targets[row_index].item()),
                    "prediction": int(probabilities[row_index].argmax().item()),
                }
                for class_index in range(int(probabilities.size(1))):
                    row[f"prob_{class_index}"] = float(
                        probabilities[row_index, class_index].item()
                    )
                rows.append(row)
    return rows


def _evaluate_condition(
    *,
    condition: str,
    brightness: Optional[float],
    contrast: Optional[float],
    models: Mapping[str, nn.Module],
    base_dataset: MangoYOLOCropDataset,
    transform,
    holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> tuple[Dict[str, list[Dict[str, object]]], Dict[str, object]]:
    if condition == "clean":
        loader, loader_summary = _make_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context="class1_reference_agem_clean_holdout",
            seed=int(args.seed) + seed_offset,
        )
    else:
        if brightness is None or contrast is None:
            raise ValueError("Lighting conditions require brightness and contrast.")
        loader, loader_summary = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            brightness=float(brightness),
            contrast=float(contrast),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"class1_reference_agem_{condition}",
            seed=int(args.seed) + seed_offset,
        )
    predictions: Dict[str, list[Dict[str, object]]] = {}
    for name, model in models.items():
        model.to(device).eval()
        predictions[name] = _predict_fp32(model=model, loader=loader, device=device)
        model.cpu().eval()
        gc.collect()
        torch.cuda.empty_cache()
    return predictions, loader_summary


def _parameter_group(name: str) -> str:
    if name.startswith("stem."):
        return "stem"
    if name.startswith(
        ("patch_embed.", "cls_token", "register_tokens", "pos_embed", "branch_")
    ):
        return "input_tokens"
    if name.startswith("blocks."):
        return "transformer"
    if name.startswith("norm."):
        return "normalization"
    if name.startswith("fine_grained_pool."):
        return "fine_grained_pool"
    if name.startswith("head."):
        return "classifier"
    if name.startswith(("cnn_fusion_", "cnn_fusion_head.")):
        return "cnn_fusion"
    return "other"


def _gradient_group_rows(
    names: Sequence[str],
    current: Sequence[Tensor],
    reference: Sequence[Tensor],
    projected: Sequence[Tensor],
) -> list[Dict[str, object]]:
    totals: Dict[str, Dict[str, float]] = {}
    for name, current_value, reference_value, projected_value in zip(
        names, current, reference, projected
    ):
        group = _parameter_group(name)
        row = totals.setdefault(
            group,
            {
                "parameters": 0.0,
                "elements": 0.0,
                "current_squared": 0.0,
                "reference_squared": 0.0,
                "projected_squared": 0.0,
                "current_reference_dot": 0.0,
                "projected_reference_dot": 0.0,
            },
        )
        current64 = current_value.detach().cpu().double()
        reference64 = reference_value.detach().cpu().double()
        projected64 = projected_value.detach().cpu().double()
        row["parameters"] += 1
        row["elements"] += int(current_value.numel())
        row["current_squared"] += float(current64.square().sum().item())
        row["reference_squared"] += float(reference64.square().sum().item())
        row["projected_squared"] += float(projected64.square().sum().item())
        row["current_reference_dot"] += float((current64 * reference64).sum().item())
        row["projected_reference_dot"] += float(
            (projected64 * reference64).sum().item()
        )
    rows = []
    for group, values in sorted(totals.items()):
        rows.append(
            {
                "group": group,
                "parameters": int(values["parameters"]),
                "elements": int(values["elements"]),
                "current_norm": math.sqrt(values["current_squared"]),
                "reference_norm": math.sqrt(values["reference_squared"]),
                "projected_norm": math.sqrt(values["projected_squared"]),
                "current_reference_dot": values["current_reference_dot"],
                "projected_reference_dot": values["projected_reference_dot"],
            }
        )
    return rows


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    clean_raw_candidate: Mapping[str, object],
    clean_raw_control: Mapping[str, object],
    clean_control_candidate: Mapping[str, object],
    illumination_raw_candidate: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    clean_delta = clean_raw_candidate["delta"]
    clean_transitions = clean_raw_candidate["transitions"]
    raw_support = int(clean_raw_candidate["control"]["predicted_support"][FOCUS_CLASS])
    candidate_support = int(
        clean_raw_candidate["candidate"]["predicted_support"][FOCUS_CLASS]
    )
    raw_control_breaks = int(clean_raw_control["transitions"]["focus_tp_break"])
    raw_candidate_breaks = int(clean_transitions["focus_tp_break"])
    clean_checks = {
        "macro_f1_delta_nonnegative": float(clean_delta["macro_f1"]) >= 0.0,
        "class1_f1_delta_gte_0p005": float(clean_delta["class1_f1"]) >= 0.005,
        "class1_precision_delta_gte_0p005": float(clean_delta["class1_precision"])
        >= 0.005,
        "class1_recall_delta_gte_minus_0p005": float(clean_delta["class1_recall"])
        >= -0.005,
        "restricted_focus_fp_reduction_gte_2": int(
            clean_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "focus_rescues_gte_tp_breaks": int(clean_transitions["focus_fn_rescue"])
        >= raw_candidate_breaks,
        "corrections_gt_harms": int(clean_transitions["candidate_correction"])
        > int(clean_transitions["candidate_harm"]),
        "max_nonfocus_f1_drop_lte_0p010": float(
            clean_raw_candidate["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "class1_support_at_least_95pct_raw": candidate_support
        >= int(math.ceil(0.95 * raw_support)),
        "candidate_class1_f1_not_lower_than_control": float(
            clean_control_candidate["delta"]["class1_f1"]
        )
        >= 0.0,
        "candidate_tp_breaks_no_greater_than_control": raw_candidate_breaks
        <= raw_control_breaks,
    }
    precision_nonnegative = sum(
        float(row["delta"]["class1_precision"]) >= 0.0
        for row in illumination_raw_candidate
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
        "precision_nonnegative_in_at_least_two_conditions": precision_nonnegative
        >= 2,
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
    conditions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    fieldnames = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for variant in ("raw", "control", "candidate"):
        fieldnames.append(f"{variant}_prediction")
        fieldnames.extend(f"{variant}_prob_{index}" for index in range(5))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, variants in conditions.items():
            indexed = {
                name: {int(row["sample_index"]): row for row in values}
                for name, values in variants.items()
            }
            expected_indices = set(indexed["raw"])
            if any(set(values) != expected_indices for values in indexed.values()):
                raise ValueError(f"Prediction index mismatch for {condition}.")
            for sample_index in sorted(expected_indices):
                source = source_rows[sample_index]
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source.source_stem,
                    "image_path": str(source.image_path),
                    "fold": source.fold,
                    "target": source.target,
                }
                for variant in ("raw", "control", "candidate"):
                    value = indexed[variant][sample_index]
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
    geometry = summary["gradient_geometry"]
    clean = summary["comparisons"]["clean_raw_candidate"]
    lines = [
        "# Class-1-Reference A-GEM Readiness Result",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Raw gradient cosine: `{geometry['raw_cosine']:.9f}`",
        f"- Projected retained norm: `{geometry['projected_retained_norm_ratio']:.6f}`",
        f"- Clean macro F1 delta: `{clean['delta']['macro_f1']:+.6f}`",
        f"- Clean class-1 F1 delta: `{clean['delta']['class1_f1']:+.6f}`",
        f"- Clean class-1 precision delta: `{clean['delta']['class1_precision']:+.6f}`",
        f"- Clean class-1 recall delta: `{clean['delta']['class1_recall']:+.6f}`",
        f"- Restricted false-positive reduction: `{clean['transitions']['restricted_focus_fp_reduction']}`",
        "",
        "This audit used only source-disjoint folds of `yolo_f/train`; validation and test were not accessed.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, artifacts: Sequence[Path]) -> Dict[str, object]:
    files = []
    for artifact in artifacts:
        resolved = Path(artifact).resolve()
        files.append(
            {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    payload = {"files": files, "binary_model_artifacts_written": False}
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked A-GEM A0 requires CUDA.")
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    if _is_relative_to(output_path, paths["data"].parent):
        raise ValueError("Audit output cannot be written inside the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))

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
    trainable_parameters = [value for value in prototype.parameters() if value.requires_grad]
    trainable_parameter_count = sum(int(value.numel()) for value in trainable_parameters)

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
        raise ValueError("Dataset sample paths violate the locked train-only CIDT order.")
    transform = _build_eval_transform(semantics)

    hard_loader, hard_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=cohorts["hard_indices"],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="class1_reference_agem_hard_gradient",
        seed=int(args.seed) + 10,
    )
    reference_loader, reference_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=cohorts["reference_indices"],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="class1_reference_agem_reference_gradient",
        seed=int(args.seed) + 20,
    )
    gradient_model = copy.deepcopy(prototype).to(device).eval()
    hard_gradient, hard_gradient_summary = _mean_cross_entropy_gradient(
        model=gradient_model,
        loader=hard_loader,
        device=device,
        expected_rows=EXPECTED_HARD_ROWS,
        cohort_name="restricted_hard_negatives",
    )
    reference_gradient, reference_gradient_summary = _mean_cross_entropy_gradient(
        model=gradient_model,
        loader=reference_loader,
        device=device,
        expected_rows=EXPECTED_REFERENCE_ROWS,
        cohort_name="class1_reference",
    )
    del hard_loader, reference_loader, gradient_model
    gc.collect()
    torch.cuda.empty_cache()

    projected_gradient, geometry = project_agem_gradient(
        hard_gradient, reference_gradient
    )
    gradient_group_rows = _gradient_group_rows(
        parameter_names,
        hard_gradient,
        reference_gradient,
        projected_gradient,
    )

    raw_model = prototype
    control_model = copy.deepcopy(prototype).eval()
    candidate_model = copy.deepcopy(prototype).eval()
    control_initial_sha256 = _state_sha256(control_model)
    candidate_initial_sha256 = _state_sha256(candidate_model)
    _, control_schema_sha256 = _parameter_schema(control_model)
    _, candidate_schema_sha256 = _parameter_schema(candidate_model)
    control_step = _apply_normalized_parameter_step(
        control_model,
        hard_gradient,
        parameter_ratio=float(args.step_ratio),
    )
    candidate_step = _apply_normalized_parameter_step(
        candidate_model,
        projected_gradient,
        parameter_ratio=float(args.step_ratio),
    )

    models = {
        "raw": raw_model,
        "control": control_model,
        "candidate": candidate_model,
    }
    condition_predictions: Dict[str, Dict[str, list[Dict[str, object]]]] = {}
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
        condition_predictions[condition] = predictions
        loader_summaries[condition] = loader_summary

    raw_cidt_mismatches = sum(
        int(row["prediction"]) != rows[int(row["sample_index"])].keeper_prediction
        for row in condition_predictions["clean"]["raw"]
    )
    clean_raw_candidate = _comparison(
        control_rows=condition_predictions["clean"]["raw"],
        candidate_rows=condition_predictions["clean"]["candidate"],
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    clean_raw_control = _comparison(
        control_rows=condition_predictions["clean"]["raw"],
        candidate_rows=condition_predictions["clean"]["control"],
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    clean_control_candidate = _comparison(
        control_rows=condition_predictions["clean"]["control"],
        candidate_rows=condition_predictions["clean"]["candidate"],
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    illumination_raw_candidate = []
    illumination_raw_control = []
    for condition, _, _ in LIGHTING_CONDITIONS:
        raw_candidate = _comparison(
            control_rows=condition_predictions[condition]["raw"],
            candidate_rows=condition_predictions[condition]["candidate"],
            num_classes=5,
            focus_class=FOCUS_CLASS,
        )
        raw_candidate["condition"] = condition
        illumination_raw_candidate.append(raw_candidate)
        raw_control = _comparison(
            control_rows=condition_predictions[condition]["raw"],
            candidate_rows=condition_predictions[condition]["control"],
            num_classes=5,
            focus_class=FOCUS_CLASS,
        )
        raw_control["condition"] = condition
        illumination_raw_control.append(raw_control)

    control_actual_ratio_error = abs(control_step["actual_ratio"] - STEP_RATIO)
    candidate_actual_ratio_error = abs(candidate_step["actual_ratio"] - STEP_RATIO)
    update_norm_mismatch_relative = abs(
        control_step["actual_step_norm"] - candidate_step["actual_step_norm"]
    ) / max(1e-12, control_step["actual_step_norm"])
    structural_checks = {
        "locked_sources_verified": True,
        "official_worktree_clean": bool(provenance["official_agem_worktree_clean"]),
        "train_only_provenance": not provenance["validation_predictions_used"]
        and not provenance["test_data_used"],
        "source_groups_disjoint": not cohorts["source_overlap"],
        "cohort_contract_exact": all(cohorts["checks"].values()),
        "sample_paths_exact": sample_paths_exact,
        "train_paths_only": train_paths_only,
        "raw_predictions_match_cidt_1843_of_1843": raw_cidt_mismatches == 0
        and len(condition_predictions["clean"]["raw"]) == EXPECTED_HOLDOUT_ROWS,
        "all_gradients_finite": bool(hard_gradient_summary["gradient_finite"])
        and bool(reference_gradient_summary["gradient_finite"])
        and bool(geometry["finite"]),
        "hard_gradient_nonzero": float(geometry["current_norm"]) > 0.0,
        "reference_gradient_nonzero": float(geometry["reference_norm"]) > 0.0,
        "active_conflict_detected": bool(geometry["violation_active"])
        and float(geometry["raw_dot"]) < 0.0,
        "equation_error_lte_1e_minus_7": float(
            geometry["direct_flat_max_abs_error"]
        )
        <= 1e-7,
        "projected_dot_gte_minus_1e_minus_8": float(geometry["projected_dot"])
        >= -1e-8,
        "projected_cosine_abs_lte_1e_minus_6": abs(
            float(geometry["projected_cosine"])
        )
        <= 1e-6,
        "projected_retained_norm_gte_0p20": float(
            geometry["projected_retained_norm_ratio"]
        )
        >= 0.20,
        "initial_states_identical": initial_state_sha256
        == control_initial_sha256
        == candidate_initial_sha256,
        "parameter_schemas_identical": parameter_schema_sha256
        == control_schema_sha256
        == candidate_schema_sha256,
        "all_trainable_parameters_included": len(parameter_names)
        == len(hard_gradient)
        == len(reference_gradient)
        == len(projected_gradient),
        "control_update_ratio_exact": control_actual_ratio_error <= 1e-8,
        "candidate_update_ratio_exact": candidate_actual_ratio_error <= 1e-8,
        "matched_update_norms": update_norm_mismatch_relative <= 1e-8,
        "candidate_first_order_reference_nonincreasing": float(
            geometry["candidate_first_order_reference_loss_change"]
        )
        <= 1e-8,
        "no_validation_or_test_access": True,
        "no_raw_data_write": not _is_relative_to(output_dir, paths["data"].parent),
        "no_binary_model_artifact": True,
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        clean_raw_candidate=clean_raw_candidate,
        clean_raw_control=clean_raw_control,
        clean_control_candidate=clean_control_candidate,
        illumination_raw_candidate=illumination_raw_candidate,
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    gradient_groups_path = output_dir / "gradient_groups.csv"
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "artifact_manifest.json"
    _write_predictions(
        predictions_path,
        source_rows=rows,
        conditions=condition_predictions,
    )
    _write_gradient_groups(gradient_groups_path, gradient_group_rows)
    summary: Dict[str, object] = {
        "protocol": "class1_reference_agem_one_macrostep_a0",
        "status": "passed" if gate["all_gates_passed"] else "closed",
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "model": {
            "class_names": class_names,
            "trainable_parameter_tensors": len(parameter_names),
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_schema_sha256": parameter_schema_sha256,
            "initial_state_sha256": initial_state_sha256,
            "control_state_sha256": _state_sha256(control_model),
            "candidate_state_sha256": _state_sha256(candidate_model),
        },
        "gradient_computation": {
            "hard": hard_gradient_summary,
            "reference": reference_gradient_summary,
            "hard_loader": hard_loader_summary,
            "reference_loader": reference_loader_summary,
            "accumulation_dtype": "float64",
            "materialized_dtype": "float32",
            "loss": "row_mean_multiclass_cross_entropy",
            "model_mode": "eval",
        },
        "gradient_geometry": geometry,
        "steps": {
            "control": control_step,
            "candidate": candidate_step,
            "update_norm_mismatch_relative": update_norm_mismatch_relative,
        },
        "comparisons": {
            "clean_raw_candidate": clean_raw_candidate,
            "clean_raw_control": clean_raw_control,
            "clean_control_candidate": clean_control_candidate,
            "illumination_raw_candidate": illumination_raw_candidate,
            "illumination_raw_control": illumination_raw_control,
        },
        "raw_cidt_mismatches": raw_cidt_mismatches,
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

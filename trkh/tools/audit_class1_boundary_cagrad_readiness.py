from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from scipy.optimize import minimize
from torch import Tensor

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.inference.inference import load_checkpoint
from trkh.models.model import create_model, load_model_state
from trkh.tools.audit_class1_boundary_gem_readiness import (
    _aligned_comparison,
    _git_worktree_clean,
    _gradient_dot,
    _gradient_norm,
    _gradient_relative_difference,
    _is_relative_to,
    _parameter_group,
)
from trkh.tools.audit_class1_boundary_vrex_readiness import (
    BOUNDARY_WEIGHT,
    CONDITIONS,
    _balanced_environment_gradient,
    _make_condition_loader,
    _weighted_sum_gradients,
    boundary_risk_from_rows,
)
from trkh.tools.audit_class1_reference_agem_readiness import (
    EXPECTED_HARD_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_REFERENCE_ROWS,
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
    STEP_RATIO,
    _apply_normalized_parameter_step,
    _cohort_serializable,
    _evaluate_condition,
    _locked_cohort_summary,
    _mean_cross_entropy_gradient,
    _parameter_schema,
    _write_manifest,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _git_commit,
    _prepare_output_dir,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


Gradient = list[Tensor]
CAGRAD_C = 0.4
PRIOR_VARIANTS = ("raw", "margin_agem", "boundary_erm", "boundary_vrex")

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_VREX_SUMMARY_SHA256 = "0b78da68af731456faaca8664a4b97dc111121a0a5c7630cfede1141e800babb"
LOCKED_VREX_PREDICTIONS_SHA256 = "7ca25f7ffad843e7cc4da4cc70567c1372209f2b28e4710599536ea42edc8de9"
LOCKED_VREX_MANIFEST_SHA256 = "1ed6bec4753f0f581f0bed0c558f4a65814ca6692524cdc56a935b8e8f1d3f1e"
LOCKED_PROTOCOL_SHA256 = "79047d2bd5bab18ee7b3da1c08d44b7e0224ef3677e6fc6512f6bb2ff48c9419"
LOCKED_CAGRAD_PAPER_SHA256 = "ad50131ada07c8c1b68fd7e31c3226f0a8980b667fc388c71e5be660f917098f"
LOCKED_CAGRAD_COMMIT = "dc3d48152b6196945cfd56144879b9d42353b095"
LOCKED_CAGRAD_SOURCE_SHA256 = "b4ab48f2409979bdd07c31ec7712421d9a6f19462fdfe0d4e82d28d9a477eef5"
LOCKED_CAGRAD_RUN_SHA256 = "571d1abf065bd23ff5bbd6b9e81f1e530ce7ac97dca8d2f4dac3b11f36f2a714"
LOCKED_CAGRAD_LICENSE_SHA256 = "057e0b7c6233c0ad86a3b8998333664967225ee761836407385d042d2583716c"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only class-1 boundary CAGrad information gate. "
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
        "--prior-vrex-summary",
        type=Path,
        default=Path("runs/audit_class1_boundary_vrex_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--prior-vrex-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_vrex_readiness_20260715/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--prior-vrex-manifest",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_vrex_readiness_20260715/artifact_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CLASS1_BOUNDARY_CAGRAD_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--cagrad-paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\cagrad_neurips2021.pdf"),
    )
    parser.add_argument(
        "--cagrad-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\CAGrad"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_class1_boundary_cagrad_readiness_20260715"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--focus-class", type=int, default=FOCUS_CLASS)
    parser.add_argument("--boundary-weight", type=float, default=BOUNDARY_WEIGHT)
    parser.add_argument("--cagrad-c", type=float, default=CAGRAD_C)
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
            float(args.boundary_weight), BOUNDARY_WEIGHT, rel_tol=0.0, abs_tol=0.0
        )
        and math.isclose(float(args.cagrad_c), CAGRAD_C, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(args.step_ratio), STEP_RATIO, rel_tol=0.0, abs_tol=1e-15)
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    cagrad_root = Path(args.cagrad_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "prior_vrex_summary": Path(args.prior_vrex_summary).resolve(),
        "prior_vrex_predictions": Path(args.prior_vrex_predictions).resolve(),
        "prior_vrex_manifest": Path(args.prior_vrex_manifest).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "cagrad_paper": Path(args.cagrad_paper).resolve(),
        "cagrad_source": cagrad_root / "nyuv2" / "utils.py",
        "cagrad_run": cagrad_root / "nyuv2" / "run.sh",
        "cagrad_license": cagrad_root / "LICENSE",
    }


def _read_prior_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    holdout_indices: Sequence[int],
) -> tuple[Dict[str, Dict[str, list[Dict[str, object]]]], Dict[str, int]]:
    result = {
        condition: {name: [] for name in PRIOR_VARIANTS} for condition in CONDITIONS
    }
    argmax_mismatches = {name: 0 for name in PRIOR_VARIANTS}
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
        for name in PRIOR_VARIANTS:
            required.add(f"{name}_prediction")
            required.update(f"{name}_prob_{index}" for index in range(5))
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prior V-REx prediction CSV is missing: {sorted(missing)}")
        for raw in reader:
            condition = str(raw["condition"]).strip()
            if condition not in result:
                raise ValueError(f"Unexpected prior V-REx condition: {condition}")
            sample_index = int(raw["sample_index"])
            if not 0 <= sample_index < len(rows):
                raise ValueError("Prior V-REx sample index is out of range.")
            source = rows[sample_index]
            if (
                source.source_stem != str(raw["source_stem"]).strip().casefold()
                or source.image_path != Path(str(raw["image_path"])).resolve()
                or source.fold != int(raw["fold"])
                or source.target != int(raw["target"])
            ):
                raise ValueError(f"Prior V-REx source row mismatch at {sample_index}.")
            for name in PRIOR_VARIANTS:
                probabilities = [float(raw[f"{name}_prob_{index}"]) for index in range(5)]
                if not all(math.isfinite(value) and value >= 0.0 for value in probabilities):
                    raise ValueError("Prior V-REx probabilities are invalid.")
                if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
                    raise ValueError("Prior V-REx probabilities are not normalized.")
                prediction = int(raw[f"{name}_prediction"])
                if int(np.argmax(probabilities)) != prediction:
                    argmax_mismatches[name] += 1
                row: Dict[str, object] = {
                    "sample_index": sample_index,
                    "target": source.target,
                    "prediction": prediction,
                }
                for class_index, value in enumerate(probabilities):
                    row[f"prob_{class_index}"] = value
                result[condition][name].append(row)
    expected = list(holdout_indices)
    for condition, variants in result.items():
        for name, values in variants.items():
            values.sort(key=lambda row: int(row["sample_index"]))
            indices = [int(row["sample_index"]) for row in values]
            if len(indices) != EXPECTED_HOLDOUT_ROWS or len(set(indices)) != len(indices):
                raise ValueError(f"Prior {condition}/{name} row contract differs.")
            if indices != expected:
                raise ValueError(f"Prior {condition}/{name} differs from locked holdout.")
    if any(argmax_mismatches.values()):
        raise ValueError(
            f"Prior V-REx stored predictions differ from argmax: {argmax_mismatches}"
        )
    return result, argmax_mismatches


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted CAGrad protocol.")
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
            paths["cidt_predictions"], LOCKED_CIDT_PREDICTIONS_SHA256, "CIDT predictions"
        ),
        "prior_vrex_summary": _verify_sha256(
            paths["prior_vrex_summary"], LOCKED_VREX_SUMMARY_SHA256, "prior V-REx summary"
        ),
        "prior_vrex_predictions": _verify_sha256(
            paths["prior_vrex_predictions"],
            LOCKED_VREX_PREDICTIONS_SHA256,
            "prior V-REx predictions",
        ),
        "prior_vrex_manifest": _verify_sha256(
            paths["prior_vrex_manifest"],
            LOCKED_VREX_MANIFEST_SHA256,
            "prior V-REx manifest",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "CAGrad protocol"
        ),
        "cagrad_paper": _verify_sha256(
            paths["cagrad_paper"], LOCKED_CAGRAD_PAPER_SHA256, "CAGrad paper"
        ),
        "cagrad_source": _verify_sha256(
            paths["cagrad_source"], LOCKED_CAGRAD_SOURCE_SHA256, "CAGrad source"
        ),
        "cagrad_run": _verify_sha256(
            paths["cagrad_run"], LOCKED_CAGRAD_RUN_SHA256, "CAGrad run recipe"
        ),
        "cagrad_license": _verify_sha256(
            paths["cagrad_license"], LOCKED_CAGRAD_LICENSE_SHA256, "CAGrad license"
        ),
    }
    cagrad_root = Path(args.cagrad_root).resolve()
    commit = _git_commit(cagrad_root)
    clean = _git_worktree_clean(cagrad_root)
    if commit != LOCKED_CAGRAD_COMMIT:
        raise ValueError(f"CAGrad commit differs: {commit} != {LOCKED_CAGRAD_COMMIT}")
    if not clean:
        raise ValueError("CAGrad worktree is not clean.")
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    prior_summary = json.loads(paths["prior_vrex_summary"].read_text(encoding="utf-8"))
    for name, payload in (("CIDT", cidt_summary), ("prior V-REx", prior_summary)):
        if bool(payload.get("test_data_used", True)):
            raise ValueError(f"{name} provenance indicates test data use.")
        if bool(payload.get("validation_predictions_used", True)):
            raise ValueError(f"{name} provenance indicates validation prediction use.")
    if bool(prior_summary.get("binary_model_artifacts_written", True)):
        raise ValueError("Prior V-REx summary indicates a binary model artifact.")
    if prior_summary.get("protocol") != "class1_boundary_balanced_vrex_a0":
        raise ValueError("Prior V-REx protocol identity differs.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    prior, argmax_mismatches = _read_prior_predictions(
        paths["prior_vrex_predictions"],
        rows=rows,
        holdout_indices=cohorts["holdout_indices"],
    )
    raw_cidt_mismatches = sum(
        int(row["prediction"]) != rows[int(row["sample_index"])].keeper_prediction
        for row in prior["clean"]["raw"]
    )
    if raw_cidt_mismatches:
        raise ValueError("Prior raw predictions differ from CIDT.")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "cagrad_commit": commit,
        "cagrad_worktree_clean": clean,
        "cohort": _cohort_serializable(cohorts),
        "prior_prediction_conditions": {
            condition: len(values["raw"]) for condition, values in prior.items()
        },
        "prior_argmax_mismatches": argmax_mismatches,
        "prior_raw_cidt_mismatches": raw_cidt_mismatches,
        "prior_vrex_status": prior_summary.get("status"),
        "validation_predictions_used": False,
        "test_data_used": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _gradient_gram(gradients: Sequence[Sequence[Tensor]]) -> np.ndarray:
    count = len(gradients)
    if count < 2 or any(len(values) != len(gradients[0]) for values in gradients):
        raise ValueError("CAGrad gradient schemas differ or are empty.")
    gram = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left, count):
            value = _gradient_dot(gradients[left], gradients[right])
            gram[left, right] = value
            gram[right, left] = value
    return gram


def cagrad_dual_terms(
    weights: Sequence[float], gram: np.ndarray, *, c: float
) -> tuple[float, np.ndarray]:
    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.shape != (vector.size, vector.size) or vector.size < 2:
        raise ValueError("CAGrad Gram/weight dimensions differ.")
    uniform = np.full(vector.size, 1.0 / vector.size, dtype=np.float64)
    g0_norm = math.sqrt(max(0.0, float(uniform @ matrix @ uniform)))
    gw_squared = max(0.0, float(vector @ matrix @ vector))
    gw_norm = math.sqrt(gw_squared)
    if g0_norm <= 0.0 or gw_norm <= 0.0:
        raise ValueError("CAGrad average or weighted gradient is zero.")
    objective = float(vector @ matrix @ uniform) + float(c) * g0_norm * gw_norm
    jacobian = matrix @ uniform + float(c) * g0_norm * (matrix @ vector) / gw_norm
    return objective, jacobian


def _solver_starts(count: int) -> list[tuple[str, np.ndarray]]:
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    starts = [("uniform", uniform)]
    for index in range(count):
        vertex = np.zeros(count, dtype=np.float64)
        vertex[index] = 1.0
        starts.append((f"vertex_{index}", vertex))
    for index in range(count):
        interior = np.full(count, 0.15, dtype=np.float64)
        interior[index] = 0.55
        starts.append((f"interior_{index}", interior))
    return starts


def solve_cagrad_dual(
    gram: np.ndarray, *, c: float
) -> tuple[np.ndarray, list[Dict[str, object]], Dict[str, object]]:
    raw_gram = np.asarray(gram, dtype=np.float64)
    if raw_gram.ndim != 2 or raw_gram.shape[0] != raw_gram.shape[1]:
        raise ValueError("CAGrad Gram matrix must be square.")
    if not np.isfinite(raw_gram).all():
        raise ValueError("CAGrad Gram matrix is nonfinite.")
    scale = float(np.max(np.abs(raw_gram)))
    if scale <= 0.0:
        raise ValueError("CAGrad Gram matrix is zero.")
    normalized = raw_gram / scale
    count = normalized.shape[0]
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = tuple((0.0, 1.0) for _ in range(count))
    rows: list[Dict[str, object]] = []
    results = []
    for name, start in _solver_starts(count):
        result = minimize(
            lambda w: cagrad_dual_terms(w, normalized, c=float(c))[0],
            start,
            jac=lambda w: cagrad_dual_terms(w, normalized, c=float(c))[1],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        weights = np.asarray(result.x, dtype=np.float64)
        objective, jacobian = cagrad_dual_terms(weights, normalized, c=float(c))
        kkt_residual = max(0.0, float(weights @ jacobian - np.min(jacobian)))
        row: Dict[str, object] = {
            "start": name,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "normalized_objective": objective,
            "raw_objective": objective * scale,
            "simplex_sum_error": abs(float(weights.sum()) - 1.0),
            "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()),
            "kkt_residual": kkt_residual,
        }
        for index, value in enumerate(weights):
            row[f"weight_{index}"] = float(value)
        rows.append(row)
        results.append((name, result, weights, objective, kkt_residual))
    primary = results[0]
    objectives = [value[3] for value in results]
    telemetry = {
        "gram_scale": scale,
        "primary_start": primary[0],
        "primary_success": bool(primary[1].success),
        "all_starts_success": all(bool(value[1].success) for value in results),
        "objective_spread": float(max(objectives) - min(objectives)),
        "maximum_kkt_residual": float(max(value[4] for value in results)),
        "minimum_weight": float(min(value[2].min() for value in results)),
        "maximum_simplex_sum_error": float(
            max(abs(float(value[2].sum()) - 1.0) for value in results)
        ),
    }
    return primary[2], rows, telemetry


def compose_cagrad_direction(
    environment_gradients: Sequence[Sequence[Tensor]],
    *,
    c: float,
) -> tuple[Gradient, Gradient, Dict[str, object], list[Dict[str, object]]]:
    gram = _gradient_gram(environment_gradients)
    weights, solver_rows, solver = solve_cagrad_dual(gram, c=float(c))
    count = len(environment_gradients)
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    g0 = _weighted_sum_gradients(environment_gradients, uniform)
    gw = _weighted_sum_gradients(environment_gradients, weights)
    g0_norm = _gradient_norm(g0)
    gw_norm = _gradient_norm(gw)
    if g0_norm <= 0.0 or gw_norm <= 0.0:
        raise ValueError("CAGrad average or weighted gradient is zero.")
    coefficient = float(c) * g0_norm / gw_norm
    direct_weights = uniform + coefficient * weights
    direction = _weighted_sum_gradients(environment_gradients, direct_weights)
    staged = _weighted_sum_gradients([g0, gw], [1.0, coefficient])
    equation_relative_error = _gradient_relative_difference(direction, staged)

    paper_gw_norm = math.sqrt(float((weights / count) @ gram @ (weights / count)))
    paper_coefficient = float(c) * g0_norm / paper_gw_norm
    paper_direct_weights = uniform + paper_coefficient * (weights / count)
    paper_direction = _weighted_sum_gradients(
        environment_gradients, paper_direct_weights
    )
    paper_scale_relative_error = _gradient_relative_difference(direction, paper_direction)

    official_coefficient = (float(c) * g0_norm + 1e-8) / (gw_norm + 1e-8)
    official_direction = _weighted_sum_gradients([g0, gw], [1.0, official_coefficient])
    official_rescaled = _weighted_sum_gradients(
        [official_direction], [1.0 / (1.0 + float(c) ** 2)]
    )
    direction_unit = _weighted_sum_gradients([direction], [1.0 / _gradient_norm(direction)])
    official_unit = _weighted_sum_gradients(
        [official_rescaled], [1.0 / _gradient_norm(official_rescaled)]
    )
    official_direction_relative_error = _gradient_relative_difference(
        direction_unit, official_unit
    )

    offset = _weighted_sum_gradients([direction, g0], [1.0, -1.0])
    task_dots = [_gradient_dot(values, direction) for values in environment_gradients]
    erm_task_dots = [_gradient_dot(values, g0) for values in environment_gradients]
    geometry = {
        "c": float(c),
        "gram_matrix": gram.tolist(),
        "weights": weights.tolist(),
        "g0_norm": g0_norm,
        "gw_norm": gw_norm,
        "direction_norm": _gradient_norm(direction),
        "direction_vs_erm_relative_l2": _gradient_relative_difference(direction, g0),
        "ball_ratio": _gradient_norm(offset) / g0_norm,
        "coefficient": coefficient,
        "equation_relative_error": equation_relative_error,
        "paper_scale_relative_error": paper_scale_relative_error,
        "official_direction_relative_error": official_direction_relative_error,
        "task_dots": task_dots,
        "erm_task_dots": erm_task_dots,
        "minimum_task_dot": min(task_dots),
        "minimum_erm_task_dot": min(erm_task_dots),
        "minimum_task_dot_gain": min(task_dots) - min(erm_task_dots),
        "average_objective_first_order_change": -_gradient_dot(g0, direction),
        "finite": bool(
            np.isfinite(gram).all()
            and all(math.isfinite(value) for value in task_dots + erm_task_dots)
            and all(
                bool(torch.isfinite(value).all().item())
                for gradient in (g0, gw, direction)
                for value in gradient
            )
        ),
        "solver": solver,
    }
    return g0, direction, geometry, solver_rows


def _risk_table(
    *,
    prior: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    current: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    hard_indices: Sequence[int],
    positive_indices: Sequence[int],
) -> Dict[str, object]:
    names = (*PRIOR_VARIANTS, "boundary_cagrad")
    variants: Dict[str, list[Dict[str, object]]] = {name: [] for name in names}
    for condition in CONDITIONS:
        sources = {
            **{name: prior[condition][name] for name in PRIOR_VARIANTS},
            "boundary_cagrad": current[condition]["boundary_cagrad"],
        }
        for name, values in sources.items():
            risk = boundary_risk_from_rows(
                values,
                hard_indices=hard_indices,
                positive_indices=positive_indices,
            )
            variants[name].append({"condition": condition, **risk})
    summary: Dict[str, object] = {}
    for name, values in variants.items():
        risks = np.asarray([float(row["boundary_risk"]) for row in values])
        summary[name] = {
            "conditions": values,
            "mean": float(risks.mean()),
            "population_variance": float(np.mean((risks - risks.mean()) ** 2)),
            "worst": float(risks.max()),
        }
    return summary


def _gradient_group_rows(
    names: Sequence[str],
    environment_gradients: Sequence[Sequence[Tensor]],
    erm: Sequence[Tensor],
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
                "erm_squared": 0.0,
                "candidate_squared": 0.0,
                **{
                    f"environment_{index}_squared": 0.0
                    for index in range(len(environment_gradients))
                },
            },
        )
        row["parameters"] += 1
        row["elements"] += int(erm[parameter_index].numel())
        row["erm_squared"] += float(
            erm[parameter_index].detach().cpu().double().square().sum().item()
        )
        row["candidate_squared"] += float(
            candidate[parameter_index].detach().cpu().double().square().sum().item()
        )
        for index, values in enumerate(environment_gradients):
            row[f"environment_{index}_squared"] += float(
                values[parameter_index].detach().cpu().double().square().sum().item()
            )
    output = []
    for group, values in sorted(totals.items()):
        row: Dict[str, object] = {
            "group": group,
            "parameters": int(values["parameters"]),
            "elements": int(values["elements"]),
            "erm_norm": math.sqrt(values["erm_squared"]),
            "candidate_norm": math.sqrt(values["candidate_squared"]),
        }
        for index in range(len(environment_gradients)):
            row[f"environment_{index}_norm"] = math.sqrt(
                values[f"environment_{index}_squared"]
            )
        output.append(row)
    return output


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    clean_raw_candidate: Mapping[str, object],
    clean_raw_erm: Mapping[str, object],
    clean_raw_vrex: Mapping[str, object],
    clean_erm_candidate: Mapping[str, object],
    clean_vrex_candidate: Mapping[str, object],
    clean_margin_candidate: Mapping[str, object],
    illumination_raw_candidate: Sequence[Mapping[str, object]],
    illumination_erm_candidate: Sequence[Mapping[str, object]],
    illumination_vrex_candidate: Sequence[Mapping[str, object]],
    risk_table: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    delta = clean_raw_candidate["delta"]
    transitions = clean_raw_candidate["transitions"]
    raw_support = int(clean_raw_candidate["control"]["predicted_support"][FOCUS_CLASS])
    candidate_support = int(
        clean_raw_candidate["candidate"]["predicted_support"][FOCUS_CLASS]
    )
    allowed_tp_breaks = min(
        int(clean_raw_erm["transitions"]["focus_tp_break"]),
        int(clean_raw_vrex["transitions"]["focus_tp_break"]),
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
        "candidate_f1_gain_vs_erm_gte_0p005": float(
            clean_erm_candidate["delta"]["class1_f1"]
        )
        >= 0.005,
        "candidate_f1_gain_vs_vrex_gte_0p005": float(
            clean_vrex_candidate["delta"]["class1_f1"]
        )
        >= 0.005,
        "candidate_within_0p002_margin_agem_f1": float(
            clean_margin_candidate["delta"]["class1_f1"]
        )
        >= -0.002,
        "candidate_tp_breaks_no_greater_than_erm_or_vrex": int(
            transitions["focus_tp_break"]
        )
        <= allowed_tp_breaks,
    }
    precision_nonnegative = sum(
        float(row["delta"]["class1_precision"]) >= 0.0
        for row in illumination_raw_candidate
    )
    raw_worst = min(
        float(row["control"]["per_class_f1"][FOCUS_CLASS])
        for row in illumination_raw_candidate
    )
    candidate_worst = min(
        float(row["candidate"]["per_class_f1"][FOCUS_CLASS])
        for row in illumination_raw_candidate
    )
    erm_worst = min(
        float(row["control"]["per_class_f1"][FOCUS_CLASS])
        for row in illumination_erm_candidate
    )
    vrex_worst = min(
        float(row["control"]["per_class_f1"][FOCUS_CLASS])
        for row in illumination_vrex_candidate
    )
    condition_sets = [
        {str(row["condition"]) for row in values}
        for values in (
            illumination_raw_candidate,
            illumination_erm_candidate,
            illumination_vrex_candidate,
        )
    ]
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
        "worst_condition_class1_f1_gain_gte_0p010": candidate_worst - raw_worst
        >= 0.010,
        "worst_condition_class1_f1_gain_vs_erm_gte_0p005": candidate_worst
        - erm_worst
        >= 0.005,
        "worst_condition_class1_f1_gain_vs_vrex_gte_0p005": candidate_worst
        - vrex_worst
        >= 0.005,
        "condition_sets_match": condition_sets[0] == condition_sets[1] == condition_sets[2],
    }
    erm_risk = risk_table["boundary_erm"]
    candidate_risk = risk_table["boundary_cagrad"]
    risk_checks = {
        "candidate_boundary_risk_mean_lte_erm_plus_0p005": float(
            candidate_risk["mean"]
        )
        <= float(erm_risk["mean"]) + 0.005,
        "candidate_worst_boundary_risk_lte_erm": float(candidate_risk["worst"])
        <= float(erm_risk["worst"]),
    }
    failed = [key for key, value in structural_checks.items() if not value]
    failed.extend(key for key, value in clean_checks.items() if not value)
    failed.extend(key for key, value in illumination_checks.items() if not value)
    failed.extend(key for key, value in risk_checks.items() if not value)
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "illumination_checks": illumination_checks,
        "risk_checks": risk_checks,
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
    variants = (*PRIOR_VARIANTS, "boundary_cagrad")
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
                **{name: prior[condition][name] for name in PRIOR_VARIANTS},
                "boundary_cagrad": current[condition]["boundary_cagrad"],
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


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    gate = summary["gate"]
    clean = summary["comparisons"]["clean_raw_cagrad"]
    geometry = summary["cagrad_geometry"]
    lines = [
        "# Class-1 Boundary CAGrad Result",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- CAGrad weights: `{geometry['weights']}`",
        f"- Ball ratio: `{geometry['ball_ratio']:.9f}`",
        f"- Minimum task-dot gain vs ERM: `{geometry['minimum_task_dot_gain']:+.9f}`",
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
        raise RuntimeError("Locked boundary CAGrad A0 requires CUDA.")
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    if _is_relative_to(output_path, paths["data"].parent):
        raise ValueError("Audit output cannot be written inside the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    prior_predictions, prior_argmax_mismatches = _read_prior_predictions(
        paths["prior_vrex_predictions"],
        rows=rows,
        holdout_indices=cohorts["holdout_indices"],
    )
    prior_summary = json.loads(paths["prior_vrex_summary"].read_text(encoding="utf-8"))

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
    environment_gradients: list[Gradient] = []
    environment_summaries = []
    environment_loader_summaries = []
    environment_flat_errors = []
    conditions = [("clean", None, None), *LIGHTING_CONDITIONS]
    for condition_index, (condition, brightness, contrast) in enumerate(conditions):
        hard_loader, hard_loader_summary = _make_condition_loader(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            base_dataset=base_dataset,
            transform=transform,
            indices=cohorts["hard_indices"],
            args=args,
            context=f"class1_boundary_cagrad_{condition}_hard",
            seed=int(args.seed) + 10 + condition_index * 2,
        )
        positive_loader, positive_loader_summary = _make_condition_loader(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            base_dataset=base_dataset,
            transform=transform,
            indices=cohorts["reference_indices"],
            args=args,
            context=f"class1_boundary_cagrad_{condition}_positive",
            seed=int(args.seed) + 11 + condition_index * 2,
        )
        hard_gradient, hard_summary = _mean_cross_entropy_gradient(
            model=gradient_model,
            loader=hard_loader,
            device=device,
            expected_rows=EXPECTED_HARD_ROWS,
            cohort_name=f"{condition}_restricted_hard_negatives",
        )
        positive_gradient, positive_summary = _mean_cross_entropy_gradient(
            model=gradient_model,
            loader=positive_loader,
            device=device,
            expected_rows=EXPECTED_REFERENCE_ROWS,
            cohort_name=f"{condition}_class1_positive",
        )
        environment_gradient, flat_error = _balanced_environment_gradient(
            hard_gradient,
            positive_gradient,
            weight=float(args.boundary_weight),
        )
        hard_loss = float(hard_summary["row_mean_cross_entropy"])
        positive_loss = float(positive_summary["row_mean_cross_entropy"])
        environment_risk = (
            float(args.boundary_weight) * hard_loss
            + (1.0 - float(args.boundary_weight)) * positive_loss
        )
        environment_gradients.append(environment_gradient)
        environment_flat_errors.append(flat_error)
        environment_summaries.append(
            {
                "condition": condition,
                "brightness": brightness,
                "contrast": contrast,
                "hard": hard_summary,
                "positive": positive_summary,
                "boundary_risk": environment_risk,
                "boundary_gradient_norm": _gradient_norm(environment_gradient),
                "flat_max_abs_error": flat_error,
            }
        )
        environment_loader_summaries.append(
            {
                "condition": condition,
                "hard": hard_loader_summary,
                "positive": positive_loader_summary,
            }
        )
        del hard_loader, positive_loader, hard_gradient, positive_gradient
        gc.collect()
    del gradient_model
    gc.collect()
    torch.cuda.empty_cache()

    erm_direction, cagrad_direction, cagrad_geometry, solver_rows = (
        compose_cagrad_direction(environment_gradients, c=float(args.cagrad_c))
    )
    gradient_group_rows = _gradient_group_rows(
        parameter_names,
        environment_gradients,
        erm_direction,
        cagrad_direction,
    )

    candidate_model = copy.deepcopy(prototype).eval()
    candidate_initial_sha256 = _state_sha256(candidate_model)
    _, candidate_schema_sha256 = _parameter_schema(candidate_model)
    candidate_step = _apply_normalized_parameter_step(
        candidate_model,
        cagrad_direction,
        parameter_ratio=float(args.step_ratio),
    )
    prior_erm_step = prior_summary["steps"]["boundary_erm"]
    step_norm_mismatch = abs(
        float(candidate_step["actual_step_norm"])
        - float(prior_erm_step["actual_step_norm"])
    ) / max(1e-12, float(prior_erm_step["actual_step_norm"]))

    models = {"boundary_cagrad": candidate_model}
    current_predictions: Dict[str, Dict[str, list[Dict[str, object]]]] = {}
    evaluation_loader_summaries: Dict[str, object] = {}
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
        evaluation_loader_summaries[condition] = loader_summary

    clean_prior = prior_predictions["clean"]
    clean_current = current_predictions["clean"]["boundary_cagrad"]
    clean_raw_cagrad = _aligned_comparison(
        control_rows=clean_prior["raw"], candidate_rows=clean_current
    )
    clean_raw_erm = _aligned_comparison(
        control_rows=clean_prior["raw"],
        candidate_rows=clean_prior["boundary_erm"],
    )
    clean_raw_vrex = _aligned_comparison(
        control_rows=clean_prior["raw"],
        candidate_rows=clean_prior["boundary_vrex"],
    )
    clean_erm_cagrad = _aligned_comparison(
        control_rows=clean_prior["boundary_erm"], candidate_rows=clean_current
    )
    clean_vrex_cagrad = _aligned_comparison(
        control_rows=clean_prior["boundary_vrex"], candidate_rows=clean_current
    )
    clean_margin_cagrad = _aligned_comparison(
        control_rows=clean_prior["margin_agem"], candidate_rows=clean_current
    )
    illumination_raw_cagrad = []
    illumination_erm_cagrad = []
    illumination_vrex_cagrad = []
    for condition, _, _ in LIGHTING_CONDITIONS:
        candidate = current_predictions[condition]["boundary_cagrad"]
        for source, output in (
            ("raw", illumination_raw_cagrad),
            ("boundary_erm", illumination_erm_cagrad),
            ("boundary_vrex", illumination_vrex_cagrad),
        ):
            comparison = _aligned_comparison(
                control_rows=prior_predictions[condition][source],
                candidate_rows=candidate,
            )
            comparison["condition"] = condition
            output.append(comparison)

    holdout_hard_indices = [
        index
        for index in cohorts["holdout_indices"]
        if rows[index].target in RESTRICTED_NEGATIVE_CLASSES
        and rows[index].keeper_prediction == FOCUS_CLASS
    ]
    holdout_positive_indices = [
        index
        for index in cohorts["holdout_indices"]
        if rows[index].target == FOCUS_CLASS
    ]
    if len(holdout_hard_indices) != 36 or len(holdout_positive_indices) != 109:
        raise ValueError("Holdout boundary-risk cohort differs from protocol.")
    risk_table = _risk_table(
        prior=prior_predictions,
        current=current_predictions,
        hard_indices=holdout_hard_indices,
        positive_indices=holdout_positive_indices,
    )
    raw_cidt_mismatches = sum(
        int(row["prediction"]) != rows[int(row["sample_index"])].keeper_prediction
        for row in clean_prior["raw"]
    )
    gradient_summaries = [
        summary[key]
        for summary in environment_summaries
        for key in ("hard", "positive")
    ]
    solver = cagrad_geometry["solver"]
    structural_checks = {
        "locked_sources_verified": True,
        "cagrad_worktree_clean": bool(provenance["cagrad_worktree_clean"]),
        "train_only_provenance": not provenance["validation_predictions_used"]
        and not provenance["test_data_used"],
        "source_groups_disjoint": not cohorts["source_overlap"],
        "cohort_contract_exact": all(cohorts["checks"].values()),
        "sample_paths_exact": sample_paths_exact,
        "train_paths_only": train_paths_only,
        "prior_argmax_exact": not any(prior_argmax_mismatches.values()),
        "prior_raw_predictions_match_cidt_1843_of_1843": raw_cidt_mismatches == 0,
        "all_eight_cohort_gradients_finite": all(
            bool(summary["gradient_finite"]) for summary in gradient_summaries
        ),
        "all_eight_cohort_gradients_nonzero": all(
            float(summary["gradient_norm"]) > 0.0 for summary in gradient_summaries
        ),
        "all_four_environment_gradients_nonzero": all(
            _gradient_norm(values) > 0.0 for values in environment_gradients
        ),
        "environment_flat_errors_lte_1e_minus_7": max(environment_flat_errors)
        <= 1e-7,
        "cagrad_geometry_finite": bool(cagrad_geometry["finite"]),
        "primary_solver_success": bool(solver["primary_success"]),
        "all_solver_starts_success": bool(solver["all_starts_success"]),
        "solver_objective_spread_lte_1e_minus_10": float(solver["objective_spread"])
        <= 1e-10,
        "solver_kkt_residual_lte_1e_minus_7": float(solver["maximum_kkt_residual"])
        <= 1e-7,
        "simplex_sum_error_lte_1e_minus_10": float(
            solver["maximum_simplex_sum_error"]
        )
        <= 1e-10,
        "simplex_minimum_weight_gte_minus_1e_minus_10": float(
            solver["minimum_weight"]
        )
        >= -1e-10,
        "paper_scale_relative_error_lte_1e_minus_10": float(
            cagrad_geometry["paper_scale_relative_error"]
        )
        <= 1e-10,
        "official_direction_relative_error_lte_1e_minus_6": float(
            cagrad_geometry["official_direction_relative_error"]
        )
        <= 1e-6,
        "equation_relative_error_lte_1e_minus_7": float(
            cagrad_geometry["equation_relative_error"]
        )
        <= 1e-7,
        "ball_ratio_exact": abs(float(cagrad_geometry["ball_ratio"]) - CAGRAD_C)
        <= 1e-7,
        "candidate_differs_from_erm_gte_0p35": float(
            cagrad_geometry["direction_vs_erm_relative_l2"]
        )
        >= 0.35,
        "minimum_task_dot_nonnegative": float(cagrad_geometry["minimum_task_dot"])
        >= 0.0,
        "minimum_task_dot_gain_vs_erm_gte_1e_minus_6": float(
            cagrad_geometry["minimum_task_dot_gain"]
        )
        >= 1e-6,
        "average_objective_first_order_change_negative": float(
            cagrad_geometry["average_objective_first_order_change"]
        )
        < 0.0,
        "initial_states_identical": initial_state_sha256 == candidate_initial_sha256,
        "parameter_schemas_identical": parameter_schema_sha256
        == candidate_schema_sha256,
        "all_trainable_parameters_included": len(parameter_names)
        == len(erm_direction)
        == len(cagrad_direction),
        "candidate_update_ratio_exact": abs(candidate_step["actual_ratio"] - STEP_RATIO)
        <= 1e-8,
        "matched_erm_update_norm": step_norm_mismatch <= 5e-7,
        "no_validation_or_test_access": True,
        "no_raw_data_write": not _is_relative_to(output_dir, paths["data"].parent),
        "no_binary_model_artifact": True,
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        clean_raw_candidate=clean_raw_cagrad,
        clean_raw_erm=clean_raw_erm,
        clean_raw_vrex=clean_raw_vrex,
        clean_erm_candidate=clean_erm_cagrad,
        clean_vrex_candidate=clean_vrex_cagrad,
        clean_margin_candidate=clean_margin_cagrad,
        illumination_raw_candidate=illumination_raw_cagrad,
        illumination_erm_candidate=illumination_erm_cagrad,
        illumination_vrex_candidate=illumination_vrex_cagrad,
        risk_table=risk_table,
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    gradient_groups_path = output_dir / "gradient_groups.csv"
    solver_starts_path = output_dir / "solver_starts.csv"
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "artifact_manifest.json"
    _write_predictions(
        predictions_path,
        source_rows=rows,
        prior=prior_predictions,
        current=current_predictions,
    )
    _write_rows(gradient_groups_path, gradient_group_rows)
    _write_rows(solver_starts_path, solver_rows)
    summary: Dict[str, object] = {
        "protocol": "class1_boundary_balanced_cagrad_a0",
        "status": "passed" if gate["all_gates_passed"] else "closed",
        "provenance": provenance,
        "cohort": {
            **_cohort_serializable(cohorts),
            "holdout_boundary_hard_rows": len(holdout_hard_indices),
            "holdout_boundary_positive_rows": len(holdout_positive_indices),
        },
        "environments": environment_summaries,
        "model": {
            "class_names": class_names,
            "trainable_parameter_tensors": len(parameter_names),
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_schema_sha256": parameter_schema_sha256,
            "initial_state_sha256": initial_state_sha256,
            "boundary_cagrad_state_sha256": _state_sha256(candidate_model),
        },
        "gradient_computation": {
            "environment_loaders": environment_loader_summaries,
            "evaluation_loaders": evaluation_loader_summaries,
            "accumulation_dtype": "float64",
            "materialized_dtype": "float32",
            "model_mode": "eval",
        },
        "cagrad_geometry": cagrad_geometry,
        "steps": {
            "locked_boundary_erm": prior_erm_step,
            "boundary_cagrad": candidate_step,
            "update_norm_mismatch_relative": step_norm_mismatch,
        },
        "comparisons": {
            "clean_raw_cagrad": clean_raw_cagrad,
            "clean_raw_erm": clean_raw_erm,
            "clean_raw_vrex": clean_raw_vrex,
            "clean_erm_cagrad": clean_erm_cagrad,
            "clean_vrex_cagrad": clean_vrex_cagrad,
            "clean_margin_agem_cagrad": clean_margin_cagrad,
            "illumination_raw_cagrad": illumination_raw_cagrad,
            "illumination_erm_cagrad": illumination_erm_cagrad,
            "illumination_vrex_cagrad": illumination_vrex_cagrad,
        },
        "holdout_boundary_risks": risk_table,
        "prior_argmax_mismatches": prior_argmax_mismatches,
        "prior_raw_cidt_mismatches": raw_cidt_mismatches,
        "gate": gate,
        "validation_predictions_used": False,
        "test_data_used": False,
        "binary_model_artifacts_written": False,
        "artifacts": {
            "predictions": str(predictions_path.resolve()),
            "gradient_groups": str(gradient_groups_path.resolve()),
            "solver_starts": str(solver_starts_path.resolve()),
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
        [
            predictions_path,
            gradient_groups_path,
            solver_starts_path,
            report_path,
            summary_path,
        ],
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

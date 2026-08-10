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
from torch import Tensor, nn

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
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _git_commit,
    _make_loader,
    _prepare_output_dir,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


Gradient = list[Tensor]

CONDITIONS = ("clean", *(value[0] for value in LIGHTING_CONDITIONS))
VREX_BETA = 1.0
BOUNDARY_WEIGHT = 0.5

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_GEM_SUMMARY_SHA256 = "1d2bdd20837ee1300698fd144601dffa7a2cd0a3b3a7df17c7f801cb994a3e0c"
LOCKED_GEM_PREDICTIONS_SHA256 = "5e66813cb0e2cbe238d7eaccfe94202a3d8362550139a75236baffa02c67cfb3"
LOCKED_GEM_MANIFEST_SHA256 = "960bbc3957f7a0eca4ee01379831f024923e5a72510c1dcc94513ea39e0f4183"
LOCKED_PROTOCOL_SHA256 = "4a798895a44e61b2c889b03edca7817cf5e223b6af55c4dd2a05c26ce0a48246"
LOCKED_VREX_PAPER_SHA256 = "99886a5074c06c277511418d1588d635f22b6096551efd334b253911a6656df1"
LOCKED_DOMAINBED_COMMIT = "b93c22a1cfc3b2428398272c1a116c8de1f4139e"
LOCKED_DOMAINBED_ALGORITHMS_SHA256 = "6c2db72489f15d2ce155ab0490b14268bab16d793446d1ea5ebf2c7f22c4847b"
LOCKED_DOMAINBED_HPARAMS_SHA256 = "3a647bc462decebeccb0e15b3aa792afd34a5cfddac2136ae7fa668403d336ca"
LOCKED_DOMAINBED_LICENSE_SHA256 = "454f8bb89faa42f38408ef1301982d51e62a1aad411ea5cb594dff57f15a26e5"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only class-1 boundary-balanced V-REx information gate. "
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
        "--prior-gem-summary",
        type=Path,
        default=Path("runs/audit_class1_boundary_gem_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--prior-gem-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_gem_readiness_20260715/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--prior-gem-manifest",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_gem_readiness_20260715/artifact_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CLASS1_BOUNDARY_VREX_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--vrex-paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\vrex_icml2021.pdf"),
    )
    parser.add_argument(
        "--domainbed-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\DomainBed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_class1_boundary_vrex_readiness_20260715"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--focus-class", type=int, default=FOCUS_CLASS)
    parser.add_argument("--boundary-weight", type=float, default=BOUNDARY_WEIGHT)
    parser.add_argument("--vrex-beta", type=float, default=VREX_BETA)
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
        and math.isclose(float(args.vrex_beta), VREX_BETA, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(args.step_ratio), STEP_RATIO, rel_tol=0.0, abs_tol=1e-15)
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    domainbed_root = Path(args.domainbed_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "prior_gem_summary": Path(args.prior_gem_summary).resolve(),
        "prior_gem_predictions": Path(args.prior_gem_predictions).resolve(),
        "prior_gem_manifest": Path(args.prior_gem_manifest).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "vrex_paper": Path(args.vrex_paper).resolve(),
        "domainbed_algorithms": domainbed_root / "domainbed" / "algorithms.py",
        "domainbed_hparams": domainbed_root / "domainbed" / "hparams_registry.py",
        "domainbed_license": domainbed_root / "LICENSE",
    }


def _read_prior_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    holdout_indices: Sequence[int],
) -> tuple[Dict[str, Dict[str, list[Dict[str, object]]]], Dict[str, int]]:
    result = {
        condition: {name: [] for name in ("raw", "margin_agem")}
        for condition in CONDITIONS
    }
    argmax_mismatches = {name: 0 for name in ("raw", "margin_agem")}
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
        for name in ("raw", "margin_agem"):
            required.add(f"{name}_prediction")
            required.update(f"{name}_prob_{index}" for index in range(5))
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prior GEM prediction CSV is missing: {sorted(missing)}")
        for raw in reader:
            condition = str(raw["condition"]).strip()
            if condition not in result:
                raise ValueError(f"Unexpected prior GEM condition: {condition}")
            sample_index = int(raw["sample_index"])
            if not 0 <= sample_index < len(rows):
                raise ValueError("Prior GEM sample index is out of range.")
            source = rows[sample_index]
            if (
                source.source_stem != str(raw["source_stem"]).strip().casefold()
                or source.image_path != Path(str(raw["image_path"])).resolve()
                or source.fold != int(raw["fold"])
                or source.target != int(raw["target"])
            ):
                raise ValueError(f"Prior GEM source row mismatch at {sample_index}.")
            for name in ("raw", "margin_agem"):
                probabilities = [float(raw[f"{name}_prob_{index}"]) for index in range(5)]
                if not all(math.isfinite(value) and value >= 0.0 for value in probabilities):
                    raise ValueError("Prior GEM probabilities are invalid.")
                if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
                    raise ValueError("Prior GEM probabilities are not normalized.")
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
        reference = None
        for name, values in variants.items():
            values.sort(key=lambda row: int(row["sample_index"]))
            indices = [int(row["sample_index"]) for row in values]
            if len(indices) != EXPECTED_HOLDOUT_ROWS or len(set(indices)) != len(indices):
                raise ValueError(f"Prior {condition}/{name} row contract differs.")
            if indices != expected:
                raise ValueError(f"Prior {condition}/{name} differs from locked holdout.")
            if reference is None:
                reference = indices
            elif indices != reference:
                raise ValueError(f"Prior {condition} variant indices differ.")
    if any(argmax_mismatches.values()):
        raise ValueError(f"Prior GEM stored predictions differ from argmax: {argmax_mismatches}")
    return result, argmax_mismatches


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted V-REx protocol.")
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
        "prior_gem_summary": _verify_sha256(
            paths["prior_gem_summary"], LOCKED_GEM_SUMMARY_SHA256, "prior GEM summary"
        ),
        "prior_gem_predictions": _verify_sha256(
            paths["prior_gem_predictions"],
            LOCKED_GEM_PREDICTIONS_SHA256,
            "prior GEM predictions",
        ),
        "prior_gem_manifest": _verify_sha256(
            paths["prior_gem_manifest"], LOCKED_GEM_MANIFEST_SHA256, "prior GEM manifest"
        ),
        "protocol": _verify_sha256(paths["protocol"], LOCKED_PROTOCOL_SHA256, "V-REx protocol"),
        "vrex_paper": _verify_sha256(
            paths["vrex_paper"], LOCKED_VREX_PAPER_SHA256, "V-REx paper"
        ),
        "domainbed_algorithms": _verify_sha256(
            paths["domainbed_algorithms"],
            LOCKED_DOMAINBED_ALGORITHMS_SHA256,
            "DomainBed algorithms",
        ),
        "domainbed_hparams": _verify_sha256(
            paths["domainbed_hparams"], LOCKED_DOMAINBED_HPARAMS_SHA256, "DomainBed hparams"
        ),
        "domainbed_license": _verify_sha256(
            paths["domainbed_license"], LOCKED_DOMAINBED_LICENSE_SHA256, "DomainBed license"
        ),
    }
    domainbed_root = Path(args.domainbed_root).resolve()
    commit = _git_commit(domainbed_root)
    clean = _git_worktree_clean(domainbed_root)
    if commit != LOCKED_DOMAINBED_COMMIT:
        raise ValueError(f"DomainBed commit differs: {commit} != {LOCKED_DOMAINBED_COMMIT}")
    if not clean:
        raise ValueError("DomainBed worktree is not clean.")
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    prior_summary = json.loads(paths["prior_gem_summary"].read_text(encoding="utf-8"))
    for name, payload in (("CIDT", cidt_summary), ("prior GEM", prior_summary)):
        if bool(payload.get("test_data_used", True)):
            raise ValueError(f"{name} provenance indicates test data use.")
        if bool(payload.get("validation_predictions_used", True)):
            raise ValueError(f"{name} provenance indicates validation prediction use.")
    if bool(prior_summary.get("binary_model_artifacts_written", True)):
        raise ValueError("Prior GEM summary indicates a binary model artifact.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    prior, argmax_mismatches = _read_prior_predictions(
        paths["prior_gem_predictions"],
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
        "domainbed_commit": commit,
        "domainbed_worktree_clean": clean,
        "cohort": _cohort_serializable(cohorts),
        "prior_prediction_conditions": {
            condition: len(values["raw"]) for condition, values in prior.items()
        },
        "prior_argmax_mismatches": argmax_mismatches,
        "prior_raw_cidt_mismatches": raw_cidt_mismatches,
        "validation_predictions_used": False,
        "test_data_used": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _weighted_sum_gradients(
    gradients: Sequence[Sequence[Tensor]], weights: Sequence[float]
) -> Gradient:
    if not gradients or len(gradients) != len(weights):
        raise ValueError("Gradient/weight counts differ or are empty.")
    width = len(gradients[0])
    if width == 0 or any(len(values) != width for values in gradients):
        raise ValueError("Gradient schemas differ or are empty.")
    result: Gradient = []
    for tensors in zip(*gradients):
        shape = tensors[0].shape
        if any(value.shape != shape for value in tensors):
            raise ValueError("Gradient tensor shapes differ.")
        accumulated = torch.zeros(shape, dtype=torch.float64)
        for value, weight in zip(tensors, weights):
            accumulated.add_(value.detach().cpu().double(), alpha=float(weight))
        result.append(accumulated.float())
    return result


def _flatten_gradient(values: Sequence[Tensor]) -> np.ndarray:
    return np.concatenate(
        [value.detach().cpu().double().reshape(-1).numpy() for value in values]
    )


def _balanced_environment_gradient(
    hard: Sequence[Tensor],
    positive: Sequence[Tensor],
    *,
    weight: float,
) -> tuple[Gradient, float]:
    environment = _weighted_sum_gradients(
        [hard, positive], [float(weight), 1.0 - float(weight)]
    )
    flat_reference = (
        float(weight) * _flatten_gradient(hard)
        + (1.0 - float(weight)) * _flatten_gradient(positive)
    ).astype(np.float32).astype(np.float64)
    flat_observed = _flatten_gradient(environment)
    error = float(np.max(np.abs(flat_observed - flat_reference)))
    return environment, error


def compose_vrex_gradients(
    risks: Sequence[float],
    environment_gradients: Sequence[Sequence[Tensor]],
    *,
    beta: float,
) -> tuple[Gradient, Gradient, Gradient, Dict[str, object]]:
    risk_array = np.asarray(risks, dtype=np.float64).reshape(-1)
    if risk_array.size != len(environment_gradients) or risk_array.size < 2:
        raise ValueError("V-REx risk/gradient environment counts differ.")
    if not np.isfinite(risk_array).all():
        raise ValueError("V-REx risks are nonfinite.")
    count = int(risk_array.size)
    mean = float(risk_array.mean())
    centered = risk_array - mean
    variance = float(np.mean(centered**2))
    erm_weights = np.full(count, 1.0 / count, dtype=np.float64)
    variance_weights = (2.0 / count) * centered
    vrex_weights = erm_weights + float(beta) * variance_weights
    erm = _weighted_sum_gradients(environment_gradients, erm_weights)
    variance_gradient = _weighted_sum_gradients(
        environment_gradients, variance_weights
    )
    vrex = _weighted_sum_gradients(environment_gradients, vrex_weights)

    matrix = np.stack([_flatten_gradient(values) for values in environment_gradients])
    references = {
        "erm": np.sum(erm_weights[:, None] * matrix, axis=0),
        "variance": np.sum(variance_weights[:, None] * matrix, axis=0),
        "vrex": np.sum(vrex_weights[:, None] * matrix, axis=0),
    }
    observed = {
        "erm": _flatten_gradient(erm),
        "variance": _flatten_gradient(variance_gradient),
        "vrex": _flatten_gradient(vrex),
    }
    flat_errors = {
        name: float(
            np.max(
                np.abs(
                    values.astype(np.float32).astype(np.float64) - observed[name]
                )
            )
        )
        for name, values in references.items()
    }
    objective_gradient_squared = _gradient_dot(vrex, vrex)
    variance_dot = _gradient_dot(variance_gradient, vrex)
    return erm, variance_gradient, vrex, {
        "environment_risks": risk_array.tolist(),
        "risk_mean": mean,
        "risk_population_variance": variance,
        "beta": float(beta),
        "erm_weights": erm_weights.tolist(),
        "variance_weights": variance_weights.tolist(),
        "vrex_weights": vrex_weights.tolist(),
        "erm_gradient_norm": _gradient_norm(erm),
        "variance_gradient_norm": _gradient_norm(variance_gradient),
        "vrex_gradient_norm": _gradient_norm(vrex),
        "vrex_vs_erm_relative_l2": _gradient_relative_difference(vrex, erm),
        "flat_max_abs_errors": flat_errors,
        "candidate_first_order_objective_change": -objective_gradient_squared,
        "candidate_first_order_variance_change": -variance_dot,
        "candidate_first_order_mean_risk_change": -_gradient_dot(erm, vrex),
        "finite": bool(
            np.isfinite(risk_array).all()
            and all(math.isfinite(value) for value in flat_errors.values())
            and all(
                bool(torch.isfinite(value).all().item())
                for gradient in (erm, variance_gradient, vrex)
                for value in gradient
            )
        ),
    }


def _make_condition_loader(
    *,
    condition: str,
    brightness: Optional[float],
    contrast: Optional[float],
    base_dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    args: argparse.Namespace,
    context: str,
    seed: int,
):
    if condition == "clean":
        return _make_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=context,
            seed=seed,
        )
    if brightness is None or contrast is None:
        raise ValueError("Lighting condition requires brightness and contrast.")
    return _make_lighting_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        brightness=float(brightness),
        contrast=float(contrast),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=context,
        seed=seed,
    )


def boundary_risk_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    hard_indices: Sequence[int],
    positive_indices: Sequence[int],
    weight: float = BOUNDARY_WEIGHT,
) -> Dict[str, float | int]:
    indexed = {int(row["sample_index"]): row for row in rows}

    def mean_nll(indices: Sequence[int]) -> float:
        values = []
        for sample_index in indices:
            row = indexed.get(int(sample_index))
            if row is None:
                raise ValueError(f"Boundary-risk row is missing: {sample_index}")
            target = int(row["target"])
            probability = float(row[f"prob_{target}"])
            if not math.isfinite(probability) or probability <= 0.0:
                raise ValueError("Boundary-risk probability is invalid.")
            values.append(-math.log(max(probability, 1e-30)))
        if not values:
            raise ValueError("Boundary-risk cohort is empty.")
        return sum(values) / len(values)

    hard = mean_nll(hard_indices)
    positive = mean_nll(positive_indices)
    risk = float(weight) * hard + (1.0 - float(weight)) * positive
    return {
        "hard_rows": len(hard_indices),
        "positive_rows": len(positive_indices),
        "hard_cross_entropy": hard,
        "positive_cross_entropy": positive,
        "boundary_risk": risk,
    }


def _risk_table(
    *,
    prior: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    current: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    hard_indices: Sequence[int],
    positive_indices: Sequence[int],
) -> Dict[str, object]:
    variants: Dict[str, list[Dict[str, object]]] = {
        name: [] for name in ("raw", "margin_agem", "boundary_erm", "boundary_vrex")
    }
    for condition in CONDITIONS:
        sources = {
            "raw": prior[condition]["raw"],
            "margin_agem": prior[condition]["margin_agem"],
            "boundary_erm": current[condition]["boundary_erm"],
            "boundary_vrex": current[condition]["boundary_vrex"],
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
    variance: Sequence[Tensor],
    vrex: Sequence[Tensor],
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
                "variance_squared": 0.0,
                "vrex_squared": 0.0,
                **{
                    f"environment_{index}_squared": 0.0
                    for index in range(len(environment_gradients))
                },
            },
        )
        row["parameters"] += 1
        row["elements"] += int(erm[parameter_index].numel())
        for key, values in (("erm", erm), ("variance", variance), ("vrex", vrex)):
            row[f"{key}_squared"] += float(
                values[parameter_index].detach().cpu().double().square().sum().item()
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
            "variance_norm": math.sqrt(values["variance_squared"]),
            "vrex_norm": math.sqrt(values["vrex_squared"]),
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
    clean_raw_control: Mapping[str, object],
    clean_control_candidate: Mapping[str, object],
    clean_margin_candidate: Mapping[str, object],
    illumination_raw_candidate: Sequence[Mapping[str, object]],
    illumination_raw_control: Sequence[Mapping[str, object]],
    risk_table: Mapping[str, object],
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
        "candidate_f1_gain_vs_erm_gte_0p002": float(
            clean_control_candidate["delta"]["class1_f1"]
        )
        >= 0.002,
        "candidate_tp_breaks_no_greater_than_erm": int(
            transitions["focus_tp_break"]
        )
        <= int(clean_raw_control["transitions"]["focus_tp_break"]),
        "candidate_within_0p002_margin_agem_f1": float(
            clean_margin_candidate["delta"]["class1_f1"]
        )
        >= -0.002,
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
    control_by_condition = {
        str(row["condition"]): row for row in illumination_raw_control
    }
    candidate_by_condition = {
        str(row["condition"]): row for row in illumination_raw_candidate
    }
    control_worst = min(
        float(row["candidate"]["per_class_f1"][FOCUS_CLASS])
        for row in illumination_raw_control
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
        "worst_condition_class1_f1_gain_gte_0p010": candidate_worst - raw_worst
        >= 0.010,
        "worst_condition_class1_f1_gain_vs_erm_gte_0p005": candidate_worst
        - control_worst
        >= 0.005,
        "condition_sets_match": set(control_by_condition) == set(candidate_by_condition),
    }
    erm_risk = risk_table["boundary_erm"]
    vrex_risk = risk_table["boundary_vrex"]
    risk_checks = {
        "candidate_boundary_risk_variance_lte_90pct_erm": float(
            vrex_risk["population_variance"]
        )
        <= 0.90 * float(erm_risk["population_variance"]),
        "candidate_boundary_risk_mean_lte_erm_plus_0p005": float(vrex_risk["mean"])
        <= float(erm_risk["mean"]) + 0.005,
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
    variants = ("raw", "margin_agem", "boundary_erm", "boundary_vrex")
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
                "margin_agem": prior[condition]["margin_agem"],
                "boundary_erm": current[condition]["boundary_erm"],
                "boundary_vrex": current[condition]["boundary_vrex"],
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
    clean = summary["comparisons"]["clean_raw_vrex"]
    geometry = summary["vrex_geometry"]
    lines = [
        "# Class-1 Boundary V-REx Result",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Environment risk variance: `{geometry['risk_population_variance']:.9f}`",
        f"- V-REx vs ERM direction L2: `{geometry['vrex_vs_erm_relative_l2']:.9f}`",
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
        raise RuntimeError("Locked boundary V-REx A0 requires CUDA.")
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    if _is_relative_to(output_path, paths["data"].parent):
        raise ValueError("Audit output cannot be written inside the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    prior_predictions, prior_argmax_mismatches = _read_prior_predictions(
        paths["prior_gem_predictions"],
        rows=rows,
        holdout_indices=cohorts["holdout_indices"],
    )

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
    environment_risks: list[float] = []
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
            context=f"class1_boundary_vrex_{condition}_hard",
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
            context=f"class1_boundary_vrex_{condition}_positive",
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
        environment_risks.append(environment_risk)
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

    erm_direction, variance_direction, vrex_direction, vrex_geometry = (
        compose_vrex_gradients(
            environment_risks,
            environment_gradients,
            beta=float(args.vrex_beta),
        )
    )
    gradient_group_rows = _gradient_group_rows(
        parameter_names,
        environment_gradients,
        erm_direction,
        variance_direction,
        vrex_direction,
    )

    erm_model = copy.deepcopy(prototype).eval()
    vrex_model = copy.deepcopy(prototype).eval()
    erm_initial_sha256 = _state_sha256(erm_model)
    vrex_initial_sha256 = _state_sha256(vrex_model)
    _, erm_schema_sha256 = _parameter_schema(erm_model)
    _, vrex_schema_sha256 = _parameter_schema(vrex_model)
    erm_step = _apply_normalized_parameter_step(
        erm_model, erm_direction, parameter_ratio=float(args.step_ratio)
    )
    vrex_step = _apply_normalized_parameter_step(
        vrex_model, vrex_direction, parameter_ratio=float(args.step_ratio)
    )

    models = {"boundary_erm": erm_model, "boundary_vrex": vrex_model}
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
    clean_current = current_predictions["clean"]
    clean_raw_vrex = _aligned_comparison(
        control_rows=clean_prior["raw"], candidate_rows=clean_current["boundary_vrex"]
    )
    clean_raw_erm = _aligned_comparison(
        control_rows=clean_prior["raw"], candidate_rows=clean_current["boundary_erm"]
    )
    clean_erm_vrex = _aligned_comparison(
        control_rows=clean_current["boundary_erm"],
        candidate_rows=clean_current["boundary_vrex"],
    )
    clean_margin_vrex = _aligned_comparison(
        control_rows=clean_prior["margin_agem"],
        candidate_rows=clean_current["boundary_vrex"],
    )
    illumination_raw_vrex = []
    illumination_raw_erm = []
    for condition, _, _ in LIGHTING_CONDITIONS:
        raw_vrex = _aligned_comparison(
            control_rows=prior_predictions[condition]["raw"],
            candidate_rows=current_predictions[condition]["boundary_vrex"],
        )
        raw_vrex["condition"] = condition
        illumination_raw_vrex.append(raw_vrex)
        raw_erm = _aligned_comparison(
            control_rows=prior_predictions[condition]["raw"],
            candidate_rows=current_predictions[condition]["boundary_erm"],
        )
        raw_erm["condition"] = condition
        illumination_raw_erm.append(raw_erm)

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
    step_norm_mismatch = abs(
        erm_step["actual_step_norm"] - vrex_step["actual_step_norm"]
    ) / max(1e-12, erm_step["actual_step_norm"])
    gradient_summaries = [
        summary[key]
        for summary in environment_summaries
        for key in ("hard", "positive")
    ]
    structural_checks = {
        "locked_sources_verified": True,
        "domainbed_worktree_clean": bool(provenance["domainbed_worktree_clean"]),
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
        "vrex_geometry_finite": bool(vrex_geometry["finite"]),
        "environment_risk_variance_gte_1e_minus_8": float(
            vrex_geometry["risk_population_variance"]
        )
        >= 1e-8,
        "variance_gradient_nonzero": float(vrex_geometry["variance_gradient_norm"])
        > 0.0,
        "vrex_differs_from_erm_gte_1e_minus_4": float(
            vrex_geometry["vrex_vs_erm_relative_l2"]
        )
        >= 1e-4,
        "environment_flat_errors_lte_1e_minus_7": max(environment_flat_errors)
        <= 1e-7,
        "composed_flat_errors_lte_1e_minus_7": max(
            float(value) for value in vrex_geometry["flat_max_abs_errors"].values()
        )
        <= 1e-7,
        "first_order_vrex_objective_decreases": float(
            vrex_geometry["candidate_first_order_objective_change"]
        )
        < 0.0,
        "first_order_variance_nonincreasing": float(
            vrex_geometry["candidate_first_order_variance_change"]
        )
        <= 0.0,
        "initial_states_identical": initial_state_sha256
        == erm_initial_sha256
        == vrex_initial_sha256,
        "parameter_schemas_identical": parameter_schema_sha256
        == erm_schema_sha256
        == vrex_schema_sha256,
        "all_trainable_parameters_included": len(parameter_names)
        == len(erm_direction)
        == len(vrex_direction),
        "control_update_ratio_exact": abs(erm_step["actual_ratio"] - STEP_RATIO)
        <= 1e-8,
        "candidate_update_ratio_exact": abs(vrex_step["actual_ratio"] - STEP_RATIO)
        <= 1e-8,
        "matched_update_norms": step_norm_mismatch <= 1e-7,
        "no_validation_or_test_access": True,
        "no_raw_data_write": not _is_relative_to(output_dir, paths["data"].parent),
        "no_binary_model_artifact": True,
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        clean_raw_candidate=clean_raw_vrex,
        clean_raw_control=clean_raw_erm,
        clean_control_candidate=clean_erm_vrex,
        clean_margin_candidate=clean_margin_vrex,
        illumination_raw_candidate=illumination_raw_vrex,
        illumination_raw_control=illumination_raw_erm,
        risk_table=risk_table,
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
        "protocol": "class1_boundary_balanced_vrex_a0",
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
            "boundary_erm_state_sha256": _state_sha256(erm_model),
            "boundary_vrex_state_sha256": _state_sha256(vrex_model),
        },
        "gradient_computation": {
            "environment_loaders": environment_loader_summaries,
            "evaluation_loaders": evaluation_loader_summaries,
            "accumulation_dtype": "float64",
            "materialized_dtype": "float32",
            "model_mode": "eval",
        },
        "vrex_geometry": vrex_geometry,
        "steps": {
            "boundary_erm": erm_step,
            "boundary_vrex": vrex_step,
            "update_norm_mismatch_relative": step_norm_mismatch,
        },
        "comparisons": {
            "clean_raw_vrex": clean_raw_vrex,
            "clean_raw_erm": clean_raw_erm,
            "clean_erm_vrex": clean_erm_vrex,
            "clean_margin_agem_vrex": clean_margin_vrex,
            "illumination_raw_vrex": illumination_raw_vrex,
            "illumination_raw_erm": illumination_raw_erm,
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

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from trkh.data.dataset import StrictBalancedBatchSampler
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _assign_source_folds,
)
from trkh.training.confusion_spectral import confusion_aware_spectral_regularizer


EXPECTED_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
MODEL_NAMES = ("keeper", "candidate")
METHODS = (("car", False), ("bicar", True))
COHORTS = (
    "focus_tp",
    "focus_fn",
    "focus_fp",
    "nonfocus_not_fp",
    "focus_all",
    "nonfocus_all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paper CAR and frequency-symmetric BiCAR logit-gradient direction "
            "on locked train-only predictions before trainer integration."
        )
    )
    parser.add_argument("--cidt-summary", type=Path, required=True)
    parser.add_argument("--expected-cidt-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--ema-momentum", type=float, default=0.5)
    parser.add_argument("--frequency-smoothing", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--paper-weight", type=float, default=0.5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_clean_rows(
    path: Path,
    *,
    num_classes: int,
) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    clean_rows = [row for row in rows if str(row.get("condition", "")) == "clean"]
    if len(clean_rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} clean train rows, found {len(clean_rows)}."
        )
    sample_indices = [int(row["sample_index"]) for row in clean_rows]
    if sample_indices != list(range(EXPECTED_ROWS)):
        raise ValueError("Clean prediction rows changed ordered sample identity.")
    arrays: Dict[str, np.ndarray] = {
        "targets": np.asarray(
            [int(row["target_index"]) for row in clean_rows], dtype=np.int64
        )
    }
    for model_name in MODEL_NAMES:
        probabilities = np.asarray(
            [
                [float(row[f"{model_name}_prob_{index}"]) for index in range(num_classes)]
                for row in clean_rows
            ],
            dtype=np.float32,
        )
        if not np.isfinite(probabilities).all() or bool((probabilities < 0.0).any()):
            raise ValueError(f"Invalid {model_name} probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"Unnormalized {model_name} probabilities.")
        arrays[model_name] = probabilities
    return clean_rows, arrays


def _cohort_name(target: int, prediction: int, focus_class: int) -> str:
    if target == focus_class:
        return "focus_tp" if prediction == focus_class else "focus_fn"
    return "focus_fp" if prediction == focus_class else "nonfocus_not_fp"


def _summarize_values(values: Sequence[float], *, desired_positive: bool) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "desired_sign_fraction": 0.0,
        }
    desired = array > 0.0 if desired_positive else array < 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(median(array.tolist())),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "desired_sign_fraction": float(desired.mean()),
    }


def _directional_probability_derivative(
    probabilities: torch.Tensor,
    gradient: torch.Tensor,
    *,
    focus_class: int,
) -> torch.Tensor:
    descent_direction = -gradient
    expected_direction = (probabilities * descent_direction).sum(dim=1)
    return probabilities[:, focus_class] * (
        descent_direction[:, focus_class] - expected_direction
    )


def _audit_model_method(
    *,
    model_name: str,
    method_name: str,
    bidirectional: bool,
    probabilities: np.ndarray,
    targets: np.ndarray,
    rows: Sequence[Mapping[str, str]],
    fold_assignments: np.ndarray,
    sampler_batches: Sequence[Sequence[int]],
    class_counts: Sequence[int],
    focus_class: int,
    ema_momentum: float,
    frequency_smoothing: float,
    margin: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    previous_ema = None
    batch_rows: List[Dict[str, object]] = []
    occurrence_rows: List[Dict[str, object]] = []
    gradient_norm_ratios: List[float] = []
    gradient_cosines: List[float] = []
    losses: List[float] = []
    every_batch_has_all_classes = True

    for batch_index, sample_indices in enumerate(sampler_batches):
        batch_targets = torch.as_tensor(
            targets[list(sample_indices)], dtype=torch.long
        )
        batch_probabilities = torch.as_tensor(
            probabilities[list(sample_indices)], dtype=torch.float32
        ).clamp_min(1e-8)
        logits = batch_probabilities.log().requires_grad_(True)
        result = confusion_aware_spectral_regularizer(
            logits,
            batch_targets,
            class_counts=class_counts,
            previous_ema=previous_ema,
            momentum=ema_momentum,
            smoothing=frequency_smoothing,
            margin=margin,
            bidirectional=bidirectional,
        )
        previous_ema = result.ema_confusion
        car_gradient = torch.autograd.grad(result.loss, logits, retain_graph=True)[0]
        ce_loss = F.cross_entropy(logits, batch_targets)
        ce_gradient = torch.autograd.grad(ce_loss, logits)[0]
        car_norm = float(car_gradient.norm().detach().cpu().item())
        ce_norm = float(ce_gradient.norm().detach().cpu().item())
        norm_ratio = car_norm / max(ce_norm, 1e-12)
        cosine = float(
            F.cosine_similarity(
                car_gradient.flatten(), ce_gradient.flatten(), dim=0
            )
            .detach()
            .cpu()
            .item()
        )
        focus_derivative = _directional_probability_derivative(
            batch_probabilities,
            car_gradient,
            focus_class=focus_class,
        )
        predictions = batch_probabilities.argmax(dim=1)
        class_histogram = torch.bincount(
            batch_targets, minlength=len(class_counts)
        ).tolist()
        every_batch_has_all_classes = every_batch_has_all_classes and all(
            int(value) > 0 for value in class_histogram
        )
        loss_value = float(result.loss.detach().cpu().item())
        losses.append(loss_value)
        gradient_norm_ratios.append(norm_ratio)
        gradient_cosines.append(cosine)
        batch_rows.append(
            {
                "model": model_name,
                "method": method_name,
                "batch_index": int(batch_index),
                "rows": int(len(sample_indices)),
                "class_histogram": json.dumps(class_histogram, separators=(",", ":")),
                "loss": loss_value,
                "car_gradient_norm": car_norm,
                "ce_gradient_norm": ce_norm,
                "gradient_norm_ratio": norm_ratio,
                "gradient_cosine_to_ce": cosine,
            }
        )
        for position, sample_index in enumerate(sample_indices):
            target = int(batch_targets[position].item())
            prediction = int(predictions[position].item())
            cohort = _cohort_name(target, prediction, focus_class)
            occurrence_rows.append(
                {
                    "model": model_name,
                    "method": method_name,
                    "batch_index": int(batch_index),
                    "batch_position": int(position),
                    "sample_index": int(sample_index),
                    "source_stem": str(rows[int(sample_index)]["source_stem"]),
                    "image_path": str(rows[int(sample_index)]["image_path"]),
                    "fold": int(fold_assignments[int(sample_index)]),
                    "target_index": target,
                    "prediction_index": prediction,
                    "cohort": cohort,
                    "focus_probability": float(batch_probabilities[position, focus_class]),
                    "focus_probability_descent_derivative": float(
                        focus_derivative[position].detach().cpu().item()
                    ),
                    "gradient_l2": float(
                        car_gradient[position].norm().detach().cpu().item()
                    ),
                }
            )

    cohort_values: Dict[str, List[float]] = {name: [] for name in COHORTS}
    unique_samples: Dict[str, set[int]] = {name: set() for name in COHORTS}
    pair_values: Dict[str, List[float]] = {}
    for row in occurrence_rows:
        cohort = str(row["cohort"])
        value = float(row["focus_probability_descent_derivative"])
        sample_index = int(row["sample_index"])
        cohort_values[cohort].append(value)
        unique_samples[cohort].add(sample_index)
        if int(row["target_index"]) == focus_class:
            cohort_values["focus_all"].append(value)
            unique_samples["focus_all"].add(sample_index)
        else:
            cohort_values["nonfocus_all"].append(value)
            unique_samples["nonfocus_all"].add(sample_index)
        pair_name = f"{int(row['target_index'])}->{int(row['prediction_index'])}"
        pair_values.setdefault(pair_name, []).append(value)

    cohorts = {}
    for name in COHORTS:
        summary = _summarize_values(
            cohort_values[name], desired_positive=name in {"focus_tp", "focus_fn", "focus_all"}
        )
        summary["unique_samples"] = int(len(unique_samples[name]))
        cohorts[name] = summary
    fp_magnitude = abs(float(cohorts["focus_fp"]["mean"]))
    nonfp_magnitude = abs(float(cohorts["nonfocus_not_fp"]["mean"]))
    tp_magnitude = abs(float(cohorts["focus_tp"]["mean"]))
    fn_magnitude = abs(float(cohorts["focus_fn"]["mean"]))
    summary = {
        "model": model_name,
        "method": method_name,
        "bidirectional": bool(bidirectional),
        "batches": int(len(sampler_batches)),
        "occurrences": int(len(occurrence_rows)),
        "every_batch_has_all_classes": bool(every_batch_has_all_classes),
        "loss": {
            "mean": float(np.mean(losses)),
            "minimum": float(np.min(losses)),
            "maximum": float(np.max(losses)),
            "last": float(losses[-1]),
        },
        "gradient": {
            "norm_ratio_to_ce_mean": float(np.mean(gradient_norm_ratios)),
            "cosine_to_ce_mean": float(np.mean(gradient_cosines)),
        },
        "cohorts": cohorts,
        "focus_fp_vs_nonfp_suppression_ratio": float(
            fp_magnitude / max(nonfp_magnitude, 1e-12)
        ),
        "focus_fn_vs_tp_boost_ratio": float(fn_magnitude / max(tp_magnitude, 1e-12)),
        "pair_direction": {
            name: _summarize_values(
                values,
                desired_positive=int(name.split("->", 1)[0]) == focus_class,
            )
            for name, values in sorted(pair_values.items())
        },
    }
    return summary, occurrence_rows, batch_rows


def _fold_summaries(
    occurrence_rows: Sequence[Mapping[str, object]],
    *,
    focus_class: int,
    folds: int,
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    combinations = sorted(
        {(str(row["model"]), str(row["method"])) for row in occurrence_rows}
    )
    for model_name, method_name in combinations:
        selected_method = [
            row
            for row in occurrence_rows
            if str(row["model"]) == model_name and str(row["method"]) == method_name
        ]
        for fold in range(folds):
            selected = [row for row in selected_method if int(row["fold"]) == fold]
            positive = [
                float(row["focus_probability_descent_derivative"])
                for row in selected
                if int(row["target_index"]) == focus_class
            ]
            negative = [
                float(row["focus_probability_descent_derivative"])
                for row in selected
                if int(row["target_index"]) != focus_class
            ]
            focus_fp = [
                float(row["focus_probability_descent_derivative"])
                for row in selected
                if str(row["cohort"]) == "focus_fp"
            ]
            nonfocus_not_fp = [
                float(row["focus_probability_descent_derivative"])
                for row in selected
                if str(row["cohort"]) == "nonfocus_not_fp"
            ]
            positive_summary = _summarize_values(positive, desired_positive=True)
            negative_summary = _summarize_values(negative, desired_positive=False)
            fp_summary = _summarize_values(focus_fp, desired_positive=False)
            nonfp_summary = _summarize_values(nonfocus_not_fp, desired_positive=False)
            summaries.append(
                {
                    "model": model_name,
                    "method": method_name,
                    "fold": int(fold),
                    "rows": int(len(selected)),
                    "focus_rows": int(len(positive)),
                    "nonfocus_rows": int(len(negative)),
                    "focus_direction_fraction": float(
                        positive_summary["desired_sign_fraction"]
                    ),
                    "nonfocus_direction_fraction": float(
                        negative_summary["desired_sign_fraction"]
                    ),
                    "focus_fp_suppression_mean": float(fp_summary["mean"]),
                    "nonfocus_not_fp_suppression_mean": float(nonfp_summary["mean"]),
                    "focus_fp_vs_nonfp_ratio": float(
                        abs(float(fp_summary["mean"]))
                        / max(abs(float(nonfp_summary["mean"])), 1e-12)
                    ),
                }
            )
    return summaries


def audit_readiness(args: argparse.Namespace) -> Dict[str, object]:
    summary_path = args.cidt_summary.resolve()
    summary_hash = _sha256(summary_path)
    if summary_hash != str(args.expected_cidt_summary_sha256).lower():
        raise ValueError(
            f"CIDT summary SHA mismatch: expected {args.expected_cidt_summary_sha256}, "
            f"found {summary_hash}."
        )
    cidt_summary = _read_json(summary_path)
    if cidt_summary.get("split") != "train":
        raise ValueError("CIDT source must be train-only.")
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT source reports test use.")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT source reports validation prediction use.")
    artifacts = cidt_summary.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts.get("predictions"):
        raise ValueError("CIDT summary is missing its prediction artifact.")
    predictions_path = (summary_path.parent / str(artifacts["predictions"])).resolve()
    prediction_hash = _sha256(predictions_path)
    class_names = [str(value) for value in cidt_summary.get("class_names", [])]
    if len(class_names) != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Unexpected class contract in CIDT summary.")
    rows, arrays = _load_clean_rows(predictions_path, num_classes=len(class_names))
    targets = arrays["targets"]
    class_counts = tuple(
        int(value) for value in np.bincount(targets, minlength=len(class_names)).tolist()
    )
    source_stems = np.asarray([str(row["source_stem"]) for row in rows], dtype=str)
    fold_assignments, fold_contract = _assign_source_folds(
        targets,
        source_stems,
        folds=int(args.folds),
        seed=int(args.seed),
    )
    sampler = StrictBalancedBatchSampler(
        targets.tolist(),
        batch_size=int(args.batch_size),
        num_classes=len(class_names),
        epoch_multiplier=1.0,
        seed=int(args.seed),
        drop_last=False,
    )
    sampler_batches = [list(batch) for batch in sampler]
    sampler_exposure = sampler.exposure_summary()

    method_summaries: Dict[str, Dict[str, object]] = {}
    occurrence_rows: List[Dict[str, object]] = []
    batch_rows: List[Dict[str, object]] = []
    for model_name in MODEL_NAMES:
        method_summaries[model_name] = {}
        for method_name, bidirectional in METHODS:
            method_summary, method_occurrences, method_batches = _audit_model_method(
                model_name=model_name,
                method_name=method_name,
                bidirectional=bidirectional,
                probabilities=arrays[model_name],
                targets=targets,
                rows=rows,
                fold_assignments=fold_assignments,
                sampler_batches=sampler_batches,
                class_counts=class_counts,
                focus_class=int(args.focus_class),
                ema_momentum=float(args.ema_momentum),
                frequency_smoothing=float(args.frequency_smoothing),
                margin=float(args.margin),
            )
            method_summaries[model_name][method_name] = method_summary
            occurrence_rows.extend(method_occurrences)
            batch_rows.extend(method_batches)
    fold_rows = _fold_summaries(
        occurrence_rows,
        focus_class=int(args.focus_class),
        folds=int(args.folds),
    )

    candidate_car = method_summaries["candidate"]["car"]
    candidate_bicar = method_summaries["candidate"]["bicar"]
    bicar_cohorts = candidate_bicar["cohorts"]
    car_cohorts = candidate_car["cohorts"]
    bicar_fold_rows = [
        row
        for row in fold_rows
        if row["model"] == "candidate" and row["method"] == "bicar"
    ]
    fp_gain = abs(float(bicar_cohorts["focus_fp"]["mean"])) / max(
        abs(float(car_cohorts["focus_fp"]["mean"])), 1e-12
    )
    fn_gain = abs(float(bicar_cohorts["focus_fn"]["mean"])) / max(
        abs(float(car_cohorts["focus_fn"]["mean"])), 1e-12
    )
    weighted_gradient_ratio = float(args.paper_weight) * float(
        candidate_bicar["gradient"]["norm_ratio_to_ce_mean"]
    )
    structural_gates = {
        "cidt_train_only_contract": cidt_summary.get("split") == "train"
        and not bool(cidt_summary.get("test_data_used"))
        and not bool(cidt_summary.get("validation_predictions_used")),
        "full_clean_rows_exact": len(rows) == EXPECTED_ROWS,
        "class_counts_exact": class_counts == EXPECTED_CLASS_COUNTS,
        "ordered_sample_identity_exact": [int(row["sample_index"]) for row in rows]
        == list(range(EXPECTED_ROWS)),
        "source_fold_assignment_complete": bool(fold_contract["assignment_complete"]),
        "source_fold_overlap_zero": int(fold_contract["source_overlap"]) == 0,
        "strict_balanced_exposure_tolerance": float(sampler_exposure["relative_gap"])
        <= 0.001,
        "every_batch_has_all_classes": all(
            bool(value[method]["every_batch_has_all_classes"])
            for value in method_summaries.values()
            for method, _ in METHODS
        ),
        "all_gradients_finite": all(
            math.isfinite(float(row["focus_probability_descent_derivative"]))
            and math.isfinite(float(row["gradient_l2"]))
            for row in occurrence_rows
        ),
    }
    readiness_gates = {
        "candidate_focus_direction_099": float(
            bicar_cohorts["focus_all"]["desired_sign_fraction"]
        )
        >= 0.99,
        "candidate_nonfocus_direction_099": float(
            bicar_cohorts["nonfocus_all"]["desired_sign_fraction"]
        )
        >= 0.99,
        "candidate_fp_focus_ratio_225": float(
            candidate_bicar["focus_fp_vs_nonfp_suppression_ratio"]
        )
        >= 2.25,
        "candidate_fn_tp_ratio_080": float(
            candidate_bicar["focus_fn_vs_tp_boost_ratio"]
        )
        >= 0.80,
        "bicar_vs_car_fp_gain_300": fp_gain >= 3.0,
        "bicar_vs_car_fn_gain_115": fn_gain >= 1.15,
        "weighted_gradient_ratio_015_030": 0.15 <= weighted_gradient_ratio <= 0.30,
        "gradient_cosine_090_099": 0.90
        <= float(candidate_bicar["gradient"]["cosine_to_ce_mean"])
        <= 0.99,
        "all_folds_signed": len(bicar_fold_rows) == int(args.folds)
        and all(
            float(row["focus_direction_fraction"]) >= 0.99
            and float(row["nonfocus_direction_fraction"]) >= 0.99
            for row in bicar_fold_rows
        ),
        "all_folds_fp_focus_ratio_175": len(bicar_fold_rows) == int(args.folds)
        and all(float(row["focus_fp_vs_nonfp_ratio"]) >= 1.75 for row in bicar_fold_rows),
    }
    scope_gates_passed = all(structural_gates.values())
    all_gates_passed = scope_gates_passed and all(readiness_gates.values())

    output_dir = args.output_dir.resolve()
    occurrence_path = output_dir / "gradient_occurrences.csv"
    batch_path = output_dir / "batch_summary.csv"
    fold_path = output_dir / "fold_summary.csv"
    _write_csv(occurrence_path, occurrence_rows)
    _write_csv(batch_path, batch_rows)
    _write_csv(fold_path, fold_rows)
    summary = {
        "mode": "confusion_aware_spectral_train_gradient_readiness",
        "split": "train",
        "test_data_used": False,
        "validation_predictions_used": False,
        "trainer_modified": False,
        "cidt_summary": str(summary_path),
        "cidt_summary_sha256": summary_hash,
        "prediction_csv": str(predictions_path),
        "prediction_csv_sha256": prediction_hash,
        "keeper_sha256": cidt_summary.get("keeper_sha256"),
        "candidate_sha256": cidt_summary.get("candidate_sha256"),
        "dataset_identity_sha256": cidt_summary.get("dataset_identity_sha256"),
        "rows": int(len(rows)),
        "source_groups": int(len(set(source_stems.tolist()))),
        "class_names": class_names,
        "class_counts": list(class_counts),
        "focus_class": int(args.focus_class),
        "protocol": {
            "paper": "Confusion-Aware Spectral Regularizer, CVPR 2026, equations 7-9",
            "paper_url": "https://arxiv.org/abs/2603.16732",
            "car_weighting": "C_ema @ Lambda",
            "bicar_weighting": "Lambda @ C_ema @ Lambda",
            "logit_reconstruction": "log(probability); exact for all difference-only CAR terms",
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "folds": int(args.folds),
            "ema_momentum": float(args.ema_momentum),
            "frequency_smoothing": float(args.frequency_smoothing),
            "margin": float(args.margin),
            "paper_weight": float(args.paper_weight),
        },
        "source_folds": fold_contract,
        "sampler_exposure": sampler_exposure,
        "methods": method_summaries,
        "candidate_comparison": {
            "bicar_vs_car_focus_fp_magnitude_gain": float(fp_gain),
            "bicar_vs_car_focus_fn_magnitude_gain": float(fn_gain),
            "paper_weighted_bicar_gradient_ratio_to_ce": weighted_gradient_ratio,
        },
        "structural_gates": structural_gates,
        "readiness_gates": readiness_gates,
        "scope_gates_passed": scope_gates_passed,
        "all_gates_passed": all_gates_passed,
        "artifacts": {
            "gradient_occurrences": occurrence_path.name,
            "batch_summary": batch_path.name,
            "fold_summary": fold_path.name,
        },
        "guardrail": (
            "Passing authorizes one matched candidate-checkpoint control/BiCAR "
            "40-batch full-validation no-test smoke only. It does not authorize "
            "weight, margin, EMA, sampler, seed, LR, or run-length sweeps."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = audit_readiness(parse_args())
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "candidate_comparison": summary["candidate_comparison"],
                "scope_gates_passed": summary["scope_gates_passed"],
                "all_gates_passed": summary["all_gates_passed"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not summary["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

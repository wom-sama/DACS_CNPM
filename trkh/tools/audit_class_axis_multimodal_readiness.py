from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _direction_auc,
    _load_cache,
    _transition_stats,
)
from trkh.tools.audit_xca_dual_axis_readiness import (
    _git_commit,
    _prepare_output_dir,
)


SEED = 20260715
CLASS_COUNT = 5
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_FIT_ROWS = 7372
EXPECTED_HOLDOUT_ROWS = 1843
EXPECTED_FEATURE_DIM = 256
EXPECTED_SOURCE_GROUPS = 8064
EXPECTED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
EXPECTED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
AXIS_DIMENSIONS_PER_CLASS = 51
GMM_COMPONENTS = 10
GMM_REG_COVAR = 1e-4
GMM_MAX_ITER = 200

LOCKED_TRAIN_CACHE_SHA256 = (
    "157805b449549c9ad87f56f11ece2ed756669bcf11332c860c99feca92314464"
)
LOCKED_CACHE_MANIFEST_SHA256 = (
    "24c617e96f6863b8434edf03f8d611a4ea2da27cb33c30df9550ac2603a46dd2"
)
LOCKED_CAGRAD_SUMMARY_SHA256 = (
    "dab24c6dd7dcf873e523915aadbdd1f8ef5337801278d9ffc55ce8c867a37750"
)
LOCKED_CAGRAD_PREDICTIONS_SHA256 = (
    "92dc296345e34d3f0ac026db2dcb2f982e40af79e1179136dfbb17f1cc2b9d9f"
)
LOCKED_PROTOCOL_SHA256 = (
    "099b797cba478854c978fcf231b04f7aee9a59ee81b7b3ac73b699b1d4e052e4"
)
LOCKED_WACV_PAPER_SHA256 = (
    "665abebce49e9771ad6d0f202b00bfb8af3c882c8f14d78e58c1e20e7f7876bd"
)
LOCKED_CCAR_PAPER_SHA256 = (
    "30847614127449c95ace5d4787bce2dd21c1909c2500c216a722b16635036e8f"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only information gate for class-axis activation and "
            "multimodal-density representation families. Validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/"
            "train_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path(
            "runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/"
            "embedding_cache_manifest.json"
        ),
    )
    parser.add_argument(
        "--cagrad-summary",
        type=Path,
        default=Path("runs/audit_class1_boundary_cagrad_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--cagrad-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_cagrad_readiness_20260715/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CLASS_AXIS_MULTIMODAL_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--wacv-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\goto_wacv2024_orthonormal_multimodal.pdf"
        ),
    )
    parser.add_argument(
        "--ccar-paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\ccar_arxiv_2604.16861.pdf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_class_axis_multimodal_readiness_20260715"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


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


def _ordered_index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in indices:
        digest.update(f"{int(value)}\n".encode("ascii"))
    return digest.hexdigest()


def _tracked_worktree_clean(root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(Path(root).resolve()),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "train_cache": Path(args.train_cache).resolve(),
        "cache_manifest": Path(args.cache_manifest).resolve(),
        "cagrad_summary": Path(args.cagrad_summary).resolve(),
        "cagrad_predictions": Path(args.cagrad_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "wacv_paper": Path(args.wacv_paper).resolve(),
        "ccar_paper": Path(args.ccar_paper).resolve(),
    }


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    paths = _source_paths(args)
    hashes = {
        "train_cache": _verify_sha256(
            paths["train_cache"], LOCKED_TRAIN_CACHE_SHA256, "train embedding cache"
        ),
        "cache_manifest": _verify_sha256(
            paths["cache_manifest"], LOCKED_CACHE_MANIFEST_SHA256, "cache manifest"
        ),
        "cagrad_summary": _verify_sha256(
            paths["cagrad_summary"], LOCKED_CAGRAD_SUMMARY_SHA256, "CAGrad summary"
        ),
        "cagrad_predictions": _verify_sha256(
            paths["cagrad_predictions"],
            LOCKED_CAGRAD_PREDICTIONS_SHA256,
            "CAGrad predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "readiness protocol"
        ),
        "wacv_paper": _verify_sha256(
            paths["wacv_paper"], LOCKED_WACV_PAPER_SHA256, "WACV paper"
        ),
        "ccar_paper": _verify_sha256(
            paths["ccar_paper"], LOCKED_CCAR_PAPER_SHA256, "CCAR paper"
        ),
    }
    return {"paths": {key: str(value) for key, value in paths.items()}, "sha256": hashes}


def _read_comparator_holdout(path: Path) -> Dict[str, np.ndarray]:
    required = {
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "raw_prediction",
        "margin_agem_prediction",
    }
    for name in ("raw", "margin_agem"):
        required.update(f"{name}_prob_{index}" for index in range(CLASS_COUNT))

    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Comparator CSV is missing columns: {missing}")
        for raw in reader:
            if str(raw["condition"]).strip() != "clean":
                continue
            if int(raw["fold"]) != 0:
                raise ValueError("The locked comparator contains a nonzero holdout fold.")
            row: Dict[str, object] = {
                "sample_index": int(raw["sample_index"]),
                "source_stem": str(raw["source_stem"]).strip().casefold(),
                "image_path": str(Path(str(raw["image_path"])).resolve()),
                "target": int(raw["target"]),
            }
            for name in ("raw", "margin_agem"):
                probabilities = np.asarray(
                    [float(raw[f"{name}_prob_{index}"]) for index in range(CLASS_COUNT)],
                    dtype=np.float64,
                )
                prediction = int(raw[f"{name}_prediction"])
                if prediction != int(probabilities.argmax()):
                    raise ValueError(f"{name} prediction does not match argmax.")
                row[f"{name}_probabilities"] = probabilities
            rows.append(row)

    rows.sort(key=lambda value: int(value["sample_index"]))
    indices = np.asarray([int(value["sample_index"]) for value in rows], dtype=np.int64)
    if len(rows) != EXPECTED_HOLDOUT_ROWS or np.unique(indices).size != len(rows):
        raise ValueError("Comparator holdout row count or uniqueness differs from protocol.")
    return {
        "sample_index": indices,
        "source_stems": np.asarray([value["source_stem"] for value in rows], dtype=object),
        "paths": np.asarray([value["image_path"] for value in rows], dtype=object),
        "labels": np.asarray([value["target"] for value in rows], dtype=np.int64),
        "raw": np.stack([value["raw_probabilities"] for value in rows]),
        "margin_agem": np.stack(
            [value["margin_agem_probabilities"] for value in rows]
        ),
    }


def _load_locked_inputs(args: argparse.Namespace) -> tuple[Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    cache = _load_cache(paths["train_cache"], split="train")
    comparator = _read_comparator_holdout(paths["cagrad_predictions"])
    cagrad_summary = json.loads(paths["cagrad_summary"].read_text(encoding="utf-8"))

    sample_index = np.asarray(cache["sample_index"], dtype=np.int64)
    holdout_indices = np.asarray(comparator["sample_index"], dtype=np.int64)
    fit_indices = np.setdiff1d(sample_index, holdout_indices, assume_unique=True)
    holdout_cache_probabilities = np.asarray(cache["probabilities"])[holdout_indices]
    raw_probabilities = np.asarray(comparator["raw"])
    raw_argmax_mismatches = int(
        np.sum(holdout_cache_probabilities.argmax(axis=1) != raw_probabilities.argmax(axis=1))
    )
    maximum_raw_probability_difference = float(
        np.max(np.abs(holdout_cache_probabilities - raw_probabilities))
    )

    cache_paths = np.asarray(
        [str(Path(str(value)).resolve()) for value in cache["paths"]], dtype=object
    )
    fit_sources = set(np.asarray(cache["source_stems"])[fit_indices].tolist())
    holdout_sources = set(np.asarray(cache["source_stems"])[holdout_indices].tolist())
    cohort = cagrad_summary.get("cohort", {})
    checks = {
        "train_rows_exact": len(sample_index) == EXPECTED_TRAIN_ROWS,
        "sample_index_is_canonical": np.array_equal(
            sample_index, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)
        ),
        "feature_dim_exact": tuple(np.asarray(cache["embeddings"]).shape)
        == (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURE_DIM),
        "source_groups_exact": len(set(cache["source_stems"].tolist()))
        == EXPECTED_SOURCE_GROUPS,
        "fit_rows_exact": len(fit_indices) == EXPECTED_FIT_ROWS,
        "holdout_rows_exact": len(holdout_indices) == EXPECTED_HOLDOUT_ROWS,
        "fit_index_sha256_exact": _ordered_index_sha256(fit_indices.tolist())
        == EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_sha256_exact": _ordered_index_sha256(holdout_indices.tolist())
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "source_groups_disjoint": not fit_sources.intersection(holdout_sources),
        "holdout_labels_exact": np.array_equal(
            np.asarray(cache["labels"])[holdout_indices], comparator["labels"]
        ),
        "holdout_paths_exact": np.array_equal(
            cache_paths[holdout_indices], comparator["paths"]
        ),
        "holdout_source_stems_exact": np.array_equal(
            np.asarray(cache["source_stems"])[holdout_indices],
            comparator["source_stems"],
        ),
        "all_cache_paths_are_train": all(
            Path(str(value)).parent.name.casefold() == "train" for value in cache_paths
        ),
        "cache_raw_argmax_mismatches_exact": raw_argmax_mismatches == 1,
        "cache_raw_probability_difference_lte_0p04": (
            maximum_raw_probability_difference <= 0.04
        ),
        "cagrad_cohort_fit_hash_exact": cohort.get("fit_index_sha256")
        == EXPECTED_FIT_INDEX_SHA256,
        "cagrad_cohort_holdout_hash_exact": cohort.get("holdout_index_sha256")
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "cagrad_validation_not_used": cagrad_summary.get("validation_predictions_used")
        is False,
        "cagrad_test_not_used": cagrad_summary.get("test_data_used") is False,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(f"Locked train-only cohort contract differs: {failed}")

    contract = {
        "checks": checks,
        "fit_indices": fit_indices,
        "holdout_indices": holdout_indices,
        "fit_rows": int(len(fit_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "fit_source_groups": int(len(fit_sources)),
        "holdout_source_groups": int(len(holdout_sources)),
        "source_overlap": [],
        "fit_index_sha256": EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_sha256": EXPECTED_HOLDOUT_INDEX_SHA256,
        "cache_raw_argmax_mismatches": raw_argmax_mismatches,
        "maximum_cache_raw_probability_difference": maximum_raw_probability_difference,
    }
    return provenance, cache, comparator, contract


def balanced_class_axis_assignment(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    *,
    class_count: int = CLASS_COUNT,
    dimensions_per_class: int = AXIS_DIMENSIONS_PER_CLASS,
) -> tuple[list[np.ndarray], Dict[str, object]]:
    features = np.asarray(fit_features, dtype=np.float64)
    labels = np.asarray(fit_labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("fit_features/fit_labels must have shapes [N,D]/[N].")
    if int(dimensions_per_class) * int(class_count) > int(features.shape[1]):
        raise ValueError("Balanced class-axis assignment exceeds feature dimension.")
    if set(np.unique(labels).tolist()) != set(range(int(class_count))):
        raise ValueError("Every class must be represented in the fit rows.")

    class_energy = np.stack(
        [np.mean(np.square(features[labels == index]), axis=0) for index in range(class_count)]
    )
    other_energy = np.stack(
        [np.mean(np.square(features[labels != index]), axis=0) for index in range(class_count)]
    )
    selectivity = (class_energy - other_energy) / (other_energy + 1e-8)
    slots = np.repeat(np.arange(class_count), int(dimensions_per_class))
    assigned_rows, assigned_dimensions = linear_sum_assignment(-selectivity[slots])
    assignments = [
        np.sort(assigned_dimensions[slots[assigned_rows] == class_index]).astype(np.int64)
        for class_index in range(class_count)
    ]
    flattened = np.concatenate(assignments)
    telemetry = {
        "class_count": int(class_count),
        "feature_dim": int(features.shape[1]),
        "dimensions_per_class": int(dimensions_per_class),
        "assigned_dimensions": [values.tolist() for values in assignments],
        "assigned_dimension_count": int(len(flattened)),
        "unique_assignment": bool(np.unique(flattened).size == len(flattened)),
        "unassigned_dimensions": sorted(
            set(range(features.shape[1])).difference(flattened.tolist())
        ),
        "mean_assigned_selectivity": [
            float(np.mean(selectivity[index, assignments[index]]))
            for index in range(class_count)
        ],
    }
    return assignments, telemetry


def class_axis_probabilities(
    fit_features: np.ndarray,
    eval_features: np.ndarray,
    assignments: Sequence[np.ndarray],
) -> tuple[np.ndarray, Dict[str, object]]:
    fit = np.asarray(fit_features, dtype=np.float64)
    evaluate = np.asarray(eval_features, dtype=np.float64)
    if fit.ndim != 2 or evaluate.ndim != 2 or fit.shape[1] != evaluate.shape[1]:
        raise ValueError("fit/eval features must be aligned 2D matrices.")
    fit_scores = np.stack(
        [np.mean(np.square(fit[:, dimensions]), axis=1) for dimensions in assignments],
        axis=1,
    )
    eval_scores = np.stack(
        [np.mean(np.square(evaluate[:, dimensions]), axis=1) for dimensions in assignments],
        axis=1,
    )
    score_scale = np.maximum(np.std(fit_scores, axis=0), 1e-6)
    probabilities = softmax(eval_scores / score_scale[None, :], axis=1)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Class-axis diagnostic produced non-finite probabilities.")
    return probabilities, {
        "fit_score_scale": score_scale.tolist(),
        "minimum_probability": float(probabilities.min()),
        "maximum_probability": float(probabilities.max()),
        "probability_sum_max_error": float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        ),
    }


def multimodal_gmm_probabilities(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    class_count: int = CLASS_COUNT,
    components: int = GMM_COMPONENTS,
    reg_covar: float = GMM_REG_COVAR,
    max_iter: int = GMM_MAX_ITER,
    seed: int = SEED,
) -> tuple[np.ndarray, Dict[str, object]]:
    fit = np.asarray(fit_features, dtype=np.float64)
    labels = np.asarray(fit_labels, dtype=np.int64)
    evaluate = np.asarray(eval_features, dtype=np.float64)
    if fit.ndim != 2 or evaluate.ndim != 2 or fit.shape[1] != evaluate.shape[1]:
        raise ValueError("fit/eval features must be aligned 2D matrices.")
    if labels.shape != (len(fit),) or int(components) < 1:
        raise ValueError("GMM labels/components are invalid.")

    log_likelihoods = []
    models = []
    support = np.bincount(labels, minlength=int(class_count))
    for class_index in range(int(class_count)):
        class_features = fit[labels == class_index]
        if len(class_features) < int(components):
            raise ValueError(f"Class {class_index} has fewer rows than GMM components.")
        model = GaussianMixture(
            n_components=int(components),
            covariance_type="diag",
            reg_covar=float(reg_covar),
            max_iter=int(max_iter),
            n_init=1,
            random_state=int(seed),
        ).fit(class_features)
        log_likelihoods.append(model.score_samples(evaluate))
        models.append(
            {
                "class_index": int(class_index),
                "support": int(len(class_features)),
                "converged": bool(model.converged_),
                "iterations": int(model.n_iter_),
                "lower_bound": float(model.lower_bound_),
            }
        )
    priors = support.astype(np.float64) / float(support.sum())
    logits = np.stack(log_likelihoods, axis=1) + np.log(priors)[None, :]
    probabilities = softmax(logits, axis=1)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Multimodal density diagnostic produced non-finite probabilities.")
    return probabilities, {
        "components_per_class": int(components),
        "covariance_type": "diag",
        "reg_covar": float(reg_covar),
        "max_iter": int(max_iter),
        "n_init": 1,
        "seed": int(seed),
        "natural_class_priors": priors.tolist(),
        "models": models,
        "all_converged": all(bool(value["converged"]) for value in models),
        "probability_sum_max_error": float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        ),
    }


def _restricted_focus_false_positives(
    targets: np.ndarray, probabilities: np.ndarray
) -> int:
    predictions = np.asarray(probabilities).argmax(axis=1)
    labels = np.asarray(targets, dtype=np.int64)
    return int(
        np.sum(
            (predictions == FOCUS_CLASS)
            & np.isin(labels, np.asarray(RESTRICTED_NEGATIVE_CLASSES))
        )
    )


def candidate_evidence(
    targets: np.ndarray,
    raw_probabilities: np.ndarray,
    margin_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(targets, dtype=np.int64)
    raw = np.asarray(raw_probabilities, dtype=np.float64)
    margin = np.asarray(margin_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    raw_metrics = _classification_metrics(labels, raw)
    margin_metrics = _classification_metrics(labels, margin)
    candidate_metrics = _classification_metrics(labels, candidate)
    transitions = _transition_stats(
        labels,
        raw,
        candidate,
        focus_class_index=FOCUS_CLASS,
    )
    direction = _direction_auc(
        labels,
        raw,
        candidate,
        focus_class_index=FOCUS_CLASS,
    )
    raw_focus = raw_metrics["per_class"][FOCUS_CLASS]
    margin_focus = margin_metrics["per_class"][FOCUS_CLASS]
    candidate_focus = candidate_metrics["per_class"][FOCUS_CLASS]
    raw_restricted_fp = _restricted_focus_false_positives(labels, raw)
    candidate_restricted_fp = _restricted_focus_false_positives(labels, candidate)
    return {
        "raw": raw_metrics,
        "margin_agem": margin_metrics,
        "candidate": candidate_metrics,
        "delta_vs_raw": {
            "macro_f1": float(candidate_metrics["macro_f1"] - raw_metrics["macro_f1"]),
            "class1_f1": float(candidate_focus["f1"] - raw_focus["f1"]),
            "class1_precision": float(
                candidate_focus["precision"] - raw_focus["precision"]
            ),
            "class1_recall": float(candidate_focus["recall"] - raw_focus["recall"]),
        },
        "class1_f1_delta_vs_margin_agem": float(
            candidate_focus["f1"] - margin_focus["f1"]
        ),
        "transitions_vs_raw": transitions,
        "direction_vs_raw": direction,
        "restricted_focus_false_positives": {
            "raw": raw_restricted_fp,
            "candidate": candidate_restricted_fp,
            "reduction": int(raw_restricted_fp - candidate_restricted_fp),
        },
    }


def assess_candidate_readiness(
    evidence: Mapping[str, object],
    *,
    structural_checks: Mapping[str, bool],
) -> Dict[str, object]:
    delta = evidence["delta_vs_raw"]
    transitions = evidence["transitions_vs_raw"]
    direction = evidence["direction_vs_raw"]
    restricted = evidence["restricted_focus_false_positives"]
    checks = {
        "macro_f1_not_below_raw": float(delta["macro_f1"]) >= 0.0,
        "class1_f1_gain_gte_0p005": float(delta["class1_f1"]) >= 0.005,
        "class1_precision_gain_gte_0p010": float(delta["class1_precision"])
        >= 0.010,
        "class1_recall_delta_gte_minus_0p005": float(delta["class1_recall"])
        >= -0.005,
        "class1_f1_within_0p002_margin_agem": float(
            evidence["class1_f1_delta_vs_margin_agem"]
        )
        >= -0.002,
        "no_focus_true_positive_broken": int(transitions["focus_true_positive_broken"])
        == 0,
        "restricted_focus_fp_reduction_gte_4": int(restricted["reduction"]) >= 4,
        "corrections_gt_harms": int(transitions["corrections"])
        > int(transitions["harms"]),
        "direction_auc_gte_0p60": direction["auc_fn_positive"] is not None
        and float(direction["auc_fn_positive"]) >= 0.60,
    }
    all_checks = {**{key: bool(value) for key, value in structural_checks.items()}, **checks}
    failed = [name for name, passed in all_checks.items() if not bool(passed)]
    return {
        "structural_checks": dict(structural_checks),
        "behavior_checks": checks,
        "failed_checks": failed,
        "shared_trainer_authorized": not failed,
    }


def _prediction_rows(
    cache: Mapping[str, np.ndarray],
    comparator: Mapping[str, np.ndarray],
    class_axis: np.ndarray,
    multimodal_gmm: np.ndarray,
) -> list[Dict[str, object]]:
    indices = np.asarray(comparator["sample_index"], dtype=np.int64)
    variants = {
        "raw": np.asarray(comparator["raw"]),
        "margin_agem": np.asarray(comparator["margin_agem"]),
        "class_axis": np.asarray(class_axis),
        "multimodal_gmm": np.asarray(multimodal_gmm),
    }
    rows = []
    for offset, sample_index in enumerate(indices.tolist()):
        row: Dict[str, object] = {
            "sample_index": int(sample_index),
            "source_stem": str(cache["source_stems"][sample_index]),
            "image_path": str(Path(str(cache["paths"][sample_index])).resolve()),
            "target": int(cache["labels"][sample_index]),
        }
        for name, probabilities in variants.items():
            row[f"{name}_prediction"] = int(probabilities[offset].argmax())
            for class_index in range(CLASS_COUNT):
                row[f"{name}_prob_{class_index}"] = float(
                    probabilities[offset, class_index]
                )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty prediction CSV.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    axis = summary["candidates"]["class_axis"]
    gmm = summary["candidates"]["multimodal_gmm"]
    lines = [
        "# Class-Axis and Multimodal Train-Only Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Shared trainer authorized: `{summary['gate']['shared_trainer_authorized']}`",
        "- Fit/holdout rows: `7372 / 1843`",
        "- Validation/test used: `false / false`",
        "",
        "## Class-axis diagnostic",
        "",
        f"- Macro F1: `{axis['evidence']['candidate']['macro_f1']:.6f}`",
        f"- Class-1 F1: `{axis['evidence']['candidate']['per_class'][1]['f1']:.6f}`",
        f"- Class-1 precision/recall: `{axis['evidence']['candidate']['per_class'][1]['precision']:.6f} / {axis['evidence']['candidate']['per_class'][1]['recall']:.6f}`",
        f"- Restricted FP reduction: `{axis['evidence']['restricted_focus_false_positives']['reduction']}`",
        f"- Failed checks: `{', '.join(axis['gate']['failed_checks']) or 'none'}`",
        "",
        "## Multimodal density diagnostic",
        "",
        f"- Macro F1: `{gmm['evidence']['candidate']['macro_f1']:.6f}`",
        f"- Class-1 F1: `{gmm['evidence']['candidate']['per_class'][1]['f1']:.6f}`",
        f"- Class-1 precision/recall: `{gmm['evidence']['candidate']['per_class'][1]['precision']:.6f} / {gmm['evidence']['candidate']['per_class'][1]['recall']:.6f}`",
        f"- Restricted FP reduction: `{gmm['evidence']['restricted_focus_false_positives']['reduction']}`",
        f"- Direction AUROC: `{gmm['evidence']['direction_vs_raw']['auc_fn_positive']:.6f}`",
        f"- Failed checks: `{', '.join(gmm['gate']['failed_checks']) or 'none'}`",
        "",
        "Failure closes only these frozen-representation head/regularizer screens. No raw data, validation, test, or model binary was used or written.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, artifacts: Sequence[Path]) -> Dict[str, object]:
    files = []
    for artifact in artifacts:
        resolved = Path(artifact).resolve()
        files.append(
            {
                "name": resolved.name,
                "bytes": int(resolved.stat().st_size),
                "sha256": _sha256(resolved),
            }
        )
    payload = {
        "mode": "class_axis_multimodal_train_only_readiness_evidence",
        "files": files,
        "contains_checkpoint": False,
        "contains_model_binary": False,
        "contains_validation_payload": False,
        "contains_test_payload": False,
        "raw_dataset_touched": False,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    provenance, cache, comparator, contract = _load_locked_inputs(args)
    paths = _source_paths(args)
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "provenance": provenance,
            "cohort": {
                key: value
                for key, value in contract.items()
                if key not in {"fit_indices", "holdout_indices"}
            },
            "validation_predictions_used": False,
            "test_data_used": False,
        }

    output_path = Path(args.output_dir).resolve()
    raw_data_root = Path(r"D:\DataAI\AIEx\newdataset").resolve()
    try:
        output_path.relative_to(raw_data_root)
    except ValueError:
        pass
    else:
        raise ValueError("Audit output cannot be written under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)

    fit_indices = np.asarray(contract["fit_indices"], dtype=np.int64)
    holdout_indices = np.asarray(contract["holdout_indices"], dtype=np.int64)
    features = np.asarray(cache["embeddings"], dtype=np.float64)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    scaler = StandardScaler().fit(features[fit_indices])
    fit_features = scaler.transform(features[fit_indices])
    holdout_features = scaler.transform(features[holdout_indices])

    assignments, assignment_telemetry = balanced_class_axis_assignment(
        fit_features,
        labels[fit_indices],
    )
    axis_probabilities, axis_telemetry = class_axis_probabilities(
        fit_features,
        holdout_features,
        assignments,
    )
    gmm_probabilities, gmm_telemetry = multimodal_gmm_probabilities(
        fit_features,
        labels[fit_indices],
        holdout_features,
    )

    common_structural = {
        "locked_sources_verified": True,
        "cohort_contract_exact": True,
        "source_groups_disjoint": True,
        "fit_only_scaler": True,
        "validation_not_used": True,
        "test_not_used": True,
        "no_hyperparameter_sweep": True,
        "tracked_worktree_clean": _tracked_worktree_clean(Path.cwd()),
    }
    axis_structural = {
        **common_structural,
        "balanced_assignment_exact": all(
            len(value) == AXIS_DIMENSIONS_PER_CLASS for value in assignments
        ),
        "assignment_unique": bool(assignment_telemetry["unique_assignment"]),
        "one_dimension_unassigned": len(assignment_telemetry["unassigned_dimensions"])
        == 1,
        "axis_probabilities_normalized": float(
            axis_telemetry["probability_sum_max_error"]
        )
        <= 1e-12,
    }
    gmm_structural = {
        **common_structural,
        "all_gmms_converged": bool(gmm_telemetry["all_converged"]),
        "gmm_probabilities_normalized": float(
            gmm_telemetry["probability_sum_max_error"]
        )
        <= 1e-12,
        "gmm_recipe_exact": bool(
            gmm_telemetry["components_per_class"] == GMM_COMPONENTS
            and math.isclose(gmm_telemetry["reg_covar"], GMM_REG_COVAR)
            and gmm_telemetry["max_iter"] == GMM_MAX_ITER
            and gmm_telemetry["seed"] == SEED
        ),
    }

    axis_evidence = candidate_evidence(
        comparator["labels"],
        comparator["raw"],
        comparator["margin_agem"],
        axis_probabilities,
    )
    gmm_evidence = candidate_evidence(
        comparator["labels"],
        comparator["raw"],
        comparator["margin_agem"],
        gmm_probabilities,
    )
    axis_gate = assess_candidate_readiness(
        axis_evidence, structural_checks=axis_structural
    )
    gmm_gate = assess_candidate_readiness(
        gmm_evidence, structural_checks=gmm_structural
    )
    authorized = bool(
        axis_gate["shared_trainer_authorized"]
        or gmm_gate["shared_trainer_authorized"]
    )

    predictions_path = output_dir / "predictions_holdout.csv"
    _write_csv(
        predictions_path,
        _prediction_rows(
            cache,
            comparator,
            axis_probabilities,
            gmm_probabilities,
        ),
    )
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "artifact_manifest.json"
    summary: Dict[str, object] = {
        "status": "authorized" if authorized else "rejected",
        "protocol": {
            "seed": SEED,
            "class_count": CLASS_COUNT,
            "focus_class": FOCUS_CLASS,
            "restricted_negative_classes": list(RESTRICTED_NEGATIVE_CLASSES),
            "selection_scope": "locked_train_only_fold0_no_sweep",
            "family_scope": (
                "necessary frozen-representation information screen; not an exact "
                "end-to-end reproduction"
            ),
        },
        "provenance": {
            **provenance,
            "git_commit": _git_commit(Path.cwd()),
        },
        "cohort": {
            key: value
            for key, value in contract.items()
            if key not in {"fit_indices", "holdout_indices"}
        },
        "comparators": {
            "raw": _classification_metrics(comparator["labels"], comparator["raw"]),
            "margin_agem": _classification_metrics(
                comparator["labels"], comparator["margin_agem"]
            ),
        },
        "candidates": {
            "class_axis": {
                "recipe": assignment_telemetry,
                "probability_telemetry": axis_telemetry,
                "evidence": axis_evidence,
                "gate": axis_gate,
            },
            "multimodal_gmm": {
                "recipe": gmm_telemetry,
                "evidence": gmm_evidence,
                "gate": gmm_gate,
            },
        },
        "gate": {
            "shared_trainer_authorized": authorized,
            "authorized_candidates": [
                name
                for name, gate in (
                    ("class_axis", axis_gate),
                    ("multimodal_gmm", gmm_gate),
                )
                if gate["shared_trainer_authorized"]
            ],
        },
        "validation_predictions_used": False,
        "test_data_used": False,
        "binary_model_artifacts_written": False,
        "raw_dataset_touched": False,
        "artifacts": {
            "summary": str(summary_path),
            "report": str(report_path),
            "predictions": str(predictions_path),
            "manifest": str(manifest_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    _write_manifest(manifest_path, [summary_path, report_path, predictions_path])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    if bool(args.preflight_only):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return
    print(json.dumps(result["gate"], indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

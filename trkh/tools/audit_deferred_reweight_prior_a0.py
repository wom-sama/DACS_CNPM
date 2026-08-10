from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from trkh.data.dataset import StrictBalancedBatchSampler


METHOD = "deferred_reweight_prior_a0"
MODE = "deferred_reweight_prior_a0_no_training_authorization"
SEED = 20260720
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
EXPECTED_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_SOURCE_COUNT = 8064
EXPECTED_BATCH_SIZE = 32
EXPECTED_STRICT_EXPOSURE = (1844, 1843, 1843, 1843, 1843)
DEFERRED_BETA = 0.999

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = REPO_ROOT / "runs" / "full_v8_yolof_randominit_30e_20260714_105524"
CIDT_ROOT = REPO_ROOT / "runs" / "audit_cidt_readiness_full_train_20260714"
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\LDAM-DRW")
PAPER_PATH = Path(r"D:\DataAI\external_sources\papers\ldam_drw_neurips2019.pdf")

LOCKED_HASHES: Mapping[str, Tuple[Path, str]] = {
    "data_yaml": (
        Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
        "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    ),
    "full_checkpoint": (
        FULL_ROOT / "checkpoints" / "best.pt",
        "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549",
    ),
    "full_resolved_config": (
        FULL_ROOT / "resolved_config.json",
        "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97",
    ),
    "full_launcher_args": (
        FULL_ROOT / "launcher_args.json",
        "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6",
    ),
    "full_history": (
        FULL_ROOT / "history.csv",
        "a6eeddd7b1ef29222e78764bad854c9d670b0bc98de38362c01e2373359404e2",
    ),
    "cidt_summary": (
        CIDT_ROOT / "summary.json",
        "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    ),
    "cidt_predictions": (
        CIDT_ROOT / "predictions_all_conditions.csv",
        "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    ),
    "keeper_checkpoint": (
        KEEPER_ROOT / "checkpoints" / "best.pt",
        "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    ),
    "protocol": (
        REPO_ROOT / "docs" / "TRKH_5CLASS_DEFERRED_REWEIGHT_A0_PROTOCOL_20260720.md",
        "1627a2b541dbbd27394ee2cd95e1673a7b73d978aec605d1e5939e2164c76276",
    ),
    "current_best_commands": (
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    ),
    "command_history": (
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
        "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    ),
    "paper": (
        PAPER_PATH,
        "f873d07d4bc6fedf6de7222a981331c88a961509c5004287346d444e96e1b1f3",
    ),
    "official_source": (
        OFFICIAL_ROOT / "cifar_train.py",
        "f2f2e10c7dbc837c6cd640b9295e97eb945064d965d2f6c19c1301f22d1f6f43",
    ),
    "official_license": (
        OFFICIAL_ROOT / "LICENSE",
        "24769cbd6cf77906da1b7c51608079de664b44c4025cf5a2ef99d1f7e2b60bc5",
    ),
}
LOCKED_OFFICIAL_COMMIT = "2536330f2afdaa65618323cb5a5850efccce762a"
LOCKED_OFFICIAL_TREE = "c076696ca0ae868e0c41038d72159a439652a146"

PROTECTED_UNTRACKED_FILES: Tuple[Tuple[Path, str], ...] = (
    (
        REPO_ROOT / "BaoCao" / "GT.md",
        "c784717657d1dda168de2e70f08256d41397c5d7c0b8a916f4b0367870c7b840",
    ),
    (
        REPO_ROOT / "BaoCao" / "mango_cls_256_merge01_4cls_architecture.dot",
        "c9aebcb97275a4dbb069e866df71d0022d9c8b6e9f9caa4e55d0f991ce528a9d",
    ),
    (
        REPO_ROOT / "BaoCao" / "mango_cls_256_merge01_4cls_architecture.png",
        "93d6f9c27b0b77bdf5113cede18d828f9a50949e5f324bba6b990f3cf204832d",
    ),
    (
        REPO_ROOT / "BaoCao" / "mango_cls_256_merge01_4cls_architecture.svg",
        "2ead05f8e42f41d256bc5b7df21df4f37b181dd2a54d89cdc8098ea074453d05",
    ),
    (
        REPO_ROOT / "BaoCao" / "mango_cls_256_merge01_4cls_architecture_summary.md",
        "d43b82beb7910f268793f38a689b74625df7764a92b85402d5fa13f8fcf0e493",
    ),
    (
        REPO_ROOT / "deep-research-report (9).md",
        "7e52cc8e2bed2e48dafc68b25dc2536318a895c866f4aa280c4fe2e817546f8a",
    ),
    (
        REPO_ROOT / "deep-research-report (10).md",
        "37f74b5e97b6160e657237844d66cedc13a98808b476b8328a58a52eb7c1522a",
    ),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked no-training prior/exposure audit for natural-first deferred reweighting."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_deferred_reweight_prior_a0_20260720",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git_value(root: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_provenance() -> Dict[str, object]:
    observed_hashes: Dict[str, str] = {}
    for label, (path, expected) in LOCKED_HASHES.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing locked {label}: {resolved}")
        observed = _sha256(resolved)
        if observed != expected:
            raise ValueError(
                f"Locked hash mismatch for {label}: observed={observed}, expected={expected}"
            )
        observed_hashes[label] = observed

    protected_hashes: Dict[str, str] = {}
    for path, expected in PROTECTED_UNTRACKED_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected untracked file: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"Protected untracked hash mismatch: {path}")
        protected_hashes[str(path.relative_to(REPO_ROOT)).replace("\\", "/")] = observed

    commit = _git_value(OFFICIAL_ROOT, "HEAD")
    tree = _git_value(OFFICIAL_ROOT, "HEAD^{tree}")
    if commit != LOCKED_OFFICIAL_COMMIT or tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(
            "Official LDAM-DRW revision mismatch: "
            f"commit={commit}, tree={tree}"
        )
    return {
        "locked_hashes": observed_hashes,
        "official_commit": commit,
        "official_tree": tree,
        "official_license": "MIT",
        "protected_untracked_hashes": protected_hashes,
    }


def _validate_locked_metadata() -> Dict[str, object]:
    resolved = _read_json(LOCKED_HASHES["full_resolved_config"][0])
    train = dict(resolved.get("train_config", {}))
    expected_train = {
        "batch_size": 32,
        "epochs": 30,
        "scheduler_total_epochs": 30,
        "seed": 42,
        "num_workers": 4,
        "eval_num_workers": 2,
        "balanced_epoch_sampling": True,
        "balanced_epoch_multiplier": 1.0,
        "use_class_weights": False,
        "classification_loss": "ldam_focal",
    }
    observed_train = {key: train.get(key) for key in expected_train}
    if observed_train != expected_train:
        raise ValueError(
            f"Locked scratch train config mismatch: {observed_train} != {expected_train}"
        )

    cidt = _read_json(LOCKED_HASHES["cidt_summary"][0])
    scope = {
        "split": cidt.get("split"),
        "rows": cidt.get("rows"),
        "validation_predictions_used": cidt.get("validation_predictions_used"),
        "test_data_used": cidt.get("test_data_used"),
        "audit_scope": cidt.get("audit_scope"),
    }
    expected_scope = {
        "split": "train",
        "rows": EXPECTED_ROWS,
        "validation_predictions_used": False,
        "test_data_used": False,
        "audit_scope": "full",
    }
    if scope != expected_scope:
        raise ValueError(f"Locked CIDT scope mismatch: {scope} != {expected_scope}")
    if cidt.get("keeper_sha256") != LOCKED_HASHES["keeper_checkpoint"][1]:
        raise ValueError("CIDT keeper hash mismatch")
    if cidt.get("candidate_sha256") != LOCKED_HASHES["full_checkpoint"][1]:
        raise ValueError("CIDT scratch candidate hash mismatch")
    return {"scratch_train_config": observed_train, "cidt_scope": scope}


def _read_clean_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "fold",
            "target_index",
            "keeper_prediction",
            "candidate_prediction",
            *(f"keeper_prob_{index}" for index in range(5)),
            *(f"candidate_prob_{index}" for index in range(5)),
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CIDT prediction CSV missing columns: {missing}")
        for raw in reader:
            if raw["condition"] != "clean":
                continue
            rows.append(
                {
                    "sample_index": int(raw["sample_index"]),
                    "source_stem": raw["source_stem"],
                    "fold": int(raw["fold"]),
                    "target": int(raw["target_index"]),
                    "keeper_prediction": int(raw["keeper_prediction"]),
                    "candidate_prediction": int(raw["candidate_prediction"]),
                    "keeper_probabilities": [
                        float(raw[f"keeper_prob_{index}"]) for index in range(5)
                    ],
                    "candidate_probabilities": [
                        float(raw[f"candidate_prob_{index}"]) for index in range(5)
                    ],
                }
            )
    rows.sort(key=lambda row: int(row["sample_index"]))
    return rows


def _validate_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} clean rows, got {len(rows)}")
    indices = [int(row["sample_index"]) for row in rows]
    if indices != list(range(EXPECTED_ROWS)):
        raise ValueError("Clean sample indices must be unique contiguous 0..9214")
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    counts = np.bincount(labels, minlength=5)
    if tuple(int(value) for value in counts) != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Class count mismatch: {counts.tolist()}")

    source_folds: Dict[str, int] = {}
    fold_counts = {fold: 0 for fold in EXPECTED_FOLDS}
    for row in rows:
        fold = int(row["fold"])
        if fold not in fold_counts:
            raise ValueError(f"Unexpected fold: {fold}")
        fold_counts[fold] += 1
        source = str(row["source_stem"])
        prior_fold = source_folds.setdefault(source, fold)
        if prior_fold != fold:
            raise ValueError(f"Source crosses folds: {source}: {prior_fold}/{fold}")
    observed_fold_counts = tuple(fold_counts[fold] for fold in EXPECTED_FOLDS)
    if observed_fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Fold count mismatch: {observed_fold_counts}")
    if len(source_folds) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"Source count mismatch: {len(source_folds)}")

    probability_checks: Dict[str, object] = {}
    for role in ("keeper", "candidate"):
        probabilities = np.asarray(
            [row[f"{role}_probabilities"] for row in rows], dtype=np.float64
        )
        persisted = np.asarray(
            [int(row[f"{role}_prediction"]) for row in rows], dtype=np.int64
        )
        derived = probabilities.argmax(axis=1)
        max_sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
        argmax_mismatches = int(np.count_nonzero(derived != persisted))
        if max_sum_error > 2e-6 or argmax_mismatches:
            raise ValueError(
                f"Invalid {role} probabilities: sum_error={max_sum_error}, "
                f"argmax_mismatches={argmax_mismatches}"
            )
        probability_checks[role] = {
            "max_probability_sum_error": max_sum_error,
            "argmax_mismatches": argmax_mismatches,
        }
    return {
        "rows": len(rows),
        "unique_sample_indices": len(set(indices)),
        "unique_sources": len(source_folds),
        "class_counts": counts.tolist(),
        "fold_counts": list(observed_fold_counts),
        "source_cross_fold_count": 0,
        "probability_checks": probability_checks,
    }


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    restricted_fp = int(
        sum(confusion[class_index, FOCUS_CLASS] for class_index in RESTRICTED_NEGATIVE_CLASSES)
    )
    return {
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "predicted_support": predicted_support.tolist(),
        "confusion_matrix": confusion.tolist(),
        "focus_precision": float(precision[FOCUS_CLASS]),
        "focus_recall": float(recall[FOCUS_CLASS]),
        "focus_f1": float(f1[FOCUS_CLASS]),
        "focus_predicted_support": int(predicted_support[FOCUS_CLASS]),
        "focus_support_ratio": float(
            predicted_support[FOCUS_CLASS] / max(1, support[FOCUS_CLASS])
        ),
        "focus_recall_precision_gap": float(
            recall[FOCUS_CLASS] - precision[FOCUS_CLASS]
        ),
        "restricted_focus_false_positives": restricted_fp,
    }


def _role_metrics(
    rows: Sequence[Mapping[str, object]],
    role: str,
) -> Dict[str, object]:
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    predictions = np.asarray(
        [int(row[f"{role}_prediction"]) for row in rows], dtype=np.int64
    )
    folds: List[Dict[str, object]] = []
    for fold in EXPECTED_FOLDS:
        selected = np.asarray([int(row["fold"]) == fold for row in rows], dtype=bool)
        metrics = _classification_metrics(labels[selected], predictions[selected])
        folds.append({"fold": fold, "rows": int(selected.sum()), **metrics})
    return {
        "aggregate": _classification_metrics(labels, predictions),
        "folds": folds,
    }


def _apply_prior_factor(probabilities: np.ndarray, factor: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] != factor.size:
        raise ValueError("Probability/factor shape mismatch")
    if np.any(factor < 0.0) or not np.all(np.isfinite(factor)):
        raise ValueError("Prior factor must be finite and nonnegative")
    return np.argmax(probabilities * factor[None, :], axis=1)


def _prior_controls(
    rows: Sequence[Mapping[str, object]],
    natural_prior: np.ndarray,
) -> List[Dict[str, object]]:
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    controls: List[Dict[str, object]] = []
    factors = {
        "sqrt_prior": np.sqrt(natural_prior / 0.2),
        "full_prior": natural_prior / 0.2,
    }
    for role in ("keeper", "candidate"):
        probabilities = np.asarray(
            [row[f"{role}_probabilities"] for row in rows], dtype=np.float64
        )
        for control_name, factor in factors.items():
            predictions = _apply_prior_factor(probabilities, factor)
            controls.append(
                {
                    "role": role,
                    "control": control_name,
                    "factor": factor.tolist(),
                    **_classification_metrics(labels, predictions),
                }
            )
    return controls


def _effective_number_weights(
    class_counts: Sequence[int],
    beta: float = DEFERRED_BETA,
) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float64)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return weights / weights.mean()


def _build_gates(
    exposure: Mapping[str, object],
    role_metrics: Mapping[str, Mapping[str, object]],
    controls: Sequence[Mapping[str, object]],
    row_validation: Mapping[str, object],
) -> List[Dict[str, object]]:
    candidate = role_metrics["candidate"]
    candidate_aggregate = dict(candidate["aggregate"])
    candidate_folds = list(candidate["folds"])
    prior_control_recalls = [
        float(row["focus_recall"])
        for row in controls
        if row["role"] == "candidate"
    ]
    return [
        {
            "name": "locked_train_scope_integrity",
            "passed": bool(
                int(row_validation["rows"]) == EXPECTED_ROWS
                and int(row_validation["unique_sample_indices"]) == EXPECTED_ROWS
                and int(row_validation["source_cross_fold_count"]) == 0
            ),
            "observed": {
                "rows": row_validation["rows"],
                "unique_sample_indices": row_validation["unique_sample_indices"],
                "source_cross_fold_count": row_validation["source_cross_fold_count"],
            },
            "requirement": "9215 unique clean train rows and zero cross-fold sources",
        },
        {
            "name": "strict_exposure_exact",
            "passed": tuple(exposure["strict_counts"]) == EXPECTED_STRICT_EXPOSURE,
            "observed": exposure["strict_counts"],
            "requirement": list(EXPECTED_STRICT_EXPOSURE),
        },
        {
            "name": "focus_exposure_mismatch",
            "passed": bool(
                float(exposure["focus_multiplier"]) >= 3.0
                and float(exposure["max_nonfocus_multiplier"]) <= 1.10
            ),
            "observed": {
                "focus_multiplier": exposure["focus_multiplier"],
                "max_nonfocus_multiplier": exposure["max_nonfocus_multiplier"],
            },
            "requirement": "focus >= 3.0 and max nonfocus <= 1.10",
        },
        {
            "name": "candidate_support_bias_fold_stable",
            "passed": all(float(row["focus_support_ratio"]) >= 1.25 for row in candidate_folds),
            "observed": [float(row["focus_support_ratio"]) for row in candidate_folds],
            "requirement": ">= 1.25 in all five fixed source partitions",
        },
        {
            "name": "candidate_precision_recall_bias_fold_stable",
            "passed": bool(
                all(
                    float(row["focus_recall_precision_gap"]) >= 0.20
                    for row in candidate_folds
                )
                and int(candidate_aggregate["restricted_focus_false_positives"]) >= 250
            ),
            "observed": {
                "fold_recall_precision_gaps": [
                    float(row["focus_recall_precision_gap"]) for row in candidate_folds
                ],
                "restricted_false_positives": candidate_aggregate[
                    "restricted_focus_false_positives"
                ],
            },
            "requirement": "gap >= 0.20 in all folds and restricted FP >= 250",
        },
        {
            "name": "posthoc_prior_control_is_unsafe",
            "passed": bool(prior_control_recalls and min(prior_control_recalls) < 0.70),
            "observed": prior_control_recalls,
            "requirement": "at least one fixed candidate control has class-1 recall < 0.70",
        },
        {
            "name": "scope_flags",
            "passed": True,
            "observed": {
                "validation_predictions_used": False,
                "test_data_used": False,
                "dataset_pixels_loaded": False,
                "checkpoint_loaded": False,
                "model_forward_used": False,
                "readout_fit": False,
                "training_used": False,
            },
            "requirement": "all forbidden operations false",
        },
    ]


def _analyze() -> Dict[str, object]:
    provenance = _verify_provenance()
    metadata = _validate_locked_metadata()
    rows = _read_clean_rows(LOCKED_HASHES["cidt_predictions"][0])
    row_validation = _validate_rows(rows)

    labels = [int(row["target"]) for row in rows]
    sampler = StrictBalancedBatchSampler(
        labels=labels,
        batch_size=EXPECTED_BATCH_SIZE,
        num_classes=5,
        epoch_multiplier=1.0,
        seed=42,
    )
    strict_counts = np.asarray(sampler.exposure_counts(), dtype=np.float64)
    natural_counts = np.asarray(EXPECTED_CLASS_COUNTS, dtype=np.float64)
    multipliers = strict_counts / natural_counts
    exposure = {
        "natural_counts": natural_counts.astype(np.int64).tolist(),
        "natural_prior": (natural_counts / natural_counts.sum()).tolist(),
        "strict_counts": strict_counts.astype(np.int64).tolist(),
        "strict_prior": (strict_counts / strict_counts.sum()).tolist(),
        "strict_total": int(strict_counts.sum()),
        "strict_relative_gap": float(sampler.exposure_summary()["relative_gap"]),
        "exposure_multipliers": multipliers.tolist(),
        "focus_multiplier": float(multipliers[FOCUS_CLASS]),
        "max_nonfocus_multiplier": float(
            max(multipliers[index] for index in range(5) if index != FOCUS_CLASS)
        ),
    }

    role_metrics = {
        role: _role_metrics(rows, role) for role in ("keeper", "candidate")
    }
    natural_prior = natural_counts / natural_counts.sum()
    controls = _prior_controls(rows, natural_prior)
    deferred_weights = _effective_number_weights(EXPECTED_CLASS_COUNTS)
    weighted_mass = natural_counts * deferred_weights
    deferred_projection = {
        "beta": DEFERRED_BETA,
        "weights": deferred_weights.tolist(),
        "weighted_mass_share": (weighted_mass / weighted_mass.sum()).tolist(),
        "focus_weight": float(deferred_weights[FOCUS_CLASS]),
        "focus_weighted_mass_share": float(
            weighted_mass[FOCUS_CLASS] / weighted_mass.sum()
        ),
        "focus_weighted_mass_over_natural": float(
            (weighted_mass[FOCUS_CLASS] / weighted_mass.sum())
            / natural_prior[FOCUS_CLASS]
        ),
        "strict_focus_mass_share": float(strict_counts[FOCUS_CLASS] / strict_counts.sum()),
        "natural_focus_mass_share": float(natural_prior[FOCUS_CLASS]),
    }
    gates = _build_gates(exposure, role_metrics, controls, row_validation)
    all_gates_passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "schema_version": 1,
        "method": METHOD,
        "mode": MODE,
        "seed": SEED,
        "known_observation_disclosure": (
            "A0 deterministically replays evidence observed before protocol lock; "
            "only Stage-B and Stage-C thresholds are prospective."
        ),
        "authorization": (
            "one_natural_only_stage_b_smoke" if all_gates_passed else "route_closed"
        ),
        "provenance": provenance,
        "metadata": metadata,
        "row_validation": row_validation,
        "exposure": exposure,
        "role_metrics": role_metrics,
        "prior_controls": controls,
        "deferred_projection": deferred_projection,
        "gates": gates,
        "all_gates_passed": all_gates_passed,
        "scope": {
            "split": "train",
            "validation_predictions_used": False,
            "test_data_used": False,
            "dataset_pixels_loaded": False,
            "checkpoint_loaded": False,
            "model_forward_used": False,
            "readout_fit": False,
            "training_used": False,
            "model_or_trainer_modified": False,
            "current_best_commands_modified": False,
        },
    }


def _write_fold_metrics(path: Path, role_metrics: Mapping[str, Mapping[str, object]]) -> None:
    fieldnames = [
        "role",
        "fold",
        "rows",
        "focus_precision",
        "focus_recall",
        "focus_f1",
        "focus_support",
        "focus_predicted_support",
        "focus_support_ratio",
        "focus_recall_precision_gap",
        "restricted_focus_false_positives",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for role in ("keeper", "candidate"):
            for row in role_metrics[role]["folds"]:
                writer.writerow(
                    {
                        "role": role,
                        "fold": row["fold"],
                        "rows": row["rows"],
                        "focus_precision": row["focus_precision"],
                        "focus_recall": row["focus_recall"],
                        "focus_f1": row["focus_f1"],
                        "focus_support": row["support"][FOCUS_CLASS],
                        "focus_predicted_support": row["focus_predicted_support"],
                        "focus_support_ratio": row["focus_support_ratio"],
                        "focus_recall_precision_gap": row["focus_recall_precision_gap"],
                        "restricted_focus_false_positives": row[
                            "restricted_focus_false_positives"
                        ],
                    }
                )


def _write_prior_controls(path: Path, controls: Iterable[Mapping[str, object]]) -> None:
    fieldnames = [
        "role",
        "control",
        "focus_precision",
        "focus_recall",
        "focus_f1",
        "focus_predicted_support",
        "focus_support_ratio",
        "restricted_focus_false_positives",
        "macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in controls:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_formal_outputs(output_dir: Path, summary: Mapping[str, object]) -> Dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Formal output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    fold_path = output_dir / "fold_metrics.csv"
    control_path = output_dir / "prior_controls.csv"
    summary_path = output_dir / "summary.json"
    _write_fold_metrics(fold_path, summary["role_metrics"])
    _write_prior_controls(control_path, summary["prior_controls"])
    _write_json(summary_path, summary)

    files = []
    for path in (fold_path, control_path, summary_path):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "method": METHOD,
        "files": files,
        "total_bytes": int(sum(int(row["bytes"]) for row in files)),
    }
    _write_json(output_dir / "audit_manifest.json", manifest)
    return manifest


def _run_replay(args: argparse.Namespace) -> Dict[str, object]:
    replay_path = Path(args.replay_summary).resolve()
    expected_hash = str(args.expected_summary_sha256 or "").strip().lower()
    if expected_hash and _sha256(replay_path) != expected_hash:
        raise ValueError("Replay summary SHA256 does not match --expected-summary-sha256")
    persisted = _read_json(replay_path)
    recomputed = _analyze()
    exact = persisted == recomputed
    replay = {
        "mode": f"{MODE}_independent_replay",
        "summary_path": str(replay_path),
        "summary_sha256": _sha256(replay_path),
        "persisted_canonical_sha256": _canonical_sha256(persisted),
        "recomputed_canonical_sha256": _canonical_sha256(recomputed),
        "exact_summary_match": exact,
        "all_gates_passed": bool(recomputed["all_gates_passed"]),
    }
    if not exact:
        raise ValueError("Independent replay did not exactly match the persisted summary")
    if args.replay_output is not None:
        output_path = Path(args.replay_output).resolve()
        if output_path.exists():
            raise FileExistsError(f"Replay output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, replay)
        manifest_path = output_path.parent / "audit_manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            files = list(manifest.get("files", []))
            if any(row.get("path") == output_path.name for row in files):
                raise ValueError("Replay artifact is already present in the manifest")
            files.append(
                {
                    "path": output_path.name,
                    "bytes": output_path.stat().st_size,
                    "sha256": _sha256(output_path),
                }
            )
            manifest["files"] = files
            manifest["total_bytes"] = int(sum(int(row["bytes"]) for row in files))
            _write_json(manifest_path, manifest)
    return replay


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.replay_summary is not None:
        print(json.dumps(_run_replay(args), sort_keys=True))
        return 0

    summary = _analyze()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "mode": MODE,
                    "preflight_only": True,
                    "all_gates_passed": summary["all_gates_passed"],
                    "authorization": summary["authorization"],
                    "output_created": False,
                },
                sort_keys=True,
            )
        )
        return 0

    output_dir = Path(args.output_dir).resolve()
    manifest = _write_formal_outputs(output_dir, summary)
    print(
        json.dumps(
            {
                "mode": MODE,
                "output_dir": str(output_dir),
                "summary_sha256": _sha256(output_dir / "summary.json"),
                "manifest_sha256": _sha256(output_dir / "audit_manifest.json"),
                "artifact_bytes": manifest["total_bytes"],
                "all_gates_passed": summary["all_gates_passed"],
                "authorization": summary["authorization"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import (
    IdentityCorruption,
    _forward_classification_with_metadata,
)
from trkh.inference.inference import load_model
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _assign_source_folds,
    _build_eval_transform,
    _classification_metrics,
    _dataset_identity,
    _split_source_stems,
    directional_event_masks,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
ROTATIONS = (0, 1, 2, 3)
PROTOCOL_SHA256 = "e4852c34143758e4a2c5b8c5c748563f38004a839d4bc0f3cab8d410af289bb3"
E2CNN_COMMIT = "022d6ca4a78ab666f3c149d1849e324ed1d40238"
E2CNN_TREE = "7673db4f3950f6aac8bbe27534af5a5d165df6ff"

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_FILES: Tuple[Tuple[str, Path, str], ...] = (
    (
        "protocol",
        REPO_ROOT / "docs" / "TRKH_5CLASS_C4_ROTATION_CONSENSUS_A0_PROTOCOL_20260720.md",
        PROTOCOL_SHA256,
    ),
    (
        "keeper",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "checkpoints"
        / "best.pt",
        "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    ),
    (
        "keeper_launcher_args",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "launcher_args.json",
        "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    ),
    (
        "keeper_resolved_config",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "resolved_config.json",
        "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    ),
    (
        "data_yaml",
        Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
        "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    ),
    (
        "cidt_predictions",
        REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
        "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    ),
    (
        "cidt_summary",
        REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "summary.json",
        "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    ),
    (
        "current_best_commands",
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    ),
    (
        "current_best_history",
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
        "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    ),
    (
        "paper",
        Path(r"D:\DataAI\external_sources\papers\group_equivariant_cnn_icml2016.pdf"),
        "e9b4c64d46f1c99569f6835bc11fc01e4635551eb236f0b9c0b7a466ed0a0b93",
    ),
    (
        "e2cnn_license",
        Path(r"D:\DataAI\external_sources\official\e2cnn_full\LICENSE"),
        "9d3ddfa2ee02769845dffb422007b836a28eafb9de37e8af3f79a6cd3f28e02c",
    ),
    (
        "e2cnn_r2convolution",
        Path(
            r"D:\DataAI\external_sources\official\e2cnn_full\e2cnn\nn\modules\r2_conv\r2convolution.py"
        ),
        "0e72b17230dee09dbde1810395aaf39409c9dd19035ccc729074e02ba1ae9e9a",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only C4 rotation-consensus information gate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_c4_rotation_consensus_a0_20260720",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Locked {label} is missing: {resolved}")
    observed = _sha256(resolved)
    if observed != str(expected).strip().lower():
        raise ValueError(
            f"Locked {label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return observed


def _git_value(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_locked_inputs() -> Dict[str, object]:
    files: Dict[str, object] = {}
    for label, path, expected in LOCKED_FILES:
        resolved = path.expanduser().resolve()
        files[label] = {
            "path": str(resolved),
            "sha256": _verify_hash(resolved, expected, label),
        }

    e2cnn_repo = Path(r"D:\DataAI\external_sources\official\e2cnn_full").resolve()
    e2cnn_commit = _git_value(e2cnn_repo, "rev-parse", "HEAD")
    e2cnn_tree = _git_value(e2cnn_repo, "rev-parse", "HEAD^{tree}")
    e2cnn_status = _git_value(e2cnn_repo, "status", "--porcelain")
    if e2cnn_commit != E2CNN_COMMIT or e2cnn_tree != E2CNN_TREE or e2cnn_status:
        raise ValueError(
            "Official e2cnn lock mismatch: "
            f"commit={e2cnn_commit}, tree={e2cnn_tree}, dirty={bool(e2cnn_status)}"
        )
    return {
        "files": files,
        "e2cnn": {
            "repo": str(e2cnn_repo),
            "commit": e2cnn_commit,
            "tree": e2cnn_tree,
            "tracked_worktree_clean": True,
        },
    }


def _repo_state() -> Dict[str, object]:
    head = _git_value(REPO_ROOT, "rev-parse", "HEAD")
    upstream = _git_value(REPO_ROOT, "rev-parse", "@{upstream}")
    tracked_status = _git_value(
        REPO_ROOT, "status", "--porcelain", "--untracked-files=no"
    )
    return {
        "head": head,
        "upstream": upstream,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": not bool(tracked_status),
        "tracked_status": tracked_status,
    }


def _require_empty_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def rotate_appearance_batch(
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    k: int,
) -> Tuple[Tensor, Dict[str, object]]:
    normalized_k = int(k) % 4
    rotated_images = torch.rot90(images, k=normalized_k, dims=(-2, -1))
    rotated_metadata: Dict[str, object] = dict(metadata)
    image_mask = metadata.get("image_mask")
    if torch.is_tensor(image_mask):
        rotated_metadata["image_mask"] = torch.rot90(
            image_mask, k=normalized_k, dims=(-2, -1)
        )
    # bbox is intentionally a source-image coordinate side channel.
    return rotated_images, rotated_metadata


def c4_statistics(angle_probabilities: np.ndarray) -> Dict[str, np.ndarray]:
    probabilities = np.asarray(angle_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1] != len(ROTATIONS):
        raise ValueError("angle probabilities must have shape [N,4,C]")
    if not np.isfinite(probabilities).all() or bool((probabilities < 0.0).any()):
        raise ValueError("angle probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("angle probabilities are not normalized")

    consensus = probabilities.mean(axis=1)
    epsilon = 1e-12
    js_per_angle = np.sum(
        probabilities
        * (
            np.log(np.clip(probabilities, epsilon, 1.0))
            - np.log(np.clip(consensus[:, None, :], epsilon, 1.0))
        ),
        axis=2,
    )
    angle_predictions = probabilities.argmax(axis=2)
    agreement_counts = np.asarray(
        [np.bincount(row, minlength=probabilities.shape[2]).max() for row in angle_predictions],
        dtype=np.int64,
    )
    focus_values = probabilities[:, :, FOCUS_CLASS]
    return {
        "consensus_probabilities": consensus,
        "angle_predictions": angle_predictions,
        "top1_agreement_count": agreement_counts,
        "top1_all_agree": agreement_counts == len(ROTATIONS),
        "mean_js_divergence": js_per_angle.mean(axis=1),
        "focus_probability_range": focus_values.max(axis=1) - focus_values.min(axis=1),
        "focus_probability_std": focus_values.std(axis=1),
    }


def _calibration_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("targets and probabilities do not align")
    epsilon = 1e-12
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == targets).astype(np.float64)
    ece = 0.0
    for index in range(10):
        low = index / 10.0
        high = (index + 1) / 10.0
        selected = (confidence >= low) & (
            confidence <= high if index == 9 else confidence < high
        )
        if bool(selected.any()):
            ece += float(selected.mean()) * abs(
                float(correctness[selected].mean()) - float(confidence[selected].mean())
            )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return {
        "nll": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(targets.size), targets], epsilon, 1.0
                )
            ).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_10": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predictions = np.asarray(probabilities).argmax(axis=1)
    result = _classification_metrics(
        targets, predictions, num_classes=int(probabilities.shape[1])
    )
    result["calibration"] = _calibration_metrics(targets, probabilities)
    return result


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, scores))


def _numeric_summary(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _directional_summary(
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> Dict[str, object]:
    events = directional_event_masks(
        targets, baseline_predictions, candidate_predictions
    )
    restricted = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
    restricted_remove = restricted & (baseline_predictions == FOCUS_CLASS) & (
        candidate_predictions != FOCUS_CLASS
    )
    restricted_create = restricted & (baseline_predictions != FOCUS_CLASS) & (
        candidate_predictions == FOCUS_CLASS
    )
    transition = np.zeros((5, 5), dtype=np.int64)
    np.add.at(transition, (baseline_predictions, candidate_predictions), 1)
    return {
        "focus_fn_rescue": int(events["focus_fn_rescue"].sum()),
        "focus_tp_break": int(events["focus_tp_break"].sum()),
        "focus_fp_remove_correct": int(events["focus_fp_remove_correct"].sum()),
        "focus_fp_create": int(events["focus_fp_create"].sum()),
        "candidate_correction": int(events["candidate_correction"].sum()),
        "candidate_harm": int(events["candidate_harm"].sum()),
        "restricted_fp_remove": int(restricted_remove.sum()),
        "restricted_fp_create": int(restricted_create.sum()),
        "restricted_fp_net_removal": int(
            restricted_remove.sum() - restricted_create.sum()
        ),
        "transition_matrix_baseline_to_candidate": transition.tolist(),
    }


def analyze_predictions(
    targets: np.ndarray,
    folds: np.ndarray,
    angle_probabilities: np.ndarray,
    *,
    peak_cuda_bytes: int,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    stats = c4_statistics(angle_probabilities)
    baseline_probabilities = np.asarray(angle_probabilities, dtype=np.float64)[:, 0]
    candidate_probabilities = stats["consensus_probabilities"]
    baseline_predictions = baseline_probabilities.argmax(axis=1)
    candidate_predictions = candidate_probabilities.argmax(axis=1)
    if targets.size != folds.size or targets.size != baseline_predictions.size:
        raise ValueError("analysis arrays do not align")

    baseline_metrics = _metrics(targets, baseline_probabilities)
    candidate_metrics = _metrics(targets, candidate_probabilities)
    directional = _directional_summary(
        targets, baseline_predictions, candidate_predictions
    )
    delta_p1 = (
        candidate_probabilities[:, FOCUS_CLASS]
        - baseline_probabilities[:, FOCUS_CLASS]
    )
    class1_tp = (targets == FOCUS_CLASS) & (baseline_predictions == FOCUS_CLASS)
    class1_fn = (targets == FOCUS_CLASS) & (baseline_predictions != FOCUS_CLASS)
    restricted_fp = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES) & (
        baseline_predictions == FOCUS_CLASS
    )
    remaining = ~(class1_tp | class1_fn | restricted_fp)
    cohort_masks = {
        "class1_tp": class1_tp,
        "class1_fn": class1_fn,
        "restricted_fp": restricted_fp,
        "remaining": remaining,
    }
    cohort_summary: Dict[str, object] = {}
    for name, mask in cohort_masks.items():
        cohort_summary[name] = {
            "rows": int(mask.sum()),
            "delta_p1": _numeric_summary(delta_p1[mask]),
            "mean_js_divergence": _numeric_summary(
                stats["mean_js_divergence"][mask]
            ),
            "focus_probability_range": _numeric_summary(
                stats["focus_probability_range"][mask]
            ),
            "all_angle_top1_agreement_rate": float(
                np.asarray(stats["top1_all_agree"])[mask].mean()
            )
            if bool(mask.any())
            else 0.0,
        }

    direction_mask = class1_tp | restricted_fp
    direction_labels = restricted_fp[direction_mask].astype(np.int64)
    delta_auc = _safe_auc(direction_labels, -delta_p1[direction_mask])
    js_auc = _safe_auc(
        direction_labels, np.asarray(stats["mean_js_divergence"])[direction_mask]
    )

    fold_rows: List[Dict[str, object]] = []
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == int(fold)
        baseline_fold = _metrics(targets[selected], baseline_probabilities[selected])
        candidate_fold = _metrics(targets[selected], candidate_probabilities[selected])
        directional_fold = _directional_summary(
            targets[selected],
            baseline_predictions[selected],
            candidate_predictions[selected],
        )
        baseline_tp = int(
            ((targets[selected] == FOCUS_CLASS) & (baseline_predictions[selected] == FOCUS_CLASS)).sum()
        )
        candidate_tp = int(
            ((targets[selected] == FOCUS_CLASS) & (candidate_predictions[selected] == FOCUS_CLASS)).sum()
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(selected.sum()),
                "baseline_macro_f1": float(baseline_fold["macro_f1"]),
                "candidate_macro_f1": float(candidate_fold["macro_f1"]),
                "baseline_class1_precision": float(
                    baseline_fold["per_class_precision"][FOCUS_CLASS]
                ),
                "candidate_class1_precision": float(
                    candidate_fold["per_class_precision"][FOCUS_CLASS]
                ),
                "baseline_class1_recall": float(
                    baseline_fold["per_class_recall"][FOCUS_CLASS]
                ),
                "candidate_class1_recall": float(
                    candidate_fold["per_class_recall"][FOCUS_CLASS]
                ),
                "baseline_class1_f1": float(
                    baseline_fold["per_class_f1"][FOCUS_CLASS]
                ),
                "candidate_class1_f1": float(
                    candidate_fold["per_class_f1"][FOCUS_CLASS]
                ),
                "restricted_fp_remove": directional_fold["restricted_fp_remove"],
                "restricted_fp_create": directional_fold["restricted_fp_create"],
                "restricted_fp_net_removal": directional_fold[
                    "restricted_fp_net_removal"
                ],
                "focus_fn_rescue": directional_fold["focus_fn_rescue"],
                "focus_tp_break": directional_fold["focus_tp_break"],
                "class1_net_tp_loss": int(baseline_tp - candidate_tp),
            }
        )

    macro_delta = float(candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"])
    class1_f1_delta = float(
        candidate_metrics["per_class_f1"][FOCUS_CLASS]
        - baseline_metrics["per_class_f1"][FOCUS_CLASS]
    )
    class1_precision_delta = float(
        candidate_metrics["per_class_precision"][FOCUS_CLASS]
        - baseline_metrics["per_class_precision"][FOCUS_CLASS]
    )
    class1_recall_delta = float(
        candidate_metrics["per_class_recall"][FOCUS_CLASS]
        - baseline_metrics["per_class_recall"][FOCUS_CLASS]
    )
    nonfocus_f1_losses = {
        str(index): float(
            baseline_metrics["per_class_f1"][index]
            - candidate_metrics["per_class_f1"][index]
        )
        for index in (0, 2, 3, 4)
    }
    tp_delta_mean = float(cohort_summary["class1_tp"]["delta_p1"]["mean"])
    fp_delta_mean = float(cohort_summary["restricted_fp"]["delta_p1"]["mean"])
    mechanism_gates = {
        "macro_f1_preserved": macro_delta >= 0.0,
        "class1_f1_gain_0005": class1_f1_delta >= 0.005,
        "class1_precision_gain_0010": class1_precision_delta >= 0.010,
        "class1_recall_delta_ge_minus0005": class1_recall_delta >= -0.005,
        "restricted_fp_net_removal_10": int(
            directional["restricted_fp_net_removal"]
        )
        >= 10,
        "corrections_not_below_harms": int(directional["candidate_correction"])
        >= int(directional["candidate_harm"]),
        "focus_tp_break_not_above_rescue": int(directional["focus_tp_break"])
        <= int(directional["focus_fn_rescue"]),
        "nonfocus_max_f1_loss_le_0010": max(nonfocus_f1_losses.values()) <= 0.010,
        "restricted_fp_delta_p1_selective_0010": tp_delta_mean - fp_delta_mean
        >= 0.010,
        "delta_p1_direction_auroc_062": delta_auc is not None and delta_auc >= 0.62,
        "fold_precision_nonworse_4": sum(
            float(row["candidate_class1_precision"])
            >= float(row["baseline_class1_precision"])
            for row in fold_rows
        )
        >= 4,
        "fold_restricted_fp_positive_4": sum(
            int(row["restricted_fp_net_removal"]) > 0 for row in fold_rows
        )
        >= 4,
        "fold_class1_net_tp_loss_le_2": all(
            int(row["class1_net_tp_loss"]) <= 2 for row in fold_rows
        ),
        "peak_cuda_allocation_le_75_gib": int(peak_cuda_bytes)
        <= int(7.5 * 1024**3),
    }
    return {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": {
            "macro_f1": macro_delta,
            "class1_f1": class1_f1_delta,
            "class1_precision": class1_precision_delta,
            "class1_recall": class1_recall_delta,
            "nonfocus_f1_losses": nonfocus_f1_losses,
        },
        "directional": directional,
        "cohorts": cohort_summary,
        "direction": {
            "rows": int(direction_mask.sum()),
            "restricted_fp_rows": int(restricted_fp.sum()),
            "class1_tp_rows": int(class1_tp.sum()),
            "delta_p1_auroc_fp_vs_tp": delta_auc,
            "js_auroc_fp_vs_tp": js_auc,
            "tp_minus_fp_mean_delta_p1": float(tp_delta_mean - fp_delta_mean),
        },
        "rotation": {
            "all_angle_top1_agreement_rate": float(
                np.asarray(stats["top1_all_agree"]).mean()
            ),
            "mean_top1_agreement_count": float(
                np.asarray(stats["top1_agreement_count"]).mean()
            ),
            "mean_js_divergence": float(
                np.asarray(stats["mean_js_divergence"]).mean()
            ),
            "mean_focus_probability_range": float(
                np.asarray(stats["focus_probability_range"]).mean()
            ),
        },
        "folds": fold_rows,
        "mechanism_gates": mechanism_gates,
        "mechanism_gates_passed": all(mechanism_gates.values()),
    }


def _read_cidt_clean(path: Path, *, num_classes: int) -> Dict[str, object]:
    rows: List[Mapping[str, str]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("condition", "")) == "clean":
                rows.append(row)
    rows.sort(key=lambda row: int(row["sample_index"]))
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"CIDT clean support mismatch: {len(rows)} != {EXPECTED_TRAIN_ROWS}"
        )
    indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)):
        raise ValueError("CIDT clean sample indices are not complete and ordered")
    probabilities = np.asarray(
        [
            [float(row[f"keeper_prob_{index}"]) for index in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "indices": indices,
        "targets": np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64),
        "predictions": np.asarray(
            [int(row["keeper_prediction"]) for row in rows], dtype=np.int64
        ),
        "probabilities": probabilities,
        "folds": np.asarray([int(row["fold"]) for row in rows], dtype=np.int64),
        "paths": [str(row["image_path"]) for row in rows],
        "source_stems": [str(row["source_stem"]) for row in rows],
    }


def _infer_c4(
    *,
    model,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context="c4_rotation_consensus",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    all_indices: List[Tensor] = []
    all_targets: List[Tensor] = []
    all_probabilities: List[Tensor] = []
    processed = 0
    started = time.perf_counter()
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_angles: List[Tensor] = []
            for k in ROTATIONS:
                rotated_images, rotated_metadata = rotate_appearance_batch(
                    images, metadata, k=k
                )
                logits, features = _forward_classification_with_metadata(
                    model, rotated_images, rotated_metadata, device=device
                )
                del features
                batch_angles.append(F.softmax(logits.float(), dim=1).detach().cpu())
            all_indices.append(metadata["sample_index"].detach().cpu().to(torch.long))
            all_targets.append(targets.detach().cpu().to(torch.long))
            all_probabilities.append(torch.stack(batch_angles, dim=1))
            processed += int(targets.numel())
            if processed % 1024 < int(targets.numel()) or processed == len(dataset):
                print(
                    json.dumps(
                        {
                            "stage": "c4_inference",
                            "processed": processed,
                            "rows": len(dataset),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    return {
        "indices": torch.cat(all_indices).numpy(),
        "targets": torch.cat(all_targets).numpy(),
        "angle_probabilities": torch.cat(all_probabilities).numpy(),
        "loader": loader_summary,
        "elapsed_seconds": elapsed,
        "throughput_source_images_per_second": float(len(dataset) / max(elapsed, 1e-12)),
        "throughput_forward_views_per_second": float(
            len(dataset) * len(ROTATIONS) / max(elapsed, 1e-12)
        ),
        "peak_cuda_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


def _prediction_rows(
    *,
    targets: np.ndarray,
    folds: np.ndarray,
    paths: Sequence[Path],
    source_stems: Sequence[str],
    angle_probabilities: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    stats = c4_statistics(angle_probabilities)
    baseline = np.asarray(angle_probabilities, dtype=np.float64)[:, 0]
    candidate = np.asarray(stats["consensus_probabilities"], dtype=np.float64)
    baseline_predictions = baseline.argmax(axis=1)
    candidate_predictions = candidate.argmax(axis=1)
    delta_p1 = candidate[:, FOCUS_CLASS] - baseline[:, FOCUS_CLASS]
    events = directional_event_masks(targets, baseline_predictions, candidate_predictions)
    restricted = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
    restricted_remove = restricted & (baseline_predictions == FOCUS_CLASS) & (
        candidate_predictions != FOCUS_CLASS
    )
    restricted_create = restricted & (baseline_predictions != FOCUS_CLASS) & (
        candidate_predictions == FOCUS_CLASS
    )
    rows: List[Dict[str, object]] = []
    for row_index in range(targets.size):
        target = int(targets[row_index])
        base_prediction = int(baseline_predictions[row_index])
        if target == FOCUS_CLASS and base_prediction == FOCUS_CLASS:
            cohort = "class1_tp"
        elif target == FOCUS_CLASS:
            cohort = "class1_fn"
        elif target in RESTRICTED_NEGATIVE_CLASSES and base_prediction == FOCUS_CLASS:
            cohort = "restricted_fp"
        else:
            cohort = "remaining"
        row: Dict[str, object] = {
            "sample_index": int(row_index),
            "image_path": str(Path(paths[row_index]).resolve()),
            "source_stem": str(source_stems[row_index]),
            "fold": int(folds[row_index]),
            "target_index": target,
            "target_name": str(class_names[target]),
            "cohort": cohort,
            "baseline_prediction": base_prediction,
            "candidate_prediction": int(candidate_predictions[row_index]),
            "delta_p1": float(delta_p1[row_index]),
            "focus_probability_range": float(stats["focus_probability_range"][row_index]),
            "focus_probability_std": float(stats["focus_probability_std"][row_index]),
            "mean_js_divergence": float(stats["mean_js_divergence"][row_index]),
            "top1_agreement_count": int(stats["top1_agreement_count"][row_index]),
            "top1_all_agree": bool(stats["top1_all_agree"][row_index]),
            "focus_fn_rescue": bool(events["focus_fn_rescue"][row_index]),
            "focus_tp_break": bool(events["focus_tp_break"][row_index]),
            "focus_fp_remove_correct": bool(
                events["focus_fp_remove_correct"][row_index]
            ),
            "focus_fp_create": bool(events["focus_fp_create"][row_index]),
            "candidate_correction": bool(events["candidate_correction"][row_index]),
            "candidate_harm": bool(events["candidate_harm"][row_index]),
            "restricted_fp_remove": bool(restricted_remove[row_index]),
            "restricted_fp_create": bool(restricted_create[row_index]),
        }
        for class_index in range(len(class_names)):
            row[f"baseline_prob_{class_index}"] = float(
                baseline[row_index, class_index]
            )
            row[f"candidate_prob_{class_index}"] = float(
                candidate[row_index, class_index]
            )
            for angle_index, k in enumerate(ROTATIONS):
                row[f"rot{k}_prob_{class_index}"] = float(
                    angle_probabilities[row_index, angle_index, class_index]
                )
        rows.append(row)
    return rows


def _artifact_record(path: Path, *, root: Path) -> Dict[str, object]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_manifest(output_dir: Path) -> Path:
    manifest_path = output_dir / "artifact_manifest.json"
    payloads = [
        _artifact_record(path, root=output_dir)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "mode": "c4_rotation_consensus_a0_artifact_manifest",
            "payload_count": len(payloads),
            "payload_bytes": int(sum(int(row["bytes"]) for row in payloads)),
            "payloads": payloads,
        },
    )
    return manifest_path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(payload.get("payloads", []))
    for row in rows:
        path = output_dir / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Manifest payload is missing: {path}")
        if int(path.stat().st_size) != int(row["bytes"]):
            raise ValueError(f"Manifest size mismatch: {path}")
        if _sha256(path) != str(row["sha256"]):
            raise ValueError(f"Manifest SHA-256 mismatch: {path}")
    return {
        "payload_count": len(rows),
        "payload_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "manifest_sha256": _sha256(manifest_path),
    }


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError("Replay mapping keys differ")
        return max(
            (_recursive_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Replay list lengths differ")
        return max(
            (_recursive_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        if bool(left) != bool(right):
            raise ValueError("Replay boolean values differ")
        return 0.0
    if left is None or right is None:
        if left is not right:
            raise ValueError("Replay null values differ")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError(f"Replay values differ: {left!r} != {right!r}")
    return 0.0


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    output_dir = resolved.parent
    manifest = _verify_manifest(output_dir)
    rows: List[Mapping[str, str]] = []
    with (output_dir / "predictions.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows.extend(csv.DictReader(handle))
    targets = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    class_count = len(summary["class_names"])
    angle_probabilities = np.asarray(
        [
            [
                [float(row[f"rot{k}_prob_{class_index}"]) for class_index in range(class_count)]
                for k in ROTATIONS
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    replayed = analyze_predictions(
        targets,
        folds,
        angle_probabilities,
        peak_cuda_bytes=int(summary["runtime"]["peak_cuda_bytes"]),
    )
    maximum_difference = _recursive_numeric_difference(summary["analysis"], replayed)
    if maximum_difference > 1e-12:
        raise ValueError(
            f"Replay numerical mismatch exceeds tolerance: {maximum_difference}"
        )
    return {
        "replay_passed": True,
        "rows": len(rows),
        "maximum_numeric_difference": float(maximum_difference),
        "mechanism_gates_passed": bool(replayed["mechanism_gates_passed"]),
        **manifest,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1 or int(args.num_workers) < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if int(args.folds) != 5 or int(args.seed) != 20260714:
        raise ValueError("The prospective protocol locks folds=5 and seed=20260714")
    if int(args.batch_size) != 64 or int(args.num_workers) != 4:
        raise ValueError("The prospective protocol locks batch-size=64 and num-workers=4")

    provenance = verify_locked_inputs()
    repo_state = _repo_state()
    if not bool(repo_state["head_matches_upstream"]) or not bool(
        repo_state["tracked_worktree_clean"]
    ):
        raise ValueError(f"Formal audit requires clean pushed tracked state: {repo_state}")

    output_dir = _require_empty_output(args.output_dir)
    device_name = str(args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    set_seed(int(args.seed), deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True

    file_locks = provenance["files"]
    keeper_path = Path(file_locks["keeper"]["path"])
    data_path = Path(file_locks["data_yaml"]["path"])
    model, checkpoint, class_names = load_model(keeper_path, device)
    semantics = _eval_semantics(checkpoint)
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("C4 A0 supports temporal_frames=1 only")
    if bool(semantics.get("classification_source_context_aux", False)):
        raise ValueError("C4 A0 excludes source-context auxiliary input")

    data_spec = load_data_spec(
        data_path, class_name_mode="raw", expected_num_classes=len(class_names)
    )
    if list(data_spec.class_names) != list(class_names):
        raise ValueError("Dataset and checkpoint class orders differ")
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
    if len(base_dataset) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Full train support mismatch: {len(base_dataset)} != {EXPECTED_TRAIN_ROWS}"
        )
    labels = np.asarray(base_dataset.labels(), dtype=np.int64)
    paths = [Path(path).resolve() for path in base_dataset.sample_paths()]
    source_stems = np.asarray([path.stem.casefold() for path in paths], dtype=str)
    folds, fold_assignment = _assign_source_folds(
        labels, source_stems, folds=int(args.folds), seed=int(args.seed)
    )
    dataset_identity = _dataset_identity(base_dataset, range(len(base_dataset)))

    train_sources = set(source_stems.tolist())
    val_sources = _split_source_stems(data_spec.val_images)
    test_sources = (
        _split_source_stems(data_spec.test_images)
        if data_spec.test_images is not None
        else set()
    )
    train_val_overlap = sorted(train_sources.intersection(val_sources))
    train_test_overlap = sorted(train_sources.intersection(test_sources))

    transform = _build_eval_transform(semantics)
    inference_dataset = _SelectedConditionDataset(
        base_dataset,
        list(range(len(base_dataset))),
        corruption=IdentityCorruption(),
        transform=transform,
    )
    inference = _infer_c4(
        model=model,
        dataset=inference_dataset,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    indices = np.asarray(inference["indices"], dtype=np.int64)
    targets = np.asarray(inference["targets"], dtype=np.int64)
    angle_probabilities = np.asarray(inference["angle_probabilities"], dtype=np.float64)

    cidt = _read_cidt_clean(
        Path(file_locks["cidt_predictions"]["path"]), num_classes=len(class_names)
    )
    baseline_probabilities = angle_probabilities[:, 0]
    baseline_predictions = baseline_probabilities.argmax(axis=1)
    cidt_probability_max_abs = float(
        np.max(np.abs(baseline_probabilities - cidt["probabilities"]))
    )
    cidt_checks = {
        "sample_indices_exact": bool(np.array_equal(indices, cidt["indices"])),
        "targets_exact": bool(np.array_equal(targets, cidt["targets"])),
        "predictions_exact": bool(
            np.array_equal(baseline_predictions, cidt["predictions"])
        ),
        "folds_exact": bool(np.array_equal(folds, cidt["folds"])),
        "paths_exact": [str(path) for path in paths] == list(cidt["paths"]),
        "source_stems_exact": source_stems.tolist() == list(cidt["source_stems"]),
        "probability_max_abs_le_2e_5": cidt_probability_max_abs <= 2e-5,
    }

    analysis = analyze_predictions(
        targets,
        folds,
        angle_probabilities,
        peak_cuda_bytes=int(inference["peak_cuda_bytes"]),
    )
    rows = _prediction_rows(
        targets=targets,
        folds=folds,
        paths=paths,
        source_stems=source_stems.tolist(),
        angle_probabilities=angle_probabilities,
        class_names=class_names,
    )
    prediction_path = output_dir / "predictions.csv"
    fold_path = output_dir / "fold_summary.csv"
    cohort_path = output_dir / "cohort_summary.json"
    _write_csv(prediction_path, rows)
    _write_csv(fold_path, analysis["folds"])
    _write_json(cohort_path, analysis["cohorts"])

    structural_gates = {
        "locked_hashes_verified": True,
        "repo_clean_and_pushed": bool(repo_state["head_matches_upstream"])
        and bool(repo_state["tracked_worktree_clean"]),
        "train_split_only": True,
        "full_train_support_9215": len(base_dataset) == EXPECTED_TRAIN_ROWS,
        "ordered_indices_complete": bool(
            np.array_equal(indices, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64))
        ),
        "targets_match_dataset": bool(np.array_equal(targets, labels)),
        "four_angles_complete": angle_probabilities.shape
        == (EXPECTED_TRAIN_ROWS, len(ROTATIONS), len(class_names)),
        "probabilities_finite_normalized": bool(np.isfinite(angle_probabilities).all())
        and bool(
            np.allclose(
                angle_probabilities.sum(axis=2), 1.0, atol=1e-5, rtol=0.0
            )
        ),
        "source_fold_assignment_complete": bool(
            fold_assignment["assignment_complete"]
        ),
        "source_fold_overlap_zero": int(fold_assignment["source_overlap"]) == 0,
        "train_val_source_overlap_zero": len(train_val_overlap) == 0,
        "train_test_source_overlap_zero": len(train_test_overlap) == 0,
        "cidt_alignment_complete": all(cidt_checks.values()),
        "peak_cuda_allocation_le_75_gib": int(inference["peak_cuda_bytes"])
        <= int(7.5 * 1024**3),
    }
    structural_passed = all(structural_gates.values())
    all_gates_passed = structural_passed and bool(analysis["mechanism_gates_passed"])

    artifact_hashes = {
        "predictions": _artifact_record(prediction_path, root=output_dir),
        "fold_summary": _artifact_record(fold_path, root=output_dir),
        "cohort_summary": _artifact_record(cohort_path, root=output_dir),
    }
    summary = {
        "mode": "c4_rotation_consensus_a0_train_information_gate",
        "status": "passed" if all_gates_passed else "rejected",
        "split": "train",
        "test_data_used": False,
        "validation_data_used": False,
        "validation_test_filename_scan_only": True,
        "rows": len(base_dataset),
        "class_names": list(class_names),
        "focus_class": FOCUS_CLASS,
        "restricted_negative_classes": list(RESTRICTED_NEGATIVE_CLASSES),
        "rotations_ccw_quarter_turns": list(ROTATIONS),
        "aggregation": "arithmetic_mean_softmax_probability",
        "bbox_policy": "source_coordinate_bbox_frozen",
        "dataset_identity_sha256": dataset_identity,
        "eval_semantics": semantics,
        "provenance": provenance,
        "repo_state": repo_state,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "batch_size": int(args.batch_size),
            "requested_num_workers": int(args.num_workers),
            "loader": inference["loader"],
            "elapsed_seconds": inference["elapsed_seconds"],
            "throughput_source_images_per_second": inference[
                "throughput_source_images_per_second"
            ],
            "throughput_forward_views_per_second": inference[
                "throughput_forward_views_per_second"
            ],
            "peak_cuda_bytes": inference["peak_cuda_bytes"],
            "peak_cuda_gib": float(inference["peak_cuda_bytes"] / 1024**3),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "source_audit": {
            "train_sources": len(train_sources),
            "val_sources": len(val_sources),
            "test_sources": len(test_sources),
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "overlap_examples": (train_val_overlap + train_test_overlap)[:20],
            "fold_assignment": fold_assignment,
            "note": "Validation/test filenames only; pixels, labels, predictions, and metrics were not opened.",
        },
        "cidt_replay": {
            "checks": cidt_checks,
            "probability_max_abs_difference": cidt_probability_max_abs,
        },
        "analysis": analysis,
        "structural_gates": structural_gates,
        "structural_gates_passed": structural_passed,
        "all_gates_passed": all_gates_passed,
        "artifacts": artifact_hashes,
        "guardrail": (
            "Only a complete pass authorizes one standard-op, parameter-neutral C4 "
            "stem-orbit implementation. Any failure closes this route before model "
            "changes, validation, XAI, smoke, probe, test, or full train."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(output_dir)
    summary["artifact_manifest"] = {
        "path": manifest_path.name,
        "sha256": _sha256(manifest_path),
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.preflight_only and args.replay_summary is not None:
        raise ValueError("preflight-only and replay-summary are mutually exclusive")
    if args.preflight_only:
        if Path(args.output_dir).expanduser().resolve().exists():
            raise FileExistsError(
                f"Preflight output path must not exist: {Path(args.output_dir).resolve()}"
            )
        payload = {
            "preflight_passed": True,
            "provenance": verify_locked_inputs(),
            "repo_state": _repo_state(),
            "output_created": False,
        }
        print(json.dumps(to_serializable(payload), indent=2), flush=True)
        return
    if args.replay_summary is not None:
        print(
            json.dumps(to_serializable(replay_summary(args.replay_summary)), indent=2),
            flush=True,
        )
        return
    summary = run_audit(args)
    print(
        json.dumps(
            to_serializable(
                {
                    "status": summary["status"],
                    "rows": summary["rows"],
                    "deltas": summary["analysis"]["deltas"],
                    "directional": summary["analysis"]["directional"],
                    "failed_mechanism_gates": [
                        name
                        for name, passed in summary["analysis"][
                            "mechanism_gates"
                        ].items()
                        if not bool(passed)
                    ],
                    "structural_gates_passed": summary[
                        "structural_gates_passed"
                    ],
                    "all_gates_passed": summary["all_gates_passed"],
                    "artifact_manifest": summary["artifact_manifest"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    if not bool(summary["structural_gates_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)


SEED = 20260712
FOLDS = 5
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
PRIOR_WEIGHT = 1.0
FOCUS_CLASS_INDEX = 1
CALIBRATION_BINS = 15


class KeeperResidualReadout(nn.Module):
    def __init__(self, feature_dim: int, class_count: int) -> None:
        super().__init__()
        self.residual = nn.Linear(int(feature_dim), int(class_count))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features: Tensor, base_log_probabilities: Tensor) -> Tensor:
        return base_log_probabilities + self.residual(features)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only readiness audit for a keeper-preserving, two-stage "
            "Re-EDL residual readout. The audit never reads test or writes a model."
        )
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def _assert_non_test_cache(path: Path, split: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if str(split).casefold() == "test" or "/test/" in normalized or "_test" in path.stem.casefold():
        raise ValueError(f"Test input is forbidden for readiness audit: {path}")


def _load_cache(path: Path, *, split: str) -> Dict[str, np.ndarray]:
    _assert_non_test_cache(path, split)
    with np.load(path, allow_pickle=True) as cache:
        required = (
            "embeddings",
            "probabilities",
            "labels",
            "base_predictions",
            "paths",
            "sample_index",
        )
        missing = [key for key in required if key not in cache.files]
        if missing:
            raise ValueError(f"Embedding cache is missing {missing}: {path}")
        payload = {key: np.asarray(cache[key]) for key in required}

    rows = int(payload["labels"].shape[0])
    if rows == 0 or any(int(payload[key].shape[0]) != rows for key in required):
        raise ValueError(f"Embedding cache arrays have invalid row counts: {path}")
    payload["embeddings"] = np.asarray(payload["embeddings"], dtype=np.float32)
    payload["probabilities"] = np.asarray(payload["probabilities"], dtype=np.float32)
    payload["labels"] = np.asarray(payload["labels"], dtype=np.int64)
    payload["base_predictions"] = np.asarray(payload["base_predictions"], dtype=np.int64)
    payload["paths"] = np.asarray(payload["paths"], dtype=object)
    payload["sample_index"] = np.asarray(payload["sample_index"], dtype=np.int64)
    if payload["embeddings"].ndim != 2 or payload["probabilities"].ndim != 2:
        raise ValueError(f"Expected 2D embeddings and probabilities: {path}")
    if not np.isfinite(payload["embeddings"]).all() or not np.isfinite(payload["probabilities"]).all():
        raise ValueError(f"Embedding cache contains non-finite values: {path}")
    if np.any(payload["probabilities"] < 0.0):
        raise ValueError(f"Embedding cache contains negative probabilities: {path}")
    probability_sums = payload["probabilities"].sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=2e-4):
        raise ValueError(f"Embedding cache probabilities are not normalized: {path}")
    if np.unique(payload["sample_index"]).size != rows:
        raise ValueError(f"Embedding cache sample_index is not unique: {path}")
    derived_predictions = payload["probabilities"].argmax(axis=1)
    if not np.array_equal(derived_predictions, payload["base_predictions"]):
        raise ValueError(f"Embedding cache base predictions do not match probabilities: {path}")
    path_splits = {
        Path(str(value)).parent.name.casefold()
        for value in payload["paths"].tolist()
        if str(value).strip()
    }
    if "test" in path_splits:
        raise ValueError(f"Test rows are forbidden for readiness audit: {path}")
    payload["source_stems"] = np.asarray(
        [Path(str(value)).stem.casefold() for value in payload["paths"]],
        dtype=object,
    )
    return payload


def reedl_projected_probabilities(
    logits: Tensor,
    *,
    prior_weight: float = PRIOR_WEIGHT,
) -> Tuple[Tensor, Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"Expected [batch, classes] logits, got {tuple(logits.shape)}")
    prior = float(prior_weight)
    if prior <= 0.0:
        raise ValueError("prior_weight must be positive")
    evidence = F.softplus(logits)
    alpha = evidence + prior
    strength = alpha.sum(dim=1, keepdim=True)
    probabilities = alpha / strength
    uncertainty = prior * float(logits.shape[1]) / strength.squeeze(1)
    return probabilities, uncertainty, evidence


def _fit_scaler(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(features, dtype=np.float64).mean(axis=0)
    scale = np.asarray(features, dtype=np.float64).std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _apply_scaler(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - mean) / scale).astype(
        np.float32,
        copy=False,
    )


def _train_stage(
    model: KeeperResidualReadout,
    *,
    features: np.ndarray,
    base_probabilities: np.ndarray,
    labels: np.ndarray,
    mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    prior_weight: float,
    seed: int,
    device: torch.device,
    curve_prefix: Mapping[str, object],
) -> list[Dict[str, object]]:
    if mode not in {"ce", "reedl"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    base_log = torch.from_numpy(
        np.log(np.clip(np.asarray(base_probabilities, dtype=np.float32), 1e-7, 1.0))
    ).to(device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    rows = int(y.numel())
    output: list[Dict[str, object]] = []
    for epoch in range(int(epochs)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + epoch)
        order = torch.randperm(rows, generator=generator)
        loss_sum = 0.0
        correct = 0
        for start in range(0, rows, int(batch_size)):
            indices = order[start : start + int(batch_size)].to(device)
            logits = model(x[indices], base_log[indices])
            if mode == "ce":
                loss = F.cross_entropy(logits, y[indices])
                probabilities = logits.softmax(dim=1)
            else:
                probabilities, _, _ = reedl_projected_probabilities(
                    logits,
                    prior_weight=prior_weight,
                )
                loss = F.nll_loss(torch.log(probabilities.clamp_min(1e-8)), y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_rows = int(indices.numel())
            loss_sum += float(loss.detach().cpu()) * batch_rows
            correct += int((probabilities.argmax(dim=1) == y[indices]).sum().detach().cpu())
        output.append(
            {
                **dict(curve_prefix),
                "mode": mode,
                "epoch": int(epoch + 1),
                "loss": float(loss_sum / rows),
                "accuracy": float(correct / rows),
            }
        )
    return output


@torch.inference_mode()
def _predict(
    model: KeeperResidualReadout,
    *,
    features: np.ndarray,
    base_probabilities: np.ndarray,
    mode: str,
    prior_weight: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    base_log = torch.from_numpy(
        np.log(np.clip(np.asarray(base_probabilities, dtype=np.float32), 1e-7, 1.0))
    ).to(device)
    logits = model(x, base_log)
    if mode == "ce":
        probabilities = logits.softmax(dim=1)
        uncertainty = _entropy_tensor(probabilities)
    elif mode == "reedl":
        probabilities, uncertainty, _ = reedl_projected_probabilities(
            logits,
            prior_weight=prior_weight,
        )
    else:
        raise ValueError(f"Unsupported prediction mode: {mode}")
    return (
        probabilities.cpu().numpy().astype(np.float32, copy=False),
        uncertainty.cpu().numpy().astype(np.float32, copy=False),
        logits.cpu().numpy().astype(np.float32, copy=False),
    )


def _entropy_tensor(probabilities: Tensor) -> Tensor:
    class_count = int(probabilities.shape[1])
    return -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-8)),
        dim=1,
    ) / math.log(max(2, class_count))


def _classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    class_count = int(probabilities.shape[1])
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    per_class = []
    f1_values = []
    for class_index in range(class_count):
        tp = int(confusion[class_index, class_index])
        support = int(confusion[class_index].sum())
        predicted = int(confusion[:, class_index].sum())
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": int(class_index),
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    selected = probabilities[np.arange(targets.size), targets]
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    return {
        "samples": int(targets.size),
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(f1_values)),
        "nll": float(-np.log(np.clip(selected, 1e-8, 1.0)).mean()),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _transition_stats(
    targets: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class_index: int,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    reference = np.asarray(reference_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    changed = reference != candidate
    corrections = changed & (reference != targets) & (candidate == targets)
    harms = changed & (reference == targets) & (candidate != targets)
    neutral = changed & ~corrections & ~harms
    focus = int(focus_class_index)
    return {
        "changed": int(changed.sum()),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "neutral": int(neutral.sum()),
        "focus_false_positive_removed": int(
            ((reference == focus) & (targets != focus) & (candidate != focus)).sum()
        ),
        "focus_false_positive_created": int(
            ((reference != focus) & (targets != focus) & (candidate == focus)).sum()
        ),
        "focus_false_negative_rescued": int(
            ((targets == focus) & (reference != focus) & (candidate == focus)).sum()
        ),
        "focus_true_positive_broken": int(
            ((targets == focus) & (reference == focus) & (candidate != focus)).sum()
        ),
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def _direction_auc(
    targets: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class_index: int,
) -> Dict[str, object]:
    control_predictions = np.asarray(control_probabilities).argmax(axis=1)
    focus = int(focus_class_index)
    false_positive = (control_predictions == focus) & (targets != focus)
    false_negative = (targets == focus) & (control_predictions != focus)
    mask = false_positive | false_negative
    labels = false_negative[mask].astype(np.int64)
    score = (
        np.asarray(candidate_probabilities)[mask, focus]
        - np.asarray(control_probabilities)[mask, focus]
    )
    return {
        "auc_fn_positive": _safe_auc(labels, score),
        "samples": int(mask.sum()),
        "false_positives": int(false_positive.sum()),
        "false_negatives": int(false_negative.sum()),
        "mean_delta_on_false_positive": (
            float(score[labels == 0].mean()) if bool(np.any(labels == 0)) else None
        ),
        "mean_delta_on_false_negative": (
            float(score[labels == 1].mean()) if bool(np.any(labels == 1)) else None
        ),
    }


def _calibration_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = CALIBRATION_BINS,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)

    def ece(labels: np.ndarray, scores: np.ndarray) -> float:
        total = int(labels.size)
        value = 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            include_right = right == 1.0
            mask = (scores >= left) & ((scores <= right) if include_right else (scores < right))
            if not mask.any():
                continue
            value += float(mask.sum() / total) * abs(
                float(labels[mask].mean()) - float(scores[mask].mean())
            )
        return float(value)

    classwise = []
    for class_index in range(probabilities.shape[1]):
        labels = (targets == class_index).astype(np.float64)
        classwise.append(
            {
                "class_index": int(class_index),
                "ece": ece(labels, probabilities[:, class_index]),
            }
        )
    return {
        "top_label_ece": ece(correct, confidence),
        "mean_classwise_ece": float(np.mean([row["ece"] for row in classwise])),
        "per_class": classwise,
    }


def _risk_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    *,
    model_name: str,
    split: str,
    focus_class_index: int,
) -> Tuple[Dict[str, object], list[Dict[str, object]]]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    errors = (predictions != targets).astype(np.int64)
    order = np.argsort(uncertainty, kind="mergesort")
    cumulative_errors = np.cumsum(errors[order])
    risk = cumulative_errors / np.arange(1, len(order) + 1)
    rows = []
    for coverage in (0.50, 0.70, 0.80, 0.90, 0.95, 1.00):
        count = min(len(order), max(1, int(math.ceil(coverage * len(order)))))
        selected = order[:count]
        metrics = _classification_metrics(targets[selected], probabilities[selected])
        rows.append(
            {
                "split": split,
                "model": model_name,
                "coverage": float(count / len(order)),
                "accepted": int(count),
                "risk": float(np.mean(errors[selected])),
                "macro_f1": float(metrics["macro_f1"]),
                "focus_f1": float(metrics["per_class"][int(focus_class_index)]["f1"]),
                "focus_support": int(metrics["per_class"][int(focus_class_index)]["support"]),
            }
        )
    error_auroc = _safe_auc(errors, uncertainty)
    error_average_precision = (
        float(average_precision_score(errors, uncertainty))
        if np.unique(errors).size >= 2
        else None
    )
    return (
        {
            "aurc": float(risk.mean()),
            "error_auroc": error_auroc,
            "error_average_precision": error_average_precision,
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_correct_mean": (
                float(uncertainty[errors == 0].mean()) if bool(np.any(errors == 0)) else None
            ),
            "uncertainty_error_mean": (
                float(uncertainty[errors == 1].mean()) if bool(np.any(errors == 1)) else None
            ),
            "coverage_rows": rows,
        },
        rows,
    )


def assess_two_stage_reedl_readiness(
    *,
    train_summary: Mapping[str, object],
    val_summary: Mapping[str, object],
    matched_source_folds: bool,
    focus_class_index: int,
) -> Dict[str, object]:
    focus = int(focus_class_index)

    def metric(split: Mapping[str, object], model: str, name: str) -> float:
        return float(split["models"][model]["metrics"][name])

    def focus_metric(split: Mapping[str, object], model: str, name: str) -> float:
        return float(split["models"][model]["metrics"]["per_class"][focus][name])

    train_control_macro = metric(train_summary, "ce_control", "macro_f1")
    train_candidate_macro = metric(train_summary, "two_stage_reedl", "macro_f1")
    train_control_focus = focus_metric(train_summary, "ce_control", "f1")
    train_candidate_focus = focus_metric(train_summary, "two_stage_reedl", "f1")
    val_control_macro = metric(val_summary, "ce_control", "macro_f1")
    val_candidate_macro = metric(val_summary, "two_stage_reedl", "macro_f1")
    val_control_focus = focus_metric(val_summary, "ce_control", "f1")
    val_candidate_focus = focus_metric(val_summary, "two_stage_reedl", "f1")
    val_keeper_macro = metric(val_summary, "keeper", "macro_f1")
    val_keeper_focus = focus_metric(val_summary, "keeper", "f1")
    val_keeper_recall = focus_metric(val_summary, "keeper", "recall")
    val_candidate_recall = focus_metric(val_summary, "two_stage_reedl", "recall")
    train_control_auc = train_summary["models"]["ce_control"]["risk"]["error_auroc"]
    train_candidate_auc = train_summary["models"]["two_stage_reedl"]["risk"]["error_auroc"]
    val_control_auc = val_summary["models"]["ce_control"]["risk"]["error_auroc"]
    val_candidate_auc = val_summary["models"]["two_stage_reedl"]["risk"]["error_auroc"]
    train_control_aurc = float(train_summary["models"]["ce_control"]["risk"]["aurc"])
    train_candidate_aurc = float(train_summary["models"]["two_stage_reedl"]["risk"]["aurc"])
    val_control_aurc = float(val_summary["models"]["ce_control"]["risk"]["aurc"])
    val_candidate_aurc = float(val_summary["models"]["two_stage_reedl"]["risk"]["aurc"])

    def risk_at(split: Mapping[str, object], model: str, coverage: float) -> float:
        rows = split["models"][model]["risk"]["coverage_rows"]
        row = min(rows, key=lambda value: abs(float(value["coverage"]) - float(coverage)))
        return float(row["risk"])

    train_candidate_risk_80 = risk_at(train_summary, "two_stage_reedl", 0.80)
    train_candidate_risk_100 = risk_at(train_summary, "two_stage_reedl", 1.00)
    val_candidate_risk_80 = risk_at(val_summary, "two_stage_reedl", 0.80)
    val_candidate_risk_100 = risk_at(val_summary, "two_stage_reedl", 1.00)
    train_direction = train_summary["direction"]["auc_fn_positive"]
    val_direction = val_summary["direction"]["auc_fn_positive"]
    transitions = val_summary["transitions_vs_keeper"]

    checks = {
        "matched_source_folds": bool(matched_source_folds),
        "oof_macro_gain_ge_0p002": train_candidate_macro - train_control_macro >= 0.002,
        "oof_focus_gain_ge_0p010": train_candidate_focus - train_control_focus >= 0.010,
        "val_macro_gain_ge_0p002": val_candidate_macro - val_control_macro >= 0.002,
        "val_focus_gain_ge_0p015": val_candidate_focus - val_control_focus >= 0.015,
        "val_macro_not_below_keeper": val_candidate_macro >= val_keeper_macro,
        "val_focus_milestone_ge_0p70": val_candidate_focus >= 0.70,
        "val_focus_recall_protected": val_candidate_recall >= val_keeper_recall - 0.01,
        "val_focus_fp_not_increased": int(transitions["focus_false_positive_removed"])
        >= int(transitions["focus_false_positive_created"]),
        "val_focus_tp_recall_action_safe": int(transitions["focus_false_negative_rescued"])
        >= int(transitions["focus_true_positive_broken"]),
        "val_corrections_exceed_harms": int(transitions["corrections"])
        > int(transitions["harms"]),
        "oof_uncertainty_auc_gain_ge_0p02": (
            train_candidate_auc is not None
            and train_control_auc is not None
            and float(train_candidate_auc) - float(train_control_auc) >= 0.02
        ),
        "val_uncertainty_auc_gain_ge_0p02": (
            val_candidate_auc is not None
            and val_control_auc is not None
            and float(val_candidate_auc) - float(val_control_auc) >= 0.02
        ),
        "val_uncertainty_auc_ge_0p65": (
            val_candidate_auc is not None and float(val_candidate_auc) >= 0.65
        ),
        "oof_aurc_not_worse_than_control": train_candidate_aurc <= train_control_aurc,
        "val_aurc_not_worse_than_control": val_candidate_aurc <= val_control_aurc,
        "oof_risk_decreases_at_80pct_coverage": (
            train_candidate_risk_80 < train_candidate_risk_100
        ),
        "val_risk_decreases_at_80pct_coverage": (
            val_candidate_risk_80 < val_candidate_risk_100
        ),
        "fn_fp_direction_train_ge_0p55": (
            train_direction is not None and float(train_direction) >= 0.55
        ),
        "fn_fp_direction_val_ge_0p55": (
            val_direction is not None and float(val_direction) >= 0.55
        ),
        "fn_fp_direction_gap_le_0p15": (
            train_direction is not None
            and val_direction is not None
            and abs(float(train_direction) - float(val_direction)) <= 0.15
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "smoke_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": {
            "oof_macro_gain": 0.002,
            "oof_focus_gain": 0.010,
            "val_macro_gain": 0.002,
            "val_focus_gain": 0.015,
            "val_focus_milestone": 0.70,
            "max_focus_recall_drop": 0.01,
            "uncertainty_auc_gain": 0.02,
            "val_uncertainty_auc": 0.65,
            "fn_fp_direction_auc": 0.55,
            "max_fn_fp_direction_gap": 0.15,
        },
        "observed": {
            "oof_macro_gain": train_candidate_macro - train_control_macro,
            "oof_focus_gain": train_candidate_focus - train_control_focus,
            "val_macro_gain": val_candidate_macro - val_control_macro,
            "val_focus_gain": val_candidate_focus - val_control_focus,
            "val_candidate_macro": val_candidate_macro,
            "val_candidate_focus": val_candidate_focus,
            "val_keeper_macro": val_keeper_macro,
            "val_keeper_focus": val_keeper_focus,
            "val_focus_recall_delta_vs_keeper": val_candidate_recall - val_keeper_recall,
            "oof_uncertainty_auc_gain": (
                float(train_candidate_auc) - float(train_control_auc)
                if train_candidate_auc is not None and train_control_auc is not None
                else None
            ),
            "val_uncertainty_auc_gain": (
                float(val_candidate_auc) - float(val_control_auc)
                if val_candidate_auc is not None and val_control_auc is not None
                else None
            ),
            "oof_candidate_minus_control_aurc": train_candidate_aurc - train_control_aurc,
            "val_candidate_minus_control_aurc": val_candidate_aurc - val_control_aurc,
            "oof_candidate_risk_80": train_candidate_risk_80,
            "oof_candidate_risk_100": train_candidate_risk_100,
            "val_candidate_risk_80": val_candidate_risk_80,
            "val_candidate_risk_100": val_candidate_risk_100,
            "oof_fn_fp_direction_auc": train_direction,
            "val_fn_fp_direction_auc": val_direction,
        },
    }


def _build_split_summary(
    *,
    split: str,
    targets: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    control_uncertainty: np.ndarray,
    candidate_uncertainty: np.ndarray,
    focus_class_index: int,
) -> Tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    keeper_uncertainty = (
        -np.sum(
            keeper_probabilities * np.log(np.clip(keeper_probabilities, 1e-8, 1.0)),
            axis=1,
        )
        / math.log(keeper_probabilities.shape[1])
    )
    models = {}
    risk_rows: list[Dict[str, object]] = []
    for name, probabilities, uncertainty in (
        ("keeper", keeper_probabilities, keeper_uncertainty),
        ("ce_control", control_probabilities, control_uncertainty),
        ("two_stage_reedl", candidate_probabilities, candidate_uncertainty),
    ):
        risk, rows = _risk_metrics(
            targets,
            probabilities,
            uncertainty,
            model_name=name,
            split=split,
            focus_class_index=focus_class_index,
        )
        risk_rows.extend(rows)
        models[name] = {
            "metrics": _classification_metrics(targets, probabilities),
            "calibration": _calibration_metrics(targets, probabilities),
            "risk": risk,
        }
    return (
        {
            "split": split,
            "samples": int(len(targets)),
            "models": models,
            "direction": _direction_auc(
                targets,
                control_probabilities,
                candidate_probabilities,
                focus_class_index=focus_class_index,
            ),
            "transitions_vs_control": _transition_stats(
                targets,
                control_probabilities,
                candidate_probabilities,
                focus_class_index=focus_class_index,
            ),
            "transitions_vs_keeper": _transition_stats(
                targets,
                keeper_probabilities,
                candidate_probabilities,
                focus_class_index=focus_class_index,
            ),
        },
        risk_rows,
        [
            {
                "split": split,
                "model": name,
                "top_label_ece": models[name]["calibration"]["top_label_ece"],
                "mean_classwise_ece": models[name]["calibration"]["mean_classwise_ece"],
                **{
                    f"class_{row['class_index']}_ece": row["ece"]
                    for row in models[name]["calibration"]["per_class"]
                },
            }
            for name in models
        ],
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(
    *,
    split: str,
    cache: Mapping[str, np.ndarray],
    fold_assignment: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    control_uncertainty: np.ndarray,
    candidate_uncertainty: np.ndarray,
    focus_class_index: int,
) -> list[Dict[str, object]]:
    rows = []
    focus = int(focus_class_index)
    for index in range(len(cache["labels"])):
        row: Dict[str, object] = {
            "split": split,
            "sample_index": int(cache["sample_index"][index]),
            "fold": int(fold_assignment[index]),
            "source_stem": str(cache["source_stems"][index]),
            "image_path": str(cache["paths"][index]),
            "target_index": int(cache["labels"][index]),
            "keeper_prediction_index": int(cache["probabilities"][index].argmax()),
            "control_prediction_index": int(control_probabilities[index].argmax()),
            "candidate_prediction_index": int(candidate_probabilities[index].argmax()),
            "control_uncertainty": float(control_uncertainty[index]),
            "candidate_uncertainty": float(candidate_uncertainty[index]),
            "candidate_minus_control_focus_probability": float(
                candidate_probabilities[index, focus] - control_probabilities[index, focus]
            ),
        }
        for class_index in range(control_probabilities.shape[1]):
            row[f"keeper_prob_{class_index}"] = float(cache["probabilities"][index, class_index])
            row[f"control_prob_{class_index}"] = float(control_probabilities[index, class_index])
            row[f"candidate_prob_{class_index}"] = float(candidate_probabilities[index, class_index])
        rows.append(row)
    return rows


def _plot_risk_coverage(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, split in zip(axes, ("train_oof", "val")):
        split_rows = [row for row in rows if row["split"] == split]
        for model in ("keeper", "ce_control", "two_stage_reedl"):
            model_rows = [row for row in split_rows if row["model"] == model]
            axis.plot(
                [float(row["coverage"]) for row in model_rows],
                [float(row["risk"]) for row in model_rows],
                marker="o",
                label=model,
            )
        axis.set_title(split)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective risk")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_uncertainty(
    *,
    train_targets: np.ndarray,
    train_probabilities: np.ndarray,
    train_uncertainty: np.ndarray,
    val_targets: np.ndarray,
    val_probabilities: np.ndarray,
    val_uncertainty: np.ndarray,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, split, targets, probabilities, uncertainty in (
        (axes[0], "train_oof", train_targets, train_probabilities, train_uncertainty),
        (axes[1], "val", val_targets, val_probabilities, val_uncertainty),
    ):
        errors = probabilities.argmax(axis=1) != targets
        axis.hist(uncertainty[~errors], bins=30, alpha=0.65, density=True, label="correct")
        axis.hist(uncertainty[errors], bins=30, alpha=0.65, density=True, label="error")
        axis.set_title(split)
        axis.set_xlabel("Re-EDL uncertainty mass")
        axis.set_ylabel("Density")
    axes[1].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _protocol_is_locked(args: argparse.Namespace) -> bool:
    return bool(
        int(args.folds) == FOLDS
        and int(args.stage1_epochs) == STAGE1_EPOCHS
        and int(args.stage2_epochs) == STAGE2_EPOCHS
        and int(args.batch_size) == BATCH_SIZE
        and math.isclose(float(args.learning_rate), LEARNING_RATE)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY)
        and math.isclose(float(args.prior_weight), PRIOR_WEIGHT)
        and int(args.focus_class_index) == FOCUS_CLASS_INDEX
        and int(args.seed) == SEED
        and str(args.device).casefold() == "cpu"
    )


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.folds) < 2:
        raise ValueError("folds must be at least two")
    if int(args.stage1_epochs) <= 0 or int(args.stage2_epochs) <= 0:
        raise ValueError("both training stages must be positive")
    if int(args.stage1_epochs) + int(args.stage2_epochs) > 30:
        raise ValueError("two-stage budget must not exceed 30 epochs")
    if int(args.batch_size) < 16:
        raise ValueError("batch_size must be at least 16")
    if float(args.learning_rate) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("invalid optimizer settings")
    if float(args.prior_weight) <= 0.0:
        raise ValueError("prior_weight must be positive")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(str(args.device))
    if device.type != "cpu":
        raise ValueError("This low-cost readiness audit is intentionally CPU-only")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    train = _load_cache(Path(args.train_cache), split="train")
    val = _load_cache(Path(args.val_cache), split="val")
    if train["embeddings"].shape[1] != val["embeddings"].shape[1]:
        raise ValueError("Train/validation embedding dimensions differ")
    if train["probabilities"].shape[1] != val["probabilities"].shape[1]:
        raise ValueError("Train/validation class counts differ")
    class_count = int(train["probabilities"].shape[1])
    focus = int(args.focus_class_index)
    if not 0 <= focus < class_count:
        raise ValueError(f"focus_class_index={focus} is outside 0..{class_count - 1}")
    if np.any(train["labels"] < 0) or np.any(train["labels"] >= class_count):
        raise ValueError("Train labels are outside the class range")
    if np.any(val["labels"] < 0) or np.any(val["labels"] >= class_count):
        raise ValueError("Validation labels are outside the class range")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    folds = list(
        StratifiedGroupKFold(
            n_splits=int(args.folds),
            shuffle=True,
            random_state=int(args.seed),
        ).split(
            np.zeros(len(train["labels"])),
            train["labels"],
            groups=train["source_stems"],
        )
    )

    train_rows = len(train["labels"])
    train_control = np.zeros_like(train["probabilities"], dtype=np.float32)
    train_candidate = np.zeros_like(train["probabilities"], dtype=np.float32)
    train_control_uncertainty = np.zeros(train_rows, dtype=np.float32)
    train_candidate_uncertainty = np.zeros(train_rows, dtype=np.float32)
    fold_assignment = np.full(train_rows, -1, dtype=np.int64)
    fold_rows: list[Dict[str, object]] = []
    training_curve: list[Dict[str, object]] = []

    for fold_index, (fit_indices, hold_indices) in enumerate(folds):
        fit_groups = set(train["source_stems"][fit_indices].tolist())
        hold_groups = set(train["source_stems"][hold_indices].tolist())
        overlap = len(fit_groups.intersection(hold_groups))
        if overlap:
            raise RuntimeError(f"Source leakage in fold {fold_index}: {overlap}")
        mean, scale = _fit_scaler(train["embeddings"][fit_indices])
        fit_features = _apply_scaler(train["embeddings"][fit_indices], mean, scale)
        hold_features = _apply_scaler(train["embeddings"][hold_indices], mean, scale)
        stage1 = KeeperResidualReadout(train["embeddings"].shape[1], class_count).to(device)
        training_curve.extend(
            _train_stage(
                stage1,
                features=fit_features,
                base_probabilities=train["probabilities"][fit_indices],
                labels=train["labels"][fit_indices],
                mode="ce",
                epochs=int(args.stage1_epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                prior_weight=float(args.prior_weight),
                seed=int(args.seed) + fold_index * 1000,
                device=device,
                curve_prefix={"fit_scope": "oof", "fold": int(fold_index), "stage": 1},
            )
        )
        control = deepcopy(stage1)
        candidate = deepcopy(stage1)
        stage2_seed = int(args.seed) + fold_index * 1000 + 500
        training_curve.extend(
            _train_stage(
                control,
                features=fit_features,
                base_probabilities=train["probabilities"][fit_indices],
                labels=train["labels"][fit_indices],
                mode="ce",
                epochs=int(args.stage2_epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                prior_weight=float(args.prior_weight),
                seed=stage2_seed,
                device=device,
                curve_prefix={"fit_scope": "oof", "fold": int(fold_index), "stage": 2},
            )
        )
        training_curve.extend(
            _train_stage(
                candidate,
                features=fit_features,
                base_probabilities=train["probabilities"][fit_indices],
                labels=train["labels"][fit_indices],
                mode="reedl",
                epochs=int(args.stage2_epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                prior_weight=float(args.prior_weight),
                seed=stage2_seed,
                device=device,
                curve_prefix={"fit_scope": "oof", "fold": int(fold_index), "stage": 2},
            )
        )
        control_probabilities, control_uncertainty, _ = _predict(
            control,
            features=hold_features,
            base_probabilities=train["probabilities"][hold_indices],
            mode="ce",
            prior_weight=float(args.prior_weight),
            device=device,
        )
        candidate_probabilities, candidate_uncertainty, _ = _predict(
            candidate,
            features=hold_features,
            base_probabilities=train["probabilities"][hold_indices],
            mode="reedl",
            prior_weight=float(args.prior_weight),
            device=device,
        )
        train_control[hold_indices] = control_probabilities
        train_candidate[hold_indices] = candidate_probabilities
        train_control_uncertainty[hold_indices] = control_uncertainty
        train_candidate_uncertainty[hold_indices] = candidate_uncertainty
        fold_assignment[hold_indices] = fold_index
        control_metrics = _classification_metrics(train["labels"][hold_indices], control_probabilities)
        candidate_metrics = _classification_metrics(train["labels"][hold_indices], candidate_probabilities)
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_groups": int(len(fit_groups)),
                "hold_groups": int(len(hold_groups)),
                "source_overlap": int(overlap),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "control_focus_f1": float(control_metrics["per_class"][focus]["f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "candidate_focus_f1": float(candidate_metrics["per_class"][focus]["f1"]),
                "candidate_minus_control_macro_f1": float(
                    candidate_metrics["macro_f1"] - control_metrics["macro_f1"]
                ),
                "candidate_minus_control_focus_f1": float(
                    candidate_metrics["per_class"][focus]["f1"]
                    - control_metrics["per_class"][focus]["f1"]
                ),
            }
        )

    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF prediction coverage is incomplete")

    mean, scale = _fit_scaler(train["embeddings"])
    fit_features = _apply_scaler(train["embeddings"], mean, scale)
    val_features = _apply_scaler(val["embeddings"], mean, scale)
    stage1 = KeeperResidualReadout(train["embeddings"].shape[1], class_count).to(device)
    training_curve.extend(
        _train_stage(
            stage1,
            features=fit_features,
            base_probabilities=train["probabilities"],
            labels=train["labels"],
            mode="ce",
            epochs=int(args.stage1_epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            prior_weight=float(args.prior_weight),
            seed=int(args.seed) + 9000,
            device=device,
            curve_prefix={"fit_scope": "full_train", "fold": -1, "stage": 1},
        )
    )
    control = deepcopy(stage1)
    candidate = deepcopy(stage1)
    stage2_seed = int(args.seed) + 9500
    training_curve.extend(
        _train_stage(
            control,
            features=fit_features,
            base_probabilities=train["probabilities"],
            labels=train["labels"],
            mode="ce",
            epochs=int(args.stage2_epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            prior_weight=float(args.prior_weight),
            seed=stage2_seed,
            device=device,
            curve_prefix={"fit_scope": "full_train", "fold": -1, "stage": 2},
        )
    )
    training_curve.extend(
        _train_stage(
            candidate,
            features=fit_features,
            base_probabilities=train["probabilities"],
            labels=train["labels"],
            mode="reedl",
            epochs=int(args.stage2_epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            prior_weight=float(args.prior_weight),
            seed=stage2_seed,
            device=device,
            curve_prefix={"fit_scope": "full_train", "fold": -1, "stage": 2},
        )
    )
    val_control, val_control_uncertainty, _ = _predict(
        control,
        features=val_features,
        base_probabilities=val["probabilities"],
        mode="ce",
        prior_weight=float(args.prior_weight),
        device=device,
    )
    val_candidate, val_candidate_uncertainty, _ = _predict(
        candidate,
        features=val_features,
        base_probabilities=val["probabilities"],
        mode="reedl",
        prior_weight=float(args.prior_weight),
        device=device,
    )

    train_summary, train_risk_rows, train_calibration_rows = _build_split_summary(
        split="train_oof",
        targets=train["labels"],
        keeper_probabilities=train["probabilities"],
        control_probabilities=train_control,
        candidate_probabilities=train_candidate,
        control_uncertainty=train_control_uncertainty,
        candidate_uncertainty=train_candidate_uncertainty,
        focus_class_index=focus,
    )
    val_summary, val_risk_rows, val_calibration_rows = _build_split_summary(
        split="val",
        targets=val["labels"],
        keeper_probabilities=val["probabilities"],
        control_probabilities=val_control,
        candidate_probabilities=val_candidate,
        control_uncertainty=val_control_uncertainty,
        candidate_uncertainty=val_candidate_uncertainty,
        focus_class_index=focus,
    )
    matched_source_folds = bool(
        len(fold_rows) == int(args.folds)
        and all(int(row["source_overlap"]) == 0 for row in fold_rows)
    )
    gate = assess_two_stage_reedl_readiness(
        train_summary=train_summary,
        val_summary=val_summary,
        matched_source_folds=matched_source_folds,
        focus_class_index=focus,
    )

    risk_rows = [*train_risk_rows, *val_risk_rows]
    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(output_dir / "training_curve.csv", training_curve)
    _write_csv(output_dir / "risk_coverage.csv", risk_rows)
    _write_csv(
        output_dir / "calibration.csv",
        [*train_calibration_rows, *val_calibration_rows],
    )
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            cache=train,
            fold_assignment=fold_assignment,
            control_probabilities=train_control,
            candidate_probabilities=train_candidate,
            control_uncertainty=train_control_uncertainty,
            candidate_uncertainty=train_candidate_uncertainty,
            focus_class_index=focus,
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            cache=val,
            fold_assignment=np.full(len(val["labels"]), -1, dtype=np.int64),
            control_probabilities=val_control,
            candidate_probabilities=val_candidate,
            control_uncertainty=val_control_uncertainty,
            candidate_uncertainty=val_candidate_uncertainty,
            focus_class_index=focus,
        ),
    )
    _plot_risk_coverage(risk_rows, output_dir / "risk_coverage.png")
    _plot_uncertainty(
        train_targets=train["labels"],
        train_probabilities=train_candidate,
        train_uncertainty=train_candidate_uncertainty,
        val_targets=val["labels"],
        val_probabilities=val_candidate,
        val_uncertainty=val_candidate_uncertainty,
        path=output_dir / "uncertainty_histogram.png",
    )

    summary: Dict[str, object] = {
        "mode": "two_stage_reedl_keeper_residual_readiness",
        "output_dir": str(output_dir.resolve()),
        "protocol_locked": _protocol_is_locked(args),
        "protocol": {
            "seed": int(args.seed),
            "folds": int(args.folds),
            "focus_class_index": focus,
            "stage1": {
                "loss": "cross_entropy",
                "epochs": int(args.stage1_epochs),
            },
            "stage2_control": {
                "loss": "cross_entropy",
                "epochs": int(args.stage2_epochs),
            },
            "stage2_candidate": {
                "loss": "Re-EDL projected-Dirichlet cross_entropy",
                "epochs": int(args.stage2_epochs),
                "evidence": "softplus(logits)",
                "alpha": f"evidence + {float(args.prior_weight):g}",
                "variance_term": False,
                "kl_regularizer": False,
            },
            "total_epochs_per_branch": int(args.stage1_epochs) + int(args.stage2_epochs),
            "readout": "zero-initialized linear residual over keeper log probabilities",
            "feature_scaling": "fit-split-only standardization",
            "sampling": "natural frequency; no oversampling or class weights",
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "device": str(device),
            "matched_source_folds": matched_source_folds,
        },
        "inputs": {
            "train_cache": str(Path(args.train_cache).resolve()),
            "val_cache": str(Path(args.val_cache).resolve()),
            "train_rows": int(len(train["labels"])),
            "val_rows": int(len(val["labels"])),
            "train_source_groups": int(np.unique(train["source_stems"]).size),
            "val_source_groups": int(np.unique(val["source_stems"]).size),
            "train_val_source_overlap": int(
                len(
                    set(train["source_stems"].tolist()).intersection(
                        set(val["source_stems"].tolist())
                    )
                )
            ),
            "feature_dim": int(train["embeddings"].shape[1]),
            "class_count": class_count,
        },
        "train_oof": train_summary,
        "validation": val_summary,
        "folds": fold_rows,
        "gate": gate,
        "research_sources": [
            {
                "name": "TEDL",
                "url": "https://arxiv.org/abs/2209.05522",
                "use": "CE first stage followed by evidential second stage for stability",
            },
            {
                "name": "Re-EDL",
                "url": "https://arxiv.org/abs/2410.00393",
                "use": "optimize projected Dirichlet mean without variance or KL terms",
            },
            {
                "name": "Official Re-EDL implementation",
                "url": "https://github.com/MengyuanChen21/Re-EDL/blob/main/code_classical/models/ModifiedEvidentialN.py",
                "use": "softplus evidence and alpha=evidence+prior implementation reference",
            },
        ],
        "guardrails": {
            "test_read": False,
            "raw_data_modified": False,
            "trainable_manifest_written": False,
            "checkpoint_written": False,
            "model_binary_written": False,
            "validation_used_for_hyperparameter_selection": False,
            "full_train_permission": False,
        },
        "limitations": [
            "The cached keeper representation and base probabilities are in-sample on train; grouped OOF applies only to the residual readout.",
            "This is a frozen-representation linear-head readiness audit, not an image-model reproduction of TEDL or Re-EDL.",
            "Closed-set validation risk is audited; no OOD claim is made.",
        ],
        "seconds": float(time.perf_counter() - started),
        "conclusion": (
            "Readiness gate passed; one checkpoint-preserving image-model smoke may be designed."
            if bool(gate["smoke_permission"])
            else "Readiness gate failed; do not implement or train a generic evidential head on the current keeper representation."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Two-stage Re-EDL keeper-residual readiness\n\n"
        "Train/validation-only, source-grouped frozen-cache audit. The candidate and "
        "CE control share the same zero-residual initialization, first-stage state, "
        "optimizer settings, batch order, and 30-epoch budget. Validation does not "
        "select hyperparameters. No test row, raw-data edit, checkpoint, model binary, "
        "or trainable manifest is produced.\n\n"
        f"Smoke permission: `{bool(gate['smoke_permission'])}`.\n\n"
        f"Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(
        output_dir,
        mode="two_stage_reedl_keeper_residual_readiness_evidence_manifest",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_audit(parse_args(argv))
    focus = int(summary["protocol"]["focus_class_index"])
    compact = {
        "output_dir": str(summary["output_dir"]),
        "protocol_locked": bool(summary["protocol_locked"]),
        "oof": {
            model: {
                "macro_f1": summary["train_oof"]["models"][model]["metrics"]["macro_f1"],
                "focus_f1": summary["train_oof"]["models"][model]["metrics"]["per_class"][
                    focus
                ]["f1"],
            }
            for model in ("keeper", "ce_control", "two_stage_reedl")
        },
        "validation": {
            model: {
                "macro_f1": summary["validation"]["models"][model]["metrics"]["macro_f1"],
                "focus_f1": summary["validation"]["models"][model]["metrics"]["per_class"][
                    focus
                ]["f1"],
            }
            for model in ("keeper", "ce_control", "two_stage_reedl")
        },
        "smoke_permission": bool(summary["gate"]["smoke_permission"]),
        "failed_checks": summary["gate"]["failed_checks"],
        "seconds": summary["seconds"],
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

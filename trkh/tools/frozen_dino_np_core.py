from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


TOKEN_WIDTH = 384
PREFIX_TOKENS = 5
REGISTER_TOKENS = 4
PATCH_TOKENS = 256
DESCRIPTOR_WIDTH = 3 * TOKEN_WIDTH
VARIANCE_FLOOR = 1e-4
RISK_ALPHA = 0.008
RISK_DELTA = 0.05
MIN_CALIBRATION_GROUPS = 373


def frozen_prefix_descriptor(tokens: Tensor) -> Tensor:
    """Return the locked parameter-free 1,152-D B8 descriptor."""

    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
        PREFIX_TOKENS + PATCH_TOKENS,
        TOKEN_WIDTH,
    ):
        raise ValueError(
            "tokens must have shape [B,261,384], got "
            f"{tuple(tokens.shape)}"
        )
    values = tokens.float()
    prefix = values[:, :PREFIX_TOKENS]
    patch_mean = values[:, PREFIX_TOKENS:].mean(dim=1)
    residual = prefix - patch_mean.unsqueeze(1)
    # torch.layer_norm is non-affine here and uses the protocol-locked eps.
    normalized = torch.nn.functional.layer_norm(
        residual,
        normalized_shape=(TOKEN_WIDTH,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    cls = normalized[:, 0]
    registers = normalized[:, 1:]
    register_mean = registers.mean(dim=1)
    register_variance = (
        registers - register_mean.unsqueeze(1)
    ).square().mean(dim=1)
    descriptor = torch.cat((cls, register_mean, register_variance), dim=1)
    if tuple(descriptor.shape) != (int(tokens.size(0)), DESCRIPTOR_WIDTH):
        raise RuntimeError("B8 descriptor shape contract changed")
    if not bool(torch.isfinite(descriptor).all()):
        raise FloatingPointError("B8 descriptor contains non-finite values")
    return descriptor


def source_balanced_diagonal_gaussian(
    descriptors: np.ndarray,
    source_groups: Sequence[object],
    *,
    variance_floor: float = VARIANCE_FLOOR,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Fit group-balanced diagonal moments without using calibration rows."""

    values = np.asarray(descriptors, dtype=np.float64)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    if values.ndim != 2 or values.shape[1] != DESCRIPTOR_WIDTH:
        raise ValueError("descriptors must have shape [N,1152]")
    if values.shape[0] != groups.size or values.shape[0] == 0:
        raise ValueError("descriptors/source_groups are empty or misaligned")
    if not bool(np.isfinite(values).all()):
        raise ValueError("descriptors contain non-finite values")
    floor = float(variance_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    unique, inverse, counts = np.unique(
        groups.astype(str), return_inverse=True, return_counts=True
    )
    weights = 1.0 / (float(unique.size) * counts[inverse].astype(np.float64))
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("source-balanced weights do not sum to one")
    mean = np.sum(values * weights[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * weights[:, None], axis=0)
    variance = np.maximum(variance, floor)
    if not (np.isfinite(mean).all() and np.isfinite(variance).all()):
        raise FloatingPointError("B8 diagonal moments are non-finite")
    return (
        mean.astype(np.float32),
        variance.astype(np.float32),
        {
            "rows": int(values.shape[0]),
            "source_groups": int(unique.size),
            "variance_floor": floor,
            "minimum_variance": float(variance.min()),
            "maximum_variance": float(variance.max()),
            "maximum_group_weight_error": float(
                max(
                    abs(float(weights[inverse == index].sum()) - 1.0 / unique.size)
                    for index in range(unique.size)
                )
            ),
        },
    )


def class2_conformity_score(
    descriptors: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """Higher score means closer to the locked class-2 diagonal Gaussian."""

    values = np.asarray(descriptors, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(variance, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != DESCRIPTOR_WIDTH:
        raise ValueError("descriptors must have shape [N,1152]")
    if center.size != DESCRIPTOR_WIDTH or scale.size != DESCRIPTOR_WIDTH:
        raise ValueError("mean/variance must each contain 1,152 values")
    if not (
        np.isfinite(values).all()
        and np.isfinite(center).all()
        and np.isfinite(scale).all()
        and bool((scale > 0.0).all())
    ):
        raise ValueError("score inputs must be finite with positive variance")
    score = -np.mean((values - center[None, :]) ** 2 / scale[None, :], axis=1)
    if not bool(np.isfinite(score).all()):
        raise FloatingPointError("B8 conformity score is non-finite")
    return score.astype(np.float64, copy=False)


def maximum_group_threshold(
    scores: np.ndarray,
    source_groups: Sequence[object],
) -> Tuple[float, Dict[str, object]]:
    """Calibrate the strict-veto threshold from source-group maxima."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1).astype(str)
    if values.size != groups.size or values.size == 0:
        raise ValueError("scores/source_groups are empty or misaligned")
    if not bool(np.isfinite(values).all()):
        raise ValueError("calibration scores contain non-finite values")
    unique = np.unique(groups)
    maxima = np.asarray(
        [values[groups == group].max() for group in unique], dtype=np.float64
    )
    threshold = float(maxima.max())
    exceedances = int(np.sum(maxima > threshold))
    return threshold, {
        "calibration_rows": int(values.size),
        "calibration_source_groups": int(unique.size),
        "calibration_group_exceedances": exceedances,
        "threshold": threshold,
        "strict_comparison": "score > threshold",
    }


def maximum_order_tolerance_upper(
    calibration_groups: int,
    *,
    delta: float = RISK_DELTA,
) -> float:
    """One-sided tolerance upper bound for a maximum order statistic."""

    count = int(calibration_groups)
    failure_probability = float(delta)
    if count <= 0 or not 0.0 < failure_probability < 1.0:
        raise ValueError("invalid calibration_groups or delta")
    return float(1.0 - failure_probability ** (1.0 / float(count)))


def selective_2to1_predictions(
    b2_logits: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply only the locked B2 top1=1, runner-up=2, strict-score veto."""

    logits = np.asarray(b2_logits, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if logits.ndim != 2 or logits.shape[1] != 5 or logits.shape[0] != values.size:
        raise ValueError("b2_logits must be [N,5] and align with scores")
    if not (np.isfinite(logits).all() and np.isfinite(values).all()):
        raise ValueError("routing inputs contain non-finite values")
    base = np.argmax(logits, axis=1).astype(np.int64)
    non_focus = logits.copy()
    non_focus[:, 1] = -np.inf
    runner = np.argmax(non_focus, axis=1).astype(np.int64)
    action = np.logical_and.reduce(
        (base == 1, runner == 2, values > float(threshold))
    )
    corrected = base.copy()
    corrected[action] = 2
    if bool(np.any(corrected[~action] != base[~action])):
        raise RuntimeError("B8 changed a prediction outside its action mask")
    return base, runner, action, corrected


def readiness_from_metrics(
    *,
    aligned: Mapping[str, object],
    controls: Mapping[str, Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
    integrity_checks: Mapping[str, bool],
) -> Dict[str, object]:
    """Evaluate only the pass/fail conditions committed in the B8 protocol."""

    checks: Dict[str, bool] = {
        f"integrity:{key}": bool(value) for key, value in integrity_checks.items()
    }
    checks.update(
        {
            "every_fold_recall_protected": all(
                float(row["aligned_class1_recall_delta"]) >= -0.005
                for row in fold_rows
            ),
            "aggregate_tp_retention": float(aligned["class1_tp_retention"]) >= 0.99,
            "minimum_2to1_corrections": int(aligned["corrected_2to1"]) >= 14,
            "positive_effect_folds": sum(
                int(row["aligned_corrected_2to1"]) > 0 for row in fold_rows
            )
            >= 4,
            "no_new_0to1_or_4to1": int(aligned["new_0to1_or_4to1"]) == 0,
            "action_domain_exact": int(aligned["outside_domain_changes"]) == 0,
        }
    )
    aligned_reduction = float(aligned["fp_2to1_reduction"])
    for name, control in controls.items():
        checks[f"aligned_exceeds_{name}_aggregate"] = (
            aligned_reduction - float(control["fp_2to1_reduction"]) >= 0.05
        )
        checks[f"aligned_exceeds_{name}_folds"] = sum(
            float(row["aligned_fp_2to1_reduction"])
            > float(row[f"{name}_fp_2to1_reduction"])
            for row in fold_rows
        ) >= 4
    failed = [key for key, value in checks.items() if not value]
    return {
        "frozen_dino_np_2to1_ready": not failed,
        "implementation_permission": not failed,
        "validation_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": failed,
    }

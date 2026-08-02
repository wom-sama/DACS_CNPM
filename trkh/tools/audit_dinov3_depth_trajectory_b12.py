from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from trkh.models.model import build_model_from_checkpoint  # noqa: E402
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_DATA_SHA256,
    EXPECTED_TRAIN_SAMPLES,
    _data_root_from_yaml,
    _relative_train_path,
    _sha256,
    _validate_b9_checkpoint,
    assert_train_only_paths,
)
from trkh.tools.probe_embedding_prototypes import (  # noqa: E402
    _build_dataset,
    _collate_classification,
    _resolve_device,
)
from trkh.tools.screen_dinov3_cgaer_b11_sourcefold import (  # noqa: E402
    read_locked_assignment,
    validate_a0_cache,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B12_DEPTH_TRAJECTORY_SIGNAL_20260802_R2"
SEED = 20260802
FOLDS = 5
LAYERS = (2, 5, 8, 11)
RIVALS = (0, 2, 4)
PREFIX_TOKENS = 5
PATCH_GRID = 16
PATCH_COUNT = PATCH_GRID * PATCH_GRID
EMBED_DIM = 384
FEATURES_PER_MAP = 5
MAPS_PER_PAIR = 2
FEATURES_PER_PAIR = FEATURES_PER_MAP * MAPS_PER_PAIR
FEATURES_PER_ARM = FEATURES_PER_PAIR * len(RIVALS)
LOGISTIC_INPUTS = FEATURES_PER_PAIR + 1
LOGISTIC_C = 1.0
LOGISTIC_TOL = 1e-9
LOGISTIC_MAX_ITER = 2_000
BOOTSTRAP_REPLICATES = 5_000
PARITY_ATOL = 1e-6
# Match the immutable A0 cache extraction batch so the locked cache-logit
# parity check is bit-exact instead of depending on CUDA GEMM batch shape.
DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked TRAIN-only B12 same-patch depth-trajectory signal gate. "
            "Validation and test datasets are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refuse to overwrite artifact: {path}")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refuse to overwrite artifact: {path}")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    digest = _sha256(partial)
    partial.replace(path)
    if _sha256(path) != digest:
        raise RuntimeError("atomic NPZ promotion changed its digest")
    return digest


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_paths() -> Dict[str, Path]:
    root = _repository_root()
    return {
        "runner": Path(__file__).resolve(),
        "protocol": root
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B12_DEPTH_TRAJECTORY_SIGNAL_PROTOCOL_20260802.md",
        "model_builder": root / "trkh" / "models" / "model.py",
        "a0_precheck": root
        / "trkh"
        / "tools"
        / "precheck_dinov3_xcnorm_a0_sourcefold.py",
        "b11_cache_guard": root
        / "trkh"
        / "tools"
        / "screen_dinov3_cgaer_b11_sourcefold.py",
        "dataset_builder": root
        / "trkh"
        / "tools"
        / "probe_embedding_prototypes.py",
    }


def _source_hashes() -> Dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"B12 bound sources are missing: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def _dependency_contract() -> Dict[str, object]:
    module_files: Dict[str, str] = {}
    for name in ("timm.models.eva", "sklearn.linear_model._logistic"):
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        if origin is None or not origin.is_file():
            raise RuntimeError(f"B12 cannot bind dependency source: {name}")
        module_files[name] = _sha256(origin)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": importlib.metadata.version("numpy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "timm": importlib.metadata.version("timm"),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda or ""),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "module_source_sha256": module_files,
    }


def _device_contract(requested: str) -> Dict[str, object]:
    device = _resolve_device(requested)
    payload: Dict[str, object] = {
        "requested": str(requested),
        "resolved": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return payload


def _git_contract() -> Dict[str, object]:
    root = _repository_root()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"head": head, "tracked_worktree_clean": not bool(dirty), "dirty": dirty}


def _linear_quantiles(sorted_maps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    if sorted_maps.size(-1) != PATCH_COUNT:
        raise ValueError("B12 quantiles require exactly 256 aligned patches")
    # (256 - 1) * (0.1, 0.5, 0.9) = (25.5, 127.5, 229.5).
    return (
        0.5 * (sorted_maps[..., 25] + sorted_maps[..., 26]),
        0.5 * (sorted_maps[..., 127] + sorted_maps[..., 128]),
        0.5 * (sorted_maps[..., 229] + sorted_maps[..., 230]),
    )


def summarize_pair_maps(maps: Tensor) -> Tensor:
    """Summarize maps [B,2,3,256] as locked [B,30] FP32 features."""

    if maps.ndim != 4 or tuple(maps.shape[1:]) != (
        MAPS_PER_PAIR,
        len(RIVALS),
        PATCH_COUNT,
    ):
        raise ValueError(f"unexpected B12 map shape: {tuple(maps.shape)}")
    values = maps.float()
    ordered = values.sort(dim=-1).values
    q10, q50, q90 = _linear_quantiles(ordered)
    summary = torch.stack(
        (
            values.mean(dim=-1),
            values.std(dim=-1, unbiased=False),
            q10,
            q50,
            q90,
        ),
        dim=-1,
    )
    flattened = summary.reshape(values.size(0), FEATURES_PER_ARM)
    if not bool(torch.isfinite(flattened).all()):
        raise FloatingPointError("B12 summaries contain non-finite values")
    return flattened


def depth_trajectory_features(
    x2: Tensor,
    x5: Tensor,
    x8: Tensor,
    directions: Tensor,
) -> Dict[str, Tensor]:
    """Build aligned, latest-only, and correspondence-deranged feature arms."""

    states = (x2, x5, x8)
    if any(
        state.ndim != 3 or tuple(state.shape[1:]) != (PATCH_COUNT, EMBED_DIM)
        for state in states
    ):
        raise ValueError("B12 states must each be [B,256,384]")
    if len({int(state.size(0)) for state in states}) != 1:
        raise ValueError("B12 state batches are not aligned")
    if tuple(directions.shape) != (len(RIVALS), EMBED_DIM):
        raise ValueError("B12 classifier directions must be [3,384]")
    typed = tuple(state.float() for state in states)
    direction = directions.float()

    candidate_delta = torch.stack((typed[1] - typed[0], typed[2] - typed[1]), dim=1)
    candidate_maps = torch.einsum("btpd,rd->btrp", candidate_delta, direction)

    last_grid = typed[2].reshape(-1, PATCH_GRID, PATCH_GRID, EMBED_DIM)
    neighbours = 0.25 * (
        torch.roll(last_grid, 1, dims=1)
        + torch.roll(last_grid, -1, dims=1)
        + torch.roll(last_grid, 1, dims=2)
        + torch.roll(last_grid, -1, dims=2)
    )
    latest_maps = torch.stack(
        (
            torch.einsum("bpd,rd->brp", typed[2], direction),
            torch.einsum(
                "bpd,rd->brp",
                (last_grid - neighbours).reshape(-1, PATCH_COUNT, EMBED_DIM),
                direction,
            ),
        ),
        dim=1,
    )

    x2_roll = torch.roll(
        typed[0].reshape(-1, PATCH_GRID, PATCH_GRID, EMBED_DIM), 8, dims=2
    ).reshape(-1, PATCH_COUNT, EMBED_DIM)
    x5_roll = torch.roll(
        typed[1].reshape(-1, PATCH_GRID, PATCH_GRID, EMBED_DIM), 8, dims=1
    ).reshape(-1, PATCH_COUNT, EMBED_DIM)
    deranged_delta = torch.stack((typed[1] - x2_roll, typed[2] - x5_roll), dim=1)
    deranged_maps = torch.einsum("btpd,rd->btrp", deranged_delta, direction)

    return {
        "candidate": summarize_pair_maps(candidate_maps),
        "latest": summarize_pair_maps(latest_maps),
        "deranged": summarize_pair_maps(deranged_maps),
    }


def pair_features(features: np.ndarray, rival_position: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURES_PER_ARM:
        raise ValueError("B12 arm features must be [N,30]")
    position = int(rival_position)
    if position < 0 or position >= len(RIVALS):
        raise ValueError("invalid B12 rival position")
    shaped = values.reshape(values.shape[0], MAPS_PER_PAIR, len(RIVALS), FEATURES_PER_MAP)
    return shaped[:, :, position, :].reshape(values.shape[0], FEATURES_PER_PAIR)


def _pair_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("pair metrics require aligned binary labels and scores")
    if not np.isfinite(values).all():
        raise FloatingPointError("pair scores contain non-finite values")
    predicted = values >= 0.0
    positive = y == 1
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1e-15, precision + recall))
    return {
        "auroc": float(roc_auc_score(y, values)),
        "class1_precision": precision,
        "class1_recall": recall,
        "class1_f1": f1,
        "class1_tp": tp,
        "class1_fn": fn,
        "rival_fp": fp,
        "samples": int(y.size),
    }


def _standardize_fit_apply(
    fit: np.ndarray, held: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(fit, dtype=np.float64).mean(axis=0)
    std = np.asarray(fit, dtype=np.float64).std(axis=0)
    scale = np.maximum(std, 1e-8)
    return (fit - mean) / scale, (held - mean) / scale, mean, scale


def fit_oof_arm(
    *,
    features: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    base_logits = np.asarray(base_logits, dtype=np.float64)
    if labels.shape != folds.shape or base_logits.shape != (labels.size, 5):
        raise ValueError("B12 OOF arrays are misaligned")
    scores = np.full((labels.size, len(RIVALS)), np.nan, dtype=np.float64)
    fold_models: List[Dict[str, object]] = []
    converged = True
    for rival_position, rival in enumerate(RIVALS):
        pair_mask = (labels == 1) | (labels == rival)
        arm = pair_features(features, rival_position)
        margin = (base_logits[:, 1] - base_logits[:, rival]).reshape(-1, 1)
        design = np.concatenate((arm, margin), axis=1)
        if design.shape[1] != LOGISTIC_INPUTS:
            raise RuntimeError("B12 logistic input width drifted")
        binary = (labels == 1).astype(np.int64)
        for fold in range(FOLDS):
            fit_mask = pair_mask & (folds != fold)
            held_mask = pair_mask & (folds == fold)
            fit_x, held_x, mean, scale = _standardize_fit_apply(
                design[fit_mask], design[held_mask]
            )
            fit_y = binary[fit_mask]
            counts = np.bincount(fit_y, minlength=2).astype(np.float64)
            if bool((counts <= 0).any()):
                raise ValueError("B12 fit pair lost one class")
            sample_weight = np.power(counts[fit_y], -0.5)
            sample_weight /= sample_weight.mean()
            classifier = LogisticRegression(
                C=LOGISTIC_C,
                solver="lbfgs",
                tol=LOGISTIC_TOL,
                max_iter=LOGISTIC_MAX_ITER,
                fit_intercept=True,
                random_state=SEED,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(fit_x, fit_y, sample_weight=sample_weight)
            warning_messages = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            iteration = int(np.asarray(classifier.n_iter_).max())
            this_converged = not warning_messages and iteration < LOGISTIC_MAX_ITER
            converged = converged and this_converged
            scores[held_mask, rival_position] = classifier.decision_function(held_x)
            fold_models.append(
                {
                    "rival": int(rival),
                    "fold": int(fold),
                    "fit_samples": int(fit_mask.sum()),
                    "held_samples": int(held_mask.sum()),
                    "iterations": iteration,
                    "converged": bool(this_converged),
                    "warnings": warning_messages,
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                    "coef": np.asarray(classifier.coef_, dtype=np.float64).reshape(-1).tolist(),
                    "intercept": float(np.asarray(classifier.intercept_).reshape(-1)[0]),
                }
            )
    for rival_position, rival in enumerate(RIVALS):
        expected = (labels == 1) | (labels == rival)
        if not np.isfinite(scores[expected, rival_position]).all() or not np.isnan(
            scores[~expected, rival_position]
        ).all():
            raise RuntimeError("B12 pair OOF coverage is incomplete")
    return {"scores": scores, "fold_models": fold_models, "converged": converged}


def summarize_oof_scores(
    labels: np.ndarray, folds: np.ndarray, scores: np.ndarray
) -> Dict[str, object]:
    pair_rows: Dict[str, object] = {}
    fold_rows: List[Dict[str, object]] = []
    for rival_position, rival in enumerate(RIVALS):
        mask = (labels == 1) | (labels == rival)
        y = (labels[mask] == 1).astype(np.int64)
        pair_rows[str(rival)] = _pair_metrics(y, scores[mask, rival_position])
        for fold in range(FOLDS):
            held = mask & (folds == fold)
            fold_y = (labels[held] == 1).astype(np.int64)
            metrics = _pair_metrics(fold_y, scores[held, rival_position])
            fold_rows.append({"fold": fold, "rival": rival, **metrics})
    return {
        "pairs": pair_rows,
        "fold_pairs": fold_rows,
        "mean_pair_auroc": float(np.mean([row["auroc"] for row in pair_rows.values()])),
        "aggregate_class1_tp": int(sum(row["class1_tp"] for row in pair_rows.values())),
        "aggregate_rival_fp": int(sum(row["rival_fp"] for row in pair_rows.values())),
    }


def _base_pair_scores(labels: np.ndarray, base_logits: np.ndarray) -> np.ndarray:
    scores = np.full((labels.size, len(RIVALS)), np.nan, dtype=np.float64)
    for position, rival in enumerate(RIVALS):
        mask = (labels == 1) | (labels == rival)
        scores[mask, position] = base_logits[mask, 1] - base_logits[mask, rival]
    return scores


def _fold_mean_aurocs(summary: Mapping[str, object]) -> np.ndarray:
    values = np.full(FOLDS, np.nan, dtype=np.float64)
    rows = summary.get("fold_pairs", [])
    if not isinstance(rows, list):
        raise ValueError("fold-pair summary is missing")
    for fold in range(FOLDS):
        selected = [float(row["auroc"]) for row in rows if int(row["fold"]) == fold]
        if len(selected) != len(RIVALS):
            raise ValueError("fold-pair summary is incomplete")
        values[fold] = np.mean(selected)
    return values


def _pair_fold_deltas(
    candidate: Mapping[str, object], latest: Mapping[str, object]
) -> List[Dict[str, float | int]]:
    latest_by_key = {
        (int(row["fold"]), int(row["rival"])): row
        for row in latest["fold_pairs"]  # type: ignore[index]
    }
    rows: List[Dict[str, float | int]] = []
    for row in candidate["fold_pairs"]:  # type: ignore[index]
        key = (int(row["fold"]), int(row["rival"]))
        rows.append(
            {
                "fold": key[0],
                "rival": key[1],
                "candidate_minus_latest_auroc": float(row["auroc"])
                - float(latest_by_key[key]["auroc"]),
            }
        )
    return rows


def _bootstrap_mean_auc_deltas(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    base_scores: np.ndarray,
    latest_scores: np.ndarray,
    deranged_scores: np.ndarray,
    candidate_scores: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> Dict[str, object]:
    if int(replicates) <= 0:
        raise ValueError("B12 bootstrap replicate count must be positive")
    members: Dict[int, np.ndarray] = {}
    fold_groups: Dict[int, List[int]] = {fold: [] for fold in range(FOLDS)}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        observed_folds = np.unique(folds[positions])
        if observed_folds.size != 1:
            raise ValueError("B12 bootstrap component crosses folds")
        members[int(group)] = positions
        fold_groups[int(observed_folds[0])].append(int(group))
    values = {
        name: np.empty(int(replicates), dtype=np.float64)
        for name in ("candidate_vs_latest", "candidate_vs_deranged", "candidate_vs_base")
    }
    rng = np.random.default_rng(int(seed))
    draw_hash = hashlib.sha256()
    arms = {
        "base": base_scores,
        "latest": latest_scores,
        "deranged": deranged_scores,
        "candidate": candidate_scores,
    }
    for replicate in range(int(replicates)):
        chunks: List[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draw_hash.update(np.asarray([replicate, fold], dtype="<i8").tobytes())
            draw_hash.update(draw.astype("<i8").tobytes())
            chunks.extend(members[available[int(index)]] for index in draw)
        selected = np.concatenate(chunks)
        mean_auc: Dict[str, float] = {}
        for name, arm_scores in arms.items():
            pair_auc: List[float] = []
            for position, rival in enumerate(RIVALS):
                pair = (labels[selected] == 1) | (labels[selected] == rival)
                y = (labels[selected][pair] == 1).astype(np.int64)
                pair_auc.append(float(roc_auc_score(y, arm_scores[selected][pair, position])))
            mean_auc[name] = float(np.mean(pair_auc))
        values["candidate_vs_latest"][replicate] = mean_auc["candidate"] - mean_auc["latest"]
        values["candidate_vs_deranged"][replicate] = mean_auc["candidate"] - mean_auc["deranged"]
        values["candidate_vs_base"][replicate] = mean_auc["candidate"] - mean_auc["base"]

    def interval(name: str) -> Dict[str, float]:
        sample = values[name]
        return {
            "lower": float(np.quantile(sample, 0.025)),
            "upper": float(np.quantile(sample, 0.975)),
        }

    return {
        "method": "paired_fold_stratified_union_component_percentile_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "draws_int64_sha256": draw_hash.hexdigest(),
        **{name: interval(name) for name in values},
    }


def assess_readiness(
    *,
    summaries: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, object],
    pair_fold_deltas: Sequence[Mapping[str, object]],
    integrity_ok: bool,
) -> Dict[str, object]:
    candidate = summaries["candidate"]
    latest = summaries["latest"]
    deranged = summaries["deranged"]
    base = summaries["base"]
    candidate_auc = float(candidate["mean_pair_auroc"])
    latest_auc = float(latest["mean_pair_auroc"])
    deranged_auc = float(deranged["mean_pair_auroc"])
    base_auc = float(base["mean_pair_auroc"])
    fold_delta = _fold_mean_aurocs(candidate) - _fold_mean_aurocs(latest)
    pair_delta = {
        rival: float(candidate["pairs"][str(rival)]["auroc"])  # type: ignore[index]
        - float(latest["pairs"][str(rival)]["auroc"])  # type: ignore[index]
        for rival in RIVALS
    }
    candidate_tp = int(candidate["aggregate_class1_tp"])
    candidate_fp = int(candidate["aggregate_rival_fp"])
    latest_tp = int(latest["aggregate_class1_tp"])
    base_tp = int(base["aggregate_class1_tp"])
    latest_fp = int(latest["aggregate_rival_fp"])
    checks = {
        "candidate_vs_latest_effect_and_lcb": candidate_auc - latest_auc >= 0.001
        and float(bootstrap["candidate_vs_latest"]["lower"]) > 0.0,  # type: ignore[index]
        "candidate_vs_deranged_effect_and_lcb": candidate_auc - deranged_auc >= 0.001
        and float(bootstrap["candidate_vs_deranged"]["lower"]) > 0.0,  # type: ignore[index]
        "fold_and_pairfold_wins": int(np.sum(fold_delta > 0.0)) >= 4
        and sum(float(row["candidate_minus_latest_auroc"]) > 0.0 for row in pair_fold_deltas)
        >= 10,
        "critical_pairs_and_no_harm": pair_delta[0] >= 0.0002
        and pair_delta[2] >= 0.0002
        and min(pair_delta.values()) >= -0.0002,
        "candidate_vs_base_effect_and_lcb": candidate_auc - base_auc >= 0.0002
        and float(bootstrap["candidate_vs_base"]["lower"]) > 0.0,  # type: ignore[index]
        "tp_retention_and_fp_control": candidate_tp >= 0.98 * base_tp
        and candidate_tp >= 0.98 * latest_tp
        and candidate_fp <= latest_fp,
        "integrity_complete_train_only": bool(integrity_ok),
    }
    passed = all(checks.values())
    return {
        "signal_gate_passed": passed,
        "exact_b12_signal_family_closed": not passed,
        "b12_model_preregistration_permission": passed,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "point_deltas": {
            "candidate_vs_latest_mean_pair_auroc": candidate_auc - latest_auc,
            "candidate_vs_deranged_mean_pair_auroc": candidate_auc - deranged_auc,
            "candidate_vs_base_mean_pair_auroc": candidate_auc - base_auc,
            "candidate_vs_latest_by_rival": {str(key): value for key, value in pair_delta.items()},
            "candidate_vs_latest_by_fold": fold_delta.tolist(),
        },
    }


def _prepare_inputs(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) != DEFAULT_BATCH_SIZE or int(args.workers) != DEFAULT_WORKERS:
        raise ValueError("B12 local extraction is locked to batch=32, workers=0")
    data_yaml = args.data.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if _sha256(data_yaml) != EXPECTED_DATA_SHA256:
        raise ValueError("B12 data YAML is not canonical class_f")
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("B12 checkpoint is not a mapping")
    checkpoint_contract = _validate_b9_checkpoint(checkpoint, checkpoint_sha)
    cache = validate_a0_cache(args.cache_dir)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    assignment = read_locked_assignment(args.assignment_csv, cache["paths"], labels)
    dataset, classes = _build_dataset(
        data_yaml=data_yaml,
        split="train",
        checkpoint=checkpoint,
        class_name_mode="raw",
        max_samples=0,
    )
    if tuple(classes) != EXPECTED_CLASSES or len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("B12 canonical train support/order changed")
    data_root = _data_root_from_yaml(data_yaml)
    dataset_paths = [Path(value) for value in dataset.sample_paths()]
    assert_train_only_paths(dataset_paths, data_root / "train")
    relative_paths = [_relative_train_path(path, data_root) for path in dataset_paths]
    if relative_paths != cache["paths"]:
        raise ValueError("B12 live dataset order differs from immutable A0 cache")
    if not np.array_equal(np.asarray(dataset.labels(), dtype=np.int64), labels):
        raise ValueError("B12 live labels differ from immutable A0 cache")
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("B12 output must remain outside immutable class_f")
    return {
        "data_yaml": data_yaml,
        "data_root": data_root,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "checkpoint_contract": checkpoint_contract,
        "cache": cache,
        "assignment": assignment,
        "dataset": dataset,
        "relative_paths": relative_paths,
        "output_dir": output_dir,
    }


def _preflight_payload(args: argparse.Namespace, prepared: Mapping[str, object]) -> Dict[str, object]:
    cache = prepared["cache"]
    assignment = prepared["assignment"]
    git = _git_contract()
    if not git["tracked_worktree_clean"]:
        raise RuntimeError(f"B12 preflight requires a clean tracked tree: {git['dirty']}")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_scope": "conditional_train_only_patch_depth_trajectory_signal_gate",
        "source_hashes": _source_hashes(),
        "dependencies": _dependency_contract(),
        "git": git,
        "checkpoint_contract": prepared["checkpoint_contract"],
        "cache_hashes": cache["hashes"],  # type: ignore[index]
        "assignment": {
            key: assignment[key]  # type: ignore[index]
            for key in (
                "csv_sha256",
                "assignment_int64_sha256",
                "path_fold_sha256",
                "group_vector_int64_sha256",
                "fold_rows",
            )
        },
        "representation": {
            "layers_zero_based": list(LAYERS),
            "rivals": list(RIVALS),
            "patch_grid": [PATCH_GRID, PATCH_GRID],
            "features_per_arm": FEATURES_PER_ARM,
            "features_per_pair_plus_base_margin": LOGISTIC_INPUTS,
            "candidate": "same_patch_deltas_2to5_and_5to8",
            "latest_control": "block8_level_and_circular_N4_highpass",
            "deranged_control": "half_grid_roll_x_then_roll_y",
        },
        "probe": {
            "seed": SEED,
            "folds": FOLDS,
            "solver": "lbfgs",
            "C": LOGISTIC_C,
            "tol": LOGISTIC_TOL,
            "max_iter": LOGISTIC_MAX_ITER,
            "tempered_class_power": 0.5,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "torch_threads": int(args.torch_threads),
            "device": _device_contract(args.device),
            "fp32": True,
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
        "validation_permission": False,
        "test_permission": False,
    }


def _validate_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if args.preflight_artifact is None:
        raise ValueError("formal B12 requires --preflight-artifact")
    path = args.preflight_artifact.expanduser().resolve()
    digest = _sha256(path)
    expected = str(args.preflight_sha256).strip().casefold()
    if len(expected) != 64 or digest != expected:
        raise ValueError("B12 preflight artifact SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("B12 preflight protocol mismatch")
    if payload.get("source_hashes") != _source_hashes():
        raise ValueError("B12 source changed after preflight")
    git = _git_contract()
    if not git["tracked_worktree_clean"] or git["head"] != payload.get("git", {}).get("head"):
        raise ValueError("B12 git state changed after preflight")
    probe = payload.get("probe", {})
    expected_probe = {
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "torch_threads": int(args.torch_threads),
    }
    if any(probe.get(key) != value for key, value in expected_probe.items()):
        raise ValueError("B12 runtime arguments differ from preflight")
    if payload.get("validation_split_used") or payload.get("test_split_used"):
        raise ValueError("B12 preflight is not TRAIN-only")
    return {"path": str(path), "sha256": digest, "payload": payload}


def _extract_features(
    *,
    model: nn.Module,
    dataset: object,
    base_logits: np.ndarray,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Linear) or tuple(head.weight.shape) != (5, EMBED_DIM):
        raise ValueError("B12 requires the locked linear B9 head [5,384]")
    direction = head.weight.detach().float()[1:2] - head.weight.detach().float()[list(RIVALS)]
    direction = torch.nn.functional.normalize(direction, dim=-1, eps=1e-12)
    arms = {
        name: np.full((EXPECTED_TRAIN_SAMPLES, FEATURES_PER_ARM), np.nan, dtype=np.float32)
        for name in ("candidate", "latest", "deranged")
    }
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(workers),
        collate_fn=_collate_classification,
    )
    cursor = 0
    max_parity = 0.0
    model.eval()
    with torch.inference_mode():
        for images, _targets, _metadata in tqdm(
            loader, desc="B12 TRAIN depth trajectory", dynamic_ncols=True
        ):
            images = images.to(device=device, dtype=torch.float32)
            intermediates = model.forward_intermediates(
                images,
                indices=list(LAYERS),
                norm=True,
                output_fmt="NLC",
                intermediates_only=True,
            )
            if not isinstance(intermediates, list) or len(intermediates) != len(LAYERS):
                raise TypeError("B12 did not receive four intermediate patch states")
            if any(tuple(value.shape[1:]) != (PATCH_COUNT, EMBED_DIM) for value in intermediates):
                raise ValueError("B12 intermediate patch geometry changed")
            final_logits = head(intermediates[-1].mean(dim=1)).float()
            end = cursor + int(images.size(0))
            reference = torch.from_numpy(
                np.asarray(base_logits[cursor:end], dtype=np.float32).copy()
            ).to(device=device)
            parity = float((final_logits - reference).abs().max().cpu())
            max_parity = max(max_parity, parity)
            if parity > PARITY_ATOL:
                raise RuntimeError(f"B12 block11/base parity failed: {parity}")
            batch_features = depth_trajectory_features(
                intermediates[0], intermediates[1], intermediates[2], direction
            )
            for name, values in batch_features.items():
                arms[name][cursor:end] = values.detach().cpu().numpy().astype(np.float32)
            cursor = end
    if cursor != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(f"B12 extraction row count changed: {cursor}")
    if any(not np.isfinite(values).all() for values in arms.values()):
        raise FloatingPointError("B12 extracted features are incomplete/non-finite")
    return arms, max_parity


def run(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) <= 0:
        raise ValueError("B12 torch thread count must be positive")
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    prepared = _prepare_inputs(args)
    output_dir = prepared["output_dir"]
    if args.preflight_only:
        payload = _preflight_payload(args, prepared)
        _atomic_json(output_dir / "preflight.json", payload)  # type: ignore[operator]
        return payload

    preflight = _validate_preflight(args)
    formal_contract = _preflight_payload(args, prepared)
    frozen = preflight["payload"]
    for key in (
        "source_hashes",
        "dependencies",
        "checkpoint_contract",
        "cache_hashes",
        "assignment",
        "representation",
        "probe",
    ):
        if formal_contract.get(key) != frozen.get(key):
            raise ValueError(f"B12 formal input drifted from preflight: {key}")
    checkpoint = prepared["checkpoint"]
    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("B12 B9 backbone/head did not freeze")
    device = _resolve_device(args.device)
    model.to(device=device, dtype=torch.float32)
    cache = prepared["cache"]
    labels = np.asarray(cache["labels"], dtype=np.int64)
    base_logits = np.asarray(cache["logits"], dtype=np.float32)
    assignment = prepared["assignment"]
    folds = np.asarray(assignment["folds"], dtype=np.int64)
    groups = np.asarray(assignment["groups"], dtype=np.int64)
    started = time.perf_counter()
    arms, parity = _extract_features(
        model=model,
        dataset=prepared["dataset"],
        base_logits=base_logits,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    feature_path = output_dir / "train_depth_trajectory_features.npz"  # type: ignore[operator]
    feature_sha = _atomic_npz(feature_path, **arms)

    fitted = {
        name: fit_oof_arm(
            features=features,
            base_logits=base_logits,
            labels=labels,
            folds=folds,
        )
        for name, features in arms.items()
    }
    base_scores = _base_pair_scores(labels, base_logits)
    summaries = {
        "base": summarize_oof_scores(labels, folds, base_scores),
        **{
            name: summarize_oof_scores(labels, folds, result["scores"])
            for name, result in fitted.items()
        },
    }
    pair_fold_deltas = _pair_fold_deltas(summaries["candidate"], summaries["latest"])
    oof_path = output_dir / "train_oof_pair_scores.npz"  # type: ignore[operator]
    oof_sha = _atomic_npz(
        oof_path,
        labels=labels.astype(np.int64),
        folds=folds.astype(np.int64),
        union_groups=groups.astype(np.int64),
        base_scores=base_scores.astype(np.float64),
        latest_scores=np.asarray(fitted["latest"]["scores"], dtype=np.float64),
        deranged_scores=np.asarray(fitted["deranged"]["scores"], dtype=np.float64),
        candidate_scores=np.asarray(fitted["candidate"]["scores"], dtype=np.float64),
    )
    bootstrap = _bootstrap_mean_auc_deltas(
        labels=labels,
        folds=folds,
        groups=groups,
        base_scores=base_scores,
        latest_scores=np.asarray(fitted["latest"]["scores"], dtype=np.float64),
        deranged_scores=np.asarray(fitted["deranged"]["scores"], dtype=np.float64),
        candidate_scores=np.asarray(fitted["candidate"]["scores"], dtype=np.float64),
    )
    integrity_ok = bool(
        parity <= PARITY_ATOL
        and all(bool(result["converged"]) for result in fitted.values())
        and all(np.isfinite(values).all() for values in arms.values())
        and labels.size == EXPECTED_TRAIN_SAMPLES
        and formal_contract["validation_split_used"] is False
        and formal_contract["test_split_used"] is False
    )
    readiness = assess_readiness(
        summaries=summaries,
        bootstrap=bootstrap,
        pair_fold_deltas=pair_fold_deltas,
        integrity_ok=integrity_ok,
    )
    correlations = []
    for column in range(FEATURES_PER_ARM):
        left = arms["candidate"][:, column].astype(np.float64)
        right = arms["deranged"][:, column].astype(np.float64)
        if left.std() <= 1e-12 or right.std() <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    summary = {
        **formal_contract,
        "preflight_artifact": {"path": preflight["path"], "sha256": preflight["sha256"]},
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "block11_base_max_abs_error": parity,
        "feature_artifact": str(feature_path),
        "feature_artifact_sha256": feature_sha,
        "oof_artifact": str(oof_path),
        "oof_artifact_sha256": oof_sha,
        "feature_rank": {
            name: int(np.linalg.matrix_rank(values.astype(np.float64)))
            for name, values in arms.items()
        },
        "candidate_deranged_feature_correlation": {
            "per_column": correlations,
            "mean_absolute": float(np.mean(np.abs(correlations))),
        },
        "screen": summaries,
        "pair_fold_deltas": pair_fold_deltas,
        "fold_models": {name: result["fold_models"] for name, result in fitted.items()},
        "bootstrap": bootstrap,
        "readiness": readiness,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    _atomic_json(output_dir / "summary.json", summary)  # type: ignore[operator]
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run(args)
    readiness = summary.get("readiness", {})
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "preflight_only": bool(args.preflight_only),
                "signal_gate_passed": bool(readiness.get("signal_gate_passed", False)),
                "validation_permission": False,
                "test_permission": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

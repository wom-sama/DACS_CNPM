from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from trkh.models.dinov3_cgaer_bridge_b11 import (  # noqa: E402
    CGAER_B11_CANDIDATE_MODE,
    CGAER_B11_CONTROL_MODE,
    CGAER_B11_PARAMETER_COUNT,
    DinoV3CGAERBridgeB11,
)
from trkh.tools import precheck_dinov3_cgaer_b11_sourcefold as b11_preflight  # noqa: E402
from trkh.tools.export_timm_predictions import _state_dict_sha256  # noqa: E402
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_SELECTED_STATE_SHA256,
    _array_finite,
    _sha256,
)
from trkh.tools.probe_embedding_prototypes import _resolve_device  # noqa: E402


PROTOCOL_ID = b11_preflight.PROTOCOL_ID
FOLDS = 5
SEED = b11_preflight.SEED
EPOCHS = 5
LR = 1.5e-4
WEIGHT_DECAY = 0.0
ADAPTER_BATCH_SIZE = 128
TEMPERED_POWER = 0.5
CONFLICT_LOSS_WEIGHT = 0.10
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = SEED
FP_SOURCE_CLASSES = (0, 2, 3, 4)
EXPECTED_TRAIN_SAMPLES = b11_preflight.EXPECTED_SAMPLES
EXPECTED_CLASS_COUNTS = b11_preflight.EXPECTED_COUNTS
EXPECTED_A0_CACHE_MANIFEST_SHA256 = b11_preflight.EXPECTED_CACHE_HASHES["manifest"]
EXPECTED_A0_TOKENS_SHA256 = b11_preflight.EXPECTED_CACHE_HASHES["tokens"]
EXPECTED_A0_LOGITS_SHA256 = b11_preflight.EXPECTED_CACHE_HASHES["logits"]
EXPECTED_A0_LABELS_SHA256 = b11_preflight.EXPECTED_CACHE_HASHES["labels"]
EXPECTED_A0_PATHS_SHA256 = b11_preflight.EXPECTED_CACHE_HASHES["paths"]
EXPECTED_ASSIGNMENT_CSV_SHA256 = b11_preflight.EXPECTED_ASSIGNMENT_CSV_SHA256
EXPECTED_CACHE_ORDER_FOLD_SHA256 = b11_preflight.EXPECTED_CACHE_ORDER_FOLD_SHA256
EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256 = (
    b11_preflight.EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256
)
EXPECTED_CACHE_ORDER_GROUP_SHA256 = b11_preflight.EXPECTED_CACHE_ORDER_GROUP_SHA256
CACHE_FILENAMES = b11_preflight.CACHE_FILES
CACHE_HASHES = b11_preflight.EXPECTED_CACHE_HASHES


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked TRAIN-only B11 conflict-gated adjacent-energy residual screen. "
            "It consumes only the exact A0 cache and never constructs validation/test."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path, required=True)
    parser.add_argument("--preflight-artifact-sha256", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-batch-size", type=int, default=ADAPTER_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refuse to overwrite artifact: {path}")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> str:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refuse to overwrite state artifact: {path}")
    torch.save(payload, partial)
    digest = _sha256(partial)
    partial.replace(path)
    if _sha256(path) != digest:
        raise RuntimeError("atomic state promotion hash drifted")
    return digest


def validate_preflight_artifact(
    path: Path, expected_sha256: str
) -> Tuple[Dict[str, object], str]:
    """Delegate schema/source validation to the B11 preflight module."""

    resolved = Path(path).expanduser().resolve()
    digest = _sha256(resolved)
    expected = str(expected_sha256).strip().casefold()
    if len(expected) != 64 or digest != expected:
        raise ValueError(
            f"B11 preflight SHA-256 mismatch: expected={expected}, observed={digest}"
        )
    validator = getattr(b11_preflight, "validate_runner_preflight", None)
    if not callable(validator):
        raise RuntimeError("B11 preflight lacks validate_runner_preflight")
    payload = validator(resolved, expected_sha256=digest)
    if not isinstance(payload, Mapping):
        raise ValueError("B11 preflight validator must return a mapping")
    return dict(payload), digest


def validate_a0_cache(cache_dir: Path) -> Dict[str, object]:
    """Hash and memory-map the sole admissible lossless TRAIN cache."""

    root = Path(cache_dir).expanduser().resolve()
    files = {name: root / filename for name, filename in CACHE_FILENAMES.items()}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"A0 cache artifacts missing: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    mismatches = {
        name: {"expected": CACHE_HASHES[name], "observed": observed[name]}
        for name in CACHE_HASHES
        if observed[name] != CACHE_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"immutable A0 cache hash mismatch: {mismatches}")
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    required_manifest = {
        "selected_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        "tokens_sha256": EXPECTED_A0_TOKENS_SHA256,
        "logits_sha256": EXPECTED_A0_LOGITS_SHA256,
        "labels_sha256": EXPECTED_A0_LABELS_SHA256,
        "paths_sha256": EXPECTED_A0_PATHS_SHA256,
        "token_shape": [EXPECTED_TRAIN_SAMPLES, 256, 384],
        "logit_shape": [EXPECTED_TRAIN_SAMPLES, 5],
        "label_shape": [EXPECTED_TRAIN_SAMPLES],
        "token_dtype": "float32",
        "logit_dtype": "float32",
        "label_dtype": "int64",
        "class_counts": list(EXPECTED_CLASS_COUNTS),
        "lossy_token_quantization": False,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
    }
    drift = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in required_manifest.items()
        if manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"A0 cache manifest contract drifted: {drift}")
    paths = json.loads(files["paths"].read_text(encoding="utf-8"))
    if (
        not isinstance(paths, list)
        or len(paths) != EXPECTED_TRAIN_SAMPLES
        or not all(isinstance(value, str) for value in paths)
    ):
        raise ValueError("A0 cache path order/content changed")
    if any("/val/" in f"/{p.casefold()}/" or "/test/" in f"/{p.casefold()}/" for p in paths):
        raise ValueError("A0 cache path list is not TRAIN-only")
    arrays = {
        "tokens": np.load(files["tokens"], mmap_mode="r", allow_pickle=False),
        "logits": np.load(files["logits"], mmap_mode="r", allow_pickle=False),
        "labels": np.load(files["labels"], mmap_mode="r", allow_pickle=False),
    }
    if arrays["tokens"].shape != (EXPECTED_TRAIN_SAMPLES, 256, 384) or arrays["tokens"].dtype != np.float32:
        raise ValueError("A0 token shape/dtype changed")
    if arrays["logits"].shape != (EXPECTED_TRAIN_SAMPLES, 5) or arrays["logits"].dtype != np.float32:
        raise ValueError("A0 logit shape/dtype changed")
    if arrays["labels"].shape != (EXPECTED_TRAIN_SAMPLES,) or arrays["labels"].dtype != np.int64:
        raise ValueError("A0 label shape/dtype changed")
    if not _array_finite(arrays["tokens"]) or not _array_finite(arrays["logits"]):
        raise ValueError("A0 cache contains non-finite floating values")
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if tuple(np.bincount(labels, minlength=5).tolist()) != EXPECTED_CLASS_COUNTS:
        raise ValueError("A0 cached class support changed")
    return {
        **arrays,
        "paths": paths,
        "manifest": manifest,
        "files": {key: str(value) for key, value in files.items()},
        "hashes": observed,
    }


def pooled_features_from_tokens(tokens: np.ndarray, chunk_size: int = 32) -> np.ndarray:
    """Mean-pool lossless patch tokens in bounded RAM chunks."""

    if tuple(tokens.shape) != (EXPECTED_TRAIN_SAMPLES, 256, 384) or tokens.dtype != np.float32:
        raise ValueError("B11 pooling requires exact [8278,256,384] FP32 tokens")
    if int(chunk_size) <= 0:
        raise ValueError("pooling chunk size must be positive")
    pooled = np.full((EXPECTED_TRAIN_SAMPLES, 384), np.nan, dtype=np.float32)
    for start in range(0, EXPECTED_TRAIN_SAMPLES, int(chunk_size)):
        end = min(EXPECTED_TRAIN_SAMPLES, start + int(chunk_size))
        pooled[start:end] = np.asarray(tokens[start:end], dtype=np.float32).mean(
            axis=1, dtype=np.float32
        )
    if not np.isfinite(pooled).all():
        raise RuntimeError("pooled B11 features are incomplete/non-finite")
    return pooled


def read_locked_assignment(
    path: Path, paths: Sequence[str], labels: np.ndarray
) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    digest = _sha256(resolved)
    if digest != EXPECTED_ASSIGNMENT_CSV_SHA256:
        raise ValueError(
            "fold assignment CSV hash mismatch: "
            f"expected={EXPECTED_ASSIGNMENT_CSV_SHA256}, observed={digest}"
        )
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != [
            "relative_path", "label", "union_group", "fold"
        ]:
            raise ValueError("fold assignment CSV columns changed")
        rows = list(reader)
    if len(rows) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("fold assignment CSV does not cover 8,278 rows")
    if any(
        row["relative_path"] != paths[index]
        or int(row["label"]) != int(labels[index])
        for index, row in enumerate(rows)
    ):
        raise ValueError("fold assignment order/labels differ from A0 cache")
    groups = np.asarray([int(row["union_group"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if bool((groups < 0).any()) or set(np.unique(folds).tolist()) != set(range(FOLDS)):
        raise ValueError("invalid group/fold IDs in assignment CSV")
    assignment_sha = hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest()
    path_fold = "\n".join(f"{paths[i]}\t{int(folds[i])}" for i in range(len(paths))) + "\n"
    path_fold_sha = hashlib.sha256(path_fold.encode("utf-8")).hexdigest()
    group_sha = hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest()
    if (
        assignment_sha != EXPECTED_CACHE_ORDER_FOLD_SHA256
        or path_fold_sha != EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256
        or group_sha != EXPECTED_CACHE_ORDER_GROUP_SHA256
    ):
        raise ValueError("fold assignment vector hashes drifted")
    fold_rows: List[Dict[str, object]] = []
    for fold in range(FOLDS):
        held = folds == fold
        fit = ~held
        overlap = set(groups[fit].tolist()) & set(groups[held].tolist())
        counts = np.bincount(labels[held], minlength=5).astype(int).tolist()
        if overlap or any(value <= 0 for value in counts):
            raise ValueError(f"fold {fold} is not group-disjoint/class-complete")
        fold_rows.append(
            {
                "fold": fold,
                "fit_samples": int(fit.sum()),
                "held_samples": int(held.sum()),
                "held_class_counts": counts,
                "group_overlap": 0,
            }
        )
    return {
        "rows": rows,
        "groups": groups,
        "folds": folds,
        "fold_rows": fold_rows,
        "csv_sha256": digest,
        "assignment_int64_sha256": assignment_sha,
        "path_fold_sha256": path_fold_sha,
        "group_vector_int64_sha256": group_sha,
    }


def component_conflict_targets(
    labels: np.ndarray, groups: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build normalized-Gini targets; mask singletons and balance components."""

    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if labels.ndim != 1 or labels.shape != groups.shape:
        raise ValueError("conflict labels/groups are not aligned")
    targets = np.zeros(labels.size, dtype=np.float32)
    eligible = np.zeros(labels.size, dtype=bool)
    mixed = np.zeros(labels.size, dtype=bool)
    inverse_sizes = np.zeros(labels.size, dtype=np.float32)
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        size = int(positions.size)
        if size <= 1:
            continue
        eligible[positions] = True
        inverse_sizes[positions] = np.float32(1.0 / size)
        counts = np.bincount(labels[positions], minlength=5).astype(np.float64)
        represented = counts[counts > 0]
        r = int(represented.size)
        if r > 1:
            p = represented / represented.sum()
            target = (1.0 - float(np.square(p).sum())) / (1.0 - 1.0 / r)
            if not (0.0 <= target <= 1.0 + 1e-12):
                raise RuntimeError("normalized component conflict target escaped [0,1]")
            targets[positions] = np.float32(np.clip(target, 0.0, 1.0))
            mixed[positions] = True
    return {
        "targets": targets,
        "eligible": eligible,
        "mixed": mixed,
        "inverse_sizes": inverse_sizes,
    }


def conflict_weighted_loss(
    logits: Tensor,
    targets: Tensor,
    conflict_logits: Tensor,
    conflict_targets: Tensor,
    conflict_eligible: Tensor,
    inverse_component_sizes: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Hard five-way CE plus masked, component-balanced conflict BCE."""

    hard_ce = F.cross_entropy(logits, targets)
    mask = conflict_eligible.to(dtype=torch.bool)
    if bool(mask.any().item()):
        weights = inverse_component_sizes[mask]
        if not bool(torch.isfinite(weights).all().item()) or bool((weights <= 0).any().item()):
            raise ValueError("eligible conflict weights must be finite and positive")
        raw = F.binary_cross_entropy_with_logits(
            conflict_logits[mask], conflict_targets[mask], reduction="none"
        )
        conflict_bce = (raw * weights).sum() / weights.sum()
    else:
        conflict_bce = conflict_logits.sum() * 0.0
    total = hard_ce + CONFLICT_LOSS_WEIGHT * conflict_bce
    return total, hard_ce.detach(), conflict_bce.detach()


def tempered_all_class_indices(
    fit_indices: np.ndarray, labels: np.ndarray, seed: int, batch_size: int
) -> np.ndarray:
    """Locked power sampler over all five classes (unlike A0 pair training)."""

    fit = np.asarray(fit_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if fit.ndim != 1 or fit.size == 0 or int(batch_size) <= 0:
        raise ValueError("tempered sampler inputs are invalid")
    fit_labels = labels[fit]
    counts = np.bincount(fit_labels, minlength=5).astype(np.float64)
    if bool((counts <= 0).any()):
        raise ValueError("all five classes must occur in every fit fold")
    weights = np.power(counts[fit_labels], -TEMPERED_POWER)
    batches = (fit.size + int(batch_size) - 1) // int(batch_size)
    rng = np.random.default_rng(int(seed))
    return rng.choice(
        fit,
        size=batches * int(batch_size),
        replace=True,
        p=weights / weights.sum(),
    ).astype(np.int64, copy=False)


def _build_paired_bridges(seed: int) -> Tuple[nn.Module, nn.Module, Dict[str, object]]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        candidate = DinoV3CGAERBridgeB11(mode=CGAER_B11_CANDIDATE_MODE)
        control = DinoV3CGAERBridgeB11(mode=CGAER_B11_CONTROL_MODE)
    control.load_state_dict(candidate.state_dict(), strict=True)
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if tuple(control_state) != tuple(candidate_state) or any(
        not torch.equal(control_state[key], candidate_state[key]) for key in control_state
    ):
        raise RuntimeError("B11 paired arms do not share an identical initial state")
    counts = [sum(parameter.numel() for parameter in module.parameters()) for module in (control, candidate)]
    if counts != [CGAER_B11_PARAMETER_COUNT, CGAER_B11_PARAMETER_COUNT]:
        raise RuntimeError(f"B11 parameter-count contract changed: {counts}")
    return control, candidate, {
        "control_parameters": counts[0],
        "candidate_parameters": counts[1],
        "identical_initial_state": True,
        "initial_state_sha256": _state_dict_sha256(control_state),
    }


def metrics_all_fp(labels: np.ndarray, logits: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits)
    if logits.shape != (labels.size, 5) or not np.isfinite(logits).all():
        raise ValueError("metric logits are incomplete/non-finite")
    predicted = logits.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=list(range(5)))
    per_f1: List[float] = []
    for cls in range(5):
        tp = int(matrix[cls, cls])
        fp = int(matrix[:, cls].sum() - tp)
        fn = int(matrix[cls, :].sum() - tp)
        per_f1.append(2.0 * tp / max(1, 2 * tp + fp + fn))
    tp1 = int(matrix[1, 1])
    fp1 = int(matrix[:, 1].sum() - tp1)
    fn1 = int(matrix[1, :].sum() - tp1)
    fp_sources = {str(cls): int(matrix[cls, 1]) for cls in FP_SOURCE_CLASSES}
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_f1": float(np.mean(per_f1)),
        "class1_precision": float(tp1 / max(1, tp1 + fp1)),
        "class1_recall": float(tp1 / max(1, tp1 + fn1)),
        "class1_f1": float(per_f1[1]),
        "class1_tp": tp1,
        "class1_fn": fn1,
        "class1_support": int(matrix[1].sum()),
        "total_fp_to_class1": int(sum(fp_sources.values())),
        "fp_to_class1": fp_sources,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _class1_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(metrics_all_fp(labels, logits)["class1_f1"])


def _macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(metrics_all_fp(labels, logits)["macro_f1"])


def _fp_rate(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(logits).argmax(axis=1)
    mask = labels != 1
    return float(np.sum(mask & (predicted == 1)) / max(1, mask.sum()))


def _gate_aurocs(
    labels: np.ndarray,
    candidate_logits: np.ndarray,
    gates: np.ndarray,
    eligible: np.ndarray,
    mixed: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(candidate_logits).argmax(axis=1)
    gates = np.asarray(gates, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    mixed = np.asarray(mixed, dtype=bool)
    if not np.isfinite(gates).all():
        raise ValueError("gate scores are incomplete/non-finite")
    if np.unique(mixed[eligible]).size != 2:
        raise ValueError("mixed/pure gate AUROC lacks binary support")
    errors = np.logical_or(
        np.logical_and(labels == 1, predicted != 1),
        np.logical_and(labels != 1, predicted == 1),
    )
    boundary = np.logical_or(labels == 1, predicted == 1)
    conditional_errors = predicted[boundary] != labels[boundary]
    if np.unique(errors).size != 2 or np.unique(conditional_errors).size != 2:
        raise ValueError("class-1 boundary gate AUROC lacks binary support")
    return {
        "mixed_vs_pure": float(roc_auc_score(mixed[eligible], gates[eligible])),
        "mixed_vs_pure_samples": int(eligible.sum()),
        "class1_boundary_error": float(roc_auc_score(errors, gates)),
        "class1_boundary_error_conditional": float(
            roc_auc_score(conditional_errors, gates[boundary])
        ),
        "class1_boundary_samples": int(labels.size),
        "class1_boundary_conditional_samples": int(boundary.sum()),
        "class1_boundary_errors": int(errors.sum()),
    }


def _summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("telemetry is empty/non-finite")
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def run_oof_screen(
    *,
    features: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    paths: Sequence[str],
    groups: np.ndarray,
    folds: np.ndarray,
    conflict: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    preflight_sha256: str,
    cache_hashes: Mapping[str, str],
) -> Dict[str, object]:
    n = int(labels.size)
    if n != EXPECTED_TRAIN_SAMPLES or int(batch_size) != ADAPTER_BATCH_SIZE:
        raise ValueError("B11 OOF shape/batch contract changed")
    outputs = {
        name: np.full((n, 5), np.nan, dtype=np.float32)
        for name in ("control", "candidate", "candidate_force_gate")
    }
    vectors = {
        name: np.full(n, np.nan, dtype=np.float32)
        for name in ("control_gate", "candidate_gate", "control_residual", "candidate_residual")
    }
    state_dir = output_dir / "fold_adapter_states"
    state_dir.mkdir(parents=False, exist_ok=False)
    training_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    paired_contracts: List[Dict[str, object]] = []
    expected_updates = completed_updates = skipped_updates = nonfinite_updates = 0
    branch_off_error = 0.0

    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        updates_per_epoch = (fit.size + batch_size - 1) // batch_size
        expected_updates += EPOCHS * updates_per_epoch
        control, candidate, paired = _build_paired_bridges(SEED + fold)
        control = control.to(device=device, dtype=torch.float32)
        candidate = candidate.to(device=device, dtype=torch.float32)
        paired_contracts.append({"fold": fold, **paired})
        optimizers = tuple(
            torch.optim.AdamW(module.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            for module in (control, candidate)
        )
        for epoch in range(EPOCHS):
            schedule_seed = SEED + fold * 100 + epoch
            schedule = tempered_all_class_indices(fit, labels, schedule_seed, batch_size)
            schedule_row = {
                "fold": fold,
                "epoch": epoch + 1,
                "seed": schedule_seed,
                "samples": int(schedule.size),
                "class_counts": np.bincount(labels[schedule], minlength=5).astype(int).tolist(),
                "schedule_int64_sha256": hashlib.sha256(schedule.astype("<i8").tobytes()).hexdigest(),
            }
            totals = np.zeros((2, 3), dtype=np.float64)
            seen = 0
            control.train()
            candidate.train()
            for start in range(0, schedule.size, batch_size):
                selected = schedule[start : start + batch_size]
                z = torch.from_numpy(features[selected]).to(device)
                base = torch.from_numpy(np.asarray(base_logits[selected], dtype=np.float32)).to(device)
                target = torch.from_numpy(labels[selected]).to(device)
                q = torch.from_numpy(conflict["targets"][selected]).to(device)
                mask = torch.from_numpy(conflict["eligible"][selected]).to(device)
                inv = torch.from_numpy(conflict["inverse_sizes"][selected]).to(device)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                arm_losses: List[Tuple[Tensor, Tensor, Tensor]] = []
                valid = True
                for module in (control, candidate):
                    logits, trace = module(z, base, return_trace=True)
                    losses = conflict_weighted_loss(
                        logits, target, trace["conflict_logit"], q, mask, inv
                    )
                    arm_losses.append(tuple(value.detach() for value in losses))
                    if not bool(torch.isfinite(losses[0]).item()):
                        valid = False
                        break
                    losses[0].backward()
                    valid = all(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in module.parameters()
                    )
                    if not valid:
                        break
                if not valid or len(arm_losses) != 2:
                    nonfinite_updates += 1
                    skipped_updates += 1
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    continue
                for optimizer in optimizers:
                    optimizer.step()
                if not all(
                    bool(torch.isfinite(parameter).all().item())
                    for module in (control, candidate) for parameter in module.parameters()
                ):
                    raise FloatingPointError("B11 parameter became non-finite")
                completed_updates += 1
                count = int(selected.size)
                for arm, loss_tuple in enumerate(arm_losses):
                    totals[arm] += np.asarray([float(v.cpu()) for v in loss_tuple]) * count
                seen += count
            training_rows.append(
                {
                    **schedule_row,
                    "completed_samples": seen,
                    "control_loss": float(totals[0, 0] / max(1, seen)),
                    "control_ce": float(totals[0, 1] / max(1, seen)),
                    "control_conflict_bce": float(totals[0, 2] / max(1, seen)),
                    "candidate_loss": float(totals[1, 0] / max(1, seen)),
                    "candidate_ce": float(totals[1, 1] / max(1, seen)),
                    "candidate_conflict_bce": float(totals[1, 2] / max(1, seen)),
                }
            )

        final_states = {
            name: {key: value.detach().cpu() for key, value in module.state_dict().items()}
            for name, module in (("control", control), ("candidate", candidate))
        }
        state_path = state_dir / f"fold_{fold}_final_bridges.pt"
        state_sha = _atomic_torch_save(
            state_path,
            {
                "protocol_id": PROTOCOL_ID,
                "fold": fold,
                "preflight_artifact_sha256": preflight_sha256,
                "cache_hashes": dict(cache_hashes),
                "control_state": final_states["control"],
                "candidate_state": final_states["candidate"],
            },
        )
        paired_contracts[-1].update(
            {
                "control_final_state_sha256": _state_dict_sha256(final_states["control"]),
                "candidate_final_state_sha256": _state_dict_sha256(final_states["candidate"]),
                "state_artifact": str(state_path),
                "state_artifact_sha256": state_sha,
            }
        )
        control.eval()
        candidate.eval()
        with torch.inference_mode():
            for start in range(0, held.size, batch_size):
                selected = held[start : start + batch_size]
                z = torch.from_numpy(features[selected]).to(device)
                base = torch.from_numpy(np.asarray(base_logits[selected], dtype=np.float32)).to(device)
                for arm_name, module in (("control", control), ("candidate", candidate)):
                    logits, trace = module(z, base, return_trace=True)
                    outputs[arm_name][selected] = logits.cpu().numpy()
                    vectors[f"{arm_name}_gate"][selected] = trace["gate"].cpu().numpy()
                    vectors[f"{arm_name}_residual"][selected] = trace["residual_l2_norm"].cpu().numpy()
                forced = candidate(z, base, force_gate=0.5)
                outputs["candidate_force_gate"][selected] = forced.cpu().numpy()
                off = candidate(z, base, branch_off=True)
                branch_off_error = max(branch_off_error, float((off - base).abs().max().cpu()))
        control_fold = metrics_all_fp(labels[held], outputs["control"][held])
        candidate_fold = metrics_all_fp(labels[held], outputs["candidate"][held])
        fold_rows.append(
            {
                "fold": fold,
                "control": control_fold,
                "candidate": candidate_fold,
                "candidate_minus_control_class1_f1": float(
                    candidate_fold["class1_f1"] - control_fold["class1_f1"]
                ),
                "candidate_win": bool(candidate_fold["class1_f1"] > control_fold["class1_f1"]),
            }
        )

    for name, array in {**outputs, **vectors}.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"B11 OOF {name} is incomplete/non-finite")
    oof_path = output_dir / "train_oof_logits.npz"
    partial = oof_path.with_suffix(oof_path.suffix + ".partial")
    if oof_path.exists() or partial.exists():
        raise RuntimeError("refuse to overwrite B11 OOF artifact")
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            relative_paths=np.asarray(paths, dtype=np.str_), labels=labels, folds=folds,
            union_groups=groups, base_logits=np.asarray(base_logits, dtype=np.float32),
            control_logits=outputs["control"], candidate_logits=outputs["candidate"],
            candidate_force_gate_logits=outputs["candidate_force_gate"],
            control_gates=vectors["control_gate"], candidate_gates=vectors["candidate_gate"],
            control_residual_l2_norms=vectors["control_residual"],
            candidate_residual_l2_norms=vectors["candidate_residual"],
            conflict_targets=conflict["targets"], conflict_eligible=conflict["eligible"],
            conflict_mixed=conflict["mixed"], inverse_component_sizes=conflict["inverse_sizes"],
        )
    oof_sha = _sha256(partial)
    partial.replace(oof_path)
    if _sha256(oof_path) != oof_sha:
        raise RuntimeError("atomic OOF promotion hash drifted")
    validate_oof_artifact(oof_path, n)
    base_metrics = metrics_all_fp(labels, base_logits)
    control_metrics = metrics_all_fp(labels, outputs["control"])
    candidate_metrics = metrics_all_fp(labels, outputs["candidate"])
    force_metrics = metrics_all_fp(labels, outputs["candidate_force_gate"])
    return {
        "base": base_metrics,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "candidate_force_gate_0_5": force_metrics,
        "fold_rows": fold_rows,
        "fold_wins": int(sum(row["candidate_win"] for row in fold_rows)),
        "gate_aurocs": {
            arm: _gate_aurocs(labels, outputs[arm], vectors[f"{arm}_gate"], conflict["eligible"], conflict["mixed"])
            for arm in ("control", "candidate")
        },
        "residual_l2_norms": {arm: _summary(vectors[f"{arm}_residual"]) for arm in ("control", "candidate")},
        "branch_off_max_abs_error": branch_off_error,
        "training_rows": training_rows,
        "schedule_contract_sha256": hashlib.sha256(
            json.dumps(training_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "paired_contracts": paired_contracts,
        "expected_updates": expected_updates,
        "completed_updates": completed_updates,
        "skipped_updates": skipped_updates,
        "nonfinite_updates": nonfinite_updates,
        "oof_artifact": str(oof_path),
        "oof_artifact_sha256": oof_sha,
        "oof_artifact_validated": True,
        "group_vector_int64_sha256": hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest(),
        "fold_vector_int64_sha256": hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest(),
    }


def validate_oof_artifact(path: Path, expected_samples: int = EXPECTED_TRAIN_SAMPLES) -> None:
    required = {
        "relative_paths": (expected_samples,),
        "labels": (expected_samples,), "folds": (expected_samples,),
        "union_groups": (expected_samples,), "base_logits": (expected_samples, 5),
        "control_logits": (expected_samples, 5), "candidate_logits": (expected_samples, 5),
        "candidate_force_gate_logits": (expected_samples, 5),
        "control_gates": (expected_samples,), "candidate_gates": (expected_samples,),
        "control_residual_l2_norms": (expected_samples,),
        "candidate_residual_l2_norms": (expected_samples,),
        "conflict_targets": (expected_samples,), "conflict_eligible": (expected_samples,),
        "conflict_mixed": (expected_samples,), "inverse_component_sizes": (expected_samples,),
    }
    with np.load(path, allow_pickle=False) as artifact:
        missing = [name for name in required if name not in artifact]
        if missing:
            raise ValueError(f"OOF artifact fields missing: {missing}")
        for name, shape in required.items():
            value = artifact[name]
            if value.shape != shape or (value.dtype.kind == "f" and not np.isfinite(value).all()):
                raise ValueError(f"OOF artifact {name} is incomplete/invalid")
        if artifact["relative_paths"].dtype.kind not in "SU" or np.unique(artifact["relative_paths"]).size != expected_samples:
            raise ValueError("OOF relative paths are invalid/non-unique")
        if any(artifact[name].dtype.kind not in "iu" for name in ("labels", "folds", "union_groups")):
            raise ValueError("OOF labels/folds/groups must be integral")
        if bool((artifact["labels"] < 0).any()) or bool((artifact["labels"] >= 5).any()):
            raise ValueError("OOF labels escaped the five-class range")
        if bool((artifact["folds"] < 0).any()) or bool((artifact["union_groups"] < 0).any()):
            raise ValueError("OOF fold/group IDs are invalid")
        if not set(np.unique(artifact["folds"]).tolist()).issubset(set(range(FOLDS))):
            raise ValueError("OOF fold IDs escaped the locked range")
        if expected_samples == EXPECTED_TRAIN_SAMPLES and set(np.unique(artifact["folds"]).tolist()) != set(range(FOLDS)):
            raise ValueError("OOF artifact does not contain all five locked folds")
        if bool((artifact["conflict_targets"] < 0).any()) or bool((artifact["conflict_targets"] > 1).any()):
            raise ValueError("OOF conflict targets escaped [0,1]")
        if bool((artifact["inverse_component_sizes"] < 0).any()):
            raise ValueError("OOF inverse component sizes are negative")
        for group in np.unique(artifact["union_groups"]):
            if np.unique(artifact["folds"][artifact["union_groups"] == group]).size != 1:
                raise ValueError("OOF union component crosses held folds")
        expected = component_conflict_targets(artifact["labels"], artifact["union_groups"])
        for saved, key in (
            ("conflict_targets", "targets"), ("conflict_eligible", "eligible"),
            ("conflict_mixed", "mixed"), ("inverse_component_sizes", "inverse_sizes"),
        ):
            if not np.array_equal(artifact[saved], expected[key]):
                raise ValueError(f"OOF {saved} differs from groups/labels")


def _interval(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("bootstrap interval values are incomplete/non-finite")
    return {"lower": float(np.quantile(values, 0.025)), "upper": float(np.quantile(values, 0.975))}


def fold_stratified_component_bootstrap(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    base_logits: np.ndarray,
    control_logits: np.ndarray,
    candidate_logits: np.ndarray,
    force_logits: np.ndarray,
    candidate_gates: np.ndarray,
    conflict_eligible: np.ndarray,
    conflict_mixed: np.ndarray,
    oof_artifact: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, object]:
    """Paired resampling of complete components, independently within folds."""

    labels, folds, groups = (np.asarray(value, dtype=np.int64) for value in (labels, folds, groups))
    if labels.shape != folds.shape or labels.shape != groups.shape or int(replicates) <= 0:
        raise ValueError("bootstrap vectors/replicate count are invalid")
    arrays = (base_logits, control_logits, candidate_logits, force_logits)
    if any(np.asarray(value).shape != (labels.size, 5) for value in arrays):
        raise ValueError("bootstrap logits must be [N,5]")
    members: Dict[int, np.ndarray] = {}
    fold_groups: Dict[int, List[int]] = {fold: [] for fold in range(FOLDS)}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        group_folds = np.unique(folds[positions])
        if group_folds.size != 1:
            raise ValueError(f"union component {group} crosses held folds")
        members[int(group)] = positions
        fold_groups[int(group_folds[0])].append(int(group))
    if any(not fold_groups[fold] for fold in range(FOLDS)):
        raise ValueError("bootstrap requires components in all five folds")
    names = (
        "c1_vs_base", "c1_vs_control", "macro_vs_base", "macro_vs_control",
        "fp_vs_base", "fp_vs_control", "mixed_gate_auc", "boundary_gate_auc",
        "active_vs_force_c1",
    )
    values = {name: np.empty(int(replicates), dtype=np.float64) for name in names}
    rng = np.random.default_rng(int(seed))
    draw_hash = hashlib.sha256()
    for replicate in range(int(replicates)):
        chunks: List[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draw_hash.update(np.asarray([replicate, fold], dtype="<i8").tobytes())
            draw_hash.update(draw.astype("<i8").tobytes())
            chunks.extend(members[available[int(index)]] for index in draw)
        selected = np.concatenate(chunks)
        y = labels[selected]
        base, control, candidate, force = (np.asarray(value)[selected] for value in arrays)
        values["c1_vs_base"][replicate] = _class1_f1(y, candidate) - _class1_f1(y, base)
        values["c1_vs_control"][replicate] = _class1_f1(y, candidate) - _class1_f1(y, control)
        values["macro_vs_base"][replicate] = _macro_f1(y, candidate) - _macro_f1(y, base)
        values["macro_vs_control"][replicate] = _macro_f1(y, candidate) - _macro_f1(y, control)
        values["fp_vs_base"][replicate] = _fp_rate(y, candidate) - _fp_rate(y, base)
        values["fp_vs_control"][replicate] = _fp_rate(y, candidate) - _fp_rate(y, control)
        gate_metrics = _gate_aurocs(
            y, candidate, np.asarray(candidate_gates)[selected],
            np.asarray(conflict_eligible)[selected], np.asarray(conflict_mixed)[selected],
        )
        values["mixed_gate_auc"][replicate] = float(gate_metrics["mixed_vs_pure"])
        values["boundary_gate_auc"][replicate] = float(gate_metrics["class1_boundary_error"])
        values["active_vs_force_c1"][replicate] = _class1_f1(y, candidate) - _class1_f1(y, force)
    points = {
        "class1_f1_delta_vs_b9": _class1_f1(labels, candidate_logits) - _class1_f1(labels, base_logits),
        "class1_f1_delta_vs_control": _class1_f1(labels, candidate_logits) - _class1_f1(labels, control_logits),
        "macro_f1_delta_vs_b9": _macro_f1(labels, candidate_logits) - _macro_f1(labels, base_logits),
        "macro_f1_delta_vs_control": _macro_f1(labels, candidate_logits) - _macro_f1(labels, control_logits),
        "fp_rate_delta_vs_b9": _fp_rate(labels, candidate_logits) - _fp_rate(labels, base_logits),
        "fp_rate_delta_vs_control": _fp_rate(labels, candidate_logits) - _fp_rate(labels, control_logits),
        "active_vs_force_class1_f1": _class1_f1(labels, candidate_logits) - _class1_f1(labels, force_logits),
    }
    gate_points = _gate_aurocs(labels, candidate_logits, candidate_gates, conflict_eligible, conflict_mixed)
    mapping = {
        "class1_f1_delta_vs_b9": "c1_vs_base", "class1_f1_delta_vs_control": "c1_vs_control",
        "macro_f1_delta_vs_b9": "macro_vs_base", "macro_f1_delta_vs_control": "macro_vs_control",
        "fp_rate_delta_vs_b9": "fp_vs_base", "fp_rate_delta_vs_control": "fp_vs_control",
        "active_vs_force_class1_f1": "active_vs_force_c1",
    }
    result = {name: {"point": float(point), **_interval(values[mapping[name]])} for name, point in points.items()}
    result["mixed_vs_pure_gate_auroc"] = {
        "point": float(gate_points["mixed_vs_pure"]), **_interval(values["mixed_gate_auc"])
    }
    result["class1_boundary_error_gate_auroc"] = {
        "point": float(gate_points["class1_boundary_error"]), **_interval(values["boundary_gate_auc"])
    }
    return {
        "method": "paired_fold_stratified_union_component_percentile_bootstrap",
        "replicates": int(replicates), "seed": int(seed),
        "draws_int64_sha256": draw_hash.hexdigest(),
        "oof_artifact_sha256": _sha256(Path(oof_artifact)),
        "group_vector_int64_sha256": hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest(),
        "fold_vector_int64_sha256": hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest(),
        **result,
    }


def assess_readiness(screen: Mapping[str, object], bootstrap: Mapping[str, object]) -> Dict[str, object]:
    base, control, candidate = (screen[name] for name in ("base", "control", "candidate"))
    force = screen["candidate_force_gate_0_5"]
    c1_base = bootstrap["class1_f1_delta_vs_b9"]
    c1_control = bootstrap["class1_f1_delta_vs_control"]
    macro_base = bootstrap["macro_f1_delta_vs_b9"]
    macro_control = bootstrap["macro_f1_delta_vs_control"]
    fp_base = bootstrap["fp_rate_delta_vs_b9"]
    fp_control = bootstrap["fp_rate_delta_vs_control"]
    gate = bootstrap["mixed_vs_pure_gate_auroc"]
    boundary = bootstrap["class1_boundary_error_gate_auroc"]
    candidate_fp, base_fp, control_fp = (
        value["fp_to_class1"] for value in (candidate, base, control)
    )
    rows = list(screen.get("training_rows", []))
    paired = list(screen.get("paired_contracts", []))
    residual_p95 = float(screen["residual_l2_norms"]["candidate"]["p95"])
    checks = {
        "preflight_artifact_verified": bool(screen.get("preflight_artifact_verified")),
        "immutable_a0_cache_verified": bool(screen.get("immutable_a0_cache_verified")),
        "locked_assignment_verified": bool(screen.get("locked_assignment_verified")),
        "paired_capacity_and_initial_state": len(paired) == FOLDS and all(
            int(row.get("control_parameters", -1)) == CGAER_B11_PARAMETER_COUNT
            and int(row.get("candidate_parameters", -1)) == CGAER_B11_PARAMETER_COUNT
            and row.get("identical_initial_state") is True for row in paired
        ),
        "all_five_class_tempered_schedule": len(rows) == FOLDS * EPOCHS and all(
            len(row.get("class_counts", [])) == 5 and all(int(v) > 0 for v in row["class_counts"])
            for row in rows
        ),
        "bootstrap_contract_locked": int(bootstrap.get("replicates", -1)) == BOOTSTRAP_REPLICATES
        and int(bootstrap.get("seed", -1)) == BOOTSTRAP_SEED
        and bootstrap.get("oof_artifact_sha256") == screen.get("oof_artifact_sha256")
        and bootstrap.get("group_vector_int64_sha256") == screen.get("group_vector_int64_sha256")
        and bootstrap.get("fold_vector_int64_sha256") == screen.get("fold_vector_int64_sha256"),
        "class1_gain_vs_b9": float(c1_base["point"]) >= 0.005 and float(c1_base["lower"]) > 0.0,
        "class1_gain_vs_control": float(c1_control["point"]) >= 0.005 and float(c1_control["lower"]) > 0.0,
        "four_of_five_fold_wins": int(screen.get("fold_wins", -1)) >= 4,
        "class1_tp_retention_and_control": int(candidate["class1_tp"]) >= 0.98 * int(base["class1_tp"])
        and int(candidate["class1_tp"]) >= int(control["class1_tp"]),
        "total_fp_point_gates": int(candidate["total_fp_to_class1"]) <= 0.90 * int(base["total_fp_to_class1"])
        and int(candidate["total_fp_to_class1"]) <= int(control["total_fp_to_class1"]),
        "fp_rate_ucb_vs_b9_strictly_negative": float(fp_base["upper"]) < 0.0,
        "fp_rate_ucb_vs_control_nonpositive": float(fp_control["upper"]) <= 0.0,
        "each_source_fp_no_higher": all(
            int(candidate_fp[str(cls)]) <= int(base_fp[str(cls)])
            and int(candidate_fp[str(cls)]) <= int(control_fp[str(cls)]) for cls in FP_SOURCE_CLASSES
        ),
        "macro_noninferior_vs_b9": float(macro_base["lower"]) >= -0.002,
        "macro_noninferior_vs_control": float(macro_control["lower"]) >= -0.002,
        "mixed_pure_gate_auroc": float(gate["point"]) >= 0.75 and float(gate["lower"]) >= 0.70,
        "class1_boundary_gate_auroc": float(boundary["point"]) >= 0.75,
        "force_gate_causal_loss": float(bootstrap["active_vs_force_class1_f1"]["point"]) >= 0.002
        and int(force["total_fp_to_class1"]) >= int(candidate["total_fp_to_class1"]),
        "candidate_residual_p95_active_bounded": 0.01 <= residual_p95 < 0.5,
        "branch_off_exact": float(screen.get("branch_off_max_abs_error", 1.0)) == 0.0,
        "oof_complete_and_validated": screen.get("oof_artifact_validated") is True,
        "finite_complete_paired_updates": int(screen.get("nonfinite_updates", -1)) == 0
        and int(screen.get("skipped_updates", -1)) == 0
        and int(screen.get("completed_updates", -1)) == int(screen.get("expected_updates", -2)),
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "train_only_screen_passed": passed,
        "exact_b11_closed": not passed,
        "frozen_validation_protocol_design_permission": passed,
        "full_validation_permission": False, "full_train_permission": False, "test_permission": False,
        "checks": checks, "failed_checks": [name for name, value in checks.items() if not value],
    }


def run_screen(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.adapter_batch_size) != ADAPTER_BATCH_SIZE:
        raise ValueError(f"adapter batch size is protocol-locked to {ADAPTER_BATCH_SIZE}")
    if int(args.torch_threads) <= 0:
        raise ValueError("torch threads must be positive")
    preflight, preflight_sha = validate_preflight_artifact(
        args.preflight_artifact, args.preflight_artifact_sha256
    )
    cache_dir = args.cache_dir.expanduser().resolve()
    assignment_path = args.assignment_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == cache_dir or cache_dir in output_dir.parents:
        raise ValueError("B11 output must remain outside immutable A0 cache")
    if output_dir.exists():
        raise RuntimeError("B11 output directory must be new")

    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    cache = validate_a0_cache(cache_dir)
    assignment = read_locked_assignment(assignment_path, cache["paths"], cache["labels"])
    # The preflight validator already binds these inputs.  Recheck the returned
    # contract so a validator/schema regression also fails closed here.
    locked_inputs = preflight.get("locked_inputs", {})
    required_inputs = {
        "a0_cache_manifest_sha256": EXPECTED_A0_CACHE_MANIFEST_SHA256,
        "a0_tokens_sha256": EXPECTED_A0_TOKENS_SHA256,
        "a0_logits_sha256": EXPECTED_A0_LOGITS_SHA256,
        "a0_labels_sha256": EXPECTED_A0_LABELS_SHA256,
        "a0_paths_sha256": EXPECTED_A0_PATHS_SHA256,
        "assignment_csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
    }
    if not isinstance(locked_inputs, Mapping) or any(
        locked_inputs.get(key) != value for key, value in required_inputs.items()
    ):
        raise ValueError("B11 preflight does not bind the exact runner inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    features = pooled_features_from_tokens(cache["tokens"])
    conflict = component_conflict_targets(cache["labels"], assignment["groups"])
    screen = run_oof_screen(
        features=features, base_logits=cache["logits"], labels=cache["labels"],
        paths=cache["paths"], groups=assignment["groups"], folds=assignment["folds"],
        conflict=conflict, device=_resolve_device(args.device),
        batch_size=int(args.adapter_batch_size), output_dir=output_dir,
        preflight_sha256=preflight_sha, cache_hashes=cache["hashes"],
    )
    screen.update(
        {
            "preflight_artifact_verified": True,
            "immutable_a0_cache_verified": True,
            "locked_assignment_verified": True,
            "assignment_contract": {
                key: assignment[key] for key in (
                    "csv_sha256", "assignment_int64_sha256", "path_fold_sha256",
                    "group_vector_int64_sha256", "fold_rows",
                )
            },
        }
    )
    with np.load(screen["oof_artifact"], allow_pickle=False) as oof:
        bootstrap = fold_stratified_component_bootstrap(
            labels=oof["labels"], folds=oof["folds"], groups=oof["union_groups"],
            base_logits=oof["base_logits"], control_logits=oof["control_logits"],
            candidate_logits=oof["candidate_logits"],
            force_logits=oof["candidate_force_gate_logits"],
            candidate_gates=oof["candidate_gates"],
            conflict_eligible=oof["conflict_eligible"],
            conflict_mixed=oof["conflict_mixed"],
            oof_artifact=Path(screen["oof_artifact"]),
        )
    readiness = assess_readiness(screen, bootstrap)
    summary = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "scientific_scope": "conditional_train_only_b11_cgaer_sourcefold_screen",
        "preflight_artifact": str(Path(args.preflight_artifact).resolve()),
        "preflight_artifact_sha256": preflight_sha,
        "source_files": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "model_sha256": _sha256(Path(b11_preflight.__file__).parents[1] / "models" / "dinov3_cgaer_bridge_b11.py"),
            "preflight_sha256": _sha256(Path(b11_preflight.__file__).resolve()),
        },
        "locked_optimization": {
            "folds": FOLDS, "seed": SEED, "epochs": EPOCHS, "optimizer": "AdamW",
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "adapter_batch_size": ADAPTER_BATCH_SIZE, "tempered_sampling_power": TEMPERED_POWER,
            "classification_loss": "hard_five_way_cross_entropy",
            "conflict_loss": "0.10_inverse_component_size_weighted_masked_BCE",
            "paired_schedule": True, "held_fold_selection": False,
        },
        "cache_dir": str(cache_dir), "cache_hashes": cache["hashes"],
        "assignment_csv": str(assignment_path),
        "class_names": list(EXPECTED_CLASSES), "class_counts": list(EXPECTED_CLASS_COUNTS),
        "screen": screen, "bootstrap": bootstrap, "readiness": readiness,
        "elapsed_seconds": time.perf_counter() - started,
        "train_split_used": True, "validation_split_used": False, "test_split_used": False,
        "validation_dataset_constructed": False, "test_dataset_constructed": False,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_screen(_parse_args(argv))
    print(json.dumps({
        "train_only_screen_passed": summary["readiness"]["train_only_screen_passed"],
        "exact_b11_closed": summary["readiness"]["exact_b11_closed"],
        "full_validation_permission": False, "test_permission": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

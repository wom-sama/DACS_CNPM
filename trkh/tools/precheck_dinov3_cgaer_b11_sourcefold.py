from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trkh.models.dinov3_cgaer_bridge_b11 import (
    CGAER_B11_CANDIDATE_MODE,
    CGAER_B11_CONTROL_MODE,
    CGAER_B11_PARAMETER_COUNT,
    CGAER_B11_RESIDUAL_L2_CAP,
    DinoV3CGAERBridgeB11,
)
from trkh.tools.export_timm_predictions import _state_dict_sha256
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import _array_finite, _sha256


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B11_CGAER_SOURCEFOLD_20260802"
SEED = 20260731
EXPECTED_SAMPLES = 8_278
EXPECTED_COUNTS = (1_987, 497, 1_326, 2_080, 2_388)
EXPECTED_CACHE_HASHES = {
    "manifest": "b3870c201dfee4437988f74f3a7ef900406d07b44c02ccae54fe6a2351690e49",
    "tokens": "7d720ef0f6897d5339ac060ac0cc70d8e718621aaee00e473108c91fe8304dbd",
    "logits": "643cd63059df171a48770f7f7e9b15ccbc9df7001d09da2bb1200db4048d2f22",
    "labels": "97a79688410600e6808140a441ddbfaaafddaf2294ffbbf77849adf0cd337a55",
    "paths": "a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb",
}
EXPECTED_ASSIGNMENT_CSV_SHA256 = (
    "afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f"
)
EXPECTED_CACHE_ORDER_FOLD_SHA256 = (
    "fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b"
)
EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256 = (
    "0f3856bf6020ba31d8564a48394e2a6fba7ad6f415e279e0063c5baadff20cb7"
)
EXPECTED_CACHE_ORDER_GROUP_SHA256 = (
    "1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2"
)
CACHE_FILES = {
    "manifest": "cache_manifest.json",
    "tokens": "train_postnorm_patch_tokens_f32.npy",
    "logits": "train_base_logits_f32.npy",
    "labels": "train_labels_i64.npy",
    "paths": "train_paths.json",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRAIN-only structural/readiness preflight for locked B11 CGAER."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refuse to overwrite preflight artifact: {path}")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _source_hashes() -> Dict[str, str]:
    tool = Path(__file__).resolve()
    model = tool.parents[1] / "models" / "dinov3_cgaer_bridge_b11.py"
    return {"preflight_tool_sha256": _sha256(tool), "model_sha256": _sha256(model)}


def _load_cache(cache_dir: Path) -> Dict[str, object]:
    root = Path(cache_dir).expanduser().resolve()
    files = {key: root / name for key, name in CACHE_FILES.items()}
    missing = [key for key, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"immutable A0 cache is incomplete: {missing}")
    hashes = {key: _sha256(path) for key, path in files.items()}
    drift = {
        key: {"expected": EXPECTED_CACHE_HASHES[key], "observed": value}
        for key, value in hashes.items()
        if value != EXPECTED_CACHE_HASHES[key]
    }
    if drift:
        raise ValueError(f"immutable A0 cache hash mismatch: {drift}")
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    required = {
        "token_shape": [EXPECTED_SAMPLES, 256, 384],
        "logit_shape": [EXPECTED_SAMPLES, 5],
        "label_shape": [EXPECTED_SAMPLES],
        "token_dtype": "float32",
        "logit_dtype": "float32",
        "label_dtype": "int64",
        "class_counts": list(EXPECTED_COUNTS),
        "lossy_token_quantization": False,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
    }
    bad = {key: [value, manifest.get(key)] for key, value in required.items() if manifest.get(key) != value}
    if bad:
        raise ValueError(f"immutable A0 manifest contract mismatch: {bad}")
    paths = json.loads(files["paths"].read_text(encoding="utf-8"))
    if not isinstance(paths, list) or len(paths) != EXPECTED_SAMPLES:
        raise ValueError("cached TRAIN path list changed")
    if any("/val/" in f"/{str(value).casefold()}/" or "/test/" in f"/{str(value).casefold()}/" for value in paths):
        raise ValueError("cache is not TRAIN-only")
    tokens = np.load(files["tokens"], mmap_mode="r", allow_pickle=False)
    logits = np.load(files["logits"], mmap_mode="r", allow_pickle=False)
    labels = np.load(files["labels"], mmap_mode="r", allow_pickle=False)
    if tokens.shape != (EXPECTED_SAMPLES, 256, 384) or tokens.dtype != np.float32:
        raise ValueError("token cache shape/dtype changed")
    if logits.shape != (EXPECTED_SAMPLES, 5) or logits.dtype != np.float32:
        raise ValueError("logit cache shape/dtype changed")
    if labels.shape != (EXPECTED_SAMPLES,) or labels.dtype != np.int64:
        raise ValueError("label cache shape/dtype changed")
    if not _array_finite(tokens) or not _array_finite(logits):
        raise ValueError("cache contains non-finite values")
    if tuple(np.bincount(labels, minlength=5).tolist()) != EXPECTED_COUNTS:
        raise ValueError("cached class support changed")
    return {"tokens": tokens, "logits": logits, "labels": labels, "paths": paths, "hashes": hashes}


def _pool_tokens(tokens: np.ndarray, chunk: int = 32) -> np.ndarray:
    if int(chunk) <= 0:
        raise ValueError("pooling chunk must be positive")
    pooled = np.full((int(tokens.shape[0]), 384), np.nan, dtype=np.float32)
    for start in range(0, int(tokens.shape[0]), int(chunk)):
        pooled[start : start + chunk] = np.asarray(
            tokens[start : start + chunk], dtype=np.float32
        ).mean(axis=1, dtype=np.float32)
    if not np.isfinite(pooled).all():
        raise RuntimeError("pooled feature cache is incomplete")
    return pooled


def _read_assignment(path: Path, paths: Sequence[str], labels: np.ndarray) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    csv_sha = _sha256(resolved)
    if csv_sha != EXPECTED_ASSIGNMENT_CSV_SHA256:
        raise ValueError("assignment CSV hash mismatch")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != ["relative_path", "label", "union_group", "fold"]:
            raise ValueError("assignment CSV columns changed")
        rows = list(reader)
    if len(rows) != len(paths) or any(
        row["relative_path"] != paths[index] or int(row["label"]) != int(labels[index])
        for index, row in enumerate(rows)
    ):
        raise ValueError("assignment rows do not align to cached TRAIN order")
    groups = np.asarray([int(row["union_group"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    fold_sha = hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest()
    group_sha = hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest()
    path_fold = "\n".join(f"{paths[i]}\t{int(folds[i])}" for i in range(len(paths))) + "\n"
    path_fold_sha = hashlib.sha256(path_fold.encode("utf-8")).hexdigest()
    if (fold_sha, path_fold_sha, group_sha) != (
        EXPECTED_CACHE_ORDER_FOLD_SHA256,
        EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256,
        EXPECTED_CACHE_ORDER_GROUP_SHA256,
    ):
        raise ValueError("cache-order fold/group identity changed")
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise ValueError("five locked folds are absent")
    fold_rows = []
    for fold in range(5):
        held = folds == fold
        if set(groups[held].tolist()) & set(groups[~held].tolist()):
            raise ValueError(f"component crosses fold {fold}")
        if bool((np.bincount(labels[held], minlength=5) <= 0).any()):
            raise ValueError(f"fold {fold} lacks a class")
        fold_rows.append(
            {
                "fold": fold,
                "fit_samples": int((~held).sum()),
                "held_samples": int(held.sum()),
                "held_class_counts": np.bincount(
                    labels[held], minlength=5
                ).astype(int).tolist(),
                "group_overlap": 0,
            }
        )
    return {
        "rows": rows,
        "groups": groups,
        "folds": folds,
        "fold_rows": fold_rows,
        "csv_sha256": csv_sha,
        "fold_vector_sha256": fold_sha,
        "assignment_int64_sha256": fold_sha,
        "path_fold_sha256": path_fold_sha,
        "group_vector_sha256": group_sha,
        "group_vector_int64_sha256": group_sha,
    }


def component_conflict_targets(labels: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if labels.ndim != 1 or labels.shape != groups.shape:
        raise ValueError("labels/groups are not aligned")
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
        counts = np.bincount(labels[positions], minlength=5)
        represented = counts[counts > 0].astype(np.float64)
        if represented.size > 1:
            p = represented / represented.sum()
            value = (1.0 - float(np.square(p).sum())) / (1.0 - 1.0 / represented.size)
            targets[positions] = np.float32(np.clip(value, 0.0, 1.0))
            mixed[positions] = True
    return {"targets": targets, "eligible": eligible, "mixed": mixed, "inverse_sizes": inverse_sizes}


def _paired_bridges() -> Tuple[DinoV3CGAERBridgeB11, DinoV3CGAERBridgeB11]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        candidate = DinoV3CGAERBridgeB11(CGAER_B11_CANDIDATE_MODE)
        control = DinoV3CGAERBridgeB11(CGAER_B11_CONTROL_MODE)
    control.load_state_dict(candidate.state_dict(), strict=True)
    return candidate, control


def _jacobian_first_two(model: DinoV3CGAERBridgeB11) -> np.ndarray:
    model = model.double()
    raw = torch.zeros(1, 5, dtype=torch.float64, requires_grad=True)
    base = torch.tensor([[0.4, 0.3, 0.1, -0.2, -0.4]], dtype=torch.float64)
    residual = model.uncapped_residual_from_raw_outputs(raw, base)[:, :3]
    rows = [torch.autograd.grad(residual[0, index], raw, retain_graph=True)[0][0, :2] for index in range(3)]
    return torch.stack(rows).detach().cpu().numpy()


def _architecture_smoke(features: np.ndarray, base_logits: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    candidate, control = _paired_bridges()
    state_equal = all(torch.equal(value, control.state_dict()[key]) for key, value in candidate.state_dict().items())
    z = torch.from_numpy(np.asarray(features[:16], dtype=np.float32))
    base = torch.from_numpy(
        np.array(base_logits[:16], dtype=np.float32, copy=True)
    )
    target = torch.from_numpy(np.array(labels[:16], dtype=np.int64, copy=True))
    with torch.no_grad():
        active_candidate, trace = candidate(z, base, return_trace=True)
        active_control = control(z, base)
        off = candidate(z, base, branch_off=True)
        huge = torch.full((16, 5), 100.0)
        capped = candidate.residual_from_raw_outputs(huge, base)
    active_error = max(
        float((active_candidate - base).abs().max()),
        float((active_control - base).abs().max()),
    )
    branch_error = float((off - base).abs().max())
    cap_max = float(torch.linalg.vector_norm(capped, dim=1).max())
    candidate_j = _jacobian_first_two(candidate)
    control_j = _jacobian_first_two(control)
    jacobian_error = float(np.max(np.abs(candidate_j - control_j)))
    candidate = candidate.float()
    z.requires_grad_(True)
    base.requires_grad_(True)
    logits, grad_trace = candidate(z, base, return_trace=True)
    loss = F.cross_entropy(logits, target) + 0.1 * F.binary_cross_entropy_with_logits(
        grad_trace["conflict_logit"], torch.zeros_like(grad_trace["conflict_logit"])
    )
    loss.backward()
    gradients = [parameter.grad for parameter in candidate.parameters()]
    finite_gradients = all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
    checks = {
        "parameter_count": candidate.added_parameter_count() == control.added_parameter_count() == CGAER_B11_PARAMETER_COUNT,
        "identical_initial_state": state_equal,
        "active_zero_parity": active_error == 0.0,
        "branch_off_exact": branch_error == 0.0,
        "jacobian_match": jacobian_error <= 1e-7,
        "residual_cap": cap_max < CGAER_B11_RESIDUAL_L2_CAP,
        "head_gradients_finite": finite_gradients,
        "input_and_base_detached": z.grad is None and base.grad is None,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parameter_count_each": candidate.added_parameter_count(),
        "paired_initial_state_sha256": _state_dict_sha256(candidate.state_dict()),
        "active_zero_max_abs_error": active_error,
        "branch_off_max_abs_error": branch_error,
        "candidate_control_jacobian_max_abs_error": jacobian_error,
        "candidate_jacobian_singular_values": np.linalg.svd(candidate_j, compute_uv=False).tolist(),
        "control_jacobian_singular_values": np.linalg.svd(control_j, compute_uv=False).tolist(),
        "stress_residual_l2_max": cap_max,
        "initial_gate_mean": float(trace["gate"].mean()),
    }


def _readiness_diagnostic(
    features: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    conflict: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    design = np.concatenate((features, np.asarray(base_logits, dtype=np.float32)), axis=1)
    mixed = np.asarray(conflict["mixed"], dtype=np.int64)
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    for fold in range(5):
        fit, held = folds != fold, folds == fold
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=1_000,
                class_weight="balanced",
                solver="liblinear",
                random_state=SEED,
            ),
        )
        estimator.fit(design[fit], mixed[fit])
        scores[held] = estimator.predict_proba(design[held])[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("readiness OOF scores are incomplete")
    eligible = np.asarray(conflict["eligible"], dtype=bool)
    prediction = np.asarray(base_logits).argmax(axis=1)
    boundary_error = np.logical_or(
        np.logical_and(labels == 1, prediction != 1),
        np.logical_and(labels != 1, prediction == 1),
    )
    boundary = np.logical_or(labels == 1, prediction == 1)
    conditional_error = prediction[boundary] != labels[boundary]
    mixed_auc = float(roc_auc_score(mixed[eligible], scores[eligible]))
    error_auc = float(roc_auc_score(boundary_error, scores))
    conditional_auc = float(roc_auc_score(conditional_error, scores[boundary]))
    return {
        "status": "non_authorizing_fixed_linear_train_only_readiness",
        "input": "mean_postnorm_B9_feature_plus_B9_logits",
        "folds": 5,
        "seed": SEED,
        "mixed_vs_pure_auroc": mixed_auc,
        "b9_class1_boundary_error_auroc": error_auc,
        "b9_class1_boundary_error_conditional_auroc": conditional_auc,
        "mixed_samples": int(mixed.sum()),
        "boundary_errors": int(boundary_error.sum()),
        "passed": mixed_auc >= 0.75 and error_auc >= 0.75,
    }


def validate_runner_preflight(path: Path, *, expected_sha256: str) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    observed_sha = _sha256(resolved)
    if observed_sha != str(expected_sha256).strip().casefold():
        raise ValueError("B11 preflight artifact SHA-256 mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("B11 preflight artifact must be a JSON object")
    locked = payload.get("locked_inputs", {})
    expected_inputs = {
        "a0_cache_manifest_sha256": EXPECTED_CACHE_HASHES["manifest"],
        "a0_tokens_sha256": EXPECTED_CACHE_HASHES["tokens"],
        "a0_logits_sha256": EXPECTED_CACHE_HASHES["logits"],
        "a0_labels_sha256": EXPECTED_CACHE_HASHES["labels"],
        "a0_paths_sha256": EXPECTED_CACHE_HASHES["paths"],
        "assignment_csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
    }
    checks = {
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "passed": payload.get("preflight_passed") is True,
        "sources_current": payload.get("source_files") == _source_hashes(),
        "locked_inputs": isinstance(locked, Mapping) and all(locked.get(key) == value for key, value in expected_inputs.items()),
        "cache_order_folds": isinstance(locked, Mapping)
        and locked.get("cache_order_fold_sha256") == EXPECTED_CACHE_ORDER_FOLD_SHA256
        and locked.get("cache_order_path_fold_sha256") == EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256
        and locked.get("cache_order_group_sha256") == EXPECTED_CACHE_ORDER_GROUP_SHA256,
        "architecture": payload.get("architecture_smoke", {}).get("passed") is True,
        "readiness": payload.get("readiness_diagnostic", {}).get("passed") is True,
        "train_only": payload.get("train_split_used") is True
        and payload.get("validation_split_used") is False
        and payload.get("test_split_used") is False,
        "no_candidate_oof": payload.get("candidate_oof_training_run") is False,
        "permissions_closed": payload.get("full_validation_permission") is False
        and payload.get("full_train_permission") is False
        and payload.get("test_permission") is False,
    }
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise ValueError(f"B11 preflight artifact is stale/unsafe: {failed}")
    return payload


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir == cache_dir or cache_dir in output_dir.parents:
        raise RuntimeError("preflight output must be a new directory outside immutable cache")
    cache = _load_cache(cache_dir)
    assignment = _read_assignment(args.assignment_csv, cache["paths"], cache["labels"])
    pooled = _pool_tokens(cache["tokens"])
    conflict = component_conflict_targets(cache["labels"], assignment["groups"])
    architecture = _architecture_smoke(pooled, cache["logits"], cache["labels"])
    readiness = _readiness_diagnostic(
        pooled, cache["logits"], cache["labels"], assignment["folds"], conflict
    )
    counts = Counter(assignment["groups"].tolist())
    mixed_groups = len(set(assignment["groups"][conflict["mixed"]].tolist()))
    checks = {
        "architecture": architecture["passed"],
        "readiness": readiness["passed"],
        "mixed_target_present": int(conflict["mixed"].sum()) == 1_604,
        "component_count": len(counts) == 3_117,
        "mixed_component_count": mixed_groups == 145,
        "singletons_masked": all(
            not bool(conflict["eligible"][index])
            for index, group in enumerate(assignment["groups"])
            if counts[int(group)] == 1
        ),
    }
    locked_inputs = {
        "a0_cache_manifest_sha256": cache["hashes"]["manifest"],
        "a0_tokens_sha256": cache["hashes"]["tokens"],
        "a0_logits_sha256": cache["hashes"]["logits"],
        "a0_labels_sha256": cache["hashes"]["labels"],
        "a0_paths_sha256": cache["hashes"]["paths"],
        "assignment_csv_sha256": assignment["csv_sha256"],
        "cache_order_fold_sha256": assignment["fold_vector_sha256"],
        "cache_order_path_fold_sha256": assignment["path_fold_sha256"],
        "cache_order_group_sha256": assignment["group_vector_sha256"],
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "preflight_passed": all(checks.values()),
        "checks": checks,
        "source_files": _source_hashes(),
        "locked_inputs": locked_inputs,
        "architecture_smoke": architecture,
        "readiness_diagnostic": readiness,
        "component_contract": {
            "components": len(counts),
            "mixed_components": mixed_groups,
            "mixed_samples": int(conflict["mixed"].sum()),
            "eligible_non_singleton_samples": int(conflict["eligible"].sum()),
            "target_min": float(conflict["targets"].min()),
            "target_max": float(conflict["targets"].max()),
            "ambiguity_weighting": "inverse_component_size_then_normalize",
        },
        "future_locked_oof": {
            "folds": 5,
            "seed": SEED,
            "epochs": 5,
            "optimizer": "AdamW",
            "learning_rate": 1.5e-4,
            "weight_decay": 0.0,
            "batch_size": 128,
            "tempered_power": 0.5,
            "bootstrap_replicates": 5_000,
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
        "candidate_oof_training_run": False,
        "full_validation_permission": False,
        "full_train_permission": False,
        "test_permission": False,
    }
    if not payload["preflight_passed"]:
        raise RuntimeError(f"B11 preflight failed: {[key for key, value in checks.items() if not value]}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "preflight.json", payload)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    payload = run_preflight(_parse_args(argv))
    print(json.dumps({
        "preflight_passed": payload["preflight_passed"],
        "mixed_vs_pure_auroc": payload["readiness_diagnostic"]["mixed_vs_pure_auroc"],
        "boundary_error_auroc": payload["readiness_diagnostic"]["b9_class1_boundary_error_auroc"],
        "full_validation_permission": False,
        "test_permission": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

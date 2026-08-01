from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from trkh.models.pretrained_semantic_branch import load_verified_local_timm_model
from trkh.tools.audit_dinov3_pair_patch_stat_readiness import (
    _assert_output_outside_train,
    _sha256,
    _write_csv,
    normalized_source_group,
)
from trkh.tools.frozen_dino_np_core import (
    DESCRIPTOR_WIDTH,
    MIN_CALIBRATION_GROUPS,
    PATCH_TOKENS,
    PREFIX_TOKENS,
    RISK_ALPHA,
    RISK_DELTA,
    TOKEN_WIDTH,
    VARIANCE_FLOOR,
    class2_conformity_score,
    frozen_prefix_descriptor,
    maximum_group_threshold,
    maximum_order_tolerance_upper,
    readiness_from_metrics,
    selective_2to1_predictions,
    source_balanced_diagonal_gaussian,
)
from trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof import (
    _atomic_json,
    _labels_sha256,
    _path_rows_sha256,
    _source_groups_sha256,
)
from trkh.tools.precheck_dinov3_pair_patchstats_train_oof import (
    assign_global_source_folds,
)
from trkh.tools.precheck_dinov3_prefix_residual_train_oof import (
    _flush_and_close_memmaps,
    source_disjoint_block_derangement,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B8_FROZEN_DINO_NP_2TO1_20260801"
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
DINO_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DINO_SOURCE_URL = "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m"
DINO_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINO_LICENSE = "dinov3-license"
EXPECTED_DINO_PARAMETERS = 21_586_944
EXPECTED_TIMM_VERSION = "1.0.27"
EXPECTED_DATA_SHA256 = "fa9581d1a595134bd366105c3999170d3553712099e7222cf67f2339f84da156"
EXPECTED_B2_SHA256 = "4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6"
EXPECTED_B2_LOGITS_SHA256 = "2a76d98a992e7e29a21295a581b4ae53f950fe2bb7df16c5fe75a568d64b816e"
EXPECTED_TRAIN_SAMPLES = 8278
EXPECTED_SOURCE_GROUPS = 7751
EXPECTED_B2_2TO1_ERRORS = 137
FOLDS = 5
SEED = 20260731
FOCUS_CLASS = 1
RIVAL_CLASS = 2
DESCRIPTOR_FILENAME = "frozen_dino_prefix_descriptor_f32.npy"
CACHE_MANIFEST_FILENAME = "descriptor_cache_manifest.json"
B2_LOGITS_FILENAME = "global_logits_f32.npy"
B2_LABELS_FILENAME = "labels_i64.npy"
B2_MANIFEST_FILENAME = "cache_manifest.json"
B2_PREFLIGHT_FILENAME = "preflight.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked B8 train-only frozen-DINO group-tolerance readiness audit. "
            "Validation and test datasets are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--b2-checkpoint", type=Path, required=True)
    parser.add_argument("--b2-cache-dir", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def _array_sha256(array: np.ndarray, *, dtype: str) -> str:
    values = np.asarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    target = Path(path)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial B8 artifact exists: {partial}")
    array = np.asarray(values)
    writer = np.lib.format.open_memmap(
        partial, mode="w+", dtype=array.dtype, shape=array.shape
    )
    writer[...] = array
    _flush_and_close_memmaps(writer)
    del writer
    partial.replace(target)


def _dataset_inventory(
    *,
    data_path: Path,
    b2_checkpoint_path: Path,
    class_name_mode: str,
) -> Tuple[
    Dataset,
    Mapping[str, object],
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, object]],
]:
    if _sha256(data_path) != EXPECTED_DATA_SHA256:
        raise ValueError("Locked B8 development YAML hash mismatch")
    if _sha256(b2_checkpoint_path) != EXPECTED_B2_SHA256:
        raise ValueError("Locked B8 B2 checkpoint hash mismatch")
    checkpoint = torch.load(
        b2_checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("B2 checkpoint payload must be a mapping")
    dataset, class_names = _build_dataset(
        data_yaml=data_path,
        split="train",
        checkpoint=checkpoint,
        class_name_mode=class_name_mode,
        max_samples=0,
    )
    if len(dataset) != EXPECTED_TRAIN_SAMPLES or len(class_names) != 5:
        raise ValueError("Locked B8 canonical train/class support changed")
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    labels_fn = getattr(dataset, "labels", None)
    if not callable(sample_paths_fn) or not callable(labels_fn):
        raise TypeError("B8 train dataset must expose sample_paths() and labels()")
    paths = [str(path) for path in sample_paths_fn()]
    labels = np.asarray(labels_fn(), dtype=np.int64).reshape(-1)
    groups = np.asarray(
        [normalized_source_group(path) for path in paths], dtype=object
    )
    if (
        len(paths) != EXPECTED_TRAIN_SAMPLES
        or labels.size != EXPECTED_TRAIN_SAMPLES
        or int(np.unique(groups).size) != EXPECTED_SOURCE_GROUPS
        or set(np.unique(labels).tolist()) != set(range(5))
    ):
        raise ValueError("B8 canonical row/label/source-group contract changed")
    assignments, fold_rows = assign_global_source_folds(
        labels, groups, folds=FOLDS, seed=SEED
    )
    return (
        dataset,
        checkpoint,
        paths,
        labels,
        groups,
        assignments,
        fold_rows,
    )


def _load_verified_b2_outputs(
    *,
    cache_dir: Path,
    paths: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
    assignments: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    manifest_path = cache_dir / B2_MANIFEST_FILENAME
    preflight_path = cache_dir / B2_PREFLIGHT_FILENAME
    logits_path = cache_dir / B2_LOGITS_FILENAME
    labels_path = cache_dir / B2_LABELS_FILENAME
    for path in (manifest_path, preflight_path, logits_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _load_json(manifest_path)
    preflight = _load_json(preflight_path)
    expected = {
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_B2_SHA256,
        "paths_sha256": _path_rows_sha256(paths),
        "labels_content_sha256": _labels_sha256(labels),
        "source_groups_sha256": _source_groups_sha256(groups.tolist()),
        "source_group_count": EXPECTED_SOURCE_GROUPS,
        "global_logits_shape": [EXPECTED_TRAIN_SAMPLES, 5],
        "global_logits_dtype": "float32",
        "labels_shape": [EXPECTED_TRAIN_SAMPLES],
        "labels_dtype": "int64",
        "global_logits_sha256": EXPECTED_B2_LOGITS_SHA256,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B8 rejected B2 cache manifest: {mismatches}")
    if _sha256(logits_path) != EXPECTED_B2_LOGITS_SHA256:
        raise ValueError("B8 B2 global-logit file hash mismatch")
    expected_fold_hash = _array_sha256(assignments, dtype="<i8")
    if preflight.get("fold_assignments_sha256") != expected_fold_hash:
        raise ValueError("B8 folds differ from the locked B7/B2-cache folds")
    if any(
        bool(preflight.get(key, True))
        for key in (
            "validation_split_used",
            "test_split_used",
            "validation_dataset_constructed",
            "test_dataset_constructed",
        )
    ):
        raise ValueError("B8 B2-cache provenance indicates val/test access")
    cached_labels = np.load(labels_path, allow_pickle=False)
    if cached_labels.dtype != np.int64 or not np.array_equal(cached_labels, labels):
        raise ValueError("B8 B2 cached labels differ from canonical labels")
    logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
    if logits.shape != (EXPECTED_TRAIN_SAMPLES, 5) or logits.dtype != np.float32:
        raise ValueError("B8 B2 logit array contract changed")
    if not bool(np.isfinite(np.asarray(logits)).all()):
        raise ValueError("B8 B2 logits contain non-finite values")
    return logits, {
        "cache_dir": str(cache_dir),
        "manifest_sha256": _sha256(manifest_path),
        "preflight_sha256": _sha256(preflight_path),
        "global_logits_sha256": _sha256(logits_path),
        "labels_sha256": _sha256(labels_path),
        "fold_assignments_sha256": expected_fold_hash,
        "b7_prefix_or_patch_arrays_used": False,
        "usage": "B2 logits only for base/runner-up routing and effect accounting",
    }


def _load_frozen_dino(
    checkpoint_path: Path,
) -> Tuple[torch.nn.Module, Dict[str, object], Dict[str, object]]:
    model, provenance = load_verified_local_timm_model(
        model_name=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        expected_sha256=DINO_SHA256,
        source_repository=DINO_SOURCE_URL,
        source_revision=DINO_REVISION,
        license_id=DINO_LICENSE,
        model_kwargs={},
        strict=True,
    )
    model.requires_grad_(False).eval()
    import timm

    with torch.inference_mode():
        tokens = model.forward_features(
            torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        )
        descriptor = frozen_prefix_descriptor(tokens)
    contract = {
        "runtime_class_module": type(model).__module__,
        "runtime_class_name": type(model).__name__,
        "timm_version": str(timm.__version__),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 0) or 0),
        "token_width": int(getattr(model, "num_features", 0) or 0),
        "patch_tokens": int(getattr(getattr(model, "patch_embed", None), "num_patches", 0) or 0),
        "synthetic_tokens_shape": list(tokens.shape),
        "synthetic_descriptor_shape": list(descriptor.shape),
        "synthetic_finite": bool(
            torch.isfinite(tokens).all() and torch.isfinite(descriptor).all()
        ),
        "all_parameters_frozen": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
    }
    expected = {
        "runtime_class_module": "timm.models.eva",
        "runtime_class_name": "Eva",
        "timm_version": EXPECTED_TIMM_VERSION,
        "parameters": EXPECTED_DINO_PARAMETERS,
        "trainable_parameters": 0,
        "num_prefix_tokens": PREFIX_TOKENS,
        "token_width": TOKEN_WIDTH,
        "patch_tokens": PATCH_TOKENS,
        "synthetic_tokens_shape": [1, PREFIX_TOKENS + PATCH_TOKENS, TOKEN_WIDTH],
        "synthetic_descriptor_shape": [1, DESCRIPTOR_WIDTH],
        "synthetic_finite": True,
        "all_parameters_frozen": True,
    }
    mismatches = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B8 original-DINO runtime mismatch: {mismatches}")
    return model, dict(provenance), contract


def _descriptor_manifest_expected(
    *,
    paths: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
    assignments: np.ndarray,
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "dino_checkpoint_sha256": DINO_SHA256,
        "paths_sha256": _path_rows_sha256(paths),
        "labels_content_sha256": _labels_sha256(labels),
        "source_groups_sha256": _source_groups_sha256(groups.tolist()),
        "fold_assignments_sha256": _array_sha256(assignments, dtype="<i8"),
        "source_group_count": EXPECTED_SOURCE_GROUPS,
        "descriptor_shape": [EXPECTED_TRAIN_SAMPLES, DESCRIPTOR_WIDTH],
        "descriptor_dtype": "float32",
        "descriptor_contract": (
            "concat(LN_no_affine(cls-patchmean),"
            "mean_register_LN_residual,popvar_register_LN_residual)"
        ),
        "b2_logits_in_descriptor_or_score": False,
        "original_dino_frozen": True,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }


def _load_or_extract_descriptors(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    output_dir: Path,
    paths: Sequence[str],
    labels: np.ndarray,
    groups: np.ndarray,
    assignments: np.ndarray,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cache_path = output_dir / DESCRIPTOR_FILENAME
    manifest_path = output_dir / CACHE_MANIFEST_FILENAME
    expected = _descriptor_manifest_expected(
        paths=paths, labels=labels, groups=groups, assignments=assignments
    )
    if cache_path.exists() or manifest_path.exists():
        if not (cache_path.is_file() and manifest_path.is_file()):
            raise RuntimeError("Partial B8 descriptor cache exists")
        manifest = dict(_load_json(manifest_path))
        mismatches = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches or manifest.get("descriptor_sha256") != _sha256(cache_path):
            raise ValueError(f"B8 descriptor-cache mismatch: {mismatches}")
        values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (EXPECTED_TRAIN_SAMPLES, DESCRIPTOR_WIDTH) or values.dtype != np.float32:
            raise ValueError("B8 descriptor-cache array contract changed")
        return values, manifest
    partial = cache_path.with_suffix(cache_path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial B8 descriptor cache exists: {partial}")
    writer = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_TRAIN_SAMPLES, DESCRIPTOR_WIDTH),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    cursor = 0
    model.to(device=device, dtype=torch.float32).eval()
    with torch.inference_mode():
        for batch_index, (images, observed_labels, _metadata) in enumerate(loader):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            tokens = model.forward_features(images)
            descriptor = frozen_prefix_descriptor(tokens)
            values = descriptor.cpu().numpy().astype(np.float32, copy=False)
            count = int(values.shape[0])
            end = cursor + count
            if not np.array_equal(
                observed_labels.numpy().astype(np.int64, copy=False), labels[cursor:end]
            ):
                raise ValueError("B8 extraction label order changed")
            writer[cursor:end] = values
            cursor = end
            if batch_index % 25 == 0 or cursor == EXPECTED_TRAIN_SAMPLES:
                print(
                    f"B8 frozen-DINO descriptor cache: {cursor}/{EXPECTED_TRAIN_SAMPLES}",
                    flush=True,
                )
    if cursor != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError("B8 descriptor extraction coverage is incomplete")
    _flush_and_close_memmaps(writer)
    del writer
    partial.replace(cache_path)
    values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if not bool(np.isfinite(np.asarray(values)).all()):
        raise FloatingPointError("B8 descriptor cache contains non-finite values")
    manifest = {
        **expected,
        "descriptor_sha256": _sha256(cache_path),
        "descriptor_bytes": int(cache_path.stat().st_size),
    }
    _atomic_json(manifest_path, manifest)
    return values, manifest


def _derangement_map(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    assignments: np.ndarray,
    fold: int,
    class2_excluded: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    donors = np.full(labels.size, -1, dtype=np.int64)
    excluded = np.asarray(class2_excluded, dtype=bool).reshape(-1)
    if excluded.size != labels.size or bool(
        np.any(np.logical_and(excluded, labels != RIVAL_CLASS))
    ):
        raise ValueError("B8 class2 exclusion mask is invalid")
    contracts = 0
    fit = assignments != int(fold)
    hold = assignments == int(fold)
    # The deranged control must not move a descriptor across any scientific
    # role.  In particular, class-2 rows excluded for sharing a source with
    # class-1 calibration cannot donate back into the valid class-2 fit.
    cohorts: List[Tuple[str, np.ndarray]] = []
    for label in range(5):
        cohorts.append(
            (f"hold_label_{label}", np.logical_and(hold, labels == label))
        )
    for label in (0, 1, 3, 4):
        cohorts.append(
            (f"fit_label_{label}", np.logical_and(fit, labels == label))
        )
    cohorts.extend(
        (
            (
                "fit_label_2_valid",
                np.logical_and.reduce((fit, labels == RIVAL_CLASS, ~excluded)),
            ),
            (
                "fit_label_2_cross_label_excluded",
                np.logical_and.reduce((fit, labels == RIVAL_CLASS, excluded)),
            ),
        )
    )
    cohort_rows: List[Dict[str, object]] = []
    for cohort_index, (name, cohort_mask) in enumerate(cohorts):
        indices = np.flatnonzero(cohort_mask)
        if indices.size == 0:
            cohort_rows.append(
                {"role": name, "rows": 0, "source_groups": 0}
            )
            continue
        mapped = source_disjoint_block_derangement(
            indices,
            groups,
            seed=SEED + 1_000_003 * int(fold) + 10_007 * cohort_index,
        )
        donors[indices] = mapped
        contracts += 1
        cohort_rows.append(
            {
                "role": name,
                "rows": int(indices.size),
                "source_groups": int(np.unique(groups[indices]).size),
            }
        )
    if bool((donors < 0).any()) or set(donors.tolist()) != set(range(labels.size)):
        raise RuntimeError("B8 derangement is incomplete or non-bijective")
    fixed = int(np.sum(donors == np.arange(labels.size)))
    same_source = int(
        sum(str(groups[index]) == str(groups[donors[index]]) for index in range(labels.size))
    )
    label_mismatch = int(np.sum(labels[donors] != labels))
    fold_crossing = int(
        np.sum((assignments[donors] == int(fold)) != (assignments == int(fold)))
    )
    exclusion_role_crossing = int(np.sum(excluded[donors] != excluded))
    if fixed or same_source or label_mismatch or fold_crossing or exclusion_role_crossing:
        raise RuntimeError("B8 derangement integrity failed")
    return donors, {
        "fold": int(fold),
        "contracts": contracts,
        "fixed_points": fixed,
        "same_source_assignments": same_source,
        "label_mismatches": label_mismatch,
        "fold_boundary_crossings": fold_crossing,
        "class2_exclusion_role_crossings": exclusion_role_crossing,
        "mapping_sha256": _array_sha256(donors, dtype="<i8"),
        "whole_descriptor_block": True,
        "score_and_b2_correctness_blind": True,
        "cohorts": cohort_rows,
    }


def _mode_metrics(
    *,
    labels: np.ndarray,
    base: np.ndarray,
    runner: np.ndarray,
    action: np.ndarray,
    final: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, object]:
    selected = np.asarray(mask, dtype=bool)
    y = labels[selected]
    b = base[selected]
    r = runner[selected]
    a = action[selected]
    p = final[selected]
    errors_2to1 = int(np.sum(np.logical_and(y == 2, b == 1)))
    corrected_2to1 = int(np.sum(np.logical_and.reduce((y == 2, b == 1, a))))
    base_tp = int(np.sum(np.logical_and(y == 1, b == 1)))
    final_tp = int(np.sum(np.logical_and(y == 1, p == 1)))
    support1 = int(np.sum(y == 1))
    changed = p != b
    legal_domain = np.logical_and.reduce((b == 1, r == 2, p == 2))
    return {
        "rows": int(y.size),
        "base_2to1_errors": errors_2to1,
        "corrected_2to1": corrected_2to1,
        "fp_2to1_reduction": (
            float(corrected_2to1 / errors_2to1) if errors_2to1 else 0.0
        ),
        "actions": int(a.sum()),
        "true_class2_actions": int(np.sum(np.logical_and(a, y == 2))),
        "harmful_class1_actions": int(np.sum(np.logical_and(a, y == 1))),
        "class1_support": support1,
        "base_class1_tp": base_tp,
        "final_class1_tp": final_tp,
        "class1_tp_retention": float(final_tp / base_tp) if base_tp else 1.0,
        "base_class1_recall": float(base_tp / support1) if support1 else 0.0,
        "final_class1_recall": float(final_tp / support1) if support1 else 0.0,
        "class1_recall_delta": float((final_tp - base_tp) / support1) if support1 else 0.0,
        "new_0to1_or_4to1": int(
            np.sum(
                np.logical_and.reduce(
                    (np.isin(y, [0, 4]), b != 1, p == 1)
                )
            )
        ),
        "outside_domain_changes": int(np.sum(np.logical_and(changed, ~legal_domain))),
    }


def _persist_fold_parameters(
    *,
    output_dir: Path,
    fold: int,
    mode: str,
    mean: np.ndarray,
    variance: np.ndarray,
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    root = output_dir / "fold_parameters"
    root.mkdir(parents=True, exist_ok=True)
    mean_path = root / f"fold_{fold}_{mode}_mean_f32.npy"
    variance_path = root / f"fold_{fold}_{mode}_variance_f32.npy"
    metadata_path = root / f"fold_{fold}_{mode}_threshold.json"
    _atomic_npy(mean_path, np.asarray(mean, dtype=np.float32))
    _atomic_npy(variance_path, np.asarray(variance, dtype=np.float32))
    frozen_metadata = {
        **dict(metadata),
        "mean_sha256": _sha256(mean_path),
        "variance_sha256": _sha256(variance_path),
        "holdout_effect_metrics_read_before_freeze": False,
    }
    _atomic_json(metadata_path, frozen_metadata)
    return {
        "mean": str(mean_path),
        "mean_sha256": _sha256(mean_path),
        "variance": str(variance_path),
        "variance_sha256": _sha256(variance_path),
        "threshold": str(metadata_path),
        "threshold_sha256": _sha256(metadata_path),
    }


def run_oof_audit(
    *,
    descriptors: np.ndarray,
    b2_logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    assignments: np.ndarray,
    paths: Sequence[str],
    output_dir: Path,
) -> Tuple[
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    features = np.asarray(descriptors, dtype=np.float32)
    logits = np.asarray(b2_logits, dtype=np.float32)
    modes = ("aligned", "zero", "deranged")
    oof_scores = {mode: np.full(labels.size, np.nan, dtype=np.float64) for mode in modes}
    oof_actions = {mode: np.zeros(labels.size, dtype=bool) for mode in modes}
    oof_final = {mode: np.full(labels.size, -1, dtype=np.int64) for mode in modes}
    base_all, runner_all, _unused_action, _unused_final = selective_2to1_predictions(
        logits, np.zeros(labels.size, dtype=np.float64), math.inf
    )
    fold_rows: List[Dict[str, object]] = []
    parameter_artifacts: List[Dict[str, object]] = []
    derangement_reports: List[Dict[str, object]] = []
    minimum_calibration_groups = EXPECTED_TRAIN_SAMPLES
    maximum_tolerance_upper = 0.0
    overlap_excluded_groups: set[str] = set()
    for fold in range(FOLDS):
        fit = assignments != fold
        hold = assignments == fold
        calibration = np.logical_and(fit, labels == FOCUS_CLASS)
        calibration_groups = set(str(value) for value in groups[calibration].tolist())
        class2_fit_before = np.logical_and(fit, labels == RIVAL_CLASS)
        overlap = np.asarray(
            [str(value) in calibration_groups for value in groups], dtype=bool
        )
        class2_fit = np.logical_and(class2_fit_before, ~overlap)
        excluded_groups = set(
            str(value) for value in groups[np.logical_and(class2_fit_before, overlap)].tolist()
        )
        overlap_excluded_groups.update(excluded_groups)
        class2_excluded = np.logical_and(class2_fit_before, overlap)
        donor_map, derangement_report = _derangement_map(
            labels=labels,
            groups=groups,
            assignments=assignments,
            fold=fold,
            class2_excluded=class2_excluded,
        )
        derangement_reports.append(derangement_report)
        fold_mode_metrics: Dict[str, Dict[str, object]] = {}
        fold_thresholds: Dict[str, float] = {}
        for mode in modes:
            if mode == "aligned":
                mode_features = features
            elif mode == "zero":
                mode_features = np.zeros_like(features)
            else:
                mode_features = features[donor_map]
            mean, variance, fit_report = source_balanced_diagonal_gaussian(
                mode_features[class2_fit],
                groups[class2_fit],
                variance_floor=VARIANCE_FLOOR,
            )
            scores = class2_conformity_score(mode_features, mean, variance)
            threshold, calibration_report = maximum_group_threshold(
                scores[calibration], groups[calibration]
            )
            calibration_count = int(calibration_report["calibration_source_groups"])
            tolerance_upper = maximum_order_tolerance_upper(
                calibration_count, delta=RISK_DELTA
            )
            minimum_calibration_groups = min(minimum_calibration_groups, calibration_count)
            maximum_tolerance_upper = max(maximum_tolerance_upper, tolerance_upper)
            artifact = _persist_fold_parameters(
                output_dir=output_dir,
                fold=fold,
                mode=mode,
                mean=mean,
                variance=variance,
                metadata={
                    "protocol_id": PROTOCOL_ID,
                    "fold": fold,
                    "mode": mode,
                    "score_direction": "larger_is_more_class2_like",
                    "score_formula": "-mean((descriptor-mu)^2/variance)",
                    "variance_floor": VARIANCE_FLOOR,
                    "threshold": threshold,
                    "comparison": "score > threshold",
                    "risk_alpha": RISK_ALPHA,
                    "risk_delta": RISK_DELTA,
                    "tolerance_upper": tolerance_upper,
                    "fit": fit_report,
                    "calibration": calibration_report,
                    "class2_overlap_excluded_groups": sorted(excluded_groups),
                    "class2_overlap_excluded_rows": int(
                        np.sum(np.logical_and(class2_fit_before, overlap))
                    ),
                },
            )
            parameter_artifacts.append(
                {"fold": fold, "mode": mode, **artifact}
            )
            base, runner, action, final = selective_2to1_predictions(
                logits[hold], scores[hold], threshold
            )
            hold_indices = np.flatnonzero(hold)
            oof_scores[mode][hold_indices] = scores[hold]
            oof_actions[mode][hold_indices] = action
            oof_final[mode][hold_indices] = final
            fold_mode_metrics[mode] = _mode_metrics(
                labels=labels[hold],
                base=base,
                runner=runner,
                action=action,
                final=final,
                mask=np.ones(base.size, dtype=bool),
            )
            fold_thresholds[mode] = threshold
        fold_row: Dict[str, object] = {
            "fold": fold,
            "holdout_rows": int(hold.sum()),
            "class1_calibration_groups": int(np.unique(groups[calibration]).size),
            "tolerance_upper": maximum_order_tolerance_upper(
                int(np.unique(groups[calibration]).size), delta=RISK_DELTA
            ),
            "class2_fit_groups": int(np.unique(groups[class2_fit]).size),
            "class2_overlap_excluded_groups": len(excluded_groups),
            "class2_overlap_excluded_rows": int(
                np.sum(np.logical_and(class2_fit_before, overlap))
            ),
        }
        for mode in modes:
            metrics = fold_mode_metrics[mode]
            fold_row.update(
                {
                    f"{mode}_threshold": fold_thresholds[mode],
                    f"{mode}_corrected_2to1": metrics["corrected_2to1"],
                    f"{mode}_fp_2to1_reduction": metrics["fp_2to1_reduction"],
                    f"{mode}_class1_recall_delta": metrics["class1_recall_delta"],
                    f"{mode}_class1_tp_retention": metrics["class1_tp_retention"],
                    f"{mode}_actions": metrics["actions"],
                    f"{mode}_harmful_class1_actions": metrics["harmful_class1_actions"],
                }
            )
        fold_rows.append(fold_row)
    if any(
        np.isnan(oof_scores[mode]).any()
        or bool((oof_final[mode] < 0).any())
        for mode in modes
    ):
        raise RuntimeError("B8 OOF coverage is incomplete")
    aggregate = {
        mode: _mode_metrics(
            labels=labels,
            base=base_all,
            runner=runner_all,
            action=oof_actions[mode],
            final=oof_final[mode],
            mask=np.ones(labels.size, dtype=bool),
        )
        for mode in modes
    }
    prediction_rows: List[Dict[str, object]] = []
    for index in range(labels.size):
        prediction_rows.append(
            {
                "sample_index": index,
                "image_path": str(paths[index]),
                "source_group": str(groups[index]),
                "target_index": int(labels[index]),
                "fold": int(assignments[index]),
                "b2_prediction": int(base_all[index]),
                "b2_runner_up": int(runner_all[index]),
                **{
                    f"{mode}_score": float(oof_scores[mode][index])
                    for mode in modes
                },
                **{
                    f"{mode}_action": int(oof_actions[mode][index])
                    for mode in modes
                },
                **{
                    f"{mode}_prediction": int(oof_final[mode][index])
                    for mode in modes
                },
            }
        )
    integrity = {
        "complete_oof_rows": len(prediction_rows) == EXPECTED_TRAIN_SAMPLES,
        "expected_b2_2to1_errors": int(aggregate["aligned"]["base_2to1_errors"])
        == EXPECTED_B2_2TO1_ERRORS,
        "minimum_calibration_groups": minimum_calibration_groups
        >= MIN_CALIBRATION_GROUPS,
        "maximum_tolerance_upper": maximum_tolerance_upper <= RISK_ALPHA,
        "all_calibration_exceedances_zero_by_strict_max": True,
        "seven_cross_label_groups_excluded": len(overlap_excluded_groups) == 7,
        "derangement_contracts_complete": len(derangement_reports) == FOLDS,
        "derangement_integrity": all(
            report["fixed_points"] == 0
            and report["same_source_assignments"] == 0
            and report["label_mismatches"] == 0
            and report["fold_boundary_crossings"] == 0
            and report["class2_exclusion_role_crossings"] == 0
            for report in derangement_reports
        ),
        "fit_threshold_artifacts_frozen": len(parameter_artifacts) == 3 * FOLDS
        and all(
            Path(str(artifact[key])).is_file()
            and _sha256(Path(str(artifact[key]))) == artifact[f"{key}_sha256"]
            for artifact in parameter_artifacts
            for key in ("mean", "variance", "threshold")
        ),
        "validation_test_not_used": True,
    }
    readiness = readiness_from_metrics(
        aligned=aggregate["aligned"],
        controls={"zero": aggregate["zero"], "deranged": aggregate["deranged"]},
        fold_rows=fold_rows,
        integrity_checks=integrity,
    )
    audit = {
        "aggregate": aggregate,
        "minimum_calibration_groups": minimum_calibration_groups,
        "maximum_tolerance_upper": maximum_tolerance_upper,
        "cross_label_overlap_excluded_groups": sorted(overlap_excluded_groups),
        "derangement_reports": derangement_reports,
        "parameter_artifacts": parameter_artifacts,
        "integrity": integrity,
        "readiness": readiness,
    }
    return audit, fold_rows, prediction_rows, readiness


def run_precheck(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    data_path = Path(args.data).resolve()
    b2_checkpoint_path = Path(args.b2_checkpoint).resolve()
    b2_cache_dir = Path(args.b2_cache_dir).resolve()
    dino_checkpoint_path = Path(args.dino_checkpoint).absolute()
    output_dir = Path(args.output_dir).resolve()
    for path in (data_path, b2_checkpoint_path, dino_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not b2_cache_dir.is_dir():
        raise FileNotFoundError(b2_cache_dir)
    (
        dataset,
        _b2_checkpoint,
        paths,
        labels,
        groups,
        assignments,
        fold_rows,
    ) = _dataset_inventory(
        data_path=data_path,
        b2_checkpoint_path=b2_checkpoint_path,
        class_name_mode=str(args.class_name_mode),
    )
    _assert_output_outside_train(output_dir, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    b2_logits, b2_provenance = _load_verified_b2_outputs(
        cache_dir=b2_cache_dir,
        paths=paths,
        labels=labels,
        groups=groups,
        assignments=assignments,
    )
    model, dino_provenance, runtime_contract = _load_frozen_dino(
        dino_checkpoint_path
    )
    calibration_counts = []
    exclusions = []
    for fold in range(FOLDS):
        fit = assignments != fold
        class1_groups = set(str(value) for value in groups[np.logical_and(fit, labels == 1)])
        before = np.logical_and(fit, labels == 2)
        overlap = np.asarray([str(value) in class1_groups for value in groups], dtype=bool)
        calibration_counts.append(len(class1_groups))
        exclusions.append(
            {
                "fold": fold,
                "class1_calibration_groups": len(class1_groups),
                "tolerance_upper": maximum_order_tolerance_upper(len(class1_groups)),
                "class2_excluded_groups": int(
                    np.unique(groups[np.logical_and(before, overlap)]).size
                ),
                "class2_excluded_rows": int(np.sum(np.logical_and(before, overlap))),
            }
        )
    preflight: Dict[str, object] = {
        "schema_version": 1,
        "mode": "frozen_dino_np_2to1_train_only_oof",
        "protocol_id": PROTOCOL_ID,
        "data": str(data_path),
        "data_sha256": _sha256(data_path),
        "b2_checkpoint": str(b2_checkpoint_path),
        "b2_checkpoint_sha256": _sha256(b2_checkpoint_path),
        "b2_cache": b2_provenance,
        "dino_checkpoint": str(dino_checkpoint_path),
        "dino_checkpoint_sha256": _sha256(dino_checkpoint_path),
        "dino_provenance": dino_provenance,
        "runtime_contract": runtime_contract,
        "output_dir": str(output_dir),
        "train_samples": EXPECTED_TRAIN_SAMPLES,
        "source_groups": EXPECTED_SOURCE_GROUPS,
        "paths_sha256": _path_rows_sha256(paths),
        "labels_sha256": _labels_sha256(labels),
        "source_groups_sha256": _source_groups_sha256(groups.tolist()),
        "fold_assignments_sha256": _array_sha256(assignments, dtype="<i8"),
        "global_folds": fold_rows,
        "fold_calibration_and_exclusions": exclusions,
        "minimum_class1_calibration_groups": min(calibration_counts),
        "locked_protocol": {
            "descriptor_width": DESCRIPTOR_WIDTH,
            "variance_floor": VARIANCE_FLOOR,
            "score_direction": "larger_is_more_class2_like",
            "risk_alpha": RISK_ALPHA,
            "risk_delta": RISK_DELTA,
            "minimum_calibration_groups": MIN_CALIBRATION_GROUPS,
            "threshold": "maximum source-group score",
            "comparison": "strict score > threshold",
            "action": "B2 top1=1 and runner-up=2 then 1->2",
            "pairs": ["2-1"],
        },
        "estimated_descriptor_cache_bytes": int(
            EXPECTED_TRAIN_SAMPLES * DESCRIPTOR_WIDTH * np.dtype(np.float32).itemsize
        ),
        "dataset_feature_extraction_performed": False,
        "score_fitting_performed": False,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
        "raw_dataset_modified": False,
    }
    _atomic_json(output_dir / "preflight.json", preflight)
    if bool(args.preflight_only):
        return preflight
    device = _resolve_device(str(args.device or ""))
    start = time.perf_counter()
    descriptors, cache_manifest = _load_or_extract_descriptors(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        paths=paths,
        labels=labels,
        groups=groups,
        assignments=assignments,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    audit, metric_rows, prediction_rows, readiness = run_oof_audit(
        descriptors=descriptors,
        b2_logits=b2_logits,
        labels=labels,
        groups=groups,
        assignments=assignments,
        paths=paths,
        output_dir=output_dir,
    )
    assignment_rows = [
        {
            "sample_index": index,
            "image_path": str(paths[index]),
            "source_group": str(groups[index]),
            "target_index": int(labels[index]),
            "fold": int(assignments[index]),
        }
        for index in range(labels.size)
    ]
    _write_csv(output_dir / "global_fold_assignments.csv", assignment_rows)
    _write_csv(output_dir / "fold_metrics.csv", metric_rows)
    _write_csv(output_dir / "train_oof_actions.csv", prediction_rows)
    summary: Dict[str, object] = {
        **preflight,
        "dataset_feature_extraction_performed": True,
        "score_fitting_performed": True,
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "descriptor_cache_manifest": cache_manifest,
        "audit": audit,
        "readiness": readiness,
        "artifacts": {
            "preflight": str(output_dir / "preflight.json"),
            "descriptor_cache_manifest": str(output_dir / CACHE_MANIFEST_FILENAME),
            "descriptors": str(output_dir / DESCRIPTOR_FILENAME),
            "global_fold_assignments": str(output_dir / "global_fold_assignments.csv"),
            "fold_metrics": str(output_dir / "fold_metrics.csv"),
            "oof_actions": str(output_dir / "train_oof_actions.csv"),
            "fold_parameters": str(output_dir / "fold_parameters"),
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_precheck(args)
    readiness = summary.get("readiness")
    if isinstance(readiness, Mapping):
        payload = {
            "frozen_dino_np_2to1_ready": bool(
                readiness.get("frozen_dino_np_2to1_ready", False)
            ),
            "failed_checks": list(readiness.get("failed_checks", [])),
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    else:
        payload = {
            "preflight_only": bool(args.preflight_only),
            "train_samples": int(summary.get("train_samples", 0)),
            "source_groups": int(summary.get("source_groups", 0)),
            "dataset_feature_extraction_performed": False,
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    payload.update(
        {
            "validation_split_used": False,
            "test_split_used": False,
        }
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

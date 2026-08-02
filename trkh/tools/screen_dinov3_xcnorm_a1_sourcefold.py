from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.models.dinov3_xcnorm_pair_adapter_a1 import (
    XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT,
    XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT,
)
from trkh.models.model import build_model_from_checkpoint
from trkh.tools import precheck_dinov3_xcnorm_a1_sourcefold as a1_preflight
from trkh.tools.export_timm_predictions import _state_dict_sha256
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (
    ADAPTER_BATCH_SIZE,
    EPOCHS,
    EXPECTED_ASSIGNMENT_INT64_SHA256,
    EXPECTED_CLASSES,
    EXPECTED_DATA_SHA256,
    EXPECTED_INTEGRITY_MANIFEST_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PATH_FOLD_SHA256,
    EXPECTED_SELECTED_STATE_SHA256,
    EXPECTED_TRAIN_SAMPLES,
    FOLDS,
    LR,
    PATCH_SHAPE,
    RIVALS,
    SEED,
    TEMPERED_POWER,
    WEIGHT_DECAY,
    _array_finite,
    _data_root_from_yaml,
    _flush_and_close_memmaps,
    _json_sha256,
    _pair_aurocs,
    _pairwise_bce,
    _path_hash,
    _read_integrity_train_rows,
    _read_train_rows,
    _relative_train_path,
    _sha256,
    _tempered_indices,
    _validate_b9_checkpoint,
    assert_train_only_paths,
    assign_locked_folds,
    build_train_union_groups,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)
from timm.models.eva import Eva


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_XCNORM_A1_SOURCEFOLD_OOF_20260802"
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = SEED
FP_SOURCE_CLASSES = (0, 2, 3, 4)
INJECTION_P95_CAP = 0.04
FINAL_TOKEN_P95_CAP = 0.08

CACHE_FILES = {
    "z_patch": "train_final_norm1_patch_tokens_f32.npy",
    "u_patch": "train_post_final_mhsa_patch_tokens_f32.npy",
    "logits": "train_exact_b9_logits_f32.npy",
    "labels": "train_labels_i64.npy",
    "paths": "train_paths.json",
    "manifest": "cache_manifest.json",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked TRAIN-only five-fold OOF screen for the sole parallel-final-"
            "MHSA Conv/XCNorm A1. Validation and test are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="optional reusable hash-verified cache directory, separate from OOF output",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-integrity-manifest", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path, required=True)
    parser.add_argument("--preflight-artifact-sha256", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--adapter-batch-size", type=int, default=ADAPTER_BATCH_SIZE
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"stale partial output exists: {partial}")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV artifact")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"stale partial output exists: {partial}")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def validate_or_write_assignment_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> str:
    """Permit resume only when the existing assignment is byte-semantically exact."""

    if not rows:
        raise ValueError("fold-assignment rows must not be empty")
    fields = list(rows[0])
    expected = [
        {field: str(row[field]) for field in fields}
        for row in rows
    ]
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or ()) != fields:
                raise ValueError("existing fold-assignment columns changed")
            observed = [dict(row) for row in reader]
        if observed != expected:
            raise ValueError("existing fold-assignment rows changed")
    else:
        _atomic_write_csv(path, rows)
    return _sha256(path)


def validate_preflight_artifact(
    path: Path, expected_sha256: str
) -> Tuple[Dict[str, object], str]:
    """Require the committed actual-B9 TRAIN smoke before allocating cache."""

    resolved = Path(path).expanduser().resolve()
    observed_sha256 = _sha256(resolved)
    expected_sha256 = str(expected_sha256).strip().casefold()
    if len(expected_sha256) != 64 or observed_sha256 != expected_sha256:
        raise ValueError(
            "preflight artifact SHA-256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preflight artifact must contain a JSON object")
    smoke = payload.get("smoke", {})
    canonical = payload.get("canonical_train", {})
    locked_oof = payload.get("future_locked_oof_optimization", {})
    source_files = payload.get("source_files", {})
    preflight_tool = Path(a1_preflight.__file__).resolve()
    expected_source_files = {
        "tool_sha256": _sha256(preflight_tool),
        "a1_model_sha256": _sha256(
            preflight_tool.parents[1] / "models" / "dinov3_xcnorm_pair_adapter_a1.py"
        ),
        "a0_operator_model_sha256": _sha256(
            preflight_tool.parents[1] / "models" / "dinov3_xcnorm_pair_adapter_a0.py"
        ),
        "a0_shared_protocol_sha256": _sha256(
            preflight_tool.with_name("precheck_dinov3_xcnorm_a0_sourcefold.py")
        ),
        "model_builder_sha256": _sha256(
            preflight_tool.parents[1] / "models" / "model.py"
        ),
        "dataset_pipeline_sha256": _sha256(
            preflight_tool.parents[1] / "data" / "dataset.py"
        ),
        "probe_loader_sha256": _sha256(
            preflight_tool.with_name("probe_embedding_prototypes.py")
        ),
        "timm_eva_source_sha256": _sha256(
            Path(inspect.getsourcefile(Eva) or "").resolve()
        ),
    }
    required = {
        "protocol_id": payload.get("protocol_id") == a1_preflight.PROTOCOL_ID,
        "preflight_passed": payload.get("preflight_passed") is True,
        "actual_b9_smoke": isinstance(smoke, Mapping)
        and smoke.get("actual_b9_train_smoke_passed") is True,
        "branch_off_exact": isinstance(smoke, Mapping)
        and float(smoke.get("parity", {}).get("branch_off_logit_max_abs_error", 1.0))
        == 0.0,
        "smoke_paths_locked": payload.get("smoke_paths_sha256")
        == a1_preflight.EXPECTED_SMOKE_PATHS_SHA256
        and isinstance(smoke, Mapping)
        and smoke.get("smoke_paths_sha256")
        == a1_preflight.EXPECTED_SMOKE_PATHS_SHA256,
        "smoke_frozen_state_unchanged": isinstance(smoke, Mapping)
        and smoke.get("state_hashes_before") == smoke.get("state_hashes_after")
        and smoke.get("state_hashes_before") == payload.get("locked_state_hashes"),
        "smoke_adapter_only_gradient": isinstance(smoke, Mapping)
        and smoke.get("frozen_model_gradients_present") is False
        and float(smoke.get("residual_gradient_through_frozen_tail_norm", 0.0))
        > 0.0,
        "train_only": payload.get("train_split_used") is True
        and payload.get("validation_split_used") is False
        and payload.get("test_split_used") is False,
        "no_prior_cache_or_oof": payload.get("large_cache_created") is False
        and payload.get("oof_training_run") is False,
        "permissions_closed": payload.get("full_validation_permission") is False
        and payload.get("test_permission") is False,
        "canonical_support": isinstance(canonical, Mapping)
        and int(canonical.get("samples", -1)) == EXPECTED_TRAIN_SAMPLES
        and canonical.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256
        and canonical.get("integrity_manifest_sha256")
        == EXPECTED_INTEGRITY_MANIFEST_SHA256,
        "canonical_folds": isinstance(canonical, Mapping)
        and canonical.get("assignment_int64_sha256")
        == EXPECTED_ASSIGNMENT_INT64_SHA256
        and canonical.get("path_fold_sha256") == EXPECTED_PATH_FOLD_SHA256,
        "locked_oof": isinstance(locked_oof, Mapping)
        and int(locked_oof.get("folds", -1)) == FOLDS
        and int(locked_oof.get("seed", -1)) == SEED
        and int(locked_oof.get("epochs", -1)) == EPOCHS
        and float(locked_oof.get("learning_rate", -1.0)) == LR
        and int(locked_oof.get("adapter_batch_size", -1)) == ADAPTER_BATCH_SIZE
        and float(locked_oof.get("weight_decay", -1.0)) == WEIGHT_DECAY
        and float(locked_oof.get("tempered_sampling_power", -1.0))
        == TEMPERED_POWER
        and locked_oof.get("status") == "not_run_preflight_only",
        "all_eight_preflight_sources_current": isinstance(source_files, Mapping)
        and all(source_files.get(key) == value for key, value in expected_source_files.items()),
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise ValueError(f"preflight artifact is not admissible: {failed}")
    state_hashes = payload.get("locked_state_hashes", {})
    if not isinstance(state_hashes, Mapping):
        raise ValueError("preflight locked state hashes are missing")
    a1_preflight._assert_locked_state_hashes(state_hashes)
    return payload, observed_sha256


def _cache_paths(output_dir: Path) -> Dict[str, Path]:
    return {key: output_dir / value for key, value in CACHE_FILES.items()}


def _promote_memmap_cache(
    arrays: Mapping[str, np.memmap],
    partials: Mapping[str, Path],
    finals: Mapping[str, Path],
) -> Dict[str, str]:
    """Close Windows mappings, hash complete partials, then atomically replace."""

    names = tuple(arrays)
    _flush_and_close_memmaps(*(arrays[name] for name in names))
    hashes = {name: _sha256(partials[name]) for name in names}
    for name in names:
        partials[name].replace(finals[name])
        if _sha256(finals[name]) != hashes[name]:
            raise RuntimeError(f"atomic cache promotion hash drifted for {name}")
    return hashes


def _cache_expected(
    *,
    relative_paths: Sequence[str],
    labels: np.ndarray,
    preflight_sha256: str,
    state_hashes: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "preflight_artifact_sha256": preflight_sha256,
        "selected_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        "state_hashes": dict(state_hashes),
        "path_order_sha256": _path_hash(relative_paths),
        "z_patch_shape": [EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE],
        "u_patch_shape": [EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE],
        "logit_shape": [EXPECTED_TRAIN_SAMPLES, 5],
        "label_shape": [EXPECTED_TRAIN_SAMPLES],
        "floating_dtype": "float32",
        "label_dtype": "int64",
        "label_content_sha256": hashlib.sha256(
            np.asarray(labels, dtype="<i8").tobytes()
        ).hexdigest(),
        "class_counts": np.bincount(labels, minlength=5).astype(int).tolist(),
        "z_contract": "block11.norm1(x_pre)_patch_tokens",
        "u_contract": (
            "x_pre+drop_path1(gamma1*attn(norm1(x_pre),rope))_patch_tokens"
        ),
        "base_logit_contract": "exact_frozen_B9_EVA_final_tail_and_head",
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
    }


def load_or_extract_cache(
    *,
    model: nn.Module,
    dataset: Dataset,
    relative_paths: Sequence[str],
    labels: np.ndarray,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    preflight_sha256: str,
    state_hashes: Mapping[str, str],
) -> Dict[str, object]:
    paths = _cache_paths(output_dir)
    present = {key: value.is_file() for key, value in paths.items()}
    expected = _cache_expected(
        relative_paths=relative_paths,
        labels=labels,
        preflight_sha256=preflight_sha256,
        state_hashes=state_hashes,
    )
    if any(present.values()):
        if not all(present.values()):
            raise RuntimeError(f"partial A1 cache is forbidden: {present}")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        mismatch = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        for name in ("z_patch", "u_patch", "logits", "labels"):
            if _sha256(paths[name]) != manifest.get(f"{name}_sha256"):
                mismatch[f"{name}_sha256"] = "content hash mismatch"
        if _sha256(paths["paths"]) != manifest.get("paths_file_sha256"):
            mismatch["paths_file_sha256"] = "content hash mismatch"
        if mismatch:
            raise ValueError(f"A1 cache manifest mismatch: {mismatch}")
        stored_paths = json.loads(paths["paths"].read_text(encoding="utf-8"))
        if list(stored_paths) != list(relative_paths):
            raise ValueError("cached paths differ from canonical TRAIN order")
        arrays = {
            "z_patch": np.load(paths["z_patch"], mmap_mode="r", allow_pickle=False),
            "u_patch": np.load(paths["u_patch"], mmap_mode="r", allow_pickle=False),
            "logits": np.load(paths["logits"], mmap_mode="r", allow_pickle=False),
            "labels": np.load(paths["labels"], mmap_mode="r", allow_pickle=False),
        }
        if arrays["z_patch"].shape != (EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE):
            raise ValueError("cached z_patch shape changed")
        if arrays["u_patch"].shape != (EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE):
            raise ValueError("cached u_patch shape changed")
        if arrays["logits"].shape != (EXPECTED_TRAIN_SAMPLES, 5):
            raise ValueError("cached base-logit shape changed")
        if arrays["labels"].shape != (EXPECTED_TRAIN_SAMPLES,):
            raise ValueError("cached label shape changed")
        if arrays["z_patch"].dtype != np.float32 or arrays["u_patch"].dtype != np.float32:
            raise ValueError("cached A1 streams must remain FP32")
        if arrays["logits"].dtype != np.float32 or arrays["labels"].dtype != np.int64:
            raise ValueError("cached logit/label dtype changed")
        if not all(_array_finite(arrays[name]) for name in ("z_patch", "u_patch", "logits")):
            raise ValueError("cached A1 arrays contain non-finite values")
        if not np.array_equal(np.asarray(arrays["labels"]), labels):
            raise ValueError("cached labels differ from canonical TRAIN")
        return {**arrays, "manifest": manifest}

    array_names = ("z_patch", "u_patch", "logits", "labels")
    partials = {
        name: paths[name].with_suffix(paths[name].suffix + ".partial")
        for name in array_names
    }
    if any(path.exists() for path in partials.values()):
        raise RuntimeError("stale partial A1 cache exists; use a new output directory")
    arrays = {
        "z_patch": np.lib.format.open_memmap(
            partials["z_patch"], "w+", dtype=np.float32,
            shape=(EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE),
        ),
        "u_patch": np.lib.format.open_memmap(
            partials["u_patch"], "w+", dtype=np.float32,
            shape=(EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE),
        ),
        "logits": np.lib.format.open_memmap(
            partials["logits"], "w+", dtype=np.float32,
            shape=(EXPECTED_TRAIN_SAMPLES, 5),
        ),
        "labels": np.lib.format.open_memmap(
            partials["labels"], "w+", dtype=np.int64,
            shape=(EXPECTED_TRAIN_SAMPLES,),
        ),
    }
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        collate_fn=_collate_classification,
    )
    cursor = 0
    patch_tail_parity = 0.0
    model.eval()
    with torch.inference_mode():
        for images, targets, _metadata in tqdm(
            loader, desc="A1 exact TRAIN stream cache", dynamic_ncols=False,
            mininterval=1.0,
        ):
            images = images.to(device=device, dtype=torch.float32)
            x_pre, rope = a1_preflight.eva_pre_final_tokens(model, images)
            if rope is None:
                raise ValueError("locked DINOv3 cache requires native RoPE")
            z_full, u_full = a1_preflight.decompose_final_mhsa(
                model.blocks[a1_preflight.FINAL_BLOCK_INDEX], x_pre, rope
            )
            base_logits = model(images).float()
            prefix = XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT
            z_patch = z_full[:, prefix:].float()
            u_patch = u_full[:, prefix:].float()
            if tuple(z_patch.shape[1:]) != PATCH_SHAPE or tuple(u_patch.shape[1:]) != PATCH_SHAPE:
                raise ValueError("actual B9 A1 cache stream shape changed")
            patch_logits, _tail_tokens = a1_preflight.frozen_tail_logits(
                model.blocks[a1_preflight.FINAL_BLOCK_INDEX],
                model.norm,
                model.head,
                u_patch,
                torch.zeros_like(u_patch),
            )
            patch_tail_parity = max(
                patch_tail_parity,
                float((patch_logits.float() - base_logits.float()).abs().max().cpu()),
            )
            values = (z_patch, u_patch, base_logits.float())
            if not all(bool(torch.isfinite(value).all().item()) for value in values):
                raise FloatingPointError("non-finite actual B9 cache value")
            end = cursor + int(targets.numel())
            arrays["z_patch"][cursor:end] = z_patch.cpu().numpy()
            arrays["u_patch"][cursor:end] = u_patch.cpu().numpy()
            arrays["logits"][cursor:end] = base_logits.float().cpu().numpy()
            arrays["labels"][cursor:end] = targets.numpy().astype(np.int64)
            cursor = end
    if cursor != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(f"A1 cache row mismatch: {cursor}")
    if patch_tail_parity > a1_preflight.PATCH_TAIL_PARITY_ATOL:
        raise RuntimeError(f"patch-only frozen tail parity failed: {patch_tail_parity}")
    hashes = _promote_memmap_cache(arrays, partials, paths)
    del arrays
    _atomic_write_json(paths["paths"], list(relative_paths))
    manifest = {
        **expected,
        "maximum_patch_tail_logit_parity_error": patch_tail_parity,
        **{f"{name}_sha256": digest for name, digest in hashes.items()},
        "paths_file_sha256": _sha256(paths["paths"]),
    }
    _atomic_write_json(paths["manifest"], manifest)
    return {
        "z_patch": np.load(paths["z_patch"], mmap_mode="r", allow_pickle=False),
        "u_patch": np.load(paths["u_patch"], mmap_mode="r", allow_pickle=False),
        "logits": np.load(paths["logits"], mmap_mode="r", allow_pickle=False),
        "labels": np.load(paths["labels"], mmap_mode="r", allow_pickle=False),
        "manifest": manifest,
    }


def metrics_all_fp(labels: np.ndarray, logits: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(logits).argmax(axis=1)
    per_class = f1_score(
        labels, predicted, labels=list(range(5)), average=None, zero_division=0
    )
    fp_by_source = {
        str(source): int(np.sum((labels == source) & (predicted == 1)))
        for source in FP_SOURCE_CLASSES
    }
    return {
        "macro_f1": float(np.mean(per_class)),
        "class1_f1": float(per_class[1]),
        "class1_tp": int(np.sum((labels == 1) & (predicted == 1))),
        "class1_fn": int(np.sum((labels == 1) & (predicted != 1))),
        "total_fp_to_class1": int(sum(fp_by_source.values())),
        "fp_to_class1": fp_by_source,
    }


def _telemetry_summary(values: Sequence[np.ndarray]) -> Dict[str, float]:
    if not values:
        raise RuntimeError("A1 telemetry is empty")
    array = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    if not np.isfinite(array).all():
        raise FloatingPointError("A1 telemetry contains non-finite values")
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _atomic_torch_save(path: Path, payload: object) -> str:
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists() or path.exists():
        raise RuntimeError(f"refuse to overwrite state artifact: {path}")
    torch.save(payload, partial)
    digest = _sha256(partial)
    partial.replace(path)
    if _sha256(path) != digest:
        raise RuntimeError("atomic adapter-state promotion hash drifted")
    return digest


def _adapter_logits(
    model: nn.Module,
    adapter: nn.Module,
    z_patch: Tensor,
    u_patch: Tensor,
    base_logits: Tensor,
    *,
    branch_off: bool = False,
    return_trace: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    """Compose only already-preflighted A1 residual and frozen-tail helpers."""

    residual_result = adapter.residual_from_final_norm1(
        z_patch,
        u_patch,
        model.head.weight,
        branch_off=bool(branch_off),
        return_trace=bool(return_trace),
    )
    if return_trace:
        residual, adapter_trace = residual_result
    else:
        residual = residual_result
        adapter_trace = {}
    logits, tail_trace = a1_preflight.tail_adjusted_logits(
        model.blocks[a1_preflight.FINAL_BLOCK_INDEX],
        model.norm,
        model.head,
        u_patch,
        residual,
        base_logits,
    )
    if return_trace:
        base_final = tail_trace["base_final_patch_tokens"].detach()
        active_final = tail_trace["active_final_patch_tokens"]
        final_ratio = (active_final.float() - base_final.float()).norm(
            dim=-1
        ) / base_final.float().norm(dim=-1).clamp_min(1e-12)
        return logits, {
            **adapter_trace,
            **tail_trace,
            "logit_delta": logits - base_logits.detach(),
            "final_token_perturbation_ratio": final_ratio,
        }
    return logits


def run_oof_screen(
    *,
    model: nn.Module,
    z_patch: np.ndarray,
    u_patch: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    relative_paths: Sequence[str],
    device: torch.device,
    batch_size: int,
    state_dir: Path,
    cache_manifest_sha256: str,
    preflight_sha256: str,
    assignment_hashes: Mapping[str, str],
    group_fold_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if any(array.shape != (EXPECTED_TRAIN_SAMPLES,) for array in (labels, folds, groups)):
        raise ValueError("A1 labels/folds/groups must cover all 8,278 TRAIN rows")
    if int(batch_size) != ADAPTER_BATCH_SIZE:
        raise ValueError(f"adapter batch size is locked to {ADAPTER_BATCH_SIZE}")
    state_dir.mkdir(parents=True, exist_ok=False)
    frozen_weight = model.head.weight.detach().to(device=device, dtype=torch.float32)
    if frozen_weight.requires_grad:
        raise RuntimeError("B9 classifier must remain frozen")
    model.eval()
    model.requires_grad_(False)
    frozen_before = a1_preflight._state_subset_hashes(model.state_dict())
    a1_preflight._assert_locked_state_hashes(frozen_before)

    # NaN initialization makes any missing/duplicated held-fold write fail
    # closed instead of leaking uninitialized memory into metrics.
    control_oof = np.full(base_logits.shape, np.nan, dtype=np.float32)
    candidate_oof = np.full(base_logits.shape, np.nan, dtype=np.float32)
    telemetry: Dict[str, Dict[str, List[np.ndarray]]] = {
        arm: {name: [] for name in ("injection", "final_token", "logit_delta")}
        for arm in ("control", "candidate")
    }
    fold_rows: List[Dict[str, object]] = []
    training_rows: List[Dict[str, object]] = []
    schedule_rows: List[Dict[str, object]] = []
    paired_contracts: List[Dict[str, object]] = []
    expected_updates = 0
    completed_updates = 0
    skipped_updates = 0
    nonfinite_updates = 0
    branch_off_error = 0.0

    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        eligible = int(np.sum(labels[fit] != 3))
        expected_updates += EPOCHS * ((eligible + batch_size - 1) // batch_size)
        control, candidate, paired = a1_preflight.build_paired_adapters(
            frozen_weight.detach().cpu(), seed=SEED + fold
        )
        control = control.to(device=device, dtype=torch.float32)
        candidate = candidate.to(device=device, dtype=torch.float32)
        control.bind_classifier_weight(frozen_weight)
        candidate.bind_classifier_weight(frozen_weight)
        paired_contracts.append({"fold": fold, **paired})
        optimizers = (
            torch.optim.AdamW(control.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
            torch.optim.AdamW(candidate.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        )

        for epoch in range(EPOCHS):
            schedule_seed = SEED + fold * 100 + epoch
            schedule = _tempered_indices(
                fit, labels, schedule_seed, int(batch_size)
            )
            schedule_row = {
                "fold": fold,
                "epoch": epoch + 1,
                "seed": schedule_seed,
                "samples": int(schedule.size),
                "class_counts": np.bincount(
                    labels[schedule], minlength=5
                ).astype(int).tolist(),
                "schedule_int64_sha256": hashlib.sha256(
                    np.asarray(schedule, dtype="<i8").tobytes()
                ).hexdigest(),
            }
            schedule_rows.append(schedule_row)
            losses = [0.0, 0.0]
            seen = 0
            control.train()
            candidate.train()
            for start in range(0, schedule.size, batch_size):
                selected = schedule[start : start + batch_size]
                z_batch = torch.from_numpy(
                    np.asarray(z_patch[selected], dtype=np.float32)
                ).to(device)
                u_batch = torch.from_numpy(
                    np.asarray(u_patch[selected], dtype=np.float32)
                ).to(device)
                base_batch = torch.from_numpy(
                    np.asarray(base_logits[selected], dtype=np.float32)
                ).to(device)
                target_batch = torch.from_numpy(labels[selected]).to(device)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                batch_losses: List[Tensor] = []
                gradients_valid = True
                # Keep only one frozen-tail graph live at a time on the 8 GiB GPU.
                # Neither arm steps until both losses and both gradient sets pass.
                for module in (control, candidate):
                    output = _adapter_logits(
                        model, module, z_batch, u_batch, base_batch
                    )
                    loss = _pairwise_bce(output, target_batch)
                    batch_losses.append(loss.detach())
                    if not bool(torch.isfinite(loss).item()):
                        gradients_valid = False
                        del output, loss
                        break
                    loss.backward()
                    gradients_valid = gradients_valid and all(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in module.parameters()
                    )
                    del output, loss
                    if not gradients_valid:
                        break
                if not gradients_valid or len(batch_losses) != 2:
                    nonfinite_updates += 1
                    skipped_updates += 1
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    continue
                for optimizer in optimizers:
                    optimizer.step()
                if not all(
                    bool(torch.isfinite(parameter).all().item())
                    for module in (control, candidate)
                    for parameter in module.parameters()
                ):
                    raise FloatingPointError("A1 adapter parameter became non-finite")
                completed_updates += 1
                count = int(selected.size)
                for arm, loss in enumerate(batch_losses):
                    losses[arm] += float(loss.detach().cpu()) * count
                seen += count
            training_rows.append(
                {
                    **schedule_row,
                    "completed_samples": seen,
                    "control_loss": losses[0] / max(1, seen),
                    "candidate_loss": losses[1] / max(1, seen),
                }
            )

        control.eval()
        candidate.eval()
        control_state = {
            key: value.detach().cpu()
            for key, value in control.state_dict().items()
        }
        candidate_state = {
            key: value.detach().cpu()
            for key, value in candidate.state_dict().items()
        }
        state_path = state_dir / f"fold_{fold}_final_adapters.pt"
        state_sha = _atomic_torch_save(
            state_path,
            {
                "protocol_id": PROTOCOL_ID,
                "preflight_artifact_sha256": preflight_sha256,
                "cache_manifest_sha256": cache_manifest_sha256,
                "fold": fold,
                "epochs": EPOCHS,
                "held_fold_selection": False,
                "adapter_ema": False,
                "control_state": control_state,
                "candidate_state": candidate_state,
            },
        )
        paired_contracts[-1].update(
            {
                "control_final_state_sha256": _state_dict_sha256(control_state),
                "candidate_final_state_sha256": _state_dict_sha256(candidate_state),
                "state_artifact": str(state_path),
                "state_artifact_sha256": state_sha,
            }
        )

        arm_chunks: Dict[str, List[np.ndarray]] = {
            "control": [], "candidate": []
        }
        with torch.inference_mode():
            for start in range(0, held.size, batch_size):
                selected = held[start : start + batch_size]
                z_batch = torch.from_numpy(
                    np.asarray(z_patch[selected], dtype=np.float32)
                ).to(device)
                u_batch = torch.from_numpy(
                    np.asarray(u_patch[selected], dtype=np.float32)
                ).to(device)
                base_batch = torch.from_numpy(
                    np.asarray(base_logits[selected], dtype=np.float32)
                ).to(device)
                for arm_name, module in (
                    ("control", control), ("candidate", candidate)
                ):
                    result, trace = _adapter_logits(
                        model,
                        module,
                        z_batch,
                        u_batch,
                        base_batch,
                        return_trace=True,
                    )
                    arm_chunks[arm_name].append(result.cpu().numpy())
                    telemetry[arm_name]["injection"].append(
                        trace["residual_norm_ratio"].cpu().numpy()
                    )
                    telemetry[arm_name]["final_token"].append(
                        trace["final_token_perturbation_ratio"].cpu().numpy()
                    )
                    telemetry[arm_name]["logit_delta"].append(
                        trace["logit_delta"].float().norm(dim=-1).cpu().numpy()
                    )
                off = _adapter_logits(
                    model,
                    candidate,
                    z_batch,
                    u_batch,
                    base_batch,
                    branch_off=True,
                )
                branch_off_error = max(
                    branch_off_error,
                    float((off - base_batch).abs().max().cpu()),
                )
        control_fold = np.concatenate(arm_chunks["control"])
        candidate_fold = np.concatenate(arm_chunks["candidate"])
        control_oof[held] = control_fold
        candidate_oof[held] = candidate_fold
        control_auc = _pair_aurocs(labels[held], control_fold)
        candidate_auc = _pair_aurocs(labels[held], candidate_fold)
        fold_rows.append(
            {
                "fold": fold,
                "control_pair_auroc": control_auc,
                "candidate_pair_auroc": candidate_auc,
                "mean_pair_auroc_gain": float(
                    np.mean(
                        [
                            candidate_auc[str(rival)]
                            - control_auc[str(rival)]
                            for rival in RIVALS
                        ]
                    )
                ),
            }
        )

    if not np.isfinite(control_oof).all() or not np.isfinite(candidate_oof).all():
        raise RuntimeError("A1 OOF logits are incomplete or non-finite")

    frozen_after = a1_preflight._state_subset_hashes(model.state_dict())
    if frozen_after != frozen_before:
        raise RuntimeError("frozen B9 state changed during A1 OOF")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A1 OOF leaked gradients into frozen B9")

    base_metrics = metrics_all_fp(labels, base_logits)
    control_metrics = metrics_all_fp(labels, control_oof)
    candidate_metrics = metrics_all_fp(labels, candidate_oof)
    control_auc = _pair_aurocs(labels, control_oof)
    candidate_auc = _pair_aurocs(labels, candidate_oof)
    pair_gains = {
        str(rival): float(
            candidate_auc[str(rival)] - control_auc[str(rival)]
        )
        for rival in RIVALS
    }
    telemetry_summary = {
        arm: {
            "injection_ratio": _telemetry_summary(values["injection"]),
            "final_token_perturbation_ratio": _telemetry_summary(
                values["final_token"]
            ),
            "logit_delta_l2": _telemetry_summary(values["logit_delta"]),
        }
        for arm, values in telemetry.items()
    }

    oof_path = state_dir.parent / "train_oof_logits.npz"
    partial_oof = oof_path.with_suffix(oof_path.suffix + ".partial")
    if partial_oof.exists() or oof_path.exists():
        raise RuntimeError("refuse to overwrite A1 OOF artifact")
    with partial_oof.open("wb") as handle:
        np.savez_compressed(
            handle,
            relative_paths=np.asarray(list(relative_paths), dtype=np.str_),
            labels=labels,
            folds=folds,
            union_groups=groups,
            base_logits=np.asarray(base_logits, dtype=np.float32),
            control_logits=control_oof,
            candidate_logits=candidate_oof,
        )
    oof_sha = _sha256(partial_oof)
    partial_oof.replace(oof_path)
    if _sha256(oof_path) != oof_sha:
        raise RuntimeError("atomic OOF promotion hash drifted")
    return {
        "base": base_metrics,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "control_pair_auroc": control_auc,
        "candidate_pair_auroc": candidate_auc,
        "pair_auroc_gains": pair_gains,
        "mean_pair_auroc_gain": float(np.mean(list(pair_gains.values()))),
        "fold_rows": fold_rows,
        "fold_wins": int(
            sum(float(row["mean_pair_auroc_gain"]) > 0.0 for row in fold_rows)
        ),
        "telemetry": telemetry_summary,
        "branch_off_max_abs_error": branch_off_error,
        "training_rows": training_rows,
        "schedule_contract_sha256": _json_sha256(schedule_rows),
        "paired_contracts": paired_contracts,
        "completed_updates": completed_updates,
        "expected_updates": expected_updates,
        "nonfinite_updates": nonfinite_updates,
        "skipped_updates": skipped_updates,
        "frozen_state_hashes_before": frozen_before,
        "frozen_state_hashes_after": frozen_after,
        "assignment_hashes": dict(assignment_hashes),
        "group_fold_contract": [dict(row) for row in group_fold_rows],
        "oof_artifact": str(oof_path),
        "oof_artifact_sha256": oof_sha,
        "group_vector_int64_sha256": hashlib.sha256(
            groups.astype("<i8").tobytes()
        ).hexdigest(),
        "fold_vector_int64_sha256": hashlib.sha256(
            folds.astype("<i8").tobytes()
        ).hexdigest(),
    }


def _mean_pair_auc_delta(
    labels: np.ndarray, candidate: np.ndarray, control: np.ndarray
) -> float:
    values: List[float] = []
    for rival in RIVALS:
        mask = np.logical_or(labels == 1, labels == rival)
        binary = (labels[mask] == 1).astype(np.int64)
        if np.unique(binary).size != 2:
            raise RuntimeError("bootstrap replicate lost binary pair support")
        values.append(
            float(
                roc_auc_score(binary, candidate[mask, 1] - candidate[mask, rival])
                - roc_auc_score(binary, control[mask, 1] - control[mask, rival])
            )
        )
    return float(np.mean(values))


def _class1_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(
        f1_score(
            labels == 1,
            np.asarray(logits).argmax(axis=1) == 1,
            zero_division=0,
        )
    )


def _macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            np.asarray(logits).argmax(axis=1),
            labels=list(range(5)),
            average="macro",
            zero_division=0,
        )
    )


def _fp_rate_to_class1(labels: np.ndarray, logits: np.ndarray) -> float:
    predicted = np.asarray(logits).argmax(axis=1)
    non_class1 = labels != 1
    return float(np.sum(non_class1 & (predicted == 1)) / max(1, non_class1.sum()))


def _interval(values: np.ndarray) -> Dict[str, float]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise RuntimeError("bootstrap metric vector is empty/non-finite")
    return {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def fold_stratified_union_group_bootstrap(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    base_logits: np.ndarray,
    control_logits: np.ndarray,
    candidate_logits: np.ndarray,
    oof_artifact: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, object]:
    """Resample complete union groups with replacement independently per fold."""

    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    arrays = (base_logits, control_logits, candidate_logits)
    if labels.shape != folds.shape or labels.shape != groups.shape:
        raise ValueError("bootstrap labels/folds/groups are not aligned")
    if any(np.asarray(array).shape != (labels.size, 5) for array in arrays):
        raise ValueError("bootstrap logits must have shape [N,5]")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    group_folds: Dict[int, int] = {}
    members: Dict[int, np.ndarray] = {}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        group_fold_values = np.unique(folds[positions])
        if group_fold_values.size != 1:
            raise ValueError(f"union group {group} crosses held folds")
        group_folds[int(group)] = int(group_fold_values[0])
        members[int(group)] = positions
    fold_groups = {
        fold: sorted(
            group for group, group_fold in group_folds.items() if group_fold == fold
        )
        for fold in range(FOLDS)
    }
    if any(not values for values in fold_groups.values()):
        raise ValueError("bootstrap requires union groups in all five folds")

    rng = np.random.default_rng(int(seed))
    draws_digest = hashlib.sha256()
    pair_values = np.empty(int(replicates), dtype=np.float64)
    class1_values = np.empty(int(replicates), dtype=np.float64)
    macro_values = np.empty(int(replicates), dtype=np.float64)
    fp_rate_values = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled_positions: List[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draws_digest.update(np.asarray([replicate, fold], dtype="<i8").tobytes())
            draws_digest.update(np.asarray(draw, dtype="<i8").tobytes())
            sampled_positions.extend(members[available[int(index)]] for index in draw)
        selected = np.concatenate(sampled_positions)
        y = labels[selected]
        base = np.asarray(base_logits)[selected]
        control = np.asarray(control_logits)[selected]
        candidate = np.asarray(candidate_logits)[selected]
        pair_values[replicate] = _mean_pair_auc_delta(y, candidate, control)
        class1_values[replicate] = _class1_f1(y, candidate) - _class1_f1(y, base)
        macro_values[replicate] = _macro_f1(y, candidate) - _macro_f1(y, base)
        fp_rate_values[replicate] = _fp_rate_to_class1(
            y, candidate
        ) - _fp_rate_to_class1(y, control)

    oof_sha = _sha256(Path(oof_artifact))
    group_sha = hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest()
    fold_sha = hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest()
    return {
        "method": "paired_fold_stratified_union_group_percentile_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "oof_artifact_sha256": oof_sha,
        "group_vector_int64_sha256": group_sha,
        "fold_vector_int64_sha256": fold_sha,
        "draws_int64_sha256": draws_digest.hexdigest(),
        "mean_pair_auroc_delta": {
            "point": _mean_pair_auc_delta(labels, candidate_logits, control_logits),
            **_interval(pair_values),
        },
        "class1_f1_delta_vs_b9": {
            "point": _class1_f1(labels, candidate_logits)
            - _class1_f1(labels, base_logits),
            **_interval(class1_values),
        },
        "macro_f1_delta_vs_b9": {
            "point": _macro_f1(labels, candidate_logits)
            - _macro_f1(labels, base_logits),
            **_interval(macro_values),
        },
        "total_fp_rate_delta_vs_control": {
            "point": _fp_rate_to_class1(labels, candidate_logits)
            - _fp_rate_to_class1(labels, control_logits),
            **_interval(fp_rate_values),
        },
    }


def assess_readiness(
    screen: Mapping[str, object], bootstrap: Mapping[str, object]
) -> Dict[str, object]:
    base = screen.get("base", {})
    control = screen.get("control", {})
    candidate = screen.get("candidate", {})
    if not all(isinstance(value, Mapping) for value in (base, control, candidate)):
        raise ValueError("A1 screen metrics are missing")
    base_fp = base.get("fp_to_class1", {})
    control_fp = control.get("fp_to_class1", {})
    candidate_fp = candidate.get("fp_to_class1", {})
    if not all(isinstance(value, Mapping) for value in (base_fp, control_fp, candidate_fp)):
        raise ValueError("A1 all-source FP metrics are missing")
    telemetry = screen.get("telemetry", {})
    assignment = screen.get("assignment_hashes", {})
    pair_gains = screen.get("pair_auroc_gains", {})
    training_rows = list(screen.get("training_rows", []))
    paired = list(screen.get("paired_contracts", []))
    group_folds = list(screen.get("group_fold_contract", []))
    if not all(isinstance(value, Mapping) for value in (telemetry, assignment, pair_gains)):
        raise ValueError("A1 structural metrics are missing")

    pair_ci = bootstrap.get("mean_pair_auroc_delta", {})
    class1_ci = bootstrap.get("class1_f1_delta_vs_b9", {})
    macro_ci = bootstrap.get("macro_f1_delta_vs_b9", {})
    fp_ci = bootstrap.get("total_fp_rate_delta_vs_control", {})
    if not all(isinstance(value, Mapping) for value in (pair_ci, class1_ci, macro_ci, fp_ci)):
        raise ValueError("A1 bootstrap intervals are missing")
    base_total_fp = float(base.get("total_fp_to_class1", 0))
    candidate_total_fp = float(candidate.get("total_fp_to_class1", 0))
    tp_retention = float(candidate.get("class1_tp", 0)) / max(
        1.0, float(base.get("class1_tp", 0))
    )
    fp_reduction = (base_total_fp - candidate_total_fp) / max(1.0, base_total_fp)

    telemetry_caps = True
    for arm in ("control", "candidate"):
        arm_values = telemetry.get(arm, {})
        if not isinstance(arm_values, Mapping):
            telemetry_caps = False
            continue
        injection = arm_values.get("injection_ratio", {})
        final_token = arm_values.get("final_token_perturbation_ratio", {})
        telemetry_caps = telemetry_caps and isinstance(injection, Mapping)
        telemetry_caps = telemetry_caps and isinstance(final_token, Mapping)
        telemetry_caps = telemetry_caps and float(injection.get("p95", float("inf"))) <= INJECTION_P95_CAP + 1e-6
        telemetry_caps = telemetry_caps and float(final_token.get("p95", float("inf"))) <= FINAL_TOKEN_P95_CAP + 1e-6

    expected_schedule = FOLDS * EPOCHS
    checks = {
        "preflight_artifact_verified": bool(screen.get("preflight_artifact_verified", False)),
        "canonical_fold_assignment_locked": assignment.get("assignment_int64_sha256") == EXPECTED_ASSIGNMENT_INT64_SHA256
        and assignment.get("path_fold_sha256") == EXPECTED_PATH_FOLD_SHA256,
        "five_group_disjoint_folds": len(group_folds) == FOLDS
        and all(int(row.get("group_overlap", 1)) == 0 for row in group_folds),
        "paired_capacity_exact_3672": len(paired) == FOLDS
        and all(
            int(row.get("control_parameters", -1)) == XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT
            and int(row.get("candidate_parameters", -1)) == XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT
            for row in paired
        ),
        "locked_batch_schedule_recorded": len(training_rows) == expected_schedule
        and len(str(screen.get("schedule_contract_sha256", ""))) == 64
        and all(
            int(row.get("samples", 0)) > 0
            and len(str(row.get("schedule_int64_sha256", ""))) == 64
            and len(list(row.get("class_counts", []))) == 5
            and int(list(row.get("class_counts", [0] * 5))[3]) == 0
            and all(
                int(list(row.get("class_counts", [0] * 5))[index]) > 0
                for index in (0, 1, 2, 4)
            )
            for row in training_rows
        ),
        "bootstrap_contract_locked": int(bootstrap.get("replicates", -1)) == BOOTSTRAP_REPLICATES
        and int(bootstrap.get("seed", -1)) == BOOTSTRAP_SEED
        and bootstrap.get("oof_artifact_sha256") == screen.get("oof_artifact_sha256")
        and bootstrap.get("group_vector_int64_sha256") == screen.get("group_vector_int64_sha256")
        and bootstrap.get("fold_vector_int64_sha256") == screen.get("fold_vector_int64_sha256"),
        "candidate_mean_pair_auroc_gain": float(screen.get("mean_pair_auroc_gain", float("-inf"))) >= 0.0002,
        "candidate_pair_auroc_lcb_positive": float(pair_ci.get("lower", float("-inf"))) > 0.0,
        "candidate_fold_wins": int(screen.get("fold_wins", -1)) >= 4,
        "no_pair_below_minus_0001": len(pair_gains) == len(RIVALS)
        and min((float(value) for value in pair_gains.values()), default=float("-inf")) >= -0.0001,
        "class1_f1_gain_vs_b9": float(candidate.get("class1_f1", 0.0)) - float(base.get("class1_f1", 0.0)) >= 0.005,
        "class1_f1_gain_lcb_positive": float(class1_ci.get("lower", float("-inf"))) > 0.0,
        "macro_f1_delta_lcb_noninferior": float(macro_ci.get("lower", float("-inf"))) >= -0.002,
        "class1_tp_retention": tp_retention >= 0.98,
        "each_source_fp_no_higher_than_base_and_control": all(
            int(candidate_fp.get(str(source), 10**9)) <= int(base_fp.get(str(source), -1))
            and int(candidate_fp.get(str(source), 10**9)) <= int(control_fp.get(str(source), -1))
            for source in FP_SOURCE_CLASSES
        ),
        "total_fp_reduced_10pct_vs_b9": fp_reduction >= 0.10,
        "total_fp_rate_delta_vs_control_ucb_nonpositive": float(fp_ci.get("upper", float("inf"))) <= 0.0,
        "telemetry_caps": telemetry_caps,
        "branch_off_exact": float(screen.get("branch_off_max_abs_error", 1.0)) == 0.0,
        "frozen_b9_hashes_unchanged": screen.get("frozen_state_hashes_before") == screen.get("frozen_state_hashes_after"),
        "finite_updates": int(screen.get("nonfinite_updates", -1)) == 0,
        "zero_skipped_updates": int(screen.get("skipped_updates", -1)) == 0,
        "all_fixed_updates_completed": int(screen.get("completed_updates", -1)) == int(screen.get("expected_updates", -2)),
    }
    clean_pass = all(bool(value) for value in checks.values())
    return {
        "conditional_adapter_screen": True,
        "end_to_end_oof_or_generalization_claim": False,
        "clean_train_gates_passed": clean_pass,
        "photometric_train_permission": clean_pass,
        "full_validation_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "observed": {
            "class1_tp_retention_vs_b9": tp_retention,
            "total_fp_reduction_vs_b9": fp_reduction,
        },
    }


def run_screen(args: argparse.Namespace) -> Dict[str, object]:
    # Fail before filesystem/model work when the locked execution contract drifts.
    if int(args.adapter_batch_size) != ADAPTER_BATCH_SIZE:
        raise ValueError(
            f"adapter batch size is protocol-locked to {ADAPTER_BATCH_SIZE}"
        )
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("cache batch must be positive and workers non-negative")
    preflight, preflight_sha = validate_preflight_artifact(
        args.preflight_artifact, args.preflight_artifact_sha256
    )

    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    data_yaml = args.data.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else output_dir
    )
    if _sha256(data_yaml) != EXPECTED_DATA_SHA256:
        raise ValueError("data YAML is not locked canonical class_f")
    data_root = _data_root_from_yaml(data_yaml)
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("output-dir must remain outside immutable class_f")
    try:
        cache_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("cache-dir must remain outside immutable class_f")
    manifest_path = (args.manifest or data_root / "manifest.csv").expanduser().resolve()
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("canonical TRAIN manifest hash changed")
    integrity_path = args.train_integrity_manifest.expanduser().resolve()
    if _sha256(integrity_path) != EXPECTED_INTEGRITY_MANIFEST_SHA256:
        raise ValueError("TRAIN integrity manifest hash changed")

    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("B9 checkpoint payload is not a mapping")
    checkpoint_contract = _validate_b9_checkpoint(checkpoint, checkpoint_sha)
    if preflight.get("checkpoint_contract") != checkpoint_contract:
        raise ValueError("preflight and runner B9 checkpoint contracts differ")

    dataset, classes = _build_dataset(
        data_yaml=data_yaml,
        split="train",
        checkpoint=checkpoint,
        class_name_mode="raw",
        max_samples=0,
    )
    if tuple(classes) != EXPECTED_CLASSES or len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("canonical TRAIN class order/support changed")
    dataset_paths = [Path(path) for path in dataset.sample_paths()]
    assert_train_only_paths(dataset_paths, data_root / "train")
    relative_paths = [_relative_train_path(path, data_root) for path in dataset_paths]
    labels = np.asarray(dataset.labels(), dtype=np.int64)
    canonical_rows = [
        {"relative_path": path, "label": int(labels[index])}
        for index, path in enumerate(relative_paths)
    ]
    manifest_rows = _read_train_rows(manifest_path, data_root)
    manifest_map = {
        str(row["relative_path"]).casefold(): int(row["label"])
        for row in manifest_rows
    }
    canonical_map = {
        str(row["relative_path"]).casefold(): int(row["label"])
        for row in canonical_rows
    }
    if manifest_map != canonical_map:
        raise ValueError("dataset loader differs from canonical TRAIN manifest")

    integrity_rows = _read_integrity_train_rows(
        integrity_path, canonical_rows, data_root
    )
    phashes = [int(row["phash_value"]) for row in integrity_rows]
    groups_sorted, group_stats = build_train_union_groups(integrity_rows, phashes)
    assignments_sorted, fold_rows, assignment_hashes = assign_locked_folds(
        integrity_rows, groups_sorted, canonical_lock=True
    )
    fold_by_path = {
        str(row["relative_path"]).casefold(): int(assignments_sorted[index])
        for index, row in enumerate(integrity_rows)
    }
    group_by_path = {
        str(row["relative_path"]).casefold(): int(groups_sorted[index])
        for index, row in enumerate(integrity_rows)
    }
    folds = np.asarray(
        [fold_by_path[path.casefold()] for path in relative_paths], dtype=np.int64
    )
    groups = np.asarray(
        [group_by_path[path.casefold()] for path in relative_paths], dtype=np.int64
    )
    if preflight.get("canonical_train", {}).get("folds") != fold_rows:
        raise ValueError("preflight and runner source-fold summaries differ")

    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model.requires_grad_(False)
    runtime = a1_preflight._runtime_contract(model)
    if runtime != preflight.get("runtime_contract"):
        raise ValueError("preflight and runner actual EVA runtime differ")
    state_hashes = a1_preflight._state_subset_hashes(model.state_dict())
    a1_preflight._assert_locked_state_hashes(state_hashes)
    if state_hashes != preflight.get("locked_state_hashes"):
        raise ValueError("preflight and runner frozen B9 state hashes differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise RuntimeError("refuse to overwrite completed A1 summary")
    assignment_rows = [
        {
            "relative_path": path,
            "label": int(labels[index]),
            "union_group": int(groups[index]),
            "fold": int(folds[index]),
        }
        for index, path in enumerate(relative_paths)
    ]
    assignment_path = output_dir / "train_fold_assignments.csv"
    assignment_sha = validate_or_write_assignment_csv(
        assignment_path, assignment_rows
    )

    device = _resolve_device(args.device)
    model.to(device=device, dtype=torch.float32)
    started = time.perf_counter()
    cache = load_or_extract_cache(
        model=model,
        dataset=dataset,
        relative_paths=relative_paths,
        labels=labels,
        output_dir=cache_dir,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        preflight_sha256=preflight_sha,
        state_hashes=state_hashes,
    )
    if a1_preflight._state_subset_hashes(model.state_dict()) != state_hashes:
        raise RuntimeError("frozen B9 state changed while extracting cache")
    cache_manifest_path = cache_dir / CACHE_FILES["manifest"]
    cache_manifest_sha = _sha256(cache_manifest_path)
    screen = run_oof_screen(
        model=model,
        z_patch=cache["z_patch"],
        u_patch=cache["u_patch"],
        base_logits=cache["logits"],
        labels=cache["labels"],
        folds=folds,
        groups=groups,
        relative_paths=relative_paths,
        device=device,
        batch_size=int(args.adapter_batch_size),
        state_dir=output_dir / "fold_adapter_states",
        cache_manifest_sha256=cache_manifest_sha,
        preflight_sha256=preflight_sha,
        assignment_hashes=assignment_hashes,
        group_fold_rows=fold_rows,
    )
    screen["preflight_artifact_verified"] = True
    screen["preflight_artifact_sha256"] = preflight_sha
    with np.load(screen["oof_artifact"], allow_pickle=False) as oof:
        bootstrap = fold_stratified_union_group_bootstrap(
            labels=oof["labels"],
            folds=oof["folds"],
            groups=oof["union_groups"],
            base_logits=oof["base_logits"],
            control_logits=oof["control_logits"],
            candidate_logits=oof["candidate_logits"],
            oof_artifact=Path(screen["oof_artifact"]),
        )
    readiness = assess_readiness(screen, bootstrap)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_scope": "conditional_train_only_parallel_final_mhsa_a1_screen",
        "preflight_artifact": str(Path(args.preflight_artifact).resolve()),
        "preflight_artifact_sha256": preflight_sha,
        "checkpoint_contract": checkpoint_contract,
        "frozen_state_hashes": state_hashes,
        "runtime_contract": runtime,
        "source_files": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "preflight_tool_sha256": _sha256(Path(a1_preflight.__file__).resolve()),
        },
        "canonical_train": {
            "samples": EXPECTED_TRAIN_SAMPLES,
            "class_names": list(classes),
            "class_counts": np.bincount(labels, minlength=5).astype(int).tolist(),
            "group_stats": group_stats,
            "folds": fold_rows,
            **assignment_hashes,
        },
        "locked_optimization": {
            "folds": FOLDS,
            "seed": SEED,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "adapter_batch_size": ADAPTER_BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "tempered_sampling_power": TEMPERED_POWER,
            "paired_schedule": True,
            "held_fold_selection": False,
            "adapter_ema": False,
        },
        "cache_manifest": cache["manifest"],
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_dir": str(cache_dir),
        "train_fold_assignments_sha256": assignment_sha,
        "screen": screen,
        "bootstrap": bootstrap,
        "readiness": readiness,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_screen(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "clean_train_gates_passed": bool(
                    summary["readiness"]["clean_train_gates_passed"]
                ),
                "photometric_train_permission": bool(
                    summary["readiness"]["photometric_train_permission"]
                ),
                "full_validation_permission": False,
                "test_permission": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

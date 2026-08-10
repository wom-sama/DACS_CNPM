from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PARENT_COMMIT = "1553f0d5b7357ef6fbc19811ae78a2a4fb989102"
BRANCH = "classification-only-research"
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_PAIR_SURFACE_DDF_A0_PROTOCOL_20260725.md"
)
LOCK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")

CACHE_ROOT = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_materialized_20260725"
)
COHORT_ARRAYS = CACHE_ROOT / "cohort_arrays.npz"
MODEL_SRGB = CACHE_ROOT / "model_srgb_uint8.npy"
VALID_MASKS_PACKBITS = CACHE_ROOT / "image_valid_masks_packbits.npy"
CACHE_SUMMARY = CACHE_ROOT / "summary.json"
CACHE_REPLAY_SUMMARY = CACHE_ROOT / "replay_summary.json"
CACHE_FORMAL_MANIFEST = CACHE_ROOT / "formal_manifest.json"
CACHE_ARTIFACT_MANIFEST = CACHE_ROOT / "artifact_set_manifest.json"

CIDT_PREDICTIONS = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_cidt_readiness_full_train_20260714"
    / "predictions_all_conditions.csv"
)
KEEPER_ROOT = (
    REPOSITORY_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
KEEPER_CHECKPOINT = KEEPER_ROOT / "checkpoints" / "best.pt"
KEEPER_CONFIG = KEEPER_ROOT / "resolved_config.json"
KEEPER_LAUNCHER_ARGS = KEEPER_ROOT / "launcher_args.json"
CURRENT_BEST_COMMANDS = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
)
CURRENT_BEST_HISTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
)

DDF_REPOSITORY = Path(r"D:\DataAI\external_sources\repositories\ddfnet")
DDF_MODULE = DDF_REPOSITORY / "ddf" / "ddf.py"
DDF_RESNET = DDF_REPOSITORY / "ddf_resnet.py"
DDF_LICENSE = DDF_REPOSITORY / "LICENSE"
DDF_README = DDF_REPOSITORY / "README.md"
DDF_PAPER = Path(
    r"D:\DataAI\external_sources\papers"
    r"\zhou2021_decoupled_dynamic_filter_cvpr.pdf"
)
DDF_COMMIT = "4e21b4ad55ffe039b075be4722eaa8968894408c"

NO_REPEAT_DOCS = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_POST_CAP_SUCCESSOR_NO_RUN_SCREEN_20260725.md",
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_META_ACON_ACTIVATION_SIGNAL_PROTOCOL_20260717.md",
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_META_ACON_ACTIVATION_SIGNAL_A0_CLOSURE_20260717.md",
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_VIG_MAX_RELATIVE_GRAPH_READINESS_PROTOCOL_20260715.md",
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_FIT_CLOSURE_20260725.md",
)

EXPECTED_SHA256 = {
    "cohort_arrays": "a835d3498319d3d25d83c0279ab24d6fcc703a2266ff5f091dc2273bbb8baa9d",
    "model_srgb": "6c56dabe69ae02f94f6754669614966227806b52be11a8559adc5fb82e05af97",
    "valid_masks_packbits": "1c742f6d3067014d78ae299fab07f3baeb4c0ec0014dc112975c4d7199e4771e",
    "cache_summary": "3ba9964bc7b06caf56fd32e817a7550156d38f10840c23c28c400587e2b646db",
    "cache_replay_summary": "ffe0aabebecd92ebe3970dab25d8212bf4807a44a3387e0814e7543d76b04284",
    "cache_formal_manifest": "322cf7fcc600e46bc5312c8c2fdc69aad18e12ad351117b33e8b8dd536a52769",
    "cache_artifact_manifest": "84811728ab7f795f4be15226962dbb3f33e694aff871cda4fa568cecae1df6aa",
    "cidt_predictions": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "keeper_config": "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    "keeper_launcher_args": "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    "current_best_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "current_best_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    "ddf_module": "34ea8c797b9902b0a9138bf6a72d2073494791f32e71046c9decd79a9c61d96e",
    "ddf_resnet": "e7c397e0319c70bc0582ec28348a49fe47af6846b7d5f501d3ff270d8ee0467e",
    "ddf_license": "b157bbd165963809e4360d163301dbfee0d72afaa697bcb0edd16b410673d253",
    "ddf_readme": "25d1f4c40be84d3411fd13a717debd0cf22fb1a6f54245106513c1c8d83f2bef",
    "ddf_paper": "35a1f1b7be36b3b072a26c73f7b631e8daec03ff76e0edbb6f1b69ec0e85b20c",
}

ROWS = 763
SOURCES = 735
CLASSES = 5
FOLDS = 5
CALIBRATION_FOLDS = (2, 3, 4, 2, 3)
EPOCHS = 20
BATCH_SIZE = 64
PRIMARY_SEED = 20260725
REPEAT_OFFSET = 100000
SATTOLO_OFFSET = 200000
DEPHASE_OFFSET = 300000
EXPECTED_COHORT_INDEX_SHA256 = (
    "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05"
)
CLASS_ORDER = (0, 1, 2, 4)
RIVALS = (0, 2, 4)
MAP_RESOLUTIONS = (32, 16)

VISUAL_SAMPLE_INDICES = np.asarray(
    [
        163,
        2370,
        856,
        338,
        58,
        2563,
        1939,
        2698,
        341,
        2004,
        1066,
        313,
        2,
        2906,
        345,
        923,
        195,
        2411,
        791,
        363,
        3624,
        3934,
        4012,
        4040,
        4613,
        4626,
        5379,
        5381,
        5732,
        5740,
    ],
    dtype=np.int64,
)

PROTECTED_UNTRACKED = (
    REPOSITORY_ROOT / "BaoCao" / "GT.md",
    REPOSITORY_ROOT
    / "BaoCao"
    / "mango_cls_256_merge01_4cls_architecture.dot",
    REPOSITORY_ROOT
    / "BaoCao"
    / "mango_cls_256_merge01_4cls_architecture.png",
    REPOSITORY_ROOT
    / "BaoCao"
    / "mango_cls_256_merge01_4cls_architecture.svg",
    REPOSITORY_ROOT
    / "BaoCao"
    / "mango_cls_256_merge01_4cls_architecture_summary.md",
    REPOSITORY_ROOT / "deep-research-report (9).md",
    REPOSITORY_ROOT / "deep-research-report (10).md",
)

PROSPECTIVE_PATHS = (
    PROTOCOL_PATH,
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_POST_CAP_SUCCESSOR_NO_RUN_SCREEN_20260725.md",
    REPOSITORY_ROOT / "docs" / "TRKH_5CLASS_RESEARCH_JOURNAL.md",
    REPOSITORY_ROOT / "docs" / "TODO_TRKH_5CLASS.md",
    Path(__file__).resolve(),
    REPOSITORY_ROOT
    / "tests"
    / "test_build_trkh_pair_surface_ddf_a0_lock.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _strings_sha256(values: Iterable[object]) -> str:
    encoded = [str(value).encode("utf-8") for value in values]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
    for value in encoded:
        digest.update(np.asarray([len(value)], dtype=np.int64).tobytes())
        digest.update(value)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _npz_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    payload = {
        name: _array_sha256(np.asarray(arrays[name]))
        for name in sorted(arrays)
    }
    return _json_sha256(payload)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _external_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(DDF_REPOSITORY), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _immutable(path: Path) -> Dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_cohort() -> Dict[str, np.ndarray]:
    with np.load(COHORT_ARRAYS, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    expected = {
        "sample_indices": ((ROWS,), np.dtype(np.int64)),
        "targets": ((ROWS,), np.dtype(np.int64)),
        "folds": ((ROWS,), np.dtype(np.int64)),
        "source_stems": ((ROWS,), None),
        "keeper_probabilities": ((ROWS, CLASSES), np.dtype(np.float32)),
        "model_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "crop_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "image_paths": ((ROWS,), None),
        "label_paths": ((ROWS,), None),
    }
    if set(arrays) != set(expected):
        raise ValueError(f"Cohort keys differ: {sorted(arrays)}")
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape:
            raise ValueError(f"{name} shape differs: {value.shape}")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} dtype differs: {value.dtype}")
    if len(set(arrays["sample_indices"].tolist())) != ROWS:
        raise ValueError("Cohort sample indices are not unique")
    if len(set(arrays["source_stems"].astype(str).tolist())) != SOURCES:
        raise ValueError("Cohort unique-source count differs")
    if not np.isfinite(arrays["keeper_probabilities"]).all():
        raise ValueError("Keeper probabilities are non-finite")
    if not np.isfinite(arrays["model_boxes"]).all():
        raise ValueError("Model boxes are non-finite")
    if bool((arrays["model_boxes"] < 0).any()):
        raise ValueError("Model boxes contain a negative component")
    return arrays


def _load_cidt_clean() -> Dict[int, Mapping[str, str]]:
    rows: Dict[int, Mapping[str, str]] = {}
    with CIDT_PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "clean":
                continue
            sample_index = int(row["sample_index"])
            if sample_index in rows:
                raise ValueError(f"Duplicate CIDT clean row {sample_index}")
            rows[sample_index] = row
    if len(rows) != 9215 or sorted(rows) != list(range(9215)):
        raise ValueError("CIDT clean rows differ from 0..9214")
    forbidden = {"val", "valid", "validation", "test"}
    for sample_index, row in rows.items():
        parts = {part.casefold() for part in Path(row["image_path"]).parts}
        if "train" not in parts or parts.intersection(forbidden):
            raise ValueError(f"CIDT row {sample_index} is not train-only")
    return rows


def _bbox_masks(model_boxes: np.ndarray) -> np.ndarray:
    masks = np.zeros((ROWS, 16, 16), dtype=np.bool_)
    for index, (cx, cy, width, height) in enumerate(
        model_boxes.astype(np.float64)
    ):
        left = float(np.clip(cx - width / 2.0, 0.0, 1.0))
        right = float(np.clip(cx + width / 2.0, 0.0, 1.0))
        top = float(np.clip(cy - height / 2.0, 0.0, 1.0))
        bottom = float(np.clip(cy + height / 2.0, 0.0, 1.0))
        x0 = min(int(math.floor(left * 16.0)), 15)
        y0 = min(int(math.floor(top * 16.0)), 15)
        x1 = max(x0 + 1, min(int(math.ceil(right * 16.0)), 16))
        y1 = max(y0 + 1, min(int(math.ceil(bottom * 16.0)), 16))
        masks[index, y0:y1, x0:x1] = True
    if bool((masks.sum(axis=(1, 2)) == 0).any()):
        raise ValueError("A rasterized bbox mask is empty")
    return masks


def _epoch_orders(indices: np.ndarray, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    orders: List[np.ndarray] = []
    hashes: List[str] = []
    for _ in range(EPOCHS):
        order = indices[rng.permutation(indices.size)].astype(
            np.int64, copy=False
        )
        if set(order.tolist()) != set(indices.tolist()):
            raise ValueError("An epoch order is not a complete permutation")
        orders.append(order)
        hashes.append(_array_sha256(order))
    return {
        "algorithm": "numpy_default_rng_pcg64_permutation",
        "seed": int(seed),
        "rows_per_epoch": int(indices.size),
        "batches_per_epoch": int(
            (int(indices.size) + BATCH_SIZE - 1) // BATCH_SIZE
        ),
        "optimizer_steps": int(
            EPOCHS * ((int(indices.size) + BATCH_SIZE - 1) // BATCH_SIZE)
        ),
        "per_epoch_sha256": hashes,
        "all_epochs_sha256": _array_sha256(np.concatenate(orders)),
    }


def _sattolo_partners(
    held: np.ndarray,
    sources: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    for attempt in range(1, 1001):
        permutation = np.arange(held.size, dtype=np.int64)
        for index in range(held.size - 1, 0, -1):
            partner = int(rng.integers(0, index))
            permutation[index], permutation[partner] = (
                permutation[partner],
                permutation[index],
            )
        partners = held[permutation]
        if bool((partners == held).any()):
            raise RuntimeError("Sattolo permutation contains a fixed point")
        if not bool((sources[partners] == sources[held]).any()):
            return partners.astype(np.int64, copy=False), attempt
    raise RuntimeError("Unable to construct a source-different Sattolo cycle")


def _dephase_offsets(
    held: np.ndarray,
    sample_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    offsets = np.empty((held.size, len(MAP_RESOLUTIONS), 2), dtype=np.int64)
    for row, cohort_position in enumerate(held.tolist()):
        sample_index = int(sample_indices[cohort_position])
        for block, resolution in enumerate(MAP_RESOLUTIONS):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, sample_index, block])
            )
            while True:
                dy = int(rng.integers(0, resolution))
                dx = int(rng.integers(0, resolution))
                if dy != 0 or dx != 0:
                    break
            offsets[row, block] = (dy, dx)
    if bool(((offsets[..., 0] == 0) & (offsets[..., 1] == 0)).any()):
        raise RuntimeError("A dephase offset is zero")
    return offsets


def _partition_summary(
    indices: np.ndarray,
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_indices: np.ndarray,
    sources: np.ndarray,
) -> Dict[str, object]:
    return {
        "rows": int(indices.size),
        "cohort_positions_sha256": _array_sha256(indices),
        "sample_indices_sha256": _array_sha256(sample_indices[indices]),
        "targets_sha256": _array_sha256(targets[indices]),
        "sources_sha256": _strings_sha256(sources[indices]),
        "target_counts": {
            str(target): int((targets[indices] == target).sum())
            for target in CLASS_ORDER
        },
        "tp1": int(
            ((targets[indices] == 1) & (predictions[indices] == 1)).sum()
        ),
        "fn1": int(
            ((targets[indices] == 1) & (predictions[indices] != 1)).sum()
        ),
        "restricted_fp": int(
            (
                np.isin(targets[indices], RIVALS)
                & (predictions[indices] == 1)
            ).sum()
        ),
    }


def _fold_payload(
    outer_fold: int,
    *,
    folds: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_indices: np.ndarray,
    sources: np.ndarray,
) -> Dict[str, object]:
    calibration_fold = CALIBRATION_FOLDS[outer_fold]
    held = np.flatnonzero(folds == outer_fold).astype(np.int64)
    calibration = np.flatnonzero(folds == calibration_fold).astype(np.int64)
    fit = np.flatnonzero(
        (folds != outer_fold) & (folds != calibration_fold)
    ).astype(np.int64)
    partitions = {"fit": fit, "calibration": calibration, "held": held}
    source_sets = {
        name: set(sources[indices].tolist())
        for name, indices in partitions.items()
    }
    overlaps = {
        f"{left}_{right}": sorted(source_sets[left].intersection(source_sets[right]))
        for left, right in (
            ("fit", "calibration"),
            ("fit", "held"),
            ("calibration", "held"),
        )
    }
    if any(overlaps.values()):
        raise ValueError(f"Fold {outer_fold} has source overlap: {overlaps}")
    if int((targets[calibration] == 4).sum()) != 3:
        raise ValueError(
            f"Fold {outer_fold} calibration class-4 support differs"
        )
    if int((targets[fit] == 4).sum()) < 4:
        raise ValueError(f"Fold {outer_fold} fit class-4 support is too small")

    primary_init_seed = PRIMARY_SEED + 100 * outer_fold
    primary_order_seed = primary_init_seed + 1
    repeat_init_seed = PRIMARY_SEED + REPEAT_OFFSET + 100 * outer_fold
    repeat_order_seed = repeat_init_seed + 1
    sattolo_seed = PRIMARY_SEED + SATTOLO_OFFSET + outer_fold
    dephase_seed = PRIMARY_SEED + DEPHASE_OFFSET + outer_fold
    partners, attempts = _sattolo_partners(held, sources, sattolo_seed)
    offsets = _dephase_offsets(held, sample_indices, dephase_seed)
    return {
        "outer_fold": outer_fold,
        "calibration_fold": calibration_fold,
        "fit_folds": sorted(
            set(range(FOLDS)) - {outer_fold, calibration_fold}
        ),
        "source_overlap": 0,
        "fit": _partition_summary(
            fit,
            targets=targets,
            predictions=predictions,
            sample_indices=sample_indices,
            sources=sources,
        ),
        "calibration": _partition_summary(
            calibration,
            targets=targets,
            predictions=predictions,
            sample_indices=sample_indices,
            sources=sources,
        ),
        "held": _partition_summary(
            held,
            targets=targets,
            predictions=predictions,
            sample_indices=sample_indices,
            sources=sources,
        ),
        "primary_initialization_seed": primary_init_seed,
        "repeat_initialization_seed": repeat_init_seed,
        "primary_orders": _epoch_orders(fit, primary_order_seed),
        "repeat_orders": _epoch_orders(fit, repeat_order_seed),
        "sattolo": {
            "algorithm": "sattolo_rejection_until_source_stem_differs",
            "seed": sattolo_seed,
            "attempts": attempts,
            "victim_cohort_positions_sha256": _array_sha256(held),
            "partner_cohort_positions_sha256": _array_sha256(partners),
            "partner_sample_indices_sha256": _array_sha256(
                sample_indices[partners]
            ),
            "fixed_points": int((partners == held).sum()),
            "same_source_pairs": int(
                (sources[partners] == sources[held]).sum()
            ),
        },
        "dephase": {
            "algorithm": "seedsequence_sample_block_pcg64_nonzero_roll",
            "seed": dephase_seed,
            "map_resolutions": list(MAP_RESOLUTIONS),
            "shape": list(offsets.shape),
            "offsets_sha256": _array_sha256(offsets),
            "zero_pairs": int(
                ((offsets[..., 0] == 0) & (offsets[..., 1] == 0)).sum()
            ),
        },
    }


def _parameter_oracles() -> Dict[str, object]:
    common = {
        "conv_3_16_k3": 3 * 16 * 3 * 3,
        "bn_16_pre": 2 * 16,
        "bn_16_post": 2 * 16,
        "conv_16_32_k3": 16 * 32 * 3 * 3,
        "bn_32_pre": 2 * 32,
        "bn_32_post": 2 * 32,
        "evidence_and_attention_heads": 2 * (32 * 4 + 4),
    }
    ddf = {
        "ddf16_spatial": 16 * 9 + 9,
        "ddf16_channel": (16 * 4 + 4) + (4 * 144 + 144) + 144,
        "ddf32_spatial": 32 * 9 + 9,
        "ddf32_channel": (32 * 6 + 6) + (6 * 288 + 288) + 288,
    }
    static = {
        "static16": 16 * 9 + (16 * 28 + 28) + (28 * 16 + 16),
        "static32": 32 * 9 + (32 * 39 + 39) + (39 * 32 + 32),
    }
    common_total = sum(common.values())
    ddf_total = common_total + sum(ddf.values())
    static_total = common_total + sum(static.values())
    if ddf_total != 9380 or static_total != 9435:
        raise RuntimeError(
            f"Parameter oracle differs: ddf={ddf_total}, static={static_total}"
        )
    return {
        "common_components": common,
        "ddf_components": ddf,
        "static_components": static,
        "ddf_full": ddf_total,
        "static_matched": static_total,
        "static_minus_ddf": static_total - ddf_total,
        "static_relative_excess": static_total / ddf_total - 1.0,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _first_package_version(*names: str) -> str:
    for name in names:
        value = _package_version(name)
        if value != "not-installed":
            return value
    return "not-installed"


def _keeper_information(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    probabilities64 = probabilities.astype(np.float64)
    strongest_positions = np.argmax(
        probabilities64[:, list(RIVALS)], axis=1
    )
    strongest_rivals = np.asarray(RIVALS, dtype=np.int64)[strongest_positions]
    strongest_probabilities = probabilities64[
        np.arange(ROWS), strongest_rivals
    ]
    union_margin = np.log(probabilities64[:, 1]) - np.log(
        strongest_probabilities
    )
    union_target = (targets == 1).astype(np.int64)
    pairs: Dict[str, object] = {}
    for rival in RIVALS:
        mask = np.isin(targets, [1, rival])
        pair_target = (targets[mask] == 1).astype(np.int64)
        pair_margin = np.log(probabilities64[mask, 1]) - np.log(
            probabilities64[mask, rival]
        )
        pairs[f"1_vs_{rival}"] = {
            "rows": int(mask.sum()),
            "cohort_positions_sha256": _array_sha256(
                np.flatnonzero(mask).astype(np.int64)
            ),
            "margin_sha256": _array_sha256(pair_margin),
            "auroc": float(roc_auc_score(pair_target, pair_margin)),
            "auprc": float(average_precision_score(pair_target, pair_margin)),
        }
    return {
        "definition": "log_p1_minus_log_max_p0_p2_p4",
        "tie_break": "lower_class_index",
        "strongest_rivals_sha256": _array_sha256(strongest_rivals),
        "union_margin_sha256": _array_sha256(union_margin),
        "union_auroc": float(roc_auc_score(union_target, union_margin)),
        "union_auprc": float(
            average_precision_score(union_target, union_margin)
        ),
        "pairs": pairs,
    }


def _full_train_baseline(
    cidt: Mapping[int, Mapping[str, str]],
) -> Dict[str, object]:
    targets = np.asarray(
        [int(cidt[index]["target_index"]) for index in range(9215)],
        dtype=np.int64,
    )
    predictions = np.asarray(
        [int(cidt[index]["keeper_prediction"]) for index in range(9215)],
        dtype=np.int64,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[1],
        zero_division=0,
    )
    return {
        "rows": 9215,
        "targets_sha256": _array_sha256(targets),
        "keeper_predictions_sha256": _array_sha256(predictions),
        "target_counts": np.bincount(targets, minlength=CLASSES).tolist(),
        "prediction_counts": np.bincount(
            predictions, minlength=CLASSES
        ).tolist(),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "class1_precision": float(precision[0]),
        "class1_recall": float(recall[0]),
        "class1_f1": float(f1[0]),
        "class1_support": int(support[0]),
    }


def build_lock() -> Dict[str, object]:
    immutable_paths = {
        "cohort_arrays": COHORT_ARRAYS,
        "model_srgb": MODEL_SRGB,
        "valid_masks_packbits": VALID_MASKS_PACKBITS,
        "cache_summary": CACHE_SUMMARY,
        "cache_replay_summary": CACHE_REPLAY_SUMMARY,
        "cache_formal_manifest": CACHE_FORMAL_MANIFEST,
        "cache_artifact_manifest": CACHE_ARTIFACT_MANIFEST,
        "cidt_predictions": CIDT_PREDICTIONS,
        "keeper_checkpoint": KEEPER_CHECKPOINT,
        "keeper_config": KEEPER_CONFIG,
        "keeper_launcher_args": KEEPER_LAUNCHER_ARGS,
        "current_best_commands": CURRENT_BEST_COMMANDS,
        "current_best_history": CURRENT_BEST_HISTORY,
        "ddf_module": DDF_MODULE,
        "ddf_resnet": DDF_RESNET,
        "ddf_license": DDF_LICENSE,
        "ddf_readme": DDF_README,
        "ddf_paper": DDF_PAPER,
    }
    required = [
        PROTOCOL_PATH,
        *immutable_paths.values(),
        *NO_REPEAT_DOCS,
        *PROTECTED_UNTRACKED,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing lock inputs: {missing}")
    immutable = {
        name: _immutable(path) for name, path in immutable_paths.items()
    }
    for name, expected in EXPECTED_SHA256.items():
        if immutable[name]["sha256"] != expected:
            raise ValueError(f"{name} SHA-256 differs")

    if _external_git("rev-parse", "HEAD") != DDF_COMMIT:
        raise ValueError("Official DDF repository commit differs")
    if _external_git("status", "--porcelain=v1"):
        raise ValueError("Official DDF repository is dirty")

    cache_summary = json.loads(CACHE_SUMMARY.read_text(encoding="utf-8"))
    cache_replay = json.loads(
        CACHE_REPLAY_SUMMARY.read_text(encoding="utf-8")
    )
    if not cache_summary["automatic_passed"]:
        raise ValueError("Materialized cache automatic checks did not pass")
    if not cache_replay["passed"]:
        raise ValueError("Materialized cache replay did not pass")
    if cache_summary["validation_data_used"] or cache_summary["test_data_used"]:
        raise ValueError("Materialized cache reports validation/test access")
    if (
        cache_replay["validation_data_used"]
        or cache_replay["test_data_used"]
    ):
        raise ValueError("Materialized replay reports validation/test access")

    cohort = _load_cohort()
    sample_indices = cohort["sample_indices"].astype(np.int64, copy=False)
    targets = cohort["targets"].astype(np.int64, copy=False)
    folds = cohort["folds"].astype(np.int64, copy=False)
    sources = cohort["source_stems"].astype(str)
    probabilities = cohort["keeper_probabilities"].astype(
        np.float32, copy=False
    )
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    if _array_sha256(sample_indices) != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError("Cohort sample-index hash differs")

    image_cache = np.load(MODEL_SRGB, mmap_mode="r", allow_pickle=False)
    valid_cache = np.load(
        VALID_MASKS_PACKBITS, mmap_mode="r", allow_pickle=False
    )
    try:
        if image_cache.shape != (ROWS, 3, 256, 256):
            raise ValueError(f"Image cache shape differs: {image_cache.shape}")
        if image_cache.dtype != np.uint8:
            raise ValueError(f"Image cache dtype differs: {image_cache.dtype}")
        if valid_cache.shape != (ROWS, 8192):
            raise ValueError(f"Valid-mask cache shape differs: {valid_cache.shape}")
        if valid_cache.dtype != np.uint8:
            raise ValueError(f"Valid-mask cache dtype differs: {valid_cache.dtype}")
    finally:
        del image_cache
        del valid_cache

    cidt = _load_cidt_clean()
    maximum_probability_delta = 0.0
    for position, sample_index in enumerate(sample_indices.tolist()):
        row = cidt[int(sample_index)]
        if int(row["target_index"]) != int(targets[position]):
            raise ValueError(f"CIDT target mismatch at {sample_index}")
        if int(row["fold"]) != int(folds[position]):
            raise ValueError(f"CIDT fold mismatch at {sample_index}")
        if row["source_stem"] != sources[position]:
            raise ValueError(f"CIDT source mismatch at {sample_index}")
        if Path(row["image_path"]).resolve() != Path(
            str(cohort["image_paths"][position])
        ).resolve():
            raise ValueError(f"CIDT image path mismatch at {sample_index}")
        if int(row["keeper_prediction"]) != int(predictions[position]):
            raise ValueError(f"CIDT prediction mismatch at {sample_index}")
        cidt_probabilities = np.asarray(
            [float(row[f"keeper_prob_{index}"]) for index in range(CLASSES)],
            dtype=np.float32,
        )
        maximum_probability_delta = max(
            maximum_probability_delta,
            float(np.max(np.abs(cidt_probabilities - probabilities[position]))),
        )

    counts = {
        "tp1": int(((targets == 1) & (predictions == 1)).sum()),
        "fn1": int(((targets == 1) & (predictions != 1)).sum()),
        "restricted_fp": int(
            (np.isin(targets, RIVALS) & (predictions == 1)).sum()
        ),
        "0_to_1": int(((targets == 0) & (predictions == 1)).sum()),
        "2_to_1": int(((targets == 2) & (predictions == 1)).sum()),
        "4_to_1": int(((targets == 4) & (predictions == 1)).sum()),
    }
    expected_counts = {
        "tp1": 528,
        "fn1": 13,
        "restricted_fp": 222,
        "0_to_1": 158,
        "2_to_1": 54,
        "4_to_1": 10,
    }
    if counts != expected_counts:
        raise ValueError(f"Cohort counts differ: {counts}")

    source_fold_counts: Dict[str, set] = {}
    for source, fold in zip(sources.tolist(), folds.tolist()):
        source_fold_counts.setdefault(source, set()).add(int(fold))
    if any(len(values) != 1 for values in source_fold_counts.values()):
        raise ValueError("A source stem spans multiple folds")

    bbox_masks = _bbox_masks(
        cohort["model_boxes"].astype(np.float32, copy=False)
    )
    fold_payloads = [
        _fold_payload(
            fold,
            folds=folds,
            targets=targets,
            predictions=predictions,
            sample_indices=sample_indices,
            sources=sources,
        )
        for fold in range(FOLDS)
    ]
    visual_positions = np.asarray(
        [
            int(np.flatnonzero(sample_indices == selected)[0])
            for selected in VISUAL_SAMPLE_INDICES.tolist()
        ],
        dtype=np.int64,
    )
    protocol = _immutable(PROTOCOL_PATH)
    protected = {
        str(path.relative_to(REPOSITORY_ROOT)): _immutable(path)
        for path in PROTECTED_UNTRACKED
    }
    no_repeat = {
        str(path.relative_to(REPOSITORY_ROOT)): _immutable(path)
        for path in NO_REPEAT_DOCS
    }
    prospective_sources = {
        str(path.relative_to(REPOSITORY_ROOT)): _immutable(path)
        for path in (
            Path(__file__).resolve(),
            REPOSITORY_ROOT
            / "tests"
            / "test_build_trkh_pair_surface_ddf_a0_lock.py",
        )
    }
    array_hashes = {
        name: _array_sha256(value)
        for name, value in sorted(cohort.items())
    }

    return {
        "protocol_id": "trkh_pair_surface_ddf_a0_20260725",
        "state": "prospective_no_candidate_implementation_or_observation",
        "created_utc": "2026-07-25T00:00:00Z",
        "lock_parent_commit": LOCK_PARENT_COMMIT,
        "repository": {
            "root": str(REPOSITORY_ROOT),
            "branch": _git("branch", "--show-current"),
            "head_at_lock_build": _git("rev-parse", "HEAD"),
            "upstream_at_lock_build": _git(
                "rev-parse", f"origin/{BRANCH}"
            ),
        },
        "protocol": protocol,
        "prospective_sources": prospective_sources,
        "immutable_inputs": immutable,
        "no_repeat_documents": no_repeat,
        "protected_untracked": protected,
        "official_source": {
            "name": "Decoupled Dynamic Filter Networks",
            "paper_url": (
                "https://openaccess.thecvf.com/content/CVPR2021/html/"
                "Zhou_Decoupled_Dynamic_Filter_Networks_CVPR_2021_paper.html"
            ),
            "repository_url": "https://github.com/theFoxofSky/ddfnet",
            "repository_commit": DDF_COMMIT,
            "repository_clean": True,
            "license": "MIT",
            "pretrained_parameters_allowed": False,
        },
        "cache": {
            "rows": ROWS,
            "unique_sources": SOURCES,
            "image_shape": [ROWS, 3, 256, 256],
            "image_dtype": "uint8",
            "valid_mask_packbits_shape": [ROWS, 8192],
            "valid_mask_unpack_shape": [ROWS, 256, 256],
            "formal_checks_passed": True,
            "fresh_process_replay_passed": True,
            "array_payload_sha256": _npz_payload_sha256(cohort),
            "array_sha256": array_hashes,
        },
        "cohort": {
            "sample_indices_sha256": _array_sha256(sample_indices),
            "targets_sha256": _array_sha256(targets),
            "folds_sha256": _array_sha256(folds),
            "source_stems_sha256": _strings_sha256(sources),
            "keeper_probabilities_sha256": _array_sha256(probabilities),
            "model_boxes_sha256": _array_sha256(
                cohort["model_boxes"].astype(np.float32)
            ),
            "crop_boxes_sha256": _array_sha256(
                cohort["crop_boxes"].astype(np.float32)
            ),
            "image_paths_sha256": _strings_sha256(cohort["image_paths"]),
            "label_paths_sha256": _strings_sha256(cohort["label_paths"]),
            "bbox_masks_sha256": _array_sha256(bbox_masks),
            "bbox_mask_min_cells": int(bbox_masks.sum(axis=(1, 2)).min()),
            "bbox_mask_max_cells": int(bbox_masks.sum(axis=(1, 2)).max()),
            "counts": counts,
            "cidt_predictions_exact": True,
            "cidt_probability_max_abs_delta": maximum_probability_delta,
            "probability_source_for_calibration": "materialized_cohort_arrays",
            "prediction_source_for_full_diagnostic": "cidt_clean_table",
        },
        "keeper_information": _keeper_information(targets, probabilities),
        "full_train_baseline": _full_train_baseline(cidt),
        "folds": fold_payloads,
        "input_transform": {
            "source": "model_srgb_uint8.npy",
            "uint8_scale": 1.0 / 255.0,
            "dtype": "float32",
            "normalization_order": "normalize_then_resize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": {
                "source_size": [256, 256],
                "target_size": [64, 64],
                "mode": "bilinear",
                "align_corners": False,
                "antialias": False,
            },
            "augmentation": False,
            "bbox_is_inference_input": False,
            "valid_mask_is_inference_input": False,
        },
        "architecture": {
            "roles": [
                "ddf_full",
                "static_matched",
                "ddf_spatial_only",
                "ddf_channel_only",
                "ddf_full_repeat",
            ],
            "score_order": [
                "union",
                "class1_vs_0",
                "class1_vs_2",
                "class1_vs_4",
            ],
            "channels": [3, 16, 32],
            "map_resolutions": list(MAP_RESOLUTIONS),
            "kernel_size": 3,
            "padding": 1,
            "squeeze_widths": [4, 6],
            "ddf_combine": "multiplicative",
            "ddf_spatial_filter": "sample_and_pixel_specific",
            "ddf_channel_filter": "sample_and_channel_specific",
            "filter_tap_order": "row_major",
            "filter_std_correction": 1,
            "filter_epsilon": 1e-10,
            "filter_gain": math.sqrt(2.0) / 3.0,
            "neutral_factor": 1.0,
            "static_intermediate_widths": [28, 39],
            "parameter_oracles": _parameter_oracles(),
            "maximum_candidate_parameters": 15000,
            "pretrained": False,
            "custom_runtime_operator": False,
        },
        "initialization": {
            "tensor_seed": (
                "first_63_bits_sha256_decimal_seed_colon_canonical_name"
            ),
            "generation_device": "cpu",
            "conv_weight": "kaiming_uniform_a_sqrt5",
            "conv_bias": "uniform_plus_minus_inverse_sqrt_fan_in",
            "batchnorm_weight": 1.0,
            "batchnorm_bias": 0.0,
            "channel_filter_scale": "normal_mean0_std_sqrt2_over3",
            "matching_tensor_copy": "bit_exact",
        },
        "optimization": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "scheduler": "cosine_to_zero",
            "warmup": False,
            "precision": "float32",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "cpu_numeric_threads": 1,
            "augmentation": False,
            "dropout": False,
            "ema": False,
            "early_stopping": False,
            "checkpoint_selection": False,
            "oversampling": False,
            "row_duplication": False,
            "bbox_attention_weight": 0.05,
            "bbox_attention_clamp_min": 1e-8,
            "pair_loss_weight": 0.5,
            "binary_loss": "bce_with_logits",
            "pair_loss_reduction": (
                "fit_partition_fixed_balanced_weights_sum_div_batch_rows"
            ),
        },
        "calibration": {
            "scaler": "sklearn_StandardScaler",
            "classifier": "sklearn_LogisticRegression",
            "C": 0.1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-8,
            "random_state": PRIMARY_SEED,
            "union_features": [
                "keeper_log_p1_minus_log_max_p0_p2_p4",
                "raw_union_score",
                "raw_pair_score_for_keeper_strongest_rival",
            ],
            "pair_features": [
                "keeper_log_p1_minus_log_prival",
                "raw_corresponding_pair_score",
            ],
            "threshold_candidates": (
                "sorted_unique_finite_calibration_probabilities_plus_no_action"
            ),
            "suppression_comparison": "candidate_probability_le_threshold",
            "minimum_tp_retention": 0.97,
            "minimum_correction_fraction_of_all_changes": 2.0 / 3.0,
            "replacement": "keeper_highest_probability_nonclass1",
            "replacement_tie_break": "lower_class_index",
            "production_artifact": False,
        },
        "scientific_gates": {
            "pooled_auroc_over_keeper": 0.015,
            "pooled_auprc_over_keeper": 0.005,
            "pooled_auroc_over_static": 0.015,
            "pooled_auroc_over_single_branch": 0.010,
            "fold_wins_over_keeper_and_static": 4,
            "pair_auroc_keeper_tolerance": -0.005,
            "pair_auroc_over_static_all": True,
            "cross_sample_raw_auroc_drop": 0.020,
            "spatial_dephase_raw_auroc_drop": 0.015,
            "beats_both_neutral_factor_ablations": True,
            "oof_tp_retention": 0.97,
            "oof_restricted_fp_rejected": 45,
            "correction_to_harm_ratio": 2.0,
            "full_class1_precision_gain": 0.030,
            "full_class1_f1_gain": 0.015,
            "full_class1_recall_min_delta": -0.030,
            "full_macro_f1_gain": 0.003,
            "repeat_auroc_max_abs_delta": 0.010,
            "repeat_action_agreement": 0.97,
            "mean_attention_bbox_mass": 0.85,
            "rows_attention_bbox_mass_at_least_070_fraction": 0.90,
            "finite_everywhere": True,
            "conjunctive": True,
        },
        "engineering_gates": {
            "fp64_numpy_oracle_max_abs": 1e-10,
            "onnx_opset": 17,
            "onnx_standard_domains_only": True,
            "onnxruntime_max_abs": 1e-5,
            "tensorrt_plugin_allowed": False,
            "tensorrt_parser_and_engine_required": True,
            "branch_batch1_fp16_mean_ms": 2.0,
            "branch_batch1_fp16_p95_ms": 2.5,
            "branch_batch32_fp16_mean_ms": 2.5,
            "branch_batch32_fp16_p95_ms": 3.0,
            "branch_peak_cuda_bytes": 134217728,
            "future_keeper_batch1_mean_ratio": 1.15,
            "future_keeper_batch1_p95_ratio": 1.15,
            "future_keeper_throughput_ratio": 0.85,
            "future_keeper_peak_cuda_ratio": 1.15,
            "future_keeper_peak_cuda_bytes": 6442450944,
        },
        "synthetic_engineering_pilot": {
            "candidate_metric": False,
            "cohort_pixel_or_label_read": False,
            "parameters": 8687,
            "batch1_fp16_mean_ms": 1.806,
            "batch1_fp16_p95_ms": 2.208,
            "batch32_fp16_mean_ms": 1.616,
            "batch32_fp16_p95_ms": 1.903,
            "batch1_peak_cuda_bytes": 1699328,
            "batch32_peak_cuda_bytes": 52781056,
            "onnxruntime_max_abs": 1.1920928955078125e-7,
            "onnx_nodes": 354,
            "tensorrt_version": "10.7",
            "tensorrt_engine_bytes": 2295020,
            "temporary_files_retained": False,
        },
        "keeper_engineering_reference": {
            "random_input_eager_fp16_batch1_mean_ms": 17.569,
            "random_input_eager_fp16_batch1_p95_ms": 19.665,
            "peak_cuda_bytes": 62196736,
            "parameters": 7245590,
        },
        "replay": {
            "fresh_process_required": True,
            "same_host_and_runtime_required": True,
            "numeric_arrays_max_abs": 0.0,
            "states_byte_exact": True,
            "decisions_exact": True,
            "ordered_access_ledger_exact": True,
            "hashes_exact": True,
        },
        "visual": {
            "sample_indices": VISUAL_SAMPLE_INDICES.tolist(),
            "sample_indices_sha256": _array_sha256(VISUAL_SAMPLE_INDICES),
            "cohort_positions_sha256": _array_sha256(visual_positions),
            "rows": int(VISUAL_SAMPLE_INDICES.size),
            "all_target4_rows": int(
                (targets[visual_positions[-10:]] == 4).sum()
            ),
            "authorized_only_after_clean_scientific_pass": True,
        },
        "resource_limits_per_formal_or_replay": {
            "wall_seconds": 1800,
            "peak_cuda_bytes": 2147483648,
            "peak_process_rss_bytes": 8589934592,
            "temporary_bytes": 1073741824,
            "retained_bytes": 268435456,
            "surviving_child_processes": 0,
        },
        "access": {
            "dynamic_file_open_ledger_required": True,
            "ledger_before_candidate_inputs": True,
            "raw_dataset_open_forbidden": True,
            "validation_forbidden": True,
            "test_forbidden": True,
            "keeper_forward_forbidden": True,
            "external_model_forward_forbidden": True,
            "formal_runs_authorized": 0,
            "replay_runs_authorized": 0,
            "separate_execution_authorization_required": True,
        },
        "forbidden": {
            "candidate_implementation_before_lock_push": True,
            "validation": True,
            "test": True,
            "raw_dataset_read_or_write": True,
            "external_or_pretrained_weights": True,
            "production_integration": True,
            "full_train": True,
            "current_best_command_edit": True,
            "nearby_sweep_after_failure": True,
        },
        "observations": {
            "candidate_implementation_sha256": None,
            "candidate_state_sha256": None,
            "candidate_scores_sha256": None,
            "candidate_thresholds_sha256": None,
            "candidate_actions_sha256": None,
            "candidate_metrics": None,
            "validation_metrics": None,
            "test_metrics": None,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "numpy": np.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "pillow": _package_version("Pillow"),
            "onnx": _package_version("onnx"),
            "onnxruntime": _first_package_version(
                "onnxruntime", "onnxruntime-gpu"
            ),
            "tensorrt": _package_version("tensorrt"),
        },
    }


def write_lock(lock: Mapping[str, object]) -> str:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = _sha256(LOCK_PATH)
    LOCK_SHA_PATH.write_text(
        f"{digest}  {LOCK_PATH.name}\n",
        encoding="ascii",
    )
    return digest


def _status_paths() -> List[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    )
    records = [record for record in result.stdout.split(b"\0") if record]
    paths: List[str] = []
    for record in records:
        decoded = record.decode("utf-8", errors="surrogateescape")
        status = decoded[:2]
        if status.startswith("R") or status.startswith("C"):
            raise RuntimeError("Rename/copy status is not allowed at lock write")
        paths.append(decoded[3:].replace("\\", "/"))
    return paths


def _relative(path: Path) -> str:
    return str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")


def _assert_lock_parent_for_write() -> None:
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", f"origin/{BRANCH}")
    branch = _git("branch", "--show-current")
    if head != LOCK_PARENT_COMMIT or upstream != LOCK_PARENT_COMMIT:
        raise RuntimeError(
            "Prospective lock writes require HEAD and upstream at "
            f"{LOCK_PARENT_COMMIT}; got HEAD={head}, upstream={upstream}. "
            "Use --check-only after the prospective boundary is committed."
        )
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, got {branch}")
    expected = {
        *(_relative(path) for path in PROTECTED_UNTRACKED),
        *(_relative(path) for path in PROSPECTIVE_PATHS),
    }
    optional_lock_paths = {_relative(LOCK_PATH), _relative(LOCK_SHA_PATH)}
    observed = set(_status_paths())
    if observed not in (expected, expected | optional_lock_paths):
        unexpected = sorted(observed - expected - optional_lock_paths)
        missing = sorted(expected - observed)
        raise RuntimeError(
            "Prospective lock worktree differs: "
            f"unexpected={unexpected}, missing={missing}"
        )


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(
            "Unable to verify prospective-lock ancestry: "
            f"{result.stderr.strip()}"
        )
    return result.returncode == 0


def _assert_check_only_matches_frozen(lock: Mapping[str, object]) -> None:
    if not LOCK_PATH.is_file() or not LOCK_SHA_PATH.is_file():
        raise FileNotFoundError("Frozen prospective lock or SHA file is missing")
    sha_parts = LOCK_SHA_PATH.read_text(encoding="ascii").strip().split()
    if len(sha_parts) != 2 or sha_parts[1] != LOCK_PATH.name:
        raise ValueError("Frozen prospective lock SHA record is malformed")
    actual_sha = _sha256(LOCK_PATH)
    if sha_parts[0] != actual_sha:
        raise ValueError(
            "Frozen prospective lock SHA differs: "
            f"recorded={sha_parts[0]}, actual={actual_sha}"
        )

    frozen = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    generated = json.loads(json.dumps(lock, ensure_ascii=False))
    for field in ("head_at_lock_build", "upstream_at_lock_build"):
        if frozen["repository"][field] != LOCK_PARENT_COMMIT:
            raise ValueError(f"Frozen repository {field} differs from parent")
        generated["repository"][field] = frozen["repository"][field]
    if generated != frozen:
        raise ValueError(
            "Regenerated prospective payload differs from frozen lock: "
            f"generated={_json_sha256(generated)}, "
            f"frozen={_json_sha256(frozen)}"
        )

    current_head = _git("rev-parse", "HEAD")
    current_upstream = _git("rev-parse", f"origin/{BRANCH}")
    if not _is_ancestor(LOCK_PARENT_COMMIT, current_head):
        raise RuntimeError("Lock parent is not an ancestor of current HEAD")
    if not _is_ancestor(LOCK_PARENT_COMMIT, current_upstream):
        raise RuntimeError("Lock parent is not an ancestor of current upstream")
    allowed = {_relative(path) for path in PROTECTED_UNTRACKED}
    if (
        current_head == LOCK_PARENT_COMMIT
        and current_upstream == LOCK_PARENT_COMMIT
    ):
        allowed.update(_relative(path) for path in PROSPECTIVE_PATHS)
        allowed.update({_relative(LOCK_PATH), _relative(LOCK_SHA_PATH)})
    unexpected = sorted(set(_status_paths()) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected post-lock worktree paths: {unexpected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the prospective TRKH Pair-Surface DDF A0 lock."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build and validate in memory without writing lock files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = build_lock()
    if args.check_only:
        _assert_check_only_matches_frozen(lock)
        print(
            json.dumps(
                {
                    "state": lock["state"],
                    "rows": lock["cache"]["rows"],
                    "protocol_sha256": lock["protocol"]["sha256"],
                    "cohort_payload_sha256": lock["cache"][
                        "array_payload_sha256"
                    ],
                    "sattolo_sha256": [
                        fold["sattolo"]["partner_cohort_positions_sha256"]
                        for fold in lock["folds"]
                    ],
                    "dephase_sha256": [
                        fold["dephase"]["offsets_sha256"]
                        for fold in lock["folds"]
                    ],
                },
                indent=2,
            )
        )
        return
    _assert_lock_parent_for_write()
    digest = write_lock(lock)
    print(f"Wrote {LOCK_PATH}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()

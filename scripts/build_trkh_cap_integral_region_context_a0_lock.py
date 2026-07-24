from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PARENT_COMMIT = "10a35b2a64c6b2d374b8a6a8314bb18fc0a2378e"
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_CAP_INTEGRAL_REGION_CONTEXT_A0_PROTOCOL_20260724.md"
)
LOCK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_CAP_INTEGRAL_REGION_CONTEXT_A0_LOCK_20260724.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")

MAXSEP_ROOT = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_attention_maxsep_prototype_a0_20260721"
)
FEATURE_CACHE = MAXSEP_ROOT / "cohort_block2_features.npy"
GEOMETRY_CACHE = MAXSEP_ROOT / "cohort_geometry.npz"
CIDT_PREDICTIONS = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_cidt_readiness_full_train_20260714"
    / "predictions_all_conditions.csv"
)
KEEPER_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
    / "checkpoints"
    / "best.pt"
)
KEEPER_CONFIG = KEEPER_CHECKPOINT.parents[1] / "resolved_config.json"
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

CAP_REPOSITORY = Path(r"D:\DataAI\external_sources\cap_official")
CAP_PAPER = Path(
    r"D:\DataAI\external_sources\papers"
    r"\Behera_Context_Aware_Attentional_Pooling_AAAI2021.pdf"
)

EXPECTED = {
    "cap_commit": "cae53050af583c2b3fa38081c0abc227199e7dc1",
    "cap_tree": "70f47d05ad0f6dafc8c43db32022a4955f4ce14a",
    "cap_paper": "2fd8cb834e44cd5819894ddf049e972c2f35d227a27727e0def36ec3f0624cb2",
    "cap_license": "624c3e2c494268065b8d00c99087e24520456eef496637d86558da6326f2e12e",
    "cap_train": "099652789ba95dce375f970a947ccd797bd13102c84b005781d877bf3e994ad9",
    "cap_self_attention": "ad67bff1c4a3c0a2c89c3edf08a8d657a86553c44d1a99003e601c67c8a2d8db",
    "cap_seq_attention": "e6ab256d596ab8a8bdd59d69a8e988f36f783d3ef41d4243a4877fd7657cd5fa",
    "cap_roi_pool": "cdf12820d27d64354a7c67fb93052ca2d585a8b46936cdc3c4b7ca2882900eed",
    "cap_loupe": "22816655f907fb79d7ad42a0d35759dda06ac6cfe559a6f143a60699542b5c1e",
    "feature_cache": "e4a833b20b29af09adb0d88a6eddb0768b1ce85a1e13bfd3cfb6215b78d8de2a",
    "geometry_cache": "69f581e1cca66bc64136352ea7f11668e1e50cf4715b2f9ba8c49e8e6f66dc72",
    "cidt_predictions": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "keeper_config": "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    "current_best_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "current_best_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}

ROWS = 763
CHANNELS = 256
FEATURE_SIZE = 16
CLASSES = 5
FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 32
PRIMARY_SEED = 20260724
REPEAT_OFFSET = 100000
SPATIAL_DERANGEMENT_SEED = PRIMARY_SEED + 5000
VISUAL_SAMPLE_INDICES = np.asarray(
    [
        163,
        2370,
        856,
        4613,
        58,
        2563,
        1939,
        2698,
        341,
        2004,
        1066,
        3934,
        2,
        2906,
        345,
        4040,
        195,
        2411,
        791,
        3624,
    ],
    dtype=np.int64,
)
TRAINABLE_ROLES = (
    "global_gap_linear_control",
    "integral_self_only_control",
    "cap_context_candidate",
    "cap_spatial_deranged_control",
    "cap_cross_sample_context_control",
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
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _immutable(path: Path) -> Dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _epoch_orders(indices: np.ndarray, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    hashes: List[str] = []
    arrays: List[np.ndarray] = []
    for _ in range(EPOCHS):
        ordered = indices[rng.permutation(indices.size)].astype(
            np.int64, copy=False
        )
        arrays.append(ordered)
        hashes.append(_array_sha256(ordered))
    return {
        "seed": seed,
        "per_epoch_sha256": hashes,
        "all_epochs_sha256": _array_sha256(np.concatenate(arrays)),
        "rows_per_epoch": int(indices.size),
        "optimizer_steps": int(
            EPOCHS * ((indices.size + BATCH_SIZE - 1) // BATCH_SIZE)
        ),
    }


def _derangement(
    indices: np.ndarray, sample_indices: np.ndarray, sources: Sequence[str], seed: int
) -> np.ndarray:
    ordered = indices[np.argsort(sample_indices[indices], kind="stable")]
    rng = np.random.default_rng(seed)
    offsets = rng.permutation(np.arange(1, ordered.size, dtype=np.int64))
    for offset in offsets:
        partners = np.roll(ordered, -int(offset))
        if all(
            sources[int(left)] != sources[int(right)]
            for left, right in zip(ordered.tolist(), partners.tolist())
        ):
            mapping = np.empty(sample_indices.size, dtype=np.int64)
            mapping.fill(-1)
            mapping[ordered] = partners
            return mapping
    raise ValueError("No label-blind different-source derangement exists")


def _partitioned_derangement(
    partitions: Mapping[str, np.ndarray],
    sample_indices: np.ndarray,
    sources: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    mapping = np.full(sample_indices.size, -1, dtype=np.int64)
    details: Dict[str, object] = {}
    for offset, (name, indices) in enumerate(partitions.items()):
        local = _derangement(
            indices,
            sample_indices,
            sources,
            seed + offset,
        )
        partners = local[indices]
        allowed = set(indices.tolist())
        outside_partition = sum(
            int(partner) not in allowed for partner in partners.tolist()
        )
        source_overlap = sum(
            sources[int(left)] == sources[int(right)]
            for left, right in zip(indices.tolist(), partners.tolist())
        )
        self_pairs = int((indices == partners).sum())
        if (
            outside_partition
            or source_overlap
            or self_pairs
            or len(set(partners.tolist())) != int(indices.size)
        ):
            raise ValueError(f"Invalid {name} cross-sample derangement")
        mapping[indices] = partners
        details[name] = {
            "rows": int(indices.size),
            "mapping_sha256": _array_sha256(partners),
            "source_overlap": int(source_overlap),
            "outside_partition": int(outside_partition),
            "self_pairs": int(self_pairs),
            "unique_partners": len(set(partners.tolist())),
        }
    if bool((mapping < 0).any()):
        raise ValueError("Partitioned cross-sample derangement is incomplete")
    return mapping, details


def _sattolo_permutation(size: int, seed: np.random.SeedSequence) -> np.ndarray:
    values = np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for upper in range(size - 1, 0, -1):
        lower = int(rng.integers(0, upper))
        values[upper], values[lower] = values[lower], values[upper]
    if bool((values == np.arange(size, dtype=np.int64)).any()):
        raise ValueError("Sattolo permutation contains a fixed point")
    return values


def _spatial_permutations(sample_indices: np.ndarray) -> np.ndarray:
    permutations = np.stack(
        [
            _sattolo_permutation(
                27,
                np.random.SeedSequence(
                    [
                        SPATIAL_DERANGEMENT_SEED,
                        int(sample_index),
                    ]
                ),
            )
            for sample_index in sample_indices.tolist()
        ],
        axis=0,
    )
    expected = np.arange(27, dtype=np.int64)
    if not all(
        np.array_equal(np.sort(permutation), expected)
        for permutation in permutations
    ):
        raise ValueError("A spatial row is not a permutation")
    if bool((permutations == expected[None, :]).any()):
        raise ValueError("A spatial permutation contains a fixed point")
    return permutations


def _valid_support_boxes(valid_masks: np.ndarray) -> np.ndarray:
    boxes: List[List[int]] = []
    for mask in valid_masks:
        rows, columns = np.where(mask)
        y0 = int(rows.min())
        y1 = int(rows.max()) + 1
        x0 = int(columns.min())
        x1 = int(columns.max()) + 1
        if not bool(mask[y0:y1, x0:x1].all()):
            raise ValueError("Valid support is not rectangular")
        if int(mask.sum()) != (y1 - y0) * (x1 - x0):
            raise ValueError("Valid support has pixels outside its rectangle")
        boxes.append([x0, y0, x1, y1])
    return np.asarray(boxes, dtype=np.int64)


def _load_cidt_clean() -> Dict[int, Mapping[str, str]]:
    result: Dict[int, Mapping[str, str]] = {}
    with CIDT_PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "clean":
                continue
            sample_index = int(row["sample_index"])
            if sample_index in result:
                raise ValueError(f"Duplicate CIDT sample index {sample_index}")
            result[sample_index] = row
    if len(result) != 9215 or sorted(result) != list(range(9215)):
        raise ValueError("CIDT clean rows differ from 0..9214")
    return result


def _load_inputs() -> Dict[str, np.ndarray]:
    features = np.load(FEATURE_CACHE, allow_pickle=False, mmap_mode="r")
    if features.shape != (ROWS, CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError(f"Feature shape differs: {features.shape}")
    if features.dtype != np.dtype(np.float16):
        raise ValueError(f"Feature dtype differs: {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("Features are non-finite")
    with np.load(GEOMETRY_CACHE, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected = {
        "sample_indices": ((ROWS,), np.dtype(np.int64)),
        "targets": ((ROWS,), np.dtype(np.int64)),
        "folds": ((ROWS,), np.dtype(np.int64)),
        "source_stems": ((ROWS,), None),
        "keeper_probabilities": ((ROWS, CLASSES), np.dtype(np.float32)),
        "valid_masks": ((ROWS, FEATURE_SIZE, FEATURE_SIZE), np.dtype(np.bool_)),
        "bbox_masks": ((ROWS, FEATURE_SIZE, FEATURE_SIZE), np.dtype(np.bool_)),
        "model_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "crop_boxes": ((ROWS, 4), np.dtype(np.float32)),
    }
    if set(arrays) != set(expected):
        raise ValueError(f"Geometry keys differ: {sorted(arrays)}")
    for key, (shape, dtype) in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} shape differs: {arrays[key].shape}")
        if dtype is not None and arrays[key].dtype != dtype:
            raise ValueError(f"{key} dtype differs: {arrays[key].dtype}")
    if len(set(arrays["sample_indices"].tolist())) != ROWS:
        raise ValueError("Cohort sample indices are not unique")
    if not np.isfinite(arrays["keeper_probabilities"]).all():
        raise ValueError("Keeper probabilities are non-finite")
    if not arrays["valid_masks"].reshape(ROWS, -1).any(axis=1).all():
        raise ValueError("A valid mask is empty")
    if bool((arrays["bbox_masks"] & ~arrays["valid_masks"]).any()):
        raise ValueError("A bbox mask leaves valid support")
    arrays["features"] = np.asarray(features)
    return arrays


def _region_boxes() -> np.ndarray:
    resolution = 42
    grid_size = 3
    min_size = 2
    step = resolution / grid_size
    boxes: List[List[int]] = []
    for column1 in range(grid_size + 1):
        for column2 in range(grid_size + 1):
            for row1 in range(grid_size + 1):
                for row2 in range(grid_size + 1):
                    x0 = int(column1 * step)
                    x1 = int(column2 * step)
                    y0 = int(row1 * step)
                    y1 = int(row2 * step)
                    width = x1 - x0
                    height = y1 - y0
                    valid_size = width >= step * min_size or height >= step * min_size
                    if (
                        x1 > x0
                        and y1 > y0
                        and valid_size
                        and not (x0 == y0 == 0 and x1 == y1 == resolution)
                    ):
                        boxes.append([x0, y0, width, height])
    boxes.append([0, 0, resolution, resolution])
    value = np.asarray(boxes, dtype=np.int64)
    if value.shape != (27, 4):
        raise ValueError(f"CAP region geometry differs: {value.shape}")
    return value


def _fold_payload(
    outer_fold: int,
    folds: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_indices: np.ndarray,
    sources: Sequence[str],
) -> Dict[str, object]:
    calibration_fold = (outer_fold + 1) % FOLDS
    held = np.flatnonzero(folds == outer_fold).astype(np.int64)
    calibration = np.flatnonzero(folds == calibration_fold).astype(np.int64)
    fit = np.flatnonzero(
        (folds != outer_fold) & (folds != calibration_fold)
    ).astype(np.int64)
    partitions = {
        "fit": fit,
        "calibration": calibration,
        "held": held,
    }
    source_sets = {
        name: {sources[int(index)] for index in values}
        for name, values in partitions.items()
    }
    if any(
        source_sets[left].intersection(source_sets[right])
        for left, right in (
            ("fit", "calibration"),
            ("fit", "held"),
            ("calibration", "held"),
        )
    ):
        raise ValueError(f"Fold {outer_fold} has source overlap")
    payload: Dict[str, object] = {
        "outer_fold": outer_fold,
        "calibration_fold": calibration_fold,
        "fit_folds": sorted(set(range(FOLDS)) - {outer_fold, calibration_fold}),
        "source_overlap": 0,
    }
    for name, values in partitions.items():
        payload[name] = {
            "rows": int(values.size),
            "indices_sha256": _array_sha256(values),
            "sample_indices_sha256": _array_sha256(sample_indices[values]),
            "targets_sha256": _array_sha256(targets[values]),
            "sources_sha256": _strings_sha256(sources[int(index)] for index in values),
            "tp1": int(((targets[values] == 1) & (predictions[values] == 1)).sum()),
            "fn1": int(((targets[values] == 1) & (predictions[values] != 1)).sum()),
            "restricted_fp": int(
                (np.isin(targets[values], [0, 2, 4]) & (predictions[values] == 1)).sum()
            ),
        }
    primary_seed = PRIMARY_SEED + outer_fold
    repeat_seed = PRIMARY_SEED + REPEAT_OFFSET + outer_fold
    payload["primary_seed"] = primary_seed
    payload["repeat_seed"] = repeat_seed
    payload["primary_orders"] = _epoch_orders(fit, primary_seed)
    payload["repeat_orders"] = _epoch_orders(fit, repeat_seed)
    mapping, mapping_details = _partitioned_derangement(
        partitions,
        sample_indices,
        sources,
        PRIMARY_SEED + 7000 + outer_fold,
    )
    payload["cross_sample_derangement_sha256"] = _array_sha256(mapping)
    payload["cross_sample_derangement_partitions"] = mapping_details
    payload["cross_sample_derangement_source_overlap"] = int(
        sum(
            sources[index] == sources[int(mapping[index])]
            for index in range(ROWS)
        )
    )
    payload["cross_sample_derangement_partition_escape"] = int(
        sum(
            int(detail["outside_partition"])
            for detail in mapping_details.values()
        )
    )
    return payload


def build_lock() -> Dict[str, object]:
    source_files = {
        "cap_paper": CAP_PAPER,
        "cap_license": CAP_REPOSITORY / "LICENSE",
        "cap_train": CAP_REPOSITORY / "train_CAP.py",
        "cap_self_attention": CAP_REPOSITORY / "SelfAttention.py",
        "cap_seq_attention": CAP_REPOSITORY / "SeqAttention.py",
        "cap_roi_pool": CAP_REPOSITORY / "RoiPoolingConv.py",
        "cap_loupe": CAP_REPOSITORY / "loupe_keras.py",
    }
    scientific_files = {
        "feature_cache": FEATURE_CACHE,
        "geometry_cache": GEOMETRY_CACHE,
        "cidt_predictions": CIDT_PREDICTIONS,
        "keeper_checkpoint": KEEPER_CHECKPOINT,
        "keeper_config": KEEPER_CONFIG,
        "current_best_commands": CURRENT_BEST_COMMANDS,
        "current_best_history": CURRENT_BEST_HISTORY,
    }
    required = [PROTOCOL_PATH, *source_files.values(), *scientific_files.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing lock inputs: {missing}")
    cap_commit = _git(CAP_REPOSITORY, "rev-parse", "HEAD")
    cap_tree = _git(CAP_REPOSITORY, "rev-parse", "HEAD^{tree}")
    cap_status = _git(CAP_REPOSITORY, "status", "--porcelain")
    if cap_commit != EXPECTED["cap_commit"] or cap_tree != EXPECTED["cap_tree"]:
        raise ValueError("Pinned CAP repository revision differs")
    if cap_status:
        raise ValueError(f"Pinned CAP repository is dirty: {cap_status}")
    immutable = {
        name: _immutable(path)
        for name, path in {**source_files, **scientific_files}.items()
    }
    for name, metadata in immutable.items():
        if metadata["sha256"] != EXPECTED[name]:
            raise ValueError(f"{name} SHA-256 differs")
    arrays = _load_inputs()
    cidt = _load_cidt_clean()
    sample_indices = arrays["sample_indices"].astype(np.int64, copy=False)
    targets = arrays["targets"].astype(np.int64, copy=False)
    folds = arrays["folds"].astype(np.int64, copy=False)
    probabilities = arrays["keeper_probabilities"].astype(np.float32, copy=False)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    sources = arrays["source_stems"].astype(str).tolist()
    for position, sample_index in enumerate(sample_indices.tolist()):
        row = cidt[sample_index]
        if int(row["target_index"]) != int(targets[position]):
            raise ValueError(f"CIDT target mismatch at {sample_index}")
        if int(row["fold"]) != int(folds[position]):
            raise ValueError(f"CIDT fold mismatch at {sample_index}")
        if row["source_stem"] != sources[position]:
            raise ValueError(f"CIDT source mismatch at {sample_index}")
        if int(row["keeper_prediction"]) != int(predictions[position]):
            raise ValueError(f"CIDT prediction mismatch at {sample_index}")
    counts = {
        "tp1": int(((targets == 1) & (predictions == 1)).sum()),
        "fn1": int(((targets == 1) & (predictions != 1)).sum()),
        "restricted_fp": int(
            (np.isin(targets, [0, 2, 4]) & (predictions == 1)).sum()
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
        raise ValueError(f"Locked cohort counts differ: {counts}")
    if not set(VISUAL_SAMPLE_INDICES.tolist()).issubset(set(sample_indices.tolist())):
        raise ValueError("A visual sample is outside the cohort")
    fold_payloads = [
        _fold_payload(
            fold,
            folds,
            targets,
            predictions,
            sample_indices,
            sources,
        )
        for fold in range(FOLDS)
    ]
    region_boxes = _region_boxes()
    spatial_permutations = _spatial_permutations(sample_indices)
    valid_support_boxes = _valid_support_boxes(arrays["valid_masks"])
    lock: Dict[str, object] = {
        "protocol_id": "trkh_cap_integral_region_context_a0_20260724",
        "state": "prospective_no_candidate_metric",
        "lock_parent_commit": LOCK_PARENT_COMMIT,
        "protocol": _immutable(PROTOCOL_PATH),
        "official_cap": {
            "repository": str(CAP_REPOSITORY),
            "commit": cap_commit,
            "tree": cap_tree,
            "status": cap_status,
            "license": "MIT",
        },
        "immutable_inputs": immutable,
        "runtime": {
            "python": sys.version,
            "pip": importlib.metadata.version("pip"),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cohort": {
            "rows": ROWS,
            "feature_shape": [ROWS, CHANNELS, FEATURE_SIZE, FEATURE_SIZE],
            "feature_dtype": "float16",
            "sample_indices_sha256": _array_sha256(sample_indices),
            "targets_sha256": _array_sha256(targets),
            "folds_sha256": _array_sha256(folds),
            "sources_sha256": _strings_sha256(sources),
            "probabilities_sha256": _array_sha256(probabilities),
            "valid_masks_sha256": _array_sha256(arrays["valid_masks"]),
            "bbox_masks_sha256": _array_sha256(arrays["bbox_masks"]),
            "valid_support_boxes_sha256": _array_sha256(
                valid_support_boxes
            ),
            "counts": counts,
        },
        "geometry": {
            "upsample_size": 42,
            "grid_size": 3,
            "min_grid_size": 2,
            "roi_pool_size": 7,
            "regions": 27,
            "region_boxes": region_boxes.tolist(),
            "region_boxes_sha256": _array_sha256(region_boxes),
            "visual_sample_indices": VISUAL_SAMPLE_INDICES.tolist(),
            "visual_sample_indices_sha256": _array_sha256(
                VISUAL_SAMPLE_INDICES
            ),
            "spatial_derangement": {
                "algorithm": "sample_index_seeded_sattolo_single_cycle_v1",
                "seed": SPATIAL_DERANGEMENT_SEED,
                "permutations_sha256": _array_sha256(
                    spatial_permutations
                ),
                "fixed_points": int(
                    (
                        spatial_permutations
                        == np.arange(27, dtype=np.int64)[None, :]
                    ).sum()
                ),
            },
        },
        "architecture": {
            "valid_support_canonicalization": (
                "crop_exact_rectangular_valid_support_then_bilinear_16x16"
            ),
            "valid_support_align_corners": False,
            "padding_enters_cap": False,
            "pre_cap_channel_recalibration": "none_paper_equation",
            "pixel_projection_normalization": "none_paper_equation",
            "pixel_query_key_channels": 32,
            "pixel_value_channels": 256,
            "pixel_gamma_initial": 0.0,
            "region_query_key_dim": 32,
            "region_attention_pre_softmax_activation": (
                "none_paper_equation"
            ),
            "lstm_hidden_dim": 128,
            "netvlad_clusters": 32,
            "candidate_uses_keeper_probability": False,
            "candidate_uses_bbox": False,
            "cross_sample_context_control": {
                "queries": "own_regions",
                "keys": "partition_mapped_different_source_regions",
                "values": "own_regions",
                "mapping_uses_labels": False,
            },
        },
        "roles": [
            "keeper_margin_control",
            *TRAINABLE_ROLES[:3],
            "cap_context_seed_repeat",
            *TRAINABLE_ROLES[3:],
        ],
        "optimization": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "loss": "unweighted_bce_with_logits",
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "warmup_epochs": 3,
            "score_epoch": 30,
            "early_stopping": False,
            "primary_seed": PRIMARY_SEED,
            "repeat_seed_offset": REPEAT_OFFSET,
        },
        "calibration": {
            "tp_retention_target": 0.97,
            "reject_rule": "score_strictly_below_threshold",
            "replacement": "keeper_highest_probability_non_class1",
            "fn_rescue_enabled": False,
        },
        "performance_gates": {
            "candidate_auroc_min": 0.82,
            "candidate_auroc_gain_vs_keeper_min": 0.02,
            "candidate_auroc_gain_vs_strongest_trainable_control_min": 0.01,
            "candidate_auroc_gain_vs_each_derangement_min": 0.015,
            "fn_vs_restricted_fp_auroc_min": 0.65,
            "fn_vs_restricted_fp_gain_vs_keeper_min": 0.03,
            "control_fold_wins_min": 4,
            "tp_retention_min": 0.97,
            "minimum_fold_tp_retention": 0.94,
            "restricted_fp_rejection_min": 0.25,
            "minimum_fold_restricted_fp_rejection": 0.10,
            "minimum_positive_fp_rejection_folds": 5,
            "correction_to_harm_ratio_min": 2.0,
            "net_corrections_min": 25,
            "full_macro_f1_gain_min": 0.002,
            "full_class1_precision_gain_min": 0.03,
            "full_class1_f1_gain_min": 0.01,
            "full_class1_recall_drop_max": 0.03,
            "nonfocus_f1_drop_max": 0.003,
            "repeat_auroc_difference_max": 0.01,
            "repeat_action_agreement_min": 0.95,
        },
        "resource_gates": {
            "cuda_peak_gib_max": 6.0,
            "rss_peak_gib_max": 10.0,
        },
        "visual_gates": {
            "rows": 20,
            "valid_support_mass_min": 0.95,
            "bbox_mass_min": 0.75,
            "manual_review_required": True,
        },
        "replay": {
            "fresh_process": True,
            "numeric_error_max": 1e-7,
            "discrete_exact": True,
            "ledger_exact": True,
        },
        "folds": fold_payloads,
        "forbidden": [
            "validation",
            "test",
            "trainer_integration_before_full_pass",
            "smoke_before_full_pass",
            "probe",
            "full_train",
            "current_best_command_update",
            "post_result_sweep",
        ],
    }
    lock["scientific_payload_sha256"] = _json_sha256(lock)
    return lock


def write_lock(lock: Mapping[str, object]) -> str:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    LOCK_PATH.write_text(payload, encoding="utf-8")
    digest = _sha256(LOCK_PATH)
    LOCK_SHA_PATH.write_text(f"{digest}  {LOCK_PATH.name}\n", encoding="ascii")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the prospective CAP integral-region context A0 lock."
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = build_lock()
    if args.check:
        if not LOCK_PATH.exists() or not LOCK_SHA_PATH.exists():
            raise FileNotFoundError("Committed CAP lock files are missing")
        observed = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if observed != lock:
            raise ValueError("Committed CAP lock payload differs")
        digest = _sha256(LOCK_PATH)
        expected_line = f"{digest}  {LOCK_PATH.name}\n"
        if LOCK_SHA_PATH.read_text(encoding="ascii") != expected_line:
            raise ValueError("Committed CAP lock SHA file differs")
        print(json.dumps({"passed": True, "lock_sha256": digest}, indent=2))
        return
    digest = write_lock(lock)
    print(json.dumps({"lock": str(LOCK_PATH), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

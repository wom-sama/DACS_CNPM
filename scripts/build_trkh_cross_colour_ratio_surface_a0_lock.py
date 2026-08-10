from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PARENT_COMMIT = "12f08e71fc0325654cccaaae7b3821b135c08e07"
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_PROTOCOL_20260725.md"
)
LOCK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_LOCK_20260725.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")

GEOMETRY_CACHE = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_attention_maxsep_prototype_a0_20260721"
    / "cohort_geometry.npz"
)
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
YOLO_DATA_YAML = Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml")
GEVERS_PAPER = Path(
    r"D:\DataAI\external_sources\papers"
    r"\Gevers_Smeulders_Color_Based_Object_Recognition_PR_1999.pdf"
)
BASLAMISLI_PAPER = Path(
    r"D:\DataAI\external_sources\papers"
    r"\Baslamisli_Invariant_Descriptors_JOSAA_2021.pdf"
)
IDTRANSFORMER_PAPER = Path(
    r"D:\DataAI\external_sources\papers"
    r"\Das_IDTransformer_ICCVW_2023.pdf"
)

EXPECTED_SHA256 = {
    "geometry_cache": "69f581e1cca66bc64136352ea7f11668e1e50cf4715b2f9ba8c49e8e6f66dc72",
    "cidt_predictions": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "keeper_config": "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    "keeper_launcher_args": "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    "current_best_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "current_best_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    "yolo_data_yaml": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "gevers_paper": "f8d03a11ad0ca4970165a75749d058ef3f9f19f40a661bb75d60b3f6f5286678",
    "baslamisli_paper": "38f0d8781b413c5466a70d86357f8edbe6c3be13d81ba3deb9a806729c719513",
    "idtransformer_paper": "471717bbf6dfd7722f33d85b505275e67f46ff530af888d60cd523900a5d110f",
}

ROWS = 763
CLASSES = 5
FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64
PRIMARY_SEED = 20260725
REPEAT_OFFSET = 100000
DEPHASE_SEED = PRIMARY_SEED + 5000
VIEW_SIZE = 96
DESCRIPTOR_CHANNELS = 6
HEAD_PARAMETERS = 3004
CLASS_ORDER = (0, 1, 2, 4)
EXPECTED_COHORT_INDEX_SHA256 = (
    "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05"
)

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


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
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


def _load_geometry() -> Dict[str, np.ndarray]:
    with np.load(GEOMETRY_CACHE, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    expected = {
        "sample_indices": ((ROWS,), np.dtype(np.int64)),
        "targets": ((ROWS,), np.dtype(np.int64)),
        "folds": ((ROWS,), np.dtype(np.int64)),
        "source_stems": ((ROWS,), None),
        "keeper_probabilities": ((ROWS, CLASSES), np.dtype(np.float32)),
        "valid_masks": ((ROWS, 16, 16), np.dtype(np.bool_)),
        "bbox_masks": ((ROWS, 16, 16), np.dtype(np.bool_)),
        "model_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "crop_boxes": ((ROWS, 4), np.dtype(np.float32)),
    }
    if set(arrays) != set(expected):
        raise ValueError(f"Geometry keys differ: {sorted(arrays)}")
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape:
            raise ValueError(f"{name} shape differs: {value.shape}")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} dtype differs: {value.dtype}")
    if len(set(arrays["sample_indices"].tolist())) != ROWS:
        raise ValueError("Cohort sample indices are not unique")
    if not np.isfinite(arrays["keeper_probabilities"]).all():
        raise ValueError("Keeper probabilities are non-finite")
    if bool((arrays["bbox_masks"] & ~arrays["valid_masks"]).any()):
        raise ValueError("A bbox mask leaves valid support")
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
    return rows


def _label_path(image_path: Path) -> Path:
    resolved = image_path.resolve()
    parts = list(resolved.parts)
    lowered = [part.casefold() for part in parts]
    try:
        image_component = lowered.index("images")
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {resolved}") from error
    if image_component + 1 >= len(parts) or lowered[image_component + 1] != "train":
        raise ValueError(f"Image path is not in images/train: {resolved}")
    parts[image_component] = "labels"
    return Path(*parts[:-1], f"{resolved.stem}.txt").resolve()


def _ordered_file_manifest(paths: Sequence[Path]) -> Dict[str, object]:
    ordered = [path.resolve() for path in paths]
    unique = sorted(set(ordered), key=lambda path: str(path).casefold())
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing locked train files: {missing[:10]}")
    metadata = {
        path: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in unique
    }
    ordered_hashes = [metadata[path]["sha256"] for path in ordered]
    unique_rows = [metadata[path] for path in unique]
    return {
        "ordered_rows": len(ordered),
        "unique_files": len(unique),
        "unique_bytes": int(sum(int(row["bytes"]) for row in unique_rows)),
        "ordered_paths_sha256": _strings_sha256(str(path) for path in ordered),
        "ordered_content_sha256": _strings_sha256(ordered_hashes),
        "unique_manifest_sha256": _json_sha256(unique_rows),
    }


def _epoch_orders(indices: np.ndarray, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    orders: List[np.ndarray] = []
    hashes: List[str] = []
    for _ in range(EPOCHS):
        order = indices[rng.permutation(indices.size)].astype(np.int64, copy=False)
        orders.append(order)
        hashes.append(_array_sha256(order))
    return {
        "seed": int(seed),
        "rows_per_epoch": int(indices.size),
        "optimizer_steps": int(
            EPOCHS * ((int(indices.size) + BATCH_SIZE - 1) // BATCH_SIZE)
        ),
        "per_epoch_sha256": hashes,
        "all_epochs_sha256": _array_sha256(np.concatenate(orders)),
    }


def _dephase_offsets(sample_indices: np.ndarray) -> np.ndarray:
    offsets = np.empty((ROWS, DESCRIPTOR_CHANNELS, 2), dtype=np.int64)
    for position, sample_index in enumerate(sample_indices.tolist()):
        for channel in range(DESCRIPTOR_CHANNELS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [DEPHASE_SEED, int(sample_index), int(channel)]
                )
            )
            offsets[position, channel, 0] = int(rng.integers(1, VIEW_SIZE))
            offsets[position, channel, 1] = int(rng.integers(1, VIEW_SIZE))
    if bool((offsets <= 0).any()) or bool((offsets >= VIEW_SIZE).any()):
        raise ValueError("A dephasing offset is zero or outside the view")
    return offsets


def _partition_summary(
    indices: np.ndarray,
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    sample_indices: np.ndarray,
    sources: Sequence[str],
) -> Dict[str, object]:
    return {
        "rows": int(indices.size),
        "indices_sha256": _array_sha256(indices),
        "sample_indices_sha256": _array_sha256(sample_indices[indices]),
        "targets_sha256": _array_sha256(targets[indices]),
        "sources_sha256": _strings_sha256(sources[int(index)] for index in indices),
        "target_counts": {
            str(target): int((targets[indices] == target).sum())
            for target in CLASS_ORDER
        },
        "tp1": int(((targets[indices] == 1) & (predictions[indices] == 1)).sum()),
        "fn1": int(((targets[indices] == 1) & (predictions[indices] != 1)).sum()),
        "restricted_fp": int(
            (
                np.isin(targets[indices], [0, 2, 4])
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
    sources: Sequence[str],
) -> Dict[str, object]:
    calibration_fold = (outer_fold + 1) % FOLDS
    held = np.flatnonzero(folds == outer_fold).astype(np.int64)
    calibration = np.flatnonzero(folds == calibration_fold).astype(np.int64)
    fit = np.flatnonzero(
        (folds != outer_fold) & (folds != calibration_fold)
    ).astype(np.int64)
    partitions = {"fit": fit, "calibration": calibration, "held": held}
    source_sets = {
        name: {sources[int(index)] for index in indices}
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
    primary_seed = PRIMARY_SEED + outer_fold
    repeat_seed = PRIMARY_SEED + REPEAT_OFFSET + outer_fold
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
        "primary_orders": _epoch_orders(fit, primary_seed),
        "repeat_orders": _epoch_orders(fit, repeat_seed),
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


def build_lock() -> Dict[str, object]:
    immutable_paths = {
        "geometry_cache": GEOMETRY_CACHE,
        "cidt_predictions": CIDT_PREDICTIONS,
        "keeper_checkpoint": KEEPER_CHECKPOINT,
        "keeper_config": KEEPER_CONFIG,
        "keeper_launcher_args": KEEPER_LAUNCHER_ARGS,
        "current_best_commands": CURRENT_BEST_COMMANDS,
        "current_best_history": CURRENT_BEST_HISTORY,
        "yolo_data_yaml": YOLO_DATA_YAML,
        "gevers_paper": GEVERS_PAPER,
        "baslamisli_paper": BASLAMISLI_PAPER,
        "idtransformer_paper": IDTRANSFORMER_PAPER,
    }
    required = [PROTOCOL_PATH, *immutable_paths.values(), *PROTECTED_UNTRACKED]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing lock inputs: {missing}")
    immutable = {name: _immutable(path) for name, path in immutable_paths.items()}
    for name, expected in EXPECTED_SHA256.items():
        if immutable[name]["sha256"] != expected:
            raise ValueError(f"{name} SHA-256 differs")

    geometry = _load_geometry()
    cidt = _load_cidt_clean()
    sample_indices = geometry["sample_indices"].astype(np.int64, copy=False)
    targets = geometry["targets"].astype(np.int64, copy=False)
    folds = geometry["folds"].astype(np.int64, copy=False)
    probabilities = geometry["keeper_probabilities"].astype(np.float32, copy=False)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    sources = geometry["source_stems"].astype(str).tolist()
    image_paths: List[Path] = []
    for position, sample_index in enumerate(sample_indices.tolist()):
        row = cidt[int(sample_index)]
        if int(row["target_index"]) != int(targets[position]):
            raise ValueError(f"CIDT target mismatch at {sample_index}")
        if int(row["fold"]) != int(folds[position]):
            raise ValueError(f"CIDT fold mismatch at {sample_index}")
        if row["source_stem"] != sources[position]:
            raise ValueError(f"CIDT source mismatch at {sample_index}")
        if int(row["keeper_prediction"]) != int(predictions[position]):
            raise ValueError(f"CIDT prediction mismatch at {sample_index}")
        path = Path(row["image_path"]).resolve()
        parts = {part.casefold() for part in path.parts}
        if "train" not in parts or parts.intersection(
            {"val", "valid", "validation", "test"}
        ):
            raise ValueError(f"Non-train path in cohort: {path}")
        image_paths.append(path)
    label_paths = [_label_path(path) for path in image_paths]

    index_hash = _array_sha256(sample_indices)
    if index_hash != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError(f"Cohort sample-index hash differs: {index_hash}")
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
        raise ValueError(f"Cohort counts differ: {counts}")

    offsets = _dephase_offsets(sample_indices)
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
    protected = {
        str(path.relative_to(REPOSITORY_ROOT)): _immutable(path)
        for path in PROTECTED_UNTRACKED
    }
    protocol = _immutable(PROTOCOL_PATH)
    head_parameter_oracle = (
        6 * 24 * 3 * 3
        + 2 * 24
        + 24 * 3 * 3
        + 24 * 48
        + 2 * 48
        + 48 * 4
        + 4
    )
    if head_parameter_oracle != HEAD_PARAMETERS:
        raise RuntimeError(f"Head parameter oracle differs: {head_parameter_oracle}")

    return {
        "protocol_id": "trkh_cross_colour_ratio_surface_a0_20260725",
        "state": "prospective_no_candidate_observation",
        "created_utc": "2026-07-25T00:00:00Z",
        "lock_parent_commit": LOCK_PARENT_COMMIT,
        "repository": {
            "root": str(REPOSITORY_ROOT),
            "branch": _git("branch", "--show-current"),
            "head_at_lock_build": _git("rev-parse", "HEAD"),
            "upstream_at_lock_build": _git(
                "rev-parse", "origin/classification-only-research"
            ),
        },
        "protocol": protocol,
        "immutable_inputs": immutable,
        "protected_untracked": protected,
        "cohort": {
            "rows": ROWS,
            "sample_indices_sha256": index_hash,
            "targets_sha256": _array_sha256(targets),
            "folds_sha256": _array_sha256(folds),
            "sources_sha256": _strings_sha256(sources),
            "probabilities_sha256": _array_sha256(probabilities),
            "valid_masks_sha256": _array_sha256(
                geometry["valid_masks"].astype(np.bool_)
            ),
            "bbox_masks_sha256": _array_sha256(
                geometry["bbox_masks"].astype(np.bool_)
            ),
            "model_boxes_sha256": _array_sha256(
                geometry["model_boxes"].astype(np.float32)
            ),
            "crop_boxes_sha256": _array_sha256(
                geometry["crop_boxes"].astype(np.float32)
            ),
            "counts": counts,
            "image_files": _ordered_file_manifest(image_paths),
            "label_files": _ordered_file_manifest(label_paths),
        },
        "folds": fold_payloads,
        "descriptor": {
            "model_image_size": 256,
            "view_size": VIEW_SIZE,
            "input_channels": DESCRIPTOR_CHANNELS,
            "srgb_inverse_transfer": "iec_61966_2_1",
            "linear_rgb_log_epsilon": 1.0 / 255.0,
            "saturation_low_exclusive": 1.0 / 255.0,
            "saturation_high_exclusive": 254.0 / 255.0,
            "gaussian_sigma": 1.0,
            "gaussian_truncate_radius": 3,
            "signed_response_clip": [-8.0, 8.0],
            "colour_ratio_planes": ["log_r", "log_g", "log_b"],
            "cross_colour_ratio_planes": [
                "log_r_minus_log_g",
                "log_r_minus_log_b",
                "log_g_minus_log_b",
            ],
            "derivative_axes": ["x", "y"],
            "dephase_seed": DEPHASE_SEED,
            "dephase_offsets_shape": list(offsets.shape),
            "dephase_offsets_sha256": _array_sha256(offsets),
            "dephase_minimum_offset": int(offsets.min()),
            "dephase_maximum_offset": int(offsets.max()),
        },
        "head": {
            "class_order": list(CLASS_ORDER),
            "trainable_parameters": HEAD_PARAMETERS,
            "parameter_oracle": head_parameter_oracle,
            "architecture": [
                "conv_6_24_k3_bias_false",
                "groupnorm_6_24_gelu_avgpool2",
                "depthwise_24_k3_bias_false",
                "pointwise_24_48_bias_false",
                "groupnorm_8_48_gelu_avgpool2",
                "class_evidence_conv_48_4_k1_bias_true",
                "reliability_masked_spatial_mean",
            ],
            "pretrained": False,
            "bbox_is_input": False,
            "keeper_probability_is_input": False,
        },
        "optimization": {
            "roles": [
                "colour_ratio_control",
                "cross_colour_ratio_candidate",
                "cross_colour_ratio_seed_repeat",
                "cross_colour_ratio_spatial_dephased_control",
            ],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "requested_workers": 4,
            "required_effective_workers": 4,
            "primary_seed": PRIMARY_SEED,
            "repeat_offset": REPEAT_OFFSET,
            "loss": "unweighted_four_class_cross_entropy",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "gradient_clip_norm": 1.0,
            "warmup_epochs": 2,
            "schedule_after_warmup": "cosine_to_zero",
            "class_weight": None,
            "oversampling": False,
            "augmentation": False,
            "early_stopping": False,
        },
        "calibration": {
            "minimum_class1_retention": 0.97,
            "score": "logit1_minus_logsumexp_logits_0_2_4",
            "action": "suppress_keeper_pred1_below_threshold",
            "replacement": "keeper_highest_probability_nonclass1",
            "production_threshold_authorized": False,
        },
        "visual": {
            "sample_indices": VISUAL_SAMPLE_INDICES.tolist(),
            "sample_indices_sha256": _array_sha256(VISUAL_SAMPLE_INDICES),
            "cohort_positions_sha256": _array_sha256(visual_positions),
            "rows": int(VISUAL_SAMPLE_INDICES.size),
            "all_target4_rows": int(
                (targets[visual_positions[-10:]] == 4).sum()
            ),
        },
        "resource_limits": {
            "clean_wall_seconds": 1200,
            "temporary_cache_bytes": 429496730,
            "retained_bytes": 107374183,
            "head_peak_cuda_bytes": 2147483648,
            "end_to_end_peak_cuda_bytes": 6442450944,
            "process_rss_bytes": 10737418240,
            "batch1_mean_ratio": 1.15,
            "batch1_p95_ratio": 1.15,
            "batch_throughput_ratio": 0.85,
            "end_to_end_peak_cuda_ratio": 1.15,
            "onnx_max_probability_error": 1e-5,
            "tensorrt_fp32_max_probability_error": 1e-4,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "numpy": np.__version__,
            "pillow": _package_version("Pillow"),
            "scikit_learn": _package_version("scikit-learn"),
            "onnx": _package_version("onnx"),
            "onnxruntime": _first_package_version(
                "onnxruntime", "onnxruntime-gpu"
            ),
            "tensorrt": _package_version("tensorrt"),
        },
        "forbidden": {
            "validation": True,
            "test": True,
            "raw_dataset_write": True,
            "external_or_pretrained_weights": True,
            "trainer_edit": True,
            "current_best_command_edit": True,
            "descriptor_or_gate_sweep": True,
        },
        "observations": {
            "candidate_scores": None,
            "candidate_thresholds": None,
            "candidate_actions": None,
            "candidate_metrics": None,
            "validation_metrics": None,
            "test_metrics": None,
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


def _assert_lock_parent_for_write() -> None:
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "origin/classification-only-research")
    if head != LOCK_PARENT_COMMIT or upstream != LOCK_PARENT_COMMIT:
        raise RuntimeError(
            "Prospective lock writes require HEAD and upstream at "
            f"{LOCK_PARENT_COMMIT}; got HEAD={head}, upstream={upstream}. "
            "Use --check-only after the prospective boundary is committed."
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
    current_upstream = _git(
        "rev-parse", "origin/classification-only-research"
    )
    if not _is_ancestor(LOCK_PARENT_COMMIT, current_head):
        raise RuntimeError("Lock parent is not an ancestor of current HEAD")
    if not _is_ancestor(LOCK_PARENT_COMMIT, current_upstream):
        raise RuntimeError("Lock parent is not an ancestor of current upstream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the prospective TRKH CCR surface A0 machine lock."
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
                    "rows": lock["cohort"]["rows"],
                    "protocol_sha256": lock["protocol"]["sha256"],
                    "image_manifest_sha256": lock["cohort"]["image_files"][
                        "unique_manifest_sha256"
                    ],
                    "dephase_offsets_sha256": lock["descriptor"][
                        "dephase_offsets_sha256"
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

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F


EXPECTED_ROWS = 763
GRID_SIZE = 16
EXPECTED_COHORT_SHA256 = (
    "a835d3498319d3d25d83c0279ab24d6fcc703a2266ff5f091dc2273bbb8baa9d"
)
EXPECTED_VALID_MASKS_SHA256 = (
    "1c742f6d3067014d78ae299fab07f3baeb4c0ec0014dc112975c4d7199e4771e"
)
EXPECTED_SAMPLE_INDICES_SHA256 = (
    "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05"
)
EXPECTED_BBOX_UNUSABLE_ROWS = 12
EXPECTED_BBOX_UNUSABLE_RECORD_SHA256 = (
    "cbe5d09bc0ef6305a6610297ddc1dd747e1e018fd10cdc87d2ed87d224b91534"
)
EXPECTED_ALL_Q_MEAN = 0.35292472290342536
EXPECTED_ALL_Q_MEDIAN = 0.34615384615384615
EXPECTED_USABLE_Q_MEAN = 0.3585639994345054
EXPECTED_USABLE_Q_MEDIAN = 0.34615384615384615
LEGACY_WRONG_BIG_ALL_Q_MEAN = 0.3498597859836392
LEGACY_WRONG_BIG_ALL_Q_MEDIAN = 0.34375
LEGACY_WRONG_BIG_USABLE_Q_MEAN = 0.35545008882225926
LEGACY_WRONG_BIG_USABLE_Q_MEDIAN = 0.34615384615384615
REQUIRED_COHORT_ARRAYS = ("sample_indices", "targets", "model_boxes")
FORBIDDEN_INPUT_COMPONENTS = frozenset({"val", "valid", "validation", "test"})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-table geometry-only preflight for the Pair-Surface DDF v2 "
            "valid-mask and bbox definitions. Frozen cohort targets are read only "
            "to stratify unusable geometry; no RGB, raw label file/path, score, "
            "validation, or test input is read."
        )
    )
    parser.add_argument("--cohort-arrays", type=Path, required=True)
    parser.add_argument("--valid-masks-packbits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_train_table_input(path: Path, *, suffix: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Geometry input does not exist: {resolved}")
    if resolved.suffix.casefold() != suffix.casefold():
        raise ValueError(
            f"Unexpected geometry input suffix for {resolved}; expected {suffix}"
        )
    components = {
        component.casefold()
        for component in str(resolved).replace("\\", "/").split("/")
        if component
    }
    forbidden = sorted(components.intersection(FORBIDDEN_INPUT_COMPONENTS))
    if forbidden:
        raise ValueError(
            "Validation/test path components are forbidden for geometry input: "
            f"{forbidden}"
        )
    return resolved


def _load_cohort_geometry(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_COHORT_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"Cohort archive is missing geometry arrays: {missing}")
        # Deliberately do not read keeper_probabilities, folds, paths, or stems.
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in REQUIRED_COHORT_ARRAYS
        }

    expected = {
        "sample_indices": ((EXPECTED_ROWS,), np.dtype(np.int64)),
        "targets": ((EXPECTED_ROWS,), np.dtype(np.int64)),
        "model_boxes": ((EXPECTED_ROWS, 4), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"Unexpected cohort {name}: shape={value.shape}, dtype={value.dtype}; "
                f"expected shape={shape}, dtype={dtype}"
            )
    if np.unique(arrays["sample_indices"]).size != EXPECTED_ROWS:
        raise ValueError("Cohort sample indices are not unique")
    if not set(arrays["targets"].tolist()).issubset({0, 1, 2, 4}):
        raise ValueError("Cohort targets must be in {0,1,2,4}")
    boxes = arrays["model_boxes"]
    if not np.isfinite(boxes).all() or bool((boxes < 0).any()):
        raise ValueError("Model boxes must be finite and non-negative")
    return arrays


def _load_packed_valid_masks(path: Path) -> np.ndarray:
    packed = np.load(path, mmap_mode="r", allow_pickle=False)
    if packed.shape != (EXPECTED_ROWS, 8192) or packed.dtype != np.uint8:
        raise ValueError(
            "Unexpected packed valid-mask cache: "
            f"shape={packed.shape}, dtype={packed.dtype}"
        )
    return np.asarray(packed).copy()


def unpack_valid_masks(packed: np.ndarray, *, bitorder: str) -> np.ndarray:
    values = np.asarray(packed)
    if values.shape != (EXPECTED_ROWS, 8192) or values.dtype != np.uint8:
        raise ValueError("packed must have shape [763,8192] and dtype uint8")
    if bitorder not in {"little", "big"}:
        raise ValueError("bitorder must be 'little' or 'big'")
    unpacked = np.unpackbits(
        values,
        axis=1,
        count=256 * 256,
        bitorder=bitorder,
    )
    return unpacked.reshape(EXPECTED_ROWS, 256, 256).astype(np.bool_)


def _max_pool3_stride2_pad1_numpy(masks: np.ndarray) -> np.ndarray:
    values = np.asarray(masks, dtype=np.bool_)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Validity masks must have shape [N,H,H]")
    height = int(values.shape[1])
    output_size = (height + 1) // 2
    padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    pooled = np.zeros((values.shape[0], output_size, output_size), dtype=np.bool_)
    for offset_y in range(3):
        for offset_x in range(3):
            pooled |= padded[
                :,
                offset_y : offset_y + 2 * output_size : 2,
                offset_x : offset_x + 2 * output_size : 2,
            ]
    return pooled


def propagate_valid_masks_numpy(valid256: np.ndarray) -> Tuple[np.ndarray, ...]:
    values = np.asarray(valid256, dtype=np.bool_)
    if values.ndim != 3 or values.shape[1:] != (256, 256):
        raise ValueError("valid256 must have shape [N,256,256]")
    valid64 = values.reshape(values.shape[0], 64, 4, 64, 4).any(axis=(2, 4))
    valid32 = _max_pool3_stride2_pad1_numpy(valid64)
    valid16 = _max_pool3_stride2_pad1_numpy(valid32)
    return valid64, valid32, valid16


def propagate_valid_masks_torch(valid256: np.ndarray) -> Tuple[np.ndarray, ...]:
    values = torch.from_numpy(np.asarray(valid256, dtype=np.bool_)).to(
        dtype=torch.float32
    )[:, None]
    valid64 = F.avg_pool2d(values, kernel_size=4, stride=4) > 0
    valid32 = F.max_pool2d(
        valid64.to(dtype=torch.float32), kernel_size=3, stride=2, padding=1
    ) > 0
    valid16 = F.max_pool2d(
        valid32.to(dtype=torch.float32), kernel_size=3, stride=2, padding=1
    ) > 0
    return tuple(
        value[:, 0].cpu().numpy().astype(np.bool_, copy=False)
        for value in (valid64, valid32, valid16)
    )


def rasterize_model_boxes(model_boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(model_boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("model_boxes must have shape [N,4]")
    masks = np.zeros((boxes.shape[0], GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    for index, (cx, cy, width, height) in enumerate(boxes.astype(np.float64)):
        left = float(np.clip(cx - width / 2.0, 0.0, 1.0))
        right = float(np.clip(cx + width / 2.0, 0.0, 1.0))
        top = float(np.clip(cy - height / 2.0, 0.0, 1.0))
        bottom = float(np.clip(cy + height / 2.0, 0.0, 1.0))
        x0 = min(int(math.floor(left * GRID_SIZE)), GRID_SIZE - 1)
        y0 = min(int(math.floor(top * GRID_SIZE)), GRID_SIZE - 1)
        x1 = max(x0 + 1, min(int(math.ceil(right * GRID_SIZE)), GRID_SIZE))
        y1 = max(y0 + 1, min(int(math.ceil(bottom * GRID_SIZE)), GRID_SIZE))
        masks[index, y0:y1, x0:x1] = True
    if bool((masks.sum(axis=(1, 2)) == 0).any()):
        raise ValueError("Rasterized model bbox is empty")
    return masks


def _float_stats(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Statistics require a finite non-empty vector")
    return {
        "mean": float(array.mean(dtype=np.float64)),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "p05_linear": float(np.quantile(array, 0.05, method="linear")),
        "p95_linear": float(np.quantile(array, 0.95, method="linear")),
    }


def _counter_payload(values: np.ndarray) -> Dict[str, int]:
    counter = Counter(int(value) for value in np.asarray(values).tolist())
    return {str(key): int(counter[key]) for key in sorted(counter)}


def build_report(
    cohort_arrays: Path,
    valid_masks_packbits: Path,
    *,
    enforce_locked_inputs: bool = True,
) -> Dict[str, object]:
    cohort_path = _assert_train_table_input(cohort_arrays, suffix=".npz")
    valid_path = _assert_train_table_input(valid_masks_packbits, suffix=".npy")
    cohort_sha256 = _sha256_file(cohort_path)
    valid_sha256 = _sha256_file(valid_path)
    if enforce_locked_inputs and cohort_sha256 != EXPECTED_COHORT_SHA256:
        raise ValueError("Cohort archive SHA-256 differs from the locked train table")
    if enforce_locked_inputs and valid_sha256 != EXPECTED_VALID_MASKS_SHA256:
        raise ValueError("Packed valid-mask SHA-256 differs from the locked train table")

    cohort = _load_cohort_geometry(cohort_path)
    packed_valid_masks = _load_packed_valid_masks(valid_path)
    valid256 = unpack_valid_masks(packed_valid_masks, bitorder="little")
    numpy_masks = propagate_valid_masks_numpy(valid256)
    torch_masks = propagate_valid_masks_torch(valid256)
    stage_names = ("valid64", "valid32", "valid16")
    parity = {
        name: bool(np.array_equal(numpy_value, torch_value))
        for name, numpy_value, torch_value in zip(
            stage_names, numpy_masks, torch_masks
        )
    }
    if not all(parity.values()):
        raise ValueError(f"NumPy/Torch validity propagation differs: {parity}")

    valid64, valid32, valid16 = numpy_masks
    bbox16 = rasterize_model_boxes(cohort["model_boxes"])
    intersection = bbox16 & valid16
    valid_cells = valid16.sum(axis=(1, 2), dtype=np.int64)
    bbox_cells = bbox16.sum(axis=(1, 2), dtype=np.int64)
    intersection_cells = intersection.sum(axis=(1, 2), dtype=np.int64)
    if bool((valid_cells <= 0).any()):
        raise ValueError("A v2 row has no valid16 support")
    q = intersection_cells.astype(np.float64) / valid_cells.astype(np.float64)
    usable = intersection_cells > 0
    unusable_positions = np.flatnonzero(~usable)
    unusable_records = [
        {
            "sample_index": int(cohort["sample_indices"][position]),
            "target": int(cohort["targets"][position]),
        }
        for position in unusable_positions.tolist()
    ]
    unusable_records.sort(key=lambda row: (row["sample_index"], row["target"]))
    unusable_record_sha256 = _json_sha256(unusable_records)

    all_q = _float_stats(q)
    usable_q = _float_stats(q[usable])

    # Regression counterexample: NumPy's default unpack order is big-endian,
    # while the retained cache was explicitly packed little-endian. Bit reversal
    # preserves row bit counts and even the unusable-row list here, which made
    # the former hand calculation look plausible. Retain both signatures.
    legacy_big_valid16 = propagate_valid_masks_numpy(
        unpack_valid_masks(packed_valid_masks, bitorder="big")
    )[2]
    legacy_big_intersection_cells = (bbox16 & legacy_big_valid16).sum(
        axis=(1, 2), dtype=np.int64
    )
    legacy_big_valid_cells = legacy_big_valid16.sum(
        axis=(1, 2), dtype=np.int64
    )
    legacy_big_usable = legacy_big_intersection_cells > 0
    legacy_big_q = (
        legacy_big_intersection_cells.astype(np.float64)
        / legacy_big_valid_cells.astype(np.float64)
    )
    legacy_big_all_q = _float_stats(legacy_big_q)
    legacy_big_usable_q = _float_stats(legacy_big_q[legacy_big_usable])
    legacy_big_unusable_records = [
        {
            "sample_index": int(cohort["sample_indices"][position]),
            "target": int(cohort["targets"][position]),
        }
        for position in np.flatnonzero(~legacy_big_usable).tolist()
    ]
    legacy_big_unusable_records.sort(
        key=lambda row: (row["sample_index"], row["target"])
    )
    legacy_big_record_sha256 = _json_sha256(legacy_big_unusable_records)
    sample_indices_sha256 = _array_sha256(cohort["sample_indices"])
    locked_checks = {
        "cohort_file_sha256": cohort_sha256 == EXPECTED_COHORT_SHA256,
        "valid_masks_file_sha256": valid_sha256 == EXPECTED_VALID_MASKS_SHA256,
        "sample_indices_sha256": sample_indices_sha256
        == EXPECTED_SAMPLE_INDICES_SHA256,
        "numpy_torch_mask_parity": all(parity.values()),
        "bbox_unusable_row_count": len(unusable_records)
        == EXPECTED_BBOX_UNUSABLE_ROWS,
        "bbox_unusable_record_sha256": unusable_record_sha256
        == EXPECTED_BBOX_UNUSABLE_RECORD_SHA256,
        "all_q_mean": math.isclose(
            all_q["mean"], EXPECTED_ALL_Q_MEAN, rel_tol=0.0, abs_tol=1e-15
        ),
        "all_q_median": math.isclose(
            all_q["median"], EXPECTED_ALL_Q_MEDIAN, rel_tol=0.0, abs_tol=1e-15
        ),
        "usable_q_mean": math.isclose(
            usable_q["mean"], EXPECTED_USABLE_Q_MEAN, rel_tol=0.0, abs_tol=1e-15
        ),
        "usable_q_median": math.isclose(
            usable_q["median"], EXPECTED_USABLE_Q_MEDIAN, rel_tol=0.0, abs_tol=1e-15
        ),
        "legacy_wrong_big_signature": all(
            (
                math.isclose(
                    legacy_big_all_q["mean"],
                    LEGACY_WRONG_BIG_ALL_Q_MEAN,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                math.isclose(
                    legacy_big_all_q["median"],
                    LEGACY_WRONG_BIG_ALL_Q_MEDIAN,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                math.isclose(
                    legacy_big_usable_q["mean"],
                    LEGACY_WRONG_BIG_USABLE_Q_MEAN,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                math.isclose(
                    legacy_big_usable_q["median"],
                    LEGACY_WRONG_BIG_USABLE_Q_MEDIAN,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                legacy_big_record_sha256
                == EXPECTED_BBOX_UNUSABLE_RECORD_SHA256,
            )
        ),
    }
    if enforce_locked_inputs and not all(locked_checks.values()):
        failed = sorted(name for name, passed in locked_checks.items() if not passed)
        raise ValueError(f"Locked v2 geometry checks failed: {failed}")

    return {
        "schema_version": 1,
        "state": "pair_surface_ddf_v2_train_table_geometry_preflight",
        "automatic_passed": bool(all(locked_checks.values())),
        "locked_checks": locked_checks,
        "access_contract": {
            "train_table_only": True,
            "cohort_members_read": list(REQUIRED_COHORT_ARRAYS),
            "keeper_probability_arrays_read": 0,
            "old_or_v2_fold_arrays_read": 0,
            "rgb_images_read": 0,
            "image_or_label_paths_opened": 0,
            "candidate_scores_read": 0,
            "validation_data_used": False,
            "test_data_used": False,
            "fold_assignment_performed": False,
            "training_launched": False,
        },
        "inputs": {
            "cohort_arrays": {
                "path": str(cohort_path),
                "bytes": cohort_path.stat().st_size,
                "sha256": cohort_sha256,
            },
            "valid_masks_packbits": {
                "path": str(valid_path),
                "bytes": valid_path.stat().st_size,
                "sha256": valid_sha256,
                "bitorder": "little",
                "unpacked_shape": [EXPECTED_ROWS, 256, 256],
            },
            "array_sha256": {
                name: _array_sha256(value) for name, value in cohort.items()
            },
        },
        "definitions": {
            "valid64": "any valid256 cell in each non-overlapping 4x4 block",
            "valid32": "max_pool2d(valid64,kernel=3,stride=2,padding=1)>0",
            "valid16": "max_pool2d(valid32,kernel=3,stride=2,padding=1)>0",
            "bbox16": (
                "model_boxes cxcywh clipped to [0,1], left/top floor(16*x), "
                "right/bottom ceil(16*x), at least one cell"
            ),
            "q": "sum(bbox16 AND valid16) / sum(valid16)",
            "bbox_usable": "sum(bbox16 AND valid16) > 0",
        },
        "cohort": {
            "rows": EXPECTED_ROWS,
            "bbox_usable_rows": int(usable.sum()),
            "bbox_unusable_rows": len(unusable_records),
            "bbox_unusable_target_counts": _counter_payload(
                cohort["targets"][~usable]
            ),
            "bbox_unusable_records": unusable_records,
            "bbox_unusable_records_sha256": unusable_record_sha256,
        },
        "mask_propagation": {
            "independent_implementations": ["numpy_boolean", "torch_pooling"],
            "stage_exact_parity": parity,
            "array_sha256": {
                name: _array_sha256(mask)
                for name, mask in zip(stage_names, numpy_masks)
            },
            "valid_cell_statistics": {
                name: _float_stats(mask.sum(axis=(1, 2), dtype=np.int64))
                for name, mask in zip(stage_names, numpy_masks)
            },
        },
        "bitorder_regression": {
            "cache_pack_bitorder": "little",
            "required_unpack_bitorder": "little",
            "numpy_default_unpack_bitorder": "big",
            "root_cause": (
                "The former q=0.3498597859836392 value used big-endian "
                "unpacking on a little-endian packed cache. Per-byte bit reversal "
                "preserved bit counts and the 12-row unusable signature, so those "
                "checks alone could not detect the error."
            ),
            "wrong_big_q_all_763_rows": legacy_big_all_q,
            "wrong_big_q_bbox_usable_751_rows": legacy_big_usable_q,
            "wrong_big_bbox_unusable_rows": int((~legacy_big_usable).sum()),
            "wrong_big_bbox_unusable_records_sha256": legacy_big_record_sha256,
            "wrong_big_and_correct_unusable_records_match": bool(
                legacy_big_unusable_records == unusable_records
            ),
            "correct_and_wrong_valid16_sha256_differ": bool(
                _array_sha256(valid16) != _array_sha256(legacy_big_valid16)
            ),
        },
        "bbox_geometry": {
            "bbox16_sha256": _array_sha256(bbox16),
            "bbox_cell_statistics": _float_stats(bbox_cells),
            "intersection_cell_statistics_all_rows": _float_stats(
                intersection_cells
            ),
            "q_all_763_rows": all_q,
            "q_bbox_usable_751_rows": usable_q,
            "uniform_attention_geometry_normalized_lift": 0.0,
        },
    }


def _write_outputs(report: Mapping[str, object], output_dir: Path) -> None:
    resolved = output_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    summary_path = resolved / "summary.json"
    csv_path = resolved / "bbox_unusable_rows.csv"
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.tmp-{os.getpid()}"
    )
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    try:
        with temporary_summary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        with temporary_csv.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_index", "target"))
            writer.writeheader()
            writer.writerows(report["cohort"]["bbox_unusable_records"])
        os.replace(temporary_summary, summary_path)
        os.replace(temporary_csv, csv_path)
    finally:
        for temporary in (temporary_summary, temporary_csv):
            if temporary.exists():
                temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = build_report(args.cohort_arrays, args.valid_masks_packbits)
    _write_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "automatic_passed": report["automatic_passed"],
                "bbox_usable_rows": report["cohort"]["bbox_usable_rows"],
                "bbox_unusable_rows": report["cohort"]["bbox_unusable_rows"],
                "usable_q_mean": report["bbox_geometry"][
                    "q_bbox_usable_751_rows"
                ]["mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

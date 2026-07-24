from __future__ import annotations

"""Fail-closed same-tensor materializer for the locked CCR Surface A0."""

import argparse
import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import psutil
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from trkh.data.dataset import bbox_xywh_to_xyxy, build_eval_transform
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics
from trkh.tools.cross_colour_ratio_surface_a0_engine import (
    INPUT_MEAN,
    INPUT_STD,
    MODEL_IMAGE_SIZE,
    array_sha256,
    margin_crop_box,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).resolve()
LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_LOCK_20260725.json"
)
ERRATUM_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_PREIMPLEMENTATION_ERRATUM_20260725.md"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_materialized_20260725"
)

EXPECTED_LOCK_SHA256 = (
    "6c2a604deb4f976a3556ffb0ad8ddb60c7f376e81e374ae16272c6e6d49fc835"
)
EXPECTED_ERRATUM_SHA256 = (
    "f9a901b14098b042272a2c3c777f7686e3e6ac61e6cfff8ed1c2d93b0777d03b"
)
TRANSFORM_SOURCE_SHA256 = {
    REPO_ROOT / "trkh" / "data" / "dataset.py": (
        "7ea29b5f3146995b63179600a5d31f338d154164221f1179b3352846ba4275d5"
    ),
    REPO_ROOT / "trkh" / "tools" / "build_precision_ensemble_checkpoint.py": (
        "481b7c1cdc0add86541e4be08175c3a42031d942cf2c86ed6d9e89320f43c7fe"
    ),
    REPO_ROOT / "trkh" / "evaluation" / "input_normalization.py": (
        "513f452063260c2f7724d40c621b4b34506e2463bfb78670f8cfb7d58c995e2f"
    ),
}

PROTOCOL_ID = "trkh_cross_colour_ratio_surface_a0_20260725"
ROWS = 763
UNIQUE_FILES = 735
FEATURE_SIZE = 16
PACKED_MASK_BYTES = MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE // 8
SRGB_CACHE_NAME = "model_srgb_uint8.npy"
VALID_MASK_CACHE_NAME = "image_valid_masks_packbits.npy"
COHORT_ARRAYS_NAME = "cohort_arrays.npz"
FORBIDDEN_SPLIT_COMPONENTS = frozenset(
    {"val", "valid", "validation", "test"}
)
TRACKED_DOMAIN_ROOTS = {
    "dataset": Path(r"D:\DataAI\AIEx\newdataset"),
    "runs": REPO_ROOT / "runs",
    "external": Path(r"D:\DataAI\external_sources"),
}
REQUIRED_EXECUTION_CONSTRAINTS = {
    "cpu_only": True,
    "formal_materializations": 1,
    "fresh_process_replays": 1,
    "descriptor_creation": False,
    "head_fit": False,
    "validation": False,
    "test": False,
    "production_integration": False,
}

EXPECTED_TRANSFORM_SEMANTICS = {
    "image_size": 256,
    "temporal_frames": 1,
    "input_mean": tuple(INPUT_MEAN),
    "input_std": tuple(INPUT_STD),
    "crop_to_primary_object": True,
    "classification_object_crops": True,
    "crop_margin_ratio": 0.05,
    "resize_mode": "pad",
    "illumination_normalization": True,
    "illumination_normalization_strength": 0.35,
    "foreground_crop_mode": "none",
    "foreground_crop_margin_ratio": 0.08,
    "foreground_crop_min_mask_area_ratio": 0.03,
    "foreground_crop_max_mask_area_ratio": 0.92,
    "foreground_crop_max_crop_area_ratio": 0.98,
    "background_suppression_mode": "desaturate_blur",
    "background_suppression_margin": 0.08,
    "background_suppression_blur_radius": 7.0,
    "surface_detail_amplification_mode": "none",
    "surface_detail_amplification_strength": 0.0,
    "surface_detail_amplification_blur_radius": 1.25,
    "surface_detail_amplification_foreground_weight": 0.85,
    "eval_surface_detail_amplification": False,
    "classification_source_context_aux": False,
}


@dataclass(frozen=True)
class LabelObject:
    label: int
    bbox: Tuple[float, float, float, float]
    object_index: int


@dataclass(frozen=True)
class CohortRecord:
    position: int
    sample_index: int
    target: int
    fold: int
    source_stem: str
    image_path: Path
    label_path: Path
    model_bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class DirectTensorSample:
    model_input: Tensor
    image_valid_mask: Tensor
    model_bbox: Tensor
    crop_bbox: Tensor
    crop_box: Tuple[int, int, int, int]
    primary_object_index: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strings_sha256(values: Iterable[object]) -> str:
    encoded = [str(value).encode("utf-8") for value in values]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
    for value in encoded:
        digest.update(np.asarray([len(value)], dtype=np.int64).tobytes())
        digest.update(value)
    return digest.hexdigest()


def _resolve_locked_path(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def _load_json_with_sidecar(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> Tuple[Dict[str, object], str]:
    resolved = Path(path).resolve()
    observed_sha = sha256_file(resolved)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise ValueError(f"Locked JSON SHA-256 differs: {resolved}")
    sidecar = resolved.with_suffix(".sha256")
    expected_sidecar_sha, expected_name = sidecar.read_text(
        encoding="utf-8"
    ).strip().split(maxsplit=1)
    if expected_name != resolved.name or expected_sidecar_sha != observed_sha:
        raise ValueError(f"Locked JSON sidecar differs: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Locked JSON is not an object: {resolved}")
    return payload, observed_sha


def load_lock(path: Path = LOCK_PATH) -> Tuple[Dict[str, object], str]:
    lock, observed_sha = _load_json_with_sidecar(
        path,
        expected_sha256=EXPECTED_LOCK_SHA256,
    )
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("CCR protocol ID differs")
    return lock, observed_sha


def load_authorization(path: Path) -> Tuple[Dict[str, object], str]:
    authorization, observed_sha = _load_json_with_sidecar(path)
    if authorization.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("CCR materializer authorization protocol differs")
    if authorization.get("state") != "materializer_authorized_no_fit":
        raise ValueError("CCR materializer authorization state differs")
    return authorization, observed_sha


def _normalized_path(value: object) -> Optional[str]:
    if isinstance(value, int):
        return None
    try:
        raw = os.fspath(value)
    except (TypeError, ValueError):
        return None
    if not raw:
        return None
    normalized = os.path.normcase(
        os.path.normpath(os.path.realpath(os.path.abspath(raw)))
    )
    if normalized.startswith("\\\\?\\"):
        normalized = normalized[4:]
    return normalized


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _path_components(path: object) -> Tuple[str, ...]:
    return tuple(
        part
        for part in str(path).replace("/", "\\").casefold().split("\\")
        if part
    )


_ACCESS_LOCK = threading.RLock()
_ACTIVE_LEDGER: Optional["DataAccessLedger"] = None
_HOOK_INSTALLED = False
_HOOK_PROBE_SEEN = False


def _audit_hook(event: str, arguments: Tuple[object, ...]) -> None:
    global _HOOK_PROBE_SEEN
    if event == "trkh.cross_colour_ratio_surface.materializer_access_probe":
        _HOOK_PROBE_SEEN = True
        return
    if event != "open":
        return
    with _ACCESS_LOCK:
        ledger = _ACTIVE_LEDGER
    if ledger is not None:
        ledger.observe_open(arguments)


def _ensure_audit_hook() -> None:
    global _HOOK_INSTALLED
    global _HOOK_PROBE_SEEN
    with _ACCESS_LOCK:
        if not _HOOK_INSTALLED:
            _HOOK_PROBE_SEEN = False
            sys.addaudithook(_audit_hook)
            sys.audit(
                "trkh.cross_colour_ratio_surface.materializer_access_probe"
            )
            if not _HOOK_PROBE_SEEN:
                raise RuntimeError("CCR materializer access probe was not observed")
            _HOOK_INSTALLED = True


class DataAccessLedger:
    def __init__(
        self,
        lock: Mapping[str, object],
        *,
        lock_path: Path,
        extra_allowed_paths: Sequence[Path] = (),
        excluded_roots: Sequence[Path] = (),
        domain_roots: Optional[Mapping[str, Path]] = None,
    ) -> None:
        roots = domain_roots or TRACKED_DOMAIN_ROOTS
        self.domain_roots = {
            str(name): _normalized_path(path)
            for name, path in roots.items()
        }
        if any(value is None for value in self.domain_roots.values()):
            raise ValueError("CCR materializer data-access roots are invalid")
        self.excluded_roots = tuple(
            value
            for value in (
                _normalized_path(path) for path in excluded_roots
            )
            if value is not None
        )
        immutable = lock.get("immutable_inputs")
        if not isinstance(immutable, Mapping):
            raise ValueError("CCR immutable-input lock is missing")
        records: List[Mapping[str, object]] = [
            value
            for value in immutable.values()
            if isinstance(value, Mapping) and "path" in value
        ]
        protocol = lock.get("protocol")
        if isinstance(protocol, Mapping):
            records.append(protocol)
        allowed_paths = [
            _resolve_locked_path(record["path"]) for record in records
        ]
        allowed_paths.extend(
            (
                Path(lock_path).resolve(),
                Path(lock_path).resolve().with_suffix(".sha256"),
                ERRATUM_PATH,
                ERRATUM_PATH.with_suffix(".sha256"),
                *extra_allowed_paths,
            )
        )
        self.allowed_exact_paths = {
            value
            for value in (
                _normalized_path(path) for path in allowed_paths
            )
            if value is not None
        }
        self.cohort_paths_authorized = False
        self.events: List[Dict[str, object]] = []
        self.blocked_attempts: List[Dict[str, object]] = []

    @staticmethod
    def _write_requested(mode: object, flags: object) -> bool:
        text = "" if mode is None else str(mode).casefold()
        if any(marker in text for marker in ("w", "a", "x", "+")):
            return True
        if isinstance(flags, int):
            write_flags = (
                os.O_WRONLY
                | os.O_RDWR
                | os.O_CREAT
                | os.O_TRUNC
                | os.O_APPEND
            )
            return bool(int(flags) & int(write_flags))
        return False

    def _domain(self, path: str) -> Optional[str]:
        for name, root in self.domain_roots.items():
            assert root is not None
            if _path_is_within(path, root):
                return name
        return None

    def _is_excluded(self, path: str) -> bool:
        return any(
            _path_is_within(path, root) for root in self.excluded_roots
        )

    def authorize_cohort_paths(
        self,
        records: Sequence[CohortRecord],
    ) -> None:
        if len(records) != ROWS:
            raise ValueError("CCR cohort authorization row count differs")
        paths = {
            *[record.image_path for record in records],
            *[record.label_path for record in records],
        }
        for path in paths:
            components = set(_path_components(path))
            if components.intersection(FORBIDDEN_SPLIT_COMPONENTS):
                raise ValueError(f"Forbidden split in cohort path: {path}")
            normalized = _normalized_path(path)
            if normalized is None:
                raise ValueError(f"Cannot normalize cohort path: {path}")
            self.allowed_exact_paths.add(normalized)
        self.cohort_paths_authorized = True

    def observe_open(self, arguments: Tuple[object, ...]) -> None:
        if not arguments:
            return
        path = _normalized_path(arguments[0])
        if path is None or self._is_excluded(path):
            return
        domain = self._domain(path)
        if domain is None:
            return
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else None
        forbidden = sorted(
            set(_path_components(path)).intersection(
                FORBIDDEN_SPLIT_COMPONENTS
            )
        )
        reason: Optional[str] = None
        if forbidden:
            reason = "forbidden_split_component"
        elif self._write_requested(mode, flags):
            reason = "write_to_data_domain"
        elif path not in self.allowed_exact_paths:
            reason = "undeclared_input"
        record = {
            "sequence": len(self.events),
            "domain": domain,
            "path": path,
            "mode": None if mode is None else str(mode),
            "flags": None if not isinstance(flags, int) else int(flags),
            "decision": "blocked" if reason else "allowed",
            "reason": reason,
            "forbidden_components": forbidden,
            "cohort_paths_authorized": bool(
                self.cohort_paths_authorized
            ),
        }
        self.events.append(record)
        if reason is not None:
            self.blocked_attempts.append(dict(record))
            raise PermissionError(
                f"CCR materializer blocked {reason}: {path}"
            )

    def __enter__(self) -> "DataAccessLedger":
        global _ACTIVE_LEDGER
        _ensure_audit_hook()
        with _ACCESS_LOCK:
            if _ACTIVE_LEDGER is not None:
                raise RuntimeError("A data-access ledger is already active")
            _ACTIVE_LEDGER = self
        return self

    def __exit__(self, *args: object) -> None:
        global _ACTIVE_LEDGER
        with _ACCESS_LOCK:
            if _ACTIVE_LEDGER is self:
                _ACTIVE_LEDGER = None

    def snapshot(self) -> Dict[str, object]:
        events = [dict(row) for row in self.events]
        canonical = json.dumps(
            events,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path_counts: Dict[str, int] = {}
        for row in events:
            path = str(row["path"])
            path_counts[path] = path_counts.get(path, 0) + 1
        logical_path_counts: Dict[str, int] = {}
        previous: Optional[Mapping[str, object]] = None
        for row in events:
            duplicate_os_event = bool(
                row["mode"] is None
                and previous is not None
                and previous["mode"] is not None
                and previous["path"] == row["path"]
                and previous["flags"] == row["flags"]
            )
            if not duplicate_os_event:
                path = str(row["path"])
                logical_path_counts[path] = (
                    logical_path_counts.get(path, 0) + 1
                )
            previous = row
        forbidden_counts = {
            name: sum(
                int(name in row["forbidden_components"]) for row in events
            )
            for name in sorted(FORBIDDEN_SPLIT_COMPONENTS)
        }
        return {
            "schema_version": 1,
            "hook_installed": bool(_HOOK_INSTALLED),
            "hook_probe_seen": bool(_HOOK_PROBE_SEEN),
            "domain_roots": self.domain_roots,
            "excluded_roots": list(self.excluded_roots),
            "cohort_paths_authorized": bool(
                self.cohort_paths_authorized
            ),
            "observed_event_count": len(events),
            "unique_path_count": len(path_counts),
            "path_counts": dict(sorted(path_counts.items())),
            "logical_open_count": int(
                sum(logical_path_counts.values())
            ),
            "logical_path_counts": dict(
                sorted(logical_path_counts.items())
            ),
            "blocked_attempt_count": len(self.blocked_attempts),
            "forbidden_component_counts": forbidden_counts,
            "validation_open_count": (
                forbidden_counts["val"]
                + forbidden_counts["valid"]
                + forbidden_counts["validation"]
            ),
            "test_open_count": forbidden_counts["test"],
            "ordered_events_sha256": hashlib.sha256(canonical).hexdigest(),
            "events": events,
            "blocked_attempts": [
                dict(row) for row in self.blocked_attempts
            ],
            "passed": (
                bool(_HOOK_INSTALLED)
                and bool(_HOOK_PROBE_SEEN)
                and len(events) > 0
                and not self.blocked_attempts
            ),
        }


def _read_locked_bytes(
    name: str,
    record: Mapping[str, object],
) -> bytes:
    path = _resolve_locked_path(record["path"])
    payload = path.read_bytes()
    if (
        len(payload) != int(record["bytes"])
        or sha256_bytes(payload) != str(record["sha256"])
    ):
        raise ValueError(f"Locked CCR input differs: {name}")
    return payload


def _verify_locked_stream(
    name: str,
    record: Mapping[str, object],
) -> Dict[str, object]:
    path = _resolve_locked_path(record["path"])
    observed = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    observed["passed"] = bool(
        int(observed["bytes"]) == int(record["bytes"])
        and observed["sha256"] == record["sha256"]
    )
    if not observed["passed"]:
        raise ValueError(f"Locked CCR input differs: {name}")
    return observed


def _geometry_from_bytes(
    payload: bytes,
    lock: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    expected = {
        "sample_indices": ((ROWS,), np.dtype(np.int64)),
        "targets": ((ROWS,), np.dtype(np.int64)),
        "folds": ((ROWS,), np.dtype(np.int64)),
        "source_stems": ((ROWS,), None),
        "keeper_probabilities": ((ROWS, 5), np.dtype(np.float32)),
        "valid_masks": ((ROWS, FEATURE_SIZE, FEATURE_SIZE), np.dtype(np.bool_)),
        "bbox_masks": ((ROWS, FEATURE_SIZE, FEATURE_SIZE), np.dtype(np.bool_)),
        "model_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "crop_boxes": ((ROWS, 4), np.dtype(np.float32)),
    }
    if set(arrays) != set(expected):
        raise ValueError("CCR geometry keys differ")
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"CCR geometry shape differs: {name}")
        if dtype is not None and arrays[name].dtype != dtype:
            raise ValueError(f"CCR geometry dtype differs: {name}")
    cohort = lock["cohort"]
    hashes = {
        "sample_indices": "sample_indices_sha256",
        "targets": "targets_sha256",
        "folds": "folds_sha256",
        "keeper_probabilities": "probabilities_sha256",
        "valid_masks": "valid_masks_sha256",
        "bbox_masks": "bbox_masks_sha256",
        "model_boxes": "model_boxes_sha256",
        "crop_boxes": "crop_boxes_sha256",
    }
    for name, key in hashes.items():
        if array_sha256(arrays[name]) != cohort[key]:
            raise ValueError(f"CCR geometry array SHA differs: {name}")
    sources = arrays["source_stems"].astype(str).tolist()
    if strings_sha256(sources) != cohort["sources_sha256"]:
        raise ValueError("CCR geometry source SHA differs")
    return arrays


def _cidt_clean_from_bytes(payload: bytes) -> Dict[int, Mapping[str, str]]:
    rows: Dict[int, Mapping[str, str]] = {}
    text = payload.decode("utf-8")
    for row in csv.DictReader(io.StringIO(text, newline="")):
        if row["condition"] != "clean":
            continue
        sample_index = int(row["sample_index"])
        if sample_index in rows:
            raise ValueError(f"Duplicate CIDT clean row: {sample_index}")
        rows[sample_index] = row
    if len(rows) != 9215 or sorted(rows) != list(range(9215)):
        raise ValueError("CIDT clean rows differ from 0..9214")
    return rows


def label_path_for_image(image_path: Path) -> Path:
    resolved = Path(image_path).resolve()
    parts = list(resolved.parts)
    lowered = [part.casefold() for part in parts]
    try:
        image_component = lowered.index("images")
    except ValueError as error:
        raise ValueError(
            f"Image path has no images component: {resolved}"
        ) from error
    if (
        image_component + 1 >= len(parts)
        or lowered[image_component + 1] != "train"
    ):
        raise ValueError(f"Image path is not in images/train: {resolved}")
    parts[image_component] = "labels"
    return Path(
        *parts[:-1],
        f"{resolved.stem}.txt",
    ).resolve()


def build_cohort_records(
    geometry: Mapping[str, np.ndarray],
    cidt: Mapping[int, Mapping[str, str]],
    lock: Mapping[str, object],
) -> List[CohortRecord]:
    records: List[CohortRecord] = []
    probabilities = np.asarray(
        geometry["keeper_probabilities"],
        dtype=np.float32,
    )
    predictions = probabilities.argmax(axis=1)
    sources = np.asarray(geometry["source_stems"]).astype(str)
    for position, sample_index in enumerate(
        np.asarray(geometry["sample_indices"], dtype=np.int64).tolist()
    ):
        row = cidt[int(sample_index)]
        target = int(geometry["targets"][position])
        fold = int(geometry["folds"][position])
        source = str(sources[position])
        if int(row["target_index"]) != target:
            raise ValueError(f"CIDT target mismatch at {sample_index}")
        if int(row["fold"]) != fold:
            raise ValueError(f"CIDT fold mismatch at {sample_index}")
        if str(row["source_stem"]) != source:
            raise ValueError(f"CIDT source mismatch at {sample_index}")
        if int(row["keeper_prediction"]) != int(predictions[position]):
            raise ValueError(f"CIDT prediction mismatch at {sample_index}")
        image_path = Path(row["image_path"]).resolve()
        components = set(_path_components(image_path))
        if (
            "train" not in components
            or components.intersection(FORBIDDEN_SPLIT_COMPONENTS)
            or image_path.stem.casefold() != source.casefold()
        ):
            raise ValueError(f"CIDT path identity differs: {image_path}")
        records.append(
            CohortRecord(
                position=position,
                sample_index=int(sample_index),
                target=target,
                fold=fold,
                source_stem=source,
                image_path=image_path,
                label_path=label_path_for_image(image_path),
                model_bbox=tuple(
                    float(value)
                    for value in geometry["model_boxes"][position].tolist()
                ),
            )
        )
    image_paths = [record.image_path for record in records]
    label_paths = [record.label_path for record in records]
    cohort = lock["cohort"]
    if (
        len(records) != ROWS
        or len(set(image_paths)) != UNIQUE_FILES
        or len(set(label_paths)) != UNIQUE_FILES
        or strings_sha256(image_paths)
        != cohort["image_files"]["ordered_paths_sha256"]
        or strings_sha256(label_paths)
        != cohort["label_files"]["ordered_paths_sha256"]
    ):
        raise ValueError("CCR cohort path lock differs")
    return records


def parse_yolo_label_bytes(
    payload: bytes,
    *,
    num_classes: int = 5,
) -> List[LabelObject]:
    objects: List[LabelObject] = []
    lines = payload.decode("utf-8", errors="replace").splitlines()
    for object_index, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            label = int(parts[0])
            bbox = tuple(float(value) for value in parts[1:5])
        except ValueError:
            continue
        if label < 0 or label >= int(num_classes):
            continue
        if (
            any(value < 0.0 or value > 1.0 for value in bbox)
            or bbox[2] <= 0.0
            or bbox[3] <= 0.0
        ):
            continue
        objects.append(
            LabelObject(
                label=label,
                bbox=bbox,
                object_index=object_index,
            )
        )
    if not objects:
        raise ValueError("Locked label has no valid objects")
    return objects


def select_locked_object(
    objects: Sequence[LabelObject],
    *,
    target: int,
    model_bbox: Sequence[float],
) -> LabelObject:
    locked = np.asarray(model_bbox, dtype=np.float32)
    matches = [
        obj
        for obj in objects
        if obj.label == int(target)
        and np.array_equal(
            np.asarray(obj.bbox, dtype=np.float32),
            locked,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "Locked target/bbox does not select exactly one label object"
        )
    return matches[0]


def crop_primary_object_view(
    image: Image.Image,
    objects: Sequence[LabelObject],
    primary: LabelObject,
    *,
    margin_ratio: float = 0.05,
) -> Tuple[
    Image.Image,
    Tensor,
    Tensor,
    Tuple[int, int, int, int],
]:
    width, height = image.size
    crop_box = margin_crop_box(
        width,
        height,
        primary.bbox,
        margin_ratio=margin_ratio,
    )
    left, top, right, bottom = crop_box
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    kept_labels: List[int] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for obj in objects:
        x1, y1, x2, y2 = bbox_xywh_to_xyxy(
            obj.bbox,
            width=width,
            height=height,
        )
        inter_x1 = max(float(left), x1)
        inter_y1 = max(float(top), y1)
        inter_x2 = min(float(right), x2)
        inter_y2 = min(float(bottom), y2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue
        box_width = inter_x2 - inter_x1
        box_height = inter_y2 - inter_y1
        kept_labels.append(int(obj.label))
        kept_boxes.append(
            (
                float(
                    ((inter_x1 + inter_x2) / 2.0 - left)
                    / crop_width
                ),
                float(
                    ((inter_y1 + inter_y2) / 2.0 - top)
                    / crop_height
                ),
                float(box_width / crop_width),
                float(box_height / crop_height),
            )
        )
    if not kept_boxes:
        raise ValueError("Locked crop has no intersecting object")
    return (
        image.crop(crop_box),
        torch.tensor(kept_labels, dtype=torch.long),
        torch.tensor(kept_boxes, dtype=torch.float32).clamp(0.0, 1.0),
        crop_box,
    )


def build_frozen_eval_transform(semantics: Mapping[str, object]):
    return build_eval_transform(
        image_size=int(semantics["image_size"]),
        resize_mode=str(semantics["resize_mode"]),
        illumination_normalization=bool(
            semantics["illumination_normalization"]
        ),
        illumination_normalization_strength=float(
            semantics["illumination_normalization_strength"]
        ),
        foreground_crop_mode=str(semantics["foreground_crop_mode"]),
        foreground_crop_margin_ratio=float(
            semantics["foreground_crop_margin_ratio"]
        ),
        foreground_crop_min_mask_area_ratio=float(
            semantics["foreground_crop_min_mask_area_ratio"]
        ),
        foreground_crop_max_mask_area_ratio=float(
            semantics["foreground_crop_max_mask_area_ratio"]
        ),
        foreground_crop_max_crop_area_ratio=float(
            semantics["foreground_crop_max_crop_area_ratio"]
        ),
        background_suppression_mode=str(
            semantics["background_suppression_mode"]
        ),
        background_suppression_margin=float(
            semantics["background_suppression_margin"]
        ),
        background_suppression_blur_radius=float(
            semantics["background_suppression_blur_radius"]
        ),
        surface_detail_amplification_mode=str(
            semantics["surface_detail_amplification_mode"]
        ),
        surface_detail_amplification_strength=float(
            semantics["surface_detail_amplification_strength"]
        ),
        surface_detail_amplification_blur_radius=float(
            semantics["surface_detail_amplification_blur_radius"]
        ),
        surface_detail_amplification_foreground_weight=float(
            semantics["surface_detail_amplification_foreground_weight"]
        ),
        eval_surface_detail_amplification=bool(
            semantics["eval_surface_detail_amplification"]
        ),
        mean=semantics["input_mean"],
        std=semantics["input_std"],
    )


def direct_tensor_sample(
    image: Image.Image,
    objects: Sequence[LabelObject],
    *,
    target: int,
    model_bbox: Sequence[float],
    transform,
) -> DirectTensorSample:
    primary = select_locked_object(
        objects,
        target=target,
        model_bbox=model_bbox,
    )
    cropped, labels, boxes, crop_box = crop_primary_object_view(
        image,
        objects,
        primary,
    )
    transformed = transform(
        cropped,
        target={
            "labels": labels,
            "boxes": boxes,
            "augmentation_scale": torch.tensor(
                [1.0],
                dtype=torch.float32,
            ),
        },
    )
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise TypeError("Frozen eval transform did not return tensor/target")
    model_input, transformed_target = transformed
    transformed_boxes = transformed_target["boxes"]
    if (
        transformed_boxes.ndim != 2
        or transformed_boxes.shape[1] != 4
        or transformed_boxes.shape[0] == 0
    ):
        raise ValueError("Frozen eval transform produced no bbox")
    areas = transformed_boxes[:, 2] * transformed_boxes[:, 3]
    crop_bbox = transformed_boxes[int(torch.argmax(areas).item())]
    image_valid_mask = transformed_target.get("image_mask")
    if (
        not torch.is_tensor(image_valid_mask)
        or image_valid_mask.shape
        != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    ):
        raise ValueError("Frozen eval transform produced no valid mask")
    return DirectTensorSample(
        model_input=model_input.to(dtype=torch.float32),
        image_valid_mask=image_valid_mask.to(dtype=torch.bool),
        model_bbox=torch.tensor(primary.bbox, dtype=torch.float32),
        crop_bbox=crop_bbox.to(dtype=torch.float32).clamp(0.0, 1.0),
        crop_box=crop_box,
        primary_object_index=int(primary.object_index),
    )


def feature_geometry(
    image_valid_mask: Tensor,
    crop_bbox: Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = image_valid_mask.to(dtype=torch.float32).view(
        1,
        1,
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
    )
    valid = F.interpolate(
        mask,
        size=(FEATURE_SIZE, FEATURE_SIZE),
        mode="area",
    )[0, 0] >= 0.5
    box = crop_bbox[:4].to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = box.unbind()
    yy = (
        torch.arange(FEATURE_SIZE, dtype=torch.float32) + 0.5
    ) / float(FEATURE_SIZE)
    xx = (
        torch.arange(FEATURE_SIZE, dtype=torch.float32) + 0.5
    ) / float(FEATURE_SIZE)
    dx = (
        (xx.view(1, FEATURE_SIZE) - cx)
        / (0.5 * width.clamp_min(1e-6))
    ).expand(FEATURE_SIZE, -1)
    dy = (
        (yy.view(FEATURE_SIZE, 1) - cy)
        / (0.5 * height.clamp_min(1e-6))
    ).expand(-1, FEATURE_SIZE)
    bbox = (dx.abs() <= 1.0) & (dy.abs() <= 1.0) & valid
    return valid.numpy(), bbox.numpy()


def model_input_to_srgb_uint8(
    model_input: Tensor,
    *,
    mean: Sequence[float] = INPUT_MEAN,
    std: Sequence[float] = INPUT_STD,
) -> Tuple[Tensor, float, bool]:
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    srgb = model_input.to(dtype=torch.float32) * std_tensor + mean_tensor
    if (
        not torch.isfinite(srgb).all()
        or float(srgb.min()) < -1e-6
        or float(srgb.max()) > 1.0 + 1e-6
    ):
        raise ValueError("Model input cannot be inverted to bounded sRGB")
    quantized = torch.round(srgb.clamp(0.0, 1.0) * 255.0).to(
        dtype=torch.uint8
    )
    replay = TF.normalize(
        quantized.to(dtype=torch.float32).div(255.0),
        mean=list(mean),
        std=list(std),
    )
    error = float(
        (replay - model_input.to(dtype=torch.float32)).abs().max()
    )
    return quantized, error, bool(torch.equal(replay, model_input))


def pack_valid_mask(mask: Tensor) -> np.ndarray:
    value = mask.detach().cpu().numpy().astype(np.bool_, copy=False)
    if value.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE):
        raise ValueError("Image valid mask shape differs")
    packed = np.packbits(value.reshape(-1), bitorder="little")
    if packed.shape != (PACKED_MASK_BYTES,):
        raise RuntimeError("Packed valid mask shape differs")
    return packed


def unpack_valid_masks(packed: np.ndarray) -> np.ndarray:
    value = np.asarray(packed, dtype=np.uint8)
    if value.ndim != 2 or value.shape[1] != PACKED_MASK_BYTES:
        raise ValueError("Packed valid-mask cache shape differs")
    unpacked = np.unpackbits(
        value,
        axis=1,
        count=MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE,
        bitorder="little",
    )
    return unpacked.reshape(
        value.shape[0],
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
    ).astype(np.bool_)


def _verify_transform_sources() -> Dict[str, object]:
    observed: Dict[str, object] = {}
    for path, expected_sha in TRANSFORM_SOURCE_SHA256.items():
        sha = sha256_file(path)
        if sha != expected_sha:
            raise ValueError(f"Frozen transform source differs: {path}")
        observed[str(path.relative_to(REPO_ROOT))] = {
            "sha256": sha,
            "passed": True,
        }
    erratum_sha = sha256_file(ERRATUM_PATH)
    if erratum_sha != EXPECTED_ERRATUM_SHA256:
        raise ValueError("CCR same-tensor erratum differs")
    return {
        "erratum_sha256": erratum_sha,
        "sources": observed,
        "passed": True,
    }


def _verify_transform_semantics(
    resolved_config: Mapping[str, object],
) -> Dict[str, object]:
    observed = _eval_semantics(resolved_config)
    differences = {
        key: {
            "expected": EXPECTED_TRANSFORM_SEMANTICS[key],
            "observed": observed.get(key),
        }
        for key in EXPECTED_TRANSFORM_SEMANTICS
        if observed.get(key) != EXPECTED_TRANSFORM_SEMANTICS[key]
    }
    if differences:
        raise ValueError(
            f"Frozen CCR transform semantics differ: {differences}"
        )
    return {
        "semantics": {
            key: observed[key]
            for key in EXPECTED_TRANSFORM_SEMANTICS
        },
        "sha256": json_sha256(
            {
                key: observed[key]
                for key in EXPECTED_TRANSFORM_SEMANTICS
            }
        ),
        "passed": True,
    }


def load_scientific_inputs(
    lock: Mapping[str, object],
    *,
    ledger: DataAccessLedger,
) -> Dict[str, object]:
    immutable = lock["immutable_inputs"]
    required_payload_names = {
        "geometry_cache",
        "cidt_predictions",
        "keeper_config",
    }
    payloads: Dict[str, bytes] = {}
    verified: Dict[str, object] = {}
    for name in sorted(immutable):
        record = immutable[name]
        if name in required_payload_names:
            payload = _read_locked_bytes(name, record)
            payloads[name] = payload
            verified[name] = {
                "path": str(_resolve_locked_path(record["path"])),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "passed": True,
            }
        else:
            verified[name] = _verify_locked_stream(name, record)
    protocol = lock["protocol"]
    verified["protocol"] = _verify_locked_stream("protocol", protocol)
    sources = _verify_transform_sources()
    geometry = _geometry_from_bytes(
        payloads["geometry_cache"],
        lock,
    )
    cidt = _cidt_clean_from_bytes(payloads["cidt_predictions"])
    config = json.loads(payloads["keeper_config"].decode("utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Keeper resolved config is not a mapping")
    transform = _verify_transform_semantics(config)
    records = build_cohort_records(geometry, cidt, lock)
    ledger.authorize_cohort_paths(records)
    return {
        "geometry": geometry,
        "records": records,
        "resolved_config": config,
        "transform": transform,
        "source_verification": sources,
        "immutable_verification": verified,
    }


def _predicted_cache_bytes() -> Dict[str, int]:
    srgb_payload = ROWS * 3 * MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE
    valid_payload = ROWS * PACKED_MASK_BYTES
    conservative_headers_and_metadata = 4 * 1024 * 1024
    return {
        "srgb_payload": srgb_payload,
        "valid_mask_payload": valid_payload,
        "conservative_headers_and_metadata": conservative_headers_and_metadata,
        "total_upper_bound": (
            srgb_payload
            + valid_payload
            + conservative_headers_and_metadata
        ),
    }


def structural_preflight(
    *,
    lock_path: Path = LOCK_PATH,
) -> Dict[str, object]:
    lock, lock_sha = load_lock(lock_path)
    with DataAccessLedger(
        lock,
        lock_path=lock_path,
    ) as ledger:
        inputs = load_scientific_inputs(lock, ledger=ledger)
        access = ledger.snapshot()
    predicted = _predicted_cache_bytes()
    temporary_limit = int(lock["resource_limits"]["temporary_cache_bytes"])
    checks = {
        "no_image_or_label_open": all(
            not (
                "yolo_f\\images\\train" in str(row["path"])
                or "yolo_f\\labels\\train" in str(row["path"])
            )
            for row in access["events"]
        ),
        "rows_exact": len(inputs["records"]) == ROWS,
        "unique_images_exact": len(
            {record.image_path for record in inputs["records"]}
        )
        == UNIQUE_FILES,
        "unique_labels_exact": len(
            {record.label_path for record in inputs["records"]}
        )
        == UNIQUE_FILES,
        "transform_exact": bool(inputs["transform"]["passed"]),
        "sources_exact": bool(
            inputs["source_verification"]["passed"]
        ),
        "access_ledger_passed": bool(access["passed"]),
        "temporary_cache_within_lock": (
            predicted["total_upper_bound"] <= temporary_limit
        ),
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "state": "structural_preflight_no_image_read_no_fit",
        "lock_sha256": lock_sha,
        "module_sha256": sha256_file(MODULE_PATH),
        "rows": len(inputs["records"]),
        "unique_images": len(
            {record.image_path for record in inputs["records"]}
        ),
        "unique_labels": len(
            {record.label_path for record in inputs["records"]}
        ),
        "transform": inputs["transform"],
        "source_verification": inputs["source_verification"],
        "cache_budget": {
            **predicted,
            "temporary_limit": temporary_limit,
        },
        "access_ledger": access,
        "checks": checks,
        "passed": all(checks.values()),
        "candidate_descriptor_created": False,
        "candidate_model_state_created": False,
        "candidate_metric_observed": False,
        "validation_data_used": False,
        "test_data_used": False,
    }


def _file_manifest_from_observed(
    ordered_paths: Sequence[Path],
    observed: Mapping[Path, Mapping[str, object]],
) -> Dict[str, object]:
    ordered = [Path(path).resolve() for path in ordered_paths]
    unique = sorted(
        set(ordered),
        key=lambda path: str(path).casefold(),
    )
    unique_rows = [dict(observed[path]) for path in unique]
    return {
        "ordered_rows": len(ordered),
        "unique_files": len(unique),
        "unique_bytes": int(
            sum(int(row["bytes"]) for row in unique_rows)
        ),
        "ordered_paths_sha256": strings_sha256(ordered),
        "ordered_content_sha256": strings_sha256(
            observed[path]["sha256"] for path in ordered
        ),
        "unique_manifest_sha256": json_sha256(unique_rows),
    }


def _verify_observed_manifest(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    name: str,
) -> None:
    if dict(observed) != dict(expected):
        raise ValueError(f"CCR {name} manifest differs")


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(payload)


def _close_memmap(value: np.memmap) -> None:
    value.flush()
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _artifact_record(path: Path) -> Dict[str, object]:
    return {
        "path": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_file(path),
    }


def _npz_arrays_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            digest.update(name.encode("utf-8"))
            digest.update(
                bytes.fromhex(array_sha256(np.asarray(archive[name])))
            )
    return digest.hexdigest()


def _directory_manifest(
    root: Path,
    *,
    excluded_names: Sequence[str] = (),
) -> Dict[str, object]:
    excluded = set(excluded_names)
    files = [
        path
        for path in sorted(
            Path(root).rglob("*"),
            key=lambda value: str(value.relative_to(root)).casefold(),
        )
        if path.is_file() and path.name not in excluded
    ]
    rows = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "schema_version": 1,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "rows_sha256": json_sha256(rows),
    }


def _verify_authorization(
    authorization: Mapping[str, object],
    *,
    authorization_path: Path,
    lock: Mapping[str, object],
    lock_sha256: str,
    output: Path,
) -> Dict[str, object]:
    expected = authorization.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError("CCR materializer authorization expected block missing")
    paths = {
        "lock": LOCK_PATH,
        "erratum": ERRATUM_PATH,
        "module": MODULE_PATH,
        "engine": (
            REPO_ROOT
            / "trkh"
            / "tools"
            / "cross_colour_ratio_surface_a0_engine.py"
        ),
        "test": (
            REPO_ROOT
            / "tests"
            / "test_cross_colour_ratio_surface_a0_materializer.py"
        ),
    }
    observed = {
        name: sha256_file(path)
        for name, path in paths.items()
    }
    checks = {
        f"{name}_sha256": observed[name] == expected.get(
            f"{name}_sha256"
        )
        for name in paths
    }
    checks["lock_argument_sha256"] = lock_sha256 == observed["lock"]
    checks["authorization_output_exact"] = (
        str(Path(output).resolve())
        == str(_resolve_locked_path(authorization["output_dir"]))
    )
    checks["authorization_sidecar_present"] = (
        Path(authorization_path).with_suffix(".sha256").is_file()
    )
    authorized_limits = authorization.get("resource_limits")
    checks["resource_limits_exact"] = bool(
        isinstance(authorized_limits, Mapping)
        and dict(authorized_limits) == dict(lock["resource_limits"])
    )
    execution_constraints = authorization.get("execution_constraints")
    checks["execution_constraints_exact"] = bool(
        isinstance(execution_constraints, Mapping)
        and dict(execution_constraints) == REQUIRED_EXECUTION_CONSTRAINTS
    )
    if not all(checks.values()):
        raise ValueError(
            f"CCR materializer authorization differs: {checks}"
        )
    return {
        "observed": observed,
        "checks": checks,
        "passed": True,
    }


def _repository_state(
    *,
    required_ancestor: str,
) -> Dict[str, object]:
    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=check,
            capture_output=True,
            text=True,
        )

    head = git("rev-parse", "HEAD").stdout.strip()
    upstream = git(
        "rev-parse",
        "origin/classification-only-research",
    ).stdout.strip()
    ancestor = (
        git(
            "merge-base",
            "--is-ancestor",
            str(required_ancestor),
            head,
            check=False,
        ).returncode
        == 0
    )
    status = [
        line
        for line in git(
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ).stdout.splitlines()
        if line
    ]
    protected = {
        "?? BaoCao/",
        "?? deep-research-report (9).md",
        "?? deep-research-report (10).md",
    }
    unexpected = [line for line in status if line not in protected]
    result = {
        "branch": git("branch", "--show-current").stdout.strip(),
        "head": head,
        "upstream": upstream,
        "required_ancestor": str(required_ancestor),
        "required_ancestor_present": ancestor,
        "status": status,
        "unexpected_status": unexpected,
        "passed": bool(
            head == upstream
            and ancestor
            and not unexpected
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"CCR materializer repository gate failed: {result}")
    return result


def materialize(
    *,
    output: Path,
    authorization_path: Path,
    lock_path: Path = LOCK_PATH,
    authorization_output: Optional[Path] = None,
) -> Dict[str, object]:
    output = Path(output).resolve()
    authorization_path = Path(authorization_path).resolve()
    if output.exists():
        raise FileExistsError(f"CCR materializer output exists: {output}")
    lock, lock_sha = load_lock(lock_path)
    authorization, authorization_sha = load_authorization(
        authorization_path
    )
    authorization_check = _verify_authorization(
        authorization,
        authorization_path=authorization_path,
        lock=lock,
        lock_sha256=lock_sha,
        output=(
            output
            if authorization_output is None
            else Path(authorization_output).resolve()
        ),
    )
    repository = _repository_state(
        required_ancestor=str(authorization["required_ancestor_commit"])
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("CCR materializer requires an uninitialized CUDA context")
    predicted = _predicted_cache_bytes()
    temporary_limit = int(lock["resource_limits"]["temporary_cache_bytes"])
    if predicted["total_upper_bound"] > temporary_limit:
        raise RuntimeError("CCR materializer predicted cache exceeds lock")
    process = psutil.Process(os.getpid())
    initial_rss = int(process.memory_info().rss)
    if initial_rss > int(lock["resource_limits"]["process_rss_bytes"]):
        raise RuntimeError("CCR materializer initial RSS exceeds lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output.parent)
    if usage.free < 2 * predicted["total_upper_bound"]:
        raise RuntimeError("CCR materializer free disk gate failed")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=str(output.parent),
        )
    ).resolve()
    started = time.perf_counter()
    peak_rss = initial_rss
    srgb_cache: Optional[np.memmap] = None
    valid_cache: Optional[np.memmap] = None
    try:
        with DataAccessLedger(
            lock,
            lock_path=lock_path,
            extra_allowed_paths=(
                authorization_path,
                authorization_path.with_suffix(".sha256"),
            ),
            excluded_roots=(temporary, output),
        ) as ledger:
            inputs = load_scientific_inputs(lock, ledger=ledger)
            geometry = inputs["geometry"]
            records: List[CohortRecord] = inputs["records"]
            transform = build_frozen_eval_transform(
                inputs["transform"]["semantics"]
            )
            srgb_path = temporary / SRGB_CACHE_NAME
            valid_path = temporary / VALID_MASK_CACHE_NAME
            srgb_cache = np.lib.format.open_memmap(
                srgb_path,
                mode="w+",
                dtype=np.uint8,
                shape=(
                    ROWS,
                    3,
                    MODEL_IMAGE_SIZE,
                    MODEL_IMAGE_SIZE,
                ),
            )
            valid_cache = np.lib.format.open_memmap(
                valid_path,
                mode="w+",
                dtype=np.uint8,
                shape=(ROWS, PACKED_MASK_BYTES),
            )

            grouped: Dict[Path, List[CohortRecord]] = {}
            for record in records:
                grouped.setdefault(record.image_path, []).append(record)
            image_observed: Dict[Path, Mapping[str, object]] = {}
            label_observed: Dict[Path, Mapping[str, object]] = {}
            completed = np.zeros(ROWS, dtype=np.bool_)
            maximum_crop_bbox_error = 0.0
            maximum_model_bbox_error = 0.0
            maximum_roundtrip_error = 0.0
            crop_boxes_exact = True
            model_boxes_exact = True
            valid_masks_exact = True
            bbox_masks_exact = True
            model_tensor_roundtrip_exact = True
            for image_path in sorted(
                grouped,
                key=lambda path: str(path).casefold(),
            ):
                rows = sorted(
                    grouped[image_path],
                    key=lambda record: record.position,
                )
                label_path = rows[0].label_path
                if any(record.label_path != label_path for record in rows):
                    raise ValueError("One image maps to multiple labels")
                image_bytes = image_path.read_bytes()
                label_bytes = label_path.read_bytes()
                image_observed[image_path] = {
                    "path": str(image_path),
                    "bytes": len(image_bytes),
                    "sha256": sha256_bytes(image_bytes),
                }
                label_observed[label_path] = {
                    "path": str(label_path),
                    "bytes": len(label_bytes),
                    "sha256": sha256_bytes(label_bytes),
                }
                with Image.open(io.BytesIO(image_bytes)) as decoded:
                    image = decoded.convert("RGB").copy()
                objects = parse_yolo_label_bytes(label_bytes)
                for record in rows:
                    sample = direct_tensor_sample(
                        image,
                        objects,
                        target=record.target,
                        model_bbox=record.model_bbox,
                        transform=transform,
                    )
                    position = record.position
                    locked_model = np.asarray(
                        geometry["model_boxes"][position],
                        dtype=np.float32,
                    )
                    observed_model = (
                        sample.model_bbox.detach().cpu().numpy()
                    )
                    locked_crop = np.asarray(
                        geometry["crop_boxes"][position],
                        dtype=np.float32,
                    )
                    observed_crop = (
                        sample.crop_bbox.detach().cpu().numpy()
                    )
                    model_error = float(
                        np.max(
                            np.abs(
                                observed_model.astype(np.float64)
                                - locked_model.astype(np.float64)
                            )
                        )
                    )
                    crop_error = float(
                        np.max(
                            np.abs(
                                observed_crop.astype(np.float64)
                                - locked_crop.astype(np.float64)
                            )
                        )
                    )
                    maximum_model_bbox_error = max(
                        maximum_model_bbox_error,
                        model_error,
                    )
                    maximum_crop_bbox_error = max(
                        maximum_crop_bbox_error,
                        crop_error,
                    )
                    model_boxes_exact = (
                        model_boxes_exact
                        and np.array_equal(observed_model, locked_model)
                    )
                    crop_boxes_exact = (
                        crop_boxes_exact
                        and np.array_equal(observed_crop, locked_crop)
                    )
                    valid16, bbox16 = feature_geometry(
                        sample.image_valid_mask,
                        sample.crop_bbox,
                    )
                    valid_masks_exact = (
                        valid_masks_exact
                        and np.array_equal(
                            valid16,
                            geometry["valid_masks"][position],
                        )
                    )
                    bbox_masks_exact = (
                        bbox_masks_exact
                        and np.array_equal(
                            bbox16,
                            geometry["bbox_masks"][position],
                        )
                    )
                    srgb, roundtrip_error, roundtrip_exact = (
                        model_input_to_srgb_uint8(sample.model_input)
                    )
                    maximum_roundtrip_error = max(
                        maximum_roundtrip_error,
                        roundtrip_error,
                    )
                    model_tensor_roundtrip_exact = (
                        model_tensor_roundtrip_exact
                        and roundtrip_exact
                    )
                    srgb_cache[position] = srgb.numpy()
                    valid_cache[position] = pack_valid_mask(
                        sample.image_valid_mask
                    )
                    completed[position] = True
                current_rss = int(process.memory_info().rss)
                peak_rss = max(peak_rss, current_rss)
                if current_rss > int(
                    lock["resource_limits"]["process_rss_bytes"]
                ):
                    raise RuntimeError(
                        "CCR materializer process RSS exceeds lock"
                    )
                if time.perf_counter() - started > int(
                    lock["resource_limits"]["clean_wall_seconds"]
                ):
                    raise RuntimeError(
                        "CCR materializer clean wall time exceeds lock"
                    )
                if torch.cuda.is_initialized():
                    raise RuntimeError(
                        "CCR materializer initialized CUDA during image pass"
                    )

            image_manifest = _file_manifest_from_observed(
                [record.image_path for record in records],
                image_observed,
            )
            label_manifest = _file_manifest_from_observed(
                [record.label_path for record in records],
                label_observed,
            )
            _verify_observed_manifest(
                image_manifest,
                lock["cohort"]["image_files"],
                name="image",
            )
            _verify_observed_manifest(
                label_manifest,
                lock["cohort"]["label_files"],
                name="label",
            )
            parity_checks = {
                "all_rows_completed": bool(completed.all()),
                "image_manifest_exact": True,
                "label_manifest_exact": True,
                "model_boxes_exact": model_boxes_exact,
                "crop_boxes_exact": crop_boxes_exact,
                "valid_masks_exact": valid_masks_exact,
                "bbox_masks_exact": bbox_masks_exact,
                "model_tensor_roundtrip_exact": (
                    model_tensor_roundtrip_exact
                ),
            }
            if not all(parity_checks.values()):
                raise RuntimeError(
                    f"CCR same-tensor parity failed: {parity_checks}"
                )
            _close_memmap(srgb_cache)
            srgb_cache = None
            _close_memmap(valid_cache)
            valid_cache = None
            stored_valid = np.load(
                valid_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            unpacked = unpack_valid_masks(stored_valid)
            peak_rss = max(
                peak_rss,
                int(process.memory_info().rss),
            )
            valid_256_replay_exact = True
            for record in records:
                valid16 = F.interpolate(
                    torch.from_numpy(
                        unpacked[record.position]
                    ).float().view(
                        1,
                        1,
                        MODEL_IMAGE_SIZE,
                        MODEL_IMAGE_SIZE,
                    ),
                    size=(FEATURE_SIZE, FEATURE_SIZE),
                    mode="area",
                )[0, 0].numpy() >= 0.5
                valid_256_replay_exact = (
                    valid_256_replay_exact
                    and np.array_equal(
                        valid16,
                        geometry["valid_masks"][record.position],
                    )
                )
            del stored_valid
            if not valid_256_replay_exact:
                raise RuntimeError("Packed valid-mask replay differs")
            np.savez(
                temporary / COHORT_ARRAYS_NAME,
                sample_indices=np.asarray(
                    geometry["sample_indices"],
                    dtype=np.int64,
                ),
                targets=np.asarray(
                    geometry["targets"],
                    dtype=np.int64,
                ),
                folds=np.asarray(
                    geometry["folds"],
                    dtype=np.int64,
                ),
                source_stems=np.asarray(
                    geometry["source_stems"]
                ).astype(str),
                keeper_probabilities=np.asarray(
                    geometry["keeper_probabilities"],
                    dtype=np.float32,
                ),
                model_boxes=np.asarray(
                    geometry["model_boxes"],
                    dtype=np.float32,
                ),
                crop_boxes=np.asarray(
                    geometry["crop_boxes"],
                    dtype=np.float32,
                ),
                image_paths=np.asarray(
                    [str(record.image_path) for record in records]
                ),
                label_paths=np.asarray(
                    [str(record.label_path) for record in records]
                ),
            )
            peak_rss = max(
                peak_rss,
                int(process.memory_info().rss),
            )
            access = ledger.snapshot()

        elapsed = time.perf_counter() - started
        output_records = {
            SRGB_CACHE_NAME: _artifact_record(
                temporary / SRGB_CACHE_NAME
            ),
            VALID_MASK_CACHE_NAME: _artifact_record(
                temporary / VALID_MASK_CACHE_NAME
            ),
            COHORT_ARRAYS_NAME: _artifact_record(
                temporary / COHORT_ARRAYS_NAME
            ),
        }
        for name, record in output_records.items():
            record["path"] = str((output / name).resolve())
        output_records[COHORT_ARRAYS_NAME][
            "array_payload_sha256"
        ] = _npz_arrays_sha256(temporary / COHORT_ARRAYS_NAME)
        cache_bytes = int(
            sum(int(row["bytes"]) for row in output_records.values())
        )
        resource_checks = {
            "temporary_cache_bytes": cache_bytes <= temporary_limit,
            "process_rss_bytes": peak_rss
            <= int(lock["resource_limits"]["process_rss_bytes"]),
            "clean_wall_seconds": elapsed
            <= int(lock["resource_limits"]["clean_wall_seconds"]),
            "cuda_not_used": not torch.cuda.is_initialized(),
        }
        access_checks = {
            "ledger_passed": bool(access["passed"]),
            "blocked_zero": int(access["blocked_attempt_count"]) == 0,
            "validation_zero": int(access["validation_open_count"]) == 0,
            "test_zero": int(access["test_open_count"]) == 0,
            "unique_images_opened_once": all(
                int(access["logical_path_counts"].get(
                    _normalized_path(path),
                    0,
                ))
                == 1
                for path in image_observed
            ),
            "unique_labels_opened_once": all(
                int(access["logical_path_counts"].get(
                    _normalized_path(path),
                    0,
                ))
                == 1
                for path in label_observed
            ),
        }
        automatic_checks = {
            **parity_checks,
            "packed_valid_mask_replay_exact": (
                valid_256_replay_exact
            ),
            **resource_checks,
            **access_checks,
        }
        if not all(automatic_checks.values()):
            raise RuntimeError(
                f"CCR materializer automatic gate failed: {automatic_checks}"
            )
        _write_json(temporary / "access_ledger.json", access)
        summary = {
            "protocol_id": PROTOCOL_ID,
            "state": "materialized_fresh_process_replay_pending",
            "process_id": os.getpid(),
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "lock_sha256": lock_sha,
            "authorization_sha256": authorization_sha,
            "authorization": authorization_check,
            "repository": repository,
            "module_sha256": sha256_file(MODULE_PATH),
            "transform": inputs["transform"],
            "source_verification": inputs["source_verification"],
            "rows": ROWS,
            "unique_images": UNIQUE_FILES,
            "unique_labels": UNIQUE_FILES,
            "image_manifest": image_manifest,
            "label_manifest": label_manifest,
            "parity": {
                "maximum_model_bbox_error": maximum_model_bbox_error,
                "maximum_crop_bbox_error": maximum_crop_bbox_error,
                "maximum_model_tensor_roundtrip_error": (
                    maximum_roundtrip_error
                ),
                "checks": parity_checks,
                "packed_valid_mask_replay_exact": (
                    valid_256_replay_exact
                ),
            },
            "outputs": output_records,
            "resource": {
                "elapsed_seconds": elapsed,
                "peak_process_rss_bytes": peak_rss,
                "cache_bytes": cache_bytes,
                "limits": lock["resource_limits"],
                "checks": resource_checks,
            },
            "access_ledger_sha256": access[
                "ordered_events_sha256"
            ],
            "automatic_checks": automatic_checks,
            "automatic_passed": True,
            "candidate_descriptor_created": False,
            "candidate_model_state_created": False,
            "candidate_metric_observed": False,
            "validation_data_used": False,
            "test_data_used": False,
            "downstream_authorization": {
                "fit": False,
                "validation": False,
                "test": False,
                "production_integration": False,
                "full_train": False,
                "current_best_command_update": False,
            },
        }
        _write_json(temporary / "summary.json", summary)
        manifest = _directory_manifest(
            temporary,
            excluded_names=("formal_manifest.json",),
        )
        _write_json(temporary / "formal_manifest.json", manifest)
        if _directory_manifest(
            temporary,
            excluded_names=("formal_manifest.json",),
        ) != manifest:
            raise RuntimeError("CCR formal materializer manifest differs")
        os.replace(temporary, output)
        return {
            "output": str(output),
            "summary": summary,
            "manifest": manifest,
        }
    except Exception:
        if srgb_cache is not None:
            _close_memmap(srgb_cache)
        if valid_cache is not None:
            _close_memmap(valid_cache)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_output_summary(output: Path) -> Dict[str, object]:
    summary = json.loads(
        (Path(output) / "summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (Path(output) / "formal_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    observed = _directory_manifest(
        output,
        excluded_names=(
            "formal_manifest.json",
            "replay_summary.json",
            "artifact_set_manifest.json",
        ),
    )
    if manifest != observed:
        raise RuntimeError("CCR materializer formal manifest is not intact")
    return summary


def replay_materialization(
    *,
    output: Path,
    authorization_path: Path,
    lock_path: Path = LOCK_PATH,
) -> Dict[str, object]:
    output = Path(output).resolve()
    reference = _load_output_summary(output)
    if int(reference["process_id"]) == os.getpid():
        raise RuntimeError("CCR materializer replay must use a fresh process")
    for name in ("replay_summary.json", "artifact_set_manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"CCR replay artifact exists: {name}")
    replay_output = output.parent / f".{output.name}.replay"
    if replay_output.exists():
        raise FileExistsError(f"CCR replay output exists: {replay_output}")
    started = time.perf_counter()
    try:
        replay = materialize(
            output=replay_output,
            authorization_path=authorization_path,
            lock_path=lock_path,
            authorization_output=output,
        )
        observed = replay["summary"]
        scientific_checks = {
            "fresh_process": int(reference["process_id"])
            != int(observed["process_id"]),
            "lock_exact": reference["lock_sha256"]
            == observed["lock_sha256"],
            "authorization_exact": reference["authorization_sha256"]
            == observed["authorization_sha256"],
            "module_exact": reference["module_sha256"]
            == observed["module_sha256"],
            "transform_exact": json_sha256(reference["transform"])
            == json_sha256(observed["transform"]),
            "image_manifest_exact": reference["image_manifest"]
            == observed["image_manifest"],
            "label_manifest_exact": reference["label_manifest"]
            == observed["label_manifest"],
            "parity_exact": reference["parity"] == observed["parity"],
            "access_ledger_exact": reference["access_ledger_sha256"]
            == observed["access_ledger_sha256"],
            "srgb_cache_exact": (
                reference["outputs"][SRGB_CACHE_NAME]["sha256"]
                == observed["outputs"][SRGB_CACHE_NAME]["sha256"]
            ),
            "valid_mask_cache_exact": (
                reference["outputs"][VALID_MASK_CACHE_NAME]["sha256"]
                == observed["outputs"][VALID_MASK_CACHE_NAME]["sha256"]
            ),
            "cohort_arrays_exact": (
                reference["outputs"][COHORT_ARRAYS_NAME][
                    "array_payload_sha256"
                ]
                == observed["outputs"][COHORT_ARRAYS_NAME][
                    "array_payload_sha256"
                ]
            ),
            "automatic_passed": bool(observed["automatic_passed"]),
        }
        replay_summary = {
            "protocol_id": PROTOCOL_ID,
            "state": "fresh_process_materializer_replay_complete",
            "reference_process_id": int(reference["process_id"]),
            "replay_process_id": int(observed["process_id"]),
            "elapsed_seconds": time.perf_counter() - started,
            "checks": scientific_checks,
            "passed": all(scientific_checks.values()),
            "fit_authorized": False,
            "validation_data_used": False,
            "test_data_used": False,
        }
        if not replay_summary["passed"]:
            raise RuntimeError(
                f"CCR materializer replay differs: {scientific_checks}"
            )
        _write_json(output / "replay_summary.json", replay_summary)
        artifact_manifest = _directory_manifest(
            output,
            excluded_names=("artifact_set_manifest.json",),
        )
        _write_json(
            output / "artifact_set_manifest.json",
            artifact_manifest,
        )
        if _directory_manifest(
            output,
            excluded_names=("artifact_set_manifest.json",),
        ) != artifact_manifest:
            raise RuntimeError("CCR artifact-set manifest differs")
        return replay_summary
    except Exception:
        for name in ("replay_summary.json", "artifact_set_manifest.json"):
            path = output / name
            if path.exists():
                path.unlink()
        raise
    finally:
        if replay_output.exists():
            shutil.rmtree(replay_output)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the locked CCR same-tensor cohort."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--replay", action="store_true")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        result = structural_preflight(lock_path=args.lock)
    else:
        if args.authorization is None:
            raise ValueError(
                "--authorization is required for materialize/replay"
            )
        if args.materialize:
            result = materialize(
                output=args.output_dir,
                authorization_path=args.authorization,
                lock_path=args.lock,
            )
        else:
            result = replay_materialization(
                output=args.output_dir,
                authorization_path=args.authorization,
                lock_path=args.lock,
            )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(result.get("passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

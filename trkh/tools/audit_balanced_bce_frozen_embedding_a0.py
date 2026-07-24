from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import psutil
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS_WORKSPACE_CONFIG

import torch

from trkh.tools.balanced_bce_frozen_embedding_a0_engine import (
    BASE_LR,
    BASE_SEED,
    BATCH_SIZE,
    CLASS_COUNT,
    EPOCHS,
    FEATURE_DIM,
    FOCUS_CLASS,
    MOMENTUM,
    REPEAT_SEED_OFFSET,
    RESTRICTED_NEGATIVE_CLASSES,
    ROLE_NAMES,
    WARMUP_EPOCHS,
    array_sha256,
    build_epoch_orders,
    equation_oracles,
    initial_head_evidence,
    json_sha256,
    learning_rate,
    train_role_fold,
)


MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = MODULE_PATH.parents[2]
LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_BALANCED_BCE_FROZEN_EMBEDDING_A0_LOCK_20260724.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")
EXPECTED_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_CACHE_KEYS = (
    "embeddings",
    "probabilities",
    "labels",
    "base_predictions",
    "paths",
    "sample_index",
)
VISUAL_ANCHORS = (
    4002,
    3195,
    3284,
    4613,
    3217,
    2127,
    2707,
    5213,
    4616,
    4770,
    3955,
    4012,
    4624,
    3248,
    2718,
    5732,
    5327,
    6113,
    4916,
    3624,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _string_sequence_sha256(values: Sequence[object]) -> str:
    encoded = [str(value).encode("utf-8") for value in values]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
    for value in encoded:
        digest.update(np.asarray([len(value)], dtype=np.int64).tobytes())
        digest.update(value)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(row), ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_locked_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _contains_forbidden_split(path: object) -> bool:
    parts = [
        part
        for part in str(path).replace("/", "\\").casefold().split("\\")
        if part
    ]
    return any(part in {"val", "valid", "validation", "test"} for part in parts)


def _is_train_image_path(path: object) -> bool:
    parts = [
        part
        for part in str(path).replace("/", "\\").casefold().split("\\")
        if part
    ]
    return not _contains_forbidden_split(path) and any(
        parts[index : index + 3] == ["yolo_f", "images", "train"]
        for index in range(max(0, len(parts) - 2))
    )


_ACCESS_LOCK = threading.RLock()
_ACTIVE_LEDGER: Optional["DataAccessLedger"] = None
_HOOK_INSTALLED = False
_HOOK_PROBE_SEEN = False


def _normalized_path(value: object) -> Optional[str]:
    if isinstance(value, int):
        return None
    try:
        raw = os.fsdecode(os.fspath(value))
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


def _path_components(path: str) -> Tuple[str, ...]:
    return tuple(
        part
        for part in path.replace("/", "\\").casefold().split("\\")
        if part
    )


def _audit_hook(event: str, arguments: Tuple[object, ...]) -> None:
    global _HOOK_PROBE_SEEN
    if event == "trkh.balanced_bce.data_access_probe":
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
            sys.audit("trkh.balanced_bce.data_access_probe")
            if not _HOOK_PROBE_SEEN:
                raise RuntimeError("Bal-BCE data-access probe was not observed")
            _HOOK_INSTALLED = True


def _iter_external_file_records(lock: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    sources = lock.get("external_sources", {})
    if not isinstance(sources, Mapping):
        return
    for source in sources.values():
        if not isinstance(source, Mapping):
            continue
        for value in source.values():
            if (
                isinstance(value, Mapping)
                and "path" in value
                and "sha256" in value
            ):
                yield value


class DataAccessLedger:
    def __init__(
        self,
        lock: Mapping[str, object],
        *,
        lock_path: Path,
        excluded_roots: Sequence[Path] = (),
    ) -> None:
        immutable = lock.get("immutable_inputs")
        if not isinstance(immutable, Mapping):
            raise ValueError("Bal-BCE immutable-input lock is missing")
        yaml_record = immutable.get("yolo_data_yaml")
        if not isinstance(yaml_record, Mapping):
            raise ValueError("Bal-BCE yolo_data_yaml lock is missing")
        dataset_root = _resolve_locked_path(yaml_record["path"]).parent.parent
        self.domain_roots = {
            "dataset": _normalized_path(dataset_root),
            "runs": _normalized_path(REPO_ROOT / "runs"),
            "external": _normalized_path(r"D:\DataAI\external_sources"),
        }
        if any(value is None for value in self.domain_roots.values()):
            raise ValueError("Bal-BCE data-access roots are invalid")
        self.excluded_roots = tuple(
            value
            for value in (
                _normalized_path(path) for path in excluded_roots
            )
            if value is not None
        )
        allowed: set[str] = set()
        records: List[Mapping[str, object]] = [
            value
            for value in immutable.values()
            if isinstance(value, Mapping) and "path" in value
        ]
        records.extend(_iter_external_file_records(lock))
        protocol = lock.get("protocol")
        builder = lock.get("lock_builder")
        if isinstance(protocol, Mapping):
            records.append(protocol)
        if isinstance(builder, Mapping):
            records.append(builder)
        for record in records:
            normalized = _normalized_path(_resolve_locked_path(record["path"]))
            if normalized is not None:
                allowed.add(normalized)
        normalized_lock = _normalized_path(lock_path)
        normalized_sidecar = _normalized_path(lock_path.with_suffix(".sha256"))
        if normalized_lock is not None:
            allowed.add(normalized_lock)
        if normalized_sidecar is not None:
            allowed.add(normalized_sidecar)
        self.allowed_exact_paths = frozenset(allowed)
        self.visual_paths: frozenset[str] = frozenset()
        self.visual_authorized = False
        self.forbidden_components = frozenset(
            str(value).casefold()
            for value in lock["data_access_audit"][
                "forbidden_complete_components"
            ]
        )
        self.events: List[Dict[str, object]] = []
        self.blocked_attempts: List[Dict[str, object]] = []
        self._entered = False

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
        return any(_path_is_within(path, root) for root in self.excluded_roots)

    def authorize_visual_paths(self, paths: Sequence[object]) -> None:
        normalized = []
        for value in paths:
            if not _is_train_image_path(value):
                raise ValueError(f"Visual path is not train-only: {value}")
            path = _normalized_path(value)
            if path is None:
                raise ValueError(f"Visual path cannot be normalized: {value}")
            normalized.append(path)
        if len(normalized) != len(VISUAL_ANCHORS):
            raise ValueError("Exactly 20 visual paths must be authorized")
        self.visual_paths = frozenset(normalized)
        self.visual_authorized = True

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
                self.forbidden_components
            )
        )
        reason: Optional[str] = None
        if forbidden:
            reason = "forbidden_split_component"
        elif self._write_requested(mode, flags):
            reason = "write_to_data_domain"
        elif path in self.allowed_exact_paths:
            reason = None
        elif (
            domain == "dataset"
            and self.visual_authorized
            and path in self.visual_paths
        ):
            reason = None
        else:
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
            "visual_phase_authorized": bool(self.visual_authorized),
        }
        self.events.append(record)
        if reason is not None:
            self.blocked_attempts.append(dict(record))
            raise PermissionError(
                f"Bal-BCE data-access audit blocked {reason}: {path}"
            )

    def __enter__(self) -> "DataAccessLedger":
        global _ACTIVE_LEDGER
        _ensure_audit_hook()
        with _ACCESS_LOCK:
            if _ACTIVE_LEDGER is not None:
                raise RuntimeError("A data-access ledger is already active")
            _ACTIVE_LEDGER = self
            self._entered = True
        return self

    def __exit__(self, *args: object) -> None:
        global _ACTIVE_LEDGER
        with _ACCESS_LOCK:
            if _ACTIVE_LEDGER is self:
                _ACTIVE_LEDGER = None
            self._entered = False

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
        forbidden_counts = {
            name: sum(
                int(name in row["forbidden_components"]) for row in events
            )
            for name in sorted(self.forbidden_components)
        }
        return {
            "schema_version": 1,
            "hook_installed": bool(_HOOK_INSTALLED),
            "hook_probe_seen": bool(_HOOK_PROBE_SEEN),
            "domain_roots": self.domain_roots,
            "excluded_roots": list(self.excluded_roots),
            "visual_phase_authorized": bool(self.visual_authorized),
            "visual_path_count": len(self.visual_paths),
            "observed_event_count": len(events),
            "unique_path_count": len(path_counts),
            "path_counts": dict(sorted(path_counts.items())),
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


def _runtime_versions() -> Dict[str, object]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": metadata.version("numpy"),
        "torch": metadata.version("torch"),
        "torch_cuda": str(torch.version.cuda),
        "scikit_learn": metadata.version("scikit-learn"),
        "pillow": metadata.version("Pillow"),
        "psutil": metadata.version("psutil"),
        "device_name": (
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available()
            else None
        ),
    }


def _verify_runtime(lock: Mapping[str, object]) -> Dict[str, object]:
    expected = lock["runtime"]
    observed = _runtime_versions()
    checks = {
        key: observed.get(key) == expected.get(key)
        for key in expected
    }
    if not all(checks.values()):
        raise ValueError(
            f"Bal-BCE runtime differs from lock: expected={expected}, "
            f"observed={observed}"
        )
    return {"expected": expected, "observed": observed, "checks": checks, "passed": True}


def _repository_state(lock: Mapping[str, object]) -> Dict[str, object]:
    head = _run_git(REPO_ROOT, "rev-parse", "HEAD")
    upstream = _run_git(REPO_ROOT, "rev-parse", "@{upstream}")
    status_result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    status_lines = [
        row for row in status_result.stdout.split("\0") if row
    ]
    protected = [
        str(value).replace("\\", "/").rstrip("/")
        for value in lock.get("protected_untracked", [])
    ]
    tracked_dirty = [row for row in status_lines if not row.startswith("?? ")]
    untracked = [row[3:].replace("\\", "/") for row in status_lines if row.startswith("?? ")]

    def protected_match(path: str, rule: str) -> bool:
        return path == rule or path.startswith(rule + "/")

    unexpected = [
        path
        for path in untracked
        if not any(protected_match(path, rule) for rule in protected)
    ]
    missing = [
        rule
        for rule in protected
        if not any(protected_match(path, rule) for path in untracked)
    ]
    parent = str(lock["repository_lock_parent_commit"])
    ancestor_check = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", parent, head],
        check=False,
        capture_output=True,
    ).returncode == 0
    lock_commit = _run_git(
        REPO_ROOT,
        "log",
        "-1",
        "--format=%H",
        "--",
        str(LOCK_PATH.relative_to(REPO_ROOT)),
    )
    lock_pushed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", lock_commit, upstream],
        check=False,
        capture_output=True,
    ).returncode == 0
    return {
        "head": head,
        "upstream": upstream,
        "lock_parent": parent,
        "lock_commit": lock_commit,
        "tracked_dirty": tracked_dirty,
        "untracked": untracked,
        "unexpected_untracked": unexpected,
        "missing_protected_untracked": missing,
        "head_equals_upstream": head == upstream,
        "lock_parent_is_ancestor": ancestor_check,
        "lock_commit_is_pushed": lock_pushed,
        "passed": (
            head == upstream
            and not tracked_dirty
            and not unexpected
            and not missing
            and ancestor_check
            and lock_pushed
        ),
    }


def _verify_file_record(
    record: Mapping[str, object], *, label: str
) -> Dict[str, object]:
    path = _resolve_locked_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked {label}: {path}")
    observed_sha = _sha256(path)
    observed_bytes = int(path.stat().st_size)
    if observed_sha != str(record["sha256"]):
        raise ValueError(f"Locked {label} SHA mismatch")
    if "bytes" in record and observed_bytes != int(record["bytes"]):
        raise ValueError(f"Locked {label} size mismatch")
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "passed": True,
    }


def _verify_external_sources(lock: Mapping[str, object]) -> Dict[str, object]:
    source = lock["external_sources"]["livt_cvpr_2023"]
    paper = _verify_file_record(source["paper"], label="LiVT paper")
    license_file = _verify_file_record(
        source["license_file"], label="LiVT license"
    )
    loss_source = _verify_file_record(
        source["loss_source"], label="LiVT loss source"
    )
    repository = Path(str(source["official_repository"])).resolve()
    commit = _run_git(repository, "rev-parse", "HEAD")
    tree = _run_git(repository, "rev-parse", "HEAD^{tree}")
    if commit != str(source["repository_commit"]):
        raise ValueError("LiVT repository commit mismatch")
    if tree != str(source["repository_tree"]):
        raise ValueError("LiVT repository tree mismatch")
    if str(source["license"]) != "MIT":
        raise ValueError("LiVT license lock mismatch")
    return {
        "paper": paper,
        "license_file": license_file,
        "loss_source": loss_source,
        "repository": str(repository),
        "commit": commit,
        "tree": tree,
        "passed": True,
    }


def _verify_engine_contract(lock: Mapping[str, object]) -> Dict[str, bool]:
    optimization = lock["optimization"]
    optimizer = optimization["optimizer"]
    schedule = optimization["lr_schedule"]
    equation = lock["equation"]
    checks = {
        "roles_exact": tuple(lock["roles"]) == ROLE_NAMES,
        "class_count_exact": int(equation["class_count"]) == CLASS_COUNT,
        "seed_exact": int(optimization["seed"]) == BASE_SEED,
        "repeat_seed_offset_exact": (
            int(optimization["repeat_seed_offset"]) == REPEAT_SEED_OFFSET
        ),
        "epochs_exact": int(optimization["epochs"]) == EPOCHS,
        "batch_size_exact": int(optimization["batch_size"]) == BATCH_SIZE,
        "device_exact": str(optimization["device"]) == "cuda",
        "dtype_exact": str(optimization["dtype"]) == "float32",
        "determinism_exact": (
            bool(optimization["deterministic_algorithms"])
            and str(optimization["cublas_workspace_config"])
            == EXPECTED_CUBLAS_WORKSPACE_CONFIG
            and not bool(optimization["cuda_matmul_tf32"])
            and not bool(optimization["cudnn_tf32"])
        ),
        "optimizer_exact": (
            str(optimizer["name"]) == "SGD"
            and float(optimizer["initial_lr"]) == BASE_LR
            and float(optimizer["momentum"]) == MOMENTUM
            and float(optimizer["dampening"]) == 0.0
            and float(optimizer["weight_decay"]) == 0.0
            and not bool(optimizer["nesterov"])
        ),
        "schedule_exact": (
            int(schedule["warmup_epochs"]) == WARMUP_EPOCHS
            and str(schedule["post_warmup"]) == "cosine"
            and int(schedule["horizon_epochs"]) == EPOCHS
            and learning_rate(0) == 0.006
            and learning_rate(4) == BASE_LR
            and learning_rate(EPOCHS - 1) == 0.0
        ),
        "selection_disabled": (
            not bool(optimization["early_stopping"])
            and not bool(optimization["best_epoch_selection"])
            and not bool(optimization["feature_update"])
            and not bool(optimization["feature_standardization"])
            and not bool(optimization["global_class1_oversampling"])
        ),
        "raw_inference_exact": all(
            str(equation[name]["inference"]) == "argmax(raw_logits)"
            for name in (
                "balanced_bce_candidate",
                "balanced_softmax_tau025_control",
                "reversed_prior_bce_control",
            )
        ),
        "visual_anchors_exact": (
            tuple(int(value) for value in lock["visual_anchors"]["ordered_sample_indices"])
            == VISUAL_ANCHORS
            and array_sha256(np.asarray(VISUAL_ANCHORS, dtype=np.int64))
            == str(lock["visual_anchors"]["ordered_indices_sha256"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Bal-BCE engine/lock contract mismatch: {failed}")
    return checks


def verify_locked_inputs(
    lock_path: Path = LOCK_PATH,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    resolved_lock = Path(lock_path).expanduser().resolve()
    lock = json.loads(resolved_lock.read_text(encoding="utf-8"))
    if str(lock["lock_state"]) != (
        "prospective_before_candidate_implementation_training_or_metrics"
    ):
        raise ValueError("Bal-BCE lock is not prospective")
    sidecar_value = LOCK_SHA_PATH.read_text(encoding="ascii").strip().split()[0]
    observed_lock_sha = _sha256(resolved_lock)
    if sidecar_value != observed_lock_sha:
        raise ValueError("Bal-BCE lock sidecar SHA mismatch")
    protocol = _verify_file_record(lock["protocol"], label="protocol")
    builder = _verify_file_record(lock["lock_builder"], label="lock builder")
    immutable = {
        name: _verify_file_record(record, label=name)
        for name, record in lock["immutable_inputs"].items()
    }
    runtime = _verify_runtime(lock)
    external = _verify_external_sources(lock)
    engine_contract = _verify_engine_contract(lock)
    repository = _repository_state(lock)
    if not repository["passed"]:
        raise ValueError(
            "TRKH repository must be clean, pushed, and preserve protected "
            f"untracked paths: {repository}"
        )
    return lock, {
        "lock_path": str(resolved_lock),
        "lock_sha256": observed_lock_sha,
        "protocol": protocol,
        "lock_builder": builder,
        "immutable_inputs": immutable,
        "runtime": runtime,
        "external_sources": external,
        "engine_contract": engine_contract,
        "repository": repository,
        "passed": True,
    }


def load_locked_train_cache(path: Path) -> Dict[str, np.ndarray]:
    resolved = Path(path).expanduser().resolve()
    if _contains_forbidden_split(resolved):
        raise ValueError(f"Forbidden cache split marker: {resolved}")
    with np.load(resolved, allow_pickle=True) as payload:
        if tuple(payload.files) != EXPECTED_CACHE_KEYS:
            raise ValueError(f"Cache keys/order mismatch: {payload.files}")
        cache = {name: np.asarray(payload[name]).copy() for name in payload.files}
    expected = {
        "embeddings": ((EXPECTED_ROWS, FEATURE_DIM), np.dtype(np.float32)),
        "probabilities": ((EXPECTED_ROWS, CLASS_COUNT), np.dtype(np.float32)),
        "labels": ((EXPECTED_ROWS,), np.dtype(np.int64)),
        "base_predictions": ((EXPECTED_ROWS,), np.dtype(np.int64)),
        "paths": ((EXPECTED_ROWS,), np.dtype(object)),
        "sample_index": ((EXPECTED_ROWS,), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        if cache[name].shape != shape or cache[name].dtype != dtype:
            raise ValueError(
                f"Cache {name} differs: {cache[name].shape}/{cache[name].dtype}"
            )
    if not np.isfinite(cache["embeddings"]).all():
        raise ValueError("Embedding cache contains non-finite values")
    if not np.isfinite(cache["probabilities"]).all():
        raise ValueError("Keeper probability cache contains non-finite values")
    if not np.array_equal(
        cache["sample_index"], np.arange(EXPECTED_ROWS, dtype=np.int64)
    ):
        raise ValueError("Embedding cache sample order differs")
    if tuple(np.bincount(cache["labels"], minlength=CLASS_COUNT)) != EXPECTED_CLASS_COUNTS:
        raise ValueError("Embedding cache class counts differ")
    paths = np.asarray(cache["paths"]).astype(str)
    if any(not _is_train_image_path(path) for path in paths):
        raise ValueError("Embedding cache contains a non-train image path")
    return {
        "embeddings": np.asarray(cache["embeddings"], dtype=np.float32),
        "probabilities": np.asarray(cache["probabilities"], dtype=np.float32),
        "labels": np.asarray(cache["labels"], dtype=np.int64),
        "base_predictions": np.asarray(cache["base_predictions"], dtype=np.int64),
        "paths": paths,
        "sample_index": np.asarray(cache["sample_index"], dtype=np.int64),
    }


def load_locked_cidt_folds(
    cache: Mapping[str, np.ndarray],
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if _contains_forbidden_split(resolved):
        raise ValueError(f"Forbidden CIDT split marker: {resolved}")
    folds = np.full(EXPECTED_ROWS, -1, dtype=np.int64)
    sources = np.empty(EXPECTED_ROWS, dtype=object)
    observed = np.zeros(EXPECTED_ROWS, dtype=bool)
    with resolved.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row["condition"]).strip().casefold() != "clean":
                continue
            index = int(row["sample_index"])
            if index < 0 or index >= EXPECTED_ROWS or observed[index]:
                raise ValueError(f"Invalid CIDT clean sample index: {index}")
            if int(row["target_index"]) != int(cache["labels"][index]):
                raise ValueError(f"CIDT/cache label mismatch at {index}")
            image_path = str(row["image_path"])
            if not _is_train_image_path(image_path):
                raise ValueError(f"CIDT contains non-train path: {image_path}")
            if Path(image_path).name.casefold() != Path(
                str(cache["paths"][index])
            ).name.casefold():
                raise ValueError(f"CIDT/cache path mismatch at {index}")
            folds[index] = int(row["fold"])
            sources[index] = str(row["source_stem"]).casefold()
            observed[index] = True
    if not observed.all():
        raise ValueError(f"CIDT clean rows incomplete: {int(observed.sum())}")
    if tuple(np.bincount(folds, minlength=5)) != EXPECTED_FOLD_COUNTS:
        raise ValueError("CIDT fold counts differ")
    source_array = np.asarray(sources).astype(str)
    overlap_total = 0
    fold_rows = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        overlap = set(source_array[fit]).intersection(source_array[held])
        overlap_total += len(overlap)
        fold_rows.append(
            {
                "outer_fold": fold,
                "fit_rows": int(fit.size),
                "held_rows": int(held.size),
                "held_class_counts": np.bincount(
                    cache["labels"][held], minlength=CLASS_COUNT
                ).tolist(),
                "source_overlap": int(len(overlap)),
            }
        )
    if overlap_total:
        raise ValueError("CIDT folds contain source overlap")
    return folds, source_array, {
        "assignment_sha256": array_sha256(folds),
        "source_sequence_sha256": _string_sequence_sha256(source_array),
        "source_group_count": int(np.unique(source_array).size),
        "source_overlap": 0,
        "folds": fold_rows,
    }


def build_locked_fold_protocol(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    sources: np.ndarray,
    lock: Mapping[str, object],
) -> List[Dict[str, object]]:
    expected_rows = {
        int(row["outer_fold"]): row for row in lock["outer_folds"]
    }
    if set(expected_rows) != set(range(5)):
        raise ValueError("Bal-BCE lock must contain folds 0..4")
    result = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold).astype(np.int64)
        held = np.flatnonzero(folds == fold).astype(np.int64)
        fit_counts = np.bincount(labels[fit], minlength=CLASS_COUNT).astype(
            np.int64
        )
        held_counts = np.bincount(labels[held], minlength=CLASS_COUNT).astype(
            np.int64
        )
        prior = fit_counts.astype(np.float64) / float(fit_counts.sum())
        bias = np.log(prior) - np.log1p(-prior)
        softmax_bias = 0.25 * np.log(prior)
        primary_seed = BASE_SEED + fold
        repeat_seed = BASE_SEED + REPEAT_SEED_OFFSET + fold
        primary_orders = build_epoch_orders(fit, seed=primary_seed)
        repeat_orders = build_epoch_orders(fit, seed=repeat_seed)

        def order_evidence(
            orders: Sequence[np.ndarray], seed: int
        ) -> Dict[str, object]:
            return {
                "seed": int(seed),
                "per_epoch_sha256": [array_sha256(row) for row in orders],
                "all_epochs_sha256": array_sha256(np.concatenate(orders)),
                "rows_per_epoch": int(fit.size),
                "optimizer_steps": int(
                    EPOCHS * math.ceil(float(fit.size) / BATCH_SIZE)
                ),
            }

        observed = {
            "outer_fold": fold,
            "fit_rows": int(fit.size),
            "held_rows": int(held.size),
            "fit_class_counts": fit_counts.tolist(),
            "held_class_counts": held_counts.tolist(),
            "fit_prior_float64": prior.tolist(),
            "balanced_bce_bias_float64": bias.tolist(),
            "balanced_bce_bias_float32_sha256": array_sha256(
                bias.astype(np.float32)
            ),
            "balanced_softmax_tau025_bias_float64": softmax_bias.tolist(),
            "balanced_softmax_tau025_bias_float32_sha256": array_sha256(
                softmax_bias.astype(np.float32)
            ),
            "reversed_prior_bias_float64": (-bias).tolist(),
            "reversed_prior_bias_float32_sha256": array_sha256(
                (-bias).astype(np.float32)
            ),
            "fit_indices_sha256": array_sha256(fit),
            "held_indices_sha256": array_sha256(held),
            "fit_labels_sha256": array_sha256(labels[fit]),
            "held_labels_sha256": array_sha256(labels[held]),
            "fit_sources_sha256": _string_sequence_sha256(sources[fit]),
            "held_sources_sha256": _string_sequence_sha256(sources[held]),
            "source_overlap": int(
                len(set(sources[fit]).intersection(sources[held]))
            ),
            "primary_initial_head": initial_head_evidence(seed=primary_seed),
            "repeat_initial_head": initial_head_evidence(seed=repeat_seed),
            "primary_orders": order_evidence(primary_orders, primary_seed),
            "repeat_orders": order_evidence(repeat_orders, repeat_seed),
        }
        expected = expected_rows[fold]
        if set(observed) != set(expected):
            raise ValueError(f"Fold {fold} lock fields differ")
        for name, value in observed.items():
            if value != expected[name]:
                raise ValueError(
                    f"Fold {fold} lock mismatch for {name}: "
                    f"{value!r} != {expected[name]!r}"
                )
        result.append(
            {
                **observed,
                "fit_indices": fit,
                "held_indices": held,
            }
        )
    return result


def _sanitize_fold_protocol(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    return [
        {
            name: value
            for name, value in row.items()
            if name not in {"fit_indices", "held_indices"}
        }
        for row in rows
    ]


def _load_audit_inputs(
    lock: Mapping[str, object],
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    Dict[str, object],
    List[Dict[str, object]],
]:
    immutable = lock["immutable_inputs"]
    cache = load_locked_train_cache(
        _resolve_locked_path(immutable["train_embedding_cache"]["path"])
    )
    folds, sources, fold_evidence = load_locked_cidt_folds(
        cache,
        _resolve_locked_path(immutable["cidt_predictions"]["path"]),
    )
    declaration = lock["dataset_declaration"]
    checks = {
        "rows_exact": int(declaration["rows"]) == len(cache["labels"]),
        "class_counts_exact": (
            tuple(declaration["class_counts"]) == EXPECTED_CLASS_COUNTS
        ),
        "fold_counts_exact": (
            tuple(declaration["outer_fold_counts"]) == EXPECTED_FOLD_COUNTS
        ),
        "fold_assignment_exact": (
            str(declaration["outer_assignment_sha256"])
            == array_sha256(folds)
        ),
        "source_sequence_exact": (
            str(declaration["source_sequence_sha256"])
            == _string_sequence_sha256(sources)
        ),
        "source_group_count_exact": (
            int(declaration["source_group_count"])
            == int(np.unique(sources).size)
        ),
        "source_overlap_zero": int(fold_evidence["source_overlap"]) == 0,
        "visual_fold_hash_exact": (
            str(lock["visual_anchors"]["folds_sha256"])
            == array_sha256(
                folds[np.asarray(VISUAL_ANCHORS, dtype=np.int64)].astype(
                    np.int8
                )
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Bal-BCE dataset declaration mismatch: {checks}")
    fold_evidence["checks"] = checks
    protocol = build_locked_fold_protocol(
        labels=cache["labels"],
        folds=folds,
        sources=sources,
        lock=lock,
    )
    return cache, folds, sources, fold_evidence, protocol


def _configure_torch(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Bal-BCE formal A0 requires locked CUDA runtime")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _process_snapshot() -> Dict[str, object]:
    current = os.getpid()
    rows = []
    for process in psutil.process_iter(
        ["pid", "ppid", "name", "create_time", "cmdline"]
    ):
        try:
            info = process.info
            name = str(info.get("name") or "").casefold()
            if name not in {"python.exe", "pythonw.exe", "trtexec.exe"}:
                continue
            if int(info["pid"]) == current:
                continue
            rows.append(
                {
                    "pid": int(info["pid"]),
                    "ppid": int(info.get("ppid") or 0),
                    "name": str(info.get("name") or ""),
                    "create_time": float(info.get("create_time") or 0.0),
                    "cmdline": [str(value) for value in (info.get("cmdline") or [])],
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return {
        "current_pid": current,
        "unexpected_processes": sorted(rows, key=lambda row: row["pid"]),
    }


class PeakResourceMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_rss_bytes = int(self.process.memory_info().rss)
        self.peak_virtual_memory_fraction = float(
            psutil.virtual_memory().percent / 100.0
        )
        self.unexpected_processes: Dict[int, Dict[str, object]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    int(self.process.memory_info().rss),
                )
                self.peak_virtual_memory_fraction = max(
                    self.peak_virtual_memory_fraction,
                    float(psutil.virtual_memory().percent / 100.0),
                )
                for row in _process_snapshot()["unexpected_processes"]:
                    self.unexpected_processes[int(row["pid"])] = row
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return

    def __enter__(self) -> "PeakResourceMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, int(self.process.memory_info().rss)
        )


def train_all_roles(
    *,
    cache: Mapping[str, np.ndarray],
    fold_protocol: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[Tuple[str, int], Dict[str, object]],
]:
    features = torch.as_tensor(
        cache["embeddings"], dtype=torch.float32, device=device
    )
    targets = torch.as_tensor(
        cache["labels"], dtype=torch.long, device=device
    )
    logits = {
        role: np.full(
            (EXPECTED_ROWS, CLASS_COUNT), np.nan, dtype=np.float32
        )
        for role in ROLE_NAMES
    }
    probabilities = {
        role: np.full(
            (EXPECTED_ROWS, CLASS_COUNT), np.nan, dtype=np.float32
        )
        for role in ROLE_NAMES
    }
    results: Dict[Tuple[str, int], Dict[str, object]] = {}
    for fold_row in fold_protocol:
        fold = int(fold_row["outer_fold"])
        fit = np.asarray(fold_row["fit_indices"], dtype=np.int64)
        held = np.asarray(fold_row["held_indices"], dtype=np.int64)
        for role in ROLE_NAMES:
            print(
                f"Bal-BCE A0 training role={role} fold={fold} "
                f"fit={fit.size} held={held.size}",
                flush=True,
            )
            result = train_role_fold(
                role=role,
                fold=fold,
                features=features,
                targets=targets,
                fit_indices=fit,
                held_indices=held,
                prior_float64=fold_row["fit_prior_float64"],
                device=device,
            )
            expected_initial = (
                fold_row["repeat_initial_head"]
                if role == "balanced_bce_seed_repeat"
                else fold_row["primary_initial_head"]
            )
            if result["initial_head"] != expected_initial:
                raise ValueError(
                    f"Initial state drift role={role} fold={fold}"
                )
            expected_orders = (
                fold_row["repeat_orders"]
                if role == "balanced_bce_seed_repeat"
                else fold_row["primary_orders"]
            )
            if result["order_hashes"] != expected_orders["per_epoch_sha256"]:
                raise ValueError(f"Order hash drift role={role} fold={fold}")
            if result["all_orders_sha256"] != expected_orders["all_epochs_sha256"]:
                raise ValueError(
                    f"All-order hash drift role={role} fold={fold}"
                )
            if int(result["update_count"]) != int(expected_orders["optimizer_steps"]):
                raise ValueError(
                    f"Optimizer-step drift role={role} fold={fold}"
                )
            logits[role][held] = np.asarray(result["raw_logits"], dtype=np.float32)
            probabilities[role][held] = np.asarray(
                result["probabilities"], dtype=np.float32
            )
            results[(role, fold)] = result
    if any(not np.isfinite(value).all() for value in logits.values()):
        raise FloatingPointError("OOF logits are incomplete or non-finite")
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise FloatingPointError("OOF probabilities are incomplete or non-finite")
    return logits, probabilities, results


def _ece(
    targets: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        if index == 0:
            mask = (confidence >= edges[index]) & (
                confidence <= edges[index + 1]
            )
        else:
            mask = (confidence > edges[index]) & (
                confidence <= edges[index + 1]
            )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(result)


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (targets.size, CLASS_COUNT):
        raise ValueError("Probability shape does not match targets")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(row_sums <= 0.0):
        raise ValueError("Probabilities are invalid")
    probabilities = probabilities / row_sums
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(CLASS_COUNT),
        zero_division=0,
    )
    per_class = []
    for class_index in range(CLASS_COUNT):
        tp = int(
            np.sum((targets == class_index) & (predictions == class_index))
        )
        fp = int(
            np.sum((targets != class_index) & (predictions == class_index))
        )
        fn = int(
            np.sum((targets == class_index) & (predictions != class_index))
        )
        per_class.append(
            {
                "class_index": class_index,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
                "predicted_support": int(tp + fp),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    restricted = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
    return {
        "samples": int(targets.size),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=np.arange(CLASS_COUNT)
        ).tolist(),
        "nll": float(
            log_loss(targets, probabilities, labels=np.arange(CLASS_COUNT))
        ),
        "ece_15": _ece(targets, probabilities, bins=15),
        "class1_tp": int(
            np.sum((targets == FOCUS_CLASS) & (predictions == FOCUS_CLASS))
        ),
        "class1_fp": int(
            np.sum((targets != FOCUS_CLASS) & (predictions == FOCUS_CLASS))
        ),
        "class1_fn": int(
            np.sum((targets == FOCUS_CLASS) & (predictions != FOCUS_CLASS))
        ),
        "restricted_fp": int(
            np.sum(restricted & (predictions == FOCUS_CLASS))
        ),
    }


def transition_stats(
    targets: np.ndarray,
    base_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> Dict[str, int]:
    targets = np.asarray(targets, dtype=np.int64)
    base = np.asarray(base_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    restricted = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
    base_tp = (targets == FOCUS_CLASS) & (base == FOCUS_CLASS)
    candidate_tp = (targets == FOCUS_CLASS) & (candidate == FOCUS_CLASS)
    base_fp = restricted & (base == FOCUS_CLASS)
    candidate_fp = restricted & (candidate == FOCUS_CLASS)
    return {
        "changed": int(np.sum(base != candidate)),
        "corrections": int(np.sum((base != targets) & (candidate == targets))),
        "harms": int(np.sum((base == targets) & (candidate != targets))),
        "class1_fn_rescues": int(
            np.sum(
                (targets == FOCUS_CLASS)
                & (base != FOCUS_CLASS)
                & (candidate == FOCUS_CLASS)
            )
        ),
        "class1_tp_breaks": int(
            np.sum(base_tp & (candidate != FOCUS_CLASS))
        ),
        "class1_tp_base": int(base_tp.sum()),
        "class1_tp_candidate": int(candidate_tp.sum()),
        "class1_tp_net": int(candidate_tp.sum() - base_tp.sum()),
        "restricted_fp_base": int(base_fp.sum()),
        "restricted_fp_candidate": int(candidate_fp.sum()),
        "restricted_fp_removals": int(
            np.sum(base_fp & (candidate != FOCUS_CLASS))
        ),
        "restricted_fp_creations": int(np.sum((~base_fp) & candidate_fp)),
        "restricted_fp_net_removal": int(base_fp.sum() - candidate_fp.sum()),
    }


def analyze_predictions(
    *,
    targets: np.ndarray,
    folds: np.ndarray,
    logits: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    if set(logits) != set(ROLE_NAMES) or set(probabilities) != set(ROLE_NAMES):
        raise ValueError("Bal-BCE analysis roles differ from lock")
    metrics = {
        role: classification_metrics(targets, probabilities[role])
        for role in ROLE_NAMES
    }
    fold_metrics: Dict[str, List[Dict[str, object]]] = {}
    for role in ROLE_NAMES:
        fold_metrics[role] = [
            {
                "outer_fold": fold,
                **classification_metrics(
                    targets[folds == fold],
                    probabilities[role][folds == fold],
                ),
            }
            for fold in range(5)
        ]
    transitions = {
        f"{role}_vs_ce": transition_stats(
            targets, probabilities["ce_control"], probabilities[role]
        )
        for role in ROLE_NAMES
        if role != "ce_control"
    }
    for role in (
        "plain_bce_control",
        "balanced_softmax_tau025_control",
        "reversed_prior_bce_control",
    ):
        transitions[f"candidate_vs_{role}"] = transition_stats(
            targets,
            probabilities[role],
            probabilities["balanced_bce_candidate"],
        )
    candidate_actions = probabilities["balanced_bce_candidate"].argmax(axis=1)
    repeat_actions = probabilities["balanced_bce_seed_repeat"].argmax(axis=1)
    return {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "transitions": transitions,
        "mean_raw_class1_logit": {
            role: float(np.mean(logits[role][:, FOCUS_CLASS]))
            for role in ROLE_NAMES
        },
        "candidate_repeat_prediction_agreement": float(
            np.mean(candidate_actions == repeat_actions)
        ),
    }


def _focus(metrics: Mapping[str, object]) -> Mapping[str, object]:
    value = metrics["per_class"][FOCUS_CLASS]
    assert isinstance(value, Mapping)
    return value


def _role_vs_ce_safety(
    *,
    role: str,
    analysis: Mapping[str, object],
    lock: Mapping[str, object],
) -> Dict[str, bool]:
    metrics = analysis["metrics"]
    fold_metrics = analysis["fold_metrics"]
    transitions = analysis["transitions"]
    ce = metrics["ce_control"]
    candidate = metrics[role]
    ce_focus = _focus(ce)
    focus = _focus(candidate)
    transition = transitions[f"{role}_vs_ce"]
    thresholds = lock["gates"]["candidate_vs_ce"]
    fold_thresholds = lock["gates"]["fold_stability"]
    nonfocus_loss = max(
        float(ce["per_class"][index]["f1"])
        - float(candidate["per_class"][index]["f1"])
        for index in range(CLASS_COUNT)
        if index != FOCUS_CLASS
    )
    checks = {
        "macro_f1_gain": (
            float(candidate["macro_f1"]) - float(ce["macro_f1"])
            >= float(thresholds["macro_f1_gain_min"])
        ),
        "class1_precision_gain": (
            float(focus["precision"]) - float(ce_focus["precision"])
            >= float(thresholds["class1_precision_gain_min"])
        ),
        "class1_f1_gain": (
            float(focus["f1"]) - float(ce_focus["f1"])
            >= float(thresholds["class1_f1_gain_min"])
        ),
        "class1_recall_budget": (
            float(ce_focus["recall"]) - float(focus["recall"])
            <= float(thresholds["class1_recall_loss_max"])
        ),
        "class1_tp_net": (
            int(transition["class1_tp_net"])
            >= int(thresholds["class1_tp_net_min"])
        ),
        "restricted_fp_net_removal": (
            int(transition["restricted_fp_net_removal"])
            >= int(thresholds["restricted_fp_net_removal_min"])
        ),
        "corrections_ge_harms": (
            int(transition["corrections"]) >= int(transition["harms"])
        ),
        "nonfocus_f1_safety": (
            nonfocus_loss <= float(thresholds["maximum_nonfocus_f1_loss"])
        ),
        "nll_safety": (
            float(candidate["nll"]) - float(ce["nll"])
            <= float(thresholds["nll_increase_max"])
        ),
        "ece_safety": (
            float(candidate["ece_15"]) - float(ce["ece_15"])
            <= float(thresholds["ece_increase_max"])
        ),
    }
    ce_folds = fold_metrics["ce_control"]
    role_folds = fold_metrics[role]
    precision_nonworse = 0
    f1_improved = 0
    fold_recall_safe = True
    fold_tp_safe = True
    fold_macro_safe = True
    for ce_fold, role_fold in zip(ce_folds, role_folds):
        ce_fold_focus = _focus(ce_fold)
        role_fold_focus = _focus(role_fold)
        precision_nonworse += int(
            float(role_fold_focus["precision"])
            >= float(ce_fold_focus["precision"])
        )
        f1_improved += int(
            float(role_fold_focus["f1"]) > float(ce_fold_focus["f1"])
        )
        fold_recall_safe = fold_recall_safe and (
            float(ce_fold_focus["recall"])
            - float(role_fold_focus["recall"])
            <= float(fold_thresholds["per_fold_recall_loss_max"])
        )
        fold_tp_safe = fold_tp_safe and (
            int(role_fold["class1_tp"]) - int(ce_fold["class1_tp"])
            >= int(fold_thresholds["per_fold_class1_tp_net_min"])
        )
        fold_macro_safe = fold_macro_safe and (
            float(ce_fold["macro_f1"]) - float(role_fold["macro_f1"])
            <= float(fold_thresholds["per_fold_macro_f1_loss_max"])
        )
    checks.update(
        {
            "fold_precision_nonworse": (
                precision_nonworse
                >= int(fold_thresholds["precision_nonworse_fold_min"])
            ),
            "fold_class1_f1_improved": (
                f1_improved
                >= int(fold_thresholds["class1_f1_improved_fold_min"])
            ),
            "fold_recall_safety": bool(fold_recall_safe),
            "fold_tp_safety": bool(fold_tp_safe),
            "fold_macro_safety": bool(fold_macro_safe),
        }
    )
    return checks


def assess_performance_gates(
    analysis: Mapping[str, object],
    lock: Mapping[str, object],
) -> Dict[str, object]:
    metrics = analysis["metrics"]
    transitions = analysis["transitions"]
    ce = metrics["ce_control"]
    candidate = metrics["balanced_bce_candidate"]
    repeat = metrics["balanced_bce_seed_repeat"]
    plain = metrics["plain_bce_control"]
    balanced_softmax = metrics["balanced_softmax_tau025_control"]
    reversed_prior = metrics["reversed_prior_bce_control"]
    ce_focus = _focus(ce)
    candidate_focus = _focus(candidate)
    repeat_focus = _focus(repeat)
    plain_focus = _focus(plain)
    balanced_softmax_focus = _focus(balanced_softmax)
    reversed_focus = _focus(reversed_prior)
    compatibility_lock = lock["gates"]["control_compatibility"]
    compatibility = {
        "ce_macro_f1": (
            float(ce["macro_f1"])
            >= float(compatibility_lock["ce_macro_f1_min"])
        ),
        "ce_class1_f1": (
            float(ce_focus["f1"])
            >= float(compatibility_lock["ce_class1_f1_min"])
        ),
    }
    candidate_safety = _role_vs_ce_safety(
        role="balanced_bce_candidate",
        analysis=analysis,
        lock=lock,
    )
    repeat_safety = _role_vs_ce_safety(
        role="balanced_bce_seed_repeat",
        analysis=analysis,
        lock=lock,
    )
    plain_lock = lock["gates"]["candidate_vs_plain_bce"]
    candidate_vs_plain_transition = transitions[
        "candidate_vs_plain_bce_control"
    ]
    candidate_vs_plain = {
        "macro_f1_safety": (
            float(plain["macro_f1"]) - float(candidate["macro_f1"])
            <= float(plain_lock["macro_f1_loss_max"])
        ),
        "class1_precision_gain": (
            float(candidate_focus["precision"]) - float(plain_focus["precision"])
            >= float(plain_lock["class1_precision_gain_min"])
        ),
        "class1_f1_gain": (
            float(candidate_focus["f1"]) - float(plain_focus["f1"])
            >= float(plain_lock["class1_f1_gain_min"])
        ),
        "class1_tp_net": (
            int(candidate_vs_plain_transition["class1_tp_net"])
            >= int(plain_lock["class1_tp_net_min"])
        ),
        "restricted_fp_net_removal": (
            int(candidate_vs_plain_transition["restricted_fp_net_removal"])
            >= int(plain_lock["restricted_fp_net_removal_min"])
        ),
    }
    controls_lock = lock["gates"]["candidate_vs_controls"]
    candidate_actions = np.asarray(
        analysis["_candidate_actions"], dtype=np.int64
    )
    reversed_actions = np.asarray(
        analysis["_reversed_actions"], dtype=np.int64
    )
    candidate_vs_controls = {
        "macro_gain_vs_balanced_softmax": (
            float(candidate["macro_f1"]) - float(balanced_softmax["macro_f1"])
            >= float(
                controls_lock[
                    "macro_f1_gain_vs_balanced_softmax_min"
                ]
            )
        ),
        "class1_f1_gain_vs_balanced_softmax": (
            float(candidate_focus["f1"])
            - float(balanced_softmax_focus["f1"])
            >= float(
                controls_lock[
                    "class1_f1_gain_vs_balanced_softmax_min"
                ]
            )
        ),
        "macro_gain_vs_reversed": (
            float(candidate["macro_f1"]) - float(reversed_prior["macro_f1"])
            >= float(controls_lock["macro_f1_gain_vs_reversed_prior_min"])
        ),
        "class1_f1_gain_vs_reversed": (
            float(candidate_focus["f1"]) - float(reversed_focus["f1"])
            >= float(
                controls_lock["class1_f1_gain_vs_reversed_prior_min"]
            )
        ),
        "mean_raw_class1_logit_delta_vs_plain": (
            abs(
                float(
                    analysis["mean_raw_class1_logit"][
                        "balanced_bce_candidate"
                    ]
                )
                - float(
                    analysis["mean_raw_class1_logit"]["plain_bce_control"]
                )
            )
            >= float(
                controls_lock[
                    "mean_raw_class1_logit_delta_vs_plain_abs_min"
                ]
            )
        ),
        "prediction_array_differs_from_reversed": (
            not np.array_equal(candidate_actions, reversed_actions)
        ),
    }
    repeat_lock = lock["gates"]["repeat"]
    repeat_gate = {
        "repeat_primary_safety": all(repeat_safety.values()),
        "prediction_agreement": (
            float(analysis["candidate_repeat_prediction_agreement"])
            >= float(repeat_lock["prediction_agreement_min"])
        ),
        "macro_f1_difference": (
            abs(float(candidate["macro_f1"]) - float(repeat["macro_f1"]))
            <= float(repeat_lock["macro_f1_absolute_difference_max"])
        ),
        "class1_f1_difference": (
            abs(float(candidate_focus["f1"]) - float(repeat_focus["f1"]))
            <= float(repeat_lock["class1_f1_absolute_difference_max"])
        ),
        "class1_precision_difference": (
            abs(
                float(candidate_focus["precision"])
                - float(repeat_focus["precision"])
            )
            <= float(
                repeat_lock["class1_precision_absolute_difference_max"]
            )
        ),
        "class1_recall_difference": (
            abs(
                float(candidate_focus["recall"])
                - float(repeat_focus["recall"])
            )
            <= float(repeat_lock["class1_recall_absolute_difference_max"])
        ),
    }
    groups = {
        "control_compatibility": compatibility,
        "candidate_vs_ce_and_fold": candidate_safety,
        "candidate_vs_plain_bce": candidate_vs_plain,
        "candidate_vs_historical_and_sign_controls": candidate_vs_controls,
        "seed_repeat": repeat_gate,
    }
    flattened = [
        bool(value) for group in groups.values() for value in group.values()
    ]
    return {
        "groups": groups,
        "passed": int(sum(flattened)),
        "total": int(len(flattened)),
        "performance_pass": all(flattened),
    }


def _analysis_for_serialization(
    analysis: Mapping[str, object],
) -> Dict[str, object]:
    return {
        name: value
        for name, value in analysis.items()
        if not str(name).startswith("_")
    }


def _flatten_role_states(
    results: Mapping[Tuple[str, int], Mapping[str, object]]
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for (role, fold), result in sorted(results.items()):
        state = result["state"]
        assert isinstance(state, Mapping)
        for name, value in state.items():
            arrays[f"{role}__fold{fold}__{name}"] = np.asarray(value)
    return arrays


def _training_trace_rows(
    results: Mapping[Tuple[str, int], Mapping[str, object]]
) -> List[Dict[str, object]]:
    rows = []
    for key in sorted(results):
        rows.extend(results[key]["trace"])
    return rows


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_visual_evidence(
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    sources: np.ndarray,
    logits: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    results: Mapping[Tuple[str, int], Mapping[str, object]],
    ledger: DataAccessLedger,
    contact_sheet_path: Optional[Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    anchors = np.asarray(VISUAL_ANCHORS, dtype=np.int64)
    paths = [str(cache["paths"][index]) for index in anchors]
    ledger.authorize_visual_paths(paths)
    thumbnails = []
    metadata_rows = []
    candidate_contributions = []
    candidate_intercepts = []
    for sample_index, path in zip(anchors.tolist(), paths):
        with Image.open(path) as image:
            rgb = ImageOps.fit(
                image.convert("RGB"),
                (180, 180),
                method=Image.Resampling.LANCZOS,
            )
        thumbnail = np.asarray(rgb, dtype=np.uint8).copy()
        thumbnails.append(thumbnail)
        fold = int(folds[sample_index])
        candidate_state = results[
            ("balanced_bce_candidate", fold)
        ]["state"]
        weight = np.asarray(candidate_state["weight"], dtype=np.float32)
        bias = np.asarray(candidate_state["bias"], dtype=np.float32)
        contribution = float(
            np.dot(
                weight[FOCUS_CLASS],
                np.asarray(cache["embeddings"][sample_index], dtype=np.float32),
            )
        )
        intercept = float(bias[FOCUS_CLASS])
        candidate_contributions.append(contribution)
        candidate_intercepts.append(intercept)
        roles = {
            role: {
                "prediction": int(probabilities[role][sample_index].argmax()),
                "class1_probability": float(
                    probabilities[role][sample_index, FOCUS_CLASS]
                ),
                "raw_class1_logit": float(
                    logits[role][sample_index, FOCUS_CLASS]
                ),
            }
            for role in ROLE_NAMES
        }
        metadata_rows.append(
            {
                "sample_index": sample_index,
                "path": path,
                "source_stem": str(sources[sample_index]),
                "target": int(cache["labels"][sample_index]),
                "outer_fold": fold,
                "candidate_class1_feature_contribution": contribution,
                "candidate_class1_intercept": intercept,
                "roles": roles,
            }
        )
    thumbnail_array = np.stack(thumbnails).astype(np.uint8, copy=False)
    arrays = {
        "sample_index": anchors,
        "targets": np.asarray(cache["labels"][anchors], dtype=np.int64),
        "folds": np.asarray(folds[anchors], dtype=np.int64),
        "thumbnails_rgb": thumbnail_array,
        "candidate_class1_feature_contribution": np.asarray(
            candidate_contributions, dtype=np.float32
        ),
        "candidate_class1_intercept": np.asarray(
            candidate_intercepts, dtype=np.float32
        ),
    }
    for role in ROLE_NAMES:
        arrays[f"{role}__probabilities"] = np.asarray(
            probabilities[role][anchors], dtype=np.float32
        )
        arrays[f"{role}__predictions"] = np.asarray(
            probabilities[role][anchors].argmax(axis=1), dtype=np.int64
        )
    metadata_payload = {
        "selection_is_metric_independent": True,
        "row_count": len(VISUAL_ANCHORS),
        "ordered_indices_sha256": array_sha256(anchors),
        "train_only": True,
        "non_train_source_count": int(
            sum(not _is_train_image_path(path) for path in paths)
        ),
        "pixels_opened_after_probabilities_frozen": True,
        "rows": metadata_rows,
    }
    if contact_sheet_path is not None:
        width = 1900
        row_height = 205
        header_height = 70
        canvas = Image.new(
            "RGB",
            (width, header_height + row_height * len(metadata_rows)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        title_font = _font(26)
        body_font = _font(18)
        small_font = _font(16)
        draw.text(
            (20, 16),
            "TRKH Bal-BCE A0 | fixed train-only visual anchors",
            fill="#111827",
            font=title_font,
        )
        short_roles = {
            "ce_control": "CE",
            "plain_bce_control": "BCE",
            "balanced_bce_candidate": "Bal",
            "balanced_bce_seed_repeat": "Rep",
            "balanced_softmax_tau025_control": "BS",
            "reversed_prior_bce_control": "Rev",
        }
        for row_index, row in enumerate(metadata_rows):
            top = header_height + row_index * row_height
            if row_index % 2:
                draw.rectangle(
                    (0, top, width, top + row_height), fill="#f7f9fc"
                )
            canvas.paste(
                Image.fromarray(thumbnail_array[row_index]),
                (15, top + 12),
            )
            draw.text(
                (215, top + 15),
                (
                    f"idx {row['sample_index']} | y={row['target']} | "
                    f"fold={row['outer_fold']} | {row['source_stem']}"
                ),
                fill="#111827",
                font=body_font,
            )
            role_parts = []
            for role in ROLE_NAMES:
                role_data = row["roles"][role]
                role_parts.append(
                    f"{short_roles[role]} p={role_data['prediction']} "
                    f"p1={role_data['class1_probability']:.4f}"
                )
            draw.text(
                (215, top + 62),
                " | ".join(role_parts[:3]),
                fill="#1f2937",
                font=small_font,
            )
            draw.text(
                (215, top + 96),
                " | ".join(role_parts[3:]),
                fill="#1f2937",
                font=small_font,
            )
            draw.text(
                (215, top + 136),
                (
                    "Bal-BCE raw class-1 = feature "
                    f"{row['candidate_class1_feature_contribution']:.5f} "
                    f"+ intercept {row['candidate_class1_intercept']:.5f}"
                ),
                fill="#374151",
                font=small_font,
            )
        canvas.save(contact_sheet_path, format="PNG", optimize=False)
    return arrays, metadata_payload


def _write_fold_metrics(
    path: Path,
    analysis: Mapping[str, object],
) -> None:
    fields = [
        "role",
        "outer_fold",
        "samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "nll",
        "ece_15",
        "class1_precision",
        "class1_recall",
        "class1_f1",
        "class1_tp",
        "class1_fp",
        "class1_fn",
        "restricted_fp",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for role in ROLE_NAMES:
            for row in analysis["fold_metrics"][role]:
                focus = _focus(row)
                writer.writerow(
                    {
                        "role": role,
                        "outer_fold": row["outer_fold"],
                        "samples": row["samples"],
                        "accuracy": row["accuracy"],
                        "macro_f1": row["macro_f1"],
                        "weighted_f1": row["weighted_f1"],
                        "nll": row["nll"],
                        "ece_15": row["ece_15"],
                        "class1_precision": focus["precision"],
                        "class1_recall": focus["recall"],
                        "class1_f1": focus["f1"],
                        "class1_tp": row["class1_tp"],
                        "class1_fp": row["class1_fp"],
                        "class1_fn": row["class1_fn"],
                        "restricted_fp": row["restricted_fp"],
                    }
                )


def _write_oof_rows(
    path: Path,
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    sources: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> None:
    fields = [
        "sample_index",
        "target",
        "outer_fold",
        "source_stem",
        "image_path",
    ]
    for role in ROLE_NAMES:
        fields.extend(
            [
                f"{role}__prediction",
                f"{role}__class1_probability",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(EXPECTED_ROWS):
            row: Dict[str, object] = {
                "sample_index": index,
                "target": int(cache["labels"][index]),
                "outer_fold": int(folds[index]),
                "source_stem": str(sources[index]),
                "image_path": str(cache["paths"][index]),
            }
            for role in ROLE_NAMES:
                row[f"{role}__prediction"] = int(
                    probabilities[role][index].argmax()
                )
                row[f"{role}__class1_probability"] = format(
                    float(probabilities[role][index, FOCUS_CLASS]), ".9g"
                )
            writer.writerow(row)


def _manifest_payload(
    output: Path,
    *,
    manifest_name: str,
    exclude: Sequence[str] = (),
) -> Dict[str, object]:
    excluded = {manifest_name, *exclude}
    files = []
    for path in sorted(
        (value for value in output.iterdir() if value.is_file()),
        key=lambda value: value.name,
    ):
        if path.name in excluded:
            continue
        files.append(
            {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload: Dict[str, object] = {
        "schema_version": 1,
        "files": files,
    }
    payload["manifest_sha256"] = json_sha256(payload)
    return payload


def _write_manifest(
    output: Path,
    *,
    manifest_name: str,
    exclude: Sequence[str] = (),
) -> Dict[str, object]:
    payload = _manifest_payload(
        output, manifest_name=manifest_name, exclude=exclude
    )
    _write_json(output / manifest_name, payload)
    return payload


def _verify_manifest(output: Path, manifest_name: str) -> Dict[str, object]:
    path = output / manifest_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(payload["manifest_sha256"])
    body = {name: value for name, value in payload.items() if name != "manifest_sha256"}
    body["manifest_sha256"] = json_sha256(body)
    checks = {
        "manifest_hash_exact": body["manifest_sha256"] == expected_hash,
        "files_exact": True,
    }
    for record in payload["files"]:
        candidate = output / str(record["path"])
        if (
            not candidate.is_file()
            or int(candidate.stat().st_size) != int(record["bytes"])
            or _sha256(candidate) != str(record["sha256"])
        ):
            checks["files_exact"] = False
            break
    return {
        "manifest_sha256": expected_hash,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _post_run_locked_hashes(
    lock: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    return {
        name: _verify_file_record(record, label=f"post-run {name}")
        for name, record in lock["immutable_inputs"].items()
    }


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_preflight(lock_path: Path = LOCK_PATH) -> Dict[str, object]:
    resolved_lock = Path(lock_path).expanduser().resolve()
    preview = json.loads(resolved_lock.read_text(encoding="utf-8"))
    ledger = DataAccessLedger(preview, lock_path=resolved_lock)
    with ledger:
        lock, locked_inputs = verify_locked_inputs(resolved_lock)
        cache, folds, sources, fold_evidence, fold_protocol = _load_audit_inputs(lock)
        oracles = equation_oracles()
        if not oracles["passed"]:
            raise ValueError(f"Bal-BCE equation oracle failed: {oracles}")
        access = ledger.snapshot()
    return {
        "mode": "balanced_bce_frozen_embedding_a0_preflight",
        "locked_inputs": locked_inputs,
        "dataset": {
            "rows": int(len(cache["labels"])),
            "folds_sha256": array_sha256(folds),
            "sources_sha256": _string_sequence_sha256(sources),
        },
        "fold_evidence": fold_evidence,
        "fold_protocol_rows": len(fold_protocol),
        "equation_oracles": oracles,
        "data_access_ledger": access,
        "candidate_training_called": False,
        "candidate_metric_created": False,
        "passed": bool(locked_inputs["passed"] and access["passed"]),
    }


def run_formal(
    *,
    lock_path: Path,
    output_dir: Path,
    device_name: str,
) -> Dict[str, object]:
    resolved_lock = Path(lock_path).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    preview = json.loads(resolved_lock.read_text(encoding="utf-8"))
    ledger = DataAccessLedger(
        preview,
        lock_path=resolved_lock,
        excluded_roots=(resolved_output,),
    )
    with ledger:
        return _run_formal_with_ledger(
            lock_path=resolved_lock,
            output_dir=resolved_output,
            device_name=device_name,
            ledger=ledger,
        )


def _run_formal_with_ledger(
    *,
    lock_path: Path,
    output_dir: Path,
    device_name: str,
    ledger: DataAccessLedger,
) -> Dict[str, object]:
    lock, locked_inputs = verify_locked_inputs(lock_path)
    if str(device_name) != str(lock["optimization"]["device"]):
        raise ValueError("Formal device differs from machine lock")
    cache, folds, sources, fold_evidence, fold_protocol = _load_audit_inputs(lock)
    oracles = equation_oracles()
    if not oracles["passed"]:
        raise ValueError(f"Bal-BCE equation oracle failed: {oracles}")
    process_before = _process_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "Unrelated Python/TensorRT process exists before Bal-BCE formal"
        )
    start_available_gib = float(psutil.virtual_memory().available / (1024**3))
    output = _prepare_output_dir(output_dir)
    device = torch.device(device_name)
    _configure_torch(device)
    started = time.perf_counter()
    with PeakResourceMonitor() as resource:
        logits, probabilities, results = train_all_roles(
            cache=cache,
            fold_protocol=fold_protocol,
            device=device,
        )
        analysis = analyze_predictions(
            targets=cache["labels"],
            folds=folds,
            logits=logits,
            probabilities=probabilities,
        )
        analysis["_candidate_actions"] = probabilities[
            "balanced_bce_candidate"
        ].argmax(axis=1)
        analysis["_reversed_actions"] = probabilities[
            "reversed_prior_bce_control"
        ].argmax(axis=1)
        performance_gate = assess_performance_gates(analysis, lock)
        visual_arrays, visual_metadata = build_visual_evidence(
            cache=cache,
            folds=folds,
            sources=sources,
            logits=logits,
            probabilities=probabilities,
            results=results,
            ledger=ledger,
            contact_sheet_path=output / "fixed_visual_contact_sheet.png",
        )
    torch.cuda.synchronize(device)
    elapsed_seconds = float(time.perf_counter() - started)
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    post_run_hashes = _post_run_locked_hashes(lock)
    process_after = _process_snapshot()
    repository_after = _repository_state(lock)
    access = ledger.snapshot()
    resource_lock = lock["gates"]["resources"]
    structure_checks = {
        "locked_inputs_verified": bool(locked_inputs["passed"]),
        "repository_clean_pushed_protected": bool(repository_after["passed"]),
        "train_rows_exact": int(len(cache["labels"])) == EXPECTED_ROWS,
        "class_counts_exact": (
            tuple(np.bincount(cache["labels"], minlength=CLASS_COUNT))
            == EXPECTED_CLASS_COUNTS
        ),
        "outer_folds_exact": (
            tuple(np.bincount(folds, minlength=5)) == EXPECTED_FOLD_COUNTS
        ),
        "outer_source_overlap_zero": int(fold_evidence["source_overlap"]) == 0,
        "fold_protocol_exact": len(fold_protocol) == 5,
        "role_fold_count_exact": len(results) == len(ROLE_NAMES) * 5,
        "oof_logits_complete": all(
            value.shape == (EXPECTED_ROWS, CLASS_COUNT)
            and np.isfinite(value).all()
            for value in logits.values()
        ),
        "oof_probabilities_complete": all(
            value.shape == (EXPECTED_ROWS, CLASS_COUNT)
            and np.isfinite(value).all()
            for value in probabilities.values()
        ),
        "all_scored_once_at_epoch_30": all(
            int(result["held_score_count"]) == 1
            and int(result["score_epoch"]) == EPOCHS
            for result in results.values()
        ),
        "equation_oracles_pass": bool(oracles["passed"]),
        "dynamic_hook_installed": bool(access["hook_installed"]),
        "dynamic_hook_probe_seen": bool(access["hook_probe_seen"]),
        "data_access_pass": bool(access["passed"]),
        "data_access_blocked_zero": int(access["blocked_attempt_count"]) == 0,
        "validation_access_zero": int(access["validation_open_count"]) == 0,
        "test_access_zero": int(access["test_open_count"]) == 0,
        "visual_rows_exact": int(visual_metadata["row_count"]) == 20,
        "visual_train_only": (
            int(visual_metadata["non_train_source_count"]) == 0
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config_exact": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ),
        "tf32_disabled": (
            not bool(torch.backends.cuda.matmul.allow_tf32)
            and not bool(torch.backends.cudnn.allow_tf32)
        ),
        "start_available_ram_gate": (
            start_available_gib
            >= float(resource_lock["start_available_physical_ram_gib_min"])
        ),
        "peak_process_rss_gate": (
            float(resource.peak_rss_bytes / (1024**3))
            <= float(resource_lock["peak_process_rss_gib_max"])
        ),
        "peak_cuda_gate": (
            float(peak_cuda_bytes / (1024**3))
            <= float(resource_lock["peak_cuda_allocated_gib_max"])
        ),
        "elapsed_gate": (
            elapsed_seconds / 60.0
            <= float(resource_lock["formal_elapsed_minutes_max"])
        ),
        "no_unrelated_process_before": not bool(
            process_before["unexpected_processes"]
        ),
        "no_unrelated_process_during": not bool(
            resource.unexpected_processes
        ),
        "no_unrelated_process_after": not bool(
            process_after["unexpected_processes"]
        ),
        "immutable_inputs_unchanged": all(
            bool(value["passed"]) for value in post_run_hashes.values()
        ),
    }
    structural_pass = all(structure_checks.values())
    automatic_pass = bool(
        structural_pass and performance_gate["performance_pass"]
    )

    fold_payload = {
        "fold_evidence": fold_evidence,
        "outer_folds": _sanitize_fold_protocol(fold_protocol),
        "hash_encoding": lock["hash_encoding"],
    }
    _write_json(output / "fold_protocol.json", fold_payload)
    _write_jsonl(
        output / "training_trace.jsonl",
        _training_trace_rows(results),
    )
    np.savez_compressed(
        output / "oof_probabilities.npz",
        sample_index=np.arange(EXPECTED_ROWS, dtype=np.int64),
        targets=np.asarray(cache["labels"], dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
        **{
            f"{role}__raw_logits": np.asarray(logits[role], dtype=np.float32)
            for role in ROLE_NAMES
        },
        **{
            f"{role}__probabilities": np.asarray(
                probabilities[role], dtype=np.float32
            )
            for role in ROLE_NAMES
        },
    )
    _write_oof_rows(
        output / "oof_rows.csv",
        cache=cache,
        folds=folds,
        sources=sources,
        probabilities=probabilities,
    )
    serial_analysis = _analysis_for_serialization(analysis)
    _write_fold_metrics(output / "fold_metrics.csv", serial_analysis)
    np.savez_compressed(
        output / "role_states.npz", **_flatten_role_states(results)
    )
    _write_json(output / "data_access_ledger.json", access)
    np.savez_compressed(
        output / "fixed_visual_arrays.npz", **visual_arrays
    )
    _write_json(
        output / "fixed_visual_metadata.json", visual_metadata
    )
    summary = {
        "mode": "balanced_bce_frozen_embedding_a0",
        "status": (
            "automatic_pass_replay_and_manual_pending"
            if automatic_pass
            else "automatic_reject_replay_and_manual_pending"
        ),
        "lock_sha256": locked_inputs["lock_sha256"],
        "protocol_sha256": locked_inputs["protocol"]["sha256"],
        "git_head": locked_inputs["repository"]["head"],
        "train_only": True,
        "encoder_frozen": True,
        "validation_access_count": int(access["validation_open_count"]),
        "test_access_count": int(access["test_open_count"]),
        "raw_dataset_modified": False,
        "dataset": {
            "rows": EXPECTED_ROWS,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "embeddings_sha256": array_sha256(cache["embeddings"]),
            "labels_sha256": array_sha256(cache["labels"]),
            "folds_sha256": array_sha256(folds),
            "sources_sha256": _string_sequence_sha256(sources),
        },
        "equation_oracles": oracles,
        "analysis": serial_analysis,
        "performance_gate": performance_gate,
        "structure_checks": structure_checks,
        "structural_pass": structural_pass,
        "automatic_pass": automatic_pass,
        "data_access_evidence": access,
        "visual_artifact": {
            "rows": int(visual_metadata["row_count"]),
            "arrays": "fixed_visual_arrays.npz",
            "metadata": "fixed_visual_metadata.json",
            "contact_sheet": "fixed_visual_contact_sheet.png",
            "manual_review_required": True,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "start_available_physical_ram_gib": start_available_gib,
            "peak_process_rss_bytes": int(resource.peak_rss_bytes),
            "peak_process_rss_gib": float(
                resource.peak_rss_bytes / (1024**3)
            ),
            "peak_virtual_memory_fraction": float(
                resource.peak_virtual_memory_fraction
            ),
            "peak_cuda_bytes": peak_cuda_bytes,
            "peak_cuda_gib": float(peak_cuda_bytes / (1024**3)),
            "process_before": process_before,
            "process_after": process_after,
            "unexpected_processes_during": sorted(
                resource.unexpected_processes.values(),
                key=lambda row: int(row["pid"]),
            ),
        },
        "post_run_locked_hashes": post_run_hashes,
        "fresh_process_replay_required": True,
        "manual_visual_review_required": True,
        "authorization": {
            "default_off_production_integration": False,
            "validation_smoke": False,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_update": False,
        },
        "claim_boundary": (
            "Frozen source-held train-embedding readout gate only; not "
            "encoder-OOF, validation, test, or end-to-end evidence."
        ),
    }
    _write_json(output / "summary.json", summary)
    manifest = _write_manifest(
        output,
        manifest_name="artifact_manifest.json",
        exclude=(
            "replay.json",
            "formal_visual_review.json",
            "final_decision.json",
            "final_manifest.json",
        ),
    )
    manifest_check = _verify_manifest(output, "artifact_manifest.json")
    return {
        "status": summary["status"],
        "automatic_pass": automatic_pass,
        "summary_sha256": _sha256(output / "summary.json"),
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "artifact_manifest_verified": manifest_check["passed"],
        "structural_pass": structural_pass,
        "performance_pass": performance_gate["performance_pass"],
        "authorization": summary["authorization"],
    }


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def _maximum_array_error(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> float:
    if set(left) != set(right):
        return math.inf
    maximum = 0.0
    for name in left:
        a = np.asarray(left[name])
        b = np.asarray(right[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            return math.inf
        if a.dtype.kind in {"i", "u", "b", "U", "S"}:
            if not np.array_equal(a, b):
                return math.inf
        else:
            maximum = max(
                maximum, float(np.max(np.abs(a.astype(float) - b.astype(float))))
            )
    return maximum


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (
                _recursive_numeric_difference(left[name], right[name])
                for name in left
            ),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max(
            (
                _recursive_numeric_difference(a, b)
                for a, b in zip(left, right)
            ),
            default=0.0,
        )
    if isinstance(left, (bool, np.bool_)) or isinstance(
        right, (bool, np.bool_)
    ):
        if not isinstance(left, (bool, np.bool_)) or not isinstance(
            right, (bool, np.bool_)
        ):
            return math.inf
        return 0.0 if bool(left) == bool(right) else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return 0.0 if left == right else math.inf
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def replay_formal(
    *,
    lock_path: Path,
    summary_path: Path,
    device_name: str,
) -> Dict[str, object]:
    resolved_lock = Path(lock_path).expanduser().resolve()
    resolved_summary = Path(summary_path).expanduser().resolve()
    preview = json.loads(resolved_lock.read_text(encoding="utf-8"))
    ledger = DataAccessLedger(
        preview,
        lock_path=resolved_lock,
        excluded_roots=(resolved_summary.parent,),
    )
    with ledger:
        return _replay_formal_with_ledger(
            lock_path=resolved_lock,
            summary_path=resolved_summary,
            device_name=device_name,
            ledger=ledger,
        )


def _replay_formal_with_ledger(
    *,
    lock_path: Path,
    summary_path: Path,
    device_name: str,
    ledger: DataAccessLedger,
) -> Dict[str, object]:
    output = summary_path.parent
    if (output / "replay.json").exists():
        raise FileExistsError("Bal-BCE replay evidence already exists")
    manifest = _verify_manifest(output, "artifact_manifest.json")
    summary_sha = _sha256(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock, locked_inputs = verify_locked_inputs(lock_path)
    if str(device_name) != str(lock["optimization"]["device"]):
        raise ValueError("Replay device differs from machine lock")
    if str(summary["git_head"]) != str(locked_inputs["repository"]["head"]):
        raise ValueError("Replay git HEAD differs from formal")
    if str(summary["lock_sha256"]) != str(locked_inputs["lock_sha256"]):
        raise ValueError("Replay lock SHA differs from formal")
    process_before = _process_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "Unrelated Python/TensorRT process exists before Bal-BCE replay"
        )
    cache, folds, sources, fold_evidence, fold_protocol = _load_audit_inputs(lock)
    oracles = equation_oracles()
    device = torch.device(device_name)
    _configure_torch(device)
    started = time.perf_counter()
    with PeakResourceMonitor() as resource:
        logits, probabilities, results = train_all_roles(
            cache=cache,
            fold_protocol=fold_protocol,
            device=device,
        )
        analysis = analyze_predictions(
            targets=cache["labels"],
            folds=folds,
            logits=logits,
            probabilities=probabilities,
        )
        analysis["_candidate_actions"] = probabilities[
            "balanced_bce_candidate"
        ].argmax(axis=1)
        analysis["_reversed_actions"] = probabilities[
            "reversed_prior_bce_control"
        ].argmax(axis=1)
        performance_gate = assess_performance_gates(analysis, lock)
        visual_arrays, visual_metadata = build_visual_evidence(
            cache=cache,
            folds=folds,
            sources=sources,
            logits=logits,
            probabilities=probabilities,
            results=results,
            ledger=ledger,
            contact_sheet_path=None,
        )
    torch.cuda.synchronize(device)
    elapsed_seconds = float(time.perf_counter() - started)
    post_run_hashes = _post_run_locked_hashes(lock)
    access = ledger.snapshot()
    process_after = _process_snapshot()

    recomputed_oof = {
        "sample_index": np.arange(EXPECTED_ROWS, dtype=np.int64),
        "targets": np.asarray(cache["labels"], dtype=np.int64),
        "folds": np.asarray(folds, dtype=np.int64),
        **{
            f"{role}__raw_logits": np.asarray(logits[role], dtype=np.float32)
            for role in ROLE_NAMES
        },
        **{
            f"{role}__probabilities": np.asarray(
                probabilities[role], dtype=np.float32
            )
            for role in ROLE_NAMES
        },
    }
    oof_error = _maximum_array_error(
        _read_npz(output / "oof_probabilities.npz"),
        recomputed_oof,
    )
    state_error = _maximum_array_error(
        _read_npz(output / "role_states.npz"),
        _flatten_role_states(results),
    )
    visual_error = _maximum_array_error(
        _read_npz(output / "fixed_visual_arrays.npz"),
        visual_arrays,
    )
    stored_trace = [
        json.loads(row)
        for row in (output / "training_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if row
    ]
    trace_error = _recursive_numeric_difference(
        stored_trace, _training_trace_rows(results)
    )
    recomputed_fold_payload = {
        "fold_evidence": fold_evidence,
        "outer_folds": _sanitize_fold_protocol(fold_protocol),
        "hash_encoding": lock["hash_encoding"],
    }
    fold_error = _recursive_numeric_difference(
        json.loads((output / "fold_protocol.json").read_text(encoding="utf-8")),
        recomputed_fold_payload,
    )
    serial_analysis = _analysis_for_serialization(analysis)
    analysis_error = _recursive_numeric_difference(
        summary["analysis"], serial_analysis
    )
    performance_error = _recursive_numeric_difference(
        summary["performance_gate"], performance_gate
    )
    oracle_error = _recursive_numeric_difference(
        summary["equation_oracles"], oracles
    )
    visual_metadata_error = _recursive_numeric_difference(
        json.loads(
            (output / "fixed_visual_metadata.json").read_text(
                encoding="utf-8"
            )
        ),
        visual_metadata,
    )
    stored_access = json.loads(
        (output / "data_access_ledger.json").read_text(encoding="utf-8")
    )
    access_error = _recursive_numeric_difference(stored_access, access)
    thresholds = lock["gates"]["replay"]
    checks = {
        "formal_manifest_verified": bool(manifest["passed"]),
        "locked_inputs_verified": bool(locked_inputs["passed"]),
        "git_head_exact": (
            str(summary["git_head"]) == str(locked_inputs["repository"]["head"])
        ),
        "oof_logits_probabilities_within_tolerance": (
            oof_error
            <= float(thresholds["logit_probability_state_max_error"])
        ),
        "integer_actions_exact": all(
            np.array_equal(
                recomputed_oof[f"{role}__probabilities"].argmax(axis=1),
                _read_npz(output / "oof_probabilities.npz")[
                    f"{role}__probabilities"
                ].argmax(axis=1),
            )
            for role in ROLE_NAMES
        ),
        "states_within_tolerance": (
            state_error
            <= float(thresholds["logit_probability_state_max_error"])
        ),
        "training_trace_within_tolerance": (
            trace_error <= float(thresholds["nested_metric_max_error"])
        ),
        "fold_protocol_exact": fold_error == 0.0,
        "analysis_within_tolerance": (
            analysis_error <= float(thresholds["nested_metric_max_error"])
        ),
        "performance_gate_within_tolerance": (
            performance_error <= float(thresholds["nested_metric_max_error"])
        ),
        "equation_oracles_exact": oracle_error == 0.0,
        "visual_arrays_exact": visual_error == 0.0,
        "visual_metadata_exact": visual_metadata_error == 0.0,
        "data_access_ledger_exact": access_error == 0.0,
        "data_access_ledger_pass": (
            bool(access["passed"])
            and int(access["blocked_attempt_count"]) == 0
            and int(access["validation_open_count"]) == 0
            and int(access["test_open_count"]) == 0
        ),
        "immutable_inputs_unchanged": all(
            bool(value["passed"]) for value in post_run_hashes.values()
        ),
        "no_unrelated_process_during": not bool(resource.unexpected_processes),
        "no_unrelated_process_after": not bool(
            process_after["unexpected_processes"]
        ),
    }
    passed = all(checks.values())
    replay = {
        "mode": "balanced_bce_frozen_embedding_a0_replay",
        "status": "passed" if passed else "failed",
        "formal_summary_sha256": summary_sha,
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "git_head": locked_inputs["repository"]["head"],
        "checks": checks,
        "passed": passed,
        "errors": {
            "oof_max_error": oof_error,
            "state_max_error": state_error,
            "training_trace_max_error": trace_error,
            "fold_protocol_max_error": fold_error,
            "analysis_max_error": analysis_error,
            "performance_gate_max_error": performance_error,
            "equation_oracle_max_error": oracle_error,
            "visual_array_max_error": visual_error,
            "visual_metadata_max_error": visual_metadata_error,
            "data_access_ledger_max_error": access_error,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "process_before": process_before,
            "process_after": process_after,
            "unexpected_processes_during": sorted(
                resource.unexpected_processes.values(),
                key=lambda row: int(row["pid"]),
            ),
        },
        "validation_access_count": int(access["validation_open_count"]),
        "test_access_count": int(access["test_open_count"]),
        "authorization": {
            "default_off_production_integration": False,
            "validation_smoke": False,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_update": False,
        },
    }
    _write_json(output / "replay.json", replay)
    return {
        "status": replay["status"],
        "passed": passed,
        "replay_sha256": _sha256(output / "replay.json"),
        "errors": replay["errors"],
        "authorization": replay["authorization"],
    }


def finalize_visual_review(
    *,
    lock_path: Path,
    output_dir: Path,
    expected_summary_sha256: str,
    decision: str,
    review_note: str,
) -> Dict[str, object]:
    output = Path(output_dir).expanduser().resolve()
    if (output / "formal_visual_review.json").exists():
        raise FileExistsError("Bal-BCE visual review already exists")
    if decision not in {"pass", "reject"}:
        raise ValueError("Manual decision must be pass or reject")
    if len(review_note.strip()) < 20:
        raise ValueError("Manual review note is too short")
    manifest = _verify_manifest(output, "artifact_manifest.json")
    summary_path = output / "summary.json"
    observed_summary_sha = _sha256(summary_path)
    if observed_summary_sha != expected_summary_sha256:
        raise ValueError("Expected summary SHA differs")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay_path = output / "replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if str(replay["formal_summary_sha256"]) != observed_summary_sha:
        raise ValueError("Replay references another formal summary")
    lock, locked_inputs = verify_locked_inputs(lock_path)
    visual_metadata = json.loads(
        (output / "fixed_visual_metadata.json").read_text(encoding="utf-8")
    )
    checklist = {
        "all_twenty_rows_reviewed": int(visual_metadata["row_count"]) == 20,
        "all_sources_train_only": (
            int(visual_metadata["non_train_source_count"]) == 0
        ),
        "fixed_metric_independent_selection": bool(
            visual_metadata["selection_is_metric_independent"]
        ),
        "class1_tp_breaks_reviewed": True,
        "restricted_fp_removals_reviewed": True,
        "prior_intercept_failure_reviewed": True,
    }
    manual_pass = decision == "pass" and all(checklist.values())
    authorized = bool(
        summary["automatic_pass"] and replay["passed"] and manual_pass
    )
    review = {
        "mode": "balanced_bce_frozen_embedding_a0_visual_review",
        "formal_summary_sha256": observed_summary_sha,
        "replay_sha256": _sha256(replay_path),
        "contact_sheet_sha256": _sha256(
            output / "fixed_visual_contact_sheet.png"
        ),
        "visual_metadata_sha256": _sha256(
            output / "fixed_visual_metadata.json"
        ),
        "reviewed_rows": 20,
        "reviewed_at_original_detail": True,
        "decision": decision,
        "manual_pass": manual_pass,
        "review_note": review_note.strip(),
        "checklist": checklist,
    }
    _write_json(output / "formal_visual_review.json", review)
    post_hashes = _post_run_locked_hashes(lock)
    final_decision = {
        "mode": "balanced_bce_frozen_embedding_a0_final_decision",
        "status": "passed" if authorized else "rejected",
        "formal_summary_sha256": observed_summary_sha,
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "replay_sha256": _sha256(replay_path),
        "visual_review_sha256": _sha256(
            output / "formal_visual_review.json"
        ),
        "automatic_pass": bool(summary["automatic_pass"]),
        "replay_pass": bool(replay["passed"]),
        "manual_pass": manual_pass,
        "authorization_conjunction": authorized,
        "locked_inputs_reverified": bool(locked_inputs["passed"]),
        "post_run_locked_hashes": post_hashes,
        "authorization": {
            "default_off_production_integration": authorized,
            "validation_smoke": authorized,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_update": False,
        },
    }
    _write_json(output / "final_decision.json", final_decision)
    final_manifest = _write_manifest(
        output,
        manifest_name="final_manifest.json",
    )
    final_check = _verify_manifest(output, "final_manifest.json")
    return {
        "status": final_decision["status"],
        "authorized": authorized,
        "final_decision_sha256": _sha256(
            output / "final_decision.json"
        ),
        "final_manifest_sha256": final_manifest["manifest_sha256"],
        "final_manifest_verified": final_check["passed"],
        "authorization": final_decision["authorization"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective train-only Balanced BCE frozen-embedding A0 "
            "auditor for TRKH 5-class."
        )
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-a0", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--decision", choices=("pass", "reject"))
    parser.add_argument("--review-note", default="")
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    modes = [
        bool(args.preflight_only),
        bool(args.run_a0),
        args.replay_summary is not None,
        bool(args.finalize_visual_review),
    ]
    if sum(modes) != 1:
        raise ValueError("Choose exactly one Bal-BCE audit phase")
    if args.preflight_only:
        return run_preflight(args.lock)
    if args.run_a0:
        if args.output_dir is None:
            raise ValueError("--output-dir is required for formal A0")
        return run_formal(
            lock_path=args.lock,
            output_dir=args.output_dir,
            device_name=str(args.device),
        )
    if args.replay_summary is not None:
        return replay_formal(
            lock_path=args.lock,
            summary_path=args.replay_summary,
            device_name=str(args.device),
        )
    if args.output_dir is None:
        raise ValueError("--output-dir is required for finalization")
    if not args.expected_summary_sha256:
        raise ValueError("--expected-summary-sha256 is required")
    if args.decision is None:
        raise ValueError("--decision is required")
    return finalize_visual_review(
        lock_path=args.lock,
        output_dir=args.output_dir,
        expected_summary_sha256=str(args.expected_summary_sha256),
        decision=str(args.decision),
        review_note=str(args.review_note),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        result = run_audit(parse_args(argv))
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

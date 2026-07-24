from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS_WORKSPACE_CONFIG

import torch
from torch import nn
from torch.nn import functional as F

from trkh.tools.cap_integral_region_context_a0_engine import (
    ADAM_EPSILON,
    BASE_LR,
    BATCH_SIZE,
    BETAS,
    CAP_ROLES,
    CHANNELS,
    EPOCHS,
    FEATURE_SIZE,
    FOCUS_CLASS,
    GRADIENT_CLIP_NORM,
    LSTM_HIDDEN_DIM,
    NETVLAD_CLUSTERS,
    PIXEL_QUERY_KEY_CHANNELS,
    PRIMARY_SEED,
    REGION_BOXES,
    REGION_COUNT,
    REGION_POOL_SIZE,
    REGION_QUERY_KEY_DIM,
    REPEAT_SEED_OFFSET,
    RESTRICTED_NEGATIVE_CLASSES,
    ROLE_NAMES,
    SAME_BUDGET_CAP_ROLES,
    SPATIAL_DERANGEMENT_SEED,
    TRAINABLE_ROLES,
    UPSAMPLE_SIZE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    CAPIntegralRegionBinaryHead,
    apply_keeper_suppression,
    array_sha256,
    build_epoch_orders,
    build_partitioned_cross_sample_mapping,
    build_spatial_permutations,
    calibrate_tp_retention_threshold,
    engineering_oracles,
    initialize_role_model,
    json_sha256,
    keeper_margin,
    model_state_arrays,
    parameter_contract,
    state_arrays_sha256,
    string_sequence_sha256,
    train_role_fold,
    valid_support_boxes,
    valid_support_global_average,
)


MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = MODULE_PATH.parents[2]
LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_CAP_INTEGRAL_REGION_CONTEXT_A0_LOCK_20260724.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")
DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "audit_cap_integral_region_context_a0_20260724"
)
EXPECTED_ROWS = 763
EXPECTED_FULL_ROWS = 9215
EXPECTED_FEATURE_SHAPE = (EXPECTED_ROWS, CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
VISUAL_ANCHORS = (
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
)
FORBIDDEN_SPLIT_COMPONENTS = frozenset(
    {"val", "valid", "validation", "test"}
)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
                    _jsonable(row),
                    ensure_ascii=False,
                    sort_keys=True,
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


def _contains_forbidden_split(path: object) -> bool:
    return bool(
        set(_path_components(str(path))).intersection(
            FORBIDDEN_SPLIT_COMPONENTS
        )
    )


def _is_train_image_path(path: object) -> bool:
    parts = list(_path_components(str(path)))
    return not _contains_forbidden_split(path) and any(
        parts[index : index + 3] == ["yolo_f", "images", "train"]
        for index in range(max(0, len(parts) - 2))
    )


_ACCESS_LOCK = threading.RLock()
_ACTIVE_LEDGER: Optional["DataAccessLedger"] = None
_HOOK_INSTALLED = False
_HOOK_PROBE_SEEN = False


def _audit_hook(event: str, arguments: Tuple[object, ...]) -> None:
    global _HOOK_PROBE_SEEN
    if event == "trkh.cap_integral_region_context.data_access_probe":
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
            sys.audit("trkh.cap_integral_region_context.data_access_probe")
            if not _HOOK_PROBE_SEEN:
                raise RuntimeError("CAP data-access probe was not observed")
            _HOOK_INSTALLED = True


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
            raise ValueError("CAP immutable-input lock is missing")
        self.domain_roots = {
            "dataset": _normalized_path(
                Path(r"D:\DataAI\AIEx\newdataset")
            ),
            "runs": _normalized_path(REPO_ROOT / "runs"),
            "external": _normalized_path(
                Path(r"D:\DataAI\external_sources")
            ),
        }
        if any(value is None for value in self.domain_roots.values()):
            raise ValueError("CAP data-access roots are invalid")
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
        protocol = lock.get("protocol")
        if isinstance(protocol, Mapping):
            records.append(protocol)
        for record in records:
            normalized = _normalized_path(
                _resolve_locked_path(record["path"])
            )
            if normalized is not None:
                allowed.add(normalized)
        for path in (lock_path, lock_path.with_suffix(".sha256")):
            normalized = _normalized_path(path)
            if normalized is not None:
                allowed.add(normalized)
        self.allowed_exact_paths = frozenset(allowed)
        self.visual_paths: frozenset[str] = frozenset()
        self.visual_authorized = False
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

    def authorize_visual_paths(self, paths: Sequence[object]) -> None:
        normalized: List[str] = []
        for value in paths:
            if not _is_train_image_path(value):
                raise ValueError(f"Visual path is not train-only: {value}")
            path = _normalized_path(value)
            if path is None:
                raise ValueError(f"Visual path cannot be normalized: {value}")
            normalized.append(path)
        if len(normalized) != len(VISUAL_ANCHORS):
            raise ValueError("Exactly 20 visual paths must be authorized")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Visual paths must be unique")
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
                FORBIDDEN_SPLIT_COMPONENTS
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
                f"CAP data-access audit blocked {reason}: {path}"
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


def load_lock(lock_path: Path = LOCK_PATH) -> Dict[str, object]:
    path = Path(lock_path).resolve()
    lock = json.loads(path.read_text(encoding="utf-8"))
    sidecar = path.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).strip()
    expected_sha, expected_name = sidecar.split(maxsplit=1)
    if expected_name != path.name or _sha256(path) != expected_sha:
        raise ValueError("CAP machine-lock sidecar differs")
    return lock


def _verify_file_record(
    name: str,
    record: Mapping[str, object],
) -> Dict[str, object]:
    path = _resolve_locked_path(record["path"])
    observed = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }
    observed["passed"] = bool(
        observed["exists"]
        and int(observed["bytes"]) == int(record["bytes"])
        and observed["sha256"] == record["sha256"]
    )
    if not observed["passed"]:
        raise ValueError(f"Locked CAP file differs: {name}")
    return observed


def runtime_versions() -> Dict[str, object]:
    return {
        "python_full": sys.version,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": metadata.version("numpy"),
        "torch": metadata.version("torch"),
        "torch_cuda": str(torch.version.cuda),
        "scikit_learn": metadata.version("scikit-learn"),
        "pillow": metadata.version("pillow"),
        "psutil": metadata.version("psutil"),
    }


def verify_runtime(lock: Mapping[str, object]) -> Dict[str, object]:
    observed = runtime_versions()
    expected = lock["runtime"]
    checks = {
        "python": str(observed["python_full"]) == str(expected["python"]),
        "numpy": observed["numpy"] == expected["numpy"],
        "torch": observed["torch"] == expected["torch"],
        "cuda": observed["torch_cuda"] == expected["cuda"],
    }
    return {
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def repository_state(
    lock: Mapping[str, object],
    *,
    expected_commit: str | None,
) -> Dict[str, object]:
    head = _run_git(REPO_ROOT, "rev-parse", "HEAD")
    tracked_status = _run_git(
        REPO_ROOT, "status", "--porcelain", "--untracked-files=no"
    )
    lock_parent = str(lock["lock_parent_commit"])
    parent_is_ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                lock_parent,
                head,
            ],
            check=False,
        ).returncode
        == 0
    )
    remote = _run_git(
        REPO_ROOT,
        "rev-parse",
        "origin/classification-only-research",
    )
    checks = {
        "expected_commit_matches": (
            expected_commit is None or head == str(expected_commit)
        ),
        "lock_parent_is_ancestor": parent_is_ancestor,
        "tracked_worktree_clean": tracked_status == "",
        "pushed_branch_matches": remote == head,
    }
    return {
        "head": head,
        "expected_commit": expected_commit,
        "lock_parent_commit": lock_parent,
        "origin_classification_only_research": remote,
        "tracked_status": tracked_status,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_engine_contract(
    lock: Mapping[str, object],
) -> Dict[str, object]:
    architecture = lock["architecture"]
    optimization = lock["optimization"]
    geometry = lock["geometry"]
    candidate = initialize_role_model(
        "cap_context_candidate",
        fold=0,
        device=torch.device("cpu"),
    )
    contracts = {
        role: parameter_contract(
            initialize_role_model(
                role, fold=0, device=torch.device("cpu")
            )
        )
        for role in CAP_ROLES
    }
    candidate_state = model_state_arrays(candidate)
    candidate_hash = state_arrays_sha256(candidate_state)
    primary_hashes = {
        role: state_arrays_sha256(
            model_state_arrays(
                initialize_role_model(
                    role, fold=0, device=torch.device("cpu")
                )
            )
        )
        for role in CAP_ROLES
        if role != "cap_context_seed_repeat"
    }
    repeat_hash = state_arrays_sha256(
        model_state_arrays(
            initialize_role_model(
                "cap_context_seed_repeat",
                fold=0,
                device=torch.device("cpu"),
            )
        )
    )
    module_names = [
        type(module).__name__.casefold() for module in candidate.modules()
    ]
    parameter_names = [
        name.casefold() for name, _ in candidate.named_parameters()
    ]
    checks = {
        "roles_exact": list(ROLE_NAMES) == list(lock["roles"]),
        "channels_exact": CHANNELS == int(lock["cohort"]["feature_shape"][1]),
        "pixel_query_key_exact": (
            PIXEL_QUERY_KEY_CHANNELS
            == int(architecture["pixel_query_key_channels"])
        ),
        "region_query_key_exact": (
            REGION_QUERY_KEY_DIM
            == int(architecture["region_query_key_dim"])
        ),
        "lstm_hidden_exact": (
            LSTM_HIDDEN_DIM == int(architecture["lstm_hidden_dim"])
        ),
        "netvlad_clusters_exact": (
            NETVLAD_CLUSTERS == int(architecture["netvlad_clusters"])
        ),
        "region_geometry_exact": (
            REGION_BOXES.tolist() == geometry["region_boxes"]
            and array_sha256(REGION_BOXES)
            == geometry["region_boxes_sha256"]
        ),
        "roi_sizes_exact": (
            UPSAMPLE_SIZE == int(geometry["upsample_size"])
            and REGION_POOL_SIZE == int(geometry["roi_pool_size"])
            and REGION_COUNT == int(geometry["regions"])
        ),
        "optimization_exact": (
            EPOCHS == int(optimization["epochs"])
            and BATCH_SIZE == int(optimization["batch_size"])
            and BASE_LR == float(optimization["learning_rate"])
            and WARMUP_EPOCHS == int(optimization["warmup_epochs"])
            and PRIMARY_SEED == int(optimization["primary_seed"])
            and REPEAT_SEED_OFFSET
            == int(optimization["repeat_seed_offset"])
            and list(BETAS) == list(optimization["betas"])
            and ADAM_EPSILON == float(optimization["epsilon"])
            and WEIGHT_DECAY == float(optimization["weight_decay"])
            and GRADIENT_CLIP_NORM
            == float(optimization["gradient_clip_norm"])
        ),
        "spatial_seed_exact": (
            SPATIAL_DERANGEMENT_SEED
            == int(geometry["spatial_derangement"]["seed"])
        ),
        "visual_anchors_exact": (
            list(VISUAL_ANCHORS) == geometry["visual_sample_indices"]
            and array_sha256(
                np.asarray(VISUAL_ANCHORS, dtype=np.int64)
            )
            == geometry["visual_sample_indices_sha256"]
        ),
        "gamma_zero": bool(
            torch.equal(
                candidate.pixel_context.gamma.detach(),
                torch.zeros_like(candidate.pixel_context.gamma),
            )
        ),
        "same_budget_contracts_exact": all(
            contracts[role] == contracts["cap_context_candidate"]
            for role in SAME_BUDGET_CAP_ROLES
        ),
        "self_only_same_parameter_contract": (
            contracts["integral_self_only_control"]
            == contracts["cap_context_candidate"]
        ),
        "primary_initial_states_exact": (
            len(set(primary_hashes.values())) == 1
            and next(iter(primary_hashes.values())) == candidate_hash
        ),
        "repeat_initial_state_differs": repeat_hash != candidate_hash,
        "no_sigmoid_module": "sigmoid" not in module_names,
        "no_squeeze_excitation_module": not any(
            "squeeze" in name or "excitation" in name
            for name in module_names
        ),
        "no_spectral_norm_parameter": not any(
            "parametrizations" in name or "weight_orig" in name
            for name in parameter_names
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "contracts": contracts,
        "candidate_initial_state_sha256": candidate_hash,
        "repeat_initial_state_sha256": repeat_hash,
    }


def _load_cidt_clean(path: Path) -> Dict[str, np.ndarray]:
    rows: Dict[int, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "clean":
                continue
            sample_index = int(row["sample_index"])
            if sample_index in rows:
                raise ValueError(f"Duplicate CIDT clean row {sample_index}")
            rows[sample_index] = row
    if len(rows) != EXPECTED_FULL_ROWS or sorted(rows) != list(
        range(EXPECTED_FULL_ROWS)
    ):
        raise ValueError("CIDT clean rows differ from 0..9214")
    ordered = [rows[index] for index in range(EXPECTED_FULL_ROWS)]
    probabilities = np.asarray(
        [
            [float(row[f"keeper_prob_{class_index}"]) for class_index in range(5)]
            for row in ordered
        ],
        dtype=np.float32,
    )
    return {
        "sample_indices": np.arange(EXPECTED_FULL_ROWS, dtype=np.int64),
        "targets": np.asarray(
            [int(row["target_index"]) for row in ordered],
            dtype=np.int64,
        ),
        "folds": np.asarray(
            [int(row["fold"]) for row in ordered], dtype=np.int64
        ),
        "sources": np.asarray(
            [row["source_stem"] for row in ordered]
        ).astype(str),
        "image_paths": np.asarray(
            [row["image_path"] for row in ordered]
        ).astype(str),
        "keeper_probabilities": probabilities,
        "keeper_predictions": probabilities.argmax(axis=1).astype(np.int64),
    }


def load_locked_inputs(
    lock: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    immutable = lock["immutable_inputs"]
    features = np.load(
        _resolve_locked_path(immutable["feature_cache"]["path"]),
        allow_pickle=False,
    )
    if features.shape != EXPECTED_FEATURE_SHAPE or features.dtype != np.float16:
        raise ValueError("CAP feature cache shape/dtype differs")
    if not bool(np.isfinite(features).all()):
        raise ValueError("CAP feature cache is non-finite")
    with np.load(
        _resolve_locked_path(immutable["geometry_cache"]["path"]),
        allow_pickle=False,
    ) as archive:
        geometry = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    expected_geometry = {
        "sample_indices",
        "targets",
        "folds",
        "source_stems",
        "keeper_probabilities",
        "valid_masks",
        "bbox_masks",
        "model_boxes",
        "crop_boxes",
    }
    if set(geometry) != expected_geometry:
        raise ValueError("CAP geometry keys differ")
    cidt = _load_cidt_clean(
        _resolve_locked_path(immutable["cidt_predictions"]["path"])
    )
    sample_indices = geometry["sample_indices"].astype(np.int64)
    if len(set(sample_indices.tolist())) != EXPECTED_ROWS:
        raise ValueError("CAP cohort sample indices are not unique")
    if not np.array_equal(
        cidt["targets"][sample_indices],
        geometry["targets"].astype(np.int64),
    ):
        raise ValueError("CAP cohort targets differ from CIDT")
    if not np.array_equal(
        cidt["folds"][sample_indices],
        geometry["folds"].astype(np.int64),
    ):
        raise ValueError("CAP cohort folds differ from CIDT")
    if not np.array_equal(
        cidt["sources"][sample_indices],
        geometry["source_stems"].astype(str),
    ):
        raise ValueError("CAP cohort sources differ from CIDT")
    geometry_probabilities = geometry["keeper_probabilities"].astype(
        np.float32
    )
    cidt_cohort_probabilities = cidt["keeper_probabilities"][
        sample_indices
    ]
    if not np.array_equal(
        cidt_cohort_probabilities.argmax(axis=1),
        geometry_probabilities.argmax(axis=1),
    ):
        raise ValueError("CAP cohort keeper predictions differ from CIDT")
    geometry_rivals = geometry_probabilities.copy()
    cidt_rivals = cidt_cohort_probabilities.copy()
    geometry_rivals[:, FOCUS_CLASS] = -np.inf
    cidt_rivals[:, FOCUS_CLASS] = -np.inf
    support_boxes = valid_support_boxes(geometry["valid_masks"])
    spatial_permutations = build_spatial_permutations(sample_indices)
    cohort = lock["cohort"]
    locked_geometry = lock["geometry"]
    checks = {
        "sample_indices": (
            array_sha256(sample_indices)
            == cohort["sample_indices_sha256"]
        ),
        "targets": (
            array_sha256(geometry["targets"].astype(np.int64))
            == cohort["targets_sha256"]
        ),
        "folds": (
            array_sha256(geometry["folds"].astype(np.int64))
            == cohort["folds_sha256"]
        ),
        "sources": (
            string_sequence_sha256(geometry["source_stems"].astype(str))
            == cohort["sources_sha256"]
        ),
        "probabilities": (
            array_sha256(
                geometry["keeper_probabilities"].astype(np.float32)
            )
            == cohort["probabilities_sha256"]
        ),
        "valid_masks": (
            array_sha256(geometry["valid_masks"].astype(np.bool_))
            == cohort["valid_masks_sha256"]
        ),
        "bbox_masks": (
            array_sha256(geometry["bbox_masks"].astype(np.bool_))
            == cohort["bbox_masks_sha256"]
        ),
        "support_boxes": (
            array_sha256(support_boxes)
            == cohort["valid_support_boxes_sha256"]
        ),
        "spatial_permutations": (
            array_sha256(spatial_permutations)
            == locked_geometry["spatial_derangement"][
                "permutations_sha256"
            ]
        ),
        "visual_anchors": (
            list(VISUAL_ANCHORS)
            == locked_geometry["visual_sample_indices"]
            and array_sha256(
                np.asarray(VISUAL_ANCHORS, dtype=np.int64)
            )
            == locked_geometry["visual_sample_indices_sha256"]
        ),
        "cidt_keeper_predictions": np.array_equal(
            cidt_cohort_probabilities.argmax(axis=1),
            geometry_probabilities.argmax(axis=1),
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"CAP cohort array hashes differ: {checks}")
    visual_paths = cidt["image_paths"][
        np.asarray(VISUAL_ANCHORS, dtype=np.int64)
    ]
    if not all(_is_train_image_path(path) for path in visual_paths):
        raise ValueError("A CAP visual path is not yolo_f train")
    return {
        "features": np.asarray(features),
        **geometry,
        "support_boxes": support_boxes,
        "spatial_permutations": spatial_permutations,
        "full_targets": cidt["targets"],
        "full_folds": cidt["folds"],
        "full_sources": cidt["sources"],
        "full_image_paths": cidt["image_paths"],
        "full_keeper_probabilities": cidt["keeper_probabilities"],
        "full_keeper_predictions": cidt["keeper_predictions"],
        "cidt_geometry_probability_max_abs_error": np.asarray(
            np.max(
                np.abs(
                    cidt_cohort_probabilities.astype(np.float64)
                    - geometry_probabilities.astype(np.float64)
                )
            ),
            dtype=np.float64,
        ),
        "cidt_geometry_rival_argmax_difference_count": np.asarray(
            np.count_nonzero(
                cidt_rivals.argmax(axis=1)
                != geometry_rivals.argmax(axis=1)
            ),
            dtype=np.int64,
        ),
        "input_checks": checks,
    }


def build_fold_protocol(
    inputs: Mapping[str, np.ndarray],
    lock: Mapping[str, object],
) -> List[Dict[str, object]]:
    folds = np.asarray(inputs["folds"], dtype=np.int64)
    targets = np.asarray(inputs["targets"], dtype=np.int64)
    sample_indices = np.asarray(inputs["sample_indices"], dtype=np.int64)
    sources = np.asarray(inputs["source_stems"]).astype(str).tolist()
    predictions = np.asarray(
        inputs["keeper_probabilities"], dtype=np.float32
    ).argmax(axis=1)
    records: List[Dict[str, object]] = []
    for locked in lock["folds"]:
        outer = int(locked["outer_fold"])
        calibration_fold = int(locked["calibration_fold"])
        held = np.flatnonzero(folds == outer).astype(np.int64)
        calibration = np.flatnonzero(
            folds == calibration_fold
        ).astype(np.int64)
        fit = np.flatnonzero(
            (folds != outer) & (folds != calibration_fold)
        ).astype(np.int64)
        partitions = {
            "fit": fit,
            "calibration": calibration,
            "held": held,
        }
        mapping, mapping_details = build_partitioned_cross_sample_mapping(
            partitions,
            sample_indices,
            sources,
            PRIMARY_SEED + 7000 + outer,
        )
        primary_orders = build_epoch_orders(
            fit, seed=PRIMARY_SEED + outer
        )
        repeat_orders = build_epoch_orders(
            fit, seed=PRIMARY_SEED + REPEAT_SEED_OFFSET + outer
        )
        checks: Dict[str, bool] = {
            "calibration_fold": calibration_fold
            == int(locked["calibration_fold"]),
            "fit_indices": array_sha256(fit)
            == locked["fit"]["indices_sha256"],
            "calibration_indices": array_sha256(calibration)
            == locked["calibration"]["indices_sha256"],
            "held_indices": array_sha256(held)
            == locked["held"]["indices_sha256"],
            "mapping": array_sha256(mapping)
            == locked["cross_sample_derangement_sha256"],
            "mapping_details": mapping_details
            == locked["cross_sample_derangement_partitions"],
            "primary_orders": [
                array_sha256(order) for order in primary_orders
            ]
            == locked["primary_orders"]["per_epoch_sha256"],
            "repeat_orders": [
                array_sha256(order) for order in repeat_orders
            ]
            == locked["repeat_orders"]["per_epoch_sha256"],
            "primary_all_orders": array_sha256(
                np.concatenate(primary_orders)
            )
            == locked["primary_orders"]["all_epochs_sha256"],
            "repeat_all_orders": array_sha256(
                np.concatenate(repeat_orders)
            )
            == locked["repeat_orders"]["all_epochs_sha256"],
        }
        for name, indices in partitions.items():
            expected = locked[name]
            checks[f"{name}_rows"] = int(indices.size) == int(
                expected["rows"]
            )
            checks[f"{name}_sample_indices"] = (
                array_sha256(sample_indices[indices])
                == expected["sample_indices_sha256"]
            )
            checks[f"{name}_targets"] = (
                array_sha256(targets[indices])
                == expected["targets_sha256"]
            )
            checks[f"{name}_sources"] = (
                string_sequence_sha256(
                    sources[int(index)] for index in indices
                )
                == expected["sources_sha256"]
            )
            checks[f"{name}_tp1"] = int(
                ((targets[indices] == 1) & (predictions[indices] == 1)).sum()
            ) == int(expected["tp1"])
            checks[f"{name}_fn1"] = int(
                ((targets[indices] == 1) & (predictions[indices] != 1)).sum()
            ) == int(expected["fn1"])
            checks[f"{name}_restricted_fp"] = int(
                (
                    np.isin(
                        targets[indices], RESTRICTED_NEGATIVE_CLASSES
                    )
                    & (predictions[indices] == 1)
                ).sum()
            ) == int(expected["restricted_fp"])
        if not all(checks.values()):
            raise ValueError(f"CAP fold {outer} differs: {checks}")
        records.append(
            {
                "outer_fold": outer,
                "calibration_fold": calibration_fold,
                "fit_indices": fit,
                "calibration_indices": calibration,
                "held_indices": held,
                "cross_sample_mapping": mapping,
                "cross_sample_mapping_sha256": array_sha256(mapping),
                "cross_sample_mapping_details": mapping_details,
                "primary_order_hashes": [
                    array_sha256(order) for order in primary_orders
                ],
                "repeat_order_hashes": [
                    array_sha256(order) for order in repeat_orders
                ],
                "checks": checks,
            }
        )
    return records


def _configure_torch(device: torch.device) -> torch.device:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CAP formal A0 requires the locked CUDA runtime")
    resolved = torch.device(
        "cuda",
        torch.cuda.current_device()
        if device.index is None
        else int(device.index),
    )
    torch.cuda.set_device(resolved)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(PRIMARY_SEED)
    torch.cuda.manual_seed_all(PRIMARY_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(resolved)
    return resolved


def process_snapshot() -> Dict[str, object]:
    current = psutil.Process()
    allowed = {current.pid}
    try:
        allowed.update(process.pid for process in current.parents())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    try:
        allowed.update(
            process.pid for process in current.children(recursive=True)
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    observed: List[Dict[str, object]] = []
    unexpected: List[Dict[str, object]] = []
    for process in psutil.process_iter(
        ["pid", "ppid", "name", "create_time", "cmdline"]
    ):
        try:
            info = process.info
            name = str(info.get("name") or "").casefold()
            if name not in {"python.exe", "pythonw.exe", "trtexec.exe"}:
                continue
            row = {
                "pid": int(info["pid"]),
                "ppid": int(info.get("ppid") or 0),
                "name": str(info.get("name") or ""),
                "create_time": float(info.get("create_time") or 0.0),
                "cmdline": [
                    str(value) for value in (info.get("cmdline") or [])
                ],
                "owned_by_auditor_chain": int(info["pid"]) in allowed,
            }
            observed.append(row)
            if not row["owned_by_auditor_chain"]:
                unexpected.append(row)
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue
    return {
        "current_pid": current.pid,
        "allowed_pids": sorted(allowed),
        "processes": sorted(observed, key=lambda row: int(row["pid"])),
        "unexpected_processes": sorted(
            unexpected, key=lambda row: int(row["pid"])
        ),
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
                for row in process_snapshot()["unexpected_processes"]:
                    self.unexpected_processes[int(row["pid"])] = row
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return

    def __enter__(self) -> "PeakResourceMonitor":
        self._thread = threading.Thread(
            target=self._sample, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            int(self.process.memory_info().rss),
        )


def synthetic_gradient_checks(
    device: torch.device,
) -> Dict[str, object]:
    resolved = _configure_torch(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(781)
    features = torch.randn(
        (2, CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        generator=generator,
        dtype=torch.float32,
    ).to(resolved)
    boxes = np.asarray([[2, 1, 15, 14], [0, 3, 16, 13]], dtype=np.int64)
    spatial = torch.as_tensor(
        build_spatial_permutations(
            np.asarray([101, 202], dtype=np.int64)
        ),
        dtype=torch.long,
        device=resolved,
    )
    targets = torch.tensor([1.0, 0.0], device=resolved)
    records: Dict[str, object] = {}
    for role in TRAINABLE_ROLES:
        model = initialize_role_model(role, fold=0, device=resolved)
        kwargs: Dict[str, object] = {}
        if role == "global_gap_linear_control":
            logits = model(features.mean(dim=(2, 3)))
        else:
            if role == "cap_spatial_deranged_control":
                kwargs["spatial_permutations"] = spatial
            if role == "cap_cross_sample_context_control":
                kwargs["partner_features"] = torch.flip(
                    features, dims=[0]
                )
                kwargs["partner_support_boxes"] = boxes[::-1].copy()
            logits = model(features, boxes, **kwargs)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            gradient
            for parameter in model.parameters()
            if (gradient := parameter.grad) is not None
        ]
        records[role] = {
            "logits_finite": bool(torch.isfinite(logits).all()),
            "loss_finite": bool(torch.isfinite(loss)),
            "gradient_tensor_count": len(gradients),
            "gradients_finite": bool(
                gradients
                and all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                )
            ),
        }
        del model, logits, loss, gradients
    torch.cuda.synchronize(resolved)
    torch.cuda.empty_cache()
    checks = {
        role: bool(
            record["logits_finite"]
            and record["loss_finite"]
            and record["gradients_finite"]
        )
        for role, record in records.items()
    }
    return {
        "roles": records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def synthetic_resource_check(
    device: torch.device,
    lock: Mapping[str, object],
) -> Dict[str, object]:
    resolved = _configure_torch(device)
    before = process_snapshot()
    records: Dict[str, object] = {}
    with PeakResourceMonitor() as monitor:
        for name, dtype in (
            ("fp32", torch.float32),
            ("bf16", torch.bfloat16),
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(resolved)
            model = CAPIntegralRegionBinaryHead(
                context_mode="cross_sample_context"
            ).to(device=resolved, dtype=dtype)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(991 if name == "fp32" else 997)
            features = torch.randn(
                (
                    BATCH_SIZE,
                    CHANNELS,
                    FEATURE_SIZE,
                    FEATURE_SIZE,
                ),
                generator=generator,
                dtype=torch.float32,
            ).to(device=resolved, dtype=dtype)
            partner = torch.flip(features, dims=[0])
            boxes = np.tile(
                np.asarray([[0, 0, 16, 16]], dtype=np.int64),
                (BATCH_SIZE, 1),
            )
            logits = model(
                features,
                boxes,
                partner_features=partner,
                partner_support_boxes=boxes,
            )
            targets = torch.arange(
                BATCH_SIZE, device=resolved, dtype=torch.int64
            ).remainder(2).to(dtype=dtype)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
            torch.cuda.synchronize(resolved)
            records[name] = {
                "finite_logits": bool(torch.isfinite(logits).all()),
                "finite_loss": bool(torch.isfinite(loss)),
                "peak_cuda_bytes": int(
                    torch.cuda.max_memory_allocated(resolved)
                ),
                "peak_cuda_gib": float(
                    torch.cuda.max_memory_allocated(resolved) / (1024**3)
                ),
            }
            del model, features, partner, logits, targets, loss
    torch.cuda.empty_cache()
    after = process_snapshot()
    cuda_peak = max(
        float(record["peak_cuda_gib"]) for record in records.values()
    )
    rss_peak = float(monitor.peak_rss_bytes / (1024**3))
    gates = lock["resource_gates"]
    checks = {
        "fp32_finite": bool(
            records["fp32"]["finite_logits"]
            and records["fp32"]["finite_loss"]
        ),
        "bf16_finite": bool(
            records["bf16"]["finite_logits"]
            and records["bf16"]["finite_loss"]
        ),
        "cuda_peak_below_6_gib": (
            cuda_peak < float(gates["cuda_peak_gib_max"])
        ),
        "rss_peak_below_10_gib": (
            rss_peak < float(gates["rss_peak_gib_max"])
        ),
        "no_unexpected_process_before": not before[
            "unexpected_processes"
        ],
        "no_unexpected_process_during": not monitor.unexpected_processes,
        "no_unexpected_process_after": not after[
            "unexpected_processes"
        ],
    }
    return {
        "records": records,
        "peak_cuda_gib": cuda_peak,
        "peak_rss_gib": rss_peak,
        "peak_virtual_memory_fraction": (
            monitor.peak_virtual_memory_fraction
        ),
        "process_before": before,
        "unexpected_processes_during": list(
            monitor.unexpected_processes.values()
        ),
        "process_after": after,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_locked_files(
    lock: Mapping[str, object],
) -> Dict[str, object]:
    files = {
        name: _verify_file_record(name, record)
        for name, record in lock["immutable_inputs"].items()
    }
    files["protocol"] = _verify_file_record(
        "protocol", lock["protocol"]
    )
    return {
        "files": files,
        "passed": all(
            bool(record["passed"]) for record in files.values()
        ),
    }


def run_preflight(
    *,
    lock_path: Path = LOCK_PATH,
    expected_commit: str | None = None,
    device: torch.device = torch.device("cuda"),
    include_resource: bool = True,
) -> Dict[str, object]:
    lock_path = Path(lock_path).resolve()
    lock = load_lock(lock_path)
    ledger = DataAccessLedger(
        lock,
        lock_path=lock_path,
        excluded_roots=(),
    )
    with ledger:
        files = verify_locked_files(lock)
        inputs = load_locked_inputs(lock)
        folds = build_fold_protocol(inputs, lock)
    ledger_evidence = ledger.snapshot()
    runtime = verify_runtime(lock)
    repository = repository_state(
        lock, expected_commit=expected_commit
    )
    engine = verify_engine_contract(lock)
    oracles = engineering_oracles()
    gradients = synthetic_gradient_checks(device)
    resource = (
        synthetic_resource_check(device, lock)
        if include_resource
        else {"passed": True, "skipped": True}
    )
    checks = {
        "files": bool(files["passed"]),
        "inputs": all(inputs["input_checks"].values()),
        "folds": len(folds) == 5
        and all(all(record["checks"].values()) for record in folds),
        "ledger": bool(ledger_evidence["passed"]),
        "runtime": bool(runtime["passed"]),
        "repository": bool(repository["passed"]),
        "engine": bool(engine["passed"]),
        "oracles": bool(oracles["passed"]),
        "gradients": bool(gradients["passed"]),
        "resource": bool(resource["passed"]),
    }
    return {
        "protocol_id": lock["protocol_id"],
        "checks": checks,
        "passed": all(checks.values()),
        "locked_files": files,
        "input_checks": inputs["input_checks"],
        "fold_checks": [
            {
                "outer_fold": record["outer_fold"],
                "checks": record["checks"],
            }
            for record in folds
        ],
        "ledger": ledger_evidence,
        "runtime": runtime,
        "repository": repository,
        "engine": engine,
        "oracles": oracles,
        "gradients": gradients,
        "resource": resource,
    }


def _sanitize_training_evidence(
    evidence: Mapping[str, object],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], List[Dict[str, object]]]:
    record = dict(evidence)
    states = {
        str(name): np.asarray(value)
        for name, value in record.pop("state").items()
    }
    record.pop("orders")
    trace = [
        dict(row) for row in record.pop("trace")
    ]
    for name in (
        "calibration_logits",
        "calibration_scores",
        "held_logits",
        "held_scores",
    ):
        record.pop(name)
    return record, states, trace


def train_all_roles(
    *,
    inputs: Mapping[str, np.ndarray],
    fold_protocol: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    resolved = _configure_torch(device)
    features = torch.as_tensor(
        np.asarray(inputs["features"], dtype=np.float32),
        dtype=torch.float32,
        device=resolved,
    )
    targets = torch.as_tensor(
        np.asarray(inputs["targets"], dtype=np.int64),
        dtype=torch.long,
        device=resolved,
    )
    valid_masks = torch.as_tensor(
        np.asarray(inputs["valid_masks"], dtype=np.bool_),
        dtype=torch.bool,
        device=resolved,
    )
    gap_features = valid_support_global_average(features, valid_masks)
    support_boxes = np.asarray(inputs["support_boxes"], dtype=np.int64)
    spatial = np.asarray(
        inputs["spatial_permutations"], dtype=np.int64
    )
    if spatial.shape != (EXPECTED_ROWS, REGION_COUNT):
        raise RuntimeError("Locked spatial permutations have an invalid shape")
    keeper_probabilities = np.asarray(
        inputs["keeper_probabilities"], dtype=np.float32
    )
    keeper_predictions = keeper_probabilities.argmax(axis=1)
    target_values = np.asarray(inputs["targets"], dtype=np.int64)

    logits = {
        role: np.full(EXPECTED_ROWS, np.nan, dtype=np.float32)
        for role in ROLE_NAMES
    }
    scores = {
        role: np.full(EXPECTED_ROWS, np.nan, dtype=np.float32)
        for role in ROLE_NAMES
    }
    calibration_logits = {
        role: np.full(EXPECTED_ROWS, np.nan, dtype=np.float32)
        for role in ROLE_NAMES
    }
    calibration_scores = {
        role: np.full(EXPECTED_ROWS, np.nan, dtype=np.float32)
        for role in ROLE_NAMES
    }
    suppressed = {
        role: np.zeros(EXPECTED_ROWS, dtype=np.bool_)
        for role in ROLE_NAMES
    }
    thresholds: Dict[str, Dict[str, object]] = {
        role: {} for role in ROLE_NAMES
    }
    training_records: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    state_arrays: Dict[str, np.ndarray] = {}
    model_bank: Dict[int, Dict[str, nn.Module]] = {}

    fixed_margin = keeper_margin(keeper_probabilities).astype(np.float32)
    for fold_record in fold_protocol:
        fold = int(fold_record["outer_fold"])
        fit = np.asarray(fold_record["fit_indices"], dtype=np.int64)
        calibration = np.asarray(
            fold_record["calibration_indices"], dtype=np.int64
        )
        held = np.asarray(
            fold_record["held_indices"], dtype=np.int64
        )
        cross_mapping = np.asarray(
            fold_record["cross_sample_mapping"], dtype=np.int64
        )
        logits["keeper_margin_control"][held] = fixed_margin[held]
        scores["keeper_margin_control"][held] = fixed_margin[held]
        calibration_logits["keeper_margin_control"][
            calibration
        ] = fixed_margin[calibration]
        calibration_scores["keeper_margin_control"][
            calibration
        ] = fixed_margin[calibration]

        for role in TRAINABLE_ROLES:
            print(
                f"CAP A0 training role={role} fold={fold} "
                f"fit={fit.size} calibration={calibration.size} "
                f"held={held.size}",
                flush=True,
            )
            trained = train_role_fold(
                role=role,
                fold=fold,
                features=features,
                gap_features=gap_features,
                targets=targets,
                support_boxes=support_boxes,
                fit_indices=fit,
                calibration_indices=calibration,
                held_indices=held,
                spatial_permutations=spatial,
                cross_sample_mapping=cross_mapping,
                device=resolved,
            )
            evidence = trained.evidence
            expected_hashes = (
                fold_record["repeat_order_hashes"]
                if role == "cap_context_seed_repeat"
                else fold_record["primary_order_hashes"]
            )
            if evidence["order_hashes"] != expected_hashes:
                raise RuntimeError(
                    f"CAP order drift role={role} fold={fold}"
                )
            expected_steps = EPOCHS * (
                (fit.size + BATCH_SIZE - 1) // BATCH_SIZE
            )
            if int(evidence["update_count"]) != int(expected_steps):
                raise RuntimeError(
                    f"CAP optimizer-step drift role={role} fold={fold}"
                )
            logits[role][held] = np.asarray(
                evidence["held_logits"], dtype=np.float32
            )
            scores[role][held] = np.asarray(
                evidence["held_scores"], dtype=np.float32
            )
            calibration_logits[role][calibration] = np.asarray(
                evidence["calibration_logits"], dtype=np.float32
            )
            calibration_scores[role][calibration] = np.asarray(
                evidence["calibration_scores"], dtype=np.float32
            )
            sanitized, role_states, role_trace = (
                _sanitize_training_evidence(evidence)
            )
            sanitized["outer_fold"] = fold
            training_records.append(sanitized)
            trace_rows.extend(role_trace)
            for name, value in role_states.items():
                state_arrays[f"fold{fold}.{role}.{name}"] = value
            if role in {
                "cap_context_candidate",
                "cap_spatial_deranged_control",
            }:
                model_bank.setdefault(fold, {})[role] = trained.model.to(
                    device=torch.device("cpu")
                )
            del trained
            torch.cuda.empty_cache()

        calibration_tp = (
            (target_values[calibration] == FOCUS_CLASS)
            & (keeper_predictions[calibration] == FOCUS_CLASS)
        )
        for role in ROLE_NAMES:
            threshold = calibrate_tp_retention_threshold(
                calibration_scores[role][calibration],
                calibration_tp,
                maximum_break_fraction=0.03,
            )
            thresholds[role][str(fold)] = threshold
            actions = apply_keeper_suppression(
                keeper_probabilities[held],
                scores[role][held],
                float(threshold["threshold"]),
            )
            suppressed[role][held] = actions["suppressed"]

    for name, values in (
        *[(f"logits.{role}", value) for role, value in logits.items()],
        *[(f"scores.{role}", value) for role, value in scores.items()],
        *[
            (f"calibration_logits.{role}", value)
            for role, value in calibration_logits.items()
        ],
        *[
            (f"calibration_scores.{role}", value)
            for role, value in calibration_scores.items()
        ],
    ):
        if not bool(np.isfinite(values).all()):
            raise FloatingPointError(f"CAP {name} is incomplete")
    return {
        "logits": logits,
        "scores": scores,
        "calibration_logits": calibration_logits,
        "calibration_scores": calibration_scores,
        "suppressed": suppressed,
        "thresholds": thresholds,
        "training_records": training_records,
        "trace_rows": trace_rows,
        "state_arrays": state_arrays,
        "state_arrays_sha256": state_arrays_sha256(state_arrays),
        "model_bank": model_bank,
        "spatial_permutations": spatial,
        "features_device": features,
    }


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, object]:
    target_values = np.asarray(targets, dtype=np.int64)
    predicted_values = np.asarray(predictions, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target_values,
        predicted_values,
        labels=np.arange(5, dtype=np.int64),
        zero_division=0,
    )
    matrix = confusion_matrix(
        target_values,
        predicted_values,
        labels=np.arange(5, dtype=np.int64),
    )
    return {
        "rows": int(target_values.size),
        "accuracy": float(
            accuracy_score(target_values, predicted_values)
        ),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(5)
        },
        "confusion_matrix": matrix.astype(np.int64).tolist(),
    }


def _binary_auroc(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive_mask, dtype=np.bool_)
    negative = np.asarray(negative_mask, dtype=np.bool_)
    selected = positive | negative
    labels = positive[selected].astype(np.int64)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("AUROC subset does not contain both classes")
    return float(roc_auc_score(labels, values[selected]))


def _full_predictions_for_role(
    *,
    inputs: Mapping[str, np.ndarray],
    role_suppressed: np.ndarray,
) -> np.ndarray:
    full_predictions = np.asarray(
        inputs["full_keeper_predictions"], dtype=np.int64
    ).copy()
    cohort_probabilities = np.asarray(
        inputs["keeper_probabilities"], dtype=np.float64
    )
    rivals = cohort_probabilities.copy()
    rivals[:, FOCUS_CLASS] = -np.inf
    replacements = rivals.argmax(axis=1).astype(np.int64)
    sample_indices = np.asarray(
        inputs["sample_indices"], dtype=np.int64
    )
    suppressed = np.asarray(role_suppressed, dtype=np.bool_)
    full_predictions[sample_indices[suppressed]] = replacements[suppressed]
    return full_predictions


def analyze_roles(
    *,
    inputs: Mapping[str, np.ndarray],
    training: Mapping[str, object],
) -> Dict[str, object]:
    targets = np.asarray(inputs["targets"], dtype=np.int64)
    folds = np.asarray(inputs["folds"], dtype=np.int64)
    keeper_probabilities = np.asarray(
        inputs["keeper_probabilities"], dtype=np.float32
    )
    keeper_predictions = keeper_probabilities.argmax(axis=1)
    keeper_tp = (
        (targets == FOCUS_CLASS)
        & (keeper_predictions == FOCUS_CLASS)
    )
    focus_fn = (
        (targets == FOCUS_CLASS)
        & (keeper_predictions != FOCUS_CLASS)
    )
    restricted_fp = (
        np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
        & (keeper_predictions == FOCUS_CLASS)
    )
    full_targets = np.asarray(
        inputs["full_targets"], dtype=np.int64
    )
    full_keeper_predictions = np.asarray(
        inputs["full_keeper_predictions"], dtype=np.int64
    )
    keeper_metrics = classification_metrics(
        full_targets, full_keeper_predictions
    )
    scores = training["scores"]
    suppressed = training["suppressed"]
    roles: Dict[str, object] = {}
    full_predictions: Dict[str, np.ndarray] = {}
    for role in ROLE_NAMES:
        role_scores = np.asarray(scores[role], dtype=np.float32)
        role_suppressed = np.asarray(
            suppressed[role], dtype=np.bool_
        )
        updated = _full_predictions_for_role(
            inputs=inputs, role_suppressed=role_suppressed
        )
        full_predictions[role] = updated
        metrics = classification_metrics(full_targets, updated)
        keeper_correct = full_keeper_predictions == full_targets
        updated_correct = updated == full_targets
        changed = updated != full_keeper_predictions
        corrections = int(
            ((~keeper_correct) & updated_correct).sum()
        )
        harms = int((keeper_correct & (~updated_correct)).sum())
        neutral = int(
            (changed & (~keeper_correct) & (~updated_correct)).sum()
        )
        fold_rows: List[Dict[str, object]] = []
        fold_aurocs: Dict[str, float] = {}
        for fold in range(5):
            selected = folds == fold
            fold_tp = selected & keeper_tp
            fold_fp = selected & restricted_fp
            fold_fn = selected & focus_fn
            fold_auc = _binary_auroc(
                role_scores, fold_tp, fold_fp
            )
            fold_aurocs[str(fold)] = fold_auc
            fold_rows.append(
                {
                    "outer_fold": fold,
                    "tp_rows": int(fold_tp.sum()),
                    "fn_rows": int(fold_fn.sum()),
                    "restricted_fp_rows": int(fold_fp.sum()),
                    "tp_retention": float(
                        (~role_suppressed[fold_tp]).mean()
                    ),
                    "restricted_fp_rejection": float(
                        role_suppressed[fold_fp].mean()
                    ),
                    "restricted_fp_rejected": int(
                        role_suppressed[fold_fp].sum()
                    ),
                    "tp_harms": int(role_suppressed[fold_tp].sum()),
                    "tp_vs_restricted_fp_auroc": fold_auc,
                    "fn_vs_restricted_fp_auroc": _binary_auroc(
                        role_scores, fold_fn, fold_fp
                    ),
                }
            )
        role_metrics = {
            "tp_vs_restricted_fp_auroc": _binary_auroc(
                role_scores, keeper_tp, restricted_fp
            ),
            "fn_vs_restricted_fp_auroc": _binary_auroc(
                role_scores, focus_fn, restricted_fp
            ),
            "tp_retention": float(
                (~role_suppressed[keeper_tp]).mean()
            ),
            "restricted_fp_rejection": float(
                role_suppressed[restricted_fp].mean()
            ),
            "restricted_fp_rejected": int(
                role_suppressed[restricted_fp].sum()
            ),
            "tp_harms": int(role_suppressed[keeper_tp].sum()),
            "suppression_count": int(role_suppressed.sum()),
            "corrections": corrections,
            "harms": harms,
            "neutral_changes": neutral,
            "net_corrections": corrections - harms,
            "full_metrics": metrics,
            "macro_f1_gain": float(
                metrics["macro_f1"] - keeper_metrics["macro_f1"]
            ),
            "class1_precision_gain": float(
                metrics["per_class"]["1"]["precision"]
                - keeper_metrics["per_class"]["1"]["precision"]
            ),
            "class1_f1_gain": float(
                metrics["per_class"]["1"]["f1"]
                - keeper_metrics["per_class"]["1"]["f1"]
            ),
            "class1_recall_drop": float(
                keeper_metrics["per_class"]["1"]["recall"]
                - metrics["per_class"]["1"]["recall"]
            ),
            "maximum_nonfocus_f1_drop": max(
                float(
                    keeper_metrics["per_class"][str(index)]["f1"]
                    - metrics["per_class"][str(index)]["f1"]
                )
                for index in (0, 2, 3, 4)
            ),
            "fold_aurocs": fold_aurocs,
            "per_fold": fold_rows,
        }
        roles[role] = role_metrics
    return {
        "cohort": {
            "rows": EXPECTED_ROWS,
            "keeper_tp": int(keeper_tp.sum()),
            "focus_fn": int(focus_fn.sum()),
            "restricted_fp": int(restricted_fp.sum()),
        },
        "keeper_full_metrics": keeper_metrics,
        "roles": roles,
        "_full_predictions": full_predictions,
    }


def _role_core_gate_checks(
    role: str,
    *,
    analysis: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> Dict[str, bool]:
    roles = analysis["roles"]
    record = roles[role]
    keeper = roles["keeper_margin_control"]
    controls = (
        roles["global_gap_linear_control"],
        roles["integral_self_only_control"],
    )
    deranged = (
        roles["cap_spatial_deranged_control"],
        roles["cap_cross_sample_context_control"],
    )
    control_fold_wins = sum(
        all(
            float(record["fold_aurocs"][str(fold)])
            > float(control["fold_aurocs"][str(fold)])
            for control in controls
        )
        for fold in range(5)
    )
    return {
        "auroc_min": float(record["tp_vs_restricted_fp_auroc"])
        >= float(thresholds["candidate_auroc_min"]),
        "auroc_gain_vs_keeper": (
            float(record["tp_vs_restricted_fp_auroc"])
            - float(keeper["tp_vs_restricted_fp_auroc"])
            >= float(thresholds["candidate_auroc_gain_vs_keeper_min"])
        ),
        "auroc_gain_vs_strongest_trainable_control": (
            float(record["tp_vs_restricted_fp_auroc"])
            - max(
                float(control["tp_vs_restricted_fp_auroc"])
                for control in controls
            )
            >= float(
                thresholds[
                    "candidate_auroc_gain_vs_strongest_trainable_control_min"
                ]
            )
        ),
        "auroc_gain_vs_each_derangement": all(
            float(record["tp_vs_restricted_fp_auroc"])
            - float(control["tp_vs_restricted_fp_auroc"])
            >= float(
                thresholds[
                    "candidate_auroc_gain_vs_each_derangement_min"
                ]
            )
            for control in deranged
        ),
        "fn_vs_fp_auroc_min": float(
            record["fn_vs_restricted_fp_auroc"]
        )
        >= float(thresholds["fn_vs_restricted_fp_auroc_min"]),
        "fn_vs_fp_gain_vs_keeper": (
            float(record["fn_vs_restricted_fp_auroc"])
            - float(keeper["fn_vs_restricted_fp_auroc"])
            >= float(
                thresholds["fn_vs_restricted_fp_gain_vs_keeper_min"]
            )
        ),
        "control_fold_wins": control_fold_wins
        >= int(thresholds["control_fold_wins_min"]),
        "tp_retention": float(record["tp_retention"])
        >= float(thresholds["tp_retention_min"]),
        "every_fold_tp_retention": min(
            float(row["tp_retention"]) for row in record["per_fold"]
        )
        >= float(thresholds["minimum_fold_tp_retention"]),
        "restricted_fp_rejection": float(
            record["restricted_fp_rejection"]
        )
        >= float(thresholds["restricted_fp_rejection_min"]),
        "every_fold_restricted_fp_rejection": min(
            float(row["restricted_fp_rejection"])
            for row in record["per_fold"]
        )
        >= float(thresholds["minimum_fold_restricted_fp_rejection"]),
        "positive_fp_rejection_every_fold": sum(
            int(row["restricted_fp_rejected"]) > 0
            for row in record["per_fold"]
        )
        >= int(thresholds["minimum_positive_fp_rejection_folds"]),
        "correction_to_harm_ratio": int(record["corrections"])
        >= float(thresholds["correction_to_harm_ratio_min"])
        * int(record["tp_harms"]),
        "net_corrections": int(record["net_corrections"])
        >= int(thresholds["net_corrections_min"]),
        "full_macro_f1_gain": float(record["macro_f1_gain"])
        >= float(thresholds["full_macro_f1_gain_min"]),
        "full_class1_precision_gain": float(
            record["class1_precision_gain"]
        )
        >= float(thresholds["full_class1_precision_gain_min"]),
        "full_class1_f1_gain": float(record["class1_f1_gain"])
        >= float(thresholds["full_class1_f1_gain_min"]),
        "full_class1_recall_drop": float(record["class1_recall_drop"])
        <= float(thresholds["full_class1_recall_drop_max"]),
        "nonfocus_f1_safety": float(
            record["maximum_nonfocus_f1_drop"]
        )
        <= float(thresholds["nonfocus_f1_drop_max"]),
    }


def assess_performance_gates(
    *,
    analysis: Mapping[str, object],
    training: Mapping[str, object],
    inputs: Mapping[str, np.ndarray],
    lock: Mapping[str, object],
) -> Dict[str, object]:
    thresholds = lock["performance_gates"]
    candidate_checks = _role_core_gate_checks(
        "cap_context_candidate",
        analysis=analysis,
        thresholds=thresholds,
    )
    repeat_checks = _role_core_gate_checks(
        "cap_context_seed_repeat",
        analysis=analysis,
        thresholds=thresholds,
    )
    candidate = analysis["roles"]["cap_context_candidate"]
    repeat = analysis["roles"]["cap_context_seed_repeat"]
    keeper_focus = (
        np.asarray(inputs["keeper_probabilities"]).argmax(axis=1)
        == FOCUS_CLASS
    )
    candidate_actions = np.asarray(
        training["suppressed"]["cap_context_candidate"],
        dtype=np.bool_,
    )
    repeat_actions = np.asarray(
        training["suppressed"]["cap_context_seed_repeat"],
        dtype=np.bool_,
    )
    action_agreement = float(
        (
            candidate_actions[keeper_focus]
            == repeat_actions[keeper_focus]
        ).mean()
    )
    repeatability_checks = {
        "seed_repeat_auroc_difference": abs(
            float(candidate["tp_vs_restricted_fp_auroc"])
            - float(repeat["tp_vs_restricted_fp_auroc"])
        )
        <= float(thresholds["repeat_auroc_difference_max"]),
        "seed_repeat_suppression_agreement": action_agreement
        >= float(thresholds["repeat_action_agreement_min"]),
        "seed_repeat_decision_identical": (
            all(candidate_checks.values())
            == all(repeat_checks.values())
        ),
    }
    checks = {
        **{
            f"candidate_{name}": value
            for name, value in candidate_checks.items()
        },
        **repeatability_checks,
    }
    return {
        "checks": checks,
        "candidate_core_checks": candidate_checks,
        "repeat_core_checks": repeat_checks,
        "repeatability": {
            "candidate_auroc": float(
                candidate["tp_vs_restricted_fp_auroc"]
            ),
            "repeat_auroc": float(
                repeat["tp_vs_restricted_fp_auroc"]
            ),
            "absolute_auroc_difference": abs(
                float(candidate["tp_vs_restricted_fp_auroc"])
                - float(repeat["tp_vs_restricted_fp_auroc"])
            ),
            "suppression_agreement": action_agreement,
            "candidate_core_passed": all(candidate_checks.values()),
            "repeat_core_passed": all(repeat_checks.values()),
        },
        "passed": all(checks.values()),
    }


def _normalize_mass_map(value: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(value, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return (array / total).astype(np.float32)


def _resize_map(
    value: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    tensor = torch.as_tensor(
        np.asarray(value, dtype=np.float64)
    )[None, None]
    resized = F.interpolate(
        tensor,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.numpy()


def _project_canonical_map(
    canonical: np.ndarray,
    support_box: Sequence[int],
) -> np.ndarray:
    x0, y0, x1, y1 = [int(value) for value in support_box]
    canvas = np.zeros((FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64)
    canvas[y0:y1, x0:x1] = _resize_map(
        canonical, y1 - y0, x1 - x0
    )
    return _normalize_mass_map(canvas)


def _region_attention_map(
    attention: np.ndarray,
    *,
    region_order: np.ndarray | None = None,
) -> np.ndarray:
    alpha = np.asarray(attention, dtype=np.float64)
    if alpha.shape != (REGION_COUNT, REGION_COUNT):
        raise ValueError("CAP region attention must have shape [27, 27]")
    weights = alpha.mean(axis=0)
    order = (
        np.arange(REGION_COUNT, dtype=np.int64)
        if region_order is None
        else np.asarray(region_order, dtype=np.int64)
    )
    if not np.array_equal(
        np.sort(order), np.arange(REGION_COUNT, dtype=np.int64)
    ):
        raise ValueError("CAP region order is not a permutation")
    canvas = np.zeros((UPSAMPLE_SIZE, UPSAMPLE_SIZE), dtype=np.float64)
    for sequence_position, region_index in enumerate(order.tolist()):
        x0, y0, width, height = [
            int(value) for value in REGION_BOXES[region_index]
        ]
        canvas[y0 : y0 + height, x0 : x0 + width] += (
            float(weights[sequence_position]) / float(width * height)
        )
    return _normalize_mass_map(canvas)


def _heatmap_image(value: np.ndarray, *, size: int) -> Image.Image:
    array = np.asarray(value, dtype=np.float64)
    maximum = max(float(array.max()), 1e-12)
    normalized = np.clip(array / maximum, 0.0, 1.0)
    red = (255.0 * normalized).astype(np.uint8)
    green = (255.0 * np.sqrt(normalized)).astype(np.uint8)
    blue = (80.0 * (1.0 - normalized)).astype(np.uint8)
    image = Image.fromarray(np.stack((red, green, blue), axis=2))
    return image.resize((size, size), Image.Resampling.NEAREST)


def _draw_feature_masks(
    image: Image.Image,
    valid_mask: np.ndarray,
    bbox_mask: np.ndarray,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    scale_x = result.width / FEATURE_SIZE
    scale_y = result.height / FEATURE_SIZE
    for mask, color, width in (
        (valid_mask, (255, 255, 255), 2),
        (bbox_mask, (20, 20, 20), 3),
    ):
        points = np.argwhere(np.asarray(mask, dtype=np.bool_))
        if points.size == 0:
            continue
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        draw.rectangle(
            (
                int(x0 * scale_x),
                int(y0 * scale_y),
                max(int(x1 * scale_x) - 1, int(x0 * scale_x)),
                max(int(y1 * scale_y) - 1, int(y0 * scale_y)),
            ),
            outline=color,
            width=width,
        )
    return result


def _draw_normalized_bbox(
    image: Image.Image,
    normalized_xywh: Sequence[float],
) -> Image.Image:
    result = image.copy()
    center_x, center_y, width, height = [
        float(value) for value in normalized_xywh
    ]
    x0 = np.clip(center_x - width * 0.5, 0.0, 1.0)
    y0 = np.clip(center_y - height * 0.5, 0.0, 1.0)
    x1 = np.clip(center_x + width * 0.5, 0.0, 1.0)
    y1 = np.clip(center_y + height * 0.5, 0.0, 1.0)
    draw = ImageDraw.Draw(result)
    draw.rectangle(
        (
            int(x0 * result.width),
            int(y0 * result.height),
            max(int(x1 * result.width) - 1, int(x0 * result.width)),
            max(int(y1 * result.height) - 1, int(y0 * result.height)),
        ),
        outline=(255, 255, 255),
        width=4,
    )
    draw.rectangle(
        (
            int(x0 * result.width) + 4,
            int(y0 * result.height) + 4,
            max(int(x1 * result.width) - 5, int(x0 * result.width) + 4),
            max(int(y1 * result.height) - 5, int(y0 * result.height) + 4),
        ),
        outline=(20, 20, 20),
        width=1,
    )
    return result


def _visual_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(arrays[name])))
    return digest.hexdigest()


def build_visual_evidence(
    *,
    inputs: Mapping[str, np.ndarray],
    training: Mapping[str, object],
    lock: Mapping[str, object],
    ledger: DataAccessLedger,
    device: torch.device,
    sheet_path: Path | None,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    resolved = _configure_torch(device)
    sample_indices = np.asarray(
        inputs["sample_indices"], dtype=np.int64
    )
    position_by_sample = {
        int(sample_index): position
        for position, sample_index in enumerate(sample_indices.tolist())
    }
    if any(anchor not in position_by_sample for anchor in VISUAL_ANCHORS):
        raise ValueError("A fixed visual anchor is absent from CAP cohort")
    positions = np.asarray(
        [position_by_sample[anchor] for anchor in VISUAL_ANCHORS],
        dtype=np.int64,
    )
    visual_paths = np.asarray(inputs["full_image_paths"])[
        np.asarray(VISUAL_ANCHORS, dtype=np.int64)
    ]
    ledger.authorize_visual_paths(visual_paths.tolist())

    features = training["features_device"]
    support_boxes = np.asarray(inputs["support_boxes"], dtype=np.int64)
    valid_masks = np.asarray(inputs["valid_masks"], dtype=np.bool_)
    bbox_masks = np.asarray(inputs["bbox_masks"], dtype=np.bool_)
    model_boxes = np.asarray(inputs["model_boxes"], dtype=np.float32)
    folds = np.asarray(inputs["folds"], dtype=np.int64)
    targets = np.asarray(inputs["targets"], dtype=np.int64)
    keeper_predictions = np.asarray(
        inputs["keeper_probabilities"]
    ).argmax(axis=1)
    spatial_permutations = np.asarray(
        training["spatial_permutations"], dtype=np.int64
    )
    candidate_scores = np.asarray(
        training["scores"]["cap_context_candidate"], dtype=np.float32
    )
    spatial_scores = np.asarray(
        training["scores"]["cap_spatial_deranged_control"],
        dtype=np.float32,
    )
    cross_scores = np.asarray(
        training["scores"]["cap_cross_sample_context_control"],
        dtype=np.float32,
    )
    candidate_suppressed = np.asarray(
        training["suppressed"]["cap_context_candidate"],
        dtype=np.bool_,
    )
    thresholds = training["thresholds"]["cap_context_candidate"]
    model_bank = training["model_bank"]

    region_maps: List[np.ndarray] = []
    gradient_maps: List[np.ndarray] = []
    spatial_maps: List[np.ndarray] = []
    contrast_maps: List[np.ndarray] = []
    valid_masses: List[float] = []
    bbox_masses: List[float] = []
    reconstructed_scores: List[float] = []
    raw_hashes: List[str] = []
    records: List[Dict[str, object]] = []
    raw_images: List[Image.Image] = []

    active_fold: int | None = None
    candidate_model: nn.Module | None = None
    spatial_model: nn.Module | None = None
    for anchor, position, image_path in zip(
        VISUAL_ANCHORS, positions.tolist(), visual_paths.tolist()
    ):
        fold = int(folds[position])
        if fold != active_fold:
            if candidate_model is not None:
                candidate_model.to(torch.device("cpu"))
            if spatial_model is not None:
                spatial_model.to(torch.device("cpu"))
            candidate_model = model_bank[fold][
                "cap_context_candidate"
            ].to(resolved)
            spatial_model = model_bank[fold][
                "cap_spatial_deranged_control"
            ].to(resolved)
            candidate_model.eval()
            spatial_model.eval()
            for model in (candidate_model, spatial_model):
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            active_fold = fold

        leaf = (
            features[position : position + 1]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        candidate_output = candidate_model(
            leaf,
            support_boxes[position : position + 1],
            return_auxiliary=True,
        )
        candidate_logit, candidate_aux = candidate_output
        gradient = torch.autograd.grad(
            candidate_logit.sum(),
            leaf,
            retain_graph=False,
            create_graph=False,
        )[0]
        attribution = (
            (gradient * leaf).abs().sum(dim=1)[0].detach().cpu().numpy()
        )
        attribution = _normalize_mass_map(attribution)
        attention = (
            candidate_aux["region_attention"][0]
            .detach()
            .cpu()
            .numpy()
        )
        canonical_region = _region_attention_map(attention)
        canonical_region = _resize_map(
            canonical_region, FEATURE_SIZE, FEATURE_SIZE
        )
        projected_region = _project_canonical_map(
            canonical_region, support_boxes[position]
        )

        with torch.no_grad():
            _, spatial_aux = spatial_model(
                features[position : position + 1],
                support_boxes[position : position + 1],
                spatial_permutations=torch.as_tensor(
                    spatial_permutations[position : position + 1],
                    dtype=torch.long,
                    device=resolved,
                ),
                return_auxiliary=True,
            )
        spatial_attention = (
            spatial_aux["region_attention"][0]
            .detach()
            .cpu()
            .numpy()
        )
        canonical_spatial = _region_attention_map(
            spatial_attention,
            region_order=spatial_permutations[position],
        )
        canonical_spatial = _resize_map(
            canonical_spatial, FEATURE_SIZE, FEATURE_SIZE
        )
        projected_spatial = _project_canonical_map(
            canonical_spatial, support_boxes[position]
        )
        contrast = _normalize_mass_map(
            np.abs(projected_region - projected_spatial)
        )
        valid_mass = float(attribution[valid_masks[position]].sum())
        bbox_mass = float(attribution[bbox_masks[position]].sum())

        path = Path(str(image_path))
        raw_hashes.append(_sha256(path))
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
        raw_images.append(rgb)
        reconstructed = float(
            torch.sigmoid(candidate_logit.detach())[0].cpu()
        )
        threshold = float(thresholds[str(fold)]["threshold"])
        record = {
            "sample_index": int(anchor),
            "cohort_position": int(position),
            "outer_fold": fold,
            "target": int(targets[position]),
            "keeper_prediction": int(keeper_predictions[position]),
            "score": float(candidate_scores[position]),
            "reconstructed_score": reconstructed,
            "score_reconstruction_error": abs(
                reconstructed - float(candidate_scores[position])
            ),
            "threshold": threshold,
            "suppressed": bool(candidate_suppressed[position]),
            "spatial_score": float(spatial_scores[position]),
            "cross_sample_score": float(cross_scores[position]),
            "candidate_spatial_score_delta": float(
                candidate_scores[position] - spatial_scores[position]
            ),
            "candidate_cross_sample_score_delta": float(
                candidate_scores[position] - cross_scores[position]
            ),
            "attribution_valid_mass": valid_mass,
            "attribution_bbox_mass": bbox_mass,
            "region_map_std": float(projected_region.std()),
            "gradient_map_std": float(attribution.std()),
            "spatial_contrast_std": float(contrast.std()),
            "raw_image_path": str(path),
            "raw_image_sha256": raw_hashes[-1],
        }
        records.append(record)
        region_maps.append(projected_region)
        gradient_maps.append(attribution)
        spatial_maps.append(projected_spatial)
        contrast_maps.append(contrast)
        valid_masses.append(valid_mass)
        bbox_masses.append(bbox_mass)
        reconstructed_scores.append(reconstructed)

    if candidate_model is not None:
        candidate_model.to(torch.device("cpu"))
    if spatial_model is not None:
        spatial_model.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    arrays = {
        "sample_indices": np.asarray(VISUAL_ANCHORS, dtype=np.int64),
        "cohort_positions": positions,
        "region_maps": np.stack(region_maps).astype(np.float32),
        "gradient_maps": np.stack(gradient_maps).astype(np.float32),
        "spatial_maps": np.stack(spatial_maps).astype(np.float32),
        "spatial_contrast_maps": np.stack(contrast_maps).astype(np.float32),
        "candidate_scores": candidate_scores[positions].astype(np.float32),
        "reconstructed_scores": np.asarray(
            reconstructed_scores, dtype=np.float32
        ),
        "spatial_scores": spatial_scores[positions].astype(np.float32),
        "cross_sample_scores": cross_scores[positions].astype(np.float32),
        "valid_masses": np.asarray(valid_masses, dtype=np.float32),
        "bbox_masses": np.asarray(bbox_masses, dtype=np.float32),
        "suppressed": candidate_suppressed[positions].astype(np.bool_),
    }

    if sheet_path is not None:
        cell = 176
        label_height = 56
        header_height = 34
        columns = (
            "RGB + bbox",
            "CAP region",
            "|grad * feature|",
            "spatial control",
            "absolute contrast",
        )
        sheet = Image.new(
            "RGB",
            (
                len(columns) * cell,
                header_height
                + len(VISUAL_ANCHORS) * (cell + label_height),
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for column, title in enumerate(columns):
            draw.text(
                (column * cell + 5, 10),
                title,
                fill="black",
                font=font,
            )
        for row_index, (position, raw, record) in enumerate(
            zip(positions.tolist(), raw_images, records)
        ):
            top = header_height + row_index * (cell + label_height)
            background = raw.resize(
                (cell, cell), Image.Resampling.LANCZOS
            )
            background = _draw_normalized_bbox(
                background, model_boxes[position]
            )
            panels = [
                background,
                _draw_feature_masks(
                    _heatmap_image(region_maps[row_index], size=cell),
                    valid_masks[position],
                    bbox_masks[position],
                ),
                _draw_feature_masks(
                    _heatmap_image(gradient_maps[row_index], size=cell),
                    valid_masks[position],
                    bbox_masks[position],
                ),
                _draw_feature_masks(
                    _heatmap_image(spatial_maps[row_index], size=cell),
                    valid_masks[position],
                    bbox_masks[position],
                ),
                _draw_feature_masks(
                    _heatmap_image(contrast_maps[row_index], size=cell),
                    valid_masks[position],
                    bbox_masks[position],
                ),
            ]
            for column, panel in enumerate(panels):
                sheet.paste(panel, (column * cell, top))
            label = (
                f"idx={record['sample_index']} fold={record['outer_fold']} "
                f"y={record['target']} keeper={record['keeper_prediction']} "
                f"score={record['score']:.5f} thr={record['threshold']:.5f} "
                f"suppressed={int(record['suppressed'])}\n"
                f"valid={record['attribution_valid_mass']:.4f} "
                f"bbox={record['attribution_bbox_mass']:.4f} "
                f"d_sp={record['candidate_spatial_score_delta']:.5f} "
                f"d_cross={record['candidate_cross_sample_score_delta']:.5f}"
            )
            draw.text(
                (5, top + cell + 4),
                label,
                fill="black",
                font=font,
            )
        sheet.save(sheet_path)

    candidate_all = np.asarray(
        training["scores"]["cap_context_candidate"], dtype=np.float64
    )
    spatial_all = np.asarray(
        training["scores"]["cap_spatial_deranged_control"],
        dtype=np.float64,
    )
    cross_all = np.asarray(
        training["scores"]["cap_cross_sample_context_control"],
        dtype=np.float64,
    )
    lock_visual = lock["visual_gates"]
    checks = {
        "exact_20_rows": len(records) == 20,
        "maps_finite": all(
            bool(np.isfinite(value).all())
            for name, value in arrays.items()
            if "maps" in name
        ),
        "region_maps_nonconstant": all(
            float(value.std()) > 0.0 for value in arrays["region_maps"]
        ),
        "gradient_maps_nonconstant": all(
            float(value.std()) > 0.0
            for value in arrays["gradient_maps"]
        ),
        "spatial_contrast_nonconstant": all(
            float(value.std()) > 0.0
            for value in arrays["spatial_contrast_maps"]
        ),
        "mean_valid_support_mass": float(
            arrays["valid_masses"].mean()
        )
        >= float(lock_visual["valid_support_mass_min"]),
        "mean_bbox_mass": float(arrays["bbox_masses"].mean())
        >= float(lock_visual["bbox_mass_min"]),
        "candidate_differs_from_spatial_derangement": float(
            np.max(np.abs(candidate_all - spatial_all))
        )
        > 1e-7,
        "candidate_differs_from_cross_sample_derangement": float(
            np.max(np.abs(candidate_all - cross_all))
        )
        > 1e-7,
        "every_anchor_differs_from_spatial_derangement": float(
            np.min(
                np.abs(
                    arrays["candidate_scores"]
                    - arrays["spatial_scores"]
                )
            )
        )
        > 1e-7,
        "every_anchor_differs_from_cross_sample_derangement": float(
            np.min(
                np.abs(
                    arrays["candidate_scores"]
                    - arrays["cross_sample_scores"]
                )
            )
        )
        > 1e-7,
        "raw_images_unique_and_stable": len(set(raw_hashes)) == 20,
    }
    evidence = {
        "definition": {
            "feature_attribution": "absolute_gradient_times_frozen_block2_feature_channel_sum",
            "region_map": "mean_received_region_attention_distributed_uniformly_by_region_area",
            "canonical_projection": "bilinear_to_exact_valid_support_then_zero_outside",
            "bbox_gate_map": "feature_attribution",
            "context_difference_epsilon": 1e-7,
        },
        "records": records,
        "mean_attribution_valid_mass": float(
            arrays["valid_masses"].mean()
        ),
        "mean_attribution_bbox_mass": float(
            arrays["bbox_masses"].mean()
        ),
        "candidate_spatial_mean_absolute_score_delta": float(
            np.mean(np.abs(candidate_all - spatial_all))
        ),
        "candidate_cross_sample_mean_absolute_score_delta": float(
            np.mean(np.abs(candidate_all - cross_all))
        ),
        "maximum_score_reconstruction_error": max(
            float(record["score_reconstruction_error"])
            for record in records
        ),
        "visual_arrays_sha256": _visual_arrays_sha256(arrays),
        "checks": checks,
        "automatic_passed": all(checks.values()),
        "manual_review": {
            "required": True,
            "status": "pending",
            "approved": False,
        },
    }
    return evidence, arrays


def _sanitize_fold_protocol(
    records: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for record in records:
        result.append(
            {
                "outer_fold": int(record["outer_fold"]),
                "calibration_fold": int(record["calibration_fold"]),
                "fit_indices": np.asarray(
                    record["fit_indices"], dtype=np.int64
                ).tolist(),
                "calibration_indices": np.asarray(
                    record["calibration_indices"], dtype=np.int64
                ).tolist(),
                "held_indices": np.asarray(
                    record["held_indices"], dtype=np.int64
                ).tolist(),
                "cross_sample_mapping": np.asarray(
                    record["cross_sample_mapping"], dtype=np.int64
                ).tolist(),
                "cross_sample_mapping_sha256": record[
                    "cross_sample_mapping_sha256"
                ],
                "cross_sample_mapping_details": record[
                    "cross_sample_mapping_details"
                ],
                "primary_order_hashes": record[
                    "primary_order_hashes"
                ],
                "repeat_order_hashes": record["repeat_order_hashes"],
                "checks": record["checks"],
            }
        )
    return result


def _prediction_arrays(
    *,
    inputs: Mapping[str, np.ndarray],
    training: Mapping[str, object],
    analysis: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "sample_indices": np.asarray(
            inputs["sample_indices"], dtype=np.int64
        ),
        "targets": np.asarray(inputs["targets"], dtype=np.int64),
        "folds": np.asarray(inputs["folds"], dtype=np.int64),
        "keeper_probabilities": np.asarray(
            inputs["keeper_probabilities"], dtype=np.float32
        ),
    }
    for role in ROLE_NAMES:
        arrays[f"logits.{role}"] = np.asarray(
            training["logits"][role], dtype=np.float32
        )
        arrays[f"scores.{role}"] = np.asarray(
            training["scores"][role], dtype=np.float32
        )
        arrays[f"calibration_logits.{role}"] = np.asarray(
            training["calibration_logits"][role], dtype=np.float32
        )
        arrays[f"calibration_scores.{role}"] = np.asarray(
            training["calibration_scores"][role], dtype=np.float32
        )
        arrays[f"suppressed.{role}"] = np.asarray(
            training["suppressed"][role], dtype=np.bool_
        )
        arrays[f"full_predictions.{role}"] = np.asarray(
            analysis["_full_predictions"][role], dtype=np.int64
        )
    return arrays


def _analysis_for_serialization(
    analysis: Mapping[str, object],
) -> Dict[str, object]:
    return {
        str(name): value
        for name, value in analysis.items()
        if name != "_full_predictions"
    }


def _execute_scientific_run(
    *,
    lock: Mapping[str, object],
    lock_path: Path,
    output_root: Path,
    device: torch.device,
    sheet_path: Path | None,
) -> Dict[str, object]:
    ledger = DataAccessLedger(
        lock,
        lock_path=lock_path,
        excluded_roots=(output_root,),
    )
    with ledger:
        locked_files = verify_locked_files(lock)
        inputs = load_locked_inputs(lock)
        fold_protocol = build_fold_protocol(inputs, lock)
        engine = verify_engine_contract(lock)
        oracles = engineering_oracles()
        resolved = _configure_torch(device)
        process_before = process_snapshot()
        torch.cuda.reset_peak_memory_stats(resolved)
        with PeakResourceMonitor() as monitor:
            training = train_all_roles(
                inputs=inputs,
                fold_protocol=fold_protocol,
                device=resolved,
            )
            analysis = analyze_roles(
                inputs=inputs,
                training=training,
            )
            performance = assess_performance_gates(
                analysis=analysis,
                training=training,
                inputs=inputs,
                lock=lock,
            )
            visual, visual_arrays = build_visual_evidence(
                inputs=inputs,
                training=training,
                lock=lock,
                ledger=ledger,
                device=resolved,
                sheet_path=sheet_path,
            )
        torch.cuda.synchronize(resolved)
        peak_cuda_bytes = int(
            torch.cuda.max_memory_allocated(resolved)
        )
        process_after = process_snapshot()
    ledger_evidence = ledger.snapshot()
    resource_gates = lock["resource_gates"]
    resource_checks = {
        "cuda_peak_below_6_gib": (
            peak_cuda_bytes / (1024**3)
            < float(resource_gates["cuda_peak_gib_max"])
        ),
        "rss_peak_below_10_gib": (
            monitor.peak_rss_bytes / (1024**3)
            < float(resource_gates["rss_peak_gib_max"])
        ),
        "no_unexpected_process_before": not process_before[
            "unexpected_processes"
        ],
        "no_unexpected_process_during": not monitor.unexpected_processes,
        "no_unexpected_process_after": not process_after[
            "unexpected_processes"
        ],
    }
    resource = {
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_gib": float(peak_cuda_bytes / (1024**3)),
        "peak_rss_bytes": int(monitor.peak_rss_bytes),
        "peak_rss_gib": float(monitor.peak_rss_bytes / (1024**3)),
        "peak_virtual_memory_fraction": float(
            monitor.peak_virtual_memory_fraction
        ),
        "process_before": process_before,
        "unexpected_processes_during": list(
            monitor.unexpected_processes.values()
        ),
        "process_after": process_after,
        "checks": resource_checks,
        "passed": all(resource_checks.values()),
    }
    input_checks = {
        str(name): bool(value)
        for name, value in inputs["input_checks"].items()
    }
    fold_checks = all(
        all(record["checks"].values()) for record in fold_protocol
    )
    engineering = {
        "locked_files": locked_files,
        "input_checks": input_checks,
        "fold_checks": fold_checks,
        "engine": engine,
        "oracles": oracles,
        "ledger": ledger_evidence,
        "checks": {
            "locked_files": bool(locked_files["passed"]),
            "inputs": all(input_checks.values()),
            "folds": fold_checks,
            "engine": bool(engine["passed"]),
            "oracles": bool(oracles["passed"]),
            "ledger": bool(ledger_evidence["passed"]),
        },
    }
    engineering["passed"] = all(engineering["checks"].values())
    automatic_checks = {
        "engineering": bool(engineering["passed"]),
        "performance": bool(performance["passed"]),
        "resource": bool(resource["passed"]),
        "visual_automatic": bool(visual["automatic_passed"]),
        "validation_open_count_zero": int(
            ledger_evidence["validation_open_count"]
        )
        == 0,
        "test_open_count_zero": int(
            ledger_evidence["test_open_count"]
        )
        == 0,
    }
    del training["features_device"]
    del training["model_bank"]
    torch.cuda.empty_cache()
    return {
        "locked_files": locked_files,
        "inputs": inputs,
        "fold_protocol": fold_protocol,
        "training": training,
        "analysis": analysis,
        "performance": performance,
        "visual": visual,
        "visual_arrays": visual_arrays,
        "engineering": engineering,
        "resource": resource,
        "ledger": ledger_evidence,
        "automatic_checks": automatic_checks,
        "automatic_passed": all(automatic_checks.values()),
    }


def _prepare_output(path: Path) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"CAP formal output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _manifest_payload(
    output: Path,
    *,
    excluded_names: Sequence[str],
) -> Dict[str, object]:
    excluded = set(excluded_names)
    files = []
    for path in sorted(
        value for value in output.rglob("*") if value.is_file()
    ):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload: Dict[str, object] = {
        "schema_version": 1,
        "excluded_names": sorted(excluded),
        "files": files,
    }
    payload["files_sha256"] = json_sha256(files)
    return payload


def _verify_manifest(
    output: Path,
    manifest_name: str,
) -> Dict[str, object]:
    manifest_path = output / manifest_name
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = {
        str(value) for value in payload.get("excluded_names", ())
    }
    excluded_errors = sorted(
        value
        for value in excluded
        if (
            Path(value).is_absolute()
            or ".." in Path(value).parts
            or "/" in value
            or "\\" in value
        )
    )
    checks = []
    expected_paths: List[str] = []
    record_errors: List[str] = []
    for record in payload["files"]:
        relative = str(record["path"])
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative_path.as_posix() != relative
        ):
            record_errors.append(relative)
            checks.append(False)
            continue
        expected_paths.append(relative)
        path = output / relative_path
        checks.append(
            path.is_file()
            and path.stat().st_size == int(record["bytes"])
            and _sha256(path) == record["sha256"]
        )
    observed_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.relative_to(output).as_posix() not in excluded
    }
    expected_path_set = set(expected_paths)
    duplicate_paths = len(expected_paths) != len(expected_path_set)
    exact_file_set = observed_paths == expected_path_set
    schema_exact = payload.get("schema_version") == 1
    files_digest_exact = (
        payload.get("files_sha256") == json_sha256(payload["files"])
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "file_count": len(checks),
        "missing_files": sorted(expected_path_set - observed_paths),
        "unexpected_files": sorted(observed_paths - expected_path_set),
        "record_errors": record_errors,
        "excluded_name_errors": excluded_errors,
        "duplicate_paths": duplicate_paths,
        "exact_file_set": exact_file_set,
        "schema_exact": schema_exact,
        "files_digest_exact": files_digest_exact,
        "passed": bool(
            checks
            and all(checks)
            and not record_errors
            and not excluded_errors
            and not duplicate_paths
            and exact_file_set
            and schema_exact
            and files_digest_exact
        ),
    }


def run_formal(
    *,
    output: Path = DEFAULT_OUTPUT,
    lock_path: Path = LOCK_PATH,
    expected_commit: str,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, object]:
    lock_path = Path(lock_path).resolve()
    lock = load_lock(lock_path)
    preflight = run_preflight(
        lock_path=lock_path,
        expected_commit=expected_commit,
        device=device,
        include_resource=True,
    )
    if not preflight["passed"]:
        raise RuntimeError("CAP formal preflight failed")
    output = _prepare_output(output)
    started = time.perf_counter()
    run = _execute_scientific_run(
        lock=lock,
        lock_path=lock_path,
        output_root=output,
        device=device,
        sheet_path=output / "xai_contact_sheet.png",
    )
    elapsed = time.perf_counter() - started
    repository = repository_state(
        lock, expected_commit=expected_commit
    )
    if not repository["passed"]:
        raise RuntimeError("Repository state changed during CAP formal")

    predictions = _prediction_arrays(
        inputs=run["inputs"],
        training=run["training"],
        analysis=run["analysis"],
    )
    np.savez(output / "formal_predictions.npz", **predictions)
    np.savez(
        output / "formal_states.npz",
        **run["training"]["state_arrays"],
    )
    np.savez(output / "visual_arrays.npz", **run["visual_arrays"])
    _write_json(
        output / "fold_protocol.json",
        _sanitize_fold_protocol(run["fold_protocol"]),
    )
    _write_json(
        output / "fold_thresholds.json",
        run["training"]["thresholds"],
    )
    _write_json(
        output / "training_records.json",
        run["training"]["training_records"],
    )
    _write_jsonl(
        output / "training_trace.jsonl",
        run["training"]["trace_rows"],
    )
    _write_json(
        output / "analysis.json",
        _analysis_for_serialization(run["analysis"]),
    )
    _write_json(
        output / "performance_gates.json",
        run["performance"],
    )
    _write_json(
        output / "visual_evidence.json",
        run["visual"],
    )
    _write_json(
        output / "engineering_checks.json",
        run["engineering"],
    )
    _write_json(
        output / "resource_evidence.json",
        run["resource"],
    )
    _write_json(output / "access_ledger.json", run["ledger"])
    _write_json(output / "preflight_evidence.json", preflight)
    summary = {
        "protocol_id": lock["protocol_id"],
        "state": "formal_complete_manual_visual_review_pending",
        "process_id": os.getpid(),
        "repository": repository,
        "lock_sha256": _sha256(lock_path),
        "engine_sha256": _sha256(
            REPO_ROOT
            / "trkh"
            / "tools"
            / "cap_integral_region_context_a0_engine.py"
        ),
        "auditor_sha256": _sha256(MODULE_PATH),
        "elapsed_seconds": float(elapsed),
        "automatic_checks": run["automatic_checks"],
        "automatic_passed": bool(run["automatic_passed"]),
        "manual_visual_review_required": True,
        "manual_visual_review_status": "pending",
        "overall_passed": False,
        "performance_passed": bool(run["performance"]["passed"]),
        "visual_automatic_passed": bool(
            run["visual"]["automatic_passed"]
        ),
        "state_arrays_sha256": run["training"][
            "state_arrays_sha256"
        ],
        "visual_arrays_sha256": run["visual"][
            "visual_arrays_sha256"
        ],
        "access_ledger_sha256": run["ledger"][
            "ordered_events_sha256"
        ],
        "downstream_authorization": {
            "trainer_integration": False,
            "validation_smoke": False,
            "validation": False,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_command_update": False,
        },
    }
    _write_json(output / "summary.json", summary)
    formal_manifest = _manifest_payload(
        output,
        excluded_names=(
            "formal_manifest.json",
            "replay_summary.json",
            "replay_manifest.json",
            "manual_visual_review.json",
            "final_decision.json",
            "artifact_set_manifest.json",
        ),
    )
    _write_json(output / "formal_manifest.json", formal_manifest)
    manifest_check = _verify_manifest(output, "formal_manifest.json")
    if not manifest_check["passed"]:
        raise RuntimeError("CAP formal artifact manifest failed")
    return {
        "output": str(output),
        "summary": summary,
        "manifest": manifest_check,
    }


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }


def _maximum_array_error(
    reference: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
) -> Tuple[float, bool]:
    if set(reference) != set(observed):
        return float("inf"), False
    maximum = 0.0
    exact_discrete = True
    for name in reference:
        left = np.asarray(reference[name])
        right = np.asarray(observed[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            return float("inf"), False
        if left.dtype.kind in {"b", "i", "u", "U", "S"}:
            exact_discrete = exact_discrete and np.array_equal(left, right)
        else:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            left.astype(np.float64)
                            - right.astype(np.float64)
                        ),
                        initial=0.0,
                    )
                ),
            )
    return maximum, exact_discrete


def _recursive_numeric_difference(
    left: object,
    right: object,
) -> Tuple[float, bool]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf"), False
        maximum = 0.0
        exact = True
        for key in left:
            difference, child_exact = _recursive_numeric_difference(
                left[key], right[key]
            )
            maximum = max(maximum, difference)
            exact = exact and child_exact
        return maximum, exact
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf"), False
        maximum = 0.0
        exact = True
        for left_item, right_item in zip(left, right):
            difference, child_exact = _recursive_numeric_difference(
                left_item, right_item
            )
            maximum = max(maximum, difference)
            exact = exact and child_exact
        return maximum, exact
    boolean_types = (bool, np.bool_)
    integer_types = (int, np.integer)
    floating_types = (float, np.floating)
    left_boolean = isinstance(left, boolean_types)
    right_boolean = isinstance(right, boolean_types)
    if left_boolean or right_boolean:
        return (
            0.0,
            left_boolean
            and right_boolean
            and bool(left) == bool(right),
        )
    left_integer = isinstance(left, integer_types)
    right_integer = isinstance(right, integer_types)
    if left_integer or right_integer:
        if left_integer and right_integer:
            difference = abs(int(left) - int(right))
            return float(difference), difference == 0
        if (
            left_integer
            and isinstance(right, floating_types)
        ) or (
            right_integer
            and isinstance(left, floating_types)
        ):
            return abs(float(left) - float(right)), False
        return 0.0, False
    if isinstance(left, floating_types) and isinstance(
        right, floating_types
    ):
        return abs(float(left) - float(right)), True
    return (0.0, type(left) is type(right) and left == right)


def replay_formal(
    *,
    output: Path = DEFAULT_OUTPUT,
    lock_path: Path = LOCK_PATH,
    expected_commit: str,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, object]:
    output = Path(output).resolve()
    lock_path = Path(lock_path).resolve()
    for name in (
        "replay_summary.json",
        "replay_manifest.json",
        "manual_visual_review.json",
        "final_decision.json",
        "artifact_set_manifest.json",
    ):
        if (output / name).exists():
            raise FileExistsError(
                f"CAP replay/finalization artifact already exists: {name}"
            )
    manifest = _verify_manifest(output, "formal_manifest.json")
    if not manifest["passed"]:
        raise RuntimeError("CAP formal manifest is not intact")
    reference = {
        "summary": json.loads(
            (output / "summary.json").read_text(encoding="utf-8")
        ),
        "predictions": _read_npz(output / "formal_predictions.npz"),
        "states": _read_npz(output / "formal_states.npz"),
        "visual_arrays": _read_npz(output / "visual_arrays.npz"),
        "analysis": json.loads(
            (output / "analysis.json").read_text(encoding="utf-8")
        ),
        "thresholds": json.loads(
            (output / "fold_thresholds.json").read_text(
                encoding="utf-8"
            )
        ),
        "training_records": json.loads(
            (output / "training_records.json").read_text(
                encoding="utf-8"
            )
        ),
        "performance": json.loads(
            (output / "performance_gates.json").read_text(
                encoding="utf-8"
            )
        ),
        "visual": json.loads(
            (output / "visual_evidence.json").read_text(
                encoding="utf-8"
            )
        ),
        "ledger": json.loads(
            (output / "access_ledger.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    lock = load_lock(lock_path)
    fresh_process = int(reference["summary"]["process_id"]) != os.getpid()
    if not fresh_process:
        raise RuntimeError("CAP replay must run in a fresh process")
    repository = repository_state(
        lock, expected_commit=expected_commit
    )
    if not repository["passed"]:
        raise RuntimeError("CAP replay repository state differs")
    started = time.perf_counter()
    replay = _execute_scientific_run(
        lock=lock,
        lock_path=lock_path,
        output_root=output,
        device=device,
        sheet_path=None,
    )
    elapsed = time.perf_counter() - started
    observed_predictions = _prediction_arrays(
        inputs=replay["inputs"],
        training=replay["training"],
        analysis=replay["analysis"],
    )
    prediction_error, prediction_discrete = _maximum_array_error(
        reference["predictions"], observed_predictions
    )
    state_error, state_discrete = _maximum_array_error(
        reference["states"], replay["training"]["state_arrays"]
    )
    visual_error, visual_discrete = _maximum_array_error(
        reference["visual_arrays"], replay["visual_arrays"]
    )
    analysis_error, analysis_exact = _recursive_numeric_difference(
        reference["analysis"],
        _analysis_for_serialization(replay["analysis"]),
    )
    threshold_error, threshold_exact = _recursive_numeric_difference(
        reference["thresholds"], replay["training"]["thresholds"]
    )
    training_error, training_exact = _recursive_numeric_difference(
        reference["training_records"],
        replay["training"]["training_records"],
    )
    performance_error, performance_exact = _recursive_numeric_difference(
        reference["performance"], replay["performance"]
    )
    visual_metric_error, visual_metric_exact = (
        _recursive_numeric_difference(
            reference["visual"], replay["visual"]
        )
    )
    ledger_exact = (
        reference["ledger"]["ordered_events_sha256"]
        == replay["ledger"]["ordered_events_sha256"]
        and reference["ledger"]["events"] == replay["ledger"]["events"]
    )
    tolerance = float(lock["replay"]["numeric_error_max"])
    checks = {
        "fresh_process": fresh_process,
        "formal_manifest_intact": bool(manifest["passed"]),
        "prediction_numeric_error": prediction_error <= tolerance,
        "prediction_discrete_exact": prediction_discrete,
        "state_numeric_error": state_error <= tolerance,
        "state_discrete_exact": state_discrete,
        "analysis_numeric_error": analysis_error <= tolerance,
        "analysis_discrete_exact": analysis_exact,
        "threshold_numeric_error": threshold_error <= tolerance,
        "threshold_discrete_exact": threshold_exact,
        "training_numeric_error": training_error <= tolerance,
        "training_discrete_exact": training_exact,
        "performance_numeric_error": performance_error <= tolerance,
        "performance_discrete_exact": performance_exact,
        "visual_numeric_array_error": visual_error <= tolerance,
        "visual_discrete_array_exact": visual_discrete,
        "visual_metric_error": visual_metric_error <= tolerance,
        "visual_metric_discrete_exact": visual_metric_exact,
        "ledger_exact": ledger_exact,
        "validation_open_count_zero": int(
            replay["ledger"]["validation_open_count"]
        )
        == 0,
        "test_open_count_zero": int(
            replay["ledger"]["test_open_count"]
        )
        == 0,
    }
    summary = {
        "protocol_id": lock["protocol_id"],
        "state": "fresh_process_replay_complete",
        "elapsed_seconds": float(elapsed),
        "formal_manifest": manifest,
        "repository": repository,
        "numeric_errors": {
            "predictions": prediction_error,
            "states": state_error,
            "analysis": analysis_error,
            "thresholds": threshold_error,
            "training": training_error,
            "performance": performance_error,
            "visual_arrays": visual_error,
            "visual_metrics": visual_metric_error,
        },
        "replay_access_ledger_sha256": replay["ledger"][
            "ordered_events_sha256"
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json(output / "replay_summary.json", summary)
    replay_manifest = _manifest_payload(
        output,
        excluded_names=(
            "replay_manifest.json",
            "manual_visual_review.json",
            "final_decision.json",
            "artifact_set_manifest.json",
        ),
    )
    _write_json(output / "replay_manifest.json", replay_manifest)
    replay_manifest_check = _verify_manifest(
        output, "replay_manifest.json"
    )
    if not replay_manifest_check["passed"]:
        raise RuntimeError("CAP replay artifact manifest failed")
    return summary


def finalize_visual_review(
    *,
    output: Path = DEFAULT_OUTPUT,
    approved: bool,
    notes: str,
) -> Dict[str, object]:
    output = Path(output).resolve()
    review_notes = str(notes).strip()
    if not review_notes:
        raise ValueError("CAP manual visual review notes are required")
    for name in (
        "manual_visual_review.json",
        "final_decision.json",
        "artifact_set_manifest.json",
    ):
        if (output / name).exists():
            raise FileExistsError(
                f"CAP finalization artifact already exists: {name}"
            )
    formal_manifest = _verify_manifest(output, "formal_manifest.json")
    replay_manifest = _verify_manifest(output, "replay_manifest.json")
    if not formal_manifest["passed"]:
        raise RuntimeError("CAP formal manifest is not intact at finalization")
    if not replay_manifest["passed"]:
        raise RuntimeError("CAP replay manifest is not intact at finalization")
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (output / "replay_summary.json").read_text(encoding="utf-8")
    )
    visual = json.loads(
        (output / "visual_evidence.json").read_text(encoding="utf-8")
    )
    sheet = output / "xai_contact_sheet.png"
    if not sheet.is_file():
        raise FileNotFoundError("CAP XAI contact sheet is absent")
    review = {
        "reviewed_rows": len(visual["records"]),
        "required_rows": len(VISUAL_ANCHORS),
        "approved": bool(approved),
        "notes": review_notes,
        "contact_sheet_sha256": _sha256(sheet),
        "visual_arrays_sha256": visual["visual_arrays_sha256"],
        "passed": bool(
            approved and len(visual["records"]) == len(VISUAL_ANCHORS)
        ),
    }
    _write_json(output / "manual_visual_review.json", review)
    all_passed = bool(
        summary["automatic_passed"]
        and replay["passed"]
        and review["passed"]
    )
    decision = {
        "protocol_id": summary["protocol_id"],
        "automatic_passed": bool(summary["automatic_passed"]),
        "replay_passed": bool(replay["passed"]),
        "manual_visual_review_passed": bool(review["passed"]),
        "formal_manifest_verified": bool(formal_manifest["passed"]),
        "formal_manifest_sha256": formal_manifest["manifest_sha256"],
        "replay_manifest_verified": bool(replay_manifest["passed"]),
        "replay_manifest_sha256": replay_manifest["manifest_sha256"],
        "all_gates_passed": all_passed,
        "decision": (
            "authorize_separately_locked_default_off_validation_smoke"
            if all_passed
            else "reject_exact_cap_a0"
        ),
        "downstream_authorization": {
            "trainer_integration": all_passed,
            "separately_locked_validation_smoke": all_passed,
            "validation": False,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_command_update": False,
        },
    }
    _write_json(output / "final_decision.json", decision)
    artifact_manifest = _manifest_payload(
        output,
        excluded_names=("artifact_set_manifest.json",),
    )
    _write_json(
        output / "artifact_set_manifest.json", artifact_manifest
    )
    artifact_check = _verify_manifest(
        output, "artifact_set_manifest.json"
    )
    if not artifact_check["passed"]:
        raise RuntimeError("CAP final artifact-set manifest failed")
    return decision


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the locked train-only CAP integral-region A0"
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "formal", "replay", "finalize"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=LOCK_PATH,
    )
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--skip-resource",
        action="store_true",
        help="Unit-test helper for preflight only; formal never skips it",
    )
    parser.add_argument("--approve-visual", action="store_true")
    parser.add_argument("--visual-notes", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if args.mode == "preflight":
        result = run_preflight(
            lock_path=args.lock,
            expected_commit=args.expected_commit,
            device=device,
            include_resource=not args.skip_resource,
        )
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if args.mode in {"formal", "replay"} and not args.expected_commit:
        raise ValueError(
            "--expected-commit is required for formal/replay"
        )
    if args.mode == "formal":
        result = run_formal(
            output=args.output,
            lock_path=args.lock,
            expected_commit=args.expected_commit,
            device=device,
        )
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    if args.mode == "replay":
        result = replay_formal(
            output=args.output,
            lock_path=args.lock,
            expected_commit=args.expected_commit,
            device=device,
        )
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0 if result["passed"] else 3
    result = finalize_visual_review(
        output=args.output,
        approved=bool(args.approve_visual),
        notes=args.visual_notes,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

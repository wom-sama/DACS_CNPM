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
import tempfile
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
from sklearn.model_selection import StratifiedGroupKFold

EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXPECTED_CUBLAS_WORKSPACE_CONFIG

import torch
from torch import Tensor

from trkh.tools.restricted_negative_learnable_isda_a0_engine import (
    BASE_SEED,
    CLASS_COUNT,
    CLEAN_EPOCHS,
    COVNET_LR,
    COVNET_REPEAT_SEED_OFFSET,
    CovariancePredictor,
    ELIGIBLE_CLASSES,
    EPOCHS,
    FEATURE_DIM,
    FOCUS_CLASS,
    HEAD_LR,
    HEAD_MOMENTUM,
    LAMBDA_0,
    META_BATCH_SIZE,
    META_INTERVAL,
    META_ROLES,
    META_SAMPLES_PER_CLASS,
    PARTITION_BATCH_SIZE,
    ROLE_NAMES,
    array_sha256,
    balanced_meta_slot_count,
    build_balanced_meta_schedule,
    build_outer_holdout_swap_donors,
    build_same_label_source_derangement,
    cosine_nearest_proxy,
    covariance_state_arrays,
    draw_semantic_feature,
    initialize_linear_head,
    isda_logits,
    population_diagonal_variance,
    predict_covariance,
    stable_seed,
    state_arrays_sha256,
    string_sequence_sha256,
    train_role_fold,
)


MODULE_PATH = Path(__file__).resolve()
_ROOT_CANDIDATE = MODULE_PATH.parents[2]
if not (_ROOT_CANDIDATE / "trkh").is_dir():
    _ROOT_CANDIDATE = Path(
        os.environ.get("TRKH_REPO_ROOT", r"D:\DataAI\AIEx\TRKH")
    )
REPO_ROOT = _ROOT_CANDIDATE.resolve()
LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_RESTRICTED_NEGATIVE_LEARNABLE_ISDA_A0_LOCK_20260724.json"
)
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


_ACCESS_AUDIT_LOCK = threading.RLock()
_ACTIVE_DATA_ACCESS_LEDGER: Optional["DataAccessLedger"] = None
_DATA_ACCESS_AUDIT_HOOK_INSTALLED = False
_DATA_ACCESS_AUDIT_PROBE_SEEN = False


def _normalized_audit_path(value: object) -> Optional[str]:
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


def _audit_path_components(path: str) -> Tuple[str, ...]:
    return tuple(
        part
        for part in path.replace("/", "\\").casefold().split("\\")
        if part
    )


def _data_access_audit_hook(event: str, arguments: Tuple[object, ...]) -> None:
    global _DATA_ACCESS_AUDIT_PROBE_SEEN
    if event == "trkh.rn_lisda.data_access_probe":
        _DATA_ACCESS_AUDIT_PROBE_SEEN = True
        return
    if event != "open":
        return
    with _ACCESS_AUDIT_LOCK:
        ledger = _ACTIVE_DATA_ACCESS_LEDGER
    if ledger is not None:
        ledger.observe_open(arguments)


def _ensure_data_access_audit_hook() -> None:
    global _DATA_ACCESS_AUDIT_HOOK_INSTALLED
    global _DATA_ACCESS_AUDIT_PROBE_SEEN
    with _ACCESS_AUDIT_LOCK:
        if not _DATA_ACCESS_AUDIT_HOOK_INSTALLED:
            _DATA_ACCESS_AUDIT_PROBE_SEEN = False
            sys.addaudithook(_data_access_audit_hook)
            sys.audit("trkh.rn_lisda.data_access_probe")
            if not _DATA_ACCESS_AUDIT_PROBE_SEEN:
                raise RuntimeError("RN-LISDA data-access audit hook probe failed")
            _DATA_ACCESS_AUDIT_HOOK_INSTALLED = True


class DataAccessLedger:
    def __init__(
        self,
        lock: Mapping[str, object],
        *,
        excluded_roots: Sequence[Path] = (),
    ) -> None:
        policy = lock.get("data_access_audit")
        if not isinstance(policy, Mapping):
            raise ValueError("RN-LISDA data-access audit lock is missing")
        roots = tuple(
            path
            for path in (
                _normalized_audit_path(value)
                for value in policy.get("roots", ())
            )
            if path is not None
        )
        if len(roots) != 2:
            raise ValueError("RN-LISDA data-access roots are incomplete")
        self.roots = roots
        self.dataset_root = roots[0]
        self.runs_root = roots[1]
        self.excluded_roots = tuple(
            path
            for path in (
                _normalized_audit_path(value) for value in excluded_roots
            )
            if path is not None
        )
        self.forbidden_components = frozenset(
            str(value).casefold()
            for value in policy.get("forbidden_complete_components", ())
        )
        if self.forbidden_components != {
            "val",
            "valid",
            "validation",
            "test",
        }:
            raise ValueError("RN-LISDA forbidden split components drifted")
        immutable = lock.get("immutable_inputs")
        if not isinstance(immutable, Mapping):
            raise ValueError("RN-LISDA immutable inputs are missing")
        allowed_paths = set()
        for record in immutable.values():
            if not isinstance(record, Mapping) or "path" not in record:
                continue
            raw_path = Path(str(record["path"])).expanduser()
            if not raw_path.is_absolute():
                raw_path = REPO_ROOT / raw_path
            normalized = _normalized_audit_path(raw_path)
            if normalized is not None:
                allowed_paths.add(normalized)
        self.allowed_exact_paths = frozenset(allowed_paths)
        data_yaml_record = immutable.get("yolo_data_yaml")
        if data_yaml_record is None:
            data_yaml_record = immutable.get("data_yaml")
        if not isinstance(data_yaml_record, Mapping):
            raise ValueError("RN-LISDA locked data YAML is missing")
        raw_data_yaml = Path(str(data_yaml_record["path"])).expanduser()
        if not raw_data_yaml.is_absolute():
            raw_data_yaml = REPO_ROOT / raw_data_yaml
        locked_dataset_root = _normalized_audit_path(raw_data_yaml.parent)
        if locked_dataset_root is None or not _path_is_within(
            locked_dataset_root, self.dataset_root
        ):
            raise ValueError("RN-LISDA locked dataset root is invalid")
        self.locked_dataset_root = locked_dataset_root
        self.allowed_dataset_train_roots = tuple(
            path
            for path in (
                _normalized_audit_path(
                    Path(locked_dataset_root) / category / "train"
                )
                for category in ("images", "labels")
            )
            if path is not None
        )
        if len(self.allowed_dataset_train_roots) != 2:
            raise ValueError("RN-LISDA train-only dataset roots are invalid")
        self.events: List[Dict[str, object]] = []
        self.blocked_attempts: List[Dict[str, object]] = []
        self._entered = False

    @staticmethod
    def _write_requested(mode: object, flags: object) -> bool:
        mode_text = "" if mode is None else str(mode).casefold()
        if any(marker in mode_text for marker in ("w", "a", "x", "+")):
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

    def _is_excluded(self, path: str) -> bool:
        return any(_path_is_within(path, root) for root in self.excluded_roots)

    def _domain(self, path: str) -> Optional[str]:
        if _path_is_within(path, self.dataset_root):
            return "dataset"
        if _path_is_within(path, self.runs_root):
            return "runs"
        return None

    def _dataset_path_allowed(self, path: str) -> bool:
        if path in self.allowed_exact_paths:
            return True
        return any(
            _path_is_within(path, root)
            for root in self.allowed_dataset_train_roots
        )

    def observe_open(self, arguments: Tuple[object, ...]) -> None:
        if not arguments:
            return
        path = _normalized_audit_path(arguments[0])
        if path is None or self._is_excluded(path):
            return
        domain = self._domain(path)
        if domain is None:
            return
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else None
        parts = _audit_path_components(path)
        forbidden = sorted(
            set(parts).intersection(self.forbidden_components)
        )
        reason: Optional[str] = None
        if forbidden:
            reason = "forbidden_split_component"
        elif self._write_requested(mode, flags):
            reason = "write_to_locked_data_domain"
        elif domain == "runs" and path not in self.allowed_exact_paths:
            reason = "undeclared_runs_artifact"
        elif domain == "dataset" and not self._dataset_path_allowed(path):
            reason = "undeclared_dataset_artifact"
        record = {
            "sequence": len(self.events),
            "domain": domain,
            "path": path,
            "mode": None if mode is None else str(mode),
            "flags": None if not isinstance(flags, int) else int(flags),
            "decision": "blocked" if reason else "allowed",
            "reason": reason,
            "forbidden_components": forbidden,
        }
        self.events.append(record)
        if reason is not None:
            self.blocked_attempts.append(dict(record))
            raise PermissionError(
                f"RN-LISDA data-access audit blocked {reason}: {path}"
            )

    def __enter__(self) -> "DataAccessLedger":
        global _ACTIVE_DATA_ACCESS_LEDGER
        _ensure_data_access_audit_hook()
        with _ACCESS_AUDIT_LOCK:
            if _ACTIVE_DATA_ACCESS_LEDGER is not None:
                raise RuntimeError("A data-access ledger is already active")
            _ACTIVE_DATA_ACCESS_LEDGER = self
            self._entered = True
        return self

    def __exit__(self, *args: object) -> None:
        global _ACTIVE_DATA_ACCESS_LEDGER
        with _ACCESS_AUDIT_LOCK:
            if _ACTIVE_DATA_ACCESS_LEDGER is self:
                _ACTIVE_DATA_ACCESS_LEDGER = None
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
            key = str(row["path"])
            path_counts[key] = path_counts.get(key, 0) + 1
        forbidden_counts = {
            name: sum(
                int(name in row["forbidden_components"]) for row in events
            )
            for name in sorted(self.forbidden_components)
        }
        return {
            "schema_version": 1,
            "hook_installed": bool(_DATA_ACCESS_AUDIT_HOOK_INSTALLED),
            "hook_probe_seen": bool(_DATA_ACCESS_AUDIT_PROBE_SEEN),
            "roots": list(self.roots),
            "excluded_roots": list(self.excluded_roots),
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
                bool(_DATA_ACCESS_AUDIT_HOOK_INSTALLED)
                and bool(_DATA_ACCESS_AUDIT_PROBE_SEEN)
                and len(events) > 0
                and not self.blocked_attempts
            ),
        }


def _runtime_versions() -> Dict[str, object]:
    device_name = (
        str(torch.cuda.get_device_name(0))
        if torch.cuda.is_available()
        else None
    )
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": metadata.version("numpy"),
        "torch": metadata.version("torch"),
        "torch_cuda": str(torch.version.cuda),
        "scikit_learn": metadata.version("scikit-learn"),
        "pillow": metadata.version("Pillow"),
        "psutil": metadata.version("psutil"),
        "device_name": device_name,
    }


def _verify_runtime_lock(lock: Mapping[str, object]) -> Dict[str, object]:
    expected = lock.get("runtime")
    if not isinstance(expected, Mapping):
        raise ValueError("RN-LISDA runtime lock is missing")
    observed = _runtime_versions()
    expected_normalized = {str(key): value for key, value in expected.items()}
    if set(observed) != set(expected_normalized):
        raise ValueError(
            "RN-LISDA runtime lock fields differ: "
            f"{sorted(observed)} != {sorted(expected_normalized)}"
        )
    checks = {
        key: str(observed[key]) == str(expected_normalized[key])
        for key in observed
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "RN-LISDA runtime differs from lock: "
            f"{failed}; observed={observed}"
        )
    return {
        "expected": expected_normalized,
        "observed": observed,
        "checks": checks,
        "passed": True,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _repository_state(root: Path, protected: object) -> Dict[str, object]:
    if isinstance(protected, Mapping):
        protected_hash_lock = {
            str(path).replace("\\", "/"): str(expected)
            for path, expected in protected.items()
        }
    elif isinstance(protected, Sequence) and not isinstance(
        protected, (str, bytes)
    ):
        protected_hash_lock = {
            str(path).replace("\\", "/"): "" for path in protected
        }
    else:
        raise TypeError("Protected untracked lock must be a mapping or list")
    status_lines = [
        line
        for line in _run_git(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        ).split("\0")
        if line
    ]
    tracked_dirty = [line for line in status_lines if not line.startswith("?? ")]
    untracked = sorted(
        line[3:].replace("\\", "/")
        for line in status_lines
        if line.startswith("?? ")
    )
    expected_untracked = sorted(protected_hash_lock)
    protected_hashes = {}
    protected_hashes_exact = True
    resolved_root = root.resolve()
    for relative_path, expected_sha256 in protected_hash_lock.items():
        path = (resolved_root / relative_path).resolve()
        try:
            path.relative_to(resolved_root)
            inside_root = True
        except ValueError:
            inside_root = False
        observed_sha256 = (
            _sha256(path) if inside_root and path.is_file() else None
        )
        passed = bool(
            inside_root
            and path.is_file()
            and (
                not expected_sha256
                or str(observed_sha256) == str(expected_sha256)
            )
        )
        protected_hashes[relative_path] = {
            "expected_sha256": expected_sha256 or None,
            "observed_sha256": observed_sha256,
            "inside_repository": inside_root,
            "passed": passed,
        }
        protected_hashes_exact &= passed
    head = _run_git(root, "rev-parse", "HEAD")
    upstream = _run_git(root, "rev-parse", "@{u}")
    return {
        "head": head,
        "upstream": upstream,
        "head_equals_upstream": head == upstream,
        "tracked_dirty": tracked_dirty,
        "tracked_clean": not tracked_dirty,
        "untracked": untracked,
        "protected_untracked": expected_untracked,
        "protected_untracked_exact": untracked == expected_untracked,
        "protected_untracked_hashes": protected_hashes,
        "protected_untracked_hashes_exact": protected_hashes_exact,
        "passed": (
            head == upstream
            and not tracked_dirty
            and untracked == expected_untracked
            and protected_hashes_exact
        ),
    }


def _resolve_locked_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _verify_file(
    root: Path, record: Mapping[str, object], *, label: str
) -> Dict[str, object]:
    path = _resolve_locked_path(root, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked {label}: {path}")
    observed = _sha256(path)
    expected = str(record["sha256"])
    if observed != expected:
        raise ValueError(
            f"Locked {label} SHA mismatch: {observed} != {expected}"
        )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": observed,
        "passed": True,
    }


def _verify_external_sources(
    lock: Mapping[str, object],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    sources = lock["external_sources"]
    assert isinstance(sources, Mapping)
    for name in (
        "isda_neurips_2019",
        "metasaug_cvpr_2021",
        "learnable_isda_tip_2024",
    ):
        source = sources[name]
        assert isinstance(source, Mapping)
        paper = Path(str(source["local_paper"])).resolve()
        observed_paper = _sha256(paper)
        if observed_paper != str(source["paper_sha256"]):
            raise ValueError(f"{name} paper SHA mismatch")
        entry: Dict[str, object] = {
            "paper": str(paper),
            "paper_sha256": observed_paper,
            "passed": True,
        }
        if "official_repository" in source:
            repository = Path(str(source["official_repository"])).resolve()
            commit = _run_git(repository, "rev-parse", "HEAD")
            tree = _run_git(repository, "rev-parse", "HEAD^{tree}")
            if commit != str(source["repository_commit"]):
                raise ValueError(f"{name} repository commit mismatch")
            if tree != str(source["repository_tree"]):
                raise ValueError(f"{name} repository tree mismatch")
            entry.update(
                {
                    "repository": str(repository),
                    "commit": commit,
                    "tree": tree,
                }
            )
            reference_hashes = source.get("reference_file_sha256", {})
            if isinstance(reference_hashes, Mapping):
                verified_references = {}
                for relative, expected in reference_hashes.items():
                    path = repository / str(relative)
                    observed = _sha256(path)
                    if observed != str(expected):
                        raise ValueError(
                            f"{name} reference SHA mismatch: {relative}"
                        )
                    verified_references[str(relative)] = observed
                entry["reference_file_sha256"] = verified_references
        result[name] = entry
    return result


def _verify_engine_lock(lock: Mapping[str, object]) -> Dict[str, bool]:
    optimization = lock["optimization"]
    equation = lock["equation"]
    head_sgd = optimization["head_sgd"]
    covnet_sgd = optimization["covnet_sgd"]
    lr_schedule = optimization["lr_schedule"]
    lambda_schedule = optimization["lambda_schedule"]
    meta_schedule = optimization["meta_schedule"]
    covnet = equation["covnet"]
    mechanism_gate = lock["gates"]["mechanism"]
    visual = lock["visual_anchors"]
    explicit_draw = visual["explicit_draw"]
    nearest_proxy = visual["nearest_proxy"]
    locked_anchor_indices = tuple(
        int(value) for value in visual["ordered_sample_indices"]
    )
    checks = {
        "roles_exact": tuple(lock["roles"]) == tuple(ROLE_NAMES),
        "seed_exact": int(optimization["seed"]) == BASE_SEED,
        "covnet_repeat_seed_offset_exact": int(
            optimization["covnet_repeat_seed_offset"]
        )
        == COVNET_REPEAT_SEED_OFFSET,
        "epochs_exact": int(optimization["epochs"]) == EPOCHS,
        "clean_epochs_exact": int(optimization["clean_epochs"])
        == CLEAN_EPOCHS,
        "semantic_epochs_exact": int(optimization["semantic_epochs"])
        == EPOCHS - CLEAN_EPOCHS,
        "partition_batch_size_exact": int(
            optimization["partition_batch_size"]
        )
        == PARTITION_BATCH_SIZE,
        "paired_batch_size_exact": int(
            optimization["total_paired_batch_size"]
        )
        == 2 * PARTITION_BATCH_SIZE,
        "drop_last_exact": bool(optimization["drop_last"]),
        "head_optimizer_exact": (
            float(head_sgd["initial_lr"]) == HEAD_LR
            and float(head_sgd["momentum"]) == HEAD_MOMENTUM
            and float(head_sgd["dampening"]) == 0.0
            and float(head_sgd["weight_decay"]) == 0.0
            and not bool(head_sgd["nesterov"])
        ),
        "covnet_optimizer_exact": (
            float(covnet_sgd["initial_lr"]) == COVNET_LR
            and float(covnet_sgd["momentum"]) == 0.0
            and float(covnet_sgd["weight_decay"]) == 0.0
        ),
        "lr_schedule_exact": (
            int(lr_schedule["warmup_epochs"]) == 5
            and str(lr_schedule["post_warmup"]) == "cosine"
            and int(lr_schedule["horizon_epochs"]) == EPOCHS
            and bool(lr_schedule["same_lr_for_both_exchange_directions"])
        ),
        "lambda_schedule_exact": (
            float(lambda_schedule["lambda_0"]) == LAMBDA_0
            and float(lambda_schedule["first_semantic_value"]) == 0.25
            and float(lambda_schedule["final_value"]) == LAMBDA_0
        ),
        "meta_schedule_exact": (
            int(meta_schedule["paired_iteration_interval"]) == META_INTERVAL
            and bool(meta_schedule["continuous_across_semantic_epochs"])
            and int(meta_schedule["first_semantic_paired_iteration"]) == 0
            and bool(meta_schedule["updates_both_exchange_directions"])
            and int(meta_schedule["balanced_meta_batch_size"])
            == META_BATCH_SIZE
            and int(meta_schedule["balanced_meta_rows_per_class"])
            == META_SAMPLES_PER_CLASS
            and bool(meta_schedule["balanced_meta_objective_only"])
            and str(meta_schedule["real_head_batch_distribution"])
            == "natural partition order"
            and str(meta_schedule["meta_partition"])
            == "opposite source-disjoint inner partition"
            and str(meta_schedule["sampler"])
            == "deterministic rotating-class-priority least-used seeded-permutation source-unique"
        ),
        "balanced_meta_gate_exact": (
            tuple(
                int(value)
                for value in mechanism_gate[
                    "balanced_meta_class_counts_exact"
                ]
            )
            == (META_SAMPLES_PER_CLASS,) * CLASS_COUNT
            and bool(mechanism_gate["balanced_meta_rows_unique"])
            and bool(mechanism_gate["balanced_meta_sources_unique"])
            and int(
                mechanism_gate[
                    "balanced_meta_max_row_use_spread_per_class"
                ]
            )
            == 1
            and int(
                mechanism_gate["pseudo_meta_source_overlap_count_max"]
            )
            == 0
        ),
        "holdout_epoch_exact": int(
            optimization["outer_holdout_score_epoch"]
        )
        == EPOCHS,
        "selection_disabled_exact": (
            not bool(optimization["early_stopping"])
            and not bool(optimization["best_epoch_selection"])
            and not bool(optimization["class_weights"])
            and not bool(optimization["global_class1_oversampling"])
        ),
        "equation_focus_exact": (
            int(equation["focus_class"]) == FOCUS_CLASS
            and tuple(int(value) for value in equation["eligible_true_classes"])
            == tuple(ELIGIBLE_CLASSES)
            and tuple(int(value) for value in equation["clean_true_classes"])
            == (1, 3)
            and int(equation["only_rival_column"]) == FOCUS_CLASS
        ),
        "covnet_shape_exact": (
            int(covnet["input_dim"]) == FEATURE_DIM
            and int(covnet["hidden_dim"]) == FEATURE_DIM // 2
            and int(covnet["hidden_layers"]) == 1
            and str(covnet["hidden_activation"]) == "relu"
            and int(covnet["output_dim"]) == FEATURE_DIM
            and str(covnet["output_activation"]) == "2*sigmoid"
            and str(covnet["output_weight_init"]) == "zeros"
            and str(covnet["output_bias_init"]) == "zeros"
            and bool(covnet["input_detached"])
        ),
        "device_and_dtype_exact": (
            str(optimization["device"]) == "cuda"
            and str(optimization["dtype"]) == "float32"
        ),
        "determinism_exact": (
            bool(optimization["deterministic_algorithms"])
            and str(optimization["cublas_workspace_config"])
            == EXPECTED_CUBLAS_WORKSPACE_CONFIG
            and not bool(optimization["cuda_matmul_tf32"])
            and not bool(optimization["cudnn_tf32"])
        ),
        "visual_anchor_contract_exact": (
            locked_anchor_indices == tuple(VISUAL_ANCHORS)
            and str(visual["ordered_indices_sha256"])
            == array_sha256(np.asarray(VISUAL_ANCHORS, dtype=np.int64))
            and int(visual["rows"]) == len(VISUAL_ANCHORS)
            and bool(visual["selection_is_metric_independent"])
            and int(visual["fold_1_true_4_fallback"]) == 5213
            and bool(visual["class1_augmentation_disabled"])
            and len(visual["held_swap_anchor_donors"])
            == len(VISUAL_ANCHORS)
            and {
                int(value)
                for value in visual["held_swap_anchor_donors"].keys()
            }
            == set(VISUAL_ANCHORS)
        ),
        "visual_draw_contract_exact": (
            str(explicit_draw["dtype"]) == "float32"
            and str(explicit_draw["rng"]) == "numpy.default_rng"
            and int(explicit_draw["fixed_classwise_draws"]) == 1
            and int(explicit_draw["candidate_draws"]) == 2
            and int(explicit_draw["deranged_candidate_draws"]) == 1
            and bool(explicit_draw["deranged_reuses_candidate_draw_0_epsilon"])
            and str(nearest_proxy["pool"]) == "real outer-fit rows only"
            and str(nearest_proxy["distance"])
            == "cosine distance on L2-normalized frozen embeddings"
            and bool(nearest_proxy["exclude_anchor_source"])
            and not bool(nearest_proxy["label_filter"])
            and str(nearest_proxy["tie_break"]) == "ascending sample_index"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Engine/lock contract mismatch: {failed}")
    return checks


def verify_locked_inputs(
    lock_path: Path = LOCK_PATH,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    lock_path = Path(lock_path).expanduser().resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("lock_state") != "prospective_pre_candidate":
        raise ValueError("RN-LISDA lock is not prospective")
    runtime = _verify_runtime_lock(lock)
    engine_contract = _verify_engine_lock(lock)
    protocol_record = lock["protocol"]
    protocol = _resolve_locked_path(REPO_ROOT, str(protocol_record["path"]))
    protocol_sha = _sha256(protocol)
    if protocol_sha != str(protocol_record["sha256"]):
        raise ValueError("RN-LISDA protocol SHA mismatch")
    benchmark_record = lock.get("engineering_benchmark")
    if not isinstance(benchmark_record, Mapping):
        raise ValueError("RN-LISDA engineering benchmark lock is missing")
    benchmark = _verify_file(
        REPO_ROOT,
        benchmark_record,
        label="engineering_benchmark",
    )
    immutable = lock["immutable_inputs"]
    assert isinstance(immutable, Mapping)
    verified = {
        name: _verify_file(REPO_ROOT, record, label=name)
        for name, record in immutable.items()
        if isinstance(record, Mapping) and "path" in record and "sha256" in record
    }
    protected = lock.get("protected_untracked", [])
    if not isinstance(protected, (list, Mapping)):
        raise ValueError("Protected untracked lock must be a mapping or list")
    repository = _repository_state(REPO_ROOT, protected)
    if not repository["passed"]:
        raise ValueError(
            "TRKH repository must be clean, pushed, and preserve protected "
            f"untracked paths: {repository}"
        )
    external = _verify_external_sources(lock)
    return lock, {
        "lock_path": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "protocol_path": str(protocol),
        "protocol_sha256": protocol_sha,
        "engineering_benchmark": benchmark,
        "immutable_inputs": verified,
        "repository": repository,
        "external_sources": external,
        "runtime": runtime,
        "engine_contract": engine_contract,
        "passed": True,
    }


def _contains_forbidden_split(path: str) -> bool:
    parts = [
        part
        for part in str(path).replace("/", "\\").casefold().split("\\")
        if part
    ]
    return any(part in {"val", "valid", "validation", "test"} for part in parts)


def _is_train_image_path(path: str) -> bool:
    parts = [
        part
        for part in str(path).replace("/", "\\").casefold().split("\\")
        if part
    ]
    return not _contains_forbidden_split(path) and any(
        parts[index : index + 2] == ["images", "train"]
        for index in range(max(0, len(parts) - 1))
    )


def load_locked_train_cache(path: Path) -> Dict[str, np.ndarray]:
    resolved = Path(path).expanduser().resolve()
    if _contains_forbidden_split(str(resolved)):
        raise ValueError(f"Forbidden cache split marker: {resolved}")
    with np.load(resolved, allow_pickle=True) as payload:
        if tuple(payload.files) != EXPECTED_CACHE_KEYS:
            raise ValueError(f"Cache keys/order mismatch: {payload.files}")
        cache = {key: np.asarray(payload[key]).copy() for key in payload.files}
    expected_shapes = {
        "embeddings": (EXPECTED_ROWS, FEATURE_DIM),
        "probabilities": (EXPECTED_ROWS, CLASS_COUNT),
        "labels": (EXPECTED_ROWS,),
        "base_predictions": (EXPECTED_ROWS,),
        "paths": (EXPECTED_ROWS,),
        "sample_index": (EXPECTED_ROWS,),
    }
    for key, shape in expected_shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"Cache shape mismatch for {key}: {cache[key].shape}")
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    probabilities = np.asarray(cache["probabilities"], dtype=np.float32)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    predictions = np.asarray(cache["base_predictions"], dtype=np.int64)
    paths = np.asarray(cache["paths"]).astype(str)
    indices = np.asarray(cache["sample_index"], dtype=np.int64)
    if not np.isfinite(embeddings).all() or not np.isfinite(probabilities).all():
        raise ValueError("Cache contains non-finite values")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=0.0
    ):
        raise ValueError("Cache probabilities are invalid")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise ValueError("Cache prediction mismatch")
    if not np.array_equal(indices, np.arange(EXPECTED_ROWS, dtype=np.int64)):
        raise ValueError("Cache sample order is incomplete")
    if tuple(np.bincount(labels, minlength=CLASS_COUNT)) != EXPECTED_CLASS_COUNTS:
        raise ValueError("Cache class counts differ from lock")
    if any(not _is_train_image_path(path) for path in paths.tolist()):
        raise ValueError("Cache contains non-train image path")
    return {
        "embeddings": embeddings,
        "probabilities": probabilities,
        "labels": labels,
        "base_predictions": predictions,
        "paths": paths,
        "sample_index": indices,
    }


def load_locked_cidt_folds(
    cache: Mapping[str, np.ndarray], path: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if _contains_forbidden_split(str(resolved)):
        raise ValueError(f"Forbidden CIDT split marker: {resolved}")
    folds = np.full(EXPECTED_ROWS, -1, dtype=np.int64)
    sources = np.empty(EXPECTED_ROWS, dtype=object)
    observed = np.zeros(EXPECTED_ROWS, dtype=bool)
    anchor_rows: Dict[int, Dict[str, object]] = {}
    with resolved.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row["condition"]).strip().casefold() != "clean":
                continue
            index = int(row["sample_index"])
            if index < 0 or index >= EXPECTED_ROWS or observed[index]:
                raise ValueError(f"Invalid CIDT sample index: {index}")
            path_value = str(row["image_path"])
            if not _is_train_image_path(path_value):
                raise ValueError(f"CIDT contains non-train path: {path_value}")
            if int(row["target_index"]) != int(cache["labels"][index]):
                raise ValueError(f"CIDT/cache label mismatch at {index}")
            if Path(path_value).name.casefold() != Path(
                str(cache["paths"][index])
            ).name.casefold():
                raise ValueError(f"CIDT/cache path mismatch at {index}")
            folds[index] = int(row["fold"])
            sources[index] = str(row["source_stem"]).casefold()
            observed[index] = True
            if index in VISUAL_ANCHORS:
                anchor_rows[index] = {
                    "sample_index": index,
                    "fold": int(row["fold"]),
                    "target_index": int(row["target_index"]),
                    "source_stem": str(row["source_stem"]).casefold(),
                    "image_path": path_value,
                    "keeper_prediction": int(row["keeper_prediction"]),
                    "keeper_focus_probability": float(
                        row["keeper_focus_probability"]
                    ),
                }
    if not observed.all():
        raise ValueError(f"CIDT clean rows incomplete: {observed.sum()}")
    if tuple(np.bincount(folds, minlength=5)) != EXPECTED_FOLD_COUNTS:
        raise ValueError("CIDT fold counts differ from lock")
    if len(anchor_rows) != len(VISUAL_ANCHORS):
        raise ValueError("Locked visual anchors are incomplete")
    source_array = np.asarray(sources).astype(str)
    fold_rows = []
    overlap_total = 0
    for fold in range(5):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        overlap = set(source_array[fit]).intersection(source_array[held])
        overlap_total += len(overlap)
        fold_rows.append(
            {
                "outer_fold": fold,
                "fit_rows": int(len(fit)),
                "held_rows": int(len(held)),
                "held_class_counts": np.bincount(
                    cache["labels"][held], minlength=CLASS_COUNT
                ).tolist(),
                "source_overlap": int(len(overlap)),
            }
        )
    if overlap_total != 0:
        raise ValueError("CIDT outer folds leak source stems")
    return folds, source_array, {
        "assignment_sha256": array_sha256(folds),
        "folds": fold_rows,
        "source_overlap": 0,
        "declared_train_image_rows": int(observed.sum()),
        "forbidden_split_path_count": 0,
        "visual_anchor_rows": [
            anchor_rows[index] for index in VISUAL_ANCHORS
        ],
    }


def build_locked_partitions(
    labels: np.ndarray,
    folds: np.ndarray,
    sources: np.ndarray,
    lock: Mapping[str, object],
) -> List[Dict[str, object]]:
    expected_rows = lock["inner_partition_rule"]["partitions"]
    expected_by_fold = {
        int(row["outer_fold"]): row for row in expected_rows
    }
    if len(expected_rows) != 5 or set(expected_by_fold) != set(range(5)):
        raise ValueError("Partition lock must contain exactly folds 0..4")
    result = []
    for outer_fold in range(5):
        expected = expected_by_fold[outer_fold]
        fit = np.flatnonzero(folds != outer_fold).astype(np.int64)
        held = np.flatnonzero(folds == outer_fold).astype(np.int64)
        splitter = StratifiedGroupKFold(
            n_splits=2,
            shuffle=True,
            random_state=BASE_SEED + outer_fold,
        )
        b_relative, a_relative = next(
            splitter.split(fit, labels[fit], groups=sources[fit])
        )
        a = fit[a_relative].astype(np.int64)
        b = fit[b_relative].astype(np.int64)
        assignment = np.full(EXPECTED_ROWS, -1, dtype=np.int8)
        assignment[a] = 0
        assignment[b] = 1
        source_overlap = set(sources[a]).intersection(sources[b])
        deranged_a_map = build_same_label_source_derangement(
            a,
            labels,
            sources,
            outer_fold=outer_fold,
            partition_name="a",
        )
        deranged_b_map = build_same_label_source_derangement(
            b,
            labels,
            sources,
            outer_fold=outer_fold,
            partition_name="b",
        )
        deranged_a = deranged_a_map[a]
        deranged_b = deranged_b_map[b]
        held_donors = build_outer_holdout_swap_donors(
            fit,
            held,
            labels,
            sources,
            outer_fold=outer_fold,
        )
        steps = min(
            len(a) // PARTITION_BATCH_SIZE,
            len(b) // PARTITION_BATCH_SIZE,
        )
        meta_slots = balanced_meta_slot_count(steps)
        balanced_meta_a = build_balanced_meta_schedule(
            a,
            labels,
            sources,
            outer_fold=outer_fold,
            partition_name="a",
            slot_count=meta_slots,
        )
        balanced_meta_b = build_balanced_meta_schedule(
            b,
            labels,
            sources,
            outer_fold=outer_fold,
            partition_name="b",
            slot_count=meta_slots,
        )
        observed = {
            "outer_fold": outer_fold,
            "fit_rows": int(len(fit)),
            "a_rows": int(len(a)),
            "b_rows": int(len(b)),
            "a_class_counts": np.bincount(
                labels[a], minlength=CLASS_COUNT
            ).tolist(),
            "b_class_counts": np.bincount(
                labels[b], minlength=CLASS_COUNT
            ).tolist(),
            "source_overlap": int(len(source_overlap)),
            "assignment_sha256": array_sha256(assignment),
            "a_indices_sha256": array_sha256(a),
            "b_indices_sha256": array_sha256(b),
            "a_sources_sha256": string_sequence_sha256(sources[a]),
            "b_sources_sha256": string_sequence_sha256(sources[b]),
            "a_deranged_donors_sha256": array_sha256(deranged_a),
            "a_deranged_pairs_sha256": array_sha256(
                np.column_stack([a, deranged_a]).astype(np.int64)
            ),
            "b_deranged_donors_sha256": array_sha256(deranged_b),
            "b_deranged_pairs_sha256": array_sha256(
                np.column_stack([b, deranged_b]).astype(np.int64)
            ),
            "held_indices_sha256": array_sha256(held),
            "held_swap_donors_sha256": array_sha256(held_donors),
            "balanced_meta_slot_count": int(meta_slots),
            "a_balanced_meta_schedule_sha256": array_sha256(
                balanced_meta_a
            ),
            "b_balanced_meta_schedule_sha256": array_sha256(
                balanced_meta_b
            ),
        }
        if set(expected) != set(observed):
            raise ValueError(
                f"Partition lock fields differ for fold {outer_fold}"
            )
        for key, value in observed.items():
            if key in expected and value != expected[key]:
                raise ValueError(
                    f"Partition lock mismatch fold={outer_fold} key={key}: "
                    f"{value} != {expected[key]}"
                )
        result.append(
            {
                **observed,
                "fit_indices": fit,
                "held_indices": held,
                "partition_a": a,
                "partition_b": b,
                "held_swap_donors": held_donors,
            }
        )
    return result


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
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
    targets: np.ndarray, probabilities: np.ndarray
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (targets.size, CLASS_COUNT):
        raise ValueError("Probability shape does not match targets")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("Probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Probability row sum is non-positive")
    probabilities = probabilities / row_sums
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(CLASS_COUNT),
        zero_division=0,
    )
    predicted = np.bincount(predictions, minlength=CLASS_COUNT)
    per_class = [
        {
            "class_index": class_index,
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
            "support": int(support[class_index]),
            "predicted_support": int(predicted[class_index]),
        }
        for class_index in range(CLASS_COUNT)
    ]
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[targets]
    restricted = np.isin(targets, ELIGIBLE_CLASSES)
    return {
        "samples": int(targets.size),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=np.arange(CLASS_COUNT)
        ).tolist(),
        "nll": float(
            log_loss(targets, probabilities, labels=np.arange(CLASS_COUNT))
        ),
        "brier": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "ece_15": _ece(targets, probabilities, bins=15),
        "class1_tp": int(
            np.sum((targets == FOCUS_CLASS) & (predictions == FOCUS_CLASS))
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
    restricted = np.isin(targets, ELIGIBLE_CLASSES)
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
                & candidate_tp
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
    targets: np.ndarray,
    folds: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    if set(outputs) != set(ROLE_NAMES):
        raise ValueError("Output roles differ from lock")
    metrics = {
        role: classification_metrics(targets, outputs[role])
        for role in ROLE_NAMES
    }
    fold_metrics: Dict[str, List[Dict[str, object]]] = {}
    for role in ROLE_NAMES:
        rows = []
        for fold in range(5):
            mask = folds == fold
            rows.append(
                {
                    "outer_fold": fold,
                    **classification_metrics(targets[mask], outputs[role][mask]),
                }
            )
        fold_metrics[role] = rows
    transitions: Dict[str, object] = {}
    for role in ROLE_NAMES:
        if role == "ce_control":
            continue
        transitions[f"{role}_vs_ce"] = transition_stats(
            targets, outputs["ce_control"], outputs[role]
        )
    for role in ROLE_NAMES:
        if role == "rn_lisda_candidate":
            continue
        transitions[f"candidate_vs_{role}"] = transition_stats(
            targets, outputs[role], outputs["rn_lisda_candidate"]
        )
    agreement = float(
        np.mean(
            outputs["rn_lisda_candidate"].argmax(axis=1)
            == outputs["rn_lisda_covnet_seed_repeat"].argmax(axis=1)
        )
    )
    return {
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "transitions": transitions,
        "candidate_repeat_prediction_agreement": agreement,
    }


def _focus(metrics: Mapping[str, object]) -> Mapping[str, object]:
    per_class = metrics["per_class"]
    assert isinstance(per_class, list)
    value = per_class[FOCUS_CLASS]
    assert isinstance(value, Mapping)
    return value


def _role_vs_ce_checks(
    *,
    role: str,
    analysis: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> Dict[str, bool]:
    metrics = analysis["metrics"]
    fold_metrics = analysis["fold_metrics"]
    transitions = analysis["transitions"]
    assert isinstance(metrics, Mapping)
    assert isinstance(fold_metrics, Mapping)
    assert isinstance(transitions, Mapping)
    ce = metrics["ce_control"]
    candidate = metrics[role]
    assert isinstance(ce, Mapping) and isinstance(candidate, Mapping)
    ce_focus = _focus(ce)
    candidate_focus = _focus(candidate)
    transition = transitions[f"{role}_vs_ce"]
    assert isinstance(transition, Mapping)
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
            float(candidate_focus["precision"]) - float(ce_focus["precision"])
            >= float(thresholds["class1_precision_gain_min"])
        ),
        "class1_f1_gain": (
            float(candidate_focus["f1"]) - float(ce_focus["f1"])
            >= float(thresholds["class1_f1_gain_min"])
        ),
        "class1_recall_floor": (
            float(candidate_focus["recall"])
            >= float(thresholds["class1_recall_min"])
        ),
        "class1_recall_budget": (
            float(ce_focus["recall"]) - float(candidate_focus["recall"])
            <= float(thresholds["class1_recall_loss_max"])
        ),
        "restricted_fp_net_removal": (
            int(transition["restricted_fp_net_removal"])
            >= int(thresholds["restricted_fp_net_removal_min"])
        ),
        "class1_tp_net": (
            int(transition["class1_tp_net"])
            >= int(thresholds["class1_tp_net_min"])
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
    candidate_folds = fold_metrics[role]
    assert isinstance(ce_folds, list) and isinstance(candidate_folds, list)
    precision_nonworse = 0
    f1_improved = 0
    maximum_recall_loss = -math.inf
    minimum_tp_net = math.inf
    for ce_fold, candidate_fold in zip(ce_folds, candidate_folds):
        ce_fold_focus = _focus(ce_fold)
        candidate_fold_focus = _focus(candidate_fold)
        if float(candidate_fold_focus["precision"]) >= float(
            ce_fold_focus["precision"]
        ):
            precision_nonworse += 1
        if float(candidate_fold_focus["f1"]) > float(ce_fold_focus["f1"]):
            f1_improved += 1
        maximum_recall_loss = max(
            maximum_recall_loss,
            float(ce_fold_focus["recall"])
            - float(candidate_fold_focus["recall"]),
        )
        minimum_tp_net = min(
            minimum_tp_net,
            int(candidate_fold["class1_tp"]) - int(ce_fold["class1_tp"]),
        )
    fold_thresholds = analysis["_fold_thresholds"]
    assert isinstance(fold_thresholds, Mapping)
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
            "fold_recall_safety": (
                maximum_recall_loss
                <= float(fold_thresholds["per_fold_recall_loss_max"])
            ),
            "fold_tp_safety": (
                minimum_tp_net
                >= int(fold_thresholds["per_fold_class1_tp_net_min"])
            ),
        }
    )
    return checks


def assess_performance_gates(
    analysis: Mapping[str, object],
    lock: Mapping[str, object],
) -> Dict[str, object]:
    gate_lock = lock["gates"]
    assert isinstance(gate_lock, Mapping)
    metrics = analysis["metrics"]
    transitions = analysis["transitions"]
    assert isinstance(metrics, Mapping) and isinstance(transitions, Mapping)
    ce = metrics["ce_control"]
    classwise = metrics["restricted_classwise_isda"]
    candidate = metrics["rn_lisda_candidate"]
    repeat = metrics["rn_lisda_covnet_seed_repeat"]
    assert all(
        isinstance(value, Mapping)
        for value in (ce, classwise, candidate, repeat)
    )
    ce_focus = _focus(ce)
    classwise_focus = _focus(classwise)
    candidate_focus = _focus(candidate)
    repeat_focus = _focus(repeat)

    compatibility_thresholds = gate_lock["control_compatibility"]
    assert isinstance(compatibility_thresholds, Mapping)
    classwise_transition = transitions["restricted_classwise_isda_vs_ce"]
    assert isinstance(classwise_transition, Mapping)
    compatibility = {
        "ce_macro_f1": (
            float(ce["macro_f1"])
            >= float(compatibility_thresholds["ce_macro_f1_min"])
        ),
        "ce_class1_f1": (
            float(ce_focus["f1"])
            >= float(compatibility_thresholds["ce_class1_f1_min"])
        ),
        "classwise_precision_gain": (
            float(classwise_focus["precision"]) - float(ce_focus["precision"])
            >= float(
                compatibility_thresholds[
                    "classwise_class1_precision_gain_vs_ce_min"
                ]
            )
        ),
        "classwise_restricted_fp_removal": (
            int(classwise_transition["restricted_fp_net_removal"])
            >= int(
                compatibility_thresholds[
                    "classwise_restricted_fp_net_removal_vs_ce_min"
                ]
            )
        ),
    }

    analysis_with_thresholds = dict(analysis)
    analysis_with_thresholds["_fold_thresholds"] = gate_lock["fold_stability"]
    candidate_checks = _role_vs_ce_checks(
        role="rn_lisda_candidate",
        analysis=analysis_with_thresholds,
        thresholds=gate_lock["candidate_vs_ce"],
    )
    repeat_checks = _role_vs_ce_checks(
        role="rn_lisda_covnet_seed_repeat",
        analysis=analysis_with_thresholds,
        thresholds=gate_lock["candidate_vs_ce"],
    )

    recovery_thresholds = gate_lock["candidate_vs_classwise"]
    assert isinstance(recovery_thresholds, Mapping)
    candidate_transition = transitions["rn_lisda_candidate_vs_ce"]
    assert isinstance(candidate_transition, Mapping)
    classwise_precision_gain = float(classwise_focus["precision"]) - float(
        ce_focus["precision"]
    )
    classwise_fp_removal = int(
        classwise_transition["restricted_fp_net_removal"]
    )

    def classwise_recovery(
        role_metrics: Mapping[str, object],
        role_transition: Mapping[str, object],
    ) -> Dict[str, bool]:
        role_focus = _focus(role_metrics)
        return {
            "recall_recovery": (
                float(role_focus["recall"]) - float(classwise_focus["recall"])
                >= float(recovery_thresholds["class1_recall_gain_min"])
            ),
            "f1_recovery": (
                float(role_focus["f1"]) - float(classwise_focus["f1"])
                >= float(recovery_thresholds["class1_f1_gain_min"])
            ),
            "precision_gain_preserved": (
                float(role_focus["precision"]) - float(ce_focus["precision"])
                >= float(
                    recovery_thresholds[
                        "preserve_fraction_of_precision_gain_min"
                    ]
                )
                * classwise_precision_gain
            ),
            "fp_removal_preserved": (
                int(role_transition["restricted_fp_net_removal"])
                >= float(
                    recovery_thresholds[
                        "preserve_fraction_of_restricted_fp_net_removal_min"
                    ]
                )
                * classwise_fp_removal
            ),
        }

    candidate_recovery = classwise_recovery(
        candidate, candidate_transition
    )
    repeat_transition = transitions["rn_lisda_covnet_seed_repeat_vs_ce"]
    assert isinstance(repeat_transition, Mapping)
    repeat_recovery = classwise_recovery(repeat, repeat_transition)

    causal_thresholds = gate_lock["causal_controls"]
    assert isinstance(causal_thresholds, Mapping)
    restricted_all = metrics["restricted_all_rival_meta"]
    full_all = metrics["full_all_class_meta"]
    deranged = metrics["rn_lisda_deranged_input"]
    joint = metrics["rn_lisda_joint_no_meta"]
    assert all(
        isinstance(value, Mapping)
        for value in (restricted_all, full_all, deranged, joint)
    )
    causal = {
        "restricted_all_rival_class1_f1": (
            float(candidate_focus["f1"]) - float(_focus(restricted_all)["f1"])
            >= float(
                causal_thresholds[
                    "class1_f1_gain_vs_restricted_all_rival_min"
                ]
            )
        ),
        "restricted_all_rival_macro_safety": (
            float(restricted_all["macro_f1"]) - float(candidate["macro_f1"])
            <= float(
                causal_thresholds[
                    "macro_loss_vs_restricted_all_rival_max"
                ]
            )
        ),
        "full_all_class_class1_f1": (
            float(candidate_focus["f1"]) - float(_focus(full_all)["f1"])
            >= float(
                causal_thresholds["class1_f1_gain_vs_full_all_class_min"]
            )
        ),
        "full_all_class_macro_safety": (
            float(full_all["macro_f1"]) - float(candidate["macro_f1"])
            <= float(
                causal_thresholds["macro_loss_vs_full_all_class_max"]
            )
        ),
        "deranged_precision": (
            float(candidate_focus["precision"])
            - float(_focus(deranged)["precision"])
            >= float(
                causal_thresholds[
                    "class1_precision_gain_vs_deranged_min"
                ]
            )
        ),
        "deranged_class1_f1": (
            float(candidate_focus["f1"]) - float(_focus(deranged)["f1"])
            >= float(causal_thresholds["class1_f1_gain_vs_deranged_min"])
        ),
        "joint_no_meta_class1_f1": (
            float(candidate_focus["f1"]) - float(_focus(joint)["f1"])
            >= float(
                causal_thresholds["class1_f1_gain_vs_joint_no_meta_min"]
            )
        ),
    }

    repeat_thresholds = gate_lock["repeat"]
    assert isinstance(repeat_thresholds, Mapping)
    repeat_gate = {
        "repeat_primary_safety": all(repeat_checks.values()),
        "repeat_classwise_recovery": all(repeat_recovery.values()),
        "prediction_agreement": (
            float(analysis["candidate_repeat_prediction_agreement"])
            >= float(repeat_thresholds["prediction_agreement_min"])
        ),
        "macro_f1_difference": (
            abs(float(candidate["macro_f1"]) - float(repeat["macro_f1"]))
            <= float(
                repeat_thresholds["macro_f1_absolute_difference_max"]
            )
        ),
        "class1_f1_difference": (
            abs(float(candidate_focus["f1"]) - float(repeat_focus["f1"]))
            <= float(
                repeat_thresholds["class1_f1_absolute_difference_max"]
            )
        ),
        "class1_precision_difference": (
            abs(
                float(candidate_focus["precision"])
                - float(repeat_focus["precision"])
            )
            <= float(
                repeat_thresholds[
                    "class1_precision_absolute_difference_max"
                ]
            )
        ),
        "class1_recall_difference": (
            abs(
                float(candidate_focus["recall"])
                - float(repeat_focus["recall"])
            )
            <= float(
                repeat_thresholds["class1_recall_absolute_difference_max"]
            )
        ),
    }
    groups = {
        "control_compatibility": compatibility,
        "candidate_vs_ce_and_fold": candidate_checks,
        "candidate_vs_classwise": candidate_recovery,
        "causal_controls": causal,
        "paired_covnet_seed_repeat": repeat_gate,
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


def _flatten_role_states(
    results: Mapping[Tuple[str, int], Mapping[str, object]]
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for (role, fold), result in sorted(results.items()):
        head_state = result["head_state"]
        covnet_state = result["covnet_state"]
        assert isinstance(head_state, Mapping)
        assert isinstance(covnet_state, Mapping)
        for name, value in head_state.items():
            arrays[f"{role}__fold{fold}__head__{name}"] = np.asarray(value)
        for name, value in covnet_state.items():
            arrays[f"{role}__fold{fold}__covnet__{name}"] = np.asarray(value)
    return arrays


def _initial_state_evidence(
    lock: Mapping[str, object],
) -> List[Dict[str, object]]:
    optimization = lock["optimization"]
    if int(optimization["covnet_repeat_seed_offset"]) != int(
        COVNET_REPEAT_SEED_OFFSET
    ):
        raise ValueError("CovNet repeat seed offset differs from lock")
    if not bool(optimization["covnet_repeat_same_head_initialization"]):
        raise ValueError("CovNet repeat must preserve head initialization")
    if not bool(optimization["covnet_repeat_same_batch_orders"]):
        raise ValueError("CovNet repeat must preserve batch orders")
    expected_rows = optimization["initial_states"]
    expected_by_fold = {
        int(row["outer_fold"]): row for row in expected_rows
    }
    if len(expected_rows) != 5 or set(expected_by_fold) != set(range(5)):
        raise ValueError("Initial-state lock must contain exactly folds 0..4")
    rows = []
    for fold in range(5):
        primary_head = initialize_linear_head(
            feature_dim=FEATURE_DIM,
            class_count=CLASS_COUNT,
            seed=stable_seed("rn_lisda_head", BASE_SEED, fold),
            device=torch.device("cpu"),
        )
        repeat_head = initialize_linear_head(
            feature_dim=FEATURE_DIM,
            class_count=CLASS_COUNT,
            seed=stable_seed("rn_lisda_head", BASE_SEED, fold),
            device=torch.device("cpu"),
        )
        primary_covnet = CovariancePredictor(
            input_dim=FEATURE_DIM,
            hidden_dim=FEATURE_DIM // 2,
            output_dim=FEATURE_DIM,
            seed=stable_seed("rn_lisda_covnet", BASE_SEED, fold),
        )
        repeat_covnet = CovariancePredictor(
            input_dim=FEATURE_DIM,
            hidden_dim=FEATURE_DIM // 2,
            output_dim=FEATURE_DIM,
            seed=stable_seed(
                "rn_lisda_covnet",
                BASE_SEED + COVNET_REPEAT_SEED_OFFSET,
                fold,
            ),
        )
        observed = {
            "outer_fold": fold,
            "primary_head_sha256": state_arrays_sha256(
                primary_head.clone_cpu()
            ),
            "primary_covnet_sha256": state_arrays_sha256(
                covariance_state_arrays(primary_covnet)
            ),
            "repeat_head_sha256": state_arrays_sha256(
                repeat_head.clone_cpu()
            ),
            "repeat_covnet_sha256": state_arrays_sha256(
                covariance_state_arrays(repeat_covnet)
            ),
        }
        expected = expected_by_fold[fold]
        if set(expected) != set(observed):
            raise ValueError(f"Initial-state lock fields differ for fold {fold}")
        for key, value in observed.items():
            if key in expected and expected[key] != value:
                raise ValueError(
                    f"Initial-state lock mismatch fold={fold} key={key}"
                )
        rows.append(observed)
    return rows


def _configure_torch(device: torch.device) -> None:
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != EXPECTED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG changed after the pre-import lock"
        )
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED % (2**32))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the RN-LISDA lock")
        torch.cuda.manual_seed_all(BASE_SEED)
        torch.cuda.reset_peak_memory_stats(device)


def _process_snapshot() -> Dict[str, object]:
    current = psutil.Process()
    allowed = {current.pid}
    try:
        allowed.update(process.pid for process in current.parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        allowed.update(process.pid for process in current.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    observed = []
    unexpected = []
    for process in psutil.process_iter(
        ["pid", "ppid", "name", "cmdline", "create_time"]
    ):
        try:
            name = str(process.info.get("name") or "")
            lowered = name.casefold()
            if "python" not in lowered and "trtexec" not in lowered:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            row = {
                "pid": int(process.info["pid"]),
                "parent_pid": int(process.info.get("ppid") or 0),
                "name": name,
                "command_line": command,
                "create_time": float(process.info.get("create_time") or 0.0),
                "owned_by_auditor_chain": int(process.info["pid"]) in allowed,
            }
            observed.append(row)
            if not row["owned_by_auditor_chain"]:
                unexpected.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
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
    def __init__(
        self,
        interval_seconds: float = 0.05,
        process_interval_seconds: float = 1.0,
    ) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process_interval_seconds = float(process_interval_seconds)
        self.peak_rss_bytes = 0
        self.peak_virtual_memory_fraction = 0.0
        self.unexpected_processes: Dict[int, Dict[str, object]] = {}
        self._last_process_check = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample_once(self) -> None:
        total_rss = 0
        process = psutil.Process()
        candidates = [process]
        try:
            candidates.extend(process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for candidate in candidates:
            try:
                total_rss += int(candidate.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_rss_bytes = max(self.peak_rss_bytes, total_rss)
        self.peak_virtual_memory_fraction = max(
            self.peak_virtual_memory_fraction,
            float(psutil.virtual_memory().percent / 100.0),
        )
        now = time.monotonic()
        if now - self._last_process_check >= self.process_interval_seconds:
            snapshot = _process_snapshot()
            for row in snapshot["unexpected_processes"]:
                self.unexpected_processes[int(row["pid"])] = dict(row)
            self._last_process_check = now

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "PeakResourceMonitor":
        self._sample_once()
        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample_once()


def _raw_train_metadata_sha256(
    paths: Sequence[str], data_yaml: Path
) -> Tuple[str, int]:
    unique: Dict[str, Path] = {str(data_yaml.resolve()): data_yaml.resolve()}
    for value in paths:
        image = Path(str(value)).resolve()
        unique[str(image)] = image
        parts = list(image.parts)
        lowered = [part.casefold() for part in parts]
        for index in range(len(parts) - 1):
            if lowered[index : index + 2] == ["images", "train"]:
                label_parts = parts.copy()
                label_parts[index] = "labels"
                label = Path(*label_parts).with_suffix(".txt")
                unique[str(label)] = label
                break
    digest = hashlib.sha256()
    for key in sorted(unique, key=str.casefold):
        path = unique[key]
        stat = path.stat()
        encoded = str(path).encode("utf-8")
        digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
        digest.update(encoded)
        digest.update(
            np.asarray(
                [int(stat.st_size), int(stat.st_mtime_ns)], dtype=np.int64
            ).tobytes()
        )
    return digest.hexdigest(), len(unique)


def train_all_roles(
    *,
    cache: Mapping[str, np.ndarray],
    partitions: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[Tuple[str, int], Dict[str, object]],
]:
    features = torch.as_tensor(
        np.asarray(cache["embeddings"], dtype=np.float32), device=device
    )
    labels = torch.as_tensor(
        np.asarray(cache["labels"], dtype=np.int64), device=device
    )
    sources = np.asarray(cache["source_stems"]).astype(str)
    outputs = {
        role: np.full(
            (EXPECTED_ROWS, CLASS_COUNT), np.nan, dtype=np.float32
        )
        for role in ROLE_NAMES
    }
    results: Dict[Tuple[str, int], Dict[str, object]] = {}
    for role in ROLE_NAMES:
        for partition in partitions:
            fold = int(partition["outer_fold"])
            print(
                json.dumps(
                    {
                        "stage": "rn_lisda_role_fold_start",
                        "role": role,
                        "outer_fold": fold,
                    }
                ),
                flush=True,
            )
            result = train_role_fold(
                role=role,
                features=features,
                labels=labels,
                source_stems=sources,
                outer_fold=fold,
                partition_a=np.asarray(partition["partition_a"]),
                partition_b=np.asarray(partition["partition_b"]),
                holdout=np.asarray(partition["held_indices"]),
            )
            held = np.asarray(result["holdout_indices"], dtype=np.int64)
            outputs[role][held] = np.asarray(
                result["probabilities"], dtype=np.float32
            )
            results[(role, fold)] = result
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for role, values in outputs.items():
        if not np.isfinite(values).all():
            raise ValueError(f"OOF output is incomplete for {role}")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"OOF output is not normalized for {role}")
    return outputs, results


def _load_predictor(
    state: Mapping[str, np.ndarray],
    *,
    fold: int,
    covnet_repeat: bool,
    device: torch.device,
) -> CovariancePredictor:
    seed_offset = COVNET_REPEAT_SEED_OFFSET if covnet_repeat else 0
    predictor = CovariancePredictor(
        input_dim=FEATURE_DIM,
        hidden_dim=FEATURE_DIM // 2,
        output_dim=FEATURE_DIM,
        seed=stable_seed(
            "rn_lisda_covnet", BASE_SEED + seed_offset, int(fold)
        ),
    ).to(device=device)
    tensor_state = {
        name: torch.as_tensor(value, device=device)
        for name, value in state.items()
    }
    predictor.load_state_dict(tensor_state, strict=True)
    predictor.eval()
    return predictor


def equation_diagnostics() -> Dict[str, object]:
    generator = np.random.default_rng(BASE_SEED)
    features = generator.normal(size=(19, 11)).astype(np.float64)
    labels = np.asarray(([0, 1, 2, 3, 4] * 4)[:19], dtype=np.int64)
    weight = generator.normal(size=(CLASS_COUNT, 11)).astype(np.float64)
    bias = generator.normal(size=CLASS_COUNT).astype(np.float64)
    covariance = generator.uniform(0.01, 0.4, size=(19, 11)).astype(
        np.float64
    )
    strength = 3.25
    augmented, clean, delta = isda_logits(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
        weight=torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
        covariance=torch.from_numpy(covariance),
        lambda_strength=strength,
    )
    expected_clean = features @ weight.T + bias
    expected_delta = np.zeros((len(labels), CLASS_COUNT), dtype=np.float64)
    for row, target in enumerate(labels.tolist()):
        if target not in ELIGIBLE_CLASSES:
            continue
        difference = weight[FOCUS_CLASS] - weight[target]
        expected_delta[row, FOCUS_CLASS] = (
            0.5
            * strength
            * float(np.sum(np.square(difference) * covariance[row]))
        )
    clean_error = float(
        np.max(np.abs(clean.detach().numpy() - expected_clean))
    )
    delta_error = float(
        np.max(np.abs(delta.detach().numpy() - expected_delta))
    )
    augmented_error = float(
        np.max(
            np.abs(
                augmented.detach().numpy()
                - (expected_clean + expected_delta)
            )
        )
    )
    clean_class_mask = np.isin(labels, [1, 3])
    nonfocus_columns = [0, 2, 3, 4]
    predictor = CovariancePredictor(
        input_dim=11,
        hidden_dim=6,
        output_dim=11,
        seed=BASE_SEED,
    )
    base = torch.from_numpy(
        generator.uniform(0.01, 0.5, size=(CLASS_COUNT, 11)).astype(
            np.float32
        )
    )
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    label_tensor = torch.from_numpy(labels)
    predicted_covariance, scales = predict_covariance(
        predictor, feature_tensor, label_tensor, base
    )
    initial_error = float(
        torch.max(
            torch.abs(predicted_covariance - base[label_tensor])
        ).item()
    )
    return {
        "clean_logits_max_error": clean_error,
        "delta_max_error": delta_error,
        "augmented_logits_max_error": augmented_error,
        "equation_oracle_max_error": max(
            clean_error, delta_error, augmented_error
        ),
        "clean_class_delta_exact_zero": bool(
            np.count_nonzero(delta.detach().numpy()[clean_class_mask]) == 0
        ),
        "nonfocus_delta_exact_zero": bool(
            np.count_nonzero(
                delta.detach().numpy()[:, nonfocus_columns]
            )
            == 0
        ),
        "initial_scale_max_error_from_one": float(
            torch.max(torch.abs(scales - 1.0)).item()
        ),
        "initial_covariance_max_error": initial_error,
    }


def mechanism_evidence(
    *,
    cache: Mapping[str, np.ndarray],
    partitions: Sequence[Mapping[str, object]],
    results: Mapping[Tuple[str, int], Mapping[str, object]],
    lock: Mapping[str, object],
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[int, Dict[str, np.ndarray]]]:
    features = torch.as_tensor(
        np.asarray(cache["embeddings"], dtype=np.float32), device=device
    )
    labels = torch.as_tensor(
        np.asarray(cache["labels"], dtype=np.int64), device=device
    )
    diagnostics = equation_diagnostics()
    expected_initial = {
        int(row["outer_fold"]): row
        for row in lock["optimization"]["initial_states"]
    }
    expected_meta = lock["optimization"]["meta_schedule"][
        "meta_updates_by_outer_fold"
    ]
    expected_steps = lock["optimization"]["paired_steps_by_outer_fold"]
    expected_head = lock["optimization"]["head_updates_by_outer_fold"]
    partition_by_fold = {
        int(row["outer_fold"]): row for row in partitions
    }
    empty_state_sha256 = state_arrays_sha256({})
    reference_order_hashes: Dict[int, Tuple[Tuple[str, str], ...]] = {}
    primary_meta_rows = []
    fold_rows = []
    views: Dict[int, Dict[str, np.ndarray]] = {}
    all_candidate_covariances = []
    all_joint_covariances = []
    all_candidate_scales = []
    all_effective_ranks = []
    all_swap_changes = []
    all_role_values_finite = True
    all_update_counts_exact = True
    all_meta_role_updates_nonzero = True
    all_joint_direct_gradients_nonzero = True
    all_initial_states_exact = True
    all_partition_provenance_exact = True
    all_holdout_score_contract_exact = True
    all_paired_orders_common = True
    all_balanced_meta_schedules_exact = True
    all_balanced_meta_batches_exact = True

    for role in ROLE_NAMES:
        for fold in range(5):
            result = results[(role, fold)]
            expected_state = expected_initial[fold]
            expected_covnet_sha256 = (
                str(expected_state["repeat_covnet_sha256"])
                if role == "rn_lisda_covnet_seed_repeat"
                else (
                    str(expected_state["primary_covnet_sha256"])
                    if role in META_ROLES or role == "rn_lisda_joint_no_meta"
                    else empty_state_sha256
                )
            )
            all_initial_states_exact &= (
                str(result["initial_head_sha256"])
                == str(expected_state["primary_head_sha256"])
                and str(result["initial_covnet_sha256"])
                == expected_covnet_sha256
            )
            partition = partition_by_fold[fold]
            all_partition_provenance_exact &= (
                str(result["partition_a_indices_sha256"])
                == str(partition["a_indices_sha256"])
                and str(result["partition_b_indices_sha256"])
                == str(partition["b_indices_sha256"])
                and str(result["holdout_indices_sha256"])
                == str(partition["held_indices_sha256"])
            )
            all_balanced_meta_schedules_exact &= (
                str(result["balanced_meta_schedule_a_sha256"])
                == str(partition["a_balanced_meta_schedule_sha256"])
                and str(result["balanced_meta_schedule_b_sha256"])
                == str(partition["b_balanced_meta_schedule_sha256"])
                and int(
                    result[
                        "balanced_meta_schedule_a_max_class_usage_spread"
                    ]
                )
                <= int(
                    lock["gates"]["mechanism"][
                        "balanced_meta_max_row_use_spread_per_class"
                    ]
                )
                and int(
                    result[
                        "balanced_meta_schedule_b_max_class_usage_spread"
                    ]
                )
                <= int(
                    lock["gates"]["mechanism"][
                        "balanced_meta_max_row_use_spread_per_class"
                    ]
                )
            )
            all_holdout_score_contract_exact &= (
                int(result["holdout_score_calls"]) == 1
                and int(result["holdout_score_epoch"]) == 30
            )
            order_hashes = tuple(
                (
                    str(row["partition_a_consumed_indices_sha256"]),
                    str(row["partition_b_consumed_indices_sha256"]),
                )
                for row in result["trace"]
            )
            if role == "ce_control":
                reference_order_hashes[fold] = order_hashes
            else:
                all_paired_orders_common &= (
                    reference_order_hashes.get(fold) == order_hashes
                )
            state_arrays = {
                **{
                    f"head.{name}": np.asarray(value)
                    for name, value in result["head_state"].items()
                },
                **{
                    f"covnet.{name}": np.asarray(value)
                    for name, value in result["covnet_state"].items()
                },
            }
            traces = [
                *result["trace"],
                *result["meta_trace"],
            ]
            all_role_values_finite &= all(
                np.isfinite(value).all() for value in state_arrays.values()
            )
            all_role_values_finite &= all(
                math.isfinite(float(value))
                for row in traces
                for value in row.values()
                if isinstance(value, (int, float))
            )
            all_update_counts_exact &= (
                int(result["head_update_count"]) == int(expected_head[fold])
                and int(result["paired_steps"]) == int(expected_steps[fold])
            )
            expected_role_meta = (
                int(expected_meta[fold]) if role in META_ROLES else 0
            )
            all_update_counts_exact &= (
                int(result["meta_update_count"]) == expected_role_meta
            )
            expected_meta_slots = (
                int(partition["balanced_meta_slot_count"])
                if role in META_ROLES
                else 0
            )
            all_update_counts_exact &= (
                int(result["balanced_meta_slot_count"])
                == expected_meta_slots
            )
            if role in META_ROLES:
                meta_rows = result["meta_trace"]
                all_meta_role_updates_nonzero &= (
                    int(result["meta_update_count"]) > 0
                    and any(
                        float(row["output_gradient_norm"]) > 0.0
                        for row in meta_rows
                    )
                )
                all_balanced_meta_batches_exact &= all(
                    tuple(int(value) for value in row["meta_class_counts"])
                    == (META_SAMPLES_PER_CLASS,) * CLASS_COUNT
                    and int(row["meta_unique_row_count"]) == META_BATCH_SIZE
                    and int(row["meta_unique_source_count"])
                    == META_BATCH_SIZE
                    and int(row["train_meta_source_overlap_count"]) == 0
                    and len(str(row["meta_indices_sha256"])) == 64
                    for row in meta_rows
                )
            if role == "rn_lisda_joint_no_meta":
                semantic_gradients = [
                    float(row["mean_direct_covnet_gradient_norm"])
                    for row in result["trace"]
                    if int(row["epoch"]) >= CLEAN_EPOCHS
                ]
                all_joint_direct_gradients_nonzero &= bool(
                    semantic_gradients
                ) and all(value > 0.0 for value in semantic_gradients)

    for partition in partitions:
        fold = int(partition["outer_fold"])
        held = np.asarray(partition["held_indices"], dtype=np.int64)
        fit = np.asarray(partition["fit_indices"], dtype=np.int64)
        donors = np.asarray(
            partition["held_swap_donors"], dtype=np.int64
        )
        fit_tensor = torch.as_tensor(fit, device=device)
        held_tensor = torch.as_tensor(held, device=device)
        donor_tensor = torch.as_tensor(donors, device=device)
        base_fit = population_diagonal_variance(
            features[fit_tensor], labels[fit_tensor]
        )

        candidate_result = results[("rn_lisda_candidate", fold)]
        candidate_state = candidate_result["covnet_state"]
        assert isinstance(candidate_state, Mapping)
        candidate_predictor = _load_predictor(
            candidate_state,
            fold=fold,
            covnet_repeat=False,
            device=device,
        )
        joint_state = results[("rn_lisda_joint_no_meta", fold)][
            "covnet_state"
        ]
        assert isinstance(joint_state, Mapping)
        joint_predictor = _load_predictor(
            joint_state,
            fold=fold,
            covnet_repeat=False,
            device=device,
        )
        deranged_state = results[("rn_lisda_deranged_input", fold)][
            "covnet_state"
        ]
        assert isinstance(deranged_state, Mapping)
        deranged_predictor = _load_predictor(
            deranged_state,
            fold=fold,
            covnet_repeat=False,
            device=device,
        )
        with torch.no_grad():
            candidate_covariance, candidate_scale = predict_covariance(
                candidate_predictor,
                features[held_tensor],
                labels[held_tensor],
                base_fit,
            )
            swapped_covariance, swapped_scale = predict_covariance(
                candidate_predictor,
                features[donor_tensor],
                labels[held_tensor],
                base_fit,
            )
            joint_covariance, joint_scale = predict_covariance(
                joint_predictor,
                features[held_tensor],
                labels[held_tensor],
                base_fit,
            )
            deranged_role_covariance, deranged_role_scale = predict_covariance(
                deranged_predictor,
                features[donor_tensor],
                labels[held_tensor],
                base_fit,
            )
        candidate_covariance_np = candidate_covariance.cpu().numpy()
        candidate_scale_np = candidate_scale.cpu().numpy()
        swapped_covariance_np = swapped_covariance.cpu().numpy()
        swapped_scale_np = swapped_scale.cpu().numpy()
        joint_covariance_np = joint_covariance.cpu().numpy()
        joint_scale_np = joint_scale.cpu().numpy()
        deranged_role_covariance_np = deranged_role_covariance.cpu().numpy()
        deranged_role_scale_np = deranged_role_scale.cpu().numpy()
        base_fit_np = base_fit.cpu().numpy()
        held_labels = np.asarray(cache["labels"])[held]
        eligible = np.isin(held_labels, ELIGIBLE_CLASSES)
        eligible_covariance = candidate_covariance_np[eligible]
        eligible_joint = joint_covariance_np[eligible]
        eligible_scale = candidate_scale_np[eligible]
        eligible_swapped = swapped_covariance_np[eligible]
        numerator = np.square(eligible_covariance.sum(axis=1))
        denominator = np.square(eligible_covariance).sum(axis=1)
        effective_rank = numerator / np.maximum(denominator, 1e-12)
        swap_change = np.abs(
            eligible_covariance - eligible_swapped
        ).sum(axis=1) / np.maximum(
            np.abs(eligible_covariance).sum(axis=1), 1e-12
        )
        all_candidate_covariances.append(eligible_covariance)
        all_joint_covariances.append(eligible_joint)
        all_candidate_scales.append(eligible_scale)
        all_effective_ranks.append(effective_rank)
        all_swap_changes.append(swap_change)

        meta_trace = candidate_result["meta_trace"]
        assert isinstance(meta_trace, list)
        if len(meta_trace) < 2:
            raise ValueError(f"Candidate fold {fold} has insufficient meta trace")
        first = meta_trace[0]
        second = meta_trace[1]
        later = meta_trace[2:]
        initial_sha = str(
            expected_initial[fold]["primary_covnet_sha256"]
        )
        final_sha = state_arrays_sha256(
            {name: np.asarray(value) for name, value in candidate_state.items()}
        )
        primary_meta_rows.append(
            {
                "outer_fold": fold,
                "meta_updates": int(len(meta_trace)),
                "first_hidden_gradient_norm": float(
                    first["hidden_gradient_norm"]
                ),
                "first_output_gradient_norm": float(
                    first["output_gradient_norm"]
                ),
                "second_hidden_gradient_norm": float(
                    second["hidden_gradient_norm"]
                ),
                "second_output_gradient_norm": float(
                    second["output_gradient_norm"]
                ),
                "later_hidden_min": float(
                    min(
                        float(row["hidden_gradient_norm"]) for row in later
                    )
                ),
                "later_output_min": float(
                    min(
                        float(row["output_gradient_norm"]) for row in later
                    )
                ),
                "initial_covnet_sha256": initial_sha,
                "final_covnet_sha256": final_sha,
                "state_changed": final_sha != initial_sha,
            }
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "held_rows": int(len(held)),
                "eligible_held_rows": int(eligible.sum()),
                "candidate_scale_min": float(candidate_scale_np.min()),
                "candidate_scale_max": float(candidate_scale_np.max()),
                "candidate_effective_rank_median": float(
                    np.median(effective_rank)
                ),
                "candidate_row_mean_scale_std": float(
                    np.std(
                        eligible_scale.mean(axis=1), ddof=0
                    )
                ),
                "input_swap_relative_change_mean": float(
                    np.mean(swap_change)
                ),
                "candidate_mean_covariance": float(
                    np.mean(eligible_covariance)
                ),
                "joint_mean_covariance": float(np.mean(eligible_joint)),
            }
        )
        views[fold] = {
            "held_indices": held,
            "held_swap_donors": donors,
            "base_fit": base_fit_np,
            "candidate_covariance": candidate_covariance_np,
            "candidate_scale": candidate_scale_np,
            "swapped_covariance": swapped_covariance_np,
            "swapped_scale": swapped_scale_np,
            "joint_covariance": joint_covariance_np,
            "joint_scale": joint_scale_np,
            "deranged_role_covariance": deranged_role_covariance_np,
            "deranged_role_scale": deranged_role_scale_np,
        }

    candidate_covariance_all = np.concatenate(
        all_candidate_covariances, axis=0
    )
    joint_covariance_all = np.concatenate(all_joint_covariances, axis=0)
    candidate_scale_all = np.concatenate(all_candidate_scales, axis=0)
    effective_rank_all = np.concatenate(all_effective_ranks, axis=0)
    swap_change_all = np.concatenate(all_swap_changes, axis=0)
    mechanism_thresholds = lock["gates"]["mechanism"]
    checks = {
        "equation_oracle": (
            float(diagnostics["equation_oracle_max_error"])
            <= float(mechanism_thresholds["equation_oracle_error_max"])
        ),
        "clean_class_delta_exact_zero": bool(
            diagnostics["clean_class_delta_exact_zero"]
        ),
        "only_focus_rival_delta_nonzero": bool(
            diagnostics["nonfocus_delta_exact_zero"]
        ),
        "initial_covariance": (
            float(diagnostics["initial_covariance_max_error"])
            <= float(mechanism_thresholds["initial_covariance_error_max"])
        ),
        "first_output_meta_gradient_nonzero": all(
            float(row["first_output_gradient_norm"]) > 0.0
            for row in primary_meta_rows
        ),
        "first_hidden_meta_gradient_zero": all(
            float(row["first_hidden_gradient_norm"])
            <= float(
                mechanism_thresholds[
                    "first_hidden_meta_gradient_abs_max"
                ]
            )
            for row in primary_meta_rows
        ),
        "second_meta_gradients_nonzero": all(
            float(row["second_hidden_gradient_norm"]) > 0.0
            and float(row["second_output_gradient_norm"]) > 0.0
            for row in primary_meta_rows
        ),
        "later_meta_gradients_nonzero": all(
            float(row["later_hidden_min"]) > 0.0
            and float(row["later_output_min"]) > 0.0
            for row in primary_meta_rows
        ),
        "covnet_state_changes_each_fold": all(
            bool(row["state_changed"]) for row in primary_meta_rows
        ),
        "candidate_covariance_positive_finite": (
            bool(
                mechanism_thresholds[
                    "covariance_strictly_positive_and_finite"
                ]
            )
            and np.isfinite(candidate_covariance_all).all()
            and np.all(candidate_covariance_all > 0.0)
        ),
        "covariance_scale_range": (
            float(candidate_scale_all.min())
            >= float(mechanism_thresholds["covariance_scale_min"])
            and float(candidate_scale_all.max())
            <= float(mechanism_thresholds["covariance_scale_max"])
        ),
        "median_effective_rank": (
            float(np.median(effective_rank_all))
            >= float(mechanism_thresholds["median_effective_rank_min"])
        ),
        "samplewise_scale_std": (
            float(np.std(candidate_scale_all.mean(axis=1), ddof=0))
            >= float(mechanism_thresholds["samplewise_scale_std_min"])
        ),
        "same_label_input_swap_change": (
            float(np.mean(swap_change_all))
            >= float(
                mechanism_thresholds[
                    "same_label_input_swap_relative_change_min"
                ]
            )
        ),
        "candidate_to_joint_covariance_ratio": (
            float(candidate_covariance_all.mean())
            / max(float(joint_covariance_all.mean()), 1e-12)
            >= float(
                mechanism_thresholds[
                    "candidate_to_joint_mean_covariance_ratio_min"
                ]
            )
        ),
        "all_role_values_finite": all_role_values_finite,
        "all_update_counts_exact": all_update_counts_exact,
        "all_meta_role_updates_nonzero": all_meta_role_updates_nonzero,
        "joint_direct_gradients_nonzero": all_joint_direct_gradients_nonzero,
        "all_initial_states_exact": all_initial_states_exact,
        "all_partition_provenance_exact": all_partition_provenance_exact,
        "all_holdout_score_contract_exact": all_holdout_score_contract_exact,
        "all_paired_orders_common": all_paired_orders_common,
        "all_balanced_meta_schedules_exact": (
            all_balanced_meta_schedules_exact
        ),
        "all_balanced_meta_batches_exact": all_balanced_meta_batches_exact,
    }
    telemetry = {
        "equation": diagnostics,
        "primary_meta_trace": primary_meta_rows,
        "folds": fold_rows,
        "aggregate": {
            "candidate_scale_min": float(candidate_scale_all.min()),
            "candidate_scale_max": float(candidate_scale_all.max()),
            "candidate_effective_rank_median": float(
                np.median(effective_rank_all)
            ),
            "candidate_row_mean_scale_std": float(
                np.std(candidate_scale_all.mean(axis=1), ddof=0)
            ),
            "same_label_input_swap_relative_change_mean": float(
                np.mean(swap_change_all)
            ),
            "candidate_mean_covariance": float(
                candidate_covariance_all.mean()
            ),
            "joint_mean_covariance": float(joint_covariance_all.mean()),
            "candidate_to_joint_mean_covariance_ratio": float(
                candidate_covariance_all.mean()
                / max(float(joint_covariance_all.mean()), 1e-12)
            ),
        },
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": int(len(checks)),
        "mechanism_pass": all(bool(value) for value in checks.values()),
    }
    return telemetry, views


def _dataset_for_visuals(cache: Mapping[str, np.ndarray]):
    from trkh.data.dataset import MangoYOLOCropDataset

    first_image = Path(str(cache["paths"][0])).resolve()
    images_dir = first_image.parent
    parts = list(images_dir.parts)
    lowered = [part.casefold() for part in parts]
    labels_dir: Optional[Path] = None
    for index in range(len(parts) - 1):
        if lowered[index : index + 2] == ["images", "train"]:
            label_parts = parts.copy()
            label_parts[index] = "labels"
            labels_dir = Path(*label_parts)
            break
    if labels_dir is None:
        raise ValueError("Cannot derive train label directory")
    dataset = MangoYOLOCropDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=None,
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        fallback_to_full_image=True,
        num_classes=CLASS_COUNT,
        primary_object_strategy="largest",
        split="train",
        class_aware_augmentation=False,
        classification_target=True,
        classification_object_crops=True,
        class_crop_margin_scale_threshold=1.5,
        class_crop_margin_max_ratio=0.16,
        class_crop_margin_scales=[1.0] * CLASS_COUNT,
        classification_source_context=False,
        classification_bbox_metadata=False,
    )
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"Visual dataset row mismatch: {len(dataset)}")
    dataset_paths = np.asarray(
        [str(sample.image_path) for sample in dataset.samples]
    )
    dataset_labels = np.asarray(
        [int(sample.primary_label) for sample in dataset.samples],
        dtype=np.int64,
    )
    if not np.array_equal(
        np.char.lower(dataset_paths), np.char.lower(cache["paths"].astype(str))
    ):
        raise ValueError("Visual dataset/cache path order mismatch")
    if not np.array_equal(dataset_labels, cache["labels"]):
        raise ValueError("Visual dataset/cache label order mismatch")
    return dataset


def _load_visual_crop(dataset: object, index: int) -> Image.Image:
    item = dataset[int(index)]
    if not isinstance(item, tuple) or len(item) < 2:
        raise TypeError("Unexpected visual dataset item")
    image = item[0]
    if not isinstance(image, Image.Image):
        raise TypeError("Visual dataset did not return a PIL crop")
    return image.convert("RGB")


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _render_visual_contact_sheet(
    *,
    rows: Sequence[Mapping[str, object]],
    dataset: object,
    output_path: Path,
) -> None:
    columns = (
        "anchor crop",
        "nearest same-label",
        "fixed-classwise proxy",
        "candidate proxy 0",
        "candidate proxy 1",
        "deranged-input proxy",
    )
    tile_width = 300
    tile_height = 260
    header_height = 58
    title_height = 34
    canvas = Image.new(
        "RGB",
        (tile_width * len(columns), title_height + header_height + tile_height * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(16)
    body_font = _font(12)
    small_font = _font(10)
    draw.text(
        (8, 7),
        "RN-LISDA A0 fixed train-only semantic proxy audit",
        fill="black",
        font=title_font,
    )
    for column, label in enumerate(columns):
        x = column * tile_width
        draw.rectangle(
            (x, title_height, x + tile_width - 1, title_height + header_height - 1),
            outline="#707070",
            fill="#e9eef2",
        )
        draw.text(
            (x + 6, title_height + 18),
            label,
            fill="black",
            font=body_font,
        )
    for row_number, row in enumerate(rows):
        y = title_height + header_height + row_number * tile_height
        proxy_indices = row["proxy_indices"]
        proxy_labels = row["proxy_labels"]
        distances = row["proxy_distances"]
        assert isinstance(proxy_indices, list)
        assert isinstance(proxy_labels, list)
        assert isinstance(distances, list)
        cell_indices = [int(row["sample_index"]), *proxy_indices]
        for column, index in enumerate(cell_indices):
            x = column * tile_width
            draw.rectangle(
                (x, y, x + tile_width - 1, y + tile_height - 1),
                outline="#9a9a9a",
                fill="white",
            )
            if index >= 0:
                image = _load_visual_crop(dataset, index)
                fitted = ImageOps.contain(image, (288, 215))
                image_x = x + (tile_width - fitted.width) // 2
                image_y = y + 4
                canvas.paste(fitted, (image_x, image_y))
            else:
                draw.rectangle(
                    (x + 6, y + 6, x + tile_width - 7, y + 215),
                    fill="#f1f1f1",
                )
                draw.text(
                    (x + 16, y + 98),
                    "disabled",
                    fill="#555555",
                    font=body_font,
                )
            if column == 0:
                caption = (
                    f"row {row_number + 1}/20 i={index} "
                    f"f={row['outer_fold']} y={row['target_index']}"
                )
            elif index < 0:
                caption = "class-1 augmentation disabled"
            else:
                distance = float(distances[column - 1])
                caption = (
                    f"i={index} y={int(proxy_labels[column - 1])} "
                    f"d={distance:.4f}"
                )
            draw.text(
                (x + 5, y + 229),
                caption,
                fill="black",
                font=small_font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_visual_evidence(
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    sources: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    views: Mapping[int, Mapping[str, np.ndarray]],
    lock: Mapping[str, object],
    contact_sheet_path: Path,
    render_contact: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    dataset = _dataset_for_visuals(cache)
    features = np.asarray(cache["embeddings"], dtype=np.float32)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    anchors = np.asarray(VISUAL_ANCHORS, dtype=np.int64)
    fixed_proxy = np.full(len(anchors), -1, dtype=np.int64)
    candidate_proxy_0 = np.full(len(anchors), -1, dtype=np.int64)
    candidate_proxy_1 = np.full(len(anchors), -1, dtype=np.int64)
    deranged_proxy = np.full(len(anchors), -1, dtype=np.int64)
    nearest_same_label = np.full(len(anchors), -1, dtype=np.int64)
    fixed_draws = np.zeros((len(anchors), FEATURE_DIM), dtype=np.float32)
    candidate_draws_0 = np.zeros_like(fixed_draws)
    candidate_draws_1 = np.zeros_like(fixed_draws)
    deranged_draws = np.zeros_like(fixed_draws)
    fixed_epsilon = np.zeros_like(fixed_draws)
    candidate_epsilon_0 = np.zeros_like(fixed_draws)
    candidate_epsilon_1 = np.zeros_like(fixed_draws)
    fixed_covariance = np.zeros_like(fixed_draws)
    candidate_covariance = np.zeros_like(fixed_draws)
    deranged_covariance = np.zeros_like(fixed_draws)
    enabled = np.zeros(len(anchors), dtype=np.bool_)
    rows = []
    for row_number, anchor in enumerate(anchors.tolist()):
        fold = int(folds[anchor])
        view = views[fold]
        held = np.asarray(view["held_indices"], dtype=np.int64)
        position = int(np.searchsorted(held, anchor))
        if position >= len(held) or int(held[position]) != anchor:
            raise ValueError(f"Visual anchor {anchor} is outside held fold")
        fit = np.flatnonzero(folds != fold).astype(np.int64)
        same_label_pool = fit[labels[fit] == labels[anchor]]
        same_index, same_distance = cosine_nearest_proxy(
            features[anchor],
            features,
            same_label_pool,
            sources,
            excluded_source=str(sources[anchor]),
        )
        nearest_same_label[row_number] = same_index
        proxy_indices = [same_index, -1, -1, -1, -1]
        proxy_labels = [int(labels[same_index]), -1, -1, -1, -1]
        proxy_distances: List[Optional[float]] = [
            same_distance,
            None,
            None,
            None,
            None,
        ]
        if int(labels[anchor]) in ELIGIBLE_CLASSES:
            enabled[row_number] = True
            base = np.asarray(view["base_fit"], dtype=np.float32)[labels[anchor]]
            candidate_cov = np.asarray(
                view["candidate_covariance"], dtype=np.float32
            )[position]
            swapped_cov = np.asarray(
                view["deranged_role_covariance"], dtype=np.float32
            )[position]
            fixed_covariance[row_number] = base
            candidate_covariance[row_number] = candidate_cov
            deranged_covariance[row_number] = swapped_cov
            fixed_child, fixed_noise = draw_semantic_feature(
                features[anchor],
                base,
                outer_fold=fold,
                sample_index=anchor,
                role="fixed_classwise",
                draw_index=0,
            )
            candidate_child_0, candidate_noise_0 = draw_semantic_feature(
                features[anchor],
                candidate_cov,
                outer_fold=fold,
                sample_index=anchor,
                role="candidate",
                draw_index=0,
            )
            candidate_child_1, candidate_noise_1 = draw_semantic_feature(
                features[anchor],
                candidate_cov,
                outer_fold=fold,
                sample_index=anchor,
                role="candidate",
                draw_index=1,
            )
            deranged_child, deranged_noise = draw_semantic_feature(
                features[anchor],
                swapped_cov,
                outer_fold=fold,
                sample_index=anchor,
                role="candidate",
                draw_index=0,
            )
            if not np.array_equal(candidate_noise_0, deranged_noise):
                raise ValueError("Deranged visual did not reuse candidate epsilon")
            fixed_draws[row_number] = fixed_child
            candidate_draws_0[row_number] = candidate_child_0
            candidate_draws_1[row_number] = candidate_child_1
            deranged_draws[row_number] = deranged_child
            fixed_epsilon[row_number] = fixed_noise
            candidate_epsilon_0[row_number] = candidate_noise_0
            candidate_epsilon_1[row_number] = candidate_noise_1
            children = (
                fixed_child,
                candidate_child_0,
                candidate_child_1,
                deranged_child,
            )
            resolved = []
            for child in children:
                proxy, distance = cosine_nearest_proxy(
                    child,
                    features,
                    fit,
                    sources,
                    excluded_source=str(sources[anchor]),
                )
                resolved.append((proxy, distance))
            fixed_proxy[row_number] = resolved[0][0]
            candidate_proxy_0[row_number] = resolved[1][0]
            candidate_proxy_1[row_number] = resolved[2][0]
            deranged_proxy[row_number] = resolved[3][0]
            proxy_indices[1:] = [value[0] for value in resolved]
            proxy_labels[1:] = [
                int(labels[value[0]]) for value in resolved
            ]
            proxy_distances[1:] = [float(value[1]) for value in resolved]
        for proxy in proxy_indices:
            if proxy < 0:
                continue
            if int(folds[proxy]) == fold:
                raise ValueError("Visual proxy is outside outer fit")
            if str(sources[proxy]) == str(sources[anchor]):
                raise ValueError("Visual proxy reuses anchor source")
        donor = int(np.asarray(view["held_swap_donors"])[position])
        expected_donor = int(
            lock["visual_anchors"]["held_swap_anchor_donors"][str(anchor)]
        )
        if donor != expected_donor:
            raise ValueError(f"Locked visual donor mismatch at {anchor}")
        rows.append(
            {
                "row": row_number + 1,
                "sample_index": anchor,
                "outer_fold": fold,
                "target_index": int(labels[anchor]),
                "source_stem": str(sources[anchor]),
                "held_swap_donor": donor,
                "augmentation_enabled": bool(enabled[row_number]),
                "proxy_indices": proxy_indices,
                "proxy_labels": proxy_labels,
                "proxy_distances": proxy_distances,
                "ce_prediction": int(outputs["ce_control"][anchor].argmax()),
                "candidate_prediction": int(
                    outputs["rn_lisda_candidate"][anchor].argmax()
                ),
                "class1_note": (
                    ""
                    if enabled[row_number]
                    else "disabled by restricted-negative design"
                ),
            }
        )
    arrays = {
        "anchor_indices": anchors,
        "augmentation_enabled": enabled,
        "nearest_same_label_proxy": nearest_same_label,
        "fixed_proxy": fixed_proxy,
        "candidate_proxy_0": candidate_proxy_0,
        "candidate_proxy_1": candidate_proxy_1,
        "deranged_proxy": deranged_proxy,
        "fixed_draws": fixed_draws,
        "candidate_draws_0": candidate_draws_0,
        "candidate_draws_1": candidate_draws_1,
        "deranged_draws": deranged_draws,
        "fixed_epsilon": fixed_epsilon,
        "candidate_epsilon_0": candidate_epsilon_0,
        "candidate_epsilon_1": candidate_epsilon_1,
        "fixed_covariance": fixed_covariance,
        "candidate_covariance": candidate_covariance,
        "deranged_covariance": deranged_covariance,
    }
    if render_contact:
        _render_visual_contact_sheet(
            rows=rows,
            dataset=dataset,
            output_path=contact_sheet_path,
        )
    metadata = {
        "rows": rows,
        "row_count": len(rows),
        "ordered_indices_sha256": array_sha256(anchors),
        "class1_rows_disabled": int(np.sum(~enabled)),
        "all_proxies_outer_fit": True,
        "all_proxies_source_disjoint": True,
        "candidate_deranged_epsilon_exact": True,
        "contact_sheet": str(contact_sheet_path),
    }
    return arrays, metadata


def _build_replay_visual_evidence(
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    sources: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    views: Mapping[int, Mapping[str, np.ndarray]],
    lock: Mapping[str, object],
    output: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    formal_contact_sheet = output / "fixed_visual_contact_sheet.png"
    with tempfile.TemporaryDirectory(
        prefix="trkh_rn_lisda_replay_visual_"
    ) as temp_dir:
        replay_contact_sheet = (
            Path(temp_dir) / "fixed_visual_contact_sheet.png"
        )
        arrays, metadata = build_visual_evidence(
            cache=cache,
            folds=folds,
            sources=sources,
            outputs=outputs,
            views=views,
            lock=lock,
            contact_sheet_path=replay_contact_sheet,
            render_contact=True,
        )
    corrected_metadata = dict(metadata)
    corrected_metadata["contact_sheet"] = str(formal_contact_sheet)
    return arrays, corrected_metadata


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(_jsonable(row), ensure_ascii=True) + "\n"
            )


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(value)
    return rows


def _write_oof_rows(
    path: Path,
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    sources: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> None:
    fields = [
        "sample_index",
        "source_stem",
        "image_path",
        "outer_fold",
        "target_index",
    ]
    for role in ROLE_NAMES:
        fields.extend(
            f"{role}_prob_{class_index}"
            for class_index in range(CLASS_COUNT)
        )
        fields.append(f"{role}_prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(EXPECTED_ROWS):
            row: Dict[str, object] = {
                "sample_index": index,
                "source_stem": str(sources[index]),
                "image_path": str(cache["paths"][index]),
                "outer_fold": int(folds[index]),
                "target_index": int(cache["labels"][index]),
            }
            for role in ROLE_NAMES:
                for class_index in range(CLASS_COUNT):
                    row[f"{role}_prob_{class_index}"] = format(
                        float(outputs[role][index, class_index]), ".10g"
                    )
                row[f"{role}_prediction"] = int(
                    outputs[role][index].argmax()
                )
            writer.writerow(row)


def _read_oof_rows(
    path: Path,
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    sources: np.ndarray,
) -> Dict[str, np.ndarray]:
    outputs = {
        role: np.full(
            (EXPECTED_ROWS, CLASS_COUNT), np.nan, dtype=np.float64
        )
        for role in ROLE_NAMES
    }
    actions = {
        role: np.full(EXPECTED_ROWS, -1, dtype=np.int64)
        for role in ROLE_NAMES
    }
    observed = np.zeros(EXPECTED_ROWS, dtype=bool)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            index = int(row["sample_index"])
            if index < 0 or index >= EXPECTED_ROWS or observed[index]:
                raise ValueError(f"Invalid OOF CSV sample index: {index}")
            if str(row["source_stem"]).casefold() != str(
                sources[index]
            ).casefold():
                raise ValueError(f"OOF CSV source mismatch at {index}")
            if Path(str(row["image_path"])).name.casefold() != Path(
                str(cache["paths"][index])
            ).name.casefold():
                raise ValueError(f"OOF CSV path mismatch at {index}")
            if int(row["outer_fold"]) != int(folds[index]):
                raise ValueError(f"OOF CSV fold mismatch at {index}")
            if int(row["target_index"]) != int(cache["labels"][index]):
                raise ValueError(f"OOF CSV target mismatch at {index}")
            for role in ROLE_NAMES:
                outputs[role][index] = [
                    float(row[f"{role}_prob_{class_index}"])
                    for class_index in range(CLASS_COUNT)
                ]
                actions[role][index] = int(row[f"{role}_prediction"])
            observed[index] = True
    if not observed.all():
        raise ValueError(f"OOF CSV rows incomplete: {int(observed.sum())}")
    for role in ROLE_NAMES:
        if not np.isfinite(outputs[role]).all():
            raise ValueError(f"OOF CSV contains non-finite values: {role}")
        if not np.array_equal(actions[role], outputs[role].argmax(axis=1)):
            raise ValueError(f"OOF CSV action mismatch: {role}")
    return {
        **{role: outputs[role] for role in ROLE_NAMES},
        **{f"{role}__actions": actions[role] for role in ROLE_NAMES},
    }


def _write_fold_metrics(
    path: Path, analysis: Mapping[str, object]
) -> None:
    fields = [
        "role",
        "outer_fold",
        "samples",
        "accuracy",
        "macro_f1",
        "nll",
        "brier",
        "ece_15",
        "class1_precision",
        "class1_recall",
        "class1_f1",
        "class1_tp",
        "class1_fn",
        "restricted_fp",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        fold_metrics = analysis["fold_metrics"]
        assert isinstance(fold_metrics, Mapping)
        for role in ROLE_NAMES:
            role_rows = fold_metrics[role]
            assert isinstance(role_rows, list)
            for metrics in role_rows:
                focus = _focus(metrics)
                writer.writerow(
                    {
                        "role": role,
                        "outer_fold": int(metrics["outer_fold"]),
                        "samples": int(metrics["samples"]),
                        "accuracy": format(
                            float(metrics["accuracy"]), ".12g"
                        ),
                        "macro_f1": format(
                            float(metrics["macro_f1"]), ".12g"
                        ),
                        "nll": format(float(metrics["nll"]), ".12g"),
                        "brier": format(float(metrics["brier"]), ".12g"),
                        "ece_15": format(
                            float(metrics["ece_15"]), ".12g"
                        ),
                        "class1_precision": format(
                            float(focus["precision"]), ".12g"
                        ),
                        "class1_recall": format(
                            float(focus["recall"]), ".12g"
                        ),
                        "class1_f1": format(float(focus["f1"]), ".12g"),
                        "class1_tp": int(metrics["class1_tp"]),
                        "class1_fn": int(metrics["class1_fn"]),
                        "restricted_fp": int(metrics["restricted_fp"]),
                    }
                )


def _write_manifest(
    output_dir: Path,
    *,
    manifest_name: str,
    exclude: Sequence[str] = (),
) -> Dict[str, object]:
    manifest_path = output_dir / manifest_name
    excluded = {manifest_name, *exclude}
    files = []
    for path in sorted(
        (
            item
            for item in output_dir.rglob("*")
            if item.is_file()
            and item.relative_to(output_dir).as_posix() not in excluded
        ),
        key=lambda item: item.relative_to(output_dir).as_posix(),
    ):
        relative = path.relative_to(output_dir).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "manifest_name": manifest_name,
        "payload_count": len(files),
        "payload_bytes": int(sum(row["bytes"] for row in files)),
        "files": files,
    }
    _write_json(manifest_path, payload)
    return {
        **payload,
        "manifest_sha256": _sha256(manifest_path),
    }


def _verify_manifest(
    output_dir: Path, manifest_name: str
) -> Dict[str, object]:
    resolved_output = Path(output_dir).resolve()
    path = resolved_output / manifest_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Manifest has no file list: {path}")
    if int(payload.get("payload_count", -1)) != len(files):
        raise ValueError(f"Manifest payload count differs: {path}")
    listed_paths = [str(row["path"]) for row in files]
    if len(set(listed_paths)) != len(listed_paths):
        raise ValueError(f"Manifest contains duplicate paths: {path}")
    allowed_post_manifest = (
        {
            "replay.json",
            "formal_visual_review.json",
            "final_decision.json",
            "final_manifest.json",
        }
        if manifest_name == "artifact_manifest.json"
        else set()
    )
    ignored = {manifest_name, *allowed_post_manifest}
    actual_paths = {
        item.relative_to(resolved_output).as_posix()
        for item in resolved_output.rglob("*")
        if item.is_file()
        and item.relative_to(resolved_output).as_posix() not in ignored
    }
    if actual_paths != set(listed_paths):
        missing = sorted(set(listed_paths) - actual_paths)
        extra = sorted(actual_paths - set(listed_paths))
        raise ValueError(
            f"Manifest file set differs: missing={missing[:10]} "
            f"extra={extra[:10]}"
        )
    failures = []
    for row in files:
        target = (resolved_output / str(row["path"])).resolve()
        try:
            target.relative_to(resolved_output)
        except ValueError:
            failures.append(f"outside:{row['path']}")
            continue
        if not target.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        if int(target.stat().st_size) != int(row["bytes"]):
            failures.append(f"size:{row['path']}")
        if _sha256(target) != str(row["sha256"]):
            failures.append(f"sha256:{row['path']}")
    if failures:
        raise ValueError(f"Manifest verification failed: {failures[:10]}")
    observed_bytes = int(sum(int(row["bytes"]) for row in files))
    if int(payload.get("payload_bytes", -1)) != observed_bytes:
        raise ValueError(f"Manifest payload bytes differ: {path}")
    return {
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "payload_count": len(files),
        "payload_bytes": observed_bytes,
        "passed": True,
    }


def _recursive_numeric_difference(
    left: object, right: object, path: str = "root"
) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(f"Replay keys differ at {path}")
        return max(
            (
                _recursive_numeric_difference(
                    left[key], right[key], f"{path}.{key}"
                )
                for key in left
            ),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"Replay lengths differ at {path}")
        return max(
            (
                _recursive_numeric_difference(
                    a, b, f"{path}[{index}]"
                )
                for index, (a, b) in enumerate(zip(left, right))
            ),
            default=0.0,
        )
    left_is_boolean = isinstance(left, (bool, np.bool_))
    right_is_boolean = isinstance(right, (bool, np.bool_))
    if left_is_boolean or right_is_boolean:
        if (
            not left_is_boolean
            or not right_is_boolean
            or bool(left) != bool(right)
        ):
            raise ValueError(f"Replay boolean differs at {path}")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        if not math.isfinite(difference):
            raise ValueError(f"Replay non-finite difference at {path}")
        return difference
    if left != right:
        raise ValueError(f"Replay value differs at {path}: {left!r} != {right!r}")
    return 0.0


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def _maximum_array_error(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> float:
    if set(left) != set(right):
        raise ValueError("Replay array keys differ")
    maximum = 0.0
    for key in left:
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Replay array metadata differs: {key}")
        if np.issubdtype(a.dtype, np.number):
            maximum = max(
                maximum,
                float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                if a.size
                else 0.0,
            )
        elif not np.array_equal(a, b):
            raise ValueError(f"Replay nonnumeric array differs: {key}")
    return maximum


def _sanitized_partition_rows(
    partitions: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    excluded = {
        "fit_indices",
        "held_indices",
        "partition_a",
        "partition_b",
        "held_swap_donors",
    }
    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in partitions
    ]


def _training_trace_rows(
    results: Mapping[Tuple[str, int], Mapping[str, object]],
    key: str,
) -> Iterable[Dict[str, object]]:
    for role in ROLE_NAMES:
        for fold in range(5):
            rows = results[(role, fold)][key]
            assert isinstance(rows, list)
            for row in rows:
                yield dict(row)


def _post_run_locked_hashes(
    lock: Mapping[str, object],
) -> Dict[str, object]:
    names = (
        "current_best_commands",
        "current_best_history",
        "keeper_checkpoint",
    )
    immutable = lock["immutable_inputs"]
    return {
        name: _verify_file(
            REPO_ROOT, immutable[name], label=f"post_run_{name}"
        )
        for name in names
    }


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
    cache_path = _resolve_locked_path(
        REPO_ROOT, str(immutable["train_embedding_cache"]["path"])
    )
    cidt_path = _resolve_locked_path(
        REPO_ROOT, str(immutable["cidt_predictions"]["path"])
    )
    cache = load_locked_train_cache(cache_path)
    folds, sources, fold_evidence = load_locked_cidt_folds(cache, cidt_path)
    declaration = lock["dataset_declaration"]
    declaration_checks = {
        "rows_exact": int(declaration["rows"]) == int(len(cache["labels"])),
        "embedding_dim_exact": int(declaration["embedding_dim"])
        == int(cache["embeddings"].shape[1]),
        "class_count_exact": int(declaration["classes"]) == CLASS_COUNT,
        "class_counts_exact": tuple(
            int(value) for value in declaration["class_counts"]
        )
        == tuple(np.bincount(cache["labels"], minlength=CLASS_COUNT)),
        "outer_fold_counts_exact": tuple(
            int(value) for value in declaration["outer_fold_counts"]
        )
        == tuple(np.bincount(folds, minlength=5)),
        "outer_assignment_exact": str(
            declaration["outer_assignment_sha256"]
        )
        == str(fold_evidence["assignment_sha256"]),
        "outer_source_overlap_zero": (
            int(declaration["outer_source_overlap"]) == 0
            and int(fold_evidence["source_overlap"]) == 0
        ),
        "split_policy_exact": (
            tuple(str(value) for value in declaration["cache_split_allowlist"])
            == ("train",)
            and set(
                str(value)
                for value in declaration["cache_split_denylist"]
            )
            == {"val", "validation", "test"}
        ),
        "raw_mutation_forbidden": not bool(
            declaration["raw_dataset_mutation_allowed"]
        ),
        "derived_augmentation_allowed_exact": bool(
            declaration["derived_preprocessing_augmentation_allowed"]
        ),
        "train_derived_synthetic_allowed_exact": bool(
            declaration["synthetic_from_existing_train_allowed"]
        ),
    }
    if not all(declaration_checks.values()):
        failed = [
            name for name, passed in declaration_checks.items() if not passed
        ]
        raise ValueError(f"Dataset declaration mismatch: {failed}")
    fold_evidence["dataset_declaration_checks"] = declaration_checks
    cache["source_stems"] = sources
    partitions = build_locked_partitions(
        cache["labels"], folds, sources, lock
    )
    return cache, folds, sources, fold_evidence, partitions


def _data_access_evidence(
    cache: Mapping[str, np.ndarray],
    fold_evidence: Mapping[str, object],
    ledger: Mapping[str, object],
) -> Dict[str, object]:
    forbidden_cache_path_count = sum(
        int(_contains_forbidden_split(str(path)))
        for path in np.asarray(cache["paths"]).tolist()
    )
    forbidden_count = forbidden_cache_path_count + int(
        fold_evidence["forbidden_split_path_count"]
    )
    return {
        "cache_train_image_rows": int(len(cache["paths"])),
        "cidt_declared_train_image_rows": int(
            fold_evidence["declared_train_image_rows"]
        ),
        "forbidden_split_path_count": int(forbidden_count),
        "dynamic_hook_installed": bool(ledger["hook_installed"]),
        "dynamic_hook_probe_seen": bool(ledger["hook_probe_seen"]),
        "observed_data_domain_open_count": int(
            ledger["observed_event_count"]
        ),
        "observed_data_domain_unique_path_count": int(
            ledger["unique_path_count"]
        ),
        "blocked_attempt_count": int(ledger["blocked_attempt_count"]),
        "validation_open_count": int(ledger["validation_open_count"]),
        "test_open_count": int(ledger["test_open_count"]),
        "ordered_events_sha256": str(ledger["ordered_events_sha256"]),
        "dynamic_ledger_pass": bool(ledger["passed"]),
    }


def run_preflight(lock_path: Path) -> Dict[str, object]:
    resolved = Path(lock_path).expanduser().resolve()
    preview = json.loads(resolved.read_text(encoding="utf-8"))
    ledger = DataAccessLedger(preview)
    with ledger:
        return _run_preflight_with_ledger(resolved, ledger)


def _run_preflight_with_ledger(
    lock_path: Path,
    access_ledger: DataAccessLedger,
) -> Dict[str, object]:
    lock, locked_inputs = verify_locked_inputs(lock_path)
    cache, folds, sources, fold_evidence, partitions = _load_audit_inputs(lock)
    initial_states = _initial_state_evidence(lock)
    equation = equation_diagnostics()
    process = _process_snapshot()
    if process["unexpected_processes"]:
        raise RuntimeError(
            "Unrelated Python/TensorRT process exists before RN-LISDA A0"
        )
    ledger_snapshot = access_ledger.snapshot()
    return {
        "mode": "rn_lisda_a0_preflight",
        "passed": bool(ledger_snapshot["passed"]),
        "locked_inputs": locked_inputs,
        "dataset": {
            "rows": int(len(cache["labels"])),
            "class_counts": np.bincount(
                cache["labels"], minlength=CLASS_COUNT
            ).tolist(),
            "fold_counts": np.bincount(folds, minlength=5).tolist(),
            "source_count": int(len(set(sources.tolist()))),
            "validation_access_count": int(
                ledger_snapshot["validation_open_count"]
            ),
            "test_access_count": int(ledger_snapshot["test_open_count"]),
        },
        "data_access_evidence": _data_access_evidence(
            cache, fold_evidence, ledger_snapshot
        ),
        "data_access_ledger": ledger_snapshot,
        "fold_evidence": fold_evidence,
        "partitions": _sanitized_partition_rows(partitions),
        "initial_states": initial_states,
        "equation": equation,
        "process": process,
        "candidate_training_called": False,
        "candidate_metric_created": False,
    }


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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
        excluded_roots=(resolved_output,),
    )
    with ledger:
        return _run_formal_with_ledger(
            lock_path=resolved_lock,
            output_dir=resolved_output,
            device_name=device_name,
            access_ledger=ledger,
        )


def _run_formal_with_ledger(
    *,
    lock_path: Path,
    output_dir: Path,
    device_name: str,
    access_ledger: DataAccessLedger,
) -> Dict[str, object]:
    lock, locked_inputs = verify_locked_inputs(lock_path)
    if str(device_name) != str(lock["optimization"]["device"]):
        raise ValueError("Requested formal device differs from machine lock")
    cache, folds, sources, fold_evidence, partitions = _load_audit_inputs(lock)
    initial_states = _initial_state_evidence(lock)
    process_before = _process_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "Unrelated Python/TensorRT process exists before RN-LISDA formal"
        )
    start_memory = psutil.virtual_memory()
    start_available_gib = float(start_memory.available / (1024**3))
    data_yaml = _resolve_locked_path(
        REPO_ROOT,
        str(lock["immutable_inputs"]["yolo_data_yaml"]["path"]),
    )
    raw_before, raw_file_count = _raw_train_metadata_sha256(
        cache["paths"].tolist(), data_yaml
    )
    output = _prepare_output_dir(output_dir)
    device = torch.device(device_name)
    _configure_torch(device)
    started = time.perf_counter()
    with PeakResourceMonitor() as resource:
        outputs, results = train_all_roles(
            cache=cache,
            partitions=partitions,
            device=device,
        )
        analysis = analyze_predictions(cache["labels"], folds, outputs)
        performance_gate = assess_performance_gates(analysis, lock)
        mechanism, views = mechanism_evidence(
            cache=cache,
            partitions=partitions,
            results=results,
            lock=lock,
            device=device,
        )
        visual_arrays, visual_metadata = build_visual_evidence(
            cache=cache,
            folds=folds,
            sources=sources,
            outputs=outputs,
            views=views,
            lock=lock,
            contact_sheet_path=output / "fixed_visual_contact_sheet.png",
            render_contact=True,
        )
    elapsed_seconds = float(time.perf_counter() - started)
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    raw_after, raw_file_count_after = _raw_train_metadata_sha256(
        cache["paths"].tolist(), data_yaml
    )
    process_after = _process_snapshot()
    post_run_hashes = _post_run_locked_hashes(lock)
    repository_after = _repository_state(
        REPO_ROOT, lock.get("protected_untracked", [])
    )
    resource_lock = lock["gates"]["resources"]
    partition_hash_count = sum(
        sum(
            1
            for key in row
            if str(key).endswith("_sha256")
            and key
            not in {
                "raw_metadata_sha256",
            }
        )
        for row in partitions
    )
    data_access_ledger = access_ledger.snapshot()
    data_access_evidence = _data_access_evidence(
        cache, fold_evidence, data_access_ledger
    )
    declared_forbidden_path_count = int(
        data_access_evidence["forbidden_split_path_count"]
    )
    access_lock = lock["gates"]["data_access"]
    structural_checks = {
        "locked_inputs_verified": bool(locked_inputs["passed"]),
        "repo_clean_pushed_and_protected_exact": bool(
            repository_after["passed"]
        ),
        "train_rows_exact": int(len(cache["labels"])) == EXPECTED_ROWS,
        "train_class_counts_exact": tuple(
            np.bincount(cache["labels"], minlength=CLASS_COUNT)
        )
        == EXPECTED_CLASS_COUNTS,
        "outer_fold_counts_exact": tuple(
            np.bincount(folds, minlength=5)
        )
        == EXPECTED_FOLD_COUNTS,
        "outer_source_overlap_zero": (
            int(fold_evidence["source_overlap"]) == 0
        ),
        "all_65_partition_hashes_exact": (
            len(partitions) == 5 and partition_hash_count == 65
        ),
        "all_inner_source_overlaps_zero": all(
            int(row["source_overlap"]) == 0 for row in partitions
        ),
        "all_initial_state_hashes_exact": len(initial_states) == 5,
        "exact_role_count": len(results) == len(ROLE_NAMES) * 5,
        "all_oof_outputs_complete": all(
            outputs[role].shape == (EXPECTED_ROWS, CLASS_COUNT)
            and np.isfinite(outputs[role]).all()
            for role in ROLE_NAMES
        ),
        "all_initial_states_exact": bool(
            mechanism["checks"]["all_initial_states_exact"]
        ),
        "all_partition_provenance_exact": bool(
            mechanism["checks"]["all_partition_provenance_exact"]
        ),
        "all_paired_orders_common": bool(
            mechanism["checks"]["all_paired_orders_common"]
        ),
        "outer_holdout_scored_once_after_epoch_30": bool(
            mechanism["checks"]["all_holdout_score_contract_exact"]
        ),
        "dynamic_data_access_hook_installed": (
            bool(data_access_evidence["dynamic_hook_installed"])
            == bool(access_lock["dynamic_hook_installed"])
        ),
        "dynamic_data_access_hook_probe_seen": (
            bool(data_access_evidence["dynamic_hook_probe_seen"])
            == bool(access_lock["dynamic_hook_probe_seen"])
        ),
        "observed_data_domain_paths_train_only": (
            bool(data_access_evidence["dynamic_ledger_pass"])
            == bool(access_lock["observed_data_domain_paths_train_only"])
        ),
        "data_access_blocked_attempt_count_zero": (
            int(data_access_evidence["blocked_attempt_count"])
            <= int(access_lock["blocked_attempt_count_max"])
        ),
        "validation_access_count_zero": (
            declared_forbidden_path_count == 0
            and int(data_access_evidence["validation_open_count"])
            <= int(access_lock["validation_open_count_max"])
        ),
        "test_access_count_zero": (
            declared_forbidden_path_count == 0
            and int(data_access_evidence["test_open_count"])
            <= int(access_lock["test_open_count_max"])
        ),
        "raw_dataset_unchanged": (
            raw_before == raw_after
            and raw_file_count == raw_file_count_after
        ),
        "all_update_counts_exact": bool(
            mechanism["checks"]["all_update_counts_exact"]
        ),
        "all_states_losses_gradients_finite": bool(
            mechanism["checks"]["all_role_values_finite"]
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config_exact": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            == EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ),
        "cuda_matmul_tf32_disabled": not bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_tf32_disabled": not bool(torch.backends.cudnn.allow_tf32),
        "visual_rows_exact_20": (
            int(visual_metadata["row_count"])
            == int(lock["gates"]["mechanism"]["fixed_visual_rows"])
        ),
        "visual_proxy_source_and_fold_safe": bool(
            visual_metadata["all_proxies_outer_fit"]
            and visual_metadata["all_proxies_source_disjoint"]
        ),
        "start_available_ram_at_least_4gib": (
            start_available_gib
            >= float(
                resource_lock["start_available_physical_ram_gib_min"]
            )
        ),
        "peak_process_rss_below_4gib": (
            float(resource.peak_rss_bytes / (1024**3))
            < float(resource_lock["peak_process_rss_gib_max"])
        ),
        "peak_cuda_below_2gib": (
            float(peak_cuda_bytes / (1024**3))
            < float(resource_lock["peak_cuda_allocated_gib_max"])
        ),
        "elapsed_below_45_minutes": (
            elapsed_seconds / 60.0
            < float(resource_lock["formal_elapsed_minutes_max"])
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
        "current_best_and_keeper_bit_identical": all(
            bool(row["passed"]) for row in post_run_hashes.values()
        ),
    }
    structural_pass = all(structural_checks.values())
    automatic_pass = bool(
        structural_pass
        and mechanism["mechanism_pass"]
        and performance_gate["performance_pass"]
    )

    fold_protocol = {
        "fold_evidence": fold_evidence,
        "partitions": _sanitized_partition_rows(partitions),
        "initial_states": initial_states,
        "hash_encoding": lock["hash_encoding"],
    }
    _write_json(output / "fold_protocol.json", fold_protocol)
    _write_jsonl(
        output / "training_trace.jsonl",
        _training_trace_rows(results, "trace"),
    )
    _write_jsonl(
        output / "meta_trace.jsonl",
        _training_trace_rows(results, "meta_trace"),
    )
    np.savez_compressed(
        output / "oof_probabilities.npz",
        sample_index=np.arange(EXPECTED_ROWS, dtype=np.int64),
        targets=np.asarray(cache["labels"], dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
        **{
            role: np.asarray(outputs[role], dtype=np.float32)
            for role in ROLE_NAMES
        },
    )
    _write_oof_rows(
        output / "oof_rows.csv",
        cache=cache,
        folds=folds,
        sources=sources,
        outputs=outputs,
    )
    _write_fold_metrics(output / "fold_metrics.csv", analysis)
    np.savez_compressed(
        output / "role_states.npz", **_flatten_role_states(results)
    )
    _write_json(output / "mechanism_telemetry.json", mechanism)
    _write_json(output / "data_access_ledger.json", data_access_ledger)
    np.savez_compressed(
        output / "fixed_visual_arrays.npz", **visual_arrays
    )
    _write_json(output / "fixed_visual_metadata.json", visual_metadata)

    summary = {
        "mode": "restricted_negative_learnable_isda_a0",
        "status": (
            "automatic_pass_manual_and_replay_pending"
            if automatic_pass
            else "automatic_reject_manual_and_replay_pending"
        ),
        "lock_sha256": locked_inputs["lock_sha256"],
        "protocol_sha256": locked_inputs["protocol_sha256"],
        "git_head": locked_inputs["repository"]["head"],
        "train_only": True,
        "encoder_oof": False,
        "validation_access_count": int(
            data_access_evidence["validation_open_count"]
        ),
        "test_access_count": int(data_access_evidence["test_open_count"]),
        "raw_dataset_modified": raw_before != raw_after,
        "dataset": {
            "rows": EXPECTED_ROWS,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "embedding_sha256": array_sha256(cache["embeddings"]),
            "labels_sha256": array_sha256(cache["labels"]),
            "folds_sha256": array_sha256(folds),
            "source_stems_sha256": string_sequence_sha256(sources),
            "raw_metadata_sha256": raw_after,
            "raw_metadata_file_count": raw_file_count_after,
        },
        "analysis": analysis,
        "performance_gate": performance_gate,
        "mechanism_gate": mechanism,
        "visual_artifact": {
            "metadata": "fixed_visual_metadata.json",
            "arrays": "fixed_visual_arrays.npz",
            "contact_sheet": "fixed_visual_contact_sheet.png",
            "rows": int(visual_metadata["row_count"]),
            "manual_review_required": True,
        },
        "structural_checks": structural_checks,
        "structural_pass": structural_pass,
        "automatic_pass": automatic_pass,
        "fresh_process_replay_required": True,
        "manual_visual_review_required": True,
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
        "authorization": {
            "default_off_production_integration": False,
            "validation_smoke": False,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_update": False,
        },
        "claim_boundary": (
            "Frozen train-embedding source-held readout gate only; not "
            "encoder-OOF, validation, test, or end-to-end evidence."
        ),
        "data_access_evidence": data_access_evidence,
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
    verified_manifest = _verify_manifest(output, "artifact_manifest.json")
    return {
        "status": summary["status"],
        "automatic_pass": automatic_pass,
        "summary_sha256": _sha256(output / "summary.json"),
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "artifact_manifest_verified": verified_manifest["passed"],
        "structural_pass": structural_pass,
        "mechanism_pass": mechanism["mechanism_pass"],
        "performance_pass": performance_gate["performance_pass"],
        "authorization": summary["authorization"],
    }


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
        excluded_roots=(resolved_summary.parent,),
    )
    with ledger:
        return _replay_formal_with_ledger(
            lock_path=resolved_lock,
            summary_path=resolved_summary,
            device_name=device_name,
            access_ledger=ledger,
        )


def _replay_formal_with_ledger(
    *,
    lock_path: Path,
    summary_path: Path,
    device_name: str,
    access_ledger: DataAccessLedger,
) -> Dict[str, object]:
    summary_path = Path(summary_path).expanduser().resolve()
    output = summary_path.parent
    if (output / "replay.json").exists():
        raise FileExistsError("Replay evidence already exists")
    formal_manifest = _verify_manifest(output, "artifact_manifest.json")
    formal_summary_sha = _sha256(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock, locked_inputs = verify_locked_inputs(lock_path)
    if str(device_name) != str(lock["optimization"]["device"]):
        raise ValueError("Requested replay device differs from machine lock")
    if str(summary["git_head"]) != str(
        locked_inputs["repository"]["head"]
    ):
        raise ValueError("Replay git HEAD differs from formal HEAD")
    if str(summary["lock_sha256"]) != str(locked_inputs["lock_sha256"]):
        raise ValueError("Replay lock SHA differs from formal")
    process_before = _process_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError("Unrelated Python/TensorRT process exists at replay")
    cache, folds, sources, fold_evidence, partitions = _load_audit_inputs(lock)
    _initial_state_evidence(lock)
    data_yaml = _resolve_locked_path(
        REPO_ROOT,
        str(lock["immutable_inputs"]["yolo_data_yaml"]["path"]),
    )
    replay_raw_metadata, replay_raw_file_count = _raw_train_metadata_sha256(
        cache["paths"].tolist(),
        data_yaml,
    )
    device = torch.device(device_name)
    _configure_torch(device)
    started = time.perf_counter()
    with PeakResourceMonitor() as replay_resource:
        outputs, results = train_all_roles(
            cache=cache,
            partitions=partitions,
            device=device,
        )
    analysis = analyze_predictions(cache["labels"], folds, outputs)
    performance_gate = assess_performance_gates(analysis, lock)
    mechanism, views = mechanism_evidence(
        cache=cache,
        partitions=partitions,
        results=results,
        lock=lock,
        device=device,
    )
    visual_arrays, visual_metadata = _build_replay_visual_evidence(
        cache=cache,
        folds=folds,
        sources=sources,
        outputs=outputs,
        views=views,
        lock=lock,
        output=output,
    )
    elapsed_seconds = float(time.perf_counter() - started)
    post_run_hashes = _post_run_locked_hashes(lock)
    data_access_ledger = access_ledger.snapshot()
    data_access_evidence = _data_access_evidence(
        cache, fold_evidence, data_access_ledger
    )
    stored_data_access_ledger = json.loads(
        (output / "data_access_ledger.json").read_text(encoding="utf-8")
    )
    data_access_ledger_error = _recursive_numeric_difference(
        stored_data_access_ledger, data_access_ledger
    )

    stored_oof = _read_npz(output / "oof_probabilities.npz")
    recomputed_oof = {
        "sample_index": np.arange(EXPECTED_ROWS, dtype=np.int64),
        "targets": np.asarray(cache["labels"], dtype=np.int64),
        "folds": np.asarray(folds, dtype=np.int64),
        **{
            role: np.asarray(outputs[role], dtype=np.float32)
            for role in ROLE_NAMES
        },
    }
    probability_error = _maximum_array_error(stored_oof, recomputed_oof)
    actions_exact = all(
        np.array_equal(
            stored_oof[role].argmax(axis=1),
            recomputed_oof[role].argmax(axis=1),
        )
        for role in ROLE_NAMES
    )
    stored_csv = _read_oof_rows(
        output / "oof_rows.csv",
        cache=cache,
        folds=folds,
        sources=sources,
    )
    csv_probability_error = max(
        float(
            np.max(
                np.abs(
                    stored_csv[role]
                    - np.asarray(outputs[role], dtype=np.float64)
                )
            )
        )
        for role in ROLE_NAMES
    )
    csv_actions_exact = all(
        np.array_equal(
            stored_csv[f"{role}__actions"],
            outputs[role].argmax(axis=1),
        )
        for role in ROLE_NAMES
    )
    stored_states = _read_npz(output / "role_states.npz")
    recomputed_states = _flatten_role_states(results)
    state_error = _maximum_array_error(stored_states, recomputed_states)
    stored_training_trace = _read_jsonl(output / "training_trace.jsonl")
    recomputed_training_trace = list(
        _training_trace_rows(results, "trace")
    )
    training_trace_error = _recursive_numeric_difference(
        stored_training_trace, recomputed_training_trace
    )
    stored_meta_trace = _read_jsonl(output / "meta_trace.jsonl")
    recomputed_meta_trace = list(_training_trace_rows(results, "meta_trace"))
    meta_trace_error = _recursive_numeric_difference(
        stored_meta_trace, recomputed_meta_trace
    )
    stored_fold_protocol = json.loads(
        (output / "fold_protocol.json").read_text(encoding="utf-8")
    )
    recomputed_fold_protocol = {
        "fold_evidence": fold_evidence,
        "partitions": _sanitized_partition_rows(partitions),
        "initial_states": _initial_state_evidence(lock),
        "hash_encoding": lock["hash_encoding"],
    }
    fold_protocol_error = _recursive_numeric_difference(
        stored_fold_protocol, recomputed_fold_protocol
    )
    analysis_error = _recursive_numeric_difference(
        summary["analysis"], analysis
    )
    performance_error = _recursive_numeric_difference(
        summary["performance_gate"], performance_gate
    )
    mechanism_error = _recursive_numeric_difference(
        summary["mechanism_gate"], mechanism
    )
    data_access_error = _recursive_numeric_difference(
        summary["data_access_evidence"], data_access_evidence
    )
    stored_visual_metadata = json.loads(
        (output / "fixed_visual_metadata.json").read_text(encoding="utf-8")
    )
    visual_metadata_error = _recursive_numeric_difference(
        stored_visual_metadata, visual_metadata
    )
    stored_visual_arrays = _read_npz(output / "fixed_visual_arrays.npz")
    visual_keys_exact = set(stored_visual_arrays) == set(visual_arrays)
    visual_arrays_exact = visual_keys_exact and all(
        array_sha256(stored_visual_arrays[key])
        == array_sha256(visual_arrays[key])
        for key in stored_visual_arrays
    )
    replay_thresholds = lock["gates"]["replay"]
    checks = {
        "pre_replay_manifest_verified": bool(formal_manifest["passed"]),
        "locked_inputs_verified": bool(locked_inputs["passed"]),
        "git_head_exact": (
            str(summary["git_head"])
            == str(locked_inputs["repository"]["head"])
        ),
        "probabilities_within_tolerance": (
            probability_error
            <= float(replay_thresholds["probability_max_error"])
        ),
        "actions_exact": actions_exact,
        "oof_csv_probabilities_within_tolerance": (
            csv_probability_error
            <= float(replay_thresholds["probability_max_error"])
        ),
        "oof_csv_actions_exact": csv_actions_exact,
        "states_within_tolerance": (
            state_error <= float(replay_thresholds["state_max_error"])
        ),
        "training_trace_within_tolerance": (
            training_trace_error
            <= float(replay_thresholds["nested_metric_max_error"])
        ),
        "meta_trace_within_tolerance": (
            meta_trace_error
            <= float(replay_thresholds["nested_metric_max_error"])
        ),
        "fold_protocol_exact": fold_protocol_error == 0.0,
        "analysis_within_tolerance": (
            analysis_error
            <= float(replay_thresholds["nested_metric_max_error"])
        ),
        "performance_gate_within_tolerance": (
            performance_error
            <= float(replay_thresholds["nested_metric_max_error"])
        ),
        "mechanism_within_tolerance": (
            mechanism_error
            <= float(replay_thresholds["nested_metric_max_error"])
        ),
        "data_access_evidence_exact": data_access_error == 0.0,
        "data_access_ledger_exact": data_access_ledger_error == 0.0,
        "data_access_ledger_pass": (
            bool(data_access_ledger["passed"])
            and int(data_access_ledger["blocked_attempt_count"]) == 0
            and int(data_access_ledger["validation_open_count"]) == 0
            and int(data_access_ledger["test_open_count"]) == 0
        ),
        "raw_dataset_metadata_exact": (
            str(summary["dataset"]["raw_metadata_sha256"])
            == replay_raw_metadata
            and int(summary["dataset"]["raw_metadata_file_count"])
            == replay_raw_file_count
        ),
        "visual_metadata_exact": visual_metadata_error == 0.0,
        "visual_arrays_exact": visual_arrays_exact,
        "current_best_and_keeper_bit_identical": all(
            row["passed"] for row in post_run_hashes.values()
        ),
    }
    process_after = _process_snapshot()
    checks["no_unrelated_process_after"] = not bool(
        process_after["unexpected_processes"]
    )
    checks["no_unrelated_process_during"] = not bool(
        replay_resource.unexpected_processes
    )
    passed = all(checks.values())
    replay = {
        "mode": "restricted_negative_learnable_isda_a0_replay",
        "status": "passed" if passed else "failed",
        "formal_summary_sha256": formal_summary_sha,
        "formal_manifest_sha256": formal_manifest["manifest_sha256"],
        "git_head": locked_inputs["repository"]["head"],
        "checks": checks,
        "passed": passed,
        "errors": {
            "probability_max_error": probability_error,
            "oof_csv_probability_max_error": csv_probability_error,
            "state_max_error": state_error,
            "training_trace_max_error": training_trace_error,
            "meta_trace_max_error": meta_trace_error,
            "fold_protocol_max_error": fold_protocol_error,
            "analysis_max_error": analysis_error,
            "performance_gate_max_error": performance_error,
            "mechanism_max_error": mechanism_error,
            "data_access_max_error": data_access_error,
            "data_access_ledger_max_error": data_access_ledger_error,
            "visual_metadata_max_error": visual_metadata_error,
            "raw_metadata_sha256": replay_raw_metadata,
            "raw_metadata_file_count": replay_raw_file_count,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "process_before": process_before,
            "process_after": process_after,
            "unexpected_processes_during": sorted(
                replay_resource.unexpected_processes.values(),
                key=lambda row: int(row["pid"]),
            ),
        },
        "validation_access_count": int(
            data_access_evidence["validation_open_count"]
        ),
        "test_access_count": int(data_access_evidence["test_open_count"]),
        "scientific_artifacts_modified": False,
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
        raise FileExistsError("Formal visual review already exists")
    if decision not in {"pass", "reject"}:
        raise ValueError("Manual decision must be pass or reject")
    if len(review_note.strip()) < 20:
        raise ValueError("Manual review note is too short")
    formal_manifest = _verify_manifest(output, "artifact_manifest.json")
    summary_path = output / "summary.json"
    observed_summary_sha = _sha256(summary_path)
    if observed_summary_sha != expected_summary_sha256:
        raise ValueError("Expected formal summary SHA does not match")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay_path = output / "replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if str(replay["formal_summary_sha256"]) != observed_summary_sha:
        raise ValueError("Replay references a different formal summary")
    lock, locked_inputs = verify_locked_inputs(lock_path)
    metadata = json.loads(
        (output / "fixed_visual_metadata.json").read_text(encoding="utf-8")
    )
    if int(metadata["row_count"]) != 20:
        raise ValueError("Manual review requires all 20 fixed rows")
    contact = output / "fixed_visual_contact_sheet.png"
    if not contact.is_file():
        raise FileNotFoundError("Fixed contact sheet is missing")
    checklist = {
        "all_rows_reviewed": True,
        "class1_augmentation_disabled": (
            int(metadata["class1_rows_disabled"]) == 5
        ),
        "all_proxies_outer_fit": bool(metadata["all_proxies_outer_fit"]),
        "all_proxies_source_disjoint": bool(
            metadata["all_proxies_source_disjoint"]
        ),
        "candidate_deranged_epsilon_exact": bool(
            metadata["candidate_deranged_epsilon_exact"]
        ),
    }
    manual_pass = decision == "pass" and all(checklist.values())
    automatic_pass = bool(summary["automatic_pass"])
    replay_pass = bool(replay["passed"])
    authorized = bool(automatic_pass and replay_pass and manual_pass)
    review = {
        "mode": "restricted_negative_learnable_isda_a0_visual_review",
        "formal_summary_sha256": observed_summary_sha,
        "replay_sha256": _sha256(replay_path),
        "contact_sheet_sha256": _sha256(contact),
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
        "mode": "restricted_negative_learnable_isda_a0_final_decision",
        "status": "passed" if authorized else "rejected",
        "formal_summary_sha256": observed_summary_sha,
        "formal_manifest_sha256": formal_manifest["manifest_sha256"],
        "replay_sha256": _sha256(replay_path),
        "visual_review_sha256": _sha256(
            output / "formal_visual_review.json"
        ),
        "automatic_pass": automatic_pass,
        "replay_pass": replay_pass,
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
    verified_final_manifest = _verify_manifest(output, "final_manifest.json")
    return {
        "status": final_decision["status"],
        "authorized": authorized,
        "final_decision_sha256": _sha256(output / "final_decision.json"),
        "final_manifest_sha256": final_manifest["manifest_sha256"],
        "final_manifest_verified": verified_final_manifest["passed"],
        "authorization": final_decision["authorization"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective train-only restricted-negative LearnableISDA A0 "
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
        raise ValueError("Choose exactly one RN-LISDA audit phase")
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
    result = run_audit(parse_args(argv))
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

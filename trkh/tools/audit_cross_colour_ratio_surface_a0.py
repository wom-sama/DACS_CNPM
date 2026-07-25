from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from PIL import Image, ImageEnhance
import psutil
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from trkh.tools.cross_colour_ratio_surface_a0_engine import (
    BATCH_SIZE,
    CLASS_ORDER,
    COLOUR_RATIO_MODE,
    CROSS_COLOUR_RATIO_MODE,
    EPOCHS,
    FOCUS_CLASS,
    ROLE_NAMES,
    VIEW_SIZE,
    ColourDerivativeDescriptorExtractor,
    CrossColourRatioEvidenceHead,
    apply_keeper_suppression,
    array_sha256,
    build_dephase_offsets,
    build_epoch_orders,
    calibrate_class1_retention_threshold,
    class1_evidence_score,
    initialize_role_head,
    json_sha256,
    keeper_margin,
    learning_rate,
    map_targets_to_head_indices,
    model_state_arrays,
    parameter_contract,
    role_seed,
    spatially_dephase,
    state_arrays_sha256,
)
from trkh.tools.cross_colour_ratio_surface_a0_materializer import (
    unpack_valid_masks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_LOCK_20260725.json"
)
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / (
        "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_"
        "MATERIALIZER_RECOVERY_EVIDENCE_20260725.json"
    )
)
DEFAULT_AUTHORIZATION = (
    REPO_ROOT
    / "docs"
    / (
        "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_"
        "FIT_RECOVERY_AUTHORIZATION_20260725.json"
    )
)
FAILURE_EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / (
        "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_"
        "FIT_FAILURE_20260725.json"
    )
)
DEFAULT_CACHE = (
    REPO_ROOT
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_materialized_20260725"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_20260725"
)
ENGINE_PATH = (
    REPO_ROOT
    / "trkh"
    / "tools"
    / "cross_colour_ratio_surface_a0_engine.py"
)
RUNNER_PATH = Path(__file__).resolve()
RUNNER_TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_audit_cross_colour_ratio_surface_a0.py"
)
LAUNCHER_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_trkh_cross_colour_ratio_surface_a0.ps1"
)
IMPLEMENTATION_NOTE_PATH = (
    REPO_ROOT
    / "docs"
    / (
        "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_"
        "FIT_IMPLEMENTATION_20260725.md"
    )
)

SRGB_CACHE_NAME = "model_srgb_uint8.npy"
VALID_MASK_CACHE_NAME = "image_valid_masks_packbits.npy"
COHORT_ARRAYS_NAME = "cohort_arrays.npz"
FORMAL_MANIFEST_NAME = "formal_manifest.json"
ARTIFACT_SET_MANIFEST_NAME = "artifact_set_manifest.json"
REPLAY_SUMMARY_NAME = "replay_summary.json"
REPLAY_MANIFEST_NAME = "replay_manifest.json"

ROWS = 763
FULL_TRAIN_ROWS = 9215
FOLDS = 5
DESCRIPTOR_BATCH_SIZE = 32
SCORE_BATCH_SIZE = 64
REQUESTED_WORKERS = 4
MINIMUM_CLASS1_RETENTION = 0.97
REPLAY_NUMERIC_TOLERANCE = 1e-7
RESTRICTED_RIVALS = (0, 2, 4)
CONSUMED_AUTHORIZATION_SHA256 = (
    "3ecb6a82d14980e25cd947007561294fa"
    "39cfe5419032cb3961389a30d1697b5"
)

CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)

PROTECTED_REPOSITORY_UNTRACKED = frozenset(
    {
        "BaoCao/",
        "deep-research-report (9).md",
        "deep-research-report (10).md",
    }
)
FORBIDDEN_SPLIT_COMPONENTS = frozenset(
    {"val", "valid", "validation", "test"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings_sha256(values: Iterable[object]) -> str:
    encoded = [str(value).encode("utf-8") for value in values]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
    for value in encoded:
        digest.update(np.asarray([len(value)], dtype=np.int64).tobytes())
        digest.update(value)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_sidecar_json(path: Path) -> tuple[Dict[str, object], str]:
    path = Path(path).resolve()
    sidecar_path = path.with_suffix(".sha256")
    sidecar = sidecar_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) != 2 or sidecar[1] != path.name:
        raise ValueError(f"Invalid SHA sidecar: {sidecar_path}")
    observed = sha256_file(path)
    if observed != sidecar[0]:
        raise ValueError(f"SHA sidecar differs: {path}")
    return _read_json(path), observed


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    payload = {
        str(name): np.ascontiguousarray(value)
        for name, value in values.items()
    }
    if any(array.dtype.hasobject for array in payload.values()):
        raise ValueError("Object arrays are forbidden in CCR artifacts")
    np.savez_compressed(path, **payload)


def _npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
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


def _artifact_record(path: Path) -> Dict[str, object]:
    path = Path(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


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


def _verify_manifest(root: Path, name: str) -> Dict[str, object]:
    path = Path(root) / name
    stored = _read_json(path)
    excluded = {name}
    if name == FORMAL_MANIFEST_NAME:
        excluded.update(
            {
                ARTIFACT_SET_MANIFEST_NAME,
                REPLAY_MANIFEST_NAME,
                REPLAY_SUMMARY_NAME,
            }
        )
    elif name == REPLAY_MANIFEST_NAME:
        excluded.update({ARTIFACT_SET_MANIFEST_NAME})
    observed = _directory_manifest(root, excluded_names=tuple(excluded))
    passed = stored == observed
    if not passed:
        raise RuntimeError(f"CCR artifact manifest differs: {name}")
    return {
        "passed": True,
        "manifest_sha256": sha256_file(path),
        "rows_sha256": stored["rows_sha256"],
        "file_count": stored["file_count"],
        "total_bytes": stored["total_bytes"],
    }


def _normalized_path(value: object) -> Optional[str]:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        return os.path.normcase(
            os.path.abspath(os.fsdecode(value))
        ).replace("/", "\\")
    except (OSError, TypeError, ValueError):
        return None


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _path_components(path: str) -> set[str]:
    return {
        component.casefold()
        for component in Path(path).parts
        if component not in ("", os.sep)
    }


_ACCESS_LOCK = threading.RLock()
_ACTIVE_LEDGER: Optional["FitDataAccessLedger"] = None
_HOOK_INSTALLED = False
_HOOK_PROBE_SEEN = False


def _audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    global _HOOK_PROBE_SEEN
    if event == "trkh.cross_colour_ratio_surface.fit_data_access_probe":
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
                "trkh.cross_colour_ratio_surface.fit_data_access_probe"
            )
            if not _HOOK_PROBE_SEEN:
                raise RuntimeError("CCR fit data-access probe was not observed")
            _HOOK_INSTALLED = True


class FitDataAccessLedger:
    def __init__(
        self,
        *,
        allowed_paths: Sequence[Path],
        excluded_roots: Sequence[Path] = (),
        domain_roots: Optional[Mapping[str, Path]] = None,
    ) -> None:
        roots = domain_roots or {
            "dataset": Path(r"D:\DataAI\AIEx\newdataset"),
            "runs": REPO_ROOT / "runs",
            "external": Path(r"D:\DataAI\external_sources"),
        }
        self.domain_roots = {
            name: _normalized_path(path)
            for name, path in roots.items()
        }
        if any(value is None for value in self.domain_roots.values()):
            raise ValueError("CCR fit ledger has an invalid domain root")
        self.allowed_exact_paths = frozenset(
            value
            for value in (
                _normalized_path(path) for path in allowed_paths
            )
            if value is not None
        )
        self.excluded_roots = tuple(
            value
            for value in (
                _normalized_path(path) for path in excluded_roots
            )
            if value is not None
        )
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

    def observe_open(self, arguments: tuple[object, ...]) -> None:
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
            _path_components(path).intersection(
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
        }
        self.events.append(record)
        if reason is not None:
            self.blocked_attempts.append(dict(record))
            raise PermissionError(
                f"CCR fit data-access audit blocked {reason}: {path}"
            )

    def __enter__(self) -> "FitDataAccessLedger":
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
                int(name in row["forbidden_components"])
                for row in events
            )
            for name in sorted(FORBIDDEN_SPLIT_COMPONENTS)
        }
        return {
            "schema_version": 1,
            "hook_installed": bool(_HOOK_INSTALLED),
            "hook_probe_seen": bool(_HOOK_PROBE_SEEN),
            "domain_roots": self.domain_roots,
            "excluded_roots": list(self.excluded_roots),
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
            "ordered_events_sha256": hashlib.sha256(
                canonical
            ).hexdigest(),
            "events": events,
            "blocked_attempts": [
                dict(row) for row in self.blocked_attempts
            ],
            "passed": bool(
                _HOOK_INSTALLED
                and _HOOK_PROBE_SEEN
                and events
                and not self.blocked_attempts
            ),
        }


def _repository_state(required_ancestor: str) -> Dict[str, object]:
    def git(
        *arguments: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
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
        record
        for record in git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=normal",
        ).stdout.split("\0")
        if record
    ]

    def protected(record: str) -> bool:
        if not record.startswith("?? "):
            return False
        path = record[3:].replace("\\", "/")
        return path in PROTECTED_REPOSITORY_UNTRACKED or any(
            protected_path.endswith("/")
            and path.startswith(protected_path)
            for protected_path in PROTECTED_REPOSITORY_UNTRACKED
        )

    unexpected = [record for record in status if not protected(record)]
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
        raise RuntimeError(f"CCR fit repository gate failed: {result}")
    return result


def _verify_file_record(
    path: Path,
    record: Mapping[str, object],
) -> Dict[str, object]:
    path = Path(path).resolve()
    observed = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    observed["passed"] = bool(
        observed["exists"]
        and int(observed["bytes"]) == int(record["bytes"])
        and observed["sha256"] == record["sha256"]
    )
    if not observed["passed"]:
        raise ValueError(f"CCR fit input differs: {path}")
    return observed


def _verify_materialized_cache(
    *,
    cache_dir: Path,
    evidence: Mapping[str, object],
) -> Dict[str, object]:
    cache_dir = Path(cache_dir).resolve()
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("CCR materializer evidence has no artifacts block")
    expected_dir = (REPO_ROOT / str(artifacts["output_dir"])).resolve()
    if cache_dir != expected_dir:
        raise ValueError("CCR materializer cache path differs")
    file_records = artifacts.get("files")
    if not isinstance(file_records, Mapping):
        raise ValueError("CCR materializer evidence has no file records")
    observed = {
        name: _verify_file_record(
            cache_dir / str(name),
            record,
        )
        for name, record in file_records.items()
        if isinstance(record, Mapping)
    }
    for manifest_name, evidence_key in (
        (FORMAL_MANIFEST_NAME, "formal_manifest"),
        (ARTIFACT_SET_MANIFEST_NAME, "artifact_set_manifest"),
    ):
        manifest_record = artifacts.get(evidence_key)
        if not isinstance(manifest_record, Mapping):
            raise ValueError(f"Missing evidence record: {evidence_key}")
        manifest_path = cache_dir / manifest_name
        manifest = _read_json(manifest_path)
        if (
            sha256_file(manifest_path)
            != manifest_record["file_sha256"]
            or manifest["file_count"]
            != manifest_record["file_count"]
            or manifest["total_bytes"]
            != manifest_record["total_bytes"]
            or manifest["rows_sha256"]
            != manifest_record["rows_sha256"]
        ):
            raise ValueError(f"CCR cache manifest differs: {manifest_name}")
    artifact_set_path = cache_dir / ARTIFACT_SET_MANIFEST_NAME
    stored_artifact_set = _read_json(artifact_set_path)
    observed_artifact_set = _directory_manifest(
        cache_dir,
        excluded_names=(ARTIFACT_SET_MANIFEST_NAME,),
    )
    if observed_artifact_set != stored_artifact_set:
        raise ValueError("CCR cache exact artifact set differs")
    return {
        "passed": True,
        "cache_dir": str(cache_dir),
        "files": observed,
        "artifact_set_manifest_sha256": sha256_file(
            cache_dir / ARTIFACT_SET_MANIFEST_NAME
        ),
        "source_cache_bytes": int(
            sum(
                path.stat().st_size
                for path in cache_dir.iterdir()
                if path.is_file()
            )
        ),
    }


def _verify_authorization(
    authorization: Mapping[str, object],
    *,
    authorization_sha256: str,
    lock_sha256: str,
    evidence_sha256: str,
    failure_evidence: Mapping[str, object],
    failure_evidence_sha256: str,
    cache_dir: Path,
    output: Path,
) -> Dict[str, object]:
    expected = authorization.get("expected")
    constraints = authorization.get("execution_constraints")
    if not isinstance(expected, Mapping) or not isinstance(
        constraints,
        Mapping,
    ):
        raise ValueError("CCR fit authorization contract is incomplete")
    observed = {
        "authorization_sha256": authorization_sha256,
        "lock_sha256": lock_sha256,
        "evidence_sha256": evidence_sha256,
        "failure_evidence_sha256": failure_evidence_sha256,
        "engine_sha256": sha256_file(ENGINE_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "runner_test_sha256": sha256_file(RUNNER_TEST_PATH),
        "launcher_sha256": sha256_file(LAUNCHER_PATH),
        "implementation_note_sha256": sha256_file(
            IMPLEMENTATION_NOTE_PATH
        ),
        "cache_artifact_set_manifest_sha256": sha256_file(
            Path(cache_dir) / ARTIFACT_SET_MANIFEST_NAME
        ),
        "cache_dir": str(Path(cache_dir).resolve()),
        "output_dir": str(Path(output).resolve()),
    }
    checks = {
        "schema_version": authorization.get("schema_version") == 1,
        "protocol_id": authorization.get("protocol_id")
        == "trkh_cross_colour_ratio_surface_a0_20260725",
        "state": authorization.get("state")
        == "fit_recovery_authorized_train_only_no_validation_test",
        "recovery_failure_evidence": authorization.get(
            "recovery_of_failure_sha256"
        )
        == failure_evidence_sha256,
        "supersedes_consumed_authorization": authorization.get(
            "supersedes_consumed_authorization_sha256"
        )
        == CONSUMED_AUTHORIZATION_SHA256,
        "failure_state": failure_evidence.get("state")
        == "formal_attempt_1_harness_failure_no_candidate_metric",
        "failure_authorization_consumed": failure_evidence.get(
            "authorization_consumed"
        )
        is True,
        "failure_authorization_reuse_forbidden": failure_evidence.get(
            "authorization_reuse_allowed"
        )
        is False,
        "failure_authorization_sha256": failure_evidence.get(
            "authorization_sha256"
        )
        == CONSUMED_AUTHORIZATION_SHA256,
        "lock_sha256": expected.get("lock_sha256") == lock_sha256,
        "evidence_sha256": expected.get("evidence_sha256")
        == evidence_sha256,
        "engine_sha256": expected.get("engine_sha256")
        == observed["engine_sha256"],
        "runner_sha256": expected.get("runner_sha256")
        == observed["runner_sha256"],
        "runner_test_sha256": expected.get("runner_test_sha256")
        == observed["runner_test_sha256"],
        "launcher_sha256": expected.get("launcher_sha256")
        == observed["launcher_sha256"],
        "implementation_note_sha256": expected.get(
            "implementation_note_sha256"
        )
        == observed["implementation_note_sha256"],
        "cache_artifact_set_manifest_sha256": expected.get(
            "cache_artifact_set_manifest_sha256"
        )
        == observed["cache_artifact_set_manifest_sha256"],
        "implementation_commit_present": isinstance(
            expected.get("implementation_commit"),
            str,
        )
        and len(str(expected.get("implementation_commit"))) == 40,
        "upstream_commit_matches_implementation": expected.get(
            "upstream_commit"
        )
        == expected.get("implementation_commit"),
        "cache_dir": Path(
            str(authorization.get("materialized_cache_dir"))
        ).resolve()
        == Path(cache_dir).resolve(),
        "output_dir": Path(
            str(authorization.get("output_dir"))
        ).resolve()
        == Path(output).resolve(),
        "formal_runs": constraints.get("formal_runs") == 1,
        "fresh_process_replays": constraints.get(
            "fresh_process_replays"
        )
        == 1,
        "descriptor_creation": constraints.get("descriptor_creation")
        is True,
        "head_fit": constraints.get("head_fit") is True,
        "scratch_only": constraints.get("scratch_only") is True,
        "conditions": constraints.get("conditions")
        == ["clean"],
        "epochs": constraints.get("epochs") == EPOCHS,
        "folds": constraints.get("folds") == FOLDS,
        "roles": constraints.get("roles") == list(ROLE_NAMES),
        "requested_workers": constraints.get("requested_workers")
        == REQUESTED_WORKERS,
        "raw_dataset_read": constraints.get("raw_dataset_read") is False,
        "raw_dataset_write": constraints.get("raw_dataset_write") is False,
        "keeper_forward": constraints.get("keeper_forward") is False,
        "validation": constraints.get("validation") is False,
        "test": constraints.get("test") is False,
        "production_integration": constraints.get(
            "production_integration"
        )
        is False,
        "full_train": constraints.get("full_train") is False,
        "current_best_command_update": constraints.get(
            "current_best_command_update"
        )
        is False,
    }
    if not all(checks.values()):
        raise ValueError(f"CCR fit authorization differs: {checks}")
    return {
        "observed": observed,
        "checks": checks,
        "passed": True,
    }


@dataclass(frozen=True)
class CohortArrays:
    sample_indices: np.ndarray
    targets: np.ndarray
    folds: np.ndarray
    source_stems: np.ndarray
    keeper_probabilities: np.ndarray
    model_boxes: np.ndarray
    crop_boxes: np.ndarray
    image_paths: np.ndarray
    label_paths: np.ndarray


def _load_cohort_arrays(
    path: Path,
    lock: Mapping[str, object],
) -> CohortArrays:
    arrays = _npz_arrays(path)
    expected = {
        "sample_indices": ((ROWS,), np.dtype(np.int64)),
        "targets": ((ROWS,), np.dtype(np.int64)),
        "folds": ((ROWS,), np.dtype(np.int64)),
        "source_stems": ((ROWS,), None),
        "keeper_probabilities": ((ROWS, 5), np.dtype(np.float32)),
        "model_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "crop_boxes": ((ROWS, 4), np.dtype(np.float32)),
        "image_paths": ((ROWS,), None),
        "label_paths": ((ROWS,), None),
    }
    if set(arrays) != set(expected):
        raise ValueError("CCR fit cohort-array keys differ")
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"CCR fit cohort shape differs: {name}")
        if dtype is not None and arrays[name].dtype != dtype:
            raise ValueError(f"CCR fit cohort dtype differs: {name}")
        if arrays[name].dtype.hasobject:
            raise ValueError(f"CCR fit cohort object dtype: {name}")
    cohort = lock["cohort"]
    hash_pairs = {
        "sample_indices": "sample_indices_sha256",
        "targets": "targets_sha256",
        "folds": "folds_sha256",
        "keeper_probabilities": "probabilities_sha256",
        "model_boxes": "model_boxes_sha256",
        "crop_boxes": "crop_boxes_sha256",
    }
    for name, lock_name in hash_pairs.items():
        if array_sha256(arrays[name]) != cohort[lock_name]:
            raise ValueError(f"CCR fit cohort hash differs: {name}")
    sources = arrays["source_stems"].astype(str)
    if strings_sha256(sources.tolist()) != cohort["sources_sha256"]:
        raise ValueError("CCR fit source hash differs")
    if (
        len(set(arrays["sample_indices"].tolist())) != ROWS
        or not np.isin(arrays["targets"], CLASS_ORDER).all()
        or sorted(np.unique(arrays["folds"]).tolist())
        != list(range(FOLDS))
    ):
        raise ValueError("CCR fit cohort identities differ")
    return CohortArrays(
        sample_indices=arrays["sample_indices"],
        targets=arrays["targets"],
        folds=arrays["folds"],
        source_stems=arrays["source_stems"].astype(str),
        keeper_probabilities=arrays["keeper_probabilities"],
        model_boxes=arrays["model_boxes"],
        crop_boxes=arrays["crop_boxes"],
        image_paths=arrays["image_paths"].astype(str),
        label_paths=arrays["label_paths"].astype(str),
    )


def _fold_indices(
    fold: int,
    cohort: CohortArrays,
    lock: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    held = np.flatnonzero(cohort.folds == int(fold)).astype(np.int64)
    calibration_fold = (int(fold) + 1) % FOLDS
    calibration = np.flatnonzero(
        cohort.folds == calibration_fold
    ).astype(np.int64)
    fit = np.flatnonzero(
        (cohort.folds != int(fold))
        & (cohort.folds != calibration_fold)
    ).astype(np.int64)
    payload = lock["folds"][int(fold)]
    partitions = {
        "fit": fit,
        "calibration": calibration,
        "held": held,
    }
    for name, indices in partitions.items():
        if array_sha256(indices) != payload[name]["indices_sha256"]:
            raise ValueError(
                f"CCR fit fold partition differs: {fold}/{name}"
            )
    source_sets = {
        name: set(cohort.source_stems[indices].tolist())
        for name, indices in partitions.items()
    }
    if (
        source_sets["fit"].intersection(source_sets["calibration"])
        or source_sets["fit"].intersection(source_sets["held"])
        or source_sets["calibration"].intersection(source_sets["held"])
    ):
        raise ValueError(f"CCR fit source leakage in fold {fold}")
    return partitions


def _verify_epoch_orders(
    *,
    role: str,
    fold: int,
    fit_indices: np.ndarray,
    lock: Mapping[str, object],
) -> List[np.ndarray]:
    seed = role_seed(role, fold)
    orders = build_epoch_orders(fit_indices, seed=seed)
    payload_name = (
        "repeat_orders"
        if role == "cross_colour_ratio_seed_repeat"
        else "primary_orders"
    )
    payload = lock["folds"][fold][payload_name]
    hashes = [array_sha256(order) for order in orders]
    if (
        hashes != payload["per_epoch_sha256"]
        or array_sha256(np.concatenate(orders))
        != payload["all_epochs_sha256"]
    ):
        raise ValueError(f"CCR fit epoch order differs: {fold}/{role}")
    return orders


class MutableOrderSampler(Sampler[int]):
    def __init__(self, order: Sequence[int]) -> None:
        self.set_order(order)

    def set_order(self, order: Sequence[int]) -> None:
        values = np.asarray(order, dtype=np.int64).reshape(-1)
        if values.size == 0 or np.unique(values).size != values.size:
            raise ValueError("CCR fit sampler order must be unique")
        self._order = values.tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)


class SharedDescriptorDataset(Dataset):
    def __init__(
        self,
        descriptors: Tensor,
        reliability: Tensor,
        head_targets: Tensor,
        offsets: Tensor,
    ) -> None:
        rows = int(descriptors.size(0))
        if (
            descriptors.shape != (rows, 6, VIEW_SIZE, VIEW_SIZE)
            or reliability.shape != (rows, 1, VIEW_SIZE, VIEW_SIZE)
            or head_targets.shape != (rows,)
            or offsets.shape != (rows, 6, 2)
        ):
            raise ValueError("CCR shared descriptor dataset shapes differ")
        self.descriptors = descriptors
        self.reliability = reliability
        self.head_targets = head_targets
        self.offsets = offsets

    def __len__(self) -> int:
        return int(self.descriptors.size(0))

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int, int]:
        position = int(index)
        return (
            self.descriptors[position],
            self.reliability[position],
            self.head_targets[position],
            self.offsets[position],
            position,
            os.getpid(),
        )


def _shutdown_loader(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def _set_determinism(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _current_rss() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _directory_bytes(path: Path) -> int:
    if not Path(path).exists():
        return 0
    return int(
        sum(
            item.stat().st_size
            for item in Path(path).rglob("*")
            if item.is_file()
        )
    )


def _condition_srgb(
    srgb_uint8: np.ndarray,
    *,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    values = np.asarray(srgb_uint8, dtype=np.uint8)
    output = np.empty_like(values)
    for index in range(values.shape[0]):
        image = Image.fromarray(
            np.transpose(values[index], (1, 2, 0)),
        )
        image = ImageEnhance.Brightness(image).enhance(
            float(brightness)
        )
        image = ImageEnhance.Contrast(image).enhance(float(contrast))
        output[index] = np.transpose(
            np.asarray(image, dtype=np.uint8),
            (2, 0, 1),
        )
    return output


def _close_numpy_memmap(value: object, *, flush: bool = False) -> None:
    if not isinstance(value, np.memmap):
        return
    if flush:
        value.flush()
    handle = getattr(value, "_mmap", None)
    if handle is not None and not handle.closed:
        handle.close()


def _extract_descriptor_cache(
    *,
    srgb_path: Path,
    packed_mask_path: Path,
    output_path: Path,
    reliability_path: Path,
    mode: str,
    device: torch.device,
    condition: tuple[str, float, float] = CONDITIONS[0],
    batch_size: int = DESCRIPTOR_BATCH_SIZE,
) -> Dict[str, object]:
    name, brightness, contrast = condition
    srgb = np.load(srgb_path, mmap_mode="r", allow_pickle=False)
    packed = np.load(
        packed_mask_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if srgb.shape != (ROWS, 3, 256, 256) or srgb.dtype != np.uint8:
        raise ValueError("CCR fit sRGB cache contract differs")
    if packed.shape != (ROWS, 8192) or packed.dtype != np.uint8:
        raise ValueError("CCR fit valid-mask cache contract differs")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(ROWS, 6, VIEW_SIZE, VIEW_SIZE),
    )
    reliability_exists = reliability_path.exists()
    if reliability_exists:
        reliability_output = np.load(
            reliability_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    else:
        reliability_output = np.lib.format.open_memmap(
            reliability_path,
            mode="w+",
            dtype=np.uint8,
            shape=(ROWS, 1, VIEW_SIZE, VIEW_SIZE),
        )
    extractor: Optional[ColourDerivativeDescriptorExtractor] = None
    try:
        extractor = ColourDerivativeDescriptorExtractor(mode=mode).to(
            device=device,
            dtype=torch.float32,
        )
        extractor.eval()
        clip_fractions: List[np.ndarray] = []
        reliable_counts: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, ROWS, int(batch_size)):
                stop = min(ROWS, start + int(batch_size))
                srgb_batch = np.asarray(srgb[start:stop]).copy()
                if name != "clean":
                    srgb_batch = _condition_srgb(
                        srgb_batch,
                        brightness=brightness,
                        contrast=contrast,
                    )
                mask_batch = unpack_valid_masks(
                    np.asarray(packed[start:stop])
                )
                srgb_tensor = torch.from_numpy(srgb_batch).to(
                    device=device,
                    dtype=torch.float32,
                )
                srgb_tensor = srgb_tensor / 255.0
                mean = extractor.input_mean
                std = extractor.input_std
                model_input = (srgb_tensor - mean) / std
                mask_tensor = torch.from_numpy(mask_batch[:, None]).to(
                    device=device,
                    dtype=torch.bool,
                )
                descriptors, reliability, clip_fraction = extractor(
                    model_input,
                    mask_tensor,
                )
                descriptors_np = (
                    descriptors.detach().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )
                reliability_np = (
                    reliability.detach().cpu().numpy() > 0.5
                ).astype(np.uint8)
                if not np.isfinite(descriptors_np).all():
                    raise RuntimeError(
                        "CCR descriptor contains non-finite values"
                    )
                support = reliability_np.sum(axis=(1, 2, 3))
                if bool((support == 0).any()):
                    raise RuntimeError(
                        "CCR descriptor has empty reliable support"
                    )
                output[start:stop] = descriptors_np
                if reliability_exists:
                    if not np.array_equal(
                        np.asarray(reliability_output[start:stop]),
                        reliability_np,
                    ):
                        raise RuntimeError(
                            "CCR/Colour Ratio reliability masks differ"
                        )
                else:
                    reliability_output[start:stop] = reliability_np
                clip_fractions.append(
                    clip_fraction.detach().cpu().numpy().astype(np.float64)
                )
                reliable_counts.append(support.astype(np.int64))
        output.flush()
        if hasattr(reliability_output, "flush"):
            reliability_output.flush()
        descriptor_hash = array_sha256(np.asarray(output))
        reliability_hash = array_sha256(
            np.asarray(reliability_output)
        )
        clips = np.concatenate(clip_fractions)
        supports = np.concatenate(reliable_counts)
        result = {
            "mode": mode,
            "condition": name,
            "shape": [ROWS, 6, VIEW_SIZE, VIEW_SIZE],
            "dtype": "float32",
            "descriptor_sha256": descriptor_hash,
            "reliability_sha256": reliability_hash,
            "clip_fraction_max": float(clips.max()),
            "clip_fraction_mean": float(clips.mean()),
            "reliable_pixels_min": int(supports.min()),
            "reliable_pixels_mean": float(supports.mean()),
            "descriptor_bytes": output_path.stat().st_size,
            "reliability_bytes": reliability_path.stat().st_size,
        }
    finally:
        _close_numpy_memmap(output, flush=True)
        _close_numpy_memmap(
            reliability_output,
            flush=not reliability_exists,
        )
        _close_numpy_memmap(srgb)
        _close_numpy_memmap(packed)
        del extractor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _load_shared_descriptor_tensors(
    *,
    descriptor_path: Path,
    reliability_path: Path,
    cohort: CohortArrays,
    offsets: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    descriptor_mmap = np.load(
        descriptor_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    reliability_mmap = np.load(
        reliability_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    descriptors = torch.from_numpy(
        np.asarray(descriptor_mmap, dtype=np.float32).copy()
    )
    reliability = torch.from_numpy(
        np.asarray(reliability_mmap, dtype=np.uint8).copy()
    ).to(dtype=torch.float32)
    head_targets = map_targets_to_head_indices(cohort.targets)
    offset_tensor = torch.from_numpy(
        np.asarray(offsets, dtype=np.int64).copy()
    )
    del descriptor_mmap
    del reliability_mmap
    descriptors.share_memory_()
    reliability.share_memory_()
    head_targets.share_memory_()
    offset_tensor.share_memory_()
    return descriptors, reliability, head_targets, offset_tensor


def _train_head(
    *,
    role: str,
    fold: int,
    descriptors: Tensor,
    reliability: Tensor,
    head_targets: Tensor,
    offsets: Tensor,
    orders: Sequence[np.ndarray],
    device: torch.device,
    num_workers: int = REQUESTED_WORKERS,
    require_effective_workers: Optional[int] = REQUESTED_WORKERS,
) -> tuple[
    CrossColourRatioEvidenceHead,
    Dict[str, np.ndarray],
    Dict[str, object],
]:
    if len(orders) != EPOCHS:
        raise ValueError("CCR fit requires exactly 20 epoch orders")
    _set_determinism(role_seed(role, fold))
    model = initialize_role_head(
        role,
        fold=fold,
        device=device,
    )
    contract = parameter_contract(model)
    if contract["trainable_parameter_count"] != 3004:
        raise RuntimeError("CCR head parameter count differs")
    initial_state = model_state_arrays(model)
    initial_parameters = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    dataset = SharedDescriptorDataset(
        descriptors,
        reliability,
        head_targets,
        offsets,
    )
    sampler = MutableOrderSampler(orders[0])
    loader_kwargs: Dict[str, object] = {
        "batch_size": BATCH_SIZE,
        "sampler": sampler,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(num_workers) > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    loader = DataLoader(dataset, **loader_kwargs)
    use_autocast = device.type == "cuda"
    first_gradient: Optional[Dict[str, object]] = None
    epoch_records: List[Dict[str, object]] = []
    worker_pids: set[int] = set()
    update_count = 0
    try:
        model.train()
        for epoch_index, order in enumerate(orders):
            sampler.set_order(order)
            epoch_lr = learning_rate(epoch_index)
            for group in optimizer.param_groups:
                group["lr"] = float(epoch_lr)
            losses: List[float] = []
            order_seen: List[int] = []
            for batch in loader:
                (
                    batch_descriptors,
                    batch_reliability,
                    batch_targets,
                    batch_offsets,
                    positions,
                    process_ids,
                ) = batch
                worker_pids.update(
                    int(value)
                    for value in process_ids.reshape(-1).tolist()
                )
                order_seen.extend(
                    int(value)
                    for value in positions.reshape(-1).tolist()
                )
                batch_descriptors = batch_descriptors.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_reliability = batch_reliability.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_targets = batch_targets.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                if role == "cross_colour_ratio_spatial_dephased_control":
                    batch_descriptors = spatially_dephase(
                        batch_descriptors,
                        batch_offsets,
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    logits, _, _, _ = model(
                        batch_descriptors,
                        batch_reliability,
                    )
                    loss = F.cross_entropy(logits, batch_targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("CCR fit loss is non-finite")
                loss.backward()
                if update_count == 0:
                    gradient_rows: Dict[str, object] = {}
                    for name, parameter in model.named_parameters():
                        gradient = parameter.grad
                        gradient_rows[name] = {
                            "present": gradient is not None,
                            "finite": bool(
                                gradient is not None
                                and torch.isfinite(gradient).all()
                            ),
                            "absolute_sum": (
                                float(gradient.detach().abs().sum())
                                if gradient is not None
                                else None
                            ),
                        }
                    first_gradient = {
                        "parameters": gradient_rows,
                        "all_present": all(
                            bool(row["present"])
                            for row in gradient_rows.values()
                        ),
                        "all_finite": all(
                            bool(row["finite"])
                            for row in gradient_rows.values()
                        ),
                    }
                    if not (
                        first_gradient["all_present"]
                        and first_gradient["all_finite"]
                    ):
                        raise RuntimeError(
                            "CCR first-update gradient contract failed"
                        )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("CCR gradient norm is non-finite")
                optimizer.step()
                update_count += 1
                losses.append(float(loss.detach().cpu()))
            observed_order = np.asarray(order_seen, dtype=np.int64)
            if not np.array_equal(observed_order, order):
                raise RuntimeError(
                    f"CCR loader order differs: {fold}/{role}/{epoch_index}"
                )
            epoch_records.append(
                {
                    "epoch": epoch_index,
                    "learning_rate": float(epoch_lr),
                    "order_sha256": array_sha256(observed_order),
                    "updates": int(len(losses)),
                    "loss_mean": float(np.mean(losses)),
                    "loss_last": float(losses[-1]),
                }
            )
    finally:
        _shutdown_loader(loader)
        del loader
        gc.collect()
    if first_gradient is None:
        raise RuntimeError("CCR fit made no optimizer update")
    expected_updates = int(
        sum(
            math.ceil(len(order) / BATCH_SIZE)
            for order in orders
        )
    )
    changed = {
        name: not torch.equal(
            initial_parameters[name],
            parameter.detach().cpu(),
        )
        for name, parameter in model.named_parameters()
    }
    optimizer_fp32 = all(
        not torch.is_tensor(value)
        or not value.dtype.is_floating_point
        or value.dtype == torch.float32
        for state in optimizer.state.values()
        for value in state.values()
    )
    if update_count != expected_updates or not all(changed.values()):
        raise RuntimeError(
            f"CCR fit update/state contract failed: "
            f"{update_count}/{expected_updates}, {changed}"
        )
    if not optimizer_fp32:
        raise RuntimeError("CCR optimizer state is not FP32")
    if require_effective_workers is not None and (
        len(worker_pids) != int(require_effective_workers)
    ):
        raise RuntimeError(
            "CCR effective worker count differs: "
            f"{sorted(worker_pids)}"
        )
    final_state = model_state_arrays(model)
    record = {
        "role": role,
        "fold": int(fold),
        "seed": int(role_seed(role, fold)),
        "initial_state_sha256": state_arrays_sha256(initial_state),
        "final_state_sha256": state_arrays_sha256(final_state),
        "parameter_contract": contract,
        "first_gradient": first_gradient,
        "all_parameters_changed": bool(all(changed.values())),
        "parameter_changed": changed,
        "optimizer_state_fp32": optimizer_fp32,
        "requested_workers": int(num_workers),
        "effective_worker_count": len(worker_pids),
        "effective_worker_pids": sorted(worker_pids),
        "updates": update_count,
        "expected_updates": expected_updates,
        "epochs": epoch_records,
    }
    return model, final_state, record


def _score_head(
    *,
    model: CrossColourRatioEvidenceHead,
    descriptors: Tensor,
    reliability: Tensor,
    offsets: Tensor,
    indices: np.ndarray,
    device: torch.device,
    dephase: bool,
    retain_maps: bool,
) -> Dict[str, np.ndarray]:
    positions = np.asarray(indices, dtype=np.int64).reshape(-1)
    logits_rows: List[np.ndarray] = []
    pair_rows: List[np.ndarray] = []
    reliability_rows: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, positions.size, SCORE_BATCH_SIZE):
            selected = positions[start : start + SCORE_BATCH_SIZE]
            selected_tensor = torch.from_numpy(selected)
            batch_descriptors = descriptors[selected_tensor].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            batch_reliability = reliability[selected_tensor].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if dephase:
                batch_descriptors = spatially_dephase(
                    batch_descriptors,
                    offsets[selected_tensor],
                )
            logits, _, pair_maps, evidence_reliability = model(
                batch_descriptors,
                batch_reliability,
            )
            logits_rows.append(
                logits.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )
            if retain_maps:
                pair_rows.append(
                    pair_maps.detach().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )
                reliability_rows.append(
                    evidence_reliability.detach().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                )
    output = {
        "logits": np.concatenate(logits_rows),
    }
    output["scores"] = np.asarray(
        class1_evidence_score(output["logits"]),
        dtype=np.float64,
    )
    if retain_maps:
        output["pair_maps"] = np.concatenate(pair_rows)
        output["evidence_reliability"] = np.concatenate(
            reliability_rows
        )
    return output


def _empty_role_output() -> Dict[str, object]:
    return {
        "logits": np.full((ROWS, 4), np.nan, dtype=np.float32),
        "scores": np.full((ROWS,), np.nan, dtype=np.float64),
        "pair_maps": np.full(
            (ROWS, 3, VIEW_SIZE // 4, VIEW_SIZE // 4),
            np.nan,
            dtype=np.float32,
        ),
        "evidence_reliability": np.full(
            (ROWS, 1, VIEW_SIZE // 4, VIEW_SIZE // 4),
            np.nan,
            dtype=np.float32,
        ),
        "thresholds": np.full((FOLDS,), np.nan, dtype=np.float64),
        "suppressed": np.zeros((ROWS,), dtype=np.bool_),
        "predictions": np.full((ROWS,), -1, dtype=np.int64),
    }


def _place_fold_output(
    *,
    output: Dict[str, object],
    fold: int,
    held: np.ndarray,
    held_result: Mapping[str, np.ndarray],
    calibration_result: Mapping[str, np.ndarray],
    cohort: CohortArrays,
) -> Dict[str, object]:
    calibration = np.flatnonzero(
        cohort.folds == ((int(fold) + 1) % FOLDS)
    ).astype(np.int64)
    threshold = calibrate_class1_retention_threshold(
        calibration_result["scores"],
        cohort.targets[calibration] == FOCUS_CLASS,
        minimum_retention=MINIMUM_CLASS1_RETENTION,
    )
    action = apply_keeper_suppression(
        cohort.keeper_probabilities[held],
        held_result["scores"],
        float(threshold["threshold"]),
    )
    output["logits"][held] = held_result["logits"]
    output["scores"][held] = held_result["scores"]
    output["pair_maps"][held] = held_result["pair_maps"]
    output["evidence_reliability"][held] = held_result[
        "evidence_reliability"
    ]
    output["thresholds"][fold] = threshold["threshold"]
    output["suppressed"][held] = action["suppressed"]
    output["predictions"][held] = action["predictions"]
    return threshold


def _assert_role_output_complete(
    role: str,
    output: Mapping[str, object],
) -> None:
    numeric_names = (
        "logits",
        "scores",
        "pair_maps",
        "evidence_reliability",
        "thresholds",
    )
    for name in numeric_names:
        if not np.isfinite(np.asarray(output[name])).all():
            raise RuntimeError(f"CCR role output incomplete: {role}/{name}")
    if bool((np.asarray(output["predictions"]) < 0).any()):
        raise RuntimeError(f"CCR role predictions incomplete: {role}")


def _binary_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    labels = (np.asarray(targets) == FOCUS_CLASS).astype(np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if (
        labels.shape != values.shape
        or not np.isfinite(values).all()
        or np.unique(labels).size != 2
    ):
        raise ValueError("CCR binary metric inputs differ")
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
    }


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, object]:
    target_values = np.asarray(targets, dtype=np.int64)
    prediction_values = np.asarray(predictions, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target_values,
        prediction_values,
        labels=list(range(5)),
        zero_division=0,
    )
    return {
        "accuracy": float((target_values == prediction_values).mean()),
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
    }


def _load_full_train_clean(path: Path) -> Dict[str, np.ndarray]:
    rows: Dict[int, Mapping[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "clean":
                continue
            sample_index = int(row["sample_index"])
            if sample_index in rows:
                raise ValueError(
                    f"Duplicate CIDT clean row: {sample_index}"
                )
            rows[sample_index] = row
    if len(rows) != FULL_TRAIN_ROWS or sorted(rows) != list(
        range(FULL_TRAIN_ROWS)
    ):
        raise ValueError("CIDT clean rows differ from 0..9214")
    return {
        "targets": np.asarray(
            [
                int(rows[index]["target_index"])
                for index in range(FULL_TRAIN_ROWS)
            ],
            dtype=np.int64,
        ),
        "keeper_predictions": np.asarray(
            [
                int(rows[index]["keeper_prediction"])
                for index in range(FULL_TRAIN_ROWS)
            ],
            dtype=np.int64,
        ),
        "folds": np.asarray(
            [
                int(rows[index]["fold"])
                for index in range(FULL_TRAIN_ROWS)
            ],
            dtype=np.int64,
        ),
        "source_stems": np.asarray(
            [
                rows[index]["source_stem"]
                for index in range(FULL_TRAIN_ROWS)
            ]
        ),
    }


def _transition_counts(
    before: np.ndarray,
    after: np.ndarray,
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for first, second in zip(
        np.asarray(before).tolist(),
        np.asarray(after).tolist(),
    ):
        key = f"{int(first)}->{int(second)}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _role_metrics(
    *,
    role: str,
    output: Mapping[str, object],
    cohort: CohortArrays,
    full_train: Mapping[str, np.ndarray],
    baseline_full_metrics: Mapping[str, object],
) -> Dict[str, object]:
    targets = cohort.targets
    scores = np.asarray(output["scores"], dtype=np.float64)
    logits = np.asarray(output["logits"], dtype=np.float64)
    thresholds = np.asarray(output["thresholds"], dtype=np.float64)
    suppressed = np.asarray(output["suppressed"], dtype=np.bool_)
    predictions = np.asarray(output["predictions"], dtype=np.int64)
    keeper_predictions = cohort.keeper_probabilities.argmax(axis=1)
    row_thresholds = thresholds[cohort.folds]
    positive_support = scores >= row_thresholds
    target1 = targets == FOCUS_CLASS
    keeper_tp = target1 & (keeper_predictions == FOCUS_CLASS)
    keeper_fn = target1 & (keeper_predictions != FOCUS_CLASS)
    restricted = np.isin(targets, RESTRICTED_RIVALS) & (
        keeper_predictions == FOCUS_CLASS
    )
    if int(restricted.sum()) != 222:
        raise ValueError("CCR restricted-FP cohort differs")
    aggregate = _binary_metrics(targets, scores)
    fold_metrics = {
        str(fold): _binary_metrics(
            targets[cohort.folds == fold],
            scores[cohort.folds == fold],
        )
        for fold in range(FOLDS)
    }
    head_index = {target: index for index, target in enumerate(CLASS_ORDER)}
    pair_metrics: Dict[str, object] = {}
    for rival in RESTRICTED_RIVALS:
        selected = (targets == FOCUS_CLASS) | (targets == rival)
        pair_score = (
            logits[:, head_index[FOCUS_CLASS]]
            - logits[:, head_index[rival]]
        )
        pair_metrics[f"1_vs_{rival}"] = {
            **_binary_metrics(targets[selected], pair_score[selected]),
            "rows": int(selected.sum()),
            "positive_rows": int((targets[selected] == FOCUS_CLASS).sum()),
            "rival_rows": int((targets[selected] == rival).sum()),
            "margin_mean_class1": float(pair_score[target1].mean()),
            "margin_mean_rival": float(
                pair_score[targets == rival].mean()
            ),
        }
    target_rejection = {
        str(rival): {
            "rows": int((targets == rival).sum()),
            "rejected": int(((targets == rival) & suppressed).sum()),
            "rate": float(
                suppressed[targets == rival].mean()
            ),
        }
        for rival in RESTRICTED_RIVALS
    }
    full_predictions = np.asarray(
        full_train["keeper_predictions"],
        dtype=np.int64,
    ).copy()
    full_predictions[cohort.sample_indices] = predictions
    full_metrics = _classification_metrics(
        full_train["targets"],
        full_predictions,
    )
    baseline_per_class = baseline_full_metrics["per_class"]
    full_gains = {
        "macro_f1": float(
            full_metrics["macro_f1"]
            - baseline_full_metrics["macro_f1"]
        ),
        "class1_precision": float(
            full_metrics["per_class"]["1"]["precision"]
            - baseline_per_class["1"]["precision"]
        ),
        "class1_recall": float(
            full_metrics["per_class"]["1"]["recall"]
            - baseline_per_class["1"]["recall"]
        ),
        "class1_f1": float(
            full_metrics["per_class"]["1"]["f1"]
            - baseline_per_class["1"]["f1"]
        ),
        "per_class_f1": {
            str(index): float(
                full_metrics["per_class"][str(index)]["f1"]
                - baseline_per_class[str(index)]["f1"]
            )
            for index in range(5)
        },
    }
    corrections = (
        (keeper_predictions != targets) & (predictions == targets)
    )
    harms = (
        (keeper_predictions == targets) & (predictions != targets)
    )
    return {
        "role": role,
        "aggregate": aggregate,
        "folds": fold_metrics,
        "pairs": pair_metrics,
        "thresholds": thresholds.tolist(),
        "action": {
            "all_class1_rows": int(target1.sum()),
            "all_class1_supported": int(
                (target1 & positive_support).sum()
            ),
            "all_class1_retention": float(
                positive_support[target1].mean()
            ),
            "keeper_tp_rows": int(keeper_tp.sum()),
            "keeper_tp_retained": int(
                (keeper_tp & positive_support).sum()
            ),
            "keeper_tp_retention": float(
                positive_support[keeper_tp].mean()
            ),
            "keeper_fn_rows": int(keeper_fn.sum()),
            "keeper_fn_supported": int(
                (keeper_fn & positive_support).sum()
            ),
            "restricted_fp_rows": int(restricted.sum()),
            "restricted_fp_rejected": int(
                (restricted & suppressed).sum()
            ),
            "restricted_fp_rejection": float(
                suppressed[restricted].mean()
            ),
            "target_rejection": target_rejection,
            "corrections": int(corrections.sum()),
            "tp_harms": int((keeper_tp & suppressed).sum()),
            "all_harms": int(harms.sum()),
            "changed": int(
                (keeper_predictions != predictions).sum()
            ),
            "transitions": _transition_counts(
                keeper_predictions,
                predictions,
            ),
        },
        "full_train": {
            "baseline": baseline_full_metrics,
            "updated": full_metrics,
            "gains": full_gains,
            "changed": int(
                (
                    np.asarray(full_train["keeper_predictions"])
                    != full_predictions
                ).sum()
            ),
        },
    }


def _core_promotion_checks(
    *,
    role_metrics: Mapping[str, object],
    role_output: Mapping[str, object],
    keeper_metrics: Mapping[str, float],
    colour_metrics: Mapping[str, object],
    dephased_metrics: Mapping[str, object],
    same_weight_metrics: Mapping[str, object],
) -> Dict[str, bool]:
    aggregate = role_metrics["aggregate"]
    action = role_metrics["action"]
    pairs = role_metrics["pairs"]
    full_gains = role_metrics["full_train"]["gains"]
    candidate_fold_wins = sum(
        float(role_metrics["folds"][str(fold)]["auroc"])
        > max(
            float(colour_metrics["folds"][str(fold)]["auroc"]),
            float(
                dephased_metrics["folds"][str(fold)]["auroc"]
            ),
        )
        for fold in range(FOLDS)
    )
    target_rejection = action["target_rejection"]
    nonfocus_f1 = full_gains["per_class_f1"]
    return {
        "aggregate_auroc_gte_0p86": aggregate["auroc"] >= 0.86,
        "aggregate_auprc_gte_0p92": aggregate["auprc"] >= 0.92,
        "auroc_gain_over_keeper_gte_0p02": (
            aggregate["auroc"] - keeper_metrics["auroc"]
        )
        >= 0.02,
        "auroc_gain_over_colour_ratio_gte_0p02": (
            aggregate["auroc"]
            - colour_metrics["aggregate"]["auroc"]
        )
        >= 0.02,
        "auroc_gain_over_trained_dephase_gte_0p04": (
            aggregate["auroc"]
            - dephased_metrics["aggregate"]["auroc"]
        )
        >= 0.04,
        "same_weight_dephase_drop_gte_0p03": (
            aggregate["auroc"]
            - same_weight_metrics["aggregate"]["auroc"]
        )
        >= 0.03,
        "fold_wins_gte_4_of_5": candidate_fold_wins >= 4,
        "pair_1_vs_0_auroc_gte_0p82": (
            pairs["1_vs_0"]["auroc"] >= 0.82
        ),
        "pair_1_vs_2_auroc_gte_0p80": (
            pairs["1_vs_2"]["auroc"] >= 0.80
        ),
        "pair_1_vs_4_auroc_gte_0p75": (
            pairs["1_vs_4"]["auroc"] >= 0.75
        ),
        "all_class1_retention_gte_0p95": (
            action["all_class1_retention"] >= 0.95
        ),
        "keeper_tp_retention_gte_0p95": (
            action["keeper_tp_retention"] >= 0.95
        ),
        "restricted_fp_rejection_gte_0p25": (
            action["restricted_fp_rejection"] >= 0.25
        ),
        "target0_rejection_gte_0p20": (
            target_rejection["0"]["rate"] >= 0.20
        ),
        "target2_rejection_gte_0p20": (
            target_rejection["2"]["rate"] >= 0.20
        ),
        "target4_rejected_gte_2": (
            target_rejection["4"]["rejected"] >= 2
        ),
        "keeper_fn_supported_gte_8": (
            action["keeper_fn_supported"] >= 8
        ),
        "corrections_plus_fn_support_ge_twice_tp_harms": (
            action["corrections"] + action["keeper_fn_supported"]
        )
        >= (2 * action["tp_harms"]),
        "full_macro_f1_gain_gte_0p002": (
            full_gains["macro_f1"] >= 0.002
        ),
        "full_class1_precision_gain_gte_0p03": (
            full_gains["class1_precision"] >= 0.03
        ),
        "full_class1_f1_gain_gte_0p01": (
            full_gains["class1_f1"] >= 0.01
        ),
        "full_class1_recall_drop_lte_0p03": (
            full_gains["class1_recall"] >= -0.03
        ),
        "nonfocus_f1_drop_lte_0p003": all(
            nonfocus_f1[str(index)] >= -0.003
            for index in (0, 2, 3, 4)
        ),
    }


def _build_clean_gate(
    *,
    role_metrics: Mapping[str, Mapping[str, object]],
    role_outputs: Mapping[str, Mapping[str, object]],
    keeper_metrics: Mapping[str, float],
    same_weight_metrics: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    colour = role_metrics["colour_ratio_control"]
    dephased = role_metrics[
        "cross_colour_ratio_spatial_dephased_control"
    ]
    candidate = role_metrics["cross_colour_ratio_candidate"]
    repeat = role_metrics["cross_colour_ratio_seed_repeat"]
    candidate_checks = _core_promotion_checks(
        role_metrics=candidate,
        role_output=role_outputs["cross_colour_ratio_candidate"],
        keeper_metrics=keeper_metrics,
        colour_metrics=colour,
        dephased_metrics=dephased,
        same_weight_metrics=same_weight_metrics[
            "cross_colour_ratio_candidate"
        ],
    )
    repeat_checks = _core_promotion_checks(
        role_metrics=repeat,
        role_output=role_outputs[
            "cross_colour_ratio_seed_repeat"
        ],
        keeper_metrics=keeper_metrics,
        colour_metrics=colour,
        dephased_metrics=dephased,
        same_weight_metrics=same_weight_metrics[
            "cross_colour_ratio_seed_repeat"
        ],
    )
    candidate_auroc = float(candidate["aggregate"]["auroc"])
    repeat_auroc = float(repeat["aggregate"]["auroc"])
    candidate_actions = np.asarray(
        role_outputs["cross_colour_ratio_candidate"]["suppressed"],
        dtype=np.bool_,
    )
    repeat_actions = np.asarray(
        role_outputs["cross_colour_ratio_seed_repeat"]["suppressed"],
        dtype=np.bool_,
    )
    action_agreement = float(
        (candidate_actions == repeat_actions).mean()
    )
    stability = {
        "seed_repeat_auroc_difference_lte_0p01": (
            abs(candidate_auroc - repeat_auroc) <= 0.01
        ),
        "seed_repeat_action_agreement_gte_0p95": (
            action_agreement >= 0.95
        ),
        "seed_repeat_pass_decision_identical": (
            all(candidate_checks.values())
            == all(repeat_checks.values())
        ),
    }
    checks = {
        **{
            f"candidate__{name}": value
            for name, value in candidate_checks.items()
        },
        **stability,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed_checks": [
            name for name, value in checks.items() if not value
        ],
        "candidate_core_passed": bool(
            all(candidate_checks.values())
        ),
        "repeat_core_passed": bool(all(repeat_checks.values())),
        "candidate_core_checks": candidate_checks,
        "repeat_core_checks": repeat_checks,
        "seed_repeat_auroc_difference": abs(
            candidate_auroc - repeat_auroc
        ),
        "seed_repeat_action_agreement": action_agreement,
    }


def _load_head_state(
    *,
    role: str,
    fold: int,
    state: Mapping[str, np.ndarray],
    device: torch.device,
) -> CrossColourRatioEvidenceHead:
    model = initialize_role_head(role, fold=fold, device=device)
    expected = {
        f"model.{name}": value
        for name, value in model.state_dict().items()
    }
    if set(state) != set(expected):
        raise ValueError(f"CCR state keys differ: {role}/{fold}")
    loaded = {
        name.removeprefix("model."): torch.from_numpy(
            np.asarray(state[name]).copy()
        )
        for name in sorted(state)
    }
    model.load_state_dict(loaded, strict=True)
    return model


def _place_same_weight_fold_output(
    *,
    output: Dict[str, object],
    fold: int,
    held: np.ndarray,
    held_result: Mapping[str, np.ndarray],
    threshold: float,
    cohort: CohortArrays,
) -> None:
    action = apply_keeper_suppression(
        cohort.keeper_probabilities[held],
        held_result["scores"],
        float(threshold),
    )
    output["logits"][held] = held_result["logits"]
    output["scores"][held] = held_result["scores"]
    output["pair_maps"][held] = held_result["pair_maps"]
    output["evidence_reliability"][held] = held_result[
        "evidence_reliability"
    ]
    output["thresholds"][fold] = float(threshold)
    output["suppressed"][held] = action["suppressed"]
    output["predictions"][held] = action["predictions"]


def _state_artifact_arrays(
    states: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for role in sorted(states):
        for fold in sorted(states[role]):
            for name, value in sorted(states[role][fold].items()):
                key = f"{role}.fold_{fold}.{name}"
                if key in arrays:
                    raise RuntimeError(f"Duplicate CCR state key: {key}")
                arrays[key] = np.asarray(value)
    return arrays


def _output_artifact_arrays(
    *,
    role_outputs: Mapping[str, Mapping[str, object]],
    same_weight_outputs: Mapping[str, Mapping[str, object]],
    keeper_scores: np.ndarray,
    cohort: CohortArrays,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "cohort.sample_indices": cohort.sample_indices,
        "cohort.targets": cohort.targets,
        "cohort.folds": cohort.folds,
        "keeper.scores": np.asarray(keeper_scores, dtype=np.float64),
    }
    for family, outputs in (
        ("role", role_outputs),
        ("same_weight_dephased", same_weight_outputs),
    ):
        for role in sorted(outputs):
            for name, value in sorted(outputs[role].items()):
                array = np.asarray(value)
                if array.dtype.hasobject:
                    raise ValueError(
                        f"Object output is forbidden: {family}/{role}/{name}"
                    )
                arrays[f"{family}.{role}.{name}"] = array
    return arrays


def _process_snapshot() -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    own = psutil.Process(os.getpid())
    ancestor_pids = {int(process.pid) for process in own.parents()}
    try:
        child_pids = {
            int(process.pid) for process in own.children(recursive=True)
        }
    except psutil.Error:
        child_pids = set()
    for process in psutil.process_iter(
        ["pid", "name", "create_time", "cmdline"]
    ):
        try:
            name = str(process.info.get("name") or "").casefold()
            if name not in {"python.exe", "python", "trtexec.exe", "trtexec"}:
                continue
            rows.append(
                {
                    "pid": int(process.info["pid"]),
                    "name": str(process.info.get("name") or ""),
                    "create_time": float(
                        process.info.get("create_time") or 0.0
                    ),
                    "cmdline": [
                        str(value)
                        for value in (process.info.get("cmdline") or [])
                    ],
                    "current": int(process.info["pid"]) == os.getpid(),
                    "ancestor": int(process.info["pid"]) in ancestor_pids,
                    "child": int(process.info["pid"]) in child_pids,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    gpu_rows: List[Dict[str, object]] = []
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            gpu_rows.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "used_memory_mib": (
                        int(parts[2]) if parts[2].isdigit() else None
                    ),
                }
            )
    compute_names = {"python.exe", "python", "trtexec.exe", "trtexec"}
    foreign_gpu = [
        row
        for row in gpu_rows
        if int(row["pid"]) != os.getpid()
        and Path(str(row["process_name"])).name.casefold()
        in compute_names
    ]
    compute_children = [
        row for row in rows if bool(row["child"])
    ]
    gpu_state: Dict[str, Optional[int]] = {
        "utilization_percent": None,
        "memory_used_mib": None,
        "memory_total_mib": None,
    }
    state_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if state_query.returncode == 0 and state_query.stdout.splitlines():
        values = [
            value.strip()
            for value in state_query.stdout.splitlines()[0].split(",")
        ]
        if len(values) == 3:
            gpu_state = {
                "utilization_percent": (
                    int(values[0]) if values[0].isdigit() else None
                ),
                "memory_used_mib": (
                    int(values[1]) if values[1].isdigit() else None
                ),
                "memory_total_mib": (
                    int(values[2]) if values[2].isdigit() else None
                ),
            }
    return {
        "process_id": os.getpid(),
        "python_or_trtexec": sorted(rows, key=lambda row: int(row["pid"])),
        "gpu_compute_processes": sorted(
            gpu_rows,
            key=lambda row: int(row["pid"]),
        ),
        "gpu_state": gpu_state,
        "compute_children": compute_children,
        "compute_child_process_count": len(compute_children),
        "foreign_gpu_compute_processes": foreign_gpu,
        "foreign_gpu_compute_process_count": len(foreign_gpu),
    }


class _ResourceMonitor:
    def __init__(self) -> None:
        self.peak_rss = _current_rss()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._sample,
            name="ccr-resource-monitor",
            daemon=True,
        )

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.wait(0.05):
            try:
                self.peak_rss = max(
                    self.peak_rss,
                    int(process.memory_info().rss),
                )
            except psutil.Error:
                return

    def __enter__(self) -> "_ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.peak_rss = max(self.peak_rss, _current_rss())


def _fit_and_score_role(
    *,
    role: str,
    fold: int,
    cohort: CohortArrays,
    lock: Mapping[str, object],
    descriptors: Tensor,
    reliability: Tensor,
    head_targets: Tensor,
    offsets: Tensor,
    device: torch.device,
    output: Dict[str, object],
    same_weight_output: Optional[Dict[str, object]] = None,
    num_workers: int = REQUESTED_WORKERS,
    require_effective_workers: Optional[int] = REQUESTED_WORKERS,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    partitions = _fold_indices(fold, cohort, lock)
    orders = _verify_epoch_orders(
        role=role,
        fold=fold,
        fit_indices=partitions["fit"],
        lock=lock,
    )
    model, state, record = _train_head(
        role=role,
        fold=fold,
        descriptors=descriptors,
        reliability=reliability,
        head_targets=head_targets,
        offsets=offsets,
        orders=orders,
        device=device,
        num_workers=num_workers,
        require_effective_workers=require_effective_workers,
    )
    trained_dephase = (
        role == "cross_colour_ratio_spatial_dephased_control"
    )
    calibration_result = _score_head(
        model=model,
        descriptors=descriptors,
        reliability=reliability,
        offsets=offsets,
        indices=partitions["calibration"],
        device=device,
        dephase=trained_dephase,
        retain_maps=False,
    )
    held_result = _score_head(
        model=model,
        descriptors=descriptors,
        reliability=reliability,
        offsets=offsets,
        indices=partitions["held"],
        device=device,
        dephase=trained_dephase,
        retain_maps=True,
    )
    threshold = _place_fold_output(
        output=output,
        fold=fold,
        held=partitions["held"],
        held_result=held_result,
        calibration_result=calibration_result,
        cohort=cohort,
    )
    record["calibration"] = threshold
    if same_weight_output is not None:
        same_weight_result = _score_head(
            model=model,
            descriptors=descriptors,
            reliability=reliability,
            offsets=offsets,
            indices=partitions["held"],
            device=device,
            dephase=True,
            retain_maps=True,
        )
        _place_same_weight_fold_output(
            output=same_weight_output,
            fold=fold,
            held=partitions["held"],
            held_result=same_weight_result,
            threshold=float(threshold["threshold"]),
            cohort=cohort,
        )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, record


def _run_clean_science(
    *,
    cache_dir: Path,
    temporary: Path,
    lock: Mapping[str, object],
    cohort: CohortArrays,
    full_train: Mapping[str, np.ndarray],
    device: torch.device,
    num_workers: int = REQUESTED_WORKERS,
    require_effective_workers: Optional[int] = REQUESTED_WORKERS,
) -> Dict[str, object]:
    offsets = build_dephase_offsets(cohort.sample_indices)
    if (
        offsets.shape != tuple(lock["descriptor"]["dephase_offsets_shape"])
        or array_sha256(offsets)
        != lock["descriptor"]["dephase_offsets_sha256"]
    ):
        raise ValueError("CCR dephase offsets differ from lock")
    baseline_full_metrics = _classification_metrics(
        full_train["targets"],
        full_train["keeper_predictions"],
    )
    keeper_scores = np.asarray(
        keeper_margin(cohort.keeper_probabilities),
        dtype=np.float64,
    )
    keeper_metrics = {
        **_binary_metrics(cohort.targets, keeper_scores),
        "folds": {
            str(fold): _binary_metrics(
                cohort.targets[cohort.folds == fold],
                keeper_scores[cohort.folds == fold],
            )
            for fold in range(FOLDS)
        },
    }
    role_outputs = {
        role: _empty_role_output() for role in ROLE_NAMES
    }
    same_weight_outputs = {
        role: _empty_role_output()
        for role in (
            "cross_colour_ratio_candidate",
            "cross_colour_ratio_seed_repeat",
        )
    }
    states: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        role: {} for role in ROLE_NAMES
    }
    training: Dict[str, Dict[str, object]] = {
        role: {} for role in ROLE_NAMES
    }
    descriptor_records: Dict[str, object] = {}
    source_cache_bytes = _directory_bytes(cache_dir)
    maximum_combined_temporary_bytes = source_cache_bytes
    srgb_path = cache_dir / SRGB_CACHE_NAME
    packed_mask_path = cache_dir / VALID_MASK_CACHE_NAME
    reliability_path = temporary / "reliability_uint8.npy"
    descriptor_path = temporary / "descriptor_float32.npy"

    def update_disk_peak() -> None:
        nonlocal maximum_combined_temporary_bytes
        maximum_combined_temporary_bytes = max(
            maximum_combined_temporary_bytes,
            source_cache_bytes + _directory_bytes(temporary),
        )

    try:
        descriptor_records[COLOUR_RATIO_MODE] = (
            _extract_descriptor_cache(
                srgb_path=srgb_path,
                packed_mask_path=packed_mask_path,
                output_path=descriptor_path,
                reliability_path=reliability_path,
                mode=COLOUR_RATIO_MODE,
                device=device,
            )
        )
        update_disk_peak()
        (
            descriptors,
            reliability,
            head_targets,
            offset_tensor,
        ) = _load_shared_descriptor_tensors(
            descriptor_path=descriptor_path,
            reliability_path=reliability_path,
            cohort=cohort,
            offsets=offsets,
        )
        for fold in range(FOLDS):
            state, record = _fit_and_score_role(
                role="colour_ratio_control",
                fold=fold,
                cohort=cohort,
                lock=lock,
                descriptors=descriptors,
                reliability=reliability,
                head_targets=head_targets,
                offsets=offset_tensor,
                device=device,
                output=role_outputs["colour_ratio_control"],
                num_workers=num_workers,
                require_effective_workers=require_effective_workers,
            )
            states["colour_ratio_control"][fold] = state
            training["colour_ratio_control"][str(fold)] = record
        del descriptors, reliability, head_targets, offset_tensor
        descriptor_path.unlink()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        descriptor_records[CROSS_COLOUR_RATIO_MODE] = (
            _extract_descriptor_cache(
                srgb_path=srgb_path,
                packed_mask_path=packed_mask_path,
                output_path=descriptor_path,
                reliability_path=reliability_path,
                mode=CROSS_COLOUR_RATIO_MODE,
                device=device,
            )
        )
        if (
            descriptor_records[COLOUR_RATIO_MODE][
                "reliability_sha256"
            ]
            != descriptor_records[CROSS_COLOUR_RATIO_MODE][
                "reliability_sha256"
            ]
        ):
            raise RuntimeError("CCR and CR reliability hashes differ")
        update_disk_peak()
        (
            descriptors,
            reliability,
            head_targets,
            offset_tensor,
        ) = _load_shared_descriptor_tensors(
            descriptor_path=descriptor_path,
            reliability_path=reliability_path,
            cohort=cohort,
            offsets=offsets,
        )
        for role in (
            "cross_colour_ratio_candidate",
            "cross_colour_ratio_seed_repeat",
            "cross_colour_ratio_spatial_dephased_control",
        ):
            for fold in range(FOLDS):
                state, record = _fit_and_score_role(
                    role=role,
                    fold=fold,
                    cohort=cohort,
                    lock=lock,
                    descriptors=descriptors,
                    reliability=reliability,
                    head_targets=head_targets,
                    offsets=offset_tensor,
                    device=device,
                    output=role_outputs[role],
                    same_weight_output=(
                        same_weight_outputs[role]
                        if role in same_weight_outputs
                        else None
                    ),
                    num_workers=num_workers,
                    require_effective_workers=require_effective_workers,
                )
                states[role][fold] = state
                training[role][str(fold)] = record
        del descriptors, reliability, head_targets, offset_tensor
    finally:
        for path in (descriptor_path, reliability_path):
            if path.exists():
                path.unlink()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for role, output in role_outputs.items():
        _assert_role_output_complete(role, output)
    for role, output in same_weight_outputs.items():
        _assert_role_output_complete(f"{role}/same_weight", output)
    primary_roles = (
        "colour_ratio_control",
        "cross_colour_ratio_candidate",
        "cross_colour_ratio_spatial_dephased_control",
    )
    initial_state_checks: Dict[str, bool] = {}
    for fold in range(FOLDS):
        hashes = [
            training[role][str(fold)]["initial_state_sha256"]
            for role in primary_roles
        ]
        initial_state_checks[str(fold)] = len(set(hashes)) == 1
    if not all(initial_state_checks.values()):
        raise RuntimeError(
            f"CCR primary initial states differ: {initial_state_checks}"
        )
    role_metrics = {
        role: _role_metrics(
            role=role,
            output=role_outputs[role],
            cohort=cohort,
            full_train=full_train,
            baseline_full_metrics=baseline_full_metrics,
        )
        for role in ROLE_NAMES
    }
    same_weight_metrics = {
        role: _role_metrics(
            role=f"{role}_same_weight_dephased",
            output=same_weight_outputs[role],
            cohort=cohort,
            full_train=full_train,
            baseline_full_metrics=baseline_full_metrics,
        )
        for role in same_weight_outputs
    }
    clean_gate = _build_clean_gate(
        role_metrics=role_metrics,
        role_outputs=role_outputs,
        keeper_metrics=keeper_metrics,
        same_weight_metrics=same_weight_metrics,
    )
    return {
        "descriptor_records": descriptor_records,
        "states": states,
        "training": training,
        "role_outputs": role_outputs,
        "same_weight_outputs": same_weight_outputs,
        "metrics": {
            "baseline_full_train": baseline_full_metrics,
            "keeper_margin": keeper_metrics,
            "roles": role_metrics,
            "same_weight_dephased": same_weight_metrics,
            "clean_gate": clean_gate,
        },
        "keeper_scores": keeper_scores,
        "dephase_offsets_sha256": array_sha256(offsets),
        "primary_initial_state_checks": initial_state_checks,
        "maximum_combined_temporary_bytes": (
            maximum_combined_temporary_bytes
        ),
    }


def structural_preflight(
    *,
    lock_path: Path = LOCK_PATH,
    cache_dir: Path = DEFAULT_CACHE,
    output: Path = DEFAULT_OUTPUT,
    require_clean_repository: bool = True,
) -> Dict[str, object]:
    lock, lock_sha = _load_sidecar_json(lock_path)
    evidence, evidence_sha = _load_sidecar_json(EVIDENCE_PATH)
    protocol = lock["protocol"]
    checks: Dict[str, bool] = {
        "lock_sha256": lock_sha
        == "6c2a604deb4f976a3556ffb0ad8ddb60c7f376e81e374ae16272c6e6d49fc835",
        "protocol_sha256": sha256_file(Path(str(protocol["path"])))
        == protocol["sha256"],
        "materializer_evidence_sha256": evidence_sha
        == (
            "e992c297c2416976152694dec95df6d99692dd63af6395f00"
            "df99b9efa98c3ed"
        ),
        "engine_sha256": sha256_file(ENGINE_PATH)
        == (
            "ac610d7c37eda1b96a54257f6814a3e8c7ef5bfc243ec3bb"
            "3c7ae13a86d26fe9"
        ),
        "rows": lock["cohort"]["rows"] == ROWS,
        "folds": len(lock["folds"]) == FOLDS,
        "roles": lock["optimization"]["roles"] == list(ROLE_NAMES),
        "epochs": lock["optimization"]["epochs"] == EPOCHS,
        "batch_size": lock["optimization"]["batch_size"] == BATCH_SIZE,
        "workers": lock["optimization"]["requested_workers"]
        == REQUESTED_WORKERS,
        "scratch_only": lock["head"]["pretrained"] is False,
        "parameter_oracle": lock["head"]["parameter_oracle"] == 3004,
        "validation_forbidden": lock["forbidden"]["validation"] is True,
        "test_forbidden": lock["forbidden"]["test"] is True,
        "output_absent": not Path(output).resolve().exists(),
        "temporary_sibling_absent": not any(
            Path(output).resolve().parent.glob(
                f".{Path(output).resolve().name}.formal.*"
            )
        ),
    }
    cache_record = evidence["artifacts"]
    expected_cache = (
        REPO_ROOT / str(cache_record["output_dir"])
    ).resolve()
    checks["cache_path"] = Path(cache_dir).resolve() == expected_cache
    checks["cache_manifest_sha256"] = (
        sha256_file(
            Path(cache_dir).resolve() / ARTIFACT_SET_MANIFEST_NAME
        )
        == cache_record["artifact_set_manifest"]["file_sha256"]
    )
    primary_hashes: List[str] = []
    for role in (
        "colour_ratio_control",
        "cross_colour_ratio_candidate",
        "cross_colour_ratio_spatial_dephased_control",
    ):
        model = initialize_role_head(
            role,
            fold=0,
            device=torch.device("cpu"),
        )
        primary_hashes.append(
            state_arrays_sha256(model_state_arrays(model))
        )
    repeat = initialize_role_head(
        "cross_colour_ratio_seed_repeat",
        fold=0,
        device=torch.device("cpu"),
    )
    checks["primary_initial_states_identical"] = (
        len(set(primary_hashes)) == 1
    )
    checks["repeat_initial_state_distinct"] = (
        state_arrays_sha256(model_state_arrays(repeat))
        != primary_hashes[0]
    )
    checks["parameter_contract"] = (
        parameter_contract(repeat)["trainable_parameter_count"] == 3004
    )
    synthetic = torch.linspace(
        -1.0,
        1.0,
        steps=8 * 6 * VIEW_SIZE * VIEW_SIZE,
        dtype=torch.float32,
    ).reshape(8, 6, VIEW_SIZE, VIEW_SIZE)
    reliability = torch.ones(
        (8, 1, VIEW_SIZE, VIEW_SIZE),
        dtype=torch.float32,
    )
    offsets = build_dephase_offsets(np.arange(8, dtype=np.int64))
    dephased = spatially_dephase(synthetic, offsets)
    logits, evidence_maps, pair_maps, evidence_reliability = repeat(
        dephased,
        reliability,
    )
    loss = F.cross_entropy(
        logits,
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    loss.backward()
    checks["synthetic_forward_finite"] = bool(
        torch.isfinite(logits).all()
        and torch.isfinite(evidence_maps).all()
        and torch.isfinite(pair_maps).all()
        and torch.isfinite(evidence_reliability).all()
    )
    checks["synthetic_gradient_finite"] = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in repeat.parameters()
    )
    checks["cuda_available"] = torch.cuda.is_available()
    checks["cuda_bf16_supported"] = bool(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    descriptor_bytes = (
        ROWS * 6 * VIEW_SIZE * VIEW_SIZE * np.dtype(np.float32).itemsize
        + 4096
    )
    reliability_bytes = (
        ROWS * VIEW_SIZE * VIEW_SIZE * np.dtype(np.uint8).itemsize
        + 4096
    )
    source_cache_bytes = _directory_bytes(Path(cache_dir))
    checks["predicted_temporary_bytes"] = (
        source_cache_bytes + descriptor_bytes + reliability_bytes
        <= int(lock["resource_limits"]["temporary_cache_bytes"])
    )
    checks["available_ram_gte_3gib"] = (
        int(psutil.virtual_memory().available) >= 3 * 1024**3
    )
    repository: Optional[Dict[str, object]] = None
    if require_clean_repository:
        repository = _repository_state(
            str(lock["lock_parent_commit"])
        )
        checks["repository"] = bool(repository["passed"])
    process = _process_snapshot()
    checks["foreign_gpu_compute_processes_zero"] = (
        process["foreign_gpu_compute_process_count"] == 0
    )
    gpu_utilization = process["gpu_state"]["utilization_percent"]
    checks["gpu_utilization_lte_30_percent"] = (
        gpu_utilization is not None and int(gpu_utilization) <= 30
    )
    result = {
        "protocol_id": lock["protocol_id"],
        "state": "fit_structural_preflight",
        "lock_sha256": lock_sha,
        "materializer_evidence_sha256": evidence_sha,
        "runner_sha256": sha256_file(RUNNER_PATH),
        "cache_dir": str(Path(cache_dir).resolve()),
        "output_dir": str(Path(output).resolve()),
        "repository": repository,
        "process": process,
        "resource_prediction": {
            "source_cache_bytes": source_cache_bytes,
            "descriptor_bytes_upper_bound": descriptor_bytes,
            "reliability_bytes_upper_bound": reliability_bytes,
            "combined_bytes_upper_bound": (
                source_cache_bytes
                + descriptor_bytes
                + reliability_bytes
            ),
            "limit": lock["resource_limits"][
                "temporary_cache_bytes"
            ],
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "candidate_descriptor_created": False,
        "candidate_state_created": False,
        "candidate_metric_observed": False,
        "validation_data_used": False,
        "test_data_used": False,
    }
    if not result["passed"]:
        raise RuntimeError(f"CCR fit preflight failed: {checks}")
    return result


def _formal_allowed_paths(
    *,
    cache_dir: Path,
    lock: Mapping[str, object],
) -> List[Path]:
    paths = [
        path
        for path in Path(cache_dir).iterdir()
        if path.is_file()
    ]
    paths.append(
        Path(str(lock["immutable_inputs"]["cidt_predictions"]["path"]))
    )
    return paths


def run_formal(
    *,
    output: Path,
    authorization_path: Path,
    lock_path: Path = LOCK_PATH,
    cache_dir: Path = DEFAULT_CACHE,
    authorization_output: Optional[Path] = None,
    num_workers: int = REQUESTED_WORKERS,
    require_effective_workers: Optional[int] = REQUESTED_WORKERS,
) -> Dict[str, object]:
    output = Path(output).resolve()
    authorized_output = Path(
        authorization_output if authorization_output is not None else output
    ).resolve()
    cache_dir = Path(cache_dir).resolve()
    if output.exists():
        raise FileExistsError(f"CCR fit output already exists: {output}")
    temporary = output.parent / f".{output.name}.formal.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"CCR fit temporary output exists: {temporary}")
    lock, lock_sha = _load_sidecar_json(lock_path)
    evidence, evidence_sha = _load_sidecar_json(EVIDENCE_PATH)
    failure_evidence, failure_evidence_sha = _load_sidecar_json(
        FAILURE_EVIDENCE_PATH
    )
    authorization, authorization_sha = _load_sidecar_json(
        authorization_path
    )
    authorization_check = _verify_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        lock_sha256=lock_sha,
        evidence_sha256=evidence_sha,
        failure_evidence=failure_evidence,
        failure_evidence_sha256=failure_evidence_sha,
        cache_dir=cache_dir,
        output=authorized_output,
    )
    required_ancestor = str(
        authorization["expected"]["implementation_commit"]
    )
    repository = _repository_state(required_ancestor)
    process_before = _process_snapshot()
    gpu_utilization = process_before["gpu_state"][
        "utilization_percent"
    ]
    if (
        process_before["foreign_gpu_compute_process_count"] != 0
        or gpu_utilization is None
        or int(gpu_utilization) > 30
    ):
        raise RuntimeError(
            "CCR formal process-isolation gate failed: "
            f"foreign={process_before['foreign_gpu_compute_processes']}, "
            f"gpu_utilization={gpu_utilization}"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CCR formal requires CUDA BF16 support")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    temporary.mkdir(parents=False)
    started = time.perf_counter()
    try:
        allowed_paths = _formal_allowed_paths(
            cache_dir=cache_dir,
            lock=lock,
        )
        cidt_path = Path(
            str(lock["immutable_inputs"]["cidt_predictions"]["path"])
        ).resolve()
        with _ResourceMonitor() as monitor:
            with FitDataAccessLedger(
                allowed_paths=allowed_paths,
                excluded_roots=(temporary, output),
            ) as ledger:
                cache_verification = _verify_materialized_cache(
                    cache_dir=cache_dir,
                    evidence=evidence,
                )
                cohort = _load_cohort_arrays(
                    cache_dir / COHORT_ARRAYS_NAME,
                    lock,
                )
                _verify_file_record(
                    cidt_path,
                    lock["immutable_inputs"]["cidt_predictions"],
                )
                full_train = _load_full_train_clean(cidt_path)
                science = _run_clean_science(
                    cache_dir=cache_dir,
                    temporary=temporary,
                    lock=lock,
                    cohort=cohort,
                    full_train=full_train,
                    device=device,
                    num_workers=num_workers,
                    require_effective_workers=require_effective_workers,
                )
                access = ledger.snapshot()
        process_after = _process_snapshot()
        elapsed = time.perf_counter() - started
        peak_cuda = int(torch.cuda.max_memory_allocated(device))
        resource_checks = {
            "clean_wall_seconds": elapsed
            <= float(lock["resource_limits"]["clean_wall_seconds"]),
            "head_peak_cuda_bytes": peak_cuda
            <= int(lock["resource_limits"]["head_peak_cuda_bytes"]),
            "process_rss_bytes": monitor.peak_rss
            <= int(lock["resource_limits"]["process_rss_bytes"]),
            "temporary_cache_bytes": int(
                science["maximum_combined_temporary_bytes"]
            )
            <= int(lock["resource_limits"]["temporary_cache_bytes"]),
        }
        access_checks = {
            "ledger_passed": bool(access["passed"]),
            "blocked_zero": int(access["blocked_attempt_count"]) == 0,
            "validation_zero": int(access["validation_open_count"]) == 0,
            "test_zero": int(access["test_open_count"]) == 0,
            "write_like_zero": all(
                row["decision"] == "allowed"
                for row in access["events"]
            ),
            "cidt_clean_read": int(
                access["logical_path_counts"].get(
                    _normalized_path(cidt_path),
                    0,
                )
            )
            >= 1,
            "raw_dataset_not_read": all(
                row["domain"] != "dataset"
                for row in access["events"]
            ),
        }
        automatic_checks = {
            **resource_checks,
            **access_checks,
            "cache_verified": bool(cache_verification["passed"]),
            "dephase_hash_exact": science["dephase_offsets_sha256"]
            == lock["descriptor"]["dephase_offsets_sha256"],
            "primary_initial_states_identical": all(
                science["primary_initial_state_checks"].values()
            ),
            "all_role_outputs_complete": True,
            "orphan_compute_children_zero": (
                process_after["compute_child_process_count"] == 0
            ),
            "clean_only": True,
        }
        if not all(automatic_checks.values()):
            raise RuntimeError(
                f"CCR formal automatic gate failed: {automatic_checks}"
            )
        _write_npz(
            temporary / "states.npz",
            _state_artifact_arrays(science["states"]),
        )
        _write_npz(
            temporary / "held_outputs.npz",
            _output_artifact_arrays(
                role_outputs=science["role_outputs"],
                same_weight_outputs=science["same_weight_outputs"],
                keeper_scores=science["keeper_scores"],
                cohort=cohort,
            ),
        )
        _write_json(
            temporary / "training_records.json",
            {
                "descriptor_records": science["descriptor_records"],
                "roles": science["training"],
                "primary_initial_state_checks": science[
                    "primary_initial_state_checks"
                ],
            },
        )
        _write_json(temporary / "metrics.json", science["metrics"])
        _write_json(temporary / "access_ledger.json", access)
        clean_passed = bool(
            science["metrics"]["clean_gate"]["passed"]
        )
        value_label = (
            "mechanism-only"
            if clean_passed
            else "negative-but-reusable"
        )
        summary = {
            "schema_version": 1,
            "protocol_id": lock["protocol_id"],
            "state": "clean_fit_fresh_process_replay_pending",
            "value_label": value_label,
            "process_id": os.getpid(),
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "lock_sha256": lock_sha,
            "materializer_evidence_sha256": evidence_sha,
            "authorization_sha256": authorization_sha,
            "authorization": authorization_check,
            "repository": repository,
            "process_before": process_before,
            "process_after": process_after,
            "runner_sha256": sha256_file(RUNNER_PATH),
            "engine_sha256": sha256_file(ENGINE_PATH),
            "cache_verification": cache_verification,
            "descriptor_records": science["descriptor_records"],
            "dephase_offsets_sha256": science[
                "dephase_offsets_sha256"
            ],
            "resource": {
                "elapsed_seconds": elapsed,
                "peak_process_rss_bytes": monitor.peak_rss,
                "peak_cuda_allocated_bytes": peak_cuda,
                "maximum_combined_temporary_bytes": science[
                    "maximum_combined_temporary_bytes"
                ],
                "limits": lock["resource_limits"],
                "checks": resource_checks,
            },
            "access_ledger_sha256": access[
                "ordered_events_sha256"
            ],
            "automatic_checks": automatic_checks,
            "automatic_passed": True,
            "clean_gate": science["metrics"]["clean_gate"],
            "clean_gate_passed": clean_passed,
            "shifted_conditions_run": False,
            "shifted_conditions_authorized": clean_passed,
            "xai_run": False,
            "deployment_run": False,
            "candidate_descriptor_created": True,
            "candidate_state_created": True,
            "candidate_metric_observed": True,
            "validation_data_used": False,
            "test_data_used": False,
            "downstream_authorization": {
                "shifted_conditions": clean_passed,
                "xai": False,
                "deployment": False,
                "validation": False,
                "test": False,
                "production_integration": False,
                "full_train": False,
                "current_best_command_update": False,
            },
        }
        _write_json(temporary / "summary.json", summary)
        retained_before_manifest = _directory_bytes(temporary)
        retained_limit = int(lock["resource_limits"]["retained_bytes"])
        if retained_before_manifest + 65536 > retained_limit:
            raise RuntimeError(
                "CCR retained payload exceeds prospective limit: "
                f"{retained_before_manifest}/{retained_limit}"
            )
        manifest = _directory_manifest(
            temporary,
            excluded_names=(FORMAL_MANIFEST_NAME,),
        )
        _write_json(temporary / FORMAL_MANIFEST_NAME, manifest)
        if (
            _directory_manifest(
                temporary,
                excluded_names=(FORMAL_MANIFEST_NAME,),
            )
            != manifest
        ):
            raise RuntimeError("CCR formal manifest differs after write")
        if _directory_bytes(temporary) > retained_limit:
            raise RuntimeError("CCR retained artifact set exceeds limit")
        os.replace(temporary, output)
        return {
            "passed": True,
            "output": str(output),
            "summary": summary,
            "manifest": manifest,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _compare_npz_artifacts(
    reference: Path,
    observed: Path,
    *,
    tolerance: float,
) -> Dict[str, object]:
    first = _npz_arrays(reference)
    second = _npz_arrays(observed)
    if set(first) != set(second):
        raise RuntimeError(
            f"CCR replay NPZ keys differ: {reference.name}"
        )
    maximum_error = 0.0
    exact = True
    per_array: Dict[str, object] = {}
    for name in sorted(first):
        left = first[name]
        right = second[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"CCR replay array contract differs: "
                f"{reference.name}/{name}"
            )
        if left.dtype.kind in {"f", "c"}:
            error = float(
                np.max(np.abs(left - right), initial=0.0)
            )
            finite = bool(
                np.isfinite(left).all() and np.isfinite(right).all()
            )
            passed = finite and error <= float(tolerance)
        else:
            error = 0.0 if np.array_equal(left, right) else math.inf
            passed = error == 0.0
        maximum_error = max(maximum_error, error)
        exact = exact and bool(np.array_equal(left, right))
        per_array[name] = {
            "maximum_absolute_error": error,
            "exact": bool(np.array_equal(left, right)),
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                f"CCR replay array differs: "
                f"{reference.name}/{name}/{error}"
            )
    return {
        "array_count": len(first),
        "maximum_absolute_error": maximum_error,
        "all_exact": exact,
        "per_array": per_array,
        "passed": True,
    }


def _scrub_training_record(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_training_record(item)
            for key, item in value.items()
            if key != "effective_worker_pids"
        }
    if isinstance(value, list):
        return [_scrub_training_record(item) for item in value]
    return value


def replay_formal(
    *,
    output: Path,
    authorization_path: Path,
    lock_path: Path = LOCK_PATH,
    cache_dir: Path = DEFAULT_CACHE,
) -> Dict[str, object]:
    output = Path(output).resolve()
    reference_summary = _read_json(output / "summary.json")
    if int(reference_summary["process_id"]) == os.getpid():
        raise RuntimeError("CCR replay must run in a fresh process")
    _verify_manifest(output, FORMAL_MANIFEST_NAME)
    for name in (
        REPLAY_SUMMARY_NAME,
        REPLAY_MANIFEST_NAME,
        ARTIFACT_SET_MANIFEST_NAME,
    ):
        if (output / name).exists():
            raise FileExistsError(f"CCR replay artifact exists: {name}")
    replay_output = output.parent / f".{output.name}.replay.{os.getpid()}"
    if replay_output.exists():
        raise FileExistsError(
            f"CCR replay temporary output exists: {replay_output}"
        )
    started = time.perf_counter()
    try:
        replay = run_formal(
            output=replay_output,
            authorization_path=authorization_path,
            lock_path=lock_path,
            cache_dir=cache_dir,
            authorization_output=output,
        )
        observed_summary = replay["summary"]
        state_comparison = _compare_npz_artifacts(
            output / "states.npz",
            replay_output / "states.npz",
            tolerance=REPLAY_NUMERIC_TOLERANCE,
        )
        output_comparison = _compare_npz_artifacts(
            output / "held_outputs.npz",
            replay_output / "held_outputs.npz",
            tolerance=REPLAY_NUMERIC_TOLERANCE,
        )
        reference_metrics = _read_json(output / "metrics.json")
        observed_metrics = _read_json(replay_output / "metrics.json")
        reference_training = _scrub_training_record(
            _read_json(output / "training_records.json")
        )
        observed_training = _scrub_training_record(
            _read_json(replay_output / "training_records.json")
        )
        checks = {
            "fresh_process": int(reference_summary["process_id"])
            != int(observed_summary["process_id"]),
            "lock_exact": reference_summary["lock_sha256"]
            == observed_summary["lock_sha256"],
            "authorization_exact": reference_summary[
                "authorization_sha256"
            ]
            == observed_summary["authorization_sha256"],
            "runner_exact": reference_summary["runner_sha256"]
            == observed_summary["runner_sha256"],
            "engine_exact": reference_summary["engine_sha256"]
            == observed_summary["engine_sha256"],
            "descriptor_records_exact": reference_summary[
                "descriptor_records"
            ]
            == observed_summary["descriptor_records"],
            "dephase_offsets_exact": reference_summary[
                "dephase_offsets_sha256"
            ]
            == observed_summary["dephase_offsets_sha256"],
            "states_within_1e_7": bool(state_comparison["passed"]),
            "outputs_within_1e_7": bool(output_comparison["passed"]),
            "metrics_exact": json_sha256(reference_metrics)
            == json_sha256(observed_metrics),
            "training_science_exact": json_sha256(reference_training)
            == json_sha256(observed_training),
            "access_ledger_exact": reference_summary[
                "access_ledger_sha256"
            ]
            == observed_summary["access_ledger_sha256"],
            "clean_gate_decision_exact": reference_summary[
                "clean_gate_passed"
            ]
            == observed_summary["clean_gate_passed"],
            "formal_automatic_passed": bool(
                observed_summary["automatic_passed"]
            ),
            "validation_unused": not bool(
                observed_summary["validation_data_used"]
            ),
            "test_unused": not bool(
                observed_summary["test_data_used"]
            ),
        }
        replay_summary = {
            "schema_version": 1,
            "protocol_id": reference_summary["protocol_id"],
            "state": "fresh_process_clean_fit_replay_complete",
            "reference_process_id": int(reference_summary["process_id"]),
            "replay_process_id": int(observed_summary["process_id"]),
            "elapsed_seconds": time.perf_counter() - started,
            "numeric_tolerance": REPLAY_NUMERIC_TOLERANCE,
            "state_comparison": state_comparison,
            "output_comparison": output_comparison,
            "checks": checks,
            "passed": bool(all(checks.values())),
            "clean_gate_passed": bool(
                reference_summary["clean_gate_passed"]
            ),
            "validation_data_used": False,
            "test_data_used": False,
            "downstream_authorization": {
                "validation": False,
                "test": False,
                "production_integration": False,
                "full_train": False,
                "current_best_command_update": False,
            },
        }
        if not replay_summary["passed"]:
            raise RuntimeError(f"CCR replay differs: {checks}")
        _write_json(output / REPLAY_SUMMARY_NAME, replay_summary)
        replay_manifest = _directory_manifest(
            output,
            excluded_names=(
                REPLAY_MANIFEST_NAME,
                ARTIFACT_SET_MANIFEST_NAME,
            ),
        )
        _write_json(output / REPLAY_MANIFEST_NAME, replay_manifest)
        _verify_manifest(output, REPLAY_MANIFEST_NAME)
        artifact_set = _directory_manifest(
            output,
            excluded_names=(ARTIFACT_SET_MANIFEST_NAME,),
        )
        _write_json(
            output / ARTIFACT_SET_MANIFEST_NAME,
            artifact_set,
        )
        _verify_manifest(output, ARTIFACT_SET_MANIFEST_NAME)
        return replay_summary
    except Exception:
        for name in (
            REPLAY_SUMMARY_NAME,
            REPLAY_MANIFEST_NAME,
            ARTIFACT_SET_MANIFEST_NAME,
        ):
            path = output / name
            if path.exists():
                path.unlink()
        raise
    finally:
        if replay_output.exists():
            shutil.rmtree(replay_output)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prospectively locked train-only CCR information gate."
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--formal", action="store_true")
    action.add_argument("--replay", action="store_true")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        result = structural_preflight(
            lock_path=args.lock,
            cache_dir=args.cache_dir,
            output=args.output_dir,
        )
    else:
        if args.authorization is None:
            raise ValueError(
                "--authorization is required for formal/replay"
            )
        if args.formal:
            result = run_formal(
                output=args.output_dir,
                authorization_path=args.authorization,
                lock_path=args.lock,
                cache_dir=args.cache_dir,
            )
        else:
            result = replay_formal(
                output=args.output_dir,
                authorization_path=args.authorization,
                lock_path=args.lock,
                cache_dir=args.cache_dir,
            )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0 if bool(result.get("passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

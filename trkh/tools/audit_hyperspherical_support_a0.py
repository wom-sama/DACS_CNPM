from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import psutil
import scipy
import sklearn
from scipy.special import ndtr, ndtri
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from threadpoolctl import threadpool_info, threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20260721
FOLDS = 5
CLASS_COUNT = 5
FOCUS_CLASS = 1
RESTRICTED_RIVALS = (0, 2, 4)
EXPECTED_ROWS = 9215
EXPECTED_EMBEDDING_DIM = 256
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
ROLES = (
    "natural_control",
    "duplicate_control",
    "cap_candidate",
    "cap_seed_repeat",
    "nearest_center_deranged",
)

PROTOCOL_COMMIT = "b28d9611f7d8c3fc3cdb47aab1699bfd0aafe7c7"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_HYPERSPHERICAL_SUPPORT_A0_PROTOCOL_20260721.md"
)
ERRATUM_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_HYPERSPHERICAL_SUPPORT_A0_INCOMPLETE_FINALIZATION_ERRATUM_20260721.md"
)
TRAIN_CACHE_PATH = (
    REPO_ROOT
    / "runs"
    / "diagnostic_reslt_embedding_cache_keeper_yolof_20260712"
    / "train_embeddings.npz"
)
CACHE_MANIFEST_PATH = TRAIN_CACHE_PATH.parent / "embedding_cache_manifest.json"
CIDT_DIR = REPO_ROOT / "runs" / "audit_cidt_readiness_full_train_20260714"
CIDT_PREDICTIONS_PATH = CIDT_DIR / "predictions_all_conditions.csv"
CIDT_SUMMARY_PATH = CIDT_DIR / "summary.json"
KEEPER_PATH = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
    / "checkpoints"
    / "best.pt"
)
DATA_YAML_PATH = Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml")
CURRENT_BEST_PATH = (
    REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
)
CURRENT_BEST_HISTORY_PATH = (
    REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
)

LOCKED_FILES = (
    (
        "protocol",
        PROTOCOL_PATH,
        "f9485b7bb0ed22b8ff5d6b4cc2f79af0f766b409d43eef2be3f70518410b280c",
    ),
    (
        "incomplete_finalization_erratum",
        ERRATUM_PATH,
        "1c19383dfde7c6216220a88a8d7833b07ba8df3c6432642d79580b1c6e6d1258",
    ),
    (
        "train_embedding_cache",
        TRAIN_CACHE_PATH,
        "157805b449549c9ad87f56f11ece2ed756669bcf11332c860c99feca92314464",
    ),
    (
        "embedding_cache_manifest",
        CACHE_MANIFEST_PATH,
        "24c617e96f6863b8434edf03f8d611a4ea2da27cb33c30df9550ac2603a46dd2",
    ),
    (
        "cidt_predictions",
        CIDT_PREDICTIONS_PATH,
        "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    ),
    (
        "cidt_summary",
        CIDT_SUMMARY_PATH,
        "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    ),
    (
        "keeper_checkpoint",
        KEEPER_PATH,
        "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    ),
    (
        "data_yaml",
        DATA_YAML_PATH,
        "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    ),
    (
        "current_best_commands",
        CURRENT_BEST_PATH,
        "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    ),
    (
        "current_best_history",
        CURRENT_BEST_HISTORY_PATH,
        "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    ),
)

INCOMPLETE_PAYLOADS = {
    "fold_assignment.json": {
        "size_bytes": 4246,
        "sha256": "162f75c959957ce89c72a43b44a916ebf56b458b7044fc016bb6991727ef97fc",
    },
    "fold_metrics.csv": {
        "size_bytes": 3008,
        "sha256": "79ccb358096a893d94a6f42a1e7a307052e9ccd06bf814659560e3da0c6e662c",
    },
    "fold_protocol.json": {
        "size_bytes": 25721,
        "sha256": "b45852a53dc015740fb4b945be459e91d76b50fbaa34851c8e133728b8a51966",
    },
    "predictions.csv": {
        "size_bytes": 6221489,
        "sha256": "ec86e45c8d539e48d1bcb6801e09476acb80226f82f4839f8eb48eecc467da20",
    },
    "synthetic_geometry.csv": {
        "size_bytes": 51637,
        "sha256": "051f454e47c133fc6ba97b21c67b5eb60c6baa4ce42c87fbda44827888577134",
    },
    "synthetic_geometry.json": {
        "size_bytes": 119813,
        "sha256": "9ed3d9241718b49931de26c8640a95f4f97ca7e0ac4207318e67e1ade7fad796",
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prospectively locked train-only hyperspherical-support A0."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_hyperspherical_support_a0_20260721",
    )
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-incomplete", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _string_sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Locked {label} is missing: {resolved}")
    observed = _sha256(resolved)
    if observed != str(expected).strip().lower():
        raise ValueError(
            f"Locked {label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return observed


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _protocol_is_ancestor(head: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            PROTOCOL_COMMIT,
            head,
        ],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def verify_locked_inputs() -> Dict[str, object]:
    files: Dict[str, object] = {}
    for label, path, expected in LOCKED_FILES:
        resolved = path.expanduser().resolve()
        files[label] = {
            "path": str(resolved),
            "sha256": _verify_hash(resolved, expected, label),
        }

    head = _git_value("rev-parse", "HEAD")
    upstream = _git_value("rev-parse", "@{upstream}")
    tracked_status = _git_value(
        "status", "--porcelain", "--untracked-files=no"
    )
    if head != upstream or tracked_status or not _protocol_is_ancestor(head):
        raise ValueError(
            "Formal A0 requires a clean pushed tracked worktree with the locked "
            f"protocol ancestor: head={head}, upstream={upstream}, "
            f"dirty={bool(tracked_status)}"
        )
    return {
        "files": files,
        "repo": {
            "head": head,
            "upstream": upstream,
            "head_matches_upstream": True,
            "tracked_worktree_clean": True,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_commit_is_ancestor": True,
        },
        "python": sys.version,
        "numpy": np.__version__,
    }


def verify_preflight_checks() -> Dict[str, object]:
    commands = {
        "py_compile": [
            sys.executable,
            "-m",
            "py_compile",
            "trkh/tools/audit_hyperspherical_support_a0.py",
            "tests/test_audit_hyperspherical_support_a0.py",
        ],
        "pyflakes": [
            sys.executable,
            "-m",
            "pyflakes",
            "trkh/tools/audit_hyperspherical_support_a0.py",
            "tests/test_audit_hyperspherical_support_a0.py",
        ],
        "focused_tests": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_audit_hyperspherical_support_a0.py",
            "-q",
        ],
        "git_diff_check": ["git", "diff", "--check"],
    }
    results: Dict[str, object] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "return_code": int(completed.returncode),
            "stdout_tail": completed.stdout.strip()[-2000:],
            "stderr_tail": completed.stderr.strip()[-2000:],
        }
        if completed.returncode != 0:
            raise RuntimeError(
                f"Preflight check failed: {name}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
    return results


def _require_empty_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_absent_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"Preflight output path must not exist: {resolved}")
    return resolved


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
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
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        Path(path).write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


class _PeakRssMonitor:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.peak_bytes = int(self.process.memory_info().rss)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(
                self.peak_bytes, int(self.process.memory_info().rss)
            )

    def __enter__(self) -> "_PeakRssMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, int(self.process.memory_info().rss))


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


def load_locked_train_cache(path: Path = TRAIN_CACHE_PATH) -> Dict[str, np.ndarray]:
    resolved = Path(path).expanduser().resolve()
    if _contains_forbidden_split(str(resolved)):
        raise ValueError(f"Cache path contains a forbidden split marker: {resolved}")
    with np.load(resolved, allow_pickle=True) as payload:
        observed_keys = tuple(payload.files)
        if observed_keys != EXPECTED_CACHE_KEYS:
            raise ValueError(
                f"Cache keys/order mismatch: {observed_keys} != {EXPECTED_CACHE_KEYS}"
            )
        cache = {key: np.asarray(payload[key]).copy() for key in payload.files}

    expected_shapes = {
        "embeddings": (EXPECTED_ROWS, EXPECTED_EMBEDDING_DIM),
        "probabilities": (EXPECTED_ROWS, CLASS_COUNT),
        "labels": (EXPECTED_ROWS,),
        "base_predictions": (EXPECTED_ROWS,),
        "paths": (EXPECTED_ROWS,),
        "sample_index": (EXPECTED_ROWS,),
    }
    for key, shape in expected_shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"Cache {key} shape mismatch: {cache[key].shape} != {shape}")

    embeddings = np.asarray(cache["embeddings"], dtype=np.float64)
    probabilities = np.asarray(cache["probabilities"], dtype=np.float64)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    predictions = np.asarray(cache["base_predictions"], dtype=np.int64)
    paths = np.asarray(cache["paths"]).astype(str)
    sample_index = np.asarray(cache["sample_index"], dtype=np.int64)
    if not np.isfinite(embeddings).all() or not np.isfinite(probabilities).all():
        raise ValueError("Cache contains non-finite embeddings or probabilities")
    if bool((probabilities < 0.0).any()) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=0.0
    ):
        raise ValueError("Cache probabilities are invalid")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise ValueError("Cache base predictions do not match probabilities")
    if not np.array_equal(sample_index, np.arange(EXPECTED_ROWS, dtype=np.int64)):
        raise ValueError("Cache sample indices are not complete and ordered")
    if tuple(np.bincount(labels, minlength=CLASS_COUNT).tolist()) != EXPECTED_CLASS_COUNTS:
        raise ValueError("Cache class counts differ from the prospective lock")
    if any(not _is_train_image_path(value) for value in paths.tolist()):
        raise ValueError("Cache contains a non-train or forbidden image path")

    return {
        "embeddings": embeddings,
        "probabilities": probabilities,
        "labels": labels,
        "base_predictions": predictions,
        "paths": paths,
        "sample_index": sample_index,
        "source_stems": np.asarray(
            [Path(value).stem.casefold() for value in paths], dtype=str
        ),
    }


def _summarize_folds(
    labels: np.ndarray, groups: np.ndarray, folds: np.ndarray
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).astype(str).reshape(-1)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    if not (labels.size == groups.size == folds.size):
        raise ValueError("Fold labels, groups, and assignments do not align")
    if set(np.unique(folds).tolist()) != set(range(FOLDS)):
        raise ValueError("Fold assignment does not contain exactly folds 0..4")
    rows: List[Dict[str, object]] = []
    total_overlap = 0
    for fold in range(FOLDS):
        fit_indices = np.flatnonzero(folds != fold)
        hold_indices = np.flatnonzero(folds == fold)
        fit_sources = set(groups[fit_indices].tolist())
        hold_sources = set(groups[hold_indices].tolist())
        overlap = fit_sources.intersection(hold_sources)
        if set(np.unique(labels[fit_indices]).tolist()) != set(range(CLASS_COUNT)):
            raise ValueError(f"Fit fold {fold} is missing a class")
        if set(np.unique(labels[hold_indices]).tolist()) != set(range(CLASS_COUNT)):
            raise ValueError(f"Holdout fold {fold} is missing a class")
        total_overlap += len(overlap)
        rows.append(
            {
                "fold": fold,
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(hold_indices.size),
                "fit_sources": int(len(fit_sources)),
                "holdout_sources": int(len(hold_sources)),
                "source_overlap": int(len(overlap)),
                "fit_class_counts": np.bincount(
                    labels[fit_indices], minlength=CLASS_COUNT
                ).tolist(),
                "holdout_class_counts": np.bincount(
                    labels[hold_indices], minlength=CLASS_COUNT
                ).tolist(),
                "fit_indices_sha256": _array_sha256(
                    fit_indices.astype(np.int64)
                ),
                "holdout_indices_sha256": _array_sha256(
                    hold_indices.astype(np.int64)
                ),
                "fit_sources_sha256": _string_sequence_sha256(
                    sorted(fit_sources)
                ),
                "holdout_sources_sha256": _string_sequence_sha256(
                    sorted(hold_sources)
                ),
            }
        )
    return {
        "fold_count": FOLDS,
        "assignment_complete": True,
        "assignment_counts": np.bincount(folds, minlength=FOLDS).tolist(),
        "assignment_sha256": _array_sha256(folds),
        "source_overlap": int(total_overlap),
        "folds": rows,
    }


def load_locked_cidt_folds(
    cache: Mapping[str, np.ndarray], path: Path = CIDT_PREDICTIONS_PATH
) -> Tuple[np.ndarray, Dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if _contains_forbidden_split(str(resolved)):
        raise ValueError(f"CIDT path contains a forbidden split marker: {resolved}")
    assignments = np.full(EXPECTED_ROWS, -1, dtype=np.int64)
    observed = np.zeros(EXPECTED_ROWS, dtype=bool)
    with resolved.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row["condition"]).strip().casefold() != "clean":
                continue
            sample_index = int(row["sample_index"])
            if sample_index < 0 or sample_index >= EXPECTED_ROWS or observed[sample_index]:
                raise ValueError(f"CIDT clean sample index is invalid: {sample_index}")
            image_path = str(row["image_path"])
            source_stem = str(row["source_stem"]).casefold()
            if not _is_train_image_path(image_path):
                raise ValueError(f"CIDT contains a forbidden image path: {image_path}")
            if int(row["target_index"]) != int(cache["labels"][sample_index]):
                raise ValueError(f"CIDT/cache target mismatch at {sample_index}")
            if source_stem != str(cache["source_stems"][sample_index]).casefold():
                raise ValueError(f"CIDT/cache source mismatch at {sample_index}")
            if Path(image_path).name.casefold() != Path(
                str(cache["paths"][sample_index])
            ).name.casefold():
                raise ValueError(f"CIDT/cache path mismatch at {sample_index}")
            assignments[sample_index] = int(row["fold"])
            observed[sample_index] = True
    if not observed.all():
        raise ValueError(
            f"CIDT clean assignment is incomplete: {int(observed.sum())}/{EXPECTED_ROWS}"
        )
    if tuple(np.bincount(assignments, minlength=FOLDS).tolist()) != EXPECTED_FOLD_COUNTS:
        raise ValueError("CIDT fold counts differ from the prospective lock")
    summary = _summarize_folds(
        cache["labels"], cache["source_stems"], assignments
    )
    if int(summary["source_overlap"]) != 0:
        raise ValueError("CIDT source-fold assignment leaks source stems")
    return assignments, summary


def l2_normalize(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool((norms <= epsilon).any()):
        raise ValueError("Cannot normalize a zero embedding")
    return array / norms


def estimate_confidence_support(
    features: np.ndarray, *, alpha: float = 0.99
) -> Dict[str, object]:
    normalized = l2_normalize(features)
    center_raw = normalized.mean(axis=0)
    center_norm = float(np.linalg.norm(center_raw))
    if center_norm <= 1e-12:
        raise ValueError("Class support center is degenerate")
    center = center_raw / center_norm
    angles = np.arccos(np.clip(normalized @ center, -1.0, 1.0))
    quantile = float(np.quantile(angles, float(alpha), method="linear"))
    simplex_radius = float(0.5 * math.acos(-1.0 / float(CLASS_COUNT - 1)))
    radius = min(quantile, simplex_radius)
    return {
        "center": center,
        "center_raw_norm": center_norm,
        "angle_mean": float(angles.mean()),
        "angle_std": float(angles.std(ddof=0)),
        "angle_min": float(angles.min()),
        "angle_max": float(angles.max()),
        "angle_quantile_099": quantile,
        "simplex_radius": simplex_radius,
        "cap_radius": radius,
        "angles_sha256": _array_sha256(angles),
    }


def _sample_truncated_angles(
    *,
    mean: float,
    std: float,
    radius: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 0 or radius < 0.0:
        raise ValueError("Synthetic count and cap radius must be nonnegative")
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if std < 1e-12:
        return np.full(count, np.clip(mean, 0.0, radius), dtype=np.float64)
    lower = float(ndtr((0.0 - mean) / std))
    upper = float(ndtr((radius - mean) / std))
    if not math.isfinite(lower + upper) or upper - lower <= 1e-15:
        return np.full(count, np.clip(mean, 0.0, radius), dtype=np.float64)
    probabilities = lower + (upper - lower) * rng.random(count)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    angles = mean + std * ndtri(probabilities)
    return np.clip(angles, 0.0, radius).astype(np.float64, copy=False)


def generate_hyperspherical_support(
    features: np.ndarray,
    *,
    count: int,
    seed: int,
    center_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    support = estimate_confidence_support(features)
    empirical_center = np.asarray(support["center"], dtype=np.float64)
    center = (
        empirical_center
        if center_override is None
        else np.asarray(center_override, dtype=np.float64).reshape(-1)
    )
    center_norm = float(np.linalg.norm(center))
    if center.shape != empirical_center.shape or center_norm <= 1e-12:
        raise ValueError("Synthetic support center override is invalid")
    center = center / center_norm
    rng = np.random.default_rng(int(seed))
    angles = _sample_truncated_angles(
        mean=float(support["angle_mean"]),
        std=float(support["angle_std"]),
        radius=float(support["cap_radius"]),
        count=int(count),
        rng=rng,
    )
    if count == 0:
        generated = np.empty((0, center.size), dtype=np.float64)
        tangent_max_abs_dot = 0.0
        angle_reconstruction_error = 0.0
        norm_error = 0.0
        max_angle_excess = 0.0
    else:
        tangent = rng.standard_normal((int(count), center.size), dtype=np.float64)
        tangent -= (tangent @ center)[:, None] * center[None, :]
        tangent_norm = np.linalg.norm(tangent, axis=1)
        degenerate = tangent_norm <= 1e-12
        while bool(degenerate.any()):
            replacements = rng.standard_normal(
                (int(degenerate.sum()), center.size), dtype=np.float64
            )
            replacements -= (replacements @ center)[:, None] * center[None, :]
            tangent[degenerate] = replacements
            tangent_norm = np.linalg.norm(tangent, axis=1)
            degenerate = tangent_norm <= 1e-12
        tangent /= tangent_norm[:, None]
        generated = (
            np.cos(angles)[:, None] * center[None, :]
            + np.sin(angles)[:, None] * tangent
        )
        generated = l2_normalize(generated)
        observed_angles = np.arccos(np.clip(generated @ center, -1.0, 1.0))
        tangent_max_abs_dot = float(np.max(np.abs(tangent @ center)))
        angle_reconstruction_error = float(np.max(np.abs(observed_angles - angles)))
        norm_error = float(np.max(np.abs(np.linalg.norm(generated, axis=1) - 1.0)))
        max_angle_excess = float(
            max(0.0, observed_angles.max() - float(support["cap_radius"]))
        )

    record = {
        key: value for key, value in support.items() if key != "center"
    }
    record.update(
        {
            "seed": int(seed),
            "generated_count": int(count),
            "empirical_center_sha256": _array_sha256(empirical_center),
            "generation_center_sha256": _array_sha256(center),
            "center_deranged": center_override is not None,
            "sampled_angle_min": float(angles.min()) if angles.size else 0.0,
            "sampled_angle_max": float(angles.max()) if angles.size else 0.0,
            "sampled_angle_mean": float(angles.mean()) if angles.size else 0.0,
            "sampled_angles_sha256": _array_sha256(angles),
            "generated_sha256": _array_sha256(generated),
            "maximum_norm_error": norm_error,
            "maximum_cap_angle_excess": max_angle_excess,
            "maximum_tangent_abs_dot": tangent_max_abs_dot,
            "maximum_angle_reconstruction_error": angle_reconstruction_error,
        }
    )
    return generated, record


def nearest_rival_centers(centers: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(centers)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return similarities.argmax(axis=1).astype(np.int64)


def duplicate_balance(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=CLASS_COUNT)
    target = int(counts.max())
    appended_features = [features]
    appended_labels = [labels]
    records: List[Dict[str, object]] = []
    for class_index in range(CLASS_COUNT):
        indices = np.flatnonzero(labels == class_index)
        deficit = target - int(indices.size)
        rng = np.random.default_rng(int(seed) + 131 * class_index)
        order = indices[rng.permutation(indices.size)]
        selected = np.resize(order, deficit) if deficit else np.empty(0, dtype=np.int64)
        if deficit:
            appended_features.append(features[selected])
            appended_labels.append(np.full(deficit, class_index, dtype=np.int64))
        records.append(
            {
                "class_index": class_index,
                "real_count": int(indices.size),
                "duplicate_count": int(deficit),
                "selected_indices_sha256": _array_sha256(selected),
            }
        )
    balanced_features = np.concatenate(appended_features, axis=0)
    balanced_labels = np.concatenate(appended_labels, axis=0)
    balanced_counts = np.bincount(balanced_labels, minlength=CLASS_COUNT)
    if not np.array_equal(balanced_counts, np.full(CLASS_COUNT, target)):
        raise RuntimeError("Duplicate balancing failed to produce exact class counts")
    return balanced_features, balanced_labels, {
        "target_count": target,
        "balanced_counts": balanced_counts.tolist(),
        "rows": int(balanced_labels.size),
        "classes": records,
    }


def _cap_balance(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    fold: int,
    seed_offset: int = 0,
    deranged: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=CLASS_COUNT)
    target = int(counts.max())
    supports = [
        estimate_confidence_support(features[labels == class_index])
        for class_index in range(CLASS_COUNT)
    ]
    centers = np.stack([np.asarray(item["center"]) for item in supports], axis=0)
    radii = np.asarray([float(item["cap_radius"]) for item in supports])
    center_angles = np.arccos(np.clip(centers @ centers.T, -1.0, 1.0))
    rivals = nearest_rival_centers(centers)

    appended_features = [features]
    appended_labels = [labels]
    records: List[Dict[str, object]] = []
    for class_index in range(CLASS_COUNT):
        class_features = features[labels == class_index]
        deficit = target - int(class_features.shape[0])
        center_override = centers[rivals[class_index]] if deranged else None
        seed = SEED + 1009 * int(fold) + 131 * class_index + int(seed_offset)
        generated, record = generate_hyperspherical_support(
            class_features,
            count=deficit,
            seed=seed,
            center_override=center_override,
        )
        if deficit:
            appended_features.append(generated)
            appended_labels.append(np.full(deficit, class_index, dtype=np.int64))
        rival_margins = [
            float(center_angles[class_index, rival] - radii[class_index] - radii[rival])
            for rival in range(CLASS_COUNT)
            if rival != class_index
        ]
        record.update(
            {
                "fold": int(fold),
                "class_index": class_index,
                "real_count": int(class_features.shape[0]),
                "target_count": target,
                "nearest_rival": int(rivals[class_index]),
                "nearest_rival_cosine": float(
                    centers[class_index] @ centers[rivals[class_index]]
                ),
                "minimum_cap_separation_margin": float(min(rival_margins)),
                "nonoverlap_rival_count": int(
                    sum(margin >= 0.0 for margin in rival_margins)
                ),
            }
        )
        records.append(record)
    balanced_features = np.concatenate(appended_features, axis=0)
    balanced_labels = np.concatenate(appended_labels, axis=0)
    if not np.array_equal(
        np.bincount(balanced_labels, minlength=CLASS_COUNT),
        np.full(CLASS_COUNT, target),
    ):
        raise RuntimeError("Cap balancing failed to produce exact class counts")
    return balanced_features, balanced_labels, records


def _fit_readout(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    holdout_features: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    model = LogisticRegression(
        C=0.3,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-9,
        class_weight=None,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(fit_features, fit_labels)
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(CLASS_COUNT, dtype=np.int64)):
        raise ValueError(f"Readout classes are incomplete or reordered: {classes}")
    probabilities = np.asarray(model.predict_proba(holdout_features), dtype=np.float64)
    iterations = np.asarray(model.n_iter_, dtype=np.int64)
    record = {
        "classes": classes.tolist(),
        "iterations": iterations.tolist(),
        "maximum_iterations": int(iterations.max(initial=0)),
        "convergence_warnings": convergence_warnings,
        "converged": not convergence_warnings
        and int(iterations.max(initial=0)) < 2000,
        "coefficient_shape": list(model.coef_.shape),
        "coefficient_sha256": _array_sha256(model.coef_),
        "intercept_sha256": _array_sha256(model.intercept_),
    }
    return probabilities, record


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
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
        raise ValueError("Probability matrix shape does not match targets")
    if not np.isfinite(probabilities).all() or bool((probabilities < 0.0).any()):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("Probabilities are not normalized")
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(CLASS_COUNT),
        zero_division=0,
    )
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[targets]
    per_class = []
    predicted_support = np.bincount(predictions, minlength=CLASS_COUNT)
    for class_index in range(CLASS_COUNT):
        per_class.append(
            {
                "class_index": class_index,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
                "predicted_support": int(predicted_support[class_index]),
            }
        )
    return {
        "samples": int(targets.size),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=np.arange(CLASS_COUNT)
        ).tolist(),
        "nll": float(log_loss(targets, probabilities, labels=np.arange(CLASS_COUNT))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_15": _ece(targets, probabilities, bins=15),
    }


def normalize_keeper_reference(
    probabilities: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CLASS_COUNT:
        raise ValueError("Keeper reference must have shape [N,5]")
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("Keeper reference contains invalid values")
    row_sums = values.sum(axis=1, keepdims=True)
    if bool((row_sums <= 0.0).any()):
        raise ValueError("Keeper reference contains a zero probability row")
    maximum_error = float(np.max(np.abs(row_sums[:, 0] - 1.0)))
    if maximum_error > 2e-4:
        raise ValueError(
            "Keeper reference row-sum error exceeds the locked cache tolerance: "
            f"{maximum_error}"
        )
    normalized = values / row_sums
    return normalized, {
        "normalization_applied": True,
        "source_dtype": str(np.asarray(probabilities).dtype),
        "maximum_pre_normalization_row_sum_error": maximum_error,
        "maximum_post_normalization_row_sum_error": float(
            np.max(np.abs(normalized.sum(axis=1) - 1.0))
        ),
        "normalized_probabilities_sha256": _array_sha256(normalized),
        "enters_candidate_or_gate": False,
    }


def _focus(metrics: Mapping[str, object]) -> Mapping[str, object]:
    return metrics["per_class"][FOCUS_CLASS]


def transition_stats(
    targets: np.ndarray,
    base_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> Dict[str, int]:
    targets = np.asarray(targets, dtype=np.int64)
    base = np.asarray(base_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    restricted = np.isin(targets, RESTRICTED_RIVALS)
    base_tp = (targets == FOCUS_CLASS) & (base == FOCUS_CLASS)
    candidate_tp = (targets == FOCUS_CLASS) & (candidate == FOCUS_CLASS)
    base_fp = restricted & (base == FOCUS_CLASS)
    candidate_fp = restricted & (candidate == FOCUS_CLASS)
    return {
        "changed": int(np.sum(base != candidate)),
        "candidate_correction": int(np.sum((base != targets) & (candidate == targets))),
        "candidate_harm": int(np.sum((base == targets) & (candidate != targets))),
        "focus_fn_rescue": int(
            np.sum((targets == FOCUS_CLASS) & (base != FOCUS_CLASS) & candidate_tp)
        ),
        "focus_tp_break": int(
            np.sum(base_tp & (candidate != FOCUS_CLASS))
        ),
        "focus_tp_base": int(base_tp.sum()),
        "focus_tp_candidate": int(candidate_tp.sum()),
        "focus_tp_net": int(candidate_tp.sum() - base_tp.sum()),
        "restricted_fp_base": int(base_fp.sum()),
        "restricted_fp_candidate": int(candidate_fp.sum()),
        "restricted_fp_remove": int(np.sum(base_fp & (candidate != FOCUS_CLASS))),
        "restricted_fp_create": int(np.sum((~base_fp) & candidate_fp)),
        "restricted_fp_net_removal": int(base_fp.sum() - candidate_fp.sum()),
    }


def analyze_predictions(
    targets: np.ndarray,
    folds: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    if set(outputs) != set(ROLES):
        raise ValueError(f"Output roles differ from the protocol: {tuple(outputs)}")
    if targets.size != folds.size or set(np.unique(folds).tolist()) != set(range(FOLDS)):
        raise ValueError("Targets and fixed folds are incomplete")
    probabilities = {
        role: np.asarray(outputs[role], dtype=np.float64) for role in ROLES
    }
    metrics = {
        role: classification_metrics(targets, probabilities[role]) for role in ROLES
    }
    transitions = {
        "candidate_vs_natural": transition_stats(
            targets,
            probabilities["natural_control"],
            probabilities["cap_candidate"],
        ),
        "candidate_vs_duplicate": transition_stats(
            targets,
            probabilities["duplicate_control"],
            probabilities["cap_candidate"],
        ),
        "repeat_vs_duplicate": transition_stats(
            targets,
            probabilities["duplicate_control"],
            probabilities["cap_seed_repeat"],
        ),
    }

    fold_rows: List[Dict[str, object]] = []
    precision_nonworse_folds = 0
    positive_fp_net_folds = 0
    fold_tp_safety = True
    for fold in range(FOLDS):
        mask = folds == fold
        fold_metrics = {
            role: classification_metrics(targets[mask], probabilities[role][mask])
            for role in ROLES
        }
        fold_transition = transition_stats(
            targets[mask],
            probabilities["duplicate_control"][mask],
            probabilities["cap_candidate"][mask],
        )
        candidate_focus = _focus(fold_metrics["cap_candidate"])
        duplicate_focus = _focus(fold_metrics["duplicate_control"])
        precision_nonworse = float(candidate_focus["precision"]) >= float(
            duplicate_focus["precision"]
        )
        positive_fp_net = int(fold_transition["restricted_fp_net_removal"]) > 0
        tp_safe = int(fold_transition["focus_tp_net"]) >= -1
        precision_nonworse_folds += int(precision_nonworse)
        positive_fp_net_folds += int(positive_fp_net)
        fold_tp_safety = fold_tp_safety and tp_safe
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "metrics": fold_metrics,
                "candidate_vs_duplicate": fold_transition,
                "candidate_precision_nonworse": precision_nonworse,
                "candidate_positive_restricted_fp_net_removal": positive_fp_net,
                "candidate_focus_tp_net_ge_minus1": tp_safe,
            }
        )

    natural = metrics["natural_control"]
    duplicate = metrics["duplicate_control"]
    candidate = metrics["cap_candidate"]
    repeat = metrics["cap_seed_repeat"]
    deranged = metrics["nearest_center_deranged"]
    natural_focus = _focus(natural)
    duplicate_focus = _focus(duplicate)
    candidate_focus = _focus(candidate)
    repeat_focus = _focus(repeat)
    deranged_focus = _focus(deranged)
    candidate_transition = transitions["candidate_vs_duplicate"]
    repeat_transition = transitions["repeat_vs_duplicate"]

    nonfocus_losses = {
        str(class_index): float(duplicate["per_class"][class_index]["f1"])
        - float(candidate["per_class"][class_index]["f1"])
        for class_index in range(CLASS_COUNT)
        if class_index != FOCUS_CLASS
    }
    primary_predictions = probabilities["cap_candidate"].argmax(axis=1)
    repeat_predictions = probabilities["cap_seed_repeat"].argmax(axis=1)
    seed_stability = {
        "prediction_agreement": float(np.mean(primary_predictions == repeat_predictions)),
        "maximum_probability_difference": float(
            np.max(
                np.abs(
                    probabilities["cap_candidate"]
                    - probabilities["cap_seed_repeat"]
                )
            )
        ),
        "macro_f1_absolute_difference": abs(
            float(candidate["macro_f1"]) - float(repeat["macro_f1"])
        ),
        "focus_f1_absolute_difference": abs(
            float(candidate_focus["f1"]) - float(repeat_focus["f1"])
        ),
    }

    def aggregate_safety(
        role_metrics: Mapping[str, object],
        role_transition: Mapping[str, int],
    ) -> Dict[str, bool]:
        focus = _focus(role_metrics)
        return {
            "focus_f1_ge_070": float(focus["f1"]) >= 0.70,
            "focus_precision_ge_duplicate_plus0015": float(focus["precision"])
            >= float(duplicate_focus["precision"]) + 0.015,
            "focus_precision_ge_natural_minus0005": float(focus["precision"])
            >= float(natural_focus["precision"]) - 0.005,
            "focus_recall_ge_075": float(focus["recall"]) >= 0.75,
            "focus_recall_ge_duplicate_minus0015": float(focus["recall"])
            >= float(duplicate_focus["recall"]) - 0.015,
            "restricted_fp_net_removal_ge_10": int(
                role_transition["restricted_fp_net_removal"]
            )
            >= 10,
            "focus_tp_net_ge_minus2": int(role_transition["focus_tp_net"]) >= -2,
        }

    primary_safety = aggregate_safety(candidate, candidate_transition)
    repeat_safety = aggregate_safety(repeat, repeat_transition)
    gates = {
        "macro_f1_ge_duplicate_plus0002": float(candidate["macro_f1"])
        >= float(duplicate["macro_f1"]) + 0.002,
        "macro_f1_ge_natural_minus0002": float(candidate["macro_f1"])
        >= float(natural["macro_f1"]) - 0.002,
        "focus_f1_ge_070": float(candidate_focus["f1"]) >= 0.70,
        "focus_f1_ge_duplicate_plus0010": float(candidate_focus["f1"])
        >= float(duplicate_focus["f1"]) + 0.010,
        "focus_precision_ge_duplicate_plus0015": float(candidate_focus["precision"])
        >= float(duplicate_focus["precision"]) + 0.015,
        "focus_precision_ge_natural_minus0005": float(candidate_focus["precision"])
        >= float(natural_focus["precision"]) - 0.005,
        "focus_recall_ge_075": float(candidate_focus["recall"]) >= 0.75,
        "focus_recall_ge_duplicate_minus0015": float(candidate_focus["recall"])
        >= float(duplicate_focus["recall"]) - 0.015,
        "restricted_fp_net_removal_ge_10": int(
            candidate_transition["restricted_fp_net_removal"]
        )
        >= 10,
        "focus_tp_net_ge_minus2": int(candidate_transition["focus_tp_net"]) >= -2,
        "corrections_ge_harms": int(candidate_transition["candidate_correction"])
        >= int(candidate_transition["candidate_harm"]),
        "focus_tp_break_le_fn_rescue": int(candidate_transition["focus_tp_break"])
        <= int(candidate_transition["focus_fn_rescue"]),
        "maximum_nonfocus_f1_loss_le_0010": max(nonfocus_losses.values()) <= 0.010,
        "precision_nonworse_in_at_least_4_folds": precision_nonworse_folds >= 4,
        "positive_fp_net_removal_in_at_least_4_folds": positive_fp_net_folds >= 4,
        "all_fold_focus_tp_net_ge_minus1": fold_tp_safety,
        "focus_f1_ge_deranged_plus0020": float(candidate_focus["f1"])
        >= float(deranged_focus["f1"]) + 0.020,
        "seed_prediction_agreement_ge_099": float(
            seed_stability["prediction_agreement"]
        )
        >= 0.99,
        "seed_macro_f1_difference_le_0005": float(
            seed_stability["macro_f1_absolute_difference"]
        )
        <= 0.005,
        "seed_focus_f1_difference_le_0005": float(
            seed_stability["focus_f1_absolute_difference"]
        )
        <= 0.005,
        "primary_aggregate_safety_passed": all(primary_safety.values()),
        "repeat_aggregate_safety_passed": all(repeat_safety.values()),
    }
    return {
        "metrics": metrics,
        "transitions": transitions,
        "folds": fold_rows,
        "fold_gate_counts": {
            "precision_nonworse_folds": precision_nonworse_folds,
            "positive_restricted_fp_net_removal_folds": positive_fp_net_folds,
        },
        "nonfocus_f1_loss_vs_duplicate": nonfocus_losses,
        "seed_stability": seed_stability,
        "primary_aggregate_safety": primary_safety,
        "repeat_aggregate_safety": repeat_safety,
        "mechanism_gates": gates,
        "mechanism_gates_passed": all(gates.values()),
    }


def _prediction_rows(
    *,
    cache: Mapping[str, np.ndarray],
    folds: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    targets = np.asarray(cache["labels"], dtype=np.int64)
    duplicate_predictions = outputs["duplicate_control"].argmax(axis=1)
    candidate_predictions = outputs["cap_candidate"].argmax(axis=1)
    restricted = np.isin(targets, RESTRICTED_RIVALS)
    for index in range(targets.size):
        row: Dict[str, object] = {
            "sample_index": int(cache["sample_index"][index]),
            "source_stem": str(cache["source_stems"][index]),
            "image_path": str(cache["paths"][index]),
            "fold": int(folds[index]),
            "target_index": int(targets[index]),
            "duplicate_prediction": int(duplicate_predictions[index]),
            "candidate_prediction": int(candidate_predictions[index]),
            "candidate_correction": bool(
                duplicate_predictions[index] != targets[index]
                and candidate_predictions[index] == targets[index]
            ),
            "candidate_harm": bool(
                duplicate_predictions[index] == targets[index]
                and candidate_predictions[index] != targets[index]
            ),
            "focus_fn_rescue": bool(
                targets[index] == FOCUS_CLASS
                and duplicate_predictions[index] != FOCUS_CLASS
                and candidate_predictions[index] == FOCUS_CLASS
            ),
            "focus_tp_break": bool(
                targets[index] == FOCUS_CLASS
                and duplicate_predictions[index] == FOCUS_CLASS
                and candidate_predictions[index] != FOCUS_CLASS
            ),
            "restricted_fp_remove": bool(
                restricted[index]
                and duplicate_predictions[index] == FOCUS_CLASS
                and candidate_predictions[index] != FOCUS_CLASS
            ),
            "restricted_fp_create": bool(
                restricted[index]
                and duplicate_predictions[index] != FOCUS_CLASS
                and candidate_predictions[index] == FOCUS_CLASS
            ),
        }
        for role in ROLES:
            row[f"{role}_prediction"] = int(outputs[role][index].argmax())
            for class_index in range(CLASS_COUNT):
                row[f"{role}_prob_{class_index}"] = float(
                    outputs[role][index, class_index]
                )
        row["candidate_minus_duplicate_focus_probability"] = float(
            outputs["cap_candidate"][index, FOCUS_CLASS]
            - outputs["duplicate_control"][index, FOCUS_CLASS]
        )
        rows.append(row)
    return rows


def _fold_metric_rows(analysis: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for fold_record in analysis["folds"]:
        for role in ROLES:
            metrics = fold_record["metrics"][role]
            focus = _focus(metrics)
            rows.append(
                {
                    "fold": int(fold_record["fold"]),
                    "role": role,
                    "rows": int(metrics["samples"]),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "focus_precision": float(focus["precision"]),
                    "focus_recall": float(focus["recall"]),
                    "focus_f1": float(focus["f1"]),
                }
            )
    return rows


def _artifact_record(path: Path, *, root: Path) -> Dict[str, object]:
    return {
        "name": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest = {
        "mode": "hyperspherical_support_a0_artifact_manifest",
        "payload_count": len(files),
        "payload_size_bytes": int(sum(path.stat().st_size for path in files)),
        "contains_checkpoint_payload": False,
        "contains_model_binary": False,
        "contains_validation_or_test_payload": False,
        "raw_dataset_touched": False,
        "files": [_artifact_record(path, root=output_dir) for path in files],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(item["name"]): item for item in manifest["files"]}
    observed_paths = {
        path.relative_to(output_dir).as_posix(): path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if set(expected) != set(observed_paths):
        raise ValueError("Artifact manifest payload set differs from disk")
    for name, path in observed_paths.items():
        record = expected[name]
        if int(record["size_bytes"]) != int(path.stat().st_size):
            raise ValueError(f"Artifact size mismatch: {name}")
        observed_hash = _sha256(path)
        if str(record["sha256"]) != observed_hash:
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")
    return {
        "verified": True,
        "payload_count": len(observed_paths),
        "manifest_sha256": _sha256(manifest_path),
    }


def verify_incomplete_payloads(
    output_dir: Path,
    expected: Mapping[str, Mapping[str, object]] = INCOMPLETE_PAYLOADS,
) -> Dict[str, object]:
    resolved = Path(output_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Incomplete output directory is missing: {resolved}")
    if (resolved / "summary.json").exists() or (
        resolved / "artifact_manifest.json"
    ).exists():
        raise FileExistsError(
            "Incomplete finalization requires both summary and manifest to be absent"
        )
    observed = {
        path.name: path for path in sorted(resolved.iterdir()) if path.is_file()
    }
    if set(observed) != set(expected):
        raise ValueError(
            "Incomplete payload set differs from the locked erratum: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    records: Dict[str, object] = {}
    for name, lock in expected.items():
        path = observed[name]
        size = int(path.stat().st_size)
        digest = _sha256(path)
        if size != int(lock["size_bytes"]):
            raise ValueError(f"Incomplete payload size mismatch: {name}")
        if digest != str(lock["sha256"]):
            raise ValueError(f"Incomplete payload SHA-256 mismatch: {name}")
        records[name] = {
            "size_bytes": size,
            "sha256": digest,
        }
    return {
        "verified": True,
        "payload_count": len(records),
        "files": records,
    }


def load_prediction_payload(
    path: Path,
    *,
    cache: Mapping[str, np.ndarray],
    expected_folds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    expected_rows = int(np.asarray(cache["labels"]).shape[0])
    targets = np.full(expected_rows, -1, dtype=np.int64)
    folds = np.full(expected_rows, -1, dtype=np.int64)
    outputs = {
        role: np.full((expected_rows, CLASS_COUNT), np.nan, dtype=np.float64)
        for role in ROLES
    }
    observed = np.zeros(expected_rows, dtype=bool)
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample_index = int(row["sample_index"])
            if sample_index < 0 or sample_index >= expected_rows or observed[sample_index]:
                raise ValueError(f"Prediction sample index is invalid: {sample_index}")
            if str(row["source_stem"]).casefold() != str(
                cache["source_stems"][sample_index]
            ).casefold():
                raise ValueError(f"Prediction/cache source mismatch at {sample_index}")
            if Path(str(row["image_path"])).name.casefold() != Path(
                str(cache["paths"][sample_index])
            ).name.casefold():
                raise ValueError(f"Prediction/cache path mismatch at {sample_index}")
            target = int(row["target_index"])
            fold = int(row["fold"])
            if target != int(cache["labels"][sample_index]):
                raise ValueError(f"Prediction/cache target mismatch at {sample_index}")
            if fold != int(expected_folds[sample_index]):
                raise ValueError(f"Prediction/CIDT fold mismatch at {sample_index}")
            targets[sample_index] = target
            folds[sample_index] = fold
            for role in ROLES:
                outputs[role][sample_index] = [
                    float(row[f"{role}_prob_{class_index}"])
                    for class_index in range(CLASS_COUNT)
                ]
            observed[sample_index] = True
    if not observed.all():
        raise ValueError(
            f"Prediction payload is incomplete: {int(observed.sum())}/{expected_rows}"
        )
    if any(not np.isfinite(outputs[role]).all() for role in ROLES):
        raise ValueError("Prediction payload contains non-finite role probabilities")
    return targets, folds, outputs


def _recursive_difference(left: object, right: object, path: str = "root") -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(f"Replay mapping keys differ at {path}")
        return max(
            (
                _recursive_difference(left[key], right[key], f"{path}.{key}")
                for key in left
            ),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"Replay sequence lengths differ at {path}")
        return max(
            (
                _recursive_difference(a, b, f"{path}[{index}]")
                for index, (a, b) in enumerate(zip(left, right))
            ),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise ValueError(f"Replay boolean differs at {path}: {left} != {right}")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        if not math.isfinite(difference):
            raise ValueError(f"Replay numeric value is non-finite at {path}")
        return difference
    if left != right:
        raise ValueError(f"Replay value differs at {path}: {left!r} != {right!r}")
    return 0.0


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    output_dir = resolved.parent
    manifest = _verify_manifest(output_dir)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    predictions_path = output_dir / "predictions.csv"
    targets: List[int] = []
    folds: List[int] = []
    outputs: Dict[str, List[List[float]]] = {role: [] for role in ROLES}
    with predictions_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            targets.append(int(row["target_index"]))
            folds.append(int(row["fold"]))
            for role in ROLES:
                outputs[role].append(
                    [float(row[f"{role}_prob_{index}"]) for index in range(CLASS_COUNT)]
                )
    replayed = analyze_predictions(
        np.asarray(targets, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
        {role: np.asarray(values, dtype=np.float64) for role, values in outputs.items()},
    )
    maximum_difference = _recursive_difference(summary["analysis"], replayed)
    if maximum_difference > 1e-12:
        raise ValueError(
            f"Replay numeric difference exceeds 1e-12: {maximum_difference}"
        )
    return {
        "replay_passed": True,
        "rows": len(targets),
        "maximum_numeric_difference": maximum_difference,
        "manifest": manifest,
        "summary_sha256": _sha256(resolved),
    }


def _run_readouts(
    cache: Mapping[str, np.ndarray], folds: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, object], List[Dict[str, object]]]:
    labels = np.asarray(cache["labels"], dtype=np.int64)
    normalized = l2_normalize(cache["embeddings"])
    outputs = {
        role: np.full((EXPECTED_ROWS, CLASS_COUNT), np.nan, dtype=np.float64)
        for role in ROLES
    }
    fold_protocols: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    global_rng_state_before = np.random.get_state()

    for fold in range(FOLDS):
        fit_indices = np.flatnonzero(folds != fold)
        hold_indices = np.flatnonzero(folds == fold)
        fit_features = normalized[fit_indices]
        fit_labels = labels[fit_indices]
        hold_features = normalized[hold_indices]

        duplicate_features, duplicate_labels, duplicate_record = duplicate_balance(
            fit_features, fit_labels, seed=SEED + 1009 * fold
        )
        candidate_features, candidate_labels, candidate_geometry = _cap_balance(
            fit_features, fit_labels, fold=fold
        )
        repeat_features, repeat_labels, repeat_geometry = _cap_balance(
            fit_features, fit_labels, fold=fold, seed_offset=7919
        )
        deranged_features, deranged_labels, deranged_geometry = _cap_balance(
            fit_features, fit_labels, fold=fold, deranged=True
        )

        role_inputs = {
            "natural_control": (fit_features, fit_labels),
            "duplicate_control": (duplicate_features, duplicate_labels),
            "cap_candidate": (candidate_features, candidate_labels),
            "cap_seed_repeat": (repeat_features, repeat_labels),
            "nearest_center_deranged": (deranged_features, deranged_labels),
        }
        readouts: Dict[str, object] = {}
        for role in ROLES:
            probabilities, record = _fit_readout(
                role_inputs[role][0], role_inputs[role][1], hold_features
            )
            outputs[role][hold_indices] = probabilities
            record["fit_rows"] = int(role_inputs[role][1].size)
            record["holdout_rows"] = int(hold_indices.size)
            readouts[role] = record

        for role, records in (
            ("cap_candidate", candidate_geometry),
            ("cap_seed_repeat", repeat_geometry),
            ("nearest_center_deranged", deranged_geometry),
        ):
            for record in records:
                geometry_rows.append({"role": role, **record})
        fold_protocols.append(
            {
                "fold": fold,
                "fit_indices_sha256": _array_sha256(fit_indices.astype(np.int64)),
                "holdout_indices_sha256": _array_sha256(
                    hold_indices.astype(np.int64)
                ),
                "fit_class_counts": np.bincount(
                    fit_labels, minlength=CLASS_COUNT
                ).tolist(),
                "duplicate_balance": duplicate_record,
                "readouts": readouts,
            }
        )

    global_rng_state_after = np.random.get_state()
    global_rng_unchanged = all(
        np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        for left, right in zip(global_rng_state_before, global_rng_state_after)
    )
    if any(not np.isfinite(outputs[role]).all() for role in ROLES):
        raise RuntimeError("A readout did not cover every source-held row")
    return outputs, {
        "folds": fold_protocols,
        "global_numpy_rng_unchanged": global_rng_unchanged,
    }, geometry_rows


def _geometry_structural_gates(
    geometry_rows: Sequence[Mapping[str, object]], fold_protocol: Mapping[str, object]
) -> Dict[str, bool]:
    expected_records = FOLDS * CLASS_COUNT * 3
    converged = all(
        bool(readout["converged"])
        for fold in fold_protocol["folds"]
        for readout in fold["readouts"].values()
    )
    return {
        "geometry_record_count_exact": len(geometry_rows) == expected_records,
        "generated_norm_error_le_1e10": max(
            float(row["maximum_norm_error"]) for row in geometry_rows
        )
        <= 1e-10,
        "generated_cap_angle_excess_le_1e10": max(
            float(row["maximum_cap_angle_excess"]) for row in geometry_rows
        )
        <= 1e-10,
        "generated_tangent_abs_dot_le_1e10": max(
            float(row["maximum_tangent_abs_dot"]) for row in geometry_rows
        )
        <= 1e-10,
        "angle_reconstruction_error_le_1e10": max(
            float(row["maximum_angle_reconstruction_error"])
            for row in geometry_rows
        )
        <= 1e-10,
        "balanced_target_counts_exact": all(
            int(row["real_count"]) + int(row["generated_count"])
            == int(row["target_count"])
            for row in geometry_rows
        ),
        "global_numpy_rng_unchanged": bool(
            fold_protocol["global_numpy_rng_unchanged"]
        ),
        "all_readouts_converged": converged,
    }


def finalize_incomplete(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    locked = verify_locked_inputs()
    preflight = verify_preflight_checks()
    output_dir = Path(args.output_dir).expanduser().resolve()
    incomplete = verify_incomplete_payloads(output_dir)

    cache = load_locked_train_cache()
    cidt_folds, fold_assignment = load_locked_cidt_folds(cache)
    stored_fold_assignment = json.loads(
        (output_dir / "fold_assignment.json").read_text(encoding="utf-8")
    )
    fold_assignment_difference = _recursive_difference(
        stored_fold_assignment, _jsonable(fold_assignment)
    )
    if fold_assignment_difference > 0.0:
        raise ValueError("Stored fold assignment differs from the locked CIDT replay")

    targets, prediction_folds, outputs = load_prediction_payload(
        output_dir / "predictions.csv",
        cache=cache,
        expected_folds=cidt_folds,
    )
    if not np.array_equal(targets, cache["labels"]):
        raise ValueError("Finalizer prediction targets differ from the locked cache")
    if not np.array_equal(prediction_folds, cidt_folds):
        raise ValueError("Finalizer prediction folds differ from the locked CIDT folds")

    fold_protocol = json.loads(
        (output_dir / "fold_protocol.json").read_text(encoding="utf-8")
    )
    geometry_payload = json.loads(
        (output_dir / "synthetic_geometry.json").read_text(encoding="utf-8")
    )
    geometry_rows = geometry_payload["rows"]
    geometry_gates = _geometry_structural_gates(geometry_rows, fold_protocol)
    geometry_gate_difference = _recursive_difference(
        geometry_payload["structural_gates"], geometry_gates
    )
    if geometry_gate_difference > 0.0:
        raise ValueError("Stored geometry gates differ from the locked replay")

    analysis = analyze_predictions(targets, prediction_folds, outputs)
    normalized_keeper, keeper_normalization = normalize_keeper_reference(
        cache["probabilities"]
    )
    structural_gates = {
        "locked_inputs_verified": True,
        "repo_clean_pushed_and_protocol_ancestor": True,
        "incomplete_payload_set_and_hashes_exact": bool(incomplete["verified"]),
        "train_rows_exact": int(cache["labels"].size) == EXPECTED_ROWS,
        "class_counts_exact": tuple(
            np.bincount(cache["labels"], minlength=CLASS_COUNT).tolist()
        )
        == EXPECTED_CLASS_COUNTS,
        "fold_counts_exact": tuple(
            np.bincount(cidt_folds, minlength=FOLDS).tolist()
        )
        == EXPECTED_FOLD_COUNTS,
        "source_overlap_zero": int(fold_assignment["source_overlap"]) == 0,
        "stored_fold_assignment_exact": fold_assignment_difference == 0.0,
        "stored_geometry_gates_exact": geometry_gate_difference == 0.0,
        "all_outputs_cover_all_rows": all(
            outputs[role].shape == (EXPECTED_ROWS, CLASS_COUNT)
            and np.isfinite(outputs[role]).all()
            for role in ROLES
        ),
        "preflight_checks_passed": all(
            bool(record["passed"]) for record in preflight.values()
        ),
        "validation_test_unopened": True,
        **geometry_gates,
    }
    structural_passed = all(structural_gates.values())
    passed = structural_passed and bool(analysis["mechanism_gates_passed"])
    finalization_seconds = time.perf_counter() - started

    with threadpool_limits(limits=int(args.blas_threads)):
        effective_threadpools = threadpool_info()
    summary = {
        "mode": "hyperspherical_support_a0",
        "status": "pass" if passed else "fail",
        "locked_inputs": locked,
        "protocol": {
            "seed": SEED,
            "folds": FOLDS,
            "alpha": 0.99,
            "logistic_c": 0.3,
            "logistic_max_iter": 2000,
            "logistic_tolerance": 1e-9,
            "class_weight": None,
            "roles": list(ROLES),
            "blas_threads_requested": int(args.blas_threads),
            "paper_scope": "independent Eq. 7/10/14-16 empirical-cap information gate",
            "complete_featrecon_reproduction": False,
            "dependency_versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "effective_threadpools_during_finalization": effective_threadpools,
        },
        "input_summary": {
            "rows": EXPECTED_ROWS,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "embedding_array_sha256": _array_sha256(cache["embeddings"]),
            "label_array_sha256": _array_sha256(cache["labels"]),
            "fold_array_sha256": _array_sha256(cidt_folds),
        },
        "analysis": analysis,
        "keeper_in_sample_reference": {
            "normalization": keeper_normalization,
            "metrics": classification_metrics(cache["labels"], normalized_keeper),
        },
        "structural_gates": structural_gates,
        "structural_gates_passed": structural_passed,
        "all_gates_passed": passed,
        "runtime": {
            "formal_compute_seconds": None,
            "formal_compute_peak_rss_bytes": None,
            "formal_compute_unavailable_reason": (
                "The reporting exception occurred before summary serialization; "
                "the missing values were not inferred."
            ),
            "incomplete_finalization_seconds": finalization_seconds,
        },
        "incomplete_finalization": {
            "erratum_path": str(ERRATUM_PATH.resolve()),
            "erratum_sha256": _sha256(ERRATUM_PATH),
            "second_readout_run_performed": False,
            "support_generation_called": False,
            "logistic_fitting_called": False,
            "locked_pre_summary_payloads": incomplete,
            "pre_summary_payloads_rewritten": False,
            "keeper_reference_only_normalized": True,
        },
        "preflight": preflight,
        "authorization": {
            "default_off_representation_smoke_authorized": passed,
            "validation_authorized": False,
            "test_authorized": False,
            "probe_authorized": False,
            "full_train_authorized": False,
            "current_best_update_authorized": False,
        },
        "raw_dataset_touched": False,
        "validation_test_opened": False,
        "note": (
            "Train-only frozen-keeper source-held readout evidence finalized from "
            "the exact locked interrupted payloads. Embeddings are not encoder-OOF "
            "and the audit is not a generalization estimate."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = _write_manifest(output_dir)
    verified_manifest = _verify_manifest(output_dir)
    return {
        "status": summary["status"],
        "all_gates_passed": passed,
        "analysis": analysis,
        "structural_gates": structural_gates,
        "authorization": summary["authorization"],
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "artifact_manifest_sha256": verified_manifest["manifest_sha256"],
        "artifact_payload_count": manifest["payload_count"],
        "second_readout_run_performed": False,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    active_modes = sum(
        bool(value)
        for value in (
            args.preflight_only,
            args.finalize_incomplete,
            args.replay_summary is not None,
        )
    )
    if active_modes > 1:
        raise ValueError(
            "Choose only one of preflight, incomplete finalization, or replay"
        )
    if args.replay_summary is not None:
        return replay_summary(args.replay_summary)
    if args.finalize_incomplete:
        return finalize_incomplete(args)
    output_path = Path(args.output_dir)
    if args.preflight_only:
        _require_absent_output(output_path)
        locked = verify_locked_inputs()
        return {
            "preflight_passed": True,
            "output_created": False,
            "locked_inputs": locked,
            "validation_test_opened": False,
        }

    locked = verify_locked_inputs()
    preflight = verify_preflight_checks()
    output_dir = _require_empty_output(output_path)
    cache = load_locked_train_cache()
    folds, fold_assignment = load_locked_cidt_folds(cache)

    started = time.perf_counter()
    with threadpool_limits(limits=int(args.blas_threads)):
        effective_threadpools = threadpool_info()
        with _PeakRssMonitor() as memory:
            outputs, fold_protocol, geometry_rows = _run_readouts(cache, folds)
            analysis = analyze_predictions(cache["labels"], folds, outputs)
    runtime_seconds = time.perf_counter() - started

    geometry_gates = _geometry_structural_gates(geometry_rows, fold_protocol)
    structural_gates = {
        "locked_inputs_verified": True,
        "repo_clean_pushed_and_protocol_ancestor": True,
        "train_rows_exact": int(cache["labels"].size) == EXPECTED_ROWS,
        "class_counts_exact": tuple(
            np.bincount(cache["labels"], minlength=CLASS_COUNT).tolist()
        )
        == EXPECTED_CLASS_COUNTS,
        "fold_counts_exact": tuple(
            np.bincount(folds, minlength=FOLDS).tolist()
        )
        == EXPECTED_FOLD_COUNTS,
        "source_overlap_zero": int(fold_assignment["source_overlap"]) == 0,
        "all_outputs_cover_all_rows": all(
            outputs[role].shape == (EXPECTED_ROWS, CLASS_COUNT)
            and np.isfinite(outputs[role]).all()
            for role in ROLES
        ),
        "preflight_checks_passed": all(
            bool(record["passed"]) for record in preflight.values()
        ),
        "validation_test_unopened": True,
        **geometry_gates,
    }
    structural_passed = all(structural_gates.values())
    passed = structural_passed and bool(analysis["mechanism_gates_passed"])

    _write_csv(
        output_dir / "predictions.csv",
        _prediction_rows(cache=cache, folds=folds, outputs=outputs),
    )
    _write_csv(output_dir / "fold_metrics.csv", _fold_metric_rows(analysis))
    _write_csv(output_dir / "synthetic_geometry.csv", geometry_rows)
    _write_json(output_dir / "fold_assignment.json", fold_assignment)
    _write_json(output_dir / "fold_protocol.json", fold_protocol)
    _write_json(
        output_dir / "synthetic_geometry.json",
        {
            "rows": geometry_rows,
            "structural_gates": geometry_gates,
            "minimum_cap_separation_margin": float(
                min(float(row["minimum_cap_separation_margin"]) for row in geometry_rows)
            ),
        },
    )
    normalized_keeper, keeper_normalization = normalize_keeper_reference(
        cache["probabilities"]
    )
    summary = {
        "mode": "hyperspherical_support_a0",
        "status": "pass" if passed else "fail",
        "locked_inputs": locked,
        "protocol": {
            "seed": SEED,
            "folds": FOLDS,
            "alpha": 0.99,
            "logistic_c": 0.3,
            "logistic_max_iter": 2000,
            "logistic_tolerance": 1e-9,
            "class_weight": None,
            "roles": list(ROLES),
            "blas_threads_requested": int(args.blas_threads),
            "paper_scope": "independent Eq. 7/10/14-16 empirical-cap information gate",
            "complete_featrecon_reproduction": False,
            "dependency_versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "effective_threadpools": effective_threadpools,
        },
        "input_summary": {
            "rows": EXPECTED_ROWS,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "embedding_array_sha256": _array_sha256(cache["embeddings"]),
            "label_array_sha256": _array_sha256(cache["labels"]),
            "fold_array_sha256": _array_sha256(folds),
        },
        "analysis": analysis,
        "keeper_in_sample_reference": {
            "normalization": keeper_normalization,
            "metrics": classification_metrics(cache["labels"], normalized_keeper),
        },
        "structural_gates": structural_gates,
        "structural_gates_passed": structural_passed,
        "all_gates_passed": passed,
        "runtime": {
            "seconds": runtime_seconds,
            "peak_rss_bytes": int(memory.peak_bytes),
            "peak_rss_gib": float(memory.peak_bytes / (1024**3)),
        },
        "preflight": preflight,
        "authorization": {
            "default_off_representation_smoke_authorized": passed,
            "validation_authorized": False,
            "test_authorized": False,
            "probe_authorized": False,
            "full_train_authorized": False,
            "current_best_update_authorized": False,
        },
        "raw_dataset_touched": False,
        "validation_test_opened": False,
        "note": (
            "Train-only frozen-keeper source-held readout evidence. Embeddings are "
            "not encoder-OOF and the audit is not a generalization estimate."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = _write_manifest(output_dir)
    verified_manifest = _verify_manifest(output_dir)
    return {
        "status": summary["status"],
        "all_gates_passed": passed,
        "analysis": analysis,
        "structural_gates": structural_gates,
        "authorization": summary["authorization"],
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "artifact_manifest_sha256": verified_manifest["manifest_sha256"],
        "artifact_payload_count": manifest["payload_count"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_audit(parse_args(argv))
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

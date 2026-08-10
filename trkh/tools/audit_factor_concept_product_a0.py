from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from trkh.core.config import to_serializable


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_SOURCE_GROUPS = 8064
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_CLASSES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
EXPECTED_CACHE_KEYS = (
    "head",
    "all_second_order",
    "core_first_order",
    "core_second_order",
    "ring_second_order",
    "context_second_order",
    "head_core_second_order",
    "probabilities",
    "labels",
    "sample_index",
    "paths",
    "source_stem",
    "classes",
)
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
MATURITY_CODE = np.asarray((0, 0, 1, 1, 2), dtype=np.int64)
TRANSPORT_CODE = np.asarray((0, 1, 0, 2, 2), dtype=np.int64)
PROTOCOL_SHA256 = "bf3cb5acb7f05a25f57f3a5627184bc2abe5b04e315f1fa958be09c9ef09b78b"
CONCEPT_REPO_COMMIT = "d6353f270702b92feb5b084a6fd065f891d583f8"
CONCEPT_REPO_TREE = "d93ca72552f8d6e405a8c44d8b9bdf70d9af4eb9"
FOLDS = 5
SEED = 20260720
LOGISTIC_C = 0.3
LOGISTIC_MAX_ITER = 1000
BLAS_THREADS = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = (
    REPO_ROOT
    / "runs"
    / "diagnostic_interior_secondorder_rank24_erode020_full_train_val_20260711"
    / "train_interior_second_order_descriptors.npz"
)
LOCKED_FILES: Tuple[Tuple[str, Path, str], ...] = (
    (
        "protocol",
        REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_FACTOR_CONCEPT_PRODUCT_A0_PROTOCOL_20260720.md",
        PROTOCOL_SHA256,
    ),
    (
        "train_descriptor_cache",
        CACHE_PATH,
        "917bdbd5cd83861f458df551f5aa9dbba4f88a13eacb8282e1769427a16c1c4e",
    ),
    (
        "parent_summary",
        CACHE_PATH.parent / "summary.json",
        "71437ec00935dbd64c77d287d13a67417afc4d7fbff816e6c373657401cfaaf7",
    ),
    (
        "keeper",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "checkpoints"
        / "best.pt",
        "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    ),
    (
        "data_yaml",
        Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
        "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    ),
    (
        "current_best_commands",
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    ),
    (
        "current_best_history",
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
        "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    ),
    (
        "paper",
        Path(r"D:\DataAI\external_sources\papers\concept_bottleneck_icml2020.pdf"),
        "250cf7d6fe86575c9a3100983441134e7f407e32d064a166d674b852a48e9b4b",
    ),
    (
        "concept_repo_license",
        Path(
            r"D:\DataAI\external_sources\official\concept-bottleneck-icml2020\LICENSE"
        ),
        "dbd16fda64f9c246a6f33a9aa449c9313806f1cfa26329c1ce51d703fcc4ccde",
    ),
    (
        "concept_repo_model",
        Path(
            r"D:\DataAI\external_sources\official\concept-bottleneck-icml2020\CUB\template_model.py"
        ),
        "fa9939dde51fc5e6eec1cc5aea6ac357cf8d92e3eff1c388b4eeee3316e76d84",
    ),
    (
        "concept_repo_train",
        Path(
            r"D:\DataAI\external_sources\official\concept-bottleneck-icml2020\CUB\train.py"
        ),
        "ad71ce78fa318e58358e4c4bf661baad0b17872b5a994db6b886fd41522d1f26",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only factor-concept product information gate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_factor_concept_product_a0_20260720",
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--blas-threads", type=int, default=BLAS_THREADS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _git_value(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_locked_inputs() -> Dict[str, object]:
    files: Dict[str, object] = {}
    for label, path, expected in LOCKED_FILES:
        resolved = path.expanduser().resolve()
        files[label] = {
            "path": str(resolved),
            "sha256": _verify_hash(resolved, expected, label),
        }

    concept_repo = Path(
        r"D:\DataAI\external_sources\official\concept-bottleneck-icml2020"
    ).resolve()
    commit = _git_value(concept_repo, "rev-parse", "HEAD")
    tree = _git_value(concept_repo, "rev-parse", "HEAD^{tree}")
    status = _git_value(concept_repo, "status", "--porcelain")
    if commit != CONCEPT_REPO_COMMIT or tree != CONCEPT_REPO_TREE or status:
        raise ValueError(
            "Official ConceptBottleneck lock mismatch: "
            f"commit={commit}, tree={tree}, dirty={bool(status)}"
        )
    return {
        "files": files,
        "concept_bottleneck_repo": {
            "path": str(concept_repo),
            "commit": commit,
            "tree": tree,
            "tracked_and_untracked_worktree_clean": True,
        },
    }


def _repo_state() -> Dict[str, object]:
    head = _git_value(REPO_ROOT, "rev-parse", "HEAD")
    upstream = _git_value(REPO_ROOT, "rev-parse", "@{upstream}")
    tracked_status = _git_value(
        REPO_ROOT, "status", "--porcelain", "--untracked-files=no"
    )
    return {
        "head": head,
        "upstream": upstream,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": not bool(tracked_status),
        "tracked_status": tracked_status,
    }


def verify_preflight_checks() -> Dict[str, object]:
    commands = {
        "py_compile": [
            sys.executable,
            "-m",
            "py_compile",
            "trkh/tools/audit_factor_concept_product_a0.py",
            "tests/test_audit_factor_concept_product_a0.py",
        ],
        "pyflakes": [
            sys.executable,
            "-m",
            "pyflakes",
            "trkh/tools/audit_factor_concept_product_a0.py",
            "tests/test_audit_factor_concept_product_a0.py",
        ],
        "focused_tests": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_audit_factor_concept_product_a0.py",
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


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in materialized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _string_sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


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
    parts = [part for part in str(path).replace("/", "\\").casefold().split("\\") if part]
    return any(part in {"val", "valid", "validation", "test"} for part in parts)


def _is_train_image_path(path: str) -> bool:
    parts = [part for part in str(path).replace("/", "\\").casefold().split("\\") if part]
    return not _contains_forbidden_split(path) and any(
        parts[index : index + 2] == ["images", "train"]
        for index in range(max(0, len(parts) - 1))
    )


def _validate_probability_matrix(
    probabilities: np.ndarray, *, rows: int, classes: int, label: str
) -> np.ndarray:
    result = np.asarray(probabilities, dtype=np.float64)
    if result.shape != (int(rows), int(classes)):
        raise ValueError(
            f"{label} shape mismatch: {result.shape} != {(int(rows), int(classes))}"
        )
    if not np.isfinite(result).all() or bool((result < 0.0).any()):
        raise ValueError(f"{label} must be finite and nonnegative")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{label} probabilities are not normalized")
    return result


def load_locked_train_cache(path: Path) -> Dict[str, object]:
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
        "head": (EXPECTED_TRAIN_ROWS, 256),
        "core_second_order": (EXPECTED_TRAIN_ROWS, 324),
        "ring_second_order": (EXPECTED_TRAIN_ROWS, 324),
        "context_second_order": (EXPECTED_TRAIN_ROWS, 324),
        "probabilities": (EXPECTED_TRAIN_ROWS, 5),
        "labels": (EXPECTED_TRAIN_ROWS,),
        "sample_index": (EXPECTED_TRAIN_ROWS,),
        "paths": (EXPECTED_TRAIN_ROWS,),
        "source_stem": (EXPECTED_TRAIN_ROWS,),
        "classes": (5,),
    }
    for key, shape in expected_shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"Cache {key} shape mismatch: {cache[key].shape} != {shape}")

    labels = np.asarray(cache["labels"], dtype=np.int64)
    indices = np.asarray(cache["sample_index"], dtype=np.int64)
    paths = np.asarray(cache["paths"]).astype(str)
    sources = np.asarray(cache["source_stem"]).astype(str)
    classes = tuple(np.asarray(cache["classes"]).astype(str).tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Cache class order mismatch: {classes}")
    if tuple(np.bincount(labels, minlength=5).tolist()) != EXPECTED_CLASS_COUNTS:
        raise ValueError("Cache class counts differ from the prospective lock")
    if not np.array_equal(indices, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)):
        raise ValueError("Cache sample indices are not complete and ordered")
    if len(set(value.casefold() for value in sources.tolist())) != EXPECTED_SOURCE_GROUPS:
        raise ValueError("Cache source-group count differs from the prospective lock")
    if any(not _is_train_image_path(path) for path in paths.tolist()):
        raise ValueError("Cache contains a non-train or forbidden image path")
    if any(
        Path(path).stem.casefold() != source.casefold()
        for path, source in zip(paths.tolist(), sources.tolist())
    ):
        raise ValueError("Cache source stems do not match image paths")

    finite_keys = (
        "head",
        "core_second_order",
        "ring_second_order",
        "context_second_order",
    )
    if any(not np.isfinite(np.asarray(cache[key])).all() for key in finite_keys):
        raise ValueError("Cache descriptors contain non-finite values")
    cache["probabilities"] = _validate_probability_matrix(
        cache["probabilities"], rows=EXPECTED_TRAIN_ROWS, classes=5, label="keeper"
    )
    cache["labels"] = labels
    cache["sample_index"] = indices
    cache["paths"] = paths
    cache["source_stem"] = np.asarray([value.casefold() for value in sources], dtype=str)
    cache["classes"] = classes
    cache["_observed_keys"] = observed_keys
    return cache


def assign_source_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int = FOLDS,
    seed: int = SEED,
) -> Tuple[np.ndarray, Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).astype(str).reshape(-1)
    if labels.size != groups.size:
        raise ValueError("Fold labels and groups do not align")
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    assignments = np.full(labels.size, -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    for fold, (fit_rows, holdout_rows) in enumerate(
        splitter.split(np.zeros(labels.size), labels, groups)
    ):
        fit_sources = set(groups[fit_rows].tolist())
        holdout_sources = set(groups[holdout_rows].tolist())
        overlap = fit_sources.intersection(holdout_sources)
        assignments[holdout_rows] += int(fold) + 1
        fold_rows.append(
            {
                "fold": int(fold),
                "fit_rows": int(fit_rows.size),
                "holdout_rows": int(holdout_rows.size),
                "fit_sources": int(len(fit_sources)),
                "holdout_sources": int(len(holdout_sources)),
                "source_overlap": int(len(overlap)),
                "fit_class_counts": np.bincount(labels[fit_rows], minlength=5).tolist(),
                "holdout_class_counts": np.bincount(
                    labels[holdout_rows], minlength=5
                ).tolist(),
                "fit_indices_sha256": _array_sha256(fit_rows.astype(np.int64)),
                "holdout_indices_sha256": _array_sha256(
                    holdout_rows.astype(np.int64)
                ),
                "fit_sources_sha256": _string_sequence_sha256(sorted(fit_sources)),
                "holdout_sources_sha256": _string_sequence_sha256(
                    sorted(holdout_sources)
                ),
            }
        )
    if bool((assignments < 0).any()) or not np.all(
        np.bincount(assignments, minlength=int(folds)) > 0
    ):
        raise RuntimeError("Source-fold assignment is incomplete")
    return assignments, {
        "fold_count": int(folds),
        "assignment_complete": True,
        "assignment_sha256": _array_sha256(assignments),
        "assignment_counts": np.bincount(assignments, minlength=int(folds)).tolist(),
        "source_overlap": int(sum(int(row["source_overlap"]) for row in fold_rows)),
        "folds": fold_rows,
    }


def factor_product_probabilities(
    maturity_probabilities: np.ndarray, transport_probabilities: np.ndarray
) -> np.ndarray:
    maturity = np.asarray(maturity_probabilities, dtype=np.float64)
    transport = np.asarray(transport_probabilities, dtype=np.float64)
    if maturity.ndim != 2 or maturity.shape[1] != 3 or transport.shape != maturity.shape:
        raise ValueError("Factor probabilities must both have shape [N,3]")
    _validate_probability_matrix(
        maturity, rows=maturity.shape[0], classes=3, label="maturity"
    )
    _validate_probability_matrix(
        transport, rows=transport.shape[0], classes=3, label="transport"
    )
    epsilon = 1e-12
    logits = np.log(np.clip(maturity[:, MATURITY_CODE], epsilon, 1.0))
    logits += np.log(np.clip(transport[:, TRANSPORT_CODE], epsilon, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    unnormalized = np.exp(logits)
    return unnormalized / unnormalized.sum(axis=1, keepdims=True)


def _expand_probabilities(
    classes: np.ndarray, probabilities: np.ndarray, *, class_count: int
) -> np.ndarray:
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != classes.size:
        raise ValueError("Classifier classes and probability columns do not align")
    if bool((classes < 0).any()) or bool((classes >= int(class_count)).any()):
        raise ValueError("Classifier emitted an out-of-range class")
    expanded = np.zeros((probabilities.shape[0], int(class_count)), dtype=np.float64)
    expanded[:, classes] = probabilities
    return expanded


def _fit_readout(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    holdout_features: np.ndarray,
    *,
    class_count: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    model = LogisticRegression(
        C=LOGISTIC_C,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
        class_weight="balanced",
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(fit_features, fit_targets)
    convergence_warnings = [
        str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    probabilities = _expand_probabilities(
        model.classes_, model.predict_proba(holdout_features), class_count=class_count
    )
    iterations = np.asarray(model.n_iter_, dtype=np.int64)
    record = {
        "classes": np.asarray(model.classes_, dtype=np.int64).tolist(),
        "iterations": iterations.tolist(),
        "maximum_iterations": int(iterations.max(initial=0)),
        "convergence_warnings": convergence_warnings,
        "converged": not convergence_warnings
        and int(iterations.max(initial=0)) < LOGISTIC_MAX_ITER,
        "coefficient_shape": list(model.coef_.shape),
        "coefficient_sha256": _array_sha256(model.coef_),
        "intercept_sha256": _array_sha256(model.intercept_),
    }
    return probabilities, record


def _scaler_record(scaler: StandardScaler) -> Dict[str, object]:
    return {
        "features": int(scaler.n_features_in_),
        "mean_sha256": _array_sha256(scaler.mean_),
        "scale_sha256": _array_sha256(scaler.scale_),
        "variance_sha256": _array_sha256(scaler.var_),
    }


def fit_oof_readouts(
    cache: Mapping[str, object], folds: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    labels = np.asarray(cache["labels"], dtype=np.int64)
    groups = np.asarray(cache["source_stem"]).astype(str)
    head = np.asarray(cache["head"], dtype=np.float64)
    core = np.asarray(cache["core_second_order"], dtype=np.float64)
    ring = np.asarray(cache["ring_second_order"], dtype=np.float64)
    context = np.asarray(cache["context_second_order"], dtype=np.float64)
    signed_surface = core - ring
    sum_surface = core + ring
    maturity_targets = MATURITY_CODE[labels]
    transport_targets = TRANSPORT_CODE[labels]

    outputs = {
        "control": np.full((labels.size, 5), np.nan, dtype=np.float64),
        "factor": np.full((labels.size, 5), np.nan, dtype=np.float64),
        "maturity": np.full((labels.size, 3), np.nan, dtype=np.float64),
        "transport": np.full((labels.size, 3), np.nan, dtype=np.float64),
        "transport_sum_placebo": np.full((labels.size, 3), np.nan, dtype=np.float64),
        "transport_context_placebo": np.full((labels.size, 3), np.nan, dtype=np.float64),
    }
    coverage = np.zeros(labels.size, dtype=np.int64)
    fold_records: List[Dict[str, object]] = []
    started = time.perf_counter()

    with threadpool_limits(limits=BLAS_THREADS):
        for fold in range(FOLDS):
            fit_rows = np.flatnonzero(folds != fold)
            holdout_rows = np.flatnonzero(folds == fold)
            fit_sources = set(groups[fit_rows].tolist())
            holdout_sources = set(groups[holdout_rows].tolist())
            if fit_sources.intersection(holdout_sources):
                raise ValueError(f"Fold {fold} has source leakage")
            if np.unique(labels[fit_rows]).size != 5 or np.unique(labels[holdout_rows]).size != 5:
                raise ValueError(f"Fold {fold} does not contain all five classes")
            if np.unique(maturity_targets[fit_rows]).size != 3 or np.unique(transport_targets[fit_rows]).size != 3:
                raise ValueError(f"Fold {fold} fit rows do not contain all factor states")

            head_scaler = StandardScaler().fit(head[fit_rows])
            signed_scaler = StandardScaler().fit(signed_surface[fit_rows])
            sum_scaler = StandardScaler().fit(sum_surface[fit_rows])
            context_scaler = StandardScaler().fit(context[fit_rows])
            fit_head = head_scaler.transform(head[fit_rows])
            holdout_head = head_scaler.transform(head[holdout_rows])
            fit_signed = signed_scaler.transform(signed_surface[fit_rows])
            holdout_signed = signed_scaler.transform(signed_surface[holdout_rows])
            fit_sum = sum_scaler.transform(sum_surface[fit_rows])
            holdout_sum = sum_scaler.transform(sum_surface[holdout_rows])
            fit_context = context_scaler.transform(context[fit_rows])
            holdout_context = context_scaler.transform(context[holdout_rows])

            control, control_record = _fit_readout(
                np.concatenate((fit_head, fit_signed), axis=1),
                labels[fit_rows],
                np.concatenate((holdout_head, holdout_signed), axis=1),
                class_count=5,
            )
            maturity, maturity_record = _fit_readout(
                fit_head,
                maturity_targets[fit_rows],
                holdout_head,
                class_count=3,
            )
            transport, transport_record = _fit_readout(
                fit_signed,
                transport_targets[fit_rows],
                holdout_signed,
                class_count=3,
            )
            sum_placebo, sum_record = _fit_readout(
                fit_sum,
                transport_targets[fit_rows],
                holdout_sum,
                class_count=3,
            )
            context_placebo, context_record = _fit_readout(
                fit_context,
                transport_targets[fit_rows],
                holdout_context,
                class_count=3,
            )
            factor = factor_product_probabilities(maturity, transport)

            outputs["control"][holdout_rows] = control
            outputs["factor"][holdout_rows] = factor
            outputs["maturity"][holdout_rows] = maturity
            outputs["transport"][holdout_rows] = transport
            outputs["transport_sum_placebo"][holdout_rows] = sum_placebo
            outputs["transport_context_placebo"][holdout_rows] = context_placebo
            coverage[holdout_rows] += 1
            fold_records.append(
                {
                    "fold": int(fold),
                    "fit_rows": int(fit_rows.size),
                    "holdout_rows": int(holdout_rows.size),
                    "source_overlap": 0,
                    "fit_indices_sha256": _array_sha256(fit_rows),
                    "holdout_indices_sha256": _array_sha256(holdout_rows),
                    "scalers": {
                        "head": _scaler_record(head_scaler),
                        "signed_surface": _scaler_record(signed_scaler),
                        "sum_placebo": _scaler_record(sum_scaler),
                        "context_placebo": _scaler_record(context_scaler),
                    },
                    "readouts": {
                        "control": control_record,
                        "maturity": maturity_record,
                        "transport": transport_record,
                        "transport_sum_placebo": sum_record,
                        "transport_context_placebo": context_record,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "factor_concept_oof",
                        "fold_completed": int(fold),
                        "elapsed_seconds": time.perf_counter() - started,
                        "all_readouts_converged": all(
                            bool(row["converged"])
                            for row in fold_records[-1]["readouts"].values()
                        ),
                    }
                ),
                flush=True,
            )

    if not np.all(coverage == 1):
        raise RuntimeError("OOF coverage is not exactly one prediction per row")
    for name, probabilities in outputs.items():
        _validate_probability_matrix(
            probabilities,
            rows=labels.size,
            classes=5 if name in {"control", "factor"} else 3,
            label=name,
        )
    return outputs, {
        "coverage_sha256": _array_sha256(coverage),
        "coverage_counts": {
            str(value): int((coverage == value).sum()) for value in np.unique(coverage)
        },
        "all_readouts_converged": all(
            bool(readout["converged"])
            for fold_record in fold_records
            for readout in fold_record["readouts"].values()
        ),
        "folds": fold_records,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, *, class_count: int
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    confusion = np.zeros((int(class_count), int(class_count)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / max(1.0, support.sum())),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(np.int64).tolist(),
        "predicted_support": predicted_support.astype(np.int64).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    probabilities = _validate_probability_matrix(
        probabilities,
        rows=np.asarray(targets).size,
        classes=np.asarray(probabilities).shape[1],
        label="metric",
    )
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = probabilities.argmax(axis=1)
    result = _classification_metrics(
        targets, predictions, class_count=probabilities.shape[1]
    )
    confidence = probabilities.max(axis=1)
    correctness = (predictions == targets).astype(np.float64)
    ece = 0.0
    for index in range(10):
        low = index / 10.0
        high = (index + 1) / 10.0
        selected = (confidence >= low) & (
            confidence <= high if index == 9 else confidence < high
        )
        if bool(selected.any()):
            ece += float(selected.mean()) * abs(
                float(correctness[selected].mean()) - float(confidence[selected].mean())
            )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    result["calibration"] = {
        "nll": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(targets.size), targets], 1e-12, 1.0
                )
            ).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_10": float(ece),
        "mean_confidence": float(confidence.mean()),
    }
    return result


def _numeric_summary(values: np.ndarray) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        return None
    return float(average_precision_score(labels, scores))


def _directional_summary(
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    baseline = np.asarray(baseline_predictions, dtype=np.int64).reshape(-1)
    candidate = np.asarray(candidate_predictions, dtype=np.int64).reshape(-1)
    restricted = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
    restricted_remove = restricted & (baseline == FOCUS_CLASS) & (
        candidate != FOCUS_CLASS
    )
    restricted_create = restricted & (baseline != FOCUS_CLASS) & (
        candidate == FOCUS_CLASS
    )
    transition = np.zeros((5, 5), dtype=np.int64)
    np.add.at(transition, (baseline, candidate), 1)
    return {
        "focus_fn_rescue": int(
            ((targets == FOCUS_CLASS) & (baseline != FOCUS_CLASS) & (candidate == FOCUS_CLASS)).sum()
        ),
        "focus_tp_break": int(
            ((targets == FOCUS_CLASS) & (baseline == FOCUS_CLASS) & (candidate != FOCUS_CLASS)).sum()
        ),
        "focus_fp_remove_correct": int(
            ((targets != FOCUS_CLASS) & (baseline == FOCUS_CLASS) & (candidate == targets)).sum()
        ),
        "focus_fp_create": int(
            ((targets != FOCUS_CLASS) & (baseline == targets) & (candidate == FOCUS_CLASS)).sum()
        ),
        "candidate_correction": int(((baseline != targets) & (candidate == targets)).sum()),
        "candidate_harm": int(((baseline == targets) & (candidate != targets)).sum()),
        "restricted_fp_remove": int(restricted_remove.sum()),
        "restricted_fp_create": int(restricted_create.sum()),
        "restricted_fp_net_removal": int(restricted_remove.sum() - restricted_create.sum()),
        "transition_matrix_baseline_to_candidate": transition.tolist(),
    }


def _transport_log_odds(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.log(np.clip(probabilities[:, 1], 1e-12, 1.0)) - np.log(
        np.clip(probabilities[:, 0] + probabilities[:, 2], 1e-12, 1.0)
    )


def analyze_predictions(
    targets: np.ndarray,
    folds: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    factor_probabilities: np.ndarray,
    maturity_probabilities: np.ndarray,
    transport_probabilities: np.ndarray,
    sum_placebo_probabilities: np.ndarray,
    context_placebo_probabilities: np.ndarray,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    if folds.size != targets.size:
        raise ValueError("Targets and folds do not align")
    matrices = {
        "keeper": _validate_probability_matrix(
            keeper_probabilities, rows=targets.size, classes=5, label="keeper"
        ),
        "control": _validate_probability_matrix(
            control_probabilities, rows=targets.size, classes=5, label="control"
        ),
        "factor": _validate_probability_matrix(
            factor_probabilities, rows=targets.size, classes=5, label="factor"
        ),
        "maturity": _validate_probability_matrix(
            maturity_probabilities, rows=targets.size, classes=3, label="maturity"
        ),
        "transport": _validate_probability_matrix(
            transport_probabilities, rows=targets.size, classes=3, label="transport"
        ),
        "sum_placebo": _validate_probability_matrix(
            sum_placebo_probabilities, rows=targets.size, classes=3, label="sum_placebo"
        ),
        "context_placebo": _validate_probability_matrix(
            context_placebo_probabilities,
            rows=targets.size,
            classes=3,
            label="context_placebo",
        ),
    }
    predictions = {name: values.argmax(axis=1) for name, values in matrices.items()}
    metrics = {
        name: _metrics(targets, matrices[name]) for name in ("keeper", "control", "factor")
    }
    factor_states = {
        "maturity": _metrics(MATURITY_CODE[targets], matrices["maturity"]),
        "transport": _metrics(TRANSPORT_CODE[targets], matrices["transport"]),
        "transport_sum_placebo": _metrics(
            TRANSPORT_CODE[targets], matrices["sum_placebo"]
        ),
        "transport_context_placebo": _metrics(
            TRANSPORT_CODE[targets], matrices["context_placebo"]
        ),
    }
    directional_control = _directional_summary(
        targets, predictions["control"], predictions["factor"]
    )
    directional_keeper = _directional_summary(
        targets, predictions["keeper"], predictions["factor"]
    )

    class1_tp = (targets == FOCUS_CLASS) & (predictions["keeper"] == FOCUS_CLASS)
    class1_fn = (targets == FOCUS_CLASS) & (predictions["keeper"] != FOCUS_CLASS)
    restricted_fp = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES) & (
        predictions["keeper"] == FOCUS_CLASS
    )
    remaining = ~(class1_tp | class1_fn | restricted_fp)
    transport_scores = {
        "signed": _transport_log_odds(matrices["transport"]),
        "sum_placebo": _transport_log_odds(matrices["sum_placebo"]),
        "context_placebo": _transport_log_odds(matrices["context_placebo"]),
    }
    delta_p1 = matrices["factor"][:, FOCUS_CLASS] - matrices["control"][:, FOCUS_CLASS]
    cohorts: Dict[str, object] = {}
    for name, mask in {
        "class1_tp": class1_tp,
        "class1_fn": class1_fn,
        "restricted_fp": restricted_fp,
        "remaining": remaining,
    }.items():
        cohorts[name] = {
            "rows": int(mask.sum()),
            "factor_minus_control_p1": _numeric_summary(delta_p1[mask]),
            "signed_transport_log_odds": _numeric_summary(transport_scores["signed"][mask]),
            "sum_placebo_transport_log_odds": _numeric_summary(
                transport_scores["sum_placebo"][mask]
            ),
            "context_placebo_transport_log_odds": _numeric_summary(
                transport_scores["context_placebo"][mask]
            ),
        }

    tp_fp_mask = class1_tp | restricted_fp
    tp_fp_labels = class1_tp[tp_fp_mask].astype(np.int64)
    signed_auc = _safe_auc(tp_fp_labels, transport_scores["signed"][tp_fp_mask])
    sum_auc = _safe_auc(tp_fp_labels, transport_scores["sum_placebo"][tp_fp_mask])
    context_auc = _safe_auc(
        tp_fp_labels, transport_scores["context_placebo"][tp_fp_mask]
    )
    fn_fp_mask = class1_fn | restricted_fp
    fn_fp_labels = class1_fn[fn_fp_mask].astype(np.int64)
    direction_auc = _safe_auc(fn_fp_labels, delta_p1[fn_fp_mask])
    direction = {
        "tp_vs_restricted_fp_rows": int(tp_fp_mask.sum()),
        "class1_tp_rows": int(class1_tp.sum()),
        "class1_fn_rows": int(class1_fn.sum()),
        "restricted_fp_rows": int(restricted_fp.sum()),
        "signed_transport_tp_vs_fp_auroc": signed_auc,
        "signed_transport_tp_vs_fp_average_precision": _safe_ap(
            tp_fp_labels, transport_scores["signed"][tp_fp_mask]
        ),
        "sum_placebo_tp_vs_fp_auroc": sum_auc,
        "context_placebo_tp_vs_fp_auroc": context_auc,
        "signed_minus_best_placebo_auroc": None
        if signed_auc is None or sum_auc is None or context_auc is None
        else float(signed_auc - max(sum_auc, context_auc)),
        "factor_minus_control_p1_fn_vs_fp_auroc": direction_auc,
        "factor_minus_control_p1_fn_vs_fp_average_precision": _safe_ap(
            fn_fp_labels, delta_p1[fn_fp_mask]
        ),
    }

    fold_details: List[Dict[str, object]] = []
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == int(fold)
        control_metrics = _metrics(targets[selected], matrices["control"][selected])
        factor_metrics = _metrics(targets[selected], matrices["factor"][selected])
        directional = _directional_summary(
            targets[selected],
            predictions["control"][selected],
            predictions["factor"][selected],
        )
        control_tp = int(
            ((targets[selected] == FOCUS_CLASS) & (predictions["control"][selected] == FOCUS_CLASS)).sum()
        )
        factor_tp = int(
            ((targets[selected] == FOCUS_CLASS) & (predictions["factor"][selected] == FOCUS_CLASS)).sum()
        )
        fold_details.append(
            {
                "fold": int(fold),
                "rows": int(selected.sum()),
                "control_metrics": control_metrics,
                "factor_metrics": factor_metrics,
                "directional": directional,
                "class1_net_tp_loss": int(control_tp - factor_tp),
            }
        )

    control_metrics = metrics["control"]
    factor_metrics = metrics["factor"]
    deltas = {
        "macro_f1": float(factor_metrics["macro_f1"] - control_metrics["macro_f1"]),
        "class1_f1": float(
            factor_metrics["per_class_f1"][FOCUS_CLASS]
            - control_metrics["per_class_f1"][FOCUS_CLASS]
        ),
        "class1_precision": float(
            factor_metrics["per_class_precision"][FOCUS_CLASS]
            - control_metrics["per_class_precision"][FOCUS_CLASS]
        ),
        "class1_recall": float(
            factor_metrics["per_class_recall"][FOCUS_CLASS]
            - control_metrics["per_class_recall"][FOCUS_CLASS]
        ),
        "nonfocus_f1_losses": {
            str(index): float(
                control_metrics["per_class_f1"][index]
                - factor_metrics["per_class_f1"][index]
            )
            for index in (0, 2, 3, 4)
        },
    }
    mechanism_gates = {
        "macro_f1_delta_ge_minus0002": float(deltas["macro_f1"]) >= -0.002,
        "class1_f1_gain_0010": float(deltas["class1_f1"]) >= 0.010,
        "class1_f1_ge_080": float(factor_metrics["per_class_f1"][FOCUS_CLASS]) >= 0.80,
        "class1_precision_gain_0015": float(deltas["class1_precision"]) >= 0.015,
        "class1_recall_delta_ge_minus0005": float(deltas["class1_recall"]) >= -0.005,
        "restricted_fp_net_removal_10": int(
            directional_control["restricted_fp_net_removal"]
        )
        >= 10,
        "corrections_not_below_harms": int(directional_control["candidate_correction"])
        >= int(directional_control["candidate_harm"]),
        "focus_tp_break_not_above_rescue": int(directional_control["focus_tp_break"])
        <= int(directional_control["focus_fn_rescue"]),
        "nonfocus_max_f1_loss_le_0010": max(
            float(value) for value in deltas["nonfocus_f1_losses"].values()
        )
        <= 0.010,
        "signed_transport_tp_fp_auroc_ge_065": signed_auc is not None
        and signed_auc >= 0.65,
        "signed_transport_auroc_beats_placebos_002": signed_auc is not None
        and sum_auc is not None
        and context_auc is not None
        and signed_auc - max(sum_auc, context_auc) >= 0.02,
        "factor_control_fn_fp_direction_auroc_ge_062": direction_auc is not None
        and direction_auc >= 0.62,
        "fold_precision_nonworse_4": sum(
            float(row["factor_metrics"]["per_class_precision"][FOCUS_CLASS])
            >= float(row["control_metrics"]["per_class_precision"][FOCUS_CLASS])
            for row in fold_details
        )
        >= 4,
        "fold_restricted_fp_positive_4": sum(
            int(row["directional"]["restricted_fp_net_removal"]) > 0
            for row in fold_details
        )
        >= 4,
        "fold_class1_net_tp_loss_le_2": all(
            int(row["class1_net_tp_loss"]) <= 2 for row in fold_details
        ),
    }
    return {
        "metrics": metrics,
        "factor_state_metrics": factor_states,
        "deltas": deltas,
        "factor_vs_control": directional_control,
        "factor_vs_keeper": directional_keeper,
        "cohorts": cohorts,
        "direction": direction,
        "folds": fold_details,
        "mechanism_gates": mechanism_gates,
        "mechanism_gates_passed": all(mechanism_gates.values()),
    }


def _prediction_rows(
    *,
    cache: Mapping[str, object],
    folds: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    targets = np.asarray(cache["labels"], dtype=np.int64)
    keeper = np.asarray(cache["probabilities"], dtype=np.float64)
    control = np.asarray(outputs["control"], dtype=np.float64)
    factor = np.asarray(outputs["factor"], dtype=np.float64)
    maturity = np.asarray(outputs["maturity"], dtype=np.float64)
    transport = np.asarray(outputs["transport"], dtype=np.float64)
    sum_placebo = np.asarray(outputs["transport_sum_placebo"], dtype=np.float64)
    context_placebo = np.asarray(
        outputs["transport_context_placebo"], dtype=np.float64
    )
    keeper_prediction = keeper.argmax(axis=1)
    control_prediction = control.argmax(axis=1)
    factor_prediction = factor.argmax(axis=1)
    signed_score = _transport_log_odds(transport)
    sum_score = _transport_log_odds(sum_placebo)
    context_score = _transport_log_odds(context_placebo)
    factor_vs_control = _directional_masks(
        targets, control_prediction, factor_prediction
    )
    factor_vs_keeper = _directional_masks(targets, keeper_prediction, factor_prediction)
    rows: List[Dict[str, object]] = []
    for index in range(targets.size):
        row: Dict[str, object] = {
            "sample_index": int(cache["sample_index"][index]),
            "image_path": str(cache["paths"][index]),
            "source_stem": str(cache["source_stem"][index]),
            "fold": int(folds[index]),
            "target_index": int(targets[index]),
            "target_name": str(cache["classes"][targets[index]]),
            "keeper_prediction": int(keeper_prediction[index]),
            "control_prediction": int(control_prediction[index]),
            "factor_prediction": int(factor_prediction[index]),
            "maturity_target": int(MATURITY_CODE[targets[index]]),
            "maturity_prediction": int(maturity[index].argmax()),
            "transport_target": int(TRANSPORT_CODE[targets[index]]),
            "transport_prediction": int(transport[index].argmax()),
            "factor_minus_control_p1": float(
                factor[index, FOCUS_CLASS] - control[index, FOCUS_CLASS]
            ),
            "signed_transport_log_odds": float(signed_score[index]),
            "sum_placebo_transport_log_odds": float(sum_score[index]),
            "context_placebo_transport_log_odds": float(context_score[index]),
        }
        for prefix, masks in (
            ("control", factor_vs_control),
            ("keeper", factor_vs_keeper),
        ):
            for name, mask in masks.items():
                row[f"factor_vs_{prefix}_{name}"] = bool(mask[index])
        for class_index in range(5):
            row[f"keeper_prob_{class_index}"] = float(keeper[index, class_index])
            row[f"control_prob_{class_index}"] = float(control[index, class_index])
            row[f"factor_prob_{class_index}"] = float(factor[index, class_index])
        for state in range(3):
            row[f"maturity_prob_{state}"] = float(maturity[index, state])
            row[f"transport_prob_{state}"] = float(transport[index, state])
            row[f"sum_placebo_prob_{state}"] = float(sum_placebo[index, state])
            row[f"context_placebo_prob_{state}"] = float(context_placebo[index, state])
        rows.append(row)
    return rows


def _directional_masks(
    targets: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, np.ndarray]:
    restricted = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES)
    return {
        "focus_fn_rescue": (targets == FOCUS_CLASS)
        & (baseline != FOCUS_CLASS)
        & (candidate == FOCUS_CLASS),
        "focus_tp_break": (targets == FOCUS_CLASS)
        & (baseline == FOCUS_CLASS)
        & (candidate != FOCUS_CLASS),
        "candidate_correction": (baseline != targets) & (candidate == targets),
        "candidate_harm": (baseline == targets) & (candidate != targets),
        "restricted_fp_remove": restricted
        & (baseline == FOCUS_CLASS)
        & (candidate != FOCUS_CLASS),
        "restricted_fp_create": restricted
        & (baseline != FOCUS_CLASS)
        & (candidate == FOCUS_CLASS),
    }


def _fold_csv_rows(folds: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for fold in folds:
        control = fold["control_metrics"]
        factor = fold["factor_metrics"]
        directional = fold["directional"]
        rows.append(
            {
                "fold": int(fold["fold"]),
                "rows": int(fold["rows"]),
                "control_macro_f1": float(control["macro_f1"]),
                "factor_macro_f1": float(factor["macro_f1"]),
                "control_class1_precision": float(
                    control["per_class_precision"][FOCUS_CLASS]
                ),
                "factor_class1_precision": float(
                    factor["per_class_precision"][FOCUS_CLASS]
                ),
                "control_class1_recall": float(
                    control["per_class_recall"][FOCUS_CLASS]
                ),
                "factor_class1_recall": float(
                    factor["per_class_recall"][FOCUS_CLASS]
                ),
                "control_class1_f1": float(control["per_class_f1"][FOCUS_CLASS]),
                "factor_class1_f1": float(factor["per_class_f1"][FOCUS_CLASS]),
                "restricted_fp_remove": int(directional["restricted_fp_remove"]),
                "restricted_fp_create": int(directional["restricted_fp_create"]),
                "restricted_fp_net_removal": int(
                    directional["restricted_fp_net_removal"]
                ),
                "focus_fn_rescue": int(directional["focus_fn_rescue"]),
                "focus_tp_break": int(directional["focus_tp_break"]),
                "candidate_correction": int(directional["candidate_correction"]),
                "candidate_harm": int(directional["candidate_harm"]),
                "class1_net_tp_loss": int(fold["class1_net_tp_loss"]),
            }
        )
    return rows


def _artifact_record(path: Path, *, root: Path) -> Dict[str, object]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_manifest(output_dir: Path) -> Path:
    manifest_path = output_dir / "artifact_manifest.json"
    payloads = [
        _artifact_record(path, root=output_dir)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "mode": "factor_concept_product_a0_artifact_manifest",
            "payload_count": len(payloads),
            "payload_bytes": int(sum(int(row["bytes"]) for row in payloads)),
            "payloads": payloads,
        },
    )
    return manifest_path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(payload.get("payloads", []))
    for row in rows:
        path = output_dir / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Manifest payload is missing: {path}")
        if int(path.stat().st_size) != int(row["bytes"]):
            raise ValueError(f"Manifest size mismatch: {path}")
        if _sha256(path) != str(row["sha256"]):
            raise ValueError(f"Manifest SHA-256 mismatch: {path}")
    actual_payloads = {
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    recorded_payloads = {str(row["path"]) for row in rows}
    if actual_payloads != recorded_payloads:
        raise ValueError("Manifest payload set differs from output directory")
    return {
        "payload_count": len(rows),
        "payload_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "manifest_sha256": _sha256(manifest_path),
    }


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError("Replay mapping keys differ")
        return max(
            (_recursive_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Replay list lengths differ")
        return max(
            (_recursive_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        if bool(left) != bool(right):
            raise ValueError("Replay boolean values differ")
        return 0.0
    if left is None or right is None:
        if left is not right:
            raise ValueError("Replay null values differ")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError(f"Replay values differ: {left!r} != {right!r}")
    return 0.0


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    output_dir = resolved.parent
    manifest = _verify_manifest(output_dir)
    with (output_dir / "predictions.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    targets = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)

    def probabilities(prefix: str, classes: int) -> np.ndarray:
        return np.asarray(
            [
                [float(row[f"{prefix}_prob_{index}"]) for index in range(classes)]
                for row in rows
            ],
            dtype=np.float64,
        )

    replayed = analyze_predictions(
        targets,
        folds,
        probabilities("keeper", 5),
        probabilities("control", 5),
        probabilities("factor", 5),
        probabilities("maturity", 3),
        probabilities("transport", 3),
        probabilities("sum_placebo", 3),
        probabilities("context_placebo", 3),
    )
    maximum_difference = _recursive_numeric_difference(summary["analysis"], replayed)
    if maximum_difference > 1e-12:
        raise ValueError(
            f"Replay numerical mismatch exceeds tolerance: {maximum_difference}"
        )
    return {
        "replay_passed": True,
        "rows": len(rows),
        "maximum_numeric_difference": float(maximum_difference),
        "mechanism_gates_passed": bool(replayed["mechanism_gates_passed"]),
        **manifest,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.folds) != FOLDS or int(args.seed) != SEED:
        raise ValueError(f"The protocol locks folds={FOLDS} and seed={SEED}")
    if int(args.blas_threads) != BLAS_THREADS:
        raise ValueError(f"The implementation locks blas-threads={BLAS_THREADS}")
    preflight_checks = verify_preflight_checks()
    provenance = verify_locked_inputs()
    repo_state = _repo_state()
    if not bool(repo_state["head_matches_upstream"]) or not bool(
        repo_state["tracked_worktree_clean"]
    ):
        raise ValueError(f"Formal audit requires clean pushed tracked state: {repo_state}")
    output_dir = _require_empty_output(args.output_dir)
    started = time.perf_counter()
    rss_start = int(psutil.Process().memory_info().rss)
    with _PeakRssMonitor() as memory_monitor:
        cache = load_locked_train_cache(
            Path(provenance["files"]["train_descriptor_cache"]["path"])
        )
        folds, fold_assignment = assign_source_folds(
            cache["labels"], cache["source_stem"], folds=FOLDS, seed=SEED
        )
        outputs, fit_provenance = fit_oof_readouts(cache, folds)
        analysis = analyze_predictions(
            cache["labels"],
            folds,
            cache["probabilities"],
            outputs["control"],
            outputs["factor"],
            outputs["maturity"],
            outputs["transport"],
            outputs["transport_sum_placebo"],
            outputs["transport_context_placebo"],
        )
        rows = _prediction_rows(cache=cache, folds=folds, outputs=outputs)
        predictions_path = output_dir / "predictions.csv"
        fold_path = output_dir / "fold_summary.csv"
        cohort_path = output_dir / "cohort_summary.json"
        provenance_path = output_dir / "fold_provenance.json"
        _write_csv(predictions_path, rows)
        _write_csv(fold_path, _fold_csv_rows(analysis["folds"]))
        _write_json(cohort_path, analysis["cohorts"])
        _write_json(
            provenance_path,
            {
                "fold_assignment": fold_assignment,
                "fit": fit_provenance,
            },
        )

    elapsed = float(time.perf_counter() - started)
    all_probabilities = [cache["probabilities"], *outputs.values()]
    structural_gates = {
        "locked_hashes_verified": True,
        "repo_clean_and_pushed": bool(repo_state["head_matches_upstream"])
        and bool(repo_state["tracked_worktree_clean"]),
        "train_cache_only": True,
        "validation_data_unopened": True,
        "test_data_unopened": True,
        "cache_keys_exact": tuple(cache["_observed_keys"]) == EXPECTED_CACHE_KEYS,
        "rows_9215": len(cache["labels"]) == EXPECTED_TRAIN_ROWS,
        "class_counts_exact": tuple(
            np.bincount(cache["labels"], minlength=5).tolist()
        )
        == EXPECTED_CLASS_COUNTS,
        "class_order_exact": tuple(cache["classes"]) == EXPECTED_CLASSES,
        "sample_order_exact": bool(
            np.array_equal(
                cache["sample_index"], np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)
            )
        ),
        "source_groups_8064": len(set(cache["source_stem"].tolist()))
        == EXPECTED_SOURCE_GROUPS,
        "train_paths_only": all(_is_train_image_path(path) for path in cache["paths"]),
        "dimensions_256_324_324_324": cache["head"].shape[1] == 256
        and cache["core_second_order"].shape[1] == 324
        and cache["ring_second_order"].shape[1] == 324
        and cache["context_second_order"].shape[1] == 324,
        "five_fold_oof_complete": fit_provenance["coverage_counts"]
        == {"1": EXPECTED_TRAIN_ROWS},
        "source_fold_overlap_zero": int(fold_assignment["source_overlap"]) == 0,
        "all_fold_classes_present": all(
            all(int(value) > 0 for value in row["fit_class_counts"])
            and all(int(value) > 0 for value in row["holdout_class_counts"])
            for row in fold_assignment["folds"]
        ),
        "all_readouts_converged": bool(fit_provenance["all_readouts_converged"]),
        "all_probabilities_finite_normalized": all(
            np.isfinite(probabilities).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
            for probabilities in all_probabilities
        ),
        "compile_pyflakes_tests_diff_check_passed": all(
            bool(row["passed"]) for row in preflight_checks.values()
        ),
    }
    structural_passed = all(structural_gates.values())
    all_gates_passed = structural_passed and bool(analysis["mechanism_gates_passed"])
    artifact_hashes = {
        "predictions": _artifact_record(predictions_path, root=output_dir),
        "fold_summary": _artifact_record(fold_path, root=output_dir),
        "cohort_summary": _artifact_record(cohort_path, root=output_dir),
        "fold_provenance": _artifact_record(provenance_path, root=output_dir),
    }
    summary = {
        "mode": "factor_concept_product_a0_train_information_gate",
        "status": "passed" if all_gates_passed else "rejected",
        "split": "train",
        "validation_data_used": False,
        "test_data_used": False,
        "rows": EXPECTED_TRAIN_ROWS,
        "class_names": list(EXPECTED_CLASSES),
        "focus_class": FOCUS_CLASS,
        "restricted_negative_classes": list(RESTRICTED_NEGATIVE_CLASSES),
        "maturity_code": MATURITY_CODE.tolist(),
        "transport_code": TRANSPORT_CODE.tolist(),
        "surface_equation": "core_second_order - ring_second_order",
        "decoder": "softmax(log_p_maturity[m(c)] + log_p_transport[t(c)])",
        "readout": {
            "solver": "lbfgs",
            "C": LOGISTIC_C,
            "max_iter": LOGISTIC_MAX_ITER,
            "class_weight": "balanced",
            "random_state": SEED,
            "blas_threads": BLAS_THREADS,
        },
        "provenance": provenance,
        "repo_state": repo_state,
        "preflight_checks": preflight_checks,
        "fold_assignment": fold_assignment,
        "fit_provenance": fit_provenance,
        "runtime": {
            "elapsed_seconds": elapsed,
            "rss_start_bytes": rss_start,
            "peak_sampled_rss_bytes": int(memory_monitor.peak_bytes),
            "peak_sampled_rss_gib": float(memory_monitor.peak_bytes / 1024**3),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
        },
        "analysis": analysis,
        "structural_gates": structural_gates,
        "structural_gates_passed": structural_passed,
        "all_gates_passed": all_gates_passed,
        "artifacts": artifact_hashes,
        "guardrail": (
            "Any failed gate closes this fixed factor code, signed descriptor, and "
            "product decoder before validation, XAI, model/trainer changes, smoke, "
            "probe, test, or full training."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(output_dir)
    summary["artifact_manifest"] = {
        "path": manifest_path.name,
        "sha256": _sha256(manifest_path),
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.preflight_only and args.replay_summary is not None:
        raise ValueError("preflight-only and replay-summary are mutually exclusive")
    if args.preflight_only:
        output = Path(args.output_dir).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Preflight output path must not exist: {output}")
        print(
            json.dumps(
                to_serializable(
                    {
                        "preflight_passed": True,
                        "checks": verify_preflight_checks(),
                        "provenance": verify_locked_inputs(),
                        "repo_state": _repo_state(),
                        "cache_opened": False,
                        "output_created": False,
                    }
                ),
                indent=2,
            ),
            flush=True,
        )
        return
    if args.replay_summary is not None:
        print(
            json.dumps(to_serializable(replay_summary(args.replay_summary)), indent=2),
            flush=True,
        )
        return
    summary = run_audit(args)
    print(
        json.dumps(
            to_serializable(
                {
                    "status": summary["status"],
                    "deltas": summary["analysis"]["deltas"],
                    "factor_class1": {
                        "precision": summary["analysis"]["metrics"]["factor"][
                            "per_class_precision"
                        ][FOCUS_CLASS],
                        "recall": summary["analysis"]["metrics"]["factor"][
                            "per_class_recall"
                        ][FOCUS_CLASS],
                        "f1": summary["analysis"]["metrics"]["factor"][
                            "per_class_f1"
                        ][FOCUS_CLASS],
                    },
                    "direction": summary["analysis"]["direction"],
                    "failed_mechanism_gates": [
                        name
                        for name, passed in summary["analysis"]["mechanism_gates"].items()
                        if not bool(passed)
                    ],
                    "structural_gates_passed": summary["structural_gates_passed"],
                    "all_gates_passed": summary["all_gates_passed"],
                    "artifact_manifest": summary["artifact_manifest"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    if not bool(summary["structural_gates_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

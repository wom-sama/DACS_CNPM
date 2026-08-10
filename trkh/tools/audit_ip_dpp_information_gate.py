from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from dppy.finite_dpps import FiniteDPP
from scipy.stats import spearmanr

from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)


EXPECTED_ROWS = 9215
EXPECTED_SOURCE_GROUPS = 8064
EXPECTED_CLASS_COUNTS = {0: 1941, 1: 541, 2: 1920, 3: 2520, 4: 2293}
EXPECTED_FOLD_COUNTS = {0: 1843, 1: 1830, 2: 1828, 3: 1851, 4: 1863}
EXPECTED_CONDITIONS = (
    "clean",
    "lighting_dim",
    "lighting_bright",
    "low_contrast",
)
MODEL_PREFIXES = ("candidate", "keeper")
CLASS_ORDER = (4, 3, 2, 1, 0)
FOCUS_CLASS = 1
RESTRICTED_FOCUS_SOURCES = (0, 2, 4)
K_PER_CLASS = 541
OFFICIAL_SEED = 42
NULL_SEED_BASE = 20260715
NULL_PROPOSALS = 256
DPPY_DETERMINANT_TOLERANCE = 1e-9
CODE_ODDS_GATE = 1.10

DEFAULT_CACHE = Path(
    "runs/audit_cidt_readiness_full_train_20260714/"
    "predictions_all_conditions.csv"
)
DEFAULT_PROTOCOL = Path(
    "docs/TRKH_5CLASS_IP_DPP_INFORMATION_GATE_PROTOCOL_20260715.md"
)
DEFAULT_OFFICIAL_SOURCE = Path(
    "D:/DataAI/external_sources/BNS_IPDPP/sampling/ip_dpp_smapler.py"
)
LOCKED_SHA256 = {
    "cache": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "protocol": "652c1fa4c2e525be7e89b0983184978d2b5e8c2c28ce1af4e60119913e9cd77f",
    "official_source": (
        "2754011d4bf556fbd4a8ada40d0169b57c03a6e62cb18ad0c7292a14cb133f57"
    ),
}
OFFICIAL_COMMIT = "20d69a676215e854d34fbdefa4ba4b3e165c41c0"
EXPECTED_VERSIONS = {
    "dppy": "0.3.3",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only IP-DPP paper/code/random information gate. "
            "This tool never constructs validation or test data."
        )
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--official-source", type=Path, default=DEFAULT_OFFICIAL_SOURCE
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_sha256(indices: np.ndarray) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames: List[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_locked_sources(args: argparse.Namespace) -> Dict[str, object]:
    paths = {
        "cache": Path(args.cache).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "official_source": Path(args.official_source).resolve(),
    }
    hashes: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Locked {name} is missing: {path}")
        observed = _sha256(path)
        if observed != LOCKED_SHA256[name]:
            raise ValueError(
                f"Locked {name} SHA mismatch: {observed} != {LOCKED_SHA256[name]}"
            )
        hashes[name] = observed

    repository = paths["official_source"].parents[1]
    commit = _git_output(repository, "rev-parse", "HEAD")
    status = _git_output(repository, "status", "--porcelain")
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"Official repository commit drift: {commit}")
    if status:
        raise ValueError("Official repository worktree is not clean")

    versions = {
        "dppy": importlib.metadata.version("dppy"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    if versions != EXPECTED_VERSIONS:
        raise ValueError(f"Locked numerical dependency drift: {versions}")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "official_repository": str(repository),
        "official_commit": commit,
        "official_worktree_clean": True,
        "versions": versions,
    }


def load_train_only_cache(path: Path) -> Dict[str, object]:
    labels = np.full(EXPECTED_ROWS, -1, dtype=np.int64)
    folds = np.full(EXPECTED_ROWS, -1, dtype=np.int64)
    sources = np.full(EXPECTED_ROWS, "", dtype=object)
    image_paths = np.full(EXPECTED_ROWS, "", dtype=object)
    seen = {
        condition: np.zeros(EXPECTED_ROWS, dtype=bool)
        for condition in EXPECTED_CONDITIONS
    }
    probabilities = {
        model: {
            condition: np.full((EXPECTED_ROWS, 5), np.nan, dtype=np.float64)
            for condition in EXPECTED_CONDITIONS
        }
        for model in MODEL_PREFIXES
    }

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "fold",
            "target_index",
            *(f"{model}_prediction" for model in MODEL_PREFIXES),
            *(f"{model}_target_probability" for model in MODEL_PREFIXES),
            *(
                f"{model}_prob_{class_index}"
                for model in MODEL_PREFIXES
                for class_index in range(5)
            ),
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Train cache is missing columns: {sorted(missing)}")
        for raw in reader:
            condition = str(raw["condition"]).strip()
            if condition not in seen:
                raise ValueError(f"Unexpected cache condition: {condition}")
            index = int(raw["sample_index"])
            if not 0 <= index < EXPECTED_ROWS:
                raise ValueError(f"Out-of-range sample index: {index}")
            if seen[condition][index]:
                raise ValueError(f"Duplicate cache row: {condition}/{index}")
            seen[condition][index] = True

            label = int(raw["target_index"])
            fold = int(raw["fold"])
            source = str(raw["source_stem"]).strip().replace("/", "\\").casefold()
            image_path = str(Path(str(raw["image_path"])).resolve())
            if labels[index] < 0:
                labels[index] = label
                folds[index] = fold
                sources[index] = source
                image_paths[index] = image_path
            elif (
                labels[index] != label
                or folds[index] != fold
                or sources[index] != source
                or image_paths[index] != image_path
            ):
                raise ValueError(f"Condition metadata drift for sample {index}")

            for model in MODEL_PREFIXES:
                row_probabilities = np.asarray(
                    [float(raw[f"{model}_prob_{class_index}"]) for class_index in range(5)],
                    dtype=np.float64,
                )
                if not np.isfinite(row_probabilities).all() or not math.isclose(
                    float(row_probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=2e-5
                ):
                    raise ValueError(
                        f"Invalid {model} probabilities at {condition}/{index}"
                    )
                target_probability = float(raw[f"{model}_target_probability"])
                if not math.isclose(
                    target_probability,
                    float(row_probabilities[label]),
                    rel_tol=0.0,
                    abs_tol=2e-6,
                ):
                    raise ValueError(
                        f"{model} target-probability drift at {condition}/{index}"
                    )
                prediction = int(raw[f"{model}_prediction"])
                if prediction != int(np.argmax(row_probabilities)):
                    raise ValueError(
                        f"{model} prediction drift at {condition}/{index}"
                    )
                probabilities[model][condition][index] = row_probabilities

    missing_rows = {
        condition: int((~condition_seen).sum())
        for condition, condition_seen in seen.items()
        if not condition_seen.all()
    }
    if missing_rows:
        raise ValueError(f"Incomplete condition coverage: {missing_rows}")
    if np.any(labels < 0) or np.any(folds < 0):
        raise ValueError("Incomplete clean metadata")

    class_counts = {
        class_index: int((labels == class_index).sum()) for class_index in range(5)
    }
    fold_counts = {fold: int((folds == fold).sum()) for fold in range(5)}
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {class_counts}")
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Unexpected fold counts: {fold_counts}")
    if int(np.unique(sources).size) != EXPECTED_SOURCE_GROUPS:
        raise ValueError("Unexpected source-group count")

    train_root = Path("D:/DataAI/AIEx/newdataset/yolo_f/images/train").resolve()
    invalid_paths = [
        value
        for value in image_paths.tolist()
        if not _is_relative_to(Path(str(value)).resolve(), train_root)
    ]
    if invalid_paths:
        raise ValueError(f"Cache contains a non-train image path: {invalid_paths[0]}")

    source_folds: Dict[str, set[int]] = {}
    for source, fold in zip(sources.tolist(), folds.tolist()):
        source_folds.setdefault(str(source), set()).add(int(fold))
    overlap = {source: values for source, values in source_folds.items() if len(values) > 1}
    if overlap:
        raise ValueError(f"CIDT source-fold leakage: {next(iter(overlap.items()))}")

    return {
        "labels": labels,
        "folds": folds,
        "sources": sources,
        "image_paths": image_paths,
        "probabilities": probabilities,
        "class_counts": class_counts,
        "fold_counts": fold_counts,
        "source_groups": int(np.unique(sources).size),
        "source_fold_overlap": 0,
        "conditions": list(EXPECTED_CONDITIONS),
        "rows_per_condition": {
            condition: int(condition_seen.sum())
            for condition, condition_seen in seen.items()
        },
    }


def target_probabilities(
    data: Mapping[str, object], model: str, condition: str
) -> np.ndarray:
    labels = np.asarray(data["labels"], dtype=np.int64)
    probabilities = np.asarray(data["probabilities"][model][condition])
    return probabilities[np.arange(labels.size), labels]


def kernel_components(probabilities: np.ndarray, mode: str) -> Dict[str, object]:
    values = np.asarray(probabilities, dtype=np.float64)
    count = int(values.size)
    if count <= 0 or not np.isfinite(values).all():
        raise ValueError("Kernel probabilities must be finite and nonempty")
    if mode == "paper_n":
        alpha = 1.0 / count
    elif mode == "code_n2":
        alpha = 1.0 / (count * count)
    else:
        raise ValueError(f"Unknown kernel mode: {mode}")
    diagonal_base = 1.0 - alpha * values * float(values.sum())
    if np.any(diagonal_base <= 0.0):
        raise ValueError(f"Nonpositive diagonal base for {mode}")
    rank_terms = values * values / diagonal_base
    return {
        "mode": mode,
        "alpha": float(alpha),
        "probabilities": values,
        "diagonal_base": diagonal_base,
        "rank_terms": rank_terms,
    }


def dense_kernel(probabilities: np.ndarray, mode: str) -> np.ndarray:
    components = kernel_components(probabilities, mode)
    values = np.asarray(components["probabilities"])
    alpha = float(components["alpha"])
    matrix = alpha * np.outer(values, values)
    np.fill_diagonal(
        matrix,
        np.asarray(components["diagonal_base"]) + alpha * values * values,
    )
    return matrix


def subset_log_determinant(
    components: Mapping[str, object], subset: np.ndarray
) -> float:
    selected = np.asarray(subset, dtype=np.int64)
    diagonal_base = np.asarray(components["diagonal_base"], dtype=np.float64)
    rank_terms = np.asarray(components["rank_terms"], dtype=np.float64)
    alpha = float(components["alpha"])
    if selected.ndim != 1 or np.unique(selected).size != selected.size:
        raise ValueError("Subset indices must be a unique vector")
    if selected.size and (selected.min() < 0 or selected.max() >= diagonal_base.size):
        raise ValueError("Subset index is outside the kernel")
    return float(
        np.log(diagonal_base[selected]).sum()
        + np.log1p(alpha * rank_terms[selected].sum())
    )


def determinant_log_upper_bound(
    components: Mapping[str, object], size: int
) -> float:
    diagonal_base = np.asarray(components["diagonal_base"], dtype=np.float64)
    rank_terms = np.asarray(components["rank_terms"], dtype=np.float64)
    k = int(size)
    if not 0 <= k <= diagonal_base.size:
        raise ValueError("Invalid determinant-bound cardinality")
    if k == 0:
        return 0.0
    largest_log_diagonal = np.partition(np.log(diagonal_base), -k)[-k:]
    largest_rank_terms = np.partition(rank_terms, -k)[-k:]
    return float(
        largest_log_diagonal.sum()
        + np.log1p(float(components["alpha"]) * largest_rank_terms.sum())
    )


def verify_rank_one_equations() -> Dict[str, object]:
    probabilities = np.asarray([0.11, 0.29, 0.57, 0.83, 0.41], dtype=np.float64)
    subsets = (
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3, 4], dtype=np.int64),
    )
    maximum_logdet_error = 0.0
    maximum_symmetry_error = 0.0
    maximum_row_sum_error = 0.0
    minimum_eigenvalue = float("inf")
    maximum_eigenvalue = float("-inf")
    for mode in ("paper_n", "code_n2"):
        matrix = dense_kernel(probabilities, mode)
        maximum_symmetry_error = max(
            maximum_symmetry_error, float(np.max(np.abs(matrix - matrix.T)))
        )
        maximum_row_sum_error = max(
            maximum_row_sum_error,
            float(np.max(np.abs(matrix.sum(axis=1) - 1.0))),
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        minimum_eigenvalue = min(minimum_eigenvalue, float(eigenvalues.min()))
        maximum_eigenvalue = max(maximum_eigenvalue, float(eigenvalues.max()))
        components = kernel_components(probabilities, mode)
        for subset in subsets:
            sign, dense_logdet = np.linalg.slogdet(matrix[np.ix_(subset, subset)])
            if sign <= 0:
                raise ValueError("Small-fixture kernel is not positive definite")
            maximum_logdet_error = max(
                maximum_logdet_error,
                abs(float(dense_logdet) - subset_log_determinant(components, subset)),
            )
    return {
        "maximum_logdet_error": maximum_logdet_error,
        "maximum_symmetry_error": maximum_symmetry_error,
        "maximum_row_sum_error": maximum_row_sum_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "passed": bool(
            maximum_logdet_error <= 1e-12
            and maximum_symmetry_error <= 1e-15
            and maximum_row_sum_error <= 1e-12
            and minimum_eigenvalue >= -1e-12
            and maximum_eigenvalue <= 1.0 + 1e-12
        ),
    }


def build_kernel_audit(data: Mapping[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, Dict[int, Dict[str, Mapping[str, object]]]]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    rows: List[Dict[str, object]] = []
    components_by_model: Dict[str, Dict[int, Dict[str, Mapping[str, object]]]] = {}
    for model in MODEL_PREFIXES:
        clean_targets = target_probabilities(data, model, "clean")
        model_components: Dict[int, Dict[str, Mapping[str, object]]] = {}
        for class_index in range(5):
            class_mask = labels == class_index
            class_probabilities = clean_targets[class_mask]
            class_components: Dict[str, Mapping[str, object]] = {}
            for mode in ("paper_n", "code_n2"):
                components = kernel_components(class_probabilities, mode)
                class_components[mode] = components
                diagonal_base = np.asarray(components["diagonal_base"])
                count = int(class_probabilities.size)
                determinant_bound = determinant_log_upper_bound(
                    components, min(K_PER_CLASS, count)
                )
                rows.append(
                    {
                        "model": model,
                        "class_index": class_index,
                        "class_count": count,
                        "k": min(K_PER_CLASS, count),
                        "kernel_mode": mode,
                        "alpha": float(components["alpha"]),
                        "diagonal_base_min": float(diagonal_base.min()),
                        "diagonal_base_max": float(diagonal_base.max()),
                        "psd_eigenvalue_lower_bound": float(diagonal_base.min()),
                        "row_stochastic_eigenvalue_max": 1.0,
                        "row_sum_max_abs_error_analytic": 0.0,
                        "determinant_log_upper_bound": determinant_bound,
                        "determinant_upper_bound": float(
                            math.exp(max(-745.0, determinant_bound))
                        ),
                        "dppy_initialization_possible_at_tol": bool(
                            determinant_bound
                            > math.log(DPPY_DETERMINANT_TOLERANCE)
                        ),
                    }
                )
            model_components[class_index] = class_components
        components_by_model[model] = model_components
    return rows, components_by_model


def _copy_random_state(source: np.random.RandomState) -> np.random.RandomState:
    copied = np.random.RandomState()
    copied.set_state(source.get_state())
    return copied


def run_official_code_sampler(
    data: Mapping[str, object], model: str
) -> Dict[str, object]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    clean_targets = target_probabilities(data, model, "clean")
    random_state = np.random.RandomState(OFFICIAL_SEED)
    first_global: List[np.ndarray] = []
    last_global: List[np.ndarray] = []
    uniform_global: List[np.ndarray] = []
    class_rows: List[Dict[str, object]] = []
    total_seconds = 0.0
    for class_index in CLASS_ORDER:
        global_indices = np.flatnonzero(labels == class_index)
        count = int(global_indices.size)
        if count <= K_PER_CLASS:
            selected_local = np.arange(count, dtype=np.int64)
            first_local = selected_local
            last_local = selected_local
            uniform_local = selected_local
            chain_length = 1
            accepted_exchanges = 0
            elapsed = 0.0
        else:
            class_probabilities = clean_targets[global_indices]
            matrix = dense_kernel(class_probabilities, "code_n2")
            reference_state = _copy_random_state(random_state)
            uniform_local = np.asarray(
                reference_state.choice(count, size=K_PER_CLASS, replace=False),
                dtype=np.int64,
            )
            dpp = FiniteDPP("likelihood", L=matrix)
            start = time.perf_counter()
            returned = dpp.sample_mcmc_k_dpp(
                size=K_PER_CLASS, random_state=random_state
            )
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            chain = [np.asarray(value, dtype=np.int64) for value in dpp.list_of_samples[0]]
            first_local = chain[0]
            last_local = np.asarray(returned, dtype=np.int64)
            chain_length = len(chain)
            accepted_exchanges = sum(
                set(previous.tolist()) != set(current.tolist())
                for previous, current in zip(chain, chain[1:])
            )
        first_global.append(global_indices[first_local])
        last_global.append(global_indices[last_local])
        uniform_global.append(global_indices[uniform_local])
        class_rows.append(
            {
                "model": model,
                "class_index": class_index,
                "class_count": count,
                "k": int(first_local.size),
                "chain_length": chain_length,
                "accepted_exchanges": accepted_exchanges,
                "first_equals_uniform_initialization": bool(
                    np.array_equal(np.sort(first_local), np.sort(uniform_local))
                ),
                "first_last_jaccard": jaccard(first_local, last_local),
                "elapsed_seconds": elapsed,
            }
        )
    return {
        "model": model,
        "first_indices": np.sort(np.concatenate(first_global)),
        "last_indices": np.sort(np.concatenate(last_global)),
        "uniform_indices": np.sort(np.concatenate(uniform_global)),
        "class_rows": class_rows,
        "elapsed_seconds": total_seconds,
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.asarray(left, dtype=np.int64).tolist())
    right_set = set(np.asarray(right, dtype=np.int64).tolist())
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def balanced_random_subset(data: Mapping[str, object], seed: int) -> np.ndarray:
    labels = np.asarray(data["labels"], dtype=np.int64)
    random_state = np.random.RandomState(int(seed))
    selected: List[np.ndarray] = []
    for class_index in CLASS_ORDER:
        global_indices = np.flatnonzero(labels == class_index)
        if global_indices.size <= K_PER_CLASS:
            selected.append(global_indices)
        else:
            local = random_state.choice(
                global_indices.size, size=K_PER_CLASS, replace=False
            )
            selected.append(global_indices[np.asarray(local, dtype=np.int64)])
    return np.sort(np.concatenate(selected))


def evaluate_subset(
    data: Mapping[str, object], model: str, indices: np.ndarray
) -> Dict[str, object]:
    selected = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    folds = np.asarray(data["folds"], dtype=np.int64)
    sources = np.asarray(data["sources"], dtype=object)
    if selected.ndim != 1:
        raise ValueError("Selected indices must be a vector")
    duplicate_count = int(selected.size - np.unique(selected).size)
    result: Dict[str, object] = {
        "selected_rows": int(selected.size),
        "duplicate_samples": duplicate_count,
        "unique_sources": int(np.unique(sources[selected]).size),
        "class1_rows": int((labels[selected] == FOCUS_CLASS).sum()),
    }
    for class_index in range(5):
        result[f"class_{class_index}_rows"] = int(
            (labels[selected] == class_index).sum()
        )
    for fold in range(5):
        result[f"fold_{fold}_rows"] = int((folds[selected] == fold).sum())

    restricted_mask = np.isin(labels[selected], RESTRICTED_FOCUS_SOURCES)
    for condition in EXPECTED_CONDITIONS:
        probabilities = np.asarray(data["probabilities"][model][condition])
        selected_probabilities = probabilities[selected]
        selected_targets = labels[selected]
        selected_target_probabilities = selected_probabilities[
            np.arange(selected.size), selected_targets
        ]
        predictions = np.argmax(selected_probabilities, axis=1)
        result[f"{condition}_mean_target_probability"] = float(
            selected_target_probabilities.mean()
        )
        result[f"{condition}_mean_self_information"] = float(
            (-np.log(np.clip(selected_target_probabilities, 1e-12, 1.0))).mean()
        )
        result[f"{condition}_wrong_count"] = int(
            (predictions != selected_targets).sum()
        )
        result[f"{condition}_restricted_focus_fp"] = int(
            (restricted_mask & (predictions == FOCUS_CLASS)).sum()
        )
        for fold in range(5):
            result[f"{condition}_fold_{fold}_restricted_focus_fp"] = int(
                (
                    restricted_mask
                    & (predictions == FOCUS_CLASS)
                    & (folds[selected] == fold)
                ).sum()
            )
    return result


def _local_subset(
    labels: np.ndarray, global_indices: np.ndarray, class_index: int
) -> np.ndarray:
    class_global = np.flatnonzero(labels == class_index)
    selected_global = global_indices[labels[global_indices] == class_index]
    local = np.searchsorted(class_global, selected_global)
    if not np.array_equal(class_global[local], selected_global):
        raise ValueError("Global-to-local subset conversion failed")
    return local.astype(np.int64)


def aggregate_logdet(
    labels: np.ndarray,
    indices: np.ndarray,
    components: Mapping[int, Mapping[str, Mapping[str, object]]],
    mode: str,
) -> float:
    return float(
        sum(
            subset_log_determinant(
                components[class_index][mode],
                _local_subset(labels, indices, class_index),
            )
            for class_index in range(5)
        )
    )


def quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Quantiles require finite observations")
    return {
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def build_null_bank(
    data: Mapping[str, object],
    components_by_model: Mapping[str, Mapping[int, Mapping[str, Mapping[str, object]]]],
) -> Tuple[List[Dict[str, object]], Dict[str, List[np.ndarray]]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    rows: List[Dict[str, object]] = []
    selections: Dict[str, List[np.ndarray]] = {model: [] for model in MODEL_PREFIXES}
    shared_subsets = [
        balanced_random_subset(data, NULL_SEED_BASE + replication)
        for replication in range(NULL_PROPOSALS)
    ]
    for model in MODEL_PREFIXES:
        for replication, subset in enumerate(shared_subsets):
            metrics = evaluate_subset(data, model, subset)
            row = {
                "model": model,
                "replication": replication,
                "seed": NULL_SEED_BASE + replication,
                "paper_logdet": aggregate_logdet(
                    labels, subset, components_by_model[model], "paper_n"
                ),
                "code_logdet": aggregate_logdet(
                    labels, subset, components_by_model[model], "code_n2"
                ),
                **metrics,
            }
            rows.append(row)
            selections[model].append(subset)
    return rows, selections


def summarize_null_bank(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_names = [
        "paper_logdet",
        "code_logdet",
        "unique_sources",
        *(f"fold_{fold}_rows" for fold in range(5)),
        *(
            f"{condition}_{suffix}"
            for condition in EXPECTED_CONDITIONS
            for suffix in (
                "mean_target_probability",
                "mean_self_information",
                "wrong_count",
                "restricted_focus_fp",
            )
        ),
        *(f"clean_fold_{fold}_restricted_focus_fp" for fold in range(5)),
    ]
    for model in MODEL_PREFIXES:
        model_rows = [row for row in rows if row["model"] == model]
        summary[model] = {
            metric: quantiles([float(row[metric]) for row in model_rows])
            for metric in metric_names
        }
        paper = np.asarray([float(row["paper_logdet"]) for row in model_rows])
        code = np.asarray([float(row["code_logdet"]) for row in model_rows])
        information = np.asarray(
            [float(row["clean_mean_self_information"]) for row in model_rows]
        )
        paper_correlation = spearmanr(paper, information).statistic
        code_correlation = spearmanr(code, information).statistic
        summary[model]["derived"] = {
            "paper_q95_q05_determinant_odds": float(
                math.exp(min(700.0, np.quantile(paper, 0.95) - np.quantile(paper, 0.05)))
            ),
            "code_q95_q05_determinant_odds": float(
                math.exp(min(700.0, np.quantile(code, 0.95) - np.quantile(code, 0.05)))
            ),
            "paper_logdet_self_information_spearman": float(paper_correlation),
            "code_logdet_self_information_spearman": float(code_correlation),
        }
    return summary


def _selected_rows(
    data: Mapping[str, object],
    model: str,
    method: str,
    indices: np.ndarray,
) -> Iterable[Dict[str, object]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    folds = np.asarray(data["folds"], dtype=np.int64)
    sources = np.asarray(data["sources"], dtype=object)
    paths = np.asarray(data["image_paths"], dtype=object)
    for index in np.asarray(indices, dtype=np.int64):
        row: Dict[str, object] = {
            "model": model,
            "method": method,
            "sample_index": int(index),
            "source_stem": str(sources[index]),
            "image_path": str(paths[index]),
            "fold": int(folds[index]),
            "target_index": int(labels[index]),
        }
        for condition in EXPECTED_CONDITIONS:
            probabilities = np.asarray(data["probabilities"][model][condition])[index]
            target_probability = float(probabilities[labels[index]])
            prediction = int(np.argmax(probabilities))
            row[f"{condition}_prediction"] = prediction
            row[f"{condition}_target_probability"] = target_probability
            row[f"{condition}_self_information"] = float(
                -math.log(max(1e-12, target_probability))
            )
            row[f"{condition}_restricted_focus_fp"] = bool(
                labels[index] in RESTRICTED_FOCUS_SOURCES
                and prediction == FOCUS_CLASS
            )
        yield row


def _fold_rows(
    data: Mapping[str, object],
    model: str,
    metrics: Mapping[str, object],
    null_summary: Mapping[str, Mapping[str, float]],
) -> Iterable[Dict[str, object]]:
    for fold in range(5):
        bounds = null_summary[f"fold_{fold}_rows"]
        yield {
            "model": model,
            "fold": fold,
            "official_first_rows": int(metrics[f"fold_{fold}_rows"]),
            "random_q05_rows": bounds["q05"],
            "random_q50_rows": bounds["q50"],
            "random_q95_rows": bounds["q95"],
            "inside_random_envelope": bool(
                bounds["q05"]
                <= float(metrics[f"fold_{fold}_rows"])
                <= bounds["q95"]
            ),
            "official_clean_restricted_focus_fp": int(
                metrics[f"clean_fold_{fold}_restricted_focus_fp"]
            ),
            "random_q05_clean_restricted_focus_fp": null_summary[
                f"clean_fold_{fold}_restricted_focus_fp"
            ]["q05"],
            "random_q50_clean_restricted_focus_fp": null_summary[
                f"clean_fold_{fold}_restricted_focus_fp"
            ]["q50"],
            "random_q95_clean_restricted_focus_fp": null_summary[
                f"clean_fold_{fold}_restricted_focus_fp"
            ]["q95"],
            "clean_restricted_focus_fp_above_random_q95": bool(
                float(metrics[f"clean_fold_{fold}_restricted_focus_fp"])
                > null_summary[f"clean_fold_{fold}_restricted_focus_fp"]["q95"]
            ),
        }


def assess_gate(
    *,
    equation_audit: Mapping[str, object],
    kernel_rows: Sequence[Mapping[str, object]],
    official: Mapping[str, Mapping[str, object]],
    official_metrics: Mapping[str, Mapping[str, object]],
    null_summary: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Dict[str, object]:
    paper_majority = [
        row
        for row in kernel_rows
        if row["kernel_mode"] == "paper_n"
        and int(row["class_count"]) > K_PER_CLASS
    ]
    paper_initialization_feasible = all(
        bool(row["dppy_initialization_possible_at_tol"])
        for row in paper_majority
    )
    official_first_nonrandom = all(
        not np.array_equal(
            np.asarray(official[model]["first_indices"]),
            np.asarray(official[model]["uniform_indices"]),
        )
        for model in MODEL_PREFIXES
    )
    official_hardness: Dict[str, Dict[str, bool]] = {}
    for model in MODEL_PREFIXES:
        metrics = official_metrics[model]
        model_null = null_summary[model]
        official_hardness[model] = {}
        for condition in EXPECTED_CONDITIONS:
            information = float(metrics[f"{condition}_mean_self_information"])
            restricted = float(metrics[f"{condition}_restricted_focus_fp"])
            official_hardness[model][condition] = bool(
                information
                > model_null[f"{condition}_mean_self_information"]["q95"]
                and restricted
                > model_null[f"{condition}_restricted_focus_fp"]["q95"]
            )

    checks = {
        "input_train_only_integrity": True,
        "rank_one_dense_equation_parity": bool(equation_audit["passed"]),
        "all_kernel_spectral_bounds_valid": all(
            float(row["psd_eigenvalue_lower_bound"]) > 0.0
            and float(row["row_stochastic_eigenvalue_max"]) <= 1.0
            for row in kernel_rows
        ),
        "official_default_changes_majority_selection": False,
        "official_first_state_is_not_uniform_initialization": official_first_nonrandom,
        "paper_and_code_kernel_scale_agree": False,
        "paper_kernel_dppy_initialization_feasible": paper_initialization_feasible,
        "code_kernel_determinant_odds_ge_1p10": all(
            null_summary[model]["derived"][
                "code_q95_q05_determinant_odds"
            ]
            >= CODE_ODDS_GATE
            for model in MODEL_PREFIXES
        ),
        "official_selection_has_exact_cardinality_and_class1_retention": all(
            int(official_metrics[model]["selected_rows"]) == 5 * K_PER_CLASS
            and int(official_metrics[model]["class1_rows"]) == K_PER_CLASS
            and int(official_metrics[model]["duplicate_samples"]) == 0
            for model in MODEL_PREFIXES
        ),
        "official_source_coverage_above_random_q05": all(
            float(official_metrics[model]["unique_sources"])
            >= null_summary[model]["unique_sources"]["q05"]
            for model in MODEL_PREFIXES
        ),
        "official_fold_composition_inside_random_envelope": all(
            null_summary[model][f"fold_{fold}_rows"]["q05"]
            <= float(official_metrics[model][f"fold_{fold}_rows"])
            <= null_summary[model][f"fold_{fold}_rows"]["q95"]
            for model in MODEL_PREFIXES
            for fold in range(5)
        ),
        "official_clean_and_three_condition_hardness_above_random_q95": all(
            official_hardness[model]["clean"]
            and sum(official_hardness[model].values()) >= 3
            for model in MODEL_PREFIXES
        ),
        "deterministic_official_replay": all(
            bool(official[model]["deterministic_replay"])
            for model in MODEL_PREFIXES
        ),
        "test_unused": True,
        "raw_dataset_unmodified": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "official_condition_hardness": official_hardness,
        "trainer_integration_authorized": not failed,
        "validation_smoke_authorized": False,
        "test_permission": False,
        "full_train_permission": False,
    }


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# IP-DPP Train-Only Information Gate",
        "",
        f"Decision: **{'PASS' if summary['gate']['trainer_integration_authorized'] else 'REJECT'}**",
        "",
        "The official paper default retains all 9,215 TRKH rows. The local diagnostic",
        "uses k=541 only to explain the source mechanism; it cannot promote a trainer.",
        "",
        "| Source | Official first = uniform init | Code odds q95/q05 | Class-1 retained |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in MODEL_PREFIXES:
        item = summary["official_sampling"][model]
        metrics = summary["official_metrics"][model]
        odds = summary["null_summary"][model]["derived"][
            "code_q95_q05_determinant_odds"
        ]
        lines.append(
            f"| {model} | {str(item['first_equals_uniform_initialization']).lower()} "
            f"| {odds:.6f} | {metrics['class1_rows']} / {K_PER_CLASS} |"
        )
    lines.extend(
        [
            "",
            "Failed checks:",
            *[f"- `{name}`" for name in summary["gate"]["failed_checks"]],
            "",
            "No validation/test data or raw-data modification was used.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    source_audit = verify_locked_sources(args)
    data = load_train_only_cache(Path(args.cache).resolve())
    preflight = {
        "mode": "train_only_ip_dpp_information_gate_preflight",
        "sources": source_audit,
        "rows": EXPECTED_ROWS,
        "source_groups": data["source_groups"],
        "class_counts": data["class_counts"],
        "fold_counts": data["fold_counts"],
        "conditions": data["conditions"],
        "test_used": False,
        "raw_dataset_modified": False,
    }
    if bool(args.preflight_only):
        print(json.dumps(preflight, indent=2))
        return preflight
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --preflight-only is used")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    equation_audit = verify_rank_one_equations()
    kernel_rows, components_by_model = build_kernel_audit(data)
    null_rows, null_selections = build_null_bank(data, components_by_model)
    null_summary = summarize_null_bank(null_rows)

    official: Dict[str, Dict[str, object]] = {}
    official_metrics: Dict[str, Dict[str, object]] = {}
    selection_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    proposal_best: Dict[str, Dict[str, object]] = {}
    for model in MODEL_PREFIXES:
        primary = run_official_code_sampler(data, model)
        replay = run_official_code_sampler(data, model)
        deterministic_replay = bool(
            np.array_equal(primary["first_indices"], replay["first_indices"])
            and np.array_equal(primary["last_indices"], replay["last_indices"])
        )
        first_equals_uniform = bool(
            np.array_equal(primary["first_indices"], primary["uniform_indices"])
        )
        metrics = evaluate_subset(data, model, primary["first_indices"])
        official_metrics[model] = metrics
        official[model] = {
            "first_indices": primary["first_indices"],
            "last_indices": primary["last_indices"],
            "uniform_indices": primary["uniform_indices"],
            "class_rows": primary["class_rows"],
            "elapsed_seconds": primary["elapsed_seconds"],
            "first_equals_uniform_initialization": first_equals_uniform,
            "first_indices_sha256": _index_sha256(primary["first_indices"]),
            "last_indices_sha256": _index_sha256(primary["last_indices"]),
            "uniform_indices_sha256": _index_sha256(primary["uniform_indices"]),
            "replay_first_indices_sha256": _index_sha256(
                replay["first_indices"]
            ),
            "replay_last_indices_sha256": _index_sha256(replay["last_indices"]),
            "first_last_jaccard": jaccard(
                primary["first_indices"], primary["last_indices"]
            ),
            "deterministic_replay": deterministic_replay,
        }
        fold_rows.extend(_fold_rows(data, model, metrics, null_summary[model]))

        model_null_rows = [row for row in null_rows if row["model"] == model]
        paper_best_index = int(
            np.argmax([float(row["paper_logdet"]) for row in model_null_rows])
        )
        code_best_index = int(
            np.argmax([float(row["code_logdet"]) for row in model_null_rows])
        )
        paper_subset = null_selections[model][paper_best_index]
        code_subset = null_selections[model][code_best_index]
        proposal_best[model] = {
            "paper": {
                "replication": paper_best_index,
                "seed": NULL_SEED_BASE + paper_best_index,
                "metrics": evaluate_subset(data, model, paper_subset),
                "logdet": float(model_null_rows[paper_best_index]["paper_logdet"]),
            },
            "code": {
                "replication": code_best_index,
                "seed": NULL_SEED_BASE + code_best_index,
                "metrics": evaluate_subset(data, model, code_subset),
                "logdet": float(model_null_rows[code_best_index]["code_logdet"]),
            },
            "paper_code_jaccard": jaccard(paper_subset, code_subset),
            "warning": "proposal_bank_maximum_is_not_an_ip_dpp_sample",
        }
        for method, indices in (
            ("official_first", primary["first_indices"]),
            ("official_last_default10", primary["last_indices"]),
            ("paper_best_random_proposal_diagnostic", paper_subset),
            ("code_best_random_proposal_diagnostic", code_subset),
        ):
            selection_rows.extend(_selected_rows(data, model, method, indices))

    gate = assess_gate(
        equation_audit=equation_audit,
        kernel_rows=kernel_rows,
        official=official,
        official_metrics=official_metrics,
        null_summary=null_summary,
    )
    serializable_official = {
        model: {
            key: value
            for key, value in values.items()
            if key not in {"first_indices", "last_indices", "uniform_indices"}
        }
        for model, values in official.items()
    }
    direct_default_k = 10 * min(EXPECTED_CLASS_COUNTS.values())
    summary: Dict[str, object] = {
        "mode": "train_only_ip_dpp_information_gate",
        "status": "completed",
        "sources": source_audit,
        "data_integrity": {
            "rows": EXPECTED_ROWS,
            "source_groups": data["source_groups"],
            "class_counts": data["class_counts"],
            "fold_counts": data["fold_counts"],
            "source_fold_overlap": data["source_fold_overlap"],
            "conditions": data["conditions"],
            "rows_per_condition": data["rows_per_condition"],
            "all_paths_train_only": True,
        },
        "locked_design": {
            "k_per_class": K_PER_CLASS,
            "official_seed": OFFICIAL_SEED,
            "class_order": list(CLASS_ORDER),
            "null_seed_base": NULL_SEED_BASE,
            "null_proposals": NULL_PROPOSALS,
        },
        "direct_official_default": {
            "k_equals_10_n_min": direct_default_k,
            "effective_class_counts": {
                class_index: min(direct_default_k, count)
                for class_index, count in EXPECTED_CLASS_COUNTS.items()
            },
            "selected_rows": sum(EXPECTED_CLASS_COUNTS.values()),
            "changes_any_majority_selection": False,
        },
        "source_discrepancies": {
            "paper_off_diagonal_scale": "1/N",
            "official_code_off_diagonal_scale": "1/N^2",
            "scales_agree": False,
            "official_extraction": "list_of_samples[0][0]",
            "dppy_first_chain_state": "uniform_random_initialization",
            "dppy_default_chain_length": 10,
        },
        "equation_audit": equation_audit,
        "kernel_audit": kernel_rows,
        "official_sampling": serializable_official,
        "official_metrics": official_metrics,
        "null_summary": null_summary,
        "proposal_bank_best_diagnostic": proposal_best,
        "gate": gate,
        "test_used": False,
        "raw_dataset_modified": False,
    }

    _write_csv(output_dir / "kernel_by_class.csv", kernel_rows)
    _write_csv(output_dir / "null_distribution.csv", null_rows)
    _write_csv(output_dir / "official_fold_composition.csv", fold_rows)
    _write_csv(output_dir / "selected_rows.csv", selection_rows)
    _write_json(
        output_dir / "locked_protocol.json",
        {
            "mode": "ip_dpp_information_gate_locked_protocol",
            "protocol_sha256": LOCKED_SHA256["protocol"],
            "cache_sha256": LOCKED_SHA256["cache"],
            "official_source_sha256": LOCKED_SHA256["official_source"],
            "official_commit": OFFICIAL_COMMIT,
            "constants": summary["locked_design"],
            "test_used": False,
        },
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    _write_artifact_manifest(
        output_dir, mode="ip_dpp_information_gate_evidence_manifest"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "trainer_integration_authorized": gate[
                    "trainer_integration_authorized"
                ],
                "failed_checks": gate["failed_checks"],
                "test_used": False,
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()

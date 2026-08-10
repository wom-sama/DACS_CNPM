from __future__ import annotations

import hashlib
import math
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from trkh.tools.pair_surface_ddf_v2_engine import (
    LOCKED_NONWRAP_OFFSETS,
    PRIMARY_SEED,
    RIVALS,
    ROLE_NAMES,
    ROWS,
    SCORE_NAMES,
    array_sha256,
    json_sha256,
)

NO_ACTION = "NO_ACTION"
COMPONENT_BOOTSTRAP_REPLICATES = 2000
LOCKED_COMPONENT_COUNT = 158
CALIBRATOR_C = 0.1
CALIBRATOR_MAX_ITER = 1000
CALIBRATOR_TOLERANCE = 1e-8

# Exact v2R2 A0 identities.  The integer arrays are canonicalized to explicit
# little-endian int64 before hashing.  The pair digest is json_sha256 over the
# ordered ``[[sample_index, held_fold], ...]`` list.
EXACT_A0_SAMPLE_INDICES_I64_SHA256 = (
    "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05"
)
EXACT_A0_TARGETS_I64_SHA256 = (
    "15c43ecc7335c7a7febc4e0fbf622df7ad593fc52e5d9f14e5cc27acc16fc000"
)
EXACT_A0_HELD_FOLDS_I64_SHA256 = (
    "924a3e98e5ef1bd814511675951d83ee79ff9788894762b4688fb93fabbea255"
)
EXACT_A0_SAMPLE_FOLD_PAIRS_JSON_SHA256 = (
    "fd61cd1c20c9ca66c113db1ec65d25c8e0ea8c4ec851f80f3c15516f993921df"
)
EXACT_COMPONENT_ORDER_JSON_SHA256 = (
    "e10e1045e6ede63ade0cb436b684be7a6ab377d0d27f74480ac8721756045ccb"
)
EXACT_BBOX_UNUSABLE_RECORDS_JSON_SHA256 = (
    "cbe5d09bc0ef6305a6610297ddc1dd747e1e018fd10cdc87d2ed87d224b91534"
)
EXACT_CIDT_SAMPLE_INDICES_I64_SHA256 = (
    "44a4e9101166892870890c4c24e54a1a240b70f9706ddd12603d85dda68d5bc7"
)
EXACT_CIDT_TARGETS_I64_SHA256 = (
    "0b54f448d890b9032dcc336f5a1798927c19bf6b35b2a094bf318208e4785e7d"
)
EXACT_CIDT_BASELINE_PREDICTIONS_I64_SHA256 = (
    "71295e8183f3b705992c16210da69470531b546108b65b65987e2fb72313b2d5"
)

UNION_SCORE_AXES = (
    "K",
    "KU",
    "KUH",
    "KUP",
    "static_KUP",
    "spatial_KUP",
    "channel_KUP",
    "repeat_KUP",
)
PAIR_SCORE_AXES = ("K", "KP", "static_KP")
CAUSAL_SCORE_AXES = (
    "clean",
    "support_matched_self",
    "cross_fold_substitution",
    "nonwrap_displacement",
    "spatial_neutral",
    "channel_neutral",
)
ACTION_ROLE_AXES = ("primary", "repeat")
TRACE_ROLE_COUNT = 5
TRACE_FOLD_COUNT = 5
TRACE_STEPS = 160

def _finite_float64(
    value: np.ndarray | Sequence[float],
    *,
    name: str,
    ndim: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains a non-finite value")
    return array

def _strictly_increasing_int64(
    value: np.ndarray | Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size == 0 or not bool(np.all(array[1:] > array[:-1])):
        raise ValueError(f"{name} must be non-empty, unique and ascending")
    return array

def keeper_predictions(keeper_probabilities: np.ndarray) -> np.ndarray:
    probabilities = _validate_keeper_probabilities(keeper_probabilities)
    return np.argmax(probabilities, axis=1).astype(np.int64, copy=False)

def keeper_selected_pair_rival(keeper_probabilities: np.ndarray) -> np.ndarray:
    probabilities = _validate_keeper_probabilities(keeper_probabilities)
    rivals = np.asarray(RIVALS, dtype=np.int64)
    positions = np.argmax(probabilities[:, rivals], axis=1)
    return rivals[positions]

def keeper_replacement_rival(keeper_probabilities: np.ndarray) -> np.ndarray:
    probabilities = _validate_keeper_probabilities(keeper_probabilities)
    values = probabilities.copy()
    values[:, 1] = -np.inf
    return np.argmax(values, axis=1).astype(np.int64, copy=False)

def _validate_keeper_probabilities(value: np.ndarray) -> np.ndarray:
    probabilities = _finite_float64(value, name="keeper probabilities", ndim=2)
    if probabilities.shape[1] != 5:
        raise ValueError("Keeper probabilities must have shape [N, 5]")
    if bool((probabilities < 0.0).any() or (probabilities > 1.0).any()):
        raise ValueError("Keeper probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("Keeper probability rows must sum to one")
    return probabilities

@dataclass(frozen=True)
class DonorCovariates:
    sample_indices: np.ndarray
    component_ids: Tuple[str, ...]
    keeper_probabilities: np.ndarray
    valid32: np.ndarray
    valid16: np.ndarray
    bbox_valid16: np.ndarray
    strongest_rival: np.ndarray
    bbox_area_rank: np.ndarray
    valid_fraction_rank: np.ndarray

def _validate_donor_covariates(
    value: DonorCovariates,
    *,
    name: str,
) -> DonorCovariates:
    indices = _strictly_increasing_int64(value.sample_indices, name=f"{name} indices")
    rows = int(indices.size)
    components = tuple(str(item) for item in value.component_ids)
    if len(components) != rows or any(
        re.fullmatch(r"[0-9a-f]{64}", item) is None for item in components
    ):
        raise ValueError(f"{name} component IDs are not aligned")
    valid32 = np.asarray(value.valid32, dtype=np.bool_)
    valid16 = np.asarray(value.valid16, dtype=np.bool_)
    bbox = np.asarray(value.bbox_valid16, dtype=np.bool_)
    if valid32.shape != (rows, 32, 32):
        raise ValueError(f"{name} valid32 must have shape [N,32,32]")
    if valid16.shape != (rows, 16, 16):
        raise ValueError(f"{name} valid16 must have shape [N,16,16]")
    if bbox.shape != (rows, 16, 16):
        raise ValueError(f"{name} bbox-valid16 must have shape [N,16,16]")
    if bool((bbox & ~valid16).any()):
        raise ValueError(f"{name} bbox-valid16 is not a subset of valid16")
    probabilities = _validate_keeper_probabilities(value.keeper_probabilities)
    if probabilities.shape[0] != rows:
        raise ValueError(f"{name} keeper probabilities are not aligned")
    rival = np.asarray(value.strongest_rival, dtype=np.int64).reshape(-1)
    bbox_rank = np.asarray(value.bbox_area_rank, dtype=np.int64).reshape(-1)
    valid_rank = np.asarray(value.valid_fraction_rank, dtype=np.int64).reshape(-1)
    expected_rival = keeper_selected_pair_rival(probabilities)
    if rival.shape != (rows,) or not np.array_equal(rival, expected_rival):
        raise ValueError(f"{name} strongest rival does not match keeper probabilities")
    if bbox_rank.shape != (rows,) or valid_rank.shape != (rows,):
        raise ValueError(f"{name} rank vectors are not aligned")
    if bool((bbox_rank < 0).any() or (valid_rank < 0).any()):
        raise ValueError(f"{name} ranks must be non-negative")
    bbox_counts = bbox.sum(axis=(1, 2), dtype=np.int64)
    valid_counts = valid16.sum(axis=(1, 2), dtype=np.int64)
    expected_bbox_rank = _canonical_ordinal_ranks(bbox_counts, indices)
    expected_valid_rank = _canonical_ordinal_ranks(valid_counts, indices)
    if not np.array_equal(bbox_rank, expected_bbox_rank) or not np.array_equal(
        valid_rank,
        expected_valid_rank,
    ):
        raise ValueError(f"{name} stored ranks do not match canonical geometry ranks")
    return DonorCovariates(
        sample_indices=indices,
        component_ids=components,
        keeper_probabilities=probabilities.copy(),
        valid32=np.ascontiguousarray(valid32),
        valid16=np.ascontiguousarray(valid16),
        bbox_valid16=np.ascontiguousarray(bbox),
        strongest_rival=rival,
        bbox_area_rank=bbox_rank,
        valid_fraction_rank=valid_rank,
    )

def _canonical_ordinal_ranks(
    values: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(values, dtype=np.int64).reshape(-1)
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if observed.shape != indices.shape:
        raise ValueError("Ordinal-rank values and sample indices are not aligned")
    order = np.lexsort((indices, observed))
    ranks = np.empty(observed.size, dtype=np.int64)
    ranks[order] = np.arange(observed.size, dtype=np.int64)
    return ranks

def build_target_blind_donor_cost_vectors(
    victims: DonorCovariates,
    donors: DonorCovariates,
) -> np.ndarray:
    """Return six target-free lexicographic edge costs.

    The dimensions are valid32 XOR, valid16 XOR, bbox-valid16 XOR,
    keeper-rival mismatch, bbox-area-rank distance and valid-fraction-rank
    distance.  Targets are intentionally absent from this API.
    """

    left = _validate_donor_covariates(victims, name="victim")
    right = _validate_donor_covariates(donors, name="donor")
    if set(left.sample_indices.tolist()) & set(right.sample_indices.tolist()):
        raise ValueError("Held victim and calibration donor indices overlap")
    if set(left.component_ids) & set(right.component_ids):
        raise ValueError("Held and calibration components overlap")
    costs = np.empty(
        (left.sample_indices.size, right.sample_indices.size, 6),
        dtype=np.int64,
    )
    for victim in range(left.sample_indices.size):
        for donor in range(right.sample_indices.size):
            costs[victim, donor] = (
                np.count_nonzero(left.valid32[victim] ^ right.valid32[donor]),
                np.count_nonzero(left.valid16[victim] ^ right.valid16[donor]),
                np.count_nonzero(
                    left.bbox_valid16[victim] ^ right.bbox_valid16[donor]
                ),
                int(left.strongest_rival[victim] != right.strongest_rival[donor]),
                abs(
                    int(left.bbox_area_rank[victim])
                    - int(right.bbox_area_rank[donor])
                ),
                abs(
                    int(left.valid_fraction_rank[victim])
                    - int(right.valid_fraction_rank[donor])
                ),
            )
    return costs

def lexicographic_cost_coefficients(
    maxima: Sequence[int],
    *,
    assignments: int,
) -> Tuple[int, ...]:
    maximum = [int(item) for item in maxima]
    if assignments <= 0 or not maximum or any(item < 0 for item in maximum):
        raise ValueError("Cost maxima and assignment count must be non-negative")
    coefficients = [0] * len(maximum)
    coefficients[-1] = 1
    for index in range(len(maximum) - 2, -1, -1):
        lower_total = assignments * sum(
            maximum[lower] * coefficients[lower]
            for lower in range(index + 1, len(maximum))
        )
        coefficients[index] = lower_total + 1
    return tuple(coefficients)

def _hungarian_rectangular_exact(
    costs: Sequence[Sequence[int]],
) -> Tuple[List[int], int]:
    """Exact rectangular Hungarian assignment over arbitrary-size integers."""

    rows = len(costs)
    if rows == 0:
        return [], 0
    columns = len(costs[0])
    if columns < rows or any(len(row) != columns for row in costs):
        raise ValueError("Hungarian costs must be rectangular with rows <= columns")
    u = [0] * (rows + 1)
    v = [0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum: List[Optional[int]] = [None] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta: Optional[int] = None
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = int(costs[row0 - 1][column - 1]) - u[row0] - v[column]
                if minimum[column] is None or reduced < minimum[column]:
                    minimum[column] = reduced
                    way[column] = column0
                candidate = minimum[column]
                assert candidate is not None
                if delta is None or candidate < delta or (
                    candidate == delta and column < column1
                ):
                    delta = candidate
                    column1 = column
            if delta is None:
                raise RuntimeError("Exact Hungarian assignment is infeasible")
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                elif minimum[column] is not None:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("Exact Hungarian assignment left an unmatched row")
    total = sum(int(costs[row][column]) for row, column in enumerate(assignment))
    return assignment, total

@dataclass(frozen=True)
class DonorAssignment:
    victim_sample_indices: np.ndarray
    donor_sample_indices: np.ndarray
    donor_positions: np.ndarray
    donor_use_counts: np.ndarray
    per_priority_totals: np.ndarray
    priority_coefficients: Tuple[int, ...]
    primary_scalar_total: int
    lexicographic_donor_vector: Tuple[int, ...]

def assign_capacity_balanced_donors(
    victims: DonorCovariates,
    donors: DonorCovariates,
) -> DonorAssignment:
    left = _validate_donor_covariates(victims, name="victim")
    right = _validate_donor_covariates(donors, name="donor")
    vectors = build_target_blind_donor_cost_vectors(left, right)
    victim_count, donor_count, priorities = vectors.shape
    lower, extra = divmod(victim_count, donor_count)
    upper = lower + (1 if extra else 0)
    maxima = vectors.max(axis=(0, 1)).astype(np.int64)
    coefficients = lexicographic_cost_coefficients(
        maxima.tolist(),
        assignments=victim_count,
    )
    scalar = [
        [
            sum(
                int(vectors[victim, donor, priority]) * coefficients[priority]
                for priority in range(priorities)
            )
            for donor in range(donor_count)
        ]
        for victim in range(victim_count)
    ]

    # One base-(D+1) digit per canonical victim makes the donor vector itself
    # the exact secondary objective, without floating perturbations.
    tie_base = donor_count + 1
    tie_factor = tie_base ** victim_count
    tie_powers = [
        tie_base ** (victim_count - 1 - victim)
        for victim in range(victim_count)
    ]
    edge = [
        [
            scalar[victim][donor] * tie_factor
            + donor * tie_powers[victim]
            for donor in range(donor_count)
        ]
        for victim in range(victim_count)
    ]
    edge_min = min(min(row) for row in edge)
    edge_max = max(max(row) for row in edge)
    mandatory_bonus = victim_count * (edge_max - edge_min) + 1

    slot_donors: List[int] = []
    slot_mandatory: List[bool] = []
    for donor in range(donor_count):
        for _ in range(lower):
            slot_donors.append(donor)
            slot_mandatory.append(True)
        if upper > lower:
            slot_donors.append(donor)
            slot_mandatory.append(False)
    costs = [
        [
            edge[victim][donor]
            - (mandatory_bonus if mandatory else 0)
            for donor, mandatory in zip(slot_donors, slot_mandatory)
        ]
        for victim in range(victim_count)
    ]
    slot_assignment, _ = _hungarian_rectangular_exact(costs)
    selected_slots = set(slot_assignment)
    mandatory_slots = {
        position for position, mandatory in enumerate(slot_mandatory) if mandatory
    }
    if not mandatory_slots.issubset(selected_slots):
        raise RuntimeError("Capacity assignment did not consume every lower-bound slot")
    donor_positions = np.asarray(
        [slot_donors[position] for position in slot_assignment],
        dtype=np.int64,
    )
    use_counts = np.bincount(donor_positions, minlength=donor_count).astype(
        np.int64,
        copy=False,
    )
    if int(use_counts.sum()) != victim_count or bool(
        ((use_counts < lower) | (use_counts > upper)).any()
    ):
        raise RuntimeError("Donor capacity bounds were not satisfied")
    if int(np.count_nonzero(use_counts == upper)) != extra and upper > lower:
        raise RuntimeError("The number of duplicated/unused donors is incorrect")
    totals = vectors[np.arange(victim_count), donor_positions].sum(axis=0)
    scalar_total = sum(
        int(totals[priority]) * coefficients[priority]
        for priority in range(priorities)
    )
    return DonorAssignment(
        victim_sample_indices=left.sample_indices.copy(),
        donor_sample_indices=right.sample_indices[donor_positions].copy(),
        donor_positions=donor_positions,
        donor_use_counts=use_counts,
        per_priority_totals=totals.astype(np.int64, copy=False),
        priority_coefficients=coefficients,
        primary_scalar_total=scalar_total,
        lexicographic_donor_vector=tuple(int(item) for item in donor_positions),
    )

def build_nonwrap_offset_mapping(
    sample_indices: np.ndarray | Sequence[int],
    *,
    fold: int,
) -> np.ndarray:
    indices = _strictly_increasing_int64(sample_indices, name="held sample indices")
    if not 0 <= int(fold) < 5:
        raise ValueError("Fold must lie in [0,5)")
    output = np.empty((indices.size, 2, 2), dtype=np.int64)
    position_by_sample = {int(sample): position for position, sample in enumerate(indices)}
    for block in range(2):
        ordered = sorted(
            (int(sample) for sample in indices),
            key=lambda sample: (
                hashlib.sha256(
                    f"{PRIMARY_SEED}|{int(fold)}|{block}|{sample}".encode("utf-8")
                ).hexdigest(),
                sample,
            ),
        )
        for order_position, sample in enumerate(ordered):
            output[position_by_sample[sample], block] = LOCKED_NONWRAP_OFFSETS[
                (order_position + int(fold) + 2 * block)
                % len(LOCKED_NONWRAP_OFFSETS)
            ]
    return output

def generate_invalid_fill_noise(
    *,
    rows: int = ROWS,
    seed: int = PRIMARY_SEED,
) -> np.ndarray:
    if rows <= 0:
        raise ValueError("Invalid-fill noise needs at least one row")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return rng.integers(
        0,
        256,
        size=(int(rows), 3, 256, 256),
        dtype=np.uint8,
        endpoint=False,
    )

def generate_component_bootstrap_draws(
    *,
    component_count: int = LOCKED_COMPONENT_COUNT,
    replicates: int = COMPONENT_BOOTSTRAP_REPLICATES,
    seed: int = PRIMARY_SEED,
) -> np.ndarray:
    if component_count <= 0 or replicates <= 0:
        raise ValueError("Bootstrap dimensions must be positive")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return rng.integers(
        0,
        int(component_count),
        size=(int(replicates), int(component_count)),
        dtype=np.int64,
        endpoint=False,
    )

def build_union_feature_sets(
    keeper_probabilities: np.ndarray,
    raw_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    probabilities = _validate_keeper_probabilities(keeper_probabilities)
    scores = _finite_float64(raw_scores, name="raw sidecar scores", ndim=2)
    if scores.shape != (probabilities.shape[0], 4):
        raise ValueError("Raw sidecar scores must have shape [N,4]")
    rival = keeper_selected_pair_rival(probabilities)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    strongest = np.maximum.reduce(
        (clipped[:, 0], clipped[:, 2], clipped[:, 4])
    )
    keeper_margin = np.log(clipped[:, 1]) - np.log(strongest)
    indicator2 = (rival == 2).astype(np.float64)
    indicator4 = (rival == 4).astype(np.float64)
    selected_scores = np.zeros((scores.shape[0], 3), dtype=np.float64)
    for position, rival_class in enumerate(RIVALS):
        selected_scores[:, position] = np.where(
            rival == rival_class,
            scores[:, position + 1],
            np.float64(0.0),
        )
    return {
        "K": keeper_margin.reshape(-1, 1),
        "KU": np.column_stack((keeper_margin, scores[:, 0])),
        "KUH": np.column_stack(
            (keeper_margin, scores[:, 0], indicator2, indicator4)
        ),
        "KUP": np.column_stack(
            (
                keeper_margin,
                scores[:, 0],
                indicator2,
                indicator4,
                selected_scores,
            )
        ),
    }

def build_pair_feature_sets(
    keeper_probabilities: np.ndarray,
    raw_scores: np.ndarray,
    *,
    rival: int,
) -> Dict[str, np.ndarray]:
    if int(rival) not in RIVALS:
        raise ValueError("Pair rival must be one of 0/2/4")
    probabilities = _validate_keeper_probabilities(keeper_probabilities)
    scores = _finite_float64(raw_scores, name="raw sidecar scores", ndim=2)
    if scores.shape != (probabilities.shape[0], 4):
        raise ValueError("Raw sidecar scores must have shape [N,4]")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    margin = np.log(clipped[:, 1]) - np.log(clipped[:, int(rival)])
    task_index = RIVALS.index(int(rival)) + 1
    return {
        f"K_{int(rival)}": margin.reshape(-1, 1),
        f"KP_{int(rival)}": np.column_stack((margin, scores[:, task_index])),
    }

@dataclass(frozen=True)
class CalibratorState:
    feature_names: Tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_var: np.ndarray
    scaler_scale: np.ndarray
    logistic_coef: np.ndarray
    logistic_intercept: np.ndarray
    classes: np.ndarray
    n_iter: np.ndarray

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        values = _finite_float64(features, name="calibrator features", ndim=2)
        if values.shape[1] != len(self.feature_names):
            raise ValueError("Calibrator feature width changed")
        standardized = (values - self.scaler_mean) / self.scaler_scale
        logits = standardized @ self.logistic_coef[0] + self.logistic_intercept[0]
        probabilities = expit(logits)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Calibrator produced non-finite probabilities")
        return np.asarray(probabilities, dtype=np.float64)

def fit_locked_calibrator(
    features: np.ndarray,
    labels: np.ndarray | Sequence[int],
    *,
    feature_names: Sequence[str],
) -> CalibratorState:
    values = _finite_float64(features, name="calibration features", ndim=2)
    names = tuple(str(item) for item in feature_names)
    if values.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Calibrator feature names are not aligned and unique")
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if targets.shape != (values.shape[0],) or not bool(np.isin(targets, (0, 1)).all()):
        raise ValueError("Calibration labels must be an aligned binary vector")
    if np.unique(targets).size != 2:
        raise ValueError("Calibration labels lack binary support")
    population_variance = values.var(axis=0, ddof=0, dtype=np.float64)
    if bool((population_variance <= 0.0).any()):
        raise ValueError("A calibrator feature has zero variance")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler = StandardScaler(with_mean=True, with_std=True)
        standardized = scaler.fit_transform(values)
        logistic = LogisticRegression(
            C=CALIBRATOR_C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=CALIBRATOR_MAX_ITER,
            tol=CALIBRATOR_TOLERANCE,
            random_state=PRIMARY_SEED,
        )
        logistic.fit(standardized, targets)
    arrays = (
        scaler.mean_,
        scaler.var_,
        scaler.scale_,
        logistic.coef_,
        logistic.intercept_,
        logistic.classes_,
        logistic.n_iter_,
    )
    if any(not np.isfinite(np.asarray(item)).all() for item in arrays):
        raise FloatingPointError("Calibrator state is non-finite")
    if bool((logistic.n_iter_ >= CALIBRATOR_MAX_ITER).any()):
        raise RuntimeError("Locked logistic calibrator did not converge")
    return CalibratorState(
        feature_names=names,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64).copy(),
        scaler_var=np.asarray(scaler.var_, dtype=np.float64).copy(),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64).copy(),
        logistic_coef=np.asarray(logistic.coef_, dtype=np.float64).copy(),
        logistic_intercept=np.asarray(logistic.intercept_, dtype=np.float64).copy(),
        classes=np.asarray(logistic.classes_, dtype=np.int64).copy(),
        n_iter=np.asarray(logistic.n_iter_, dtype=np.int64).copy(),
    )

@dataclass(frozen=True)
class ActionPolicy:
    threshold_token: str
    numeric_threshold: Optional[float]

@dataclass(frozen=True)
class ActionMetrics:
    changed: int
    corrections: int
    harms: int
    neutral_changes: int
    keeper_class1_tp: int
    retained_class1_tp: int
    tp_retention: float
    restricted_fp_denominator: int
    restricted_fp_rejected: int
    restricted_fp_rejection_rate: float
    correction_ratio: Optional[float]

@dataclass(frozen=True)
class ActionSelection:
    policy: ActionPolicy
    calibration_metrics: ActionMetrics

@dataclass(frozen=True)
class AppliedAction:
    keeper_predictions: np.ndarray
    replacement_predictions: np.ndarray
    final_predictions: np.ndarray
    changed: np.ndarray

def apply_action_policy(
    calibrated_probabilities: np.ndarray,
    keeper_probabilities: np.ndarray,
    policy: ActionPolicy,
) -> AppliedAction:
    candidate = _finite_float64(
        calibrated_probabilities,
        name="calibrated action probabilities",
        ndim=1,
    )
    if bool((candidate < 0.0).any() or (candidate > 1.0).any()):
        raise ValueError("Calibrated action probabilities must lie in [0,1]")
    keeper = _validate_keeper_probabilities(keeper_probabilities)
    if candidate.shape != (keeper.shape[0],):
        raise ValueError("Action probabilities are not row-aligned")
    original = keeper_predictions(keeper)
    replacement = keeper_replacement_rival(keeper)
    if policy.threshold_token == NO_ACTION:
        if policy.numeric_threshold is not None:
            raise ValueError("NO_ACTION cannot carry a numeric threshold")
        changed = np.zeros(candidate.shape[0], dtype=np.bool_)
    else:
        if policy.numeric_threshold is None or not math.isfinite(
            policy.numeric_threshold
        ):
            raise ValueError("Numeric action policy lacks a finite threshold")
        if policy.threshold_token != float(policy.numeric_threshold).hex():
            raise ValueError("Numeric action threshold token is not canonical")
        changed = (original == 1) & (candidate <= policy.numeric_threshold)
    final = original.copy()
    final[changed] = replacement[changed]
    return AppliedAction(
        keeper_predictions=original,
        replacement_predictions=replacement,
        final_predictions=final,
        changed=changed,
    )

def evaluate_applied_actions(
    applied: AppliedAction,
    targets: np.ndarray | Sequence[int],
) -> ActionMetrics:
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    rows = applied.final_predictions.shape[0]
    if truth.shape != (rows,):
        raise ValueError("Action targets are not row-aligned")
    if not bool(np.isin(truth, np.arange(5, dtype=np.int64)).all()):
        raise ValueError("Action targets must lie in classes 0..4")
    changed = np.asarray(applied.changed, dtype=np.bool_)
    original = np.asarray(applied.keeper_predictions, dtype=np.int64)
    replacement = np.asarray(applied.replacement_predictions, dtype=np.int64)
    final = np.asarray(applied.final_predictions, dtype=np.int64)
    if any(value.shape != (rows,) for value in (changed, original, replacement, final)):
        raise ValueError("Applied-action arrays are not aligned vectors")
    if not bool(
        np.isin(original, np.arange(5)).all()
        and np.isin(replacement, np.arange(5)).all()
        and np.isin(final, np.arange(5)).all()
    ):
        raise ValueError("Applied-action predictions must lie in classes 0..4")
    if bool((replacement == 1).any()):
        raise ValueError("Replacement predictions may never be class 1")
    if not np.array_equal(changed, final != original):
        raise ValueError("Applied-action changed mask is inconsistent")
    if not np.array_equal(final[~changed], original[~changed]) or not np.array_equal(
        final[changed],
        replacement[changed],
    ):
        raise ValueError("Applied-action final predictions are inconsistent")
    if bool((original[changed] != 1).any()):
        raise ValueError("Only keeper class-1 predictions may change")
    originally_correct = original == truth
    finally_correct = final == truth
    corrections = int(np.count_nonzero(changed & ~originally_correct & finally_correct))
    harms = int(np.count_nonzero(changed & originally_correct & ~finally_correct))
    neutral = int(np.count_nonzero(changed & ~originally_correct & ~finally_correct))
    keeper_tp_mask = (truth == 1) & (original == 1)
    keeper_tp = int(np.count_nonzero(keeper_tp_mask))
    if keeper_tp <= 0:
        raise ValueError("Action evaluation lacks keeper class-1 true positives")
    retained_tp = int(np.count_nonzero(keeper_tp_mask & ~changed))
    restricted_fp_mask = np.isin(truth, RIVALS) & (original == 1)
    restricted_denominator = int(np.count_nonzero(restricted_fp_mask))
    rejected = int(np.count_nonzero(restricted_fp_mask & changed))
    rejection_rate = (
        float(rejected) / float(restricted_denominator)
        if restricted_denominator > 0
        else 0.0
    )
    changed_count = int(np.count_nonzero(changed))
    ratio = (
        float(corrections) / float(changed_count) if changed_count > 0 else None
    )
    return ActionMetrics(
        changed=changed_count,
        corrections=corrections,
        harms=harms,
        neutral_changes=neutral,
        keeper_class1_tp=keeper_tp,
        retained_class1_tp=retained_tp,
        tp_retention=float(retained_tp) / float(keeper_tp),
        restricted_fp_denominator=restricted_denominator,
        restricted_fp_rejected=rejected,
        restricted_fp_rejection_rate=rejection_rate,
        correction_ratio=ratio,
    )

def action_candidate_is_better(
    candidate_metrics: ActionMetrics,
    candidate_threshold: float,
    candidate_token: str,
    incumbent_metrics: ActionMetrics,
    incumbent_threshold: float,
    incumbent_token: str,
) -> bool:
    """Exact action tie order: rejection, retention, threshold, float-hex."""

    if candidate_metrics.restricted_fp_rejected != (
        incumbent_metrics.restricted_fp_rejected
    ):
        return (
            candidate_metrics.restricted_fp_rejected
            > incumbent_metrics.restricted_fp_rejected
        )
    if candidate_metrics.tp_retention != incumbent_metrics.tp_retention:
        return candidate_metrics.tp_retention > incumbent_metrics.tp_retention
    if candidate_threshold != incumbent_threshold:
        return candidate_threshold > incumbent_threshold
    return str(candidate_token) < str(incumbent_token)

def select_calibration_action_threshold(
    calibrated_probabilities: np.ndarray,
    keeper_probabilities: np.ndarray,
    calibration_targets: np.ndarray | Sequence[int],
) -> ActionSelection:
    candidate = _finite_float64(
        calibrated_probabilities,
        name="calibration action probabilities",
        ndim=1,
    )
    if bool((candidate < 0.0).any() or (candidate > 1.0).any()):
        raise ValueError("Calibration action probabilities must lie in [0,1]")
    keeper = _validate_keeper_probabilities(keeper_probabilities)
    truth = np.asarray(calibration_targets, dtype=np.int64).reshape(-1)
    if candidate.shape != (keeper.shape[0],) or truth.shape != candidate.shape:
        raise ValueError("Calibration action inputs are not aligned")
    if not bool(np.isin(truth, np.arange(5, dtype=np.int64)).all()):
        raise ValueError("Calibration action targets must lie in classes 0..4")
    best: Optional[Tuple[ActionPolicy, ActionMetrics]] = None
    for threshold in np.unique(candidate):
        numeric = float(threshold)
        policy = ActionPolicy(numeric.hex(), numeric)
        metrics = evaluate_applied_actions(
            apply_action_policy(candidate, keeper, policy),
            truth,
        )
        feasible = (
            metrics.changed > 0
            and metrics.restricted_fp_rejected > 0
            and metrics.tp_retention >= 0.97
            and metrics.corrections
            >= 2 * (metrics.harms + metrics.neutral_changes)
        )
        if not feasible:
            continue
        if best is None or action_candidate_is_better(
            metrics,
            numeric,
            policy.threshold_token,
            best[1],
            float(best[0].numeric_threshold),
            best[0].threshold_token,
        ):
            best = (policy, metrics)
    if best is not None:
        return ActionSelection(policy=best[0], calibration_metrics=best[1])
    policy = ActionPolicy(NO_ACTION, None)
    metrics = evaluate_applied_actions(
        apply_action_policy(candidate, keeper, policy),
        truth,
    )
    if metrics.changed != 0 or metrics.correction_ratio is not None:
        raise RuntimeError("NO_ACTION metrics are not canonical")
    return ActionSelection(policy=policy, calibration_metrics=metrics)

@dataclass(frozen=True)
class BootstrapResult:
    auroc_delta: np.ndarray
    auprc_delta: np.ndarray
    auroc_interval: Tuple[float, float]
    auprc_interval: Tuple[float, float]

def component_bootstrap_row_positions(
    component_ids: Sequence[str],
    component_order: Sequence[str],
    draw: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Concatenate component rows left-to-right, retaining every repeat."""

    components = np.asarray([str(item) for item in component_ids], dtype=object)
    order = tuple(str(item) for item in component_order)
    if components.ndim != 1 or tuple(sorted(set(components.tolist()))) != order:
        raise ValueError("Component rows/order are not canonical")
    selected = np.asarray(draw, dtype=np.int64).reshape(-1)
    if selected.size == 0 or bool((selected < 0).any() or (selected >= len(order)).any()):
        raise ValueError("Component draw contains an out-of-range index")
    rows_by_component = {
        component: np.flatnonzero(components == component).astype(np.int64, copy=False)
        for component in order
    }
    return np.concatenate(
        [rows_by_component[order[int(component)]] for component in selected]
    ).astype(np.int64, copy=False)

def paired_component_bootstrap(
    *,
    sample_indices: np.ndarray,
    component_ids: Sequence[str],
    component_order: Sequence[str],
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    control_scores: np.ndarray,
    draws: np.ndarray,
    expected_component_order_sha256: str,
    expected_draws_sha256: str,
) -> BootstrapResult:
    indices = _strictly_increasing_int64(sample_indices, name="bootstrap sample indices")
    rows = indices.size
    components = np.asarray([str(item) for item in component_ids], dtype=object)
    order = tuple(str(item) for item in component_order)
    if (
        len(order) != LOCKED_COMPONENT_COUNT
        or components.shape != (rows,)
        or tuple(sorted(set(components.tolist()))) != order
        or any(re.fullmatch(r"[0-9a-f]{64}", item) is None for item in order)
    ):
        raise ValueError("Bootstrap component order is not canonical")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(expected_component_order_sha256)) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_draws_sha256)) is None
    ):
        raise ValueError("Bootstrap expected hashes are not canonical SHA-256")
    observed_order_sha = json_sha256(list(order))
    if observed_order_sha != str(expected_component_order_sha256):
        raise ValueError("Bootstrap component-order SHA-256 changed")
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    candidate = _finite_float64(candidate_scores, name="candidate scores", ndim=1)
    control = _finite_float64(control_scores, name="control scores", ndim=1)
    if truth.shape != (rows,) or candidate.shape != (rows,) or control.shape != (rows,):
        raise ValueError("Bootstrap row arrays are not aligned")
    if not bool(np.isin(truth, (0, 1)).all()):
        raise ValueError("Bootstrap labels must be binary")
    draw_matrix = np.asarray(draws, dtype=np.int64)
    if draw_matrix.shape != (
        COMPONENT_BOOTSTRAP_REPLICATES,
        LOCKED_COMPONENT_COUNT,
    ):
        raise ValueError("Bootstrap draw matrix has the wrong shape")
    if array_sha256(draw_matrix) != str(expected_draws_sha256):
        raise ValueError("Bootstrap draw matrix SHA-256 changed")
    if bool((draw_matrix < 0).any() or (draw_matrix >= len(order)).any()):
        raise ValueError("Bootstrap draw index is out of range")
    auroc_delta = np.empty(draw_matrix.shape[0], dtype=np.float64)
    auprc_delta = np.empty(draw_matrix.shape[0], dtype=np.float64)
    for replicate, draw in enumerate(draw_matrix):
        positions = component_bootstrap_row_positions(
            components.tolist(),
            order,
            draw,
        )
        replicate_truth = truth[positions]
        if np.unique(replicate_truth).size != 2:
            raise RuntimeError("A component-bootstrap replicate has one class")
        auroc_delta[replicate] = roc_auc_score(
            replicate_truth,
            candidate[positions],
        ) - roc_auc_score(replicate_truth, control[positions])
        auprc_delta[replicate] = average_precision_score(
            replicate_truth,
            candidate[positions],
        ) - average_precision_score(replicate_truth, control[positions])
    auroc_interval = tuple(
        float(item)
        for item in np.quantile(
            auroc_delta,
            (0.025, 0.975),
            method="linear",
        )
    )
    auprc_interval = tuple(
        float(item)
        for item in np.quantile(
            auprc_delta,
            (0.025, 0.975),
            method="linear",
        )
    )
    return BootstrapResult(
        auroc_delta=auroc_delta,
        auprc_delta=auprc_delta,
        auroc_interval=auroc_interval,
        auprc_interval=auprc_interval,
    )

@dataclass(frozen=True)
class MapStatistics:
    targets: np.ndarray
    active: np.ndarray
    bbox_usable: np.ndarray
    bbox_stat_valid: np.ndarray
    bbox_mass: np.ndarray
    geometry_lift: np.ndarray
    normalized_entropy: np.ndarray
    attention_cv: np.ndarray
    evidence_dispersion: np.ndarray
    invalid_attention_mass: np.ndarray
    focus_row_mask: np.ndarray
    pair_attention_cosine: Mapping[str, np.ndarray]
    pair_evidence_abs_pearson: Mapping[str, np.ndarray]

def compute_map_statistics(
    *,
    attention_maps: np.ndarray,
    evidence_maps: np.ndarray,
    valid16: np.ndarray,
    bbox_valid16: np.ndarray,
    targets: np.ndarray,
) -> MapStatistics:
    attention = _finite_float64(attention_maps, name="attention maps", ndim=4)
    evidence = _finite_float64(evidence_maps, name="evidence maps", ndim=4)
    if attention.shape != evidence.shape or attention.shape[1:] != (4, 16, 16):
        raise ValueError("Evidence and attention maps must be [N,4,16,16]")
    rows = attention.shape[0]
    valid = np.asarray(valid16, dtype=np.bool_)
    bbox = np.asarray(bbox_valid16, dtype=np.bool_)
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    if valid.shape == (rows, 1, 16, 16):
        valid = valid[:, 0]
    if valid.shape != (rows, 16, 16) or bbox.shape != (rows, 16, 16):
        raise ValueError("Map validity and bbox masks must be [N,16,16]")
    if truth.shape != (rows,) or not bool(
        np.isin(truth, np.arange(5, dtype=np.int64)).all()
    ):
        raise ValueError("Map targets must be aligned classes 0..4")
    active_mask = np.zeros((rows, 4), dtype=np.bool_)
    active_mask[:, 0] = True
    active_mask[truth == 1, 1:] = True
    for task, rival in enumerate(RIVALS, start=1):
        active_mask[truth == rival, task] = True
    if bool((bbox & ~valid).any()):
        raise ValueError("BBox map must already be intersected with valid16")
    if bool((attention < 0.0).any()):
        raise ValueError("Attention maps must be non-negative")
    valid_counts = valid.sum(axis=(1, 2)).astype(np.int64)
    if bool((valid_counts <= 1).any()):
        raise ValueError("Map statistics require more than one valid16 cell")
    invalid_mass = np.where(
        ~valid[:, None],
        attention,
        0.0,
    ).sum(axis=(2, 3), dtype=np.float64)
    valid_attention_sum = np.where(
        valid[:, None],
        attention,
        0.0,
    ).sum(axis=(2, 3), dtype=np.float64)
    if not np.allclose(valid_attention_sum, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Each attention map must have unit valid-cell mass")

    bbox_usable = bbox.any(axis=(1, 2))
    bbox_stat_valid = active_mask & bbox_usable[:, None]
    bbox_mass = np.zeros((rows, 4), dtype=np.float64)
    geometry_lift = np.zeros((rows, 4), dtype=np.float64)
    entropy = np.zeros((rows, 4), dtype=np.float64)
    coefficient_variation = np.zeros((rows, 4), dtype=np.float64)
    evidence_dispersion = np.zeros((rows, 4), dtype=np.float64)
    q = bbox.sum(axis=(1, 2), dtype=np.float64) / valid_counts
    for row in range(rows):
        selected = valid[row]
        for task in range(4):
            if not active_mask[row, task]:
                continue
            attention_vector = attention[row, task, selected]
            evidence_vector = evidence[row, task, selected]
            entropy[row, task] = -np.sum(
                attention_vector * np.log(np.maximum(attention_vector, 1e-12)),
                dtype=np.float64,
            ) / math.log(float(valid_counts[row]))
            coefficient_variation[row, task] = attention_vector.std(
                ddof=0,
                dtype=np.float64,
            ) / (attention_vector.mean(dtype=np.float64) + 1e-12)
            evidence_dispersion[row, task] = evidence_vector.std(
                ddof=0,
                dtype=np.float64,
            ) / math.sqrt(
                float(np.mean(np.square(evidence_vector), dtype=np.float64)) + 1e-12
            )
            if bbox_stat_valid[row, task]:
                mass = float(attention[row, task, bbox[row]].sum(dtype=np.float64))
                bbox_mass[row, task] = mass
                geometry_lift[row, task] = (mass - q[row]) / max(
                    1.0 - q[row],
                    1e-8,
                )

    focus = truth == 1
    pair_cosine: Dict[str, np.ndarray] = {}
    pair_pearson: Dict[str, np.ndarray] = {}
    pair_tasks = (1, 2, 3)
    for left_position, left_task in enumerate(pair_tasks):
        for right_task in pair_tasks[left_position + 1 :]:
            key = f"{SCORE_NAMES[left_task]}__{SCORE_NAMES[right_task]}"
            cosine_values: List[float] = []
            pearson_values: List[float] = []
            for row in np.flatnonzero(focus):
                selected = valid[row]
                left_attention = attention[row, left_task, selected]
                right_attention = attention[row, right_task, selected]
                denominator = np.linalg.norm(left_attention) * np.linalg.norm(
                    right_attention
                )
                if denominator <= 0.0 or not math.isfinite(float(denominator)):
                    raise FloatingPointError("Pair attention cosine has zero norm")
                cosine_values.append(
                    float(np.dot(left_attention, right_attention) / denominator)
                )
                left_evidence = evidence[row, left_task, selected]
                right_evidence = evidence[row, right_task, selected]
                left_centered = left_evidence - left_evidence.mean(dtype=np.float64)
                right_centered = right_evidence - right_evidence.mean(dtype=np.float64)
                evidence_denominator = np.linalg.norm(left_centered) * np.linalg.norm(
                    right_centered
                )
                if evidence_denominator <= 0.0 or not math.isfinite(
                    float(evidence_denominator)
                ):
                    raise FloatingPointError("Pair evidence Pearson has zero variance")
                pearson_values.append(
                    abs(
                        float(
                            np.dot(left_centered, right_centered)
                            / evidence_denominator
                        )
                    )
                )
            pair_cosine[key] = np.asarray(cosine_values, dtype=np.float64)
            pair_pearson[key] = np.asarray(pearson_values, dtype=np.float64)
    return MapStatistics(
        targets=truth.copy(),
        active=active_mask,
        bbox_usable=bbox_usable,
        bbox_stat_valid=bbox_stat_valid,
        bbox_mass=bbox_mass,
        geometry_lift=geometry_lift,
        normalized_entropy=entropy,
        attention_cv=coefficient_variation,
        evidence_dispersion=evidence_dispersion,
        invalid_attention_mass=invalid_mass,
        focus_row_mask=focus,
        pair_attention_cosine=pair_cosine,
        pair_evidence_abs_pearson=pair_pearson,
    )

def spatial_filter_dispersion(
    spatial_filters: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    filters = _finite_float64(spatial_filters, name="spatial filters", ndim=4)
    if filters.shape[1] != 9:
        raise ValueError("Spatial filters must have shape [N,9,H,W]")
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if valid.shape == (filters.shape[0], 1, filters.shape[2], filters.shape[3]):
        valid = valid[:, 0]
    if valid.shape != (filters.shape[0], filters.shape[2], filters.shape[3]):
        raise ValueError("Spatial-filter validity is not aligned")
    if bool((valid.sum(axis=(1, 2)) <= 0).any()):
        raise ValueError("Every spatial filter row needs valid support")
    per_sample = np.stack(
        [
            np.asarray(
                [filters[row, tap, valid[row]].mean(dtype=np.float64) for tap in range(9)]
            )
            for row in range(filters.shape[0])
        ]
    )
    return _sample_filter_dispersion(per_sample)

def channel_filter_dispersion(channel_filters: np.ndarray) -> float:
    filters = _finite_float64(channel_filters, name="channel filters", ndim=3)
    if filters.shape[2] != 9:
        raise ValueError("Channel filters must have shape [N,C,9]")
    return _sample_filter_dispersion(filters.reshape(filters.shape[0], -1))

def _sample_filter_dispersion(per_sample: np.ndarray) -> float:
    if per_sample.shape[0] <= 1:
        raise ValueError("Filter dispersion requires more than one sample")
    numerator = float(
        np.mean(per_sample.var(axis=0, ddof=0, dtype=np.float64), dtype=np.float64)
    )
    denominator = float(np.mean(np.square(per_sample), dtype=np.float64)) + 1e-12
    value = numerator / denominator
    if not math.isfinite(value):
        raise FloatingPointError("Filter dispersion is non-finite")
    return value

@dataclass(frozen=True)
class GateResult:
    passed: bool
    details: Mapping[str, object]

def evaluate_map_gate(statistics: MapStatistics) -> GateResult:
    targets = np.asarray(statistics.targets, dtype=np.int64).reshape(-1)
    rows = targets.size
    if rows == 0 or not bool(np.isin(targets, np.arange(5)).all()):
        raise ValueError("Map-gate targets must be non-empty classes 0..4")
    expected_active = np.zeros((rows, 4), dtype=np.bool_)
    expected_active[:, 0] = True
    expected_active[targets == 1, 1:] = True
    for task, rival in enumerate(RIVALS, start=1):
        expected_active[targets == rival, task] = True
    active = np.asarray(statistics.active, dtype=np.bool_)
    bbox_usable = np.asarray(statistics.bbox_usable, dtype=np.bool_).reshape(-1)
    bbox_valid = np.asarray(statistics.bbox_stat_valid, dtype=np.bool_)
    focus = np.asarray(statistics.focus_row_mask, dtype=np.bool_).reshape(-1)
    numeric_arrays = (
        statistics.bbox_mass,
        statistics.geometry_lift,
        statistics.normalized_entropy,
        statistics.attention_cv,
        statistics.evidence_dispersion,
        statistics.invalid_attention_mass,
    )
    if (
        active.shape != (rows, 4)
        or bbox_usable.shape != (rows,)
        or bbox_valid.shape != (rows, 4)
        or focus.shape != (rows,)
        or not np.array_equal(active, expected_active)
        or not np.array_equal(bbox_valid, active & bbox_usable[:, None])
        or not np.array_equal(focus, targets == 1)
        or bool((bbox_valid & ~active).any())
        or any(np.asarray(item).shape != (rows, 4) for item in numeric_arrays)
        or any(not np.isfinite(np.asarray(item, dtype=np.float64)).all() for item in numeric_arrays)
    ):
        raise ValueError("Map-gate statistics are structurally inconsistent")
    if bool(
        (np.asarray(statistics.bbox_mass) < 0.0).any()
        or (np.asarray(statistics.bbox_mass) > 1.0 + 1e-12).any()
        or (np.asarray(statistics.normalized_entropy) < 0.0).any()
        or (np.asarray(statistics.attention_cv) < 0.0).any()
        or (np.asarray(statistics.evidence_dispersion) < 0.0).any()
        or (np.asarray(statistics.invalid_attention_mass) < 0.0).any()
    ):
        raise ValueError("Map-gate statistic domains are invalid")
    if not (
        np.equal(np.asarray(statistics.bbox_mass)[~bbox_valid], 0.0).all()
        and np.equal(np.asarray(statistics.geometry_lift)[~bbox_valid], 0.0).all()
        and np.equal(np.asarray(statistics.normalized_entropy)[~active], 0.0).all()
        and np.equal(np.asarray(statistics.attention_cv)[~active], 0.0).all()
        and np.equal(np.asarray(statistics.evidence_dispersion)[~active], 0.0).all()
    ):
        raise ValueError("Inactive/null map statistics must be exact zero")
    expected_pair_keys = {
        f"{SCORE_NAMES[left]}__{SCORE_NAMES[right]}"
        for left, right in ((1, 2), (1, 3), (2, 3))
    }
    if (
        set(statistics.pair_attention_cosine) != expected_pair_keys
        or set(statistics.pair_evidence_abs_pearson) != expected_pair_keys
    ):
        raise ValueError("Map-gate pair-statistic keys are incomplete")
    for key in expected_pair_keys:
        cosine = _finite_float64(
            statistics.pair_attention_cosine[key],
            name=f"attention cosine {key}",
            ndim=1,
        )
        pearson = _finite_float64(
            statistics.pair_evidence_abs_pearson[key],
            name=f"evidence Pearson {key}",
            ndim=1,
        )
        if cosine.shape != (int(focus.sum()),) or pearson.shape != cosine.shape:
            raise ValueError("Map-gate pair-statistic support changed")
        if bool(
            (cosine < -1.0 - 1e-12).any()
            or (cosine > 1.0 + 1e-12).any()
            or (pearson < 0.0).any()
            or (pearson > 1.0 + 1e-12).any()
        ):
            raise ValueError("Map-gate pair-statistic domains are invalid")
    details: Dict[str, object] = {}
    passed = True
    for task, name in enumerate(SCORE_NAMES):
        selected = statistics.bbox_stat_valid[:, task]
        if not bool(selected.any()):
            task_pass = False
            summary = {"support": 0}
        else:
            mass = statistics.bbox_mass[selected, task]
            lift = statistics.geometry_lift[selected, task]
            summary = {
                "support": int(selected.sum()),
                "mean_bbox_mass": float(mass.mean(dtype=np.float64)),
                "bbox_mass_ge_0_70_fraction": float(np.mean(mass >= 0.70)),
                "mean_geometry_lift": float(lift.mean(dtype=np.float64)),
                "geometry_lift_ge_0_20_fraction": float(np.mean(lift >= 0.20)),
            }
            task_pass = (
                summary["mean_bbox_mass"] >= 0.85
                and summary["bbox_mass_ge_0_70_fraction"] >= 0.90
                and summary["mean_geometry_lift"] >= 0.50
                and summary["geometry_lift_ge_0_20_fraction"] >= 0.90
            )
        summary["passed"] = task_pass
        details[f"bbox_{name}"] = summary
        passed = passed and task_pass
    for task in (1, 2, 3):
        selected = statistics.active[:, task]
        entropy = statistics.normalized_entropy[selected, task]
        cv = statistics.attention_cv[selected, task]
        dispersion = statistics.evidence_dispersion[selected, task]
        head_pass = bool(selected.any()) and (
            float(np.median(entropy)) <= 0.995
            and float(np.mean(cv >= 0.05)) >= 0.90
            and float(np.mean(dispersion >= 0.01)) >= 0.90
        )
        details[f"shape_{SCORE_NAMES[task]}"] = {
            "support": int(selected.sum()),
            "median_entropy": float(np.median(entropy)) if entropy.size else None,
            "cv_ge_0_05_fraction": float(np.mean(cv >= 0.05)) if cv.size else None,
            "dispersion_ge_0_01_fraction": (
                float(np.mean(dispersion >= 0.01)) if dispersion.size else None
            ),
            "passed": head_pass,
        }
        passed = passed and head_pass
    for key in sorted(statistics.pair_attention_cosine):
        cosine = statistics.pair_attention_cosine[key]
        pearson = statistics.pair_evidence_abs_pearson[key]
        pair_pass = (
            cosine.size > 0
            and pearson.size > 0
            and float(np.median(cosine)) <= 0.995
            and float(np.median(pearson)) <= 0.995
        )
        details[f"pair_{key}"] = {
            "median_attention_cosine": (
                float(np.median(cosine)) if cosine.size else None
            ),
            "median_evidence_abs_pearson": (
                float(np.median(pearson)) if pearson.size else None
            ),
            "passed": pair_pass,
        }
        passed = passed and pair_pass
    invalid_exact = bool(np.equal(statistics.invalid_attention_mass, 0.0).all())
    details["invalid_attention_exact_zero"] = invalid_exact
    passed = passed and invalid_exact
    return GateResult(passed=bool(passed), details=details)

def evaluate_filter_gate(dispersions: np.ndarray) -> GateResult:
    values = _finite_float64(dispersions, name="filter dispersions", ndim=3)
    if values.shape != (5, 2, 2):
        raise ValueError("Filter dispersions must have shape [5 folds,2 blocks,2 factors]")
    passed = bool((values >= 1e-4).all())
    return GateResult(
        passed=passed,
        details={
            "minimum": float(values.min()),
            "all_20_ge_1e_4": passed,
        },
    )

# Raw scientific evidence; no caller-supplied metrics, calibrators or policies.
UNION_CALIBRATOR_NAMES = ("K", "KU", "KUH", "KUP")
PAIR_CALIBRATOR_NAMES = tuple(
    name for rival in RIVALS for name in (f"K_{rival}", f"KP_{rival}")
)
CAUSAL_SCORE_NAMES = (
    "clean",
    "support_matched_self",
    "cross_fold_substitution",
    "nonwrap_displacement",
    "spatial_neutral",
    "channel_neutral",
)
ARTIFACT_NAMES = (
    "sidecar_preimages",
    "causal_scores",
    "invalid_fill_scores",
    "invalid_fill_maps",
    "attention_maps",
    "evidence_maps",
    "spatial_filters32",
    "spatial_filters16",
    "channel_filters16",
    "channel_filters32",
    "donor_assignment",
    "nonwrap_offsets",
)
_SENTINEL_SHA256 = frozenset(
    {"0" * 64, "f" * 64, hashlib.sha256(b"").hexdigest()}
)

def _canonical_sha256(value: object, *, name: str) -> str:
    observed = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", observed) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256")
    if observed in _SENTINEL_SHA256:
        raise ValueError(f"{name} uses a forbidden sentinel SHA-256")
    return observed

def _exact_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype,
    shape: Tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} dtype changed: {value.dtype} != {np.dtype(dtype)}")
    if value.shape != shape:
        raise ValueError(f"{name} shape changed: {value.shape} != {shape}")
    if finite and not bool(np.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains a non-finite value")
    return value

def _digest_array(value: object, *, name: str, shape: Tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}")
    if value.dtype.kind != "U" or value.dtype.itemsize != 64 * 4:
        raise TypeError(f"{name} must use fixed-width <U64")
    for position, item in enumerate(value.reshape(-1)):
        _canonical_sha256(item, name=f"{name}[{position}]")
    return value

@dataclass(frozen=True)
class FrozenArrayIdentity:
    dtype: str
    shape: Tuple[int, ...]
    array_sha256: str

def frozen_array_identity(value: np.ndarray) -> FrozenArrayIdentity:
    if not isinstance(value, np.ndarray):
        raise TypeError("Frozen identity input must be a NumPy array")
    return FrozenArrayIdentity(
        dtype=str(value.dtype),
        shape=tuple(int(item) for item in value.shape),
        array_sha256=array_sha256(value),
    )

def _identity_json(value: FrozenArrayIdentity) -> Mapping[str, object]:
    return {
        "dtype": value.dtype,
        "shape": list(value.shape),
        "array_sha256": value.array_sha256,
    }

def _validate_identity(
    value: np.ndarray,
    identity: FrozenArrayIdentity,
    *,
    name: str,
) -> None:
    if not isinstance(identity, FrozenArrayIdentity):
        raise TypeError(f"{name} identity is not typed")
    _canonical_sha256(identity.array_sha256, name=f"{name} identity")
    if frozen_array_identity(value) != identity:
        raise ValueError(f"{name} differs from its frozen lineage identity")

@dataclass(frozen=True)
class FrozenTargetSlice:
    sample_indices: FrozenArrayIdentity
    targets: FrozenArrayIdentity

@dataclass(frozen=True)
class TargetShardLock:
    outer_fold: int
    fit: FrozenTargetSlice
    calibration: FrozenTargetSlice
    held: FrozenTargetSlice

@dataclass(frozen=True)
class NamedArtifactIdentity:
    name: str
    sha256: str

@dataclass(frozen=True)
class SidecarFoldPreimage:
    role: str
    outer_fold: int
    calibration_sample_indices: np.ndarray
    calibration_raw_scores: np.ndarray
    held_sample_indices: np.ndarray
    held_raw_scores: np.ndarray

@dataclass(frozen=True)
class TrainingTraceEvidence:
    losses: np.ndarray
    gradient_sha256: np.ndarray
    optimizer_state_sha256: np.ndarray
    initialized_model_state_sha256: np.ndarray
    final_model_state_sha256: np.ndarray
    prepared_input_sha256: np.ndarray

@dataclass(frozen=True)
class LineageLock:
    schema: str
    authorization_sha256: str
    machine_lock_sha256: str
    s1_payload_root_sha256: str
    s1_replay_payload_root_sha256: str
    s2_consumed_s1_root_sha256: str
    s2a_inner_root_sha256: str
    s2a_replay_inner_root_sha256: str
    s2b_consumed_s2a_root_sha256: str
    s2b_payload_root_sha256: str
    s2b_replay_payload_root_sha256: str
    s2_payload_root_sha256: str
    s2_replay_payload_root_sha256: str
    component_order_sha256: str
    sample_component_mapping_sha256: str
    bootstrap_draws: FrozenArrayIdentity
    keeper_probabilities: FrozenArrayIdentity
    valid64: FrozenArrayIdentity
    valid32: FrozenArrayIdentity
    valid16: FrozenArrayIdentity
    bbox_valid16: FrozenArrayIdentity
    nonwrap_offsets: FrozenArrayIdentity
    donor_artifact_sha256: str
    sidecar_preimage_registry_sha256: str
    artifact_registry_sha256: str
    bbox_unusable_sample_indices: FrozenArrayIdentity
    bbox_unusable_records_sha256: str
    target_shards: Tuple[TargetShardLock, ...]
    target_shard_registry_sha256: str
    cidt_sample_indices: FrozenArrayIdentity
    cidt_targets: FrozenArrayIdentity
    cidt_baseline_predictions: FrozenArrayIdentity
    cidt_registry_sha256: str
    frozen_record_sha256: str

@dataclass(frozen=True)
class RawScientificRun:
    sample_indices: np.ndarray
    targets: np.ndarray
    held_folds: np.ndarray
    component_ids: Tuple[str, ...]
    component_order: Tuple[str, ...]
    bootstrap_draws: np.ndarray
    keeper_probabilities: np.ndarray
    sidecar_preimages: Tuple[SidecarFoldPreimage, ...]
    causal_scores: np.ndarray
    invalid_fill_scores: np.ndarray
    invalid_fill_maps: np.ndarray
    attention_maps: np.ndarray
    evidence_maps: np.ndarray
    valid64: np.ndarray
    valid32: np.ndarray
    valid16: np.ndarray
    bbox_valid16: np.ndarray
    nonwrap_offsets: np.ndarray
    donor_artifact_sha256: str
    artifact_identities: Tuple[NamedArtifactIdentity, ...]
    spatial_filters32: np.ndarray
    spatial_filters16: np.ndarray
    channel_filters16: np.ndarray
    channel_filters32: np.ndarray
    cidt_sample_indices: np.ndarray
    cidt_targets: np.ndarray
    cidt_baseline_predictions: np.ndarray
    trace: TrainingTraceEvidence

@dataclass(frozen=True)
class RawScientificEvidence:
    lineage: LineageLock
    formal: RawScientificRun
    replay: RawScientificRun

class IncompleteScientificBoundaryError(RuntimeError):
    """Raised while the trusted formal runner boundary remains unfrozen."""

def _target_slice_json(value: FrozenTargetSlice) -> Mapping[str, object]:
    return {
        "sample_indices": _identity_json(value.sample_indices),
        "targets": _identity_json(value.targets),
    }

def target_shard_registry_sha256(shards: Sequence[TargetShardLock]) -> str:
    return json_sha256(
        [
            {
                "outer_fold": int(shard.outer_fold),
                "fit": _target_slice_json(shard.fit),
                "calibration": _target_slice_json(shard.calibration),
                "held": _target_slice_json(shard.held),
            }
            for shard in shards
        ]
    )

def cidt_registry_sha256(
    sample_indices: FrozenArrayIdentity,
    targets: FrozenArrayIdentity,
    baseline_predictions: FrozenArrayIdentity,
) -> str:
    return json_sha256(
        {
            "sample_indices": _identity_json(sample_indices),
            "targets": _identity_json(targets),
            "baseline_predictions": _identity_json(baseline_predictions),
        }
    )

def build_target_shard_locks(
    sample_indices: np.ndarray,
    targets: np.ndarray,
    held_folds: np.ndarray,
) -> Tuple[TargetShardLock, ...]:
    indices = _exact_array(
        sample_indices, name="shard IDs", dtype=np.dtype("<i8"), shape=(ROWS,)
    )
    truth = _exact_array(
        targets, name="shard targets", dtype=np.dtype("<i8"), shape=(ROWS,)
    )
    folds = _exact_array(
        held_folds, name="shard folds", dtype=np.dtype("<i8"), shape=(ROWS,)
    )
    output: List[TargetShardLock] = []
    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        masks = (
            (folds != outer_fold) & (folds != calibration_fold),
            folds == calibration_fold,
            folds == outer_fold,
        )
        slices: List[FrozenTargetSlice] = []
        for mask in masks:
            slices.append(
                FrozenTargetSlice(
                    frozen_array_identity(
                        np.ascontiguousarray(indices[mask], dtype=np.dtype("<i8"))
                    ),
                    frozen_array_identity(
                        np.ascontiguousarray(truth[mask], dtype=np.dtype("<i8"))
                    ),
                )
            )
        output.append(TargetShardLock(outer_fold, *slices))
    return tuple(output)

def sidecar_preimage_registry_sha256(
    preimages: Sequence[SidecarFoldPreimage],
) -> str:
    return json_sha256(
        [
            {
                "role": item.role,
                "outer_fold": int(item.outer_fold),
                "calibration_sample_indices": _identity_json(
                    frozen_array_identity(item.calibration_sample_indices)
                ),
                "calibration_raw_scores": _identity_json(
                    frozen_array_identity(item.calibration_raw_scores)
                ),
                "held_sample_indices": _identity_json(
                    frozen_array_identity(item.held_sample_indices)
                ),
                "held_raw_scores": _identity_json(
                    frozen_array_identity(item.held_raw_scores)
                ),
            }
            for item in preimages
        ]
    )

def _expected_artifact_identities(
    run: RawScientificRun,
) -> Tuple[NamedArtifactIdentity, ...]:
    arrays = {
        "causal_scores": run.causal_scores,
        "invalid_fill_scores": run.invalid_fill_scores,
        "invalid_fill_maps": run.invalid_fill_maps,
        "attention_maps": run.attention_maps,
        "evidence_maps": run.evidence_maps,
        "spatial_filters32": run.spatial_filters32,
        "spatial_filters16": run.spatial_filters16,
        "channel_filters16": run.channel_filters16,
        "channel_filters32": run.channel_filters32,
        "nonwrap_offsets": run.nonwrap_offsets,
    }
    digests = {
        name: array_sha256(value) for name, value in arrays.items()
    }
    digests["sidecar_preimages"] = sidecar_preimage_registry_sha256(
        run.sidecar_preimages
    )
    digests["donor_assignment"] = run.donor_artifact_sha256
    return tuple(
        NamedArtifactIdentity(name, digests[name]) for name in ARTIFACT_NAMES
    )

def artifact_registry_sha256(
    identities: Sequence[NamedArtifactIdentity],
) -> str:
    return json_sha256(
        [{"name": item.name, "sha256": item.sha256} for item in identities]
    )

def _component_mapping_sha256(run: RawScientificRun) -> str:
    return json_sha256(
        [
            [int(sample_index), component]
            for sample_index, component in zip(run.sample_indices, run.component_ids)
        ]
    )

def lineage_lock_sha256(lock: LineageLock) -> str:
    payload = {
        key: getattr(lock, key)
        for key in (
            "schema",
            "authorization_sha256",
            "machine_lock_sha256",
            "s1_payload_root_sha256",
            "s1_replay_payload_root_sha256",
            "s2_consumed_s1_root_sha256",
            "s2a_inner_root_sha256",
            "s2a_replay_inner_root_sha256",
            "s2b_consumed_s2a_root_sha256",
            "s2b_payload_root_sha256",
            "s2b_replay_payload_root_sha256",
            "s2_payload_root_sha256",
            "s2_replay_payload_root_sha256",
            "component_order_sha256",
            "sample_component_mapping_sha256",
            "donor_artifact_sha256",
            "sidecar_preimage_registry_sha256",
            "artifact_registry_sha256",
            "bbox_unusable_records_sha256",
            "target_shard_registry_sha256",
            "cidt_registry_sha256",
        )
    }
    for name in (
        "bootstrap_draws",
        "keeper_probabilities",
        "valid64",
        "valid32",
        "valid16",
        "bbox_valid16",
        "nonwrap_offsets",
        "bbox_unusable_sample_indices",
    ):
        payload[name] = _identity_json(getattr(lock, name))
    return json_sha256(payload)

def _run_identity_payload(run: RawScientificRun) -> Mapping[str, object]:
    arrays = {
        name: _identity_json(frozen_array_identity(getattr(run, name)))
        for name in (
            "sample_indices",
            "targets",
            "held_folds",
            "bootstrap_draws",
            "keeper_probabilities",
            "causal_scores",
            "invalid_fill_scores",
            "invalid_fill_maps",
            "attention_maps",
            "evidence_maps",
            "valid64",
            "valid32",
            "valid16",
            "bbox_valid16",
            "nonwrap_offsets",
            "spatial_filters32",
            "spatial_filters16",
            "channel_filters16",
            "channel_filters32",
            "cidt_sample_indices",
            "cidt_targets",
            "cidt_baseline_predictions",
        )
    }
    trace = {
        name: _identity_json(frozen_array_identity(getattr(run.trace, name)))
        for name in (
            "losses",
            "gradient_sha256",
            "optimizer_state_sha256",
            "initialized_model_state_sha256",
            "final_model_state_sha256",
            "prepared_input_sha256",
        )
    }
    return {
        "arrays": arrays,
        "component_mapping_sha256": _component_mapping_sha256(run),
        "component_order": list(run.component_order),
        "sidecar_registry_sha256": sidecar_preimage_registry_sha256(
            run.sidecar_preimages
        ),
        "donor_artifact_sha256": run.donor_artifact_sha256,
        "artifact_registry_sha256": artifact_registry_sha256(
            run.artifact_identities
        ),
        "trace": trace,
    }

def raw_scientific_evidence_sha256(evidence: RawScientificEvidence) -> str:
    return json_sha256(
        {
            "lineage_sha256": lineage_lock_sha256(evidence.lineage),
            "formal": _run_identity_payload(evidence.formal),
            "replay": _run_identity_payload(evidence.replay),
        }
    )

def _validate_lineage_structure(lock: LineageLock) -> None:
    if not isinstance(lock, LineageLock):
        raise TypeError("Scientific evidence lacks a typed LineageLock")
    if lock.schema != "trkh_pair_surface_ddf_v2_lineage_lock/v2":
        raise ValueError("Lineage lock schema is not v2")
    digest_names = (
        "authorization_sha256",
        "machine_lock_sha256",
        "s1_payload_root_sha256",
        "s1_replay_payload_root_sha256",
        "s2_consumed_s1_root_sha256",
        "s2a_inner_root_sha256",
        "s2a_replay_inner_root_sha256",
        "s2b_consumed_s2a_root_sha256",
        "s2b_payload_root_sha256",
        "s2b_replay_payload_root_sha256",
        "s2_payload_root_sha256",
        "s2_replay_payload_root_sha256",
        "component_order_sha256",
        "sample_component_mapping_sha256",
        "donor_artifact_sha256",
        "sidecar_preimage_registry_sha256",
        "artifact_registry_sha256",
        "bbox_unusable_records_sha256",
        "target_shard_registry_sha256",
        "cidt_registry_sha256",
        "frozen_record_sha256",
    )
    for name in digest_names:
        _canonical_sha256(getattr(lock, name), name=name)
    if not (
        lock.s1_payload_root_sha256
        == lock.s1_replay_payload_root_sha256
        == lock.s2_consumed_s1_root_sha256
    ):
        raise ValueError("S1 formal/replay/consumed roots differ")
    if not (
        lock.s2a_inner_root_sha256
        == lock.s2a_replay_inner_root_sha256
        == lock.s2b_consumed_s2a_root_sha256
    ):
        raise ValueError("S2A formal/replay/consumed roots differ")
    if lock.s2b_payload_root_sha256 != lock.s2b_replay_payload_root_sha256:
        raise ValueError("S2B formal/replay roots differ")
    if lock.s2_payload_root_sha256 != lock.s2_replay_payload_root_sha256:
        raise ValueError("S2 formal/replay roots differ")
    distinct = {
        lock.authorization_sha256,
        lock.machine_lock_sha256,
        lock.s1_payload_root_sha256,
        lock.s2a_inner_root_sha256,
        lock.s2b_payload_root_sha256,
        lock.s2_payload_root_sha256,
    }
    if len(distinct) != 6:
        raise ValueError("Distinct lineage stages reuse a root")
    if tuple(shard.outer_fold for shard in lock.target_shards) != (0, 1, 2, 3, 4):
        raise ValueError("Target shards do not cover outer folds 0..4")
    if target_shard_registry_sha256(lock.target_shards) != lock.target_shard_registry_sha256:
        raise ValueError("Target-shard registry root changed")
    if cidt_registry_sha256(
        lock.cidt_sample_indices,
        lock.cidt_targets,
        lock.cidt_baseline_predictions,
    ) != lock.cidt_registry_sha256:
        raise ValueError("CIDT registry root changed")
    if lineage_lock_sha256(lock) != lock.frozen_record_sha256:
        raise ValueError("Lineage frozen-record root changed")

def _maxpool3_stride2(mask: np.ndarray) -> np.ndarray:
    rows, height, width = mask.shape
    padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    output = np.zeros((rows, (height + 1) // 2, (width + 1) // 2), dtype=np.bool_)
    for dy in range(3):
        for dx in range(3):
            output |= padded[:, dy : dy + height : 2, dx : dx + width : 2]
    return output

def _validate_trace(trace: TrainingTraceEvidence, *, name: str) -> None:
    if not isinstance(trace, TrainingTraceEvidence):
        raise TypeError(f"{name} trace is not typed")
    _exact_array(
        trace.losses,
        name=f"{name} losses",
        dtype=np.dtype("<f8"),
        shape=(5, 5, 160),
        finite=True,
    )
    for field, shape in (
        ("gradient_sha256", (5, 5, 160)),
        ("optimizer_state_sha256", (5, 5, 161)),
        ("initialized_model_state_sha256", (5, 5)),
        ("final_model_state_sha256", (5, 5)),
        ("prepared_input_sha256", (2, ROWS)),
    ):
        _digest_array(getattr(trace, field), name=f"{name} {field}", shape=shape)
    if not np.array_equal(trace.prepared_input_sha256[0], trace.prepared_input_sha256[1]):
        raise ValueError("Invalid-fill prepared inputs differ")

def _validate_sidecar_preimages(run: RawScientificRun, lock: LineageLock) -> None:
    expected_axis = tuple((role, fold) for role in ROLE_NAMES for fold in range(5))
    observed_axis = tuple((item.role, item.outer_fold) for item in run.sidecar_preimages)
    if observed_axis != expected_axis:
        raise ValueError("Sidecar preimages do not have exact role/fold axes")
    for item in run.sidecar_preimages:
        held_mask = run.held_folds == item.outer_fold
        calibration_mask = run.held_folds == ((item.outer_fold + 1) % 5)
        expected_calibration = np.ascontiguousarray(
            run.sample_indices[calibration_mask], dtype=np.dtype("<i8")
        )
        expected_held = np.ascontiguousarray(
            run.sample_indices[held_mask], dtype=np.dtype("<i8")
        )
        _exact_array(
            item.calibration_sample_indices,
            name=f"{item.role}/{item.outer_fold} calibration IDs",
            dtype=np.dtype("<i8"),
            shape=expected_calibration.shape,
        )
        _exact_array(
            item.held_sample_indices,
            name=f"{item.role}/{item.outer_fold} held IDs",
            dtype=np.dtype("<i8"),
            shape=expected_held.shape,
        )
        if not np.array_equal(item.calibration_sample_indices, expected_calibration):
            raise ValueError("Sidecar calibration row identity changed")
        if not np.array_equal(item.held_sample_indices, expected_held):
            raise ValueError("Sidecar held row identity changed")
        _exact_array(
            item.calibration_raw_scores,
            name=f"{item.role}/{item.outer_fold} calibration raw scores",
            dtype=np.dtype("<f8"),
            shape=(expected_calibration.size, 4),
            finite=True,
        )
        _exact_array(
            item.held_raw_scores,
            name=f"{item.role}/{item.outer_fold} held raw scores",
            dtype=np.dtype("<f8"),
            shape=(expected_held.size, 4),
            finite=True,
        )
    observed_root = sidecar_preimage_registry_sha256(run.sidecar_preimages)
    if observed_root != lock.sidecar_preimage_registry_sha256:
        raise ValueError("Sidecar preimages differ from frozen S2/formal lineage")

def _validate_components_and_offsets(run: RawScientificRun, lock: LineageLock) -> None:
    if len(run.component_ids) != ROWS or len(run.component_order) != 158:
        raise ValueError("Component axes must contain 763 rows and 158 components")
    for item in (*run.component_ids, *run.component_order):
        _canonical_sha256(item, name="component identity")
    if tuple(sorted(set(run.component_ids))) != run.component_order:
        raise ValueError("Component order is not canonical")
    if json_sha256(list(run.component_order)) != lock.component_order_sha256:
        raise ValueError("Component order differs from lineage")
    if _component_mapping_sha256(run) != lock.sample_component_mapping_sha256:
        raise ValueError("Sample-to-component mapping differs from lineage")
    component_folds: Dict[str, int] = {}
    for component, fold in zip(run.component_ids, run.held_folds):
        if component_folds.setdefault(component, int(fold)) != int(fold):
            raise ValueError("A component crosses outer folds")
    draws = _exact_array(
        run.bootstrap_draws,
        name="bootstrap draws",
        dtype=np.dtype("<i8"),
        shape=(2000, 158),
    )
    if not np.array_equal(draws, generate_component_bootstrap_draws()):
        raise ValueError("Bootstrap draws differ from locked PCG64 output")
    _validate_identity(draws, lock.bootstrap_draws, name="bootstrap draws")
    offsets = _exact_array(
        run.nonwrap_offsets,
        name="nonwrap offsets",
        dtype=np.dtype("<i8"),
        shape=(ROWS, 2, 2),
    )
    for fold in range(5):
        selected = run.held_folds == fold
        expected = build_nonwrap_offset_mapping(run.sample_indices[selected], fold=fold)
        if not np.array_equal(offsets[selected], expected):
            raise ValueError("Nonwrap offset preimage differs from formula")
    _validate_identity(offsets, lock.nonwrap_offsets, name="nonwrap offsets")

def _bbox_records(run: RawScientificRun) -> Tuple[np.ndarray, str]:
    unusable = ~run.bbox_valid16.any(axis=(1, 2))
    indices = np.ascontiguousarray(run.sample_indices[unusable], dtype=np.dtype("<i8"))
    records = [
        {"sample_index": int(index), "target": int(target)}
        for index, target in zip(indices, run.targets[unusable])
    ]
    return indices, json_sha256(records)

def _held_raw_scores(run: RawScientificRun, role: str) -> np.ndarray:
    output = np.empty((ROWS, 4), dtype=np.dtype("<f8"))
    seen = np.zeros(ROWS, dtype=np.bool_)
    for item in run.sidecar_preimages:
        if item.role != role:
            continue
        positions = np.searchsorted(run.sample_indices, item.held_sample_indices)
        if bool(seen[positions].any()):
            raise ValueError(f"Held raw scores for {role} duplicate rows")
        output[positions] = item.held_raw_scores
        seen[positions] = True
    if not bool(seen.all()):
        raise ValueError(f"Held raw scores for {role} omit rows")
    return output

def _validate_run(
    run: RawScientificRun,
    lock: LineageLock,
    *,
    name: str,
    exact_a0: bool,
) -> None:
    if not isinstance(run, RawScientificRun):
        raise TypeError(f"{name} is not RawScientificRun")
    indices = _exact_array(
        run.sample_indices, name=f"{name} IDs", dtype=np.dtype("<i8"), shape=(ROWS,)
    )
    targets = _exact_array(
        run.targets, name=f"{name} targets", dtype=np.dtype("<i8"), shape=(ROWS,)
    )
    folds = _exact_array(
        run.held_folds, name=f"{name} folds", dtype=np.dtype("<i8"), shape=(ROWS,)
    )
    if not bool(np.all(indices[1:] > indices[:-1])):
        raise ValueError("A0 sample indices are not strictly increasing")
    if not bool(np.isin(targets, np.arange(5)).all() and np.isin(folds, np.arange(5)).all()):
        raise ValueError("A0 targets/folds are out of domain")
    _validate_components_and_offsets(run, lock)
    keeper = _exact_array(
        run.keeper_probabilities,
        name=f"{name} keeper probabilities",
        dtype=np.dtype("<f8"),
        shape=(ROWS, 5),
        finite=True,
    )
    _validate_keeper_probabilities(keeper)
    _validate_identity(keeper, lock.keeper_probabilities, name="keeper probabilities")
    _validate_sidecar_preimages(run, lock)
    for field, shape in (
        ("causal_scores", (6, ROWS)),
        ("invalid_fill_scores", (2, ROWS)),
        ("invalid_fill_maps", (2, ROWS, 4, 16, 16)),
        ("attention_maps", (ROWS, 4, 16, 16)),
        ("evidence_maps", (ROWS, 4, 16, 16)),
        ("spatial_filters32", (ROWS, 9, 32, 32)),
        ("spatial_filters16", (ROWS, 9, 16, 16)),
        ("channel_filters16", (ROWS, 16, 9)),
        ("channel_filters32", (ROWS, 32, 9)),
    ):
        _exact_array(
            getattr(run, field),
            name=f"{name} {field}",
            dtype=np.dtype("<f8"),
            shape=shape,
            finite=True,
        )
    if not np.array_equal(run.invalid_fill_scores[0], run.invalid_fill_scores[1]):
        raise ValueError("Invalid-fill scores differ")
    if not np.array_equal(run.invalid_fill_maps[0], run.invalid_fill_maps[1]):
        raise ValueError("Invalid-fill maps differ")
    for field, shape in (
        ("valid64", (ROWS, 64, 64)),
        ("valid32", (ROWS, 32, 32)),
        ("valid16", (ROWS, 16, 16)),
        ("bbox_valid16", (ROWS, 16, 16)),
    ):
        value = _exact_array(
            getattr(run, field),
            name=f"{name} {field}",
            dtype=np.dtype(np.bool_),
            shape=shape,
        )
        _validate_identity(value, getattr(lock, field), name=field)
    if not np.array_equal(_maxpool3_stride2(run.valid64), run.valid32):
        raise ValueError("valid32 is not the exact valid64 propagation")
    if not np.array_equal(_maxpool3_stride2(run.valid32), run.valid16):
        raise ValueError("valid16 is not the exact valid32 propagation")
    if bool((run.bbox_valid16 & ~run.valid16).any()):
        raise ValueError("bbox_valid16 extends outside valid16")
    unusable_indices, unusable_records = _bbox_records(run)
    if unusable_indices.size != 12:
        raise ValueError("BBox usability must be exactly 751/12")
    _validate_identity(
        unusable_indices,
        lock.bbox_unusable_sample_indices,
        name="bbox-unusable IDs",
    )
    if unusable_records != lock.bbox_unusable_records_sha256:
        raise ValueError("BBox-unusable records differ from lineage")
    primary_raw = _held_raw_scores(run, ROLE_NAMES[0])
    if not np.array_equal(run.causal_scores[0], primary_raw[:, 0]):
        raise ValueError("Causal clean scores differ from primary clean raw union")
    map_scores = np.sum(
        run.attention_maps * run.evidence_maps,
        axis=(2, 3),
        dtype=np.float64,
    )
    if not np.allclose(map_scores, primary_raw, atol=1e-10, rtol=0.0):
        raise ValueError("Map-derived DDF scores differ from primary raw scores")
    _canonical_sha256(run.donor_artifact_sha256, name="donor artifact")
    if run.donor_artifact_sha256 != lock.donor_artifact_sha256:
        raise ValueError("Donor artifact differs from frozen lineage")
    expected_artifacts = _expected_artifact_identities(run)
    if run.artifact_identities != expected_artifacts:
        raise ValueError("Named raw artifact identities do not match their preimages")
    if artifact_registry_sha256(run.artifact_identities) != lock.artifact_registry_sha256:
        raise ValueError("Named artifact registry differs from lineage")
    if sidecar_preimage_registry_sha256(run.sidecar_preimages) != lock.sidecar_preimage_registry_sha256:
        raise ValueError("Sidecar registry differs from lineage")
    cidt_indices = _exact_array(
        run.cidt_sample_indices,
        name=f"{name} CIDT IDs",
        dtype=np.dtype("<i8"),
        shape=(9215,),
    )
    cidt_targets = _exact_array(
        run.cidt_targets,
        name=f"{name} CIDT targets",
        dtype=np.dtype("<i8"),
        shape=(9215,),
    )
    cidt_baseline = _exact_array(
        run.cidt_baseline_predictions,
        name=f"{name} CIDT baseline",
        dtype=np.dtype("<i8"),
        shape=(9215,),
    )
    if not np.array_equal(cidt_indices, np.arange(9215, dtype=np.dtype("<i8"))):
        raise ValueError("CIDT IDs are not exactly 0..9214")
    if not bool(np.isin(cidt_targets, np.arange(5)).all() and np.isin(cidt_baseline, np.arange(5)).all()):
        raise ValueError("CIDT targets/baseline are out of domain")
    for value, identity, label in (
        (cidt_indices, lock.cidt_sample_indices, "CIDT IDs"),
        (cidt_targets, lock.cidt_targets, "CIDT targets"),
        (cidt_baseline, lock.cidt_baseline_predictions, "CIDT baseline"),
    ):
        _validate_identity(value, identity, name=label)
    if build_target_shard_locks(indices, targets, folds) != lock.target_shards:
        raise ValueError("A0 arrays differ from frozen target shards")
    _validate_trace(run.trace, name=name)
    if exact_a0:
        exact = (
            (indices, EXACT_A0_SAMPLE_INDICES_I64_SHA256),
            (targets, EXACT_A0_TARGETS_I64_SHA256),
            (folds, EXACT_A0_HELD_FOLDS_I64_SHA256),
            (cidt_indices, EXACT_CIDT_SAMPLE_INDICES_I64_SHA256),
            (cidt_targets, EXACT_CIDT_TARGETS_I64_SHA256),
            (cidt_baseline, EXACT_CIDT_BASELINE_PREDICTIONS_I64_SHA256),
        )
        if any(array_sha256(value) != expected for value, expected in exact):
            raise ValueError("Exact A0/CIDT identities changed")
        pairs = [[int(index), int(fold)] for index, fold in zip(indices, folds)]
        if json_sha256(pairs) != EXACT_A0_SAMPLE_FOLD_PAIRS_JSON_SHA256:
            raise ValueError("Exact A0 sample/fold mapping changed")
        if lock.component_order_sha256 != EXACT_COMPONENT_ORDER_JSON_SHA256:
            raise ValueError("Exact component order changed")
        if unusable_records != EXACT_BBOX_UNUSABLE_RECORDS_JSON_SHA256:
            raise ValueError("Exact bbox-unusable records changed")

@dataclass(frozen=True)
class NamedCalibratorState:
    role: str
    outer_fold: int
    name: str
    state: CalibratorState

@dataclass(frozen=True)
class RepeatActionComparisonV2:
    eligible_rows: int
    decision_agreement: float
    suppressed_jaccard: float

@dataclass(frozen=True)
class CIDTDiagnosticResult:
    class1_precision_delta: float
    class1_f1_delta: float
    macro_f1_delta: float
    class1_recall_delta: float
    outside_cohort_rows: int
    outside_cohort_changed: int

@dataclass(frozen=True)
class DerivedRunEvidence:
    held_raw_scores: np.ndarray
    union_probabilities: np.ndarray
    pair_probabilities: np.ndarray
    action_probabilities: np.ndarray
    action_selections: Tuple[Tuple[ActionSelection, ...], ...]
    calibrators: Tuple[NamedCalibratorState, ...]
    pooled_auroc: np.ndarray
    pooled_auprc: np.ndarray
    fold_auroc: np.ndarray
    fold_auprc: np.ndarray
    pair_auroc: np.ndarray
    pair_auprc: np.ndarray
    causal_auroc: np.ndarray
    causal_auprc: np.ndarray
    bootstrap: BootstrapResult
    primary_action: AppliedAction
    repeat_action: AppliedAction
    primary_action_metrics: ActionMetrics
    repeat_action_metrics: ActionMetrics
    repeat_comparison: RepeatActionComparisonV2
    cidt: CIDTDiagnosticResult
    map_statistics: MapStatistics
    filter_dispersions: np.ndarray

@dataclass(frozen=True)
class ScientificComparisonRecord:
    category: str
    name: str
    kind: str
    formal_value: object
    replay_value: object

@dataclass(frozen=True)
class ConjunctiveGateEvaluationV2:
    all_passed: bool
    gates: Mapping[int, GateResult]
    formal: DerivedRunEvidence
    replay: DerivedRunEvidence
    comparison_records: Tuple[ScientificComparisonRecord, ...]

_UNION_FEATURE_NAMES = {
    "K": ("keeper_margin",),
    "KU": ("keeper_margin", "raw_union"),
    "KUH": ("keeper_margin", "raw_union", "rival_is_2", "rival_is_4"),
    "KUP": (
        "keeper_margin",
        "raw_union",
        "rival_is_2",
        "rival_is_4",
        "selected_pair_0",
        "selected_pair_2",
        "selected_pair_4",
    ),
}

def _calibrator_identity(record: NamedCalibratorState) -> Tuple[object, ...]:
    state = record.state
    arrays = (
        state.scaler_mean,
        state.scaler_var,
        state.scaler_scale,
        state.logistic_coef,
        state.logistic_intercept,
        state.classes,
        state.n_iter,
    )
    return (
        record.role,
        record.outer_fold,
        record.name,
        state.feature_names,
        *(frozen_array_identity(value) for value in arrays),
    )

def named_calibrator_registry_sha256(
    calibrators: Sequence[NamedCalibratorState],
) -> str:
    payload = []
    for record in calibrators:
        identity = _calibrator_identity(record)
        payload.append(
            {
                "role": identity[0],
                "outer_fold": identity[1],
                "name": identity[2],
                "feature_names": list(identity[3]),
                "arrays": [_identity_json(item) for item in identity[4:]],
            }
        )
    return json_sha256(payload)

def _fit_all_calibrators(
    run: RawScientificRun,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[NamedCalibratorState, ...],
    Tuple[Tuple[ActionSelection, ...], ...],
]:
    union_probabilities = np.empty((5, 4, ROWS), dtype=np.dtype("<f8"))
    pair_probabilities = np.empty((5, 3, 2, ROWS), dtype=np.dtype("<f8"))
    calibrators: List[NamedCalibratorState] = []
    action_selections: List[List[ActionSelection]] = [[], []]
    action_role_positions = {ROLE_NAMES[0]: 0, ROLE_NAMES[4]: 1}
    for preimage in run.sidecar_preimages:
        role_position = ROLE_NAMES.index(preimage.role)
        calibration_positions = np.searchsorted(
            run.sample_indices, preimage.calibration_sample_indices
        )
        held_positions = np.searchsorted(run.sample_indices, preimage.held_sample_indices)
        calibration_keeper = run.keeper_probabilities[calibration_positions]
        held_keeper = run.keeper_probabilities[held_positions]
        calibration_targets = run.targets[calibration_positions]
        union_labels = (calibration_targets == 1).astype(np.int64)
        calibration_union = build_union_feature_sets(
            calibration_keeper, preimage.calibration_raw_scores
        )
        held_union = build_union_feature_sets(held_keeper, preimage.held_raw_scores)
        calibration_kup_probability: Optional[np.ndarray] = None
        for score_position, score_name in enumerate(UNION_CALIBRATOR_NAMES):
            state = fit_locked_calibrator(
                calibration_union[score_name],
                union_labels,
                feature_names=_UNION_FEATURE_NAMES[score_name],
            )
            calibrators.append(
                NamedCalibratorState(
                    preimage.role, preimage.outer_fold, score_name, state
                )
            )
            union_probabilities[role_position, score_position, held_positions] = (
                state.predict_probability(held_union[score_name])
            )
            if score_name == "KUP":
                calibration_kup_probability = state.predict_probability(
                    calibration_union[score_name]
                )
        assert calibration_kup_probability is not None
        if preimage.role in action_role_positions:
            selection = select_calibration_action_threshold(
                calibration_kup_probability,
                calibration_keeper,
                calibration_targets,
            )
            action_selections[action_role_positions[preimage.role]].append(selection)
        for rival_position, rival in enumerate(RIVALS):
            calibration_pair = build_pair_feature_sets(
                calibration_keeper,
                preimage.calibration_raw_scores,
                rival=rival,
            )
            held_pair = build_pair_feature_sets(
                held_keeper,
                preimage.held_raw_scores,
                rival=rival,
            )
            active = np.isin(calibration_targets, (1, rival))
            pair_labels = (calibration_targets[active] == 1).astype(np.int64)
            for pair_position, pair_name in enumerate((f"K_{rival}", f"KP_{rival}")):
                feature_names = (
                    (f"keeper_margin_1v{rival}",)
                    if pair_position == 0
                    else (f"keeper_margin_1v{rival}", f"raw_pair_1v{rival}")
                )
                state = fit_locked_calibrator(
                    calibration_pair[pair_name][active],
                    pair_labels,
                    feature_names=feature_names,
                )
                calibrators.append(
                    NamedCalibratorState(
                        preimage.role,
                        preimage.outer_fold,
                        pair_name,
                        state,
                    )
                )
                pair_probabilities[
                    role_position, rival_position, pair_position, held_positions
                ] = state.predict_probability(held_pair[pair_name])
    expected_names = tuple(
        (role, fold, name)
        for role in ROLE_NAMES
        for fold in range(5)
        for name in (*UNION_CALIBRATOR_NAMES, *PAIR_CALIBRATOR_NAMES)
    )
    observed_names = tuple(
        (item.role, item.outer_fold, item.name) for item in calibrators
    )
    if observed_names != expected_names:
        raise RuntimeError("Named calibrator registry axes changed")
    if any(len(items) != 5 for items in action_selections):
        raise RuntimeError("Action policy derivation did not cover five folds")
    return (
        union_probabilities,
        pair_probabilities,
        tuple(calibrators),
        tuple(tuple(items) for items in action_selections),
    )

EVALUATION_SCORE_NAMES = (
    "K",
    "KU",
    "KUH",
    "KUP",
    "static_KUP",
    "spatial_KUP",
    "channel_KUP",
    "repeat_KUP",
)

def _evaluation_scores(union_probabilities: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            union_probabilities[0, 0],
            union_probabilities[0, 1],
            union_probabilities[0, 2],
            union_probabilities[0, 3],
            union_probabilities[1, 3],
            union_probabilities[2, 3],
            union_probabilities[3, 3],
            union_probabilities[4, 3],
        ),
        axis=0,
    )

def _binary_metrics(labels: np.ndarray, scores: np.ndarray, *, name: str) -> Tuple[float, float]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != values.shape or truth.size == 0 or np.unique(truth).size != 2:
        raise ValueError(f"{name} lacks aligned binary support")
    if not bool(np.isfinite(values).all()):
        raise FloatingPointError(f"{name} scores are non-finite")
    return float(roc_auc_score(truth, values)), float(average_precision_score(truth, values))

def _derive_score_metrics(
    run: RawScientificRun,
    union_probabilities: np.ndarray,
    pair_probabilities: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    scores = _evaluation_scores(union_probabilities)
    union_target = (run.targets == 1).astype(np.int64)
    pooled_auroc = np.empty(8, dtype=np.dtype("<f8"))
    pooled_auprc = np.empty(8, dtype=np.dtype("<f8"))
    fold_auroc = np.empty((8, 5), dtype=np.dtype("<f8"))
    fold_auprc = np.empty((8, 5), dtype=np.dtype("<f8"))
    for score_position, score_name in enumerate(EVALUATION_SCORE_NAMES):
        pooled_auroc[score_position], pooled_auprc[score_position] = _binary_metrics(
            union_target, scores[score_position], name=f"pooled {score_name}"
        )
        for fold in range(5):
            selected = run.held_folds == fold
            fold_auroc[score_position, fold], fold_auprc[score_position, fold] = (
                _binary_metrics(
                    union_target[selected],
                    scores[score_position, selected],
                    name=f"fold-{fold} {score_name}",
                )
            )
    pair_auroc = np.empty((3, 3), dtype=np.dtype("<f8"))
    pair_auprc = np.empty((3, 3), dtype=np.dtype("<f8"))
    for rival_position, rival in enumerate(RIVALS):
        selected = np.isin(run.targets, (1, rival))
        labels = (run.targets[selected] == 1).astype(np.int64)
        pair_scores = (
            pair_probabilities[0, rival_position, 0],
            pair_probabilities[0, rival_position, 1],
            pair_probabilities[1, rival_position, 1],
        )
        for score_position, values in enumerate(pair_scores):
            pair_auroc[rival_position, score_position], pair_auprc[
                rival_position, score_position
            ] = _binary_metrics(labels, values[selected], name=f"pair-{rival}")
    causal_auroc = np.empty(6, dtype=np.dtype("<f8"))
    causal_auprc = np.empty(6, dtype=np.dtype("<f8"))
    for position, name in enumerate(CAUSAL_SCORE_NAMES):
        causal_auroc[position], causal_auprc[position] = _binary_metrics(
            union_target, run.causal_scores[position], name=f"causal {name}"
        )
    return (
        pooled_auroc,
        pooled_auprc,
        fold_auroc,
        fold_auprc,
        pair_auroc,
        pair_auprc,
        causal_auroc,
        causal_auprc,
    )

def _apply_derived_actions(
    run: RawScientificRun,
    union_probabilities: np.ndarray,
    selections: Tuple[Tuple[ActionSelection, ...], ...],
) -> Tuple[np.ndarray, AppliedAction, AppliedAction]:
    action_probabilities = np.stack(
        (union_probabilities[0, 3], union_probabilities[4, 3]), axis=0
    )
    actions: List[AppliedAction] = []
    for role_position in range(2):
        keeper = keeper_predictions(run.keeper_probabilities)
        replacement = keeper_replacement_rival(run.keeper_probabilities)
        final = keeper.copy()
        changed = np.zeros(ROWS, dtype=np.bool_)
        for fold in range(5):
            selected = run.held_folds == fold
            applied = apply_action_policy(
                action_probabilities[role_position, selected],
                run.keeper_probabilities[selected],
                selections[role_position][fold].policy,
            )
            final[selected] = applied.final_predictions
            changed[selected] = applied.changed
        actions.append(AppliedAction(keeper, replacement, final, changed))
    return action_probabilities, actions[0], actions[1]

def _action_gate(metrics: ActionMetrics) -> bool:
    return bool(
        metrics.keeper_class1_tp == 528
        and metrics.tp_retention >= 0.97
        and metrics.restricted_fp_denominator == 222
        and metrics.restricted_fp_rejected >= 45
        and metrics.restricted_fp_rejection_rate >= 45.0 / 222.0
        and metrics.restricted_fp_rejected
        == metrics.corrections + metrics.neutral_changes
        and metrics.corrections >= 2 * (metrics.harms + metrics.neutral_changes)
    )

def _repeat_comparison(primary: AppliedAction, repeat: AppliedAction) -> RepeatActionComparisonV2:
    eligible = primary.keeper_predictions == 1
    left, right = primary.changed[eligible], repeat.changed[eligible]
    union = int(np.count_nonzero(left | right))
    return RepeatActionComparisonV2(
        int(np.count_nonzero(eligible)),
        float(np.mean(left == right)),
        1.0 if union == 0 else float(np.count_nonzero(left & right)) / union,
    )

def _derive_cidt(run: RawScientificRun, action: AppliedAction) -> CIDTDiagnosticResult:
    positions = np.searchsorted(run.cidt_sample_indices, run.sample_indices)
    if not np.array_equal(run.cidt_sample_indices[positions], run.sample_indices):
        raise ValueError("CIDT projection omits A0 rows")
    if not np.array_equal(run.cidt_targets[positions], run.targets):
        raise ValueError("CIDT/A0 targets differ")
    if not np.array_equal(run.cidt_baseline_predictions[positions], action.keeper_predictions):
        raise ValueError("CIDT baseline differs from keeper probabilities")
    after = run.cidt_baseline_predictions.copy()
    after[positions] = action.final_predictions
    before_precision, before_recall, before_f1, _ = precision_recall_fscore_support(
        run.cidt_targets,
        run.cidt_baseline_predictions,
        labels=np.arange(5),
        zero_division=0,
    )
    after_precision, after_recall, after_f1, _ = precision_recall_fscore_support(
        run.cidt_targets, after, labels=np.arange(5), zero_division=0
    )
    outside = np.ones(9215, dtype=np.bool_)
    outside[positions] = False
    return CIDTDiagnosticResult(
        float(after_precision[1] - before_precision[1]),
        float(after_f1[1] - before_f1[1]),
        float(np.mean(after_f1) - np.mean(before_f1)),
        float(after_recall[1] - before_recall[1]),
        int(np.count_nonzero(outside)),
        int(np.count_nonzero(after[outside] != run.cidt_baseline_predictions[outside])),
    )

def _filter_dispersions(run: RawScientificRun) -> np.ndarray:
    output = np.empty((5, 2, 2), dtype=np.dtype("<f8"))
    for fold in range(5):
        selected = run.held_folds == fold
        output[fold, 0] = (
            spatial_filter_dispersion(run.spatial_filters32[selected], run.valid32[selected]),
            channel_filter_dispersion(run.channel_filters16[selected]),
        )
        output[fold, 1] = (
            spatial_filter_dispersion(run.spatial_filters16[selected], run.valid16[selected]),
            channel_filter_dispersion(run.channel_filters32[selected]),
        )
    return output

def _derive_run(run: RawScientificRun, lock: LineageLock) -> DerivedRunEvidence:
    union_probabilities, pair_probabilities, calibrators, selections = (
        _fit_all_calibrators(run)
    )
    metrics = _derive_score_metrics(run, union_probabilities, pair_probabilities)
    action_probabilities, primary_action, repeat_action = _apply_derived_actions(
        run, union_probabilities, selections
    )
    primary_metrics = evaluate_applied_actions(primary_action, run.targets)
    repeat_metrics = evaluate_applied_actions(repeat_action, run.targets)
    map_statistics = compute_map_statistics(
        attention_maps=run.attention_maps,
        evidence_maps=run.evidence_maps,
        valid16=run.valid16,
        bbox_valid16=run.bbox_valid16,
        targets=run.targets,
    )
    held_raw = np.stack(
        [_held_raw_scores(run, role) for role in ROLE_NAMES], axis=0
    )
    evaluation_scores = _evaluation_scores(union_probabilities)
    bootstrap = paired_component_bootstrap(
        sample_indices=run.sample_indices,
        component_ids=run.component_ids,
        component_order=run.component_order,
        labels=(run.targets == 1).astype(np.int64),
        candidate_scores=evaluation_scores[3],
        control_scores=evaluation_scores[0],
        draws=run.bootstrap_draws,
        expected_component_order_sha256=lock.component_order_sha256,
        expected_draws_sha256=lock.bootstrap_draws.array_sha256,
    )
    return DerivedRunEvidence(
        held_raw_scores=held_raw,
        union_probabilities=union_probabilities,
        pair_probabilities=pair_probabilities,
        action_probabilities=action_probabilities,
        action_selections=selections,
        calibrators=calibrators,
        pooled_auroc=metrics[0],
        pooled_auprc=metrics[1],
        fold_auroc=metrics[2],
        fold_auprc=metrics[3],
        pair_auroc=metrics[4],
        pair_auprc=metrics[5],
        causal_auroc=metrics[6],
        causal_auprc=metrics[7],
        bootstrap=bootstrap,
        primary_action=primary_action,
        repeat_action=repeat_action,
        primary_action_metrics=primary_metrics,
        repeat_action_metrics=repeat_metrics,
        repeat_comparison=_repeat_comparison(primary_action, repeat_action),
        cidt=_derive_cidt(run, primary_action),
        map_statistics=map_statistics,
        filter_dispersions=_filter_dispersions(run),
    )

def _gates_1_to_13(run: RawScientificRun, derived: DerivedRunEvidence) -> Dict[int, GateResult]:
    index = {name: position for position, name in enumerate(EVALUATION_SCORE_NAMES)}
    gates: Dict[int, GateResult] = {}
    auc_delta = derived.pooled_auroc[index["KUP"]] - derived.pooled_auroc[index["K"]]
    ap_delta = derived.pooled_auprc[index["KUP"]] - derived.pooled_auprc[index["K"]]
    gates[1] = GateResult(
        bool(auc_delta >= 0.015 and derived.bootstrap.auroc_interval[0] > 0.0),
        {"delta": float(auc_delta), "bootstrap_lower": derived.bootstrap.auroc_interval[0]},
    )
    gates[2] = GateResult(
        bool(ap_delta >= 0.005 and derived.bootstrap.auprc_interval[0] >= 0.0),
        {"delta": float(ap_delta), "bootstrap_lower": derived.bootstrap.auprc_interval[0]},
    )
    control_deltas = {
        name: float(derived.pooled_auroc[index["KUP"]] - derived.pooled_auroc[index[name]])
        for name in ("static_KUP", "spatial_KUP", "channel_KUP")
    }
    gates[3] = GateResult(
        bool(
            control_deltas["static_KUP"] >= 0.015
            and control_deltas["spatial_KUP"] >= 0.010
            and control_deltas["channel_KUP"] >= 0.010
        ),
        control_deltas,
    )
    keeper_delta = derived.fold_auroc[index["KUP"]] - derived.fold_auroc[index["K"]]
    static_delta = (
        derived.fold_auroc[index["KUP"]] - derived.fold_auroc[index["static_KUP"]]
    )
    joint = (keeper_delta > 0.0) & (static_delta > 0.0)
    gates[4] = GateResult(
        bool(
            np.count_nonzero(joint) >= 4
            and np.median(keeper_delta) > 0.0
            and np.median(static_delta) > 0.0
        ),
        {
            "joint_wins": int(np.count_nonzero(joint)),
            "keeper_median_delta": float(np.median(keeper_delta)),
            "static_median_delta": float(np.median(static_delta)),
        },
    )
    pair_checks: Dict[str, bool] = {}
    for position, rival in enumerate(RIVALS):
        pair_checks[f"pair_{rival}_noninferior"] = bool(
            derived.pair_auroc[position, 1] >= derived.pair_auroc[position, 0] - 0.005
        )
        if rival in (0, 2):
            pair_checks[f"pair_{rival}_beats_static"] = bool(
                derived.pair_auroc[position, 1] > derived.pair_auroc[position, 2]
            )
    gates[5] = GateResult(all(pair_checks.values()), pair_checks)
    attribution = derived.pooled_auroc[index["KUP"]] - derived.pooled_auroc[index["KUH"]]
    fold_attribution = derived.fold_auroc[index["KUP"]] - derived.fold_auroc[index["KUH"]]
    gates[6] = GateResult(
        bool(
            attribution >= 0.005
            and np.count_nonzero(fold_attribution >= 0.0) >= 4
            and np.median(fold_attribution) > 0.0
        ),
        {
            "pooled_delta": float(attribution),
            "nonnegative_folds": int(np.count_nonzero(fold_attribution >= 0.0)),
            "median_fold_delta": float(np.median(fold_attribution)),
        },
    )
    causal = dict(zip(CAUSAL_SCORE_NAMES, derived.causal_auroc.tolist()))
    gate7 = {
        "support_close": abs(causal["support_matched_self"] - causal["clean"]) <= 0.005,
        "substitution_drop": causal["support_matched_self"] - causal["cross_fold_substitution"] >= 0.020,
        "displacement_drop": causal["clean"] - causal["nonwrap_displacement"] >= 0.015,
        "invalid_fill_exact": bool(
            np.array_equal(run.invalid_fill_scores[0], run.invalid_fill_scores[1])
            and np.array_equal(run.invalid_fill_maps[0], run.invalid_fill_maps[1])
            and np.array_equal(run.trace.prepared_input_sha256[0], run.trace.prepared_input_sha256[1])
        ),
    }
    gates[7] = GateResult(all(gate7.values()), gate7)
    gate8 = {
        "spatial": causal["clean"] - causal["spatial_neutral"] >= 0.010,
        "channel": causal["clean"] - causal["channel_neutral"] >= 0.010,
    }
    gates[8] = GateResult(all(gate8.values()), gate8)
    gates[9] = GateResult(
        _action_gate(derived.primary_action_metrics),
        {"metrics": derived.primary_action_metrics},
    )
    cidt = derived.cidt
    gates[10] = GateResult(
        bool(
            cidt.class1_precision_delta >= 0.030
            and cidt.class1_f1_delta >= 0.015
            and cidt.macro_f1_delta >= 0.003
            and cidt.class1_recall_delta >= -0.030
            and cidt.outside_cohort_rows == 8452
            and cidt.outside_cohort_changed == 0
        ),
        {
            "class1_precision": cidt.class1_precision_delta,
            "class1_f1": cidt.class1_f1_delta,
            "macro_f1": cidt.macro_f1_delta,
            "class1_recall": cidt.class1_recall_delta,
        },
    )
    repeat = derived.repeat_comparison
    gates[11] = GateResult(
        bool(
            abs(derived.pooled_auroc[index["repeat_KUP"]] - derived.pooled_auroc[index["KUP"]]) <= 0.010
            and _action_gate(derived.repeat_action_metrics)
            and repeat.eligible_rows == 750
            and repeat.decision_agreement >= 0.97
            and repeat.suppressed_jaccard >= 0.80
        ),
        {
            "eligible_rows": repeat.eligible_rows,
            "agreement": repeat.decision_agreement,
            "jaccard": repeat.suppressed_jaccard,
        },
    )
    gates[12] = evaluate_map_gate(derived.map_statistics)
    gates[13] = evaluate_filter_gate(derived.filter_dispersions)
    return gates


# This ordered registry is a protocol surface, not a convenience list.  A
# production replay is accepted only when every record is generated internally
# and compares exactly in this order.
REQUIRED_COMPARISON_RECORDS = (
    ("lineage", "frozen_record_sha256", "identity"),
    ("input", "sample_indices_i64", "array_exact"),
    ("input", "targets_i64", "array_exact"),
    ("input", "held_folds_i64", "array_exact"),
    ("input", "sample_component_mapping", "identity"),
    ("input", "component_order", "identity"),
    ("random", "bootstrap_draws_i64", "array_exact"),
    ("input", "keeper_probabilities_f64", "array_exact"),
    ("raw", "sidecar_preimage_registry", "identity"),
    ("raw", "sidecar_preimages", "identity"),
    ("raw", "artifact_registry", "identity"),
    ("raw", "artifact_identities", "identity"),
    ("raw", "causal_scores_f64", "array_exact"),
    ("raw", "invalid_fill_scores_f64", "array_exact"),
    ("raw", "invalid_fill_maps_f64", "array_exact"),
    ("raw", "attention_maps_f64", "array_exact"),
    ("raw", "evidence_maps_f64", "array_exact"),
    ("geometry", "valid64_bool", "array_exact"),
    ("geometry", "valid32_bool", "array_exact"),
    ("geometry", "valid16_bool", "array_exact"),
    ("geometry", "bbox_valid16_bool", "array_exact"),
    ("causal", "nonwrap_offsets_i64", "array_exact"),
    ("causal", "donor_artifact_sha256", "identity"),
    ("filter", "spatial_filters32_f64", "array_exact"),
    ("filter", "spatial_filters16_f64", "array_exact"),
    ("filter", "channel_filters16_f64", "array_exact"),
    ("filter", "channel_filters32_f64", "array_exact"),
    ("cidt", "sample_indices_i64", "array_exact"),
    ("cidt", "targets_i64", "array_exact"),
    ("cidt", "baseline_predictions_i64", "array_exact"),
    ("trace", "losses_f64", "array_exact"),
    ("trace", "gradient_sha256", "array_exact"),
    ("trace", "optimizer_state_sha256", "array_exact"),
    ("trace", "initialized_model_state_sha256", "array_exact"),
    ("trace", "final_model_state_sha256", "array_exact"),
    ("trace", "prepared_input_sha256", "array_exact"),
    ("derived", "held_raw_scores_f64", "array_exact"),
    ("derived", "union_probabilities_f64", "array_exact"),
    ("derived", "pair_probabilities_f64", "array_exact"),
    ("derived", "action_kup_probabilities_f64", "array_exact"),
    ("derived", "named_calibrator_registry_sha256", "identity"),
    ("derived", "named_calibrator_identities", "identity"),
    ("derived", "action_selections", "identity"),
    ("metric", "pooled_auroc_f64", "array_exact"),
    ("metric", "pooled_auprc_f64", "array_exact"),
    ("metric", "fold_auroc_f64", "array_exact"),
    ("metric", "fold_auprc_f64", "array_exact"),
    ("metric", "pair_auroc_f64", "array_exact"),
    ("metric", "pair_auprc_f64", "array_exact"),
    ("metric", "causal_auroc_f64", "array_exact"),
    ("metric", "causal_auprc_f64", "array_exact"),
    ("bootstrap", "auroc_delta_f64", "array_exact"),
    ("bootstrap", "auprc_delta_f64", "array_exact"),
    ("bootstrap", "intervals", "identity"),
    ("action", "primary_keeper_i64", "array_exact"),
    ("action", "primary_replacement_i64", "array_exact"),
    ("action", "primary_final_i64", "array_exact"),
    ("action", "primary_changed_bool", "array_exact"),
    ("action", "repeat_keeper_i64", "array_exact"),
    ("action", "repeat_replacement_i64", "array_exact"),
    ("action", "repeat_final_i64", "array_exact"),
    ("action", "repeat_changed_bool", "array_exact"),
    ("action", "primary_metrics", "identity"),
    ("action", "repeat_metrics", "identity"),
    ("action", "repeat_comparison", "identity"),
    ("cidt", "derived_diagnostic", "identity"),
    ("map", "statistics", "identity"),
    ("filter", "dispersions_f64", "array_exact"),
    ("gate", "gates_1_to_13", "identity"),
)

# Deliberately hard-coded below; tests also recompute this value from the
# ordered triples so accidental registry edits cannot silently redefine Gate 14.
REQUIRED_COMPARISON_REGISTRY_SHA256 = (
    "689730f64f918e3b482c1f2d565cf71be4e03a89cafafc09626c08008cbe2dd6"
)


def _sidecar_identities(run: RawScientificRun) -> Tuple[object, ...]:
    return tuple(
        (
            item.role,
            item.outer_fold,
            frozen_array_identity(item.calibration_sample_indices),
            frozen_array_identity(item.calibration_raw_scores),
            frozen_array_identity(item.held_sample_indices),
            frozen_array_identity(item.held_raw_scores),
        )
        for item in run.sidecar_preimages
    )


def _map_statistics_identity(statistics: MapStatistics) -> Tuple[object, ...]:
    arrays = tuple(
        frozen_array_identity(getattr(statistics, name))
        for name in (
            "targets",
            "active",
            "bbox_usable",
            "bbox_stat_valid",
            "bbox_mass",
            "geometry_lift",
            "normalized_entropy",
            "attention_cv",
            "evidence_dispersion",
            "invalid_attention_mass",
            "focus_row_mask",
        )
    )
    pairs = tuple(
        (
            key,
            frozen_array_identity(statistics.pair_attention_cosine[key]),
            frozen_array_identity(statistics.pair_evidence_abs_pearson[key]),
        )
        for key in sorted(statistics.pair_attention_cosine)
    )
    return arrays + pairs


def _comparison_values(
    lock: LineageLock,
    run: RawScientificRun,
    derived: DerivedRunEvidence,
    gates: Mapping[int, GateResult],
) -> Tuple[object, ...]:
    primary = derived.primary_action
    repeat = derived.repeat_action
    return (
        lock.frozen_record_sha256,
        run.sample_indices,
        run.targets,
        run.held_folds,
        tuple(zip(run.sample_indices.tolist(), run.component_ids)),
        run.component_order,
        run.bootstrap_draws,
        run.keeper_probabilities,
        sidecar_preimage_registry_sha256(run.sidecar_preimages),
        _sidecar_identities(run),
        artifact_registry_sha256(run.artifact_identities),
        run.artifact_identities,
        run.causal_scores,
        run.invalid_fill_scores,
        run.invalid_fill_maps,
        run.attention_maps,
        run.evidence_maps,
        run.valid64,
        run.valid32,
        run.valid16,
        run.bbox_valid16,
        run.nonwrap_offsets,
        run.donor_artifact_sha256,
        run.spatial_filters32,
        run.spatial_filters16,
        run.channel_filters16,
        run.channel_filters32,
        run.cidt_sample_indices,
        run.cidt_targets,
        run.cidt_baseline_predictions,
        run.trace.losses,
        run.trace.gradient_sha256,
        run.trace.optimizer_state_sha256,
        run.trace.initialized_model_state_sha256,
        run.trace.final_model_state_sha256,
        run.trace.prepared_input_sha256,
        derived.held_raw_scores,
        derived.union_probabilities,
        derived.pair_probabilities,
        derived.action_probabilities,
        named_calibrator_registry_sha256(derived.calibrators),
        tuple(_calibrator_identity(item) for item in derived.calibrators),
        derived.action_selections,
        derived.pooled_auroc,
        derived.pooled_auprc,
        derived.fold_auroc,
        derived.fold_auprc,
        derived.pair_auroc,
        derived.pair_auprc,
        derived.causal_auroc,
        derived.causal_auprc,
        derived.bootstrap.auroc_delta,
        derived.bootstrap.auprc_delta,
        (derived.bootstrap.auroc_interval, derived.bootstrap.auprc_interval),
        primary.keeper_predictions,
        primary.replacement_predictions,
        primary.final_predictions,
        primary.changed,
        repeat.keeper_predictions,
        repeat.replacement_predictions,
        repeat.final_predictions,
        repeat.changed,
        derived.primary_action_metrics,
        derived.repeat_action_metrics,
        derived.repeat_comparison,
        derived.cidt,
        _map_statistics_identity(derived.map_statistics),
        derived.filter_dispersions,
        tuple((index, gates[index].passed, gates[index].details) for index in range(1, 14)),
    )


def _comparison_equal(kind: str, formal: object, replay: object) -> bool:
    if kind == "array_exact":
        return bool(
            isinstance(formal, np.ndarray)
            and isinstance(replay, np.ndarray)
            and formal.dtype == replay.dtype
            and formal.shape == replay.shape
            and np.array_equal(formal, replay, equal_nan=False)
        )
    if kind != "identity":
        raise RuntimeError(f"Unknown scientific comparison kind: {kind}")
    return bool(formal == replay)


def _build_comparison_records(
    lock: LineageLock,
    formal_run: RawScientificRun,
    replay_run: RawScientificRun,
    formal: DerivedRunEvidence,
    replay: DerivedRunEvidence,
    formal_gates: Mapping[int, GateResult],
    replay_gates: Mapping[int, GateResult],
) -> Tuple[Tuple[ScientificComparisonRecord, ...], GateResult]:
    if json_sha256([list(item) for item in REQUIRED_COMPARISON_RECORDS]) != (
        REQUIRED_COMPARISON_REGISTRY_SHA256
    ):
        raise RuntimeError("Gate-14 comparison registry hash changed")
    formal_values = _comparison_values(lock, formal_run, formal, formal_gates)
    replay_values = _comparison_values(lock, replay_run, replay, replay_gates)
    if not (
        len(REQUIRED_COMPARISON_RECORDS)
        == len(formal_values)
        == len(replay_values)
    ):
        raise RuntimeError("Gate-14 registry/value cardinality changed")
    records = tuple(
        ScientificComparisonRecord(category, name, kind, left, right)
        for (category, name, kind), left, right in zip(
            REQUIRED_COMPARISON_RECORDS, formal_values, replay_values
        )
    )
    failed = tuple(
        f"{record.category}/{record.name}/{record.kind}"
        for record in records
        if not _comparison_equal(
            record.kind, record.formal_value, record.replay_value
        )
    )
    return records, GateResult(
        passed=not failed,
        details={
            "record_count": len(records),
            "registry_sha256": REQUIRED_COMPARISON_REGISTRY_SHA256,
            "failed_records": failed,
        },
    )


def _evaluate_raw_scientific_evidence(
    evidence: RawScientificEvidence,
    *,
    exact_a0: bool,
) -> ConjunctiveGateEvaluationV2:
    if not isinstance(evidence, RawScientificEvidence):
        raise TypeError("Scientific input must be RawScientificEvidence")
    _validate_lineage_structure(evidence.lineage)
    _validate_run(evidence.formal, evidence.lineage, name="formal", exact_a0=exact_a0)
    _validate_run(evidence.replay, evidence.lineage, name="replay", exact_a0=exact_a0)
    formal = _derive_run(evidence.formal, evidence.lineage)
    replay = _derive_run(evidence.replay, evidence.lineage)
    gates = _gates_1_to_13(evidence.formal, formal)
    replay_gates = _gates_1_to_13(evidence.replay, replay)
    records, gate14 = _build_comparison_records(
        evidence.lineage,
        evidence.formal,
        evidence.replay,
        formal,
        replay,
        gates,
        replay_gates,
    )
    gates[14] = gate14
    return ConjunctiveGateEvaluationV2(
        all_passed=bool(all(gate.passed for gate in gates.values())),
        gates=gates,
        formal=formal,
        replay=replay,
        comparison_records=records,
    )


def evaluate_synthetic_conjunctive_gates(
    evidence: RawScientificEvidence,
) -> ConjunctiveGateEvaluationV2:
    """Development-only evaluator for synthetic, non-production evidence.

    Mutation rejection here proves consistency with a supplied frozen lineage;
    it does not establish artifact origin or enforce target-release order.  Those
    properties belong to the future trusted fresh-process runner.
    """

    return _evaluate_raw_scientific_evidence(evidence, exact_a0=False)


def evaluate_exact_a0_conjunctive_gates(
    evidence: RawScientificEvidence,
) -> ConjunctiveGateEvaluationV2:
    """Fail closed until the trusted fresh-process runner contract is frozen."""

    del evidence
    raise IncompleteScientificBoundaryError(
        "Exact A0 evaluation is unavailable: trusted fresh-process runner "
        "integration and release-order enforcement are not yet frozen"
    )

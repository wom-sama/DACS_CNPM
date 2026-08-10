from __future__ import annotations

import hashlib
import inspect
import itertools
from functools import lru_cache
from dataclasses import replace

import numpy as np
import pytest

import trkh.tools.pair_surface_ddf_v2_scientific as scientific
from trkh.tools.pair_surface_ddf_v2_engine import array_sha256, json_sha256
from trkh.tools.pair_surface_ddf_v2_scientific import (
    NO_ACTION,
    ActionMetrics,
    ActionPolicy,
    DonorCovariates,
    REQUIRED_COMPARISON_RECORDS,
    REQUIRED_COMPARISON_REGISTRY_SHA256,
    action_candidate_is_better,
    apply_action_policy,
    assign_capacity_balanced_donors,
    build_nonwrap_offset_mapping,
    build_pair_feature_sets,
    build_target_blind_donor_cost_vectors,
    build_union_feature_sets,
    channel_filter_dispersion,
    component_bootstrap_row_positions,
    compute_map_statistics,
    evaluate_applied_actions,
    evaluate_exact_a0_conjunctive_gates,
    evaluate_filter_gate,
    evaluate_map_gate,
    evaluate_synthetic_conjunctive_gates,
    fit_locked_calibrator,
    generate_component_bootstrap_draws,
    generate_invalid_fill_noise,
    keeper_replacement_rival,
    keeper_selected_pair_rival,
    lexicographic_cost_coefficients,
    paired_component_bootstrap,
    select_calibration_action_threshold,
    spatial_filter_dispersion,
)


def _component_id(prefix: str, index: int) -> str:
    return hashlib.sha256(f"{prefix}-{index}".encode("utf-8")).hexdigest()


def _donor_covariates(
    sample_indices: list[int],
    *,
    prefix: str,
) -> DonorCovariates:
    rows = len(sample_indices)
    valid32 = np.zeros((rows, 32, 32), dtype=np.bool_)
    valid16 = np.zeros((rows, 16, 16), dtype=np.bool_)
    bbox = np.zeros((rows, 16, 16), dtype=np.bool_)
    for row in range(rows):
        valid32[row, 2:30, 2:30] = True
        valid16[row, 1:15, 1:15] = True
        bbox[row, 4:12, 4:12] = True
    return DonorCovariates(
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        component_ids=tuple(_component_id(prefix, row) for row in range(rows)),
        keeper_probabilities=np.tile(
            np.asarray([0.4, 0.5, 0.03, 0.02, 0.05]),
            (rows, 1),
        ),
        valid32=valid32,
        valid16=valid16,
        bbox_valid16=bbox,
        strongest_rival=np.zeros(rows, dtype=np.int64),
        bbox_area_rank=np.arange(rows, dtype=np.int64),
        valid_fraction_rank=np.arange(rows, dtype=np.int64),
    )


def test_target_blind_donor_costs_and_capacity_lex_tie_match_bruteforce() -> None:
    victims = _donor_covariates([10, 20, 30, 40, 50], prefix="victim")
    donors = _donor_covariates([110, 120, 130], prefix="donor")
    costs = build_target_blind_donor_cost_vectors(victims, donors)
    assert costs.shape == (5, 3, 6)
    assert costs.dtype == np.int64
    assert np.equal(costs[:, :, :4], 0).all()
    assert np.array_equal(costs[:, :, 4], costs[:, :, 5])
    assert "target" not in inspect.signature(
        build_target_blind_donor_cost_vectors
    ).parameters

    assignment = assign_capacity_balanced_donors(victims, donors)
    brute: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for candidate in itertools.product(range(3), repeat=5):
        counts = np.bincount(candidate, minlength=3)
        if sorted(counts.tolist()) != [1, 2, 2]:
            continue
        totals = tuple(
            int(costs[np.arange(5), np.asarray(candidate), priority].sum())
            for priority in range(6)
        )
        brute.append((totals, candidate))
    expected_totals, expected_vector = min(brute)
    assert assignment.lexicographic_donor_vector == expected_vector
    assert tuple(assignment.per_priority_totals.tolist()) == expected_totals
    assert sorted(assignment.donor_use_counts.tolist()) == [1, 2, 2]
    assert assignment.donor_sample_indices.tolist() == [
        donors.sample_indices[position] for position in expected_vector
    ]


def test_donor_priority_coefficients_prove_aggregate_dominance() -> None:
    maxima = (1024, 256, 256, 1, 152, 152)
    coefficients = lexicographic_cost_coefficients(maxima, assignments=153)
    assert coefficients[-1] == 1
    for priority in range(len(maxima) - 1):
        lower_maximum = 153 * sum(
            maxima[lower] * coefficients[lower]
            for lower in range(priority + 1, len(maxima))
        )
        assert coefficients[priority] > lower_maximum

    victims = _donor_covariates([1, 2], prefix="shared")
    donors = _donor_covariates([3, 4], prefix="donor")
    donors = replace(
        donors,
        component_ids=(victims.component_ids[0], donors.component_ids[1]),
    )
    with pytest.raises(ValueError, match="components overlap"):
        build_target_blind_donor_cost_vectors(victims, donors)
    with pytest.raises(ValueError, match="canonical geometry ranks"):
        build_target_blind_donor_cost_vectors(
            replace(victims, bbox_area_rank=np.asarray([0, 0])),
            _donor_covariates([3, 4], prefix="clean-donor"),
        )
    with pytest.raises(ValueError, match="strongest rival"):
        build_target_blind_donor_cost_vectors(
            replace(victims, strongest_rival=np.asarray([2, 0])),
            _donor_covariates([3, 4], prefix="clean-donor-rival"),
        )
    with pytest.raises(ValueError, match="canonical geometry ranks"):
        build_target_blind_donor_cost_vectors(
            replace(victims, valid_fraction_rank=np.asarray([1, 0])),
            _donor_covariates([3, 4], prefix="clean-donor-valid-rank"),
        )


def test_locked_offsets_noise_and_bootstrap_draws_are_exact() -> None:
    indices = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
    offsets = build_nonwrap_offset_mapping(indices, fold=0)
    assert offsets.tolist() == [
        [[0, 1], [-1, 1]],
        [[0, -1], [0, 1]],
        [[-1, -1], [0, -1]],
        [[-1, 0], [1, -1]],
        [[-1, 1], [1, 0]],
    ]
    assert not np.equal(offsets, 0).all(axis=2).any()
    for block in range(2):
        _, counts = np.unique(offsets[:, block], axis=0, return_counts=True)
        assert int(counts.max() - counts.min()) <= 1

    small_noise = generate_invalid_fill_noise(rows=2)
    rng = np.random.Generator(np.random.PCG64(20260729))
    expected_noise = rng.integers(
        0,
        256,
        size=(2, 3, 256, 256),
        dtype=np.uint8,
        endpoint=False,
    )
    assert np.array_equal(small_noise, expected_noise)
    assert array_sha256(small_noise) == (
        "8f03670b1dbe3018d36ae53f79d93e6d8395c1fb56894b92c54515ab3b366a75"
    )
    full_noise = generate_invalid_fill_noise()
    assert array_sha256(full_noise) == (
        "9c18e8ab55f6bbf15d1372876bf2578cd094f2a3f1a755e89992c086bf365596"
    )
    del full_noise

    draws = generate_component_bootstrap_draws()
    assert draws.shape == (2000, 158)
    assert draws.dtype == np.int64
    assert array_sha256(draws) == (
        "d0b5041be27e27cd17fa55acfb44e1a072e6b6dde360c115fb21f022030b981c"
    )


def _feature_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026072911)
    probabilities = rng.dirichlet(np.asarray([2.0, 3.0, 2.0, 1.0, 2.0]), size=90)
    # Ensure all three selected heads occur without changing probability validity.
    for row, rival in enumerate((0, 2, 4) * 30):
        probabilities[row, [0, 2, 4]] = 0.04
        probabilities[row, rival] = 0.24 + 0.001 * (row % 7)
        probabilities[row, 1] = 0.50 - 0.001 * (row % 11)
        probabilities[row, 3] = 1.0 - probabilities[row, [0, 1, 2, 4]].sum()
    raw_scores = rng.normal(size=(90, 4))
    labels = (np.arange(90) % 2).astype(np.int64)
    return probabilities, raw_scores, labels


def test_k_ku_kuh_kup_pair_features_and_locked_calibrator_are_deterministic() -> None:
    probabilities, raw_scores, labels = _feature_fixture()
    features = build_union_feature_sets(probabilities, raw_scores)
    assert {name: value.shape[1] for name, value in features.items()} == {
        "K": 1,
        "KU": 2,
        "KUH": 4,
        "KUP": 7,
    }
    selected = keeper_selected_pair_rival(probabilities)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    expected_k = np.log(clipped[:, 1]) - np.log(
        np.max(clipped[:, [0, 2, 4]], axis=1)
    )
    assert np.array_equal(features["K"][:, 0], expected_k)
    for position, rival in enumerate((0, 2, 4)):
        expected_selected_score = raw_scores[:, position + 1] * (selected == rival)
        assert np.array_equal(features["KUP"][:, 4 + position], expected_selected_score)
        inactive = selected != rival
        assert not np.signbit(features["KUP"][inactive, 4 + position]).any()

    pair = build_pair_feature_sets(probabilities, raw_scores, rival=2)
    expected_pair_margin = np.log(clipped[:, 1]) - np.log(clipped[:, 2])
    assert np.array_equal(pair["K_2"][:, 0], expected_pair_margin)
    assert np.array_equal(pair["KP_2"][:, 1], raw_scores[:, 2])

    names = ("K", "U", "I2", "I4", "S0", "S2", "S4")
    first = fit_locked_calibrator(features["KUP"], labels, feature_names=names)
    second = fit_locked_calibrator(features["KUP"], labels, feature_names=names)
    assert np.array_equal(first.scaler_mean, second.scaler_mean)
    assert np.array_equal(first.logistic_coef, second.logistic_coef)
    assert np.array_equal(
        first.predict_probability(features["KUP"]),
        second.predict_probability(features["KUP"]),
    )
    assert np.isfinite(first.predict_probability(features["KUP"])).all()
    with pytest.raises(ValueError, match="zero variance"):
        fit_locked_calibrator(
            np.ones((10, 2), dtype=np.float64),
            np.asarray([0, 1] * 5),
            feature_names=("a", "b"),
        )
    with pytest.raises(ValueError, match="binary support"):
        fit_locked_calibrator(
            np.column_stack((np.arange(10), np.arange(10) ** 2)),
            np.zeros(10, dtype=np.int64),
            feature_names=("a", "b"),
        )


def _keeper_action_probabilities(rows: int) -> np.ndarray:
    return np.tile(
        np.asarray([0.35, 0.55, 0.03, 0.03, 0.04], dtype=np.float64),
        (rows, 1),
    )


def test_calibration_threshold_selection_and_label_blind_application() -> None:
    targets = np.concatenate(
        (np.ones(100, dtype=np.int64), np.zeros(10, dtype=np.int64))
    )
    probabilities = _keeper_action_probabilities(110)
    calibrated = np.concatenate(
        (
            np.full(97, 0.9),
            np.full(3, 0.3),
            np.full(10, 0.2),
        )
    )
    selection = select_calibration_action_threshold(
        calibrated,
        probabilities,
        targets,
    )
    assert selection.policy.threshold_token == float(0.2).hex()
    assert selection.policy.numeric_threshold == 0.2
    assert selection.calibration_metrics.changed == 10
    assert selection.calibration_metrics.corrections == 10
    assert selection.calibration_metrics.harms == 0
    assert selection.calibration_metrics.tp_retention == 1.0
    assert "target" not in inspect.signature(apply_action_policy).parameters

    applied = apply_action_policy(calibrated, probabilities, selection.policy)
    assert np.count_nonzero(applied.changed) == 10
    metrics = evaluate_applied_actions(applied, targets)
    assert metrics == selection.calibration_metrics
    assert np.equal(applied.final_predictions[applied.changed], 0).all()
    assert np.equal(keeper_replacement_rival(probabilities), 0).all()

    no_action_probabilities = np.concatenate(
        (np.full(100, 0.1), np.full(10, 0.9))
    )
    no_action = select_calibration_action_threshold(
        no_action_probabilities,
        probabilities,
        targets,
    )
    assert no_action.policy == ActionPolicy(NO_ACTION, None)
    assert no_action.calibration_metrics.changed == 0
    assert no_action.calibration_metrics.correction_ratio is None


def test_action_tie_order_and_fail_closed_domains() -> None:
    base = ActionMetrics(
        changed=10,
        corrections=10,
        harms=0,
        neutral_changes=0,
        keeper_class1_tp=100,
        retained_class1_tp=98,
        tp_retention=0.98,
        restricted_fp_denominator=20,
        restricted_fp_rejected=10,
        restricted_fp_rejection_rate=0.5,
        correction_ratio=1.0,
    )
    assert action_candidate_is_better(
        replace(base, restricted_fp_rejected=11),
        0.1,
        "z",
        base,
        0.9,
        "a",
    )
    assert action_candidate_is_better(
        replace(base, tp_retention=0.99),
        0.1,
        "z",
        base,
        0.9,
        "a",
    )
    assert action_candidate_is_better(base, 0.8, "z", base, 0.7, "a")
    assert action_candidate_is_better(base, 0.8, "a", base, 0.8, "b")

    keeper = _keeper_action_probabilities(2)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        apply_action_policy(np.asarray([0.2, 1.1]), keeper, ActionPolicy(NO_ACTION, None))
    applied = apply_action_policy(
        np.asarray([0.2, 0.3]),
        keeper,
        ActionPolicy(NO_ACTION, None),
    )
    with pytest.raises(ValueError, match="classes 0..4"):
        evaluate_applied_actions(applied, np.asarray([1, 5]))
    malformed = replace(
        applied,
        changed=np.asarray([True, False]),
    )
    with pytest.raises(ValueError, match="changed mask"):
        evaluate_applied_actions(malformed, np.asarray([1, 0]))


def test_exact_component_bootstrap_preserves_order_and_repeats() -> None:
    component_a = _component_id("unequal", 0)
    component_b = _component_id("unequal", 1)
    row_positions = component_bootstrap_row_positions(
        component_ids=(component_a, component_a, component_b, component_b, component_b),
        component_order=(component_a, component_b),
        draw=np.asarray([1, 0, 1], dtype=np.int64),
    )
    assert row_positions.tolist() == [2, 3, 4, 0, 1, 2, 3, 4]

    component_order = tuple(
        sorted(_component_id("bootstrap", index) for index in range(158))
    )
    sample_indices = np.arange(1, 317, dtype=np.int64)
    component_ids: list[str] = []
    labels: list[int] = []
    candidate: list[float] = []
    control: list[float] = []
    for component in component_order:
        component_ids.extend((component, component))
        labels.extend((0, 1))
        candidate.extend((0.1, 0.9))
        control.extend((0.5, 0.5))
    draws = generate_component_bootstrap_draws()
    result = paired_component_bootstrap(
        sample_indices=sample_indices,
        component_ids=component_ids,
        component_order=component_order,
        labels=np.asarray(labels),
        candidate_scores=np.asarray(candidate),
        control_scores=np.asarray(control),
        draws=draws,
        expected_component_order_sha256=json_sha256(list(component_order)),
        expected_draws_sha256=array_sha256(draws),
    )
    assert result.auroc_delta.shape == (2000,)
    assert np.equal(result.auroc_delta, 0.5).all()
    assert np.equal(result.auprc_delta, 0.5).all()
    assert result.auroc_interval == (0.5, 0.5)
    assert result.auprc_interval == (0.5, 0.5)
    with pytest.raises(ValueError, match="SHA-256"):
        paired_component_bootstrap(
            sample_indices=sample_indices,
            component_ids=component_ids,
            component_order=component_order,
            labels=np.asarray(labels),
            candidate_scores=np.asarray(candidate),
            control_scores=np.asarray(control),
            draws=draws,
            expected_component_order_sha256="0" * 64,
            expected_draws_sha256=array_sha256(draws),
        )
    with pytest.raises(ValueError, match="draw matrix SHA-256"):
        paired_component_bootstrap(
            sample_indices=sample_indices,
            component_ids=component_ids,
            component_order=component_order,
            labels=np.asarray(labels),
            candidate_scores=np.asarray(candidate),
            control_scores=np.asarray(control),
            draws=draws,
            expected_component_order_sha256=json_sha256(list(component_order)),
            expected_draws_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="wrong shape"):
        paired_component_bootstrap(
            sample_indices=sample_indices,
            component_ids=component_ids,
            component_order=component_order,
            labels=np.asarray(labels),
            candidate_scores=np.asarray(candidate),
            control_scores=np.asarray(control),
            draws=draws[:-1],
            expected_component_order_sha256=json_sha256(list(component_order)),
            expected_draws_sha256=array_sha256(draws[:-1]),
        )


def _passing_map_statistics():
    rng = np.random.default_rng(2026072912)
    targets = np.asarray(([1, 0, 2, 4, 3] * 8), dtype=np.int64)
    rows = targets.size
    valid = np.zeros((rows, 16, 16), dtype=np.bool_)
    valid[:, 1:15, 1:15] = True
    bbox = np.zeros_like(valid)
    bbox[:, 5:11, 5:11] = True
    attention = np.zeros((rows, 4, 16, 16), dtype=np.float64)
    evidence = np.zeros_like(attention)
    for row in range(rows):
        inside = bbox[row]
        outside = valid[row] & ~bbox[row]
        for task in range(4):
            inside_weights = rng.uniform(0.1, 1.0, size=int(inside.sum()))
            outside_weights = rng.uniform(0.1, 1.0, size=int(outside.sum()))
            attention[row, task, inside] = 0.9 * inside_weights / inside_weights.sum()
            attention[row, task, outside] = 0.1 * outside_weights / outside_weights.sum()
            evidence[row, task, valid[row]] = rng.normal(
                loc=0.1 * task,
                scale=1.0,
                size=int(valid[row].sum()),
            )
    return compute_map_statistics(
        attention_maps=attention,
        evidence_maps=evidence,
        valid16=valid,
        bbox_valid16=bbox,
        targets=targets,
    )


def test_map_and_filter_statistics_match_formulas_and_gates() -> None:
    statistics = _passing_map_statistics()
    assert statistics.active.shape == (40, 4)
    assert statistics.active[statistics.targets == 1].all()
    assert statistics.active[statistics.targets == 3, 0].all()
    assert not statistics.active[statistics.targets == 3, 1:].any()
    assert np.allclose(
        statistics.bbox_mass[statistics.bbox_stat_valid],
        0.9,
        atol=2e-15,
        rtol=0.0,
    )
    q = 36.0 / 196.0
    expected_lift = (0.9 - q) / (1.0 - q)
    assert np.allclose(
        statistics.geometry_lift[statistics.bbox_stat_valid],
        expected_lift,
        atol=3e-15,
        rtol=0.0,
    )
    map_gate = evaluate_map_gate(statistics)
    assert map_gate.passed

    rng = np.random.default_rng(2026072913)
    spatial = rng.normal(size=(12, 9, 5, 6))
    valid = np.ones((12, 5, 6), dtype=np.bool_)
    channel = rng.normal(size=(12, 7, 9))
    spatial_value = spatial_filter_dispersion(spatial, valid)
    channel_value = channel_filter_dispersion(channel)
    expected_spatial_vectors = spatial.mean(axis=(2, 3))
    expected_spatial = expected_spatial_vectors.var(axis=0).mean() / (
        np.square(expected_spatial_vectors).mean() + 1e-12
    )
    expected_channel = channel.reshape(12, -1).var(axis=0).mean() / (
        np.square(channel).mean() + 1e-12
    )
    assert spatial_value == pytest.approx(expected_spatial, abs=2e-15)
    assert channel_value == pytest.approx(expected_channel, abs=2e-15)
    dispersions = np.full((5, 2, 2), 0.001, dtype=np.float64)
    assert evaluate_filter_gate(dispersions).passed
    dispersions[4, 1, 1] = 0.00009
    assert not evaluate_filter_gate(dispersions).passed

    invalid = statistics.invalid_attention_mass.copy()
    invalid[0, 0] = 1e-12
    assert not evaluate_map_gate(
        replace(statistics, invalid_attention_mass=invalid)
    ).passed
    with pytest.raises(ValueError, match="targets"):
        compute_map_statistics(
            attention_maps=np.ones((1, 4, 16, 16)) / 256.0,
            evidence_maps=np.ones((1, 4, 16, 16)),
            valid16=np.ones((1, 16, 16), dtype=np.bool_),
            bbox_valid16=np.ones((1, 16, 16), dtype=np.bool_),
            targets=np.asarray([5]),
        )


def _raw_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_class_counts = (
        (32, 109, 10, 2),
        (32, 108, 11, 2),
        (32, 108, 11, 2),
        (31, 108, 11, 2),
        (31, 108, 11, 2),
    )
    targets: list[int] = []
    folds: list[int] = []
    for fold, counts in enumerate(fold_class_counts):
        fold_targets: list[int] = []
        for label, count in zip((0, 1, 2, 4), counts):
            fold_targets.extend([label] * count)
        targets.extend(fold_targets)
        folds.extend([fold] * len(fold_targets))
    return (
        np.arange(763, dtype=np.dtype("<i8")),
        np.asarray(targets, dtype=np.dtype("<i8")),
        np.asarray(folds, dtype=np.dtype("<i8")),
    )


def _raw_components(held_folds: np.ndarray) -> tuple[tuple[str, ...], tuple[str, ...]]:
    order = tuple(sorted(_component_id("raw-component", index) for index in range(158)))
    component_ids = [""] * held_folds.size
    cursor = 0
    for fold, component_count in enumerate((32, 32, 32, 31, 31)):
        pool = order[cursor : cursor + component_count]
        cursor += component_count
        positions = np.flatnonzero(held_folds == fold)
        for local_position, row in enumerate(positions):
            component_ids[int(row)] = pool[local_position % component_count]
    return tuple(component_ids), order


def _noise(sample_indices: np.ndarray, salt: int) -> np.ndarray:
    integer = (sample_indices * (83 + 2 * salt) + 31 * salt + 7) % 1009
    return np.asarray(2.0 * integer / 1008.0 - 1.0, dtype=np.dtype("<f8"))


def _binary_score(
    sample_indices: np.ndarray,
    targets: np.ndarray,
    *,
    signal: float,
    salt: int,
) -> np.ndarray:
    signed = np.where(targets == 1, 1.0, -1.0)
    return np.asarray(signal * signed + _noise(sample_indices, salt), dtype="<f8")


def _keeper_probabilities(
    sample_indices: np.ndarray,
    targets: np.ndarray,
    held_folds: np.ndarray,
) -> np.ndarray:
    keeper = np.zeros((763, 5), dtype=np.dtype("<f8"))
    missed: list[int] = []
    for fold, count in enumerate((3, 3, 3, 2, 2)):
        missed.extend(np.flatnonzero((held_folds == fold) & (targets == 1))[:count])
    missed_set = set(missed)
    for row, (sample_index, target) in enumerate(zip(sample_indices, targets)):
        if row in missed_set:
            keeper[row] = np.asarray([0.55, 0.25, 0.08, 0.05, 0.07])
            continue
        rival = int(target) if target in (0, 2, 4) else (0, 2, 4)[row % 3]
        margin = 0.06 + 0.18 * ((_noise(np.asarray([sample_index]), 19)[0] + 1.0) / 2.0)
        keeper[row, 1] = 0.50
        keeper[row, rival] = 0.50 - margin
        for label in ({0, 2, 3, 4} - {rival}):
            keeper[row, label] = margin / 3.0
    return keeper


def _role_raw_scores(
    sample_indices: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    output = np.empty((5, 763, 4), dtype=np.dtype("<f8"))
    role_signals = (
        (0.30, 0.75, 0.78, 0.81),
        (0.06, 0.08, 0.07, 0.09),
        (0.13, 0.18, 0.17, 0.19),
        (0.14, 0.20, 0.18, 0.21),
        (0.30, 0.75, 0.78, 0.81),
    )
    for role, signals in enumerate(role_signals):
        for task, signal in enumerate(signals):
            salt = 30 + 7 * task if role == 4 else 30 + 41 * role + 7 * task
            output[role, :, task] = _binary_score(
                sample_indices,
                targets,
                signal=signal,
                salt=salt,
            )
    return output


def _sidecar_preimages(
    sample_indices: np.ndarray,
    held_folds: np.ndarray,
    role_scores: np.ndarray,
) -> tuple[scientific.SidecarFoldPreimage, ...]:
    output: list[scientific.SidecarFoldPreimage] = []
    for role_position, role in enumerate(scientific.ROLE_NAMES):
        for outer_fold in range(5):
            calibration = held_folds == ((outer_fold + 1) % 5)
            held = held_folds == outer_fold
            output.append(
                scientific.SidecarFoldPreimage(
                    role=role,
                    outer_fold=outer_fold,
                    calibration_sample_indices=np.ascontiguousarray(
                        sample_indices[calibration], dtype="<i8"
                    ),
                    calibration_raw_scores=np.ascontiguousarray(
                        role_scores[role_position, calibration], dtype="<f8"
                    ),
                    held_sample_indices=np.ascontiguousarray(
                        sample_indices[held], dtype="<i8"
                    ),
                    held_raw_scores=np.ascontiguousarray(
                        role_scores[role_position, held], dtype="<f8"
                    ),
                )
            )
    return tuple(output)


def _raw_maps(
    primary_raw_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid64 = np.ones((763, 64, 64), dtype=np.bool_)
    valid32 = np.ones((763, 32, 32), dtype=np.bool_)
    valid16 = np.ones((763, 16, 16), dtype=np.bool_)
    bbox = np.zeros((763, 16, 16), dtype=np.bool_)
    bbox[:751].reshape(751, -1)[:, :128] = True
    attention = np.zeros((763, 4, 16, 16), dtype=np.dtype("<f8"))
    evidence = np.zeros_like(attention)
    flat_attention = attention.reshape(763, 4, 256)
    flat_evidence = evidence.reshape(763, 4, 256)
    coordinates = np.arange(256, dtype=np.float64)
    for task in range(4):
        selected_bbox = np.arange(task, 128, 4)
        flat_attention[:, task, selected_bbox] = 0.90 / selected_bbox.size
        flat_attention[:, task, 128:] = 0.10 / 128.0
        pattern = np.sin(2.0 * np.pi * (task + 1) * coordinates / 256.0)
        weighted_mean = float(np.dot(flat_attention[0, task], pattern))
        centered = pattern - weighted_mean
        flat_evidence[:, task] = primary_raw_scores[:, task, None] + centered
    return attention, evidence, valid64, valid32, valid16, bbox


def _raw_filters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = np.arange(763, dtype=np.float64)[:, None]
    per_tap = (1.0 + 0.25 * np.sin(0.31 * row)) * (
        1.0 + 0.01 * np.arange(9, dtype=np.float64)[None, :]
    )
    spatial32 = np.broadcast_to(per_tap[:, :, None, None], (763, 9, 32, 32)).copy()
    spatial16 = np.broadcast_to(per_tap[:, :, None, None], (763, 9, 16, 16)).copy()
    channel16 = np.broadcast_to(per_tap[:, None, :], (763, 16, 9)).copy()
    channel32 = np.broadcast_to(per_tap[:, None, :], (763, 32, 9)).copy()
    return spatial32, spatial16, channel16, channel32


def _raw_trace() -> scientific.TrainingTraceEvidence:
    digest = _component_id("trace", 0)

    def digests(shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, digest, dtype="<U64")

    return scientific.TrainingTraceEvidence(
        losses=np.full((5, 5, 160), 0.5, dtype=np.dtype("<f8")),
        gradient_sha256=digests((5, 5, 160)),
        optimizer_state_sha256=digests((5, 5, 161)),
        initialized_model_state_sha256=digests((5, 5)),
        final_model_state_sha256=digests((5, 5)),
        prepared_input_sha256=digests((2, 763)),
    )


def _raw_cidt(
    sample_indices: np.ndarray,
    targets: np.ndarray,
    keeper_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cidt_indices = np.arange(9215, dtype=np.dtype("<i8"))
    outside_targets = np.asarray((0, 2, 3, 4), dtype=np.dtype("<i8"))[
        np.arange(9215 - 763) % 4
    ]
    cidt_targets = np.concatenate((targets, outside_targets)).astype("<i8", copy=False)
    baseline = cidt_targets.copy()
    baseline[sample_indices] = np.argmax(keeper_probabilities, axis=1).astype("<i8")
    return cidt_indices, cidt_targets, baseline


def _nonwrap_offsets(sample_indices: np.ndarray, held_folds: np.ndarray) -> np.ndarray:
    offsets = np.empty((763, 2, 2), dtype=np.dtype("<i8"))
    for fold in range(5):
        selected = held_folds == fold
        offsets[selected] = build_nonwrap_offset_mapping(sample_indices[selected], fold=fold)
    return offsets


def _lineage_lock(run: scientific.RawScientificRun) -> scientific.LineageLock:
    target_shards = scientific.build_target_shard_locks(
        run.sample_indices, run.targets, run.held_folds
    )
    unusable_mask = ~run.bbox_valid16.any(axis=(1, 2))
    unusable = np.ascontiguousarray(run.sample_indices[unusable_mask], dtype="<i8")
    unusable_records = [
        {"sample_index": int(index), "target": int(target)}
        for index, target in zip(unusable, run.targets[unusable_mask])
    ]
    cidt_ids = scientific.frozen_array_identity(run.cidt_sample_indices)
    cidt_targets = scientific.frozen_array_identity(run.cidt_targets)
    cidt_baseline = scientific.frozen_array_identity(run.cidt_baseline_predictions)
    roots = [_component_id("lineage-root", index) for index in range(6)]
    lock = scientific.LineageLock(
        schema="trkh_pair_surface_ddf_v2_lineage_lock/v2",
        authorization_sha256=roots[0],
        machine_lock_sha256=roots[1],
        s1_payload_root_sha256=roots[2],
        s1_replay_payload_root_sha256=roots[2],
        s2_consumed_s1_root_sha256=roots[2],
        s2a_inner_root_sha256=roots[3],
        s2a_replay_inner_root_sha256=roots[3],
        s2b_consumed_s2a_root_sha256=roots[3],
        s2b_payload_root_sha256=roots[4],
        s2b_replay_payload_root_sha256=roots[4],
        s2_payload_root_sha256=roots[5],
        s2_replay_payload_root_sha256=roots[5],
        component_order_sha256=json_sha256(list(run.component_order)),
        sample_component_mapping_sha256=json_sha256(
            [[int(index), component] for index, component in zip(run.sample_indices, run.component_ids)]
        ),
        bootstrap_draws=scientific.frozen_array_identity(run.bootstrap_draws),
        keeper_probabilities=scientific.frozen_array_identity(run.keeper_probabilities),
        valid64=scientific.frozen_array_identity(run.valid64),
        valid32=scientific.frozen_array_identity(run.valid32),
        valid16=scientific.frozen_array_identity(run.valid16),
        bbox_valid16=scientific.frozen_array_identity(run.bbox_valid16),
        nonwrap_offsets=scientific.frozen_array_identity(run.nonwrap_offsets),
        donor_artifact_sha256=run.donor_artifact_sha256,
        sidecar_preimage_registry_sha256=scientific.sidecar_preimage_registry_sha256(
            run.sidecar_preimages
        ),
        artifact_registry_sha256=scientific.artifact_registry_sha256(
            run.artifact_identities
        ),
        bbox_unusable_sample_indices=scientific.frozen_array_identity(unusable),
        bbox_unusable_records_sha256=json_sha256(unusable_records),
        target_shards=target_shards,
        target_shard_registry_sha256=scientific.target_shard_registry_sha256(
            target_shards
        ),
        cidt_sample_indices=cidt_ids,
        cidt_targets=cidt_targets,
        cidt_baseline_predictions=cidt_baseline,
        cidt_registry_sha256=scientific.cidt_registry_sha256(
            cidt_ids, cidt_targets, cidt_baseline
        ),
        frozen_record_sha256=_component_id("temporary-lineage-record", 0),
    )
    return replace(lock, frozen_record_sha256=scientific.lineage_lock_sha256(lock))


def _refresh_artifact_lock(
    lock: scientific.LineageLock,
    run: scientific.RawScientificRun,
) -> scientific.LineageLock:
    updated = replace(
        lock,
        artifact_registry_sha256=scientific.artifact_registry_sha256(
            run.artifact_identities
        ),
    )
    return replace(
        updated,
        frozen_record_sha256=scientific.lineage_lock_sha256(updated),
    )


@lru_cache(maxsize=1)
def _raw_evidence() -> scientific.RawScientificEvidence:
    sample_indices, targets, held_folds = _raw_axes()
    component_ids, component_order = _raw_components(held_folds)
    keeper = _keeper_probabilities(sample_indices, targets, held_folds)
    role_scores = _role_raw_scores(sample_indices, targets)
    sidecars = _sidecar_preimages(sample_indices, held_folds, role_scores)
    attention, evidence, valid64, valid32, valid16, bbox = _raw_maps(role_scores[0])
    spatial32, spatial16, channel16, channel32 = _raw_filters()
    cidt_indices, cidt_targets, cidt_baseline = _raw_cidt(
        sample_indices, targets, keeper
    )
    causal = np.stack(
        (
            role_scores[0, :, 0],
            role_scores[0, :, 0],
            _binary_score(sample_indices, targets, signal=0.06, salt=211),
            _binary_score(sample_indices, targets, signal=0.12, salt=223),
            _binary_score(sample_indices, targets, signal=0.10, salt=227),
            _binary_score(sample_indices, targets, signal=0.09, salt=229),
        ),
        axis=0,
    )
    invalid_scores = np.stack((causal[0], causal[0]), axis=0)
    invalid_maps = np.stack((attention, attention), axis=0)
    run = scientific.RawScientificRun(
        sample_indices=sample_indices,
        targets=targets,
        held_folds=held_folds,
        component_ids=component_ids,
        component_order=component_order,
        bootstrap_draws=generate_component_bootstrap_draws(),
        keeper_probabilities=keeper,
        sidecar_preimages=sidecars,
        causal_scores=np.ascontiguousarray(causal, dtype="<f8"),
        invalid_fill_scores=invalid_scores,
        invalid_fill_maps=invalid_maps,
        attention_maps=attention,
        evidence_maps=evidence,
        valid64=valid64,
        valid32=valid32,
        valid16=valid16,
        bbox_valid16=bbox,
        nonwrap_offsets=_nonwrap_offsets(sample_indices, held_folds),
        donor_artifact_sha256=_component_id("donor-assignment", 0),
        artifact_identities=(),
        spatial_filters32=spatial32,
        spatial_filters16=spatial16,
        channel_filters16=channel16,
        channel_filters32=channel32,
        cidt_sample_indices=cidt_indices,
        cidt_targets=cidt_targets,
        cidt_baseline_predictions=cidt_baseline,
        trace=_raw_trace(),
    )
    run = replace(run, artifact_identities=scientific._expected_artifact_identities(run))
    lock = _lineage_lock(run)
    return scientific.RawScientificEvidence(lineage=lock, formal=run, replay=run)


@lru_cache(maxsize=1)
def _raw_result() -> scientific.ConjunctiveGateEvaluationV2:
    return evaluate_synthetic_conjunctive_gates(_raw_evidence())


def test_raw_evaluator_derives_calibrators_actions_metrics_and_fixed_registry() -> None:
    result = _raw_result()
    assert result.all_passed
    assert set(result.gates) == set(range(1, 15))
    assert len(result.comparison_records) == 69
    assert json_sha256([list(item) for item in REQUIRED_COMPARISON_RECORDS]) == (
        REQUIRED_COMPARISON_REGISTRY_SHA256
    )
    assert result.formal.held_raw_scores.shape == (5, 763, 4)
    assert result.formal.union_probabilities.shape == (5, 4, 763)
    assert result.formal.pair_probabilities.shape == (5, 3, 2, 763)
    assert result.formal.action_probabilities.shape == (2, 763)
    assert len(result.formal.calibrators) == 250
    assert result.formal.primary_action_metrics.keeper_class1_tp == 528
    assert result.formal.primary_action_metrics.restricted_fp_denominator == 222
    assert result.formal.primary_action_metrics.restricted_fp_rejected >= 45
    assert result.formal.repeat_comparison.eligible_rows == 750
    assert result.formal.cidt.outside_cohort_rows == 8452
    assert result.gates[14].details["record_count"] == 69


def test_action_policy_is_derived_from_calibration_not_a_stored_point_two() -> None:
    assert "action_policies" not in scientific.RawScientificRun.__dataclass_fields__
    assert "calibrated_pair_probabilities" not in scientific.RawScientificRun.__dataclass_fields__
    probabilities = np.asarray([0.1, 0.9], dtype=np.dtype("<f8"))
    keeper = np.asarray(
        [[0.3, 0.6, 0.03, 0.02, 0.05], [0.3, 0.6, 0.03, 0.02, 0.05]],
        dtype=np.dtype("<f8"),
    )
    selection = select_calibration_action_threshold(
        probabilities, keeper, np.asarray([0, 1], dtype=np.dtype("<i8"))
    )
    assert selection.policy.numeric_threshold == 0.1
    assert selection.policy.threshold_token != float(0.2).hex()


def test_frozen_lineage_rejects_target_oracle_sidecar_mutation() -> None:
    # This is mutation integrity, not artifact-origin or release-order security.
    evidence = _raw_evidence()
    first = evidence.formal.sidecar_preimages[0]
    held = first.held_raw_scores.copy()
    held[:, 0] = (evidence.formal.targets[evidence.formal.held_folds == 0] == 1)
    sidecars = (replace(first, held_raw_scores=held),) + evidence.formal.sidecar_preimages[1:]
    poisoned = replace(evidence.formal, sidecar_preimages=sidecars)
    with pytest.raises(ValueError, match="Sidecar preimages differ"):
        evaluate_synthetic_conjunctive_gates(replace(evidence, formal=poisoned))


def test_frozen_lineage_rejects_keeper_same_argmax_probability_mutation() -> None:
    evidence = _raw_evidence()
    keeper = evidence.formal.keeper_probabilities.copy()
    assert int(np.argmax(keeper[0])) == 1
    keeper[0, 2] += 0.01
    keeper[0, 3] -= 0.01
    assert int(np.argmax(keeper[0])) == 1
    poisoned = replace(evidence.formal, keeper_probabilities=keeper)
    with pytest.raises(ValueError, match="keeper probabilities differs"):
        evaluate_synthetic_conjunctive_gates(replace(evidence, formal=poisoned))


def test_frozen_lineage_rejects_within_fold_sample_component_swap() -> None:
    evidence = _raw_evidence()
    components = list(evidence.formal.component_ids)
    assert evidence.formal.held_folds[0] == evidence.formal.held_folds[1]
    components[0], components[1] = components[1], components[0]
    poisoned = replace(evidence.formal, component_ids=tuple(components))
    with pytest.raises(ValueError, match="Sample-to-component mapping"):
        evaluate_synthetic_conjunctive_gates(replace(evidence, formal=poisoned))


def test_frozen_lineage_rejects_single_geometry_cell_mutation() -> None:
    evidence = _raw_evidence()
    valid64 = evidence.formal.valid64.copy()
    valid64[0, 0, 0] = False
    poisoned = replace(evidence.formal, valid64=valid64)
    with pytest.raises(ValueError, match="valid64 differs"):
        evaluate_synthetic_conjunctive_gates(replace(evidence, formal=poisoned))


def test_map_score_mismatch_survives_registry_refresh_and_is_rejected() -> None:
    evidence = _raw_evidence()
    maps = evidence.formal.evidence_maps.copy()
    maps[0, 0, 0, 0] += 1.0
    run = replace(evidence.formal, evidence_maps=maps, artifact_identities=())
    run = replace(run, artifact_identities=scientific._expected_artifact_identities(run))
    lock = _refresh_artifact_lock(evidence.lineage, run)
    with pytest.raises(ValueError, match="Map-derived DDF scores"):
        evaluate_synthetic_conjunctive_gates(
            replace(evidence, lineage=lock, formal=run)
        )


def test_causal_clean_must_equal_primary_raw_union() -> None:
    evidence = _raw_evidence()
    causal = evidence.formal.causal_scores.copy()
    causal[0, 0] += 1.0
    run = replace(evidence.formal, causal_scores=causal, artifact_identities=())
    run = replace(run, artifact_identities=scientific._expected_artifact_identities(run))
    lock = _refresh_artifact_lock(evidence.lineage, run)
    with pytest.raises(ValueError, match="Causal clean scores"):
        evaluate_synthetic_conjunctive_gates(
            replace(evidence, lineage=lock, formal=run)
        )


def test_replay_trace_mismatch_fails_generated_gate14() -> None:
    evidence = _raw_evidence()
    losses = evidence.replay.trace.losses.copy()
    losses[0, 0, 0] += 1e-12
    replay = replace(evidence.replay, trace=replace(evidence.replay.trace, losses=losses))
    result = evaluate_synthetic_conjunctive_gates(replace(evidence, replay=replay))
    assert not result.gates[14].passed
    assert not result.all_passed
    assert "trace/losses_f64/array_exact" in result.gates[14].details["failed_records"]


def test_gate4_requires_four_joint_fold_wins() -> None:
    evidence = _raw_evidence()
    formal = _raw_result().formal
    fold_auroc = formal.fold_auroc.copy()
    kup = scientific.EVALUATION_SCORE_NAMES.index("KUP")
    keeper = scientific.EVALUATION_SCORE_NAMES.index("K")
    static = scientific.EVALUATION_SCORE_NAMES.index("static_KUP")
    fold_auroc[kup] = 0.80
    fold_auroc[keeper] = np.asarray([0.79, 0.79, 0.79, 0.79, 0.81])
    fold_auroc[static] = np.asarray([0.81, 0.79, 0.79, 0.79, 0.79])
    gates = scientific._gates_1_to_13(
        evidence.formal, replace(formal, fold_auroc=fold_auroc)
    )
    assert gates[4].details["joint_wins"] == 3
    assert not gates[4].passed


def test_exact_entry_is_unconditionally_closed_until_runner_is_frozen() -> None:
    assert not hasattr(scientific, "GuardVerifiedEvidenceToken")
    assert not hasattr(scientific, "FormalEvidenceVerifier")
    assert not hasattr(scientific, "_verify_guard_token")
    with pytest.raises(
        scientific.IncompleteScientificBoundaryError,
        match="trusted fresh-process runner",
    ):
        evaluate_exact_a0_conjunctive_gates(_raw_evidence())
    assert "non-production" in (evaluate_synthetic_conjunctive_gates.__doc__ or "")

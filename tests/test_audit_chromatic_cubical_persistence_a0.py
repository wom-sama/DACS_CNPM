from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gudhi", minversion="3.11.0")

from trkh.tools.audit_chromatic_cubical_persistence_a0 import (  # noqa: E402
    CHANNEL_NAMES,
    CONDITIONS,
    DIAGRAM_FEATURE_DIM,
    FIT_FOLDS,
    GLOBAL_COLOR_DIM,
    HOMOLOGY_DIMENSIONS,
    POLARITY_NAMES,
    ROLE_DIM,
    ROLES,
    ROI_SIZE,
    TOPOLOGY_DIM,
    _action_predictions,
    _coordinate_geometry,
    _nearest_valid_fill,
    _predict_saved_readout,
    _read_prediction_artifact,
    _source_mappings,
    _validate_locked_args,
    _write_predictions,
    assemble_role_features,
    assess_chromatic_persistence_a0,
    build_chromatic_rank_maps,
    cubical_topology_descriptor,
    parse_args,
    pixel_permutation_placebo,
    select_retention_threshold,
    synthetic_oracle_checks,
    vectorize_persistence_intervals,
)


def _chromatic_surface() -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, ROI_SIZE, dtype=np.float32)
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    red = 55.0 + 170.0 * x
    green = 190.0 - 100.0 * y
    blue = 40.0 + 80.0 * np.square(x - y)
    return np.clip(np.rint(np.stack((red, green, blue), axis=2)), 0, 255).astype(
        np.uint8
    )


def _role_metric(
    *,
    auc: float,
    removals: int,
    breaks: int = 0,
    tp_retention: float = 1.0,
    fn_accepted: int = 11,
    correlation: float = 1.0,
) -> dict[str, object]:
    per_fold = {
        str(fold): {
            "tp_rows": 100,
            "tp_breaks": 1 if breaks else 0,
            "restricted_fp_rows": 40,
            "restricted_fp_removals": max(2, removals // 4),
        }
        for fold in FIT_FOLDS
    }
    return {
        "auc": auc,
        "tp_retention": tp_retention,
        "tp_breaks": breaks,
        "fn_accepted": fn_accepted,
        "restricted_fp_rejection": removals / 186.0,
        "restricted_fp_removals": removals,
        "corrections": removals,
        "harms": breaks,
        "clean_score_spearman": correlation,
        "per_fold": per_fold,
    }


def _passing_results() -> dict[str, dict[str, dict[str, object]]]:
    results: dict[str, dict[str, dict[str, object]]] = {}
    for condition, _brightness, _contrast in CONDITIONS:
        shifted = condition != "clean"
        candidate = _role_metric(
            auc=0.68 if shifted else 0.75,
            removals=28 if shifted else 38,
            breaks=10 if shifted else 4,
            tp_retention=0.97 if shifted else 0.99,
            fn_accepted=8 if shifted else 9,
            correlation=0.80,
        )
        results[condition] = {
            "candidate": candidate,
            "control": _role_metric(auc=0.64 if shifted else 0.69, removals=20),
            "pixel_placebo": _role_metric(
                auc=0.64 if shifted else 0.68, removals=20
            ),
            "source_placebo": _role_metric(
                auc=0.64 if shifted else 0.67, removals=20
            ),
        }
    return results


def test_protocol_dimensions_are_internally_consistent() -> None:
    assert TOPOLOGY_DIM == (
        len(CHANNEL_NAMES)
        * len(POLARITY_NAMES)
        * len(HOMOLOGY_DIMENSIONS)
        * DIAGRAM_FEATURE_DIM
    )
    assert ROLE_DIM == 5 + GLOBAL_COLOR_DIM + TOPOLOGY_DIM


def test_locked_a0_rejects_cpu_execution() -> None:
    args = parse_args(["--device", "cpu"])
    with pytest.raises(ValueError, match="device=cuda"):
        _validate_locked_args(args)


def test_nearest_valid_fill_uses_existing_values() -> None:
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    valid = np.zeros((5, 5), dtype=bool)
    valid[0, 0] = True
    valid[4, 4] = True
    filled = _nearest_valid_fill(values, valid)
    assert set(np.unique(filled).tolist()) == {0.0, 24.0}
    assert filled[0, 0] == 0.0
    assert filled[4, 4] == 24.0


def test_rank_maps_are_finite_and_keep_uniform_roi_valid() -> None:
    uniform = np.full((ROI_SIZE, ROI_SIZE, 3), (150, 150, 70), dtype=np.uint8)
    control, maps, valid, telemetry = build_chromatic_rank_maps(uniform)
    _x, _y, ellipse = _coordinate_geometry()
    assert control.shape == (GLOBAL_COLOR_DIM,)
    assert maps.shape == (len(CHANNEL_NAMES), ROI_SIZE, ROI_SIZE)
    assert np.isfinite(maps).all()
    assert int(valid.sum()) == int(ellipse.sum())
    assert telemetry["highlight_fraction"] == 0.0


def test_synthetic_h0_h1_oracle_passes() -> None:
    result = synthetic_oracle_checks()
    assert result["passed"] is True
    assert result["checks"]["two_components_h0_finite_one"] is True
    assert result["checks"]["ring_h1_finite_one"] is True


def test_persistence_vector_and_descriptor_are_rotation_invariant() -> None:
    vector = vectorize_persistence_intervals(
        np.asarray([[0.1, 0.4], [0.2, 0.9]], dtype=np.float64)
    )
    assert vector.shape == (DIAGRAM_FEATURE_DIM,)
    assert vector[-5] == 2.0

    _control, maps, _valid, _telemetry = build_chromatic_rank_maps(
        _chromatic_surface()
    )
    original, _counts = cubical_topology_descriptor(maps)
    rotated, _counts = cubical_topology_descriptor(
        np.rot90(maps, axes=(1, 2)).copy()
    )
    assert original.shape == (TOPOLOGY_DIM,)
    assert np.allclose(original, rotated, atol=1e-6, rtol=1e-6)


def test_pixel_placebo_is_deterministic_and_histogram_exact() -> None:
    _control, maps, _valid, _telemetry = build_chromatic_rank_maps(
        _chromatic_surface()
    )
    first = pixel_permutation_placebo(maps, sample_index=17)
    second = pixel_permutation_placebo(maps, sample_index=17)
    assert np.array_equal(first, second)
    for channel in range(len(CHANNEL_NAMES)):
        assert np.array_equal(
            np.sort(first[channel], axis=None), np.sort(maps[channel], axis=None)
        )
        assert not np.array_equal(first[channel], maps[channel])


def test_role_features_match_dimensions_and_placebos() -> None:
    rng = np.random.default_rng(9)
    rows = 12
    probabilities = rng.dirichlet(np.ones(5), size=rows)
    global_color = rng.normal(size=(rows, GLOBAL_COLOR_DIM))
    topology = rng.normal(size=(rows, TOPOLOGY_DIM))
    pixel = rng.normal(size=(rows, TOPOLOGY_DIM))
    mapping = np.arange(rows - 1, -1, -1)
    features = {
        role: assemble_role_features(
            probabilities=probabilities,
            global_color=global_color,
            topology=topology,
            pixel_topology=pixel,
            role=role,
            source_mapping=mapping if role == "source_placebo" else None,
        )
        for role in ROLES
    }
    assert all(value.shape == (rows, ROLE_DIM) for value in features.values())
    assert np.count_nonzero(features["control"][:, -TOPOLOGY_DIM:]) == 0
    assert np.array_equal(features["candidate"][:, -TOPOLOGY_DIM:], topology)
    assert np.array_equal(features["pixel_placebo"][:, -TOPOLOGY_DIM:], pixel)
    assert np.array_equal(
        features["source_placebo"][:, -TOPOLOGY_DIM:], topology[mapping]
    )


def test_source_mappings_stay_within_partitions_and_change_source() -> None:
    folds = np.tile(np.asarray(FIT_FOLDS, dtype=np.int64), 20)
    sources = np.asarray([f"source_{index}" for index in range(folds.size)])
    positions = np.arange(folds.size)
    mappings = _source_mappings(folds, sources)
    for outer_fold, mapping in mappings.items():
        assert np.all(sources[mapping] != sources)
        assert set(mapping[folds == outer_fold]) == set(positions[folds == outer_fold])
        assert set(mapping[folds != outer_fold]) == set(positions[folds != outer_fold])


def test_retention_threshold_selects_largest_eligible_score() -> None:
    scores = np.asarray([0.9, 0.8, 0.4, 0.3, 0.2, 0.7, 0.1])
    labels = np.asarray([1, 1, 1, 1, 1, 0, 0])
    threshold, record = select_retention_threshold(
        scores, labels, target_retention=0.8
    )
    assert threshold == pytest.approx(0.3)
    assert record["inner_positive_retention"] == pytest.approx(0.8)


def test_saved_readout_prediction_reconstructs_sigmoid() -> None:
    coefficient = np.zeros(ROLE_DIM, dtype=np.float64)
    coefficient[0] = 2.0
    record = {
        "scaler_mean": np.zeros(ROLE_DIM).tolist(),
        "scaler_scale": np.ones(ROLE_DIM).tolist(),
        "coef": coefficient.tolist(),
        "intercept": -1.0,
    }
    features = np.zeros((2, ROLE_DIM), dtype=np.float64)
    features[:, 0] = (0.5, 1.0)
    probabilities = _predict_saved_readout(record, features)
    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_gate_requires_candidate_to_beat_controls_and_placebos() -> None:
    structural = {"all_structural_checks": True}
    passing = assess_chromatic_persistence_a0(
        structural_checks=structural, results=_passing_results()
    )
    assert passing["automatic_gates_passed"] is True
    assert passing["full_train_authorized"] is False

    failing_results = _passing_results()
    failing_results["clean"]["candidate"]["auc"] = 0.70
    failing = assess_chromatic_persistence_a0(
        structural_checks=structural, results=failing_results
    )
    assert failing["automatic_gates_passed"] is False
    assert failing["clean_checks"]["auc_gain_over_pixel_placebo"] is False


def test_prediction_artifact_round_trips_probabilities_and_actions(
    tmp_path: Path,
) -> None:
    rows = 618
    selected = np.arange(rows, dtype=np.int64)
    probabilities = np.tile(
        np.asarray([[0.10, 0.55, 0.15, 0.10, 0.10]], dtype=np.float64),
        (rows, 1),
    )
    conditions = {
        name: {
            "probabilities": probabilities.copy(),
            "target": np.where(selected < 432, 1, 0),
            "fold": np.asarray(FIT_FOLDS, dtype=np.int64)[selected % 4],
            "source": np.asarray([f"source_{index}" for index in selected]),
        }
        for name, _brightness, _contrast in CONDITIONS
    }
    accepted = selected % 3 != 0
    outputs = {
        name: {
            role: {
                "score": np.linspace(0.0, 1.0, rows),
                "threshold": np.full(rows, 0.5),
                "accepted": accepted.copy(),
            }
            for role in ROLES
        }
        for name, _brightness, _contrast in CONDITIONS
    }
    path = tmp_path / "predictions.csv"
    _write_predictions(
        path,
        outputs=outputs,
        conditions=conditions,
        selected_indices=selected,
    )
    replay_outputs, replay_rows, replay_selected = _read_prediction_artifact(path)
    assert np.array_equal(replay_selected, selected)
    for name, _brightness, _contrast in CONDITIONS:
        assert np.array_equal(replay_rows[name]["probabilities"], probabilities)
        for role in ROLES:
            _base, expected_action = _action_predictions(probabilities, accepted)
            assert np.array_equal(
                replay_outputs[name][role]["action_prediction"], expected_action
            )

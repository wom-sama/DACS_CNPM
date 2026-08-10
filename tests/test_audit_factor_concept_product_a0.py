from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import trkh.tools.audit_factor_concept_product_a0 as factor_a0


def _probability(prediction: int, confidence: float = 0.90) -> np.ndarray:
    values = np.full(5, (1.0 - confidence) / 4.0, dtype=np.float64)
    values[int(prediction)] = confidence
    return values


def _synthetic_predictions() -> tuple[np.ndarray, ...]:
    targets = []
    folds = []
    keeper = []
    control = []
    factor = []
    maturity = []
    transport = []
    sum_placebo = []
    context_placebo = []

    def add(
        target: int,
        fold: int,
        keeper_probability: np.ndarray,
        control_probability: np.ndarray,
        factor_probability: np.ndarray,
        transport_probability: np.ndarray,
    ) -> None:
        targets.append(target)
        folds.append(fold)
        keeper.append(keeper_probability)
        control.append(control_probability)
        factor.append(factor_probability)
        maturity_probability = np.full(3, 0.05, dtype=np.float64)
        maturity_probability[factor_a0.MATURITY_CODE[target]] = 0.90
        maturity.append(maturity_probability)
        transport.append(transport_probability)
        sum_placebo.append([0.25, 0.50, 0.25])
        context_placebo.append([0.25, 0.50, 0.25])

    for fold in range(5):
        for _ in range(20):
            add(
                1,
                fold,
                _probability(1, 0.80),
                _probability(1, 0.80),
                _probability(1, 0.90),
                np.asarray([0.05, 0.90, 0.05]),
            )
        for _ in range(2):
            add(
                1,
                fold,
                _probability(0, 0.60),
                _probability(1, 0.70),
                _probability(1, 0.90),
                np.asarray([0.05, 0.90, 0.05]),
            )
        for index in range(4):
            add(
                0,
                fold,
                _probability(1, 0.60),
                _probability(1, 0.60),
                _probability(0, 0.90)
                if index < 2
                else _probability(1, 0.55),
                np.asarray([0.85, 0.10, 0.05]),
            )
        for target in (0, 2, 3, 4):
            for _ in range(8):
                transport_state = factor_a0.TRANSPORT_CODE[target]
                transport_probability = np.full(3, 0.05, dtype=np.float64)
                transport_probability[transport_state] = 0.90
                add(
                    target,
                    fold,
                    _probability(target),
                    _probability(target),
                    _probability(target),
                    transport_probability,
                )
    return tuple(
        np.asarray(values, dtype=np.int64 if index < 2 else np.float64)
        for index, values in enumerate(
            (
                targets,
                folds,
                keeper,
                control,
                factor,
                maturity,
                transport,
                sum_placebo,
                context_placebo,
            )
        )
    )


def test_factor_product_uses_only_five_valid_conjunctions() -> None:
    maturity = np.asarray([[0.6, 0.3, 0.1]], dtype=np.float64)
    transport = np.asarray([[0.2, 0.5, 0.3]], dtype=np.float64)
    observed = factor_a0.factor_product_probabilities(maturity, transport)
    expected = np.asarray([[0.12, 0.30, 0.06, 0.09, 0.03]], dtype=np.float64)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)
    assert int(observed.argmax(axis=1)[0]) == 1


def test_expand_probabilities_preserves_classifier_class_order() -> None:
    observed = factor_a0._expand_probabilities(
        np.asarray([0, 2]),
        np.asarray([[0.25, 0.75], [0.80, 0.20]]),
        class_count=3,
    )
    np.testing.assert_allclose(observed, [[0.25, 0.0, 0.75], [0.80, 0.0, 0.20]])


def test_analysis_passes_only_for_selective_factor_veto() -> None:
    values = _synthetic_predictions()
    analysis = factor_a0.analyze_predictions(*values)
    assert analysis["mechanism_gates_passed"] is True
    assert analysis["factor_vs_control"]["restricted_fp_net_removal"] == 10
    assert analysis["factor_vs_control"]["candidate_harm"] == 0
    assert analysis["direction"]["signed_transport_tp_vs_fp_auroc"] == pytest.approx(1.0)
    assert analysis["direction"]["sum_placebo_tp_vs_fp_auroc"] == pytest.approx(0.5)
    assert analysis["direction"]["factor_minus_control_p1_fn_vs_fp_auroc"] == pytest.approx(1.0)
    assert all(analysis["mechanism_gates"].values())


def test_analysis_rejects_broad_class1_contraction() -> None:
    values = list(_synthetic_predictions())
    targets, control, factor = values[0], values[3], values[4].copy()
    control_predictions = control.argmax(axis=1)
    true_positive = (targets == 1) & (control_predictions == 1)
    factor[true_positive] = _probability(0, 0.90)
    values[4] = factor
    analysis = factor_a0.analyze_predictions(*values)
    assert analysis["mechanism_gates_passed"] is False
    assert analysis["mechanism_gates"]["class1_recall_delta_ge_minus0005"] is False
    assert analysis["mechanism_gates"]["focus_tp_break_not_above_rescue"] is False


def test_source_folds_are_complete_and_source_disjoint() -> None:
    base_labels = np.tile(np.arange(5, dtype=np.int64), 10)
    labels = np.repeat(base_labels, 2)
    groups = np.repeat([f"source_{index}" for index in range(base_labels.size)], 2)
    assignments, summary = factor_a0.assign_source_folds(labels, groups)
    assert assignments.shape == labels.shape
    assert set(assignments.tolist()) == set(range(5))
    assert summary["source_overlap"] == 0
    for fold in range(5):
        fit_sources = set(groups[assignments != fold].tolist())
        holdout_sources = set(groups[assignments == fold].tolist())
        assert not fit_sources.intersection(holdout_sources)


@pytest.mark.parametrize(
    "path",
    [
        r"D:\dataset\images\val\a.jpg",
        r"D:\dataset\images\validation\a.jpg",
        r"D:\dataset\images\test\a.jpg",
    ],
)
def test_train_path_guard_rejects_validation_and_test(path: str) -> None:
    assert factor_a0._contains_forbidden_split(path) is True
    assert factor_a0._is_train_image_path(path) is False


def test_output_directory_must_be_empty(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    assert factor_a0._require_empty_output(output) == output.resolve()
    (output / "payload.txt").write_text("locked", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        factor_a0._require_empty_output(output)


def test_manifest_and_replay_recompute_analysis(tmp_path: Path) -> None:
    values = _synthetic_predictions()
    targets, folds = values[:2]
    cache = {
        "labels": targets,
        "sample_index": np.arange(targets.size),
        "paths": np.asarray([str(tmp_path / f"image_{index}.jpg") for index in range(targets.size)]),
        "source_stem": np.asarray([f"image_{index}" for index in range(targets.size)]),
        "classes": factor_a0.EXPECTED_CLASSES,
        "probabilities": values[2],
    }
    outputs = {
        "control": values[3],
        "factor": values[4],
        "maturity": values[5],
        "transport": values[6],
        "transport_sum_placebo": values[7],
        "transport_context_placebo": values[8],
    }
    rows = factor_a0._prediction_rows(cache=cache, folds=folds, outputs=outputs)
    factor_a0._write_csv(tmp_path / "predictions.csv", rows)
    analysis = factor_a0.analyze_predictions(*values)
    factor_a0._write_json(tmp_path / "summary.json", {"analysis": analysis})
    factor_a0._write_manifest(tmp_path)
    replay = factor_a0.replay_summary(tmp_path / "summary.json")
    assert replay["replay_passed"] is True
    assert replay["rows"] == targets.size
    assert replay["maximum_numeric_difference"] <= 1e-12
    assert replay["payload_count"] == 2


def test_manifest_detects_payload_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"value": 1}), encoding="utf-8")
    factor_a0._write_manifest(tmp_path)
    payload.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        factor_a0._verify_manifest(tmp_path)


def test_manifest_detects_unrecorded_payload(tmp_path: Path) -> None:
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    factor_a0._write_manifest(tmp_path)
    (tmp_path / "late.txt").write_text("late", encoding="utf-8")
    with pytest.raises(ValueError, match="payload set differs"):
        factor_a0._verify_manifest(tmp_path)

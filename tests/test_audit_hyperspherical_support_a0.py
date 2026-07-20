from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import trkh.tools.audit_hyperspherical_support_a0 as support_a0


def _probability(prediction: int, confidence: float = 0.90) -> np.ndarray:
    values = np.full(5, (1.0 - confidence) / 4.0, dtype=np.float64)
    values[int(prediction)] = confidence
    return values


def _passing_predictions() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    targets = []
    folds = []
    outputs = {role: [] for role in support_a0.ROLES}

    def add(target: int, fold: int, duplicate_prediction: int, candidate_prediction: int) -> None:
        targets.append(target)
        folds.append(fold)
        duplicate = _probability(duplicate_prediction)
        candidate = _probability(candidate_prediction)
        deranged = _probability(0 if target == 1 else 1)
        outputs["natural_control"].append(duplicate)
        outputs["duplicate_control"].append(duplicate)
        outputs["cap_candidate"].append(candidate)
        outputs["cap_seed_repeat"].append(candidate)
        outputs["nearest_center_deranged"].append(deranged)

    for fold in range(5):
        for index in range(20):
            add(1, fold, 1 if index < 18 else 0, 1)
        for target in (0, 2, 4):
            for index in range(20):
                add(target, fold, 1 if index < 4 else target, target)
        for _ in range(20):
            add(3, fold, 3, 3)
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
        {
            role: np.asarray(probabilities, dtype=np.float64)
            for role, probabilities in outputs.items()
        },
    )


def test_generated_support_respects_cap_and_does_not_touch_global_rng() -> None:
    rng = np.random.default_rng(7)
    features = rng.standard_normal((80, 16))
    state_before = np.random.get_state()
    generated, record = support_a0.generate_hyperspherical_support(
        features, count=120, seed=99
    )
    state_after = np.random.get_state()
    assert generated.shape == (120, 16)
    np.testing.assert_allclose(np.linalg.norm(generated, axis=1), 1.0, atol=1e-12)
    assert record["maximum_cap_angle_excess"] <= 1e-12
    assert record["maximum_tangent_abs_dot"] <= 1e-12
    assert all(
        np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        for left, right in zip(state_before, state_after)
    )


def test_duplicate_balance_produces_exact_counts() -> None:
    features = np.arange(18, dtype=np.float64).reshape(9, 2) + 1.0
    labels = np.asarray([0, 0, 0, 1, 2, 2, 3, 4, 4], dtype=np.int64)
    balanced, balanced_labels, record = support_a0.duplicate_balance(
        features, labels, seed=11
    )
    assert balanced.shape == (15, 2)
    assert np.bincount(balanced_labels, minlength=5).tolist() == [3, 3, 3, 3, 3]
    assert record["balanced_counts"] == [3, 3, 3, 3, 3]


def test_nearest_rival_never_selects_self() -> None:
    centers = np.eye(5, dtype=np.float64)
    centers[1] = np.asarray([0.9, 0.1, 0.0, 0.0, 0.0])
    rivals = support_a0.nearest_rival_centers(centers)
    assert rivals.shape == (5,)
    assert all(int(rivals[index]) != index for index in range(5))
    assert int(rivals[1]) == 0


def test_analysis_passes_only_for_selective_candidate() -> None:
    targets, folds, outputs = _passing_predictions()
    analysis = support_a0.analyze_predictions(targets, folds, outputs)
    assert analysis["mechanism_gates_passed"] is True
    assert analysis["transitions"]["candidate_vs_duplicate"][
        "restricted_fp_net_removal"
    ] == 60
    assert analysis["transitions"]["candidate_vs_duplicate"]["focus_tp_net"] == 10
    assert analysis["fold_gate_counts"]["precision_nonworse_folds"] == 5
    assert all(analysis["mechanism_gates"].values())


def test_analysis_rejects_broad_focus_contraction() -> None:
    targets, folds, outputs = _passing_predictions()
    candidate = outputs["cap_candidate"].copy()
    candidate[targets == 1] = _probability(0)
    outputs["cap_candidate"] = candidate
    analysis = support_a0.analyze_predictions(targets, folds, outputs)
    assert analysis["mechanism_gates_passed"] is False
    assert analysis["mechanism_gates"]["focus_recall_ge_075"] is False
    assert analysis["mechanism_gates"]["focus_tp_net_ge_minus2"] is False


def test_source_folds_are_complete_and_disjoint() -> None:
    base_labels = np.tile(np.arange(5, dtype=np.int64), 5)
    labels = np.repeat(base_labels, 2)
    groups = np.repeat([f"source_{index}" for index in range(base_labels.size)], 2)
    folds = np.repeat(np.arange(5, dtype=np.int64), 10)
    summary = support_a0._summarize_folds(labels, groups, folds)
    assert summary["assignment_complete"] is True
    assert summary["source_overlap"] == 0
    assert summary["assignment_counts"] == [10, 10, 10, 10, 10]


@pytest.mark.parametrize(
    "path",
    [
        r"D:\dataset\images\val\a.jpg",
        r"D:\dataset\images\validation\a.jpg",
        r"D:\dataset\images\test\a.jpg",
    ],
)
def test_train_path_guard_rejects_validation_and_test(path: str) -> None:
    assert support_a0._contains_forbidden_split(path) is True
    assert support_a0._is_train_image_path(path) is False


def test_output_directory_must_be_empty(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    assert support_a0._require_empty_output(output) == output.resolve()
    (output / "payload.txt").write_text("locked", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        support_a0._require_empty_output(output)


def test_manifest_and_replay_recompute_analysis(tmp_path: Path) -> None:
    targets, folds, outputs = _passing_predictions()
    cache = {
        "labels": targets,
        "sample_index": np.arange(targets.size, dtype=np.int64),
        "paths": np.asarray(
            [str(tmp_path / "images" / "train" / f"image_{i}.jpg") for i in range(targets.size)]
        ),
        "source_stems": np.asarray([f"image_{i}" for i in range(targets.size)]),
    }
    support_a0._write_csv(
        tmp_path / "predictions.csv",
        support_a0._prediction_rows(cache=cache, folds=folds, outputs=outputs),
    )
    analysis = support_a0.analyze_predictions(targets, folds, outputs)
    support_a0._write_json(tmp_path / "summary.json", {"analysis": analysis})
    support_a0._write_manifest(tmp_path)
    replay = support_a0.replay_summary(tmp_path / "summary.json")
    assert replay["replay_passed"] is True
    assert replay["rows"] == targets.size
    assert replay["maximum_numeric_difference"] <= 1e-12


def test_manifest_detects_payload_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"value": 1}), encoding="utf-8")
    support_a0._write_manifest(tmp_path)
    payload.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        support_a0._verify_manifest(tmp_path)

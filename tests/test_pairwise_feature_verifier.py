import numpy as np
import pytest

from trkh.tools.probe_pairwise_feature_verifier import (
    _fit_pair_models,
    _pair_verifier_probabilities_for_split,
    apply_pairwise_verifiers,
    build_verifier_features,
    parse_pairs,
)


class FakePairModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_proba(self, features):
        return self.probabilities[: features.shape[0]]


def test_parse_pairs_deduplicates_and_sorts() -> None:
    assert parse_pairs("1-0,1-2,2-1") == [(0, 1), (1, 2)]


def test_parse_pairs_rejects_self_pair() -> None:
    with pytest.raises(ValueError):
        parse_pairs("1-1")


def test_build_verifier_features_appends_probability_context() -> None:
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    probabilities = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)
    features = build_verifier_features(embeddings, probabilities)
    assert features.shape == (1, 10)
    assert np.allclose(features[0, :2], embeddings[0])
    assert np.allclose(features[0, 2:5], probabilities[0])
    assert np.isclose(features[0, -2], 0.5)
    assert np.isclose(features[0, -1], 0.7)


def test_apply_pairwise_verifiers_uses_fixed_thresholds() -> None:
    probabilities = np.array(
        [
            [0.35, 0.39, 0.10, 0.08, 0.08],
            [0.02, 0.80, 0.10, 0.04, 0.04],
        ],
        dtype=np.float32,
    )
    base = np.array([1, 1], dtype=np.int64)
    features = np.zeros((2, 8), dtype=np.float32)
    models = {(0, 1): FakePairModel([[0.75, 0.25], [0.75, 0.25]])}
    final, changes = apply_pairwise_verifiers(
        probabilities,
        base,
        models,
        min_pair_probability=0.02,
        max_pair_margin=0.40,
        verifier_confidence_threshold=0.60,
        features=features,
    )
    assert final.tolist() == [0, 1]
    assert len(changes) == 1
    assert changes[0]["pair"] == "0-1"


def test_fit_pair_models_accepts_sample_weights() -> None:
    features = np.array(
        [
            [0.0],
            [0.1],
            [0.9],
            [1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    weights = np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32)

    models, summaries, oof = _fit_pair_models(
        features,
        labels,
        [(0, 1)],
        c_value=1.0,
        max_iter=100,
        seed=7,
        oof_folds=2,
        sample_weights=weights,
    )

    assert (0, 1) in models
    assert summaries[0]["sample_weight_min"] == 0.5
    assert summaries[0]["sample_weight_max"] == 1.0
    assert oof[(0, 1)].shape == (4, 2)


def test_fit_pair_models_supports_extra_trees_verifier() -> None:
    features = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [0.8, 0.9],
            [1.0, 0.7],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    models, summaries, oof = _fit_pair_models(
        features,
        labels,
        [(0, 1)],
        model_type="extra_trees",
        c_value=1.0,
        max_iter=100,
        seed=7,
        oof_folds=2,
        extra_trees_n_estimators=20,
        extra_trees_min_samples_leaf=1,
    )

    assert (0, 1) in models
    assert summaries[0]["model_type"] == "extra_trees"
    assert oof[(0, 1)].shape == (4, 2)


def test_pair_verifier_probabilities_use_oof_for_train_pair_rows() -> None:
    features = np.zeros((4, 2), dtype=np.float32)
    labels = np.array([0, 2, 1, 4], dtype=np.int64)
    model = FakePairModel(
        [
            [0.10, 0.90],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.40, 0.60],
        ]
    )
    oof = {(0, 1): np.array([[0.91, 0.09], [0.12, 0.88]], dtype=np.float32)}

    pair_probs = _pair_verifier_probabilities_for_split(
        split="train",
        features=features,
        labels=labels,
        models={(0, 1): model},
        oof_probabilities=oof,
    )[(0, 1)]

    assert np.allclose(pair_probs[0], [0.91, 0.09])
    assert np.allclose(pair_probs[2], [0.12, 0.88])
    assert np.allclose(pair_probs[1], [0.20, 0.80])
    assert np.allclose(pair_probs[3], [0.40, 0.60])

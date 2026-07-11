import numpy as np

from trkh.tools.probe_embedding_spectral_noise import (
    _class_summary,
    score_embedding_spectral_noise,
)


def test_spectral_noise_scores_flag_neighbor_disagreement() -> None:
    embeddings = np.asarray(
        [
            [1.00, 0.00],
            [0.95, 0.05],
            [0.00, 1.00],
            [0.02, 0.98],
            [0.10, 0.90],
            [0.90, 0.10],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    base_predictions = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.85, 0.15],
            [0.30, 0.70],
            [0.15, 0.85],
            [0.20, 0.80],
            [0.65, 0.35],
        ],
        dtype=np.float32,
    )

    scores = score_embedding_spectral_noise(
        embeddings,
        labels,
        class_count=2,
        probabilities=probabilities,
        base_predictions=base_predictions,
        neighbors_k=2,
    )

    assert scores["neighbor_same_label_fraction"][0] == 0.5
    assert scores["neighbor_same_label_fraction"][2] == 0.0
    assert scores["neighbor_same_label_fraction"][5] == 0.0
    assert scores["ambiguity_score"][2] > scores["ambiguity_score"][1]
    assert scores["ambiguity_score"][5] > scores["ambiguity_score"][4]
    assert scores["base_error"].tolist() == [0, 0, 1, 0, 0, 1]
    assert np.all(scores["fine_clean_score"] >= 0.0)
    assert np.all(scores["fine_clean_score"] <= 1.0)


def test_class_summary_reports_top_risk_error_rate() -> None:
    embeddings = np.asarray(
        [
            [1.00, 0.00],
            [0.95, 0.05],
            [0.00, 1.00],
            [0.02, 0.98],
            [0.10, 0.90],
            [0.90, 0.10],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = score_embedding_spectral_noise(
        embeddings,
        labels,
        class_count=2,
        base_predictions=np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64),
        neighbors_k=2,
    )

    summary = _class_summary(
        labels=labels,
        scores=scores,
        class_names=["a", "b"],
        top_fraction=1.0 / 3.0,
    )

    assert summary["0"]["samples"] == 3
    assert summary["0"]["top_count"] == 1
    assert summary["0"]["top_risk_base_error_rate"] == 1.0
    assert summary["1"]["top_risk_base_error_rate"] == 1.0

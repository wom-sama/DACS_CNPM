import numpy as np
import torch

from trkh.tools.probe_embedding_prototypes import (
    _classification_metrics,
    _collate_classification,
    _fit_predict_etf_ridge,
    _predict_knn_cosine,
    _predict_nearest_centroid_cosine,
    _predict_nearest_centroid_euclidean,
    _predict_soft_centroid_cosine,
    _predict_t3a_templates,
    _simplex_etf_targets,
    _temperature_scale_probabilities,
    _write_embedding_cache,
)


def test_centroid_and_knn_predictions_separate_tiny_embeddings():
    train_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    eval_embeddings = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)

    assert _predict_nearest_centroid_cosine(
        train_embeddings,
        train_labels,
        eval_embeddings,
        2,
    ).tolist() == [0, 1]
    assert _predict_nearest_centroid_euclidean(
        train_embeddings,
        train_labels,
        eval_embeddings,
        2,
    ).tolist() == [0, 1]
    assert _predict_knn_cosine(
        train_embeddings,
        train_labels,
        eval_embeddings,
        2,
        k=1,
    ).tolist() == [0, 1]


def test_classification_metrics_reports_class1_f1():
    targets = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)
    metrics = _classification_metrics(targets, predictions, ["c0", "c1"])

    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]
    class1 = metrics["per_class"][1]
    assert class1["tp"] == 2
    assert class1["fp"] == 1
    assert class1["fn"] == 1
    assert abs(class1["f1"] - (2.0 / 3.0)) < 1e-6


def test_simplex_etf_targets_have_equal_pairwise_angles():
    targets = _simplex_etf_targets(4)
    gram = targets @ targets.T

    assert np.allclose(np.diag(gram), 1.0)
    off_diag = gram[~np.eye(4, dtype=bool)]
    assert np.allclose(off_diag, -1.0 / 3.0)


def test_etf_ridge_predictions_separate_tiny_embeddings():
    train_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    eval_embeddings = np.asarray([[0.85, 0.15], [0.15, 0.85]], dtype=np.float32)

    predictions = _fit_predict_etf_ridge(
        train_embeddings,
        train_labels,
        eval_embeddings,
        2,
        alpha=0.1,
        class_weight="balanced",
    )

    assert predictions is not None
    assert predictions.tolist() == [0, 1]


def test_t3a_templates_use_low_entropy_target_supports():
    eval_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.55, 0.45],
            [0.10, 0.90],
            [0.45, 0.55],
        ],
        dtype=np.float32,
    )
    templates = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    predictions, stats = _predict_t3a_templates(
        eval_embeddings,
        probabilities,
        templates,
        2,
        support_per_class=1,
        source_weight=0.0,
        min_confidence=0.0,
    )

    assert predictions.tolist() == [0, 0, 1, 1]
    assert stats["selected_counts"] == [1, 1]


def test_t3a_templates_fall_back_to_source_when_no_target_support():
    eval_embeddings = np.asarray([[0.0, 1.0]], dtype=np.float32)
    probabilities = np.asarray([[0.80, 0.20]], dtype=np.float32)
    templates = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    predictions, stats = _predict_t3a_templates(
        eval_embeddings,
        probabilities,
        templates,
        2,
        support_per_class=2,
        source_weight=1.0,
        min_confidence=0.95,
    )

    assert predictions.tolist() == [1]
    assert stats["selected_counts"] == [0, 0]


def test_soft_centroid_uses_train_probabilities_without_hard_labels():
    train_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    train_probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.15, 0.85],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )
    eval_embeddings = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)

    predictions, stats = _predict_soft_centroid_cosine(
        train_embeddings,
        train_probabilities,
        eval_embeddings,
        2,
        temperature=1.0,
        min_confidence=0.0,
    )

    assert predictions.tolist() == [0, 1]
    assert stats["kept_rows"] == 4
    assert stats["class_weight_sums"][0] > 0.0
    assert stats["class_weight_sums"][1] > 0.0


def test_temperature_scale_probabilities_sharpens_when_temperature_is_low():
    probabilities = np.asarray([[0.70, 0.30]], dtype=np.float32)

    sharpened = _temperature_scale_probabilities(probabilities, temperature=0.5)
    softened = _temperature_scale_probabilities(probabilities, temperature=2.0)

    assert sharpened[0, 0] > probabilities[0, 0]
    assert softened[0, 0] < probabilities[0, 0]
    assert np.allclose(sharpened.sum(axis=1), 1.0)
    assert np.allclose(softened.sum(axis=1), 1.0)


def test_collate_classification_preserves_bbox_and_mask_metadata():
    batch = [
        (
            torch.zeros(3, 4, 4),
            0,
            {
                "bbox": torch.tensor([0.5, 0.5, 0.4, 0.4]),
                "image_mask": torch.ones(4, 4, dtype=torch.bool),
                "image_path": "a.jpg",
            },
        ),
        (
            torch.ones(3, 4, 4),
            1,
            {
                "bbox": torch.tensor([0.4, 0.4, 0.3, 0.3]),
                "image_mask": torch.zeros(4, 4, dtype=torch.bool),
                "image_path": "b.jpg",
            },
        ),
    ]

    images, labels, metadata = _collate_classification(batch)

    assert images.shape == (2, 3, 4, 4)
    assert labels.tolist() == [0, 1]
    assert metadata["paths"] == ["a.jpg", "b.jpg"]
    assert metadata["bbox"].shape == (2, 4)
    assert metadata["image_mask"].shape == (2, 4, 4)


def test_write_embedding_cache_preserves_alignment(tmp_path):
    path = tmp_path / "val_embeddings.npz"
    payload = {
        "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "probabilities": np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "base_predictions": np.asarray([0, 1], dtype=np.int64),
        "paths": ["a.jpg", "b.jpg"],
    }

    _write_embedding_cache(path, payload)

    cache = np.load(path, allow_pickle=True)
    assert cache["embeddings"].shape == (2, 2)
    assert cache["sample_index"].tolist() == [0, 1]
    assert cache["paths"].tolist() == ["a.jpg", "b.jpg"]

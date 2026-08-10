import numpy as np

from trkh.tools.audit_self_neighbor_consistency import (
    _align_feature_cache,
    _auc_positive_greater,
    _compute_neighbor_rows,
    _deduplicate_review_rows,
    _decision_hint,
    _js_divergence,
    _nearest_neighbors,
)


def test_align_feature_cache_uses_explicit_feature_key(tmp_path) -> None:
    cache_path = tmp_path / "features.npz"
    np.savez_compressed(
        cache_path,
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        source_embeddings=np.asarray([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32),
        sample_index=np.asarray([7, 8], dtype=np.int64),
        labels=np.asarray([0, 1], dtype=np.int64),
    )

    aligned = _align_feature_cache(
        prediction_sample_indices=np.asarray([8, 7], dtype=np.int64),
        feature_cache_npz=cache_path,
        feature_key="source_embeddings",
    )

    assert aligned["feature_key"] == "source_embeddings"
    assert np.allclose(aligned["features"][0], np.asarray([0.8, 0.6], dtype=np.float32))


def test_js_divergence_is_zero_for_identical_distributions() -> None:
    probs = np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32)

    jsd = _js_divergence(probs, probs)

    assert float(jsd[0]) < 1e-7


def test_auc_positive_greater_handles_ties() -> None:
    auc = _auc_positive_greater([0.5, 0.7], [0.5, 0.6])

    assert auc == 0.625


def test_review_deduplication_prevents_queue_overlap_weighting() -> None:
    rows = [
        {"sample_index": 10, "strict_review_side": "recall_protector", "review_source_csv": "a.csv"},
        {"sample_index": 10, "strict_review_side": "recall_protector", "review_source_csv": "b.csv"},
        {"sample_index": 11, "strict_review_side": "fp_suppressor", "review_source_csv": "a.csv"},
    ]

    unique_rows, summary = _deduplicate_review_rows(rows)

    assert len(unique_rows) == 2
    assert summary["duplicate_row_count"] == 1
    assert unique_rows[0]["review_duplicate_count"] == 2


def test_decision_hint_never_grants_smoke_permission() -> None:
    decision = _decision_hint(
        {
            "auc_recall_gt_suppressor": {
                "neighbor_class1_label_fraction": 0.95,
                "neighbor_class1_probability": 0.90,
                "clean_like_score": 0.80,
            }
        }
    )

    assert decision["signal_separation_passed"] is True
    assert decision["smoke_ready"] is False
    assert decision["training_permission"] is False


def test_nearest_neighbors_excludes_same_source() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    stems = np.asarray(["a", "a", "b", "c"], dtype=object)

    neighbors = _nearest_neighbors(
        features,
        stems,
        top_k=1,
        chunk_size=2,
        include_same_source=False,
    )

    assert neighbors[0, 0] == 2


def test_compute_neighbor_rows_reports_class1_support() -> None:
    sample_indices = np.asarray([10, 11, 12], dtype=np.int64)
    targets = np.asarray([1, 0, 1], dtype=np.int64)
    preds = np.asarray([1, 1, 0], dtype=np.int64)
    probs = np.asarray(
        [
            [0.1, 0.8, 0.1],
            [0.3, 0.5, 0.2],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    confidences = probs[np.arange(3), preds]
    labels = targets.copy()
    neighbor_indices = np.asarray([[2, 1], [0, 2], [0, 1]], dtype=np.int64)

    rows = _compute_neighbor_rows(
        sample_indices=sample_indices,
        targets=targets,
        preds=preds,
        probs=probs,
        confidences=confidences,
        labels=labels,
        neighbor_indices=neighbor_indices,
        class1_index=1,
    )

    assert rows[0]["neighbor_class1_label_fraction"] == 0.5
    assert rows[1]["neighbor_class1_label_fraction"] == 1.0
    assert rows[2]["neighbor_same_target_fraction"] == 0.5

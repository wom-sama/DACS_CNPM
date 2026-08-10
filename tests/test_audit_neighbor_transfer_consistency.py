import numpy as np

from trkh.tools.audit_neighbor_transfer_consistency import (
    _decision,
    _nearest_reference_neighbors,
)


def test_nearest_reference_neighbors_excludes_same_source() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    reference = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    query_stems = np.asarray(["same"], dtype=object)
    reference_stems = np.asarray(["same", "other", "far"], dtype=object)

    neighbors = _nearest_reference_neighbors(
        query,
        reference,
        query_stems,
        reference_stems,
        top_k=1,
        chunk_size=1,
        include_same_source=False,
    )

    assert neighbors.tolist() == [[1]]


def test_decision_requires_stable_auc_across_k() -> None:
    per_k = {
        "15": {
            "auc_recall_gt_suppressor": {
                "neighbor_class1_label_fraction": 0.80,
                "neighbor_class1_probability": 0.75,
                "same_target_fraction": 0.72,
                "clean_like_score": 0.74,
            }
        },
        "30": {
            "auc_recall_gt_suppressor": {
                "neighbor_class1_label_fraction": 0.60,
                "neighbor_class1_probability": 0.65,
                "same_target_fraction": 0.62,
                "clean_like_score": 0.64,
            }
        },
    }

    decision = _decision(per_k)

    assert decision["signal_transfer_passed"] is False
    assert decision["smoke_ready"] is False
    assert decision["training_permission"] is False

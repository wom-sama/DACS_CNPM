import numpy as np
import pytest

from trkh.tools.build_local_neighbor_feature_cache import build_local_neighbor_targets


def _row_norms(values):
    return np.linalg.norm(np.asarray(values, dtype=np.float32), axis=1)


def test_local_neighbor_targets_are_normalized_and_sample_aligned():
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 2, 2, 0, 0], dtype=np.int64)

    targets, summary = build_local_neighbor_targets(
        embeddings,
        labels,
        min_neighbor_rank=1,
        max_neighbor_rank=2,
        neighbors_per_sample=1,
        blend_alpha=0.10,
        focus_class_index=1,
        focus_blend_alpha=0.30,
        boundary_rival_classes={0, 2},
        boundary_blend_alpha=0.20,
    )

    assert targets.shape == embeddings.shape
    assert np.allclose(_row_norms(targets), 1.0, atol=1e-6)
    assert summary["rows_with_neighbor_target"] == len(labels)
    assert summary["by_reason"]["focus_same_class_neighbor"] == 2
    assert summary["by_reason"]["same_class_neighbor"] == 4
    assert summary["fallback_rows"] == 0


def test_focus_class_alpha_moves_target_more_than_base_alpha():
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)

    targets, _ = build_local_neighbor_targets(
        embeddings,
        labels,
        min_neighbor_rank=1,
        max_neighbor_rank=1,
        neighbors_per_sample=1,
        blend_alpha=0.10,
        focus_class_index=1,
        focus_blend_alpha=0.40,
    )
    normalized = embeddings / np.maximum(_row_norms(embeddings)[:, None], 1e-12)
    focus_shift = 1.0 - float(np.dot(targets[0], normalized[0]))
    base_shift = 1.0 - float(np.dot(targets[2], normalized[2]))

    assert focus_shift > base_shift


def test_boundary_predicted_focus_uses_boundary_alpha():
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 2, 2], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.10, 0.80, 0.10],
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
        ],
        dtype=np.float32,
    )

    _, summary = build_local_neighbor_targets(
        embeddings,
        labels,
        probabilities=probabilities,
        min_neighbor_rank=1,
        max_neighbor_rank=1,
        neighbors_per_sample=1,
        blend_alpha=0.10,
        focus_class_index=1,
        boundary_rival_classes={0},
        boundary_blend_alpha=0.25,
    )

    assert summary["by_reason"]["boundary_predicted_focus_neighbor"] == 1
    assert summary["by_reason"]["same_class_neighbor"] == 3


def test_local_neighbor_targets_validate_empty_and_mismatched_inputs():
    with pytest.raises(ValueError, match="empty embedding"):
        build_local_neighbor_targets(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    with pytest.raises(ValueError, match="length mismatch"):
        build_local_neighbor_targets(
            np.ones((2, 4), dtype=np.float32),
            np.ones((1,), dtype=np.int64),
        )

import numpy as np
import torch

from trkh.tools.probe_patch_part_prototypes import (
    _select_patch_tokens,
    build_part_histogram_features,
)


def test_select_patch_tokens_prefers_high_bbox_weight_and_falls_back_to_valid():
    patches = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    valid = torch.tensor(
        [
            [True, True, True, True],
            [False, True, True, False],
        ]
    )
    weights = torch.tensor(
        [
            [0.01, 0.90, 0.20, 0.70],
            [0.00, 0.01, 0.02, 0.00],
        ],
        dtype=torch.float32,
    )

    selected, owners, selected_weights = _select_patch_tokens(
        patches,
        valid_mask=valid,
        patch_weights=weights,
        max_patches_per_sample=2,
        bbox_threshold=0.05,
    )

    assert selected.shape == (4, 3)
    assert owners.tolist() == [0, 0, 1, 1]
    assert selected[:2].tolist() == [patches[0, 1].tolist(), patches[0, 3].tolist()]
    assert selected[2:].tolist() == [patches[1, 1].tolist(), patches[1, 2].tolist()]
    assert np.all(selected_weights > 0.0)


def test_part_histogram_features_are_sample_index_aligned_and_normalized():
    centroids = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    patch_tokens = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    owners = np.asarray([0, 0, 1, 1], dtype=np.int64)
    weights = np.asarray([1.0, 3.0, 2.0, 2.0], dtype=np.float32)

    features = build_part_histogram_features(
        patch_tokens=patch_tokens,
        patch_sample_indices=owners,
        patch_weights=weights,
        centroids=centroids,
        sample_count=3,
    )

    # K histogram + K max-similarity + entropy/active/peak/mean-similarity.
    assert features.shape == (3, 8)
    np.testing.assert_allclose(features[0, :2], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(features[1, :2], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(features[2, :2], [0.0, 0.0], atol=1e-6)
    assert features[0, 2] > 0.99
    assert features[1, 3] > 0.99
    assert features[0, 6] == 1.0
    assert features[2, 6] == 0.0

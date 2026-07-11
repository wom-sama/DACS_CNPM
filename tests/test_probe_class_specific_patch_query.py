import numpy as np
import torch

from trkh.tools.probe_class_specific_patch_query import (
    ClassSpecificPatchQueryReadout,
    build_padded_patch_cache,
)


def test_build_padded_patch_cache_keeps_owner_order_and_empty_fallback() -> None:
    tokens = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [9.0, 9.0],
        ],
        dtype=np.float32,
    )
    owners = np.array([0, 0, 2, 9], dtype=np.int64)
    weights = np.array([0.2, 0.3, 0.4, 0.9], dtype=np.float32)

    padded, mask, padded_weights, counts = build_padded_patch_cache(
        patch_tokens=tokens,
        patch_sample_indices=owners,
        patch_weights=weights,
        sample_count=3,
        max_patches_per_sample=2,
    )

    assert padded.shape == (3, 2, 2)
    assert np.allclose(padded[0], [[1.0, 0.0], [2.0, 0.0]])
    assert np.allclose(padded[2, 0], [0.0, 3.0])
    assert counts.tolist() == [2, 0, 1]
    assert mask.tolist() == [[True, True], [True, False], [True, False]]
    assert np.allclose(padded_weights[1], [1.0, 0.0])


def test_class_specific_patch_query_readout_masks_invalid_tokens() -> None:
    module = ClassSpecificPatchQueryReadout(
        dim=4,
        num_classes=3,
        attention_temperature=0.5,
        dropout=0.0,
        base_logit_scale=1.0,
        residual_logit_scale=0.2,
    )
    patches = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [
            [True, True, False, False, False],
            [False, False, False, False, False],
        ]
    )
    weights = torch.ones(2, 5)
    base = torch.full((2, 3), 1.0 / 3.0)

    logits, attention = module(patches, mask, weights, base, return_attention=True)

    assert logits.shape == (2, 3)
    assert attention.shape == (2, 3, 5)
    assert torch.allclose(attention[0, :, 2:], torch.zeros(3, 3), atol=1e-6)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 3), atol=1e-5)

from __future__ import annotations

import torch

from trkh.training.train import _paired_view_patch_token_alignment_loss


def test_paired_view_patch_token_alignment_prefers_matching_local_tokens() -> None:
    primary = {
        "patches": torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.2, 0.2, 0.2],
                ]
            ]
        )
    }
    matching = {
        "patches": torch.tensor(
            [
                [
                    [3.5, 0.0, 0.0],
                    [0.0, 2.5, 0.0],
                    [0.0, 0.0, 0.4],
                    [0.1, 0.1, 0.1],
                ]
            ]
        )
    }
    mismatched = {
        "patches": torch.tensor(
            [
                [
                    [0.0, 0.0, 4.0],
                    [0.0, 0.0, 3.0],
                    [0.5, 0.0, 0.0],
                    [0.2, 0.2, 0.2],
                ]
            ]
        )
    }

    good_loss = _paired_view_patch_token_alignment_loss(
        primary_features=primary,
        paired_features=matching,
        top_k=2,
        temperature=0.08,
    )
    bad_loss = _paired_view_patch_token_alignment_loss(
        primary_features=primary,
        paired_features=mismatched,
        top_k=2,
        temperature=0.08,
    )

    assert good_loss is not None
    assert bad_loss is not None
    assert good_loss.item() < bad_loss.item()


def test_paired_view_patch_token_alignment_respects_padding_mask() -> None:
    primary_patches = torch.randn(2, 4, 6, requires_grad=True)
    paired_patches = primary_patches.detach().clone().requires_grad_(True)
    features = {
        "patches": primary_patches,
        "key_padding_mask": torch.tensor(
            [
                [False, False, True, True],
                [False, True, True, True],
            ]
        ),
    }
    paired_features = {
        "patches": paired_patches,
        "key_padding_mask": torch.tensor(
            [
                [False, False, True, True],
                [False, True, True, True],
            ]
        ),
    }

    loss = _paired_view_patch_token_alignment_loss(
        primary_features=features,
        paired_features=paired_features,
        top_k=3,
        temperature=0.12,
    )

    assert loss is not None
    loss.backward()
    assert primary_patches.grad is not None
    assert paired_patches.grad is not None

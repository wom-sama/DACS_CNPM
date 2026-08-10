import torch

from trkh.tools.probe_bbox_token_region_separability import (
    _region_weights,
    _weighted_token_mean,
)


def test_region_weights_select_foreground_and_background_tokens() -> None:
    prior = torch.tensor([[0.0, 0.1, 0.7, 1.0]], dtype=torch.float32)

    foreground, background, valid, stats = _region_weights(
        bbox_prior=prior,
        foreground_threshold=0.2,
        background_threshold=0.2,
    )

    assert valid.tolist() == [[True, True, True, True]]
    assert torch.allclose(foreground.sum(dim=1), torch.ones(1))
    assert torch.allclose(background.sum(dim=1), torch.ones(1))
    assert foreground[0, 2] > 0
    assert foreground[0, 3] > foreground[0, 2]
    assert background[0, 0] > 0
    assert background[0, 1] > 0
    assert stats["empty_foreground_fallbacks"] == 0
    assert stats["empty_background_fallbacks"] == 0


def test_region_weights_fallback_when_no_background_survives() -> None:
    prior = torch.tensor([[0.9, 1.0]], dtype=torch.float32)

    _foreground, background, _valid, stats = _region_weights(
        bbox_prior=prior,
        foreground_threshold=0.2,
        background_threshold=0.05,
    )

    assert torch.allclose(background.sum(dim=1), torch.ones(1))
    assert stats["empty_background_fallbacks"] == 1


def test_weighted_token_mean_uses_normalized_weights() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [3.0, 2.0]]])
    weights = torch.tensor([[0.25, 0.75]])

    pooled = _weighted_token_mean(tokens, weights)

    assert torch.allclose(pooled, torch.tensor([[2.5, 1.5]]))

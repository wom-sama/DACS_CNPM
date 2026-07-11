import torch

from trkh.tools.evaluate_paired_view_fusion import (
    _fused_probabilities,
    parse_weight_grid,
)


def test_parse_weight_grid_defaults_and_deduplicates():
    assert parse_weight_grid("") == [round(index / 10.0, 2) for index in range(11)]
    assert parse_weight_grid("0.2, 0.1;0.2") == [0.1, 0.2]


def test_fused_probabilities_logit_and_probability_modes():
    primary = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    paired = torch.tensor([[0.0, 2.0], [2.0, 0.0]])

    primary_probs = _fused_probabilities(
        primary_logits=primary,
        paired_logits=paired,
        weight=0.0,
        mode="logit",
    )
    assert torch.equal(primary_probs.argmax(dim=1), torch.tensor([0, 1]))

    paired_probs = _fused_probabilities(
        primary_logits=primary,
        paired_logits=paired,
        weight=1.0,
        mode="probability",
    )
    assert torch.equal(paired_probs.argmax(dim=1), torch.tensor([1, 0]))

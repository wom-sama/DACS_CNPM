from __future__ import annotations

import torch

from trkh.tools.screen_dinov3_margin_subpatch_b28 import (
    DINO_FEATURE_DIM,
    select_margin_degradation_route,
)


def test_margin_degradation_selects_largest_predicted_boundary_drop() -> None:
    attention = torch.full((2, 3, 256), 1e-4)
    attention[0, 0, :3] = torch.tensor([0.7, 0.2, 0.1])
    attention[0, 1, 3:6] = torch.tensor([0.7, 0.2, 0.1])
    attention[0, 2, 6:9] = torch.tensor([0.7, 0.2, 0.1])
    attention[1] = attention[0]
    attention /= attention.sum(dim=-1, keepdim=True)

    degraded = torch.zeros(2, 3, DINO_FEATURE_DIM)
    degraded[0, :, 0] = torch.tensor([1.0, 0.0, 1.5])
    degraded[1, :, 0] = torch.tensor([1.0, 1.5, 0.0])
    folds = torch.tensor([0, 1], dtype=torch.long)
    base_scores = torch.tensor([[2.0, 0.0, -1.0, -2.0, -3.0]]).repeat(2, 1)
    coefficients = torch.zeros(5, 5, DINO_FEATURE_DIM)
    coefficients[:, 0, 0] = 1.0
    intercepts = torch.zeros(5, 5)
    means = torch.zeros(5, DINO_FEATURE_DIM)
    scales = torch.ones(5, DINO_FEATURE_DIM)

    result = select_margin_degradation_route(
        attention,
        degraded,
        folds,
        base_scores,
        coefficients,
        intercepts,
        means,
        scales,
    )

    assert result["selected_head"].tolist() == [1, 2]
    assert result["indices"][0].tolist() == [3, 4, 5]
    assert result["indices"][1].tolist() == [6, 7, 8]
    assert torch.allclose(result["margin_drop"], torch.tensor([2.0, 2.0]))
    assert torch.isfinite(result["attention_entropy"]).all()

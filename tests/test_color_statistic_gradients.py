from __future__ import annotations

import torch

from trkh.models.model import ColorStatisticFusion


def test_lab_chroma_is_input_gradient_safe_at_zero_chroma() -> None:
    module = ColorStatisticFusion(num_classes=1, hidden_dim=16, dropout=0.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    normalized_black = ((torch.zeros(1, 3, 32, 32) - mean) / std).requires_grad_(
        True
    )

    statistics = module.extract_stats(normalized_black)
    gradient = torch.autograd.grad(statistics.sum(), normalized_black)[0]

    assert torch.isfinite(statistics).all()
    assert torch.isfinite(gradient).all()
    assert int(torch.count_nonzero(gradient)) > 0

from __future__ import annotations

import numpy as np
import torch

from trkh.tools.extract_pwca_token_cache import (
    _overlay_selected_tokens,
    select_top_attention_tokens,
    selected_bbox_fraction,
)


def test_top_attention_selection_excludes_padding_and_preserves_mass() -> None:
    patches = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    attention = torch.tensor(
        [[0.05, 0.40, 0.30, 0.20, 0.05], [0.10, 0.20, 0.50, 0.15, 0.05]]
    )
    indices = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    valid = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    selected, selected_indices, weights, mass = select_top_attention_tokens(
        patches,
        attention,
        indices,
        valid,
        top_k=2,
    )
    assert selected.shape == (2, 2, 3)
    assert selected_indices.tolist() == [[1, 2], [7, 6]]
    assert torch.allclose(weights, torch.tensor([[0.40, 0.30], [0.50, 0.20]]))
    assert torch.allclose(mass, torch.tensor([0.70 / 0.75, 0.70 / 0.95]))


def test_bbox_fraction_and_overlay_are_well_formed() -> None:
    indices = torch.tensor([[0, 5, 10, 15]])
    bbox = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    fraction = selected_bbox_fraction(indices, bbox, grid_size=(4, 4))
    assert torch.allclose(fraction, torch.tensor([0.5]))
    rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    overlay = _overlay_selected_tokens(
        rgb,
        np.asarray([0, 5]),
        np.asarray([0.2, 0.8]),
        grid_size=(4, 4),
    )
    assert overlay.shape == rgb.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, rgb)

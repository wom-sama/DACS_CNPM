from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from trkh.tools.screen_dinov3_sparse_subpatch_b27 import (
    DETAIL_CHANNELS,
    DINO_FEATURE_DIM,
    TOP_K,
    _selected_subpatch_descriptor,
    aggregate_contrasts,
    haar_contrasts,
    spatial_half_shift,
)


def test_spatial_half_shift_is_an_involution_without_identity() -> None:
    indices = torch.tensor([[0, 15, 240], [17, 88, 255]], dtype=torch.long)
    shifted = spatial_half_shift(indices)
    assert torch.equal(spatial_half_shift(shifted), indices)
    assert not torch.any(shifted == indices)
    assert int(shifted.min()) >= 0 and int(shifted.max()) < 256


def test_haar_contrasts_have_expected_signed_axes() -> None:
    quadrants = torch.tensor([[[[1.0], [3.0], [5.0], [7.0]]]])
    observed = haar_contrasts(quadrants)
    expected = torch.tensor([[[[2.0], [4.0], [0.0]]]])
    assert torch.equal(observed, expected)


def test_aggregate_contrasts_normalizes_route_weights() -> None:
    contrasts = torch.zeros(1, 3, DETAIL_CHANNELS, 2)
    contrasts[:, 0] = 1.0
    contrasts[:, 1] = 2.0
    contrasts[:, 2] = 4.0
    observed = aggregate_contrasts(contrasts, torch.tensor([[1.0, 1.0, 2.0]]))
    assert torch.allclose(observed, torch.full((1, DETAIL_CHANNELS, 2), 2.75))


def test_selected_subpatch_descriptor_has_locked_geometry_and_signal() -> None:
    generator = torch.Generator().manual_seed(7)
    projection = nn.Conv2d(3, DINO_FEATURE_DIM, kernel_size=16, stride=16, bias=True)
    model = SimpleNamespace(
        patch_embed=SimpleNamespace(proj=projection, norm=nn.Identity())
    )
    images = torch.randn(2, 3, 256, 256, generator=generator)
    indices = torch.tensor([[0, 1, 16], [17, 18, 34]], dtype=torch.long)
    weights = torch.ones(2, TOP_K)
    descriptor = _selected_subpatch_descriptor(model, images, indices, weights)
    assert descriptor.shape == (2, DETAIL_CHANNELS, DINO_FEATURE_DIM)
    assert torch.isfinite(descriptor).all()
    assert float(descriptor.abs().max()) > 0.0

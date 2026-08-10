"""Reusable luminance-only relighting for classification training."""

from __future__ import annotations

import torch
from torch import Tensor

from trkh.core.config import IMAGENET_MEAN, IMAGENET_STD


def balanced_polarities(
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return shuffled dim/bright signs whose counts differ by at most one."""

    if count < 0:
        raise ValueError("count must be non-negative")
    pair_count = count // 2
    polarity = torch.cat(
        (
            -torch.ones(pair_count, device=device, dtype=dtype),
            torch.ones(pair_count, device=device, dtype=dtype),
        )
    )
    if count % 2:
        extra = torch.where(
            torch.rand((), device=device, generator=generator) < 0.5,
            torch.full((), -1.0, device=device, dtype=dtype),
            torch.full((), 1.0, device=device, dtype=dtype),
        ).reshape(1)
        polarity = torch.cat((polarity, extra))
    if count > 1:
        polarity = polarity.index_select(
            0,
            torch.randperm(count, device=device, generator=generator),
        )
    return polarity


def relight_luminance(
    images: Tensor,
    polarities: Tensor,
    *,
    brightness: float = 0.25,
    contrast: float = 0.10,
) -> Tensor:
    """Relight normalized RGB while preserving RGB differences until clipping."""

    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError("relighting expects normalized RGB tensor (B, 3, H, W)")
    signs = polarities.to(device=images.device, dtype=images.dtype).reshape(-1)
    if signs.numel() != images.size(0):
        raise ValueError("one relighting polarity is required per image")
    if not bool(torch.all((signs == -1) | (signs == 1)).item()):
        raise ValueError("relighting polarities must be exactly -1 or +1")
    brightness_delta = float(brightness)
    contrast_delta = float(contrast)
    if not 0.0 <= brightness_delta <= 0.95:
        raise ValueError("relighting brightness must be in [0, 0.95]")
    if not 0.0 <= contrast_delta <= 0.95:
        raise ValueError("relighting contrast must be in [0, 0.95]")

    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = images * std + mean
    signs = signs.view(-1, 1, 1, 1)
    luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    luma_mean = luma.mean(dim=(2, 3), keepdim=True)
    relit_luma = (
        (luma - luma_mean) * (1.0 + signs * contrast_delta) + luma_mean
    ) * (1.0 + signs * brightness_delta)
    relit_rgb = (rgb + relit_luma - luma).clamp(0.0, 1.0)
    return ((relit_rgb - mean) / std).to(dtype=images.dtype)

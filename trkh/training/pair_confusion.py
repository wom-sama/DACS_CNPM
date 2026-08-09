from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from trkh.training.confusion_spectral import differentiable_margin_confusion


@dataclass(frozen=True)
class BidirectionalPairConfusionResult:
    loss: Tensor
    batch_confusion: Tensor
    ema_confusion: Tensor
    directional_confusions: Tensor


def bidirectional_pair_confusion_mean(
    logits: Tensor,
    targets: Tensor,
    *,
    previous_ema: Tensor | None = None,
    momentum: float = 0.5,
    margin: float = 0.1,
) -> BidirectionalPairConfusionResult:
    """Average both class-conditional error directions of a two-class pair.

    For a two-class off-diagonal confusion matrix ``[[0, a], [b, 0]]``, the
    spectral norm is ``max(|a|, |b|)`` and can therefore ignore the smaller
    direction.  This functional uses ``(a + b) / 2``.  Each confusion column is
    already a per-true-class mean, so both directions receive equal explicit
    pressure without introducing a dataset-frequency scale.
    """

    if logits.ndim != 2 or logits.size(1) != 2:
        raise ValueError("pair logits must have shape [B, 2].")
    if targets.ndim != 1 or targets.numel() != logits.size(0):
        raise ValueError("pair targets must have shape [B].")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("momentum must be in [0, 1).")
    resolved_targets = targets.to(device=logits.device, dtype=torch.long)
    if bool(((resolved_targets < 0) | (resolved_targets > 1)).any().item()):
        raise ValueError("pair targets must contain only 0 and 1.")
    if not bool((resolved_targets == 0).any().item()) or not bool(
        (resolved_targets == 1).any().item()
    ):
        raise ValueError("both pair classes must be present.")

    batch_confusion = differentiable_margin_confusion(
        logits,
        resolved_targets,
        margin=margin,
    )
    if previous_ema is None:
        previous = torch.zeros_like(batch_confusion)
    else:
        if tuple(previous_ema.shape) != (2, 2):
            raise ValueError("previous_ema must have shape [2, 2].")
        previous = previous_ema.detach().to(
            device=batch_confusion.device,
            dtype=batch_confusion.dtype,
        )
        if not torch.isfinite(previous).all():
            raise ValueError("previous_ema must be finite.")
    ema_confusion = (
        float(momentum) * previous
        + (1.0 - float(momentum)) * batch_confusion
    )
    directional_confusions = torch.stack(
        (ema_confusion[0, 1], ema_confusion[1, 0])
    )
    loss = directional_confusions.float().mean().to(dtype=logits.dtype)
    return BidirectionalPairConfusionResult(
        loss=loss,
        batch_confusion=batch_confusion,
        ema_confusion=ema_confusion.detach(),
        directional_confusions=directional_confusions,
    )

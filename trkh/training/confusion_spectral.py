from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConfusionSpectralResult:
    loss: Tensor
    batch_confusion: Tensor
    ema_confusion: Tensor
    class_weights: Tensor
    weighted_confusion: Tensor


@dataclass
class ConfusionSpectralEMAState:
    ema_confusion: Tensor | None = None
    updates: int = 0

    def update(self, ema_confusion: Tensor) -> None:
        if ema_confusion.ndim != 2 or ema_confusion.size(0) != ema_confusion.size(1):
            raise ValueError("ema_confusion must be a square matrix.")
        if not torch.isfinite(ema_confusion).all():
            raise ValueError("ema_confusion must be finite.")
        self.ema_confusion = ema_confusion.detach()
        self.updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "ema_confusion": (
                self.ema_confusion.detach().cpu()
                if self.ema_confusion is not None
                else None
            ),
            "updates": int(self.updates),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, object] | None,
        *,
        num_classes: int,
        device: torch.device,
    ) -> "ConfusionSpectralEMAState":
        state = cls()
        if payload is None:
            return state
        updates = int(payload.get("updates", 0) or 0)
        if updates < 0:
            raise ValueError("confusion spectral update count must be non-negative.")
        raw_ema = payload.get("ema_confusion")
        if raw_ema is None:
            if updates != 0:
                raise ValueError("confusion spectral EMA is missing despite non-zero updates.")
            return state
        ema = torch.as_tensor(raw_ema, dtype=torch.float32, device=device)
        expected_shape = (int(num_classes), int(num_classes))
        if tuple(ema.shape) != expected_shape:
            raise ValueError(
                f"confusion spectral EMA shape {tuple(ema.shape)} != {expected_shape}."
            )
        if not torch.isfinite(ema).all():
            raise ValueError("confusion spectral EMA checkpoint contains non-finite values.")
        state.ema_confusion = ema.detach()
        state.updates = updates
        return state


def confusion_frequency_weights(
    class_counts: Sequence[int] | Tensor,
    *,
    smoothing: float = 0.2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    counts = torch.as_tensor(class_counts, device=device, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("class_counts must be a one-dimensional vector with at least two classes.")
    if not torch.isfinite(counts).all() or bool((counts <= 0).any().item()):
        raise ValueError("class_counts must contain finite positive values.")
    if not 0.0 < float(smoothing):
        raise ValueError("smoothing must be positive.")
    frequencies = counts / counts.sum()
    return (frequencies + float(smoothing)).rsqrt().to(dtype=dtype)


def differentiable_margin_confusion(
    logits: Tensor,
    targets: Tensor,
    *,
    margin: float = 0.1,
) -> Tensor:
    """Equation (7) from CAR, with the true class excluded from soft argmax."""
    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError("logits must have shape [B, C] with C >= 2.")
    if targets.ndim != 1 or targets.numel() != logits.size(0):
        raise ValueError("targets must have shape [B].")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite.")
    num_classes = int(logits.size(1))
    targets = targets.to(device=logits.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= num_classes)).any().item()):
        raise ValueError("targets contain an out-of-range class index.")

    columns = []
    class_indices = torch.arange(num_classes, device=logits.device)
    for true_class in range(num_classes):
        selected = logits[targets == true_class]
        if selected.numel() == 0:
            columns.append(logits.new_zeros(num_classes))
            continue
        true_logits = selected[:, true_class : true_class + 1]
        soft_margin_gate = torch.sigmoid(float(margin) + selected - true_logits)
        competitor_logits = selected - true_logits
        competitor_logits = competitor_logits.masked_fill(
            class_indices.unsqueeze(0) == true_class,
            -torch.inf,
        )
        soft_competitor = torch.softmax(competitor_logits, dim=1)
        column = (soft_margin_gate * soft_competitor).mean(dim=0)
        column = column * (class_indices != true_class).to(dtype=column.dtype)
        columns.append(column)
    return torch.stack(columns, dim=1)


def confusion_aware_spectral_regularizer(
    logits: Tensor,
    targets: Tensor,
    *,
    class_counts: Sequence[int] | Tensor,
    previous_ema: Tensor | None = None,
    momentum: float = 0.5,
    smoothing: float = 0.2,
    margin: float = 0.1,
    bidirectional: bool = False,
) -> ConfusionSpectralResult:
    """Compute paper CAR or frequency-symmetric BiCAR without retaining EMA graphs."""
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("momentum must be in [0, 1).")
    batch_confusion = differentiable_margin_confusion(logits, targets, margin=margin)
    if previous_ema is None:
        previous = torch.zeros_like(batch_confusion)
    else:
        if tuple(previous_ema.shape) != tuple(batch_confusion.shape):
            raise ValueError("previous_ema shape must match the confusion matrix.")
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
    class_weights = confusion_frequency_weights(
        class_counts,
        smoothing=smoothing,
        device=logits.device,
        dtype=logits.dtype,
    )
    diagonal = torch.diag(class_weights)
    weighted_confusion = ema_confusion @ diagonal
    if bidirectional:
        weighted_confusion = diagonal @ weighted_confusion
    loss = torch.linalg.matrix_norm(weighted_confusion.float(), ord=2).to(
        dtype=logits.dtype
    )
    return ConfusionSpectralResult(
        loss=loss,
        batch_confusion=batch_confusion,
        ema_confusion=ema_confusion.detach(),
        class_weights=class_weights,
        weighted_confusion=weighted_confusion,
    )

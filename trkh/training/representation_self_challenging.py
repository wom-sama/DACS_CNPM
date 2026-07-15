from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProtectedRSCResult:
    gradient: Tensor
    preliminary_mask: Tensor
    final_mask: Tensor
    preliminary_logits: Tensor
    confidence_drop: Tensor
    eligible: Tensor
    positive_drop: Tensor
    selected: Tensor
    drop_count: int
    selection_limit: int


def true_logit_gradient(
    *,
    logits: Tensor,
    targets: Tensor,
    representation: Tensor,
    retain_graph: bool = True,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if representation.ndim != 2:
        raise ValueError("representation must have shape [batch, channels]")
    if targets.ndim != 1 or int(targets.numel()) != int(logits.size(0)):
        raise ValueError("targets must have shape [batch]")
    if int(representation.size(0)) != int(logits.size(0)):
        raise ValueError("representation and logits batch sizes differ")
    if not representation.requires_grad:
        raise ValueError("representation must require gradients")
    if bool((targets < 0).any().item()) or bool((targets >= logits.size(1)).any().item()):
        raise ValueError("targets contain a class outside logits")

    target_score = logits.gather(1, targets.long().unsqueeze(1)).sum()
    gradient = torch.autograd.grad(
        target_score,
        representation,
        retain_graph=bool(retain_graph),
        create_graph=False,
        only_inputs=True,
    )[0]
    if gradient is None or tuple(gradient.shape) != tuple(representation.shape):
        raise RuntimeError("true-logit gradient is missing or has an invalid shape")
    return gradient.detach()


def signed_top_gradient_mask(
    gradient: Tensor,
    *,
    drop_fraction: float,
    protected_rows: Tensor | None = None,
) -> tuple[Tensor, int]:
    if gradient.ndim != 2 or int(gradient.size(1)) <= 0:
        raise ValueError("gradient must have shape [batch, channels]")
    fraction = float(drop_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("drop_fraction must be in (0,1)")
    if not bool(torch.isfinite(gradient).all().item()):
        raise ValueError("gradient must be finite")

    drop_count = min(
        int(gradient.size(1)) - 1,
        max(1, int(math.ceil(int(gradient.size(1)) * fraction))),
    )
    indices = gradient.float().topk(
        k=drop_count,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    mask = torch.ones_like(gradient)
    mask.scatter_(1, indices, 0.0)

    if protected_rows is not None:
        if protected_rows.ndim != 1 or int(protected_rows.numel()) != int(gradient.size(0)):
            raise ValueError("protected_rows must have shape [batch]")
        mask[protected_rows.to(device=mask.device, dtype=torch.bool)] = 1.0
    return mask.detach(), drop_count


def select_positive_drop_rows(
    *,
    clean_logits: Tensor,
    challenged_logits: Tensor,
    targets: Tensor,
    eligible_classes: Sequence[int],
    batch_fraction: float,
    minimum_drop: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    if clean_logits.ndim != 2 or challenged_logits.shape != clean_logits.shape:
        raise ValueError("clean/challenged logits must have equal [batch, classes] shapes")
    if targets.ndim != 1 or int(targets.numel()) != int(clean_logits.size(0)):
        raise ValueError("targets must have shape [batch]")
    fraction = float(batch_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("batch_fraction must be in (0,1]")
    if float(minimum_drop) < 0.0:
        raise ValueError("minimum_drop must be nonnegative")

    classes = sorted({int(value) for value in eligible_classes})
    if not classes:
        raise ValueError("eligible_classes must not be empty")
    if classes[0] < 0 or classes[-1] >= int(clean_logits.size(1)):
        raise ValueError("eligible_classes contain a class outside logits")

    clean_probability = clean_logits.float().softmax(dim=1)
    challenged_probability = challenged_logits.float().softmax(dim=1)
    target_column = targets.long().unsqueeze(1)
    confidence_drop = (
        clean_probability.gather(1, target_column).squeeze(1)
        - challenged_probability.gather(1, target_column).squeeze(1)
        - float(minimum_drop)
    )
    eligible = torch.zeros_like(targets, dtype=torch.bool)
    for class_index in classes:
        eligible |= targets.eq(int(class_index))
    positive_drop = eligible & confidence_drop.gt(0.0)
    eligible_count = int(eligible.sum().item())
    selection_limit = int(math.ceil(eligible_count * fraction)) if eligible_count else 0

    selected = torch.zeros_like(eligible)
    positive_indices = torch.nonzero(positive_drop, as_tuple=False).flatten()
    if selection_limit > 0 and int(positive_indices.numel()) > 0:
        ranked_local = torch.argsort(
            confidence_drop.index_select(0, positive_indices),
            descending=True,
            stable=True,
        )
        selected_indices = positive_indices.index_select(
            0,
            ranked_local[:selection_limit],
        )
        selected[selected_indices] = True
    return (
        selected.detach(),
        confidence_drop.detach(),
        eligible.detach(),
        positive_drop.detach(),
        selection_limit,
    )


def locate_protected_rsc(
    *,
    representation: Tensor,
    clean_logits: Tensor,
    targets: Tensor,
    forward_from_representation: Callable[[Tensor], Tensor],
    focus_class: int = 1,
    eligible_classes: Sequence[int] = (0, 2, 3, 4),
    drop_fraction: float = 1.0 / 3.0,
    batch_fraction: float = 1.0 / 3.0,
    minimum_drop: float = 1e-4,
) -> ProtectedRSCResult:
    if int(focus_class) in {int(value) for value in eligible_classes}:
        raise ValueError("focus_class cannot be eligible for RSC")
    protected = targets.eq(int(focus_class))
    gradient = true_logit_gradient(
        logits=clean_logits,
        targets=targets,
        representation=representation,
        retain_graph=True,
    )
    preliminary_mask, drop_count = signed_top_gradient_mask(
        gradient,
        drop_fraction=drop_fraction,
        protected_rows=protected,
    )
    preliminary_logits = forward_from_representation(
        representation * preliminary_mask.to(dtype=representation.dtype)
    )
    selected, confidence_drop, eligible, positive_drop, selection_limit = (
        select_positive_drop_rows(
            clean_logits=clean_logits,
            challenged_logits=preliminary_logits,
            targets=targets,
            eligible_classes=eligible_classes,
            batch_fraction=batch_fraction,
            minimum_drop=minimum_drop,
        )
    )
    final_mask = torch.where(
        selected.unsqueeze(1),
        preliminary_mask,
        torch.ones_like(preliminary_mask),
    ).detach()
    if bool((final_mask[protected] != 1).any().item()):
        raise RuntimeError("class-1 protection invariant failed")
    return ProtectedRSCResult(
        gradient=gradient,
        preliminary_mask=preliminary_mask,
        final_mask=final_mask,
        preliminary_logits=preliminary_logits.detach(),
        confidence_drop=confidence_drop,
        eligible=eligible,
        positive_drop=positive_drop,
        selected=selected,
        drop_count=int(drop_count),
        selection_limit=int(selection_limit),
    )

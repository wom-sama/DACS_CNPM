from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ParetoMarginTarget:
    logits: Tensor
    true_class_shift: Tensor
    control_true_margin: Tensor
    robust_true_margin: Tensor
    projected_true_margin: Tensor
    control_focus_margin: Tensor
    robust_focus_margin: Tensor
    projected_focus_margin: Tensor


@dataclass(frozen=True)
class DualTeacherLoss:
    total: Tensor
    clean_logits: Tensor
    clean_focus_margin: Tensor
    robust_logits: Tensor


def centered_logits(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or int(logits.size(1)) < 2:
        raise ValueError("logits must have shape [B, C] with C >= 2")
    return logits - logits.mean(dim=1, keepdim=True)


def true_class_margin(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2 or labels.ndim != 1 or int(logits.size(0)) != int(labels.size(0)):
        raise ValueError("logits/labels are not batch aligned")
    labels = labels.to(device=logits.device, dtype=torch.long)
    if bool(((labels < 0) | (labels >= int(logits.size(1)))).any()):
        raise ValueError("labels contain an out-of-range class index")
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    rival_logits = logits.clone()
    rival_logits.scatter_(1, labels[:, None], torch.finfo(logits.dtype).min)
    return true_logits - rival_logits.max(dim=1).values


def focus_boundary_margin(
    logits: Tensor,
    labels: Tensor,
    *,
    focus_class: int = 1,
    focus_rivals: Sequence[int] = (0, 2, 4),
) -> Tensor:
    if logits.ndim != 2 or labels.ndim != 1 or int(logits.size(0)) != int(labels.size(0)):
        raise ValueError("logits/labels are not batch aligned")
    class_count = int(logits.size(1))
    focus = int(focus_class)
    rivals = tuple(int(value) for value in focus_rivals)
    if not 0 <= focus < class_count:
        raise ValueError("focus_class is out of range")
    if not rivals or focus in rivals or any(value < 0 or value >= class_count for value in rivals):
        raise ValueError("focus_rivals must be valid non-focus classes")
    labels = labels.to(device=logits.device, dtype=torch.long)
    general = true_class_margin(logits, labels)
    result = general.clone()

    focus_rows = labels == focus
    if bool(focus_rows.any()):
        rival_index = torch.tensor(rivals, device=logits.device, dtype=torch.long)
        result[focus_rows] = (
            logits[focus_rows, focus]
            - logits[focus_rows].index_select(1, rival_index).max(dim=1).values
        )

    rival_rows = torch.zeros_like(labels, dtype=torch.bool)
    for rival in rivals:
        rival_rows |= labels == rival
    if bool(rival_rows.any()):
        row_labels = labels[rival_rows]
        result[rival_rows] = (
            logits[rival_rows].gather(1, row_labels[:, None]).squeeze(1)
            - logits[rival_rows, focus]
        )
    return result


def pareto_project_robust_teacher(
    control_logits: Tensor,
    robust_logits: Tensor,
    labels: Tensor,
    *,
    focus_class: int = 1,
    focus_rivals: Sequence[int] = (0, 2, 4),
) -> ParetoMarginTarget:
    if control_logits.shape != robust_logits.shape:
        raise ValueError("control and robust logits must have identical shapes")
    labels = labels.to(device=robust_logits.device, dtype=torch.long)
    control_true = true_class_margin(control_logits, labels)
    robust_true = true_class_margin(robust_logits, labels)
    control_focus = focus_boundary_margin(
        control_logits,
        labels,
        focus_class=focus_class,
        focus_rivals=focus_rivals,
    )
    robust_focus = focus_boundary_margin(
        robust_logits,
        labels,
        focus_class=focus_class,
        focus_rivals=focus_rivals,
    )
    required_shift = torch.maximum(
        (control_true - robust_true).clamp_min(0.0),
        (control_focus - robust_focus).clamp_min(0.0),
    )
    projected = robust_logits.clone()
    projected.scatter_add_(1, labels[:, None], required_shift[:, None])
    projected_true = true_class_margin(projected, labels)
    projected_focus = focus_boundary_margin(
        projected,
        labels,
        focus_class=focus_class,
        focus_rivals=focus_rivals,
    )
    return ParetoMarginTarget(
        logits=projected,
        true_class_shift=required_shift,
        control_true_margin=control_true,
        robust_true_margin=robust_true,
        projected_true_margin=projected_true,
        control_focus_margin=control_focus,
        robust_focus_margin=robust_focus,
        projected_focus_margin=projected_focus,
    )


def clean_anchored_dual_teacher_loss(
    student_clean_logits: Tensor,
    student_robust_logits: Tensor,
    control_clean_logits: Tensor,
    projected_robust_logits: Tensor,
    labels: Tensor,
    *,
    focus_class: int = 1,
    focus_rivals: Sequence[int] = (0, 2, 4),
) -> DualTeacherLoss:
    shapes = {
        tuple(student_clean_logits.shape),
        tuple(student_robust_logits.shape),
        tuple(control_clean_logits.shape),
        tuple(projected_robust_logits.shape),
    }
    if len(shapes) != 1:
        raise ValueError("all logit tensors must have one shared shape")
    clean_logits_loss = F.smooth_l1_loss(
        centered_logits(student_clean_logits),
        centered_logits(control_clean_logits),
    )
    clean_focus_loss = F.smooth_l1_loss(
        focus_boundary_margin(
            student_clean_logits,
            labels,
            focus_class=focus_class,
            focus_rivals=focus_rivals,
        ),
        focus_boundary_margin(
            control_clean_logits,
            labels,
            focus_class=focus_class,
            focus_rivals=focus_rivals,
        ),
    )
    robust_logits_loss = F.smooth_l1_loss(
        centered_logits(student_robust_logits),
        centered_logits(projected_robust_logits),
    )
    return DualTeacherLoss(
        total=clean_logits_loss + clean_focus_loss + robust_logits_loss,
        clean_logits=clean_logits_loss,
        clean_focus_margin=clean_focus_loss,
        robust_logits=robust_logits_loss,
    )

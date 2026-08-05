from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


NUM_CLASSES = 5
CLASS1 = 1
GAIN_CAP = 2.0
CLASS1_WEIGHT = 4.0
TASK_WEIGHT = 1.0
DISTILL_WEIGHT = 0.5
RETENTION_WEIGHT = 0.25


def orchestrated_margin_targets(
    primary_scores: Tensor,
    teacher_scores: Tensor,
    labels: Tensor,
    *,
    gain_cap: float = GAIN_CAP,
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep only teacher corrections that improve a true-vs-rival margin."""
    if (
        primary_scores.ndim != 2
        or teacher_scores.shape != primary_scores.shape
        or primary_scores.shape[1] != NUM_CLASSES
        or labels.shape != primary_scores.shape[:1]
        or not 0.0 < float(gain_cap) <= 6.0
    ):
        raise ValueError("B30 expects scores [B,5], labels [B], and a valid cap")
    rows = torch.arange(labels.numel(), device=labels.device)
    rival_mask = torch.ones_like(primary_scores, dtype=torch.bool)
    rival_mask[rows, labels] = False
    primary_margin = primary_scores[rows, labels, None] - primary_scores
    teacher_margin = teacher_scores.detach()[rows, labels, None] - teacher_scores.detach()
    useful_gain = (teacher_margin - primary_margin.detach()).clamp(0.0, float(gain_cap))
    return primary_margin.detach() + useful_gain, rival_mask, useful_gain


def trust_margin_distillation_loss(
    logits: Tensor,
    primary_scores: Tensor,
    teacher_scores: Tensor,
    labels: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_margin, rival_mask, useful_gain = orchestrated_margin_targets(
        primary_scores, teacher_scores, labels
    )
    rows = torch.arange(labels.numel(), device=labels.device)
    student_margin = logits[rows, labels, None] - logits
    sample_weight = torch.where(
        labels.eq(CLASS1),
        torch.full_like(labels, CLASS1_WEIGHT, dtype=logits.dtype),
        torch.ones_like(labels, dtype=logits.dtype),
    )
    pair_weight = sample_weight[:, None].expand_as(logits)[rival_mask]
    pair_error = F.smooth_l1_loss(
        student_margin[rival_mask], target_margin[rival_mask], reduction="none"
    )
    distill = (pair_error * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)

    primary_other = primary_scores.detach().masked_fill(~rival_mask, -torch.inf)
    student_other = logits.masked_fill(~rival_mask, -torch.inf)
    primary_true_margin = primary_scores.detach()[rows, labels] - primary_other.amax(1)
    student_true_margin = logits[rows, labels] - student_other.amax(1)
    protected = primary_scores.detach().argmax(1).eq(labels)
    protected_weight = sample_weight[protected]
    retention_error = F.relu(
        primary_true_margin[protected] - student_true_margin[protected]
    )
    retention = (
        (retention_error * protected_weight).sum()
        / protected_weight.sum().clamp_min(1.0)
        if bool(protected.any())
        else logits.sum() * 0.0
    )
    task = F.cross_entropy(logits, labels)
    total = (
        TASK_WEIGHT * task
        + DISTILL_WEIGHT * distill
        + RETENTION_WEIGHT * retention
    )
    rival_gain = useful_gain[rival_mask]
    class1_rivals = rival_mask & labels.eq(CLASS1)[:, None]
    return total, {
        "task": task.detach(),
        "kd": distill.detach(),
        "retention": retention.detach(),
        "total": total.detach(),
        "positive_gain_fraction": rival_gain.gt(0).float().mean().detach(),
        "gain_mean": rival_gain.mean().detach(),
        "class1_positive_gain_fraction": (
            useful_gain[class1_rivals].gt(0).float().mean().detach()
            if bool(class1_rivals.any())
            else logits.detach().sum() * 0.0
        ),
    }

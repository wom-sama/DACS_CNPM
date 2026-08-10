from __future__ import annotations

import torch

from trkh.training.pareto_margin_distillation import (
    centered_logits,
    clean_anchored_dual_teacher_loss,
    focus_boundary_margin,
    pareto_project_robust_teacher,
    true_class_margin,
)


def test_pareto_projection_preserves_non_target_geometry_and_both_margins() -> None:
    control = torch.tensor(
        [
            [3.0, 2.4, 0.2, -1.0, 0.1],
            [1.0, 2.5, 2.0, -0.5, 0.0],
            [0.0, 1.2, 2.4, -0.4, 0.3],
        ]
    )
    robust = torch.tensor(
        [
            [2.2, 2.3, 0.4, -0.8, 0.0],
            [1.5, 2.1, 2.0, -0.2, 0.4],
            [0.3, 1.8, 1.9, -0.1, 0.5],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    target = pareto_project_robust_teacher(control, robust, labels)

    assert torch.all(target.projected_true_margin >= target.control_true_margin - 1e-6)
    assert torch.all(target.projected_true_margin >= target.robust_true_margin - 1e-6)
    assert torch.all(target.projected_focus_margin >= target.control_focus_margin - 1e-6)
    assert torch.all(target.projected_focus_margin >= target.robust_focus_margin - 1e-6)
    for row, label in enumerate(labels.tolist()):
        non_targets = [index for index in range(5) if index != label]
        before = robust[row, non_targets][:, None] - robust[row, non_targets][None, :]
        after = target.logits[row, non_targets][:, None] - target.logits[row, non_targets][None, :]
        torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_projection_is_exact_when_robust_teacher_already_pareto_dominates() -> None:
    control = torch.tensor([[1.0, 2.0, 0.5, 0.0, -0.5]])
    robust = torch.tensor([[0.5, 3.0, 0.2, 0.1, -0.2]])
    labels = torch.tensor([1])
    target = pareto_project_robust_teacher(control, robust, labels)
    torch.testing.assert_close(target.logits, robust, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target.true_class_shift, torch.zeros(1), rtol=0.0, atol=0.0)


def test_focus_margin_uses_class1_boundary_for_relevant_labels() -> None:
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0, 9.0, 0.0],
            [3.0, 2.0, 9.0, 8.0, 0.0],
            [0.0, 1.0, 2.0, 5.0, 3.0],
        ]
    )
    labels = torch.tensor([1, 0, 3])
    observed = focus_boundary_margin(logits, labels)
    expected = torch.tensor([1.0, 1.0, 2.0])
    torch.testing.assert_close(observed, expected)
    assert true_class_margin(logits, labels)[0].item() == -6.0


def test_dual_teacher_loss_is_zero_at_both_targets_and_backpropagates() -> None:
    clean_teacher = torch.randn(4, 5)
    robust_teacher = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 4])
    clean_student = clean_teacher.clone().requires_grad_(True)
    robust_student = robust_teacher.clone().requires_grad_(True)
    exact = clean_anchored_dual_teacher_loss(
        clean_student,
        robust_student,
        clean_teacher,
        robust_teacher,
        labels,
    )
    assert exact.total.item() == 0.0

    shifted_clean = (clean_teacher + torch.randn_like(clean_teacher) * 0.1).requires_grad_(True)
    shifted_robust = (robust_teacher + torch.randn_like(robust_teacher) * 0.1).requires_grad_(True)
    active = clean_anchored_dual_teacher_loss(
        shifted_clean,
        shifted_robust,
        clean_teacher,
        robust_teacher,
        labels,
    )
    active.total.backward()
    assert active.total.item() > 0.0
    assert shifted_clean.grad is not None and float(shifted_clean.grad.norm()) > 0.0
    assert shifted_robust.grad is not None and float(shifted_robust.grad.norm()) > 0.0
    torch.testing.assert_close(centered_logits(clean_teacher).sum(dim=1), torch.zeros(4), atol=1e-6, rtol=0.0)

from __future__ import annotations

import torch

from trkh.models.efficientvim_m1 import EfficientViMM1
from trkh.models.efficientvim_residual_student_b29 import (
    RESIDUAL_SCALE,
    EfficientViMResidualStudent,
    load_student_state_dict,
    student_state_dict,
)
from trkh.tools.run_b29_efficientvim_teacher_residual_fold import (
    MIN_LR,
    PEAK_LR,
    learning_rate,
    residual_student_loss,
)


def _model() -> EfficientViMResidualStudent:
    torch.manual_seed(7)
    return EfficientViMResidualStudent(EfficientViMM1(num_classes=5))


def test_zero_head_preserves_primary_scores_exactly() -> None:
    model = _model().eval()
    images = torch.randn(2, 3, 224, 224)
    primary = torch.randn(2, 5)
    with torch.inference_mode():
        logits, trace = model.forward_with_trace(images, primary)
    assert torch.equal(logits, primary.float())
    assert torch.count_nonzero(trace["residual_scores"]) == 0


def test_residual_is_bounded_and_gradients_reach_heads_then_backbone() -> None:
    model = _model().train()
    images = torch.randn(2, 3, 224, 224)
    primary = torch.randn(2, 5)
    first = model(images, primary).sum()
    first.backward()
    assert all(head.weight.grad is not None for head in model.student.heads)
    stem = model.student.patch_embed.conv[0].conv.weight
    assert stem.grad is not None
    assert torch.count_nonzero(stem.grad) == 0

    for head in model.student.heads:
        with torch.no_grad():
            head.weight.normal_(std=0.01)
    model.zero_grad(set_to_none=True)
    logits, trace = model.forward_with_trace(images, primary)
    logits.sum().backward()
    assert float(trace["residual_scores"].abs().max()) < RESIDUAL_SCALE
    assert torch.count_nonzero(stem.grad) > 0


def test_student_state_round_trip_is_exact() -> None:
    source = _model()
    target = _model()
    with torch.no_grad():
        source.student.weights.add_(0.25)
    state = student_state_dict(source)
    load_student_state_dict(target, state)
    for name, value in source.student.state_dict().items():
        assert torch.equal(value, target.student.state_dict()[name])


def test_learning_rate_hits_locked_warmup_and_final_endpoints() -> None:
    assert learning_rate(0, 60, 10) == PEAK_LR / 10
    assert learning_rate(9, 60, 10) == PEAK_LR
    assert learning_rate(59, 60, 10) == MIN_LR
    assert all(
        learning_rate(index, 60, 10) >= learning_rate(index + 1, 60, 10)
        for index in range(9, 59)
    )


def test_residual_student_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(4, 5, requires_grad=True)
    primary = torch.randn(4, 5)
    teacher = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    loss, parts = residual_student_loss(logits, primary, teacher, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert set(parts) == {"task", "kd", "retention", "total"}

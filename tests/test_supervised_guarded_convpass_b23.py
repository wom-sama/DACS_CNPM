from __future__ import annotations

import torch
from torch import nn

from trkh.models.supervised_guarded_convpass_b23 import (
    adapter_state_dict,
    class1_margin,
    configure_adapter_only,
    guarded_adapter_loss,
    is_adapter_parameter,
    load_adapter_state_dict,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.adapter_attn = nn.Linear(3, 3)
        self.adapter_mlp = nn.Linear(3, 3)
        self.head = nn.Linear(3, 5)


def test_configure_and_roundtrip_adapter_only() -> None:
    model = _ToyModel()
    counts = configure_adapter_only(model)
    assert counts["trainable"] == 24
    assert counts["frozen"] > counts["trainable"]
    assert all(
        parameter.requires_grad == is_adapter_parameter(name)
        for name, parameter in model.named_parameters()
    )
    state = adapter_state_dict(model)
    clone = _ToyModel()
    load_adapter_state_dict(clone, state)
    assert set(state) == {
        name for name in clone.state_dict() if is_adapter_parameter(name)
    }


def test_class1_margin_uses_locked_rivals_only() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0, 99.0, -1.0]])
    assert torch.equal(class1_margin(logits), torch.tensor([1.0]))


def test_guard_penalizes_only_lost_labelled_class1_margin() -> None:
    teacher = torch.tensor(
        [[0.0, 2.0, 1.0, -1.0, 0.0], [1.0, 0.0, 2.0, -1.0, 0.0]]
    )
    targets = torch.tensor([1, 2])
    student = teacher.clone().requires_grad_(True)
    student.data[0, 1] -= 0.50
    zero_task = lambda values, labels: values.sum() * 0.0
    loss, parts = guarded_adapter_loss(
        student,
        teacher,
        targets,
        zero_task,
        retention_weight=1.0,
        distillation_weight=0.0,
        retention_slack=0.05,
    )
    assert torch.allclose(parts["retention"], torch.tensor(0.45), atol=1e-6)
    assert int(parts["class1_rows"].item()) == 1
    loss.backward()
    assert student.grad is not None
    assert float(student.grad[0, 1]) < 0.0


def test_guard_allows_improved_margin_and_no_class1_batch() -> None:
    teacher = torch.tensor([[0.0, 2.0, 1.0, -1.0, 0.0]])
    improved = teacher.clone()
    improved[:, 1] += 0.25
    zero_task = lambda values, labels: values.sum() * 0.0
    _, improved_parts = guarded_adapter_loss(
        improved,
        teacher,
        torch.tensor([1]),
        zero_task,
        distillation_weight=0.0,
    )
    _, absent_parts = guarded_adapter_loss(
        teacher,
        teacher,
        torch.tensor([0]),
        zero_task,
        distillation_weight=0.0,
    )
    assert float(improved_parts["retention"]) == 0.0
    assert float(absent_parts["retention"]) == 0.0


def test_numerical_kl_roundoff_cannot_reduce_total_loss() -> None:
    logits = torch.tensor(
        [[-0.7599, 0.1994, -0.4695, 0.1094, -0.1592]], dtype=torch.float32
    )
    zero_task = lambda values, labels: values.sum() * 0.0
    total, parts = guarded_adapter_loss(
        logits,
        logits.clone(),
        torch.tensor([0]),
        zero_task,
        retention_weight=0.0,
        distillation_weight=1.0,
    )
    assert float(parts["distillation"]) >= 0.0
    assert float(total) >= 0.0

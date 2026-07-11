import torch
from torch import nn

from trkh.tools.pretrain_internal_barlow import (
    build_v8_model_config,
    dino_self_distillation_loss,
    update_ema_model,
)


def test_dino_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(17)
    student_a = torch.randn(4, 12, requires_grad=True)
    student_b = torch.randn(4, 12, requires_grad=True)
    teacher_a = torch.randn(4, 12)
    teacher_b = torch.randn(4, 12)
    center = torch.zeros(12)

    loss = dino_self_distillation_loss(
        student_a,
        student_b,
        teacher_a,
        teacher_b,
        center=center,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert student_a.grad is not None
    assert student_b.grad is not None
    assert torch.isfinite(student_a.grad).all()
    assert torch.isfinite(student_b.grad).all()


def test_ema_update_moves_target_toward_source() -> None:
    source = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(1.0)
        target.weight.zero_()
        target.bias.zero_()

    update_ema_model(source, target, momentum=0.5)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 1.0))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 0.5))


def test_ssl_v8_config_can_preserve_bbox_spatial_fusion_head() -> None:
    config = build_v8_model_config(
        256,
        bbox_spatial_fusion=True,
        bbox_spatial_fusion_hidden_dim=48,
        bbox_spatial_fusion_dropout=0.03,
        bbox_spatial_fusion_logit_scale=0.17,
    )

    assert config.bbox_spatial_fusion is True
    assert config.bbox_spatial_fusion_hidden_dim == 48
    assert config.bbox_spatial_fusion_dropout == 0.03
    assert config.bbox_spatial_fusion_logit_scale == 0.17

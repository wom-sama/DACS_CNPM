from __future__ import annotations

import pytest
import torch

from trkh.training.train import (
    _distillation_loss,
    _probability_distillation_loss,
    build_configs,
    parse_args,
)


@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0, 4.0])
def test_offline_probabilities_match_online_logits_at_same_temperature(
    temperature: float,
) -> None:
    torch.manual_seed(20260729)
    online_student = torch.randn(7, 5, requires_grad=True)
    offline_student = online_student.detach().clone().requires_grad_(True)
    teacher_logits = torch.randn(7, 5)
    teacher_probabilities_at_t1 = teacher_logits.softmax(dim=1)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 0])

    online_loss = _distillation_loss(
        student_logits=online_student,
        teacher_logits=teacher_logits,
        targets=targets,
        temperature=temperature,
        focus_class_index=1,
        focus_class_weight=1.4,
    )
    offline_loss = _probability_distillation_loss(
        student_logits=offline_student,
        teacher_probabilities=teacher_probabilities_at_t1,
        targets=targets,
        temperature=temperature,
        focus_class_index=1,
        focus_class_weight=1.4,
    )

    assert torch.allclose(offline_loss, online_loss, rtol=1e-6, atol=1e-7)
    online_loss.backward()
    offline_loss.backward()
    assert torch.allclose(
        offline_student.grad,
        online_student.grad,
        rtol=1e-5,
        atol=1e-7,
    )


def test_global_online_and_offline_teacher_sources_are_mutually_exclusive(
    tmp_path,
) -> None:
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_csv = tmp_path / "teacher.csv"
    teacher_checkpoint.write_bytes(b"placeholder")
    teacher_csv.write_text("path,prob_0\n", encoding="utf-8")
    args = parse_args(
        [
            "--pretrained-distillation",
            "--distillation-teacher-checkpoint",
            str(teacher_checkpoint),
            "--distillation-teacher-csv",
            str(teacher_csv),
        ]
    )
    with pytest.raises(ValueError, match="exactly one global distillation source"):
        build_configs(args)

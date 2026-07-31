from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from trkh.training.train import DistillationTeacherInputAdapter


class _RecordingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_images = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_images = images.detach().clone()
        return images.mean(dim=(2, 3))


def test_distillation_teacher_adapter_resizes_and_renormalizes() -> None:
    teacher = _RecordingTeacher()
    adapter = DistillationTeacherInputAdapter(
        teacher,
        student_mean=(0.5, 0.4, 0.3),
        student_std=(0.2, 0.3, 0.4),
        teacher_mean=(0.1, 0.2, 0.3),
        teacher_std=(0.5, 0.25, 0.2),
        teacher_image_size=12,
    )
    torch.manual_seed(20260729)
    student_images = torch.randn(2, 3, 8, 8)

    outputs = adapter(student_images)

    student_mean = torch.tensor((0.5, 0.4, 0.3)).view(1, 3, 1, 1)
    student_std = torch.tensor((0.2, 0.3, 0.4)).view(1, 3, 1, 1)
    teacher_mean = torch.tensor((0.1, 0.2, 0.3)).view(1, 3, 1, 1)
    teacher_std = torch.tensor((0.5, 0.25, 0.2)).view(1, 3, 1, 1)
    rgb = student_images * student_std + student_mean
    expected = F.interpolate(
        rgb,
        size=(12, 12),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    expected = (expected - teacher_mean) / teacher_std

    assert teacher.last_images is not None
    torch.testing.assert_close(teacher.last_images, expected)
    torch.testing.assert_close(outputs, expected.mean(dim=(2, 3)))


def test_distillation_teacher_adapter_rejects_non_rgb_inputs() -> None:
    adapter = DistillationTeacherInputAdapter(
        _RecordingTeacher(),
        student_mean=(0.0, 0.0, 0.0),
        student_std=(1.0, 1.0, 1.0),
        teacher_mean=(0.0, 0.0, 0.0),
        teacher_std=(1.0, 1.0, 1.0),
        teacher_image_size=8,
    )
    try:
        adapter(torch.randn(1, 1, 8, 8))
    except ValueError as exc:
        assert "[B,3,H,W]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected non-RGB input rejection.")

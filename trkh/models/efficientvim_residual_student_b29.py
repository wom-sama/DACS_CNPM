from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from trkh.models.efficientvim_m1 import EfficientViMM1


NUM_CLASSES = 5
RESIDUAL_SCALE = 6.0
M1_PARAMETERS = 5_720_278


class EfficientViMResidualStudent(nn.Module):
    """Bounded mobile residual over an externally supplied primary score path."""

    def __init__(
        self,
        student: EfficientViMM1,
        *,
        residual_scale: float = RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if (
            not isinstance(student, EfficientViMM1)
            or int(student.num_classes) != NUM_CLASSES
            or sum(parameter.numel() for parameter in student.parameters())
            != M1_PARAMETERS
            or not 0.0 < float(residual_scale) <= 10.0
        ):
            raise ValueError("B29 EfficientViM student topology changed")
        self.student = student
        self.residual_scale = float(residual_scale)
        for head in self.student.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.model_type = "b29_efficientvim_residual_student"
        self.research_track = "pretrained"

    def forward_with_trace(
        self, images: Tensor, primary_scores: Tensor
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        if primary_scores.ndim != 2 or tuple(primary_scores.shape[1:]) != (
            NUM_CLASSES,
        ):
            raise ValueError("B29 primary scores must be [B,5]")
        raw = self.student(images).float()
        residual = self.residual_scale * torch.tanh(raw / self.residual_scale)
        logits = primary_scores.detach().float() + residual
        return logits, {
            "primary_scores": primary_scores.detach().float(),
            "student_raw": raw,
            "residual_scores": residual,
        }

    def forward(self, images: Tensor, primary_scores: Tensor) -> Tensor:
        return self.forward_with_trace(images, primary_scores)[0]


def student_state_dict(model: EfficientViMResidualStudent) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.student.state_dict().items()
    }


def load_student_state_dict(
    model: EfficientViMResidualStudent, state: Mapping[str, Tensor]
) -> None:
    model.student.load_state_dict(dict(state), strict=True)

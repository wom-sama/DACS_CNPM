from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


BIAS_ONLY = "bias_only"
LOGITS_ONLY = "logits_only"
FEATURE_LOGITS = "feature_logits"
SUPPORTED_EXPERT_MODES = (BIAS_ONLY, LOGITS_ONLY, FEATURE_LOGITS)


class RecallMonotonicLogitExpert(nn.Module):
    """A bounded expert that can only increase one class logit.

    The base model remains frozen.  Because the correction is non-negative and
    every other logit is byte-preserved, a sample already assigned to the focus
    class can never leave it.  This is the hard recall-preservation property
    that signed residual specialists used earlier in TRKH did not provide.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        mode: str,
        focus_class: int = 1,
        max_correction: float = 4.0,
        initial_bias: float = -8.0,
    ) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_EXPERT_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_EXPERT_MODES}; got {mode!r}."
            )
        if int(feature_dim) <= 0 or int(num_classes) < 2:
            raise ValueError("feature_dim and num_classes must be positive.")
        if not 0 <= int(focus_class) < int(num_classes):
            raise ValueError("focus_class is outside the class range.")
        if not float(max_correction) > 0.0:
            raise ValueError("max_correction must be positive.")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.mode = resolved_mode
        self.focus_class = int(focus_class)
        self.max_correction = float(max_correction)

        if self.mode == BIAS_ONLY:
            self.raw_bias = nn.Parameter(torch.tensor(float(initial_bias)))
            self.linear = None
        else:
            input_dim = self.num_classes
            if self.mode == FEATURE_LOGITS:
                input_dim += self.feature_dim
            self.linear = nn.Linear(input_dim, 1)
            nn.init.zeros_(self.linear.weight)
            nn.init.constant_(self.linear.bias, float(initial_bias))
            self.register_parameter("raw_bias", None)

    def raw_score(self, features: Tensor, base_logits: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be [B,{self.feature_dim}], got {tuple(features.shape)}."
            )
        if base_logits.ndim != 2 or base_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"base_logits must be [B,{self.num_classes}], "
                f"got {tuple(base_logits.shape)}."
            )
        if features.shape[0] != base_logits.shape[0]:
            raise ValueError("features and base_logits batch sizes differ.")
        if self.mode == BIAS_ONLY:
            return self.raw_bias.expand(base_logits.shape[0])
        centered_logits = base_logits - base_logits.mean(dim=1, keepdim=True)
        inputs = centered_logits
        if self.mode == FEATURE_LOGITS:
            inputs = torch.cat((features, centered_logits), dim=1)
        return self.linear(inputs).squeeze(1)

    def forward(self, features: Tensor, base_logits: Tensor) -> Tuple[Tensor, Tensor]:
        raw = self.raw_score(features, base_logits)
        correction = torch.sigmoid(raw) * self.max_correction
        adjusted = base_logits.clone()
        adjusted[:, self.focus_class] = (
            adjusted[:, self.focus_class] + correction
        )
        return adjusted, correction

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class PatchStyleRecalibration(nn.Module):
    """SRM-style mean/std gating over patch hidden channels."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive.")
        if not 0.0 < float(momentum) <= 1.0:
            raise ValueError("momentum must be in (0, 1].")
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.cfc = nn.Parameter(torch.zeros(self.hidden_dim, 2))
        self.bn = nn.BatchNorm1d(
            self.hidden_dim,
            eps=self.eps,
            momentum=float(momentum),
            affine=True,
            track_running_stats=True,
        )
        self._last_trace: Dict[str, Tensor] = {}

    def style_statistics(self, patch_hidden: Tensor) -> Tensor:
        if patch_hidden.ndim != 3 or int(patch_hidden.size(-1)) != self.hidden_dim:
            raise ValueError(
                "patch_hidden must have shape [batch, patch_tokens, hidden_dim]."
            )
        if int(patch_hidden.size(1)) <= 0:
            raise ValueError("Patch-style recalibration requires at least one patch token.")
        channel_mean = patch_hidden.mean(dim=1)
        channel_variance = (patch_hidden - channel_mean.unsqueeze(1)).square().mean(dim=1)
        channel_std = (channel_variance + self.eps).sqrt()
        return torch.stack((channel_mean, channel_std), dim=-1)

    def gate_from_statistics(self, style: Tensor) -> Tensor:
        if style.ndim != 3 or tuple(style.shape[-2:]) != (self.hidden_dim, 2):
            raise ValueError("style must have shape [batch, hidden_dim, 2].")
        encoded = (style * self.cfc.unsqueeze(0)).sum(dim=-1)
        return torch.sigmoid(self.bn(encoded))

    def forward(
        self,
        hidden: Tensor,
        *,
        prefix_count: int,
        return_gate: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if hidden.ndim != 3 or int(hidden.size(-1)) != self.hidden_dim:
            raise ValueError("hidden must have shape [batch, tokens, hidden_dim].")
        resolved_prefix_count = int(prefix_count)
        if not 0 <= resolved_prefix_count < int(hidden.size(1)):
            raise ValueError("prefix_count must leave at least one patch token.")

        patch_hidden = hidden[:, resolved_prefix_count:]
        style = self.style_statistics(patch_hidden)
        gate = self.gate_from_statistics(style)
        recalibrated_patches = patch_hidden * gate.unsqueeze(1)
        if resolved_prefix_count:
            output = torch.cat(
                (hidden[:, :resolved_prefix_count], recalibrated_patches),
                dim=1,
            )
        else:
            output = recalibrated_patches

        with torch.no_grad():
            gate_float = gate.detach().float()
            style_float = style.detach().float()
            denominator = patch_hidden.detach().float().norm().clamp_min(1e-12)
            self._last_trace = {
                "gate_values": gate_float,
                "gate_min": gate_float.amin(),
                "gate_mean": gate_float.mean(),
                "gate_max": gate_float.amax(),
                "gate_channel_std": gate_float.std(dim=1, correction=0).mean(),
                "gate_sample_std": gate_float.std(dim=0, correction=0).mean(),
                "style_mean_abs": style_float[..., 0].abs().mean(),
                "style_std_mean": style_float[..., 1].mean(),
                "cfc_l2_norm": self.cfc.detach().float().norm(),
                "patch_hidden_norm_ratio": (
                    recalibrated_patches.detach().float().norm() / denominator
                ),
                "patch_count": torch.tensor(
                    int(patch_hidden.size(1)),
                    device=hidden.device,
                    dtype=torch.long,
                ),
            }
        if return_gate:
            return output, gate
        return output

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

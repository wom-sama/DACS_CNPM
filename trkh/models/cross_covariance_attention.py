from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SharedProjectionCrossCovarianceAttention(nn.Module):
    """Patch-only XCA that reuses a parent block's qkv/projection layers."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("dim must be positive.")
        if int(num_heads) <= 0 or int(dim) % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self._last_trace: Dict[str, Tensor] = {}

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        qkv_projection: nn.Linear,
        output_projection: nn.Linear,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if patch_tokens.ndim != 3 or int(patch_tokens.size(-1)) != self.dim:
            raise ValueError(
                "patch_tokens must have shape [batch, tokens, dim] with the configured dim."
            )
        batch_size, token_count, _ = patch_tokens.shape
        if int(token_count) <= 0:
            raise ValueError("Cross-covariance attention requires at least one patch token.")

        normalized = self.norm(patch_tokens)
        qkv = qkv_projection(normalized)
        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 4, 1)
        query, key, value = qkv.unbind(dim=0)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = attention @ value
        output = output.permute(0, 3, 1, 2).reshape(
            batch_size,
            token_count,
            self.dim,
        )
        output = output_projection(output)

        with torch.no_grad():
            probability = attention.detach().float().clamp_min(1e-12)
            normalized_entropy = -(
                probability * probability.log()
            ).sum(dim=-1).mean() / math.log(float(self.head_dim))
            self._last_trace = {
                "temperature_min": self.temperature.detach().float().amin(),
                "temperature_mean": self.temperature.detach().float().mean(),
                "temperature_max": self.temperature.detach().float().amax(),
                "normalized_entropy": normalized_entropy,
                "diagonal_mass": torch.diagonal(
                    probability,
                    dim1=-2,
                    dim2=-1,
                ).mean(),
                "query_norm_max_error": (
                    query.detach().float().norm(dim=-1) - 1.0
                ).abs().amax(),
                "key_norm_max_error": (
                    key.detach().float().norm(dim=-1) - 1.0
                ).abs().amax(),
                "patch_count": torch.tensor(
                    int(token_count),
                    device=patch_tokens.device,
                    dtype=torch.long,
                ),
            }
        if return_attention:
            return output, attention
        return output

    def record_residual_norm_ratio(self, ratio: Tensor) -> None:
        self._last_trace["residual_norm_ratio"] = ratio.detach().float()

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

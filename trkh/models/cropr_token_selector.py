from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _trunc_normal_tf_(tensor: Tensor, *, std: float) -> Tensor:
    """TF-style truncated normal used by the official Cropr query init."""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        tensor.mul_(float(std))
    return tensor


class CroprTokenSelector(nn.Module):
    """Task-supervised spatial-token scorer adapted from Token Cropr.

    The query/scoring and detached auxiliary path follow the official MIT
    implementation by Bergner et al. (CVPR 2025), commit ``fa259e9``. TRKH
    passes only decision-eligible spatial tokens because its seven prefix
    tokens are retained unconditionally.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_classes: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.mlp_ratio = float(mlp_ratio)
        if self.dim < 1:
            raise ValueError("Cropr selector dimension must be positive.")
        if self.num_classes < 2:
            raise ValueError("Cropr selector requires at least two classes.")
        if self.mlp_ratio <= 0.0:
            raise ValueError("Cropr selector MLP ratio must be positive.")

        hidden_dim = max(1, int(self.dim * self.mlp_ratio))
        self.scale = self.dim**-0.5
        self.query = nn.Parameter(torch.empty(1, 1, self.dim))
        self.mlp_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
        )
        self.head_norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        query_std = self.dim**-0.5
        _trunc_normal_tf_(self.query, std=query_std)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_scorer(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("Cropr selector expects patch tokens [B,N,D].")
        batch_size, token_count, dim = patch_tokens.shape
        if token_count < 1 or int(dim) != self.dim:
            raise ValueError(
                "Cropr selector received an empty or dimension-mismatched token set."
            )

        query = self.query.expand(batch_size, -1, -1)
        query = query.reshape(batch_size, 1, 1, self.dim).transpose(1, 2)
        keys = patch_tokens.reshape(batch_size, token_count, 1, self.dim).transpose(1, 2)
        attention_logits = query @ keys.transpose(-2, -1)
        return attention_logits.sum(dim=(1, 2))

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        collect_auxiliary: bool = False,
        return_trace: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]:
        scorer_input = patch_tokens.detach() if collect_auxiliary else patch_tokens
        scores = self.forward_scorer(scorer_input)
        if not collect_auxiliary:
            trace = {"raw_scores": scores.detach().float()} if return_trace else {}
            return scores, None, trace

        batch_size, token_count, _ = scorer_input.shape
        values = scorer_input.reshape(
            batch_size,
            token_count,
            1,
            self.dim,
        ).transpose(1, 2)
        attention = (scores * self.scale).softmax(dim=-1)
        aggregated = attention.unsqueeze(1).unsqueeze(2) @ values
        aggregated = aggregated.transpose(1, 2).reshape(batch_size, 1, self.dim).squeeze(1)
        aggregated = aggregated + self.mlp(self.mlp_norm(aggregated))
        auxiliary_logits = self.head(self.head_norm(aggregated))

        normalized_entropy = -(
            attention.float()
            * attention.float().clamp(min=1e-12).log()
        ).sum(dim=1) / max(math.log(float(token_count)), 1e-12)
        trace = (
            {
                "raw_scores": scores.detach().float(),
                "attention": attention.detach().float(),
                "aggregated": aggregated.detach().float(),
                "score_variance": scores.detach().float().var(
                    dim=1, unbiased=False
                ),
                "normalized_entropy": normalized_entropy.detach(),
                "auxiliary_logits": auxiliary_logits.detach().float(),
            }
            if return_trace
            else {}
        )
        return scores, auxiliary_logits, trace

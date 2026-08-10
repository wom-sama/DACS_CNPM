from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def visual_contrast_lambda_init(block_depth: int) -> float:
    depth = int(block_depth)
    if depth < 0:
        raise ValueError("block_depth must be nonnegative.")
    return 0.8 - 0.6 * math.exp(-0.3 * float(depth))


class VisualContrastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("RMSNorm dimension must be positive.")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=inputs.dtype) * self.weight.to(dtype=inputs.dtype)


class VisualContrastAttention(nn.Module):
    """Prefix-aware Visual-Contrast Attention for a dense 2-D patch grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        visual_contrast_tokens: int = 64,
        block_depth: int = 0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        contrast_tokens = int(visual_contrast_tokens)
        contrast_side = math.isqrt(contrast_tokens)
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("VCA dimension must be positive and divisible by num_heads.")
        if contrast_tokens <= 0 or contrast_side * contrast_side != contrast_tokens:
            raise ValueError("visual_contrast_tokens must be a positive perfect square.")
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
        if not 0.0 <= float(projection_dropout) < 1.0:
            raise ValueError("projection_dropout must be in [0, 1).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.visual_contrast_tokens = contrast_tokens
        self.visual_contrast_side = contrast_side
        self.block_depth = int(block_depth)
        self.lambda_init = visual_contrast_lambda_init(self.block_depth)

        # Preserve the existing TRKH attention key schema for provenance and
        # checkpoint tooling while changing only the interaction mechanism.
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.projection_dropout = nn.Dropout(float(projection_dropout))

        self.positive_embedding = nn.Parameter(
            torch.empty(1, contrast_tokens, dim)
        )
        self.negative_embedding = nn.Parameter(
            torch.empty(1, contrast_tokens, dim)
        )
        self.stage1_lambda_q1 = nn.Parameter(torch.empty(self.head_dim))
        self.stage1_lambda_k1 = nn.Parameter(torch.empty(self.head_dim))
        self.stage1_lambda_q2 = nn.Parameter(torch.empty(self.head_dim))
        self.stage1_lambda_k2 = nn.Parameter(torch.empty(self.head_dim))
        self.stage2_lambda_q1 = nn.Parameter(torch.empty(self.head_dim))
        self.stage2_lambda_k1 = nn.Parameter(torch.empty(self.head_dim))
        self.stage2_lambda_q2 = nn.Parameter(torch.empty(self.head_dim))
        self.stage2_lambda_k2 = nn.Parameter(torch.empty(self.head_dim))
        self.stage1_norm = VisualContrastRMSNorm(self.head_dim)
        self.stage2_norm = VisualContrastRMSNorm(self.head_dim)
        self.depthwise_value = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )
        self._last_trace: Dict[str, Tensor] = {}
        self.reset_visual_contrast_parameters()

    def reset_visual_contrast_parameters(self) -> None:
        # These initializers match the official LinearDiff implementation.
        nn.init.normal_(self.positive_embedding)
        nn.init.normal_(self.negative_embedding)
        for parameter in (
            self.stage1_lambda_q1,
            self.stage1_lambda_k1,
            self.stage1_lambda_q2,
            self.stage1_lambda_k2,
            self.stage2_lambda_q1,
            self.stage2_lambda_k1,
            self.stage2_lambda_q2,
            self.stage2_lambda_k2,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.1)

    @staticmethod
    def _validate_dense_grid(
        inputs: Tensor,
        *,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor],
    ) -> Tuple[int, int, int]:
        if inputs.ndim != 3:
            raise ValueError("VCA expects tokens with shape [B, N, C].")
        if grid_size is None or len(grid_size) != 2:
            raise ValueError("VCA requires an explicit dense patch grid_size.")
        grid_height, grid_width = (int(grid_size[0]), int(grid_size[1]))
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("VCA grid dimensions must be positive.")
        prefix_count = int(prefix_count)
        if prefix_count < 0 or prefix_count > int(inputs.size(1)):
            raise ValueError("VCA prefix_count is out of range.")
        patch_count = int(inputs.size(1)) - prefix_count
        grid_patch_count = grid_height * grid_width
        if patch_count != grid_patch_count:
            raise ValueError(
                "VCA requires the complete dense patch grid; token pruning is unsupported."
            )
        if torch.is_tensor(patch_indices):
            if tuple(patch_indices.shape) != (int(inputs.size(0)), patch_count):
                raise ValueError("VCA patch_indices do not match the dense patch grid.")
            expected = torch.arange(
                patch_count,
                device=patch_indices.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(inputs.size(0)), -1)
            if not torch.equal(patch_indices.to(dtype=torch.long), expected):
                raise ValueError("VCA requires identity-ordered dense patch_indices.")
        return grid_height, grid_width, prefix_count

    def _lambda_value(
        self,
        q1: Tensor,
        k1: Tensor,
        q2: Tensor,
        k2: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        value = (
            torch.exp((q1.float() * k1.float()).sum())
            - torch.exp((q2.float() * k2.float()).sum())
            + float(self.lambda_init)
        )
        return value.to(device=device, dtype=dtype)

    @staticmethod
    def _normalized_positive_effective_attention(
        stage2_signed: Tensor,
        stage1_signed: Tensor,
    ) -> Tensor:
        effective = stage2_signed @ stage1_signed
        positive = effective.clamp_min(0.0)
        positive_mass = positive.sum(dim=-1, keepdim=True)
        absolute = effective.abs()
        selected = torch.where(positive_mass > 1e-12, positive, absolute)
        selected_mass = selected.sum(dim=-1, keepdim=True)
        normalized = selected / selected_mass.clamp_min(1e-12)
        uniform = torch.full_like(normalized, 1.0 / float(normalized.size(-1)))
        return torch.where(selected_mass > 1e-12, normalized, uniform)

    def forward(
        self,
        inputs: Tensor,
        return_attention: bool = False,
        *,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_count: int = 0,
        patch_indices: Optional[Tensor] = None,
    ):
        grid_height, grid_width, prefix_count = self._validate_dense_grid(
            inputs,
            grid_size=grid_size,
            prefix_count=prefix_count,
            patch_indices=patch_indices,
        )
        batch_size, token_count, dim = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        patch_query = query[:, :, prefix_count:, :]
        patch_query_map = patch_query.permute(0, 1, 3, 2).reshape(
            batch_size,
            dim,
            grid_height,
            grid_width,
        )
        pooled_query = F.adaptive_avg_pool2d(
            patch_query_map,
            output_size=(self.visual_contrast_side, self.visual_contrast_side),
        ).flatten(2).transpose(1, 2)
        positive_tokens = pooled_query + self.positive_embedding.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        negative_tokens = pooled_query + self.negative_embedding.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        positive_tokens = positive_tokens.reshape(
            batch_size,
            self.visual_contrast_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        negative_tokens = negative_tokens.reshape(
            batch_size,
            self.visual_contrast_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        contrast_tokens = torch.cat((positive_tokens, negative_tokens), dim=2)

        lambda1 = self._lambda_value(
            self.stage1_lambda_q1,
            self.stage1_lambda_k1,
            self.stage1_lambda_q2,
            self.stage1_lambda_k2,
            dtype=query.dtype,
            device=query.device,
        )
        stage1_attention = torch.softmax(
            (contrast_tokens * self.scale) @ key.transpose(-2, -1),
            dim=-1,
        )
        # The released classifier defines attention dropout for API parity but
        # applies the normalized maps directly in both contrast stages.
        stage1_values = stage1_attention @ value
        positive_values = stage1_values[:, :, : self.visual_contrast_tokens]
        negative_values = stage1_values[:, :, self.visual_contrast_tokens :]
        contrast_values = self.stage1_norm(
            positive_values - lambda1 * negative_values
        ) * (1.0 - float(self.lambda_init))

        lambda2 = self._lambda_value(
            self.stage2_lambda_q1,
            self.stage2_lambda_k1,
            self.stage2_lambda_q2,
            self.stage2_lambda_k2,
            dtype=query.dtype,
            device=query.device,
        )
        # The released LinearDiff classifier normalizes the concatenated
        # positive/negative token bank jointly at Stage II.
        stage2_attention = torch.softmax(
            (query * self.scale) @ contrast_tokens.transpose(-2, -1),
            dim=-1,
        )
        positive_query = stage2_attention[:, :, :, : self.visual_contrast_tokens]
        negative_query = stage2_attention[:, :, :, self.visual_contrast_tokens :]
        stage2_signed = positive_query - lambda2 * negative_query
        output = self.stage2_norm(stage2_signed @ contrast_values) * (
            1.0 - float(self.lambda_init)
        )
        stage2_contrast_output = output
        output = output.transpose(1, 2).reshape(batch_size, token_count, dim)

        value_patches = value[:, :, prefix_count:, :].permute(0, 1, 3, 2).reshape(
            batch_size,
            dim,
            grid_height,
            grid_width,
        )
        spatial_residual = self.depthwise_value(value_patches).flatten(2).transpose(1, 2)
        if prefix_count:
            prefix_residual = spatial_residual.new_zeros(
                batch_size,
                prefix_count,
                dim,
            )
            spatial_residual = torch.cat((prefix_residual, spatial_residual), dim=1)
        output = output + spatial_residual
        output = self.projection_dropout(self.proj(output))

        stage1_signed = (
            stage1_attention[:, :, : self.visual_contrast_tokens]
            - lambda1 * stage1_attention[:, :, self.visual_contrast_tokens :]
        )
        positive_stage1_strength = positive_values.detach().float().norm(dim=-1).mean()
        negative_stage1_strength = (
            lambda1.detach().float().abs()
            * negative_values.detach().float().norm(dim=-1).mean()
        )
        stage1_strength = (positive_stage1_strength + negative_stage1_strength).clamp_min(
            1e-12
        )
        self._last_trace = {
            "lambda_stage1": lambda1.detach().float(),
            "lambda_stage2": lambda2.detach().float(),
            "stage1_positive_mass": positive_stage1_strength / stage1_strength,
            "stage1_negative_mass": negative_stage1_strength / stage1_strength,
            "stage2_positive_mass": stage2_attention[
                :, :, :, : self.visual_contrast_tokens
            ].detach().float().sum(dim=-1).mean(),
            "stage2_negative_mass": stage2_attention[
                :, :, :, self.visual_contrast_tokens :
            ].detach().float().sum(dim=-1).mean(),
            "stage1_contrast_norm": contrast_values.detach().float().norm(dim=-1).mean(),
            "stage2_contrast_norm": (
                stage2_contrast_output.detach().float().norm(dim=-1).mean()
            ),
        }
        if not return_attention:
            return output
        attention = self._normalized_positive_effective_attention(
            stage2_signed,
            stage1_signed,
        )
        return output, attention

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

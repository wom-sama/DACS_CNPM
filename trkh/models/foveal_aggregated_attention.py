from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@torch.no_grad()
def _sequence_length_and_mask(
    input_resolution: Tuple[int, int],
    window_size: int,
) -> Tuple[Tensor, Tensor]:
    height, width = (int(input_resolution[0]), int(input_resolution[1]))
    support = F.unfold(
        torch.ones(1, 1, height, width),
        kernel_size=int(window_size),
        padding=int(window_size) // 2,
        stride=1,
    )
    local_length = support.sum(dim=-2).squeeze(0).unsqueeze(-1)
    padding_mask = support.squeeze(0).permute(1, 0).eq(0)
    return local_length, padding_mask


@torch.no_grad()
def _relative_position_cpb(
    query_size: Tuple[int, int],
    key_size: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    query_height, query_width = (int(query_size[0]), int(query_size[1]))
    key_height, key_width = (int(key_size[0]), int(key_size[1]))
    axis_qh = torch.arange(query_height, dtype=torch.float32)
    axis_qw = torch.arange(query_width, dtype=torch.float32)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_height).squeeze(0)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_width).squeeze(0)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw, indexing="ij")
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw, indexing="ij")
    axis_qh = axis_qh.reshape(-1)
    axis_qw = axis_qw.reshape(-1)
    axis_kh = axis_kh.reshape(-1)
    axis_kw = axis_kw.reshape(-1)
    height_denominator = max(1, query_height - 1)
    width_denominator = max(1, query_width - 1)
    relative_height = (
        (axis_qh[:, None] - axis_kh[None, :])
        / float(height_denominator)
        * 8.0
    )
    relative_width = (
        (axis_qw[:, None] - axis_kw[None, :])
        / float(width_denominator)
        * 8.0
    )
    relative = torch.stack((relative_height, relative_width), dim=-1).reshape(-1, 2)
    table, index = torch.unique(relative, return_inverse=True, dim=0)
    table = (
        torch.sign(table)
        * torch.log2(table.abs() + 1.0)
        / math.log2(8.0)
    )
    return index, table


@torch.no_grad()
def _local_patch_indices(
    input_resolution: Tuple[int, int],
    window_size: int,
) -> Tuple[Tensor, Tensor]:
    height, width = (int(input_resolution[0]), int(input_resolution[1]))
    values = torch.arange(height * width, dtype=torch.float32).add_(1.0)
    values = values.reshape(1, 1, height, width)
    unfolded = F.unfold(
        values,
        kernel_size=int(window_size),
        padding=int(window_size) // 2,
        stride=1,
    )
    indices = unfolded.squeeze(0).permute(1, 0).to(dtype=torch.long).sub_(1)
    valid = indices.ge(0)
    return indices.clamp_min_(0), valid


@torch.no_grad()
def _pool_to_patch_matrix(
    input_resolution: Tuple[int, int],
    pool_resolution: Tuple[int, int],
) -> Tensor:
    height, width = (int(input_resolution[0]), int(input_resolution[1]))
    patch_count = height * width
    basis = torch.eye(patch_count, dtype=torch.float32).reshape(
        patch_count,
        1,
        height,
        width,
    )
    pooled = F.adaptive_avg_pool2d(basis, output_size=pool_resolution)
    return pooled.squeeze(1).flatten(1).transpose(0, 1).contiguous()


class FovealAggregatedAttention(nn.Module):
    """Prefix-aware TransNeXt Aggregated Attention on one dense patch grid."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 8,
        *,
        window_size: int = 3,
        fixed_pool_size: int = 4,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        window_size = int(window_size)
        fixed_pool_size = int(fixed_pool_size)
        input_resolution = (
            int(input_resolution[0]),
            int(input_resolution[1]),
        )
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("FAA dimension must be positive and divisible by num_heads.")
        if min(input_resolution) <= 0:
            raise ValueError("FAA input_resolution must be positive.")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("FAA window_size must be odd and >= 3.")
        if fixed_pool_size <= 0 or fixed_pool_size >= min(input_resolution):
            raise ValueError(
                "FAA fixed_pool_size must be positive and smaller than the patch grid."
            )
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("FAA attention_dropout must be in [0, 1).")
        if not 0.0 <= float(projection_dropout) < 1.0:
            raise ValueError("FAA projection_dropout must be in [0, 1).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.local_length = window_size * window_size
        self.pool_resolution = (fixed_pool_size, fixed_pool_size)
        self.pool_length = fixed_pool_size * fixed_pool_size

        # qkv/proj preserve the standard TRKH attention state-key schema.
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.projection_dropout = nn.Dropout(float(projection_dropout))

        initial_temperature = torch.log(
            torch.exp(torch.ones(num_heads, 1, 1) / 0.24) - 1.0
        )
        self.temperature = nn.Parameter(initial_temperature)
        self.query_embedding = nn.Parameter(torch.empty(num_heads, 1, self.head_dim))
        nn.init.trunc_normal_(self.query_embedding, mean=0.0, std=0.02)

        self.unfold = nn.Unfold(
            kernel_size=window_size,
            padding=window_size // 2,
            stride=1,
        )
        self.pool = nn.AdaptiveAvgPool2d(self.pool_resolution)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.pool_norm = nn.LayerNorm(dim)
        self.pool_activation = nn.GELU()
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_activation = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)
        self.relative_position_bias_local = nn.Parameter(
            torch.empty(num_heads, self.local_length)
        )
        nn.init.trunc_normal_(
            self.relative_position_bias_local,
            mean=0.0,
            std=0.0004,
        )
        self.learnable_tokens = nn.Parameter(
            torch.empty(num_heads, self.head_dim, self.local_length)
        )
        nn.init.trunc_normal_(self.learnable_tokens, mean=0.0, std=0.02)
        self.learnable_bias = nn.Parameter(
            torch.zeros(num_heads, 1, self.local_length)
        )

        local_length, padding_mask = _sequence_length_and_mask(
            input_resolution,
            window_size,
        )
        sequence_scale = torch.as_tensor(
            np.log(local_length.numpy() + float(self.pool_length)),
            dtype=torch.float32,
        )
        relative_position_index, relative_coords_table = _relative_position_cpb(
            input_resolution,
            self.pool_resolution,
        )
        local_indices, local_valid = _local_patch_indices(
            input_resolution,
            window_size,
        )
        self.register_buffer("sequence_length_scale", sequence_scale, persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )
        self.register_buffer(
            "relative_coords_table",
            relative_coords_table,
            persistent=False,
        )
        self.register_buffer("local_indices", local_indices, persistent=False)
        self.register_buffer("local_valid", local_valid, persistent=False)
        self.register_buffer(
            "pool_to_patch",
            _pool_to_patch_matrix(input_resolution, self.pool_resolution),
            persistent=False,
        )
        self._last_trace: Dict[str, Tensor] = {}

    def copy_shared_projections_from(self, source: nn.Module) -> None:
        source_qkv = getattr(source, "qkv", None)
        source_proj = getattr(source, "proj", None)
        if not isinstance(source_qkv, nn.Linear) or not isinstance(source_proj, nn.Linear):
            raise TypeError("FAA projection source must expose Linear qkv and proj modules.")
        if source_qkv.weight.shape != self.qkv.weight.shape:
            raise ValueError("FAA qkv projection shape differs from source attention.")
        if source_proj.weight.shape != self.proj.weight.shape:
            raise ValueError("FAA output projection shape differs from source attention.")
        with torch.no_grad():
            self.qkv.weight.copy_(source_qkv.weight)
            self.qkv.bias.copy_(source_qkv.bias)
            self.proj.weight.copy_(source_proj.weight)
            self.proj.bias.copy_(source_proj.bias)

    def _validate_dense_grid(
        self,
        inputs: Tensor,
        *,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor],
    ) -> int:
        if inputs.ndim != 3 or int(inputs.size(-1)) != self.dim:
            raise ValueError("FAA expects tokens with shape [B, N, dim].")
        if grid_size is None or tuple(int(value) for value in grid_size) != self.input_resolution:
            raise ValueError(
                f"FAA requires the locked dense grid {self.input_resolution}."
            )
        prefix_count = int(prefix_count)
        if prefix_count < 0 or prefix_count > int(inputs.size(1)):
            raise ValueError("FAA prefix_count is out of range.")
        patch_count = int(inputs.size(1)) - prefix_count
        expected_patch_count = self.input_resolution[0] * self.input_resolution[1]
        if patch_count != expected_patch_count:
            raise ValueError(
                "FAA requires the complete dense patch grid before token pruning."
            )
        if torch.is_tensor(patch_indices):
            if tuple(patch_indices.shape) != (int(inputs.size(0)), patch_count):
                raise ValueError("FAA patch_indices do not match the dense patch grid.")
            expected = torch.arange(
                patch_count,
                device=patch_indices.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(inputs.size(0)), -1)
            if not torch.equal(patch_indices.to(dtype=torch.long), expected):
                raise ValueError("FAA requires identity-ordered dense patch_indices.")
        return prefix_count

    def _project_query(self, inputs: Tensor) -> Tensor:
        return F.linear(
            inputs,
            self.qkv.weight[: self.dim],
            self.qkv.bias[: self.dim],
        )

    def _project_key_value(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        key_value = F.linear(
            inputs,
            self.qkv.weight[self.dim :],
            self.qkv.bias[self.dim :],
        )
        return key_value.chunk(2, dim=-1)

    def _dense_attention(
        self,
        *,
        prefix_attention: Tensor,
        local_attention: Tensor,
        pooled_attention: Tensor,
        prefix_count: int,
    ) -> Tensor:
        batch_size, num_heads, patch_count, _ = local_attention.shape
        local_dense = local_attention.new_zeros(
            batch_size,
            num_heads,
            patch_count,
            patch_count,
        )
        local_indices = self.local_indices.to(device=local_attention.device)
        scatter_indices = local_indices.view(1, 1, patch_count, self.local_length)
        scatter_indices = scatter_indices.expand(batch_size, num_heads, -1, -1)
        valid = self.local_valid.to(
            device=local_attention.device,
            dtype=local_attention.dtype,
        ).view(1, 1, patch_count, self.local_length)
        local_dense.scatter_add_(
            dim=-1,
            index=scatter_indices,
            src=local_attention * valid,
        )
        pooled_dense = torch.matmul(
            pooled_attention,
            self.pool_to_patch.to(
                device=pooled_attention.device,
                dtype=pooled_attention.dtype,
            ),
        )
        patch_dense = local_dense + pooled_dense
        if prefix_count:
            patch_prefix = patch_dense.new_zeros(
                batch_size,
                num_heads,
                patch_count,
                prefix_count,
            )
            patch_dense = torch.cat((patch_prefix, patch_dense), dim=-1)
            return torch.cat((prefix_attention, patch_dense), dim=-2)
        return patch_dense

    def forward(
        self,
        inputs: Tensor,
        return_attention: bool = False,
        *,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_count: int = 0,
        patch_indices: Optional[Tensor] = None,
    ):
        prefix_count = self._validate_dense_grid(
            inputs,
            grid_size=grid_size,
            prefix_count=prefix_count,
            patch_indices=patch_indices,
        )
        batch_size, token_count, _ = inputs.shape
        patch_count = token_count - prefix_count

        query_linear = self._project_query(inputs)
        key_linear, value_linear = self._project_key_value(inputs)
        query = query_linear.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        key = key_linear.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        value = value_linear.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        if prefix_count:
            prefix_logits = (
                query[:, :, :prefix_count] @ key.transpose(-2, -1)
            ) * self.scale
            prefix_attention_raw = prefix_logits.softmax(dim=-1)
            prefix_attention = self.attention_dropout(prefix_attention_raw)
            prefix_output = prefix_attention @ value
        else:
            prefix_attention_raw = query.new_empty(
                batch_size,
                self.num_heads,
                0,
                token_count,
            )
            prefix_attention = prefix_attention_raw
            prefix_output = query.new_empty(
                batch_size,
                self.num_heads,
                0,
                self.head_dim,
            )

        patch_inputs = inputs[:, prefix_count:]
        query_patch = query[:, :, prefix_count:]
        key_patch = key[:, :, prefix_count:]
        value_patch = value[:, :, prefix_count:]
        query_normalized = F.normalize(query_patch, dim=-1)
        query_scaled = (
            query_normalized
            + self.query_embedding.to(device=inputs.device, dtype=inputs.dtype)
        )
        query_scaled = (
            query_scaled
            * F.softplus(self.temperature).to(dtype=inputs.dtype)
            * self.sequence_length_scale.to(
                device=inputs.device,
                dtype=inputs.dtype,
            )
        )

        key_patch_normalized = F.normalize(key_patch, dim=-1)
        key_flat = key_patch_normalized.permute(0, 2, 1, 3).reshape(
            batch_size,
            patch_count,
            self.dim,
        )
        value_flat = value_patch.permute(0, 2, 1, 3).reshape(
            batch_size,
            patch_count,
            self.dim,
        )
        local_key_value = torch.cat((key_flat, value_flat), dim=-1)
        local_key_value = local_key_value.permute(0, 2, 1).reshape(
            batch_size,
            2 * self.dim,
            self.input_resolution[0],
            self.input_resolution[1],
        )
        local_key, local_value = self.unfold(local_key_value).reshape(
            batch_size,
            2 * self.num_heads,
            self.head_dim,
            self.local_length,
            patch_count,
        ).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        local_logits = (
            (query_scaled.unsqueeze(-2) @ local_key).squeeze(-2)
            + self.relative_position_bias_local.unsqueeze(1)
        )
        local_logits = local_logits.masked_fill(
            self.padding_mask.to(device=inputs.device).view(
                1,
                1,
                patch_count,
                self.local_length,
            ),
            float("-inf"),
        )

        patch_map = patch_inputs.permute(0, 2, 1).reshape(
            batch_size,
            self.dim,
            self.input_resolution[0],
            self.input_resolution[1],
        ).contiguous()
        pooled_inputs = self.pool(self.pool_activation(self.sr(patch_map)))
        pooled_inputs = pooled_inputs.reshape(
            batch_size,
            self.dim,
            self.pool_length,
        ).permute(0, 2, 1)
        pooled_inputs = self.pool_norm(pooled_inputs)
        pooled_key_linear, pooled_value_linear = self._project_key_value(pooled_inputs)
        pooled_key = pooled_key_linear.reshape(
            batch_size,
            self.pool_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        pooled_value = pooled_value_linear.reshape(
            batch_size,
            self.pool_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        pooled_bias = self.cpb_fc2(
            self.cpb_activation(self.cpb_fc1(self.relative_coords_table))
        ).transpose(0, 1)
        pooled_bias = pooled_bias[:, self.relative_position_index.reshape(-1)].reshape(
            self.num_heads,
            patch_count,
            self.pool_length,
        )
        pooled_logits = (
            query_scaled @ F.normalize(pooled_key, dim=-1).transpose(-2, -1)
            + pooled_bias
        )

        aggregated_attention_raw = torch.cat(
            (local_logits, pooled_logits),
            dim=-1,
        ).softmax(dim=-1)
        aggregated_attention = self.attention_dropout(aggregated_attention_raw)
        local_attention, pooled_attention = torch.split(
            aggregated_attention,
            (self.local_length, self.pool_length),
            dim=-1,
        )
        local_output = (
            (
                (query_normalized @ self.learnable_tokens)
                + self.learnable_bias
                + local_attention
            ).unsqueeze(-2)
            @ local_value.transpose(-2, -1)
        ).squeeze(-2)
        pooled_output = pooled_attention @ pooled_value
        patch_output = local_output + pooled_output

        output = torch.cat((prefix_output, patch_output), dim=2)
        output = output.transpose(1, 2).reshape(batch_size, token_count, self.dim)
        output = self.projection_dropout(self.proj(output))

        raw_local, raw_pooled = torch.split(
            aggregated_attention_raw,
            (self.local_length, self.pool_length),
            dim=-1,
        )
        local_mass = raw_local.detach().float().sum(dim=-1)
        pooled_mass = raw_pooled.detach().float().sum(dim=-1)
        self._last_trace = {
            "local_mass_map": local_mass.mean(dim=1),
            "local_mass_mean": local_mass.mean(),
            "local_mass_min": local_mass.amin(),
            "local_mass_max": local_mass.amax(),
            "pooled_mass_map": pooled_mass.mean(dim=1),
            "pooled_mass_mean": pooled_mass.mean(),
            "pooled_mass_min": pooled_mass.amin(),
            "pooled_mass_max": pooled_mass.amax(),
            "dual_route_fraction": (
                local_mass.ge(0.05) & pooled_mass.ge(0.05)
            ).float().mean(),
            "temperature_mean": F.softplus(self.temperature.detach().float()).mean(),
            "patch_count": torch.tensor(float(patch_count), device=inputs.device),
        }
        if return_attention:
            dense_attention = self._dense_attention(
                prefix_attention=prefix_attention,
                local_attention=local_attention,
                pooled_attention=pooled_attention,
                prefix_count=prefix_count,
            )
            return output, dense_attention
        return output

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

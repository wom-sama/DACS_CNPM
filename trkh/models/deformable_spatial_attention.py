from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _LayerNormProxy(nn.Module):
    """Apply LayerNorm over channels while preserving BCHW layout."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DeformableSpatialAttention(nn.Module):
    """Prefix-aware CVPR-2022 DAT attention on one dense patch grid."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 8,
        *,
        offset_groups: int = 2,
        offset_kernel_size: int = 5,
        offset_stride: int = 1,
        offset_range_factor: float = 2.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        offset_groups = int(offset_groups)
        offset_kernel_size = int(offset_kernel_size)
        offset_stride = int(offset_stride)
        input_resolution = tuple(int(value) for value in input_resolution)
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("DAT dimension must be positive and divisible by heads.")
        if offset_groups <= 0 or dim % offset_groups != 0:
            raise ValueError("DAT dimension must be divisible by offset_groups.")
        if num_heads % offset_groups != 0:
            raise ValueError("DAT heads must be divisible by offset_groups.")
        if len(input_resolution) != 2 or min(input_resolution) <= 0:
            raise ValueError("DAT input_resolution must contain two positive values.")
        if offset_kernel_size < 3 or offset_kernel_size % 2 == 0:
            raise ValueError("DAT offset kernel must be odd and >= 3.")
        if offset_stride != 1:
            raise ValueError("The locked TRKH DAT route requires offset_stride=1.")
        if float(offset_range_factor) <= 0.0:
            raise ValueError("DAT offset_range_factor must be positive.")
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("DAT attention_dropout must be in [0, 1).")
        if not 0.0 <= float(projection_dropout) < 1.0:
            raise ValueError("DAT projection_dropout must be in [0, 1).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.input_resolution = input_resolution
        self.patch_count = input_resolution[0] * input_resolution[1]
        self.offset_groups = offset_groups
        self.group_channels = dim // offset_groups
        self.heads_per_group = num_heads // offset_groups
        self.offset_kernel_size = offset_kernel_size
        self.offset_stride = offset_stride
        self.offset_range_factor = float(offset_range_factor)

        # Preserve the standard TRKH attention state-key schema.
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.projection_dropout = nn.Dropout(float(projection_dropout))

        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channels,
                self.group_channels,
                kernel_size=offset_kernel_size,
                stride=offset_stride,
                padding=offset_kernel_size // 2,
                groups=self.group_channels,
            ),
            _LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv2d(
                self.group_channels,
                2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        height, width = input_resolution
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_heads, 2 * height - 1, 2 * width - 1)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.01)
        self._last_trace: Dict[str, Tensor] = {}

    def copy_shared_projections_from(self, source: nn.Module) -> None:
        source_qkv = getattr(source, "qkv", None)
        source_proj = getattr(source, "proj", None)
        if not isinstance(source_qkv, nn.Linear) or not isinstance(source_proj, nn.Linear):
            raise TypeError("DAT projection source must expose Linear qkv and proj modules.")
        if source_qkv.weight.shape != self.qkv.weight.shape:
            raise ValueError("DAT qkv projection shape differs from source attention.")
        if source_proj.weight.shape != self.proj.weight.shape:
            raise ValueError("DAT output projection shape differs from source attention.")
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
            raise ValueError("DAT expects tokens with shape [B, N, dim].")
        observed_grid = (
            tuple(int(value) for value in grid_size)
            if grid_size is not None
            else None
        )
        if observed_grid != self.input_resolution:
            raise ValueError(f"DAT requires the locked dense grid {self.input_resolution}.")
        prefix_count = int(prefix_count)
        if prefix_count < 0 or prefix_count > int(inputs.size(1)):
            raise ValueError("DAT prefix_count is out of range.")
        patch_count = int(inputs.size(1)) - prefix_count
        if patch_count != self.patch_count:
            raise ValueError("DAT requires the complete patch grid before token pruning.")
        if torch.is_tensor(patch_indices):
            if tuple(patch_indices.shape) != (int(inputs.size(0)), self.patch_count):
                raise ValueError("DAT patch_indices do not match the dense patch grid.")
            expected = torch.arange(
                self.patch_count,
                device=patch_indices.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(inputs.size(0)), -1)
            if not torch.equal(patch_indices.to(dtype=torch.long), expected):
                raise ValueError("DAT requires identity-ordered dense patch_indices.")
        return prefix_count

    @torch.no_grad()
    def _reference_points(
        self,
        *,
        height: int,
        width: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        reference_y, reference_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
            indexing="ij",
        )
        reference_y = reference_y / float(height) * 2.0 - 1.0
        reference_x = reference_x / float(width) * 2.0 - 1.0
        reference = torch.stack((reference_y, reference_x), dim=-1)
        return reference.unsqueeze(0).expand(
            int(batch_size) * self.offset_groups,
            -1,
            -1,
            -1,
        )

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

    def _sample_patch_features(
        self,
        patch_map: Tensor,
        query_map: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, height, width = patch_map.shape
        grouped_query = query_map.reshape(
            batch_size * self.offset_groups,
            self.group_channels,
            height,
            width,
        )
        offset_channels = self.conv_offset(grouped_query)
        sampled_height, sampled_width = offset_channels.shape[-2:]
        if (sampled_height, sampled_width) != self.input_resolution:
            raise ValueError("Locked DAT sampling must preserve the 16x16 grid.")
        offset_range = offset_channels.new_tensor(
            [1.0 / sampled_height, 1.0 / sampled_width]
        ).reshape(1, 2, 1, 1)
        offset_channels = (
            offset_channels.tanh()
            * offset_range
            * self.offset_range_factor
        )
        offsets = offset_channels.permute(0, 2, 3, 1)
        reference = self._reference_points(
            height=sampled_height,
            width=sampled_width,
            batch_size=batch_size,
            dtype=patch_map.dtype,
            device=patch_map.device,
        ).to(dtype=offsets.dtype)
        positions = (offsets + reference).to(dtype=patch_map.dtype)
        grouped_inputs = patch_map.reshape(
            batch_size * self.offset_groups,
            self.group_channels,
            height,
            width,
        )
        sampled = F.grid_sample(
            grouped_inputs,
            positions[..., (1, 0)].to(dtype=grouped_inputs.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(
            batch_size,
            self.dim,
            sampled_height * sampled_width,
        ).transpose(1, 2)
        return sampled, positions, reference, offsets

    def _relative_position_bias(
        self,
        *,
        positions: Tensor,
        batch_size: int,
        query_height: int,
        query_width: int,
    ) -> Tensor:
        sample_count = int(positions.size(1) * positions.size(2))
        query_grid = self._reference_points(
            height=query_height,
            width=query_width,
            batch_size=batch_size,
            dtype=positions.dtype,
            device=positions.device,
        ).to(dtype=positions.dtype)
        displacement = (
            query_grid.reshape(
                batch_size * self.offset_groups,
                query_height * query_width,
                2,
            ).unsqueeze(2)
            - positions.reshape(
                batch_size * self.offset_groups,
                sample_count,
                2,
            ).unsqueeze(1)
        ) * 0.5
        bias_table = self.relative_position_bias_table.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )
        bias = F.grid_sample(
            bias_table.reshape(
                batch_size * self.offset_groups,
                self.heads_per_group,
                2 * query_height - 1,
                2 * query_width - 1,
            ),
            displacement[..., (1, 0)].to(dtype=bias_table.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return bias.reshape(
            batch_size,
            self.num_heads,
            query_height * query_width,
            sample_count,
        )

    def _positions_per_head(self, positions: Tensor, batch_size: int) -> Tensor:
        height, width = positions.shape[1:3]
        return positions.reshape(
            batch_size,
            self.offset_groups,
            height,
            width,
            2,
        ).unsqueeze(2).expand(
            -1,
            -1,
            self.heads_per_group,
            -1,
            -1,
            -1,
        ).reshape(batch_size, self.num_heads, height * width, 2)

    def sampled_valid_interpolation_mass(
        self,
        attention: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Return retained bilinear mass without materializing a dense proxy."""

        if attention.ndim != 4:
            raise ValueError("DAT sampled attention must have shape [B,H,Q,S].")
        batch_size, num_heads, _, sample_count = attention.shape
        if num_heads != self.num_heads:
            raise ValueError("DAT sampled attention head count does not match the module.")
        position_heads = self._positions_per_head(positions, batch_size)
        if int(position_heads.size(2)) != sample_count:
            raise ValueError("DAT sampled positions do not match attention samples.")
        height, width = self.input_resolution
        y = (position_heads[..., 0] + 1.0) * 0.5 * float(height - 1)
        x = (position_heads[..., 1] + 1.0) * 0.5 * float(width - 1)
        y0 = torch.floor(y)
        x0 = torch.floor(x)
        y1 = y0 + 1.0
        x1 = x0 + 1.0
        sample_valid_mass = attention.new_zeros(position_heads.shape[:-1])
        for yy, xx, weight in (
            (y0, x0, (y1 - y) * (x1 - x)),
            (y0, x1, (y1 - y) * (x - x0)),
            (y1, x0, (y - y0) * (x1 - x)),
            (y1, x1, (y - y0) * (x - x0)),
        ):
            valid = yy.ge(0) & yy.lt(height) & xx.ge(0) & xx.lt(width)
            typed_weight = weight.to(dtype=attention.dtype)
            sample_valid_mass = sample_valid_mass + typed_weight * valid.to(
                dtype=attention.dtype
            )
        return (attention * sample_valid_mass.unsqueeze(2)).sum(dim=-1)

    def bilinear_scatter_attention(
        self,
        attention: Tensor,
        positions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Scatter sampled-key attention onto source patch columns."""

        if attention.ndim != 4:
            raise ValueError("DAT sampled attention must have shape [B,H,Q,S].")
        batch_size, num_heads, query_count, sample_count = attention.shape
        if num_heads != self.num_heads or query_count != self.patch_count:
            raise ValueError("DAT sampled attention shape does not match the module.")
        position_heads = self._positions_per_head(positions, batch_size)
        if int(position_heads.size(2)) != sample_count:
            raise ValueError("DAT sampled positions do not match attention samples.")

        height, width = self.input_resolution
        y = (position_heads[..., 0] + 1.0) * 0.5 * float(height - 1)
        x = (position_heads[..., 1] + 1.0) * 0.5 * float(width - 1)
        y0 = torch.floor(y)
        x0 = torch.floor(x)
        y1 = y0 + 1.0
        x1 = x0 + 1.0

        dense = attention.new_zeros(
            batch_size,
            num_heads,
            query_count,
            self.patch_count,
        )
        for yy, xx, weight in (
            (y0, x0, (y1 - y) * (x1 - x)),
            (y0, x1, (y1 - y) * (x - x0)),
            (y1, x0, (y - y0) * (x1 - x)),
            (y1, x1, (y - y0) * (x - x0)),
        ):
            valid = yy.ge(0) & yy.lt(height) & xx.ge(0) & xx.lt(width)
            flat_index = (
                yy.clamp(0, height - 1).to(dtype=torch.long) * width
                + xx.clamp(0, width - 1).to(dtype=torch.long)
            )
            scatter_index = flat_index.unsqueeze(2).expand(-1, -1, query_count, -1)
            typed_weight = weight.to(dtype=attention.dtype)
            source = attention * (
                typed_weight * valid.to(dtype=attention.dtype)
            ).unsqueeze(2)
            dense.scatter_add_(dim=-1, index=scatter_index, src=source)

        valid_mass = dense.sum(dim=-1)
        dense = dense / valid_mass.clamp_min(torch.finfo(dense.dtype).eps).unsqueeze(-1)
        return dense, valid_mass

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
        height, width = self.input_resolution

        query_linear, all_key_linear, all_value_linear = self.qkv(inputs).chunk(
            3, dim=-1
        )
        query = query_linear.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        all_key = all_key_linear.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        all_value = all_value_linear.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        if prefix_count:
            prefix_logits = (
                query[:, :, :prefix_count] @ all_key.transpose(-2, -1)
            ) * self.scale
            prefix_attention_raw = prefix_logits.softmax(dim=-1)
            prefix_attention = self.attention_dropout(prefix_attention_raw)
            prefix_output = prefix_attention @ all_value
        else:
            prefix_attention = query.new_empty(
                batch_size,
                self.num_heads,
                0,
                token_count,
            )
            prefix_output = query.new_empty(
                batch_size,
                self.num_heads,
                0,
                self.head_dim,
            )

        patch_inputs = inputs[:, prefix_count:]
        patch_map = patch_inputs.transpose(1, 2).reshape(
            batch_size,
            self.dim,
            height,
            width,
        ).contiguous()
        patch_query_map = query_linear[:, prefix_count:].transpose(1, 2).reshape(
            batch_size,
            self.dim,
            height,
            width,
        ).contiguous()
        sampled_inputs, positions, reference, offsets = self._sample_patch_features(
            patch_map,
            patch_query_map,
        )
        sampled_key_linear, sampled_value_linear = self._project_key_value(
            sampled_inputs
        )
        sampled_key = sampled_key_linear.reshape(
            batch_size,
            self.patch_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        sampled_value = sampled_value_linear.reshape(
            batch_size,
            self.patch_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        patch_query = query[:, :, prefix_count:]
        patch_logits = (
            patch_query @ sampled_key.transpose(-2, -1)
        ) * self.scale
        patch_logits = patch_logits + self._relative_position_bias(
            positions=positions,
            batch_size=batch_size,
            query_height=height,
            query_width=width,
        )
        patch_attention_raw = patch_logits.softmax(dim=-1)
        patch_attention = self.attention_dropout(patch_attention_raw)
        patch_output = patch_attention @ sampled_value

        output = torch.cat((prefix_output, patch_output), dim=2)
        output = output.transpose(1, 2).reshape(batch_size, token_count, self.dim)
        output = self.projection_dropout(self.proj(output))

        position_groups = positions.reshape(
            batch_size,
            self.offset_groups,
            height,
            width,
            2,
        )
        offset_groups = offsets.reshape_as(position_groups)
        if self.offset_groups > 1:
            group_delta = position_groups[:, 1:] - position_groups[:, :-1]
            group_rms = group_delta.float().square().mean().sqrt()
        else:
            group_rms = positions.new_zeros((), dtype=torch.float32)
        patch_dense = None
        if return_attention:
            patch_dense, proxy_valid_mass = self.bilinear_scatter_attention(
                patch_attention,
                positions,
            )
        else:
            proxy_valid_mass = None
        if proxy_valid_mass is not None and not self.training:
            valid_mass = proxy_valid_mass.detach()
        else:
            with torch.no_grad():
                valid_mass = self.sampled_valid_interpolation_mass(
                    patch_attention_raw.detach(), positions.detach()
                )
        self._last_trace = {
            "positions": position_groups.detach(),
            "reference_positions": reference.reshape_as(position_groups).detach(),
            "offsets": offset_groups.detach(),
            "offset_rms": offset_groups.detach().float().square().mean().sqrt(),
            "offset_rms_per_group": offset_groups.detach().float().square().mean(
                dim=(0, 2, 3, 4)
            ).sqrt(),
            "inter_group_position_rms": group_rms.detach(),
            "valid_interpolation_mass_mean": valid_mass.detach().float().mean(),
            "valid_interpolation_mass_min": valid_mass.detach().float().amin(),
            "sample_attention_mean": patch_attention_raw.detach().float().mean(dim=2),
            "position_min": position_groups.detach().float().amin(),
            "position_max": position_groups.detach().float().amax(),
        }

        if return_attention:
            # Keep the native sampled-key probabilities only for explicit audit/XAI
            # calls; retaining this O(B*H*P^2) tensor during ordinary training would
            # materially increase memory use.
            self._last_trace["sample_attention"] = patch_attention_raw.detach()
            if patch_dense is None:
                raise RuntimeError("DAT pruning proxy was not constructed.")
            if prefix_count:
                patch_prefix = patch_dense.new_zeros(
                    batch_size,
                    self.num_heads,
                    self.patch_count,
                    prefix_count,
                )
                patch_dense = torch.cat((patch_prefix, patch_dense), dim=-1)
                dense_attention = torch.cat((prefix_attention, patch_dense), dim=-2)
            else:
                dense_attention = patch_dense
            return output, dense_attention
        return output

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class BiLevelRoutingAttention(nn.Module):
    """Prefix-aware BiFormer routing attention on one dense patch grid.

    Adapted from the MIT-licensed official BiFormer implementation:
    https://github.com/rayleizhu/BiFormer at
    1697bbbeafb8680524898f1dcaac10defd0604be.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 8,
        *,
        regions_per_axis: int = 4,
        topk: int = 4,
        local_context_kernel_size: int = 5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        regions_per_axis = int(regions_per_axis)
        topk = int(topk)
        local_context_kernel_size = int(local_context_kernel_size)
        input_resolution = tuple(int(value) for value in input_resolution)

        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("BRA dimension must be positive and divisible by num_heads.")
        if len(input_resolution) != 2 or min(input_resolution) <= 0:
            raise ValueError("BRA input_resolution must contain two positive values.")
        if regions_per_axis <= 0:
            raise ValueError("BRA regions_per_axis must be positive.")
        if any(value % regions_per_axis != 0 for value in input_resolution):
            raise ValueError("BRA patch grid must be divisible by regions_per_axis.")
        region_count = regions_per_axis * regions_per_axis
        if not 1 <= topk <= region_count:
            raise ValueError("BRA topk must be in [1, regions_per_axis**2].")
        if local_context_kernel_size <= 0 or local_context_kernel_size % 2 == 0:
            raise ValueError("BRA local_context_kernel_size must be positive and odd.")
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("BRA attention_dropout must be in [0, 1).")
        if not 0.0 <= float(projection_dropout) < 1.0:
            raise ValueError("BRA projection_dropout must be in [0, 1).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.input_resolution = input_resolution
        self.regions_per_axis = regions_per_axis
        self.region_count = region_count
        self.topk = topk
        self.region_height = input_resolution[0] // regions_per_axis
        self.region_width = input_resolution[1] // regions_per_axis
        self.tokens_per_region = self.region_height * self.region_width
        self.patch_count = input_resolution[0] * input_resolution[1]

        # Preserve the standard TRKH attention checkpoint schema.
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.proj = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(float(projection_dropout))
        self.local_context = nn.Conv2d(
            dim,
            dim,
            kernel_size=local_context_kernel_size,
            stride=1,
            padding=local_context_kernel_size // 2,
            groups=dim,
            bias=True,
        )

        region_token_indices = self._partition_patch_indices()
        region_flat_indices = region_token_indices.flatten()
        inverse_region_order = torch.empty_like(region_flat_indices)
        inverse_region_order[region_flat_indices] = torch.arange(self.patch_count)
        region_rows = torch.arange(regions_per_axis).repeat_interleave(
            regions_per_axis
        )
        region_columns = torch.arange(regions_per_axis).repeat(regions_per_axis)
        region_coordinates = torch.stack((region_rows, region_columns), dim=-1)
        self.register_buffer(
            "region_token_indices",
            region_token_indices,
            persistent=False,
        )
        self.register_buffer(
            "inverse_region_order",
            inverse_region_order,
            persistent=False,
        )
        self.register_buffer(
            "region_coordinates",
            region_coordinates,
            persistent=False,
        )
        self._last_trace: Dict[str, Tensor] = {}

    def _partition_patch_indices(self) -> Tensor:
        height, width = self.input_resolution
        regions = self.regions_per_axis
        return (
            torch.arange(height * width, dtype=torch.long)
            .reshape(height, width)
            .reshape(
                regions,
                self.region_height,
                regions,
                self.region_width,
            )
            .permute(0, 2, 1, 3)
            .reshape(self.region_count, self.tokens_per_region)
            .contiguous()
        )

    def copy_shared_projections_from(self, source: nn.Module) -> None:
        source_qkv = getattr(source, "qkv", None)
        source_proj = getattr(source, "proj", None)
        if not isinstance(source_qkv, nn.Linear) or not isinstance(source_proj, nn.Linear):
            raise TypeError("BRA projection source must expose Linear qkv and proj modules.")
        if source_qkv.weight.shape != self.qkv.weight.shape:
            raise ValueError("BRA qkv projection shape differs from source attention.")
        if source_proj.weight.shape != self.proj.weight.shape:
            raise ValueError("BRA output projection shape differs from source attention.")
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
            raise ValueError("BRA expects tokens with shape [B, N, dim].")
        if grid_size is None or tuple(int(value) for value in grid_size) != self.input_resolution:
            raise ValueError(f"BRA requires the locked dense grid {self.input_resolution}.")
        prefix_count = int(prefix_count)
        if prefix_count < 0 or prefix_count > int(inputs.size(1)):
            raise ValueError("BRA prefix_count is out of range.")
        if int(inputs.size(1)) - prefix_count != self.patch_count:
            raise ValueError("BRA requires the complete dense patch grid before pruning.")
        if torch.is_tensor(patch_indices):
            if tuple(patch_indices.shape) != (int(inputs.size(0)), self.patch_count):
                raise ValueError("BRA patch_indices do not match the dense patch grid.")
            expected = torch.arange(
                self.patch_count,
                device=patch_indices.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(inputs.size(0)), -1)
            if not torch.equal(patch_indices.to(dtype=torch.long), expected):
                raise ValueError("BRA requires identity-ordered dense patch_indices.")
        return prefix_count

    def _to_regions(self, tensor: Tensor) -> Tensor:
        batch_size, num_heads, patch_count, head_dim = tensor.shape
        if patch_count != self.patch_count or head_dim != self.head_dim:
            raise ValueError("BRA patch projection shape does not match its locked grid.")
        return (
            tensor.reshape(
                batch_size,
                num_heads,
                self.input_resolution[0],
                self.input_resolution[1],
                head_dim,
            )
            .reshape(
                batch_size,
                num_heads,
                self.regions_per_axis,
                self.region_height,
                self.regions_per_axis,
                self.region_width,
                head_dim,
            )
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(
                batch_size,
                num_heads,
                self.region_count,
                self.tokens_per_region,
                head_dim,
            )
            .contiguous()
        )

    def _from_regions(self, tensor: Tensor) -> Tensor:
        flattened = tensor.flatten(2, 3)
        inverse = self.inverse_region_order.to(device=tensor.device)
        return flattened.index_select(2, inverse)

    def _routing_graph(self, query_patch: Tensor, key_patch: Tensor) -> Tuple[Tensor, Tensor]:
        query_regions = self._to_regions(query_patch)
        key_regions = self._to_regions(key_patch)
        batch_size = int(query_patch.size(0))
        query_region = (
            query_regions.permute(0, 2, 3, 1, 4)
            .reshape(
                batch_size,
                self.region_count,
                self.tokens_per_region,
                self.dim,
            )
            .mean(dim=2)
            .detach()
        )
        key_region = (
            key_regions.permute(0, 2, 3, 1, 4)
            .reshape(
                batch_size,
                self.region_count,
                self.tokens_per_region,
                self.dim,
            )
            .mean(dim=2)
            .detach()
        )
        affinity = query_region @ key_region.transpose(-2, -1)
        if self.topk == self.region_count:
            indices = torch.arange(
                self.region_count,
                device=affinity.device,
                dtype=torch.long,
            ).view(1, 1, -1).expand(batch_size, self.region_count, -1)
        else:
            indices = torch.topk(affinity, k=self.topk, dim=-1).indices
        return affinity, indices

    def _gather_regions(self, tensor: Tensor, route_indices: Tensor) -> Tensor:
        regions = self._to_regions(tensor)
        batch_size, num_heads = int(tensor.size(0)), int(tensor.size(1))
        source = regions.unsqueeze(2).expand(
            -1,
            -1,
            self.region_count,
            -1,
            -1,
            -1,
        )
        gather_index = route_indices[:, None, :, :, None, None].expand(
            batch_size,
            num_heads,
            self.region_count,
            self.topk,
            self.tokens_per_region,
            self.head_dim,
        )
        return torch.gather(source, dim=3, index=gather_index).flatten(3, 4)

    def _selected_patch_indices(self, route_indices: Tensor) -> Tensor:
        source = self.region_token_indices.to(device=route_indices.device)
        source = source.view(1, 1, self.region_count, self.tokens_per_region)
        source = source.expand(int(route_indices.size(0)), self.region_count, -1, -1)
        gather_index = route_indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            self.tokens_per_region,
        )
        return torch.gather(source, dim=2, index=gather_index).flatten(2, 3)

    def _sparse_patch_attention(
        self,
        query_patch: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        prefix_count: int,
        route_indices: Tensor,
        return_dense_attention: bool,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        query_regions = self._to_regions(query_patch)
        key_regions = self._gather_regions(key[:, :, prefix_count:], route_indices)
        value_regions = self._gather_regions(value[:, :, prefix_count:], route_indices)
        batch_size, num_heads = int(query_patch.size(0)), int(query_patch.size(1))

        if prefix_count:
            prefix_key = key[:, :, :prefix_count].unsqueeze(2).expand(
                -1,
                -1,
                self.region_count,
                -1,
                -1,
            )
            prefix_value = value[:, :, :prefix_count].unsqueeze(2).expand(
                -1,
                -1,
                self.region_count,
                -1,
                -1,
            )
            key_regions = torch.cat((prefix_key, key_regions), dim=3)
            value_regions = torch.cat((prefix_value, value_regions), dim=3)

        logits = (query_regions @ key_regions.transpose(-2, -1)) * self.scale
        native_attention = self.attention_dropout(logits.softmax(dim=-1))
        patch_output = self._from_regions(native_attention @ value_regions)

        if not return_dense_attention:
            return patch_output, native_attention, None

        selected_patch_indices = self._selected_patch_indices(route_indices)
        dense_region = native_attention.new_zeros(
            batch_size,
            num_heads,
            self.region_count,
            self.tokens_per_region,
            prefix_count + self.patch_count,
        )
        if prefix_count:
            dense_region[..., :prefix_count] = native_attention[..., :prefix_count]
        sparse_patch_attention = native_attention[..., prefix_count:]
        scatter_index = selected_patch_indices[:, None, :, None, :].expand(
            batch_size,
            num_heads,
            self.region_count,
            self.tokens_per_region,
            -1,
        )
        dense_region.scatter_add_(
            dim=-1,
            index=scatter_index + prefix_count,
            src=sparse_patch_attention,
        )
        dense_patch_attention = self._from_regions(dense_region)
        return patch_output, native_attention, dense_patch_attention

    def _full_patch_attention(
        self,
        query_patch: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        logits = (query_patch @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(logits.softmax(dim=-1))
        return attention @ value, attention

    def _local_context(self, value_patch: Tensor) -> Tensor:
        batch_size = int(value_patch.size(0))
        value_map = (
            value_patch.transpose(1, 2)
            .reshape(batch_size, self.patch_count, self.dim)
            .transpose(1, 2)
            .reshape(
                batch_size,
                self.dim,
                self.input_resolution[0],
                self.input_resolution[1],
            )
        )
        context = self.local_context(value_map)
        return (
            context.flatten(2)
            .transpose(1, 2)
            .reshape(batch_size, self.patch_count, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

    def _route_metrics(
        self,
        affinity: Tensor,
        route_indices: Tensor,
        *,
        local_context: Tensor,
        patch_output: Tensor,
    ) -> Dict[str, Tensor]:
        selected = torch.gather(affinity, dim=-1, index=route_indices)
        if self.topk < self.region_count:
            selected_mask = torch.zeros_like(affinity, dtype=torch.bool)
            selected_mask.scatter_(dim=-1, index=route_indices, value=True)
            excluded = affinity.masked_fill(selected_mask, float("-inf"))
            selected_excluded_margin = selected.min(dim=-1).values - excluded.max(
                dim=-1
            ).values
        else:
            selected_excluded_margin = affinity.new_zeros(affinity.shape[:2])

        coordinates = self.region_coordinates.to(
            device=route_indices.device,
            dtype=torch.float32,
        )
        query_coordinates = coordinates.view(1, self.region_count, 1, 2)
        selected_coordinates = coordinates[route_indices]
        route_distance = (selected_coordinates - query_coordinates).square().sum(
            dim=-1
        ).sqrt()
        nonlocal_fraction = route_distance.gt(2.0**0.5).float().mean(dim=(1, 2))

        sorted_routes = route_indices.sort(dim=-1).values
        distinct_route_sets = []
        pairwise_jaccard = []
        for batch_index in range(int(route_indices.size(0))):
            rows = sorted_routes[batch_index]
            distinct_route_sets.append(torch.unique(rows, dim=0).size(0))
            equal = rows[:, None, :, None].eq(rows[None, :, None, :])
            intersection = equal.any(dim=-1).sum(dim=-1).float()
            union = float(2 * self.topk) - intersection
            upper = torch.triu(
                torch.ones(
                    self.region_count,
                    self.region_count,
                    dtype=torch.bool,
                    device=rows.device,
                ),
                diagonal=1,
            )
            pairwise_jaccard.append((intersection / union.clamp_min(1.0))[upper].mean())

        context_norm = local_context.float().norm(dim=(-2, -1))
        output_norm = patch_output.float().norm(dim=(-2, -1)).clamp_min(1e-12)
        return {
            "selected_affinity_margin": selected_excluded_margin.detach(),
            "selected_affinity_margin_mean": selected_excluded_margin.float()
            .mean()
            .detach(),
            "route_distance_mean": route_distance.mean().detach(),
            "nonlocal_route_fraction": nonlocal_fraction.mean().detach(),
            "distinct_route_sets_mean": torch.as_tensor(
                distinct_route_sets,
                device=affinity.device,
                dtype=torch.float32,
            )
            .mean()
            .detach(),
            "pairwise_route_jaccard_mean": torch.stack(pairwise_jaccard).mean().detach(),
            "local_context_norm_ratio": (context_norm / output_norm).mean().detach(),
        }

    def forward(
        self,
        inputs: Tensor,
        return_attention: bool = False,
        *,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_count: int = 0,
        patch_indices: Optional[Tensor] = None,
        collect_trace: Optional[bool] = None,
    ):
        prefix_count = self._validate_dense_grid(
            inputs,
            grid_size=grid_size,
            prefix_count=prefix_count,
            patch_indices=patch_indices,
        )
        batch_size, token_count, _ = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        collect_trace = bool(return_attention) if collect_trace is None else bool(
            collect_trace
        )
        if collect_trace and not return_attention:
            raise ValueError("BRA trace collection requires return_attention=True.")

        if prefix_count:
            prefix_logits = (
                query[:, :, :prefix_count] @ key.transpose(-2, -1)
            ) * self.scale
            prefix_attention = self.attention_dropout(prefix_logits.softmax(dim=-1))
            prefix_output = prefix_attention @ value
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

        query_patch = query[:, :, prefix_count:]
        key_patch = key[:, :, prefix_count:]
        value_patch = value[:, :, prefix_count:]
        affinity, route_indices = self._routing_graph(query_patch, key_patch)
        if self.topk == self.region_count:
            patch_output, dense_patch_attention = self._full_patch_attention(
                query_patch,
                key,
                value,
            )
            native_attention = dense_patch_attention
        else:
            patch_output, native_attention, dense_patch_attention = (
                self._sparse_patch_attention(
                    query_patch,
                    key,
                    value,
                    prefix_count=prefix_count,
                    route_indices=route_indices,
                    return_dense_attention=return_attention,
                )
            )

        local_context = self._local_context(value_patch)
        patch_output = patch_output + local_context
        output = torch.cat((prefix_output, patch_output), dim=2)
        output = output.transpose(1, 2).reshape(batch_size, token_count, self.dim)
        output = self.projection_dropout(self.proj(output))

        if not return_attention:
            if not torch.jit.is_tracing() and not torch.jit.is_scripting():
                self._last_trace = {"route_indices": route_indices.detach()}
            return output

        if dense_patch_attention is None:
            raise RuntimeError("BRA dense attention was not constructed.")
        dense_attention = torch.cat((prefix_attention, dense_patch_attention), dim=2)
        if collect_trace:
            metrics = self._route_metrics(
                affinity,
                route_indices,
                local_context=local_context,
                patch_output=patch_output,
            )
            self._last_trace = {
                "route_indices": route_indices.detach(),
                "region_affinity": affinity.detach(),
                "selected_patch_indices": self._selected_patch_indices(
                    route_indices
                ).detach(),
                "native_sparse_attention": native_attention.detach(),
                "dense_attention": dense_attention.detach(),
                "topk": torch.tensor(float(self.topk), device=inputs.device),
                "region_count": torch.tensor(
                    float(self.region_count),
                    device=inputs.device,
                ),
                "patch_count": torch.tensor(
                    float(self.patch_count), device=inputs.device
                ),
                **metrics,
            }
        elif not torch.jit.is_tracing() and not torch.jit.is_scripting():
            self._last_trace = {"route_indices": route_indices.detach()}
        if return_attention:
            return output, dense_attention
        raise AssertionError("Unreachable BRA return path.")

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

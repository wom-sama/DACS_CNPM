from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _DeterministicLinear(nn.Module):
    """Linear projection whose construction never consumes the global RNG."""

    def __init__(self, weight: Tensor, *, bias: bool) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have shape [out_features, in_features].")
        self.weight = nn.Parameter(weight.clone())
        self.bias = (
            nn.Parameter(weight.new_zeros(int(weight.size(0)))) if bool(bias) else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight, self.bias)


class MaxRelativeDynamicGraphMixer(nn.Module):
    """Patch-only ViG-style dynamic kNN max-relative graph residual."""

    def __init__(
        self,
        dim: int,
        *,
        bottleneck_dim: int = 64,
        k: int = 9,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.k = int(k)
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.bottleneck_dim <= 0 or self.dim % self.bottleneck_dim != 0:
            raise ValueError("bottleneck_dim must be positive and divide dim.")
        if self.k <= 0:
            raise ValueError("k must be positive.")

        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        group_width = self.dim // self.bottleneck_dim
        input_weight = torch.zeros(self.bottleneck_dim, self.dim)
        group_scale = 1.0 / math.sqrt(float(group_width))
        for output_index in range(self.bottleneck_dim):
            start = output_index * group_width
            input_weight[output_index, start : start + group_width] = group_scale
        self.input_projection = _DeterministicLinear(input_weight, bias=False)
        self.output_projection = _DeterministicLinear(
            torch.zeros(self.dim, self.bottleneck_dim * 2),
            bias=True,
        )
        self.activation = nn.GELU()
        self._last_trace: Dict[str, Tensor] = {}

    def project_nodes(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 3 or int(patch_tokens.size(-1)) != self.dim:
            raise ValueError(
                "patch_tokens must have shape [batch, patch_tokens, dim]."
            )
        if int(patch_tokens.size(1)) <= 0:
            raise ValueError("Dynamic graph mixing requires at least one patch token.")
        return self.input_projection(self.norm(patch_tokens))

    def knn_indices(self, projected_nodes: Tensor) -> Tuple[Tensor, Tensor]:
        if projected_nodes.ndim != 3:
            raise ValueError("projected_nodes must have shape [batch, nodes, channels].")
        token_count = int(projected_nodes.size(1))
        if token_count <= 0:
            raise ValueError("Dynamic graph mixing requires at least one node.")
        neighbor_count = min(self.k, token_count)
        with torch.no_grad():
            normalized = F.normalize(projected_nodes.detach().float(), p=2.0, dim=-1)
            squared_norm = normalized.square().sum(dim=-1, keepdim=True)
            distance = (
                squared_norm
                - 2.0 * torch.matmul(normalized, normalized.transpose(1, 2))
                + squared_norm.transpose(1, 2)
            ).clamp_min(0.0)
            neighbor_distance, indices = torch.topk(
                -distance,
                k=neighbor_count,
                dim=-1,
                largest=True,
                sorted=True,
            )
        return indices, -neighbor_distance

    @staticmethod
    def gather_neighbors(projected_nodes: Tensor, indices: Tensor) -> Tensor:
        if projected_nodes.ndim != 3 or indices.ndim != 3:
            raise ValueError("Expected nodes [B,N,D] and indices [B,N,K].")
        batch_size, token_count, channel_count = projected_nodes.shape
        if tuple(indices.shape[:2]) != (batch_size, token_count):
            raise ValueError("Neighbor indices must match the node batch and token axes.")
        flat_indices = indices.reshape(batch_size, -1)
        gathered = projected_nodes.gather(
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, channel_count),
        )
        return gathered.reshape(
            batch_size,
            token_count,
            int(indices.size(2)),
            channel_count,
        )

    def max_relative_message(
        self,
        projected_nodes: Tensor,
        indices: Tensor,
    ) -> Tensor:
        neighbors = self.gather_neighbors(projected_nodes, indices)
        return (neighbors - projected_nodes.unsqueeze(2)).amax(dim=2)

    def _record_trace(
        self,
        *,
        patch_tokens: Tensor,
        projected_nodes: Tensor,
        residual: Tensor,
        indices: Tensor,
        neighbor_distances: Tensor,
        patch_indices: Optional[Tensor],
        grid_size: Optional[Tuple[int, int]],
    ) -> None:
        batch_size, token_count, _ = patch_tokens.shape
        neighbor_count = int(indices.size(-1))
        device = patch_tokens.device
        center_indices = torch.arange(
            token_count,
            device=device,
            dtype=indices.dtype,
        ).view(1, token_count, 1)
        nonself_count = (indices != center_indices).float().sum(dim=-1).mean()

        counts = torch.zeros(
            batch_size,
            token_count,
            device=device,
            dtype=torch.float32,
        )
        counts.scatter_add_(
            1,
            indices.reshape(batch_size, -1),
            torch.ones(
                batch_size,
                token_count * neighbor_count,
                device=device,
                dtype=torch.float32,
            ),
        )
        probability = counts / float(token_count * neighbor_count)
        entropy_denominator = math.log(float(max(2, token_count)))
        selection_entropy = -(
            probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()
        ).sum(dim=1).mean() / entropy_denominator

        nonlocal_fraction = patch_tokens.new_tensor(float("nan"), dtype=torch.float32)
        resolved_indices: Optional[Tensor] = None
        if (
            torch.is_tensor(patch_indices)
            and patch_indices.ndim == 2
            and tuple(patch_indices.shape) == (batch_size, token_count)
        ):
            resolved_indices = patch_indices.to(device=device, dtype=torch.long)
        elif grid_size is not None and int(grid_size[0]) * int(grid_size[1]) == token_count:
            resolved_indices = torch.arange(
                token_count,
                device=device,
                dtype=torch.long,
            ).view(1, token_count).expand(batch_size, -1)
        if resolved_indices is not None and grid_size is not None:
            grid_width = max(1, int(grid_size[1]))
            neighbor_patch_indices = resolved_indices.gather(
                1,
                indices.reshape(batch_size, -1),
            ).reshape(batch_size, token_count, neighbor_count)
            center_patch_indices = resolved_indices.unsqueeze(-1)
            row_distance = (
                torch.div(neighbor_patch_indices, grid_width, rounding_mode="floor")
                - torch.div(center_patch_indices, grid_width, rounding_mode="floor")
            ).abs()
            column_distance = (
                torch.remainder(neighbor_patch_indices, grid_width)
                - torch.remainder(center_patch_indices, grid_width)
            ).abs()
            nonlocal_fraction = (
                torch.maximum(row_distance, column_distance) > 1
            ).float().mean()

        denominator = patch_tokens.detach().float().norm().clamp_min(1e-12)
        self._last_trace = {
            "residual_norm_ratio": residual.detach().float().norm() / denominator,
            "projected_node_norm": projected_nodes.detach().float().norm(dim=-1).mean(),
            "neighbor_distance_mean": neighbor_distances.detach().float().mean(),
            "nonself_neighbor_count": nonself_count,
            "nonlocal_neighbor_fraction": nonlocal_fraction,
            "neighbor_selection_entropy": selection_entropy,
            "patch_count": torch.tensor(token_count, device=device, dtype=torch.long),
            "neighbor_count": torch.tensor(neighbor_count, device=device, dtype=torch.long),
        }

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        patch_indices: Optional[Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        projected_nodes = self.project_nodes(patch_tokens)
        indices, neighbor_distances = self.knn_indices(projected_nodes)
        message = self.max_relative_message(projected_nodes, indices)
        residual = self.activation(
            self.output_projection(torch.cat((projected_nodes, message), dim=-1))
        )

        if not torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                self._record_trace(
                    patch_tokens=patch_tokens,
                    projected_nodes=projected_nodes,
                    residual=residual,
                    indices=indices,
                    neighbor_distances=neighbor_distances,
                    patch_indices=patch_indices,
                    grid_size=grid_size,
                )
        if return_details:
            return residual, {
                "projected_nodes": projected_nodes,
                "neighbor_indices": indices,
                "neighbor_distances": neighbor_distances,
                "max_relative_message": message,
            }
        return residual

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

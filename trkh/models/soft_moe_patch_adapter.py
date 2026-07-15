from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _SoftMoEExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, slot: Tensor) -> Tensor:
        return self.output_projection(self.activation(self.input_projection(slot)))


class SoftMoEPatchAdapter(nn.Module):
    """Soft-MoE residual over patch tokens with one slot per expert."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int = 64,
        num_experts: int = 4,
        residual_scale: float = 0.10,
        router_scale_init: float = 10.0,
        init_seed: int = 20260715,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.residual_scale = float(residual_scale)
        self.router_scale_init = float(router_scale_init)
        self.init_seed = int(init_seed)
        if self.dim <= 0:
            raise ValueError("SoftMoEPatchAdapter dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("SoftMoEPatchAdapter hidden_dim must be positive.")
        if self.num_experts <= 1:
            raise ValueError("SoftMoEPatchAdapter requires at least two experts.")
        if self.residual_scale < 0.0:
            raise ValueError("SoftMoEPatchAdapter residual_scale must be >= 0.")
        if self.router_scale_init <= 0.0:
            raise ValueError("SoftMoEPatchAdapter router_scale_init must be positive.")

        # torch.manual_seed also changes CUDA generators, so fork all visible devices.
        cuda_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.init_seed)
            self.router_slots = nn.Parameter(torch.empty(self.dim, self.num_experts))
            self.router_scale = nn.Parameter(torch.tensor(self.router_scale_init))
            self.experts = nn.ModuleList(
                [
                    _SoftMoEExpert(self.dim, self.hidden_dim)
                    for _ in range(self.num_experts)
                ]
            )
            nn.init.trunc_normal_(
                self.router_slots,
                std=1.0 / math.sqrt(float(self.dim)),
            )
        self._last_trace: Dict[str, Tensor] = {}

    @property
    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def routing_weights(self, patch_tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if patch_tokens.ndim != 3 or int(patch_tokens.size(-1)) != self.dim:
            raise ValueError("patch_tokens must have shape [B, N, dim].")
        if int(patch_tokens.size(1)) <= 0:
            raise ValueError("Soft MoE requires at least one patch token.")
        normalized_tokens = F.normalize(
            patch_tokens.float(),
            p=2.0,
            dim=-1,
            eps=1e-6,
        )
        normalized_slots = F.normalize(
            self.router_slots.float(),
            p=2.0,
            dim=0,
            eps=1e-6,
        )
        logits = torch.einsum(
            "bnd,de->bne",
            normalized_tokens,
            normalized_slots * self.router_scale.float(),
        )
        dispatch = logits.softmax(dim=1)
        combine = logits.softmax(dim=2)
        return logits, dispatch, combine

    def mix_patches(
        self,
        patch_tokens: Tensor,
        *,
        return_details: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        logits, dispatch, combine = self.routing_weights(patch_tokens)
        slots = torch.einsum(
            "bne,bnd->bed",
            dispatch.to(dtype=patch_tokens.dtype),
            patch_tokens,
        )
        expert_outputs = torch.stack(
            [
                expert(slots[:, expert_index])
                for expert_index, expert in enumerate(self.experts)
            ],
            dim=1,
        )
        residual = torch.einsum(
            "bne,bed->bnd",
            combine.to(dtype=expert_outputs.dtype),
            expert_outputs,
        )
        updated = patch_tokens + residual.to(dtype=patch_tokens.dtype) * self.residual_scale

        if not torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                entropy = -(
                    combine.clamp_min(1e-12) * combine.clamp_min(1e-12).log()
                ).sum(dim=-1) / math.log(float(self.num_experts))
                dispatch_vectors = dispatch.transpose(1, 2)
                normalized_dispatch = F.normalize(
                    dispatch_vectors,
                    p=2.0,
                    dim=-1,
                    eps=1e-12,
                )
                dispatch_similarity = torch.matmul(
                    normalized_dispatch,
                    normalized_dispatch.transpose(1, 2),
                )
                off_diagonal = ~torch.eye(
                    self.num_experts,
                    device=patch_tokens.device,
                    dtype=torch.bool,
                ).unsqueeze(0)
                self._last_trace = {
                    "router_logits": logits.detach(),
                    "dispatch_weights": dispatch.detach(),
                    "combine_weights": combine.detach(),
                    "slots": slots.detach(),
                    "expert_outputs": expert_outputs.detach(),
                    "residual_norm": residual.detach().float().norm(dim=-1),
                    "residual_norm_ratio": (
                        residual.detach().float().norm()
                        / patch_tokens.detach().float().norm().clamp_min(1e-12)
                    ),
                    "combine_mass": combine.detach().float().mean(dim=1),
                    "combine_entropy": entropy.detach().float().mean(dim=1),
                    "dispatch_similarity_off_diagonal": (
                        dispatch_similarity.masked_select(off_diagonal).reshape(
                            int(patch_tokens.size(0)), -1
                        )
                    ),
                }
        if return_details:
            return updated, {
                "router_logits": logits,
                "dispatch_weights": dispatch,
                "combine_weights": combine,
                "slots": slots,
                "expert_outputs": expert_outputs,
                "residual": residual,
            }
        return updated

    def forward(
        self,
        tokens: Tensor,
        *,
        prefix_count: int,
        patch_indices: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if tokens.ndim != 3 or int(tokens.size(-1)) != self.dim:
            raise ValueError("tokens must have shape [B, N, dim].")
        resolved_prefix_count = int(prefix_count)
        if not 0 <= resolved_prefix_count < int(tokens.size(1)):
            raise ValueError("prefix_count must leave at least one patch token.")
        prefix_tokens = tokens[:, :resolved_prefix_count]
        result = self.mix_patches(
            tokens[:, resolved_prefix_count:],
            return_details=return_details,
        )
        if return_details:
            updated_patches, details = result
        else:
            updated_patches = result
        updated = (
            torch.cat((prefix_tokens, updated_patches), dim=1)
            if resolved_prefix_count
            else updated_patches
        )
        if not torch.onnx.is_in_onnx_export():
            patch_count = int(updated_patches.size(1))
            if patch_indices is None:
                resolved_patch_indices = torch.arange(
                    patch_count,
                    device=tokens.device,
                    dtype=torch.long,
                ).view(1, patch_count).expand(int(tokens.size(0)), -1)
            elif (
                patch_indices.ndim == 2
                and tuple(patch_indices.shape)
                == (int(tokens.size(0)), patch_count)
            ):
                resolved_patch_indices = patch_indices.to(
                    device=tokens.device,
                    dtype=torch.long,
                )
            else:
                raise ValueError("patch_indices must match the patch-token axes.")
            self._last_trace["patch_indices"] = resolved_patch_indices.detach()
        if return_details:
            return updated, details
        return updated

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)

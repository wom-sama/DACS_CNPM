from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class DeepClassPrompt(nn.Module):
    """Per-layer class prompts with a shared class-slot readout."""

    def __init__(
        self,
        *,
        depth: int,
        num_classes: int,
        embed_dim: int,
        init_seed: int = 20260715,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.init_seed = int(init_seed)
        if self.depth <= 0:
            raise ValueError("DeepClassPrompt depth must be positive.")
        if self.num_classes <= 0:
            raise ValueError("DeepClassPrompt num_classes must be positive.")
        if self.embed_dim <= 0:
            raise ValueError("DeepClassPrompt embed_dim must be positive.")

        # The optional extension must not alter the legacy model constructor RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.prompt_embeddings = nn.Parameter(
                torch.empty(self.depth, self.num_classes, self.embed_dim)
            )
            self.norm = nn.LayerNorm(self.embed_dim)
            self.classifier = nn.Linear(self.embed_dim, 1)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @property
    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def insert(
        self,
        tokens: Tensor,
        *,
        block_index: int,
        base_prefix_count: int,
    ) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("DeepClassPrompt expects tokens [B, N, D].")
        if int(tokens.size(-1)) != self.embed_dim:
            raise ValueError("DeepClassPrompt token dimension mismatch.")
        if not 0 <= int(block_index) < self.depth:
            raise IndexError("DeepClassPrompt block_index is outside configured depth.")
        prefix_count = int(base_prefix_count)
        if not 0 <= prefix_count < int(tokens.size(1)):
            raise ValueError("base_prefix_count must leave at least one patch token.")
        prompts = self.prompt_embeddings[int(block_index)].unsqueeze(0).expand(
            int(tokens.size(0)), -1, -1
        )
        prompts = prompts.to(device=tokens.device, dtype=tokens.dtype)
        return torch.cat(
            (tokens[:, :prefix_count], prompts, tokens[:, prefix_count:]),
            dim=1,
        )

    def extract_and_remove(
        self,
        tokens: Tensor,
        *,
        base_prefix_count: int,
    ) -> Tuple[Tensor, Tensor]:
        prefix_count = int(base_prefix_count)
        prompt_end = prefix_count + self.num_classes
        if tokens.ndim != 3 or int(tokens.size(1)) <= prompt_end:
            raise ValueError("Prompt-enabled sequence must retain at least one patch token.")
        prompt_tokens = tokens[:, prefix_count:prompt_end]
        base_tokens = torch.cat(
            (tokens[:, :prefix_count], tokens[:, prompt_end:]),
            dim=1,
        )
        return base_tokens, prompt_tokens

    def logits(self, prompt_tokens: Tensor) -> Tensor:
        expected = (self.num_classes, self.embed_dim)
        if prompt_tokens.ndim != 3 or tuple(prompt_tokens.shape[1:]) != expected:
            raise ValueError(
                "DeepClassPrompt logits expect [B, num_classes, embed_dim]."
            )
        return self.classifier(self.norm(prompt_tokens)).squeeze(-1)

    def sanitize_native_attention(
        self,
        attention: Tensor,
        *,
        base_prefix_count: int,
    ) -> Tensor:
        if attention.ndim != 4 or int(attention.size(-2)) != int(attention.size(-1)):
            raise ValueError("DeepClassPrompt attention must be square [B, H, N, N].")
        prefix_count = int(base_prefix_count)
        prompt_end = prefix_count + self.num_classes
        if int(attention.size(-1)) <= prompt_end:
            raise ValueError("Prompt attention must retain at least one patch token.")
        kept_rows = torch.cat(
            (attention[:, :, :prefix_count], attention[:, :, prompt_end:]),
            dim=2,
        )
        return torch.cat(
            (kept_rows[:, :, :, :prefix_count], kept_rows[:, :, :, prompt_end:]),
            dim=3,
        )

    def class_to_patch_attention(
        self,
        attention: Tensor,
        *,
        base_prefix_count: int,
        patch_indices: Tensor,
        original_patch_count: int,
    ) -> Tensor:
        if attention.ndim != 4:
            raise ValueError("DeepClassPrompt attention must be [B, H, N, N].")
        if patch_indices.ndim != 2:
            raise ValueError("patch_indices must be [B, N].")
        prefix_count = int(base_prefix_count)
        patch_start = prefix_count + self.num_classes
        patch_attention = attention[
            :, :, prefix_count:patch_start, patch_start:
        ].mean(dim=1)
        if tuple(patch_attention.shape[:1] + patch_attention.shape[2:]) != tuple(
            patch_indices.shape
        ):
            raise ValueError("Prompt attention and patch_indices disagree.")
        output = patch_attention.new_zeros(
            (int(attention.size(0)), self.num_classes, int(original_patch_count))
        )
        scatter_indices = patch_indices.to(device=attention.device, dtype=torch.long)
        scatter_indices = scatter_indices.unsqueeze(1).expand(-1, self.num_classes, -1)
        return output.scatter(2, scatter_indices, patch_attention)

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor, nn


DIRECT_MODE = "direct"
SPATIAL_MODE = "spatial"
DEPHASED_MODE = "dephased"
SUPPORTED_MODES = (DIRECT_MODE, SPATIAL_MODE, DEPHASED_MODE)

CANONICAL_EMBED_DIM = 384
CANONICAL_PREFIX_TOKENS = 5
CANONICAL_PATCH_COUNT = 256
CANONICAL_BLOCK_COUNT = 12
CANONICAL_BOTTLENECK_DIM = 8
CANONICAL_ADAPTER_SCALE = 1.0
CANONICAL_ADAPTER_SEED = 20260805
ADDED_PARAMETER_COUNT = 170_880


class DinoV3ConvPassContractError(RuntimeError):
    """Raised when the locked DINOv3/EVA token contract has drifted."""


class QuickGELU(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.sigmoid(1.702 * inputs)


class SpatialConvPassAdapter(nn.Module):
    """A deterministic ConvPass bottleneck for DINO patch and prefix tokens.

    This keeps the core ConvPass operator (down projection, dense 3x3 spatial
    convolution, up projection) but deliberately has no private dropout or
    stochastic depth.  Direct and specialist arms therefore consume identical
    backbone RNG streams.  The output projection is zero initialized, so all
    modes are exactly the native DINO function at step zero.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        bottleneck_dim: int,
        prefix_tokens: int,
        patch_count: int,
        mode: str,
        permutation: Tensor,
    ) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported ConvPass mode: {mode!r}.")
        grid_size = int(math.isqrt(int(patch_count)))
        if grid_size * grid_size != int(patch_count):
            raise ValueError("patch_count must form a square spatial grid.")
        if tuple(permutation.shape) != (int(patch_count),):
            raise ValueError("permutation shape must equal patch_count.")
        permutation = permutation.to(dtype=torch.long, device="cpu")
        if sorted(permutation.tolist()) != list(range(int(patch_count))):
            raise ValueError("permutation must contain every patch index once.")

        self.embed_dim = int(embed_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.prefix_tokens = int(prefix_tokens)
        self.patch_count = int(patch_count)
        self.grid_size = int(grid_size)
        self.mode = resolved_mode
        self.down = nn.Linear(self.embed_dim, self.bottleneck_dim)
        self.conv = nn.Conv2d(
            self.bottleneck_dim,
            self.bottleneck_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = QuickGELU()
        self.up = nn.Linear(self.bottleneck_dim, self.embed_dim)
        self.register_buffer("permutation", permutation.clone(), persistent=True)
        self.register_buffer(
            "inverse_permutation",
            torch.argsort(permutation),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            identity = torch.eye(
                self.bottleneck_dim,
                device=self.conv.weight.device,
                dtype=self.conv.weight.dtype,
            )
            self.conv.weight[:, :, 1, 1].copy_(identity)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def set_mode(self, mode: str) -> None:
        resolved = str(mode).strip().lower()
        if resolved not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported ConvPass mode: {mode!r}.")
        self.mode = resolved

    def _spatial_tokens(self, patch_tokens: Tensor) -> Tensor:
        if self.mode == DEPHASED_MODE:
            patch_tokens = patch_tokens.index_select(1, self.permutation)
        batch_size = int(patch_tokens.size(0))
        feature_map = patch_tokens.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.bottleneck_dim,
        ).permute(0, 3, 1, 2)
        feature_map = self.conv(feature_map)
        patch_tokens = feature_map.permute(0, 2, 3, 1).reshape(
            batch_size,
            self.patch_count,
            self.bottleneck_dim,
        )
        if self.mode == DEPHASED_MODE:
            patch_tokens = patch_tokens.index_select(1, self.inverse_permutation)
        return patch_tokens

    def _prefix_tokens(self, prefix_tokens: Tensor) -> Tensor:
        if self.prefix_tokens == 0:
            return prefix_tokens
        batch_size = int(prefix_tokens.size(0))
        prefix_map = prefix_tokens.reshape(
            batch_size * self.prefix_tokens,
            self.bottleneck_dim,
            1,
            1,
        )
        prefix_map = self.conv(prefix_map)
        return prefix_map.reshape(
            batch_size,
            self.prefix_tokens,
            self.bottleneck_dim,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if self.mode == DIRECT_MODE:
            return torch.zeros_like(tokens)
        if tokens.ndim != 3 or int(tokens.size(-1)) != self.embed_dim:
            raise DinoV3ConvPassContractError(
                f"Expected [B,N,{self.embed_dim}] tokens, got {tuple(tokens.shape)}."
            )
        expected_tokens = self.prefix_tokens + self.patch_count
        if not torch.jit.is_tracing() and int(tokens.size(1)) != expected_tokens:
            raise DinoV3ConvPassContractError(
                "Runtime token count changed: "
                f"observed={int(tokens.size(1))}, expected={expected_tokens}."
            )

        reduced = self.act(self.down(tokens))
        prefix = reduced[:, : self.prefix_tokens]
        patches = reduced[:, self.prefix_tokens :]
        prefix = self._prefix_tokens(prefix)
        patches = self._spatial_tokens(patches)
        mixed = torch.cat((prefix, patches), dim=1)
        return self.up(self.act(mixed))


class _ConvPassEvaBlock(nn.Module):
    """EVA block with deterministic ConvPass residuals parallel to both paths."""

    def __init__(
        self,
        block: nn.Module,
        *,
        embed_dim: int,
        bottleneck_dim: int,
        prefix_tokens: int,
        patch_count: int,
        mode: str,
        scale: float,
        permutation: Tensor,
    ) -> None:
        super().__init__()
        required = (
            "norm1",
            "attn",
            "drop_path1",
            "norm2",
            "mlp",
            "drop_path2",
        )
        missing = [name for name in required if not hasattr(block, name)]
        if missing:
            raise DinoV3ConvPassContractError(
                f"DINO block is not an EVA block; missing={missing}."
            )

        # Preserve native state-key names instead of nesting the source block.
        self.norm1 = block.norm1
        self.attn = block.attn
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.drop_path2 = block.drop_path2
        gamma_1 = getattr(block, "gamma_1", None)
        gamma_2 = getattr(block, "gamma_2", None)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.scale = float(scale)
        self.prefix_tokens = int(prefix_tokens)
        self.adapter_attn = SpatialConvPassAdapter(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            prefix_tokens=prefix_tokens,
            patch_count=patch_count,
            mode=mode,
            permutation=permutation,
        )
        self.adapter_mlp = SpatialConvPassAdapter(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            prefix_tokens=prefix_tokens,
            patch_count=patch_count,
            mode=mode,
            permutation=permutation,
        )
        self.capture_trace = False
        self.last_trace: Dict[str, Tensor] = {}

    @property
    def mode(self) -> str:
        return self.adapter_attn.mode

    def set_mode(self, mode: str) -> None:
        self.adapter_attn.set_mode(mode)
        self.adapter_mlp.set_mode(mode)

    @staticmethod
    def _ratio(residual: Tensor, reference: Tensor) -> Tensor:
        return residual.detach().float().norm(dim=-1) / reference.detach().float().norm(
            dim=-1
        ).clamp_min(1e-12)

    def forward(
        self,
        tokens: Tensor,
        rope: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        norm1_tokens = self.norm1(tokens)
        attention = self.attn(
            norm1_tokens,
            rope=rope,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
        )
        if self.gamma_1 is not None:
            attention = self.gamma_1 * attention
        attention = self.drop_path1(attention)
        adapter_attention = self.adapter_attn(norm1_tokens) * self.scale
        attention_input = tokens
        tokens = tokens + attention
        if self.mode != DIRECT_MODE:
            tokens = tokens + adapter_attention

        norm2_tokens = self.norm2(tokens)
        mlp = self.mlp(norm2_tokens)
        if self.gamma_2 is not None:
            mlp = self.gamma_2 * mlp
        mlp = self.drop_path2(mlp)
        adapter_mlp = self.adapter_mlp(norm2_tokens) * self.scale
        mlp_input = tokens
        tokens = tokens + mlp
        if self.mode != DIRECT_MODE:
            tokens = tokens + adapter_mlp

        if self.capture_trace:
            patch_slice = slice(self.prefix_tokens, None)
            self.last_trace = {
                "attention_residual_ratio": self._ratio(
                    adapter_attention[:, patch_slice],
                    attention_input[:, patch_slice],
                ),
                "mlp_residual_ratio": self._ratio(
                    adapter_mlp[:, patch_slice],
                    mlp_input[:, patch_slice],
                ),
            }
        return tokens


class DinoV3MultiDepthConvPassB21(nn.Module):
    """DINOv3-S classifier with a lightweight spatial bypass in every block.

    Unlike the retired late-fusion hybrids, B21 exposes local 3x3 evidence at
    every attention and MLP depth and lets all remaining DINO computation
    integrate it.  ``direct`` is the exact ordinary-backbone control;
    ``dephased`` keeps identical parameters/operators while destroying image
    neighbourhoods, and is reserved for the confirmatory OOF phase.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,
        mode: str = SPATIAL_MODE,
        expected_embed_dim: int = CANONICAL_EMBED_DIM,
        expected_prefix_tokens: int = CANONICAL_PREFIX_TOKENS,
        expected_patch_count: int = CANONICAL_PATCH_COUNT,
        expected_block_count: int = CANONICAL_BLOCK_COUNT,
        bottleneck_dim: int = CANONICAL_BOTTLENECK_DIM,
        adapter_scale: float = CANONICAL_ADAPTER_SCALE,
        adapter_seed: int = CANONICAL_ADAPTER_SEED,
        externally_pretrained: bool = True,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}; got {mode!r}.")
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2.")
        if int(bottleneck_dim) <= 0:
            raise ValueError("bottleneck_dim must be positive.")
        if not math.isfinite(float(adapter_scale)) or float(adapter_scale) <= 0.0:
            raise ValueError("adapter_scale must be finite and positive.")

        embed_dim = int(
            getattr(backbone, "num_features", getattr(backbone, "embed_dim", 0))
            or 0
        )
        prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 0) or 0)
        blocks = getattr(backbone, "blocks", None)
        if embed_dim != int(expected_embed_dim):
            raise DinoV3ConvPassContractError(
                f"DINO embed_dim changed: {embed_dim} != {int(expected_embed_dim)}."
            )
        if prefix_tokens != int(expected_prefix_tokens):
            raise DinoV3ConvPassContractError(
                "DINO prefix-token count changed: "
                f"{prefix_tokens} != {int(expected_prefix_tokens)}."
            )
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != int(
            expected_block_count
        ):
            raise DinoV3ConvPassContractError(
                "DINO block contract changed: "
                f"observed={len(blocks) if isinstance(blocks, nn.ModuleList) else None}, "
                f"expected={int(expected_block_count)}."
            )
        if str(getattr(backbone, "global_pool", "")).strip().lower() != "avg":
            raise DinoV3ConvPassContractError("B21 requires average patch pooling.")
        classifier = (
            backbone.get_classifier()
            if callable(getattr(backbone, "get_classifier", None))
            else getattr(backbone, "head", None)
        )
        if not isinstance(classifier, nn.Module) or int(
            getattr(classifier, "out_features", 0) or 0
        ) != int(num_classes):
            raise DinoV3ConvPassContractError("DINO classifier contract changed.")

        self.backbone = backbone
        self.mode = resolved_mode
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_prefix_tokens = prefix_tokens
        self.expected_patch_count = int(expected_patch_count)
        self.bottleneck_dim = int(bottleneck_dim)
        self.adapter_scale = float(adapter_scale)
        self.adapter_seed = int(adapter_seed)

        # Construction is reproducible but does not alter the experiment RNG.
        with torch.random.fork_rng(devices=[]):
            # Do not call torch.manual_seed here: it also mutates every CUDA
            # generator and previously broke paired-arm reproducibility.  Set
            # only the forked CPU generator used by module constructors.
            adapter_generator = torch.Generator(device="cpu")
            adapter_generator.manual_seed(self.adapter_seed)
            torch.random.set_rng_state(adapter_generator.get_state())
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.adapter_seed + 1)
            permutation = torch.randperm(self.expected_patch_count, generator=generator)
            wrapped = []
            for block in list(self.backbone.blocks):
                wrapped.append(
                    _ConvPassEvaBlock(
                        block,
                        embed_dim=self.embed_dim,
                        bottleneck_dim=self.bottleneck_dim,
                        prefix_tokens=self.num_prefix_tokens,
                        patch_count=self.expected_patch_count,
                        mode=self.mode,
                        scale=self.adapter_scale,
                        permutation=permutation,
                    )
                )
            self.backbone.blocks = nn.ModuleList(wrapped)

        self.is_timm_classifier = True
        self.is_pretrained_timm_classifier = bool(externally_pretrained)
        self.uses_timm_backbone_lr_split = True
        self.is_dinov3_multidepth_convpass_b21 = True
        classifier_names = tuple(
            name
            for name, module in self.backbone.named_modules()
            if name and module is classifier
        )
        if not classifier_names:
            raise DinoV3ConvPassContractError("Cannot identify classifier parameter path.")
        self.pretrained_classifier_parameter_prefixes = tuple(
            f"backbone.{name}." for name in classifier_names
        )
        self.pretrained_task_parameter_prefixes = tuple(
            prefix
            for block_index in range(len(self.backbone.blocks))
            for prefix in (
                f"backbone.blocks.{block_index}.adapter_attn.",
                f"backbone.blocks.{block_index}.adapter_mlp.",
            )
        )
        inherited = getattr(backbone, "pretrained_provenance", {})
        provenance = source_provenance if source_provenance is not None else inherited
        if provenance and not isinstance(provenance, Mapping):
            raise TypeError("source_provenance must be a mapping.")
        self.pretrained_provenance = {
            "schema_version": 1,
            "research_track": "pretrained",
            "architecture": type(self).__name__,
            "primary_backbone": copy.deepcopy(dict(provenance or {})),
            "external_initialization_used": bool(externally_pretrained),
            "fusion": self.fusion_provenance(),
        }
        self.set_mode(self.mode)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.backbone.blocks

    @property
    def patch_embed(self) -> nn.Module:
        return self.backbone.patch_embed

    @property
    def head(self) -> nn.Module:
        return self.get_classifier()

    def get_classifier(self) -> nn.Module:
        getter = getattr(self.backbone, "get_classifier", None)
        return getter() if callable(getter) else self.backbone.head

    def reset_classifier(self, num_classes: int) -> None:
        reset = getattr(self.backbone, "reset_classifier", None)
        if not callable(reset):
            raise DinoV3ConvPassContractError("DINO backbone cannot reset classifier.")
        reset(int(num_classes))

    def no_weight_decay(self) -> set[str]:
        inherited = getattr(self.backbone, "no_weight_decay", None)
        names = set(str(name) for name in inherited()) if callable(inherited) else set()
        return {f"backbone.{name}" for name in names}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        hook = getattr(self.backbone, "set_grad_checkpointing", None)
        if not callable(hook):
            raise DinoV3ConvPassContractError("DINO backbone has no checkpoint hook.")
        hook(enable=bool(enable))

    def set_mode(self, mode: str) -> None:
        resolved = str(mode).strip().lower()
        if resolved not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported ConvPass mode: {mode!r}.")
        self.mode = resolved
        for block in self.backbone.blocks:
            block.set_mode(resolved)
        enabled = resolved != DIRECT_MODE
        for name, parameter in self.named_parameters():
            if ".adapter_attn." in name or ".adapter_mlp." in name:
                parameter.requires_grad_(enabled)

    def forward_features(
        self,
        images: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        return self.backbone.forward_features(
            images,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
        )

    def forward_head(self, tokens: Tensor, pre_logits: bool = False) -> Tensor:
        return self.backbone.forward_head(tokens, pre_logits=bool(pre_logits))

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(images), pre_logits=False)

    def added_parameter_count(self, *, trainable_only: bool = False) -> int:
        return int(
            sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if (".adapter_attn." in name or ".adapter_mlp." in name)
                and (not trainable_only or parameter.requires_grad)
            )
        )

    def fusion_provenance(self) -> Dict[str, Any]:
        return {
            "protocol_id": "TRKH_B21_DINOV3_MULTIDEPTH_CONVPASS_20260805",
            "mode": self.mode,
            "placement": "parallel_to_mhsa_and_mlp_in_all_12_eva_blocks",
            "bottleneck_dim": int(self.bottleneck_dim),
            "adapter_scale": float(self.adapter_scale),
            "adapter_seed": int(self.adapter_seed),
            "private_stochastic_operations": False,
            "up_projection_initialization": "zeros_exact_native_identity",
            "conv_initialization": "center_identity",
            "dephased_control": "fixed_patch_permutation_then_inverse_after_conv",
            "observed_added_parameter_count": self.added_parameter_count(),
        }

    def forward_with_adapter_trace(self, images: Tensor) -> tuple[Tensor, Dict[str, Any]]:
        for block in self.backbone.blocks:
            block.capture_trace = True
            block.last_trace = {}
        try:
            logits = self(images)
            attention = []
            mlp = []
            for block in self.backbone.blocks:
                attention.append(block.last_trace["attention_residual_ratio"].reshape(-1))
                mlp.append(block.last_trace["mlp_residual_ratio"].reshape(-1))
            attention_values = torch.cat(attention)
            mlp_values = torch.cat(mlp)
            all_values = torch.cat((attention_values, mlp_values))
            trace = {
                "attention_residual_ratio": attention_values,
                "mlp_residual_ratio": mlp_values,
                "all_residual_ratio": all_values,
            }
            return logits, trace
        finally:
            for block in self.backbone.blocks:
                block.capture_trace = False


def expected_added_parameter_count(
    *,
    block_count: int,
    embed_dim: int,
    bottleneck_dim: int,
) -> int:
    per_adapter = (
        int(embed_dim) * int(bottleneck_dim)
        + int(bottleneck_dim)
        + 9 * int(bottleneck_dim) * int(bottleneck_dim)
        + int(bottleneck_dim)
        + int(bottleneck_dim) * int(embed_dim)
        + int(embed_dim)
    )
    return int(block_count) * 2 * per_adapter


if ADDED_PARAMETER_COUNT != expected_added_parameter_count(
    block_count=CANONICAL_BLOCK_COUNT,
    embed_dim=CANONICAL_EMBED_DIM,
    bottleneck_dim=CANONICAL_BOTTLENECK_DIM,
):
    raise AssertionError("Locked B21 adapter parameter count is inconsistent.")

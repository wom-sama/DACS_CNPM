from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


LOCAL_SURFACE_MODE = "local_surface"
GENERIC_TOKEN_ADAPTER_MODE = "generic_token_adapter"
SUPPORTED_FUSION_MODES = (LOCAL_SURFACE_MODE, GENERIC_TOKEN_ADAPTER_MODE)

CANONICAL_EMBED_DIM = 384
CANONICAL_PREFIX_TOKENS = 5
CANONICAL_PATCH_COUNT = 256
CANONICAL_PATCH_SIZE = (16, 16)
CANONICAL_BLOCK_COUNT = 12
CANONICAL_FUSION_BLOCK_INDEX = 11
CANONICAL_SURFACE_CHANNELS = (16, 24, 32, 64)
CANONICAL_GENERIC_HIDDEN_DIM = 37

# These counts deliberately exclude the DINOv3 backbone and its classifier.
# They are locked so the generic token control is capacity matched to within 1%.
LOCAL_SURFACE_ADDED_PARAMETER_COUNT = 29_825
GENERIC_TOKEN_ADAPTER_ADDED_PARAMETER_COUNT = 29_990


class DinoV3SurfaceHybridContractError(RuntimeError):
    """Raised when the primary backbone or patch grid violates the V2 contract."""


def _parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def _validate_channel_triplet(name: str, values: Sequence[float]) -> Tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if len(resolved) != 3 or not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} must contain three finite values.")
    return resolved


class _ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels at each spatial cell; no batch/grid leakage."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels), eps=float(eps))

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs.permute(0, 2, 3, 1)
        outputs = self.norm(outputs)
        return outputs.permute(0, 3, 1, 2)


class _ExactPatchConvDownsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            _ChannelLayerNorm2d(int(out_channels)),
            nn.Hardswish(inplace=False),
        )


class _DepthwiseSeparableDownsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(in_channels),
                kernel_size=2,
                stride=2,
                padding=0,
                groups=int(in_channels),
                bias=False,
            ),
            _ChannelLayerNorm2d(int(in_channels)),
            nn.Hardswish(inplace=False),
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=1,
                bias=False,
            ),
            _ChannelLayerNorm2d(int(out_channels)),
            nn.Hardswish(inplace=False),
        )


class _LocalSurfaceEncoder(nn.Module):
    """A mobile stride-16 encoder whose cells exactly match DINO patches.

    Four non-overlapping ``2x2/2`` stages give each output cell a receptive
    field of exactly one ``16x16`` input patch.  No padded convolution can pull
    evidence from the preceding DINO patch before fusion.
    """

    output_dim = CANONICAL_SURFACE_CHANNELS[-1]

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = CANONICAL_SURFACE_CHANNELS
        self.stem = _ExactPatchConvDownsample(6, c1)
        self.downsample_1 = _DepthwiseSeparableDownsample(c1, c2)
        self.downsample_2 = _DepthwiseSeparableDownsample(c2, c3)
        self.downsample_3 = _DepthwiseSeparableDownsample(c3, c4)

    def forward_with_shapes(self, inputs: Tensor) -> Tuple[Tensor, Dict[str, list[int]]]:
        stem = self.stem(inputs)
        stage_2 = self.downsample_1(stem)
        stage_3 = self.downsample_2(stage_2)
        stage_4 = self.downsample_3(stage_3)
        shapes = {
            "surface_descriptor": [int(value) for value in inputs.shape],
            "surface_stage_1": [int(value) for value in stem.shape],
            "surface_stage_2": [int(value) for value in stage_2.shape],
            "surface_stage_3": [int(value) for value in stage_3.shape],
            "surface_stage_4": [int(value) for value in stage_4.shape],
        }
        return stage_4, shapes

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.stem(inputs)
        outputs = self.downsample_1(outputs)
        outputs = self.downsample_2(outputs)
        return self.downsample_3(outputs)


class _GenericTokenAdapter(nn.Module):
    """Capacity-matched DINO-only patch adapter for the causal control."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(embed_dim))
        self.fc1 = nn.Linear(int(embed_dim), CANONICAL_GENERIC_HIDDEN_DIM)
        self.act = nn.Hardswish(inplace=False)
        self.fc2 = nn.Linear(CANONICAL_GENERIC_HIDDEN_DIM, int(embed_dim))

    def forward_with_context(self, patch_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        context = self.norm(patch_tokens)
        residual = self.fc2(self.act(self.fc1(context)))
        return residual, context

    def forward(self, patch_tokens: Tensor) -> Tensor:
        residual, _ = self.forward_with_context(patch_tokens)
        return residual


class DinoV3SurfacePatchHybridV2(nn.Module):
    """DINOv3-primary classifier with a bounded, patch-aligned local residual.

    The pretrained DINOv3 classifier remains the primary path. In
    ``local_surface`` mode a lightweight exact-bin stride-16 CNN consumes
    denormalized RGB and its exact-patch mean, producing one feature per DINO
    patch.
    A semantic gate conditions each local residual on the corresponding
    *detached* DINO patch before the final pretrained DINO block.  That final
    attention/MLP block integrates the added local evidence before pooling.
    Detaching only the gate context prevents the optional branch from adding a
    second Jacobian into the primary backbone, while the ordinary DINO path is
    still fine-tuned normally.

    The gate is bounded by ``max_gate_scale`` and starts at a strictly positive
    value.  The final branch projection is zero-initialized, so the configured
    model is exactly the native DINO classifier at step zero while the first
    classification backward still trains that projection.  Upstream specialist
    layers become active after the projection's first update; this avoids both
    random residual injection and the zero-gate starvation of the retired
    global hybrid. ``generic_token_adapter`` supplies a nearly exact
    parameter-matched DINO-only control without access to new RGB evidence.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,
        mode: str = LOCAL_SURFACE_MODE,
        input_mean: Sequence[float] = (0.485, 0.456, 0.406),
        input_std: Sequence[float] = (0.229, 0.224, 0.225),
        expected_embed_dim: int = CANONICAL_EMBED_DIM,
        expected_prefix_tokens: int = CANONICAL_PREFIX_TOKENS,
        expected_patch_count: int = CANONICAL_PATCH_COUNT,
        initial_gate_scale: float = 0.05,
        max_gate_scale: float = 0.25,
        externally_pretrained: bool = True,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module.")
        resolved_mode = str(mode or "").strip().lower()
        if resolved_mode not in SUPPORTED_FUSION_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_FUSION_MODES!r}; got {mode!r}."
            )
        resolved_num_classes = int(num_classes)
        resolved_embed_dim = int(expected_embed_dim)
        resolved_prefix_tokens = int(expected_prefix_tokens)
        resolved_patch_count = int(expected_patch_count)
        if resolved_num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if resolved_embed_dim <= 0:
            raise ValueError("expected_embed_dim must be positive.")
        if resolved_prefix_tokens <= 0:
            raise ValueError("expected_prefix_tokens must be positive.")
        if resolved_patch_count <= 0:
            raise ValueError("expected_patch_count must be positive.")
        resolved_max_scale = float(max_gate_scale)
        resolved_initial_scale = float(initial_gate_scale)
        if not math.isfinite(resolved_max_scale) or resolved_max_scale <= 0.0:
            raise ValueError("max_gate_scale must be finite and positive.")
        if not (
            math.isfinite(resolved_initial_scale)
            and 0.0 < resolved_initial_scale < resolved_max_scale
        ):
            raise ValueError(
                "initial_gate_scale must be finite and strictly inside "
                "(0, max_gate_scale)."
            )

        observed_embed_dim = int(
            getattr(backbone, "num_features", getattr(backbone, "embed_dim", 0))
            or 0
        )
        observed_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 0) or 0)
        observed_global_pool = str(getattr(backbone, "global_pool", "")).strip().lower()
        if observed_embed_dim != resolved_embed_dim:
            raise DinoV3SurfaceHybridContractError(
                "DINO embedding dimension changed: "
                f"observed={observed_embed_dim}, expected={resolved_embed_dim}."
            )
        if observed_prefix_tokens != resolved_prefix_tokens:
            raise DinoV3SurfaceHybridContractError(
                "DINO prefix-token count changed: "
                f"observed={observed_prefix_tokens}, expected={resolved_prefix_tokens}."
            )
        if observed_global_pool != "avg":
            raise DinoV3SurfaceHybridContractError(
                "Hybrid V2 requires DINO average patch pooling; "
                f"observed global_pool={observed_global_pool!r}."
            )
        if not callable(getattr(backbone, "forward_features", None)):
            raise DinoV3SurfaceHybridContractError(
                "DINO backbone must implement forward_features."
            )
        if not callable(getattr(backbone, "forward_head", None)):
            raise DinoV3SurfaceHybridContractError(
                "DINO backbone must implement forward_head."
            )
        required_manual_path = {
            "patch_embed": callable(getattr(backbone, "patch_embed", None)),
            "_pos_embed": callable(getattr(backbone, "_pos_embed", None)),
            "norm_pre": isinstance(getattr(backbone, "norm_pre", None), nn.Module),
            "blocks": isinstance(getattr(backbone, "blocks", None), nn.ModuleList),
            "norm": isinstance(getattr(backbone, "norm", None), nn.Module),
        }
        missing_manual_path = sorted(
            name for name, available in required_manual_path.items() if not available
        )
        if missing_manual_path:
            raise DinoV3SurfaceHybridContractError(
                "DINO backbone cannot expose the locked pre-final-block fusion path: "
                f"missing={missing_manual_path}."
            )
        observed_block_count = len(backbone.blocks)
        if observed_block_count != CANONICAL_BLOCK_COUNT:
            raise DinoV3SurfaceHybridContractError(
                "DINO block count changed: "
                f"observed={observed_block_count}, expected={CANONICAL_BLOCK_COUNT}."
            )
        observed_patch_size_raw = getattr(backbone.patch_embed, "patch_size", None)
        if isinstance(observed_patch_size_raw, int):
            observed_patch_size = (
                int(observed_patch_size_raw),
                int(observed_patch_size_raw),
            )
        elif isinstance(observed_patch_size_raw, Sequence):
            observed_patch_size = tuple(int(value) for value in observed_patch_size_raw)
        else:
            observed_patch_size = ()
        if observed_patch_size != CANONICAL_PATCH_SIZE:
            raise DinoV3SurfaceHybridContractError(
                "DINO patch size changed: "
                f"observed={observed_patch_size}, expected={CANONICAL_PATCH_SIZE}."
            )
        rope_mixed = getattr(backbone, "rope_mixed", False)
        if not isinstance(rope_mixed, bool):
            raise DinoV3SurfaceHybridContractError(
                "DINO rope_mixed contract must be boolean."
            )
        classifier = (
            backbone.get_classifier()
            if callable(getattr(backbone, "get_classifier", None))
            else getattr(backbone, "head", None)
        )
        if not isinstance(classifier, nn.Module):
            raise DinoV3SurfaceHybridContractError(
                "DINO backbone does not expose a classifier module."
            )
        classifier_out_features = int(getattr(classifier, "out_features", 0) or 0)
        if classifier_out_features != resolved_num_classes:
            raise DinoV3SurfaceHybridContractError(
                "DINO classifier class count changed: "
                f"observed={classifier_out_features}, expected={resolved_num_classes}."
            )

        classifier_module_names = tuple(
            name
            for name, module in backbone.named_modules()
            if name and module is classifier
        )
        if not classifier_module_names:
            raise DinoV3SurfaceHybridContractError(
                "Cannot identify the DINO classifier parameter prefix."
            )

        mean = _validate_channel_triplet("input_mean", input_mean)
        std = _validate_channel_triplet("input_std", input_std)
        if any(value <= 0.0 for value in std):
            raise ValueError("input_std values must be positive.")

        self.backbone = backbone
        self.mode = resolved_mode
        self.num_classes = resolved_num_classes
        self.embed_dim = resolved_embed_dim
        self.num_features = resolved_embed_dim
        self.num_prefix_tokens = resolved_prefix_tokens
        self.expected_patch_count = resolved_patch_count
        self.fusion_block_index = CANONICAL_FUSION_BLOCK_INDEX
        self.global_pool = observed_global_pool
        self.initial_gate_scale = resolved_initial_scale
        self.max_gate_scale = resolved_max_scale
        self.register_buffer(
            "input_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
        )

        # Module construction must not advance the global training RNG.  The
        # direct DINO, generic-control and local-surface arms therefore start
        # data loading/dropout from the same RNG state under the same seed.
        with torch.random.fork_rng(devices=[]):
            if self.mode == LOCAL_SURFACE_MODE:
                self.surface_encoder: nn.Module | None = _LocalSurfaceEncoder()
                self.surface_norm: nn.Module | None = nn.LayerNorm(
                    _LocalSurfaceEncoder.output_dim
                )
                self.surface_projection: nn.Module | None = nn.Linear(
                    _LocalSurfaceEncoder.output_dim,
                    self.embed_dim,
                )
                nn.init.zeros_(self.surface_projection.weight)
                nn.init.zeros_(self.surface_projection.bias)
                self.token_adapter: nn.Module | None = None
                gate_input_dim = _LocalSurfaceEncoder.output_dim + self.embed_dim
                self.pretrained_task_parameter_prefixes = (
                    "surface_encoder.",
                    "surface_norm.",
                    "surface_projection.",
                    "fusion_gate.",
                )
            else:
                self.surface_encoder = None
                self.surface_norm = None
                self.surface_projection = None
                self.token_adapter = _GenericTokenAdapter(self.embed_dim)
                nn.init.zeros_(self.token_adapter.fc2.weight)
                nn.init.zeros_(self.token_adapter.fc2.bias)
                gate_input_dim = self.embed_dim
                self.pretrained_task_parameter_prefixes = (
                    "token_adapter.",
                    "fusion_gate.",
                )

            self.fusion_gate = nn.Linear(gate_input_dim, 1)
            nn.init.zeros_(self.fusion_gate.weight)
            initial_ratio = self.initial_gate_scale / self.max_gate_scale
            initial_raw_gate = math.log(initial_ratio / (1.0 - initial_ratio))
            nn.init.constant_(self.fusion_gate.bias, initial_raw_gate)

        self.is_timm_classifier = True
        self.is_pretrained_timm_classifier = bool(externally_pretrained)
        # Keep the optimizer contract identical for the pretrained arms and
        # their explicit random-init causal control.  This flag controls only
        # LR grouping; it does not claim external initialization provenance.
        self.uses_timm_backbone_lr_split = True
        self.is_pretrained_surface_hybrid = True
        self.is_pretrained_surface_patch_hybrid_v2 = True
        self.pretrained_classifier_parameter_prefixes = tuple(
            f"backbone.{name}." for name in classifier_module_names
        )
        self.research_track = "pretrained"

        inherited_provenance = getattr(backbone, "pretrained_provenance", {})
        selected_provenance = (
            source_provenance
            if source_provenance is not None
            else inherited_provenance
        )
        if selected_provenance and not isinstance(selected_provenance, Mapping):
            raise TypeError("source_provenance must be a mapping when provided.")
        self.pretrained_provenance = {
            "schema_version": 2,
            "research_track": "pretrained",
            "architecture": type(self).__name__,
            "primary_backbone": copy.deepcopy(dict(selected_provenance or {})),
            "external_initialization_used": bool(externally_pretrained),
            "fusion": self.fusion_provenance(),
            "attention_audit": {
                "native_attention_source": "backbone_dinov3_final_block_after_fusion",
                "surface_specialist_native_attention": False,
                "surface_specialist_xai": (
                    "use specialist Grad-CAM plus full-model perturbation"
                ),
            },
        }

    @property
    def blocks(self):
        return self.backbone.blocks

    @property
    def patch_embed(self):
        return self.backbone.patch_embed

    @property
    def head(self):
        return self.get_classifier()

    def get_classifier(self) -> nn.Module:
        getter = getattr(self.backbone, "get_classifier", None)
        return getter() if callable(getter) else self.backbone.head

    def no_weight_decay(self) -> set[str]:
        inherited = getattr(self.backbone, "no_weight_decay", None)
        names = set(str(name) for name in inherited()) if callable(inherited) else set()
        prefixed = {f"backbone.{name}" for name in names}
        prefixed.update({"fusion_gate.weight", "fusion_gate.bias"})
        return prefixed

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        hook = getattr(self.backbone, "set_grad_checkpointing", None)
        if not callable(hook):
            raise DinoV3SurfaceHybridContractError(
                "DINO backbone has no gradient-checkpointing hook."
            )
        hook(enable=bool(enable))

    def _validate_images(self, images: Tensor) -> None:
        if torch.jit.is_tracing():
            return
        if images.ndim != 4 or int(images.size(1)) != 3:
            raise DinoV3SurfaceHybridContractError(
                f"Expected images [B,3,H,W], got {tuple(images.shape)}."
            )
        if int(images.size(-2)) % 16 != 0 or int(images.size(-1)) % 16 != 0:
            raise DinoV3SurfaceHybridContractError(
                "Hybrid V2 requires image height and width divisible by 16."
            )

    def _validate_dino_tokens(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        if not torch.jit.is_tracing() and (
            not torch.is_tensor(tokens) or tokens.ndim != 3
        ):
            raise DinoV3SurfaceHybridContractError(
                "DINO forward_features must return one [B,N,D] tensor."
            )
        if not torch.jit.is_tracing() and int(tokens.size(-1)) != self.embed_dim:
            raise DinoV3SurfaceHybridContractError(
                "Runtime DINO token dimension changed: "
                f"observed={int(tokens.size(-1))}, expected={self.embed_dim}."
            )
        patch_count = (
            self.expected_patch_count
            if torch.jit.is_tracing()
            else int(tokens.size(1)) - self.num_prefix_tokens
        )
        if patch_count != self.expected_patch_count:
            raise DinoV3SurfaceHybridContractError(
                "Runtime DINO patch count changed: "
                f"observed={patch_count}, expected={self.expected_patch_count}."
            )
        return (
            tokens[:, : self.num_prefix_tokens],
            tokens[:, self.num_prefix_tokens :],
        )

    def surface_descriptor(self, images: Tensor) -> Tensor:
        """Return RGB plus its exact-bin mean; never cross DINO patch borders."""

        self._validate_images(images)
        mean = self.input_mean.to(device=images.device, dtype=images.dtype)
        std = self.input_std.to(device=images.device, dtype=images.dtype)
        rgb = (images * std + mean).clamp(min=0.0, max=1.0)
        patch_mean = F.avg_pool2d(
            rgb,
            kernel_size=16,
            stride=16,
        )
        patch_mean = F.interpolate(
            patch_mean,
            scale_factor=16.0,
            mode="nearest",
        )
        return torch.cat((rgb, patch_mean), dim=1)

    def _normalize_delta(
        self,
        residual: Tensor,
        detached_patch_tokens: Tensor,
    ) -> Tensor:
        residual_float = residual.float()
        # Smooth norm saturation: ||r / sqrt(1 + ||r||^2)|| is always < 1,
        # but its derivative at r=0 is the identity.  This is what allows a
        # zero-initialized output projection to preserve native DINO logits at
        # step zero without starving that projection's first backward pass.
        bounded_delta = residual_float / torch.sqrt(
            1.0 + residual_float.square().sum(dim=-1, keepdim=True)
        )
        primary_norm = detached_patch_tokens.float().norm(
            dim=-1,
            keepdim=True,
        )
        return (bounded_delta * primary_norm).to(dtype=residual.dtype)

    def _local_surface_residual(
        self,
        images: Tensor,
        detached_patch_tokens: Tensor,
        *,
        return_trace: bool,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        if (
            self.surface_encoder is None
            or self.surface_norm is None
            or self.surface_projection is None
        ):
            raise RuntimeError("local_surface modules are not initialized.")
        descriptor = self.surface_descriptor(images)
        if return_trace:
            surface_map, stage_shapes = self.surface_encoder.forward_with_shapes(
                descriptor
            )
        else:
            surface_map = self.surface_encoder(descriptor)
            stage_shapes = {}
        surface_tokens = surface_map.flatten(2).transpose(1, 2)
        if (
            not torch.jit.is_tracing()
            and int(surface_tokens.size(1)) != self.expected_patch_count
        ):
            raise DinoV3SurfaceHybridContractError(
                "Surface/DINO patch grids are misaligned: "
                f"surface={int(surface_tokens.size(1))}, "
                f"dino={self.expected_patch_count}."
            )
        surface_tokens = self.surface_norm(surface_tokens)
        raw_delta = self.surface_projection(surface_tokens)
        dino_gate_context = F.layer_norm(
            detached_patch_tokens.float(),
            normalized_shape=(self.embed_dim,),
            eps=1e-6,
        ).to(dtype=surface_tokens.dtype)
        gate_context = torch.cat((surface_tokens, dino_gate_context), dim=-1)
        trace: Dict[str, Any] = {
            **stage_shapes,
            "surface_descriptor_tensor": descriptor,
            "surface_feature_map": surface_map,
            "surface_tokens": surface_tokens,
        }
        return raw_delta, gate_context, trace

    def _generic_token_residual(
        self,
        detached_patch_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        if not isinstance(self.token_adapter, _GenericTokenAdapter):
            raise RuntimeError("generic_token_adapter modules are not initialized.")
        raw_delta, gate_context = self.token_adapter.forward_with_context(
            detached_patch_tokens
        )
        return raw_delta, gate_context, {
            "generic_token_context": gate_context,
        }

    def _build_gated_patch_residual(
        self,
        images: Tensor,
        pre_fusion_tokens: Tensor,
        patch_tokens: Tensor,
        detached_patch_tokens: Tensor,
        *,
        return_trace: bool,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Build the V2 residual behind a protected V3-compatible hook."""

        del pre_fusion_tokens

        if self.mode == LOCAL_SURFACE_MODE:
            raw_delta, gate_context, mode_trace = self._local_surface_residual(
                images,
                detached_patch_tokens,
                return_trace=return_trace,
            )
        else:
            raw_delta, gate_context, mode_trace = self._generic_token_residual(
                detached_patch_tokens
            )

        normalized_delta = self._normalize_delta(
            raw_delta,
            detached_patch_tokens,
        )
        raw_gate = self.fusion_gate(gate_context)
        gate = (
            torch.sigmoid(raw_gate.float()) * self.max_gate_scale
        ).to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        typed_delta = normalized_delta.to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        gated_residual = gate * typed_delta
        return gated_residual, {
            **mode_trace,
            "raw_surface_delta": raw_delta,
            "normalized_surface_delta": normalized_delta,
            "raw_gate": raw_gate,
            "gate": gate,
        }

    def _run_backbone_block(
        self,
        block_index: int,
        tokens: Tensor,
        rot_pos_embed: Tensor | None,
        *,
        attn_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        block = self.backbone.blocks[int(block_index)]
        block_rope = (
            rot_pos_embed[int(block_index)]
            if bool(getattr(self.backbone, "rope_mixed", False))
            and rot_pos_embed is not None
            else rot_pos_embed
        )
        if bool(getattr(self.backbone, "grad_checkpointing", False)) and not torch.jit.is_scripting():
            return checkpoint(
                block,
                tokens,
                rope=block_rope,
                attn_mask=attn_mask,
                is_causal=bool(is_causal),
                use_reentrant=False,
            )
        return block(
            tokens,
            rope=block_rope,
            attn_mask=attn_mask,
            is_causal=bool(is_causal),
        )

    def _forward_to_fusion_point(
        self,
        images: Tensor,
        *,
        attn_mask: Tensor | None,
        is_causal: bool,
    ) -> Tuple[Tensor, Tensor | None]:
        tokens = self.backbone.patch_embed(images)
        positioned = self.backbone._pos_embed(tokens)
        if not isinstance(positioned, tuple) or len(positioned) != 2:
            raise DinoV3SurfaceHybridContractError(
                "DINO _pos_embed must return (tokens, rotary_position_embedding)."
            )
        tokens, rot_pos_embed = positioned
        tokens = self.backbone.norm_pre(tokens)
        for block_index in range(self.fusion_block_index):
            tokens = self._run_backbone_block(
                block_index,
                tokens,
                rot_pos_embed,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        return tokens, rot_pos_embed

    def _forward_features_impl(
        self,
        images: Tensor,
        *,
        return_trace: bool,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        self._validate_images(images)
        pre_fusion_tokens, rot_pos_embed = self._forward_to_fusion_point(
            images,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        prefix_tokens, patch_tokens = self._validate_dino_tokens(pre_fusion_tokens)
        detached_patch_tokens = patch_tokens.detach()
        gated_residual, fusion_trace = self._build_gated_patch_residual(
            images,
            pre_fusion_tokens,
            patch_tokens,
            detached_patch_tokens,
            return_trace=return_trace,
        )
        fused_patch_tokens = patch_tokens + gated_residual
        injected_tokens = torch.cat((prefix_tokens, fused_patch_tokens), dim=1)
        post_fusion_tokens = self._run_backbone_block(
            self.fusion_block_index,
            injected_tokens,
            rot_pos_embed,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        post_fusion_tokens = self.backbone.norm(post_fusion_tokens)

        if not return_trace:
            return post_fusion_tokens, {}
        gate = fusion_trace["gate"]
        patch_norm = patch_tokens.detach().float().norm(dim=-1)
        gated_residual_norm = gated_residual.detach().float().norm(dim=-1)
        trace: Dict[str, Any] = {
            **fusion_trace,
            "mode": self.mode,
            "dino_tokens": pre_fusion_tokens,
            "dino_prefix_tokens": prefix_tokens,
            "dino_patch_tokens": patch_tokens,
            "gated_residual": gated_residual,
            "fused_patch_tokens": fused_patch_tokens,
            "injected_tokens": injected_tokens,
            "fused_tokens": post_fusion_tokens,
            "post_fusion_tokens": post_fusion_tokens,
            "dino_token_shape": [int(value) for value in pre_fusion_tokens.shape],
            "prefix_token_shape": [int(value) for value in prefix_tokens.shape],
            "patch_token_shape": [int(value) for value in patch_tokens.shape],
            "gate_shape": [int(value) for value in gate.shape],
            "gate_mean": gate.detach().float().mean(),
            "gate_min": gate.detach().float().amin(),
            "gate_max": gate.detach().float().amax(),
            "gate_std": gate.detach().float().std(unbiased=False),
            "gated_residual_norm_ratio": (
                gated_residual_norm / patch_norm.clamp_min(1e-12)
            ),
            "primary_gate_context_detached": True,
            "fusion_block_index_zero_based": int(self.fusion_block_index),
            "fusion_tail_block_count": int(
                len(self.backbone.blocks) - self.fusion_block_index
            ),
            "prefix_tokens_modified_at_injection": False,
        }
        return post_fusion_tokens, trace

    def forward_features(
        self,
        images: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        fused_tokens, _ = self._forward_features_impl(
            images,
            return_trace=False,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return fused_tokens

    def forward_features_with_fusion_trace(
        self,
        images: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        return self._forward_features_impl(images, return_trace=True)

    def forward_head(self, tokens: Tensor, pre_logits: bool = False) -> Tensor:
        return self.backbone.forward_head(tokens, pre_logits=bool(pre_logits))

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(images), pre_logits=False)

    def added_parameter_count(self, *, trainable_only: bool = False) -> int:
        return int(
            sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if not name.startswith("backbone.")
                and (not trainable_only or parameter.requires_grad)
            )
        )

    def parameter_telemetry(self) -> Dict[str, int]:
        return {
            "backbone_parameter_count": _parameter_count(self.backbone),
            "added_parameter_count": self.added_parameter_count(),
            "total_parameter_count": _parameter_count(self),
            "trainable_backbone_parameter_count": _parameter_count(
                self.backbone,
                trainable_only=True,
            ),
            "trainable_added_parameter_count": self.added_parameter_count(
                trainable_only=True
            ),
            "trainable_total_parameter_count": _parameter_count(
                self,
                trainable_only=True,
            ),
        }

    def fusion_runtime_parameter_telemetry(self) -> Dict[str, float]:
        """Small checkpoint-safe signal for detecting gate collapse or drift."""

        gate_weight = self.fusion_gate.weight.detach().float()
        gate_bias = self.fusion_gate.bias.detach().float()
        bias_only_scale = torch.sigmoid(gate_bias) * self.max_gate_scale
        return {
            "gate_weight_l2": float(gate_weight.norm().cpu().item()),
            "gate_bias": float(gate_bias.mean().cpu().item()),
            "bias_only_gate_scale": float(bias_only_scale.mean().cpu().item()),
            "max_gate_scale": float(self.max_gate_scale),
        }

    def fusion_provenance(self) -> Dict[str, Any]:
        expected_added = (
            LOCAL_SURFACE_ADDED_PARAMETER_COUNT
            if self.mode == LOCAL_SURFACE_MODE
            else GENERIC_TOKEN_ADAPTER_ADDED_PARAMETER_COUNT
        )
        return {
            "schema_version": 1,
            "mode": self.mode,
            "primary_path": (
                "dinov3_tokens_plus_bounded_patch_residual_before_final_block"
            ),
            "primary_gate_context_detached": True,
            "prefix_tokens_modified_at_injection": False,
            "prefix_tokens_can_change_in_final_block": True,
            "pooling": "backbone_avg_patch_pool",
            "fusion_point": "before_backbone_block_11_zero_based",
            "fusion_tail_block_count": 1,
            "expected_embed_dim": int(self.embed_dim),
            "expected_prefix_tokens": int(self.num_prefix_tokens),
            "expected_patch_count": int(self.expected_patch_count),
            "expected_patch_size": list(CANONICAL_PATCH_SIZE),
            "surface_channels": list(CANONICAL_SURFACE_CHANNELS),
            "surface_patch_alignment": "four_nonoverlapping_k2_s2_p0_stages",
            "surface_spatial_normalization": (
                "per_cell_channel_layer_norm_no_batch_or_grid_statistics"
            ),
            "generic_hidden_dim": int(CANONICAL_GENERIC_HIDDEN_DIM),
            "task_module_rng_isolation": "torch.random.fork_rng(devices=[])",
            "gate": {
                "parameterization": "max_gate_scale*sigmoid(linear_context)",
                "context": (
                    "concat(local_surface_token,detached_dino_patch)"
                    if self.mode == LOCAL_SURFACE_MODE
                    else "normalized_detached_dino_patch"
                ),
                "weight_initialization": "zero",
                "initial_scale": float(self.initial_gate_scale),
                "max_scale": float(self.max_gate_scale),
                "strictly_positive_at_initialization": True,
            },
            "delta_normalization": (
                "smooth_r_over_sqrt_1_plus_l2_squared_times_"
                "detached_primary_patch_l2"
            ),
            "branch_output_initialization": "zero_native_dino_identity",
            "expected_added_parameter_count_for_embed_dim_384": int(
                expected_added
            ),
            "observed_added_parameter_count": self.added_parameter_count(),
        }

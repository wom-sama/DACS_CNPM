from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trkh.models.dinov3_surface_patch_hybrid_v2 import (
    CANONICAL_EMBED_DIM,
    CANONICAL_PATCH_COUNT,
    CANONICAL_PATCH_SIZE,
    CANONICAL_PREFIX_TOKENS,
    LOCAL_SURFACE_MODE,
    DinoV3SurfaceHybridContractError,
    DinoV3SurfacePatchHybridV2,
    _parameter_count,
)


RELATIVE_SURFACE_PAIR_MODE = "relative_surface_pair"
GENERIC_TOKEN_PAIR_MODE = "generic_token_pair"
SUPPORTED_PAIR_MODES = (RELATIVE_SURFACE_PAIR_MODE, GENERIC_TOKEN_PAIR_MODE)

FOCUS_CLASS_INDEX = 1
PAIR_COMPETITOR_INDICES = (0, 2, 4)
PAIR_DESCRIPTOR_DIM = 7
PAIR_HIDDEN_DIM = 16
PAIR_RESIDUAL_RATIO_CAP = 0.04
PAIR_EVIDENCE_PARAMETER_COUNT = 160
PATCH_LUMA_FLOOR = 1.0 / 16.0
DESCRIPTOR_EPS = 1e-6


class DinoV3SurfacePairHybridV3(DinoV3SurfacePatchHybridV2):
    """Patch-relative, class-pair-constrained surface adapter for DINOv3.

    V2 allowed an RGB specialist to create any 384-D residual and then learned
    a nearly constant scalar gate.  V3 instead learns only three signed local
    evidence scores for class 1 versus classes 0, 2 and 4.  Each score can move
    a patch only along the corresponding detached classifier direction.  The
    branch is therefore small, interpretable and unable to invent an arbitrary
    token direction.

    ``relative_surface_pair`` consumes seven fixed within-patch statistics.
    ``generic_token_pair`` is the same-capacity DINO-only causal control.  The
    pair head's final layer is zero initialized, preserving exact native-DINO
    logits at step zero while retaining a first-step classification gradient.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,
        mode: str = RELATIVE_SURFACE_PAIR_MODE,
        input_mean: Sequence[float] = (0.485, 0.456, 0.406),
        input_std: Sequence[float] = (0.229, 0.224, 0.225),
        expected_embed_dim: int = CANONICAL_EMBED_DIM,
        expected_prefix_tokens: int = CANONICAL_PREFIX_TOKENS,
        expected_patch_count: int = CANONICAL_PATCH_COUNT,
        externally_pretrained: bool = True,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_mode = str(mode or "").strip().lower()
        if resolved_mode not in SUPPORTED_PAIR_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_PAIR_MODES!r}; got {mode!r}."
            )
        if int(num_classes) != 5:
            raise ValueError("Surface Pair Hybrid V3 is locked to five classes.")

        self._pair_v3_ready = False
        super().__init__(
            backbone,
            num_classes=num_classes,
            mode=LOCAL_SURFACE_MODE,
            input_mean=input_mean,
            input_std=input_std,
            expected_embed_dim=expected_embed_dim,
            expected_prefix_tokens=expected_prefix_tokens,
            expected_patch_count=expected_patch_count,
            initial_gate_scale=0.01,
            max_gate_scale=PAIR_RESIDUAL_RATIO_CAP,
            externally_pretrained=externally_pretrained,
            source_provenance=source_provenance,
        )

        # Retire every V2 task module.  The parent constructor is RNG-isolated,
        # and the replacement is isolated too, so direct/generic/surface arms
        # still enter data loading and dropout with the same global RNG state.
        self.surface_encoder = None
        self.surface_norm = None
        self.surface_projection = None
        self.token_adapter = None
        self.fusion_gate = None
        self.mode = resolved_mode
        with torch.random.fork_rng(devices=[]):
            self.pair_evidence_head = nn.Sequential(
                nn.Linear(PAIR_DESCRIPTOR_DIM, PAIR_HIDDEN_DIM, bias=False),
                nn.Hardswish(inplace=False),
                nn.Linear(
                    PAIR_HIDDEN_DIM,
                    len(PAIR_COMPETITOR_INDICES),
                    bias=False,
                ),
            )
            output_layer = self.pair_evidence_head[-1]
            if not isinstance(output_layer, nn.Linear):
                raise RuntimeError("Pair evidence output layer contract changed.")
            nn.init.zeros_(output_layer.weight)

        channel_positions = (
            torch.arange(self.embed_dim, dtype=torch.float32) + 0.5
        ).unsqueeze(0)
        frequencies = torch.arange(
            1,
            PAIR_DESCRIPTOR_DIM + 1,
            dtype=torch.float32,
        ).unsqueeze(1)
        generic_projection = torch.cos(
            math.pi * frequencies * channel_positions / float(self.embed_dim)
        )
        generic_projection = F.normalize(generic_projection, dim=1)
        self.register_buffer(
            "generic_descriptor_projection",
            generic_projection,
            persistent=False,
        )
        self.register_buffer(
            "surface_luma_weights",
            torch.tensor((0.2126, 0.7152, 0.0722), dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )

        classifier = self.get_classifier()
        classifier_weight = getattr(classifier, "weight", None)
        if not torch.is_tensor(classifier_weight) or tuple(
            classifier_weight.shape
        ) != (5, self.embed_dim):
            raise DinoV3SurfaceHybridContractError(
                "Surface Pair Hybrid V3 requires a linear 5-class DINO head "
                f"with weight shape [5,{self.embed_dim}]."
            )
        if not isinstance(getattr(self.backbone, "fc_norm", None), nn.Identity):
            raise DinoV3SurfaceHybridContractError(
                "Surface Pair Hybrid V3 requires the locked identity fc_norm."
            )
        head_drop = getattr(self.backbone, "head_drop", None)
        if not isinstance(head_drop, nn.Dropout) or float(head_drop.p) != 0.0:
            raise DinoV3SurfaceHybridContractError(
                "Surface Pair Hybrid V3 requires the locked zero-probability head dropout."
            )

        self.pretrained_task_parameter_prefixes = ("pair_evidence_head.",)
        self.is_pretrained_surface_patch_hybrid_v2 = False
        self.is_pretrained_surface_pair_hybrid_v3 = True
        self.is_pretrained_surface_hybrid = True
        self._pair_v3_ready = True
        self.pretrained_provenance.update(
            {
                "architecture": type(self).__name__,
                "fusion": self.fusion_provenance(),
                "attention_audit": {
                    "native_attention_source": (
                        "backbone_dinov3_final_block_after_pair_residual"
                    ),
                    "surface_specialist_native_attention": False,
                    "surface_specialist_xai": (
                        "trace signed pair-evidence maps plus full-model perturbation"
                    ),
                },
            }
        )

    def no_weight_decay(self) -> set[str]:
        inherited = getattr(self.backbone, "no_weight_decay", None)
        names = set(str(name) for name in inherited()) if callable(inherited) else set()
        return {f"backbone.{name}" for name in names}

    @staticmethod
    def _zero_safe_rms(mean_square: Tensor) -> Tensor:
        return torch.sqrt(mean_square + DESCRIPTOR_EPS) - math.sqrt(DESCRIPTOR_EPS)

    @staticmethod
    def _smooth_unit_l2(descriptor: Tensor) -> Tensor:
        denominator = torch.sqrt(
            descriptor.square().sum(dim=-1, keepdim=True) + DESCRIPTOR_EPS
        )
        return descriptor / denominator

    def relative_surface_descriptor(
        self,
        images: Tensor,
        *,
        return_map: bool = True,
    ) -> Tuple[Tensor, Tensor | None]:
        """Return seven illumination-normalized statistics per exact patch."""

        self._validate_images(images)
        mean = self.input_mean.to(device=images.device, dtype=torch.float32)
        std = self.input_std.to(device=images.device, dtype=torch.float32)
        rgb = (images.float() * std + mean).clamp(min=0.0, max=1.0)
        rgb = torch.where(rgb < DESCRIPTOR_EPS, torch.zeros_like(rgb), rgb)
        # Keep the batch extent symbolic for the standard dynamic-batch ONNX
        # export contract.  Converting it to ``int`` here silently freezes a
        # batch-1 trace even though the exporter advertises a dynamic axis.
        batch_size = rgb.size(0)
        grid_height = rgb.size(-2) // CANONICAL_PATCH_SIZE[0]
        grid_width = rgb.size(-1) // CANONICAL_PATCH_SIZE[1]
        patches = rgb.reshape(
            batch_size,
            3,
            grid_height,
            CANONICAL_PATCH_SIZE[0],
            grid_width,
            CANONICAL_PATCH_SIZE[1],
        ).permute(0, 2, 4, 1, 3, 5)
        patch_mean_rgb = patches.mean(dim=(-1, -2))

        luma_weights = self.surface_luma_weights.to(
            device=images.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, 3, 1, 1)
        patch_luma = (patches * luma_weights).sum(dim=3)
        patch_mean_luma = patch_luma.mean(dim=(-1, -2))
        safe_patch_luma = patch_mean_luma.clamp_min(PATCH_LUMA_FLOOR)
        relative_rgb = (
            patches - patch_mean_rgb[..., None, None]
        ) / safe_patch_luma[..., None, None, None]
        relative_rgb_rms = self._zero_safe_rms(
            relative_rgb.square().mean(dim=(-1, -2))
        )
        patch_mean_chromaticity = (
            patches
            / patches.sum(dim=3, keepdim=True).clamp_min(PATCH_LUMA_FLOOR)
        ).mean(dim=(-1, -2))
        relative_luma = (
            patch_luma - patch_mean_luma[..., None, None]
        ) / safe_patch_luma[..., None, None]
        relative_luma_std = self._zero_safe_rms(
            relative_luma.square().mean(dim=(-1, -2))
        )
        descriptor_grid = torch.cat(
            (
                relative_rgb_rms,
                patch_mean_chromaticity,
                relative_luma_std.unsqueeze(-1),
            ),
            dim=-1,
        )
        descriptor_tokens = descriptor_grid.reshape(
            batch_size,
            grid_height * grid_width,
            PAIR_DESCRIPTOR_DIM,
        )
        descriptor_tokens = self._smooth_unit_l2(descriptor_tokens)
        descriptor_map = None
        if return_map:
            descriptor_map = descriptor_tokens.transpose(1, 2).reshape(
                batch_size,
                PAIR_DESCRIPTOR_DIM,
                grid_height,
                grid_width,
            )
        return descriptor_tokens, descriptor_map

    def generic_token_descriptor(self, detached_patch_tokens: Tensor) -> Tensor:
        normalized_tokens = F.layer_norm(
            detached_patch_tokens.float(),
            normalized_shape=(self.embed_dim,),
            eps=1e-6,
        )
        projection = self.generic_descriptor_projection.to(
            device=normalized_tokens.device,
            dtype=normalized_tokens.dtype,
        )
        descriptor_tokens = F.linear(normalized_tokens, projection)
        return self._smooth_unit_l2(descriptor_tokens)

    def _preview_routing(
        self,
        pre_fusion_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            preview_patches = self.backbone.norm(
                pre_fusion_tokens[:, self.num_prefix_tokens :].detach()
            )
            preview_features = preview_patches.mean(dim=1)
            classifier = self.get_classifier()
            preview_logits = F.linear(
                preview_features,
                classifier.weight,
                classifier.bias,
            ).float()
            preview_probabilities = torch.softmax(preview_logits, dim=-1)
            competitor_probabilities = preview_probabilities[
                :, list(PAIR_COMPETITOR_INDICES)
            ]
            competitor_weights = competitor_probabilities / (
                competitor_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            )
            non_class3_mass = 1.0 - preview_probabilities[:, 3:4]
        return competitor_weights, non_class3_mass, preview_logits

    def _semantic_pair_directions(self) -> Tensor:
        classifier_weight = self.get_classifier().weight.detach().float()
        focus_weight = classifier_weight[FOCUS_CLASS_INDEX : FOCUS_CLASS_INDEX + 1]
        competitor_weight = classifier_weight[list(PAIR_COMPETITOR_INDICES)]
        return F.normalize(focus_weight - competitor_weight, dim=-1, eps=1e-12)

    def _build_gated_patch_residual(
        self,
        images: Tensor,
        pre_fusion_tokens: Tensor,
        patch_tokens: Tensor,
        detached_patch_tokens: Tensor,
        *,
        return_trace: bool,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if self.mode == RELATIVE_SURFACE_PAIR_MODE:
            descriptor_tokens, descriptor_map = self.relative_surface_descriptor(
                images,
                return_map=return_trace,
            )
            descriptor_kind = (
                "unit_l2_patch_relative_rgb_chromaticity_luma_std"
            )
        else:
            descriptor_tokens = self.generic_token_descriptor(detached_patch_tokens)
            descriptor_map = None
            if return_trace:
                grid_height = images.size(-2) // CANONICAL_PATCH_SIZE[0]
                grid_width = images.size(-1) // CANONICAL_PATCH_SIZE[1]
                descriptor_map = descriptor_tokens.transpose(1, 2).reshape(
                    descriptor_tokens.size(0),
                    PAIR_DESCRIPTOR_DIM,
                    grid_height,
                    grid_width,
                )
            descriptor_kind = (
                "unit_l2_fixed_dct_summary_of_detached_dino_tokens"
            )
        if not torch.jit.is_tracing() and int(descriptor_tokens.size(1)) != int(
            patch_tokens.size(1)
        ):
            raise DinoV3SurfaceHybridContractError(
                "Pair descriptor/DINO patch grids are misaligned: "
                f"descriptor={int(descriptor_tokens.size(1))}, "
                f"dino={int(patch_tokens.size(1))}."
            )

        raw_pair_scores = self.pair_evidence_head(descriptor_tokens)
        competitor_weights, non_class3_mass, preview_logits = self._preview_routing(
            pre_fusion_tokens
        )
        signed_pair_coefficients = (
            torch.tanh(raw_pair_scores.float())
            * competitor_weights[:, None, :]
            * non_class3_mass[:, None, :]
        )
        semantic_directions = self._semantic_pair_directions().to(
            device=patch_tokens.device,
            dtype=signed_pair_coefficients.dtype,
        )
        raw_delta = torch.matmul(signed_pair_coefficients, semantic_directions)
        normalized_delta = self._normalize_delta(raw_delta, detached_patch_tokens)
        typed_delta = normalized_delta.to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        gated_residual = PAIR_RESIDUAL_RATIO_CAP * typed_delta
        if not return_trace:
            return gated_residual, {}
        if descriptor_map is None:
            raise RuntimeError("Pair descriptor trace map was not constructed.")
        routing_strength = (
            signed_pair_coefficients.abs().sum(dim=-1, keepdim=True).clamp_max(1.0)
            * PAIR_RESIDUAL_RATIO_CAP
        ).to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        return gated_residual, {
            "surface_descriptor_tensor": descriptor_tokens,
            "surface_feature_map": descriptor_map,
            "surface_descriptor_kind": descriptor_kind,
            "surface_pair_scores": raw_pair_scores,
            "signed_pair_coefficients": signed_pair_coefficients,
            "pair_competitor_weights": competitor_weights,
            "pair_non_class3_mass": non_class3_mass,
            "pair_preview_logits": preview_logits,
            "semantic_pair_directions": semantic_directions,
            "raw_surface_delta": raw_delta,
            "normalized_surface_delta": normalized_delta,
            "raw_gate": raw_pair_scores,
            "pair_routing_strength": routing_strength,
            # Compatibility alias consumed by the shared V2 trace envelope.
            "gate": routing_strength,
        }

    def fusion_runtime_parameter_telemetry(self) -> Dict[str, float]:
        first = self.pair_evidence_head[0]
        output = self.pair_evidence_head[-1]
        if not isinstance(first, nn.Linear) or not isinstance(output, nn.Linear):
            raise RuntimeError("Pair evidence head contract changed.")
        return {
            "pair_hidden_weight_l2": float(
                first.weight.detach().float().norm().cpu().item()
            ),
            "pair_output_weight_l2": float(
                output.weight.detach().float().norm().cpu().item()
            ),
            "max_residual_ratio": float(PAIR_RESIDUAL_RATIO_CAP),
        }

    def fusion_provenance(self) -> Dict[str, Any]:
        if not bool(getattr(self, "_pair_v3_ready", False)):
            return super().fusion_provenance()
        return {
            "schema_version": 1,
            "mode": self.mode,
            "primary_path": (
                "dinov3_tokens_plus_bounded_semantic_pair_residual_before_final_block"
            ),
            "primary_context_detached": True,
            "prefix_tokens_modified_at_injection": False,
            "pooling": "backbone_avg_patch_pool",
            "fusion_point": "before_backbone_block_11_zero_based",
            "fusion_tail_block_count": 1,
            "expected_embed_dim": int(self.embed_dim),
            "expected_prefix_tokens": int(self.num_prefix_tokens),
            "expected_patch_count": int(self.expected_patch_count),
            "expected_patch_size": list(CANONICAL_PATCH_SIZE),
            "descriptor_dim": int(PAIR_DESCRIPTOR_DIM),
            "descriptor_normalization": "per_patch_smooth_unit_l2",
            "pair_hidden_dim": int(PAIR_HIDDEN_DIM),
            "focus_class": int(FOCUS_CLASS_INDEX),
            "competitor_classes": list(PAIR_COMPETITOR_INDICES),
            "semantic_directions": "detach_and_normalize(W_focus-W_competitor)",
            "competitor_routing": "detached_pre_final_preview_softmax",
            "class3_safety": "fixed_one_minus_preview_probability",
            "learned_scalar_gate": False,
            "auxiliary_loss": False,
            "evidence_head_bias": False,
            "residual_ratio_cap": float(PAIR_RESIDUAL_RATIO_CAP),
            "branch_output_initialization": "zero_native_dino_identity",
            "task_module_rng_isolation": "torch.random.fork_rng(devices=[])",
            "expected_added_parameter_count": int(PAIR_EVIDENCE_PARAMETER_COUNT),
            "observed_added_parameter_count": int(self.added_parameter_count()),
            "observed_total_parameter_count": int(_parameter_count(self)),
        }

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LOCAL_CONV_A0_MODE = "local_conv_a0"
LOCAL_XCNORM_A0_MODE = "local_xcnorm_a0"
SUPPORTED_XCNORM_PAIR_MODES = (LOCAL_CONV_A0_MODE, LOCAL_XCNORM_A0_MODE)

XCNORM_PAIR_EMBED_DIM = 384
XCNORM_PAIR_HIDDEN_DIM = 8
XCNORM_PAIR_GRID_SIZE = 16
XCNORM_PAIR_PATCH_COUNT = XCNORM_PAIR_GRID_SIZE * XCNORM_PAIR_GRID_SIZE
XCNORM_PAIR_FOCUS_CLASS = 1
XCNORM_PAIR_COMPETITORS = (0, 2, 4)
XCNORM_PAIR_RESIDUAL_RATIO_CAP = 0.04
XCNORM_PAIR_ADDED_PARAMETER_COUNT = 3_672
XCNORM_EPS = 1e-6


class DinoV3XCNormPairContractError(ValueError):
    """Raised when the locked A0 representation contract is violated."""


class DinoV3XCNormPairAdapterA0(nn.Module):
    """A bounded post-DINO-norm local pair adapter.

    The module deliberately owns only three tensors: a bias-free 384->8 token
    projection, one 8x8x3x3 local template bank, and a zero-initialized
    bias-free 8->3 pair projection.  ``local_conv_a0`` and
    ``local_xcnorm_a0`` therefore have identical capacity and state-dict
    geometry.  The latter replaces ordinary convolution with fixed normalized
    cross-correlation over each valid 3x3 neighborhood.

    Inputs are normalized DINO patch tokens shaped ``[B,256,384]``.  A fixed
    (parameter-free) layer norm precedes the learned projection.  The 14x14
    local response is Hardswish-activated and constant-zero padded back to the
    16x16 patch grid.  Three signed scores use a fixed minimum-norm map in the
    span of the raw detached classifier-margin rows W1-W{0,2,4}; normalized
    directions are retained only as telemetry.  This avoids cross-Gram margin
    coupling while retaining a strict smooth 4%
    token-norm cap.  The map must be bound once from a frozen classifier before
    the active branch is used; no matrix inverse enters the forward graph.
    """

    def __init__(
        self,
        mode: str = LOCAL_XCNORM_A0_MODE,
        *,
        embed_dim: int = XCNORM_PAIR_EMBED_DIM,
        grid_size: int = XCNORM_PAIR_GRID_SIZE,
        residual_ratio_cap: float = XCNORM_PAIR_RESIDUAL_RATIO_CAP,
        eps: float = XCNORM_EPS,
    ) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_XCNORM_PAIR_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_XCNORM_PAIR_MODES!r}; "
                f"got {mode!r}."
            )
        if int(embed_dim) != XCNORM_PAIR_EMBED_DIM:
            raise DinoV3XCNormPairContractError(
                f"A0 is locked to embed_dim={XCNORM_PAIR_EMBED_DIM}."
            )
        if int(grid_size) != XCNORM_PAIR_GRID_SIZE:
            raise DinoV3XCNormPairContractError(
                f"A0 is locked to a {XCNORM_PAIR_GRID_SIZE}x"
                f"{XCNORM_PAIR_GRID_SIZE} patch grid."
            )
        if not (0.0 < float(residual_ratio_cap) <= XCNORM_PAIR_RESIDUAL_RATIO_CAP):
            raise DinoV3XCNormPairContractError(
                "residual_ratio_cap must be in (0, 0.04]."
            )
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive.")

        self.mode = resolved_mode
        self.embed_dim = int(embed_dim)
        self.grid_size = int(grid_size)
        self.patch_count = self.grid_size * self.grid_size
        self.residual_ratio_cap = float(residual_ratio_cap)
        self.eps = float(eps)

        self.token_projection = nn.Linear(
            self.embed_dim,
            XCNORM_PAIR_HIDDEN_DIM,
            bias=False,
        )
        self.local_weight = nn.Parameter(
            torch.empty(
                XCNORM_PAIR_HIDDEN_DIM,
                XCNORM_PAIR_HIDDEN_DIM,
                3,
                3,
            )
        )
        self.pair_projection = nn.Linear(
            XCNORM_PAIR_HIDDEN_DIM,
            len(XCNORM_PAIR_COMPETITORS),
            bias=False,
        )
        nn.init.kaiming_uniform_(self.local_weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.pair_projection.weight)

        self.register_buffer(
            "local_sum_kernel",
            torch.ones(1, XCNORM_PAIR_HIDDEN_DIM, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "classifier_pair_rows",
            torch.zeros(len(XCNORM_PAIR_COMPETITORS), self.embed_dim),
        )
        self.register_buffer(
            "semantic_pair_directions_buffer",
            torch.zeros(len(XCNORM_PAIR_COMPETITORS), self.embed_dim),
        )
        self.register_buffer(
            "pair_margin_mapping",
            torch.zeros(len(XCNORM_PAIR_COMPETITORS), self.embed_dim),
        )
        self.register_buffer(
            "classifier_binding_ready",
            torch.tensor(False, dtype=torch.bool),
        )
        observed = self.added_parameter_count()
        if observed != XCNORM_PAIR_ADDED_PARAMETER_COUNT:
            raise RuntimeError(
                "A0 parameter contract changed: "
                f"expected {XCNORM_PAIR_ADDED_PARAMETER_COUNT}, observed {observed}."
            )

    def _validate_patch_tokens(self, patch_tokens: Tensor) -> None:
        if patch_tokens.ndim != 3 or tuple(patch_tokens.shape[1:]) != (
            self.patch_count,
            self.embed_dim,
        ):
            raise DinoV3XCNormPairContractError(
                "patch_tokens must have shape "
                f"[B,{self.patch_count},{self.embed_dim}]; got "
                f"{tuple(patch_tokens.shape)}."
            )
        if not patch_tokens.is_floating_point():
            raise DinoV3XCNormPairContractError(
                "patch_tokens must use a floating-point dtype."
            )

    def _validate_classifier_weight(self, classifier_weight: Tensor) -> None:
        if classifier_weight.ndim != 2 or tuple(classifier_weight.shape) != (
            5,
            self.embed_dim,
        ):
            raise DinoV3XCNormPairContractError(
                "classifier_weight must have shape "
                f"[5,{self.embed_dim}]; got {tuple(classifier_weight.shape)}."
            )

    def project_token_grid(self, patch_tokens: Tensor) -> Tensor:
        """Apply fixed channel LayerNorm and form the learned 8x16x16 grid."""

        self._validate_patch_tokens(patch_tokens)
        frozen_tokens = patch_tokens.detach()
        normalized = F.layer_norm(
            frozen_tokens.float(),
            normalized_shape=(self.embed_dim,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        projected = F.linear(normalized, self.token_projection.weight.float())
        return projected.transpose(1, 2).reshape(
            frozen_tokens.size(0),
            XCNORM_PAIR_HIDDEN_DIM,
            self.grid_size,
            self.grid_size,
        )

    def _xcnorm_response(self, feature_map: Tensor) -> Tensor:
        # One correlation vector contains every channel and 3x3 location.
        # Centering the template makes Conv(x, centered_w) algebraically equal
        # to Conv(centered_x, centered_w), while two ordinary convolutions
        # provide the local sum and square-sum without unfold or resize.
        sample_size = float(XCNORM_PAIR_HIDDEN_DIM * 3 * 3)
        output_dtype = feature_map.dtype
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            feature_float = feature_map.float()
            # Remove the sample-wide offset before the local reductions.  NCC
            # is invariant to this common offset, while the shift prevents a
            # large constant input from leaving a floating-point Conv residue
            # that would otherwise be amplified by the epsilon denominator.
            shifted_feature = feature_float - feature_float.mean(
                dim=(1, 2, 3),
                keepdim=True,
            )
            weight_float = self.local_weight.float()
            template_mean = weight_float.sum(
                dim=(1, 2, 3),
                keepdim=True,
            ) / sample_size
            centered_template = weight_float - template_mean
            numerator = F.conv2d(shifted_feature, centered_template, bias=None)

            sum_kernel = self.local_sum_kernel.to(
                device=shifted_feature.device,
                dtype=torch.float32,
            )
            local_sum = F.conv2d(shifted_feature, sum_kernel, bias=None)
            local_square_sum = F.conv2d(
                shifted_feature.square(),
                sum_kernel,
                bias=None,
            )
            centered_input_square_sum = (
                local_square_sum - local_sum.square() / sample_size
            ).clamp_min(0.0)
            centered_template_square_sum = centered_template.square().sum(
                dim=(1, 2, 3),
            ).reshape(1, XCNORM_PAIR_HIDDEN_DIM, 1, 1)
            # Locked A0 stabilizer: ||x_c||*||w_c|| + eps.  For an isotropic
            # 72-vector NCC has variance 1/72, while the matched Kaiming Conv
            # has variance about 1/9 after the preceding projection.  The
            # fixed sqrt(8) scale therefore matches their expected RMS without
            # a data-fitted calibration axis.
            valid_energy = (
                centered_input_square_sum > self.eps * self.eps
            ) & (centered_template_square_sum > self.eps * self.eps)
            energy_product = (
                centered_input_square_sum * centered_template_square_sum
            )
            safe_energy_product = torch.where(
                valid_energy,
                energy_product,
                torch.ones_like(energy_product),
            )
            denominator = torch.sqrt(safe_energy_product) + self.eps
            response = (
                numerator * math.sqrt(XCNORM_PAIR_HIDDEN_DIM) / denominator
            )
            response = torch.where(
                valid_energy,
                response,
                torch.zeros_like(response),
            )
        return response.to(dtype=output_dtype)

    def local_response(self, feature_map: Tensor) -> Tensor:
        """Return the valid 14x14 local response before Hardswish."""

        expected_shape = (
            XCNORM_PAIR_HIDDEN_DIM,
            self.grid_size,
            self.grid_size,
        )
        if feature_map.ndim != 4 or tuple(feature_map.shape[1:]) != expected_shape:
            raise DinoV3XCNormPairContractError(
                "feature_map must have shape "
                f"[B,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}]; "
                f"got {tuple(feature_map.shape)}."
            )
        if self.mode == LOCAL_CONV_A0_MODE:
            output_dtype = feature_map.dtype
            with torch.autocast(
                device_type=feature_map.device.type,
                enabled=False,
            ):
                response = F.conv2d(
                    feature_map.float(),
                    self.local_weight.float(),
                    bias=None,
                )
            return response.to(dtype=output_dtype)
        return self._xcnorm_response(feature_map)

    def pair_scores_from_tokens(
        self,
        patch_tokens: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return three signed pair-score maps aligned to all 256 patches."""

        projected_map = self.project_token_grid(patch_tokens)
        local_response = self.local_response(projected_map)
        activated_valid = F.hardswish(local_response, inplace=False)
        activated_grid = F.pad(
            activated_valid,
            (1, 1, 1, 1),
            mode="constant",
            value=0.0,
        )
        hidden_tokens = activated_grid.flatten(2).transpose(1, 2)
        pair_scores = self.pair_projection(hidden_tokens)
        return pair_scores, {
            "projected_feature_map": projected_map,
            "local_response_valid": local_response,
            "activated_local_map": activated_grid,
            "hidden_tokens": hidden_tokens,
        }

    def semantic_pair_directions(self, classifier_weight: Tensor) -> Tensor:
        """Return detached unit directions W1-W0, W1-W2 and W1-W4."""

        self._validate_classifier_weight(classifier_weight)
        detached_weight = classifier_weight.detach().float()
        focus = detached_weight[
            XCNORM_PAIR_FOCUS_CLASS : XCNORM_PAIR_FOCUS_CLASS + 1
        ]
        competitors = detached_weight[list(XCNORM_PAIR_COMPETITORS)]
        return F.normalize(focus - competitors, dim=-1, eps=1e-12)

    @torch.no_grad()
    def bind_classifier_weight(self, classifier_weight: Tensor) -> None:
        """Freeze the three-direction minimum-norm map outside forward/export."""

        self._validate_classifier_weight(classifier_weight)
        detached_weight = classifier_weight.detach().float()
        focus = detached_weight[
            XCNORM_PAIR_FOCUS_CLASS : XCNORM_PAIR_FOCUS_CLASS + 1
        ]
        rows = focus - detached_weight[list(XCNORM_PAIR_COMPETITORS)]
        directions = F.normalize(rows, dim=-1, eps=1e-12)
        gram = torch.matmul(rows, rows.transpose(0, 1))
        try:
            mapping = torch.linalg.solve(gram, rows)
        except RuntimeError as exc:
            raise DinoV3XCNormPairContractError(
                "classifier pair directions are not linearly independent."
            ) from exc
        reconstructed = torch.matmul(mapping, rows.transpose(0, 1))
        identity = torch.eye(
            len(XCNORM_PAIR_COMPETITORS),
            device=reconstructed.device,
            dtype=reconstructed.dtype,
        )
        if not bool(torch.isfinite(mapping).all()) or not torch.allclose(
            reconstructed,
            identity,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise DinoV3XCNormPairContractError(
                "classifier pair directions do not admit a stable A0 mapping."
            )
        self.classifier_pair_rows.copy_(
            rows.to(
                device=self.classifier_pair_rows.device,
                dtype=self.classifier_pair_rows.dtype,
            )
        )
        self.semantic_pair_directions_buffer.copy_(
            directions.to(
                device=self.semantic_pair_directions_buffer.device,
                dtype=self.semantic_pair_directions_buffer.dtype,
            )
        )
        self.pair_margin_mapping.copy_(
            mapping.to(
                device=self.pair_margin_mapping.device,
                dtype=self.pair_margin_mapping.dtype,
            )
        )
        self.classifier_binding_ready.fill_(True)

    def classifier_binding_matches(
        self,
        classifier_weight: Tensor,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> bool:
        """Explicit preflight for the frozen-classifier contract."""

        if not bool(self.classifier_binding_ready.item()):
            return False
        self._validate_classifier_weight(classifier_weight)
        detached_weight = classifier_weight.detach().float()
        current_rows = (
            detached_weight[
                XCNORM_PAIR_FOCUS_CLASS : XCNORM_PAIR_FOCUS_CLASS + 1
            ]
            - detached_weight[list(XCNORM_PAIR_COMPETITORS)]
        ).to(
            device=self.classifier_pair_rows.device,
            dtype=self.classifier_pair_rows.dtype,
        )
        return bool(
            torch.allclose(
                current_rows,
                self.classifier_pair_rows,
                atol=float(atol),
                rtol=float(rtol),
            )
        )

    def bounded_pair_residual(
        self,
        pair_scores: Tensor,
        patch_tokens: Tensor,
        classifier_weight: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Map pair scores to a smooth residual whose norm ratio is below 4%."""

        self._validate_patch_tokens(patch_tokens)
        frozen_tokens = patch_tokens.detach()
        expected_score_shape = (
            patch_tokens.size(0),
            self.patch_count,
            len(XCNORM_PAIR_COMPETITORS),
        )
        if tuple(pair_scores.shape) != expected_score_shape:
            raise DinoV3XCNormPairContractError(
                f"pair_scores must have shape {expected_score_shape}; got "
                f"{tuple(pair_scores.shape)}."
            )
        self._validate_classifier_weight(classifier_weight)
        if not bool(self.classifier_binding_ready.item()):
            raise DinoV3XCNormPairContractError(
                "call bind_classifier_weight() with the frozen 5-class head "
                "before enabling the A0 branch."
            )
        if not self.classifier_binding_matches(classifier_weight):
            raise DinoV3XCNormPairContractError(
                "classifier_weight no longer matches the frozen head used to "
                "bind the A0 minimum-norm mapping."
            )
        directions = self.semantic_pair_directions_buffer.to(
            device=frozen_tokens.device,
            dtype=torch.float32,
        )
        classifier_pair_rows = self.classifier_pair_rows.to(
            device=frozen_tokens.device,
            dtype=torch.float32,
        )
        coefficients = torch.tanh(pair_scores.float())
        mapping = self.pair_margin_mapping.to(
            device=frozen_tokens.device,
            dtype=torch.float32,
        )
        raw_delta = torch.matmul(coefficients, mapping)
        raw_norm_square = raw_delta.square().sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(0.0)
        unit_bounded_delta = raw_delta / torch.sqrt(1.0 + raw_norm_square)
        token_norm = frozen_tokens.float().norm(dim=-1, keepdim=True)
        residual_float = (
            self.residual_ratio_cap * token_norm * unit_bounded_delta
        )
        residual = residual_float.to(
            device=frozen_tokens.device,
            dtype=frozen_tokens.dtype,
        )
        residual_ratio = residual_float.norm(dim=-1) / token_norm.squeeze(
            -1
        ).clamp_min(1e-12)
        return residual, {
            "signed_pair_coefficients": coefficients,
            "semantic_pair_directions": directions,
            "classifier_pair_rows": classifier_pair_rows,
            "pair_margin_mapping": mapping,
            "raw_pair_delta": raw_delta,
            "residual_norm_ratio": residual_ratio,
        }

    def _residual_with_trace(
        self,
        patch_tokens: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        self._validate_patch_tokens(patch_tokens)
        self._validate_classifier_weight(classifier_weight)
        frozen_tokens = patch_tokens.detach()
        if branch_off:
            return torch.zeros_like(frozen_tokens), {
                "mode": self.mode,
                "branch_off": True,
                "residual": torch.zeros_like(frozen_tokens),
            }
        pair_scores, local_trace = self.pair_scores_from_tokens(frozen_tokens)
        residual, residual_trace = self.bounded_pair_residual(
            pair_scores,
            frozen_tokens,
            classifier_weight,
        )
        return residual, {
            "mode": self.mode,
            "branch_off": False,
            "pair_scores": pair_scores,
            "residual": residual,
            **local_trace,
            **residual_trace,
        }

    def forward(
        self,
        patch_tokens: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool = False,
    ) -> Tensor:
        residual, _ = self._residual_with_trace(
            patch_tokens,
            classifier_weight,
            branch_off=bool(branch_off),
        )
        if branch_off:
            return patch_tokens.detach()
        return patch_tokens.detach() + residual

    def forward_with_trace(
        self,
        patch_tokens: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        residual, trace = self._residual_with_trace(
            patch_tokens,
            classifier_weight,
            branch_off=bool(branch_off),
        )
        frozen_tokens = patch_tokens.detach()
        fused = frozen_tokens if branch_off else frozen_tokens + residual
        trace["fused_patch_tokens"] = fused
        return fused, trace

    def logit_delta_from_cached_tokens(
        self,
        patch_tokens: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool = False,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """Return mean-pool/head delta while preserving any cached base logits."""

        residual, trace = self._residual_with_trace(
            patch_tokens,
            classifier_weight,
            branch_off=bool(branch_off),
        )
        delta = F.linear(
            residual.mean(dim=1),
            classifier_weight.detach(),
            bias=None,
        )
        if return_trace:
            trace["logit_delta"] = delta
            return delta, trace
        return delta

    def logits_from_cached_tokens(
        self,
        patch_tokens: Tensor,
        classifier_weight: Tensor,
        classifier_bias: Tensor | None = None,
        *,
        base_logits: Tensor | None = None,
        branch_off: bool = False,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """Apply the adapter to cached tokens without running the backbone.

        ``base_logits`` may contain the exact backbone output for a nonstandard
        pooling path.  In that case only the average-patch residual's linear
        head delta is added.  Without it, the method uses ordinary average
        patch pooling as the complete baseline.
        """

        self._validate_patch_tokens(patch_tokens)
        self._validate_classifier_weight(classifier_weight)
        frozen_tokens = patch_tokens.detach()
        frozen_classifier_weight = classifier_weight.detach()
        if base_logits is None:
            baseline = F.linear(
                frozen_tokens.mean(dim=1),
                frozen_classifier_weight,
                None if classifier_bias is None else classifier_bias.detach(),
            )
        else:
            if tuple(base_logits.shape) != (patch_tokens.size(0), 5):
                raise DinoV3XCNormPairContractError(
                    "base_logits must have shape "
                    f"[B,5]; got {tuple(base_logits.shape)}."
                )
            baseline = base_logits.detach()
        delta_result = self.logit_delta_from_cached_tokens(
            patch_tokens,
            classifier_weight,
            branch_off=bool(branch_off),
            return_trace=bool(return_trace),
        )
        if return_trace:
            delta, trace = delta_result
            logits = baseline if branch_off else baseline + delta
            trace["base_logits"] = baseline
            trace["logits"] = logits
            return logits, trace
        delta = delta_result
        return baseline if branch_off else baseline + delta

    def added_parameter_count(self, *, trainable_only: bool = False) -> int:
        return int(
            sum(
                parameter.numel()
                for parameter in self.parameters()
                if not trainable_only or parameter.requires_grad
            )
        )

    def parameter_telemetry(self) -> Dict[str, int]:
        return {
            "added_parameter_count": self.added_parameter_count(),
            "trainable_added_parameter_count": self.added_parameter_count(
                trainable_only=True
            ),
            "token_projection_parameter_count": int(
                self.token_projection.weight.numel()
            ),
            "local_template_parameter_count": int(self.local_weight.numel()),
            "pair_projection_parameter_count": int(
                self.pair_projection.weight.numel()
            ),
        }

    def fusion_runtime_parameter_telemetry(self) -> Dict[str, float]:
        return {
            "token_projection_weight_l2": float(
                self.token_projection.weight.detach().float().norm().cpu().item()
            ),
            "local_template_weight_l2": float(
                self.local_weight.detach().float().norm().cpu().item()
            ),
            "pair_projection_weight_l2": float(
                self.pair_projection.weight.detach().float().norm().cpu().item()
            ),
            "max_residual_ratio": float(self.residual_ratio_cap),
        }

    def fusion_provenance(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "architecture": type(self).__name__,
            "mode": self.mode,
            "research_track": "pretrained",
            "input": "post_dinov3_norm_patch_tokens",
            "input_shape": [None, self.patch_count, self.embed_dim],
            "fixed_token_normalization": "layer_norm_no_affine",
            "local_operator": (
                "ordinary_valid_conv3x3"
                if self.mode == LOCAL_CONV_A0_MODE
                else "normalized_cross_correlation_valid3x3"
            ),
            "xcnorm_statistics_precision": "fp32_autocast_disabled",
            "xcnorm_denominator": (
                "sqrt(centered_x_ss*centered_w_ss)+1e-6_with_"
                "degenerate_energy_zero_mask"
            ),
            "xcnorm_fixed_response_scale": (
                "sqrt(8)_analytic_kaiming_rms_match_no_data_calibration"
            ),
            "local_output_alignment": "constant_zero_pad_14x14_to_16x16",
            "activation": "hardswish",
            "focus_class": int(XCNORM_PAIR_FOCUS_CLASS),
            "competitor_classes": list(XCNORM_PAIR_COMPETITORS),
            "semantic_directions": "normalized_W1-W{0,2,4}_telemetry_only",
            "pair_margin_mapping": (
                "fixed_minimum_norm_R_transpose_inverse_RR_transpose_"
                "from_raw_classifier_margin_rows"
            ),
            "classifier_mapping_contract": "explicitly_bound_frozen_head",
            "gradient_contract": "cached_tokens_and_classifier_are_detached",
            "pooling_contract": "average_patch_residual_before_linear_head",
            "learned_gate": False,
            "bias": False,
            "batch_normalization": False,
            "threshold": False,
            "residual_ratio_cap": float(self.residual_ratio_cap),
            "branch_output_initialization": "zero_exact_base_identity",
            "expected_added_parameter_count": int(
                XCNORM_PAIR_ADDED_PARAMETER_COUNT
            ),
            "observed_added_parameter_count": self.added_parameter_count(),
            "onnx_operator_policy": (
                "conv_reduce_sum_square_sqrt_div_hardswish_constant_pad"
            ),
            "scientific_scope": (
                "conditional_local_token_pattern_screen_not_image_illumination_"
                "invariance_or_complete_deployment_architecture"
            ),
        }

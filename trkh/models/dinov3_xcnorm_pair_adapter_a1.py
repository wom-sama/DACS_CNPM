from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn

from trkh.models.dinov3_xcnorm_pair_adapter_a0 import (
    LOCAL_CONV_A0_MODE,
    LOCAL_XCNORM_A0_MODE,
    XCNORM_PAIR_ADDED_PARAMETER_COUNT,
    XCNORM_PAIR_EMBED_DIM,
    XCNORM_PAIR_PATCH_COUNT,
    XCNORM_PAIR_RESIDUAL_RATIO_CAP,
    DinoV3XCNormPairAdapterA0,
    DinoV3XCNormPairContractError,
)


LOCAL_CONV_A1_MODE = "local_conv_a1"
LOCAL_XCNORM_A1_MODE = "local_xcnorm_a1"
SUPPORTED_XCNORM_PAIR_A1_MODES = (LOCAL_CONV_A1_MODE, LOCAL_XCNORM_A1_MODE)
XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT = XCNORM_PAIR_ADDED_PARAMETER_COUNT
XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT = 5
XCNORM_PAIR_A1_TOTAL_TOKEN_COUNT = (
    XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT + XCNORM_PAIR_PATCH_COUNT
)

_A1_TO_A0_MODE = {
    LOCAL_CONV_A1_MODE: LOCAL_CONV_A0_MODE,
    LOCAL_XCNORM_A1_MODE: LOCAL_XCNORM_A0_MODE,
}


class DinoV3XCNormPairA1ContractError(DinoV3XCNormPairContractError):
    """Raised when the final-MHSA A1 composition contract is violated."""


class DinoV3XCNormPairAdapterA1(nn.Module):
    """Compose the A0 local core as a branch parallel to final DINO MHSA.

    ``z_patch`` is specifically the 256 patch-token slice of the frozen final
    block's ``norm1`` output.  This module returns a bounded patch residual; a
    caller adds it to the corresponding patch slice after the frozen final
    MHSA residual addition and before the frozen final MLP and final LayerNorm.

    The fixed classifier-axis map is only an injection-space prior in A1.  The
    downstream frozen MLP and final LayerNorm are nonlinear, so the map is not
    claimed to produce exact output-logit margin changes.
    """

    def __init__(
        self,
        mode: str = LOCAL_XCNORM_A1_MODE,
        *,
        residual_ratio_cap: float = XCNORM_PAIR_RESIDUAL_RATIO_CAP,
    ) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_XCNORM_PAIR_A1_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_XCNORM_PAIR_A1_MODES!r}; "
                f"got {mode!r}."
            )
        self.mode = resolved_mode
        self.core = DinoV3XCNormPairAdapterA0(
            _A1_TO_A0_MODE[resolved_mode],
            residual_ratio_cap=float(residual_ratio_cap),
        )
        if self.added_parameter_count() != XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT:
            raise RuntimeError("A1 must contain exactly the unchanged A0 core.")

    def bind_classifier_weight(self, classifier_weight: Tensor) -> None:
        """Bind the fixed head-axis prior from a genuinely frozen 5-class head."""

        if bool(classifier_weight.requires_grad):
            raise DinoV3XCNormPairA1ContractError(
                "A1 requires a frozen classifier_weight with requires_grad=False."
            )
        self.core.bind_classifier_weight(classifier_weight)

    def classifier_binding_matches(self, classifier_weight: Tensor) -> bool:
        return self.core.classifier_binding_matches(classifier_weight)

    @staticmethod
    def _validate_residual_inputs(
        z_patch: Tensor,
        residual_reference: Tensor,
    ) -> None:
        expected_tail = (XCNORM_PAIR_PATCH_COUNT, XCNORM_PAIR_EMBED_DIM)
        if z_patch.ndim != 3 or tuple(z_patch.shape[1:]) != expected_tail:
            raise DinoV3XCNormPairA1ContractError(
                "z_patch must have shape "
                f"[B,{expected_tail[0]},{expected_tail[1]}]; got "
                f"{tuple(z_patch.shape)}."
            )
        if (
            residual_reference.ndim != 3
            or tuple(residual_reference.shape[1:]) != expected_tail
        ):
            raise DinoV3XCNormPairA1ContractError(
                "residual_reference must have shape "
                f"[B,{expected_tail[0]},{expected_tail[1]}]; got "
                f"{tuple(residual_reference.shape)}."
            )
        if tuple(z_patch.shape) != tuple(residual_reference.shape):
            raise DinoV3XCNormPairA1ContractError(
                "z_patch and residual_reference shapes must match exactly."
            )
        if z_patch.device != residual_reference.device:
            raise DinoV3XCNormPairA1ContractError(
                "z_patch and residual_reference devices must match exactly."
            )
        if z_patch.dtype != residual_reference.dtype:
            raise DinoV3XCNormPairA1ContractError(
                "z_patch and residual_reference dtypes must match exactly."
            )
        if not z_patch.is_floating_point():
            raise DinoV3XCNormPairA1ContractError(
                "z_patch and residual_reference must use a floating-point dtype."
            )

    @staticmethod
    def _validate_classifier_weight(classifier_weight: Tensor) -> None:
        if classifier_weight.ndim != 2 or tuple(classifier_weight.shape) != (
            5,
            XCNORM_PAIR_EMBED_DIM,
        ):
            raise DinoV3XCNormPairA1ContractError(
                "classifier_weight must have shape "
                f"[5,{XCNORM_PAIR_EMBED_DIM}]; got "
                f"{tuple(classifier_weight.shape)}."
            )
        if bool(classifier_weight.requires_grad):
            raise DinoV3XCNormPairA1ContractError(
                "A1 requires a frozen classifier_weight with requires_grad=False."
            )

    def residual_from_final_norm1(
        self,
        z_patch: Tensor,
        residual_reference: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool = False,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """Score ``z_patch`` and bound the residual against frozen ``u_patch``.

        ``residual_reference`` is the patch slice of the native frozen
        post-final-MHSA stream ``u = x_pre + drop_path1(MHSA(norm1(x_pre)))``.
        Keeping it separate from ``z_patch`` prevents a silent scale mismatch
        between the score source and the actual residual injection stream.
        """

        self._validate_residual_inputs(z_patch, residual_reference)
        self._validate_classifier_weight(classifier_weight)
        frozen_z = z_patch.detach()
        frozen_reference = residual_reference.detach()
        if branch_off:
            residual = torch.zeros_like(frozen_reference)
            trace: Dict[str, Any] = {
                "mode": self.mode,
                "core_mode": self.core.mode,
                "branch_off": True,
                "residual": residual,
            }
        else:
            pair_scores, local_trace = self.core.pair_scores_from_tokens(frozen_z)
            residual, residual_trace = self.core.bounded_pair_residual(
                pair_scores,
                frozen_reference,
                classifier_weight,
            )
            trace = {
                "mode": self.mode,
                "core_mode": self.core.mode,
                "branch_off": False,
                "pair_scores": pair_scores,
                "residual": residual,
                **local_trace,
                **residual_trace,
            }
        if return_trace:
            trace = {
                **trace,
                "input_tensor": "final_block_norm1_patch_tokens",
                "residual_bound_reference": (
                    "frozen_post_final_mhsa_patch_tokens_u"
                ),
                "merge_point": "parallel_final_mhsa_before_final_mlp_and_norm",
            }
            return residual, trace
        return residual

    def forward(
        self,
        z_patch: Tensor,
        residual_reference: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool = False,
    ) -> Tensor:
        return self.residual_from_final_norm1(
            z_patch,
            residual_reference,
            classifier_weight,
            branch_off=bool(branch_off),
            return_trace=False,
        )

    def merge_parallel_final_mhsa(
        self,
        post_mhsa_tokens: Tensor,
        z_patch: Tensor,
        classifier_weight: Tensor,
        *,
        branch_off: bool = False,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """Add the patch residual to the frozen post-final-MHSA token stream."""

        expected_tail = (XCNORM_PAIR_A1_TOTAL_TOKEN_COUNT, XCNORM_PAIR_EMBED_DIM)
        if post_mhsa_tokens.ndim != 3 or tuple(post_mhsa_tokens.shape[1:]) != (
            expected_tail
        ):
            raise DinoV3XCNormPairA1ContractError(
                "post_mhsa_tokens must have shape "
                f"[B,{expected_tail[0]},{expected_tail[1]}]; got "
                f"{tuple(post_mhsa_tokens.shape)}."
            )
        if int(post_mhsa_tokens.size(0)) != int(z_patch.size(0)):
            raise DinoV3XCNormPairA1ContractError(
                "post_mhsa_tokens and z_patch batch sizes must match."
            )
        frozen_post_mhsa = post_mhsa_tokens.detach()
        residual_reference = frozen_post_mhsa[
            :, XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT :, :
        ]
        residual_result = self.residual_from_final_norm1(
            z_patch,
            residual_reference,
            classifier_weight,
            branch_off=bool(branch_off),
            return_trace=bool(return_trace),
        )
        if return_trace:
            residual, trace = residual_result
        else:
            residual = residual_result
            trace = {}
        if branch_off:
            merged = frozen_post_mhsa
        else:
            prefix_zeros = torch.zeros(
                residual.size(0),
                XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT,
                residual.size(2),
                device=residual.device,
                dtype=residual.dtype,
            )
            merged = frozen_post_mhsa + torch.cat((prefix_zeros, residual), dim=1)
        if return_trace:
            trace["parallel_mhsa_residual"] = residual
            trace["merged_post_mhsa_tokens"] = merged
            trace["prefix_tokens_modified_at_merge"] = False
            return merged, trace
        return merged

    def added_parameter_count(self, *, trainable_only: bool = False) -> int:
        return self.core.added_parameter_count(trainable_only=trainable_only)

    def parameter_telemetry(self) -> Dict[str, int]:
        return self.core.parameter_telemetry()

    def fusion_runtime_parameter_telemetry(self) -> Dict[str, float]:
        return self.core.fusion_runtime_parameter_telemetry()

    def fusion_provenance(self) -> Dict[str, Any]:
        core = self.core.fusion_provenance()
        return {
            **core,
            "architecture": type(self).__name__,
            "mode": self.mode,
            "reused_core_mode": self.core.mode,
            "input": "frozen_final_block_norm1_patch_tokens",
            "placement": "parallel_to_frozen_final_mhsa",
            "merge_point": (
                "post_final_mhsa_residual_before_frozen_final_mlp_and_final_norm"
            ),
            "backbone_contract": "frozen_eval",
            "classifier_contract": "frozen_and_explicitly_bound",
            "classifier_axis_semantics": (
                "fixed_injection_space_prior_not_exact_output_margin_mapping_"
                "after_frozen_mlp_and_final_norm"
            ),
            "local_score_input": (
                "z_patch=frozen_final_block_norm1_patch_tokens"
            ),
            "residual_bound_reference": (
                "u_patch=frozen_post_final_mhsa_residual_patch_tokens"
            ),
            "cache_precheck_attestation": (
                "u_full=x_pre+drop_path1(gamma_1*MHSA(z_full,rope));"
                "z_full=norm1(x_pre)"
            ),
            "cache_precheck_required_fields": (
                "final_block_gamma_1_hash,final_block_gamma_2_hash,"
                "branch_off_logit_parity"
            ),
            "downstream_frozen_contract": (
                "x_out=(u+r)+drop_path2(gamma_2*MLP(norm2(u+r)));"
                "y=final_norm(x_out)"
            ),
            "residual_ratio_cap_semantics": (
                "norm(r_patch)_2/norm(u_patch)_2<0.04_per_patch"
            ),
            "pooling_contract": (
                "frozen_final_mlp_then_final_norm_then_mean_patch_pool_then_head"
            ),
            "branch_off": "exact_zero_residual_and_exact_frozen_stream_identity",
            "expected_added_parameter_count": int(
                XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT
            ),
            "observed_added_parameter_count": self.added_parameter_count(),
            "scientific_scope": "final_mhsa_parallel_local_token_pattern_screen",
        }

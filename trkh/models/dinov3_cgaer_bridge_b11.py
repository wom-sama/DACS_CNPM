from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


CGAER_B11_CANDIDATE_MODE = "cgaer_candidate_b11"
CGAER_B11_CONTROL_MODE = "categorical_control_b11"
SUPPORTED_CGAER_B11_MODES = (
    CGAER_B11_CANDIDATE_MODE,
    CGAER_B11_CONTROL_MODE,
)
CGAER_B11_EMBED_DIM = 384
CGAER_B11_NUM_CLASSES = 5
CGAER_B11_HIDDEN_DIM = 8
CGAER_B11_PARAMETER_COUNT = 3_125
CGAER_B11_DENSE_MACS = 3_112
CGAER_B11_RESIDUAL_L2_CAP = 0.5

class DinoV3CGAERBridgeB11(nn.Module):
    """Conflict-gated adjacent-energy residual on frozen B9 outputs."""

    def __init__(self, mode: str = CGAER_B11_CANDIDATE_MODE) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_CGAER_B11_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_CGAER_B11_MODES!r}; "
                f"got {mode!r}."
            )
        self.mode = resolved_mode
        self.input_norm = nn.LayerNorm(
            CGAER_B11_EMBED_DIM,
            elementwise_affine=False,
        )
        self.input_projection = nn.Linear(
            CGAER_B11_EMBED_DIM,
            CGAER_B11_HIDDEN_DIM,
        )
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(
            CGAER_B11_HIDDEN_DIM,
            CGAER_B11_NUM_CLASSES,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        centers = torch.tensor((-1.0, 0.0, 1.0))
        stage = torch.tensor((-1.0, 0.0, 1.0)) / math.sqrt(2.0)
        curvature = torch.tensor((-1.0, 2.0, -1.0)) / math.sqrt(6.0)
        ordered_basis = torch.stack((stage, curvature), dim=1)
        ordered_gains = torch.tensor(
            (8.0 * math.sqrt(2.0) / 9.0, math.sqrt(6.0) / 27.0)
        )
        cluster_vs_34 = torch.tensor(
            (
                math.sqrt(2.0 / 15.0),
                math.sqrt(2.0 / 15.0),
                math.sqrt(2.0 / 15.0),
                -math.sqrt(3.0 / 10.0),
                -math.sqrt(3.0 / 10.0),
            )
        )
        class_3_vs_4 = torch.tensor(
            (0.0, 0.0, 0.0, 1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0))
        )
        self.register_buffer("ordered_centers", centers)
        self.register_buffer("ordered_basis", ordered_basis)
        self.register_buffer("ordered_gains", ordered_gains)
        self.register_buffer(
            "G",
            torch.stack((cluster_vs_34, class_3_vs_4), dim=1),
        )

        if self.added_parameter_count() != CGAER_B11_PARAMETER_COUNT:
            raise RuntimeError("B11 trainable capacity contract was violated.")

    @staticmethod
    def _validate_pair(first: Tensor, second: Tensor, first_dim: int) -> None:
        if first.ndim != 2 or int(first.size(1)) != first_dim:
            raise ValueError(f"first input must have shape [B,{first_dim}].")
        if second.ndim != 2 or int(second.size(1)) != CGAER_B11_NUM_CLASSES:
            raise ValueError("base_logits must have shape [B,5].")
        if int(first.size(0)) != int(second.size(0)):
            raise ValueError("input batch sizes must match.")
        if first.device != second.device or first.dtype != second.dtype:
            raise ValueError("inputs must have the same device and dtype.")
        if not first.is_floating_point():
            raise ValueError("inputs must use a floating-point dtype.")
        if not bool(torch.isfinite(first).all()) or not bool(
            torch.isfinite(second).all()
        ):
            raise ValueError("inputs must be finite.")

    @staticmethod
    def _forced_gate(raw_gate: Tensor, force_gate: float | None) -> Tensor:
        if force_gate is None:
            return torch.sigmoid(raw_gate)
        value = float(force_gate)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("force_gate must be finite and in [0,1].")
        return torch.full_like(raw_gate, value)

    def candidate_ordered_energy(
        self,
        raw_location: Tensor,
        raw_spread: Tensor,
        gate: Tensor,
    ) -> Tensor:
        location = 2.0 * torch.tanh(raw_location)
        alpha = 1.0 + 0.5 * torch.tanh(raw_spread)
        temperature = 1.0 + gate
        centers = self.ordered_centers.to(raw_location)
        phi = -alpha.unsqueeze(-1) * torch.square(
            location.unsqueeze(-1) - centers
        ) / (2.0 * torch.square(temperature.unsqueeze(-1)))
        baseline = -torch.square(centers) / (2.0 * (1.5**2))
        delta = phi - baseline
        return delta - delta.mean(dim=-1, keepdim=True)

    def control_ordered_contrasts(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or int(coordinates.size(1)) != 2:
            raise ValueError("control coordinates must have shape [B,2].")
        scaled_basis = self.ordered_basis * self.ordered_gains.unsqueeze(0)
        return coordinates @ scaled_basis.transpose(0, 1)

    def fixed_categorical_basis(self) -> Tensor:
        return self.G.detach().clone()

    def ordered_basis_gains(self) -> Tensor:
        return self.ordered_gains.detach().clone()

    def uncapped_residual_from_raw_outputs(
        self,
        raw_outputs: Tensor,
        base_logits: Tensor,
        *,
        force_gate: float | None = None,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        self._validate_pair(raw_outputs, base_logits, CGAER_B11_NUM_CLASSES)
        gate = self._forced_gate(raw_outputs[:, 4], force_gate)
        if self.mode == CGAER_B11_CANDIDATE_MODE:
            ordered = self.candidate_ordered_energy(
                raw_outputs[:, 0], raw_outputs[:, 1], gate
            )
        else:
            ordered = self.control_ordered_contrasts(raw_outputs[:, :2])

        centered_base = base_logits[:, :3] - base_logits[:, :3].mean(
            dim=-1, keepdim=True
        )
        flattening = -(gate - 0.5).unsqueeze(-1) * centered_base
        cluster_residual = ordered + flattening
        ordered_five = torch.cat(
            (cluster_residual, torch.zeros_like(base_logits[:, 3:])), dim=-1
        )
        categorical = raw_outputs[:, 2:4] @ self.G.transpose(0, 1)
        residual = ordered_five + categorical
        if not return_trace:
            return residual
        return residual, {
            "mode": self.mode,
            "raw_outputs": raw_outputs,
            "conflict_logit": raw_outputs[:, 4],
            "gate": gate,
            "forced_gate": force_gate is not None,
            "ordered_residual_012": ordered,
            "base_ambiguity_flattening_012": flattening,
            "categorical_residual": categorical,
            "uncapped_residual": residual,
        }

    def residual_from_raw_outputs(
        self,
        raw_outputs: Tensor,
        base_logits: Tensor,
        *,
        force_gate: float | None = None,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        result = self.uncapped_residual_from_raw_outputs(
            raw_outputs,
            base_logits,
            force_gate=force_gate,
            return_trace=True,
        )
        uncapped, trace = result
        # Keep the trust-region arithmetic in FP32 so a finite, large FP16/
        # BF16 vector cannot overflow its squared norm and collapse to zero.
        low_precision = uncapped.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        cap_work = uncapped.float() if low_precision else uncapped
        uncapped_norm = torch.linalg.vector_norm(
            cap_work, dim=-1, keepdim=True
        )
        scale = torch.rsqrt(
            1.0 + torch.square(uncapped_norm / CGAER_B11_RESIDUAL_L2_CAP)
        )
        if low_precision:
            # Leave one FP16-ulp-scale guard below the mathematical cap after
            # casting the five-vector back to its inference dtype.
            scale = scale * (1.0 - 2.0**-10)
        residual = (cap_work * scale).to(dtype=uncapped.dtype)
        if not return_trace:
            return residual
        trace.update(
            {
                "residual": residual,
                "uncapped_residual_l2_norm": uncapped_norm.squeeze(-1),
                "residual_l2_norm": torch.linalg.vector_norm(
                    residual.float(), dim=-1
                ),
                "residual_scale": scale.squeeze(-1),
            }
        )
        return residual, trace

    def forward(
        self,
        z: Tensor,
        base_logits: Tensor,
        *,
        branch_off: bool = False,
        force_gate: float | None = None,
        return_trace: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        self._validate_pair(z, base_logits, CGAER_B11_EMBED_DIM)
        frozen_base = base_logits.detach()
        if branch_off:
            output = frozen_base
            if not return_trace:
                return output
            zero = torch.zeros_like(frozen_base)
            return output, {
                "mode": self.mode,
                "branch_off": True,
                "raw_outputs": zero,
                "conflict_logit": zero[:, 4],
                "gate": torch.full_like(zero[:, 4], 0.5),
                "uncapped_residual": zero,
                "residual": zero,
                "uncapped_residual_l2_norm": zero[:, 0],
                "residual_l2_norm": zero[:, 0],
            }

        hidden = self.activation(self.input_projection(self.input_norm(z.detach())))
        raw_outputs = self.output_projection(hidden)
        residual, trace = self.residual_from_raw_outputs(
            raw_outputs,
            frozen_base,
            force_gate=force_gate,
            return_trace=True,
        )
        output = frozen_base + residual
        if return_trace:
            trace["branch_off"] = False
            return output, trace
        return output

    def added_parameter_count(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(int(parameter.numel()) for parameter in parameters)

    def dense_macs(self) -> int:
        return CGAER_B11_DENSE_MACS

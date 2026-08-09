"""Block-local task-vector consolidation with learned elementwise coefficients.

The module keeps a frozen initialization block and two frozen endpoint task
vectors.  Only the two coefficient tensors are optimized.  After optimization,
``materialize_state_dict`` returns an ordinary block state, so inference keeps
the original graph and parameter count.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.func import functional_call


class ProgressiveTaskVectorContractError(RuntimeError):
    """Raised when base and endpoint blocks do not share an exact state schema."""


def _detached_state(state: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
    return OrderedDict(
        (name, value.detach().clone())
        for name, value in state.items()
    )


class ElementwiseTaskVectorBlock(nn.Module):
    """Learn ``base + alpha_a * delta_a + alpha_b * delta_b`` for one block."""

    def __init__(
        self,
        base_block: nn.Module,
        endpoint_a_state: Mapping[str, Tensor],
        endpoint_b_state: Mapping[str, Tensor],
        *,
        coefficient_init: float = 0.5,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(coefficient_init)):
            raise ValueError("coefficient_init must be finite")

        self.base_block = copy.deepcopy(base_block).eval().requires_grad_(False)
        base_state = _detached_state(self.base_block.state_dict())
        endpoint_a = _detached_state(endpoint_a_state)
        endpoint_b = _detached_state(endpoint_b_state)
        if set(endpoint_a) != set(base_state) or set(endpoint_b) != set(base_state):
            raise ProgressiveTaskVectorContractError(
                "Base and endpoint state keys must match exactly."
            )

        parameter_names = {name for name, _ in self.base_block.named_parameters()}
        self._learned_names: list[str] = []
        self._static_names: list[str] = []
        self.delta_a = nn.ParameterList()
        self.delta_b = nn.ParameterList()
        self.alpha_a = nn.ParameterList()
        self.alpha_b = nn.ParameterList()

        for name, base in base_state.items():
            a = endpoint_a[name]
            b = endpoint_b[name]
            if a.shape != base.shape or b.shape != base.shape:
                raise ProgressiveTaskVectorContractError(
                    f"State shape mismatch for {name!r}: "
                    f"base={tuple(base.shape)}, a={tuple(a.shape)}, b={tuple(b.shape)}"
                )
            if a.dtype != base.dtype or b.dtype != base.dtype:
                raise ProgressiveTaskVectorContractError(
                    f"State dtype mismatch for {name!r}."
                )
            if name not in parameter_names:
                if not torch.equal(a, base) or not torch.equal(b, base):
                    raise ProgressiveTaskVectorContractError(
                        f"Non-parameter state {name!r} differs across endpoints."
                    )
                self._static_names.append(name)
                continue
            if not base.is_floating_point():
                raise ProgressiveTaskVectorContractError(
                    f"Learned parameter {name!r} is not floating point."
                )
            delta_a = a - base
            delta_b = b - base
            if not bool(torch.count_nonzero(delta_a).item()) and not bool(
                torch.count_nonzero(delta_b).item()
            ):
                self._static_names.append(name)
                continue
            self._learned_names.append(name)
            # Frozen task vectors are Parameters solely so device/dtype moves are
            # handled by nn.Module without a parallel buffer-name registry.
            self.delta_a.append(nn.Parameter(delta_a, requires_grad=False))
            self.delta_b.append(nn.Parameter(delta_b, requires_grad=False))
            self.alpha_a.append(
                nn.Parameter(torch.full_like(base, float(coefficient_init)))
            )
            self.alpha_b.append(
                nn.Parameter(torch.full_like(base, float(coefficient_init)))
            )

        if not self._learned_names:
            raise ProgressiveTaskVectorContractError("The block has no learnable parameters.")

    @property
    def learned_state_names(self) -> tuple[str, ...]:
        return tuple(self._learned_names)

    def coefficient_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.alpha_a) + tuple(self.alpha_b)

    def _merged_state(self) -> OrderedDict[str, Tensor]:
        state = OrderedDict(self.base_block.state_dict())
        for index, name in enumerate(self._learned_names):
            base = state[name]
            state[name] = (
                base
                + self.alpha_a[index] * self.delta_a[index]
                + self.alpha_b[index] * self.delta_b[index]
            )
        return state

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        return functional_call(
            self.base_block,
            self._merged_state(),
            args=args,
            kwargs=kwargs,
            strict=True,
        )

    @torch.no_grad()
    def materialize_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in self._merged_state().items()
        )

    @torch.no_grad()
    def coefficient_summary(self) -> dict[str, Any]:
        def summarize(parameters: nn.ParameterList) -> dict[str, float]:
            total = sum(int(value.numel()) for value in parameters)
            total_sum = sum(float(value.detach().double().sum().cpu()) for value in parameters)
            minimum = min(float(value.detach().min().cpu()) for value in parameters)
            maximum = max(float(value.detach().max().cpu()) for value in parameters)
            return {
                "count": int(total),
                "mean": total_sum / float(total),
                "min": minimum,
                "max": maximum,
            }

        return {
            "endpoint_a": summarize(self.alpha_a),
            "endpoint_b": summarize(self.alpha_b),
            "learned_state_count": len(self._learned_names),
        }

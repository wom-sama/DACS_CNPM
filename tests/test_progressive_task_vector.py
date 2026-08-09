from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.func import functional_call

from trkh.training.progressive_task_vector import (
    ElementwiseTaskVectorBlock,
    ProgressiveTaskVectorContractError,
)


def _endpoint(module: nn.Module, *, offset: float) -> dict[str, torch.Tensor]:
    state = copy.deepcopy(module.state_dict())
    for name, value in state.items():
        if name.endswith("weight") or name.endswith("bias"):
            state[name] = value + float(offset)
    return state


def test_initial_function_is_exact_endpoint_midpoint() -> None:
    torch.manual_seed(7)
    base = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2))
    endpoint_a = _endpoint(base, offset=0.2)
    endpoint_b = _endpoint(base, offset=-0.1)
    merger = ElementwiseTaskVectorBlock(base, endpoint_a, endpoint_b)
    inputs = torch.randn(4, 3)

    expected_state = {
        name: base.state_dict()[name] + 0.5 * (endpoint_a[name] - base.state_dict()[name])
        + 0.5 * (endpoint_b[name] - base.state_dict()[name])
        for name in base.state_dict()
    }
    expected = functional_call(base, expected_state, (inputs,), strict=True)
    assert torch.equal(merger(inputs), expected)


def test_coefficients_reduce_loss_and_materialize_exactly() -> None:
    torch.manual_seed(11)
    base = nn.Linear(4, 3)
    endpoint_a = _endpoint(base, offset=0.4)
    endpoint_b = _endpoint(base, offset=-0.3)
    merger = ElementwiseTaskVectorBlock(base, endpoint_a, endpoint_b)
    inputs = torch.randn(16, 4)
    target = functional_call(base, endpoint_a, (inputs,), strict=True).detach()
    optimizer = torch.optim.Adam(merger.coefficient_parameters(), lr=0.05)

    initial = float(torch.nn.functional.mse_loss(merger(inputs), target).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(merger(inputs), target)
        loss.backward()
        optimizer.step()
    final = float(torch.nn.functional.mse_loss(merger(inputs), target).detach())

    replay = functional_call(base, merger.materialize_state_dict(), (inputs,), strict=True)
    assert final < initial * 0.1
    assert torch.equal(merger(inputs), replay)
    assert all(parameter.grad is None for parameter in merger.delta_a)
    assert all(parameter.grad is None for parameter in merger.delta_b)


def test_rejects_key_shape_and_nonparameter_drift() -> None:
    base = nn.BatchNorm1d(3)
    endpoint_a = copy.deepcopy(base.state_dict())
    endpoint_b = copy.deepcopy(base.state_dict())
    endpoint_a.pop("bias")
    with pytest.raises(ProgressiveTaskVectorContractError, match="keys"):
        ElementwiseTaskVectorBlock(base, endpoint_a, endpoint_b)

    endpoint_a = copy.deepcopy(base.state_dict())
    endpoint_a["weight"] = torch.ones(4)
    with pytest.raises(ProgressiveTaskVectorContractError, match="shape mismatch"):
        ElementwiseTaskVectorBlock(base, endpoint_a, endpoint_b)

    endpoint_a = copy.deepcopy(base.state_dict())
    endpoint_a["running_mean"] = torch.ones_like(endpoint_a["running_mean"])
    with pytest.raises(ProgressiveTaskVectorContractError, match="Non-parameter"):
        ElementwiseTaskVectorBlock(base, endpoint_a, endpoint_b)


def test_identical_endpoint_parameters_do_not_allocate_coefficients() -> None:
    base = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    endpoint_a = copy.deepcopy(base.state_dict())
    endpoint_b = copy.deepcopy(base.state_dict())
    endpoint_a["0.weight"] = endpoint_a["0.weight"] + 0.1
    endpoint_b["0.weight"] = endpoint_b["0.weight"] - 0.1
    merger = ElementwiseTaskVectorBlock(base, endpoint_a, endpoint_b)
    assert merger.learned_state_names == ("0.weight",)
    assert len(merger.coefficient_parameters()) == 2

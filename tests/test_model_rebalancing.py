from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from trkh.models.model_rebalancing import (
    freeze_model_rebalancing_base,
    inject_model_rebalancing_convs,
    merge_model_rebalancing_convs_,
    model_rebalancing_general_only,
    model_rebalancing_modules,
    model_rebalancing_tail_sha256,
    more_discrepancy_loss,
    more_sinusoidal_weight,
)


class TinyConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
        )
        self.head = nn.Linear(12, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).mean(dim=(-2, -1))
        return self.head(features)


def _model_and_inputs():
    torch.manual_seed(7)
    model = TinyConvClassifier().eval()
    images = torch.randn(5, 3, 16, 16)
    return model, images


def test_zero_tail_is_keeper_exact_and_initialization_is_deterministic() -> None:
    base, images = _model_and_inputs()
    first = copy.deepcopy(base)
    second = copy.deepcopy(base)
    expected = base(images)

    first_records = inject_model_rebalancing_convs(first, rank_ratio=0.1, seed=42)
    second_records = inject_model_rebalancing_convs(second, rank_ratio=0.1, seed=42)

    assert [record.name for record in first_records] == ["features.0", "features.2"]
    assert first_records == second_records
    assert torch.equal(first(images), expected)
    assert model_rebalancing_tail_sha256(first) == model_rebalancing_tail_sha256(second)
    assert all(
        torch.count_nonzero(module.tail_b.weight).item() == 0
        for module in model_rebalancing_modules(first).values()
    )


def test_general_only_restores_state_and_stays_on_base_path() -> None:
    base, images = _model_and_inputs()
    expected = base(images)
    inject_model_rebalancing_convs(base, rank_ratio=0.25, seed=42)
    for module in model_rebalancing_modules(base).values():
        nn.init.normal_(module.tail_b.weight, std=0.02)
    full = base(images)
    with model_rebalancing_general_only(base):
        general = base(images)
        assert all(not module.tail_enabled for module in model_rebalancing_modules(base).values())
    assert all(module.tail_enabled for module in model_rebalancing_modules(base).values())
    assert torch.equal(general, expected)
    assert not torch.equal(full, general)


def test_tail_gradients_become_finite_for_both_factors() -> None:
    model, images = _model_and_inputs()
    inject_model_rebalancing_convs(model, rank_ratio=0.25, seed=42)
    tail_parameters = freeze_model_rebalancing_base(model)
    optimizer = torch.optim.SGD(tail_parameters, lr=0.2)
    targets = torch.tensor([0, 1, 2, 1, 0])

    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(images), targets).backward()
    first_b_gradients = [module.tail_b.weight.grad for module in model_rebalancing_modules(model).values()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in first_b_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in first_b_gradients)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(images), targets).backward()
    for module in model_rebalancing_modules(model).values():
        assert module.tail_a.weight.grad is not None
        assert module.tail_b.weight.grad is not None
        assert torch.isfinite(module.tail_a.weight.grad).all()
        assert torch.isfinite(module.tail_b.weight.grad).all()
    assert all(not parameter.requires_grad for parameter in model.head.parameters())


def test_merge_replaces_wrappers_and_preserves_outputs() -> None:
    model, images = _model_and_inputs()
    original_keys = set(model.state_dict())
    inject_model_rebalancing_convs(model, rank_ratio=0.25, seed=42)
    for module in model_rebalancing_modules(model).values():
        nn.init.normal_(module.tail_b.weight, std=0.01)
    before = model(images)

    merged_names = merge_model_rebalancing_convs_(model)
    after = model(images)

    assert merged_names == ["features.0", "features.2"]
    assert not model_rebalancing_modules(model)
    assert set(model.state_dict()) == original_keys
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)


def test_discrepancy_uses_natural_priors_and_has_gradients() -> None:
    full = torch.tensor([[2.0, 0.0], [0.2, 1.3]], requires_grad=True)
    general = torch.tensor([[1.0, 0.0], [0.2, 0.3]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss, telemetry = more_discrepancy_loss(
        full,
        general,
        targets,
        class_counts=[9, 1],
    )
    assert telemetry["class_priors"].tolist() == pytest.approx([0.9, 0.1])
    assert telemetry["sample_weights"].tolist() == pytest.approx([0.9, 0.1])
    assert float(loss.item()) == pytest.approx(0.5)
    loss.backward()
    assert full.grad is not None and torch.isfinite(full.grad).all()
    assert general.grad is not None and torch.isfinite(general.grad).all()


def test_sinusoidal_weight_starts_and_ends_at_zero() -> None:
    assert more_sinusoidal_weight(0, 10, peak_amplitude=10.0) == pytest.approx(0.0)
    assert more_sinusoidal_weight(5, 10, peak_amplitude=10.0) == pytest.approx(10.0)
    assert more_sinusoidal_weight(10, 10, peak_amplitude=10.0) == pytest.approx(0.0)


def test_injection_rejects_grouped_convolution() -> None:
    model = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4))
    with pytest.raises(ValueError, match="Grouped convolution"):
        inject_model_rebalancing_convs(model)

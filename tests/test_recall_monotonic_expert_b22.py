from __future__ import annotations

import pytest
import torch

from trkh.models.recall_monotonic_expert_b22 import (
    BIAS_ONLY,
    FEATURE_LOGITS,
    LOGITS_ONLY,
    RecallMonotonicLogitExpert,
)


@pytest.mark.parametrize(
    ("mode", "expected_parameters"),
    ((BIAS_ONLY, 1), (LOGITS_ONLY, 6), (FEATURE_LOGITS, 390)),
)
def test_parameter_contract(mode: str, expected_parameters: int) -> None:
    expert = RecallMonotonicLogitExpert(
        feature_dim=384,
        num_classes=5,
        mode=mode,
    )
    assert expert.parameter_count() == expected_parameters


@pytest.mark.parametrize("mode", (BIAS_ONLY, LOGITS_ONLY, FEATURE_LOGITS))
def test_correction_is_bounded_and_only_changes_focus_logit(mode: str) -> None:
    torch.manual_seed(7)
    features = torch.randn(11, 384)
    logits = torch.randn(11, 5)
    expert = RecallMonotonicLogitExpert(
        feature_dim=384,
        num_classes=5,
        mode=mode,
        max_correction=3.0,
    )
    adjusted, correction = expert(features, logits)
    assert bool((correction >= 0.0).all())
    assert bool((correction <= 3.0).all())
    assert torch.equal(adjusted[:, 0], logits[:, 0])
    assert torch.equal(adjusted[:, 2:], logits[:, 2:])
    assert torch.equal(adjusted[:, 1], logits[:, 1] + correction)


def test_existing_focus_predictions_are_preserved_for_any_parameters() -> None:
    torch.manual_seed(11)
    features = torch.randn(17, 384)
    logits = torch.randn(17, 5)
    logits[:, 1] = logits.max(dim=1).values + 0.25
    expert = RecallMonotonicLogitExpert(
        feature_dim=384,
        num_classes=5,
        mode=FEATURE_LOGITS,
    )
    with torch.no_grad():
        expert.linear.weight.normal_()
        expert.linear.bias.normal_()
    adjusted, _ = expert(features, logits)
    assert torch.equal(logits.argmax(dim=1), torch.ones(17, dtype=torch.long))
    assert torch.equal(adjusted.argmax(dim=1), torch.ones(17, dtype=torch.long))


def test_small_nonzero_initialization_can_reach_feature_weights() -> None:
    torch.manual_seed(13)
    features = torch.randn(23, 384)
    logits = torch.randn(23, 5)
    expert = RecallMonotonicLogitExpert(
        feature_dim=384,
        num_classes=5,
        mode=FEATURE_LOGITS,
    )
    adjusted, correction = expert(features, logits)
    assert float(correction.max()) < 0.002
    adjusted[:, 1].sum().backward()
    assert expert.linear.weight.grad is not None
    assert int(torch.count_nonzero(expert.linear.weight.grad)) > 0


def test_invalid_shapes_fail_closed() -> None:
    expert = RecallMonotonicLogitExpert(
        feature_dim=384,
        num_classes=5,
        mode=LOGITS_ONLY,
    )
    with pytest.raises(ValueError):
        expert(torch.randn(2, 383), torch.randn(2, 5))


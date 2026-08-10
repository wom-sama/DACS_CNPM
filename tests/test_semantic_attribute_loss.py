import pytest
import torch

from trkh.training.train import (
    _parse_semantic_attribute_specs,
    _semantic_attribute_loss_from_logits,
)


def test_parse_semantic_attribute_specs_rejects_duplicate_class():
    with pytest.raises(ValueError, match="lap class"):
        _parse_semantic_attribute_specs("bad:0,1|1,2", num_classes=5)


def test_semantic_attribute_loss_prefers_correct_group_mass():
    targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    specs = "maturity:0,1|2,3|4;transport:0,2|1|3,4"
    good_logits = torch.full((5, 5), -3.0)
    good_logits[0, 0] = 3.0
    good_logits[1, 1] = 3.0
    good_logits[2, 2] = 3.0
    good_logits[3, 3] = 3.0
    good_logits[4, 4] = 3.0
    bad_logits = -good_logits

    good_loss, good_tasks = _semantic_attribute_loss_from_logits(
        good_logits,
        targets,
        specs,
    )
    bad_loss, bad_tasks = _semantic_attribute_loss_from_logits(
        bad_logits,
        targets,
        specs,
    )

    assert good_tasks == 2.0
    assert bad_tasks == 2.0
    assert good_loss.item() < bad_loss.item()
    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)


def test_semantic_attribute_loss_backpropagates():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)

    loss, tasks = _semantic_attribute_loss_from_logits(
        logits,
        targets,
        "quality:0,1,2|3|4",
    )
    loss.backward()

    assert tasks == 1.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0

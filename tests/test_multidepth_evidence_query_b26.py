from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trkh.models.model import build_model_from_checkpoint
from trkh.models.multidepth_evidence_query_b26 import (
    MultiDepthEvidenceQuery,
    boundary_guarded_loss,
    configure_query_only,
    is_query_parameter,
    load_query_state_dict,
    query_state_dict,
)


CHECKPOINT = Path(
    r"D:\DataAI\AIEx\TRKH_pretrained\runs"
    r"\pretrained_dinov3_classf_b9_b2_reference_completion_full_20260801_b9_b2_ref_full_r1"
    r"\checkpoints\best.pt"
)


def _backbone() -> nn.Module:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return build_model_from_checkpoint(checkpoint).eval()


def test_b26_preserves_b9_at_step_zero_and_query_state_reloads() -> None:
    backbone = _backbone()
    model = MultiDepthEvidenceQuery(backbone).eval()
    counts = configure_query_only(model)
    assert counts == {
        "total": 21_647_622,
        "trainable": 58_753,
        "frozen": 21_588_869,
    }
    images = torch.randn(2, 3, 256, 256)
    with torch.inference_mode():
        native_features = backbone.forward_features(images)
        native = backbone.forward_head(native_features).float()
        logits, trace = model.forward_with_trace(images)
    assert torch.equal(logits, native)
    assert tuple(trace["class_to_patch_attention"].shape) == (2, 3, 4, 5, 256)
    assert torch.allclose(
        trace["class_to_patch_attention"].sum(dim=-1),
        torch.ones_like(trace["class_to_patch_attention"].sum(dim=-1)),
        atol=1e-6,
    )
    state = query_state_dict(model)
    clone = MultiDepthEvidenceQuery(_backbone()).eval()
    load_query_state_dict(clone, state)
    with torch.inference_mode():
        observed = clone(images)
    assert torch.equal(observed, logits)


def test_b26_two_steps_reach_decoder_but_never_backbone() -> None:
    model = MultiDepthEvidenceQuery(_backbone()).eval()
    configure_query_only(model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.005,
    )
    images = torch.randn(2, 3, 256, 256)
    targets = torch.tensor([1, 2])
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits, trace = model.forward_with_trace(images, return_attention=False)
        loss, parts = boundary_guarded_loss(
            logits, trace["base_logits"], targets, nn.CrossEntropyLoss()
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in parts.values())
        assert all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not is_query_parameter(name)
        )
        optimizer.step()
    gradients = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    }
    assert "class_queries" in gradients
    assert any(name.startswith("evidence_query.patch_projection") for name in gradients)
    assert any(name.startswith("evidence_query.cross_attention") for name in gradients)

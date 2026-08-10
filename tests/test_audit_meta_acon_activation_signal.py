from __future__ import annotations

import os

import numpy as np
import torch

from trkh.tools.audit_meta_acon_activation_signal import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    CONDITIONS,
    EPOCHS,
    LEARNING_RATE,
    META_OBJECT_BETA_ROLE,
    REDUCTION,
    ROLES,
    WEIGHT_DECAY,
    AconC,
    MetaAconC,
    ObjectActivationReadout,
    _independent_meta_acon,
    _load_module,
    assess_information_gate,
    object_mask_from_bbox,
    parse_args,
    select_recall_constrained_threshold,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.epochs == EPOCHS == 12
    assert args.learning_rate == LEARNING_RATE == 0.003
    assert args.weight_decay == WEIGHT_DECAY == 0.0001
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 3
    assert ROLES == ("identity", "acon_static", "meta_acon")
    assert len(CONDITIONS) == 4


def test_paper_initialization_and_independent_equation_match() -> None:
    torch.manual_seed(17)
    module = MetaAconC(16, reduction=REDUCTION).double().eval()
    assert torch.equal(module.p1, torch.ones_like(module.p1))
    assert torch.equal(module.p2, torch.zeros_like(module.p2))
    x = torch.randn(3, 16, 7, 9, dtype=torch.float64)
    observed, beta = module.forward_with_beta(x)
    expected, expected_beta = _independent_meta_acon(module, x)
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(beta, expected_beta, atol=1e-12, rtol=0.0)

    static = AconC(16).double()
    delta = static.p1 * x - static.p2 * x
    expected_static = delta * torch.sigmoid(static.beta * delta) + static.p2 * x
    assert torch.allclose(static(x), expected_static, atol=1e-12, rtol=0.0)


def test_official_source_loader_does_not_write_bytecode(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 7\n", encoding="utf-8")
    module = _load_module(source)
    assert module.VALUE == 7
    assert not (tmp_path / "__pycache__").exists()


def test_object_mask_and_identity_readout_ignore_outside_pixels() -> None:
    torch.manual_seed(19)
    model = ObjectActivationReadout("identity", width=8).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(1, 8, 16, 16)
    mask = object_mask_from_bbox(boxes, 16, 16)
    changed = x.clone()
    changed[:, :, ~mask[0]] += 100.0
    first = model(x, boxes)[0]
    second = model(changed, boxes)[0]
    assert torch.equal(first, second)


def test_object_only_beta_is_invariant_to_outside_pixels() -> None:
    torch.manual_seed(23)
    model = ObjectActivationReadout("meta_acon", width=16).eval()
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(1, 16, 16, 16)
    mask = object_mask_from_bbox(boxes, 16, 16)
    changed = x.clone()
    changed[:, :, ~mask[0]] -= 50.0
    first_logit, first_beta, _ = model(x, boxes, object_only_beta=True)
    second_logit, second_beta, _ = model(
        changed, boxes, object_only_beta=True
    )
    assert first_beta is not None and second_beta is not None
    assert torch.allclose(first_beta, second_beta, atol=1e-6, rtol=0.0)
    assert torch.allclose(first_logit, second_logit, atol=1e-6, rtol=0.0)


def test_meta_acon_gradient_families_are_live() -> None:
    torch.manual_seed(29)
    model = ObjectActivationReadout("meta_acon", width=16).train()
    x = torch.randn(8, 16, 12, 12, requires_grad=True)
    boxes = torch.tensor([[0.5, 0.5, 0.6, 0.6]], dtype=torch.float32).repeat(8, 1)
    target = torch.tensor([1.0, 0.0] * 4)
    logits, _, _ = model(x, boxes)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = {name: value.grad for name, value in model.named_parameters()}
    for name in (
        "activation.p1",
        "activation.p2",
        "activation.fc1.weight",
        "activation.fc2.weight",
        "head.weight",
    ):
        assert gradients[name] is not None
        assert torch.isfinite(gradients[name]).all()
        assert torch.count_nonzero(gradients[name]) > 0


def test_recall_constrained_threshold_uses_highest_valid_score() -> None:
    scores = np.asarray([0.95, 0.90, 0.80, 0.70, 0.60, 0.55], dtype=np.float64)
    labels = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.int64)
    threshold = select_recall_constrained_threshold(
        scores, labels, min_tp_retention=0.80
    )
    assert threshold == 0.70
    assert np.mean(scores[labels == 1] >= threshold) == 0.80


def _metric(
    auroc: float,
    *,
    tp: float = 0.92,
    fp: float = 0.25,
    tp_median: float = 0.70,
    fp_median: float = 0.30,
) -> dict[str, float]:
    return {
        "auroc": auroc,
        "tp_retention": tp,
        "fp_rejection": fp,
        "tp_median_score": tp_median,
        "fp_median_score": fp_median,
    }


def _passing_gate_inputs() -> tuple[dict[str, object], dict[str, object]]:
    condition_names = [name for name, _, _ in CONDITIONS]
    metrics: dict[str, object] = {
        "identity": {name: _metric(0.64) for name in condition_names},
        "acon_static": {name: _metric(0.65) for name in condition_names},
        "meta_acon": {name: _metric(0.69) for name in condition_names},
        META_OBJECT_BETA_ROLE: {
            name: _metric(0.68) for name in condition_names
        },
    }
    diagnostics = {
        "folds": [
            {"clean_auroc_delta_meta_vs_static": value}
            for value in (0.04, 0.03, 0.02, -0.01)
        ],
        "activity": {
            "between_sample_beta_rms": 0.006,
            "activation_delta_rms_mean": 0.02,
        },
    }
    return metrics, diagnostics


def test_information_gate_requires_precision_and_object_signal() -> None:
    metrics, diagnostics = _passing_gate_inputs()
    result = assess_information_gate(metrics, diagnostics)
    assert result["passed"] is True
    assert result["failed_checks"] == []

    metrics[META_OBJECT_BETA_ROLE]["clean"]["auroc"] = 0.60
    result = assess_information_gate(metrics, diagnostics)
    assert result["passed"] is False
    assert "object_beta_delta_vs_static_gte_0p02" in result["failed_checks"]

    metrics, diagnostics = _passing_gate_inputs()
    metrics["meta_acon"]["clean"]["fp_rejection"] = 0.19
    result = assess_information_gate(metrics, diagnostics)
    assert result["passed"] is False
    assert "clean_fp_rejection_gte_0p20" in result["failed_checks"]

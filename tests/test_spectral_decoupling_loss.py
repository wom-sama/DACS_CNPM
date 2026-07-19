import pytest
import torch
from pathlib import Path

from trkh.training.losses import (
    FocalCrossEntropyLoss,
    SpectralDecouplingCrossEntropyLoss,
)
from trkh.training.train import build_configs, parse_args
from trkh.tools.audit_spectral_decoupling_preflight import _equation_checks
from trkh.tools.audit_spectral_decoupling_smoke_pair import (
    assess_spectral_decoupling_pair,
)


def test_spectral_decoupling_matches_locked_hard_target_equation():
    regularization_lambda = 0.01
    logits = torch.tensor(
        [[2.0, -0.5, 0.25], [-1.0, 1.5, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1])
    loss_fn = SpectralDecouplingCrossEntropyLoss(
        regularization_lambda=regularization_lambda
    )

    observed = loss_fn.per_sample_loss(logits, targets)
    ce = FocalCrossEntropyLoss(gamma=0.0, focal_mix=0.0).per_sample_loss(
        logits, targets
    )
    expected = ce + 0.5 * regularization_lambda * logits.square().mean(dim=1)

    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-7)
    observed.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_spectral_decoupling_supports_soft_targets_and_class_multipliers():
    loss_fn = SpectralDecouplingCrossEntropyLoss(
        weight=torch.tensor([1.0, 2.0, 0.5]),
        label_smoothing=0.02,
        regularization_lambda=0.01,
    )
    loss_fn.set_class_weight_multipliers(torch.tensor([0.9, 1.1, 1.0]))
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor(
        [
            [0.90, 0.10, 0.00],
            [0.20, 0.70, 0.10],
            [0.00, 0.25, 0.75],
            [0.33, 0.33, 0.34],
        ],
        dtype=torch.float32,
    )

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_spectral_decoupling_lambda_zero_is_ordinary_cross_entropy():
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    spectral = SpectralDecouplingCrossEntropyLoss(
        regularization_lambda=0.0
    ).per_sample_loss(logits, targets)
    ordinary = FocalCrossEntropyLoss(
        gamma=0.0,
        focal_mix=0.0,
    ).per_sample_loss(logits, targets)

    assert torch.equal(spectral, ordinary)


def test_spectral_decoupling_rejects_negative_lambda():
    with pytest.raises(ValueError, match="lambda"):
        SpectralDecouplingCrossEntropyLoss(regularization_lambda=-1e-3)


def test_spectral_decoupling_cli_round_trip():
    args = parse_args(
        [
            "--classification-loss",
            "spectral_decoupling",
            "--spectral-decoupling-lambda",
            "0.0125",
        ]
    )
    _, train_config, _ = build_configs(args)

    assert train_config.classification_loss == "spectral_decoupling"
    assert train_config.spectral_decoupling_lambda == pytest.approx(0.0125)


def test_v8_launcher_exposes_spectral_decoupling_configuration():
    launcher = Path("scripts/run_trkh_5class_attention_views_v8.ps1").read_text(
        encoding="utf-8"
    )

    assert '"spectral_decoupling"' in launcher
    assert "$SpectralDecouplingLambda = 0.01" in launcher
    assert '"--spectral-decoupling-lambda", "$SpectralDecouplingLambda"' in launcher


def test_spectral_decoupling_stage_a_equation_checks_all_pass():
    assert all(_equation_checks().values())


def test_spectral_decoupling_locked_metric_gate_passes_only_precision_safe_gain():
    def metrics(macro_f1, precision, recall, f1):
        per_class = [
            {"precision": 0.90, "recall": 0.90, "f1": 0.90}
            for _ in range(5)
        ]
        per_class[1] = {"precision": precision, "recall": recall, "f1": f1}
        return {"macro_f1": macro_f1, "per_class": per_class}

    gate = assess_spectral_decoupling_pair(
        keeper_metrics=metrics(0.884, 0.61, 0.78, 0.686),
        control_metrics=metrics(0.870, 0.60, 0.78, 0.68),
        candidate_metrics=metrics(0.871, 0.606, 0.775, 0.686),
        transitions={
            "focus_false_positive_net_reduction": 2,
            "corrections": 5,
            "harms": 4,
        },
        runtime_ratio=1.01,
        aligned_rows=2606,
        probabilities_valid=True,
    )

    assert gate["metric_gate_passed"] is True
    assert gate["post_smoke_audit_required"] is True
    assert gate["full_train_permission"] is False


def test_spectral_decoupling_locked_metric_gate_rejects_precision_drop():
    per_class = [
        {"precision": 0.90, "recall": 0.90, "f1": 0.90}
        for _ in range(5)
    ]
    keeper = {"macro_f1": 0.884, "per_class": per_class}
    control = {"macro_f1": 0.870, "per_class": list(per_class)}
    candidate_class = [dict(value) for value in per_class]
    control["per_class"][1] = {"precision": 0.61, "recall": 0.78, "f1": 0.68}
    candidate_class[1] = {"precision": 0.60, "recall": 0.80, "f1": 0.69}
    candidate = {"macro_f1": 0.871, "per_class": candidate_class}

    gate = assess_spectral_decoupling_pair(
        keeper_metrics=keeper,
        control_metrics=control,
        candidate_metrics=candidate,
        transitions={
            "focus_false_positive_net_reduction": 2,
            "corrections": 5,
            "harms": 4,
        },
        runtime_ratio=1.0,
        aligned_rows=2606,
        probabilities_valid=True,
    )

    assert gate["metric_gate_passed"] is False
    assert "class1_precision_delta_gte_0p005" in gate["failed_checks"]

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from trkh.tools.audit_gsfl_shared_feature_readiness import (
    BATCH_SIZE,
    CENTER_UPDATE_RATE,
    EPOCHS,
    GSFLAdapter,
    GSFLCEControl,
    HIDDEN_DIM,
    SEED,
    _seed_torch,
    assess_gsfl_readiness,
    feature_expression_loss,
    l2_normalize_embeddings,
    parse_args,
    update_feature_centers,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert args.device == "cuda"
    assert args.preflight_only is False
    assert EPOCHS == 30
    assert BATCH_SIZE == 42
    assert HIDDEN_DIM == 128
    assert CENTER_UPDATE_RATE == 0.01
    assert SEED == 20260715


def test_l2_normalization_and_signed_decoder_loss_are_finite() -> None:
    raw = np.asarray([[1.0, -2.0, 2.0, 0.0], [-1.0, 0.0, 0.0, 3.0]])
    normalized = l2_normalize_embeddings(raw)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-7)
    model = GSFLAdapter(input_dim=4, hidden_dim=3, class_count=2, dropout=0.0)
    with torch.no_grad():
        model.decoder[-1].bias.fill_(-0.5)
    features = torch.from_numpy(normalized)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    outputs = model(features)
    assert outputs["reconstruction"].shape == features.shape
    assert bool(torch.any(outputs["reconstruction"] < 0.0))
    centers = torch.zeros(2, 3)
    shared = torch.zeros(3)
    loss, components = feature_expression_loss(
        outputs,
        features,
        labels,
        centers,
        shared,
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "total",
        "ce",
        "reconstruction",
        "class_center",
        "shared_center",
    }
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_center_update_matches_official_minibatch_equation() -> None:
    class_centers = torch.zeros(2, 2)
    total_centers = torch.zeros(2, 2)
    discriminative = torch.tensor([[1.0, 2.0], [2.0, 4.0], [5.0, 7.0]])
    labels = torch.tensor([0, 0, 1], dtype=torch.int64)
    shared = update_feature_centers(
        class_centers,
        total_centers,
        discriminative,
        labels,
        update_rate=0.01,
    )
    expected_zero = 0.01 * torch.tensor([3.0, 6.0]) / 3.0
    expected_one = 0.01 * torch.tensor([5.0, 7.0]) / 2.0
    assert torch.allclose(class_centers[0], expected_zero)
    assert torch.allclose(class_centers[1], expected_one)
    assert torch.equal(class_centers, total_centers)
    assert torch.allclose(shared, total_centers.mean(dim=0))


def test_control_and_candidate_share_initial_dropout_path() -> None:
    _seed_torch(17)
    candidate = GSFLAdapter(input_dim=4, hidden_dim=3, class_count=2, dropout=0.5)
    control = GSFLCEControl(input_dim=4, hidden_dim=3, class_count=2, dropout=0.5)
    control.discriminative_encoder.load_state_dict(
        copy.deepcopy(candidate.discriminative_encoder.state_dict())
    )
    control.classifier.load_state_dict(copy.deepcopy(candidate.classifier.state_dict()))
    candidate.train()
    control.train()
    features = F.normalize(torch.randn(8, 4), dim=1)
    _seed_torch(91)
    control_outputs = control(features)
    _seed_torch(91)
    candidate_outputs = candidate(features)
    assert torch.equal(
        control_outputs["discriminative"], candidate_outputs["discriminative"]
    )
    assert torch.equal(control_outputs["logits"], candidate_outputs["logits"])


def _passing_evidence() -> dict[str, object]:
    return {
        "delta_vs_raw": {
            "macro_f1": 0.003,
            "class1_f1": 0.007,
            "class1_precision": 0.015,
            "class1_recall": -0.003,
        },
        "class1_f1_delta_vs_margin_agem": -0.001,
        "candidate": {
            "macro_f1": 0.95,
            "per_class": [{"f1": 0.9}, {"f1": 0.87}],
        },
        "transitions_vs_raw": {
            "focus_true_positive_broken": 0,
            "corrections": 8,
            "harms": 3,
        },
        "direction_vs_raw": {"auc_fn_positive": 0.64},
        "restricted_focus_false_positives": {
            "raw": 20,
            "candidate": 15,
            "reduction": 5,
        },
    }


def test_gsfl_gate_requires_control_gain_and_zero_tp_break() -> None:
    control = {"macro_f1": 0.948, "per_class": [{"f1": 0.9}, {"f1": 0.86}]}
    result = assess_gsfl_readiness(
        _passing_evidence(),
        control,
        structural_checks={"structural": True},
    )
    assert result["shared_trainer_authorized"] is True
    assert result["failed_checks"] == []

    evidence = _passing_evidence()
    evidence["transitions_vs_raw"]["focus_true_positive_broken"] = 1
    evidence["candidate"]["per_class"][1]["f1"] = 0.862
    result = assess_gsfl_readiness(
        evidence,
        control,
        structural_checks={"structural": True},
    )
    assert result["shared_trainer_authorized"] is False
    assert "zero_raw_class1_tp_broken" in result["failed_checks"]
    assert "class1_f1_gain_vs_control_gte_0p005" in result["failed_checks"]

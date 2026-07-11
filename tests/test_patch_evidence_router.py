from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import (
    PatchEvidenceLinearVerifier,
    classification_logits_from_features,
    create_model,
    patch_evidence_summary_features,
)
from trkh.training.train import (
    _patch_evidence_router_loss_from_features,
    _patch_evidence_router_teacher_loss_from_features,
)
from trkh.tools.probe_patch_evidence_mil import summarize_patch_evidence


def _small_config(**overrides):
    values = {
        "model_type": "vit_registers",
        "image_size": 32,
        "patch_size": 8,
        "use_cnn_stem": False,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "num_registers": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_patch_evidence_router_extends_checkpoint_without_changing_logits() -> None:
    base = create_model(num_classes=5, model_config=_small_config())
    router_model = create_model(
        num_classes=5,
        model_config=_small_config(
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
            patch_evidence_router_dropout=0.0,
        ),
    )

    missing, unexpected = router_model.load_flexible_state_dict(
        base.state_dict(),
        strict=False,
    )

    assert not unexpected
    assert any(str(key).startswith("patch_evidence_router_head.") for key in missing)

    base.eval()
    router_model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), router_model(images), atol=1e-6)


def test_patch_evidence_router_adjusts_pair_logits_and_traces_attention() -> None:
    model = create_model(
        num_classes=5,
        model_config=_small_config(
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
            patch_evidence_router_dropout=0.0,
            patch_evidence_router_logit_scale=0.20,
            patch_evidence_router_routing=False,
        ),
    )
    with torch.no_grad():
        model.patch_evidence_router_head.output[-1].bias.fill_(0.50)

    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    features = model.forward_features(images, bbox_token_prior=bbox, return_trace=True)
    base_logits = model.head(model.head_input_from_features(features))
    logits = classification_logits_from_features(model, features)

    expected = torch.zeros_like(logits)
    expected[:, 0] = -0.10
    expected[:, 1] = 0.10

    assert torch.allclose(logits - base_logits, expected, atol=1e-5)
    assert tuple(features["patch_evidence_router_logits"].shape) == (2, 1)
    assert tuple(features["trace"]["patch_evidence_router_attention"].shape) == (2, 16)
    assert tuple(features["trace"]["patch_evidence_router_route_weights"].shape) == (2, 1)
    assert torch.allclose(
        features["trace"]["patch_evidence_router_attention"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
    assert "patch_evidence_router_margin_prior" in features["trace"]


def test_patch_evidence_router_margin_prior_is_checkpoint_safe_and_active() -> None:
    base = create_model(num_classes=5, model_config=_small_config())
    router_model = create_model(
        num_classes=5,
        model_config=_small_config(
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
            patch_evidence_router_dropout=0.0,
            patch_evidence_router_logit_scale=0.20,
            patch_evidence_router_routing=False,
            patch_evidence_router_margin_prior_mode="max",
            patch_evidence_router_margin_prior_scale=1.0,
        ),
    )
    router_model.load_flexible_state_dict(base.state_dict(), strict=False)

    images = torch.randn(2, 3, 32, 32)
    router_model.eval()
    with torch.no_grad():
        features = router_model.forward_features(images, return_trace=True)
        base_logits = router_model.head(router_model.head_input_from_features(features))
        patch_logits = router_model.head(features["patches"])
        logits = classification_logits_from_features(router_model, features)
        patch_margin = patch_logits[..., 1] - patch_logits[..., 0]
        selected_indices = features["trace"]["patch_evidence_router_selected_indices"]
        expected_prior = patch_margin.gather(1, selected_indices).max(dim=1).values

    assert torch.allclose(
        features["trace"]["patch_evidence_router_margin_prior"],
        expected_prior,
        atol=1e-5,
    )
    assert torch.allclose(logits[:, 1] - base_logits[:, 1], 0.20 * expected_prior, atol=1e-5)
    assert torch.allclose(logits[:, 0] - base_logits[:, 0], -0.20 * expected_prior, atol=1e-5)


def test_patch_evidence_router_summary_stats_are_checkpoint_safe_and_traced() -> None:
    base = create_model(num_classes=5, model_config=_small_config())
    router_model = create_model(
        num_classes=5,
        model_config=_small_config(
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
            patch_evidence_router_dropout=0.0,
            patch_evidence_router_summary_stats=True,
        ),
    )
    router_model.load_flexible_state_dict(base.state_dict(), strict=False)

    images = torch.randn(2, 3, 32, 32)
    router_model.eval()
    with torch.no_grad():
        features = router_model.forward_features(images, return_trace=True)
        base_logits = router_model.head(router_model.head_input_from_features(features))
        logits = classification_logits_from_features(router_model, features)

    assert torch.allclose(base_logits, logits, atol=1e-6)
    assert tuple(features["trace"]["patch_evidence_router_summary_stats"].shape) == (2, 13)
    assert torch.isfinite(features["trace"]["patch_evidence_router_summary_stats"]).all()


def test_patch_evidence_router_routing_requires_exact_pair_boundary() -> None:
    model = create_model(
        num_classes=5,
        model_config=_small_config(
            depth=1,
            num_registers=1,
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
            patch_evidence_router_route_max_probability_margin=0.25,
        ),
    )
    like_logits = torch.tensor(
        [
            [2.00, 1.95, 0.00, -1.00, -2.00],
            [2.00, 0.10, 1.95, -1.00, -2.00],
        ]
    )

    route_weights = model.patch_evidence_router_route_weights(like_logits)

    assert tuple(route_weights.shape) == (2, 1)
    assert route_weights[0, 0] > 0.0
    assert route_weights[1, 0] == 0.0


def test_patch_evidence_router_loss_uses_only_configured_pair() -> None:
    model = create_model(
        num_classes=5,
        model_config=_small_config(
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
        ),
    )
    features = {
        "patch_evidence_router_logits": torch.tensor([[0.0], [1.0], [4.0]]),
    }
    targets = torch.tensor([0, 1, 2])

    loss, stats = _patch_evidence_router_loss_from_features(
        model=model,
        features=features,
        targets=targets,
        positive_weight=1.25,
    )

    assert loss.item() > 0.0
    assert abs(stats["patch_evidence_router_fraction"] - (2.0 / 3.0)) < 1e-6
    assert stats["patch_evidence_router_positive_logit"] == 1.0
    assert stats["patch_evidence_router_negative_logit"] == 0.0


def test_patch_evidence_router_teacher_loss_filters_unreliable_oof_targets() -> None:
    model = create_model(
        num_classes=5,
        model_config=_small_config(
            patch_evidence_router_head=True,
            patch_evidence_router_pair="0-1",
        ),
    )
    features = {
        "patch_evidence_router_logits": torch.tensor([[-2.0], [2.0], [0.0], [1.0]]),
    }
    targets = torch.tensor([0, 1, 0, 1])
    teacher_probabilities = torch.tensor(
        [
            [0.90, 0.10, 0.00, 0.00, 0.00],
            [0.20, 0.80, 0.00, 0.00, 0.00],
            [0.40, 0.60, 0.00, 0.00, 0.00],
            [0.49, 0.51, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    loss, stats = _patch_evidence_router_teacher_loss_from_features(
        model=model,
        features=features,
        teacher_probabilities=teacher_probabilities,
        hard_labels=targets,
        targets=targets,
        min_confidence=0.70,
        min_pair_mass=0.20,
        positive_weight=1.0,
    )

    assert loss.item() > 0.0
    assert abs(stats["patch_evidence_router_teacher_pair_fraction"] - 1.0) < 1e-6
    assert abs(stats["patch_evidence_router_teacher_fraction"] - 0.5) < 1e-6
    assert abs(stats["patch_evidence_router_teacher_positive_fraction"] - 0.5) < 1e-6
    assert abs(stats["patch_evidence_router_teacher_target"] - 0.45) < 1e-6


def test_patch_evidence_linear_verifier_summary_matches_probe_layout() -> None:
    torch.manual_seed(7)
    local_logits = torch.randn(3, 7, 5)
    key_padding_mask = torch.tensor(
        [
            [False, False, False, True, False, False, False],
            [False, True, True, True, True, True, True],
            [False, False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    bbox_prior = torch.tensor(
        [
            [0.90, 0.80, 0.10, 0.00, 0.20, 0.00, 0.40],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        ],
        dtype=torch.float32,
    )

    expected, _method_logits, _diagnostics = summarize_patch_evidence(
        local_logits,
        key_padding_mask=key_padding_mask,
        bbox_prior=bbox_prior,
        pairs=[(0, 1)],
        top_k=4,
    )
    actual = patch_evidence_summary_features(
        local_logits,
        key_padding_mask=key_padding_mask,
        bbox_prior=bbox_prior,
        pairs=[(0, 1)],
        top_k=4,
    )

    assert tuple(actual.shape) == (3, 50)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_patch_evidence_linear_verifier_routes_and_protects_right_class() -> None:
    torch.manual_seed(11)
    head_input = torch.zeros(2, 4)
    logits = torch.tensor(
        [
            [1.00, 0.95, -5.00, -5.00, -5.00],
            [0.95, 1.00, -5.00, -5.00, -5.00],
        ],
        dtype=torch.float32,
    )
    local_logits = torch.randn(2, 6, 5)
    feature_dim = (
        head_input.size(1)
        + logits.size(1)
        + logits.size(1)
        + 2
        + patch_evidence_summary_features(local_logits, pairs=[(0, 1)]).size(1)
    )

    right_verifier = PatchEvidenceLinearVerifier(
        raw_coef=torch.zeros(feature_dim),
        raw_intercept=10.0,
        pair=(0, 1),
        logit_boost=5.0,
    )
    right_features = right_verifier.build_feature_vector(
        head_input=head_input,
        logits=logits,
        local_logits=local_logits,
    )
    adjusted, trace = right_verifier(logits, right_features, return_trace=True)

    assert adjusted[0].argmax().item() == 1
    assert adjusted[1].argmax().item() == 1
    assert trace["route_mask"].tolist() == [True, False]

    left_verifier = PatchEvidenceLinearVerifier(
        raw_coef=torch.zeros(feature_dim),
        raw_intercept=-10.0,
        pair=(0, 1),
        logit_boost=5.0,
        protect_right_min_probability=0.40,
    )
    left_features = left_verifier.build_feature_vector(
        head_input=head_input,
        logits=logits,
        local_logits=local_logits,
    )
    protected, protected_trace = left_verifier(logits, left_features, return_trace=True)

    assert protected[0].argmax().item() == 0
    assert protected[1].argmax().item() == 1
    assert protected_trace["route_mask"].tolist() == [False, False]


def test_patch_evidence_linear_verifier_loads_export_default_off(tmp_path) -> None:
    model = create_model(num_classes=5, model_config=_small_config())
    assert getattr(model, "patch_evidence_linear_verifier", None) is None
    assert not any(key.startswith("patch_evidence_linear_verifier.") for key in model.state_dict())

    export_path = tmp_path / "pair_verifier_model_params.json"
    feature_dim = 94
    export_path.write_text(
        (
            '{"metadata":{"top_k":4,"spatial_evidence_features":false,'
            '"source_domain_feature":false},"feature_dim":94,'
            '"pairs":[{"pair":"0-1","status":"exported","feature_dim":94,'
            '"raw_coef":['
            + ",".join(["0.0"] * feature_dim)
            + '],"raw_intercept":0.0}]}'
        ),
        encoding="utf-8",
    )

    summary = model.load_patch_evidence_linear_verifier_export(export_path)

    assert summary["enabled"] is True
    assert summary["feature_dim"] == feature_dim
    assert summary["pair"] == [0, 1]
    assert getattr(model, "patch_evidence_linear_verifier", None) is not None
    assert not any(
        key.startswith("patch_evidence_linear_verifier.") for key in model.state_dict()
    )


def test_patch_evidence_linear_verifier_training_soft_adjustment_has_gradient() -> None:
    torch.manual_seed(19)
    head_input = torch.randn(2, 4, requires_grad=True)
    logits = torch.tensor(
        [
            [0.70, 0.68, -1.0, -1.0, -1.0],
            [0.66, 0.71, -1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    local_logits = torch.randn(2, 6, 5, requires_grad=True)
    feature_dim = (
        head_input.size(1)
        + logits.size(1)
        + logits.size(1)
        + 2
        + patch_evidence_summary_features(local_logits, pairs=[(0, 1)]).size(1)
    )
    raw_coef = torch.zeros(feature_dim)
    raw_coef[0] = 1.0
    verifier = PatchEvidenceLinearVerifier(
        raw_coef=raw_coef,
        raw_intercept=0.0,
        pair=(0, 1),
        confidence_threshold=0.0,
        training_soft_adjustment=True,
        training_soft_logit_scale=0.20,
        training_soft_gate_temperature=0.05,
    )
    verifier.train()

    feature_vector = verifier.build_feature_vector(
        head_input=head_input,
        logits=logits,
        local_logits=local_logits,
    )
    adjusted, trace = verifier(logits, feature_vector, return_trace=True)
    loss = adjusted[:, 1].sum() - adjusted[:, 0].sum()
    loss.backward()

    assert trace["soft_delta"].abs().sum().item() > 0.0
    assert head_input.grad is not None
    assert head_input.grad.abs().sum().item() > 0.0

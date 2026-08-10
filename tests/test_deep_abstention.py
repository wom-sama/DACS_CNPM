import math

import pytest
import torch

from trkh.evaluation.evaluate import _build_prediction_records
from trkh.models.model import (
    VisionTransformerWithRegisters,
    classification_logits_from_features,
)
from trkh.training.train import (
    _deep_abstention_loss,
    _load_model_state_allowing_extensions,
)


def test_deep_abstention_loss_matches_dac_objective() -> None:
    logits = torch.tensor([[2.0, -0.5], [-1.0, 1.5]], requires_grad=True)
    abstention_logits = torch.tensor([-1.2, 0.3], requires_grad=True)
    targets = torch.tensor([0, 0])
    penalty = 1.3

    loss, stats = _deep_abstention_loss(
        logits=logits,
        abstention_logits=abstention_logits,
        targets=targets,
        penalty=penalty,
    )

    joint = torch.cat((logits, abstention_logits[:, None]), dim=1).softmax(dim=1)
    abstention_probability = joint[:, -1]
    non_abstention_probability = 1.0 - abstention_probability
    conditional_target_probability = (
        joint[torch.arange(targets.numel()), targets] / non_abstention_probability
    )
    expected = (
        non_abstention_probability * -conditional_target_probability.log()
        + penalty * -non_abstention_probability.log()
    ).mean()

    assert torch.allclose(loss, expected, atol=1e-6)
    assert math.isclose(
        stats["abstention_probability"],
        float(abstention_probability.detach().mean()),
        abs_tol=1e-6,
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(abstention_logits.grad).all()


def test_deep_abstention_gradient_separates_easy_and_hard_samples() -> None:
    logits = torch.tensor([[5.0, -5.0], [-2.0, 2.0]], requires_grad=True)
    abstention_logits = torch.full((2,), -4.0, requires_grad=True)
    targets = torch.tensor([0, 0])

    loss, _ = _deep_abstention_loss(
        logits=logits,
        abstention_logits=abstention_logits,
        targets=targets,
        penalty=1.3,
    )
    loss.backward()

    assert abstention_logits.grad[0].item() > 0.0
    assert abstention_logits.grad[1].item() < 0.0


def _tiny_model(*, enabled: bool) -> VisionTransformerWithRegisters:
    return VisionTransformerWithRegisters(
        image_size=32,
        patch_size=4,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        deep_abstention_head=enabled,
        deep_abstention_dropout=0.0,
        deep_abstention_initial_probability=0.01,
    )


def test_deep_abstention_head_exports_configured_initial_probability() -> None:
    model = _tiny_model(enabled=True)
    model.eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        features = {"pooled": torch.randn(3, model.embed_dim), "trace": {}}
        logits = classification_logits_from_features(model, features)
        joint = torch.cat(
            (logits, features["deep_abstention_logit"][:, None]), dim=1
        ).softmax(dim=1)

    assert features["deep_abstention_logit"].shape == (3,)
    assert "deep_abstention_logit" in features["trace"]
    assert torch.allclose(joint[:, -1], torch.full((3,), 0.01), atol=1e-6)


def test_deep_abstention_is_checkpoint_neutral_when_disabled() -> None:
    model = _tiny_model(enabled=False)
    assert model.deep_abstention_head is None
    assert not any("deep_abstention" in key for key in model.state_dict())


def test_resume_allows_only_declared_deep_abstention_extension() -> None:
    source = _tiny_model(enabled=False)
    target = _tiny_model(enabled=True)
    summary = _load_model_state_allowing_extensions(
        target,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert set(summary["allowed_missing_keys"]) == {
        "deep_abstention_norm.weight",
        "deep_abstention_norm.bias",
        "deep_abstention_head.weight",
        "deep_abstention_head.bias",
    }

    invalid_state = dict(target.state_dict())
    invalid_state.pop("head.weight")
    with pytest.raises(RuntimeError, match="Resume partial load"):
        _load_model_state_allowing_extensions(
            _tiny_model(enabled=True),
            invalid_state,
            allow_extensions=True,
        )


def test_prediction_records_export_abstention_probability() -> None:
    records = _build_prediction_records(
        targets=torch.tensor([1, 0]),
        predictions=torch.tensor([2, 0]),
        probabilities=torch.tensor([[0.1, 0.4, 0.5], [0.8, 0.1, 0.1]]),
        class_names=["a", "b", "c"],
        sample_paths=["first.jpg", "second.jpg"],
        abstention_probabilities=torch.tensor([0.25, 0.02]),
    )

    assert records[0]["abstention_probability"] == 0.25
    assert math.isclose(records[1]["abstention_probability"], 0.02, abs_tol=1e-6)

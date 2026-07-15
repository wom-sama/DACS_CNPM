from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.deep_class_prompt import DeepClassPrompt
from trkh.models.model import create_model
from trkh.training.train import (
    _load_model_state_allowing_extensions,
    parse_args,
)


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=3,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="1,2",
        token_keep_rates="0.75,0.50",
        deep_class_prompt=True,
        deep_class_prompt_logit_scale=0.10,
        deep_class_prompt_init_seed=20260715,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_locked_module_has_exact_parameter_and_state_contract() -> None:
    module = DeepClassPrompt(
        depth=8,
        num_classes=5,
        embed_dim=256,
        init_seed=20260715,
    )

    assert module.added_parameter_count == 11_009
    assert set(module.state_dict()) == {
        "prompt_embeddings",
        "norm.weight",
        "norm.bias",
        "classifier.weight",
        "classifier.bias",
    }
    assert module.prompt_embeddings.shape == (8, 5, 256)
    assert module.classifier.weight.shape == (1, 256)


def test_constructor_preserves_legacy_rng_and_base_state_bit_exact() -> None:
    seed = 23
    torch.manual_seed(seed)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(deep_class_prompt=False),
    )
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = create_model(num_classes=5, model_config=_tiny_config())
    candidate_rng = torch.get_rng_state().clone()

    assert torch.equal(control_rng, candidate_rng)
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    for key, value in control_state.items():
        torch.testing.assert_close(candidate_state[key], value, rtol=0.0, atol=0.0)


def test_prompt_maps_preserve_native_attention_schema_and_pruning_contract() -> None:
    torch.manual_seed(29)
    model = create_model(num_classes=5, model_config=_tiny_config()).eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        dense = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
        pruned = model.forward_features(images, return_trace=True)

    assert set(dense["attentions"]) == {0, 1, 2}
    assert set(dense["deep_class_prompt_attentions"]) == {0, 1, 2}
    assert all(
        attention.shape == (2, 4, 19, 19)
        for attention in dense["attentions"].values()
    )
    assert all(
        attention.shape == (2, 5, 16)
        for attention in dense["deep_class_prompt_attentions"].values()
    )
    assert [
        tuple(indices.shape)
        for indices in pruned["trace"]["block_patch_indices"]
    ] == [(2, 12), (2, 8), (2, 8)]
    assert pruned["patches"].shape == (2, 8, 32)
    assert pruned["tokens"].shape == (2, 11, 32)
    assert all(
        attention.shape == (2, 5, 16)
        for attention in pruned["deep_class_prompt_attentions"].values()
    )


def test_prompt_fusion_and_all_parameter_families_receive_gradients() -> None:
    torch.manual_seed(31)
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(token_pruning=False),
    ).train()
    images = torch.randn(3, 3, 32, 32)
    features = model.forward_features(images)
    base_logits = model.head(features["pooled"])
    expected = base_logits + 0.10 * features["deep_class_prompt_logits"]
    actual = model(images)
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    required = {
        "prompts": model.deep_class_prompt.prompt_embeddings,
        "norm_weight": model.deep_class_prompt.norm.weight,
        "norm_bias": model.deep_class_prompt.norm.bias,
        "classifier_weight": model.deep_class_prompt.classifier.weight,
        "classifier_bias": model.deep_class_prompt.classifier.bias,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert int(torch.count_nonzero(parameter.grad).item()) > 0, name


def test_checkpoint_extension_has_exact_missing_keys() -> None:
    torch.manual_seed(37)
    source = create_model(
        num_classes=5,
        model_config=_tiny_config(deep_class_prompt=False),
    )
    torch.manual_seed(37)
    target = create_model(num_classes=5, model_config=_tiny_config())

    summary = _load_model_state_allowing_extensions(
        target,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert set(summary["allowed_missing_keys"]) == {
        "deep_class_prompt.prompt_embeddings",
        "deep_class_prompt.norm.weight",
        "deep_class_prompt.norm.bias",
        "deep_class_prompt.classifier.weight",
        "deep_class_prompt.classifier.bias",
    }


def test_defaults_cli_validation_and_v8_launcher_controls() -> None:
    defaults = ModelConfig()
    assert defaults.deep_class_prompt is False
    assert defaults.deep_class_prompt_logit_scale == pytest.approx(0.10)
    assert defaults.deep_class_prompt_init_seed == 20260715
    args = parse_args(["--deep-class-prompt"])
    assert args.deep_class_prompt is True
    assert args.deep_class_prompt_logit_scale == pytest.approx(0.10)
    assert args.deep_class_prompt_init_seed == 20260715
    with pytest.raises(ValueError, match=">= 0"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(deep_class_prompt_logit_scale=-0.01),
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$DeepClassPrompt = $false" in script
    assert "[double]$DeepClassPromptLogitScale = 0.10" in script
    assert '"--deep-class-prompt"' in script
    assert '"--deep-class-prompt-logit-scale"' in script
    assert '"--deep-class-prompt-init-seed"' in script

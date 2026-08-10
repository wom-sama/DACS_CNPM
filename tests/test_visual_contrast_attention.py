from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.model import MultiHeadSelfAttention, create_model
from trkh.models.visual_contrast_attention import (
    VisualContrastAttention,
    visual_contrast_lambda_init,
)
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_model_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
        token_pruning=False,
        visual_contrast_attention=True,
        visual_contrast_attention_layers="1,2",
        visual_contrast_tokens=4,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _dense_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    inputs = torch.randn(2, 21, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    return inputs, indices


def test_visual_contrast_attention_shape_and_normalized_effective_map() -> None:
    inputs, indices = _dense_inputs()
    module = VisualContrastAttention(
        dim=32,
        num_heads=4,
        visual_contrast_tokens=4,
        block_depth=2,
    )
    output, attention = module(
        inputs,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    assert output.shape == inputs.shape
    assert attention.shape == (2, 4, 21, 21)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention).all()
    assert bool((attention >= 0).all())
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=1e-6,
        rtol=0.0,
    )


def test_visual_contrast_attention_all_component_families_receive_gradients() -> None:
    inputs, indices = _dense_inputs()
    module = VisualContrastAttention(
        dim=32,
        num_heads=4,
        visual_contrast_tokens=4,
        block_depth=1,
    )
    output = module(
        inputs,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    output.square().mean().backward()
    required = {
        "qkv.weight",
        "proj.weight",
        "positive_embedding",
        "negative_embedding",
        "stage1_lambda_q1",
        "stage2_lambda_q1",
        "stage1_norm.weight",
        "stage2_norm.weight",
        "depthwise_value.weight",
    }
    gradients = dict(module.named_parameters())
    for name in required:
        gradient = gradients[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient).item()) > 0, name


def test_visual_contrast_attention_rejects_sparse_or_malformed_grids() -> None:
    inputs, indices = _dense_inputs()
    module = VisualContrastAttention(dim=32, num_heads=4, visual_contrast_tokens=4)
    with pytest.raises(ValueError, match="complete dense patch grid"):
        module(inputs[:, :-1], grid_size=(4, 4), prefix_count=5)
    reversed_indices = indices.flip(1)
    with pytest.raises(ValueError, match="identity-ordered"):
        module(
            inputs,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=reversed_indices,
        )
    with pytest.raises(ValueError, match="perfect square"):
        VisualContrastAttention(dim=32, num_heads=4, visual_contrast_tokens=6)
    with pytest.raises(ValueError, match="divisible"):
        VisualContrastAttention(dim=30, num_heads=4, visual_contrast_tokens=4)


def test_visual_contrast_attention_seed_and_state_roundtrip_are_exact() -> None:
    torch.manual_seed(42)
    first = VisualContrastAttention(
        dim=32,
        num_heads=4,
        visual_contrast_tokens=4,
        block_depth=3,
    )
    torch.manual_seed(42)
    second = VisualContrastAttention(
        dim=32,
        num_heads=4,
        visual_contrast_tokens=4,
        block_depth=3,
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    restored = copy.deepcopy(first)
    restored.load_state_dict(first.state_dict(), strict=True)
    inputs, indices = _dense_inputs()
    first.eval()
    restored.eval()
    with torch.inference_mode():
        expected = first(
            inputs,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=indices,
        )
        observed = restored(
            inputs,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=indices,
        )
    assert torch.equal(expected, observed)


def test_visual_contrast_lambda_schedule_matches_paper_formula() -> None:
    assert visual_contrast_lambda_init(0) == pytest.approx(0.2)
    assert visual_contrast_lambda_init(4) > visual_contrast_lambda_init(0)
    with pytest.raises(ValueError, match="nonnegative"):
        visual_contrast_lambda_init(-1)


@pytest.mark.parametrize("model_type", ["vit_registers", "vit_registers_hybrid"])
def test_visual_contrast_attention_model_wiring_trace_and_xai_map(
    model_type: str,
) -> None:
    torch.manual_seed(13)
    model = create_model(
        num_classes=5,
        model_config=_tiny_model_config(model_type=model_type),
    ).eval()
    assert all(isinstance(block.attn, VisualContrastAttention) for block in model.blocks)
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        features = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
    assert set(features["attentions"]) == {0, 1}
    assert features["attention_representation"] == "vca_effective_positive"
    assert features["attention_representations"] == {
        0: "vca_effective_positive",
        1: "vca_effective_positive",
    }
    for attention in features["attentions"].values():
        assert attention.shape == (2, 4, 19, 19)
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones_like(attention.sum(dim=-1)),
            atol=1e-6,
            rtol=0.0,
        )
    trace = features["trace"]
    assert trace["visual_contrast_attention_layers"].tolist() == [1, 2]
    assert trace["visual_contrast_lambda_stage1"].shape == (2,)
    assert trace["visual_contrast_lambda_stage2"].shape == (2,)
    torch.testing.assert_close(
        trace["visual_contrast_stage1_positive_mass"]
        + trace["visual_contrast_stage1_negative_mass"],
        torch.ones(2),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["visual_contrast_stage2_positive_mass"]
        + trace["visual_contrast_stage2_negative_mass"],
        torch.ones(2),
        atol=1e-6,
        rtol=0.0,
    )


def test_visual_contrast_attention_resume_reuses_compatible_mhsa_parameters() -> None:
    torch.manual_seed(17)
    source = create_model(
        num_classes=5,
        model_config=_tiny_model_config(visual_contrast_attention=False),
    )
    target = create_model(num_classes=5, model_config=_tiny_model_config())
    source_qkv = source.blocks[0].attn.qkv.weight.detach().clone()

    summary = _load_model_state_allowing_extensions(
        target,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert summary["allowed_missing_keys"]
    assert all(".attn." in str(key) for key in summary["allowed_missing_keys"])
    torch.testing.assert_close(target.blocks[0].attn.qkv.weight, source_qkv)
    torch.testing.assert_close(
        target.blocks[0].attn.proj.weight,
        source.blocks[0].attn.proj.weight,
    )


def test_visual_contrast_attention_defaults_and_dense_grid_conflicts() -> None:
    default_model = create_model(
        num_classes=5,
        model_config=_tiny_model_config(visual_contrast_attention=False),
    )
    assert all(isinstance(block.attn, MultiHeadSelfAttention) for block in default_model.blocks)
    with pytest.raises(ValueError, match="dense patch grid"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                token_pruning=True,
                token_prune_layers="1",
                token_keep_rates="0.5",
            ),
        )
    with pytest.raises(ValueError, match="gated relative position"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                gated_relative_position_attention=True,
                gated_relative_position_attention_layers="1,2",
            ),
        )


def test_v8_launcher_exposes_dense_visual_contrast_controls() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$TokenPruning = $true" in script
    assert "[bool]$VisualContrastAttention = $false" in script
    assert '"--visual-contrast-attention"' in script
    assert '"--visual-contrast-attention-layers"' in script
    assert '"--visual-contrast-tokens"' in script
    assert "VisualContrastAttention -and $TokenPruning" in script

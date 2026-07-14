from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import build_model_from_checkpoint, create_model
from trkh.models.starnet_s2_tokenizer import (
    StarNetBlock,
    StarNetS2Tokenizer,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=256,
        patch_size=16,
        stem_architecture="starnet_s2_tokenizer",
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=False,
        token_pruning=False,
    )


def test_starnet_s2_preserves_official_stages_and_grid() -> None:
    tokenizer = StarNetS2Tokenizer()
    tokenizer.eval()
    with torch.no_grad():
        output, trace = tokenizer.forward_with_trace(torch.randn(2, 3, 256, 256))

    assert tokenizer.stage_depths == (1, 2, 6)
    assert tokenizer.stage_channels == (32, 64, 128)
    assert tokenizer.stage_width_depth_pairs == ((32, 1), (64, 2), (128, 6))
    assert tokenizer.block_count == 9
    assert output.shape == (2, 128, 16, 16)
    assert trace["stem_output"].shape == (2, 32, 128, 128)
    assert trace["stage1_output"].shape == (2, 32, 64, 64)
    assert trace["stage2_output"].shape == (2, 64, 32, 32)
    assert trace["stage3_output"].shape == (2, 128, 16, 16)
    assert all(torch.isfinite(value).all() for value in trace.values())


def test_starnet_block_uses_live_star_interaction_and_same_weight_sum_differs() -> None:
    torch.manual_seed(42)
    block = StarNetBlock(32, expansion=4)
    block.eval()
    x = torch.randn(2, 32, 20, 20, requires_grad=True)
    features = block.interaction_features(x)
    star_output = block.forward_with_interaction(x, interaction="star")
    sum_output = block.forward_with_interaction(x, interaction="sum")

    assert features["f1"].shape == (2, 128, 20, 20)
    assert features["f2"].shape == (2, 128, 20, 20)
    assert torch.count_nonzero(features["star"]) > 0
    assert not torch.equal(features["star"], features["f1"])
    assert not torch.equal(features["star"], features["f2"])
    assert float((star_output - sum_output).abs().mean()) > 1e-6

    star_output.square().mean().backward()
    for parameter in (block.f1.conv.weight, block.f2.conv.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert int(torch.count_nonzero(parameter.grad)) > 0


def test_starnet_trkh_forward_and_component_gradients_are_live() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_model_config())
    logits = model(torch.randn(2, 3, 256, 256))
    logits.square().mean().backward()

    assert isinstance(model.stem, StarNetS2Tokenizer)
    assert model.patch_embed.patch_size == 1
    assert model.patch_embed.num_patches == 256
    assert logits.shape == (2, 5)
    parameters = dict(model.named_parameters())
    required = (
        "stem.stem.0.conv.weight",
        "stem.stages.0.downsample.conv.weight",
        "stem.stages.2.downsample.conv.weight",
        "stem.stages.0.blocks.0.dwconv.conv.weight",
        "stem.stages.1.blocks.1.f1.conv.weight",
        "stem.stages.2.blocks.5.f2.conv.weight",
        "stem.stages.2.blocks.5.g.conv.weight",
        "stem.stages.2.blocks.5.dwconv2.conv.weight",
        "patch_embed.proj.weight",
    )
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient)) > 0, name


def test_starnet_checkpoint_and_official_initialization() -> None:
    config = _model_config()
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )

    stem_conv = source.stem.stem[0].conv
    expected_bound = 1.0 / (3 * 3 * 3) ** 0.5
    assert float(stem_conv.weight.abs().max()) <= expected_bound + 1e-6
    assert torch.allclose(source.stem.stem[0].bn.weight, torch.ones(32))
    assert torch.allclose(source.stem.stem[0].bn.bias, torch.zeros(32))
    assert isinstance(restored.stem, StarNetS2Tokenizer)
    assert set(restored.state_dict()) == set(source.state_dict())


def test_starnet_full_recipe_fusions_use_stem_width() -> None:
    config = _model_config()
    config.cnn_feature_fusion = True
    config.multi_branch_fusion = True
    config.branch_color_tokens = 1
    config.branch_edge_tokens = 1
    config.branch_cnn_tokens = 0
    config.head_pooling = "cls_branch_register_mean"
    model = create_model(num_classes=5, model_config=config)

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 256, 256))

    assert model.cnn_fusion_norm.normalized_shape == (128,)
    assert model.cnn_fusion_head.in_features == 128
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_starnet_rejects_invalid_input_or_interaction() -> None:
    try:
        StarNetS2Tokenizer(in_channels=1)
    except ValueError as error:
        assert "three-channel" in str(error)
    else:
        raise AssertionError("Expected non-RGB StarNet tokenizer rejection")

    block = StarNetBlock(8)
    try:
        block.forward_with_interaction(torch.randn(1, 8, 8, 8), interaction="max")
    except ValueError as error:
        assert "interaction" in str(error)
    else:
        raise AssertionError("Expected unknown StarNet interaction rejection")

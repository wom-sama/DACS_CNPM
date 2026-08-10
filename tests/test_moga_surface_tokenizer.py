from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.moga_surface_tokenizer import (
    MogaMultiOrderDWConv,
    MogaXTTokenizer,
)
from trkh.models.model import build_model_from_checkpoint, create_model


def _model_config(stem_architecture: str = "moganet_xt_tokenizer") -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=256,
        patch_size=16,
        stem_architecture=stem_architecture,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=False,
        token_pruning=False,
    )


def test_moga_xt_tokenizer_preserves_locked_stage_depths_and_grid() -> None:
    tokenizer = MogaXTTokenizer()
    tokenizer.eval()
    with torch.no_grad():
        output, trace = tokenizer.forward_with_trace(torch.randn(2, 3, 256, 256))

    assert tokenizer.stage_depths == (3, 3, 10)
    assert tokenizer.block_count == 16
    assert output.shape == (2, 96, 16, 16)
    assert trace["stage1_output"].shape == (2, 32, 64, 64)
    assert trace["stage2_output"].shape == (2, 64, 32, 32)
    assert trace["stage3_output"].shape == (2, 96, 16, 16)
    assert all(torch.isfinite(value).all() for value in trace.values())


def test_moga_multi_order_branches_are_nonzero_and_distinct() -> None:
    module = MogaMultiOrderDWConv(32)
    branches = module.branch_features(torch.randn(2, 32, 24, 24))

    assert tuple(branches) == ("low", "middle", "high")
    assert [branches[key].shape[1] for key in branches] == [4, 12, 16]
    assert all(float(value.square().mean()) > 0.0 for value in branches.values())
    means = [float(value.mean()) for value in branches.values()]
    assert len({round(value, 7) for value in means}) == 3


def test_moga_trkh_forward_and_component_gradients_are_live() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_model_config())
    logits = model(torch.randn(2, 3, 256, 256))
    logits.square().mean().backward()

    assert isinstance(model.stem, MogaXTTokenizer)
    assert model.stem.downsample_factor == 16
    assert model.patch_embed.patch_size == 1
    assert model.patch_embed.num_patches == 256
    assert logits.shape == (2, 5)
    required = (
        "stem.patch_embed1.projection.0.weight",
        "stem.blocks1.0.spatial.value.dw_conv_low.weight",
        "stem.blocks2.0.spatial.value.dw_conv_middle.weight",
        "stem.blocks3.0.spatial.value.dw_conv_high.weight",
        "stem.blocks3.9.spatial.gate.weight",
        "stem.blocks3.9.spatial.sigma.scale",
        "stem.blocks3.9.channel.dwconv.weight",
        "stem.blocks3.9.channel.sigma.scale",
        "patch_embed.proj.weight",
    )
    parameters = dict(model.named_parameters())
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient)) > 0, name


def test_moga_checkpoint_rebuilds_without_external_runtime() -> None:
    config = _model_config()
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )

    assert isinstance(restored.stem, MogaXTTokenizer)
    assert restored.stem_architecture == "moganet_xt_tokenizer"
    assert set(restored.state_dict()) == set(source.state_dict())


def test_moga_full_recipe_cnn_and_branch_fusion_use_stem_width() -> None:
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

    assert model.cnn_fusion_norm.normalized_shape == (96,)
    assert model.cnn_fusion_head.in_features == 96
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()

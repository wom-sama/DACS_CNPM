from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.inceptionnext_atto_tokenizer import (
    InceptionNeXtAttoTokenizer,
    InceptionNeXtDWConv2d,
)
from trkh.models.model import build_model_from_checkpoint, create_model


def _model_config() -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=256,
        patch_size=16,
        stem_architecture="inceptionnext_atto_tokenizer",
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=False,
        token_pruning=False,
    )


def test_inceptionnext_atto_preserves_official_stages_and_grid() -> None:
    tokenizer = InceptionNeXtAttoTokenizer()
    tokenizer.eval()
    with torch.no_grad():
        output, trace = tokenizer.forward_with_trace(torch.randn(2, 3, 256, 256))

    assert tokenizer.stage_depths == (2, 2, 6)
    assert tokenizer.stage_channels == (40, 80, 160)
    assert tokenizer.stage_splits == (
        (10, 10, 10, 10),
        (20, 20, 20, 20),
        (40, 40, 40, 40),
    )
    assert tokenizer.block_count == 10
    assert output.shape == (2, 160, 16, 16)
    assert trace["stage1_output"].shape == (2, 40, 64, 64)
    assert trace["stage2_output"].shape == (2, 80, 32, 32)
    assert trace["stage3_output"].shape == (2, 160, 16, 16)
    assert all(torch.isfinite(value).all() for value in trace.values())


def test_inceptionnext_four_branches_are_nonzero_and_distinct() -> None:
    module = InceptionNeXtDWConv2d(40)
    branches = module.branch_features(torch.randn(2, 40, 24, 24))

    assert tuple(branches) == ("identity", "square", "horizontal", "vertical")
    assert [value.shape[1] for value in branches.values()] == [10, 10, 10, 10]
    assert all(float(value.square().mean()) > 0.0 for value in branches.values())
    signatures = {
        (round(float(value.mean()), 7), round(float(value.std()), 7))
        for value in branches.values()
    }
    assert len(signatures) == 4


def test_inceptionnext_trkh_forward_and_component_gradients_are_live() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_model_config())
    logits = model(torch.randn(2, 3, 256, 256))
    logits.square().mean().backward()

    assert isinstance(model.stem, InceptionNeXtAttoTokenizer)
    assert model.patch_embed.patch_size == 1
    assert model.patch_embed.num_patches == 256
    assert logits.shape == (2, 5)
    parameters = dict(model.named_parameters())
    required = (
        "stem.stem.0.weight",
        "stem.stages.1.downsample.1.weight",
        "stem.stages.2.downsample.1.weight",
        "stem.stages.0.blocks.0.token_mixer.dwconv_hw.weight",
        "stem.stages.1.blocks.0.token_mixer.dwconv_w.weight",
        "stem.stages.2.blocks.5.token_mixer.dwconv_h.weight",
        "stem.stages.2.blocks.5.mlp.fc1.weight",
        "stem.stages.2.blocks.5.mlp.fc2.weight",
        "stem.stages.2.blocks.5.gamma",
        "patch_embed.proj.weight",
    )
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient)) > 0, name


def test_inceptionnext_checkpoint_and_official_initialization() -> None:
    config = _model_config()
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )

    stem_weight = source.stem.stem[0].weight.detach()
    assert 0.015 < float(stem_weight.std()) < 0.025
    assert torch.allclose(
        source.stem.stages[2].blocks[5].gamma,
        torch.full((160,), 1e-6),
    )
    assert isinstance(restored.stem, InceptionNeXtAttoTokenizer)
    assert set(restored.state_dict()) == set(source.state_dict())


def test_inceptionnext_full_recipe_fusions_use_stem_width() -> None:
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

    assert model.cnn_fusion_norm.normalized_shape == (160,)
    assert model.cnn_fusion_head.in_features == 160
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()

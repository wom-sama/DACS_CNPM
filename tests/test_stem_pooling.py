from dataclasses import asdict

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import (
    HybridConvStem,
    MaxSoftPool2d,
    SoftPool2d,
    build_model_from_checkpoint,
    create_model,
)


def _tiny_config(pooling_mode: str) -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=True,
        stem_channels=4,
        stem_pooling_mode=pooling_mode,
        cnn_feature_fusion=False,
        fine_grained_pooling=False,
        multi_branch_fusion=False,
        detail_patch_enhancement=False,
        token_pruning=False,
        pairwise_margin_head=False,
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )


def test_soft_pool_preserves_shape_and_local_range() -> None:
    pool = SoftPool2d(kernel_size=2, stride=2)
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    output = pool(values)

    assert output.shape == (1, 1, 1, 1)
    assert 2.5 < float(output.item()) < 4.0


def test_max_soft_pool_blends_legacy_and_detail_preserving_outputs() -> None:
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    max_value = torch.nn.functional.max_pool2d(values, 2, 2)
    soft_value = SoftPool2d(2, 2)(values)
    pool = MaxSoftPool2d(2, 2, soft_blend=0.15)

    output = pool(values)

    assert torch.allclose(output, torch.lerp(max_value, soft_value, 0.15))


def test_soft_and_max_stems_have_identical_state_dict_schema() -> None:
    max_stem = HybridConvStem(3, stem_channels=4, embed_dim=16, pooling_mode="max")
    soft_stem = HybridConvStem(3, stem_channels=4, embed_dim=16, pooling_mode="soft")

    assert max_stem.state_dict().keys() == soft_stem.state_dict().keys()
    output = soft_stem(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 16, 4, 4)
    assert torch.isfinite(output).all()

    mixed_stem = HybridConvStem(
        3,
        stem_channels=4,
        embed_dim=16,
        pooling_mode="max_soft",
        softpool_blend=0.15,
    )
    assert max_stem.state_dict().keys() == mixed_stem.state_dict().keys()


def test_checkpoint_can_override_stem_pooling_strictly() -> None:
    config = _tiny_config("max")
    model = create_model(num_classes=5, model_config=config)
    checkpoint = {
        "class_names": [f"c{index}" for index in range(5)],
        "model_config": asdict(config),
        "model_state": model.state_dict(),
    }

    restored = build_model_from_checkpoint(
        checkpoint,
        override_stem_pooling_mode="soft",
    )

    assert restored.stem_pooling_mode == "soft"
    assert restored.stem.pooling_mode == "soft"
    logits = restored(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 5)

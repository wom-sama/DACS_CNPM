import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.model import CoAtNetMBConvStem, build_model_from_checkpoint, create_model


pytest.importorskip("timm")


def _model_config(stem_architecture: str) -> ModelConfig:
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


def test_coatnet_mbconv_stem_preserves_trkh_stride_and_forward_shape() -> None:
    model = create_model(num_classes=5, model_config=_model_config("coatnet_mbconv"))
    assert isinstance(model.stem, CoAtNetMBConvStem)
    assert model.stem.downsample_factor == 8
    assert model.stem.out_channels == 64

    model.eval()
    with torch.no_grad():
        stem_features = model.stem(torch.randn(1, 3, 256, 256))
        logits = model(torch.randn(1, 3, 256, 256))
    assert stem_features.shape == (1, 64, 32, 32)
    assert logits.shape == (1, 5)


def test_coatnet_mbconv_checkpoint_rebuilds_without_pretrained_weights() -> None:
    config = _model_config("coatnet_mbconv")
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )
    assert isinstance(restored.stem, CoAtNetMBConvStem)
    assert restored.stem_architecture == "coatnet_mbconv"
    assert set(restored.state_dict()) == set(source.state_dict())


def test_default_stem_architecture_remains_legacy_conv_pool() -> None:
    config = _model_config("conv_pool")
    model = create_model(num_classes=5, model_config=config)
    assert model.stem_architecture == "conv_pool"
    assert not any("mbconv_stages" in key for key in model.state_dict())
    assert not any("blocks3" in key for key in model.state_dict())
    assert not any("token_mixer.dwconv_w" in key for key in model.state_dict())
    assert not any("stem.stages.0.blocks.0.f1" in key for key in model.state_dict())
    assert not any("stem.blocks.0.octave" in key for key in model.state_dict())


def test_unknown_stem_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="stem_architecture"):
        create_model(num_classes=5, model_config=_model_config("unknown"))

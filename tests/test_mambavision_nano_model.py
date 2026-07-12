import pytest
import torch

from trkh.core.config import IMAGENET_MEAN, IMAGENET_STD, ModelConfig
from trkh.models.model import (
    MAMBAVISION_NANO_SPEC,
    build_model_from_checkpoint,
    create_model,
)
from trkh.training.train import _apply_timm_input_normalization


pytest.importorskip("mambavision")


def _model_config(*, pretrained: bool = False) -> ModelConfig:
    return ModelConfig(
        model_type="mambavision_nano",
        pretrained=pretrained,
        image_size=256,
    )


def test_mambavision_nano_has_locked_structure():
    model = create_model(num_classes=5, model_config=_model_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_172_197
    assert model.trkh_architecture_spec == MAMBAVISION_NANO_SPEC
    assert [len(level.blocks) for level in model.levels] == [1, 2, 4, 2]
    assert [
        type(block.mixer).__name__ for block in model.levels[2].blocks
    ] == ["MambaVisionMixer", "MambaVisionMixer", "Attention", "Attention"]
    assert [
        type(block.mixer).__name__ for block in model.levels[3].blocks
    ] == ["MambaVisionMixer", "Attention"]



@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba-ssm selective scan requires CUDA")
def test_mambavision_nano_cuda_forward_shape():
    model = create_model(num_classes=5, model_config=_model_config()).cuda().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 256, 256, device="cuda"))
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()


def test_mambavision_nano_rebuilds_strictly_from_checkpoint_config():
    source_model = create_model(num_classes=5, model_config=_model_config())
    rebuilt = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": {
                "model_type": "mambavision_nano",
                "pretrained": False,
                "image_size": 256,
            },
            "model_state": source_model.state_dict(),
        },
        num_classes=5,
    )
    assert rebuilt.model_type == "mambavision_nano"
    assert sum(parameter.numel() for parameter in rebuilt.parameters()) == 6_172_197
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key])


def test_mambavision_nano_rejects_pretrained_weights():
    with pytest.raises(ValueError, match="pretrained"):
        create_model(num_classes=5, model_config=_model_config(pretrained=True))


def test_mambavision_nano_rejects_unlocked_input_or_temporal_shape():
    with pytest.raises(ValueError, match="locked input size"):
        create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="mambavision_nano",
                pretrained=False,
                image_size=224,
            ),
        )
    with pytest.raises(ValueError, match="temporal_frames=1"):
        create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="mambavision_nano",
                pretrained=False,
                image_size=256,
                temporal_frames=2,
            ),
        )


def test_mambavision_nano_uses_locked_imagenet_normalization():
    config = _model_config()
    summary = _apply_timm_input_normalization(config)
    assert summary["enabled"] is True
    assert summary["source"] == "locked_mambavision_nano_imagenet"
    assert tuple(config.input_mean) == tuple(IMAGENET_MEAN)
    assert tuple(config.input_std) == tuple(IMAGENET_STD)
    assert summary["input_size"] == [3, 256, 256]

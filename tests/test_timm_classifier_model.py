import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.model import build_model_from_checkpoint, create_model


pytest.importorskip("timm")


def test_timm_classifier_forward_no_pretrained():
    model = create_model(
        num_classes=5,
        model_config=ModelConfig(
            model_type="timm_classifier",
            timm_model_name="mobilenetv3_large_100.ra_in1k",
            pretrained=False,
        ),
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 5)


def test_timm_classifier_rebuilds_from_checkpoint_config():
    source_model = create_model(
        num_classes=5,
        model_config=ModelConfig(
            model_type="timm_classifier",
            timm_model_name="mobilenetv3_large_100.ra_in1k",
            pretrained=False,
        ),
    )
    model = build_model_from_checkpoint(
        {
            "class_names": [f"class_{idx}" for idx in range(5)],
            "model_config": {
                "model_type": "timm_classifier",
                "timm_model_name": "mobilenetv3_large_100.ra_in1k",
                "pretrained": False,
            },
            "model_state": source_model.state_dict(),
        },
        num_classes=5,
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 5)


def test_timm_coatnet_nano_forward_at_fixed_native_size():
    model = create_model(
        num_classes=5,
        model_config=ModelConfig(
            model_type="timm_classifier",
            timm_model_name="coatnet_nano_rw_224",
            pretrained=False,
        ),
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 5)

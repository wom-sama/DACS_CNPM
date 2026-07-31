import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.core.utils import build_optimizer_param_groups
from trkh.models.model import build_model_from_checkpoint, create_model
from trkh.training.train import _forward_model_outputs


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
    assert model.is_timm_classifier is True
    assert model.is_pretrained_timm_classifier is False


def test_timm_classifier_exposes_mobile_feature_contract():
    model = create_model(
        num_classes=5,
        model_config=ModelConfig(
            model_type="timm_classifier",
            timm_model_name="mobilenetv3_large_100.ra_in1k",
            pretrained=False,
        ),
    ).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        features, logits = _forward_model_outputs(model, images)

    assert features is not None
    assert logits.shape == (2, 5)
    assert features["pooled"].ndim == 2
    assert features["patches"].ndim == 3
    assert features["patches"].shape[0] == 2


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
                "research_track": "no_pretrain",
            },
            "model_state": source_model.state_dict(),
        },
        num_classes=5,
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 5)
    assert model.is_timm_classifier is True
    assert model.is_pretrained_timm_classifier is False
    assert not hasattr(model, "pretrained_provenance")


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


def test_pretrained_timm_optimizer_honors_exact_and_vector_no_decay() -> None:
    class _TimmLikeClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cls_token = torch.nn.Parameter(torch.ones(1, 1, 4))
            self.vector_scale = torch.nn.Parameter(torch.ones(4))
            self.backbone = torch.nn.Linear(4, 4, bias=False)
            self.classifier = torch.nn.Linear(4, 5)
            self.is_pretrained_timm_classifier = True
            self.pretrained_classifier_parameter_prefixes = ("classifier.",)

        def no_weight_decay(self):
            return {"cls_token"}

    model = _TimmLikeClassifier()
    groups = build_optimizer_param_groups(
        model,
        weight_decay=0.05,
        learning_rate=1.5e-4,
        backbone_lr_scale=0.1,
    )
    decay_by_parameter_id = {
        id(parameter): float(group["weight_decay"])
        for group in groups
        for parameter in group["params"]
    }

    assert decay_by_parameter_id[id(model.cls_token)] == 0.0
    assert decay_by_parameter_id[id(model.vector_scale)] == 0.0
    assert decay_by_parameter_id[id(model.backbone.weight)] == 0.05

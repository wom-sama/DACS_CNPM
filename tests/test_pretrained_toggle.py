from __future__ import annotations

from unittest.mock import patch

import pytest
from torch import nn

from trkh.models.model import create_model


class _TinyResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1000)

    def forward(self, x):
        return self.fc(x)


def test_resnet50_pretrained_flag_uses_torchvision_default_weights() -> None:
    calls = {}

    def fake_resnet50(*, weights=None):
        calls["weights"] = weights
        return _TinyResNet()

    with patch("trkh.models.model.tv_models.resnet50", fake_resnet50):
        model = create_model(
            num_classes=5,
            model_config={"model_type": "resnet50", "pretrained": True},
        )

    assert calls["weights"] is not None
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 5


def test_resnet50_no_pretrained_keeps_weights_none() -> None:
    calls = {}

    def fake_resnet50(*, weights=None):
        calls["weights"] = weights
        return _TinyResNet()

    with patch("trkh.models.model.tv_models.resnet50", fake_resnet50):
        model = create_model(
            num_classes=5,
            model_config={"model_type": "resnet50", "pretrained": False},
        )

    assert calls["weights"] is None
    assert model.fc.out_features == 5


def test_custom_trkh_model_rejects_pretrained_flag() -> None:
    with pytest.raises(ValueError, match="pretrained.*resnet50"):
        create_model(
            num_classes=5,
            model_config={"model_type": "vit_registers", "pretrained": True},
        )


def test_external_weight_path_stays_blocked() -> None:
    with pytest.raises(ValueError, match="external checkpoint|weights path|Tham so bi chan"):
        create_model(
            num_classes=5,
            model_config={"model_type": "resnet50", "weights_path": "leaky.pt"},
        )

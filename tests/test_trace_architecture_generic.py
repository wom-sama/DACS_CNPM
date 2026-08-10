import torch
from torch import nn

from trkh.tools.trace_architecture import (
    _forward_generic_feature_trace,
    _generic_spatial_activation_map,
    _supports_trkh_feature_trace,
)


class _GenericStageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.levels = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(16, 24, 3, padding=1), nn.ReLU()),
            ]
        )
        self.norm = nn.BatchNorm2d(24)
        self.head = nn.Linear(24, 5)

    def forward_features(self, images):
        features = self.patch_embed(images)
        for level in self.levels:
            features = level(features)
        return self.norm(features)

    def forward(self, images):
        features = self.forward_features(images)
        return self.head(features.mean(dim=(2, 3)))


class _TrkhTraceStub(nn.Module):
    def forward_features(self, images, *, bbox_token_prior=None, return_trace=False):
        return {"images": images, "trace": return_trace, "bbox": bbox_token_prior}


def test_generic_stage_trace_captures_spatial_features_and_logits():
    model = _GenericStageClassifier().eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        logits, features = _forward_generic_feature_trace(model, images)

    assert logits.shape == (2, 5)
    assert list(features) == ["patch_embed", "levels_01", "levels_02", "norm"]
    assert features["patch_embed"].shape == (2, 8, 8, 8)
    assert features["levels_02"].shape == (2, 24, 8, 8)
    activation_map = _generic_spatial_activation_map(features["levels_02"])
    assert activation_map is not None
    assert activation_map.shape == (8, 8)
    assert torch.isfinite(activation_map).all()


def test_trace_mode_detection_distinguishes_generic_and_trkh_signatures():
    assert _supports_trkh_feature_trace(_GenericStageClassifier()) is False
    assert _supports_trkh_feature_trace(_TrkhTraceStub()) is True

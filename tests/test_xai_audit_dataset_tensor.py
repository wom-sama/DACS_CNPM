import torch
from PIL import Image

from trkh.evaluation.attention_viz import (
    _disable_inplace_modules_for_hooks,
    _supports_trkh_feature_metadata as _viz_supports_trkh_feature_metadata,
)
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.xai_audit import (
    _crop_image_from_tensor,
    _dataset_tensor_and_crop,
    _forward_logits_with_optional_bbox,
    _supports_trkh_feature_metadata,
    _tensor_from_crop,
)


class _TimmLikeClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(1, 2)
        self.forward_features_called = False

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_features_called = True
        raise AssertionError("timm-style forward_features must not receive TRKH metadata")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.head(pooled)


def test_crop_image_from_normalized_tensor_round_trips_shape() -> None:
    image = torch.zeros(3, 12, 10)
    image[0].fill_((0.70 - 0.485) / 0.229)
    image[1].fill_((0.50 - 0.456) / 0.224)
    image[2].fill_((0.30 - 0.406) / 0.225)

    crop = _crop_image_from_tensor(image)

    assert crop.mode == "RGB"
    assert crop.size == (10, 12)
    red, green, blue = crop.getpixel((0, 0))
    assert abs(red - 178) <= 1
    assert abs(green - 128) <= 1
    assert abs(blue - 76) <= 1


def test_dataset_tensor_and_crop_uses_dataset_tensor() -> None:
    image = torch.rand(3, 8, 8)
    crop, batch = _dataset_tensor_and_crop((image, 1, {}), device=torch.device("cpu"))

    assert crop.size == (8, 8)
    assert tuple(batch.shape) == (1, 3, 8, 8)
    assert torch.allclose(batch.squeeze(0), image)


def test_dataset_tensor_and_crop_rejects_non_tensor_item() -> None:
    assert _dataset_tensor_and_crop(("not-a-tensor", 1, {}), device=torch.device("cpu")) is None


def test_checkpoint_input_normalization_uses_timm_values() -> None:
    checkpoint = {
        "model_config": {
            "input_mean": [0.5, 0.45, 0.4],
            "input_std": [0.2, 0.25, 0.3],
        }
    }

    assert checkpoint_input_normalization(checkpoint) == (
        (0.5, 0.45, 0.4),
        (0.2, 0.25, 0.3),
    )


def test_tensor_from_crop_uses_checkpoint_input_normalization() -> None:
    checkpoint = {
        "model_config": {
            "image_size": 4,
            "input_mean": [0.5, 0.25, 0.125],
            "input_std": [0.5, 0.25, 0.125],
        },
        "augmentation_config": {"resize_mode": "stretch"},
    }
    image = Image.new("RGB", (4, 4), (128, 64, 32))

    tensor = _tensor_from_crop(checkpoint, image, torch.device("cpu")).squeeze(0)

    expected_rgb = torch.tensor([128 / 255, 64 / 255, 32 / 255], dtype=torch.float32)
    expected = torch.tensor(
        [
            (expected_rgb[0] - 0.5) / 0.5,
            (expected_rgb[1] - 0.25) / 0.25,
            (expected_rgb[2] - 0.125) / 0.125,
        ],
        dtype=torch.float32,
    ).view(3, 1, 1)
    assert torch.allclose(tensor, expected.expand_as(tensor), atol=1e-5)


def test_crop_image_from_tensor_uses_supplied_normalization() -> None:
    image = torch.zeros(3, 6, 6)
    image[0].fill_((0.20 - 0.5) / 0.5)
    image[1].fill_((0.25 - 0.25) / 0.25)
    image[2].fill_((0.125 - 0.125) / 0.125)

    crop = _crop_image_from_tensor(
        image,
        mean=(0.5, 0.25, 0.125),
        std=(0.5, 0.25, 0.125),
    )

    assert crop.getpixel((0, 0)) == (51, 64, 32)


def test_forward_logits_falls_back_for_timm_style_forward_features() -> None:
    model = _TimmLikeClassifier()
    images = torch.ones(2, 3, 8, 8)
    bbox = torch.zeros(2, 4)
    image_valid_mask = torch.ones(2, 8, 8, dtype=torch.bool)

    logits = _forward_logits_with_optional_bbox(
        model,
        images,
        bbox,
        image_valid_mask=image_valid_mask,
    )

    assert tuple(logits.shape) == (2, 2)
    assert not model.forward_features_called
    assert not _supports_trkh_feature_metadata(model)
    assert not _viz_supports_trkh_feature_metadata(model)


def test_disable_inplace_modules_for_hooked_gradcam_models() -> None:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=1),
        torch.nn.Hardswish(inplace=True),
        torch.nn.ReLU(inplace=True),
    )

    changed = _disable_inplace_modules_for_hooks(model)

    assert changed == ["1", "2"]
    assert model[1].inplace is False
    assert model[2].inplace is False
    assert _disable_inplace_modules_for_hooks(model) == []

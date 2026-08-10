from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from trkh.data.dataset import (
    _amplify_surface_detail_image,
    _surface_detail_foreground_mask_array,
    build_eval_transform,
    build_train_transform,
)


def _synthetic_surface_image(size: int = 96) -> Image.Image:
    image = Image.new("RGB", (size, size), (175, 165, 95))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 14, size - 10, size - 12), fill=(176, 178, 80))
    for x in range(22, size - 18, 12):
        draw.line((x, 22, x + 14, size - 28), fill=(130, 112, 48), width=1)
    for x, y in ((38, 36), (56, 48), (44, 62), (66, 38)):
        draw.ellipse((x - 3, y - 2, x + 3, y + 2), fill=(92, 66, 32))
    return image


def test_surface_detail_amplification_none_returns_same_pixels() -> None:
    image = _synthetic_surface_image()
    result = _amplify_surface_detail_image(
        image,
        mode="none",
        strength=0.8,
    )
    assert np.array_equal(np.asarray(image), np.asarray(result))


def test_surface_detail_amplification_foreground_unsharp_changes_surface() -> None:
    image = _synthetic_surface_image()
    result = _amplify_surface_detail_image(
        image,
        mode="foreground_unsharp",
        strength=0.7,
        blur_radius=1.2,
        foreground_weight=0.9,
    )
    delta = np.abs(np.asarray(result, dtype=np.int16) - np.asarray(image, dtype=np.int16))
    assert float(delta.mean()) > 0.05
    assert result.size == image.size


def test_surface_detail_mask_is_not_full_frame_on_padded_image() -> None:
    image = Image.new("RGB", (96, 96), (128, 128, 128))
    fruit = _synthetic_surface_image(size=72)
    image.paste(fruit, (12, 12))

    mask = _surface_detail_foreground_mask_array(image, margin=0.08)
    fraction = float(mask.mean())

    assert 0.10 < fraction < 0.80


def test_surface_detail_amplification_train_and_eval_transform_shapes() -> None:
    image = _synthetic_surface_image()
    train_transform = build_train_transform(
        image_size=64,
        resize_mode="pad",
        surface_detail_amplification_mode="foreground_luma",
        surface_detail_amplification_probability=1.0,
        surface_detail_amplification_strength=0.6,
        random_erasing_probability=0.0,
        randaugment_num_ops=0,
        lighting_probability=0.0,
    )
    eval_transform = build_eval_transform(
        image_size=64,
        resize_mode="pad",
        surface_detail_amplification_mode="foreground_luma",
        surface_detail_amplification_strength=0.6,
    )

    train_tensor = train_transform(image)
    eval_tensor = eval_transform(image)

    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(eval_tensor, torch.Tensor)
    assert tuple(train_tensor.shape) == (3, 64, 64)
    assert tuple(eval_tensor.shape) == (3, 64, 64)
    assert torch.isfinite(train_tensor).all()
    assert torch.isfinite(eval_tensor).all()

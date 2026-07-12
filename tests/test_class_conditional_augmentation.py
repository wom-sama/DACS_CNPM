from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from trkh.data.dataset import (
    ClassificationFolderDataset,
    HybridImageTransform,
    build_train_transform,
    normalize_class_conditional_augmentation_scales,
)
from trkh.training.train import parse_class_conditional_augmentation_scales


def _gradient_image(width: int = 48, height: int = 40) -> Image.Image:
    values = []
    for y in range(height):
        for x in range(width):
            values.append((x * 5 % 256, y * 7 % 256, (x * 3 + y * 2) % 256))
    image = Image.new("RGB", (width, height))
    image.putdata(values)
    return image


def _target(scale: float | None) -> dict[str, torch.Tensor]:
    target = {
        "labels": torch.tensor([1], dtype=torch.long),
        "boxes": torch.tensor([[0.5, 0.5, 0.70, 0.65]], dtype=torch.float32),
    }
    if scale is not None:
        target["augmentation_scale"] = torch.tensor([scale], dtype=torch.float32)
    return target


def _transform() -> HybridImageTransform:
    return build_train_transform(
        image_size=64,
        resize_mode="pad",
        scale_min=0.88,
        scale_crop_probability=0.35,
        brightness=0.04,
        contrast=0.04,
        saturation=0.02,
        hue=0.01,
        random_erasing_probability=0.0,
        random_affine_degrees=3.0,
        random_affine_translate=0.02,
        random_affine_scale_min=0.96,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.0,
        rotate90_probability=0.03,
        lighting_probability=0.0,
        illumination_normalization=True,
        illumination_normalization_strength=0.35,
        background_suppression_mode="desaturate_blur",
        background_suppression_probability=0.8,
        background_suppression_margin=0.08,
        background_suppression_blur_radius=7.0,
        local_exposure_probability=0.15,
        local_exposure_strength=0.25,
        obstacle_probability=0.04,
        obstacle_max_area=0.08,
        scale_photometric_with_augmentation=False,
    )


def test_scale_parser_is_fail_closed() -> None:
    expected = [1.0, 0.5, 1.0, 1.0, 1.0]
    assert parse_class_conditional_augmentation_scales("1,.5,1,1,1", 5) == expected
    assert parse_class_conditional_augmentation_scales("", 5) == []
    assert normalize_class_conditional_augmentation_scales(expected, num_classes=5) == expected

    with pytest.raises(ValueError, match="dung 5"):
        parse_class_conditional_augmentation_scales("1,.5,1", 5)
    with pytest.raises(ValueError, match=r"\[0.1, 1.0\]"):
        parse_class_conditional_augmentation_scales("1,0,1,1,1", 5)
    with pytest.raises(ValueError, match="cac so"):
        parse_class_conditional_augmentation_scales("1,bad,1,1,1", 5)


def test_explicit_policy_changes_only_selected_class_scale(tmp_path: Path) -> None:
    class_names = [f"class_{index}" for index in range(5)]
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        Image.new("RGB", (16, 16), color=(30, 80, 120)).save(class_dir / "sample.png")

    dataset = ClassificationFolderDataset(
        root_dir=tmp_path,
        class_names=class_names,
        class_aware_augmentation=False,
        class_conditional_augmentation_scales=[1.0, 0.5, 1.0, 1.0, 1.0],
    )

    assert [dataset._augmentation_scale_for_label(index) for index in range(5)] == [
        1.0,
        0.5,
        1.0,
        1.0,
        1.0,
    ]
    report = dataset.quality_report()
    assert report["class_conditional_augmentation_scales"] == [1.0, 0.5, 1.0, 1.0, 1.0]
    assert report["effective_class_augmentation_scales"] == [1.0, 0.5, 1.0, 1.0, 1.0]


def test_empty_policy_keeps_scale_one_transform_bit_exact() -> None:
    transform = _transform()
    image = _gradient_image()

    torch.manual_seed(1701)
    legacy_tensor, legacy_target = transform(image, target=_target(None))
    torch.manual_seed(1701)
    explicit_tensor, explicit_target = transform(image, target=_target(1.0))

    assert torch.equal(legacy_tensor, explicit_tensor)
    assert torch.equal(legacy_target["boxes"], explicit_target["boxes"])
    assert torch.equal(legacy_target["image_mask"], explicit_target["image_mask"])


def test_half_scale_reduces_transform_and_preserves_target_contract() -> None:
    transform = _transform()
    image = _gradient_image()

    torch.manual_seed(90210)
    control_tensor, control_target = transform(image, target=_target(1.0))
    torch.manual_seed(90210)
    candidate_tensor, candidate_target = transform(image, target=_target(0.5))

    assert not torch.equal(control_tensor, candidate_tensor)
    assert candidate_target["labels"].tolist() == [1]
    assert candidate_target["boxes"].shape == (1, 4)
    assert bool(((candidate_target["boxes"] >= 0.0) & (candidate_target["boxes"] <= 1.0)).all())
    assert candidate_target["image_mask"].dtype == torch.bool
    assert candidate_target["image_mask"].shape == control_target["image_mask"].shape
    assert HybridImageTransform._probability_scale(0.5, boost_power=0.5) == pytest.approx(0.5)
    assert HybridImageTransform._probability_scale(1.8, boost_power=0.5) == pytest.approx(
        1.8**0.5
    )


def test_v8_launcher_forwards_policy_only_when_requested() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")

    assert '[string]$ClassConditionalAugmentationScales = ""' in script
    assert '"--class-conditional-augmentation-scales"' in script
    assert "IsNullOrWhiteSpace($ClassConditionalAugmentationScales)" in script
    assert "class_conditional_augmentation_scales = $ClassConditionalAugmentationScales" in script

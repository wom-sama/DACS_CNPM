from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    MixedTrainDataset,
    PairedViewTrainDataset,
    TrainBatchCollator,
)


def test_yolo_classification_source_context_keeps_full_frame_and_spotlights_target(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    draw.rectangle((65, 20, 85, 60), fill=(30, 30, 210))
    image.save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                "1 0.250000 0.500000 0.200000 0.500000",
                "0 0.750000 0.500000 0.200000 0.500000",
            ]
        ),
        encoding="utf-8",
    )

    dataset = MangoYOLOCropDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=None,
        num_classes=2,
        classification_target=True,
        classification_object_crops=True,
        classification_source_context=True,
        classification_source_context_mode="gray",
        classification_source_context_margin_ratio=0.0,
        classification_source_context_background_alpha=0.0,
        classification_source_context_blur_radius=1.0,
    )

    first_image, first_label = dataset[0]
    second_image, second_label = dataset[1]

    assert len(dataset) == 2
    assert first_label == 1
    assert second_label == 0
    assert first_image.size == (100, 80)
    assert second_image.size == (100, 80)
    assert dataset.quality_report()["crop_to_primary_object"] is False
    assert dataset.quality_report()["classification_source_context"]["enabled"] is True

    target_pixel = first_image.getpixel((25, 40))
    other_object_pixel = first_image.getpixel((75, 40))
    assert target_pixel[0] > target_pixel[1] + 80
    assert abs(other_object_pixel[0] - other_object_pixel[1]) <= 2
    assert abs(other_object_pixel[1] - other_object_pixel[2]) <= 2


def test_yolo_classification_source_context_crop_inset_preserves_object_crop(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    draw.rectangle((65, 20, 85, 60), fill=(30, 30, 210))
    image.save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                "1 0.250000 0.500000 0.200000 0.500000",
                "0 0.750000 0.500000 0.200000 0.500000",
            ]
        ),
        encoding="utf-8",
    )

    dataset = MangoYOLOCropDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=None,
        crop_margin_ratio=0.0,
        num_classes=2,
        classification_target=True,
        classification_object_crops=True,
        classification_source_context=True,
        classification_source_context_layout="crop_inset",
        classification_source_context_mode="gray",
        classification_source_context_margin_ratio=0.0,
        classification_source_context_background_alpha=0.0,
        classification_source_context_blur_radius=1.0,
        classification_source_context_inset_scale=0.45,
    )

    first_image, first_label = dataset[0]

    assert first_label == 1
    assert first_image.size[0] < 100
    assert first_image.size[1] < 80
    report = dataset.quality_report()["classification_source_context"]
    assert report["enabled"] is True
    assert report["layout"] == "crop_inset"
    center_pixel = first_image.getpixel((first_image.size[0] // 2, first_image.size[1] // 2))
    assert center_pixel[0] > center_pixel[1] + 80


def test_yolo_classification_source_context_aux_returns_crop_plus_full_context_metadata(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    draw.rectangle((65, 20, 85, 60), fill=(30, 30, 210))
    image.save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                "1 0.250000 0.500000 0.200000 0.500000",
                "0 0.750000 0.500000 0.200000 0.500000",
            ]
        ),
        encoding="utf-8",
    )

    def transform(img, target):
        transformed_target = dict(target)
        transformed_target["image_mask"] = torch.ones(
            (img.height, img.width),
            dtype=torch.bool,
        )
        return TF.to_tensor(img), transformed_target

    dataset = MangoYOLOCropDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=transform,
        crop_margin_ratio=0.0,
        num_classes=2,
        classification_target=True,
        classification_object_crops=True,
        classification_source_context_aux=True,
        classification_source_context_mode="gray",
        classification_source_context_margin_ratio=0.0,
        classification_source_context_background_alpha=0.0,
        classification_source_context_blur_radius=1.0,
    )

    image_tensor, label, metadata = dataset[0]

    assert label == 1
    assert image_tensor.shape[-2:] == (40, 20)
    assert metadata["source_context_image"].shape[-2:] == (80, 100)
    assert metadata["source_context_image_mask"].shape == (80, 100)
    assert metadata["source_context_image_mask"].dtype == torch.bool
    assert torch.allclose(
        metadata["source_context_bbox"],
        torch.tensor([0.25, 0.5, 0.2, 0.5]),
        atol=1e-5,
    )
    assert metadata["image_mask"].shape == (40, 20)
    report = dataset.quality_report()["classification_source_context"]
    assert report["enabled"] is False
    assert report["aux_enabled"] is True

    collator = TrainBatchCollator(num_classes=2, batch_mix_probability=0.0)
    images, labels, batch_metadata = collator([dataset[0], dataset[1]])
    assert images.shape == (2, 3, 40, 20)
    assert labels.tolist() == [1, 0]
    assert batch_metadata["source_context_image"].shape == (2, 3, 80, 100)
    assert batch_metadata["source_context_image_mask"].shape == (2, 80, 100)
    assert batch_metadata["source_context_image_mask"].dtype == torch.bool
    assert batch_metadata["source_context_bbox"].shape == (2, 4)
    assert batch_metadata["image_mask"].shape == (2, 40, 20)
    assert batch_metadata["image_mask"].dtype == torch.bool


def test_classification_folder_can_pair_yolo_bbox_and_source_context_aux(
    tmp_path: Path,
) -> None:
    class_root = tmp_path / "class_f" / "train"
    (class_root / "class0").mkdir(parents=True)
    (class_root / "class1").mkdir(parents=True)
    crop_path = class_root / "class1" / "Image_1_box000.jpg"
    Image.new("RGB", (48, 48), (90, 140, 80)).save(crop_path)

    yolo_images_dir = tmp_path / "yolo_f" / "images" / "train"
    yolo_labels_dir = tmp_path / "yolo_f" / "labels" / "train"
    yolo_images_dir.mkdir(parents=True)
    yolo_labels_dir.mkdir(parents=True)
    source = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(source)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    source.save(yolo_images_dir / "Image_1.jpg")
    yolo_labels_dir.joinpath("Image_1.txt").write_text(
        "0 0.250000 0.500000 0.200000 0.500000\n",
        encoding="utf-8",
    )

    def transform(img, target):
        transformed_target = dict(target)
        transformed_target["image_mask"] = torch.ones(
            (img.height, img.width),
            dtype=torch.bool,
        )
        return TF.to_tensor(img), transformed_target

    dataset = ClassificationFolderDataset(
        root_dir=class_root,
        class_names=["class0", "class1"],
        transform=transform,
        split="train",
        paired_yolo_images_dir=yolo_images_dir,
        paired_yolo_labels_dir=yolo_labels_dir,
        paired_yolo_class_names=["class0", "class1"],
        classification_source_context_aux=True,
        classification_source_context_mode="gray",
        classification_source_context_margin_ratio=0.0,
        classification_source_context_background_alpha=0.0,
        classification_source_context_blur_radius=1.0,
        classification_bbox_metadata=True,
    )

    assert len(dataset) == 1
    image_tensor, label, metadata = dataset[0]

    assert label == 1
    assert image_tensor.shape[-2:] == (48, 48)
    assert torch.allclose(
        metadata["bbox"],
        torch.tensor([0.25, 0.5, 0.2, 0.5]),
        atol=1e-5,
    )
    assert torch.allclose(
        metadata["crop_bbox"],
        torch.tensor([0.5, 0.5, 1.0, 1.0]),
        atol=1e-5,
    )
    assert metadata["source_context_image"].shape[-2:] == (80, 100)
    assert metadata["source_context_image_mask"].shape == (80, 100)
    assert metadata["source_context_image_mask"].dtype == torch.bool
    assert torch.allclose(
        metadata["source_context_bbox"],
        torch.tensor([0.25, 0.5, 0.2, 0.5]),
        atol=1e-5,
    )
    assert metadata["image_mask"].shape == (48, 48)
    report = dataset.quality_report()
    assert report["paired_yolo"]["enabled"] is True
    assert report["paired_yolo"]["mapped_count"] == 1
    assert report["paired_yolo"]["label_mismatch_count"] == 1
    assert report["classification_bbox_metadata"] is True
    assert report["classification_source_context"]["aux_enabled"] is True

    collator = TrainBatchCollator(num_classes=2, batch_mix_probability=0.0)
    images, labels, batch_metadata = collator([dataset[0], dataset[0]])
    assert images.shape == (2, 3, 48, 48)
    assert labels.tolist() == [1, 1]
    assert batch_metadata["bbox"].shape == (2, 4)
    assert batch_metadata["crop_bbox"].shape == (2, 4)
    assert batch_metadata["source_context_image"].shape == (2, 3, 80, 100)
    assert batch_metadata["source_context_image_mask"].shape == (2, 80, 100)
    assert batch_metadata["source_context_image_mask"].dtype == torch.bool
    assert batch_metadata["source_context_bbox"].shape == (2, 4)
    assert batch_metadata["image_mask"].shape == (2, 48, 48)
    assert batch_metadata["image_mask"].dtype == torch.bool


def test_mixed_train_dataset_concatenates_yolo_and_paired_classification_metadata(
    tmp_path: Path,
) -> None:
    yolo_images_dir = tmp_path / "yolo_f" / "images" / "train"
    yolo_labels_dir = tmp_path / "yolo_f" / "labels" / "train"
    yolo_images_dir.mkdir(parents=True)
    yolo_labels_dir.mkdir(parents=True)
    source = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(source)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    source.save(yolo_images_dir / "Image_1.jpg")
    yolo_labels_dir.joinpath("Image_1.txt").write_text(
        "1 0.250000 0.500000 0.200000 0.500000\n",
        encoding="utf-8",
    )

    class_root = tmp_path / "class_f" / "train"
    (class_root / "class0").mkdir(parents=True)
    (class_root / "class1").mkdir(parents=True)
    Image.new("RGB", (48, 48), (90, 140, 80)).save(
        class_root / "class1" / "Image_1_box000.jpg"
    )

    def transform(img, target):
        resized = img.resize((48, 48))
        transformed_target = dict(target)
        transformed_target["image_mask"] = torch.ones(
            (48, 48),
            dtype=torch.bool,
        )
        return TF.to_tensor(resized), transformed_target

    yolo_dataset = MangoYOLOCropDataset(
        images_dir=yolo_images_dir,
        labels_dir=yolo_labels_dir,
        transform=transform,
        crop_margin_ratio=0.0,
        num_classes=2,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    class_dataset = ClassificationFolderDataset(
        root_dir=class_root,
        class_names=["class0", "class1"],
        transform=transform,
        split="train",
        paired_yolo_images_dir=yolo_images_dir,
        paired_yolo_labels_dir=yolo_labels_dir,
        paired_yolo_class_names=["class0", "class1"],
        classification_bbox_metadata=True,
    )
    mixed = MixedTrainDataset(
        datasets=[yolo_dataset, class_dataset],
        source_names=["yolo", "class"],
        source_weights=[1.0, 1.0],
        seed=7,
    )

    assert len(mixed) == 2
    assert mixed.class_counts(2) == [0, 2]
    report = mixed.quality_report()
    assert report["data_format"] == "mixed_train"
    assert report["mixed_train"]["source_count"] == 2

    batch = [mixed[0], mixed[1]]
    collator = TrainBatchCollator(num_classes=2, batch_mix_probability=0.0)
    images, labels, metadata = collator(batch)

    assert images.shape == (2, 3, 48, 48)
    assert labels.tolist() == [1, 1]
    assert metadata["bbox"].shape == (2, 4)
    assert metadata["crop_bbox"].shape == (2, 4)
    assert metadata["image_mask"].shape == (2, 48, 48)
    assert metadata["image_mask"].dtype == torch.bool


def test_paired_view_train_dataset_attaches_class_crop_view_by_yolo_object_key(
    tmp_path: Path,
) -> None:
    yolo_images_dir = tmp_path / "yolo_f" / "images" / "train"
    yolo_labels_dir = tmp_path / "yolo_f" / "labels" / "train"
    yolo_images_dir.mkdir(parents=True)
    yolo_labels_dir.mkdir(parents=True)
    source = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(source)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    source.save(yolo_images_dir / "Image_1.jpg")
    yolo_labels_dir.joinpath("Image_1.txt").write_text(
        "1 0.250000 0.500000 0.200000 0.500000\n",
        encoding="utf-8",
    )

    class_root = tmp_path / "class_f" / "train"
    (class_root / "class0").mkdir(parents=True)
    (class_root / "class1").mkdir(parents=True)
    Image.new("RGB", (48, 48), (90, 140, 80)).save(
        class_root / "class1" / "Image_1_box000.jpg"
    )

    def transform(img, target):
        resized = img.resize((48, 48))
        transformed_target = dict(target)
        transformed_target["image_mask"] = torch.ones((48, 48), dtype=torch.bool)
        return TF.to_tensor(resized), transformed_target

    yolo_dataset = MangoYOLOCropDataset(
        images_dir=yolo_images_dir,
        labels_dir=yolo_labels_dir,
        transform=transform,
        crop_margin_ratio=0.0,
        num_classes=2,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    class_dataset = ClassificationFolderDataset(
        root_dir=class_root,
        class_names=["class0", "class1"],
        transform=transform,
        split="train",
        paired_yolo_images_dir=yolo_images_dir,
        paired_yolo_labels_dir=yolo_labels_dir,
        paired_yolo_class_names=["class0", "class1"],
        classification_bbox_metadata=True,
    )
    paired = PairedViewTrainDataset(
        primary_dataset=yolo_dataset,
        paired_dataset=class_dataset,
        primary_name="yolo",
        paired_name="class",
    )

    assert len(paired) == 1
    assert paired.class_counts(2) == [0, 1]
    image, label, metadata = paired[0]
    assert label == 1
    assert image.shape == (3, 48, 48)
    assert metadata["paired_view_image"].shape == (3, 48, 48)
    assert metadata["paired_view_label"] == 1
    assert torch.allclose(
        metadata["paired_view_bbox"],
        torch.tensor([0.25, 0.5, 0.2, 0.5]),
        atol=1e-5,
    )

    collator = TrainBatchCollator(num_classes=2, batch_mix_probability=0.0)
    images, labels, batch_metadata = collator([paired[0], paired[0]])
    assert images.shape == (2, 3, 48, 48)
    assert labels.tolist() == [1, 1]
    assert batch_metadata["paired_view_image"].shape == (2, 3, 48, 48)
    assert batch_metadata["paired_view_label"].tolist() == [1, 1]
    assert batch_metadata["paired_view_bbox"].shape == (2, 4)
    assert batch_metadata["paired_view_image_mask"].shape == (2, 48, 48)
    report = paired.quality_report()
    assert report["data_format"] == "paired_view_train"
    assert report["paired_view_train"]["matched_samples"] == 1

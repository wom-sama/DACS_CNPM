from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from trkh.core.config import DataSpec
from trkh.tools.evaluate_embedding_retrieval import OrderedYoloObjectDataset


def test_ordered_yolo_object_dataset_returns_sample_index_and_group_stem(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

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

    data_spec = DataSpec(
        data_yaml=tmp_path / "data.yaml",
        root=tmp_path,
        train_images=images_dir,
        train_labels=labels_dir,
        val_images=images_dir,
        val_labels=labels_dir,
        class_names=["negative", "focus"],
        class_name_mode="raw",
        data_format="yolo",
    )

    dataset = OrderedYoloObjectDataset(
        data_spec=data_spec,
        split="train",
        transform=TF.to_tensor,
        max_samples_per_class=1,
    )

    first_image, first_label, first_path, first_index, first_stem = dataset[0]
    second_image, second_label, second_path, second_index, second_stem = dataset[1]

    assert len(dataset) == 2
    assert first_image.ndim == 3
    assert second_image.ndim == 3
    assert {first_label, second_label} == {0, 1}
    assert first_path == second_path
    assert first_stem == second_stem == "sample"
    assert {first_index, second_index} == {0, 1}

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import yaml

from trkh.inference.deploy import (
    build_eval_loader,
    deterministic_stratified_indices,
)


CLASS_NAMES = [f"class_{index}" for index in range(5)]


def _write_classification_dataset(root: Path) -> Path:
    for split, count in (("train", 6), ("val", 2)):
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(count):
                Image.new(
                    "RGB",
                    (20, 16),
                    color=(20 * class_index, 10 * image_index, 80),
                ).save(class_dir / f"image_{image_index:02d}.png")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "format": "classification_folder",
                "path": str(root),
                "train": "train",
                "val": "val",
                "nc": 5,
                "class_name_mode": "raw",
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def _checkpoint() -> dict:
    return {
        "class_names": list(CLASS_NAMES),
        "model_config": {
            "model_type": "timm_classifier",
            "image_size": 16,
            "temporal_frames": 1,
            "input_mean": [0.485, 0.456, 0.406],
            "input_std": [0.229, 0.224, 0.225],
        },
        "augmentation_config": {
            "resize_mode": "stretch",
        },
    }


def test_deploy_loader_supports_classification_folder(tmp_path: Path) -> None:
    data_yaml = _write_classification_dataset(tmp_path / "class_f")
    loader, class_names = build_eval_loader(
        checkpoint=_checkpoint(),
        data_yaml=data_yaml,
        batch_size=4,
        num_workers=0,
        split="val",
        class_name_mode="raw",
        expected_num_classes=5,
    )

    images, labels = next(iter(loader))[:2]
    assert class_names == CLASS_NAMES
    assert len(loader.dataset) == 10
    assert images.shape == (4, 3, 16, 16)
    assert labels.shape == (4,)
    assert loader.trkh_sampling_summary["mode"] == "full_sequential"
    assert loader.trkh_sampling_summary["per_class"] == {
        str(index): 2 for index in range(5)
    }


def test_deploy_calibration_subset_is_stratified_and_deterministic(
    tmp_path: Path,
) -> None:
    data_yaml = _write_classification_dataset(tmp_path / "class_f")
    kwargs = dict(
        checkpoint=_checkpoint(),
        data_yaml=data_yaml,
        batch_size=4,
        num_workers=0,
        split="train",
        class_name_mode="raw",
        expected_num_classes=5,
        stratified_max_samples=12,
        sampling_seed=20260731,
    )
    first_loader, _ = build_eval_loader(**kwargs)
    second_loader, _ = build_eval_loader(**kwargs)

    assert first_loader.dataset.indices == second_loader.dataset.indices
    assert first_loader.trkh_sampling_summary["per_class"] == {
        "0": 3,
        "1": 3,
        "2": 2,
        "3": 2,
        "4": 2,
    }
    observed = []
    for batch in first_loader:
        labels = batch[1]
        observed.extend(int(label) for label in labels.tolist())
    assert len(observed) == 12
    assert set(observed) == set(range(5))


def test_deploy_loader_rejects_class_order_mismatch(tmp_path: Path) -> None:
    data_yaml = _write_classification_dataset(tmp_path / "class_f")
    checkpoint = _checkpoint()
    checkpoint["class_names"] = list(reversed(CLASS_NAMES))

    with pytest.raises(ValueError, match="class order mismatch"):
        build_eval_loader(
            checkpoint=checkpoint,
            data_yaml=data_yaml,
            batch_size=2,
            num_workers=0,
            split="val",
            class_name_mode="raw",
            expected_num_classes=5,
        )


def test_stratified_indices_do_not_duplicate_samples() -> None:
    indices, counts = deterministic_stratified_indices(
        [0, 0, 0, 1, 1, 2],
        max_samples=6,
        seed=42,
    )

    assert len(indices) == len(set(indices)) == 6
    assert counts == {0: 3, 1: 2, 2: 1}

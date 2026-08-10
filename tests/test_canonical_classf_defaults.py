from __future__ import annotations

from trkh.core.config import default_data_yaml, load_data_spec, project_dir
from trkh.tools.validate_canonical_classf import (
    CANONICAL_CLASS_NAMES,
    _image_tree_sha256,
)


def test_default_data_yaml_points_to_canonical_classification_export() -> None:
    normalized = default_data_yaml().as_posix().lower()
    assert normalized.endswith("/newdataset/class_f/data.yaml")


def test_checked_in_class_f_config_keeps_explicit_canonical_order() -> None:
    config_path = project_dir() / "configs" / "class_f_5class.yaml"
    data_spec = load_data_spec(
        config_path,
        class_name_mode="raw",
        expected_num_classes=5,
    )

    assert data_spec.data_format == "classification_folder"
    assert tuple(data_spec.class_names) == CANONICAL_CLASS_NAMES


def test_image_tree_digest_binds_relative_path_and_content(tmp_path) -> None:
    first = tmp_path / "train" / "class_a" / "sample.jpg"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    original = _image_tree_sha256(
        tmp_path,
        ["train/class_a/sample.jpg"],
    )

    first.write_bytes(b"changed")
    changed_content = _image_tree_sha256(
        tmp_path,
        ["train/class_a/sample.jpg"],
    )
    moved = tmp_path / "val" / "class_a" / "sample.jpg"
    moved.parent.mkdir(parents=True)
    moved.write_bytes(b"first")
    changed_path = _image_tree_sha256(
        tmp_path,
        ["val/class_a/sample.jpg"],
    )

    assert original != changed_content
    assert original != changed_path

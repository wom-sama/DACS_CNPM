from pathlib import Path

from trkh.tools.audit_yolo_source_groups import audit_yolo_source_groups


def _write_label(path: Path, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{label} 0.5 0.5 0.2 0.2" for label in labels]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_audit_yolo_source_groups_counts_mixed_and_same_label_groups(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (data_root / "images" / split).mkdir(parents=True)
        (data_root / "labels" / split).mkdir(parents=True)
    (data_root / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 3",
                "class_name_mode: raw",
                "names: [a, b, c]",
            ]
        ),
        encoding="utf-8",
    )
    _write_label(data_root / "labels" / "train" / "single.txt", [0])
    _write_label(data_root / "labels" / "train" / "mixed.txt", [0, 1, 1])
    _write_label(data_root / "labels" / "train" / "same_label.txt", [1, 1])
    _write_label(data_root / "labels" / "val" / "val_single.txt", [2])
    _write_label(data_root / "labels" / "test" / "test_single.txt", [1])

    output_dir = tmp_path / "audit"
    summary = audit_yolo_source_groups(data_root, output_dir, max_examples=10)

    train = summary["splits"]["train"]
    assert train["images_with_objects"] == 3
    assert train["objects"] == 6
    assert train["class_counts"] == {"0": 2, "1": 4}
    assert train["multi_object_images"] == 2
    assert train["same_label_multi_object_images"] == 1
    assert train["mixed_label_images"] == 1
    assert train["class1_multi_object_images"] == 2
    assert train["class1_same_label_multi_object_images"] == 1
    assert train["class1_mixed_label_images"] == 1
    assert train["mixed_label_pair_counts"] == {"0-1": 1}
    assert (output_dir / "source_group_summary.json").exists()
    assert "mixed.txt" in (output_dir / "mixed_label_examples.csv").read_text(encoding="utf-8")

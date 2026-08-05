from pathlib import Path

from trkh.tools.build_classification_grouped_split import (
    ClassificationRecord,
    build_groups,
)


def _record(path: Path, number: int, class_id: int) -> ClassificationRecord:
    path.write_bytes(f"image-{number}-{class_id}".encode("ascii"))
    return ClassificationRecord(
        source_path=path,
        original_split="train",
        class_id=class_id,
        class_name=f"class_{class_id}",
        source_id=f"Image_{number}",
        image_number=number,
    )


def test_near_image_sequence_stays_grouped_across_class_transitions(tmp_path: Path) -> None:
    records = [
        _record(tmp_path / "image_100.jpg", 100, 0),
        _record(tmp_path / "image_101.jpg", 101, 1),
        _record(tmp_path / "image_102.jpg", 102, 0),
    ]

    groups = build_groups(records, near_id_window=3, group_exact_duplicates=False)

    assert {frozenset(group) for group in groups} == {frozenset({0, 1, 2})}


def test_near_image_sequence_does_not_bridge_a_gap_beyond_window(tmp_path: Path) -> None:
    records = [
        _record(tmp_path / "image_100.jpg", 100, 0),
        _record(tmp_path / "image_101.jpg", 101, 1),
        _record(tmp_path / "image_105.jpg", 105, 2),
    ]

    groups = build_groups(records, near_id_window=3, group_exact_duplicates=False)

    assert {frozenset(group) for group in groups} == {
        frozenset({0, 1}),
        frozenset({2}),
    }

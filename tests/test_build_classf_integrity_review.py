from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from trkh.tools.build_classf_integrity_review import build_classf_integrity_review


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pattern(path: Path, *, invert: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), "white" if invert else "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 7, 31, 35), fill="black" if invert else "white")
    draw.ellipse((20, 11, 43, 40), fill=(90, 90, 90))
    image.save(path)


def _make_dataset(root: Path) -> tuple[Path, Path]:
    paths = {
        "train/a/Image_100_box000.png": (0, "a", "Image_100.jpg"),
        "train/b/Image_101_box000.png": (1, "b", "Image_101.jpg"),
        "val/a/Image_102_box000.png": (0, "a", "Image_102.jpg"),
        "train/a/Other_5_box000.png": (0, "a", "Other_5.jpg"),
    }
    first = root / "train/a/Image_100_box000.png"
    _pattern(first)
    for relative in ("train/b/Image_101_box000.png", "val/a/Image_102_box000.png"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(first.read_bytes())
    _pattern(root / "train/a/Other_5_box000.png", invert=True)
    sealed = root / "test/a/do_not_open.png"
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"not an image: the builder must never inspect canonical test")

    manifest = root / "manifest.csv"
    fieldnames = [
        "split",
        "source_image",
        "output_image",
        "class_id",
        "class_name",
        "source",
        "source_split",
        "leakage_group",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for relative, (class_id, class_name, source_name) in sorted(paths.items()):
            split = relative.split("/", 1)[0]
            writer.writerow(
                {
                    "split": split,
                    "source_image": str(root.parent / "provenance" / source_name),
                    "output_image": str((root / relative).resolve()),
                    "class_id": class_id,
                    "class_name": class_name,
                    "source": "0",
                    "source_split": "capture_a",
                    "leakage_group": f"group-{source_name}",
                }
            )
        writer.writerow(
            {
                "split": "test",
                "source_image": "sealed",
                "output_image": str(sealed.resolve()),
                "class_id": 0,
                "class_name": "a",
                "source": "sealed",
                "source_split": "sealed",
                "leakage_group": "sealed",
            }
        )

    archive = root / ".cvat_nhai_classification_archive"
    archive.mkdir()
    operations = archive / "operations.jsonl"
    current = paths["train/b/Image_101_box000.png"]
    after = {
        "split": "train",
        "source_image": str(root.parent / "provenance" / current[2]),
        "output_image": str((root / "train/b/Image_101_box000.png").resolve()),
        "class_id": "1",
        "class_name": "b",
        "source": "0",
        "source_split": "capture_a",
        "leakage_group": f"group-{current[2]}",
    }
    before = {**after, "class_id": "0", "class_name": "a"}
    operation = {
        "action": "change_classification",
        "status": "committed",
        "operation_id": "op-1",
        "timestamp": "2026-01-01T00:00:00",
        "manifest_changed_rows": [{"before": before, "after": after}],
    }
    sealed_operation = {
        **operation,
        "operation_id": "sealed-op",
        "manifest_changed_rows": [
            {"before": {**before, "split": "test"}, "after": {**after, "split": "test"}}
        ],
    }
    operations.write_text(
        json.dumps(operation, sort_keys=True) + "\n" + json.dumps(sealed_operation, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed, operations


def test_builds_deterministic_train_val_only_review_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "class_f"
    sealed, _ = _make_dataset(data_root)
    sealed_hash = _sha256(sealed)

    first_dir = tmp_path / "audit_a"
    second_dir = tmp_path / "audit_b"
    first = build_classf_integrity_review(data_root, first_dir, phash_distance=0, source_adjacency=1)
    second = build_classf_integrity_review(data_root, second_dir, phash_distance=0, source_adjacency=1)

    assert first["policy"]["canonical_splits_read"] == ["train", "val"]
    assert first["policy"]["canonical_test_accessed"] is False
    assert first["policy"]["model_predictions_used"] is False
    assert first["counts"]["samples"] == 4
    assert first["counts"]["relabelled_samples"] == 1
    assert first["attestation"] == second["attestation"]
    assert _sha256(sealed) == sealed_hash

    with (first_dir / "classf_train_val_integrity_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert all(row["split"] in {"train", "val"} for row in manifest_rows)
    relabelled = next(row for row in manifest_rows if "Image_101" in row["relative_path"])
    assert relabelled["was_relabelled"] == "True"
    assert relabelled["original_class_ids"] == "0"
    assert "relabel_history" in relabelled["train_review_flags"]
    assert "near_crop_cross_label" in relabelled["train_review_flags"]

    with (first_dir / "classf_human_review_queue.csv").open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    issue_types = {row["issue_type"] for row in review_rows}
    assert "relabel_history" in issue_types
    assert "source_adjacent_label_conflict" in issue_types
    assert "near_crop_label_conflict" in issue_types
    assert "near_crop_cross_split" in issue_types
    assert {row["recommended_action"] for row in review_rows} == {
        "human_review_only_no_auto_relabel_or_move"
    }


def test_refuses_to_write_inside_immutable_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "class_f"
    _make_dataset(data_root)
    with pytest.raises(ValueError, match="outside the immutable dataset"):
        build_classf_integrity_review(data_root, data_root / "derived")

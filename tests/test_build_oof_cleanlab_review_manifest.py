import csv
from pathlib import Path

from trkh.tools.build_oof_cleanlab_review_manifest import build_oof_cleanlab_review_manifest


def _write_cleanlab(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "path",
        "true_index",
        "true_name",
        "suggested_index",
        "suggested_name",
        "is_label_issue",
        "issue_rank",
        "label_quality",
        "self_confidence",
        "top1_confidence",
        "top2_margin",
        "normalized_entropy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_oof_cleanlab_review_manifest_filters_pairs_and_train_only(tmp_path: Path) -> None:
    train_a = tmp_path / "class_f" / "train" / "a" / "a.jpg"
    train_b = tmp_path / "class_f" / "train" / "b" / "b.jpg"
    val_a = tmp_path / "class_f" / "val" / "a" / "c.jpg"
    train_a.parent.mkdir(parents=True)
    train_b.parent.mkdir(parents=True)
    val_a.parent.mkdir(parents=True)
    train_a.write_bytes(b"a")
    train_b.write_bytes(b"b")
    val_a.write_bytes(b"c")

    cleanlab_csv = tmp_path / "cleanlab.csv"
    _write_cleanlab(
        cleanlab_csv,
        [
            {
                "path": str(train_a),
                "true_index": 0,
                "true_name": "a",
                "suggested_index": 1,
                "suggested_name": "b",
                "is_label_issue": 1,
                "issue_rank": 2,
                "label_quality": 0.1,
                "self_confidence": 0.1,
                "top1_confidence": 0.9,
                "top2_margin": 0.7,
                "normalized_entropy": 0.2,
            },
            {
                "path": str(train_b),
                "true_index": 1,
                "true_name": "b",
                "suggested_index": 0,
                "suggested_name": "a",
                "is_label_issue": 1,
                "issue_rank": 1,
                "label_quality": 0.2,
                "self_confidence": 0.2,
                "top1_confidence": 0.8,
                "top2_margin": 0.5,
                "normalized_entropy": 0.3,
            },
            {
                "path": str(val_a),
                "true_index": 0,
                "true_name": "a",
                "suggested_index": 1,
                "suggested_name": "b",
                "is_label_issue": 1,
                "issue_rank": 3,
                "label_quality": 0.3,
                "self_confidence": 0.3,
                "top1_confidence": 0.7,
                "top2_margin": 0.4,
                "normalized_entropy": 0.4,
            },
        ],
    )

    summary = build_oof_cleanlab_review_manifest(
        cleanlab_manifest=cleanlab_csv,
        output_dir=tmp_path / "review",
        pairs="0-1",
        max_per_direction=1,
        image_mode="copy",
    )

    assert summary["selected_rows"] == 2
    assert summary["skipped_non_train"] == 1
    assert summary["selected_by_direction"] == {"0->1": 1, "1->0": 1}
    assert summary["image_actions"] == {"copy": 2}

    with (tmp_path / "review" / "boundary_review_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["review_id"] for row in rows] == ["oof_00001", "oof_00002"]
    assert rows[0]["reason"] == "oof_cleanlab_1->0"
    assert rows[1]["boundary_pair"] == "0-1"

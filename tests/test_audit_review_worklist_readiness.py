import csv
from pathlib import Path

from trkh.tools.audit_review_worklist_readiness import audit_review_worklist_readiness


FIELDNAMES = [
    "review_id",
    "sample_index",
    "image_path",
    "transition",
    "strict_review_side",
    "strict_cluster_id",
    "manual_label_status",
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_review_worklist_readiness_blocks_empty_manual_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    _write_csv(
        csv_path,
        [
            {
                "review_id": "a",
                "sample_index": 1,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/a.jpg",
                "transition": "1->0",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "c1",
                "manual_label_status": "",
            },
            {
                "review_id": "b",
                "sample_index": 2,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/b.jpg",
                "transition": "0->1",
                "strict_review_side": "fp_suppressor",
                "strict_cluster_id": "c2",
                "manual_label_status": "",
            },
        ],
    )

    summary = audit_review_worklist_readiness(csv_path=csv_path)

    assert summary["ready_for_training_manifest"] is False
    assert summary["manual_label_status_filled"] == 0
    assert "manual_field_not_filled:2" in summary["blocking_reasons"]
    assert "actionable_recall_below_min:0<1" in summary["blocking_reasons"]
    assert "actionable_fp_below_min:0<1" in summary["blocking_reasons"]


def test_review_worklist_readiness_passes_balanced_actionable_review(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    _write_csv(
        csv_path,
        [
            {
                "review_id": "a",
                "sample_index": 1,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/a.jpg",
                "transition": "1->0",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "c1",
                "manual_label_status": "correct",
            },
            {
                "review_id": "b",
                "sample_index": 2,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/b.jpg",
                "transition": "0->1",
                "strict_review_side": "fp_suppressor",
                "strict_cluster_id": "c2",
                "manual_label_status": "wrong",
            },
        ],
    )

    summary = audit_review_worklist_readiness(csv_path=csv_path)

    assert summary["ready_for_training_manifest"] is True
    assert summary["manual_label_status_filled"] == 2
    assert summary["actionable_recall_count"] == 1
    assert summary["actionable_fp_count"] == 1
    assert summary["blocking_reasons"] == []


def test_review_worklist_readiness_flags_mixed_side_clusters(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    _write_csv(
        csv_path,
        [
            {
                "review_id": "a",
                "sample_index": 1,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/a.jpg",
                "transition": "1->0",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "same",
                "manual_label_status": "correct",
            },
            {
                "review_id": "b",
                "sample_index": 2,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/b.jpg",
                "transition": "0->1",
                "strict_review_side": "fp_suppressor",
                "strict_cluster_id": "same",
                "manual_label_status": "wrong",
            },
        ],
    )

    summary = audit_review_worklist_readiness(csv_path=csv_path)

    assert summary["ready_for_training_manifest"] is False
    assert "mixed_actionable_side_clusters:1" in summary["blocking_reasons"]
    assert summary["mixed_side_clusters"][0]["sides"] == {
        "recall_protector": 1,
        "fp_suppressor": 1,
    }
    assert summary["blocking_mixed_side_clusters"][0]["blocks_training"] is True


def test_review_worklist_readiness_accepts_multiple_side_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    _write_csv(
        csv_path,
        [
            {
                "review_id": "a",
                "sample_index": 1,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/a.jpg",
                "transition": "1->0",
                "strict_review_side": "fn1_all",
                "strict_cluster_id": "c1",
                "manual_label_status": "correct",
            },
            {
                "review_id": "b",
                "sample_index": 2,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/b.jpg",
                "transition": "1->1",
                "strict_review_side": "tp1_low_margin_recall_protector",
                "strict_cluster_id": "c2",
                "manual_label_status": "correct",
            },
            {
                "review_id": "c",
                "sample_index": 3,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/c.jpg",
                "transition": "0->1",
                "strict_review_side": "fp0_to_1_suppressor_review",
                "strict_cluster_id": "c3",
                "manual_label_status": "wrong",
            },
        ],
    )

    summary = audit_review_worklist_readiness(
        csv_path=csv_path,
        recall_side_value="fn1_all,tp1_low_margin_recall_protector",
        fp_side_value="fp0_to_1_suppressor_review",
        min_actionable_recall=2,
        min_actionable_fp=1,
    )

    assert summary["ready_for_training_manifest"] is True
    assert summary["actionable_recall_count"] == 2
    assert summary["actionable_fp_count"] == 1
    assert summary["recall_side_values"] == ["fn1_all", "tp1_low_margin_recall_protector"]


def test_review_worklist_readiness_treats_anchor_cluster_as_warning(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    _write_csv(
        csv_path,
        [
            {
                "review_id": "a",
                "sample_index": 1,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/a.jpg",
                "transition": "0->1",
                "strict_review_side": "fp_suppressor",
                "strict_cluster_id": "same",
                "manual_label_status": "wrong",
            },
            {
                "review_id": "b",
                "sample_index": 2,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/b.jpg",
                "transition": "0->0",
                "strict_review_side": "anchor",
                "strict_cluster_id": "same",
                "manual_label_status": "ignore",
            },
            {
                "review_id": "c",
                "sample_index": 3,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/c.jpg",
                "transition": "1->0",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "other",
                "manual_label_status": "correct",
            },
        ],
    )

    summary = audit_review_worklist_readiness(csv_path=csv_path)

    assert summary["ready_for_training_manifest"] is True
    assert len(summary["mixed_side_clusters"]) == 1
    assert summary["mixed_side_clusters"][0]["blocks_training"] is False
    assert summary["blocking_mixed_side_clusters"] == []

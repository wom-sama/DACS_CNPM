import csv
import json
from pathlib import Path

from trkh.tools.build_review_worklist_readiness_matrix import build_review_worklist_readiness_matrix


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_review_worklist_readiness_matrix_summarizes_ready_and_blocked(tmp_path: Path) -> None:
    ready_csv = tmp_path / "ready.csv"
    blocked_csv = tmp_path / "blocked.csv"
    _write_csv(
        ready_csv,
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
    _write_csv(
        blocked_csv,
        [
            {
                "review_id": "c",
                "sample_index": 3,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/c.jpg",
                "transition": "1->2",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "c3",
                "manual_label_status": "",
            }
        ],
    )
    config = tmp_path / "matrix_config.json"
    config.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "name": "ready queue",
                        "csv_path": str(ready_csv),
                        "min_actionable_recall": 1,
                        "min_actionable_fp": 1,
                    },
                    {
                        "name": "blocked queue",
                        "csv_path": str(blocked_csv),
                        "min_actionable_recall": 1,
                        "min_actionable_fp": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = build_review_worklist_readiness_matrix(
        config_json=config,
        output_dir=tmp_path / "matrix",
        created_at="2026-07-06T00:00:00",
        guardrail="matrix only",
    )

    assert summary["ready_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["raw_dataset_touched"] is False
    assert summary["trainable_manifest_written"] is False
    by_name = {item["name"]: item for item in summary["items"]}
    assert by_name["ready queue"]["ready_for_training_manifest"] is True
    assert by_name["blocked queue"]["ready_for_training_manifest"] is False
    assert "manual_field_not_filled:1" in by_name["blocked queue"]["blocking_reasons"]
    assert (tmp_path / "matrix" / "ready_queue" / "readiness_summary.json").is_file()
    assert (tmp_path / "matrix" / "summary.json").is_file()

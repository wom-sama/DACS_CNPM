import csv
import json
from pathlib import Path

from trkh.tools.build_review_minimum_fill_plan import build_review_minimum_fill_plan


FIELDNAMES = [
    "review_id",
    "sample_index",
    "image_path",
    "target_index",
    "prediction_index",
    "risk_transition",
    "strict_review_side",
    "strict_cluster_id",
    "manual_label_status",
    "manual_expected_class",
    "top2_margin",
    "manual_label_status_suggestion",
    "manual_expected_class_suggestion",
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_review_minimum_fill_plan_prioritizes_mixed_clusters_without_manifest(tmp_path: Path) -> None:
    queue_csv = tmp_path / "queue.csv"
    _write_csv(
        queue_csv,
        [
            {
                "review_id": "r1",
                "sample_index": 1,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/a.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "risk_transition": "1->0",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "safe_a",
                "manual_label_status": "",
                "top2_margin": "0.20",
            },
            {
                "review_id": "r2",
                "sample_index": 2,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/b.jpg",
                "target_index": 1,
                "prediction_index": 2,
                "risk_transition": "1->2",
                "strict_review_side": "recall_protector",
                "strict_cluster_id": "mixed_1",
                "manual_label_status": "",
                "top2_margin": "0.01",
            },
            {
                "review_id": "r3",
                "sample_index": 3,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/c.jpg",
                "target_index": 2,
                "prediction_index": 1,
                "risk_transition": "2->1",
                "strict_review_side": "fp_suppressor",
                "strict_cluster_id": "mixed_1",
                "manual_label_status": "",
                "top2_margin": "0.02",
            },
            {
                "review_id": "r4",
                "sample_index": 4,
                "image_path": "D:/DataAI/AIEx/newdataset/yolo_f/images/train/d.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "risk_transition": "0->1",
                "strict_review_side": "fp_suppressor",
                "strict_cluster_id": "safe_b",
                "manual_label_status": "",
                "top2_margin": "0.03",
            },
        ],
    )
    matrix_dir = tmp_path / "matrix"
    item_dir = matrix_dir / "rival_queue"
    item_dir.mkdir(parents=True)
    readiness_summary = {
        "blocking_mixed_side_clusters": [
            {
                "cluster": "mixed_1",
                "rows": 2,
                "sides": {"recall_protector": 1, "fp_suppressor": 1},
                "blocks_training": True,
            }
        ]
    }
    (item_dir / "readiness_summary.json").write_text(json.dumps(readiness_summary), encoding="utf-8")
    config_json = tmp_path / "matrix_config.json"
    config_json.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "name": "rival_queue",
                        "csv_path": str(queue_csv),
                        "side_column": "strict_review_side",
                        "recall_side_value": "recall_protector",
                        "fp_side_value": "fp_suppressor",
                        "transition_column": "risk_transition",
                        "cluster_column": "strict_cluster_id",
                        "neutral_side_value": "anchor,<blank>",
                        "min_actionable_recall": 2,
                        "min_actionable_fp": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix_summary = tmp_path / "matrix_summary.json"
    matrix_summary.write_text(
        json.dumps(
            {
                "config_json": str(config_json),
                "items": [
                    {
                        "name": "rival_queue",
                        "output_dir": str(item_dir),
                        "ready_for_training_manifest": False,
                        "actionable_recall_count": 0,
                        "actionable_fp_count": 0,
                        "blocking_reasons": [
                            "manual_field_not_filled:4",
                            "actionable_recall_below_min:0<2",
                            "actionable_fp_below_min:0<1",
                            "mixed_actionable_side_clusters:1",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    gate_summary = tmp_path / "gate_summary.json"
    gate_summary.write_text(
        json.dumps(
            {
                "smoke_gate_ready": False,
                "transition_coverage": {
                    "critical_transitions": [
                        {"transition": "1->2", "val_error_count": 11},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_review_minimum_fill_plan(
        matrix_summary_path=matrix_summary,
        gate_summary_path=gate_summary,
        output_dir=tmp_path / "fill_plan",
        created_at="2026-07-06T00:00:00",
    )

    assert summary["raw_dataset_touched"] is False
    assert summary["test_split_used"] is False
    assert summary["trainable_manifest_written"] is False
    assert summary["auto_labeling_performed"] is False
    assert summary["smoke_permission"] is False
    assert summary["selected_plan_rows"] == 4
    assert summary["queue_summaries"][0]["blocking_mixed_cluster_count"] == 1
    assert summary["queue_summaries"][0]["critical_transition_blank_rows"]["1->2"] == 1

    with (tmp_path / "fill_plan" / "priority_fill_plan.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fill_role"] == "mixed_cluster_resolution"
    assert rows[1]["fill_role"] == "mixed_cluster_resolution"
    assert {rows[0]["cluster_id"], rows[1]["cluster_id"]} == {"mixed_1"}
    assert any(row["transition"] == "1->2" for row in rows)
    assert (tmp_path / "fill_plan" / "summary.json").is_file()

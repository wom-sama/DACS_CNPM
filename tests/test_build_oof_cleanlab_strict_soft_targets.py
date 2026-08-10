import csv
from pathlib import Path

from trkh.tools.build_oof_cleanlab_strict_soft_targets import build_oof_cleanlab_strict_soft_targets


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_index",
        "image_path",
        "target_index",
        "prediction_index",
        "is_label_issue",
        "issue_rank",
        "target_probability",
        "confidence",
        "top2_margin",
        "prob_0",
        "prob_1",
        "prob_2",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_oof_cleanlab_strict_soft_targets_filters_and_writes_probabilities(tmp_path: Path) -> None:
    source = tmp_path / "review.csv"
    _write_manifest(
        source,
        [
            {
                "sample_index": 11,
                "image_path": tmp_path / "class_f" / "train" / "a" / "a.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "is_label_issue": 1,
                "issue_rank": 1,
                "target_probability": 0.01,
                "confidence": 0.99,
                "top2_margin": 0.90,
                "prob_0": 0.01,
                "prob_1": 0.99,
                "prob_2": 0.0,
            },
            {
                "sample_index": 12,
                "image_path": tmp_path / "class_f" / "val" / "a" / "b.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "is_label_issue": 1,
                "issue_rank": 2,
                "target_probability": 0.01,
                "confidence": 0.99,
                "top2_margin": 0.90,
                "prob_0": 0.99,
                "prob_1": 0.01,
                "prob_2": 0.0,
            },
            {
                "sample_index": 13,
                "image_path": tmp_path / "class_f" / "train" / "b" / "c.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "is_label_issue": 1,
                "issue_rank": 3,
                "target_probability": 0.20,
                "confidence": 0.80,
                "top2_margin": 0.30,
                "prob_0": 0.80,
                "prob_1": 0.20,
                "prob_2": 0.0,
            },
        ],
    )

    summary = build_oof_cleanlab_strict_soft_targets(
        source_manifest=source,
        output_dir=tmp_path / "out",
        pairs="0-1",
        alpha=0.85,
        min_top1_confidence=0.95,
        max_self_confidence=0.05,
        min_top2_margin=0.80,
    )

    assert summary["selected_rows"] == 1
    assert summary["rows_with_sample_index"] == 1
    assert summary["skipped_non_train"] == 1
    assert summary["skipped_threshold"] == 1
    assert summary["selected_by_direction"] == {"0->1": 1}

    with (tmp_path / "out" / "strict_oof_cleanlab_soft_targets_train_only.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_index"] == "11"
    assert rows[0]["soft_0"] == "0.15"
    assert rows[0]["soft_1"] == "0.85"
    assert rows[0]["soft_2"] == "0"

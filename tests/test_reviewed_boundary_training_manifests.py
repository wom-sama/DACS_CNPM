import csv
from pathlib import Path

import pytest

from trkh.tools.build_reviewed_boundary_training_manifests import build_manifests
from trkh.tools.build_reviewed_boundary_training_manifests import main as builder_main


FIELDNAMES = [
    "review_id",
    "sample_index",
    "source_row_index",
    "image_path",
    "target_index",
    "target_name",
    "prediction_index",
    "prediction_name",
    "top2_index",
    "top2_margin",
    "reason",
    "manual_label_status",
    "manual_expected_class",
    "quality_lighting",
    "quality_dirty_obstacle",
    "quality_partial_fruit",
    "quality_background_mask",
    "manual_sample_weight",
    "review_decision",
    "review_notes",
]


def _write_reviewed_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_reviewed_boundary_manifest_builds_training_artifacts(tmp_path):
    train_root = tmp_path / "class_f" / "train"
    reviewed = tmp_path / "reviewed.csv"
    _write_reviewed_manifest(
        reviewed,
        [
            {
                "review_id": "train_00001",
                "image_path": train_root / "c0" / "a.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "top2_index": 0,
                "top2_margin": 0.03,
                "reason": "focus_false_positive",
                "manual_label_status": "correct",
            },
            {
                "review_id": "train_00002",
                "image_path": train_root / "c1" / "b.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "top2_index": 1,
                "top2_margin": 0.02,
                "reason": "focus_false_negative",
                "manual_label_status": "ambiguous",
                "quality_lighting": "glare",
            },
            {
                "review_id": "train_00003",
                "image_path": train_root / "c1" / "c.jpg",
                "target_index": 1,
                "prediction_index": 4,
                "top2_index": 1,
                "top2_margin": 0.01,
                "reason": "focus_false_negative",
                "manual_label_status": "wrong",
                "manual_expected_class": 4,
            },
            {
                "review_id": "train_00004",
                "image_path": train_root / "c2" / "d.jpg",
                "target_index": 2,
                "prediction_index": 1,
                "top2_index": 2,
                "top2_margin": 0.01,
                "reason": "focus_false_positive",
                "manual_label_status": "needs_crop",
            },
        ],
    )

    summary = build_manifests(
        reviewed_manifest=reviewed,
        output_dir=tmp_path / "out",
        num_classes=5,
        soft_target_pairs="0-1,1-2,2-3",
        soft_alpha=0.25,
        correct_error_weight=1.10,
        ambiguous_weight=0.65,
        wrong_weight=0.20,
        needs_crop_weight=0.45,
        lighting_weight=0.85,
        dirty_weight=0.80,
        partial_weight=0.65,
        background_weight=0.85,
        targeted_margin=0.10,
        targeted_margin_weight=0.70,
        max_targeted_margin_weight=1.25,
        allow_non_train_paths=False,
        dry_run=False,
    )

    assert summary["sample_weight_rows"] == 4
    assert summary["soft_target_rows"] == 1
    assert summary["targeted_margin_rows"] == 1
    assert summary["relabel_candidate_rows"] == 1
    assert summary["by_soft_pair"] == {"1->0": 1}
    assert summary["by_targeted_pair"] == {"0->1": 1}
    assert summary["sample_index_rows"]["sample_weight"] == 0

    with (tmp_path / "out" / "ambiguous_soft_targets_reviewed_train_only.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        soft_rows = list(csv.DictReader(handle))
    assert soft_rows[0]["soft_0"] == "0.25"
    assert soft_rows[0]["soft_1"] == "0.75"

    with (tmp_path / "out" / "sample_weights_reviewed_train_only.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        weight_rows = {row["review_id"]: row for row in csv.DictReader(handle)}
    assert float(weight_rows["train_00001"]["sample_weight"]) == pytest.approx(1.10)
    assert float(weight_rows["train_00002"]["sample_weight"]) == pytest.approx(0.65)
    assert float(weight_rows["train_00003"]["sample_weight"]) == pytest.approx(0.20)
    assert float(weight_rows["train_00004"]["sample_weight"]) == pytest.approx(0.45)


def test_reviewed_boundary_manifest_preserves_sample_index_duplicates(tmp_path):
    train_root = tmp_path / "yolo_f" / "images" / "train"
    reviewed = tmp_path / "reviewed.csv"
    shared_path = train_root / "multi_object.jpg"
    _write_reviewed_manifest(
        reviewed,
        [
            {
                "review_id": "train_00001",
                "sample_index": 10,
                "image_path": shared_path,
                "target_index": 0,
                "prediction_index": 1,
                "reason": "focus_false_positive",
                "manual_label_status": "correct",
            },
            {
                "review_id": "train_00002",
                "sample_index": 11,
                "image_path": shared_path,
                "target_index": 1,
                "prediction_index": 0,
                "reason": "focus_false_negative",
                "manual_label_status": "correct",
            },
        ],
    )

    summary = build_manifests(
        reviewed_manifest=reviewed,
        output_dir=tmp_path / "out",
        num_classes=5,
        soft_target_pairs="0-1,1-2,2-3",
        soft_alpha=0.25,
        correct_error_weight=1.10,
        ambiguous_weight=0.65,
        wrong_weight=0.20,
        needs_crop_weight=0.45,
        lighting_weight=0.85,
        dirty_weight=0.80,
        partial_weight=0.65,
        background_weight=0.85,
        targeted_margin=0.10,
        targeted_margin_weight=0.70,
        max_targeted_margin_weight=1.25,
        allow_non_train_paths=False,
        dry_run=False,
    )

    assert summary["sample_weight_rows"] == 2
    assert summary["targeted_margin_rows"] == 2
    assert summary["sample_index_rows"]["sample_weight"] == 2
    assert summary["sample_index_rows"]["targeted_margin"] == 2

    with (tmp_path / "out" / "targeted_margin_reviewed_train_only.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        targeted_rows = list(csv.DictReader(handle))
    assert {row["sample_index"] for row in targeted_rows} == {"10", "11"}


def test_reviewed_boundary_manifest_rejects_non_train_rows(tmp_path):
    reviewed = tmp_path / "reviewed.csv"
    _write_reviewed_manifest(
        reviewed,
        [
            {
                "review_id": "val_00001",
                "image_path": tmp_path / "class_f" / "val" / "c0" / "a.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "manual_label_status": "correct",
            }
        ],
    )

    with pytest.raises(ValueError, match="train-only"):
        build_manifests(
            reviewed_manifest=reviewed,
            output_dir=tmp_path / "out",
            num_classes=5,
            soft_target_pairs="0-1,1-2,2-3",
            soft_alpha=0.25,
            correct_error_weight=1.10,
            ambiguous_weight=0.65,
            wrong_weight=0.20,
            needs_crop_weight=0.45,
            lighting_weight=0.85,
            dirty_weight=0.80,
            partial_weight=0.65,
            background_weight=0.85,
            targeted_margin=0.10,
            targeted_margin_weight=0.70,
            max_targeted_margin_weight=1.25,
            allow_non_train_paths=False,
            dry_run=False,
        )


def test_reviewed_boundary_manifest_accepts_manual_sample_weight_override(tmp_path):
    train_root = tmp_path / "class_f" / "train"
    reviewed = tmp_path / "reviewed.csv"
    _write_reviewed_manifest(
        reviewed,
        [
            {
                "review_id": "train_00001",
                "image_path": train_root / "c1" / "recall.jpg",
                "target_index": 1,
                "prediction_index": 1,
                "reason": "recall_protector",
                "manual_label_status": "correct",
                "manual_sample_weight": "1.07",
            },
            {
                "review_id": "train_00002",
                "image_path": train_root / "c0" / "glare.jpg",
                "target_index": 0,
                "prediction_index": 0,
                "reason": "low_quality_anchor",
                "manual_label_status": "correct",
                "manual_sample_weight": "1.20",
                "quality_lighting": "glare",
            },
        ],
    )

    summary = build_manifests(
        reviewed_manifest=reviewed,
        output_dir=tmp_path / "out",
        num_classes=5,
        soft_target_pairs="0-1,1-2,2-3",
        soft_alpha=0.25,
        correct_error_weight=1.10,
        ambiguous_weight=0.65,
        wrong_weight=0.20,
        needs_crop_weight=0.45,
        lighting_weight=0.85,
        dirty_weight=0.80,
        partial_weight=0.65,
        background_weight=0.85,
        targeted_margin=0.10,
        targeted_margin_weight=0.70,
        max_targeted_margin_weight=1.25,
        allow_non_train_paths=False,
        dry_run=False,
    )

    assert summary["manual_sample_weight_rows"] == 2
    assert summary["by_decision"] == {"manual_weight": 1, "quality_downweight": 1}
    with (tmp_path / "out" / "sample_weights_reviewed_train_only.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = {row["review_id"]: row for row in csv.DictReader(handle)}
    assert float(rows["train_00001"]["sample_weight"]) == pytest.approx(1.07)
    assert float(rows["train_00002"]["sample_weight"]) == pytest.approx(0.85)
    assert "manual_sample_weight=1.07" in rows["train_00001"]["reason"]


def test_reviewed_boundary_cli_blocks_non_dry_run_when_readiness_fails(tmp_path):
    train_root = tmp_path / "yolo_f" / "images" / "train"
    reviewed = tmp_path / "reviewed.csv"
    _write_reviewed_manifest(
        reviewed,
        [
            {
                "review_id": "train_00001",
                "sample_index": 10,
                "image_path": train_root / "c1" / "fn.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "reason": "focus_false_negative",
                "manual_label_status": "",
            },
            {
                "review_id": "train_00002",
                "sample_index": 11,
                "image_path": train_root / "c0" / "fp.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "reason": "focus_false_positive",
                "manual_label_status": "",
            },
        ],
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="readiness check failed"):
        builder_main(
            [
                "--reviewed-manifest",
                str(reviewed),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert (output_dir / "readiness_audit" / "readiness_summary.json").is_file()
    assert not (output_dir / "sample_weights_reviewed_train_only.csv").exists()


def test_reviewed_boundary_cli_allows_dry_run_with_readiness_warning(tmp_path, capsys):
    train_root = tmp_path / "yolo_f" / "images" / "train"
    reviewed = tmp_path / "reviewed.csv"
    _write_reviewed_manifest(
        reviewed,
        [
            {
                "review_id": "train_00001",
                "sample_index": 10,
                "image_path": train_root / "c1" / "fn.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "reason": "focus_false_negative",
                "manual_label_status": "correct",
            },
            {
                "review_id": "train_00002",
                "sample_index": 11,
                "image_path": train_root / "c0" / "fp.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "reason": "focus_false_positive",
                "manual_label_status": "correct",
            },
        ],
    )
    output_dir = tmp_path / "out"

    builder_main(
        [
            "--reviewed-manifest",
            str(reviewed),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert '"ready_for_training_manifest": false' in captured.out
    assert (output_dir / "readiness_audit" / "readiness_summary.json").is_file()
    assert not (output_dir / "sample_weights_reviewed_train_only.csv").exists()

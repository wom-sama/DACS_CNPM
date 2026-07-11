from __future__ import annotations

import csv
from pathlib import Path

from trkh.tools.audit_hflip_decision_instability import audit_instability


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_hflip_instability_audit_groups_softboost_class1_errors(tmp_path: Path) -> None:
    hflip_csv = tmp_path / "hflip.csv"
    softboost_csv = tmp_path / "softboost.csv"
    output_dir = tmp_path / "audit"
    _write_csv(
        hflip_csv,
        [
            {
                "index": 0,
                "target": 1,
                "clean_pred": 0,
                "flip_pred": 1,
                "clean_conf": 0.40,
                "flip_conf": 0.45,
                "clean_entropy": 1.2,
                "flip_entropy": 1.1,
            },
            {
                "index": 1,
                "target": 0,
                "clean_pred": 1,
                "flip_pred": 0,
                "clean_conf": 0.38,
                "flip_conf": 0.41,
                "clean_entropy": 1.3,
                "flip_entropy": 1.2,
            },
            {
                "index": 2,
                "target": 2,
                "clean_pred": 2,
                "flip_pred": 2,
                "clean_conf": 0.60,
                "flip_conf": 0.58,
                "clean_entropy": 0.9,
                "flip_entropy": 0.95,
            },
        ],
    )
    _write_csv(
        softboost_csv,
        [
            {
                "sample_index": 0,
                "image_path": "a.jpg",
                "target_index": 1,
                "prediction_index": 0,
                "confidence": 0.40,
                "prob_0_class0": 0.40,
                "prob_1_class1": 0.30,
                "prob_2_class2": 0.30,
            },
            {
                "sample_index": 1,
                "image_path": "b.jpg",
                "target_index": 0,
                "prediction_index": 1,
                "confidence": 0.38,
                "prob_0_class0": 0.35,
                "prob_1_class1": 0.38,
                "prob_2_class2": 0.27,
            },
            {
                "sample_index": 2,
                "image_path": "c.jpg",
                "target_index": 2,
                "prediction_index": 2,
                "confidence": 0.60,
                "prob_0_class0": 0.20,
                "prob_1_class1": 0.20,
                "prob_2_class2": 0.60,
            },
        ],
    )

    summary = audit_instability(
        hflip_predictions=hflip_csv,
        softboost_predictions=softboost_csv,
        output_dir=output_dir,
        top_k_cases=10,
    )

    assert summary["support"] == 3
    assert summary["overall"]["hflip_changed"] == 2
    assert summary["groups"]["softboost_class1_fn"]["count"] == 1
    assert summary["groups"]["softboost_class1_fn"]["flip_corrects_softboost_errors"] == 1
    assert summary["groups"]["softboost_class1_fp"]["count"] == 1
    assert summary["groups"]["softboost_class1_fp"]["clean_class1_to_flip_non1"] == 1
    assert summary["decision"]["training_manifest_written"] is False
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "hflip_softboost_instability_cases.csv").exists()


def test_hflip_instability_audit_reports_target_mismatch(tmp_path: Path) -> None:
    hflip_csv = tmp_path / "hflip.csv"
    softboost_csv = tmp_path / "softboost.csv"
    output_dir = tmp_path / "audit"
    _write_csv(
        hflip_csv,
        [
            {
                "index": 0,
                "target": 2,
                "clean_pred": 2,
                "flip_pred": 2,
                "clean_conf": 0.5,
                "flip_conf": 0.5,
                "clean_entropy": 1.0,
                "flip_entropy": 1.0,
            }
        ],
    )
    _write_csv(
        softboost_csv,
        [
            {
                "sample_index": 0,
                "target_index": 1,
                "prediction_index": 1,
                "confidence": 0.5,
                "prob_0_class0": 0.2,
                "prob_1_class1": 0.5,
                "prob_2_class2": 0.3,
            }
        ],
    )

    summary = audit_instability(
        hflip_predictions=hflip_csv,
        softboost_predictions=softboost_csv,
        output_dir=output_dir,
    )

    assert summary["target_mismatch_count"] == 1
    assert summary["target_mismatch_preview"] == [0]

import csv
import json
from pathlib import Path

import pytest

from trkh.tools.reconcile_fp32_xai_cohort import (
    classify_fp32_change,
    reconcile_cohort,
)


def _write_case(root: Path, sample_index: int, target: int, prediction: int) -> None:
    case_dir = root / f"case_{sample_index:03d}"
    case_dir.mkdir(parents=True)
    image_path = root.parent / "images" / f"{sample_index}.jpg"
    payload = {
        "sample_index": sample_index,
        "image_path": str(image_path),
        "target_index": target,
        "prediction_index": prediction,
        "confidence": 0.25,
        "margin": 0.001,
        "correct": int(target == prediction),
    }
    (case_dir / "case.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_cohort(path: Path) -> None:
    rows = [
        {
            "sample_index": "1",
            "image_path": str(path.parent / "images" / "1.jpg"),
            "target_index": "2",
            "control_prediction_index": "2",
            "candidate_prediction_index": "3",
            "xai_category": "harm",
        },
        {
            "sample_index": "2",
            "image_path": str(path.parent / "images" / "2.jpg"),
            "target_index": "0",
            "control_prediction_index": "0",
            "candidate_prediction_index": "1",
            "xai_category": "focus_fp_added",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_classify_fp32_change_covers_precision_and_unchanged_cases() -> None:
    assert classify_fp32_change(2, 1, 2) == "focus_fp_removed"
    assert classify_fp32_change(0, 0, 1) == "focus_fp_added"
    assert classify_fp32_change(1, 2, 1) == "focus_fn_rescue"
    assert classify_fp32_change(1, 1, 0) == "focus_tp_break"
    assert classify_fp32_change(3, 3, 2) == "new_3_to_2_harm"
    assert classify_fp32_change(2, 3, 3) == "fp32_unchanged"


def test_reconcile_preserves_selection_and_records_backend_drift(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.csv"
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _write_cohort(cohort)
    _write_case(control, 1, 2, 3)
    _write_case(candidate, 1, 2, 3)
    _write_case(control, 2, 0, 0)
    _write_case(candidate, 2, 0, 1)

    summary = reconcile_cohort(
        cohort_csv=cohort,
        control_xai_dir=control,
        candidate_xai_dir=candidate,
        output_dir=tmp_path / "out",
        expected_cases=2,
    )

    assert summary["cases"] == 2
    assert summary["control"]["prediction_drift_count"] == 1
    assert summary["candidate"]["prediction_drift_count"] == 0
    assert summary["any_prediction_drift_count"] == 1
    assert summary["category_change_count"] == 1
    with Path(summary["reconciled_cohort"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_index"] for row in rows] == ["1", "2"]
    assert rows[0]["original_control_prediction_index"] == "2"
    assert rows[0]["control_prediction_index"] == "3"
    assert rows[0]["original_xai_category"] == "harm"
    assert rows[0]["xai_category"] == "fp32_unchanged"


def test_reconcile_rejects_case_path_mismatch(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.csv"
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _write_cohort(cohort)
    _write_case(control, 1, 2, 2)
    _write_case(candidate, 1, 2, 3)
    _write_case(control, 2, 0, 0)
    _write_case(candidate, 2, 0, 1)
    case_path = next(control.glob("case_001/case.json"))
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["image_path"] = str(tmp_path / "wrong.jpg")
    case_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Control image path mismatch"):
        reconcile_cohort(
            cohort_csv=cohort,
            control_xai_dir=control,
            candidate_xai_dir=candidate,
            output_dir=tmp_path / "out",
        )


def test_post_smoke_wrapper_reconciles_fp32_and_resumes_valid_json() -> None:
    script = Path("scripts/run_trkh_learnable_gabor_post_smoke_audits.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "[switch]$ResumeCompleted" in script
    assert "function Test-CompletedJson" in script
    assert '"-m", "trkh.tools.reconcile_fp32_xai_cohort"' in script
    assert '"--cohort", $Fp32CohortCsv' in script

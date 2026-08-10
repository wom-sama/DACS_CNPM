from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trkh.tools.audit_saspa_synthetic_a0_f1_finalize import (
    evaluate_stage_checks,
    validate_fixed_artifacts,
)


def _evidence(*, fidelity: bool, blind: bool):
    return evaluate_stage_checks(
        generation={"passed_generation": True, "saved_output_count": 10},
        fidelity={"passed": fidelity, "row_count": 10},
        blind={"passed": blind},
        duplicate={"passed": True, "reference_file_count": 24996},
        duplicate_replay={"passed": True},
        xai={
            "status": "diagnostic_complete_upstream_rejected",
            "case_count": 10,
        },
        xai_replay={"passed": True},
        visual_review={
            "decision_authority": "diagnostic_only",
            "a1_authorized": False,
            "training_authorized": False,
        },
    )


def test_final_decision_rejects_failed_fidelity_and_blind_review() -> None:
    result = _evidence(fidelity=False, blind=False)
    assert result["decision"] == "Rejected"
    assert result["failed_checks"] == [
        "fidelity_passed",
        "blind_review_passed",
    ]


def test_final_decision_requires_every_stage() -> None:
    result = _evidence(fidelity=True, blind=True)
    assert result["decision"] == "Passed"
    assert result["passed"] is True
    assert result["failed_checks"] == []


def test_fixed_artifacts_are_hash_checked(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text(json.dumps({"passed": False}) + "\n", encoding="utf-8")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lock = {"fixed_artifacts": {"evidence.json": expected}}
    result = validate_fixed_artifacts(lock=lock, generation_root=tmp_path)
    assert result["passed"] is True
    assert result["artifact_count"] == 1
    artifact.write_text("{}\n", encoding="utf-8")
    changed = validate_fixed_artifacts(lock=lock, generation_root=tmp_path)
    assert changed["passed"] is False
    assert changed["failures"] == ["hash:evidence.json"]

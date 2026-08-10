import json
from pathlib import Path

import pytest

from trkh.tools.apply_frozen_precision_ensemble import (
    _load_frozen_protocol,
    _verify_frozen_summary_hash,
)
from trkh.tools.audit_precision_ensemble_readiness import _sha256


def _write_protocol(tmp_path: Path, *, passed: bool = True) -> Path:
    keeper = tmp_path / "keeper_val.csv"
    candidate = tmp_path / "candidate_val.csv"
    keeper.write_text("sample_index\n0\n", encoding="utf-8")
    candidate.write_text("sample_index\n0\n", encoding="utf-8")
    summary = {
        "mode": "validation_only_precision_ensemble_readiness",
        "all_gates_passed": passed,
        "gates": {"one": passed},
        "inputs": {
            "keeper_csv": str(keeper),
            "keeper_sha256": _sha256(keeper),
            "candidate_csv": str(candidate),
            "candidate_sha256": _sha256(candidate),
            "test_data_used": False,
        },
        "locked": {"candidate_weight": 0.4, "focus_margin_offset": 0.034},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_load_frozen_protocol_accepts_hashed_passing_validation_summary(
    tmp_path: Path,
) -> None:
    path = _write_protocol(tmp_path)

    summary = _load_frozen_protocol(path)

    assert summary["locked"]["candidate_weight"] == 0.4


def test_load_frozen_protocol_rejects_failed_gate(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path, passed=False)

    with pytest.raises(ValueError, match="did not pass"):
        _load_frozen_protocol(path)


def test_load_frozen_protocol_rejects_changed_validation_input(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    Path(summary["inputs"]["keeper_csv"]).write_text(
        "sample_index\n0\n1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash changed"):
        _load_frozen_protocol(path)


def test_frozen_summary_hash_is_required_and_exact_for_test(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path)

    with pytest.raises(ValueError, match="requires --expected-summary-sha256"):
        _verify_frozen_summary_hash(path, None, required=True)
    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        _verify_frozen_summary_hash(path, "abc", required=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_frozen_summary_hash(path, "0" * 64, required=True)

    assert _verify_frozen_summary_hash(path, _sha256(path), required=True) == _sha256(path)

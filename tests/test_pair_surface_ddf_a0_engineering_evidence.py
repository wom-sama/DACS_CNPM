from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "TRKH_5CLASS_PAIR_SURFACE_DDF_A0_ENGINEERING_EVIDENCE_20260725.json"
)
SIDECAR = EVIDENCE.with_suffix(".sha256")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ddf_engineering_evidence_sidecar_and_sources_are_exact() -> None:
    sidecar = SIDECAR.read_text(encoding="ascii").split()
    assert sidecar == [_sha256(EVIDENCE), EVIDENCE.name]
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    expected_sources = {
        "engine": ROOT / "trkh" / "tools" / "pair_surface_ddf_a0_engine.py",
        "engine_test": ROOT / "tests" / "test_pair_surface_ddf_a0_engine.py",
        "preflight": (
            ROOT / "trkh" / "tools" / "preflight_pair_surface_ddf_a0.py"
        ),
        "preflight_test": (
            ROOT / "tests" / "test_preflight_pair_surface_ddf_a0.py"
        ),
    }
    assert evidence["source_sha256"] == {
        name: _sha256(path) for name, path in expected_sources.items()
    }
    lock = (
        ROOT
        / "docs"
        / "TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.json"
    )
    assert evidence["lock_sha256"] == _sha256(lock)


def test_ddf_engineering_evidence_is_synthetic_and_passes_locked_limits() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    final = evidence["final_attempt"]
    limits = evidence["limits"]

    assert evidence["state"] == "synthetic_engineering_passed_mechanism_unproven"
    assert not any(evidence["candidate_data"].values())
    assert evidence["first_attempt"]["passed"] is False
    assert evidence["recovery_attempt"]["passed"] is False
    assert final["passed"] is True
    assert final["batch1"]["maximum_eager_graph_error"] == 0.0
    assert final["batch32"]["maximum_eager_graph_error"] == 0.0
    assert final["batch1"]["mean_ms"] <= limits["batch1_mean_ms"]
    assert final["batch1"]["p95_ms"] <= limits["batch1_p95_ms"]
    assert final["batch32"]["mean_ms"] <= limits["batch32_mean_ms"]
    assert final["batch32"]["p95_ms"] <= limits["batch32_p95_ms"]
    assert (
        final["equation_maximum_absolute_error"]
        <= limits["equation_maximum_absolute_error"]
    )
    assert (
        final["onnx"]["runtime_maximum_absolute_error"]
        <= limits["onnxruntime_maximum_absolute_error"]
    )
    assert final["tensorrt"]["parse_passed"] is True
    assert final["tensorrt"]["build_passed"] is True

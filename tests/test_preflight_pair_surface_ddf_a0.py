from __future__ import annotations

from pathlib import Path

from trkh.tools import preflight_pair_surface_ddf_a0 as preflight


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_reads_only_prospective_sources_not_candidate_data() -> None:
    source = (
        ROOT
        / "trkh"
        / "tools"
        / "preflight_pair_surface_ddf_a0.py"
    ).read_text(encoding="utf-8")
    lowered = source.casefold()

    assert "newdataset" not in lowered
    assert "model_srgb_uint8" not in lowered
    assert "cohort_arrays" not in lowered
    assert "predictions_all_conditions" not in lowered
    assert "\\validation\\" not in lowered
    assert "/validation/" not in lowered
    assert "test_read" in lowered


def test_preflight_lock_equation_and_role_contracts_pass() -> None:
    lock, digest = preflight._read_lock()
    equation = preflight._equation_oracle()
    roles = preflight._role_contracts()

    assert len(digest) == 64
    assert lock["access"]["formal_runs_authorized"] == 0
    assert lock["access"]["replay_runs_authorized"] == 0
    assert equation["maximum_absolute_error"] <= 1e-10
    assert equation["passed"] is True
    assert roles["matching_primary_tensors"] > 0
    assert roles["matching_primary_tensors_bit_exact"] is True
    assert roles["passed"] is True

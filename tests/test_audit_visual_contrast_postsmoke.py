from __future__ import annotations

import json
from pathlib import Path

from trkh.tools.audit_visual_contrast_postsmoke import (
    EXPECTED_CONDITIONS,
    run_audit,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_postsmoke_closes_failed_metric_gate_and_aggregates_robustness(
    tmp_path: Path,
) -> None:
    pair = _write(
        tmp_path / "pair.json",
        {
            "sources": {"test_used": False},
            "gate": {
                "metric_gate_passed": False,
                "five_epoch_permission": False,
                "full_train_permission": False,
                "failed_checks": ["class1_precision_gain"],
            },
        },
    )
    trace = _write(
        tmp_path / "trace.json",
        {"checks": {"test_used": False, "all_values_finite": True}, "diagnostics": {}},
    )
    xai = _write(
        tmp_path / "xai.json",
        {
            "test_data_used": False,
            "cases": 16,
            "categories": {
                "all": {
                    "candidate_minus_control": {
                        "attention_foreground_mass": -0.1,
                        "grad_rollout_foreground_mass": -0.2,
                    }
                }
            },
            "grad_rollout_provenance": {"fallback_case_count": {"candidate": 16}},
        },
    )
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    for index, condition in enumerate(EXPECTED_CONDITIONS):
        _write(
            control / condition / "summary.json",
            {"metrics": {"accuracy": 0.8, "macro_f1": 0.7}},
        )
        _write(
            candidate / condition / "summary.json",
            {
                "metrics": {
                    "accuracy": 0.81,
                    "macro_f1": 0.71 if index == 0 else 0.69,
                }
            },
        )

    control_forensics = _write(
        tmp_path / "forensics_control.json",
        {
            "samples": 2606,
            "pair_metrics": [{"pair": "0-1", "f1": 0.8}],
            "calibration": {"ece": 0.5},
        },
    )
    candidate_forensics = _write(
        tmp_path / "forensics_candidate.json",
        {
            "samples": 2606,
            "pair_metrics": [{"pair": "0-1", "f1": 0.79}],
            "calibration": {"ece": 0.49},
        },
    )
    control_confusions = _write(
        tmp_path / "confusions_control.json",
        {
            "samples": 2606,
            "focus_metrics": {
                "tp": 90,
                "fp": 140,
                "fn": 61,
                "precision": 0.39,
                "recall": 0.60,
                "f1": 0.47,
            },
            "error_counts": {"4->1": 68},
        },
    )
    candidate_confusions = _write(
        tmp_path / "confusions_candidate.json",
        {
            "samples": 2606,
            "focus_metrics": {
                "tp": 80,
                "fp": 160,
                "fn": 71,
                "precision": 0.33,
                "recall": 0.53,
                "f1": 0.41,
            },
            "error_counts": {"4->1": 82},
        },
    )

    summary = run_audit(
        pair_summary=pair,
        trace_audit=trace,
        paired_xai=xai,
        robustness_control=control,
        robustness_candidate=candidate,
        forensics_control=control_forensics,
        forensics_candidate=candidate_forensics,
        confusions_control=control_confusions,
        confusions_candidate=candidate_confusions,
        output_dir=tmp_path / "out",
    )

    assert summary["decision"]["close_method_without_hyperparameter_sweep"]
    assert not summary["decision"]["five_epoch_permission"]
    assert summary["robustness"]["macro_f1_win_count"] == 1
    assert summary["xai"]["candidate_grad_rollout_fallback_cases"] == 16
    assert not summary["xai"]["candidate_grad_rollout_valid_for_attribution"]
    assert "grad_rollout_foreground_mass" not in summary["xai"]["candidate_minus_control"]
    assert summary["xai"]["invalid_candidate_minus_control_fields"] == [
        "grad_rollout_foreground_mass"
    ]
    assert summary["forensics"]["error_count_deltas"]["4->1"] == 14
    assert summary["forensics"]["pair_f1_deltas"]["0-1"] == -0.010000000000000009
    assert summary["test_used"] is False

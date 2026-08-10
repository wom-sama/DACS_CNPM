import csv
import json
from pathlib import Path

from trkh.tools.build_trkh_signal_gap_readiness import build_trkh_signal_gap_readiness


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_signal_gap_includes_rival24_review_rows_without_opening_gate(tmp_path: Path) -> None:
    softboost_val = _write_csv(
        tmp_path / "softboost_val.csv",
        [
            {"sample_index": 1, "target_index": 1, "prediction_index": 2},
            {"sample_index": 2, "target_index": 1, "prediction_index": 4},
            {"sample_index": 3, "target_index": 0, "prediction_index": 1},
            {"sample_index": 4, "target_index": 2, "prediction_index": 2},
        ],
    )
    transition_budget = _write_csv(
        tmp_path / "transition_budget.csv",
        [
            {"transition": "1->2", "target": 1, "prediction": 2, "count": 11, "class1_delta": "0.04", "class1_f1": "0.74"},
            {"transition": "1->4", "target": 1, "prediction": 4, "count": 5, "class1_delta": "0.02", "class1_f1": "0.72"},
            {"transition": "0->1", "target": 0, "prediction": 1, "count": 41, "class1_delta": "0.10", "class1_f1": "0.80"},
        ],
    )
    milestone_budget = _write_csv(
        tmp_path / "milestones.csv",
        [{"target_class1_f1": 0.75, "min_total_corrections": 13, "fp1_corrections": 0, "fn1_corrections": 13}],
    )
    external_summary = _write_json(
        tmp_path / "external_summary.json",
        {
            "diagnostic_consensus_route": {
                "metrics": {
                    "macro_f1": 0.90,
                    "per_class": [{"class_index": 1, "precision": 0.71, "recall": 0.72, "f1": 0.723}],
                },
                "stats": {"changes": 99, "corrections": 67, "harms": 28},
            }
        },
    )
    external_remaining = _write_csv(
        tmp_path / "external_remaining.csv",
        [
            {
                "target_index": 1,
                "base_prediction_index": 2,
                "error_pair": "1->2",
                "external_correct_count": 0,
                "external_consensus_prediction": 2,
                "external_consensus_votes": 2,
                "external_consensus_tied": 0,
            },
            {
                "target_index": 1,
                "base_prediction_index": 4,
                "error_pair": "1->4",
                "external_correct_count": 1,
                "external_consensus_prediction": 1,
                "external_consensus_votes": 2,
                "external_consensus_tied": 0,
            },
        ],
    )
    train_oof = _write_csv(
        tmp_path / "train_oof.csv",
        [
            {
                "target_index": 1,
                "base_pred": 4,
                "bucket": "base_fn1_rescue_candidate",
                "min_conf": 0.75,
                "external_correct_count": 1,
                "external_same_1": 1,
                "external_same_non1": 0,
            }
        ],
    )
    train_summary = _write_json(tmp_path / "train_summary.json", {"mode": "train_oof_external_reliability_full"})
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "ready_count": 0,
            "blocked_count": 1,
            "items": [{"name": "rival24", "manual_label_status_filled": 0, "manual_label_status_blank": 2}],
        },
    )
    oof_policy = _write_json(tmp_path / "oof_policy.json", {"candidate_rows": 188, "strict_rows": 0, "issue_pairs": {"0->1": 70}})
    rival_review = _write_csv(
        tmp_path / "rival_review.csv",
        [
            {"sample_index": 10, "target_index": 1, "prediction_index": 1, "risk_transition": "1->2", "manual_label_status": ""},
            {"sample_index": 11, "target_index": 1, "prediction_index": 1, "risk_transition": "1->4", "manual_label_status": ""},
            {"sample_index": 12, "target_index": 2, "prediction_index": 1, "risk_transition": "2->1", "manual_label_status": ""},
        ],
    )

    summary = build_trkh_signal_gap_readiness(
        softboost_val=softboost_val,
        transition_budget=transition_budget,
        milestone_budget=milestone_budget,
        external_support_summary=external_summary,
        external_remaining_support=external_remaining,
        train_oof_external_support=train_oof,
        train_oof_external_summary=train_summary,
        review_readiness=readiness,
        oof_neighbor_policy=oof_policy,
        review_csvs={"rival24": rival_review},
        output_dir=tmp_path / "signal_gap",
        created_at="2026-07-06T23:00:00",
    )

    fn_pairs = summary["transition_focus_gaps"]["class1_false_negative_pairs"]
    assert fn_pairs["1->2"]["review_rows_total"] == 1
    assert fn_pairs["1->2"]["review_rows_rival24"] == 1
    assert fn_pairs["1->2"]["review_manual_filled_total"] == 0
    assert fn_pairs["1->4"]["review_rows_total"] == 1
    assert summary["unreviewed_critical_true_class1_transitions"] == ["1->2", "1->4"]
    assert summary["smoke_gate_ready"] is False
    assert "critical_true_class1_review_rows_unfilled" in summary["blocking_reasons"]
    assert summary["raw_dataset_touched"] is False
    assert summary["test_split_used"] is False
    assert summary["trainable_manifest_written"] is False
    assert (tmp_path / "signal_gap" / "summary.json").is_file()
    assert (tmp_path / "signal_gap" / "transition_gap_rows.csv").is_file()

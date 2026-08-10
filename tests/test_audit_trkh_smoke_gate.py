import json
from pathlib import Path

from trkh.tools.audit_trkh_smoke_gate import audit_trkh_smoke_gate


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _base_inputs(tmp_path: Path) -> dict[str, Path]:
    correction = _write_json(
        tmp_path / "correction.json",
        {
            "base_class1": {"f1": 0.703030303, "precision": 0.648, "recall": 0.768, "tp": 116, "fp": 63, "fn": 35},
            "budgets": [
                {"target_class1_f1": 0.75, "min_total_corrections": 13, "fp1_corrections": 0, "fn1_corrections": 13},
                {"target_class1_f1": 0.8, "min_total_corrections": 27, "fp1_corrections": 0, "fn1_corrections": 27},
            ],
        },
    )
    external = _write_json(
        tmp_path / "external.json",
        {
            "diagnostic_consensus_route": {
                "metrics": {
                    "macro_f1": 0.9034,
                    "per_class": [
                        {"class_index": 0, "f1": 0.92},
                        {"class_index": 1, "precision": 0.719, "recall": 0.728, "f1": 0.7237},
                    ],
                },
                "stats": {"changes": 99, "corrections": 67, "harms": 28},
            }
        },
    )
    signal = _write_json(
        tmp_path / "signal_gap.json",
        {
            "smoke_gate_ready": False,
            "trainable_manifest_written": False,
            "test_split_used": False,
            "blocking_reasons": ["val_external_consensus_covers_zero_1_to_2_errors"],
            "transition_focus_gaps": {
                "class1_false_negative_pairs": {
                    "1->2": {
                        "val_error_count": 11,
                        "val_consensus_correct": 0,
                        "train_oof_rows": 0,
                        "review_rows_total": 0,
                        "review_manual_filled_total": 0,
                    },
                    "1->4": {
                        "val_error_count": 5,
                        "val_consensus_correct": 1,
                        "train_oof_rows": 1,
                        "review_rows_total": 2,
                        "review_manual_filled_total": 0,
                    },
                }
            },
        },
    )
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "ready_count": 0,
            "blocked_count": 1,
            "trainable_manifest_written": False,
            "test_split_used": False,
            "items": [
                {
                    "name": "manual_softboost_train_class1_rival24_gap_review",
                    "input_rows": 155,
                    "ready_for_training_manifest": False,
                    "manual_label_status_filled": 0,
                    "manual_label_status_blank": 155,
                    "actionable_recall_count": 0,
                    "actionable_fp_count": 0,
                    "blocking_mixed_side_cluster_count": 1,
                    "blocking_reasons": ["manual_field_not_filled:155", "mixed_actionable_side_clusters:1"],
                }
            ],
        },
    )
    xai = _write_json(
        tmp_path / "xai.json",
        {
            "selected_cases": 8,
            "transition_count": 2,
            "test_split_used": False,
            "transition_rows": [
                {
                    "transition": "1->2",
                    "cases": 4,
                    "top_flags": "gradcam_border_attention:4",
                    "background_blur_original_prediction_drop_mean": 0.001,
                    "background_gray_original_prediction_drop_mean": 0.002,
                    "object_desaturate_original_prediction_drop_mean": 0.11,
                    "attention_foreground_mass_mean": 0.95,
                    "gradcam_foreground_mass_mean": 0.97,
                },
                {
                    "transition": "2->1",
                    "cases": 4,
                    "top_flags": "object_color_sensitive:3;rollout_border_attention:2",
                    "background_blur_original_prediction_drop_mean": 0.001,
                    "background_gray_original_prediction_drop_mean": -0.001,
                    "object_desaturate_original_prediction_drop_mean": 0.18,
                    "attention_foreground_mass_mean": 0.97,
                    "gradcam_foreground_mass_mean": 0.93,
                },
            ],
        },
    )
    return {
        "correction_budget_summary": correction,
        "external_support_summary": external,
        "signal_gap_summary": signal,
        "readiness_matrix_summary": readiness,
        "xai_transition_summary": xai,
    }


def test_audit_trkh_smoke_gate_blocks_current_artifacts(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)

    summary = audit_trkh_smoke_gate(
        **inputs,
        output_dir=tmp_path / "gate",
        created_at="2026-07-06T22:00:00",
    )

    assert summary["smoke_gate_ready"] is False
    assert summary["raw_dataset_touched"] is False
    assert summary["trainable_manifest_written"] is False
    assert summary["correction_budget_for_next_milestone"]["min_total_corrections"] == 13
    assert summary["review_readiness"]["manual_label_status_filled_total"] == 0
    assert summary["transition_coverage"]["insufficient_transitions"] == ["1->2", "1->4"]
    assert summary["xai_assessment"]["surface_boundary_dominant"] is True
    blockers = set(summary["blocking_reasons"])
    assert "signal_gap_smoke_gate_false" in blockers
    assert "review_readiness_ready_count_zero" in blockers
    assert "all_review_manual_fields_empty" in blockers
    assert "rival24_mixed_actionable_cluster_present" in blockers
    assert "external_consensus_class1_below_milestone" in blockers
    assert "external_consensus_true_class1_recall_coverage_insufficient" in blockers
    assert "xai_supports_surface_boundary_not_background_context" in blockers
    assert (tmp_path / "gate" / "summary.json").is_file()
    assert "Smoke gate ready: `false`" in (tmp_path / "gate" / "README.md").read_text(encoding="utf-8")


def test_audit_trkh_smoke_gate_allows_declared_new_surface_signal(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)

    summary = audit_trkh_smoke_gate(
        **inputs,
        output_dir=tmp_path / "gate_ready",
        created_at="2026-07-06T22:05:00",
        new_surface_boundary_signal_ready=True,
    )

    assert summary["source_signal_ready"] is True
    assert summary["new_surface_boundary_signal_ready"] is True
    assert summary["smoke_gate_ready"] is True
    assert summary["blocking_reasons"] == []
    assert summary["xai_assessment"]["surface_boundary_dominant"] is True

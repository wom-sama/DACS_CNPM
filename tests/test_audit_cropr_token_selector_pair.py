from __future__ import annotations

import subprocess
from pathlib import Path

import torch
from torch import nn

from trkh.tools.audit_cropr_token_selector_pair import (
    EXPECTED_HOLDOUT_ROWS,
    FAILED_AUDIT_DIR,
    FAILED_AUDIT_HEAD,
    LOCKED_FAILED_MANIFEST_SHA256,
    LOCKED_FAILED_SUMMARY_SHA256,
    LOCKED_FAILURE_RECORD_SHA256,
    LOCKED_PAIR_MANIFEST_SHA256,
    LOCKED_PREFLIGHT_SUMMARY_SHA256,
    _compare_failed_artifacts,
    _forward_trace,
    _maximum_tree_error,
    _perturbation_summary,
    _run_provenance,
    _selector_summary_with_expected,
    _set_jaccard,
    _verify_failed_attempt,
    parse_args,
)
from trkh.tools.audit_foveal_aggregated_attention_pair import _sha256


def _selector_rows(count: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in (
        "clean",
        "lighting_dim",
        "lighting_bright",
        "low_contrast",
    ):
        for role in ("control", "candidate"):
            for layer in (2, 5):
                for index in range(count):
                    target = index % 5
                    hard_negative = target == 0
                    margin = 2.0 if target == 1 else (-2.0 if hard_negative else 0.0)
                    row: dict[str, object] = {
                        "condition": condition,
                        "role": role,
                        "layer": layer,
                        "local_index": index,
                        "sample_index": 1000 + index,
                        "target": target,
                        "auxiliary_prediction": target,
                        "auxiliary_class1_margin": margin,
                        "normalized_entropy": 0.75,
                        "score_variance": 0.02,
                        "cropr_bbox_mass": 0.55,
                        "native_bbox_mass": 0.50,
                        "selected_object_patch_recall": 0.90,
                        "standard_trace_maximum_error": 0.0,
                        "standard_trace_prediction_match": True,
                        "partition_valid": True,
                        "cropr_routing": role == "candidate",
                        "hard_negative": hard_negative,
                        "kept_indices": "0;1;2" if layer == 2 else "0;1",
                    }
                    rows.append(row)
    return rows


def test_cropr_pair_locked_artifact_hashes_and_cli_defaults() -> None:
    preflight = Path("runs/audit_cropr_token_selector_preflight_20260716/summary.json")
    pair = Path("runs/cropr_a0_pair_manifest_20260716.json")
    assert _sha256(preflight) == LOCKED_PREFLIGHT_SUMMARY_SHA256
    assert _sha256(pair) == LOCKED_PAIR_MANIFEST_SHA256

    args = parse_args(["--output-dir", "runs/unit_cropr_pair"])
    assert args.batch_size == 32
    assert args.xai_batch_size == 2
    assert args.seed == 42
    assert args.control_checkpoint.name == "last.pt"
    assert args.candidate_checkpoint.name == "last.pt"
    assert EXPECTED_HOLDOUT_ROWS == 1843


def test_cropr_failed_attempt_is_hash_locked_before_xai_correction() -> None:
    failed = _verify_failed_attempt(
        current_head=FAILED_AUDIT_HEAD,
        require_correction_commit=False,
    )
    assert failed["all_checks_pass"], failed["failed_checks"]
    assert failed["summary_sha256"] == LOCKED_FAILED_SUMMARY_SHA256
    assert failed["failure_record_sha256"] == LOCKED_FAILURE_RECORD_SHA256
    assert failed["artifact_manifest_sha256"] == LOCKED_FAILED_MANIFEST_SHA256
    replay = _compare_failed_artifacts(failed, output_dir=FAILED_AUDIT_DIR)
    assert replay["all_checks_pass"], replay["failed_checks"]


def test_cropr_pair_provenance_passes_without_loading_holdout() -> None:
    provenance = _run_provenance(
        control_run=Path("runs/probe_cropr_a0_native_control_5e_20260716"),
        candidate_run=Path("runs/probe_cropr_a0_learned_candidate_5e_20260716"),
        control_checkpoint=Path(
            "runs/probe_cropr_a0_native_control_5e_20260716/checkpoints/last.pt"
        ),
        candidate_checkpoint=Path(
            "runs/probe_cropr_a0_learned_candidate_5e_20260716/checkpoints/last.pt"
        ),
        pair_manifest_path=Path("runs/cropr_a0_pair_manifest_20260716.json"),
        preflight_path=Path(
            "runs/audit_cropr_token_selector_preflight_20260716/summary.json"
        ),
        fold_summary_path=Path(
            "runs/yolof_cidt_fold0_trainonly_20260716/summary.json"
        ),
        protocol_path=Path(
            "docs/TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md"
        ),
        require_clean_worktree=False,
    )
    assert provenance["all_checks_pass"], provenance["failed_checks"]
    assert provenance["tracked_worktree_clean_required"] is False
    assert provenance["occurrence"]["control"]["epochs"] == provenance[
        "occurrence"
    ]["candidate"]["epochs"]


def test_cropr_trace_keeps_the_deployment_pruning_path_enabled() -> None:
    class TraceProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Identity()
            self.forward_kwargs: dict[str, object] = {}

        def forward_features(self, images: torch.Tensor, **kwargs):
            self.forward_kwargs = dict(kwargs)
            return {
                "pooled": torch.zeros((int(images.size(0)), 5)),
                "trace": {"pruning": []},
            }

    model = TraceProbe()
    logits, _ = _forward_trace(
        model,
        torch.zeros((2, 3, 16, 16)),
        {
            "image_mask": torch.ones((2, 16, 16), dtype=torch.bool),
            "bbox": torch.zeros((2, 4)),
        },
    )
    assert tuple(logits.shape) == (2, 5)
    assert model.forward_kwargs["return_trace"] is True
    assert model.forward_kwargs["return_attention"] is False


def test_cropr_selector_summary_and_replay_helpers_are_independent() -> None:
    rows = _selector_rows()
    summary = _selector_summary_with_expected(rows, expected_rows=10)
    candidate = summary["conditions"]["clean"]["candidate"]["layers"]
    assert candidate["2"]["auxiliary"]["macro_f1"] == 1.0
    assert candidate["5"]["auxiliary"]["per_class_f1"][1] == 1.0
    assert candidate["5"]["auxiliary_class1_hard_negative_auroc"] == 1.0
    assert candidate["2"]["foreground_mass_gain_mean"] > 0.049
    assert summary["route_stability"]["lighting_dim"]["candidate"]["layers"][
        "5"
    ]["mean_jaccard"] == 1.0
    assert summary["standard_trace"]["maximum_error"] == 0.0
    assert _maximum_tree_error(summary, summary) == 0.0
    assert _set_jaccard([0, 1, 2], [1, 2, 3]) == 0.5


def test_cropr_perturbation_summary_checks_causality_and_false_positives() -> None:
    rows: list[dict[str, object]] = []
    for role in ("control", "candidate"):
        for mode, shift in (("clean", 0.0), ("object", 2.0), ("far_background", 0.5)):
            for index in range(4):
                rows.append(
                    {
                        "role": role,
                        "mode": mode,
                        "local_index": index,
                        "target": 1 if index == 0 else 0,
                        "prediction": 1 if index == 0 else 0,
                        "class1_restricted_margin": float(index) + shift,
                        "hard_negative": index != 0,
                    }
                )
    summary = _perturbation_summary(rows)
    assert summary["cohort_rows"] == 4
    assert summary["roles"]["candidate"]["object_more_causal"]
    assert summary["roles"]["candidate"]["restricted_fp_no_increase"]


def test_cropr_pair_gate_and_wrapper_keep_locked_scope() -> None:
    source = Path("trkh/tools/audit_cropr_token_selector_pair.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "clean_class1_f1_delta_gte_0p015",
        "clean_class1_precision_delta_gte_0p025",
        "clean_restricted_fp_reduction_gte_4",
        "auxiliary_layer5_class1_f1_gte_0p40",
        "selected_set_shift_jaccard_gte_0p65",
        "auxiliary_margin_auroc_clean_gte_0p65",
        "object_perturbation_more_causal_than_background",
        "xai_all_events_covered",
        "failed_attempt_preserved",
        "unaffected_artifact_replay_exact",
        '"official_validation_permission": False',
        '"test_permission": False',
    ):
        assert marker in source
    run_start = source.index("def run_audit")
    assert source.index("failed_attempt = _verify_failed_attempt", run_start) < source.index(
        "all_rows = common._read_clean_train_rows", run_start
    )
    xai_start = source.index("def _collect_xai_maps")
    xai_end = source.index("def _bbox_overlay", xai_start)
    xai_source = source[xai_start:xai_end]
    standard_forward = xai_source.index("_forward_classification_with_metadata")
    trace_forward = xai_source.index("_forward_trace")
    assert standard_forward < trace_forward
    wrapper = Path(
        "scripts/run_trkh_cropr_token_selector_a0_audit.ps1"
    ).read_text(encoding="utf-8")
    assert "checkpoints\\last.pt" in wrapper
    assert "-RunAudit" in wrapper
    assert "-FinalizePass" in wrapper
    assert "ExpectedSummarySha256" in wrapper
    assert "git status --short --untracked-files=no" in wrapper
    assert "audit_cropr_token_selector_a0_pair_corrected_20260717" in wrapper

    command = (
        "$errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path 'scripts/run_trkh_cropr_token_selector_a0_audit.ps1'),"
        "[ref]$null,[ref]$errors); if($errors.Count){exit 1}"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        text=True,
    )

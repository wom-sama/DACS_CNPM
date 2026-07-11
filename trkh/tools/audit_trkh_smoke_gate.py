from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


DEFAULT_GUARDRAIL = (
    "Smoke-gate audit only. No trainable manifests, relabels, sample weights, "
    "targeted margins, soft targets, raw-data edits, test tuning, or training launches."
)


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _nested(mapping: Mapping[str, object], keys: Sequence[str], default: object = None) -> object:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key, default)
    return current


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_value(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _add_unique(values: List[str], item: str) -> None:
    if item and item not in values:
        values.append(item)


def _per_class(metrics: Mapping[str, object], class_index: int) -> Dict[str, object]:
    per_class = metrics.get("per_class", [])
    if isinstance(per_class, list):
        for row in per_class:
            if isinstance(row, Mapping) and _int_value(row.get("class_index"), -1) == class_index:
                return dict(row)
    return {}


def _budget_for_milestone(
    correction_budget: Mapping[str, object],
    next_class1_milestone: float,
) -> Dict[str, object]:
    selected: Dict[str, object] = {}
    selected_target = float("inf")
    budgets = correction_budget.get("budgets", [])
    if not isinstance(budgets, list):
        return selected
    for row in budgets:
        if not isinstance(row, Mapping):
            continue
        target = _float_value(row.get("target_class1_f1"), -1.0)
        if target >= next_class1_milestone and target < selected_target:
            selected = dict(row)
            selected_target = target
    return selected


def _current_class1(
    correction_budget: Mapping[str, object],
    signal_gap: Mapping[str, object],
) -> Dict[str, object]:
    candidate = correction_budget.get("base_class1")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    candidate = signal_gap.get("current_softboost_class1")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _external_consensus(
    external_support: Mapping[str, object],
    signal_gap: Mapping[str, object],
    focus_class_index: int,
) -> Dict[str, object]:
    route = external_support.get("diagnostic_consensus_route")
    if isinstance(route, Mapping):
        metrics = route.get("metrics", {})
        if isinstance(metrics, Mapping):
            class_row = _per_class(metrics, focus_class_index)
            stats = route.get("stats", {})
            return {
                "macro_f1": _float_value(metrics.get("macro_f1")),
                "class1_f1": _float_value(class_row.get("f1")),
                "class1_precision": _float_value(class_row.get("precision")),
                "class1_recall": _float_value(class_row.get("recall")),
                "stats": dict(stats) if isinstance(stats, Mapping) else {},
            }
    candidate = signal_gap.get("external_consensus_upper_bound")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _review_readiness(readiness_matrix: Mapping[str, object]) -> Dict[str, object]:
    items = readiness_matrix.get("items", [])
    if not isinstance(items, list):
        items = []
    flattened: List[Dict[str, object]] = []
    total_filled = 0
    total_blank = 0
    blocking_mixed = 0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        filled = _int_value(item.get("manual_label_status_filled"))
        blank = _int_value(item.get("manual_label_status_blank"))
        mixed = _int_value(item.get("blocking_mixed_side_cluster_count"))
        total_filled += filled
        total_blank += blank
        blocking_mixed += mixed
        flattened.append(
            {
                "name": str(item.get("name", "")),
                "input_rows": _int_value(item.get("input_rows")),
                "ready_for_training_manifest": _bool_value(item.get("ready_for_training_manifest")),
                "manual_label_status_filled": filled,
                "manual_label_status_blank": blank,
                "actionable_recall_count": _int_value(item.get("actionable_recall_count")),
                "actionable_fp_count": _int_value(item.get("actionable_fp_count")),
                "blocking_mixed_side_cluster_count": mixed,
                "blocking_reasons": list(item.get("blocking_reasons", []))
                if isinstance(item.get("blocking_reasons"), list)
                else [],
            }
        )
    return {
        "ready_count": _int_value(readiness_matrix.get("ready_count")),
        "blocked_count": _int_value(readiness_matrix.get("blocked_count")),
        "queue_count": len(flattened),
        "manual_label_status_filled_total": total_filled,
        "manual_label_status_blank_total": total_blank,
        "blocking_mixed_side_cluster_count": blocking_mixed,
        "items": flattened,
    }


def _transition_coverage(
    signal_gap: Mapping[str, object],
    critical_transitions: Sequence[str],
    min_consensus_correct: int,
) -> Dict[str, object]:
    fn_pairs = _nested(signal_gap, ("transition_focus_gaps", "class1_false_negative_pairs"), {})
    rows: List[Dict[str, object]] = []
    insufficient: List[str] = []
    if not isinstance(fn_pairs, Mapping):
        fn_pairs = {}
    for transition in critical_transitions:
        row = fn_pairs.get(transition, {})
        if not isinstance(row, Mapping):
            row = {}
        val_errors = _int_value(row.get("val_error_count"))
        consensus_correct = _int_value(row.get("val_consensus_correct"))
        train_oof_rows = _int_value(row.get("train_oof_rows"))
        review_rows = _int_value(row.get("review_rows_total"))
        review_manual = _int_value(row.get("review_manual_filled_total"))
        compact = {
            "transition": transition,
            "val_error_count": val_errors,
            "val_consensus_correct": consensus_correct,
            "train_oof_rows": train_oof_rows,
            "review_rows_total": review_rows,
            "review_manual_filled_total": review_manual,
        }
        rows.append(compact)
        if val_errors > 0 and consensus_correct < min_consensus_correct:
            insufficient.append(transition)
    return {
        "critical_transitions": rows,
        "min_consensus_correct": int(min_consensus_correct),
        "insufficient_transitions": insufficient,
        "true_class1_recall_coverage_sufficient": not insufficient,
    }


def _weighted_mean(rows: Sequence[Mapping[str, object]], keys: Sequence[str], *, absolute: bool = False) -> float:
    total = 0.0
    weight_total = 0
    for row in rows:
        weight = max(1, _int_value(row.get("cases"), 1))
        values = []
        for key in keys:
            if key in row:
                value = _float_value(row.get(key))
                values.append(abs(value) if absolute else value)
        if not values:
            continue
        total += (sum(values) / len(values)) * weight
        weight_total += weight
    return float(total / weight_total) if weight_total else 0.0


def _xai_assessment(xai_summary: Mapping[str, object]) -> Dict[str, object]:
    transition_rows = xai_summary.get("transition_rows", [])
    if not isinstance(transition_rows, list):
        transition_rows = []
    rows = [dict(row) for row in transition_rows if isinstance(row, Mapping)]
    background_drop = _weighted_mean(
        rows,
        (
            "background_blur_original_prediction_drop_mean",
            "background_gray_original_prediction_drop_mean",
        ),
        absolute=True,
    )
    object_drop = _weighted_mean(
        rows,
        ("object_desaturate_original_prediction_drop_mean",),
        absolute=True,
    )
    foreground_mass = _weighted_mean(
        rows,
        (
            "attention_foreground_mass_mean",
            "rollout_foreground_mass_mean",
            "grad_rollout_foreground_mass_mean",
            "gradcam_foreground_mass_mean",
        ),
    )
    border_transition_count = 0
    background_transition_count = 0
    for row in rows:
        flags = str(row.get("top_flags", ""))
        if "border_attention" in flags:
            border_transition_count += 1
        if "background_attention" in flags:
            background_transition_count += 1

    surface_boundary_dominant = object_drop >= max(0.02, background_drop * 3.0) and foreground_mass >= 0.75
    background_context_supported = background_drop >= 0.03 and background_transition_count >= max(1, len(rows) // 2)
    if surface_boundary_dominant:
        conclusion = "surface_boundary_dominant"
    elif background_context_supported:
        conclusion = "background_context_supported"
    else:
        conclusion = "inconclusive"

    return {
        "selected_cases": _int_value(xai_summary.get("selected_cases")),
        "transition_count": _int_value(xai_summary.get("transition_count"), len(rows)),
        "background_original_prediction_drop_abs_mean": background_drop,
        "object_desaturate_original_prediction_drop_abs_mean": object_drop,
        "foreground_mass_mean": foreground_mass,
        "border_attention_transition_count": border_transition_count,
        "background_attention_transition_count": background_transition_count,
        "surface_boundary_dominant": surface_boundary_dominant,
        "background_context_supported": background_context_supported,
        "conclusion": conclusion,
    }


def audit_trkh_smoke_gate(
    *,
    correction_budget_summary: Path,
    external_support_summary: Path,
    signal_gap_summary: Path,
    readiness_matrix_summary: Path,
    xai_transition_summary: Path,
    output_dir: Path,
    forensics_summary: Optional[Path] = None,
    created_at: Optional[str] = None,
    focus_class_index: int = 1,
    next_class1_milestone: float = 0.75,
    min_ready_review_queues: int = 1,
    min_true_class1_transition_consensus: int = 2,
    new_surface_boundary_signal_ready: bool = False,
    fold_safe_reliability_signal_ready: bool = False,
    guardrail: str = DEFAULT_GUARDRAIL,
) -> Dict[str, object]:
    correction_budget = _load_json(Path(correction_budget_summary))
    external_support = _load_json(Path(external_support_summary))
    signal_gap = _load_json(Path(signal_gap_summary))
    readiness_matrix = _load_json(Path(readiness_matrix_summary))
    xai_summary = _load_json(Path(xai_transition_summary))
    forensics = _load_json(Path(forensics_summary)) if forensics_summary else {}

    current_class1 = _current_class1(correction_budget, signal_gap)
    next_budget = _budget_for_milestone(correction_budget, next_class1_milestone)
    external_consensus = _external_consensus(external_support, signal_gap, focus_class_index)
    review = _review_readiness(readiness_matrix)
    transition_coverage = _transition_coverage(
        signal_gap,
        critical_transitions=("1->2", "1->4"),
        min_consensus_correct=min_true_class1_transition_consensus,
    )
    xai = _xai_assessment(xai_summary)

    review_signal_ready = _int_value(review.get("ready_count")) >= min_ready_review_queues
    explicit_signal_ready = bool(new_surface_boundary_signal_ready or fold_safe_reliability_signal_ready)
    source_signal_ready = bool(review_signal_ready or explicit_signal_ready)

    blockers: List[str] = []
    warnings: List[str] = []
    source_blockers: List[str] = []
    signal_blockers = signal_gap.get("blocking_reasons", [])
    if isinstance(signal_blockers, list):
        source_blockers = [str(item) for item in signal_blockers]

    any_test_input = any(
        _bool_value(payload.get("test_split_used"), False)
        for payload in (signal_gap, readiness_matrix, xai_summary)
        if isinstance(payload, Mapping)
    )
    if any_test_input:
        _add_unique(blockers, "test_split_input_detected")
    if _bool_value(signal_gap.get("trainable_manifest_written"), False) or _bool_value(
        readiness_matrix.get("trainable_manifest_written"), False
    ):
        _add_unique(blockers, "trainable_manifest_input_detected")

    if not source_signal_ready:
        _add_unique(blockers, "no_fold_safe_manual_or_new_surface_signal_ready")
    if not _bool_value(signal_gap.get("smoke_gate_ready"), False) and not source_signal_ready:
        _add_unique(blockers, "signal_gap_smoke_gate_false")
    if not review_signal_ready and not explicit_signal_ready:
        _add_unique(blockers, "review_readiness_ready_count_zero")
    if (
        _int_value(review.get("manual_label_status_filled_total")) == 0
        and _int_value(review.get("manual_label_status_blank_total")) > 0
        and not explicit_signal_ready
    ):
        _add_unique(blockers, "all_review_manual_fields_empty")
    if _int_value(review.get("blocking_mixed_side_cluster_count")) > 0 and not explicit_signal_ready:
        _add_unique(blockers, "review_blocking_mixed_actionable_clusters_present")
        for item in review.get("items", []):
            if isinstance(item, Mapping) and _int_value(item.get("blocking_mixed_side_cluster_count")) > 0:
                if "rival24" in str(item.get("name", "")).lower():
                    _add_unique(blockers, "rival24_mixed_actionable_cluster_present")

    external_class1_f1 = _float_value(external_consensus.get("class1_f1"))
    if external_class1_f1 < next_class1_milestone and not source_signal_ready:
        _add_unique(blockers, "external_consensus_class1_below_milestone")
    if (
        not _bool_value(transition_coverage.get("true_class1_recall_coverage_sufficient"), False)
        and not source_signal_ready
    ):
        _add_unique(blockers, "external_consensus_true_class1_recall_coverage_insufficient")
    if _bool_value(xai.get("surface_boundary_dominant"), False) and not source_signal_ready:
        _add_unique(blockers, "xai_supports_surface_boundary_not_background_context")

    if next_budget:
        required = _int_value(next_budget.get("min_total_corrections"))
        if required >= 10:
            _add_unique(warnings, "class1_budget_requires_large_fn_fp_corrections")
    if not _bool_value(xai.get("background_context_supported"), False):
        _add_unique(warnings, "xai_does_not_support_another_background_context_route")

    next_allowed_actions: List[str] = [
        "Use external models only as diagnostics, teacher/upper-bound references, or review-priority signals until a fold-safe/manual target is ready.",
        "Keep validation XAI and forensics as audit evidence, not as threshold selectors or training rows.",
        "Run py_compile and focused pytest before any future smoke.",
    ]
    if not review_signal_ready:
        next_allowed_actions.append(
            "Fill explicit manual review fields and rerun the readiness matrix before using any current review queue."
        )
    if _int_value(review.get("blocking_mixed_side_cluster_count")) > 0:
        next_allowed_actions.append("Resolve mixed actionable-side source clusters before manifest generation.")
    if not explicit_signal_ready:
        next_allowed_actions.append(
            "Open the next TRKH-native smoke only after a genuinely new surface/boundary representation target or fold-safe reliability target exists."
        )

    blocked_actions = [
        "Do not launch a full 30-epoch train from the current gate state.",
        "Do not smoke from current external consensus, current review queues, patch-threshold variants, or validation XAI case indices.",
        "Do not convert diagnostic validation transitions into routers, sample weights, soft targets, or targeted margins.",
    ]

    smoke_gate_ready = not blockers
    decision = (
        "Smoke gate is ready for a short TRKH-native probe/smoke with the declared new signal."
        if smoke_gate_ready
        else "Smoke gate remains closed for automatic TRKH training from the current artifacts."
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs: Dict[str, str] = {
        "correction_budget_summary": str(Path(correction_budget_summary).resolve()),
        "external_support_summary": str(Path(external_support_summary).resolve()),
        "signal_gap_summary": str(Path(signal_gap_summary).resolve()),
        "readiness_matrix_summary": str(Path(readiness_matrix_summary).resolve()),
        "xai_transition_summary": str(Path(xai_transition_summary).resolve()),
    }
    if forensics_summary:
        inputs["forensics_summary"] = str(Path(forensics_summary).resolve())

    summary: Dict[str, object] = {
        "mode": "trkh_smoke_gate_audit",
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs,
        "focus_class_index": int(focus_class_index),
        "next_class1_milestone": float(next_class1_milestone),
        "min_ready_review_queues": int(min_ready_review_queues),
        "min_true_class1_transition_consensus": int(min_true_class1_transition_consensus),
        "raw_dataset_touched": False,
        "test_split_used": any_test_input,
        "trainable_manifest_written": False,
        "current_class1": current_class1,
        "correction_budget_for_next_milestone": next_budget,
        "external_consensus": external_consensus,
        "transition_coverage": transition_coverage,
        "review_readiness": review,
        "xai_assessment": xai,
        "forensics_focus_class": forensics.get("focus_class", {}) if isinstance(forensics, Mapping) else {},
        "source_blocking_reasons": source_blockers,
        "source_signal_ready": source_signal_ready,
        "review_signal_ready": review_signal_ready,
        "new_surface_boundary_signal_ready": bool(new_surface_boundary_signal_ready),
        "fold_safe_reliability_signal_ready": bool(fold_safe_reliability_signal_ready),
        "smoke_gate_ready": smoke_gate_ready,
        "blocking_reasons": blockers,
        "warning_reasons": warnings,
        "next_allowed_actions": next_allowed_actions,
        "blocked_actions": blocked_actions,
        "decision": decision,
        "guardrail": guardrail,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    blocker_text = ", ".join(blockers) if blockers else "none"
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# TRKH Smoke Gate Audit",
                "",
                guardrail,
                "",
                f"Smoke gate ready: `{str(smoke_gate_ready).lower()}`",
                f"Next class-1 milestone: `{next_class1_milestone}`",
                f"Blocking reasons: `{blocker_text}`",
                "",
                decision,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether current TRKH artifacts justify another smoke.")
    parser.add_argument("--correction-budget-summary", type=Path, required=True)
    parser.add_argument("--external-support-summary", type=Path, required=True)
    parser.add_argument("--signal-gap-summary", type=Path, required=True)
    parser.add_argument("--readiness-matrix-summary", type=Path, required=True)
    parser.add_argument("--xai-transition-summary", type=Path, required=True)
    parser.add_argument("--forensics-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", type=str, default="")
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--next-class1-milestone", type=float, default=0.75)
    parser.add_argument("--min-ready-review-queues", type=int, default=1)
    parser.add_argument("--min-true-class1-transition-consensus", type=int, default=2)
    parser.add_argument("--new-surface-boundary-signal-ready", action="store_true")
    parser.add_argument("--fold-safe-reliability-signal-ready", action="store_true")
    parser.add_argument("--guardrail", type=str, default=DEFAULT_GUARDRAIL)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = audit_trkh_smoke_gate(
        correction_budget_summary=args.correction_budget_summary,
        external_support_summary=args.external_support_summary,
        signal_gap_summary=args.signal_gap_summary,
        readiness_matrix_summary=args.readiness_matrix_summary,
        xai_transition_summary=args.xai_transition_summary,
        forensics_summary=args.forensics_summary,
        output_dir=args.output_dir,
        created_at=args.created_at or None,
        focus_class_index=args.focus_class_index,
        next_class1_milestone=args.next_class1_milestone,
        min_ready_review_queues=args.min_ready_review_queues,
        min_true_class1_transition_consensus=args.min_true_class1_transition_consensus,
        new_surface_boundary_signal_ready=args.new_surface_boundary_signal_ready,
        fold_safe_reliability_signal_ready=args.fold_safe_reliability_signal_ready,
        guardrail=args.guardrail,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

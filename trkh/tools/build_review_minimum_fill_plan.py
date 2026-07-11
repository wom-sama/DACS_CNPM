from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_GUARDRAIL = (
    "Manual-review fill plan only. No trainable manifests, relabels, sample weights, "
    "targeted margins, soft targets, raw data edits, test usage, auto-labeling, or training launches."
)


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _text(row: Mapping[str, object], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return str(value if value is not None else default).strip()


def _split_values(value: object) -> List[str]:
    values: List[str] = []
    for part in str(value or "").replace(";", ",").split(","):
        item = part.strip()
        if item:
            values.append(item)
    return values


def _int_value(row: Mapping[str, object], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe.strip("_") or "review_item"


def _normalize_transition(value: str) -> str:
    return str(value or "").strip().replace("=>", "->").replace(" - > ", "->")


def _manual_blank(row: Mapping[str, str], manual_field: str) -> bool:
    return not _text(row, manual_field)


def _transition(row: Mapping[str, str], preferred_column: str) -> str:
    for key in (preferred_column, "risk_transition", "transition", "buckets", "boundary_pair"):
        if key and _text(row, key):
            return _normalize_transition(_text(row, key))
    return ""


def _row_identity(row: Mapping[str, str], index: int) -> str:
    for key in ("review_id", "sample_index", "source_row_index", "dataset_index"):
        value = _text(row, key)
        if value:
            return f"{key}:{value}"
    return f"row:{index}"


def _float_sort_value(row: Mapping[str, str], keys: Sequence[str]) -> float:
    for key in keys:
        value = _text(row, key)
        if not value:
            continue
        try:
            return abs(float(value))
        except ValueError:
            continue
    return 999999.0


def _plan_row_float(row: Mapping[str, object], *keys: str) -> float:
    for key in keys:
        value = _text(row, key)
        if not value:
            continue
        try:
            return abs(float(value))
        except ValueError:
            continue
    return 999999.0


def _candidate_sort_key(
    row: Mapping[str, str],
    *,
    transition_column: str,
    critical_transitions: set[str],
    row_index: int,
) -> Tuple[int, float, float, int]:
    transition = _transition(row, transition_column)
    critical_rank = 0 if transition in critical_transitions else 1
    margin_rank = _float_sort_value(
        row,
        (
            "class1_minus_rival_probability",
            "top2_margin",
            "margin",
            "confidence_margin",
        ),
    )
    try:
        review_rank = -float(_text(row, "review_priority_score") or "0")
    except ValueError:
        review_rank = 0.0
    return critical_rank, margin_rank, review_rank, row_index


def _global_plan_sort_key(row: Mapping[str, object], critical_transitions: set[str]) -> Tuple[int, int, float, str, int]:
    role = _text(row, "fill_role")
    transition = _normalize_transition(_text(row, "transition"))
    if role == "mixed_cluster_resolution":
        role_rank = 0
    elif transition in critical_transitions:
        role_rank = 1
    elif role == "recall_required":
        role_rank = 2
    elif role == "fp_required":
        role_rank = 3
    else:
        role_rank = 4
    transition_rank = 0 if transition in critical_transitions else 1
    margin_rank = _plan_row_float(row, "class1_minus_rival_probability", "top2_margin")
    return role_rank, transition_rank, margin_rank, _text(row, "queue_name"), _int_value(row, "row_index", 0)


def _load_config_items(matrix_summary: Mapping[str, object]) -> List[Mapping[str, object]]:
    config_json = Path(_text(matrix_summary, "config_json"))
    if not config_json.is_file():
        raise FileNotFoundError(f"Matrix config not found: {config_json}")
    config = _read_json(config_json)
    items = config.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Matrix config must contain a non-empty items list: {config_json}")
    return [item for item in items if isinstance(item, Mapping)]


def _critical_transitions_from_gate(gate_summary: Optional[Mapping[str, object]]) -> List[str]:
    if not gate_summary:
        return []
    coverage = gate_summary.get("transition_coverage", {})
    if not isinstance(coverage, Mapping):
        return []
    transitions = coverage.get("critical_transitions", [])
    if not isinstance(transitions, list):
        return []
    result: List[str] = []
    for item in transitions:
        if isinstance(item, Mapping):
            transition = _normalize_transition(_text(item, "transition"))
            if transition:
                result.append(transition)
    return result


def _side_role(side: str, recall_values: set[str], fp_values: set[str], neutral_values: set[str]) -> str:
    if side in recall_values:
        return "recall_required"
    if side in fp_values:
        return "fp_required"
    if side in neutral_values:
        return "anchor_context"
    return "unknown_side"


def _selected_row_payload(
    *,
    queue_name: str,
    rank: int,
    fill_role: str,
    row: Mapping[str, str],
    row_index: int,
    side_column: str,
    transition_column: str,
    cluster_column: str,
    selection_reason: str,
) -> Dict[str, object]:
    transition = _transition(row, transition_column)
    return {
        "queue_name": queue_name,
        "priority_rank": rank,
        "fill_role": fill_role,
        "selection_reason": selection_reason,
        "review_id": _text(row, "review_id"),
        "row_index": row_index,
        "sample_index": _text(row, "sample_index") or _text(row, "source_row_index") or _text(row, "dataset_index"),
        "source_stem": _text(row, "source_stem") or Path(_text(row, "image_path")).stem,
        "cluster_id": _text(row, cluster_column) if cluster_column else "",
        "side": _text(row, side_column) if side_column else "",
        "transition": transition,
        "target_index": _text(row, "target_index"),
        "prediction_index": _text(row, "prediction_index") or _text(row, "top1_index"),
        "manual_label_status": _text(row, "manual_label_status"),
        "manual_expected_class": _text(row, "manual_expected_class"),
        "manual_label_status_suggestion": _text(row, "manual_label_status_suggestion"),
        "manual_expected_class_suggestion": _text(row, "manual_expected_class_suggestion"),
        "top2_margin": _text(row, "top2_margin"),
        "class1_minus_rival_probability": _text(row, "class1_minus_rival_probability"),
        "review_priority_score": _text(row, "review_priority_score"),
        "image_path": _text(row, "image_path"),
        "guardrail": "review_plan_only_not_training_label",
    }


def build_review_minimum_fill_plan(
    *,
    matrix_summary_path: Path,
    output_dir: Path,
    gate_summary_path: Optional[Path] = None,
    created_at: Optional[str] = None,
    rows_per_role_cap: int = 0,
    guardrail: str = DEFAULT_GUARDRAIL,
) -> Dict[str, object]:
    matrix_summary = _read_json(Path(matrix_summary_path))
    gate_summary = _read_json(Path(gate_summary_path)) if gate_summary_path else None
    config_items = _load_config_items(matrix_summary)
    matrix_items = {
        _text(item, "name"): item
        for item in matrix_summary.get("items", [])
        if isinstance(item, Mapping)
    }
    critical_transitions = set(_critical_transitions_from_gate(gate_summary))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: List[Dict[str, object]] = []
    queue_summaries: List[Dict[str, object]] = []
    unique_samples: set[str] = set()
    rank = 1

    for item in config_items:
        name = _text(item, "name")
        source_csv = Path(_text(item, "csv_path", _text(item, "source_csv")))
        side_column = _text(item, "side_column", "strict_review_side")
        transition_column = _text(item, "transition_column", "transition")
        cluster_column = _text(item, "cluster_column", "strict_cluster_id")
        manual_field = _text(item, "manual_field", "manual_label_status")
        recall_values = set(_split_values(item.get("recall_side_value", "recall_protector")))
        fp_values = set(_split_values(item.get("fp_side_value", "fp_suppressor")))
        neutral_values = set(_split_values(item.get("neutral_side_value", "anchor,<blank>")))
        min_recall = _int_value(item, "min_actionable_recall", 1)
        min_fp = _int_value(item, "min_actionable_fp", 1)
        flattened = matrix_items.get(name, {})
        current_recall = _int_value(flattened, "actionable_recall_count", 0)
        current_fp = _int_value(flattened, "actionable_fp_count", 0)
        needed_recall = max(0, min_recall - current_recall)
        needed_fp = max(0, min_fp - current_fp)

        rows, fieldnames = _read_csv(source_csv)
        if side_column and side_column not in fieldnames:
            raise ValueError(f"Side column {side_column!r} missing from {source_csv}")
        if manual_field and manual_field not in fieldnames:
            raise ValueError(f"Manual field {manual_field!r} missing from {source_csv}")

        blank_rows = [(idx, row) for idx, row in enumerate(rows) if _manual_blank(row, manual_field)]
        role_counts_all = Counter()
        transition_counts_all = Counter()
        available_by_role = Counter()
        for idx, row in enumerate(rows):
            side = _text(row, side_column) if side_column else ""
            role = _side_role(side, recall_values, fp_values, neutral_values)
            role_counts_all[role] += 1
            transition_counts_all[_transition(row, transition_column) or "<blank>"] += 1
            if _manual_blank(row, manual_field):
                available_by_role[role] += 1

        item_out_dir = Path(_text(flattened, "output_dir", ""))
        readiness_summary_path = item_out_dir / "readiness_summary.json" if item_out_dir else Path()
        readiness_summary = _read_json(readiness_summary_path) if readiness_summary_path.is_file() else {}
        blocking_clusters = readiness_summary.get("blocking_mixed_side_clusters", [])
        blocking_cluster_ids = {
            _text(cluster, "cluster")
            for cluster in blocking_clusters
            if isinstance(cluster, Mapping) and _text(cluster, "cluster")
        }

        local_selected_keys: set[str] = set()
        for cluster_id in sorted(blocking_cluster_ids):
            for idx, row in blank_rows:
                if cluster_column and _text(row, cluster_column) == cluster_id:
                    key = _row_identity(row, idx)
                    if key in local_selected_keys:
                        continue
                    selected = _selected_row_payload(
                        queue_name=name,
                        rank=rank,
                        fill_role="mixed_cluster_resolution",
                        row=row,
                        row_index=idx,
                        side_column=side_column,
                        transition_column=transition_column,
                        cluster_column=cluster_column,
                        selection_reason=f"blocking mixed actionable cluster {cluster_id}",
                    )
                    selected_rows.append(selected)
                    local_selected_keys.add(key)
                    if selected["sample_index"]:
                        unique_samples.add(str(selected["sample_index"]))
                    rank += 1

        def select_role(role_name: str, need: int, reason: str) -> int:
            nonlocal rank
            if need <= 0:
                return 0
            candidates = [
                (idx, row)
                for idx, row in blank_rows
                if _side_role(_text(row, side_column) if side_column else "", recall_values, fp_values, neutral_values)
                == role_name
                and _row_identity(row, idx) not in local_selected_keys
            ]
            candidates.sort(
                key=lambda pair: _candidate_sort_key(
                    pair[1],
                    transition_column=transition_column,
                    critical_transitions=critical_transitions,
                    row_index=pair[0],
                )
            )
            limit = need if rows_per_role_cap <= 0 else min(need, rows_per_role_cap)
            count = 0
            for idx, row in candidates[:limit]:
                key = _row_identity(row, idx)
                selected = _selected_row_payload(
                    queue_name=name,
                    rank=rank,
                    fill_role=role_name,
                    row=row,
                    row_index=idx,
                    side_column=side_column,
                    transition_column=transition_column,
                    cluster_column=cluster_column,
                    selection_reason=reason,
                )
                selected_rows.append(selected)
                local_selected_keys.add(key)
                if selected["sample_index"]:
                    unique_samples.add(str(selected["sample_index"]))
                rank += 1
                count += 1
            return count

        selected_recall = select_role(
            "recall_required",
            needed_recall,
            f"manual recall coverage needed {current_recall}<{min_recall}",
        )
        selected_fp = select_role(
            "fp_required",
            needed_fp,
            f"manual false-positive coverage needed {current_fp}<{min_fp}",
        )

        critical_available = {
            transition: sum(1 for _, row in blank_rows if _transition(row, transition_column) == transition)
            for transition in sorted(critical_transitions)
        }
        queue_summaries.append(
            {
                "name": name,
                "source_csv": str(source_csv.resolve()),
                "input_rows": int(len(rows)),
                "manual_blank_rows": int(len(blank_rows)),
                "min_actionable_recall": int(min_recall),
                "min_actionable_fp": int(min_fp),
                "current_actionable_recall": int(current_recall),
                "current_actionable_fp": int(current_fp),
                "needed_recall_to_fill": int(needed_recall),
                "needed_fp_to_fill": int(needed_fp),
                "selected_recall_rows": int(selected_recall),
                "selected_fp_rows": int(selected_fp),
                "blocking_mixed_cluster_count": int(len(blocking_cluster_ids)),
                "blocking_mixed_clusters": sorted(blocking_cluster_ids),
                "side_counts_all_rows": dict(role_counts_all),
                "available_blank_by_role": dict(available_by_role),
                "transition_counts_all_rows": dict(transition_counts_all),
                "critical_transition_blank_rows": critical_available,
                "ready_before_fill": bool(flattened.get("ready_for_training_manifest", False)),
                "blocking_reasons_before_fill": list(flattened.get("blocking_reasons", [])),
            }
        )

    fieldnames = [
        "queue_name",
        "priority_rank",
        "fill_role",
        "selection_reason",
        "review_id",
        "row_index",
        "sample_index",
        "source_stem",
        "cluster_id",
        "side",
        "transition",
        "target_index",
        "prediction_index",
        "manual_label_status",
        "manual_expected_class",
        "manual_label_status_suggestion",
        "manual_expected_class_suggestion",
        "top2_margin",
        "class1_minus_rival_probability",
        "review_priority_score",
        "image_path",
        "guardrail",
    ]
    selected_rows.sort(key=lambda row: _global_plan_sort_key(row, critical_transitions))
    for new_rank, row in enumerate(selected_rows, start=1):
        row["priority_rank"] = new_rank
    _write_csv(output_dir / "priority_fill_plan.csv", selected_rows, fieldnames)

    gate_context: Dict[str, object] = {}
    if gate_summary:
        gate_context = {
            "current_class1": gate_summary.get("current_class1", {}),
            "correction_budget_for_next_milestone": gate_summary.get(
                "correction_budget_for_next_milestone", {}
            ),
            "critical_transitions": sorted(critical_transitions),
            "smoke_gate_ready_before_fill_plan": bool(gate_summary.get("smoke_gate_ready", False)),
            "gate_blocking_reasons_before_fill_plan": list(gate_summary.get("blocking_reasons", [])),
        }

    summary_payload: Dict[str, object] = {
        "mode": "review_minimum_fill_plan",
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "matrix_summary": str(Path(matrix_summary_path).resolve()),
        "gate_summary": str(Path(gate_summary_path).resolve()) if gate_summary_path else "",
        "output_dir": str(output_dir.resolve()),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "auto_labeling_performed": False,
        "smoke_permission": False,
        "manual_review_required": True,
        "critical_transitions": sorted(critical_transitions),
        "gate_context": gate_context,
        "queue_count": int(len(queue_summaries)),
        "selected_plan_rows": int(len(selected_rows)),
        "unique_sample_indices_in_plan": int(len(unique_samples)),
        "queue_summaries": queue_summaries,
        "priority_fill_plan_csv": str((output_dir / "priority_fill_plan.csv").resolve()),
        "decision": (
            "This plan identifies manual-review rows needed before reviewed-manifest tooling can be "
            "considered. It does not open the TRKH smoke gate."
        ),
        "guardrail": guardrail,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Review Minimum Fill Plan",
                "",
                "This artifact prioritizes manual review rows needed by the active readiness matrix.",
                guardrail,
                "",
                f"Selected plan rows: `{len(selected_rows)}`",
                "Smoke permission: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimum manual-review fill plan from readiness matrix evidence.")
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-summary", type=Path)
    parser.add_argument("--created-at", type=str, default="")
    parser.add_argument("--rows-per-role-cap", type=int, default=0)
    parser.add_argument("--guardrail", type=str, default=DEFAULT_GUARDRAIL)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_review_minimum_fill_plan(
        matrix_summary_path=args.matrix_summary,
        output_dir=args.output_dir,
        gate_summary_path=args.gate_summary,
        created_at=args.created_at or None,
        rows_per_role_cap=args.rows_per_role_cap,
        guardrail=args.guardrail,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

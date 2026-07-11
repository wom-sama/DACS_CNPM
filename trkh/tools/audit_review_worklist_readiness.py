from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


VALID_MANUAL_STATUSES = {"correct", "ambiguous", "wrong", "needs_crop", "ignore"}
ACTIONABLE_MANUAL_STATUSES = {"correct", "ambiguous", "wrong"}


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _text(row: Mapping[str, str], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def _split_values(value: str | Sequence[str]) -> List[str]:
    chunks = [value] if isinstance(value, str) else list(value)
    values: List[str] = []
    for chunk in chunks:
        for part in str(chunk).replace(";", ",").split(","):
            item = part.strip()
            if item:
                values.append(item)
    return values


def _manual_status(row: Mapping[str, str], manual_field: str) -> str:
    return _text(row, manual_field).lower()


def _has_sample_index(row: Mapping[str, str]) -> bool:
    for key in ("sample_index", "source_row_index", "dataset_index"):
        value = _text(row, key)
        if not value:
            continue
        try:
            return int(float(value)) >= 0
        except ValueError:
            continue
    return False


def _path_value(row: Mapping[str, str]) -> str:
    return _text(row, "image_path") or _text(row, "path") or _text(row, "sample_path")


def _looks_like_non_train(path_text: str) -> bool:
    parts = [part for part in str(path_text or "").replace("\\", "/").lower().split("/") if part]
    return "val" in parts or "valid" in parts or "validation" in parts or "test" in parts


def _count_by(rows: Sequence[Mapping[str, str]], column: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = _text(row, column) if column else ""
        counts[value or "<blank>"] += 1
    return dict(counts)


def _side_counts(
    rows: Sequence[Mapping[str, str]],
    *,
    side_column: str,
    manual_field: str,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        status = _manual_status(row, manual_field)
        if status not in ACTIONABLE_MANUAL_STATUSES:
            continue
        side = _text(row, side_column) if side_column else ""
        counts[side or "<blank>"] += 1
    return dict(counts)


def _cluster_side_conflicts(
    rows: Sequence[Mapping[str, str]],
    *,
    cluster_column: str,
    side_column: str,
    neutral_side_values: set[str],
) -> List[Dict[str, object]]:
    if not cluster_column or not side_column:
        return []
    clusters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cluster = _text(row, cluster_column)
        side = _text(row, side_column)
        if cluster and side:
            clusters[cluster][side] += 1
    conflicts = []
    for cluster, sides in clusters.items():
        if len(sides) > 1:
            non_neutral_sides = {
                side: count for side, count in sides.items() if side not in neutral_side_values
            }
            conflicts.append(
                {
                    "cluster": cluster,
                    "rows": int(sum(sides.values())),
                    "sides": dict(sides),
                    "non_neutral_sides": non_neutral_sides,
                    "blocks_training": len(non_neutral_sides) > 1,
                }
            )
    return sorted(conflicts, key=lambda item: (-int(item["rows"]), str(item["cluster"])))


def audit_review_worklist_readiness(
    *,
    csv_path: Path,
    output_dir: Optional[Path] = None,
    manual_field: str = "manual_label_status",
    side_column: str = "strict_review_side",
    recall_side_value: str = "recall_protector",
    fp_side_value: str = "fp_suppressor",
    transition_column: str = "transition",
    cluster_column: str = "strict_cluster_id",
    neutral_side_value: str = "anchor,<blank>",
    require_all_reviewed: bool = True,
    require_sample_index: bool = True,
    allow_non_train_paths: bool = False,
    min_actionable_recall: int = 1,
    min_actionable_fp: int = 1,
) -> Dict[str, object]:
    rows, fieldnames = _read_csv(Path(csv_path))
    missing_columns = [manual_field]
    if side_column:
        missing_columns.append(side_column)
    missing_columns = [column for column in missing_columns if column and column not in fieldnames]
    if missing_columns:
        raise ValueError(f"Columns not found in {csv_path}: {', '.join(missing_columns)}")
    recall_side_values = set(_split_values(recall_side_value))
    fp_side_values = set(_split_values(fp_side_value))
    neutral_side_values = set(_split_values(neutral_side_value))

    manual_filled_rows = [row for row in rows if _manual_status(row, manual_field)]
    manual_blank_rows = [row for row in rows if not _manual_status(row, manual_field)]
    invalid_status_rows = [
        row for row in manual_filled_rows if _manual_status(row, manual_field) not in VALID_MANUAL_STATUSES
    ]
    actionable_rows = [
        row for row in rows if _manual_status(row, manual_field) in ACTIONABLE_MANUAL_STATUSES
    ]
    side_actionable = _side_counts(rows, side_column=side_column, manual_field=manual_field)
    recall_actionable = sum(side_actionable.get(value, 0) for value in recall_side_values)
    fp_actionable = sum(side_actionable.get(value, 0) for value in fp_side_values)
    missing_sample_index_rows = [row for row in rows if not _has_sample_index(row)]
    missing_path_rows = [row for row in rows if not _path_value(row)]
    non_train_path_rows = [row for row in rows if _path_value(row) and _looks_like_non_train(_path_value(row))]
    conflicts = _cluster_side_conflicts(
        rows,
        cluster_column=cluster_column,
        side_column=side_column,
        neutral_side_values=neutral_side_values,
    )
    blocking_conflicts = [conflict for conflict in conflicts if conflict["blocks_training"]]

    reasons: List[str] = []
    if require_all_reviewed and manual_blank_rows:
        reasons.append(f"manual_field_not_filled:{len(manual_blank_rows)}")
    if invalid_status_rows:
        reasons.append(f"invalid_manual_status:{len(invalid_status_rows)}")
    if require_sample_index and missing_sample_index_rows:
        reasons.append(f"missing_sample_index:{len(missing_sample_index_rows)}")
    if missing_path_rows:
        reasons.append(f"missing_image_path:{len(missing_path_rows)}")
    if not allow_non_train_paths and non_train_path_rows:
        reasons.append(f"non_train_paths:{len(non_train_path_rows)}")
    if recall_actionable < int(min_actionable_recall):
        reasons.append(f"actionable_recall_below_min:{recall_actionable}<{int(min_actionable_recall)}")
    if fp_actionable < int(min_actionable_fp):
        reasons.append(f"actionable_fp_below_min:{fp_actionable}<{int(min_actionable_fp)}")
    if blocking_conflicts:
        reasons.append(f"mixed_actionable_side_clusters:{len(blocking_conflicts)}")

    summary: Dict[str, object] = {
        "source_csv": str(Path(csv_path).resolve()),
        "input_rows": int(len(rows)),
        "manual_field": manual_field,
        "manual_label_status_filled": int(len(manual_filled_rows)),
        "manual_label_status_blank": int(len(manual_blank_rows)),
        "manual_status_counts": _count_by(rows, manual_field),
        "actionable_review_rows": int(len(actionable_rows)),
        "side_actionable_counts": side_actionable,
        "recall_side_value": recall_side_value,
        "fp_side_value": fp_side_value,
        "recall_side_values": sorted(recall_side_values),
        "fp_side_values": sorted(fp_side_values),
        "neutral_side_values": sorted(neutral_side_values),
        "actionable_recall_count": int(recall_actionable),
        "actionable_fp_count": int(fp_actionable),
        "transition_counts": _count_by(rows, transition_column) if transition_column in fieldnames else {},
        "side_counts_all_rows": _count_by(rows, side_column) if side_column in fieldnames else {},
        "missing_sample_index_rows": int(len(missing_sample_index_rows)),
        "missing_image_path_rows": int(len(missing_path_rows)),
        "non_train_path_rows": int(len(non_train_path_rows)),
        "mixed_side_clusters": conflicts,
        "blocking_mixed_side_clusters": blocking_conflicts,
        "ready_for_training_manifest": not reasons,
        "blocking_reasons": reasons,
        "guardrail": (
            "Readiness audit only. This tool writes no sample-weight, soft-target, "
            "targeted-margin, relabel, or raw-data files."
        ),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "readiness_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "README.md").write_text(
            "# Review Worklist Readiness Audit\n\n"
            "This artifact audits whether a train-only manual review worklist is ready "
            "to be passed to reviewed-manifest tooling. It does not create trainable "
            "manifests or edit raw data.\n\n"
            f"Ready: `{summary['ready_for_training_manifest']}`\n",
            encoding="utf-8",
        )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a manual review worklist before manifest building.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manual-field", type=str, default="manual_label_status")
    parser.add_argument("--side-column", type=str, default="strict_review_side")
    parser.add_argument("--recall-side-value", type=str, default="recall_protector")
    parser.add_argument("--fp-side-value", type=str, default="fp_suppressor")
    parser.add_argument("--transition-column", type=str, default="transition")
    parser.add_argument("--cluster-column", type=str, default="strict_cluster_id")
    parser.add_argument("--neutral-side-value", type=str, default="anchor,<blank>")
    parser.add_argument("--allow-partial-review", action="store_true")
    parser.add_argument("--allow-missing-sample-index", action="store_true")
    parser.add_argument("--allow-non-train-paths", action="store_true")
    parser.add_argument("--min-actionable-recall", type=int, default=1)
    parser.add_argument("--min-actionable-fp", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = audit_review_worklist_readiness(
        csv_path=args.csv,
        output_dir=args.output_dir,
        manual_field=args.manual_field,
        side_column=args.side_column,
        recall_side_value=args.recall_side_value,
        fp_side_value=args.fp_side_value,
        transition_column=args.transition_column,
        cluster_column=args.cluster_column,
        neutral_side_value=args.neutral_side_value,
        require_all_reviewed=not args.allow_partial_review,
        require_sample_index=not args.allow_missing_sample_index,
        allow_non_train_paths=args.allow_non_train_paths,
        min_actionable_recall=args.min_actionable_recall,
        min_actionable_fp=args.min_actionable_fp,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

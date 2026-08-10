from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from trkh.tools.audit_review_worklist_readiness import audit_review_worklist_readiness


DEFAULT_GUARDRAIL = (
    "Readiness matrix only. No trainable manifests, relabels, sample weights, "
    "targeted margins, soft targets, raw data edits, or test usage."
)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe.strip("_") or "review_item"


def _bool_value(item: Mapping[str, object], key: str, default: bool) -> bool:
    value = item.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int_value(item: Mapping[str, object], key: str, default: int) -> int:
    try:
        return int(item.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _str_value(item: Mapping[str, object], key: str, default: str = "") -> str:
    value = item.get(key, default)
    return str(value if value is not None else default)


def _flatten_item_summary(name: str, source_csv: Path, output_dir: Path, summary: Mapping[str, object]) -> Dict[str, object]:
    mixed = summary.get("mixed_side_clusters", [])
    blocking_mixed = summary.get("blocking_mixed_side_clusters", [])
    return {
        "name": name,
        "source_csv": str(source_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_rows": int(summary.get("input_rows", 0)),
        "ready_for_training_manifest": bool(summary.get("ready_for_training_manifest", False)),
        "manual_label_status_filled": int(summary.get("manual_label_status_filled", 0)),
        "manual_label_status_blank": int(summary.get("manual_label_status_blank", 0)),
        "actionable_review_rows": int(summary.get("actionable_review_rows", 0)),
        "actionable_recall_count": int(summary.get("actionable_recall_count", 0)),
        "actionable_fp_count": int(summary.get("actionable_fp_count", 0)),
        "missing_sample_index_rows": int(summary.get("missing_sample_index_rows", 0)),
        "missing_image_path_rows": int(summary.get("missing_image_path_rows", 0)),
        "non_train_path_rows": int(summary.get("non_train_path_rows", 0)),
        "mixed_side_cluster_count": len(mixed) if isinstance(mixed, list) else 0,
        "blocking_mixed_side_cluster_count": len(blocking_mixed) if isinstance(blocking_mixed, list) else 0,
        "blocking_reasons": list(summary.get("blocking_reasons", [])),
    }


def _load_items(config_json: Path) -> List[Mapping[str, object]]:
    payload = json.loads(Path(config_json).read_text(encoding="utf-8-sig"))
    if isinstance(payload, Mapping):
        items = payload.get("items")
    else:
        items = payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"Config must contain a non-empty items list: {config_json}")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"Config item {index} must be an object.")
        if not str(item.get("name", "")).strip():
            raise ValueError(f"Config item {index} is missing name.")
        if not str(item.get("csv_path", "") or item.get("source_csv", "")).strip():
            raise ValueError(f"Config item {index} is missing csv_path/source_csv.")
    return items


def build_review_worklist_readiness_matrix(
    *,
    config_json: Path,
    output_dir: Path,
    created_at: Optional[str] = None,
    guardrail: str = DEFAULT_GUARDRAIL,
) -> Dict[str, object]:
    items = _load_items(Path(config_json))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_items: List[Dict[str, object]] = []
    for item in items:
        name = _str_value(item, "name")
        source_csv = Path(_str_value(item, "csv_path", _str_value(item, "source_csv")))
        item_output_dir = output_dir / _safe_name(name)
        summary = audit_review_worklist_readiness(
            csv_path=source_csv,
            output_dir=item_output_dir,
            manual_field=_str_value(item, "manual_field", "manual_label_status"),
            side_column=_str_value(item, "side_column", "strict_review_side"),
            recall_side_value=_str_value(item, "recall_side_value", "recall_protector"),
            fp_side_value=_str_value(item, "fp_side_value", "fp_suppressor"),
            transition_column=_str_value(item, "transition_column", "transition"),
            cluster_column=_str_value(item, "cluster_column", "strict_cluster_id"),
            neutral_side_value=_str_value(item, "neutral_side_value", "anchor,<blank>"),
            require_all_reviewed=_bool_value(item, "require_all_reviewed", True),
            require_sample_index=_bool_value(item, "require_sample_index", True),
            allow_non_train_paths=_bool_value(item, "allow_non_train_paths", False),
            min_actionable_recall=_int_value(item, "min_actionable_recall", 1),
            min_actionable_fp=_int_value(item, "min_actionable_fp", 1),
        )
        matrix_items.append(_flatten_item_summary(name, source_csv, item_output_dir, summary))

    ready_count = sum(1 for item in matrix_items if item["ready_for_training_manifest"])
    summary_payload: Dict[str, object] = {
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "config_json": str(Path(config_json).resolve()),
        "output_root": str(output_dir.resolve()),
        "ready_count": int(ready_count),
        "blocked_count": int(len(matrix_items) - ready_count),
        "items": matrix_items,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "guardrail": guardrail,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Review Worklist Readiness Matrix",
                "",
                "This artifact audits multiple manual review worklists before any reviewed-manifest path is used.",
                guardrail,
                "",
                f"Ready: `{ready_count}/{len(matrix_items)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readiness matrix for manual review worklists.")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", type=str, default="")
    parser.add_argument("--guardrail", type=str, default=DEFAULT_GUARDRAIL)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_review_worklist_readiness_matrix(
        config_json=args.config_json,
        output_dir=args.output_dir,
        created_at=args.created_at or None,
        guardrail=args.guardrail,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

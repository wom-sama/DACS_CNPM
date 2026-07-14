from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FOCUS_CLASS = 1
FOCUS_FP_CLASSES = {0, 2, 4}
DEFAULT_CATEGORY_QUOTAS: Tuple[Tuple[str, int], ...] = (
    ("focus_fp_removed", 3),
    ("focus_fp_added", 3),
    ("focus_fn_rescue", 2),
    ("focus_tp_break", 2),
    ("new_3_to_2_harm", 2),
    ("correction", 2),
    ("harm", 2),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(row: Mapping[str, str], key: str) -> int:
    return int(str(row.get(key, "") or "0"))


def classify_change(row: Mapping[str, str]) -> str:
    target = _as_int(row, "target")
    control = _as_int(row, "control_prediction")
    candidate = _as_int(row, "candidate_prediction")
    if target in FOCUS_FP_CLASSES and control == FOCUS_CLASS and candidate != FOCUS_CLASS:
        return "focus_fp_removed"
    if target in FOCUS_FP_CLASSES and control != FOCUS_CLASS and candidate == FOCUS_CLASS:
        return "focus_fp_added"
    if _as_int(row, "focus_fn_rescue"):
        return "focus_fn_rescue"
    if _as_int(row, "focus_tp_break"):
        return "focus_tp_break"
    if _as_int(row, "new_3_to_2_harm"):
        return "new_3_to_2_harm"
    if _as_int(row, "correction"):
        return "correction"
    if _as_int(row, "harm"):
        return "harm"
    return "other_changed"


def _selection_score(row: Mapping[str, str]) -> float:
    return abs(float(row["candidate_prob_1"]) - float(row["control_prob_1"]))


def select_cohort_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    max_cases: int = 16,
    category_quotas: Sequence[Tuple[str, int]] = DEFAULT_CATEGORY_QUOTAS,
) -> List[Dict[str, str]]:
    if int(max_cases) <= 0:
        raise ValueError("max_cases must be positive")
    prepared: List[Dict[str, str]] = []
    seen = set()
    for raw_row in rows:
        row = dict(raw_row)
        sample_index = str(row.get("sample_index", "") or "").strip()
        if not sample_index:
            raise ValueError("changed case is missing sample_index")
        if sample_index in seen:
            raise ValueError(f"duplicate sample_index in changed cases: {sample_index}")
        seen.add(sample_index)
        image_parts = {part.casefold() for part in Path(row["image_path"]).parts}
        if "test" in image_parts:
            raise ValueError(f"test-split case is forbidden: sample_index={sample_index}")
        row["xai_category"] = classify_change(row)
        row["focus_probability_delta"] = str(
            float(row["candidate_prob_1"]) - float(row["control_prob_1"])
        )
        row["selection_score"] = str(_selection_score(row))
        prepared.append(row)

    def order_key(row: Mapping[str, str]) -> Tuple[float, int]:
        return (-float(row["selection_score"]), int(row["sample_index"]))

    by_category: Dict[str, List[Dict[str, str]]] = {}
    for row in prepared:
        by_category.setdefault(row["xai_category"], []).append(row)
    for category_rows in by_category.values():
        category_rows.sort(key=order_key)

    selected: List[Dict[str, str]] = []
    selected_indices = set()
    for category, quota in category_quotas:
        for row in by_category.get(category, [])[: max(0, int(quota))]:
            selected.append(row)
            selected_indices.add(row["sample_index"])
            if len(selected) >= int(max_cases):
                break
        if len(selected) >= int(max_cases):
            break

    if len(selected) < int(max_cases):
        leftovers = [
            row for row in prepared if row["sample_index"] not in selected_indices
        ]
        leftovers.sort(key=order_key)
        selected.extend(leftovers[: int(max_cases) - len(selected)])

    result: List[Dict[str, str]] = []
    for rank, row in enumerate(selected[: int(max_cases)], start=1):
        result.append(
            {
                "cohort_rank": str(rank),
                "sample_index": row["sample_index"],
                "image_path": row["image_path"],
                "source_stem": row.get("source_stem", ""),
                "object_index": row.get("object_index", ""),
                "target_index": row["target"],
                "control_prediction_index": row["control_prediction"],
                "candidate_prediction_index": row["candidate_prediction"],
                "xai_category": row["xai_category"],
                "control_focus_probability": row["control_prob_1"],
                "candidate_focus_probability": row["candidate_prob_1"],
                "focus_probability_delta": row["focus_probability_delta"],
                "selection_score": row["selection_score"],
            }
        )
    return result


def build_cohort(
    *, changed_cases: Path, output_dir: Path, max_cases: int = 16
) -> Dict[str, object]:
    changed_cases = Path(changed_cases).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Cohort output already exists and is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with changed_cases.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_cohort_rows(rows, max_cases=max_cases)
    if len(selected) != min(int(max_cases), len(rows)):
        raise RuntimeError("cohort selection returned an unexpected number of cases")

    cohort_path = output_dir / "xai_priority_cases.csv"
    fields = list(selected[0]) if selected else []
    with cohort_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    category_counts = Counter(row["xai_category"] for row in selected)
    summary: Dict[str, object] = {
        "mode": "validation_only_visual_contrast_changed_case_cohort",
        "source": str(changed_cases),
        "source_sha256": _sha256(changed_cases),
        "changed_case_count": len(rows),
        "selected_case_count": len(selected),
        "category_counts": dict(sorted(category_counts.items())),
        "selection": {
            "category_quotas": dict(DEFAULT_CATEGORY_QUOTAS),
            "ranking": "absolute candidate-control class-1 probability delta, then sample_index",
        },
        "cohort": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path),
        "raw_dataset_modified": False,
        "test_used": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic validation-only VCA changed-case XAI cohort."
    )
    parser.add_argument("--changed-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_cohort(
        changed_cases=args.changed_cases,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

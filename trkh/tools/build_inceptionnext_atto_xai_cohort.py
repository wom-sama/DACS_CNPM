from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

from trkh.tools.build_visual_contrast_xai_cohort import (
    DEFAULT_CATEGORY_QUOTAS,
    _sha256,
    select_cohort_rows,
)


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
    if not selected:
        raise ValueError("At least one changed validation decision is required for paired XAI")

    cohort_path = output_dir / "xai_priority_cases.csv"
    with cohort_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    category_counts = Counter(row["xai_category"] for row in selected)
    summary: Dict[str, object] = {
        "mode": "validation_only_inceptionnext_atto_changed_case_cohort",
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
        description="Build a deterministic validation-only InceptionNeXt changed-case XAI cohort."
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

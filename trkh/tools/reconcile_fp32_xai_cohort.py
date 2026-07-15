from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


FOCUS_CLASS = 1
FOCUS_FP_CLASSES = {0, 2, 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_test_image_path(value: object) -> None:
    parts = {part.casefold() for part in Path(str(value)).parts}
    if "test" in parts:
        raise ValueError(f"XAI cohort reconciliation is validation-only: {value}")


def _read_cohort(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_index" not in reader.fieldnames:
            raise ValueError(f"Cohort CSV must contain sample_index: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Cohort CSV is empty: {path}")
    order: List[str] = []
    for row in rows:
        key = str(row.get("sample_index", "") or "").strip()
        if not key:
            raise ValueError("Cohort contains an empty sample_index")
        if key in order:
            raise ValueError(f"Duplicate cohort sample_index={key}")
        order.append(key)
    return order, rows


def _read_xai_cases(root: Path) -> Tuple[Dict[str, Dict[str, object]], str]:
    cases: Dict[str, Dict[str, object]] = {}
    digest = hashlib.sha256()
    for path in sorted(root.glob("case_*/case.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = str(payload.get("sample_index", "") or "").strip()
        if not key:
            raise ValueError(f"XAI case has no sample_index: {path}")
        if key in cases:
            raise ValueError(f"Duplicate XAI sample_index={key} under {root}")
        payload["_case_json"] = str(path.resolve())
        cases[key] = payload
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    if not cases:
        raise ValueError(f"No case_*/case.json artifacts found under {root}")
    return cases, digest.hexdigest()


def classify_fp32_change(target: int, control: int, candidate: int) -> str:
    if target in FOCUS_FP_CLASSES and control == FOCUS_CLASS and candidate != FOCUS_CLASS:
        return "focus_fp_removed"
    if target in FOCUS_FP_CLASSES and control != FOCUS_CLASS and candidate == FOCUS_CLASS:
        return "focus_fp_added"
    if target == FOCUS_CLASS and control != FOCUS_CLASS and candidate == FOCUS_CLASS:
        return "focus_fn_rescue"
    if target == FOCUS_CLASS and control == FOCUS_CLASS and candidate != FOCUS_CLASS:
        return "focus_tp_break"
    if target == 3 and control == 3 and candidate == 2:
        return "new_3_to_2_harm"
    control_correct = control == target
    candidate_correct = candidate == target
    if not control_correct and candidate_correct:
        return "correction"
    if control_correct and not candidate_correct:
        return "harm"
    if control == candidate:
        return "fp32_unchanged"
    return "other_changed"


def _normalized_path(value: object) -> str:
    return os.path.normcase(str(Path(str(value)).resolve()))


def _validate_case(
    *,
    key: str,
    row: Mapping[str, str],
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    target = int(str(row.get("target_index", "") or "-1"))
    if int(control.get("target_index", -2)) != target:
        raise ValueError(f"Control target mismatch for sample_index={key}")
    if int(candidate.get("target_index", -2)) != target:
        raise ValueError(f"Candidate target mismatch for sample_index={key}")
    if int(control.get("sample_index", -1)) != int(key):
        raise ValueError(f"Control sample_index mismatch for sample_index={key}")
    if int(candidate.get("sample_index", -1)) != int(key):
        raise ValueError(f"Candidate sample_index mismatch for sample_index={key}")
    cohort_path = str(row.get("image_path", "") or "").strip()
    if cohort_path:
        _reject_test_image_path(cohort_path)
        _reject_test_image_path(control.get("image_path", ""))
        _reject_test_image_path(candidate.get("image_path", ""))
        expected = _normalized_path(cohort_path)
        if _normalized_path(control.get("image_path", "")) != expected:
            raise ValueError(f"Control image path mismatch for sample_index={key}")
        if _normalized_path(candidate.get("image_path", "")) != expected:
            raise ValueError(f"Candidate image path mismatch for sample_index={key}")


def reconcile_cohort(
    *,
    cohort_csv: Path,
    control_xai_dir: Path,
    candidate_xai_dir: Path,
    output_dir: Path,
    expected_cases: int = 0,
) -> Dict[str, object]:
    cohort_csv = Path(cohort_csv).resolve()
    control_xai_dir = Path(control_xai_dir).resolve()
    candidate_xai_dir = Path(candidate_xai_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is nonempty: {output_dir}")

    order, rows = _read_cohort(cohort_csv)
    control_cases, control_set_sha = _read_xai_cases(control_xai_dir)
    candidate_cases, candidate_set_sha = _read_xai_cases(candidate_xai_dir)
    expected = set(order)
    if set(control_cases) != expected or set(candidate_cases) != expected:
        raise ValueError(
            "XAI sample set mismatch: "
            f"cohort={len(expected)}, control={len(control_cases)}, "
            f"candidate={len(candidate_cases)}"
        )
    if int(expected_cases) > 0 and len(order) != int(expected_cases):
        raise ValueError(f"Expected {expected_cases} cases, found {len(order)}")

    reconciled: List[Dict[str, str]] = []
    drift_rows: List[Dict[str, object]] = []
    original_categories: Counter[str] = Counter()
    fp32_categories: Counter[str] = Counter()
    for row in rows:
        key = str(row["sample_index"])
        control_case = control_cases[key]
        candidate_case = candidate_cases[key]
        _validate_case(
            key=key,
            row=row,
            control=control_case,
            candidate=candidate_case,
        )
        original_control = int(row["control_prediction_index"])
        original_candidate = int(row["candidate_prediction_index"])
        fp32_control = int(control_case["prediction_index"])
        fp32_candidate = int(candidate_case["prediction_index"])
        target = int(row["target_index"])
        original_category = str(row.get("xai_category", "") or "uncategorized")
        fp32_category = classify_fp32_change(target, fp32_control, fp32_candidate)
        control_drift = original_control != fp32_control
        candidate_drift = original_candidate != fp32_candidate

        updated = dict(row)
        updated.update(
            {
                "original_control_prediction_index": str(original_control),
                "original_candidate_prediction_index": str(original_candidate),
                "original_xai_category": original_category,
                "fp32_control_prediction_index": str(fp32_control),
                "fp32_candidate_prediction_index": str(fp32_candidate),
                "fp32_xai_category": fp32_category,
                "control_prediction_index": str(fp32_control),
                "candidate_prediction_index": str(fp32_candidate),
                "xai_category": fp32_category,
                "control_backend_prediction_drift": str(control_drift).lower(),
                "candidate_backend_prediction_drift": str(candidate_drift).lower(),
                "selection_backend": "original_full_validation_amp",
                "xai_backend": "fp32_disable_amp",
            }
        )
        reconciled.append(updated)
        original_categories[original_category] += 1
        fp32_categories[fp32_category] += 1
        if control_drift or candidate_drift or original_category != fp32_category:
            drift_rows.append(
                {
                    "sample_index": key,
                    "target_index": target,
                    "original_control_prediction_index": original_control,
                    "fp32_control_prediction_index": fp32_control,
                    "original_candidate_prediction_index": original_candidate,
                    "fp32_candidate_prediction_index": fp32_candidate,
                    "original_xai_category": original_category,
                    "fp32_xai_category": fp32_category,
                    "control_prediction_drift": control_drift,
                    "candidate_prediction_drift": candidate_drift,
                    "control_fp32_confidence": float(control_case.get("confidence", 0.0)),
                    "control_fp32_margin": float(control_case.get("margin", 0.0)),
                    "candidate_fp32_confidence": float(candidate_case.get("confidence", 0.0)),
                    "candidate_fp32_margin": float(candidate_case.get("margin", 0.0)),
                    "control_case_json": str(control_case["_case_json"]),
                    "candidate_case_json": str(candidate_case["_case_json"]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "xai_priority_cases_fp32.csv"
    fieldnames: List[str] = []
    for row in reconciled:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with cohort_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reconciled)

    drift_path = output_dir / "backend_prediction_drifts.csv"
    drift_fields = list(drift_rows[0]) if drift_rows else [
        "sample_index",
        "target_index",
        "original_control_prediction_index",
        "fp32_control_prediction_index",
        "original_candidate_prediction_index",
        "fp32_candidate_prediction_index",
        "original_xai_category",
        "fp32_xai_category",
        "control_prediction_drift",
        "candidate_prediction_drift",
    ]
    with drift_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=drift_fields)
        writer.writeheader()
        writer.writerows(drift_rows)

    summary: Dict[str, object] = {
        "mode": "validation_only_fp32_xai_cohort_reconciliation",
        "source_cohort": str(cohort_csv),
        "source_cohort_sha256": _sha256(cohort_csv),
        "selection_backend": "original_full_validation_amp",
        "xai_backend": "fp32_disable_amp",
        "selection_policy": (
            "Keep the exact original AMP-selected sample set and ranking; replace only "
            "the prediction/category fields consumed by paired FP32 XAI validation."
        ),
        "cases": len(reconciled),
        "control": {
            "xai_dir": str(control_xai_dir),
            "case_json_set_sha256": control_set_sha,
            "prediction_drift_count": sum(
                row["original_control_prediction_index"]
                != row["fp32_control_prediction_index"]
                for row in drift_rows
            ),
        },
        "candidate": {
            "xai_dir": str(candidate_xai_dir),
            "case_json_set_sha256": candidate_set_sha,
            "prediction_drift_count": sum(
                row["original_candidate_prediction_index"]
                != row["fp32_candidate_prediction_index"]
                for row in drift_rows
            ),
        },
        "any_prediction_drift_count": sum(
            bool(row["control_prediction_drift"] or row["candidate_prediction_drift"])
            for row in drift_rows
        ),
        "category_change_count": sum(
            row["original_xai_category"] != row["fp32_xai_category"]
            for row in drift_rows
        ),
        "original_category_counts": dict(sorted(original_categories.items())),
        "fp32_category_counts": dict(sorted(fp32_categories.items())),
        "reconciled_cohort": str(cohort_path),
        "reconciled_cohort_sha256": _sha256(cohort_path),
        "backend_prediction_drifts": str(drift_path),
        "backend_prediction_drifts_sha256": _sha256(drift_path),
        "raw_dataset_modified": False,
        "test_used": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile an AMP-selected cohort with exact FP32 XAI predictions."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--control-xai-dir", type=Path, required=True)
    parser.add_argument("--candidate-xai-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = reconcile_cohort(
        cohort_csv=args.cohort,
        control_xai_dir=args.control_xai_dir,
        candidate_xai_dir=args.candidate_xai_dir,
        output_dir=args.output_dir,
        expected_cases=args.expected_cases,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

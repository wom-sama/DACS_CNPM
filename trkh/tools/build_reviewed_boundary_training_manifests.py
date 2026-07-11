from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trkh.tools.audit_review_worklist_readiness import audit_review_worklist_readiness


def _read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reviewed boundary manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _read_fieldnames(path: Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return list(reader.fieldnames)


def _infer_readiness_config(fieldnames: Sequence[str]) -> Dict[str, str]:
    fields = set(fieldnames)
    if "strict_review_side" in fields:
        return {
            "side_column": "strict_review_side",
            "recall_side_value": "recall_protector",
            "fp_side_value": "fp_suppressor",
            "cluster_column": "strict_cluster_id" if "strict_cluster_id" in fields else "",
        }
    if "triage_group" in fields:
        return {
            "side_column": "triage_group",
            "recall_side_value": "fn1_all,tp1_low_margin_recall_protector",
            "fp_side_value": (
                "fp0_to_1_suppressor_review,fp2_to_1_suppressor_review,"
                "fp_other_to_1_complex_review"
            ),
            "cluster_column": "strict_cluster_id" if "strict_cluster_id" in fields else "",
        }
    if "reason" in fields:
        return {
            "side_column": "reason",
            "recall_side_value": "focus_false_negative",
            "fp_side_value": "focus_false_positive",
            "cluster_column": "strict_cluster_id" if "strict_cluster_id" in fields else "",
        }
    return {
        "side_column": "",
        "recall_side_value": "",
        "fp_side_value": "",
        "cluster_column": "",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _path_value(row: Mapping[str, str]) -> str:
    return (
        str(row.get("image_path", "") or "").strip()
        or str(row.get("path", "") or "").strip()
        or str(row.get("sample_path", "") or "").strip()
    )


def _looks_like_non_train(path_text: str) -> bool:
    parts = [part for part in str(path_text or "").replace("\\", "/").lower().split("/") if part]
    return "val" in parts or "valid" in parts or "validation" in parts or "test" in parts


def _int_value(row: Mapping[str, str], *keys: str, default: int = -1) -> int:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            return int(default)
    return int(default)


def _sample_index_value(row: Mapping[str, str]) -> Optional[int]:
    for key in ("sample_index", "source_row_index", "dataset_index"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            parsed = int(float(value))
        except ValueError:
            continue
        if parsed >= 0:
            return int(parsed)
    return None


def _float_value(row: Mapping[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return float(parsed)
    return float(default)


def _parse_pairs(text: str) -> set[Tuple[int, int]]:
    pairs: set[Tuple[int, int]] = set()
    for raw_item in str(text or "").replace(";", ",").split(","):
        item = raw_item.strip().replace(":", "-")
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid pair {raw_item!r}; expected A-B.")
        left, right = [part.strip() for part in item.split("-", 1)]
        a, b = int(left), int(right)
        pairs.add(tuple(sorted((a, b))))
    return pairs


def _pair_allowed(a: int, b: int, pairs: set[Tuple[int, int]]) -> bool:
    if not pairs:
        return True
    return tuple(sorted((int(a), int(b)))) in pairs


def _parse_expected_class(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return -1
    try:
        return int(float(text))
    except ValueError:
        for token in text.replace(";", ",").replace("/", ",").split(","):
            token = token.strip()
            if token.isdigit():
                return int(token)
    return -1


def _manual_status(row: Mapping[str, str]) -> str:
    return str(row.get("manual_label_status", "") or "").strip().lower()


def _quality_notes(row: Mapping[str, str]) -> List[str]:
    notes: List[str] = []
    for key in (
        "quality_lighting",
        "quality_dirty_obstacle",
        "quality_partial_fruit",
        "quality_background_mask",
    ):
        value = str(row.get(key, "") or "").strip()
        if value:
            notes.append(f"{key}={value}")
    return notes


def _quality_weight_cap(
    row: Mapping[str, str],
    *,
    lighting_weight: float,
    dirty_weight: float,
    partial_weight: float,
    background_weight: float,
) -> float:
    cap = 1.0
    lighting = str(row.get("quality_lighting", "") or "").strip().lower()
    if lighting and lighting not in {"ok", "good", "normal", "none"}:
        cap = min(cap, float(lighting_weight))
    dirty = str(row.get("quality_dirty_obstacle", "") or "").strip().lower()
    if dirty and dirty not in {"ok", "clean", "none", "minor"}:
        cap = min(cap, float(dirty_weight))
    partial = str(row.get("quality_partial_fruit", "") or "").strip().lower()
    if partial and partial not in {"ok", "full", "none"}:
        cap = min(cap, float(partial_weight))
    background = str(row.get("quality_background_mask", "") or "").strip().lower()
    if background and background not in {"ok", "clean", "none"}:
        cap = min(cap, float(background_weight))
    return float(cap)


def _manual_sample_weight_override(row: Mapping[str, str]) -> Optional[float]:
    for key in ("manual_sample_weight", "review_sample_weight"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            return float(parsed)
    return None


def _choose_soft_index(row: Mapping[str, str], status: str) -> int:
    target = _int_value(row, "target_index", "y_true")
    manual_expected = _parse_expected_class(str(row.get("manual_expected_class", "") or ""))
    if status == "ambiguous" and manual_expected >= 0 and manual_expected != target:
        return int(manual_expected)
    prediction = _int_value(row, "prediction_index", "y_pred")
    if prediction >= 0 and prediction != target:
        return int(prediction)
    top2 = _int_value(row, "top2_index")
    if top2 >= 0 and top2 != target:
        return int(top2)
    return -1


def _soft_row(
    *,
    image_path: str,
    sample_index: Optional[int],
    target_index: int,
    soft_index: int,
    alpha: float,
    num_classes: int,
    reason: str,
    source_row: Mapping[str, str],
) -> Dict[str, object]:
    probabilities = [0.0 for _ in range(num_classes)]
    probabilities[target_index] = 1.0 - float(alpha)
    probabilities[soft_index] += float(alpha)
    row: Dict[str, object] = {
        "image_path": image_path,
        "sample_index": "" if sample_index is None else int(sample_index),
        "target_index": int(target_index),
        "soft_target_index": int(soft_index),
        "alpha": f"{float(alpha):.10g}",
        "reason": reason,
        "review_id": source_row.get("review_id", ""),
        "prediction_index": source_row.get("prediction_index", ""),
        "top2_margin": source_row.get("top2_margin", ""),
    }
    row.update({f"soft_{index}": f"{probabilities[index]:.10g}" for index in range(num_classes)})
    return row


def _resolved_key(path_text: str) -> str:
    return str(Path(path_text).resolve()).lower()


def _dedup_key(row: Mapping[str, object]) -> str:
    sample_index = row.get("sample_index", "")
    sample_text = str(sample_index if sample_index is not None else "").strip()
    if sample_text:
        try:
            parsed = int(float(sample_text))
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return f"sample:{parsed}"
    return f"path:{_resolved_key(str(row.get('image_path', '') or ''))}"


def _deduplicate_sample_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        key = _dedup_key(row)
        if not key or key == "path:":
            continue
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = dict(row)
            continue
        previous_weight = _float_object(previous.get("sample_weight"), default=1.0)
        current_weight = _float_object(row.get("sample_weight"), default=1.0)
        if current_weight < previous_weight:
            deduped[key] = dict(row)
    return list(deduped.values())


def _deduplicate_by_path(
    rows: Sequence[Dict[str, object]],
    *,
    allowed_path_keys: Optional[set[str]] = None,
) -> List[Dict[str, object]]:
    deduped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        key = _dedup_key(row)
        if not key or key == "path:":
            continue
        if allowed_path_keys is not None and key not in allowed_path_keys:
            continue
        if key not in deduped:
            deduped[key] = dict(row)
    return list(deduped.values())


def _float_object(value: object, *, default: float) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def build_manifests(
    *,
    reviewed_manifest: Path,
    output_dir: Path,
    num_classes: int,
    soft_target_pairs: str,
    soft_alpha: float,
    correct_error_weight: float,
    ambiguous_weight: float,
    wrong_weight: float,
    needs_crop_weight: float,
    lighting_weight: float,
    dirty_weight: float,
    partial_weight: float,
    background_weight: float,
    targeted_margin: float,
    targeted_margin_weight: float,
    max_targeted_margin_weight: float,
    allow_non_train_paths: bool,
    dry_run: bool,
) -> Dict[str, object]:
    rows, fieldnames = _read_csv(Path(reviewed_manifest))
    if "manual_label_status" not in fieldnames:
        raise ValueError("Reviewed manifest must contain manual_label_status.")
    soft_pairs = _parse_pairs(soft_target_pairs)
    num_classes = max(1, int(num_classes))
    soft_alpha = max(0.0, min(1.0, float(soft_alpha)))
    targeted_margin_weight = max(0.0, min(float(max_targeted_margin_weight), float(targeted_margin_weight)))

    sample_rows: List[Dict[str, object]] = []
    soft_rows: List[Dict[str, object]] = []
    targeted_rows: List[Dict[str, object]] = []
    relabel_rows: List[Dict[str, object]] = []
    reviewed_rows: List[Dict[str, object]] = []
    skipped_non_train = 0
    skipped_unreviewed = 0
    skipped_invalid = 0
    by_status: Counter[str] = Counter()
    by_decision: Counter[str] = Counter()
    by_soft_pair: Counter[str] = Counter()
    by_targeted_pair: Counter[str] = Counter()
    manual_sample_weight_rows = 0

    for row in rows:
        image_path = _path_value(row)
        if not image_path:
            skipped_invalid += 1
            continue
        if _looks_like_non_train(image_path) and not allow_non_train_paths:
            skipped_non_train += 1
            continue
        target = _int_value(row, "target_index", "y_true")
        prediction = _int_value(row, "prediction_index", "y_pred")
        sample_index = _sample_index_value(row)
        if not (0 <= target < num_classes):
            skipped_invalid += 1
            continue
        status = _manual_status(row)
        if not status:
            skipped_unreviewed += 1
            continue
        if status not in {"correct", "ambiguous", "wrong", "needs_crop"}:
            skipped_invalid += 1
            continue

        by_status[status] += 1
        reason_parts = [f"manual_{status}"]
        source_reason = str(row.get("reason", "") or "").strip()
        if source_reason:
            reason_parts.append(source_reason)
        quality_notes = _quality_notes(row)
        if quality_notes:
            reason_parts.extend(quality_notes)
        reason = ";".join(reason_parts)
        quality_cap = _quality_weight_cap(
            row,
            lighting_weight=lighting_weight,
            dirty_weight=dirty_weight,
            partial_weight=partial_weight,
            background_weight=background_weight,
        )

        decision = "keep"
        weight = 1.0
        if status == "correct":
            if prediction >= 0 and prediction != target:
                decision = "targeted_margin"
                weight = (
                    min(float(correct_error_weight), quality_cap)
                    if quality_cap < 1.0
                    else float(correct_error_weight)
                )
                if targeted_margin_weight > 0.0:
                    targeted_rows.append(
                        {
                            "image_path": image_path,
                            "sample_index": "" if sample_index is None else int(sample_index),
                            "target_index": int(target),
                            "negative_index": int(prediction),
                            "prediction_index": int(prediction),
                            "targeted_margin": f"{float(targeted_margin):.10g}",
                            "targeted_margin_weight": f"{float(targeted_margin_weight):.10g}",
                            "reason": reason,
                            "review_id": row.get("review_id", ""),
                        }
                    )
                    by_targeted_pair[f"{target}->{prediction}"] += 1
            else:
                weight = min(1.0, quality_cap)
                if weight < 1.0:
                    decision = "quality_downweight"
        elif status == "ambiguous":
            decision = "soft_boundary"
            weight = min(float(ambiguous_weight), quality_cap)
            soft_index = _choose_soft_index(row, status)
            if (
                0 <= soft_index < num_classes
                and soft_index != target
                and _pair_allowed(target, soft_index, soft_pairs)
            ):
                soft_rows.append(
                    _soft_row(
                        image_path=image_path,
                        sample_index=sample_index,
                        target_index=int(target),
                        soft_index=int(soft_index),
                        alpha=soft_alpha,
                        num_classes=num_classes,
                        reason=reason,
                        source_row=row,
                    )
                )
                by_soft_pair[f"{target}->{soft_index}"] += 1
            else:
                decision = "ambiguous_downweight"
        elif status == "wrong":
            decision = "relabel_candidate_downweight"
            weight = min(float(wrong_weight), quality_cap)
            relabel_rows.append(
                {
                    "image_path": image_path,
                    "sample_index": "" if sample_index is None else int(sample_index),
                    "target_index": int(target),
                    "manual_expected_class": _parse_expected_class(
                        str(row.get("manual_expected_class", "") or "")
                    ),
                    "prediction_index": int(prediction),
                    "reason": reason,
                    "review_id": row.get("review_id", ""),
                }
            )
        elif status == "needs_crop":
            decision = "needs_crop_downweight"
            weight = min(float(needs_crop_weight), quality_cap)

        manual_weight = _manual_sample_weight_override(row)
        if manual_weight is not None:
            manual_sample_weight_rows += 1
            weight = float(manual_weight)
            if quality_cap < 1.0:
                weight = min(weight, quality_cap)
            if decision == "keep" and abs(float(weight) - 1.0) > 1e-9:
                decision = "manual_weight"
            reason = f"{reason};manual_sample_weight={float(weight):.10g}"

        weight = max(1e-6, float(weight))
        sample_rows.append(
            {
                "image_path": image_path,
                "sample_index": "" if sample_index is None else int(sample_index),
                "target_index": int(target),
                "target_name": row.get("target_name", ""),
                "prediction_index": "" if prediction < 0 else int(prediction),
                "prediction_name": row.get("prediction_name", ""),
                "sample_weight": f"{weight:.10g}",
                "reason": reason,
                "review_decision": decision,
                "review_id": row.get("review_id", ""),
            }
        )
        reviewed_copy = dict(row)
        reviewed_copy["review_decision"] = row.get("review_decision", "") or decision
        reviewed_rows.append(reviewed_copy)
        by_decision[decision] += 1

    if skipped_non_train and not allow_non_train_paths:
        raise ValueError(
            "Reviewed manifest must be train-only unless allow_non_train_paths=True; "
            f"skipped_non_train={skipped_non_train}"
        )
    raw_sample_rows = len(sample_rows)
    raw_soft_rows = len(soft_rows)
    raw_targeted_rows = len(targeted_rows)
    raw_relabel_rows = len(relabel_rows)
    sample_rows = _deduplicate_sample_rows(sample_rows)
    sample_decisions_by_key = {
        _dedup_key(row): str(row.get("review_decision", "") or "")
        for row in sample_rows
    }
    soft_allowed = {
        key for key, decision in sample_decisions_by_key.items() if decision == "soft_boundary"
    }
    targeted_allowed = {
        key for key, decision in sample_decisions_by_key.items() if decision == "targeted_margin"
    }
    relabel_allowed = {
        key for key, decision in sample_decisions_by_key.items() if decision == "relabel_candidate_downweight"
    }
    soft_rows = _deduplicate_by_path(soft_rows, allowed_path_keys=soft_allowed)
    targeted_rows = _deduplicate_by_path(targeted_rows, allowed_path_keys=targeted_allowed)
    relabel_rows = _deduplicate_by_path(relabel_rows, allowed_path_keys=relabel_allowed)
    by_decision = Counter(str(row.get("review_decision", "") or "") for row in sample_rows)
    by_soft_pair = Counter(
        f"{row.get('target_index')}->{row.get('soft_target_index')}" for row in soft_rows
    )
    by_targeted_pair = Counter(
        f"{row.get('target_index')}->{row.get('negative_index')}" for row in targeted_rows
    )
    if not sample_rows:
        raise ValueError("Reviewed manifest produced no valid sample-weight rows.")

    output_dir = Path(output_dir)
    sample_path = output_dir / "sample_weights_reviewed_train_only.csv"
    soft_path = output_dir / "ambiguous_soft_targets_reviewed_train_only.csv"
    targeted_path = output_dir / "targeted_margin_reviewed_train_only.csv"
    relabel_path = output_dir / "relabel_candidates_reviewed_train_only.csv"
    reviewed_path = output_dir / "boundary_review_manifest_reviewed.csv"

    sample_fieldnames = [
        "image_path",
        "sample_index",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "sample_weight",
        "reason",
        "review_decision",
        "review_id",
    ]
    soft_fieldnames = [
        "image_path",
        "sample_index",
        "target_index",
        "soft_target_index",
        "alpha",
        "reason",
        "review_id",
        "prediction_index",
        "top2_margin",
        *[f"soft_{index}" for index in range(num_classes)],
    ]
    targeted_fieldnames = [
        "image_path",
        "sample_index",
        "target_index",
        "negative_index",
        "prediction_index",
        "targeted_margin",
        "targeted_margin_weight",
        "reason",
        "review_id",
    ]
    relabel_fieldnames = [
        "image_path",
        "sample_index",
        "target_index",
        "manual_expected_class",
        "prediction_index",
        "reason",
        "review_id",
    ]
    reviewed_fieldnames = list(fieldnames)
    if "review_decision" not in reviewed_fieldnames:
        reviewed_fieldnames.append("review_decision")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(sample_path, sample_rows, sample_fieldnames)
        _write_csv(soft_path, soft_rows, soft_fieldnames)
        _write_csv(targeted_path, targeted_rows, targeted_fieldnames)
        _write_csv(relabel_path, relabel_rows, relabel_fieldnames)
        _write_csv(reviewed_path, reviewed_rows, reviewed_fieldnames)

    weights = [float(row["sample_weight"]) for row in sample_rows]
    summary = {
        "reviewed_manifest": str(Path(reviewed_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "dry_run": bool(dry_run),
        "input_rows": int(len(rows)),
        "sample_weight_rows": int(len(sample_rows)),
        "soft_target_rows": int(len(soft_rows)),
        "targeted_margin_rows": int(len(targeted_rows)),
        "relabel_candidate_rows": int(len(relabel_rows)),
        "raw_rows_before_path_dedup": {
            "sample_weight": int(raw_sample_rows),
            "soft_target": int(raw_soft_rows),
            "targeted_margin": int(raw_targeted_rows),
            "relabel_candidate": int(raw_relabel_rows),
        },
        "sample_index_rows": {
            "sample_weight": int(sum(1 for row in sample_rows if str(row.get("sample_index", "") or "").strip())),
            "soft_target": int(sum(1 for row in soft_rows if str(row.get("sample_index", "") or "").strip())),
            "targeted_margin": int(sum(1 for row in targeted_rows if str(row.get("sample_index", "") or "").strip())),
            "relabel_candidate": int(sum(1 for row in relabel_rows if str(row.get("sample_index", "") or "").strip())),
        },
        "manual_sample_weight_rows": int(manual_sample_weight_rows),
        "skipped_non_train": int(skipped_non_train),
        "skipped_unreviewed": int(skipped_unreviewed),
        "skipped_invalid": int(skipped_invalid),
        "by_status": dict(by_status),
        "by_decision": dict(by_decision),
        "by_soft_pair": dict(by_soft_pair),
        "by_targeted_pair": dict(by_targeted_pair),
        "weight_range": {
            "min": float(min(weights)) if weights else 0.0,
            "max": float(max(weights)) if weights else 0.0,
            "mean": float(sum(weights) / max(1, len(weights))) if weights else 0.0,
        },
        "outputs": {
            "sample_weights": str(sample_path.resolve()),
            "ambiguous_soft_targets": str(soft_path.resolve()),
            "targeted_margin": str(targeted_path.resolve()),
            "relabel_candidates": str(relabel_path.resolve()),
            "reviewed_manifest": str(reviewed_path.resolve()),
        },
        "policy": {
            "soft_target_pairs": sorted([f"{a}-{b}" for a, b in soft_pairs]),
            "soft_alpha": float(soft_alpha),
            "correct_error_weight": float(correct_error_weight),
            "ambiguous_weight": float(ambiguous_weight),
            "wrong_weight": float(wrong_weight),
            "needs_crop_weight": float(needs_crop_weight),
            "targeted_margin": float(targeted_margin),
            "targeted_margin_weight": float(targeted_margin_weight),
        },
        "leakage_guard": "train-only by default; non-train paths raise unless allow_non_train_paths=True",
    }
    if not dry_run:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a manually reviewed train-only boundary manifest into training manifests."
    )
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--soft-target-pairs", type=str, default="0-1,1-2,2-3")
    parser.add_argument("--soft-alpha", type=float, default=0.25)
    parser.add_argument("--correct-error-weight", type=float, default=1.10)
    parser.add_argument("--ambiguous-weight", type=float, default=0.65)
    parser.add_argument("--wrong-weight", type=float, default=0.20)
    parser.add_argument("--needs-crop-weight", type=float, default=0.45)
    parser.add_argument("--lighting-weight", type=float, default=0.85)
    parser.add_argument("--dirty-weight", type=float, default=0.80)
    parser.add_argument("--partial-weight", type=float, default=0.65)
    parser.add_argument("--background-weight", type=float, default=0.85)
    parser.add_argument("--targeted-margin", type=float, default=0.10)
    parser.add_argument("--targeted-margin-weight", type=float, default=0.70)
    parser.add_argument("--max-targeted-margin-weight", type=float, default=1.25)
    parser.add_argument("--allow-non-train-paths", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--skip-readiness-check",
        action="store_true",
        default=False,
        help="Bypass the review readiness guard. Use only for legacy/debug workflows.",
    )
    parser.add_argument(
        "--readiness-output-dir",
        type=Path,
        default=None,
        help="Where to write readiness_summary.json. Defaults to output-dir/readiness_audit.",
    )
    parser.add_argument("--readiness-side-column", type=str, default=None)
    parser.add_argument("--readiness-recall-side-value", type=str, default=None)
    parser.add_argument("--readiness-fp-side-value", type=str, default=None)
    parser.add_argument("--readiness-cluster-column", type=str, default=None)
    parser.add_argument("--readiness-min-actionable-recall", type=int, default=20)
    parser.add_argument("--readiness-min-actionable-fp", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    readiness_summary = None
    if not args.skip_readiness_check:
        inferred = _infer_readiness_config(_read_fieldnames(args.reviewed_manifest))
        readiness_output_dir = (
            Path(args.readiness_output_dir)
            if args.readiness_output_dir is not None
            else Path(args.output_dir) / "readiness_audit"
        )
        side_column = (
            args.readiness_side_column
            if args.readiness_side_column is not None
            else inferred["side_column"]
        )
        recall_side_value = (
            args.readiness_recall_side_value
            if args.readiness_recall_side_value is not None
            else inferred["recall_side_value"]
        )
        fp_side_value = (
            args.readiness_fp_side_value
            if args.readiness_fp_side_value is not None
            else inferred["fp_side_value"]
        )
        cluster_column = (
            args.readiness_cluster_column
            if args.readiness_cluster_column is not None
            else inferred["cluster_column"]
        )
        min_recall = int(args.readiness_min_actionable_recall)
        min_fp = int(args.readiness_min_actionable_fp)
        if not side_column:
            min_recall = 0
            min_fp = 0
        readiness_summary = audit_review_worklist_readiness(
            csv_path=args.reviewed_manifest,
            output_dir=readiness_output_dir,
            side_column=side_column,
            recall_side_value=recall_side_value,
            fp_side_value=fp_side_value,
            cluster_column=cluster_column,
            allow_non_train_paths=args.allow_non_train_paths,
            min_actionable_recall=min_recall,
            min_actionable_fp=min_fp,
        )
        if not args.dry_run and not bool(readiness_summary["ready_for_training_manifest"]):
            reasons = ", ".join(str(item) for item in readiness_summary["blocking_reasons"])
            raise ValueError(
                "Reviewed manifest readiness check failed; refusing non-dry-run "
                f"manifest build. reasons=[{reasons}]. "
                f"See {readiness_output_dir / 'readiness_summary.json'}"
            )
    summary = build_manifests(
        reviewed_manifest=args.reviewed_manifest,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        soft_target_pairs=args.soft_target_pairs,
        soft_alpha=args.soft_alpha,
        correct_error_weight=args.correct_error_weight,
        ambiguous_weight=args.ambiguous_weight,
        wrong_weight=args.wrong_weight,
        needs_crop_weight=args.needs_crop_weight,
        lighting_weight=args.lighting_weight,
        dirty_weight=args.dirty_weight,
        partial_weight=args.partial_weight,
        background_weight=args.background_weight,
        targeted_margin=args.targeted_margin,
        targeted_margin_weight=args.targeted_margin_weight,
        max_targeted_margin_weight=args.max_targeted_margin_weight,
        allow_non_train_paths=args.allow_non_train_paths,
        dry_run=args.dry_run,
    )
    if readiness_summary is not None:
        summary["readiness_check"] = {
            "ready_for_training_manifest": bool(readiness_summary["ready_for_training_manifest"]),
            "blocking_reasons": list(readiness_summary["blocking_reasons"]),
            "readiness_output_dir": str(
                (
                    Path(args.readiness_output_dir)
                    if args.readiness_output_dir is not None
                    else Path(args.output_dir) / "readiness_audit"
                ).resolve()
            ),
        }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

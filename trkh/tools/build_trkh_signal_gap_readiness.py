from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed == parsed else float(default)


def _load_json(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _normalize_transition(value: object) -> str:
    text = str(value or "").strip().replace("=>", "->").replace("→", "->")
    if "->" not in text and "-" in text:
        text = text.replace("-", "->", 1)
    text = text.replace(" ", "")
    if "->" not in text:
        return ""
    left, right = text.split("->", 1)
    left_index = _safe_int(left, -1)
    right_index = _safe_int(right, -1)
    if left_index < 0 or right_index < 0:
        return ""
    return f"{left_index}->{right_index}"


def _transition_from_row(row: Mapping[str, object]) -> str:
    for key in ("risk_transition", "transition", "error_pair", "pair"):
        transition = _normalize_transition(row.get(key, ""))
        if transition:
            return transition
    target = _safe_int(row.get("target_index", row.get("target", "")), -1)
    prediction = _safe_int(
        row.get(
            "prediction_index",
            row.get("base_prediction_index", row.get("base_pred", row.get("prediction", ""))),
        ),
        -1,
    )
    if target >= 0 and prediction >= 0:
        return f"{target}->{prediction}"
    return ""


def _manual_filled(row: Mapping[str, object]) -> bool:
    return bool(str(row.get("manual_label_status", "") or "").strip())


def _classification_metrics_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    num_classes: int,
) -> Dict[str, object]:
    confusion = [[0 for _ in range(int(num_classes))] for _ in range(int(num_classes))]
    for row in rows:
        target = _safe_int(row.get("target_index", row.get("target", "")), -1)
        prediction = _safe_int(
            row.get("prediction_index", row.get("prediction", row.get("pred_index", ""))),
            -1,
        )
        if 0 <= target < int(num_classes) and 0 <= prediction < int(num_classes):
            confusion[target][prediction] += 1
    per_class: list[dict[str, object]] = []
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(int(num_classes)))
    for index in range(int(num_classes)):
        tp = float(confusion[index][index])
        support = float(sum(confusion[index]))
        predicted = float(sum(confusion[row][index] for row in range(int(num_classes))))
        precision = tp / predicted if predicted > 0.0 else 0.0
        recall = tp / support if support > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        per_class.append(
            {
                "class_index": int(index),
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(predicted - tp),
                "fn": int(support - tp),
            }
        )
    return {
        "samples": int(total),
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "macro_f1": float(sum(float(item["f1"]) for item in per_class) / max(1, int(num_classes))),
        "per_class": per_class,
    }


def _focus_class_metrics(metrics: Mapping[str, object], focus_class_index: int) -> Dict[str, object]:
    per_class = metrics.get("per_class", [])
    if isinstance(per_class, list):
        for item in per_class:
            if isinstance(item, Mapping) and _safe_int(item.get("class_index", -1), -1) == int(focus_class_index):
                return {
                    "f1": float(_safe_float(item.get("f1"), 0.0)),
                    "precision": float(_safe_float(item.get("precision"), 0.0)),
                    "recall": float(_safe_float(item.get("recall"), 0.0)),
                    "tp": int(_safe_int(item.get("tp"), 0)),
                    "fp": int(_safe_int(item.get("fp"), 0)),
                    "fn": int(_safe_int(item.get("fn"), 0)),
                }
    return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}


def _external_consensus_summary(
    external_support_summary: Mapping[str, object],
    *,
    focus_class_index: int,
) -> Dict[str, object]:
    route = external_support_summary.get("diagnostic_consensus_route", {})
    if not isinstance(route, Mapping):
        route = {}
    metrics = route.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    focus = _focus_class_metrics(metrics, focus_class_index)
    return {
        "macro_f1": float(_safe_float(metrics.get("macro_f1"), 0.0)),
        "class1_f1": float(focus["f1"]),
        "class1_precision": float(focus["precision"]),
        "class1_recall": float(focus["recall"]),
        "stats": route.get("stats", {}) if isinstance(route.get("stats", {}), Mapping) else {},
    }


def _focus_consensus_summary(
    external_support_summary: Mapping[str, object],
    *,
    focus_class_index: int,
) -> Dict[str, object]:
    route = external_support_summary.get("diagnostic_focus_consensus_route", {})
    if not isinstance(route, Mapping):
        return {}
    metrics = route.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    focus = _focus_class_metrics(metrics, focus_class_index)
    return {
        "macro_f1": float(_safe_float(metrics.get("macro_f1"), 0.0)),
        "class1_f1": float(focus["f1"]),
        "class1_precision": float(focus["precision"]),
        "class1_recall": float(focus["recall"]),
        "stats": route.get("stats", {}) if isinstance(route.get("stats", {}), Mapping) else {},
    }


def _budget_by_transition(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        transition = _normalize_transition(row.get("transition", ""))
        if not transition:
            target = _safe_int(row.get("target"), -1)
            prediction = _safe_int(row.get("prediction"), -1)
            if target >= 0 and prediction >= 0:
                transition = f"{target}->{prediction}"
        if transition:
            output[transition] = dict(row)
    return output


def _external_transition_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        transition = _transition_from_row(row)
        if not transition:
            continue
        target = _safe_int(row.get("target_index"), -1)
        base_prediction = _safe_int(row.get("base_prediction_index"), -1)
        consensus_prediction = _safe_int(row.get("external_consensus_prediction"), -1)
        consensus_votes = _safe_int(row.get("external_consensus_votes"), 0)
        consensus_tied = _safe_int(row.get("external_consensus_tied"), 0)
        external_correct_count = _safe_int(row.get("external_correct_count"), 0)
        if external_correct_count > 0:
            counts[transition]["val_any_external_correct"] += 1
        if external_correct_count >= 2:
            counts[transition]["val_both_external_correct"] += 1
        if consensus_votes >= 2 and consensus_tied == 0 and consensus_prediction == target:
            counts[transition]["val_consensus_correct"] += 1
        elif (
            consensus_votes >= 2
            and consensus_tied == 0
            and consensus_prediction >= 0
            and consensus_prediction != target
            and consensus_prediction != base_prediction
        ):
            counts[transition]["val_consensus_wrong_change"] += 1
    return counts


def _train_oof_transition_counts(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    by_transition: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    thresholds = (0.0, 0.5, 0.7, 0.9)
    for row in rows:
        transition = _transition_from_row(row)
        if transition:
            by_transition[transition]["train_oof_rows"] += 1
        bucket = str(row.get("bucket", "") or "").strip() or "unknown"
        min_conf = _safe_float(row.get("min_conf"), 0.0)
        external_correct = _safe_int(row.get("external_correct_count"), 0)
        same_1 = _safe_int(row.get("external_same_1"), 0)
        same_non1 = _safe_int(row.get("external_same_non1"), 0)
        for threshold in thresholds:
            suffix = f"{threshold:g}"
            if same_1 and min_conf >= threshold:
                by_bucket[bucket][f"same_1_minconf_{suffix}"] += 1
                if transition:
                    by_transition[transition][f"train_oof_same1_minconf_{suffix.replace('.', 'p')}"] += 1
            if same_non1 and min_conf >= threshold:
                by_bucket[bucket][f"same_non1_minconf_{suffix}"] += 1
                if transition:
                    by_transition[transition][f"train_oof_samenon1_minconf_{suffix.replace('.', 'p')}"] += 1
            if external_correct >= 2 and min_conf >= threshold:
                by_bucket[bucket][f"both_true_minconf_{suffix}"] += 1
                if transition:
                    by_transition[transition][f"train_oof_both_correct_minconf_{suffix.replace('.', 'p')}"] += 1
    return by_transition, by_bucket


def _review_counts(review_csvs: Mapping[str, Path]) -> tuple[dict[str, Counter[str]], dict[str, dict[str, object]]]:
    by_transition: dict[str, Counter[str]] = defaultdict(Counter)
    source_totals: dict[str, dict[str, object]] = {}
    for name, path in review_csvs.items():
        rows = _read_csv(Path(path))
        manual_filled = 0
        for row in rows:
            transition = _transition_from_row(row)
            if not transition:
                continue
            by_transition[transition]["review_rows_total"] += 1
            by_transition[transition][f"review_rows_{name}"] += 1
            if _manual_filled(row):
                by_transition[transition]["review_manual_filled_total"] += 1
                manual_filled += 1
        source_totals[name] = {"rows": int(len(rows)), "manual_filled": int(manual_filled)}
    return by_transition, source_totals


def _transition_payload(
    transition: str,
    *,
    budget: Mapping[str, object],
    external_counts: Mapping[str, int],
    train_oof_counts: Mapping[str, int],
    review_counts: Mapping[str, int],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "transition": transition,
        "val_error_count": int(_safe_int(budget.get("count"), 0)),
        "oracle_class1_delta_if_all_corrected": str(budget.get("class1_delta", "")),
        "oracle_class1_f1_if_all_corrected": str(budget.get("class1_f1", "")),
    }
    for key in (
        "val_any_external_correct",
        "val_both_external_correct",
        "val_consensus_correct",
        "val_consensus_wrong_change",
    ):
        payload[key] = int(external_counts.get(key, 0))
    payload["train_oof_rows"] = int(train_oof_counts.get("train_oof_rows", 0))
    payload["train_oof_both_correct_minconf_0p7"] = int(
        train_oof_counts.get("train_oof_both_correct_minconf_0p7", 0)
    )
    payload["train_oof_same1_minconf_0p7"] = int(train_oof_counts.get("train_oof_same1_minconf_0p7", 0))
    payload["train_oof_samenon1_minconf_0p7"] = int(
        train_oof_counts.get("train_oof_samenon1_minconf_0p7", 0)
    )
    payload["review_rows_total"] = int(review_counts.get("review_rows_total", 0))
    payload["review_manual_filled_total"] = int(review_counts.get("review_manual_filled_total", 0))
    for key, value in sorted(review_counts.items()):
        if key.startswith("review_rows_") and key != "review_rows_total":
            payload[key] = int(value)
    return payload


def _parse_review_csv_specs(specs: Optional[Sequence[str]]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for spec in specs or []:
        if "=" not in str(spec):
            raise ValueError(f"Review CSV spec must be NAME=PATH: {spec!r}")
        name, path = str(spec).split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Review CSV spec has empty name: {spec!r}")
        output[name] = Path(path.strip())
    return output


def build_trkh_signal_gap_readiness(
    *,
    softboost_val: Path,
    transition_budget: Path,
    milestone_budget: Path,
    external_support_summary: Path,
    external_remaining_support: Path,
    train_oof_external_support: Path,
    train_oof_external_summary: Optional[Path],
    review_readiness: Path,
    oof_neighbor_policy: Path,
    review_csvs: Mapping[str, Path],
    output_dir: Path,
    created_at: Optional[str] = None,
    focus_class_index: int = 1,
    next_class1_milestone: float = 0.75,
    min_true_class1_transition_consensus: int = 2,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    softboost_rows = _read_csv(Path(softboost_val))
    transition_rows = _read_csv(Path(transition_budget))
    milestone_rows = _read_csv(Path(milestone_budget))
    external_summary = _load_json(Path(external_support_summary))
    external_rows = _read_csv(Path(external_remaining_support))
    train_oof_rows = _read_csv(Path(train_oof_external_support))
    train_oof_summary = _load_json(Path(train_oof_external_summary)) if train_oof_external_summary else {}
    readiness_summary = _load_json(Path(review_readiness))
    oof_policy_summary = _load_json(Path(oof_neighbor_policy))

    metrics = _classification_metrics_from_rows(softboost_rows, num_classes=5)
    current_focus = _focus_class_metrics(metrics, int(focus_class_index))
    budgets = _budget_by_transition(transition_rows)
    external_counts = _external_transition_counts(external_rows)
    train_counts, train_bucket_counts = _train_oof_transition_counts(train_oof_rows)
    review_transition_counts, review_source_totals = _review_counts(review_csvs)

    fn_pairs: dict[str, dict[str, object]] = {}
    fp_pairs: dict[str, dict[str, object]] = {}
    for transition, budget in sorted(budgets.items()):
        left, right = [int(part) for part in transition.split("->", 1)]
        payload = _transition_payload(
            transition,
            budget=budget,
            external_counts=external_counts.get(transition, {}),
            train_oof_counts=train_counts.get(transition, {}),
            review_counts=review_transition_counts.get(transition, {}),
        )
        if left == int(focus_class_index) and right != int(focus_class_index):
            fn_pairs[transition] = payload
        elif left != int(focus_class_index) and right == int(focus_class_index):
            fp_pairs[transition] = payload

    critical_transitions = [
        transition
        for transition, payload in fn_pairs.items()
        if int(payload.get("val_error_count", 0)) > 0
        and transition in {f"{focus_class_index}->2", f"{focus_class_index}->4"}
    ]
    insufficient_transitions = [
        transition
        for transition in critical_transitions
        if int(fn_pairs[transition].get("val_consensus_correct", 0)) < int(min_true_class1_transition_consensus)
    ]
    unreviewed_critical_transitions = [
        transition
        for transition in critical_transitions
        if int(fn_pairs[transition].get("review_manual_filled_total", 0)) == 0
    ]

    review_ready_count = _safe_int(readiness_summary.get("ready_count"), 0)
    strict_rows = _safe_int(oof_policy_summary.get("strict_rows"), 0)
    external_consensus = _external_consensus_summary(external_summary, focus_class_index=int(focus_class_index))
    focus_consensus = _focus_consensus_summary(external_summary, focus_class_index=int(focus_class_index))

    blockers: list[str] = []
    if review_ready_count <= 0:
        blockers.append("all_current_review_queues_blocked_by_empty_manual_fields")
    if strict_rows <= 0:
        blockers.append("patch_oof_cleanlab_multibank_strict_rows_zero")
    fn_rescue_bucket = train_bucket_counts.get("base_fn1_rescue_candidate", Counter())
    if int(fn_rescue_bucket.get("both_true_minconf_0.7", 0)) < int(min_true_class1_transition_consensus):
        blockers.append("train_oof_external_fn1_rescue_support_sparse")
    for transition in insufficient_transitions:
        if transition == f"{focus_class_index}->2":
            blockers.append("val_external_consensus_covers_zero_1_to_2_errors")
        elif transition == f"{focus_class_index}->4":
            blockers.append("val_external_consensus_covers_only_one_or_less_1_to_4_errors")
        else:
            blockers.append(f"val_external_consensus_insufficient:{transition}")
    if float(external_consensus.get("class1_f1", 0.0)) < float(next_class1_milestone):
        blockers.append("external_consensus_upper_bound_below_next_0p75_class1_milestone")
    if unreviewed_critical_transitions:
        blockers.append("critical_true_class1_review_rows_unfilled")

    smoke_gate_ready = not blockers
    inputs = {
        "softboost_val": str(Path(softboost_val).resolve()),
        "transition_budget": str(Path(transition_budget).resolve()),
        "milestone_budget": str(Path(milestone_budget).resolve()),
        "external_remaining_support": str(Path(external_remaining_support).resolve()),
        "external_support_summary": str(Path(external_support_summary).resolve()),
        "train_oof_external_support": str(Path(train_oof_external_support).resolve()),
        "train_oof_external_summary": str(Path(train_oof_external_summary).resolve())
        if train_oof_external_summary
        else "",
        "review_readiness": str(Path(review_readiness).resolve()),
        "oof_neighbor_policy": str(Path(oof_neighbor_policy).resolve()),
    }
    summary = {
        "mode": "trkh_signal_gap_readiness",
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "note": (
            "No-test/no-train diagnostic. It sizes whether existing fold-safe/manual/external "
            "support sources justify another TRKH smoke. It writes no trainable manifests and "
            "does not touch raw data."
        ),
        "inputs": inputs,
        "review_csvs": {name: str(Path(path).resolve()) for name, path in sorted(review_csvs.items())},
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "current_softboost_class1": current_focus,
        "external_consensus_upper_bound": external_consensus,
        "focus_consensus_upper_bound": focus_consensus,
        "milestone_budget": milestone_rows,
        "transition_focus_gaps": {
            "class1_false_negative_pairs": fn_pairs,
            "class1_false_positive_pairs": fp_pairs,
        },
        "train_oof_external_support_by_bucket": {
            name: dict(counts) for name, counts in sorted(train_bucket_counts.items())
        },
        "train_oof_external_summary": train_oof_summary,
        "review_readiness": readiness_summary,
        "review_source_totals": review_source_totals,
        "oof_neighbor_policy": {
            "candidate_rows": _safe_int(oof_policy_summary.get("candidate_rows"), 0),
            "strict_rows": strict_rows,
            "issue_pairs": oof_policy_summary.get("issue_pairs", {}),
            "strict_pairs": oof_policy_summary.get("strict_pairs", {}),
        },
        "critical_true_class1_transitions": critical_transitions,
        "unreviewed_critical_true_class1_transitions": unreviewed_critical_transitions,
        "smoke_gate_ready": bool(smoke_gate_ready),
        "blocking_reasons": blockers,
        "decision": (
            "Do not launch an automatic TRKH smoke from existing external/review/OOF sources."
            if not smoke_gate_ready
            else "Existing source summaries clear this diagnostic gate; run the explicit smoke gate auditor next."
        ),
        "next_valid_signal_requirement": (
            "Need a new fold-safe/manual reliability source or representation target that covers "
            "1->2 and 1->4 recall while suppressing 0/2/4->1 false positives; validation/test "
            "thresholds are not acceptable as train signals."
        ),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rows_path = output_dir / "transition_gap_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "transition",
            "side",
            "val_error_count",
            "val_consensus_correct",
            "train_oof_rows",
            "review_rows_total",
            "review_manual_filled_total",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for side, items in (("fn", fn_pairs), ("fp", fp_pairs)):
            for transition, payload in sorted(items.items()):
                writer.writerow(
                    {
                        "transition": transition,
                        "side": side,
                        "val_error_count": payload.get("val_error_count", 0),
                        "val_consensus_correct": payload.get("val_consensus_correct", 0),
                        "train_oof_rows": payload.get("train_oof_rows", 0),
                        "review_rows_total": payload.get("review_rows_total", 0),
                        "review_manual_filled_total": payload.get("review_manual_filled_total", 0),
                    }
                )
    blocker_text = ", ".join(blockers) if blockers else "none"
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# TRKH Signal Gap Readiness",
                "",
                str(summary["note"]),
                "",
                f"Smoke gate ready: `{str(smoke_gate_ready).lower()}`",
                f"Blocking reasons: `{blocker_text}`",
                f"Critical true-class1 transitions: `{', '.join(critical_transitions)}`",
                f"Unreviewed critical transitions: `{', '.join(unreviewed_critical_transitions)}`",
                "",
                str(summary["decision"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current TRKH signal-gap readiness crosswalk.")
    parser.add_argument("--softboost-val", type=Path, required=True)
    parser.add_argument("--transition-budget", type=Path, required=True)
    parser.add_argument("--milestone-budget", type=Path, required=True)
    parser.add_argument("--external-support-summary", type=Path, required=True)
    parser.add_argument("--external-remaining-support", type=Path, required=True)
    parser.add_argument("--train-oof-external-support", type=Path, required=True)
    parser.add_argument("--train-oof-external-summary", type=Path, default=None)
    parser.add_argument("--review-readiness", type=Path, required=True)
    parser.add_argument("--oof-neighbor-policy", type=Path, required=True)
    parser.add_argument("--review-csv", action="append", default=None, help="Repeat as NAME=PATH.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", type=str, default="")
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--next-class1-milestone", type=float, default=0.75)
    parser.add_argument("--min-true-class1-transition-consensus", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_trkh_signal_gap_readiness(
        softboost_val=args.softboost_val,
        transition_budget=args.transition_budget,
        milestone_budget=args.milestone_budget,
        external_support_summary=args.external_support_summary,
        external_remaining_support=args.external_remaining_support,
        train_oof_external_support=args.train_oof_external_support,
        train_oof_external_summary=args.train_oof_external_summary,
        review_readiness=args.review_readiness,
        oof_neighbor_policy=args.oof_neighbor_policy,
        review_csvs=_parse_review_csv_specs(args.review_csv),
        output_dir=args.output_dir,
        created_at=args.created_at or None,
        focus_class_index=args.focus_class_index,
        next_class1_milestone=args.next_class1_milestone,
        min_true_class1_transition_consensus=args.min_true_class1_transition_consensus,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

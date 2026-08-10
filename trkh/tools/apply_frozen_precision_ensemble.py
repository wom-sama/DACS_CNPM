from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from trkh.tools.audit_precision_ensemble_readiness import (
    _aligned_arrays,
    _metric_summary,
    _prediction_rows,
    _sha256,
    _transition_audit,
    _validate_prediction_row_split,
    _write_csv,
    apply_precision_ensemble,
)
from trkh.tools.soft_ensemble_predictions import _read_table, _source_path


def _load_frozen_protocol(summary_path: Path) -> Dict[str, object]:
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Readiness summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "validation_only_precision_ensemble_readiness":
        raise ValueError(f"Unexpected readiness mode: {summary.get('mode')!r}")
    if not bool(summary.get("all_gates_passed", False)):
        raise ValueError("Frozen readiness summary did not pass all gates.")
    gates = summary.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(bool(value) for value in gates.values()):
        raise ValueError("Frozen readiness summary has missing or failed gates.")
    inputs = summary.get("inputs")
    if not isinstance(inputs, Mapping) or bool(inputs.get("test_data_used", True)):
        raise ValueError("Frozen readiness summary is not validation-only.")
    for name in ("keeper", "candidate"):
        path_value = inputs.get(f"{name}_csv")
        hash_value = str(inputs.get(f"{name}_sha256", "") or "").lower()
        if not path_value or not hash_value:
            raise ValueError(f"Frozen summary is missing {name} validation provenance.")
        path = Path(str(path_value))
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {name} validation input no longer exists: {path}")
        actual_hash = _sha256(path).lower()
        if actual_hash != hash_value:
            raise ValueError(
                f"Frozen {name} validation input hash changed: expected={hash_value}, "
                f"actual={actual_hash}"
            )
    locked = summary.get("locked")
    if not isinstance(locked, Mapping):
        raise ValueError("Frozen summary has no locked parameters.")
    candidate_weight = float(locked.get("candidate_weight", -1.0))
    focus_margin_offset = float(locked.get("focus_margin_offset", -1.0))
    if (
        not math.isfinite(candidate_weight)
        or not math.isfinite(focus_margin_offset)
        or not 0.0 <= candidate_weight <= 1.0
        or focus_margin_offset < 0.0
    ):
        raise ValueError("Frozen summary contains invalid locked parameters.")
    return summary


def _verify_frozen_summary_hash(
    summary_path: Path,
    expected_sha256: Optional[str],
    *,
    required: bool,
) -> str:
    expected = str(expected_sha256 or "").strip().lower()
    if not expected:
        if required:
            raise ValueError(
                "Test application requires --expected-summary-sha256 from the locked "
                "validation protocol."
            )
    elif re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("expected_summary_sha256 must be exactly 64 hexadecimal characters.")
    actual = _sha256(Path(summary_path)).lower()
    if expected and actual != expected:
        raise ValueError(
            f"Frozen summary hash mismatch: expected={expected}, actual={actual}"
        )
    return actual


def _transition_rows(
    *,
    keys: Sequence[str],
    base_rows: Sequence[Mapping[str, str]],
    class_names: Sequence[str],
    targets: np.ndarray,
    keeper_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    frozen_predictions: np.ndarray,
    focus_margins: np.ndarray,
    focus_class: int,
) -> List[Dict[str, object]]:
    _summary, categories = _transition_audit(
        targets,
        keeper_predictions,
        frozen_predictions,
        focus_class=focus_class,
    )
    rows: List[Dict[str, object]] = []
    for index, key in enumerate(keys):
        category = str(categories[index])
        target = int(targets[index])
        frozen_prediction = int(frozen_predictions[index])
        if category == "unchanged" and not (
            target == int(focus_class) or frozen_prediction == int(focus_class)
        ):
            continue
        rows.append(
            {
                "sample_index": key,
                "image_path": _source_path(base_rows[index]),
                "target_index": target,
                "target_name": class_names[target],
                "keeper_prediction_index": int(keeper_predictions[index]),
                "candidate_prediction_index": int(candidate_predictions[index]),
                "frozen_prediction_index": frozen_prediction,
                "transition": category,
                "focus_decision_margin": float(focus_margins[index]),
            }
        )
    return rows


def apply_frozen_precision_ensemble(
    *,
    frozen_summary: Path,
    keeper_csv: Path,
    candidate_csv: Path,
    output_dir: Path,
    split: str,
    expected_rows: int,
    key_column: str = "sample_index",
    expected_summary_sha256: Optional[str] = None,
) -> Dict[str, object]:
    split = str(split).strip().lower()
    if split not in {"val", "validation", "test"}:
        raise ValueError(f"split must be val or test, got {split!r}")
    frozen_summary = Path(frozen_summary)
    frozen_summary_sha256 = _verify_frozen_summary_hash(
        frozen_summary,
        expected_summary_sha256,
        required=split == "test",
    )
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Frozen apply output already exists and is nonempty: {output_dir}"
        )
    protocol = _load_frozen_protocol(frozen_summary)
    keeper = _read_table("keeper", Path(keeper_csv), key_column)
    candidate = _read_table("candidate", Path(candidate_csv), key_column)
    keys, targets, groups, keeper_probs, candidate_probs, base_rows = _aligned_arrays(
        keeper,
        candidate,
    )
    _validate_prediction_row_split(base_rows, expected_split=split)
    if int(expected_rows) > 0 and len(keys) != int(expected_rows):
        raise ValueError(f"Expected {expected_rows} rows, found {len(keys)}")
    class_names = list(keeper.class_names)
    protocol_class_names = list(protocol.get("class_names", []))
    if class_names != protocol_class_names:
        raise ValueError(
            f"Class order differs from frozen protocol: {class_names} vs {protocol_class_names}"
        )
    focus_class = int(protocol["focus_class"])
    locked = protocol["locked"]
    assert isinstance(locked, Mapping)
    candidate_weight = float(locked["candidate_weight"])
    focus_margin_offset = float(locked["focus_margin_offset"])
    probabilities, predictions, focus_margins = apply_precision_ensemble(
        keeper_probs,
        candidate_probs,
        candidate_weight=candidate_weight,
        focus_margin_offset=focus_margin_offset,
        focus_class=focus_class,
    )
    keeper_predictions = keeper_probs.argmax(axis=1)
    candidate_predictions = candidate_probs.argmax(axis=1)
    metrics = _metric_summary(targets, predictions, class_names, focus_class)
    keeper_metrics = _metric_summary(
        targets,
        keeper_predictions,
        class_names,
        focus_class,
    )
    candidate_metrics = _metric_summary(
        targets,
        candidate_predictions,
        class_names,
        focus_class,
    )
    vs_keeper, _ = _transition_audit(
        targets,
        keeper_predictions,
        predictions,
        focus_class=focus_class,
    )
    vs_candidate, _ = _transition_audit(
        targets,
        candidate_predictions,
        predictions,
        focus_class=focus_class,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = _prediction_rows(
        keys=keys,
        base_rows=base_rows,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        focus_margins=focus_margins,
        class_names=class_names,
    )
    predictions_path = output_dir / "predictions_detailed.csv"
    _write_csv(predictions_path, prediction_rows)
    transition_rows = _transition_rows(
        keys=keys,
        base_rows=base_rows,
        class_names=class_names,
        targets=targets,
        keeper_predictions=keeper_predictions,
        candidate_predictions=candidate_predictions,
        frozen_predictions=predictions,
        focus_margins=focus_margins,
        focus_class=focus_class,
    )
    transitions_path = output_dir / "transition_rows.csv"
    _write_csv(transitions_path, transition_rows)
    summary: Dict[str, object] = {
        "mode": "frozen_precision_ensemble_apply",
        "split": "val" if split in {"val", "validation"} else "test",
        "selection_parameters_frozen": True,
        "test_data_used_for_tuning": False,
        "frozen_protocol": {
            "summary": str(frozen_summary.resolve()),
            "summary_sha256": frozen_summary_sha256,
            "candidate_weight": candidate_weight,
            "keeper_weight": float(1.0 - candidate_weight),
            "focus_margin_offset": focus_margin_offset,
            "focus_class": focus_class,
        },
        "inputs": {
            "keeper_csv": str(Path(keeper_csv).resolve()),
            "keeper_sha256": _sha256(Path(keeper_csv)),
            "candidate_csv": str(Path(candidate_csv).resolve()),
            "candidate_sha256": _sha256(Path(candidate_csv)),
        },
        "rows": int(len(keys)),
        "source_groups": int(len(set(groups.tolist()))),
        "class_names": class_names,
        "metrics": metrics,
        "keeper_metrics": keeper_metrics,
        "candidate_metrics": candidate_metrics,
        "vs_keeper": vs_keeper,
        "vs_candidate": vs_candidate,
        "artifacts": {
            "predictions": "predictions_detailed.csv",
            "predictions_sha256": _sha256(predictions_path),
            "transitions": "transition_rows.csv",
            "transitions_sha256": _sha256(transitions_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply validation-frozen precision-ensemble parameters without any search or tuning."
        )
    )
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--key-column", default="sample_index")
    parser.add_argument(
        "--expected-summary-sha256",
        help="Required for --split test; SHA-256 of the locked validation summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = apply_frozen_precision_ensemble(
        frozen_summary=args.frozen_summary,
        keeper_csv=args.keeper,
        candidate_csv=args.candidate,
        output_dir=args.output_dir,
        split=args.split,
        expected_rows=args.expected_rows,
        key_column=args.key_column,
        expected_summary_sha256=args.expected_summary_sha256,
    )
    focus = summary["metrics"]["focus"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "split": summary["split"],
                "rows": summary["rows"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "focus_precision": focus["precision"],
                "focus_recall": focus["recall"],
                "focus_f1": focus["f1"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

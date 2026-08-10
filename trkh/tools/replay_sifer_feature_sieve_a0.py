from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from trkh.tools.audit_sifer_feature_sieve_a0 import (
    EXPECTED_FORGET_STEPS,
    EXPECTED_FOLD_COUNTS,
    EXPECTED_TRAIN_BATCHES,
    _mean_probability_difference,
    assess_a0,
)
from trkh.tools.audit_xca_dual_axis_readiness import _comparison, _sha256


DEFAULT_REPLAY_ABS_TOLERANCE = 1e-12
STEM_AGGREGATE_ABS_TOLERANCE = 5e-10


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent CSV/history replay for the locked SIFER A0 audit."
    )
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--illumination", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", type=str, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _variant_rows(
    rows: Sequence[Mapping[str, str]], variant: str
) -> list[Dict[str, object]]:
    result = []
    for row in rows:
        result.append(
            {
                "sample_index": int(row["sample_index"]),
                "target": int(row["target"]),
                "prediction": int(row[f"{variant}_prediction"]),
                **{
                    f"prob_{index}": float(row[f"{variant}_prob_{index}"])
                    for index in range(5)
                },
            }
        )
    return result


def _assert_close(observed: object, expected: object, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise AssertionError(f"{path}: expected mapping, found {type(observed)}")
        if set(observed) != set(expected):
            raise AssertionError(
                f"{path}: key mismatch {sorted(observed)} != {sorted(expected)}"
            )
        for key in expected:
            _assert_close(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"{path}: list shape mismatch")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise AssertionError(f"{path}: {observed!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)):
        tolerance = (
            STEM_AGGREGATE_ABS_TOLERANCE
            if path == "mechanism.stem_mean_absolute_difference"
            else DEFAULT_REPLAY_ABS_TOLERANCE
        )
        if not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
        ):
            raise AssertionError(f"{path}: {observed!r} != {expected!r}")
        return
    if observed != expected:
        raise AssertionError(f"{path}: {observed!r} != {expected!r}")


def replay(args: argparse.Namespace) -> Dict[str, object]:
    paths = {
        "clean": Path(args.clean).resolve(),
        "illumination": Path(args.illumination).resolve(),
        "history": Path(args.history).resolve(),
        "contact_manifest": Path(args.contact_manifest).resolve(),
        "summary": Path(args.summary).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact is missing: {path}")
    summary_sha = _sha256(paths["summary"])
    expected_summary_sha = str(args.expected_summary_sha256).strip().casefold()
    if summary_sha != expected_summary_sha:
        raise ValueError(
            f"Summary SHA-256 mismatch: {summary_sha} != {expected_summary_sha}"
        )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError("Replay requires the immutable pre-review SIFER summary.")
    artifact_bindings = {
        "clean": "clean_predictions",
        "illumination": "illumination_predictions",
        "history": "training_history",
        "contact_manifest": "contact_sheet_manifest",
    }
    for input_name, artifact_name in artifact_bindings.items():
        artifact_path = Path(str(summary["artifacts"][artifact_name]["path"])).resolve()
        if paths[input_name] != artifact_path:
            raise ValueError(
                f"Replay input {input_name!r} is not the summary-bound artifact: "
                f"{paths[input_name]} != {artifact_path}"
            )

    clean_csv = _read_csv(paths["clean"])
    if len(clean_csv) != EXPECTED_FOLD_COUNTS[0]:
        raise ValueError(
            f"Clean replay expected {EXPECTED_FOLD_COUNTS[0]} rows, found {len(clean_csv)}."
        )
    sample_indices = [int(row["sample_index"]) for row in clean_csv]
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("Clean replay contains duplicate sample indices.")
    control_clean = _variant_rows(clean_csv, "control")
    candidate_clean = _variant_rows(clean_csv, "candidate")
    replayed_clean = _comparison(
        control_rows=control_clean,
        candidate_rows=candidate_clean,
        num_classes=5,
        focus_class=1,
    )
    _assert_close(replayed_clean, summary["clean"], "clean")

    illumination_csv = _read_csv(paths["illumination"])
    expected_illumination_rows = 3 * EXPECTED_FOLD_COUNTS[0]
    if len(illumination_csv) != expected_illumination_rows:
        raise ValueError(
            "Illumination replay row count differs: "
            f"{len(illumination_csv)} != {expected_illumination_rows}"
        )
    replayed_illumination = []
    for summary_condition in summary["illumination"]:
        condition = str(summary_condition["condition"])
        condition_csv = [
            row for row in illumination_csv if str(row["condition"]) == condition
        ]
        if len(condition_csv) != EXPECTED_FOLD_COUNTS[0]:
            raise ValueError(f"Condition {condition!r} has an invalid row count.")
        comparison = _comparison(
            control_rows=_variant_rows(condition_csv, "control"),
            candidate_rows=_variant_rows(condition_csv, "candidate"),
            num_classes=5,
            focus_class=1,
        )
        comparison["condition"] = condition
        comparison["brightness"] = float(summary_condition["brightness"])
        comparison["contrast"] = float(summary_condition["contrast"])
        replayed_illumination.append(comparison)
    _assert_close(replayed_illumination, summary["illumination"], "illumination")

    history = json.loads(paths["history"].read_text(encoding="utf-8"))
    control_history = history.get("control", [])
    candidate_history = history.get("candidate", [])
    if (
        len(control_history) != EXPECTED_TRAIN_BATCHES
        or len(candidate_history) != EXPECTED_TRAIN_BATCHES
    ):
        raise ValueError("Replay training history does not contain exactly 60 steps.")
    forget_rows = [
        row["forget"] for row in candidate_history if row.get("forget") is not None
    ]
    forget_steps = tuple(int(row["step"]) for row in forget_rows)
    if forget_steps != EXPECTED_FORGET_STEPS:
        raise ValueError(f"Forget sequence differs: {forget_steps}")
    identify_decrease_count = sum(
        bool(candidate_history[step]["identify"]["loss_decreased"])
        for step in EXPECTED_FORGET_STEPS
    )
    uniform_nonincrease_count = sum(
        bool(row["uniform_ce_nonincreasing"]) for row in forget_rows
    )
    entropy_nondecrease_count = sum(
        bool(row["entropy_nondecreasing"]) for row in forget_rows
    )
    stem_difference = float(
        np.mean(
            [float(row["stem_mean_absolute_difference"]) for row in clean_csv]
        )
    )
    probability_difference = _mean_probability_difference(
        control_clean, candidate_clean
    )
    mechanism = dict(summary["mechanism"])
    replayed_mechanism = {
        **mechanism,
        "identify_steps": len(candidate_history),
        "forget_steps": list(forget_steps),
        "identify_decrease_count": identify_decrease_count,
        "uniform_ce_nonincrease_count": uniform_nonincrease_count,
        "entropy_nondecrease_count": entropy_nondecrease_count,
        "stem_mean_absolute_difference": stem_difference,
        "probability_mean_absolute_difference": probability_difference,
        "forget_uniform_ce_deltas": [
            float(row["after"]["uniform_ce"])
            - float(row["before"]["uniform_ce"])
            for row in forget_rows
        ],
        "forget_entropy_deltas": [
            float(row["after"]["entropy_mean"])
            - float(row["before"]["entropy_mean"])
            for row in forget_rows
        ],
    }
    _assert_close(replayed_mechanism, summary["mechanism"], "mechanism")

    contact_manifest = json.loads(
        paths["contact_manifest"].read_text(encoding="utf-8")
    )
    if int(contact_manifest.get("sheet_count", 0)) != 4:
        raise ValueError("Contact manifest does not contain four sheets.")
    for sheet in contact_manifest["sheets"]:
        path = Path(str(sheet["path"]))
        if not path.is_file() or _sha256(path) != str(sheet["sha256"]):
            raise ValueError(f"Contact sheet hash differs: {path}")

    structural = summary["gate"]["groups"]["structural"]
    replayed_gate = assess_a0(
        structural_checks=structural,
        mechanism=replayed_mechanism,
        clean=replayed_clean,
        illumination=replayed_illumination,
    )
    if replayed_gate["automatic_failed_checks"] != summary["gate"][
        "automatic_failed_checks"
    ]:
        raise AssertionError("Replayed automatic failed-check set differs.")
    artifact_checks = {}
    for name, artifact in summary["artifacts"].items():
        artifact_path = Path(str(artifact["path"]))
        artifact_checks[name] = bool(
            artifact_path.is_file()
            and _sha256(artifact_path) == str(artifact["sha256"])
        )
    checks = {
        "summary_hash_exact": True,
        "clean_row_count_exact": len(clean_csv) == EXPECTED_FOLD_COUNTS[0],
        "illumination_row_count_exact": len(illumination_csv)
        == expected_illumination_rows,
        "clean_metrics_exact": True,
        "illumination_metrics_exact": True,
        "mechanism_exact": True,
        "forget_sequence_exact": forget_steps == EXPECTED_FORGET_STEPS,
        "contact_sheets_verified": True,
        "automatic_gate_exact": True,
        "all_summary_artifacts_verified": all(artifact_checks.values()),
    }
    result = {
        "method": "sifer_feature_sieve_a0_independent_replay",
        "passed": all(checks.values()),
        "checks": checks,
        "artifact_checks": artifact_checks,
        "summary_sha256": summary_sha,
        "clean_rows": len(clean_csv),
        "illumination_rows": len(illumination_csv),
        "forget_steps": list(forget_steps),
        "automatic_failed_checks": replayed_gate["automatic_failed_checks"],
    }
    output = (
        Path(args.output).resolve()
        if args.output is not None
        else paths["summary"].parent / "independent_replay.json"
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    result["output"] = str(output)
    result["output_sha256"] = _sha256(output)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    result = replay(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result["passed"]):
        raise SystemExit("SIFER independent replay checks failed.")


if __name__ == "__main__":
    main()

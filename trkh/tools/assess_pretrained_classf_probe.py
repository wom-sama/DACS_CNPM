from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


RESTRICTED_FALSE_POSITIVE_SOURCES = (0, 2, 4)
SUPPORTED_PROTOCOLS = ("b10", "hybrid-v2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_val_only_path(path: Path) -> None:
    forbidden = {
        "test",
        "test_only",
        "test_split",
        "eval_test",
        "final_test",
    }
    offending = [part for part in path.parts if part.strip().lower() in forbidden]
    if offending:
        raise ValueError(
            f"Probe assessment accepts validation artifacts only; got {offending}."
        )


def _finite_float(value: object, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return resolved


def load_val_metrics(path: Path) -> Dict[str, object]:
    path = Path(path).resolve()
    _assert_val_only_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Metrics must contain a JSON object: {path}.")
    per_class = payload.get("per_class")
    confusion = payload.get("confusion_matrix")
    if not isinstance(per_class, list) or not isinstance(confusion, list):
        raise ValueError(f"Missing per_class/confusion_matrix in {path}.")
    class1 = next(
        (
            row
            for row in per_class
            if isinstance(row, Mapping) and int(row.get("class_index", -1)) == 1
        ),
        None,
    )
    if not isinstance(class1, Mapping):
        raise ValueError(f"Class-1 metrics are missing from {path}.")
    if len(confusion) != 5 or any(
        not isinstance(row, list) or len(row) != 5 for row in confusion
    ):
        raise ValueError(f"Expected a 5x5 confusion matrix in {path}.")
    matrix = [[int(value) for value in row] for row in confusion]
    if any(value < 0 for row in matrix for value in row):
        raise ValueError(f"Confusion counts must be non-negative in {path}.")
    restricted_fp = int(
        sum(matrix[source][1] for source in RESTRICTED_FALSE_POSITIVE_SOURCES)
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "macro_f1": _finite_float(payload.get("macro_f1"), name="macro_f1"),
        "class1_precision": _finite_float(
            class1.get("precision"), name="class1_precision"
        ),
        "class1_recall": _finite_float(
            class1.get("recall"), name="class1_recall"
        ),
        "class1_f1": _finite_float(class1.get("f1"), name="class1_f1"),
        "restricted_0_2_4_to_1_fp": restricted_fp,
        "class0_to_1_fp": int(matrix[0][1]),
        "class2_to_1_fp": int(matrix[2][1]),
        "class1_true_positive": int(matrix[1][1]),
        "confusion_matrix": matrix,
    }


def _relative_reduction(candidate: int, control: int) -> float:
    if control == 0:
        return 1.0 if candidate == 0 else -1.0
    return 1.0 - float(candidate) / float(control)


def assess_b10(
    candidate: Mapping[str, object],
    control: Mapping[str, object],
) -> Dict[str, object]:
    deltas = {
        "macro_f1": float(candidate["macro_f1"]) - float(control["macro_f1"]),
        "class1_f1": float(candidate["class1_f1"]) - float(control["class1_f1"]),
        "class2_to_1_fp": int(candidate["class2_to_1_fp"])
        - int(control["class2_to_1_fp"]),
    }
    class2_fp_reduction = _relative_reduction(
        int(candidate["class2_to_1_fp"]),
        int(control["class2_to_1_fp"]),
    )
    gates = {
        "macro_f1_nonnegative_delta": deltas["macro_f1"] >= 0.0,
        "class1_f1_delta_gte_0p005": deltas["class1_f1"] >= 0.005,
        "class2_to_1_fp_reduction_gte_0p10": class2_fp_reduction >= 0.10 - 1e-12,
    }
    return {
        "deltas": {**deltas, "class2_to_1_fp_reduction": class2_fp_reduction},
        "gates": gates,
        "metric_gate_passed": all(gates.values()),
    }


def assess_hybrid_v2(
    candidate: Mapping[str, object],
    controls: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    required_controls = {"pure_dino", "generic_adapter"}
    if set(controls) != required_controls:
        raise ValueError(
            "Hybrid V2 assessment requires exactly pure_dino and generic_adapter "
            f"controls; got {sorted(controls)}."
        )
    comparisons: Dict[str, Dict[str, float]] = {}
    gates: Dict[str, bool] = {
        "absolute_class1_f1_gte_0p72": float(candidate["class1_f1"]) >= 0.72,
    }
    for name in sorted(controls):
        control = controls[name]
        comparison = {
            "macro_f1_delta": float(candidate["macro_f1"])
            - float(control["macro_f1"]),
            "class1_f1_delta": float(candidate["class1_f1"])
            - float(control["class1_f1"]),
            "restricted_fp_reduction": _relative_reduction(
                int(candidate["restricted_0_2_4_to_1_fp"]),
                int(control["restricted_0_2_4_to_1_fp"]),
            ),
            "class1_true_positive_retention": (
                float(candidate["class1_true_positive"])
                / max(float(control["class1_true_positive"]), 1.0)
            ),
            "class0_to_1_fp_delta": float(candidate["class0_to_1_fp"])
            - float(control["class0_to_1_fp"]),
            "class2_to_1_fp_delta": float(candidate["class2_to_1_fp"])
            - float(control["class2_to_1_fp"]),
        }
        comparisons[name] = comparison
        gates[f"class1_f1_delta_gte_0p010_vs_{name}"] = (
            comparison["class1_f1_delta"] >= 0.010
        )
        gates[f"macro_f1_nonnegative_delta_vs_{name}"] = (
            comparison["macro_f1_delta"] >= 0.0
        )
        gates[f"restricted_fp_reduction_gte_0p20_vs_{name}"] = (
            comparison["restricted_fp_reduction"] >= 0.20
        )
        gates[f"class1_true_positive_retention_gte_0p97_vs_{name}"] = (
            comparison["class1_true_positive_retention"] >= 0.97
        )
        gates[f"class0_to_1_fp_not_increased_vs_{name}"] = (
            comparison["class0_to_1_fp_delta"] <= 0.0
        )
        gates[f"class2_to_1_fp_not_increased_vs_{name}"] = (
            comparison["class2_to_1_fp_delta"] <= 0.0
        )
    return {
        "comparisons": comparisons,
        "gates": gates,
        "metric_gate_passed": all(gates.values()),
    }


def _parse_control(value: str) -> Tuple[str, Path]:
    if "=" not in str(value):
        raise argparse.ArgumentTypeError("Control must use NAME=PATH.")
    name, raw_path = str(value).split("=", 1)
    name = name.strip().lower()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("Control must use non-empty NAME=PATH.")
    return name, Path(raw_path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate locked validation-only TRKH pretrained probe gates."
    )
    parser.add_argument("--protocol", choices=SUPPORTED_PROTOCOLS, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--control",
        type=_parse_control,
        action="append",
        required=True,
        help="Named validation metrics artifact as NAME=PATH; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    candidate = load_val_metrics(args.candidate)
    controls = {name: load_val_metrics(path) for name, path in args.control}
    if len(controls) != len(args.control):
        raise ValueError("Control names must be unique.")
    if args.protocol == "b10":
        if set(controls) != {"b2"}:
            raise ValueError("B10 assessment requires exactly one control named b2.")
        assessment = assess_b10(candidate, controls["b2"])
    else:
        assessment = assess_hybrid_v2(candidate, controls)
    payload = {
        "schema_version": 1,
        "protocol": args.protocol,
        "split": "val_only",
        "test_split_opened": False,
        "candidate": candidate,
        "controls": controls,
        **assessment,
        "full_train_authorized_by_this_artifact": False,
        "note": (
            "Passing these metric gates still requires mechanism, robustness, "
            "XAI and trained-device deployment audits before promotion."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()

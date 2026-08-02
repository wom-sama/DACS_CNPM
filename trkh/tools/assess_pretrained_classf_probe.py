from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


RESTRICTED_FALSE_POSITIVE_SOURCES = (0, 2, 4)
SUPPORTED_PROTOCOLS = ("b10", "hybrid-v2", "hybrid-v3")
CANONICAL_CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
CANONICAL_VAL_SUPPORTS = (558, 158, 380, 494, 889)
CANONICAL_VAL_SAMPLES = sum(CANONICAL_VAL_SUPPORTS)


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
    class_rows = sorted(
        (row for row in per_class if isinstance(row, Mapping)),
        key=lambda row: int(row.get("class_index", -1)),
    )
    routing_audit = payload.get("surface_pair_routing_audit")
    residual_ratio_p95 = None
    if isinstance(routing_audit, Mapping):
        statistics = routing_audit.get("statistics")
        if isinstance(statistics, Mapping):
            residual = statistics.get("residual_to_token_ratio")
            if isinstance(residual, Mapping) and residual.get("p95") is not None:
                residual_ratio_p95 = _finite_float(
                    residual.get("p95"),
                    name="surface_pair_residual_ratio_p95",
                )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "split": str(payload.get("split", "")).strip().lower(),
        "samples": int(payload.get("samples", -1)),
        "class_names": [str(row.get("class_name", "")) for row in class_rows],
        "supports": [int(row.get("support", -1)) for row in class_rows],
        "checkpoint": str(payload.get("checkpoint", "")),
        "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")).lower(),
        "prediction_file_sha256": str(
            payload.get("prediction_file_sha256", "")
        ).lower(),
        "sample_identity_sha256": str(
            payload.get("sample_identity_sha256", "")
        ).lower(),
        "model_type": str(payload.get("model_type", "")).strip().lower(),
        "dinov3_surface_hybrid_mode": str(
            payload.get("dinov3_surface_hybrid_mode", "")
        ).strip().lower(),
        "surface_pair_branch_off": bool(
            payload.get("surface_pair_branch_off", False)
        ),
        "surface_pair_residual_ratio_p95": residual_ratio_p95,
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


def _canonical_relative_val_path(raw_path: str) -> str:
    parts = Path(str(raw_path)).parts
    indices = [index for index, part in enumerate(parts) if part.lower() == "val"]
    if not indices:
        raise ValueError(f"Prediction row is not rooted in the val split: {raw_path}")
    return "/".join(parts[indices[-1] :])


def validate_canonical_v3_artifact(metrics: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(metrics["path"])).resolve()
    if metrics.get("split") != "val":
        raise ValueError(f"Hybrid V3 requires split=val: {path}")
    if int(metrics.get("samples", -1)) != CANONICAL_VAL_SAMPLES:
        raise ValueError(
            f"Hybrid V3 requires {CANONICAL_VAL_SAMPLES} full-val rows: {path}"
        )
    if tuple(metrics.get("class_names", [])) != CANONICAL_CLASS_NAMES:
        raise ValueError(f"Hybrid V3 class order differs from canonical class_f: {path}")
    if tuple(metrics.get("supports", [])) != CANONICAL_VAL_SUPPORTS:
        raise ValueError(f"Hybrid V3 validation supports are not canonical: {path}")
    matrix = metrics["confusion_matrix"]
    if tuple(sum(int(value) for value in row) for row in matrix) != CANONICAL_VAL_SUPPORTS:
        raise ValueError(f"Hybrid V3 confusion support disagrees with per-class support: {path}")

    prediction_path = path.with_name("predictions_val.csv")
    _assert_val_only_path(prediction_path)
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Hybrid V3 metrics require the sibling predictions_val.csv: {path}"
        )
    identity_digest = hashlib.sha256()
    observed_supports = [0] * len(CANONICAL_CLASS_NAMES)
    observed_confusion = [
        [0] * len(CANONICAL_CLASS_NAMES) for _ in CANONICAL_CLASS_NAMES
    ]
    seen_paths = set()
    row_count = 0
    with prediction_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "y_true", "y_pred", "true_name", "pred_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Prediction CSV is missing required columns: {prediction_path}")
        for row in reader:
            canonical_path = _canonical_relative_val_path(str(row["path"]))
            if canonical_path in seen_paths:
                raise ValueError(f"Duplicate validation prediction path: {canonical_path}")
            seen_paths.add(canonical_path)
            target = int(row["y_true"])
            prediction = int(row["y_pred"])
            if target not in range(5) or prediction not in range(5):
                raise ValueError(f"Prediction class index is outside [0,4]: {prediction_path}")
            if str(row["true_name"]) != CANONICAL_CLASS_NAMES[target]:
                raise ValueError(f"Prediction true_name/class index mismatch: {canonical_path}")
            if str(row["pred_name"]) != CANONICAL_CLASS_NAMES[prediction]:
                raise ValueError(f"Prediction pred_name/class index mismatch: {canonical_path}")
            observed_supports[target] += 1
            observed_confusion[target][prediction] += 1
            identity_digest.update(
                f"{canonical_path}\t{target}\n".encode("utf-8")
            )
            row_count += 1
    if row_count != CANONICAL_VAL_SAMPLES:
        raise ValueError(f"Prediction CSV is not the full canonical validation split: {prediction_path}")
    if tuple(observed_supports) != CANONICAL_VAL_SUPPORTS:
        raise ValueError(f"Prediction CSV validation supports are not canonical: {prediction_path}")
    if observed_confusion != matrix:
        raise ValueError(f"Prediction CSV and metrics confusion matrices differ: {path}")

    prediction_sha256 = _sha256(prediction_path)
    recorded_prediction_sha256 = str(metrics.get("prediction_file_sha256", ""))
    if recorded_prediction_sha256 and recorded_prediction_sha256 != prediction_sha256:
        raise ValueError(f"Recorded prediction hash differs from its CSV: {path}")
    sample_identity_sha256 = identity_digest.hexdigest()
    recorded_identity_sha256 = str(metrics.get("sample_identity_sha256", ""))
    if recorded_identity_sha256 and recorded_identity_sha256 != sample_identity_sha256:
        raise ValueError(f"Recorded validation identity hash differs from its CSV: {path}")

    checkpoint = Path(str(metrics.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Metrics checkpoint is unavailable: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    recorded_checkpoint_sha256 = str(metrics.get("checkpoint_sha256", ""))
    if recorded_checkpoint_sha256 and recorded_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(f"Recorded checkpoint hash differs from its file: {path}")
    metrics.update(
        {
            "prediction_path": str(prediction_path),
            "prediction_file_sha256": prediction_sha256,
            "sample_identity_sha256": sample_identity_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    return metrics


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


def assess_hybrid_v3(
    candidate: Mapping[str, object],
    controls: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    required_controls = {"b2", "pair_generic", "branch_off"}
    if set(controls) != required_controls:
        raise ValueError(
            "Hybrid V3 assessment requires exactly b2, pair_generic and "
            f"branch_off controls; got {sorted(controls)}."
        )
    comparisons: Dict[str, Dict[str, float]] = {}
    gates: Dict[str, bool] = {
        "absolute_class1_f1_gte_0p72": float(candidate["class1_f1"])
        >= 0.72 - 1e-12,
        "surface_pair_residual_ratio_p95_lte_0p04": (
            candidate.get("surface_pair_residual_ratio_p95") is not None
            and float(candidate["surface_pair_residual_ratio_p95"])
            <= 0.04 + 1e-12
        ),
    }
    for name in ("b2", "pair_generic"):
        control = controls[name]
        comparison = {
            "macro_f1_delta": float(candidate["macro_f1"])
            - float(control["macro_f1"]),
            "class1_f1_delta": float(candidate["class1_f1"])
            - float(control["class1_f1"]),
        }
        comparisons[name] = comparison
        gates[f"class1_f1_delta_gte_0p010_vs_{name}"] = (
            comparison["class1_f1_delta"] >= 0.010 - 1e-12
        )
        gates[f"macro_f1_delta_gte_minus_0p002_vs_{name}"] = (
            comparison["macro_f1_delta"] >= -0.002 - 1e-12
        )

    generic = controls["pair_generic"]
    generic_safety = {
        "class2_to_1_fp_reduction": _relative_reduction(
            int(candidate["class2_to_1_fp"]),
            int(generic["class2_to_1_fp"]),
        ),
        "restricted_fp_reduction": _relative_reduction(
            int(candidate["restricted_0_2_4_to_1_fp"]),
            int(generic["restricted_0_2_4_to_1_fp"]),
        ),
        "class1_true_positive_retention": (
            float(candidate["class1_true_positive"])
            / max(float(generic["class1_true_positive"]), 1.0)
        ),
        "class1_true_positive_delta": float(candidate["class1_true_positive"])
        - float(generic["class1_true_positive"]),
    }
    comparisons["pair_generic"].update(generic_safety)
    gates["class2_to_1_fp_reduction_gte_0p20_vs_pair_generic"] = (
        generic_safety["class2_to_1_fp_reduction"] >= 0.20 - 1e-12
    )
    gates["restricted_fp_reduction_gte_0p10_vs_pair_generic"] = (
        generic_safety["restricted_fp_reduction"] >= 0.10 - 1e-12
    )
    gates["class1_true_positive_retention_gte_0p98"] = (
        generic_safety["class1_true_positive_retention"] >= 0.98 - 1e-12
    )
    gates["class1_true_positive_loss_lte_2"] = (
        generic_safety["class1_true_positive_delta"] >= -2.0
    )

    branch_delta = float(candidate["class1_f1"]) - float(
        controls["branch_off"]["class1_f1"]
    )
    comparisons["branch_off"] = {"class1_f1_delta": branch_delta}
    gates["active_minus_branch_off_class1_f1_gte_0p005"] = (
        branch_delta >= 0.005 - 1e-12
    )
    return {
        "comparisons": comparisons,
        "gates": gates,
        "metric_gate_passed": all(gates.values()),
    }


def validate_hybrid_v3_comparison_contract(
    candidate: Dict[str, object],
    controls: Dict[str, Dict[str, object]],
) -> None:
    required_controls = {"b2", "pair_generic", "branch_off"}
    if set(controls) != required_controls:
        raise ValueError(
            "Hybrid V3 comparison requires exactly b2, pair_generic and branch_off."
        )
    artifacts = {"candidate": candidate, **controls}
    for artifact in artifacts.values():
        validate_canonical_v3_artifact(artifact)
    identities = {
        str(artifact["sample_identity_sha256"]) for artifact in artifacts.values()
    }
    if len(identities) != 1:
        raise ValueError("Hybrid V3 artifacts do not cover the same ordered val samples.")

    expected_roles = {
        "candidate": ("relative_surface_pair", False),
        "pair_generic": ("generic_token_pair", False),
        "branch_off": ("relative_surface_pair", True),
    }
    for name, (expected_mode, expected_off) in expected_roles.items():
        artifact = artifacts[name]
        if artifact.get("model_type") != "dinov3_surface_pair_hybrid_v3":
            raise ValueError(f"{name} is not a Surface Pair Hybrid V3 artifact.")
        if artifact.get("dinov3_surface_hybrid_mode") != expected_mode:
            raise ValueError(f"{name} has the wrong V3 evidence mode.")
        if bool(artifact.get("surface_pair_branch_off")) is not expected_off:
            raise ValueError(f"{name} has the wrong branch-off state.")
    if candidate["checkpoint_sha256"] != controls["branch_off"]["checkpoint_sha256"]:
        raise ValueError("Candidate and branch_off must use the identical checkpoint.")
    if candidate.get("surface_pair_residual_ratio_p95") is None:
        raise ValueError("Candidate is missing the full-val surface-pair routing audit.")


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
    elif args.protocol == "hybrid-v2":
        assessment = assess_hybrid_v2(candidate, controls)
    else:
        validate_hybrid_v3_comparison_contract(candidate, controls)
        assessment = assess_hybrid_v3(candidate, controls)
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

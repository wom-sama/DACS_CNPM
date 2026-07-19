from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from trkh.tools.audit_schedule_free_adamw_a0 import (
    EXPECTED_CALIBRATION_BATCHES,
    EXPECTED_TRAIN_BATCHES,
    LEARNING_RATE,
    WARMUP_STEPS,
    _equation_diagnostics,
    _focus_tp_retention,
    _json_max_abs_difference,
    _locked_args_exact,
    _ordered_index_sha256,
    _parameter_max_abs_difference,
    _prediction_artifact_integrity,
    _rows_from_csv,
    _scalar_trajectory,
    _select_xai_rows,
    _source_paths,
    _tensor_batch_sha256,
    _validate_prediction_pair,
    assess_a0,
    parse_args,
)


def _comparison_payload(
    *,
    macro_delta: float = 0.004,
    class1_f1_delta: float = 0.011,
    precision_delta: float = 0.016,
    recall_delta: float = 0.0,
    fp_reduction: int = 4,
) -> dict[str, object]:
    return {
        "control": {"per_class_f1": [0.8, 0.7, 0.8, 0.8, 0.8]},
        "candidate": {"per_class_f1": [0.8, 0.711, 0.8, 0.8, 0.8]},
        "delta": {
            "macro_f1": macro_delta,
            "class1_f1": class1_f1_delta,
            "class1_precision": precision_delta,
            "class1_recall": recall_delta,
        },
        "transitions": {
            "restricted_focus_fp_reduction": fp_reduction,
            "candidate_correction": 7,
            "candidate_harm": 3,
            "focus_fn_rescue": 2,
            "focus_tp_break": 1,
        },
    }


def test_locked_defaults_match_protocol() -> None:
    args = parse_args([])
    assert _locked_args_exact(args)
    assert args.max_train_batches == EXPECTED_TRAIN_BATCHES
    assert args.calibration_batches == EXPECTED_CALIBRATION_BATCHES
    assert args.warmup_steps == WARMUP_STEPS
    assert args.learning_rate == LEARNING_RATE


def test_scalar_trajectory_is_stable() -> None:
    result = _scalar_trajectory(
        0.6,
        [0.4, -0.2, 0.7, -0.5, 0.1],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.05,
        warmup_steps=3,
    )
    assert result["y"] == pytest.approx(0.5928316290290837, abs=1e-15)
    assert result["z"] == pytest.approx(0.5931187374636118, abs=1e-15)
    assert result["x"] == pytest.approx(0.592799728091914, abs=1e-15)
    assert [row["scheduled_lr"] for row in result["rows"]] == pytest.approx(
        [0.01 / 3.0, 0.02 / 3.0, 0.01, 0.01, 0.01]
    )


def test_official_equation_and_mode_contract_when_source_is_present() -> None:
    args = parse_args([])
    paths = _source_paths(args)
    if not paths["optimizer"].is_file() or not paths["reference"].is_file():
        pytest.skip("Locked official Schedule-Free checkout is unavailable.")
    result = _equation_diagnostics(paths)
    assert result["passed"] is True
    assert result["maximum_error"] <= 1e-12
    assert result["mode_roundtrip_y_error"] == 0.0
    assert result["mode_roundtrip_x_error"] == 0.0
    assert result["x_y_max_abs_difference"] > 0.0
    assert result["x_z_max_abs_difference"] > 0.0


def test_focus_tp_retention_uses_control_true_positives() -> None:
    control = [
        {"sample_index": 0, "target": 1, "prediction": 1},
        {"sample_index": 1, "target": 1, "prediction": 1},
        {"sample_index": 2, "target": 1, "prediction": 0},
        {"sample_index": 3, "target": 0, "prediction": 1},
    ]
    candidate = [
        {"sample_index": 0, "target": 1, "prediction": 1},
        {"sample_index": 1, "target": 1, "prediction": 0},
        {"sample_index": 2, "target": 1, "prediction": 1},
        {"sample_index": 3, "target": 0, "prediction": 0},
    ]
    result = _focus_tp_retention(control, candidate, focus_class=1)
    assert result == {"control_tp": 2, "retained_tp": 1, "ratio": 0.5}


def test_assess_a0_requires_every_clean_and_shift_gate() -> None:
    clean = _comparison_payload()
    illumination = []
    retention = []
    for condition in ("lighting_dim", "lighting_bright", "low_contrast"):
        row = _comparison_payload(
            macro_delta=0.0,
            class1_f1_delta=0.0,
            precision_delta=0.011,
            recall_delta=0.0,
            fp_reduction=1,
        )
        row["condition"] = condition
        illumination.append(row)
        retention.append({"ratio": 1.0})
    passed = assess_a0(
        structural_checks={"structure": True},
        clean=clean,
        clean_tp_retention={"ratio": 1.0},
        illumination=illumination,
        illumination_tp_retention=retention,
    )
    assert passed["automatic_gates_passed"] is True
    assert passed["xai_review_required"] is True
    assert passed["stage_b_smoke_authorized"] is False

    failed_illumination = json.loads(json.dumps(illumination))
    failed_illumination[1]["delta"]["class1_precision"] = -0.001
    failed = assess_a0(
        structural_checks={"structure": True},
        clean=clean,
        clean_tp_retention={"ratio": 1.0},
        illumination=failed_illumination,
        illumination_tp_retention=retention,
    )
    assert failed["automatic_gates_passed"] is False
    assert "lighting_bright_class1_precision_nonnegative" in failed["failed_checks"]


def test_parameter_and_json_differences_fail_closed() -> None:
    left = {"weight": torch.tensor([1.0, 2.0]), "counter": torch.tensor(1)}
    right = {"weight": torch.tensor([1.0, 2.25]), "counter": torch.tensor(2)}
    assert _parameter_max_abs_difference(left, right) == pytest.approx(0.25)
    assert _json_max_abs_difference({"a": [1.0, True]}, {"a": [1.1, True]}) == pytest.approx(0.1)
    assert _json_max_abs_difference({"a": 1}, {"b": 1}) == float("inf")


def test_rows_from_csv_reconstructs_locked_role(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    fields = [
        "condition",
        "sample_index",
        "target",
        "candidate_prediction",
        *[f"candidate_prob_{index}" for index in range(5)],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "condition": "clean",
                "sample_index": 7,
                "target": 1,
                "candidate_prediction": 1,
                "candidate_prob_0": 0.1,
                "candidate_prob_1": 0.6,
                "candidate_prob_2": 0.1,
                "candidate_prob_3": 0.1,
                "candidate_prob_4": 0.1,
            }
        )
    rows = _rows_from_csv(path, role="candidate", condition="clean")
    assert rows == [
        {
            "sample_index": 7,
            "target": 1,
            "prediction": 1,
            "prob_0": 0.1,
            "prob_1": 0.6,
            "prob_2": 0.1,
            "prob_3": 0.1,
            "prob_4": 0.1,
        }
    ]


def test_prediction_pair_alignment_fails_closed() -> None:
    left = [
        {"sample_index": 3, "target": 1},
        {"sample_index": 7, "target": 0},
    ]
    right = [dict(row) for row in left]
    _validate_prediction_pair(left, right)
    with pytest.raises(ValueError, match="order"):
        _validate_prediction_pair(left, list(reversed(right)))
    with pytest.raises(ValueError, match="duplicate"):
        _validate_prediction_pair(left, [dict(left[0]), dict(left[0])])


def test_batch_hash_covers_metadata_tensors() -> None:
    images = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    targets = torch.tensor([1])
    metadata = {
        "sample_index": torch.tensor([4]),
        "bbox": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
        "ignored_text": ["same"],
    }
    baseline = _tensor_batch_sha256(images, targets, metadata)
    changed = dict(metadata)
    changed["bbox"] = torch.tensor([[0.5, 0.5, 0.3, 0.4]])
    assert _tensor_batch_sha256(images, targets, changed) != baseline


def test_xai_selection_reserves_available_transition_categories() -> None:
    control = [
        {"sample_index": 0, "target": 0, "prediction": 1, "prob_1": 0.8},
        {"sample_index": 1, "target": 2, "prediction": 2, "prob_1": 0.1},
        {"sample_index": 2, "target": 1, "prediction": 1, "prob_1": 0.9},
        {"sample_index": 3, "target": 1, "prediction": 0, "prob_1": 0.2},
    ]
    candidate = [
        {"sample_index": 0, "target": 0, "prediction": 0, "prob_1": 0.2},
        {"sample_index": 1, "target": 2, "prediction": 1, "prob_1": 0.7},
        {"sample_index": 2, "target": 1, "prediction": 0, "prob_1": 0.3},
        {"sample_index": 3, "target": 1, "prediction": 1, "prob_1": 0.8},
    ]
    selected, categories = _select_xai_rows(
        control=control, candidate=candidate, limit=4
    )
    assert set(selected) == {0, 1, 2, 3}
    assert "restricted_fp_removal" in categories[0]
    assert "restricted_fp_creation" in categories[1]
    assert "class1_tp_break" in categories[2]
    assert "class1_fn_rescue" in categories[3]


def test_prediction_artifact_integrity_checks_all_locked_roles(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    roles = ("raw", "control", "candidate", "candidate_z")
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        *[f"{role}_prediction" for role in roles],
        *[
            f"{role}_prob_{class_index}"
            for role in roles
            for class_index in range(5)
        ],
    ]
    conditions = ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in conditions:
            for sample_index, target in ((2, 1), (5, 0)):
                row: dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": f"source_{sample_index}",
                    "image_path": f"image_{sample_index}.jpg",
                    "fold": 0,
                    "target": target,
                    "keeper_prediction": target,
                }
                for role in roles:
                    row[f"{role}_prediction"] = target
                    for class_index in range(5):
                        row[f"{role}_prob_{class_index}"] = float(
                            class_index == target
                        )
                writer.writerow(row)
    result = _prediction_artifact_integrity(
        path,
        expected_order_sha256=_ordered_index_sha256([2, 5]),
        expected_rows=2,
    )
    assert result["passed"] is True
    assert result["raw_keeper_prediction_mismatches"] == 0

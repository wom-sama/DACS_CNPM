from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trkh.tools.audit_validation_information_ceiling import (
    InputSpec,
    audit_validation_information_ceiling,
)


CLASS_NAMES = ["class_0", "class_1", "class_2"]


def _write_predictions(
    path: Path,
    targets: list[int],
    predictions: list[int],
    *,
    split: str = "val",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "source_stem",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        *[f"prob_{index}_{name}" for index, name in enumerate(CLASS_NAMES)],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (target, prediction) in enumerate(zip(targets, predictions)):
            row = {
                "sample_index": index,
                "image_path": f"D:/dataset/images/{split}/image_{index // 2}.jpg",
                "source_stem": f"image_{index // 2}",
                "target_index": target,
                "target_name": CLASS_NAMES[target],
                "prediction_index": prediction,
                "prediction_name": CLASS_NAMES[prediction],
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"prob_{class_index}_{class_name}"] = float(class_index == prediction)
            writer.writerow(row)


def test_information_ceiling_is_strict_aligned_and_deterministic(tmp_path: Path) -> None:
    targets = [0, 1, 1, 2, 0, 2]
    base_predictions = [0, 0, 1, 2, 1, 2]
    expert_predictions = [0, 1, 0, 2, 0, 1]
    base_csv = tmp_path / "base.csv"
    expert_csv = tmp_path / "expert.csv"
    _write_predictions(base_csv, targets, base_predictions)
    _write_predictions(expert_csv, targets, expert_predictions)

    kwargs = {
        "input_specs": [
            InputSpec("no_pretrain", "base", base_csv),
            InputSpec("pretrained", "expert", expert_csv),
        ],
        "base_name": "base",
        "expected_samples": 6,
        "focus_class_index": 1,
        "target_focus_f1": 0.98,
        "bootstrap_iterations": 40,
        "bootstrap_seed": 7,
        "bootstrap_mode": "source_group",
    }
    first = audit_validation_information_ceiling(output_dir=tmp_path / "first", **kwargs)
    second = audit_validation_information_ceiling(output_dir=tmp_path / "second", **kwargs)

    assert first["validation_guard"]["test_used"] is False
    assert first["model_metrics"]["base"]["macro_f1"] == pytest.approx(
        second["model_metrics"]["base"]["macro_f1"]
    )
    assert first["exact_oracle_metrics"]["all_models"]["accuracy"] == pytest.approx(1.0)
    assert first["focus_binary_ceiling"]["all_models"]["f1"] == pytest.approx(1.0)
    assert first["focus_target_budget"]["minimum_independent_corrections"][
        "minimum_total_corrections"
    ] == 2
    assert (tmp_path / "first" / "bootstrap_delta_vs_base.csv").is_file()
    assert (tmp_path / "first" / "all_models_wrong_cases.csv").is_file()
    assert (tmp_path / "first" / "focus_binary_residual_cases.csv").is_file()
    assert (tmp_path / "first" / "exact_oracle_per_class_metrics.csv").is_file()
    assert "focus_binary_oracle:all_models" in (
        tmp_path / "first" / "bootstrap_metrics.csv"
    ).read_text()
    assert (tmp_path / "first" / "bootstrap_metrics.csv").read_text() == (
        tmp_path / "second" / "bootstrap_metrics.csv"
    ).read_text()


def test_information_ceiling_rejects_target_mismatch(tmp_path: Path) -> None:
    base_csv = tmp_path / "base.csv"
    mismatch_csv = tmp_path / "mismatch.csv"
    _write_predictions(base_csv, [0, 1], [0, 0])
    _write_predictions(mismatch_csv, [0, 2], [0, 2])

    with pytest.raises(ValueError, match="Target mismatch"):
        audit_validation_information_ceiling(
            input_specs=[
                InputSpec("a", "base", base_csv),
                InputSpec("b", "mismatch", mismatch_csv),
            ],
            base_name="base",
            output_dir=tmp_path / "out",
            expected_samples=2,
            bootstrap_iterations=2,
        )


def test_information_ceiling_rejects_test_split(tmp_path: Path) -> None:
    base_csv = tmp_path / "base.csv"
    test_csv = tmp_path / "test.csv"
    _write_predictions(base_csv, [0, 1], [0, 0])
    _write_predictions(test_csv, [0, 1], [0, 1], split="test")

    with pytest.raises(ValueError, match="Validation-only guard"):
        audit_validation_information_ceiling(
            input_specs=[
                InputSpec("a", "base", base_csv),
                InputSpec("b", "test_model", test_csv),
            ],
            base_name="base",
            output_dir=tmp_path / "out",
            expected_samples=2,
            bootstrap_iterations=2,
        )

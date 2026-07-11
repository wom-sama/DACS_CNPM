import csv
from pathlib import Path

import numpy as np
import pytest

from trkh.tools.audit_adaptive_ldr_readiness import (
    adaptive_lambda_ratio,
    build_adaptive_ldr_readiness,
    target_gradient_strength,
)


def _write_predictions(path: Path, rows: list[tuple[int, int, list[float]]], *, split: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_count = len(rows[0][2])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "sample_index",
                "source_stem",
                "image_path",
                "target_index",
                "prediction_index",
                *[f"prob_{index}" for index in range(class_count)],
            ],
        )
        writer.writeheader()
        for sample_index, target, probabilities in rows:
            prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
            row = {
                "split": split,
                "sample_index": sample_index,
                "source_stem": f"image_{sample_index}",
                "image_path": f"/dataset/{split}/image_{sample_index}.jpg",
                "target_index": target,
                "prediction_index": prediction,
            }
            row.update({f"prob_{index}": value for index, value in enumerate(probabilities)})
            writer.writerow(row)
    return path


def test_adaptive_lambda_formula_and_target_gradient_are_finite() -> None:
    probabilities = np.asarray(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.99, 0.0025, 0.0025, 0.0025, 0.0025],
        ],
        dtype=np.float64,
    )
    ratios = adaptive_lambda_ratio(probabilities, lambda_ref=1.0, alpha=2.0)
    assert ratios[0] == pytest.approx(1.0)
    assert 0.5 < ratios[1] < 0.6
    gradients = target_gradient_strength(
        probabilities,
        np.asarray([0, 1]),
        ratios,
        margin=1.0,
    )
    assert np.isfinite(gradients).all()
    assert (gradients > 0.0).all()
    with pytest.raises(ValueError, match="greater than 1"):
        adaptive_lambda_ratio(probabilities, lambda_ref=1.0, alpha=1.0)


def test_adaptive_ldr_readiness_rejects_near_fixed_update_and_test_rows(tmp_path: Path) -> None:
    rows = [
        (0, 0, [0.24, 0.20, 0.19, 0.18, 0.19]),
        (1, 0, [0.20, 0.24, 0.19, 0.18, 0.19]),
        (2, 1, [0.24, 0.20, 0.19, 0.18, 0.19]),
        (3, 1, [0.20, 0.24, 0.19, 0.18, 0.19]),
        (4, 2, [0.19, 0.24, 0.20, 0.18, 0.19]),
        (5, 2, [0.19, 0.20, 0.24, 0.18, 0.19]),
        (6, 3, [0.19, 0.20, 0.18, 0.24, 0.19]),
        (7, 4, [0.19, 0.20, 0.18, 0.19, 0.24]),
    ]
    train_csv = _write_predictions(tmp_path / "train.csv", rows, split="train")
    validation_csv = _write_predictions(tmp_path / "validation.csv", rows, split="val")
    summary = build_adaptive_ldr_readiness(
        train_csv=train_csv,
        validation_csv=validation_csv,
        output_dir=tmp_path / "audit",
        bootstrap_replicates=20,
    )
    assert summary["smoke_gate_ready"] is False
    assert any("lambda_span_below_min" in reason for reason in summary["blocking_reasons"])
    assert summary["test_split_used"] is False
    assert (tmp_path / "audit" / "summary.json").is_file()
    assert (tmp_path / "audit" / "sample_adaptive_ldr_diagnostics.csv").is_file()

    test_csv = _write_predictions(tmp_path / "locked.csv", rows, split="test")
    with pytest.raises(ValueError, match="Locked test"):
        build_adaptive_ldr_readiness(
            train_csv=train_csv,
            validation_csv=test_csv,
            output_dir=tmp_path / "blocked",
            bootstrap_replicates=0,
        )

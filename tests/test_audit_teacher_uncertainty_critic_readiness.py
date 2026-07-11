import csv
from pathlib import Path

import pytest

from trkh.tools.audit_teacher_uncertainty_critic_readiness import (
    build_teacher_uncertainty_critic_readiness,
)


def _write_predictions(
    path: Path,
    rows: list[tuple[int, int, list[float]]],
    *,
    split: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_count = len(rows[0][2])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "sample_index",
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
                "image_path": f"/dataset/{split}/image_{sample_index}.jpg",
                "target_index": target,
                "prediction_index": prediction,
            }
            row.update({f"prob_{index}": value for index, value in enumerate(probabilities)})
            writer.writerow(row)
    return path


def test_uncertainty_readiness_is_sample_index_strict_and_blocks_sparse_transition(tmp_path: Path) -> None:
    base_rows = [
        (0, 0, [0.8, 0.1, 0.1, 0.0, 0.0]),
        (1, 0, [0.2, 0.7, 0.1, 0.0, 0.0]),
        (2, 1, [0.7, 0.2, 0.1, 0.0, 0.0]),
        (3, 1, [0.1, 0.8, 0.1, 0.0, 0.0]),
        (4, 2, [0.1, 0.7, 0.2, 0.0, 0.0]),
        (5, 2, [0.1, 0.2, 0.7, 0.0, 0.0]),
    ]
    expert_a = [
        (0, 0, [0.9, 0.05, 0.05, 0.0, 0.0]),
        (1, 0, [0.8, 0.1, 0.1, 0.0, 0.0]),
        (2, 1, [0.1, 0.8, 0.1, 0.0, 0.0]),
        (3, 1, [0.1, 0.8, 0.1, 0.0, 0.0]),
        (4, 2, [0.1, 0.1, 0.8, 0.0, 0.0]),
        (5, 2, [0.1, 0.1, 0.8, 0.0, 0.0]),
    ]
    expert_b = [
        (0, 0, [0.85, 0.10, 0.05, 0.0, 0.0]),
        (1, 0, [0.75, 0.15, 0.10, 0.0, 0.0]),
        (2, 1, [0.15, 0.75, 0.10, 0.0, 0.0]),
        (3, 1, [0.10, 0.80, 0.10, 0.0, 0.0]),
        (4, 2, [0.10, 0.15, 0.75, 0.0, 0.0]),
        (5, 2, [0.10, 0.10, 0.80, 0.0, 0.0]),
    ]
    base_train = _write_predictions(tmp_path / "base_train.csv", base_rows, split="train")
    base_val = _write_predictions(tmp_path / "base_val.csv", base_rows, split="val")
    train_a = _write_predictions(tmp_path / "expert_a_oof.csv", expert_a, split="train")
    train_b = _write_predictions(tmp_path / "expert_b_oof.csv", expert_b, split="train")
    val_a = _write_predictions(tmp_path / "expert_a_val.csv", expert_a, split="val")
    val_b = _write_predictions(tmp_path / "expert_b_val.csv", expert_b, split="val")

    output_dir = tmp_path / "audit"
    summary = build_teacher_uncertainty_critic_readiness(
        base_train=base_train,
        train_experts={"a": train_a, "b": train_b},
        base_val=base_val,
        val_experts={"a": val_a, "b": val_b},
        output_dir=output_dir,
        focus_class_index=1,
        min_train_fp_transition_corrections=1,
        min_train_fn_transition_corrections=1,
    )

    assert summary["test_split_used"] is False
    assert summary["raw_dataset_touched"] is False
    assert summary["trainable_manifest_written"] is False
    assert summary["smoke_gate_ready"] is False
    assert any("1->2" in reason for reason in summary["blocking_reasons"])
    assert summary["train"]["transition_support"]["0->1"]["teacher_corrections"] == 1
    assert summary["train"]["transition_support"]["1->0"]["teacher_corrections"] == 1
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "sample_uncertainty_diagnostics.csv").is_file()
    assert (output_dir / "risk_coverage.csv").is_file()
    assert (output_dir / "transition_support.csv").is_file()


def test_uncertainty_readiness_refuses_locked_test_rows(tmp_path: Path) -> None:
    rows = [
        (0, 0, [0.8, 0.1, 0.1, 0.0, 0.0]),
        (1, 1, [0.1, 0.8, 0.1, 0.0, 0.0]),
    ]
    base_train = _write_predictions(tmp_path / "base_train.csv", rows, split="train")
    train_expert = _write_predictions(tmp_path / "expert_oof.csv", rows, split="train")
    base_test = _write_predictions(tmp_path / "base_locked.csv", rows, split="test")
    val_expert = _write_predictions(tmp_path / "expert_val.csv", rows, split="val")

    with pytest.raises(ValueError, match="Locked test"):
        build_teacher_uncertainty_critic_readiness(
            base_train=base_train,
            train_experts={"expert": train_expert},
            base_val=base_test,
            val_experts={"expert": val_expert},
            output_dir=tmp_path / "audit",
            focus_class_index=1,
        )

import csv
from pathlib import Path

import numpy as np
import pytest

from trkh.tools.audit_class_noise_transition_readiness import (
    build_class_noise_transition_readiness,
    proxy_forward_transition,
)
from trkh.tools.audit_teacher_uncertainty_critic_readiness import _load_prediction_table


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


def test_proxy_forward_transition_uses_predicted_class_as_latent_row(tmp_path: Path) -> None:
    rows = [
        (0, 0, [0.9, 0.1]),
        (1, 1, [0.8, 0.2]),
        (2, 1, [0.1, 0.9]),
    ]
    path = _write_predictions(tmp_path / "expert_oof.csv", rows, split="train")
    table = _load_prediction_table("expert", path)
    transition, support = proxy_forward_transition(table)
    assert support.tolist() == [2, 1]
    assert transition[0].tolist() == pytest.approx([0.5, 0.5])
    assert transition[1].tolist() == pytest.approx([0.0, 1.0])


def test_transition_readiness_accepts_stable_experts_and_refuses_test(tmp_path: Path) -> None:
    rows = [
        (0, 0, [0.9, 0.05, 0.05]),
        (1, 1, [0.05, 0.9, 0.05]),
        (2, 2, [0.05, 0.05, 0.9]),
        (3, 0, [0.9, 0.05, 0.05]),
        (4, 1, [0.05, 0.9, 0.05]),
        (5, 2, [0.05, 0.05, 0.9]),
    ]
    train_a = _write_predictions(tmp_path / "a_oof.csv", rows, split="train")
    train_b = _write_predictions(tmp_path / "b_oof.csv", rows, split="train")
    val_a = _write_predictions(tmp_path / "a_val.csv", rows, split="val")
    val_b = _write_predictions(tmp_path / "b_val.csv", rows, split="val")
    summary = build_class_noise_transition_readiness(
        train_experts={"a": train_a, "b": train_b},
        validation_experts={"a": val_a, "b": val_b},
        output_dir=tmp_path / "audit",
        focus_class_index=1,
        min_focus_latent_support=1,
    )
    assert summary["smoke_gate_ready"] is True
    assert summary["test_split_used"] is False
    assert np.asarray(summary["validation"]["a"]["matrix"]).shape == (3, 3)
    assert (tmp_path / "audit" / "proxy_transition_matrices.csv").is_file()

    test_a = _write_predictions(tmp_path / "a_locked.csv", rows, split="test")
    with pytest.raises(ValueError, match="Locked test"):
        build_class_noise_transition_readiness(
            train_experts={"a": train_a, "b": train_b},
            validation_experts={"a": test_a, "b": val_b},
            output_dir=tmp_path / "blocked",
            min_focus_latent_support=1,
        )

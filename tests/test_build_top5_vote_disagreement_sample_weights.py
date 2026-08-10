import csv
import json
from pathlib import Path

import pytest

from trkh.tools.build_top5_vote_disagreement_sample_weights import (
    _weight_for_vote_pattern,
    build_manifest,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_expert(root: Path, name: str, rows) -> None:
    expert_dir = root / name
    expert_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"classes": ["C", "A", "B"], "samples": len(rows)}
    (expert_dir / "metrics_train.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (expert_dir / "predictions_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "prob_0", "prob_1", "prob_2"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(stem: str, source_prediction: str):
    # Source order is C,A,B. Target YOLO order in the fixture is A,B,C.
    probs_by_source = {
        "C": ("0.90", "0.05", "0.05"),
        "A": ("0.05", "0.90", "0.05"),
        "B": ("0.05", "0.05", "0.90"),
    }
    prob_0, prob_1, prob_2 = probs_by_source[source_prediction]
    return {
        "path": f"{stem}_box000.jpg",
        "prob_0": prob_0,
        "prob_1": prob_1,
        "prob_2": prob_2,
    }


def test_weight_policy_protects_focus_class_and_downweights_nonfocus_class1_votes():
    focus_weight, focus_reason = _weight_for_vote_pattern(
        target_index=1,
        majority_index=0,
        vote_counts={1: 2, 0: 3},
        num_experts=5,
        focus_class=1,
        protect_focus_class=True,
    )
    assert focus_weight == 1.0
    assert focus_reason == "focus_class_recall_protected"

    nonfocus_weight, nonfocus_reason = _weight_for_vote_pattern(
        target_index=0,
        majority_index=0,
        vote_counts={0: 4, 1: 1},
        num_experts=5,
        focus_class=1,
        class1_vote_weight=0.82,
        protect_focus_class=True,
    )
    assert nonfocus_weight == pytest.approx(0.82)
    assert nonfocus_reason == "nonfocus_single_class1_vote"


def test_build_manifest_remaps_experts_and_writes_sample_index_weights(tmp_path):
    data_root = tmp_path / "yolo"
    for stem in ("Image_A", "Image_B", "Image_C"):
        _touch(data_root / "images" / "train" / f"{stem}.jpg")
    labels_dir = data_root / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "Image_A.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "Image_B.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "Image_C.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = data_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: images/train",
                "val: images/train",
                "names:",
                "  0: A",
                "  1: B",
                "  2: C",
            ]
        ),
        encoding="utf-8",
    )

    expert_root = tmp_path / "experts"
    _write_expert(
        expert_root,
        "expert1",
        [_row("Image_A", "A"), _row("Image_B", "B"), _row("Image_C", "C")],
    )
    _write_expert(
        expert_root,
        "expert2",
        [_row("Image_A", "A"), _row("Image_B", "A"), _row("Image_C", "A")],
    )
    _write_expert(
        expert_root,
        "expert3",
        [_row("Image_A", "B"), _row("Image_B", "B"), _row("Image_C", "A")],
    )

    output_dir = tmp_path / "out"
    summary = build_manifest(
        data=data_yaml,
        expert_root=expert_root,
        output_dir=output_dir,
        expected_num_classes=3,
        focus_class=1,
        class1_vote_weight=0.82,
        majority_conflict_weight=0.80,
        protect_focus_class=True,
    )

    assert summary["samples"] == 3
    assert summary["disagreement_rows"] == 3
    assert summary["focus_protected_rows"] == 1
    assert summary["manifest_rows"] == 2
    with (output_dir / "top5_vote_disagreement_sample_weights_train_only.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [int(row["sample_index"]) for row in rows] == [0, 2]
    assert [row["reason"] for row in rows] == [
        "nonfocus_single_class1_vote",
        "nonfocus_majority_conflict",
    ]
    assert float(rows[0]["sample_weight"]) == pytest.approx(0.82)
    assert float(rows[1]["sample_weight"]) == pytest.approx(0.80)
    assert int(rows[0]["prediction_index"]) == 0
    assert int(rows[1]["prediction_index"]) == 0

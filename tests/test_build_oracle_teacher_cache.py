import pytest

from trkh.tools.build_oracle_teacher_cache import _read_csv, build_oracle_teacher


def _row(sample_index, path, target, pred, probs):
    row = {
        "sample_index": str(sample_index),
        "image_path": path,
        "target_index": str(target),
        "target_name": f"c{target}",
        "prediction_index": str(pred),
        "prediction_name": f"c{pred}",
    }
    row.update({f"prob_{index}_c{index}": str(value) for index, value in enumerate(probs)})
    return row


def test_build_oracle_teacher_uses_correct_experts_by_sample_index():
    named_rows = {
        "base": {
            "sample:0": _row(0, "same.jpg", 0, 1, [0.2, 0.7, 0.1]),
            "sample:1": _row(1, "same.jpg", 1, 1, [0.1, 0.8, 0.1]),
        },
        "color": {
            "sample:0": _row(0, "same.jpg", 0, 0, [0.75, 0.2, 0.05]),
            "sample:1": _row(1, "same.jpg", 1, 2, [0.1, 0.2, 0.7]),
        },
        "fgbg": {
            "sample:0": _row(0, "same.jpg", 0, 0, [0.6, 0.3, 0.1]),
            "sample:1": _row(1, "same.jpg", 1, 1, [0.1, 0.7, 0.2]),
        },
    }

    records, summary = build_oracle_teacher(
        named_rows,
        selection="correct_mean",
        hard_target_blend=0.0,
        temperature=1.0,
    )

    assert [record["sample_index"] for record in records] == ["0", "1"]
    assert records[0]["selected_experts"] == "color,fgbg"
    assert records[0]["teacher_pred_index"] == 0
    assert records[1]["selected_experts"] == "base,fgbg"
    assert records[1]["teacher_pred_index"] == 1
    assert summary["selection_counts"] == {"correct_expert": 2}


def test_build_oracle_teacher_hard_target_blend_sharpens_target():
    named_rows = {
        "base": {"sample:0": _row(0, "a.jpg", 1, 1, [0.2, 0.5, 0.3])},
        "fgbg": {"sample:0": _row(0, "a.jpg", 1, 1, [0.1, 0.6, 0.3])},
    }

    records, _ = build_oracle_teacher(
        named_rows,
        selection="correct_mean",
        hard_target_blend=0.5,
        temperature=1.0,
    )

    assert records[0]["prob_1"] > 0.75


def test_read_csv_rejects_duplicate_sample_index(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sample_index,image_path,target_index,prediction_index,prob_0_c0,prob_1_c1",
                "0,a.jpg,0,0,0.8,0.2",
                "0,b.jpg,1,1,0.1,0.9",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample keys"):
        _read_csv(csv_path)

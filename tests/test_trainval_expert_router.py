import json

import pytest

from trkh.tools.train_trainval_expert_router import _read_csv, _source_class_names


def test_source_class_names_prefers_probability_suffix_over_metrics_order(tmp_path):
    metrics_path = tmp_path / "metrics_val.json"
    metrics_path.write_text(
        json.dumps({"classes": ["sorted_c0", "sorted_c1"]}),
        encoding="utf-8",
    )
    rows = {
        "sample.jpg": {
            "path": "sample.jpg",
            "target_index": "0",
            "target_name": "raw_c0",
            "prob_0_raw_c0": "0.7",
            "prob_1_raw_c1": "0.3",
        }
    }

    assert _source_class_names(rows, metrics_path) == ["raw_c0", "raw_c1"]


def test_read_csv_uses_sample_index_for_object_level_rows(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sample_index,image_path,target_name,prediction_name,prob_0_c0,prob_1_c1",
                "0,same.jpg,c0,c0,0.8,0.2",
                "1,same.jpg,c1,c1,0.1,0.9",
            ]
        ),
        encoding="utf-8",
    )

    rows = _read_csv(csv_path)

    assert sorted(rows) == ["sample_index:0", "sample_index:1"]
    assert rows["sample_index:1"]["path"] == "same.jpg"


def test_read_csv_rejects_duplicate_path_without_sample_index(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "image_path,target_name,prediction_name,prob_0_c0,prob_1_c1",
                "same.jpg,c0,c0,0.8,0.2",
                "same.jpg,c1,c1,0.1,0.9",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate routing key"):
        _read_csv(csv_path)

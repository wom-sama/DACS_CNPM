from pathlib import Path

import pytest

from trkh.tools.audit_label_issues_cleanlab import (
    _probabilities,
    _probability_columns,
    _read_rows,
)


def test_read_rows_accepts_evaluate_prediction_aliases(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions_detailed.csv"
    csv_path.write_text(
        "\n".join(
            [
                "image_path,target_name,prob_0_a,prob_1_b",
                "D:\\data\\train\\a\\sample.jpg,a,0.75,0.25",
            ]
        ),
        encoding="utf-8",
    )

    rows = _read_rows(csv_path)

    assert rows[0]["path"] == "D:\\data\\train\\a\\sample.jpg"
    assert rows[0]["true_name"] == "a"


def test_read_rows_accepts_target_index_and_preserves_sample_index(tmp_path: Path) -> None:
    csv_path = tmp_path / "oof_predictions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sample_index,path,target_index,prob_0,prob_1",
                "17,D:\\data\\train\\a\\sample.jpg,1,0.05,0.95",
            ]
        ),
        encoding="utf-8",
    )

    rows = _read_rows(csv_path)

    assert rows[0]["sample_index"] == "17"
    assert rows[0]["true_index"] == "1"


def test_probability_columns_accept_class_suffix_and_normalize() -> None:
    row = {
        "prob_0_Xoai_Song": "2.0",
        "prob_1_Xoai_Chin": "1.0",
        "prob_2_Xoai_Hu": "1.0",
    }

    columns = _probability_columns(row, 3)
    probabilities = _probabilities(row, 3, columns)

    assert columns == ["prob_0_Xoai_Song", "prob_1_Xoai_Chin", "prob_2_Xoai_Hu"]
    assert probabilities == [0.5, 0.25, 0.25]


def test_probability_columns_reject_duplicate_class_index() -> None:
    row = {
        "prob_0": "0.5",
        "prob_0_Xoai_Song": "0.5",
        "prob_1": "0.5",
    }

    with pytest.raises(ValueError, match="Duplicate probability column"):
        _probability_columns(row, 2)

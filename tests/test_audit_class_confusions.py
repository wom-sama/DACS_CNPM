from trkh.tools.audit_class_confusions import _index_from_row


def test_index_from_row_prefers_dataset_name_mapping() -> None:
    row = {
        "y_true": "0",
        "true_name": "class_one",
    }

    result = _index_from_row(
        row,
        index_keys=("target_index", "y_true"),
        name_keys=("target_name", "true_name"),
        class_to_index={"class_zero": 0, "class_one": 1},
    )

    assert result == 1


def test_index_from_row_falls_back_to_numeric_index() -> None:
    row = {
        "target_index": "3",
        "target_name": "",
    }

    result = _index_from_row(
        row,
        index_keys=("target_index", "y_true"),
        name_keys=("target_name", "true_name"),
        class_to_index={},
    )

    assert result == 3

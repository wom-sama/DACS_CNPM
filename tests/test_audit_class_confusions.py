import pytest

from trkh.tools.audit_class_confusions import _index_from_row, _prob_key


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


def test_index_from_row_rejects_unknown_name_with_authoritative_mapping() -> None:
    with pytest.raises(ValueError, match="Unknown class name"):
        _index_from_row(
            {"target_index": "0", "target_name": "wrong_name"},
            index_keys=("target_index",),
            name_keys=("target_name",),
            class_to_index={"class_zero": 0},
            require_known_name=True,
        )


def test_index_from_row_rejects_missing_name_and_index() -> None:
    with pytest.raises(ValueError, match="missing both"):
        _index_from_row(
            {},
            index_keys=("target_index",),
            name_keys=("target_name",),
            class_to_index={},
        )


def test_probability_key_prefers_semantic_name_over_source_numeric_order() -> None:
    row = {
        "prob_1_other_class": "0.1",
        "prob_3_focus_class": "0.7",
    }

    assert _prob_key(row, class_index=1, class_name="focus_class") == "prob_3_focus_class"


def test_probability_key_does_not_fallback_to_numeric_when_name_is_authoritative() -> None:
    row = {"prob_1_other_class": "0.7", "prob_1": "0.7"}

    assert _prob_key(row, class_index=1, class_name="focus_class") is None

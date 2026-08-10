import pytest

from trkh.evaluation.evaluate import _resolve_checkpoint_class_names


def test_class_name_contract_accepts_exact_order() -> None:
    names = ["class_0", "class_1"]
    assert _resolve_checkpoint_class_names({"class_names": names}, names) == names


def test_class_name_contract_supports_legacy_missing_names() -> None:
    assert _resolve_checkpoint_class_names({}, ["class_0", "class_1"]) == [
        "class_0",
        "class_1",
    ]


@pytest.mark.parametrize(
    "checkpoint_names",
    (["class_1", "class_0"], ["class_0", "renamed"]),
)
def test_class_name_contract_rejects_same_size_remap(checkpoint_names) -> None:
    with pytest.raises(ValueError, match="class-name order mismatch"):
        _resolve_checkpoint_class_names(
            {"class_names": checkpoint_names},
            ["class_0", "class_1"],
        )

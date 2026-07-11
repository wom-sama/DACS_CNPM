from __future__ import annotations

import pytest
import torch

from trkh.inference.stream_infer import build_classification_prediction_result


def test_build_classification_prediction_result_filters_and_preserves_top_prediction() -> None:
    result = build_classification_prediction_result(
        logits=torch.tensor([0.0, 2.0, 1.0]),
        class_names=["zero", "one", "two"],
        top_k=3,
        confidence_threshold=0.20,
    )

    assert result["top_prediction"]["class_index"] == 1
    assert [item["class_index"] for item in result["predictions"]] == [1, 2]


def test_build_classification_prediction_result_keeps_top_one_when_threshold_rejects_all() -> None:
    result = build_classification_prediction_result(
        logits=torch.tensor([0.1, 0.0]),
        class_names=["zero", "one"],
        top_k=0,
        confidence_threshold=0.99,
    )

    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == result["top_prediction"]


def test_build_classification_prediction_result_rejects_metadata_mismatch() -> None:
    with pytest.raises(ValueError, match="class_names/logits mismatch"):
        build_classification_prediction_result(
            logits=torch.tensor([0.0, 1.0]),
            class_names=["only_one_name"],
        )

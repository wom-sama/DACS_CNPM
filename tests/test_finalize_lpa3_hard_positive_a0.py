from __future__ import annotations

import pytest

from trkh.tools.finalize_lpa3_hard_positive_a0 import (
    candidate_feature_difference_count,
    parse_args,
)


def test_finalizer_requires_explicit_visual_confirmation() -> None:
    args = parse_args(["--audit-dir", "formal"])
    assert args.confirm_visual_review is False


def test_candidate_feature_difference_count_is_strict() -> None:
    rows = [
        {
            "variant": "candidate",
            "sample_index": "1",
            "aug_prediction_index": "1",
            "selected_step": "5",
            "feature_distance": "0.2",
        },
        {
            "variant": "feature_only",
            "sample_index": "1",
            "aug_prediction_index": "0",
            "selected_step": "5",
            "feature_distance": "0.2",
        },
        {
            "variant": "candidate",
            "sample_index": "2",
            "aug_prediction_index": "1",
            "selected_step": "5",
            "feature_distance": "0.2",
        },
        {
            "variant": "feature_only",
            "sample_index": "2",
            "aug_prediction_index": "1",
            "selected_step": "5",
            "feature_distance": "0.2",
        },
    ]
    assert candidate_feature_difference_count(rows) == 1
    with pytest.raises(ValueError, match="coverage differ"):
        candidate_feature_difference_count(rows[:-1])

from __future__ import annotations

import pytest

from trkh.tools.probe_token_prune_sensitivity import (
    build_changed_prediction_rows,
    build_variant_checkpoint,
    parse_variant_spec,
    parse_variant_specs,
)


def test_parse_variant_spec_supports_checkpoint_off_and_override() -> None:
    baseline = parse_variant_spec("baseline=checkpoint")
    assert baseline["mode"] == "checkpoint"
    assert baseline["token_pruning"] is None

    no_prune = parse_variant_spec("no_prune=off")
    assert no_prune["token_pruning"] is False
    assert no_prune["token_prune_layers"] == ""

    override = parse_variant_spec("keep95_85=2,5|0.95,0.85")
    assert override["token_pruning"] is True
    assert override["token_prune_layers"] == "2,5"
    assert override["token_keep_rates"] == "0.95,0.85"

    with pytest.raises(ValueError):
        parse_variant_spec("bad=2,5")


def test_parse_variant_specs_adds_baseline_and_rejects_duplicates() -> None:
    specs = parse_variant_specs(["keep95_85=2,5|0.95,0.85"])
    assert [item["name"] for item in specs] == ["baseline", "keep95_85"]

    with pytest.raises(ValueError):
        parse_variant_specs(["a=off", "a=checkpoint"])


def test_build_variant_checkpoint_only_overrides_model_config() -> None:
    checkpoint = {
        "model_config": {
            "token_pruning": True,
            "token_prune_layers": "2,5",
            "token_keep_rates": "0.85,0.65",
            "other": 3,
        },
        "class_names": ["0", "1"],
    }
    variant = parse_variant_spec("no_prune=off")
    updated = build_variant_checkpoint(checkpoint, variant)

    assert updated["model_config"]["token_pruning"] is False
    assert updated["model_config"]["token_prune_layers"] == ""
    assert updated["model_config"]["token_keep_rates"] == ""
    assert updated["model_config"]["other"] == 3
    assert checkpoint["model_config"]["token_pruning"] is True


def test_changed_rows_count_class1_related_transitions() -> None:
    baseline = [
        {"sample_index": 0, "target_index": 0, "prediction_index": 1, "confidence": 0.7},
        {"sample_index": 1, "target_index": 1, "prediction_index": 1, "confidence": 0.8},
        {"sample_index": 2, "target_index": 2, "prediction_index": 0, "confidence": 0.6},
    ]
    current = [
        {"sample_index": 0, "target_index": 0, "prediction_index": 0, "confidence": 0.6},
        {"sample_index": 1, "target_index": 1, "prediction_index": 0, "confidence": 0.7},
        {"sample_index": 2, "target_index": 2, "prediction_index": 1, "confidence": 0.5},
    ]

    rows, counts = build_changed_prediction_rows(
        baseline_records=baseline,
        current_records=current,
        variant_name="v",
        class_names=["0", "1", "2"],
    )

    assert [row["transition"] for row in rows] == ["0:1->0", "1:1->0", "2:0->1"]
    assert counts == {"corrections": 1, "harms": 1, "neutral": 1, "class1_related": 3}

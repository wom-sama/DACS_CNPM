from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from trkh.tools.audit_dart_recurrent_aggregation_readiness import (
    _average_state_dicts_strict,
    _event_diversity_passed,
    _largest_remainder_quotas,
    _logical_batch_map,
    _logical_seed,
    _maximum_numeric_difference,
    _pairwise_delta_diagnostics,
    _process_snapshot,
    _protocol_index_sha256,
    _role_transform_config,
    _select_grouped_rows,
    assess_dart_a0,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow


def _row(index: int, target: int, source: str, *, fold: int = 1) -> CleanTrainRow:
    probabilities = tuple(1.0 if value == target else 0.0 for value in range(5))
    return CleanTrainRow(
        sample_index=index,
        source_stem=source,
        image_path=Path(f"train/{source}.jpg"),
        fold=fold,
        target=target,
        keeper_prediction=target,
        keeper_probabilities=probabilities,
    )


def test_largest_remainder_uses_natural_distribution_and_class_tie_break() -> None:
    assert _largest_remainder_quotas({0: 382, 1: 115, 2: 367, 3: 510, 4: 456}, 256) == (
        54,
        16,
        51,
        71,
        64,
    )
    assert _largest_remainder_quotas({index: 1 for index in range(5)}, 3) == (
        1,
        1,
        1,
        0,
        0,
    )


def test_grouped_selection_is_deterministic_exact_and_never_splits_source() -> None:
    rows = [
        _row(index, index % 5, f"source_{index}")
        for index in range(25)
    ]
    first, first_summary = _select_grouped_rows(
        rows,
        fold=1,
        limit=10,
        namespace="unit",
    )
    second, second_summary = _select_grouped_rows(
        rows,
        fold=1,
        limit=10,
        namespace="unit",
    )
    assert [row.sample_index for row in first] == [row.sample_index for row in second]
    assert first_summary == second_summary
    assert len(first) == 10
    assert first_summary["class_counts"] == [2, 2, 2, 2, 2]
    assert first_summary["source_groups"] == 10


def test_logical_seed_is_role_step_specific_and_repeatable() -> None:
    observed = {
        _logical_seed(role, step)
        for role in ("context_geometry", "illumination_surface")
        for step in range(3)
    }
    assert len(observed) == 6
    assert _logical_seed("context_geometry", 2) == _logical_seed(
        "context_geometry", 2
    )


def test_protocol_index_hash_has_no_implicit_trailing_newline() -> None:
    import hashlib

    assert _protocol_index_sha256([7, 11]) == hashlib.sha256(b"7\n11").hexdigest()
    assert _protocol_index_sha256([7, 11]) != hashlib.sha256(b"7\n11\n").hexdigest()


def test_strict_average_matches_fp32_mean_and_preserves_integer_buffer() -> None:
    states = []
    for index in range(4):
        states.append(
            {
                "weight": torch.tensor([float(index), float(index + 2)]),
                "counter": torch.tensor(7, dtype=torch.int64),
            }
        )
    averaged, summary = _average_state_dicts_strict(states)
    assert torch.equal(averaged["weight"], torch.tensor([1.5, 3.5]))
    assert torch.equal(averaged["counter"], torch.tensor(7, dtype=torch.int64))
    assert summary["nonfloating_bit_exact"] is True
    assert summary["nonfloating_keys"] == ["counter"]


def test_strict_average_rejects_differing_nonfloating_buffer() -> None:
    states = [
        {"weight": torch.tensor([float(index)]), "counter": torch.tensor(index % 2)}
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="nonfloating buffer differs"):
        _average_state_dicts_strict(states)


def test_pairwise_delta_diagnostics_detect_useful_diversity() -> None:
    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    models = [nn.Linear(2, 1, bias=False) for _ in range(4)]
    values = ((1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5))
    for model, value in zip(models, values):
        with torch.no_grad():
            model.weight.copy_(torch.tensor([value]))
    rows = _pairwise_delta_diagnostics(
        models,
        {"weight": torch.zeros_like(base.weight)},
    )
    event = {"pairwise_delta": rows}
    assert len(rows) == 6
    assert _event_diversity_passed(event) is True
    assert all(row["endpoint_distance"] > 0.0 for row in rows)


def test_logical_batch_map_rejects_duplicate_role_step() -> None:
    history = [
        {
            "role": "keeper_mixed",
            "role_step": 0,
            "batch_sha256": "a",
            "logical_seed": 1,
        },
        {
            "role": "keeper_mixed",
            "role_step": 0,
            "batch_sha256": "b",
            "logical_seed": 2,
        },
    ]
    with pytest.raises(ValueError, match="Duplicate DART logical batch key"):
        _logical_batch_map(history)


def _comparison_payload() -> dict[str, object]:
    return {
        "control": {"predicted_support": [100, 100, 100, 100, 100]},
        "candidate": {"predicted_support": [100, 98, 100, 100, 100]},
        "delta": {
            "macro_f1": 0.002,
            "class1_f1": 0.010,
            "class1_precision": 0.010,
            "class1_recall": 0.001,
        },
        "transitions": {
            "restricted_focus_fp_reduction": 3,
            "focus_fn_rescue": 4,
            "focus_tp_break": 2,
            "candidate_correction": 8,
            "candidate_harm": 3,
        },
        "maximum_nonfocus_f1_drop": 0.003,
    }


def test_assessment_requires_both_controls_and_all_lighting_gates() -> None:
    mixed = _comparison_payload()
    final = _comparison_payload()
    result = assess_dart_a0(
        structural_checks={"structure": True},
        clean_mixed=mixed,
        clean_final=final,
        illumination_mixed=[_comparison_payload() for _ in range(3)],
    )
    assert result["all_nonvisual_gates_passed"] is True
    failed = _comparison_payload()
    failed["delta"] = dict(failed["delta"], class1_precision=-0.001)
    result = assess_dart_a0(
        structural_checks={"structure": True},
        clean_mixed=mixed,
        clean_final=failed,
        illumination_mixed=[_comparison_payload() for _ in range(3)],
    )
    assert result["all_nonvisual_gates_passed"] is False
    assert "final_class1_precision_delta_nonnegative" in result["failed_checks"]


def test_numeric_replay_difference_is_recursive_and_exact() -> None:
    left = {"a": [1, {"b": 2.5}], "c": True}
    assert _maximum_numeric_difference(left, left) == 0.0
    assert _maximum_numeric_difference(left, {"a": [1, {"b": 2.6}], "c": True}) == pytest.approx(0.1)


def test_role_configs_keep_domains_distinct_without_mix_families() -> None:
    context = _role_transform_config("context_geometry")
    illumination = _role_transform_config("illumination_surface")
    occlusion = _role_transform_config("occlusion_parts")
    keeper = _role_transform_config("keeper_mixed")
    assert context["background_suppression_probability"] == 0.0
    assert illumination["lighting_probability"] == 0.20
    assert illumination["local_exposure_probability"] == 0.30
    assert occlusion["random_erasing_probability"] == 0.08
    assert occlusion["obstacle_probability"] == 0.12
    assert keeper["brightness"] == 0.04
    assert keeper["background_suppression_probability"] == 0.80


def test_process_snapshot_marks_current_windows_python_chain_as_owned() -> None:
    snapshot = _process_snapshot()
    current = [
        row for row in snapshot["processes"] if row["pid"] == snapshot["current_pid"]
    ]
    assert len(current) == 1
    assert current[0]["current_auditor"] is True
    assert current[0]["current_auditor_chain"] is True

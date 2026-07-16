from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from trkh.tools.audit_mutual_channel_patch_readiness import (
    EXPECTED_ADAPTER_PARAMETERS,
    MutualChannelPatchAdapter,
    _equation_diagnostics,
    _stream_tensor_sha256,
    assess_stage_a,
    mcl_drop_index,
    mutual_channel_terms,
)


def test_adapter_schema_scatter_and_zero_residual_are_exact() -> None:
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state().clone()
    adapter = MutualChannelPatchAdapter()
    rng_after = torch.get_rng_state().clone()
    assert torch.equal(rng_before, rng_after)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == (
        EXPECTED_ADAPTER_PARAMETERS
    )
    patches = torch.arange(2 * 4 * 256, dtype=torch.float32).reshape(2, 4, 256)
    indices = torch.tensor([[0, 3, 17, 255], [1, 4, 32, 128]])
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    raw = torch.randn(2, 5)
    output = adapter.forward_sparse(patches, indices, valid, raw)
    assert output["dense"].shape == (2, 256, 16, 16)
    assert output["valid_mask"].shape == (2, 1, 16, 16)
    assert int(output["valid_mask"].sum()) == 6
    assert torch.count_nonzero(output["dense"][0, :, 1, 1]).item() == 0
    assert torch.equal(output["logits"], raw)


def test_locked_drop_seed_examples_are_stable() -> None:
    observed = [
        mcl_drop_index(
            seed=42,
            epoch=0,
            batch=0,
            sample_index=17,
            class_index=class_index,
        )
        for class_index in range(5)
    ]
    assert observed == [2, 1, 2, 1, 0]


def test_streaming_tensor_hash_matches_contiguous_bytes() -> None:
    value = torch.arange(67 * 11, dtype=torch.float32).reshape(67, 11)
    import hashlib

    expected = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    assert _stream_tensor_sha256(value, rows_per_chunk=8) == expected


def test_vectorized_mutual_channel_equations_match_independent_reference() -> None:
    diagnostics = _equation_diagnostics()
    assert diagnostics["all_equations_match_fp32_lte_1e6"]
    assert max(diagnostics["maximum_absolute_errors"].values()) <= 1e-6


def test_control_detached_diagnostics_do_not_change_ce_gradients() -> None:
    first = MutualChannelPatchAdapter()
    second = copy.deepcopy(first)
    patches = torch.randn(3, 7, 256)
    indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13, 14], [16, 17, 18, 19, 20, 21, 22]]
    )
    valid = torch.ones(3, 7, dtype=torch.bool)
    raw = torch.randn(3, 5)
    targets = torch.tensor([0, 1, 4])
    drops = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2], [2, 0, 1, 2, 0]])

    output = first.forward_sparse(patches, indices, valid, raw)
    F.cross_entropy(output["logits"], targets).backward()
    baseline = {
        name: parameter.grad.clone()
        for name, parameter in first.named_parameters()
    }

    output = second.forward_sparse(patches, indices, valid, raw)
    terms = mutual_channel_terms(
        output["projected"], output["valid_mask"], drops, targets
    )
    loss = F.cross_entropy(output["logits"], targets) + 0.0 * (
        1.5 * terms["loss_dis"] + 20.0 * terms["loss_div"]
    ).detach()
    loss.backward()
    observed = {
        name: parameter.grad.clone()
        for name, parameter in second.named_parameters()
    }
    assert baseline.keys() == observed.keys()
    assert all(torch.equal(baseline[name], observed[name]) for name in baseline)


def _comparison_payload() -> dict[str, object]:
    return {
        "delta": {
            "macro_f1": 0.01,
            "class1_f1": 0.01,
            "class1_precision": 0.01,
            "class1_recall": 0.0,
        },
        "transitions": {
            "focus_tp_break": 0,
            "focus_fn_rescue": 1,
            "restricted_focus_fp_reduction": 3,
            "candidate_correction": 4,
            "candidate_harm": 1,
        },
        "maximum_nonfocus_f1_drop": 0.0,
    }


def _comparisons() -> dict[str, object]:
    return {
        condition: {
            "raw_candidate": _comparison_payload(),
            "control_candidate": _comparison_payload(),
        }
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }


def _mechanism() -> dict[str, object]:
    return {
        "control": {
            "loss_dis": 1.0,
            "mean_group_coverage": 1.2,
            "class1_spatial_entropy": 0.5,
            "maximum_class1_within_group_cosine": 0.9,
            "class1_residual_margin_auroc_vs_0_2_4": 0.71,
        },
        "candidate": {
            "loss_dis": 0.95,
            "mean_group_coverage": 1.3,
            "class1_spatial_entropy": 0.5,
            "maximum_class1_within_group_cosine": 0.9,
            "class1_residual_margin_auroc_vs_0_2_4": 0.75,
        },
    }


def test_stage_a_gate_is_all_or_nothing() -> None:
    passing = assess_stage_a(
        structural_checks={"structural": True},
        comparisons=_comparisons(),
        holdout_mechanism=_mechanism(),
    )
    assert passing["all_gates_passed"]
    assert passing["stage_b_authorized"]
    assert not passing["test_authorized"]
    assert not passing["full_train_authorized"]

    failing_comparisons = _comparisons()
    failing_comparisons["clean"]["raw_candidate"]["delta"]["class1_precision"] = 0.0
    failing = assess_stage_a(
        structural_checks={"structural": True},
        comparisons=failing_comparisons,
        holdout_mechanism=_mechanism(),
    )
    assert not failing["all_gates_passed"]
    assert not failing["stage_b_authorized"]
    assert "class1_precision_gain_vs_raw_gte_0p005" in failing["failed_checks"]

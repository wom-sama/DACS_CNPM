from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from trkh.tools.audit_support_constrained_capsule_alm_readiness import (
    EXPECTED_ADAPTER_PARAMETERS,
    SupportConstrainedCapsuleAdapter,
    _dense_patch_map,
    _equation_diagnostics,
    assess_stage_a,
    augmented_lagrangian_loss,
    boundary_schedule_sha256,
    interleaved_boundary_batches,
    normalized_alm_constraints,
)


def test_adapter_schema_rng_isolation_and_zero_residual_are_exact() -> None:
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state().clone()
    adapter = SupportConstrainedCapsuleAdapter()
    rng_after = torch.get_rng_state().clone()
    assert torch.equal(rng_before, rng_after)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == (
        EXPECTED_ADAPTER_PARAMETERS
    )
    assert list(adapter.vote_weights.shape) == [256, 8, 5, 16]
    assert list(adapter.class_capsule_bias.shape) == [5, 16]
    patches = torch.randn(2, 7, 256)
    indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [8, 9, 10, 11, 12, 13, 14],
        ]
    )
    valid = torch.tensor(
        [[True, True, False, True, True, True, True], [True] * 7]
    )
    raw = torch.randn(2, 5)
    output = adapter.forward_sparse(patches, indices, valid, raw)
    assert torch.equal(output["logits"], raw)
    assert output["couplings"].shape == (2, 7, 5)
    assert torch.equal(output["couplings"][~valid], torch.zeros_like(output["couplings"][~valid]))
    assert torch.allclose(
        output["couplings"].sum(dim=2)[valid],
        torch.ones_like(output["couplings"].sum(dim=2)[valid]),
        atol=1e-6,
        rtol=0.0,
    )


def test_vectorized_routing_and_alm_match_independent_references() -> None:
    diagnostics = _equation_diagnostics()
    assert diagnostics["routing_matches_reference_lte_1e6"]
    assert diagnostics["constraints_match_reference_lte_1e7"]
    assert diagnostics["coupling_contract_exact"]
    assert max(diagnostics["maximum_absolute_errors"].values()) <= 1e-6


def test_normalized_alm_support_is_zero_at_raw_and_rank_is_per_positive() -> None:
    positive = torch.tensor(
        [[0.0, 1.0, 0.2, -0.1, 0.3], [0.2, 0.8, 0.1, 0.0, -0.2]],
        requires_grad=True,
    )
    negative = torch.tensor(
        [[0.1, 0.9, 0.0, -0.2, 0.2], [0.4, 0.4, 0.1, 0.0, -0.1]],
        requires_grad=True,
    )
    constraints = normalized_alm_constraints(
        positive, negative, positive.detach().clone()
    )
    assert constraints["q_rank"].shape == (2,)
    assert constraints["q_support"].shape == (2,)
    assert torch.count_nonzero(constraints["q_support"]).item() == 0
    multipliers = torch.tensor([0.2, 0.4])
    loss = augmented_lagrangian_loss(
        constraints["q_rank"],
        constraints["q_support"],
        multipliers,
        torch.zeros_like(multipliers),
    )["loss"]
    loss.backward()
    assert positive.grad is not None and torch.count_nonzero(positive.grad).item() > 0
    assert negative.grad is not None and torch.count_nonzero(negative.grad).item() > 0


def test_detached_control_boundary_terms_do_not_change_ce_gradients() -> None:
    first = SupportConstrainedCapsuleAdapter()
    second = copy.deepcopy(first)
    patches = torch.randn(4, 9, 256)
    indices = torch.arange(9).reshape(1, 9).expand(4, -1)
    valid = torch.ones(4, 9, dtype=torch.bool)
    raw = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 2, 4])

    baseline = first.forward_sparse(patches, indices, valid, raw)
    F.cross_entropy(baseline["logits"], targets).backward()
    baseline_gradients = {
        name: parameter.grad.clone()
        for name, parameter in first.named_parameters()
    }

    observed = second.forward_sparse(patches, indices, valid, raw)
    constraints = normalized_alm_constraints(
        observed["logits"][:2], observed["logits"][2:], raw[:2]
    )
    zeros = torch.zeros(2)
    diagnostic = augmented_lagrangian_loss(
        constraints["q_rank"], constraints["q_support"], zeros, zeros
    )["loss"]
    loss = F.cross_entropy(observed["logits"], targets) + 0.0 * diagnostic.detach()
    loss.backward()
    observed_gradients = {
        name: parameter.grad.clone()
        for name, parameter in second.named_parameters()
    }
    assert baseline_gradients.keys() == observed_gradients.keys()
    assert all(
        torch.equal(baseline_gradients[name], observed_gradients[name])
        for name in baseline_gradients
    )


def test_boundary_serialization_interleave_and_dense_invalid_contract() -> None:
    events = [
        {
            "epoch": 0,
            "condition": "clean",
            "batch": 0,
            "positive_indices": [2, 4],
            "negative_indices": [1, 3],
        },
        {
            "epoch": 0,
            "condition": "lighting_dim",
            "batch": 0,
            "positive_indices": [6, 8],
            "negative_indices": [5, 7],
        },
    ]
    assert boundary_schedule_sha256(events) == (
        "9c0fdf53e858598be34284e66d1ff160ba30e6f6d61223e4c556895de6a19f81"
    )
    interleave = interleaved_boundary_batches(natural_batches=231)
    assert len(interleave) == 108
    assert len(set(interleave)) == 108
    assert interleave[0] == 0 and interleave[-1] == 228
    values = torch.tensor([0.3, 0.7, 0.9])
    indices = torch.tensor([0, 17, 255])
    valid = torch.tensor([True, False, True])
    dense = _dense_patch_map(values, indices, valid)
    assert dense.shape == (16, 16)
    assert torch.isclose(dense[0, 0], torch.tensor(0.3))
    assert dense[1, 1].item() == 0.0
    assert torch.isclose(dense[15, 15], torch.tensor(0.9))


def _comparison_payload() -> dict[str, object]:
    return {
        "candidate": {
            "per_class_f1": [0.9, 0.82, 0.9, 0.9, 0.9],
            "per_class_precision": [0.9, 0.84, 0.9, 0.9, 0.9],
            "per_class_recall": [0.9, 0.81, 0.9, 0.9, 0.9],
        },
        "delta": {
            "macro_f1": 0.01,
            "class1_f1": 0.01,
            "class1_precision": 0.02,
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


def _mcl_comparisons() -> dict[str, object]:
    payload = {
        "candidate": {
            "per_class_f1": [0.7, 0.6, 0.7, 0.7, 0.7],
            "per_class_precision": [0.7, 0.6, 0.7, 0.7, 0.7],
            "per_class_recall": [0.7, 0.6, 0.7, 0.7, 0.7],
        }
    }
    return {
        condition: {"raw_candidate": copy.deepcopy(payload)}
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }


def _mechanisms() -> dict[str, object]:
    control = {
        "rank_violation_mean": 1.0,
        "support_violation_mean": 0.2,
        "support_violation_p95": 0.3,
        "class1_capsule_residual_margin_auroc_vs_0_2_4": 0.72,
    }
    candidate = {
        "rank_violation_mean": 0.8,
        "support_violation_mean": 0.1,
        "support_violation_p95": 0.2,
        "class1_capsule_residual_margin_auroc_vs_0_2_4": 0.75,
        "normalized_class1_routing_entropy_mean": 0.8,
        "effective_class1_routed_patch_count_mean": 50.0,
        "coupling_parent_sum_maximum_error": 1e-7,
        "invalid_coupling_mass": 0.0,
        "dynamic_uniform_maximum_errors": {
            "couplings": 0.1,
            "class_capsules": 0.1,
            "capsule_lengths": 0.1,
            "residual_logits": 0.1,
            "logits": 0.1,
        },
        "dynamic_uniform_logit_changed_rows": 1,
    }
    return {
        condition: {"control": copy.deepcopy(control), "candidate": copy.deepcopy(candidate)}
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }


def test_stage_a_gate_is_all_or_nothing() -> None:
    dual = {
        "finite_nonnegative": True,
        "rank_nonzero": 10,
        "rank_std": 0.1,
        "largest_source_share": 0.01,
    }
    passing = assess_stage_a(
        structural_checks={"structural": True},
        comparisons=_comparisons(),
        mechanisms=_mechanisms(),
        mcl_comparisons=_mcl_comparisons(),
        dual=dual,
    )
    assert passing["all_gates_passed"]
    assert passing["stage_b_authorized"]
    assert not passing["test_authorized"]
    assert not passing["full_train_authorized"]

    failing_comparisons = _comparisons()
    failing_comparisons["clean"]["raw_candidate"]["delta"][
        "class1_precision"
    ] = 0.0
    failing = assess_stage_a(
        structural_checks={"structural": True},
        comparisons=failing_comparisons,
        mechanisms=_mechanisms(),
        mcl_comparisons=_mcl_comparisons(),
        dual=dual,
    )
    assert not failing["all_gates_passed"]
    assert not failing["stage_b_authorized"]
    assert "class1_precision_gain_vs_raw_gte_0p010" in failing["failed_checks"]

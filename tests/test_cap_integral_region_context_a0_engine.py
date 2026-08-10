from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.cap_integral_region_context_a0_engine import (
    CAPIntegralRegionBinaryHead,
    CAP_ROLES,
    CHANNELS,
    FEATURE_SIZE,
    REGION_BOXES,
    REGION_COUNT,
    SAME_BUDGET_CAP_ROLES,
    apply_keeper_suppression,
    array_sha256,
    build_epoch_orders,
    build_partitioned_cross_sample_mapping,
    build_spatial_permutations,
    calibrate_tp_retention_threshold,
    canonicalize_valid_support,
    engineering_oracles,
    initialize_role_model,
    parameter_contract,
    role_seed,
    state_arrays_sha256,
    model_state_arrays,
    valid_support_boxes,
)


REPOSITORY = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    REPOSITORY
    / "docs"
    / "TRKH_5CLASS_CAP_INTEGRAL_REGION_CONTEXT_A0_LOCK_20260724.json"
)
GEOMETRY_PATH = (
    REPOSITORY
    / "runs"
    / "audit_attention_maxsep_prototype_a0_20260721"
    / "cohort_geometry.npz"
)


def _lock() -> dict[str, object]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_engine_equations_match_independent_numpy_oracles() -> None:
    evidence = engineering_oracles()
    assert evidence["passed"]
    assert max(evidence["errors"].values()) <= 3e-15


def test_region_geometry_and_derangements_reproduce_lock() -> None:
    lock = _lock()
    assert REGION_BOXES.tolist() == lock["geometry"]["region_boxes"]
    assert array_sha256(REGION_BOXES) == lock["geometry"]["region_boxes_sha256"]
    with np.load(GEOMETRY_PATH, allow_pickle=False) as archive:
        sample_indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        folds = np.asarray(archive["folds"], dtype=np.int64)
        sources = np.asarray(archive["source_stems"]).astype(str).tolist()
    spatial = build_spatial_permutations(sample_indices)
    assert (
        array_sha256(spatial)
        == lock["geometry"]["spatial_derangement"]["permutations_sha256"]
    )
    expected = np.arange(REGION_COUNT, dtype=np.int64)
    assert not bool((spatial == expected[None, :]).any())

    for fold_record in lock["folds"]:
        outer_fold = int(fold_record["outer_fold"])
        calibration_fold = int(fold_record["calibration_fold"])
        partitions = {
            "fit": np.flatnonzero(
                (folds != outer_fold) & (folds != calibration_fold)
            ).astype(np.int64),
            "calibration": np.flatnonzero(
                folds == calibration_fold
            ).astype(np.int64),
            "held": np.flatnonzero(folds == outer_fold).astype(np.int64),
        }
        mapping, details = build_partitioned_cross_sample_mapping(
            partitions,
            sample_indices,
            sources,
            20260724 + 7000 + outer_fold,
        )
        assert (
            array_sha256(mapping)
            == fold_record["cross_sample_derangement_sha256"]
        )
        assert details == fold_record["cross_sample_derangement_partitions"]
        orders = build_epoch_orders(
            partitions["fit"],
            seed=int(fold_record["primary_seed"]),
        )
        assert [array_sha256(order) for order in orders] == fold_record[
            "primary_orders"
        ]["per_epoch_sha256"]
        assert array_sha256(np.concatenate(orders)) == fold_record[
            "primary_orders"
        ]["all_epochs_sha256"]


def test_valid_support_is_rectangular_and_padding_cannot_enter_cap() -> None:
    masks = np.zeros((2, FEATURE_SIZE, FEATURE_SIZE), dtype=np.bool_)
    masks[0, 2:14, 3:13] = True
    masks[1] = True
    boxes = valid_support_boxes(masks)
    assert boxes.tolist() == [[3, 2, 13, 14], [0, 0, 16, 16]]

    features = torch.zeros((2, CHANNELS, FEATURE_SIZE, FEATURE_SIZE))
    features[0, :, 2:14, 3:13] = 2.0
    features[0, :, ~masks[0]] = 1000.0
    features[1] = torch.arange(
        FEATURE_SIZE * FEATURE_SIZE, dtype=torch.float32
    ).reshape(1, FEATURE_SIZE, FEATURE_SIZE)
    restored = canonicalize_valid_support(features, boxes)
    assert tuple(restored.shape) == tuple(features.shape)
    assert torch.equal(restored[0], torch.full_like(restored[0], 2.0))
    assert torch.equal(restored[1], features[1])

    invalid = masks[:1].copy()
    invalid[0, 5, 5] = False
    with pytest.raises(ValueError, match="not rectangular"):
        valid_support_boxes(invalid)


def test_all_cap_roles_share_parameter_budget_and_locked_initialization() -> None:
    device = torch.device("cpu")
    contracts: dict[str, dict[str, object]] = {}
    state_hashes: dict[str, str] = {}
    for role in CAP_ROLES:
        model = initialize_role_model(role, fold=2, device=device)
        contracts[role] = parameter_contract(model)
        state_hashes[role] = state_arrays_sha256(model_state_arrays(model))
    reference = contracts["cap_context_candidate"]
    for role in SAME_BUDGET_CAP_ROLES:
        assert contracts[role] == reference
    assert contracts["integral_self_only_control"] == reference
    assert (
        state_hashes["cap_context_candidate"]
        == state_hashes["cap_spatial_deranged_control"]
        == state_hashes["cap_cross_sample_context_control"]
        == state_hashes["integral_self_only_control"]
    )
    assert (
        state_hashes["cap_context_seed_repeat"]
        != state_hashes["cap_context_candidate"]
    )
    assert role_seed("cap_context_seed_repeat", 2) == 20360726
    assert role_seed("cap_context_candidate", 2) == 20260726


@pytest.mark.parametrize("role", CAP_ROLES)
def test_cap_role_forward_backward_is_finite(role: str) -> None:
    torch.manual_seed(7)
    batch_size = 2
    features = torch.randn(
        batch_size,
        CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
        dtype=torch.float32,
        requires_grad=True,
    )
    boxes = np.asarray([[2, 1, 15, 14], [0, 3, 16, 13]], dtype=np.int64)
    model = initialize_role_model(role, fold=0, device=torch.device("cpu"))
    kwargs: dict[str, object] = {}
    if role == "cap_spatial_deranged_control":
        kwargs["spatial_permutations"] = torch.as_tensor(
            build_spatial_permutations(
                np.asarray([101, 202], dtype=np.int64)
            )
        )
    if role == "cap_cross_sample_context_control":
        kwargs["partner_features"] = torch.flip(features, dims=[0])
        kwargs["partner_support_boxes"] = boxes[::-1].copy()
    logits, auxiliary = model(
        features,
        boxes,
        return_auxiliary=True,
        **kwargs,
    )
    assert tuple(logits.shape) == (batch_size,)
    assert tuple(auxiliary["region_attention"].shape) == (
        batch_size,
        REGION_COUNT,
        REGION_COUNT,
    )
    loss = logits.square().mean()
    loss.backward()
    assert torch.isfinite(logits).all()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_context_gap_optimization_matches_full_region_construction() -> None:
    torch.manual_seed(11)
    model = CAPIntegralRegionBinaryHead(context_mode="context")
    regions = torch.randn(
        2, REGION_COUNT, CHANNELS, 7, 7, dtype=torch.float32
    )
    flattened = regions.flatten(2)
    compact, attention = model.region_context(flattened)
    full = torch.bmm(attention, flattened)
    full_gap = full.reshape(2, REGION_COUNT, CHANNELS, 7, 7).mean(
        dim=(-1, -2)
    )
    assert torch.allclose(compact, full_gap, atol=2e-6, rtol=0.0)


def test_threshold_ties_and_suppression_semantics_are_fail_closed() -> None:
    result = calibrate_tp_retention_threshold(
        np.asarray([0.1, 0.2, 0.2, 0.3, 0.9], dtype=np.float64),
        np.ones(5, dtype=np.bool_),
        maximum_break_fraction=0.4,
    )
    assert result["boundary"] == "tie_next_lower"
    assert result["allowed_tp_breaks"] == 2
    assert result["observed_tp_breaks"] == 1

    probabilities = np.asarray(
        [
            [0.1, 0.7, 0.05, 0.1, 0.05],
            [0.6, 0.2, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.05, 0.1, 0.05],
        ],
        dtype=np.float64,
    )
    action = apply_keeper_suppression(
        probabilities,
        np.asarray([0.1, 0.1, 0.2], dtype=np.float64),
        0.2,
    )
    assert action["suppressed"].tolist() == [True, False, False]
    assert action["predictions"].tolist() == [0, 0, 1]

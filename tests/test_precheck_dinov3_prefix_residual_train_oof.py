from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import trkh.tools.precheck_dinov3_prefix_residual_train_oof as b7
from trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof import (
    expand_pair_rows,
)
from trkh.tools.precheck_dinov3_prefix_residual_train_oof import (
    CACHE_MANIFEST_SCHEMA_VERSION,
    DESCRIPTOR_WIDTH,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DATA_SHA256,
    EXPECTED_HEAD_MACS_ALL_PAIRS,
    EXPECTED_HEAD_PARAMETERS,
    EXPECTED_SOURCE_GROUPS,
    EXPECTED_TRAIN_SAMPLES,
    FOLDS,
    GLOBAL_LOGIT_WIDTH,
    MAX_HEAD_MACS_ALL_PAIRS,
    PAIR_COUNT,
    PAIR_RIVALS,
    PATCH_TOKENS,
    PREFIX_TOKENS,
    PROTOCOL_ID,
    SOFT_RECALL_TARGET,
    TOKEN_WIDTH,
    _PairPrefixDataset,
    _cache_manifest_expected,
    _cache_paths,
    _load_existing_cache,
    _parse_args,
    _sha256,
    _train_readout,
    architecture_self_check,
    assess_prefix_residual_readiness,
    build_derangement_source_map,
    estimate_head_macs_all_pairs,
    primal_dual_soft_recall_loss,
    source_disjoint_block_derangement,
    update_dual_values,
)


def test_locked_architecture_capacity_all_pair_path_and_invariances() -> None:
    result = architecture_self_check()
    assert result["candidate_parameters"] == EXPECTED_HEAD_PARAMETERS == 7_345
    assert result["control_parameters"] == EXPECTED_HEAD_PARAMETERS
    assert result["deranged_parameters"] == EXPECTED_HEAD_PARAMETERS
    assert result["all_parameter_counts_equal"]
    assert result["estimated_head_macs_all_pairs"] == (
        EXPECTED_HEAD_MACS_ALL_PAIRS
    ) == 34_080
    assert estimate_head_macs_all_pairs() <= MAX_HEAD_MACS_ALL_PAIRS
    assert result["candidate_control_initial_state_max_abs_error"] == 0.0
    assert result["candidate_deranged_initial_state_max_abs_error"] == 0.0
    assert result["zero_residual_candidate_control_max_abs_error"] <= 1e-6
    assert result["all_pairs_path_max_abs_error"] <= 1e-6
    assert result["register_permutation_max_abs_error"] <= 1e-6
    assert result["register_variance_unbiased"] is False
    assert result["layer_norm_elementwise_affine"] is False
    assert result["descriptor_shape"] == [6, DESCRIPTOR_WIDTH]
    assert result["output_shape"] == [6]


def test_block_derangement_is_deterministic_bijective_source_disjoint_and_label_blind() -> None:
    indices = np.arange(12, dtype=np.int64)
    groups = np.asarray(
        ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        dtype=object,
    )
    first = source_disjoint_block_derangement(indices, groups, seed=17)
    second = source_disjoint_block_derangement(indices, groups, seed=17)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == set(indices.tolist())
    assert not np.any(first == indices)
    assert all(groups[int(source)] != groups[int(target)] for target, source in enumerate(first))

    source_map, report = build_derangement_source_map(
        sample_count=12,
        fit_indices=np.arange(8, dtype=np.int64),
        hold_indices=np.arange(8, 12, dtype=np.int64),
        source_groups=groups,
        seed=19,
    )
    assert source_map.shape == (12, PREFIX_TOKENS)
    assert np.all(source_map == source_map[:, :1])
    assert report["bijective_permutation_contracts"] == 2
    assert report["expected_permutation_contracts"] == 2
    assert report["fixed_points"] == 0
    assert report["same_source_group_assignments"] == 0
    assert report["fit_to_hold_crossings"] == 0
    assert report["hold_to_fit_crossings"] == 0
    assert report["block_donor_consistent"]
    assert report["label_blind_construction"]


def test_deranged_dataset_moves_a_coherent_residual_block_and_keeps_alignment() -> None:
    labels = np.asarray([0, 1, 2, 4], dtype=np.int64)
    groups = np.asarray(["a", "b", "c", "d"], dtype=object)
    source_map, _report = build_derangement_source_map(
        sample_count=4,
        fit_indices=np.asarray([0, 1], dtype=np.int64),
        hold_indices=np.asarray([2, 3], dtype=np.int64),
        source_groups=groups,
        seed=23,
    )
    patch_mean = np.arange(4 * TOKEN_WIDTH, dtype=np.float32).reshape(4, TOKEN_WIDTH)
    residual = np.arange(
        4 * PREFIX_TOKENS * TOKEN_WIDTH, dtype=np.float32
    ).reshape(4, PREFIX_TOKENS, TOKEN_WIDTH) / 1000.0
    prefixes = (patch_mean[:, None, :] + residual).astype(np.float16)
    patch_mean_cache = patch_mean.astype(np.float16)
    global_logits = np.arange(
        4 * GLOBAL_LOGIT_WIDTH, dtype=np.float32
    ).reshape(4, GLOBAL_LOGIT_WIDTH)
    rows = expand_pair_rows(labels)
    dataset = _PairPrefixDataset(
        prefix_cache=prefixes,
        patch_mean_cache=patch_mean_cache,
        global_logits=global_logits,
        rows=rows,
        mode="deranged",
        derangement_sources=source_map,
    )
    row_index = int(np.flatnonzero(rows["sample_indices"] == 1)[0])
    observed_prefixes, observed_mean, observed_logits, *_rest = dataset[row_index]
    donor = int(source_map[1, 0])
    expected_residual = (
        prefixes[donor].astype(np.float32)
        - patch_mean_cache[donor].astype(np.float32)[None, :]
    )
    assert np.allclose(
        observed_prefixes.numpy() - observed_mean.numpy()[None, :],
        expected_residual,
    )
    assert np.array_equal(observed_mean.numpy(), patch_mean_cache[1].astype(np.float32))
    assert np.array_equal(observed_logits.numpy(), global_logits[1])


def test_primal_dual_formula_update_and_absent_positive_pair() -> None:
    logits = torch.zeros(6, dtype=torch.float32)
    targets = torch.as_tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
    pair_ids = torch.as_tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    weights = torch.ones(6, dtype=torch.float32)
    dual = torch.as_tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    loss, bce, violations, active = primal_dual_soft_recall_loss(
        logits=logits,
        targets=targets,
        pair_ids=pair_ids,
        weights=weights,
        dual_values=dual,
    )
    assert torch.allclose(violations, torch.full((3,), SOFT_RECALL_TARGET - 0.5))
    assert active.tolist() == [0, 1, 2]
    expected_constraint = (
        dual * violations + 0.5 * violations.square()
    ).mean()
    assert torch.allclose(loss, bce + expected_constraint)
    updated = update_dual_values(dual, violations, active)
    assert torch.allclose(updated, dual + 0.05 * violations)

    zero_targets = torch.zeros(3, dtype=torch.float32)
    no_constraint, no_constraint_bce, empty_g, empty_active = (
        primal_dual_soft_recall_loss(
            logits=torch.zeros(3),
            targets=zero_targets,
            pair_ids=torch.arange(3),
            weights=torch.ones(3),
            dual_values=dual,
        )
    )
    assert torch.equal(no_constraint, no_constraint_bce)
    assert empty_g.numel() == 0 and empty_active.numel() == 0
    assert torch.equal(update_dual_values(dual, empty_g, empty_active), dual)


def test_training_telemetry_locks_reset_support_and_finite_final_epoch() -> None:
    labels = np.asarray([0, 1, 2, 4, 0, 1, 2, 4], dtype=np.int64)
    rows = expand_pair_rows(labels)
    rng = np.random.default_rng(31)
    prefixes = rng.normal(
        size=(labels.size, PREFIX_TOKENS, TOKEN_WIDTH)
    ).astype(np.float16)
    patch_mean = rng.normal(size=(labels.size, TOKEN_WIDTH)).astype(np.float16)
    logits = rng.normal(size=(labels.size, GLOBAL_LOGIT_WIDTH)).astype(np.float32)
    dataset = _PairPrefixDataset(
        prefix_cache=prefixes,
        patch_mean_cache=patch_mean,
        global_logits=logits,
        rows=rows,
        mode="candidate",
    )
    _model, telemetry = _train_readout(
        dataset=dataset,
        device=torch.device("cpu"),
        seed=101,
        epochs=1,
        batch_size=len(dataset),
    )
    assert telemetry["initial_dual_values"] == [0.0] * PAIR_COUNT
    assert telemetry["seed"] == telemetry["data_order_seed"] == 101
    assert all(value > 0 for value in telemetry["active_constraint_updates_by_pair"])
    assert all(value > 0 for value in telemetry["final_epoch_active_batches_by_pair"])
    assert telemetry["successful_optimizer_updates"] == 1
    assert telemetry["nonfinite_steps"] == 0
    assert telemetry["dual_values_finite"]
    assert np.isfinite(telemetry["final_dual_values"]).all()
    assert np.isfinite(telemetry["final_epoch_mean_soft_recall_by_pair"]).all()
    assert np.isfinite(telemetry["final_epoch_mean_violation_by_pair"]).all()


def test_cache_loader_rejects_actual_dtype_even_with_matching_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b7, "EXPECTED_TRAIN_SAMPLES", 4)
    paths = [str(tmp_path / f"image_{index}.jpg") for index in range(4)]
    groups = np.asarray(["a", "b", "c", "d"], dtype=object)
    labels = np.asarray([0, 1, 2, 4], dtype=np.int64)
    cache_paths = _cache_paths(tmp_path)
    # Wrong float32 dtype is deliberately paired with a truthful file hash.
    np.save(cache_paths["prefixes"], np.zeros((4, PREFIX_TOKENS, TOKEN_WIDTH), dtype=np.float32))
    np.save(cache_paths["patch_mean"], np.zeros((4, TOKEN_WIDTH), dtype=np.float16))
    np.save(cache_paths["global_logits"], np.zeros((4, GLOBAL_LOGIT_WIDTH), dtype=np.float32))
    np.save(cache_paths["labels"], labels)
    manifest = {
        **_cache_manifest_expected(
            paths=paths, source_groups=groups, expected_labels=labels
        ),
        "maximum_pool_parity_error": 0.0,
        "maximum_prefix_float16_quantization_error": 0.0,
        "maximum_patch_mean_float16_quantization_error": 0.0,
        "prefixes_sha256": _sha256(cache_paths["prefixes"]),
        "patch_mean_sha256": _sha256(cache_paths["patch_mean"]),
        "global_logits_sha256": _sha256(cache_paths["global_logits"]),
        "labels_sha256": _sha256(cache_paths["labels"]),
    }
    cache_paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cache array contract mismatch"):
        _load_existing_cache(
            output_dir=tmp_path,
            paths=paths,
            source_groups=groups,
            expected_labels=labels,
        )


def _passing_readiness_inputs():
    pair_rows = []
    fold_rows = []
    for rival in PAIR_RIVALS:
        pair_rows.append(
            {
                "pair": f"{rival}-1",
                "rival_class": rival,
                "delta_auroc": 0.02,
                "delta_f1_class1": 0.02,
                "delta_recall_class1": -0.005,
                "candidate_deranged_delta_auroc": 0.01,
                "control_tp": 100.0,
                "candidate_tp": 99.0,
                "control_fp": 50.0,
                "candidate_fp": 45.0,
                "rival_fp_reduction": 0.10,
            }
        )
        for fold in range(FOLDS):
            fold_rows.append(
                {
                    "pair": f"{rival}-1",
                    "rival_class": rival,
                    "fold": fold,
                    "source_overlap": 0,
                    "candidate_auroc": 0.72,
                    "control_auroc": 0.70,
                    "deranged_auroc": 0.71,
                    "candidate_control_delta_auroc": 0.02,
                    "candidate_deranged_delta_auroc": 0.01,
                }
            )
    global_rows = [{"fold": fold, "source_overlap": 0} for fold in range(FOLDS)]
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_group_count": EXPECTED_SOURCE_GROUPS,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    integrity = {
        "derangement": {
            "observed_bijective_permutation_contracts": 2 * FOLDS,
            "same_source_group_assignments": 0,
            "fold_boundary_crossings": 0,
            "fixed_points": 0,
            "all_block_donors_consistent": True,
            "label_blind_construction": True,
        },
        "training_telemetry": [
            {
                "fold": fold,
                "seed": b7.SEED + fold,
                "readouts": {
                    mode: {
                        "seed": b7.SEED + fold,
                        "data_order_seed": b7.SEED + fold,
                        "initial_dual_values": [0.0] * PAIR_COUNT,
                        "final_dual_values": [0.1] * PAIR_COUNT,
                        "active_constraint_updates_by_pair": [10] * PAIR_COUNT,
                        "final_epoch_mean_soft_recall_by_pair": [0.96] * PAIR_COUNT,
                        "final_epoch_mean_violation_by_pair": [-0.01] * PAIR_COUNT,
                        "final_epoch_active_batches_by_pair": [10] * PAIR_COUNT,
                        "successful_optimizer_updates": 20,
                        "nonfinite_steps": 0,
                        "dual_values_finite": True,
                    }
                    for mode in ("candidate", "control", "deranged")
                },
            }
            for fold in range(FOLDS)
        ],
    }
    runtime = {"mismatches": {}, "observed": {"synthetic_pool_parity_error": 0.0}}
    return pair_rows, fold_rows, global_rows, manifest, integrity, runtime


def test_readiness_locks_all_incremental_and_derangement_gates() -> None:
    pairs, folds, global_rows, manifest, integrity, runtime = (
        _passing_readiness_inputs()
    )
    result = assess_prefix_residual_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_rows,
        train_samples=EXPECTED_TRAIN_SAMPLES,
        source_groups=EXPECTED_SOURCE_GROUPS,
        architecture=architecture_self_check(),
        runtime_contract=runtime,
        cache_manifest=manifest,
        cache_finite=True,
        oof_integrity=integrity,
    )
    assert result["prefix_residual_alignment_ready"]
    assert result["failed_checks"] == []
    assert not result["validation_permission"]
    assert not result["test_permission"]

    pairs[0]["candidate_deranged_delta_auroc"] = -0.02
    folds[0]["candidate_deranged_delta_auroc"] = -0.02
    rejected = assess_prefix_residual_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_rows,
        train_samples=EXPECTED_TRAIN_SAMPLES,
        source_groups=EXPECTED_SOURCE_GROUPS,
        architecture=architecture_self_check(),
        runtime_contract=runtime,
        cache_manifest=manifest,
        cache_finite=True,
        oof_integrity=integrity,
    )
    assert not rejected["prefix_residual_alignment_ready"]
    assert "candidate_beats_deranged_mean" in rejected["failed_checks"]


def test_cli_and_launcher_keep_default_path_preflight_only() -> None:
    args = _parse_args(
        [
            "--data",
            "train.yaml",
            "--checkpoint",
            "best.pt",
            "--output-dir",
            "run",
        ]
    )
    assert not args.preflight_only
    assert not hasattr(args, "validation")
    assert not hasattr(args, "test")
    assert not hasattr(args, "full_train")
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_pretrained_classf_b7_prefix_residual_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "[switch]$Run" in launcher
    assert "if (-not $Run)" in launcher
    assert '$Arguments += "--preflight-only"' in launcher
    assert "feature extraction and head training are disabled" in launcher
    assert "No validation/test dataset" in launcher

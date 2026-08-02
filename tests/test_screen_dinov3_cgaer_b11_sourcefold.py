from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from trkh.tools.screen_dinov3_cgaer_b11_sourcefold import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CGAER_B11_PARAMETER_COUNT,
    EPOCHS,
    FOLDS,
    _build_paired_bridges,
    assess_readiness,
    component_conflict_targets,
    conflict_weighted_loss,
    fold_stratified_component_bootstrap,
    metrics_all_fp,
    tempered_all_class_indices,
    validate_oof_artifact,
)


def test_all_class_sampler_is_deterministic_and_paired() -> None:
    labels = np.repeat(np.arange(5), [70, 20, 45, 80, 100]).astype(np.int64)
    fit = np.arange(labels.size, dtype=np.int64)
    left = tempered_all_class_indices(fit, labels, seed=123, batch_size=32)
    right = tempered_all_class_indices(fit, labels, seed=123, batch_size=32)
    np.testing.assert_array_equal(left, right)
    assert left.size % 32 == 0
    assert set(labels[left].tolist()) == set(range(5))

    control, candidate, contract = _build_paired_bridges(77)
    assert contract["identical_initial_state"]
    assert contract["control_parameters"] == contract["candidate_parameters"] == CGAER_B11_PARAMETER_COUNT
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)


def test_conflict_targets_and_inverse_component_masked_loss() -> None:
    # group 0 singleton (masked); group 1 pure; group 2 has 2:1 label split.
    labels = np.asarray([4, 0, 0, 1, 1, 2], dtype=np.int64)
    groups = np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int64)
    conflict = component_conflict_targets(labels, groups)
    assert not conflict["eligible"][0]
    assert conflict["targets"][1] == 0.0
    assert conflict["targets"][3] == pytest.approx(8.0 / 9.0)
    assert conflict["inverse_sizes"][1] == pytest.approx(0.5)
    assert conflict["inverse_sizes"][3] == pytest.approx(1.0 / 3.0)

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    target = torch.tensor([0, 1, 1])
    conflict_logits = torch.tensor([100.0, -1.0, 2.0])
    q = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([False, True, True])
    weights = torch.tensor([0.0, 0.5, 0.25])
    total, ce, bce = conflict_weighted_loss(logits, target, conflict_logits, q, mask, weights)
    expected_bce = (
        F.binary_cross_entropy_with_logits(conflict_logits[1], q[1]) * 0.5
        + F.binary_cross_entropy_with_logits(conflict_logits[2], q[2]) * 0.25
    ) / 0.75
    torch.testing.assert_close(ce, F.cross_entropy(logits, target))
    torch.testing.assert_close(bce, expected_bce)
    torch.testing.assert_close(total, ce + 0.10 * bce)


def _bootstrap_fixture():
    labels, folds, groups, mixed, eligible = [], [], [], [], []
    for fold in range(FOLDS):
        for local_group in range(20):
            group = fold * 20 + local_group
            if local_group < 10:
                group_labels = [0, 1, 2, 3, 4]
                is_mixed = True
            else:
                group_labels = [local_group % 5] * 5
                is_mixed = False
            labels.extend(group_labels)
            folds.extend([fold] * 5)
            groups.extend([group] * 5)
            mixed.extend([is_mixed] * 5)
            eligible.extend([True] * 5)
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    base = np.full((labels.size, 5), -1.0, dtype=np.float32)
    control = base.copy()
    candidate = base.copy()
    force = base.copy()
    for logits, strength in ((base, 1.0), (control, 0.7), (candidate, 2.0), (force, 1.5)):
        logits[np.arange(labels.size), labels] = strength
    # Every mixed component has both a C1 TP and a 0->1 boundary error.
    mixed_zero = np.asarray(mixed) & (labels == 0)
    candidate[mixed_zero, 1] = 2.5
    control[mixed_zero, 1] = 2.7
    base[mixed_zero, 1] = 2.6
    force[mixed_zero, 1] = 2.8
    gates = np.where(np.asarray(mixed), 0.9, 0.1).astype(np.float32)
    gates[mixed_zero] = 0.99
    return labels, folds, groups, base, control, candidate, force, gates, np.asarray(eligible), np.asarray(mixed)


def test_bootstrap_resamples_whole_components_and_is_hash_deterministic(tmp_path: Path) -> None:
    labels, folds, groups, base, control, candidate, force, gates, eligible, mixed = _bootstrap_fixture()
    artifact = tmp_path / "oof.npz"
    np.savez(artifact, labels=labels)
    kwargs = dict(
        labels=labels, folds=folds, groups=groups, base_logits=base,
        control_logits=control, candidate_logits=candidate, force_logits=force,
        candidate_gates=gates, conflict_eligible=eligible, conflict_mixed=mixed,
        oof_artifact=artifact, replicates=24, seed=91,
    )
    first = fold_stratified_component_bootstrap(**kwargs)
    second = fold_stratified_component_bootstrap(**kwargs)
    assert first == second
    assert len(first["draws_int64_sha256"]) == 64
    crossed = groups.copy()
    crossed[100] = crossed[0]
    with pytest.raises(ValueError, match="crosses held folds"):
        fold_stratified_component_bootstrap(**{**kwargs, "groups": crossed})


def _passing_gate_inputs():
    base = {"macro_f1": 0.90, "class1_f1": 0.90, "class1_tp": 100, "total_fp_to_class1": 100,
            "fp_to_class1": {"0": 25, "2": 25, "3": 25, "4": 25}}
    control = {"macro_f1": 0.90, "class1_f1": 0.90, "class1_tp": 99, "total_fp_to_class1": 90,
               "fp_to_class1": {"0": 23, "2": 23, "3": 22, "4": 22}}
    candidate = {"macro_f1": 0.901, "class1_f1": 0.91, "class1_tp": 99, "total_fp_to_class1": 85,
                 "fp_to_class1": {"0": 22, "2": 22, "3": 21, "4": 20}}
    force = {"class1_f1": 0.905, "total_fp_to_class1": 86}
    screen = {
        "base": base, "control": control, "candidate": candidate,
        "candidate_force_gate_0_5": force, "fold_wins": 4,
        "gate_aurocs": {"candidate": {"mixed_vs_pure": 0.8, "class1_boundary_error": 0.8}},
        "residual_l2_norms": {"candidate": {"p95": 0.1}},
        "preflight_artifact_verified": True, "immutable_a0_cache_verified": True,
        "locked_assignment_verified": True, "branch_off_max_abs_error": 0.0,
        "oof_artifact_validated": True, "completed_updates": 100,
        "expected_updates": 100, "skipped_updates": 0, "nonfinite_updates": 0,
        "oof_artifact_sha256": "a" * 64,
        "group_vector_int64_sha256": "b" * 64,
        "fold_vector_int64_sha256": "c" * 64,
        "training_rows": [{"class_counts": [1, 1, 1, 1, 1]} for _ in range(FOLDS * EPOCHS)],
        "paired_contracts": [{"control_parameters": CGAER_B11_PARAMETER_COUNT,
                              "candidate_parameters": CGAER_B11_PARAMETER_COUNT,
                              "identical_initial_state": True} for _ in range(FOLDS)],
    }
    interval = lambda point, lower, upper: {"point": point, "lower": lower, "upper": upper}
    bootstrap = {
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "oof_artifact_sha256": "a" * 64,
        "group_vector_int64_sha256": "b" * 64,
        "fold_vector_int64_sha256": "c" * 64,
        "class1_f1_delta_vs_b9": interval(.01, .001, .02),
        "class1_f1_delta_vs_control": interval(.01, .001, .02),
        "macro_f1_delta_vs_b9": interval(.001, -.001, .003),
        "macro_f1_delta_vs_control": interval(.001, -.001, .003),
        "fp_rate_delta_vs_b9": interval(-.01, -.02, -.001),
        "fp_rate_delta_vs_control": interval(-.005, -.01, 0.0),
        "mixed_vs_pure_gate_auroc": interval(.80, .71, .88),
        "class1_boundary_error_gate_auroc": interval(.80, .70, .90),
        "active_vs_force_class1_f1": interval(.005, .001, .01),
    }
    return screen, bootstrap


def test_all_preregistered_gates_fail_closed() -> None:
    screen, bootstrap = _passing_gate_inputs()
    passed = assess_readiness(screen, bootstrap)
    assert passed["train_only_screen_passed"]
    assert not passed["full_validation_permission"] and not passed["test_permission"]
    bootstrap["mixed_vs_pure_gate_auroc"]["lower"] = 0.69
    failed = assess_readiness(screen, bootstrap)
    assert failed["exact_b11_closed"]
    assert "mixed_pure_gate_auroc" in failed["failed_checks"]


def test_metrics_include_class3_fp_and_confusion() -> None:
    labels = np.arange(5, dtype=np.int64)
    logits = np.eye(5, dtype=np.float32)
    logits[3] = [0, 2, 0, 1, 0]
    result = metrics_all_fp(labels, logits)
    assert result["fp_to_class1"]["3"] == 1
    assert result["confusion_matrix"][3][1] == 1


def test_oof_artifact_rejects_missing_and_nan(tmp_path: Path) -> None:
    path = tmp_path / "oof.npz"
    np.savez(path, labels=np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="fields missing"):
        validate_oof_artifact(path, expected_samples=3)
    arrays = {
        "relative_paths": np.asarray(["a", "b", "c"]),
        "labels": np.zeros(3, np.int64), "folds": np.zeros(3, np.int64),
        "union_groups": np.arange(3), "base_logits": np.zeros((3, 5), np.float32),
        "control_logits": np.zeros((3, 5), np.float32),
        "candidate_logits": np.zeros((3, 5), np.float32),
        "candidate_force_gate_logits": np.zeros((3, 5), np.float32),
        "control_gates": np.zeros(3, np.float32), "candidate_gates": np.zeros(3, np.float32),
        "control_residual_l2_norms": np.zeros(3, np.float32),
        "candidate_residual_l2_norms": np.asarray([0.0, np.nan, 0.0], np.float32),
        "conflict_targets": np.zeros(3, np.float32),
        "conflict_eligible": np.zeros(3, bool), "conflict_mixed": np.zeros(3, bool),
        "inverse_component_sizes": np.zeros(3, np.float32),
    }
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="incomplete/invalid"):
        validate_oof_artifact(path, expected_samples=3)

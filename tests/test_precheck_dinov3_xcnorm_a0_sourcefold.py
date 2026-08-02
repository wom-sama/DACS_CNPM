from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.models.dinov3_xcnorm_pair_adapter_a0 import (
    XCNORM_PAIR_ADDED_PARAMETER_COUNT,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (
    EXPECTED_ASSIGNMENT_INT64_SHA256,
    EXPECTED_PATH_FOLD_SHA256,
    FOLDS,
    assert_train_only_paths,
    assess_readiness,
    assign_locked_folds,
    build_paired_adapters,
    build_train_union_groups,
    _flush_and_close_memmaps,
)


def _row(index: int, label: int) -> dict[str, object]:
    return {
        "relative_path": f"train/class_{label}/image_{index}_box000.jpg",
        "label": label,
        "source": "source_a",
        "source_split": "train",
        "source_root": f"image_{index}",
        "source_prefix": "image",
        "source_number": index * 10,
        "leakage_group": f"declared_{index}",
    }


def test_union_groups_join_each_locked_train_relation() -> None:
    rows = [_row(index, index % 5) for index in range(8)]
    rows[1]["source_root"] = rows[0]["source_root"]
    rows[3]["leakage_group"] = rows[2]["leakage_group"]
    rows[5]["source_number"] = int(rows[4]["source_number"]) + 1
    # 0b1 is within pHash radius three of zero, while the high bits are far.
    phashes = [
        0,
        0xFFFFFFFFFFFFFFFF,
        0xFFFF0000FFFF0000,
        0x0000FFFF0000FFFF,
        0xAAAAAAAAAAAAAAAA,
        0x5555555555555555,
        0b1,
        0x3333333333333333,
    ]
    groups, stats = build_train_union_groups(rows, phashes)
    assert groups[0] == groups[1]  # same source crop root
    assert groups[2] == groups[3]  # declared leakage_group
    assert groups[4] == groups[5]  # numeric adjacency in one provenance cohort
    assert groups[0] == groups[6]  # pHash Hamming <= 3
    assert int(stats["components"]) == 4


def test_locked_folds_have_all_classes_and_zero_group_overlap() -> None:
    rows = [_row(index, index % 5) for index in range(100)]
    groups = np.asarray([f"g{index}" for index in range(100)], dtype=object)
    assignments, fold_rows, hashes = assign_locked_folds(rows, groups)
    assert set(assignments.tolist()) == set(range(FOLDS))
    assert all(row["group_overlap"] == 0 for row in fold_rows)
    assert all(all(value > 0 for value in row["held_class_counts"]) for row in fold_rows)
    assert len(hashes["assignment_int64_sha256"]) == 64
    assert len(hashes["path_fold_sha256"]) == 64


def test_train_only_guard_rejects_val_and_test(tmp_path: Path) -> None:
    train = tmp_path / "class_f" / "train"
    assert_train_only_paths([train / "class0" / "a.jpg"], train)
    with pytest.raises(ValueError, match="not inside canonical train"):
        assert_train_only_paths([tmp_path / "class_f" / "val" / "class0" / "a.jpg"], train)
    with pytest.raises(ValueError, match="not inside canonical train"):
        assert_train_only_paths([tmp_path / "class_f" / "test" / "class0" / "a.jpg"], train)


def test_deterministic_cuda_workspace_is_set_before_runtime() -> None:
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_memmap_is_closed_before_atomic_promotion(tmp_path: Path) -> None:
    partial = tmp_path / "cache.npy.partial"
    final = tmp_path / "cache.npy"
    values = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(3,),
    )
    values[:] = [1.0, 2.0, 3.0]
    mapping = values._mmap
    _flush_and_close_memmaps(values)
    assert mapping.closed
    del values

    partial.replace(final)
    np.testing.assert_array_equal(
        np.load(final, allow_pickle=False),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_paired_adapters_are_identical_capacity_and_head_is_frozen() -> None:
    generator = torch.Generator().manual_seed(123)
    classifier_weight = torch.randn(5, 384, generator=generator)
    classifier_weight.requires_grad_(False)
    control, candidate, contract = build_paired_adapters(classifier_weight, seed=77)
    assert contract["paired_initial_state_sha256"]
    assert not contract["classifier_weight_requires_grad"]
    assert control.added_parameter_count() == XCNORM_PAIR_ADDED_PARAMETER_COUNT
    assert candidate.added_parameter_count() == XCNORM_PAIR_ADDED_PARAMETER_COUNT
    assert all(parameter.requires_grad for parameter in control.parameters())
    assert all(
        torch.equal(control.state_dict()[key], candidate.state_dict()[key])
        for key in control.state_dict()
    )


def _passing_screen() -> dict[str, object]:
    base = {
        "macro_f1": 0.80,
        "class1_f1": 0.70,
        "class1_tp": 100,
        "restricted_fp": 100,
        "fp_to_class1": {"0": 30, "2": 50, "4": 20},
    }
    candidate = {
        "macro_f1": 0.799,
        "class1_f1": 0.72,
        "class1_tp": 99,
        "restricted_fp": 90,
        "fp_to_class1": {"0": 27, "2": 45, "4": 18},
    }
    return {
        "base": base,
        "candidate": candidate,
        "pair_auroc_gains": {"0": 0.021, "2": 0.021, "4": 0.021},
        "mean_pair_auroc_gain": 0.021,
        "fold_wins": 4,
        "residual_ratio_p95": 0.039,
        "branch_off_max_abs_error": 0.0,
        "nonfinite_updates": 0,
        "skipped_updates": 0,
        "completed_updates": 100,
        "expected_updates": 100,
        "assignment_hashes": {
            "assignment_int64_sha256": EXPECTED_ASSIGNMENT_INT64_SHA256,
            "path_fold_sha256": EXPECTED_PATH_FOLD_SHA256,
        },
        "group_fold_contract": [
            {"fold": fold, "group_overlap": 0} for fold in range(FOLDS)
        ],
        "paired_contracts": [
            {
                "control_parameters": XCNORM_PAIR_ADDED_PARAMETER_COUNT,
                "candidate_parameters": XCNORM_PAIR_ADDED_PARAMETER_COUNT,
            }
            for _ in range(FOLDS)
        ],
    }


def test_readiness_fails_closed_until_photometric_train_recaches() -> None:
    deferred = assess_readiness(_passing_screen())
    assert deferred["clean_train_gates_passed"]
    assert deferred["photometric_recache_permission"]
    assert not deferred["full_validation_permission"]
    assert deferred["failed_checks"] == ["photometric_train_recaches_complete"]


def test_readiness_rejects_class2_false_positive_regression() -> None:
    screen = _passing_screen()
    screen["candidate"]["fp_to_class1"]["2"] = 51
    result = assess_readiness(screen)
    assert not result["full_validation_permission"]
    assert "b9_2_to_1_not_increased" in result["failed_checks"]

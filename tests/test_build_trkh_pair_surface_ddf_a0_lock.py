from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_trkh_pair_surface_ddf_a0_lock.py"


def _module():
    spec = importlib.util.spec_from_file_location("ddf_a0_lock_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ddf_parameter_oracles_are_exact_and_static_is_matched() -> None:
    module = _module()
    oracle = module._parameter_oracles()

    assert oracle["ddf_full"] == 9380
    assert oracle["static_matched"] == 9435
    assert oracle["static_minus_ddf"] == 55
    assert 0.0 <= oracle["static_relative_excess"] <= 0.05


def test_ddf_bbox_rasterization_is_nonempty_bounded_and_stable() -> None:
    module = _module()
    cohort = module._load_cohort()
    masks = module._bbox_masks(cohort["model_boxes"])
    replay = module._bbox_masks(cohort["model_boxes"])

    assert masks.shape == (763, 16, 16)
    assert masks.dtype == np.bool_
    assert int(masks.sum(axis=(1, 2)).min()) >= 1
    assert int(masks.sum(axis=(1, 2)).max()) <= 256
    assert np.array_equal(masks, replay)
    assert module._array_sha256(masks) == module._array_sha256(replay)


def test_ddf_fold_orders_partners_and_offsets_are_locked() -> None:
    module = _module()
    cohort = module._load_cohort()
    folds = cohort["folds"].astype(np.int64)
    targets = cohort["targets"].astype(np.int64)
    probabilities = cohort["keeper_probabilities"].astype(np.float32)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    samples = cohort["sample_indices"].astype(np.int64)
    sources = cohort["source_stems"].astype(str)

    payloads = [
        module._fold_payload(
            fold,
            folds=folds,
            targets=targets,
            predictions=predictions,
            sample_indices=samples,
            sources=sources,
        )
        for fold in range(5)
    ]

    assert [payload["calibration_fold"] for payload in payloads] == [
        2,
        3,
        4,
        2,
        3,
    ]
    assert all(payload["source_overlap"] == 0 for payload in payloads)
    assert all(payload["sattolo"]["fixed_points"] == 0 for payload in payloads)
    assert all(
        payload["sattolo"]["same_source_pairs"] == 0 for payload in payloads
    )
    assert all(payload["dephase"]["zero_pairs"] == 0 for payload in payloads)
    assert all(
        payload["dephase"]["shape"][1:] == [2, 2] for payload in payloads
    )
    assert all(
        len(payload["primary_orders"]["per_epoch_sha256"]) == 20
        for payload in payloads
    )
    assert all(
        len(payload["repeat_orders"]["per_epoch_sha256"]) == 20
        for payload in payloads
    )
    assert sum(payload["held"]["rows"] for payload in payloads) == 763
    assert all(
        payload["calibration"]["target_counts"]["4"] == 3
        for payload in payloads
    )
    assert all(
        payload["fit"]["target_counts"]["4"] >= 4 for payload in payloads
    )


def test_ddf_keeper_information_matches_locked_boundary() -> None:
    module = _module()
    cohort = module._load_cohort()
    information = module._keeper_information(
        cohort["targets"],
        cohort["keeper_probabilities"],
    )

    assert information["definition"] == "log_p1_minus_log_max_p0_p2_p4"
    assert information["union_auroc"] == pytest.approx(
        0.8166974738139248, abs=1e-15
    )
    assert information["union_auprc"] == pytest.approx(
        0.921070118462715, abs=1e-15
    )
    assert information["pairs"]["1_vs_0"]["auroc"] == pytest.approx(
        0.8358875968085356, abs=1e-15
    )
    assert information["pairs"]["1_vs_2"]["auroc"] == pytest.approx(
        0.9328404189772027, abs=1e-15
    )
    assert information["pairs"]["1_vs_4"]["auroc"] == pytest.approx(
        0.9266173752310535, abs=1e-15
    )


def test_ddf_lock_is_prospective_and_contains_exact_gates() -> None:
    module = _module()
    lock = module.build_lock()

    assert (
        lock["state"]
        == "prospective_no_candidate_implementation_or_observation"
    )
    assert lock["cache"]["rows"] == 763
    assert lock["cache"]["unique_sources"] == 735
    assert lock["cohort"]["counts"] == {
        "tp1": 528,
        "fn1": 13,
        "restricted_fp": 222,
        "0_to_1": 158,
        "2_to_1": 54,
        "4_to_1": 10,
    }
    assert lock["architecture"]["parameter_oracles"]["ddf_full"] == 9380
    assert lock["architecture"]["parameter_oracles"]["static_matched"] == 9435
    assert lock["optimization"]["epochs"] == 20
    assert lock["optimization"]["num_workers"] == 0
    assert lock["optimization"]["binary_loss"] == "bce_with_logits"
    assert lock["optimization"]["bbox_attention_clamp_min"] == 1e-8
    assert lock["scientific_gates"]["oof_restricted_fp_rejected"] == 45
    assert lock["scientific_gates"]["full_class1_precision_gain"] == 0.030
    assert lock["replay"]["numeric_arrays_max_abs"] == 0.0
    assert lock["access"]["formal_runs_authorized"] == 0
    assert lock["access"]["replay_runs_authorized"] == 0
    assert lock["forbidden"]["validation"] is True
    assert lock["forbidden"]["test"] is True
    assert all(value is None for value in lock["observations"].values())
    assert lock["visual"]["rows"] == 30
    assert lock["visual"]["all_target4_rows"] == 10


def test_ddf_lock_write_guard_requires_parent_and_expected_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    expected_paths = [
        *(
            module._relative(path)
            for path in module.PROTECTED_UNTRACKED
        ),
        *(module._relative(path) for path in module.PROSPECTIVE_PATHS),
    ]

    def matching_git(*args):
        if args[:2] == ("branch", "--show-current"):
            return module.BRANCH
        return module.LOCK_PARENT_COMMIT

    monkeypatch.setattr(module, "_git", matching_git)
    monkeypatch.setattr(module, "_status_paths", lambda: expected_paths)
    module._assert_lock_parent_for_write()

    monkeypatch.setattr(module, "_git", lambda *args: "0" * 40)
    with pytest.raises(RuntimeError, match="Use --check-only"):
        module._assert_lock_parent_for_write()


def test_ddf_check_only_replays_complete_frozen_payload() -> None:
    module = _module()
    lock = module.build_lock()

    module._assert_check_only_matches_frozen(lock)
    sha_parts = module.LOCK_SHA_PATH.read_text(encoding="ascii").split()
    assert sha_parts == [
        module._sha256(module.LOCK_PATH),
        module.LOCK_PATH.name,
    ]

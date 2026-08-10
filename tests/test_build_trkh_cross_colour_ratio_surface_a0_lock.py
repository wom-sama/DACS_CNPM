from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "build_trkh_cross_colour_ratio_surface_a0_lock.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("ccr_lock_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ccr_direct_ratio_log_identity_and_photometric_cancellation() -> None:
    rng = np.random.default_rng(20260725)
    reflectance1 = rng.uniform(0.08, 0.72, size=(128, 3))
    reflectance2 = rng.uniform(0.08, 0.72, size=(128, 3))
    geometry1 = rng.uniform(0.55, 1.0, size=(128, 1))
    geometry2 = rng.uniform(0.55, 1.0, size=(128, 1))
    illuminant = rng.uniform(0.75, 1.15, size=(128, 3))
    pixel1 = reflectance1 * geometry1 * illuminant
    pixel2 = reflectance2 * geometry2 * illuminant

    direct_rg = (pixel1[:, 0] * pixel2[:, 1]) / (
        pixel2[:, 0] * pixel1[:, 1]
    )
    log_rg = (
        np.log(pixel1[:, 0])
        - np.log(pixel1[:, 1])
        - np.log(pixel2[:, 0])
        + np.log(pixel2[:, 1])
    )
    expected = (
        reflectance1[:, 0] * reflectance2[:, 1]
    ) / (reflectance2[:, 0] * reflectance1[:, 1])

    assert np.max(np.abs(np.log(direct_rg) - log_rg)) <= 2e-15
    assert np.max(np.abs(direct_rg - expected)) <= 5e-15

    reverse = (
        np.log(pixel2[:, 0])
        - np.log(pixel2[:, 1])
        - np.log(pixel1[:, 0])
        + np.log(pixel1[:, 1])
    )
    assert np.max(np.abs(log_rg + reverse)) <= 2e-15


def test_ccr_dephase_offsets_are_nonzero_bounded_and_hash_stable() -> None:
    module = _module()
    geometry = module._load_geometry()
    offsets = module._dephase_offsets(geometry["sample_indices"])
    replay = module._dephase_offsets(geometry["sample_indices"])

    assert offsets.shape == (763, 6, 2)
    assert offsets.dtype == np.int64
    assert int(offsets.min()) >= 1
    assert int(offsets.max()) < 96
    assert np.array_equal(offsets, replay)
    assert module._array_sha256(offsets) == module._array_sha256(replay)


def test_ccr_folds_are_source_disjoint_and_orders_are_complete() -> None:
    module = _module()
    geometry = module._load_geometry()
    folds = geometry["folds"].astype(np.int64)
    targets = geometry["targets"].astype(np.int64)
    probabilities = geometry["keeper_probabilities"].astype(np.float32)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    samples = geometry["sample_indices"].astype(np.int64)
    sources = geometry["source_stems"].astype(str).tolist()

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
        1,
        2,
        3,
        4,
        0,
    ]
    assert all(payload["source_overlap"] == 0 for payload in payloads)
    assert all(
        len(payload["primary_orders"]["per_epoch_sha256"]) == 20
        for payload in payloads
    )
    assert all(
        len(payload["repeat_orders"]["per_epoch_sha256"]) == 20
        for payload in payloads
    )
    assert sum(payload["held"]["rows"] for payload in payloads) == 763
    assert sum(
        payload["held"]["target_counts"]["4"] for payload in payloads
    ) == 10


def test_ccr_visual_rows_cover_each_main_stratum_and_all_target4_rows() -> None:
    module = _module()
    geometry = module._load_geometry()
    samples = geometry["sample_indices"].astype(np.int64)
    targets = geometry["targets"].astype(np.int64)
    folds = geometry["folds"].astype(np.int64)
    predictions = geometry["keeper_probabilities"].argmax(axis=1)
    selected = module.VISUAL_SAMPLE_INDICES
    positions = np.asarray(
        [int(np.flatnonzero(samples == sample)[0]) for sample in selected],
        dtype=np.int64,
    )

    assert selected.shape == (30,)
    assert len(set(selected.tolist())) == 30
    for fold in range(5):
        current = positions[fold * 4 : (fold + 1) * 4]
        assert np.all(folds[current] == fold)
        pairs = [
            (int(targets[position]), int(predictions[position]))
            for position in current
        ]
        assert pairs[0] == (1, 1)
        assert pairs[1][0] == 1 and pairs[1][1] != 1
        assert pairs[2] == (0, 1)
        assert pairs[3] == (2, 1)

    target4_positions = positions[-10:]
    assert np.all(targets[target4_positions] == 4)
    assert np.all(predictions[target4_positions] == 1)
    expected_target4 = set(
        samples[(targets == 4) & (predictions == 1)].tolist()
    )
    assert set(selected[-10:].tolist()) == expected_target4


def test_ccr_lock_is_prospective_and_contains_exact_resource_boundary() -> None:
    module = _module()
    lock = module.build_lock()

    assert lock["state"] == "prospective_no_candidate_observation"
    assert lock["cohort"]["rows"] == 763
    assert lock["cohort"]["counts"] == {
        "tp1": 528,
        "fn1": 13,
        "restricted_fp": 222,
        "0_to_1": 158,
        "2_to_1": 54,
        "4_to_1": 10,
    }
    assert lock["head"]["trainable_parameters"] == 3004
    assert lock["head"]["parameter_oracle"] == 3004
    assert lock["head"]["pretrained"] is False
    assert lock["head"]["bbox_is_input"] is False
    assert lock["optimization"]["epochs"] == 20
    assert lock["optimization"]["class_weight"] is None
    assert lock["optimization"]["oversampling"] is False
    assert lock["resource_limits"]["clean_wall_seconds"] == 1200
    assert lock["forbidden"]["validation"] is True
    assert lock["forbidden"]["test"] is True
    assert all(value is None for value in lock["observations"].values())
    assert lock["cohort"]["image_files"]["ordered_rows"] == 763
    assert lock["cohort"]["label_files"]["ordered_rows"] == 763
    assert lock["visual"]["rows"] == 30
    assert lock["visual"]["all_target4_rows"] == 10


def test_ccr_lock_write_guard_requires_exact_parent_and_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "_git",
        lambda *args: module.LOCK_PARENT_COMMIT,
    )
    module._assert_lock_parent_for_write()

    monkeypatch.setattr(module, "_git", lambda *args: "0" * 40)
    with pytest.raises(RuntimeError, match="Use --check-only"):
        module._assert_lock_parent_for_write()


def test_ccr_check_only_replays_complete_frozen_payload() -> None:
    module = _module()
    lock = module.build_lock()

    module._assert_check_only_matches_frozen(lock)
    sha_parts = module.LOCK_SHA_PATH.read_text(encoding="ascii").split()
    assert sha_parts == [
        module._sha256(module.LOCK_PATH),
        module.LOCK_PATH.name,
    ]

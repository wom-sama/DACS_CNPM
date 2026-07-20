from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools import audit_cutpaste_surface_response_a0 as cutpaste


def _synthetic_image() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinate = torch.linspace(0.0, 1.0, 256)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    image = torch.stack((xx, yy, 0.6 * xx + 0.4 * yy), dim=0)
    bbox = torch.tensor([0.5, 0.5, 0.70, 0.62])
    mask = torch.ones((256, 256), dtype=torch.bool)
    mask[:16] = False
    return image, bbox, mask


def test_regular_cutpaste_is_deterministic_paired_and_local_rng_only() -> None:
    image, bbox, mask = _synthetic_image()
    before = cutpaste._global_rng_snapshot()
    first_candidate, first_cutout, first_record = cutpaste.make_cutpaste_pair(
        image,
        bbox,
        mask,
        sample_index=31,
        draw=1,
        family="regular",
    )
    after = cutpaste._global_rng_snapshot()
    second_candidate, second_cutout, second_record = cutpaste.make_cutpaste_pair(
        image,
        bbox,
        mask,
        sample_index=31,
        draw=1,
        family="regular",
    )
    assert cutpaste._global_rng_equal(before, after)
    assert first_record == second_record
    assert torch.equal(first_candidate, second_candidate)
    assert torch.equal(first_cutout, second_cutout)
    assert first_record["candidate_destination_mask_sha256"] == first_record[
        "cutout_destination_mask_sha256"
    ]
    assert first_record["source_destination_iou"] <= 0.05
    assert first_record["identity_max_abs_error"] <= 1e-7
    assert first_record["candidate_changed_subset_destination"] is True
    assert first_record["cutout_changed_subset_destination"] is True
    assert first_record["support_algorithm"] == (
        "surface_mask_cropbbox_imagemask_ellipse_erode"
    )
    assert first_record["source_candidate_count"] > 0
    assert first_record["destination_candidate_count"] > 0


def test_scar_cutpaste_geometry_obeys_locked_ranges() -> None:
    image, bbox, mask = _synthetic_image()
    candidate, cutout, record = cutpaste.make_cutpaste_pair(
        image,
        bbox,
        mask,
        sample_index=89,
        draw=3,
        family="scar",
    )
    assert candidate.shape == image.shape
    assert cutout.shape == image.shape
    assert 2 <= record["scar_short_side"] <= 16
    assert 10 <= record["scar_long_side"] <= 25
    assert -45.0 <= record["angle_degrees"] <= 45.0
    assert record["source_fully_valid"] is True
    assert record["destination_fully_valid"] is True


def test_precomputed_surface_support_replays_pair_exactly() -> None:
    image, bbox, mask = _synthetic_image()
    support = cutpaste._bbox_support(image, bbox, mask)
    direct = cutpaste.make_cutpaste_pair(
        image, bbox, mask, sample_index=113, draw=0, family="regular"
    )
    cached = cutpaste.make_cutpaste_pair(
        image,
        bbox,
        mask,
        sample_index=113,
        draw=0,
        family="regular",
        support=support,
    )
    assert torch.equal(direct[0], cached[0])
    assert torch.equal(direct[1], cached[1])
    assert direct[2] == cached[2]


def test_placement_candidates_enumerate_every_fully_valid_location() -> None:
    support = torch.ones((5, 6), dtype=torch.bool)
    shape = torch.ones((2, 3), dtype=torch.bool)
    coordinates, _ = cutpaste._placement_candidates(support, shape)
    assert coordinates.tolist() == [
        [row, column] for row in range(4) for column in range(4)
    ]
    forbidden = torch.zeros_like(support)
    forbidden[:2, :3] = True
    filtered, _ = cutpaste._placement_candidates(
        support, shape, forbidden=forbidden, maximum_iou=0.05
    )
    assert filtered.tolist() == [
        [0, 3],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
    ]


def test_engineering_preflight_passes_without_dataset_or_model() -> None:
    result = cutpaste.cutpaste_engineering_checks()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert len(result["records"]) == 2


def test_response_scalars_are_zero_for_identity_and_finite_for_change() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((4, 5), generator=generator)
    probabilities = logits.softmax(dim=1)
    pooled = torch.randn((4, 12), generator=generator)
    identity = cutpaste.response_scalars(
        logits, probabilities, pooled, logits, probabilities, pooled
    )
    assert torch.allclose(identity, torch.zeros_like(identity), atol=2e-7)
    changed_logits = logits.clone()
    changed_logits[:, 1] += 0.25
    changed_probabilities = changed_logits.softmax(dim=1)
    changed_pooled = pooled + 0.05
    changed = cutpaste.response_scalars(
        logits,
        probabilities,
        pooled,
        changed_logits,
        changed_probabilities,
        changed_pooled,
    )
    assert changed.shape == (4, 5)
    assert torch.isfinite(changed).all()
    assert torch.allclose(changed[:, 0], torch.full((4,), 0.25), atol=1e-6)


def test_response_aggregation_has_locked_order_and_population_std() -> None:
    raw = np.zeros((2, 2, 4, 5), dtype=np.float64)
    raw[0, 0, :, 0] = [1.0, 2.0, 3.0, 4.0]
    raw[1, 1, :, 4] = [2.0, 2.0, 6.0, 10.0]
    result = cutpaste.aggregate_responses(raw)
    assert result.shape == (2, 40)
    assert np.allclose(result[0, :4], [2.5, np.std([1, 2, 3, 4]), 1.0, 4.0])
    assert np.allclose(
        result[1, -4:], [5.0, np.std([2, 2, 6, 10]), 2.0, 10.0]
    )


def test_cohort_index_hash_uses_locked_comma_serialization() -> None:
    expected = hashlib.sha256(b"2,58,103").hexdigest()
    assert cutpaste._cohort_index_sha256([2, 58, 103]) == expected


def test_source_derangement_is_fold_quartile_locked_and_different_source() -> None:
    rows = 80
    responses = np.arange(rows * 40, dtype=np.float64).reshape(rows, 40)
    folds = np.repeat(np.arange(5), rows // 5)
    margins = np.linspace(-1.0, 1.0, rows)
    indices = np.arange(rows, dtype=np.int64)
    sources = [f"source_{index}" for index in indices]
    first, first_mapping = cutpaste.build_source_derangement(
        responses,
        folds=folds,
        clean_margins=margins,
        source_stems=sources,
        sample_indices=indices,
    )
    second, second_mapping = cutpaste.build_source_derangement(
        responses,
        folds=folds,
        clean_margins=margins,
        source_stems=sources,
        sample_indices=indices,
    )
    assert np.array_equal(first, second)
    assert first_mapping == second_mapping
    assert len(first_mapping) == rows
    assert all(
        row["destination_source_stem"] != row["response_source_stem"]
        for row in first_mapping
    )


def _readout_fixture():
    generator = np.random.default_rng(20260720)
    rows = 250
    labels = np.tile(np.asarray([1] * 7 + [0] * 3, dtype=np.int64), rows // 10)
    folds = np.repeat(np.arange(5, dtype=np.int64), rows // 5)
    sources = [f"source_{index}" for index in range(rows)]
    signal = (2.0 * labels - 1.0)[:, None]
    features = {}
    for role_index, role in enumerate(cutpaste.ROLE_NAMES):
        width = 6 if role == "base_only" else 46
        value = generator.normal(0.0, 0.1, size=(rows, width))
        value[:, :1] += signal * (1.0 + role_index * 0.05)
        features[role] = value
    return features, labels, folds, sources


def test_oof_readout_converges_and_persisted_equations_replay() -> None:
    features, labels, folds, sources = _readout_fixture()
    scores, actions, states = cutpaste.fit_clean_oof_readouts(
        features, labels, folds, sources
    )
    replay_scores, replay_actions = cutpaste.apply_readout_states(
        features, folds, states
    )
    assert all(
        np.allclose(scores[role], replay_scores[role], atol=1e-12, rtol=0.0)
        for role in cutpaste.ROLE_NAMES
    )
    assert all(
        np.array_equal(actions[role], replay_actions[role])
        for role in cutpaste.ROLE_NAMES
    )
    assert cutpaste._readout_states_converged(states)
    assert all(
        fold["fit_tp_retention"] >= 0.97
        for role in cutpaste.ROLE_NAMES
        for fold in states[role]["folds"]
    )


def _analysis_fixture():
    targets = []
    folds = []
    for fold in range(5):
        targets.extend([1] * 20 + [0] * 8 + [2] * 8 + [4] * 2)
        folds.extend([fold] * 38)
    target_array = np.asarray(targets, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=np.int64)
    labels = (target_array == 1).astype(np.int64)
    rows = labels.size
    clean = np.full((rows, 5), 0.02, dtype=np.float64)
    clean[:, 1] = 0.55
    for index, target in enumerate(target_array):
        if target == 1:
            clean[index, 1] = 0.82
            clean[index, 0] = 0.12
        else:
            clean[index, target] = 0.37
    clean /= clean.sum(axis=1, keepdims=True)
    flat = np.full(rows, 0.5, dtype=np.float64)
    candidate = np.where(labels == 1, 0.9, 0.1)
    scores = {
        "base_only": flat.copy(),
        "cutout_control": flat.copy(),
        "cutpaste_candidate": candidate,
        "paired_contrast": candidate.copy(),
        "source_deranged": flat.copy(),
    }
    actions = {
        "base_only": np.ones(rows, dtype=bool),
        "cutout_control": np.ones(rows, dtype=bool),
        "cutpaste_candidate": labels.astype(bool),
        "paired_contrast": labels.astype(bool),
        "source_deranged": np.ones(rows, dtype=bool),
    }
    stats = {"effective_rank": 12.0}
    return labels, target_array, fold_array, scores, actions, clean, stats


def test_analysis_passes_only_for_selective_surface_response() -> None:
    labels, targets, folds, scores, actions, clean, stats = _analysis_fixture()
    result = cutpaste.build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=scores,
        actions=actions,
        clean_probabilities=clean,
        candidate_descriptor_stats=stats,
    )
    assert result["mechanism_gates_passed"] is True
    assert all(result["mechanism_gates"].values())


def test_analysis_rejects_precision_gain_from_broad_tp_suppression() -> None:
    labels, targets, folds, scores, actions, clean, stats = _analysis_fixture()
    candidate_actions = actions["cutpaste_candidate"].copy()
    candidate_actions[np.flatnonzero(labels == 1)[:60]] = False
    actions["cutpaste_candidate"] = candidate_actions
    result = cutpaste.build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=scores,
        actions=actions,
        clean_probabilities=clean,
        candidate_descriptor_stats=stats,
    )
    assert result["mechanism_gates_passed"] is False
    assert result["mechanism_gates"]["candidate_tp_retention_ge_095"] is False
    assert result["mechanism_gates"]["candidate_corrections_ge_2x_harms"] is False


def test_manifest_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "payload.txt").write_text("locked\n", encoding="utf-8")
    cutpaste._write_manifest(tmp_path)
    assert cutpaste._verify_manifest(tmp_path)["passed"] is True
    (tmp_path / "payload.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest payload differs"):
        cutpaste._verify_manifest(tmp_path)


def test_visual_review_cannot_rescue_failed_automatic_gate(tmp_path: Path) -> None:
    contact = tmp_path / "cutpaste_contact_sheet_clean.png"
    contact.write_bytes(b"locked-sheet")
    summary_path = tmp_path / "summary.json"
    cutpaste._write_json(
        summary_path,
        {
            "mode": "cutpaste_surface_response_a0_train_information_gate",
            "status": "rejected_automated_gate",
            "automated_gate_passed": False,
            "external_geometry_replay": {"passed": True},
            "contact_sheet": {"sha256": cutpaste._sha256(contact)},
            "visual_review": {"required": True, "completed": False},
        },
    )
    cutpaste._write_manifest(tmp_path)
    expected = cutpaste._sha256(summary_path)
    result = cutpaste.finalize_visual_review(
        summary_path, result="pass", expected_summary_sha256=expected
    )
    assert result["clean_a0_gates_passed"] is False
    assert result["stage_b_conditions_authorized"] is False
    assert result["full_train_authorized"] is False

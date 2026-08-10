from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.models.model import ConvStemBlock
from trkh.tools import audit_sparse_class_evidence_a0 as audit
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow


def test_evidence_logits_and_objectives_match_independent_numpy_fp64() -> None:
    generator = np.random.default_rng(91)
    maps = generator.normal(size=(5, 2, 7, 9)).astype(np.float64)
    targets = np.asarray([0, 1, 0, 1, 1], dtype=np.int64)
    tensor = torch.from_numpy(maps)
    assert np.max(
        np.abs(audit.evidence_logits_torch(tensor).numpy() - audit.evidence_logits_numpy(maps))
    ) <= 1e-12
    for regularizer in ("none", "map_l1", "logit_l1"):
        observed = audit.evidence_objective_torch(
            tensor,
            torch.from_numpy(targets),
            regularizer=regularizer,
            regularizer_lambda=0.137,
        )
        expected = audit.evidence_objective_numpy(
            maps,
            targets,
            regularizer=regularizer,
            regularizer_lambda=0.137,
        )
        assert max(abs(float(value) - oracle) for value, oracle in zip(observed, expected)) <= 1e-12


def test_map_l1_regularizes_every_batch_item_and_rejects_invalid_maps() -> None:
    maps = torch.randn(4, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    objective, _, _ = audit.evidence_objective_torch(
        maps,
        torch.tensor([0, 1, 0, 1]),
        regularizer="map_l1",
        regularizer_lambda=0.2,
    )
    objective.backward()
    assert maps.grad is not None
    assert torch.all(maps.grad.abs().flatten(1).sum(dim=1) > 0.0)
    with pytest.raises(ValueError, match="shape"):
        audit.evidence_logits_torch(torch.randn(2, 3, 5, 5))
    with pytest.raises(ValueError, match="finite"):
        invalid = torch.randn(2, 2, 5, 5)
        invalid[0, 0, 0, 0] = torch.nan
        audit.evidence_logits_torch(invalid)


def test_engineering_checks_cover_equations_symmetry_and_dephasing() -> None:
    result = audit.engineering_checks()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert max(result["errors"]["objectives"].values()) <= 1e-12


def test_channel_dephasing_is_deterministic_marginal_preserving_and_rng_local() -> None:
    generator = torch.Generator().manual_seed(13)
    features = torch.randn(
        2,
        audit.FEATURE_CHANNELS,
        audit.FEATURE_SIZE,
        audit.FEATURE_SIZE,
        generator=generator,
    )
    before = audit._global_rng_snapshot()
    first = audit.channel_dephase(features, [31, 87], fold=2)
    second = audit.channel_dephase(features, [31, 87], fold=2)
    after = audit._global_rng_snapshot()
    assert torch.equal(first, second)
    assert audit._global_rng_equal(before, after)
    assert torch.equal(
        torch.sort(first.flatten(2), dim=2).values,
        torch.sort(features.flatten(2), dim=2).values,
    )
    assert not torch.equal(first, features)
    dy, dx = audit._dephase_offsets([31, 87], fold=2)
    assert np.all((dy != 0) | (dx != 0))


def test_sparse_evidence_branch_has_locked_map_shape_and_exact_spatial_mean() -> None:
    branch = audit.SparseEvidenceBranch(ConvStemBlock(64, 256)).eval()
    features = torch.randn(3, 64, 48, 48)
    maps = branch.evidence_maps(features)
    logits = branch(features)
    assert maps.shape == (3, 2, 24, 24)
    assert torch.equal(logits, maps.mean(dim=(2, 3)))


def test_matched_branches_are_byte_identical_and_rng_local() -> None:
    local_block = ConvStemBlock(64, 256)
    before = audit._global_rng_snapshot()
    branches = audit.build_matched_branches(local_block, fold=4)
    after = audit._global_rng_snapshot()
    assert audit._global_rng_equal(before, after)
    hashes = {audit._parameter_sha256(branch) for branch in branches.values()}
    assert len(hashes) == 1


def test_valid_block2_crop_removes_padding_and_builds_geometry_masks() -> None:
    block2 = torch.arange(
        2 * 64 * 64 * 64, dtype=torch.float32
    ).reshape(2, 64, 64, 64)
    valid = torch.zeros(2, 1, 256, 256, dtype=torch.bool)
    valid[0, :, 32:224, 64:192] = True
    valid[1, :, 64:192, 16:240] = True
    bbox = torch.tensor(
        [[0.5, 0.5, 0.35, 0.55], [0.5, 0.5, 0.65, 0.35]],
        dtype=torch.float32,
    )
    rgb = torch.rand(2, 3, 256, 256)
    features, valid_masks, foreground, rgb_crops, geometry = (
        audit.crop_valid_block2_maps(block2, valid, bbox, rgb)
    )
    assert features.shape == (2, 64, 48, 48)
    assert valid_masks.shape == (2, 24, 24)
    assert foreground.shape == (2, 24, 24)
    assert rgb_crops.shape == (2, 256, 256, 3)
    assert geometry[0]["block2_x0"] == 16
    assert geometry[0]["block2_x1_exclusive"] == 48
    assert geometry[0]["block2_y0"] == 8
    assert geometry[0]["block2_y1_exclusive"] == 56
    assert bool(foreground.reshape(2, -1).any(axis=1).all())


def test_positive_threshold_is_highest_score_retaining_locked_fraction() -> None:
    positives = np.linspace(0.01, 1.0, 100)
    scores = np.concatenate((positives, np.asarray([0.2, 0.7, 0.9])))
    labels = np.concatenate((np.ones(100, dtype=np.int64), np.zeros(3, dtype=np.int64)))
    threshold = audit._positive_threshold(scores, labels)
    assert threshold == pytest.approx(positives[3])
    assert np.mean(positives >= threshold) == pytest.approx(0.97)
    assert np.mean(positives >= np.nextafter(threshold, np.inf)) < 0.97


def test_role_metrics_separate_keeper_tp_and_keeper_fn_support() -> None:
    targets = np.asarray([1, 1, 0, 2] * 5, dtype=np.int64)
    keeper = np.asarray([1, 0, 1, 1] * 5, dtype=np.int64)
    folds = np.repeat(np.arange(5), 4)
    scores = np.asarray([0.9, 0.8, 0.2, 0.7] * 5)
    actions = scores >= 0.5
    result = audit.role_metrics(
        scores=scores,
        actions=actions,
        targets=targets,
        keeper_predictions=keeper,
        folds=folds,
    )
    assert result["keeper_tp_broken"] == 0
    assert result["keeper_fn_supported"] == 5
    assert result["restricted_fp_rejected"] == 5
    assert result["all_class1_retention"] == 1.0


def test_map_statistics_measure_sparsity_foreground_border_and_padding() -> None:
    maps = np.zeros((2, 2, 24, 24), dtype=np.float32)
    maps[:, 1, 8:16, 8:16] = 2.0
    maps[:, 0, 8:16, 8:16] = -1.0
    valid = np.ones((2, 24, 24), dtype=np.bool_)
    foreground = np.zeros_like(valid)
    foreground[:, 6:18, 6:18] = True
    result = audit.map_statistics(
        maps, valid_masks=valid, foreground_masks=foreground
    )
    assert result["foreground_mass_mean"] == pytest.approx(1.0)
    assert result["border_mass_mean"] == pytest.approx(0.0)
    assert result["padding_mass_maximum"] == pytest.approx(0.0)
    assert result["zero_map_count"] == 0
    assert result["constant_map_count"] == 0
    assert result["margin_reconstruction_max_abs_error"] <= 1e-12


def _row(index: int, fold: int, category: str) -> CleanTrainRow:
    if category == "tp":
        target, keeper = 1, 1
    elif category == "fn":
        target, keeper = 1, 0
    else:
        target, keeper = 0, 1
    return CleanTrainRow(
        sample_index=index,
        source_stem=f"source_{index}",
        image_path=Path(f"D:/train/source_{index}.jpg"),
        fold=fold,
        target=target,
        keeper_prediction=keeper,
        keeper_probabilities=(0.1, 0.6, 0.1, 0.1, 0.1),
    )


def test_fixed_xai_selection_is_lowest_index_per_fold_and_category() -> None:
    rows = []
    for fold in audit.FOLDS:
        rows.extend(
            [
                _row(1000 + fold * 10 + 4, fold, "tp"),
                _row(1000 + fold * 10 + 1, fold, "tp"),
                _row(1000 + fold * 10 + 5, fold, "fn"),
                _row(1000 + fold * 10 + 2, fold, "fn"),
                _row(1000 + fold * 10 + 6, fold, "fp"),
                _row(1000 + fold * 10 + 3, fold, "fp"),
            ]
        )
    positions = audit.fixed_xai_positions(rows)
    selected = [rows[position].sample_index for position in positions]
    expected = []
    for fold in audit.FOLDS:
        expected.extend(
            [1000 + fold * 10 + 1, 1000 + fold * 10 + 2, 1000 + fold * 10 + 3]
        )
    assert selected == expected


def test_temporary_memmap_cleanup_is_verified(tmp_path: Path) -> None:
    path = tmp_path / "temporary.npy"
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(2, 3))
    cache[:] = 1.0
    cache.flush()
    extraction = {
        "features": cache,
        "feature_cache": {"path": str(path), "sha256": audit._array_sha256(cache)},
    }
    result = audit._close_delete_feature_cache(extraction)
    assert result["deleted"] is True
    assert not path.exists()
    assert "features" not in extraction


def test_replay_reconstructs_thresholds_scores_actions_metrics_and_map_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_per_fold = 4
    total = len(audit.FOLDS) * rows_per_fold
    targets = np.tile(np.asarray([1, 1, 0, 2], dtype=np.int64), len(audit.FOLDS))
    keeper = np.tile(np.asarray([1, 0, 1, 1], dtype=np.int64), len(audit.FOLDS))
    folds = np.repeat(np.asarray(audit.FOLDS, dtype=np.int64), rows_per_fold)
    indices = np.arange(500, 500 + total, dtype=np.int64)
    monkeypatch.setattr(audit, "EXPECTED_COHORT_ROWS", total)
    monkeypatch.setattr(
        audit,
        "EXPECTED_ORDERED_INDEX_SHA256",
        audit._cohort_index_sha256(indices.tolist()),
    )
    generator = np.random.default_rng(77)
    evidence_maps = {}
    scores = {}
    actions = {role: np.zeros(total, dtype=np.bool_) for role in audit.ROLE_NAMES}
    metadata = {"folds": {}}
    fit_payload = {}
    labels = (targets == 1).astype(np.int64)
    for role_number, role in enumerate(audit.ROLE_NAMES):
        maps = generator.normal(0.0, 0.2, size=(total, 2, 24, 24)).astype(np.float32)
        maps[:, 1] += np.where(labels[:, None, None] == 1, 0.7, -0.7)
        maps[:, 0] -= np.where(labels[:, None, None] == 1, 0.7, -0.7)
        maps[:, 1] += role_number * 0.01
        evidence_maps[role] = maps
        scores[role] = audit._probabilities_from_maps(maps)
    for fold in audit.FOLDS:
        fit_positions = np.flatnonzero(folds != fold)
        held_positions = np.flatnonzero(folds == fold)
        thresholds = {}
        for role in audit.TRAINED_ROLES:
            fit_scores = scores[role][fit_positions]
            fit_payload[f"fold{fold}__{role}"] = fit_scores
            thresholds[role] = audit._positive_threshold(
                fit_scores, labels[fit_positions]
            )
            actions[role][held_positions] = (
                scores[role][held_positions] >= thresholds[role]
            )
        fit_payload[f"fold{fold}__{audit.SAME_WEIGHT_ROLE}"] = scores[
            audit.SAME_WEIGHT_ROLE
        ][fit_positions]
        thresholds[audit.SAME_WEIGHT_ROLE] = thresholds[audit.CANDIDATE_ROLE]
        actions[audit.SAME_WEIGHT_ROLE][held_positions] = (
            scores[audit.SAME_WEIGHT_ROLE][held_positions]
            >= thresholds[audit.CANDIDATE_ROLE]
        )
        metadata["folds"][str(fold)] = {
            "fit_positions": fit_positions.tolist(),
            "held_positions": held_positions.tolist(),
            "thresholds": thresholds,
        }
    np.savez_compressed(tmp_path / "oof_evidence_maps.npz", **evidence_maps)
    valid = np.ones((total, 24, 24), dtype=np.bool_)
    foreground = np.ones_like(valid)
    np.savez_compressed(
        tmp_path / "evidence_geometry_masks.npz",
        valid=valid,
        foreground=foreground,
    )
    np.savez_compressed(tmp_path / "fit_scores.npz", **fit_payload)
    (tmp_path / "replay_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with (tmp_path / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "position",
            "sample_index",
            "source_stem",
            "fold",
            "target",
            "keeper_prediction",
            "binary_target",
        ] + [
            item
            for role in audit.ROLE_NAMES
            for item in (f"score__{role}", f"action__{role}")
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position in range(total):
            row = {
                "position": position,
                "sample_index": int(indices[position]),
                "source_stem": f"source_{position}",
                "fold": int(folds[position]),
                "target": int(targets[position]),
                "keeper_prediction": int(keeper[position]),
                "binary_target": int(labels[position]),
            }
            for role in audit.ROLE_NAMES:
                row[f"score__{role}"] = float(scores[role][position])
                row[f"action__{role}"] = int(actions[role][position])
            writer.writerow(row)
    (tmp_path / "training_evidence.json").write_text(
        json.dumps({"records": []}), encoding="utf-8"
    )
    expected = audit.build_clean_analysis(
        scores=scores,
        actions=actions,
        evidence_maps=evidence_maps,
        targets=targets,
        keeper_predictions=keeper,
        folds=folds,
        valid_masks=valid,
        foreground_masks=foreground,
        training_records=[],
    )
    result = audit.replay_artifacts(tmp_path, expected_analysis=expected)
    assert result["passed"] is True
    assert result["threshold_max_abs_error"] == 0.0
    assert result["score_max_abs_error"] <= audit.MAX_REPLAY_ERROR
    assert result["actions_exact"] is True
    assert result["analysis_maximum_numeric_difference"] <= audit.MAX_REPLAY_ERROR

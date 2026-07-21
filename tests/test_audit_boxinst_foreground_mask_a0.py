from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab
import torch

from trkh.tools import audit_boxinst_foreground_mask_a0 as audit


def test_projection_and_pairwise_equations_match_independent_numpy_fp64() -> None:
    generator = np.random.default_rng(81)
    mask = generator.uniform(0.01, 0.99, size=(3, 7, 9)).astype(np.float64)
    box = np.zeros_like(mask)
    box[:, 1:6, 2:8] = 1.0
    valid = np.ones_like(mask)
    valid[:, 0] = 0.0
    lab = generator.normal(size=(3, 3, 7, 9)).astype(np.float64)

    observed_projection = audit.projection_loss_torch(
        torch.from_numpy(mask), torch.from_numpy(box)
    ).numpy()
    expected_projection = audit.projection_loss_numpy(mask, box)
    assert np.max(np.abs(observed_projection - expected_projection)) <= 1e-10

    observed_loss, observed_similarity, observed_weights = audit.pairwise_affinity_torch(
        torch.from_numpy(mask),
        torch.from_numpy(lab),
        torch.from_numpy(valid),
        torch.from_numpy(box),
    )
    expected_loss, expected_similarity, expected_weights = audit.pairwise_affinity_numpy(
        mask, lab, valid, box
    )
    assert np.max(np.abs(observed_loss.numpy() - expected_loss)) <= 1e-10
    assert np.max(np.abs(observed_similarity.numpy() - expected_similarity)) <= 1e-10
    assert np.array_equal(observed_weights.numpy(), expected_weights)


def test_projection_orientation_dice_symmetry_and_pairwise_complement() -> None:
    mask = np.zeros((1, 5, 7), dtype=np.float64)
    mask[:, 1:4, 2:6] = 0.8
    box = np.zeros_like(mask)
    box[:, 1:4, 2:6] = 1.0
    loss = audit.projection_loss_numpy(mask, box)
    transposed = audit.projection_loss_numpy(
        np.transpose(mask, (0, 2, 1)), np.transpose(box, (0, 2, 1))
    )
    assert np.array_equal(loss, transposed)
    assert np.max(
        np.abs(audit.dice_loss_numpy(mask, box) - audit.dice_loss_numpy(box, mask))
    ) <= 1e-12
    neighbor = np.roll(mask, 1, axis=2)
    assert np.array_equal(
        audit.pairwise_same_probability_numpy(mask, neighbor),
        audit.pairwise_same_probability_numpy(1.0 - mask, 1.0 - neighbor),
    )


def test_pairwise_valid_edge_mask_excludes_padding_and_supports_singletons() -> None:
    mask = torch.full((1, 1, 5, 5), 0.7, dtype=torch.float64)
    lab = torch.zeros(1, 3, 5, 5, dtype=torch.float64)
    valid = torch.zeros(1, 5, 5, dtype=torch.float64)
    valid[:, 2, 2] = 1.0
    box = torch.ones_like(valid)
    loss, _, weights = audit.pairwise_affinity_torch(mask, lab, valid, box)
    assert loss.shape == (1,)
    assert float(loss[0]) == 0.0
    assert int(weights.sum()) == 0


def test_srgb_to_lab_matches_skimage_for_anchors_and_random_colors() -> None:
    generator = np.random.default_rng(19)
    colors = np.concatenate(
        (
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            generator.uniform(size=(64, 3)),
        ),
        axis=0,
    )
    rgb = colors.T.reshape(1, 3, 1, colors.shape[0])
    expected = rgb2lab(colors.reshape(1, colors.shape[0], 3)).transpose(2, 0, 1)[None]
    torch_result = audit.srgb_to_lab_torch(torch.from_numpy(rgb)).numpy()
    numpy_result = audit.srgb_to_lab_numpy(rgb)
    assert np.max(np.abs(torch_result - expected)) <= 5e-4
    assert np.max(np.abs(numpy_result - expected)) <= 5e-4
    assert np.isfinite(torch_result).all()


def test_common_roll_is_deterministic_rng_local_and_multiset_preserving() -> None:
    generator = torch.Generator().manual_seed(13)
    value = torch.randn(3, 3, 64, 64, generator=generator)
    before = audit._global_rng_snapshot()
    first = audit.common_toroidal_roll(
        value, [21, 42, 87], fold=3, purpose="affinity"
    )
    second = audit.common_toroidal_roll(
        value, [21, 42, 87], fold=3, purpose="affinity"
    )
    after = audit._global_rng_snapshot()
    assert torch.equal(first, second)
    assert audit._global_rng_equal(before, after)
    assert not torch.equal(first, value)
    assert torch.equal(
        torch.sort(first.flatten(2), dim=2).values,
        torch.sort(value.flatten(2), dim=2).values,
    )
    aligned = torch.sort(audit._periodic_affinity_multiset(value).flatten(1), dim=1).values
    rolled = torch.sort(audit._periodic_affinity_multiset(first).flatten(1), dim=1).values
    assert torch.max(torch.abs(aligned - rolled)).item() <= 1e-6


def test_bbox_coordinates_and_masks_follow_locked_geometry() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.25]], dtype=torch.float32)
    coordinates = audit.bbox_coordinate_channels(boxes)
    mask = audit.normalized_bbox_masks(boxes)
    assert coordinates.shape == (1, 2, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert bool(mask.any())
    assert int(mask.sum()) == 32 * 16
    assert float(coordinates.max()) <= 2.0
    assert float(coordinates.min()) >= -2.0


def test_geometry_downsampling_reconstructs_bbox_inside_letterbox() -> None:
    valid = torch.zeros(2, 1, 256, 256, dtype=torch.bool)
    valid[0, :, 32:224, 64:192] = True
    valid[1, :, 64:192, 16:240] = True
    boxes = torch.tensor(
        [[0.5, 0.5, 0.25, 0.50], [0.5, 0.5, 0.50, 0.25]],
        dtype=torch.float32,
    )
    valid64, bbox64, records = audit._geometry_from_metadata(valid, boxes)
    assert valid64.shape == (2, 64, 64)
    assert bbox64.shape == valid64.shape
    assert not bool((bbox64 & ~valid64).any())
    assert all(record["bbox_outside_valid_pixels"] == 0 for record in records)
    assert [record["valid_pixels"] for record in records] == [1536, 1792]


def test_fixed_projection_is_orthonormal_deterministic_and_rng_local() -> None:
    before = audit._global_rng_snapshot()
    first = audit.fixed_projection()
    second = audit.fixed_projection()
    after = audit._global_rng_snapshot()
    assert audit._global_rng_equal(before, after)
    assert np.array_equal(first, second)
    np.testing.assert_allclose(
        first.T @ first,
        np.eye(audit.PROJECTION_DIM),
        atol=1e-12,
    )


def test_matched_heads_are_rng_local_byte_exact_and_trainable() -> None:
    before = audit._global_rng_snapshot()
    heads = audit.build_matched_heads(fold=2)
    after = audit._global_rng_snapshot()
    assert audit._global_rng_equal(before, after)
    assert len({audit._parameter_sha256(head) for head in heads.values()}) == 1
    features = torch.randn(2, 64, 64, 64)
    boxes = torch.tensor([[0.5, 0.5, 0.6, 0.5], [0.4, 0.6, 0.3, 0.4]])
    output = heads[audit.CANDIDATE_ROLE](features, boxes)
    assert output.shape == (2, 1, 64, 64)
    assert bool(torch.all((output > 0.0) & (output < 1.0)))


def test_descriptor_bank_has_locked_roles_dimensions_and_finite_values() -> None:
    generator = np.random.default_rng(31)
    rows = 4
    features = generator.normal(size=(rows, 24, 64, 64))
    valid = np.ones((rows, 64, 64), dtype=np.bool_)
    valid[:, :8] = False
    bbox = np.zeros_like(valid)
    bbox[:, 12:54, 14:50] = True
    candidate = generator.uniform(0.05, 0.95, size=valid.shape)
    masks = {
        audit.PROJECTION_ROLE: candidate * 0.9,
        audit.CANDIDATE_ROLE: candidate,
        audit.DEPHASED_ROLE: candidate * 0.8,
        audit.VALID_ROLE: valid.astype(np.float64),
        audit.BBOX_ROLE: bbox.astype(np.float64),
        audit.ROLLED_ROLE: np.roll(candidate, 7, axis=2),
    }
    keeper = generator.dirichlet(np.ones(5), size=rows)
    descriptors = audit.build_descriptors(features, masks, valid, bbox, keeper)
    assert set(descriptors) == set(audit.ROLE_NAMES)
    for role in audit.MASK_ROLES:
        assert descriptors[role].shape == (rows, audit.DESCRIPTOR_DIM)
    assert descriptors[audit.WITHOUT_KEEPER_ROLE].shape == (
        rows,
        audit.DESCRIPTOR_WITHOUT_KEEPER_DIM,
    )
    assert descriptors[audit.KEEPER_ROLE].shape == (rows, 5)
    assert all(np.isfinite(value).all() for value in descriptors.values())
    np.testing.assert_allclose(
        descriptors[audit.CANDIDATE_ROLE][:, : audit.DESCRIPTOR_WITHOUT_KEEPER_DIM],
        descriptors[audit.WITHOUT_KEEPER_ROLE],
    )


def test_torch_export_descriptor_matches_numpy_descriptor() -> None:
    generator = np.random.default_rng(103)
    rows = 2
    features = generator.normal(size=(rows, 64, 64, 64)).astype(np.float32)
    valid = np.ones((rows, 64, 64), dtype=np.float32)
    valid[:, :7] = 0.0
    bbox = np.zeros_like(valid)
    bbox[:, 13:53, 11:55] = 1.0
    masks = generator.uniform(0.05, 0.95, size=valid.shape).astype(np.float32) * valid
    keeper = generator.dirichlet(np.ones(5), size=rows).astype(np.float32)
    projection = audit.fixed_projection().astype(np.float32)
    observed = audit._torch_descriptor_batch(
        torch.from_numpy(features),
        torch.from_numpy(masks),
        torch.from_numpy(valid),
        torch.from_numpy(bbox),
        torch.from_numpy(keeper),
        torch.from_numpy(projection),
    ).numpy()
    projected = np.einsum("bchw,cd->bdhw", features, projection)
    bank = {role: masks for role in audit.MASK_ROLES}
    expected = audit.build_descriptors(
        projected,
        bank,
        valid.astype(np.bool_),
        bbox.astype(np.bool_),
        keeper,
    )[audit.CANDIDATE_ROLE]
    np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=1e-5)


def test_epoch_orders_are_deterministic_complete_and_epoch_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "EPOCHS", 3)
    positions = np.asarray([2, 4, 7, 9, 11, 15], dtype=np.int64)
    first, first_hash = audit._epoch_orders(positions, fold=3)
    second, second_hash = audit._epoch_orders(positions, fold=3)
    assert first_hash == second_hash
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(np.array_equal(np.sort(order), positions) for order in first)
    assert len({order.tobytes() for order in first}) == 3


def test_oof_readout_states_replay_scores_and_actions() -> None:
    generator = np.random.default_rng(55)
    rows_per_fold = 8
    rows = rows_per_fold * len(audit.FOLDS)
    folds = np.repeat(np.asarray(audit.FOLDS), rows_per_fold)
    labels = np.tile(np.asarray([1, 1, 1, 1, 0, 0, 0, 0]), len(audit.FOLDS))
    sources = [f"source_{index}" for index in range(rows)]
    descriptors = {}
    for role in audit.ROLE_NAMES:
        dimension = (
            5
            if role == audit.KEEPER_ROLE
            else audit.DESCRIPTOR_WITHOUT_KEEPER_DIM
            if role == audit.WITHOUT_KEEPER_ROLE
            else audit.DESCRIPTOR_DIM
        )
        values = generator.normal(size=(rows, dimension))
        values[:, 0] += labels * 0.7
        descriptors[role] = values
    scores, actions, states = audit.fit_oof_readouts(
        descriptors, labels, folds, sources
    )
    replay_scores, replay_actions = audit.apply_readout_states(
        descriptors, folds, states
    )
    for role in audit.ROLE_NAMES:
        np.testing.assert_allclose(scores[role], replay_scores[role], atol=1e-12)
        assert np.array_equal(actions[role], replay_actions[role])


def test_role_metrics_separates_tp_fn_and_restricted_fp() -> None:
    targets = np.asarray([1, 1, 0, 2] * 5, dtype=np.int64)
    keeper = np.asarray([1, 0, 1, 1] * 5, dtype=np.int64)
    folds = np.repeat(np.asarray(audit.FOLDS), 4)
    scores = np.asarray([0.9, 0.8, 0.2, 0.3] * 5)
    result = audit.role_metrics(
        scores=scores,
        actions=scores >= 0.5,
        targets=targets,
        keeper_predictions=keeper,
        folds=folds,
    )
    assert result["keeper_tp_broken"] == 0
    assert result["keeper_fn_supported"] == 5
    assert result["restricted_fp_rejected"] == 10
    assert result["precision"] == pytest.approx(1.0)


def test_engineering_checks_cover_locked_equations_and_controls() -> None:
    result = audit.engineering_checks()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["errors"]["projection"] <= 1e-10
    assert result["errors"]["pairwise"] <= 1e-10
    assert max(result["errors"]["lab_torch"], result["errors"]["lab_numpy"]) <= 5e-4


def test_temporary_feature_and_lab_caches_are_deleted(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npy"
    lab_path = tmp_path / "lab.npy"
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16, shape=(2, 3)
    )
    lab = np.lib.format.open_memmap(lab_path, mode="w+", dtype=np.float16, shape=(2, 2))
    features[:] = 1.0
    lab[:] = 2.0
    extraction = {
        "features": features,
        "lab": lab,
        "features_cache": {"path": str(feature_path)},
        "lab_cache": {"path": str(lab_path)},
    }
    result = audit._close_delete_caches(extraction)
    assert result["deleted"] is True
    assert not feature_path.exists()
    assert not lab_path.exists()
    assert "features" not in extraction
    assert "lab" not in extraction

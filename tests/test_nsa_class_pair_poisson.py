from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.nsa_class_pair_poisson import (
    ClassQueryMismatchAdapter,
    PoissonGeometry,
    SourceGeometry,
    TargetGeometry,
    array_sha256,
    deterministic_deranged_queries,
    geometry_sha256,
    make_poisson_view,
    map_summaries,
    masked_bce_per_sample,
    parameter_count,
    rng_snapshot,
    rng_snapshot_equal,
    sample_poisson_geometry,
    sample_source_geometry,
    sample_target_geometry,
    validate_role_parameter_parity,
    valid_rectangle_positions,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow
from trkh.tools.audit_nsa_class_pair_poisson_a0 import build_preview_plan


def _support(size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return ((xx - size / 2) ** 2 / (size * 0.40) ** 2) + (
        (yy - size / 2) ** 2 / (size * 0.43) ** 2
    ) <= 1.0


def _images(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    target = np.stack(
        (
            40 + 2 * xx,
            70 + yy,
            35 + (xx + yy) // 2,
        ),
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    source = np.stack(
        (
            210 - 2 * yy,
            30 + 3 * (xx % 17),
            180 - 2 * (yy % 23),
        ),
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    return target, source


def test_valid_rectangle_positions_are_fully_supported() -> None:
    support = np.zeros((12, 14), dtype=bool)
    support[2:10, 3:12] = True
    positions = valid_rectangle_positions(support, height=4, width=5)
    assert positions.shape[0] == (8 - 4 + 1) * (9 - 5 + 1)
    for top, left in positions.tolist():
        assert support[top : top + 4, left : left + 5].all()


def test_geometry_is_deterministic_local_and_inside_support() -> None:
    support = _support()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    before = rng_snapshot()
    first = sample_poisson_geometry(
        support,
        support,
        sample_index=123,
        epoch=1,
        target_role_id=7,
        source_role_id=8,
    )
    after = rng_snapshot()
    second = sample_poisson_geometry(
        support,
        support,
        sample_index=123,
        epoch=1,
        target_role_id=7,
        source_role_id=8,
    )
    assert rng_snapshot_equal(before, after)
    assert first == second
    assert geometry_sha256(first) == geometry_sha256(second)
    target = first.target
    source = first.source
    assert support[
        target.top : target.top + target.height,
        target.left : target.left + target.width,
    ].all()
    assert support[
        source.top : source.top + source.height,
        source.left : source.left + source.width,
    ].all()


def test_matched_target_geometry_changes_only_source_draw() -> None:
    support = _support()
    target = sample_target_geometry(
        support,
        sample_index=31,
        epoch=0,
        role_id=1,
    )
    first = sample_source_geometry(
        support,
        target,
        sample_index=31,
        epoch=0,
        role_id=2,
    )
    second = sample_source_geometry(
        support,
        target,
        sample_index=31,
        epoch=0,
        role_id=3,
    )
    assert first.seed != second.seed
    assert target.height > 2 and target.width > 2


def test_poisson_view_has_nonempty_local_intensity_target() -> None:
    support = _support()
    target_rgb, source_rgb = _images()
    geometry = PoissonGeometry(
        target=TargetGeometry(
            seed=1,
            top=22,
            left=22,
            height=20,
            width=20,
            shape_attempt=1,
            target_candidate_count=1,
            half_height_ratio=0.2,
            half_width_ratio=0.2,
        ),
        source=SourceGeometry(
            seed=2,
            top=22,
            left=22,
            height=20,
            width=20,
            resize_scale=1.0,
            source_candidate_count=1,
            source_attempt=1,
        ),
    )
    view = make_poisson_view(
        target_rgb,
        source_rgb,
        support,
        support,
        geometry,
    )
    assert view.composite_rgb.shape == target_rgb.shape
    assert view.composite_rgb.dtype == np.uint8
    assert view.intensity_target.dtype == np.float32
    assert view.changed_mask.any()
    assert float(view.intensity_target.max()) > 0.0
    assert float(view.intensity_target.min()) >= 0.0
    assert float(view.intensity_target.max()) <= 1.0
    assert not view.changed_mask[~support].any()
    assert view.maximum_outside_target_delta == 0
    assert array_sha256(view.composite_rgb) == array_sha256(
        make_poisson_view(
            target_rgb,
            source_rgb,
            support,
            support,
            geometry,
        ).composite_rgb
    )


def test_adapter_roles_are_capacity_matched_and_shape_stable() -> None:
    torch.manual_seed(17)
    roles = {
        name: ClassQueryMismatchAdapter()
        for name in ("candidate", "clean", "no_query", "permuted")
    }
    count, counts = validate_role_parameter_parity(roles)
    assert count == parameter_count(roles["candidate"])
    assert len(set(counts.values())) == 1
    stem = torch.randn(3, 256, 32, 32)
    query = torch.tensor([0, 1, 4])
    output = roles["candidate"](stem, query)
    assert output.shape == (3, 1, 32, 32)
    with pytest.raises(ValueError, match="outside"):
        roles["candidate"](stem, torch.tensor([0, 1, 5]))


def test_masked_bce_ignores_pixels_outside_support() -> None:
    logits = torch.zeros(2, 1, 4, 4)
    target = torch.zeros_like(logits)
    support = torch.zeros_like(logits, dtype=torch.bool)
    support[:, :, 1:3, 1:3] = True
    first = masked_bce_per_sample(logits, target, support)
    modified = logits.clone()
    modified[:, :, 0, 0] = 100.0
    second = masked_bce_per_sample(modified, target, support)
    assert torch.equal(first, second)
    assert torch.allclose(first, torch.full((2,), np.log(2.0)))


def test_map_summaries_follow_locked_definition() -> None:
    probabilities = torch.tensor(
        [[[[0.0, 0.1], [0.8, 1.0]]]],
        dtype=torch.float32,
    )
    support = torch.ones_like(probabilities, dtype=torch.bool)
    summary = map_summaries(probabilities, support)
    assert summary["support_mean"].item() == pytest.approx(0.475)
    assert summary["top_quintile_mean"].item() == pytest.approx(1.0)
    assert summary["percentile_90"].item() == pytest.approx(0.94)
    assert summary["compatibility"].item() == pytest.approx(0.2625)


def test_deranged_queries_are_deterministic_and_not_labels() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4])
    indices = torch.tensor([10, 11, 12, 13, 14])
    first = deterministic_deranged_queries(
        labels,
        sample_indices=indices,
        epoch=1,
        namespace=99,
    )
    second = deterministic_deranged_queries(
        labels,
        sample_indices=indices,
        epoch=1,
        namespace=99,
    )
    assert torch.equal(first, second)
    assert bool((first != labels).all())


def test_preview_plan_is_fixed_distinct_and_source_disjoint() -> None:
    rows = []
    sample_index = 0
    for target in range(5):
        for local_index in range(40):
            rows.append(
                CleanTrainRow(
                    sample_index=sample_index,
                    source_stem=f"class{target}_source{local_index}",
                    image_path=Path(f"C:/train/class{target}_source{local_index}.jpg"),
                    fold=local_index % 5,
                    target=target,
                    keeper_prediction=target,
                    keeper_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
                )
            )
            sample_index += 1
    first = build_preview_plan(rows)
    second = build_preview_plan(rows)
    assert first == second
    assert len(first) == 24
    assert len({row["target_sample_index"] for row in first}) == 24
    assert {row["target_class"] for row in first} == {0, 1, 2, 4}
    for item in first:
        target_source = item["target_source_stem"]
        same_sources = {
            rows[index].source_stem for index in item["same_candidate_indices"]
        }
        cross_sources = {
            rows[index].source_stem for index in item["cross_candidate_indices"]
        }
        assert target_source not in same_sources
        assert target_source not in cross_sources

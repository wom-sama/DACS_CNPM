from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

import trkh.tools.nsa_class_pair_poisson_a0_engine as nsa_engine
from trkh.tools.nsa_class_pair_poisson import (
    ClassQueryMismatchAdapter,
    PoissonGeometry,
    SourceGeometry,
    TargetGeometry,
    array_sha256,
    deterministic_deranged_queries,
    exclude_entering_chromatic_occluders,
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
    state_dict_sha256,
    validate_role_parameter_parity,
    valid_rectangle_positions,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow
from trkh.tools.audit_nsa_class_pair_poisson_a0 import (
    _compare_fold_scores,
    _compare_nested,
    _load_fold_scores,
    _save_fold_scores,
    build_preview_plan,
    render_formal_contact_sheet,
)
from trkh.tools.nsa_class_pair_poisson_a0_engine import (
    READOUT_ROLES,
    ROLE_NAMES,
    apply_saved_readouts,
    assess_mechanism_gate,
    build_balanced_fit_panel,
    cross_class_schedule,
    fixed_visual_indices,
    geometry_records_sha256,
    initialize_roles,
    role_metrics,
)


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


def test_occluder_filter_removes_only_entering_component() -> None:
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[:] = np.array([80, 170, 70], dtype=np.uint8)
    support = np.zeros((80, 80), dtype=bool)
    support[20:65, 20:65] = True
    hand_color = np.array([180, 138, 157], dtype=np.uint8)
    rgb[25:50, 0:32] = hand_color
    rgb[52:59, 52:59] = hand_color
    final, record = exclude_entering_chromatic_occluders(rgb, support)
    assert record["selected_component_count"] == 1
    assert record["excluded_support_pixels"] > 0
    assert not final[30:45, 20:30].any()
    assert final[52:59, 52:59].all()
    assert final[30, 40]
    assert final.sum() < support.sum()


def _engine_rows(per_class: int = 15) -> list[CleanTrainRow]:
    rows = []
    for target in range(5):
        for local_index in range(per_class):
            sample_index = len(rows)
            probabilities = np.full(5, 0.025, dtype=np.float64)
            probabilities[target] = 0.9
            rows.append(
                CleanTrainRow(
                    sample_index=sample_index,
                    source_stem=f"engine_{target}_{local_index}",
                    image_path=Path(
                        f"C:/train/engine_{target}_{local_index}.jpg"
                    ),
                    fold=local_index % 5,
                    target=target,
                    keeper_prediction=target,
                    keeper_probabilities=tuple(probabilities.tolist()),
                )
            )
    return rows


def test_balanced_panel_is_deterministic_and_excludes_held_fold() -> None:
    rows = _engine_rows()
    first, first_summary = build_balanced_fit_panel(rows, held_fold=2)
    second, second_summary = build_balanced_fit_panel(rows, held_fold=2)
    assert first == second
    assert first_summary == second_summary
    assert len(first) == 5 * 12
    assert len(first) == len(set(first))
    assert {rows[index].fold for index in first} == {0, 1, 3, 4}
    assert {
        target: sum(rows[index].target == target for index in first)
        for target in range(5)
    } == {target: 12 for target in range(5)}


@pytest.mark.parametrize("target", range(5))
def test_cross_class_schedule_covers_each_other_class_once(target: int) -> None:
    observed = [
        value
        for epoch in range(2)
        for value in cross_class_schedule(target, sample_index=37, epoch=epoch)
    ]
    assert sorted(observed) == [value for value in range(5) if value != target]


def test_role_initialization_is_equal_and_preserves_global_rng() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    roles, summary = initialize_roles(held_fold=3, device=torch.device("cpu"))
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert tuple(roles) == ROLE_NAMES
    assert len({state_dict_sha256(module) for module in roles.values()}) == 1
    assert summary["seed"] == 20260727


def test_role_metrics_counts_corrections_without_subtracting_harms() -> None:
    cohort = []
    labels = []
    folds = []
    scores = []
    actions = []
    for fold in range(5):
        for target in (1, 0):
            index = len(cohort)
            probabilities = (
                (0.1, 0.7, 0.1, 0.05, 0.05)
                if target == 1
                else (0.7, 0.2, 0.05, 0.025, 0.025)
            )
            cohort.append(
                CleanTrainRow(
                    sample_index=index,
                    source_stem=f"metric_{index}",
                    image_path=Path(f"C:/train/metric_{index}.jpg"),
                    fold=fold,
                    target=target,
                    keeper_prediction=1,
                    keeper_probabilities=probabilities,
                )
            )
            labels.append(int(target == 1))
            folds.append(fold)
            scores.append(0.9 if target == 1 else 0.1)
            actions.append(fold != 0 if target == 1 else fold == 4)
    metrics = role_metrics(
        cohort=cohort,
        labels=np.asarray(labels, dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float64),
        actions=np.asarray(actions),
    )
    assert metrics["corrections"] == 4
    assert metrics["harms"] == 1
    assert metrics["fp_rejected"] == 4


def test_mechanism_gate_is_strictly_conjunctive() -> None:
    folds = [
        {"fold": fold, "auroc": 0.90, "tp_retention": 0.99}
        for fold in range(5)
    ]
    candidate = {
        "auroc": 0.90,
        "tp_retention": 0.99,
        "fp_rejection": 0.20,
        "fp_rejected": 25,
        "corrections": 9,
        "harms": 3,
        "target_rejection": {"0": 0.10, "2": 0.10, "4": 0.10},
        "folds": folds,
    }
    control_folds = [
        {"fold": fold, "auroc": 0.87, "tp_retention": 0.99}
        for fold in range(5)
    ]
    metrics = {
        "poisson_query_candidate": candidate,
        "base_context": {"auroc": 0.87, "folds": control_folds},
        "clean_query_control": {
            "auroc": 0.88,
            "fp_rejected": 20,
            "folds": control_folds,
        },
        "no_query_control": {"auroc": 0.87},
        "permuted_query_control": {"auroc": 0.87},
    }
    query = {
        "true_query_better_fraction": 0.75,
        "five_query_accuracy": 0.50,
        "effective_spatial_rank": 2.0,
    }
    diagnostics = {
        "cross_map_mass_inside_intensity": 0.80,
        "same_support_pixel_mismatch_rate": 0.10,
    }
    passed = assess_mechanism_gate(
        metrics=metrics,
        query=query,
        diagnostics=diagnostics,
        candidate_query_gradients_nonzero=True,
        candidate_query_updates_nonzero=True,
    )
    assert passed["mechanism_pass"]
    candidate["auroc"] = 0.889
    failed = assess_mechanism_gate(
        metrics=metrics,
        query=query,
        diagnostics=diagnostics,
        candidate_query_gradients_nonzero=True,
        candidate_query_updates_nonzero=True,
    )
    assert not failed["mechanism_pass"]
    candidate["auroc"] = 0.90
    failed_update = assess_mechanism_gate(
        metrics=metrics,
        query=query,
        diagnostics=diagnostics,
        candidate_query_gradients_nonzero=True,
        candidate_query_updates_nonzero=False,
    )
    assert not failed_update["mechanism_pass"]


def test_saved_readouts_apply_without_refitting() -> None:
    cohort = []
    for fold in range(5):
        for target in (1, 0):
            index = len(cohort)
            probabilities = (
                (0.05, 0.8, 0.05, 0.05, 0.05)
                if target == 1
                else (0.8, 0.05, 0.05, 0.05, 0.05)
            )
            cohort.append(
                CleanTrainRow(
                    sample_index=index,
                    source_stem=f"readout_{index}",
                    image_path=Path(f"C:/train/readout_{index}.jpg"),
                    fold=fold,
                    target=target,
                    keeper_prediction=1,
                    keeper_probabilities=probabilities,
                )
            )
    metadata = {
        "bbox": np.tile(
            np.asarray([[0.5, 0.5, 0.4, 0.4]], dtype=np.float64),
            (len(cohort), 1),
        ),
        "valid_fraction": np.ones(len(cohort), dtype=np.float64),
        "support_fraction": np.full(len(cohort), 0.5, dtype=np.float64),
    }
    fold_scores = {}
    for fold in range(5):
        fold_scores[fold] = {
            "metadata": metadata,
            "roles": {
                role: {
                    "support_mean": np.full(len(cohort), 0.2),
                    "top_quintile_mean": np.full(len(cohort), 0.3),
                    "percentile_90": np.full(len(cohort), 0.25),
                }
                for role in ROLE_NAMES
            },
        }
    states = []
    for fold in range(5):
        for role in READOUT_ROLES:
            feature_dim = 14 if role == "base_context" else 17
            states.append(
                {
                    "held_fold": fold,
                    "role": role,
                    "threshold": 0.5,
                    "scaler_mean": [0.0] * feature_dim,
                    "scaler_scale": [1.0] * feature_dim,
                    "coefficient": [0.0] * feature_dim,
                    "intercept": [0.0],
                }
            )
    replay = apply_saved_readouts(
        cohort=cohort,
        fold_scores=fold_scores,
        states=states,
    )
    for role in READOUT_ROLES:
        assert np.array_equal(
            replay["scores"][role],
            np.full(len(cohort), 0.5),
        )
        assert replay["actions"][role].all()


def test_fixed_visual_panel_is_score_independent_and_covers_available_cells() -> None:
    cohort = []
    for fold in range(5):
        for target in (1, 0, 2, 4):
            if fold == 1 and target == 4:
                continue
            for repeat in range(2):
                index = len(cohort)
                cohort.append(
                    CleanTrainRow(
                        sample_index=index,
                        source_stem=f"visual_{fold}_{target}_{repeat}",
                        image_path=Path(
                            f"C:/train/visual_{fold}_{target}_{repeat}.jpg"
                        ),
                        fold=fold,
                        target=target,
                        keeper_prediction=1,
                        keeper_probabilities=(0.2,) * 5,
                    )
                )
    first = fixed_visual_indices(cohort)
    second = fixed_visual_indices(list(reversed(cohort)))
    assert first == second
    assert len(first) == 19
    assert {
        (cohort[index].fold, cohort[index].target) for index in first
    } == {
        (fold, target)
        for fold in range(5)
        for target in (1, 0, 2, 4)
        if not (fold == 1 and target == 4)
    }


def test_geometry_record_hash_is_order_sensitive() -> None:
    records = [{"sample": 1, "hash": "a"}, {"sample": 2, "hash": "b"}]
    assert geometry_records_sha256(records) == geometry_records_sha256(
        [dict(record) for record in records]
    )
    assert geometry_records_sha256(records) != geometry_records_sha256(
        list(reversed(records))
    )


def test_cross_view_retries_geometry_after_blend_failure(monkeypatch) -> None:
    row = CleanTrainRow(
        sample_index=0,
        source_stem="retry_target",
        image_path=Path("C:/train/retry_target.jpg"),
        fold=0,
        target=1,
        keeper_prediction=1,
        keeper_probabilities=(0.1, 0.6, 0.1, 0.1, 0.1),
    )
    calls = []
    sentinel = object()

    def fake_find(*args, **kwargs):
        calls.append(kwargs.get("source_role_id"))
        if len(calls) == 1:
            raise RuntimeError("empty intensity")
        return 0, {"support": _support()}, sentinel

    monkeypatch.setattr(nsa_engine, "_find_donor_view", fake_find)
    donor_index, _, view, retry = (
        nsa_engine._find_cross_view_with_geometry_retries(
            dataset=object(),
            semantics={},
            rows=[row],
            target={"support": _support()},
            target_row=row,
            candidates=[0],
            epoch=0,
            target_role_id=11,
            source_role_id=300,
        )
    )
    assert donor_index == 0
    assert view is sentinel
    assert retry == 1
    assert calls == [300, 10300]


def test_formal_fold_score_artifact_round_trip(tmp_path: Path) -> None:
    row_count = 3
    scores = {
        "metadata": {
            "sample_index": np.arange(row_count, dtype=np.int64),
            "target": np.arange(row_count, dtype=np.int64),
            "fold": np.arange(row_count, dtype=np.int64),
            "bbox": np.ones((row_count, 4), dtype=np.float32),
            "valid_fraction": np.ones(row_count, dtype=np.float32),
            "support_fraction": np.full(row_count, 0.5, dtype=np.float32),
        },
        "roles": {
            role: {
                summary: np.full(row_count, 0.2, dtype=np.float32)
                for summary in (
                    "support_mean",
                    "top_quintile_mean",
                    "percentile_90",
                    "compatibility",
                )
            }
            for role in ROLE_NAMES
        },
        "query_compatibility": np.full(
            (row_count, 5), 0.4, dtype=np.float32
        ),
        "candidate_maps": np.full(
            (row_count, 1, 32, 32), 0.3, dtype=np.float32
        ),
    }
    path = tmp_path / "scores.npz"
    artifact = _save_fold_scores(path, scores)
    loaded = _load_fold_scores(path)
    comparison = _compare_fold_scores(scores, loaded)
    assert artifact["sha256"]
    assert comparison["passed"]
    assert comparison["maximum_absolute_error"] == 0.0


def test_nested_comparison_tracks_numeric_error_and_structure() -> None:
    exact = _compare_nested(
        {"a": [1.0, {"b": True}]},
        {"a": [1.0, {"b": True}]},
    )
    assert exact["structure_exact"]
    assert exact["maximum_absolute_error"] == 0.0
    changed = _compare_nested(
        {"a": [1.0, {"b": True}]},
        {"a": [1.0001, {"b": False}]},
    )
    assert not changed["structure_exact"]
    assert changed["maximum_absolute_error"] == pytest.approx(0.0001)


def test_formal_contact_sheet_renders_locked_rows(tmp_path: Path) -> None:
    rows = []
    for index in range(19):
        rows.append(
            {
                "sample_index": index,
                "target": index % 5,
                "fold": index % 5,
                "clean_rgb": np.full((64, 64, 3), 100, dtype=np.uint8),
                "support": np.ones((64, 64), dtype=bool),
                "same_rgb": np.full((64, 64, 3), 110, dtype=np.uint8),
                "cross_rgb": np.full((64, 64, 3), 120, dtype=np.uint8),
                "cross_target": np.full((32, 32), 0.3, dtype=np.float32),
                "role_maps": {
                    role: np.full((32, 32), 0.2, dtype=np.float32)
                    for role in ROLE_NAMES
                },
                "query_maps": {
                    str(query): np.full(
                        (32, 32), query / 5.0, dtype=np.float32
                    )
                    for query in range(5)
                },
            }
        )
    output = tmp_path / "contact.png"
    summary = render_formal_contact_sheet(rows, output)
    assert output.is_file()
    assert summary["rows"] == 19
    assert summary["columns"] == 14
    assert summary["sha256"]

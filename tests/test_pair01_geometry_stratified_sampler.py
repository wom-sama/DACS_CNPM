from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from trkh.data.dataset import StrictBalancedBatchSampler, TeacherProbabilityDataset
from trkh.training.train import build_configs, build_weighted_sampler, parse_args


LABELS = [0] * 16 + [1] * 16 + [2] * 16
AREAS = (
    [
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.20,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.40,
        0.41,
        0.42,
        0.43,
    ]
    + [
        0.03,
        0.035,
        0.04,
        0.045,
        0.05,
        0.055,
        0.06,
        0.07,
        0.10,
        0.20,
        0.22,
        0.25,
        0.40,
        0.42,
        0.45,
        0.48,
    ]
    + [0.10 + 0.01 * index for index in range(16)]
)


class _GeometryDataset(Dataset):
    def __len__(self) -> int:
        return len(LABELS)

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), LABELS[int(index)], {}

    def labels(self) -> list[int]:
        return list(LABELS)

    def sample_paths(self) -> list[Path]:
        return [Path(f"train_sample_{index}.jpg") for index in range(len(self))]

    def sample_bboxes(self) -> list[tuple[float, float, float, float]]:
        return [(0.5, 0.5, float(area), 1.0) for area in AREAS]


def _area_bin(area: float, edges: list[float]) -> int:
    if area <= edges[0]:
        return 0
    if area <= edges[1]:
        return 1
    return 2


def test_geometry_stratification_preserves_class_exposure_and_matches_pair_bins() -> None:
    sampler = StrictBalancedBatchSampler(
        labels=LABELS,
        batch_size=6,
        num_classes=3,
        epoch_multiplier=1.0,
        seed=17,
        geometry_areas=AREAS,
        geometry_stratified_pair=(0, 1),
    )

    summary = sampler.exposure_summary()
    geometry = summary["geometry_stratification"]
    planned = geometry["planned_bin_exposure_counts"]
    assert summary["class_exposure_counts"] == [16, 16, 16]
    assert planned["0"] == planned["1"]

    observed_class_counts = [0, 0, 0]
    observed_pair_bins = {0: [0, 0, 0], 1: [0, 0, 0]}
    edges = geometry["bin_edges"]
    for batch in sampler:
        assert len(batch) == 6
        for sample_index in batch:
            class_index = LABELS[sample_index]
            observed_class_counts[class_index] += 1
            if class_index in observed_pair_bins:
                observed_pair_bins[class_index][
                    _area_bin(AREAS[sample_index], edges)
                ] += 1

    assert observed_class_counts == summary["class_exposure_counts"]
    assert observed_pair_bins[0] == planned["0"]
    assert observed_pair_bins[1] == planned["1"]


def test_default_sampler_path_remains_bitwise_deterministic() -> None:
    first = StrictBalancedBatchSampler(
        labels=LABELS,
        batch_size=6,
        num_classes=3,
        seed=42,
    )
    second = StrictBalancedBatchSampler(
        labels=LABELS,
        batch_size=6,
        num_classes=3,
        seed=42,
    )

    assert list(first) == list(second)
    assert "geometry_stratification" not in first.exposure_summary()


def test_geometry_stratification_fails_closed_without_complete_geometry() -> None:
    with pytest.raises(ValueError, match="bbox geometry train-only"):
        StrictBalancedBatchSampler(
            labels=LABELS,
            batch_size=6,
            num_classes=3,
            geometry_stratified_pair=(0, 1),
        )

    sparse_labels = [0] * 8 + [1] * 8
    sparse_areas = [0.05, 0.06, 0.20, 0.21, 0.22, 0.23, 0.40, 0.41] * 2
    with pytest.raises(ValueError, match="it nhat 4 samples"):
        StrictBalancedBatchSampler(
            labels=sparse_labels,
            batch_size=4,
            num_classes=2,
            geometry_areas=sparse_areas,
            geometry_stratified_pair=(0, 1),
        )

    with pytest.raises(ValueError, match="length mismatch"):
        StrictBalancedBatchSampler(
            labels=LABELS,
            batch_size=6,
            num_classes=3,
            geometry_areas=AREAS[:-1],
            geometry_stratified_pair=(0, 1),
        )

    invalid = list(AREAS)
    invalid[3] = float("nan")
    with pytest.raises(ValueError, match="khong hop le"):
        StrictBalancedBatchSampler(
            labels=LABELS,
            batch_size=6,
            num_classes=3,
            geometry_areas=invalid,
            geometry_stratified_pair=(0, 1),
        )


def test_actual_teacher_probability_wrapper_preserves_bbox_label_alignment() -> None:
    base = _GeometryDataset()
    probabilities = {
        index: [1.0 if class_index == LABELS[index] else 0.0 for class_index in range(3)]
        for index in range(len(base))
    }
    wrapped = TeacherProbabilityDataset(
        base,
        {},
        num_classes=3,
        probabilities_by_sample_index=probabilities,
    )

    sampler = build_weighted_sampler(
        wrapped,
        batch_size=6,
        num_classes=3,
        epoch_multiplier=1.0,
        pair01_geometry_stratified_sampling=True,
    )
    geometry = sampler.exposure_summary()["geometry_stratification"]

    assert wrapped.labels() == base.labels()
    assert wrapped.sample_bboxes() == base.sample_bboxes()
    assert geometry["planned_bin_exposure_counts"]["0"] == (
        geometry["planned_bin_exposure_counts"]["1"]
    )


def test_24_batch_prefix_keeps_pair_bin_schedules_aligned_and_on_target() -> None:
    class_counts = [1941, 541, 1920, 2520, 2293]
    labels = [
        class_index
        for class_index, count in enumerate(class_counts)
        for _ in range(count)
    ]
    areas = (
        [0.10] * 486
        + [0.20] * 970
        + [0.40] * 485
        + [0.08] * 217
        + [0.18] * 182
        + [0.35] * 142
        + [0.20] * sum(class_counts[2:])
    )
    sampler = StrictBalancedBatchSampler(
        labels=labels,
        batch_size=32,
        num_classes=5,
        epoch_multiplier=1.0,
        seed=42,
        geometry_areas=areas,
        geometry_stratified_pair=(0, 1),
    )
    geometry = sampler.exposure_summary()["geometry_stratification"]
    edges = geometry["bin_edges"]
    target = geometry["target_bin_distribution"]
    observed = {0: [0, 0, 0], 1: [0, 0, 0]}

    for batch_index, batch in enumerate(sampler):
        if batch_index >= 24:
            break
        for sample_index in batch:
            class_index = labels[sample_index]
            if class_index in observed:
                observed[class_index][
                    _area_bin(areas[sample_index], edges)
                ] += 1

    assert all(abs(observed[0][index] - observed[1][index]) <= 1 for index in range(3))
    for class_index in (0, 1):
        prefix_total = sum(observed[class_index])
        assert all(
            abs(observed[class_index][bin_index] - prefix_total * target[bin_index])
                <= 1.0 + 1e-12
            for bin_index in range(3)
        )

def test_pair01_geometry_cli_round_trip_and_conflict_guard() -> None:
    args = parse_args(["--pair01-geometry-stratified-sampling"])
    _, train_config, _ = build_configs(args)
    assert train_config.pair01_geometry_stratified_sampling is True
    assert train_config.balanced_epoch_sampling is True

    with pytest.raises(ValueError, match="strict balanced"):
        build_configs(
            parse_args(
                [
                    "--pair01-geometry-stratified-sampling",
                    "--disable-balanced-epoch-sampling",
                ]
            )
        )

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from trkh.core.config import TrainConfig
from trkh.data.dataset import TemperedClassBatchSampler
from trkh.data.leakage_groups import load_tempered_leakage_group_ids
from trkh.training.train import (
    build_configs,
    parse_args,
    resolve_imbalance_strategy,
)


class _PathLabelDataset:
    def __init__(self, paths: list[Path], labels: list[int]) -> None:
        self._paths = list(paths)
        self._labels = list(labels)

    def __len__(self) -> int:
        return len(self._paths)

    def sample_paths(self) -> list[Path]:
        return list(self._paths)

    def labels(self) -> list[int]:
        return list(self._labels)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "source_image",
                "output_image",
                "class_id",
                "class_name",
                "source",
                "source_split",
                "leakage_group",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(
    *,
    split: str,
    output_image: str,
    class_id: int,
    class_name: str,
    group: str,
) -> dict[str, object]:
    return {
        "split": split,
        "source_image": f"source/{group}.jpg",
        "output_image": output_image,
        "class_id": class_id,
        "class_name": class_name,
        "source": 0,
        "source_split": "train",
        "leakage_group": group,
    }


def test_group_balancing_preserves_tempered_quota_and_cycles_members() -> None:
    labels = [0] * 7 + [1] * 4
    group_ids = ["a"] * 5 + ["b", "c"] + ["d"] * 3 + ["e"]
    common = {
        "labels": labels,
        "batch_size": 4,
        "num_classes": 2,
        "power": 0.5,
        "epoch_multiplier": 4.0,
        "seed": 17,
    }
    baseline = TemperedClassBatchSampler(**common)
    grouped = TemperedClassBatchSampler(**common, group_ids=group_ids)

    assert grouped.exposure_counts() == baseline.exposure_counts()
    epoch_zero = list(grouped)
    assert epoch_zero == list(grouped)
    assert [
        Counter(labels[index] for index in batch) for batch in epoch_zero
    ] == [
        Counter(labels[index] for index in batch) for batch in baseline
    ]

    sampled_indices = [index for batch in epoch_zero for index in batch]
    group_exposure = Counter((labels[index], group_ids[index]) for index in sampled_indices)
    for class_index, class_groups in ((0, ("a", "b", "c")), (1, ("d", "e"))):
        counts = [group_exposure[(class_index, group_id)] for group_id in class_groups]
        assert max(counts) - min(counts) <= 1

    sample_exposure = Counter(sampled_indices)
    for class_index, group_id in ((0, "a"), (1, "d")):
        members = [
            index
            for index, (label, observed_group) in enumerate(zip(labels, group_ids))
            if label == class_index and observed_group == group_id
        ]
        counts = [sample_exposure[index] for index in members]
        assert max(counts) - min(counts) <= 1

    summary = grouped.exposure_summary()
    assert summary["group_sampling_enabled"] is True
    assert summary["source_group_counts_by_class"] == [3, 2]
    assert summary["source_max_group_sizes_by_class"] == [5, 3]
    assert summary["max_within_class_group_exposure_gap"] <= 1

    grouped.set_epoch(1)
    assert list(grouped) != epoch_zero
    assert grouped.exposure_counts() == baseline.exposure_counts()


def test_group_ids_default_off_preserves_legacy_iterator() -> None:
    common = {
        "labels": [0] * 9 + [1] * 3 + [2] * 5,
        "batch_size": 6,
        "num_classes": 3,
        "power": 0.5,
        "seed": 42,
    }
    omitted = TemperedClassBatchSampler(**common)
    explicit_none = TemperedClassBatchSampler(**common, group_ids=None)

    assert list(omitted) == list(explicit_none)
    assert "group_sampling_enabled" not in omitted.exposure_summary()


def test_sampler_provenance_distinguishes_tempered_rows_from_groups() -> None:
    row_tempered = resolve_imbalance_strategy(
        TrainConfig(tempered_class_sampling_power=0.5),
        [20, 5],
    )
    group_tempered = resolve_imbalance_strategy(
        TrainConfig(
            tempered_class_sampling_power=0.5,
            tempered_leakage_group_manifest="manifest.csv",
        ),
        [20, 5],
    )

    assert row_tempered["sampler_type"] == "tempered_class_prior"
    assert row_tempered["use_tempered_group_sampler"] is False
    assert group_tempered["sampler_type"] == (
        "tempered_class_leakage_group_uniform"
    )
    assert group_tempered["use_tempered_group_sampler"] is True


def test_group_manifest_cli_requires_tempered_class_sampling(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--data",
            str(tmp_path / "data.yaml"),
            "--tempered-leakage-group-manifest",
            str(tmp_path / "manifest.csv"),
        ]
    )

    with pytest.raises(ValueError, match="yeu cau"):
        build_configs(args)


@pytest.mark.parametrize(
    "group_ids",
    (
        ["a"],
        ["a", ""],
        ["a", "   "],
    ),
)
def test_group_sampler_rejects_invalid_group_ids(group_ids: list[str]) -> None:
    with pytest.raises(ValueError):
        TemperedClassBatchSampler(
            labels=[0, 1],
            batch_size=2,
            num_classes=2,
            power=0.5,
            group_ids=group_ids,
        )


def test_manifest_loader_maps_windows_paths_to_runtime_train_root(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "class_f" / "train"
    path0 = train_root / "C0" / "a.jpg"
    path1 = train_root / "C1" / "b.jpg"
    path0.parent.mkdir(parents=True)
    path1.parent.mkdir(parents=True)
    path0.touch()
    path1.touch()
    manifest = tmp_path / "class_f" / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                split="train",
                output_image=r"D:\old\class_f\train\C0\a.jpg",
                class_id=0,
                class_name="C0",
                group="g0",
            ),
            _manifest_row(
                split="train",
                output_image=r"D:\old\class_f\train\C1\b.jpg",
                class_id=1,
                class_name="C1",
                group="g1",
            ),
            _manifest_row(
                split="val",
                output_image=r"D:\old\class_f\val\C0\v.jpg",
                class_id=0,
                class_name="C0",
                group="gv",
            ),
        ],
    )

    dataset = _PathLabelDataset([path0, path1], [0, 1])
    group_ids, summary = load_tempered_leakage_group_ids(
        manifest,
        sample_paths=dataset.sample_paths(),
        labels=dataset.labels(),
        train_root=train_root,
        class_names=("C0", "C1"),
    )

    assert group_ids == ["g0", "g1"]
    assert summary["mapped_samples"] == 2
    assert summary["skipped_non_train_rows"] == 1
    assert summary["group_counts_by_class"] == [1, 1]
    assert len(str(summary["manifest_sha256"])) == 64


@pytest.mark.parametrize(
    "mutation, expected_message",
    (
        ("missing", "missing from manifest"),
        ("extra", "absent from final dataset"),
        ("duplicate", "Duplicate train output_image"),
        ("label_mismatch", "does not match manifest"),
        ("cross_split", "cross declared splits"),
    ),
)
def test_manifest_loader_fails_closed(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    train_root = tmp_path / mutation / "train"
    path0 = train_root / "C0" / "a.jpg"
    path1 = train_root / "C1" / "b.jpg"
    path0.parent.mkdir(parents=True)
    path1.parent.mkdir(parents=True)
    path0.touch()
    path1.touch()
    rows = [
        _manifest_row(
            split="train",
            output_image="train/C0/a.jpg",
            class_id=0,
            class_name="C0",
            group="g0",
        ),
        _manifest_row(
            split="train",
            output_image="train/C1/b.jpg",
            class_id=1,
            class_name="C1",
            group="g1",
        ),
    ]
    labels = [0, 1]
    if mutation == "missing":
        rows = rows[:1]
    elif mutation == "extra":
        rows.append(
            _manifest_row(
                split="train",
                output_image="train/C0/extra.jpg",
                class_id=0,
                class_name="C0",
                group="g-extra",
            )
        )
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "label_mismatch":
        labels = [1, 1]
    elif mutation == "cross_split":
        rows.append(
            _manifest_row(
                split="val",
                output_image="val/C0/v.jpg",
                class_id=0,
                class_name="C0",
                group="g0",
            )
        )
    manifest = tmp_path / mutation / "manifest.csv"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=expected_message):
        load_tempered_leakage_group_ids(
            manifest,
            sample_paths=[path0, path1],
            labels=labels,
            train_root=train_root,
            class_names=("C0", "C1"),
        )

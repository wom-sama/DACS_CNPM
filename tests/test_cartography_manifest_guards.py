import csv
from pathlib import Path

from trkh.tools.build_ambiguous_soft_targets import build_manifest as build_soft_targets
from trkh.tools.build_targeted_margin_manifest import (
    _parse_pairs,
    build_manifest as build_targeted_margin,
)


def _write_cartography_predictions(path: Path, train_root: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "target_index",
                "prediction_index",
                "top2_index",
                "top2_margin",
                "seen_count",
                "prob_0",
                "prob_1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(train_root / "class0" / "unseen.jpg"),
                "target_index": 0,
                "prediction_index": -1,
                "top2_index": -1,
                "top2_margin": 0.0,
                "seen_count": 0,
                "prob_0": 0.0,
                "prob_1": 0.0,
            }
        )
        writer.writerow(
            {
                "image_path": str(train_root / "class0" / "seen.jpg"),
                "target_index": 0,
                "prediction_index": 1,
                "top2_index": 0,
                "top2_margin": 0.05,
                "seen_count": 2,
                "prob_0": 0.45,
                "prob_1": 0.55,
            }
        )


def test_ambiguous_soft_targets_skip_unseen_cartography_rows(tmp_path):
    predictions = tmp_path / "cartography.csv"
    _write_cartography_predictions(predictions, tmp_path / "train")

    summary = build_soft_targets(
        predictions=predictions,
        output=tmp_path / "ambiguous.csv",
        pairs="0-1",
        alpha=0.22,
        margin_threshold=0.10,
        max_samples=0,
        require_train_paths=True,
        include_errors=True,
        include_low_margin=True,
    )

    assert summary["rows"] == 1
    assert summary["skipped_unseen"] == 1


def test_targeted_margin_skips_unseen_cartography_rows(tmp_path):
    predictions = tmp_path / "cartography.csv"
    _write_cartography_predictions(predictions, tmp_path / "train")

    summary = build_targeted_margin(
        predictions=predictions,
        output=tmp_path / "targeted_margin.csv",
        focus_class_index=1,
        pairs=_parse_pairs("0-1"),
        margin_threshold=0.10,
        target_margin=0.12,
        false_positive_weight=1.35,
        false_negative_weight=1.10,
        low_margin_weight=1.00,
        max_weight=1.60,
        max_samples=0,
        max_per_pair=0,
        include_low_margin_correct=False,
        allow_non_train_paths=False,
        dry_run=False,
    )

    assert summary["rows"] == 1
    assert summary["skipped_unseen"] == 1

from pathlib import Path

import pytest
import torch

from trkh.training.losses import LDAMFocalLoss
from trkh.evaluation.metrics import plot_confusion_matrix
from trkh.training.train import (
    build_configs,
    parse_args,
    resolve_ldam_class_counts,
)


NATURAL_COUNTS = [1941, 541, 1920, 2520, 2293]
SAMPLER_EXPOSURES = [1844, 1844, 1843, 1843, 1843]


def test_sampler_exposure_counts_remove_class_specific_ldam_margin_bias() -> None:
    resolved, summary = resolve_ldam_class_counts(
        NATURAL_COUNTS,
        source="sampler_exposure",
        sampler_exposure_summary={
            "enabled": True,
            "class_exposure_counts": SAMPLER_EXPOSURES,
        },
        max_margin=0.3,
    )
    criterion = LDAMFocalLoss(
        class_counts=resolved,
        max_margin=summary["effective_max_margin"],
        scale=18.0,
    )

    assert resolved == SAMPLER_EXPOSURES
    assert summary["sampler_aligned"] is True
    assert summary["effective_max_margin"] == pytest.approx(0.22998, abs=1e-4)
    assert summary["resolved_mean_margin"] == pytest.approx(
        summary["natural_mean_margin"],
        abs=1e-8,
    )
    assert torch.max(criterion.class_margins) - torch.min(criterion.class_margins) < 1e-4


def test_natural_count_source_preserves_existing_ldam_behavior() -> None:
    resolved, summary = resolve_ldam_class_counts(
        NATURAL_COUNTS,
        source="natural",
        max_margin=0.3,
    )
    criterion = LDAMFocalLoss(class_counts=resolved, max_margin=0.3, scale=18.0)

    assert resolved == NATURAL_COUNTS
    assert summary["sampler_aligned"] is False
    assert criterion.class_margins[1] == torch.max(criterion.class_margins)
    assert criterion.class_margins[1] - criterion.class_margins[0] == pytest.approx(
        0.082021,
        abs=1e-5,
    )
    legacy = LDAMFocalLoss(class_counts=NATURAL_COUNTS, max_margin=0.3, scale=18.0)
    logits = torch.tensor([[0.4, -0.1, 0.2, 0.0, -0.3]], dtype=torch.float32)
    target = torch.tensor([1])
    assert torch.equal(criterion.class_margins, legacy.class_margins)
    assert torch.equal(criterion(logits, target), legacy(logits, target))


def test_sampler_exposure_source_fails_closed_without_strict_sampler() -> None:
    with pytest.raises(ValueError, match="strict balanced sampler"):
        resolve_ldam_class_counts(
            NATURAL_COUNTS,
            source="sampler_exposure",
            sampler_exposure_summary={"enabled": False},
        )


def test_sampler_aware_ldam_cli_round_trip() -> None:
    args = parse_args(
        [
            "--classification-loss",
            "ldam_focal",
            "--ldam-class-count-source",
            "sampler_exposure",
        ]
    )
    _, train_config, _ = build_configs(args)

    assert train_config.ldam_class_count_source == "sampler_exposure"


def test_v8_launcher_exposes_sampler_aware_ldam() -> None:
    launcher = Path("scripts/run_trkh_5class_attention_views_v8.ps1").read_text(
        encoding="utf-8"
    )

    assert '[ValidateSet("natural", "sampler_exposure")]' in launcher
    assert '[string]$LdamClassCountSource = "natural"' in launcher
    assert '"--ldam-class-count-source", "$LdamClassCountSource"' in launcher


def test_confusion_plot_exports_column_normalized_precision_view(tmp_path: Path) -> None:
    output_path = tmp_path / "column_normalized.png"
    plot_confusion_matrix(
        confusion=[[8, 2], [1, 9]],
        class_names=["a", "b"],
        output_path=output_path,
        normalize=True,
        normalize_by="predicted",
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_confusion_plot_rejects_unknown_normalization_axis(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="normalize_by"):
        plot_confusion_matrix(
            confusion=[[1, 0], [0, 1]],
            class_names=["a", "b"],
            output_path=tmp_path / "invalid.png",
            normalize=True,
            normalize_by="diagonal",
        )

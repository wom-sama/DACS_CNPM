from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.audit_deepten_stem_texture_readiness import (
    BATCH_SIZE,
    CODEWORDS,
    EPOCHS,
    FOLDS,
    GRID_SIZE,
    HIDDEN_DIM,
    INTERIOR_ERODE_RATIO,
    LEARNING_RATE,
    OFFICIAL_SOURCE_COMMIT,
    PROJECTION_DIM,
    RESIDUAL_SCALE,
    DeepTENResidualHead,
    MaskedTextureEncoding,
    MeanStdResidualControl,
    _validate_protocol,
    assess_deepten_stem_texture_readiness,
    build_interior_grid_mask,
    parse_args,
    residual_probabilities,
    select_xai_cases,
)


def _args(tmp_path, *extra: str):
    return parse_args(
        [
            "--data",
            str(tmp_path / "dataset" / "data.yaml"),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_protocol_defaults_are_locked(tmp_path) -> None:
    args = _args(tmp_path)
    assert args.grid_size == GRID_SIZE == 12
    assert args.projection_dim == PROJECTION_DIM == 32
    assert args.codewords == CODEWORDS == 8
    assert args.hidden_dim == HIDDEN_DIM == 64
    assert args.folds == FOLDS == 5
    assert args.epochs == EPOCHS == 20
    assert args.batch_size == BATCH_SIZE == 256
    assert args.learning_rate == LEARNING_RATE == 1e-3
    assert args.residual_scale == RESIDUAL_SCALE == 0.12
    assert args.interior_erode_ratio == INTERIOR_ERODE_RATIO == 0.12
    assert OFFICIAL_SOURCE_COMMIT == "ac748410dfc8d7d70a2ce7f5add08050af2fae20"
    _validate_protocol(args)


def test_protocol_rejects_non_class1_and_over_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="class 1 only"):
        _validate_protocol(_args(tmp_path, "--focus-class-index", "2"))
    with pytest.raises(ValueError, match="epoch budget"):
        _validate_protocol(_args(tmp_path, "--epochs", "31"))


def test_interior_mask_respects_bbox_erosion_and_valid_padding() -> None:
    bbox = torch.tensor([[0.5, 0.5, 0.8, 0.8]], dtype=torch.float32)
    valid = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    valid[:, :, :, -2:] = False
    mask = build_interior_grid_mask(
        bbox,
        valid,
        grid_size=8,
        erode_ratio=0.125,
    ).view(1, 8, 8)
    assert mask.any()
    assert not mask[:, :, -2:].any()
    assert not mask[:, 0].any()
    assert mask[:, 3:5, 3:5].all()


def test_interior_mask_guarantees_nine_nearest_valid_tokens() -> None:
    bbox = torch.tensor([[0.5, 0.5, 0.01, 0.01]], dtype=torch.float32)
    valid = torch.ones((1, 1, 48, 48), dtype=torch.bool)

    mask = build_interior_grid_mask(
        bbox,
        valid,
        grid_size=12,
        erode_ratio=0.12,
    ).view(12, 12)

    assert int(mask.sum()) == 9
    coordinates = torch.nonzero(mask, as_tuple=False).float()
    assert torch.allclose(coordinates.mean(dim=0), torch.tensor([5.5, 5.5]), atol=0.75)


def test_masked_texture_encoding_ignores_masked_descriptors() -> None:
    torch.manual_seed(7)
    layer = MaskedTextureEncoding(feature_dim=3, codewords=4)
    descriptors = torch.randn(2, 6, 3)
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True, False, True, False, True, False]]
    )
    changed = descriptors.clone()
    changed[~mask] = 1000.0
    expected = layer(descriptors, mask)
    actual = layer(changed, mask)
    assert expected.shape == (2, 4, 3)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=0.0)


def test_deepten_codewords_scales_and_inputs_receive_gradients() -> None:
    torch.manual_seed(11)
    layer = MaskedTextureEncoding(feature_dim=4, codewords=3)
    descriptors = torch.randn(3, 7, 4, requires_grad=True)
    mask = torch.ones((3, 7), dtype=torch.bool)
    output = layer(descriptors, mask)
    output.square().mean().backward()
    assert descriptors.grad is not None and torch.isfinite(descriptors.grad).all()
    assert layer.codewords.grad is not None and layer.codewords.grad.abs().sum() > 0
    assert layer.scale.grad is not None and layer.scale.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "model",
    [
        MeanStdResidualControl(feature_dim=4, hidden_dim=8, class_count=5),
        DeepTENResidualHead(feature_dim=4, codewords=3, hidden_dim=8, class_count=5),
    ],
)
def test_zero_initialized_residual_preserves_keeper_probabilities(model) -> None:
    torch.manual_seed(13)
    descriptors = torch.randn(2, 9, 4)
    mask = torch.ones((2, 9), dtype=torch.bool)
    base = torch.tensor(
        [[0.60, 0.10, 0.10, 0.10, 0.10], [0.05, 0.65, 0.10, 0.10, 0.10]],
        dtype=torch.float32,
    )
    probabilities, residual = residual_probabilities(
        model,
        descriptors,
        mask,
        base,
        residual_scale=RESIDUAL_SCALE,
    )
    assert torch.count_nonzero(residual) == 0
    assert torch.allclose(probabilities, base, atol=1e-7, rtol=0.0)


def _metrics(macro: float, focus_f1: float, focus_recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.85, "recall": 0.85},
            {"f1": focus_f1, "precision": 0.75, "recall": focus_recall},
        ],
    }


def _transitions() -> dict:
    return {
        "changed": 20,
        "corrections": 12,
        "harms": 5,
        "neutral": 3,
        "focus_false_positive_removed": 8,
        "focus_false_positive_created": 2,
        "focus_false_negative_rescued": 5,
        "focus_true_positive_broken": 1,
    }


def _direction(auc: float) -> dict:
    return {"auc_fn_positive": auc, "samples": 30}


def test_gate_accepts_recall_safe_texture_signal() -> None:
    result = assess_deepten_stem_texture_readiness(
        train_rows=9215,
        val_rows=2606,
        train_val_source_overlap=0,
        fold_source_overlap=0,
        features_finite=True,
        minimum_mask_tokens=16,
        codeword_utilization_entropy=0.88,
        scale_positive_fraction=0.0,
        keeper_oof=_metrics(0.94, 0.81, 0.82),
        control_oof=_metrics(0.94, 0.80, 0.81),
        candidate_oof=_metrics(0.945, 0.82, 0.83),
        keeper_val=_metrics(0.884, 0.684, 0.781),
        control_val=_metrics(0.884, 0.68, 0.77),
        candidate_val=_metrics(0.90, 0.72, 0.79),
        transitions_vs_keeper=_transitions(),
        candidate_vs_control_oof_direction=_direction(0.68),
        candidate_vs_control_val_direction=_direction(0.66),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["image_smoke_permission"]
    assert result["failed_checks"] == []


def test_gate_rejects_collapsed_recall_damaging_texture_signal() -> None:
    transitions = _transitions()
    transitions.update(
        {
            "corrections": 3,
            "harms": 15,
            "focus_false_negative_rescued": 0,
            "focus_true_positive_broken": 10,
        }
    )
    result = assess_deepten_stem_texture_readiness(
        train_rows=2000,
        val_rows=500,
        train_val_source_overlap=1,
        fold_source_overlap=1,
        features_finite=False,
        minimum_mask_tokens=2,
        codeword_utilization_entropy=0.20,
        scale_positive_fraction=0.25,
        keeper_oof=_metrics(0.94, 0.81, 0.82),
        control_oof=_metrics(0.94, 0.80, 0.81),
        candidate_oof=_metrics(0.92, 0.70, 0.68),
        keeper_val=_metrics(0.884, 0.684, 0.781),
        control_val=_metrics(0.88, 0.67, 0.75),
        candidate_val=_metrics(0.85, 0.60, 0.55),
        transitions_vs_keeper=transitions,
        candidate_vs_control_oof_direction=_direction(0.42),
        candidate_vs_control_val_direction=_direction(0.80),
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["image_smoke_permission"]
    assert "codeword_utilization_entropy_ge_0p55" in result["failed_checks"]
    assert "all_scales_nonpositive" in result["failed_checks"]
    assert "val_focus_recall_preserved_within_0p01" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]


def test_xai_selection_balances_class1_transition_types() -> None:
    labels = np.asarray([1, 1, 0, 2, 1, 4, 0, 2], dtype=np.int64)
    keeper_predictions = np.asarray([0, 1, 1, 0, 2, 1, 0, 2], dtype=np.int64)
    candidate_predictions = np.asarray([1, 0, 0, 1, 1, 4, 1, 0], dtype=np.int64)
    keeper = np.eye(5, dtype=np.float32)[keeper_predictions]
    candidate = np.eye(5, dtype=np.float32)[candidate_predictions]
    selected = select_xai_cases(labels, keeper, candidate, maximum_cases=8)
    categories = {row["category"] for row in selected}
    assert {
        "focus_fn_rescued",
        "focus_tp_broken",
        "focus_fp_removed",
        "focus_fp_created",
    }.issubset(categories)
    assert len({row["row_index"] for row in selected}) == len(selected)

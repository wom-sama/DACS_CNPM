from __future__ import annotations

import pytest
import torch

from trkh.models.photometric_invariant import (
    ColorInvariantW,
    apply_diagonal_color_constancy,
    estimate_gray_edge_illuminant,
    invariant_response_to_model_view,
    suppress_invalid_boundary,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _effective_rank,
    assess_photometric_invariant_readiness,
)
from trkh.tools.probe_context_gray_edge_color_constancy import (
    assess_context_color_constancy_readiness,
)


def test_color_invariant_w_is_finite_and_standardized() -> None:
    generator = torch.Generator().manual_seed(11)
    image = 0.15 + 0.55 * torch.rand((3, 3, 48, 48), generator=generator)
    response = ColorInvariantW(scale=0.0)(image)
    assert response.shape == (3, 1, 48, 48)
    assert torch.isfinite(response).all()
    assert torch.allclose(response.mean(dim=(-2, -1)), torch.zeros((3, 1)), atol=1e-4)
    assert torch.allclose(response.std(dim=(-2, -1), unbiased=False), torch.ones((3, 1)), atol=2e-3)


def test_color_invariant_w_resists_global_intensity_scaling() -> None:
    y = torch.linspace(0.0, 1.0, 40).view(1, 1, 40, 1)
    x = torch.linspace(0.0, 1.0, 40).view(1, 1, 1, 40)
    image = torch.cat(
        (
            0.10 + 0.18 * x.expand(1, 1, 40, 40),
            0.12 + 0.16 * y.expand(1, 1, 40, 40),
            0.11 + 0.10 * (x + y),
        ),
        dim=1,
    )
    invariant = ColorInvariantW(scale=0.0)
    first = invariant(image)
    second = invariant(image * 1.8)
    assert float((first - second).abs().mean()) < 0.03


def test_invalid_boundary_suppression_removes_padding_halo() -> None:
    response = torch.ones((1, 1, 8, 8))
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    valid[:, :, :2] = False
    masked = suppress_invalid_boundary(response, valid, boundary_width=1)
    assert torch.count_nonzero(masked[:, :, :, :3]) == 0
    assert torch.all(masked[:, :, :, 4:] == 1)


def test_invariant_response_can_be_replicated_for_frozen_rgb_probe() -> None:
    response = torch.tensor([[[[-5.0, 0.0, 5.0]]]])
    view = invariant_response_to_model_view(response, clip=3.0)
    assert view.shape == (1, 3, 1, 3)
    assert torch.equal(view[:, 0], view[:, 1])
    assert float(view.min()) == pytest.approx(-3.0)
    assert float(view.max()) == pytest.approx(3.0)


def test_effective_rank_handles_numpy_features() -> None:
    features = torch.eye(8).numpy()
    rank = _effective_rank(features)
    assert 6.9 <= rank <= 7.1


def test_gray_edge_recovers_colored_illuminant_from_achromatic_structure() -> None:
    y = torch.arange(48).view(1, 1, 48, 1)
    x = torch.arange(48).view(1, 1, 1, 48)
    checker = ((x // 6 + y // 6) % 2).float()
    reflectance = 0.12 + 0.28 * checker
    light = torch.tensor([[1.30, 0.90, 0.60]], dtype=torch.float32)
    image = reflectance.repeat(1, 3, 1, 1) * light.view(1, 3, 1, 1)
    estimated, support = estimate_gray_edge_illuminant(
        image,
        sigma=1.0,
        minkowski_p=6.0,
    )
    expected = light / light.norm(dim=1, keepdim=True)
    assert torch.nn.functional.cosine_similarity(estimated, expected).item() > 0.999
    assert support.item() > 0.5

    corrected, gains = apply_diagonal_color_constancy(image, estimated)
    channel_means = corrected.mean(dim=(-2, -1))
    assert float(channel_means.std(dim=1).item()) < 1e-4
    assert gains.shape == (1, 3)


def test_gray_edge_excludes_target_bbox_and_invalid_padding() -> None:
    image = torch.rand((2, 3, 40, 40), generator=torch.Generator().manual_seed(4)) * 0.7 + 0.1
    valid = torch.ones((2, 40, 40), dtype=torch.bool)
    valid[:, :5] = False
    bbox = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.4, 0.6, 0.2, 0.3]])
    illuminant, support = estimate_gray_edge_illuminant(
        image,
        valid_mask=valid,
        excluded_bbox=bbox,
        sigma=1.0,
    )
    assert illuminant.shape == (2, 3)
    assert torch.isfinite(illuminant).all()
    assert torch.all(support > 0.1)
    assert torch.all(support < 0.8)


def test_photometric_gate_requires_fold_safe_and_class1_gains() -> None:
    ready = assess_photometric_invariant_readiness(
        train_samples=9215,
        val_samples=2606,
        alignment_ok=True,
        rgb_effective_rank=60.0,
        invariant_effective_rank=80.0,
        invariant_border_mass=0.20,
        direct_metrics={"macro_f1": 0.884, "focus_f1": 0.686},
        rgb_oof_metrics={"macro_f1": 0.84, "focus_f1": 0.64},
        combined_oof_metrics={"macro_f1": 0.85, "focus_f1": 0.66},
        rgb_val_metrics={"macro_f1": 0.87, "focus_f1": 0.68},
        combined_val_metrics={"macro_f1": 0.885, "focus_f1": 0.705},
        direct_transitions={
            "corrections": 20,
            "harms": 10,
            "class1_fn_rescued": 8,
            "class1_tp_broken": 2,
            "class1_fp_removed": 10,
            "class1_fp_created": 3,
        },
        class1_error_delta_auc=0.70,
    )
    assert ready["smoke_ready"] is True
    assert ready["full_train_permission"] is False

    rejected = assess_photometric_invariant_readiness(
        train_samples=9215,
        val_samples=2606,
        alignment_ok=True,
        rgb_effective_rank=60.0,
        invariant_effective_rank=80.0,
        invariant_border_mass=0.20,
        direct_metrics={"macro_f1": 0.884, "focus_f1": 0.686},
        rgb_oof_metrics={"macro_f1": 0.84, "focus_f1": 0.64},
        combined_oof_metrics={"macro_f1": 0.841, "focus_f1": 0.645},
        rgb_val_metrics={"macro_f1": 0.87, "focus_f1": 0.68},
        combined_val_metrics={"macro_f1": 0.871, "focus_f1": 0.685},
        direct_transitions={
            "corrections": 5,
            "harms": 12,
            "class1_fn_rescued": 2,
            "class1_tp_broken": 5,
            "class1_fp_removed": 3,
            "class1_fp_created": 9,
        },
        class1_error_delta_auc=0.48,
    )
    assert rejected["smoke_ready"] is False
    assert "val_class1_milestone" in rejected["failed_checks"]
    assert "class1_false_positive_control" in rejected["failed_checks"]


def test_context_color_constancy_gate_accepts_only_a_safe_path() -> None:
    ready = assess_context_color_constancy_readiness(
        train_samples=9215,
        val_samples=2606,
        alignment_ok=True,
        mean_support_fraction=0.55,
        clipped_gain_fraction=0.01,
        rgb_effective_rank=80.0,
        corrected_effective_rank=90.0,
        base_metrics={"macro_f1": 0.884, "focus_f1": 0.686},
        corrected_metrics={"macro_f1": 0.889, "focus_f1": 0.705},
        corrected_transitions={
            "corrections": 18,
            "harms": 8,
            "class1_fn_rescued": 7,
            "class1_tp_broken": 2,
            "class1_fp_removed": 8,
            "class1_fp_created": 3,
        },
        rgb_oof_metrics={"macro_f1": 0.93, "focus_f1": 0.79},
        combined_oof_metrics={"macro_f1": 0.92, "focus_f1": 0.78},
        rgb_val_metrics={"macro_f1": 0.88, "focus_f1": 0.67},
        combined_val_metrics={"macro_f1": 0.87, "focus_f1": 0.64},
        combined_transitions={
            "corrections": 5,
            "harms": 9,
            "class1_fn_rescued": 1,
            "class1_tp_broken": 4,
            "class1_fp_removed": 3,
            "class1_fp_created": 5,
        },
        class1_error_delta_auc=0.50,
    )
    assert ready["smoke_ready"] is True
    assert ready["selected_path"] == "direct_preprocess"
    assert ready["full_train_permission"] is False

    rejected = assess_context_color_constancy_readiness(
        train_samples=9215,
        val_samples=2606,
        alignment_ok=True,
        mean_support_fraction=0.55,
        clipped_gain_fraction=0.01,
        rgb_effective_rank=80.0,
        corrected_effective_rank=90.0,
        base_metrics={"macro_f1": 0.884, "focus_f1": 0.686},
        corrected_metrics={"macro_f1": 0.875, "focus_f1": 0.65},
        corrected_transitions={
            "corrections": 5,
            "harms": 15,
            "class1_fn_rescued": 2,
            "class1_tp_broken": 8,
            "class1_fp_removed": 3,
            "class1_fp_created": 7,
        },
        rgb_oof_metrics={"macro_f1": 0.93, "focus_f1": 0.79},
        combined_oof_metrics={"macro_f1": 0.925, "focus_f1": 0.78},
        rgb_val_metrics={"macro_f1": 0.88, "focus_f1": 0.67},
        combined_val_metrics={"macro_f1": 0.87, "focus_f1": 0.64},
        combined_transitions={
            "corrections": 4,
            "harms": 12,
            "class1_fn_rescued": 1,
            "class1_tp_broken": 6,
            "class1_fp_removed": 2,
            "class1_fp_created": 8,
        },
        class1_error_delta_auc=0.45,
    )
    assert rejected["smoke_ready"] is False
    assert rejected["selected_path"] == "none"

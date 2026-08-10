from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.audit_foveal_aggregated_attention_pair import (
    _RegionPerturbationDataset,
    _binary_auroc,
    _event_categories,
    _margin,
    _prefix_attention_map,
    _token_gradcam,
)


def test_binary_auroc_handles_ties_and_perfect_order() -> None:
    assert _binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert _binary_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both positive and negative"):
        _binary_auroc([1, 1], [0.2, 0.3])


def test_class1_event_categories_are_restricted_and_directional() -> None:
    assert _event_categories(0, 1, 0) == [
        "changed_class1_event",
        "restricted_fp_removal",
    ]
    assert _event_categories(4, 4, 1) == [
        "changed_class1_event",
        "restricted_fp_creation",
    ]
    assert _event_categories(1, 1, 2) == [
        "changed_class1_event",
        "class1_tp_break",
    ]
    assert _event_categories(1, 2, 1) == [
        "changed_class1_event",
        "class1_fn_rescue",
    ]
    assert _event_categories(3, 3, 2) == []


def test_margin_uses_only_locked_restricted_negative_classes() -> None:
    logits = torch.tensor(
        [[2.0, 3.0, 1.0, 100.0, 2.5], [0.0, -1.0, -2.0, 50.0, -3.0]]
    )
    torch.testing.assert_close(_margin(logits), torch.tensor([0.5, -1.0]))


def test_token_gradcam_and_prefix_attention_maps_have_locked_grid_shape() -> None:
    activations = torch.ones(2, 7, 3)
    gradients = torch.zeros_like(activations)
    gradients[:, 3:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    heat = _token_gradcam(
        activations,
        gradients,
        prefix_count=3,
        grid_size=(2, 2),
    )
    assert heat.shape == (2, 2, 2)
    assert torch.isfinite(heat).all()

    attention = torch.zeros(2, 2, 7, 7)
    attention[:, :, :3, 3:] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    prefix = _prefix_attention_map(
        attention,
        prefix_count=3,
        grid_size=(2, 2),
    )
    assert prefix.shape == (2, 2, 2)
    assert float(prefix[:, -1, -1].mean()) == pytest.approx(1.0)


def test_region_perturbations_change_only_requested_region() -> None:
    from PIL import Image

    array = np.zeros((20, 20, 3), dtype=np.uint8)
    array[:, :10] = (255, 0, 0)
    array[:, 10:] = (0, 255, 0)
    image = Image.fromarray(array)
    bbox = torch.tensor([0.5, 0.5, 0.4, 0.4])
    object_view = np.asarray(
        _RegionPerturbationDataset._perturb(image, bbox, mode="object")
    )
    background_view = np.asarray(
        _RegionPerturbationDataset._perturb(image, bbox, mode="far_background")
    )
    assert np.array_equal(object_view[0, 0], array[0, 0])
    assert not np.array_equal(object_view[10, 10], array[10, 10])
    assert not np.array_equal(background_view[0, 0], array[0, 0])
    assert np.array_equal(background_view[10, 10], array[10, 10])

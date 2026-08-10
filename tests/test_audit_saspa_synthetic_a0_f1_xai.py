from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from trkh.tools.audit_saspa_synthetic_a0_f1_xai import (
    bbox_mask,
    heatmap_bbox_focus,
    normalize_input_gradient,
    perturb_image,
)


def _test_image() -> Image.Image:
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, :] = (20, 80, 160)
    array[2:6, 2:6] = (210, 60, 10)
    return Image.fromarray(array)


def test_bbox_mask_uses_locked_normalized_xywh() -> None:
    mask = bbox_mask((8, 8), (0.5, 0.5, 0.5, 0.5))
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 16
    assert bool(mask[2:6, 2:6].all())
    assert not bool(mask[:2].any())


def test_bbox_mask_clamps_out_of_bounds_box() -> None:
    mask = bbox_mask((8, 6), (0.0, 0.0, 0.6, 0.8))
    assert mask.shape == (6, 8)
    assert int(mask.sum()) > 0
    assert bool(mask[0, 0])


def test_background_gray_preserves_bbox_object() -> None:
    image = _test_image()
    mask = bbox_mask(image.size, (0.5, 0.5, 0.5, 0.5))
    result = np.asarray(perturb_image(image, mask, "background_gray"))
    original = np.asarray(image)
    assert np.array_equal(result[mask], original[mask])
    assert np.all(result[~mask, 0] == result[~mask, 1])
    assert np.all(result[~mask, 1] == result[~mask, 2])


def test_object_desaturate_preserves_background() -> None:
    image = _test_image()
    mask = bbox_mask(image.size, (0.5, 0.5, 0.5, 0.5))
    result = np.asarray(perturb_image(image, mask, "object_desaturate"))
    original = np.asarray(image)
    assert np.array_equal(result[~mask], original[~mask])
    assert np.all(result[mask, 0] == result[mask, 1])
    assert np.all(result[mask, 1] == result[mask, 2])


def test_unknown_perturbation_is_rejected() -> None:
    image = _test_image()
    mask = bbox_mask(image.size, (0.5, 0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb_image(image, mask, "unknown")


def test_input_gradient_normalization_is_finite_and_bounded() -> None:
    gradient = torch.zeros(1, 3, 2, 2)
    gradient[0, 0, 0, 0] = 3.0
    gradient[0, 1, 0, 0] = 4.0
    heatmap = normalize_input_gradient(gradient)
    assert heatmap.shape == (2, 2)
    assert float(heatmap.max()) == pytest.approx(1.0)
    assert float(heatmap.min()) == pytest.approx(0.0)


def test_input_gradient_rejects_nonfinite_values() -> None:
    gradient = torch.zeros(1, 3, 2, 2)
    gradient[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        normalize_input_gradient(gradient)


def test_heatmap_bbox_focus_reports_exact_mass() -> None:
    mask = bbox_mask((4, 4), (0.5, 0.5, 0.5, 0.5))
    heatmap = np.ones((4, 4), dtype=np.float32)
    focus = heatmap_bbox_focus(heatmap, mask)
    assert focus["bbox_mass"] == pytest.approx(0.25)
    assert focus["outside_bbox_mass"] == pytest.approx(0.75)

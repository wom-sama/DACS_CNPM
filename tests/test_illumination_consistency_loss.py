import torch

from trkh.training.train import (
    _augmix_consistency_loss,
    _foreground_chroma_consistency_loss,
    _illumination_consistency_loss,
    _sample_augmix_consistency_images,
    _sample_foreground_chroma_consistency_images,
    _sample_illumination_consistency_images,
)


def test_sample_illumination_consistency_images_keeps_shape_and_finite_values():
    torch.manual_seed(7)
    images = torch.randn(3, 3, 16, 16)

    augmented = _sample_illumination_consistency_images(
        images,
        brightness=0.06,
        contrast=0.05,
        gamma=0.08,
    )

    assert augmented.shape == images.shape
    assert torch.isfinite(augmented).all()
    assert not torch.allclose(augmented, images)


def test_illumination_consistency_loss_backpropagates_to_both_logits():
    logits_a = torch.randn(4, 5, requires_grad=True)
    logits_b = torch.randn(4, 5, requires_grad=True)

    loss = _illumination_consistency_loss(logits_a, logits_b, temperature=0.9)
    loss.backward()

    assert loss.item() >= 0.0
    assert logits_a.grad is not None
    assert logits_b.grad is not None
    assert torch.isfinite(logits_a.grad).all()
    assert torch.isfinite(logits_b.grad).all()
    assert logits_a.grad.abs().sum().item() > 0.0
    assert logits_b.grad.abs().sum().item() > 0.0


def test_sample_augmix_consistency_images_keeps_shape_and_changes_values():
    torch.manual_seed(13)
    images = torch.randn(4, 3, 18, 18)

    augmented = _sample_augmix_consistency_images(
        images,
        severity=0.20,
        width=2,
        depth=2,
        alpha=1.0,
    )

    assert augmented.shape == images.shape
    assert torch.isfinite(augmented).all()
    assert not torch.allclose(augmented, images)


def test_augmix_consistency_loss_zero_for_identical_logits():
    logits = torch.randn(4, 5, requires_grad=True)

    loss = _augmix_consistency_loss(logits, logits, logits, temperature=1.0)
    loss.backward()

    assert abs(loss.item()) < 1e-6
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_augmix_consistency_loss_backpropagates_to_all_logits():
    logits_clean = torch.randn(4, 5, requires_grad=True)
    logits_aug_a = torch.randn(4, 5, requires_grad=True)
    logits_aug_b = torch.randn(4, 5, requires_grad=True)

    loss = _augmix_consistency_loss(
        logits_clean,
        logits_aug_a,
        logits_aug_b,
        temperature=0.9,
    )
    loss.backward()

    assert loss.item() >= 0.0
    for logits in (logits_clean, logits_aug_a, logits_aug_b):
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum().item() > 0.0


def test_sample_foreground_chroma_consistency_images_uses_bbox_mask():
    torch.manual_seed(11)
    images = torch.zeros(2, 3, 12, 12)
    images[:, 0] = 1.0
    bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    selected = torch.tensor([True, False])

    augmented, fraction, mask_fraction = _sample_foreground_chroma_consistency_images(
        images,
        bboxes,
        selected_mask=selected,
        saturation_delta=0.25,
        hue_delta=0.04,
        bbox_margin_ratio=0.0,
    )

    assert augmented.shape == images.shape
    assert torch.isfinite(augmented).all()
    assert fraction == 0.5
    assert mask_fraction > 0.0
    assert not torch.allclose(augmented[0], images[0])
    torch.testing.assert_close(augmented[1], images[1])


def test_foreground_chroma_consistency_loss_backpropagates_to_both_logits():
    logits_a = torch.randn(4, 5, requires_grad=True)
    logits_b = torch.randn(4, 5, requires_grad=True)

    loss = _foreground_chroma_consistency_loss(logits_a, logits_b, temperature=1.1)
    loss.backward()

    assert loss.item() >= 0.0
    assert logits_a.grad is not None
    assert logits_b.grad is not None
    assert torch.isfinite(logits_a.grad).all()
    assert torch.isfinite(logits_b.grad).all()

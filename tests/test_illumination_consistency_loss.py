import torch

from trkh.training.train import (
    _augmix_consistency_loss,
    _foreground_chroma_consistency_loss,
    _illumination_consistency_loss,
    _pairwise_relighting_margin_retention_loss,
    _sample_augmix_consistency_images,
    _sample_foreground_chroma_consistency_images,
    _sample_illumination_consistency_images,
    _sample_pairwise_relighting_images,
    train_one_epoch,
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


def test_pairwise_relighting_balances_polarity_and_preserves_unclipped_chroma():
    torch.manual_seed(19)
    rgb = torch.full((4, 3, 8, 8), 0.45)
    rgb[:, 0] += 0.04
    rgb[:, 2] -= 0.03
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (rgb - mean) / std

    relit = _sample_pairwise_relighting_images(
        images,
        brightness=0.25,
        contrast=0.10,
    )
    relit_rgb = relit * std + mean

    assert relit.shape == images.shape
    assert torch.isfinite(relit).all()
    assert (relit_rgb.mean(dim=(1, 2, 3)) > rgb.mean()).sum().item() == 2
    assert (relit_rgb.mean(dim=(1, 2, 3)) < rgb.mean()).sum().item() == 2
    torch.testing.assert_close(
        relit_rgb[:, 0] - relit_rgb[:, 1],
        rgb[:, 0] - rgb[:, 1],
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        relit_rgb[:, 2] - relit_rgb[:, 1],
        rgb[:, 2] - rgb[:, 1],
        atol=1e-6,
        rtol=0.0,
    )


def test_pairwise_relighting_randomizes_unmatched_polarity_for_odd_batches():
    torch.manual_seed(1901)
    rgb = torch.full((3, 3, 8, 8), 0.45)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (rgb - mean) / std
    extra_dim = 0
    extra_bright = 0

    for _ in range(32):
        relit = _sample_pairwise_relighting_images(
            images,
            brightness=0.25,
            contrast=0.10,
        )
        relit_rgb = relit * std + mean
        dim_count = int(
            (relit_rgb.mean(dim=(1, 2, 3)) < rgb.mean()).sum().item()
        )
        bright_count = int(
            (relit_rgb.mean(dim=(1, 2, 3)) > rgb.mean()).sum().item()
        )
        assert sorted((dim_count, bright_count)) == [1, 2]
        extra_dim += int(dim_count == 2)
        extra_bright += int(bright_count == 2)

    assert extra_dim > 0
    assert extra_bright > 0


def test_pairwise_relighting_margin_retention_is_zero_when_positive_margins_are_kept():
    clean = torch.tensor(
        [[1.2, 3.0, 0.7, -0.2, 0.1], [0.3, -0.4, 2.2, 0.0, 0.1]],
        requires_grad=True,
    )
    relit = clean.detach().clone().requires_grad_(True)
    targets = torch.tensor([1, 2])

    loss = _pairwise_relighting_margin_retention_loss(
        clean,
        relit,
        targets,
        focus_class=1,
        negative_classes="0,2,4",
        retention=0.8,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert clean.grad is None
    assert relit.grad is not None
    assert relit.grad.abs().sum().item() == 0.0


def test_pairwise_relighting_margin_retention_only_updates_relit_branch_in_both_directions():
    clean = torch.tensor(
        [[0.0, 2.0, 0.5, -1.0, 0.2], [0.0, -0.5, 2.0, -1.0, 0.1]],
        requires_grad=True,
    )
    relit = torch.tensor(
        [[0.0, 0.4, 0.5, -1.0, 0.2], [0.0, 0.8, 0.6, -1.0, 0.1]],
        requires_grad=True,
    )
    targets = torch.tensor([1, 2])

    loss = _pairwise_relighting_margin_retention_loss(
        clean,
        relit,
        targets,
        focus_class=1,
        negative_classes=(0, 2, 4),
        retention=0.8,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert clean.grad is None
    assert relit.grad is not None
    assert torch.isfinite(relit.grad).all()
    assert relit.grad[0, 1].item() < 0.0
    assert relit.grad[0, 2].item() > 0.0
    assert relit.grad[1, 2].item() < 0.0
    assert relit.grad[1, 1].item() > 0.0


def test_pairwise_relighting_margin_retention_excludes_wrong_clean_pairs():
    clean = torch.tensor([[2.0, 0.1, 1.0, 0.0, 0.5]], requires_grad=True)
    relit = torch.tensor([[0.0, -2.0, 3.0, 0.0, 0.5]], requires_grad=True)

    loss = _pairwise_relighting_margin_retention_loss(
        clean,
        relit,
        torch.tensor([1]),
        focus_class=1,
        negative_classes="0,2,4",
        retention=1.0,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert clean.grad is None
    assert relit.grad is not None
    assert relit.grad.abs().sum().item() == 0.0


def test_pairwise_relighting_margin_retention_balances_protect_and_suppress_directions():
    clean = torch.tensor(
        [[0.0, 2.0, 0.0, 4.0, 0.0], [0.0, 0.0, 2.0, 4.0, 0.0]],
    )
    relit = torch.tensor(
        [[0.0, 0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 1.0, 4.0, 0.0]],
        requires_grad=True,
    )

    loss, stats = _pairwise_relighting_margin_retention_loss(
        clean,
        relit,
        torch.tensor([1, 2]),
        focus_class=1,
        negative_classes="0;2;4",
        retention=1.0,
        return_stats=True,
    )

    torch.testing.assert_close(loss, torch.tensor(1.5))
    assert stats["protect_active_samples"].item() == 1.0
    assert stats["protect_active_pairs"].item() == 3.0
    assert stats["suppress_active_samples"].item() == 1.0
    assert stats["protect_violation_count"].item() == 3.0
    assert stats["suppress_violation_count"].item() == 1.0
    assert stats["protect_violation_fraction"].item() == 1.0
    assert stats["suppress_violation_fraction"].item() == 1.0
    assert stats["active_directions"].item() == 2.0


class _CountingImageClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3 * 4 * 4, 5)
        self.forward_count = 0

    def forward(self, images):
        self.forward_count += 1
        return self.classifier(images.flatten(1))


def _one_batch_forward_count(*, epoch: int, start_epoch: int, soft_targets: bool) -> int:
    torch.manual_seed(23)
    model = _CountingImageClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    hard_targets = torch.tensor([1, 2, 0, 4])
    targets = (
        torch.nn.functional.one_hot(hard_targets, num_classes=5).float()
        if soft_targets
        else hard_targets
    )

    class _SoftCrossEntropy(torch.nn.Module):
        def per_sample_loss(self, logits, labels):
            return -(labels * torch.nn.functional.log_softmax(logits, dim=1)).sum(dim=1)

        def forward(self, logits, labels):
            return self.per_sample_loss(logits, labels).mean()

    train_one_epoch(
        model=model,
        dataloader=[(torch.randn(4, 3, 4, 4), targets)],
        criterion=_SoftCrossEntropy() if soft_targets else torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device=torch.device("cpu"),
        amp=False,
        grad_clip_norm=1.0,
        epoch_index=epoch,
        illumination_consistency_loss_weight=0.15,
        illumination_consistency_probability=1.0,
        illumination_consistency_brightness=0.10,
        illumination_consistency_contrast=0.05,
        illumination_consistency_gamma=0.0,
        illumination_consistency_mode="pairwise_margin_retention",
        illumination_consistency_focus_class=1,
        illumination_consistency_negative_classes="0,2,4",
        illumination_consistency_margin_retention=0.8,
        illumination_consistency_start_epoch=start_epoch,
    )
    return model.forward_count


def test_prmr_extra_forward_obeys_start_epoch_and_hard_target_gate():
    assert _one_batch_forward_count(epoch=1, start_epoch=2, soft_targets=False) == 1
    assert _one_batch_forward_count(epoch=2, start_epoch=2, soft_targets=False) == 2
    assert _one_batch_forward_count(epoch=2, start_epoch=2, soft_targets=True) == 1


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

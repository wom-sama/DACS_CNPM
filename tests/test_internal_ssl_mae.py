import torch
from torch import nn

from trkh.models.model import MaskedPatchReconstructionHead
from trkh.training.train import _masked_reconstruction_loss
from trkh.tools.pretrain_internal_barlow import (
    MaskedPatchDecoder,
    TwoViewDataset,
    apply_patch_mask,
    build_mae_patch_mask,
    load_model_init_checkpoint,
    mae_reconstruction_loss,
    patchify_images,
)


class TinyPatchModel(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 8) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)

    def forward_features(self, images: torch.Tensor):
        patches = patchify_images(images, self.patch_size)
        return {"patches": self.proj(patches)}


class TinyTrainTimeMimModel(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 8) -> None:
        super().__init__()
        self.input_patch_size = int(patch_size)
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.masked_reconstruction_head = MaskedPatchReconstructionHead(
            input_dim=embed_dim,
            hidden_dim=16,
            patch_dim=3 * patch_size * patch_size,
        )

    def forward_features(
        self,
        images: torch.Tensor,
        image_valid_mask=None,
        bbox_token_prior=None,
    ):
        del image_valid_mask, bbox_token_prior
        patches = patchify_images(images, self.input_patch_size)
        return {"patches": self.proj(patches)}


class TinyTupleDataset(torch.utils.data.Dataset):
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = float(offset)

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        return torch.full((3, 8, 8), float(index) + self.offset), int(index % 2)


def test_patchify_images_returns_flat_patch_grid() -> None:
    images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)

    patches = patchify_images(images, patch_size=4)

    assert patches.shape == (2, 4, 48)
    assert torch.equal(patches[0, 0], images[0, :, :4, :4].reshape(-1))


def test_two_view_dataset_can_pair_transformed_tuple_datasets() -> None:
    dataset = TwoViewDataset(
        TinyTupleDataset(offset=0.0),
        dataset_b=TinyTupleDataset(offset=10.0),
    )

    view_a, view_b, label = dataset[2]

    assert label == 0
    assert torch.all(view_a == 2.0)
    assert torch.all(view_b == 12.0)


def test_load_model_init_checkpoint_loads_weights_only(tmp_path) -> None:
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(0.75)
        target.weight.zero_()
        target.bias.zero_()
    checkpoint = tmp_path / "init.pt"
    torch.save(
        {
            "checkpoint_kind": "supervised_probe",
            "model_state": source.state_dict(),
            "class_names": ["a", "b"],
            "epoch": 3,
            "best_epoch": 2,
            "optimizer_state": {"ignored": True},
        },
        checkpoint,
    )

    summary = load_model_init_checkpoint(
        target,
        checkpoint,
        expected_class_names=["a", "b"],
    )

    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)
    assert summary["enabled"] is True
    assert summary["epoch"] == 3
    assert summary["missing_key_count"] == 0
    assert summary["unexpected_key_count"] == 0


def test_load_model_init_checkpoint_rejects_class_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "init.pt"
    torch.save(
        {
            "model_state": nn.Linear(2, 2).state_dict(),
            "class_names": ["a", "b"],
        },
        checkpoint,
    )

    try:
        load_model_init_checkpoint(
            nn.Linear(2, 2),
            checkpoint,
            expected_class_names=["a", "c"],
        )
    except ValueError as exc:
        assert "Class names" in str(exc)
    else:
        raise AssertionError("Expected class mismatch to raise ValueError.")


def test_mae_patch_mask_has_expected_ratio_and_shape() -> None:
    torch.manual_seed(5)
    images = torch.randn(3, 3, 16, 16)

    mask = build_mae_patch_mask(
        images,
        patch_size=4,
        mask_ratio=0.5,
        foreground_weight=0.0,
        detail_weight=0.0,
    )

    assert mask.shape == (3, 16)
    assert mask.dtype == torch.bool
    assert torch.equal(mask.sum(dim=1), torch.full((3,), 8))


def test_apply_patch_mask_replaces_selected_patch_pixels() -> None:
    images = torch.ones(1, 3, 8, 8)
    patch_mask = torch.tensor([[True, False, False, False]])

    masked = apply_patch_mask(images, patch_mask, patch_size=4, mask_value=0.0)

    assert torch.all(masked[:, :, :4, :4] == 0.0)
    assert torch.all(masked[:, :, :4, 4:] == 1.0)
    assert torch.all(masked[:, :, 4:, :] == 1.0)


def test_mae_reconstruction_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(11)
    model = TinyPatchModel(patch_size=4, embed_dim=8)
    decoder = MaskedPatchDecoder(input_dim=8, hidden_dim=16, patch_dim=3 * 4 * 4)
    images = torch.randn(2, 3, 16, 16)

    loss, stats = mae_reconstruction_loss(
        model,
        decoder,
        images,
        patch_size=4,
        mask_ratio=0.5,
        foreground_weight=0.0,
        detail_weight=0.0,
    )

    assert torch.isfinite(loss)
    assert stats["mask_fraction"] == 0.5
    assert stats["patch_count"] == 16.0
    loss.backward()
    assert model.proj.weight.grad is not None
    assert torch.isfinite(model.proj.weight.grad).all()


def test_train_time_masked_reconstruction_loss_uses_model_head() -> None:
    torch.manual_seed(17)
    model = TinyTrainTimeMimModel(patch_size=4, embed_dim=8)
    images = torch.randn(2, 3, 16, 16)
    crop_bboxes = torch.tensor(
        [[0.5, 0.5, 0.50, 0.50], [0.25, 0.25, 0.40, 0.40]],
        dtype=torch.float32,
    )

    loss, stats = _masked_reconstruction_loss(
        model=model,
        images=images,
        image_valid_mask=None,
        bbox_metadata=crop_bboxes,
        bbox_token_prior=crop_bboxes,
        crop_bbox_metadata=crop_bboxes,
        patch_size=4,
        mask_ratio=0.50,
        foreground_weight=0.0,
        detail_weight=0.0,
        bbox_weight=0.50,
        bbox_margin_ratio=0.02,
    )

    assert torch.isfinite(loss)
    assert stats["mask_fraction"] == 0.50
    assert stats["patch_count"] == 16.0
    assert stats["bbox_prior_mean"] > 0.0
    loss.backward()
    assert model.proj.weight.grad is not None
    assert model.masked_reconstruction_head.net[-1].weight.grad is not None

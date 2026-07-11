from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from trkh.data.dataset import MangoYOLOCropDataset, TrainBatchCollator
from trkh.models.model import VisionTransformerWithRegisters, classification_logits_from_features
from trkh.training.train import (
    _bbox_foreground_dropout_images,
    _bbox_object_erasure_images,
    _bbox_object_erasure_negative_loss,
)


def _tensor_transform(image, target=None):
    return TF.to_tensor(image), target


def test_yolo_classification_can_return_bbox_metadata_for_collator(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = Image.new("RGB", (100, 80), (20, 180, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 20, 35, 60), fill=(210, 30, 30))
    draw.rectangle((65, 20, 85, 60), fill=(30, 30, 210))
    image.save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text(
        "\n".join(
            [
                "1 0.250000 0.500000 0.200000 0.500000",
                "0 0.750000 0.500000 0.200000 0.500000",
            ]
        ),
        encoding="utf-8",
    )

    dataset = MangoYOLOCropDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=_tensor_transform,
        crop_margin_ratio=0.0,
        num_classes=2,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )

    image_tensor, label, metadata = dataset[0]

    assert image_tensor.ndim == 3
    assert label == 1
    assert torch.allclose(
        metadata["bbox"],
        torch.tensor([0.25, 0.5, 0.2, 0.5], dtype=torch.float32),
    )
    assert torch.allclose(
        metadata["crop_bbox"],
        torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32),
    )
    assert dataset.quality_report()["classification_bbox_metadata"] is True

    collator = TrainBatchCollator(num_classes=2, batch_mix_probability=0.0)
    batch_images, batch_labels, batch_metadata = collator([dataset[0], dataset[1]])

    assert batch_images.shape[0] == 2
    assert batch_labels.tolist() == [1, 0]
    assert batch_metadata["bbox"].shape == (2, 4)
    assert batch_metadata["crop_bbox"].shape == (2, 4)


def test_bbox_foreground_dropout_uses_crop_bbox_region() -> None:
    torch.manual_seed(5)
    images = torch.zeros(2, 3, 32, 32)
    images[:, 0] = 1.0
    crop_bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    dropped, selected_indices, fraction = _bbox_foreground_dropout_images(
        images,
        crop_bboxes,
        probability=1.0,
        min_area_ratio=0.25,
        max_area_ratio=0.25,
        fill="zero",
    )

    assert selected_indices.tolist() == [0]
    assert fraction == 0.5
    assert not torch.equal(dropped[0], images[0])
    assert torch.allclose(dropped[1], images[1], atol=1e-6)
    changed = ((dropped[0] - images[0]).abs() > 1e-5).any(dim=0)
    assert changed[:8].sum() == 0
    assert changed[24:].sum() == 0
    assert changed[:, :8].sum() == 0
    assert changed[:, 24:].sum() == 0


def test_bbox_foreground_dropout_boundary_band_touches_bbox_edge() -> None:
    torch.manual_seed(13)
    images = torch.zeros(1, 3, 32, 32)
    images[:, 0] = 1.0
    crop_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    dropped, selected_indices, fraction = _bbox_foreground_dropout_images(
        images,
        crop_bboxes,
        probability=1.0,
        min_area_ratio=0.25,
        max_area_ratio=0.25,
        mode="boundary_band",
        fill="zero",
    )

    assert selected_indices.tolist() == [0]
    assert fraction == 1.0
    changed = ((dropped[0] - images[0]).abs() > 1e-5).any(dim=0)
    assert changed[:8].sum() == 0
    assert changed[24:].sum() == 0
    assert changed[:, :8].sum() == 0
    assert changed[:, 24:].sum() == 0
    assert (
        changed[8].any()
        or changed[23].any()
        or changed[:, 8].any()
        or changed[:, 23].any()
    )


def test_bbox_object_erasure_covers_expanded_crop_bbox() -> None:
    torch.manual_seed(17)
    images = torch.zeros(1, 3, 32, 32)
    images[:, 0] = 1.0
    crop_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    erased, selected_indices, fraction = _bbox_object_erasure_images(
        images,
        crop_bboxes,
        probability=1.0,
        margin_ratio=0.0,
        fill="zero",
    )

    assert selected_indices.tolist() == [0]
    assert fraction == 1.0
    changed = ((erased[0] - images[0]).abs() > 1e-5).any(dim=0)
    assert changed[:8].sum() == 0
    assert changed[24:].sum() == 0
    assert changed[:, :8].sum() == 0
    assert changed[:, 24:].sum() == 0
    assert changed[8:24, 8:24].all()


def test_bbox_object_erasure_negative_loss_is_finite() -> None:
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 3))
    images = torch.rand(2, 3, 16, 16)
    logits = model(images)
    crop_bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.4, 0.4],
        ],
        dtype=torch.float32,
    )

    loss, fraction, max_probability = _bbox_object_erasure_negative_loss(
        model=model,
        images=images,
        logits=logits,
        crop_bboxes=crop_bboxes,
        bbox_metadata=None,
        bbox_token_prior=None,
        probability=1.0,
        margin_ratio=0.0,
        fill="mean",
        blur_kernel=5,
        temperature=1.0,
        amp=False,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss).item()
    assert fraction == 1.0
    assert 0.0 <= max_probability <= 1.0


def test_bbox_spatial_fusion_is_zero_init_residual() -> None:
    torch.manual_seed(7)
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        in_channels=3,
        use_cnn_stem=False,
        num_classes=2,
        embed_dim=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        bbox_spatial_fusion=True,
        bbox_spatial_fusion_hidden_dim=16,
        bbox_spatial_fusion_dropout=0.0,
        bbox_spatial_fusion_logit_scale=0.25,
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.25, 0.50, 0.20, 0.40],
            [0.75, 0.45, 0.30, 0.35],
        ],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        features_without_bbox = model.forward_features(images)
        logits_without_bbox = classification_logits_from_features(model, features_without_bbox)
        features_with_bbox = model.forward_features(images)
        features_with_bbox["bbox"] = bbox
        logits_with_bbox = classification_logits_from_features(model, features_with_bbox)

    assert torch.allclose(logits_without_bbox, logits_with_bbox, atol=1e-6)
    assert torch.allclose(
        features_with_bbox["bbox_spatial_logits"],
        torch.zeros_like(features_with_bbox["bbox_spatial_logits"]),
        atol=1e-6,
    )


def test_bbox_token_prior_changes_pruned_patch_selection() -> None:
    torch.manual_seed(11)
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        in_channels=3,
        use_cnn_stem=False,
        num_classes=2,
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        token_pruning=True,
        token_prune_layers="1",
        token_keep_rates="0.25",
        token_prune_foreground_weight=1.0,
        token_prune_bbox_weight=12.0,
        token_prune_bbox_margin_ratio=0.0,
    )
    model.eval()
    images = torch.zeros(1, 3, 32, 32)
    left_bbox = torch.tensor([[0.25, 0.50, 0.30, 0.80]], dtype=torch.float32)
    right_bbox = torch.tensor([[0.75, 0.50, 0.30, 0.80]], dtype=torch.float32)

    with torch.inference_mode():
        left_features = model.forward_features(
            images,
            bbox_token_prior=left_bbox,
            return_trace=True,
        )
        right_features = model.forward_features(
            images,
            bbox_token_prior=right_bbox,
            return_trace=True,
        )

    left_kept = left_features["trace"]["pruning"][0]["kept_indices"][0]
    right_kept = right_features["trace"]["pruning"][0]["kept_indices"][0]
    assert "bbox_patch_prior" in left_features["trace"]
    assert not torch.equal(left_kept, right_kept)

    left_columns = left_kept.remainder(4).float().mean()
    right_columns = right_kept.remainder(4).float().mean()
    assert left_columns < right_columns


def test_early_token_mask_records_layer_zero_pruning() -> None:
    torch.manual_seed(17)
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        in_channels=3,
        use_cnn_stem=False,
        num_classes=2,
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        token_pruning=True,
        token_prune_layers="1",
        token_keep_rates="0.25",
        token_prune_foreground_weight=1.0,
        token_prune_bbox_weight=12.0,
        token_prune_bbox_margin_ratio=0.0,
        early_token_mask_keep_rate=0.5,
    )
    model.eval()
    images = torch.zeros(1, 3, 32, 32)
    bbox = torch.tensor([[0.25, 0.50, 0.30, 0.80]], dtype=torch.float32)

    with torch.inference_mode():
        features = model.forward_features(
            images,
            bbox_token_prior=bbox,
            return_trace=True,
        )

    pruning = features["trace"]["pruning"]
    assert int(pruning[0]["layer"].item()) == 0
    assert int(pruning[0]["before_count"].item()) == 16
    assert int(pruning[0]["after_count"].item()) == 8
    assert int(pruning[1]["layer"].item()) == 1
    assert int(pruning[1]["after_count"].item()) == 4
    assert features["patches"].shape[1] == 4

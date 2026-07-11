import torch

from trkh.training.train import (
    _border_attention_suppression_loss_from_features,
    _grid_frame_border_mask,
    _register_attention_alignment_loss_from_features,
)


def test_grid_frame_border_mask_marks_only_outer_ring() -> None:
    mask = _grid_frame_border_mask(
        grid_size=(4, 4),
        frame_width=1,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 16)
    assert int(mask.sum().item()) == 12
    assert mask.view(4, 4)[1, 1].item() == 0.0
    assert mask.view(4, 4)[0, 0].item() == 1.0


def test_border_attention_suppression_penalizes_frame_energy_more_than_center() -> None:
    frame_patches = torch.zeros(1, 16, 4)
    center_patches = torch.zeros(1, 16, 4)
    frame_patches[:, 0, :] = 5.0
    center_patches[:, 5, :] = 5.0

    frame_loss, frame_stats = _border_attention_suppression_loss_from_features(
        features={"patches": frame_patches, "grid_size": (4, 4)},
        targets=torch.tensor([1]),
        bboxes=None,
        frame_width=1,
        bbox_weight=0.0,
        temperature=0.10,
        classes="1",
    )
    center_loss, center_stats = _border_attention_suppression_loss_from_features(
        features={"patches": center_patches, "grid_size": (4, 4)},
        targets=torch.tensor([1]),
        bboxes=None,
        frame_width=1,
        bbox_weight=0.0,
        temperature=0.10,
        classes="1",
    )

    assert frame_loss.item() > center_loss.item()
    assert frame_stats["frame_mass"] > 0.99
    assert center_stats["frame_mass"] < 0.01
    assert frame_stats["valid_fraction"] == 1.0


def test_border_attention_suppression_respects_class_filter() -> None:
    patches = torch.zeros(1, 16, 4)
    patches[:, 0, :] = 5.0

    loss, stats = _border_attention_suppression_loss_from_features(
        features={"patches": patches, "grid_size": (4, 4)},
        targets=torch.tensor([3]),
        bboxes=None,
        frame_width=1,
        bbox_weight=0.0,
        temperature=0.10,
        classes="1",
    )

    assert loss.item() == 0.0
    assert stats["valid_fraction"] == 0.0


def _attention_alignment_features(
    cls_patch_attention: torch.Tensor,
    register_patch_attention: torch.Tensor,
) -> dict:
    attention = torch.zeros(1, 1, 7, 7)
    attention[:, :, 0, 3:] = cls_patch_attention.view(1, 1, 4)
    attention[:, :, 1, 3:] = register_patch_attention.view(1, 1, 4)
    attention[:, :, 2, 3:] = register_patch_attention.view(1, 1, 4)
    return {
        "patches": torch.zeros(1, 4, 8),
        "registers": torch.zeros(1, 2, 8),
        "branch_tokens": torch.zeros(1, 0, 8),
        "grid_size": (2, 2),
        "attentions": {7: attention},
    }


def test_register_attention_alignment_rewards_bbox_aligned_register_agreement() -> None:
    bbox = torch.tensor([[0.75, 0.25, 0.50, 0.50]], dtype=torch.float32)
    targets = torch.tensor([1])
    aligned_features = _attention_alignment_features(
        torch.tensor([0.05, 0.85, 0.05, 0.05]),
        torch.tensor([0.05, 0.85, 0.05, 0.05]),
    )
    divergent_features = _attention_alignment_features(
        torch.tensor([0.85, 0.05, 0.05, 0.05]),
        torch.tensor([0.05, 0.05, 0.05, 0.85]),
    )

    aligned_loss, aligned_stats = _register_attention_alignment_loss_from_features(
        features=aligned_features,
        targets=targets,
        bboxes=bbox,
        classes="1",
        agreement_weight=1.0,
        foreground_weight=0.25,
    )
    divergent_loss, divergent_stats = _register_attention_alignment_loss_from_features(
        features=divergent_features,
        targets=targets,
        bboxes=bbox,
        classes="1",
        agreement_weight=1.0,
        foreground_weight=0.25,
    )

    assert aligned_loss.item() < divergent_loss.item()
    assert aligned_stats["cls_foreground_mass"] > 0.80
    assert divergent_stats["outside_mass"] > aligned_stats["outside_mass"]
    assert divergent_stats["js_divergence"] > aligned_stats["js_divergence"]


def test_register_attention_alignment_respects_class_filter() -> None:
    features = _attention_alignment_features(
        torch.tensor([0.05, 0.85, 0.05, 0.05]),
        torch.tensor([0.05, 0.85, 0.05, 0.05]),
    )

    loss, stats = _register_attention_alignment_loss_from_features(
        features=features,
        targets=torch.tensor([3]),
        bboxes=torch.tensor([[0.75, 0.25, 0.50, 0.50]], dtype=torch.float32),
        classes="1",
    )

    assert loss.item() == 0.0
    assert stats["valid_fraction"] == 0.0

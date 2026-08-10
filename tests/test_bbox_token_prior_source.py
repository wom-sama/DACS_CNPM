import torch

from trkh.training.train import _select_bbox_token_prior_metadata


def test_bbox_token_prior_source_defaults_to_source_bbox() -> None:
    bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]], dtype=torch.float32)
    crop_bbox = torch.tensor([[0.50, 0.50, 0.90, 0.90]], dtype=torch.float32)

    selected = _select_bbox_token_prior_metadata(
        bbox_metadata=bbox,
        crop_bbox_metadata=crop_bbox,
        source="bbox",
    )

    assert selected is bbox


def test_bbox_token_prior_source_can_use_transformed_crop_bbox() -> None:
    bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]], dtype=torch.float32)
    crop_bbox = torch.tensor([[0.50, 0.50, 0.90, 0.90]], dtype=torch.float32)

    selected = _select_bbox_token_prior_metadata(
        bbox_metadata=bbox,
        crop_bbox_metadata=crop_bbox,
        source="crop_bbox",
    )

    assert selected is crop_bbox


def test_bbox_token_prior_source_falls_back_when_crop_bbox_missing() -> None:
    bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]], dtype=torch.float32)

    selected = _select_bbox_token_prior_metadata(
        bbox_metadata=bbox,
        crop_bbox_metadata=None,
        source="crop_bbox",
    )

    assert selected is bbox

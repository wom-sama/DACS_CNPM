from __future__ import annotations

import torch

from trkh.models.deformable_spatial_attention import DeformableSpatialAttention
from trkh.tools.audit_deformable_spatial_attention_pair import (
    _attention_key_map,
    _candidate_spatial_batch,
    _position_bbox_statistics,
    _position_density,
    _standard_logits_with_block2_trace,
)


class _TraceSeparationModel(torch.nn.Module):
    num_registers = 1

    def forward_features(
        self,
        images,
        *,
        image_valid_mask=None,
        bbox_token_prior=None,
        return_attention=False,
        attention_layers=None,
        return_trace=False,
    ):
        del image_valid_mask, bbox_token_prior, attention_layers, return_trace
        logits = torch.tensor([[9.0, -9.0]], device=images.device).expand(
            images.size(0), -1
        )
        if return_attention:
            logits = -logits
            return {
                "logits": logits,
                "grid_size": (1, 1),
                "attentions": {1: torch.ones(images.size(0), 1, 2, 2)},
            }
        return {"logits": logits}

    def forward_heads(self, features):
        return features["logits"]


def test_standard_logits_are_not_taken_from_attention_trace() -> None:
    model = _TraceSeparationModel().eval()
    images = torch.zeros(2, 3, 4, 4)
    logits, trace = _standard_logits_with_block2_trace(
        model,
        images,
        {},
        device=torch.device("cpu"),
    )
    assert logits.argmax(dim=1).tolist() == [0, 0]
    assert trace["logits"].argmax(dim=1).tolist() == [1, 1]
    assert 1 in trace["attentions"]


def test_attention_key_map_aggregates_only_patch_columns() -> None:
    attention = torch.zeros(1, 2, 5, 5)
    attention[:, :, :, 2] = 1.0
    heat = _attention_key_map(attention, prefix_count=1, grid_size=(2, 2))
    assert heat.shape == (1, 2, 2)
    assert torch.isfinite(heat).all()
    assert float(heat[0, 0, 1]) == 1.0
    assert int(torch.count_nonzero(heat).item()) == 1
    torch.testing.assert_close(heat.sum(dim=(1, 2)), torch.ones(1))


def test_position_bbox_statistics_detects_foreground_shift() -> None:
    references = torch.tensor(
        [[[[[0.0, -0.8], [0.0, 0.8]]], [[[0.0, -0.8], [0.0, 0.8]]]]]
    )
    positions = torch.tensor(
        [[[[[0.0, -0.4], [0.0, 0.4]]], [[[0.0, -0.4], [0.0, 0.4]]]]]
    )
    bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    result = _position_bbox_statistics(positions, references, bboxes)
    torch.testing.assert_close(result["bbox_hit_fraction"], torch.tensor([1.0]))
    torch.testing.assert_close(
        result["reference_bbox_hit_fraction"], torch.tensor([0.0])
    )
    torch.testing.assert_close(result["bbox_hit_delta"], torch.tensor([1.0]))
    torch.testing.assert_close(result["outside_distance"], torch.tensor([0.0]))
    torch.testing.assert_close(
        result["reference_outside_distance"], torch.tensor([0.15]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        result["outside_distance_reduction_fraction"],
        torch.tensor([1.0]),
    )


def test_position_density_is_finite_and_normalized_per_row() -> None:
    positions = torch.tensor(
        [
            [
                [[[-0.75, -0.75], [-0.75, 0.75]], [[0.75, -0.75], [0.75, 0.75]]],
                [[[-0.75, -0.75], [-0.75, 0.75]], [[0.75, -0.75], [0.75, 0.75]]],
            ]
        ]
    )
    density = _position_density(positions, (2, 2))
    assert density.shape == (1, 2, 2)
    assert torch.isfinite(density).all()
    torch.testing.assert_close(density, torch.zeros_like(density))


def test_candidate_spatial_batch_reports_each_sample_and_group() -> None:
    torch.manual_seed(17)
    module = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_stride=1,
        offset_range_factor=2.0,
    ).eval()
    inputs = torch.randn(2, 19, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    with torch.inference_mode():
        output, attention = module(
            inputs,
            return_attention=True,
            grid_size=(4, 4),
            prefix_count=3,
            patch_indices=indices,
        )
    assert output.shape == inputs.shape
    assert attention.shape == (2, 4, 19, 19)
    result = _candidate_spatial_batch(
        module,
        crop_bboxes=torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [0.4, 0.6, 0.3, 0.4]]
        ),
    )
    assert result["offset_rms"].shape == (2,)
    assert result["offset_rms_per_group"].shape == (2, 2)
    assert result["inter_group_position_rms"].shape == (2,)
    assert result["valid_interpolation_mass"].shape == (2,)
    assert all(torch.isfinite(value).all() for value in result.values())

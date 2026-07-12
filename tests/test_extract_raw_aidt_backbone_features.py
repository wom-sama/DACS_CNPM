from __future__ import annotations

import torch
from torch import nn

from trkh.tools.extract_raw_aidt_backbone_features import (
    BATCH_SIZE,
    IMAGE_SIZE,
    RESNET_NAME,
    VIT_NAME,
    _normalization_tensors,
    _selected_pretrained_config,
    parse_args,
    state_dict_sha256,
)


def test_protocol_defaults_match_local_aidt_backbones() -> None:
    args = parse_args(["--data", "data.yaml", "--output-dir", "out"])
    assert args.resnet == RESNET_NAME == "resnet50.a1_in1k"
    assert args.vit == VIT_NAME == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    assert args.image_size == IMAGE_SIZE == 224
    assert args.batch_size == BATCH_SIZE == 32
    assert args.max_samples_per_class == 0
    assert not args.amp


def test_state_dict_hash_is_deterministic_and_weight_sensitive() -> None:
    model = nn.Linear(3, 2)
    first = state_dict_sha256(model)
    assert state_dict_sha256(model) == first
    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    assert state_dict_sha256(model) != first


def test_selected_pretrained_config_is_json_friendly_and_bounded() -> None:
    selected = _selected_pretrained_config(
        {
            "architecture": "example",
            "input_size": (3, 224, 224),
            "mean": (0.1, 0.2, 0.3),
            "unrelated": object(),
        }
    )
    assert selected == {
        "architecture": "example",
        "input_size": [3, 224, 224],
        "mean": [0.1, 0.2, 0.3],
    }


def test_normalization_tensors_broadcast_over_image_batches() -> None:
    mean, std = _normalization_tensors(
        {"mean": (0.1, 0.2, 0.3), "std": (0.4, 0.5, 0.6)},
        device=torch.device("cpu"),
    )
    images = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    normalized = (images - mean) / std
    assert mean.shape == std.shape == (1, 3, 1, 1)
    assert torch.allclose(normalized[0, :, 0, 0], torch.tensor([2.25, 1.6, 7.0 / 6.0]))

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.model import (
    HybridConvStem,
    InstanceBatchNorm2d,
    create_model,
)
from trkh.training.train import parse_args


def _config(stem_normalization: str = "batch") -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=32,
        patch_size=16,
        stem_channels=8,
        stem_architecture="conv_pool",
        stem_normalization=stem_normalization,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=False,
        fine_grained_pooling=False,
        token_pruning=False,
    )


def test_ibn_a_matches_independent_direct_equation() -> None:
    torch.manual_seed(7)
    layer = InstanceBatchNorm2d(8, ratio=0.5).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(0.7, 1.3, 8))
        layer.bias.copy_(torch.linspace(-0.2, 0.2, 8))
        layer.running_mean.copy_(torch.linspace(-0.4, 0.4, 8))
        layer.running_var.copy_(torch.linspace(0.6, 1.4, 8))
    inputs = torch.randn(3, 8, 9, 7)

    actual = layer(inputs)
    expected_instance = F.instance_norm(
        inputs[:, :4].contiguous(),
        weight=layer.weight[:4],
        bias=layer.bias[:4],
        use_input_stats=True,
        momentum=0.0,
        eps=layer.eps,
    )
    expected_batch = F.batch_norm(
        inputs[:, 4:],
        layer.running_mean[4:],
        layer.running_var[4:],
        layer.weight[4:],
        layer.bias[4:],
        training=False,
        momentum=layer.momentum,
        eps=layer.eps,
    )
    expected = torch.cat((expected_instance, expected_batch), dim=1)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    standardized = (
        actual[:, :4] - layer.bias[:4].view(1, -1, 1, 1)
    ) / layer.weight[:4].view(1, -1, 1, 1)
    assert standardized.mean(dim=(-2, -1)).abs().max().item() < 1e-5
    assert (
        standardized.var(dim=(-2, -1), unbiased=False)
        - torch.ones(1, 4)
    ).abs().max().item() < 2e-4


def test_ibn_a_preserves_batchnorm_state_schema_and_strict_loading() -> None:
    control = nn.BatchNorm2d(8)
    candidate = InstanceBatchNorm2d(8)
    assert list(control.state_dict()) == list(candidate.state_dict())
    candidate.load_state_dict(control.state_dict(), strict=True)
    control.load_state_dict(candidate.state_dict(), strict=True)


def test_ibn_a_is_affine_style_invariant_only_in_first_channel_half() -> None:
    torch.manual_seed(11)
    control = nn.BatchNorm2d(8).eval()
    candidate = InstanceBatchNorm2d(8).eval()
    candidate.load_state_dict(control.state_dict(), strict=True)
    inputs = torch.randn(3, 8, 11, 9)
    scale = torch.tensor([0.55, 1.25, 1.8]).view(3, 1, 1, 1)
    shift = torch.tensor([-0.7, 0.35, 1.1]).view(3, 1, 1, 1)
    transformed = inputs * scale + shift

    base_control = control(inputs)
    shifted_control = control(transformed)
    base_candidate = candidate(inputs)
    shifted_candidate = candidate(transformed)
    control_error = (shifted_control[:, :4] - base_control[:, :4]).abs().mean()
    candidate_error = (shifted_candidate[:, :4] - base_candidate[:, :4]).abs().mean()

    assert candidate_error.item() <= 0.05 * control_error.item()
    assert torch.equal(candidate(inputs)[:, 4:], control(inputs)[:, 4:])


def test_model_candidate_changes_only_first_stem_normalization() -> None:
    torch.manual_seed(19)
    control = create_model(num_classes=5, model_config=_config("batch"))
    torch.manual_seed(19)
    candidate = create_model(num_classes=5, model_config=_config("ibn_a_first"))

    assert isinstance(control.stem, HybridConvStem)
    assert isinstance(candidate.stem, HybridConvStem)
    assert type(control.stem.blocks[0].block.norm) is nn.BatchNorm2d
    assert isinstance(candidate.stem.blocks[0].block.norm, InstanceBatchNorm2d)
    assert type(candidate.stem.blocks[1].block.norm) is nn.BatchNorm2d
    assert type(candidate.stem.blocks[2].block.norm) is nn.BatchNorm2d
    assert candidate.stem.blocks[0].block.norm.instance_channels == 4
    assert list(control.state_dict()) == list(candidate.state_dict())
    assert all(
        torch.equal(value, candidate.state_dict()[name])
        for name, value in control.state_dict().items()
    )
    candidate.load_state_dict(control.state_dict(), strict=True)
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )


def test_explicit_batch_default_is_bit_identical() -> None:
    explicit = vars(_config("batch")).copy()
    omitted = dict(explicit)
    omitted.pop("stem_normalization")
    torch.manual_seed(23)
    explicit_model = create_model(num_classes=5, model_config=explicit).eval()
    torch.manual_seed(23)
    omitted_model = create_model(num_classes=5, model_config=omitted).eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        explicit_logits = explicit_model(inputs)
        omitted_logits = omitted_model(inputs)
    assert all(
        torch.equal(value, omitted_model.state_dict()[name])
        for name, value in explicit_model.state_dict().items()
    )
    assert torch.equal(explicit_logits, omitted_logits)


def test_ibn_a_rejects_nonlegacy_stem_and_cli_roundtrips() -> None:
    with pytest.raises(ValueError, match="only supported"):
        create_model(
            num_classes=5,
            model_config=replace(
                _config("ibn_a_first"),
                stem_architecture="dbb_conv_pool",
            ),
        )
    assert parse_args([]).stem_normalization == "batch"
    assert (
        parse_args(["--stem-normalization", "ibn_a_first"]).stem_normalization
        == "ibn_a_first"
    )


def test_v8_launcher_forwards_and_records_stem_normalization() -> None:
    launcher = Path("scripts/run_trkh_5class_attention_views_v8.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert '[ValidateSet("batch", "ibn_a_first")]' in launcher
    assert '"--stem-normalization", "$StemNormalization"' in launcher
    assert launcher.count("stem_normalization = $StemNormalization") == 2

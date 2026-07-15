from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.model import HybridConvStem, build_model_from_checkpoint, create_model
from trkh.models.octave_conv_stem import OctaveConv2d, OctaveConvStem
from trkh.training.train import parse_args


def _model_config(stem_architecture: str = "octave_conv") -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=256,
        patch_size=16,
        stem_architecture=stem_architecture,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=False,
        token_pruning=False,
    )


def test_octave_operator_matches_all_four_primary_equation_paths() -> None:
    torch.manual_seed(42)
    operator = OctaveConv2d(
        8,
        16,
        alpha_in=0.25,
        alpha_out=0.25,
    )
    high = torch.randn(2, 6, 16, 16)
    low = torch.randn(2, 2, 8, 8)
    high_output, low_output, paths = operator.forward_with_paths(high, low)

    assert tuple(paths) == ("hh", "hl", "ll", "lh")
    assert low_output is not None
    assert torch.equal(paths["hh"], operator.conv_hh(high))
    assert torch.equal(
        paths["hl"],
        operator.conv_hl(F.avg_pool2d(high, kernel_size=2, stride=2)),
    )
    assert torch.equal(paths["ll"], operator.conv_ll(low))
    assert torch.equal(
        paths["lh"],
        F.interpolate(operator.conv_lh(low), size=(16, 16), mode="nearest"),
    )
    assert torch.equal(high_output, paths["hh"] + paths["lh"])
    assert torch.equal(low_output, paths["ll"] + paths["hl"])


def test_octave_stem_preserves_locked_channels_stride_and_patch_grid() -> None:
    stem = OctaveConvStem(embed_dim=256)
    stem.eval()
    with torch.no_grad():
        output, trace = stem.forward_with_trace(torch.randn(2, 3, 256, 256))

    assert stem.alpha == 0.125
    assert stem.downsample_factor == 8
    assert stem.out_channels == 256
    assert stem.kernel_parameter_count == 3 * 32 * 9 + 32 * 64 * 9 + 64 * 256 * 9
    assert stem.blocks[0].octave.available_paths == ("hh", "hl")
    assert stem.blocks[1].octave.available_paths == ("hh", "hl", "ll", "lh")
    assert stem.blocks[2].octave.available_paths == ("hh", "lh")
    assert trace["block1_high_output"].shape == (2, 28, 128, 128)
    assert trace["block1_low_output"].shape == (2, 4, 64, 64)
    assert trace["block2_high_output"].shape == (2, 56, 64, 64)
    assert trace["block2_low_output"].shape == (2, 8, 32, 32)
    assert output.shape == (2, 256, 32, 32)
    assert torch.equal(output, trace["stem_output"])
    assert all(torch.isfinite(value).all() for value in trace.values())


def test_octave_and_control_are_parameter_rng_and_nonstem_paired() -> None:
    control_config = _model_config("conv_pool")
    candidate_config = replace(control_config, stem_architecture="octave_conv")

    torch.manual_seed(42)
    control = create_model(num_classes=5, model_config=control_config)
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_rng = torch.get_rng_state().clone()

    assert isinstance(control.stem, HybridConvStem)
    assert isinstance(candidate.stem, OctaveConvStem)
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    assert torch.equal(control_rng, candidate_rng)
    control_nonstem = {
        name: value for name, value in control.state_dict().items() if not name.startswith("stem.")
    }
    candidate_nonstem = {
        name: value
        for name, value in candidate.state_dict().items()
        if not name.startswith("stem.")
    }
    assert control_nonstem.keys() == candidate_nonstem.keys()
    assert all(
        torch.equal(value, candidate_nonstem[name])
        for name, value in control_nonstem.items()
    )
    control_kernels = [block.block.conv.weight for block in control.stem.blocks]
    assert all(
        torch.equal(control_kernel, candidate_kernel)
        for control_kernel, candidate_kernel in zip(
            control_kernels,
            candidate.stem.reconstructed_vanilla_kernels(),
        )
    )


def test_octave_trkh_forward_and_every_path_gradient_are_live() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_model_config())
    logits = model(torch.randn(2, 3, 256, 256))
    logits.square().mean().backward()

    assert isinstance(model.stem, OctaveConvStem)
    assert model.patch_embed.patch_size == 2
    assert model.patch_embed.num_patches == 256
    assert logits.shape == (2, 5)
    path_weights = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("stem.blocks.") and ".octave.conv_" in name and name.endswith("weight")
    }
    assert len(path_weights) == 8
    for name, parameter in path_weights.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert int(torch.count_nonzero(parameter.grad)) > 0, name


def test_octave_checkpoint_cli_and_invalid_paths_fail_closed() -> None:
    config = _model_config()
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )
    parsed = parse_args(["--stem-architecture", "octave_conv"])

    assert isinstance(restored.stem, OctaveConvStem)
    assert restored.stem_architecture == "octave_conv"
    assert set(restored.state_dict()) == set(source.state_dict())
    assert parsed.stem_architecture == "octave_conv"
    with pytest.raises(ValueError, match="unavailable OctConv paths"):
        source.stem.blocks[0].octave.forward_with_paths(
            torch.randn(1, 3, 16, 16),
            disabled_paths=("lh",),
        )
    with pytest.raises(ValueError, match="stem_architecture"):
        create_model(num_classes=5, model_config=_model_config("octave_unknown"))

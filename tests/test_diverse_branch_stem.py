from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.diverse_branch_stem import (
    DiverseBranchConvBN,
    DiverseBranchConvStem,
    add_equivalent_branches,
    average_pool_kernel,
    compose_1x1_kxk,
    convert_dbb_modules_to_deploy,
    fuse_conv_batch_norm,
    pad_kernel_to,
)
from trkh.models.model import HybridConvStem, build_model_from_checkpoint, create_model
from trkh.training.train import parse_args


def _model_config(stem_architecture: str = "dbb_conv_pool") -> ModelConfig:
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


def test_dbb_transform_equations_replay_each_branch() -> None:
    torch.manual_seed(42)
    block = DiverseBranchConvBN(3, 5).eval()
    inputs = torch.randn(2, 3, 9, 9)
    with torch.no_grad():
        observed, branches = block.forward_with_branches(inputs)
        paths = block.equivalent_path_kernels()
        replayed = {
            name: F.conv2d(inputs, kernel, bias, padding=1)
            for name, (kernel, bias) in paths.items()
        }
        kernel, bias = block.get_equivalent_kernel_bias()
        fused = F.conv2d(inputs, kernel, bias, padding=1)

    assert tuple(branches) == block.branch_names
    for name in block.branch_names:
        assert torch.allclose(branches[name], replayed[name], atol=1e-6, rtol=1e-6)
    assert torch.allclose(observed, sum(replayed.values()), atol=1e-6, rtol=1e-6)
    assert torch.allclose(observed, fused, atol=1e-6, rtol=1e-6)

    first_kernel, first_bias = fuse_conv_batch_norm(
        block.dbb_1x1_kxk.idconv1.actual_kernel(),
        block.dbb_1x1_kxk.norm1,
    )
    second_kernel, second_bias = fuse_conv_batch_norm(
        block.dbb_1x1_kxk.conv2.weight,
        block.dbb_1x1_kxk.norm2,
    )
    sequential_kernel, sequential_bias = compose_1x1_kxk(
        first_kernel,
        first_bias,
        second_kernel,
        second_bias,
    )
    assert torch.equal(sequential_kernel, paths["sequential"][0])
    assert torch.equal(sequential_bias, paths["sequential"][1])

    pointwise_kernel, pointwise_bias = fuse_conv_batch_norm(
        block.dbb_1x1.conv.weight,
        block.dbb_1x1.norm,
    )
    assert torch.equal(pad_kernel_to(pointwise_kernel, 3), paths["pointwise"][0])
    assert torch.equal(pointwise_bias, paths["pointwise"][1])
    avg = average_pool_kernel(
        5,
        3,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    assert avg.shape == (5, 5, 3, 3)
    assert torch.allclose(avg.sum(dim=(1, 2, 3)), torch.ones(5))

    added_kernel, added_bias = add_equivalent_branches(
        tuple(paths[name][0] for name in block.branch_names),
        tuple(paths[name][1] for name in block.branch_names),
    )
    assert torch.equal(added_kernel, kernel)
    assert torch.equal(added_bias, bias)


def test_dbb_switch_to_deploy_is_exact_and_removes_training_paths() -> None:
    torch.manual_seed(42)
    block = DiverseBranchConvBN(4, 7).eval()
    inputs = torch.randn(3, 4, 17, 17)
    with torch.no_grad():
        expected = block(inputs)
        block.switch_to_deploy()
        observed = block(inputs)

    assert torch.allclose(expected, observed, atol=1e-5, rtol=1e-5)
    assert hasattr(block, "reparam")
    assert block.reparam.bias is not None
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in block.modules())
    assert not any(isinstance(module, torch.nn.AvgPool2d) for module in block.modules())
    assert not hasattr(block, "dbb_origin")
    assert not hasattr(block, "dbb_1x1")
    assert not hasattr(block, "dbb_1x1_kxk")
    assert not hasattr(block, "dbb_avg")


def test_dbb_and_control_are_rng_origin_and_nonstem_paired() -> None:
    control_config = _model_config("conv_pool")
    candidate_config = replace(control_config, stem_architecture="dbb_conv_pool")

    torch.manual_seed(42)
    control = create_model(num_classes=5, model_config=control_config)
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_rng = torch.get_rng_state().clone()

    assert isinstance(control.stem, HybridConvStem)
    assert isinstance(candidate.stem, DiverseBranchConvStem)
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
    for control_block, candidate_block in zip(control.stem.blocks, candidate.stem.blocks):
        assert torch.equal(
            control_block.block.conv.weight,
            candidate_block.dbb.dbb_origin.conv.weight,
        )
        assert control_block.block.norm.state_dict().keys() == (
            candidate_block.dbb.dbb_origin.norm.state_dict().keys()
        )
        assert all(
            torch.equal(value, candidate_block.dbb.dbb_origin.norm.state_dict()[name])
            for name, value in control_block.block.norm.state_dict().items()
        )
        for norm in candidate_block.dbb.terminal_norms().values():
            assert torch.equal(norm.weight, torch.ones_like(norm.weight))


def test_dbb_trkh_forward_and_all_twelve_path_gradients_are_live() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_model_config())
    logits = model(torch.randn(2, 3, 256, 256))
    logits.square().mean().backward()

    assert isinstance(model.stem, DiverseBranchConvStem)
    assert model.patch_embed.patch_size == 2
    assert model.patch_embed.num_patches == 256
    assert logits.shape == (2, 5)
    terminal_norms = {
        f"block{block_index}.{path_name}": norm
        for block_index, block in enumerate(model.stem.blocks, start=1)
        for path_name, norm in block.dbb.terminal_norms().items()
    }
    assert len(terminal_norms) == 12
    for name, norm in terminal_norms.items():
        assert norm.weight.grad is not None, name
        assert torch.isfinite(norm.weight.grad).all(), name
        assert int(torch.count_nonzero(norm.weight.grad)) > 0, name
    identity_weights = [
        block.dbb.dbb_1x1_kxk.idconv1.weight for block in model.stem.blocks
    ]
    for parameter in identity_weights:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert int(torch.count_nonzero(parameter.grad)) > 0


def test_dbb_model_deploy_copy_preserves_source_and_logits() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_model_config()).eval()
    inputs = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        expected = model(inputs)
        deployed = convert_dbb_modules_to_deploy(model, inplace=False).eval()
        observed = deployed(inputs)

    assert isinstance(model.stem, DiverseBranchConvStem)
    assert not model.stem.is_deployed
    assert isinstance(deployed.stem, DiverseBranchConvStem)
    assert deployed.stem.is_deployed
    assert torch.allclose(expected, observed, atol=1e-5, rtol=1e-5)
    assert torch.equal(expected.argmax(dim=1), observed.argmax(dim=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dbb_cuda_conversion_disables_tf32_for_kernel_composition() -> None:
    torch.manual_seed(42)
    block = DiverseBranchConvBN(32, 64).cuda().eval()
    inputs = torch.randn(2, 32, 64, 64, device="cuda")
    previous = bool(torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32 = True
        deployed = convert_dbb_modules_to_deploy(block, inplace=False).eval()
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            expected = block(inputs)
            observed = deployed(inputs)
    finally:
        torch.backends.cudnn.allow_tf32 = previous

    assert torch.allclose(expected, observed, atol=1e-5, rtol=1e-5)


def test_dbb_checkpoint_cli_and_invalid_pooling_fail_closed() -> None:
    config = _model_config()
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )
    parsed = parse_args(["--stem-architecture", "dbb_conv_pool"])

    assert isinstance(restored.stem, DiverseBranchConvStem)
    assert restored.stem_architecture == "dbb_conv_pool"
    assert set(restored.state_dict()) == set(source.state_dict())
    assert parsed.stem_architecture == "dbb_conv_pool"
    with pytest.raises(ValueError, match="stem_pooling_mode=max"):
        create_model(
            num_classes=5,
            model_config=replace(config, stem_pooling_mode="soft"),
        )

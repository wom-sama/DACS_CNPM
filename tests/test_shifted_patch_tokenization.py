import torch

from trkh.models.model import (
    ShiftedPatchTokenResidual,
    VisionTransformerWithRegisters,
)
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_model(enabled: bool) -> VisionTransformerWithRegisters:
    return VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        in_channels=3,
        use_cnn_stem=True,
        stem_channels=4,
        shifted_patch_tokenization=enabled,
        shifted_patch_shift=1,
        shifted_patch_residual_scale=0.10,
        cnn_feature_fusion=False,
        fine_grained_pooling=False,
        multi_branch_fusion=False,
        detail_patch_enhancement=False,
        token_pruning=False,
        num_classes=5,
        embed_dim=16,
        depth=1,
        num_heads=2,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )


def test_shifted_views_zero_pad_instead_of_wrapping() -> None:
    module = ShiftedPatchTokenResidual(
        in_channels=1,
        embed_dim=1,
        patch_size=1,
        shift=1,
    )
    image = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)

    top_left, _, _, bottom_right = module.shifted_views(image)

    torch.testing.assert_close(
        top_left,
        torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 4.0, 5.0]]]]),
    )
    torch.testing.assert_close(
        bottom_right,
        torch.tensor([[[[5.0, 6.0, 0.0], [8.0, 9.0, 0.0], [0.0, 0.0, 0.0]]]]),
    )


def test_shifted_patch_residual_is_zero_initialized_and_trainable() -> None:
    module = ShiftedPatchTokenResidual(
        in_channels=2,
        embed_dim=3,
        patch_size=2,
        shift=1,
        residual_scale=0.25,
    )
    module.zero_init_residual()
    image = torch.randn(2, 2, 8, 8, requires_grad=True)

    zero_output = module(image)
    assert zero_output.shape == (2, 16, 3)
    torch.testing.assert_close(zero_output, torch.zeros_like(zero_output))

    with torch.no_grad():
        module.proj.weight.fill_(0.01)
    output = module(image)
    output.square().mean().backward()
    assert module.proj.weight.grad is not None
    assert float(module.proj.weight.grad.abs().sum()) > 0.0


def test_resume_extension_preserves_base_logits_at_zero_init() -> None:
    torch.manual_seed(7)
    source = _tiny_model(enabled=False).eval()
    target = _tiny_model(enabled=True).eval()

    summary = _load_model_state_allowing_extensions(
        target,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert summary["allowed_missing_keys"]
    assert all(
        key.startswith("shifted_patch_token_residual.")
        for key in summary["allowed_missing_keys"]
    )
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        source_logits = source(image)
        target_logits = target(image)
        trace = target.forward_features(image, return_trace=True)["trace"]
    torch.testing.assert_close(target_logits, source_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        trace["shifted_patch_residual_norm"],
        torch.zeros_like(trace["shifted_patch_residual_norm"]),
    )

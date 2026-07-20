from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.model import (
    HybridConvStem,
    ValidityPartialConv2d,
    VisionTransformerWithRegisters,
    classification_logits_from_features,
)
from trkh.training.train import parse_args as parse_train_args


def _copy_state(source: nn.Module, target: nn.Module) -> None:
    result = target.load_state_dict(source.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_validity_partial_conv_matches_locked_equation() -> None:
    torch.manual_seed(3)
    layer = ValidityPartialConv2d(2, 3, kernel_size=3, padding=1, bias=False).eval()
    inputs = torch.randn(2, 2, 9, 11)
    mask = torch.ones(2, 1, 9, 11, dtype=torch.bool)
    mask[0, :, 2:7, :3] = False
    mask[1, :, :4, 7:] = False

    actual, updated, valid_count = layer.forward_with_mask(inputs, mask)
    expected_count = F.conv2d(
        F.pad(mask.float(), (1, 1, 1, 1), value=1.0),
        torch.ones(1, 1, 3, 3),
    )
    expected_updated = expected_count > 0.0
    expected_ratio = torch.where(
        expected_updated,
        (9.0 + 1e-6) / (expected_count + 1e-6),
        torch.zeros_like(expected_count),
    )
    expected = F.conv2d(inputs * mask, layer.weight, padding=1) * expected_ratio

    torch.testing.assert_close(valid_count, expected_count, rtol=0.0, atol=0.0)
    assert torch.equal(updated, expected_updated)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_validity_partial_stem_is_state_compatible_and_all_valid_equivalent() -> None:
    torch.manual_seed(7)
    control = HybridConvStem(3, 4, 16, convolution="standard").eval()
    candidate = HybridConvStem(3, 4, 16, convolution="validity_partial").eval()
    _copy_state(control, candidate)

    assert list(control.state_dict()) == list(candidate.state_dict())
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    inputs = torch.randn(2, 3, 32, 32)
    all_valid = torch.ones(2, 32, 32, dtype=torch.bool)
    torch.testing.assert_close(
        candidate(inputs, all_valid),
        control(inputs),
        rtol=0.0,
        atol=0.0,
    )


def test_validity_partial_stem_is_invariant_to_masked_fill() -> None:
    torch.manual_seed(11)
    control = HybridConvStem(3, 4, 16, convolution="standard").eval()
    candidate = HybridConvStem(3, 4, 16, convolution="validity_partial").eval()
    _copy_state(control, candidate)

    mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    mask[:, :, :, :9] = False
    first = torch.randn(2, 3, 32, 32)
    second = first.clone()
    second = torch.where(mask, second, torch.full_like(second, 7.0))
    first = torch.where(mask, first, torch.full_like(first, -5.0))

    torch.testing.assert_close(
        candidate(first, mask),
        candidate(second, mask),
        rtol=0.0,
        atol=0.0,
    )
    assert float((control(first) - control(second)).abs().max()) > 1e-4


def test_validity_partial_stem_updates_mask_and_preserves_geometry() -> None:
    torch.manual_seed(13)
    stem = HybridConvStem(3, 4, 16, convolution="validity_partial").eval()
    inputs = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
    mask[:, :, 8:56, 16:48] = True

    output, trace = stem(inputs, mask, return_mask_trace=True)
    assert output.shape == (2, 16, 8, 8)
    assert trace["mode"] == "validity_partial"
    assert len(trace["blocks"]) == 3
    expected_shapes = [(32, 32), (16, 16), (8, 8)]
    valid_fractions = [float(mask.float().mean())]
    for block_trace, expected_shape in zip(trace["blocks"], expected_shapes):
        assert tuple(block_trace["pooled_mask"].shape[-2:]) == expected_shape
        assert float(block_trace["valid_count"].min()) >= 0.0
        assert float(block_trace["valid_count"].max()) <= 9.0
        valid_fractions.append(float(block_trace["pooled_mask"].float().mean()))
    assert valid_fractions == sorted(valid_fractions)
    assert torch.equal(trace["final_mask"], trace["blocks"][-1]["pooled_mask"])


def test_validity_partial_stem_gradients_are_finite_and_nonzero() -> None:
    torch.manual_seed(17)
    stem = HybridConvStem(3, 4, 16, convolution="validity_partial").train()
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    mask[:, :, :, :8] = False
    stem(inputs, mask).square().mean().backward()

    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for block in stem.blocks:
        gradient = block.block.conv.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_validity_partial_complete_model_is_all_valid_equivalent() -> None:
    common = dict(
        image_size=32,
        patch_size=16,
        stem_channels=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        num_classes=5,
    )
    torch.manual_seed(19)
    control = VisionTransformerWithRegisters(**common, stem_convolution="standard").eval()
    candidate = VisionTransformerWithRegisters(
        **common,
        stem_convolution="validity_partial",
    ).eval()
    _copy_state(control, candidate)
    inputs = torch.randn(2, 3, 32, 32)
    mask = torch.ones(2, 32, 32, dtype=torch.bool)

    control_features = control.forward_features(inputs, image_valid_mask=mask)
    candidate_features = candidate.forward_features(inputs, image_valid_mask=mask)
    control_logits = classification_logits_from_features(control, control_features)
    candidate_logits = classification_logits_from_features(candidate, candidate_features)
    torch.testing.assert_close(candidate_logits, control_logits, rtol=2e-6, atol=2e-6)


def test_validity_partial_complete_model_masks_raw_image_branches() -> None:
    common = dict(
        image_size=32,
        patch_size=16,
        stem_channels=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        detail_patch_enhancement=True,
        multi_branch_fusion=True,
        num_classes=5,
    )
    torch.manual_seed(23)
    candidate = VisionTransformerWithRegisters(
        **common,
        stem_convolution="validity_partial",
    ).eval()
    mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    mask[:, :, :, :8] = False
    base = torch.randn(2, 3, 32, 32)
    first = torch.where(mask, base, torch.full_like(base, -4.0))
    second = torch.where(mask, base, torch.full_like(base, 6.0))

    with torch.inference_mode():
        first_features = candidate.forward_features(first, image_valid_mask=mask)
        second_features = candidate.forward_features(second, image_valid_mask=mask)
        first_logits = classification_logits_from_features(candidate, first_features)
        second_logits = classification_logits_from_features(candidate, second_features)
    torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=0.0)


def test_validity_partial_configuration_guards() -> None:
    assert ModelConfig().stem_convolution == "standard"
    try:
        HybridConvStem(3, 4, 16, pooling_mode="soft", convolution="validity_partial")
    except ValueError as error:
        assert "requires max pooling" in str(error)
    else:
        raise AssertionError("validity_partial unexpectedly accepted soft pooling")

    try:
        VisionTransformerWithRegisters(
            image_size=32,
            patch_size=16,
            stem_architecture="coatnet_mbconv",
            stem_convolution="validity_partial",
            embed_dim=16,
            depth=1,
            num_heads=4,
        )
    except ValueError as error:
        assert "only supported" in str(error)
    else:
        raise AssertionError("validity_partial unexpectedly accepted a non-conv_pool stem")


def test_validity_partial_cli_and_v8_launcher_roundtrip() -> None:
    assert parse_train_args([]).stem_convolution == "standard"
    assert (
        parse_train_args(["--stem-convolution", "validity_partial"]).stem_convolution
        == "validity_partial"
    )
    launcher = Path("scripts/run_trkh_5class_attention_views_v8.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert '[ValidateSet("standard", "validity_partial")]' in launcher
    assert '"--stem-convolution", "$StemConvolution"' in launcher
    assert launcher.count("stem_convolution = $StemConvolution") == 2


def test_pairwise_margin_route_export_preserves_dynamic_batch() -> None:
    import onnxruntime as ort

    class RouteWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = VisionTransformerWithRegisters(
                image_size=32,
                patch_size=16,
                stem_channels=4,
                embed_dim=16,
                depth=1,
                num_heads=4,
                num_registers=1,
                pairwise_margin_head=True,
                pairwise_margin_routing=True,
                num_classes=5,
            ).eval()

        def forward(self, logits: torch.Tensor) -> torch.Tensor:
            return self.model.pairwise_margin_route_weights(logits)

    wrapper = RouteWrapper().eval()
    batch1 = torch.tensor([[3.0, 2.8, 0.1, -0.2, -0.4]])
    batch2 = torch.cat((batch1, batch1.flip(1)), dim=0)
    with tempfile.TemporaryDirectory(prefix="trkh_pairroute_onnx_") as directory:
        path = Path(directory) / "route.onnx"
        torch.onnx.export(
            wrapper,
            batch1,
            path,
            input_names=["logits"],
            output_names=["route_weights"],
            dynamic_axes={"logits": {0: "batch"}, "route_weights": {0: "batch"}},
            opset_version=17,
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        observed = session.run(None, {"logits": batch2.numpy()})[0]
    expected = wrapper(batch2).detach().numpy()
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-6)
    assert observed.shape == (2, len(wrapper.model.pairwise_margin_pairs))

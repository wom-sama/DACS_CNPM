import torch

from trkh.models.model import PatchMemoryAdapter, VisionTransformerWithRegisters
from trkh.training.train import _load_model_state_allowing_extensions


def test_patch_memory_adapter_zero_init_is_identity_for_matching_grid():
    adapter = PatchMemoryAdapter(embed_dim=8, dropout=0.0, zero_init=True)
    tokens = torch.randn(2, 6, 8, requires_grad=True)

    adapted = adapter(tokens, grid_size=(2, 3))

    assert torch.allclose(adapted, tokens)
    adapted.sum().backward()
    assert tokens.grad is not None


def test_patch_memory_adapter_forward_trace_on_small_classifier():
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        stem_channels=8,
        fine_grained_pooling=True,
        patch_memory_adapter=True,
        patch_memory_adapter_dropout=0.0,
        head_pooling="cls_register_mean",
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        features = model.forward_features(images, return_trace=True)
        head_input = model.head_input_from_features(features)

    assert head_input.shape == (2, 32)
    assert features["trace"]["patch_memory_adapter_enabled"] is True


def test_patch_memory_adapter_resume_extension_is_allowlisted():
    base_model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        stem_channels=8,
        fine_grained_pooling=True,
        patch_memory_adapter=False,
        head_pooling="cls_register_mean",
    )
    extended_model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        stem_channels=8,
        fine_grained_pooling=True,
        patch_memory_adapter=True,
        patch_memory_adapter_dropout=0.0,
        head_pooling="cls_register_mean",
    )

    summary = _load_model_state_allowing_extensions(
        extended_model,
        base_model.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert summary["allowed_missing_keys"]
    assert all(
        str(key).startswith("patch_memory_adapter.")
        for key in summary["allowed_missing_keys"]
    )

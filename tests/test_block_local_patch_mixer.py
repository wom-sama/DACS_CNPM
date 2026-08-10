import torch

from trkh.core.config import ModelConfig
from trkh.models.model import BlockLocalPatchMixer, create_model
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=3,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_block_local_patch_mixer_zero_init_is_identity_with_prefix_tokens() -> None:
    mixer = BlockLocalPatchMixer(embed_dim=8, dropout=0.0, residual_scale=0.10, zero_init=True)
    tokens = torch.randn(2, 7, 8, requires_grad=True)

    mixed = mixer(tokens, grid_size=(2, 2), prefix_count=3)

    assert torch.allclose(mixed, tokens)
    mixed.sum().backward()
    assert tokens.grad is not None


def test_block_local_patch_mixer_supports_pruned_patch_indices() -> None:
    mixer = BlockLocalPatchMixer(embed_dim=8, dropout=0.0, residual_scale=0.10, zero_init=True)
    tokens = torch.randn(2, 6, 8)
    patch_indices = torch.tensor([[0, 2, 5, 8], [1, 3, 4, 7]], dtype=torch.long)

    mixed = mixer(
        tokens,
        grid_size=(3, 3),
        prefix_count=2,
        patch_indices=patch_indices,
    )

    assert torch.allclose(mixed, tokens)
    assert mixed.requires_grad
    mixed[:, 2:].sum().backward()
    assert mixer.pointwise.weight.grad is not None


def test_block_local_patch_mixer_trace_and_resume_extension_keep_logits() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    extended = create_model(
        num_classes=5,
        model_config=_tiny_config(
            block_local_patch_mixer=True,
            block_local_patch_mixer_layers="1,3",
            block_local_patch_mixer_dropout=0.0,
            block_local_patch_mixer_scale=0.10,
        ),
    )

    summary = _load_model_state_allowing_extensions(
        extended,
        base.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert summary["allowed_missing_keys"]
    assert all(".local_patch_mixer." in str(key) for key in summary["allowed_missing_keys"])

    base.eval()
    extended.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), extended(images), atol=1e-6)
        features = extended.forward_features(images, return_trace=True)

    assert features["trace"]["block_local_patch_mixer_layers"].tolist() == [1, 3]

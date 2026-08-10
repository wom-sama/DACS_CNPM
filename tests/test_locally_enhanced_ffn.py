import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.model import (
    FeedForward,
    LocallyEnhancedFeedForward,
    build_model_from_checkpoint,
    create_model,
)


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
        token_pruning=False,
        head_pooling="cls_register_mean",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_locally_enhanced_ffn_supports_dense_and_pruned_patch_grids() -> None:
    leff = LocallyEnhancedFeedForward(
        dim=8,
        hidden_dim=16,
        dropout=0.0,
        kernel_size=3,
    )
    dense = torch.randn(2, 11, 8, requires_grad=True)
    dense_output = leff(dense, grid_size=(3, 3), prefix_count=2)
    assert dense_output.shape == dense.shape

    sparse = torch.randn(2, 6, 8, requires_grad=True)
    patch_indices = torch.tensor([[0, 2, 5, 8], [1, 3, 4, 7]])
    sparse_output = leff(
        sparse,
        grid_size=(3, 3),
        prefix_count=2,
        patch_indices=patch_indices,
    )
    assert sparse_output.shape == sparse.shape
    (dense_output.mean() + sparse_output.mean()).backward()
    assert torch.isfinite(leff.depthwise.weight.grad).all()


def test_locally_enhanced_ffn_is_layer_selective_and_traced() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            locally_enhanced_ffn=True,
            locally_enhanced_ffn_layers="1,3",
            locally_enhanced_ffn_kernel_size=3,
        ),
    )
    assert isinstance(model.blocks[0].mlp, LocallyEnhancedFeedForward)
    assert isinstance(model.blocks[1].mlp, FeedForward)
    assert isinstance(model.blocks[2].mlp, LocallyEnhancedFeedForward)

    model.eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(2, 3, 32, 32), return_trace=True)
        logits = model(torch.randn(2, 3, 32, 32))
    assert features["trace"]["locally_enhanced_ffn_layers"].tolist() == [1, 3]
    assert logits.shape == (2, 5)


def test_locally_enhanced_ffn_checkpoint_roundtrip_and_default_schema() -> None:
    default_model = create_model(num_classes=5, model_config=_tiny_config())
    explicit_default = create_model(
        num_classes=5,
        model_config=_tiny_config(locally_enhanced_ffn=False),
    )
    assert set(default_model.state_dict()) == set(explicit_default.state_dict())
    assert all(isinstance(block.mlp, FeedForward) for block in default_model.blocks)

    config = _tiny_config(
        locally_enhanced_ffn=True,
        locally_enhanced_ffn_layers="1,2,3",
        locally_enhanced_ffn_kernel_size=3,
    )
    source = create_model(num_classes=5, model_config=config)
    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(config),
            "model_state": source.state_dict(),
        }
    )
    assert set(restored.state_dict()) == set(source.state_dict())
    assert all(isinstance(block.mlp, LocallyEnhancedFeedForward) for block in restored.blocks)


def test_locally_enhanced_ffn_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        LocallyEnhancedFeedForward(dim=8, hidden_dim=16, kernel_size=4)

import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.model import (
    ConcurrentLocalGlobalCoupling,
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


def test_concurrent_coupling_supports_dense_and_pruned_grids() -> None:
    coupling = ConcurrentLocalGlobalCoupling(
        embed_dim=8,
        local_dim=4,
        kernel_size=3,
    )
    dense_tokens = torch.randn(2, 11, 8, requires_grad=True)
    dense_local = torch.randn(2, 4, 3, 3, requires_grad=True)
    dense_output, dense_state = coupling(
        dense_tokens,
        dense_local,
        grid_size=(3, 3),
        prefix_count=2,
    )
    assert dense_output.shape == dense_tokens.shape
    assert dense_state.shape == dense_local.shape
    assert torch.equal(dense_output[:, :2], dense_tokens[:, :2])

    sparse_tokens = torch.randn(2, 6, 8, requires_grad=True)
    sparse_local = torch.randn(2, 4, 3, 3, requires_grad=True)
    patch_indices = torch.tensor([[0, 2, 5, 8], [1, 3, 4, 7]])
    sparse_output, sparse_state, trace = coupling(
        sparse_tokens,
        sparse_local,
        grid_size=(3, 3),
        prefix_count=2,
        patch_indices=patch_indices,
        return_trace=True,
    )
    assert sparse_output.shape == sparse_tokens.shape
    assert sparse_state.shape == sparse_local.shape
    assert trace["local_state_norm"].shape == (2,)
    assert trace["token_residual_norm"].shape == (2,)
    assert torch.equal(sparse_output[:, :2], sparse_tokens[:, :2])

    (dense_output.mean() + dense_state.mean() + sparse_output.mean() + sparse_state.mean()).backward()
    assert torch.isfinite(coupling.token_to_local.weight.grad).all()
    assert torch.isfinite(coupling.local_update[0].weight.grad).all()
    assert torch.isfinite(coupling.local_to_token[0].weight.grad).all()


def test_concurrent_coupling_is_layer_selective_and_traced() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            concurrent_local_global_coupling=True,
            concurrent_local_global_layers="1,3",
            concurrent_local_global_dim=8,
            concurrent_local_global_kernel_size=3,
        ),
    )
    assert set(model.concurrent_local_couplings.keys()) == {"1", "3"}

    model.eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(2, 3, 32, 32), return_trace=True)
        logits = model(torch.randn(2, 3, 32, 32))
    trace = features["trace"]
    assert trace["concurrent_local_global_layers"].tolist() == [1, 3]
    assert trace["concurrent_local_state_norm"].shape == (2, 2)
    assert trace["concurrent_token_residual_norm"].shape == (2, 2)
    assert trace["concurrent_local_activation"].shape == (2, 1, 4, 4)
    assert logits.shape == (2, 5)


def test_concurrent_coupling_checkpoint_roundtrip_and_default_schema() -> None:
    default_model = create_model(num_classes=5, model_config=_tiny_config())
    explicit_default = create_model(
        num_classes=5,
        model_config=_tiny_config(concurrent_local_global_coupling=False),
    )
    assert set(default_model.state_dict()) == set(explicit_default.state_dict())
    assert not default_model.concurrent_local_couplings

    config = _tiny_config(
        concurrent_local_global_coupling=True,
        concurrent_local_global_layers="1,2,3",
        concurrent_local_global_dim=8,
        concurrent_local_global_kernel_size=3,
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
    assert set(restored.concurrent_local_couplings.keys()) == {"1", "2", "3"}


def test_concurrent_coupling_rejects_invalid_dimensions_and_kernel() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        ConcurrentLocalGlobalCoupling(embed_dim=8, local_dim=0)
    with pytest.raises(ValueError, match="kernel_size"):
        ConcurrentLocalGlobalCoupling(embed_dim=8, local_dim=4, kernel_size=4)

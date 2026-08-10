import torch

from trkh.models.model import VisionTransformerWithRegisters
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_branch_model(branch_cnn_tokens: int) -> VisionTransformerWithRegisters:
    return VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        in_channels=3,
        use_cnn_stem=True,
        stem_channels=4,
        cnn_feature_fusion=True,
        multi_branch_fusion=True,
        branch_color_tokens=1,
        branch_edge_tokens=1,
        branch_cnn_tokens=branch_cnn_tokens,
        branch_token_dropout=0.0,
        detail_patch_enhancement=True,
        num_classes=5,
        embed_dim=16,
        depth=1,
        num_heads=2,
        num_registers=1,
        register_positional_embedding=True,
        fine_grained_pooling=True,
        head_pooling="cls_branch_register_mean",
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )


def test_resume_can_add_cnn_branch_token_without_losing_existing_branch_embeddings():
    source = _tiny_branch_model(branch_cnn_tokens=0)
    with torch.no_grad():
        source.branch_token_fusion.branch_type_embed.copy_(
            torch.arange(32, dtype=torch.float32).view(1, 2, 16)
        )
    state = source.state_dict()

    target = _tiny_branch_model(branch_cnn_tokens=1)
    summary = _load_model_state_allowing_extensions(
        target,
        state,
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert all(
        key.startswith("branch_token_fusion.cnn_branch.")
        for key in summary["allowed_missing_keys"]
    )
    assert tuple(target.branch_token_fusion.branch_type_embed.shape) == (1, 3, 16)
    torch.testing.assert_close(
        target.branch_token_fusion.branch_type_embed[:, :2],
        source.branch_token_fusion.branch_type_embed,
    )

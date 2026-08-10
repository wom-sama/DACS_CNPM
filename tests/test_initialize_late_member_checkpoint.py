from pathlib import Path

import torch

from trkh.models.model import VisionTransformerWithRegisters
from trkh.tools.initialize_late_member_checkpoint import (
    initialize_late_member_checkpoint,
)


def _checkpoint() -> dict[str, object]:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    return {
        "class_names": [f"class_{index}" for index in range(5)],
        "model_config": {
            "model_type": "vit_registers",
            "image_size": 32,
            "patch_size": 8,
            "use_cnn_stem": False,
            "embed_dim": 32,
            "depth": 4,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "num_registers": 2,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "drop_path_rate": 0.0,
        },
        "model_state": model.state_dict(),
        "train_model_state": {"bad": torch.tensor(1.0)},
        "optimizer_state": {"large": True},
    }


def test_initializer_preserves_primary_and_materializes_late_member() -> None:
    checkpoint = _checkpoint()
    output, summary = initialize_late_member_checkpoint(
        checkpoint,
        source_path=Path("keeper.pt"),
        source_sha256="a" * 64,
        fork_after_block=2,
        candidate_weight=0.4,
        focus_class=1,
        focus_margin_offset=0.034,
    )

    output_state = output["model_state"]
    assert isinstance(output_state, dict)
    for name, tensor in checkpoint["model_state"].items():
        assert torch.equal(output_state[name], tensor)
    assert torch.equal(
        output_state["late_member_blocks.0.attn.qkv.weight"],
        output_state["blocks.2.attn.qkv.weight"],
    )
    assert output["model_config"]["late_member_branch"] is True
    assert output["model_config"]["late_member_focus_margin_offset"] == 0.034
    assert summary["primary_tensors_bit_identical"] is True
    assert summary["state_numel_ratio"] > 1.0
    assert "train_model_state" not in output
    assert "optimizer_state" not in output

import pytest
import torch
import torch.nn.functional as F

from trkh.models.model import VisionTransformerWithRegisters
from trkh.training.train import (
    _apply_trainable_module_prefixes,
    _keep_late_member_primary_path_eval,
)


def _model(*, late_member: bool, focus_offset: float = 0.0):
    return VisionTransformerWithRegisters(
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
        late_member_branch=late_member,
        late_member_fork_after_block=2,
        late_member_candidate_weight=0.4,
        late_member_focus_class=1,
        late_member_focus_margin_offset=focus_offset,
    )


def test_late_member_loads_legacy_state_into_both_paths() -> None:
    torch.manual_seed(7)
    base = _model(late_member=False)
    late = _model(late_member=True)

    missing, unexpected = late.load_flexible_state_dict(base.state_dict(), strict=True)

    assert missing == []
    assert unexpected == []
    late_state = late.state_dict()
    assert torch.equal(
        late_state["late_member_blocks.0.attn.qkv.weight"],
        late_state["blocks.2.attn.qkv.weight"],
    )
    assert torch.equal(late_state["late_member_head.weight"], late_state["head.weight"])


def test_identical_late_member_preserves_base_probabilities() -> None:
    torch.manual_seed(11)
    base = _model(late_member=False).eval()
    late = _model(late_member=True).eval()
    late.load_flexible_state_dict(base.state_dict(), strict=True)
    images = torch.randn(3, 3, 32, 32)

    with torch.inference_mode():
        base_probabilities = F.softmax(base(images), dim=1)
        late_probabilities = F.softmax(late(images), dim=1)

    assert torch.allclose(late_probabilities, base_probabilities, atol=2e-6, rtol=1e-6)


def test_late_member_features_expose_primary_candidate_and_fused_logits() -> None:
    model = _model(late_member=True).eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        features = model.forward_features(images)
        logits = model.head(features["pooled"])
        assert "late_member_tokens" in features
        assert features["late_member_tokens"].shape == features["tokens"].shape
        fused = model.fuse_late_member_logits(logits, logits)

    assert torch.allclose(F.softmax(fused, dim=1), F.softmax(logits, dim=1), atol=1e-7)


def test_late_member_probability_margin_matches_frozen_rule() -> None:
    model = _model(late_member=True, focus_offset=0.034).eval()
    primary = torch.tensor([[2.0, 1.8, 0.2, -1.0, -2.0]])
    candidate = torch.tensor([[1.4, 2.1, 0.0, -1.0, -2.0]])

    fused = model.fuse_late_member_logits(primary, candidate)
    expected = 0.6 * F.softmax(primary.float(), dim=1) + 0.4 * F.softmax(
        candidate.float(), dim=1
    )
    expected[:, 1] = (expected[:, 1] - 0.034).clamp_min(1e-8)
    expected = expected / expected.sum(dim=1, keepdim=True)

    assert torch.allclose(F.softmax(fused, dim=1), expected, atol=1e-7)


def test_late_member_rejects_token_pruning_after_fork() -> None:
    with pytest.raises(ValueError, match="token pruning"):
        VisionTransformerWithRegisters(
            image_size=32,
            patch_size=8,
            use_cnn_stem=False,
            num_classes=5,
            embed_dim=32,
            depth=4,
            num_heads=4,
            num_registers=2,
            token_pruning=True,
            token_prune_layers="3",
            token_keep_rates="0.5",
            late_member_branch=True,
            late_member_fork_after_block=2,
        )


def test_late_member_only_training_keeps_primary_path_in_eval_mode() -> None:
    model = _model(late_member=True)
    _apply_trainable_module_prefixes(
        model,
        ["late_member_blocks", "late_member_norm", "late_member_head"],
    )

    model.train()
    frozen_children = _keep_late_member_primary_path_eval(model)

    assert frozen_children > 0
    assert model.training is True
    assert model.blocks.training is False
    assert model.head.training is False
    assert model.late_member_blocks.training is True
    assert model.late_member_head.training is True


def test_late_member_native_attention_uses_shared_then_candidate_blocks() -> None:
    model = _model(late_member=True).eval()

    with torch.inference_mode():
        features = model.forward_features(
            torch.randn(1, 3, 32, 32),
            return_attention=True,
        )

    assert features["attention_member"] == "late_member"
    assert list(features["primary_attentions"]) == [0, 1, 2, 3]
    assert list(features["late_member_attentions"]) == [0, 1, 2, 3]
    assert features["attentions"] is features["late_member_attentions"]

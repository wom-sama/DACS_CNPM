from pathlib import Path

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import GatedPatchRelativePositionAttention, create_model
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="1",
        token_keep_rates="0.5",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_zero_gate_is_exact_identity_and_has_gradient() -> None:
    module = GatedPatchRelativePositionAttention(
        num_heads=4,
        max_mix=0.25,
        locality_strength=1.0,
    )
    logits = torch.randn(2, 4, 9, 9)
    attention = logits.softmax(dim=-1)
    target = torch.randn_like(attention)

    mixed = module(
        attention,
        grid_size=(2, 3),
        prefix_count=3,
    )

    torch.testing.assert_close(mixed, attention, rtol=0.0, atol=0.0)
    (mixed * target).sum().backward()
    assert module.gate.grad is not None
    assert float(module.gate.grad.abs().sum()) > 0.0


def test_negative_raw_gate_stays_identity_but_can_recover_during_training() -> None:
    module = GatedPatchRelativePositionAttention(num_heads=2, max_mix=0.25)
    module.train()
    with torch.no_grad():
        module.gate.fill_(-0.05)
    attention = torch.randn(1, 2, 7, 7).softmax(dim=-1)
    target = torch.randn_like(attention)

    mixed = module(attention, grid_size=(2, 2), prefix_count=3)

    torch.testing.assert_close(mixed, attention, rtol=0.0, atol=0.0)
    (mixed * target).sum().backward()
    assert module.gate.grad is not None
    assert float(module.gate.grad.abs().sum()) > 0.0


def test_patch_mix_preserves_prefix_attention_and_patch_mass_after_pruning() -> None:
    module = GatedPatchRelativePositionAttention(
        num_heads=2,
        max_mix=0.30,
        locality_strength=1.5,
    )
    with torch.no_grad():
        module.gate.fill_(0.20)
    attention = torch.randn(2, 2, 7, 7).softmax(dim=-1)
    patch_indices = torch.tensor(
        [[0, 2, 5, 8], [1, 3, 4, 7]],
        dtype=torch.long,
    )

    mixed = module(
        attention,
        grid_size=(3, 3),
        prefix_count=3,
        patch_indices=patch_indices,
    )

    torch.testing.assert_close(mixed[:, :, :3], attention[:, :, :3])
    torch.testing.assert_close(mixed[:, :, 3:, :3], attention[:, :, 3:, :3])
    torch.testing.assert_close(
        mixed[:, :, 3:, 3:].sum(dim=-1),
        attention[:, :, 3:, 3:].sum(dim=-1),
        rtol=1e-6,
        atol=1e-7,
    )
    trace = module.trace()
    assert trace["gate"].tolist() == [0.20000000298023224, 0.20000000298023224]
    assert trace["center_map"].numel() == 9
    assert torch.all(trace["position_local_mass"] > 0.0)


def test_resume_extension_keeps_logits_and_traces_pruned_layers() -> None:
    torch.manual_seed(11)
    source = create_model(num_classes=5, model_config=_tiny_config()).eval()
    target = create_model(
        num_classes=5,
        model_config=_tiny_config(
            gated_relative_position_attention=True,
            gated_relative_position_attention_layers="1,2",
            gated_relative_position_attention_max_mix=0.25,
            gated_relative_position_attention_locality_strength=1.0,
        ),
    ).eval()

    summary = _load_model_state_allowing_extensions(
        target,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert summary["allowed_missing_keys"]
    assert all(
        ".attn.relative_position_attention." in str(key)
        for key in summary["allowed_missing_keys"]
    )
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        source_logits = source(images)
        target_logits = target(images)
        features = target.forward_features(images, return_trace=True)
    torch.testing.assert_close(target_logits, source_logits, rtol=0.0, atol=0.0)
    trace = features["trace"]
    assert trace["relative_position_attention_layers"].tolist() == [1, 2]
    torch.testing.assert_close(
        trace["relative_position_attention_gate"],
        torch.zeros_like(trace["relative_position_attention_gate"]),
    )
    assert trace["relative_position_attention_center_map"].shape == (2, 16)
    assert features["patch_indices"].shape[1] == 8


def test_v8_launcher_exposes_gated_relative_position_attention() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")

    assert "[bool]$GatedRelativePositionAttention = $false" in script
    assert '"--gated-relative-position-attention"' in script
    assert (
        '"--gated-relative-position-attention-layers", '
        '"$GatedRelativePositionAttentionLayers"'
    ) in script
    assert "gated_relative_position_attention_max_mix" in script

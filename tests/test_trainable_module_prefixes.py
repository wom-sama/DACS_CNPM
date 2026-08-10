from __future__ import annotations

import torch

from trkh.models.model import VisionTransformerWithRegisters
from trkh.training.train import (
    ModelEMA,
    _apply_trainable_module_prefixes,
    _keep_frozen_norm_modules_eval,
    _resolve_model_ema_resume_updates,
    _state_names_matching_trainable_prefixes,
)


def test_trainable_module_prefixes_freezes_everything_else() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        source_context_feature_fusion=True,
        source_context_fusion_hidden_dim=16,
    )

    summary = _apply_trainable_module_prefixes(
        model,
        ["source_context_fusion_head"],
    )

    assert summary["enabled"] is True
    assert summary["trainable_parameters"] > 0
    assert summary["frozen_parameters"] > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith("source_context_fusion_head.")


def test_trainable_module_prefixes_supports_paired_view_fusion_head() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=4,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        paired_view_feature_fusion=True,
        paired_view_fusion_hidden_dim=16,
    )

    summary = _apply_trainable_module_prefixes(
        model,
        ["paired_view_fusion_head"],
    )

    assert summary["enabled"] is True
    assert summary["trainable_parameters"] > 0
    assert summary["frozen_parameters"] > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith("paired_view_fusion_head.")


def test_trainable_module_prefixes_freezes_batchnorm_running_stats() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=True,
        stem_channels=4,
        num_classes=3,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_registers=1,
        paired_view_feature_fusion=True,
        paired_view_fusion_hidden_dim=16,
    )

    summary = _apply_trainable_module_prefixes(
        model,
        ["paired_view_fusion_head"],
    )

    assert summary["frozen_norm_modules"] > 0
    for module in model.modules():
        if module.__class__.__name__.startswith("BatchNorm"):
            assert module.momentum == 0.0


def test_trainable_module_prefixes_keep_batchnorm_eval_after_model_train() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=True,
        stem_channels=4,
        num_classes=3,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_registers=1,
        paired_view_feature_fusion=True,
        paired_view_fusion_hidden_dim=16,
    )
    _apply_trainable_module_prefixes(model, ["paired_view_fusion_head"])
    before = [
        module.num_batches_tracked.clone()
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]

    model.train()
    _keep_frozen_norm_modules_eval(model)
    with torch.no_grad():
        model(torch.rand(2, 3, 32, 32))

    after = [
        module.num_batches_tracked.clone()
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert [int(value.item()) for value in after] == [int(value.item()) for value in before]


def test_model_ema_update_can_be_limited_to_trainable_prefix_state() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=4,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        paired_view_feature_fusion=True,
        paired_view_fusion_hidden_dim=16,
    )
    ema = ModelEMA(model, decay=0.5, updates=100)
    _apply_trainable_module_prefixes(model, ["paired_view_fusion_head"])
    update_names = _state_names_matching_trainable_prefixes(
        model,
        ["paired_view_fusion_head"],
    )

    frozen_name, frozen_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("paired_view_fusion_head.")
    )
    trainable_name, trainable_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("paired_view_fusion_head.")
    )
    frozen_before = ema.state_dict()[frozen_name].clone()
    trainable_before = ema.state_dict()[trainable_name].clone()

    with torch.no_grad():
        frozen_parameter.add_(1.0)
        trainable_parameter.add_(1.0)
    ema.update(model, update_state_names=update_names)

    assert torch.equal(ema.state_dict()[frozen_name], frozen_before)
    assert not torch.equal(ema.state_dict()[trainable_name], trainable_before)


def test_partial_architecture_ema_load_resets_warmup_counter() -> None:
    checkpoint = {"ema_updates": 240}

    assert _resolve_model_ema_resume_updates(checkpoint, None) == 240
    assert (
        _resolve_model_ema_resume_updates(
            checkpoint,
            {"allowed_missing_keys": ["new_adapter.weight"]},
        )
        == 0
    )

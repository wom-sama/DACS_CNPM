from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.model import FeedForward, MultiHeadSelfAttention, create_model
from trkh.models.patch_style_recalibration import PatchStyleRecalibration
from trkh.training.train import (
    _load_model_state_allowing_extensions,
    parse_args,
)


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=6,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="2,5",
        token_keep_rates="0.75,0.50",
        patch_style_recalibration=True,
        patch_style_recalibration_layers="2,5",
    )
    values.update(overrides)
    return ModelConfig(**values)


def _expected_missing_keys(layer_indices: tuple[int, ...]) -> set[str]:
    suffixes = (
        "cfc",
        "bn.weight",
        "bn.bias",
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
    )
    return {
        f"blocks.{layer}.mlp.style_recalibration.{suffix}"
        for layer in layer_indices
        for suffix in suffixes
    }


def test_srm_matches_explicit_population_style_equation_and_is_patch_only() -> None:
    module = PatchStyleRecalibration(hidden_dim=4).eval()
    with torch.no_grad():
        module.cfc.copy_(
            torch.tensor(
                [
                    [0.20, -0.10],
                    [-0.15, 0.05],
                    [0.07, 0.12],
                    [-0.04, -0.08],
                ]
            )
        )
        module.bn.running_mean.copy_(torch.tensor([0.10, -0.05, 0.20, -0.10]))
        module.bn.running_var.copy_(torch.tensor([0.70, 1.30, 0.90, 1.10]))
        module.bn.weight.copy_(torch.tensor([1.10, 0.80, 1.25, 0.95]))
        module.bn.bias.copy_(torch.tensor([-0.10, 0.20, 0.05, -0.15]))

    hidden = torch.tensor(
        [
            [
                [9.0, 8.0, 7.0, 6.0],
                [0.2, 0.5, -0.3, 1.0],
                [0.8, -0.1, 0.7, -0.4],
                [-0.5, 0.4, 1.2, 0.3],
            ],
            [
                [5.0, 4.0, 3.0, 2.0],
                [-0.4, 0.9, 0.1, 0.2],
                [0.6, 0.3, -0.8, 1.1],
                [1.0, -0.7, 0.5, -0.2],
            ],
        ]
    )
    output, gate = module(hidden, prefix_count=1, return_gate=True)

    patches = hidden[:, 1:]
    mean = patches.mean(dim=1)
    variance = (patches - mean.unsqueeze(1)).square().mean(dim=1)
    style = torch.stack((mean, (variance + module.eps).sqrt()), dim=-1)
    encoded = (style * module.cfc.unsqueeze(0)).sum(dim=-1)
    normalized = (
        (encoded - module.bn.running_mean)
        / torch.sqrt(module.bn.running_var + module.bn.eps)
        * module.bn.weight
        + module.bn.bias
    )
    expected_gate = torch.sigmoid(normalized)

    torch.testing.assert_close(gate, expected_gate)
    torch.testing.assert_close(output[:, :1], hidden[:, :1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[:, 1:], patches * expected_gate.unsqueeze(1))


def test_srm_initial_eval_gate_is_exactly_half() -> None:
    module = PatchStyleRecalibration(hidden_dim=8).eval()
    hidden = torch.randn(3, 7, 8)
    output, gate = module(hidden, prefix_count=2, return_gate=True)

    torch.testing.assert_close(gate, torch.full_like(gate, 0.5), rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[:, :2], hidden[:, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[:, 2:], hidden[:, 2:] * 0.5)
    assert float(module.trace()["patch_hidden_norm_ratio"]) == pytest.approx(0.5)


def test_default_feed_forward_schema_and_behavior_remain_exact() -> None:
    torch.manual_seed(5)
    feed_forward = FeedForward(dim=16, hidden_dim=32, dropout=0.0).eval()
    inputs = torch.randn(2, 9, 16)

    assert feed_forward.style_recalibration is None
    assert set(feed_forward.state_dict()) == {
        "net.0.weight",
        "net.0.bias",
        "net.3.weight",
        "net.3.bias",
    }
    with torch.inference_mode():
        actual = feed_forward(inputs, prefix_count=3)
        expected = feed_forward.net(inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_model_wiring_preserves_spatial_attention_pruning_and_emits_srm_trace() -> None:
    torch.manual_seed(7)
    model = create_model(num_classes=5, model_config=_tiny_config()).eval()
    assert all(isinstance(block.attn, MultiHeadSelfAttention) for block in model.blocks)
    assert [
        index + 1
        for index, block in enumerate(model.blocks)
        if block.mlp.style_recalibration is not None
    ] == [2, 5]
    images = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        spatial = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
        pruned = model.forward_features(images, return_trace=True)

    assert set(spatial["attentions"]) == {0, 1, 2, 3, 4, 5}
    assert spatial["attention_representation"] == "mhsa_probability"
    trace = spatial["trace"]
    assert trace["patch_style_recalibration_layers"].tolist() == [2, 5]
    for key in (
        "patch_style_gate_min",
        "patch_style_gate_mean",
        "patch_style_gate_max",
        "patch_style_gate_channel_std",
        "patch_style_gate_sample_std",
        "patch_style_style_mean_abs",
        "patch_style_style_std_mean",
        "patch_style_cfc_l2_norm",
        "patch_style_patch_hidden_norm_ratio",
        "patch_style_patch_count",
    ):
        assert trace[key].shape == (2,), key
    torch.testing.assert_close(
        trace["patch_style_gate_mean"],
        torch.full((2,), 0.5),
        rtol=0.0,
        atol=0.0,
    )
    assert [int(item["layer"]) for item in pruned["trace"]["pruning"]] == [2, 5]


def test_srm_resume_extension_has_exact_keys_and_preserves_existing_tensors() -> None:
    torch.manual_seed(11)
    source = create_model(
        num_classes=5,
        model_config=_tiny_config(patch_style_recalibration=False),
    )
    torch.manual_seed(11)
    candidate = create_model(num_classes=5, model_config=_tiny_config())

    summary = _load_model_state_allowing_extensions(
        candidate,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    expected_state_additions = _expected_missing_keys((1, 4))
    assert set(candidate.state_dict()) - set(source.state_dict()) == expected_state_additions
    # PyTorch deliberately suppresses missing BatchNorm batch counters for
    # backward compatibility, so the loader reports the other ten entries.
    assert set(summary["allowed_missing_keys"]) == {
        key for key in expected_state_additions if not key.endswith("num_batches_tracked")
    }
    candidate_state = candidate.state_dict()
    for key, value in source.state_dict().items():
        torch.testing.assert_close(candidate_state[key], value, rtol=0.0, atol=0.0)


def test_exact_candidate_parameter_increment_is_8192() -> None:
    base = dict(
        image_size=16,
        patch_size=8,
        embed_dim=256,
        depth=5,
        num_heads=8,
        token_pruning=False,
    )
    source = create_model(
        num_classes=5,
        model_config=_tiny_config(**base, patch_style_recalibration=False),
    )
    candidate = create_model(
        num_classes=5,
        model_config=_tiny_config(**base),
    )
    source_count = sum(parameter.numel() for parameter in source.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    assert candidate_count - source_count == 8_192


def test_constructor_and_forward_do_not_advance_rng_streams() -> None:
    seed = 13
    torch.manual_seed(seed)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(patch_style_recalibration=False),
    ).eval()
    control_constructor_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = create_model(num_classes=5, model_config=_tiny_config()).eval()
    candidate_constructor_rng = torch.get_rng_state().clone()
    assert torch.equal(control_constructor_rng, candidate_constructor_rng)

    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(17)
    with torch.inference_mode():
        control(images)
    control_forward_rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    with torch.inference_mode():
        candidate(images)
    candidate_forward_rng = torch.get_rng_state().clone()
    assert torch.equal(control_forward_rng, candidate_forward_rng)


def test_warmed_srm_and_shared_model_families_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(19)
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            depth=3,
            patch_style_recalibration_layers="1,2",
            token_pruning=False,
        ),
    ).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    warm_images = torch.randn(8, 3, 32, 32)
    warm_loss = F.cross_entropy(
        model(warm_images),
        torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
    )
    warm_loss.backward()
    for block in model.blocks:
        module = block.mlp.style_recalibration
        if module is None:
            continue
        assert module.cfc.grad is not None
        assert int(torch.count_nonzero(module.cfc.grad).item()) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    images = torch.randn(8, 3, 32, 32, requires_grad=True)
    loss = F.cross_entropy(
        model(images),
        torch.tensor([1, 2, 3, 4, 0, 1, 2, 3]),
    )
    loss.backward()
    first_srm = model.blocks[0].mlp.style_recalibration
    required = {
        "cfc": first_srm.cfc,
        "bn.weight": first_srm.bn.weight,
        "bn.bias": first_srm.bn.bias,
        "ffn.fc1": model.blocks[0].mlp.net[0].weight,
        "ffn.fc2": model.blocks[0].mlp.net[3].weight,
        "head": model.head.weight,
        "input": images,
    }
    for name, tensor in required.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert int(torch.count_nonzero(tensor.grad).item()) > 0, name


def test_defaults_conflicts_parser_and_launcher_controls() -> None:
    defaults = ModelConfig()
    assert defaults.patch_style_recalibration is False
    assert defaults.patch_style_recalibration_layers == "2,5"
    parsed = parse_args(
        [
            "--patch-style-recalibration",
            "--patch-style-recalibration-layers",
            "1,4",
        ]
    )
    assert parsed.patch_style_recalibration is True
    assert parsed.patch_style_recalibration_layers == "1,4"

    with pytest.raises(ValueError, match="locally enhanced FFN"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(
                locally_enhanced_ffn=True,
                locally_enhanced_ffn_layers="1,2,3,4,5",
            ),
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$PatchStyleRecalibration = $false" in script
    assert '[string]$PatchStyleRecalibrationLayers = "2,5"' in script
    assert '"--patch-style-recalibration"' in script
    assert '"--patch-style-recalibration-layers"' in script

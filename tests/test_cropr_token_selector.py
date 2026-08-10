from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.cropr_token_selector import CroprTokenSelector
from trkh.models.model import classification_logits_from_features, create_model
from trkh.training.train import _cropr_auxiliary_loss_from_features


OFFICIAL_CROPR_PATH = (
    Path(r"D:\DataAI\external_sources\official\cropr-cvpr2025")
    / "cls"
    / "cropr.py"
)


def _tiny_config(*, selector: bool, routing: bool = True, **overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "model_type": "vit_registers",
        "image_size": 32,
        "patch_size": 8,
        "use_cnn_stem": False,
        "embed_dim": 32,
        "depth": 6,
        "num_heads": 4,
        "num_registers": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "multi_branch_fusion": True,
        "branch_color_tokens": 1,
        "branch_edge_tokens": 1,
        "branch_cnn_tokens": 0,
        "branch_token_dropout": 0.0,
        "head_pooling": "cls_branch_register_mean",
        "token_pruning": True,
        "token_prune_layers": "2,5",
        "token_keep_rates": "0.85,0.65",
        "token_prune_foreground_weight": 0.35,
        "cropr_token_selector": selector,
        "cropr_token_selector_routing": routing,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _official_cropr_class():
    if not OFFICIAL_CROPR_PATH.is_file():
        pytest.skip(f"Locked official Cropr source is absent: {OFFICIAL_CROPR_PATH}")
    spec = importlib.util.spec_from_file_location(
        "locked_cvpr2025_cropr",
        OFFICIAL_CROPR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the locked official Cropr source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Cropr


def _copy_candidate_to_official(
    candidate: CroprTokenSelector,
    official: torch.nn.Module,
) -> None:
    with torch.no_grad():
        official.cross_attn.queries.copy_(candidate.query)
        official.cross_attn.mlp_norm.load_state_dict(candidate.mlp_norm.state_dict())
        official.cross_attn.mlp.fc1.load_state_dict(candidate.mlp[0].state_dict())
        official.cross_attn.mlp.fc2.load_state_dict(candidate.mlp[2].state_dict())
        official.head.norm.load_state_dict(candidate.head_norm.state_dict())
        official.head.head.load_state_dict(candidate.head.state_dict())


def test_cropr_scores_aggregation_logits_and_gradients_match_official() -> None:
    official_class = _official_cropr_class()
    torch.manual_seed(41)
    candidate = CroprTokenSelector(dim=16, num_classes=5, mlp_ratio=4.0)
    official = official_class(
        pruning_rate=3,
        num_queries=1,
        num_classes=5,
        embed_dim=16,
        num_heads=1,
        pre_attn_norm=False,
        q_proj=False,
        k_proj=False,
        v_proj=False,
        mlp=True,
        mlp_ratio=4.0,
        training=True,
    )
    _copy_candidate_to_official(candidate, official)

    torch.manual_seed(19)
    candidate_input = torch.randn(3, 9, 16, requires_grad=True)
    official_input = candidate_input.detach().clone().requires_grad_(True)
    scores, logits, trace = candidate(
        candidate_input,
        collect_auxiliary=True,
        return_trace=True,
    )
    official_aggregate, official_scores = official.cross_attn(
        official_input.detach()
    )
    official_logits = official.head(official_aggregate)

    assert logits is not None
    torch.testing.assert_close(scores, official_scores, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        trace["attention"],
        (official_scores * (16**-0.5)).softmax(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        trace["aggregated"], official_aggregate, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(logits, official_logits, atol=1e-6, rtol=1e-6)
    assert torch.equal(
        torch.topk(scores, k=6, dim=1).indices,
        torch.topk(official_scores, k=6, dim=1).indices,
    )

    targets = torch.tensor([0, 2, 4])
    F.cross_entropy(logits.float(), targets).backward()
    F.cross_entropy(official_logits.float(), targets).backward()
    assert candidate_input.grad is None
    assert official_input.grad is None
    mappings = {
        "query": "cross_attn.queries",
        "mlp_norm.weight": "cross_attn.mlp_norm.weight",
        "mlp_norm.bias": "cross_attn.mlp_norm.bias",
        "mlp.0.weight": "cross_attn.mlp.fc1.weight",
        "mlp.0.bias": "cross_attn.mlp.fc1.bias",
        "mlp.2.weight": "cross_attn.mlp.fc2.weight",
        "mlp.2.bias": "cross_attn.mlp.fc2.bias",
        "head_norm.weight": "head.norm.weight",
        "head_norm.bias": "head.norm.bias",
        "head.weight": "head.head.weight",
        "head.bias": "head.head.bias",
    }
    candidate_parameters = dict(candidate.named_parameters())
    official_parameters = dict(official.named_parameters())
    for candidate_name, official_name in mappings.items():
        observed = candidate_parameters[candidate_name].grad
        expected = official_parameters[official_name].grad
        assert observed is not None, candidate_name
        assert expected is not None, official_name
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)


def test_cropr_direct_equation_and_parameter_gradients_match() -> None:
    torch.manual_seed(7)
    selector = CroprTokenSelector(dim=12, num_classes=5, mlp_ratio=4.0)
    patch_tokens = torch.randn(4, 7, 12, requires_grad=True)
    scores, logits, trace = selector(
        patch_tokens,
        collect_auxiliary=True,
        return_trace=True,
    )
    assert logits is not None

    clones = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in selector.named_parameters()
    }
    detached_input = patch_tokens.detach()
    manual_scores = (
        clones["query"].expand(4, -1, -1) @ detached_input.transpose(1, 2)
    ).squeeze(1)
    manual_attention = (manual_scores * (12**-0.5)).softmax(dim=-1)
    manual_aggregate = manual_attention.unsqueeze(1) @ detached_input
    manual_aggregate = manual_aggregate.squeeze(1)
    normalized = F.layer_norm(
        manual_aggregate,
        (12,),
        clones["mlp_norm.weight"],
        clones["mlp_norm.bias"],
        1e-6,
    )
    hidden = F.gelu(
        F.linear(normalized, clones["mlp.0.weight"], clones["mlp.0.bias"])
    )
    manual_aggregate = manual_aggregate + F.linear(
        hidden,
        clones["mlp.2.weight"],
        clones["mlp.2.bias"],
    )
    manual_head_input = F.layer_norm(
        manual_aggregate,
        (12,),
        clones["head_norm.weight"],
        clones["head_norm.bias"],
        1e-5,
    )
    manual_logits = F.linear(
        manual_head_input,
        clones["head.weight"],
        clones["head.bias"],
    )

    torch.testing.assert_close(scores, manual_scores, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        trace["attention"], manual_attention, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        trace["aggregated"], manual_aggregate, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(logits, manual_logits, atol=1e-6, rtol=1e-6)

    targets = torch.tensor([0, 1, 3, 4])
    F.cross_entropy(logits.float(), targets).backward()
    F.cross_entropy(manual_logits.float(), targets).backward()
    for name, parameter in selector.named_parameters():
        assert parameter.grad is not None, name
        assert clones[name].grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            clones[name].grad,
            atol=1e-6,
            rtol=1e-6,
        )
    assert patch_tokens.grad is None


def test_cropr_detached_auxiliary_path_updates_every_parameter_family() -> None:
    selector = CroprTokenSelector(dim=16, num_classes=5)
    patches = torch.randn(6, 11, 16, requires_grad=True)
    _, logits, _ = selector(patches, collect_auxiliary=True)
    assert logits is not None
    F.cross_entropy(logits.float(), torch.tensor([0, 1, 2, 3, 4, 0])).backward()
    assert patches.grad is None
    for name, parameter in selector.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert int(torch.count_nonzero(parameter.grad).item()) > 0, name


def test_cropr_synthetic_relevant_token_is_learned() -> None:
    torch.manual_seed(101)
    batch_size, token_count, dim = 160, 8, 16
    labels = torch.arange(batch_size) % 5
    relevant_indices = torch.arange(batch_size) % token_count
    patches = 0.03 * torch.randn(batch_size, token_count, dim)
    patches[:, :, 0] -= 1.0
    rows = torch.arange(batch_size)
    patches[rows, relevant_indices, 0] = 5.0
    patches[rows, relevant_indices, 1 + labels] = 3.0

    selector = CroprTokenSelector(dim=dim, num_classes=5)
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.03)
    with torch.no_grad():
        _, initial_logits, _ = selector(patches, collect_auxiliary=True)
        assert initial_logits is not None
        initial_loss = F.cross_entropy(initial_logits.float(), labels).item()
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        scores, logits, _ = selector(patches, collect_auxiliary=True)
        assert logits is not None
        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        scores, final_logits, _ = selector(patches, collect_auxiliary=True)
        assert final_logits is not None
        final_loss = F.cross_entropy(final_logits.float(), labels).item()
        hit_rate = (scores.argmax(dim=1) == relevant_indices).float().mean().item()
    assert final_loss <= 0.50 * initial_loss
    assert hit_rate >= 0.95


def test_cropr_disabled_is_bit_exact_to_default_construction() -> None:
    torch.manual_seed(42)
    default_model = create_model(
        num_classes=5,
        model_config=_tiny_config(selector=False),
    ).eval()
    default_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    explicit_disabled = create_model(
        num_classes=5,
        model_config=_tiny_config(
            selector=False,
            cropr_token_selector_routing=False,
        ),
    ).eval()
    explicit_rng = torch.random.get_rng_state().clone()
    assert torch.equal(default_rng, explicit_rng)
    assert list(default_model.state_dict()) == list(explicit_disabled.state_dict())
    for name, value in default_model.state_dict().items():
        assert torch.equal(value, explicit_disabled.state_dict()[name]), name

    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        default_features = default_model.forward_features(images, return_trace=True)
        disabled_features = explicit_disabled.forward_features(
            images, return_trace=True
        )
    torch.testing.assert_close(
        classification_logits_from_features(default_model, default_features),
        classification_logits_from_features(explicit_disabled, disabled_features),
        atol=0.0,
        rtol=0.0,
    )
    assert set(default_features["trace"]) == set(disabled_features["trace"])
    for default_stage, disabled_stage in zip(
        default_features["trace"]["pruning"],
        disabled_features["trace"]["pruning"],
    ):
        assert set(default_stage) == set(disabled_stage)
        for key in default_stage:
            assert torch.equal(default_stage[key], disabled_stage[key]), key


def test_cropr_control_and_candidate_have_identical_state_and_rng() -> None:
    torch.manual_seed(42)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(selector=True, routing=False),
    )
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(
        num_classes=5,
        model_config=_tiny_config(selector=True, routing=True),
    )
    candidate_rng = torch.random.get_rng_state().clone()
    assert torch.equal(control_rng, candidate_rng)
    assert list(control.state_dict()) == list(candidate.state_dict())
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    for name, value in control.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
        assert torch.isfinite(value).all(), name
        assert "routing" not in name


def test_cropr_control_routing_is_bit_exact_to_native_pruning() -> None:
    torch.manual_seed(29)
    native = create_model(
        num_classes=5,
        model_config=_tiny_config(selector=False),
    ).eval()
    torch.manual_seed(29)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(selector=True, routing=False),
    ).eval()
    native_state = native.state_dict()
    for name, value in native_state.items():
        assert torch.equal(value, control.state_dict()[name]), name

    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        native_features = native.forward_features(images, return_trace=True)
        control_features = control.forward_features(images, return_trace=True)
    native_logits = classification_logits_from_features(native, native_features)
    control_logits = classification_logits_from_features(control, control_features)
    torch.testing.assert_close(native_logits, control_logits, atol=0.0, rtol=0.0)
    assert torch.equal(native_features["patch_indices"], control_features["patch_indices"])
    assert set(control_features["cropr_auxiliary_logits"]) == {"2", "5"}
    for native_stage, control_stage in zip(
        native_features["trace"]["pruning"],
        control_features["trace"]["pruning"],
    ):
        assert torch.equal(native_stage["kept_indices"], control_stage["kept_indices"])
        assert torch.equal(native_stage["scores"], control_stage["scores"])
        assert not bool(control_stage["cropr_routing"].item())


def test_cropr_candidate_standard_trace_parity_and_complete_partitions() -> None:
    torch.manual_seed(31)
    candidate = create_model(
        num_classes=5,
        model_config=_tiny_config(selector=True, routing=True),
    ).eval()
    images = torch.randn(5, 3, 32, 32)
    with torch.no_grad():
        standard = candidate.forward_features(images)
        traced = candidate.forward_features(images, return_trace=True)
    standard_logits = classification_logits_from_features(candidate, standard)
    traced_logits = classification_logits_from_features(candidate, traced)
    torch.testing.assert_close(standard_logits, traced_logits, atol=1e-6, rtol=0.0)
    assert torch.equal(standard["patch_indices"], traced["patch_indices"])
    assert "cropr_auxiliary_logits" not in standard
    assert set(traced["cropr_auxiliary_logits"]) == {"2", "5"}
    assert traced["trace"]["cropr_token_selector_layers"].tolist() == [2, 5]
    assert bool(traced["trace"]["cropr_token_selector_routing"].item())

    previous = torch.arange(16).unsqueeze(0).expand(images.size(0), -1)
    expected_counts = [14, 11]
    for stage, expected_count in zip(traced["trace"]["pruning"], expected_counts):
        kept = stage["kept_indices"]
        dropped = stage["dropped_indices"]
        assert kept.shape[1] == expected_count
        assert torch.equal(kept, kept.sort(dim=1).values)
        assert bool(stage["cropr_routing"].item())
        assert stage["raw_scores"].shape == previous.shape
        assert stage["attention"].shape == previous.shape
        assert stage["native_scores"].shape == previous.shape
        assert stage["cropr_scores"].shape == previous.shape
        torch.testing.assert_close(
            stage["attention"].sum(dim=1),
            torch.ones(images.size(0)),
            atol=1e-6,
            rtol=0.0,
        )
        for row in range(images.size(0)):
            kept_set = set(kept[row].tolist())
            dropped_set = set(dropped[row].tolist())
            previous_set = set(previous[row].tolist())
            assert kept_set.isdisjoint(dropped_set)
            assert kept_set | dropped_set == previous_set
        previous = kept


def test_cropr_auxiliary_loss_is_unweighted_sum_of_natural_ce() -> None:
    logits_2 = torch.randn(6, 5, requires_grad=True)
    logits_5 = torch.randn(6, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 1])
    loss, stats = _cropr_auxiliary_loss_from_features(
        features={"cropr_auxiliary_logits": {"2": logits_2, "5": logits_5}},
        targets=targets,
    )
    expected = F.cross_entropy(logits_2.float(), targets) + F.cross_entropy(
        logits_5.float(), targets
    )
    torch.testing.assert_close(loss, expected, atol=0.0, rtol=0.0)
    assert stats["layer_count"] == 2
    assert stats["used_layers"] == ["2", "5"]
    assert set(stats["loss_by_layer"]) == {"2", "5"}
    loss.backward()
    assert logits_2.grad is not None
    assert logits_5.grad is not None

    with pytest.raises(ValueError, match="hard class-index targets"):
        _cropr_auxiliary_loss_from_features(
            features={"cropr_auxiliary_logits": {"2": logits_2.detach()}},
            targets=F.one_hot(targets, num_classes=5).float(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"token_pruning": False},
        {"token_prune_layers": "2,4"},
        {"token_keep_rates": "0.80,0.65"},
        {"token_prune_foreground_weight": 0.45},
        {"early_token_mask_keep_rate": 0.90},
        {"inattentive_token_fusion": True},
        {"deep_class_prompt": True},
        {
            "concurrent_local_global_coupling": True,
            "concurrent_local_global_layers": "1,2,3,4,5,6",
        },
        {"late_member_branch": True, "late_member_fork_after_block": 3},
    ],
)
def test_cropr_locked_configuration_rejects_confounds(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="cropr_token_selector"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(selector=True, **overrides),
        )


def test_v8_launcher_exposes_locked_cropr_control_and_candidate_flags() -> None:
    launcher = Path("scripts/run_trkh_5class_attention_views_v8.ps1").read_text(
        encoding="utf-8"
    )
    assert "[bool]$CroprTokenSelector = $false" in launcher
    assert "[bool]$CroprTokenSelectorRouting = $true" in launcher
    assert "[double]$TokenPruneForegroundWeight = 0.45" in launcher
    assert '"--cropr-token-selector"' in launcher
    assert '"--disable-cropr-token-selector-routing"' in launcher
    assert '"--token-prune-foreground-weight", "$TokenPruneForegroundWeight"' in launcher
    assert "CroprTokenSelector A0 yeu cau TokenPruneForegroundWeight=0.35" in launcher

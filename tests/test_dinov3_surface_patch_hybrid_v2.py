from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trkh.core.utils import build_optimizer_param_groups
from trkh.models.dinov3_surface_patch_hybrid_v2 import (
    GENERIC_TOKEN_ADAPTER_ADDED_PARAMETER_COUNT,
    GENERIC_TOKEN_ADAPTER_MODE,
    LOCAL_SURFACE_ADDED_PARAMETER_COUNT,
    LOCAL_SURFACE_MODE,
    DinoV3SurfaceHybridContractError,
    DinoV3SurfacePatchHybridV2,
)


class _FakePatchEmbed(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = (16, 16)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


class _FakeEvaBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.randn(embed_dim) * 0.001)
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del rope, attn_mask, is_causal
        return tokens + torch.tanh(tokens * self.scale + self.bias)


class _FakeDinoBackbone(nn.Module):
    num_features = 384
    embed_dim = 384
    num_prefix_tokens = 5

    def __init__(
        self,
        *,
        num_classes: int = 5,
        global_pool: str = "avg",
        prefix_tokens: int = 5,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.num_features = int(embed_dim)
        self.embed_dim = int(embed_dim)
        self.num_prefix_tokens = int(prefix_tokens)
        self.global_pool = str(global_pool)
        self.patch_embed = _FakePatchEmbed(self.embed_dim)
        self.prefix = nn.Parameter(
            torch.randn(1, self.num_prefix_tokens, self.embed_dim) * 0.01
        )
        self.norm_pre = nn.Identity()
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, int(num_classes))
        self.blocks = nn.ModuleList(
            [_FakeEvaBlock(self.embed_dim) for _ in range(12)]
        )
        self.grad_checkpointing = False
        self.rope_mixed = False

    def _pos_embed(
        self,
        patches: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        prefix = self.prefix.expand(int(patches.size(0)), -1, -1)
        return torch.cat((prefix, patches), dim=1), None

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images)
        tokens, rope = self._pos_embed(patches)
        tokens = self.norm_pre(tokens)
        for block in self.blocks:
            tokens = block(tokens, rope=rope, attn_mask=None, is_causal=False)
        return self.norm(tokens)

    def forward_head(
        self,
        tokens: torch.Tensor,
        pre_logits: bool = False,
    ) -> torch.Tensor:
        pooled = tokens[:, self.num_prefix_tokens :].mean(dim=1)
        return pooled if pre_logits else self.head(pooled)

    def get_classifier(self) -> nn.Module:
        return self.head

    def no_weight_decay(self) -> set[str]:
        return {"prefix"}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = bool(enable)


def _build_model(mode: str = LOCAL_SURFACE_MODE) -> DinoV3SurfacePatchHybridV2:
    return DinoV3SurfacePatchHybridV2(
        _FakeDinoBackbone(),
        num_classes=5,
        mode=mode,
        expected_embed_dim=384,
        expected_prefix_tokens=5,
        expected_patch_count=4,
        initial_gate_scale=0.05,
        max_gate_scale=0.25,
        externally_pretrained=True,
        source_provenance={"checkpoint": {"sha256": "a" * 64}},
    )


def _nonzero_gradient_in_prefix(model: nn.Module, prefix: str) -> bool:
    return any(
        parameter.grad is not None
        and int(torch.count_nonzero(parameter.grad).item()) > 0
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    )


def test_local_surface_shapes_prefix_identity_gate_and_parameter_contract() -> None:
    torch.manual_seed(11)
    model = _build_model(LOCAL_SURFACE_MODE).eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        fused_tokens, trace = model.forward_features_with_fusion_trace(images)
        logits = model.forward_head(fused_tokens)

    assert fused_tokens.shape == (2, 9, 384)
    assert logits.shape == (2, 5)
    assert trace["dino_token_shape"] == [2, 9, 384]
    assert trace["prefix_token_shape"] == [2, 5, 384]
    assert trace["patch_token_shape"] == [2, 4, 384]
    assert trace["gate_shape"] == [2, 4, 1]
    assert trace["surface_descriptor"] == [2, 6, 32, 32]
    assert trace["surface_stage_1"] == [2, 16, 16, 16]
    assert trace["surface_stage_2"] == [2, 24, 8, 8]
    assert trace["surface_stage_3"] == [2, 32, 4, 4]
    assert trace["surface_stage_4"] == [2, 64, 2, 2]
    assert torch.equal(
        trace["injected_tokens"][:, :5],
        trace["dino_prefix_tokens"],
    )
    assert torch.allclose(
        trace["gate"],
        torch.full_like(trace["gate"], 0.05),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.all(trace["gate"] > 0.0)
    assert torch.all(trace["gate"] < 0.25)
    assert torch.count_nonzero(trace["gated_residual"]) == 0
    assert model.added_parameter_count() == LOCAL_SURFACE_ADDED_PARAMETER_COUNT
    telemetry = model.parameter_telemetry()
    assert telemetry["added_parameter_count"] == LOCAL_SURFACE_ADDED_PARAMETER_COUNT
    assert telemetry["total_parameter_count"] == (
        telemetry["backbone_parameter_count"] + LOCAL_SURFACE_ADDED_PARAMETER_COUNT
    )
    assert model.fusion_provenance()["observed_added_parameter_count"] == 29_825
    assert trace["fusion_block_index_zero_based"] == 11
    assert trace["fusion_tail_block_count"] == 1


def test_generic_control_is_capacity_matched_and_uses_no_surface_encoder() -> None:
    model = _build_model(GENERIC_TOKEN_ADAPTER_MODE).eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        fused_tokens, trace = model.forward_features_with_fusion_trace(images)
        logits = model(images)

    assert fused_tokens.shape == (2, 9, 384)
    assert logits.shape == (2, 5)
    assert model.surface_encoder is None
    assert model.token_adapter is not None
    assert "generic_token_context" in trace
    assert "surface_feature_map" not in trace
    assert model.added_parameter_count() == GENERIC_TOKEN_ADAPTER_ADDED_PARAMETER_COUNT
    difference = abs(
        GENERIC_TOKEN_ADAPTER_ADDED_PARAMETER_COUNT
        - LOCAL_SURFACE_ADDED_PARAMETER_COUNT
    )
    assert difference == 165
    assert difference / LOCAL_SURFACE_ADDED_PARAMETER_COUNT < 0.01


def test_surface_encoder_bins_are_exactly_aligned_to_dino_patches() -> None:
    torch.manual_seed(29)
    model = _build_model(LOCAL_SURFACE_MODE).eval()
    assert model.surface_encoder is not None
    with torch.no_grad():
        baseline = model.surface_encoder(torch.zeros(1, 6, 32, 32))
        impulse = torch.zeros(1, 6, 32, 32)
        impulse[:, 0, 17, 1] = 1.0
        observed = model.surface_encoder(impulse) - baseline

    active_cells = torch.nonzero(
        observed.abs().sum(dim=1).flatten() > 0,
        as_tuple=False,
    ).flatten()
    assert active_cells.tolist() == [2]


def test_surface_encoder_has_no_cross_sample_batch_statistics() -> None:
    torch.manual_seed(37)
    model = _build_model(LOCAL_SURFACE_MODE).train()
    assert model.surface_encoder is not None
    target = torch.randn(1, 6, 32, 32)
    distractor = torch.randn(1, 6, 32, 32) * 20.0 + 10.0
    with torch.no_grad():
        alone = model.surface_encoder(target)
        batched = model.surface_encoder(torch.cat((target, distractor), dim=0))[:1]

    assert torch.allclose(alone, batched, atol=2e-5, rtol=2e-5)


def test_architecture_trace_exposes_the_novel_fusion_path() -> None:
    from trkh.tools.trace_architecture import (
        _forward_generic_feature_trace,
        _generic_spatial_activation_map,
    )

    model = _build_model(LOCAL_SURFACE_MODE).eval()
    with torch.no_grad():
        logits, features = _forward_generic_feature_trace(
            model,
            torch.randn(1, 3, 32, 32),
        )

    assert logits.shape == (1, 5)
    assert set(features) == {
        "surface_descriptor",
        "surface_feature_map",
        "pre_fusion_patches",
        "fusion_gate",
        "gated_residual",
        "post_fusion_patches",
    }
    for name in (
        "surface_feature_map",
        "fusion_gate",
        "gated_residual",
        "post_fusion_patches",
    ):
        activation = _generic_spatial_activation_map(features[name])
        assert activation is not None, name
        assert activation.shape == (2, 2)


def test_gate_is_a_true_relative_patch_residual_bound() -> None:
    model = _build_model(LOCAL_SURFACE_MODE).eval()
    assert model.surface_projection is not None
    with torch.no_grad():
        model.surface_projection.weight.normal_(mean=0.0, std=0.02)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        _, trace = model.forward_features_with_fusion_trace(images)

    ratios = trace["gated_residual_norm_ratio"]
    assert torch.all(ratios >= 0.0)
    assert torch.all(ratios < trace["gate"].squeeze(-1))
    assert float(ratios.max()) <= model.max_gate_scale


@pytest.mark.parametrize("mode", (LOCAL_SURFACE_MODE, GENERIC_TOKEN_ADAPTER_MODE))
def test_configured_step_zero_matches_native_backbone_exactly(mode: str) -> None:
    torch.manual_seed(17)
    model = _build_model(mode).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model.backbone.forward_head(
            model.backbone.forward_features(images)
        )
        observed = model(images)
        _, trace = model.forward_features_with_fusion_trace(images)

    assert torch.equal(observed, expected)
    assert torch.count_nonzero(trace["raw_surface_delta"]) == 0
    assert torch.count_nonzero(trace["gated_residual"]) == 0
    assert model.fusion_provenance()["branch_output_initialization"] == (
        "zero_native_dino_identity"
    )


def test_smooth_delta_bound_has_nonzero_gradient_at_zero() -> None:
    model = _build_model(LOCAL_SURFACE_MODE)
    residual = torch.zeros(2, 4, 384, requires_grad=True)
    primary = torch.randn(2, 4, 384)

    bounded = model._normalize_delta(residual, primary)
    bounded.sum().backward()

    expected = primary.norm(dim=-1, keepdim=True).expand_as(residual)
    assert torch.equal(bounded, torch.zeros_like(bounded))
    assert torch.allclose(residual.grad, expected, atol=1e-6, rtol=1e-6)


def test_zero_gate_manual_path_matches_native_backbone_exactly() -> None:
    torch.manual_seed(19)
    model = _build_model(LOCAL_SURFACE_MODE).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        model.fusion_gate.bias.fill_(-1e9)
        expected_tokens = model.backbone.forward_features(images)
        expected = model.backbone.forward_head(expected_tokens)
        observed = model(images)

    assert torch.equal(observed, expected)


def test_task_modules_do_not_shift_global_training_rng() -> None:
    outputs = {}
    for name, mode in (
        ("direct", None),
        ("generic", GENERIC_TOKEN_ADAPTER_MODE),
        ("local", LOCAL_SURFACE_MODE),
    ):
        torch.manual_seed(67)
        backbone = _FakeDinoBackbone()
        if mode is not None:
            DinoV3SurfacePatchHybridV2(
                backbone,
                num_classes=5,
                mode=mode,
                expected_patch_count=4,
            )
        outputs[name] = torch.rand(16)

    assert torch.equal(outputs["direct"], outputs["generic"])
    assert torch.equal(outputs["direct"], outputs["local"])


@pytest.mark.parametrize(
    "mode,first_step_prefixes,delayed_prefixes",
    (
        (
            LOCAL_SURFACE_MODE,
            (
                "backbone.patch_embed.",
                "backbone.blocks.11.",
                "backbone.head.",
                "surface_projection.",
            ),
            ("surface_encoder.", "surface_norm.", "fusion_gate."),
        ),
        (
            GENERIC_TOKEN_ADAPTER_MODE,
            (
                "backbone.patch_embed.",
                "backbone.blocks.11.",
                "backbone.head.",
                "token_adapter.fc2.",
            ),
            ("token_adapter.norm.", "token_adapter.fc1.", "fusion_gate."),
        ),
    ),
)
def test_identity_initialization_opens_branch_without_random_injection(
    mode: str,
    first_step_prefixes: tuple[str, ...],
    delayed_prefixes: tuple[str, ...],
) -> None:
    torch.manual_seed(23)
    model = _build_model(mode).train()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    loss = F.cross_entropy(model(images), labels)
    loss.backward()

    for prefix in first_step_prefixes:
        assert _nonzero_gradient_in_prefix(model, prefix), prefix
    for prefix in delayed_prefixes:
        assert not _nonzero_gradient_in_prefix(model, prefix), prefix

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), labels).backward()
    for prefix in delayed_prefixes:
        assert _nonzero_gradient_in_prefix(model, prefix), prefix


def test_gate_context_does_not_add_a_jacobian_into_primary_patch_tokens() -> None:
    torch.manual_seed(31)
    model = _build_model(LOCAL_SURFACE_MODE).eval()
    images = torch.randn(2, 3, 32, 32)
    _, trace = model.forward_features_with_fusion_trace(images)
    patch_tokens = trace["dino_patch_tokens"]
    gradient = torch.autograd.grad(
        trace["fused_patch_tokens"].sum(),
        patch_tokens,
        retain_graph=True,
    )[0]

    assert trace["primary_gate_context_detached"] is True
    assert torch.equal(gradient, torch.ones_like(gradient))


def test_primary_backbone_is_unchanged_by_wrapper_construction() -> None:
    torch.manual_seed(41)
    backbone = _FakeDinoBackbone()
    before = copy.deepcopy(backbone.state_dict())
    model = DinoV3SurfacePatchHybridV2(
        backbone,
        num_classes=5,
        mode=LOCAL_SURFACE_MODE,
        expected_patch_count=4,
    )

    assert before.keys() == model.backbone.state_dict().keys()
    for key, expected in before.items():
        assert torch.equal(model.backbone.state_dict()[key], expected), key


def test_strict_state_roundtrip_preserves_logits_and_provenance_contract() -> None:
    torch.manual_seed(53)
    source = _build_model(LOCAL_SURFACE_MODE).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = source(images)

    restored = _build_model(LOCAL_SURFACE_MODE).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        observed = restored(images)

    assert torch.equal(observed, expected)
    assert (
        restored.pretrained_provenance["primary_backbone"]["checkpoint"]["sha256"]
        == "a" * 64
    )
    assert restored.pretrained_provenance["external_initialization_used"] is True


def test_timm_and_optimizer_metadata_are_explicit() -> None:
    model = _build_model(LOCAL_SURFACE_MODE)

    assert model.is_timm_classifier is True
    assert model.is_pretrained_timm_classifier is True
    assert model.uses_timm_backbone_lr_split is True
    assert model.pretrained_classifier_parameter_prefixes == ("backbone.head.",)
    assert model.pretrained_task_parameter_prefixes == (
        "surface_encoder.",
        "surface_norm.",
        "surface_projection.",
        "fusion_gate.",
    )
    assert model.no_weight_decay() == {
        "backbone.prefix",
        "fusion_gate.weight",
        "fusion_gate.bias",
    }
    model.set_grad_checkpointing(True)
    assert model.backbone.grad_checkpointing is True


def test_gradient_checkpointing_accepts_eva_keyword_contract_and_backpropagates() -> None:
    model = _build_model(LOCAL_SURFACE_MODE).train()
    assert model.surface_projection is not None
    with torch.no_grad():
        model.surface_projection.weight.normal_(mean=0.0, std=1e-3)
    model.set_grad_checkpointing(True)
    images = torch.randn(2, 3, 32, 32)
    loss = model(images).square().mean()
    loss.backward()

    assert _nonzero_gradient_in_prefix(model, "backbone.blocks.11.")
    assert _nonzero_gradient_in_prefix(model, "surface_encoder.")


@pytest.mark.parametrize("externally_pretrained", (False, True))
def test_optimizer_contract_is_matched_for_pretrained_and_random_control(
    externally_pretrained: bool,
) -> None:
    model = DinoV3SurfacePatchHybridV2(
        _FakeDinoBackbone(),
        num_classes=5,
        mode=LOCAL_SURFACE_MODE,
        expected_patch_count=4,
        externally_pretrained=externally_pretrained,
    )
    groups = build_optimizer_param_groups(
        model,
        weight_decay=0.05,
        learning_rate=1.5e-4,
        backbone_lr_scale=0.1,
    )
    group_by_parameter_id = {
        id(parameter): group
        for group in groups
        for parameter in group["params"]
    }

    for name, parameter in model.named_parameters():
        group = group_by_parameter_id[id(parameter)]
        if name.startswith("backbone.") and not name.startswith("backbone.head."):
            assert str(group["name"]).startswith("backbone_")
            assert group["lr"] == pytest.approx(1.5e-5)
        else:
            assert str(group["name"]).startswith("head_")
            assert "lr" not in group


def test_create_model_wires_hybrid_v2_without_replaying_external_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trkh.models import model as model_module

    observed: dict[str, object] = {}

    def fake_timm_builder(**kwargs: object) -> nn.Module:
        observed.update(kwargs)
        backbone = _FakeDinoBackbone()
        backbone.is_pretrained_timm_classifier = False
        backbone.pretrained_provenance = {
            "initialization_source": "random_control"
        }
        return backbone

    monkeypatch.setattr(model_module, "_build_timm_classifier", fake_timm_builder)
    model = model_module.create_model(
        num_classes=5,
        model_config={
            "model_type": "dinov3_surface_patch_hybrid_v2",
            "research_track": "pretrained",
            "pretrained": False,
            "timm_model_name": "vit_small_patch16_dinov3.lvd1689m",
            "image_size": 256,
            "dinov3_surface_hybrid_mode": "local_surface",
            "dinov3_surface_initial_gate_scale": 0.05,
            "dinov3_surface_max_gate_scale": 0.25,
            "input_mean": (0.1, 0.2, 0.3),
            "input_std": (0.4, 0.5, 0.6),
        },
    ).eval()

    assert model.model_type == "dinov3_surface_patch_hybrid_v2"
    assert model.research_track == "pretrained"
    assert model.pretrained_provenance["external_initialization_used"] is False
    assert observed["pretrained"] is False
    assert observed["architecture_only_checkpoint_rebuild"] is False
    assert model.input_mean.flatten().tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert model.input_std.flatten().tolist() == pytest.approx([0.4, 0.5, 0.6])
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 256, 256)).shape == (1, 5)


def test_random_init_checkpoint_rebuild_restores_false_external_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trkh.models import model as model_module

    torch.manual_seed(73)
    source = DinoV3SurfacePatchHybridV2(
        _FakeDinoBackbone(),
        num_classes=5,
        mode=LOCAL_SURFACE_MODE,
        expected_patch_count=256,
        externally_pretrained=False,
        source_provenance={"initialization_source": "random_control"},
    ).eval()

    def fake_timm_builder(**kwargs: object) -> nn.Module:
        assert kwargs["architecture_only_checkpoint_rebuild"] is True
        backbone = _FakeDinoBackbone()
        # Architecture-only reconstruction is provisionally marked pretrained;
        # serialized provenance must correct this after strict state loading.
        backbone.is_pretrained_timm_classifier = True
        backbone.pretrained_provenance = {"rebuild": True}
        return backbone

    monkeypatch.setattr(model_module, "_build_timm_classifier", fake_timm_builder)
    checkpoint_payload = {
        "class_names": [f"class_{index}" for index in range(5)],
        "model_config": {
            "model_type": "dinov3_surface_patch_hybrid_v2",
            "research_track": "pretrained",
            "pretrained": False,
            "timm_model_name": "vit_small_patch16_dinov3.lvd1689m",
            "image_size": 256,
            "dinov3_surface_hybrid_mode": "local_surface",
            "dinov3_surface_initial_gate_scale": 0.05,
            "dinov3_surface_max_gate_scale": 0.25,
        },
        "model_state": source.state_dict(),
        "pretrained_provenance": copy.deepcopy(source.pretrained_provenance),
    }
    restored = model_module.build_model_from_checkpoint(checkpoint_payload).eval()
    images = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        expected = source(images)
        observed = restored(images)

    assert torch.equal(observed, expected)
    assert restored.is_pretrained_timm_classifier is False
    assert restored.uses_timm_backbone_lr_split is True
    assert restored.pretrained_provenance["external_initialization_used"] is False
    assert restored.pretrained_provenance["external_initialization_replayed"] is False


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"mode": "unknown"}, "mode must be one of"),
        ({"expected_embed_dim": 192}, "embedding dimension changed"),
        ({"expected_prefix_tokens": 4}, "prefix-token count changed"),
        ({"initial_gate_scale": 0.0}, "strictly inside"),
        ({"initial_gate_scale": 0.25}, "strictly inside"),
    ),
)
def test_constructor_fails_closed(kwargs: dict[str, object], match: str) -> None:
    arguments = {
        "num_classes": 5,
        "expected_embed_dim": 384,
        "expected_prefix_tokens": 5,
        "expected_patch_count": 4,
        **kwargs,
    }
    with pytest.raises((ValueError, DinoV3SurfaceHybridContractError), match=match):
        DinoV3SurfacePatchHybridV2(_FakeDinoBackbone(), **arguments)


def test_constructor_rejects_non_average_pooling_and_wrong_classifier() -> None:
    with pytest.raises(DinoV3SurfaceHybridContractError, match="average patch pooling"):
        DinoV3SurfacePatchHybridV2(
            _FakeDinoBackbone(global_pool="token"),
            num_classes=5,
            expected_patch_count=4,
        )
    with pytest.raises(DinoV3SurfaceHybridContractError, match="class count changed"):
        DinoV3SurfacePatchHybridV2(
            _FakeDinoBackbone(num_classes=4),
            num_classes=5,
            expected_patch_count=4,
        )
    changed_patch_size = _FakeDinoBackbone()
    changed_patch_size.patch_embed.patch_size = (14, 14)
    with pytest.raises(DinoV3SurfaceHybridContractError, match="patch size changed"):
        DinoV3SurfacePatchHybridV2(
            changed_patch_size,
            num_classes=5,
            expected_patch_count=4,
        )
    changed_depth = _FakeDinoBackbone()
    changed_depth.blocks = nn.ModuleList(list(changed_depth.blocks)[:-1])
    with pytest.raises(DinoV3SurfaceHybridContractError, match="block count changed"):
        DinoV3SurfacePatchHybridV2(
            changed_depth,
            num_classes=5,
            expected_patch_count=4,
        )


def test_runtime_fails_closed_on_patch_count_and_input_geometry() -> None:
    model = _build_model(LOCAL_SURFACE_MODE)
    with pytest.raises(DinoV3SurfaceHybridContractError, match="patch count changed"):
        model(torch.randn(1, 3, 48, 48))
    with pytest.raises(DinoV3SurfaceHybridContractError, match="divisible by 16"):
        model(torch.randn(1, 3, 31, 32))
    with pytest.raises(DinoV3SurfaceHybridContractError, match=r"\[B,3,H,W\]"):
        model(torch.randn(1, 1, 32, 32))

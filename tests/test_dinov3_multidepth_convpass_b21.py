from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from trkh.core.utils import build_optimizer_param_groups
from trkh.models.dinov3_multidepth_convpass_b21 import (
    DEPHASED_MODE,
    DIRECT_MODE,
    SPATIAL_MODE,
    DinoV3MultiDepthConvPassB21,
    expected_added_parameter_count,
)


class _PatchEmbed(nn.Module):
    patch_size = (16, 16)

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del rope, attn_mask, is_causal
        return torch.tanh(self.proj(tokens))


class _EvaBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim)
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop_path2 = nn.Identity()
        self.gamma_1 = nn.Parameter(torch.full((dim,), 0.8))
        self.gamma_2 = nn.Parameter(torch.full((dim,), 0.7))

    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        tokens = tokens + self.drop_path1(
            self.gamma_1
            * self.attn(
                self.norm1(tokens),
                rope=rope,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        )
        return tokens + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(tokens)))


class _FakeDino(nn.Module):
    global_pool = "avg"
    rope_mixed = False
    grad_checkpointing = False

    def __init__(self, dim: int = 16, prefix_tokens: int = 2, blocks: int = 2) -> None:
        super().__init__()
        self.num_features = dim
        self.embed_dim = dim
        self.num_prefix_tokens = prefix_tokens
        self.patch_embed = _PatchEmbed(dim)
        self.prefix = nn.Parameter(torch.randn(1, prefix_tokens, dim) * 0.01)
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([_EvaBlock(dim) for _ in range(blocks)])
        self.norm = nn.LayerNorm(dim)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Identity()
        self.head = nn.Linear(dim, 5)

    def _pos_embed(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prefix = self.prefix.expand(patches.size(0), -1, -1)
        return torch.cat((prefix, patches), dim=1), None

    def forward_features(
        self,
        images: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        tokens, rope = self._pos_embed(self.patch_embed(images))
        tokens = self.norm_pre(tokens)
        for block in self.blocks:
            tokens = block(
                tokens,
                rope=rope,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        return self.norm(tokens)

    def forward_head(
        self, tokens: torch.Tensor, pre_logits: bool = False
    ) -> torch.Tensor:
        pooled = tokens[:, self.num_prefix_tokens :].mean(dim=1)
        return pooled if pre_logits else self.head(pooled)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(images))

    def get_classifier(self) -> nn.Module:
        return self.head

    def no_weight_decay(self) -> set[str]:
        return {"prefix"}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = bool(enable)


def _build(
    mode: str = SPATIAL_MODE,
    *,
    backbone: _FakeDino | None = None,
) -> DinoV3MultiDepthConvPassB21:
    return DinoV3MultiDepthConvPassB21(
        backbone or _FakeDino(),
        num_classes=5,
        mode=mode,
        expected_embed_dim=16,
        expected_prefix_tokens=2,
        expected_patch_count=4,
        expected_block_count=2,
        bottleneck_dim=4,
        adapter_scale=1.0,
        adapter_seed=99,
        source_provenance={"checkpoint": {"sha256": "a" * 64}},
    )


def _adapter_parameters(model: nn.Module, suffix: str = ""):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if (".adapter_attn." in name or ".adapter_mlp." in name)
        and (not suffix or name.endswith(suffix))
    ]


def test_construction_does_not_advance_global_rng_and_count_is_locked() -> None:
    torch.manual_seed(17)
    backbone = _FakeDino()
    before = torch.random.get_rng_state().clone()
    model = _build(backbone=backbone)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)
    assert model.added_parameter_count() == expected_added_parameter_count(
        block_count=2,
        embed_dim=16,
        bottleneck_dim=4,
    )
    assert model.fusion_provenance()["private_stochastic_operations"] is False


def test_construction_does_not_advance_cuda_rng_when_cuda_is_initialized() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed_all(71)
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    _build(backbone=_FakeDino())
    after = torch.cuda.get_rng_state_all()
    assert len(before) == len(after)
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_step_zero_and_direct_mode_are_bit_exact_native_dino() -> None:
    torch.manual_seed(23)
    backbone = _FakeDino()
    native = copy.deepcopy(backbone).eval()
    spatial = _build(SPATIAL_MODE, backbone=backbone).eval()
    images = torch.randn(3, 3, 32, 32)

    with torch.no_grad():
        expected = native(images)
        observed = spatial(images)
        spatial.set_mode(DIRECT_MODE)
        direct = spatial(images)

    assert torch.equal(observed, expected)
    assert torch.equal(direct, expected)
    assert all(not parameter.requires_grad for parameter in _adapter_parameters(spatial))


def test_zero_up_first_step_then_gradients_reach_conv_and_down() -> None:
    torch.manual_seed(29)
    model = _build(SPATIAL_MODE).train()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    first_loss = F.cross_entropy(model(images), labels)
    first_loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in _adapter_parameters(model, "up.weight")
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in (
            _adapter_parameters(model, "down.weight")
            + _adapter_parameters(model, "conv.weight")
        )
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = F.cross_entropy(model(images), labels)
    second_loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in _adapter_parameters(model, "down.weight")
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in _adapter_parameters(model, "conv.weight")
    )


def test_dephased_control_has_identical_state_but_different_neighbour_graph() -> None:
    torch.manual_seed(31)
    spatial = _build(SPATIAL_MODE).eval()
    dephased = copy.deepcopy(spatial).eval()
    dephased.set_mode(DEPHASED_MODE)
    for model in (spatial, dephased):
        with torch.no_grad():
            for block in model.blocks:
                for adapter in (block.adapter_attn, block.adapter_mlp):
                    adapter.conv.weight.normal_(mean=0.0, std=0.08)
                    adapter.up.weight.normal_(mean=0.0, std=0.08)
    # Restore bit-identical parameters after the deterministic mutations above.
    dephased.load_state_dict(spatial.state_dict())
    dephased.set_mode(DEPHASED_MODE)
    assert spatial.added_parameter_count() == dephased.added_parameter_count()
    assert all(
        torch.equal(left, right)
        for left, right in zip(spatial.state_dict().values(), dephased.state_dict().values())
    )

    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        spatial_logits = spatial(images)
        dephased_logits = dephased(images)
    assert not torch.equal(spatial_logits, dephased_logits)


def test_optimizer_places_only_head_and_adapters_at_task_learning_rate() -> None:
    model = _build(SPATIAL_MODE)
    groups = build_optimizer_param_groups(
        model,
        weight_decay=0.05,
        learning_rate=1.5e-4,
        backbone_lr_scale=0.1,
    )
    group_by_parameter = {
        id(parameter): str(group["name"])
        for group in groups
        for parameter in group["params"]
    }
    for name, parameter in model.named_parameters():
        if ".adapter_attn." in name or ".adapter_mlp." in name or name.startswith(
            "backbone.head."
        ):
            assert group_by_parameter[id(parameter)].startswith("head_")
        else:
            assert group_by_parameter[id(parameter)].startswith("backbone_")


def test_trace_reports_both_multidepth_paths_after_activation() -> None:
    model = _build(SPATIAL_MODE).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.adapter_attn.up.weight.normal_(mean=0.0, std=0.02)
            block.adapter_mlp.up.weight.normal_(mean=0.0, std=0.02)
        logits, trace = model.forward_with_adapter_trace(
            torch.randn(2, 3, 32, 32)
        )

    assert logits.shape == (2, 5)
    assert trace["attention_residual_ratio"].numel() == 2 * 2 * 4
    assert trace["mlp_residual_ratio"].numel() == 2 * 2 * 4
    assert torch.isfinite(trace["all_residual_ratio"]).all()
    assert torch.count_nonzero(trace["all_residual_ratio"]) > 0

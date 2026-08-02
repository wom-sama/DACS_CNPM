from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import trkh.tools.precheck_dinov3_xcnorm_a1_sourcefold as precheck
from trkh.tools.precheck_dinov3_xcnorm_a1_sourcefold import (
    ADAPTER_BATCH_SIZE,
    build_paired_adapters,
    decompose_final_mhsa,
    frozen_tail_logits,
    run_actual_b9_train_smoke,
    run_precheck,
    select_fixed_smoke_indices,
    tail_adjusted_logits,
    _runtime_contract,
)


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
        return self.value * value


class _ToyAttention(nn.Module):
    def __init__(self, scale: float = 0.25) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(
        self,
        value: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del attn_mask, is_causal
        if rope is None:
            raise RuntimeError("toy attention requires the RoPE object")
        return self.scale * value


class _ToyFinalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(384)
        self.attn = _ToyAttention()
        self.drop_path1 = nn.Identity()
        self.gamma_1 = nn.Parameter(torch.full((384,), 0.2))
        self.norm2 = nn.LayerNorm(384)
        self.mlp = nn.Sequential(
            nn.Linear(384, 32), nn.GELU(), nn.Linear(32, 384)
        )
        self.drop_path2 = nn.Identity()
        self.gamma_2 = nn.Parameter(torch.full((384,), 0.3))

    def forward(
        self,
        value: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        value = value + self.drop_path1(
            self.gamma_1
            * self.attn(
                self.norm1(value),
                rope=rope,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        )
        return value + self.drop_path2(
            self.gamma_2 * self.mlp(self.norm2(value))
        )


class _PassBlock(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del rope, attn_mask, is_causal
        return value


def _classifier_weight() -> torch.Tensor:
    weight = torch.zeros(5, 384)
    weight[0, 0] = -1.0
    weight[2, :2] = torch.tensor((-0.7, -1.0))
    weight[4, :3] = torch.tensor((-0.4, -0.6, -1.0))
    weight[3, 3] = 0.5
    return weight


class _ToyEva(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Identity()
        self.patch_embed.grid_size = (16, 16)
        self.norm_pre = nn.Identity()
        self.rope = nn.Identity()
        self.blocks = nn.ModuleList(
            [_PassBlock() for _ in range(11)] + [_ToyFinalBlock()]
        )
        self.norm = nn.LayerNorm(384)
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(p=0.0)
        self.head = nn.Linear(384, 5, bias=False)
        self.num_prefix_tokens = 5
        self.embed_dim = 384
        self.global_pool = "avg"
        self.dynamic_img_size = True
        self.rope_mixed = False
        with torch.no_grad():
            self.head.weight.copy_(_classifier_weight())

    def _pos_embed(self, value: torch.Tensor):
        return value, torch.ones(1, device=value.device, dtype=value.dtype)

    def forward_head(
        self, value: torch.Tensor, pre_logits: bool = False
    ) -> torch.Tensor:
        pooled = value[:, self.num_prefix_tokens :].mean(dim=1)
        return pooled if pre_logits else self.head(pooled)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.patch_embed(value)
        value, rope = self._pos_embed(value)
        value = self.norm_pre(value)
        for block in self.blocks:
            value = block(value, rope=rope)
        return self.forward_head(self.norm(value), pre_logits=False)


def test_gamma1_and_gamma2_formulas_are_exact() -> None:
    torch.manual_seed(3)
    block = _ToyFinalBlock().eval()
    x_pre = torch.randn(2, 261, 384)
    rope = torch.ones(1)
    z, u = decompose_final_mhsa(block, x_pre, rope)
    expected_z = block.norm1(x_pre)
    expected_u = x_pre + block.gamma_1 * block.attn(expected_z, rope=rope)
    assert torch.equal(z, expected_z)
    assert torch.equal(u, expected_u)

    u_patch = u[:, 5:]
    residual = 0.01 * torch.randn_like(u_patch)
    logits, final_patch = frozen_tail_logits(
        block, nn.Identity(), nn.Identity(), u_patch, residual
    )
    v = u_patch + residual
    expected_final = v + block.gamma_2 * block.mlp(block.norm2(v))
    assert torch.equal(final_patch, expected_final)
    assert torch.equal(logits, expected_final.mean(dim=1))


def test_cached_base_delta_makes_zero_tail_bit_exact() -> None:
    torch.manual_seed(5)
    block = _ToyFinalBlock().eval()
    final_norm = nn.LayerNorm(384).eval()
    head = nn.Linear(384, 5).eval()
    u = torch.randn(3, 256, 384)
    cached = torch.randn(3, 5)
    adjusted, trace = tail_adjusted_logits(
        block, final_norm, head, u, torch.zeros_like(u), cached
    )
    assert torch.equal(adjusted, cached)
    assert torch.equal(
        trace["active_tail_logits"], trace["base_tail_logits"]
    )


def test_paired_a1_capacity_and_state_are_identical() -> None:
    weight = _classifier_weight().requires_grad_(False)
    control, candidate, contract = build_paired_adapters(weight, seed=17)
    assert contract["paired_initial_state_sha256"]
    assert contract["control_parameters"] == 3_672
    assert contract["candidate_parameters"] == 3_672
    assert all(
        torch.equal(control.state_dict()[key], candidate.state_dict()[key])
        for key in control.state_dict()
    )


def test_fixed_train_smoke_selection_is_one_per_class_and_hashed() -> None:
    paths = [
        "train/c0/z.jpg",
        "train/c0/a.jpg",
        "train/c1/b.jpg",
        "train/c2/c.jpg",
        "train/c3/d.jpg",
        "train/c4/e.jpg",
    ]
    labels = [0, 0, 1, 2, 3, 4]
    indices1, selected1, digest1 = select_fixed_smoke_indices(
        paths, labels, expected_paths_sha256=None
    )
    indices2, selected2, digest2 = select_fixed_smoke_indices(
        paths, labels, expected_paths_sha256=None
    )
    assert indices1 == indices2
    assert selected1 == selected2
    assert digest1 == digest2 and len(digest1) == 64
    assert [labels[index] for index in indices1] == list(range(5))


def test_actual_smoke_proves_parity_and_adapter_only_gradient(monkeypatch) -> None:
    locked = {
        "selected_full_state_sha256": "a" * 64,
        "final_block_state_sha256": "b" * 64,
        "gamma1_state_sha256": "c" * 64,
        "gamma2_state_sha256": "d" * 64,
        "final_norm_head_state_sha256": "e" * 64,
    }
    monkeypatch.setattr(precheck, "_state_subset_hashes", lambda _state: dict(locked))
    monkeypatch.setattr(precheck, "_assert_locked_state_hashes", lambda _hashes: None)
    torch.manual_seed(7)
    model = _ToyEva().eval().requires_grad_(False)
    images = torch.randn(5, 261, 384)
    labels = torch.arange(5, dtype=torch.long)
    result = run_actual_b9_train_smoke(
        model=model,
        images=images,
        labels=labels,
        relative_paths=[f"train/c{index}/x.jpg" for index in range(5)],
    )
    assert result["actual_b9_train_smoke_passed"]
    assert result["state_hashes_before"] == result["state_hashes_after"]
    assert result["parity"]["post_mhsa_u_max_abs_error"] == 0.0
    assert result["parity"]["branch_off_logit_max_abs_error"] == 0.0
    assert result["residual_gradient_through_frozen_tail_norm"] > 0.0
    assert not result["frozen_model_gradients_present"]
    assert all(value > 0.0 for value in result["adapter_gradient_norms"].values())


def test_actual_smoke_rejects_a_self_consistent_but_wrong_extractor(
    monkeypatch,
) -> None:
    locked = {
        "selected_full_state_sha256": "a" * 64,
        "final_block_state_sha256": "b" * 64,
        "gamma1_state_sha256": "c" * 64,
        "gamma2_state_sha256": "d" * 64,
        "final_norm_head_state_sha256": "e" * 64,
    }
    monkeypatch.setattr(precheck, "_state_subset_hashes", lambda _state: dict(locked))
    monkeypatch.setattr(precheck, "_assert_locked_state_hashes", lambda _hashes: None)
    native_extractor = precheck.eva_pre_final_tokens

    def corrupted_extractor(model, images):
        value, rope = native_extractor(model, images)
        value = value.clone()
        value[..., 0] += 0.125
        return value, rope

    monkeypatch.setattr(precheck, "eva_pre_final_tokens", corrupted_extractor)
    model = _ToyEva().eval().requires_grad_(False)
    with pytest.raises(RuntimeError, match="native_direct_logits"):
        run_actual_b9_train_smoke(
            model=model,
            images=torch.randn(5, 261, 384),
            labels=torch.arange(5, dtype=torch.long),
            relative_paths=[f"train/c{index}/x.jpg" for index in range(5)],
        )


def test_batch_lock_and_preflight_only_fail_before_io() -> None:
    with pytest.raises(ValueError, match=f"locked to {ADAPTER_BATCH_SIZE}"):
        run_precheck(
            SimpleNamespace(
                adapter_batch_size=64,
                preflight_only=True,
                workers=0,
            )
        )
    with pytest.raises(RuntimeError, match="--preflight-only"):
        run_precheck(
            SimpleNamespace(
                adapter_batch_size=ADAPTER_BATCH_SIZE,
                preflight_only=False,
                workers=0,
            )
        )


def test_cublas_workspace_is_configured_before_runtime() -> None:
    assert precheck.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_runtime_contract_records_exact_eva_integration_surface(
    monkeypatch,
) -> None:
    monkeypatch.setattr(precheck, "package_version", lambda name: "locked-timm")
    model = _ToyEva().eval().requires_grad_(False)
    contract = _runtime_contract(model)
    assert contract["timm_version"] == "locked-timm"
    assert contract["model_training"] is False
    assert contract["model_parameters_frozen"] is True
    assert contract["patch_grid_size"] == [16, 16]
    assert contract["num_prefix_tokens"] == 5
    assert contract["rope_mixed"] is False
    assert contract["final_block_index"] == 11
    assert contract["attention_class"] == "_ToyAttention"
    assert contract["drop_path1_class"] == "Identity"
    assert contract["drop_path2_class"] == "Identity"
    assert contract["head_shape"] == [5, 384]

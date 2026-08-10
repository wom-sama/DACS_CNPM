from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Deterministic cuBLAS must be configured before the first CUDA operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from trkh.models.dinov3_xcnorm_pair_adapter_a1 import (
    LOCAL_CONV_A1_MODE,
    LOCAL_XCNORM_A1_MODE,
    XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT,
    XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT,
    DinoV3XCNormPairAdapterA1,
)
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.export_timm_predictions import _state_dict_sha256
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (
    ADAPTER_BATCH_SIZE,
    EPOCHS,
    EXPECTED_ASSIGNMENT_INT64_SHA256,
    EXPECTED_CLASSES,
    EXPECTED_DATA_SHA256,
    EXPECTED_HELD_CLASS_COUNTS,
    EXPECTED_INTEGRITY_MANIFEST_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PATH_FOLD_SHA256,
    EXPECTED_SELECTED_STATE_SHA256,
    EXPECTED_TRAIN_SAMPLES,
    FOLDS,
    LR,
    SEED,
    TEMPERED_POWER,
    WEIGHT_DECAY,
    _data_root_from_yaml,
    _json_sha256,
    _read_integrity_train_rows,
    _read_train_rows,
    _relative_train_path,
    _sha256,
    _validate_b9_checkpoint,
    assert_train_only_paths,
    assign_locked_folds,
    build_train_union_groups,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_XCNORM_A1_SOURCEFOLD_PREFLIGHT_20260802"
EXPECTED_TIMM_VERSION = "1.0.27"
FINAL_BLOCK_INDEX = 11
FINAL_BLOCK_COUNT = 12
PATCH_COUNT = 256
EMBED_DIM = 384
SMOKE_BATCH_SIZE = 5
EXPECTED_SMOKE_PATHS_SHA256 = (
    "1b9c22bcbb40b6379de355d69f9a21ec33e2a84ebaf0752da15a3bc30c80b842"
)

# These hashes are over the exact selected EMA state using
# export_timm_predictions._state_dict_sha256 and original state-dict names.
EXPECTED_FINAL_BLOCK_STATE_SHA256 = (
    "adb3d5b2314c9667c5956cb40f1b815ea3d3f81f0c265fc6fcccf31bb96b415d"
)
EXPECTED_GAMMA1_STATE_SHA256 = (
    "1c8dbbc9b7230d6f00a927fa3f80d3f575f514be514bfa9ed84797621bb88196"
)
EXPECTED_GAMMA2_STATE_SHA256 = (
    "5b82320031483379d1ac71980a48342012cfa66d2f8268d2b8708bd43035bbde"
)
EXPECTED_FINAL_NORM_HEAD_STATE_SHA256 = (
    "c9954140962809418953b9aaa5672ba7b7c6a2aaae1d5e1cb47620368d5ea0d2"
)

STRUCTURAL_PARITY_ATOL = 1e-6
# The frozen token-wise MLP is evaluated on 256 cached patches rather than the
# native 261-token tensor. Equivalent FP32 GEMMs can differ by a few ulps when
# their flattened leading dimension changes; cached-base delta correction still
# keeps branch-off logits bit exact.
PATCH_TAIL_PARITY_ATOL = 1e-5


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict TRAIN-only integration preflight for the single locked "
            "DINOv3 parallel-final-MHSA XCNorm A1. This stage deliberately "
            "does not create the large token cache or run OOF training."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-integrity-manifest", type=Path, required=True)
    parser.add_argument(
        "--adapter-batch-size",
        type=int,
        default=ADAPTER_BATCH_SIZE,
        help=f"future A1 OOF batch lock; must remain {ADAPTER_BATCH_SIZE}",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"stale partial output exists: {partial}")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _state_subset_hashes(state: Mapping[str, Tensor]) -> Dict[str, str]:
    subsets = {
        "selected_full_state_sha256": dict(state),
        "final_block_state_sha256": {
            key: value
            for key, value in state.items()
            if str(key).startswith(f"blocks.{FINAL_BLOCK_INDEX}.")
        },
        "gamma1_state_sha256": {
            key: value
            for key, value in state.items()
            if str(key) == f"blocks.{FINAL_BLOCK_INDEX}.gamma_1"
        },
        "gamma2_state_sha256": {
            key: value
            for key, value in state.items()
            if str(key) == f"blocks.{FINAL_BLOCK_INDEX}.gamma_2"
        },
        "final_norm_head_state_sha256": {
            key: value
            for key, value in state.items()
            if str(key).startswith("norm.") or str(key).startswith("head.")
        },
    }
    expected_sizes = {
        "final_block_state_sha256": 13,
        "gamma1_state_sha256": 1,
        "gamma2_state_sha256": 1,
        "final_norm_head_state_sha256": 4,
    }
    for name, size in expected_sizes.items():
        if len(subsets[name]) != size:
            raise ValueError(f"locked B9 state subset {name} has changed")
    return {name: _state_dict_sha256(values) for name, values in subsets.items()}


def _assert_locked_state_hashes(hashes: Mapping[str, str]) -> None:
    expected = {
        "selected_full_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        "final_block_state_sha256": EXPECTED_FINAL_BLOCK_STATE_SHA256,
        "gamma1_state_sha256": EXPECTED_GAMMA1_STATE_SHA256,
        "gamma2_state_sha256": EXPECTED_GAMMA2_STATE_SHA256,
        "final_norm_head_state_sha256": EXPECTED_FINAL_NORM_HEAD_STATE_SHA256,
    }
    mismatch = {
        key: {"expected": value, "observed": hashes.get(key)}
        for key, value in expected.items()
        if hashes.get(key) != value
    }
    if mismatch:
        raise ValueError(f"actual B9 final-block state hash mismatch: {mismatch}")


def build_paired_adapters(
    classifier_weight: Tensor, *, seed: int = SEED
) -> Tuple[DinoV3XCNormPairAdapterA1, DinoV3XCNormPairAdapterA1, Dict[str, object]]:
    if classifier_weight.requires_grad:
        raise ValueError("classifier weight must be frozen before A1 binding")
    torch.manual_seed(int(seed))
    control = DinoV3XCNormPairAdapterA1(LOCAL_CONV_A1_MODE)
    control.bind_classifier_weight(classifier_weight)
    candidate = DinoV3XCNormPairAdapterA1(LOCAL_XCNORM_A1_MODE)
    candidate.load_state_dict(control.state_dict(), strict=True)
    candidate.bind_classifier_weight(classifier_weight)
    control_hash = _state_dict_sha256(control.state_dict())
    candidate_hash = _state_dict_sha256(candidate.state_dict())
    if control_hash != candidate_hash:
        raise RuntimeError("paired A1 arms did not receive identical state")
    for module in (control, candidate):
        if module.added_parameter_count(trainable_only=True) != 3_672:
            raise RuntimeError("A1 adapter capacity changed from 3,672")
        if not module.classifier_binding_matches(classifier_weight):
            raise RuntimeError("A1 classifier-axis binding mismatch")
    return control, candidate, {
        "paired_initial_state_sha256": control_hash,
        "control_parameters": control.added_parameter_count(),
        "candidate_parameters": candidate.added_parameter_count(),
        "expected_parameters": XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT,
        "classifier_weight_requires_grad": False,
    }


def _rope_for_block(
    model: nn.Module, rotary: Optional[Tensor], block_index: int
) -> Optional[Tensor]:
    if rotary is None:
        return None
    if bool(getattr(model, "rope_mixed", False)):
        return rotary[int(block_index)]
    return rotary


def eva_pre_final_tokens(
    model: nn.Module, images: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    """Run exact native EVA preprocessing and blocks 0..10."""

    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != FINAL_BLOCK_COUNT:
        raise ValueError("A1 is locked to the 12-block B9 EVA backbone")
    x = model.patch_embed(images)
    x, rotary = model._pos_embed(x)
    x = model.norm_pre(x)
    for index, block in enumerate(blocks[:-1]):
        x = block(x, rope=_rope_for_block(model, rotary, index))
    return x, _rope_for_block(model, rotary, FINAL_BLOCK_INDEX)


def decompose_final_mhsa(
    block: nn.Module, x_pre: Tensor, rope: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Exact EVA gamma-1 final-MHSA residual decomposition."""

    gamma1 = getattr(block, "gamma_1", None)
    gamma2 = getattr(block, "gamma_2", None)
    if gamma1 is None or gamma2 is None:
        raise ValueError("locked B9 final block must expose gamma_1 and gamma_2")
    z = block.norm1(x_pre)
    attention = block.attn(z, rope=rope, attn_mask=None, is_causal=False)
    u = x_pre + block.drop_path1(gamma1 * attention)
    return z, u


def frozen_tail_logits(
    block: nn.Module,
    final_norm: nn.Module,
    head: nn.Module,
    u_patch: Tensor,
    residual: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Run the exact frozen gamma-2 MLP tail on patch tokens only."""

    if tuple(u_patch.shape) != tuple(residual.shape):
        raise ValueError("u_patch/residual shapes must match")
    gamma2 = getattr(block, "gamma_2", None)
    if gamma2 is None:
        raise ValueError("locked B9 final block must expose gamma_2")
    v = u_patch + residual
    y = v + block.drop_path2(gamma2 * block.mlp(block.norm2(v)))
    final_patch = final_norm(y)
    logits = head(final_patch.mean(dim=1))
    return logits, final_patch


def tail_adjusted_logits(
    block: nn.Module,
    final_norm: nn.Module,
    head: nn.Module,
    u_patch: Tensor,
    residual: Tensor,
    cached_base_logits: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Keep branch-off bit exact via cached-base plus active-minus-base tail."""

    zeros = torch.zeros_like(residual)
    base_tail_logits, base_final = frozen_tail_logits(
        block, final_norm, head, u_patch, zeros
    )
    active_tail_logits, active_final = frozen_tail_logits(
        block, final_norm, head, u_patch, residual
    )
    adjusted = cached_base_logits + (active_tail_logits - base_tail_logits)
    return adjusted, {
        "base_tail_logits": base_tail_logits,
        "active_tail_logits": active_tail_logits,
        "base_final_patch_tokens": base_final,
        "active_final_patch_tokens": active_final,
    }


def select_fixed_smoke_indices(
    relative_paths: Sequence[str],
    labels: Sequence[int],
    *,
    expected_paths_sha256: Optional[str] = EXPECTED_SMOKE_PATHS_SHA256,
) -> Tuple[List[int], List[str], str]:
    """Select exactly one immutable TRAIN path per class by path digest."""

    if len(relative_paths) != len(labels):
        raise ValueError("smoke paths/labels are not aligned")
    selected: List[int] = []
    for class_index in range(5):
        support = [
            index for index, label in enumerate(labels) if int(label) == class_index
        ]
        if not support:
            raise ValueError(f"smoke selection lacks class {class_index}")
        selected.append(
            min(
                support,
                key=lambda index: (
                    hashlib.sha256(
                        str(relative_paths[index]).encode("utf-8")
                    ).hexdigest(),
                    str(relative_paths[index]),
                ),
            )
        )
    paths = [str(relative_paths[index]) for index in selected]
    observed_hash = _json_sha256(paths)
    if expected_paths_sha256 and observed_hash != expected_paths_sha256:
        raise ValueError(
            "fixed TRAIN smoke path hash changed: "
            f"expected={expected_paths_sha256}, observed={observed_hash}"
        )
    return selected, paths, observed_hash


def _maximum_error(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def _native_final_block_capture(
    block: nn.Module, x_pre: Tensor, rope: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    captured: Dict[str, Optional[Tensor]] = {}

    def capture_attention(
        _module: nn.Module, args: Tuple[Tensor, ...], kwargs: Dict[str, object]
    ) -> None:
        captured["z"] = args[0].detach()
        value = kwargs.get("rope")
        captured["rope"] = value.detach() if torch.is_tensor(value) else None

    def capture_norm2(_module: nn.Module, args: Tuple[Tensor, ...]) -> None:
        captured["u"] = args[0].detach()

    attention_hook = block.attn.register_forward_pre_hook(
        capture_attention, with_kwargs=True
    )
    norm2_hook = block.norm2.register_forward_pre_hook(capture_norm2)
    try:
        output = block(x_pre, rope=rope, attn_mask=None, is_causal=False)
    finally:
        attention_hook.remove()
        norm2_hook.remove()
    if not all(key in captured for key in ("z", "u", "rope")):
        raise RuntimeError("native EVA final-block capture was incomplete")
    return output, captured["z"], captured["u"], captured["rope"]


def _pairwise_bce(logits: Tensor, targets: Tensor) -> Tensor:
    losses: List[Tensor] = []
    for rival in (0, 2, 4):
        mask = torch.logical_or(targets == 1, targets == rival)
        if int(mask.sum().item()) == 0:
            raise RuntimeError(f"smoke batch lacks the 1-vs-{rival} pair")
        margin = logits[mask, 1] - logits[mask, rival]
        binary = (targets[mask] == 1).to(dtype=margin.dtype)
        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(margin, binary)
        )
    return torch.stack(losses).mean()


def run_actual_b9_train_smoke(
    *,
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    relative_paths: Sequence[str],
) -> Dict[str, object]:
    """Falsify placement/parity/gradient errors on five locked TRAIN rows."""

    if images.size(0) != SMOKE_BATCH_SIZE or labels.tolist() != list(range(5)):
        raise ValueError("actual B9 smoke requires one ordered TRAIN row per class")
    if len(relative_paths) != SMOKE_BATCH_SIZE:
        raise ValueError("actual B9 smoke path count changed")
    model.eval()
    model.requires_grad_(False)
    before = _state_subset_hashes(model.state_dict())
    _assert_locked_state_hashes(before)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("B9 backbone/tail/head did not freeze")

    block = model.blocks[FINAL_BLOCK_INDEX]
    prefix = int(getattr(model, "num_prefix_tokens", 0) or 0)
    if prefix != XCNORM_PAIR_A1_PREFIX_TOKEN_COUNT:
        raise ValueError("B9 prefix-token contract changed")

    with torch.inference_mode():
        direct_b9_logits = model(images).float()
        x_pre, rope = eva_pre_final_tokens(model, images)
        if rope is None:
            raise ValueError("locked DINOv3 A1 requires native RoPE")
        z, u = decompose_final_mhsa(block, x_pre, rope)
        native_tail, native_z, native_u, native_rope = _native_final_block_capture(
            block, x_pre, rope
        )
        native_final = model.norm(native_tail)
        native_logits = model.forward_head(native_final, pre_logits=False).float()
        native_direct_logit_error = _maximum_error(
            native_logits, direct_b9_logits
        )
        z_error = _maximum_error(z, native_z)
        u_error = _maximum_error(u, native_u)
        rope_error = _maximum_error(rope, native_rope)
        u_patch = u[:, prefix:].detach()
        z_patch = z[:, prefix:].detach()
        manual_logits, manual_final_patch = frozen_tail_logits(
            block,
            model.norm,
            model.head,
            u_patch,
            torch.zeros_like(u_patch),
        )
        tail_token_error = _maximum_error(
            manual_final_patch, native_final[:, prefix:]
        )
        tail_logit_error = _maximum_error(manual_logits, direct_b9_logits)

    for name, value in {
        "norm1_capture": z_error,
        "post_mhsa_u": u_error,
        "rope_capture": rope_error,
        "native_direct_logits": native_direct_logit_error,
    }.items():
        if value > STRUCTURAL_PARITY_ATOL:
            raise RuntimeError(f"actual B9 {name} parity failed: {value}")
    for name, value in {
        "tail_tokens": tail_token_error,
        "tail_logits": tail_logit_error,
    }.items():
        if value > PATCH_TAIL_PARITY_ATOL:
            raise RuntimeError(f"actual B9 {name} parity failed: {value}")

    frozen_weight_cpu = model.head.weight.detach().cpu().float()
    _control, candidate, paired = build_paired_adapters(frozen_weight_cpu)
    candidate = candidate.to(device=images.device, dtype=torch.float32)
    frozen_weight = model.head.weight.detach().float()
    candidate.bind_classifier_weight(frozen_weight)

    with torch.no_grad():
        zero_residual = candidate.residual_from_final_norm1(
            z_patch, u_patch, frozen_weight
        )
        branch_off_residual = candidate.residual_from_final_norm1(
            z_patch, u_patch, frozen_weight, branch_off=True
        )
        zero_logits, _ = tail_adjusted_logits(
            block,
            model.norm,
            model.head,
            u_patch,
            zero_residual,
            direct_b9_logits,
        )
        branch_off_logits, _ = tail_adjusted_logits(
            block,
            model.norm,
            model.head,
            u_patch,
            branch_off_residual,
            direct_b9_logits,
        )
    if torch.count_nonzero(zero_residual) or torch.count_nonzero(branch_off_residual):
        raise RuntimeError("A1 zero/branch-off residual is not exactly zero")
    if not torch.equal(zero_logits, direct_b9_logits) or not torch.equal(
        branch_off_logits, direct_b9_logits
    ):
        raise RuntimeError("A1 branch-off final logits are not bit exact B9")

    # Activate only the adapter's already-registered projection for a gradient
    # plumbing check. No B9 state is changed and no optimizer step is taken.
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        candidate.core.pair_projection.weight.fill_(0.01)
    residual, residual_trace = candidate.residual_from_final_norm1(
        z_patch, u_patch, frozen_weight, return_trace=True
    )
    residual.retain_grad()
    active_logits, active_trace = tail_adjusted_logits(
        block,
        model.norm,
        model.head,
        u_patch,
        residual,
        direct_b9_logits,
    )
    loss = _pairwise_bce(active_logits, labels)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("A1 actual-B9 smoke loss is non-finite")
    loss.backward()
    gradient_norms = {
        name: float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in candidate.named_parameters()
        if parameter.grad is not None
    }
    expected_gradient_names = {name for name, _ in candidate.named_parameters()}
    if set(gradient_norms) != expected_gradient_names:
        raise RuntimeError("not every A1 parameter received a gradient")
    if not all(np.isfinite(value) and value > 0.0 for value in gradient_norms.values()):
        raise RuntimeError(f"A1 adapter gradient is zero/non-finite: {gradient_norms}")
    residual_gradient_norm = float(residual.grad.detach().float().norm().cpu())
    if not np.isfinite(residual_gradient_norm) or residual_gradient_norm <= 0.0:
        raise RuntimeError("gradient did not cross the frozen gamma2/MLP/final norm")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("gradient leaked into frozen B9")

    after = _state_subset_hashes(model.state_dict())
    if after != before:
        raise RuntimeError("frozen B9 state hashes changed during A1 smoke")

    final_delta = active_trace["active_final_patch_tokens"] - active_trace[
        "base_final_patch_tokens"
    ]
    final_ratio = final_delta.float().norm(dim=-1) / active_trace[
        "base_final_patch_tokens"
    ].float().norm(dim=-1).clamp_min(1e-12)
    injection_ratio = residual_trace["residual_norm_ratio"].detach().float()
    logit_delta = (active_logits - direct_b9_logits).detach().float().norm(dim=-1)
    return {
        "smoke_paths": list(relative_paths),
        "smoke_paths_sha256": _json_sha256(list(relative_paths)),
        "labels": labels.detach().cpu().tolist(),
        "state_hashes_before": before,
        "state_hashes_after": after,
        "rope_state_sha256": _state_dict_sha256(
            {"final_block_rope": rope.detach().cpu()}
        ),
        "gamma_formula": (
            "u=x_pre+drop_path1(gamma_1*attn(norm1(x_pre),rope))"
        ),
        "tail_formula": (
            "v=u+r;y=v+drop_path2(gamma_2*mlp(norm2(v)));"
            "final_norm;mean_256_patch;frozen_head"
        ),
        "parity": {
            "norm1_capture_max_abs_error": z_error,
            "post_mhsa_u_max_abs_error": u_error,
            "rope_capture_max_abs_error": rope_error,
            "native_direct_logit_max_abs_error": native_direct_logit_error,
            "tail_token_max_abs_error": tail_token_error,
            "tail_logit_max_abs_error": tail_logit_error,
            "zero_initialization_logit_max_abs_error": _maximum_error(
                zero_logits, native_logits
            ),
            "branch_off_logit_max_abs_error": _maximum_error(
                branch_off_logits, native_logits
            ),
        },
        "adapter_gradient_norms": gradient_norms,
        "residual_gradient_through_frozen_tail_norm": residual_gradient_norm,
        "frozen_model_gradients_present": False,
        "smoke_loss": float(loss.detach().cpu()),
        "telemetry": {
            "injection_ratio_p95": float(torch.quantile(injection_ratio, 0.95).cpu()),
            "final_token_perturb_ratio_p95": float(
                torch.quantile(final_ratio, 0.95).detach().cpu()
            ),
            "logit_delta_l2_p95": float(
                torch.quantile(logit_delta, 0.95).detach().cpu()
            ),
        },
        "paired_adapter": paired,
        "actual_b9_train_smoke_passed": True,
    }


def _runtime_contract(model: nn.Module) -> Dict[str, object]:
    block = model.blocks[FINAL_BLOCK_INDEX]
    return {
        "timm_version": package_version("timm"),
        "model_class": type(model).__name__,
        "model_training": bool(model.training),
        "model_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "global_pool": str(getattr(model, "global_pool", "")),
        "dynamic_img_size": bool(getattr(model, "dynamic_img_size", False)),
        "embed_dim": int(getattr(model, "embed_dim", 0) or 0),
        "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 0) or 0),
        "patch_grid_size": list(getattr(model.patch_embed, "grid_size", ())),
        "patch_embed_class": type(model.patch_embed).__name__,
        "norm_pre_class": type(model.norm_pre).__name__,
        "rope_class": type(getattr(model, "rope", None)).__name__,
        "rope_mixed": bool(getattr(model, "rope_mixed", False)),
        "block_count": len(model.blocks),
        "final_block_index": FINAL_BLOCK_INDEX,
        "final_block_class": type(block).__name__,
        "norm1_class": type(block.norm1).__name__,
        "attention_class": type(block.attn).__name__,
        "drop_path1_class": type(block.drop_path1).__name__,
        "gamma1_shape": list(block.gamma_1.shape),
        "norm2_class": type(block.norm2).__name__,
        "mlp_class": type(block.mlp).__name__,
        "drop_path2_class": type(block.drop_path2).__name__,
        "gamma2_shape": list(block.gamma_2.shape),
        "final_norm_class": type(model.norm).__name__,
        "fc_norm_class": type(model.fc_norm).__name__,
        "head_drop_class": type(model.head_drop).__name__,
        "head_drop_probability": float(model.head_drop.p),
        "head_class": type(model.head).__name__,
        "head_shape": [int(model.head.out_features), int(model.head.in_features)],
    }


def run_precheck(args: argparse.Namespace) -> Dict[str, object]:
    # These fail before filesystem/model access and are covered by unit tests.
    if int(args.adapter_batch_size) != ADAPTER_BATCH_SIZE:
        raise ValueError(
            f"adapter batch size is protocol-locked to {ADAPTER_BATCH_SIZE}"
        )
    if not bool(args.preflight_only):
        raise RuntimeError(
            "A1 cache/OOF is intentionally not implemented yet; "
            "run this hard gate with --preflight-only"
        )
    if int(args.workers) < 0:
        raise ValueError("workers must be non-negative")

    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    data_yaml = args.data.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if _sha256(data_yaml) != EXPECTED_DATA_SHA256:
        raise ValueError("data YAML is not the locked canonical class_f contract")
    data_root = _data_root_from_yaml(data_yaml)
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("output-dir must be outside immutable class_f")

    manifest_path = (args.manifest or data_root / "manifest.csv").expanduser().resolve()
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest is not the locked canonical class_f manifest")
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    checkpoint_contract = _validate_b9_checkpoint(checkpoint, checkpoint_sha)

    dataset, classes = _build_dataset(
        data_yaml=data_yaml,
        split="train",
        checkpoint=checkpoint,
        class_name_mode="raw",
        max_samples=0,
    )
    if tuple(classes) != EXPECTED_CLASSES or len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("canonical class order/train support changed")
    dataset_paths = [Path(path) for path in dataset.sample_paths()]
    assert_train_only_paths(dataset_paths, data_root / "train")
    relative_paths = [_relative_train_path(path, data_root) for path in dataset_paths]
    dataset_labels = np.asarray(dataset.labels(), dtype=np.int64)
    canonical_rows = [
        {"relative_path": path, "label": int(dataset_labels[index])}
        for index, path in enumerate(relative_paths)
    ]
    manifest_rows = _read_train_rows(manifest_path, data_root)
    manifest_map = {
        str(row["relative_path"]).casefold(): int(row["label"])
        for row in manifest_rows
    }
    dataset_map = {
        str(row["relative_path"]).casefold(): int(row["label"])
        for row in canonical_rows
    }
    if manifest_map != dataset_map:
        raise ValueError("dataset loader does not exactly match canonical manifest")

    integrity_path = args.train_integrity_manifest.expanduser().resolve()
    if _sha256(integrity_path) != EXPECTED_INTEGRITY_MANIFEST_SHA256:
        raise ValueError("train integrity manifest is not the locked radius-3 artifact")
    integrity_rows = _read_integrity_train_rows(
        integrity_path, canonical_rows, data_root
    )
    phashes = [int(row["phash_value"]) for row in integrity_rows]
    groups, group_stats = build_train_union_groups(integrity_rows, phashes)
    _folds, fold_rows, assignment_hashes = assign_locked_folds(
        integrity_rows, groups, canonical_lock=True
    )
    if (
        assignment_hashes["assignment_int64_sha256"]
        != EXPECTED_ASSIGNMENT_INT64_SHA256
        or assignment_hashes["path_fold_sha256"] != EXPECTED_PATH_FOLD_SHA256
        or tuple(tuple(row["held_class_counts"]) for row in fold_rows)
        != EXPECTED_HELD_CLASS_COUNTS
    ):
        raise RuntimeError("canonical A1 source-fold lock changed")

    smoke_indices, smoke_paths, smoke_hash = select_fixed_smoke_indices(
        relative_paths, dataset_labels
    )
    smoke_loader = DataLoader(
        Subset(dataset, smoke_indices),
        batch_size=SMOKE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_classification,
    )
    images, targets, _metadata = next(iter(smoke_loader))
    if targets.tolist() != list(range(5)):
        raise RuntimeError("fixed smoke sample order/labels changed")

    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model.requires_grad_(False)
    runtime = _runtime_contract(model)
    expected_runtime = {
        "timm_version": EXPECTED_TIMM_VERSION,
        "model_class": "Eva",
        "model_training": False,
        "model_parameters_frozen": True,
        "global_pool": "avg",
        "dynamic_img_size": True,
        "embed_dim": 384,
        "num_prefix_tokens": 5,
        "patch_grid_size": [16, 16],
        "patch_embed_class": "PatchEmbed",
        "norm_pre_class": "Identity",
        "rope_class": "RotaryEmbeddingDinoV3",
        "rope_mixed": False,
        "block_count": 12,
        "final_block_index": 11,
        "final_block_class": "EvaBlock",
        "norm1_class": "LayerNorm",
        "attention_class": "EvaAttention",
        "drop_path1_class": "Identity",
        "gamma1_shape": [384],
        "norm2_class": "LayerNorm",
        "mlp_class": "Mlp",
        "drop_path2_class": "Identity",
        "gamma2_shape": [384],
        "final_norm_class": "LayerNorm",
        "fc_norm_class": "Identity",
        "head_drop_class": "Dropout",
        "head_drop_probability": 0.0,
        "head_class": "Linear",
        "head_shape": [5, 384],
    }
    if runtime != expected_runtime:
        raise ValueError(f"B9 EVA runtime contract mismatch: {runtime}")
    _assert_locked_state_hashes(_state_subset_hashes(model.state_dict()))

    device = _resolve_device(args.device)
    model.to(device=device, dtype=torch.float32)
    smoke = run_actual_b9_train_smoke(
        model=model,
        images=images.to(device=device, dtype=torch.float32),
        labels=targets.to(device=device),
        relative_paths=smoke_paths,
    )
    preflight = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "actual_b9_train_integration_preflight_only",
        "checkpoint_contract": checkpoint_contract,
        "locked_state_hashes": smoke["state_hashes_before"],
        "runtime_contract": runtime,
        "canonical_train": {
            "samples": len(dataset),
            "class_names": list(classes),
            "class_counts": np.bincount(dataset_labels, minlength=5).tolist(),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "integrity_manifest_sha256": EXPECTED_INTEGRITY_MANIFEST_SHA256,
            "group_stats": group_stats,
            "folds": fold_rows,
            **assignment_hashes,
        },
        "future_locked_oof_optimization": {
            "folds": FOLDS,
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "adapter_batch_size": ADAPTER_BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "tempered_sampling_power": TEMPERED_POWER,
            "status": "not_run_preflight_only",
        },
        "smoke": smoke,
        "smoke_paths_sha256": smoke_hash,
        "source_files": {
            "tool_sha256": _sha256(Path(__file__).resolve()),
            "a1_model_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "models"
                / "dinov3_xcnorm_pair_adapter_a1.py"
            ),
            "a0_operator_model_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "models"
                / "dinov3_xcnorm_pair_adapter_a0.py"
            ),
            "a0_shared_protocol_sha256": _sha256(
                Path(__file__).resolve().with_name(
                    "precheck_dinov3_xcnorm_a0_sourcefold.py"
                )
            ),
            "model_builder_sha256": _sha256(
                Path(__file__).resolve().parents[1] / "models" / "model.py"
            ),
            "dataset_pipeline_sha256": _sha256(
                Path(__file__).resolve().parents[1] / "data" / "dataset.py"
            ),
            "probe_loader_sha256": _sha256(
                Path(__file__).resolve().with_name(
                    "probe_embedding_prototypes.py"
                )
            ),
            "timm_eva_source_sha256": _sha256(
                Path(inspect.getsourcefile(type(model)) or "").resolve()
            ),
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "large_cache_created": False,
        "oof_training_run": False,
        "photometric_train_permission": False,
        "full_validation_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "preflight_passed": True,
    }
    _atomic_write_json(output_dir / "preflight.json", preflight)
    return preflight


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_precheck(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "preflight_passed": bool(result["preflight_passed"]),
                "large_cache_created": False,
                "validation_split_used": False,
                "test_split_used": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

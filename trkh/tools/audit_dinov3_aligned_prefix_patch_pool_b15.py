from __future__ import annotations

import argparse
import gc
import hashlib
import io
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import random
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from trkh.tools.audit_dinov3_depth_trajectory_b12 import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
)
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (  # noqa: E402
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    CLASSES,
    DINO_FEATURE_DIM,
    DINO_IMAGE_SIZE,
    DINO_PREFIX_TOKENS,
    DINO_WEIGHT_SHA256,
    FOLDS,
    ORT_ITERATIONS,
    ORT_THREADS,
    ORT_TRIALS,
    ORT_WARMUPS,
    RIVALS,
    SEED as B13_SEED,
    WORKERS,
    _atomic_npy,
    _build_dino,
    _dino_transform,
    _export_onnx,
    _fast_f1_from_predictions,
    _git_contract,
    _ort_session,
    _resolve_device,
    _resolve_dino_weight,
    _sha256,
    _timed_ort,
    _train_content_contract,
    classification_summary,
    fit_oof_readout,
)
from trkh.tools.audit_iformer_s_frozen_transfer_b14 import (  # noqa: E402
    B13_OOF_SHA256,
    B13_PATHS_SHA256,
    B13_RUNNER_SHA256,
    B13_SUMMARY_SHA256,
    EXPECTED_BOOTSTRAP_DRAW_SHA256,
    EXPECTED_BRANCH,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_DATA_YAML_SHA256,
    EXPECTED_DINO_METRICS,
    EXPECTED_FOLD_VECTOR_SHA256,
    EXPECTED_GROUP_VECTOR_SHA256,
    EXPECTED_TRAIN_CONTENT_SHA256,
    EXPECTED_TRAIN_ROWS,
    _int64_sha,
    _resolve_train_paths,
    _runtime_contract,
    _validate_b13_summary_payload,
    _validate_dino_metrics,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B15_ALIGNED_PREFIX_PATCH_POOL_20260804"
DONOR_SEED = 20260804
DESCRIPTOR_DIM = DINO_FEATURE_DIM
PATCH_TOKENS = 256
TOKEN_COUNT = DINO_PREFIX_TOKENS + PATCH_TOKENS
LAYER_NORM_EPS = 1e-6
ARMS = ("uniform", "prefix_only", "aligned", "deranged")
RETAINED_DESCRIPTOR_ARMS = ("prefix_only", "aligned", "deranged")
RETAINED_DESCRIPTOR_FILENAMES = {
    "prefix_only": "train_prefix_only_f32.npy",
    "aligned": "train_aligned_prefix_patch_pool_f32.npy",
    "deranged": "train_deranged_prefix_patch_pool_f32.npy",
}
BOOTSTRAP_COMPARATORS = ("uniform", "prefix_only")
PAIR2 = "2"
WALL_BUDGET_SECONDS = 15 * 60
VRAM_BUDGET_BYTES = 6 * 1024**3
ARTIFACT_BUDGET_BYTES = 100 * 1024**2
PRE_METRIC_ARTIFACT_RESERVE_BYTES = 16 * 1024**2
EXPECTED_CUDA_NAME = "NVIDIA GeForce RTX 4060 Laptop GPU"
EXPECTED_CUDA_CAPABILITY = (8, 9)
EXPECTED_CUDA_MEMORY_BYTES = 8_585_216_000
PREFLIGHT_OUTPUT_PREFIX = "preflight_b15_aligned_prefix_patch_pool_"
FORMAL_OUTPUT_PREFIX = "pretrained_dinov3_classf_b15_aligned_prefix_patch_pool_"
EXPECTED_DINO_DESCRIPTOR_SHA256 = (
    "0429260ab2f619cf8312848e08af1a13f4c81caa95a93daebda6e6e22240dc38"
)
EXPECTED_DINO_WEIGHT_BYTES = 86_362_376
EXPECTED_TIMM_SOURCE_HASHES = {
    "_factory.py": "30a6eecdaba750af470cfae3196186fd647052c72338a21c928914aac06163e4",
    "eva.py": "23314ef536d7ce9cc3737e54f424841b46f426e93af651c357b4a75a8b00a12d",
}
B14_EVIDENCE_ROOT = "pretrained_iformer_s_classf_b14_frozen_transfer_c797a6f_r1"
B14_EVIDENCE_HASHES = {
    "summary.json": "9ad53ca882ceec174cbe19aca9dfd8e268c1c4380ae8172ee3c745d0f166ba90",
    "train_oof_readouts.npz": "ae3ad2420f070d79300701e4c222e8465afa45079a5c81a7ac0f5e30e329ffde",
    "train_paths.json": "a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb",
    "train_iformer_s_pooled320_f32.npy": "255141e85097a1f671ab53e06b5d19b1890f1d96bcd597137e877deae6ce3217",
}
B14_ACCEPTED_PREFLIGHT_SHA256 = (
    "aa59fef3d61be981c6540569eb35d1c7d885de5cf73675c28230af18aaf77f0a"
)
B14_PROTOCOL_SHA256 = "74302e9ded89099866aed060203973f297e71fbed8779fb2893e509a52d5a8c5"
B14_RUNNER_SHA256 = "c9eafebeec471db175e397119ecb648d591b98593d2cdc2e35a602c5a0c080fa"
OPERATOR_CHECK_NAMES = (
    "operator_parameters_zero",
    "operator_buffers_zero",
    "wrapper_parameter_counts_equal",
    "descriptor_geometry_exact",
    "uniform_identity_exact",
    "prefix_identity_exact",
    "direct_formula_parity",
    "attention_rows_sum_one",
    "prefix_order_permutation_invariant",
    "joint_patch_weight_permutation_invariant",
    "uniform_query_weights_exact",
    "uniform_query_mean_pool_max_abs_lte_1e_6",
    "donor_centering_formula_parity",
    "donor_centered_reconstruction_parity",
    "donor_common_offset_invariant",
    "forbidden_current_center_changes_output",
    "attention_entropy_finite_bounded",
    "effective_patch_count_finite_bounded",
    "donor_repeat_exact",
    "donor_fixed_points_zero",
    "donor_fold_crossings_zero",
    "donor_same_union_group_zero",
    "donor_bijective_all_folds",
    "onnx_standard_domains_only",
    "onnx_parity_max_abs_lte_1e_5",
    "onnx_argmax_exact",
    "operator_p95_overhead_lte_5pct",
)
SYNTHETIC_CHECK_NAMES = frozenset(
    {
        "descriptor_geometry_exact",
        "uniform_identity_exact",
        "prefix_identity_exact",
        "direct_formula_parity",
        "attention_rows_sum_one",
        "prefix_order_permutation_invariant",
        "joint_patch_weight_permutation_invariant",
        "uniform_query_weights_exact",
        "uniform_query_mean_pool_max_abs_lte_1e_6",
        "donor_centering_formula_parity",
        "donor_centered_reconstruction_parity",
        "donor_common_offset_invariant",
        "forbidden_current_center_changes_output",
        "attention_entropy_finite_bounded",
        "effective_patch_count_finite_bounded",
        "donor_repeat_exact",
        "donor_fixed_points_zero",
        "donor_fold_crossings_zero",
        "donor_same_union_group_zero",
        "donor_bijective_all_folds",
    }
)
MOBILE_OPERATOR_CHECK_NAMES = frozenset(
    {
        "operator_parameters_zero",
        "operator_buffers_zero",
        "wrapper_parameter_counts_equal",
        "descriptor_geometry_exact",
        "onnx_standard_domains_only",
        "onnx_parity_max_abs_lte_1e_5",
        "onnx_argmax_exact",
        "operator_p95_overhead_lte_5pct",
    }
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _defaults() -> dict[str, Path]:
    root = _repository_root()
    return {
        "data_yaml": Path(r"D:\DataAI\AIEx\newdataset\class_f\data.yaml"),
        "data_root": Path(r"D:\DataAI\AIEx\newdataset\class_f"),
        "b13_root": root
        / "runs"
        / "pretrained_efficientvim_classf_b13_frozen_transfer_f229649_r1",
        "b14_root": root / "runs" / B14_EVIDENCE_ROOT,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Locked B15 TRAIN-only parameter-free aligned prefix-patch pool screen"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, default=defaults["data_yaml"])
    parser.add_argument("--data-root", type=Path, default=defaults["data_root"])
    parser.add_argument("--b13-root", type=Path, default=defaults["b13_root"])
    parser.add_argument("--b14-root", type=Path, default=defaults["b14_root"])
    parser.add_argument("--dino-weight", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _assert_locked_locations(args: argparse.Namespace, *, include_data: bool) -> None:
    if not include_data:
        return
    defaults = _defaults()
    mismatches = {}
    for name in ("data_yaml", "data_root", "b13_root", "b14_root"):
        observed = Path(getattr(args, name)).expanduser().resolve()
        expected = defaults[name].expanduser().resolve()
        if observed != expected:
            mismatches[name] = {"observed": str(observed), "expected": str(expected)}
    if mismatches:
        raise ValueError(f"B15 locked artifact locations changed: {mismatches}")


def _source_paths() -> dict[str, Path]:
    root = _repository_root()
    return {
        "runner": Path(__file__).resolve(),
        "runner_test": root
        / "tests"
        / "test_audit_dinov3_aligned_prefix_patch_pool_b15.py",
        "protocol": root
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B15_ALIGNED_PREFIX_PATCH_POOL_PROTOCOL_20260804.md",
        "b13_runner": root
        / "trkh"
        / "tools"
        / "audit_efficientvim_m1_frozen_transfer_b13.py",
        "b14_runner": root
        / "trkh"
        / "tools"
        / "audit_iformer_s_frozen_transfer_b14.py",
    }


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = _source_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"B15 bound source is missing: {name}={path}")
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"B15 bound source is not tracked: {name}={path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes["b13_runner"] != B13_RUNNER_SHA256:
        raise ValueError("B15 locked B13 helper source changed")
    if hashes["b14_runner"] != B14_RUNNER_SHA256:
        raise ValueError("B15 locked B14 evidence loader source changed")
    return hashes


def _dino_source_contract() -> dict[str, object]:
    import timm.models._factory as factory
    import timm.models.eva as eva

    paths = {
        "_factory.py": Path(factory.__file__).resolve(),
        "eva.py": Path(eva.__file__).resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_TIMM_SOURCE_HASHES:
        raise ValueError(f"B15 locked timm source changed: {hashes}")
    return {
        "files": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
    }


def _dino_model_contract(model: nn.Module, weight: Path) -> dict[str, object]:
    import timm

    data_config = timm.data.resolve_model_data_config(model)
    observed = {
        "runtime_class": f"{type(model).__module__}.{type(model).__name__}",
        "features": int(getattr(model, "num_features", -1)),
        "prefix_tokens": int(getattr(model, "num_prefix_tokens", -1)),
        "grid": [int(value) for value in model.patch_embed.grid_size],
        "weight_bytes": int(weight.stat().st_size),
        "weight_sha256": _sha256(weight),
        "mean": [float(value) for value in data_config["mean"]],
        "std": [float(value) for value in data_config["std"]],
    }
    expected = {
        "runtime_class": "timm.models.eva.Eva",
        "features": DESCRIPTOR_DIM,
        "prefix_tokens": DINO_PREFIX_TOKENS,
        "grid": [16, 16],
        "weight_bytes": EXPECTED_DINO_WEIGHT_BYTES,
        "weight_sha256": DINO_WEIGHT_SHA256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    if observed != expected:
        raise ValueError(f"B15 DINO model contract changed: {observed}")
    return observed


def _focused_test_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "tests/test_audit_dinov3_aligned_prefix_patch_pool_b15.py",
    ]


def _run_focused_tests() -> dict[str, object]:
    command = _focused_test_command()
    result = subprocess.run(
        command,
        cwd=_repository_root(),
        capture_output=True,
        text=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "skipped" in output.casefold():
        raise RuntimeError(f"B15 focused tests failed or skipped:\n{output}")
    return {"command": command, "returncode": result.returncode, "output": output}


class AlignedPrefixPatchPool(nn.Module):
    """Locked zero-parameter operator over final post-normalization DINO tokens."""

    def forward(
        self,
        tokens: Tensor,
        prefix_tokens: Tensor | None = None,
        prefix_center: Tensor | None = None,
    ) -> Tensor:
        return _descriptor_arms(
            tokens,
            prefix_tokens=prefix_tokens,
            prefix_center=prefix_center,
        )["aligned"]


def _validate_tokens(tokens: Tensor) -> None:
    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("B15 tokens must be a rank-3 tensor")
    if tuple(tokens.shape[1:]) != (TOKEN_COUNT, DESCRIPTOR_DIM):
        raise ValueError(f"B15 final token geometry changed: {tuple(tokens.shape)}")
    if not tokens.is_floating_point():
        raise TypeError("B15 tokens must be floating point")


def _descriptor_arms(
    tokens: Tensor,
    *,
    prefix_tokens: Tensor | None = None,
    prefix_center: Tensor | None = None,
) -> dict[str, Tensor]:
    uniform, prefix_only, aligned, _weights = _descriptor_details(
        tokens,
        prefix_tokens=prefix_tokens,
        prefix_center=prefix_center,
    )
    return {
        "uniform": uniform,
        "prefix_only": prefix_only,
        "aligned": aligned,
    }


def _descriptor_details(
    tokens: Tensor,
    *,
    prefix_tokens: Tensor | None = None,
    prefix_center: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_tokens(tokens)
    own_prefix = tokens[:, :DINO_PREFIX_TOKENS]
    patches = tokens[:, DINO_PREFIX_TOKENS:]
    uniform = patches.mean(dim=1)
    prefix_only = own_prefix.mean(dim=1)
    prefixes = own_prefix if prefix_tokens is None else prefix_tokens
    if tuple(prefixes.shape) != tuple(own_prefix.shape):
        raise ValueError("B15 donor prefix block must be [B,5,384]")
    if prefixes.device != tokens.device or prefixes.dtype != tokens.dtype:
        raise ValueError("B15 donor prefixes must share token device and dtype")
    center = uniform if prefix_center is None else prefix_center
    if tuple(center.shape) != tuple(uniform.shape):
        raise ValueError("B15 prefix center must be [B,384]")
    if center.device != tokens.device or center.dtype != tokens.dtype:
        raise ValueError("B15 prefix center must share token device and dtype")
    q = F.layer_norm(
        prefixes - center[:, None],
        (DESCRIPTOR_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )
    k = F.layer_norm(
        patches - uniform[:, None],
        (DESCRIPTOR_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )
    scores = torch.einsum("bjd,bid->bji", q, k).sum(dim=1)
    scores = scores / (DINO_PREFIX_TOKENS * math.sqrt(DESCRIPTOR_DIM))
    weights = torch.softmax(scores, dim=1)
    aligned = torch.einsum("bi,bid->bd", weights, patches)
    return uniform, prefix_only, aligned, weights


class _DinoUniformDescriptor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.model.forward_features(images)
        _validate_tokens(tokens)
        return tokens[:, DINO_PREFIX_TOKENS:].mean(dim=1)


class _DinoAlignedDescriptor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.pool = AlignedPrefixPatchPool()

    def forward(self, images: Tensor) -> Tensor:
        return self.pool(self.model.forward_features(images))


def _in_memory_npy_contract(array: np.ndarray) -> dict[str, object]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    view = buffer.getbuffer()
    payload = {
        "sha256": hashlib.sha256(view).hexdigest(),
        "bytes": int(len(view)),
        "retained_on_disk": False,
    }
    del view
    buffer.close()
    return payload


def build_fold_contained_donor_map(
    folds: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = DONOR_SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    fold_array = np.asarray(folds, dtype=np.int64).reshape(-1)
    group_array = np.asarray(groups).reshape(-1)
    if fold_array.shape != group_array.shape or fold_array.size < 2:
        raise ValueError("B15 donor fold/group rows are not aligned")
    observed_folds = sorted(int(value) for value in np.unique(fold_array))
    if observed_folds != list(range(FOLDS)):
        raise ValueError("B15 donor map requires the five locked folds")
    donors = np.full(fold_array.size, -1, dtype=np.int64)
    fold_reports: list[dict[str, object]] = []
    for fold in range(FOLDS):
        rows = np.flatnonzero(fold_array == fold).astype(np.int64)
        rng = np.random.default_rng(int(seed) + fold)
        grouped: dict[int, list[int]] = {}
        for row in rows.tolist():
            grouped.setdefault(int(group_array[row]), []).append(int(row))
        group_keys = list(grouped)
        rng.shuffle(group_keys)
        ordered: list[int] = []
        maximum_group_size = 0
        for key in group_keys:
            members = np.asarray(grouped[key], dtype=np.int64)
            rng.shuffle(members)
            ordered.extend(int(value) for value in members.tolist())
            maximum_group_size = max(maximum_group_size, int(members.size))
        if maximum_group_size * 2 > rows.size:
            raise ValueError(
                f"B15 fold {fold} has no union-group-disjoint derangement"
            )
        ordered_array = np.asarray(ordered, dtype=np.int64)
        rolled = np.roll(ordered_array, -maximum_group_size)
        mapping = {
            int(recipient): int(donor)
            for recipient, donor in zip(ordered_array.tolist(), rolled.tolist())
        }
        selected = np.asarray([mapping[int(row)] for row in rows], dtype=np.int64)
        donors[rows] = selected
        fold_reports.append(
            {
                "fold": fold,
                "seed": int(seed) + fold,
                "rows": int(rows.size),
                "largest_union_group": maximum_group_size,
                "twice_largest_lte_rows": 2 * maximum_group_size <= rows.size,
                "fixed_points": int(np.sum(selected == rows)),
                "fold_crossings": int(np.sum(fold_array[selected] != fold)),
                "same_union_group_assignments": int(
                    np.sum(group_array[selected] == group_array[rows])
                ),
                "bijective": set(selected.tolist()) == set(rows.tolist()),
            }
        )
    fixed = int(np.sum(donors == np.arange(fold_array.size)))
    crossings = int(np.sum(fold_array[donors] != fold_array))
    same_union_group = int(np.sum(group_array[donors] == group_array))
    all_bijective = all(bool(row["bijective"]) for row in fold_reports)
    if (
        bool((donors < 0).any())
        or fixed
        or crossings
        or same_union_group
        or not all_bijective
    ):
        raise RuntimeError("B15 donor derangement integrity failed")
    report = {
        "seed": int(seed),
        "label_blind_construction": True,
        "fold_contained": True,
        "whole_prefix_block": True,
        "rows": int(donors.size),
        "fixed_points": fixed,
        "fold_crossings": crossings,
        "same_union_group_assignments": same_union_group,
        "all_folds_bijective": all_bijective,
        "donor_int64_sha256": _int64_sha(donors),
        "fold_reports": fold_reports,
    }
    return donors, report


def _synthetic_label_free_contract() -> dict[str, object]:
    generator = torch.Generator().manual_seed(DONOR_SEED + 31)
    tokens = torch.randn(3, TOKEN_COUNT, DESCRIPTOR_DIM, generator=generator)
    uniform, prefix_only, aligned, weights = _descriptor_details(tokens)
    raw_uniform = tokens[:, DINO_PREFIX_TOKENS:].mean(dim=1)
    raw_prefix = tokens[:, :DINO_PREFIX_TOKENS].mean(dim=1)
    prefixes = tokens[:, :DINO_PREFIX_TOKENS]
    patches = tokens[:, DINO_PREFIX_TOKENS:]
    q = F.layer_norm(
        prefixes - raw_uniform[:, None],
        (DESCRIPTOR_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )
    k = F.layer_norm(
        patches - raw_uniform[:, None],
        (DESCRIPTOR_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )
    direct_scores = (q[:, :, None] * k[:, None]).sum(dim=-1).sum(dim=1)
    direct_scores = direct_scores / (DINO_PREFIX_TOKENS * math.sqrt(DESCRIPTOR_DIM))
    direct_weights = torch.softmax(direct_scores, dim=1)
    direct_aligned = (direct_weights[:, :, None] * patches).sum(dim=1)

    prefix_permutation = torch.tensor([4, 2, 0, 3, 1], dtype=torch.long)
    permuted_prefix = prefixes[:, prefix_permutation]
    _u, _p, prefix_permuted_aligned, prefix_permuted_weights = _descriptor_details(
        tokens, prefix_tokens=permuted_prefix
    )
    patch_permutation = torch.randperm(PATCH_TOKENS, generator=generator)
    patch_permuted_tokens = torch.cat(
        (prefixes, patches[:, patch_permutation]), dim=1
    )
    _pu, _pp, patch_permuted_aligned, patch_permuted_weights = _descriptor_details(
        patch_permuted_tokens
    )

    zero_query_tokens = tokens.clone()
    zero_query_tokens[:, :DINO_PREFIX_TOKENS] = raw_uniform[:, None]
    zero_uniform, _zero_prefix, zero_aligned, zero_weights = _descriptor_details(
        zero_query_tokens
    )
    entropy = -(weights * torch.log(weights)).sum(dim=1)
    effective = torch.exp(entropy)
    donor_prefixes = torch.roll(prefixes, shifts=1, dims=0)
    donor_centers = torch.roll(raw_uniform, shifts=1, dims=0)
    _du, _dp, donor_aligned, _dw = _descriptor_details(
        tokens,
        prefix_tokens=donor_prefixes,
        prefix_center=donor_centers,
    )
    donor_q = F.layer_norm(
        donor_prefixes - donor_centers[:, None],
        (DESCRIPTOR_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )
    donor_scores = torch.einsum("bjd,bid->bji", donor_q, k).sum(dim=1)
    donor_scores = donor_scores / (DINO_PREFIX_TOKENS * math.sqrt(DESCRIPTOR_DIM))
    donor_weights = torch.softmax(donor_scores, dim=1)
    donor_direct = torch.einsum("bi,bid->bd", donor_weights, patches)
    donor_offset = torch.randn(3, 1, DESCRIPTOR_DIM, generator=generator)
    _ou, _op, offset_aligned, _ow = _descriptor_details(
        tokens,
        prefix_tokens=donor_prefixes + donor_offset,
        prefix_center=donor_centers + donor_offset[:, 0],
    )
    _fu, _fp, forbidden_current_center, _fw = _descriptor_details(
        tokens,
        prefix_tokens=donor_prefixes,
        prefix_center=raw_uniform,
    )
    reconstructed_prefixes = raw_uniform[:, None] + (
        donor_prefixes - donor_centers[:, None]
    )
    _ru, _rp, reconstructed_aligned, _rw = _descriptor_details(
        tokens,
        prefix_tokens=reconstructed_prefixes,
        prefix_center=raw_uniform,
    )

    folds = np.repeat(np.arange(FOLDS, dtype=np.int64), 4)
    groups = np.arange(folds.size, dtype=np.int64)
    donor_a, donor_report = build_fold_contained_donor_map(folds, groups)
    donor_b, _ = build_fold_contained_donor_map(folds, groups)
    checks = {
        "descriptor_geometry_exact": all(
            tuple(value.shape) == (3, DESCRIPTOR_DIM)
            for value in (uniform, prefix_only, aligned)
        ),
        "uniform_identity_exact": torch.equal(uniform, raw_uniform),
        "prefix_identity_exact": torch.equal(prefix_only, raw_prefix),
        "direct_formula_parity": torch.allclose(
            aligned, direct_aligned, rtol=1e-6, atol=1e-6
        ),
        "attention_rows_sum_one": torch.allclose(
            weights.sum(dim=1), torch.ones(weights.size(0)), rtol=0.0, atol=1e-6
        ),
        "prefix_order_permutation_invariant": torch.allclose(
            aligned, prefix_permuted_aligned, rtol=1e-6, atol=1e-6
        )
        and torch.allclose(weights, prefix_permuted_weights, rtol=1e-6, atol=1e-6),
        "joint_patch_weight_permutation_invariant": torch.allclose(
            aligned, patch_permuted_aligned, rtol=1e-6, atol=1e-6
        )
        and torch.allclose(
            weights[:, patch_permutation],
            patch_permuted_weights,
            rtol=1e-6,
            atol=1e-6,
        ),
        "uniform_query_weights_exact": torch.equal(
            zero_weights,
            torch.full_like(zero_weights, 1.0 / PATCH_TOKENS),
        ),
        "uniform_query_mean_pool_max_abs_lte_1e_6": float(
            torch.max(torch.abs(zero_aligned - zero_uniform))
        )
        <= 1e-6,
        "donor_centering_formula_parity": torch.allclose(
            donor_aligned, donor_direct, rtol=1e-6, atol=1e-6
        ),
        "donor_centered_reconstruction_parity": torch.allclose(
            donor_aligned, reconstructed_aligned, rtol=1e-6, atol=1e-6
        ),
        "donor_common_offset_invariant": torch.allclose(
            donor_aligned, offset_aligned, rtol=1e-6, atol=1e-6
        ),
        "forbidden_current_center_changes_output": not torch.allclose(
            donor_aligned, forbidden_current_center, rtol=1e-6, atol=1e-6
        ),
        "attention_entropy_finite_bounded": bool(
            torch.isfinite(entropy).all()
            and (entropy >= 0).all()
            and (entropy <= math.log(PATCH_TOKENS) + 1e-6).all()
        ),
        "effective_patch_count_finite_bounded": bool(
            torch.isfinite(effective).all()
            and (effective >= 1.0).all()
            and (effective <= PATCH_TOKENS + 1e-5).all()
        ),
        "donor_repeat_exact": np.array_equal(donor_a, donor_b),
        "donor_fixed_points_zero": donor_report["fixed_points"] == 0,
        "donor_fold_crossings_zero": donor_report["fold_crossings"] == 0,
        "donor_same_union_group_zero": donor_report["same_union_group_assignments"] == 0,
        "donor_bijective_all_folds": donor_report["all_folds_bijective"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"B15 synthetic label-free contract failed: {checks}")
    return {
        "formal_arrays_or_labels_read": False,
        "descriptor_signature": str(inspect.signature(_descriptor_arms)),
        "donor_signature": str(inspect.signature(build_fold_contained_donor_map)),
        "attention_entropy": entropy.tolist(),
        "effective_patch_count": effective.tolist(),
        "donor_report": donor_report,
        "checks": checks,
    }


def _operator_preflight(dino_weight: Path) -> dict[str, object]:
    uniform_model = _build_dino(dino_weight, num_classes=0).cpu().eval()
    aligned_model = _build_dino(dino_weight, num_classes=0).cpu().eval()
    model_contract = _dino_model_contract(uniform_model, dino_weight)
    if _dino_model_contract(aligned_model, dino_weight) != model_contract:
        raise RuntimeError("B15 side-by-side DINO model contracts differ")
    uniform = _DinoUniformDescriptor(uniform_model).eval()
    aligned = _DinoAlignedDescriptor(aligned_model).eval()
    pool = AlignedPrefixPatchPool()
    parameter_counts = {
        "operator": sum(parameter.numel() for parameter in pool.parameters()),
        "operator_buffers": sum(buffer.numel() for buffer in pool.buffers()),
        "uniform_wrapper": sum(parameter.numel() for parameter in uniform.parameters()),
        "aligned_wrapper": sum(parameter.numel() for parameter in aligned.parameters()),
    }
    torch.manual_seed(DONOR_SEED + 47)
    sample = torch.randn(1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)
    with torch.inference_mode():
        torch_outputs = {
            "uniform": uniform(sample).numpy(),
            "aligned": aligned(sample).numpy(),
        }
    if not all(np.isfinite(value).all() for value in torch_outputs.values()):
        raise FloatingPointError("B15 synthetic DINO descriptors are non-finite")

    with tempfile.TemporaryDirectory(prefix="trkh_b15_onnx_") as temporary:
        root = Path(temporary)
        paths = {name: root / f"{name}.onnx" for name in ("uniform", "aligned")}
        exports = {
            "uniform": _export_onnx(uniform, sample, paths["uniform"]),
            "aligned": _export_onnx(aligned, sample, paths["aligned"]),
        }
        sessions = {name: _ort_session(path) for name, path in paths.items()}
        array = sample.numpy()
        parity: dict[str, object] = {}
        for name in ("uniform", "aligned"):
            output = sessions[name].run(None, {"images": array})[0]
            parity[name] = {
                "max_abs": float(np.max(np.abs(output - torch_outputs[name]))),
                "argmax_equal": bool(
                    np.array_equal(output.argmax(axis=1), torch_outputs[name].argmax(axis=1))
                ),
            }
        for _ in range(ORT_WARMUPS):
            for name in ("uniform", "aligned"):
                sessions[name].run(None, {"images": array})
        timings = {name: [] for name in ("uniform", "aligned")}
        trial_means = {name: [] for name in timings}
        for trial in range(ORT_TRIALS):
            order = ("aligned", "uniform") if trial % 2 == 0 else ("uniform", "aligned")
            for name in order:
                samples = _timed_ort(sessions[name], array, ORT_ITERATIONS)
                timings[name].extend(samples)
                trial_means[name].append(float(np.mean(samples)))
        latency = {
            name: {
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
                "trial_mean_ms": trial_means[name],
            }
            for name, values in timings.items()
        }
    ratios = {
        "median": latency["aligned"]["median_ms"] / latency["uniform"]["median_ms"],
        "p95": latency["aligned"]["p95_ms"] / latency["uniform"]["p95_ms"],
    }
    checks = {
        "operator_parameters_zero": parameter_counts["operator"] == 0,
        "operator_buffers_zero": parameter_counts["operator_buffers"] == 0,
        "wrapper_parameter_counts_equal": parameter_counts["uniform_wrapper"]
        == parameter_counts["aligned_wrapper"],
        "descriptor_geometry_exact": all(
            value.shape == (1, DESCRIPTOR_DIM) for value in torch_outputs.values()
        ),
        "onnx_standard_domains_only": all(
            all(domain in {"", "ai.onnx"} for domain in export["operator_domains"])
            for export in exports.values()
        ),
        "onnx_parity_max_abs_lte_1e_5": all(
            float(value["max_abs"]) <= 1e-5 for value in parity.values()
        ),
        "onnx_argmax_exact": all(bool(value["argmax_equal"]) for value in parity.values()),
        "operator_p95_overhead_lte_5pct": ratios["p95"] <= 1.05,
    }
    return {
        "settings": {
            "runtime": "onnxruntime_cpu",
            "threads": ORT_THREADS,
            "batch": 1,
            "warmups": ORT_WARMUPS,
            "trials": ORT_TRIALS,
            "iterations_per_trial": ORT_ITERATIONS,
            "input": [1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
        },
        "model_contract": model_contract,
        "parameters": parameter_counts,
        "exports": exports,
        "parity": parity,
        "latency": latency,
        "ratios": ratios,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _preflight_checks(
    *,
    git: Mapping[str, object],
    synthetic_passed: bool,
    operator_passed: bool,
) -> dict[str, bool]:
    return {
        "canonical_branch_clean": git.get("branch") == EXPECTED_BRANCH
        and git.get("tracked_worktree_clean") is True,
        "protocol_and_sources_tracked": True,
        "runtime_exact": True,
        "dino_source_exact": True,
        "dino_weight_exact": True,
        "focused_tests_passed_without_skip": True,
        "synthetic_label_free_contract": bool(synthetic_passed),
        "operator_mobile_contract": bool(operator_passed),
        "train_descriptor_not_read": True,
        "validation_not_constructed": True,
        "test_not_constructed": True,
    }


def _validated_b15_output_path(output_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    runs_root = (_repository_root() / "runs").resolve()
    if output.parent != runs_root or not output.name.startswith(
        (PREFLIGHT_OUTPUT_PREFIX, FORMAL_OUTPUT_PREFIX)
    ):
        raise ValueError(
            "B15 output must be a fresh direct child of runs with a locked B15 prefix"
        )
    return output


def _validate_output_root(
    output_dir: Path,
    data_root: Path,
    *,
    expected_prefix: str,
) -> tuple[Path, Path]:
    output = _validated_b15_output_path(output_dir)
    if not output.name.startswith(expected_prefix):
        raise ValueError(f"B15 output does not match requested mode: {expected_prefix}")
    immutable = data_root.expanduser().resolve()
    try:
        output.relative_to(immutable)
    except ValueError:
        pass
    else:
        raise ValueError("B15 output must stay outside immutable class_f")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"Refuse to overwrite B15 output/partial: {output}")
    return output, partial


def build_preflight(args: argparse.Namespace) -> dict[str, object]:
    _assert_locked_locations(args, include_data=False)
    output, _partial = _validate_output_root(
        args.output_dir,
        _defaults()["data_root"],
        expected_prefix=PREFLIGHT_OUTPUT_PREFIX,
    )
    git = _git_contract()
    if git.get("branch") != EXPECTED_BRANCH or git.get("tracked_worktree_clean") is not True:
        raise RuntimeError(f"B15 preflight requires clean canonical branch: {git}")
    sources = _source_hashes()
    runtime = _runtime_contract()
    dino_source = _dino_source_contract()
    weight = _resolve_dino_weight(args.dino_weight)
    focused = _run_focused_tests()
    synthetic = _synthetic_label_free_contract()
    operator = _operator_preflight(weight)
    merged_operator_checks = {
        **synthetic["checks"],
        **operator["checks"],
    }
    checks = _preflight_checks(
        git=git,
        synthetic_passed=all(synthetic["checks"].values()),
        operator_passed=bool(operator["passed"]),
    )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "synthetic_preflight_no_dataset",
        "created_at_unix": time.time(),
        "source_hashes": sources,
        "git": git,
        "runtime": runtime,
        "dino_source": dino_source,
        "device": _locked_cuda_hardware(args.device)[1],
        "weight": {"path": str(weight), "sha256": DINO_WEIGHT_SHA256},
        "focused_tests": focused,
        "synthetic": synthetic,
        "operator": operator,
        "operator_checks": merged_operator_checks,
        "permissions": {
            "train_descriptor_read": False,
            "validation_not_constructed": True,
            "test_not_constructed": True,
            "validation_permission": False,
            "test_permission": False,
            "full_train_permission": False,
        },
        "checks": checks,
        "passed": all(checks.values()) and set(merged_operator_checks) == set(OPERATOR_CHECK_NAMES)
        and all(merged_operator_checks.values()),
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "preflight.json", payload)
    return payload


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _operator_payload_contract(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "settings",
        "model_contract",
        "parameters",
        "exports",
        "parity",
        "latency",
        "ratios",
        "checks",
        "passed",
    }:
        return False
    if value.get("passed") is not True:
        return False
    settings = value.get("settings")
    expected_settings = {
        "runtime": "onnxruntime_cpu",
        "threads": ORT_THREADS,
        "batch": 1,
        "warmups": ORT_WARMUPS,
        "trials": ORT_TRIALS,
        "iterations_per_trial": ORT_ITERATIONS,
        "input": [1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
    }
    if settings != expected_settings:
        return False
    if value.get("model_contract") != {
        "runtime_class": "timm.models.eva.Eva",
        "features": DESCRIPTOR_DIM,
        "prefix_tokens": DINO_PREFIX_TOKENS,
        "grid": [16, 16],
        "weight_bytes": EXPECTED_DINO_WEIGHT_BYTES,
        "weight_sha256": DINO_WEIGHT_SHA256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }:
        return False
    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "operator",
        "operator_buffers",
        "uniform_wrapper",
        "aligned_wrapper",
    }:
        return False
    if (
        parameters.get("operator") != 0
        or parameters.get("operator_buffers") != 0
        or parameters.get("uniform_wrapper") != parameters.get("aligned_wrapper")
        or not isinstance(parameters.get("uniform_wrapper"), int)
        or int(parameters.get("uniform_wrapper", 0)) <= 0
    ):
        return False
    exports = value.get("exports")
    parity = value.get("parity")
    latency = value.get("latency")
    ratios = value.get("ratios")
    if not all(isinstance(item, Mapping) for item in (exports, parity, latency, ratios)):
        return False
    if any(set(item) != {"uniform", "aligned"} for item in (exports, parity, latency)):
        return False
    for name in ("uniform", "aligned"):
        export = exports[name]
        parity_row = parity[name]
        latency_row = latency[name]
        if not isinstance(export, Mapping) or not (
            isinstance(export.get("bytes"), int)
            and int(export["bytes"]) > 0
            and isinstance(export.get("sha256"), str)
            and len(str(export["sha256"])) == 64
            and isinstance(export.get("operator_domains"), list)
            and all(domain in {"", "ai.onnx"} for domain in export["operator_domains"])
        ):
            return False
        if not isinstance(parity_row, Mapping) or not (
            isinstance(parity_row.get("max_abs"), (int, float))
            and float(parity_row["max_abs"]) <= 1e-5
            and parity_row.get("argmax_equal") is True
        ):
            return False
        if not isinstance(latency_row, Mapping) or not (
            _positive_finite(latency_row.get("median_ms"))
            and _positive_finite(latency_row.get("p95_ms"))
            and isinstance(latency_row.get("trial_mean_ms"), list)
            and len(latency_row["trial_mean_ms"]) == ORT_TRIALS
            and all(_positive_finite(sample) for sample in latency_row["trial_mean_ms"])
        ):
            return False
    if set(ratios) != {"median", "p95"} or not all(
        _positive_finite(ratios.get(name)) for name in ratios
    ):
        return False
    expected_median = float(latency["aligned"]["median_ms"]) / float(
        latency["uniform"]["median_ms"]
    )
    expected_p95 = float(latency["aligned"]["p95_ms"]) / float(
        latency["uniform"]["p95_ms"]
    )
    checks = value.get("checks")
    return bool(
        np.isclose(float(ratios["median"]), expected_median, rtol=0.0, atol=1e-12)
        and np.isclose(float(ratios["p95"]), expected_p95, rtol=0.0, atol=1e-12)
        and float(ratios["p95"]) <= 1.05
        and isinstance(checks, Mapping)
        and set(checks) == MOBILE_OPERATOR_CHECK_NAMES
        and all(result is True for result in checks.values())
    )


def _preflight_payload_structure_checks(payload: Mapping[str, object]) -> dict[str, bool]:
    permissions = {
        "train_descriptor_read": False,
        "validation_not_constructed": True,
        "test_not_constructed": True,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
    }
    focused = payload.get("focused_tests")
    synthetic = payload.get("synthetic")
    synthetic_checks = synthetic.get("checks") if isinstance(synthetic, Mapping) else None
    operator = payload.get("operator")
    nested_operator_checks = (
        operator.get("checks") if isinstance(operator, Mapping) else None
    )
    operator_checks = payload.get("operator_checks")
    nested_checks_match_top_level = bool(
        isinstance(synthetic_checks, Mapping)
        and isinstance(nested_operator_checks, Mapping)
        and isinstance(operator_checks, Mapping)
        and operator_checks == {**synthetic_checks, **nested_operator_checks}
    )
    expected_checks = _preflight_checks(
        git={"branch": EXPECTED_BRANCH, "tracked_worktree_clean": True},
        synthetic_passed=True,
        operator_passed=True,
    )
    return {
        "schema_exact": payload.get("schema_version") == 1,
        "protocol_exact": payload.get("protocol_id") == PROTOCOL_ID,
        "mode_exact": payload.get("mode") == "synthetic_preflight_no_dataset",
        "permissions_exact": payload.get("permissions") == permissions,
        "top_level_passed": payload.get("passed") is True,
        "checks_exact": payload.get("checks") == expected_checks,
        "focused_tests_exact": isinstance(focused, Mapping)
        and set(focused) == {"command", "returncode", "output"}
        and focused.get("command") == _focused_test_command()
        and focused.get("returncode") == 0
        and isinstance(focused.get("output"), str)
        and "passed" in str(focused.get("output")).casefold()
        and "skipped" not in str(focused.get("output")).casefold(),
        "synthetic_exact": isinstance(synthetic, Mapping)
        and synthetic.get("formal_arrays_or_labels_read") is False
        and isinstance(synthetic_checks, Mapping)
        and set(synthetic_checks) == SYNTHETIC_CHECK_NAMES
        and all(result is True for result in synthetic_checks.values()),
        "operator_contract_exact": _operator_payload_contract(operator),
        "operator_checks_exact": isinstance(operator_checks, Mapping)
        and set(operator_checks) == set(OPERATOR_CHECK_NAMES)
        and all(result is True for result in operator_checks.values())
        and nested_checks_match_top_level,
    }


def _validate_preflight(
    artifact: Path,
    expected_sha256: str,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    path = artifact.expanduser().resolve(strict=True)
    if path.name != "preflight.json":
        raise ValueError("Accepted B15 preflight must be named preflight.json")
    root = _validated_b15_output_path(path.parent)
    if not root.name.startswith(PREFLIGHT_OUTPUT_PREFIX):
        raise ValueError("Accepted B15 preflight must come from a preflight root")
    digest = _sha256(path)
    if digest != expected_sha256.casefold():
        raise ValueError("Accepted B15 preflight SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Accepted B15 preflight is not a mapping")
    weight = _resolve_dino_weight(args.dino_weight)
    checks = {
        **_preflight_payload_structure_checks(payload),
        "same_git": payload.get("git") == _git_contract(),
        "same_sources": payload.get("source_hashes") == _source_hashes(),
        "same_runtime": payload.get("runtime") == _runtime_contract(),
        "same_dino_source": payload.get("dino_source") == _dino_source_contract(),
        "same_device": payload.get("device") == _locked_cuda_hardware(args.device)[1],
        "same_weight": payload.get("weight")
        == {"path": str(weight), "sha256": DINO_WEIGHT_SHA256}
        and _sha256(weight) == DINO_WEIGHT_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Accepted B15 preflight no longer applies: {checks}")
    return {"artifact": str(path), "sha256": digest, "payload": dict(payload)}


def _verify_formal_evidence_anchors(
    b13_root: Path,
    b14_root: Path,
) -> dict[str, object]:
    b13 = b13_root.expanduser().resolve(strict=True)
    b14 = b14_root.expanduser().resolve(strict=True)
    expected_b13 = {
        "summary.json": B13_SUMMARY_SHA256,
        "train_oof_readouts.npz": B13_OOF_SHA256,
        "train_paths.json": B13_PATHS_SHA256,
        "train_dino_final_f32.npy": EXPECTED_DINO_DESCRIPTOR_SHA256,
    }
    observed_b13 = {}
    for name, expected in expected_b13.items():
        path = b13 / name
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"Locked B13 artifact changed: {path}")
        observed_b13[name] = digest
    observed_b14 = {}
    for name, expected in B14_EVIDENCE_HASHES.items():
        path = b14 / name
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"Locked B14 closure artifact changed: {path}")
        observed_b14[name] = digest
    repo = _repository_root()
    fixed_b14 = {
        repo
        / "runs"
        / "preflight_b14_iformer_s_frozen_transfer_c797a6f_r1"
        / "preflight.json": B14_ACCEPTED_PREFLIGHT_SHA256,
        repo
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B14_IFORMER_S_FROZEN_TRANSFER_PROTOCOL_20260804.md": B14_PROTOCOL_SHA256,
        repo
        / "trkh"
        / "tools"
        / "audit_iformer_s_frozen_transfer_b14.py": B14_RUNNER_SHA256,
        repo
        / "trkh"
        / "tools"
        / "audit_efficientvim_m1_frozen_transfer_b13.py": B13_RUNNER_SHA256,
    }
    fixed_hashes = {}
    for path, expected in fixed_b14.items():
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"Locked B13/B14 source/preflight anchor changed: {path}")
        fixed_hashes[str(path.relative_to(repo))] = digest
    return {
        "b13_root": str(b13),
        "b13_hashes": observed_b13,
        "b14_root": str(b14),
        "b14_hashes": observed_b14,
        "fixed_source_preflight_hashes": fixed_hashes,
    }


def _read_b13_fold_group_arrays(archive: Any) -> dict[str, np.ndarray]:
    return {
        "folds": np.asarray(archive["folds"], dtype=np.int64).copy(),
        "groups": np.asarray(archive["groups"], dtype=np.int64).copy(),
    }


def _read_b13_label_score_arrays(archive: Any) -> dict[str, np.ndarray]:
    return {
        "labels": np.asarray(archive["labels"], dtype=np.int64).copy(),
        "dino_scores": np.asarray(archive["dino_scores"], dtype=np.float64).copy(),
    }


def _load_fold_group_stage(anchor: Mapping[str, object]) -> dict[str, np.ndarray]:
    root = Path(str(anchor["b13_root"]))
    with np.load(root / "train_oof_readouts.npz", allow_pickle=False) as archive:
        arrays = _read_b13_fold_group_arrays(archive)
    folds = arrays["folds"]
    groups = arrays["groups"]
    if (
        folds.shape != (EXPECTED_TRAIN_ROWS,)
        or groups.shape != folds.shape
        or set(np.unique(folds).tolist()) != set(range(FOLDS))
        or _int64_sha(folds) != EXPECTED_FOLD_VECTOR_SHA256
        or _int64_sha(groups) != EXPECTED_GROUP_VECTOR_SHA256
    ):
        raise ValueError("Locked B13 fold/group vectors changed")
    return arrays


def _resolve_train_paths_without_labels(
    relative_paths: Sequence[str],
    data_root: Path,
    *,
    require_files: bool = True,
) -> list[Path]:
    root = data_root.expanduser().resolve()
    train_root = (root / "train").resolve()
    output: list[Path] = []
    for index, raw in enumerate(relative_paths):
        if not isinstance(raw, str):
            raise TypeError("B15 TRAIN path is not text")
        pure = PurePosixPath(raw.replace("\\", "/"))
        if (
            pure.is_absolute()
            or len(pure.parts) < 2
            or pure.parts[0] != "train"
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(f"B15 rejected non-TRAIN path at row {index}: {raw}")
        absolute = (root / Path(*pure.parts)).resolve()
        try:
            absolute.relative_to(train_root)
        except ValueError as error:
            raise ValueError(f"B15 TRAIN path escaped train root: {raw}") from error
        if require_files and not absolute.is_file():
            raise FileNotFoundError(f"B15 TRAIN image is missing: {absolute}")
        output.append(absolute)
    return output


def _load_path_only_stage(
    anchor: Mapping[str, object],
    data_root: Path,
) -> dict[str, object]:
    root = Path(str(anchor["b13_root"]))
    relative_paths = json.loads((root / "train_paths.json").read_text(encoding="utf-8"))
    if not isinstance(relative_paths, list) or len(relative_paths) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Locked B13 TRAIN path ledger changed")
    absolute_paths = _resolve_train_paths_without_labels(relative_paths, data_root)
    return {"relative_paths": relative_paths, "absolute_paths": absolute_paths}


def _load_label_score_stage(
    anchor: Mapping[str, object],
    folds: np.ndarray,
) -> dict[str, object]:
    root = Path(str(anchor["b13_root"]))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise TypeError("Locked B13 summary is not a mapping")
    _validate_b13_summary_payload(summary)
    with np.load(root / "train_oof_readouts.npz", allow_pickle=False) as archive:
        arrays = _read_b13_label_score_arrays(archive)
    labels = arrays["labels"]
    scores = arrays["dino_scores"]
    if (
        labels.shape != (EXPECTED_TRAIN_ROWS,)
        or scores.shape != (EXPECTED_TRAIN_ROWS, CLASSES)
        or not np.isfinite(scores).all()
        or tuple(np.bincount(labels, minlength=CLASSES).tolist())
        != EXPECTED_CLASS_COUNTS
    ):
        raise ValueError("Locked B13 labels/raw-DINO scores changed")
    metrics = _validate_dino_metrics(
        labels,
        folds,
        scores,
    )
    return {
        "labels": labels,
        "dino_scores": scores,
        "dino_metrics": metrics,
        "summary": dict(summary),
    }


def _open_descriptor_memmap(path: Path, rows: int) -> np.memmap:
    if path.exists():
        raise RuntimeError(f"Refuse to overwrite B15 descriptor: {path}")
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(int(rows), DESCRIPTOR_DIM),
    )


class _TrainPathDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, paths: Sequence[Path], transform: Any) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        with Image.open(self.paths[index]) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(rgb)
        return tensor, int(index)


def _ordered_loader(
    dataset: Dataset[tuple[Tensor, int]],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _extract_two_pass_descriptors(
    *,
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, int]],
    donors: np.ndarray,
    partial_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, object]:
    rows = len(dataset)
    donor_array = np.asarray(donors, dtype=np.int64)
    if donor_array.shape != (rows,):
        raise ValueError("B15 donor rows do not align with TRAIN ledger")
    paths = {
        arm: partial_dir / filename
        for arm, filename in RETAINED_DESCRIPTOR_FILENAMES.items()
    }
    arrays = {arm: _open_descriptor_memmap(path, rows) for arm, path in paths.items()}
    uniform_features = np.empty((rows, DESCRIPTOR_DIM), dtype=np.float32)
    prefixes = np.empty(
        (rows, DINO_PREFIX_TOKENS, DESCRIPTOR_DIM),
        dtype=np.float32,
    )
    loader = _ordered_loader(
        dataset,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )
    model.to(device).eval()
    uniform_patch_mean_bit_exact = True
    cursor = 0
    with torch.inference_mode():
        for images, indices in loader:
            index = indices.numpy().astype(np.int64, copy=False)
            expected = np.arange(cursor, cursor + index.size, dtype=np.int64)
            if not np.array_equal(index, expected):
                raise ValueError("B15 pass-1 DataLoader order changed")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            tokens = model.forward_features(images).float()
            _validate_tokens(tokens)
            descriptors = _descriptor_arms(tokens)
            raw_patch_mean = tokens[:, DINO_PREFIX_TOKENS:].mean(dim=1)
            uniform_patch_mean_bit_exact = uniform_patch_mean_bit_exact and torch.equal(
                descriptors["uniform"], raw_patch_mean
            )
            retained_values = {
                name: descriptors[name].cpu().numpy().astype(np.float32, copy=False)
                for name in ("prefix_only", "aligned")
            }
            uniform_values = (
                descriptors["uniform"].cpu().numpy().astype(np.float32, copy=False)
            )
            prefix_values = (
                tokens[:, :DINO_PREFIX_TOKENS]
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if uniform_values.shape != (index.size, DESCRIPTOR_DIM) or any(
                value.shape != (index.size, DESCRIPTOR_DIM)
                for value in retained_values.values()
            ):
                raise ValueError("B15 pass-1 descriptor shape changed")
            prefixes[index] = prefix_values
            uniform_features[index] = uniform_values
            for name, value in retained_values.items():
                arrays[name][index] = value
            cursor += index.size
    if cursor != rows or not uniform_patch_mean_bit_exact:
        raise RuntimeError("B15 pass-1 incomplete or uniform identity drifted")
    for name in ("prefix_only", "aligned"):
        arrays[name].flush()
    uniform_contract = _in_memory_npy_contract(uniform_features)

    cursor = 0
    with torch.inference_mode():
        for images, indices in loader:
            index = indices.numpy().astype(np.int64, copy=False)
            expected = np.arange(cursor, cursor + index.size, dtype=np.int64)
            if not np.array_equal(index, expected):
                raise ValueError("B15 pass-2 DataLoader order changed")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            tokens = model.forward_features(images).float()
            _validate_tokens(tokens)
            donor_prefix = torch.from_numpy(
                np.array(prefixes[donor_array[index]], dtype=np.float32, copy=True)
            ).to(device=device)
            donor_center = torch.from_numpy(
                np.array(
                    uniform_features[donor_array[index]],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(device=device)
            deranged = _descriptor_arms(
                tokens,
                prefix_tokens=donor_prefix,
                prefix_center=donor_center,
            )["aligned"]
            value = deranged.cpu().numpy().astype(np.float32, copy=False)
            if value.shape != (index.size, DESCRIPTOR_DIM):
                raise ValueError("B15 pass-2 descriptor shape changed")
            arrays["deranged"][index] = value
            cursor += index.size
    model.cpu()
    if cursor != rows:
        raise RuntimeError("B15 pass-2 descriptor extraction is incomplete")
    arrays["deranged"].flush()
    descriptor_hashes = {name: _sha256(path) for name, path in paths.items()}
    finite = np.isfinite(uniform_features).all() and all(
        np.isfinite(np.asarray(array)).all() for array in arrays.values()
    )
    prefix_ram_bytes = int(prefixes.nbytes)
    del prefixes
    del arrays
    gc.collect()
    if not finite:
        raise RuntimeError("B15 descriptor finiteness failed")
    return {
        "paths": paths,
        "descriptor_sha256_start": descriptor_hashes,
        "uniform_features": uniform_features,
        "uniform_npy_start": uniform_contract,
        "prefix_blocks_retained": False,
        "prefix_ram_bytes": prefix_ram_bytes,
        "uniform_patch_mean_bit_exact": uniform_patch_mean_bit_exact,
    }


def _component_members(
    folds: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    members: dict[int, np.ndarray] = {}
    fold_groups: dict[int, list[int]] = {fold: [] for fold in range(FOLDS)}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        observed = np.unique(folds[positions])
        if observed.size != 1:
            raise ValueError("B15 bootstrap component crosses folds")
        members[int(group)] = positions
        fold_groups[int(observed[0])].append(int(group))
    if any(not fold_groups[fold] for fold in range(FOLDS)):
        raise ValueError("B15 bootstrap fold has no component")
    return members, fold_groups


def _pair_aurocs(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    output = {}
    for rival in RIVALS:
        mask = (labels == 1) | (labels == rival)
        binary = (labels[mask] == 1).astype(np.int64)
        margin = scores[mask, 1] - scores[mask, rival]
        output[str(rival)] = float(roc_auc_score(binary, margin))
    return output


def paired_component_bootstrap_b15(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    scores: Mapping[str, np.ndarray],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = B13_SEED,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if labels.shape != folds.shape or labels.shape != groups.shape:
        raise ValueError("B15 bootstrap label/fold/group arrays are not aligned")
    expected_score_names = {"uniform", "prefix_only", "aligned"}
    if (
        set(scores) != expected_score_names
        or int(replicates) != BOOTSTRAP_REPLICATES
        or int(seed) != B13_SEED
    ):
        raise ValueError("B15 bootstrap arms/count/seed changed")
    score_arrays = {
        name: np.asarray(scores[name], dtype=np.float64)
        for name in ("uniform", "prefix_only", "aligned")
    }
    if any(
        value.shape != (labels.size, CLASSES) or not np.isfinite(value).all()
        for value in score_arrays.values()
    ):
        raise ValueError("B15 bootstrap scores are invalid")
    members, fold_groups = _component_members(folds, groups)
    predictions = {name: value.argmax(axis=1) for name, value in score_arrays.items()}
    values = {
        comparator: {
            metric: np.empty(int(replicates), dtype=np.float64)
            for metric in (
                "pair2_auroc_delta",
                "mean_pair_auroc_delta",
                "class1_f1_delta",
                "macro_f1_delta",
            )
        }
        for comparator in BOOTSTRAP_COMPARATORS
    }
    rng = np.random.default_rng(int(seed))
    draw_hash = hashlib.sha256()
    for replicate in range(int(replicates)):
        chunks: list[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draw_hash.update(np.asarray((replicate, fold), dtype="<i8").tobytes())
            draw_hash.update(draw.astype("<i8").tobytes())
            chunks.extend(members[available[int(position)]] for position in draw)
        selected = np.concatenate(chunks)
        aligned_macro, aligned_c1 = _fast_f1_from_predictions(
            labels[selected], predictions["aligned"][selected]
        )
        aligned_pairs = _pair_aurocs(labels[selected], score_arrays["aligned"][selected])
        for comparator in BOOTSTRAP_COMPARATORS:
            comparator_macro, comparator_c1 = _fast_f1_from_predictions(
                labels[selected], predictions[comparator][selected]
            )
            comparator_pairs = _pair_aurocs(
                labels[selected], score_arrays[comparator][selected]
            )
            values[comparator]["pair2_auroc_delta"][replicate] = (
                aligned_pairs[PAIR2] - comparator_pairs[PAIR2]
            )
            values[comparator]["mean_pair_auroc_delta"][replicate] = float(
                np.mean(list(aligned_pairs.values()))
                - np.mean(list(comparator_pairs.values()))
            )
            values[comparator]["class1_f1_delta"][replicate] = aligned_c1 - comparator_c1
            values[comparator]["macro_f1_delta"][replicate] = (
                aligned_macro - comparator_macro
            )
    finite_checks = {
        comparator: {
            metric: bool(
                samples.shape == (BOOTSTRAP_REPLICATES,)
                and np.isfinite(samples).all()
            )
            for metric, samples in comparator_values.items()
        }
        for comparator, comparator_values in values.items()
    }
    if not all(
        passed
        for comparator_checks in finite_checks.values()
        for passed in comparator_checks.values()
    ):
        raise FloatingPointError("B15 bootstrap gate replicate arrays are non-finite")
    intervals = {
        comparator: {
            metric: {
                "lower": float(np.quantile(samples, 0.025)),
                "upper": float(np.quantile(samples, 0.975)),
            }
            for metric, samples in comparator_values.items()
        }
        for comparator, comparator_values in values.items()
    }
    return {
        "method": "paired_fold_stratified_union_component_percentile_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "draws_int64_sha256": draw_hash.hexdigest(),
        "finite_gate_replicates": finite_checks,
        "all_gate_replicates_finite": True,
        "intervals": intervals,
        "replicate_arrays": values,
    }


def _uniform_parity(
    uniform_scores: np.ndarray,
    locked_scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, object]:
    observed = classification_summary(labels, uniform_scores, folds)
    maximum = float(np.max(np.abs(uniform_scores - locked_scores)))
    metric_errors = {
        name: abs(float(observed[name]) - float(EXPECTED_DINO_METRICS[name]))
        for name in EXPECTED_DINO_METRICS
    }
    checks = {
        "score_max_abs_lte_1e_8": maximum <= 1e-8,
        "argmax_exact": np.array_equal(
            uniform_scores.argmax(axis=1), locked_scores.argmax(axis=1)
        ),
        "metrics_max_abs_lte_1e_12": max(metric_errors.values()) <= 1e-12,
    }
    return {
        "score_max_abs": maximum,
        "metric_abs_errors": metric_errors,
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": observed,
    }


def _fold_pair_wins(
    metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    aligned = {int(row["fold"]): row for row in metrics["aligned"]["folds"]}
    uniform = {int(row["fold"]): row for row in metrics["uniform"]["folds"]}
    rows = []
    for fold in range(FOLDS):
        for rival in RIVALS:
            delta = float(aligned[fold]["pairs"][str(rival)]["auroc"]) - float(
                uniform[fold]["pairs"][str(rival)]["auroc"]
            )
            rows.append({"fold": fold, "rival": rival, "delta": delta, "win": delta > 0.0})
    return {
        "rows": rows,
        "pair2_wins": sum(row["win"] for row in rows if row["rival"] == 2),
        "all_pair_fold_wins": sum(row["win"] for row in rows),
    }


def assess_b15_gate(
    *,
    metrics: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, object],
    uniform_parity: Mapping[str, object],
    fold_wins: Mapping[str, object],
    readouts_converged: bool,
    integrity_complete: bool,
) -> dict[str, object]:
    aligned = metrics["aligned"]
    uniform = metrics["uniform"]
    intervals = bootstrap["intervals"]
    point = {
        "pair2_auroc_delta_vs_uniform": float(aligned["pairs"][PAIR2]["auroc"])
        - float(uniform["pairs"][PAIR2]["auroc"]),
        "mean_pair_auroc_delta_vs_uniform": float(aligned["mean_pair_auroc"])
        - float(uniform["mean_pair_auroc"]),
        "class1_f1_delta_vs_uniform": float(aligned["class1_f1"])
        - float(uniform["class1_f1"]),
        "macro_f1_delta_vs_uniform": float(aligned["macro_f1"])
        - float(uniform["macro_f1"]),
        "class1_recall_delta_vs_uniform": float(aligned["class1_recall"])
        - float(uniform["class1_recall"]),
        "pair2_auroc_delta_vs_deranged": float(aligned["pairs"][PAIR2]["auroc"])
        - float(metrics["deranged"]["pairs"][PAIR2]["auroc"]),
        "one_to_two_delta": int(aligned["confusion_matrix"][1][2])
        - int(uniform["confusion_matrix"][1][2]),
        "two_to_one_ratio": (
            int(aligned["confusion_matrix"][2][1])
            / int(uniform["confusion_matrix"][2][1])
            if int(uniform["confusion_matrix"][2][1]) > 0
            else float("inf")
        ),
    }
    uniform_ci = intervals["uniform"]
    checks = {
        "uniform_reproduces_b13": uniform_parity.get("passed") is True,
        "pair2_gain_gte_0p010": point["pair2_auroc_delta_vs_uniform"] >= 0.010,
        "pair2_lcb_gt_zero_vs_uniform": float(uniform_ci["pair2_auroc_delta"]["lower"])
        > 0.0,
        "mean_pair_gain_gte_0p003": point["mean_pair_auroc_delta_vs_uniform"] >= 0.003,
        "mean_pair_lcb_gt_zero": float(uniform_ci["mean_pair_auroc_delta"]["lower"])
        > 0.0,
        "class1_f1_gain_gte_0p010": point["class1_f1_delta_vs_uniform"] >= 0.010,
        "class1_f1_lcb_gt_zero": float(uniform_ci["class1_f1_delta"]["lower"])
        > 0.0,
        "macro_f1_lcb_gte_minus_0p002": float(uniform_ci["macro_f1_delta"]["lower"])
        >= -0.002,
        "class1_recall_delta_gte_minus_0p005": point["class1_recall_delta_vs_uniform"]
        >= -0.005,
        "one_to_two_not_increased": point["one_to_two_delta"] <= 0,
        "two_to_one_reduced_at_least_10pct": int(aligned["confusion_matrix"][2][1])
        <= 0.90 * int(uniform["confusion_matrix"][2][1])
        and int(uniform["confusion_matrix"][2][1]) > 0,
        "pair2_lcb_gt_zero_vs_prefix_only": float(
            intervals["prefix_only"]["pair2_auroc_delta"]["lower"]
        )
        > 0.0,
        "pair2_point_gt_deranged": point["pair2_auroc_delta_vs_deranged"] > 0.0,
        "pair2_fold_wins_gte_4_of_5": int(fold_wins["pair2_wins"]) >= 4,
        "all_pair_fold_wins_gte_10_of_15": int(fold_wins["all_pair_fold_wins"]) >= 10,
        "bootstrap_gate_arrays_finite": bootstrap.get(
            "all_gate_replicates_finite"
        )
        is True,
        "readouts_converged": bool(readouts_converged),
        "integrity_complete_train_only": bool(integrity_complete),
    }
    passed = all(checks.values())
    return {
        "signal_gate_passed": passed,
        "exact_aligned_prefix_patch_pool_route_closed": not passed,
        "next_protocol_permission": passed,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "point_deltas": point,
    }


def _output_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _atomic_bytes(path: Path, payload: bytes) -> str:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"Refuse to overwrite B15 byte artifact: {path}")
    partial.write_bytes(payload)
    digest = _sha256(partial)
    partial.replace(path)
    if _sha256(path) != digest:
        raise RuntimeError(f"B15 byte artifact changed during promotion: {path}")
    return digest


def _fixed_point_json_bytes(
    payload: dict[str, object],
    *,
    base_bytes: int,
    set_final_total: Any,
) -> tuple[bytes, int]:
    projected = int(base_bytes)
    for _ in range(32):
        set_final_total(payload, projected)
        serialized = json.dumps(payload, indent=2).encode("utf-8")
        updated = int(base_bytes) + len(serialized)
        if updated == projected:
            set_final_total(payload, updated)
            final = json.dumps(payload, indent=2).encode("utf-8")
            if int(base_bytes) + len(final) != updated:
                raise RuntimeError("B15 JSON final-byte projection lost its fixed point")
            return final, updated
        projected = updated
    raise RuntimeError("B15 JSON final-byte projection did not converge")


def _timed_fixed_point_json_bytes(
    payload: dict[str, object],
    *,
    base_bytes: int,
    started: float,
    set_final_total: Any,
    set_wall_seconds: Any,
    clock: Any = time.perf_counter,
) -> tuple[bytes, int, float]:
    """Serialize only when recorded wall time covers the post-serialization clock."""
    for _ in range(32):
        elapsed_before = max(0.0, float(clock()) - float(started))
        wall_upper_bound = float(math.ceil(elapsed_before))
        set_wall_seconds(payload, wall_upper_bound)
        encoded, total = _fixed_point_json_bytes(
            payload,
            base_bytes=base_bytes,
            set_final_total=set_final_total,
        )
        elapsed_after = max(0.0, float(clock()) - float(started))
        if elapsed_after <= wall_upper_bound:
            return encoded, total, wall_upper_bound
    raise RuntimeError("B15 wall-time/JSON serialization bound did not converge")


def _locked_cuda_hardware(requested_name: str) -> tuple[torch.device, dict[str, object]]:
    requested = _resolve_device(requested_name)
    if requested.type != "cuda":
        raise RuntimeError("B15 requires the locked cuda:0 device")
    index = int(requested.index if requested.index is not None else torch.cuda.current_device())
    if index != 0:
        raise RuntimeError("B15 formal descriptor extraction requires CUDA device zero")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "device": str(device),
        "name": str(properties.name),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": int(properties.total_memory),
    }
    expected_hardware = {
        "device": "cuda:0",
        "name": EXPECTED_CUDA_NAME,
        "capability": list(EXPECTED_CUDA_CAPABILITY),
        "total_memory": EXPECTED_CUDA_MEMORY_BYTES,
    }
    if hardware != expected_hardware:
        raise RuntimeError(f"B15 locked CUDA hardware changed: {hardware}")
    return device, hardware


def _configure_formal_runtime(args: argparse.Namespace) -> tuple[torch.device, dict[str, object]]:
    if int(args.torch_threads) != 4:
        raise ValueError("B15 formal execution is locked to four torch CPU threads")
    device, hardware = _locked_cuda_hardware(args.device)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("B15 CUBLAS_WORKSPACE_CONFIG changed before formal CUDA use")
    random.seed(DONOR_SEED)
    np.random.seed(DONOR_SEED)
    torch.manual_seed(DONOR_SEED)
    torch.cuda.manual_seed_all(DONOR_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    settings = {
        "mechanism_seed": DONOR_SEED,
        "readout_bootstrap_seed": B13_SEED,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_threads": int(torch.get_num_threads()),
        "autocast_used": False,
        "hardware": hardware,
    }
    expected_settings = {
        "mechanism_seed": DONOR_SEED,
        "readout_bootstrap_seed": B13_SEED,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "torch_threads": 4,
        "autocast_used": False,
        "hardware": hardware,
    }
    if settings != expected_settings:
        raise RuntimeError(f"B15 deterministic settings changed: {settings}")
    return device, settings


def _run_formal(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    _assert_locked_locations(args, include_data=True)
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B15 requires accepted preflight path and SHA-256")
    if args.batch_size != BATCH_SIZE or args.workers != WORKERS:
        raise ValueError("B15 is locked to batch-size=32 and workers=0")
    device, determinism = _configure_formal_runtime(args)
    torch.cuda.reset_peak_memory_stats(device)
    data_yaml = args.data_yaml.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    if _sha256(data_yaml) != EXPECTED_DATA_YAML_SHA256:
        raise ValueError("Canonical class_f data YAML bytes changed")
    output, partial = _validate_output_root(
        args.output_dir,
        data_root,
        expected_prefix=FORMAL_OUTPUT_PREFIX,
    )
    accepted = _validate_preflight(
        args.preflight_artifact,
        args.preflight_sha256,
        args=args,
    )
    git_start = _git_contract()
    sources_start = _source_hashes()
    runtime_start = _runtime_contract()
    dino_source_start = _dino_source_contract()
    weight = _resolve_dino_weight(args.dino_weight)
    anchors = _verify_formal_evidence_anchors(args.b13_root, args.b14_root)
    fold_group = _load_fold_group_stage(anchors)
    folds = fold_group["folds"]
    groups = fold_group["groups"]
    partial.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        partial / "run_owner.json",
        {
            "protocol_id": PROTOCOL_ID,
            "intended_output": str(output),
            "git_head": git_start["head"],
        },
    )
    donors, donor_report = build_fold_contained_donor_map(folds, groups)
    donor_path = partial / "donor_indices_i64.npy"
    donor_sha_start = _atomic_npy(donor_path, donors)
    if donor_sha_start != _sha256(donor_path):
        raise RuntimeError("B15 donor vector changed during atomic persistence")
    _atomic_json(partial / "donor_report.json", donor_report)

    # Only the locked TRAIN path ledger is opened here. Labels, scores and the
    # metric-bearing B13 summary remain unopened until all descriptors freeze.
    path_stage = _load_path_only_stage(anchors, data_root)
    relative_paths = list(path_stage["relative_paths"])
    absolute_paths = list(path_stage["absolute_paths"])
    train_content_start = _train_content_contract(relative_paths, absolute_paths)
    if train_content_start["sha256"] != EXPECTED_TRAIN_CONTENT_SHA256:
        raise ValueError("Canonical TRAIN content bytes changed")

    model = _build_dino(weight, num_classes=0)
    model_contract = _dino_model_contract(model, weight)
    dataset = _TrainPathDataset(absolute_paths, _dino_transform(model))
    train_paths_path = partial / "train_paths.json"
    _atomic_json(train_paths_path, relative_paths)
    train_paths_sha_start = _sha256(train_paths_path)
    if train_paths_sha_start != B13_PATHS_SHA256:
        raise RuntimeError("B15 TRAIN path artifact failed locked B13 byte parity")
    extraction = _extract_two_pass_descriptors(
        model=model,
        dataset=dataset,
        donors=donors,
        partial_dir=partial,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if extraction["uniform_npy_start"]["sha256"] != EXPECTED_DINO_DESCRIPTOR_SHA256:
        raise RuntimeError("B15 fresh uniform descriptor failed locked B13 byte parity")
    _atomic_json(
        partial / "descriptor_start_hashes.json",
        {
            "retained_descriptors": extraction["descriptor_sha256_start"],
            "uniform_in_memory_npy": extraction["uniform_npy_start"],
            "donor_indices_i64": donor_sha_start,
            "train_paths": train_paths_sha_start,
        },
    )

    data_stage = _load_label_score_stage(anchors, folds)
    labels = np.asarray(data_stage["labels"], dtype=np.int64)
    locked_scores = np.asarray(data_stage["dino_scores"], dtype=np.float64)
    label_validated_paths = _resolve_train_paths(relative_paths, labels, data_root)
    if label_validated_paths != absolute_paths:
        raise RuntimeError("B15 label/class-directory path validation changed order")
    pre_metric_descriptor_hashes = {
        arm: _sha256(path) for arm, path in extraction["paths"].items()
    }
    pre_metric_uniform_npy = _in_memory_npy_contract(
        extraction["uniform_features"]
    )
    pre_metric_hashes_stable = bool(
        pre_metric_descriptor_hashes == extraction["descriptor_sha256_start"]
        and pre_metric_uniform_npy == extraction["uniform_npy_start"]
        and _sha256(donor_path) == donor_sha_start
        and _sha256(train_paths_path)
        == train_paths_sha_start
        == B13_PATHS_SHA256
    )
    if not pre_metric_hashes_stable:
        raise RuntimeError("B15 retained artifact changed before metric construction")
    pre_metric_current_bytes = _output_bytes(partial)
    pre_metric_elapsed = time.perf_counter() - started
    pre_metric_peak_vram = int(torch.cuda.max_memory_allocated(device))
    pre_metric_checks = {
        "elapsed_seconds_lte_900": pre_metric_elapsed <= WALL_BUDGET_SECONDS,
        "peak_vram_lte_6gib": pre_metric_peak_vram <= VRAM_BUDGET_BYTES,
        "current_plus_16mib_reserve_lte_100mib": pre_metric_current_bytes
        + PRE_METRIC_ARTIFACT_RESERVE_BYTES
        <= ARTIFACT_BUDGET_BYTES,
        "retained_start_hashes_stable": pre_metric_hashes_stable,
    }
    pre_metric_gate = {
        "elapsed_seconds": pre_metric_elapsed,
        "peak_vram_bytes": pre_metric_peak_vram,
        "current_artifact_bytes": pre_metric_current_bytes,
        "reserve_bytes": PRE_METRIC_ARTIFACT_RESERVE_BYTES,
        "current_plus_reserve_bytes": pre_metric_current_bytes
        + PRE_METRIC_ARTIFACT_RESERVE_BYTES,
        "descriptor_sha256": pre_metric_descriptor_hashes,
        "uniform_in_memory_npy": pre_metric_uniform_npy,
        "checks": pre_metric_checks,
        "passed": all(pre_metric_checks.values()),
        "metric_boundary_crossed": False,
    }
    _atomic_json(partial / "pre_metric_resource_gate.json", pre_metric_gate)
    if not pre_metric_gate["passed"]:
        raise RuntimeError("B15 pre-metric resource gate failed")
    post_gate_elapsed = time.perf_counter() - started
    post_gate_peak_vram = int(torch.cuda.max_memory_allocated(device))
    post_gate_artifact_bytes = _output_bytes(partial)
    post_gate_checks = {
        "elapsed_seconds_lte_900": post_gate_elapsed <= WALL_BUDGET_SECONDS,
        "peak_vram_lte_6gib": post_gate_peak_vram <= VRAM_BUDGET_BYTES,
        "current_plus_16mib_reserve_lte_100mib": post_gate_artifact_bytes
        + PRE_METRIC_ARTIFACT_RESERVE_BYTES
        <= ARTIFACT_BUDGET_BYTES,
        "retained_start_hashes_stable": pre_metric_hashes_stable,
    }
    if not all(post_gate_checks.values()):
        raise RuntimeError(
            "B15 pre-metric post-gate resource check failed: "
            f"elapsed={post_gate_elapsed}, peak={post_gate_peak_vram}, "
            f"bytes={post_gate_artifact_bytes}, checks={post_gate_checks}"
        )
    _atomic_json(
        partial / "stage_metric_construction_started.json",
        {
            "metric_boundary_crossed": True,
            "retry_allowed": False,
            "post_gate_elapsed_seconds": post_gate_elapsed,
            "post_gate_peak_vram_bytes": post_gate_peak_vram,
            "post_gate_artifact_bytes": post_gate_artifact_bytes,
            "post_gate_resource_checks": post_gate_checks,
            "reason": "Fail-closed before uniform OOF parity and candidate readouts.",
        },
    )
    features: dict[str, np.ndarray] = {
        "uniform": np.asarray(extraction["uniform_features"], dtype=np.float32),
        **{
            arm: np.load(extraction["paths"][arm], mmap_mode="r", allow_pickle=False)
            for arm in RETAINED_DESCRIPTOR_ARMS
        },
    }
    readouts: dict[str, dict[str, object]] = {}
    readouts["uniform"] = fit_oof_readout(features["uniform"], labels, folds)
    uniform_scores = np.asarray(readouts["uniform"]["scores"], dtype=np.float64)
    uniform_parity = _uniform_parity(uniform_scores, locked_scores, labels, folds)
    if not uniform_parity["passed"]:
        _atomic_json(partial / "uniform_parity_failure.json", uniform_parity)
        raise RuntimeError("B15 uniform OOF failed locked B13 infrastructure parity")
    for arm in ("prefix_only", "aligned", "deranged"):
        readouts[arm] = fit_oof_readout(features[arm], labels, folds)
    score_arrays = {
        arm: np.asarray(readouts[arm]["scores"], dtype=np.float64) for arm in ARMS
    }
    score_arrays["uniform"] = locked_scores.copy()
    metrics = {
        arm: classification_summary(labels, score_arrays[arm], folds) for arm in ARMS
    }
    _atomic_json(
        partial / "stage_metrics_constructed.json",
        {
            "metrics_constructed": True,
            "retry_allowed": False,
            "reason": "The preregistered B15 gate is final once all arm metrics exist.",
        },
    )
    bootstrap = paired_component_bootstrap_b15(
        labels=labels,
        folds=folds,
        groups=groups,
        scores={
            name: score_arrays[name] for name in ("uniform", "prefix_only", "aligned")
        },
    )
    bootstrap_replicates = bootstrap.pop("replicate_arrays")
    if bootstrap["draws_int64_sha256"] != EXPECTED_BOOTSTRAP_DRAW_SHA256:
        raise ValueError("B15 bootstrap draw vector changed")
    fold_wins = _fold_pair_wins(metrics)
    oof_arrays: dict[str, np.ndarray] = {
        "labels": labels,
        "folds": folds,
        "groups": groups,
        "donors": donors,
    }
    for arm in ARMS:
        oof_arrays[f"{arm}_scores"] = score_arrays[arm]
        oof_arrays[f"{arm}_coefficients"] = np.asarray(readouts[arm]["coefficients"])
        oof_arrays[f"{arm}_intercepts"] = np.asarray(readouts[arm]["intercepts"])
        oof_arrays[f"{arm}_scaler_means"] = np.asarray(readouts[arm]["scaler_means"])
        oof_arrays[f"{arm}_scaler_scales"] = np.asarray(readouts[arm]["scaler_scales"])
    for comparator, comparator_values in bootstrap_replicates.items():
        for metric_name, replicate_values in comparator_values.items():
            oof_arrays[
                f"bootstrap_aligned_minus_{comparator}_{metric_name}"
            ] = np.asarray(replicate_values, dtype=np.float64)
    oof_sha = _atomic_npz(partial / "train_oof_readouts.npz", **oof_arrays)

    descriptor_sha_end = {
        arm: _sha256(path) for arm, path in extraction["paths"].items()
    }
    uniform_npy_end = _in_memory_npy_contract(features["uniform"])
    donor_sha_end = _sha256(donor_path)
    train_paths_sha_end = _sha256(train_paths_path)
    train_content_end = _train_content_contract(relative_paths, absolute_paths)
    git_end = _git_contract()
    sources_end = _source_hashes()
    runtime_end = _runtime_contract()
    dino_source_end = _dino_source_contract()
    model_contract_end = _dino_model_contract(model, weight)
    anchors_end = _verify_formal_evidence_anchors(args.b13_root, args.b14_root)
    retained_hashes_stable = (
        descriptor_sha_end == extraction["descriptor_sha256_start"]
        and donor_sha_end == donor_sha_start
        and train_paths_sha_end == train_paths_sha_start == B13_PATHS_SHA256
        and uniform_npy_end == extraction["uniform_npy_start"]
        and uniform_npy_end["sha256"] == EXPECTED_DINO_DESCRIPTOR_SHA256
    )
    integrity_complete_raw = bool(
        extraction["uniform_patch_mean_bit_exact"]
        and extraction["prefix_blocks_retained"] is False
        and retained_hashes_stable
        and uniform_parity["passed"]
        and bootstrap["all_gate_replicates_finite"] is True
        and all(np.isfinite(scores).all() for scores in score_arrays.values())
        and train_content_end == train_content_start
        and git_end == git_start
        and git_end["tracked_worktree_clean"] is True
        and sources_end == sources_start
        and runtime_end == runtime_start
        and dino_source_end == dino_source_start
        and model_contract_end == model_contract
        and anchors_end == anchors
        and _sha256(weight) == DINO_WEIGHT_SHA256
    )
    scientific_gate = assess_b15_gate(
        metrics=metrics,
        bootstrap=bootstrap,
        uniform_parity=uniform_parity,
        fold_wins=fold_wins,
        readouts_converged=all(bool(readouts[arm]["converged"]) for arm in ARMS),
        integrity_complete=integrity_complete_raw,
    )
    del features
    extraction.pop("uniform_features")
    gc.collect()

    peak_vram = int(torch.cuda.max_memory_allocated(device))
    artifact_bytes_before_terminal = _output_bytes(partial)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_frozen_parameter_free_pool_screen",
        "source_hashes": {"start": sources_start, "end": sources_end},
        "git": {"start": git_start, "end": git_end},
        "runtime": {"start": runtime_start, "end": runtime_end},
        "dino_source": {"start": dino_source_start, "end": dino_source_end},
        "dino_model": {"start": model_contract, "end": model_contract_end},
        "determinism": determinism,
        "accepted_preflight": {
            "artifact": accepted["artifact"],
            "sha256": accepted["sha256"],
        },
        "weight": accepted["payload"]["weight"],
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": EXPECTED_DATA_YAML_SHA256,
            "data_root": str(data_root),
            "train_rows": EXPECTED_TRAIN_ROWS,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "validation_not_constructed": True,
            "test_not_constructed": True,
            "train_content": {"start": train_content_start, "end": train_content_end},
        },
        "preprocessing": {
            "exif_transpose": True,
            "rgb": True,
            "resize": [DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
            "interpolation": "bicubic",
            "antialias": True,
            "dtype": "float32",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "autocast": False,
        },
        "comparator": {
            "root": anchors["b13_root"],
            "hashes": anchors["b13_hashes"],
            "allowed_arrays": ["labels", "folds", "groups", "dino_scores"],
            "efficientvim_scores_read": False,
            "iformer_scores_read": False,
            "b9_logits_read": False,
        },
        "closure_anchors": {
            "b14_root": anchors["b14_root"],
            "b14_hashes": anchors["b14_hashes"],
            "fixed_source_preflight_hashes": anchors[
                "fixed_source_preflight_hashes"
            ],
        },
        "operator": {
            "input": "final_postnorm_tokens_[B,261,384]",
            "prefix_tokens": DINO_PREFIX_TOKENS,
            "patch_tokens": PATCH_TOKENS,
            "dimension": DESCRIPTOR_DIM,
            "layer_norm_eps": LAYER_NORM_EPS,
            "tokens_already_final_postnorm": True,
            "second_final_norm_applied": False,
            "deranged_prefix_center": "donor_uniform_patch_mean",
            "deranged_patch_center_values": "current_row",
            "temperature_or_blend": None,
            "trainable_parameters": 0,
            "arms": list(ARMS),
        },
        "donor": donor_report,
        "donor_file_sha256": {"start": donor_sha_start, "end": donor_sha_end},
        "train_paths_sha256": {
            "start": train_paths_sha_start,
            "end": train_paths_sha_end,
        },
        "descriptor_sha256": {
            "start": extraction["descriptor_sha256_start"],
            "end": descriptor_sha_end,
        },
        "uniform_in_memory_npy": {
            "start": extraction["uniform_npy_start"],
            "end": uniform_npy_end,
        },
        "retained_descriptor_arms": list(RETAINED_DESCRIPTOR_ARMS),
        "uniform_descriptor_retained": False,
        "prefix_blocks_retained": extraction["prefix_blocks_retained"],
        "prefix_ram_bytes": extraction["prefix_ram_bytes"],
        "uniform_patch_mean_bit_exact": extraction["uniform_patch_mean_bit_exact"],
        "uniform_parity": uniform_parity,
        "readout": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "tol": 1e-8,
            "max_iter": 2000,
            "random_state": B13_SEED,
            "fold_records": {arm: readouts[arm]["fold_records"] for arm in ARMS},
        },
        "metrics": metrics,
        "fold_pair_wins": fold_wins,
        "bootstrap": bootstrap,
        "gate": {},
        "oof_sha256": oof_sha,
        "integrity_complete": False,
        "operational_budget": {
            "limits": {
                "wall_seconds": WALL_BUDGET_SECONDS,
                "peak_vram_bytes": VRAM_BUDGET_BYTES,
                "artifact_bytes": ARTIFACT_BUDGET_BYTES,
            },
            "observed": {
                "wall_seconds": 0.0,
                "peak_vram_bytes": peak_vram,
                "artifact_bytes_before_summary": artifact_bytes_before_terminal,
                "final_total_bytes": 0,
            },
            "checks": {},
        },
        "execution_status": "PENDING_FINAL_BYTE_PROJECTION",
        "execution_valid": False,
        "permissions": {
            "next_protocol_permission": False,
            "validation_permission": False,
            "test_permission": False,
            "full_train_permission": False,
            "xai_claim_permission": False,
        },
    }

    def set_summary_total(payload: dict[str, object], final_total: int) -> None:
        recorded_wall_seconds = float(
            payload["operational_budget"]["observed"]["wall_seconds"]
        )
        resource_checks = {
            "wall_seconds_lte_900": recorded_wall_seconds
            <= WALL_BUDGET_SECONDS,
            "peak_vram_lte_6gib": peak_vram <= VRAM_BUDGET_BYTES,
            "final_total_bytes_lte_100mib": int(final_total)
            <= ARTIFACT_BUDGET_BYTES,
        }
        final_valid = bool(integrity_complete_raw and all(resource_checks.values()))
        final_gate = json.loads(json.dumps(scientific_gate))
        final_gate["scientific_gate_evaluable"] = final_valid
        final_gate["next_protocol_permission"] = bool(
            final_valid and final_gate["signal_gate_passed"]
        )
        final_gate["execution_status"] = (
            "SCIENTIFIC_GATE_EVALUATED"
            if final_valid
            else "PENDING_TERMINAL_OPERATIONAL_FAILURE"
        )
        if not final_valid:
            final_gate["signal_gate_passed"] = False
            final_gate["exact_aligned_prefix_patch_pool_route_closed"] = True
        payload["gate"] = final_gate
        payload["integrity_complete"] = final_valid
        payload["execution_status"] = (
            "VALID_SCIENTIFIC_GATE_EVALUATION"
            if final_valid
            else "PENDING_TERMINAL_OPERATIONAL_FAILURE"
        )
        payload["execution_valid"] = final_valid
        payload["permissions"] = {
            "next_protocol_permission": bool(
                final_valid and final_gate["signal_gate_passed"]
            ),
            "validation_permission": False,
            "test_permission": False,
            "full_train_permission": False,
            "xai_claim_permission": False,
        }
        payload["operational_budget"]["observed"]["final_total_bytes"] = int(
            final_total
        )
        payload["operational_budget"]["checks"] = resource_checks

    def set_summary_wall(payload: dict[str, object], seconds: float) -> None:
        payload["operational_budget"]["observed"]["wall_seconds"] = float(
            seconds
        )

    summary_bytes, projected_summary_total, final_wall_upper_bound = (
        _timed_fixed_point_json_bytes(
            summary,
            base_bytes=artifact_bytes_before_terminal,
            set_final_total=set_summary_total,
            started=started,
            set_wall_seconds=set_summary_wall,
        )
    )
    if summary["execution_valid"] is not True:
        failure_reasons = [
            name
            for name, passed in {
                "final_integrity_complete": integrity_complete_raw,
                **summary["operational_budget"]["checks"],
            }.items()
            if not passed
        ]
        terminal = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "REGISTERED_OPERATIONAL_FAILURE",
            "execution_status": "TERMINAL_OPERATIONAL_ROUTE_CLOSED",
            "metric_boundary_crossed": True,
            "retry_allowed": False,
            "summary_written": False,
            "candidate_summary_projected_bytes": projected_summary_total,
            "failure_reasons": failure_reasons,
            "exact_aligned_prefix_patch_pool_route_closed_operationally": True,
            "representation_quality_conclusion": "NOT_PERMITTED",
            "statement": (
                "This registered operational failure is not evidence against "
                "representation quality."
            ),
            "operational_budget": {
                "limits": {
                    "wall_seconds": WALL_BUDGET_SECONDS,
                    "peak_vram_bytes": VRAM_BUDGET_BYTES,
                    "artifact_bytes": ARTIFACT_BUDGET_BYTES,
                },
                "observed": {
                    "wall_seconds": final_wall_upper_bound,
                    "peak_vram_bytes": peak_vram,
                    "base_artifact_bytes": artifact_bytes_before_terminal,
                    "final_total_bytes": 0,
                },
                "checks": {},
            },
            "permissions": {
                "representation_permission": False,
                "distillation_permission": False,
                "validation_permission": False,
                "test_permission": False,
                "full_train_permission": False,
            },
        }

        def set_terminal_total(payload: dict[str, object], final_total: int) -> None:
            recorded_wall_seconds = float(
                payload["operational_budget"]["observed"]["wall_seconds"]
            )
            checks = {
                "wall_seconds_lte_900": recorded_wall_seconds
                <= WALL_BUDGET_SECONDS,
                "peak_vram_lte_6gib": peak_vram <= VRAM_BUDGET_BYTES,
                "final_total_bytes_lte_100mib": int(final_total)
                <= ARTIFACT_BUDGET_BYTES,
            }
            payload["operational_budget"]["observed"]["final_total_bytes"] = int(
                final_total
            )
            payload["operational_budget"]["checks"] = checks

        def set_terminal_wall(payload: dict[str, object], seconds: float) -> None:
            payload["operational_budget"]["observed"]["wall_seconds"] = float(
                seconds
            )

        terminal_bytes, terminal_total, _terminal_wall_upper_bound = (
            _timed_fixed_point_json_bytes(
                terminal,
                base_bytes=artifact_bytes_before_terminal,
                set_final_total=set_terminal_total,
                started=started,
                set_wall_seconds=set_terminal_wall,
            )
        )
        _atomic_bytes(partial / "terminal_operational_failure.json", terminal_bytes)
        if _output_bytes(partial) != terminal_total:
            raise RuntimeError("B15 terminal manifest final-byte total changed")
        partial.replace(output)
        return terminal

    _atomic_bytes(partial / "summary.json", summary_bytes)
    actual_final_total = _output_bytes(partial)
    if (
        actual_final_total != projected_summary_total
        or actual_final_total > ARTIFACT_BUDGET_BYTES
    ):
        raise RuntimeError("B15 valid summary final-byte contract changed")
    partial.replace(output)
    return summary


def _quarantine_partial(args: argparse.Namespace, error: BaseException) -> None:
    try:
        output = _validated_b15_output_path(args.output_dir)
    except (OSError, ValueError):
        return
    partial = output.with_name(output.name + ".partial")
    owner_path = partial / "run_owner.json"
    if not partial.is_dir() or not owner_path.is_file():
        return
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(owner, Mapping) or owner.get("protocol_id") != PROTOCOL_ID:
        return
    manifest = partial / "failure_manifest.json"
    if manifest.exists():
        return
    files = [
        {
            "relative_path": path.relative_to(partial).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(partial.rglob("*"))
        if path.is_file()
    ]
    metric_boundary = (partial / "stage_metric_construction_started.json").is_file()
    metrics_constructed = (partial / "stage_metrics_constructed.json").is_file()
    _atomic_json(
        manifest,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "QUARANTINED_PARTIAL",
            "created_at_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "metric_boundary_crossed": metric_boundary,
            "metrics_constructed": metrics_constructed,
            "retry_allowed": not metric_boundary,
            "partial_root": str(partial),
            "files": files,
            "deletion_requires_reviewed_cleanup_manifest": True,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        payload = build_preflight(args)
    else:
        try:
            payload = _run_formal(args)
        except BaseException as error:
            _quarantine_partial(args, error)
            raise
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

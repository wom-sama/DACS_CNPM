from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from trkh.tools.audit_dinov3_convnext_b24 import (  # noqa: E402
    _load_reference,
    _prepare_b24_ledger,
)  # noqa: E402
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (  # noqa: E402
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    DINO_FEATURE_DIM,
    DINO_IMAGE_SIZE,
    DINO_MODEL,
    DINO_PREFIX_TOKENS,
    DINO_WEIGHT_SHA256,
    FOLDS,
    LOGISTIC_C,
    LOGISTIC_TOL,
    SEED,
    WORKERS,
    _TrainLedgerDataset,
    _atomic_json,
    _atomic_npy,
    _atomic_npz,
    _build_dino,
    _dependency_contract,
    _device_contract,
    _dino_transform,
    _git_contract,
    _sha256,
    classification_summary,
    paired_component_bootstrap,
)  # noqa: E402
from trkh.tools.probe_embedding_prototypes import _resolve_device  # noqa: E402


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B27_SPARSE_SUBPATCH_SIGNAL_20260805"
TOP_K = 3
GRID_SIZE = 16
PATCH_SIZE = 16
SUBPATCH_SIZE = 8
PCA_COMPONENTS = 8
DETAIL_CHANNELS = 3
DETAIL_DIM = PCA_COMPONENTS * DETAIL_CHANNELS
READOUT_MAX_ITER = 5_000


class B27ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B27 TRAIN-only frozen-DINO sparse high-resolution subpatch signal "
            "screen. Validation and test are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_hashes() -> dict[str, str]:
    root = _repo()
    paths = {
        "runner": Path("trkh/tools/screen_dinov3_sparse_subpatch_b27.py"),
        "test": Path("tests/test_screen_dinov3_sparse_subpatch_b27.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "b13": Path("trkh/tools/audit_efficientvim_m1_frozen_transfer_b13.py"),
        "b24": Path("trkh/tools/audit_dinov3_convnext_b24.py"),
    }
    for path in paths.values():
        check = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0 or not (root / path).is_file():
            raise B27ContractError(f"B27 source missing or untracked: {path}")
    return {name: _sha256(root / path) for name, path in paths.items()}


def spatial_half_shift(indices: Tensor) -> Tensor:
    if indices.ndim != 2:
        raise ValueError("B27 route indices must be [B,K]")
    rows = torch.div(indices, GRID_SIZE, rounding_mode="floor")
    cols = indices.remainder(GRID_SIZE)
    shifted = ((rows + GRID_SIZE // 2) % GRID_SIZE) * GRID_SIZE
    shifted = shifted + ((cols + GRID_SIZE // 2) % GRID_SIZE)
    return shifted


def haar_contrasts(quadrants: Tensor) -> Tensor:
    if quadrants.ndim != 4 or quadrants.shape[2] != 4:
        raise ValueError("B27 quadrant embeddings must be [B,K,4,D]")
    tl, tr, bl, br = quadrants.unbind(dim=2)
    horizontal = (tr + br - tl - bl) * 0.5
    vertical = (bl + br - tl - tr) * 0.5
    diagonal = (tl + br - tr - bl) * 0.5
    return torch.stack((horizontal, vertical, diagonal), dim=2)


def aggregate_contrasts(contrasts: Tensor, weights: Tensor) -> Tensor:
    if contrasts.ndim != 4 or contrasts.shape[2] != DETAIL_CHANNELS:
        raise ValueError("B27 contrasts must be [B,K,3,D]")
    if weights.shape != contrasts.shape[:2]:
        raise ValueError("B27 route weights are not aligned")
    normalized = weights.float() / weights.float().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (contrasts.float() * normalized[:, :, None, None]).sum(dim=1)


def _selected_subpatch_descriptor(
    model: nn.Module,
    images: Tensor,
    indices: Tensor,
    weights: Tensor,
) -> Tensor:
    batch = int(images.size(0))
    if tuple(images.shape[1:]) != (3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE):
        raise ValueError("B27 input geometry changed")
    if tuple(indices.shape) != (batch, TOP_K):
        raise ValueError("B27 top-K geometry changed")
    patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(
        3, PATCH_SIZE, PATCH_SIZE
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch, GRID_SIZE * GRID_SIZE, 3, PATCH_SIZE, PATCH_SIZE
    )
    rows = torch.arange(batch, device=images.device)[:, None]
    selected = patches[rows, indices]
    quadrants = torch.stack(
        (
            selected[..., :SUBPATCH_SIZE, :SUBPATCH_SIZE],
            selected[..., :SUBPATCH_SIZE, SUBPATCH_SIZE:],
            selected[..., SUBPATCH_SIZE:, :SUBPATCH_SIZE],
            selected[..., SUBPATCH_SIZE:, SUBPATCH_SIZE:],
        ),
        dim=2,
    )
    resized = F.interpolate(
        quadrants.reshape(-1, 3, SUBPATCH_SIZE, SUBPATCH_SIZE),
        size=(PATCH_SIZE, PATCH_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    embedded = model.patch_embed.proj(resized).flatten(2).transpose(1, 2)
    embedded = model.patch_embed.norm(embedded).squeeze(1)
    embedded = embedded.reshape(batch, TOP_K, 4, DINO_FEATURE_DIM)
    return aggregate_contrasts(haar_contrasts(embedded), weights)


def _attention_topk(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    captured: list[Tensor] = []
    attention_module = model.blocks[-1].attn
    original_fused = bool(attention_module.fused_attn)

    def capture(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        captured.append(output.detach())

    handle = attention_module.attn_drop.register_forward_hook(capture)
    try:
        attention_module.fused_attn = False
        with torch.inference_mode():
            model.forward_features(images)
    finally:
        attention_module.fused_attn = original_fused
        handle.remove()
    if len(captured) != 1:
        raise B27ContractError(f"B27 expected one attention tensor, got {len(captured)}")
    attention = captured[0]
    expected = (int(images.size(0)), int(attention_module.num_heads), 261, 261)
    if tuple(attention.shape) != expected or not bool(torch.isfinite(attention).all()):
        raise B27ContractError(f"B27 attention geometry/domain changed: {attention.shape}")
    patch_attention = attention[:, :, 0, DINO_PREFIX_TOKENS:].amax(dim=1)
    distribution = patch_attention / patch_attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
    values, indices = torch.topk(distribution, k=TOP_K, dim=1, largest=True, sorted=True)
    entropy = -(
        distribution.clamp_min(1e-12) * distribution.clamp_min(1e-12).log()
    ).sum(dim=1) / np.log(GRID_SIZE * GRID_SIZE)
    return indices, values, entropy


def _extract(
    model: nn.Module,
    dataset: _TrainLedgerDataset,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    count = len(dataset)
    candidate = np.full((count, DETAIL_CHANNELS, DINO_FEATURE_DIM), np.nan, np.float32)
    dephased = np.full_like(candidate, np.nan)
    topk_indices = np.full((count, TOP_K), -1, np.int16)
    topk_weights = np.full((count, TOP_K), np.nan, np.float32)
    entropy = np.full(count, np.nan, np.float32)
    expected_index = 0
    next_report = 10
    model.to(device).eval().requires_grad_(False)
    for batch_index, (images, _labels, indices) in enumerate(loader, start=1):
        observed = indices.numpy().astype(np.int64, copy=False)
        expected = np.arange(expected_index, expected_index + observed.size, dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise B27ContractError("B27 DataLoader order changed")
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        route, weights, current_entropy = _attention_topk(model, images)
        with torch.inference_mode():
            current = _selected_subpatch_descriptor(model, images, route, weights)
            shifted = _selected_subpatch_descriptor(
                model, images, spatial_half_shift(route), weights
            )
        candidate[observed] = current.cpu().numpy()
        dephased[observed] = shifted.cpu().numpy()
        topk_indices[observed] = route.cpu().numpy().astype(np.int16)
        topk_weights[observed] = weights.cpu().numpy()
        entropy[observed] = current_entropy.cpu().numpy()
        expected_index += observed.size
        percent = int(100 * batch_index / max(len(loader), 1))
        if percent >= next_report:
            print(f"B27 extract: {percent}% ({expected_index}/{count})", flush=True)
            next_report += 10
    model.cpu()
    arrays = {
        "candidate": candidate,
        "dephased": dephased,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "attention_entropy": entropy,
    }
    if expected_index != count or any(not np.isfinite(value).all() for value in arrays.values()):
        raise B27ContractError("B27 extraction incomplete/non-finite")
    if bool((topk_indices < 0).any()) or bool((topk_indices >= 256).any()):
        raise B27ContractError("B27 selected patch domain changed")
    return arrays


def fit_detail_oof(
    base: np.ndarray,
    detail: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, object]:
    base = np.asarray(base, dtype=np.float32)
    detail = np.asarray(detail, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    expected = (labels.size, DETAIL_CHANNELS, DINO_FEATURE_DIM)
    if base.shape != (labels.size, DINO_FEATURE_DIM) or detail.shape != expected:
        raise ValueError("B27 readout arrays are not aligned")
    scores = np.full((labels.size, 5), np.nan, np.float64)
    records: list[dict[str, object]] = []
    converged = True
    for fold in range(FOLDS):
        fit = folds != fold
        held = folds == fold
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
        pca.fit(base[fit].astype(np.float64, copy=False))
        projected = np.einsum(
            "ncd,qd->ncq", detail.astype(np.float64, copy=False), pca.components_
        ).reshape(labels.size, DETAIL_DIM)
        features = np.concatenate((base.astype(np.float64, copy=False), projected), axis=1)
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(features[fit])
        x_held = scaler.transform(features[held])
        classifier = LogisticRegression(
            C=LOGISTIC_C,
            class_weight="balanced",
            solver="lbfgs",
            tol=LOGISTIC_TOL,
            max_iter=READOUT_MAX_ITER,
            random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_fit, labels[fit])
        fold_converged = not any(
            issubclass(record.category, ConvergenceWarning) for record in caught
        ) and int(np.max(classifier.n_iter_)) < READOUT_MAX_ITER
        converged = converged and fold_converged
        scores[held] = classifier.decision_function(x_held)
        records.append(
            {
                "fold": fold,
                "fit_rows": int(fit.sum()),
                "held_rows": int(held.sum()),
                "n_iter": classifier.n_iter_.astype(int).tolist(),
                "converged": fold_converged,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        )
    if not np.isfinite(scores).all():
        raise B27ContractError("B27 OOF scores are incomplete/non-finite")
    return {"scores": scores, "fold_records": records, "converged": converged}


def assess_gate(
    *,
    base: Mapping[str, object],
    candidate: Mapping[str, object],
    dephased: Mapping[str, object],
    base_bootstrap: Mapping[str, object],
    dephased_bootstrap: Mapping[str, object],
    converged: bool,
    integrity: bool,
) -> dict[str, object]:
    deltas = {
        "class1_f1_vs_base": float(candidate["class1_f1"]) - float(base["class1_f1"]),
        "macro_f1_vs_base": float(candidate["macro_f1"]) - float(base["macro_f1"]),
        "pair_auroc_vs_base": float(candidate["mean_pair_auroc"])
        - float(base["mean_pair_auroc"]),
        "class1_f1_vs_dephased": float(candidate["class1_f1"])
        - float(dephased["class1_f1"]),
        "pair_auroc_vs_dephased": float(candidate["mean_pair_auroc"])
        - float(dephased["mean_pair_auroc"]),
        "class1_recall_ratio": float(candidate["class1_recall"])
        / max(float(base["class1_recall"]), 1e-12),
        "restricted_fp_rate_ratio": float(candidate["restricted_fp_rate"])
        / max(float(base["restricted_fp_rate"]), 1e-12),
    }
    base_folds = {int(row["fold"]): row for row in base["folds"]}
    candidate_folds = {int(row["fold"]): row for row in candidate["folds"]}
    fold_deltas = [
        float(candidate_folds[fold]["class1_f1"])
        - float(base_folds[fold]["class1_f1"])
        for fold in range(FOLDS)
    ]
    checks = {
        "class1_signal": deltas["class1_f1_vs_base"] >= 0.010,
        "macro_noninferiority": deltas["macro_f1_vs_base"] >= -0.002,
        "pair_signal": deltas["pair_auroc_vs_base"] >= 0.001,
        "route_specific_class1": deltas["class1_f1_vs_dephased"] >= 0.005,
        "route_specific_pair": deltas["pair_auroc_vs_dephased"] > 0.0,
        "class1_recall_retention": deltas["class1_recall_ratio"] >= 0.98,
        "restricted_fp_reduction": deltas["restricted_fp_rate_ratio"] <= 0.95,
        "fold_stability": sum(value > 0.0 for value in fold_deltas) >= 4,
        "base_bootstrap_no_c1_harm": float(
            base_bootstrap["intervals"]["class1_f1_delta"]["lower"]
        ) >= 0.0,
        "dephased_bootstrap_no_pair_harm": float(
            dephased_bootstrap["intervals"]["mean_pair_auroc_delta"]["lower"]
        ) >= 0.0,
        "readouts_converged": bool(converged),
        "integrity_train_only": bool(integrity),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "sparse_router_refiner_design_permission": passed,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
        "deltas": {**deltas, "class1_f1_by_fold": fold_deltas},
    }


def _run_tests() -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_screen_dinov3_sparse_subpatch_b27.py"]
    result = subprocess.run(command, cwd=_repo(), capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise B27ContractError(f"B27 focused tests failed:\n{result.stdout}\n{result.stderr}")
    return {"command": command, "stdout": result.stdout.strip(), "passed": True}


def _locked_runtime(args: argparse.Namespace) -> None:
    if int(args.batch_size) != BATCH_SIZE or int(args.workers) != WORKERS:
        raise ValueError(f"B27 is locked to batch-size={BATCH_SIZE}, workers={WORKERS}")
    weight = args.dino_weight.expanduser().resolve()
    if not weight.is_file() or _sha256(weight) != DINO_WEIGHT_SHA256:
        raise ValueError("B27 DINO weight missing or changed")


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    _locked_runtime(args)
    ledger = _prepare_b24_ledger(args)
    reference = _load_reference(args.b13_dir, ledger)
    git = _git_contract()
    if git["branch"] != "research/pretrained-classf-b1" or not git["tracked_worktree_clean"]:
        raise B27ContractError(f"B27 requires the clean pretrained branch: {git}")
    model = _build_dino(args.dino_weight.expanduser().resolve(), num_classes=0).eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    images = torch.randn(2, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE, generator=generator)
    route, weights, entropy = _attention_topk(model, images)
    with torch.inference_mode():
        descriptor = _selected_subpatch_descriptor(model, images, route, weights)
        dephased = _selected_subpatch_descriptor(
            model, images, spatial_half_shift(route), weights
        )
    tests = _run_tests()
    checks = {
        "git_clean_pretrained_branch": True,
        "train_ledger_exact": True,
        "validation_not_constructed": True,
        "test_not_constructed": True,
        "raw_dino_strict_weight": True,
        "route_shape": tuple(route.shape) == (2, TOP_K),
        "route_unique": all(torch.unique(row).numel() == TOP_K for row in route),
        "attention_finite": bool(torch.isfinite(entropy).all()),
        "candidate_shape": tuple(descriptor.shape) == (2, DETAIL_CHANNELS, DINO_FEATURE_DIM),
        "dephased_shape": tuple(dephased.shape) == tuple(descriptor.shape),
        "detail_nonzero": float(descriptor.abs().max()) > 0.0,
        "b13_exact_replay": bool(reference["summary"]),
        "focused_tests": bool(tests["passed"]),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at_unix": time.time(),
        "source_hashes": _source_hashes(),
        "git": git,
        "dependencies": _dependency_contract(),
        "device": _device_contract(args.device),
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "train_rows": int(np.asarray(ledger["labels"]).size),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": ledger["train_content"],
        },
        "assignment": {
            key: ledger["assignment"][key]
            for key in ("csv_sha256", "assignment_int64_sha256", "group_vector_int64_sha256", "fold_rows")
        },
        "weight": {
            "model": DINO_MODEL,
            "path": str(args.dino_weight.expanduser().resolve()),
            "sha256": DINO_WEIGHT_SHA256,
        },
        "reference": {
            "directory": str(args.b13_dir.expanduser().resolve()),
            "artifacts": reference["artifacts"],
            "metrics": reference["summary"],
        },
        "screen": {
            "top_k": TOP_K,
            "top_k_fraction": TOP_K / (GRID_SIZE * GRID_SIZE),
            "head_aggregation": "max_cls_to_patch_final_block",
            "subdivision": "2x2_bilinear_then_reuse_pretrained_patch_projection",
            "detail": "weighted_three_haar_contrasts_in_embedding_space",
            "capacity_control": "same_detail_pipeline_at_spatial_half_shift",
            "pca_components_per_contrast": PCA_COMPONENTS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "focused_tests": tests,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise B27ContractError(f"B27 preflight failed: {checks}")
    return payload


def _validate_preflight(args: argparse.Namespace, ledger: Mapping[str, object]) -> dict[str, object]:
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B27 requires an accepted preflight path and SHA-256")
    path = args.preflight_artifact.expanduser().resolve()
    digest = _sha256(path)
    if digest != args.preflight_sha256.lower():
        raise ValueError("B27 accepted preflight SHA changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "passed": payload.get("passed") is True,
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "sources": payload.get("source_hashes") == _source_hashes(),
        "git": payload.get("git") == _git_contract(),
        "train_content": payload.get("dataset", {}).get("train_content") == ledger["train_content"],
        "weight": _sha256(args.dino_weight.expanduser().resolve()) == DINO_WEIGHT_SHA256,
    }
    if not all(checks.values()):
        raise B27ContractError(f"B27 accepted preflight invalid: {checks}")
    return {"path": str(path), "sha256": digest, "checks": checks}


def _formal(args: argparse.Namespace) -> dict[str, object]:
    _locked_runtime(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B27 output: {output}")
    ledger = _prepare_b24_ledger(args)
    accepted = _validate_preflight(args, ledger)
    reference = _load_reference(args.b13_dir, ledger)
    source_start = _source_hashes()
    git_start = _git_contract()
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    model = _build_dino(args.dino_weight.expanduser().resolve(), num_classes=0)
    dataset = _TrainLedgerDataset(
        ledger["absolute_paths"], ledger["labels"], _dino_transform(model)
    )
    started = time.perf_counter()
    extracted = _extract(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    labels = np.asarray(ledger["labels"], dtype=np.int64)
    folds = np.asarray(ledger["assignment"]["folds"], dtype=np.int64)
    groups = np.asarray(ledger["assignment"]["groups"], dtype=np.int64)
    readouts = {
        "candidate": fit_detail_oof(reference["features"], extracted["candidate"], labels, folds),
        "dephased": fit_detail_oof(reference["features"], extracted["dephased"], labels, folds),
    }
    metrics = {
        "base": reference["summary"],
        "candidate": classification_summary(labels, readouts["candidate"]["scores"], folds),
        "dephased": classification_summary(labels, readouts["dephased"]["scores"], folds),
    }
    base_bootstrap = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=reference["scores"],
        candidate_scores=readouts["candidate"]["scores"],
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    dephased_bootstrap = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=readouts["dephased"]["scores"],
        candidate_scores=readouts["candidate"]["scores"],
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED + 1,
    )
    source_end = _source_hashes()
    git_end = _git_contract()
    converged = all(bool(value["converged"]) for value in readouts.values())
    integrity = bool(
        source_start == source_end
        and git_start == git_end
        and git_end["tracked_worktree_clean"] is True
        and all(np.isfinite(value).all() for value in extracted.values())
        and all(np.isfinite(value["scores"]).all() for value in readouts.values())
    )
    gate = assess_gate(
        base=metrics["base"],
        candidate=metrics["candidate"],
        dephased=metrics["dephased"],
        base_bootstrap=base_bootstrap,
        dephased_bootstrap=dephased_bootstrap,
        converged=converged,
        integrity=integrity,
    )
    output.mkdir(parents=True, exist_ok=False)
    descriptor_sha = {
        name: _atomic_npy(output / f"train_{name}.npy", value)
        for name, value in extracted.items()
    }
    oof_sha = _atomic_npz(
        output / "train_oof_readouts.npz",
        labels=labels,
        folds=folds,
        groups=groups,
        base_scores=np.asarray(reference["scores"], dtype=np.float64),
        candidate_scores=np.asarray(readouts["candidate"]["scores"], dtype=np.float64),
        dephased_scores=np.asarray(readouts["dephased"]["scores"], dtype=np.float64),
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_sparse_subpatch_signal_screen",
        "accepted_preflight": accepted,
        "source_hashes": source_start,
        "git": {"start": git_start, "end": git_end},
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "train_rows": int(labels.size),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": ledger["train_content"],
        },
        "screen": {
            "top_k": TOP_K,
            "top_k_fraction": TOP_K / 256,
            "attention_entropy_mean": float(extracted["attention_entropy"].mean()),
            "attention_entropy_p95": float(np.quantile(extracted["attention_entropy"], 0.95)),
            "detail_dimension_before_pca": DETAIL_CHANNELS * DINO_FEATURE_DIM,
            "detail_dimension_after_pca": DETAIL_DIM,
        },
        "metrics": metrics,
        "readouts": {name: value["fold_records"] for name, value in readouts.items()},
        "bootstrap": {"candidate_vs_base": base_bootstrap, "candidate_vs_dephased": dephased_bootstrap},
        "gate": gate,
        "artifacts": {"descriptors_sha256": descriptor_sha, "oof_sha256": oof_sha},
        "integrity_complete": integrity,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.preflight_only:
        if args.preflight_artifact is not None or args.preflight_sha256:
            raise ValueError("B27 preflight cannot consume another preflight")
        if output.exists():
            raise FileExistsError(f"refuse to overwrite B27 preflight: {output}")
        payload = _preflight(args)
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / "preflight.json", payload)
        print(json.dumps({"passed": payload["passed"], "output": str(output)}, indent=2))
        return 0
    summary = _formal(args)
    print(json.dumps({"gate": summary["gate"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

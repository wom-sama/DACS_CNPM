from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.tools.audit_dinov3_convnext_b24 import _load_reference, _prepare_b24_ledger
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    DINO_FEATURE_DIM,
    DINO_IMAGE_SIZE,
    DINO_PREFIX_TOKENS,
    FOLDS,
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
)
from trkh.tools.probe_embedding_prototypes import _resolve_device
from trkh.tools.screen_dinov3_sparse_subpatch_b27 import (
    DETAIL_CHANNELS,
    GRID_SIZE,
    TOP_K,
    _locked_runtime,
    _selected_subpatch_descriptor,
    assess_gate,
    fit_detail_oof,
    spatial_half_shift,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B28_MARGIN_DEGRADATION_SUBPATCH_20260805"


class B28ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B28 TRAIN-only cross-fitted margin-degradation subpatch screen; "
            "validation/test are never constructed."
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
        "runner": Path("trkh/tools/screen_dinov3_margin_subpatch_b28.py"),
        "test": Path("tests/test_screen_dinov3_margin_subpatch_b28.py"),
        "shared_b27": Path("trkh/tools/screen_dinov3_sparse_subpatch_b27.py"),
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
            raise B28ContractError(f"B28 source missing or untracked: {path}")
    return {name: _sha256(root / path) for name, path in paths.items()}


def _readout_state(reference: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(reference["artifacts"]["oof"], allow_pickle=False) as payload:
        state = {
            "folds": np.asarray(payload["folds"], dtype=np.int64),
            "base_scores": np.asarray(payload["dino_scores"], dtype=np.float32),
            "coefficients": np.asarray(payload["dino_coefficients"], dtype=np.float32),
            "intercepts": np.asarray(payload["dino_intercepts"], dtype=np.float32),
            "means": np.asarray(payload["dino_scaler_means"], dtype=np.float32),
            "scales": np.asarray(payload["dino_scaler_scales"], dtype=np.float32),
        }
    expected = {
        "folds": (8278,),
        "base_scores": (8278, 5),
        "coefficients": (FOLDS, 5, DINO_FEATURE_DIM),
        "intercepts": (FOLDS, 5),
        "means": (FOLDS, DINO_FEATURE_DIM),
        "scales": (FOLDS, DINO_FEATURE_DIM),
    }
    if any(state[name].shape != shape for name, shape in expected.items()):
        raise B28ContractError("B28 retained cross-fitted readout geometry changed")
    if any(not np.isfinite(value).all() for value in state.values()):
        raise B28ContractError("B28 retained cross-fitted readout is non-finite")
    return state


def _attention_maps(model: nn.Module, images: Tensor) -> Tensor:
    captured: list[Tensor] = []
    module = model.blocks[-1].attn
    original = bool(module.fused_attn)
    handle = module.attn_drop.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    try:
        module.fused_attn = False
        with torch.inference_mode():
            model.forward_features(images)
    finally:
        module.fused_attn = original
        handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape[2:]) != (261, 261):
        raise B28ContractError("B28 final attention capture changed")
    maps = captured[0][:, :, 0, DINO_PREFIX_TOKENS:].float()
    maps = maps / maps.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if not bool(torch.isfinite(maps).all()):
        raise B28ContractError("B28 attention map is non-finite")
    return maps


def _embedded_inputs(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor]:
    with torch.inference_mode():
        tokens = model.patch_embed(images)
        tokens, rope = model._pos_embed(tokens)
        tokens = model.norm_pre(tokens)
    if tuple(tokens.shape[1:]) != (261, DINO_FEATURE_DIM) or rope is None:
        raise B28ContractError("B28 DINO input token/RoPE geometry changed")
    return tokens, rope


def _drop_descriptor(
    model: nn.Module, tokens: Tensor, rope: Tensor, drop_indices: Tensor
) -> Tensor:
    batch = int(tokens.size(0))
    mask = torch.ones(batch, GRID_SIZE * GRID_SIZE, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, drop_indices, False)
    keep = mask.nonzero(as_tuple=False)[:, 1].reshape(batch, 256 - TOP_K)
    patches = tokens[:, DINO_PREFIX_TOKENS:].gather(
        1, keep[:, :, None].expand(-1, -1, DINO_FEATURE_DIM)
    )
    current = torch.cat((tokens[:, :DINO_PREFIX_TOKENS], patches), dim=1)
    batch_rope = rope[None].expand(batch, -1, -1).gather(
        1, keep[:, :, None].expand(-1, -1, int(rope.size(-1)))
    )
    batch_rope = batch_rope.unsqueeze(1)
    with torch.inference_mode():
        for block in model.blocks:
            current = block(current, rope=batch_rope)
        current = model.norm(current)
    return current[:, DINO_PREFIX_TOKENS:].mean(dim=1).float()


def select_margin_degradation_route(
    attention: Tensor,
    degraded_features: Tensor,
    folds: Tensor,
    base_scores: Tensor,
    coefficients: Tensor,
    intercepts: Tensor,
    means: Tensor,
    scales: Tensor,
) -> dict[str, Tensor]:
    batch, heads, patches = attention.shape
    if patches != 256 or degraded_features.shape != (batch, heads, DINO_FEATURE_DIM):
        raise ValueError("B28 attention/degraded feature geometry changed")
    order = base_scores.argsort(dim=1, descending=True)
    predicted, rival = order[:, 0], order[:, 1]
    base_margin = base_scores.gather(1, predicted[:, None]).squeeze(1)
    base_margin -= base_scores.gather(1, rival[:, None]).squeeze(1)
    fold_coefficients = coefficients[folds]
    fold_intercepts = intercepts[folds]
    fold_means = means[folds]
    fold_scales = scales[folds]
    drops: list[Tensor] = []
    for head in range(heads):
        standardized = (degraded_features[:, head] - fold_means) / fold_scales
        scores = torch.einsum("bcd,bd->bc", fold_coefficients, standardized)
        scores = scores + fold_intercepts
        margin = scores.gather(1, predicted[:, None]).squeeze(1)
        margin -= scores.gather(1, rival[:, None]).squeeze(1)
        drops.append(base_margin - margin)
    degradation = torch.stack(drops, dim=1)
    selected_head = degradation.argmax(dim=1)
    head_topk = attention.topk(TOP_K, dim=-1, largest=True, sorted=True)
    rows = torch.arange(batch, device=attention.device)
    indices = head_topk.indices[rows, selected_head]
    weights = head_topk.values[rows, selected_head]
    selected_map = attention[rows, selected_head]
    entropy = -(
        selected_map.clamp_min(1e-12) * selected_map.clamp_min(1e-12).log()
    ).sum(dim=1) / np.log(256)
    max_route = attention.amax(dim=1).topk(TOP_K, dim=1).indices
    return {
        "indices": indices,
        "weights": weights,
        "selected_head": selected_head,
        "margin_drop": degradation[rows, selected_head],
        "attention_entropy": entropy,
        "max_route": max_route,
    }


def _is_border(indices: Tensor) -> Tensor:
    rows = torch.div(indices, GRID_SIZE, rounding_mode="floor")
    cols = indices.remainder(GRID_SIZE)
    return (rows <= 1) | (rows >= 14) | (cols <= 1) | (cols >= 14)


def _extract(
    model: nn.Module,
    dataset: _TrainLedgerDataset,
    readout: Mapping[str, np.ndarray],
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
    output = {
        "candidate": np.full((count, DETAIL_CHANNELS, DINO_FEATURE_DIM), np.nan, np.float32),
        "dephased": np.full((count, DETAIL_CHANNELS, DINO_FEATURE_DIM), np.nan, np.float32),
        "topk_indices": np.full((count, TOP_K), -1, np.int16),
        "topk_weights": np.full((count, TOP_K), np.nan, np.float32),
        "selected_head": np.full(count, -1, np.int8),
        "margin_drop": np.full(count, np.nan, np.float32),
        "attention_entropy": np.full(count, np.nan, np.float32),
        "route_border": np.full(count, np.nan, np.float32),
        "max_route_border": np.full(count, np.nan, np.float32),
        "same_as_max_route": np.full(count, np.nan, np.float32),
    }
    tensors = {
        name: torch.from_numpy(readout[name]).to(device=device)
        for name in ("coefficients", "intercepts", "means", "scales")
    }
    expected_index = 0
    next_report = 10
    model.to(device).eval().requires_grad_(False)
    for batch_index, (images, _labels, indices) in enumerate(loader, start=1):
        observed = indices.numpy().astype(np.int64, copy=False)
        expected = np.arange(expected_index, expected_index + observed.size, dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise B28ContractError("B28 DataLoader order changed")
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        attention = _attention_maps(model, images)
        tokens, rope = _embedded_inputs(model, images)
        per_head_topk = attention.topk(TOP_K, dim=-1).indices
        degraded = torch.stack(
            [_drop_descriptor(model, tokens, rope, per_head_topk[:, head]) for head in range(attention.size(1))],
            dim=1,
        )
        route = select_margin_degradation_route(
            attention,
            degraded,
            torch.from_numpy(readout["folds"][observed]).to(device=device),
            torch.from_numpy(readout["base_scores"][observed]).to(device=device),
            tensors["coefficients"],
            tensors["intercepts"],
            tensors["means"],
            tensors["scales"],
        )
        with torch.inference_mode():
            candidate = _selected_subpatch_descriptor(
                model, images, route["indices"], route["weights"]
            )
            dephased = _selected_subpatch_descriptor(
                model, images, spatial_half_shift(route["indices"]), route["weights"]
            )
        output["candidate"][observed] = candidate.cpu().numpy()
        output["dephased"][observed] = dephased.cpu().numpy()
        output["topk_indices"][observed] = route["indices"].cpu().numpy().astype(np.int16)
        output["topk_weights"][observed] = route["weights"].cpu().numpy()
        output["selected_head"][observed] = route["selected_head"].cpu().numpy().astype(np.int8)
        for name in ("margin_drop", "attention_entropy"):
            output[name][observed] = route[name].cpu().numpy()
        output["route_border"][observed] = _is_border(route["indices"]).float().mean(1).cpu().numpy()
        output["max_route_border"][observed] = _is_border(route["max_route"]).float().mean(1).cpu().numpy()
        output["same_as_max_route"][observed] = (
            route["indices"].eq(route["max_route"]).all(1).float().cpu().numpy()
        )
        expected_index += observed.size
        percent = int(100 * batch_index / max(len(loader), 1))
        if percent >= next_report:
            print(f"B28 extract: {percent}% ({expected_index}/{count})", flush=True)
            next_report += 10
    model.cpu()
    if expected_index != count or any(not np.isfinite(value).all() for value in output.values()):
        raise B28ContractError("B28 extraction incomplete/non-finite")
    return output


def _run_tests() -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_screen_dinov3_margin_subpatch_b28.py"]
    result = subprocess.run(command, cwd=_repo(), capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise B28ContractError(f"B28 tests failed:\n{result.stdout}\n{result.stderr}")
    return {"command": command, "stdout": result.stdout.strip(), "passed": True}


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    _locked_runtime(args)
    ledger = _prepare_b24_ledger(args)
    reference = _load_reference(args.b13_dir, ledger)
    readout = _readout_state(reference)
    git = _git_contract()
    if git["branch"] != "research/pretrained-classf-b1" or not git["tracked_worktree_clean"]:
        raise B28ContractError(f"B28 requires the clean pretrained branch: {git}")
    device = _resolve_device(args.device)
    model = _build_dino(args.dino_weight.expanduser().resolve(), num_classes=0).to(device).eval()
    images = torch.randn(2, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE, generator=torch.Generator().manual_seed(SEED)).to(device)
    attention = _attention_maps(model, images)
    tokens, rope = _embedded_inputs(model, images)
    per_head = attention.topk(TOP_K, dim=-1).indices
    degraded = torch.stack(
        [_drop_descriptor(model, tokens, rope, per_head[:, head]) for head in range(attention.size(1))], dim=1
    )
    probe = select_margin_degradation_route(
        attention,
        degraded,
        torch.from_numpy(readout["folds"][:2]).to(device),
        torch.from_numpy(readout["base_scores"][:2]).to(device),
        *[torch.from_numpy(readout[name]).to(device) for name in ("coefficients", "intercepts", "means", "scales")],
    )
    with torch.inference_mode():
        descriptor = _selected_subpatch_descriptor(model, images, probe["indices"], probe["weights"])
    tests = _run_tests()
    checks = {
        "git_clean_pretrained_branch": True,
        "train_ledger_exact": True,
        "validation_not_constructed": True,
        "test_not_constructed": True,
        "route_shape": tuple(probe["indices"].shape) == (2, TOP_K),
        "head_domain": int(probe["selected_head"].min()) >= 0 and int(probe["selected_head"].max()) < attention.size(1),
        "degradation_finite": bool(torch.isfinite(probe["margin_drop"]).all()),
        "descriptor_shape": tuple(descriptor.shape) == (2, DETAIL_CHANNELS, DINO_FEATURE_DIM),
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
        "weight": {
            "path": str(args.dino_weight.expanduser().resolve()),
            "sha256": _sha256(args.dino_weight.expanduser().resolve()),
        },
        "reference": reference["artifacts"],
        "screen": {
            "selector": "cross_fitted_predicted_margin_degradation",
            "labels_used_by_selector": False,
            "top_k": TOP_K,
            "subpatch_and_readout": "exact_b27",
            "capacity_control": "spatial_half_shift",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "focused_tests": tests,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise B28ContractError(f"B28 preflight failed: {checks}")
    return payload


def _accepted_preflight(args: argparse.Namespace, ledger: Mapping[str, Any]) -> dict[str, Any]:
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B28 requires an accepted preflight path and SHA-256")
    path = args.preflight_artifact.expanduser().resolve()
    digest = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "sha": digest == args.preflight_sha256.lower(),
        "passed": payload.get("passed") is True,
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "sources": payload.get("source_hashes") == _source_hashes(),
        "git": payload.get("git") == _git_contract(),
        "train_content": payload.get("dataset", {}).get("train_content") == ledger["train_content"],
    }
    if not all(checks.values()):
        raise B28ContractError(f"B28 accepted preflight invalid: {checks}")
    return {"path": str(path), "sha256": digest, "checks": checks}


def _formal(args: argparse.Namespace) -> dict[str, Any]:
    _locked_runtime(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B28 output: {output}")
    ledger = _prepare_b24_ledger(args)
    accepted = _accepted_preflight(args, ledger)
    reference = _load_reference(args.b13_dir, ledger)
    readout_state = _readout_state(reference)
    source_start, git_start = _source_hashes(), _git_contract()
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    model = _build_dino(args.dino_weight.expanduser().resolve(), num_classes=0)
    dataset = _TrainLedgerDataset(ledger["absolute_paths"], ledger["labels"], _dino_transform(model))
    started = time.perf_counter()
    extracted = _extract(
        model,
        dataset,
        readout_state,
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
    bootstraps = {
        "candidate_vs_base": paired_component_bootstrap(
            labels=labels,
            folds=folds,
            groups=groups,
            dino_scores=reference["scores"],
            candidate_scores=readouts["candidate"]["scores"],
            replicates=BOOTSTRAP_REPLICATES,
            seed=SEED,
        ),
        "candidate_vs_dephased": paired_component_bootstrap(
            labels=labels,
            folds=folds,
            groups=groups,
            dino_scores=readouts["dephased"]["scores"],
            candidate_scores=readouts["candidate"]["scores"],
            replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + 1,
        ),
    }
    source_end, git_end = _source_hashes(), _git_contract()
    converged = all(bool(value["converged"]) for value in readouts.values())
    integrity = bool(
        source_start == source_end
        and git_start == git_end
        and git_end["tracked_worktree_clean"] is True
        and all(np.isfinite(value).all() for value in extracted.values())
    )
    gate = assess_gate(
        base=metrics["base"],
        candidate=metrics["candidate"],
        dephased=metrics["dephased"],
        base_bootstrap=bootstraps["candidate_vs_base"],
        dephased_bootstrap=bootstraps["candidate_vs_dephased"],
        converged=converged,
        integrity=integrity,
    )
    teacher_checks = {
        "positive_margin_degradation": float((extracted["margin_drop"] > 0).mean()) >= 0.90,
        "route_materially_differs_from_max": float(extracted["same_as_max_route"].mean()) <= 0.50,
    }
    gate["checks"].update(teacher_checks)
    gate["failed"] = [name for name, value in gate["checks"].items() if not value]
    gate["passed"] = all(gate["checks"].values())
    gate["sparse_router_refiner_design_permission"] = gate["passed"]
    output.mkdir(parents=True, exist_ok=False)
    descriptor_sha = {
        name: _atomic_npy(output / f"train_{name}.npy", value) for name, value in extracted.items()
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
        "mode": "formal_train_only_margin_degradation_subpatch_screen",
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
        "selector": {
            "labels_used": False,
            "positive_margin_degradation_fraction": float((extracted["margin_drop"] > 0).mean()),
            "margin_drop_mean": float(extracted["margin_drop"].mean()),
            "route_border_fraction": float(extracted["route_border"].mean()),
            "max_route_border_fraction": float(extracted["max_route_border"].mean()),
            "same_as_max_route_fraction": float(extracted["same_as_max_route"].mean()),
            "selected_head_counts": np.bincount(extracted["selected_head"].astype(np.int64), minlength=6).astype(int).tolist(),
            "attention_entropy_mean": float(extracted["attention_entropy"].mean()),
        },
        "metrics": metrics,
        "readouts": {name: value["fold_records"] for name, value in readouts.items()},
        "bootstrap": bootstraps,
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
        if output.exists():
            raise FileExistsError(f"refuse to overwrite B28 preflight: {output}")
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

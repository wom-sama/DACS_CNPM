from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from trkh.core.config import load_data_spec  # noqa: E402
from trkh.core.utils import build_warmup_decay_scheduler  # noqa: E402
from trkh.data.dataset import (  # noqa: E402
    ClassificationFolderDataset,
    TemperedClassBatchSampler,
)
from trkh.models.model import build_model_from_checkpoint  # noqa: E402
from trkh.models.multidepth_evidence_query_b26 import (  # noqa: E402
    MultiDepthEvidenceQuery,
    boundary_guarded_loss,
    configure_query_only,
    is_query_parameter,
    load_query_state_dict,
    query_state_dict,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SELECTED_STATE_SHA256,
    _state_dict_sha256,
)
from trkh.tools.run_b23_b9_guarded_convpass import (  # noqa: E402
    DEFAULT_B9_CHECKPOINT,
    EXPECTED_B9_CONFUSION,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_CLASS_NAMES,
    EXPECTED_DATA_YAML,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    _atomic_json,
    _configure_determinism,
    _datasets,
    _eval_loader,
    _evaluate,
    _git_snapshot,
    _load_b9,
    _seed_all,
    _seed_worker,
    _split_batch,
    build_transforms,
    sha256_file,
)
from trkh.training.losses import LDAMFocalLoss  # noqa: E402


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B26_MULTIDEPTH_EVIDENCE_QUERY_20260805"
SEED = 20260805
EPOCHS = 12
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
WORKERS = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 2
MIN_LR = 1e-6
GRAD_CLIP = 1.0
RETENTION_WEIGHT = 1.0
HARD_NEGATIVE_WEIGHT = 0.10
HARD_NEGATIVE_MARGIN = 0.10
DISTILLATION_WEIGHT = 0.10
DISTILLATION_TEMPERATURE = 2.0
QUERY_PARAMETERS = 58_753
TOTAL_PARAMETERS = 21_647_622


class B26ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B26 inherited-B9 one-way multi-depth class-evidence query screen. "
            "Validation is design-exposed; test is never constructed."
        )
    )
    parser.add_argument("--data", type=Path, default=EXPECTED_DATA_YAML)
    parser.add_argument("--b9-checkpoint", type=Path, default=DEFAULT_B9_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--accepted-preflight", type=Path, default=None)
    parser.add_argument("--accepted-preflight-sha256", type=str, default="")
    return parser.parse_args(argv)


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _sources(repo: Path) -> dict[str, str]:
    paths = {
        "model": Path("trkh/models/multidepth_evidence_query_b26.py"),
        "runner": Path("trkh/tools/run_b26_multidepth_evidence_query.py"),
        "test": Path("tests/test_multidepth_evidence_query_b26.py"),
        "b23_helpers": Path("trkh/tools/run_b23_b9_guarded_convpass.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
    }
    for relative in paths.values():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if not (repo / relative).is_file() or tracked.returncode != 0:
            raise B26ContractError(f"B26 source missing/untracked: {relative}")
    return {name: sha256_file(repo / relative) for name, relative in paths.items()}


def build_b26(backbone: nn.Module) -> MultiDepthEvidenceQuery:
    model = MultiDepthEvidenceQuery(backbone)
    counts = configure_query_only(model)
    if counts != {
        "total": TOTAL_PARAMETERS,
        "trainable": QUERY_PARAMETERS,
        "frozen": TOTAL_PARAMETERS - QUERY_PARAMETERS,
    }:
        raise B26ContractError(f"B26 parameter contract changed: {counts}")
    if _state_dict_sha256(model.backbone.state_dict()) != EXPECTED_SELECTED_STATE_SHA256:
        raise B26ContractError("B26 did not inherit the selected B9 state exactly")
    return model


def _train_loader(dataset: Any) -> tuple[DataLoader, TemperedClassBatchSampler]:
    sampler = TemperedClassBatchSampler(
        dataset.labels(),
        batch_size=BATCH_SIZE,
        num_classes=5,
        power=0.5,
        epoch_multiplier=1.0,
        seed=SEED,
        drop_last=False,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=WORKERS,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            persistent_workers=True,
            prefetch_factor=2,
        ),
        sampler,
    )


def _query_norm(model: nn.Module, *, gradients: bool = False) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not is_query_parameter(name):
            continue
        value = parameter.grad if gradients else parameter
        if value is not None:
            total += float(value.detach().float().square().sum().item())
    return math.sqrt(total)


def _gate(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trace: Mapping[str, float],
) -> dict[str, object]:
    base_c1, candidate_c1 = base["class1"], candidate["class1"]
    checks = {
        "class1_f1_gain": float(candidate_c1["f1"])
        >= float(base_c1["f1"]) + 0.005,
        "macro_noninferiority": float(candidate["macro_f1"])
        >= float(base["macro_f1"]) - 0.002,
        "accuracy_noninferiority": float(candidate["accuracy"])
        >= float(base["accuracy"]) - 0.005,
        "class1_tp_retention": int(candidate_c1["tp"])
        >= math.ceil(0.98 * int(base_c1["tp"])),
        "restricted_fp_reduction": int(candidate["restricted_fp_into_class1"])
        <= math.floor(0.95 * int(base["restricted_fp_into_class1"])),
        "two_to_one_nonincrease": int(candidate["transitions_into_class1"]["2->1"])
        <= int(base["transitions_into_class1"]["2->1"]),
        "query_residual_active_bounded": 1e-4 <= trace["residual_p95"] < 0.5,
        "query_attention_nonuniform": trace["attention_normalized_entropy"] < 0.995,
        "query_attention_finite": bool(trace["attention_finite"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
        "deltas": {
            "accuracy": float(candidate["accuracy"] - base["accuracy"]),
            "macro_f1": float(candidate["macro_f1"] - base["macro_f1"]),
            "class1_f1": float(candidate_c1["f1"] - base_c1["f1"]),
            "class1_tp": int(candidate_c1["tp"] - base_c1["tp"]),
            "restricted_fp": int(
                candidate["restricted_fp_into_class1"]
                - base["restricted_fp_into_class1"]
            ),
            "two_to_one": int(
                candidate["transitions_into_class1"]["2->1"]
                - base["transitions_into_class1"]["2->1"]
            ),
        },
        "permission": (
            "equal_parameter_mean_token_query_control"
            if all(checks.values())
            else "close_exact_b26_without_tuning"
        ),
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
    }


def _trace_summary(model: MultiDepthEvidenceQuery, images: Tensor) -> dict[str, float]:
    with torch.inference_mode():
        _logits, trace = model.forward_with_trace(images, return_attention=True)
    residual = trace["residual_logits"].abs().flatten().float()
    attention = trace["class_to_patch_attention"].float()
    entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(int(attention.size(-1)))
    return {
        "residual_p50": float(torch.quantile(residual, 0.50).cpu()),
        "residual_p95": float(torch.quantile(residual, 0.95).cpu()),
        "residual_max": float(residual.max().cpu()),
        "attention_normalized_entropy": float(entropy.mean().cpu()),
        "attention_std": float(attention.std().cpu()),
        "attention_finite": bool(torch.isfinite(attention).all().cpu()),
    }


def _preflight(args: argparse.Namespace, repo: Path, git: Mapping[str, Any]) -> dict[str, object]:
    checkpoint, backbone, checkpoint_contract = _load_b9(args.b9_checkpoint)
    train, val = _datasets(args.data, include_validation=False)
    if val is not None:
        raise B26ContractError("B26 preflight constructed validation")
    _seed_all(SEED)
    model = build_b26(backbone)
    eval_transform = build_transforms()[1]
    spec = load_data_spec(args.data, class_name_mode="raw", expected_num_classes=5)
    probe = ClassificationFolderDataset.from_data_spec(
        spec, split="train", transform=eval_transform
    )
    images = torch.stack([probe[index][0] for index in range(2)]).cuda()
    model.cuda().eval()
    with torch.inference_mode():
        native = model.backbone(images).float()
        candidate, trace = model.forward_with_trace(images, return_attention=True)
    parity = float((candidate - native).abs().max().cpu())
    if parity != 0.0:
        raise B26ContractError(f"B26 step-zero parity failed: {parity}")
    optimizer = torch.optim.SGD(list(model.query_parameters()), lr=LEARNING_RATE)
    labels = torch.tensor([1, 2], device=images.device)
    gradient_records: list[list[str]] = []
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits, current = model.forward_with_trace(images, return_attention=False)
        loss, _parts = boundary_guarded_loss(
            logits, current["base_logits"], labels, nn.CrossEntropyLoss()
        )
        loss.backward()
        escaped = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not is_query_parameter(name)
        ]
        if escaped or not bool(torch.isfinite(loss).item()):
            raise B26ContractError(f"B26 preflight gradient/loss failure: {escaped}")
        gradient_records.append(
            [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and float(parameter.grad.detach().abs().sum().item()) > 0.0
            ]
        )
        optimizer.step()
    required_step2 = ("class_queries", "evidence_query.patch_projection.weight")
    if not all(name in gradient_records[1] for name in required_step2):
        raise B26ContractError(f"B26 decoder did not open on step two: {gradient_records}")
    if _state_dict_sha256(model.backbone.state_dict()) != EXPECTED_SELECTED_STATE_SHA256:
        raise B26ContractError("B26 preflight moved the frozen B9 state")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "passed": True,
        "git": dict(git),
        "source_hashes": _sources(repo),
        "checkpoint": checkpoint_contract,
        "data": {"train_rows": len(train), "validation": False, "test": False},
        "model": {
            "total_parameters": TOTAL_PARAMETERS,
            "trainable_parameters": QUERY_PARAMETERS,
            "frozen_parameters": TOTAL_PARAMETERS - QUERY_PARAMETERS,
            "primary_state_sha256": _state_dict_sha256(model.backbone.state_dict()),
            "step_zero_max_abs": parity,
            "attention_shape": list(trace["class_to_patch_attention"].shape),
            "gradient_names_by_step": gradient_records,
        },
        "recipe": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "sgd_momentum",
            "scheduler_horizon": EPOCHS,
        },
        "train": False,
        "validation": False,
        "test": False,
    }
    del checkpoint
    return payload


def _accepted(
    args: argparse.Namespace, repo: Path, git: Mapping[str, Any]
) -> dict[str, str]:
    if args.accepted_preflight is None or not args.accepted_preflight_sha256:
        raise B26ContractError("B26 formal run requires preflight path and SHA")
    path = args.accepted_preflight.expanduser().resolve(strict=True)
    digest = sha256_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    checks = {
        "sha": digest == args.accepted_preflight_sha256.lower(),
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "passed": payload.get("passed") is True,
        "git": payload.get("git") == dict(git),
        "sources": payload.get("source_hashes") == _sources(repo),
        "train_false": payload.get("train") is False,
        "validation_false": payload.get("validation") is False,
        "test_false": payload.get("test") is False,
    }
    if not all(checks.values()):
        raise B26ContractError(f"B26 preflight validation failed: {checks}")
    return {"path": str(path), "sha256": digest}


def _run(args: argparse.Namespace, repo: Path, git: Mapping[str, Any]) -> dict[str, object]:
    accepted = _accepted(args, repo, git)
    checkpoint, base_model, checkpoint_contract = _load_b9(args.b9_checkpoint)
    train_dataset, val_dataset = _datasets(args.data, include_validation=True)
    if val_dataset is None:
        raise B26ContractError("B26 formal run failed to construct validation")
    train_loader, sampler = _train_loader(train_dataset)
    val_loader = _eval_loader(val_dataset)
    device = torch.device("cuda")
    base_model.to(device).eval().requires_grad_(False)
    base_eval = _evaluate(base_model, val_loader, device=device)
    observed_confusion = tuple(
        tuple(int(value) for value in row)
        for row in base_eval["metrics"]["confusion_matrix"]
    )
    if observed_confusion != EXPECTED_B9_CONFUSION:
        raise B26ContractError("B26 B9 validation replay changed")

    _seed_all(SEED)
    model = build_b26(build_model_from_checkpoint(checkpoint)).to(device)
    initial_primary = _state_dict_sha256(model.backbone.state_dict())
    initial_query_norm = _query_norm(model)
    criterion = LDAMFocalLoss(
        class_counts=list(EXPECTED_CLASS_COUNTS),
        gamma=1.0,
        focal_mix=0.1,
        label_smoothing=0.02,
        max_margin=0.3,
        scale=18.0,
    ).to(device)
    optimizer = torch.optim.SGD(
        list(model.query_parameters()),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_warmup_decay_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_start_factor=0.1,
        total_epochs=EPOCHS,
        min_learning_rate=MIN_LR,
        decay_style="cosine",
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    total_updates = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        sums = {
            name: 0.0
            for name in ("task", "retention", "hard_negative", "distillation", "total")
        }
        samples = 0
        final_gradient_norm = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            images, labels = _split_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, trace = model.forward_with_trace(images, return_attention=False)
                loss, parts = boundary_guarded_loss(
                    logits,
                    trace["base_logits"],
                    labels,
                    criterion,
                    retention_weight=RETENTION_WEIGHT,
                    hard_negative_weight=HARD_NEGATIVE_WEIGHT,
                    distillation_weight=DISTILLATION_WEIGHT,
                    hard_negative_margin=HARD_NEGATIVE_MARGIN,
                    temperature=DISTILLATION_TEMPERATURE,
                )
            if not bool(torch.isfinite(loss).item()):
                raise B26ContractError(f"B26 non-finite loss at epoch {epoch + 1}")
            loss.backward()
            final_gradient_norm = _query_norm(model, gradients=True)
            clipped = torch.nn.utils.clip_grad_norm_(list(model.query_parameters()), GRAD_CLIP)
            if not bool(torch.isfinite(clipped).item()):
                raise B26ContractError("B26 query gradient became non-finite")
            optimizer.step()
            progress = epoch + batch_index / max(len(train_loader), 1)
            scheduler.step(progress)
            total_updates += 1
            count = int(labels.numel())
            samples += count
            for name in sums:
                sums[name] += float(parts[name].float().cpu()) * count
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                progress_payload = {
                    "protocol_id": PROTOCOL_ID,
                    "epoch": epoch + 1,
                    "epochs": EPOCHS,
                    "batch": batch_index,
                    "batches": len(train_loader),
                    "loss": float(loss.detach().float().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "wall_seconds": time.monotonic() - started,
                }
                _atomic_json(output / "progress.json", progress_payload)
                print(
                    f"B26 epoch={epoch + 1}/{EPOCHS} batch={batch_index}/{len(train_loader)} "
                    f"loss={progress_payload['loss']:.5f} lr={progress_payload['lr']:.3e}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_updates_total": total_updates,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "mean_losses": {
                    name: value / max(samples, 1) for name, value in sums.items()
                },
                "query_gradient_norm": final_gradient_norm,
                "query_norm": _query_norm(model),
            }
        )

    model.eval()
    candidate_eval = _evaluate(model, val_loader, device=device)
    final_primary = _state_dict_sha256(model.backbone.state_dict())
    if initial_primary != EXPECTED_SELECTED_STATE_SHA256 or final_primary != initial_primary:
        raise B26ContractError("B26 frozen B9 primary state changed")
    first_images = candidate_eval["first_images"]
    if first_images is None:
        raise B26ContractError("B26 validation trace batch is missing")
    trace_summary = _trace_summary(model, first_images.to(device))
    gate = _gate(base_eval["metrics"], candidate_eval["metrics"], trace_summary)

    state_path = output / "b26_evidence_query_final.safetensors"
    save_file(
        query_state_dict(model),
        str(state_path),
        metadata={
            "protocol_id": PROTOCOL_ID,
            "base_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "base_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        },
    )
    reload_model = build_b26(build_model_from_checkpoint(checkpoint))
    load_query_state_dict(reload_model, load_file(str(state_path), device="cpu"))
    reload_model.to(device).eval()
    with torch.inference_mode():
        reloaded = reload_model(first_images.to(device)).float().cpu().numpy()
    expected = candidate_eval["logits"][: int(first_images.size(0))]
    reload_max_abs = float(np.max(np.abs(reloaded - expected)))
    if reload_max_abs != 0.0:
        raise B26ContractError(f"B26 strict reload parity failed: {reload_max_abs}")

    prediction_path = output / "validation_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "label", "b9_prediction", "b26_prediction"])
        for path, label, base_pred, candidate_pred in zip(
            val_dataset.sample_paths(),
            candidate_eval["labels"],
            base_eval["predictions"],
            candidate_eval["predictions"],
        ):
            writer.writerow(
                [
                    path.resolve().relative_to(Path(args.data).resolve().parent).as_posix(),
                    label,
                    base_pred,
                    candidate_pred,
                ]
            )

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": dict(git),
        "accepted_preflight": accepted,
        "checkpoint": checkpoint_contract,
        "recipe": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "workers": WORKERS,
            "optimizer": "sgd",
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "scheduler_horizon": EPOCHS,
            "min_lr": MIN_LR,
            "gradient_clip": GRAD_CLIP,
            "retention_weight": RETENTION_WEIGHT,
            "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
            "hard_negative_margin": HARD_NEGATIVE_MARGIN,
            "distillation_weight": DISTILLATION_WEIGHT,
            "distillation_temperature": DISTILLATION_TEMPERATURE,
            "amp": "bf16_train_fp32_eval",
        },
        "data": {
            "train_rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VAL_ROWS,
            "test_rows": 0,
        },
        "parameters": {
            "total": TOTAL_PARAMETERS,
            "trainable": QUERY_PARAMETERS,
            "frozen": TOTAL_PARAMETERS - QUERY_PARAMETERS,
        },
        "training": {
            "history": history,
            "optimizer_updates": total_updates,
            "initial_query_norm": initial_query_norm,
            "final_query_norm": _query_norm(model),
            "primary_state_sha256": final_primary,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
        },
        "b9": base_eval["metrics"],
        "b26": candidate_eval["metrics"],
        "query_trace": trace_summary,
        "promotion_gate": gate,
        "artifacts": {
            "query_state": str(state_path),
            "query_state_sha256": sha256_file(state_path),
            "predictions": str(prediction_path),
            "predictions_sha256": sha256_file(prediction_path),
            "reload_max_abs": reload_max_abs,
        },
        "validation_opened": True,
        "validation_status": "exploratory_design_exposed",
        "test_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    repo = _repo()
    _configure_determinism()
    git = _git_snapshot(repo)
    output = args.output.expanduser().resolve()
    if args.preflight_only:
        if output.exists():
            raise B26ContractError(f"refuse to overwrite B26 preflight: {output}")
        if args.accepted_preflight is not None or args.accepted_preflight_sha256:
            raise B26ContractError("B26 preflight cannot consume another preflight")
        payload = _preflight(args, repo, git)
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / "preflight.json", payload)
        print(json.dumps({"passed": True, "output": str(output)}, indent=2))
        return 0
    if output.exists():
        raise B26ContractError(f"refuse to overwrite B26 output: {output}")
    summary = _run(args, repo, git)
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"gate": summary["promotion_gate"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from trkh.models.efficientvim_residual_student_b29 import (
    M1_PARAMETERS,
    RESIDUAL_SCALE,
    load_student_state_dict,
    student_state_dict,
)
from trkh.models.trust_margin_distillation_b30 import (
    CLASS1_WEIGHT,
    GAIN_CAP,
    trust_margin_distillation_loss,
)
from trkh.tools import run_b29_efficientvim_teacher_residual_fold as b29


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B30_TRUST_MARGIN_RESIDUAL_FOLD1_20260805"
FOLD = 1
SEED = 20260805
LOG_KEYS = ("task", "kd", "retention", "total")


class B30ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B30 TRAIN-only TMORD fold-1 screen")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--b24-dir", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _source_hashes() -> dict[str, str]:
    root = b29._repo()
    paths = {
        "runner": Path("trkh/tools/run_b30_trust_margin_residual_fold1.py"),
        "loss": Path("trkh/models/trust_margin_distillation_b30.py"),
        "student": Path("trkh/models/efficientvim_residual_student_b29.py"),
        "test": Path("tests/test_trust_margin_distillation_b30.py"),
        "b29_dependency": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
    }
    for path in paths.values():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0 or not (root / path).is_file():
            raise B30ContractError(f"B30 source missing or untracked: {path}")
    return {name: b29._sha256(root / path) for name, path in paths.items()}


def _teacher(args: argparse.Namespace) -> dict[str, Any]:
    retained = b29._load_b25(args)
    labels = np.asarray(retained["arrays"]["labels"], dtype=np.int64)
    folds = np.asarray(retained["arrays"]["folds"], dtype=np.int64)
    fit, held = folds != FOLD, folds == FOLD
    base = b29._fit_readout(retained["dino"], labels, fit)
    fusion = b29._fit_readout(
        np.concatenate((retained["dino"], retained["convnext"]), axis=1), labels, fit
    )
    if not base["converged"] or not fusion["converged"] or not held.any():
        raise B30ContractError("B30 fold-1 teacher/readout contract failed")
    metrics = {
        "base": b29.classification_summary(labels[held], base["scores"][held]),
        "fusion_teacher": b29.classification_summary(
            labels[held], fusion["scores"][held]
        ),
    }
    return {
        "labels": labels,
        "folds": folds,
        "fit": fit,
        "held": held,
        "base_scores": base["scores"],
        "teacher_scores": fusion["scores"],
        "iterations": {"base": base["iterations"], "fusion": fusion["iterations"]},
        "metrics": metrics,
    }


def _run_tests() -> str:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_trust_margin_distillation_b30.py"]
    result = subprocess.run(
        command, cwd=b29._repo(), capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        raise B30ContractError(f"B30 tests failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def _train_pair(
    control: torch.nn.Module,
    candidate: torch.nn.Module,
    loader: DataLoader,
    base_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    models = {"coupled_kl_control": control.to(device).train(), "tmord_candidate": candidate.to(device).train()}
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=b29.PEAK_LR, weight_decay=b29.WEIGHT_DECAY
        )
        for name, model in models.items()
    }
    total_updates = b29.EPOCHS * len(loader)
    warmup_updates = len(loader)
    history: list[dict[str, Any]] = []
    update, peak_allocated = 0, 0
    started = time.perf_counter()
    for epoch in range(b29.EPOCHS):
        sums = {name: {key: 0.0 for key in LOG_KEYS} for name in models}
        candidate_trace = {
            key: 0.0
            for key in ("positive_gain_fraction", "gain_mean", "class1_positive_gain_fraction")
        }
        batches = 0
        for images, labels, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            index = indices.long()
            primary = base_scores[index].to(device=device, non_blocking=True)
            target = teacher_scores[index].to(device=device, non_blocking=True)
            lr = b29.learning_rate(update, total_updates, warmup_updates)
            for name, model in models.items():
                optimizer = optimizers[name]
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                logits = model(images, primary)
                if name == "coupled_kl_control":
                    loss, parts = b29.residual_student_loss(logits, primary, target, labels)
                else:
                    loss, parts = trust_margin_distillation_loss(
                        logits, primary, target, labels
                    )
                    for key in candidate_trace:
                        candidate_trace[key] += float(parts[key].cpu())
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), b29.GRAD_CLIP)
                if not bool(torch.isfinite(norm)):
                    raise B30ContractError(f"B30 {name} gradient became non-finite")
                optimizer.step()
                for key in LOG_KEYS:
                    sums[name][key] += float(parts[key].cpu())
            update += 1
            batches += 1
            if device.type == "cuda":
                peak_allocated = max(
                    peak_allocated, int(torch.cuda.max_memory_allocated(device))
                )
            if batches % 40 == 0 or batches == len(loader):
                print(
                    f"B30 epoch={epoch + 1}/{b29.EPOCHS} batch={batches}/{len(loader)} "
                    f"lr={lr:.3e} kl={sums['coupled_kl_control']['total']/batches:.4f} "
                    f"tmord={sums['tmord_candidate']['total']/batches:.4f}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "mean_losses": {
                    name: {key: value / batches for key, value in parts.items()}
                    for name, parts in sums.items()
                },
                "tmord_trace": {
                    key: value / batches for key, value in candidate_trace.items()
                },
            }
        )
    if update != total_updates or b29.learning_rate(
        total_updates - 1, total_updates, warmup_updates
    ) != b29.MIN_LR:
        raise B30ContractError("B30 schedule horizon incomplete")
    return {
        "history": history,
        "optimizer_updates_per_arm": update,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak_allocated,
    }


def _gate(metrics: Mapping[str, Mapping[str, Any]], trace: Mapping[str, Any]) -> dict[str, Any]:
    base = metrics["base"]
    control = metrics["coupled_kl_control"]
    candidate = metrics["tmord_candidate"]
    deltas = {
        "class1_f1_vs_base": float(candidate["class1_f1"]) - float(base["class1_f1"]),
        "class1_f1_vs_control": float(candidate["class1_f1"]) - float(control["class1_f1"]),
        "macro_vs_control": float(candidate["macro_f1"]) - float(control["macro_f1"]),
        "accuracy_vs_control": float(candidate["accuracy"]) - float(control["accuracy"]),
        "tp_vs_control": int(candidate["class1_tp"]) - int(control["class1_tp"]),
        "restricted_fp_vs_control": int(candidate["restricted_fp"]) - int(control["restricted_fp"]),
    }
    checks = {
        "class1_gain_over_base": deltas["class1_f1_vs_base"] >= 0.005,
        "class1_gain_over_coupled_kl": deltas["class1_f1_vs_control"] >= 0.005,
        "macro_noninferiority": deltas["macro_vs_control"] >= -0.002,
        "accuracy_noninferiority": deltas["accuracy_vs_control"] >= -0.002,
        "class1_tp_recovery": deltas["tp_vs_control"] >= 0,
        "fp_tradeoff_bounded": deltas["restricted_fp_vs_control"] <= 3,
        "residual_active_bounded": 0.05 <= float(trace["tmord_candidate"]["abs_p95"]) < RESIDUAL_SCALE,
    }
    return {
        "passed": all(checks.values()),
        "next_unseen_train_fold_permission": all(checks.values()),
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "deltas": deltas,
    }


def _formal(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B30 output: {output}")
    source_start, git_start = _source_hashes(), b29._git_contract()
    if git_start["branch"] != "research/pretrained-classf-b1" or not git_start["tracked_worktree_clean"]:
        raise B30ContractError(f"B30 requires a clean pretrained branch: {git_start}")
    tests = _run_tests()
    teacher = _teacher(args)
    ledger = b29._prepare_b24_ledger(args)
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = b29._resolve_device(args.device)
    base_model, checkpoint_sha = b29._build_student(
        args.efficientvim_checkpoint.expanduser().resolve()
    )
    control, candidate = copy.deepcopy(base_model), copy.deepcopy(base_model)
    initial_sha = b29._state_dict_sha256(base_model.state_dict())
    if any(
        b29._state_dict_sha256(model.state_dict()) != initial_sha
        for model in (control, candidate)
    ):
        raise B30ContractError("B30 arms do not share exact initialization")
    print(
        json.dumps(
            {
                "preflight": "passed",
                "fold": FOLD,
                "fit_rows": int(teacher["fit"].sum()),
                "held_rows": int(teacher["held"].sum()),
                "teacher_metrics": teacher["metrics"],
            },
            indent=2,
        ),
        flush=True,
    )
    dataset = b29._TrainLedgerDataset(
        ledger["absolute_paths"], teacher["labels"], b29._efficientvim_transform()
    )
    fit_indices = np.flatnonzero(teacher["fit"]).astype(np.int64)
    held_indices = np.flatnonzero(teacher["held"]).astype(np.int64)
    counts = np.bincount(teacher["labels"][fit_indices], minlength=5).astype(np.float64)
    sample_weights = counts[teacher["labels"][fit_indices]] ** -0.5
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights),
        num_samples=fit_indices.size,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        Subset(dataset, fit_indices.tolist()),
        batch_size=b29.BATCH_SIZE,
        sampler=sampler,
        num_workers=b29.WORKERS,
        pin_memory=device.type == "cuda",
    )
    held_loader = DataLoader(
        Subset(dataset, held_indices.tolist()),
        batch_size=b29.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=b29.WORKERS,
        pin_memory=device.type == "cuda",
    )
    base_scores = torch.from_numpy(teacher["base_scores"])
    teacher_scores = torch.from_numpy(teacher["teacher_scores"])
    training = _train_pair(
        control, candidate, train_loader, base_scores, teacher_scores, device
    )
    models = {"coupled_kl_control": control, "tmord_candidate": candidate}
    held_scores, trace = b29._evaluate_pair(
        models, held_loader, base_scores, held_indices, device=device
    )
    held_labels = teacher["labels"][held_indices]
    metrics = {
        **teacher["metrics"],
        **{
            name: b29.classification_summary(held_labels, scores)
            for name, scores in held_scores.items()
        },
    }
    gate = _gate(metrics, trace)
    output.mkdir(parents=True, exist_ok=False)
    state_artifacts = {}
    for name, model in models.items():
        path = output / f"b30_{name}.safetensors"
        save_file(student_state_dict(model), str(path))
        state_artifacts[name] = {"path": str(path), "sha256": b29._sha256(path)}
    reloaded, _ = b29._build_student(args.efficientvim_checkpoint.expanduser().resolve())
    load_student_state_dict(
        reloaded, load_file(state_artifacts["tmord_candidate"]["path"], device="cpu")
    )
    reload_scores, _ = b29._evaluate_pair(
        {"tmord_candidate": reloaded},
        held_loader,
        base_scores,
        held_indices,
        device=device,
    )
    reload_max_abs = float(
        np.max(np.abs(reload_scores["tmord_candidate"] - held_scores["tmord_candidate"]))
    )
    source_end, git_end = _source_hashes(), b29._git_contract()
    integrity = source_start == source_end and git_start == git_end and reload_max_abs == 0.0
    if not integrity:
        raise B30ContractError("B30 source/Git/reload integrity failed")
    scores_sha = b29._atomic_npz(
        output / "fold1_scores.npz",
        indices=held_indices,
        labels=held_labels,
        base_scores=teacher["base_scores"][held_indices],
        fusion_teacher_scores=teacher["teacher_scores"][held_indices],
        coupled_kl_control_scores=held_scores["coupled_kl_control"],
        tmord_candidate_scores=held_scores["tmord_candidate"],
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_fold1_loss_ablation",
        "source_hashes": source_start,
        "git": {"start": git_start, "end": git_end},
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "fit_rows": int(fit_indices.size),
            "held_rows": int(held_indices.size),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "teacher": {"readout_iterations": teacher["iterations"]},
        "student": {
            "checkpoint_sha256": checkpoint_sha,
            "parameters": M1_PARAMETERS,
            "initial_state_sha256": initial_sha,
            "residual_scale": RESIDUAL_SCALE,
            "gain_cap": GAIN_CAP,
            "class1_weight": CLASS1_WEIGHT,
            "state_artifacts": state_artifacts,
            "reload_max_abs": reload_max_abs,
        },
        "focused_tests": tests,
        "training": training,
        "metrics": metrics,
        "trace": trace,
        "gate": gate,
        "scores_sha256": scores_sha,
        "integrity_complete": integrity,
    }
    b29._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    summary = _formal(args)
    print(
        json.dumps(
            {"gate": summary["gate"], "output": str(args.output_dir.resolve())},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

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
from trkh.tools import run_b29_efficientvim_teacher_residual_fold as b29


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B31_COMPLETE_COUPLED_KD_OOF_20260805"
NEW_FOLDS = (2, 3, 4)
SEED = 20260805
FOLD0_SCORES_SHA256 = "f9e0d9347a6295d6c9467b7186989e68a822291ca4c293f26c417f3d86787a8f"
FOLD1_SCORES_SHA256 = "29937d53ffa3215d63e95505c5b5fd803c317a459677ac194fc27a521f0b199d"


class B31ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete unchanged B29/B30 coupled-KL TRAIN OOF")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--b24-dir", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--fold0-scores", type=Path, required=True)
    parser.add_argument("--fold1-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _source_hashes() -> dict[str, str]:
    root = b29._repo()
    paths = {
        "runner": Path("trkh/tools/run_b31_complete_coupled_kd_oof.py"),
        "student": Path("trkh/models/efficientvim_residual_student_b29.py"),
        "loss_dependency": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
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
            raise B31ContractError(f"B31 source missing or untracked: {path}")
    return {name: b29._sha256(root / path) for name, path in paths.items()}


def _teacher(retained: dict[str, Any], fold: int) -> dict[str, Any]:
    labels = np.asarray(retained["arrays"]["labels"], dtype=np.int64)
    folds = np.asarray(retained["arrays"]["folds"], dtype=np.int64)
    fit, held = folds != fold, folds == fold
    base = b29._fit_readout(retained["dino"], labels, fit)
    fusion = b29._fit_readout(
        np.concatenate((retained["dino"], retained["convnext"]), axis=1), labels, fit
    )
    if not base["converged"] or not fusion["converged"] or not held.any():
        raise B31ContractError(f"B31 fold {fold} readout contract failed")
    return {
        "fit": fit,
        "held": held,
        "base_scores": base["scores"],
        "teacher_scores": fusion["scores"],
        "iterations": {"base": base["iterations"], "fusion": fusion["iterations"]},
    }


def _train(
    model: torch.nn.Module,
    loader: DataLoader,
    base_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    device: torch.device,
    fold: int,
) -> dict[str, Any]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=b29.PEAK_LR, weight_decay=b29.WEIGHT_DECAY
    )
    total_updates = b29.EPOCHS * len(loader)
    warmup_updates = len(loader)
    history, update = [], 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(b29.EPOCHS):
        sums = {key: 0.0 for key in ("task", "kd", "retention", "total")}
        batches = 0
        for images, labels, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            index = indices.long()
            primary = base_scores[index].to(device=device, non_blocking=True)
            target = teacher_scores[index].to(device=device, non_blocking=True)
            lr = b29.learning_rate(update, total_updates, warmup_updates)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, primary)
            loss, parts = b29.residual_student_loss(logits, primary, target, labels)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), b29.GRAD_CLIP)
            if not bool(torch.isfinite(norm)):
                raise B31ContractError(f"B31 fold {fold} gradient became non-finite")
            optimizer.step()
            for key in sums:
                sums[key] += float(parts[key].cpu())
            update += 1
            batches += 1
            if batches % 40 == 0 or batches == len(loader):
                print(
                    f"B31 fold={fold} epoch={epoch + 1}/{b29.EPOCHS} "
                    f"batch={batches}/{len(loader)} lr={lr:.3e} "
                    f"loss={sums['total']/batches:.4f}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "mean_losses": {key: value / batches for key, value in sums.items()},
            }
        )
    if update != total_updates or b29.learning_rate(
        total_updates - 1, total_updates, warmup_updates
    ) != b29.MIN_LR:
        raise B31ContractError(f"B31 fold {fold} schedule incomplete")
    return {
        "history": history,
        "updates": update,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }


def _load_prior(
    path: Path,
    expected_sha: str,
    fold: int,
    score_key: str,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or b29._sha256(resolved) != expected_sha:
        raise B31ContractError(f"B31 retained fold {fold} scores changed")
    with np.load(resolved, allow_pickle=False) as payload:
        data = {
            "indices": np.asarray(payload["indices"], dtype=np.int64),
            "labels": np.asarray(payload["labels"], dtype=np.int64),
            "base": np.asarray(payload["base_scores"], dtype=np.float32),
            "hybrid": np.asarray(payload[score_key], dtype=np.float32),
        }
    indices = data["indices"]
    if (
        not np.all(folds[indices] == fold)
        or not np.array_equal(labels[indices], data["labels"])
        or data["base"].shape != data["hybrid"].shape
        or data["base"].shape != (indices.size, 5)
    ):
        raise B31ContractError(f"B31 retained fold {fold} payload mismatch")
    return data


def _gate(
    confirmation: dict[str, dict[str, Any]],
    per_fold: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    base, hybrid = confirmation["base"], confirmation["coupled_kl"]
    deltas = {
        "accuracy": float(hybrid["accuracy"]) - float(base["accuracy"]),
        "macro_f1": float(hybrid["macro_f1"]) - float(base["macro_f1"]),
        "class1_f1": float(hybrid["class1_f1"]) - float(base["class1_f1"]),
        "class1_tp": int(hybrid["class1_tp"]) - int(base["class1_tp"]),
        "restricted_fp": int(hybrid["restricted_fp"]) - int(base["restricted_fp"]),
    }
    fold_wins = sum(
        float(values["coupled_kl"]["class1_f1"]) > float(values["base"]["class1_f1"])
        for values in per_fold.values()
    )
    checks = {
        "confirmation_class1_gain": deltas["class1_f1"] >= 0.010,
        "confirmation_macro_gain": deltas["macro_f1"] >= 0.005,
        "confirmation_accuracy_gain": deltas["accuracy"] >= 0.005,
        "confirmation_tp_retention": deltas["class1_tp"] >= 0,
        "confirmation_fp_reduction": deltas["restricted_fp"] <= -10,
        "fold_class1_wins": fold_wins >= 2,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "matched_exploratory_validation_permission": passed,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "deltas": deltas,
        "class1_fold_wins": fold_wins,
    }


def _formal(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B31 output: {output}")
    source_start, git_start = _source_hashes(), b29._git_contract()
    if git_start["branch"] != "research/pretrained-classf-b1" or not git_start["tracked_worktree_clean"]:
        raise B31ContractError(f"B31 requires a clean pretrained branch: {git_start}")
    retained = b29._load_b25(args)
    ledger = b29._prepare_b24_ledger(args)
    labels = np.asarray(retained["arrays"]["labels"], dtype=np.int64)
    folds = np.asarray(retained["arrays"]["folds"], dtype=np.int64)
    prior = {
        0: _load_prior(
            args.fold0_scores, FOLD0_SCORES_SHA256, 0, "candidate_scores", labels, folds
        ),
        1: _load_prior(
            args.fold1_scores,
            FOLD1_SCORES_SHA256,
            1,
            "coupled_kl_control_scores",
            labels,
            folds,
        ),
    }
    torch.set_num_threads(int(args.torch_threads))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = b29._resolve_device(args.device)
    dataset = b29._TrainLedgerDataset(
        ledger["absolute_paths"], labels, b29._efficientvim_transform()
    )
    output.mkdir(parents=True, exist_ok=False)
    all_base = np.full((labels.size, 5), np.nan, np.float32)
    all_hybrid = np.full((labels.size, 5), np.nan, np.float32)
    covered = np.zeros(labels.size, dtype=bool)
    for data in prior.values():
        all_base[data["indices"]] = data["base"]
        all_hybrid[data["indices"]] = data["hybrid"]
        covered[data["indices"]] = True
    training, states, readouts, per_fold = {}, {}, {}, {}
    initial_sha = None
    for fold in NEW_FOLDS:
        fold_teacher = _teacher(retained, fold)
        fit_indices = np.flatnonzero(fold_teacher["fit"]).astype(np.int64)
        held_indices = np.flatnonzero(fold_teacher["held"]).astype(np.int64)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model, checkpoint_sha = b29._build_student(
            args.efficientvim_checkpoint.expanduser().resolve()
        )
        state_sha = b29._state_dict_sha256(model.state_dict())
        if initial_sha is None:
            initial_sha = state_sha
        elif state_sha != initial_sha:
            raise B31ContractError("B31 fold initial states differ")
        counts = np.bincount(labels[fit_indices], minlength=5).astype(np.float64)
        weights = counts[labels[fit_indices]] ** -0.5
        loader = DataLoader(
            Subset(dataset, fit_indices.tolist()),
            batch_size=b29.BATCH_SIZE,
            sampler=WeightedRandomSampler(
                torch.from_numpy(weights),
                num_samples=fit_indices.size,
                replacement=True,
                generator=torch.Generator().manual_seed(SEED),
            ),
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
        base_tensor = torch.from_numpy(fold_teacher["base_scores"])
        training[str(fold)] = _train(
            model,
            loader,
            base_tensor,
            torch.from_numpy(fold_teacher["teacher_scores"]),
            device,
            fold,
        )
        scores, trace = b29._evaluate_pair(
            {"coupled_kl": model}, held_loader, base_tensor, held_indices, device=device
        )
        hybrid_scores = scores["coupled_kl"]
        per_fold[str(fold)] = {
            "base": b29.classification_summary(labels[held_indices], fold_teacher["base_scores"][held_indices]),
            "coupled_kl": b29.classification_summary(labels[held_indices], hybrid_scores),
            "trace": trace["coupled_kl"],
        }
        state_path = output / f"b31_fold{fold}_student.safetensors"
        save_file(student_state_dict(model), str(state_path))
        states[str(fold)] = {"path": str(state_path), "sha256": b29._sha256(state_path)}
        reloaded, _ = b29._build_student(args.efficientvim_checkpoint.expanduser().resolve())
        load_student_state_dict(reloaded, load_file(str(state_path), device="cpu"))
        reload_scores, _ = b29._evaluate_pair(
            {"coupled_kl": reloaded}, held_loader, base_tensor, held_indices, device=device
        )
        reload_error = float(np.max(np.abs(reload_scores["coupled_kl"] - hybrid_scores)))
        if reload_error != 0.0:
            raise B31ContractError(f"B31 fold {fold} reload mismatch: {reload_error}")
        states[str(fold)]["reload_max_abs"] = reload_error
        readouts[str(fold)] = fold_teacher["iterations"]
        b29._atomic_npz(
            output / f"fold{fold}_scores.npz",
            indices=held_indices,
            labels=labels[held_indices],
            base_scores=fold_teacher["base_scores"][held_indices],
            coupled_kl_scores=hybrid_scores,
        )
        if covered[held_indices].any():
            raise B31ContractError(f"B31 fold {fold} overlaps retained folds")
        all_base[held_indices] = fold_teacher["base_scores"][held_indices]
        all_hybrid[held_indices] = hybrid_scores
        covered[held_indices] = True
        model.cpu()
        reloaded.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not covered.all() or not np.isfinite(all_base).all() or not np.isfinite(all_hybrid).all():
        raise B31ContractError("B31 five-fold score coverage incomplete")
    confirmation_mask = np.isin(folds, NEW_FOLDS)
    confirmation = {
        "base": b29.classification_summary(labels[confirmation_mask], all_base[confirmation_mask]),
        "coupled_kl": b29.classification_summary(
            labels[confirmation_mask], all_hybrid[confirmation_mask]
        ),
    }
    full_oof = {
        "base": b29.classification_summary(labels, all_base),
        "coupled_kl": b29.classification_summary(labels, all_hybrid),
    }
    gate = _gate(confirmation, per_fold)
    oof_sha = b29._atomic_npz(
        output / "five_fold_oof_scores.npz",
        labels=labels,
        folds=folds,
        base_scores=all_base,
        coupled_kl_scores=all_hybrid,
    )
    source_end, git_end = _source_hashes(), b29._git_contract()
    integrity = source_start == source_end and git_start == git_end
    if not integrity:
        raise B31ContractError("B31 source/Git integrity failed")
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "train_only_coupled_kl_oof_completion_folds_2_3_4",
        "source_hashes": source_start,
        "git": {"start": git_start, "end": git_end},
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "rows": int(labels.size),
            "new_folds": list(NEW_FOLDS),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "prior_scores": {
            "fold0_sha256": FOLD0_SCORES_SHA256,
            "fold1_sha256": FOLD1_SCORES_SHA256,
        },
        "student": {
            "checkpoint_sha256": checkpoint_sha,
            "parameters": M1_PARAMETERS,
            "initial_state_sha256": initial_sha,
            "residual_scale": RESIDUAL_SCALE,
            "states": states,
        },
        "readout_iterations": readouts,
        "training": training,
        "per_new_fold": per_fold,
        "confirmation_folds_2_3_4": confirmation,
        "five_fold_oof": full_oof,
        "gate": gate,
        "oof_scores_sha256": oof_sha,
        "integrity_complete": integrity,
    }
    b29._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    summary = _formal(args)
    print(
        json.dumps(
            {"gate": summary["gate"], "output": str(args.output_dir.resolve())}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

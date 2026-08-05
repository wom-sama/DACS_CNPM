from __future__ import annotations

import argparse
import copy
import json
import math
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
from safetensors.torch import load_file, save_file  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler  # noqa: E402

from trkh.models.efficientvim_residual_student_b29 import (  # noqa: E402
    M1_PARAMETERS,
    RESIDUAL_SCALE,
    EfficientViMResidualStudent,
    load_student_state_dict,
    student_state_dict,
)
from trkh.tools.audit_dinov3_convnext_b24 import _prepare_b24_ledger  # noqa: E402
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (  # noqa: E402
    OFFICIAL_M1_E450_SHA256,
    SEED,
    _TrainLedgerDataset,
    _atomic_json,
    _atomic_npz,
    _build_efficientvim,
    _copy_efficientvim_backbone,
    _dependency_contract,
    _device_contract,
    _efficientvim_transform,
    _git_contract,
    _sha256,
    classification_summary,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import _state_dict_sha256  # noqa: E402
from trkh.tools.probe_embedding_prototypes import _resolve_device  # noqa: E402
from trkh.tools.resolve_b24_fusion_readout_b25 import _load as _load_b25  # noqa: E402


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B29_EFFICIENTVIM_TEACHER_RESIDUAL_FOLD0_20260805"
FOLD = 0
EPOCHS = 6
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
WORKERS = 0
PEAK_LR = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
TEMPERATURE = 4.0
TASK_WEIGHT = 1.0
KD_WEIGHT = 0.5
RETENTION_WEIGHT = 0.25
READOUT_MAX_ITER = 10_000


class B29ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B29 fold-0 EfficientViM mobile residual: DINO-target control versus "
            "DINO+ConvNeXt teacher target. TRAIN only; validation/test sealed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--b24-dir", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_hashes() -> dict[str, str]:
    root = _repo()
    paths = {
        "runner": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
        "model": Path("trkh/models/efficientvim_residual_student_b29.py"),
        "test": Path("tests/test_efficientvim_residual_student_b29.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "b13": Path("trkh/tools/audit_efficientvim_m1_frozen_transfer_b13.py"),
        "b24": Path("trkh/tools/audit_dinov3_convnext_b24.py"),
        "b25": Path("trkh/tools/resolve_b24_fusion_readout_b25.py"),
    }
    for path in paths.values():
        check = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0 or not (root / path).is_file():
            raise B29ContractError(f"B29 source missing or untracked: {path}")
    return {name: _sha256(root / path) for name, path in paths.items()}


def learning_rate(update: int, total_updates: int, warmup_updates: int) -> float:
    if not 0 <= int(update) < int(total_updates) or not 1 <= int(warmup_updates) < int(total_updates):
        raise ValueError("B29 LR requested outside locked horizon")
    if update < warmup_updates:
        return float(PEAK_LR * (update + 1) / warmup_updates)
    if update == total_updates - 1:
        return MIN_LR
    progress = (update - warmup_updates + 1) / (total_updates - warmup_updates)
    return float(MIN_LR + (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress)) / 2.0)


def residual_student_loss(
    logits: Tensor,
    primary_scores: Tensor,
    teacher_scores: Tensor,
    labels: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    task = F.cross_entropy(logits, labels)
    temperature = TEMPERATURE
    kd = F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        F.softmax(teacher_scores.detach() / temperature, dim=1),
        reduction="batchmean",
    ).clamp_min(0.0) * (temperature * temperature)
    rows = torch.arange(labels.numel(), device=labels.device)
    primary_other = primary_scores.detach().clone()
    primary_other[rows, labels] = -torch.inf
    logits_other = logits.clone()
    logits_other[rows, labels] = -torch.inf
    primary_margin = primary_scores.detach()[rows, labels] - primary_other.amax(dim=1)
    student_margin = logits[rows, labels] - logits_other.amax(dim=1)
    primary_correct = primary_scores.detach().argmax(dim=1).eq(labels)
    retention = (
        F.relu(primary_margin[primary_correct] - student_margin[primary_correct]).mean()
        if bool(primary_correct.any())
        else logits.sum() * 0.0
    )
    total = TASK_WEIGHT * task + KD_WEIGHT * kd + RETENTION_WEIGHT * retention
    return total, {
        "task": task.detach(),
        "kd": kd.detach(),
        "retention": retention.detach(),
        "total": total.detach(),
    }


def _fit_readout(features: np.ndarray, labels: np.ndarray, fit: np.ndarray) -> dict[str, Any]:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(features[fit].astype(np.float64, copy=False))
    x_all = scaler.transform(features.astype(np.float64, copy=False))
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        tol=1e-8,
        max_iter=READOUT_MAX_ITER,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x_fit, labels[fit])
    converged = not any(
        issubclass(record.category, ConvergenceWarning) for record in caught
    ) and int(np.max(classifier.n_iter_)) < READOUT_MAX_ITER
    return {
        "scores": classifier.decision_function(x_all).astype(np.float32),
        "iterations": classifier.n_iter_.astype(int).tolist(),
        "converged": converged,
    }


def _teacher_contract(args: argparse.Namespace) -> dict[str, Any]:
    retained = _load_b25(args)
    labels = np.asarray(retained["arrays"]["labels"], dtype=np.int64)
    folds = np.asarray(retained["arrays"]["folds"], dtype=np.int64)
    fit = folds != FOLD
    held = folds == FOLD
    base = _fit_readout(retained["dino"], labels, fit)
    fusion = _fit_readout(
        np.concatenate((retained["dino"], retained["convnext"]), axis=1), labels, fit
    )
    metrics = {
        "base": classification_summary(labels[held], base["scores"][held]),
        "fusion_teacher": classification_summary(labels[held], fusion["scores"][held]),
    }
    deltas = {
        name: float(metrics["fusion_teacher"][name]) - float(metrics["base"][name])
        for name in ("accuracy", "macro_f1", "class1_f1", "mean_pair_auroc")
    }
    checks = {
        "fold0_rows": int(held.sum()) == 1823 and int(fit.sum()) == 6455,
        "readouts_converged": bool(base["converged"] and fusion["converged"]),
        "teacher_accuracy_signal": deltas["accuracy"] >= 0.020,
        "teacher_macro_signal": deltas["macro_f1"] >= 0.020,
        "teacher_class1_signal": deltas["class1_f1"] >= 0.020,
        "teacher_pair_signal": deltas["mean_pair_auroc"] >= 0.010,
        "teacher_keeps_class1_tp": int(metrics["fusion_teacher"]["class1_tp"])
        >= int(metrics["base"]["class1_tp"]),
        "teacher_reduces_restricted_fp": int(metrics["fusion_teacher"]["restricted_fp"])
        < int(metrics["base"]["restricted_fp"]),
    }
    if not all(checks.values()):
        raise B29ContractError(f"B29 fold-0 teacher premise failed: {checks}")
    return {
        "retained": retained,
        "labels": labels,
        "folds": folds,
        "fit": fit,
        "held": held,
        "base_scores": base["scores"],
        "teacher_scores": fusion["scores"],
        "readout_iterations": {"base": base["iterations"], "fusion": fusion["iterations"]},
        "metrics": metrics,
        "deltas": deltas,
        "checks": checks,
    }


def _build_student(checkpoint: Path) -> tuple[EfficientViMResidualStudent, str]:
    source, digest = _build_efficientvim(checkpoint)
    student = _copy_efficientvim_backbone(source)
    model = EfficientViMResidualStudent(student)
    if sum(parameter.numel() for parameter in model.parameters()) != M1_PARAMETERS:
        raise B29ContractError("B29 student parameter count changed")
    return model, digest


def _run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_efficientvim_residual_student_b29.py"]
    result = subprocess.run(command, cwd=_repo(), capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise B29ContractError(f"B29 tests failed:\n{result.stdout}\n{result.stderr}")
    return {"command": command, "stdout": result.stdout.strip(), "passed": True}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    teacher = _teacher_contract(args)
    ledger = _prepare_b24_ledger(args)
    git = _git_contract()
    if git["branch"] != "research/pretrained-classf-b1" or not git["tracked_worktree_clean"]:
        raise B29ContractError(f"B29 requires clean pretrained branch: {git}")
    checkpoint = args.efficientvim_checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != OFFICIAL_M1_E450_SHA256:
        raise B29ContractError("B29 official EfficientViM checkpoint changed")
    control, digest = _build_student(checkpoint)
    candidate = copy.deepcopy(control)
    initial_hash = _state_dict_sha256(control.state_dict())
    parity_hash = _state_dict_sha256(candidate.state_dict())
    images = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(SEED))
    primary = torch.randn(2, 5, generator=torch.Generator().manual_seed(SEED + 1))
    labels = torch.tensor([0, 1])
    initial = control(images, primary)
    loss, _ = residual_student_loss(initial, primary, primary, labels)
    loss.backward()
    head_grad = sum(float(head.weight.grad.abs().sum()) for head in control.student.heads)
    stem = control.student.patch_embed.conv[0].conv.weight
    tests = _run_tests()
    checks = {
        "git_clean_pretrained_branch": True,
        "train_ledger_exact": True,
        "validation_not_constructed": True,
        "test_not_constructed": True,
        "teacher_premise": all(teacher["checks"].values()),
        "official_m1_weight": digest == OFFICIAL_M1_E450_SHA256,
        "matched_initial_state": initial_hash == parity_hash,
        "step0_primary_exact": torch.equal(initial, primary.float()),
        "step1_head_gradient": head_grad > 0.0,
        "step1_backbone_gradient_zero": stem.grad is not None and torch.count_nonzero(stem.grad) == 0,
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
            "train_rows": int(teacher["labels"].size),
            "fit_rows": int(teacher["fit"].sum()),
            "held_rows": int(teacher["held"].sum()),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": ledger["train_content"],
        },
        "teacher": {
            "metrics": teacher["metrics"],
            "deltas": teacher["deltas"],
            "readout_iterations": teacher["readout_iterations"],
            "control_target": "fold0_fit_raw_dino_readout",
            "candidate_target": "fold0_fit_dino_convnext_fusion_readout",
        },
        "student": {
            "architecture": "EfficientViM-M1_canonical_four_stage_fusion",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "parameters": M1_PARAMETERS,
            "initial_state_sha256": initial_hash,
            "residual_scale": RESIDUAL_SCALE,
        },
        "recipe": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "peak_lr": PEAK_LR,
            "min_lr": MIN_LR,
            "warmup_epochs": 1,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "task_weight": TASK_WEIGHT,
            "kd_weight": KD_WEIGHT,
            "retention_weight": RETENTION_WEIGHT,
        },
        "focused_tests": tests,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise B29ContractError(f"B29 preflight failed: {checks}")
    return payload


def _accepted(args: argparse.Namespace, teacher: Mapping[str, Any]) -> dict[str, Any]:
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B29 requires accepted preflight path and SHA-256")
    path = args.preflight_artifact.expanduser().resolve()
    digest = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_deltas = payload.get("teacher", {}).get("deltas", {})
    delta_match = isinstance(recorded_deltas, Mapping) and set(recorded_deltas) == set(
        teacher["deltas"]
    ) and all(
        math.isclose(
            float(recorded_deltas[name]),
            float(teacher["deltas"][name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name in teacher["deltas"]
    )
    checks = {
        "sha": digest == args.preflight_sha256.lower(),
        "passed": payload.get("passed") is True,
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "sources": payload.get("source_hashes") == _source_hashes(),
        "git": payload.get("git") == _git_contract(),
        "teacher_deltas": delta_match,
    }
    if not all(checks.values()):
        raise B29ContractError(f"B29 accepted preflight invalid: {checks}")
    return {"path": str(path), "sha256": digest, "checks": checks}


def _train_pair(
    control: EfficientViMResidualStudent,
    candidate: EfficientViMResidualStudent,
    loader: DataLoader,
    base_scores: Tensor,
    teacher_scores: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    models = {"control": control.to(device).train(), "candidate": candidate.to(device).train()}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
        for name, model in models.items()
    }
    total_updates = EPOCHS * len(loader)
    warmup_updates = len(loader)
    update = 0
    history: list[dict[str, Any]] = []
    peak_allocated = 0
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        sums = {name: {key: 0.0 for key in ("task", "kd", "retention", "total")} for name in models}
        batches = 0
        for images, labels, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            index = indices.to(dtype=torch.long)
            primary = base_scores[index].to(device=device, non_blocking=True)
            targets = {
                "control": primary,
                "candidate": teacher_scores[index].to(device=device, non_blocking=True),
            }
            lr = learning_rate(update, total_updates, warmup_updates)
            for name in ("control", "candidate"):
                optimizer = optimizers[name]
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                logits = models[name](images, primary)
                loss, parts = residual_student_loss(logits, primary, targets[name], labels)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(models[name].parameters(), GRAD_CLIP)
                if not bool(torch.isfinite(norm)):
                    raise B29ContractError(f"B29 {name} gradient became non-finite")
                optimizer.step()
                for key in sums[name]:
                    sums[name][key] += float(parts[key].cpu())
            update += 1
            batches += 1
            if device.type == "cuda":
                peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            if batches % 40 == 0 or batches == len(loader):
                print(
                    f"B29 epoch={epoch + 1}/{EPOCHS} batch={batches}/{len(loader)} "
                    f"lr={lr:.3e} control={sums['control']['total']/batches:.4f} "
                    f"candidate={sums['candidate']['total']/batches:.4f}",
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
            }
        )
    if update != total_updates or learning_rate(total_updates - 1, total_updates, warmup_updates) != MIN_LR:
        raise B29ContractError("B29 schedule horizon incomplete")
    return {
        "history": history,
        "optimizer_updates_per_arm": update,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": peak_allocated,
    }


def _evaluate_pair(
    models: Mapping[str, EfficientViMResidualStudent],
    loader: DataLoader,
    base_scores: Tensor,
    held_indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    positions = {int(index): position for position, index in enumerate(held_indices.tolist())}
    scores = {name: np.full((held_indices.size, 5), np.nan, np.float32) for name in models}
    residual_values = {name: [] for name in models}
    for model in models.values():
        model.to(device).eval()
    with torch.inference_mode():
        for images, _labels, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            primary = base_scores[indices.long()].to(device=device, non_blocking=True)
            target_positions = np.asarray([positions[int(value)] for value in indices.tolist()], dtype=np.int64)
            for name, model in models.items():
                logits, trace = model.forward_with_trace(images, primary)
                scores[name][target_positions] = logits.cpu().numpy()
                residual_values[name].append(trace["residual_scores"].abs().cpu().numpy())
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise B29ContractError("B29 held scores incomplete/non-finite")
    trace_summary = {}
    for name, chunks in residual_values.items():
        values = np.concatenate(chunks).reshape(-1)
        trace_summary[name] = {
            "abs_p50": float(np.quantile(values, 0.50)),
            "abs_p95": float(np.quantile(values, 0.95)),
            "abs_max": float(values.max()),
        }
    return scores, trace_summary


def _gate(metrics: Mapping[str, Mapping[str, Any]], trace: Mapping[str, Any]) -> dict[str, Any]:
    base, control, candidate = metrics["base"], metrics["control"], metrics["candidate"]
    deltas = {
        "class1_f1_vs_base": float(candidate["class1_f1"]) - float(base["class1_f1"]),
        "class1_f1_vs_control": float(candidate["class1_f1"]) - float(control["class1_f1"]),
        "macro_f1_vs_control": float(candidate["macro_f1"]) - float(control["macro_f1"]),
        "accuracy_vs_control": float(candidate["accuracy"]) - float(control["accuracy"]),
        "class1_tp_vs_base": int(candidate["class1_tp"]) - int(base["class1_tp"]),
        "restricted_fp_vs_control": int(candidate["restricted_fp"]) - int(control["restricted_fp"]),
    }
    teacher_gain = float(metrics["fusion_teacher"]["class1_f1"]) - float(base["class1_f1"])
    checks = {
        "candidate_beats_base_class1": deltas["class1_f1_vs_base"] >= 0.005,
        "local_teacher_beats_control_class1": deltas["class1_f1_vs_control"] >= 0.010,
        "local_teacher_beats_control_macro": deltas["macro_f1_vs_control"] >= 0.005,
        "accuracy_noninferiority": deltas["accuracy_vs_control"] >= -0.002,
        "class1_tp_retention": deltas["class1_tp_vs_base"] >= 0,
        "restricted_fp_reduction": deltas["restricted_fp_vs_control"] <= -3,
        "teacher_gain_recovery": deltas["class1_f1_vs_base"] >= 0.25 * teacher_gain,
        "control_not_collapsed": float(control["macro_f1"]) >= float(base["macro_f1"]) - 0.020,
        "residual_active_bounded": 0.05 <= float(trace["candidate"]["abs_p95"]) < RESIDUAL_SCALE,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "five_fold_oof_permission": passed,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
        "deltas": deltas,
    }


def _formal(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B29 output: {output}")
    teacher = _teacher_contract(args)
    accepted = _accepted(args, teacher)
    ledger = _prepare_b24_ledger(args)
    source_start, git_start = _source_hashes(), _git_contract()
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    checkpoint = args.efficientvim_checkpoint.expanduser().resolve()
    base_model, digest = _build_student(checkpoint)
    control = copy.deepcopy(base_model)
    candidate = copy.deepcopy(base_model)
    initial_hash = _state_dict_sha256(base_model.state_dict())
    if _state_dict_sha256(control.state_dict()) != initial_hash or _state_dict_sha256(candidate.state_dict()) != initial_hash:
        raise B29ContractError("B29 arms do not share exact initialization")
    dataset = _TrainLedgerDataset(ledger["absolute_paths"], teacher["labels"], _efficientvim_transform())
    fit_indices = np.flatnonzero(teacher["fit"]).astype(np.int64)
    held_indices = np.flatnonzero(teacher["held"]).astype(np.int64)
    fit_subset = Subset(dataset, fit_indices.tolist())
    counts = np.bincount(teacher["labels"][fit_indices], minlength=5).astype(np.float64)
    sample_weights = counts[teacher["labels"][fit_indices]] ** -0.5
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights),
        num_samples=fit_indices.size,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        fit_subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    held_loader = DataLoader(
        Subset(dataset, held_indices.tolist()),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    base_scores = torch.from_numpy(teacher["base_scores"])
    teacher_scores = torch.from_numpy(teacher["teacher_scores"])
    training = _train_pair(
        control,
        candidate,
        train_loader,
        base_scores,
        teacher_scores,
        device=device,
    )
    held_scores, trace = _evaluate_pair(
        {"control": control, "candidate": candidate},
        held_loader,
        base_scores,
        held_indices,
        device=device,
    )
    held_labels = teacher["labels"][held_indices]
    metrics = {
        **teacher["metrics"],
        "control": classification_summary(held_labels, held_scores["control"]),
        "candidate": classification_summary(held_labels, held_scores["candidate"]),
    }
    gate = _gate(metrics, trace)
    output.mkdir(parents=True, exist_ok=False)
    state_paths = {}
    for name, model in (("control", control), ("candidate", candidate)):
        path = output / f"b29_{name}_student.safetensors"
        save_file(student_state_dict(model), str(path))
        state_paths[name] = {"path": str(path), "sha256": _sha256(path)}
    reload_candidate, _ = _build_student(checkpoint)
    load_student_state_dict(reload_candidate, load_file(state_paths["candidate"]["path"], device="cpu"))
    reload_scores, _ = _evaluate_pair(
        {"candidate": reload_candidate},
        held_loader,
        base_scores,
        held_indices,
        device=device,
    )
    reload_max_abs = float(np.max(np.abs(reload_scores["candidate"] - held_scores["candidate"])))
    source_end, git_end = _source_hashes(), _git_contract()
    integrity = bool(source_start == source_end and git_start == git_end and reload_max_abs == 0.0)
    if not integrity:
        raise B29ContractError("B29 source/Git/reload integrity failed")
    oof_sha = _atomic_npz(
        output / "fold0_scores.npz",
        indices=held_indices,
        labels=held_labels,
        base_scores=teacher["base_scores"][held_indices],
        fusion_teacher_scores=teacher["teacher_scores"][held_indices],
        control_scores=held_scores["control"],
        candidate_scores=held_scores["candidate"],
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_fold0_mobile_residual_distillation",
        "accepted_preflight": accepted,
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
        "teacher": {
            "readout_iterations": teacher["readout_iterations"],
            "deltas": teacher["deltas"],
        },
        "student": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "parameters": M1_PARAMETERS,
            "initial_state_sha256": initial_hash,
            "residual_scale": RESIDUAL_SCALE,
            "state_artifacts": state_paths,
            "reload_max_abs": reload_max_abs,
        },
        "training": training,
        "metrics": metrics,
        "trace": trace,
        "gate": gate,
        "scores_sha256": oof_sha,
        "integrity_complete": integrity,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.preflight_only:
        if output.exists():
            raise FileExistsError(f"refuse to overwrite B29 preflight: {output}")
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

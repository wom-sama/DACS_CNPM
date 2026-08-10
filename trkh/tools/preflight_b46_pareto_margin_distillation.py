"""Metric-free B46 preflight for clean-anchored Pareto-margin distillation.

The preflight reuses ten fixed TRAIN-side B43 calibration bases and its verified
fold-4 control/robust endpoints.  It tests whether one ordinary DINO graph can
move toward a label-safe robust logit target while retaining the clean control
function.  It reads no held labels, validation, test, F1, or accuracy and saves
no checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import run_b43_progressive_function_merge as b43
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.training.pareto_margin_distillation import (
    centered_logits,
    clean_anchored_dual_teacher_loss,
    pareto_project_robust_teacher,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B46_PARETO_MARGIN_DISTILLATION_PREFLIGHT_20260809"
SEED = 20260809
EXCLUDED_FOLD = 4
BASES_PER_CLASS = 2
BATCH_SIZE = 5
STEPS = 30
LEARNING_RATE = 1e-5
MAX_GRADIENT_NORM = 1.0
MAX_FINAL_ROBUST_RATIO = 0.85
MAX_CLEAN_TO_INITIAL_ROBUST = 0.25
MAX_PEAK_CUDA_BYTES = 6 * 1024**3

CONTROL_FILE_SHA256 = "c069cfd6b36e675447de4df6502f334e58a2ddcab433064cc747c92c17d30d9e"
CONTROL_STATE_SHA256 = "827687796b8f64f227cce8770e21a73db11406b36edcfb17f8dea685ed0cad63"
ROBUST_FILE_SHA256 = "cb355b0e32728667bd899cd673580429b683504e9bf122e5913c020dd8adc0b0"
ROBUST_STATE_SHA256 = "2f6bc6352ba9bfa0cc2c93f1c30359468dcab560b396a24152d992a973cf592e"


class B46ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--robust-checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/preflight_b46_pareto_margin_distillation.py"),
        "primitive": Path("trkh/training/pareto_margin_distillation.py"),
        "primitive_test": Path("tests/test_pareto_margin_distillation.py"),
        "model": Path("trkh/models/dinov3_multidepth_convpass_b21.py"),
        "model_builder": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "view_builder": Path("trkh/tools/run_b43_progressive_function_merge.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    *,
    device: torch.device,
    gradients: bool,
) -> Tensor:
    rows: list[Tensor] = []
    context = torch.enable_grad() if gradients else torch.inference_mode()
    with context:
        for start in range(0, int(images.size(0)), BATCH_SIZE):
            batch = images[start : start + BATCH_SIZE].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            rows.append(logits.float() if gradients else logits.float().cpu())
    return torch.cat(rows, dim=0)


def _loss_snapshot(
    model: nn.Module,
    clean: Tensor,
    robust: Tensor,
    control_clean: Tensor,
    projected_robust: Tensor,
    labels: Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, float], Tensor, Tensor]:
    model.eval()
    clean_logits = _forward_logits(model, clean, device=device, gradients=False)
    robust_logits = _forward_logits(model, robust, device=device, gradients=False)
    losses = clean_anchored_dual_teacher_loss(
        clean_logits,
        robust_logits,
        control_clean,
        projected_robust,
        labels,
    )
    values = {
        "total": float(losses.total),
        "clean_logits": float(losses.clean_logits),
        "clean_focus_margin": float(losses.clean_focus_margin),
        "robust_logits": float(losses.robust_logits),
    }
    return values, clean_logits, robust_logits


def _gradient_roles(model: nn.Module) -> dict[str, float]:
    squared: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("backbone.blocks."):
            parts = name.split(".")
            role = f"block_{parts[2]}"
        elif name.startswith("backbone.head"):
            role = "head"
        elif name.startswith("backbone.patch_embed"):
            role = "patch_embed"
        else:
            role = "other_backbone"
        squared[role] = squared.get(role, 0.0) + float(
            parameter.grad.detach().float().square().sum().cpu()
        )
    norms = {name: math.sqrt(value) for name, value in squared.items()}
    total = math.sqrt(sum(value for value in squared.values()))
    norms["total"] = total
    return norms


def _non_target_geometry_error(
    before: Tensor,
    after: Tensor,
    labels: Tensor,
) -> float:
    maximum = 0.0
    for row, label in enumerate(labels.tolist()):
        indices = [index for index in range(int(before.size(1))) if index != int(label)]
        source = before[row, indices]
        target = after[row, indices]
        source_pairs = source[:, None] - source[None, :]
        target_pairs = target[:, None] - target[None, :]
        maximum = max(maximum, float((source_pairs - target_pairs).abs().max()))
    return maximum


def _state_delta(
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
) -> dict[str, float]:
    if set(before) != set(after):
        raise B46ContractError("Student state keys changed during preflight.")
    squared = 0.0
    maximum = 0.0
    changed = 0
    for name in before:
        if not before[name].is_floating_point():
            if not torch.equal(before[name], after[name]):
                raise B46ContractError(f"Non-floating state changed: {name}")
            continue
        difference = after[name].detach().float().cpu() - before[name].detach().float().cpu()
        norm = float(difference.norm())
        if norm > 0.0:
            changed += 1
        squared += norm * norm
        maximum = max(maximum, float(difference.abs().max()))
    return {"l2": math.sqrt(squared), "max_abs": maximum, "changed_tensors": changed}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise B46ContractError("B46 preflight requires CUDA BF16.")
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    if not bool(git.get("clean")):
        raise B46ContractError("B46 preflight requires a clean committed worktree.")
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    clean, robust, records = b43._build_calibration_views(
        train_root,
        assignment,
        excluded_fold=EXCLUDED_FOLD,
        bases_per_class=BASES_PER_CLASS,
    )
    labels = torch.tensor([int(row["class_index"]) for row in records], dtype=torch.long)
    if tuple(torch.bincount(labels, minlength=5).tolist()) != (2, 2, 2, 2, 2):
        raise B46ContractError("Preflight calibration is not exactly class balanced.")

    control_state = b43._load_verified_state(
        args.control_checkpoint,
        file_sha=CONTROL_FILE_SHA256,
        state_sha=CONTROL_STATE_SHA256,
    )
    robust_state = b43._load_verified_state(
        args.robust_checkpoint,
        file_sha=ROBUST_FILE_SHA256,
        state_sha=ROBUST_STATE_SHA256,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    teacher = b21.build_arm_model(args.dino_weight, DIRECT_MODE)
    teacher.set_mode(DIRECT_MODE)
    teacher.load_state_dict(control_state, strict=True)
    teacher.to(device).eval().requires_grad_(False)
    control_clean = _forward_logits(teacher, clean, device=device, gradients=False)
    control_robust = _forward_logits(teacher, robust, device=device, gradients=False)
    teacher.load_state_dict(robust_state, strict=True)
    robust_robust = _forward_logits(teacher, robust, device=device, gradients=False)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    target = pareto_project_robust_teacher(control_robust, robust_robust, labels)
    projected = target.logits.detach().cpu()
    margin_tolerance = 2e-6
    projection_checks = {
        "true_margin_dominates_control": bool(
            torch.all(target.projected_true_margin >= target.control_true_margin - margin_tolerance)
        ),
        "true_margin_dominates_robust": bool(
            torch.all(target.projected_true_margin >= target.robust_true_margin - margin_tolerance)
        ),
        "focus_margin_dominates_control": bool(
            torch.all(target.projected_focus_margin >= target.control_focus_margin - margin_tolerance)
        ),
        "focus_margin_dominates_robust": bool(
            torch.all(target.projected_focus_margin >= target.robust_focus_margin - margin_tolerance)
        ),
        "non_target_geometry_exact": _non_target_geometry_error(
            robust_robust,
            projected,
            labels,
        )
        <= 1e-6,
    }

    student = b21.build_arm_model(args.dino_weight, DIRECT_MODE)
    student.set_mode(DIRECT_MODE)
    student.load_state_dict(control_state, strict=True)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_("adapter_" not in name)
    student.to(device).eval()
    initial_state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in student.state_dict().items()
    }
    initial, initial_clean_logits, initial_robust_logits = _loss_snapshot(
        student,
        clean,
        robust,
        control_clean,
        projected,
        labels,
        device=device,
    )
    initial_clean_replay = float((initial_clean_logits - control_clean).abs().max())
    if initial_clean_replay != 0.0:
        raise B46ContractError(f"Control initialization replay changed: {initial_clean_replay}")
    target_distance = float(
        torch.mean((centered_logits(projected) - centered_logits(control_robust)).square())
    )

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    first_gradient_roles: dict[str, float] | None = None
    steps: list[dict[str, float]] = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        accumulated = {"total": 0.0, "clean_logits": 0.0, "clean_focus_margin": 0.0, "robust_logits": 0.0}
        for start in range(0, int(clean.size(0)), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, int(clean.size(0)))
            weight = float(stop - start) / float(clean.size(0))
            clean_batch = clean[start:stop].to(device)
            robust_batch = robust[start:stop].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_clean = student(clean_batch).float()
                student_robust = student(robust_batch).float()
            loss = clean_anchored_dual_teacher_loss(
                student_clean,
                student_robust,
                control_clean[start:stop].to(device),
                projected[start:stop].to(device),
                labels[start:stop].to(device),
            )
            (loss.total * weight).backward()
            for name in accumulated:
                accumulated[name] += float(getattr(loss, name).detach().cpu()) * weight
        if first_gradient_roles is None:
            first_gradient_roles = _gradient_roles(student)
        preclip = float(torch.nn.utils.clip_grad_norm_(trainable, MAX_GRADIENT_NORM))
        if not math.isfinite(preclip):
            raise B46ContractError("Student gradient became non-finite.")
        optimizer.step()
        steps.append({"step": float(step + 1), "preclip_gradient_norm": preclip, **accumulated})

    final, final_clean_logits, final_robust_logits = _loss_snapshot(
        student,
        clean,
        robust,
        control_clean,
        projected,
        labels,
        device=device,
    )
    final_state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in student.state_dict().items()
    }
    state_delta = _state_delta(initial_state, final_state)
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    initial_robust = max(initial["robust_logits"], 1e-12)
    robust_ratio = final["robust_logits"] / initial_robust
    clean_ratio = (final["clean_logits"] + final["clean_focus_margin"]) / initial_robust
    clean_argmax_exact = bool(
        torch.equal(final_clean_logits.argmax(dim=1), control_clean.argmax(dim=1))
    )
    robust_function_changed = float((final_robust_logits - initial_robust_logits).abs().max())
    checks = {
        **projection_checks,
        "robust_target_nonzero": target_distance > 1e-7,
        "projection_active": bool(torch.count_nonzero(target.true_class_shift > 0).item() > 0),
        "robust_fit_ratio": robust_ratio <= MAX_FINAL_ROBUST_RATIO,
        "clean_trust_ratio": clean_ratio <= MAX_CLEAN_TO_INITIAL_ROBUST,
        "clean_argmax_exact": clean_argmax_exact,
        "student_function_changed": robust_function_changed > 1e-6,
        "student_parameters_changed": state_delta["l2"] > 0.0,
        "peak_cuda_within_budget": peak_cuda_bytes <= MAX_PEAK_CUDA_BYTES,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "git": git,
        "source_hashes": _source_hashes(repo),
        "scope": {
            "train_images_used": True,
            "held_labels_used": False,
            "classification_metric_used": False,
            "validation_used": False,
            "test_used": False,
            "checkpoint_saved": False,
            "full_train_permission": False,
        },
        "inputs": {
            "assignment_csv": str(args.assignment_csv),
            "assignment_sha256": b21.sha256_file(args.assignment_csv),
            "dino_weight": str(args.dino_weight),
            "dino_sha256": b21.sha256_file(args.dino_weight),
            "control_checkpoint": str(args.control_checkpoint),
            "control_sha256": b21.sha256_file(args.control_checkpoint),
            "robust_checkpoint": str(args.robust_checkpoint),
            "robust_sha256": b21.sha256_file(args.robust_checkpoint),
        },
        "calibration": {
            "excluded_fold": EXCLUDED_FOLD,
            "bases_per_class": BASES_PER_CLASS,
            "records": records,
        },
        "target": {
            "equation": (
                "robust logits plus the minimum true-class shift needed to weakly "
                "dominate control and robust true/focus margins; non-target geometry fixed"
            ),
            "distance_from_control_centered_mse": target_distance,
            "shifted_rows": int(torch.count_nonzero(target.true_class_shift > 0).item()),
            "shift_mean": float(target.true_class_shift.mean()),
            "shift_max": float(target.true_class_shift.max()),
            "non_target_geometry_max_abs": _non_target_geometry_error(
                robust_robust,
                projected,
                labels,
            ),
        },
        "optimization": {
            "student_initialization": "exact_control_endpoint",
            "student_graph": "ordinary direct DINO; dormant ConvPass parameters frozen",
            "mode": "eval_deterministic_gradients",
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "loss_weights": {
                "clean_centered_logits": 1.0,
                "clean_focus_margin": 1.0,
                "projected_robust_centered_logits": 1.0,
            },
            "initial": initial,
            "final": final,
            "final_to_initial_robust_ratio": robust_ratio,
            "clean_to_initial_robust_ratio": clean_ratio,
            "initial_clean_replay_max_abs": initial_clean_replay,
            "robust_function_change_max_abs": robust_function_changed,
            "first_gradient_roles": first_gradient_roles,
            "state_delta": state_delta,
            "step_trace": steps,
        },
        "resources": {
            "peak_cuda_bytes": peak_cuda_bytes,
            "peak_cuda_gib": peak_cuda_bytes / float(1024**3),
            "maximum_cuda_bytes": MAX_PEAK_CUDA_BYTES,
        },
        "next_permission": (
            "one_fresh_train_fold_screen_only"
            if all(checks.values())
            else "close_exact_b46_preflight_without_metric_run"
        ),
    }
    del student, optimizer, initial_state, final_state, control_state, robust_state
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    args.control_checkpoint = args.control_checkpoint.expanduser().resolve(strict=True)
    args.robust_checkpoint = args.robust_checkpoint.expanduser().resolve(strict=True)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B46ContractError("DINOv3 weight changed.")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must not already exist: {output}")
    output.mkdir(parents=True, exist_ok=False)
    b21._configure_determinism()
    b21._seed_all(SEED)
    payload = _run(args)
    artifact = output / "preflight.json"
    b21._atomic_json(artifact, payload)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"]}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

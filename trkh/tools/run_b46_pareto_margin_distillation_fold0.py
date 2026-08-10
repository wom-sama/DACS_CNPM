"""Run the one-use B46 clean-anchored Pareto-margin TRAIN-fold screen."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import preflight_b46_pareto_margin_distillation as b46
from trkh.tools import run_b36_head_first_dino_fold as b36
from trkh.tools import run_b38_relit_bicar_fold2 as b38
from trkh.tools import run_b39_stratified_pair_bicar_fold3 as b39
from trkh.tools import run_b41_backbone_routed_pair_mean_fold3 as b41
from trkh.tools import run_b43_progressive_function_merge as b43
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.training.pareto_margin_distillation import (
    centered_logits,
    clean_anchored_dual_teacher_loss,
    pareto_project_robust_teacher,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B46_PARETO_MARGIN_DISTILLATION_FOLD0_20260809"
FOLD = 0
SEED = 42
ENDPOINT_EPOCHS = 9
ENDPOINT_HORIZON = 30
ENDPOINT_START_EPOCH = 3
BASES_PER_CLASS = 16
CONSOLIDATION_STEPS = 30
SOURCES_PER_CLASS_PER_STEP = 2
MICROBATCH_SIZE = 5
LEARNING_RATE = b46.LEARNING_RATE
MAX_GRADIENT_NORM = b46.MAX_GRADIENT_NORM
MAX_CUDA_BYTES = 6 * 1024**3
PREFLIGHT_SHA256 = "80fd9071395cce798bd53c5fc8a04668cd5f3ee927a74eee3b454c34ef86d424"
PREFLIGHT_HEAD = "27b7def7ca3c4bda75cf2efbf43d915b74292415"


class B46FormalContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path, required=True)
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b46_pareto_margin_distillation_fold0.py"),
        "preflight_runner": Path("trkh/tools/preflight_b46_pareto_margin_distillation.py"),
        "primitive": Path("trkh/training/pareto_margin_distillation.py"),
        "model": Path("trkh/models/dinov3_multidepth_convpass_b21.py"),
        "trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "endpoint_objective": Path("trkh/tools/run_b41_backbone_routed_pair_mean_fold3.py"),
        "endpoint_common": Path("trkh/tools/run_b39_stratified_pair_bicar_fold3.py"),
        "evaluator": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "view_builder": Path("trkh/tools/run_b43_progressive_function_merge.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "primitive_test": Path("tests/test_pareto_margin_distillation.py"),
        "runner_test": Path("tests/test_run_b46_pareto_margin_distillation_fold0.py"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _read_preflight(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    digest = b21.sha256_file(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    scope = payload.get("scope", {})
    checks = {
        "sha256": digest == PREFLIGHT_SHA256,
        "protocol": payload.get("protocol_id") == b46.PROTOCOL_ID,
        "passed": payload.get("passed") is True,
        "source_head": payload.get("git", {}).get("head") == PREFLIGHT_HEAD,
        "no_validation": scope.get("validation_used") is False,
        "no_test": scope.get("test_used") is False,
        "no_checkpoint": scope.get("checkpoint_saved") is False,
        "next_permission": payload.get("next_permission") == "one_fresh_train_fold_screen_only",
    }
    if not all(checks.values()):
        raise B46FormalContractError(f"B46 preflight is not accepted: {checks}")
    return {"path": str(resolved), "sha256": digest, "source_head": PREFLIGHT_HEAD}


def _balanced_step_indices(labels: Tensor, step: int) -> list[int]:
    labels = labels.detach().cpu().to(dtype=torch.long)
    selected: list[int] = []
    for class_index in range(5):
        positions = torch.nonzero(labels == class_index, as_tuple=False).flatten().tolist()
        if len(positions) != BASES_PER_CLASS:
            raise B46FormalContractError(
                f"Class {class_index} calibration count changed: {len(positions)}"
            )
        offset = (int(step) * SOURCES_PER_CLASS_PER_STEP) % len(positions)
        for rank in range(SOURCES_PER_CLASS_PER_STEP):
            selected.append(int(positions[(offset + rank) % len(positions)]))
    return selected


def _load_model(
    dino_weight: Path,
    state: Mapping[str, Tensor],
    *,
    device: torch.device,
    trainable: bool,
) -> torch.nn.Module:
    model = b21.build_arm_model(dino_weight, DIRECT_MODE)
    model.set_mode(DIRECT_MODE)
    model.load_state_dict(state, strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(trainable and "adapter_" not in name))
    return model.to(device).eval()


def _teacher_targets(
    *,
    dino_weight: Path,
    control_state: Mapping[str, Tensor],
    robust_state: Mapping[str, Tensor],
    clean: Tensor,
    robust: Tensor,
    labels: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    teacher = _load_model(dino_weight, control_state, device=device, trainable=False)
    control_clean = b46._forward_logits(teacher, clean, device=device, gradients=False)
    control_robust = b46._forward_logits(teacher, robust, device=device, gradients=False)
    teacher.load_state_dict(robust_state, strict=True)
    robust_robust = b46._forward_logits(teacher, robust, device=device, gradients=False)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    target = pareto_project_robust_teacher(control_robust, robust_robust, labels)
    projected = target.logits.detach().cpu()
    target_mse = float(
        torch.mean((centered_logits(projected) - centered_logits(control_robust)).square())
    )
    metadata = {
        "equation": (
            "robust logits plus minimum true-logit shift dominating control/robust "
            "true and focus margins; robust non-target geometry unchanged"
        ),
        "distance_from_control_centered_mse": target_mse,
        "shifted_rows": int(torch.count_nonzero(target.true_class_shift > 0).item()),
        "shift_mean": float(target.true_class_shift.mean()),
        "shift_max": float(target.true_class_shift.max()),
        "non_target_geometry_max_abs": b46._non_target_geometry_error(
            robust_robust,
            projected,
            labels,
        ),
        "true_margin_dominates_control": bool(
            torch.all(target.projected_true_margin >= target.control_true_margin - 2e-6)
        ),
        "true_margin_dominates_robust": bool(
            torch.all(target.projected_true_margin >= target.robust_true_margin - 2e-6)
        ),
        "focus_margin_dominates_control": bool(
            torch.all(target.projected_focus_margin >= target.control_focus_margin - 2e-6)
        ),
        "focus_margin_dominates_robust": bool(
            torch.all(target.projected_focus_margin >= target.robust_focus_margin - 2e-6)
        ),
    }
    if (
        target_mse <= 1e-7
        or metadata["non_target_geometry_max_abs"] > 1e-6
        or not all(value for key, value in metadata.items() if key.endswith(("control", "robust")))
    ):
        raise B46FormalContractError(f"Formal teacher target failed: {metadata}")
    return control_clean, projected, metadata


def _consolidate(
    *,
    dino_weight: Path,
    control_state: Mapping[str, Tensor],
    clean: Tensor,
    robust: Tensor,
    labels: Tensor,
    control_clean: Tensor,
    projected_robust: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    device = torch.device("cuda")
    student = _load_model(dino_weight, control_state, device=device, trainable=True)
    initial, initial_clean, _initial_robust = b46._loss_snapshot(
        student,
        clean,
        robust,
        control_clean,
        projected_robust,
        labels,
        device=device,
    )
    clean_replay = float((initial_clean - control_clean).abs().max())
    if clean_replay != 0.0:
        raise B46FormalContractError(f"Student is not control-exact: {clean_replay}")
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    trace: list[dict[str, Any]] = []
    first_gradient_roles: dict[str, float] | None = None
    torch.cuda.reset_peak_memory_stats()
    for step in range(CONSOLIDATION_STEPS):
        indices = _balanced_step_indices(labels, step)
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "total": 0.0,
            "clean_logits": 0.0,
            "clean_focus_margin": 0.0,
            "robust_logits": 0.0,
        }
        for start in range(0, len(indices), MICROBATCH_SIZE):
            positions = indices[start : start + MICROBATCH_SIZE]
            weight = float(len(positions)) / float(len(indices))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_clean = student(clean[positions].to(device)).float()
                student_robust = student(robust[positions].to(device)).float()
            loss = clean_anchored_dual_teacher_loss(
                student_clean,
                student_robust,
                control_clean[positions].to(device),
                projected_robust[positions].to(device),
                labels[positions].to(device),
            )
            (loss.total * weight).backward()
            for name in totals:
                totals[name] += float(getattr(loss, name).detach().cpu()) * weight
        if first_gradient_roles is None:
            first_gradient_roles = b46._gradient_roles(student)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, MAX_GRADIENT_NORM))
        if not math.isfinite(gradient_norm):
            raise B46FormalContractError("B46 consolidation gradient became non-finite.")
        optimizer.step()
        trace.append(
            {
                "step": step + 1,
                "source_indices": indices,
                "preclip_gradient_norm": gradient_norm,
                **totals,
            }
        )

    final, final_clean, final_robust = b46._loss_snapshot(
        student,
        clean,
        robust,
        control_clean,
        projected_robust,
        labels,
        device=device,
    )
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in student.state_dict().items()
    }
    state_delta = b46._state_delta(control_state, state)
    initial_robust_loss = max(initial["robust_logits"], 1e-12)
    robust_ratio = final["robust_logits"] / initial_robust_loss
    clean_ratio = (final["clean_logits"] + final["clean_focus_margin"]) / initial_robust_loss
    per_row_exposure = np.bincount(
        np.asarray([index for row in trace for index in row["source_indices"]], dtype=np.int64),
        minlength=int(labels.numel()),
    )
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    telemetry = {
        "student_initialization": "exact_control_endpoint",
        "student_graph": "ordinary direct DINO; dormant ConvPass parameters frozen",
        "steps": CONSOLIDATION_STEPS,
        "sources_per_class_per_step": SOURCES_PER_CLASS_PER_STEP,
        "source_views_total": int(per_row_exposure.sum()),
        "per_row_exposure_min": int(per_row_exposure.min()),
        "per_row_exposure_max": int(per_row_exposure.max()),
        "microbatch_size": MICROBATCH_SIZE,
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
        "initial_clean_replay_max_abs": clean_replay,
        "final_clean_argmax_exact_on_calibration": bool(
            torch.equal(final_clean.argmax(dim=1), control_clean.argmax(dim=1))
        ),
        "robust_function_change_max_abs": float((final_robust - _initial_robust).abs().max()),
        "first_gradient_roles": first_gradient_roles,
        "state_delta": state_delta,
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_gib": peak_cuda_bytes / float(1024**3),
        "maximum_cuda_bytes": MAX_CUDA_BYTES,
        "trace": trace,
    }
    mechanical_checks = {
        "robust_fit_ratio": robust_ratio <= b46.MAX_FINAL_ROBUST_RATIO,
        "clean_trust_ratio": clean_ratio <= b46.MAX_CLEAN_TO_INITIAL_ROBUST,
        "calibration_clean_argmax_exact": telemetry[
            "final_clean_argmax_exact_on_calibration"
        ],
        "exact_sample_budget": telemetry["source_views_total"]
        == CONSOLIDATION_STEPS * 5 * SOURCES_PER_CLASS_PER_STEP,
        "balanced_exposure": telemetry["per_row_exposure_max"]
        - telemetry["per_row_exposure_min"]
        <= 1,
        "parameters_changed": state_delta["l2"] > 0.0,
        "cuda_budget": peak_cuda_bytes <= MAX_CUDA_BYTES,
    }
    telemetry["mechanical_checks"] = mechanical_checks
    telemetry["mechanical_passed"] = bool(all(mechanical_checks.values()))
    if not telemetry["mechanical_passed"]:
        raise B46FormalContractError(f"B46 consolidation mechanics failed: {mechanical_checks}")
    del student, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return state, telemetry


def _formal(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_preflight = _read_preflight(args.preflight_artifact)
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    train_fit, train_held, held_paths, fit_labels = b21._build_datasets(
        train_root,
        assignment,
        fold=FOLD,
    )
    class_counts = b39._fit_counts(assignment, fold=FOLD)
    control_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=ENDPOINT_EPOCHS,
        scheduler_horizon=ENDPOINT_HORIZON,
        seed=SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b46_control_ema",
        state_hash_epochs=(ENDPOINT_START_EPOCH - 1,),
    )
    endpoint_objective = b41._objective_factory(class_counts)
    robust_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=ENDPOINT_EPOCHS,
        scheduler_horizon=ENDPOINT_HORIZON,
        seed=SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b46_backbone_routed_pair_mean_ema",
        auxiliary_loss_fn=endpoint_objective,
        auxiliary_loss_weight=b41.AUXILIARY_WEIGHT,
        state_hash_epochs=(ENDPOINT_START_EPOCH - 1,),
    )
    warmup_replay = b38._matched_warmup(control_training, robust_training)
    if not warmup_replay["passed"]:
        raise B46FormalContractError(f"Endpoint warmup replay failed: {warmup_replay}")
    control_record = control_training.get("checkpoint")
    robust_record = robust_training.get("checkpoint")
    if not isinstance(control_record, Mapping) or not isinstance(robust_record, Mapping):
        raise B46FormalContractError("B46 endpoint checkpoint is missing.")
    control_state = b43._load_checkpoint_record(control_record)
    robust_state = b43._load_checkpoint_record(robust_record)

    clean, robust, records = b43._build_calibration_views(
        train_root,
        assignment,
        excluded_fold=FOLD,
        bases_per_class=BASES_PER_CLASS,
    )
    labels = torch.tensor([int(row["class_index"]) for row in records], dtype=torch.long)
    if tuple(torch.bincount(labels, minlength=5).tolist()) != (16, 16, 16, 16, 16):
        raise B46FormalContractError("Formal calibration is not exactly balanced.")
    device = torch.device("cuda")
    control_clean, projected_robust, target_metadata = _teacher_targets(
        dino_weight=args.dino_weight,
        control_state=control_state,
        robust_state=robust_state,
        clean=clean,
        robust=robust,
        labels=labels,
        device=device,
    )
    candidate_state, consolidation = _consolidate(
        dino_weight=args.dino_weight,
        control_state=control_state,
        clean=clean,
        robust=robust,
        labels=labels,
        control_clean=control_clean,
        projected_robust=projected_robust,
    )
    candidate_record = b43._save_safetensors_atomic(
        output / "b46_pareto_margin.safetensors",
        candidate_state,
    )

    datasets = {
        name: b36._held_subset(
            train_root,
            assignment,
            brightness=brightness,
            contrast=contrast,
            fold=FOLD,
        )
        for name, (brightness, contrast) in b41.CONDITIONS.items()
    }
    control_metrics, _control_features, control_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(control_record["path"])),
        expected_state_sha256=str(control_record["state_sha256"]),
        datasets=datasets,
        seed_offset=1200,
    )
    robust_metrics, _robust_features, robust_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(robust_record["path"])),
        expected_state_sha256=str(robust_record["state_sha256"]),
        datasets=datasets,
        seed_offset=1300,
    )
    candidate_metrics, _candidate_features, candidate_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(candidate_record["path"])),
        expected_state_sha256=str(candidate_record["state_sha256"]),
        datasets=datasets,
        seed_offset=1400,
    )
    labels_held = np.asarray(control_training["held_labels"], dtype=np.int64)
    if not np.array_equal(labels_held, np.asarray(robust_training["held_labels"], dtype=np.int64)):
        raise B46FormalContractError("Endpoint held-label order changed.")
    control_replay = float(
        np.max(
            np.abs(
                control_logits["clean"].numpy()
                - np.asarray(control_training["held_logits"], dtype=np.float32)
            )
        )
    )
    robust_replay = float(
        np.max(
            np.abs(
                robust_logits["clean"].numpy()
                - np.asarray(robust_training["held_logits"], dtype=np.float32)
            )
        )
    )
    if control_replay != 0.0 or robust_replay != 0.0:
        raise B46FormalContractError(
            f"Endpoint clean replay changed: {control_replay}, {robust_replay}"
        )
    gate = b41._gate(control_metrics, candidate_metrics)
    gate["next_permission"] = (
        "confirm_fixed_b46_equation_on_remaining_train_folds"
        if gate["passed"]
        else "close_exact_b46_without_validation_or_test"
    )
    score_path = output / "held_condition_scores.npz"
    arrays: dict[str, np.ndarray] = {"labels": labels_held}
    for condition in b41.CONDITIONS:
        arrays[f"control_{condition}"] = control_logits[condition].numpy()
        arrays[f"raw_robust_{condition}"] = robust_logits[condition].numpy()
        arrays[f"pareto_{condition}"] = candidate_logits[condition].numpy()
    b36._atomic_npz(score_path, **arrays)
    bright_ema = endpoint_objective.bright_state.ema_confusion
    dim_ema = endpoint_objective.dim_state.ema_confusion
    if bright_ema is None or dim_ema is None or endpoint_objective.active_calls <= 0:
        raise B46FormalContractError("Robust endpoint objective never became active.")

    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": dict(git),
        "source_hashes": _source_hashes(repo),
        "accepted_preflight": accepted_preflight,
        "dataset": {
            "fold": FOLD,
            "fit_rows": len(train_fit),
            "held_rows": len(train_held),
            "fit_class_counts": class_counts,
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "fold_status": "new_to_B38_B46_robust_sequence_but_historically_design_exposed",
        },
        "endpoint_schedule": {
            "seed": SEED,
            "epochs": ENDPOINT_EPOCHS,
            "scheduler_horizon": ENDPOINT_HORIZON,
            "warmup_replay": warmup_replay,
            "objective": b41._objective_contract(),
        },
        "control_training": b21._sanitize_arm_result(control_training),
        "raw_robust_training": b21._sanitize_arm_result(robust_training),
        "calibration": {
            "selection": "reuse fixed B43 hash key; no metric-selected rows",
            "excluded_fold": FOLD,
            "bases_per_class": BASES_PER_CLASS,
            "labels_used_for_balancing_and_teacher_margin_projection_only": True,
            "classification_metric_used": False,
            "records": records,
        },
        "target": target_metadata,
        "consolidation": consolidation,
        "candidate_checkpoint": candidate_record,
        "auxiliary_state": {
            "active_calls": endpoint_objective.active_calls,
            "bright_updates": endpoint_objective.bright_state.updates,
            "dim_updates": endpoint_objective.dim_state.updates,
            "bright_ema_confusion": bright_ema.detach().float().cpu().tolist(),
            "dim_ema_confusion": dim_ema.detach().float().cpu().tolist(),
        },
        "metrics": {
            "control": control_metrics,
            "raw_robust": robust_metrics,
            "pareto": candidate_metrics,
        },
        "strict_replay_max_abs": {
            "control": control_replay,
            "raw_robust": robust_replay,
        },
        "scores_sha256": b21.sha256_file(score_path),
        "gate": gate,
        "interpretation_guard": (
            "TRAIN component fold 0 only. It is new to the B38-B46 robust sequence "
            "but historically design-exposed by other mechanisms. Validation/test remain closed."
        ),
    }
    b21._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    args.preflight_artifact = args.preflight_artifact.expanduser().resolve(strict=True)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B46FormalContractError("DINOv3 weight changed.")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must not already exist: {output}")
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    if not bool(git.get("clean")):
        raise B46FormalContractError("Formal B46 requires a clean committed worktree.")
    b21._configure_determinism()
    output.mkdir(parents=True, exist_ok=False)
    summary = _formal(args, output, repo=repo, git=git)
    print(json.dumps(summary["gate"], indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

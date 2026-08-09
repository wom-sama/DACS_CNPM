"""Run a matched TRAIN-fold screen of relighting-conditioned BiCAR.

B38 keeps the strong B35 DINOv3 optimization recipe and changes only the
candidate training objective from epoch three onward.  The candidate evaluates
the existing bidirectional confusion-spectral regularizer on a luminance-only
relit view.  Fold 2 is held from both arms; validation and test are never read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import run_b36_head_first_dino_fold as b36
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.training.confusion_spectral import (
    ConfusionSpectralEMAState,
    confusion_aware_spectral_regularizer,
)
from trkh.training.losses import LDAMFocalLoss
from trkh.training.relighting import balanced_polarities, relight_luminance


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B38_RELIT_BICAR_FOLD2_20260809"
FOLD = 2
SEED = 42
AUXILIARY_SEED = 20260809
EPOCHS = 9
SCHEDULER_HORIZON = 30
START_EPOCH = 3
AUXILIARY_WEIGHT = 0.5
EMA_MOMENTUM = 0.5
FREQUENCY_SMOOTHING = 0.2
MARGIN = 0.1
RELIGHT_BRIGHTNESS = 0.25
RELIGHT_CONTRAST = 0.10
B35_CHECKPOINT_SHA256 = "b29bc28f5e59e5690dd3e99bba93b7de7edf5023fa834bdb76a12821ff50e548"
B35_STATE_SHA256 = "0a4fc1972811be0af800bac969ee0f31f37816f768c1cc2914832f54f249b9a1"
CONDITIONS = {
    "clean": (1.0, 1.0),
    "lighting_dim": (0.70, 0.90),
    "lighting_bright": (1.25, 1.10),
}


class B38ContractError(RuntimeError):
    pass


@dataclass
class RelitBiCARObjective:
    class_counts: tuple[int, ...]
    state: ConfusionSpectralEMAState = field(default_factory=ConfusionSpectralEMAState)
    active_calls: int = 0

    def __call__(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        _clean_logits: Tensor,
        epoch_index: int,
        batch_index: int,
    ) -> tuple[Tensor, Mapping[str, float]]:
        if int(epoch_index) + 1 < START_EPOCH:
            return _clean_logits.sum() * 0.0, {"active": 0.0}
        if images.device.type != "cuda":
            raise B38ContractError("B38 relit objective requires CUDA.")
        device_index = images.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        auxiliary_seed = (
            AUXILIARY_SEED + int(epoch_index) * 100_000 + int(batch_index)
        )
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(auxiliary_seed)
            polarities = balanced_polarities(
                int(images.size(0)),
                device=images.device,
                dtype=images.dtype,
            )
            relit_images = relight_luminance(
                images,
                polarities,
                brightness=RELIGHT_BRIGHTNESS,
                contrast=RELIGHT_CONTRAST,
            )
            relit_logits = model(relit_images)
        result = confusion_aware_spectral_regularizer(
            relit_logits,
            labels,
            class_counts=self.class_counts,
            previous_ema=self.state.ema_confusion,
            momentum=EMA_MOMENTUM,
            smoothing=FREQUENCY_SMOOTHING,
            margin=MARGIN,
            bidirectional=True,
        )
        self.state.update(result.ema_confusion)
        self.active_calls += 1
        return result.loss, {
            "active": 1.0,
            "batch_confusion_norm": float(
                torch.linalg.matrix_norm(result.batch_confusion.float(), ord=2)
                .detach()
                .cpu()
                .item()
            ),
            "ema_confusion_norm": float(
                torch.linalg.matrix_norm(result.ema_confusion.float(), ord=2)
                .detach()
                .cpu()
                .item()
            ),
            "weighted_confusion_norm": float(result.loss.detach().float().cpu().item()),
        }


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--mature-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b38_relit_bicar_fold2.py"),
        "trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "evaluator": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "confusion": Path("trkh/training/confusion_spectral.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "runner_test": Path("tests/test_run_b38_relit_bicar_fold2.py"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _read_preflight(
    path: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    checks = {
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "passed": payload.get("passed") is True,
        "scope": payload.get("train") is False
        and payload.get("validation") is False
        and payload.get("test") is False,
        "git": payload.get("git") == dict(git),
        "source_hashes": payload.get("source_hashes") == _source_hashes(repo),
    }
    if not all(checks.values()):
        raise B38ContractError(f"B38 preflight is not accepted: {checks}")
    return {"path": str(resolved), "sha256": b21.sha256_file(resolved)}


def _fit_counts(assignment: Mapping[str, Any]) -> tuple[int, ...]:
    fit = np.asarray(assignment["folds"]) != FOLD
    return tuple(
        int(value)
        for value in np.bincount(np.asarray(assignment["labels"])[fit], minlength=5)
    )


def _first_fit_batch(
    fit_dataset: torch.utils.data.Dataset,
    fit_labels: Sequence[int],
) -> tuple[Tensor, Tensor]:
    b21._seed_all(SEED)
    sampler = b21.TemperedClassBatchSampler(
        fit_labels,
        batch_size=b21.BATCH_SIZE,
        num_classes=5,
        power=0.5,
        seed=SEED,
        drop_last=False,
    )
    loader = DataLoader(fit_dataset, batch_sampler=sampler, num_workers=0)
    return b21._split_batch(next(iter(loader)))


def _oracle_directions(class_counts: Sequence[int]) -> dict[str, Any]:
    logits = torch.tensor(
        [
            [0.0, 2.0, -2.0, -3.0, -2.0],
            [2.0, 0.0, -2.0, -3.0, -2.0],
            [-2.0, -2.0, 3.0, -3.0, -2.0],
            [-2.0, -2.0, -2.0, 3.0, -2.0],
            [-2.0, -2.0, -2.0, -3.0, 3.0],
        ],
        requires_grad=True,
    )
    targets = torch.arange(5)
    result = confusion_aware_spectral_regularizer(
        logits,
        targets,
        class_counts=class_counts,
        momentum=EMA_MOMENTUM,
        smoothing=FREQUENCY_SMOOTHING,
        margin=MARGIN,
        bidirectional=True,
    )
    gradient = torch.autograd.grad(result.loss, logits)[0]
    checks = {
        "true0_wrong1_suppressed": bool(gradient[0, 1] > 0 and gradient[0, 0] < 0),
        "true1_wrong0_rescued": bool(gradient[1, 1] < 0 and gradient[1, 0] > 0),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "true0_gradient": gradient[0].tolist(),
        "true1_gradient": gradient[1].tolist(),
    }


def _run_preflight(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    fit_dataset, _held_dataset, _held_paths, fit_labels = b21._build_datasets(
        train_root,
        assignment,
        fold=FOLD,
    )
    class_counts = _fit_counts(assignment)
    images, labels = _first_fit_batch(fit_dataset, fit_labels)
    label_counts = torch.bincount(labels, minlength=5)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise B38ContractError("B38 requires CUDA BF16.")
    device = torch.device("cuda")
    model = b21.build_arm_model(args.dino_weight, DIRECT_MODE)
    model.set_mode(DIRECT_MODE)
    state = load_file(str(args.mature_checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    state_match = b21.state_sha256(model.state_dict()) == B35_STATE_SHA256
    model.to(device).train()
    images = images.to(device)
    labels = labels.to(device)
    criterion = LDAMFocalLoss(
        class_counts=class_counts,
        gamma=1.0,
        focal_mix=0.1,
        label_smoothing=0.02,
        max_margin=0.3,
        scale=18.0,
    ).to(device)
    objective = RelitBiCARObjective(class_counts=class_counts)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clean_logits = model(images)
        task_loss = criterion(clean_logits, labels)
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone()
        auxiliary_loss, auxiliary_stats = objective(
            model,
            images,
            labels,
            clean_logits,
            START_EPOCH - 1,
            0,
        )
        cpu_rng_preserved = torch.equal(cpu_rng_before, torch.random.get_rng_state())
        cuda_rng_preserved = torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device))
        weighted_auxiliary = AUXILIARY_WEIGHT * auxiliary_loss
    alignment = b21.gradient_alignment_by_role(model, task_loss, weighted_auxiliary)
    combined = task_loss + weighted_auxiliary
    combined.backward()
    oracle = _oracle_directions(class_counts)
    gradient_checks = {
        role: bool(
            values["task_norm"] > 0.0
            and values["auxiliary_norm"] > 0.0
            and 0.01 <= values["auxiliary_over_task"] <= 0.75
            and values["cosine"] >= -0.25
        )
        for role, values in alignment.items()
        if role in {"backbone", "head"}
    }
    checks = {
        "mature_state_exact": state_match,
        "all_classes_in_batch": bool((label_counts > 0).all().item()),
        "cpu_rng_preserved": cpu_rng_preserved,
        "cuda_rng_preserved": cuda_rng_preserved,
        "task_loss_finite": bool(torch.isfinite(task_loss).item()),
        "auxiliary_loss_finite_positive": bool(
            torch.isfinite(auxiliary_loss).item() and auxiliary_loss.item() > 0.0
        ),
        "ema_updated_once": objective.state.updates == 1,
        "oracle_directions": bool(oracle["passed"]),
        "backbone_gradient_scale": gradient_checks.get("backbone", False),
        "head_gradient_scale": gradient_checks.get("head", False),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": dict(git),
        "source_hashes": _source_hashes(repo),
        "train": False,
        "validation": False,
        "test": False,
        "fold": FOLD,
        "fit_rows": len(fit_dataset),
        "fit_class_counts": class_counts,
        "first_batch_class_counts": label_counts.tolist(),
        "mature_checkpoint": {
            "path": str(args.mature_checkpoint),
            "sha256": b21.sha256_file(args.mature_checkpoint),
            "expected_sha256": B35_CHECKPOINT_SHA256,
        },
        "objective": _objective_contract(),
        "task_loss": float(task_loss.detach().float().cpu().item()),
        "auxiliary_loss": float(auxiliary_loss.detach().float().cpu().item()),
        "auxiliary_stats": dict(auxiliary_stats),
        "gradient_alignment": alignment,
        "oracle": oracle,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    b21._atomic_json(output / "preflight.json", payload)
    del model, state, images, labels, clean_logits, combined
    gc.collect()
    torch.cuda.empty_cache()
    if not payload["passed"]:
        raise B38ContractError(f"B38 preflight failed: {checks}")
    return payload


def _objective_contract() -> dict[str, Any]:
    return {
        "name": "relighting_conditioned_bidirectional_confusion_spectral",
        "start_epoch": START_EPOCH,
        "weight": AUXILIARY_WEIGHT,
        "ema_momentum": EMA_MOMENTUM,
        "frequency_smoothing": FREQUENCY_SMOOTHING,
        "margin": MARGIN,
        "bidirectional": True,
        "brightness": RELIGHT_BRIGHTNESS,
        "contrast": RELIGHT_CONTRAST,
        "chroma_policy": "preserve_rgb_differences_until_clipping",
        "rng_policy": "seeded_auxiliary_fork_restores_cpu_and_active_cuda_rng",
    }


def _gate(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    clean_control = control["clean"]
    clean_candidate = candidate["clean"]
    dim_delta = float(
        candidate["lighting_dim"]["class1_f1"]
        - control["lighting_dim"]["class1_f1"]
    )
    bright_delta = float(
        candidate["lighting_bright"]["class1_f1"]
        - control["lighting_bright"]["class1_f1"]
    )
    control_bright_fp = int(control["lighting_bright"]["confusion_matrix"][0][1])
    candidate_bright_fp = int(candidate["lighting_bright"]["confusion_matrix"][0][1])
    bright_fp_reduction = (control_bright_fp - candidate_bright_fp) / max(
        control_bright_fp,
        1,
    )
    deltas = {
        "clean_accuracy": float(clean_candidate["accuracy"] - clean_control["accuracy"]),
        "clean_macro_f1": float(clean_candidate["macro_f1"] - clean_control["macro_f1"]),
        "clean_class1_f1": float(
            clean_candidate["class1_f1"] - clean_control["class1_f1"]
        ),
        "clean_pair_auroc": float(
            clean_candidate["mean_pair_auroc"] - clean_control["mean_pair_auroc"]
        ),
        "clean_class1_tp": int(clean_candidate["class1_tp"] - clean_control["class1_tp"]),
        "clean_restricted_fp": int(
            clean_candidate["restricted_fp"] - clean_control["restricted_fp"]
        ),
        "dim_class1_f1": dim_delta,
        "bright_class1_f1": bright_delta,
        "robust_mean_class1_f1": float((dim_delta + bright_delta) / 2.0),
        "bright_zero_to_one": candidate_bright_fp - control_bright_fp,
        "bright_zero_to_one_reduction_fraction": float(bright_fp_reduction),
    }
    checks = {
        "clean_class1_gain": deltas["clean_class1_f1"] >= 0.005,
        "clean_macro_noninferiority": deltas["clean_macro_f1"] >= -0.002,
        "clean_accuracy_noninferiority": deltas["clean_accuracy"] >= -0.002,
        "clean_pair_noninferiority": deltas["clean_pair_auroc"] >= -0.003,
        "clean_tp_noninferiority": deltas["clean_class1_tp"] >= 0,
        "clean_fp_nonincrease": deltas["clean_restricted_fp"] <= 0,
        "dim_gain": deltas["dim_class1_f1"] >= 0.010,
        "bright_gain": deltas["bright_class1_f1"] >= 0.015,
        "robust_mean_gain": deltas["robust_mean_class1_f1"] >= 0.015,
        "bright_zero_to_one_reduction": (
            deltas["bright_zero_to_one_reduction_fraction"] >= 0.10
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "deltas": deltas,
        "next_permission": (
            "confirm_relit_bicar_on_remaining_train_folds"
            if all(checks.values())
            else "close_exact_relighting_conditioned_bicar"
        ),
        "validation_permission": False,
        "test_permission": False,
    }


def _matched_warmup(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact replay before the candidate-only objective becomes active."""

    warmup_epochs = START_EPOCH - 1
    control_rows = list(control["epochs"])[:warmup_epochs]
    candidate_rows = list(candidate["epochs"])[:warmup_epochs]
    checks: dict[str, bool] = {
        "initial_state": (
            control["initial_primary_state_sha256"]
            == candidate["initial_primary_state_sha256"]
        ),
        "epoch_count": len(control_rows) == len(candidate_rows) == warmup_epochs,
    }
    comparable_fields = (
        "epoch",
        "mean_loss",
        "mean_task_loss",
        "optimizer_updates_total",
        "learning_rates",
        "update_over_parameter_norm",
        "model_state_sha256",
        "ema_state_sha256",
    )
    if checks["epoch_count"]:
        for index, (control_row, candidate_row) in enumerate(
            zip(control_rows, candidate_rows),
            start=1,
        ):
            for field_name in comparable_fields:
                checks[f"epoch{index}_{field_name}"] = (
                    control_row[field_name] == candidate_row[field_name]
                )
    passed = bool(all(checks.values()))
    return {"passed": passed, "checks": checks}


def _run_formal(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    if args.preflight_artifact is None:
        raise B38ContractError("Formal B38 requires --preflight-artifact.")
    accepted_preflight = _read_preflight(
        args.preflight_artifact,
        repo=repo,
        git=git,
    )
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    train_fit, train_held, held_paths, fit_labels = b21._build_datasets(
        train_root,
        assignment,
        fold=FOLD,
    )
    class_counts = _fit_counts(assignment)
    control_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=EPOCHS,
        scheduler_horizon=SCHEDULER_HORIZON,
        seed=SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b38_control_ema",
        state_hash_epochs=(START_EPOCH - 1,),
    )
    objective = RelitBiCARObjective(class_counts=class_counts)
    candidate_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=EPOCHS,
        scheduler_horizon=SCHEDULER_HORIZON,
        seed=SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b38_relit_bicar_ema",
        auxiliary_loss_fn=objective,
        auxiliary_loss_weight=AUXILIARY_WEIGHT,
        state_hash_epochs=(START_EPOCH - 1,),
    )
    warmup_replay = _matched_warmup(control_training, candidate_training)
    if not warmup_replay["passed"]:
        raise B38ContractError(
            f"B38 control/candidate warm-up diverged: {warmup_replay['checks']}"
        )
    control_checkpoint = control_training["checkpoint"]
    candidate_checkpoint = candidate_training["checkpoint"]
    if control_checkpoint is None or candidate_checkpoint is None:
        raise B38ContractError("B38 checkpoints were not saved.")

    datasets = {
        name: b36._held_subset(
            train_root,
            assignment,
            brightness=brightness,
            contrast=contrast,
            fold=FOLD,
        )
        for name, (brightness, contrast) in CONDITIONS.items()
    }
    control_metrics, _control_features, control_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(control_checkpoint["path"]),
        expected_state_sha256=control_checkpoint["state_sha256"],
        datasets=datasets,
        seed_offset=300,
    )
    candidate_metrics, _candidate_features, candidate_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(candidate_checkpoint["path"]),
        expected_state_sha256=candidate_checkpoint["state_sha256"],
        datasets=datasets,
        seed_offset=400,
    )
    labels = np.asarray(candidate_training["held_labels"], dtype=np.int64)
    control_replay = float(
        np.max(
            np.abs(
                control_logits["clean"].numpy()
                - np.asarray(control_training["held_logits"], dtype=np.float32)
            )
        )
    )
    candidate_replay = float(
        np.max(
            np.abs(
                candidate_logits["clean"].numpy()
                - np.asarray(candidate_training["held_logits"], dtype=np.float32)
            )
        )
    )
    if control_replay != 0.0 or candidate_replay != 0.0:
        raise B38ContractError(
            f"B38 clean replay changed: control={control_replay}, candidate={candidate_replay}"
        )
    if not np.array_equal(labels, np.asarray(control_training["held_labels"], dtype=np.int64)):
        raise B38ContractError("Control/candidate held label order changed.")
    gate = _gate(control_metrics, candidate_metrics)
    score_path = output / "held_condition_scores.npz"
    arrays: dict[str, np.ndarray] = {"labels": labels}
    for condition in CONDITIONS:
        arrays[f"control_{condition}"] = control_logits[condition].numpy()
        arrays[f"candidate_{condition}"] = candidate_logits[condition].numpy()
    b36._atomic_npz(score_path, **arrays)
    ema_state = objective.state.ema_confusion
    if ema_state is None or objective.active_calls <= 0:
        raise B38ContractError("B38 auxiliary objective never became active.")
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
        },
        "schedule": {
            "seed": SEED,
            "epochs": EPOCHS,
            "scheduler_horizon": SCHEDULER_HORIZON,
            "matched_clean_rng_policy": True,
            "warmup_replay": warmup_replay,
        },
        "objective": _objective_contract(),
        "control_training": b21._sanitize_arm_result(control_training),
        "candidate_training": b21._sanitize_arm_result(candidate_training),
        "auxiliary_state": {
            "updates": objective.state.updates,
            "active_calls": objective.active_calls,
            "ema_confusion": ema_state.detach().float().cpu().tolist(),
            "ema_state_sha256": b21.state_sha256({"ema_confusion": ema_state}),
        },
        "metrics": {"control": control_metrics, "candidate": candidate_metrics},
        "strict_replay_max_abs": {
            "control": control_replay,
            "candidate": candidate_replay,
        },
        "scores_sha256": b21.sha256_file(score_path),
        "gate": gate,
        "interpretation_guard": (
            "Fresh TRAIN component fold 2 only; validation/test remain closed. "
            "A pass requires unchanged confirmation on remaining TRAIN folds."
        ),
    }
    b21._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    args.mature_checkpoint = args.mature_checkpoint.expanduser().resolve(strict=True)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B38ContractError("DINOv3 weight changed.")
    if b21.sha256_file(args.mature_checkpoint) != B35_CHECKPOINT_SHA256:
        raise B38ContractError("Mature B35 preflight checkpoint changed.")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must not already exist: {output}")
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    b21._configure_determinism()
    if args.preflight_only:
        payload = _run_preflight(args, output, repo=repo, git=git)
        print(json.dumps({"passed": payload["passed"], "checks": payload["checks"]}, indent=2))
    else:
        summary = _run_formal(args, output, repo=repo, git=git)
        print(json.dumps(summary["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

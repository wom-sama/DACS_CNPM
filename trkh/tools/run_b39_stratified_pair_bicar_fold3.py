"""Run a fresh TRAIN-fold screen of stratified pair BiCAR plus fixed midpoint.

B39 changes B38 in two places only.  Bright relighting regularizes the 0/1
confusion pair, dim relighting regularizes the 1/2 pair, and the two 2x2 losses
receive equal weight.  The deployable candidate is the fixed 50/50 EMA
weight-space midpoint of the matched clean control and raw robust candidate.
Validation and test are never constructed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import run_b36_head_first_dino_fold as b36
from trkh.tools import run_b38_relit_bicar_fold2 as b38
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.tools.average_checkpoints import average_state_dicts
from trkh.training.confusion_spectral import (
    ConfusionSpectralEMAState,
    ConfusionSpectralResult,
    confusion_aware_spectral_regularizer,
)
from trkh.training.losses import LDAMFocalLoss
from trkh.training.relighting import relight_luminance


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B39_STRATIFIED_PAIR_BICAR_FOLD3_20260809"
FOLD = 3
SEED = 42
AUXILIARY_SEED = 20260809
EPOCHS = 9
SCHEDULER_HORIZON = 30
START_EPOCH = 3
AUXILIARY_WEIGHT = 0.5
PAIR_MIX_WEIGHT = 0.5
EMA_MOMENTUM = 0.5
FREQUENCY_SMOOTHING = 0.2
MARGIN = 0.1
RELIGHT_BRIGHTNESS = 0.25
RELIGHT_CONTRAST = 0.10
MIDPOINT_CONTROL_WEIGHT = 0.5
MIDPOINT_CANDIDATE_WEIGHT = 0.5
BRIGHT_PAIR = (0, 1)
DIM_PAIR = (1, 2)
B38_CONTROL_CHECKPOINT_SHA256 = (
    "5049d3e881041f42bcffde98517052e719579e0a3ca12ae0b81e8c94e606263d"
)
B38_CONTROL_STATE_SHA256 = (
    "4be3c1fffa737c2b5633aa92f6940c0eb7c172510c723ba588318d2baf7b2c95"
)
CONDITIONS = dict(b38.CONDITIONS)


class B39ContractError(RuntimeError):
    pass


def _pair_result(
    logits: Tensor,
    labels: Tensor,
    *,
    pair: tuple[int, int],
    pair_counts: tuple[int, int],
    state: ConfusionSpectralEMAState,
) -> ConfusionSpectralResult:
    if pair[0] == pair[1] or min(pair) < 0 or max(pair) >= logits.size(1):
        raise ValueError("pair must contain two distinct in-range classes")
    mask = (labels == pair[0]) | (labels == pair[1])
    selected_labels = labels[mask]
    if not bool((selected_labels == pair[0]).any().item()) or not bool(
        (selected_labels == pair[1]).any().item()
    ):
        raise B39ContractError(f"A B39 batch is missing one class from pair {pair}.")
    selected_logits = logits[mask][:, list(pair)]
    local_targets = (selected_labels == pair[1]).to(dtype=torch.long)
    return confusion_aware_spectral_regularizer(
        selected_logits,
        local_targets,
        class_counts=pair_counts,
        previous_ema=state.ema_confusion,
        momentum=EMA_MOMENTUM,
        smoothing=FREQUENCY_SMOOTHING,
        margin=MARGIN,
        bidirectional=True,
    )


@dataclass
class StratifiedPairBiCARObjective:
    class_counts: tuple[int, ...]
    bright_state: ConfusionSpectralEMAState = field(
        default_factory=ConfusionSpectralEMAState
    )
    dim_state: ConfusionSpectralEMAState = field(
        default_factory=ConfusionSpectralEMAState
    )
    active_calls: int = 0

    def _active_components(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        *,
        epoch_index: int,
        batch_index: int,
    ) -> tuple[Tensor, Tensor, Mapping[str, float]]:
        if images.device.type != "cuda":
            raise B39ContractError("B39 stratified objective requires CUDA.")
        device_index = images.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        bright_mask = (labels == BRIGHT_PAIR[0]) | (labels == BRIGHT_PAIR[1])
        dim_mask = (labels == DIM_PAIR[0]) | (labels == DIM_PAIR[1])
        if not bool(bright_mask.any().item()) or not bool(dim_mask.any().item()):
            raise B39ContractError("B39 pair subsets are empty.")
        auxiliary_seed = (
            AUXILIARY_SEED + int(epoch_index) * 100_000 + int(batch_index)
        )
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(auxiliary_seed)
            bright_images = relight_luminance(
                images[bright_mask],
                torch.ones(
                    int(bright_mask.sum().item()),
                    device=images.device,
                    dtype=images.dtype,
                ),
                brightness=RELIGHT_BRIGHTNESS,
                contrast=RELIGHT_CONTRAST,
            )
            dim_images = relight_luminance(
                images[dim_mask],
                -torch.ones(
                    int(dim_mask.sum().item()),
                    device=images.device,
                    dtype=images.dtype,
                ),
                brightness=RELIGHT_BRIGHTNESS,
                contrast=RELIGHT_CONTRAST,
            )
            bright_logits = model(bright_images)
            dim_logits = model(dim_images)
        bright_result = _pair_result(
            bright_logits,
            labels[bright_mask],
            pair=BRIGHT_PAIR,
            pair_counts=(self.class_counts[0], self.class_counts[1]),
            state=self.bright_state,
        )
        dim_result = _pair_result(
            dim_logits,
            labels[dim_mask],
            pair=DIM_PAIR,
            pair_counts=(self.class_counts[1], self.class_counts[2]),
            state=self.dim_state,
        )
        self.bright_state.update(bright_result.ema_confusion)
        self.dim_state.update(dim_result.ema_confusion)
        self.active_calls += 1
        stats = {
            "active": 1.0,
            "bright_rows": float(bright_mask.sum().item()),
            "dim_rows": float(dim_mask.sum().item()),
            "bright_pair_loss": float(bright_result.loss.detach().float().cpu()),
            "dim_pair_loss": float(dim_result.loss.detach().float().cpu()),
        }
        return bright_result.loss, dim_result.loss, stats

    def __call__(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        clean_logits: Tensor,
        epoch_index: int,
        batch_index: int,
    ) -> tuple[Tensor, Mapping[str, float]]:
        if int(epoch_index) + 1 < START_EPOCH:
            return clean_logits.sum() * 0.0, {"active": 0.0}
        bright_loss, dim_loss, stats = self._active_components(
            model,
            images,
            labels,
            epoch_index=epoch_index,
            batch_index=batch_index,
        )
        loss = PAIR_MIX_WEIGHT * bright_loss + PAIR_MIX_WEIGHT * dim_loss
        return loss, dict(stats)


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
        "runner": Path("trkh/tools/run_b39_stratified_pair_bicar_fold3.py"),
        "b38_runner": Path("trkh/tools/run_b38_relit_bicar_fold2.py"),
        "trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "evaluator": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "confusion": Path("trkh/training/confusion_spectral.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "averaging": Path("trkh/tools/average_checkpoints.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "runner_test": Path("tests/test_run_b39_stratified_pair_bicar_fold3.py"),
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
        raise B39ContractError(f"B39 preflight is not accepted: {checks}")
    return {"path": str(resolved), "sha256": b21.sha256_file(resolved)}


def _fit_counts(
    assignment: Mapping[str, Any],
    *,
    fold: int = FOLD,
) -> tuple[int, ...]:
    fit = np.asarray(assignment["folds"]) != int(fold)
    return tuple(
        int(value)
        for value in np.bincount(np.asarray(assignment["labels"])[fit], minlength=5)
    )


def _pair_oracle_directions(class_counts: Sequence[int]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    gradients: dict[str, list[list[float]]] = {}
    for name, pair in (("bright_0_1", BRIGHT_PAIR), ("dim_1_2", DIM_PAIR)):
        logits = torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor([0, 1])
        pair_counts = (int(class_counts[pair[0]]), int(class_counts[pair[1]]))
        result = confusion_aware_spectral_regularizer(
            logits,
            targets,
            class_counts=pair_counts,
            momentum=EMA_MOMENTUM,
            smoothing=FREQUENCY_SMOOTHING,
            margin=MARGIN,
            bidirectional=True,
        )
        gradient = torch.autograd.grad(result.loss, logits)[0]
        checks[f"{name}_first_direction"] = bool(
            gradient[0, 0] < 0 and gradient[0, 1] > 0
        )
        checks[f"{name}_second_direction"] = bool(
            gradient[1, 0] > 0 and gradient[1, 1] < 0
        )
        gradients[name] = gradient.tolist()
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "gradients": gradients,
    }


def _role_checks(
    alignment: Mapping[str, Mapping[str, float]],
    *,
    ratio_min: float,
    ratio_max: float,
    cosine_min: float,
) -> dict[str, bool]:
    return {
        role: bool(
            alignment[role]["task_norm"] > 0.0
            and alignment[role]["auxiliary_norm"] > 0.0
            and ratio_min <= alignment[role]["auxiliary_over_task"] <= ratio_max
            and alignment[role]["cosine"] >= cosine_min
        )
        for role in ("backbone", "head")
    }


def _objective_contract() -> dict[str, Any]:
    return {
        "name": "polarity_stratified_pair_bicar_fixed_midpoint",
        "start_epoch": START_EPOCH,
        "outer_weight": AUXILIARY_WEIGHT,
        "bright_pair": BRIGHT_PAIR,
        "dim_pair": DIM_PAIR,
        "pair_mix_weights": [PAIR_MIX_WEIGHT, PAIR_MIX_WEIGHT],
        "ema_momentum": EMA_MOMENTUM,
        "frequency_smoothing": FREQUENCY_SMOOTHING,
        "margin": MARGIN,
        "bidirectional": True,
        "brightness": RELIGHT_BRIGHTNESS,
        "contrast": RELIGHT_CONTRAST,
        "chroma_policy": "preserve_rgb_differences_until_clipping",
        "midpoint_weights": [MIDPOINT_CONTROL_WEIGHT, MIDPOINT_CANDIDATE_WEIGHT],
        "inference_parameter_delta": 0,
        "rng_policy": "seeded_pair_forward_fork_restores_cpu_and_active_cuda_rng",
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
    images, labels = b38._first_fit_batch(fit_dataset, fit_labels)
    label_counts = torch.bincount(labels, minlength=5)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise B39ContractError("B39 requires CUDA BF16.")
    device = torch.device("cuda")
    model = b21.build_arm_model(args.dino_weight, DIRECT_MODE)
    model.set_mode(DIRECT_MODE)
    state = load_file(str(args.mature_checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    state_match = b21.state_sha256(model.state_dict()) == B38_CONTROL_STATE_SHA256
    identity_state, identity_stats = average_state_dicts(
        [state, state],
        [MIDPOINT_CONTROL_WEIGHT, MIDPOINT_CANDIDATE_WEIGHT],
    )
    midpoint_identity = b21.state_sha256(identity_state) == b21.state_sha256(state)
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
    objective = StratifiedPairBiCARObjective(class_counts=class_counts)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clean_logits = model(images)
        task_loss = criterion(clean_logits, labels)
        focus_mask = labels == 1
        focus_task_loss = criterion(clean_logits[focus_mask], labels[focus_mask])
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone()
        bright_loss, dim_loss, auxiliary_stats = objective._active_components(
            model,
            images,
            labels,
            epoch_index=START_EPOCH - 1,
            batch_index=0,
        )
        cpu_rng_preserved = torch.equal(cpu_rng_before, torch.random.get_rng_state())
        cuda_rng_preserved = torch.equal(
            cuda_rng_before,
            torch.cuda.get_rng_state(device),
        )
        auxiliary_loss = PAIR_MIX_WEIGHT * bright_loss + PAIR_MIX_WEIGHT * dim_loss
        weighted_auxiliary = AUXILIARY_WEIGHT * auxiliary_loss
        weighted_bright = AUXILIARY_WEIGHT * PAIR_MIX_WEIGHT * bright_loss
        weighted_dim = AUXILIARY_WEIGHT * PAIR_MIX_WEIGHT * dim_loss
    task_alignment = b21.gradient_alignment_by_role(
        model,
        task_loss,
        weighted_auxiliary,
    )
    focus_alignment = b21.gradient_alignment_by_role(
        model,
        focus_task_loss,
        weighted_auxiliary,
    )
    pair_alignment = b21.gradient_alignment_by_role(
        model,
        weighted_bright,
        weighted_dim,
    )
    combined = task_loss + weighted_auxiliary
    combined.backward()
    oracle = _pair_oracle_directions(class_counts)
    task_checks = _role_checks(
        task_alignment,
        ratio_min=0.01,
        ratio_max=0.15,
        cosine_min=-0.25,
    )
    focus_checks = _role_checks(
        focus_alignment,
        ratio_min=0.005,
        ratio_max=0.25,
        cosine_min=0.0,
    )
    pair_checks = _role_checks(
        pair_alignment,
        ratio_min=0.25,
        ratio_max=4.0,
        cosine_min=-0.5,
    )
    checks = {
        "mature_state_exact": state_match,
        "all_classes_in_batch": bool((label_counts > 0).all().item()),
        "cpu_rng_preserved": cpu_rng_preserved,
        "cuda_rng_preserved": cuda_rng_preserved,
        "task_loss_finite": bool(torch.isfinite(task_loss).item()),
        "pair_losses_finite_positive": bool(
            torch.isfinite(bright_loss).item()
            and torch.isfinite(dim_loss).item()
            and bright_loss.item() > 0.0
            and dim_loss.item() > 0.0
        ),
        "pair_ema_updated_once": (
            objective.bright_state.updates == 1 and objective.dim_state.updates == 1
        ),
        "pair_oracle_directions": bool(oracle["passed"]),
        "backbone_task_scale": task_checks["backbone"],
        "head_task_scale": task_checks["head"],
        "backbone_focus_compatibility": focus_checks["backbone"],
        "head_focus_compatibility": focus_checks["head"],
        "backbone_pair_balance": pair_checks["backbone"],
        "head_pair_balance": pair_checks["head"],
        "midpoint_identity": midpoint_identity,
        "midpoint_nonfloat_states_match": (
            identity_stats["copied_different_non_float"] == 0
        ),
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
            "expected_sha256": B38_CONTROL_CHECKPOINT_SHA256,
        },
        "objective": _objective_contract(),
        "losses": {
            "task": float(task_loss.detach().float().cpu()),
            "focus_task": float(focus_task_loss.detach().float().cpu()),
            "bright_pair": float(bright_loss.detach().float().cpu()),
            "dim_pair": float(dim_loss.detach().float().cpu()),
            "weighted_auxiliary": float(weighted_auxiliary.detach().float().cpu()),
        },
        "auxiliary_stats": dict(auxiliary_stats),
        "task_alignment": task_alignment,
        "focus_alignment": focus_alignment,
        "pair_alignment": pair_alignment,
        "oracle": oracle,
        "midpoint_identity_stats": identity_stats,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    b21._atomic_json(output / "preflight.json", payload)
    del model, state, identity_state, images, labels, clean_logits, combined
    gc.collect()
    torch.cuda.empty_cache()
    if not payload["passed"]:
        raise B39ContractError(f"B39 preflight failed: {checks}")
    return payload


def _save_midpoint(
    control_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    output: Path,
    *,
    protocol_id: str = PROTOCOL_ID,
    filename: str = "b39_midpoint_ema.safetensors",
) -> dict[str, Any]:
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("midpoint filename must be a local .safetensors name")
    control_state = load_file(str(control_checkpoint["path"]), device="cpu")
    candidate_state = load_file(str(candidate_checkpoint["path"]), device="cpu")
    midpoint_state, stats = average_state_dicts(
        [control_state, candidate_state],
        [MIDPOINT_CONTROL_WEIGHT, MIDPOINT_CANDIDATE_WEIGHT],
    )
    if stats["copied_different_non_float"] != 0:
        raise B39ContractError(
            "Pair-midpoint endpoint states have different non-float buffers."
        )
    path = output / filename
    save_file(
        midpoint_state,
        str(path),
        metadata={
            "protocol_id": protocol_id,
            "kind": "fixed_control_candidate_midpoint",
            "control_weight": str(MIDPOINT_CONTROL_WEIGHT),
            "candidate_weight": str(MIDPOINT_CANDIDATE_WEIGHT),
        },
    )
    result = {
        "path": str(path.resolve()),
        "sha256": b21.sha256_file(path),
        "state_sha256": b21.state_sha256(midpoint_state),
        "weights": [MIDPOINT_CONTROL_WEIGHT, MIDPOINT_CANDIDATE_WEIGHT],
        "stats": stats,
    }
    del control_state, candidate_state, midpoint_state
    return result


def _gate(
    control: Mapping[str, Mapping[str, Any]],
    midpoint: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate = b38._gate(control, midpoint)
    gate["next_permission"] = (
        "confirm_stratified_pair_midpoint_on_remaining_train_folds"
        if gate["passed"]
        else "close_exact_stratified_pair_midpoint"
    )
    return gate


def run_pair_midpoint_formal(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
    protocol_id: str,
    fold: int,
    seed: int,
    epochs: int,
    scheduler_horizon: int,
    start_epoch: int,
    auxiliary_weight: float,
    objective_factory: Callable[[tuple[int, ...]], Any],
    objective_contract_fn: Callable[[], dict[str, Any]],
    source_hashes_fn: Callable[[Path], dict[str, str]],
    preflight_reader: Callable[..., dict[str, str]],
    gate_fn: Callable[
        [Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]],
        dict[str, Any],
    ],
    control_checkpoint_stem: str,
    candidate_checkpoint_stem: str,
    midpoint_filename: str,
    conditions: Mapping[str, tuple[float, float]],
    interpretation_guard: str,
) -> dict[str, Any]:
    if args.preflight_artifact is None:
        raise B39ContractError(
            "Formal pair-midpoint screen requires --preflight-artifact."
        )
    accepted_preflight = preflight_reader(
        args.preflight_artifact,
        repo=repo,
        git=git,
    )
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    train_fit, train_held, held_paths, fit_labels = b21._build_datasets(
        train_root,
        assignment,
        fold=fold,
    )
    class_counts = _fit_counts(assignment, fold=fold)
    control_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=epochs,
        scheduler_horizon=scheduler_horizon,
        seed=seed,
        protocol_id=protocol_id,
        checkpoint_stem=control_checkpoint_stem,
        state_hash_epochs=(start_epoch - 1,),
    )
    objective = objective_factory(class_counts)
    candidate_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=epochs,
        scheduler_horizon=scheduler_horizon,
        seed=seed,
        protocol_id=protocol_id,
        checkpoint_stem=candidate_checkpoint_stem,
        auxiliary_loss_fn=objective,
        auxiliary_loss_weight=auxiliary_weight,
        state_hash_epochs=(start_epoch - 1,),
    )
    warmup_replay = b38._matched_warmup(control_training, candidate_training)
    if not warmup_replay["passed"]:
        raise B39ContractError(
            "Pair-midpoint control/candidate warm-up diverged: "
            f"{warmup_replay['checks']}"
        )
    control_checkpoint = control_training["checkpoint"]
    candidate_checkpoint = candidate_training["checkpoint"]
    if not isinstance(control_checkpoint, Mapping) or not isinstance(
        candidate_checkpoint,
        Mapping,
    ):
        raise B39ContractError("Pair-midpoint checkpoints were not saved.")
    midpoint_checkpoint = _save_midpoint(
        control_checkpoint,
        candidate_checkpoint,
        output,
        protocol_id=protocol_id,
        filename=midpoint_filename,
    )
    datasets = {
        name: b36._held_subset(
            train_root,
            assignment,
            brightness=brightness,
            contrast=contrast,
            fold=fold,
        )
        for name, (brightness, contrast) in conditions.items()
    }
    control_metrics, _control_features, control_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(control_checkpoint["path"])),
        expected_state_sha256=str(control_checkpoint["state_sha256"]),
        datasets=datasets,
        seed_offset=600,
    )
    raw_metrics, _raw_features, raw_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(candidate_checkpoint["path"])),
        expected_state_sha256=str(candidate_checkpoint["state_sha256"]),
        datasets=datasets,
        seed_offset=700,
    )
    midpoint_metrics, _midpoint_features, midpoint_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(midpoint_checkpoint["path"])),
        expected_state_sha256=str(midpoint_checkpoint["state_sha256"]),
        datasets=datasets,
        seed_offset=800,
    )
    labels = np.asarray(candidate_training["held_labels"], dtype=np.int64)
    if not np.array_equal(
        labels,
        np.asarray(control_training["held_labels"], dtype=np.int64),
    ):
        raise B39ContractError("Pair-midpoint held label order changed.")
    control_replay = float(
        np.max(
            np.abs(
                control_logits["clean"].numpy()
                - np.asarray(control_training["held_logits"], dtype=np.float32)
            )
        )
    )
    raw_replay = float(
        np.max(
            np.abs(
                raw_logits["clean"].numpy()
                - np.asarray(candidate_training["held_logits"], dtype=np.float32)
            )
        )
    )
    if control_replay != 0.0 or raw_replay != 0.0:
        raise B39ContractError(
            "Pair-midpoint clean replay changed: "
            f"control={control_replay}, raw={raw_replay}"
        )
    gate = gate_fn(control_metrics, midpoint_metrics)
    score_path = output / "held_condition_scores.npz"
    arrays: dict[str, np.ndarray] = {"labels": labels}
    for condition in conditions:
        arrays[f"control_{condition}"] = control_logits[condition].numpy()
        arrays[f"raw_candidate_{condition}"] = raw_logits[condition].numpy()
        arrays[f"midpoint_{condition}"] = midpoint_logits[condition].numpy()
    b36._atomic_npz(score_path, **arrays)
    bright_ema = objective.bright_state.ema_confusion
    dim_ema = objective.dim_state.ema_confusion
    if bright_ema is None or dim_ema is None or objective.active_calls <= 0:
        raise B39ContractError("Pair-midpoint objective never became active.")
    summary = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "git": dict(git),
        "source_hashes": source_hashes_fn(repo),
        "accepted_preflight": accepted_preflight,
        "dataset": {
            "fold": fold,
            "fit_rows": len(train_fit),
            "held_rows": len(train_held),
            "fit_class_counts": class_counts,
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "schedule": {
            "seed": seed,
            "epochs": epochs,
            "scheduler_horizon": scheduler_horizon,
            "warmup_replay": warmup_replay,
        },
        "objective": objective_contract_fn(),
        "control_training": b21._sanitize_arm_result(control_training),
        "raw_candidate_training": b21._sanitize_arm_result(candidate_training),
        "midpoint_checkpoint": midpoint_checkpoint,
        "auxiliary_state": {
            "active_calls": objective.active_calls,
            "bright_updates": objective.bright_state.updates,
            "dim_updates": objective.dim_state.updates,
            "bright_ema_confusion": bright_ema.detach().float().cpu().tolist(),
            "dim_ema_confusion": dim_ema.detach().float().cpu().tolist(),
        },
        "metrics": {
            "control": control_metrics,
            "raw_candidate": raw_metrics,
            "midpoint": midpoint_metrics,
        },
        "strict_replay_max_abs": {
            "control": control_replay,
            "raw_candidate": raw_replay,
        },
        "scores_sha256": b21.sha256_file(score_path),
        "gate": gate,
        "interpretation_guard": interpretation_guard,
    }
    b21._atomic_json(output / "summary.json", summary)
    return summary


def _run_formal(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    return run_pair_midpoint_formal(
        args,
        output,
        repo=repo,
        git=git,
        protocol_id=PROTOCOL_ID,
        fold=FOLD,
        seed=SEED,
        epochs=EPOCHS,
        scheduler_horizon=SCHEDULER_HORIZON,
        start_epoch=START_EPOCH,
        auxiliary_weight=AUXILIARY_WEIGHT,
        objective_factory=lambda counts: StratifiedPairBiCARObjective(
            class_counts=counts
        ),
        objective_contract_fn=_objective_contract,
        source_hashes_fn=_source_hashes,
        preflight_reader=_read_preflight,
        gate_fn=_gate,
        control_checkpoint_stem="b39_control_ema",
        candidate_checkpoint_stem="b39_stratified_pair_ema",
        midpoint_filename="b39_midpoint_ema.safetensors",
        conditions=CONDITIONS,
        interpretation_guard=(
            "Fresh TRAIN component fold 3 only. The fixed midpoint is the gated "
            "endpoint; raw candidate is mechanism evidence. Validation/test stay closed."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    args.mature_checkpoint = args.mature_checkpoint.expanduser().resolve(strict=True)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B39ContractError("DINOv3 weight changed.")
    if b21.sha256_file(args.mature_checkpoint) != B38_CONTROL_CHECKPOINT_SHA256:
        raise B39ContractError("Mature B38 control checkpoint changed.")
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

"""Run the locked B40 pair-mean robustness screen on TRAIN component fold 3.

B40 repairs the exact B39 preflight defect: a two-by-two spectral norm is a
maximum over the two off-diagonal directions.  Separate bright 0/1 and dim 1/2
EMAs instead use the arithmetic mean of their two class-conditional confusions.
The gated endpoint remains the fixed 50/50 EMA weight midpoint.  Validation and
test are never constructed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import run_b38_relit_bicar_fold2 as b38
from trkh.tools import run_b39_stratified_pair_bicar_fold3 as b39
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.tools.average_checkpoints import average_state_dicts
from trkh.training.confusion_spectral import ConfusionSpectralEMAState
from trkh.training.losses import LDAMFocalLoss
from trkh.training.pair_confusion import (
    BidirectionalPairConfusionResult,
    bidirectional_pair_confusion_mean,
)
from trkh.training.relighting import relight_luminance


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B40_PAIR_MEAN_MIDPOINT_FOLD3_20260809"
FOLD = 3
SEED = 42
AUXILIARY_SEED = b39.AUXILIARY_SEED
EPOCHS = 9
SCHEDULER_HORIZON = 30
START_EPOCH = 3
AUXILIARY_WEIGHT = 0.15
PAIR_MIX_WEIGHT = 0.5
EMA_MOMENTUM = 0.5
MARGIN = 0.1
RELIGHT_BRIGHTNESS = 0.25
RELIGHT_CONTRAST = 0.10
MIDPOINT_CONTROL_WEIGHT = 0.5
MIDPOINT_CANDIDATE_WEIGHT = 0.5
BRIGHT_PAIR = (0, 1)
DIM_PAIR = (1, 2)
B38_CONTROL_CHECKPOINT_SHA256 = b39.B38_CONTROL_CHECKPOINT_SHA256
B38_CONTROL_STATE_SHA256 = b39.B38_CONTROL_STATE_SHA256
CONDITIONS = dict(b38.CONDITIONS)


class B40ContractError(RuntimeError):
    pass


def _pair_result(
    logits: Tensor,
    labels: Tensor,
    *,
    pair: tuple[int, int],
    state: ConfusionSpectralEMAState,
) -> BidirectionalPairConfusionResult:
    if pair[0] == pair[1] or min(pair) < 0 or max(pair) >= logits.size(1):
        raise ValueError("pair must contain two distinct in-range classes")
    mask = (labels == pair[0]) | (labels == pair[1])
    selected_labels = labels[mask]
    if not bool((selected_labels == pair[0]).any().item()) or not bool(
        (selected_labels == pair[1]).any().item()
    ):
        raise B40ContractError(f"A B40 batch is missing one class from pair {pair}.")
    selected_logits = logits[mask][:, list(pair)]
    local_targets = (selected_labels == pair[1]).to(dtype=torch.long)
    return bidirectional_pair_confusion_mean(
        selected_logits,
        local_targets,
        previous_ema=state.ema_confusion,
        momentum=EMA_MOMENTUM,
        margin=MARGIN,
    )


@dataclass
class StratifiedPairMeanObjective:
    class_counts: tuple[int, ...]
    bright_state: ConfusionSpectralEMAState = field(
        default_factory=ConfusionSpectralEMAState
    )
    dim_state: ConfusionSpectralEMAState = field(
        default_factory=ConfusionSpectralEMAState
    )
    active_calls: int = 0

    def __post_init__(self) -> None:
        if len(self.class_counts) != 5 or min(self.class_counts) <= 0:
            raise ValueError("B40 requires five positive natural TRAIN class counts.")

    def _active_components(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        *,
        epoch_index: int,
        batch_index: int,
    ) -> tuple[
        BidirectionalPairConfusionResult,
        BidirectionalPairConfusionResult,
        Mapping[str, float],
    ]:
        if images.device.type != "cuda":
            raise B40ContractError("B40 stratified objective requires CUDA.")
        device_index = images.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        bright_mask = (labels == BRIGHT_PAIR[0]) | (labels == BRIGHT_PAIR[1])
        dim_mask = (labels == DIM_PAIR[0]) | (labels == DIM_PAIR[1])
        if not bool(bright_mask.any().item()) or not bool(dim_mask.any().item()):
            raise B40ContractError("B40 pair subsets are empty.")
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
            state=self.bright_state,
        )
        dim_result = _pair_result(
            dim_logits,
            labels[dim_mask],
            pair=DIM_PAIR,
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
        return bright_result, dim_result, stats

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
        bright_result, dim_result, stats = self._active_components(
            model,
            images,
            labels,
            epoch_index=epoch_index,
            batch_index=batch_index,
        )
        loss = (
            PAIR_MIX_WEIGHT * bright_result.loss
            + PAIR_MIX_WEIGHT * dim_result.loss
        )
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
        "runner": Path("trkh/tools/run_b40_pair_mean_midpoint_fold3.py"),
        "formal_common": Path("trkh/tools/run_b39_stratified_pair_bicar_fold3.py"),
        "trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "evaluator": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "pair_confusion": Path("trkh/training/pair_confusion.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "averaging": Path("trkh/tools/average_checkpoints.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "runner_test": Path("tests/test_run_b40_pair_mean_midpoint_fold3.py"),
        "primitive_test": Path("tests/test_pair_confusion.py"),
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
        raise B40ContractError(f"B40 preflight is not accepted: {checks}")
    return {"path": str(resolved), "sha256": b21.sha256_file(resolved)}


def _pair_oracle_directions() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    gradients: dict[str, list[list[float]]] = {}
    row_ratios: dict[str, float] = {}
    for name in ("bright_0_1", "dim_1_2"):
        logits = torch.tensor(
            [[0.0, 3.0], [1.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        result = bidirectional_pair_confusion_mean(
            logits,
            torch.tensor([0, 1]),
            momentum=EMA_MOMENTUM,
            margin=MARGIN,
        )
        gradient = torch.autograd.grad(result.loss, logits)[0]
        row_norms = gradient.norm(dim=1)
        row_ratio = float(row_norms.min() / row_norms.max().clamp_min(1e-12))
        checks[f"{name}_first_direction"] = bool(
            gradient[0, 0] < 0 and gradient[0, 1] > 0
        )
        checks[f"{name}_second_direction"] = bool(
            gradient[1, 0] > 0 and gradient[1, 1] < 0
        )
        checks[f"{name}_direction_floor"] = row_ratio >= 0.10
        gradients[name] = gradient.tolist()
        row_ratios[name] = row_ratio
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "gradients": gradients,
        "row_gradient_min_over_max": row_ratios,
    }


def _objective_contract() -> dict[str, Any]:
    return {
        "name": "polarity_stratified_bidirectional_pair_confusion_mean_midpoint",
        "start_epoch": START_EPOCH,
        "outer_weight": AUXILIARY_WEIGHT,
        "outer_weight_derivation": "0.5_b39_weight_times_0.30_measured_scale",
        "bright_pair": BRIGHT_PAIR,
        "dim_pair": DIM_PAIR,
        "pair_mix_weights": [PAIR_MIX_WEIGHT, PAIR_MIX_WEIGHT],
        "pair_functional": "mean_of_two_class_conditional_offdiagonal_confusions",
        "frequency_weighting": False,
        "frequency_weighting_reason": "each_true_class_column_is_already_a_mean",
        "ema_momentum": EMA_MOMENTUM,
        "margin": MARGIN,
        "brightness": RELIGHT_BRIGHTNESS,
        "contrast": RELIGHT_CONTRAST,
        "chroma_policy": "preserve_rgb_differences_until_clipping",
        "midpoint_weights": [MIDPOINT_CONTROL_WEIGHT, MIDPOINT_CANDIDATE_WEIGHT],
        "inference_parameter_delta": 0,
        "rng_policy": "seeded_pair_forward_fork_restores_cpu_and_active_cuda_rng",
    }


def _gate(
    control: Mapping[str, Mapping[str, Any]],
    midpoint: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate = b38._gate(control, midpoint)
    gate["next_permission"] = (
        "confirm_pair_mean_midpoint_on_remaining_train_folds"
        if gate["passed"]
        else "close_exact_pair_mean_midpoint"
    )
    return gate


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
    class_counts = b39._fit_counts(assignment, fold=FOLD)
    images, labels = b38._first_fit_batch(fit_dataset, fit_labels)
    label_counts = torch.bincount(labels, minlength=5)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise B40ContractError("B40 requires CUDA BF16.")
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
    objective = StratifiedPairMeanObjective(class_counts=class_counts)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clean_logits = model(images)
        task_loss = criterion(clean_logits, labels)
        focus_mask = labels == 1
        focus_task_loss = criterion(clean_logits[focus_mask], labels[focus_mask])
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone()
        bright_result, dim_result, auxiliary_stats = objective._active_components(
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
        auxiliary_loss = (
            PAIR_MIX_WEIGHT * bright_result.loss
            + PAIR_MIX_WEIGHT * dim_result.loss
        )
        weighted_auxiliary = AUXILIARY_WEIGHT * auxiliary_loss
        weighted_bright = AUXILIARY_WEIGHT * PAIR_MIX_WEIGHT * bright_result.loss
        weighted_dim = AUXILIARY_WEIGHT * PAIR_MIX_WEIGHT * dim_result.loss
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
    (task_loss + weighted_auxiliary).backward()
    oracle = _pair_oracle_directions()
    task_checks = b39._role_checks(
        task_alignment,
        ratio_min=0.04,
        ratio_max=0.16,
        cosine_min=0.0,
    )
    focus_checks = b39._role_checks(
        focus_alignment,
        ratio_min=0.02,
        ratio_max=0.20,
        cosine_min=0.20,
    )
    pair_checks = b39._role_checks(
        pair_alignment,
        ratio_min=0.50,
        ratio_max=2.0,
        cosine_min=-0.50,
    )
    directional_values = torch.cat(
        (
            bright_result.directional_confusions.detach().float(),
            dim_result.directional_confusions.detach().float(),
        )
    )
    checks = {
        "mature_state_exact": state_match,
        "all_classes_in_batch": bool((label_counts > 0).all().item()),
        "cpu_rng_preserved": cpu_rng_preserved,
        "cuda_rng_preserved": cuda_rng_preserved,
        "task_loss_finite": bool(torch.isfinite(task_loss).item()),
        "pair_losses_finite_positive": bool(
            torch.isfinite(bright_result.loss).item()
            and torch.isfinite(dim_result.loss).item()
            and bright_result.loss.item() > 0.0
            and dim_result.loss.item() > 0.0
        ),
        "all_directional_confusions_positive": bool(
            torch.isfinite(directional_values).all().item()
            and (directional_values > 0).all().item()
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
            "bright_pair": float(bright_result.loss.detach().float().cpu()),
            "dim_pair": float(dim_result.loss.detach().float().cpu()),
            "weighted_auxiliary": float(weighted_auxiliary.detach().float().cpu()),
        },
        "auxiliary_stats": dict(auxiliary_stats),
        "task_alignment": task_alignment,
        "focus_alignment": focus_alignment,
        "pair_alignment": pair_alignment,
        "directional_confusions": {
            "bright_pred0_true1": float(directional_values[0].cpu()),
            "bright_pred1_true0": float(directional_values[1].cpu()),
            "dim_pred1_true2": float(directional_values[2].cpu()),
            "dim_pred2_true1": float(directional_values[3].cpu()),
        },
        "oracle": oracle,
        "midpoint_identity_stats": identity_stats,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    b21._atomic_json(output / "preflight.json", payload)
    del model, state, identity_state, images, labels, clean_logits
    gc.collect()
    torch.cuda.empty_cache()
    if not payload["passed"]:
        raise B40ContractError(f"B40 preflight failed: {checks}")
    return payload


def _run_formal(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    return b39.run_pair_midpoint_formal(
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
        objective_factory=lambda counts: StratifiedPairMeanObjective(
            class_counts=counts
        ),
        objective_contract_fn=_objective_contract,
        source_hashes_fn=_source_hashes,
        preflight_reader=_read_preflight,
        gate_fn=_gate,
        control_checkpoint_stem="b40_control_ema",
        candidate_checkpoint_stem="b40_pair_mean_ema",
        midpoint_filename="b40_midpoint_ema.safetensors",
        conditions=CONDITIONS,
        interpretation_guard=(
            "Fresh held TRAIN component fold 3 only. The fixed midpoint is the gated "
            "endpoint; raw candidate is mechanism evidence. Validation/test stay closed."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    args.mature_checkpoint = args.mature_checkpoint.expanduser().resolve(strict=True)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B40ContractError("DINOv3 weight changed.")
    if b21.sha256_file(args.mature_checkpoint) != B38_CONTROL_CHECKPOINT_SHA256:
        raise B40ContractError("Mature B38 control checkpoint changed.")
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

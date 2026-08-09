"""Run locked B41 backbone-routed pair-mean midpoint screen on TRAIN fold 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trkh.tools import run_b38_relit_bicar_fold2 as b38
from trkh.tools import run_b39_stratified_pair_bicar_fold3 as b39
from trkh.tools import run_b40_pair_mean_midpoint_fold3 as b40
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B41_BACKBONE_ROUTED_PAIR_MEAN_FOLD3_20260809"
FOLD = b40.FOLD
SEED = b40.SEED
EPOCHS = b40.EPOCHS
SCHEDULER_HORIZON = b40.SCHEDULER_HORIZON
START_EPOCH = b40.START_EPOCH
AUXILIARY_WEIGHT = 0.20
CONDITIONS = dict(b40.CONDITIONS)
B38_CONTROL_CHECKPOINT_SHA256 = b40.B38_CONTROL_CHECKPOINT_SHA256


class B41ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--mature-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _objective_factory(class_counts: tuple[int, ...]) -> b40.StratifiedPairMeanObjective:
    return b40.StratifiedPairMeanObjective(
        class_counts=class_counts,
        isolate_auxiliary_head=True,
    )


def _objective_contract() -> dict[str, Any]:
    contract = dict(b40._objective_contract())
    contract.update(
        {
            "name": "backbone_routed_polarity_stratified_pair_confusion_mean_midpoint",
            "outer_weight": AUXILIARY_WEIGHT,
            "outer_weight_derivation": (
                "round_0.15_times_target_0.05_over_b40_backbone_ratio_0.0388435"
            ),
            "auxiliary_head_parameter_gradient": "exact_zero",
            "auxiliary_backbone_gradient": "through_fixed_current_head_weights",
            "clean_task_head_gradient": "unchanged",
            "gradient_projection_or_surgery": False,
        }
    )
    return contract


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b41_backbone_routed_pair_mean_fold3.py"),
        "pair_mean_common": Path("trkh/tools/run_b40_pair_mean_midpoint_fold3.py"),
        "formal_common": Path("trkh/tools/run_b39_stratified_pair_bicar_fold3.py"),
        "trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "evaluator": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "pair_confusion": Path("trkh/training/pair_confusion.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "averaging": Path("trkh/tools/average_checkpoints.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "runner_test": Path("tests/test_run_b41_backbone_routed_pair_mean_fold3.py"),
        "telemetry_test": Path("tests/test_run_b38_relit_bicar_fold2.py"),
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
        raise B41ContractError(f"B41 preflight is not accepted: {checks}")
    return {"path": str(resolved), "sha256": b21.sha256_file(resolved)}


def _gate(
    control: Mapping[str, Mapping[str, Any]],
    midpoint: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate = b38._gate(control, midpoint)
    gate["next_permission"] = (
        "confirm_backbone_routed_pair_mean_on_remaining_train_folds"
        if gate["passed"]
        else "close_exact_backbone_routed_pair_mean"
    )
    return gate


def _run_preflight(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    return b40.run_pair_mean_preflight(
        args,
        output,
        repo=repo,
        git=git,
        protocol_id=PROTOCOL_ID,
        auxiliary_weight=AUXILIARY_WEIGHT,
        objective_factory=_objective_factory,
        objective_contract_fn=_objective_contract,
        source_hashes_fn=_source_hashes,
        contract_error=B41ContractError,
        head_isolated_expected=True,
        task_ratio_min=0.04,
        task_ratio_max=0.10,
        task_cosine_min=0.15,
        focus_ratio_min=0.02,
        focus_ratio_max=0.10,
        focus_cosine_min=0.30,
        pair_ratio_min=0.50,
        pair_ratio_max=2.0,
        pair_cosine_min=0.0,
    )


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
        objective_factory=_objective_factory,
        objective_contract_fn=_objective_contract,
        source_hashes_fn=_source_hashes,
        preflight_reader=_read_preflight,
        gate_fn=_gate,
        control_checkpoint_stem="b41_control_ema",
        candidate_checkpoint_stem="b41_backbone_routed_pair_mean_ema",
        midpoint_filename="b41_midpoint_ema.safetensors",
        conditions=CONDITIONS,
        interpretation_guard=(
            "Fresh held TRAIN component fold 3 only. Robust auxiliary gradients "
            "reach the backbone through fixed current head weights but cannot update "
            "head parameters. The fixed midpoint is gated; validation/test stay closed."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    args.mature_checkpoint = args.mature_checkpoint.expanduser().resolve(strict=True)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B41ContractError("DINOv3 weight changed.")
    if b21.sha256_file(args.mature_checkpoint) != B38_CONTROL_CHECKPOINT_SHA256:
        raise B41ContractError("Mature B38 control checkpoint changed.")
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

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from trkh.recipes import pretrained_classf_b4 as b4
from trkh.training.train import build_configs, parse_args as parse_train_args


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B5_POSF_QV_LORA_R4_20260731"
RUN_PREFIX = "pretrained_dinov3_classf_b5_posf_qv_lora_r4"

# B5 inherits every B2/B4 model, data, sampler, optimizer, and trainability lock.
REQUIRED_BRANCH = b4.REQUIRED_BRANCH
B2_BEST_RELATIVE_PATH = b4.B2_BEST_RELATIVE_PATH
B2_BEST_SHA256 = b4.B2_BEST_SHA256
B2_SOURCE_COMMIT = b4.B2_SOURCE_COMMIT
B2_SOURCE_TREE_SHA256 = b4.B2_SOURCE_TREE_SHA256
B2_RECIPE_TRAIN_CONTRACT_SHA256 = b4.B2_RECIPE_TRAIN_CONTRACT_SHA256
DATASET_IMAGE_TREE_SHA256 = b4.DATASET_IMAGE_TREE_SHA256
DINO_MODEL_NAME = b4.DINO_MODEL_NAME
DINO_SHA256 = b4.DINO_SHA256
LORA_LAYERS = b4.LORA_LAYERS
LORA_RANK = b4.LORA_RANK
LORA_ALPHA = b4.LORA_ALPHA
LORA_DROPOUT = b4.LORA_DROPOUT
EXPECTED_LORA_TRAINABLE_PARAMETERS = b4.EXPECTED_LORA_TRAINABLE_PARAMETERS
TRAINABLE_PREFIXES = b4.TRAINABLE_PREFIXES
STAGES = b4.STAGES

SEMANTIC_ATTRIBUTE_LOSS_WEIGHT = 0.006
SEMANTIC_ATTRIBUTE_SPECS = (
    "lower:0|1,2,3;"
    "upper:0,1|2,3;"
    "rotten:0,1,2,3|4"
)
SEMANTIC_ATTRIBUTE_TASKS = 3

ABSOLUTE_PROBE_GATES: Mapping[str, float | int] = {
    "validation_support": 2479,
    "accuracy": 0.8879,
    "macro_f1": 0.8495,
    "class1_precision": 0.610,
    "class1_recall": 0.708,
    "class1_f1": 0.665,
    "class1_tp": 112,
    "restricted_fp_max": 69,
    "class2_to_1_max": 39,
}

RELATIVE_PROBE_GATES: Mapping[str, float | int] = {
    "accuracy_delta_min": -0.0020,
    "macro_f1_delta_min": -0.0015,
    "class1_precision_delta_min": 0.010,
    "class1_recall_delta_min": -0.012,
    "class1_f1_delta_min": 0.005,
    "class1_tp_delta_min": -2,
    "restricted_fp_reduction_min": 5,
    "class2_to_1_reduction_min": 3,
}

NO_REPEAT_RATIONALE: Mapping[str, object] = {
    "closed_recipe": (
        "scratch/full-readout semantic-attribute weight=0.02 with "
        "maturity+transport+quality grouping"
    ),
    "material_differences": [
        "verified DINOv3 B2 initialization with the exact B4 Q/V-LoRA adapter",
        "classifier, normalization, and base DINO parameters remain frozen",
        "the parameter-free loss can update only the 24,576 Q/V-LoRA parameters",
        "lower/upper/rotten tasks directly bracket class 1 against 0/2/4",
        "no cumulative head, logit residual, pAUC, margin, or post-hoc threshold",
    ],
    "one_shot": True,
    "validation_guided_weight_or_spec_sweep_permitted": False,
}


def run_name(stage: str, run_tag: str) -> str:
    stage_name = str(stage).strip().lower()
    if stage_name not in STAGES:
        raise ValueError(f"Unsupported B5 stage: {stage!r}.")
    return f"{RUN_PREFIX}_{stage_name}_{run_tag}"


def _remove_value_option(args: list[str], option: str) -> str:
    try:
        return b4._remove_option(args, option)
    except AssertionError as error:
        raise ValueError(f"B5 contract is missing required option {option}.") from error


def build_train_args(
    *,
    data_yaml: Path,
    dino_checkpoint: Path,
    b2_checkpoint: Path,
    output_dir: Path,
    stage: str,
    run_tag: str,
    batch_size: int = 24,
    num_workers: int = 4,
    eval_num_workers: int = 2,
    seed: int = 42,
    source_commit: str = "",
    source_tree_sha256: str = "",
    dataset_image_tree_sha256: str = "",
) -> list[str]:
    """Apply the sole B5 POSF loss delta to the exact locked B4 command."""

    args = b4.build_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=dino_checkpoint,
        b2_checkpoint=b2_checkpoint,
        output_dir=output_dir,
        stage=stage,
        run_tag=run_tag,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_num_workers=eval_num_workers,
        seed=seed,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        dataset_image_tree_sha256=dataset_image_tree_sha256,
    )
    _remove_value_option(args, "--recipe-train-contract-sha256")
    b4._replace_option(args, "--run-name", run_name(stage, run_tag))
    b4._replace_option(args, "--experiment-protocol-id", PROTOCOL_ID)
    args.extend(
        [
            "--semantic-attribute-loss-weight",
            str(SEMANTIC_ATTRIBUTE_LOSS_WEIGHT),
            "--semantic-attribute-specs",
            SEMANTIC_ATTRIBUTE_SPECS,
        ]
    )
    args.extend(
        [
            "--recipe-train-contract-sha256",
            b4._train_contract_sha256(args),
        ]
    )
    return args


def _b4_control_view(train_args: Sequence[str]) -> list[str]:
    """Remove the declared POSF delta so B4 can revalidate all inherited locks."""

    control = list(train_args)
    _remove_value_option(control, "--semantic-attribute-loss-weight")
    _remove_value_option(control, "--semantic-attribute-specs")
    b4._replace_option(control, "--experiment-protocol-id", b4.PROTOCOL_ID)
    return control


def validate_b5_train_args(train_args: Sequence[str]) -> Dict[str, object]:
    raw_args = list(train_args)
    digest_option = "--recipe-train-contract-sha256"
    if raw_args.count(digest_option) != 1:
        raise ValueError("B5 train contract requires exactly one recipe digest.")
    digest_index = raw_args.index(digest_option)
    if digest_index != len(raw_args) - 2:
        raise ValueError("B5 recipe digest must be the final value option.")
    observed_digest = str(raw_args[digest_index + 1]).strip().lower()
    expected_digest = b4._train_contract_sha256(raw_args[:digest_index])
    if observed_digest != expected_digest:
        raise ValueError(
            "B5 recipe train-contract digest mismatch: "
            f"observed={observed_digest}, expected={expected_digest}."
        )
    for option in (
        "--semantic-attribute-loss-weight",
        "--semantic-attribute-specs",
    ):
        if raw_args.count(option) != 1:
            raise ValueError(f"B5 requires exactly one {option} option.")

    inherited = b4.validate_b4_train_args(_b4_control_view(train_args))
    parsed = parse_train_args(train_args)
    model_config, train_config, _augmentation_config = build_configs(parsed)

    checks = {
        "experiment_protocol_id": (
            train_config.experiment_protocol_id,
            PROTOCOL_ID,
        ),
        "run_name_prefix": (
            str(parsed.run_name).startswith(f"{RUN_PREFIX}_"),
            True,
        ),
        "semantic_attribute_loss_weight": (
            train_config.semantic_attribute_loss_weight,
            SEMANTIC_ATTRIBUTE_LOSS_WEIGHT,
        ),
        "semantic_attribute_specs": (
            train_config.semantic_attribute_specs,
            SEMANTIC_ATTRIBUTE_SPECS,
        ),
        "ordinal_maturity_head": (model_config.ordinal_maturity_head, False),
        "cumulative_ordinal_head": (model_config.cumulative_ordinal_head, False),
        "ordinal_maturity_loss_weight": (
            train_config.ordinal_maturity_loss_weight,
            0.0,
        ),
        "cumulative_ordinal_loss_weight": (
            train_config.cumulative_ordinal_loss_weight,
            0.0,
        ),
        "ordinal_distribution_loss_weight": (
            train_config.ordinal_distribution_loss_weight,
            0.0,
        ),
        "ordinal_boundary_loss_weight": (
            train_config.ordinal_boundary_loss_weight,
            0.0,
        ),
        "focus_auc_rank_loss_weight": (
            train_config.focus_auc_rank_loss_weight,
            0.0,
        ),
        "focus_partial_auc_loss_weight": (
            train_config.focus_partial_auc_loss_weight,
            0.0,
        ),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"B5 train-argument contract mismatch: {mismatches}.")

    inherited.update(
        {
            "protocol": PROTOCOL_ID,
            "semantic_attribute_loss_weight": SEMANTIC_ATTRIBUTE_LOSS_WEIGHT,
            "semantic_attribute_specs": SEMANTIC_ATTRIBUTE_SPECS,
            "semantic_attribute_tasks": SEMANTIC_ATTRIBUTE_TASKS,
            "no_repeat_rationale": dict(NO_REPEAT_RATIONALE),
        }
    )
    return inherited


def _required_finite_metrics(
    values: Mapping[str, object],
    names: Sequence[str],
    *,
    label: str,
) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for name in names:
        if name not in values:
            raise ValueError(f"Missing {label} probe metric: {name}.")
        try:
            value = float(values[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid {label} probe metric: {name}.") from error
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {label} probe metric: {name}={value}.")
        parsed[name] = value
    return parsed


def evaluate_probe_gate(
    candidate: Mapping[str, object],
    b4_control: Mapping[str, object],
) -> Dict[str, object]:
    """Apply the predeclared B5 absolute and matched-B4 promotion gate."""

    metric_names = (
        "accuracy",
        "macro_f1",
        "class1_precision",
        "class1_recall",
        "class1_f1",
        "class1_tp",
        "restricted_fp",
        "class0_to_1",
        "class2_to_1",
        "class4_to_1",
    )
    observed = _required_finite_metrics(candidate, metric_names, label="B5")
    control = _required_finite_metrics(b4_control, metric_names, label="B4")
    operational_names = (
        "validation_support",
        "nonfinite_steps",
        "train_semantic_attribute_tasks",
        "train_semantic_attribute_loss",
        "trainable_parameters",
    )
    operational = _required_finite_metrics(
        candidate,
        operational_names,
        label="B5 operational",
    )
    test_inference_performed = bool(
        candidate.get("test_inference_performed", True)
    )

    dynamic_thresholds = {
        "accuracy_min": max(
            float(ABSOLUTE_PROBE_GATES["accuracy"]),
            control["accuracy"]
            + float(RELATIVE_PROBE_GATES["accuracy_delta_min"]),
        ),
        "macro_f1_min": max(
            float(ABSOLUTE_PROBE_GATES["macro_f1"]),
            control["macro_f1"]
            + float(RELATIVE_PROBE_GATES["macro_f1_delta_min"]),
        ),
        "class1_precision_min": max(
            float(ABSOLUTE_PROBE_GATES["class1_precision"]),
            control["class1_precision"]
            + float(RELATIVE_PROBE_GATES["class1_precision_delta_min"]),
        ),
        "class1_recall_min": max(
            float(ABSOLUTE_PROBE_GATES["class1_recall"]),
            control["class1_recall"]
            + float(RELATIVE_PROBE_GATES["class1_recall_delta_min"]),
        ),
        "class1_f1_min": max(
            float(ABSOLUTE_PROBE_GATES["class1_f1"]),
            control["class1_f1"]
            + float(RELATIVE_PROBE_GATES["class1_f1_delta_min"]),
        ),
        "class1_tp_min": max(
            float(ABSOLUTE_PROBE_GATES["class1_tp"]),
            control["class1_tp"]
            + float(RELATIVE_PROBE_GATES["class1_tp_delta_min"]),
        ),
        "restricted_fp_max": min(
            float(ABSOLUTE_PROBE_GATES["restricted_fp_max"]),
            control["restricted_fp"]
            - float(RELATIVE_PROBE_GATES["restricted_fp_reduction_min"]),
        ),
        "class2_to_1_max": min(
            float(ABSOLUTE_PROBE_GATES["class2_to_1_max"]),
            control["class2_to_1"]
            - float(RELATIVE_PROBE_GATES["class2_to_1_reduction_min"]),
        ),
    }
    checks = {
        "validation_support_eq_2479": operational["validation_support"]
        == float(ABSOLUTE_PROBE_GATES["validation_support"]),
        "no_nonfinite_steps": operational["nonfinite_steps"] == 0.0,
        "semantic_attribute_tasks_eq_3": operational[
            "train_semantic_attribute_tasks"
        ]
        == float(SEMANTIC_ATTRIBUTE_TASKS),
        "semantic_attribute_loss_finite_positive": operational[
            "train_semantic_attribute_loss"
        ]
        > 0.0,
        "trainable_parameters_eq_24576": operational["trainable_parameters"]
        == float(EXPECTED_LORA_TRAINABLE_PARAMETERS),
        "test_inference_not_performed": not test_inference_performed,
        "accuracy_gate": observed["accuracy"] >= dynamic_thresholds["accuracy_min"],
        "macro_f1_gate": observed["macro_f1"]
        >= dynamic_thresholds["macro_f1_min"],
        "class1_precision_gate": observed["class1_precision"]
        >= dynamic_thresholds["class1_precision_min"],
        "class1_recall_gate": observed["class1_recall"]
        >= dynamic_thresholds["class1_recall_min"],
        "class1_f1_gate": observed["class1_f1"]
        >= dynamic_thresholds["class1_f1_min"],
        "class1_tp_gate": observed["class1_tp"]
        >= dynamic_thresholds["class1_tp_min"],
        "restricted_fp_gate": observed["restricted_fp"]
        <= dynamic_thresholds["restricted_fp_max"],
        "class2_to_1_gate": observed["class2_to_1"]
        <= dynamic_thresholds["class2_to_1_max"],
        "class0_to_1_not_worse": observed["class0_to_1"]
        <= control["class0_to_1"],
        "class4_to_1_not_worse": observed["class4_to_1"]
        <= control["class4_to_1"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dynamic_thresholds": dynamic_thresholds,
        "candidate": observed,
        "b4_control": control,
        "absolute_gates": dict(ABSOLUTE_PROBE_GATES),
        "relative_gates": dict(RELATIVE_PROBE_GATES),
    }


def _best_history_row(run_dir: Path) -> Dict[str, object]:
    result = b4._best_history_row(run_dir)
    summary_path = Path(run_dir) / "summary.json"
    history_path = Path(run_dir) / "history.csv"
    if not result or not summary_path.is_file() or not history_path.is_file():
        return result
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_epoch = int(summary.get("best_epoch", -1))
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected = next(
            (
                row
                for row in csv.DictReader(handle)
                if int(float(row.get("epoch", -1))) == best_epoch
            ),
            {},
        )
    result.update(
        {
            "train_semantic_attribute_loss": selected.get(
                "train_semantic_attribute_loss"
            ),
            "train_semantic_attribute_tasks": selected.get(
                "train_semantic_attribute_tasks"
            ),
        }
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked B5 partial-order semantic-factorization Q/V-LoRA "
            "smoke or probe on canonical class_f."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "probe"),
        default="preflight",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--canonical-attestation", type=Path, default=None)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=repo_root / "configs" / "class_f_5class_dev.yaml",
    )
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--b2-checkpoint",
        type=Path,
        default=repo_root / B2_BEST_RELATIVE_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--run-tag", type=str, default="20260731_local")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(8, 12, 16, 24),
        default=24,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument(
        "--skip-focused-tests",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    data_yaml = args.data.resolve()
    training_data_yaml = args.train_data.resolve()
    dino_checkpoint = b4._absolute_path_without_resolving_symlink(
        args.dino_checkpoint
    )
    b2_checkpoint = args.b2_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dino_checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {dino_checkpoint}")
    dino_sha = b4._sha256(dino_checkpoint)
    if dino_sha != DINO_SHA256:
        raise ValueError(
            f"DINOv3 SHA-256 mismatch: observed={dino_sha}, expected={DINO_SHA256}."
        )
    dataset_contract = (
        b4.load_canonical_attestation(
            args.canonical_attestation,
            data_yaml=data_yaml,
        )
        if args.canonical_attestation is not None
        else b4.validate_canonical_classf(data_yaml)
    )
    if str(dataset_contract.get("image_tree_sha256", "")).lower() != (
        DATASET_IMAGE_TREE_SHA256
    ):
        raise ValueError("B5 canonical dataset image-tree SHA-256 drifted.")
    development_contract = b4.validate_development_data_yaml(
        training_data_yaml,
        canonical_contract=dataset_contract,
    )
    b2_contract = b4.validate_locked_b2_checkpoint(
        b2_checkpoint,
        repo_root=repo_root,
        training_data_yaml=training_data_yaml,
    )

    git = b4._git_snapshot(repo_root)
    b4.validate_branch_contract(git, require_clean=args.mode == "probe")
    observed_source_tree_sha256 = b4.source_tree_sha256(repo_root)
    source_commit = str(git["commit"]).strip().lower()
    stage = "smoke" if args.mode == "preflight" else args.mode
    train_args = build_train_args(
        data_yaml=training_data_yaml,
        dino_checkpoint=dino_checkpoint,
        b2_checkpoint=b2_checkpoint,
        output_dir=output_dir,
        stage=stage,
        run_tag=args.run_tag,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_num_workers=args.eval_num_workers,
        seed=args.seed,
        source_commit=source_commit,
        source_tree_sha256=observed_source_tree_sha256,
        dataset_image_tree_sha256=DATASET_IMAGE_TREE_SHA256,
    )
    train_contract = validate_b5_train_args(train_args)
    if int(train_contract["effective_batch_size"]) != 48:
        raise RuntimeError("B5 effective batch-size contract drifted from 48.")
    model_preflight = b4.preflight_b4_model_contract(
        train_args,
        b2_checkpoint=b2_checkpoint,
    )

    focused_tests = [
        repo_root / "tests" / "test_pretrained_classf_b5_recipe.py",
        repo_root / "tests" / "test_pretrained_classf_b4_recipe.py",
        repo_root / "tests" / "test_timm_qv_lora.py",
        repo_root / "tests" / "test_semantic_attribute_loss.py",
        repo_root / "tests" / "test_pretrained_classf_recipe.py",
        repo_root / "tests" / "test_resume_weight_and_distillation_source.py",
        repo_root / "tests" / "test_canonical_classf_defaults.py",
    ]
    test_result: Dict[str, object] = {
        "skipped": bool(args.skip_focused_tests),
        "paths": [str(path) for path in focused_tests],
    }
    if not args.skip_focused_tests:
        completed = subprocess.run(
            [str(args.python.resolve()), "-m", "pytest", "-q", *map(str, focused_tests)],
            cwd=repo_root,
            check=False,
        )
        test_result["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError("Focused B5 tests failed.")

    preflight_dir = output_dir / f"preflight_classf_b5_{args.run_tag}"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "test_locked": True,
        "test_model_inference_performed": False,
        "dataset": dataset_contract,
        "development_data": development_contract,
        "dino": {
            "model_name": DINO_MODEL_NAME,
            "checkpoint": str(dino_checkpoint),
            "sha256": dino_sha,
        },
        "b2_resume": b2_contract,
        "b4_inherited_delta": {
            "layers": LORA_LAYERS,
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "trainable_prefixes": list(TRAINABLE_PREFIXES),
            "expected_trainable_parameters": (
                EXPECTED_LORA_TRAINABLE_PARAMETERS
            ),
            "base_dino_and_classifier_frozen": True,
        },
        "b5_delta": {
            "semantic_attribute_loss_weight": (
                SEMANTIC_ATTRIBUTE_LOSS_WEIGHT
            ),
            "semantic_attribute_specs": SEMANTIC_ATTRIBUTE_SPECS,
            "semantic_attribute_tasks": SEMANTIC_ATTRIBUTE_TASKS,
            "new_model_parameters": 0,
            "inference_graph_delta": False,
            "no_repeat_rationale": dict(NO_REPEAT_RATIONALE),
        },
        "probe_gate": {
            "absolute": dict(ABSOLUTE_PROBE_GATES),
            "relative_to_matched_b4": dict(RELATIVE_PROBE_GATES),
            "validation_guided_sweep_permitted": False,
        },
        "git": git,
        "source": {
            "commit": source_commit,
            "tree_sha256": observed_source_tree_sha256,
            "launcher_sha256": b4._sha256(Path(__file__)),
        },
        "runtime": {
            "python": str(args.python.resolve()),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "amp_dtype_request": args.amp_dtype,
            "batch_size": int(args.batch_size),
            "effective_batch_size": 48,
        },
        "focused_tests": test_result,
        "train_contract": train_contract,
        "model_preflight": model_preflight,
        "train_args": train_args,
    }
    manifest_path = preflight_dir / f"{args.mode}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"B5 preflight passed: {manifest_path}", flush=True)
    if args.mode == "preflight":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B5 smoke/probe training.")

    run_dir = output_dir / run_name(stage, args.run_tag)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Run directory already exists: {run_dir}. Use a new --run-tag."
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["TRKH_AMP_DTYPE"] = str(args.amp_dtype)
    environment.setdefault("OMP_NUM_THREADS", "4")
    command = [
        str(args.python.resolve()),
        "-m",
        "trkh.training.train",
        *train_args,
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
    )
    marker = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "stage": stage,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": int(completed.returncode),
        "preflight_manifest": str(manifest_path),
        "test_locked": True,
        "test_model_inference_performed": False,
        "git_commit": source_commit,
        "metrics": _best_history_row(run_dir),
        "gate_requires_matched_b4_metrics": stage == "probe",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / "b5_stage_complete.json"
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"B5 {stage} failed with code {completed.returncode}; "
            f"evidence: {run_dir}."
        )
    print(f"B5 {stage} completed: {marker_path}", flush=True)


if __name__ == "__main__":
    main()

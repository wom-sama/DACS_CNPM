from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from trkh.core.config import to_serializable
from trkh.core.utils import build_optimizer_param_groups, load_checkpoint
from trkh.models.model import create_model
from trkh.models.timm_qv_lora import (
    TimmQKVLoRALinear,
    timm_qv_lora_parameter_prefixes,
)
from trkh.recipes.pretrained_classf_b0 import (
    B2_TEMPERED_P05_PROTOCOL_ID,
    DINO_MODEL_NAME,
    DINO_SHA256,
    EXPECTED_CLASS_NAMES,
    _absolute_path_without_resolving_symlink,
    _git_snapshot,
    _sha256,
    build_train_args as build_b2_train_args,
    load_canonical_attestation,
    source_tree_sha256,
    validate_development_data_yaml,
)
from trkh.tools.validate_canonical_classf import validate_canonical_classf
from trkh.training.train import (
    _apply_trainable_module_prefixes,
    _load_model_state_allowing_extensions,
    build_configs,
    parse_args as parse_train_args,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B4_QV_LORA_R4_20260731"
REQUIRED_BRANCH = "research/pretrained-classf-b1"
RUN_PREFIX = "pretrained_dinov3_classf_b4_qv_lora_r4"

B2_BEST_RELATIVE_PATH = Path(
    "runs"
    "/pretrained_dinov3_classf_tempered_p05_probe_20260731_local_b2_tempered_p05"
    "/checkpoints/best.pt"
)
B2_BEST_SHA256 = (
    "4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6"
)
B2_SOURCE_COMMIT = "f1d79def090e347c0586efae4003b2579d368a28"
B2_SOURCE_TREE_SHA256 = (
    "61b860a5549ac172d06c64ee1c5f9fe035cc9853074312f2fe5a2de80390e747"
)
B2_RECIPE_TRAIN_CONTRACT_SHA256 = (
    "2f53fd5e3f4fc154a86a72b1985328c59fffd9a5d376b14e89f9c70ad8311aa6"
)
DATASET_IMAGE_TREE_SHA256 = (
    "70a1b7d2b4c6f80e28fe3f0f714f1ba3e8ab654a90ce50b1e9cae8e4dac4a503"
)

LORA_LAYERS = "8,9,10,11"
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.05
EXPECTED_LORA_TRAINABLE_PARAMETERS = 24_576
TRAINABLE_PREFIXES = tuple(timm_qv_lora_parameter_prefixes(LORA_LAYERS))


@dataclass(frozen=True)
class Stage:
    epochs: int
    scheduler_total_epochs: int
    warmup_epochs: int
    patience: int
    max_train_batches: int
    max_val_batches: int


STAGES: Mapping[str, Stage] = {
    "smoke": Stage(
        epochs=1,
        scheduler_total_epochs=5,
        warmup_epochs=0,
        patience=2,
        max_train_batches=4,
        max_val_batches=2,
    ),
    "probe": Stage(
        epochs=5,
        scheduler_total_epochs=5,
        warmup_epochs=1,
        patience=5,
        max_train_batches=120,
        max_val_batches=0,
    ),
}


def run_name(stage: str, run_tag: str) -> str:
    stage_name = str(stage).strip().lower()
    if stage_name not in STAGES:
        raise ValueError(f"Unsupported B4 stage: {stage!r}.")
    return f"{RUN_PREFIX}_{stage_name}_{run_tag}"


def _replace_option(args: list[str], option: str, value: object) -> None:
    try:
        index = args.index(option)
    except ValueError as error:
        raise AssertionError(f"Inherited B2 recipe is missing {option}.") from error
    if index + 1 >= len(args):
        raise AssertionError(f"Inherited B2 recipe has no value for {option}.")
    args[index + 1] = str(value)


def _remove_option(args: list[str], option: str) -> str:
    try:
        index = args.index(option)
    except ValueError as error:
        raise AssertionError(f"Inherited B2 recipe is missing {option}.") from error
    if index + 1 >= len(args):
        raise AssertionError(f"Inherited B2 recipe has no value for {option}.")
    value = args[index + 1]
    del args[index : index + 2]
    return value


def _train_contract_sha256(args_without_digest: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(args_without_digest),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
    """Build the locked B4 command by applying only the declared B4 delta to B2."""

    stage_name = str(stage).strip().lower()
    if stage_name not in STAGES:
        raise ValueError(f"Unsupported B4 stage: {stage!r}.")
    stage_spec = STAGES[stage_name]
    args = build_b2_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=dino_checkpoint,
        output_dir=output_dir,
        stage=stage_name,
        run_tag=run_tag,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_num_workers=eval_num_workers,
        seed=seed,
        auto_resume=False,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        dataset_image_tree_sha256=dataset_image_tree_sha256,
        experiment="b2-tempered-p05",
    )

    _remove_option(args, "--recipe-train-contract-sha256")
    _replace_option(args, "--run-name", run_name(stage_name, run_tag))
    _replace_option(args, "--experiment-protocol-id", PROTOCOL_ID)
    _replace_option(args, "--epochs", stage_spec.epochs)
    _replace_option(
        args,
        "--scheduler-total-epochs",
        stage_spec.scheduler_total_epochs,
    )
    _replace_option(args, "--warmup-epochs", stage_spec.warmup_epochs)
    _replace_option(args, "--patience", stage_spec.patience)
    _replace_option(args, "--max-train-batches", stage_spec.max_train_batches)
    _replace_option(args, "--max-val-batches", stage_spec.max_val_batches)
    _replace_option(args, "--learning-rate", "2e-4")
    _replace_option(args, "--backbone-lr-scale", "1")

    args.extend(
        [
            "--resume",
            str(Path(b2_checkpoint).resolve()),
            "--resume-use-cli-config",
            "--resume-weight-source",
            "selected",
            "--resume-reset-epoch",
            "--resume-reset-optimizer",
            "--resume-reset-scheduler",
            "--resume-reset-scaler",
            "--timm-qv-lora",
            "--timm-qv-lora-layers",
            LORA_LAYERS,
            "--timm-qv-lora-rank",
            str(LORA_RANK),
            "--timm-qv-lora-alpha",
            str(LORA_ALPHA),
            "--timm-qv-lora-dropout",
            str(LORA_DROPOUT),
            "--trainable-module-prefixes",
            ",".join(TRAINABLE_PREFIXES),
        ]
    )
    args.extend(
        [
            "--recipe-train-contract-sha256",
            _train_contract_sha256(args),
        ]
    )
    return args


def validate_locked_b2_checkpoint(
    checkpoint_path: Path,
    *,
    repo_root: Path,
    training_data_yaml: Path,
) -> Dict[str, object]:
    """Fail closed unless the exact selected B2 EMA checkpoint is supplied."""

    checkpoint_path = Path(checkpoint_path).resolve()
    expected_path = (Path(repo_root).resolve() / B2_BEST_RELATIVE_PATH).resolve()
    if checkpoint_path != expected_path:
        raise ValueError(
            "B4 resume path must be the locked local B2 best checkpoint: "
            f"observed={checkpoint_path}, expected={expected_path}."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Locked B2 checkpoint not found: {checkpoint_path}")
    observed_sha = _sha256(checkpoint_path)
    if observed_sha != B2_BEST_SHA256:
        raise ValueError(
            "B2 checkpoint SHA-256 mismatch: "
            f"observed={observed_sha}, expected={B2_BEST_SHA256}."
        )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if list(checkpoint.get("class_names", [])) != list(EXPECTED_CLASS_NAMES):
        raise ValueError("B2 checkpoint class order is not canonical class_f.")
    observed_data_yaml = Path(str(checkpoint.get("data_yaml", ""))).resolve()
    expected_data_yaml = Path(training_data_yaml).resolve()
    if observed_data_yaml != expected_data_yaml:
        raise ValueError(
            "B2 checkpoint data YAML mismatch: "
            f"observed={observed_data_yaml}, expected={expected_data_yaml}."
        )

    model_config = checkpoint.get("model_config", {})
    train_config = checkpoint.get("train_config", {})
    if not isinstance(model_config, Mapping) or not isinstance(train_config, Mapping):
        raise ValueError("B2 checkpoint model/train configuration is invalid.")
    expected_model = {
        "model_type": "timm_classifier",
        "research_track": "pretrained",
        "pretrained": True,
        "timm_model_name": DINO_MODEL_NAME,
        "pretrained_checkpoint_sha256": DINO_SHA256,
    }
    expected_train = {
        "experiment_protocol_id": B2_TEMPERED_P05_PROTOCOL_ID,
        "source_commit": B2_SOURCE_COMMIT,
        "source_tree_sha256": B2_SOURCE_TREE_SHA256,
        "dataset_image_tree_sha256": DATASET_IMAGE_TREE_SHA256,
        "recipe_train_contract_sha256": B2_RECIPE_TRAIN_CONTRACT_SHA256,
        "balanced_epoch_sampling": False,
        "tempered_class_sampling_power": 0.5,
        "classification_loss": "ldam_focal",
        "ldam_max_margin": 0.3,
        "ldam_scale": 18.0,
        "focal_loss_gamma": 1.0,
        "focal_loss_mix": 0.1,
        "label_smoothing": 0.02,
        "batch_size": 24,
        "grad_accum_steps": 2,
        "epochs": 5,
        "scheduler_total_epochs": 5,
        "max_train_batches": 120,
        "max_val_batches": 0,
        "warmup_epochs": 1,
    }
    mismatches: Dict[str, object] = {}
    for key, expected in expected_model.items():
        observed = model_config.get(key)
        if observed != expected:
            mismatches[f"model_config.{key}"] = {
                "observed": observed,
                "expected": expected,
            }
    for key, expected in expected_train.items():
        observed = train_config.get(key)
        if observed != expected:
            mismatches[f"train_config.{key}"] = {
                "observed": observed,
                "expected": expected,
            }
    checkpoint_contract = {
        "epoch": (checkpoint.get("epoch"), 4),
        "best_epoch": (checkpoint.get("best_epoch"), 4),
        "checkpoint_weight_source": (
            checkpoint.get("checkpoint_weight_source"),
            "ema",
        ),
        "validation_weight_source": (
            checkpoint.get("validation_weight_source"),
            "ema",
        ),
    }
    for key, (observed, expected) in checkpoint_contract.items():
        if observed != expected:
            mismatches[key] = {"observed": observed, "expected": expected}
    if mismatches:
        raise ValueError(f"Locked B2 checkpoint provenance mismatch: {mismatches}.")

    return {
        "checkpoint": str(checkpoint_path),
        "sha256": observed_sha,
        "epoch": 4,
        "best_epoch": 4,
        "weight_source": "selected_ema_model_state",
        "data_yaml": str(observed_data_yaml),
        "source_commit": B2_SOURCE_COMMIT,
        "source_tree_sha256": B2_SOURCE_TREE_SHA256,
        "recipe_train_contract_sha256": B2_RECIPE_TRAIN_CONTRACT_SHA256,
        "dataset_image_tree_sha256": DATASET_IMAGE_TREE_SHA256,
        "test_inference_performed": False,
    }


def validate_branch_contract(
    git_snapshot: Mapping[str, object],
    *,
    require_clean: bool,
) -> None:
    if not bool(git_snapshot.get("available")):
        raise RuntimeError("B4 requires an auditable Git worktree.")
    branch = str(git_snapshot.get("branch", ""))
    if branch != REQUIRED_BRANCH:
        raise RuntimeError(
            f"B4 requires branch {REQUIRED_BRANCH!r}, observed={branch!r}."
        )
    commit = str(git_snapshot.get("commit", "")).lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"B4 source commit is invalid: {commit!r}.")
    if require_clean and not bool(git_snapshot.get("clean")):
        raise RuntimeError("B4 probe requires a clean worktree.")


def validate_b4_train_args(train_args: Sequence[str]) -> Dict[str, object]:
    parsed = parse_train_args(train_args)
    model_config, train_config, _augmentation_config = build_configs(parsed)
    observed_prefixes = tuple(
        item.strip()
        for item in str(train_config.trainable_module_prefixes).split(",")
        if item.strip()
    )
    checks = {
        "model_type": (model_config.model_type, "timm_classifier"),
        "research_track": (model_config.research_track, "pretrained"),
        "timm_model_name": (model_config.timm_model_name, DINO_MODEL_NAME),
        "timm_qv_lora": (model_config.timm_qv_lora, True),
        "timm_qv_lora_layers": (model_config.timm_qv_lora_layers, LORA_LAYERS),
        "timm_qv_lora_rank": (model_config.timm_qv_lora_rank, LORA_RANK),
        "timm_qv_lora_alpha": (model_config.timm_qv_lora_alpha, LORA_ALPHA),
        "timm_qv_lora_dropout": (
            model_config.timm_qv_lora_dropout,
            LORA_DROPOUT,
        ),
        "trainable_module_prefixes": (observed_prefixes, TRAINABLE_PREFIXES),
        "learning_rate": (train_config.learning_rate, 2e-4),
        "backbone_lr_scale": (train_config.backbone_lr_scale, 1.0),
        "tempered_class_sampling_power": (
            train_config.tempered_class_sampling_power,
            0.5,
        ),
        "balanced_epoch_sampling": (
            train_config.balanced_epoch_sampling,
            False,
        ),
        "experiment_protocol_id": (
            train_config.experiment_protocol_id,
            PROTOCOL_ID,
        ),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    resume_checks = {
        "resume_use_cli_config": (parsed.resume_use_cli_config, True),
        "resume_weight_source": (parsed.resume_weight_source, "selected"),
        "resume_reset_epoch": (parsed.resume_reset_epoch, True),
        "resume_reset_optimizer": (parsed.resume_reset_optimizer, True),
        "resume_reset_scheduler": (parsed.resume_reset_scheduler, True),
        "resume_reset_scaler": (parsed.resume_reset_scaler, True),
        "skip_final_test": (parsed.skip_final_test, True),
        "max_val_batches_is_test_locked": (parsed.max_val_batches >= 0, True),
    }
    mismatches.update(
        {
            key: {"observed": observed, "expected": expected}
            for key, (observed, expected) in resume_checks.items()
            if observed != expected
        }
    )
    if mismatches:
        raise ValueError(f"B4 train-argument contract mismatch: {mismatches}.")
    return {
        "protocol": PROTOCOL_ID,
        "trainable_prefixes": list(observed_prefixes),
        "expected_trainable_parameters": EXPECTED_LORA_TRAINABLE_PARAMETERS,
        "effective_batch_size": (
            int(train_config.batch_size) * int(train_config.grad_accum_steps)
        ),
        "test_locked": bool(parsed.skip_final_test),
        "recipe_train_contract_sha256": (
            train_config.recipe_train_contract_sha256
        ),
    }


def preflight_b4_model_contract(
    train_args: Sequence[str],
    *,
    b2_checkpoint: Path,
) -> Dict[str, object]:
    """Instantiate the real DINO target and prove the adapter-only contract."""

    parsed = parse_train_args(train_args)
    model_config, train_config, _augmentation_config = build_configs(parsed)
    checkpoint = load_checkpoint(Path(b2_checkpoint), map_location="cpu")
    model = create_model(
        num_classes=len(EXPECTED_CLASS_NAMES),
        model_config=to_serializable(model_config),
    )
    partial_load = _load_model_state_allowing_extensions(
        model,
        checkpoint["model_state"],
        allow_extensions=True,
    )
    if partial_load is None:
        raise RuntimeError("B4 must add exactly the declared LoRA state to B2.")
    missing_keys = list(partial_load.get("allowed_missing_keys", []))
    if len(missing_keys) != 16 or partial_load.get("unexpected_keys"):
        raise RuntimeError(
            "B4 partial-load contract drifted: "
            f"missing={missing_keys}, "
            f"unexpected={partial_load.get('unexpected_keys')}."
        )

    prefix_summary = _apply_trainable_module_prefixes(
        model,
        TRAINABLE_PREFIXES,
    )
    if (
        int(prefix_summary.get("trainable_parameters", 0))
        != EXPECTED_LORA_TRAINABLE_PARAMETERS
        or int(prefix_summary.get("trainable_tensors", 0)) != 16
    ):
        raise RuntimeError(f"B4 trainable parameter contract drifted: {prefix_summary}.")

    named_adapters = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TimmQKVLoRALinear)
    ]
    expected_module_names = [
        f"blocks.{index}.attn.qkv" for index in (8, 9, 10, 11)
    ]
    observed_module_names = [name for name, _module in named_adapters]
    if observed_module_names != expected_module_names:
        raise RuntimeError(
            "B4 target-module contract drifted: "
            f"observed={observed_module_names}, expected={expected_module_names}."
        )

    torch.manual_seed(20260731)
    maximum_zero_init_error = 0.0
    for _name, module in named_adapters:
        module.eval()
        inputs = torch.randn(2, 5, int(module.in_features))
        base = F.linear(inputs, module.weight, module.bias)
        adapted = module(inputs)
        maximum_zero_init_error = max(
            maximum_zero_init_error,
            float((base - adapted).abs().max().item()),
        )
    if maximum_zero_init_error != 0.0:
        raise RuntimeError(
            "B4 zero-init changed the inherited B2 QKV output: "
            f"max_abs={maximum_zero_init_error}."
        )

    optimizer_groups = build_optimizer_param_groups(
        model,
        train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        backbone_lr_scale=train_config.backbone_lr_scale,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=train_config.learning_rate,
    )
    optimizer_lrs = sorted(
        {float(group["lr"]) for group in optimizer.param_groups}
    )
    if optimizer_lrs != [2e-4]:
        raise RuntimeError(f"B4 adapter optimizer LR contract drifted: {optimizer_lrs}.")

    return {
        "model_name": model_config.timm_model_name,
        "adapter_modules": observed_module_names,
        "partial_load_missing_keys": missing_keys,
        "partial_load_unexpected_keys": [],
        "trainable_parameters": int(prefix_summary["trainable_parameters"]),
        "trainable_tensors": int(prefix_summary["trainable_tensors"]),
        "frozen_parameters": int(prefix_summary["frozen_parameters"]),
        "optimizer_group_names": [
            str(group.get("name", "")) for group in optimizer.param_groups
        ],
        "optimizer_lrs": optimizer_lrs,
        "maximum_zero_init_qkv_error": maximum_zero_init_error,
        "test_model_inference_performed": False,
    }


def _best_history_row(run_dir: Path) -> Dict[str, object]:
    summary_path = Path(run_dir) / "summary.json"
    history_path = Path(run_dir) / "history.csv"
    if not summary_path.is_file() or not history_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_epoch = int(summary.get("best_epoch", -1))
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next(
        (row for row in rows if int(float(row.get("epoch", -1))) == best_epoch),
        {},
    )
    return {
        "best_epoch": best_epoch,
        "best_accuracy": selected.get("val_accuracy"),
        "best_macro_f1": selected.get("val_macro_f1"),
        "best_class1_precision": selected.get("val_class_1_precision"),
        "best_class1_recall": selected.get("val_class_1_recall"),
        "best_class1_f1": selected.get("val_class_1_f1"),
        "stop_reason": summary.get("stop_reason"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the locked B4 DINOv3 Q/V-LoRA probe on canonical class_f."
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
    dino_checkpoint = _absolute_path_without_resolving_symlink(
        args.dino_checkpoint
    )
    b2_checkpoint = args.b2_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dino_checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {dino_checkpoint}")
    dino_sha = _sha256(dino_checkpoint)
    if dino_sha != DINO_SHA256:
        raise ValueError(
            f"DINOv3 SHA-256 mismatch: observed={dino_sha}, expected={DINO_SHA256}."
        )
    dataset_contract = (
        load_canonical_attestation(
            args.canonical_attestation,
            data_yaml=data_yaml,
        )
        if args.canonical_attestation is not None
        else validate_canonical_classf(data_yaml)
    )
    if str(dataset_contract.get("image_tree_sha256", "")).lower() != (
        DATASET_IMAGE_TREE_SHA256
    ):
        raise ValueError("B4 canonical dataset image-tree SHA-256 drifted.")
    development_contract = validate_development_data_yaml(
        training_data_yaml,
        canonical_contract=dataset_contract,
    )
    b2_contract = validate_locked_b2_checkpoint(
        b2_checkpoint,
        repo_root=repo_root,
        training_data_yaml=training_data_yaml,
    )

    git = _git_snapshot(repo_root)
    validate_branch_contract(
        git,
        require_clean=args.mode == "probe",
    )
    observed_source_tree_sha256 = source_tree_sha256(repo_root)
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
    train_contract = validate_b4_train_args(train_args)
    if int(train_contract["effective_batch_size"]) != 48:
        raise RuntimeError("B4 effective batch-size contract drifted from 48.")
    model_preflight = preflight_b4_model_contract(
        train_args,
        b2_checkpoint=b2_checkpoint,
    )

    focused_tests = [
        repo_root / "tests" / "test_pretrained_classf_b4_recipe.py",
        repo_root / "tests" / "test_timm_qv_lora.py",
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
            raise RuntimeError("Focused B4 tests failed.")

    preflight_dir = output_dir / f"preflight_classf_b4_{args.run_tag}"
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
        "b4_delta": {
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
        "git": git,
        "source": {
            "commit": source_commit,
            "tree_sha256": observed_source_tree_sha256,
            "launcher_sha256": _sha256(Path(__file__)),
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
    print(f"B4 preflight passed: {manifest_path}", flush=True)
    if args.mode == "preflight":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B4 smoke/probe training.")

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
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / "b4_stage_complete.json"
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"B4 {stage} failed with code {completed.returncode}; "
            f"evidence: {run_dir}."
        )
    print(f"B4 {stage} completed: {marker_path}", flush=True)


if __name__ == "__main__":
    main()

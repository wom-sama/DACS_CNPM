from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch

from trkh.core.config import load_data_spec
from trkh.core.utils import build_optimizer_param_groups, load_checkpoint
from trkh.data.dataset import ClassificationFolderDataset
from trkh.models.model import create_model
from trkh.tools.validate_canonical_classf import (
    CANONICAL_FILE_SHA256,
    CANONICAL_IMAGE_TREE_SHA256,
    validate_canonical_classf,
)
from trkh.training.train import build_configs, parse_args as parse_train_args


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B0_20260731"
DINO_MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
DINO_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DINO_SOURCE_URL = "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m"
DINO_SOURCE_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINO_SOURCE_LICENSE = "dinov3-license"
EXPECTED_PARAMETER_COUNT = 21_588_869
EXPECTED_CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
EXPECTED_DEVELOPMENT_COUNTS = {
    "train": [1987, 497, 1326, 2080, 2388],
    "val": [558, 158, 380, 494, 889],
}


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
    "full": Stage(
        epochs=30,
        scheduler_total_epochs=30,
        warmup_epochs=2,
        patience=6,
        max_train_batches=0,
        max_val_batches=0,
    ),
}
ACCUMULATION_BY_BATCH = {8: 6, 12: 4, 16: 3, 24: 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path_without_resolving_symlink(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.absolute()


def source_tree_sha256(repo_root: Path) -> str:
    repo_root = Path(repo_root).resolve()
    files = sorted((repo_root / "trkh").rglob("*.py"))
    files.extend(sorted((repo_root / "configs").rglob("*.yaml")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_development_data_yaml(
    data_yaml: Path,
    *,
    canonical_contract: Mapping[str, object],
) -> Dict[str, object]:
    data_yaml = Path(data_yaml).resolve()
    spec = load_data_spec(
        data_yaml,
        class_name_mode="raw",
        expected_num_classes=5,
    )
    if spec.data_format != "classification_folder":
        raise ValueError("B0 development data must use classification_folder.")
    if spec.has_test_split:
        raise ValueError("B0 development YAML must omit the test split.")
    if tuple(spec.class_names) != EXPECTED_CLASS_NAMES:
        raise ValueError(
            "B0 development class order mismatch: "
            f"{list(spec.class_names)!r} != {list(EXPECTED_CLASS_NAMES)!r}."
        )
    canonical_root = Path(str(canonical_contract["root"])).resolve()
    if spec.root.resolve() != canonical_root:
        raise ValueError(
            "B0 development YAML must point to the verified canonical root: "
            f"{spec.root.resolve()} != {canonical_root}."
        )
    split_counts: Dict[str, list[int]] = {}
    split_totals: Dict[str, int] = {}
    for split, expected_counts in EXPECTED_DEVELOPMENT_COUNTS.items():
        dataset = ClassificationFolderDataset.from_data_spec(spec, split=split)
        observed_counts = dataset.class_counts(5)
        if observed_counts != expected_counts:
            raise ValueError(
                f"B0 development {split} counts mismatch: "
                f"{observed_counts} != {expected_counts}."
            )
        split_counts[split] = observed_counts
        split_totals[split] = len(dataset)
    return {
        "data_yaml": str(data_yaml),
        "root": str(spec.root.resolve()),
        "class_names": list(spec.class_names),
        "split_class_counts": split_counts,
        "split_totals": split_totals,
        "has_test_split": False,
    }


def load_canonical_attestation(
    attestation_path: Path,
    *,
    data_yaml: Path,
) -> Dict[str, object]:
    """Reuse a same-session full image hash without reading the 1 GiB tree again."""

    attestation_path = Path(attestation_path).resolve()
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Canonical attestation must contain a JSON object.")
    expected_data_yaml = Path(data_yaml).resolve()
    observed_data_yaml = Path(str(payload.get("data_yaml", ""))).resolve()
    if observed_data_yaml != expected_data_yaml:
        raise ValueError(
            "Canonical attestation data_yaml mismatch: "
            f"{observed_data_yaml} != {expected_data_yaml}."
        )
    required = {
        "contract": "TRKH_CLASS_F_CANONICAL_V1_20260730",
        "status": "passed",
        "image_tree_sha256": CANONICAL_IMAGE_TREE_SHA256,
        "image_tree_file_count": 12019,
        "file_sha256": CANONICAL_FILE_SHA256,
        "test_model_inference_performed": False,
        "test_metrics_read": False,
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": expected}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Canonical attestation mismatch: {mismatches}.")
    observed_file_hashes = {
        name: _sha256(expected_data_yaml.parent / name)
        for name in CANONICAL_FILE_SHA256
    }
    if observed_file_hashes != CANONICAL_FILE_SHA256:
        raise ValueError(
            "Canonical files changed after image-tree attestation: "
            f"{observed_file_hashes}."
        )
    payload["attestation_path"] = str(attestation_path)
    payload["attestation_sha256"] = _sha256(attestation_path)
    payload["reused_without_rehashing_images"] = True
    return payload


def validate_auto_resume_checkpoint(
    checkpoint_path: Path,
    *,
    training_data_yaml: Path,
    expected_train_args: Sequence[str] | None = None,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if list(checkpoint.get("class_names", [])) != list(EXPECTED_CLASS_NAMES):
        raise ValueError("Auto-resume checkpoint class order is not canonical class_f.")
    checkpoint_data = checkpoint.get("data_yaml")
    if not checkpoint_data:
        raise ValueError("Auto-resume checkpoint is missing data_yaml provenance.")
    observed_data = Path(str(checkpoint_data)).resolve()
    expected_data = Path(training_data_yaml).resolve()
    if observed_data != expected_data:
        raise ValueError(
            "Auto-resume data YAML mismatch: "
            f"{observed_data} != {expected_data}."
        )
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Auto-resume checkpoint model_config is invalid.")
    required = {
        "model_type": (str(model_config.get("model_type", "")), "timm_classifier"),
        "research_track": (
            str(model_config.get("research_track", "")),
            "pretrained",
        ),
        "timm_model_name": (
            str(model_config.get("timm_model_name", "")),
            DINO_MODEL_NAME,
        ),
        "pretrained_checkpoint_sha256": (
            str(model_config.get("pretrained_checkpoint_sha256", "")).lower(),
            DINO_SHA256,
        ),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in required.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"Auto-resume pretrained contract mismatch: {mismatches}.")
    train_config = checkpoint.get("train_config", {})
    if not isinstance(train_config, Mapping):
        raise ValueError("Auto-resume checkpoint train_config is invalid.")
    resume_lineage: Dict[str, object] = {}
    if expected_train_args is not None:
        expected_parsed = parse_train_args(expected_train_args)
        expected_lineage = {
            "experiment_protocol_id": str(
                expected_parsed.experiment_protocol_id
            ),
            "source_commit": str(expected_parsed.source_commit).lower(),
            "source_tree_sha256": str(
                expected_parsed.source_tree_sha256
            ).lower(),
            "dataset_image_tree_sha256": str(
                expected_parsed.dataset_image_tree_sha256
            ).lower(),
            "recipe_train_contract_sha256": str(
                expected_parsed.recipe_train_contract_sha256
            ).lower(),
        }
        lineage_mismatches = {
            key: {
                "observed": str(train_config.get(key, "")).lower(),
                "expected": expected,
            }
            for key, expected in expected_lineage.items()
            if str(train_config.get(key, "")).lower() != expected.lower()
        }
        if lineage_mismatches:
            raise ValueError(
                "Auto-resume source/protocol/config lineage mismatch: "
                f"{lineage_mismatches}."
            )
        resume_lineage = expected_lineage
    return {
        "checkpoint": str(checkpoint_path),
        "data_yaml": str(observed_data),
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "model_type": required["model_type"][0],
        "research_track": required["research_track"][0],
        "timm_model_name": required["timm_model_name"][0],
        "pretrained_checkpoint_sha256": required[
            "pretrained_checkpoint_sha256"
        ][0],
        "lineage": resume_lineage,
    }


def run_name(stage: str, run_tag: str) -> str:
    return f"pretrained_dinov3_classf_direct_{stage}_{run_tag}"


def build_train_args(
    *,
    data_yaml: Path,
    dino_checkpoint: Path,
    output_dir: Path,
    stage: str,
    run_tag: str,
    batch_size: int = 24,
    num_workers: int = 4,
    eval_num_workers: int = 2,
    seed: int = 42,
    auto_resume: bool = False,
    source_commit: str = "",
    source_tree_sha256: str = "",
    dataset_image_tree_sha256: str = "",
) -> list[str]:
    stage_name = str(stage).strip().lower()
    if stage_name not in STAGES:
        raise ValueError(f"Unsupported stage: {stage!r}.")
    if int(batch_size) not in ACCUMULATION_BY_BATCH:
        raise ValueError(
            f"batch_size must be one of {sorted(ACCUMULATION_BY_BATCH)}."
        )
    grad_accum_steps = ACCUMULATION_BY_BATCH[int(batch_size)]
    if int(batch_size) * grad_accum_steps != 48:
        raise AssertionError("B0 effective batch contract drifted from 48.")
    if not str(run_tag).replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_tag may contain only letters, numbers, '_' and '-'.")
    spec = STAGES[stage_name]

    args = [
        "--data",
        str(Path(data_yaml).resolve()),
        "--class-name-mode",
        "raw",
        "--expected-num-classes",
        "5",
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--run-name",
        run_name(stage_name, run_tag),
        "--model-type",
        "timm_classifier",
        "--research-track",
        "pretrained",
        "--pretrained",
        "--timm-model-name",
        DINO_MODEL_NAME,
        "--pretrained-checkpoint-path",
        str(_absolute_path_without_resolving_symlink(dino_checkpoint)),
        "--pretrained-checkpoint-sha256",
        DINO_SHA256,
        "--pretrained-source-url",
        DINO_SOURCE_URL,
        "--pretrained-source-revision",
        DINO_SOURCE_REVISION,
        "--pretrained-source-license",
        DINO_SOURCE_LICENSE,
        "--image-size",
        "256",
        "--batch-size",
        str(int(batch_size)),
        "--grad-accum-steps",
        str(int(grad_accum_steps)),
        "--epochs",
        str(spec.epochs),
        "--scheduler-total-epochs",
        str(spec.scheduler_total_epochs),
        "--patience",
        str(spec.patience),
        "--learning-rate",
        "1.5e-4",
        "--backbone-lr-scale",
        "0.1",
        "--min-learning-rate",
        "1e-6",
        "--warmup-epochs",
        str(spec.warmup_epochs),
        "--warmup-start-factor",
        "0.1",
        "--weight-decay",
        "0.05",
        "--grad-clip-norm",
        "0.7",
        "--max-nonfinite-grad-steps",
        "4",
        "--model-ema",
        "--model-ema-decay",
        "0.995",
        "--num-workers",
        str(max(0, int(num_workers))),
        "--eval-num-workers",
        str(max(0, int(eval_num_workers))),
        "--train-image-cache-mb",
        "0",
        "--eval-image-cache-mb",
        "0",
        "--seed",
        str(int(seed)),
        "--disable-class-weights",
        "--balanced-epoch-multiplier",
        "1.0",
        "--balanced-epoch-tolerance",
        "0.10",
        "--classification-loss",
        "ldam_focal",
        "--focal-loss-gamma",
        "1.0",
        "--focal-loss-mix",
        "0.1",
        "--label-smoothing",
        "0.02",
        "--ldam-max-margin",
        "0.3",
        "--ldam-scale",
        "18",
        "--best-metric",
        "fair_macro_f1",
        "--fair-f1-gap-target",
        "0.05",
        "--fair-f1-gap-penalty",
        "1.5",
        "--fair-f1-min-weight",
        "0.25",
        "--resize-mode",
        "pad",
        "--train-scale-min",
        "0.90",
        "--train-scale-crop-probability",
        "0.30",
        "--brightness",
        "0.05",
        "--contrast",
        "0.05",
        "--saturation",
        "0.03",
        "--hue",
        "0.01",
        "--local-exposure-probability",
        "0.15",
        "--local-exposure-strength",
        "0.20",
        "--obstacle-probability",
        "0.03",
        "--obstacle-max-area",
        "0.06",
        "--random-erasing-probability",
        "0",
        "--random-affine-degrees",
        "3",
        "--random-affine-translate",
        "0.02",
        "--random-affine-scale-min",
        "0.96",
        "--horizontal-flip-probability",
        "0.5",
        "--vertical-flip-probability",
        "0",
        "--rotate90-probability",
        "0.03",
        "--lighting-probability",
        "0",
        "--disable-class-aware-augmentation",
        "--disable-rare-class-repeat",
        "--batch-mix-probability",
        "0",
        "--mosaic-probability",
        "0",
        "--mixup-probability",
        "0",
        "--cutmix-probability",
        "0",
        "--copy-paste-probability",
        "0",
        "--targeted-copy-paste-probability",
        "0",
        "--no-pretrained-distillation",
        "--distillation-weight",
        "0",
        "--teacher-focus-binary-loss-weight",
        "0",
        "--teacher-focus-margin-loss-weight",
        "0",
        "--teacher-pairwise-margin-loss-weight",
        "0",
        "--max-train-batches",
        str(spec.max_train_batches),
        "--max-val-batches",
        str(spec.max_val_batches),
        "--skip-final-test",
    ]
    args.extend(
        [
            "--experiment-protocol-id",
            PROTOCOL_ID,
            "--source-commit",
            str(source_commit).strip().lower(),
            "--source-tree-sha256",
            str(source_tree_sha256).strip().lower(),
            "--dataset-image-tree-sha256",
            str(dataset_image_tree_sha256).strip().lower(),
        ]
    )
    train_contract_sha256 = hashlib.sha256(
        json.dumps(
            args,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    args.extend(
        [
            "--recipe-train-contract-sha256",
            train_contract_sha256,
        ]
    )
    if auto_resume:
        args.append("--auto-resume")
    return args


def _git_snapshot(repo_root: Path) -> Dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "status_short": status,
            "clean": not status,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "available": False,
            "commit": "",
            "branch": "",
            "status_short": [],
            "clean": False,
        }


def _preflight_model(train_args: Sequence[str]) -> Dict[str, object]:
    parsed = parse_train_args(train_args)
    model_config, _, _ = build_configs(parsed)
    model = create_model(num_classes=5, model_config=model_config)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "DINOv3 five-class parameter count drifted: "
            f"{parameter_count} != {EXPECTED_PARAMETER_COUNT}."
        )
    groups = build_optimizer_param_groups(
        model,
        weight_decay=0.05,
        learning_rate=1.5e-4,
        backbone_lr_scale=0.1,
    )
    group_names = sorted(str(group["name"]) for group in groups)
    weight_decay_by_parameter_id = {
        id(parameter): float(group["weight_decay"])
        for group in groups
        for parameter in group["params"]
    }
    declared_no_decay = (
        set(str(name) for name in model.no_weight_decay())
        if callable(getattr(model, "no_weight_decay", None))
        else set()
    )
    no_decay_violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name in declared_no_decay or parameter.ndim <= 1)
        and weight_decay_by_parameter_id.get(id(parameter), float("inf")) != 0.0
    ]
    if no_decay_violations:
        raise RuntimeError(
            "DINO/timm no-weight-decay contract is inactive: "
            f"{no_decay_violations}."
        )
    backbone_lrs = sorted(
        {
            float(group["lr"])
            for group in groups
            if str(group["name"]).startswith("backbone_") and "lr" in group
        }
    )
    if (
        len(backbone_lrs) != 1
        or not math.isclose(backbone_lrs[0], 1.5e-5, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Pretrained backbone LR split is inactive: {backbone_lrs}.")
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 256, 256))
    if not torch.is_tensor(logits) or tuple(logits.shape) != (1, 5):
        raise RuntimeError(f"Unexpected DINOv3 output shape: {getattr(logits, 'shape', None)}.")
    provenance = getattr(model, "pretrained_provenance", {})
    observed_sha = (
        provenance.get("initialization", {})
        .get("checkpoint", {})
        .get("sha256", "")
        if isinstance(provenance, Mapping)
        else ""
    )
    if observed_sha != DINO_SHA256:
        raise RuntimeError(
            f"Model did not consume the locked DINO file: {observed_sha!r}."
        )
    return {
        "parameter_count": parameter_count,
        "optimizer_groups": group_names,
        "declared_no_decay_parameters": sorted(declared_no_decay),
        "no_decay_violations": no_decay_violations,
        "backbone_learning_rates": backbone_lrs,
        "output_shape": list(logits.shape),
        "pretrained_checkpoint_sha256": observed_sha,
    }


def _best_history_row(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.csv"
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
        "best_macro_f1": summary.get("best_macro_f1"),
        "best_class1_f1": selected.get("val_class_1_f1"),
        "best_class1_precision": selected.get("val_class_1_precision"),
        "best_class1_recall": selected.get("val_class_1_recall"),
        "stop_reason": summary.get("stop_reason"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clean DINOv3 direct baseline on canonical class_f."
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "probe", "full"),
        default="preflight",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--canonical-attestation",
        type=Path,
        default=None,
        help=(
            "Optional JSON produced by validate_canonical_classf earlier in the "
            "same immutable runtime; avoids rehashing the 1 GiB image tree."
        ),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help=(
            "Optional test-locked YAML for train/val. --data remains the "
            "fingerprinted canonical contract."
        ),
    )
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-tag", type=str, default="20260731")
    parser.add_argument("--batch-size", type=int, choices=sorted(ACCUMULATION_BY_BATCH), default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument("--auto-resume", action="store_true", default=False)
    parser.add_argument("--confirm-full", action="store_true", default=False)
    parser.add_argument(
        "--source-commit",
        default="",
        help="Required 40-hex source commit for probe/full archive execution.",
    )
    parser.add_argument(
        "--source-tree-sha256",
        default="",
        help="Required source-tree digest for probe/full archive execution.",
    )
    parser.add_argument(
        "--skip-focused-tests",
        action="store_true",
        default=False,
        help="Engineering-only; preflight manifest records the skipped tests.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    data_yaml = args.data.resolve()
    training_data_yaml = (
        args.train_data.resolve() if args.train_data is not None else data_yaml
    )
    dino_checkpoint = _absolute_path_without_resolving_symlink(
        args.dino_checkpoint
    )
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
    development_contract = None
    if training_data_yaml != data_yaml:
        development_contract = validate_development_data_yaml(
            training_data_yaml,
            canonical_contract=dataset_contract,
        )
    git = _git_snapshot(repo_root)
    observed_source_tree_sha256 = source_tree_sha256(repo_root)
    archive_source: Dict[str, object] = {
        "commit": str(args.source_commit).strip().lower(),
        "tree_sha256": observed_source_tree_sha256,
        "expected_tree_sha256": str(args.source_tree_sha256).strip().lower(),
    }
    if git["available"]:
        branch = str(git["branch"])
        if not branch.startswith("research/pretrained-"):
            raise RuntimeError(
                f"B0 requires research/pretrained-* branch, observed={branch!r}."
            )
        if args.mode in {"probe", "full"} and not bool(git["clean"]):
            raise RuntimeError(f"{args.mode} requires a clean worktree.")
    elif args.mode in {"probe", "full"}:
        source_commit = str(args.source_commit).strip().lower()
        expected_tree_sha256 = str(args.source_tree_sha256).strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise RuntimeError(
                f"{args.mode} archive execution requires --source-commit as 40 hex."
            )
        if re.fullmatch(r"[0-9a-f]{64}", expected_tree_sha256) is None:
            raise RuntimeError(
                f"{args.mode} archive execution requires --source-tree-sha256 as 64 hex."
            )
        if expected_tree_sha256 != observed_source_tree_sha256:
            raise RuntimeError(
                "Archive source-tree SHA-256 mismatch: "
                f"observed={observed_source_tree_sha256}, "
                f"expected={expected_tree_sha256}."
            )
    if args.mode == "full" and not bool(args.confirm_full):
        raise RuntimeError("Full requires --confirm-full after reviewing the probe.")

    stage = "smoke" if args.mode == "preflight" else args.mode
    train_args = build_train_args(
        data_yaml=training_data_yaml,
        dino_checkpoint=dino_checkpoint,
        output_dir=output_dir,
        stage=stage,
        run_tag=args.run_tag,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_num_workers=args.eval_num_workers,
        seed=args.seed,
        auto_resume=bool(args.auto_resume),
        source_commit=(
            str(git.get("commit", "")).strip().lower()
            if bool(git.get("available"))
            else str(args.source_commit).strip().lower()
        ),
        source_tree_sha256=observed_source_tree_sha256,
        dataset_image_tree_sha256=str(
            dataset_contract["image_tree_sha256"]
        ).lower(),
    )

    focused_tests = [
        repo_root / "tests" / "test_pretrained_semantic_branch.py",
        repo_root / "tests" / "test_timm_classifier_model.py",
        repo_root / "tests" / "test_canonical_classf_defaults.py",
        repo_root / "tests" / "test_deploy_classification_folder.py",
        repo_root / "tests" / "test_pretrained_classf_recipe.py",
        repo_root / "tests" / "test_resume_weight_and_distillation_source.py",
        repo_root / "tests" / "test_attention_viz_headless.py",
        repo_root / "tests" / "test_mobile_onnx_quantization.py",
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
            raise RuntimeError("Focused B0 tests failed.")

    model_preflight = _preflight_model(train_args)
    preflight_dir = output_dir / f"preflight_classf_b0_{args.run_tag}"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "test_locked": True,
        "old_dataset_trained_weights_used": False,
        "old_teacher_cache_used": False,
        "dataset": dataset_contract,
        "development_data": development_contract,
        "dino": {
            "model_name": DINO_MODEL_NAME,
            "checkpoint": str(dino_checkpoint),
            "sha256": dino_sha,
            "source_url": DINO_SOURCE_URL,
            "source_revision": DINO_SOURCE_REVISION,
            "source_license": DINO_SOURCE_LICENSE,
        },
        "git": git,
        "archive_source": archive_source,
        "runtime": {
            "python": str(args.python.resolve()),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp_dtype_request": args.amp_dtype,
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(ACCUMULATION_BY_BATCH[args.batch_size]),
            "effective_batch_size": 48,
        },
        "focused_tests": test_result,
        "model_preflight": model_preflight,
        "train_args": train_args,
    }
    manifest_path = preflight_dir / f"{args.mode}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"B0 preflight passed: {manifest_path}", flush=True)
    if args.mode == "preflight":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for smoke/probe/full training.")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["TRKH_AMP_DTYPE"] = str(args.amp_dtype)
    environment.setdefault("OMP_NUM_THREADS", "4")
    command = [str(args.python.resolve()), "-m", "trkh.training.train", *train_args]
    run_dir = output_dir / run_name(stage, args.run_tag)
    if run_dir.exists() and any(run_dir.iterdir()) and not bool(args.auto_resume):
        raise RuntimeError(
            f"Run directory already exists: {run_dir}. "
            "Use a new --run-tag or explicitly pass --auto-resume."
        )
    if bool(args.auto_resume) and not (run_dir / "checkpoints" / "last.pt").is_file():
        raise RuntimeError(
            f"--auto-resume requires {run_dir / 'checkpoints' / 'last.pt'}."
        )
    resume_contract = None
    if bool(args.auto_resume):
        resume_contract = validate_auto_resume_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            training_data_yaml=training_data_yaml,
            expected_train_args=train_args,
        )
        manifest["auto_resume_checkpoint"] = resume_contract
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
        "git_commit": git.get("commit"),
        "metrics": _best_history_row(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / "b0_stage_complete.json"
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"B0 {stage} failed with code {completed.returncode}; evidence: {run_dir}."
        )
    print(f"B0 {stage} completed: {marker_path}", flush=True)


if __name__ == "__main__":
    main()

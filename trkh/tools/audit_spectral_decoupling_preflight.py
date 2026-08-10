from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch

from trkh.core.utils import json_dump
from trkh.training.losses import (
    FocalCrossEntropyLoss,
    SpectralDecouplingCrossEntropyLoss,
)
from trkh.training.train import build_configs, parse_args as parse_train_args


METHOD = "spectral_decoupling_multiclass_mean_lambda_half"
PROTOCOL_SHA256 = "fe7c14ee923f2b60ac5744831a4524ab721c61ba5f19f7745c5daa13830ac576"
SOURCE_ARGS_SHA256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _argument_value(arguments: Sequence[str], name: str) -> Optional[str]:
    try:
        index = list(arguments).index(name)
    except ValueError:
        return None
    if index + 1 >= len(arguments) or str(arguments[index + 1]).startswith("--"):
        raise ValueError(f"Argument is missing its value: {name}")
    return str(arguments[index + 1])


def _equation_checks() -> Dict[str, bool]:
    torch.manual_seed(20260720)
    regularization_lambda = 0.01
    logits = torch.tensor(
        [[2.0, -0.5, 0.25], [-1.0, 1.5, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1])
    spectral = SpectralDecouplingCrossEntropyLoss(
        regularization_lambda=regularization_lambda
    )
    ordinary = FocalCrossEntropyLoss(gamma=0.0, focal_mix=0.0)
    observed = spectral.per_sample_loss(logits, targets)
    expected = ordinary.per_sample_loss(logits, targets) + (
        0.5 * regularization_lambda * logits.square().mean(dim=1)
    )
    observed.mean().backward()

    soft_logits = torch.randn(5, 3, requires_grad=True)
    soft_targets = torch.softmax(torch.randn(5, 3), dim=1)
    soft_loss = SpectralDecouplingCrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5, 0.75]),
        label_smoothing=0.02,
        regularization_lambda=regularization_lambda,
    )
    soft_loss.set_class_weight_multipliers(torch.tensor([0.9, 1.1, 1.0]))
    soft_value = soft_loss(soft_logits, soft_targets)
    soft_value.backward()

    zero_logits = torch.randn(7, 5)
    zero_targets = torch.randint(0, 5, (7,))
    zero_spectral = SpectralDecouplingCrossEntropyLoss(
        regularization_lambda=0.0
    ).per_sample_loss(zero_logits, zero_targets)
    zero_ordinary = ordinary.per_sample_loss(zero_logits, zero_targets)

    negative_rejected = False
    try:
        SpectralDecouplingCrossEntropyLoss(regularization_lambda=-1e-6)
    except ValueError:
        negative_rejected = True

    invalid_shape_rejected = False
    try:
        spectral(torch.randn(5), torch.tensor([0]))
    except ValueError:
        invalid_shape_rejected = True

    return {
        "hard_target_equation_exact": bool(torch.equal(observed, expected)),
        "hard_target_gradient_finite": bool(
            logits.grad is not None and torch.isfinite(logits.grad).all()
        ),
        "soft_target_gradient_finite": bool(
            torch.isfinite(soft_value)
            and soft_logits.grad is not None
            and torch.isfinite(soft_logits.grad).all()
        ),
        "lambda_zero_exact_ce": bool(torch.equal(zero_spectral, zero_ordinary)),
        "negative_lambda_rejected": negative_rejected,
        "invalid_logit_shape_rejected": invalid_shape_rejected,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only Spectral Decoupling equation/configuration preflight."
    )
    parser.add_argument(
        "--source-launcher-args",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_SPECTRAL_DECOUPLING_A0_PROTOCOL_20260720.md"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = _sha256(args.source_launcher_args)
    protocol_sha256 = _sha256(args.protocol)
    source = _load_json(args.source_launcher_args)
    source_arguments = [str(value) for value in source.get("train_args", [])]

    cli_args = parse_train_args(
        [
            "--classification-loss",
            "spectral_decoupling",
            "--spectral-decoupling-lambda",
            "0.01",
            "--weight-decay",
            "0",
            "--label-smoothing",
            "0",
        ]
    )
    _, train_config, _ = build_configs(cli_args)

    train_source = Path("trkh/training/train.py").read_text(encoding="utf-8")
    branch_start = train_source.index(
        'elif classification_loss_name == "spectral_decoupling":'
    )
    branch_end = train_source.index("elif classification_loss_name", branch_start + 1)
    branch_source = train_source[branch_start:branch_end]
    launcher_source = Path(
        "scripts/run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")

    checks = {
        **_equation_checks(),
        "protocol_hash_locked": protocol_sha256 == PROTOCOL_SHA256,
        "source_args_hash_locked": source_sha256 == SOURCE_ARGS_SHA256,
        "source_is_no_pretrain": (
            "--no-pretrained" in source_arguments
            and "--no-pretrained-distillation" in source_arguments
            and "--pretrained" not in source_arguments
        ),
        "source_skips_final_test": "--skip-final-test" in source_arguments,
        "source_has_no_resume": (
            "--resume" not in source_arguments
            and "--disable-resume" in source_arguments
        ),
        "source_dataset_is_locked_yolof": (
            _argument_value(source_arguments, "--data")
            == r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
        ),
        "cli_round_trip": (
            train_config.classification_loss == "spectral_decoupling"
            and train_config.spectral_decoupling_lambda == 0.01
            and train_config.weight_decay == 0.0
            and train_config.label_smoothing == 0.0
        ),
        "train_factory_uses_sd": (
            "SpectralDecouplingCrossEntropyLoss" in branch_source
            and "regularization_lambda=train_config.spectral_decoupling_lambda"
            in branch_source
        ),
        "eval_factory_uses_ordinary_ce": (
            "eval_criterion = FocalCrossEntropyLoss" in branch_source
            and "gamma=0.0" in branch_source
            and "focal_mix=0.0" in branch_source
            and "LogitNormCrossEntropyLoss" not in branch_source
        ),
        "v8_launcher_round_trip_declared": (
            '"spectral_decoupling"' in launcher_source
            and "$SpectralDecouplingLambda = 0.01" in launcher_source
            and '"--spectral-decoupling-lambda", "$SpectralDecouplingLambda"'
            in launcher_source
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    summary = {
        "method": METHOD,
        "stage": "A_equation_and_integration_preflight",
        "sources": {
            "protocol": str(Path(args.protocol).resolve()),
            "protocol_sha256": protocol_sha256,
            "source_launcher_args": str(Path(args.source_launcher_args).resolve()),
            "source_launcher_args_sha256": source_sha256,
            "validation_loaded": False,
            "test_loaded": False,
            "raw_dataset_modified": False,
        },
        "objective": {
            "classification_loss": "spectral_decoupling",
            "spectral_decoupling_lambda": 0.01,
            "formula": "CE + (lambda / 2) * mean_class(logit^2)",
            "validation_loss": "ordinary_cross_entropy",
        },
        "gate": {
            "checks": checks,
            "failed_checks": failed,
            "smoke_permission": not failed,
            "test_permission": False,
            "full_train_permission": False,
        },
    }
    json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from trkh.tools import run_b23_b9_guarded_convpass as b23
from trkh.tools import run_b29_efficientvim_teacher_residual_fold as b29
from trkh.tools import run_b32_full_train_matched_validation as b32


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B33_B9_M1_RESIDUAL_FUSION_VALIDATION_20260805"
B9_TRAIN_LOGITS_SHA256 = "643cd63059df171a48770f7f7e9b15ccbc9df7001d09da2bb1200db4048d2f22"
B32_SUMMARY_SHA256 = "6e689d825c187453892487cf97ffe78ddc36937a0d6c85ab6ef384a3e77a2f84"
B32_TRAIN_RESIDUAL_SHA256 = "490f07f210870312c7137ba9a883f23de150fcb33a9765f10cf04b81f3482062"
B32_VALIDATION_SCORES_SHA256 = "958d9d6e7471d58f0bc2ede99166c45da12836659e0d30476f0f0c1d2b8965d5"


class B33ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B33 B9 + distilled-M1 residual validation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--b24-dir", type=Path, required=True)
    parser.add_argument("--b9-checkpoint", type=Path, required=True)
    parser.add_argument("--b9-train-logits", type=Path, required=True)
    parser.add_argument("--b32-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _source_hashes() -> dict[str, str]:
    root = b29._repo()
    paths = {
        "runner": Path("trkh/tools/run_b33_b9_m1_residual_fusion_validation.py"),
        "b9_loader": Path("trkh/tools/run_b23_b9_guarded_convpass.py"),
        "readout": Path("trkh/tools/run_b32_full_train_matched_validation.py"),
        "metric": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
    }
    for path in paths.values():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0 or not (root / path).is_file():
            raise B33ContractError(f"B33 source missing or untracked: {path}")
    return {name: b29._sha256(root / path) for name, path in paths.items()}


def _load_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    b9_logits_path = args.b9_train_logits.expanduser().resolve()
    b32_dir = args.b32_dir.expanduser().resolve()
    paths = {
        "b9_train_logits": b9_logits_path,
        "b32_summary": b32_dir / "summary.json",
        "train_residual": b32_dir / "train_m1_residual_f32.npy",
        "validation_scores": b32_dir / "validation_scores.npz",
    }
    expected = {
        "b9_train_logits": B9_TRAIN_LOGITS_SHA256,
        "b32_summary": B32_SUMMARY_SHA256,
        "train_residual": B32_TRAIN_RESIDUAL_SHA256,
        "validation_scores": B32_VALIDATION_SCORES_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or b29._sha256(path) != expected[name]:
            raise B33ContractError(f"B33 retained artifact changed: {name}")
    summary = json.loads(paths["b32_summary"].read_text(encoding="utf-8"))
    if summary.get("integrity_complete") is not True:
        raise B33ContractError("B33 B32 integrity contract failed")
    b9_logits = np.load(paths["b9_train_logits"], allow_pickle=False)
    residual = np.load(paths["train_residual"], allow_pickle=False)
    with np.load(paths["validation_scores"], allow_pickle=False) as payload:
        validation = {name: np.asarray(payload[name]) for name in payload.files}
    if b9_logits.shape != residual.shape or b9_logits.shape != (8278, 5):
        raise B33ContractError("B33 TRAIN feature shapes changed")
    return {
        "paths": paths,
        "b9_logits": np.asarray(b9_logits, dtype=np.float32),
        "residual": np.asarray(residual, dtype=np.float32),
        "validation": validation,
        "b32_summary": summary,
    }


def _screen(
    b9_logits: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    control_scores = np.full((labels.size, 5), np.nan, np.float32)
    candidate_scores = np.full_like(control_scores, np.nan)
    iterations = []
    for fold in range(5):
        fit = folds != fold
        control = b29._fit_readout(b9_logits, labels, fit)
        candidate = b29._fit_readout(
            np.concatenate((b9_logits, residual), axis=1), labels, fit
        )
        control_scores[~fit] = control["scores"][~fit]
        candidate_scores[~fit] = candidate["scores"][~fit]
        iterations.append(
            {
                "fold": fold,
                "control": control["iterations"],
                "candidate": candidate["iterations"],
            }
        )
    metrics = {
        "exact_b9": b29.classification_summary(labels, b9_logits),
        "b9_readout_control": b29.classification_summary(labels, control_scores),
        "b9_m1_candidate": b29.classification_summary(labels, candidate_scores),
    }
    checks = {
        "candidate_beats_exact_b9_class1": float(metrics["b9_m1_candidate"]["class1_f1"])
        >= float(metrics["exact_b9"]["class1_f1"]) + 0.020,
        "candidate_beats_control_class1": float(metrics["b9_m1_candidate"]["class1_f1"])
        >= float(metrics["b9_readout_control"]["class1_f1"]) + 0.020,
        "candidate_macro_gain": float(metrics["b9_m1_candidate"]["macro_f1"])
        >= float(metrics["exact_b9"]["macro_f1"]) + 0.010,
        "candidate_tp_gain": int(metrics["b9_m1_candidate"]["class1_tp"])
        >= int(metrics["exact_b9"]["class1_tp"]),
        "candidate_fp_reduction": int(metrics["b9_m1_candidate"]["restricted_fp"])
        < int(metrics["exact_b9"]["restricted_fp"]),
    }
    if not all(checks.values()):
        raise B33ContractError(f"B33 TRAIN screen did not reproduce: {checks}")
    return {
        "metrics": metrics,
        "checks": checks,
        "iterations": iterations,
        "control_scores": control_scores,
        "candidate_scores": candidate_scores,
    }


def _gate(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b9 = metrics["exact_b9"]
    control = metrics["b9_readout_control"]
    candidate = metrics["b9_m1_fusion"]
    deltas = {
        "accuracy_vs_b9": float(candidate["accuracy"]) - float(b9["accuracy"]),
        "macro_vs_b9": float(candidate["macro_f1"]) - float(b9["macro_f1"]),
        "class1_vs_b9": float(candidate["class1_f1"]) - float(b9["class1_f1"]),
        "class1_vs_control": float(candidate["class1_f1"])
        - float(control["class1_f1"]),
        "tp_vs_b9": int(candidate["class1_tp"]) - int(b9["class1_tp"]),
        "restricted_fp_vs_b9": int(candidate["restricted_fp"])
        - int(b9["restricted_fp"]),
    }
    checks = {
        "class1_milestone": float(candidate["class1_f1"]) >= 0.72,
        "class1_gain_over_b9": deltas["class1_vs_b9"] >= 0.005,
        "class1_gain_over_control": deltas["class1_vs_control"] >= 0.010,
        "tp_noninferiority": deltas["tp_vs_b9"] >= 0,
        "fp_nonincrease": deltas["restricted_fp_vs_b9"] <= 0,
        "accuracy_noninferiority": deltas["accuracy_vs_b9"] >= -0.002,
        "macro_noninferiority": deltas["macro_vs_b9"] >= -0.002,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "audit_permission": passed,
        "test_permission": False,
        "finalfit_permission": False,
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "deltas": deltas,
    }


def _formal(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B33 output: {output}")
    source_start, git_start = _source_hashes(), b29._git_contract()
    if git_start["branch"] != "research/pretrained-classf-b1" or not git_start["tracked_worktree_clean"]:
        raise B33ContractError(f"B33 requires a clean pretrained branch: {git_start}")
    retained = b29._load_b25(args)
    labels = np.asarray(retained["arrays"]["labels"], dtype=np.int64)
    folds = np.asarray(retained["arrays"]["folds"], dtype=np.int64)
    artifacts = _load_artifacts(args)
    screen = _screen(artifacts["b9_logits"], artifacts["residual"], labels, folds)
    control_readout = b32._fit_readout(artifacts["b9_logits"], labels)
    candidate_readout = b32._fit_readout(
        np.concatenate((artifacts["b9_logits"], artifacts["residual"]), axis=1), labels
    )
    torch.set_num_threads(int(args.torch_threads))
    b23._configure_determinism()
    device = b29._resolve_device(args.device)
    _checkpoint, b9_model, b9_contract = b23._load_b9(args.b9_checkpoint)
    _train_dataset, val_dataset = b23._datasets(args.data, include_validation=True)
    if val_dataset is None:
        raise B33ContractError("B33 validation dataset was not constructed")
    evaluation = b23._evaluate(b9_model.to(device), b23._eval_loader(val_dataset), device=device)
    val_labels = np.asarray(evaluation["labels"], dtype=np.int64)
    b9_logits = np.asarray(evaluation["logits"], dtype=np.float32)
    b9_predictions = np.asarray(evaluation["predictions"], dtype=np.int64)
    validation = artifacts["validation"]
    if (
        not np.array_equal(val_labels, validation["labels"])
        or not np.array_equal(b9_predictions, validation["b9_predictions"])
        or b9_logits.shape != (2479, 5)
    ):
        raise B33ContractError("B33 direct B9 replay changed")
    val_residual = np.asarray(
        validation["hybrid_scores"] - validation["raw_dino_primary_scores"],
        dtype=np.float32,
    )
    control_scores = b32._apply_readout(control_readout, b9_logits)
    fusion_scores = b32._apply_readout(
        candidate_readout, np.concatenate((b9_logits, val_residual), axis=1)
    )
    metrics = {
        "exact_b9": b29.classification_summary(val_labels, b9_logits),
        "b9_readout_control": b29.classification_summary(val_labels, control_scores),
        "b9_m1_fusion": b29.classification_summary(val_labels, fusion_scores),
    }
    gate = _gate(metrics)
    output.mkdir(parents=True, exist_ok=False)
    readout_sha = b29._atomic_npz(
        output / "b33_fusion_readouts.npz",
        control_mean=control_readout["mean"],
        control_scale=control_readout["scale"],
        control_coef=control_readout["coef"],
        control_intercept=control_readout["intercept"],
        fusion_mean=candidate_readout["mean"],
        fusion_scale=candidate_readout["scale"],
        fusion_coef=candidate_readout["coef"],
        fusion_intercept=candidate_readout["intercept"],
    )
    score_sha = b29._atomic_npz(
        output / "validation_scores.npz",
        labels=val_labels,
        exact_b9_logits=b9_logits,
        m1_residual=val_residual,
        b9_readout_control_scores=control_scores,
        b9_m1_fusion_scores=fusion_scores,
    )
    oof_sha = b29._atomic_npz(
        output / "train_oof_scores.npz",
        labels=labels,
        folds=folds,
        exact_b9_logits=artifacts["b9_logits"],
        b9_readout_control_scores=screen["control_scores"],
        b9_m1_candidate_scores=screen["candidate_scores"],
    )
    with np.load(output / "b33_fusion_readouts.npz", allow_pickle=False) as saved:
        reload_readout = {
            "mean": saved["fusion_mean"],
            "scale": saved["fusion_scale"],
            "coef": saved["fusion_coef"],
            "intercept": saved["fusion_intercept"],
        }
    reload_scores = b32._apply_readout(
        reload_readout, np.concatenate((b9_logits, val_residual), axis=1)
    )
    reload_error = float(np.max(np.abs(reload_scores - fusion_scores)))
    source_end, git_end = _source_hashes(), b29._git_contract()
    integrity = source_start == source_end and git_start == git_end and reload_error == 0.0
    if not integrity:
        raise B33ContractError("B33 source/Git/readout reload integrity failed")
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "train_screen_then_design_exposed_validation",
        "source_hashes": source_start,
        "git": {"start": git_start, "end": git_end},
        "dataset": {
            "train_rows": int(labels.size),
            "validation_rows": int(val_labels.size),
            "train_split_used": True,
            "validation_split_used": True,
            "test_split_used": False,
        },
        "b9": b9_contract,
        "b32_state_sha256": artifacts["b32_summary"]["student"]["state_sha256"],
        "train_screen": {
            "metrics": screen["metrics"],
            "checks": screen["checks"],
            "iterations": screen["iterations"],
            "oof_scores_sha256": oof_sha,
        },
        "full_readout_iterations": {
            "control": control_readout["iterations"],
            "candidate": candidate_readout["iterations"],
        },
        "metrics": metrics,
        "gate": gate,
        "deployment": {
            "inference_backbones": ["supervised_B9_DINOv3_S16", "distilled_EfficientViM_M1"],
            "raw_dino_teacher_present": False,
            "convnext_teacher_present": False,
            "total_parameters_with_readout": 21_588_869 + 5_720_278 + 55,
            "readout_sha256": readout_sha,
            "readout_reload_max_abs": reload_error,
        },
        "validation_scores_sha256": score_sha,
        "integrity_complete": integrity,
    }
    b29._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    summary = _formal(args)
    print(
        json.dumps(
            {"gate": summary["gate"], "output": str(args.output_dir.resolve())}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

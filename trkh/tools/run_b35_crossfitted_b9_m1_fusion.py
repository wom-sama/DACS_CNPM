from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Subset

from trkh.data.dataset import ClassificationFolderDataset, build_eval_transform
from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.models.efficientvim_residual_student_b29 import load_student_state_dict
from trkh.tools import diagnose_b22_recall_monotonic_expert as b22
from trkh.tools import run_b29_efficientvim_teacher_residual_fold as b29
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (
    OFFICIAL_M1_E450_SHA256,
    _TrainLedgerDataset,
    _efficientvim_transform,
    classification_summary,
)
from trkh.tools.screen_b34_crossfitted_supervised_m1_fusion import (
    _atomic_npz,
    _extract_m1_residual,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B35_CROSSFITTED_B9_M1_FUSION_20260805"
FOLD = 1
SEED = 42
EPOCHS = 9
SCHEDULER_HORIZON = 30
EXPECTED_M1_STATE_SHA256 = (
    "71ace9b7ada0035af914fc1cbb2265085ae52e6a5cc9d1553231cbdbd154e357"
)
ROWS = 8_278
CLASSES = 5


class B35ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--m1-state", type=Path, required=True)
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b35_crossfitted_b9_m1_fusion.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "b21_trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "b29_student": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
        "b34_fusion": Path("trkh/tools/screen_b34_crossfitted_supervised_m1_fusion.py"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _eval_subsets(
    train_root: Path, assignment: Mapping[str, Any]
) -> tuple[Subset, Subset]:
    dataset = ClassificationFolderDataset(
        train_root,
        b21.EXPECTED_CLASS_NAMES,
        transform=build_eval_transform(image_size=256, resize_mode="pad"),
        split="train_b35_eval",
    )
    mapped = b21.map_dataset_indices(
        dataset,
        data_root=train_root.parent,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=assignment["labels"],
    )
    folds = assignment["folds"]
    fit_positions = np.flatnonzero(folds != FOLD)
    held_positions = np.flatnonzero(folds == FOLD)
    return (
        Subset(dataset, [mapped[index] for index in fit_positions]),
        Subset(dataset, [mapped[index] for index in held_positions]),
    )


def _gate(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary = metrics["b9_like_primary"]
    control = metrics["primary_readout_control"]
    candidate = metrics["primary_m1_fusion"]
    deltas = {
        "class1_f1_vs_primary": float(candidate["class1_f1"])
        - float(primary["class1_f1"]),
        "class1_f1_vs_control": float(candidate["class1_f1"])
        - float(control["class1_f1"]),
        "macro_f1_vs_control": float(candidate["macro_f1"])
        - float(control["macro_f1"]),
        "accuracy_vs_control": float(candidate["accuracy"])
        - float(control["accuracy"]),
        "class1_tp_vs_primary": int(candidate["class1_tp"])
        - int(primary["class1_tp"]),
        "restricted_fp_vs_control": int(candidate["restricted_fp"])
        - int(control["restricted_fp"]),
        "two_to_one_vs_control": int(candidate["confusion_matrix"][2][1])
        - int(control["confusion_matrix"][2][1]),
        "pair_auroc_vs_primary": float(candidate["mean_pair_auroc"])
        - float(primary["mean_pair_auroc"]),
        "pair_auroc_vs_control": float(candidate["mean_pair_auroc"])
        - float(control["mean_pair_auroc"]),
    }
    checks = {
        "class1_gain_vs_primary": deltas["class1_f1_vs_primary"] >= 0.005,
        "class1_gain_vs_control": deltas["class1_f1_vs_control"] >= 0.010,
        "macro_noninferiority": deltas["macro_f1_vs_control"] >= -0.002,
        "accuracy_noninferiority": deltas["accuracy_vs_control"] >= -0.002,
        "class1_tp_noninferiority": deltas["class1_tp_vs_primary"] >= 0,
        "restricted_fp_nonincrease": deltas["restricted_fp_vs_control"] <= 0,
        "two_to_one_nonincrease": deltas["two_to_one_vs_control"] <= 0,
        "pair_auroc_vs_primary": deltas["pair_auroc_vs_primary"] >= -0.005,
        "pair_auroc_vs_control": deltas["pair_auroc_vs_control"] >= -0.005,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
        "deltas": deltas,
        "next_permission": (
            "complete_crossfitted_primary_m1_oof"
            if passed
            else "do_not_scale_current_linear_fusion"
        ),
        "validation_permission": False,
        "test_permission": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise B35ContractError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    fit = assignment["folds"] != FOLD
    held = ~fit
    train_fit, train_held, held_paths, fit_labels_list = b21._build_datasets(
        train_root, assignment, fold=FOLD
    )
    dino_weight = b21.logical_absolute_path(args.dino_weight)

    primary_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels_list,
        output_dir=output,
        epochs=EPOCHS,
        scheduler_horizon=SCHEDULER_HORIZON,
        seed=SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b35_primary_ema",
    )
    primary_checkpoint = primary_training["checkpoint"]
    if primary_checkpoint is None:
        raise B35ContractError("B35 primary checkpoint was not saved")

    fit_eval, held_eval = _eval_subsets(train_root, assignment)
    primary_model = b21.build_arm_model(dino_weight, DIRECT_MODE)
    primary_model.set_mode(DIRECT_MODE)
    primary_model.load_state_dict(
        load_file(primary_checkpoint["path"], device="cpu"), strict=True
    )
    if b21.state_sha256(primary_model.state_dict()) != primary_checkpoint["state_sha256"]:
        raise B35ContractError("B35 primary state changed after strict reload")
    device = torch.device("cuda")
    primary_model.to(device).eval()
    _fit_features, fit_logits, fit_labels = b22._extract_frozen(
        primary_model, fit_eval, device=device, loader_seed=SEED
    )
    _held_features, held_logits, held_labels = b22._extract_frozen(
        primary_model, held_eval, device=device, loader_seed=SEED + 1
    )
    replay_max_abs = float(
        np.max(np.abs(held_logits.numpy() - primary_training["held_logits"]))
    )
    if replay_max_abs != 0.0:
        raise B35ContractError(f"B35 primary replay changed: {replay_max_abs}")
    del primary_model, _fit_features, _held_features
    gc.collect()
    torch.cuda.empty_cache()

    labels = assignment["labels"]
    primary_logits = np.full((ROWS, CLASSES), np.nan, dtype=np.float32)
    primary_logits[fit] = fit_logits.numpy()
    primary_logits[held] = held_logits.numpy()
    if not np.array_equal(fit_labels.numpy(), labels[fit]) or not np.array_equal(
        held_labels.numpy(), labels[held]
    ):
        raise B35ContractError("Primary extraction labels changed")

    efficientvim_path = args.efficientvim_checkpoint.expanduser().resolve(strict=True)
    m1_state_path = args.m1_state.expanduser().resolve(strict=True)
    if b21.sha256_file(efficientvim_path) != OFFICIAL_M1_E450_SHA256:
        raise B35ContractError("Official EfficientViM-M1 checkpoint changed")
    if b21.sha256_file(m1_state_path) != EXPECTED_M1_STATE_SHA256:
        raise B35ContractError("B30 coupled-KL state changed")
    m1_model, _digest = b29._build_student(efficientvim_path)
    load_student_state_dict(m1_model, load_file(str(m1_state_path), device="cpu"))
    m1_dataset = _TrainLedgerDataset(
        [train_root.parent / path for path in assignment["relative_paths"]],
        labels,
        _efficientvim_transform(),
    )
    fit_indices = np.flatnonzero(fit)
    held_indices = np.flatnonzero(held)
    fit_residual, fit_m1_labels = _extract_m1_residual(
        m1_model, m1_dataset, fit_indices, device=device
    )
    held_residual, held_m1_labels = _extract_m1_residual(
        m1_model, m1_dataset, held_indices, device=device
    )
    m1_residual = np.full((ROWS, CLASSES), np.nan, dtype=np.float32)
    m1_residual[fit] = fit_residual
    m1_residual[held] = held_residual
    if not np.array_equal(fit_m1_labels, labels[fit]) or not np.array_equal(
        held_m1_labels, labels[held]
    ):
        raise B35ContractError("M1 extraction labels changed")
    del m1_model
    gc.collect()
    torch.cuda.empty_cache()

    retained_path = m1_state_path.parent / "fold1_scores.npz"
    with np.load(retained_path, allow_pickle=False) as retained:
        m1_replay = retained["base_scores"] + m1_residual[held]
        m1_replay_max_abs = float(
            np.max(np.abs(m1_replay - retained["coupled_kl_control_scores"]))
        )
        if not np.array_equal(retained["indices"], held_indices):
            raise B35ContractError("B30 held order changed")
    if m1_replay_max_abs != 0.0:
        raise B35ContractError(f"B30 M1 replay changed: {m1_replay_max_abs}")

    control = b29._fit_readout(primary_logits, labels, fit)
    candidate = b29._fit_readout(
        np.concatenate((primary_logits, m1_residual), axis=1), labels, fit
    )
    if not bool(control["converged"] and candidate["converged"]):
        raise B35ContractError("B35 readout did not converge")
    metrics = {
        "b9_like_primary": classification_summary(labels[held], primary_logits[held]),
        "primary_readout_control": classification_summary(
            labels[held], control["scores"][held]
        ),
        "primary_m1_fusion": classification_summary(
            labels[held], candidate["scores"][held]
        ),
    }
    gate = _gate(metrics)

    score_path = output / "crossfitted_scores.npz"
    _atomic_npz(
        score_path,
        labels=labels,
        folds=assignment["folds"],
        primary_logits=primary_logits,
        m1_residual=m1_residual,
        control_scores=control["scores"],
        candidate_scores=candidate["scores"],
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "fixed_epoch_b9_like_primary_plus_crossfitted_m1_fold1",
        "git": git,
        "source_hashes": _source_hashes(repo),
        "dataset": {
            "rows": ROWS,
            "fold": FOLD,
            "fit_rows": int(fit.sum()),
            "held_rows": int(held.sum()),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "primary_training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "scheduler_horizon": SCHEDULER_HORIZON,
            "selection": "fixed_historical_B9_best_epoch_without_held_selection",
            "optimizer_updates": primary_training["optimizer_updates"],
            "finite_updates": primary_training["finite_updates"],
            "epoch_records": primary_training["epochs"],
            "wall_seconds": primary_training["wall_seconds"],
            "peak_cuda_allocated_bytes": primary_training["peak_cuda_allocated_bytes"],
            "checkpoint": primary_checkpoint,
            "strict_replay_max_abs": replay_max_abs,
        },
        "m1": {
            "path": str(m1_state_path),
            "sha256": b21.sha256_file(m1_state_path),
            "strict_replay_max_abs": m1_replay_max_abs,
        },
        "readout_iterations": {
            "control": control["iterations"],
            "candidate": candidate["iterations"],
        },
        "metrics": metrics,
        "gate": gate,
        "scores_sha256": b21.sha256_file(score_path),
        "interpretation_guard": (
            "TRAIN component fold 1 only; no validation/test. One-fold evidence "
            "cannot establish the final hybrid or mobile claim."
        ),
    }
    summary_path = output / "summary.json"
    b21._atomic_json(summary_path, summary)
    print(summary_path)
    print(gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

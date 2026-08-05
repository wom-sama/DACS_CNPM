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
from torch.utils.data import DataLoader, Subset

from trkh.models.dinov3_multidepth_convpass_b21 import SPATIAL_MODE
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


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B34_CROSSFITTED_SUPERVISED_M1_FUSION_20260805"
EXPECTED_M1_STATE_SHA256 = (
    "6373f259736db50f4394bfd369ba1132c93ac3d6588c9214d6f8d987fd57122d"
)
ROWS = 8_278
CLASSES = 5
M1_BATCH_SIZE = 64


class B34ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--m1-state", type=Path, required=True)
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/screen_b34_crossfitted_supervised_m1_fusion.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "b21": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "b22": Path("trkh/tools/diagnose_b22_recall_monotonic_expert.py"),
        "b29": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


@torch.inference_mode()
def _extract_m1_residual(
    model: torch.nn.Module,
    dataset: _TrainLedgerDataset,
    ordered_indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_indices = np.asarray(ordered_indices, dtype=np.int64)
    subset = Subset(dataset, ordered_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=M1_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    residual = np.full((ordered_indices.size, CLASSES), np.nan, dtype=np.float32)
    observed_labels = np.full(ordered_indices.size, -1, dtype=np.int64)
    positions = {
        int(index): position for position, index in enumerate(ordered_indices.tolist())
    }
    model.to(device).eval()
    for images, labels, indices in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        zeros = torch.zeros((images.shape[0], CLASSES), device=device)
        _logits, trace = model.forward_with_trace(images, zeros)
        target = np.asarray([positions[int(index)] for index in indices], dtype=np.int64)
        residual[target] = trace["residual_scores"].cpu().numpy()
        observed_labels[target] = labels.numpy()
    if not np.isfinite(residual).all() or bool((observed_labels < 0).any()):
        raise B34ContractError("M1 residual extraction is incomplete or non-finite")
    return residual, observed_labels


def _gate(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary = metrics["spatial_primary"]
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
    }
    checks = {
        "class1_gain_vs_primary": deltas["class1_f1_vs_primary"] >= 0.005,
        "class1_gain_vs_control": deltas["class1_f1_vs_control"] >= 0.010,
        "macro_noninferiority": deltas["macro_f1_vs_control"] >= -0.002,
        "accuracy_noninferiority": deltas["accuracy_vs_control"] >= -0.002,
        "class1_tp_noninferiority": deltas["class1_tp_vs_primary"] >= 0,
        "restricted_fp_nonincrease": deltas["restricted_fp_vs_control"] <= 0,
        "two_to_one_nonincrease": deltas["two_to_one_vs_control"] <= 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "deltas": deltas,
        "next_permission": (
            "one_crossfitted_b9_aligned_residual_fold"
            if all(checks.values())
            else "current_raw_dino_aligned_m1_residual_closed_for_b9_fusion"
        ),
        "validation_permission": False,
        "test_permission": False,
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise B34ContractError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)

    assignment = b21.read_locked_assignment(args.assignment_csv)
    fit = assignment["folds"] != b21.FOLD
    held = ~fit
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    fit_dataset, held_dataset, held_paths = b22._build_eval_subsets(
        train_root, assignment
    )
    spatial_path = args.spatial_checkpoint.expanduser().resolve(strict=True)
    dino_weight = b21.logical_absolute_path(args.dino_weight)
    if b21.sha256_file(spatial_path) != b22.EXPECTED_SPATIAL_SHA256:
        raise B34ContractError("B21 spatial checkpoint changed")

    device = torch.device("cuda")
    b22._seed_all(b22.SEED)
    primary_model = b21.build_arm_model(dino_weight, SPATIAL_MODE)
    primary_model.set_mode(SPATIAL_MODE)
    primary_model.load_state_dict(load_file(str(spatial_path), device="cpu"), strict=True)
    if b21.state_sha256(primary_model.state_dict()) != b22.EXPECTED_SPATIAL_STATE_SHA256:
        raise B34ContractError("B21 spatial state changed after strict reload")
    primary_model.to(device).eval()
    _fit_features, fit_logits, fit_labels = b22._extract_frozen(
        primary_model, fit_dataset, device=device, loader_seed=b22.SEED
    )
    _held_features, held_logits, held_labels = b22._extract_frozen(
        primary_model, held_dataset, device=device, loader_seed=b22.SEED + 1
    )
    provenance = b22._verify_previous_predictions(
        spatial_path, held_labels, held_logits.argmax(dim=1), held_paths
    )
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
        raise B34ContractError("B21 extraction labels changed")

    efficientvim_path = args.efficientvim_checkpoint.expanduser().resolve(strict=True)
    m1_state_path = args.m1_state.expanduser().resolve(strict=True)
    if b21.sha256_file(efficientvim_path) != OFFICIAL_M1_E450_SHA256:
        raise B34ContractError("Official EfficientViM-M1 checkpoint changed")
    if b21.sha256_file(m1_state_path) != EXPECTED_M1_STATE_SHA256:
        raise B34ContractError("B29 candidate M1 state changed")
    m1_model, _digest = b29._build_student(efficientvim_path)
    load_student_state_dict(m1_model, load_file(str(m1_state_path), device="cpu"))
    data_root = train_root.parent
    m1_dataset = _TrainLedgerDataset(
        [data_root / path for path in assignment["relative_paths"]],
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
        raise B34ContractError("M1 extraction labels changed")
    del m1_model
    gc.collect()
    torch.cuda.empty_cache()

    b29_scores_path = m1_state_path.parent / "fold0_scores.npz"
    with np.load(b29_scores_path, allow_pickle=False) as retained:
        replay = retained["base_scores"] + m1_residual[held]
        replay_max_abs = float(np.max(np.abs(replay - retained["candidate_scores"])))
        if not np.array_equal(retained["indices"], np.flatnonzero(held)):
            raise B34ContractError("B29 held order changed")
    if replay_max_abs > 1e-6:
        raise B34ContractError(f"B29 candidate replay drifted: {replay_max_abs}")

    control = b29._fit_readout(primary_logits, labels, fit)
    candidate = b29._fit_readout(
        np.concatenate((primary_logits, m1_residual), axis=1), labels, fit
    )
    if not bool(control["converged"] and candidate["converged"]):
        raise B34ContractError("B34 readout did not converge")
    metrics = {
        "spatial_primary": classification_summary(labels[held], primary_logits[held]),
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
        "mode": "train_fold0_upstream_crossfit_diagnostic",
        "git": git,
        "source_hashes": _source_hashes(repo),
        "dataset": {
            "rows": ROWS,
            "fit_rows": int(fit.sum()),
            "held_rows": int(held.sum()),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "states": {
            "spatial": {"path": str(spatial_path), "sha256": b21.sha256_file(spatial_path)},
            "m1": {"path": str(m1_state_path), "sha256": b21.sha256_file(m1_state_path)},
        },
        "replay": {"b21": provenance, "b29_max_abs": replay_max_abs},
        "readout_iterations": {
            "control": control["iterations"],
            "candidate": candidate["iterations"],
        },
        "metrics": metrics,
        "gate": gate,
        "scores_sha256": b21.sha256_file(score_path),
        "interpretation_guard": (
            "Fold 0 was unseen during upstream fitting but previously exposed during "
            "component evaluation; this is a leakage-safe diagnostic, not confirmation."
        ),
    }
    summary_path = output / "summary.json"
    b21._atomic_json(summary_path, summary)
    print(summary_path)
    print(gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

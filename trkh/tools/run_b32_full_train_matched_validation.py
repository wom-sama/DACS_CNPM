from __future__ import annotations

import argparse
import csv
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from trkh.core.config import load_data_spec
from trkh.data.dataset import ClassificationFolderDataset
from trkh.models.efficientvim_residual_student_b29 import (
    M1_PARAMETERS,
    RESIDUAL_SCALE,
    load_student_state_dict,
    student_state_dict,
)
from trkh.tools import audit_efficientvim_m1_frozen_transfer_b13 as b13
from trkh.tools import run_b29_efficientvim_teacher_residual_fold as b29
from trkh.tools import run_b31_complete_coupled_kd_oof as b31


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B32_FULL_TRAIN_MATCHED_VALIDATION_20260805"
SEED = 20260805
B9_PREDICTIONS_SHA256 = "17b81f5aa78cbc95d41fcbd31b5b2e91b9f637847d89549a38ba5c613fff099c"
CANONICAL_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)


class B32ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B32 full-TRAIN raw-DINO+M1 matched validation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--b24-dir", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--b9-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _source_hashes() -> dict[str, str]:
    root = b29._repo()
    paths = {
        "runner": Path("trkh/tools/run_b32_full_train_matched_validation.py"),
        "student": Path("trkh/models/efficientvim_residual_student_b29.py"),
        "loss": Path("trkh/tools/run_b29_efficientvim_teacher_residual_fold.py"),
        "train_helper": Path("trkh/tools/run_b31_complete_coupled_kd_oof.py"),
        "descriptor_helper": Path("trkh/tools/audit_efficientvim_m1_frozen_transfer_b13.py"),
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
            raise B32ContractError(f"B32 source missing or untracked: {path}")
    return {name: b29._sha256(root / path) for name, path in paths.items()}


def _fit_readout(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.astype(np.float64, copy=False))
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        tol=1e-8,
        max_iter=10_000,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaled, labels)
    converged = not any(
        issubclass(record.category, ConvergenceWarning) for record in caught
    ) and int(np.max(classifier.n_iter_)) < 10_000
    if not converged:
        raise B32ContractError("B32 full readout did not converge")
    return {
        "scores": classifier.decision_function(scaled).astype(np.float32),
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "coef": classifier.coef_.astype(np.float64),
        "intercept": classifier.intercept_.astype(np.float64),
        "iterations": classifier.n_iter_.astype(int).tolist(),
    }


def _apply_readout(readout: dict[str, Any], features: np.ndarray) -> np.ndarray:
    scaled = (features.astype(np.float64, copy=False) - readout["mean"]) / readout["scale"]
    scores = scaled @ readout["coef"].T + readout["intercept"]
    if scores.shape != (features.shape[0], 5) or not np.isfinite(scores).all():
        raise B32ContractError("B32 primary readout application failed")
    return scores.astype(np.float32)


def _validation_ledger(data_yaml: Path) -> tuple[list[Path], np.ndarray]:
    spec = load_data_spec(
        data_yaml.expanduser().resolve(), class_name_mode="raw", expected_num_classes=5
    )
    dataset = ClassificationFolderDataset.from_data_spec(
        spec, split="val", transform=b13._efficientvim_transform()
    )
    paths = dataset.sample_paths()
    labels = np.asarray(dataset.labels(), dtype=np.int64)
    if (
        tuple(dataset.class_names) != CANONICAL_NAMES
        or len(paths) != 2479
        or labels.shape != (2479,)
        or np.bincount(labels, minlength=5).tolist() != [558, 158, 380, 494, 889]
    ):
        raise B32ContractError("B32 validation ledger changed")
    return paths, labels


def _load_b9_predictions(
    path: Path, val_paths: Sequence[Path], labels: np.ndarray
) -> np.ndarray:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or b29._sha256(resolved) != B9_PREDICTIONS_SHA256:
        raise B32ContractError("B32 retained B9 predictions changed")
    rows = list(csv.DictReader(resolved.open("r", encoding="utf-8", newline="")))
    if len(rows) != len(val_paths):
        raise B32ContractError("B32 retained B9 row count changed")
    by_path = {Path(row["path"]).resolve(): row for row in rows}
    if len(by_path) != len(rows):
        raise B32ContractError("B32 retained B9 paths are not unique")
    predictions = np.empty(len(val_paths), dtype=np.int64)
    for index, path_value in enumerate(val_paths):
        row = by_path.get(path_value.resolve())
        if row is None:
            raise B32ContractError("B32 retained B9 path set changed")
        true_label = CANONICAL_NAMES.index(row["true_name"])
        predictions[index] = CANONICAL_NAMES.index(row["pred_name"])
        if true_label != int(labels[index]):
            raise B32ContractError("B32 retained B9 labels changed")
    return predictions


def _gate(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base, hybrid, b9 = metrics["raw_dino_primary"], metrics["hybrid"], metrics["b9"]
    deltas = {
        "accuracy_vs_primary": float(hybrid["accuracy"]) - float(base["accuracy"]),
        "macro_vs_primary": float(hybrid["macro_f1"]) - float(base["macro_f1"]),
        "class1_vs_primary": float(hybrid["class1_f1"]) - float(base["class1_f1"]),
        "tp_vs_primary": int(hybrid["class1_tp"]) - int(base["class1_tp"]),
        "accuracy_vs_b9": float(hybrid["accuracy"]) - float(b9["accuracy"]),
        "macro_vs_b9": float(hybrid["macro_f1"]) - float(b9["macro_f1"]),
        "class1_vs_b9": float(hybrid["class1_f1"]) - float(b9["class1_f1"]),
        "restricted_fp_vs_b9": int(hybrid["restricted_fp"]) - int(b9["restricted_fp"]),
    }
    checks = {
        "matched_accuracy_gain": deltas["accuracy_vs_primary"] >= 0.010,
        "matched_macro_gain": deltas["macro_vs_primary"] >= 0.010,
        "matched_class1_gain": deltas["class1_vs_primary"] >= 0.020,
        "matched_tp_retention": deltas["tp_vs_primary"] >= 0,
        "b9_accuracy_noninferiority": float(hybrid["accuracy"]) >= 0.907061,
        "b9_macro_noninferiority": float(hybrid["macro_f1"]) >= 0.871324,
        "class1_milestone": float(hybrid["class1_f1"]) >= 0.72,
        "b9_restricted_fp_nonincrease": int(hybrid["restricted_fp"])
        <= int(b9["restricted_fp"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "robustness_xai_mobile_audit_permission": passed,
        "test_permission": False,
        "finalfit_permission": False,
        "checks": checks,
        "failed": [key for key, value in checks.items() if not value],
        "deltas": deltas,
    }


def _formal(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite B32 output: {output}")
    source_start, git_start = _source_hashes(), b29._git_contract()
    if git_start["branch"] != "research/pretrained-classf-b1" or not git_start["tracked_worktree_clean"]:
        raise B32ContractError(f"B32 requires a clean pretrained branch: {git_start}")
    retained = b29._load_b25(args)
    ledger = b29._prepare_b24_ledger(args)
    labels = np.asarray(retained["arrays"]["labels"], dtype=np.int64)
    primary_readout = _fit_readout(retained["dino"], labels)
    teacher_readout = _fit_readout(
        np.concatenate((retained["dino"], retained["convnext"]), axis=1), labels
    )
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = b29._resolve_device(args.device)
    model, m1_sha = b29._build_student(args.efficientvim_checkpoint.expanduser().resolve())
    initial_sha = b29._state_dict_sha256(model.state_dict())
    train_dataset = b29._TrainLedgerDataset(
        ledger["absolute_paths"], labels, b29._efficientvim_transform()
    )
    counts = np.bincount(labels, minlength=5).astype(np.float64)
    weights = counts[labels] ** -0.5
    train_loader = DataLoader(
        train_dataset,
        batch_size=b29.BATCH_SIZE,
        sampler=WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=labels.size,
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        ),
        num_workers=b29.WORKERS,
        pin_memory=device.type == "cuda",
    )
    training = b31._train(
        model,
        train_loader,
        torch.from_numpy(primary_readout["scores"]),
        torch.from_numpy(teacher_readout["scores"]),
        device,
        -1,
    )
    val_paths, val_labels = _validation_ledger(args.data)
    dino_weight = args.dino_weight.expanduser().resolve()
    if not dino_weight.is_file() or b29._sha256(dino_weight) != b13.DINO_WEIGHT_SHA256:
        raise B32ContractError("B32 DINO weight changed")
    dino = b13._build_dino(dino_weight, num_classes=0)
    val_dino_dataset = b13._TrainLedgerDataset(
        val_paths, val_labels, b13._dino_transform(dino)
    )
    val_dino_features = b13._extract_descriptors(
        name="dino",
        model=dino,
        dataset=val_dino_dataset,
        feature_dim=b13.DINO_FEATURE_DIM,
        device=device,
        batch_size=32,
        workers=0,
    )
    val_primary = _apply_readout(primary_readout, val_dino_features)
    val_m1_dataset = b29._TrainLedgerDataset(
        val_paths, val_labels, b29._efficientvim_transform()
    )
    val_loader = DataLoader(
        val_m1_dataset,
        batch_size=b29.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=b29.WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_indices = np.arange(val_labels.size, dtype=np.int64)
    hybrid_scores, trace = b29._evaluate_pair(
        {"hybrid": model},
        val_loader,
        torch.from_numpy(val_primary),
        val_indices,
        device=device,
    )
    b9_predictions = _load_b9_predictions(args.b9_predictions, val_paths, val_labels)
    b9_scores = np.eye(5, dtype=np.float32)[b9_predictions]
    metrics = {
        "raw_dino_primary": b29.classification_summary(val_labels, val_primary),
        "hybrid": b29.classification_summary(val_labels, hybrid_scores["hybrid"]),
        "b9": b29.classification_summary(val_labels, b9_scores),
    }
    gate = _gate(metrics)
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "b32_full_train_student.safetensors"
    save_file(student_state_dict(model), str(state_path))
    reloaded, _ = b29._build_student(args.efficientvim_checkpoint.expanduser().resolve())
    load_student_state_dict(reloaded, load_file(str(state_path), device="cpu"))
    reload_scores, _ = b29._evaluate_pair(
        {"hybrid": reloaded},
        val_loader,
        torch.from_numpy(val_primary),
        val_indices,
        device=device,
    )
    reload_error = float(
        np.max(np.abs(reload_scores["hybrid"] - hybrid_scores["hybrid"]))
    )
    if reload_error != 0.0:
        raise B32ContractError(f"B32 reload mismatch: {reload_error}")
    readout_sha = b29._atomic_npz(
        output / "primary_readout.npz",
        mean=primary_readout["mean"],
        scale=primary_readout["scale"],
        coef=primary_readout["coef"],
        intercept=primary_readout["intercept"],
    )
    feature_sha = b13._atomic_npy(
        output / "validation_dino_features_f32.npy", val_dino_features
    )
    score_sha = b29._atomic_npz(
        output / "validation_scores.npz",
        labels=val_labels,
        raw_dino_primary_scores=val_primary,
        hybrid_scores=hybrid_scores["hybrid"],
        b9_predictions=b9_predictions,
    )
    source_end, git_end = _source_hashes(), b29._git_contract()
    integrity = source_start == source_end and git_start == git_end
    if not integrity:
        raise B32ContractError("B32 source/Git integrity failed")
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "full_train_once_matched_exploratory_validation",
        "source_hashes": source_start,
        "git": {"start": git_start, "end": git_end},
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "train_rows": int(labels.size),
            "validation_rows": int(val_labels.size),
            "train_split_used": True,
            "validation_split_used": True,
            "test_split_used": False,
        },
        "readouts": {
            "primary_iterations": primary_readout["iterations"],
            "teacher_iterations": teacher_readout["iterations"],
            "primary_artifact_sha256": readout_sha,
        },
        "student": {
            "m1_checkpoint_sha256": m1_sha,
            "parameters": M1_PARAMETERS,
            "initial_state_sha256": initial_sha,
            "state_path": str(state_path),
            "state_sha256": b29._sha256(state_path),
            "reload_max_abs": reload_error,
            "residual_scale": RESIDUAL_SCALE,
        },
        "dino": {
            "weight_sha256": b13.DINO_WEIGHT_SHA256,
            "validation_feature_sha256": feature_sha,
        },
        "b9_predictions_sha256": B9_PREDICTIONS_SHA256,
        "training": training,
        "metrics": metrics,
        "trace": trace["hybrid"],
        "gate": gate,
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

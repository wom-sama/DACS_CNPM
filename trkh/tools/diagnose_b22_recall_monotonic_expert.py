from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from trkh.data.dataset import ClassificationFolderDataset, build_eval_transform
from trkh.models.dinov3_multidepth_convpass_b21 import SPATIAL_MODE
from trkh.models.recall_monotonic_expert_b22 import (
    BIAS_ONLY,
    FEATURE_LOGITS,
    LOGITS_ONLY,
    RecallMonotonicLogitExpert,
)
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21


PROTOCOL_ID = "TRKH_B22_RECALL_MONOTONIC_EXPERT_DIAGNOSTIC_20260805"
SEED = 20260805
FOCUS_CLASS = 1
FEATURE_DIM = 384
NUM_CLASSES = 5
MAX_CORRECTION = 4.0
INITIAL_BIAS = -8.0
TRAIN_STEPS = 400
LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.01
POSITIVE_WEIGHT_POWER = 0.5
EVAL_BATCH_SIZE = 64
WORKERS = 2
EXPECTED_SPATIAL_SHA256 = (
    "69b5902735be2b916ff127eed653bc128713398ae6f47db13aa99be4a491857c"
)
EXPECTED_SPATIAL_STATE_SHA256 = (
    "9f4623be0077c9652daa7347b628bbf877d438beb4de7cdfa7870e8c5932aedb"
)
EXPECTED_HELD_ROWS = 1823
EXPECTED_DIRECT_CLASS1_F1 = 0.6918918918918919
EXPECTED_SPATIAL_METRICS = {
    "accuracy": 0.8804168952276468,
    "macro_f1": 0.8467748826448902,
    "class1_f1": 0.6740331491712708,
    "class1_recall": 0.6354166666666666,
    "class1_fp": 24,
    "two_to_one": 17,
}
EXPERT_MODES = (BIAS_ONLY, LOGITS_ONLY, FEATURE_LOGITS)


class B22DiagnosticError(RuntimeError):
    """Raised when a fixed diagnostic input or invariant drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def focus_log_odds(logits: Tensor, focus_class: int = FOCUS_CLASS) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected logits [B,{NUM_CLASSES}], got {tuple(logits.shape)}.")
    rivals = torch.cat(
        (logits[:, :focus_class], logits[:, focus_class + 1 :]), dim=1
    )
    return logits[:, focus_class] - torch.logsumexp(rivals, dim=1)


def _metric_view(metrics: Mapping[str, Any]) -> Dict[str, float | int]:
    class1 = metrics["class1"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "class1_f1": float(class1["f1"]),
        "class1_recall": float(class1["recall"]),
        "class1_tp": int(class1["tp"]),
        "class1_fp": int(class1["fp"]),
        "two_to_one": int(metrics["transitions_into_class1"]["2->1"]),
    }


def evaluate_diagnostic_gate(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    spatial_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate = _metric_view(arms[FEATURE_LOGITS]["held_metrics"])
    bias = _metric_view(arms[BIAS_ONLY]["held_metrics"])
    logits = _metric_view(arms[LOGITS_ONLY]["held_metrics"])
    base = _metric_view(spatial_metrics)
    best_control_class1_f1 = max(
        float(bias["class1_f1"]), float(logits["class1_f1"])
    )
    checks = {
        "feature_specific_class1_gain": float(candidate["class1_f1"])
        >= best_control_class1_f1 + 0.003,
        "beats_direct_class1_anchor": float(candidate["class1_f1"])
        >= EXPECTED_DIRECT_CLASS1_F1 + 0.002,
        "macro_noninferiority": float(candidate["macro_f1"])
        >= float(base["macro_f1"]) - 0.002,
        "accuracy_noninferiority": float(candidate["accuracy"])
        >= float(base["accuracy"]) - 0.002,
        "class1_recall_monotonic": float(candidate["class1_recall"])
        >= float(base["class1_recall"]),
        "class1_fp_budget": int(candidate["class1_fp"])
        <= int(base["class1_fp"]) + 5,
        "two_to_one_budget": int(candidate["two_to_one"])
        <= int(base["two_to_one"]) + 3,
        "candidate_parameter_contract": int(arms[FEATURE_LOGITS]["parameter_count"])
        == 390,
        "all_arms_finite": all(bool(arms[mode]["finite"]) for mode in EXPERT_MODES),
        "all_arms_preserve_existing_focus": all(
            bool(arms[mode]["existing_focus_predictions_preserved"])
            for mode in EXPERT_MODES
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate": candidate,
        "bias_control": bias,
        "logits_control": logits,
        "spatial_base": base,
        "thresholds": {
            "feature_class1_gain_over_both_controls_min": 0.003,
            "class1_f1_over_direct_min": 0.002,
            "macro_delta_vs_spatial_min": -0.002,
            "accuracy_delta_vs_spatial_min": -0.002,
            "class1_fp_increase_max": 5,
            "two_to_one_increase_max": 3,
        },
        "meaning": (
            "pass_allows_new_prospectively_locked_fresh_fold_screen_only"
            if all(checks.values())
            else "fail_closes_exact_b22_diagnostic_without_validation_or_test"
        ),
    }


def _build_eval_subsets(
    train_root: Path,
    assignment: Mapping[str, Any],
) -> tuple[Subset, Subset, list[str]]:
    dataset = ClassificationFolderDataset(
        train_root,
        b21.EXPECTED_CLASS_NAMES,
        transform=build_eval_transform(image_size=256, resize_mode="pad"),
        split="train_b22_frozen_eval",
    )
    indices = b21.map_dataset_indices(
        dataset,
        data_root=train_root.parent,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=assignment["labels"],
    )
    folds = np.asarray(assignment["folds"], dtype=np.int64)
    fit_positions = np.flatnonzero(folds != b21.FOLD).tolist()
    held_positions = np.flatnonzero(folds == b21.FOLD).tolist()
    fit_indices = [indices[position] for position in fit_positions]
    held_indices = [indices[position] for position in held_positions]
    held_paths = [assignment["relative_paths"][position] for position in held_positions]
    return Subset(dataset, fit_indices), Subset(dataset, held_indices), held_paths


@torch.inference_mode()
def _extract_frozen(
    model: torch.nn.Module,
    dataset: Subset,
    *,
    device: torch.device,
    loader_seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        worker_init_fn=b21._seed_worker,
        generator=torch.Generator(device="cpu").manual_seed(int(loader_seed)),
        persistent_workers=False,
        prefetch_factor=2,
    )
    feature_chunks: list[Tensor] = []
    logit_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    for batch in loader:
        images, labels = b21._split_batch(batch)
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.forward_features(images)
            features = model.forward_head(tokens, pre_logits=True)
            logits = model.forward_head(tokens, pre_logits=False)
        if features.ndim != 2 or tuple(features.shape[1:]) != (FEATURE_DIM,):
            raise B22DiagnosticError(
                f"Unexpected frozen feature geometry: {tuple(features.shape)}."
            )
        feature_chunks.append(features.float().cpu())
        logit_chunks.append(logits.float().cpu())
        label_chunks.append(labels.long().cpu())
    return (
        torch.cat(feature_chunks, dim=0),
        torch.cat(logit_chunks, dim=0),
        torch.cat(label_chunks, dim=0),
    )


def _standardize_features(fit: Tensor, held: Tensor) -> tuple[Tensor, Tensor, Dict[str, float]]:
    mean = fit.mean(dim=0, keepdim=True)
    scale = fit.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    fit_standardized = (fit - mean) / scale
    held_standardized = (held - mean) / scale
    return fit_standardized, held_standardized, {
        "minimum_fit_scale": float(scale.min()),
        "maximum_fit_scale": float(scale.max()),
    }


def _correction_summary(correction: Tensor) -> Dict[str, float]:
    values = correction.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _train_expert(
    *,
    mode: str,
    fit_features: Tensor,
    fit_logits: Tensor,
    fit_labels: Tensor,
    held_features: Tensor,
    held_logits: Tensor,
    held_labels: Tensor,
) -> tuple[RecallMonotonicLogitExpert, Dict[str, Any]]:
    _seed_all(SEED)
    expert = RecallMonotonicLogitExpert(
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        mode=mode,
        focus_class=FOCUS_CLASS,
        max_correction=MAX_CORRECTION,
        initial_bias=INITIAL_BIAS,
    )
    optimizer = torch.optim.AdamW(
        expert.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    binary_targets = (fit_labels == FOCUS_CLASS).float()
    positives = int(binary_targets.sum().item())
    negatives = int(binary_targets.numel() - positives)
    positive_weight = torch.tensor(
        (negatives / positives) ** POSITIVE_WEIGHT_POWER,
        dtype=torch.float32,
    )
    losses: list[float] = []
    finite = True
    expert.train()
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        adjusted, _ = expert(fit_features, fit_logits)
        loss = F.binary_cross_entropy_with_logits(
            focus_log_odds(adjusted),
            binary_targets,
            pos_weight=positive_weight,
        )
        if not bool(torch.isfinite(loss)):
            finite = False
            raise B22DiagnosticError(f"Non-finite expert loss: mode={mode}, step={step}.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            finite = False
            raise B22DiagnosticError(
                f"Non-finite expert gradient: mode={mode}, step={step}."
            )
        optimizer.step()
        losses.append(float(loss.detach()))

    expert.eval()
    with torch.inference_mode():
        fit_adjusted, fit_correction = expert(fit_features, fit_logits)
        held_adjusted, held_correction = expert(held_features, held_logits)
    base_held_predictions = held_logits.argmax(dim=1)
    held_predictions = held_adjusted.argmax(dim=1)
    existing_focus = base_held_predictions == FOCUS_CLASS
    preserved = bool((held_predictions[existing_focus] == FOCUS_CLASS).all())
    result = {
        "mode": mode,
        "parameter_count": expert.parameter_count(),
        "finite": finite and all(math.isfinite(value) for value in losses),
        "train_loss_first": losses[0],
        "train_loss_final": losses[-1],
        "fit_metrics": b21.classification_metrics(
            fit_labels.tolist(), fit_adjusted.argmax(dim=1).tolist()
        ),
        "held_metrics": b21.classification_metrics(
            held_labels.tolist(), held_predictions.tolist()
        ),
        "fit_correction": _correction_summary(fit_correction),
        "held_correction": _correction_summary(held_correction),
        "existing_focus_predictions": int(existing_focus.sum()),
        "existing_focus_predictions_preserved": preserved,
        "new_focus_predictions": int(
            ((base_held_predictions != FOCUS_CLASS) & (held_predictions == FOCUS_CLASS)).sum()
        ),
        "positive_weight": float(positive_weight),
    }
    return expert, result


def _verify_previous_predictions(
    checkpoint: Path,
    labels: Tensor,
    predictions: Tensor,
    held_paths: Sequence[str],
) -> Dict[str, Any]:
    summary_path = checkpoint.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prediction_path = Path(summary["predictions"]["path"])
    if sha256_file(prediction_path) != summary["predictions"]["sha256"]:
        raise B22DiagnosticError("B21 held prediction CSV hash changed.")
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_paths = [str(row["relative_path"]).replace("\\", "/") for row in rows]
    expected_labels = [int(row["label"]) for row in rows]
    expected_predictions = [int(row["spatial_prediction"]) for row in rows]
    if list(held_paths) != expected_paths:
        raise B22DiagnosticError("B22 held path order differs from B21.")
    if labels.tolist() != expected_labels:
        raise B22DiagnosticError("B22 held labels differ from B21.")
    if predictions.tolist() != expected_predictions:
        raise B22DiagnosticError("B22 spatial checkpoint predictions differ from B21.")
    return {
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": sha256_file(prediction_path),
        "bit_exact_predictions": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise B22DiagnosticError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    if git["branch"] != "research/pretrained-classf-b1" or not git["clean"]:
        raise B22DiagnosticError("B22 requires the clean pretrained research branch.")

    checkpoint = Path(args.spatial_checkpoint).expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != EXPECTED_SPATIAL_SHA256:
        raise B22DiagnosticError("B21 spatial checkpoint SHA-256 changed.")
    dino_weight = b21.logical_absolute_path(Path(args.dino_weight))
    assignment = b21.read_locked_assignment(Path(args.assignment_csv))
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    fit_dataset, held_dataset, held_paths = _build_eval_subsets(train_root, assignment)
    if len(held_dataset) != EXPECTED_HELD_ROWS:
        raise B22DiagnosticError("Held TRAIN-fold size changed.")

    device = torch.device("cuda")
    _seed_all(SEED)
    model = b21.build_arm_model(dino_weight, SPATIAL_MODE)
    model.set_mode(SPATIAL_MODE)
    state = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise B22DiagnosticError("Strict B21 spatial checkpoint load failed.")
    if b21.state_sha256(model.state_dict()) != EXPECTED_SPATIAL_STATE_SHA256:
        raise B22DiagnosticError("B21 spatial state digest changed after reload.")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    fit_features, fit_logits, fit_labels = _extract_frozen(
        model, fit_dataset, device=device, loader_seed=SEED
    )
    held_features, held_logits, held_labels = _extract_frozen(
        model, held_dataset, device=device, loader_seed=SEED + 1
    )
    peak_cuda = int(torch.cuda.max_memory_allocated())
    del model, state
    gc.collect()
    torch.cuda.empty_cache()

    base_predictions = held_logits.argmax(dim=1)
    provenance = _verify_previous_predictions(
        checkpoint, held_labels, base_predictions, held_paths
    )
    spatial_metrics = b21.classification_metrics(
        held_labels.tolist(), base_predictions.tolist()
    )
    observed_base = _metric_view(spatial_metrics)
    expected_base = EXPECTED_SPATIAL_METRICS
    exact_base_checks = {
        key: observed_base[key] == expected_base[key]
        for key in expected_base
    }
    if not all(exact_base_checks.values()):
        raise B22DiagnosticError(
            f"B21 spatial metric anchor changed: {exact_base_checks}."
        )

    fit_features, held_features, standardization = _standardize_features(
        fit_features, held_features
    )
    arms: Dict[str, Dict[str, Any]] = {}
    experts: Dict[str, RecallMonotonicLogitExpert] = {}
    for mode in EXPERT_MODES:
        expert, result = _train_expert(
            mode=mode,
            fit_features=fit_features,
            fit_logits=fit_logits,
            fit_labels=fit_labels,
            held_features=held_features,
            held_logits=held_logits,
            held_labels=held_labels,
        )
        experts[mode] = expert
        arms[mode] = result
        print(
            mode,
            json.dumps(_metric_view(result["held_metrics"]), sort_keys=True),
            flush=True,
        )

    gate = evaluate_diagnostic_gate(arms, spatial_metrics=spatial_metrics)
    for mode, expert in experts.items():
        save_file(
            {name: value.detach().cpu() for name, value in expert.state_dict().items()},
            str(output / f"{mode}.safetensors"),
            metadata={"protocol_id": PROTOCOL_ID, "mode": mode},
        )
        arms[mode]["checkpoint"] = {
            "path": str((output / f"{mode}.safetensors").resolve()),
            "sha256": sha256_file(output / f"{mode}.safetensors"),
        }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "adaptive_train_fold_diagnostic_only",
        "adaptivity_notice": (
            "B22 was designed after B21 fold-0 metrics were observed; a pass can "
            "authorize only a newly locked fresh-fold screen, never validation/test."
        ),
        "git": git,
        "inputs": {
            "assignment_csv": str(Path(args.assignment_csv).resolve()),
            "assignment_sha256": sha256_file(Path(args.assignment_csv)),
            "dino_weight": str(dino_weight),
            "dino_sha256": sha256_file(dino_weight),
            "spatial_checkpoint": str(checkpoint),
            "spatial_checkpoint_sha256": sha256_file(checkpoint),
        },
        "fit_rows": int(fit_labels.numel()),
        "held_rows": int(held_labels.numel()),
        "training": {
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "positive_weight_power": POSITIVE_WEIGHT_POWER,
            "max_correction": MAX_CORRECTION,
            "initial_bias": INITIAL_BIAS,
            "feature_standardization": standardization,
        },
        "b21_spatial_base": spatial_metrics,
        "b21_replay": provenance,
        "arms": arms,
        "gate": gate,
        "peak_cuda_allocated_bytes": peak_cuda,
        "validation_opened": False,
        "test_opened": False,
    }
    _atomic_json(output / "summary.json", payload)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


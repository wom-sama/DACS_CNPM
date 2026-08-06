"""Screen DINOv3's official CLS + average-patch head on TRAIN component fold 1.

This is a representation-sufficiency screen, not validation or final training.
It strict-loads the retained B35 primary, freezes it, and fits fixed balanced
linear readouts on fold-1-excluded TRAIN features.  The candidate follows the
official DINOv3 linear-evaluation interface; patch-only and row-deranged CLS
controls prevent a wider readout from being mistaken for new information.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import diagnose_b22_recall_monotonic_expert as b22
from trkh.tools import run_b29_efficientvim_teacher_residual_fold as b29
from trkh.tools import run_b35_crossfitted_b9_m1_fusion as b35
from trkh.tools import run_b36_head_first_dino_fold as b36
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import classification_summary


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B37_DINOV3_CLS_PATCH_HEAD_FOLD1_20260806"
FOLD = 1
SEED = 42
TOKEN_WIDTH = 384
PREFIX_TOKENS = 5
PATCH_TOKENS = 256
CLASSES = 5
READOUT_MAX_ITER = b29.READOUT_MAX_ITER


class B37ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/screen_b37_dinov3_cls_patch_head_fold.py"),
        "extractor": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "control": Path("trkh/tools/run_b35_crossfitted_b9_m1_fusion.py"),
        "lighting": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _split_official_features(tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return CLS, average patch, and the official concatenation.

    DINOv3 register tokens occupy positions 1:5.  The official linear evaluator
    uses the class token and average patch tokens, not register concatenation.
    """

    expected = (PREFIX_TOKENS + PATCH_TOKENS, TOKEN_WIDTH)
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
        raise B37ContractError(
            f"DINO token geometry changed: {tuple(tokens.shape)} != [B,{expected[0]},{expected[1]}]."
        )
    cls = tokens[:, 0].float()
    patch = tokens[:, PREFIX_TOKENS:].float().mean(dim=1)
    return cls, patch, torch.cat((cls, patch), dim=1)


@torch.inference_mode()
def _extract(
    model: torch.nn.Module,
    dataset: Subset,
    *,
    device: torch.device,
    loader_seed: int,
) -> dict[str, np.ndarray | float]:
    loader = DataLoader(
        dataset,
        batch_size=b22.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=b22.WORKERS,
        pin_memory=True,
        worker_init_fn=b21._seed_worker,
        generator=torch.Generator(device="cpu").manual_seed(int(loader_seed)),
        persistent_workers=False,
        prefetch_factor=2 if b22.WORKERS > 0 else None,
    )
    cls_chunks: list[Tensor] = []
    patch_chunks: list[Tensor] = []
    official_chunks: list[Tensor] = []
    logit_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    maximum_pool_replay = 0.0
    for batch in loader:
        images, labels = b21._split_batch(batch)
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.forward_features(images)
            native_pool = model.forward_head(tokens, pre_logits=True)
            logits = model.forward_head(tokens, pre_logits=False)
        cls, patch, official = _split_official_features(tokens)
        maximum_pool_replay = max(
            maximum_pool_replay,
            float((native_pool.float() - patch).abs().max().cpu()),
        )
        cls_chunks.append(cls.cpu())
        patch_chunks.append(patch.cpu())
        official_chunks.append(official.cpu())
        logit_chunks.append(logits.float().cpu())
        label_chunks.append(labels.long().cpu())
    return {
        "cls": torch.cat(cls_chunks).numpy(),
        "patch": torch.cat(patch_chunks).numpy(),
        "official": torch.cat(official_chunks).numpy(),
        "logits": torch.cat(logit_chunks).numpy(),
        "labels": torch.cat(label_chunks).numpy(),
        "maximum_native_pool_replay": maximum_pool_replay,
    }


def _sattolo_indices(size: int, seed: int) -> np.ndarray:
    """Return a deterministic single-cycle permutation with no fixed points."""

    if int(size) < 2:
        raise ValueError("A derangement requires at least two rows.")
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(size), dtype=np.int64)
    for index in range(int(size) - 1, 0, -1):
        other = int(rng.integers(0, index))
        indices[index], indices[other] = indices[other], indices[index]
    if bool(np.any(indices == np.arange(int(size)))):
        raise B37ContractError("Sattolo control unexpectedly contains a fixed point.")
    return indices


def _fit_readout(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    eval_features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_features.astype(np.float64, copy=False))
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        tol=1e-8,
        max_iter=READOUT_MAX_ITER,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(fit_scaled, fit_labels)
    converged = not any(
        issubclass(record.category, ConvergenceWarning) for record in caught
    ) and int(np.max(classifier.n_iter_)) < READOUT_MAX_ITER
    scores = {
        condition: classifier.decision_function(
            scaler.transform(features.astype(np.float64, copy=False))
        ).astype(np.float32)
        for condition, features in eval_features.items()
    }
    return {
        "scores": scores,
        "iterations": classifier.n_iter_.astype(int).tolist(),
        "converged": bool(converged),
    }


def _gate(metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    raw = metrics["native_primary"]
    patch = metrics["patch_readout"]
    deranged = metrics["deranged_cls_patch"]
    candidate = metrics["official_cls_patch"]
    robust_conditions = ("lighting_dim", "lighting_bright")
    raw_robust = float(np.mean([raw[name]["class1_f1"] for name in robust_conditions]))
    candidate_robust = float(
        np.mean([candidate[name]["class1_f1"] for name in robust_conditions])
    )
    deltas = {
        "clean_class1_vs_native": float(
            candidate["clean"]["class1_f1"] - raw["clean"]["class1_f1"]
        ),
        "clean_class1_vs_patch": float(
            candidate["clean"]["class1_f1"] - patch["clean"]["class1_f1"]
        ),
        "clean_class1_vs_deranged": float(
            candidate["clean"]["class1_f1"] - deranged["clean"]["class1_f1"]
        ),
        "clean_accuracy_vs_native": float(
            candidate["clean"]["accuracy"] - raw["clean"]["accuracy"]
        ),
        "clean_macro_vs_native": float(
            candidate["clean"]["macro_f1"] - raw["clean"]["macro_f1"]
        ),
        "clean_pair_vs_native": float(
            candidate["clean"]["mean_pair_auroc"] - raw["clean"]["mean_pair_auroc"]
        ),
        "clean_tp_vs_native": int(
            candidate["clean"]["class1_tp"] - raw["clean"]["class1_tp"]
        ),
        "clean_restricted_fp_vs_native": int(
            candidate["clean"]["restricted_fp"] - raw["clean"]["restricted_fp"]
        ),
        "robust_mean_class1_vs_native": candidate_robust - raw_robust,
        "bright_class1_vs_native": float(
            candidate["lighting_bright"]["class1_f1"]
            - raw["lighting_bright"]["class1_f1"]
        ),
        "bright_zero_to_one_vs_native": int(
            candidate["lighting_bright"]["confusion_matrix"][0][1]
            - raw["lighting_bright"]["confusion_matrix"][0][1]
        ),
    }
    checks = {
        "class1_gain_vs_native": deltas["clean_class1_vs_native"] >= 0.005,
        "class1_gain_vs_patch": deltas["clean_class1_vs_patch"] >= 0.005,
        "aligned_cls_beats_deranged": deltas["clean_class1_vs_deranged"] >= 0.005,
        "clean_accuracy_noninferiority": deltas["clean_accuracy_vs_native"] >= -0.002,
        "clean_macro_noninferiority": deltas["clean_macro_vs_native"] >= -0.002,
        "clean_pair_noninferiority": deltas["clean_pair_vs_native"] >= -0.003,
        "clean_tp_noninferiority": deltas["clean_tp_vs_native"] >= 0,
        "clean_fp_nonincrease": deltas["clean_restricted_fp_vs_native"] <= 0,
        "robust_mean_gain": deltas["robust_mean_class1_vs_native"] >= 0.005,
        "bright_gain": deltas["bright_class1_vs_native"] >= 0.005,
        "bright_zero_to_one_nonincrease": deltas["bright_zero_to_one_vs_native"] <= 0,
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "checks": checks,
        "failed": [name for name, value in checks.items() if not value],
        "deltas": deltas,
        "next_permission": (
            "implement_zero_init_cls_patch_head_on_fresh_train_fold"
            if passed
            else "do_not_promote_cls_patch_head_from_this_strong_checkpoint"
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
        raise B37ContractError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    control, checkpoint, control_scores = b36._load_control(args.control_summary)
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    fit_dataset, clean_held = b35._eval_subsets(train_root, assignment)
    held_datasets: dict[str, Subset] = {"clean": clean_held}
    for condition, (brightness, contrast) in b36.CONDITIONS.items():
        if condition != "clean":
            held_datasets[condition] = b36._held_subset(
                train_root,
                assignment,
                brightness=brightness,
                contrast=contrast,
            )

    dino_weight = b21.logical_absolute_path(args.dino_weight)
    model = b21.build_arm_model(dino_weight, DIRECT_MODE)
    model.set_mode(DIRECT_MODE)
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    expected_state = control["primary_training"]["checkpoint"]["state_sha256"]
    if b21.state_sha256(model.state_dict()) != expected_state:
        raise B37ContractError("B35 primary state changed after strict load.")
    device = torch.device("cuda")
    model.to(device).eval()
    fit = _extract(model, fit_dataset, device=device, loader_seed=SEED)
    held = {
        condition: _extract(
            model,
            dataset,
            device=device,
            loader_seed=SEED + 100 + index,
        )
        for index, (condition, dataset) in enumerate(held_datasets.items())
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()

    fit_labels = np.asarray(fit["labels"], dtype=np.int64)
    held_labels = np.asarray(held["clean"]["labels"], dtype=np.int64)
    for condition in held_datasets:
        if not np.array_equal(held_labels, np.asarray(held[condition]["labels"])):
            raise B37ContractError("Held label order changed across lighting conditions.")
    assignment_folds = np.asarray(assignment["folds"], dtype=np.int64)
    assignment_labels = np.asarray(assignment["labels"], dtype=np.int64)
    if not np.array_equal(fit_labels, assignment_labels[assignment_folds != FOLD]):
        raise B37ContractError("Fit labels changed against locked assignment.")
    if not np.array_equal(held_labels, assignment_labels[assignment_folds == FOLD]):
        raise B37ContractError("Held labels changed against locked assignment.")

    fit_derangement = _sattolo_indices(len(fit_labels), SEED + 1_000)
    held_derangement = _sattolo_indices(len(held_labels), SEED + 2_000)
    eval_patch = {
        name: np.asarray(payload["patch"], dtype=np.float32)
        for name, payload in held.items()
    }
    eval_official = {
        name: np.asarray(payload["official"], dtype=np.float32)
        for name, payload in held.items()
    }
    eval_deranged = {
        name: np.concatenate(
            (
                np.asarray(payload["cls"], dtype=np.float32)[held_derangement],
                np.asarray(payload["patch"], dtype=np.float32),
            ),
            axis=1,
        )
        for name, payload in held.items()
    }
    readouts = {
        "patch_readout": _fit_readout(
            np.asarray(fit["patch"], dtype=np.float32), fit_labels, eval_patch
        ),
        "official_cls_patch": _fit_readout(
            np.asarray(fit["official"], dtype=np.float32), fit_labels, eval_official
        ),
        "deranged_cls_patch": _fit_readout(
            np.concatenate(
                (
                    np.asarray(fit["cls"], dtype=np.float32)[fit_derangement],
                    np.asarray(fit["patch"], dtype=np.float32),
                ),
                axis=1,
            ),
            fit_labels,
            eval_deranged,
        ),
    }
    if not all(bool(payload["converged"]) for payload in readouts.values()):
        raise B37ContractError(
            f"A locked readout did not converge: "
            f"{ {name: value['iterations'] for name, value in readouts.items()} }"
        )

    metrics: dict[str, dict[str, dict[str, Any]]] = {
        "native_primary": {
            name: classification_summary(
                held_labels, np.asarray(payload["logits"], dtype=np.float32)
            )
            for name, payload in held.items()
        }
    }
    for method, readout in readouts.items():
        metrics[method] = {
            condition: classification_summary(held_labels, scores)
            for condition, scores in readout["scores"].items()
        }

    with np.load(control_scores, allow_pickle=False) as retained:
        retained_held = np.asarray(retained["folds"]) == FOLD
        raw_replay = float(
            np.max(
                np.abs(
                    np.asarray(held["clean"]["logits"], dtype=np.float32)
                    - retained["primary_logits"][retained_held]
                )
            )
        )
    if raw_replay != 0.0:
        raise B37ContractError(f"B35 clean-logit replay changed: {raw_replay}")
    pool_replay = max(
        [float(fit["maximum_native_pool_replay"])]
        + [float(payload["maximum_native_pool_replay"]) for payload in held.values()]
    )
    if pool_replay != 0.0:
        raise B37ContractError(f"Native average-patch replay changed: {pool_replay}")

    gate = _gate(metrics)
    scores_path = output / "held_cls_patch_scores.npz"
    arrays: dict[str, np.ndarray] = {"labels": held_labels}
    for condition, payload in held.items():
        arrays[f"native_primary_{condition}"] = np.asarray(
            payload["logits"], dtype=np.float32
        )
    for method, readout in readouts.items():
        for condition, scores in readout["scores"].items():
            arrays[f"{method}_{condition}"] = scores
    _atomic_npz(scores_path, **arrays)

    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "frozen_strong_primary_official_cls_patch_readout_train_fold1",
        "git": git,
        "source_hashes": _source_hashes(repo),
        "dataset": {
            "fold": FOLD,
            "fit_rows": len(fit_labels),
            "held_rows": len(held_labels),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "control": {
            "summary": str(args.control_summary.expanduser().resolve()),
            "summary_sha256": b36.CONTROL_SUMMARY_SHA256,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": b36.CONTROL_CHECKPOINT_SHA256,
            "strict_clean_logit_replay_max_abs": raw_replay,
        },
        "interface": {
            "official_reference": "https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/linear.py",
            "candidate_order": ["class_token", "average_patch_tokens"],
            "register_tokens_used": False,
            "native_pool": "average_patch_tokens_only",
            "token_width": TOKEN_WIDTH,
            "candidate_width": TOKEN_WIDTH * 2,
            "maximum_native_pool_replay_abs": pool_replay,
        },
        "readouts": {
            name: {
                "iterations": payload["iterations"],
                "converged": payload["converged"],
            }
            for name, payload in readouts.items()
        },
        "same_capacity_control": {
            "method": "Sattolo row-deranged CLS concatenated with aligned patch mean",
            "fit_fixed_points": int(np.sum(fit_derangement == np.arange(len(fit_labels)))),
            "held_fixed_points": int(
                np.sum(held_derangement == np.arange(len(held_labels)))
            ),
            "fit_label_coincidence": float(
                np.mean(fit_labels[fit_derangement] == fit_labels)
            ),
            "held_label_coincidence": float(
                np.mean(held_labels[held_derangement] == held_labels)
            ),
        },
        "deployment_delta": {
            "additional_parameters": TOKEN_WIDTH * CLASSES,
            "additional_macs_per_image": TOKEN_WIDTH * CLASSES,
            "backbone_changed": False,
        },
        "metrics": metrics,
        "gate": gate,
        "scores_sha256": b21.sha256_file(scores_path),
        "interpretation_guard": (
            "Design-exposed TRAIN component fold 1 only. A pass authorizes one "
            "zero-init inherited dual head on a fresh TRAIN fold; never validation/test."
        ),
    }
    b21._atomic_json(output / "summary.json", summary)
    print(json.dumps(gate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import ImageEnhance
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import Subset

from trkh.data.dataset import ClassificationFolderDataset, build_eval_transform
from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import diagnose_b22_recall_monotonic_expert as b22
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import classification_summary


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B36_HEAD_FIRST_DINO_FOLD1_20260806"
CONTROL_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B35_CROSSFITTED_B9_M1_FUSION_20260805"
CONTROL_SUMMARY_SHA256 = "5c5e295b295085632137da23e501aea9a48ebf3d043f9eb3e78fd7a63af74e88"
CONTROL_CHECKPOINT_SHA256 = "b29bc28f5e59e5690dd3e99bba93b7de7edf5023fa834bdb76a12821ff50e548"
CONTROL_SCORE_SHA256 = "1f913b15974a92417bda39cdb58ea5aec3b76a3d096dd4ce84fc010a93dcd9e0"
FOLD = 1
SEED = 42
EPOCHS = 9
BACKBONE_FREEZE_EPOCHS = b21.WARMUP_EPOCHS
SCHEDULER_HORIZON = 30
CONDITIONS = {
    "clean": (1.0, 1.0),
    "lighting_dim": (0.70, 0.90),
    "lighting_bright": (1.25, 1.10),
}


class B36ContractError(RuntimeError):
    pass


class _LightingEvalTransform:
    def __init__(self, brightness: float, contrast: float) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.base = build_eval_transform(image_size=256, resize_mode="pad")

    def __call__(self, image, *, target=None):
        shifted = ImageEnhance.Brightness(image).enhance(self.brightness)
        shifted = ImageEnhance.Contrast(shifted).enhance(self.contrast)
        return self.base(shifted, target=target)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "trainer": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _load_control(path: Path) -> tuple[dict[str, Any], Path, Path]:
    resolved = path.expanduser().resolve(strict=True)
    if b21.sha256_file(resolved) != CONTROL_SUMMARY_SHA256:
        raise B36ContractError("B35 control summary changed.")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    checkpoint = Path(payload["primary_training"]["checkpoint"]["path"])
    score_path = resolved.parent / "crossfitted_scores.npz"
    checks = {
        "protocol": payload.get("protocol_id") == CONTROL_PROTOCOL_ID,
        "fold": int(payload.get("dataset", {}).get("fold", -1)) == FOLD,
        "train_only": bool(payload["dataset"].get("train_split_used")),
        "validation_closed": not bool(payload["dataset"].get("validation_split_used")),
        "test_closed": not bool(payload["dataset"].get("test_split_used")),
        "checkpoint": b21.sha256_file(checkpoint) == CONTROL_CHECKPOINT_SHA256,
        "scores": b21.sha256_file(score_path) == CONTROL_SCORE_SHA256,
    }
    if not all(checks.values()):
        raise B36ContractError(f"B35 control contract changed: {checks}")
    return payload, checkpoint, score_path


def _held_subset(
    train_root: Path,
    assignment: Mapping[str, Any],
    *,
    brightness: float,
    contrast: float,
    fold: int = FOLD,
) -> Subset:
    dataset = ClassificationFolderDataset(
        train_root,
        b21.EXPECTED_CLASS_NAMES,
        transform=_LightingEvalTransform(brightness, contrast),
        split="train_b36_held",
    )
    mapped = b21.map_dataset_indices(
        dataset,
        data_root=train_root.parent,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=assignment["labels"],
    )
    if int(fold) not in range(5):
        raise ValueError("fold must be in 0..4")
    held = np.flatnonzero(np.asarray(assignment["folds"]) == int(fold))
    return Subset(dataset, [mapped[int(index)] for index in held])


def _evaluate_checkpoint(
    *,
    dino_weight: Path,
    checkpoint: Path,
    expected_state_sha256: str,
    datasets: Mapping[str, Subset],
    seed_offset: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Tensor], dict[str, Tensor]]:
    model = b21.build_arm_model(dino_weight, DIRECT_MODE)
    model.set_mode(DIRECT_MODE)
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    if b21.state_sha256(model.state_dict()) != expected_state_sha256:
        raise B36ContractError(f"Checkpoint state changed: {checkpoint}")
    device = torch.device("cuda")
    model.to(device).eval()
    metrics: dict[str, dict[str, Any]] = {}
    features: dict[str, Tensor] = {}
    logits: dict[str, Tensor] = {}
    expected_labels: Tensor | None = None
    for index, (condition, dataset) in enumerate(datasets.items()):
        feature, score, labels = b22._extract_frozen(
            model,
            dataset,
            device=device,
            loader_seed=SEED + seed_offset + index,
        )
        if expected_labels is None:
            expected_labels = labels
        elif not torch.equal(expected_labels, labels):
            raise B36ContractError("Condition label order changed.")
        features[condition] = feature
        logits[condition] = score
        metrics[condition] = classification_summary(labels.numpy(), score.numpy())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, features, logits


def _invariance(features: Mapping[str, Tensor], labels: np.ndarray) -> dict[str, Any]:
    clean = features["clean"].float()
    result: dict[str, Any] = {}
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    for condition in ("lighting_dim", "lighting_bright"):
        cosine = torch.nn.functional.cosine_similarity(clean, features[condition].float(), dim=1)
        class1 = cosine[label_tensor == 1]
        result[condition] = {
            "median_cosine": float(torch.median(cosine)),
            "p05_cosine": float(torch.quantile(cosine, 0.05)),
            "class1_median_cosine": float(torch.median(class1)),
        }
    return result


def _gate(
    control: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    control_invariance: Mapping[str, Mapping[str, float]],
    candidate_invariance: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    clean_control = control["clean"]
    clean_candidate = candidate["clean"]
    robust_control = np.mean(
        [control[name]["class1_f1"] for name in ("lighting_dim", "lighting_bright")]
    )
    robust_candidate = np.mean(
        [candidate[name]["class1_f1"] for name in ("lighting_dim", "lighting_bright")]
    )
    deltas = {
        "clean_accuracy": float(clean_candidate["accuracy"] - clean_control["accuracy"]),
        "clean_macro_f1": float(clean_candidate["macro_f1"] - clean_control["macro_f1"]),
        "clean_class1_f1": float(clean_candidate["class1_f1"] - clean_control["class1_f1"]),
        "clean_pair_auroc": float(
            clean_candidate["mean_pair_auroc"] - clean_control["mean_pair_auroc"]
        ),
        "clean_class1_tp": int(clean_candidate["class1_tp"] - clean_control["class1_tp"]),
        "clean_restricted_fp": int(
            clean_candidate["restricted_fp"] - clean_control["restricted_fp"]
        ),
        "robust_mean_class1_f1": float(robust_candidate - robust_control),
        "bright_class1_f1": float(
            candidate["lighting_bright"]["class1_f1"]
            - control["lighting_bright"]["class1_f1"]
        ),
        "bright_zero_to_one": int(
            candidate["lighting_bright"]["confusion_matrix"][0][1]
            - control["lighting_bright"]["confusion_matrix"][0][1]
        ),
        "bright_class1_feature_cosine": float(
            candidate_invariance["lighting_bright"]["class1_median_cosine"]
            - control_invariance["lighting_bright"]["class1_median_cosine"]
        ),
    }
    checks = {
        "clean_class1_gain": deltas["clean_class1_f1"] >= 0.005,
        "clean_macro_noninferiority": deltas["clean_macro_f1"] >= -0.002,
        "clean_accuracy_noninferiority": deltas["clean_accuracy"] >= -0.002,
        "clean_pair_noninferiority": deltas["clean_pair_auroc"] >= -0.003,
        "clean_tp_noninferiority": deltas["clean_class1_tp"] >= 0,
        "clean_fp_nonincrease": deltas["clean_restricted_fp"] <= 0,
        "robust_mean_gain": deltas["robust_mean_class1_f1"] >= 0.010,
        "bright_gain": deltas["bright_class1_f1"] >= 0.010,
        "bright_zero_to_one_nonincrease": deltas["bright_zero_to_one"] <= 0,
        "bright_feature_preservation": deltas["bright_class1_feature_cosine"] >= 0.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "deltas": deltas,
        "next_permission": (
            "confirm_head_first_on_fresh_component_fold"
            if all(checks.values())
            else "close_exact_two_epoch_head_first_schedule"
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
        raise B36ContractError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    control, control_checkpoint, control_scores = _load_control(args.control_summary)
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    train_fit, train_held, held_paths, fit_labels = b21._build_datasets(
        train_root, assignment, fold=FOLD
    )
    dino_weight = b21.logical_absolute_path(args.dino_weight)

    candidate_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=EPOCHS,
        scheduler_horizon=SCHEDULER_HORIZON,
        seed=SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b36_head_first_ema",
        backbone_freeze_epochs=BACKBONE_FREEZE_EPOCHS,
    )
    candidate_checkpoint = candidate_training["checkpoint"]
    if candidate_checkpoint is None:
        raise B36ContractError("Candidate checkpoint was not saved.")

    datasets = {
        name: _held_subset(
            train_root,
            assignment,
            brightness=brightness,
            contrast=contrast,
        )
        for name, (brightness, contrast) in CONDITIONS.items()
    }
    control_metrics, control_features, control_logits = _evaluate_checkpoint(
        dino_weight=dino_weight,
        checkpoint=control_checkpoint,
        expected_state_sha256=control["primary_training"]["checkpoint"]["state_sha256"],
        datasets=datasets,
        seed_offset=100,
    )
    candidate_metrics, candidate_features, candidate_logits = _evaluate_checkpoint(
        dino_weight=dino_weight,
        checkpoint=Path(candidate_checkpoint["path"]),
        expected_state_sha256=candidate_checkpoint["state_sha256"],
        datasets=datasets,
        seed_offset=200,
    )
    labels = np.asarray(candidate_training["held_labels"], dtype=np.int64)
    with np.load(control_scores, allow_pickle=False) as retained:
        held = retained["folds"] == FOLD
        control_replay = float(
            np.max(np.abs(control_logits["clean"].numpy() - retained["primary_logits"][held]))
        )
        if not np.array_equal(labels, retained["labels"][held]):
            raise B36ContractError("Held label order changed against B35.")
    candidate_replay = float(
        np.max(
            np.abs(
                candidate_logits["clean"].numpy()
                - np.asarray(candidate_training["held_logits"], dtype=np.float32)
            )
        )
    )
    if control_replay != 0.0 or candidate_replay != 0.0:
        raise B36ContractError(
            f"Clean replay changed: control={control_replay}, candidate={candidate_replay}"
        )

    control_invariance = _invariance(control_features, labels)
    candidate_invariance = _invariance(candidate_features, labels)
    gate = _gate(control_metrics, candidate_metrics, control_invariance, candidate_invariance)
    scores_path = output / "held_condition_scores.npz"
    arrays: dict[str, np.ndarray] = {"labels": labels}
    for condition in CONDITIONS:
        arrays[f"control_{condition}"] = control_logits[condition].numpy()
        arrays[f"candidate_{condition}"] = candidate_logits[condition].numpy()
    _atomic_npz(scores_path, **arrays)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "paired_design_exposed_train_fold_head_first_adaptation",
        "git": git,
        "source_hashes": _source_hashes(repo),
        "dataset": {
            "fold": FOLD,
            "fit_rows": len(train_fit),
            "held_rows": len(train_held),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "schedule": {
            "seed": SEED,
            "epochs": EPOCHS,
            "scheduler_horizon": SCHEDULER_HORIZON,
            "backbone_freeze_epochs": BACKBONE_FREEZE_EPOCHS,
            "causal_difference": (
                "candidate freezes only the pretrained backbone during the existing "
                "two-epoch LR warmup; the retained B35 control updates it immediately"
            ),
        },
        "control": {
            "summary": str(args.control_summary.expanduser().resolve()),
            "summary_sha256": CONTROL_SUMMARY_SHA256,
            "checkpoint": str(control_checkpoint),
            "checkpoint_sha256": CONTROL_CHECKPOINT_SHA256,
        },
        "candidate_training": {
            "optimizer_updates": candidate_training["optimizer_updates"],
            "finite_updates": candidate_training["finite_updates"],
            "frozen_backbone_parameters": candidate_training["frozen_backbone_parameters"],
            "epoch_records": candidate_training["epochs"],
            "wall_seconds": candidate_training["wall_seconds"],
            "peak_cuda_allocated_bytes": candidate_training["peak_cuda_allocated_bytes"],
            "checkpoint": candidate_checkpoint,
            "strict_replay_max_abs": candidate_replay,
        },
        "metrics": {"control": control_metrics, "candidate": candidate_metrics},
        "feature_invariance": {
            "control": control_invariance,
            "candidate": candidate_invariance,
        },
        "control_strict_replay_max_abs": control_replay,
        "scores_sha256": b21.sha256_file(scores_path),
        "gate": gate,
        "interpretation_guard": (
            "Design-exposed TRAIN component fold 1 reuses B35 as a resource-saving "
            "control. A pass requires confirmation on a fresh fold; validation/test stay closed."
        ),
    }
    b21._atomic_json(output / "summary.json", summary)
    print(json.dumps(gate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

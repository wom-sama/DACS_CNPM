from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (  # noqa: E402
    BATCH_SIZE,
    CLASS1,
    EXPECTED_ASSIGNMENT_SHA256,
    EXPECTED_CLASS_COUNTS,
    FOLDS,
    WORKERS,
    _TrainLedgerDataset,
    _atomic_json,
    _atomic_npy,
    _atomic_npz,
    _dependency_contract,
    _device_contract,
    _git_contract,
    _sha256,
    _train_content_contract,
    classification_summary,
    fit_oof_readout,
    paired_component_bootstrap,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (  # noqa: E402
    EXPECTED_DATA_SHA256,
    EXPECTED_TRAIN_SAMPLES,
    _data_root_from_yaml,
    assert_train_only_paths,
)
from trkh.tools.probe_embedding_prototypes import _resolve_device  # noqa: E402
from trkh.tools.screen_dinov3_cgaer_b11_sourcefold import (  # noqa: E402
    read_locked_assignment,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B24_DINOV3_CONVNEXT_LOCAL_TEACHER_20260805"
SEED = 20260805
MODEL_NAME = "convnext_tiny.dinov3_lvd1689m"
MODEL_REPOSITORY = "timm/convnext_tiny.dinov3_lvd1689m"
MODEL_WEIGHT_SHA256 = (
    "a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f"
)
IMAGE_SIZE = 224
FEATURE_DIM = 768
PARAMETERS = 27_820_128
BOOTSTRAP_REPLICATES = 5_000

B13_DINO_FEATURE_SHA256 = (
    "0429260ab2f619cf8312848e08af1a13f4c81caa95a93daebda6e6e22240dc38"
)
B13_PATHS_SHA256 = (
    "a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb"
)
B13_OOF_SHA256 = (
    "82bcef6d293223b819f559122bffee822b7f2a2c80a200113dd6c0b7843def39"
)
B13_SUMMARY_SHA256 = (
    "b017d34955f1c9126a611cf064cb919b6696ed9a19bf0f2b676838ad97955a75"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked B24 TRAIN-only raw-DINO versus DINOv3-ConvNeXt-T frozen "
            "local-teacher screen. Validation and test are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--convnext-weight", type=Path, required=True)
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_paths() -> dict[str, Path]:
    root = _repository_root()
    return {
        "runner": Path(__file__).resolve(),
        "test": root / "tests" / "test_audit_dinov3_convnext_b24.py",
        "decision": root / "docs" / "TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md",
        "b13_runner": root
        / "trkh"
        / "tools"
        / "audit_efficientvim_m1_frozen_transfer_b13.py",
        "data_contract": root
        / "trkh"
        / "tools"
        / "precheck_dinov3_xcnorm_a0_sourcefold.py",
        "assignment_guard": root
        / "trkh"
        / "tools"
        / "screen_dinov3_cgaer_b11_sourcefold.py",
    }


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = _source_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"B24 source missing: {name}={path}")
        relative = path.relative_to(root)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"B24 source is not tracked: {relative}")
    return {name: _sha256(path) for name, path in paths.items()}


def _assert_locked_runtime(args: argparse.Namespace) -> None:
    if int(args.batch_size) != BATCH_SIZE or int(args.workers) != WORKERS:
        raise ValueError("B24 is locked to batch-size=32 and workers=0")


def _build_convnext(weight: Path) -> nn.Module:
    resolved = weight.expanduser().resolve()
    if not resolved.is_file() or _sha256(resolved) != MODEL_WEIGHT_SHA256:
        raise ValueError("B24 DINOv3-ConvNeXt-T weight is missing or changed")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    state = load_file(str(resolved), device="cpu")
    model.load_state_dict(state, strict=True)
    if (
        int(getattr(model, "num_features", -1)) != FEATURE_DIM
        or sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS
    ):
        raise ValueError("B24 ConvNeXt topology changed")
    model.eval().requires_grad_(False)
    return model


def _transform() -> Any:
    return v2.Compose(
        (
            v2.Resize(
                IMAGE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
    )


def _extract(
    model: nn.Module,
    dataset: _TrainLedgerDataset,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    output = np.full((len(dataset), FEATURE_DIM), np.nan, dtype=np.float32)
    expected_index = 0
    next_report = 10
    model.to(device).eval()
    with torch.inference_mode():
        for batch_index, (images, _labels, indices) in enumerate(loader, start=1):
            index_array = indices.numpy().astype(np.int64, copy=False)
            expected = np.arange(
                expected_index, expected_index + index_array.size, dtype=np.int64
            )
            if not np.array_equal(index_array, expected):
                raise ValueError("B24 DataLoader order changed")
            values = (
                model(images.to(device=device, dtype=torch.float32, non_blocking=True))
                .float()
                .cpu()
                .numpy()
            )
            if values.shape != (index_array.size, FEATURE_DIM):
                raise ValueError(f"B24 descriptor shape changed: {values.shape}")
            output[index_array] = values
            expected_index += index_array.size
            percent = int(100 * batch_index / max(len(loader), 1))
            if percent >= next_report:
                print(f"convnext: {percent}% ({expected_index}/{len(dataset)})", flush=True)
                next_report += 10
    model.cpu()
    if expected_index != len(dataset) or not np.isfinite(output).all():
        raise RuntimeError("B24 descriptor extraction incomplete/non-finite")
    return output


def _b13_paths(root: Path) -> dict[str, Path]:
    base = root.expanduser().resolve()
    paths = {
        "dino_features": base / "train_dino_final_f32.npy",
        "paths": base / "train_paths.json",
        "oof": base / "train_oof_readouts.npz",
        "summary": base / "summary.json",
    }
    expected = {
        "dino_features": B13_DINO_FEATURE_SHA256,
        "paths": B13_PATHS_SHA256,
        "oof": B13_OOF_SHA256,
        "summary": B13_SUMMARY_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or _sha256(path) != expected[name]:
            raise ValueError(f"B24 retained B13 artifact changed: {name}")
    return paths


def _prepare_b24_ledger(args: argparse.Namespace) -> dict[str, object]:
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file() or _sha256(data_yaml) != EXPECTED_DATA_SHA256:
        raise ValueError("B24 canonical class_f data YAML changed")
    data_root = _data_root_from_yaml(data_yaml)
    artifacts = _b13_paths(args.b13_dir)
    with artifacts["paths"].open("r", encoding="utf-8") as handle:
        relative_paths = json.load(handle)
    if not isinstance(relative_paths, list) or not all(
        isinstance(path, str) for path in relative_paths
    ):
        raise ValueError("B24 retained TRAIN path ledger is invalid")
    with np.load(artifacts["oof"], allow_pickle=False) as payload:
        labels = np.asarray(payload["labels"], dtype=np.int64)
        retained_folds = np.asarray(payload["folds"], dtype=np.int64)
        retained_groups = np.asarray(payload["groups"], dtype=np.int64)
    counts = tuple(np.bincount(labels, minlength=5).astype(int).tolist())
    if len(relative_paths) != EXPECTED_TRAIN_SAMPLES or counts != EXPECTED_CLASS_COUNTS:
        raise ValueError("B24 retained TRAIN support changed")
    assignment = read_locked_assignment(args.assignment_csv, relative_paths, labels)
    if assignment["csv_sha256"] != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("B24 component-fold assignment changed")
    if not np.array_equal(assignment["folds"], retained_folds) or not np.array_equal(
        assignment["groups"], retained_groups
    ):
        raise ValueError("B24 retained OOF fold/group vectors changed")
    absolute_paths = [(data_root / path).resolve() for path in relative_paths]
    assert_train_only_paths(absolute_paths, data_root / "train")
    if any(not path.is_file() for path in absolute_paths):
        raise FileNotFoundError("B24 TRAIN ledger references a missing image")
    return {
        "data_yaml": data_yaml,
        "data_root": data_root,
        "labels": labels,
        "paths": relative_paths,
        "absolute_paths": absolute_paths,
        "assignment": assignment,
        "train_content": _train_content_contract(relative_paths, absolute_paths),
    }


def _load_reference(
    b13_dir: Path,
    ledger: Mapping[str, object],
) -> dict[str, object]:
    paths = _b13_paths(b13_dir)
    with paths["paths"].open("r", encoding="utf-8") as handle:
        retained_paths = json.load(handle)
    if retained_paths != list(ledger["paths"]):
        raise ValueError("B24 retained DINO paths do not match the canonical ledger")
    labels = np.asarray(ledger["labels"], dtype=np.int64)
    folds = np.asarray(ledger["assignment"]["folds"], dtype=np.int64)
    groups = np.asarray(ledger["assignment"]["groups"], dtype=np.int64)
    dino_features = np.load(paths["dino_features"], allow_pickle=False)
    with np.load(paths["oof"], allow_pickle=False) as payload:
        for name, expected in (("labels", labels), ("folds", folds), ("groups", groups)):
            if not np.array_equal(payload[name], expected):
                raise ValueError(f"B24 retained B13 {name} changed")
        dino_scores = np.asarray(payload["dino_scores"], dtype=np.float64)
    if dino_features.shape != (labels.size, 384) or dino_scores.shape != (labels.size, 5):
        raise ValueError("B24 retained DINO array geometry changed")
    with paths["summary"].open("r", encoding="utf-8") as handle:
        b13_summary = json.load(handle)
    reproduced = classification_summary(labels, dino_scores, folds)
    for metric in ("accuracy", "macro_f1", "class1_f1", "mean_pair_auroc"):
        if reproduced[metric] != b13_summary["metrics"]["dino"][metric]:
            raise ValueError(f"B24 failed exact B13 DINO replay: {metric}")
    return {
        "features": np.asarray(dino_features, dtype=np.float32),
        "scores": dino_scores,
        "summary": reproduced,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def assess_gate(
    *,
    dino: Mapping[str, object],
    fusion: Mapping[str, object],
    bootstrap: Mapping[str, object],
    integrity_complete: bool,
    readouts_converged: bool,
) -> dict[str, object]:
    class1_delta = float(fusion["class1_f1"]) - float(dino["class1_f1"])
    macro_delta = float(fusion["macro_f1"]) - float(dino["macro_f1"])
    pair_delta = float(fusion["mean_pair_auroc"]) - float(dino["mean_pair_auroc"])
    intervals = bootstrap["intervals"]
    dino_folds = {int(row["fold"]): row for row in dino["folds"]}
    fusion_folds = {int(row["fold"]): row for row in fusion["folds"]}
    fold_deltas = [
        float(fusion_folds[fold]["class1_f1"])
        - float(dino_folds[fold]["class1_f1"])
        for fold in range(FOLDS)
    ]
    checks = {
        "new_local_signal": class1_delta >= 0.010 or pair_delta >= 0.003,
        "macro_gain": macro_delta >= 0.002,
        "bootstrap_harm_guard": (
            float(intervals["class1_f1_delta"]["lower"]) >= -0.020
            and float(intervals["macro_f1_delta"]["lower"]) >= -0.008
            and float(intervals["mean_pair_auroc_delta"]["lower"]) >= -0.004
        ),
        "class1_recall_retention": float(fusion["class1_recall"])
        >= 0.95 * float(dino["class1_recall"]),
        "restricted_fp_control": float(fusion["restricted_fp_rate"])
        <= 1.10 * float(dino["restricted_fp_rate"]),
        "fold_stability": sum(delta >= -0.010 for delta in fold_deltas) >= 3,
        "readouts_converged": bool(readouts_converged),
        "integrity_complete_train_only": bool(integrity_complete),
    }
    passed = all(checks.values())
    return {
        "signal_gate_passed": passed,
        "teacher_distillation_design_permission": passed,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "point_deltas": {
            "class1_f1": class1_delta,
            "macro_f1": macro_delta,
            "mean_pair_auroc": pair_delta,
            "class1_recall_ratio": float(fusion["class1_recall"])
            / max(float(dino["class1_recall"]), 1e-12),
            "restricted_fp_rate_ratio": float(fusion["restricted_fp_rate"])
            / max(float(dino["restricted_fp_rate"]), 1e-12),
            "class1_f1_by_fold": fold_deltas,
        },
    }


def _run_focused_tests() -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_audit_dinov3_convnext_b24.py",
    ]
    completed = subprocess.run(
        command,
        cwd=_repository_root(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"B24 focused tests failed:\n{completed.stdout}\n{completed.stderr}")
    return {"command": command, "stdout": completed.stdout.strip(), "passed": True}


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    _assert_locked_runtime(args)
    ledger = _prepare_b24_ledger(args)
    git = _git_contract()
    if git["branch"] != "research/pretrained-classf-b1" or not git["tracked_worktree_clean"]:
        raise RuntimeError(f"B24 preflight requires the clean canonical branch: {git}")
    weight = args.convnext_weight.expanduser().resolve()
    reference = _load_reference(args.b13_dir, ledger)
    model = _build_convnext(weight)
    torch.manual_seed(SEED)
    with torch.inference_mode():
        feature = model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    tests = _run_focused_tests()
    checks = {
        "git_clean_canonical_branch": True,
        "train_ledger_exact": True,
        "validation_not_constructed": True,
        "test_not_constructed": True,
        "convnext_strict_weight": _sha256(weight) == MODEL_WEIGHT_SHA256,
        "convnext_feature_shape": tuple(feature.shape) == (1, FEATURE_DIM),
        "b13_dino_exact_replay": True,
        "focused_tests": bool(tests["passed"]),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_scope": "train_only_frozen_local_teacher_and_fusion_screen",
        "created_at_unix": time.time(),
        "source_hashes": _source_hashes(),
        "git": git,
        "dependencies": _dependency_contract(),
        "device": _device_contract(args.device),
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "train_rows": int(np.asarray(ledger["labels"]).size),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": ledger["train_content"],
        },
        "assignment": {
            key: ledger["assignment"][key]
            for key in (
                "csv_sha256",
                "assignment_int64_sha256",
                "group_vector_int64_sha256",
                "fold_rows",
            )
        },
        "weight": {
            "model": MODEL_NAME,
            "repository": MODEL_REPOSITORY,
            "path": str(weight),
            "sha256": MODEL_WEIGHT_SHA256,
            "parameters": PARAMETERS,
        },
        "retained_b13": {
            "directory": str(args.b13_dir.expanduser().resolve()),
            "artifacts": reference["artifacts"],
            "dino_metrics": reference["summary"],
        },
        "preprocessing": {
            "resize_short_side": IMAGE_SIZE,
            "center_crop": IMAGE_SIZE,
            "interpolation": "bicubic",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "descriptor": "official_head_norm_global_average_pool",
            "dimension": FEATURE_DIM,
        },
        "screen": {
            "arms": ["retained_raw_dino", "convnext", "standardized_feature_fusion"],
            "selected_gate_arm": "standardized_feature_fusion",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "focused_tests": tests,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"B24 preflight failed: {checks}")
    return payload


def _validate_preflight(
    args: argparse.Namespace, ledger: Mapping[str, object]
) -> dict[str, object]:
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B24 requires accepted preflight path and SHA-256")
    artifact = args.preflight_artifact.expanduser().resolve()
    if _sha256(artifact) != args.preflight_sha256.lower():
        raise ValueError("B24 accepted preflight SHA changed")
    with artifact.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    current_git = _git_contract()
    checks = {
        "passed": payload.get("passed") is True,
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "source_hashes": payload.get("source_hashes") == _source_hashes(),
        "git": payload.get("git") == current_git,
        "train_content": payload.get("dataset", {}).get("train_content")
        == ledger["train_content"],
        "weight": _sha256(args.convnext_weight.expanduser().resolve())
        == MODEL_WEIGHT_SHA256,
        "reference": all(_b13_paths(args.b13_dir).values()),
    }
    if not all(checks.values()):
        raise ValueError(f"B24 preflight validation failed: {checks}")
    return {"artifact": str(artifact), "sha256": args.preflight_sha256.lower(), "payload": payload}


def _formal(args: argparse.Namespace) -> dict[str, object]:
    _assert_locked_runtime(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"refuse to overwrite B24 output directory: {output_dir}")
    ledger = _prepare_b24_ledger(args)
    accepted = _validate_preflight(args, ledger)
    reference = _load_reference(args.b13_dir, ledger)
    start_source = _source_hashes()
    start_git = _git_contract()
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    model = _build_convnext(args.convnext_weight)
    dataset = _TrainLedgerDataset(ledger["absolute_paths"], ledger["labels"], _transform())
    started = time.perf_counter()
    convnext_features = _extract(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    labels = np.asarray(ledger["labels"], dtype=np.int64)
    folds = np.asarray(ledger["assignment"]["folds"], dtype=np.int64)
    groups = np.asarray(ledger["assignment"]["groups"], dtype=np.int64)
    fusion_features = np.concatenate((reference["features"], convnext_features), axis=1)
    readouts = {
        "convnext": fit_oof_readout(convnext_features, labels, folds),
        "fusion": fit_oof_readout(fusion_features, labels, folds),
    }
    metrics = {
        "dino": reference["summary"],
        "convnext": classification_summary(labels, readouts["convnext"]["scores"], folds),
        "fusion": classification_summary(labels, readouts["fusion"]["scores"], folds),
    }
    bootstrap = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=reference["scores"],
        candidate_scores=readouts["fusion"]["scores"],
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    end_source = _source_hashes()
    end_git = _git_contract()
    integrity = bool(
        np.isfinite(convnext_features).all()
        and all(np.isfinite(result["scores"]).all() for result in readouts.values())
        and start_source == end_source
        and start_git == end_git
        and end_git["tracked_worktree_clean"] is True
    )
    gate = assess_gate(
        dino=metrics["dino"],
        fusion=metrics["fusion"],
        bootstrap=bootstrap,
        integrity_complete=integrity,
        readouts_converged=all(bool(result["converged"]) for result in readouts.values()),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    descriptor_sha = _atomic_npy(
        output_dir / "train_convnext_final_f32.npy", convnext_features
    )
    oof_sha = _atomic_npz(
        output_dir / "train_oof_readouts.npz",
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=np.asarray(reference["scores"], dtype=np.float64),
        convnext_scores=np.asarray(readouts["convnext"]["scores"], dtype=np.float64),
        fusion_scores=np.asarray(readouts["fusion"]["scores"], dtype=np.float64),
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_frozen_local_teacher_and_fusion_screen",
        "source_hashes": start_source,
        "git": {"start": start_git, "end": end_git},
        "accepted_preflight": accepted,
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "train_rows": int(labels.size),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": ledger["train_content"],
        },
        "weight": accepted["payload"]["weight"],
        "retained_b13": accepted["payload"]["retained_b13"],
        "preprocessing": accepted["payload"]["preprocessing"],
        "descriptor_sha256": descriptor_sha,
        "oof_sha256": oof_sha,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "gate": gate,
        "integrity_complete": integrity,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        if args.preflight_artifact is not None or args.preflight_sha256:
            raise ValueError("B24 preflight cannot consume a prior preflight")
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise RuntimeError(f"refuse to overwrite B24 preflight directory: {output_dir}")
        payload = _preflight(args)
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(output_dir / "preflight.json", payload)
        print(json.dumps({"passed": payload["passed"], "output": str(output_dir)}, indent=2))
        return 0
    summary = _formal(args)
    print(json.dumps({"gate": summary["gate"], "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

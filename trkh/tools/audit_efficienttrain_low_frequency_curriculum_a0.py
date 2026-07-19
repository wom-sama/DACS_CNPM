from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_pixel_difference_stem_signal import (
    _build_dataset,
    _full_worktree_clean,
    _git_value,
    _load_keeper_model,
    _make_condition_loader,
    _prepare_output_dir,
    _sha256,
    _tracked_worktree_clean,
    _verify_sha256,
    region_masks,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics
from trkh.tools.replay_efficienttrain_low_frequency_curriculum_a0 import (
    canonical_close,
    replay_artifacts,
)


METHOD = "efficienttrain_low_frequency_curriculum_a0"
IMAGE_SIZE = 256
PATCH_SIZE = 16
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42
BENCHMARK_BATCH_SIZE = 32
BENCHMARK_WARMUPS = 5
BENCHMARK_REPEATS = 3
MAX_EQUATION_ERROR = 1e-6
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FIT_FOLDS = (1, 2, 3, 4)
EXPECTED_COHORT_ROWS = 607
EXPECTED_POSITIVES = 421
EXPECTED_NEGATIVES = 186
EXPECTED_FOLD_COUNTS = {
    1: {"tp": 112, "fp": 45},
    2: {"tp": 100, "fp": 48},
    3: {"tp": 101, "fp": 52},
    4: {"tp": 108, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd"
)
KNOWN_NEAR_TIE_INDEX = 3657
KNOWN_NEAR_TIE_COHORT_POSITION = 337
KNOWN_NEAR_TIE_TARGET = 2
KNOWN_NEAR_TIE_DECLARED_PREDICTION = 1
KNOWN_NEAR_TIE_REPLAY_PREDICTION = 2
SENSITIVITY_COHORT_ROWS = EXPECTED_COHORT_ROWS - 1
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
SHIFTED_CONDITIONS = tuple(name for name, _, _ in CONDITIONS if name != "clean")
VIEWS = (
    ("early_b176", 176),
    ("middle_b224", 224),
    ("native_b256", 256),
)
CROPPED_VIEWS = VIEWS[:2]
TP_QUANTILE = 0.97

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "c8cd9bcf65b297c43e7dacc5e588a82f6edccf58c53a1e625f8f8a79ff071be4"
LOCKED_PAPER_SHA256 = "3191ac4650bfea859492360ca916c83b487906e7c5eae98af803d83bdb251eae"
LOCKED_SUPPLEMENTAL_SHA256 = "aa542b054afdd2cfdc53fb4bb0cb8f7df932c37a126400d822939aba0b79c92c"
LOCKED_OFFICIAL_COMMIT = "bdefd277c71ba3bfc2a88c13768b215f15301350"
LOCKED_OFFICIAL_TREE = "5c1362c15be3695eba64d636a8019944c026ac61"
LOCKED_OFFICIAL_HASHES = {
    "utils": "64f42a102eee738f45383dbdc9be36fdb0fc960810caebf9b5b3c9ae0c42a21a",
    "engine": "2aab50e607f519aa59892acd1086fb6d33594e854f9798d4fd33be2cc6c55058",
    "training": "75c0eb344d577814b06e2539df8b77d441f3111103a2d1249d19d32f1682ba7c",
    "readme": "b0a5cc82e7d9e28afd0172d9e93948400def13491d3f2b4341f7bddb459d0861",
    "license": "63e8210e6bf3e8c032dc0c69b1d1d2e3ab72c14b02cabcc0dada2618bb188b97",
}
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_INTERRUPTED_MANIFEST_SHA256 = "1ed3b08b639eaa81645ee611b2c393388a17fcfe489e0175b6e7a4cf4e65e702"
LOCKED_INTERRUPTED_TRANSCRIPT_SHA256 = "80d636a945168f86576605db59576c3ce9e54ba9f5331a73754f9f3661275ba3"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only EfficientTrain low-frequency information gate. "
            "Validation, test, training, and checkpoint writes are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_EFFICIENTTRAIN_LOW_FREQUENCY_CURRICULUM_"
            "A0_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\EfficientTrain_ICCV2023\Wang_EfficientTrain_ICCV_2023.pdf"
        ),
    )
    parser.add_argument(
        "--supplemental",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\EfficientTrain_ICCV2023\Wang_EfficientTrain_ICCV_2023_supplemental.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\EfficientTrain_ICCV2023"
        ),
    )
    parser.add_argument(
        "--interrupted-reference",
        type=Path,
        default=Path(
            "runs/audit_efficienttrain_low_frequency_curriculum_a0_"
            "interrupted_declaration_20260719"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/audit_efficienttrain_low_frequency_curriculum_a0_20260719"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument(
        "--finalize-visual-review", action="store_true", default=False
    )
    parser.add_argument(
        "--visual-review-result", choices=("pass", "fail"), default="fail"
    )
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--benchmark-repeats", type=int, default=BENCHMARK_REPEATS
    )
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
        and int(args.seed) == SEED
        and int(args.benchmark_repeats) == BENCHMARK_REPEATS
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    repository = Path(__file__).resolve().parents[2]
    interrupted = Path(args.interrupted_reference).resolve()
    return {
        "repository": repository,
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "supplemental": Path(args.supplemental).resolve(),
        "official_root": official,
        "utils": official / "utils.py",
        "engine": official / "engine.py",
        "training": official / "ET_training.py",
        "readme": official / "README.md",
        "license": official / "LICENSE",
        "interrupted_reference": interrupted,
        "interrupted_manifest": interrupted / "failure_manifest.json",
        "interrupted_transcript": interrupted / "launcher_transcript.txt",
        "current_commands": repository
        / "docs"
        / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "command_history": repository
        / "docs"
        / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
        "auditor": Path(__file__).resolve(),
        "replay": repository
        / "trkh"
        / "tools"
        / "replay_efficienttrain_low_frequency_curriculum_a0.py",
        "tests": repository
        / "tests"
        / "test_audit_efficienttrain_low_frequency_curriculum_a0.py",
        "launcher": repository
        / "scripts"
        / "run_trkh_efficienttrain_low_frequency_curriculum_a0.ps1",
    }


def _cohort_label(row: CleanTrainRow) -> Optional[str]:
    if row.fold not in FIT_FOLDS or row.keeper_prediction != FOCUS_CLASS:
        return None
    if row.target == FOCUS_CLASS:
        return "tp"
    if row.target in RESTRICTED_NEGATIVE_CLASSES:
        return "fp"
    return None


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], list[CleanTrainRow]]:
    if not _locked_args_exact(args):
        raise ValueError(
            "Arguments differ from the prospectively locked EfficientTrain A0."
        )
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"
        ),
        "launcher_args": _verify_sha256(
            paths["launcher_args"],
            LOCKED_LAUNCHER_ARGS_SHA256,
            "keeper launcher args",
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(
            paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "EfficientTrain protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "accepted EfficientTrain paper"
        ),
        "supplemental": _verify_sha256(
            paths["supplemental"],
            LOCKED_SUPPLEMENTAL_SHA256,
            "EfficientTrain supplemental",
        ),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
        "interrupted_manifest": _verify_sha256(
            paths["interrupted_manifest"],
            LOCKED_INTERRUPTED_MANIFEST_SHA256,
            "declaration-only interruption manifest",
        ),
        "interrupted_transcript": _verify_sha256(
            paths["interrupted_transcript"],
            LOCKED_INTERRUPTED_TRANSCRIPT_SHA256,
            "declaration-only interruption transcript",
        ),
    }
    for name, expected in LOCKED_OFFICIAL_HASHES.items():
        hashes[name] = _verify_sha256(
            paths[name], expected, f"official EfficientTrain {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(
        paths["official_root"], "rev-parse", "HEAD^{tree}"
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official commit differs: {official_commit} != {LOCKED_OFFICIAL_COMMIT}"
        )
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(
            f"Official tree differs: {official_tree} != {LOCKED_OFFICIAL_TREE}"
        )
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official EfficientTrain worktree must be clean.")

    cidt_summary = json.loads(
        paths["cidt_summary"].read_text(encoding="utf-8")
    )
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test-data use.")
    if bool(
        cidt_summary.get(
            "validation_predictions_used",
            cidt_summary.get("validation_data_used", True),
        )
    ):
        raise ValueError("CIDT provenance indicates validation-data use.")
    interrupted = json.loads(
        paths["interrupted_manifest"].read_text(encoding="utf-8")
    )
    expected_interruption = {
        "sample_index": KNOWN_NEAR_TIE_INDEX,
        "target": KNOWN_NEAR_TIE_TARGET,
        "expected_prediction": KNOWN_NEAR_TIE_DECLARED_PREDICTION,
        "observed_prediction": KNOWN_NEAR_TIE_REPLAY_PREDICTION,
    }
    failure = interrupted.get("failure", {})
    if any(
        int(failure.get(key, -1)) != value
        for key, value in expected_interruption.items()
    ):
        raise ValueError("Interrupted declaration tuple differs from erratum.")
    if bool(interrupted.get("candidate_b176_b224_inference_run", True)) or bool(
        interrupted.get("candidate_behavioral_metrics_read", True)
    ):
        raise ValueError("Interrupted reference contains candidate behavior.")

    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohort = [row for row in rows if _cohort_label(row) is not None]
    positives = sum(_cohort_label(row) == "tp" for row in cohort)
    negatives = sum(_cohort_label(row) == "fp" for row in cohort)
    if (len(cohort), positives, negatives) != (
        EXPECTED_COHORT_ROWS,
        EXPECTED_POSITIVES,
        EXPECTED_NEGATIVES,
    ):
        raise ValueError(
            "Locked cohort differs: "
            f"rows/TP/FP={len(cohort)}/{positives}/{negatives}."
        )
    fold_counts: Dict[int, Dict[str, int]] = {
        fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS
    }
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked cohort fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256(
        [row.sample_index for row in cohort]
    )
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    repository = paths["repository"]
    if not _tracked_worktree_clean(repository):
        raise ValueError(
            "Tracked TRKH worktree must be clean for EfficientTrain A0."
        )
    repository_commit = _git_value(repository, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repository, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError(
            "EfficientTrain A0 requires the exact pushed repository commit."
        )
    implementation: Dict[str, object] = {}
    for name in ("auditor", "replay", "tests", "launcher"):
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"Locked implementation file missing: {path}")
        implementation[name] = {
            "path": str(path.relative_to(repository)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
    return (
        {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "official_worktree_clean": True,
            "repository_commit": repository_commit,
            "upstream_commit": upstream_commit,
            "tracked_worktree_clean": True,
            "implementation": implementation,
            "ordered_cohort_index_sha256": ordered_hash,
            "cohort_rows": len(cohort),
            "positive_rows": positives,
            "negative_rows": negatives,
            "fold_counts": fold_counts,
            "validation_predictions_used": False,
            "holdout_data_used": False,
            "test_data_used": False,
            "training_used": False,
            "declaration_erratum": {
                "known_near_tie_index": KNOWN_NEAR_TIE_INDEX,
                "known_near_tie_target": KNOWN_NEAR_TIE_TARGET,
                "declared_prediction": KNOWN_NEAR_TIE_DECLARED_PREDICTION,
                "replayed_prediction": KNOWN_NEAR_TIE_REPLAY_PREDICTION,
                "interrupted_manifest_sha256": hashes[
                    "interrupted_manifest"
                ],
                "candidate_behavior_observed_in_interruption": False,
            },
        },
        rows,
        cohort,
    )


def _validate_frequency_request(images: Tensor, band_width: int) -> tuple[int, int]:
    if images.ndim != 4 or int(images.size(1)) != 3:
        raise ValueError("EfficientTrain crop requires an NCHW RGB tensor.")
    height, width = int(images.size(-2)), int(images.size(-1))
    band = int(band_width)
    if height != width or band <= 0 or band > height or band % 2:
        raise ValueError(
            f"Invalid square frequency crop H={height}, W={width}, B={band}."
        )
    if images.dtype != torch.float32:
        raise ValueError("Locked EfficientTrain FFT input must be FP32.")
    return height, band


def efficienttrain_freq_crop(images: Tensor, band_width: int) -> Tensor:
    image_size, band = _validate_frequency_request(images, band_width)
    if band == image_size:
        return images
    half = band // 2
    spectrum = torch.fft.fft2(images)
    cropped = torch.empty(
        (images.size(0), images.size(1), band, band),
        dtype=spectrum.dtype,
        device=spectrum.device,
    )
    cropped[:, :, :half, :half] = spectrum[:, :, :half, :half]
    cropped[:, :, -half:, :half] = spectrum[:, :, -half:, :half]
    cropped[:, :, :half, -half:] = spectrum[:, :, :half, -half:]
    cropped[:, :, -half:, -half:] = spectrum[:, :, -half:, -half:]
    cropped = cropped * ((band / image_size) ** 2)
    return torch.fft.ifft2(cropped).real


def independent_frequency_crop_oracle(
    images: Tensor, band_width: int
) -> Tensor:
    image_size, band = _validate_frequency_request(images, band_width)
    if band == image_size:
        return images
    shifted = torch.fft.fftshift(
        torch.fft.fft2(images), dim=(-2, -1)
    )
    start = (image_size - band) // 2
    centered = shifted[:, :, start : start + band, start : start + band]
    cropped = torch.fft.ifftshift(centered, dim=(-2, -1))
    cropped = cropped * ((band / image_size) ** 2)
    return torch.fft.ifft2(cropped).real


def same_size_low_frequency_reconstruction(
    images: Tensor, band_width: int
) -> tuple[Tensor, Tensor]:
    image_size, band = _validate_frequency_request(images, band_width)
    if band == image_size:
        return images, torch.zeros_like(images)
    half = band // 2
    spectrum = torch.fft.fft2(images)
    retained = torch.zeros_like(spectrum)
    retained[:, :, :half, :half] = spectrum[:, :, :half, :half]
    retained[:, :, -half:, :half] = spectrum[:, :, -half:, :half]
    retained[:, :, :half, -half:] = spectrum[:, :, :half, -half:]
    retained[:, :, -half:, -half:] = spectrum[:, :, -half:, -half:]
    low_frequency = torch.fft.ifft2(retained).real
    return low_frequency, images - low_frequency


def independent_same_size_reconstruction(
    images: Tensor, band_width: int
) -> tuple[Tensor, Tensor]:
    image_size, band = _validate_frequency_request(images, band_width)
    if band == image_size:
        return images, torch.zeros_like(images)
    shifted = torch.fft.fftshift(
        torch.fft.fft2(images), dim=(-2, -1)
    )
    start = (image_size - band) // 2
    retained_shifted = torch.zeros_like(shifted)
    retained_shifted[
        :, :, start : start + band, start : start + band
    ] = shifted[:, :, start : start + band, start : start + band]
    retained = torch.fft.ifftshift(retained_shifted, dim=(-2, -1))
    low_frequency = torch.fft.ifft2(retained).real
    return low_frequency, images - low_frequency


def _load_official_freq_crop(path: Path) -> Callable[[Tensor, int], Tensor]:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "freq_crop"
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ValueError("Official utils.py must contain exactly one freq_crop.")
    isolated = ast.Module(body=[matches[0]], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: Dict[str, object] = {"torch": torch}
    exec(compile(isolated, str(path), "exec"), namespace)
    function = namespace.get("freq_crop")
    if not callable(function):
        raise RuntimeError("Official freq_crop extraction failed.")
    return function


def _cropped_spectrum(images: Tensor, band_width: int) -> Tensor:
    image_size, band = _validate_frequency_request(images, band_width)
    half = band // 2
    source = torch.fft.fft2(images)
    result = torch.empty(
        (images.size(0), images.size(1), band, band),
        dtype=source.dtype,
        device=source.device,
    )
    result[:, :, :half, :half] = source[:, :, :half, :half]
    result[:, :, -half:, :half] = source[:, :, -half:, :half]
    result[:, :, :half, -half:] = source[:, :, :half, -half:]
    result[:, :, -half:, -half:] = source[:, :, -half:, -half:]
    return result * ((band / image_size) ** 2)


def _independent_cropped_spectrum(images: Tensor, band_width: int) -> Tensor:
    image_size, band = _validate_frequency_request(images, band_width)
    shifted = torch.fft.fftshift(
        torch.fft.fft2(images), dim=(-2, -1)
    )
    start = (image_size - band) // 2
    centered = shifted[:, :, start : start + band, start : start + band]
    return torch.fft.ifftshift(centered, dim=(-2, -1)) * (
        (band / image_size) ** 2
    )


def _equation_diagnostics(official_utils: Path) -> Dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    images = torch.randn(
        (2, 3, IMAGE_SIZE, IMAGE_SIZE),
        generator=generator,
        dtype=torch.float32,
    )
    official = _load_official_freq_crop(official_utils)
    cases: Dict[str, object] = {}
    for view, band in CROPPED_VIEWS:
        official_output = official(images, band)
        candidate = efficienttrain_freq_crop(images, band)
        repeated = efficienttrain_freq_crop(images, band)
        oracle = independent_frequency_crop_oracle(images, band)
        spectrum = _cropped_spectrum(images, band)
        oracle_spectrum = _independent_cropped_spectrum(images, band)
        low_frequency, residual = same_size_low_frequency_reconstruction(
            images, band
        )
        oracle_low, oracle_residual = independent_same_size_reconstruction(
            images, band
        )
        cases[view] = {
            "band_width": band,
            "scale_factor": float((band / IMAGE_SIZE) ** 2),
            "official_candidate_max_abs_error": float(
                (official_output - candidate).abs().max()
            ),
            "candidate_oracle_max_abs_error": float(
                (candidate - oracle).abs().max()
            ),
            "spectral_corner_scale_max_abs_error": float(
                (spectrum - oracle_spectrum).abs().max()
            ),
            "same_size_oracle_max_abs_error": float(
                (low_frequency - oracle_low).abs().max()
            ),
            "residual_oracle_max_abs_error": float(
                (residual - oracle_residual).abs().max()
            ),
            "reconstruction_max_abs_error": float(
                (low_frequency + residual - images).abs().max()
            ),
            "deterministic_two_pass_bit_exact": torch.equal(
                candidate, repeated
            ),
            "all_outputs_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    official_output,
                    candidate,
                    oracle,
                    low_frequency,
                    residual,
                )
            ),
        }
    identity = efficienttrain_freq_crop(images, IMAGE_SIZE)
    return {
        "official_function_extracted_with_ast": True,
        "cases": cases,
        "maximum_official_candidate_error": max(
            float(value["official_candidate_max_abs_error"])
            for value in cases.values()
        ),
        "maximum_candidate_oracle_error": max(
            float(value["candidate_oracle_max_abs_error"])
            for value in cases.values()
        ),
        "maximum_spectral_replay_error": max(
            float(value["spectral_corner_scale_max_abs_error"])
            for value in cases.values()
        ),
        "maximum_same_size_oracle_error": max(
            float(value["same_size_oracle_max_abs_error"])
            for value in cases.values()
        ),
        "maximum_reconstruction_error": max(
            float(value["reconstruction_max_abs_error"])
            for value in cases.values()
        ),
        "b256_identity_bit_exact": torch.equal(identity, images),
        "b256_same_storage": identity.data_ptr() == images.data_ptr(),
    }


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _rng_snapshot() -> Dict[str, object]:
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "cpu": _tensor_sha256(torch.random.get_rng_state()),
        "cuda": [_tensor_sha256(value) for value in cuda_states],
    }


def _gpu_snapshot() -> Dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        if len(values) != 4:
            raise ValueError(result.stdout)
        return {
            "available": True,
            "utilization_percent": int(values[0]),
            "memory_used_mib": int(values[1]),
            "memory_total_mib": int(values[2]),
            "temperature_c": int(values[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "error": str(error)}


def _forward_logits_and_features(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> tuple[Tensor, Mapping[str, object]]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits, features = _forward_classification_with_metadata(
            model, images, metadata, device=device
        )
    if not torch.is_tensor(logits) or logits.ndim != 2 or logits.size(1) != 5:
        raise ValueError("Keeper forward did not produce [B,5] logits.")
    if not isinstance(features, Mapping):
        raise ValueError("Keeper forward did not expose dynamic feature metadata.")
    logits = logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Keeper forward produced non-finite logits.")
    return logits, features


def _declaration_mismatch_policy(
    mismatches: Sequence[Mapping[str, int]],
) -> Dict[str, object]:
    expected = {
        "expected_sample_index": KNOWN_NEAR_TIE_INDEX,
        "observed_sample_index": KNOWN_NEAR_TIE_INDEX,
        "expected_target": KNOWN_NEAR_TIE_TARGET,
        "observed_target": KNOWN_NEAR_TIE_TARGET,
        "expected_prediction": KNOWN_NEAR_TIE_DECLARED_PREDICTION,
        "observed_prediction": KNOWN_NEAR_TIE_REPLAY_PREDICTION,
    }
    exact_known_exception = bool(
        len(mismatches) == 1
        and all(
            int(mismatches[0].get(key, -1)) == value
            for key, value in expected.items()
        )
    )
    return {
        "mismatch_count": len(mismatches),
        "exact_known_exception": exact_known_exception,
        "policy_passed": exact_known_exception,
        "expected_exception": expected,
    }


def _collect_native_declarations(
    *,
    model: nn.Module,
    loader,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
) -> Dict[str, object]:
    logits_out: list[Tensor] = []
    probabilities_out: list[Tensor] = []
    indices_out: list[int] = []
    targets_out: list[int] = []
    predictions_out: list[int] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logits, _ = _forward_logits_and_features(
                model, images, metadata, device=device
            )
            probabilities = F.softmax(logits, dim=1)
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Declaration loader lacks sample_index.")
            logits_out.append(logits.cpu())
            probabilities_out.append(probabilities.cpu())
            indices_out.extend(int(value) for value in sample_indices.tolist())
            targets_out.extend(int(value) for value in targets.tolist())
            predictions_out.extend(
                int(value) for value in logits.argmax(dim=1).cpu().tolist()
            )
    if len(indices_out) != len(cohort):
        raise ValueError("Clean native declaration row count differs.")
    mismatches = [
        {
            "position": position,
            "expected_sample_index": source.sample_index,
            "observed_sample_index": indices_out[position],
            "expected_target": source.target,
            "observed_target": targets_out[position],
            "expected_prediction": source.keeper_prediction,
            "observed_prediction": predictions_out[position],
        }
        for position, source in enumerate(cohort)
        if indices_out[position] != source.sample_index
        or targets_out[position] != source.target
        or predictions_out[position] != source.keeper_prediction
    ]
    policy = _declaration_mismatch_policy(mismatches)
    if not bool(policy["policy_passed"]):
        raise ValueError(
            "Clean native declaration mismatch policy failed before candidate metrics: "
            f"{mismatches[:10]}"
        )
    return {
        "rows": len(indices_out),
        "indices": indices_out,
        "targets": targets_out,
        "predictions": predictions_out,
        "logits": torch.cat(logits_out, dim=0).numpy(),
        "probabilities": torch.cat(probabilities_out, dim=0).numpy(),
        "exact": not mismatches,
        "mismatches": mismatches,
        "mismatch_policy": policy,
        "candidate_metric_read_before_declaration": False,
    }


def _new_geometry_accumulator() -> Dict[str, object]:
    return {
        "grid_sizes": set(),
        "prefix_counts": set(),
        "kept_patch_counts": set(),
        "original_patch_counts": set(),
        "positional_token_counts": set(),
        "minimum_patch_index": None,
        "maximum_patch_index": None,
        "batches": 0,
        "finite_logits": True,
        "indices_in_grid": True,
        "positional_count_exact": True,
    }


def _update_geometry(
    accumulator: Dict[str, object],
    *,
    model: nn.Module,
    features: Mapping[str, object],
    logits: Tensor,
) -> None:
    grid = features.get("grid_size")
    patches = features.get("patches")
    tokens = features.get("tokens")
    patch_indices = features.get("patch_indices")
    if (
        not isinstance(grid, (tuple, list))
        or len(grid) != 2
        or not torch.is_tensor(patches)
        or not torch.is_tensor(tokens)
        or not torch.is_tensor(patch_indices)
    ):
        raise ValueError("Dynamic forward metadata is incomplete.")
    grid_tuple = (int(grid[0]), int(grid[1]))
    grid_cells = int(grid_tuple[0] * grid_tuple[1])
    prefix_count = int(tokens.size(1) - patches.size(1))
    kept_count = int(patch_indices.size(1))
    pos_embed = model.get_interpolated_pos_embed(grid_tuple)
    register_positions = (
        int(getattr(model, "num_registers", 0))
        if bool(getattr(model, "register_positional_embedding", False))
        else 0
    )
    expected_positional = 1 + register_positions + grid_cells
    minimum = int(patch_indices.min().item())
    maximum = int(patch_indices.max().item())
    accumulator["grid_sizes"].add(grid_tuple)
    accumulator["prefix_counts"].add(prefix_count)
    accumulator["kept_patch_counts"].add(kept_count)
    accumulator["original_patch_counts"].add(grid_cells)
    accumulator["positional_token_counts"].add(int(pos_embed.size(1)))
    previous_minimum = accumulator["minimum_patch_index"]
    previous_maximum = accumulator["maximum_patch_index"]
    accumulator["minimum_patch_index"] = (
        minimum if previous_minimum is None else min(previous_minimum, minimum)
    )
    accumulator["maximum_patch_index"] = (
        maximum if previous_maximum is None else max(previous_maximum, maximum)
    )
    accumulator["batches"] = int(accumulator["batches"]) + 1
    accumulator["finite_logits"] = bool(
        accumulator["finite_logits"] and torch.isfinite(logits).all()
    )
    accumulator["indices_in_grid"] = bool(
        accumulator["indices_in_grid"]
        and minimum >= 0
        and maximum < grid_cells
    )
    accumulator["positional_count_exact"] = bool(
        accumulator["positional_count_exact"]
        and int(pos_embed.size(1)) == expected_positional
    )


def _finalize_geometry(
    accumulators: Mapping[str, Mapping[str, object]], model: nn.Module
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for view, band in VIEWS:
        source = accumulators[view]
        expected_grid = (band // PATCH_SIZE, band // PATCH_SIZE)
        output[view] = {
            "band_width": band,
            "expected_grid": list(expected_grid),
            "grid_sizes": [list(value) for value in sorted(source["grid_sizes"])],
            "prefix_counts": sorted(int(value) for value in source["prefix_counts"]),
            "expected_prefix_count": int(getattr(model, "num_prefix_tokens")),
            "kept_patch_counts": sorted(
                int(value) for value in source["kept_patch_counts"]
            ),
            "original_patch_counts": sorted(
                int(value) for value in source["original_patch_counts"]
            ),
            "positional_token_counts": sorted(
                int(value) for value in source["positional_token_counts"]
            ),
            "minimum_patch_index": int(source["minimum_patch_index"]),
            "maximum_patch_index": int(source["maximum_patch_index"]),
            "batches": int(source["batches"]),
            "finite_logits": bool(source["finite_logits"]),
            "indices_in_grid": bool(source["indices_in_grid"]),
            "positional_count_exact": bool(source["positional_count_exact"]),
            "grid_exact": source["grid_sizes"] == {expected_grid},
            "prefix_count_exact": source["prefix_counts"]
            == {int(getattr(model, "num_prefix_tokens"))},
        }
    return output


def _masked_rms(activation: Tensor, mask: Tensor) -> Tensor:
    expanded = mask[:, None].to(dtype=activation.dtype)
    denominator = expanded.sum(dim=(1, 2, 3)) * float(activation.size(1))
    if bool((denominator <= 0).any()):
        raise ValueError("Masked RMS received an empty region.")
    return (
        (activation.square() * expanded).sum(dim=(1, 2, 3)) / denominator
    ).clamp_min(0.0).sqrt()


def _collect_conditions(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    declarations: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], Dict[str, object]]:
    indices = [row.sample_index for row in cohort]
    records: list[Dict[str, object]] = []
    mechanism_rows: list[Dict[str, object]] = []
    geometry = {view: _new_geometry_accumulator() for view, _ in VIEWS}
    loader_summaries: Dict[str, object] = {}
    native_control_max_abs_error = 0.0
    b256_input_bit_exact = True
    expected_declaration_logits = np.asarray(declarations["logits"])

    for condition, brightness, contrast in CONDITIONS:
        loader, loader_summary = _make_condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=indices,
            brightness=brightness,
            contrast=contrast,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"efficienttrain_a0_{condition}",
        )
        loader_summaries[condition] = loader_summary
        offset = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for images, targets, metadata in loader:
                images = images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                batch_rows = int(images.size(0))
                sample_indices = metadata.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(
                    crop_bboxes
                ):
                    raise ValueError(
                        "Condition loader lacks sample_index or crop_bbox."
                    )
                expected_rows = cohort[offset : offset + batch_rows]
                observed_indices = [int(value) for value in sample_indices.tolist()]
                observed_targets = [int(value) for value in targets.tolist()]
                if observed_indices != [row.sample_index for row in expected_rows]:
                    raise ValueError(f"Condition {condition} changed row identity.")
                if observed_targets != [row.target for row in expected_rows]:
                    raise ValueError(f"Condition {condition} changed targets.")

                view_values: Dict[str, Dict[str, Tensor]] = {}
                for view, band in VIEWS:
                    transformed = efficienttrain_freq_crop(images, band)
                    if band == IMAGE_SIZE:
                        b256_input_bit_exact = bool(
                            b256_input_bit_exact and torch.equal(transformed, images)
                        )
                    transformed = transformed.detach()
                    logits, features = _forward_logits_and_features(
                        model, transformed, metadata, device=device
                    )
                    probabilities = F.softmax(logits, dim=1)
                    _update_geometry(
                        geometry[view],
                        model=model,
                        features=features,
                        logits=logits,
                    )
                    view_values[view] = {
                        "logits": logits.cpu(),
                        "probabilities": probabilities.cpu(),
                        "predictions": logits.argmax(dim=1).cpu(),
                    }

                if condition == "clean":
                    declaration_slice = expected_declaration_logits[
                        offset : offset + batch_rows
                    ]
                    native_control_max_abs_error = max(
                        native_control_max_abs_error,
                        float(
                            np.max(
                                np.abs(
                                    view_values["native_b256"]["logits"].numpy()
                                    - declaration_slice
                                )
                            )
                        ),
                    )

                crop_bboxes_device = crop_bboxes[:, :4].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                core, boundary, outside = region_masks(
                    crop_bboxes_device, IMAGE_SIZE, IMAGE_SIZE
                )
                object_mask = core | boundary
                object_pixels = object_mask.flatten(1).sum(dim=1)
                outside_pixels = outside.flatten(1).sum(dim=1)
                mechanism_batch: Dict[str, Dict[str, Tensor]] = {}
                for view, band in CROPPED_VIEWS:
                    _, residual = same_size_low_frequency_reconstruction(
                        images, band
                    )
                    mechanism_batch[view] = {
                        "object": _masked_rms(residual, object_mask).cpu(),
                        "outside": _masked_rms(residual, outside).cpu(),
                    }

                for local_position, source in enumerate(expected_rows):
                    cohort_position = offset + local_position
                    record: Dict[str, object] = {
                        "condition": condition,
                        "cohort_position": cohort_position,
                        "sample_index": source.sample_index,
                        "source_stem": source.source_stem,
                        "image_path": str(source.image_path),
                        "fold": source.fold,
                        "target": source.target,
                        "cohort": str(_cohort_label(source)),
                    }
                    for view, _ in VIEWS:
                        values = view_values[view]
                        logits = values["logits"][local_position]
                        probabilities = values["probabilities"][local_position]
                        for class_index in range(5):
                            record[f"{view}_logit_{class_index}"] = float(
                                logits[class_index]
                            )
                            record[f"{view}_prob_{class_index}"] = float(
                                probabilities[class_index]
                            )
                        record[f"{view}_prediction"] = int(
                            values["predictions"][local_position]
                        )
                    native_p1 = float(record["native_b256_prob_1"])
                    for view, _ in CROPPED_VIEWS:
                        record[f"{view}_suppression_score"] = native_p1 - float(
                            record[f"{view}_prob_1"]
                        )
                    records.append(record)
                    for view, _ in CROPPED_VIEWS:
                        mechanism_rows.append(
                            {
                                "condition": condition,
                                "view": view,
                                "cohort_position": cohort_position,
                                "sample_index": source.sample_index,
                                "fold": source.fold,
                                "target": source.target,
                                "cohort": str(_cohort_label(source)),
                                "object_residual_rms": float(
                                    mechanism_batch[view]["object"][local_position]
                                ),
                                "outside_residual_rms": float(
                                    mechanism_batch[view]["outside"][local_position]
                                ),
                                "object_pixels": int(object_pixels[local_position]),
                                "outside_pixels": int(outside_pixels[local_position]),
                            }
                        )
                offset += batch_rows
                if offset % 256 < batch_rows or offset == len(cohort):
                    print(
                        json.dumps(
                            {
                                "condition": condition,
                                "processed": offset,
                                "rows": len(cohort),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
        if offset != len(cohort):
            raise ValueError(f"Condition {condition} row count differs: {offset}")
    return records, mechanism_rows, {
        "loader_summaries": loader_summaries,
        "geometry": _finalize_geometry(geometry, model),
        "native_b256_input_bit_exact": b256_input_bit_exact,
        "native_b256_declaration_logit_max_abs_error": native_control_max_abs_error,
    }


def _condition_records(
    records: Sequence[Mapping[str, object]],
    condition: str,
    *,
    expected_rows: int = EXPECTED_COHORT_ROWS,
) -> list[Mapping[str, object]]:
    selected = [row for row in records if str(row["condition"]) == condition]
    selected.sort(key=lambda row: int(row["cohort_position"]))
    if len(selected) != int(expected_rows):
        raise ValueError(
            f"Condition {condition} has {len(selected)} prediction rows."
        )
    expected_positions = _expected_cohort_positions(expected_rows)
    positions = [int(row["cohort_position"]) for row in selected]
    if positions != expected_positions:
        raise ValueError(f"Condition {condition} has noncanonical cohort positions.")
    return selected


def _expected_cohort_positions(expected_rows: int) -> list[int]:
    if int(expected_rows) == EXPECTED_COHORT_ROWS:
        return list(range(EXPECTED_COHORT_ROWS))
    if int(expected_rows) == SENSITIVITY_COHORT_ROWS:
        return [
            position
            for position in range(EXPECTED_COHORT_ROWS)
            if position != KNOWN_NEAR_TIE_COHORT_POSITION
        ]
    raise ValueError(f"Unsupported expected cohort row count: {expected_rows}.")


def _fit_thresholds(
    records: Sequence[Mapping[str, object]],
    view: str,
    *,
    expected_rows: int = EXPECTED_COHORT_ROWS,
) -> Dict[int, float]:
    clean = _condition_records(records, "clean", expected_rows=expected_rows)
    score_key = f"{view}_suppression_score"
    thresholds: Dict[int, float] = {}
    for held_fold in FIT_FOLDS:
        fit_values = np.asarray(
            [
                float(row[score_key])
                for row in clean
                if str(row["cohort"]) == "tp"
                and int(row["fold"]) != held_fold
            ],
            dtype=np.float64,
        )
        if fit_values.size == 0:
            raise ValueError(f"Fold {held_fold} has no clean fit TP values.")
        thresholds[held_fold] = float(
            np.quantile(fit_values, TP_QUANTILE, method="higher")
        )
    return thresholds


def _hard_decision_summary(
    clean: Sequence[Mapping[str, object]], view: str
) -> Dict[str, object]:
    corrections = 0
    harms = 0
    tp_breaks = 0
    fp_removals = 0
    transitions: Counter[str] = Counter()
    for row in clean:
        target = int(row["target"])
        native = int(row["native_b256_prediction"])
        candidate = int(row[f"{view}_prediction"])
        transitions[f"{native}->{candidate}"] += 1
        corrections += int(native != target and candidate == target)
        harms += int(native == target and candidate != target)
        tp_breaks += int(
            str(row["cohort"]) == "tp" and native == 1 and candidate != 1
        )
        fp_removals += int(
            str(row["cohort"]) == "fp" and native == 1 and candidate != 1
        )
    return {
        "corrections": corrections,
        "harms": harms,
        "net_corrections": corrections - harms,
        "class1_tp_breaks": tp_breaks,
        "restricted_fp_removals": fp_removals,
        "transitions": dict(sorted(transitions.items())),
    }


def _summarize_view_and_assign_thresholds(
    records: Sequence[Dict[str, object]],
    view: str,
    *,
    fixed_thresholds: Optional[Mapping[int, float]] = None,
    expected_rows: int = EXPECTED_COHORT_ROWS,
) -> Dict[str, object]:
    thresholds = (
        {int(key): float(value) for key, value in fixed_thresholds.items()}
        if fixed_thresholds is not None
        else _fit_thresholds(records, view, expected_rows=expected_rows)
    )
    score_key = f"{view}_suppression_score"
    threshold_key = f"{view}_oof_threshold"
    rejected_key = f"{view}_oof_rejected"
    for row in records:
        threshold = thresholds[int(row["fold"])]
        row[threshold_key] = threshold
        row[rejected_key] = int(float(row[score_key]) > threshold)

    conditions: Dict[str, object] = {}
    for condition, _, _ in CONDITIONS:
        selected = _condition_records(
            records, condition, expected_rows=expected_rows
        )
        labels = np.asarray(
            [int(str(row["cohort"]) == "fp") for row in selected],
            dtype=np.int64,
        )
        scores = np.asarray(
            [float(row[score_key]) for row in selected], dtype=np.float64
        )
        rejected = np.asarray(
            [bool(int(row[rejected_key])) for row in selected], dtype=np.bool_
        )
        tp_mask = labels == 0
        fp_mask = labels == 1
        conditions[condition] = {
            "rows": len(selected),
            "auroc_fp_vs_tp": float(roc_auc_score(labels, scores)),
            "tp_retention": float(np.mean(~rejected[tp_mask])),
            "restricted_fp_rejection": float(np.mean(rejected[fp_mask])),
            "tp_breaks": int(rejected[tp_mask].sum()),
            "restricted_fp_removals": int(rejected[fp_mask].sum()),
            "tp_suppression_median": float(np.median(scores[tp_mask])),
            "fp_suppression_median": float(np.median(scores[fp_mask])),
        }

    clean = _condition_records(records, "clean", expected_rows=expected_rows)
    fold_metrics: Dict[str, object] = {}
    for fold in FIT_FOLDS:
        selected = [row for row in clean if int(row["fold"]) == fold]
        tp = [row for row in selected if str(row["cohort"]) == "tp"]
        fp = [row for row in selected if str(row["cohort"]) == "fp"]
        tp_breaks = sum(int(row[rejected_key]) for row in tp)
        fp_removals = sum(int(row[rejected_key]) for row in fp)
        fold_metrics[str(fold)] = {
            "threshold": thresholds[fold],
            "fit_tp_rows": sum(
                str(row["cohort"]) == "tp" and int(row["fold"]) != fold
                for row in clean
            ),
            "holdout_tp_rows": len(tp),
            "holdout_fp_rows": len(fp),
            "tp_breaks": int(tp_breaks),
            "restricted_fp_removals": int(fp_removals),
            "tp_retention": float(1.0 - tp_breaks / len(tp)),
            "restricted_fp_rejection": float(fp_removals / len(fp)),
        }
    return {
        "threshold_quantile": TP_QUANTILE,
        "threshold_method": "higher",
        "fold_thresholds": {
            str(key): value for key, value in thresholds.items()
        },
        "conditions": conditions,
        "clean_folds": fold_metrics,
        "positive_clean_fp_rejection_fold_count": sum(
            float(value["restricted_fp_rejection"]) > 0.0
            for value in fold_metrics.values()
        ),
        "hard_decisions_clean": _hard_decision_summary(clean, view),
        "csv_reconstruction": {
            "maximum_suppression_score_error": 0.0,
            "maximum_threshold_error": 0.0,
            "rejection_mismatches": 0,
        },
    }


def _summarize_views(
    records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        view: _summarize_view_and_assign_thresholds(records, view)
        for view, _ in CROPPED_VIEWS
    }


def _summarize_known_near_tie_exclusion(
    records: Sequence[Dict[str, object]],
    full_view_metrics: Mapping[str, object],
) -> Dict[str, object]:
    filtered = [
        row for row in records if int(row["sample_index"]) != KNOWN_NEAR_TIE_INDEX
    ]
    if len(filtered) != len(CONDITIONS) * SENSITIVITY_COHORT_ROWS:
        raise ValueError("Known near-tie exclusion prediction count differs.")
    views: Dict[str, object] = {}
    for view, _ in CROPPED_VIEWS:
        thresholds = {
            int(key): float(value)
            for key, value in full_view_metrics[view]["fold_thresholds"].items()
        }
        views[view] = _summarize_view_and_assign_thresholds(
            filtered,
            view,
            fixed_thresholds=thresholds,
            expected_rows=SENSITIVITY_COHORT_ROWS,
        )
    return {
        "excluded_sample_index": KNOWN_NEAR_TIE_INDEX,
        "rows_per_condition": SENSITIVITY_COHORT_ROWS,
        "thresholds_refit": False,
        "view_metrics": views,
        "information_gate": assess_information_gate(views),
    }


def assess_information_gate(
    view_metrics: Mapping[str, object]
) -> Dict[str, object]:
    early = view_metrics["early_b176"]
    middle = view_metrics["middle_b224"]
    conditions = early["conditions"]
    clean = conditions["clean"]
    hard = early["hard_decisions_clean"]
    checks = {
        "clean_auroc_gte_0p65": float(clean["auroc_fp_vs_tp"]) >= 0.65,
        "every_shifted_auroc_gte_0p60": all(
            float(conditions[name]["auroc_fp_vs_tp"]) >= 0.60
            for name in SHIFTED_CONDITIONS
        ),
        "four_condition_mean_auroc_gte_0p65": float(
            np.mean(
                [
                    float(conditions[name]["auroc_fp_vs_tp"])
                    for name, _, _ in CONDITIONS
                ]
            )
        )
        >= 0.65,
        "clean_tp_retention_gte_0p97": float(clean["tp_retention"]) >= 0.97,
        "clean_fp_rejection_gte_0p10": float(
            clean["restricted_fp_rejection"]
        )
        >= 0.10,
        "every_shifted_tp_retention_gte_0p93": all(
            float(conditions[name]["tp_retention"]) >= 0.93
            for name in SHIFTED_CONDITIONS
        ),
        "every_shifted_fp_rejection_gte_0p05": all(
            float(conditions[name]["restricted_fp_rejection"]) >= 0.05
            for name in SHIFTED_CONDITIONS
        ),
        "positive_clean_fp_rejection_in_gte_3_folds": int(
            early["positive_clean_fp_rejection_fold_count"]
        )
        >= 3,
        "clean_fp_removals_exceed_tp_breaks_by_gte_5": int(
            clean["restricted_fp_removals"]
        )
        - int(clean["tp_breaks"])
        >= 5,
        "early_clean_auroc_not_below_middle": float(clean["auroc_fp_vs_tp"])
        >= float(middle["conditions"]["clean"]["auroc_fp_vs_tp"]),
        "early_clean_fp_rejection_not_below_middle": float(
            clean["restricted_fp_rejection"]
        )
        >= float(
            middle["conditions"]["clean"]["restricted_fp_rejection"]
        ),
        "hard_corrections_exceed_harms": int(hard["corrections"])
        > int(hard["harms"]),
        "hard_class1_tp_breaks_lte_2": int(hard["class1_tp_breaks"]) <= 2,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }


def _summarize_mechanism(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_rows: int = EXPECTED_COHORT_ROWS,
) -> Dict[str, object]:
    views: Dict[str, object] = {}
    for view, _ in CROPPED_VIEWS:
        condition_values: Dict[str, object] = {}
        for condition, _, _ in CONDITIONS:
            selected = [
                row
                for row in rows
                if str(row["view"]) == view
                and str(row["condition"]) == condition
            ]
            selected.sort(key=lambda row: int(row["cohort_position"]))
            if len(selected) != int(expected_rows):
                raise ValueError(
                    f"Mechanism {view}/{condition} has {len(selected)} rows."
                )
            positions = [int(row["cohort_position"]) for row in selected]
            if positions != _expected_cohort_positions(expected_rows):
                raise ValueError(
                    f"Mechanism {view}/{condition} has noncanonical cohort positions."
                )
            labels = np.asarray(
                [int(str(row["cohort"]) == "fp") for row in selected],
                dtype=np.int64,
            )
            object_values = np.asarray(
                [float(row["object_residual_rms"]) for row in selected],
                dtype=np.float64,
            )
            outside_values = np.asarray(
                [float(row["outside_residual_rms"]) for row in selected],
                dtype=np.float64,
            )
            object_tp = float(np.median(object_values[labels == 0]))
            object_fp = float(np.median(object_values[labels == 1]))
            outside_tp = float(np.median(outside_values[labels == 0]))
            outside_fp = float(np.median(outside_values[labels == 1]))
            condition_values[condition] = {
                "rows": len(selected),
                "all_regions_nonempty": all(
                    int(row["object_pixels"]) > 0
                    and int(row["outside_pixels"]) > 0
                    for row in selected
                ),
                "object_auroc_fp_vs_tp": float(
                    roc_auc_score(labels, object_values)
                ),
                "outside_auroc_fp_vs_tp": float(
                    roc_auc_score(labels, outside_values)
                ),
                "object_tp_median": object_tp,
                "object_fp_median": object_fp,
                "outside_tp_median": outside_tp,
                "outside_fp_median": outside_fp,
                "object_fp_minus_tp_gap": object_fp - object_tp,
                "outside_fp_minus_tp_gap": outside_fp - outside_tp,
            }
        views[view] = {"conditions": condition_values}
    return {"views": views}


def assess_mechanism_gate(
    mechanism_summary: Mapping[str, object]
) -> Dict[str, object]:
    conditions = mechanism_summary["views"]["early_b176"]["conditions"]
    positive_gap_count = sum(
        float(value["object_fp_minus_tp_gap"]) > 0.0
        and float(value["object_fp_minus_tp_gap"])
        > float(value["outside_fp_minus_tp_gap"])
        for value in conditions.values()
    )
    checks = {
        "every_row_has_object_and_outside": all(
            bool(value["all_regions_nonempty"]) for value in conditions.values()
        ),
        "every_condition_object_auroc_gte_0p60": all(
            float(value["object_auroc_fp_vs_tp"]) >= 0.60
            for value in conditions.values()
        ),
        "every_condition_object_auroc_exceeds_outside_by_0p03": all(
            float(value["object_auroc_fp_vs_tp"])
            - float(value["outside_auroc_fp_vs_tp"])
            >= 0.03
            for value in conditions.values()
        ),
        "object_gap_positive_and_exceeds_outside_in_gte_3_conditions": positive_gap_count
        >= 3,
    }
    return {
        "positive_object_gap_condition_count": int(positive_gap_count),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }


def _summarize_known_near_tie_mechanism_exclusion(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    filtered = [
        row for row in rows if int(row["sample_index"]) != KNOWN_NEAR_TIE_INDEX
    ]
    expected = len(CONDITIONS) * len(CROPPED_VIEWS) * SENSITIVITY_COHORT_ROWS
    if len(filtered) != expected:
        raise ValueError("Known near-tie exclusion mechanism count differs.")
    summary = _summarize_mechanism(
        filtered, expected_rows=SENSITIVITY_COHORT_ROWS
    )
    return {
        "excluded_sample_index": KNOWN_NEAR_TIE_INDEX,
        "rows_per_condition_view": SENSITIVITY_COHORT_ROWS,
        "mechanism_summary": summary,
        "mechanism_gate": assess_mechanism_gate(summary),
    }


def _prediction_fieldnames() -> list[str]:
    fields = [
        "condition",
        "cohort_position",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "cohort",
    ]
    for view, _ in VIEWS:
        fields.extend(f"{view}_logit_{index}" for index in range(5))
        fields.extend(f"{view}_prob_{index}" for index in range(5))
        fields.append(f"{view}_prediction")
    for view, _ in CROPPED_VIEWS:
        fields.extend(
            (
                f"{view}_suppression_score",
                f"{view}_oof_threshold",
                f"{view}_oof_rejected",
            )
        )
    return fields


def _write_predictions(
    path: Path, records: Sequence[Mapping[str, object]]
) -> None:
    fieldnames = _prediction_fieldnames()
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({name: row[name] for name in fieldnames})


def _write_mechanism(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    fieldnames = [
        "condition",
        "view",
        "cohort_position",
        "sample_index",
        "fold",
        "target",
        "cohort",
        "object_residual_rms",
        "outside_residual_rms",
        "object_pixels",
        "outside_pixels",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _tensor_to_rgb(image: Tensor, semantics: Mapping[str, object]) -> Tensor:
    mean = torch.tensor(
        semantics["input_mean"], device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        semantics["input_std"], device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def _pil_from_rgb(image: Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0.0, 255.0)
        .to(dtype=torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _bbox_overlay(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    cx, cy, width, height = [float(value) for value in bbox[:4]]
    x1 = max(0.0, cx - width / 2.0) * output.width
    x2 = min(1.0, cx + width / 2.0) * output.width
    y1 = max(0.0, cy - height / 2.0) * output.height
    y2 = min(1.0, cy + height / 2.0) * output.height
    draw.rectangle((x1, y1, x2, y2), outline=(235, 40, 40), width=3)
    return output


def _energy_image(energy: Tensor, *, upper: float) -> Image.Image:
    values = energy.detach().float().cpu().numpy()
    normalized = np.clip(values / max(float(upper), 1e-8), 0.0, 1.0)
    red = np.clip(1.7 * normalized, 0.0, 1.0)
    green = np.clip(1.6 - np.abs(normalized - 0.55) * 3.0, 0.0, 1.0)
    blue = np.clip(1.25 * (1.0 - normalized), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(np.rint(rgb * 255.0)), mode="RGB")


def _visual_selection(
    records: Sequence[Mapping[str, object]], condition: str
) -> tuple[list[Mapping[str, object]], list[str]]:
    selected = _condition_records(records, condition)
    score_key = "early_b176_suppression_score"
    output: list[Mapping[str, object]] = []
    names: list[str] = []
    specifications = (
        ("fp", "highest"),
        ("fp", "lowest"),
        ("tp", "highest"),
        ("tp", "lowest"),
    )
    for cohort_name, rank_name in specifications:
        cohort_rows = [
            row for row in selected if str(row["cohort"]) == cohort_name
        ]
        ordered = sorted(
            cohort_rows,
            key=lambda row: (
                float(row[score_key]), int(row["cohort_position"])
            ),
            reverse=rank_name == "highest",
        )[:4]
        output.extend(ordered)
        names.extend(
            f"{cohort_name}_{rank_name}_{index + 1}"
            for index in range(len(ordered))
        )
    if len(output) != 16 or len({int(row["sample_index"]) for row in output}) != 16:
        raise ValueError(f"Visual selection is incomplete for {condition}.")
    return output, names


def _render_contact_sheets(
    *,
    output_dir: Path,
    checkpoint: Mapping[str, object],
    base_dataset,
    transform,
    records: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    semantics = _eval_semantics(checkpoint)
    font = ImageFont.load_default()
    sheets: list[Dict[str, object]] = []
    for condition, brightness, contrast in CONDITIONS:
        selected, selection_names = _visual_selection(records, condition)
        loader, _ = _make_condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=[int(row["sample_index"]) for row in selected],
            brightness=brightness,
            contrast=contrast,
            batch_size=16,
            num_workers=int(args.num_workers),
            context=f"efficienttrain_a0_visual_{condition}",
        )
        images_out: list[Tensor] = []
        early_out: list[Tensor] = []
        low_out: list[Tensor] = []
        residual_out: list[Tensor] = []
        bboxes_out: list[Tensor] = []
        indices_out: list[int] = []
        with torch.inference_mode():
            for images, _, metadata in loader:
                crop_bboxes = metadata.get("crop_bbox")
                sample_indices = metadata.get("sample_index")
                if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(
                    sample_indices
                ):
                    raise ValueError("Visual loader metadata is incomplete.")
                images_device = images.to(device=device, dtype=torch.float32)
                early = efficienttrain_freq_crop(images_device, 176)
                low_frequency, residual = same_size_low_frequency_reconstruction(
                    images_device, 176
                )
                images_out.append(images.float())
                early_out.append(early.cpu())
                low_out.append(low_frequency.cpu())
                residual_out.append(
                    residual.square().mean(dim=1).sqrt().cpu()
                )
                bboxes_out.append(crop_bboxes[:, :4].float())
                indices_out.extend(int(value) for value in sample_indices.tolist())
        expected_indices = [int(row["sample_index"]) for row in selected]
        if indices_out != expected_indices:
            raise ValueError(f"Visual row order differs for {condition}.")
        images = _tensor_to_rgb(torch.cat(images_out), semantics)
        early = _tensor_to_rgb(torch.cat(early_out), semantics)
        low_frequency = _tensor_to_rgb(torch.cat(low_out), semantics)
        residual_energy = torch.cat(residual_out)
        bboxes = torch.cat(bboxes_out)

        canvas = Image.new("RGB", (1600, 16 * 196 + 44), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10),
            (
                f"{condition} | red=bbox | conditioned / official B176 / "
                "same-size low-frequency / removed residual"
            ),
            fill=(0, 0, 0),
            font=font,
        )
        for row_index, (row, selection) in enumerate(
            zip(selected, selection_names)
        ):
            top = 38 + row_index * 196
            panels = [
                _bbox_overlay(
                    _pil_from_rgb(images[row_index]), bboxes[row_index].tolist()
                ).resize((180, 180), Image.Resampling.BILINEAR),
                _pil_from_rgb(early[row_index]).resize(
                    (180, 180), Image.Resampling.BILINEAR
                ),
                _bbox_overlay(
                    _pil_from_rgb(low_frequency[row_index]),
                    bboxes[row_index].tolist(),
                ).resize((180, 180), Image.Resampling.BILINEAR),
            ]
            upper = float(torch.quantile(residual_energy[row_index], 0.98))
            panels.append(
                _energy_image(
                    residual_energy[row_index], upper=upper
                ).resize((180, 180), Image.Resampling.BILINEAR)
            )
            for panel_index, panel in enumerate(panels):
                canvas.paste(panel, (12 + panel_index * 192, top))
            lines = [
                f"{selection} | sample={row['sample_index']} fold={row['fold']}",
                (
                    f"target={row['target']} native={row['native_b256_prediction']} "
                    f"B176={row['early_b176_prediction']}"
                ),
                (
                    f"suppression={float(row['early_b176_suppression_score']):.6f} "
                    f"threshold={float(row['early_b176_oof_threshold']):.6f} "
                    f"rejected={row['early_b176_oof_rejected']}"
                ),
                f"residual upper={upper:.6f}",
                str(row["image_path"]),
            ]
            draw.multiline_text(
                (790, top + 14),
                "\n".join(lines),
                fill=(10, 10, 10),
                font=font,
                spacing=9,
            )
        path = output_dir / f"contact_sheet_{condition}.png"
        canvas.save(path)
        sheets.append(
            {
                "condition": condition,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "selected_rows": len(selected),
                "sample_indices": expected_indices,
                "selections": selection_names,
            }
        )
    review_manifest = {
        "manual_review_status": "pending",
        "required_page_count": 4,
        "sheets": sheets,
    }
    manifest_path = output_dir / "contact_sheet_review_manifest.json"
    manifest_path.write_text(
        json.dumps(review_manifest, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        **review_manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }


def _benchmark_transform(
    *, device: torch.device, repeats: int
) -> Dict[str, object]:
    gpu_before = _gpu_snapshot()
    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE, device=device)
    pattern = (
        torch.sin(axis[None, :] * 7.0)
        + torch.cos(axis[:, None] * 5.0)
    )
    channels = torch.stack((pattern, pattern.square(), pattern.flip(0)), dim=0)
    images = channels.unsqueeze(0).expand(BENCHMARK_BATCH_SIZE, -1, -1, -1)
    images = images.contiguous().to(dtype=torch.float32)
    for _ in range(BENCHMARK_WARMUPS):
        warmup = efficienttrain_freq_crop(images, 176)
        del warmup
    torch.cuda.synchronize(device)
    timings_ms: list[float] = []
    peak_allocations: list[int] = []
    checksum = 0.0
    for _ in range(int(repeats)):
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = efficienttrain_freq_crop(images, 176)
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        peak_allocations.append(
            max(
                0,
                int(torch.cuda.max_memory_allocated(device)) - allocated_before,
            )
        )
        checksum += float(output[0, 0, 0, 0])
        del output
    gpu_after = _gpu_snapshot()
    result = {
        "batch_size": BENCHMARK_BATCH_SIZE,
        "warmups": BENCHMARK_WARMUPS,
        "repeats": int(repeats),
        "band_width": 176,
        "timings_ms": timings_ms,
        "median_ms": float(statistics.median(timings_ms)),
        "peak_allocation_bytes": peak_allocations,
        "median_peak_allocation_bytes": int(
            statistics.median(peak_allocations)
        ),
        "checksum": checksum,
        "finite": bool(
            all(math.isfinite(value) and value > 0.0 for value in timings_ms)
            and math.isfinite(checksum)
        ),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "timing_isolated": bool(
            gpu_before.get("available")
            and gpu_after.get("available")
            and int(gpu_before["utilization_percent"]) <= 10
            and int(gpu_after["utilization_percent"]) <= 10
        ),
        "deployed_parameter_delta": 0,
        "deployed_inference_operation_delta": 0,
        "training_view_only": True,
    }
    del images, channels, pattern, axis
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _write_launcher_transcript(
    path: Path, *, args: argparse.Namespace, started_utc: str
) -> None:
    payload = {
        "method": METHOD,
        "started_utc": started_utc,
        "argv": [str(value) for value in sys.argv],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "working_directory": str(Path.cwd().resolve()),
        "locked_arguments_exact": _locked_args_exact(args),
        "gpu_at_launcher_record": _gpu_snapshot(),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    early = summary.get("view_metrics", {}).get("early_b176", {})
    condition_metrics = early.get("conditions", {})
    lines = [
        "# EfficientTrain Low-Frequency Curriculum A0 Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Automatic gate eligible: `{summary.get('auto_gate_eligible', False)}`",
        f"- Structural failures: `{summary['structural_gate']['failed_checks']}`",
        f"- Information failures: `{summary['information_gate']['failed_checks']}`",
        f"- Mechanism failures: `{summary['mechanism_gate']['failed_checks']}`",
        "- Validation/test/training used: `False/False/False`",
        "- Current-best commands updated: `False`",
        "",
        "## Early B176",
        "",
        "| Condition | AUROC | TP retention | Restricted-FP rejection |",
        "|---|---:|---:|---:|",
    ]
    for condition, _, _ in CONDITIONS:
        if condition not in condition_metrics:
            continue
        values = condition_metrics[condition]
        lines.append(
            f"| {condition} | {float(values['auroc_fp_vs_tp']):.6f} | "
            f"{float(values['tp_retention']):.6f} | "
            f"{float(values['restricted_fp_rejection']):.6f} |"
        )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            (
                "A pass authorizes only a separately locked matched scratch "
                "protocol. It never authorizes validation, test, full training, "
                "or a current-best command update."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    artifacts = []
    forbidden_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".engine",
        ".mmap",
        ".memmap",
        ".f16",
        ".cache",
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "forbidden_artifact_count": sum(
            Path(str(row["path"])).suffix.casefold() in forbidden_suffixes
            for row in artifacts
        ),
    }
    path = output_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "path": str(path.resolve()), "sha256": _sha256(path)}


def _implementation_unchanged(provenance: Mapping[str, object]) -> bool:
    repository = Path(provenance["paths"]["repository"])
    return all(
        _sha256(repository / str(value["path"])) == str(value["sha256"])
        for value in provenance["implementation"].values()
    )


def _formal_structural_checks(
    *,
    args: argparse.Namespace,
    provenance: Mapping[str, object],
    equation: Mapping[str, object],
    dataset_mapping: Mapping[str, object],
    declarations: Mapping[str, object],
    collection: Mapping[str, object],
    model_state_unchanged: bool,
    parameter_count_unchanged: bool,
    rng_unchanged: bool,
    replay_checks: Mapping[str, bool],
    contact_sheets: Mapping[str, object],
    resource: Mapping[str, object],
    paths: Mapping[str, Path],
) -> Dict[str, bool]:
    geometry = collection["geometry"]
    repository = paths["repository"]
    return {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_commit_tree_hashes_and_worktree_exact": bool(
            provenance["official_worktree_clean"]
            and provenance["official_commit"] == LOCKED_OFFICIAL_COMMIT
            and provenance["official_tree"] == LOCKED_OFFICIAL_TREE
        ),
        "repository_commit_exactly_pushed": bool(
            provenance["repository_commit"] == provenance["upstream_commit"]
        ),
        "cohort_count_fold_order_exact": bool(
            provenance["cohort_rows"] == EXPECTED_COHORT_ROWS
            and provenance["positive_rows"] == EXPECTED_POSITIVES
            and provenance["negative_rows"] == EXPECTED_NEGATIVES
            and provenance["ordered_cohort_index_sha256"]
            == EXPECTED_ORDERED_INDEX_SHA256
        ),
        "dataset_mapping_train_only_exact": bool(
            dataset_mapping["paths_exact"]
            and dataset_mapping["train_paths_only"]
            and int(dataset_mapping["rows"]) == 9215
        ),
        "official_candidate_and_oracle_error_lte_1e6": max(
            float(equation["maximum_official_candidate_error"]),
            float(equation["maximum_candidate_oracle_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "spectral_corner_scale_replay_lte_1e6": float(
            equation["maximum_spectral_replay_error"]
        )
        <= MAX_EQUATION_ERROR,
        "same_size_reconstruction_oracle_lte_1e6": max(
            float(equation["maximum_same_size_oracle_error"]),
            float(equation["maximum_reconstruction_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "equation_deterministic_finite": all(
            bool(value["deterministic_two_pass_bit_exact"])
            and bool(value["all_outputs_finite"])
            for value in equation["cases"].values()
        ),
        "b256_equation_and_input_identity_bit_exact": bool(
            equation["b256_identity_bit_exact"]
            and equation["b256_same_storage"]
            and collection["native_b256_input_bit_exact"]
        ),
        "clean_native_declaration_erratum_exact_before_candidate_metrics": bool(
            declarations["mismatch_policy"]["policy_passed"]
            and declarations["mismatch_policy"]["exact_known_exception"]
            and int(declarations["mismatch_policy"]["mismatch_count"]) == 1
            and int(declarations["rows"]) == EXPECTED_COHORT_ROWS
            and not declarations["candidate_metric_read_before_declaration"]
        ),
        "b256_keeper_logits_bit_exact": float(
            collection["native_b256_declaration_logit_max_abs_error"]
        )
        == 0.0,
        "all_three_dynamic_grids_exact": all(
            bool(value["grid_exact"])
            and bool(value["prefix_count_exact"])
            and bool(value["positional_count_exact"])
            and bool(value["indices_in_grid"])
            and bool(value["finite_logits"])
            for value in geometry.values()
        ),
        "model_state_parameter_count_and_rng_unchanged": bool(
            model_state_unchanged and parameter_count_unchanged and rng_unchanged
        ),
        "independent_csv_replay_exact": all(replay_checks.values()),
        "four_contact_sheets_complete": bool(
            len(contact_sheets["sheets"]) == 4
            and all(
                int(value["selected_rows"]) == 16
                for value in contact_sheets["sheets"]
            )
            and int(contact_sheets["required_page_count"]) == 4
        ),
        "resource_record_finite_and_deployment_unchanged": bool(
            resource["finite"]
            and int(resource["deployed_parameter_delta"]) == 0
            and int(resource["deployed_inference_operation_delta"]) == 0
            and resource["training_view_only"]
        ),
        "no_holdout_validation_test_or_training": bool(
            not provenance["holdout_data_used"]
            and not provenance["validation_predictions_used"]
            and not provenance["test_data_used"]
            and not provenance["training_used"]
        ),
        "raw_data_and_current_commands_unchanged": bool(
            _sha256(paths["data"]) == LOCKED_DATA_SHA256
            and _sha256(paths["current_commands"])
            == LOCKED_CURRENT_COMMAND_SHA256
            and _sha256(paths["command_history"])
            == LOCKED_COMMAND_HISTORY_SHA256
            and _sha256(paths["interrupted_manifest"])
            == LOCKED_INTERRUPTED_MANIFEST_SHA256
            and _sha256(paths["interrupted_transcript"])
            == LOCKED_INTERRUPTED_TRANSCRIPT_SHA256
        ),
        "repository_and_implementation_unchanged_at_end": bool(
            _tracked_worktree_clean(repository)
            and _git_value(repository, "rev-parse", "HEAD")
            == provenance["repository_commit"]
            and _git_value(
                repository,
                "rev-parse",
                "origin/classification-only-research",
            )
            == provenance["upstream_commit"]
            and _implementation_unchanged(provenance)
        ),
        "official_source_unchanged_at_end": bool(
            _full_worktree_clean(paths["official_root"])
            and _git_value(paths["official_root"], "rev-parse", "HEAD")
            == LOCKED_OFFICIAL_COMMIT
            and _git_value(
                paths["official_root"], "rev-parse", "HEAD^{tree}"
            )
            == LOCKED_OFFICIAL_TREE
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance, rows, cohort = _load_locked_inputs(args)
    paths = _source_paths(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Locked EfficientTrain A0 requires CUDA.")
    if bool(args.preflight_only):
        equation = _equation_diagnostics(paths["utils"])
        maximum_error = max(
            float(equation["maximum_official_candidate_error"]),
            float(equation["maximum_candidate_oracle_error"]),
            float(equation["maximum_spectral_replay_error"]),
            float(equation["maximum_same_size_oracle_error"]),
            float(equation["maximum_reconstruction_error"]),
        )
        if maximum_error > MAX_EQUATION_ERROR:
            raise RuntimeError(
                f"EfficientTrain equation preflight failed: {maximum_error}"
            )
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "provenance": provenance,
            "equation": equation,
            "output_directory_created": False,
            "candidate_cohort_inference_run": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }

    output_path = Path(args.output_dir).resolve()
    raw_root = paths["data"].parent.parent.resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("Audit output cannot be under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)
    transcript_path = output_dir / "launcher_transcript.txt"
    _write_launcher_transcript(
        transcript_path, args=args, started_utc=started_utc
    )

    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    equation = _equation_diagnostics(paths["utils"])
    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    model = _load_keeper_model(checkpoint).to(device).eval()
    base_dataset, transform, dataset_mapping = _build_dataset(
        checkpoint, rows, paths["data"]
    )
    model_state_before = _state_sha256(model)
    parameter_count_before = sum(value.numel() for value in model.parameters())
    rng_before = _rng_snapshot()

    declaration_loader, declaration_loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="efficienttrain_a0_native_declaration",
    )
    declarations = _collect_native_declarations(
        model=model,
        loader=declaration_loader,
        cohort=cohort,
        device=device,
    )
    records, mechanism_rows, collection = _collect_conditions(
        model=model,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        declarations=declarations,
        args=args,
        device=device,
    )
    view_metrics = _summarize_views(records)
    information_gate = assess_information_gate(view_metrics)
    exclusion_sensitivity = _summarize_known_near_tie_exclusion(
        records, view_metrics
    )
    mechanism_summary = _summarize_mechanism(mechanism_rows)
    mechanism_gate = assess_mechanism_gate(mechanism_summary)
    mechanism_exclusion_sensitivity = (
        _summarize_known_near_tie_mechanism_exclusion(mechanism_rows)
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    mechanism_path = output_dir / "frequency_mechanism.csv"
    _write_predictions(predictions_path, records)
    _write_mechanism(mechanism_path, mechanism_rows)
    independent_replay = replay_artifacts(predictions_path, mechanism_path)
    replay_checks = {
        "view_metrics_exact_within_1e12": canonical_close(
            view_metrics, independent_replay["view_metrics"]
        ),
        "information_gate_exact": canonical_close(
            information_gate, independent_replay["information_gate"]
        ),
        "mechanism_summary_exact_within_1e12": canonical_close(
            mechanism_summary, independent_replay["mechanism_summary"]
        ),
        "mechanism_gate_exact": canonical_close(
            mechanism_gate, independent_replay["mechanism_gate"]
        ),
        "known_near_tie_exclusion_exact_within_1e12": canonical_close(
            exclusion_sensitivity,
            independent_replay["known_near_tie_exclusion_sensitivity"][
                "prediction"
            ],
        ),
        "known_near_tie_mechanism_exclusion_exact_within_1e12": canonical_close(
            mechanism_exclusion_sensitivity,
            independent_replay["known_near_tie_exclusion_sensitivity"][
                "mechanism"
            ],
        ),
    }
    replay_path = output_dir / "independent_replay.json"
    replay_path.write_text(
        json.dumps(
            {"replay": independent_replay, "checks": replay_checks},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    contact_sheets = _render_contact_sheets(
        output_dir=output_dir,
        checkpoint=checkpoint,
        base_dataset=base_dataset,
        transform=transform,
        records=records,
        args=args,
        device=device,
    )
    resource = _benchmark_transform(
        device=device, repeats=int(args.benchmark_repeats)
    )

    model_state_after = _state_sha256(model)
    parameter_count_after = sum(value.numel() for value in model.parameters())
    rng_after = _rng_snapshot()
    structural_checks = _formal_structural_checks(
        args=args,
        provenance=provenance,
        equation=equation,
        dataset_mapping=dataset_mapping,
        declarations=declarations,
        collection=collection,
        model_state_unchanged=model_state_before == model_state_after,
        parameter_count_unchanged=parameter_count_before == parameter_count_after,
        rng_unchanged=rng_before == rng_after,
        replay_checks=replay_checks,
        contact_sheets=contact_sheets,
        resource=resource,
        paths=paths,
    )
    structural_gate = {
        "checks": structural_checks,
        "failed_checks": [
            name for name, passed in structural_checks.items() if not passed
        ],
        "passed": all(structural_checks.values()),
    }
    auto_eligible = bool(
        structural_gate["passed"]
        and information_gate["passed"]
        and mechanism_gate["passed"]
        and exclusion_sensitivity["information_gate"]["passed"]
        and mechanism_exclusion_sensitivity["mechanism_gate"]["passed"]
    )
    automatic_failed_checks = [
        *(
            f"structural:{name}"
            for name in structural_gate["failed_checks"]
        ),
        *(
            f"information:{name}"
            for name in information_gate["failed_checks"]
        ),
        *(
            f"mechanism:{name}" for name in mechanism_gate["failed_checks"]
        ),
        *(
            f"sensitivity_information:{name}"
            for name in exclusion_sensitivity["information_gate"][
                "failed_checks"
            ]
        ),
        *(
            f"sensitivity_mechanism:{name}"
            for name in mechanism_exclusion_sensitivity["mechanism_gate"][
                "failed_checks"
            ]
        ),
    ]
    gate = {
        "automatic_failed_checks": automatic_failed_checks,
        "manual_visual_review_passed": None,
        "failed_checks": automatic_failed_checks,
        "all_gates_passed": False,
        "matched_scratch_protocol_authorized": False,
        "validation_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "current_best_update_authorized": False,
    }
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": "awaiting_visual_review",
        "auto_gate_eligible": auto_eligible,
        "provenance": provenance,
        "equation_audit": equation,
        "dataset_mapping": dataset_mapping,
        "native_declaration_audit": {
            key: value
            for key, value in declarations.items()
            if key not in {"logits", "probabilities", "indices", "targets", "predictions"}
        },
        "dynamic_geometry_audit": collection["geometry"],
        "native_b256_control": {
            "input_bit_exact": collection["native_b256_input_bit_exact"],
            "declaration_logit_max_abs_error": collection[
                "native_b256_declaration_logit_max_abs_error"
            ],
        },
        "loader_audit": {
            "declaration": declaration_loader_summary,
            "conditions": collection["loader_summaries"],
        },
        "view_metrics": view_metrics,
        "information_gate": information_gate,
        "mechanism_summary": mechanism_summary,
        "mechanism_gate": mechanism_gate,
        "known_near_tie_exclusion_sensitivity": {
            "prediction": exclusion_sensitivity,
            "mechanism": mechanism_exclusion_sensitivity,
        },
        "independent_replay_audit": {
            "checks": replay_checks,
            "all_exact": all(replay_checks.values()),
            "path": str(replay_path.resolve()),
            "sha256": _sha256(replay_path),
        },
        "contact_sheet_audit": contact_sheets,
        "resource_audit": resource,
        "integrity_audit": {
            "model_state_before": model_state_before,
            "model_state_after": model_state_after,
            "model_state_unchanged": model_state_before == model_state_after,
            "parameter_count_before": parameter_count_before,
            "parameter_count_after": parameter_count_after,
            "parameter_count_unchanged": parameter_count_before
            == parameter_count_after,
            "rng_before": rng_before,
            "rng_after": rng_after,
            "rng_unchanged": rng_before == rng_after,
        },
        "structural_gate": structural_gate,
        "gate": gate,
        "visual_review": None,
        "validation_predictions_used": False,
        "holdout_data_used": False,
        "test_data_used": False,
        "training_used": False,
        "raw_data_modified": False,
        "image_epochs_run": 0,
        "checkpoint_written": False,
        "current_best_commands_updated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "predictions": str(predictions_path.resolve()),
            "mechanism": str(mechanism_path.resolve()),
            "replay": str(replay_path.resolve()),
            "launcher_transcript": str(transcript_path.resolve()),
        },
    }
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    _write_report(report_path, summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir)
    if int(manifest["forbidden_artifact_count"]) != 0:
        raise RuntimeError("EfficientTrain A0 retained a forbidden artifact.")
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Formal EfficientTrain summary is missing: {summary_path}"
        )
    observed_sha = _sha256(summary_path)
    expected_sha = str(args.expected_summary_sha256).strip().casefold()
    if not expected_sha or observed_sha != expected_sha:
        raise ValueError(
            f"Pre-review summary SHA differs: {observed_sha} != {expected_sha}"
        )
    note = str(args.visual_review_note).strip()
    if not note:
        raise ValueError("Visual finalization requires a non-empty review note.")
    current_provenance, _, _ = _load_locked_inputs(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError("EfficientTrain summary is not awaiting visual review.")
    original_provenance = summary["provenance"]
    if (
        current_provenance["repository_commit"]
        != original_provenance["repository_commit"]
        or current_provenance["upstream_commit"]
        != original_provenance["upstream_commit"]
        or current_provenance["implementation"]
        != original_provenance["implementation"]
    ):
        raise ValueError("Implementation provenance changed before finalization.")
    for sheet in summary["contact_sheet_audit"]["sheets"]:
        path = Path(sheet["path"])
        if not path.is_file() or _sha256(path) != str(sheet["sha256"]):
            raise ValueError(f"Contact sheet changed before review: {path}")

    passed = str(args.visual_review_result) == "pass"
    review_manifest_path = output_dir / "contact_sheet_review_manifest.json"
    review_manifest = {
        "manual_review_status": "passed" if passed else "failed",
        "result": str(args.visual_review_result),
        "note": note,
        "pre_review_summary_sha256": observed_sha,
        "required_page_count": 4,
        "reviewed_page_count": len(summary["contact_sheet_audit"]["sheets"]),
        "sheets": summary["contact_sheet_audit"]["sheets"],
    }
    review_manifest_path.write_text(
        json.dumps(review_manifest, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    summary["contact_sheet_audit"]["manual_review_status"] = review_manifest[
        "manual_review_status"
    ]
    summary["contact_sheet_audit"]["manifest_sha256"] = _sha256(
        review_manifest_path
    )
    summary["visual_review"] = {
        "result": str(args.visual_review_result),
        "passed": passed,
        "note": note,
        "reviewed_page_count": len(summary["contact_sheet_audit"]["sheets"]),
        "pre_review_summary_sha256": observed_sha,
    }
    gate = dict(summary["gate"])
    failed = list(gate["automatic_failed_checks"])
    if not passed:
        failed.append("manual_visual_review_passed")
    gate["manual_visual_review_passed"] = passed
    gate["failed_checks"] = failed
    gate["all_gates_passed"] = not failed
    gate["matched_scratch_protocol_authorized"] = not failed
    gate["validation_authorized"] = False
    gate["test_authorized"] = False
    gate["full_train_authorized"] = False
    gate["current_best_update_authorized"] = False
    summary["gate"] = gate
    summary["status"] = "passed" if not failed else "rejected"
    summary["current_best_commands_updated"] = False
    _write_report(output_dir / "report.md", summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir)
    if int(manifest["forbidden_artifact_count"]) != 0:
        raise RuntimeError("Finalized audit contains a forbidden artifact.")
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if bool(args.finalize_visual_review):
        result = _finalize_visual_review(args)
    else:
        result = run_audit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "method": result["method"],
                "summary_path": result.get("summary_path"),
                "summary_sha256": result.get("summary_sha256"),
                "auto_gate_eligible": result.get("auto_gate_eligible", False),
                "matched_scratch_protocol_authorized": result.get(
                    "gate", {}
                ).get("matched_scratch_protocol_authorized", False),
                "validation_predictions_used": result.get(
                    "validation_predictions_used", False
                ),
                "test_data_used": result.get("test_data_used", False),
            },
            sort_keys=True,
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

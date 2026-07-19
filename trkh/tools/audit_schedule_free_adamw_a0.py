from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset, build_train_transform
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.inference.inference import load_checkpoint
from trkh.models.feature_hooks import reconstruct_attention, resolve_attention_hook
from trkh.models.model import classification_logits_from_features, create_model, load_model_state
from trkh.tools.audit_class1_protected_rsc_readiness import (
    _export_candidate,
    _features_from_batch,
    _parameter_group,
    _parameter_group_hashes,
    _predict_fp32,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _heat_overlay,
    _normalize_maps,
    _rgb_from_tensor,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _comparison,
    _make_loader,
    _prepare_output_dir,
    _sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_FIT_ROWS = 7372
EXPECTED_HOLDOUT_ROWS = 1843
EXPECTED_HOLDOUT_CLASS_COUNTS = (380, 109, 393, 503, 458)
EXPECTED_TRAIN_BATCHES = 60
EXPECTED_TRAIN_ROWS_USED = 1920
EXPECTED_CALIBRATION_BATCHES = 50
EXPECTED_CALIBRATION_ROWS = 1600

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "455d285dfd4ec559629a353d7621b8991579be8d33c0c005cf99e74b990fe499"
LOCKED_PAPER_SHA256 = "1aef42351ac417eb08ba81919cb49bd9d14d98320d0bd3d47eac72f48dcf20d3"
LOCKED_OFFICIAL_COMMIT = "d24878d3489bf8ede6148eb6390c9b272b9d93c4"
LOCKED_OFFICIAL_TREE = "3aff6c4c3c69a3c965d8ea406008be6ff2d8782f"
LOCKED_OPTIMIZER_BLOB = "e8f5a5d52f497577fa229b3c4341feca251bb6e9"
LOCKED_REFERENCE_BLOB = "b133c3fa1158cde356baa58b45660cc983109923"
LOCKED_PAPER_OPTIMIZER_BLOB = "aeb80e58f473181204d109302ceb90680ce03f6a"
LOCKED_LICENSE_BLOB = "f49a4e16e68b128803cc2dcea614603632b04eac"
LOCKED_OPTIMIZER_SHA256 = "52afca387c4a8004b669cfa8e58b58b98492d128e54e17d15137b70716202f52"
LOCKED_REFERENCE_SHA256 = "38547977ee4253fcb82e4f07d5f8777ca22e804bf5246bfb6dee3ef997caf2aa"
LOCKED_LICENSE_SHA256 = "5ad8c213095c573921d7388edf93f6a9f54490b8ebf3b165ae8ef1947cf70846"
LOCKED_TRAIN_TRANSFORM_SHA256 = "12dd45b1a03d3b19cf830fff84f92389562bf73e278dad71202e616ba0d2bd04"

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WARMUP_STEPS = 8
MAX_RUNTIME_RATIO = 1.20
MAX_PEAK_VRAM_GIB = 6.50
MAX_PEAK_VRAM_RATIO = 1.15
MAX_ONNX_ERROR = 1e-5
MAX_MODE_ROUNDTRIP_ERROR = float(torch.finfo(torch.float32).eps)
LOCKED_LIGHTING_CONDITIONS = (
    ("lighting_dim", 0.72, 1.00),
    ("lighting_bright", 1.28, 1.00),
    ("low_contrast", 1.00, 0.65),
)


def _mode_roundtrip_within_fp32_tolerance(value: object) -> bool:
    error = float(value)
    return math.isfinite(error) and 0.0 <= error <= MAX_MODE_ROUNDTRIP_ERROR


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Schedule-Free AdamW A0. Validation and test "
            "access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/launcher_args.json"
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
            "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_SCHEDULE_FREE_ADAMW_A0_PROTOCOL_20260720.md"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_schedule_free_official_20260720"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_schedule_free_arxiv_2405.15682.pdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_schedule_free_adamw_a0_20260720"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--replay-summary", type=Path, default=None)
    parser.add_argument(
        "--finalize-visual-review", action="store_true", default=False
    )
    parser.add_argument(
        "--visual-review-result", choices=("pass", "fail"), default="fail"
    )
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument(
        "--calibration-batches", type=int, default=EXPECTED_CALIBRATION_BATCHES
    )
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--contact-rows", type=int, default=8)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument(
        "--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB
    )
    parser.add_argument(
        "--max-peak-vram-ratio", type=float, default=MAX_PEAK_VRAM_RATIO
    )
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == 32
        and int(args.num_workers) == 4
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.max_train_batches) == EXPECTED_TRAIN_BATCHES
        and math.isclose(float(args.learning_rate), LEARNING_RATE, abs_tol=1e-15)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY, abs_tol=1e-15)
        and int(args.warmup_steps) == WARMUP_STEPS
        and int(args.calibration_batches) == EXPECTED_CALIBRATION_BATCHES
        and int(args.focus_class) == 1
        and int(args.contact_rows) == 8
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
        and math.isclose(float(args.max_peak_vram_ratio), MAX_PEAK_VRAM_RATIO)
    )


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "optimizer": official / "schedulefree" / "adamw_schedulefree.py",
        "reference": official / "schedulefree" / "adamw_schedulefree_reference.py",
        "paper_optimizer": official
        / "schedulefree"
        / "adamw_schedulefree_paper.py",
        "license": official / "LICENSE",
    }


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("A0 arguments differ from the prospectively locked protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"),
        "launcher_args": _verify_sha256(
            paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "paper"),
        "optimizer": _verify_sha256(
            paths["optimizer"], LOCKED_OPTIMIZER_SHA256, "official optimizer"
        ),
        "reference": _verify_sha256(
            paths["reference"], LOCKED_REFERENCE_SHA256, "official reference"
        ),
        "license": _verify_sha256(
            paths["license"], LOCKED_LICENSE_SHA256, "official license"
        ),
    }
    root = Path(args.official_root).resolve()
    commit = _git_value(root, "rev-parse", "HEAD")
    tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    status = _git_value(root, "status", "--porcelain")
    blobs = {
        "optimizer": _git_value(root, "rev-parse", "HEAD:schedulefree/adamw_schedulefree.py"),
        "reference": _git_value(
            root, "rev-parse", "HEAD:schedulefree/adamw_schedulefree_reference.py"
        ),
        "paper_optimizer": _git_value(
            root, "rev-parse", "HEAD:schedulefree/adamw_schedulefree_paper.py"
        ),
        "license": _git_value(root, "rev-parse", "HEAD:LICENSE"),
    }
    expected_blobs = {
        "optimizer": LOCKED_OPTIMIZER_BLOB,
        "reference": LOCKED_REFERENCE_BLOB,
        "paper_optimizer": LOCKED_PAPER_OPTIMIZER_BLOB,
        "license": LOCKED_LICENSE_BLOB,
    }
    if commit != LOCKED_OFFICIAL_COMMIT or tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(f"Official source revision differs: {commit}/{tree}")
    if blobs != expected_blobs:
        raise ValueError(f"Official source blobs differ: {blobs}")
    if status:
        raise ValueError(f"Official source worktree is dirty: {status}")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "official_commit": commit,
        "official_tree": tree,
        "official_blobs": blobs,
        "official_worktree_clean": True,
        "validation_predictions_used": False,
        "test_data_used": False,
    }


def _load_class(path: Path, class_name: str):
    module_name = f"trkh_locked_{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    namespace = {
        "__builtins__": __builtins__,
        "__file__": str(Path(path).resolve()),
        "__name__": module_name,
        "__package__": "",
    }
    source = Path(path).read_bytes()
    exec(compile(source, str(Path(path).resolve()), "exec"), namespace)
    value = namespace.get(class_name)
    if value is None:
        raise ImportError(f"Official module has no {class_name}: {path}")
    return value


def _scalar_trajectory(
    initial: float,
    gradients: Sequence[float],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    warmup_steps: int,
) -> Dict[str, object]:
    y = float(initial)
    z = float(initial)
    exp_avg_sq = 0.0
    weight_sum = 0.0
    lr_max = -1.0
    rows = []
    for k, gradient in enumerate(gradients):
        schedule = (k + 1) / warmup_steps if k < warmup_steps else 1.0
        scheduled_lr = lr * schedule
        bias_correction2 = 1.0 - beta2 ** (k + 1)
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient * gradient
        normalized = gradient / (math.sqrt(exp_avg_sq / bias_correction2) + eps)
        normalized += weight_decay * y
        lr_max = max(lr_max, scheduled_lr)
        weight = lr_max**2
        weight_sum += weight
        ckp1 = weight / weight_sum if weight_sum else 0.0
        y = (1.0 - ckp1) * y + ckp1 * z
        y += scheduled_lr * (beta1 * (1.0 - ckp1) - 1.0) * normalized
        z -= scheduled_lr * normalized
        rows.append({"k": k, "y": y, "z": z, "scheduled_lr": scheduled_lr})
    x = y + (1.0 - 1.0 / beta1) * (z - y)
    return {"y": y, "z": z, "x": x, "rows": rows}


def _equation_diagnostics(paths: Mapping[str, Path]) -> Dict[str, object]:
    optimized_class = _load_class(paths["optimizer"], "AdamWScheduleFree")
    reference_class = _load_class(
        paths["reference"], "AdamWScheduleFreeReference"
    )
    initial = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64)
    gradients = [
        torch.tensor(
            [math.sin(index + 1), math.cos(index + 0.5), (-1.0) ** index * 0.3],
            dtype=torch.float64,
        )
        for index in range(12)
    ]

    parameters = {
        "optimized_foreach": nn.Parameter(initial.clone()),
        "optimized_scalar": nn.Parameter(initial.clone()),
        "reference": nn.Parameter(initial.clone()),
    }
    optimizers = {
        "optimized_foreach": optimized_class(
            [parameters["optimized_foreach"]],
            lr=0.01,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=3,
            foreach=True,
        ),
        "optimized_scalar": optimized_class(
            [parameters["optimized_scalar"]],
            lr=0.01,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=3,
            foreach=False,
        ),
        "reference": reference_class(
            [parameters["reference"]],
            lr=0.01,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=3,
        ),
    }
    for optimizer in optimizers.values():
        optimizer.train()
    train_errors = []
    for gradient in gradients:
        for name, parameter in parameters.items():
            parameter.grad = gradient.clone()
            optimizers[name].step()
        train_errors.append(
            max(
                float(
                    (parameters["optimized_foreach"] - parameters[name])
                    .abs()
                    .amax()
                    .item()
                )
                for name in ("optimized_scalar", "reference")
            )
        )
    y_value = parameters["optimized_foreach"].detach().clone()
    z_value = optimizers["optimized_foreach"].state[
        parameters["optimized_foreach"]
    ]["z"].detach().clone()
    for optimizer in optimizers.values():
        optimizer.eval()
    eval_errors = [
        float(
            (parameters["optimized_foreach"] - parameters[name])
            .abs()
            .amax()
            .item()
        )
        for name in ("optimized_scalar", "reference")
    ]
    x_value = parameters["optimized_foreach"].detach().clone()
    optimizers["optimized_foreach"].train()
    y_roundtrip = parameters["optimized_foreach"].detach().clone()
    optimizers["optimized_foreach"].eval()
    x_roundtrip = parameters["optimized_foreach"].detach().clone()

    scalar_gradients = [0.4, -0.2, 0.7, -0.5, 0.1]
    scalar_oracle = _scalar_trajectory(
        0.6,
        scalar_gradients,
        lr=0.01,
        beta1=BETAS[0],
        beta2=BETAS[1],
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=3,
    )
    scalar_parameter = nn.Parameter(torch.tensor([0.6], dtype=torch.float64))
    scalar_optimizer = optimized_class(
        [scalar_parameter],
        lr=0.01,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=3,
        foreach=False,
    )
    scalar_optimizer.train()
    for gradient in scalar_gradients:
        scalar_parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        scalar_optimizer.step()
    scalar_y = float(scalar_parameter.item())
    scalar_z = float(scalar_optimizer.state[scalar_parameter]["z"].item())
    scalar_optimizer.eval()
    scalar_x = float(scalar_parameter.item())
    scalar_errors = {
        "y": abs(scalar_y - float(scalar_oracle["y"])),
        "z": abs(scalar_z - float(scalar_oracle["z"])),
        "x": abs(scalar_x - float(scalar_oracle["x"])),
    }
    maximum_error = max([*train_errors, *eval_errors, *scalar_errors.values()])
    return {
        "maximum_official_reference_error": max([*train_errors, *eval_errors]),
        "maximum_scalar_oracle_error": max(scalar_errors.values()),
        "maximum_error": maximum_error,
        "mode_roundtrip_y_error": float((y_roundtrip - y_value).abs().amax().item()),
        "mode_roundtrip_x_error": float((x_roundtrip - x_value).abs().amax().item()),
        "x_y_max_abs_difference": float((x_value - y_value).abs().amax().item()),
        "x_z_max_abs_difference": float((x_value - z_value).abs().amax().item()),
        "passed": bool(
            maximum_error <= 1e-12
            and torch.equal(y_roundtrip, y_value)
            and torch.equal(x_roundtrip, x_value)
            and not torch.equal(x_value, y_value)
            and not torch.equal(x_value, z_value)
        ),
    }


def _build_locked_train_transform(
    checkpoint: Mapping[str, object],
) -> tuple[object, Dict[str, object]]:
    augmentation = checkpoint.get("augmentation_config")
    model_config = checkpoint.get("model_config")
    if not isinstance(augmentation, Mapping) or not isinstance(model_config, Mapping):
        raise ValueError("Keeper lacks the locked augmentation/model configuration.")
    required = (
        "resize_mode",
        "random_resized_crop_scale_min",
        "random_resized_crop_probability",
        "color_jitter_brightness",
        "color_jitter_contrast",
        "color_jitter_saturation",
        "color_jitter_hue",
        "random_erasing_probability",
        "random_affine_degrees",
        "random_affine_translate",
        "random_affine_scale_min",
        "horizontal_flip_probability",
        "vertical_flip_probability",
        "rotate90_probability",
        "lighting_probability",
        "randaugment_num_ops",
        "randaugment_magnitude",
        "illumination_normalization",
        "illumination_normalization_strength",
        "foreground_crop_mode",
        "foreground_crop_probability",
        "foreground_crop_margin_ratio",
        "foreground_crop_min_mask_area_ratio",
        "foreground_crop_max_mask_area_ratio",
        "foreground_crop_max_crop_area_ratio",
        "background_suppression_mode",
        "background_suppression_probability",
        "background_suppression_margin",
        "background_suppression_blur_radius",
        "surface_detail_amplification_mode",
        "surface_detail_amplification_probability",
        "surface_detail_amplification_strength",
        "surface_detail_amplification_blur_radius",
        "surface_detail_amplification_foreground_weight",
        "local_exposure_probability",
        "local_exposure_strength",
        "obstacle_probability",
        "obstacle_max_area",
        "class_aware_photometric_augmentation",
    )
    missing = [key for key in required if key not in augmentation]
    if missing:
        raise ValueError(f"Keeper augmentation config lacks locked fields: {missing}")
    if int(model_config.get("temporal_frames", 1)) != 1:
        raise ValueError("Schedule-Free A0 supports only the keeper's single-frame path.")
    image_size = int(model_config.get("image_size", 0))
    if image_size != 256:
        raise ValueError(f"Unexpected keeper image size: {image_size}")
    mean, std = checkpoint_input_normalization(checkpoint)
    settings: Dict[str, object] = {
        "image_size": image_size,
        **{key: augmentation[key] for key in required},
        "input_mean": [float(value) for value in mean],
        "input_std": [float(value) for value in std],
    }
    serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    summary = {
        "settings": settings,
        "settings_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "train_augmentation_enabled": True,
        "source": "keeper_checkpoint_augmentation_config",
    }
    transform = build_train_transform(
        image_size=image_size,
        resize_mode=str(augmentation["resize_mode"]),
        scale_min=float(augmentation["random_resized_crop_scale_min"]),
        scale_crop_probability=float(augmentation["random_resized_crop_probability"]),
        brightness=float(augmentation["color_jitter_brightness"]),
        contrast=float(augmentation["color_jitter_contrast"]),
        saturation=float(augmentation["color_jitter_saturation"]),
        hue=float(augmentation["color_jitter_hue"]),
        random_erasing_probability=float(augmentation["random_erasing_probability"]),
        random_affine_degrees=float(augmentation["random_affine_degrees"]),
        random_affine_translate=float(augmentation["random_affine_translate"]),
        random_affine_scale_min=float(augmentation["random_affine_scale_min"]),
        horizontal_flip_probability=float(augmentation["horizontal_flip_probability"]),
        vertical_flip_probability=float(augmentation["vertical_flip_probability"]),
        rotate90_probability=float(augmentation["rotate90_probability"]),
        lighting_probability=float(augmentation["lighting_probability"]),
        randaugment_num_ops=int(augmentation["randaugment_num_ops"]),
        randaugment_magnitude=int(augmentation["randaugment_magnitude"]),
        illumination_normalization=bool(augmentation["illumination_normalization"]),
        illumination_normalization_strength=float(
            augmentation["illumination_normalization_strength"]
        ),
        foreground_crop_mode=str(augmentation["foreground_crop_mode"]),
        foreground_crop_probability=float(augmentation["foreground_crop_probability"]),
        foreground_crop_margin_ratio=float(augmentation["foreground_crop_margin_ratio"]),
        foreground_crop_min_mask_area_ratio=float(
            augmentation["foreground_crop_min_mask_area_ratio"]
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation["foreground_crop_max_mask_area_ratio"]
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation["foreground_crop_max_crop_area_ratio"]
        ),
        background_suppression_mode=str(augmentation["background_suppression_mode"]),
        background_suppression_probability=float(
            augmentation["background_suppression_probability"]
        ),
        background_suppression_margin=float(augmentation["background_suppression_margin"]),
        background_suppression_blur_radius=float(
            augmentation["background_suppression_blur_radius"]
        ),
        surface_detail_amplification_mode=str(
            augmentation["surface_detail_amplification_mode"]
        ),
        surface_detail_amplification_probability=float(
            augmentation["surface_detail_amplification_probability"]
        ),
        surface_detail_amplification_strength=float(
            augmentation["surface_detail_amplification_strength"]
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation["surface_detail_amplification_blur_radius"]
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation["surface_detail_amplification_foreground_weight"]
        ),
        local_exposure_probability=float(augmentation["local_exposure_probability"]),
        local_exposure_strength=float(augmentation["local_exposure_strength"]),
        obstacle_probability=float(augmentation["obstacle_probability"]),
        obstacle_max_area=float(augmentation["obstacle_max_area"]),
        scale_photometric_with_augmentation=bool(
            augmentation["class_aware_photometric_augmentation"]
        ),
        mean=mean,
        std=std,
    )
    return transform, summary


def _tensor_batch_sha256(
    images: Tensor, targets: Tensor, metadata: Mapping[str, object]
) -> str:
    digest = hashlib.sha256()
    tensors = [("images", images), ("targets", targets)]
    tensors.extend(
        (f"metadata.{key}", value)
        for key, value in sorted(metadata.items())
        if torch.is_tensor(value)
    )
    if not any(name == "metadata.sample_index" for name, _value in tensors):
        raise ValueError("Batch metadata is missing sample_index.")
    for name, value in tensors:
        cpu = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(cpu.shape)}:{cpu.dtype}\n".encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _optimizer_state_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _gradient_telemetry(model: nn.Module) -> tuple[Dict[str, float], Dict[str, bool], bool]:
    squared: Dict[str, float] = {}
    seen: Dict[str, bool] = {}
    finite = True
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        group = _parameter_group(name)
        finite = bool(finite and torch.isfinite(gradient).all().item())
        nonzero = bool(torch.count_nonzero(gradient).item())
        seen[group] = bool(seen.get(group, False) or nonzero)
        squared[group] = squared.get(group, 0.0) + float(
            gradient.detach().float().square().sum().item()
        )
    return (
        {key: math.sqrt(value) for key, value in sorted(squared.items())},
        seen,
        finite,
    )


def _parameter_max_abs_difference(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> float:
    values = []
    for name, left_value in left.items():
        right_value = right.get(name)
        if right_value is None or not torch.is_floating_point(left_value):
            continue
        values.append(
            float(
                (left_value.detach().float().cpu() - right_value.detach().float().cpu())
                .abs()
                .amax()
                .item()
            )
        )
    return max(values, default=0.0)


def _clone_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _make_z_model(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[nn.Module, Dict[str, object]]:
    z_model = copy.deepcopy(model).cpu()
    z_names = []
    z_finite = True
    named_z = {}
    for name, parameter in model.named_parameters():
        z_value = optimizer.state.get(parameter, {}).get("z")
        if torch.is_tensor(z_value):
            value = z_value.detach().cpu().clone()
            named_z[name] = value
            z_names.append(name)
            z_finite = bool(z_finite and torch.isfinite(value).all().item())
    z_parameters = dict(z_model.named_parameters())
    with torch.no_grad():
        for name, value in named_z.items():
            z_parameters[name].copy_(value)
    return z_model.eval(), {
        "parameter_tensors": len(z_names),
        "optimizer_parameter_tensors": sum(
            len(group["params"]) for group in optimizer.param_groups
        ),
        "all_finite": z_finite,
        "names_sha256": hashlib.sha256(
            "\n".join(sorted(z_names)).encode("utf-8")
        ).hexdigest(),
    }


def _train_variant(
    *,
    name: str,
    schedule_free: bool,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    device: torch.device,
) -> tuple[nn.Module, Optional[nn.Module], Dict[str, object]]:
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial_state = _state_sha256(model)
    initial_groups = _parameter_group_hashes(model)
    if schedule_free:
        optimizer_class = _load_class(paths["optimizer"], "AdamWScheduleFree")
        optimizer = optimizer_class(
            model.parameters(),
            lr=float(args.learning_rate),
            betas=BETAS,
            eps=EPSILON,
            weight_decay=float(args.weight_decay),
            warmup_steps=int(args.warmup_steps),
            r=0.0,
            weight_lr_power=2.0,
            inner_momentum=0.0,
            foreach=True,
        )
        optimizer.train()
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            betas=BETAS,
            eps=EPSILON,
            weight_decay=float(args.weight_decay),
        )
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="schedule_free_a0_fit",
        seed=int(args.seed) + 200,
    )
    ordered_indices = []
    batch_hashes = []
    history = []
    gradient_seen = {key: False for key in initial_groups}
    all_gradients_finite = True
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (images_cpu, targets_cpu, metadata) in enumerate(loader):
        if batch_index >= int(args.max_train_batches):
            break
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("Training metadata is missing sample_index.")
        ordered_indices.extend(int(value) for value in sample_indices.tolist())
        batch_hashes.append(
            _tensor_batch_sha256(images_cpu, targets_cpu, metadata)
        )
        images = images_cpu.to(device=device, non_blocking=True)
        targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        if not schedule_free:
            factor = min(1.0, float(batch_index + 1) / float(args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = float(args.learning_rate) * factor
        set_seed(int(args.seed) + 10_000 + batch_index, deterministic=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        features = _features_from_batch(model, images, metadata, device=device)
        logits = classification_logits_from_features(model, features)
        loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"Non-finite {name} loss at batch {batch_index}.")
        loss.backward()
        gradient_norms, batch_seen, gradients_finite = _gradient_telemetry(model)
        for group, seen in batch_seen.items():
            gradient_seen[group] = bool(gradient_seen.get(group, False) or seen)
        all_gradients_finite = bool(all_gradients_finite and gradients_finite)
        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - started)
        history.append(
            {
                "batch": batch_index,
                "rows": int(targets.numel()),
                "loss": float(loss.detach().item()),
                "step_seconds": elapsed,
                "gradient_norms": gradient_norms,
                "clipped_gradient_norm": float(clipped_norm.detach().float().item()),
                "scheduled_lr": float(
                    optimizer.param_groups[0].get(
                        "scheduled_lr", optimizer.param_groups[0]["lr"]
                    )
                ),
            }
        )

    if len(history) != EXPECTED_TRAIN_BATCHES:
        raise ValueError(f"Expected 60 train batches, observed {len(history)}.")
    if len(ordered_indices) != EXPECTED_TRAIN_ROWS_USED:
        raise ValueError(
            f"Expected 1920 train rows, observed {len(ordered_indices)}."
        )
    training_peak_vram_gib = float(
        torch.cuda.max_memory_allocated(device) / (1024.0**3)
    )
    final_y_state = _clone_state(model)
    z_model = None
    mode = {"applicable": schedule_free}
    z_summary = {
        "parameter_tensors": 0,
        "optimizer_parameter_tensors": 0,
        "all_finite": True,
    }
    if schedule_free:
        z_model, z_summary = _make_z_model(model, optimizer)
        optimizer.eval()
        final_x_state = _clone_state(model)
        optimizer.train()
        replay_y_state = _clone_state(model)
        optimizer.eval()
        replay_x_state = _clone_state(model)
        mode.update(
            {
                "y_roundtrip_max_abs_error": _parameter_max_abs_difference(
                    final_y_state, replay_y_state
                ),
                "x_roundtrip_max_abs_error": _parameter_max_abs_difference(
                    final_x_state, replay_x_state
                ),
                "x_y_max_abs_difference": _parameter_max_abs_difference(
                    final_x_state, final_y_state
                ),
                "x_z_max_abs_difference": _parameter_max_abs_difference(
                    final_x_state, z_model.state_dict()
                ),
                "optimizer_eval_mode": all(
                    not bool(group.get("train_mode", True))
                    for group in optimizer.param_groups
                ),
            }
        )
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            checkpoint_buffer,
        )
        checkpoint_buffer.seek(0)
        checkpoint_payload = torch.load(
            checkpoint_buffer, map_location=device, weights_only=False
        )
        replay_model = copy.deepcopy(prototype).to(device).eval()
        replay_model.load_state_dict(checkpoint_payload["model_state"], strict=True)
        optimizer_class = _load_class(paths["optimizer"], "AdamWScheduleFree")
        replay_optimizer = optimizer_class(
            replay_model.parameters(),
            lr=float(args.learning_rate),
            betas=BETAS,
            eps=EPSILON,
            weight_decay=float(args.weight_decay),
            warmup_steps=int(args.warmup_steps),
            r=0.0,
            weight_lr_power=2.0,
            inner_momentum=0.0,
            foreach=True,
        )
        replay_optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
        replay_optimizer.eval()
        model.eval()
        replay_images = images_cpu[:2].to(device=device)
        replay_metadata = {
            key: value[:2] if torch.is_tensor(value) else value
            for key, value in metadata.items()
        }
        with torch.inference_mode():
            original_features = _features_from_batch(
                model, replay_images, replay_metadata, device=device
            )
            original_logits = classification_logits_from_features(
                model, original_features
            ).float()
            replay_features = _features_from_batch(
                replay_model, replay_images, replay_metadata, device=device
            )
            replay_logits = classification_logits_from_features(
                replay_model, replay_features
            ).float()
        mode["checkpoint_replay"] = {
            "model_state_max_abs_error": _parameter_max_abs_difference(
                model.state_dict(), replay_model.state_dict()
            ),
            "logit_max_abs_error": float(
                (original_logits - replay_logits).abs().amax().item()
            ),
            "argmax_match": bool(
                torch.equal(original_logits.argmax(dim=1), replay_logits.argmax(dim=1))
            ),
            "optimizer_eval_mode": all(
                not bool(group.get("train_mode", True))
                for group in replay_optimizer.param_groups
            ),
            "serialized_bytes": len(checkpoint_buffer.getvalue()),
            "serialized_sha256": hashlib.sha256(
                checkpoint_buffer.getvalue()
            ).hexdigest(),
        }
        del replay_optimizer, replay_model, checkpoint_payload
    else:
        final_x_state = final_y_state

    final_groups = _parameter_group_hashes(model)
    measured = [float(row["step_seconds"]) for row in history[WARMUP_STEPS:]]
    optimizer_buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), optimizer_buffer)
    result = {
        "name": name,
        "train_batches": len(history),
        "train_rows": len(ordered_indices),
        "train_order_sha256": _ordered_index_sha256(ordered_indices),
        "batch_hashes": batch_hashes,
        "batch_hash_sequence_sha256": hashlib.sha256(
            "\n".join(batch_hashes).encode("ascii")
        ).hexdigest(),
        "initial_state_sha256": initial_state,
        "final_state_sha256": _state_sha256(model),
        "parameter_group_hashes_before": initial_groups,
        "parameter_group_hashes_after": final_groups,
        "parameter_group_movement": {
            key: initial_groups.get(key) != final_groups.get(key)
            for key in sorted(set(initial_groups).union(final_groups))
        },
        "gradient_seen": gradient_seen,
        "all_gradients_finite": all_gradients_finite,
        "optimizer_state_finite": _optimizer_state_finite(optimizer),
        "optimizer_state_sha256": hashlib.sha256(optimizer_buffer.getvalue()).hexdigest(),
        "median_step_seconds_excluding_warmup": float(np.median(measured)),
        "peak_vram_gib": training_peak_vram_gib,
        "mode": mode,
        "z": z_summary,
        "history": history,
        "loader": loader_summary,
    }
    model = model.cpu().eval()
    if z_model is not None:
        z_model = z_model.cpu().eval()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return model, z_model, result


def _batchnorm_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not any(
            name.endswith(suffix)
            for suffix in ("running_mean", "running_var", "num_batches_tracked")
        ):
            continue
        cpu = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(cpu.shape)}:{cpu.dtype}\n".encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        cpu = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(cpu.shape)}:{cpu.dtype}\n".encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _recalibrate_batchnorm(
    *,
    name: str,
    model: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    calibration_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, object]]:
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=calibration_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="schedule_free_a0_precise_bn",
        seed=int(args.seed) + 400,
    )
    model = model.to(device)
    parameter_before = _parameter_sha256(model)
    state_before = _batchnorm_state_sha256(model)
    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batchnorm_modules:
        raise ValueError("Schedule-Free A0 expected BatchNorm modules.")
    momenta = [module.momentum for module in batchnorm_modules]
    for module in batchnorm_modules:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    ordered_indices = []
    batch_hashes = []
    with torch.no_grad():
        for batch_index, (images_cpu, targets_cpu, metadata) in enumerate(loader):
            if batch_index >= int(args.calibration_batches):
                break
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Calibration metadata is missing sample_index.")
            ordered_indices.extend(int(value) for value in sample_indices.tolist())
            batch_hashes.append(
                _tensor_batch_sha256(images_cpu, targets_cpu, metadata)
            )
            set_seed(int(args.seed) + 20_000 + batch_index, deterministic=True)
            images = images_cpu.to(device=device, non_blocking=True)
            features = _features_from_batch(model, images, metadata, device=device)
            logits = classification_logits_from_features(model, features)
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError(f"Non-finite {name} calibration logits.")
    for module, momentum in zip(batchnorm_modules, momenta):
        module.momentum = momentum
    model.eval()
    tracked = [int(module.num_batches_tracked.item()) for module in batchnorm_modules]
    summary = {
        "name": name,
        "batchnorm_modules": len(batchnorm_modules),
        "batches": len(batch_hashes),
        "rows": len(ordered_indices),
        "order_sha256": _ordered_index_sha256(ordered_indices),
        "batch_hashes": batch_hashes,
        "batch_hash_sequence_sha256": hashlib.sha256(
            "\n".join(batch_hashes).encode("ascii")
        ).hexdigest(),
        "num_batches_tracked": tracked,
        "all_tracked_exact": all(
            value == EXPECTED_CALIBRATION_BATCHES for value in tracked
        ),
        "state_before_sha256": state_before,
        "state_after_sha256": _batchnorm_state_sha256(model),
        "state_changed": state_before != _batchnorm_state_sha256(model),
        "parameters_unchanged": parameter_before == _parameter_sha256(model),
        "loader": loader_summary,
    }
    model = model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, summary


def _probability_difference(
    left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]
) -> Dict[str, float]:
    _validate_prediction_pair(left, right)
    left_by_index = {int(row["sample_index"]): row for row in left}
    values = []
    for row in right:
        sample_index = int(row["sample_index"])
        match = left_by_index[sample_index]
        values.extend(
            abs(float(row[f"prob_{index}"]) - float(match[f"prob_{index}"]))
            for index in range(5)
        )
    return {
        "mean_absolute": float(np.mean(values)) if values else 0.0,
        "maximum_absolute": max(values, default=0.0),
    }


def _validate_prediction_pair(
    left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]
) -> None:
    if not left or len(left) != len(right):
        raise ValueError(
            f"Prediction roles are empty or differ in length: {len(left)}/{len(right)}"
        )
    left_keys = [
        (int(row["sample_index"]), int(row["target"])) for row in left
    ]
    right_keys = [
        (int(row["sample_index"]), int(row["target"])) for row in right
    ]
    if len({sample_index for sample_index, _target in left_keys}) != len(left_keys):
        raise ValueError("Left prediction role contains duplicate sample indices.")
    if len({sample_index for sample_index, _target in right_keys}) != len(right_keys):
        raise ValueError("Right prediction role contains duplicate sample indices.")
    if left_keys != right_keys:
        raise ValueError("Prediction roles differ in sample-index/target order.")


def _checked_comparison(
    *,
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    num_classes: int,
    focus_class: int,
) -> Dict[str, object]:
    _validate_prediction_pair(control_rows, candidate_rows)
    return _comparison(
        control_rows=control_rows,
        candidate_rows=candidate_rows,
        num_classes=num_classes,
        focus_class=focus_class,
    )


def _focus_tp_retention(
    control: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    focus_class: int,
) -> Dict[str, object]:
    _validate_prediction_pair(control, candidate)
    candidate_by_index = {int(row["sample_index"]): row for row in candidate}
    control_tp = 0
    retained = 0
    for row in control:
        if int(row["target"]) != focus_class or int(row["prediction"]) != focus_class:
            continue
        control_tp += 1
        retained += int(
            int(candidate_by_index[int(row["sample_index"])]["prediction"])
            == focus_class
        )
    return {
        "control_tp": control_tp,
        "retained_tp": retained,
        "ratio": float(retained / control_tp) if control_tp else 1.0,
    }


def assess_a0(
    *,
    structural_checks: Mapping[str, bool],
    clean: Mapping[str, object],
    clean_tp_retention: Mapping[str, object],
    illumination: Sequence[Mapping[str, object]],
    illumination_tp_retention: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    delta = clean["delta"]
    transitions = clean["transitions"]
    control_f1 = clean["control"]["per_class_f1"]
    candidate_f1 = clean["candidate"]["per_class_f1"]
    clean_checks = {
        "macro_f1_delta_gte_0p003": float(delta["macro_f1"]) >= 0.003,
        "class1_f1_delta_gte_0p010": float(delta["class1_f1"]) >= 0.010,
        "class1_precision_delta_gte_0p015": float(delta["class1_precision"])
        >= 0.015,
        "class1_recall_delta_gte_minus_0p010": float(delta["class1_recall"])
        >= -0.010,
        "class1_tp_retention_gte_0p97": float(clean_tp_retention["ratio"])
        >= 0.97,
        "restricted_focus_fp_reduction_gte_4": int(
            transitions["restricted_focus_fp_reduction"]
        )
        >= 4,
        "corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "focus_rescues_gte_tp_breaks": int(transitions["focus_fn_rescue"])
        >= int(transitions["focus_tp_break"]),
        "max_nonfocus_f1_drop_lte_0p010": max(
            float(control_f1[index]) - float(candidate_f1[index])
            for index in (0, 2, 3, 4)
        )
        <= 0.010,
    }
    shift_checks = {}
    precision_improvements = 0
    for row, retention in zip(illumination, illumination_tp_retention):
        name = str(row["condition"])
        row_delta = row["delta"]
        row_transitions = row["transitions"]
        shift_checks[f"{name}_class1_precision_nonnegative"] = float(
            row_delta["class1_precision"]
        ) >= 0.0
        shift_checks[f"{name}_class1_f1_delta_gte_minus_0p005"] = float(
            row_delta["class1_f1"]
        ) >= -0.005
        shift_checks[f"{name}_class1_tp_retention_gte_0p95"] = float(
            retention["ratio"]
        ) >= 0.95
        shift_checks[f"{name}_restricted_fp_nonincreasing"] = int(
            row_transitions["restricted_focus_fp_reduction"]
        ) >= 0
        shift_checks[f"{name}_macro_f1_delta_gte_minus_0p005"] = float(
            row_delta["macro_f1"]
        ) >= -0.005
        precision_improvements += int(float(row_delta["class1_precision"]) >= 0.010)
    shift_checks["two_shift_precision_deltas_gte_0p010"] = precision_improvements >= 2
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **clean_checks,
        **shift_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "illumination_checks": shift_checks,
        "failed_checks": failed,
        "automatic_gates_passed": not failed,
        "xai_review_required": not failed,
        "stage_b_smoke_authorized": False,
        "full_train_authorized": False,
    }


def _select_xai_rows(
    *,
    control: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    limit: int,
) -> tuple[list[int], Dict[int, list[str]]]:
    candidate_by_index = {int(row["sample_index"]): row for row in candidate}
    categories: Dict[int, list[str]] = {}
    ranking = []
    for control_row in control:
        sample_index = int(control_row["sample_index"])
        candidate_row = candidate_by_index[sample_index]
        target = int(control_row["target"])
        control_prediction = int(control_row["prediction"])
        candidate_prediction = int(candidate_row["prediction"])
        labels = []
        if target in (0, 2, 4) and control_prediction == 1 and candidate_prediction != 1:
            labels.append("restricted_fp_removal")
        if target in (0, 2, 4) and control_prediction != 1 and candidate_prediction == 1:
            labels.append("restricted_fp_creation")
        if target == 1 and control_prediction == 1 and candidate_prediction != 1:
            labels.append("class1_tp_break")
        if target == 1 and control_prediction != 1 and candidate_prediction == 1:
            labels.append("class1_fn_rescue")
        if control_prediction != candidate_prediction:
            labels.append("changed_decision")
        if labels:
            categories[sample_index] = labels
        delta = float(candidate_row["prob_1"]) - float(control_row["prob_1"])
        ranking.append((abs(delta), sample_index))

    selected = []
    for category in (
        "restricted_fp_removal",
        "restricted_fp_creation",
        "class1_tp_break",
        "class1_fn_rescue",
    ):
        candidates = [
            (score, sample_index)
            for score, sample_index in ranking
            if category in categories.get(sample_index, ())
        ]
        if candidates:
            selected.append(max(candidates)[1])
    for _score, sample_index in sorted(ranking, reverse=True):
        if sample_index not in selected:
            selected.append(sample_index)
        if len(selected) >= int(limit):
            break
    selected = selected[: int(limit)]
    return selected, {index: categories.get(index, []) for index in selected}


def _xai_maps(
    *,
    model: nn.Module,
    images_cpu: Tensor,
    metadata: Mapping[str, object],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
) -> Dict[int, Dict[str, object]]:
    model = copy.deepcopy(model).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    activation: Dict[str, Tensor] = {}
    attention_activation: Dict[str, Tensor] = {}

    def hook(_module, _inputs, output: Tensor) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Patch embedding XAI hook requires a tensor output.")
        activation["value"] = output
        output.retain_grad()

    attention_spec = resolve_attention_hook(model, 0)
    if attention_spec is None:
        raise ValueError("Native layer-0 QKV hook is unavailable.")

    def attention_hook(_module, _inputs, output: Tensor) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Native attention QKV hook requires a tensor output.")
        attention_activation["qkv"] = output

    handles = [
        model.patch_embed.proj.register_forward_hook(hook),
        attention_spec.module.register_forward_hook(attention_hook),
    ]
    images = images_cpu.detach().to(device=device).requires_grad_(True)
    bbox = metadata.get("bbox")
    display_bbox = metadata.get("crop_bbox")
    if not torch.is_tensor(display_bbox):
        display_bbox = bbox
    try:
        model.zero_grad(set_to_none=True)
        features = _features_from_batch(model, images, metadata, device=device)
        logits = classification_logits_from_features(model, features).float()
        qkv = attention_activation.get("qkv")
        if not torch.is_tensor(qkv):
            raise ValueError("Native layer-0 QKV activation is unavailable.")
        attention = reconstruct_attention(qkv.detach(), attention_spec)
        logits[:, 1].sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    value = activation.get("value")
    if not torch.is_tensor(value) or not torch.is_tensor(value.grad):
        raise ValueError("Patch embedding activation or gradient is unavailable.")
    if value.ndim != 4 or tuple(value.shape[-2:]) != (16, 16):
        raise ValueError(f"Unexpected patch embedding shape for XAI: {tuple(value.shape)}")
    gradcam = torch.relu((value.float() * value.grad.float()).sum(dim=1))
    gradcam = _normalize_maps(gradcam).detach().cpu()
    grid_height, grid_width = (int(value.shape[-2]), int(value.shape[-1]))
    prefix = int(attention_spec.prefix_tokens)
    patch_count = grid_height * grid_width
    if int(attention.size(-1)) != prefix + patch_count:
        raise ValueError(
            "Layer-0 attention token layout differs from the locked full-grid layout."
        )
    native = attention[:, :, 0, prefix : prefix + patch_count].float().mean(dim=1)
    native = _normalize_maps(
        native.reshape(-1, grid_height, grid_width)
    ).detach().cpu()
    probabilities = logits.detach().softmax(dim=1).cpu()
    sample_indices = metadata.get("sample_index")
    if not torch.is_tensor(sample_indices):
        raise ValueError("XAI metadata is missing sample_index.")
    records: Dict[int, Dict[str, object]] = {}
    for position, sample_index in enumerate(sample_indices.tolist()):
        bbox_value = (
            display_bbox[position].detach().float().cpu().tolist()
            if torch.is_tensor(display_bbox)
            else []
        )
        records[int(sample_index)] = {
            "rgb": _rgb_from_tensor(images_cpu[position], mean=mean, std=std),
            "native_attention": native[position].numpy().astype(np.float32),
            "gradcam": gradcam[position].numpy().astype(np.float32),
            "bbox": bbox_value,
            "prediction": int(probabilities[position].argmax().item()),
            "probabilities": [float(value) for value in probabilities[position].tolist()],
        }
    del model, images
    gc.collect()
    torch.cuda.empty_cache()
    return records


def _draw_bbox(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    output = image.copy()
    if len(bbox) < 4:
        return output
    cx, cy, width, height = [float(value) for value in bbox[:4]]
    x1 = int(round((cx - width / 2.0) * output.width))
    y1 = int(round((cy - height / 2.0) * output.height))
    x2 = int(round((cx + width / 2.0) * output.width))
    y2 = int(round((cy + height / 2.0) * output.height))
    draw = ImageDraw.Draw(output)
    draw.rectangle((x1, y1, x2, y2), outline=(25, 230, 80), width=3)
    return output


def _render_xai_contact_sheet(
    *,
    path: Path,
    condition: str,
    selected: Sequence[int],
    categories: Mapping[int, Sequence[str]],
    source_rows: Sequence[CleanTrainRow],
    locked_control: Sequence[Mapping[str, object]],
    locked_candidate: Sequence[Mapping[str, object]],
    control_maps: Mapping[int, Mapping[str, object]],
    candidate_maps: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    tile = 256
    caption = 58
    columns = 5
    canvas = Image.new("RGB", (columns * tile, len(selected) * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    control_by_index = {int(row["sample_index"]): row for row in locked_control}
    candidate_by_index = {int(row["sample_index"]): row for row in locked_candidate}
    rows = []
    prediction_mismatches = 0
    for row_index, sample_index in enumerate(selected):
        control = control_maps[sample_index]
        candidate = candidate_maps[sample_index]
        control_locked = control_by_index[sample_index]
        candidate_locked = candidate_by_index[sample_index]
        prediction_mismatches += int(
            int(control["prediction"]) != int(control_locked["prediction"])
        )
        prediction_mismatches += int(
            int(candidate["prediction"]) != int(candidate_locked["prediction"])
        )
        rgb = _draw_bbox(
            Image.fromarray(np.asarray(control["rgb"], dtype=np.uint8)).convert("RGB"),
            control["bbox"],
        )
        panels = [
            rgb,
            _heat_overlay(control["rgb"], control["native_attention"]),
            _heat_overlay(candidate["rgb"], candidate["native_attention"]),
            _heat_overlay(control["rgb"], control["gradcam"]),
            _heat_overlay(candidate["rgb"], candidate["gradcam"]),
        ]
        y = row_index * (tile + caption)
        for column, panel in enumerate(panels):
            canvas.paste(panel.resize((tile, tile), Image.Resampling.BILINEAR), (column * tile, y))
        source = source_rows[sample_index]
        label = (
            f"{condition} i={sample_index} t={source.target} "
            f"c={int(control_locked['prediction'])} sf={int(candidate_locked['prediction'])} "
            f"dp1={float(candidate_locked['prob_1']) - float(control_locked['prob_1']):+.4f} "
            f"{','.join(categories.get(sample_index, ()))}"
        )
        draw.text((4, y + tile + 3), label, fill="black")
        draw.text(
            (4, y + tile + 25),
            "image+bbox | control attention | candidate attention | control Grad-CAM | candidate Grad-CAM",
            fill=(45, 45, 45),
        )
        rows.append(
            {
                "sample_index": sample_index,
                "target": source.target,
                "categories": list(categories.get(sample_index, ())),
                "control_prediction": int(control_locked["prediction"]),
                "candidate_prediction": int(candidate_locked["prediction"]),
                "class1_probability_delta": float(candidate_locked["prob_1"])
                - float(control_locked["prob_1"]),
            }
        )
    canvas.save(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "rows": rows,
        "prediction_mismatches": prediction_mismatches,
        "passed": prediction_mismatches == 0 and len(rows) == len(selected),
    }


def _run_xai_review_artifacts(
    *,
    output_dir: Path,
    control_model: nn.Module,
    candidate_model: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    source_rows: Sequence[CleanTrainRow],
    condition_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    checkpoint: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    mean, std = checkpoint_input_normalization(checkpoint)
    condition_spec = {
        "clean": (1.0, 1.0),
        **{
            str(name): (float(brightness), float(contrast))
            for name, brightness, contrast in LOCKED_LIGHTING_CONDITIONS
        },
    }
    pages = {}
    arrays: Dict[str, np.ndarray] = {}
    for condition_index, (condition, (brightness, contrast)) in enumerate(
        condition_spec.items()
    ):
        locked = condition_rows[condition]
        selected, categories = _select_xai_rows(
            control=locked["control"],
            candidate=locked["candidate"],
            limit=int(args.contact_rows),
        )
        if not selected:
            raise ValueError(f"No Schedule-Free XAI rows selected for {condition}.")
        loader, loader_summary = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=selected,
            brightness=brightness,
            contrast=contrast,
            batch_size=len(selected),
            num_workers=0,
            context=f"schedule_free_xai_{condition}",
            seed=int(args.seed) + 900 + condition_index,
        )
        images_cpu, _targets, metadata = next(iter(loader))
        control_maps = _xai_maps(
            model=control_model,
            images_cpu=images_cpu,
            metadata=metadata,
            mean=mean,
            std=std,
            device=device,
        )
        candidate_maps = _xai_maps(
            model=candidate_model,
            images_cpu=images_cpu,
            metadata=metadata,
            mean=mean,
            std=std,
            device=device,
        )
        for sample_index in selected:
            prefix = f"{condition}_{sample_index}"
            arrays[f"{prefix}_rgb"] = np.asarray(
                control_maps[sample_index]["rgb"], dtype=np.uint8
            )
            for role, records in (("control", control_maps), ("candidate", candidate_maps)):
                arrays[f"{prefix}_{role}_attention"] = np.asarray(
                    records[sample_index]["native_attention"], dtype=np.float32
                )
                arrays[f"{prefix}_{role}_gradcam"] = np.asarray(
                    records[sample_index]["gradcam"], dtype=np.float32
                )
        page_path = output_dir / f"contact_sheet_{condition}.png"
        page = _render_xai_contact_sheet(
            path=page_path,
            condition=condition,
            selected=selected,
            categories=categories,
            source_rows=source_rows,
            locked_control=locked["control"],
            locked_candidate=locked["candidate"],
            control_maps=control_maps,
            candidate_maps=candidate_maps,
        )
        page["loader"] = loader_summary
        pages[condition] = page
    tensor_path = output_dir / "xai_tensors.npz"
    np.savez_compressed(tensor_path, **arrays)
    manifest = {
        "pages": pages,
        "tensor_path": tensor_path.name,
        "tensor_sha256": _sha256(tensor_path),
        "all_pages_passed": all(bool(value["passed"]) for value in pages.values()),
        "manual_review_required": True,
    }
    manifest_path = output_dir / "xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    manifest["manifest_path"] = manifest_path.name
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _write_prediction_rows(
    path: Path,
    *,
    source_rows: Sequence[CleanTrainRow],
    condition_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    role_names = sorted(next(iter(condition_rows.values())).keys())
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
    ]
    for role in role_names:
        fields.append(f"{role}_prediction")
        fields.extend(f"{role}_prob_{index}" for index in range(5))
    source = {row.sample_index: row for row in source_rows}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, roles in condition_rows.items():
            indexed = {
                role: {int(row["sample_index"]): row for row in rows}
                for role, rows in roles.items()
            }
            sample_indices = sorted(next(iter(indexed.values())))
            for sample_index in sample_indices:
                source_row = source[sample_index]
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source_row.source_stem,
                    "image_path": str(source_row.image_path),
                    "fold": source_row.fold,
                    "target": source_row.target,
                    "keeper_prediction": source_row.keeper_prediction,
                }
                for role in role_names:
                    value = indexed[role][sample_index]
                    output[f"{role}_prediction"] = int(value["prediction"])
                    for class_index in range(5):
                        output[f"{role}_prob_{class_index}"] = float(
                            value[f"prob_{class_index}"]
                        )
                writer.writerow(output)


def _rows_from_csv(
    path: Path, *, role: str, condition: str
) -> list[Dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "target",
            f"{role}_prediction",
            *{f"{role}_prob_{index}" for index in range(5)},
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Replay CSV lacks columns: {sorted(missing)}")
        for raw in reader:
            if str(raw["condition"]) != condition:
                continue
            row: Dict[str, object] = {
                "sample_index": int(raw["sample_index"]),
                "target": int(raw["target"]),
                "prediction": int(raw[f"{role}_prediction"]),
            }
            for class_index in range(5):
                row[f"prob_{class_index}"] = float(raw[f"{role}_prob_{class_index}"])
            rows.append(row)
    return rows


def _condition_rows_aligned(
    condition_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> bool:
    expected_conditions = {"clean", *[name for name, _brightness, _contrast in LOCKED_LIGHTING_CONDITIONS]}
    expected_roles = {"raw", "control", "candidate", "candidate_z"}
    if set(condition_rows) != expected_conditions:
        return False
    clean_keys: Optional[list[tuple[int, int]]] = None
    try:
        for roles in condition_rows.values():
            if set(roles) != expected_roles:
                return False
            control = roles["control"]
            for role in expected_roles.difference({"control"}):
                _validate_prediction_pair(control, roles[role])
            keys = [
                (int(row["sample_index"]), int(row["target"])) for row in control
            ]
            if len(keys) != EXPECTED_HOLDOUT_ROWS:
                return False
            if clean_keys is None:
                clean_keys = keys
            elif keys != clean_keys:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _prediction_artifact_integrity(
    path: Path,
    *,
    expected_order_sha256: str,
    expected_rows: int = EXPECTED_HOLDOUT_ROWS,
) -> Dict[str, object]:
    roles = ("raw", "control", "candidate", "candidate_z")
    conditions = ("clean", *[name for name, _brightness, _contrast in LOCKED_LIGHTING_CONDITIONS])
    required = {
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        *{f"{role}_prediction" for role in roles},
        *{
            f"{role}_prob_{class_index}"
            for role in roles
            for class_index in range(5)
        },
    }
    grouped: Dict[str, list[Dict[str, object]]] = {name: [] for name in conditions}
    probability_rows_valid = True
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prediction artifact lacks columns: {sorted(missing)}")
        unknown_conditions = set()
        for raw in reader:
            condition = str(raw["condition"])
            if condition not in grouped:
                unknown_conditions.add(condition)
                continue
            record: Dict[str, object] = {
                "sample_index": int(raw["sample_index"]),
                "source_stem": str(raw["source_stem"]),
                "image_path": str(raw["image_path"]),
                "fold": int(raw["fold"]),
                "target": int(raw["target"]),
                "keeper_prediction": int(raw["keeper_prediction"]),
            }
            for role in roles:
                prediction = int(raw[f"{role}_prediction"])
                probabilities = np.asarray(
                    [float(raw[f"{role}_prob_{index}"]) for index in range(5)],
                    dtype=np.float64,
                )
                probability_rows_valid = bool(
                    probability_rows_valid
                    and 0 <= prediction < 5
                    and bool(np.isfinite(probabilities).all())
                    and bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
                    and abs(float(probabilities.sum()) - 1.0) <= 1e-5
                    and prediction == int(probabilities.argmax())
                )
                record[f"{role}_prediction"] = prediction
            grouped[condition].append(record)
    reference = grouped["clean"]
    reference_signature = [
        (
            int(row["sample_index"]),
            str(row["source_stem"]),
            str(row["image_path"]),
            int(row["fold"]),
            int(row["target"]),
            int(row["keeper_prediction"]),
        )
        for row in reference
    ]
    condition_counts_exact = all(
        len(grouped[name]) == int(expected_rows) for name in conditions
    )
    signatures_identical = all(
        [
            (
                int(row["sample_index"]),
                str(row["source_stem"]),
                str(row["image_path"]),
                int(row["fold"]),
                int(row["target"]),
                int(row["keeper_prediction"]),
            )
            for row in grouped[name]
        ]
        == reference_signature
        for name in conditions
    )
    sample_indices = [int(row["sample_index"]) for row in reference]
    unique_indices = len(set(sample_indices)) == len(sample_indices)
    order_sha256 = _ordered_index_sha256(sample_indices)
    targets_valid = all(0 <= int(row["target"]) < 5 for row in reference)
    keeper_predictions_valid = all(
        0 <= int(row["keeper_prediction"]) < 5 for row in reference
    )
    raw_mismatches = sum(
        int(row["raw_prediction"]) != int(row["keeper_prediction"])
        for row in reference
    )
    checks = {
        "unknown_conditions_zero": not unknown_conditions,
        "condition_counts_exact": condition_counts_exact,
        "condition_signatures_identical": signatures_identical,
        "sample_indices_unique": unique_indices,
        "holdout_order_matches": order_sha256 == str(expected_order_sha256),
        "targets_valid": targets_valid,
        "keeper_predictions_valid": keeper_predictions_valid,
        "probability_rows_valid": probability_rows_valid,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "condition_counts": {name: len(grouped[name]) for name in conditions},
        "holdout_order_sha256": order_sha256,
        "raw_keeper_prediction_mismatches": raw_mismatches,
    }


def _provenance_locked(provenance: Mapping[str, object]) -> bool:
    expected_hashes = {
        "checkpoint": LOCKED_KEEPER_SHA256,
        "launcher_args": LOCKED_LAUNCHER_ARGS_SHA256,
        "data": LOCKED_DATA_SHA256,
        "cidt_summary": LOCKED_CIDT_SUMMARY_SHA256,
        "cidt_predictions": LOCKED_CIDT_PREDICTIONS_SHA256,
        "protocol": LOCKED_PROTOCOL_SHA256,
        "paper": LOCKED_PAPER_SHA256,
        "optimizer": LOCKED_OPTIMIZER_SHA256,
        "reference": LOCKED_REFERENCE_SHA256,
        "license": LOCKED_LICENSE_SHA256,
    }
    expected_blobs = {
        "optimizer": LOCKED_OPTIMIZER_BLOB,
        "reference": LOCKED_REFERENCE_BLOB,
        "paper_optimizer": LOCKED_PAPER_OPTIMIZER_BLOB,
        "license": LOCKED_LICENSE_BLOB,
    }
    return bool(
        provenance.get("hashes") == expected_hashes
        and str(provenance.get("official_commit", "")).lower()
        == LOCKED_OFFICIAL_COMMIT
        and str(provenance.get("official_tree", "")).lower() == LOCKED_OFFICIAL_TREE
        and provenance.get("official_blobs") == expected_blobs
        and provenance.get("official_worktree_clean") is True
    )


def _config_locked(config: Mapping[str, object]) -> bool:
    try:
        namespace = argparse.Namespace(**dict(config))
        return bool(
            _locked_args_exact(namespace)
            and namespace.preflight_only is False
            and namespace.replay_summary is None
            and namespace.finalize_visual_review is False
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _train_transform_locked(value: Mapping[str, object]) -> bool:
    settings = value.get("settings")
    digest = str(value.get("settings_sha256", ""))
    if not isinstance(settings, Mapping):
        return False
    serialized = json.dumps(dict(settings), sort_keys=True, separators=(",", ":"))
    return bool(
        value.get("train_augmentation_enabled") is True
        and value.get("source") == "keeper_checkpoint_augmentation_config"
        and hashlib.sha256(serialized.encode("utf-8")).hexdigest() == digest
        and digest == LOCKED_TRAIN_TRANSFORM_SHA256
        and int(settings.get("image_size", 0)) == 256
        and float(settings.get("random_resized_crop_probability", 0.0)) > 0.0
        and float(settings.get("background_suppression_probability", 0.0)) > 0.0
    )


def _build_structural_checks(
    *,
    equation: Mapping[str, object],
    raw_mismatches: int,
    control_result: Mapping[str, object],
    candidate_result: Mapping[str, object],
    control_bn: Mapping[str, object],
    candidate_bn: Mapping[str, object],
    candidate_z_bn: Mapping[str, object],
    x_z_difference: Mapping[str, object],
    runtime_ratio: float,
    peak_vram_ratio: float,
    export: Mapping[str, object],
    source_overlap: int,
    prediction_rows_aligned: bool,
    provenance: Mapping[str, object],
    config: Mapping[str, object],
    dataset: Mapping[str, object],
) -> Dict[str, bool]:
    required_groups = sorted(control_result["parameter_group_hashes_before"])
    return {
        "provenance_locked": _provenance_locked(provenance),
        "config_locked": _config_locked(config),
        "train_augmentation_locked": _train_transform_locked(
            dataset.get("train_transform", {})
        ),
        "equation_diagnostics_passed": bool(equation["passed"]),
        "raw_cidt_prediction_mismatches_zero": int(raw_mismatches) == 0,
        "prediction_rows_aligned": bool(prediction_rows_aligned),
        "train_budget_exact": all(
            int(result["train_batches"]) == EXPECTED_TRAIN_BATCHES
            and int(result["train_rows"]) == EXPECTED_TRAIN_ROWS_USED
            and len(result["batch_hashes"]) == EXPECTED_TRAIN_BATCHES
            for result in (control_result, candidate_result)
        ),
        "train_orders_identical": control_result["train_order_sha256"]
        == candidate_result["train_order_sha256"],
        "train_batches_byte_identical": control_result["batch_hashes"]
        == candidate_result["batch_hashes"],
        "initial_states_identical": control_result["initial_state_sha256"]
        == candidate_result["initial_state_sha256"],
        "all_gradients_finite": bool(control_result["all_gradients_finite"])
        and bool(candidate_result["all_gradients_finite"]),
        "all_optimizer_state_finite": bool(control_result["optimizer_state_finite"])
        and bool(candidate_result["optimizer_state_finite"]),
        "all_parameter_groups_moved": all(
            bool(control_result["parameter_group_movement"].get(group, False))
            and bool(candidate_result["parameter_group_movement"].get(group, False))
            for group in required_groups
        ),
        "all_gradient_groups_seen": all(
            bool(control_result["gradient_seen"].get(group, False))
            and bool(candidate_result["gradient_seen"].get(group, False))
            for group in required_groups
        ),
        "optimizer_eval_mode": bool(candidate_result["mode"]["optimizer_eval_mode"]),
        # Retain the v1 artifact key names, but apply the protocol's declared
        # FP32 tolerance rather than requiring bit identity after two affine
        # train/eval parameter transitions.
        "mode_y_roundtrip_exact": _mode_roundtrip_within_fp32_tolerance(
            candidate_result["mode"]["y_roundtrip_max_abs_error"]
        ),
        "mode_x_roundtrip_exact": _mode_roundtrip_within_fp32_tolerance(
            candidate_result["mode"]["x_roundtrip_max_abs_error"]
        ),
        "x_y_parameters_distinct": float(
            candidate_result["mode"]["x_y_max_abs_difference"]
        )
        > 0.0,
        "x_z_parameters_distinct": float(
            candidate_result["mode"]["x_z_max_abs_difference"]
        )
        > 0.0,
        "x_z_probabilities_distinct": float(x_z_difference["maximum_absolute"])
        > 1e-6,
        "checkpoint_replay_exact": float(
            candidate_result["mode"]["checkpoint_replay"][
                "model_state_max_abs_error"
            ]
        )
        == 0.0
        and float(candidate_result["mode"]["checkpoint_replay"]["logit_max_abs_error"])
        == 0.0
        and bool(candidate_result["mode"]["checkpoint_replay"]["argmax_match"])
        and bool(
            candidate_result["mode"]["checkpoint_replay"]["optimizer_eval_mode"]
        ),
        "z_state_complete_and_finite": bool(candidate_result["z"]["all_finite"])
        and int(candidate_result["z"]["parameter_tensors"]) > 0
        and int(candidate_result["z"]["parameter_tensors"])
        <= int(candidate_result["z"]["optimizer_parameter_tensors"]),
        "precise_bn_orders_identical": control_bn["batch_hashes"]
        == candidate_bn["batch_hashes"]
        == candidate_z_bn["batch_hashes"],
        "precise_bn_rows_exact": all(
            int(value["rows"]) == EXPECTED_CALIBRATION_ROWS
            for value in (control_bn, candidate_bn, candidate_z_bn)
        ),
        "precise_bn_counts_exact": all(
            bool(value["all_tracked_exact"])
            for value in (control_bn, candidate_bn, candidate_z_bn)
        ),
        "precise_bn_parameters_unchanged": all(
            bool(value["parameters_unchanged"])
            for value in (control_bn, candidate_bn, candidate_z_bn)
        ),
        "runtime_ratio_lte_1p20": float(runtime_ratio) <= MAX_RUNTIME_RATIO,
        "peak_vram_lte_6p50_gib": float(candidate_result["peak_vram_gib"])
        <= MAX_PEAK_VRAM_GIB,
        "peak_vram_ratio_lte_1p15": float(peak_vram_ratio)
        <= MAX_PEAK_VRAM_RATIO,
        "onnx_succeeded": bool(export.get("succeeded", False)),
        "onnx_finite": bool(export.get("finite", False)),
        "onnx_error_lte_1e5": float(export.get("maximum_absolute_error", 1e9))
        <= MAX_ONNX_ERROR,
        "onnx_argmax_match": bool(export.get("argmax_match", False)),
        "source_overlap_zero": int(source_overlap) == 0,
        "validation_predictions_unused": bool(
            provenance.get("validation_predictions_used") is False
            and dataset.get("validation_predictions_used") is False
            and dataset.get("cidt_validation_predictions_used") is False
        ),
        "test_data_unused": bool(
            provenance.get("test_data_used") is False
            and dataset.get("test_data_used") is False
            and dataset.get("cidt_test_data_used") is False
        ),
    }


def _json_max_abs_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_json_max_abs_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return math.inf
        if len(left) != len(right):
            return math.inf
        return max(
            (_json_max_abs_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left is right else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def replay_summary(summary_path: Path) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for name, relative_path in summary["artifacts"].items():
        artifact_path = summary_path.parent / str(relative_path)
        if _sha256(artifact_path) != str(summary["artifact_hashes"][name]):
            raise ValueError(f"{name} artifact hash differs during replay.")
    prediction_path = summary_path.parent / str(summary["artifacts"]["predictions"])
    dataset = summary["dataset"]
    artifact_integrity = _prediction_artifact_integrity(
        prediction_path,
        expected_order_sha256=str(dataset["holdout_order_sha256"]),
    )
    clean_control = _rows_from_csv(prediction_path, role="control", condition="clean")
    clean_candidate = _rows_from_csv(
        prediction_path, role="candidate", condition="clean"
    )
    clean_candidate_z = _rows_from_csv(
        prediction_path, role="candidate_z", condition="clean"
    )
    clean = _checked_comparison(
        control_rows=clean_control,
        candidate_rows=clean_candidate,
        num_classes=5,
        focus_class=1,
    )
    clean_retention = _focus_tp_retention(
        clean_control, clean_candidate, focus_class=1
    )
    x_z_difference = _probability_difference(clean_candidate_z, clean_candidate)
    x_z_comparison = _checked_comparison(
        control_rows=clean_candidate_z,
        candidate_rows=clean_candidate,
        num_classes=5,
        focus_class=1,
    )
    illumination = []
    retention = []
    for condition, brightness, contrast in LOCKED_LIGHTING_CONDITIONS:
        control = _rows_from_csv(
            prediction_path, role="control", condition=condition
        )
        candidate = _rows_from_csv(
            prediction_path, role="candidate", condition=condition
        )
        value = _checked_comparison(
            control_rows=control,
            candidate_rows=candidate,
            num_classes=5,
            focus_class=1,
        )
        value.update(
            {
                "condition": condition,
                "brightness": float(brightness),
                "contrast": float(contrast),
            }
        )
        illumination.append(value)
        retention.append(_focus_tp_retention(control, candidate, focus_class=1))
    precalibration_path = summary_path.parent / str(
        summary["artifacts"]["precalibration_predictions"]
    )
    precalibration_control = _rows_from_csv(
        precalibration_path, role="control", condition="clean"
    )
    precalibration_candidate = _rows_from_csv(
        precalibration_path, role="candidate", condition="clean"
    )
    precalibration = _checked_comparison(
        control_rows=precalibration_control,
        candidate_rows=precalibration_candidate,
        num_classes=5,
        focus_class=1,
    )
    control_result = summary["control"]
    candidate_result = summary["candidate"]
    runtime_ratio = float(
        candidate_result["median_step_seconds_excluding_warmup"]
        / max(float(control_result["median_step_seconds_excluding_warmup"]), 1e-12)
    )
    peak_vram_ratio = float(
        candidate_result["peak_vram_gib"]
        / max(float(control_result["peak_vram_gib"]), 1e-12)
    )
    batchnorm = summary["batchnorm"]
    structural_checks = _build_structural_checks(
        equation=summary["equation"],
        raw_mismatches=int(artifact_integrity["raw_keeper_prediction_mismatches"]),
        control_result=control_result,
        candidate_result=candidate_result,
        control_bn=batchnorm["control"],
        candidate_bn=batchnorm["candidate_x"],
        candidate_z_bn=batchnorm["candidate_z"],
        x_z_difference=x_z_difference,
        runtime_ratio=runtime_ratio,
        peak_vram_ratio=peak_vram_ratio,
        export=summary["export"],
        source_overlap=int(dataset["source_overlap"]),
        prediction_rows_aligned=bool(artifact_integrity["passed"]),
        provenance=summary["provenance"],
        config=summary["config"],
        dataset=dataset,
    )
    decision = assess_a0(
        structural_checks=structural_checks,
        clean=clean,
        clean_tp_retention=clean_retention,
        illumination=illumination,
        illumination_tp_retention=retention,
    )
    differences = {
        "raw_cidt_prediction_mismatches": _json_max_abs_difference(
            int(artifact_integrity["raw_keeper_prediction_mismatches"]),
            summary["raw_cidt_prediction_mismatches"],
        ),
        "precalibration": _json_max_abs_difference(
            precalibration, summary["precalibration"]
        ),
        "clean": _json_max_abs_difference(clean, summary["clean"]),
        "clean_tp_retention": _json_max_abs_difference(
            clean_retention, summary["clean_tp_retention"]
        ),
        "illumination": _json_max_abs_difference(
            illumination, summary["illumination"]
        ),
        "illumination_tp_retention": _json_max_abs_difference(
            retention, summary["illumination_tp_retention"]
        ),
        "candidate_x_vs_z_probability_difference": _json_max_abs_difference(
            x_z_difference,
            summary["candidate_x_vs_z"]["probability_difference"],
        ),
        "candidate_x_vs_z_comparison": _json_max_abs_difference(
            x_z_comparison, summary["candidate_x_vs_z"]["comparison"]
        ),
        "decision": _json_max_abs_difference(decision, summary["decision"]),
    }
    maximum_difference = max(differences.values())
    return {
        "summary": str(summary_path),
        "prediction_rows": sum(1 for _ in prediction_path.open("r", encoding="utf-8"))
        - 1,
        "prediction_artifact_integrity": artifact_integrity,
        "recomputed_structural_checks": structural_checks,
        "differences": differences,
        "maximum_absolute_difference": maximum_difference,
        "passed": bool(
            artifact_integrity["passed"]
            and maximum_difference <= 1e-12
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked Schedule-Free A0 requires CUDA.")
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    equation = _equation_diagnostics(paths)
    if not bool(equation["passed"]):
        raise ValueError(f"Schedule-Free equation diagnostics failed: {equation}")
    output_dir = _prepare_output_dir(args.output_dir)

    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    cidt_test_data_used = bool(cidt_summary.get("test_data_used", True))
    cidt_validation_predictions_used = bool(
        cidt_summary.get("validation_predictions_used", True)
    )
    if cidt_test_data_used:
        raise ValueError("CIDT provenance indicates test data use.")
    if cidt_validation_predictions_used:
        raise ValueError("CIDT provenance indicates validation prediction use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])

    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    source_config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if len(class_names) != 5 or not isinstance(source_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    prototype = create_model(num_classes=5, model_config=source_config)
    load_model_state(prototype, dict(state), strict=True)
    prototype.eval()
    parameter_count = sum(int(value.numel()) for value in prototype.parameters())

    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(paths["data"], class_name_mode="raw", expected_num_classes=5)
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper.")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    sample_paths = [Path(value).resolve() for value in base_dataset.sample_paths()]
    if len(sample_paths) != len(rows):
        raise ValueError("Dataset/CIDT row count mismatch.")
    for index, (row, path) in enumerate(zip(rows, sample_paths)):
        if row.sample_index != index or row.image_path != path:
            raise ValueError(f"Dataset/CIDT alignment differs at row {index}.")

    fold = int(args.fold)
    fit_indices = [row.sample_index for row in rows if row.fold != fold]
    holdout_indices = [row.sample_index for row in rows if row.fold == fold]
    fit_sources = {rows[index].source_stem for index in fit_indices}
    holdout_sources = {rows[index].source_stem for index in holdout_indices}
    source_overlap = fit_sources.intersection(holdout_sources)
    shuffled_fit = np.asarray(fit_indices, dtype=np.int64)
    np.random.default_rng(int(args.seed)).shuffle(shuffled_fit)
    train_indices = shuffled_fit[:EXPECTED_TRAIN_ROWS_USED].tolist()
    calibration_indices = fit_indices[:EXPECTED_CALIBRATION_ROWS]
    holdout_counts = tuple(
        sum(rows[index].target == class_index for index in holdout_indices)
        for class_index in range(5)
    )
    if (
        len(rows) != EXPECTED_TRAIN_ROWS
        or len(fit_indices) != EXPECTED_FIT_ROWS
        or len(holdout_indices) != EXPECTED_HOLDOUT_ROWS
        or len(train_indices) != EXPECTED_TRAIN_ROWS_USED
        or len(calibration_indices) != EXPECTED_CALIBRATION_ROWS
        or holdout_counts != EXPECTED_HOLDOUT_CLASS_COUNTS
        or source_overlap
    ):
        raise ValueError("Locked train/fold/source-disjoint contract differs.")
    train_transform, train_transform_summary = _build_locked_train_transform(checkpoint)
    transform = _build_eval_transform(semantics)
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="schedule_free_a0_holdout",
        seed=int(args.seed) + 100,
    )

    raw_gpu = copy.deepcopy(prototype).to(device).eval()
    raw_predictions = _predict_fp32(model=raw_gpu, loader=holdout_loader, device=device)
    raw_mismatches = sum(
        int(row["prediction"])
        != rows[int(row["sample_index"])].keeper_prediction
        for row in raw_predictions
    )
    del raw_gpu
    gc.collect()
    torch.cuda.empty_cache()

    control_model, _unused_control_z, control_result = _train_variant(
        name="control",
        schedule_free=False,
        prototype=prototype,
        base_dataset=base_dataset,
        transform=train_transform,
        train_indices=train_indices,
        args=args,
        paths=paths,
        device=device,
    )
    candidate_model, candidate_z_model, candidate_result = _train_variant(
        name="candidate",
        schedule_free=True,
        prototype=prototype,
        base_dataset=base_dataset,
        transform=train_transform,
        train_indices=train_indices,
        args=args,
        paths=paths,
        device=device,
    )
    if candidate_z_model is None:
        raise RuntimeError("Candidate fast iterate was not reconstructed.")

    control_pre_gpu = control_model.to(device).eval()
    candidate_pre_gpu = candidate_model.to(device).eval()
    precalibration = {
        "control": _predict_fp32(
            model=control_pre_gpu, loader=holdout_loader, device=device
        ),
        "candidate": _predict_fp32(
            model=candidate_pre_gpu, loader=holdout_loader, device=device
        ),
    }
    control_model = control_pre_gpu.cpu().eval()
    candidate_model = candidate_pre_gpu.cpu().eval()
    del control_pre_gpu, candidate_pre_gpu
    gc.collect()
    torch.cuda.empty_cache()

    control_model, control_bn = _recalibrate_batchnorm(
        name="control",
        model=control_model,
        base_dataset=base_dataset,
        transform=transform,
        calibration_indices=calibration_indices,
        args=args,
        device=device,
    )
    candidate_model, candidate_bn = _recalibrate_batchnorm(
        name="candidate_x",
        model=candidate_model,
        base_dataset=base_dataset,
        transform=transform,
        calibration_indices=calibration_indices,
        args=args,
        device=device,
    )
    candidate_z_model, candidate_z_bn = _recalibrate_batchnorm(
        name="candidate_z",
        model=candidate_z_model,
        base_dataset=base_dataset,
        transform=transform,
        calibration_indices=calibration_indices,
        args=args,
        device=device,
    )

    control_gpu = control_model.to(device).eval()
    candidate_gpu = candidate_model.to(device).eval()
    candidate_z_gpu = candidate_z_model.to(device).eval()
    clean_rows = {
        "raw": raw_predictions,
        "control": _predict_fp32(model=control_gpu, loader=holdout_loader, device=device),
        "candidate": _predict_fp32(
            model=candidate_gpu, loader=holdout_loader, device=device
        ),
        "candidate_z": _predict_fp32(
            model=candidate_z_gpu, loader=holdout_loader, device=device
        ),
    }
    clean = _checked_comparison(
        control_rows=clean_rows["control"],
        candidate_rows=clean_rows["candidate"],
        num_classes=5,
        focus_class=1,
    )
    clean_tp_retention = _focus_tp_retention(
        clean_rows["control"], clean_rows["candidate"], focus_class=1
    )
    x_z_difference = _probability_difference(
        clean_rows["candidate_z"], clean_rows["candidate"]
    )
    x_z_comparison = _checked_comparison(
        control_rows=clean_rows["candidate_z"],
        candidate_rows=clean_rows["candidate"],
        num_classes=5,
        focus_class=1,
    )

    condition_rows: Dict[str, Dict[str, Sequence[Mapping[str, object]]]] = {
        "clean": clean_rows
    }
    illumination = []
    illumination_tp_retention = []
    illumination_loaders = {}
    for condition_index, (condition, brightness, contrast) in enumerate(
        LOCKED_LIGHTING_CONDITIONS
    ):
        loader, loader_summary = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            brightness=brightness,
            contrast=contrast,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"schedule_free_a0_{condition}",
            seed=int(args.seed) + 500 + condition_index,
        )
        role_rows = {
            "raw": _predict_fp32(model=prototype.to(device), loader=loader, device=device),
            "control": _predict_fp32(model=control_gpu, loader=loader, device=device),
            "candidate": _predict_fp32(model=candidate_gpu, loader=loader, device=device),
            "candidate_z": _predict_fp32(
                model=candidate_z_gpu, loader=loader, device=device
            ),
        }
        comparison = _checked_comparison(
            control_rows=role_rows["control"],
            candidate_rows=role_rows["candidate"],
            num_classes=5,
            focus_class=1,
        )
        comparison.update(
            {
                "condition": condition,
                "brightness": float(brightness),
                "contrast": float(contrast),
            }
        )
        illumination.append(comparison)
        illumination_tp_retention.append(
            _focus_tp_retention(
                role_rows["control"], role_rows["candidate"], focus_class=1
            )
        )
        condition_rows[condition] = role_rows
        illumination_loaders[condition] = loader_summary

    export_loader, export_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=calibration_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="schedule_free_a0_export",
        seed=int(args.seed) + 600,
    )
    export_images, _export_targets, export_metadata = next(iter(export_loader))
    export = _export_candidate(
        candidate=candidate_model,
        images=export_images,
        metadata=export_metadata,
        output_dir=output_dir,
    )

    runtime_ratio = float(
        candidate_result["median_step_seconds_excluding_warmup"]
        / max(float(control_result["median_step_seconds_excluding_warmup"]), 1e-12)
    )
    peak_vram_ratio = float(
        candidate_result["peak_vram_gib"]
        / max(float(control_result["peak_vram_gib"]), 1e-12)
    )
    config_summary = to_serializable(vars(args))
    dataset_summary: Dict[str, object] = {
        "train_rows": len(rows),
        "class_counts": list(EXPECTED_CLASS_COUNTS),
        "fit_rows": len(fit_indices),
        "holdout_rows": len(holdout_indices),
        "holdout_class_counts": list(holdout_counts),
        "train_budget_rows": len(train_indices),
        "calibration_rows": len(calibration_indices),
        "source_overlap": len(source_overlap),
        "holdout_order_sha256": _ordered_index_sha256(holdout_indices),
        "train_order_sha256": _ordered_index_sha256(train_indices),
        "calibration_order_sha256": _ordered_index_sha256(calibration_indices),
        "holdout_loader": holdout_loader_summary,
        "illumination_loaders": illumination_loaders,
        "export_loader": export_loader_summary,
        "train_transform": train_transform_summary,
        "cidt_validation_predictions_used": cidt_validation_predictions_used,
        "cidt_test_data_used": cidt_test_data_used,
        "validation_predictions_used": False,
        "test_data_used": False,
    }
    structural_checks = _build_structural_checks(
        equation=equation,
        raw_mismatches=raw_mismatches,
        control_result=control_result,
        candidate_result=candidate_result,
        control_bn=control_bn,
        candidate_bn=candidate_bn,
        candidate_z_bn=candidate_z_bn,
        x_z_difference=x_z_difference,
        runtime_ratio=runtime_ratio,
        peak_vram_ratio=peak_vram_ratio,
        export=export,
        source_overlap=len(source_overlap),
        prediction_rows_aligned=_condition_rows_aligned(condition_rows),
        provenance=provenance,
        config=config_summary,
        dataset=dataset_summary,
    )
    decision = assess_a0(
        structural_checks=structural_checks,
        clean=clean,
        clean_tp_retention=clean_tp_retention,
        illumination=illumination,
        illumination_tp_retention=illumination_tp_retention,
    )

    control_model = control_gpu.cpu().eval()
    candidate_model = candidate_gpu.cpu().eval()
    candidate_z_model = candidate_z_gpu.cpu().eval()
    prototype.cpu().eval()
    xai: Dict[str, object] = {
        "required": bool(decision["xai_review_required"]),
        "generated": False,
        "reason": (
            "automatic_gates_failed"
            if not bool(decision["automatic_gates_passed"])
            else "pending_generation"
        ),
    }
    if bool(decision["automatic_gates_passed"]):
        xai = _run_xai_review_artifacts(
            output_dir=output_dir,
            control_model=control_model,
            candidate_model=candidate_model,
            base_dataset=base_dataset,
            transform=transform,
            source_rows=rows,
            condition_rows=condition_rows,
            checkpoint=checkpoint,
            args=args,
            device=device,
        )
        xai["required"] = True
        xai["generated"] = True
        xai["reason"] = "automatic_gates_passed"

    predictions_path = output_dir / "predictions_all_conditions.csv"
    _write_prediction_rows(
        predictions_path, source_rows=rows, condition_rows=condition_rows
    )
    precalibration_path = output_dir / "precalibration_clean_predictions.csv"
    _write_prediction_rows(
        precalibration_path,
        source_rows=rows,
        condition_rows={
            "clean": {
                "control": precalibration["control"],
                "candidate": precalibration["candidate"],
            }
        },
    )
    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "control": control_result["history"],
                "candidate": candidate_result["history"],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary = {
        "protocol": "schedule_free_adamw_train_only_a0_v1",
        "status": (
            "awaiting_visual_review"
            if bool(decision["automatic_gates_passed"])
            else "rejected"
        ),
        "provenance": provenance,
        "config": config_summary,
        "dataset": dataset_summary,
        "model": {
            "parameter_count": parameter_count,
            "prototype_state_sha256": _state_sha256(prototype),
            "class_names": class_names,
        },
        "equation": equation,
        "raw_cidt_prediction_mismatches": raw_mismatches,
        "control": {
            key: value for key, value in control_result.items() if key != "history"
        },
        "candidate": {
            key: value for key, value in candidate_result.items() if key != "history"
        },
        "batchnorm": {
            "control": control_bn,
            "candidate_x": candidate_bn,
            "candidate_z": candidate_z_bn,
        },
        "precalibration": _checked_comparison(
            control_rows=precalibration["control"],
            candidate_rows=precalibration["candidate"],
            num_classes=5,
            focus_class=1,
        ),
        "clean": clean,
        "clean_tp_retention": clean_tp_retention,
        "candidate_x_vs_z": {
            "probability_difference": x_z_difference,
            "comparison": x_z_comparison,
        },
        "illumination": illumination,
        "illumination_tp_retention": illumination_tp_retention,
        "runtime": {
            "control_median_step_seconds": control_result[
                "median_step_seconds_excluding_warmup"
            ],
            "candidate_median_step_seconds": candidate_result[
                "median_step_seconds_excluding_warmup"
            ],
            "candidate_control_ratio": runtime_ratio,
            "control_peak_vram_gib": control_result["peak_vram_gib"],
            "candidate_peak_vram_gib": candidate_result["peak_vram_gib"],
            "peak_vram_ratio": peak_vram_ratio,
        },
        "export": export,
        "xai": xai,
        "decision": decision,
        "artifacts": {
            "predictions": predictions_path.name,
            "precalibration_predictions": precalibration_path.name,
            "training_history": history_path.name,
        },
        "artifact_hashes": {
            "predictions": _sha256(predictions_path),
            "precalibration_predictions": _sha256(precalibration_path),
            "training_history": _sha256(history_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    replay = replay_summary(summary_path)
    replay_path = output_dir / "replay.json"
    replay_path.write_text(
        json.dumps(replay, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    if not bool(replay["passed"]):
        raise ValueError(f"Independent replay failed: {replay}")
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    del control_gpu, candidate_gpu, candidate_z_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    _verify_sources(args)
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Formal summary is missing: {summary_path}")
    expected = str(args.expected_summary_sha256).strip().lower()
    if not expected:
        raise ValueError("Finalize requires --expected-summary-sha256.")
    observed = _sha256(summary_path)
    if observed != expected:
        raise ValueError(f"Pre-review summary SHA differs: {observed} != {expected}")
    note = str(args.visual_review_note).strip()
    if not note:
        raise ValueError("Finalize requires a non-empty visual review note.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("status")) != "awaiting_visual_review":
        raise ValueError(
            f"Summary is not awaiting visual review: {summary.get('status')}"
        )
    xai = summary.get("xai")
    decision = summary.get("decision")
    if (
        not isinstance(decision, Mapping)
        or not bool(decision.get("automatic_gates_passed", False))
    ):
        raise ValueError("Automatic Schedule-Free gates are not all passed.")
    if (
        not isinstance(xai, Mapping)
        or not bool(xai.get("required", False))
        or not bool(xai.get("generated", False))
        or not bool(xai.get("all_pages_passed", False))
    ):
        raise ValueError("XAI artifacts did not pass their automatic integrity checks.")
    pages = xai.get("pages")
    if not isinstance(pages, Mapping) or set(pages) != {
        "clean",
        "lighting_dim",
        "lighting_bright",
        "low_contrast",
    }:
        raise ValueError("Finalize requires all four locked XAI pages.")
    for value in pages.values():
        page_path = output_dir / str(value["path"])
        if not page_path.is_file() or _sha256(page_path) != str(value["sha256"]):
            raise ValueError(f"XAI page integrity differs: {page_path}")
    tensor_path = output_dir / str(xai.get("tensor_path", ""))
    if not tensor_path.is_file() or _sha256(tensor_path) != str(
        xai.get("tensor_sha256", "")
    ):
        raise ValueError(f"XAI tensor integrity differs: {tensor_path}")
    xai_manifest_path = output_dir / str(xai.get("manifest_path", ""))
    if not xai_manifest_path.is_file() or _sha256(xai_manifest_path) != str(
        xai.get("manifest_sha256", "")
    ):
        raise ValueError(f"XAI manifest integrity differs: {xai_manifest_path}")
    xai_manifest = json.loads(xai_manifest_path.read_text(encoding="utf-8"))
    manifest_projection = {
        "pages": xai.get("pages"),
        "tensor_path": xai.get("tensor_path"),
        "tensor_sha256": xai.get("tensor_sha256"),
        "all_pages_passed": xai.get("all_pages_passed"),
        "manual_review_required": xai.get("manual_review_required"),
    }
    if _json_max_abs_difference(xai_manifest, manifest_projection) != 0.0:
        raise ValueError("XAI manifest content differs from the formal summary.")
    review_passed = str(args.visual_review_result) == "pass"
    summary["visual_review"] = {
        "result": str(args.visual_review_result),
        "note": note,
        "pre_review_summary_sha256": observed,
        "pages_reviewed": sorted(pages),
    }
    summary["final_decision"] = {
        "automatic_gates_passed": True,
        "visual_review_passed": review_passed,
        "stage_b_smoke_authorized": review_passed,
        "full_train_authorized": False,
    }
    summary["status"] = "passed" if review_passed else "rejected"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    replay = replay_summary(summary_path)
    if not bool(replay["passed"]):
        raise ValueError(f"Post-review replay failed: {replay}")
    replay_path = output_dir / "replay.json"
    replay_path.write_text(
        json.dumps(replay, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.finalize_visual_review:
        if args.preflight_only or args.replay_summary is not None:
            raise ValueError(
                "--finalize-visual-review is exclusive with preflight and replay."
            )
        summary = _finalize_visual_review(args)
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
        return
    if args.replay_summary is not None:
        if args.preflight_only:
            raise ValueError("--replay-summary and --preflight-only are exclusive.")
        replay = replay_summary(args.replay_summary)
        print(json.dumps(replay, indent=2, ensure_ascii=True), flush=True)
        if not bool(replay["passed"]):
            raise SystemExit(1)
        return
    provenance = _verify_sources(args)
    equation = _equation_diagnostics(_source_paths(args))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "provenance": provenance,
                    "equation": equation,
                    "dataset_opened": False,
                    "model_loaded": False,
                    "output_created": False,
                    "validation_predictions_used": False,
                    "test_data_used": False,
                },
                indent=2,
                ensure_ascii=True,
            ),
            flush=True,
        )
        if not bool(equation["passed"]):
            raise SystemExit(1)
        return
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

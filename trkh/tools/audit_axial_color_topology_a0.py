from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import minimize
from skimage.color import rgb2hsv, rgb2lab
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import IdentityCorruption, LightingShift
from trkh.inference.inference import load_checkpoint
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
    _classification_metrics,
    directional_event_masks,
)
from trkh.tools.audit_lbp_surface_texture_readiness import (
    _apply_scaler,
    _effective_rank,
    _fit_scaler,
    apply_offset_residual,
    interior_roi_boxes,
)
from trkh.tools.audit_surface_blob_morphology_readiness import (
    achromatic_highlight_mask_from_rgb,
    diffuse_color_control_descriptor,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _sha256,
    _write_artifact_manifest,
)


SEED = 20260720
FOCUS_CLASS = 1
FOLDS = 5
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)

ROI_SIZE = 128
ERODE_RATIO = 0.05
ELLIPSE_RADIUS = 0.92
AXIAL_STRIP_HALF_WIDTH = 0.70
GAUSSIAN_SIGMA = 2.0
ROBUST_SCALE_FLOOR = 1e-3
CHANNEL_CLIP = 4.0
AXES_DEGREES = (0.0, 45.0, 90.0, 135.0)
CHANNEL_NAMES = ("lab_a", "lab_b", "chromaticity_r", "chromaticity_g", "saturation")
PROFILE_NAMES = (
    "outer_mean",
    "inner_mean",
    "center",
    "outer_asymmetry",
    "inner_asymmetry",
    "monotonic_coherence",
    "total_variation",
)
CONTROL_DIM = 25
AXIAL_DIM = 140
TOTAL_DIM = 165
READOUT_C = 0.3
MAX_ITERATIONS = 300
MIN_BIN_SUPPORT = 128
MIN_AXIAL_EFFECTIVE_RANK = 12.0

CONDITIONS = (
    ("clean", 1.0, 1.0),
    ("lighting_dim", 0.7, 0.9),
    ("lighting_bright", 1.25, 1.1),
    ("low_contrast", 1.0, 0.65),
)

EXPECTED_HASHES = {
    "protocol": "038faf959e2110a507b1aa4a41e21f1d6f70f9b02ffa769a84ecc1f4074758d1",
    "paper": "ef97ba28784e8864e6c8a61bd1b349fa9de5238da13a20c490d57da49acb1f4d",
    "data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "cidt_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "cidt_predictions": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "keeper": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "current_best": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_paths() -> Dict[str, Path]:
    root = _repo_root()
    return {
        "protocol": root
        / "docs"
        / "TRKH_5CLASS_AXIAL_COLOR_TOPOLOGY_A0_PROTOCOL_20260720.md",
        "paper": Path(
            "D:/DataAI/external_sources/papers/mango_spatial_variance_jsfa_2016.pdf"
        ),
        "data": Path("D:/DataAI/AIEx/newdataset/yolo_f/data.yaml"),
        "cidt_summary": root
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "summary.json",
        "cidt_predictions": root
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
        "keeper": root
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "checkpoints"
        / "best.pt",
        "current_best": root
        / "docs"
        / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "command_history": root
        / "docs"
        / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only axial color-topology A0. It rejects validation/test "
            "access and never writes a model checkpoint."
        )
    )
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--checkpoint", type=Path, default=defaults["keeper"])
    parser.add_argument("--cidt-summary", type=Path, default=defaults["cidt_summary"])
    parser.add_argument(
        "--cidt-predictions", type=Path, default=defaults["cidt_predictions"]
    )
    parser.add_argument(
        "--paper", type=Path, default=defaults["paper"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_axial_color_topology_a0_20260720"),
    )
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--descriptor-threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) != 96:
        raise ValueError("Locked A0 requires batch-size=96")
    if int(args.num_workers) != 4:
        raise ValueError("Locked A0 requires num-workers=4")
    if int(args.descriptor_threads) != 8:
        raise ValueError("Locked A0 requires descriptor-threads=8")
    if int(args.torch_threads) != 8:
        raise ValueError("Locked A0 requires torch-threads=8")
    output = Path(args.output_dir).resolve()
    data_root = Path(args.data).resolve().parent
    try:
        output.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must remain outside the raw dataset")


def verify_provenance(args: argparse.Namespace) -> Dict[str, object]:
    paths = {
        **_default_paths(),
        "paper": Path(args.paper),
        "data": Path(args.data),
        "cidt_summary": Path(args.cidt_summary),
        "cidt_predictions": Path(args.cidt_predictions),
        "keeper": Path(args.checkpoint),
    }
    result: Dict[str, object] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = Path(paths[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked provenance file: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"Locked provenance mismatch for {name}: {observed} != {expected}"
            )
        result[name] = {
            "path": str(path),
            "sha256": observed,
            "matched": True,
        }
    return result


def _coordinate_geometry(size: int = ROI_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = ((np.arange(int(size), dtype=np.float32) + 0.5) / float(size)) * 2.0 - 1.0
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    ellipse = np.square(x) + np.square(y) <= float(ELLIPSE_RADIUS) ** 2
    return x.astype(np.float32), y.astype(np.float32), ellipse


def _axial_bin_masks(
    x: np.ndarray,
    y: np.ndarray,
    ellipse: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = float(angle_degrees)
    diagonal = float(1.0 / math.sqrt(2.0))
    if angle == 0.0:
        t, u = x, y
    elif angle == 45.0:
        t, u = (x + y) * diagonal, (y - x) * diagonal
    elif angle == 90.0:
        t, u = y, -x
    elif angle == 135.0:
        t, u = (y - x) * diagonal, -(x + y) * diagonal
    else:
        raise ValueError(f"Unsupported locked axial angle: {angle}")
    geometry = ellipse & (np.abs(u) <= float(AXIAL_STRIP_HALF_WIDTH))
    values = t[geometry]
    if values.size < 5 * MIN_BIN_SUPPORT:
        raise ValueError("Axial strip geometry has insufficient support")

    unique_values, counts = np.unique(values, return_counts=True)
    if unique_values.size < 6:
        raise ValueError("Axial strip geometry has too few coordinate levels")
    cumulative = np.cumsum(counts, dtype=np.int64)[:-1]

    def split_at(fraction: float) -> float:
        target = float(values.size) * float(fraction)
        split = int(np.argmin(np.abs(cumulative.astype(np.float64) - target))) + 1
        return 0.5 * float(unique_values[split - 1] + unique_values[split])

    lower_outer = split_at(0.20)
    lower_inner = split_at(0.40)
    edges = np.asarray(
        (
            float(unique_values[0]),
            lower_outer,
            lower_inner,
            -lower_inner,
            -lower_outer,
            float(unique_values[-1]),
        ),
        dtype=np.float64,
    )
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("Axial coordinate edges are not strictly increasing")
    masks = []
    for index in range(5):
        lower = t >= edges[index]
        upper = t <= edges[index + 1] if index == 4 else t < edges[index + 1]
        masks.append(geometry & lower & upper)
    return np.stack(masks, axis=0), edges


def _ellipse_highlight_mask(
    values: np.ndarray,
    ellipse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _global_mask, score = achromatic_highlight_mask_from_rgb(values)
    hsv = rgb2hsv(np.asarray(values, dtype=np.float32).clip(0.0, 1.0))
    ellipse_scores = score[ellipse]
    ellipse_values = hsv[..., 2][ellipse]
    threshold = float(np.quantile(ellipse_scores, 0.95))
    median_value = float(np.median(ellipse_values))

    # Strict comparison makes percentile ties conservative. In particular, a
    # flat-color ROI must not be classified entirely as specular highlight.
    highlight = (
        ellipse
        & (score > threshold)
        & (hsv[..., 2] >= median_value)
    )
    return np.asarray(highlight, dtype=bool), np.asarray(score, dtype=np.float32)


def _robust_standardize_channel(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    selected = np.asarray(values, dtype=np.float32)[valid]
    if selected.size == 0:
        raise ValueError("Axial descriptor received an empty valid mask")
    center = float(np.median(selected))
    mad = float(np.median(np.abs(selected - center)))
    scale = max(1.4826 * mad, float(ROBUST_SCALE_FLOOR))
    standardized = np.clip((values - center) / scale, -CHANNEL_CLIP, CHANNEL_CLIP)
    return np.asarray(standardized, dtype=np.float32), float(scale)


def axial_color_topology_descriptor(
    rgb_uint8: np.ndarray,
    *,
    return_details: bool = False,
):
    rgb = np.asarray(rgb_uint8)
    if rgb.shape != (ROI_SIZE, ROI_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError(
            f"Axial descriptor requires uint8 [{ROI_SIZE},{ROI_SIZE},3], got {rgb.shape}/{rgb.dtype}"
        )
    values = rgb.astype(np.float32) / 255.0
    x, y, ellipse = _coordinate_geometry(ROI_SIZE)
    highlight, _highlight_score = _ellipse_highlight_mask(values, ellipse)
    valid = ellipse & ~highlight
    if int(valid.sum()) < 4096:
        raise ValueError("Axial descriptor valid support is unexpectedly small")

    lab = rgb2lab(values).astype(np.float32)
    hsv = rgb2hsv(values).astype(np.float32)
    channel_sum = values.sum(axis=2) + 1e-6
    channels = np.stack(
        (
            lab[..., 1] / 128.0,
            lab[..., 2] / 128.0,
            values[..., 0] / channel_sum,
            values[..., 1] / channel_sum,
            hsv[..., 1],
        ),
        axis=0,
    ).astype(np.float32)
    smoothed = np.stack(
        [
            ndimage.gaussian_filter(channel, sigma=GAUSSIAN_SIGMA, mode="reflect")
            for channel in channels
        ],
        axis=0,
    ).astype(np.float32)
    standardized = []
    robust_scales = []
    for channel in smoothed:
        normalized, scale = _robust_standardize_channel(channel, valid)
        standardized.append(normalized)
        robust_scales.append(scale)
    standardized_array = np.stack(standardized, axis=0)

    profiles = np.zeros((len(AXES_DEGREES), len(CHANNEL_NAMES), 5), dtype=np.float32)
    axis_features = np.zeros(
        (len(AXES_DEGREES), len(CHANNEL_NAMES), len(PROFILE_NAMES)),
        dtype=np.float32,
    )
    support_counts = np.zeros((len(AXES_DEGREES), 5), dtype=np.int64)
    bin_maps = []
    for axis_index, angle in enumerate(AXES_DEGREES):
        bins, _edges = _axial_bin_masks(x, y, ellipse, angle)
        bin_map = np.full((ROI_SIZE, ROI_SIZE), -1, dtype=np.int8)
        for bin_index, geometry_mask in enumerate(bins):
            selected = valid & geometry_mask
            support_counts[axis_index, bin_index] = int(selected.sum())
            if int(selected.sum()) < MIN_BIN_SUPPORT:
                raise ValueError("Axial bin support fell below the locked minimum")
            bin_map[selected] = int(bin_index)
            profiles[axis_index, :, bin_index] = standardized_array[:, selected].mean(
                axis=1
            )
        bin_maps.append(bin_map)
        q = profiles[axis_index]
        axis_features[axis_index, :, 0] = 0.5 * (q[:, 0] + q[:, 4])
        axis_features[axis_index, :, 1] = 0.5 * (q[:, 1] + q[:, 3])
        axis_features[axis_index, :, 2] = q[:, 2]
        axis_features[axis_index, :, 3] = np.abs(q[:, 4] - q[:, 0])
        axis_features[axis_index, :, 4] = np.abs(q[:, 3] - q[:, 1])
        axis_features[axis_index, :, 5] = 0.5 * np.abs(
            (q[:, 4] - q[:, 0]) + (q[:, 3] - q[:, 1])
        )
        axis_features[axis_index, :, 6] = np.abs(np.diff(q, axis=1)).mean(axis=1)

    sorted_axes = np.sort(axis_features.transpose(1, 2, 0), axis=2)
    axial = sorted_axes.reshape(-1).astype(np.float32)
    control = diffuse_color_control_descriptor(values, valid).astype(np.float32)
    if control.shape != (CONTROL_DIM,) or axial.shape != (AXIAL_DIM,):
        raise RuntimeError("Axial descriptor dimensions violate the locked protocol")
    if not np.isfinite(control).all() or not np.isfinite(axial).all():
        raise RuntimeError("Axial descriptor contains non-finite values")
    telemetry = {
        "valid_fraction": float(valid.mean()),
        "highlight_fraction": float((ellipse & highlight).sum() / max(1, ellipse.sum())),
        "minimum_bin_support": int(support_counts.min()),
        "minimum_robust_scale": float(min(robust_scales)),
    }
    if not return_details:
        return control, axial, telemetry
    dominant_axis = int(
        np.argmax(axis_features[:, :, 3].mean(axis=1) + axis_features[:, :, 4].mean(axis=1))
    )
    details = {
        "valid": valid.astype(np.uint8),
        "profiles": profiles,
        "dominant_axis": dominant_axis,
        "dominant_bin_map": bin_maps[dominant_axis],
        "standardized_channels": standardized_array,
    }
    return control, axial, telemetry, details


def assemble_role_features(
    control: np.ndarray,
    axial: np.ndarray,
    *,
    role: str,
    axial_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    control_values = np.asarray(control, dtype=np.float32)
    axial_values = np.asarray(axial, dtype=np.float32)
    if control_values.ndim != 2 or control_values.shape[1] != CONTROL_DIM:
        raise ValueError("Control feature matrix has the wrong shape")
    if axial_values.ndim != 2 or axial_values.shape[1] != AXIAL_DIM:
        raise ValueError("Axial feature matrix has the wrong shape")
    if role == "control":
        selected_axial = np.zeros((control_values.shape[0], AXIAL_DIM), dtype=np.float32)
    elif role == "candidate":
        if axial_values.shape[0] != control_values.shape[0]:
            raise ValueError("Candidate control/axial rows differ")
        selected_axial = axial_values
    elif role == "placebo":
        if axial_indices is None:
            raise ValueError("Placebo requires explicit axial indices")
        selected_axial = axial_values[np.asarray(axial_indices, dtype=np.int64)]
        if selected_axial.shape[0] != control_values.shape[0]:
            raise ValueError("Placebo permutation does not match control rows")
    else:
        raise ValueError(f"Unknown role: {role}")
    output = np.concatenate((control_values, selected_axial), axis=1).astype(np.float32)
    if output.shape != (control_values.shape[0], TOTAL_DIM):
        raise RuntimeError("Role feature dimension violates the locked protocol")
    return output


def source_derangement(
    indices: Sequence[int],
    source_stems: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    sources = np.asarray(source_stems, dtype=object)
    if rows.ndim != 1 or rows.size < 2:
        raise ValueError("Derangement requires at least two rows")
    original_sources = sources[rows]
    rng = np.random.default_rng(int(seed))
    for _attempt in range(1000):
        candidate = rows[rng.permutation(rows.size)]
        if np.all(sources[candidate] != original_sources):
            return candidate
    raise RuntimeError("Unable to construct a source-safe placebo derangement")


def fit_offset_model(
    features: np.ndarray,
    base_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float = READOUT_C,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(features, dtype=np.float64)
    base = np.asarray(base_probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or base.ndim != 2 or x.shape[0] != base.shape[0] or y.shape[0] != x.shape[0]:
        raise ValueError("Offset-model fit arrays are misaligned")
    if x.shape[1] != TOTAL_DIM or base.shape[1] != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Offset-model dimensions violate the locked protocol")
    base_logits = np.log(np.clip(base, 1e-7, 1.0))
    rows, feature_dim = x.shape
    class_count = base.shape[1]

    def objective(flat: np.ndarray):
        weights = flat.reshape(feature_dim, class_count)
        logits = base_logits + x @ weights
        maximum = logits.max(axis=1, keepdims=True)
        exponent = np.exp(logits - maximum)
        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        log_normalizer = np.log(exponent.sum(axis=1)) + maximum[:, 0]
        loss = float(
            np.mean(log_normalizer - logits[np.arange(rows), y])
            + 0.5 * np.square(weights).sum() / float(c_value)
        )
        gradient_logits = probabilities.copy()
        gradient_logits[np.arange(rows), y] -= 1.0
        gradient_logits /= float(rows)
        gradient = x.T @ gradient_logits + weights / float(c_value)
        return loss, gradient.reshape(-1)

    result = minimize(
        objective,
        np.zeros(feature_dim * class_count, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 50,
        },
    )
    weights = result.x.reshape(feature_dim, class_count)
    if not np.isfinite(weights).all():
        raise RuntimeError("Offset-model weights are non-finite")
    record = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "weight_l2": float(np.linalg.norm(weights)),
        "feature_dim": int(feature_dim),
        "parameter_count": int(weights.size),
    }
    return weights, record


def _read_cidt_conditions(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    expected_names = tuple(name for name, _brightness, _contrast in CONDITIONS)
    raw_by_condition: Dict[str, list[dict[str, str]]] = {
        name: [] for name in expected_names
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "fold",
            "target_index",
            *{f"keeper_prob_{index}" for index in range(5)},
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CIDT CSV is missing columns: {sorted(missing)}")
        for row in reader:
            condition = str(row["condition"]).strip()
            if condition not in raw_by_condition:
                raise ValueError(f"Unexpected CIDT condition: {condition}")
            raw_by_condition[condition].append(row)

    output: Dict[str, Dict[str, np.ndarray]] = {}
    reference = None
    for condition in expected_names:
        rows = raw_by_condition[condition]
        if len(rows) != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                f"CIDT condition {condition} has {len(rows)} rows, expected {EXPECTED_TRAIN_ROWS}"
            )
        sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        if not np.array_equal(sample_index, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)):
            raise ValueError(f"CIDT condition {condition} row order is not 0..N-1")
        target = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
        fold = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
        source = np.asarray([str(row["source_stem"]).casefold() for row in rows], dtype=object)
        image_path = np.asarray(
            [str(Path(row["image_path"]).resolve()) for row in rows], dtype=object
        )
        probabilities = np.asarray(
            [
                [float(row[f"keeper_prob_{index}"]) for index in range(5)]
                for row in rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-5
        ):
            raise ValueError(f"CIDT condition {condition} probabilities are invalid")
        for value in image_path.tolist():
            parts = {part.casefold() for part in Path(value).parts}
            if "train" not in parts or "val" in parts or "test" in parts:
                raise ValueError(f"Non-train path rejected from CIDT rows: {value}")
        metadata = (target, fold, source, image_path)
        if reference is None:
            reference = metadata
            if np.bincount(target, minlength=5).tolist() != list(EXPECTED_CLASS_COUNTS):
                raise ValueError("CIDT class counts differ from the locked protocol")
            if np.bincount(fold, minlength=FOLDS).tolist() != list(EXPECTED_FOLD_COUNTS):
                raise ValueError("CIDT fold counts differ from the locked protocol")
        else:
            for current, expected in zip(metadata, reference):
                if not np.array_equal(current, expected):
                    raise ValueError(f"CIDT metadata differs across condition {condition}")
        output[condition] = {
            "sample_index": sample_index,
            "target": target,
            "fold": fold,
            "source": source,
            "image_path": image_path,
            "probabilities": probabilities.astype(np.float32),
        }
    return output


def _dataset_metadata_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(value).resolve() for value in paths}, key=lambda value: str(value).casefold()):
        stat = path.stat()
        digest.update(str(path).replace("/", "\\").casefold().encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    requested = str(value)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _build_dataset(
    *,
    data_path: Path,
    checkpoint: Mapping[str, object],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[MangoYOLOCropDataset, object, list[str]]:
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != 5:
        raise ValueError("Keeper checkpoint class order is invalid")
    semantics = _eval_semantics(checkpoint)
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("Axial A0 requires temporal_frames=1")
    data_spec = load_data_spec(
        Path(data_path), class_name_mode="raw", expected_num_classes=5
    )
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from the keeper checkpoint")
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    paths = [str(Path(value).resolve()) for value in dataset.sample_paths()]
    clean_paths = [str(value) for value in condition_rows["clean"]["image_path"]]
    if paths != clean_paths:
        raise ValueError("Dataset sample paths do not align with CIDT sample order")
    if len(dataset) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Dataset row count violates the locked protocol")
    return dataset, _build_eval_transform(semantics), class_names


def _condition_corruption(name: str, brightness: float, contrast: float):
    if name == "clean":
        return IdentityCorruption()
    return LightingShift(brightness=float(brightness), contrast=float(contrast))


def _make_condition_loader(
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    condition: str,
    brightness: float,
    contrast: float,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Dict[str, object]]:
    dataset = _SelectedConditionDataset(
        base_dataset,
        indices,
        corruption=_condition_corruption(condition, brightness, contrast),
        transform=transform,
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=f"axial_color_topology_{condition}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            **kwargs,
        ),
        summary,
    )


def _extract_condition_descriptors(
    *,
    condition: str,
    brightness: float,
    contrast: float,
    base_dataset: MangoYOLOCropDataset,
    transform,
    expected: Mapping[str, np.ndarray],
    checkpoint: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    descriptor_threads: int,
) -> Dict[str, object]:
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        condition=condition,
        brightness=brightness,
        contrast=contrast,
        indices=list(range(EXPECTED_TRAIN_ROWS)),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    controls = []
    axials = []
    telemetry = []
    observed_indices = []
    observed_targets = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with ThreadPoolExecutor(max_workers=int(descriptor_threads)) as executor:
        for images, targets, metadata in tqdm(
            loader, desc=f"axial-{condition}", dynamic_ncols=True
        ):
            if not isinstance(metadata, Mapping):
                raise ValueError("Axial extraction requires metadata")
            crop_bbox = metadata.get("crop_bbox")
            sample_index = metadata.get("sample_index")
            if not torch.is_tensor(crop_bbox) or not torch.is_tensor(sample_index):
                raise ValueError("Axial extraction requires crop_bbox/sample_index tensors")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
            rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            boxes = interior_roi_boxes(
                crop_bbox,
                image_height=int(rgb.size(2)),
                image_width=int(rgb.size(3)),
                erode_ratio=ERODE_RATIO,
            )
            rois = roi_align(
                rgb,
                boxes,
                output_size=(ROI_SIZE, ROI_SIZE),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
            rgb_uint8 = (
                rois.mul(255.0)
                .round()
                .clamp(0.0, 255.0)
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            results = list(executor.map(axial_color_topology_descriptor, rgb_uint8))
            for control, axial, row_telemetry in results:
                controls.append(control)
                axials.append(axial)
                telemetry.append(row_telemetry)
            observed_indices.extend(sample_index.cpu().numpy().astype(np.int64).tolist())
            observed_targets.extend(targets.cpu().numpy().astype(np.int64).tolist())
    control_array = np.stack(controls, axis=0).astype(np.float32)
    axial_array = np.stack(axials, axis=0).astype(np.float32)
    if control_array.shape != (EXPECTED_TRAIN_ROWS, CONTROL_DIM):
        raise RuntimeError("Control descriptor extraction is incomplete")
    if axial_array.shape != (EXPECTED_TRAIN_ROWS, AXIAL_DIM):
        raise RuntimeError("Axial descriptor extraction is incomplete")
    if observed_indices != list(range(EXPECTED_TRAIN_ROWS)):
        raise ValueError("Descriptor loader sample order differs from CIDT")
    if not np.array_equal(
        np.asarray(observed_targets, dtype=np.int64), expected["target"]
    ):
        raise ValueError("Descriptor loader targets differ from CIDT")
    telemetry_summary = {
        key: {
            "minimum": float(min(float(row[key]) for row in telemetry)),
            "mean": float(np.mean([float(row[key]) for row in telemetry])),
            "maximum": float(max(float(row[key]) for row in telemetry)),
        }
        for key in telemetry[0]
    }
    return {
        "control": control_array,
        "axial": axial_array,
        "telemetry": telemetry_summary,
        "loader": loader_summary,
        "seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }


def _transition_summary(
    targets: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, int]:
    control_prediction = np.asarray(control).argmax(axis=1)
    candidate_prediction = np.asarray(candidate).argmax(axis=1)
    masks = directional_event_masks(
        targets,
        control_prediction,
        candidate_prediction,
        focus_class=FOCUS_CLASS,
    )
    changed = control_prediction != candidate_prediction
    control_correct = control_prediction == targets
    candidate_correct = candidate_prediction == targets
    result = {
        "changed": int(changed.sum()),
        "corrections": int((~control_correct & candidate_correct).sum()),
        "harms": int((control_correct & ~candidate_correct).sum()),
        "neutral": int((changed & (control_correct == candidate_correct)).sum()),
    }
    result.update({key: int(mask.sum()) for key, mask in masks.items()})
    return result


def _restricted_fp_summary(
    targets: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, int]:
    restricted = np.isin(targets, np.asarray(RESTRICTED_NEGATIVE_CLASSES))
    control_fp = restricted & (np.asarray(control).argmax(axis=1) == FOCUS_CLASS)
    candidate_fp = restricted & (np.asarray(candidate).argmax(axis=1) == FOCUS_CLASS)
    return {
        "control": int(control_fp.sum()),
        "candidate": int(candidate_fp.sum()),
        "removed": int((control_fp & ~candidate_fp).sum()),
        "created": int((~control_fp & candidate_fp).sum()),
        "net_reduction": int(control_fp.sum() - candidate_fp.sum()),
    }


def _direction_auc(
    targets: np.ndarray,
    keeper_probabilities: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    keeper_prediction = np.asarray(keeper_probabilities).argmax(axis=1)
    positive = (targets == FOCUS_CLASS) & (keeper_prediction != FOCUS_CLASS)
    negative = np.isin(targets, RESTRICTED_NEGATIVE_CLASSES) & (
        keeper_prediction == FOCUS_CLASS
    )
    selected = positive | negative
    labels = positive[selected].astype(np.int64)
    scores = (
        np.asarray(candidate, dtype=np.float64)[:, FOCUS_CLASS]
        - np.asarray(control, dtype=np.float64)[:, FOCUS_CLASS]
    )[selected]
    auc = (
        float(roc_auc_score(labels, scores))
        if labels.size and np.unique(labels).size == 2
        else None
    )
    return {
        "auc": auc,
        "positive_keeper_fn": int(positive.sum()),
        "negative_restricted_keeper_fp": int(negative.sum()),
        "rows": int(selected.sum()),
        "mean_positive_score": float(scores[labels == 1].mean()) if np.any(labels == 1) else None,
        "mean_negative_score": float(scores[labels == 0].mean()) if np.any(labels == 0) else None,
    }


def _metric_bundle(
    *,
    targets: np.ndarray,
    keeper: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    placebo: np.ndarray,
) -> Dict[str, object]:
    metrics = {
        "keeper": _classification_metrics(targets, keeper.argmax(axis=1), num_classes=5),
        "control": _classification_metrics(targets, control.argmax(axis=1), num_classes=5),
        "candidate": _classification_metrics(targets, candidate.argmax(axis=1), num_classes=5),
        "placebo": _classification_metrics(targets, placebo.argmax(axis=1), num_classes=5),
    }
    candidate_metrics = metrics["candidate"]
    control_metrics = metrics["control"]
    placebo_metrics = metrics["placebo"]
    delta = {
        "macro_f1": float(candidate_metrics["macro_f1"] - control_metrics["macro_f1"]),
        "class1_precision": float(
            candidate_metrics["per_class_precision"][FOCUS_CLASS]
            - control_metrics["per_class_precision"][FOCUS_CLASS]
        ),
        "class1_recall": float(
            candidate_metrics["per_class_recall"][FOCUS_CLASS]
            - control_metrics["per_class_recall"][FOCUS_CLASS]
        ),
        "class1_f1": float(
            candidate_metrics["per_class_f1"][FOCUS_CLASS]
            - control_metrics["per_class_f1"][FOCUS_CLASS]
        ),
        "candidate_vs_placebo_macro_f1": float(
            candidate_metrics["macro_f1"] - placebo_metrics["macro_f1"]
        ),
        "candidate_vs_placebo_class1_f1": float(
            candidate_metrics["per_class_f1"][FOCUS_CLASS]
            - placebo_metrics["per_class_f1"][FOCUS_CLASS]
        ),
        "maximum_nonfocus_f1_drop": float(
            max(
                control_metrics["per_class_f1"][index]
                - candidate_metrics["per_class_f1"][index]
                for index in range(5)
                if index != FOCUS_CLASS
            )
        ),
    }
    return {
        "metrics": metrics,
        "delta": delta,
        "transitions": _transition_summary(targets, control, candidate),
        "restricted_fp": _restricted_fp_summary(targets, control, candidate),
        "direction": _direction_auc(targets, keeper, control, candidate),
    }


def assess_axial_color_topology_a0(
    *,
    structural_checks: Mapping[str, bool],
    condition_results: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    clean = condition_results["clean"]
    clean_delta = clean["delta"]
    clean_transitions = clean["transitions"]
    clean_restricted = clean["restricted_fp"]
    clean_direction = clean["direction"]
    clean_checks = {
        "clean_macro_f1_delta_gte_0p002": float(clean_delta["macro_f1"]) >= 0.002,
        "clean_class1_f1_delta_gte_0p005": float(clean_delta["class1_f1"]) >= 0.005,
        "clean_class1_precision_delta_gte_0p010": float(
            clean_delta["class1_precision"]
        )
        >= 0.010,
        "clean_class1_recall_delta_gte_minus_0p005": float(
            clean_delta["class1_recall"]
        )
        >= -0.005,
        "clean_restricted_fp_reduction_gte_4": int(
            clean_restricted["net_reduction"]
        )
        >= 4,
        "clean_corrections_gt_harms": int(clean_transitions["corrections"])
        > int(clean_transitions["harms"]),
        "clean_fn_rescues_gte_tp_breaks": int(
            clean_transitions["focus_fn_rescue"]
        )
        >= int(clean_transitions["focus_tp_break"]),
        "clean_nonfocus_f1_drop_lte_0p010": float(
            clean_delta["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "clean_candidate_vs_placebo_macro_gte_0p002": float(
            clean_delta["candidate_vs_placebo_macro_f1"]
        )
        >= 0.002,
        "clean_candidate_vs_placebo_class1_gte_0p005": float(
            clean_delta["candidate_vs_placebo_class1_f1"]
        )
        >= 0.005,
        "clean_direction_auc_gte_0p60": clean_direction["auc"] is not None
        and float(clean_direction["auc"]) >= 0.60,
    }
    shifted = [
        condition_results[name]
        for name, _brightness, _contrast in CONDITIONS
        if name != "clean"
    ]
    shifted_precision = [float(row["delta"]["class1_precision"]) for row in shifted]
    shifted_f1 = [float(row["delta"]["class1_f1"]) for row in shifted]
    shifted_recall = [float(row["delta"]["class1_recall"]) for row in shifted]
    shifted_macro = [float(row["delta"]["macro_f1"]) for row in shifted]
    shifted_auc = [
        float(row["direction"]["auc"])
        for row in shifted
        if row["direction"]["auc"] is not None
    ]
    robustness_checks = {
        "shifted_precision_nonnegative_all": all(value >= 0.0 for value in shifted_precision),
        "worst_shifted_class1_f1_delta_gte_minus_0p005": min(shifted_f1) >= -0.005,
        "worst_shifted_class1_recall_delta_gte_minus_0p015": min(shifted_recall)
        >= -0.015,
        "shifted_restricted_fp_nonnegative_each": all(
            int(row["restricted_fp"]["net_reduction"]) >= 0 for row in shifted
        ),
        "shifted_restricted_fp_aggregate_reduction_gte_6": sum(
            int(row["restricted_fp"]["net_reduction"]) for row in shifted
        )
        >= 6,
        "shifted_fn_rescues_gte_tp_breaks_each": all(
            int(row["transitions"]["focus_fn_rescue"])
            >= int(row["transitions"]["focus_tp_break"])
            for row in shifted
        ),
        "maximum_shifted_macro_drop_lte_0p005": max(-value for value in shifted_macro)
        <= 0.005,
        "median_shifted_direction_auc_gte_0p58": len(shifted_auc) == 3
        and float(np.median(shifted_auc)) >= 0.58,
    }
    checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **clean_checks,
        **robustness_checks,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "robustness_checks": robustness_checks,
        "checks": checks,
        "failed_checks": failed,
        "automatic_gates_passed": not failed,
        "visual_review_required": not failed,
        "stage_b_smoke_authorized": False,
        "full_train_authorized": False,
    }


def _fit_oof_readouts(
    *,
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Dict[str, Dict[str, np.ndarray]], list[Dict[str, object]], Dict[str, object]]:
    roles = ("control", "candidate", "placebo")
    outputs = {
        condition: {
            role: np.full((EXPECTED_TRAIN_ROWS, 5), np.nan, dtype=np.float32)
            for role in roles
        }
        for condition, _brightness, _contrast in CONDITIONS
    }
    clean_rows = condition_rows["clean"]
    targets = clean_rows["target"]
    folds = clean_rows["fold"]
    sources = clean_rows["source"]
    fit_records: list[Dict[str, object]] = []
    derangement_records = []
    for fold in range(FOLDS):
        fit_indices = np.flatnonzero(folds != fold)
        holdout_indices = np.flatnonzero(folds == fold)
        fit_sources = set(sources[fit_indices].tolist())
        holdout_sources = set(sources[holdout_indices].tolist())
        overlap = fit_sources.intersection(holdout_sources)
        if overlap:
            raise ValueError(f"CIDT fold {fold} has source overlap")
        fit_permutation = source_derangement(
            fit_indices, sources, seed=SEED + fold * 2
        )
        holdout_permutation = source_derangement(
            holdout_indices, sources, seed=SEED + fold * 2 + 1
        )
        derangement_records.append(
            {
                "fold": fold,
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "source_overlap": len(overlap),
                "fit_same_source": int(
                    np.sum(sources[fit_indices] == sources[fit_permutation])
                ),
                "holdout_same_source": int(
                    np.sum(sources[holdout_indices] == sources[holdout_permutation])
                ),
                "fit_permutation_sha256": hashlib.sha256(
                    fit_permutation.astype("<i8").tobytes()
                ).hexdigest(),
                "holdout_permutation_sha256": hashlib.sha256(
                    holdout_permutation.astype("<i8").tobytes()
                ).hexdigest(),
            }
        )
        for role in roles:
            if role == "placebo":
                fit_features = assemble_role_features(
                    descriptors["clean"]["control"][fit_indices],
                    descriptors["clean"]["axial"],
                    role=role,
                    axial_indices=fit_permutation,
                )
            else:
                fit_features = assemble_role_features(
                    descriptors["clean"]["control"][fit_indices],
                    descriptors["clean"]["axial"][fit_indices],
                    role=role,
                )
            mean, scale = _fit_scaler(fit_features)
            fit_scaled = _apply_scaler(fit_features, mean, scale)
            weights, record = fit_offset_model(
                fit_scaled,
                condition_rows["clean"]["probabilities"][fit_indices],
                targets[fit_indices],
            )
            record.update(
                {
                    "fold": fold,
                    "role": role,
                    "fit_rows": int(fit_indices.size),
                    "holdout_rows": int(holdout_indices.size),
                    "scaler_mean_sha256": hashlib.sha256(
                        np.asarray(mean, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "scaler_scale_sha256": hashlib.sha256(
                        np.asarray(scale, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "weights_sha256": hashlib.sha256(
                        np.asarray(weights, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
            fit_records.append(record)
            for condition, _brightness, _contrast in CONDITIONS:
                if role == "placebo":
                    holdout_features = assemble_role_features(
                        descriptors[condition]["control"][holdout_indices],
                        descriptors[condition]["axial"],
                        role=role,
                        axial_indices=holdout_permutation,
                    )
                else:
                    holdout_features = assemble_role_features(
                        descriptors[condition]["control"][holdout_indices],
                        descriptors[condition]["axial"][holdout_indices],
                        role=role,
                    )
                holdout_scaled = _apply_scaler(holdout_features, mean, scale)
                outputs[condition][role][holdout_indices] = apply_offset_residual(
                    condition_rows[condition]["probabilities"][holdout_indices],
                    holdout_scaled,
                    weights,
                    np.zeros(5, dtype=np.float64),
                )
    if any(
        not np.isfinite(outputs[condition][role]).all()
        for condition, _brightness, _contrast in CONDITIONS
        for role in roles
    ):
        raise RuntimeError("OOF readout probabilities are incomplete or non-finite")
    diagnostics = {
        "derangements": derangement_records,
        "all_source_deranged": all(
            row["fit_same_source"] == 0 and row["holdout_same_source"] == 0
            for row in derangement_records
        ),
        "maximum_source_overlap": max(
            int(row["source_overlap"]) for row in derangement_records
        ),
        "all_converged": all(
            bool(row["success"]) and int(row["iterations"]) < MAX_ITERATIONS
            for row in fit_records
        ),
    }
    return outputs, fit_records, diagnostics


def _condition_results(
    *,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    result = {}
    for condition, brightness, contrast in CONDITIONS:
        rows = condition_rows[condition]
        result[condition] = {
            "brightness": float(brightness),
            "contrast": float(contrast),
            **_metric_bundle(
                targets=rows["target"],
                keeper=rows["probabilities"],
                control=outputs[condition]["control"],
                candidate=outputs[condition]["candidate"],
                placebo=outputs[condition]["placebo"],
            ),
        }
    return result


def _write_predictions(
    path: Path,
    *,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for role in ("keeper", "control", "candidate", "placebo"):
        fields.append(f"{role}_prediction")
        fields.extend(f"{role}_prob_{index}" for index in range(5))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, _brightness, _contrast in CONDITIONS:
            rows = condition_rows[condition]
            role_probabilities = {
                "keeper": rows["probabilities"],
                **outputs[condition],
            }
            for index in range(EXPECTED_TRAIN_ROWS):
                output = {
                    "condition": condition,
                    "sample_index": index,
                    "source_stem": rows["source"][index],
                    "image_path": rows["image_path"][index],
                    "fold": int(rows["fold"][index]),
                    "target": int(rows["target"][index]),
                }
                for role, probabilities in role_probabilities.items():
                    output[f"{role}_prediction"] = int(probabilities[index].argmax())
                    for class_index in range(5):
                        output[f"{role}_prob_{class_index}"] = repr(
                            float(probabilities[index, class_index])
                        )
                writer.writerow(output)


def _write_fold_metrics(
    path: Path,
    *,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    fields = [
        "condition",
        "fold",
        "role",
        "rows",
        "macro_f1",
        "class1_precision",
        "class1_recall",
        "class1_f1",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, _brightness, _contrast in CONDITIONS:
            rows = condition_rows[condition]
            for fold in range(FOLDS):
                selected = rows["fold"] == fold
                for role in ("keeper", "control", "candidate", "placebo"):
                    probabilities = (
                        rows["probabilities"] if role == "keeper" else outputs[condition][role]
                    )
                    metrics = _classification_metrics(
                        rows["target"][selected],
                        probabilities[selected].argmax(axis=1),
                        num_classes=5,
                    )
                    writer.writerow(
                        {
                            "condition": condition,
                            "fold": fold,
                            "role": role,
                            "rows": int(selected.sum()),
                            "macro_f1": repr(float(metrics["macro_f1"])),
                            "class1_precision": repr(
                                float(metrics["per_class_precision"][FOCUS_CLASS])
                            ),
                            "class1_recall": repr(
                                float(metrics["per_class_recall"][FOCUS_CLASS])
                            ),
                            "class1_f1": repr(
                                float(metrics["per_class_f1"][FOCUS_CLASS])
                            ),
                        }
                    )


def _select_visual_indices(bundle: Mapping[str, object], limit_per_kind: int = 3) -> list[int]:
    metrics = bundle["arrays"]
    targets = np.asarray(metrics["target"], dtype=np.int64)
    control = np.asarray(metrics["control"], dtype=np.float32).argmax(axis=1)
    candidate = np.asarray(metrics["candidate"], dtype=np.float32).argmax(axis=1)
    kinds = (
        ("fp_remove", (targets != FOCUS_CLASS) & (control == FOCUS_CLASS) & (candidate == targets)),
        ("fn_rescue", (targets == FOCUS_CLASS) & (control != FOCUS_CLASS) & (candidate == FOCUS_CLASS)),
        ("tp_break", (targets == FOCUS_CLASS) & (control == FOCUS_CLASS) & (candidate != FOCUS_CLASS)),
        ("fp_create", (targets != FOCUS_CLASS) & (control == targets) & (candidate == FOCUS_CLASS)),
    )
    selected = []
    for _name, mask in kinds:
        selected.extend(np.flatnonzero(mask)[: int(limit_per_kind)].tolist())
    if not selected:
        changed = np.flatnonzero(control != candidate)
        selected.extend(changed[: max(1, int(limit_per_kind))].tolist())
    return sorted(set(int(value) for value in selected))


def _profile_panel(details: Mapping[str, object], size: int = 180) -> Image.Image:
    panel = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((24, 14, size - 10, size - 24), outline="black", width=1)
    profiles = np.asarray(details["profiles"], dtype=np.float32)[
        int(details["dominant_axis"])
    ]
    colors = ("#d62728", "#ffbf00", "#2ca02c", "#1f77b4", "#9467bd")
    maximum = max(1.0, float(np.max(np.abs(profiles))))
    for channel_index, values in enumerate(profiles):
        points = []
        for index, value in enumerate(values):
            x = 30 + index * (size - 48) / 4.0
            y = (size - 30) - ((float(value) / maximum + 1.0) * 0.5) * (size - 52)
            points.append((x, y))
        draw.line(points, fill=colors[channel_index], width=2)
    draw.text((4, 2), f"axis={AXES_DEGREES[int(details['dominant_axis'])]:.0f}", fill="black")
    return panel


def _render_contact_sheet(
    path: Path,
    *,
    condition: str,
    indices: Sequence[int],
    rois: Mapping[int, np.ndarray],
    targets: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    tile = 180
    header = 34
    rows = max(1, len(indices))
    canvas = Image.new("RGB", (tile * 4, header + rows * (tile + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{condition}: RGB | valid | axial bins | profiles", fill="black")
    palette = np.asarray(
        [[49, 130, 189], [107, 174, 214], [189, 189, 189], [253, 174, 107], [215, 48, 39]],
        dtype=np.uint8,
    )
    selected_rows = []
    for row_index, sample_index in enumerate(indices):
        rgb = np.asarray(rois[int(sample_index)], dtype=np.uint8)
        _control, _axial, telemetry, details = axial_color_topology_descriptor(
            rgb, return_details=True
        )
        y0 = header + row_index * (tile + 24)
        rgb_image = Image.fromarray(rgb).resize((tile, tile), Image.Resampling.BILINEAR)
        valid_image = Image.fromarray(
            np.asarray(details["valid"], dtype=np.uint8) * 255
        ).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
        bin_map = np.asarray(details["dominant_bin_map"], dtype=np.int64)
        bin_rgb = np.full((ROI_SIZE, ROI_SIZE, 3), 255, dtype=np.uint8)
        active = bin_map >= 0
        bin_rgb[active] = palette[bin_map[active]]
        bin_image = Image.fromarray(bin_rgb).resize((tile, tile), Image.Resampling.NEAREST)
        panels = (rgb_image, valid_image, bin_image, _profile_panel(details, tile))
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * tile, y0))
        control_prediction = int(control[sample_index].argmax())
        candidate_prediction = int(candidate[sample_index].argmax())
        label = (
            f"sample={sample_index} y={int(targets[sample_index])} "
            f"control={control_prediction} candidate={candidate_prediction} "
            f"support={telemetry['minimum_bin_support']}"
        )
        draw.text((8, y0 + tile + 4), label, fill="black")
        selected_rows.append(
            {
                "sample_index": int(sample_index),
                "target": int(targets[sample_index]),
                "control_prediction": control_prediction,
                "candidate_prediction": candidate_prediction,
                "dominant_axis": int(details["dominant_axis"]),
                "minimum_bin_support": int(telemetry["minimum_bin_support"]),
            }
        )
    canvas.save(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "rows": selected_rows,
        "passed": len(selected_rows) == len(indices) and len(indices) > 0,
    }


def _extract_selected_rois(
    *,
    condition: str,
    brightness: float,
    contrast: float,
    indices: Sequence[int],
    base_dataset: MangoYOLOCropDataset,
    transform,
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    loader, _summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        condition=condition,
        brightness=brightness,
        contrast=contrast,
        indices=indices,
        batch_size=max(1, len(indices)),
        num_workers=0,
    )
    images, _targets, metadata = next(iter(loader))
    mean, std = checkpoint_input_normalization(checkpoint)
    images = images.to(device=device, dtype=torch.float32)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
    crop_bbox = metadata["crop_bbox"].to(device=device, dtype=torch.float32)
    boxes = interior_roi_boxes(
        crop_bbox,
        image_height=int(rgb.size(2)),
        image_width=int(rgb.size(3)),
        erode_ratio=ERODE_RATIO,
    )
    rois = roi_align(
        rgb,
        boxes,
        output_size=(ROI_SIZE, ROI_SIZE),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    arrays = (
        rois.mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    observed = metadata["sample_index"].cpu().numpy().astype(np.int64).tolist()
    if observed != [int(value) for value in indices]:
        raise ValueError("Selected ROI order differs from requested indices")
    return {int(index): array for index, array in zip(observed, arrays)}


def _generate_contact_sheets(
    *,
    output_dir: Path,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
    base_dataset: MangoYOLOCropDataset,
    transform,
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    pages = {}
    for condition, brightness, contrast in CONDITIONS:
        rows = condition_rows[condition]
        selection_bundle = {
            "arrays": {
                "target": rows["target"],
                "control": outputs[condition]["control"],
                "candidate": outputs[condition]["candidate"],
            }
        }
        indices = _select_visual_indices(selection_bundle)
        rois = _extract_selected_rois(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            indices=indices,
            base_dataset=base_dataset,
            transform=transform,
            checkpoint=checkpoint,
            device=device,
        )
        page_path = output_dir / f"contact_{condition}.png"
        pages[condition] = _render_contact_sheet(
            page_path,
            condition=condition,
            indices=indices,
            rois=rois,
            targets=rows["target"],
            control=outputs[condition]["control"],
            candidate=outputs[condition]["candidate"],
        )
    return pages


def _descriptor_summary(
    descriptors: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    clean_axial = np.asarray(descriptors["clean"]["axial"], dtype=np.float32)
    return {
        "control_dim": CONTROL_DIM,
        "axial_dim": AXIAL_DIM,
        "role_dim": TOTAL_DIM,
        "axial_effective_rank": float(_effective_rank(clean_axial)),
        "all_finite": all(
            np.isfinite(descriptors[name][key]).all()
            for name, _brightness, _contrast in CONDITIONS
            for key in ("control", "axial")
        ),
        "minimum_bin_support": int(
            min(
                descriptors[name]["telemetry"]["minimum_bin_support"]["minimum"]
                for name, _brightness, _contrast in CONDITIONS
            )
        ),
        "conditions": {
            name: {
                "telemetry": descriptors[name]["telemetry"],
                "seconds": descriptors[name]["seconds"],
                "peak_cuda_memory_mib": descriptors[name]["peak_cuda_memory_mib"],
                "loader": descriptors[name]["loader"],
            }
            for name, _brightness, _contrast in CONDITIONS
        },
    }


def _structural_checks(
    *,
    provenance: Mapping[str, object],
    descriptor_summary: Mapping[str, object],
    readout_diagnostics: Mapping[str, object],
    fit_records: Sequence[Mapping[str, object]],
    raw_metadata_before: str,
    raw_metadata_after: str,
) -> Dict[str, bool]:
    return {
        "provenance_hashes_exact": all(
            bool(value["matched"]) for value in provenance.values()
        ),
        "train_rows_exact": True,
        "class_counts_exact": True,
        "source_folds_exact": True,
        "source_overlap_zero": int(readout_diagnostics["maximum_source_overlap"]) == 0,
        "four_conditions_exact": True,
        "descriptor_dimensions_exact": (
            int(descriptor_summary["control_dim"]) == CONTROL_DIM
            and int(descriptor_summary["axial_dim"]) == AXIAL_DIM
            and int(descriptor_summary["role_dim"]) == TOTAL_DIM
        ),
        "descriptors_finite": bool(descriptor_summary["all_finite"]),
        "minimum_bin_support_gte_128": int(descriptor_summary["minimum_bin_support"])
        >= MIN_BIN_SUPPORT,
        "axial_effective_rank_gte_12": float(
            descriptor_summary["axial_effective_rank"]
        )
        >= MIN_AXIAL_EFFECTIVE_RANK,
        "all_15_readouts_converged": len(fit_records) == 15
        and bool(readout_diagnostics["all_converged"]),
        "placebo_source_deranged": bool(readout_diagnostics["all_source_deranged"]),
        "raw_dataset_metadata_unchanged": raw_metadata_before == raw_metadata_after,
        "validation_data_unused": True,
        "test_data_unused": True,
        "model_or_checkpoint_written_false": True,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    provenance = verify_provenance(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Formal output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.torch_threads))
    set_seed(SEED, deterministic=True)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cidt_summary = json.loads(Path(args.cidt_summary).read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use")
    condition_rows = _read_cidt_conditions(Path(args.cidt_predictions))
    checkpoint = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    base_dataset, transform, class_names = _build_dataset(
        data_path=Path(args.data),
        checkpoint=checkpoint,
        condition_rows=condition_rows,
    )
    dataset_paths = [Path(value) for value in base_dataset.sample_paths()]
    raw_metadata_before = _dataset_metadata_sha256(dataset_paths)

    descriptors: Dict[str, Dict[str, object]] = {}
    for condition, brightness, contrast in CONDITIONS:
        descriptors[condition] = _extract_condition_descriptors(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            base_dataset=base_dataset,
            transform=transform,
            expected=condition_rows[condition],
            checkpoint=checkpoint,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            descriptor_threads=int(args.descriptor_threads),
        )
    descriptor_summary = _descriptor_summary(descriptors)
    outputs, fit_records, readout_diagnostics = _fit_oof_readouts(
        descriptors=descriptors,
        condition_rows=condition_rows,
    )
    results = _condition_results(outputs=outputs, condition_rows=condition_rows)
    raw_metadata_after = _dataset_metadata_sha256(dataset_paths)
    structural = _structural_checks(
        provenance=provenance,
        descriptor_summary=descriptor_summary,
        readout_diagnostics=readout_diagnostics,
        fit_records=fit_records,
        raw_metadata_before=raw_metadata_before,
        raw_metadata_after=raw_metadata_after,
    )
    gate = assess_axial_color_topology_a0(
        structural_checks=structural,
        condition_results=results,
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    fold_metrics_path = output_dir / "fold_metrics.csv"
    descriptor_path = output_dir / "descriptor_diagnostics.json"
    readout_path = output_dir / "readout_diagnostics.json"
    descriptor_cache_path = output_dir / "descriptor_cache.npz"
    _write_predictions(
        predictions_path, outputs=outputs, condition_rows=condition_rows
    )
    _write_fold_metrics(
        fold_metrics_path, outputs=outputs, condition_rows=condition_rows
    )
    descriptor_path.write_text(
        json.dumps(descriptor_summary, indent=2), encoding="utf-8"
    )
    readout_path.write_text(
        json.dumps(
            {
                **readout_diagnostics,
                "fit_records": fit_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        descriptor_cache_path,
        conditions=np.asarray([name for name, _brightness, _contrast in CONDITIONS]),
        control=np.stack(
            [descriptors[name]["control"] for name, _brightness, _contrast in CONDITIONS]
        ),
        axial=np.stack(
            [descriptors[name]["axial"] for name, _brightness, _contrast in CONDITIONS]
        ),
        target=condition_rows["clean"]["target"],
        fold=condition_rows["clean"]["fold"],
        sample_index=condition_rows["clean"]["sample_index"],
    )

    contact_sheets = {}
    if bool(gate["automatic_gates_passed"]):
        contact_sheets = _generate_contact_sheets(
            output_dir=output_dir,
            outputs=outputs,
            condition_rows=condition_rows,
            base_dataset=base_dataset,
            transform=transform,
            checkpoint=checkpoint,
            device=device,
        )
    status = (
        "awaiting_visual_review"
        if gate["automatic_gates_passed"]
        else "rejected_before_visual_review"
    )
    summary = {
        "mode": "axial_color_topology_train_only_a0",
        "status": status,
        "provenance": provenance,
        "protocol": {
            "seed": SEED,
            "folds": FOLDS,
            "roi_size": ROI_SIZE,
            "erode_ratio": ERODE_RATIO,
            "ellipse_radius": ELLIPSE_RADIUS,
            "axial_strip_half_width": AXIAL_STRIP_HALF_WIDTH,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "axes_degrees": list(AXES_DEGREES),
            "channel_names": list(CHANNEL_NAMES),
            "profile_names": list(PROFILE_NAMES),
            "control_dim": CONTROL_DIM,
            "axial_dim": AXIAL_DIM,
            "role_dim": TOTAL_DIM,
            "readout_c": READOUT_C,
            "max_iterations": MAX_ITERATIONS,
            "conditions": [
                {"name": name, "brightness": brightness, "contrast": contrast}
                for name, brightness, contrast in CONDITIONS
            ],
        },
        "dataset": {
            "split": "train",
            "rows": EXPECTED_TRAIN_ROWS,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "source_groups": int(
                len(set(condition_rows["clean"]["source"].tolist()))
            ),
            "class_names": class_names,
            "raw_metadata_sha256_before": raw_metadata_before,
            "raw_metadata_sha256_after": raw_metadata_after,
            "validation_data_used": False,
            "test_data_used": False,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "descriptor_threads": int(args.descriptor_threads),
        },
        "descriptor": descriptor_summary,
        "readout": {
            **readout_diagnostics,
            "fit_records": fit_records,
        },
        "results": results,
        "gate": gate,
        "contact_sheets": contact_sheets,
        "visual_review": None,
        "artifacts": {
            "predictions": predictions_path.name,
            "fold_metrics": fold_metrics_path.name,
            "descriptor_diagnostics": descriptor_path.name,
            "readout_diagnostics": readout_path.name,
            "descriptor_cache": descriptor_cache_path.name,
            "artifact_manifest": "artifact_manifest.json",
        },
        "guardrails": {
            "raw_dataset_touched": False,
            "validation_data_used": False,
            "test_data_used": False,
            "model_or_checkpoint_written": False,
            "current_best_commands_modified": False,
            "full_train_authorized": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_artifact_manifest(
        output_dir, mode="axial_color_topology_a0_nonbinary_evidence_manifest"
    )
    return summary


def _read_prediction_artifact(path: Path) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    raw = {name: [] for name, _brightness, _contrast in CONDITIONS}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = str(row["condition"])
            if condition not in raw:
                raise ValueError(f"Unexpected prediction condition: {condition}")
            raw[condition].append(row)
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    rows_by_condition: Dict[str, Dict[str, np.ndarray]] = {}
    reference = None
    for condition, _brightness, _contrast in CONDITIONS:
        rows = raw[condition]
        if len(rows) != EXPECTED_TRAIN_ROWS:
            raise ValueError(f"Prediction artifact is incomplete for {condition}")
        sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        if not np.array_equal(sample_index, np.arange(EXPECTED_TRAIN_ROWS)):
            raise ValueError("Prediction artifact sample order differs")
        target = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
        fold = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
        source = np.asarray([row["source_stem"] for row in rows], dtype=object)
        image_path = np.asarray([row["image_path"] for row in rows], dtype=object)
        metadata = (target, fold, source, image_path)
        if reference is None:
            reference = metadata
        elif any(not np.array_equal(a, b) for a, b in zip(metadata, reference)):
            raise ValueError("Prediction metadata differs across conditions")
        role_probabilities = {}
        for role in ("keeper", "control", "candidate", "placebo"):
            probabilities = np.asarray(
                [
                    [float(row[f"{role}_prob_{index}"]) for index in range(5)]
                    for row in rows
                ],
                dtype=np.float32,
            )
            declared = np.asarray(
                [int(row[f"{role}_prediction"]) for row in rows], dtype=np.int64
            )
            if not np.array_equal(probabilities.argmax(axis=1), declared):
                raise ValueError(f"Prediction artifact argmax mismatch for {condition}/{role}")
            role_probabilities[role] = probabilities
        outputs[condition] = {
            role: role_probabilities[role]
            for role in ("control", "candidate", "placebo")
        }
        rows_by_condition[condition] = {
            "sample_index": sample_index,
            "target": target,
            "fold": fold,
            "source": source,
            "image_path": image_path,
            "probabilities": role_probabilities["keeper"],
        }
    return outputs, rows_by_condition


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = Path(output_dir) / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = Path(output_dir) / str(row["name"])
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise ValueError(f"Artifact size differs during replay: {path}")
        if _sha256(path) != str(row["sha256"]):
            raise ValueError(f"Artifact hash differs during replay: {path}")
    return manifest


def _maximum_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return math.inf
        if len(left) != len(right):
            return math.inf
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right)),
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
    manifest = _verify_manifest(summary_path.parent)
    outputs, condition_rows = _read_prediction_artifact(
        summary_path.parent / summary["artifacts"]["predictions"]
    )
    results = _condition_results(outputs=outputs, condition_rows=condition_rows)
    gate = assess_axial_color_topology_a0(
        structural_checks=summary["gate"]["structural_checks"],
        condition_results=results,
    )
    results_difference = _maximum_numeric_difference(results, summary["results"])
    gate_difference = _maximum_numeric_difference(
        gate["checks"], summary["gate"]["checks"]
    )
    if results_difference > 1e-9 or gate_difference > 0.0:
        raise ValueError(
            "Replay differs from formal summary: "
            f"results={results_difference}, gates={gate_difference}"
        )
    return {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(summary_path.parent / "artifact_manifest.json"),
        "payload_manifest_sha256": manifest["payload_manifest_sha256"],
        "results_max_abs_difference": results_difference,
        "gate_difference": gate_difference,
        "replay_passed": True,
    }


def finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    summary_path = Path(args.output_dir).resolve() / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary for finalization: {summary_path}")
    expected = str(args.expected_summary_sha256).strip().lower()
    if not expected or _sha256(summary_path) != expected:
        raise ValueError("Finalization summary SHA-256 does not match the declared input")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError("Summary is not awaiting visual review")
    if not args.visual_review_result or not str(args.visual_review_note).strip():
        raise ValueError("Visual review result and note are required")
    pages = summary.get("contact_sheets", {})
    if set(pages) != {name for name, _brightness, _contrast in CONDITIONS}:
        raise ValueError("All four contact sheets are required for finalization")
    for page in pages.values():
        path = summary_path.parent / str(page["path"])
        if _sha256(path) != str(page["sha256"]):
            raise ValueError(f"Contact sheet hash differs: {path}")
    passed = str(args.visual_review_result) == "pass"
    summary["visual_review"] = {
        "result": str(args.visual_review_result),
        "note": str(args.visual_review_note).strip(),
        "input_summary_sha256": expected,
        "all_pages_verified": True,
    }
    summary["status"] = "stage_b_smoke_authorized" if passed else "rejected_after_visual_review"
    summary["gate"]["stage_b_smoke_authorized"] = bool(passed)
    summary["gate"]["full_train_authorized"] = False
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = _write_artifact_manifest(
        summary_path.parent,
        mode="axial_color_topology_a0_nonbinary_evidence_manifest",
    )
    return {
        "status": summary["status"],
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(summary_path.parent / "artifact_manifest.json"),
        "payload_manifest_sha256": manifest["payload_manifest_sha256"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _validate_locked_args(args)
    if args.replay_summary is not None:
        print(json.dumps(replay_summary(args.replay_summary), indent=2))
        return 0
    if args.finalize_visual_review:
        print(json.dumps(finalize_visual_review(args), indent=2))
        return 0
    if args.preflight_only:
        provenance = verify_provenance(args)
        print(
            json.dumps(
                {
                    "mode": "axial_color_topology_a0_preflight",
                    "provenance": provenance,
                    "dataset_opened": False,
                    "checkpoint_loaded": False,
                    "output_created": False,
                },
                indent=2,
            )
        )
        return 0
    summary = run_audit(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "failed_checks": summary["gate"]["failed_checks"],
                "summary": str(Path(args.output_dir).resolve() / "summary.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

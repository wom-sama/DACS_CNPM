from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence
import warnings

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata, spearmanr
from skimage.color import rgb2hsv, rgb2lab
from skimage.measure import euler_number
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.inference.inference import load_checkpoint
from trkh.tools.audit_axial_color_topology_a0 import (
    _SelectedConditionDataset,
    _build_dataset,
    _condition_corruption,
    _dataset_metadata_sha256,
    _ellipse_highlight_mask,
    _read_cidt_conditions,
    _resolve_device,
    source_derangement,
)
from trkh.tools.audit_lbp_surface_texture_readiness import interior_roi_boxes
from trkh.tools.audit_more_model_rebalancing_readiness import (
    _ordered_index_sha256,
)
from trkh.tools.audit_pixel_difference_stem_signal import (
    _git_value,
    _tracked_worktree_clean,
)
from trkh.tools.audit_surface_blob_morphology_readiness import (
    diffuse_color_control_descriptor,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _sha256,
    _write_artifact_manifest,
)


METHOD = "chromatic_cubical_persistence_a0"
SEED = 20260720
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FIT_FOLDS = (1, 2, 3, 4)

EXPECTED_COHORT_ROWS = 618
EXPECTED_POSITIVES = 432
EXPECTED_TRUE_POSITIVES = 421
EXPECTED_FALSE_NEGATIVES = 11
EXPECTED_RESTRICTED_FALSE_POSITIVES = 186
EXPECTED_FOLD_COUNTS = {
    1: {"positive": 115, "tp": 112, "fn": 3, "fp": 45},
    2: {"positive": 104, "tp": 100, "fn": 4, "fp": 48},
    3: {"positive": 103, "tp": 101, "fn": 2, "fp": 52},
    4: {"positive": 110, "tp": 108, "fn": 2, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "98b6e447ace4e3dc4d9a3fcd5402c25662b018697636c5e9d53b9f4d7e91b2dd"
)

ROI_SIZE = 48
ERODE_RATIO = 0.15
ELLIPSE_RADIUS = 0.92
GAUSSIAN_SIGMA = 0.8
PERSISTENCE_FLOOR = 1.0 / 32.0
BETTI_THRESHOLDS = np.arange(1, 17, dtype=np.float64) / 17.0
CHANNEL_NAMES = ("lab_a", "lab_b", "saturation")
POLARITY_NAMES = ("sublevel", "superlevel")
HOMOLOGY_DIMENSIONS = (0, 1)
DIAGRAM_FEATURE_DIM = 21
TOPOLOGY_DIM = 252
GLOBAL_COLOR_DIM = 25
BASE_DIM = 30
ROLE_DIM = 282
READOUT_C = 0.1
MAX_ITERATIONS = 300
TARGET_RETENTION = 0.98
ROLES = ("control", "candidate", "pixel_placebo", "source_placebo")

CONDITIONS = (
    ("clean", 1.0, 1.0),
    ("lighting_dim", 0.7, 0.9),
    ("lighting_bright", 1.25, 1.1),
    ("low_contrast", 1.0, 0.65),
)
SHIFTED_CONDITIONS = tuple(name for name, _, _ in CONDITIONS if name != "clean")

EXPECTED_HASHES = {
    "protocol": "6e91d163ef5a00dfefddfb9a271bfa07480b278266fbc0539b2f05dc3ce7892b",
    "paper": "41c862cb346b482f15f7a4a731909f41adcece65f974bb78096d23ac39af1b96",
    "wheel": "846ac5807d6a72ebc79a29b9405d12a68cfb8dad896ff5aa6a24fde343a8581f",
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
        / "TRKH_5CLASS_CHROMATIC_CUBICAL_PERSISTENCE_A0_PROTOCOL_20260720.md",
        "paper": Path("D:/DataAI/external_sources/papers/phg_net_wacv_2024.pdf"),
        "wheel": Path(
            "D:/DataAI/external_sources/wheels/"
            "gudhi-3.11.0-cp39-cp39-win_amd64.whl"
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
            "Locked train-only chromatic cubical-persistence A0. Validation, "
            "test, model writes, and parameter sweeps are rejected."
        )
    )
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--checkpoint", type=Path, default=defaults["keeper"])
    parser.add_argument("--cidt-summary", type=Path, default=defaults["cidt_summary"])
    parser.add_argument(
        "--cidt-predictions", type=Path, default=defaults["cidt_predictions"]
    )
    parser.add_argument("--paper", type=Path, default=defaults["paper"])
    parser.add_argument("--wheel", type=Path, default=defaults["wheel"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_chromatic_cubical_persistence_a0_20260720"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--descriptor-threads", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    defaults = _default_paths()
    locked_paths = {
        "data": defaults["data"],
        "checkpoint": defaults["keeper"],
        "cidt_summary": defaults["cidt_summary"],
        "cidt_predictions": defaults["cidt_predictions"],
        "paper": defaults["paper"],
        "wheel": defaults["wheel"],
    }
    for key, expected in locked_paths.items():
        observed = Path(getattr(args, key)).resolve()
        if observed != Path(expected).resolve():
            raise ValueError(f"Locked A0 rejects changed {key}: {observed}")
    if int(args.batch_size) != 64:
        raise ValueError("Locked A0 requires batch-size=64")
    if int(args.num_workers) != 4:
        raise ValueError("Locked A0 requires num-workers=4")
    if int(args.descriptor_threads) != 8:
        raise ValueError("Locked A0 requires descriptor-threads=8")
    if int(args.torch_threads) != 8:
        raise ValueError("Locked A0 requires torch-threads=8")
    if str(args.device) != "cuda":
        raise ValueError("Locked A0 requires device=cuda")
    output = Path(args.output_dir).resolve()
    data_root = Path(args.data).resolve().parent
    try:
        output.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must remain outside the raw dataset")


def _require_gudhi():
    try:
        import gudhi
    except ImportError as exc:
        raise RuntimeError(
            "gudhi==3.11.0 is required; install the protocol-locked local wheel"
        ) from exc
    if str(gudhi.__version__) != "3.11.0":
        raise RuntimeError(f"Locked A0 requires gudhi==3.11.0, got {gudhi.__version__}")
    return gudhi


def verify_provenance(args: argparse.Namespace) -> Dict[str, object]:
    defaults = _default_paths()
    paths = {
        **defaults,
        "paper": Path(args.paper),
        "wheel": Path(args.wheel),
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
        result[name] = {"path": str(path), "sha256": observed, "matched": True}
    gudhi = _require_gudhi()
    result["gudhi"] = {
        "version": str(gudhi.__version__),
        "module": str(Path(gudhi.__file__).resolve()),
        "matched": str(gudhi.__version__) == "3.11.0",
    }
    result["phg_repository"] = {
        "url": "https://github.com/yaoppeng/TopoClassification",
        "commit": "6daa5f7dba556e9882611eb4e2e1c89a67f0d2c5",
        "license_declared": False,
        "source_copied": False,
    }
    return result


def _coordinate_geometry(size: int = ROI_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = ((np.arange(int(size), dtype=np.float32) + 0.5) / float(size)) * 2.0 - 1.0
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    ellipse = np.square(x) + np.square(y) <= float(ELLIPSE_RADIUS) ** 2
    return x, y, np.asarray(ellipse, dtype=bool)


def _nearest_valid_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if array.shape != mask.shape or not mask.any():
        raise ValueError("Nearest-valid fill received invalid inputs")
    indices = ndimage.distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    return np.asarray(array[tuple(indices)], dtype=np.float32)


def _average_rank_map(values: np.ndarray, valid: np.ndarray, ellipse: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    ellipse_mask = np.asarray(ellipse, dtype=bool)
    selected = array[valid_mask]
    if selected.size < 2 or not np.isfinite(selected).all():
        raise ValueError("Rank map has insufficient finite support")
    selected_ranks = (rankdata(selected, method="average") - 1.0) / float(
        selected.size - 1
    )
    output = np.full(array.shape, 0.5, dtype=np.float64)
    output[valid_mask] = selected_ranks
    excluded_inside = ellipse_mask & ~valid_mask
    if excluded_inside.any():
        ordered = np.sort(selected)
        query = array[excluded_inside]
        left = np.searchsorted(ordered, query, side="left")
        right = np.searchsorted(ordered, query, side="right")
        average_rank = 0.5 * (left + right - 1.0)
        output[excluded_inside] = np.clip(
            average_rank / float(selected.size - 1), 0.0, 1.0
        )
    output[~ellipse_mask] = 0.5
    if not np.isfinite(output).all() or output.min() < 0.0 or output.max() > 1.0:
        raise RuntimeError("Rank map left the locked [0,1] range")
    return output.astype(np.float32)


def build_chromatic_rank_maps(
    rgb_uint8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    rgb = np.asarray(rgb_uint8)
    if rgb.shape != (ROI_SIZE, ROI_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError(
            f"Chromatic PH requires uint8 [{ROI_SIZE},{ROI_SIZE},3], got "
            f"{rgb.shape}/{rgb.dtype}"
        )
    values = rgb.astype(np.float32) / 255.0
    _x, _y, ellipse = _coordinate_geometry(ROI_SIZE)
    highlight, _score = _ellipse_highlight_mask(values, ellipse)
    valid = ellipse & ~highlight
    if int(valid.sum()) < 1024:
        raise ValueError("Chromatic PH valid support is unexpectedly small")
    lab = rgb2lab(values).astype(np.float32)
    hsv = rgb2hsv(values).astype(np.float32)
    channels = np.stack((lab[..., 1], lab[..., 2], hsv[..., 1]), axis=0)
    maps = []
    for channel in channels:
        filled = _nearest_valid_fill(channel, valid)
        smoothed = ndimage.gaussian_filter(
            filled, sigma=GAUSSIAN_SIGMA, mode="reflect"
        )
        maps.append(_average_rank_map(smoothed, valid, ellipse))
    rank_maps = np.stack(maps, axis=0).astype(np.float32)
    control = diffuse_color_control_descriptor(values, valid).astype(np.float32)
    if rank_maps.shape != (len(CHANNEL_NAMES), ROI_SIZE, ROI_SIZE):
        raise RuntimeError("Chromatic rank-map shape violates the protocol")
    if control.shape != (GLOBAL_COLOR_DIM,):
        raise RuntimeError("Global color control shape violates the protocol")
    telemetry = {
        "valid_fraction": float(valid.mean()),
        "highlight_fraction": float(highlight.sum() / max(1, ellipse.sum())),
        "minimum_channel_std": float(
            min(float(rank_maps[index][valid].std()) for index in range(3))
        ),
    }
    return control, rank_maps, valid.astype(np.uint8), telemetry


def _finite_intervals(field: np.ndarray) -> Dict[int, np.ndarray]:
    gudhi = _require_gudhi()
    values = np.asarray(field, dtype=np.float64)
    if values.shape != (ROI_SIZE, ROI_SIZE) or not np.isfinite(values).all():
        raise ValueError("Cubical field has invalid shape or values")
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=values)
    complex_.persistence(
        homology_coeff_field=2, min_persistence=float(PERSISTENCE_FLOOR)
    )
    output: Dict[int, np.ndarray] = {}
    for dimension in HOMOLOGY_DIMENSIONS:
        intervals = np.asarray(
            complex_.persistence_intervals_in_dimension(int(dimension)),
            dtype=np.float64,
        ).reshape(-1, 2)
        if intervals.size:
            intervals = intervals[np.isfinite(intervals).all(axis=1)]
            intervals = intervals[
                (intervals[:, 1] - intervals[:, 0]) > PERSISTENCE_FLOOR
            ]
            order = np.lexsort((intervals[:, 1], intervals[:, 0]))
            intervals = intervals[order]
        output[int(dimension)] = intervals
    return output


def vectorize_persistence_intervals(intervals: np.ndarray) -> np.ndarray:
    values = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if values.size and (
        not np.isfinite(values).all()
        or np.any(values[:, 1] <= values[:, 0])
    ):
        raise ValueError("Persistence intervals are invalid")
    betti = np.asarray(
        [
            np.count_nonzero(
                (values[:, 0] <= threshold) & (threshold < values[:, 1])
            )
            for threshold in BETTI_THRESHOLDS
        ],
        dtype=np.float64,
    )
    persistence = values[:, 1] - values[:, 0] if values.size else np.empty(0)
    summaries = np.asarray(
        [
            float(values.shape[0]),
            float(persistence.sum()) if persistence.size else 0.0,
            float(persistence.max()) if persistence.size else 0.0,
            float(np.quantile(persistence, 0.50)) if persistence.size else 0.0,
            float(np.quantile(persistence, 0.90)) if persistence.size else 0.0,
        ],
        dtype=np.float64,
    )
    output = np.concatenate((betti, summaries)).astype(np.float32)
    if output.shape != (DIAGRAM_FEATURE_DIM,) or not np.isfinite(output).all():
        raise RuntimeError("Persistence vectorization violates the protocol")
    return output


def cubical_topology_descriptor(rank_maps: np.ndarray) -> tuple[np.ndarray, Dict[str, int]]:
    maps = np.asarray(rank_maps, dtype=np.float32)
    if maps.shape != (len(CHANNEL_NAMES), ROI_SIZE, ROI_SIZE):
        raise ValueError("Topology descriptor received invalid rank maps")
    features = []
    counts: Dict[str, int] = {}
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        for polarity_name, field in (
            ("sublevel", maps[channel_index]),
            ("superlevel", 1.0 - maps[channel_index]),
        ):
            diagrams = _finite_intervals(field)
            for dimension in HOMOLOGY_DIMENSIONS:
                intervals = diagrams[dimension]
                features.append(vectorize_persistence_intervals(intervals))
                counts[f"{channel_name}_{polarity_name}_h{dimension}"] = int(
                    intervals.shape[0]
                )
    output = np.concatenate(features).astype(np.float32)
    if output.shape != (TOPOLOGY_DIM,) or not np.isfinite(output).all():
        raise RuntimeError("Topology descriptor dimension violates the protocol")
    return output, counts


def pixel_permutation_placebo(
    rank_maps: np.ndarray,
    *,
    sample_index: int,
) -> np.ndarray:
    maps = np.asarray(rank_maps, dtype=np.float32)
    _x, _y, ellipse = _coordinate_geometry(ROI_SIZE)
    output = maps.copy()
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        seed_bytes = hashlib.sha256(
            f"ccp-a0|{int(sample_index)}|{channel_name}".encode("ascii")
        ).digest()[:8]
        seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        values = maps[channel_index][ellipse]
        output[channel_index][ellipse] = values[rng.permutation(values.size)]
        if not np.array_equal(
            np.sort(output[channel_index].reshape(-1)),
            np.sort(maps[channel_index].reshape(-1)),
        ):
            raise RuntimeError("Pixel placebo changed the rank histogram")
    return output


def chromatic_persistence_descriptor(
    rgb_uint8: np.ndarray,
    *,
    sample_index: int,
    return_details: bool = False,
):
    control, rank_maps, valid, telemetry = build_chromatic_rank_maps(rgb_uint8)
    topology, counts = cubical_topology_descriptor(rank_maps)
    placebo_maps = pixel_permutation_placebo(
        rank_maps, sample_index=int(sample_index)
    )
    pixel_topology, pixel_counts = cubical_topology_descriptor(placebo_maps)
    changed_channels = int(
        sum(
            not np.array_equal(rank_maps[index], placebo_maps[index])
            for index in range(len(CHANNEL_NAMES))
        )
    )
    telemetry = {
        **telemetry,
        "diagram_interval_count": int(sum(counts.values())),
        "pixel_diagram_interval_count": int(sum(pixel_counts.values())),
        "pixel_changed_channels": changed_channels,
        "pixel_histogram_exact": True,
    }
    if not return_details:
        return control, topology, pixel_topology, telemetry
    return control, topology, pixel_topology, telemetry, {
        "rank_maps": rank_maps,
        "pixel_rank_maps": placebo_maps,
        "valid": valid,
        "diagram_counts": counts,
        "pixel_diagram_counts": pixel_counts,
    }


def synthetic_oracle_checks() -> Dict[str, object]:
    constant = np.full((ROI_SIZE, ROI_SIZE), 0.5, dtype=np.float32)
    constant_a = _finite_intervals(constant)
    constant_b = _finite_intervals(constant.copy())

    components = np.ones((ROI_SIZE, ROI_SIZE), dtype=np.float32)
    components[8:14, 8:14] = 0.1
    components[32:38, 32:38] = 0.1
    component_diagrams = _finite_intervals(components)
    component_binary = components <= 0.5
    component_count = int(ndimage.label(component_binary)[1])

    ring = np.ones((ROI_SIZE, ROI_SIZE), dtype=np.float32)
    ring[10:38, 10:38] = 0.2
    ring[18:30, 18:30] = 0.8
    ring_diagrams = _finite_intervals(ring)
    ring_binary = ring <= 0.5
    ring_components = int(ndimage.label(ring_binary)[1])
    ring_euler = int(euler_number(ring_binary, connectivity=1))
    ring_holes = int(ring_components - ring_euler)

    checks = {
        "constant_no_finite_intervals": all(
            constant_a[dimension].shape[0] == 0 for dimension in HOMOLOGY_DIMENSIONS
        ),
        "deterministic_two_pass": all(
            np.array_equal(constant_a[dimension], constant_b[dimension])
            for dimension in HOMOLOGY_DIMENSIONS
        ),
        "two_components_binary_oracle": component_count == 2,
        "two_components_h0_finite_one": component_diagrams[0].shape[0] == 1,
        "ring_binary_one_hole": ring_holes == 1,
        "ring_h1_finite_one": ring_diagrams[1].shape[0] == 1,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "component_h0": component_diagrams[0].tolist(),
        "ring_h1": ring_diagrams[1].tolist(),
        "persistence_floor": PERSISTENCE_FLOOR,
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    provenance = verify_provenance(args)
    cidt_summary = json.loads(Path(args.cidt_summary).read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use")
    conditions = _read_cidt_conditions(Path(args.cidt_predictions))
    clean = conditions["clean"]
    prediction = np.asarray(clean["probabilities"]).argmax(axis=1)
    target = np.asarray(clean["target"], dtype=np.int64)
    fold = np.asarray(clean["fold"], dtype=np.int64)
    selected = np.flatnonzero(
        np.isin(fold, FIT_FOLDS)
        & (
            (target == FOCUS_CLASS)
            | (
                np.isin(target, RESTRICTED_NEGATIVE_CLASSES)
                & (prediction == FOCUS_CLASS)
            )
        )
    ).astype(np.int64)
    selected_target = target[selected]
    selected_prediction = prediction[selected]
    positives = selected_target == FOCUS_CLASS
    tp = positives & (selected_prediction == FOCUS_CLASS)
    fn = positives & ~tp
    fp = ~positives
    observed = (
        int(selected.size),
        int(positives.sum()),
        int(tp.sum()),
        int(fn.sum()),
        int(fp.sum()),
    )
    expected = (
        EXPECTED_COHORT_ROWS,
        EXPECTED_POSITIVES,
        EXPECTED_TRUE_POSITIVES,
        EXPECTED_FALSE_NEGATIVES,
        EXPECTED_RESTRICTED_FALSE_POSITIVES,
    )
    if observed != expected:
        raise ValueError(f"Locked cohort differs: {observed} != {expected}")
    fold_counts: Dict[int, Dict[str, int]] = {}
    for value in FIT_FOLDS:
        mask = fold[selected] == value
        fold_counts[value] = {
            "positive": int((mask & positives).sum()),
            "tp": int((mask & tp).sum()),
            "fn": int((mask & fn).sum()),
            "fp": int((mask & fp).sum()),
        }
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked cohort fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256(selected.tolist())
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")
    provenance["cohort"] = {
        "rows": int(selected.size),
        "positives": int(positives.sum()),
        "true_positives": int(tp.sum()),
        "false_negatives": int(fn.sum()),
        "restricted_false_positives": int(fp.sum()),
        "fold_counts": fold_counts,
        "ordered_index_sha256": ordered_hash,
    }
    return provenance, conditions, selected


def _make_loader(
    *,
    base_dataset,
    transform,
    condition: str,
    brightness: float,
    contrast: float,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Dict[str, object]]:
    selected = _SelectedConditionDataset(
        base_dataset,
        indices,
        corruption=_condition_corruption(condition, brightness, contrast),
        transform=transform,
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=f"chromatic_cubical_persistence_{condition}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    return (
        DataLoader(
            selected,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            **kwargs,
        ),
        summary,
    )


def _descriptor_job(item):
    rgb, sample_index = item
    return chromatic_persistence_descriptor(
        rgb, sample_index=int(sample_index), return_details=True
    )


def _extract_condition_descriptors(
    *,
    condition: str,
    brightness: float,
    contrast: float,
    base_dataset,
    transform,
    checkpoint: Mapping[str, object],
    selected_indices: np.ndarray,
    expected: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    descriptor_threads: int,
) -> Dict[str, object]:
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        condition=condition,
        brightness=brightness,
        contrast=contrast,
        indices=selected_indices.tolist(),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    controls = []
    topology = []
    pixel_topology = []
    telemetry = []
    roi_rows = []
    rank_rows = []
    observed_indices = []
    observed_targets = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with ThreadPoolExecutor(max_workers=int(descriptor_threads)) as executor:
        for images, targets, metadata in tqdm(
            loader, desc=f"ccp-{condition}", dynamic_ncols=True
        ):
            if not isinstance(metadata, Mapping):
                raise ValueError("Chromatic PH extraction requires metadata")
            crop_bbox = metadata.get("crop_bbox")
            sample_index = metadata.get("sample_index")
            if not torch.is_tensor(crop_bbox) or not torch.is_tensor(sample_index):
                raise ValueError("Chromatic PH requires crop_bbox/sample_index tensors")
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
            batch_indices = sample_index.cpu().numpy().astype(np.int64)
            jobs = zip(rgb_uint8, batch_indices.tolist())
            results = list(executor.map(_descriptor_job, jobs))
            for roi, result in zip(rgb_uint8, results):
                control, real, pixel, row_telemetry, details = result
                controls.append(control)
                topology.append(real)
                pixel_topology.append(pixel)
                telemetry.append(row_telemetry)
                roi_rows.append(roi)
                rank_rows.append(details["rank_maps"])
            observed_indices.extend(batch_indices.tolist())
            observed_targets.extend(targets.cpu().numpy().astype(np.int64).tolist())
    if observed_indices != selected_indices.tolist():
        raise ValueError("Descriptor loader sample order differs from the locked cohort")
    expected_targets = np.asarray(expected["target"], dtype=np.int64)[selected_indices]
    if not np.array_equal(np.asarray(observed_targets, dtype=np.int64), expected_targets):
        raise ValueError("Descriptor loader targets differ from CIDT")
    arrays = {
        "global": np.stack(controls).astype(np.float32),
        "topology": np.stack(topology).astype(np.float32),
        "pixel_topology": np.stack(pixel_topology).astype(np.float32),
        "roi_rgb": np.stack(roi_rows).astype(np.uint8),
        "rank_maps": np.stack(rank_rows).astype(np.float32),
    }
    expected_shapes = {
        "global": (EXPECTED_COHORT_ROWS, GLOBAL_COLOR_DIM),
        "topology": (EXPECTED_COHORT_ROWS, TOPOLOGY_DIM),
        "pixel_topology": (EXPECTED_COHORT_ROWS, TOPOLOGY_DIM),
        "roi_rgb": (EXPECTED_COHORT_ROWS, ROI_SIZE, ROI_SIZE, 3),
        "rank_maps": (EXPECTED_COHORT_ROWS, 3, ROI_SIZE, ROI_SIZE),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"Incomplete {name}: {arrays[name].shape} != {shape}")
        if name != "roi_rgb" and not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"Non-finite descriptor values in {name}")
    telemetry_summary = {
        key: {
            "minimum": float(min(float(row[key]) for row in telemetry)),
            "mean": float(np.mean([float(row[key]) for row in telemetry])),
            "maximum": float(max(float(row[key]) for row in telemetry)),
        }
        for key in telemetry[0]
    }
    return {
        **arrays,
        "telemetry": telemetry_summary,
        "loader": loader_summary,
        "seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }


def _source_mappings(
    folds: np.ndarray,
    sources: np.ndarray,
) -> Dict[int, np.ndarray]:
    mappings: Dict[int, np.ndarray] = {}
    positions = np.arange(folds.size, dtype=np.int64)
    for outer_fold in FIT_FOLDS:
        fit = positions[folds != outer_fold]
        held = positions[folds == outer_fold]
        mapping = positions.copy()
        mapping[fit] = source_derangement(
            fit, sources.tolist(), seed=SEED + outer_fold * 101 + 1
        )
        mapping[held] = source_derangement(
            held, sources.tolist(), seed=SEED + outer_fold * 101 + 2
        )
        if np.any(sources[mapping] == sources):
            raise RuntimeError("Source placebo retained a same-source descriptor")
        mappings[int(outer_fold)] = mapping
    return mappings


def assemble_role_features(
    *,
    probabilities: np.ndarray,
    global_color: np.ndarray,
    topology: np.ndarray,
    pixel_topology: np.ndarray,
    role: str,
    source_mapping: Optional[np.ndarray] = None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    color = np.asarray(global_color, dtype=np.float64)
    real = np.asarray(topology, dtype=np.float64)
    pixel = np.asarray(pixel_topology, dtype=np.float64)
    rows = probs.shape[0]
    if probs.shape != (rows, 5) or color.shape != (rows, GLOBAL_COLOR_DIM):
        raise ValueError("Base readout features have invalid shapes")
    if real.shape != (rows, TOPOLOGY_DIM) or pixel.shape != (rows, TOPOLOGY_DIM):
        raise ValueError("Topology readout features have invalid shapes")
    base = np.concatenate((np.log(np.clip(probs, 1e-7, 1.0)), color), axis=1)
    if role == "control":
        extra = np.zeros_like(real)
    elif role == "candidate":
        extra = real
    elif role == "pixel_placebo":
        extra = pixel
    elif role == "source_placebo":
        if source_mapping is None:
            raise ValueError("Source placebo requires an explicit mapping")
        mapping = np.asarray(source_mapping, dtype=np.int64)
        if mapping.shape != (rows,):
            raise ValueError("Source-placebo mapping has invalid shape")
        extra = real[mapping]
    else:
        raise ValueError(f"Unknown readout role: {role}")
    output = np.concatenate((base, extra), axis=1).astype(np.float64)
    if output.shape != (rows, ROLE_DIM) or not np.isfinite(output).all():
        raise RuntimeError("Readout role features violate the protocol")
    return output


def _fit_readout(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    role: str,
    outer_fold: int,
    fit_kind: str,
) -> tuple[StandardScaler, LogisticRegression, Dict[str, object]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != ROLE_DIM or x.shape[0] != y.size:
        raise ValueError("Readout fit arrays are misaligned")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Readout fit requires both binary classes")
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    model = LogisticRegression(
        C=READOUT_C,
        solver="lbfgs",
        max_iter=MAX_ITERATIONS,
        tol=1e-8,
        fit_intercept=True,
        class_weight=None,
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaled, y)
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    iterations = int(np.max(model.n_iter_))
    converged = not convergence_warnings and iterations < MAX_ITERATIONS
    record = {
        "role": role,
        "outer_fold": int(outer_fold),
        "fit_kind": fit_kind,
        "rows": int(y.size),
        "positives": int(y.sum()),
        "iterations": iterations,
        "converged": bool(converged),
        "convergence_warnings": convergence_warnings,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
    }
    return scaler, model, record


def _predict_readout(
    scaler: StandardScaler,
    model: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    probabilities = model.predict_proba(scaler.transform(features))[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Readout produced non-finite probabilities")
    return probabilities.astype(np.float64)


def select_retention_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_retention: float = TARGET_RETENTION,
) -> tuple[float, Dict[str, object]]:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(values)
    eligible = []
    for threshold in candidates:
        accepted = values >= threshold
        retention = float(accepted[y == 1].mean())
        rejection = float((~accepted[y == 0]).mean())
        if retention + 1e-12 >= float(target_retention):
            eligible.append((float(threshold), retention, rejection))
    if not eligible:
        raise RuntimeError("No threshold satisfies the locked retention constraint")
    selected = max(eligible, key=lambda row: (row[0], row[2]))
    return selected[0], {
        "threshold": selected[0],
        "inner_positive_retention": selected[1],
        "inner_restricted_fp_rejection": selected[2],
        "candidate_count": int(candidates.size),
    }


def _fit_nested_oof(
    *,
    descriptors: Mapping[str, Mapping[str, object]],
    conditions: Mapping[str, Mapping[str, np.ndarray]],
    selected_indices: np.ndarray,
) -> tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, object]]:
    clean = conditions["clean"]
    target = np.asarray(clean["target"], dtype=np.int64)[selected_indices]
    labels = (target == FOCUS_CLASS).astype(np.int64)
    folds = np.asarray(clean["fold"], dtype=np.int64)[selected_indices]
    sources = np.asarray(clean["source"], dtype=object)[selected_indices]
    mappings = _source_mappings(folds, sources)
    outputs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        condition: {
            role: {
                "score": np.full(EXPECTED_COHORT_ROWS, np.nan, dtype=np.float64),
                "threshold": np.full(EXPECTED_COHORT_ROWS, np.nan, dtype=np.float64),
                "accepted": np.zeros(EXPECTED_COHORT_ROWS, dtype=bool),
            }
            for role in ROLES
        }
        for condition, _, _ in CONDITIONS
    }
    fit_records = []
    threshold_records = []
    outer_model_records = []
    positions = np.arange(EXPECTED_COHORT_ROWS, dtype=np.int64)

    for role in ROLES:
        for outer_fold in FIT_FOLDS:
            held = positions[folds == outer_fold]
            outer_fit = positions[folds != outer_fold]
            source_mapping = mappings[outer_fold] if role == "source_placebo" else None
            clean_features = assemble_role_features(
                probabilities=np.asarray(clean["probabilities"])[selected_indices],
                global_color=np.asarray(descriptors["clean"]["global"]),
                topology=np.asarray(descriptors["clean"]["topology"]),
                pixel_topology=np.asarray(descriptors["clean"]["pixel_topology"]),
                role=role,
                source_mapping=source_mapping,
            )
            inner_scores = np.full(EXPECTED_COHORT_ROWS, np.nan, dtype=np.float64)
            for inner_fold in FIT_FOLDS:
                if inner_fold == outer_fold:
                    continue
                inner_held = positions[folds == inner_fold]
                inner_train = positions[
                    (folds != outer_fold) & (folds != inner_fold)
                ]
                scaler, model, record = _fit_readout(
                    clean_features[inner_train],
                    labels[inner_train],
                    role=role,
                    outer_fold=outer_fold,
                    fit_kind=f"inner_holdout_{inner_fold}",
                )
                fit_records.append(record)
                inner_scores[inner_held] = _predict_readout(
                    scaler, model, clean_features[inner_held]
                )
            if not np.isfinite(inner_scores[outer_fit]).all():
                raise RuntimeError("Nested inner OOF scores are incomplete")
            threshold, threshold_record = select_retention_threshold(
                inner_scores[outer_fit], labels[outer_fit]
            )
            threshold_records.append(
                {
                    "role": role,
                    "outer_fold": int(outer_fold),
                    **threshold_record,
                }
            )
            scaler, model, record = _fit_readout(
                clean_features[outer_fit],
                labels[outer_fit],
                role=role,
                outer_fold=outer_fold,
                fit_kind="outer",
            )
            fit_records.append(record)
            outer_model_records.append(record)
            for condition, _, _ in CONDITIONS:
                condition_data = conditions[condition]
                condition_descriptors = descriptors[condition]
                features = assemble_role_features(
                    probabilities=np.asarray(condition_data["probabilities"])[
                        selected_indices
                    ],
                    global_color=np.asarray(condition_descriptors["global"]),
                    topology=np.asarray(condition_descriptors["topology"]),
                    pixel_topology=np.asarray(
                        condition_descriptors["pixel_topology"]
                    ),
                    role=role,
                    source_mapping=source_mapping,
                )
                scores = _predict_readout(scaler, model, features[held])
                outputs[condition][role]["score"][held] = scores
                outputs[condition][role]["threshold"][held] = threshold
                outputs[condition][role]["accepted"][held] = scores >= threshold

    for condition, _, _ in CONDITIONS:
        for role in ROLES:
            if not np.isfinite(outputs[condition][role]["score"]).all():
                raise RuntimeError(f"Incomplete OOF scores for {condition}/{role}")
            if not np.isfinite(outputs[condition][role]["threshold"]).all():
                raise RuntimeError(f"Incomplete thresholds for {condition}/{role}")
    diagnostics = {
        "fit_count": len(fit_records),
        "expected_fit_count": 64,
        "all_converged": bool(all(record["converged"] for record in fit_records)),
        "maximum_iterations": int(max(record["iterations"] for record in fit_records)),
        "fit_records": fit_records,
        "threshold_records": threshold_records,
        "source_mappings": {
            str(outer_fold): mappings[outer_fold].astype(int).tolist()
            for outer_fold in FIT_FOLDS
        },
        "source_mapping_safe": bool(
            all(
                np.all(sources[mappings[outer_fold]] != sources)
                for outer_fold in FIT_FOLDS
            )
        ),
        "outer_model_count": len(outer_model_records),
    }
    return outputs, diagnostics


def _action_predictions(
    probabilities: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    base = probs.argmax(axis=1).astype(np.int64)
    action = base.copy()
    suppressed = (base == FOCUS_CLASS) & ~np.asarray(accepted, dtype=bool)
    non_focus = probs.copy()
    non_focus[:, FOCUS_CLASS] = -np.inf
    action[suppressed] = non_focus[suppressed].argmax(axis=1)
    return base, action


def _role_metrics(
    *,
    condition: str,
    role: str,
    output: Mapping[str, np.ndarray],
    condition_rows: Mapping[str, np.ndarray],
    clean_rows: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
    clean_candidate_scores: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    target = np.asarray(clean_rows["target"], dtype=np.int64)[selected_indices]
    fold = np.asarray(clean_rows["fold"], dtype=np.int64)[selected_indices]
    clean_probabilities = np.asarray(clean_rows["probabilities"], dtype=np.float64)[
        selected_indices
    ]
    clean_prediction = clean_probabilities.argmax(axis=1)
    labels = (target == FOCUS_CLASS).astype(np.int64)
    tp_mask = (target == FOCUS_CLASS) & (clean_prediction == FOCUS_CLASS)
    fn_mask = (target == FOCUS_CLASS) & (clean_prediction != FOCUS_CLASS)
    fp_mask = np.isin(target, RESTRICTED_NEGATIVE_CLASSES) & (
        clean_prediction == FOCUS_CLASS
    )
    scores = np.asarray(output["score"], dtype=np.float64)
    thresholds = np.asarray(output["threshold"], dtype=np.float64)
    accepted = np.asarray(output["accepted"], dtype=bool)
    probabilities = np.asarray(condition_rows["probabilities"], dtype=np.float64)[
        selected_indices
    ]
    base_prediction, action_prediction = _action_predictions(probabilities, accepted)
    corrections = (base_prediction != target) & (action_prediction == target)
    harms = (base_prediction == target) & (action_prediction != target)
    per_fold = {}
    for value in FIT_FOLDS:
        mask = fold == value
        fold_tp = mask & tp_mask
        fold_fp = mask & fp_mask
        per_fold[str(value)] = {
            "tp_rows": int(fold_tp.sum()),
            "tp_breaks": int((fold_tp & ~accepted).sum()),
            "restricted_fp_rows": int(fold_fp.sum()),
            "restricted_fp_removals": int((fold_fp & ~accepted).sum()),
        }
    correlation = 1.0
    if clean_candidate_scores is not None:
        correlation_result = spearmanr(
            np.asarray(clean_candidate_scores, dtype=np.float64), scores
        )
        correlation = float(correlation_result.statistic)
    return {
        "condition": condition,
        "role": role,
        "auc": float(roc_auc_score(labels, scores)),
        "tp_rows": int(tp_mask.sum()),
        "tp_retained": int((tp_mask & accepted).sum()),
        "tp_breaks": int((tp_mask & ~accepted).sum()),
        "tp_retention": float(accepted[tp_mask].mean()),
        "fn_rows": int(fn_mask.sum()),
        "fn_accepted": int((fn_mask & accepted).sum()),
        "fn_acceptance": float(accepted[fn_mask].mean()),
        "restricted_fp_rows": int(fp_mask.sum()),
        "restricted_fp_removals": int((fp_mask & ~accepted).sum()),
        "restricted_fp_rejection": float((~accepted[fp_mask]).mean()),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "changed": int((base_prediction != action_prediction).sum()),
        "score_minimum": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_maximum": float(scores.max()),
        "threshold_minimum": float(thresholds.min()),
        "threshold_mean": float(thresholds.mean()),
        "threshold_maximum": float(thresholds.max()),
        "clean_score_spearman": correlation,
        "per_fold": per_fold,
        "base_prediction": base_prediction,
        "action_prediction": action_prediction,
    }


def _condition_results(
    *,
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    conditions: Mapping[str, Mapping[str, np.ndarray]],
    selected_indices: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    results: Dict[str, Dict[str, Dict[str, object]]] = {}
    clean_candidate_scores = np.asarray(outputs["clean"]["candidate"]["score"])
    for condition, _, _ in CONDITIONS:
        results[condition] = {}
        for role in ROLES:
            metrics = _role_metrics(
                condition=condition,
                role=role,
                output=outputs[condition][role],
                condition_rows=conditions[condition],
                clean_rows=conditions["clean"],
                selected_indices=selected_indices,
                clean_candidate_scores=(
                    clean_candidate_scores
                    if condition != "clean" and role == "candidate"
                    else None
                ),
            )
            metrics.pop("base_prediction")
            metrics.pop("action_prediction")
            results[condition][role] = metrics
    return results


def assess_chromatic_persistence_a0(
    *,
    structural_checks: Mapping[str, bool],
    results: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    clean = results["clean"]
    candidate = clean["candidate"]
    control = clean["control"]
    pixel = clean["pixel_placebo"]
    source = clean["source_placebo"]
    clean_checks = {
        "candidate_auc_minimum": float(candidate["auc"]) >= 0.68,
        "auc_gain_over_control": float(candidate["auc"]) - float(control["auc"]) >= 0.03,
        "auc_gain_over_pixel_placebo": float(candidate["auc"]) - float(pixel["auc"]) >= 0.03,
        "auc_gain_over_source_placebo": float(candidate["auc"]) - float(source["auc"]) >= 0.03,
        "tp_retention": float(candidate["tp_retention"]) >= 0.98,
        "fn_acceptance_count": int(candidate["fn_accepted"]) >= 9,
        "restricted_fp_rejection": float(candidate["restricted_fp_rejection"]) >= 0.15,
        "restricted_fp_removal_count": int(candidate["restricted_fp_removals"]) >= 28,
        "fp_gain_over_control": int(candidate["restricted_fp_removals"])
        - int(control["restricted_fp_removals"])
        >= 9,
        "fp_minus_tp_breaks": int(candidate["restricted_fp_removals"])
        - int(candidate["tp_breaks"])
        >= 20,
        "every_fold_fp_and_tp_safe": all(
            int(row["restricted_fp_removals"]) >= 2 and int(row["tp_breaks"]) <= 3
            for row in candidate["per_fold"].values()
        ),
        "corrections_exceed_harms": int(candidate["corrections"]) > int(candidate["harms"]),
    }
    shifted_checks: Dict[str, bool] = {}
    candidate_aucs = [float(candidate["auc"])]
    control_deltas = []
    pixel_deltas = []
    source_deltas = []
    shifted_net = 0
    for condition in SHIFTED_CONDITIONS:
        condition_candidate = results[condition]["candidate"]
        condition_control = results[condition]["control"]
        condition_pixel = results[condition]["pixel_placebo"]
        condition_source = results[condition]["source_placebo"]
        candidate_aucs.append(float(condition_candidate["auc"]))
        control_delta = float(condition_candidate["auc"]) - float(condition_control["auc"])
        pixel_delta = float(condition_candidate["auc"]) - float(condition_pixel["auc"])
        source_delta = float(condition_candidate["auc"]) - float(condition_source["auc"])
        control_deltas.append(control_delta)
        pixel_deltas.append(pixel_delta)
        source_deltas.append(source_delta)
        removals = int(condition_candidate["restricted_fp_removals"])
        breaks = int(condition_candidate["tp_breaks"])
        shifted_net += removals - breaks
        shifted_checks[f"{condition}_auc"] = float(condition_candidate["auc"]) >= 0.62
        shifted_checks[f"{condition}_control_delta"] = control_delta >= 0.0
        shifted_checks[f"{condition}_tp_retention"] = (
            float(condition_candidate["tp_retention"]) >= 0.95
        )
        shifted_checks[f"{condition}_fn_acceptance"] = int(condition_candidate["fn_accepted"]) >= 8
        shifted_checks[f"{condition}_fp_rejection"] = (
            float(condition_candidate["restricted_fp_rejection"]) >= 0.10
        )
        shifted_checks[f"{condition}_net_removal"] = removals > breaks
        shifted_checks[f"{condition}_score_spearman"] = (
            float(condition_candidate["clean_score_spearman"]) >= 0.65
        )
    aggregate_checks = {
        "four_condition_mean_auc": float(np.mean(candidate_aucs)) >= 0.66,
        "shifted_mean_control_auc_gain": float(np.mean(control_deltas)) >= 0.02,
        "shifted_mean_pixel_auc_gain": float(np.mean(pixel_deltas)) >= 0.02,
        "shifted_mean_source_auc_gain": float(np.mean(source_deltas)) >= 0.02,
        "shifted_aggregate_net_removal": int(shifted_net) >= 30,
    }
    structural_passed = bool(all(bool(value) for value in structural_checks.values()))
    automatic = bool(
        structural_passed
        and all(clean_checks.values())
        and all(shifted_checks.values())
        and all(aggregate_checks.values())
    )
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "shifted_checks": shifted_checks,
        "aggregate_checks": aggregate_checks,
        "structural_passed": structural_passed,
        "automatic_gates_passed": automatic,
        "visual_review_required": automatic,
        "full_train_authorized": False,
    }


def _descriptor_summary(
    descriptors: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    output = {}
    for condition, _, _ in CONDITIONS:
        row = descriptors[condition]
        topology = np.asarray(row["topology"], dtype=np.float64)
        singular = np.linalg.svd(
            topology - topology.mean(axis=0, keepdims=True),
            compute_uv=False,
        )
        energy = np.square(singular)
        effective_rank = float(
            np.square(energy.sum()) / max(float(np.square(energy).sum()), 1e-12)
        )
        output[condition] = {
            "rows": int(topology.shape[0]),
            "global_shape": list(np.asarray(row["global"]).shape),
            "topology_shape": list(topology.shape),
            "pixel_topology_shape": list(np.asarray(row["pixel_topology"]).shape),
            "topology_effective_rank": effective_rank,
            "telemetry": row["telemetry"],
            "loader": row["loader"],
            "seconds": float(row["seconds"]),
            "peak_cuda_memory_mib": float(row["peak_cuda_memory_mib"]),
        }
    return output


def _structural_checks(
    *,
    provenance: Mapping[str, object],
    descriptors: Mapping[str, Mapping[str, object]],
    diagnostics: Mapping[str, object],
    synthetic: Mapping[str, object],
    raw_before: str,
    raw_after: str,
    command_hashes_after: Mapping[str, str],
) -> Dict[str, bool]:
    descriptor_shapes = all(
        np.asarray(descriptors[condition]["global"]).shape
        == (EXPECTED_COHORT_ROWS, GLOBAL_COLOR_DIM)
        and np.asarray(descriptors[condition]["topology"]).shape
        == (EXPECTED_COHORT_ROWS, TOPOLOGY_DIM)
        and np.asarray(descriptors[condition]["pixel_topology"]).shape
        == (EXPECTED_COHORT_ROWS, TOPOLOGY_DIM)
        for condition, _, _ in CONDITIONS
    )
    finite = all(
        np.isfinite(np.asarray(descriptors[condition][key])).all()
        for condition, _, _ in CONDITIONS
        for key in ("global", "topology", "pixel_topology", "rank_maps")
    )
    histogram_exact = all(
        bool(descriptors[condition]["telemetry"]["pixel_histogram_exact"]["minimum"])
        for condition, _, _ in CONDITIONS
    )
    placebo_changed = all(
        float(descriptors[condition]["telemetry"]["pixel_changed_channels"]["minimum"])
        == float(len(CHANNEL_NAMES))
        for condition, _, _ in CONDITIONS
    )
    return {
        "provenance_hashes_exact": all(
            bool(value.get("matched", False))
            for key, value in provenance.items()
            if key in EXPECTED_HASHES or key == "gudhi"
        ),
        "cohort_exact": provenance["cohort"]["rows"] == EXPECTED_COHORT_ROWS,
        "descriptor_shapes_exact": descriptor_shapes,
        "descriptors_finite": finite,
        "pixel_histograms_exact": histogram_exact,
        "pixel_adjacency_changed": placebo_changed,
        "synthetic_h0_h1_oracle": bool(synthetic["passed"]),
        "readout_fit_count_exact": int(diagnostics["fit_count"]) == 64,
        "all_readouts_converged": bool(diagnostics["all_converged"]),
        "source_placebo_safe": bool(diagnostics["source_mapping_safe"]),
        "raw_dataset_metadata_unchanged": raw_before == raw_after,
        "current_best_hash_unchanged": command_hashes_after["current_best"]
        == EXPECTED_HASHES["current_best"],
        "command_history_hash_unchanged": command_hashes_after["command_history"]
        == EXPECTED_HASHES["command_history"],
        "validation_data_unused": True,
        "test_data_unused": True,
        "model_or_checkpoint_written_false": True,
    }


def _write_predictions(
    path: Path,
    *,
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    conditions: Mapping[str, Mapping[str, np.ndarray]],
    selected_indices: np.ndarray,
) -> None:
    fieldnames = [
        "condition",
        "cohort_position",
        "sample_index",
        "source_stem",
        "fold",
        "target_index",
        "clean_keeper_prediction",
        "condition_keeper_prediction",
        *[f"keeper_prob_{index}" for index in range(5)],
    ]
    for role in ROLES:
        fieldnames.extend(
            (
                f"{role}_score",
                f"{role}_threshold",
                f"{role}_accepted",
                f"{role}_action_prediction",
            )
        )
    clean = conditions["clean"]
    clean_probs = np.asarray(clean["probabilities"], dtype=np.float64)[selected_indices]
    clean_prediction = clean_probs.argmax(axis=1)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, _, _ in CONDITIONS:
            condition_row = conditions[condition]
            probabilities = np.asarray(
                condition_row["probabilities"], dtype=np.float64
            )[selected_indices]
            condition_prediction = probabilities.argmax(axis=1)
            action_predictions = {}
            for role in ROLES:
                _base, action = _action_predictions(
                    probabilities, outputs[condition][role]["accepted"]
                )
                action_predictions[role] = action
            for position, sample_index in enumerate(selected_indices.tolist()):
                row = {
                    "condition": condition,
                    "cohort_position": position,
                    "sample_index": sample_index,
                    "source_stem": str(clean["source"][sample_index]),
                    "fold": int(clean["fold"][sample_index]),
                    "target_index": int(clean["target"][sample_index]),
                    "clean_keeper_prediction": int(clean_prediction[position]),
                    "condition_keeper_prediction": int(condition_prediction[position]),
                }
                for class_index in range(5):
                    row[f"keeper_prob_{class_index}"] = repr(
                        float(probabilities[position, class_index])
                    )
                for role in ROLES:
                    output = outputs[condition][role]
                    row[f"{role}_score"] = repr(float(output["score"][position]))
                    row[f"{role}_threshold"] = repr(
                        float(output["threshold"][position])
                    )
                    row[f"{role}_accepted"] = int(output["accepted"][position])
                    row[f"{role}_action_prediction"] = int(
                        action_predictions[role][position]
                    )
                writer.writerow(row)


def _visual_categories(
    *,
    target: np.ndarray,
    clean_prediction: np.ndarray,
    accepted: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> list[tuple[str, int]]:
    tp = (target == FOCUS_CLASS) & (clean_prediction == FOCUS_CLASS)
    fn = (target == FOCUS_CLASS) & (clean_prediction != FOCUS_CLASS)
    fp = np.isin(target, RESTRICTED_NEGATIVE_CLASSES) & (
        clean_prediction == FOCUS_CLASS
    )
    specifications = (
        ("removed_fp", fp & ~accepted, True),
        ("retained_tp", tp & accepted, False),
        ("broken_tp", tp & ~accepted, True),
        ("accepted_fn", fn & accepted, False),
        ("missed_fp", fp & accepted, False),
    )
    selected = []
    margin = np.abs(scores - thresholds)
    for name, mask, ascending in specifications:
        positions = np.flatnonzero(mask)
        order = sorted(
            positions.tolist(),
            key=lambda index: float(margin[index]),
            reverse=not ascending,
        )
        selected.extend((name, index) for index in order[:2])
    return selected[:10]


def _generate_contact_sheets(
    *,
    output_dir: Path,
    descriptors: Mapping[str, Mapping[str, object]],
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    conditions: Mapping[str, Mapping[str, np.ndarray]],
    selected_indices: np.ndarray,
) -> Dict[str, object]:
    import matplotlib.pyplot as plt

    clean = conditions["clean"]
    target = np.asarray(clean["target"], dtype=np.int64)[selected_indices]
    clean_prediction = np.asarray(clean["probabilities"])[selected_indices].argmax(axis=1)
    manifest = []
    paths = []
    for condition, _, _ in CONDITIONS:
        candidate = outputs[condition]["candidate"]
        selections = _visual_categories(
            target=target,
            clean_prediction=clean_prediction,
            accepted=np.asarray(candidate["accepted"]),
            scores=np.asarray(candidate["score"]),
            thresholds=np.asarray(candidate["threshold"]),
        )
        if not selections:
            raise RuntimeError("Automatic pass produced no visual selections")
        figure, axes = plt.subplots(
            len(selections), 7, figsize=(16, 2.4 * len(selections)), squeeze=False
        )
        for row_index, (category, position) in enumerate(selections):
            rgb = np.asarray(descriptors[condition]["roi_rgb"])[position]
            maps = np.asarray(descriptors[condition]["rank_maps"])[position]
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 0].set_title(
                f"{category} idx={int(selected_indices[position])}\n"
                f"y={int(target[position])} s={float(candidate['score'][position]):.3f} "
                f"t={float(candidate['threshold'][position]):.3f}"
            )
            for channel_index, channel_name in enumerate(CHANNEL_NAMES):
                axes[row_index, 1 + channel_index].imshow(
                    maps[channel_index], cmap="viridis", vmin=0.0, vmax=1.0
                )
                axes[row_index, 1 + channel_index].set_title(channel_name)
            axes[row_index, 4].imshow(maps[0] <= 0.5, cmap="gray")
            axes[row_index, 4].set_title("sublevel a @ 0.5")
            axes[row_index, 5].imshow(maps[0] >= 0.5, cmap="gray")
            axes[row_index, 5].set_title("superlevel a @ 0.5")
            diagrams = _finite_intervals(maps[0])
            for dimension, color in ((0, "tab:blue"), (1, "tab:orange")):
                for interval in diagrams[dimension]:
                    axes[row_index, 6].plot(
                        interval,
                        [dimension, dimension],
                        color=color,
                        alpha=0.7,
                    )
            axes[row_index, 6].set_xlim(0.0, 1.0)
            axes[row_index, 6].set_ylim(-0.5, 1.5)
            axes[row_index, 6].set_yticks((0, 1))
            axes[row_index, 6].set_title("Lab-a H0/H1 bars")
            for axis in axes[row_index]:
                axis.tick_params(labelsize=6)
                if axis is not axes[row_index, 6]:
                    axis.axis("off")
            manifest.append(
                {
                    "condition": condition,
                    "category": category,
                    "cohort_position": int(position),
                    "sample_index": int(selected_indices[position]),
                    "target": int(target[position]),
                    "score": float(candidate["score"][position]),
                    "threshold": float(candidate["threshold"][position]),
                }
            )
        figure.tight_layout()
        path = output_dir / f"contact_sheet_{condition}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    manifest_path = output_dir / "visual_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "count": len(paths),
        "paths": [path.name for path in paths],
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not _tracked_worktree_clean(_repo_root()):
        raise ValueError("Tracked TRKH worktree must be clean for formal A0")
    head = _git_value(_repo_root(), "rev-parse", "HEAD")
    upstream = _git_value(_repo_root(), "rev-parse", "@{upstream}")
    if head != upstream:
        raise ValueError(f"Formal A0 requires pushed HEAD: {head} != {upstream}")
    provenance, conditions, selected_indices = _load_locked_inputs(args)
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

    synthetic = synthetic_oracle_checks()
    if not bool(synthetic["passed"]):
        raise RuntimeError("Synthetic cubical-persistence oracle failed")
    checkpoint = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    base_dataset, transform, class_names = _build_dataset(
        data_path=Path(args.data), checkpoint=checkpoint, condition_rows=conditions
    )
    dataset_paths = [Path(value) for value in base_dataset.sample_paths()]
    raw_before = _dataset_metadata_sha256(dataset_paths)
    descriptors: Dict[str, Dict[str, object]] = {}
    for condition, brightness, contrast in CONDITIONS:
        descriptors[condition] = _extract_condition_descriptors(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            base_dataset=base_dataset,
            transform=transform,
            checkpoint=checkpoint,
            selected_indices=selected_indices,
            expected=conditions[condition],
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            descriptor_threads=int(args.descriptor_threads),
        )
    outputs, diagnostics = _fit_nested_oof(
        descriptors=descriptors,
        conditions=conditions,
        selected_indices=selected_indices,
    )
    results = _condition_results(
        outputs=outputs, conditions=conditions, selected_indices=selected_indices
    )
    raw_after = _dataset_metadata_sha256(dataset_paths)
    defaults = _default_paths()
    command_hashes_after = {
        "current_best": _sha256(defaults["current_best"]),
        "command_history": _sha256(defaults["command_history"]),
    }
    descriptor_summary = _descriptor_summary(descriptors)
    structural = _structural_checks(
        provenance=provenance,
        descriptors=descriptors,
        diagnostics=diagnostics,
        synthetic=synthetic,
        raw_before=raw_before,
        raw_after=raw_after,
        command_hashes_after=command_hashes_after,
    )
    gate = assess_chromatic_persistence_a0(
        structural_checks=structural, results=results
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    descriptors_path = output_dir / "descriptor_payload.npz"
    diagnostics_path = output_dir / "readout_diagnostics.json"
    descriptor_summary_path = output_dir / "descriptor_diagnostics.json"
    _write_predictions(
        predictions_path,
        outputs=outputs,
        conditions=conditions,
        selected_indices=selected_indices,
    )
    np.savez_compressed(
        descriptors_path,
        condition_names=np.asarray([name for name, _, _ in CONDITIONS]),
        sample_index=selected_indices,
        target=np.asarray(conditions["clean"]["target"])[selected_indices],
        fold=np.asarray(conditions["clean"]["fold"])[selected_indices],
        source=np.asarray(conditions["clean"]["source"])[selected_indices].astype(str),
        global_color=np.stack(
            [np.asarray(descriptors[name]["global"]) for name, _, _ in CONDITIONS]
        ),
        topology=np.stack(
            [np.asarray(descriptors[name]["topology"]) for name, _, _ in CONDITIONS]
        ),
        pixel_topology=np.stack(
            [
                np.asarray(descriptors[name]["pixel_topology"])
                for name, _, _ in CONDITIONS
            ]
        ),
    )
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    descriptor_summary_path.write_text(
        json.dumps(descriptor_summary, indent=2), encoding="utf-8"
    )
    contact_sheets = {}
    if bool(gate["automatic_gates_passed"]):
        contact_sheets = _generate_contact_sheets(
            output_dir=output_dir,
            descriptors=descriptors,
            outputs=outputs,
            conditions=conditions,
            selected_indices=selected_indices,
        )
    status = (
        "awaiting_visual_review"
        if gate["automatic_gates_passed"]
        else "rejected_before_visual_review"
    )
    summary = {
        "mode": METHOD,
        "status": status,
        "repository": {"head": head, "upstream": upstream},
        "provenance": provenance,
        "protocol": {
            "seed": SEED,
            "fit_folds": list(FIT_FOLDS),
            "roi_size": ROI_SIZE,
            "erode_ratio": ERODE_RATIO,
            "ellipse_radius": ELLIPSE_RADIUS,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "persistence_floor": PERSISTENCE_FLOOR,
            "betti_thresholds": BETTI_THRESHOLDS.tolist(),
            "channel_names": list(CHANNEL_NAMES),
            "polarity_names": list(POLARITY_NAMES),
            "homology_dimensions": list(HOMOLOGY_DIMENSIONS),
            "topology_dim": TOPOLOGY_DIM,
            "role_dim": ROLE_DIM,
            "readout_c": READOUT_C,
            "target_retention": TARGET_RETENTION,
            "max_iterations": MAX_ITERATIONS,
            "conditions": [
                {"name": name, "brightness": brightness, "contrast": contrast}
                for name, brightness, contrast in CONDITIONS
            ],
        },
        "dataset": {
            "split": "train",
            "cohort_rows": EXPECTED_COHORT_ROWS,
            "class_names": class_names,
            "raw_metadata_sha256_before": raw_before,
            "raw_metadata_sha256_after": raw_after,
            "validation_data_used": False,
            "test_data_used": False,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "descriptor_threads": int(args.descriptor_threads),
        },
        "synthetic_oracle": synthetic,
        "descriptor": descriptor_summary,
        "readout": {
            "fit_count": diagnostics["fit_count"],
            "all_converged": diagnostics["all_converged"],
            "maximum_iterations": diagnostics["maximum_iterations"],
            "source_mapping_safe": diagnostics["source_mapping_safe"],
        },
        "results": results,
        "gate": gate,
        "contact_sheets": contact_sheets,
        "visual_review": None,
        "artifacts": {
            "predictions": predictions_path.name,
            "descriptor_payload": descriptors_path.name,
            "readout_diagnostics": diagnostics_path.name,
            "descriptor_diagnostics": descriptor_summary_path.name,
            "artifact_manifest": "artifact_manifest.json",
        },
        "guardrails": {
            "raw_dataset_touched": False,
            "validation_data_used": False,
            "test_data_used": False,
            "model_or_checkpoint_written": False,
            "current_best_commands_modified": False,
            "trainer_integration_authorized": False,
            "full_train_authorized": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_artifact_manifest(
        output_dir,
        mode="chromatic_cubical_persistence_a0_nonbinary_evidence_manifest",
    )
    return summary


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for row in manifest.get("files", []):
        payload = output_dir / str(row["name"])
        if not payload.is_file():
            raise FileNotFoundError(f"Manifest payload is missing: {payload}")
        if int(payload.stat().st_size) != int(row["size_bytes"]):
            raise ValueError(f"Manifest size mismatch: {payload}")
        if _sha256(payload) != str(row["sha256"]):
            raise ValueError(f"Manifest hash mismatch: {payload}")
    aggregate = hashlib.sha256()
    for row in manifest.get("files", []):
        aggregate.update(str(row["name"]).encode("utf-8"))
        aggregate.update(str(row["sha256"]).encode("ascii"))
    if aggregate.hexdigest() != str(manifest["payload_manifest_sha256"]):
        raise ValueError("Artifact manifest aggregate hash differs")
    return manifest


def _read_prediction_artifact(
    path: Path,
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, np.ndarray]],
    np.ndarray,
]:
    rows_by_condition = {name: [] for name, _, _ in CONDITIONS}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_condition[str(row["condition"])].append(row)
    outputs = {}
    condition_rows = {}
    selected_indices = None
    for condition, _, _ in CONDITIONS:
        rows = rows_by_condition[condition]
        if len(rows) != EXPECTED_COHORT_ROWS:
            raise ValueError(f"Replay row count differs for {condition}")
        positions = np.asarray([int(row["cohort_position"]) for row in rows])
        if not np.array_equal(positions, np.arange(EXPECTED_COHORT_ROWS)):
            raise ValueError("Replay cohort positions are noncanonical")
        indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        if selected_indices is None:
            selected_indices = indices
        elif not np.array_equal(indices, selected_indices):
            raise ValueError("Replay sample indices differ across conditions")
        condition_rows[condition] = {
            "target": np.asarray([int(row["target_index"]) for row in rows]),
            "fold": np.asarray([int(row["fold"]) for row in rows]),
            "source": np.asarray([str(row["source_stem"]) for row in rows]),
            "clean_keeper_prediction": np.asarray(
                [int(row["clean_keeper_prediction"]) for row in rows]
            ),
            "condition_keeper_prediction": np.asarray(
                [int(row["condition_keeper_prediction"]) for row in rows]
            ),
            "probabilities": np.asarray(
                [
                    [float(row[f"keeper_prob_{index}"]) for index in range(5)]
                    for row in rows
                ],
                dtype=np.float64,
            ),
        }
        outputs[condition] = {}
        for role in ROLES:
            outputs[condition][role] = {
                "score": np.asarray([float(row[f"{role}_score"]) for row in rows]),
                "threshold": np.asarray(
                    [float(row[f"{role}_threshold"]) for row in rows]
                ),
                "accepted": np.asarray(
                    [bool(int(row[f"{role}_accepted"])) for row in rows]
                ),
                "action_prediction": np.asarray(
                    [int(row[f"{role}_action_prediction"]) for row in rows]
                ),
            }
    assert selected_indices is not None
    return outputs, condition_rows, selected_indices


def _predict_saved_readout(
    record: Mapping[str, object], features: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(record["scaler_mean"], dtype=np.float64)
    scale = np.asarray(record["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(record["coef"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ROLE_DIM:
        raise ValueError("Replay readout features have invalid dimensions")
    if mean.shape != (ROLE_DIM,) or scale.shape != (ROLE_DIM,):
        raise ValueError("Replay scaler payload has invalid dimensions")
    if coefficient.shape != (ROLE_DIM,) or np.any(scale <= 0.0):
        raise ValueError("Replay coefficient payload is invalid")
    logits = ((values - mean) / scale) @ coefficient + float(record["intercept"])
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Replay readout produced non-finite probabilities")
    return probabilities.astype(np.float64)


def _validate_replay_readouts(
    *,
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    rows: Mapping[str, Mapping[str, np.ndarray]],
    payload: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    folds = np.asarray(rows["clean"]["fold"], dtype=np.int64)
    targets = np.asarray(rows["clean"]["target"], dtype=np.int64)
    sources = np.asarray(rows["clean"]["source"], dtype=object)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    positions = np.arange(EXPECTED_COHORT_ROWS, dtype=np.int64)

    recorded_mappings = {
        int(key): np.asarray(value, dtype=np.int64)
        for key, value in diagnostics["source_mappings"].items()
    }
    expected_mappings = _source_mappings(folds, sources)
    for outer_fold in FIT_FOLDS:
        recorded = recorded_mappings.get(int(outer_fold))
        if recorded is None or not np.array_equal(
            recorded, expected_mappings[int(outer_fold)]
        ):
            raise ValueError("Replay source-placebo mapping differs")
        held = positions[folds == outer_fold]
        fit = positions[folds != outer_fold]
        if set(recorded[held].tolist()) != set(held.tolist()):
            raise ValueError("Replay source mapping crosses the held partition")
        if set(recorded[fit].tolist()) != set(fit.tolist()):
            raise ValueError("Replay source mapping crosses the fit partition")
        if np.any(sources[recorded] == sources):
            raise ValueError("Replay source mapping retained a same-source row")

    fit_records = list(diagnostics["fit_records"])
    if len(fit_records) != 64 or int(diagnostics["fit_count"]) != 64:
        raise ValueError("Replay readout fit count differs")
    if not all(bool(record["converged"]) for record in fit_records):
        raise ValueError("Replay contains a nonconverged readout")
    fit_lookup: Dict[tuple[str, int, str], Mapping[str, object]] = {}
    for record in fit_records:
        key = (
            str(record["role"]),
            int(record["outer_fold"]),
            str(record["fit_kind"]),
        )
        if key in fit_lookup:
            raise ValueError(f"Replay contains a duplicate readout record: {key}")
        fit_lookup[key] = record

    threshold_records = list(diagnostics["threshold_records"])
    threshold_lookup: Dict[tuple[str, int], Mapping[str, object]] = {}
    for record in threshold_records:
        key = (str(record["role"]), int(record["outer_fold"]))
        if key in threshold_lookup:
            raise ValueError(f"Replay contains a duplicate threshold record: {key}")
        threshold_lookup[key] = record
    if len(threshold_lookup) != len(ROLES) * len(FIT_FOLDS):
        raise ValueError("Replay threshold record count differs")

    condition_names = [str(value) for value in payload["condition_names"].tolist()]
    expected_names = [name for name, _brightness, _contrast in CONDITIONS]
    if condition_names != expected_names:
        raise ValueError("Replay descriptor condition order differs")

    maximum_score_difference = 0.0
    maximum_threshold_difference = 0.0
    maximum_inner_record_difference = 0.0
    for role in ROLES:
        for outer_fold in FIT_FOLDS:
            held = positions[folds == outer_fold]
            outer_fit = positions[folds != outer_fold]
            source_mapping = (
                recorded_mappings[outer_fold] if role == "source_placebo" else None
            )
            clean_features = assemble_role_features(
                probabilities=np.asarray(rows["clean"]["probabilities"]),
                global_color=np.asarray(payload["global_color"])[0],
                topology=np.asarray(payload["topology"])[0],
                pixel_topology=np.asarray(payload["pixel_topology"])[0],
                role=role,
                source_mapping=source_mapping,
            )
            inner_scores = np.full(EXPECTED_COHORT_ROWS, np.nan, dtype=np.float64)
            for inner_fold in FIT_FOLDS:
                if inner_fold == outer_fold:
                    continue
                inner_held = positions[folds == inner_fold]
                key = (role, outer_fold, f"inner_holdout_{inner_fold}")
                if key not in fit_lookup:
                    raise ValueError(f"Replay is missing inner readout: {key}")
                inner_scores[inner_held] = _predict_saved_readout(
                    fit_lookup[key], clean_features[inner_held]
                )
            if not np.isfinite(inner_scores[outer_fit]).all():
                raise RuntimeError("Replay inner OOF scores are incomplete")
            threshold, threshold_record = select_retention_threshold(
                inner_scores[outer_fit], labels[outer_fit]
            )
            recorded_threshold = threshold_lookup[(role, outer_fold)]
            inner_differences = (
                abs(float(recorded_threshold["threshold"]) - threshold),
                abs(
                    float(recorded_threshold["inner_positive_retention"])
                    - float(threshold_record["inner_positive_retention"])
                ),
                abs(
                    float(recorded_threshold["inner_restricted_fp_rejection"])
                    - float(threshold_record["inner_restricted_fp_rejection"])
                ),
            )
            maximum_inner_record_difference = max(
                maximum_inner_record_difference, *inner_differences
            )
            if int(recorded_threshold["candidate_count"]) != int(
                threshold_record["candidate_count"]
            ):
                raise ValueError("Replay inner threshold candidate count differs")

            outer_key = (role, outer_fold, "outer")
            if outer_key not in fit_lookup:
                raise ValueError(f"Replay is missing outer readout: {outer_key}")
            for condition_index, (condition, _brightness, _contrast) in enumerate(
                CONDITIONS
            ):
                features = assemble_role_features(
                    probabilities=np.asarray(rows[condition]["probabilities"]),
                    global_color=np.asarray(payload["global_color"])[condition_index],
                    topology=np.asarray(payload["topology"])[condition_index],
                    pixel_topology=np.asarray(payload["pixel_topology"])[
                        condition_index
                    ],
                    role=role,
                    source_mapping=source_mapping,
                )
                reconstructed = _predict_saved_readout(
                    fit_lookup[outer_key], features[held]
                )
                observed = np.asarray(outputs[condition][role]["score"])[held]
                maximum_score_difference = max(
                    maximum_score_difference,
                    float(np.max(np.abs(reconstructed - observed))),
                )
                observed_thresholds = np.asarray(
                    outputs[condition][role]["threshold"]
                )[held]
                maximum_threshold_difference = max(
                    maximum_threshold_difference,
                    float(np.max(np.abs(observed_thresholds - threshold))),
                )
                reconstructed_accepted = reconstructed >= threshold
                observed_accepted = np.asarray(
                    outputs[condition][role]["accepted"], dtype=bool
                )[held]
                if not np.array_equal(reconstructed_accepted, observed_accepted):
                    raise ValueError("Replay accepted decisions differ from readouts")

    passed = bool(
        maximum_score_difference <= 1e-12
        and maximum_threshold_difference <= 1e-12
        and maximum_inner_record_difference <= 1e-12
    )
    return {
        "fit_count": len(fit_lookup),
        "threshold_count": len(threshold_lookup),
        "source_mapping_safe": True,
        "maximum_score_difference": maximum_score_difference,
        "maximum_threshold_difference": maximum_threshold_difference,
        "maximum_inner_record_difference": maximum_inner_record_difference,
        "passed": passed,
    }


def _replay_results(
    *,
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    rows: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    target = np.asarray(rows["clean"]["target"], dtype=np.int64)
    fold = np.asarray(rows["clean"]["fold"], dtype=np.int64)
    clean_prediction = np.asarray(
        rows["clean"]["clean_keeper_prediction"], dtype=np.int64
    )
    labels = (target == FOCUS_CLASS).astype(np.int64)
    tp = (target == FOCUS_CLASS) & (clean_prediction == FOCUS_CLASS)
    fn = (target == FOCUS_CLASS) & (clean_prediction != FOCUS_CLASS)
    fp = np.isin(target, RESTRICTED_NEGATIVE_CLASSES) & (
        clean_prediction == FOCUS_CLASS
    )
    clean_candidate = np.asarray(outputs["clean"]["candidate"]["score"])
    results = {}
    for condition, _, _ in CONDITIONS:
        results[condition] = {}
        probabilities = np.asarray(rows[condition]["probabilities"], dtype=np.float64)
        base_prediction = probabilities.argmax(axis=1).astype(np.int64)
        if not np.array_equal(
            base_prediction,
            np.asarray(rows[condition]["condition_keeper_prediction"], dtype=np.int64),
        ):
            raise ValueError("Replay keeper argmax differs from the recorded prediction")
        for role in ROLES:
            output = outputs[condition][role]
            scores = np.asarray(output["score"], dtype=np.float64)
            thresholds = np.asarray(output["threshold"], dtype=np.float64)
            accepted = np.asarray(output["accepted"], dtype=bool)
            _base, action = _action_predictions(probabilities, accepted)
            if not np.array_equal(
                action, np.asarray(output["action_prediction"], dtype=np.int64)
            ):
                raise ValueError("Replay action prediction differs from the artifact")
            corrections = (base_prediction != target) & (action == target)
            harms = (base_prediction == target) & (action != target)
            per_fold = {}
            for value in FIT_FOLDS:
                mask = fold == value
                per_fold[str(value)] = {
                    "tp_rows": int((mask & tp).sum()),
                    "tp_breaks": int((mask & tp & ~accepted).sum()),
                    "restricted_fp_rows": int((mask & fp).sum()),
                    "restricted_fp_removals": int((mask & fp & ~accepted).sum()),
                }
            correlation = 1.0
            if condition != "clean" and role == "candidate":
                correlation = float(spearmanr(clean_candidate, scores).statistic)
            results[condition][role] = {
                "condition": condition,
                "role": role,
                "auc": float(roc_auc_score(labels, scores)),
                "tp_rows": int(tp.sum()),
                "tp_retained": int((tp & accepted).sum()),
                "tp_breaks": int((tp & ~accepted).sum()),
                "tp_retention": float(accepted[tp].mean()),
                "fn_rows": int(fn.sum()),
                "fn_accepted": int((fn & accepted).sum()),
                "fn_acceptance": float(accepted[fn].mean()),
                "restricted_fp_rows": int(fp.sum()),
                "restricted_fp_removals": int((fp & ~accepted).sum()),
                "restricted_fp_rejection": float((~accepted[fp]).mean()),
                "corrections": int(corrections.sum()),
                "harms": int(harms.sum()),
                "changed": int((base_prediction != action).sum()),
                "score_minimum": float(scores.min()),
                "score_mean": float(scores.mean()),
                "score_maximum": float(scores.max()),
                "threshold_minimum": float(thresholds.min()),
                "threshold_mean": float(thresholds.mean()),
                "threshold_maximum": float(thresholds.max()),
                "clean_score_spearman": correlation,
                "per_fold": per_fold,
            }
    return results


def _maximum_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError("Replay mapping keys differ")
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Replay list lengths differ")
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        if bool(left) != bool(right):
            raise ValueError("Replay boolean values differ")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError(f"Replay scalar values differ: {left!r} != {right!r}")
    return 0.0


def replay_summary(summary_path: Path) -> Dict[str, object]:
    path = Path(summary_path).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("mode") != METHOD:
        raise ValueError("Replay summary mode differs")
    output_dir = path.parent
    manifest = _verify_manifest(output_dir)
    outputs, rows, selected_indices = _read_prediction_artifact(
        output_dir / str(summary["artifacts"]["predictions"])
    )
    if _ordered_index_sha256(selected_indices.tolist()) != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError("Replay cohort hash differs")
    payload_path = output_dir / str(summary["artifacts"]["descriptor_payload"])
    with np.load(payload_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    expected_payload_keys = {
        "condition_names",
        "sample_index",
        "target",
        "fold",
        "source",
        "global_color",
        "topology",
        "pixel_topology",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError("Replay descriptor payload keys differ")
    if payload["global_color"].shape != (
        len(CONDITIONS),
        EXPECTED_COHORT_ROWS,
        GLOBAL_COLOR_DIM,
    ):
        raise ValueError("Replay global-color payload shape differs")
    if payload["topology"].shape != (
        len(CONDITIONS),
        EXPECTED_COHORT_ROWS,
        TOPOLOGY_DIM,
    ):
        raise ValueError("Replay topology payload shape differs")
    if payload["pixel_topology"].shape != payload["topology"].shape:
        raise ValueError("Replay pixel-topology payload shape differs")
    for key in ("global_color", "topology", "pixel_topology"):
        if not np.isfinite(payload[key]).all():
            raise ValueError(f"Replay {key} payload is non-finite")
    if not np.array_equal(payload["sample_index"], selected_indices):
        raise ValueError("Replay payload sample indices differ from CSV")
    for key in ("target", "fold", "source"):
        if not np.array_equal(payload[key].astype(str), rows["clean"][key].astype(str)):
            raise ValueError(f"Replay payload {key} differs from CSV")
    for condition, _brightness, _contrast in CONDITIONS:
        for key in ("target", "fold", "source"):
            if not np.array_equal(
                rows[condition][key].astype(str), rows["clean"][key].astype(str)
            ):
                raise ValueError(f"Replay {condition} metadata differs from clean")
        clean_prediction = np.asarray(rows[condition]["clean_keeper_prediction"])
        expected_clean_prediction = np.asarray(
            rows["clean"]["probabilities"]
        ).argmax(axis=1)
        if not np.array_equal(clean_prediction, expected_clean_prediction):
            raise ValueError("Replay clean keeper prediction differs from probabilities")
    diagnostics = json.loads(
        (output_dir / str(summary["artifacts"]["readout_diagnostics"])).read_text(
            encoding="utf-8"
        )
    )
    if int(diagnostics["fit_count"]) != 64 or not bool(diagnostics["all_converged"]):
        raise ValueError("Replay readout convergence differs")
    readout_replay = _validate_replay_readouts(
        outputs=outputs,
        rows=rows,
        payload=payload,
        diagnostics=diagnostics,
    )
    replayed_results = _replay_results(outputs=outputs, rows=rows)
    difference = _maximum_numeric_difference(summary["results"], replayed_results)
    replayed_gate = assess_chromatic_persistence_a0(
        structural_checks=summary["gate"]["structural_checks"],
        results=replayed_results,
    )
    gate_difference = _maximum_numeric_difference(summary["gate"], replayed_gate)
    result = {
        "mode": "chromatic_cubical_persistence_a0_replay",
        "summary": str(path),
        "summary_sha256": _sha256(path),
        "maximum_result_difference": difference,
        "maximum_gate_difference": gate_difference,
        "readout_replay": readout_replay,
        "payload_manifest_sha256": manifest["payload_manifest_sha256"],
        "passed": bool(
            difference <= 1e-12
            and gate_difference <= 1e-12
            and readout_replay["passed"]
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"Independent replay failed: {result}")
    return result


def finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    expected = str(args.expected_summary_sha256).strip().casefold()
    if not expected or _sha256(summary_path) != expected:
        raise ValueError("Finalize summary hash differs from the locked input")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError("Visual review is not authorized for this summary")
    sheets = summary.get("contact_sheets", {})
    if int(sheets.get("count", 0)) != 4:
        raise ValueError("Finalize requires all four contact sheets")
    note = str(args.visual_review_note).strip()
    if not note:
        raise ValueError("Finalize requires a nonempty review note")
    result = str(args.visual_review_result)
    summary["visual_review"] = {
        "result": result,
        "note": note,
        "reviewed_summary_sha256": expected,
        "contact_sheet_manifest_sha256": sheets["manifest_sha256"],
    }
    summary["status"] = (
        "passed_information_gate" if result == "pass" else "rejected_visual_review"
    )
    summary["guardrails"]["trainer_integration_authorized"] = result == "pass"
    summary["guardrails"]["full_train_authorized"] = False
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_artifact_manifest(
        output_dir,
        mode="chromatic_cubical_persistence_a0_nonbinary_evidence_manifest",
    )
    return summary


def _preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_provenance(args)
    synthetic = synthetic_oracle_checks()
    if not synthetic["passed"]:
        raise RuntimeError("Synthetic cubical-persistence oracle failed")
    return {
        "mode": "chromatic_cubical_persistence_a0_preflight",
        "provenance": provenance,
        "synthetic_oracle": synthetic,
        "dataset_loaded": False,
        "model_loaded": False,
        "output_created": False,
        "validation_data_used": False,
        "test_data_used": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    modes = sum(
        (
            bool(args.preflight_only),
            args.replay_summary is not None,
            bool(args.finalize_visual_review),
        )
    )
    if modes > 1:
        raise ValueError("Choose only one preflight/replay/finalize mode")
    if args.preflight_only:
        print(json.dumps(_preflight(args), indent=2))
        return 0
    if args.replay_summary is not None:
        print(json.dumps(replay_summary(args.replay_summary), indent=2))
        return 0
    if args.finalize_visual_review:
        if args.visual_review_result is None:
            raise ValueError("Finalize requires --visual-review-result")
        print(json.dumps(finalize_visual_review(args), indent=2))
        return 0
    summary = run_audit(args)
    print(json.dumps({"status": summary["status"], "gate": summary["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

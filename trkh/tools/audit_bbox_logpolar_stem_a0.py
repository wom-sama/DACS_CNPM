from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import IdentityCorruption, LightingShift
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
    _classification_metrics,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


SEED = 20260720
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLDS = 5
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
CONDITIONS = (
    ("clean", 1.0, 1.0),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
ROLES = ("cartesian", "candidate", "linear", "source_placebo")
EXTRACTED_ROLES = ("cartesian", "candidate", "linear")
IMAGE_SIZE = 256
STEM_SIZE = 32
STEM_CHANNELS = 256
SUPPORT_SCALE = 1.10
LOG_RADIUS = 128.0
RADIAL_BANDS = 4
DESCRIPTOR_DIM = RADIAL_BANDS * 2 * STEM_CHANNELS
READOUT_C = 0.30
MAX_ITERATIONS = 2000
VISUAL_SAMPLE_INDICES = (1, 2, 3, 46, 0)
PROBABILITY_EPSILON = 1e-7

OFFICIAL_REPOSITORIES = {
    "logpolar_repo": {
        "path": Path(
            "D:/DataAI/external_sources/official/log-polar-descriptors-iccv2019"
        ),
        "commit": "45d0a922dcd58e6e64e6cba158e3dad3f62ccfe8",
        "tree": "527738e303157dee4c60997ed73ea3aeb7d20091",
    },
    "ptn_repo": {
        "path": Path(
            "D:/DataAI/external_sources/official/polar-transformer-networks-iclr2018"
        ),
        "commit": "c6a4ad613bb2feb7a5ba8b25307449bde931b7a3",
        "tree": "a0b328bb45804625cfec436a169b3c92ceaa0cdc",
    },
}

EXPECTED_HASHES = {
    "protocol": "dc04f38ea27f813dc9f1866e8bc9a325ca834475ae7c2b0a9ae1c45c60896f76",
    "logpolar_paper": "19718844bc02faaf20f89e595cb81af46cfa221f96a3726e5bc67908f1418e21",
    "logpolar_license": "3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5",
    "ptn_paper": "a49644afa3676cf2eb1219892b327fc26f4ac014565821d9ddee4e458d52e304",
    "ptn_license": "2d212fb778efe3c9a60e3d54f8718dedaf86909ac1a06fd8b773dcfcc8d205d3",
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
        / "TRKH_5CLASS_BBOX_LOGPOLAR_STEM_A0_PROTOCOL_20260720.md",
        "logpolar_paper": Path(
            "D:/DataAI/external_sources/papers/beyond_cartesian_logpolar_iccv2019.pdf"
        ),
        "logpolar_license": OFFICIAL_REPOSITORIES["logpolar_repo"]["path"]
        / "LICENSE",
        "ptn_paper": Path(
            "D:/DataAI/external_sources/papers/polar_transformer_networks_iclr2018.pdf"
        ),
        "ptn_license": OFFICIAL_REPOSITORIES["ptn_repo"]["path"] / "LICENSE",
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
            "Locked train-only bbox-centered log-polar stem A0. Validation and "
            "test are forbidden."
        )
    )
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--checkpoint", type=Path, default=defaults["keeper"])
    parser.add_argument("--cidt-summary", type=Path, default=defaults["cidt_summary"])
    parser.add_argument(
        "--cidt-predictions", type=Path, default=defaults["cidt_predictions"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_bbox_logpolar_stem_a0_20260720"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_provenance(args: argparse.Namespace) -> Dict[str, object]:
    defaults = _default_paths()
    paths = {
        **defaults,
        "data": Path(args.data),
        "cidt_summary": Path(args.cidt_summary),
        "cidt_predictions": Path(args.cidt_predictions),
        "keeper": Path(args.checkpoint),
    }
    files: Dict[str, object] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = Path(paths[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked provenance file: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"Locked provenance mismatch for {name}: {observed} != {expected}"
            )
        files[name] = {
            "path": str(path),
            "sha256": observed,
            "matched": True,
        }

    repositories: Dict[str, object] = {}
    for name, specification in OFFICIAL_REPOSITORIES.items():
        path = Path(specification["path"]).resolve()
        if not (path / ".git").exists():
            raise FileNotFoundError(f"Missing official Git checkout: {path}")
        commit = _git_value(path, "rev-parse", "HEAD")
        tree = _git_value(path, "rev-parse", "HEAD^{tree}")
        status = _git_value(path, "status", "--porcelain")
        matched = (
            commit == specification["commit"]
            and tree == specification["tree"]
            and not status
        )
        if not matched:
            raise ValueError(
                f"Official source mismatch for {name}: commit={commit} tree={tree} "
                f"dirty={bool(status)}"
            )
        repositories[name] = {
            "path": str(path),
            "commit": commit,
            "tree": tree,
            "clean": True,
            "matched": True,
        }
    return {"files": files, "repositories": repositories, "all_exact": True}


def _validate_locked_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) != 64:
        raise ValueError("Locked A0 requires batch-size=64")
    if int(args.num_workers) != 4:
        raise ValueError("Locked A0 requires num-workers=4")
    if int(args.torch_threads) != 8:
        raise ValueError("Locked A0 requires torch-threads=8")
    output = Path(args.output_dir).resolve()
    dataset_root = Path(args.data).resolve().parent
    try:
        output.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must remain outside the raw dataset")


def _resolve_device(value: str) -> torch.device:
    requested = str(value)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _dataset_tree_identity(data_path: Path) -> str:
    root = Path(data_path).resolve().parent
    paths = [Path(data_path).resolve()]
    for relative in (Path("images/train"), Path("labels/train")):
        folder = root / relative
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing locked dataset directory: {folder}")
        paths.extend(path for path in folder.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value).casefold()):
        stat = path.stat()
        digest.update(str(path).replace("/", "\\").casefold().encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def build_bbox_support_grid(
    crop_bbox: Tensor,
    *,
    output_size: int = IMAGE_SIZE,
    support_scale: float = SUPPORT_SCALE,
) -> Tensor:
    if crop_bbox.ndim != 2 or int(crop_bbox.size(1)) != 4:
        raise ValueError("crop_bbox must have shape [B,4]")
    if output_size <= 0 or support_scale <= 0.0:
        raise ValueError("output_size and support_scale must be positive")
    dtype = crop_bbox.dtype
    device = crop_bbox.device
    coordinates = (
        (torch.arange(output_size, device=device, dtype=dtype) + 0.5)
        / float(output_size)
        * 2.0
        - 1.0
    )
    target_y, target_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    center_x = crop_bbox[:, 0].view(-1, 1, 1) * 2.0 - 1.0
    center_y = crop_bbox[:, 1].view(-1, 1, 1) * 2.0 - 1.0
    span_x = crop_bbox[:, 2].view(-1, 1, 1) * float(support_scale)
    span_y = crop_bbox[:, 3].view(-1, 1, 1) * float(support_scale)
    source_x = center_x + target_x.unsqueeze(0) * span_x
    source_y = center_y + target_y.unsqueeze(0) * span_y
    return torch.stack((source_x, source_y), dim=-1)


def sample_bbox_support(
    images: Tensor,
    crop_bbox: Tensor,
    *,
    output_size: int = IMAGE_SIZE,
    support_scale: float = SUPPORT_SCALE,
) -> Tensor:
    grid = build_bbox_support_grid(
        crop_bbox, output_size=output_size, support_scale=support_scale
    )
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def build_polar_grid(
    batch_size: int,
    *,
    output_size: int,
    mode: str,
    radius: float = LOG_RADIUS,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if batch_size <= 0 or output_size <= 0 or radius <= 0.0:
        raise ValueError("batch_size, output_size, and radius must be positive")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"log", "linear"}:
        raise ValueError("mode must be log or linear")
    angle_fraction = (
        torch.arange(output_size, device=device, dtype=dtype) + 0.5
    ) / float(output_size)
    radial_fraction = (
        torch.arange(output_size, device=device, dtype=dtype) + 0.5
    ) / float(output_size)
    theta = angle_fraction * (2.0 * math.pi) - math.pi
    if normalized_mode == "log":
        rho = torch.expm1(radial_fraction * math.log1p(float(radius))) / float(radius)
    else:
        rho = radial_fraction
    source_x = torch.cos(theta).view(-1, 1) * rho.view(1, -1)
    source_y = torch.sin(theta).view(-1, 1) * rho.view(1, -1)
    grid = torch.stack((source_x, source_y), dim=-1)
    return grid.unsqueeze(0).expand(int(batch_size), -1, -1, -1).contiguous()


def polar_sample(values: Tensor, *, mode: str, output_size: int) -> Tensor:
    if values.ndim != 4:
        raise ValueError("polar_sample expects [B,C,H,W]")
    grid = build_polar_grid(
        int(values.size(0)),
        output_size=int(output_size),
        mode=mode,
        device=values.device,
        dtype=values.dtype,
    )
    return F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def periodic_stem_block_forward(block: nn.Module, values: Tensor) -> Tensor:
    modules = getattr(block, "block", None)
    if modules is None:
        raise TypeError("Periodic stem requires ConvStemBlock.block")
    conv = getattr(modules, "conv", None)
    norm = getattr(modules, "norm", None)
    activation = getattr(modules, "act", None)
    pooling = getattr(modules, "pool", None)
    if not isinstance(conv, nn.Conv2d) or any(
        module is None for module in (norm, activation, pooling)
    ):
        raise TypeError("Unexpected ConvStemBlock structure")
    if conv.kernel_size != (3, 3) or conv.padding != (1, 1):
        raise ValueError("Locked periodic path requires 3x3 padding-1 convolutions")
    padded = F.pad(values, (1, 1, 0, 0), mode="constant", value=0.0)
    padded = F.pad(padded, (0, 0, 1, 1), mode="circular")
    output = F.conv2d(
        padded,
        conv.weight,
        conv.bias,
        stride=conv.stride,
        padding=0,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    return pooling(activation(norm(output)))


def periodic_stem_forward(stem: nn.Module, values: Tensor) -> Tensor:
    blocks = getattr(stem, "blocks", None)
    if blocks is None or len(blocks) != 3:
        raise TypeError("Locked periodic path requires the three-block HybridConvStem")
    output = values
    for block in blocks:
        output = periodic_stem_block_forward(block, output)
    return output


def radial_band_descriptor(features: Tensor) -> Tensor:
    if features.ndim != 4:
        raise ValueError("Stem features must have shape [B,C,H,W]")
    if tuple(features.shape[1:]) != (STEM_CHANNELS, STEM_SIZE, STEM_SIZE):
        raise ValueError(
            f"Locked stem shape is [B,{STEM_CHANNELS},{STEM_SIZE},{STEM_SIZE}], "
            f"got {tuple(features.shape)}"
        )
    if STEM_SIZE % RADIAL_BANDS:
        raise RuntimeError("Stem radius is not divisible by radial bands")
    width = STEM_SIZE // RADIAL_BANDS
    chunks = []
    for band in range(RADIAL_BANDS):
        values = features[:, :, :, band * width : (band + 1) * width].float()
        chunks.append(values.mean(dim=(2, 3)))
        chunks.append(values.std(dim=(2, 3), unbiased=False))
    descriptor = torch.cat(chunks, dim=1)
    if int(descriptor.size(1)) != DESCRIPTOR_DIM:
        raise RuntimeError("Descriptor dimension violates the locked protocol")
    return descriptor


def _read_cidt_conditions(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    required = {
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target_index",
        "keeper_prob_0",
        "keeper_prob_1",
        "keeper_prob_2",
        "keeper_prob_3",
        "keeper_prob_4",
    }
    grouped: Dict[str, list[Dict[str, str]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("CIDT predictions lack locked columns")
        for row in reader:
            grouped.setdefault(str(row["condition"]), []).append(row)

    expected_conditions = {name for name, _brightness, _contrast in CONDITIONS}
    if set(grouped) != expected_conditions:
        raise ValueError(f"CIDT condition mismatch: {sorted(grouped)}")
    result: Dict[str, Dict[str, np.ndarray]] = {}
    clean_metadata = None
    for condition, _brightness, _contrast in CONDITIONS:
        rows = grouped[condition]
        if len(rows) != EXPECTED_TRAIN_ROWS:
            raise ValueError(f"CIDT {condition} row count mismatch")
        sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        if not np.array_equal(sample_index, np.arange(EXPECTED_TRAIN_ROWS)):
            raise ValueError(f"CIDT {condition} sample order mismatch")
        target = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
        fold = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
        source = np.asarray(
            [str(row["source_stem"]).casefold() for row in rows], dtype=object
        )
        image_path = np.asarray([str(Path(row["image_path"]).resolve()) for row in rows])
        probabilities = np.asarray(
            [
                [float(row[f"keeper_prob_{class_index}"]) for class_index in range(5)]
                for row in rows
            ],
            dtype=np.float64,
        )
        metadata = (target, fold, source, image_path)
        if clean_metadata is None:
            clean_metadata = metadata
            if np.bincount(target, minlength=5).tolist() != list(EXPECTED_CLASS_COUNTS):
                raise ValueError("CIDT class counts differ from the locked protocol")
            if np.bincount(fold, minlength=FOLDS).tolist() != list(EXPECTED_FOLD_COUNTS):
                raise ValueError("CIDT fold counts differ from the locked protocol")
        else:
            for observed, clean in zip(metadata, clean_metadata):
                if not np.array_equal(observed, clean):
                    raise ValueError(f"CIDT metadata drift in condition {condition}")
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-5
        ):
            raise ValueError(f"Invalid CIDT keeper probabilities for {condition}")
        result[condition] = {
            "sample_index": sample_index,
            "target": target,
            "fold": fold,
            "source": source,
            "image_path": image_path,
            "probabilities": probabilities,
        }
    return result


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
    if int(semantics["image_size"]) != IMAGE_SIZE:
        raise ValueError("Keeper image size differs from the locked protocol")
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("Log-polar A0 requires temporal_frames=1")
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
    if paths != [str(value) for value in condition_rows["clean"]["image_path"]]:
        raise ValueError("Dataset sample paths do not align with CIDT")
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
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Dict[str, object]]:
    dataset = _SelectedConditionDataset(
        base_dataset,
        list(range(EXPECTED_TRAIN_ROWS)),
        corruption=_condition_corruption(condition, brightness, contrast),
        transform=transform,
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=f"bbox_logpolar_{condition}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return loader, summary


def _activation_norm(features: Tensor) -> np.ndarray:
    values = torch.linalg.vector_norm(features.float(), dim=1)
    minimum = values.amin(dim=(1, 2), keepdim=True)
    maximum = values.amax(dim=(1, 2), keepdim=True)
    values = (values - minimum) / (maximum - minimum).clamp_min(1e-8)
    return values.cpu().numpy().astype(np.float32)


def _extract_condition_descriptors(
    *,
    condition: str,
    brightness: float,
    contrast: float,
    model: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    expected: Mapping[str, np.ndarray],
    checkpoint: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        condition=condition,
        brightness=brightness,
        contrast=contrast,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    descriptors = {role: [] for role in EXTRACTED_ROLES}
    observed_indices: list[int] = []
    observed_targets: list[int] = []
    visuals: Dict[int, Dict[str, np.ndarray]] = {}
    requested_visuals = set(VISUAL_SAMPLE_INDICES) if condition == "clean" else set()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stem = getattr(model, "stem", None)
    if stem is None:
        raise ValueError("Log-polar A0 requires the keeper CNN stem")
    with torch.inference_mode():
        for images, targets, metadata in tqdm(
            loader, desc=f"logpolar-{condition}", dynamic_ncols=True
        ):
            if not isinstance(metadata, Mapping):
                raise ValueError("Log-polar extraction requires metadata")
            crop_bbox = metadata.get("crop_bbox")
            sample_index = metadata.get("sample_index")
            if not torch.is_tensor(crop_bbox) or not torch.is_tensor(sample_index):
                raise ValueError("Missing crop_bbox/sample_index metadata")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
            rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            support = sample_bbox_support(rgb, crop_bbox)

            normalized_support = (support - mean_tensor) / std_tensor
            native_features = stem(normalized_support)
            cartesian_features = polar_sample(
                native_features, mode="log", output_size=STEM_SIZE
            )
            cartesian_descriptor = radial_band_descriptor(cartesian_features)

            log_rgb = polar_sample(support, mode="log", output_size=IMAGE_SIZE)
            normalized_log = (log_rgb - mean_tensor) / std_tensor
            candidate_features = periodic_stem_forward(stem, normalized_log)
            candidate_descriptor = radial_band_descriptor(candidate_features)

            linear_rgb = polar_sample(support, mode="linear", output_size=IMAGE_SIZE)
            normalized_linear = (linear_rgb - mean_tensor) / std_tensor
            linear_features = periodic_stem_forward(stem, normalized_linear)
            linear_descriptor = radial_band_descriptor(linear_features)

            batch_descriptors = {
                "cartesian": cartesian_descriptor,
                "candidate": candidate_descriptor,
                "linear": linear_descriptor,
            }
            for role, values in batch_descriptors.items():
                descriptors[role].append(values.cpu().numpy().astype(np.float32))

            indices = sample_index.cpu().numpy().astype(np.int64).tolist()
            observed_indices.extend(indices)
            observed_targets.extend(targets.cpu().numpy().astype(np.int64).tolist())
            selected_positions = [
                position
                for position, index in enumerate(indices)
                if index in requested_visuals
            ]
            if selected_positions:
                control_norm = _activation_norm(cartesian_features[selected_positions])
                candidate_norm = _activation_norm(candidate_features[selected_positions])
                support_images = (
                    support[selected_positions]
                    .mul(255.0)
                    .round()
                    .clamp(0.0, 255.0)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                linear_images = (
                    linear_rgb[selected_positions]
                    .mul(255.0)
                    .round()
                    .clamp(0.0, 255.0)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                log_images = (
                    log_rgb[selected_positions]
                    .mul(255.0)
                    .round()
                    .clamp(0.0, 255.0)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for local, position in enumerate(selected_positions):
                    index = int(indices[position])
                    visuals[index] = {
                        "support": support_images[local],
                        "linear": linear_images[local],
                        "log": log_images[local],
                        "cartesian_activation": control_norm[local],
                        "candidate_activation": candidate_norm[local],
                    }

    arrays = {
        role: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for role, values in descriptors.items()
    }
    for role, values in arrays.items():
        if values.shape != (EXPECTED_TRAIN_ROWS, DESCRIPTOR_DIM):
            raise RuntimeError(f"Incomplete {role} descriptor extraction: {values.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {role} descriptors")
    if observed_indices != list(range(EXPECTED_TRAIN_ROWS)):
        raise ValueError("Descriptor loader sample order differs from CIDT")
    if not np.array_equal(
        np.asarray(observed_targets, dtype=np.int64), expected["target"]
    ):
        raise ValueError("Descriptor targets differ from CIDT")
    return {
        **arrays,
        "visuals": visuals,
        "seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "loader": loader_summary,
    }


def source_derangement(
    indices: np.ndarray,
    source_stems: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    sources = np.asarray(source_stems, dtype=object)
    if rows.ndim != 1 or rows.size < 2:
        raise ValueError("Derangement requires at least two rows")
    original = sources[rows]
    rng = np.random.default_rng(int(seed))
    for _ in range(10000):
        candidate = rng.permutation(rows)
        if np.all(sources[candidate] != original):
            return candidate.astype(np.int64, copy=False)
    raise RuntimeError("Unable to construct a source-safe derangement")


def _fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale >= 1e-8, scale, 1.0)
    return mean, scale


def _apply_scaler(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(features, dtype=np.float64)
        - np.asarray(mean, dtype=np.float64)
    ) / np.asarray(scale, dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=1, keepdims=True)
    exponent = np.exp(values - maximum)
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_offset_readout(
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
    if x.ndim != 2 or x.shape[1] != DESCRIPTOR_DIM:
        raise ValueError("Readout descriptor dimension mismatch")
    if base.shape != (x.shape[0], 5) or y.shape != (x.shape[0],):
        raise ValueError("Readout arrays are misaligned")
    base_logits = np.log(np.clip(base, PROBABILITY_EPSILON, 1.0))
    rows, feature_dim = x.shape

    def objective(flat: np.ndarray):
        weights = flat.reshape(feature_dim, 5)
        logits = base_logits + x @ weights
        probabilities = _softmax(logits)
        maximum = logits.max(axis=1)
        log_normalizer = np.log(np.exp(logits - maximum[:, None]).sum(axis=1)) + maximum
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
        np.zeros(feature_dim * 5, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 50,
        },
    )
    weights = result.x.reshape(feature_dim, 5)
    if not np.isfinite(weights).all():
        raise RuntimeError("Readout weights are non-finite")
    return weights, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "weight_l2": float(np.linalg.norm(weights)),
        "feature_dim": int(feature_dim),
        "parameter_count": int(weights.size),
    }


def apply_offset_readout(
    features: np.ndarray,
    base_probabilities: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    scaled = _apply_scaler(features, mean, scale)
    base_logits = np.log(
        np.clip(np.asarray(base_probabilities, dtype=np.float64), PROBABILITY_EPSILON, 1.0)
    )
    return _softmax(base_logits + scaled @ np.asarray(weights, dtype=np.float64))


def fit_source_oof_readouts(
    *,
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    list[Dict[str, object]],
    Dict[str, object],
]:
    clean = condition_rows["clean"]
    targets = np.asarray(clean["target"], dtype=np.int64)
    folds = np.asarray(clean["fold"], dtype=np.int64)
    sources = np.asarray(clean["source"], dtype=object)
    outputs = {
        condition: {
            role: np.full((EXPECTED_TRAIN_ROWS, 5), np.nan, dtype=np.float64)
            for role in ROLES
        }
        for condition, _brightness, _contrast in CONDITIONS
    }
    parameters: Dict[str, np.ndarray] = {}
    fit_records: list[Dict[str, object]] = []
    derangement_records: list[Dict[str, object]] = []
    hold_map = np.full(EXPECTED_TRAIN_ROWS, -1, dtype=np.int64)

    for fold in range(FOLDS):
        fit_indices = np.flatnonzero(folds != fold)
        hold_indices = np.flatnonzero(folds == fold)
        fit_sources = set(sources[fit_indices].tolist())
        hold_sources = set(sources[hold_indices].tolist())
        overlap = fit_sources.intersection(hold_sources)
        if overlap:
            raise ValueError(f"CIDT fold {fold} has source overlap")
        fit_permutation = source_derangement(
            fit_indices, sources, seed=SEED + fold * 2
        )
        hold_permutation = source_derangement(
            hold_indices, sources, seed=SEED + fold * 2 + 1
        )
        hold_map[hold_indices] = hold_permutation
        parameters[f"fold{fold}__placebo_fit_indices"] = fit_permutation
        parameters[f"fold{fold}__placebo_hold_indices"] = hold_permutation
        derangement_records.append(
            {
                "fold": fold,
                "fit_rows": int(fit_indices.size),
                "hold_rows": int(hold_indices.size),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": len(overlap),
                "fit_same_source": int(
                    np.sum(sources[fit_indices] == sources[fit_permutation])
                ),
                "hold_same_source": int(
                    np.sum(sources[hold_indices] == sources[hold_permutation])
                ),
                "fit_permutation_sha256": hashlib.sha256(
                    fit_permutation.astype("<i8").tobytes()
                ).hexdigest(),
                "hold_permutation_sha256": hashlib.sha256(
                    hold_permutation.astype("<i8").tobytes()
                ).hexdigest(),
            }
        )

        for role in ROLES:
            if role == "source_placebo":
                fit_features = descriptors["clean"]["candidate"][fit_permutation]
            else:
                fit_features = descriptors["clean"][role][fit_indices]
            mean, scale = _fit_scaler(fit_features)
            fit_scaled = _apply_scaler(fit_features, mean, scale)
            weights, record = fit_offset_readout(
                fit_scaled,
                clean["probabilities"][fit_indices],
                targets[fit_indices],
            )
            prefix = f"fold{fold}__{role}"
            parameters[f"{prefix}__mean"] = mean
            parameters[f"{prefix}__scale"] = scale
            parameters[f"{prefix}__weights"] = weights
            record.update(
                {
                    "fold": fold,
                    "role": role,
                    "fit_rows": int(fit_indices.size),
                    "hold_rows": int(hold_indices.size),
                    "scaler_mean_sha256": hashlib.sha256(
                        mean.astype("<f8").tobytes()
                    ).hexdigest(),
                    "scaler_scale_sha256": hashlib.sha256(
                        scale.astype("<f8").tobytes()
                    ).hexdigest(),
                    "weights_sha256": hashlib.sha256(
                        weights.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )
            fit_records.append(record)
            for condition, _brightness, _contrast in CONDITIONS:
                if role == "source_placebo":
                    hold_features = descriptors[condition]["candidate"][hold_permutation]
                else:
                    hold_features = descriptors[condition][role][hold_indices]
                outputs[condition][role][hold_indices] = apply_offset_readout(
                    hold_features,
                    condition_rows[condition]["probabilities"][hold_indices],
                    mean,
                    scale,
                    weights,
                )
    parameters["placebo_hold_map"] = hold_map
    if np.any(hold_map < 0):
        raise RuntimeError("Placebo hold mapping is incomplete")
    if any(
        not np.isfinite(outputs[condition][role]).all()
        for condition, _brightness, _contrast in CONDITIONS
        for role in ROLES
    ):
        raise RuntimeError("OOF readout probabilities are incomplete")
    diagnostics = {
        "derangements": derangement_records,
        "all_source_deranged": all(
            row["fit_same_source"] == 0 and row["hold_same_source"] == 0
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
    return outputs, parameters, fit_records, diagnostics


def _calibration_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    selected = np.clip(probs[np.arange(labels.size), labels], PROBABILITY_EPSILON, 1.0)
    one_hot = np.eye(5, dtype=np.float64)[labels]
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 16)[:-1]:
        upper = lower + 1.0 / 15.0
        mask = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "nll": float(-np.log(selected).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece_15": float(ece),
    }


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predictions = np.asarray(probabilities).argmax(axis=1)
    result = _classification_metrics(targets, predictions, num_classes=5)
    result.update(_calibration_metrics(targets, probabilities))
    result["predicted_support"] = np.bincount(predictions, minlength=5).tolist()
    return result


def _comparison(
    targets: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(targets, dtype=np.int64)
    control_prediction = np.asarray(control).argmax(axis=1)
    candidate_prediction = np.asarray(candidate).argmax(axis=1)
    changed = control_prediction != candidate_prediction
    control_correct = control_prediction == labels
    candidate_correct = candidate_prediction == labels
    focus = labels == FOCUS_CLASS
    restricted = np.isin(labels, np.asarray(RESTRICTED_NEGATIVE_CLASSES))
    control_fp = restricted & (control_prediction == FOCUS_CLASS)
    candidate_fp = restricted & (candidate_prediction == FOCUS_CLASS)
    return {
        "changed": int(changed.sum()),
        "corrections": int((~control_correct & candidate_correct).sum()),
        "harms": int((control_correct & ~candidate_correct).sum()),
        "neutral": int((changed & (control_correct == candidate_correct)).sum()),
        "focus_fn_rescue": int(
            (focus & (control_prediction != FOCUS_CLASS) & (candidate_prediction == FOCUS_CLASS)).sum()
        ),
        "focus_tp_break": int(
            (focus & (control_prediction == FOCUS_CLASS) & (candidate_prediction != FOCUS_CLASS)).sum()
        ),
        "restricted_fp": {
            "control": int(control_fp.sum()),
            "candidate": int(candidate_fp.sum()),
            "removed": int((control_fp & ~candidate_fp).sum()),
            "created": int((~control_fp & candidate_fp).sum()),
            "net_reduction": int(control_fp.sum() - candidate_fp.sum()),
        },
        "prediction_transition_matrix": np.asarray(
            [
                [
                    int(((control_prediction == first) & (candidate_prediction == second)).sum())
                    for second in range(5)
                ]
                for first in range(5)
            ],
            dtype=np.int64,
        ).tolist(),
    }


def direction_auc(
    targets: np.ndarray,
    keeper: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(targets, dtype=np.int64)
    keeper_prediction = np.asarray(keeper).argmax(axis=1)
    positive = labels == FOCUS_CLASS
    negative = np.isin(labels, RESTRICTED_NEGATIVE_CLASSES) & (
        keeper_prediction == FOCUS_CLASS
    )
    selected = positive | negative
    binary = positive[selected].astype(np.int64)
    scores = (
        np.asarray(candidate, dtype=np.float64)[:, FOCUS_CLASS]
        - np.asarray(control, dtype=np.float64)[:, FOCUS_CLASS]
    )[selected]
    auc = (
        float(roc_auc_score(binary, scores))
        if binary.size and np.unique(binary).size == 2
        else None
    )
    keeper_tp = positive & (keeper_prediction == FOCUS_CLASS)
    keeper_fn = positive & (keeper_prediction != FOCUS_CLASS)
    return {
        "auc": auc,
        "positive_true_class1": int(positive.sum()),
        "positive_keeper_tp": int(keeper_tp.sum()),
        "positive_keeper_fn": int(keeper_fn.sum()),
        "negative_restricted_keeper_fp": int(negative.sum()),
        "rows": int(selected.sum()),
        "mean_positive_score": float(scores[binary == 1].mean()),
        "mean_negative_score": float(scores[binary == 0].mean()),
    }


def _cohort_summary(
    targets: np.ndarray,
    keeper: np.ndarray,
    cartesian: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(targets, dtype=np.int64)
    keeper_prediction = np.asarray(keeper).argmax(axis=1)
    delta = np.asarray(candidate)[:, FOCUS_CLASS] - np.asarray(cartesian)[:, FOCUS_CLASS]
    cohorts = {
        "true_class1_keeper_tp": (labels == FOCUS_CLASS)
        & (keeper_prediction == FOCUS_CLASS),
        "true_class1_keeper_fn": (labels == FOCUS_CLASS)
        & (keeper_prediction != FOCUS_CLASS),
        "restricted_keeper_fp": np.isin(labels, RESTRICTED_NEGATIVE_CLASSES)
        & (keeper_prediction == FOCUS_CLASS),
        "remaining_restricted_negatives": np.isin(labels, RESTRICTED_NEGATIVE_CLASSES)
        & (keeper_prediction != FOCUS_CLASS),
    }
    result: Dict[str, object] = {}
    for name, mask in cohorts.items():
        values = delta[mask]
        result[name] = {
            "rows": int(mask.sum()),
            "mean_delta_p1": float(values.mean()) if values.size else None,
            "median_delta_p1": float(np.median(values)) if values.size else None,
            "minimum_delta_p1": float(values.min()) if values.size else None,
            "maximum_delta_p1": float(values.max()) if values.size else None,
        }
    return result


def _metric_delta(candidate: Mapping[str, object], control: Mapping[str, object]) -> Dict[str, float]:
    return {
        "macro_f1": float(candidate["macro_f1"] - control["macro_f1"]),
        "class1_precision": float(
            candidate["per_class_precision"][FOCUS_CLASS]
            - control["per_class_precision"][FOCUS_CLASS]
        ),
        "class1_recall": float(
            candidate["per_class_recall"][FOCUS_CLASS]
            - control["per_class_recall"][FOCUS_CLASS]
        ),
        "class1_f1": float(
            candidate["per_class_f1"][FOCUS_CLASS]
            - control["per_class_f1"][FOCUS_CLASS]
        ),
    }


def build_condition_results(
    *,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Dict[str, object], Dict[str, object]]:
    results: Dict[str, object] = {}
    cohorts: Dict[str, object] = {}
    for condition, _brightness, _contrast in CONDITIONS:
        rows = condition_rows[condition]
        targets = rows["target"]
        keeper = rows["probabilities"]
        metrics = {"keeper": _metrics(targets, keeper)}
        metrics.update({role: _metrics(targets, outputs[condition][role]) for role in ROLES})
        comparisons = {
            "candidate_vs_cartesian": _comparison(
                targets, outputs[condition]["cartesian"], outputs[condition]["candidate"]
            ),
            "candidate_vs_linear": _comparison(
                targets, outputs[condition]["linear"], outputs[condition]["candidate"]
            ),
            "candidate_vs_source_placebo": _comparison(
                targets,
                outputs[condition]["source_placebo"],
                outputs[condition]["candidate"],
            ),
        }
        deltas = {
            "candidate_vs_cartesian": _metric_delta(
                metrics["candidate"], metrics["cartesian"]
            ),
            "candidate_vs_linear": _metric_delta(
                metrics["candidate"], metrics["linear"]
            ),
            "candidate_vs_source_placebo": _metric_delta(
                metrics["candidate"], metrics["source_placebo"]
            ),
        }
        directions = {
            role: direction_auc(
                targets,
                keeper,
                outputs[condition]["cartesian"],
                outputs[condition][role],
            )
            for role in ("candidate", "linear", "source_placebo")
        }
        results[condition] = {
            "metrics": metrics,
            "deltas": deltas,
            "comparisons": comparisons,
            "directions": directions,
        }
        cohorts[condition] = _cohort_summary(
            targets,
            keeper,
            outputs[condition]["cartesian"],
            outputs[condition]["candidate"],
        )
    return results, cohorts


def build_fold_metrics(
    *,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> list[Dict[str, object]]:
    clean = condition_rows["clean"]
    rows = []
    for fold in range(FOLDS):
        selected = clean["fold"] == fold
        role_metrics = {
            role: _metrics(clean["target"][selected], outputs["clean"][role][selected])
            for role in ROLES
        }
        rows.append(
            {
                "fold": fold,
                "rows": int(selected.sum()),
                "source_overlap": 0,
                "metrics": role_metrics,
                "candidate_vs_cartesian": _metric_delta(
                    role_metrics["candidate"], role_metrics["cartesian"]
                ),
            }
        )
    return rows


def _effective_rank(features: np.ndarray, *, maximum_rows: int = 512) -> float:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[0] > maximum_rows:
        indices = np.linspace(0, values.shape[0] - 1, maximum_rows, dtype=np.int64)
        values = values[indices]
    values = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(values, compute_uv=False, full_matrices=False)
    energy = np.square(singular)
    if not np.any(energy > 0.0):
        return 0.0
    probability = energy / energy.sum()
    probability = probability[probability > 0.0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _descriptor_correlation(left: np.ndarray, right: np.ndarray) -> float:
    rows = min(512, left.shape[0])
    indices = np.linspace(0, left.shape[0] - 1, rows, dtype=np.int64)
    first = np.asarray(left[indices], dtype=np.float64).reshape(-1)
    second = np.asarray(right[indices], dtype=np.float64).reshape(-1)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0.0 else 0.0


def descriptor_diagnostics(
    descriptors: Mapping[str, Mapping[str, np.ndarray]]
) -> Dict[str, object]:
    clean = descriptors["clean"]
    roles = {}
    for role in EXTRACTED_ROLES:
        values = clean[role]
        roles[role] = {
            "shape": list(values.shape),
            "finite": bool(np.isfinite(values).all()),
            "effective_rank": _effective_rank(values),
            "mean_l2": float(np.linalg.norm(values, axis=1).mean()),
            "maximum_absolute": float(np.abs(values).max()),
        }
    return {
        "roles": roles,
        "candidate_cartesian_correlation": _descriptor_correlation(
            clean["candidate"], clean["cartesian"]
        ),
        "candidate_linear_correlation": _descriptor_correlation(
            clean["candidate"], clean["linear"]
        ),
        "candidate_cartesian_max_abs_difference": float(
            np.abs(clean["candidate"] - clean["cartesian"]).max()
        ),
        "candidate_linear_max_abs_difference": float(
            np.abs(clean["candidate"] - clean["linear"]).max()
        ),
    }


def assess_bbox_logpolar_a0(
    *,
    structural_checks: Mapping[str, bool],
    results: Mapping[str, object],
    fold_metrics: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    clean = results["clean"]
    clean_delta = clean["deltas"]["candidate_vs_cartesian"]
    clean_events = clean["comparisons"]["candidate_vs_cartesian"]
    candidate_auc = clean["directions"]["candidate"]["auc"]
    linear_auc = clean["directions"]["linear"]["auc"]
    placebo_auc = clean["directions"]["source_placebo"]["auc"]
    fold_deltas = [row["candidate_vs_cartesian"] for row in fold_metrics]
    shifted = [results[name] for name, _b, _c in CONDITIONS if name != "clean"]

    clean_checks = {
        "clean_macro_f1_delta_gte_0p003": clean_delta["macro_f1"] >= 0.003,
        "clean_class1_f1_delta_gte_0p010": clean_delta["class1_f1"] >= 0.010,
        "clean_class1_precision_delta_gte_0p015": clean_delta["class1_precision"] >= 0.015,
        "clean_class1_recall_delta_gte_neg0p005": clean_delta["class1_recall"] >= -0.005,
        "clean_net_restricted_fp_removal_gte_10": clean_events["restricted_fp"]["net_reduction"] >= 10,
        "clean_corrections_gte_harms": clean_events["corrections"] >= clean_events["harms"],
        "clean_fn_rescue_gte_tp_break": clean_events["focus_fn_rescue"] >= clean_events["focus_tp_break"],
        "clean_four_of_five_fold_class1_f1_positive": sum(
            row["class1_f1"] > 0.0 for row in fold_deltas
        )
        >= 4,
        "clean_four_of_five_fold_class1_precision_positive": sum(
            row["class1_precision"] > 0.0 for row in fold_deltas
        )
        >= 4,
        "clean_worst_fold_class1_f1_gte_neg0p010": min(
            row["class1_f1"] for row in fold_deltas
        )
        >= -0.010,
        "clean_direction_auc_gte_0p62": candidate_auc is not None
        and candidate_auc >= 0.62,
    }
    specificity_checks = {
        "candidate_beats_linear_macro_f1": clean["deltas"]["candidate_vs_linear"]["macro_f1"] >= 0.002,
        "candidate_beats_linear_class1_f1": clean["deltas"]["candidate_vs_linear"]["class1_f1"] >= 0.005,
        "candidate_beats_linear_class1_precision": clean["deltas"]["candidate_vs_linear"]["class1_precision"] >= 0.005,
        "candidate_beats_placebo_macro_f1": clean["deltas"]["candidate_vs_source_placebo"]["macro_f1"] >= 0.002,
        "candidate_beats_placebo_class1_f1": clean["deltas"]["candidate_vs_source_placebo"]["class1_f1"] >= 0.005,
        "candidate_beats_placebo_class1_precision": clean["deltas"]["candidate_vs_source_placebo"]["class1_precision"] >= 0.005,
        "candidate_auc_beats_linear_by_0p03": candidate_auc is not None
        and linear_auc is not None
        and candidate_auc - linear_auc >= 0.03,
        "candidate_auc_beats_placebo_by_0p03": candidate_auc is not None
        and placebo_auc is not None
        and candidate_auc - placebo_auc >= 0.03,
    }
    shifted_deltas = [
        row["deltas"]["candidate_vs_cartesian"] for row in shifted
    ]
    shifted_events = [
        row["comparisons"]["candidate_vs_cartesian"] for row in shifted
    ]
    robustness_checks = {
        "shift_precision_nonnegative_all": all(
            row["class1_precision"] >= 0.0 for row in shifted_deltas
        ),
        "shift_precision_positive_at_least_two": sum(
            row["class1_precision"] > 0.0 for row in shifted_deltas
        )
        >= 2,
        "shift_worst_class1_f1_gte_neg0p005": min(
            row["class1_f1"] for row in shifted_deltas
        )
        >= -0.005,
        "shift_worst_class1_recall_gte_neg0p015": min(
            row["class1_recall"] for row in shifted_deltas
        )
        >= -0.015,
        "shift_total_net_restricted_fp_positive": sum(
            row["restricted_fp"]["net_reduction"] for row in shifted_events
        )
        > 0,
        "shift_macro_f1_all_gte_neg0p005": all(
            row["macro_f1"] >= -0.005 for row in shifted_deltas
        ),
    }
    checks = {
        **{str(name): bool(value) for name, value in structural_checks.items()},
        **clean_checks,
        **specificity_checks,
        **robustness_checks,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "specificity_checks": specificity_checks,
        "robustness_checks": robustness_checks,
        "checks": checks,
        "failed_checks": failed,
        "automatic_gates_passed": not failed,
        "visual_review_required": not failed,
        "model_integration_authorized": False,
        "validation_test_authorized": False,
        "full_train_authorized": False,
    }


def _heatmap_rgb(values: np.ndarray) -> Image.Image:
    normalized = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    red = normalized
    blue = 1.0 - normalized
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def _render_transform_sheet(
    output_path: Path,
    visuals: Mapping[int, Mapping[str, np.ndarray]],
    targets: np.ndarray,
) -> None:
    if set(visuals) != set(VISUAL_SAMPLE_INDICES):
        raise ValueError("Fixed visual sample set is incomplete")
    tile = 224
    header = 34
    row_label = 34
    columns = (
        "Cartesian support",
        "Linear-polar RGB",
        "Log-polar RGB",
        "Post-feature polar",
        "Direct log-polar",
    )
    canvas = Image.new(
        "RGB", (tile * len(columns), header + (tile + row_label) * 5), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw.text((column * tile + 6, 10), title, fill="black")
    for class_index, sample_index in enumerate(VISUAL_SAMPLE_INDICES):
        if int(targets[sample_index]) != class_index:
            raise ValueError("Fixed visual sample class order drifted")
        row_top = header + class_index * (tile + row_label)
        draw.text(
            (6, row_top + 8),
            f"class={class_index} sample={sample_index}",
            fill="black",
        )
        payload = visuals[sample_index]
        images = (
            Image.fromarray(payload["support"], mode="RGB"),
            Image.fromarray(payload["linear"], mode="RGB"),
            Image.fromarray(payload["log"], mode="RGB"),
            _heatmap_rgb(payload["cartesian_activation"]),
            _heatmap_rgb(payload["candidate_activation"]),
        )
        for column, image in enumerate(images):
            canvas.paste(
                image.resize((tile, tile), Image.Resampling.BILINEAR),
                (column * tile, row_top + row_label),
            )
    canvas.save(output_path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_fold_metrics_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    flattened = []
    for row in rows:
        for role in ROLES:
            metrics = row["metrics"][role]
            flattened.append(
                {
                    "fold": row["fold"],
                    "rows": row["rows"],
                    "role": role,
                    "macro_f1": metrics["macro_f1"],
                    "class1_precision": metrics["per_class_precision"][FOCUS_CLASS],
                    "class1_recall": metrics["per_class_recall"][FOCUS_CLASS],
                    "class1_f1": metrics["per_class_f1"][FOCUS_CLASS],
                    "restricted_fp": metrics["confusion_matrix"][0][FOCUS_CLASS]
                    + metrics["confusion_matrix"][2][FOCUS_CLASS]
                    + metrics["confusion_matrix"][4][FOCUS_CLASS],
                }
            )
    _write_csv(path, flattened)


def _prediction_rows(
    *,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for condition, _brightness, _contrast in CONDITIONS:
        source = condition_rows[condition]
        for index in range(EXPECTED_TRAIN_ROWS):
            row: Dict[str, object] = {
                "condition": condition,
                "sample_index": index,
                "source_stem": source["source"][index],
                "image_path": source["image_path"][index],
                "fold": int(source["fold"][index]),
                "target_index": int(source["target"][index]),
            }
            for role, probabilities in (
                ("keeper", source["probabilities"]),
                *((role, outputs[condition][role]) for role in ROLES),
            ):
                row[f"{role}_prediction"] = int(probabilities[index].argmax())
                for class_index in range(5):
                    row[f"{role}_prob_{class_index}"] = float(
                        probabilities[index, class_index]
                    )
            rows.append(row)
    return rows


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    files = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name.casefold()):
        if not path.is_file() or path == manifest_path:
            continue
        files.append(
            {
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    aggregate = hashlib.sha256()
    for row in files:
        aggregate.update(str(row["name"]).encode("utf-8"))
        aggregate.update(str(row["sha256"]).encode("ascii"))
    manifest = {
        "mode": "bbox_logpolar_stem_a0_evidence_manifest",
        "payload_count": len(files),
        "payload_size_bytes": int(sum(row["size_bytes"] for row in files)),
        "payload_manifest_sha256": aggregate.hexdigest(),
        "files": files,
        "contains_checkpoint": False,
        "contains_model_binary": False,
        "contains_test_payload": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_names = {row["name"] for row in manifest["files"]}
    observed_names = {
        item.name
        for item in output_dir.iterdir()
        if item.is_file() and item.name != "artifact_manifest.json"
    }
    if observed_names != expected_names:
        raise ValueError("Artifact manifest file set mismatch")
    for row in manifest["files"]:
        payload = output_dir / row["name"]
        if int(payload.stat().st_size) != int(row["size_bytes"]):
            raise ValueError(f"Artifact size mismatch: {payload}")
        if _sha256(payload) != row["sha256"]:
            raise ValueError(f"Artifact SHA mismatch: {payload}")
    return manifest


def _structural_checks(
    *,
    provenance: Mapping[str, object],
    condition_rows: Mapping[str, Mapping[str, np.ndarray]],
    descriptors: Mapping[str, Mapping[str, np.ndarray]],
    descriptor_summary: Mapping[str, object],
    fit_records: Sequence[Mapping[str, object]],
    readout_diagnostics: Mapping[str, object],
    raw_before: str,
    raw_after: str,
    visuals: Mapping[int, object],
) -> Dict[str, bool]:
    clean = condition_rows["clean"]
    source_overlap = 0
    for fold in range(FOLDS):
        fit_sources = set(clean["source"][clean["fold"] != fold].tolist())
        hold_sources = set(clean["source"][clean["fold"] == fold].tolist())
        source_overlap = max(source_overlap, len(fit_sources.intersection(hold_sources)))
    return {
        "provenance_hashes_and_git_exact": bool(provenance["all_exact"]),
        "train_rows_exact": all(
            len(condition_rows[name]["target"]) == EXPECTED_TRAIN_ROWS
            for name, _b, _c in CONDITIONS
        ),
        "class_counts_exact": np.bincount(clean["target"], minlength=5).tolist()
        == list(EXPECTED_CLASS_COUNTS),
        "fold_counts_exact": np.bincount(clean["fold"], minlength=FOLDS).tolist()
        == list(EXPECTED_FOLD_COUNTS),
        "source_overlap_zero": source_overlap == 0
        and int(readout_diagnostics["maximum_source_overlap"]) == 0,
        "all_derangements_change_source": bool(
            readout_diagnostics["all_source_deranged"]
        ),
        "four_conditions_exact": set(descriptors)
        == {name for name, _b, _c in CONDITIONS},
        "descriptor_shapes_exact": all(
            descriptors[name][role].shape == (EXPECTED_TRAIN_ROWS, DESCRIPTOR_DIM)
            for name, _b, _c in CONDITIONS
            for role in EXTRACTED_ROLES
        ),
        "descriptors_finite": all(
            np.isfinite(descriptors[name][role]).all()
            for name, _b, _c in CONDITIONS
            for role in EXTRACTED_ROLES
        ),
        "all_20_readouts_converged": len(fit_records) == 20
        and bool(readout_diagnostics["all_converged"]),
        "candidate_effective_rank_gte_24": descriptor_summary["roles"]["candidate"]["effective_rank"]
        >= 24.0,
        "candidate_differs_from_cartesian": descriptor_summary[
            "candidate_cartesian_max_abs_difference"
        ]
        > 1e-6,
        "candidate_differs_from_linear": descriptor_summary[
            "candidate_linear_max_abs_difference"
        ]
        > 1e-6,
        "fixed_visual_rows_complete": set(visuals) == set(VISUAL_SAMPLE_INDICES),
        "raw_dataset_metadata_unchanged": raw_before == raw_after,
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

    started = time.perf_counter()
    raw_before = _dataset_tree_identity(Path(args.data))
    cidt_summary = json.loads(Path(args.cidt_summary).read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use")
    condition_rows = _read_cidt_conditions(Path(args.cidt_predictions))
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Invalid keeper checkpoint")
    model = build_model_from_checkpoint(dict(checkpoint)).to(device).eval()
    if not hasattr(model, "stem"):
        raise ValueError("Keeper model lacks CNN stem")
    base_dataset, transform, class_names = _build_dataset(
        data_path=Path(args.data),
        checkpoint=checkpoint,
        condition_rows=condition_rows,
    )

    descriptors: Dict[str, Dict[str, object]] = {}
    clean_visuals: Dict[int, Dict[str, np.ndarray]] = {}
    for condition, brightness, contrast in CONDITIONS:
        payload = _extract_condition_descriptors(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            model=model,
            base_dataset=base_dataset,
            transform=transform,
            expected=condition_rows[condition],
            checkpoint=checkpoint,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        descriptors[condition] = payload
        if condition == "clean":
            clean_visuals = payload["visuals"]
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    descriptor_summary = descriptor_diagnostics(descriptors)
    outputs, parameters, fit_records, readout_diagnostics = fit_source_oof_readouts(
        descriptors=descriptors,
        condition_rows=condition_rows,
    )
    results, cohorts = build_condition_results(
        outputs=outputs, condition_rows=condition_rows
    )
    fold_metrics = build_fold_metrics(
        outputs=outputs, condition_rows=condition_rows
    )
    raw_after = _dataset_tree_identity(Path(args.data))
    structural = _structural_checks(
        provenance=provenance,
        condition_rows=condition_rows,
        descriptors=descriptors,
        descriptor_summary=descriptor_summary,
        fit_records=fit_records,
        readout_diagnostics=readout_diagnostics,
        raw_before=raw_before,
        raw_after=raw_after,
        visuals=clean_visuals,
    )
    gate = assess_bbox_logpolar_a0(
        structural_checks=structural,
        results=results,
        fold_metrics=fold_metrics,
    )

    transform_sheet = output_dir / "fixed_transform_activation_sheet.png"
    _render_transform_sheet(
        transform_sheet, clean_visuals, condition_rows["clean"]["target"]
    )
    predictions_path = output_dir / "predictions_all_conditions.csv"
    _write_csv(
        predictions_path,
        _prediction_rows(outputs=outputs, condition_rows=condition_rows),
    )
    fold_metrics_path = output_dir / "fold_metrics.csv"
    _write_fold_metrics_csv(fold_metrics_path, fold_metrics)
    descriptor_cache_path = output_dir / "descriptor_cache.npz"
    np.savez_compressed(
        descriptor_cache_path,
        conditions=np.asarray([name for name, _b, _c in CONDITIONS]),
        target=condition_rows["clean"]["target"],
        fold=condition_rows["clean"]["fold"],
        sample_index=condition_rows["clean"]["sample_index"],
        **{
            f"{condition}__{role}": descriptors[condition][role]
            for condition, _b, _c in CONDITIONS
            for role in EXTRACTED_ROLES
        },
    )
    readout_parameters_path = output_dir / "readout_parameters.npz"
    np.savez_compressed(readout_parameters_path, **parameters)
    descriptor_path = output_dir / "descriptor_diagnostics.json"
    descriptor_path.write_text(
        json.dumps(descriptor_summary, indent=2), encoding="utf-8"
    )
    readout_path = output_dir / "readout_diagnostics.json"
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
    cohort_path = output_dir / "cohort_summary.json"
    cohort_path.write_text(json.dumps(cohorts, indent=2), encoding="utf-8")

    status = (
        "awaiting_visual_review"
        if gate["automatic_gates_passed"]
        else "rejected_before_model_integration"
    )
    summary = {
        "mode": "bbox_centered_logpolar_stem_train_only_a0",
        "status": status,
        "provenance": provenance,
        "protocol": {
            "seed": SEED,
            "image_size": IMAGE_SIZE,
            "stem_shape": [STEM_CHANNELS, STEM_SIZE, STEM_SIZE],
            "support_scale": SUPPORT_SCALE,
            "log_radius": LOG_RADIUS,
            "radial_bands": RADIAL_BANDS,
            "descriptor_dim": DESCRIPTOR_DIM,
            "readout_c": READOUT_C,
            "max_iterations": MAX_ITERATIONS,
            "conditions": [
                {"name": name, "brightness": brightness, "contrast": contrast}
                for name, brightness, contrast in CONDITIONS
            ],
            "roles": list(ROLES),
            "visual_sample_indices": list(VISUAL_SAMPLE_INDICES),
        },
        "dataset": {
            "split": "train",
            "rows": EXPECTED_TRAIN_ROWS,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "source_groups": int(len(set(condition_rows["clean"]["source"].tolist()))),
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
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "batch_size": int(args.batch_size),
            "requested_num_workers": int(args.num_workers),
            "torch_threads": int(args.torch_threads),
            "condition_extraction": {
                name: {
                    "seconds": descriptors[name]["seconds"],
                    "peak_cuda_memory_mib": descriptors[name]["peak_cuda_memory_mib"],
                    "loader": descriptors[name]["loader"],
                }
                for name, _b, _c in CONDITIONS
            },
            "total_seconds": float(time.perf_counter() - started),
        },
        "descriptor_diagnostics": descriptor_summary,
        "readout_diagnostics": {
            **readout_diagnostics,
            "fit_records": fit_records,
        },
        "results": results,
        "cohorts": cohorts,
        "fold_metrics": fold_metrics,
        "gate": gate,
        "artifacts": {
            "predictions": predictions_path.name,
            "fold_metrics": fold_metrics_path.name,
            "descriptor_cache": descriptor_cache_path.name,
            "readout_parameters": readout_parameters_path.name,
            "descriptor_diagnostics": descriptor_path.name,
            "readout_diagnostics": readout_path.name,
            "cohort_summary": cohort_path.name,
            "transform_activation_sheet": transform_sheet.name,
        },
        "validation_data_used": False,
        "test_data_used": False,
        "raw_dataset_touched": False,
        "checkpoint_written": False,
        "model_integration_authorized": False,
        "full_train_authorized": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_manifest(output_dir)
    return summary


def _read_predictions(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, list[Dict[str, str]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["condition"], []).append(row)
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for condition, _brightness, _contrast in CONDITIONS:
        rows = grouped.get(condition, [])
        if len(rows) != EXPECTED_TRAIN_ROWS:
            raise ValueError(f"Prediction replay row mismatch for {condition}")
        sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
        if not np.array_equal(sample_index, np.arange(EXPECTED_TRAIN_ROWS)):
            raise ValueError("Prediction replay sample order mismatch")
        payload: Dict[str, np.ndarray] = {
            "sample_index": sample_index,
            "target": np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64),
            "fold": np.asarray([int(row["fold"]) for row in rows], dtype=np.int64),
            "source": np.asarray([row["source_stem"] for row in rows], dtype=object),
            "image_path": np.asarray([row["image_path"] for row in rows]),
        }
        for role in ("keeper", *ROLES):
            payload[role] = np.asarray(
                [
                    [float(row[f"{role}_prob_{class_index}"]) for class_index in range(5)]
                    for row in rows
                ],
                dtype=np.float64,
            )
        result[condition] = payload
    return result


def _maximum_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError("Replay mapping keys differ")
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            raise ValueError("Replay sequence types differ")
        if len(left) != len(right):
            raise ValueError("Replay sequence lengths differ")
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        if bool(left) != bool(right):
            raise ValueError("Replay boolean values differ")
        return 0.0
    if left is None or right is None:
        if left is not right:
            raise ValueError("Replay null values differ")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError(f"Replay values differ: {left!r} != {right!r}")
    return 0.0


def replay_summary(summary_path: Path) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    output_dir = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = _verify_manifest(output_dir)
    predictions = _read_predictions(output_dir / summary["artifacts"]["predictions"])
    descriptor_cache = np.load(
        output_dir / summary["artifacts"]["descriptor_cache"], allow_pickle=False
    )
    parameters = np.load(
        output_dir / summary["artifacts"]["readout_parameters"], allow_pickle=False
    )
    condition_rows = {
        condition: {
            "sample_index": predictions[condition]["sample_index"],
            "target": predictions[condition]["target"],
            "fold": predictions[condition]["fold"],
            "source": predictions[condition]["source"],
            "image_path": predictions[condition]["image_path"],
            "probabilities": predictions[condition]["keeper"],
        }
        for condition, _b, _c in CONDITIONS
    }
    outputs = {
        condition: {
            role: np.full((EXPECTED_TRAIN_ROWS, 5), np.nan, dtype=np.float64)
            for role in ROLES
        }
        for condition, _b, _c in CONDITIONS
    }
    maximum_probability_difference = 0.0
    folds = condition_rows["clean"]["fold"]
    for fold in range(FOLDS):
        hold_indices = np.flatnonzero(folds == fold)
        hold_permutation = np.asarray(
            parameters[f"fold{fold}__placebo_hold_indices"], dtype=np.int64
        )
        for role in ROLES:
            prefix = f"fold{fold}__{role}"
            mean = parameters[f"{prefix}__mean"]
            scale = parameters[f"{prefix}__scale"]
            weights = parameters[f"{prefix}__weights"]
            for condition, _brightness, _contrast in CONDITIONS:
                if role == "source_placebo":
                    features = descriptor_cache[f"{condition}__candidate"][hold_permutation]
                else:
                    features = descriptor_cache[f"{condition}__{role}"][hold_indices]
                reconstructed = apply_offset_readout(
                    features,
                    condition_rows[condition]["probabilities"][hold_indices],
                    mean,
                    scale,
                    weights,
                )
                outputs[condition][role][hold_indices] = reconstructed
                maximum_probability_difference = max(
                    maximum_probability_difference,
                    float(
                        np.abs(
                            reconstructed - predictions[condition][role][hold_indices]
                        ).max()
                    ),
                )
    results, cohorts = build_condition_results(
        outputs=outputs, condition_rows=condition_rows
    )
    fold_metrics = build_fold_metrics(
        outputs=outputs, condition_rows=condition_rows
    )
    gate = assess_bbox_logpolar_a0(
        structural_checks=summary["gate"]["structural_checks"],
        results=results,
        fold_metrics=fold_metrics,
    )
    metric_difference = max(
        _maximum_numeric_difference(results, summary["results"]),
        _maximum_numeric_difference(cohorts, summary["cohorts"]),
        _maximum_numeric_difference(fold_metrics, summary["fold_metrics"]),
        _maximum_numeric_difference(gate, summary["gate"]),
    )
    passed = (
        maximum_probability_difference <= 1e-12
        and metric_difference <= 1e-12
        and not bool(manifest.get("contains_test_payload", True))
        and not bool(manifest.get("contains_checkpoint", True))
    )
    return {
        "mode": "bbox_logpolar_stem_a0_exact_replay",
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(output_dir / "artifact_manifest.json"),
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_metric_gate_difference": metric_difference,
        "fp32_descriptor_tolerance": 1e-7,
        "fp64_probability_metric_tolerance": 1e-12,
        "manifest_verified": True,
        "passed": passed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _validate_locked_args(args)
    if args.replay_summary is not None:
        report = replay_summary(args.replay_summary)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    if args.preflight_only:
        provenance = verify_provenance(args)
        output_dir = Path(args.output_dir).resolve()
        print(
            json.dumps(
                {
                    "mode": "bbox_logpolar_stem_a0_preflight",
                    "provenance": provenance,
                    "dataset_opened": False,
                    "checkpoint_opened": False,
                    "descriptor_created": False,
                    "output_exists": output_dir.exists(),
                    "output_nonempty": output_dir.exists() and any(output_dir.iterdir()),
                    "validation_data_used": False,
                    "test_data_used": False,
                },
                indent=2,
            )
        )
        return 0
    summary = run_audit(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

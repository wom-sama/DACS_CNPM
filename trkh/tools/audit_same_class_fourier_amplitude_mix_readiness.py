from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    set_seed,
)
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.audit_friendly_foreground_adversarial_readiness import (
    build_eroded_bbox_mask,
)
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _forward_logits,
    _select_bbox_token_prior,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)
from trkh.tools.probe_embedding_prototypes import _collate_classification


SEED = 20260712
FOCUS_CLASS = 1
NEGATIVE_CLASSES = (0, 2, 4)
CROSS_CLASS_MAP = {0: 1, 1: 0, 2: 1, 3: 1, 4: 1}
MIX_LAMBDA = 0.5
SPECTRUM_RATIO = 1.0
FOLDS = 5
EXPECTED_TRAIN_COUNT = 9215
EXPECTED_VAL_COUNT = 2606
EXPECTED_KEEPER_VAL_MACRO = 0.882925
EXPECTED_KEEPER_VAL_CLASS1 = 0.678261
EXPECTED_CHECKPOINT_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)

FACT_PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2021/html/"
    "Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html"
)
FACT_CODE_URL = "https://github.com/MediaBrain-SJTU/FACT"
FDA_PAPER_URL = (
    "https://openaccess.thecvf.com/content_CVPR_2020/html/"
    "Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html"
)
FDA_CODE_URL = "https://github.com/YanchaoYang/FDA"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked no-test readiness audit for source-distinct, same-class Fourier "
            "amplitude mixing. It does not fit or save a model."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--allow-preflight",
        action="store_true",
        default=False,
        help="Allow capped prefixes for implementation checks; gate remains closed.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested or "").strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(normalized or ("cuda" if torch.cuda.is_available() else "cpu"))


def _source_stem(path: str | Path) -> str:
    stem = Path(path).stem.casefold().strip()
    if not stem:
        raise ValueError(f"sample path has no source stem: {path}")
    return stem


def fourier_amplitude_mix_rgb(
    target_rgb: Tensor,
    peer_rgb: Tensor,
    *,
    mix_lambda: float = MIX_LAMBDA,
    spectrum_ratio: float = SPECTRUM_RATIO,
) -> Tuple[Tensor, Dict[str, object]]:
    """Mix peer amplitude into target phase in unnormalized RGB space."""
    if target_rgb.shape != peer_rgb.shape:
        raise ValueError("target and peer tensors must have the same shape")
    if target_rgb.ndim != 4 or int(target_rgb.size(1)) != 3:
        raise ValueError("RGB tensors must have shape [B,3,H,W]")
    if not 0.0 <= float(mix_lambda) <= 1.0:
        raise ValueError("mix_lambda must be in [0,1]")
    if not 0.0 < float(spectrum_ratio) <= 1.0:
        raise ValueError("spectrum_ratio must be in (0,1]")

    target = target_rgb.float()
    peer = peer_rgb.to(device=target.device, dtype=torch.float32)
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    peer_fft = torch.fft.fft2(peer, dim=(-2, -1))
    target_amplitude = target_fft.abs()
    peer_amplitude = peer_fft.abs()
    mixed_amplitude = target_amplitude.clone()

    if float(spectrum_ratio) >= 1.0:
        mixed_amplitude = (
            (1.0 - float(mix_lambda)) * target_amplitude
            + float(mix_lambda) * peer_amplitude
        )
    else:
        shifted_target = torch.fft.fftshift(target_amplitude, dim=(-2, -1))
        shifted_peer = torch.fft.fftshift(peer_amplitude, dim=(-2, -1))
        shifted_mixed = shifted_target.clone()
        height, width = int(target.size(-2)), int(target.size(-1))
        side_scale = math.sqrt(float(spectrum_ratio))
        crop_height = max(1, min(height, int(round(height * side_scale))))
        crop_width = max(1, min(width, int(round(width * side_scale))))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        shifted_mixed[..., top : top + crop_height, left : left + crop_width] = (
            (1.0 - float(mix_lambda))
            * shifted_target[..., top : top + crop_height, left : left + crop_width]
            + float(mix_lambda)
            * shifted_peer[..., top : top + crop_height, left : left + crop_width]
        )
        mixed_amplitude = torch.fft.ifftshift(shifted_mixed, dim=(-2, -1))

    mixed_spectrum = mixed_amplitude * torch.exp(1j * torch.angle(target_fft))
    reconstructed = torch.fft.ifft2(mixed_spectrum, dim=(-2, -1))
    real = reconstructed.real
    finite = bool(torch.isfinite(real).all() and torch.isfinite(reconstructed.imag).all())
    outside = (real < 0.0) | (real > 1.0)
    clipped = real.clamp(0.0, 1.0)
    diagnostics: Dict[str, object] = {
        "finite": finite,
        "max_imaginary_residual": float(reconstructed.imag.abs().max().item()),
        "clip_fraction": float(outside.float().mean().item()),
        "mean_abs_delta_rgb": float((clipped - target).abs().mean().item()),
        "max_abs_delta_rgb": float((clipped - target).abs().max().item()),
    }
    return clipped, diagnostics


def _stable_choice(candidates: Sequence[int], token: str) -> int:
    if not candidates:
        raise ValueError("peer candidate list is empty")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(candidates[offset % len(candidates)])


def build_peer_plan(
    *,
    target_split: str,
    peer_split: str,
    target_indices: Sequence[int],
    target_labels: Sequence[int],
    target_paths: Sequence[str | Path],
    peer_labels: Sequence[int],
    peer_paths: Sequence[str | Path],
    seed: int = SEED,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if len(target_labels) != len(target_paths):
        raise ValueError("target label/path coverage differs")
    if len(peer_labels) != len(peer_paths):
        raise ValueError("peer label/path coverage differs")
    normalized_indices = [int(index) for index in target_indices]
    if len(normalized_indices) != len(set(normalized_indices)):
        raise ValueError("target indices must be unique")
    if any(index < 0 or index >= len(target_labels) for index in normalized_indices):
        raise ValueError("target index is outside the target dataset")

    peer_sources = [_source_stem(path) for path in peer_paths]
    candidates_by_class: Dict[int, List[int]] = {}
    for peer_index, label in enumerate(peer_labels):
        candidates_by_class.setdefault(int(label), []).append(int(peer_index))
    required_classes = set(CROSS_CLASS_MAP) | set(CROSS_CLASS_MAP.values())
    missing = sorted(label for label in required_classes if not candidates_by_class.get(label))
    if missing:
        raise ValueError(f"peer dataset lacks class support: {missing}")

    rows: List[Dict[str, object]] = []
    for target_index in normalized_indices:
        label = int(target_labels[target_index])
        if label not in CROSS_CLASS_MAP:
            raise ValueError(f"target class has no locked cross control: {label}")
        target_path = str(target_paths[target_index])
        target_source = _source_stem(target_path)
        same_candidates = [
            index
            for index in candidates_by_class[label]
            if peer_sources[index] != target_source
        ]
        cross_label = int(CROSS_CLASS_MAP[label])
        cross_candidates = [
            index
            for index in candidates_by_class[cross_label]
            if peer_sources[index] != target_source
        ]
        if not same_candidates:
            raise ValueError(
                f"target {target_split}:{target_index} has no source-distinct same-class peer"
            )
        if not cross_candidates:
            raise ValueError(
                f"target {target_split}:{target_index} has no source-distinct cross peer"
            )
        same_index = _stable_choice(
            same_candidates,
            f"{seed}|{target_split}|{target_index}|{target_source}|same|{label}",
        )
        cross_index = _stable_choice(
            cross_candidates,
            f"{seed}|{target_split}|{target_index}|{target_source}|cross|{cross_label}",
        )
        rows.append(
            {
                "target_split": str(target_split),
                "target_index": int(target_index),
                "target_path": target_path,
                "target_source_stem": target_source,
                "target_label": label,
                "peer_split": str(peer_split),
                "same_peer_index": same_index,
                "same_peer_path": str(peer_paths[same_index]),
                "same_peer_source_stem": peer_sources[same_index],
                "same_peer_label": int(peer_labels[same_index]),
                "cross_peer_index": cross_index,
                "cross_peer_path": str(peer_paths[cross_index]),
                "cross_peer_source_stem": peer_sources[cross_index],
                "cross_peer_label": int(peer_labels[cross_index]),
                "expected_cross_peer_label": cross_label,
            }
        )

    same_reuse = Counter(int(row["same_peer_index"]) for row in rows)
    cross_reuse = Counter(int(row["cross_peer_index"]) for row in rows)
    stats = {
        "target_split": str(target_split),
        "peer_split": str(peer_split),
        "support": int(len(rows)),
        "same_label_matches": int(
            sum(int(row["same_peer_label"]) == int(row["target_label"]) for row in rows)
        ),
        "same_source_distinct": int(
            sum(
                str(row["same_peer_source_stem"])
                != str(row["target_source_stem"])
                for row in rows
            )
        ),
        "cross_label_matches": int(
            sum(
                int(row["cross_peer_label"])
                == int(row["expected_cross_peer_label"])
                for row in rows
            )
        ),
        "cross_source_distinct": int(
            sum(
                str(row["cross_peer_source_stem"])
                != str(row["target_source_stem"])
                for row in rows
            )
        ),
        "same_unique_peers": int(len(same_reuse)),
        "same_max_peer_reuse": int(max(same_reuse.values(), default=0)),
        "cross_unique_peers": int(len(cross_reuse)),
        "cross_max_peer_reuse": int(max(cross_reuse.values(), default=0)),
    }
    return rows, stats


def _classification_item(dataset: Dataset, index: int) -> Tuple[Tensor, int, Dict[str, object]]:
    item = dataset[int(index)]
    if len(item) == 2:
        image, label = item
        metadata: Dict[str, object] = {}
    elif len(item) == 3:
        image, label, raw_metadata = item
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    else:
        raise ValueError("classification item must have two or three fields")
    if not torch.is_tensor(image):
        raise TypeError("classification image must be a tensor")
    return image, int(label), metadata


class FourierPeerDataset(Dataset):
    def __init__(
        self,
        *,
        target_dataset: Dataset,
        peer_dataset: Dataset,
        plan: Sequence[Mapping[str, object]],
    ) -> None:
        self.target_dataset = target_dataset
        self.peer_dataset = peer_dataset
        self.plan = [dict(row) for row in plan]

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int):
        row = self.plan[int(index)]
        target_index = int(row["target_index"])
        same_index = int(row["same_peer_index"])
        cross_index = int(row["cross_peer_index"])
        image, label, metadata = _classification_item(self.target_dataset, target_index)
        same_image, same_label, _ = _classification_item(self.peer_dataset, same_index)
        cross_image, cross_label, _ = _classification_item(self.peer_dataset, cross_index)
        if label != int(row["target_label"]):
            raise RuntimeError("target label changed after the peer plan was built")
        if same_label != int(row["same_peer_label"]):
            raise RuntimeError("same peer label changed after the peer plan was built")
        if cross_label != int(row["cross_peer_label"]):
            raise RuntimeError("cross peer label changed after the peer plan was built")
        metadata.update(
            {
                "audit_index": torch.tensor(target_index, dtype=torch.long),
                "same_peer_index": torch.tensor(same_index, dtype=torch.long),
                "cross_peer_index": torch.tensor(cross_index, dtype=torch.long),
                "same_peer_label": torch.tensor(same_label, dtype=torch.long),
                "cross_peer_label": torch.tensor(cross_label, dtype=torch.long),
                "same_peer_image": same_image,
                "cross_peer_image": cross_image,
                "image_path": str(row["target_path"]),
            }
        )
        return image, label, metadata


def _loader(
    *,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    context: str,
) -> Tuple[DataLoader, Dict[str, object]]:
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=_collate_classification,
            **kwargs,
        ),
        summary,
    )


def _model_logits(
    *,
    model: nn.Module,
    images: Tensor,
    image_valid_mask: Tensor,
    bbox: Tensor,
    crop_bbox: Tensor,
    bbox_token_prior_source: str,
    device: torch.device,
) -> Tensor:
    prior = _select_bbox_token_prior(
        bbox=bbox,
        crop_bbox=crop_bbox,
        source=bbox_token_prior_source,
    )
    logits, _ = _forward_logits(
        model=model,
        images=images,
        image_valid_mask=image_valid_mask,
        bbox_metadata=bbox,
        bbox_token_prior=prior,
        amp=False,
        device=device,
    )
    return logits.float()


def _normalization_tensors(
    mean: Sequence[float],
    std: Sequence[float],
    reference: Tensor,
) -> Tuple[Tensor, Tensor]:
    mean_tensor = torch.as_tensor(mean, device=reference.device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std_tensor = torch.as_tensor(std, device=reference.device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    if bool((std_tensor <= 0).any()):
        raise ValueError("input normalization std must be positive")
    return mean_tensor, std_tensor


def _unnormalize(images: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    mean_tensor, std_tensor = _normalization_tensors(mean, std, images)
    return (images.float() * std_tensor + mean_tensor).clamp(0.0, 1.0)


def _normalize(images: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    mean_tensor, std_tensor = _normalization_tensors(mean, std, images)
    return (images.float() - mean_tensor) / std_tensor


def focus_cohort(target: int, clean_prediction: int) -> str:
    if int(target) == FOCUS_CLASS:
        return "class1_tp" if int(clean_prediction) == FOCUS_CLASS else "class1_fn"
    if int(target) in NEGATIVE_CLASSES:
        return (
            "negative_fp"
            if int(clean_prediction) == FOCUS_CLASS
            else "negative_nonfp"
        )
    return "other"


def transition_outcome(target: int, clean_prediction: int, view_prediction: int) -> str:
    target = int(target)
    clean_prediction = int(clean_prediction)
    view_prediction = int(view_prediction)
    if target == FOCUS_CLASS and clean_prediction != FOCUS_CLASS and view_prediction == FOCUS_CLASS:
        return "fn_rescue"
    if target == FOCUS_CLASS and clean_prediction == FOCUS_CLASS and view_prediction != FOCUS_CLASS:
        return "tp_break"
    if target in NEGATIVE_CLASSES and clean_prediction == FOCUS_CLASS and view_prediction != FOCUS_CLASS:
        return "fp_remove"
    if target in NEGATIVE_CLASSES and clean_prediction != FOCUS_CLASS and view_prediction == FOCUS_CLASS:
        return "fp_create"
    if clean_prediction != target and view_prediction == target:
        return "correction"
    if clean_prediction == target and view_prediction != target:
        return "harm"
    if clean_prediction != view_prediction:
        return "neutral_change"
    return "unchanged"


def _masked_mean_per_sample(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.to(device=values.device, dtype=values.dtype)
    if expanded.ndim == 3:
        expanded = expanded.unsqueeze(1)
    if int(expanded.size(1)) == 1 and int(values.size(1)) != 1:
        expanded = expanded.expand(-1, int(values.size(1)), -1, -1)
    numerator = (values * expanded).flatten(1).sum(dim=1)
    denominator = expanded.flatten(1).sum(dim=1).clamp(min=1.0)
    return numerator / denominator


def collect_split(
    *,
    split: str,
    target_dataset: Dataset,
    peer_dataset: Dataset,
    plan: Sequence[Mapping[str, object]],
    model: nn.Module,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    plan_by_target = {int(row["target_index"]): dict(row) for row in plan}
    paired_dataset = FourierPeerDataset(
        target_dataset=target_dataset,
        peer_dataset=peer_dataset,
        plan=plan,
    )
    loader, loader_summary = _loader(
        dataset=paired_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        context=f"same_class_fourier_{split}",
    )
    rows: List[Dict[str, object]] = []
    target_batches: List[Tensor] = []
    probability_batches: Dict[str, List[Tensor]] = {
        "clean": [],
        "same": [],
        "cross": [],
    }
    identity_max_error = 0.0
    max_imaginary_residual = 0.0
    finite = True
    same_clip_weighted = 0.0
    cross_clip_weighted = 0.0
    pixel_count = 0

    with torch.inference_mode():
        for images, labels, metadata in tqdm(
            loader,
            desc=f"fourier-readiness:{split}",
            unit="batch",
        ):
            if not isinstance(metadata, Mapping):
                raise ValueError("classification metadata is required")
            required = (
                "bbox",
                "crop_bbox",
                "image_mask",
                "audit_index",
                "same_peer_index",
                "cross_peer_index",
                "same_peer_image",
                "cross_peer_image",
            )
            if any(not torch.is_tensor(metadata.get(key)) for key in required):
                raise ValueError(f"Fourier batch lacks required tensor metadata: {required}")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            bbox = metadata["bbox"].to(device=device, dtype=torch.float32, non_blocking=True)
            crop_bbox = metadata["crop_bbox"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            valid_mask = metadata["image_mask"].to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            if valid_mask.ndim == 4 and int(valid_mask.size(1)) == 1:
                valid_mask = valid_mask[:, 0]
            audit_indices = metadata["audit_index"].to(
                device=device, dtype=torch.long, non_blocking=True
            ).view(-1)
            same_indices = metadata["same_peer_index"].to(
                device=device, dtype=torch.long, non_blocking=True
            ).view(-1)
            cross_indices = metadata["cross_peer_index"].to(
                device=device, dtype=torch.long, non_blocking=True
            ).view(-1)
            same_peer = metadata["same_peer_image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            cross_peer = metadata["cross_peer_image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )

            target_rgb = _unnormalize(images, mean, std)
            same_peer_rgb = _unnormalize(same_peer, mean, std)
            cross_peer_rgb = _unnormalize(cross_peer, mean, std)
            identity_rgb, identity_diag = fourier_amplitude_mix_rgb(
                target_rgb,
                target_rgb,
            )
            same_rgb, same_diag = fourier_amplitude_mix_rgb(target_rgb, same_peer_rgb)
            cross_rgb, cross_diag = fourier_amplitude_mix_rgb(target_rgb, cross_peer_rgb)
            identity_max_error = max(
                identity_max_error,
                float((identity_rgb - target_rgb).abs().max().item()),
            )
            max_imaginary_residual = max(
                max_imaginary_residual,
                float(identity_diag["max_imaginary_residual"]),
                float(same_diag["max_imaginary_residual"]),
                float(cross_diag["max_imaginary_residual"]),
            )
            finite = finite and bool(identity_diag["finite"])
            finite = finite and bool(same_diag["finite"])
            finite = finite and bool(cross_diag["finite"])
            batch_pixels = int(target_rgb.numel())
            same_clip_weighted += float(same_diag["clip_fraction"]) * batch_pixels
            cross_clip_weighted += float(cross_diag["clip_fraction"]) * batch_pixels
            pixel_count += batch_pixels

            combined_images = torch.cat(
                [images, _normalize(same_rgb, mean, std), _normalize(cross_rgb, mean, std)],
                dim=0,
            )
            combined_valid = torch.cat([valid_mask, valid_mask, valid_mask], dim=0)
            combined_bbox = torch.cat([bbox, bbox, bbox], dim=0)
            combined_crop_bbox = torch.cat([crop_bbox, crop_bbox, crop_bbox], dim=0)
            logits = _model_logits(
                model=model,
                images=combined_images,
                image_valid_mask=combined_valid,
                bbox=combined_bbox,
                crop_bbox=combined_crop_bbox,
                bbox_token_prior_source=bbox_token_prior_source,
                device=device,
            )
            clean_logits, same_logits, cross_logits = logits.chunk(3, dim=0)
            probabilities = {
                "clean": torch.softmax(clean_logits, dim=1),
                "same": torch.softmax(same_logits, dim=1),
                "cross": torch.softmax(cross_logits, dim=1),
            }
            predictions = {
                name: value.argmax(dim=1) for name, value in probabilities.items()
            }
            same_delta = (same_rgb - target_rgb).abs()
            cross_delta = (cross_rgb - target_rgb).abs()
            object_mask = build_eroded_bbox_mask(
                crop_bbox,
                valid_mask,
                height=int(images.size(2)),
                width=int(images.size(3)),
                erode_ratio=0.0,
            )
            valid_rgb_mask = valid_mask.unsqueeze(1)
            background_mask = valid_rgb_mask & ~object_mask
            same_object_mae = _masked_mean_per_sample(same_delta, object_mask)
            same_background_mae = _masked_mean_per_sample(same_delta, background_mask)
            cross_object_mae = _masked_mean_per_sample(cross_delta, object_mask)
            cross_background_mae = _masked_mean_per_sample(cross_delta, background_mask)
            same_global_mae = same_delta.flatten(1).mean(dim=1)
            cross_global_mae = cross_delta.flatten(1).mean(dim=1)
            same_channel_shift = (same_rgb - target_rgb).mean(dim=(2, 3))
            cross_channel_shift = (cross_rgb - target_rgb).mean(dim=(2, 3))

            paths = metadata.get("paths")
            if not isinstance(paths, list) or len(paths) != int(images.size(0)):
                raise ValueError("classification paths are incomplete")
            for row_index in range(int(images.size(0))):
                target_index = int(audit_indices[row_index].item())
                plan_row = plan_by_target[target_index]
                if int(same_indices[row_index].item()) != int(plan_row["same_peer_index"]):
                    raise RuntimeError("same peer index changed inside the dataloader")
                if int(cross_indices[row_index].item()) != int(plan_row["cross_peer_index"]):
                    raise RuntimeError("cross peer index changed inside the dataloader")
                target = int(labels[row_index].item())
                clean_prediction = int(predictions["clean"][row_index].item())
                same_prediction = int(predictions["same"][row_index].item())
                cross_prediction = int(predictions["cross"][row_index].item())
                row: Dict[str, object] = dict(plan_row)
                row.update(
                    {
                        "split": str(split),
                        "sample_index": target_index,
                        "image_path": str(paths[row_index]),
                        "source_stem": str(plan_row["target_source_stem"]),
                        "target_index": target,
                        "clean_prediction_index": clean_prediction,
                        "same_prediction_index": same_prediction,
                        "cross_prediction_index": cross_prediction,
                        "focus_cohort": focus_cohort(target, clean_prediction),
                        "same_outcome": transition_outcome(
                            target, clean_prediction, same_prediction
                        ),
                        "cross_outcome": transition_outcome(
                            target, clean_prediction, cross_prediction
                        ),
                        "same_rgb_mae": float(same_global_mae[row_index].item()),
                        "same_object_rgb_mae": float(same_object_mae[row_index].item()),
                        "same_background_rgb_mae": float(
                            same_background_mae[row_index].item()
                        ),
                        "cross_rgb_mae": float(cross_global_mae[row_index].item()),
                        "cross_object_rgb_mae": float(cross_object_mae[row_index].item()),
                        "cross_background_rgb_mae": float(
                            cross_background_mae[row_index].item()
                        ),
                    }
                )
                for channel_index, channel_name in enumerate(("r", "g", "b")):
                    row[f"same_mean_{channel_name}_shift"] = float(
                        same_channel_shift[row_index, channel_index].item()
                    )
                    row[f"cross_mean_{channel_name}_shift"] = float(
                        cross_channel_shift[row_index, channel_index].item()
                    )
                for class_index in range(len(class_names)):
                    for view in ("clean", "same", "cross"):
                        row[f"{view}_prob_{class_index}"] = float(
                            probabilities[view][row_index, class_index].item()
                        )
                row["delta_p1_same"] = float(
                    probabilities["same"][row_index, FOCUS_CLASS].item()
                    - probabilities["clean"][row_index, FOCUS_CLASS].item()
                )
                row["delta_p1_cross"] = float(
                    probabilities["cross"][row_index, FOCUS_CLASS].item()
                    - probabilities["clean"][row_index, FOCUS_CLASS].item()
                )
                rows.append(row)
            target_batches.append(labels.detach().cpu())
            for name in probability_batches:
                probability_batches[name].append(probabilities[name].detach().cpu())

    targets = torch.cat(target_batches, dim=0)
    view_probabilities = {
        name: torch.cat(values, dim=0) for name, values in probability_batches.items()
    }
    metrics = {
        name: build_metrics(
            targets,
            probabilities.argmax(dim=1),
            class_names,
            probabilities=probabilities,
        )
        for name, probabilities in view_probabilities.items()
    }
    return {
        "split": str(split),
        "support": int(len(rows)),
        "rows": rows,
        "metrics": metrics,
        "numerical": {
            "identity_max_error_rgb": float(identity_max_error),
            "max_imaginary_residual": float(max_imaginary_residual),
            "finite": bool(finite),
            "same_clip_fraction": float(same_clip_weighted / max(1, pixel_count)),
            "cross_clip_fraction": float(cross_clip_weighted / max(1, pixel_count)),
            "same_mean_rgb_mae": float(
                np.mean([float(row["same_rgb_mae"]) for row in rows]) if rows else 0.0
            ),
            "same_mean_object_rgb_mae": float(
                np.mean([float(row["same_object_rgb_mae"]) for row in rows])
                if rows
                else 0.0
            ),
            "same_mean_background_rgb_mae": float(
                np.mean([float(row["same_background_rgb_mae"]) for row in rows])
                if rows
                else 0.0
            ),
            "cross_mean_rgb_mae": float(
                np.mean([float(row["cross_rgb_mae"]) for row in rows]) if rows else 0.0
            ),
        },
        "loader": loader_summary,
    }


def summarize_view(
    rows: Sequence[Mapping[str, object]],
    *,
    view: str,
    directional_auc_threshold: float = 0.55,
) -> Dict[str, object]:
    if view not in {"same", "cross"}:
        raise ValueError("view must be 'same' or 'cross'")
    prediction_key = f"{view}_prediction_index"
    outcome_key = f"{view}_outcome"
    delta_key = f"delta_p1_{view}"
    clean_correct = [
        row
        for row in rows
        if int(row["clean_prediction_index"]) == int(row["target_index"])
    ]
    class1_tp = [row for row in rows if str(row["focus_cohort"]) == "class1_tp"]
    class1_fn = [row for row in rows if str(row["focus_cohort"]) == "class1_fn"]
    negative_fp = [row for row in rows if str(row["focus_cohort"]) == "negative_fp"]
    negative_nonfp = [
        row for row in rows if str(row["focus_cohort"]) == "negative_nonfp"
    ]
    outcomes = Counter(str(row[outcome_key]) for row in rows)
    corrections = sum(
        int(row["clean_prediction_index"]) != int(row["target_index"])
        and int(row[prediction_key]) == int(row["target_index"])
        for row in rows
    )
    harms = sum(
        int(row["clean_prediction_index"]) == int(row["target_index"])
        and int(row[prediction_key]) != int(row["target_index"])
        for row in rows
    )
    fn_deltas = [float(row[delta_key]) for row in class1_fn]
    fp_deltas = [float(row[delta_key]) for row in negative_fp]
    separation_auc = float("nan")
    if fn_deltas and fp_deltas:
        labels = np.asarray([1] * len(fn_deltas) + [0] * len(fp_deltas), dtype=np.int64)
        scores = np.asarray(fn_deltas + fp_deltas, dtype=np.float64)
        separation_auc = float(roc_auc_score(labels, scores))
    fn_mean = float(np.mean(fn_deltas)) if fn_deltas else float("nan")
    fp_mean = float(np.mean(fp_deltas)) if fp_deltas else float("nan")
    directional_pass = bool(
        math.isfinite(fn_mean)
        and math.isfinite(fp_mean)
        and math.isfinite(separation_auc)
        and fn_mean > 0.0
        and fp_mean < 0.0
        and separation_auc >= float(directional_auc_threshold)
    )
    changed = sum(
        int(row["clean_prediction_index"]) != int(row[prediction_key]) for row in rows
    )
    return {
        "view": view,
        "support": int(len(rows)),
        "changed": int(changed),
        "corrections": int(corrections),
        "harms": int(harms),
        "neutral_changes": int(max(0, changed - corrections - harms)),
        "clean_correct_eligible": int(len(clean_correct)),
        "clean_correct_retained": int(
            sum(int(row[prediction_key]) == int(row["target_index"]) for row in clean_correct)
        ),
        "clean_correct_retention": float(
            sum(int(row[prediction_key]) == int(row["target_index"]) for row in clean_correct)
            / max(1, len(clean_correct))
        ),
        "class1_tp_eligible": int(len(class1_tp)),
        "class1_tp_retained": int(
            sum(int(row[prediction_key]) == FOCUS_CLASS for row in class1_tp)
        ),
        "class1_tp_retention": float(
            sum(int(row[prediction_key]) == FOCUS_CLASS for row in class1_tp)
            / max(1, len(class1_tp))
        ),
        "class1_fn_eligible": int(len(class1_fn)),
        "class1_fn_rescued": int(outcomes.get("fn_rescue", 0)),
        "class1_tp_broken": int(outcomes.get("tp_break", 0)),
        "class1_fp_eligible": int(len(negative_fp)),
        "class1_fp_removed": int(outcomes.get("fp_remove", 0)),
        "class1_fp_creation_eligible": int(len(negative_nonfp)),
        "class1_fp_created": int(outcomes.get("fp_create", 0)),
        "class1_fn_mean_delta_p1": fn_mean,
        "class1_fp_mean_delta_p1": fp_mean,
        "class1_fn_vs_fp_delta_auc": separation_auc,
        "directional_auc_threshold": float(directional_auc_threshold),
        "directional_pass": directional_pass,
        "outcomes": dict(sorted(outcomes.items())),
    }


def assign_source_grouped_folds(
    rows: Sequence[Mapping[str, object]],
    *,
    folds: int = FOLDS,
    seed: int = SEED,
) -> Dict[int, int]:
    if int(folds) < 2:
        raise ValueError("folds must be at least two")
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if len(set(sample_indices.tolist())) != len(sample_indices):
        raise ValueError("sample indices must be unique")
    labels = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row["source_stem"]) for row in rows], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    assignments: Dict[int, int] = {}
    for fold, (_, holdout) in enumerate(
        splitter.split(np.zeros(len(rows), dtype=np.float32), labels, groups)
    ):
        for row_index in holdout.tolist():
            assignments[int(sample_indices[row_index])] = int(fold)
    if len(assignments) != len(rows):
        raise RuntimeError("source-grouped fold assignment is incomplete")
    source_folds: Dict[str, set[int]] = {}
    for row in rows:
        source_folds.setdefault(str(row["source_stem"]), set()).add(
            assignments[int(row["sample_index"])]
        )
    if any(len(values) != 1 for values in source_folds.values()):
        raise RuntimeError("a source group crosses readiness folds")
    return assignments


def _class1_metrics(metrics: Mapping[str, object]) -> Mapping[str, object]:
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, list):
        return {}
    for row in per_class:
        if isinstance(row, Mapping) and int(row.get("class_index", -1)) == FOCUS_CLASS:
            return row
    return {}


def _gate_check(name: str, value: object, requirement: str, passed: bool) -> Dict[str, object]:
    return {
        "name": name,
        "value": value,
        "requirement": requirement,
        "passed": bool(passed),
    }


def assess_smoke_permission(
    *,
    preflight: bool,
    checkpoint_sha256: str,
    train_support: int,
    val_support: int,
    source_overlap_count: int,
    train_pairing: Mapping[str, object],
    val_pairing: Mapping[str, object],
    numerical: Mapping[str, object],
    clean_val_metrics: Mapping[str, object],
    same_val_metrics: Mapping[str, object],
    train_same: Mapping[str, object],
    val_same: Mapping[str, object],
    val_cross: Mapping[str, object],
    fold_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    clean_class1 = _class1_metrics(clean_val_metrics)
    same_class1 = _class1_metrics(same_val_metrics)
    stable_folds = sum(bool(row.get("directional_pass", False)) for row in fold_rows)
    pairing_complete = all(
        int(stats.get(key, -1)) == int(stats.get("support", 0))
        for stats in (train_pairing, val_pairing)
        for key in (
            "same_label_matches",
            "same_source_distinct",
            "cross_label_matches",
            "cross_source_distinct",
        )
    )
    checks = [
        _gate_check("full_run", not preflight, "preflight=false", not preflight),
        _gate_check(
            "checkpoint_sha256",
            checkpoint_sha256,
            EXPECTED_CHECKPOINT_SHA256,
            str(checkpoint_sha256).lower() == EXPECTED_CHECKPOINT_SHA256,
        ),
        _gate_check(
            "train_support",
            int(train_support),
            str(EXPECTED_TRAIN_COUNT),
            int(train_support) == EXPECTED_TRAIN_COUNT,
        ),
        _gate_check(
            "val_support",
            int(val_support),
            str(EXPECTED_VAL_COUNT),
            int(val_support) == EXPECTED_VAL_COUNT,
        ),
        _gate_check(
            "train_val_source_overlap",
            int(source_overlap_count),
            "0",
            int(source_overlap_count) == 0,
        ),
        _gate_check(
            "pairing_complete",
            pairing_complete,
            "all label/source checks equal support",
            pairing_complete,
        ),
        _gate_check(
            "validation_peers_train_only",
            val_pairing.get("peer_split"),
            "train",
            str(val_pairing.get("peer_split")) == "train",
        ),
        _gate_check(
            "identity_reconstruction",
            float(numerical.get("identity_max_error_rgb", float("inf"))),
            "<=1e-5",
            float(numerical.get("identity_max_error_rgb", float("inf"))) <= 1e-5,
        ),
        _gate_check(
            "ifft_imaginary_residual",
            float(numerical.get("max_imaginary_residual", float("inf"))),
            "<=1e-5",
            float(numerical.get("max_imaginary_residual", float("inf"))) <= 1e-5,
        ),
        _gate_check(
            "finite_outputs",
            bool(numerical.get("finite", False)),
            "true",
            bool(numerical.get("finite", False)),
        ),
        _gate_check(
            "keeper_val_macro",
            float(clean_val_metrics.get("macro_f1", 0.0)),
            f"abs(delta from {EXPECTED_KEEPER_VAL_MACRO})<=0.001",
            abs(float(clean_val_metrics.get("macro_f1", 0.0)) - EXPECTED_KEEPER_VAL_MACRO)
            <= 0.001,
        ),
        _gate_check(
            "keeper_val_class1",
            float(clean_class1.get("f1", 0.0)),
            f"abs(delta from {EXPECTED_KEEPER_VAL_CLASS1})<=0.001",
            abs(float(clean_class1.get("f1", 0.0)) - EXPECTED_KEEPER_VAL_CLASS1)
            <= 0.001,
        ),
        _gate_check(
            "active_bounded_perturbation",
            float(numerical.get("same_mean_rgb_mae", 0.0)),
            "1/255<=mean_rgb_mae<=0.25",
            1.0 / 255.0
            <= float(numerical.get("same_mean_rgb_mae", 0.0))
            <= 0.25,
        ),
        _gate_check(
            "bounded_clipping",
            float(numerical.get("same_clip_fraction", 1.0)),
            "<=0.05",
            float(numerical.get("same_clip_fraction", 1.0)) <= 0.05,
        ),
        _gate_check(
            "val_clean_correct_retention",
            float(val_same.get("clean_correct_retention", 0.0)),
            ">=0.95",
            float(val_same.get("clean_correct_retention", 0.0)) >= 0.95,
        ),
        _gate_check(
            "val_class1_tp_retention",
            float(val_same.get("class1_tp_retention", 0.0)),
            ">=0.95",
            float(val_same.get("class1_tp_retention", 0.0)) >= 0.95,
        ),
        _gate_check(
            "same_vs_cross_clean_retention",
            float(val_same.get("clean_correct_retention", 0.0))
            - float(val_cross.get("clean_correct_retention", 0.0)),
            ">=0",
            float(val_same.get("clean_correct_retention", 0.0))
            >= float(val_cross.get("clean_correct_retention", 0.0)),
        ),
        _gate_check(
            "same_vs_cross_class1_tp_retention",
            float(val_same.get("class1_tp_retention", 0.0))
            - float(val_cross.get("class1_tp_retention", 0.0)),
            ">=0",
            float(val_same.get("class1_tp_retention", 0.0))
            >= float(val_cross.get("class1_tp_retention", 0.0)),
        ),
        _gate_check(
            "val_fn_delta_p1",
            float(val_same.get("class1_fn_mean_delta_p1", float("nan"))),
            ">0",
            float(val_same.get("class1_fn_mean_delta_p1", float("nan"))) > 0.0,
        ),
        _gate_check(
            "val_fp_delta_p1",
            float(val_same.get("class1_fp_mean_delta_p1", float("nan"))),
            "<0",
            float(val_same.get("class1_fp_mean_delta_p1", float("nan"))) < 0.0,
        ),
        _gate_check(
            "val_fn_fp_delta_auc",
            float(val_same.get("class1_fn_vs_fp_delta_auc", float("nan"))),
            ">=0.60",
            float(val_same.get("class1_fn_vs_fp_delta_auc", float("nan"))) >= 0.60,
        ),
        _gate_check(
            "val_recall_safety",
            {
                "rescued": int(val_same.get("class1_fn_rescued", 0)),
                "broken": int(val_same.get("class1_tp_broken", 0)),
            },
            "rescued>=broken",
            int(val_same.get("class1_fn_rescued", 0))
            >= int(val_same.get("class1_tp_broken", 0)),
        ),
        _gate_check(
            "val_fp_safety",
            {
                "removed": int(val_same.get("class1_fp_removed", 0)),
                "created": int(val_same.get("class1_fp_created", 0)),
            },
            "removed>=created",
            int(val_same.get("class1_fp_removed", 0))
            >= int(val_same.get("class1_fp_created", 0)),
        ),
        _gate_check(
            "val_same_class1_f1",
            float(same_class1.get("f1", 0.0)),
            ">=clean class1 f1",
            float(same_class1.get("f1", 0.0)) >= float(clean_class1.get("f1", 0.0)),
        ),
        _gate_check(
            "train_aggregate_direction",
            bool(train_same.get("directional_pass", False)),
            "true",
            bool(train_same.get("directional_pass", False)),
        ),
        _gate_check(
            "train_fold_direction",
            stable_folds,
            f">=4/{FOLDS}",
            len(fold_rows) == FOLDS and stable_folds >= 4,
        ),
    ]
    passed = sum(bool(check["passed"]) for check in checks)
    return {
        "smoke_permission": passed == len(checks),
        "full_train_permission": False,
        "passed": int(passed),
        "total": int(len(checks)),
        "failed_checks": [
            str(check["name"]) for check in checks if not bool(check["passed"])
        ],
        "checks": checks,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    normalized = [dict(row) for row in rows]
    if not normalized:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in normalized:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def _tensor_to_pil(rgb: Tensor, size: int) -> Image.Image:
    array = (
        rgb.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array).resize(
        (int(size), int(size)), Image.Resampling.LANCZOS
    )


def _select_contact_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    limit: int = 12,
) -> List[Mapping[str, object]]:
    selected: List[Mapping[str, object]] = []
    selected_indices = set()
    for outcome in ("fn_rescue", "tp_break", "fp_remove", "fp_create", "correction", "harm"):
        candidates = sorted(
            (row for row in rows if str(row["same_outcome"]) == outcome),
            key=lambda row: (-abs(float(row["delta_p1_same"])), int(row["sample_index"])),
        )
        for row in candidates[:2]:
            index = int(row["sample_index"])
            if index not in selected_indices:
                selected.append(row)
                selected_indices.add(index)
    for row in sorted(
        rows,
        key=lambda value: (
            -abs(float(value["delta_p1_same"])),
            int(value["sample_index"]),
        ),
    ):
        if len(selected) >= int(limit):
            break
        index = int(row["sample_index"])
        if index not in selected_indices:
            selected.append(row)
            selected_indices.add(index)
    return selected[: int(limit)]


def build_contact_sheet(
    *,
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
    target_dataset: Dataset,
    peer_dataset: Dataset,
    mean: Sequence[float],
    std: Sequence[float],
) -> int:
    selected = _select_contact_rows(rows)
    if not selected:
        return 0
    panel_size = 176
    panel_header = 22
    row_header = 30
    gap = 6
    columns = 5
    width = columns * panel_size + (columns - 1) * gap
    row_height = row_header + panel_header + panel_size + gap
    canvas = Image.new("RGB", (width, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    panel_names = ("target", "same peer", "same mix", "cross peer", "cross mix")
    for row_number, row in enumerate(selected):
        target_image, _, _ = _classification_item(target_dataset, int(row["sample_index"]))
        same_image, _, _ = _classification_item(peer_dataset, int(row["same_peer_index"]))
        cross_image, _, _ = _classification_item(peer_dataset, int(row["cross_peer_index"]))
        target_rgb = _unnormalize(target_image.unsqueeze(0), mean, std)
        same_peer_rgb = _unnormalize(same_image.unsqueeze(0), mean, std)
        cross_peer_rgb = _unnormalize(cross_image.unsqueeze(0), mean, std)
        same_mix, _ = fourier_amplitude_mix_rgb(target_rgb, same_peer_rgb)
        cross_mix, _ = fourier_amplitude_mix_rgb(target_rgb, cross_peer_rgb)
        panels = (
            target_rgb[0],
            same_peer_rgb[0],
            same_mix[0],
            cross_peer_rgb[0],
            cross_mix[0],
        )
        top = row_number * row_height
        title = (
            f"{row['split']}:{row['sample_index']} y={row['target_index']} "
            f"pred {row['clean_prediction_index']}->{row['same_prediction_index']} "
            f"{row['same_outcome']} dp1={float(row['delta_p1_same']):+.4f}"
        )
        draw.text((2, top + 3), title, fill="black")
        for column, (name, panel) in enumerate(zip(panel_names, panels)):
            left = column * (panel_size + gap)
            draw.text((left + 2, top + row_header + 3), name, fill="black")
            image = _tensor_to_pil(panel, panel_size)
            canvas.paste(image, (left, top + row_header + panel_header))
    canvas.save(output_path, format="PNG", optimize=True)
    return int(len(selected))


def _public_split(result: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in result.items() if key != "rows"}


def _aggregate_numerical(
    train: Mapping[str, object],
    val: Mapping[str, object],
) -> Dict[str, object]:
    train_num = train["numerical"]
    val_num = val["numerical"]
    if not isinstance(train_num, Mapping) or not isinstance(val_num, Mapping):
        raise TypeError("split numerical summaries are invalid")
    train_support = int(train["support"])
    val_support = int(val["support"])
    total = max(1, train_support + val_support)

    def weighted(key: str) -> float:
        return float(
            (
                float(train_num.get(key, 0.0)) * train_support
                + float(val_num.get(key, 0.0)) * val_support
            )
            / total
        )

    return {
        "identity_max_error_rgb": max(
            float(train_num["identity_max_error_rgb"]),
            float(val_num["identity_max_error_rgb"]),
        ),
        "max_imaginary_residual": max(
            float(train_num["max_imaginary_residual"]),
            float(val_num["max_imaginary_residual"]),
        ),
        "finite": bool(train_num["finite"]) and bool(val_num["finite"]),
        "same_clip_fraction": weighted("same_clip_fraction"),
        "cross_clip_fraction": weighted("cross_clip_fraction"),
        "same_mean_rgb_mae": weighted("same_mean_rgb_mae"),
        "same_mean_object_rgb_mae": weighted("same_mean_object_rgb_mae"),
        "same_mean_background_rgb_mae": weighted("same_mean_background_rgb_mae"),
        "cross_mean_rgb_mae": weighted("cross_mean_rgb_mae"),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1:
        raise ValueError("batch-size must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num-workers must be non-negative")
    capped = int(args.max_train_samples) > 0 or int(args.max_val_samples) > 0
    if capped and not bool(args.allow_preflight):
        raise ValueError("sample caps require --allow-preflight")

    output_dir = Path(args.output_dir).resolve()
    data_path = Path(args.data).resolve()
    if _is_relative_to(output_dir, data_path.parent):
        raise ValueError("output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)
    set_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "readiness protocol is locked to the keeper checkpoint hash: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        raise ValueError("Fourier readiness is locked to bbox-aware yolo_f")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if class_names != list(data_spec.class_names):
        raise ValueError("checkpoint and dataset class orders differ")
    model = build_model_from_checkpoint(dict(checkpoint), num_classes=len(class_names))
    model.to(device=device, dtype=torch.float32)
    model.eval()
    mean, std = checkpoint_input_normalization(checkpoint)
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    transform = _build_eval_transform_from_checkpoint(dict(checkpoint), image_size=image_size)
    datasets = {
        split: _build_classification_dataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            checkpoint=dict(checkpoint),
        )
        for split in ("train", "val")
    }
    labels = {split: list(datasets[split].labels()) for split in datasets}
    paths = {
        split: [str(path) for path in datasets[split].sample_paths()]
        for split in datasets
    }
    for split in datasets:
        if len(labels[split]) != len(datasets[split]) or len(paths[split]) != len(datasets[split]):
            raise ValueError(f"{split} label/path coverage is incomplete")
    caps = {"train": int(args.max_train_samples), "val": int(args.max_val_samples)}
    target_indices = {
        split: list(range(min(len(datasets[split]), caps[split])))
        if caps[split] > 0
        else list(range(len(datasets[split])))
        for split in datasets
    }
    plans: Dict[str, List[Dict[str, object]]] = {}
    pairing: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val"):
        plans[split], pairing[split] = build_peer_plan(
            target_split=split,
            peer_split="train",
            target_indices=target_indices[split],
            target_labels=labels[split],
            target_paths=paths[split],
            peer_labels=labels["train"],
            peer_paths=paths["train"],
            seed=int(args.seed),
        )

    bbox_token_prior_source = str(
        checkpoint.get("model_config", {}).get("bbox_token_prior_source", "bbox")
        or "bbox"
    )
    split_results = {
        split: collect_split(
            split=split,
            target_dataset=datasets[split],
            peer_dataset=datasets["train"],
            plan=plans[split],
            model=model,
            class_names=class_names,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        for split in ("train", "val")
    }
    train_rows = split_results["train"]["rows"]
    val_rows = split_results["val"]["rows"]
    if not isinstance(train_rows, list) or not isinstance(val_rows, list):
        raise TypeError("split rows are invalid")

    fold_rows: List[Dict[str, object]] = []
    if not capped:
        assignments = assign_source_grouped_folds(train_rows, folds=FOLDS, seed=int(args.seed))
        for row in train_rows:
            row["readiness_fold"] = int(assignments[int(row["sample_index"])])
        for fold in range(FOLDS):
            subset = [row for row in train_rows if int(row["readiness_fold"]) == fold]
            summary = summarize_view(subset, view="same", directional_auc_threshold=0.55)
            summary["fold"] = int(fold)
            summary["source_groups"] = int(len({str(row["source_stem"]) for row in subset}))
            fold_rows.append(summary)
    else:
        for row in train_rows:
            row["readiness_fold"] = -1

    view_summaries = {
        split: {
            view: summarize_view(
                split_results[split]["rows"],
                view=view,
                directional_auc_threshold=0.55 if split == "train" else 0.60,
            )
            for view in ("same", "cross")
        }
        for split in ("train", "val")
    }
    train_sources = {_source_stem(path) for path in paths["train"]}
    val_sources = {_source_stem(path) for path in paths["val"]}
    source_overlap = sorted(train_sources & val_sources)
    numerical = _aggregate_numerical(split_results["train"], split_results["val"])
    gate = assess_smoke_permission(
        preflight=capped,
        checkpoint_sha256=checkpoint_sha,
        train_support=int(split_results["train"]["support"]),
        val_support=int(split_results["val"]["support"]),
        source_overlap_count=len(source_overlap),
        train_pairing=pairing["train"],
        val_pairing=pairing["val"],
        numerical=numerical,
        clean_val_metrics=split_results["val"]["metrics"]["clean"],
        same_val_metrics=split_results["val"]["metrics"]["same"],
        train_same=view_summaries["train"]["same"],
        val_same=view_summaries["val"]["same"],
        val_cross=view_summaries["val"]["cross"],
        fold_rows=fold_rows,
    )

    _write_csv(output_dir / "train_peer_plan.csv", plans["train"])
    _write_csv(output_dir / "val_peer_plan.csv", plans["val"])
    _write_csv(output_dir / "train_fourier_rows.csv", train_rows)
    _write_csv(output_dir / "val_fourier_rows.csv", val_rows)
    _write_csv(output_dir / "train_fold_summaries.csv", fold_rows)
    contact_count = build_contact_sheet(
        output_path=output_dir / "val_contact_sheet.png",
        rows=val_rows,
        target_dataset=datasets["val"],
        peer_dataset=datasets["train"],
        mean=mean,
        std=std,
    )

    summary = {
        "mode": "same_class_fourier_amplitude_mix_readiness",
        "protocol_locked": True,
        "preflight": bool(capped),
        "test_split_used": False,
        "raw_dataset_modified": False,
        "model_fit": False,
        "checkpoint_written": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "data": str(data_path),
        "device": str(device),
        "class_names": class_names,
        "image_size": image_size,
        "bbox_token_prior_source": bbox_token_prior_source,
        "protocol": {
            "fact_paper": FACT_PAPER_URL,
            "fact_official_code": FACT_CODE_URL,
            "fda_paper": FDA_PAPER_URL,
            "fda_official_code": FDA_CODE_URL,
            "rgb_before_normalization": True,
            "mix_lambda": MIX_LAMBDA,
            "spectrum_ratio": SPECTRUM_RATIO,
            "phase_owner": "target",
            "same_class_peer": True,
            "source_distinct_peer": True,
            "validation_peer_split": "train",
            "cross_class_control_map": CROSS_CLASS_MAP,
            "cross_class_control_training_allowed": False,
            "parameter_sweep": False,
            "folds": FOLDS,
            "seed": int(args.seed),
        },
        "source_overlap": {"count": len(source_overlap), "examples": source_overlap[:10]},
        "pairing": pairing,
        "splits": {
            split: _public_split(split_results[split]) for split in ("train", "val")
        },
        "view_summaries": view_summaries,
        "train_fold_summaries": fold_rows,
        "numerical": numerical,
        "contact_sheet_rows": int(contact_count),
        "gate": gate,
        "artifact_manifest_path": "artifact_manifest.json",
    }
    json_dump(output_dir / "summary.json", summary)
    clean_class1 = _class1_metrics(split_results["val"]["metrics"]["clean"])
    same_class1 = _class1_metrics(split_results["val"]["metrics"]["same"])
    readme = [
        "# Same-Class Fourier Amplitude Mix Readiness",
        "",
        "- Full train/validation audit; no test, fit, checkpoint, or raw-data edit.",
        (
            f"- Clean val macro/class1: `{float(split_results['val']['metrics']['clean']['macro_f1']):.6f}/"
            f"{float(clean_class1.get('f1', 0.0)):.6f}`."
        ),
        (
            f"- Same-class Fourier val macro/class1: "
            f"`{float(split_results['val']['metrics']['same']['macro_f1']):.6f}/"
            f"{float(same_class1.get('f1', 0.0)):.6f}`."
        ),
        (
            f"- Same-class val FN/FP mean delta p1: "
            f"`{float(view_summaries['val']['same']['class1_fn_mean_delta_p1']):+.6f}/"
            f"{float(view_summaries['val']['same']['class1_fp_mean_delta_p1']):+.6f}`."
        ),
        (
            f"- Smoke permission: `{gate['smoke_permission']}` "
            f"({gate['passed']}/{gate['total']} checks)."
        ),
        "",
        "Cross-class views are destructive controls only and are never training candidates.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    _write_artifact_manifest(
        output_dir,
        mode="same_class_fourier_amplitude_mix_readiness_evidence_manifest",
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _sha256,
    _write_artifact_manifest,
)
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification
from trkh.tools.probe_interior_second_order_readiness import (
    build_fixed_orthogonal_projection,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _effective_rank,
)


SEED = 20260711


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed low-rank head plus top-attention patch-token caches for "
            "a train/validation-only PWCA readiness proxy. Test is forbidden."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--projection-rank", type=int, default=64)
    parser.add_argument("--top-patches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_top_attention_tokens(
    patches: Tensor,
    attention: Tensor,
    patch_indices: Tensor,
    valid_mask: Tensor,
    *,
    top_k: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if patches.ndim != 3 or attention.ndim != 2:
        raise ValueError("patches/attention must be [B,N,D]/[B,N]")
    if tuple(patches.shape[:2]) != tuple(attention.shape):
        raise ValueError("patches and attention rows must align")
    if tuple(patch_indices.shape) != tuple(attention.shape):
        raise ValueError("patch_indices must align with attention")
    if tuple(valid_mask.shape) != tuple(attention.shape):
        raise ValueError("valid_mask must align with attention")
    top_k = int(top_k)
    if top_k <= 0 or top_k > int(attention.size(1)):
        raise ValueError("top_k must be within the patch-token count")
    valid = valid_mask.to(device=attention.device, dtype=torch.bool)
    valid_count = valid.sum(dim=1)
    if bool((valid_count < top_k).any().item()):
        raise ValueError(
            f"Every row must have at least {top_k} valid patches; "
            f"minimum={int(valid_count.min().item())}"
        )
    masked_attention = attention.float().masked_fill(~valid, float("-inf"))
    top = torch.topk(masked_attention, k=top_k, dim=1, largest=True, sorted=True)
    selected_tokens = patches.gather(
        1,
        top.indices.unsqueeze(-1).expand(-1, -1, int(patches.size(2))),
    )
    selected_patch_indices = patch_indices.gather(1, top.indices)
    selected_attention = attention.float().gather(1, top.indices)
    total_attention = attention.float().masked_fill(~valid, 0.0).sum(dim=1)
    selected_mass = selected_attention.sum(dim=1) / total_attention.clamp_min(1e-8)
    return selected_tokens, selected_patch_indices, selected_attention, selected_mass


def selected_bbox_fraction(
    selected_patch_indices: Tensor,
    crop_bbox: Tensor,
    *,
    grid_size: Tuple[int, int],
) -> Tensor:
    if selected_patch_indices.ndim != 2 or crop_bbox.ndim != 2:
        raise ValueError("selected indices and crop_bbox must be two-dimensional")
    grid_height, grid_width = [max(1, int(value)) for value in grid_size]
    indices = selected_patch_indices.to(dtype=torch.long)
    x = (indices.remainder(grid_width).float() + 0.5) / float(grid_width)
    y = (indices.div(grid_width, rounding_mode="floor").float() + 0.5) / float(
        grid_height
    )
    bbox = crop_bbox[:, :4].to(device=x.device, dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = bbox.unbind(dim=1)
    inside = (
        (x - cx[:, None]).abs() <= 0.5 * width[:, None]
    ) & ((y - cy[:, None]).abs() <= 0.5 * height[:, None])
    return inside.float().mean(dim=1)


def _overlay_selected_tokens(
    rgb: np.ndarray,
    selected_indices: np.ndarray,
    selected_attention: np.ndarray,
    *,
    grid_size: Tuple[int, int],
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    grid_height, grid_width = [max(1, int(value)) for value in grid_size]
    heat = np.zeros((grid_height, grid_width), dtype=np.float32)
    weights = np.asarray(selected_attention, dtype=np.float32)
    weights = weights / max(float(weights.max(initial=0.0)), 1e-8)
    for patch_index, weight in zip(selected_indices, weights):
        row = int(patch_index) // grid_width
        column = int(patch_index) % grid_width
        if 0 <= row < grid_height and 0 <= column < grid_width:
            heat[row, column] = max(float(heat[row, column]), float(weight))
    heat_image = Image.fromarray((heat * 255.0).round().astype(np.uint8)).resize(
        (width, height),
        getattr(Image, "Resampling", Image).NEAREST,
    )
    alpha = np.asarray(heat_image, dtype=np.float32)[..., None] / 255.0 * 0.52
    color = np.asarray([245.0, 45.0, 35.0], dtype=np.float32)
    overlay = image.astype(np.float32) * (1.0 - alpha) + color * alpha
    return overlay.clip(0.0, 255.0).round().astype(np.uint8)


def _write_preview(
    path: Path,
    preview: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> None:
    tile = 192
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index in range(len(class_names)):
        if class_index not in preview:
            continue
        rgb, overlay = preview[class_index]
        row = Image.new("RGB", (tile * 2, tile), color=(255, 255, 255))
        row.paste(Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR), (0, 0))
        row.paste(
            Image.fromarray(overlay).resize((tile, tile), resampling.BILINEAR),
            (tile, 0),
        )
        rows.append(row)
        manifest.append(
            {
                "row": len(rows) - 1,
                "class_index": int(class_index),
                "class_name": str(class_names[class_index]),
                "columns": ["keeper_object_crop", "top_attention_patch_tokens"],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 2, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(data.mean()) if data.size else 0.0,
        "p01": float(np.quantile(data, 0.01)) if data.size else 0.0,
        "p05": float(np.quantile(data, 0.05)) if data.size else 0.0,
        "p50": float(np.quantile(data, 0.50)) if data.size else 0.0,
        "p95": float(np.quantile(data, 0.95)) if data.size else 0.0,
        "p99": float(np.quantile(data, 0.99)) if data.size else 0.0,
    }


def _extract_split(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    projection: Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    top_patches: int,
    split: str,
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_classification,
        persistent_workers=bool(int(workers) > 0),
    )
    token_batches: List[np.ndarray] = []
    patch_index_batches: List[np.ndarray] = []
    attention_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    sample_index_batches: List[np.ndarray] = []
    selected_mass_batches: List[np.ndarray] = []
    selected_bbox_batches: List[np.ndarray] = []
    paths: List[str] = []
    preview: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = (
        [str(path) for path in dataset_paths_fn()]
        if callable(dataset_paths_fn)
        else []
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    projection_device = projection.to(device=device, dtype=torch.float32)
    seen = 0
    grid_size_seen: Optional[Tuple[int, int]] = None
    model.eval()

    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"pwca-token-cache-{split}", dynamic_ncols=True)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                raise ValueError("PWCA token extraction requires tensor metadata")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            crop_bbox = metadata.get("crop_bbox")
            if not torch.is_tensor(crop_bbox):
                raise ValueError("crop_bbox metadata is required")
            crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
            bbox = metadata.get("bbox")
            bbox = (
                bbox.to(device=device, dtype=torch.float32, non_blocking=True)
                if torch.is_tensor(bbox)
                else crop_bbox
            )
            image_mask = metadata.get("image_mask")
            image_mask = (
                image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
                if torch.is_tensor(image_mask)
                else None
            )
            with autocast_context(device, bool(amp)):
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=crop_bbox,
                )
                features["bbox"] = bbox
                output = (
                    model.forward_heads(features)
                    if hasattr(model, "forward_heads")
                    else classification_logits_from_features(model, features)
                )
                logits, _, _ = extract_detection_from_model_output(output)
                head = extract_head_input_from_features(model, features).float()
            patches = features.get("patches")
            patch_indices = features.get("patch_indices")
            attention = features.get("fine_grained_attention")
            if not torch.is_tensor(patches) or patches.ndim != 3:
                raise ValueError("Keeper features do not contain patch tokens")
            if not torch.is_tensor(patch_indices):
                raise ValueError("Keeper features do not contain patch indices")
            if not torch.is_tensor(attention):
                raise ValueError("Keeper does not expose fine-grained attention")
            key_padding_mask = features.get("memory_key_padding_mask")
            valid_mask = (
                ~key_padding_mask.to(device=device, dtype=torch.bool)
                if torch.is_tensor(key_padding_mask)
                else torch.ones_like(attention, dtype=torch.bool)
            )
            grid_size = tuple(int(value) for value in features.get("grid_size", (0, 0)))
            if len(grid_size) != 2 or min(grid_size) <= 0:
                raise ValueError("Keeper features do not contain a valid grid size")
            if grid_size_seen is not None and grid_size != grid_size_seen:
                raise ValueError("PWCA cache requires one fixed patch grid")
            grid_size_seen = grid_size
            selected, selected_indices, selected_attention, selected_mass = (
                select_top_attention_tokens(
                    patches,
                    attention,
                    patch_indices,
                    valid_mask,
                    top_k=int(top_patches),
                )
            )
            bbox_fraction = selected_bbox_fraction(
                selected_indices,
                crop_bbox,
                grid_size=grid_size,
            )
            proxy_tokens = torch.cat((head.unsqueeze(1), selected.float()), dim=1)
            proxy_tokens = F.layer_norm(proxy_tokens, (int(proxy_tokens.size(2)),))
            proxy_tokens = proxy_tokens @ projection_device
            token_batches.append(proxy_tokens.detach().half().cpu().numpy())
            patch_index_batches.append(
                selected_indices.detach().to(dtype=torch.int16).cpu().numpy()
            )
            attention_batches.append(selected_attention.detach().half().cpu().numpy())
            probability_batches.append(logits.detach().float().softmax(dim=1).cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            selected_mass_batches.append(selected_mass.detach().cpu().numpy())
            selected_bbox_batches.append(bbox_fraction.detach().cpu().numpy())

            batch_count = int(labels.numel())
            raw_paths = metadata.get("paths", [])
            batch_paths = (
                [str(path) for path in raw_paths]
                if isinstance(raw_paths, Sequence)
                else []
            )
            if dataset_paths and (
                len(batch_paths) != batch_count
                or not any(path.strip() for path in batch_paths)
            ):
                batch_paths = dataset_paths[seen : seen + batch_count]
            paths.extend(batch_paths)
            fallback_indices = np.arange(seen, seen + batch_count, dtype=np.int64)
            sample_index = metadata.get("sample_index")
            if torch.is_tensor(sample_index) and int(sample_index.numel()) == batch_count:
                sample_index_batches.append(
                    sample_index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                )
            else:
                sample_index_batches.append(fallback_indices)
            seen += batch_count

            for row_index, target in enumerate(labels.detach().cpu().tolist()):
                if int(target) in preview:
                    continue
                rgb = (
                    images[row_index].detach().float().cpu().unsqueeze(0) * std_tensor
                    + mean_tensor
                ).squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
                rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
                overlay = _overlay_selected_tokens(
                    rgb_u8,
                    selected_indices[row_index].detach().cpu().numpy(),
                    selected_attention[row_index].detach().cpu().numpy(),
                    grid_size=grid_size,
                )
                preview[int(target)] = (rgb_u8, overlay)

    labels_array = np.concatenate(label_batches).astype(np.int64, copy=False)
    if len(paths) != int(labels_array.size):
        raise ValueError("Extracted path count does not match labels")
    return {
        "tokens": np.concatenate(token_batches).astype(np.float16, copy=False),
        "patch_indices": np.concatenate(patch_index_batches).astype(np.int16, copy=False),
        "selected_attention": np.concatenate(attention_batches).astype(
            np.float16, copy=False
        ),
        "probabilities": np.concatenate(probability_batches).astype(
            np.float32, copy=False
        ),
        "labels": labels_array,
        "sample_index": np.concatenate(sample_index_batches).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "source_stem": np.asarray(
            [Path(path).stem.casefold() for path in paths],
            dtype=object,
        ),
        "selected_attention_mass": np.concatenate(selected_mass_batches).astype(
            np.float32, copy=False
        ),
        "selected_bbox_fraction": np.concatenate(selected_bbox_batches).astype(
            np.float32, copy=False
        ),
        "preview": preview,
        "grid_size": grid_size_seen,
    }


def _save_cache(
    path: Path,
    payload: Mapping[str, object],
    *,
    class_names: Sequence[str],
) -> None:
    np.savez(
        path,
        tokens=np.asarray(payload["tokens"], dtype=np.float16),
        patch_indices=np.asarray(payload["patch_indices"], dtype=np.int16),
        selected_attention=np.asarray(payload["selected_attention"], dtype=np.float16),
        probabilities=np.asarray(payload["probabilities"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
        paths=np.asarray(payload["paths"], dtype=object),
        source_stem=np.asarray(payload["source_stem"], dtype=object),
        selected_attention_mass=np.asarray(
            payload["selected_attention_mass"], dtype=np.float32
        ),
        selected_bbox_fraction=np.asarray(
            payload["selected_bbox_fraction"], dtype=np.float32
        ),
        grid_size=np.asarray(payload["grid_size"], dtype=np.int64),
        classes=np.asarray(class_names, dtype=object),
    )


def run_extraction(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.projection_rank) <= 0:
        raise ValueError("projection-rank must be positive")
    if int(args.top_patches) <= 0:
        raise ValueError("top-patches must be positive")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    embed_dim = int(getattr(model, "embed_dim", 0))
    if embed_dim <= 0:
        raise ValueError("Could not resolve keeper embed_dim")
    projection = build_fixed_orthogonal_projection(
        embed_dim,
        int(args.projection_rank),
        seed=SEED,
    )
    device = _resolve_device(str(args.device or ""))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    mean, std = checkpoint_input_normalization(checkpoint)
    start = time.perf_counter()
    class_names: List[str] = []
    split_payloads: Dict[str, Dict[str, object]] = {}
    max_samples = {
        "train": int(args.max_train_samples),
        "val": int(args.max_val_samples),
    }
    for split in ("train", "val"):
        dataset, split_class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples[split],
        )
        if class_names and list(split_class_names) != class_names:
            raise ValueError("Train and validation class orders differ")
        class_names = list(split_class_names)
        split_payloads[split] = _extract_split(
            model=model,
            dataset=dataset,
            projection=projection,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            top_patches=int(args.top_patches),
            split=split,
            mean=mean,
            std=std,
        )

    cache_paths: Dict[str, Path] = {}
    split_summaries: Dict[str, object] = {}
    for split, payload in split_payloads.items():
        cache_path = output_dir / f"{split}_pwca_token_cache.npz"
        _save_cache(cache_path, payload, class_names=class_names)
        cache_paths[split] = cache_path
        labels = np.asarray(payload["labels"], dtype=np.int64)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        tokens = np.asarray(payload["tokens"], dtype=np.float16)
        split_summaries[split] = {
            "samples": int(labels.size),
            "source_groups": int(np.unique(payload["source_stem"]).size),
            "class_counts": np.bincount(labels, minlength=len(class_names)).tolist(),
            "direct_keeper_metrics": _classification_metrics(
                labels,
                probabilities,
                class_names=class_names,
            ),
            "token_shape": list(tokens.shape),
            "projected_head_effective_rank": _effective_rank(
                tokens[:, 0].astype(np.float32),
                max_rows=2048,
            ),
            "selected_patch_effective_rank": _effective_rank(
                tokens[:, 1:].reshape(-1, int(tokens.shape[2])).astype(np.float32),
                max_rows=2048,
            ),
            "selected_attention_mass": _summary_stats(
                np.asarray(payload["selected_attention_mass"], dtype=np.float32)
            ),
            "selected_bbox_fraction": _summary_stats(
                np.asarray(payload["selected_bbox_fraction"], dtype=np.float32)
            ),
            "cache_size_bytes": int(cache_path.stat().st_size),
            "cache_sha256": _sha256(cache_path),
        }
        _write_preview(
            output_dir / f"{split}_top_patch_preview.png",
            payload["preview"],
            class_names,
        )

    train_groups = set(np.asarray(split_payloads["train"]["source_stem"], dtype=object))
    val_groups = set(np.asarray(split_payloads["val"]["source_stem"], dtype=object))
    summary = {
        "mode": "pwca_low_rank_top_patch_token_cache",
        "guardrail": (
            "Frozen keeper; deterministic eval transforms; fixed orthogonal projection and "
            "top-attention selection; train/validation only. No test, image-model training, "
            "checkpoint, trainable manifest, or raw-data edit."
        ),
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "class_names": class_names,
        "protocol": {
            "projection": "fixed_gaussian_qr_orthogonal",
            "projection_seed": SEED,
            "projection_rank": int(projection.shape[1]),
            "sequence": "projected_keeper_head_plus_top_fine_grained_attention_patches",
            "top_patches": int(args.top_patches),
            "token_dtype": "float16",
            "selection_sweep": False,
        },
        "splits": split_summaries,
        "train_val_source_overlap": len(train_groups.intersection(val_groups)),
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "image_model_trained": False,
        "model_written": False,
        "checkpoint_written": False,
        "trainable_manifest_written": False,
    }
    summary["artifact_manifest_path"] = str(
        (output_dir / "artifact_manifest.json").resolve()
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        "# PWCA Low-Rank Top-Patch Token Cache",
        "",
        f"- Train/validation rows: `{split_summaries['train']['samples']}/{split_summaries['val']['samples']}`",
        f"- Token sequence: `1 + {int(args.top_patches)}` at rank `{int(projection.shape[1])}`",
        f"- Train selected attention mass mean/p05: `{split_summaries['train']['selected_attention_mass']['mean']:.6f}/{split_summaries['train']['selected_attention_mass']['p05']:.6f}`",
        f"- Validation selected attention mass mean/p05: `{split_summaries['val']['selected_attention_mass']['mean']:.6f}/{split_summaries['val']['selected_attention_mass']['p05']:.6f}`",
        "",
        "The caches are diagnostic intermediates. They do not contain test rows or a model.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    _write_artifact_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_extraction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

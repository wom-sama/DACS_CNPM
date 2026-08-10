from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from trkh.models.model import build_model_from_checkpoint
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _extract_split_embeddings,
    _resolve_device,
)


RESEARCH_SOURCES = [
    {
        "name": "Constrained Mean Shift Using Distant Yet Related Neighbors",
        "url": "https://arxiv.org/abs/2112.04607",
        "note": "Use semantically constrained neighbors instead of pulling every same-label sample together.",
    },
    {
        "name": "PARTICLE: Part Discovery and Contrastive Learning for Fine-grained Recognition",
        "url": "https://arxiv.org/abs/2309.13822",
        "note": "Fine-grained recognition benefits from part/local variation rather than instance-only contrast.",
    },
    {
        "name": "Contrast to Divide",
        "url": "https://arxiv.org/abs/2103.13646",
        "note": "Self-supervised features can reduce noisy-label warm-up sensitivity.",
    },
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only local-neighbor teacher feature cache. The cache is "
            "keyed by yolo_f sample_index and can be passed to existing "
            "--teacher-feature-npz CRD/RKD losses without touching raw data."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-neighbor-rank", type=int, default=4)
    parser.add_argument("--max-neighbor-rank", type=int, default=24)
    parser.add_argument("--neighbors-per-sample", type=int, default=6)
    parser.add_argument("--blend-alpha", type=float, default=0.18)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--focus-blend-alpha", type=float, default=0.28)
    parser.add_argument("--boundary-rival-classes", type=str, default="0,2,4")
    parser.add_argument("--boundary-blend-alpha", type=float, default=0.22)
    parser.add_argument("--min-target-norm", type=float, default=0.10)
    return parser.parse_args(argv)


def _parse_int_set(text: str) -> set[int]:
    output: set[int] = set()
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            output.add(int(item))
    return output


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [N, D] feature array, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def build_local_neighbor_targets(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    probabilities: Optional[np.ndarray] = None,
    min_neighbor_rank: int = 4,
    max_neighbor_rank: int = 24,
    neighbors_per_sample: int = 6,
    blend_alpha: float = 0.18,
    focus_class_index: int = 1,
    focus_blend_alpha: float = 0.28,
    boundary_rival_classes: Optional[set[int]] = None,
    boundary_blend_alpha: float = 0.22,
    min_target_norm: float = 0.10,
) -> tuple[np.ndarray, Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    normalized = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if normalized.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Embedding/label length mismatch: {normalized.shape[0]} != {labels.shape[0]}"
        )
    sample_count, feature_dim = normalized.shape
    if sample_count <= 0 or feature_dim <= 0:
        raise ValueError("Cannot build local neighbor targets from an empty embedding set.")

    min_rank = max(1, int(min_neighbor_rank))
    max_rank = max(min_rank, int(max_neighbor_rank))
    neighbor_count = max(1, int(neighbors_per_sample))
    base_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    focus_alpha = float(np.clip(focus_blend_alpha, 0.0, 1.0))
    boundary_alpha = float(np.clip(boundary_blend_alpha, 0.0, 1.0))
    min_norm = max(1e-6, float(min_target_norm))
    rival_classes = set(boundary_rival_classes or {0, 2, 4})

    probabilities_array: Optional[np.ndarray] = None
    predictions = np.full((sample_count,), -1, dtype=np.int64)
    if probabilities is not None:
        probabilities_array = np.asarray(probabilities, dtype=np.float32)
        if probabilities_array.ndim == 2 and probabilities_array.shape[0] == sample_count:
            predictions = probabilities_array.argmax(axis=1).astype(np.int64)
        else:
            probabilities_array = None

    targets = normalized.copy()
    neighbor_counts = np.zeros((sample_count,), dtype=np.int64)
    blend_alphas = np.zeros((sample_count,), dtype=np.float32)
    fallback_rows = 0
    by_label: Dict[str, int] = Counter()
    by_reason: Dict[str, int] = Counter()
    by_top_neighbor_label: Dict[str, int] = Counter()

    for class_index in sorted(int(value) for value in np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == class_index)
        if class_indices.size <= 1:
            fallback_rows += int(class_indices.size)
            continue
        class_features = normalized[class_indices]
        sims = class_features.dot(class_features.T)
        np.fill_diagonal(sims, -np.inf)
        sorted_local = np.argsort(-sims, axis=1, kind="mergesort")
        for local_row, sample_index in enumerate(class_indices.tolist()):
            row_order = sorted_local[local_row]
            finite_order = row_order[np.isfinite(sims[local_row, row_order])]
            if finite_order.size == 0:
                fallback_rows += 1
                continue
            start = min(max(0, min_rank - 1), max(0, finite_order.size - 1))
            stop = min(finite_order.size, max_rank)
            window = finite_order[start:stop]
            if window.size < neighbor_count:
                window = finite_order[: min(finite_order.size, max(neighbor_count, window.size))]
            selected = window[: min(neighbor_count, window.size)]
            if selected.size == 0:
                fallback_rows += 1
                continue
            neighbor_indices = class_indices[selected]
            neighbor_mean = normalized[neighbor_indices].mean(axis=0)
            norm = float(np.linalg.norm(neighbor_mean))
            if not math.isfinite(norm) or norm < min_norm:
                fallback_rows += 1
                continue
            neighbor_mean = (neighbor_mean / max(norm, 1e-12)).astype(np.float32, copy=False)

            prediction = int(predictions[int(sample_index)])
            if int(class_index) == int(focus_class_index):
                alpha = focus_alpha
                reason = "focus_same_class_neighbor"
            elif prediction == int(focus_class_index) and int(class_index) in rival_classes:
                alpha = boundary_alpha
                reason = "boundary_predicted_focus_neighbor"
            else:
                alpha = base_alpha
                reason = "same_class_neighbor"
            blended = (1.0 - alpha) * normalized[int(sample_index)] + alpha * neighbor_mean
            blended_norm = float(np.linalg.norm(blended))
            if not math.isfinite(blended_norm) or blended_norm < min_norm:
                fallback_rows += 1
                continue
            targets[int(sample_index)] = (blended / max(blended_norm, 1e-12)).astype(np.float32, copy=False)
            neighbor_counts[int(sample_index)] = int(selected.size)
            blend_alphas[int(sample_index)] = float(alpha)
            by_label[str(class_index)] += 1
            by_reason[reason] += 1
            if selected.size > 0:
                by_top_neighbor_label[str(int(labels[int(neighbor_indices[0])]))] += 1

    target_norms = np.linalg.norm(targets.astype(np.float32), axis=1)
    cosine_to_source = np.sum(targets.astype(np.float32) * normalized.astype(np.float32), axis=1)
    summary = {
        "samples": int(sample_count),
        "feature_dim": int(feature_dim),
        "min_neighbor_rank": int(min_rank),
        "max_neighbor_rank": int(max_rank),
        "neighbors_per_sample": int(neighbor_count),
        "blend_alpha": float(base_alpha),
        "focus_class_index": int(focus_class_index),
        "focus_blend_alpha": float(focus_alpha),
        "boundary_rival_classes": sorted(int(value) for value in rival_classes),
        "boundary_blend_alpha": float(boundary_alpha),
        "fallback_rows": int(fallback_rows),
        "rows_with_neighbor_target": int((neighbor_counts > 0).sum()),
        "mean_neighbor_count": float(neighbor_counts[neighbor_counts > 0].mean()) if bool((neighbor_counts > 0).any()) else 0.0,
        "mean_blend_alpha": float(blend_alphas[blend_alphas > 0.0].mean()) if bool((blend_alphas > 0.0).any()) else 0.0,
        "mean_target_norm": float(target_norms.mean()) if target_norms.size else 0.0,
        "std_target_norm": float(target_norms.std()) if target_norms.size else 0.0,
        "mean_cosine_to_source": float(cosine_to_source.mean()) if cosine_to_source.size else 0.0,
        "min_cosine_to_source": float(cosine_to_source.min()) if cosine_to_source.size else 0.0,
        "by_label": dict(by_label),
        "by_reason": dict(by_reason),
        "by_top_neighbor_label": dict(by_top_neighbor_label),
    }
    return targets.astype(np.float32, copy=False), summary


def _dataset_paths(dataset, payload_paths: Sequence[str]) -> np.ndarray:
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    paths: List[str]
    if callable(sample_paths_fn):
        paths = [str(path) for path in sample_paths_fn()]
    else:
        paths = [str(path) for path in payload_paths]
    return np.asarray(paths, dtype=object)


def run(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if str(args.split).strip().lower() != "train":
        raise ValueError("Local neighbor feature cache is train-only; refusing non-train split.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    dataset, class_names = _build_dataset(
        data_yaml=Path(args.data),
        split=str(args.split),
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=int(args.max_samples),
    )
    payload = _extract_split_embeddings(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        amp=bool(args.amp),
        split=str(args.split),
    )
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    probabilities = np.asarray(payload.get("probabilities"), dtype=np.float32)
    targets, target_summary = build_local_neighbor_targets(
        embeddings,
        labels,
        probabilities=probabilities,
        min_neighbor_rank=int(args.min_neighbor_rank),
        max_neighbor_rank=int(args.max_neighbor_rank),
        neighbors_per_sample=int(args.neighbors_per_sample),
        blend_alpha=float(args.blend_alpha),
        focus_class_index=int(args.focus_class_index),
        focus_blend_alpha=float(args.focus_blend_alpha),
        boundary_rival_classes=_parse_int_set(str(args.boundary_rival_classes)),
        boundary_blend_alpha=float(args.boundary_blend_alpha),
        min_target_norm=float(args.min_target_norm),
    )
    sample_indices = np.arange(int(labels.shape[0]), dtype=np.int64)
    paths = _dataset_paths(dataset, payload.get("paths", []))
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        features=targets.astype(np.float32, copy=False),
        sample_index=sample_indices,
        labels=labels.astype(np.int64, copy=False),
        paths=paths,
        source_embeddings=_normalize_rows(embeddings).astype(np.float32, copy=False),
        class_names=np.asarray([str(name) for name in class_names], dtype=object),
    )
    summary_path = Path(args.summary_json) if args.summary_json is not None else output_npz.with_suffix(".summary.json")
    summary = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "output_npz": str(output_npz.resolve()),
        "summary_json": str(summary_path.resolve()),
        "split": str(args.split),
        "class_names": [str(name) for name in class_names],
        "class_counts": {str(index): int((labels == index).sum()) for index in range(len(class_names))},
        "cache": target_summary,
        "research_sources": RESEARCH_SOURCES,
        "leakage_guard": "Only the requested train split is embedded; non-train splits are refused.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    start = time.perf_counter()
    summary = run(args)
    summary["seconds"] = float(time.perf_counter() - start)
    Path(summary["summary_json"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_npz": summary["output_npz"],
                "samples": int(summary["cache"]["samples"]),
                "feature_dim": int(summary["cache"]["feature_dim"]),
                "rows_with_neighbor_target": int(summary["cache"]["rows_with_neighbor_target"]),
                "mean_cosine_to_source": float(summary["cache"]["mean_cosine_to_source"]),
                "seconds": round(float(summary["seconds"]), 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

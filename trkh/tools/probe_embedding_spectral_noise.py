from __future__ import annotations

import argparse
import csv
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
        "name": "FINE Samples for Learning with Noisy Labels",
        "url": "https://arxiv.org/abs/2102.11628",
        "note": "Uses latent representation eigen structure for label-noise filtering.",
    },
    {
        "name": "Fine-Grained Classification with Noisy Labels",
        "url": "https://arxiv.org/abs/2303.02404",
        "note": "Fine-grained noisy labels need discriminative representation, not only robust CE.",
    },
    {
        "name": "PASS: Peer-Agreement based Sample Selection",
        "url": "https://openresearch.surrey.ac.uk/esploro/outputs/preprint/PASS-Peer-Agreement-based-Sample-Selection-for/99783703102346",
        "note": "Feature/loss selection can confuse hard samples with noisy samples; use as diagnostic first.",
    },
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only FINE/spectral embedding diagnostic for the TRKH mango "
            "classifier. It extracts frozen checkpoint embeddings and scores "
            "class-local ambiguity; it does not modify the raw dataset."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--neighbors-k", type=int, default=9)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _rank01(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size <= 1:
        return np.zeros_like(array, dtype=np.float32)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    return (ranks / float(array.size - 1)).astype(np.float32)


def _safe_quantiles(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def _class_centroids(features: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    normalized = _normalize_rows(features)
    centroids = np.zeros((class_count, normalized.shape[1]), dtype=np.float32)
    global_centroid = normalized.mean(axis=0) if normalized.size else np.zeros((normalized.shape[1],), dtype=np.float32)
    for class_index in range(class_count):
        mask = labels == class_index
        centroids[class_index] = normalized[mask].mean(axis=0) if bool(mask.any()) else global_centroid
    return _normalize_rows(centroids)


def _neighbor_label_stats(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    chunk_size: int = 512,
) -> Dict[str, np.ndarray]:
    normalized = _normalize_rows(features)
    labels = np.asarray(labels, dtype=np.int64)
    sample_count = int(labels.shape[0])
    k = max(1, min(int(k), max(1, sample_count - 1)))
    same_fraction = np.zeros((sample_count,), dtype=np.float32)
    top_rival_label = np.full((sample_count,), -1, dtype=np.int64)
    top_rival_fraction = np.zeros((sample_count,), dtype=np.float32)
    mean_top_similarity = np.zeros((sample_count,), dtype=np.float32)

    for start in range(0, sample_count, max(1, int(chunk_size))):
        end = min(sample_count, start + chunk_size)
        sims = normalized[start:end].dot(normalized.T)
        row_indices = np.arange(end - start)
        sims[row_indices, np.arange(start, end)] = -np.inf
        if k < sample_count:
            top_indices = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            top_sims = np.take_along_axis(sims, top_indices, axis=1)
            order = np.argsort(-top_sims, axis=1)
            top_indices = np.take_along_axis(top_indices, order, axis=1)
            top_sims = np.take_along_axis(top_sims, order, axis=1)
        else:
            top_indices = np.argsort(-sims, axis=1)
            top_sims = np.take_along_axis(sims, top_indices, axis=1)
        neighbor_labels = labels[top_indices]
        current_labels = labels[start:end, None]
        same = neighbor_labels == current_labels
        same_fraction[start:end] = same.mean(axis=1).astype(np.float32)
        mean_top_similarity[start:end] = np.where(
            np.isfinite(top_sims),
            top_sims,
            0.0,
        ).mean(axis=1).astype(np.float32)
        for row_offset in range(end - start):
            rival_labels = neighbor_labels[row_offset][~same[row_offset]]
            if rival_labels.size == 0:
                continue
            counts = Counter(int(value) for value in rival_labels.tolist())
            label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            top_rival_label[start + row_offset] = int(label)
            top_rival_fraction[start + row_offset] = float(count / max(1, k))
    return {
        "neighbor_same_label_fraction": same_fraction,
        "neighbor_disagreement": (1.0 - same_fraction).astype(np.float32),
        "neighbor_top_rival_label": top_rival_label,
        "neighbor_top_rival_fraction": top_rival_fraction,
        "neighbor_mean_top_similarity": mean_top_similarity,
    }


def _spectral_scores_by_class(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    normalized = _normalize_rows(features)
    sample_count = int(labels.shape[0])
    alignment = np.zeros((sample_count,), dtype=np.float32)
    projection = np.zeros((sample_count,), dtype=np.float32)
    projection_z_abs = np.zeros((sample_count,), dtype=np.float32)
    singular_share = np.zeros((sample_count,), dtype=np.float32)

    for class_index in range(class_count):
        indices = np.flatnonzero(labels == class_index)
        if indices.size <= 1:
            continue
        class_features = normalized[indices]
        centered = class_features - class_features.mean(axis=0, keepdims=True)
        if not np.isfinite(centered).all() or float(np.linalg.norm(centered)) <= 1e-12:
            continue
        try:
            u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if singular_values.size == 0:
            continue
        pc = vt[0].astype(np.float32, copy=False)
        proj = centered.dot(pc)
        proj_std = float(np.std(proj))
        projection[indices] = proj.astype(np.float32, copy=False)
        projection_z_abs[indices] = (
            np.abs((proj - float(np.mean(proj))) / max(proj_std, 1e-12))
        ).astype(np.float32, copy=False)
        # Top left singular-vector magnitude is the FINE-style sample alignment
        # with the dominant class-local representation direction.
        alignment[indices] = (np.abs(u[:, 0]) * math.sqrt(float(indices.size))).astype(np.float32, copy=False)
        total_energy = float(np.square(singular_values).sum())
        share = float((singular_values[0] ** 2) / max(total_energy, 1e-12))
        singular_share[indices] = share
    return {
        "spectral_alignment": alignment,
        "spectral_projection": projection,
        "spectral_projection_z_abs": projection_z_abs,
        "spectral_top_energy_share": singular_share,
    }


def score_embedding_spectral_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    probabilities: Optional[np.ndarray] = None,
    base_predictions: Optional[np.ndarray] = None,
    neighbors_k: int = 9,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Embeddings and labels shape mismatch: embeddings={embeddings.shape}, labels={labels.shape}"
        )
    if labels.size == 0:
        raise ValueError("Cannot score an empty embedding set.")
    if int(class_count) <= int(labels.max(initial=0)):
        raise ValueError("class_count is smaller than a label value.")

    normalized = _normalize_rows(embeddings)
    centroids = _class_centroids(embeddings, labels, int(class_count))
    centroid_scores = normalized.dot(centroids.T)
    own_similarity = centroid_scores[np.arange(labels.shape[0]), labels]
    masked_centroid_scores = centroid_scores.copy()
    masked_centroid_scores[np.arange(labels.shape[0]), labels] = -np.inf
    rival_centroid_label = masked_centroid_scores.argmax(axis=1).astype(np.int64)
    rival_centroid_similarity = masked_centroid_scores[np.arange(labels.shape[0]), rival_centroid_label]
    centroid_margin = (own_similarity - rival_centroid_similarity).astype(np.float32, copy=False)
    centroid_distance = (1.0 - own_similarity).astype(np.float32, copy=False)

    neighbor_stats = _neighbor_label_stats(embeddings, labels, k=int(neighbors_k))
    spectral_stats = _spectral_scores_by_class(embeddings, labels, int(class_count))

    centroid_distance_rank = np.zeros_like(centroid_distance, dtype=np.float32)
    neighbor_disagreement_rank = np.zeros_like(centroid_distance, dtype=np.float32)
    spectral_outlier_rank = np.zeros_like(centroid_distance, dtype=np.float32)
    low_alignment_rank = np.zeros_like(centroid_distance, dtype=np.float32)
    for class_index in range(class_count):
        indices = np.flatnonzero(labels == class_index)
        if indices.size == 0:
            continue
        centroid_distance_rank[indices] = _rank01(centroid_distance[indices])
        neighbor_disagreement_rank[indices] = _rank01(neighbor_stats["neighbor_disagreement"][indices])
        spectral_outlier_rank[indices] = _rank01(spectral_stats["spectral_projection_z_abs"][indices])
        low_alignment_rank[indices] = _rank01(-spectral_stats["spectral_alignment"][indices])

    ambiguity_score = (
        0.40 * neighbor_disagreement_rank
        + 0.30 * centroid_distance_rank
        + 0.20 * spectral_outlier_rank
        + 0.10 * low_alignment_rank
    ).astype(np.float32, copy=False)

    output: Dict[str, np.ndarray] = {
        "centroid_own_similarity": own_similarity.astype(np.float32, copy=False),
        "centroid_rival_label": rival_centroid_label,
        "centroid_rival_similarity": rival_centroid_similarity.astype(np.float32, copy=False),
        "centroid_margin": centroid_margin,
        "centroid_distance": centroid_distance,
        "centroid_distance_rank": centroid_distance_rank,
        "neighbor_disagreement_rank": neighbor_disagreement_rank,
        "spectral_outlier_rank": spectral_outlier_rank,
        "spectral_low_alignment_rank": low_alignment_rank,
        "ambiguity_score": ambiguity_score,
        "fine_clean_score": (1.0 - ambiguity_score).astype(np.float32, copy=False),
        **neighbor_stats,
        **spectral_stats,
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float32)
        if probs.ndim == 2 and probs.shape[0] == labels.shape[0]:
            output["base_target_probability"] = probs[np.arange(labels.shape[0]), labels].astype(np.float32, copy=False)
            output["base_confidence"] = probs.max(axis=1).astype(np.float32, copy=False)
    if base_predictions is not None:
        predictions = np.asarray(base_predictions, dtype=np.int64)
        if predictions.shape[0] == labels.shape[0]:
            output["base_prediction"] = predictions
            output["base_error"] = (predictions != labels).astype(np.int64)
    return output


def _dataset_records(dataset, *, split: str) -> List[Dict[str, object]]:
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    label_fn = getattr(dataset, "labels", None)
    paths = [Path(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []
    labels = [int(value) for value in label_fn()] if callable(label_fn) else []
    samples = getattr(dataset, "samples", None)
    records: List[Dict[str, object]] = []
    sample_count = len(dataset)
    for sample_index in range(sample_count):
        image_path = paths[sample_index] if sample_index < len(paths) else Path("")
        target_index = labels[sample_index] if sample_index < len(labels) else -1
        source_stem = image_path.stem
        label_path = ""
        object_index = 0
        if isinstance(samples, Sequence) and sample_index < len(samples):
            sample = samples[sample_index]
            label_path_value = getattr(sample, "label_path", None)
            if label_path_value is not None:
                label_path = str(label_path_value)
                source_stem = Path(label_path_value).stem
            primary_object_index = getattr(sample, "primary_object_index", 0)
            try:
                object_index = int(primary_object_index)
            except (TypeError, ValueError):
                object_index = 0
        records.append(
            {
                "split": str(split),
                "sample_index": int(sample_index),
                "image_path": str(image_path),
                "label_path": str(label_path),
                "source_stem": str(source_stem),
                "object_index": int(object_index),
                "target_index": int(target_index),
            }
        )
    return records


def _class_summary(
    *,
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    top_fraction: float,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    class_count = len(class_names)
    output: Dict[str, object] = {}
    base_error = np.asarray(scores.get("base_error", np.zeros(labels.shape[0], dtype=np.int64)), dtype=np.int64)
    top_fraction = min(1.0, max(0.01, float(top_fraction)))
    for class_index in range(class_count):
        indices = np.flatnonzero(labels == class_index)
        if indices.size == 0:
            output[str(class_index)] = {
                "class_name": str(class_names[class_index]),
                "samples": 0,
            }
            continue
        class_scores = np.asarray(scores["ambiguity_score"], dtype=np.float32)[indices]
        top_count = max(1, int(math.ceil(float(indices.size) * top_fraction)))
        top_local = np.argsort(-class_scores, kind="mergesort")[:top_count]
        top_indices = indices[top_local]
        rival_labels = [
            int(value)
            for value in np.asarray(scores["neighbor_top_rival_label"], dtype=np.int64)[top_indices].tolist()
            if int(value) >= 0
        ]
        output[str(class_index)] = {
            "class_name": str(class_names[class_index]),
            "samples": int(indices.size),
            "top_fraction": float(top_fraction),
            "top_count": int(top_count),
            "base_error_rate": float(base_error[indices].mean()) if indices.size else 0.0,
            "top_risk_base_error_rate": float(base_error[top_indices].mean()) if top_indices.size else 0.0,
            "mean_ambiguity_score": float(np.mean(class_scores)),
            "top_risk_mean_ambiguity_score": float(np.mean(scores["ambiguity_score"][top_indices])),
            "mean_neighbor_same_label_fraction": float(np.mean(scores["neighbor_same_label_fraction"][indices])),
            "top_risk_mean_neighbor_same_label_fraction": float(
                np.mean(scores["neighbor_same_label_fraction"][top_indices])
            ),
            "mean_centroid_margin": float(np.mean(scores["centroid_margin"][indices])),
            "top_risk_mean_centroid_margin": float(np.mean(scores["centroid_margin"][top_indices])),
            "ambiguity_quantiles": _safe_quantiles(class_scores),
            "neighbor_top_rival_label_counts_top_risk": {
                str(label): int(count)
                for label, count in sorted(Counter(rival_labels).items(), key=lambda item: (-item[1], item[0]))
            },
        }
    return output


def _write_score_csv(
    path: Path,
    *,
    records: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "source_stem",
        "object_index",
        "image_path",
        "label_path",
        "target_index",
        "base_prediction",
        "base_error",
        "base_target_probability",
        "base_confidence",
        "ambiguity_score",
        "fine_clean_score",
        "neighbor_same_label_fraction",
        "neighbor_top_rival_label",
        "neighbor_top_rival_fraction",
        "centroid_own_similarity",
        "centroid_rival_label",
        "centroid_margin",
        "spectral_alignment",
        "spectral_projection_z_abs",
        "spectral_top_energy_share",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            row = {key: record.get(key, "") for key in fieldnames}
            row["target_index"] = int(labels[index])
            for key in fieldnames:
                if key in scores:
                    value = np.asarray(scores[key])[index]
                    row[key] = int(value) if np.asarray(scores[key]).dtype.kind in {"i", "u", "b"} else float(value)
            writer.writerow(row)


def _write_top_risk_csv(
    path: Path,
    *,
    records: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    class_index: Optional[int],
    limit: int,
) -> None:
    indices = np.arange(labels.shape[0])
    if class_index is not None:
        indices = indices[labels == int(class_index)]
    order = indices[np.argsort(-np.asarray(scores["ambiguity_score"], dtype=np.float32)[indices], kind="mergesort")]
    selected = order[: max(1, int(limit))]
    fieldnames = [
        "rank",
        "sample_index",
        "source_stem",
        "object_index",
        "image_path",
        "target_index",
        "base_prediction",
        "base_error",
        "ambiguity_score",
        "fine_clean_score",
        "neighbor_same_label_fraction",
        "neighbor_top_rival_label",
        "centroid_margin",
        "spectral_projection_z_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, index in enumerate(selected, start=1):
            record = records[int(index)]
            row = {
                "rank": int(rank),
                "sample_index": int(record.get("sample_index", index)),
                "source_stem": str(record.get("source_stem", "")),
                "object_index": int(record.get("object_index", 0)),
                "image_path": str(record.get("image_path", "")),
                "target_index": int(labels[int(index)]),
            }
            for key in fieldnames:
                if key in scores:
                    value = np.asarray(scores[key])[int(index)]
                    row[key] = int(value) if np.asarray(scores[key]).dtype.kind in {"i", "u", "b"} else float(value)
            writer.writerow(row)


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    records = _dataset_records(dataset, split=str(args.split))
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
    scores = score_embedding_spectral_noise(
        embeddings,
        labels,
        class_count=len(class_names),
        probabilities=np.asarray(payload.get("probabilities"), dtype=np.float32),
        base_predictions=np.asarray(payload.get("base_predictions"), dtype=np.int64),
        neighbors_k=int(args.neighbors_k),
    )
    if len(records) != int(labels.shape[0]):
        raise ValueError(f"Record/label length mismatch: {len(records)} != {labels.shape[0]}")

    score_csv = output_dir / f"{args.split}_spectral_noise_scores.csv"
    _write_score_csv(score_csv, records=records, labels=labels, scores=scores)
    _write_top_risk_csv(
        output_dir / f"{args.split}_top_risk_all_classes.csv",
        records=records,
        labels=labels,
        scores=scores,
        class_index=None,
        limit=200,
    )
    if len(class_names) > 1:
        _write_top_risk_csv(
            output_dir / f"{args.split}_top_risk_class1.csv",
            records=records,
            labels=labels,
            scores=scores,
            class_index=1,
            limit=200,
        )

    class_summary = _class_summary(
        labels=labels,
        scores=scores,
        class_names=class_names,
        top_fraction=float(args.top_fraction),
    )
    summary = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": str(args.split),
        "samples": int(labels.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "class_names": [str(name) for name in class_names],
        "class_counts": {
            str(index): int((labels == index).sum()) for index in range(len(class_names))
        },
        "neighbors_k": int(args.neighbors_k),
        "top_fraction": float(args.top_fraction),
        "leakage_guard": "Scores are fitted and reported on the requested split only; no test split is read.",
        "research_sources": RESEARCH_SOURCES,
        "class_summary": class_summary,
        "files": {
            "score_csv": str(score_csv),
            "top_risk_all_classes_csv": str(output_dir / f"{args.split}_top_risk_all_classes.csv"),
            "top_risk_class1_csv": str(output_dir / f"{args.split}_top_risk_class1.csv") if len(class_names) > 1 else "",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    start_time = time.perf_counter()
    summary = run_probe(args)
    summary["seconds"] = float(time.perf_counter() - start_time)
    (Path(args.output_dir) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    class1 = summary.get("class_summary", {}).get("1", {}) if isinstance(summary.get("class_summary"), Mapping) else {}
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "samples": int(summary.get("samples", 0)),
                "embedding_dim": int(summary.get("embedding_dim", 0)),
                "class1": {
                    "samples": int(class1.get("samples", 0) or 0),
                    "base_error_rate": float(class1.get("base_error_rate", 0.0) or 0.0),
                    "top_risk_base_error_rate": float(class1.get("top_risk_base_error_rate", 0.0) or 0.0),
                    "mean_neighbor_same_label_fraction": float(
                        class1.get("mean_neighbor_same_label_fraction", 0.0) or 0.0
                    ),
                    "top_risk_mean_neighbor_same_label_fraction": float(
                        class1.get("top_risk_mean_neighbor_same_label_fraction", 0.0) or 0.0
                    ),
                },
                "seconds": round(float(summary["seconds"]), 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

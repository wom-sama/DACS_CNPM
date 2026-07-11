from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from trkh.tools.audit_self_neighbor_consistency import (
    _auc_positive_greater,
    _js_divergence,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether train-neighbor consistency transfers to an independent "
            "evaluation split. Diagnostic only: no relabeling, manifest, training, "
            "raw-data edit, threshold fitting, or automatic smoke permission."
        )
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--allow-test", action="store_true", default=False)
    parser.add_argument("--top-k", type=int, nargs="+", default=[15, 30, 50])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--class1-index", type=int, default=1)
    parser.add_argument("--rival-classes", type=str, default="2,4")
    parser.add_argument("--include-same-source", action="store_true", default=False)
    return parser.parse_args(argv)


def _parse_int_set(text: str) -> set[int]:
    return {
        int(value.strip())
        for value in str(text).replace(";", ",").split(",")
        if value.strip()
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got {array.shape}.")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=True)
    required = ("embeddings", "probabilities", "labels", "base_predictions", "paths")
    for key in required:
        if key not in cache.files:
            raise ValueError(f"Embedding cache missing key {key}: {path}")
    payload = {key: np.asarray(cache[key]) for key in required}
    row_count = int(payload["labels"].shape[0])
    if any(int(payload[key].shape[0]) != row_count for key in required):
        raise ValueError(f"Embedding cache arrays have inconsistent rows: {path}")
    payload["embeddings"] = np.asarray(payload["embeddings"], dtype=np.float32)
    payload["probabilities"] = np.asarray(payload["probabilities"], dtype=np.float32)
    payload["labels"] = np.asarray(payload["labels"], dtype=np.int64)
    payload["base_predictions"] = np.asarray(payload["base_predictions"], dtype=np.int64)
    payload["paths"] = np.asarray(payload["paths"], dtype=object)
    return payload


def _source_stems(paths: np.ndarray) -> np.ndarray:
    return np.asarray([Path(str(path)).stem for path in paths], dtype=object)


def _nearest_reference_neighbors(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    query_stems: np.ndarray,
    reference_stems: np.ndarray,
    *,
    top_k: int,
    chunk_size: int,
    include_same_source: bool,
) -> np.ndarray:
    query = _normalize_rows(query_embeddings)
    reference = _normalize_rows(reference_embeddings)
    if top_k <= 0 or top_k >= reference.shape[0]:
        raise ValueError("top_k must be positive and smaller than the reference rows.")
    result = np.zeros((query.shape[0], top_k), dtype=np.int64)
    for start in range(0, query.shape[0], max(1, int(chunk_size))):
        end = min(query.shape[0], start + max(1, int(chunk_size)))
        similarity = query[start:end] @ reference.T
        if not include_same_source:
            same_source = query_stems[start:end, None] == reference_stems[None, :]
            similarity[same_source] = -np.inf
        finite_counts = np.isfinite(similarity).sum(axis=1)
        if bool(np.any(finite_counts < top_k)):
            raise ValueError("Not enough finite reference neighbors after source exclusion.")
        candidates = np.argpartition(-similarity, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(similarity, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        result[start:end] = np.take_along_axis(candidates, order, axis=1)
    return result


def _classification_metrics(targets: np.ndarray, predictions: np.ndarray, class_count: int) -> dict[str, object]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    per_class: list[dict[str, object]] = []
    for class_index in range(class_count):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class.append(
            {
                "class_index": class_index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean([float(row["f1"]) for row in per_class])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _error_role(target: int, prediction: int, class1_index: int, rivals: set[int]) -> str:
    if target == class1_index and prediction in rivals:
        return "recall_1_to_rival"
    if target in rivals and prediction == class1_index:
        return "suppress_rival_to_1"
    if target == class1_index and prediction != class1_index:
        return "other_class1_false_negative"
    if target != class1_index and prediction == class1_index:
        return "other_class1_false_positive"
    return "other"


def _auc_summary(
    *,
    metrics: Mapping[str, np.ndarray],
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> dict[str, Optional[float]]:
    return {
        name: _auc_positive_greater(values[positive_mask], values[negative_mask])
        for name, values in metrics.items()
    }


def _decision(per_k: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    metric_names = (
        "neighbor_class1_label_fraction",
        "neighbor_class1_probability",
        "same_target_fraction",
        "clean_like_score",
    )
    stable_min_by_metric: dict[str, float] = {}
    for metric in metric_names:
        values = []
        for payload in per_k.values():
            aucs = payload.get("auc_recall_gt_suppressor", {})
            if isinstance(aucs, Mapping) and aucs.get(metric) is not None:
                values.append(float(aucs[metric]))
        stable_min_by_metric[metric] = min(values) if values else 0.0
    best_metric = max(stable_min_by_metric, key=stable_min_by_metric.get)
    best_stable_auc = float(stable_min_by_metric[best_metric])
    transfer_passed = best_stable_auc >= 0.70
    return {
        "signal_transfer_passed": transfer_passed,
        "smoke_ready": False,
        "training_permission": False,
        "best_stable_metric": best_metric,
        "best_stable_recall_gt_suppressor_auc": best_stable_auc,
        "stable_min_auc_by_metric": stable_min_by_metric,
        "reason": (
            "neighbor consistency transfers across all requested k values"
            if transfer_passed
            else "neighbor consistency does not transfer reliably to the independent evaluation split"
        ),
        "guardrail": "diagnostic-only; this audit never grants smoke or training permission",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    eval_split = str(args.eval_split).strip().lower()
    if eval_split == "test" and not bool(args.allow_test):
        raise ValueError("Refusing test split without --allow-test.")
    rivals = _parse_int_set(str(args.rival_classes))
    class1_index = int(args.class1_index)
    train = _load_cache(Path(args.train_cache))
    evaluation = _load_cache(Path(args.eval_cache))
    train_embeddings = _normalize_rows(train["embeddings"])
    eval_embeddings = _normalize_rows(evaluation["embeddings"])
    train_labels = train["labels"]
    train_probabilities = train["probabilities"]
    eval_labels = evaluation["labels"]
    eval_probabilities = evaluation["probabilities"]
    eval_predictions = evaluation["base_predictions"]
    class_count = int(eval_probabilities.shape[1])
    if int(train_probabilities.shape[1]) != class_count:
        raise ValueError("Train/eval probability class counts differ.")

    train_stems = _source_stems(train["paths"])
    eval_stems = _source_stems(evaluation["paths"])
    recall_mask = (eval_labels == class1_index) & np.isin(eval_predictions, list(rivals))
    suppress_mask = np.isin(eval_labels, list(rivals)) & (eval_predictions == class1_index)
    all_fn_mask = (eval_labels == class1_index) & (eval_predictions != class1_index)
    all_fp_mask = (eval_labels != class1_index) & (eval_predictions == class1_index)
    base_metrics = _classification_metrics(eval_labels, eval_predictions, class_count)

    per_k: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for top_k in sorted(set(int(value) for value in args.top_k)):
        neighbor_indices = _nearest_reference_neighbors(
            eval_embeddings,
            train_embeddings,
            eval_stems,
            train_stems,
            top_k=top_k,
            chunk_size=int(args.chunk_size),
            include_same_source=bool(args.include_same_source),
        )
        neighbor_label_fraction = np.stack(
            [(train_labels[neighbor_indices] == class_index).mean(axis=1) for class_index in range(class_count)],
            axis=1,
        )
        neighbor_probability = train_probabilities[neighbor_indices].mean(axis=1)
        same_target = neighbor_label_fraction[np.arange(eval_labels.shape[0]), eval_labels]
        confidence = eval_probabilities[np.arange(eval_labels.shape[0]), eval_predictions]
        divergence = _js_divergence(eval_probabilities, neighbor_probability)
        clean_like = confidence * (1.0 - np.minimum(divergence, 1.0)) * same_target
        metric_arrays = {
            "neighbor_class1_label_fraction": neighbor_label_fraction[:, class1_index],
            "neighbor_class1_probability": neighbor_probability[:, class1_index],
            "same_target_fraction": same_target,
            "clean_like_score": clean_like,
        }
        knn_predictions = neighbor_label_fraction.argmax(axis=1).astype(np.int64)
        pair_auc: dict[str, object] = {}
        for rival in sorted(rivals):
            positive = (eval_labels == class1_index) & (eval_predictions == rival)
            negative = (eval_labels == rival) & (eval_predictions == class1_index)
            pair_auc[f"1_to_{rival}_gt_{rival}_to_1"] = _auc_summary(
                metrics=metric_arrays,
                positive_mask=positive,
                negative_mask=negative,
            )
        per_k[str(top_k)] = {
            "top_k": top_k,
            "counts": {
                "recall_1_to_rival": int(recall_mask.sum()),
                "suppress_rival_to_1": int(suppress_mask.sum()),
                "all_class1_false_negative": int(all_fn_mask.sum()),
                "all_class1_false_positive": int(all_fp_mask.sum()),
            },
            "auc_recall_gt_suppressor": _auc_summary(
                metrics=metric_arrays,
                positive_mask=recall_mask,
                negative_mask=suppress_mask,
            ),
            "auc_all_fn_gt_fp": _auc_summary(
                metrics=metric_arrays,
                positive_mask=all_fn_mask,
                negative_mask=all_fp_mask,
            ),
            "pair_auc": pair_auc,
            "knn_metrics": _classification_metrics(eval_labels, knn_predictions, class_count),
        }
        for index in range(eval_labels.shape[0]):
            row = {
                "top_k": top_k,
                "sample_index": index,
                "image_path": str(evaluation["paths"][index]),
                "target_index": int(eval_labels[index]),
                "prediction_index": int(eval_predictions[index]),
                "error_role": _error_role(
                    int(eval_labels[index]),
                    int(eval_predictions[index]),
                    class1_index,
                    rivals,
                ),
                "neighbor_prediction_index": int(knn_predictions[index]),
                "neighbor_class1_label_fraction": float(metric_arrays["neighbor_class1_label_fraction"][index]),
                "neighbor_class1_probability": float(metric_arrays["neighbor_class1_probability"][index]),
                "same_target_fraction": float(same_target[index]),
                "neighbor_jsd": float(divergence[index]),
                "clean_like_score": float(clean_like[index]),
            }
            rows.append(row)

    decision = _decision(per_k)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "neighbor_transfer_rows.csv", rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_cache": str(Path(args.train_cache)),
        "eval_cache": str(Path(args.eval_cache)),
        "output_dir": str(output_dir),
        "eval_split": eval_split,
        "train_rows": int(train_labels.shape[0]),
        "eval_rows": int(eval_labels.shape[0]),
        "class1_index": class1_index,
        "rival_classes": sorted(rivals),
        "top_k": sorted(per_k.keys(), key=int),
        "include_same_source": bool(args.include_same_source),
        "base_metrics": base_metrics,
        "per_k": per_k,
        "decision": decision,
        "raw_dataset_touched": False,
        "test_split_used": eval_split == "test",
        "threshold_fitted": False,
        "trainable_manifest_written": False,
        "research_sources": [
            {
                "name": "Jo-SNC",
                "url": "https://arxiv.org/html/2601.12795v1",
                "note": "Requires self/neighbor consistency to identify reliable samples; this audit checks independent-split transfer.",
            },
            {
                "name": "SNSCL / Fine-Grained Classification with Noisy Labels",
                "url": "https://arxiv.org/abs/2303.02404",
                "note": "Reliability-aware contrastive learning needs trustworthy anchors rather than in-sample neighbor separation alone.",
            },
        ],
        "seconds": round(time.perf_counter() - start, 6),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Neighbor Transfer Consistency Audit",
                "",
                "Train-reference to independent-evaluation diagnostic only.",
                "No threshold was fitted and this artifact grants no smoke permission.",
                "",
                f"Decision: {decision['reason']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    summary = run_audit(args)
    print(json.dumps({"output_dir": summary["output_dir"], "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()

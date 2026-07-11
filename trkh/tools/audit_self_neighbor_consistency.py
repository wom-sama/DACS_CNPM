from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only self/neighbor-consistency audit inspired by recent "
            "noisy-label methods such as Jo-SNC. The tool writes diagnostic "
            "CSV/JSON/README artifacts only; it does not relabel, write a "
            "trainable manifest, use test, or modify raw data."
        )
    )
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--feature-cache-npz", type=Path, required=True)
    parser.add_argument(
        "--feature-key",
        type=str,
        default="source_embeddings",
        help=(
            "NPZ feature array to audit. Local-neighbor caches store raw keeper "
            "embeddings as source_embeddings and smoothed teacher targets as features."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, action="append", default=[])
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--class1-index", type=int, default=1)
    parser.add_argument("--include-same-source", action="store_true", default=False)
    parser.add_argument("--created-at", type=str, default="")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    row_list = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in row_list:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _extract_class_index_from_prob_column(column: str) -> Optional[int]:
    if not column.startswith("prob_"):
        return None
    suffix = column[len("prob_") :]
    first = suffix.split("_", 1)[0]
    try:
        return int(first)
    except ValueError:
        return None


def _source_stem_from_path(path: object) -> str:
    text = str(path)
    if not text:
        return ""
    return Path(text).stem


def _read_prediction_csv(path: Path) -> dict[str, object]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Prediction CSV has no rows: {path}")

    prob_columns: list[tuple[int, str]] = []
    for column in fieldnames:
        class_index = _extract_class_index_from_prob_column(column)
        if class_index is not None:
            prob_columns.append((class_index, column))
    if not prob_columns:
        raise ValueError(f"Could not find prob_* columns in {path}")
    prob_columns.sort(key=lambda item: item[0])

    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    targets = np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64)
    preds = np.asarray([int(row["prediction_index"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row[column]) for _, column in prob_columns] for row in rows],
        dtype=np.float32,
    )
    confidences = np.asarray(
        [
            float(row.get("confidence") or probs[i, preds[i]])
            for i, row in enumerate(rows)
        ],
        dtype=np.float32,
    )
    source_stems = np.asarray(
        [
            str(row.get("source_stem") or _source_stem_from_path(row.get("image_path", "")))
            for row in rows
        ],
        dtype=object,
    )
    return {
        "rows": rows,
        "sample_indices": sample_indices,
        "targets": targets,
        "preds": preds,
        "probs": probs,
        "confidences": confidences,
        "source_stems": source_stems,
        "class_indices": [class_index for class_index, _ in prob_columns],
    }


def _safe_l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D features, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _align_feature_cache(
    *,
    prediction_sample_indices: np.ndarray,
    feature_cache_npz: Path,
    feature_key: str,
) -> dict[str, object]:
    cache = np.load(feature_cache_npz, allow_pickle=True)
    selected_feature_key = str(feature_key).strip()
    if not selected_feature_key:
        raise ValueError("feature_key must not be empty.")
    for key in (selected_feature_key, "sample_index", "labels"):
        if key not in cache.files:
            raise ValueError(f"Feature cache missing key {key}: {feature_cache_npz}")
    features = np.asarray(cache[selected_feature_key], dtype=np.float32)
    sample_indices = np.asarray(cache["sample_index"], dtype=np.int64)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    paths = np.asarray(cache["paths"], dtype=object) if "paths" in cache.files else np.asarray([""] * len(labels), dtype=object)
    if features.shape[0] != sample_indices.shape[0] or labels.shape[0] != sample_indices.shape[0]:
        raise ValueError("Feature cache arrays have inconsistent row counts.")

    index_to_cache_pos = {int(sample_index): i for i, sample_index in enumerate(sample_indices.tolist())}
    missing = [int(sample_index) for sample_index in prediction_sample_indices.tolist() if int(sample_index) not in index_to_cache_pos]
    if missing:
        raise ValueError(f"Feature cache missing {len(missing)} prediction sample indices; first={missing[:5]}")
    order = np.asarray([index_to_cache_pos[int(sample_index)] for sample_index in prediction_sample_indices.tolist()], dtype=np.int64)
    return {
        "features": _safe_l2_normalize(features[order]),
        "labels": labels[order],
        "paths": paths[order],
        "cache_sample_indices": sample_indices[order],
        "coverage": int(order.shape[0]),
        "feature_key": selected_feature_key,
    }


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = p_arr / np.maximum(p_arr.sum(axis=-1, keepdims=True), eps)
    q_arr = q_arr / np.maximum(q_arr.sum(axis=-1, keepdims=True), eps)
    p_arr = np.clip(p_arr, eps, 1.0)
    q_arr = np.clip(q_arr, eps, 1.0)
    midpoint = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * np.log(p_arr / midpoint), axis=-1)
    kl_qm = np.sum(q_arr * np.log(q_arr / midpoint), axis=-1)
    return (0.5 * (kl_pm + kl_qm)).astype(np.float32, copy=False)


def _auc_positive_greater(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    comparisons = (pos[:, None] > neg[None, :]).astype(np.float64)
    ties = (pos[:, None] == neg[None, :]).astype(np.float64) * 0.5
    return float((comparisons + ties).mean())


def _nearest_neighbors(
    features: np.ndarray,
    source_stems: np.ndarray,
    *,
    top_k: int,
    chunk_size: int,
    include_same_source: bool,
) -> np.ndarray:
    values = _safe_l2_normalize(features)
    n = values.shape[0]
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k >= n:
        raise ValueError("top_k must be smaller than the number of rows.")
    result = np.zeros((n, top_k), dtype=np.int64)
    stems = np.asarray(source_stems, dtype=object)
    for start in range(0, n, max(1, int(chunk_size))):
        end = min(n, start + max(1, int(chunk_size)))
        similarity = values[start:end] @ values.T
        row_indices = np.arange(start, end)
        similarity[np.arange(end - start), row_indices] = -np.inf
        if not include_same_source:
            same_source = stems[row_indices, None] == stems[None, :]
            similarity[same_source] = -np.inf
        candidate = np.argpartition(-similarity, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(similarity, candidate, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        result[start:end] = np.take_along_axis(candidate, order, axis=1)
    return result


def _compute_neighbor_rows(
    *,
    sample_indices: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
    class1_index: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    class_count = probs.shape[1]
    for i in range(sample_indices.shape[0]):
        nn = neighbor_indices[i]
        nn_labels = labels[nn]
        nn_preds = preds[nn]
        nn_probs = probs[nn]
        neighbor_prob_mean = nn_probs.mean(axis=0)
        jsd = float(_js_divergence(probs[i : i + 1], neighbor_prob_mean.reshape(1, -1))[0])
        target = int(targets[i])
        pred = int(preds[i])
        target_prob = float(probs[i, target])
        row: dict[str, object] = {
            "sample_index": int(sample_indices[i]),
            "target_index": target,
            "prediction_index": pred,
            "correct": int(target == pred),
            "confidence": float(confidences[i]),
            "neighbor_jsd": jsd,
            "neighbor_same_target_fraction": float(np.mean(nn_labels == target)),
            "neighbor_same_prediction_fraction": float(np.mean(nn_preds == pred)),
            "neighbor_class1_label_fraction": float(np.mean(nn_labels == int(class1_index))),
            "neighbor_class1_prediction_fraction": float(np.mean(nn_preds == int(class1_index))),
            "self_class1_probability": float(probs[i, int(class1_index)]),
            "neighbor_class1_probability": float(neighbor_prob_mean[int(class1_index)]),
            "self_target_probability": target_prob,
            "neighbor_target_probability": float(neighbor_prob_mean[target]),
            "neighbor_target_probability_delta": float(neighbor_prob_mean[target] - target_prob),
            "clean_like_score": float(confidences[i] * (1.0 - min(jsd, 1.0)) * np.mean(nn_labels == target)),
        }
        for class_index in range(class_count):
            row[f"neighbor_prob_{class_index}"] = float(neighbor_prob_mean[class_index])
        rows.append(row)
    return rows


def _summarize_numeric(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    values = np.asarray([float(row[key]) for row in rows if row.get(key, "") != ""], dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _load_review_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = dict(row)
                item["review_source_csv"] = str(path)
                rows.append(item)
    return rows


def _join_review_rows(
    *,
    review_rows: Sequence[Mapping[str, object]],
    consistency_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_sample = {int(row["sample_index"]): dict(row) for row in consistency_rows}
    joined: list[dict[str, object]] = []
    for review in review_rows:
        if not str(review.get("sample_index", "")).strip():
            continue
        sample_index = int(review["sample_index"])
        if sample_index not in by_sample:
            continue
        item = dict(review)
        for key, value in by_sample[sample_index].items():
            if key not in item:
                item[key] = value
            else:
                item[f"consistency_{key}"] = value
        joined.append(item)
    return joined


def _deduplicate_review_rows(
    joined_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_sample: dict[int, list[Mapping[str, object]]] = {}
    for row in joined_rows:
        by_sample.setdefault(int(row["sample_index"]), []).append(row)

    unique_rows: list[dict[str, object]] = []
    conflict_sample_indices: list[int] = []
    for sample_index, rows in sorted(by_sample.items()):
        sides = {
            str(row.get("strict_review_side", "") or "unknown")
            for row in rows
        }
        if len(sides) != 1:
            conflict_sample_indices.append(sample_index)
            continue
        item = dict(rows[0])
        item["review_source_csvs"] = "|".join(
            sorted({str(row.get("review_source_csv", "")) for row in rows})
        )
        item["review_duplicate_count"] = len(rows)
        unique_rows.append(item)

    return unique_rows, {
        "raw_joined_row_count": len(joined_rows),
        "unique_sample_count": len(by_sample),
        "decision_eligible_unique_sample_count": len(unique_rows),
        "duplicate_row_count": len(joined_rows) - len(by_sample),
        "conflicting_side_sample_count": len(conflict_sample_indices),
        "conflicting_side_sample_indices": conflict_sample_indices,
    }


def _review_side_summary(joined_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    side_key = "strict_review_side"
    sides: dict[str, list[Mapping[str, object]]] = {}
    for row in joined_rows:
        side = str(row.get(side_key, "") or "unknown")
        sides.setdefault(side, []).append(row)
    metric_keys = [
        "neighbor_class1_label_fraction",
        "neighbor_class1_prediction_fraction",
        "neighbor_class1_probability",
        "self_class1_probability",
        "neighbor_jsd",
        "neighbor_same_target_fraction",
        "clean_like_score",
    ]
    summary: dict[str, object] = {
        "row_count": len(joined_rows),
        "side_counts": {key: len(value) for key, value in sorted(sides.items())},
        "metrics_by_side": {},
        "auc_recall_gt_suppressor": {},
    }
    metrics_by_side: dict[str, object] = {}
    for side, rows in sorted(sides.items()):
        metrics_by_side[side] = {key: _summarize_numeric(rows, key) for key in metric_keys if rows}
    summary["metrics_by_side"] = metrics_by_side
    recall = sides.get("recall_protector", [])
    suppress = sides.get("fp_suppressor", [])
    aucs: dict[str, object] = {}
    for key in metric_keys:
        aucs[key] = _auc_positive_greater(
            [float(row[key]) for row in recall if key in row],
            [float(row[key]) for row in suppress if key in row],
        )
    summary["auc_recall_gt_suppressor"] = aucs
    return summary


def _transition_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        transition = str(row.get("risk_transition") or f"{row.get('target_index')}->{row.get('prediction_index')}")
        counts[transition] = counts.get(transition, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _decision_hint(review_summary: Mapping[str, object]) -> dict[str, object]:
    aucs = review_summary.get("auc_recall_gt_suppressor", {})
    if not isinstance(aucs, Mapping):
        return {"smoke_ready": False, "reason": "missing review AUC summary"}
    class1_auc = aucs.get("neighbor_class1_label_fraction")
    prob_auc = aucs.get("neighbor_class1_probability")
    clean_auc = aucs.get("clean_like_score")
    best_auc = max(
        [float(v) for v in (class1_auc, prob_auc, clean_auc) if v is not None],
        default=0.0,
    )
    separation_passed = best_auc >= 0.70
    reason = (
        "self/neighbor consistency separates recall protectors from FP suppressors"
        if separation_passed
        else "self/neighbor consistency does not separate recall protectors from FP suppressors strongly enough"
    )
    return {
        "signal_separation_passed": bool(separation_passed),
        "smoke_ready": False,
        "best_recall_gt_suppressor_auc": float(best_auc),
        "reason": reason,
        "training_permission": False,
        "guardrail": "diagnostic-only; rerun smoke gate before any trainable use",
    }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    prediction = _read_prediction_csv(args.prediction_csv)
    sample_indices = prediction["sample_indices"]
    targets = prediction["targets"]
    preds = prediction["preds"]
    probs = prediction["probs"]
    confidences = prediction["confidences"]
    source_stems = prediction["source_stems"]

    cache = _align_feature_cache(
        prediction_sample_indices=sample_indices,
        feature_cache_npz=args.feature_cache_npz,
        feature_key=str(args.feature_key),
    )
    labels = np.asarray(cache["labels"], dtype=np.int64)
    if not np.array_equal(labels, targets):
        mismatch = int(np.count_nonzero(labels != targets))
    else:
        mismatch = 0
    neighbor_indices = _nearest_neighbors(
        np.asarray(cache["features"], dtype=np.float32),
        source_stems,
        top_k=int(args.top_k),
        chunk_size=int(args.chunk_size),
        include_same_source=bool(args.include_same_source),
    )
    consistency_rows = _compute_neighbor_rows(
        sample_indices=sample_indices,
        targets=targets,
        preds=preds,
        probs=probs,
        confidences=confidences,
        labels=labels,
        neighbor_indices=neighbor_indices,
        class1_index=int(args.class1_index),
    )
    review_rows = _load_review_rows(args.review_csv)
    joined_rows = _join_review_rows(
        review_rows=review_rows,
        consistency_rows=consistency_rows,
    )
    unique_review_rows, review_deduplication = _deduplicate_review_rows(joined_rows)
    review_summary = _review_side_summary(unique_review_rows)
    decision = _decision_hint(review_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "self_neighbor_consistency_rows.csv", consistency_rows)
    _write_csv(output_dir / "review_joined_self_neighbor_consistency.csv", joined_rows)
    _write_csv(output_dir / "review_unique_self_neighbor_consistency.csv", unique_review_rows)
    summary = {
        "created_at": args.created_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "guardrail": (
            "Train-only self/neighbor-consistency diagnostic. No deletion, no raw-data "
            "edit, no test usage, no relabeling, no sample weights, no soft targets, "
            "no targeted margins, no trainable manifest, no smoke permission."
        ),
        "prediction_csv": str(args.prediction_csv),
        "feature_cache_npz": str(args.feature_cache_npz),
        "feature_key": str(cache["feature_key"]),
        "review_csv": [str(path) for path in args.review_csv],
        "output_dir": str(output_dir),
        "row_count": int(sample_indices.shape[0]),
        "feature_cache_coverage": int(cache["coverage"]),
        "target_label_mismatch_count": int(mismatch),
        "top_k": int(args.top_k),
        "include_same_source": bool(args.include_same_source),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "overall_metrics": {
            "neighbor_jsd": _summarize_numeric(consistency_rows, "neighbor_jsd"),
            "neighbor_same_target_fraction": _summarize_numeric(consistency_rows, "neighbor_same_target_fraction"),
            "neighbor_class1_label_fraction": _summarize_numeric(consistency_rows, "neighbor_class1_label_fraction"),
            "neighbor_class1_probability": _summarize_numeric(consistency_rows, "neighbor_class1_probability"),
            "clean_like_score": _summarize_numeric(consistency_rows, "clean_like_score"),
        },
        "review_summary": review_summary,
        "review_deduplication": review_deduplication,
        "review_transition_counts": _transition_counts(unique_review_rows),
        "decision": decision,
        "research_sources": [
            {
                "name": "Jo-SNC: Combating Noisy Labels through Fostering Self- and Neighbor-Consistency",
                "url": "https://arxiv.org/html/2601.12795v1",
                "note": "Motivates auditing neighbor-cleanliness and self/neighbor prediction consistency before any sample-selection policy.",
            },
            {
                "name": "Fine-Grained Classification with Noisy Labels / SNSCL",
                "url": "https://arxiv.org/abs/2303.02404",
                "note": "Fine-grained label noise requires reliability-aware representation learning; current audit checks whether reliable train anchors exist.",
            },
        ],
        "seconds": round(time.perf_counter() - start, 6),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Self/Neighbor Consistency Audit",
                "",
                "Diagnostic-only train split audit inspired by Jo-SNC style noisy-label methods.",
                "It is not a trainable manifest and does not grant smoke permission.",
                "",
                f"Rows: {summary['row_count']}",
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

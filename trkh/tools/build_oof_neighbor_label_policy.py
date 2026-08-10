from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from trkh.core.config import load_data_spec


IMAGE_ID_PATTERN = re.compile(r"Image_(\d+)", re.IGNORECASE)


def _read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _resolved_key(path_text: str) -> str:
    return str(Path(path_text).resolve()).lower()


def _looks_like_non_train(path_text: str) -> bool:
    parts = [part for part in str(path_text or "").replace("\\", "/").lower().split("/") if part]
    return "val" in parts or "valid" in parts or "validation" in parts or "test" in parts


def _int_value(row: Mapping[str, str], key: str, *, default: int = -1) -> int:
    value = str(row.get(key, "") or "").strip()
    if not value:
        return int(default)
    try:
        return int(float(value))
    except ValueError:
        return int(default)


def _float_value(row: Mapping[str, str], key: str, *, default: float = 0.0) -> float:
    value = str(row.get(key, "") or "").strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _image_id(path_text: str) -> str:
    match = IMAGE_ID_PATTERN.search(str(path_text or ""))
    return match.group(1) if match is not None else ""


def _feature_alias(path: Path, index: int) -> str:
    stem = path.stem
    stem = re.sub(r"^features_train_", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem[:40] or f"feature_{index}"


class FeatureBank:
    def __init__(self, path: Path, alias: str):
        loaded = np.load(path, allow_pickle=True)
        missing = {"features", "labels", "paths"}.difference(loaded.files)
        if missing:
            raise ValueError(f"Feature NPZ missing keys {sorted(missing)}: {path}")
        features = np.asarray(loaded["features"], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D: {path}")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        self.path = Path(path)
        self.alias = alias
        self.features = features
        self.labels = np.asarray(loaded["labels"], dtype=np.int64)
        self.paths = np.asarray(loaded["paths"]).astype(str)
        if len(self.labels) != len(self.paths) or len(self.paths) != features.shape[0]:
            raise ValueError(f"Feature NPZ arrays have inconsistent lengths: {path}")
        self.index_by_path = {_resolved_key(path_text): index for index, path_text in enumerate(self.paths.tolist())}
        sample_index_values = (
            np.asarray(loaded["sample_index"], dtype=np.int64)
            if "sample_index" in loaded.files
            else np.full((features.shape[0],), -1, dtype=np.int64)
        )
        if int(sample_index_values.shape[0]) != int(features.shape[0]):
            raise ValueError(f"sample_index length mismatch in feature NPZ: {path}")
        self.sample_index = sample_index_values
        self.index_by_sample_index = {
            int(sample_index): index
            for index, sample_index in enumerate(self.sample_index.tolist())
            if int(sample_index) >= 0
        }
        self.image_ids = np.asarray([_image_id(path_text) for path_text in self.paths.tolist()])

    def vote(
        self,
        *,
        path_text: str,
        sample_index: int,
        suggested_index: int,
        top_k: int,
        exclude_same_image_id: bool,
    ) -> Dict[str, object]:
        query_index: Optional[int] = None
        if int(sample_index) >= 0:
            query_index = self.index_by_sample_index.get(int(sample_index))
        if query_index is None:
            key = _resolved_key(path_text)
            query_index = self.index_by_path.get(key)
        if query_index is None:
            return {
                "available": 0,
                "top_index": -1,
                "top_vote_fraction": 0.0,
                "suggested_vote_fraction": 0.0,
                "mean_similarity": 0.0,
                "top_neighbor_paths": "",
                "top_neighbor_labels": "",
            }
        similarities = self.features @ self.features[query_index]
        similarities[query_index] = -np.inf
        if exclude_same_image_id:
            current_id = self.image_ids[query_index]
            if current_id:
                similarities[self.image_ids == current_id] = -np.inf
        finite = np.isfinite(similarities)
        available = int(np.count_nonzero(finite))
        if available <= 0:
            return {
                "available": 0,
                "top_index": -1,
                "top_vote_fraction": 0.0,
                "suggested_vote_fraction": 0.0,
                "mean_similarity": 0.0,
                "top_neighbor_paths": "",
                "top_neighbor_labels": "",
            }
        k = max(1, min(int(top_k), available))
        top_indices = np.argpartition(-similarities, k - 1)[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        top_labels = [int(value) for value in self.labels[top_indices].tolist()]
        counts = Counter(top_labels)
        top_index, top_count = counts.most_common(1)[0]
        suggested_count = int(counts.get(int(suggested_index), 0))
        return {
            "available": available,
            "top_index": int(top_index),
            "top_vote_fraction": float(top_count / k),
            "suggested_vote_fraction": float(suggested_count / k),
            "mean_similarity": float(np.mean(similarities[top_indices])),
            "top_neighbor_paths": "|".join(str(self.paths[index]) for index in top_indices[:5]),
            "top_neighbor_labels": "|".join(str(label) for label in top_labels[:5]),
        }


def _load_banks(feature_npz: Sequence[Path]) -> List[FeatureBank]:
    banks: List[FeatureBank] = []
    used_aliases: Counter[str] = Counter()
    for index, path in enumerate(feature_npz):
        alias = _feature_alias(Path(path), index)
        used_aliases[alias] += 1
        if used_aliases[alias] > 1:
            alias = f"{alias}_{used_aliases[alias]}"
        banks.append(FeatureBank(Path(path), alias))
    if not banks:
        raise ValueError("At least one --feature-npz is required.")
    return banks


def build_policy(
    *,
    data_yaml: Path,
    cleanlab_manifest: Path,
    feature_npz: Sequence[Path],
    output_dir: Path,
    top_k: int = 15,
    min_vote_fraction: float = 0.65,
    max_issue_rank: int = 300,
    min_oof_confidence: float = 0.70,
    max_self_confidence: float = 0.20,
    strict_weight: float = 0.05,
    soft_alpha: float = 0.75,
    exclude_same_image_id: bool = True,
    allow_non_train_paths: bool = False,
) -> Dict[str, object]:
    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    class_names = [str(value) for value in data_spec.class_names]
    rows, _ = _read_csv(cleanlab_manifest)
    banks = _load_banks(feature_npz)

    reviewed_rows: List[Dict[str, object]] = []
    strict_rows: List[Dict[str, object]] = []
    review_rows: List[Dict[str, object]] = []
    non_train_paths: List[str] = []
    issue_pairs: Counter[str] = Counter()
    strict_pairs: Counter[str] = Counter()
    strict_by_feature_count: Counter[int] = Counter()

    for row in rows:
        path_text = str(row.get("path", "") or row.get("image_path", "") or "").strip()
        if not path_text:
            continue
        if _looks_like_non_train(path_text):
            non_train_paths.append(path_text)
            if not allow_non_train_paths:
                continue
        true_index = _int_value(row, "true_index")
        suggested_index = _int_value(row, "suggested_index")
        issue_rank = _int_value(row, "issue_rank", default=10**9)
        is_issue = _int_value(row, "is_label_issue", default=0) == 1
        if true_index < 0 or suggested_index < 0 or true_index >= len(class_names) or suggested_index >= len(class_names):
            continue
        if true_index == suggested_index:
            continue
        if is_issue:
            issue_pairs[f"{true_index}->{suggested_index}"] += 1
        sample_index = _int_value(row, "sample_index", default=-1)

        enriched: Dict[str, object] = {
            "image_path": path_text,
            "sample_index": sample_index if sample_index >= 0 else "",
            "target_index": true_index,
            "target_name": class_names[true_index],
            "suggested_index": suggested_index,
            "suggested_name": class_names[suggested_index],
            "issue_rank": issue_rank,
            "label_quality": _float_value(row, "label_quality", default=1.0),
            "self_confidence": _float_value(row, "self_confidence", default=1.0),
            "top1_confidence": _float_value(row, "top1_confidence", default=0.0),
            "top2_margin": _float_value(row, "top2_margin", default=0.0),
            "normalized_entropy": _float_value(row, "normalized_entropy", default=0.0),
            "is_label_issue": int(is_issue),
        }

        vote_failures: List[str] = []
        vote_pass_count = 0
        min_suggested_vote = 1.0
        max_suggested_vote = 0.0
        for bank in banks:
            vote = bank.vote(
                path_text=path_text,
                sample_index=sample_index,
                suggested_index=suggested_index,
                top_k=top_k,
                exclude_same_image_id=exclude_same_image_id,
            )
            alias = bank.alias
            enriched[f"{alias}_available_neighbors"] = int(vote["available"])
            enriched[f"{alias}_top_index"] = int(vote["top_index"])
            enriched[f"{alias}_top_name"] = (
                class_names[int(vote["top_index"])] if 0 <= int(vote["top_index"]) < len(class_names) else ""
            )
            enriched[f"{alias}_top_vote_fraction"] = f"{float(vote['top_vote_fraction']):.10g}"
            enriched[f"{alias}_suggested_vote_fraction"] = f"{float(vote['suggested_vote_fraction']):.10g}"
            enriched[f"{alias}_mean_similarity"] = f"{float(vote['mean_similarity']):.10g}"
            enriched[f"{alias}_top_neighbor_labels"] = str(vote["top_neighbor_labels"])
            enriched[f"{alias}_top_neighbor_paths"] = str(vote["top_neighbor_paths"])
            suggested_vote = float(vote["suggested_vote_fraction"])
            min_suggested_vote = min(min_suggested_vote, suggested_vote)
            max_suggested_vote = max(max_suggested_vote, suggested_vote)
            if int(vote["top_index"]) == suggested_index and suggested_vote >= float(min_vote_fraction):
                vote_pass_count += 1
            else:
                vote_failures.append(f"{alias}_vote")

        strict_reasons: List[str] = []
        if not is_issue:
            strict_reasons.append("not_cleanlab_issue")
        if issue_rank > int(max_issue_rank):
            strict_reasons.append("issue_rank")
        if _float_value(row, "top1_confidence", default=0.0) < float(min_oof_confidence):
            strict_reasons.append("oof_confidence")
        if _float_value(row, "self_confidence", default=1.0) > float(max_self_confidence):
            strict_reasons.append("self_confidence")
        strict_reasons.extend(vote_failures)
        is_strict = not strict_reasons
        enriched["feature_vote_pass_count"] = vote_pass_count
        enriched["feature_bank_count"] = len(banks)
        enriched["min_suggested_vote_fraction"] = f"{min_suggested_vote:.10g}"
        enriched["max_suggested_vote_fraction"] = f"{max_suggested_vote:.10g}"
        enriched["policy"] = "strict_relabel_or_ignore" if is_strict else "review_only"
        enriched["policy_reason"] = "strict_consensus" if is_strict else ";".join(strict_reasons)
        reviewed_rows.append(enriched)
        if is_strict:
            strict_rows.append(enriched)
            strict_pairs[f"{true_index}->{suggested_index}"] += 1
            strict_by_feature_count[vote_pass_count] += 1
        else:
            review_rows.append(enriched)

    reviewed_rows.sort(
        key=lambda item: (
            0 if item["policy"] == "strict_relabel_or_ignore" else 1,
            int(item["issue_rank"]),
            -float(item["min_suggested_vote_fraction"]),
        )
    )
    strict_rows.sort(key=lambda item: (int(item["issue_rank"]), -float(item["min_suggested_vote_fraction"])))
    review_rows.sort(key=lambda item: (int(item["issue_rank"]), -float(item["max_suggested_vote_fraction"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "oof_neighbor_policy_review.csv"
    strict_path = output_dir / "strict_relabel_or_ignore_candidates.csv"
    sample_weight_path = output_dir / "strict_ignore_sample_weights_train_only.csv"
    soft_target_path = output_dir / "strict_relabel_soft_targets_train_only.csv"

    fieldnames = list(reviewed_rows[0].keys()) if reviewed_rows else [
        "image_path",
        "target_index",
        "target_name",
        "suggested_index",
        "suggested_name",
        "policy",
        "policy_reason",
    ]
    _write_csv(review_path, reviewed_rows, fieldnames)
    _write_csv(strict_path, strict_rows, fieldnames)

    sample_weight_rows = [
        {
            "image_path": row["image_path"],
            "sample_index": row.get("sample_index", ""),
            "target_index": row["target_index"],
            "target_name": row["target_name"],
            "prediction_index": row["suggested_index"],
            "prediction_name": row["suggested_name"],
            "sample_weight": f"{float(strict_weight):.10g}",
            "reason": "strict_oof_cleanlab_neighbor_consensus",
            "review_decision": "ignore_or_manual_relabel_candidate",
            "issue_rank": row["issue_rank"],
        }
        for row in strict_rows
    ]
    _write_csv(
        sample_weight_path,
        sample_weight_rows,
        [
            "image_path",
            "sample_index",
            "target_index",
            "target_name",
            "prediction_index",
            "prediction_name",
            "sample_weight",
            "reason",
            "review_decision",
            "issue_rank",
        ],
    )

    soft_rows: List[Dict[str, object]] = []
    soft_alpha = max(0.0, min(1.0, float(soft_alpha)))
    for row in strict_rows:
        target_index = int(row["target_index"])
        soft_index = int(row["suggested_index"])
        probabilities = [0.0 for _ in class_names]
        probabilities[target_index] = 1.0 - soft_alpha
        probabilities[soft_index] += soft_alpha
        record: Dict[str, object] = {
            "image_path": row["image_path"],
            "sample_index": row.get("sample_index", ""),
            "target_index": target_index,
            "soft_target_index": soft_index,
            "alpha": f"{soft_alpha:.10g}",
            "reason": "strict_oof_cleanlab_neighbor_consensus",
            "issue_rank": row["issue_rank"],
            "prediction_index": soft_index,
            "top2_margin": row["top2_margin"],
        }
        record.update({f"soft_{index}": f"{probabilities[index]:.10g}" for index in range(len(class_names))})
        soft_rows.append(record)
    _write_csv(
        soft_target_path,
        soft_rows,
        [
            "image_path",
            "sample_index",
            "target_index",
            "soft_target_index",
            "alpha",
            "reason",
            "issue_rank",
            "prediction_index",
            "top2_margin",
            *[f"soft_{index}" for index in range(len(class_names))],
        ],
    )

    summary: Dict[str, object] = {
        "data": str(data_yaml),
        "cleanlab_manifest": str(cleanlab_manifest),
        "feature_npz": [str(path) for path in feature_npz],
        "output_review_csv": str(review_path),
        "output_strict_csv": str(strict_path),
        "output_sample_weights": str(sample_weight_path),
        "output_soft_targets": str(soft_target_path),
        "classes": class_names,
        "rows_read": len(rows),
        "candidate_rows": len(reviewed_rows),
        "strict_rows": len(strict_rows),
        "review_only_rows": len(review_rows),
        "non_train_paths": len(non_train_paths),
        "issue_pairs": dict(sorted(issue_pairs.items())),
        "strict_pairs": dict(sorted(strict_pairs.items())),
        "strict_by_feature_count": dict(sorted(strict_by_feature_count.items())),
        "settings": {
            "top_k": int(top_k),
            "min_vote_fraction": float(min_vote_fraction),
            "max_issue_rank": int(max_issue_rank),
            "min_oof_confidence": float(min_oof_confidence),
            "max_self_confidence": float(max_self_confidence),
            "strict_weight": float(strict_weight),
            "soft_alpha": float(soft_alpha),
            "exclude_same_image_id": bool(exclude_same_image_id),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if non_train_paths and not allow_non_train_paths:
        raise ValueError(
            "Cleanlab manifest contained non-train paths; no train-signal manifest should use these. "
            f"Sample: {non_train_paths[:3]}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only strict label-policy candidates by intersecting OOF cleanlab issues "
            "with kNN feature-neighbor consensus."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cleanlab-manifest", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-vote-fraction", type=float, default=0.65)
    parser.add_argument("--max-issue-rank", type=int, default=300)
    parser.add_argument("--min-oof-confidence", type=float, default=0.70)
    parser.add_argument("--max-self-confidence", type=float, default=0.20)
    parser.add_argument("--strict-weight", type=float, default=0.05)
    parser.add_argument("--soft-alpha", type=float, default=0.75)
    parser.add_argument("--include-same-image-id", action="store_true")
    parser.add_argument("--allow-non-train-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_policy(
        data_yaml=args.data,
        cleanlab_manifest=args.cleanlab_manifest,
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        top_k=args.top_k,
        min_vote_fraction=args.min_vote_fraction,
        max_issue_rank=args.max_issue_rank,
        min_oof_confidence=args.min_oof_confidence,
        max_self_confidence=args.max_self_confidence,
        strict_weight=args.strict_weight,
        soft_alpha=args.soft_alpha,
        exclude_same_image_id=not bool(args.include_same_image_id),
        allow_non_train_paths=bool(args.allow_non_train_paths),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

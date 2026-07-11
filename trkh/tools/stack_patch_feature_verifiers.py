from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stack train-OOF patch-evidence and feature verifier outputs into a "
            "selective 0-1 reliability gate. This is a diagnostic and does not "
            "read test unless the caller explicitly passes test CSVs."
        )
    )
    parser.add_argument("--patch-train-predictions", type=Path, required=True)
    parser.add_argument("--feature-train-predictions", type=Path, required=True)
    parser.add_argument("--patch-eval-predictions", type=Path, required=True)
    parser.add_argument("--feature-eval-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-mode",
        choices=("exact_correct", "not_harm", "class1_benefit"),
        default="class1_benefit",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _read_merged(patch_csv: Path, feature_csv: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    patch = pd.read_csv(patch_csv).add_prefix("patch_")
    feature = pd.read_csv(feature_csv).add_prefix("feature_")
    merged = patch.merge(
        feature,
        left_on="patch_sample_index",
        right_on="feature_sample_index",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(patch) or len(merged) != len(feature):
        raise ValueError(
            "Patch and feature prediction CSVs must have one row per matching sample_index: "
            f"patch={len(patch)} feature={len(feature)} merged={len(merged)}"
        )
    target_mismatch = int((merged["patch_target_index"] != merged["feature_target_index"]).sum())
    path_mismatch = int((merged["patch_image_path"] != merged["feature_image_path"]).sum())
    if target_mismatch or path_mismatch:
        raise ValueError(
            "Patch and feature prediction CSVs are not aligned: "
            f"target_mismatch={target_mismatch}, path_mismatch={path_mismatch}"
        )
    audit = {
        "rows": int(len(merged)),
        "base_prediction_mismatch": int(
            (merged["patch_base_prediction"] != merged["feature_base_prediction"]).sum()
        ),
        "patch_csv": str(Path(patch_csv).resolve()),
        "feature_csv": str(Path(feature_csv).resolve()),
        "base_source": "patch",
    }
    return merged, audit


def _stack_features(df: pd.DataFrame) -> np.ndarray:
    parts = [df[f"patch_prob_{index}"].to_numpy(dtype=np.float32) for index in range(5)]
    patch_v0 = df["patch_verifier_0_1_prob_0"].to_numpy(dtype=np.float32)
    patch_v1 = df["patch_verifier_0_1_prob_1"].to_numpy(dtype=np.float32)
    feature_v0 = df["feature_verifier_0_1_prob_0"].to_numpy(dtype=np.float32)
    feature_v1 = df["feature_verifier_0_1_prob_1"].to_numpy(dtype=np.float32)
    prob0 = df["patch_prob_0"].to_numpy(dtype=np.float32)
    prob1 = df["patch_prob_1"].to_numpy(dtype=np.float32)
    probs = df[[f"patch_prob_{index}" for index in range(5)]].to_numpy(dtype=np.float32)
    sorted_probs = np.sort(probs, axis=1)
    parts.extend(
        [
            patch_v0,
            patch_v1,
            feature_v0,
            feature_v1,
            patch_v1 - feature_v1,
            (patch_v1 + feature_v1) / 2.0,
            np.minimum(patch_v1, feature_v1),
            np.maximum(patch_v1, feature_v1),
            prob1 - prob0,
            np.abs(prob1 - prob0),
            sorted_probs[:, -1],
            sorted_probs[:, -1] - sorted_probs[:, -2],
            (
                df["patch_final_prediction"].to_numpy(dtype=np.int64)
                != df["patch_base_prediction"].to_numpy(dtype=np.int64)
            ).astype(np.float32),
            (
                df["feature_final_prediction"].to_numpy(dtype=np.int64)
                != df["patch_base_prediction"].to_numpy(dtype=np.int64)
            ).astype(np.float32),
        ]
    )
    return np.vstack(parts).T.astype(np.float32, copy=False)


def _proposal_and_candidates(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = df["patch_base_prediction"].to_numpy(dtype=np.int64)
    patch_final = df["patch_final_prediction"].to_numpy(dtype=np.int64)
    feature_final = df["feature_final_prediction"].to_numpy(dtype=np.int64)
    proposal = base.copy()
    feature_changed = feature_final != base
    patch_changed = patch_final != base
    proposal[feature_changed] = feature_final[feature_changed]
    proposal[patch_changed] = patch_final[patch_changed]
    candidates = proposal != base
    source_code = np.zeros_like(base)
    source_code[feature_changed] = 1
    source_code[patch_changed] = 2
    source_code[feature_changed & patch_changed] = 3
    return proposal, candidates, source_code


def _target_for_mode(
    *,
    mode: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    proposals: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    base_predictions = np.asarray(base_predictions, dtype=np.int64)
    proposals = np.asarray(proposals, dtype=np.int64)
    if mode == "exact_correct":
        return (proposals == targets).astype(np.int64)
    if mode == "not_harm":
        return np.logical_or(base_predictions != targets, proposals == targets).astype(np.int64)
    if mode == "class1_benefit":
        correction = np.logical_and(proposals == targets, base_predictions != targets)
        suppress_false_class1 = np.logical_and.reduce(
            [base_predictions == 1, proposals != 1, targets != 1]
        )
        return np.logical_or(correction, suppress_false_class1).astype(np.int64)
    raise ValueError(f"Unsupported target mode: {mode}")


def _make_model(*, c_value: float, max_iter: int, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=max(1e-6, float(c_value)),
            class_weight="balanced",
            max_iter=max(100, int(max_iter)),
            solver="lbfgs",
            random_state=int(seed),
        ),
    )


def _metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    base_predictions: np.ndarray,
) -> Dict[str, object]:
    labels = [0, 1, 2, 3, 4]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    changed = predictions.astype(np.int64) != base_predictions.astype(np.int64)
    before_correct = base_predictions.astype(np.int64) == targets.astype(np.int64)
    after_correct = predictions.astype(np.int64) == targets.astype(np.int64)
    return {
        "accuracy": float((predictions == targets).mean()) if len(targets) else 0.0,
        "macro_f1": float(f1_score(targets, predictions, average="macro")) if len(targets) else 0.0,
        "class1_precision": float(precision[1]),
        "class1_recall": float(recall[1]),
        "class1_f1": float(f1[1]),
        "class1_support": int(support[1]),
        "changed": int(changed.sum()),
        "corrections": int(np.logical_and(changed, np.logical_and(~before_correct, after_correct)).sum()),
        "harms": int(np.logical_and(changed, np.logical_and(before_correct, ~after_correct)).sum()),
        "neutral_changes": int(np.logical_and(changed, before_correct == after_correct).sum()),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).astype(int).tolist(),
    }


def _write_predictions(
    path: Path,
    *,
    df: pd.DataFrame,
    proposals: np.ndarray,
    candidate_mask: np.ndarray,
    source_code: np.ndarray,
    meta_probability: np.ndarray,
    accepted: np.ndarray,
    final_predictions: np.ndarray,
) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "proposal_prediction",
        "final_prediction",
        "candidate",
        "accepted",
        "proposal_source",
        "meta_accept_probability",
        "patch_final_prediction",
        "feature_final_prediction",
        "patch_verifier_0_1_prob_1",
        "feature_verifier_0_1_prob_1",
    ] + [f"prob_{index}" for index in range(5)]
    path.parent.mkdir(parents=True, exist_ok=True)
    source_names = {0: "", 1: "feature", 2: "patch", 3: "patch+feature"}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in df.iterrows():
            out = {
                "split": row.get("patch_split", ""),
                "sample_index": int(row["patch_sample_index"]),
                "image_path": str(row["patch_image_path"]),
                "target_index": int(row["patch_target_index"]),
                "base_prediction": int(row["patch_base_prediction"]),
                "proposal_prediction": int(proposals[index]),
                "final_prediction": int(final_predictions[index]),
                "candidate": int(bool(candidate_mask[index])),
                "accepted": int(bool(accepted[index])),
                "proposal_source": source_names.get(int(source_code[index]), ""),
                "meta_accept_probability": float(meta_probability[index]) if candidate_mask[index] else "",
                "patch_final_prediction": int(row["patch_final_prediction"]),
                "feature_final_prediction": int(row["feature_final_prediction"]),
                "patch_verifier_0_1_prob_1": float(row["patch_verifier_0_1_prob_1"]),
                "feature_verifier_0_1_prob_1": float(row["feature_verifier_0_1_prob_1"]),
            }
            for class_index in range(5):
                out[f"prob_{class_index}"] = float(row[f"patch_prob_{class_index}"])
            writer.writerow(out)


def _write_changed_cases(path: Path, predictions_csv: Path) -> None:
    df = pd.read_csv(predictions_csv)
    changed = df[df["accepted"].astype(int) == 1].copy()
    columns = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "proposal_prediction",
        "final_prediction",
        "proposal_source",
        "meta_accept_probability",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    changed[columns].to_csv(path, index=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_audit = _read_merged(args.patch_train_predictions, args.feature_train_predictions)
    eval_df, eval_audit = _read_merged(args.patch_eval_predictions, args.feature_eval_predictions)
    train_targets = train_df["patch_target_index"].to_numpy(dtype=np.int64)
    eval_targets = eval_df["patch_target_index"].to_numpy(dtype=np.int64)
    train_base = train_df["patch_base_prediction"].to_numpy(dtype=np.int64)
    eval_base = eval_df["patch_base_prediction"].to_numpy(dtype=np.int64)
    train_proposal, train_candidates, train_source = _proposal_and_candidates(train_df)
    eval_proposal, eval_candidates, eval_source = _proposal_and_candidates(eval_df)

    train_target = _target_for_mode(
        mode=str(args.target_mode),
        targets=train_targets,
        base_predictions=train_base,
        proposals=train_proposal,
    )
    candidate_target = train_target[train_candidates]
    if int(train_candidates.sum()) < 4 or int(np.unique(candidate_target).shape[0]) < 2:
        raise ValueError(
            "Not enough positive and negative train candidates for stacked verifier: "
            f"candidates={int(train_candidates.sum())}, positives={int(candidate_target.sum())}"
        )
    folds = max(2, min(int(args.oof_folds), int(np.bincount(candidate_target, minlength=2).min())))
    model = _make_model(c_value=float(args.logistic_c), max_iter=int(args.max_iter), seed=int(args.seed))
    train_features = _stack_features(train_df)
    eval_features = _stack_features(eval_df)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(args.seed))
    train_meta_probability = np.zeros(train_targets.shape[0], dtype=np.float32)
    train_meta_probability[train_candidates] = cross_val_predict(
        model,
        train_features[train_candidates],
        candidate_target,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1].astype(np.float32, copy=False)
    model.fit(train_features[train_candidates], candidate_target)
    eval_meta_probability = np.zeros(eval_targets.shape[0], dtype=np.float32)
    eval_meta_probability[eval_candidates] = model.predict_proba(eval_features[eval_candidates])[
        :, 1
    ].astype(np.float32, copy=False)

    threshold = float(args.decision_threshold)
    train_accepted = np.logical_and(train_candidates, train_meta_probability >= threshold)
    eval_accepted = np.logical_and(eval_candidates, eval_meta_probability >= threshold)
    train_final = train_base.copy()
    eval_final = eval_base.copy()
    train_final[train_accepted] = train_proposal[train_accepted]
    eval_final[eval_accepted] = eval_proposal[eval_accepted]

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    _write_predictions(
        train_dir / "predictions.csv",
        df=train_df,
        proposals=train_proposal,
        candidate_mask=train_candidates,
        source_code=train_source,
        meta_probability=train_meta_probability,
        accepted=train_accepted,
        final_predictions=train_final,
    )
    _write_predictions(
        eval_dir / "predictions.csv",
        df=eval_df,
        proposals=eval_proposal,
        candidate_mask=eval_candidates,
        source_code=eval_source,
        meta_probability=eval_meta_probability,
        accepted=eval_accepted,
        final_predictions=eval_final,
    )
    _write_changed_cases(train_dir / "changed_cases_for_xai.csv", train_dir / "predictions.csv")
    _write_changed_cases(eval_dir / "changed_cases_for_xai.csv", eval_dir / "predictions.csv")

    summary: Dict[str, object] = {
        "output_dir": str(output_dir.resolve()),
        "method": "stacked selective verifier over union of patch-evidence and feature-verifier 0-1 proposals",
        "target_mode": str(args.target_mode),
        "decision_threshold": threshold,
        "settings": {
            "logistic_c": float(args.logistic_c),
            "max_iter": int(args.max_iter),
            "oof_folds": folds,
            "seed": int(args.seed),
        },
        "research_basis": [
            "Wolpert 1992 stacked generalization: second-level learner should use out-of-fold base predictions.",
            "Geifman & El-Yaniv selective classification: gate risky decisions instead of forcing every candidate through.",
            "SelectiveNet: integrated reject-option framing motivates an accept/reject reliability head.",
        ],
        "alignment": {"train": train_audit, "eval": eval_audit},
        "train_candidates": int(train_candidates.sum()),
        "eval_candidates": int(eval_candidates.sum()),
        "train_candidate_target_positive": int(candidate_target.sum()),
        "train": {
            "base": _metrics(train_targets, train_base, train_base),
            "proposal_union": _metrics(train_targets, train_proposal, train_base),
            "stacked": _metrics(train_targets, train_final, train_base),
        },
        "eval": {
            "base": _metrics(eval_targets, eval_base, eval_base),
            "proposal_union": _metrics(eval_targets, eval_proposal, eval_base),
            "stacked": _metrics(eval_targets, eval_final, eval_base),
        },
        "leakage_guard": (
            "Meta model is fitted on train candidates using train-OOF verifier probabilities; "
            "eval split is diagnostic only. Do not pass test CSVs unless doing a final frozen audit."
        ),
    }
    (train_dir / "metrics.json").write_text(
        json.dumps(
            {
                "base": summary["train"]["base"],
                "proposal_union": summary["train"]["proposal_union"],
                "stacked": summary["train"]["stacked"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (eval_dir / "metrics.json").write_text(
        json.dumps(
            {
                "base": summary["eval"]["base"],
                "proposal_union": summary["eval"]["proposal_union"],
                "stacked": summary["eval"]["stacked"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

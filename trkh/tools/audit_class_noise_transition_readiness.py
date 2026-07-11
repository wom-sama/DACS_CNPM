from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from trkh.tools.audit_teacher_uncertainty_critic_readiness import (
    PredictionTable,
    _load_prediction_table,
)


def proxy_forward_transition(table: PredictionTable) -> tuple[np.ndarray, np.ndarray]:
    """Estimate P(observed label | proxy latent class) from expert predictions."""
    class_count = int(table.probabilities.shape[1])
    counts = np.zeros((class_count, class_count), dtype=np.int64)
    for observed, latent in zip(table.targets.tolist(), table.predictions.tolist()):
        counts[int(latent), int(observed)] += 1
    support = counts.sum(axis=1)
    transition = counts.astype(np.float64)
    transition /= np.maximum(support[:, None], 1)
    return transition, support


def _matrix_payload(matrix: np.ndarray, support: np.ndarray) -> Dict[str, object]:
    return {
        "matrix": matrix.tolist(),
        "latent_support": support.tolist(),
        "diagonal": np.diag(matrix).tolist(),
        "condition_number": float(np.linalg.cond(matrix)),
    }


def _assert_aligned(tables: Mapping[str, PredictionTable], *, split: str) -> None:
    names = list(tables)
    if not names:
        raise ValueError(f"At least one {split} expert is required.")
    reference = tables[names[0]]
    for name in names[1:]:
        table = tables[name]
        if not np.array_equal(reference.sample_indices, table.sample_indices):
            raise ValueError(f"{split} sample_index mismatch: {names[0]} vs {name}")
        if not np.array_equal(reference.targets, table.targets):
            raise ValueError(f"{split} target mismatch: {names[0]} vs {name}")
        if reference.probabilities.shape[1] != table.probabilities.shape[1]:
            raise ValueError(f"{split} class-count mismatch: {names[0]} vs {name}")


def _pairwise_focus_differences(
    matrices: Mapping[str, np.ndarray], focus_class_index: int
) -> list[dict[str, object]]:
    names = sorted(matrices)
    rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            difference = np.abs(matrices[left] - matrices[right])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "mean_absolute_difference": float(difference.mean()),
                    "max_absolute_difference": float(difference.max()),
                    "focus_row_l1": float(difference[int(focus_class_index)].sum()),
                }
            )
    return rows


def _write_matrix_csv(
    path: Path,
    payloads: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
) -> None:
    rows = []
    for split, expert, matrix, support in payloads:
        for latent_class in range(matrix.shape[0]):
            row: Dict[str, object] = {
                "split": split,
                "expert": expert,
                "latent_class": latent_class,
                "latent_support": int(support[latent_class]),
                "diagonal_probability": float(matrix[latent_class, latent_class]),
            }
            row.update(
                {
                    f"observed_probability_{observed}": float(matrix[latent_class, observed])
                    for observed in range(matrix.shape[1])
                }
            )
            rows.append(row)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_class_noise_transition_readiness(
    *,
    train_experts: Mapping[str, Path],
    validation_experts: Mapping[str, Path],
    output_dir: Path,
    focus_class_index: int = 1,
    min_focus_diagonal: float = 0.70,
    min_other_diagonal: float = 0.80,
    min_focus_latent_support: int = 100,
    max_condition_number: float = 5.0,
    max_train_validation_mean_abs: float = 0.03,
    max_train_validation_focus_l1: float = 0.15,
    max_cross_expert_focus_l1: float = 0.15,
) -> Dict[str, object]:
    if set(train_experts) != set(validation_experts):
        raise ValueError("Train and validation expert names must match exactly.")
    if len(train_experts) < 2:
        raise ValueError("At least two independent experts are required for identifiability.")
    for name, path in train_experts.items():
        if "oof" not in str(Path(path)).lower():
            raise ValueError(f"Train expert must be explicitly OOF/fold-safe ({name}): {path}")

    train_tables = {
        name: _load_prediction_table(f"train_{name}", Path(path))
        for name, path in train_experts.items()
    }
    validation_tables = {
        name: _load_prediction_table(f"validation_{name}", Path(path))
        for name, path in validation_experts.items()
    }
    _assert_aligned(train_tables, split="train")
    _assert_aligned(validation_tables, split="validation")
    class_counts = {
        table.probabilities.shape[1]
        for table in [*train_tables.values(), *validation_tables.values()]
    }
    if len(class_counts) != 1:
        raise ValueError("Train/validation expert class counts do not match.")
    class_count = int(next(iter(class_counts)))
    focus = int(focus_class_index)
    if not 0 <= focus < class_count:
        raise ValueError("focus_class_index is outside the prediction class range.")

    matrices: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "validation": {}}
    supports: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "validation": {}}
    payloads: Dict[str, Dict[str, object]] = {"train": {}, "validation": {}}
    matrix_csv_payloads = []
    for split, tables in (("train", train_tables), ("validation", validation_tables)):
        for name, table in tables.items():
            matrix, support = proxy_forward_transition(table)
            matrices[split][name] = matrix
            supports[split][name] = support
            payloads[split][name] = {
                "source_csv": str(table.path.resolve()),
                "samples": int(len(table.targets)),
                **_matrix_payload(matrix, support),
            }
            matrix_csv_payloads.append((split, name, matrix, support))

    stability = {}
    blockers = []
    for name in sorted(train_experts):
        train_matrix = matrices["train"][name]
        validation_matrix = matrices["validation"][name]
        difference = np.abs(train_matrix - validation_matrix)
        stability[name] = {
            "mean_absolute_difference": float(difference.mean()),
            "max_absolute_difference": float(difference.max()),
            "focus_row_l1": float(difference[focus].sum()),
        }
        if float(difference.mean()) > float(max_train_validation_mean_abs):
            blockers.append(
                f"train_validation_mean_abs_above_max:{name}:"
                f"{difference.mean()}>{max_train_validation_mean_abs}"
            )
        if float(difference[focus].sum()) > float(max_train_validation_focus_l1):
            blockers.append(
                f"train_validation_focus_l1_above_max:{name}:"
                f"{difference[focus].sum()}>{max_train_validation_focus_l1}"
            )

    cross_expert = {
        split: _pairwise_focus_differences(matrices[split], focus)
        for split in ("train", "validation")
    }
    for split, rows in cross_expert.items():
        for row in rows:
            if float(row["focus_row_l1"]) > float(max_cross_expert_focus_l1):
                blockers.append(
                    f"cross_expert_focus_l1_above_max:{split}:{row['left']}:{row['right']}:"
                    f"{row['focus_row_l1']}>{max_cross_expert_focus_l1}"
                )

    matrix_checks = {}
    for split in ("train", "validation"):
        matrix_checks[split] = {}
        for name in sorted(matrices[split]):
            matrix = matrices[split][name]
            support = supports[split][name]
            diagonal = np.diag(matrix)
            condition_number = float(np.linalg.cond(matrix))
            matrix_checks[split][name] = {
                "focus_diagonal": float(diagonal[focus]),
                "minimum_focus_diagonal": float(min_focus_diagonal),
                "minimum_other_diagonal_observed": float(
                    np.min(np.delete(diagonal, focus))
                ),
                "minimum_other_diagonal": float(min_other_diagonal),
                "focus_latent_support": int(support[focus]),
                "minimum_focus_latent_support": int(min_focus_latent_support),
                "condition_number": condition_number,
                "maximum_condition_number": float(max_condition_number),
            }
            if float(diagonal[focus]) < float(min_focus_diagonal):
                blockers.append(
                    f"focus_diagonal_below_min:{split}:{name}:"
                    f"{diagonal[focus]}<{min_focus_diagonal}"
                )
            if float(np.min(np.delete(diagonal, focus))) < float(min_other_diagonal):
                blockers.append(
                    f"other_diagonal_below_min:{split}:{name}:"
                    f"{np.min(np.delete(diagonal, focus))}<{min_other_diagonal}"
                )
            if int(support[focus]) < int(min_focus_latent_support):
                blockers.append(
                    f"focus_latent_support_below_min:{split}:{name}:"
                    f"{support[focus]}<{min_focus_latent_support}"
                )
            if not np.isfinite(condition_number) or condition_number > float(max_condition_number):
                blockers.append(
                    f"condition_number_above_max:{split}:{name}:"
                    f"{condition_number}>{max_condition_number}"
                )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix_csv(output_dir / "proxy_transition_matrices.csv", matrix_csv_payloads)
    smoke_gate_ready = not blockers
    summary: Dict[str, object] = {
        "mode": "class_noise_transition_readiness",
        "research_scope": (
            "Patrini et al. CVPR 2017 forward loss correction and the ICLR 2017 "
            "noise-adaptation-layer formulation"
        ),
        "note": (
            "Expert predictions are only latent-clean proxies, not ground-truth clean labels. "
            "The audit tests whether a class-conditional transition matrix is identifiable "
            "enough to justify implementation; it does not claim to estimate the true noise process."
        ),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "focus_class_index": focus,
        "class_count": class_count,
        "thresholds": {
            "min_focus_diagonal": float(min_focus_diagonal),
            "min_other_diagonal": float(min_other_diagonal),
            "min_focus_latent_support": int(min_focus_latent_support),
            "max_condition_number": float(max_condition_number),
            "max_train_validation_mean_abs": float(max_train_validation_mean_abs),
            "max_train_validation_focus_l1": float(max_train_validation_focus_l1),
            "max_cross_expert_focus_l1": float(max_cross_expert_focus_l1),
        },
        "train": payloads["train"],
        "validation": payloads["validation"],
        "train_validation_stability": stability,
        "cross_expert_stability": cross_expert,
        "gate_checks": matrix_checks,
        "smoke_gate_ready": bool(smoke_gate_ready),
        "blocking_reasons": blockers,
        "decision": (
            "Eligible for one bounded forward-correction smoke; this is not a clean-label claim."
            if smoke_gate_ready
            else "Do not implement forward loss correction or a learnable noise-adaptation layer. "
            "The current OOF/validation experts do not identify a stable class-conditional noise matrix."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    (output_dir / "README.md").write_text(
        "# Class-noise transition readiness\n\n"
        "This no-test diagnostic treats independent expert predictions only as latent-clean proxies "
        "and checks whether their implied forward transition matrices agree across experts and "
        "train-OOF/validation. Failure means the matrix is not identifiable enough to train with.\n",
        encoding="utf-8",
    )
    return summary


def _parse_named_paths(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expert input must be NAME=CSV: {value!r}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"Expert name is empty or duplicated: {name!r}")
        output[name] = Path(path_text.strip())
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit identifiability of a class-conditional label-noise transition matrix."
    )
    parser.add_argument("--train-expert", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--validation-expert", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, object]:
    return build_class_noise_transition_readiness(
        train_experts=_parse_named_paths(args.train_expert),
        validation_experts=_parse_named_paths(args.validation_expert),
        output_dir=args.output_dir,
        focus_class_index=args.focus_class_index,
    )


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "smoke_gate_ready": summary["smoke_gate_ready"],
                "blocking_reasons": summary["blocking_reasons"],
                "output": "summary.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

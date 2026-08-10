from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


TeacherRows = Dict[int, Dict[str, str]]


def _probability_columns(fieldnames: Sequence[str]) -> List[str]:
    columns = [name for name in fieldnames if name.startswith("prob_")]
    if not columns:
        raise ValueError("Teacher CSV must contain prob_* columns.")
    return sorted(columns, key=lambda value: int(value.split("_", 1)[1]))


def _read_teacher_csv(path: Path) -> Tuple[TeacherRows, List[str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Teacher CSV has no header: {path}")
        if "sample_index" not in reader.fieldnames:
            raise ValueError(f"Teacher CSV must contain sample_index: {path}")
        if "path" not in reader.fieldnames:
            raise ValueError(f"Teacher CSV must contain path: {path}")
        prob_columns = _probability_columns(reader.fieldnames)
        rows: TeacherRows = {}
        for row in reader:
            sample_index = int(row["sample_index"])
            if sample_index in rows:
                raise ValueError(f"Duplicate sample_index={sample_index} in {path}")
            probs = [float(row[column]) for column in prob_columns]
            if not all(math.isfinite(value) and value >= 0.0 for value in probs):
                raise ValueError(f"Invalid probabilities for sample_index={sample_index} in {path}")
            if sum(probs) <= 0.0:
                raise ValueError(f"Probability sum must be positive for sample_index={sample_index} in {path}")
            rows[sample_index] = row
    if not rows:
        raise ValueError(f"Teacher CSV has no rows: {path}")
    expected = set(range(max(rows) + 1))
    missing = sorted(expected.difference(rows))
    if missing:
        raise ValueError(f"Teacher CSV sample_index is not contiguous in {path}: preview={missing[:5]}")
    return rows, prob_columns


def build_mixed_indices(
    lengths: Sequence[int],
    weights: Sequence[float],
    *,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    if len(lengths) != len(weights):
        raise ValueError("lengths and weights must have the same length.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    mixed: List[Tuple[int, int]] = []
    for source_index, (length, weight) in enumerate(zip(lengths, weights)):
        length = int(length)
        weight = max(0.0, float(weight))
        if length <= 0 or weight <= 0.0:
            continue
        whole = int(math.floor(weight))
        fraction = float(weight) - float(whole)
        if whole <= 0:
            for sample_index in range(length):
                if torch.rand(1, generator=generator).item() < fraction:
                    mixed.append((source_index, sample_index))
            continue
        for sample_index in range(length):
            mixed.extend([(source_index, sample_index)] * whole)
            if fraction > 0.0 and torch.rand(1, generator=generator).item() < fraction:
                mixed.append((source_index, sample_index))
    if len(mixed) > 1:
        order = torch.randperm(len(mixed), generator=generator).tolist()
        mixed = [mixed[index] for index in order]
    if not mixed:
        raise ValueError("Mixed teacher cache would be empty.")
    return mixed


def build_mixed_teacher_probability_cache(
    *,
    primary_csv: Path,
    auxiliary_csv: Path,
    output_csv: Path,
    auxiliary_weight: float,
    primary_weight: float = 1.0,
    seed: int = 42,
    primary_name: str = "primary",
    auxiliary_name: str = "auxiliary",
) -> Mapping[str, object]:
    primary_rows, primary_prob_columns = _read_teacher_csv(primary_csv)
    auxiliary_rows, auxiliary_prob_columns = _read_teacher_csv(auxiliary_csv)
    if primary_prob_columns != auxiliary_prob_columns:
        raise ValueError(
            "Primary and auxiliary teacher CSVs must have the same probability columns: "
            f"primary={primary_prob_columns}, auxiliary={auxiliary_prob_columns}"
        )
    sources = [
        (str(primary_name), primary_rows),
        (str(auxiliary_name), auxiliary_rows),
    ]
    weights = [float(primary_weight), float(auxiliary_weight)]
    lengths = [len(primary_rows), len(auxiliary_rows)]
    mixed_indices = build_mixed_indices(lengths, weights, seed=seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "source_name",
        "source_sample_index",
        "path",
        *primary_prob_columns,
    ]
    source_counts = [0 for _ in sources]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mixed_index, (source_index, source_sample_index) in enumerate(mixed_indices):
            source_name, rows = sources[int(source_index)]
            source_counts[int(source_index)] += 1
            row = rows[int(source_sample_index)]
            writer.writerow(
                {
                    "sample_index": str(mixed_index),
                    "source_name": source_name,
                    "source_sample_index": str(source_sample_index),
                    "path": row["path"],
                    **{column: row[column] for column in primary_prob_columns},
                }
            )
    summary = {
        "output_csv": str(output_csv),
        "seed": int(seed),
        "primary_csv": str(primary_csv),
        "auxiliary_csv": str(auxiliary_csv),
        "primary_weight": float(primary_weight),
        "auxiliary_weight": float(auxiliary_weight),
        "primary_rows": int(len(primary_rows)),
        "auxiliary_rows": int(len(auxiliary_rows)),
        "mixed_rows": int(len(mixed_indices)),
        "source_counts": {
            str(primary_name): int(source_counts[0]),
            str(auxiliary_name): int(source_counts[1]),
        },
        "probability_columns": list(primary_prob_columns),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict sample_index teacher-probability CSV for MixedTrainDataset "
            "from primary and auxiliary teacher caches."
        )
    )
    parser.add_argument("--primary-csv", type=Path, required=True)
    parser.add_argument("--auxiliary-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--auxiliary-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-name", default="primary")
    parser.add_argument("--auxiliary-name", default="auxiliary")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_mixed_teacher_probability_cache(
        primary_csv=args.primary_csv,
        auxiliary_csv=args.auxiliary_csv,
        output_csv=args.output_csv,
        primary_weight=args.primary_weight,
        auxiliary_weight=args.auxiliary_weight,
        seed=args.seed,
        primary_name=args.primary_name,
        auxiliary_name=args.auxiliary_name,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

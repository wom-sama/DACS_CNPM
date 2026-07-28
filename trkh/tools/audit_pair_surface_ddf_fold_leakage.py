from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EXPECTED_ROWS = 763
EXPECTED_FOLDS = frozenset(range(5))
REQUIRED_COHORT_ARRAYS = (
    "sample_indices",
    "targets",
    "folds",
    "source_stems",
)
EXPECTED_MANIFEST_FIELDS = (
    "split",
    "source_split",
    "leakage_group",
    "source_image",
    "output_image",
    "output_label",
)
FORBIDDEN_INPUT_COMPONENTS = frozenset({"val", "valid", "validation", "test"})
NUMERIC_WINDOWS = (1, 3)
NUMERIC_UNION_WINDOW = 3
IMAGE_NUMBER_PATTERN = re.compile(r"^image_(\d+)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CohortRow:
    position: int
    sample_index: int
    target: int
    fold: int
    source_stem: str
    stem_key: str
    leakage_group: str = ""
    image_number: Optional[int] = None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Metadata-only audit of the existing Pair-Surface DDF cohort folds. "
            "The tool reads no pixels or candidate scores and never assigns folds."
        )
    )
    parser.add_argument("--cohort-arrays", type=Path, required=True)
    parser.add_argument("--yolo-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _assert_metadata_input(path: Path, *, suffix: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Metadata input does not exist: {resolved}")
    if resolved.suffix.casefold() != suffix.casefold():
        raise ValueError(f"Unexpected metadata input suffix for {resolved}; expected {suffix}")
    components = {
        component.casefold()
        for component in str(resolved).replace("\\", "/").split("/")
        if component
    }
    forbidden = sorted(components.intersection(FORBIDDEN_INPUT_COMPONENTS))
    if forbidden:
        raise ValueError(
            f"Validation/test path components are forbidden for metadata input: {forbidden}"
        )
    return resolved


def _cohort_stem(value: object) -> Tuple[str, str]:
    stem = str(value)
    if stem != stem.strip() or not stem:
        raise ValueError(f"Unexpected source_stem value: {stem!r}")
    if "/" in stem or "\\" in stem or stem in {".", ".."}:
        raise ValueError(f"source_stem must be a stem, not a path: {stem!r}")
    return stem, stem.casefold()


def _manifest_output_stem(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Manifest output_image is empty")
    normalized = raw.replace("\\", "/")
    name = PurePosixPath(normalized).name
    stem = PurePosixPath(name).stem
    if not name or not stem or stem in {".", ".."}:
        raise ValueError(f"Manifest output_image has an unexpected format: {raw!r}")
    return stem.casefold()


def _load_cohort(path: Path) -> List[CohortRow]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_COHORT_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"Cohort archive is missing required arrays: {missing}")
        # Deliberately read only fold-audit metadata. Other archive members may
        # contain keeper outputs or geometry and are outside this tool's scope.
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in REQUIRED_COHORT_ARRAYS
        }

    for name in REQUIRED_COHORT_ARRAYS:
        if arrays[name].shape != (EXPECTED_ROWS,):
            raise ValueError(
                f"Unexpected cohort {name} shape: {arrays[name].shape}; "
                f"expected ({EXPECTED_ROWS},)"
            )
    for name in ("sample_indices", "targets", "folds"):
        if arrays[name].dtype != np.dtype(np.int64):
            raise ValueError(
                f"Unexpected cohort {name} dtype: {arrays[name].dtype}; expected int64"
            )
    if arrays["source_stems"].dtype.kind != "U":
        raise ValueError(
            "Unexpected cohort source_stems dtype: "
            f"{arrays['source_stems'].dtype}; expected fixed-width Unicode"
        )

    sample_indices = arrays["sample_indices"]
    targets = arrays["targets"]
    folds = arrays["folds"]
    if np.unique(sample_indices).size != EXPECTED_ROWS or bool((sample_indices < 0).any()):
        raise ValueError("Cohort sample_indices must be unique non-negative int64 values")
    observed_folds = {int(value) for value in folds.tolist()}
    if observed_folds != EXPECTED_FOLDS:
        raise ValueError(
            f"Unexpected cohort fold support: {sorted(observed_folds)}; expected 0..4"
        )
    if bool(((targets < 0) | (targets > 4)).any()):
        raise ValueError("Cohort targets must be class indices in [0, 4]")

    rows: List[CohortRow] = []
    for position in range(EXPECTED_ROWS):
        source_stem, stem_key = _cohort_stem(arrays["source_stems"][position])
        match = IMAGE_NUMBER_PATTERN.fullmatch(source_stem)
        rows.append(
            CohortRow(
                position=position,
                sample_index=int(sample_indices[position]),
                target=int(targets[position]),
                fold=int(folds[position]),
                source_stem=source_stem,
                stem_key=stem_key,
                image_number=int(match.group(1)) if match is not None else None,
            )
        )
    return rows


def _load_manifest_assignments(
    path: Path,
    *,
    rows: Sequence[CohortRow],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    wanted_stems = {row.stem_key for row in rows}
    row_count_by_stem = Counter(row.stem_key for row in rows)
    assignments: Dict[str, str] = {}
    manifest_rows_scanned = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_MANIFEST_FIELDS:
            raise ValueError(
                "Unexpected YOLO manifest fields: "
                f"{reader.fieldnames}; expected {list(EXPECTED_MANIFEST_FIELDS)}"
            )
        for line_number, manifest_row in enumerate(reader, start=2):
            manifest_rows_scanned += 1
            if None in manifest_row or any(value is None for value in manifest_row.values()):
                raise ValueError(f"Malformed YOLO manifest row at line {line_number}")
            stem_key = _manifest_output_stem(manifest_row["output_image"])
            if stem_key not in wanted_stems:
                continue
            if stem_key in assignments:
                raise ValueError(
                    "Casefold-exact manifest join is ambiguous for cohort stem "
                    f"{stem_key!r} at line {line_number}"
                )
            split = manifest_row["split"].strip().casefold()
            if split != "train":
                raise ValueError(
                    f"Mapped manifest split must be train for cohort stem {stem_key!r}; "
                    f"got {manifest_row['split']!r}"
                )
            leakage_group = manifest_row["leakage_group"].strip()
            if not leakage_group:
                raise ValueError(
                    f"Mapped manifest leakage_group is empty for cohort stem {stem_key!r}"
                )
            assignments[stem_key] = leakage_group

    mapped_rows = sum(row_count_by_stem[stem] for stem in assignments)
    missing_stems = sorted(wanted_stems.difference(assignments))
    if mapped_rows != EXPECTED_ROWS or missing_stems:
        preview = missing_stems[:10]
        raise ValueError(
            "Casefold-exact manifest mapping must cover all 763 cohort rows; "
            f"mapped_rows={mapped_rows}, missing_unique_stems={len(missing_stems)}, "
            f"missing_preview={preview}"
        )
    return assignments, {
        "manifest_rows_scanned": manifest_rows_scanned,
        "mapped_rows": mapped_rows,
        "mapped_unique_stems": len(assignments),
        "unmapped_rows": EXPECTED_ROWS - mapped_rows,
        "unmapped_unique_stems": len(missing_stems),
    }


def _cross_fold_pair_counts(rows: Sequence[CohortRow]) -> Dict[str, int]:
    total = 0
    same_label = 0
    cross_label = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left.fold == right.fold:
                continue
            total += 1
            if left.target == right.target:
                same_label += 1
            else:
                cross_label += 1
    return {
        "total": total,
        "same_label": same_label,
        "cross_label": cross_label,
    }


def _counter_payload(counter: Mapping[object, int]) -> Dict[str, int]:
    return {
        str(key): int(counter[key])
        for key in sorted(counter, key=lambda value: (str(type(value)), str(value)))
    }


def _group_overlap(
    rows: Sequence[CohortRow],
    *,
    group_values: Iterable[Tuple[str, CohortRow]],
) -> Dict[str, object]:
    groups: Dict[str, List[CohortRow]] = defaultdict(list)
    for group_id, row in group_values:
        groups[group_id].append(row)

    fold_span_histogram: Counter[int] = Counter()
    cross_fold_span_histogram: Counter[int] = Counter()
    aggregate_pairs = Counter({"total": 0, "same_label": 0, "cross_label": 0})
    covered_positions = set()
    covered_stems = set()
    details: List[Dict[str, object]] = []
    for group_id in sorted(groups):
        group_rows = sorted(groups[group_id], key=lambda row: row.position)
        folds = sorted({row.fold for row in group_rows})
        fold_span_histogram[len(folds)] += 1
        if len(folds) <= 1:
            continue
        cross_fold_span_histogram[len(folds)] += 1
        pair_counts = _cross_fold_pair_counts(group_rows)
        aggregate_pairs.update(pair_counts)
        covered_positions.update(row.position for row in group_rows)
        covered_stems.update(row.stem_key for row in group_rows)
        details.append(
            {
                "group_id": group_id,
                "row_count": len(group_rows),
                "unique_stem_count": len({row.stem_key for row in group_rows}),
                "folds": folds,
                "target_counts": _counter_payload(
                    Counter(row.target for row in group_rows)
                ),
                "cross_fold_row_pairs": pair_counts,
                "sample_indices": sorted(row.sample_index for row in group_rows),
                "stems": sorted({row.stem_key for row in group_rows}),
            }
        )
    return {
        "group_count": len(groups),
        "cross_fold_group_count": len(details),
        "covered_row_count": len(covered_positions),
        "covered_unique_stem_count": len(covered_stems),
        "cross_fold_row_pairs": dict(aggregate_pairs),
        "fold_span_histogram": _counter_payload(fold_span_histogram),
        "cross_fold_span_histogram": _counter_payload(cross_fold_span_histogram),
        "cross_fold_groups": details,
    }


def _rows_by_stem(rows: Sequence[CohortRow]) -> Dict[str, List[CohortRow]]:
    result: Dict[str, List[CohortRow]] = defaultdict(list)
    for row in rows:
        result[row.stem_key].append(row)
    return result


def _numeric_stem_pairs(
    rows_by_stem: Mapping[str, Sequence[CohortRow]],
    *,
    maximum_window: int,
) -> List[Tuple[int, str, Sequence[CohortRow], int, str, Sequence[CohortRow]]]:
    parsed = sorted(
        (
            int(stem_rows[0].image_number),
            stem,
            stem_rows,
        )
        for stem, stem_rows in rows_by_stem.items()
        if stem_rows[0].image_number is not None
    )
    pairs = []
    for left_index, (left_number, left_stem, left_rows) in enumerate(parsed):
        for right_number, right_stem, right_rows in parsed[left_index + 1 :]:
            distance = right_number - left_number
            if distance > maximum_window:
                break
            pairs.append(
                (
                    left_number,
                    left_stem,
                    left_rows,
                    right_number,
                    right_stem,
                    right_rows,
                )
            )
    return pairs


def _numeric_window_summary(
    pairs: Sequence[
        Tuple[int, str, Sequence[CohortRow], int, str, Sequence[CohortRow]]
    ],
    *,
    window: int,
) -> Dict[str, object]:
    candidate_pair_count = 0
    aggregate_pairs = Counter({"total": 0, "same_label": 0, "cross_label": 0})
    covered_positions = set()
    covered_stems = set()
    details: List[Dict[str, object]] = []
    for (
        left_number,
        left_stem,
        left_rows,
        right_number,
        right_stem,
        right_rows,
    ) in pairs:
        distance = right_number - left_number
        if distance > window:
            continue
        candidate_pair_count += 1
        pair_counts = _cross_fold_pair_counts([*left_rows, *right_rows])
        # Exclude within-stem pairs: this section reports distinct-stem numeric
        # adjacency, while exact-stem overlap is reported separately.
        left_internal = _cross_fold_pair_counts(left_rows)
        right_internal = _cross_fold_pair_counts(right_rows)
        pair_counts = {
            key: pair_counts[key] - left_internal[key] - right_internal[key]
            for key in ("total", "same_label", "cross_label")
        }
        if pair_counts["total"] == 0:
            continue
        aggregate_pairs.update(pair_counts)
        covered_positions.update(row.position for row in [*left_rows, *right_rows])
        covered_stems.update((left_stem, right_stem))
        details.append(
            {
                "left_stem": left_stem,
                "right_stem": right_stem,
                "left_image_number": left_number,
                "right_image_number": right_number,
                "distance": distance,
                "left_folds": sorted({row.fold for row in left_rows}),
                "right_folds": sorted({row.fold for row in right_rows}),
                "cross_fold_row_pairs": pair_counts,
            }
        )
    return {
        "window": window,
        "distance_rule": (
            "distinct casefold stems with abs(Image_N_i-Image_N_j) <= window"
        ),
        "all_unique_stem_pair_count": candidate_pair_count,
        "cross_fold_unique_stem_pair_count": len(details),
        "cross_fold_row_pair_count": int(aggregate_pairs["total"]),
        "cross_fold_row_pairs": dict(aggregate_pairs),
        "covered_row_count": len(covered_positions),
        "covered_unique_stem_count": len(covered_stems),
        "cross_fold_unique_stem_pairs": details,
    }


def _nearest_rank_percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(int(value) for value in values)
    rank = max(1, int(math.ceil(float(percentile) * len(ordered))))
    return ordered[rank - 1]


def _union_all(union_find: UnionFind, indices: Sequence[int]) -> int:
    if not indices:
        return 0
    first = int(indices[0])
    for index in indices[1:]:
        union_find.union(first, int(index))
    return max(0, len(indices) - 1)


def _transitive_component_summary(
    rows: Sequence[CohortRow],
    *,
    numeric_pairs: Sequence[
        Tuple[int, str, Sequence[CohortRow], int, str, Sequence[CohortRow]]
    ],
) -> Dict[str, object]:
    union_find = UnionFind(len(rows))
    by_stem = _rows_by_stem(rows)
    by_leakage_group: Dict[str, List[CohortRow]] = defaultdict(list)
    for row in rows:
        by_leakage_group[row.leakage_group].append(row)

    exact_links = sum(
        _union_all(union_find, [row.position for row in stem_rows])
        for stem_rows in by_stem.values()
    )
    leakage_group_links = sum(
        _union_all(union_find, [row.position for row in group_rows])
        for group_rows in by_leakage_group.values()
    )
    for _, _, left_rows, _, _, right_rows in numeric_pairs:
        union_find.union(left_rows[0].position, right_rows[0].position)

    components: Dict[int, List[CohortRow]] = defaultdict(list)
    for row in rows:
        components[union_find.find(row.position)].append(row)

    row_sizes = [len(component) for component in components.values()]
    stem_sizes = [
        len({row.stem_key for row in component})
        for component in components.values()
    ]
    component_fold_spans = Counter(
        len({row.fold for row in component})
        for component in components.values()
    )
    component_class_spans = Counter(
        len({row.target for row in component})
        for component in components.values()
    )
    component_target_sets = Counter(
        ",".join(str(value) for value in sorted({row.target for row in component}))
        for component in components.values()
    )
    cross_fold_components = [
        component
        for component in components.values()
        if len({row.fold for row in component}) > 1
    ]

    observed_classes = sorted({row.target for row in rows})
    class_support = {}
    for target in observed_classes:
        target_rows = [row for row in rows if row.target == target]
        class_support[str(target)] = {
            "row_count": len(target_rows),
            "unique_stem_count": len({row.stem_key for row in target_rows}),
            "component_count": sum(
                any(row.target == target for row in component)
                for component in components.values()
            ),
        }

    return {
        "relations": [
            "casefold_exact_stem",
            "manifest_leakage_group",
            "numeric_Image_N_adjacency_window_le_3",
        ],
        "fold_assignment_performed": False,
        "relation_link_counts": {
            "exact_stem_row_links": exact_links,
            "leakage_group_row_links": leakage_group_links,
            "numeric_unique_stem_pairs_window_le_3": len(numeric_pairs),
        },
        "component_count": len(components),
        "row_size": {
            "maximum": max(row_sizes),
            "p95_nearest_rank": _nearest_rank_percentile(row_sizes, 0.95),
        },
        "unique_stem_size": {
            "maximum": max(stem_sizes),
            "p95_nearest_rank": _nearest_rank_percentile(stem_sizes, 0.95),
        },
        "cross_fold_component_count": len(cross_fold_components),
        "cross_fold_covered_row_count": sum(
            len(component) for component in cross_fold_components
        ),
        "fold_span_histogram": _counter_payload(component_fold_spans),
        "class_support": class_support,
        "class_span_histogram": _counter_payload(component_class_spans),
        "target_set_histogram": _counter_payload(component_target_sets),
        "mixed_class_component_count": sum(
            count for span, count in component_class_spans.items() if span > 1
        ),
    }


def build_report(cohort_arrays: Path, yolo_manifest: Path) -> Dict[str, object]:
    cohort_path = _assert_metadata_input(cohort_arrays, suffix=".npz")
    manifest_path = _assert_metadata_input(yolo_manifest, suffix=".csv")
    rows = _load_cohort(cohort_path)
    assignments, mapping = _load_manifest_assignments(manifest_path, rows=rows)
    rows = [
        replace(row, leakage_group=assignments[row.stem_key])
        for row in rows
    ]
    rows_by_stem = _rows_by_stem(rows)
    numeric_pairs = _numeric_stem_pairs(
        rows_by_stem,
        maximum_window=NUMERIC_UNION_WINDOW,
    )

    parsed_stems = {
        stem
        for stem, stem_rows in rows_by_stem.items()
        if stem_rows[0].image_number is not None
    }
    parsed_rows = sum(len(rows_by_stem[stem]) for stem in parsed_stems)
    return {
        "schema_version": 1,
        "state": "metadata_only_pair_surface_ddf_fold_leakage_audit",
        "inputs": {
            "cohort_arrays": str(cohort_path),
            "yolo_manifest": str(manifest_path),
        },
        "access_contract": {
            "declared_metadata_files_read": 2,
            "pixel_files_opened": 0,
            "image_or_label_paths_opened": 0,
            "candidate_score_arrays_read": 0,
            "fold_assignment_performed": False,
        },
        "cohort": {
            "rows": len(rows),
            "unique_sample_indices": len({row.sample_index for row in rows}),
            "unique_casefold_stems": len(rows_by_stem),
            "fold_counts": _counter_payload(Counter(row.fold for row in rows)),
            "target_counts": _counter_payload(Counter(row.target for row in rows)),
        },
        "manifest_mapping": {
            "join": "casefold_exact_output_image_stem_to_source_stem",
            "all_mapped_rows_are_train": True,
            **mapping,
        },
        "old_fold_exact_stem_overlap": _group_overlap(
            rows,
            group_values=((row.stem_key, row) for row in rows),
        ),
        "manifest_leakage_group_cross_fold": _group_overlap(
            rows,
            group_values=((row.leakage_group, row) for row in rows),
        ),
        "numeric_image_n_cross_fold": {
            "pattern": r"^Image_(\d+)$ (case-insensitive)",
            "parsed_row_count": parsed_rows,
            "parsed_unique_stem_count": len(parsed_stems),
            "unparsed_row_count": len(rows) - parsed_rows,
            "unparsed_unique_stem_count": len(rows_by_stem) - len(parsed_stems),
            "windows": {
                str(window): _numeric_window_summary(
                    numeric_pairs,
                    window=window,
                )
                for window in NUMERIC_WINDOWS
            },
        },
        "transitive_union_window_le_3": _transitive_component_summary(
            rows,
            numeric_pairs=numeric_pairs,
        ),
    }


def _write_json_atomic(
    report: Mapping[str, object],
    output_json: Path,
    *,
    protected_inputs: Sequence[Path],
) -> None:
    if output_json.suffix.casefold() != ".json":
        raise ValueError(f"Output must be a JSON file: {output_json}")
    resolved_output = output_json.resolve()
    if resolved_output in {path.resolve() for path in protected_inputs}:
        raise ValueError("Output JSON must not overwrite an input")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(
        f".{resolved_output.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, resolved_output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = build_report(args.cohort_arrays, args.yolo_manifest)
    _write_json_atomic(
        report,
        args.output_json,
        protected_inputs=(args.cohort_arrays, args.yolo_manifest),
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "rows": report["cohort"]["rows"],
                "component_count": report["transitive_union_window_le_3"][
                    "component_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

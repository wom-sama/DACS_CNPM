from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import ntpath
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPOSITORY_ROOT = Path(r"D:\DataAI\AIEx\TRKH")

COHORT_PATH = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_materialized_20260725"
    / "cohort_arrays.npz"
)
YOLO_MANIFEST_PATH = (
    REPOSITORY_ROOT.parent / "newdataset" / "yolo_f" / "manifest.csv"
)
CVAT_EXPORTER_PATH = (
    REPOSITORY_ROOT.parent / "CVAT_" / "src" / "cvat_nhai" / "yolo_editor.py"
)
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_PAIR_SURFACE_DDF_V2R2_FOLD_MANIFEST_20260729.json"
)
OUTPUT_SHA_PATH = OUTPUT_PATH.with_suffix(".sha256")

PREDECESSOR_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_PAIR_SURFACE_DDF_V2_FOLD_MANIFEST_20260729.json"
)
PREDECESSOR_OUTPUT_SHA256 = (
    "8a084b171b12c4d8a4e85d09d748d30cdf46191ea7160748cdd03519b38f924b"
)
PREDECESSOR_SIDECAR_PATH = PREDECESSOR_OUTPUT_PATH.with_suffix(".sha256")
PREDECESSOR_SIDECAR_SHA256 = (
    "fe54c79b5976336768e6a6c2291d55a8d91bf004bb2095eb9ee20591b9a31912"
)
COHORT_CONTAINER_LINEAGE_SHA256 = (
    "a835d3498319d3d25d83c0279ab24d6fcc703a2266ff5f091dc2273bbb8baa9d"
)
YOLO_MANIFEST_SHA256 = (
    "eb16e09cd20fff8480b50aa79e7d5fec779e4614398911e2e257ba6a37642ff2"
)
CVAT_EXPORTER_SHA256 = (
    "65e6056dac1793eaa9775b458a448828bfd0debd7adc4d9acb58807b7b74e4b5"
)
CVAT_EXPORTER_GIT_COMMIT = "0c7fd8b873d34d18c6f6f6588b772996528ba0ec"
CVAT_EXPORTER_GIT_BLOB = "d405bf3522f8e7809488b4842f9ffb988a069195"
CANONICAL_METADATA_SHA256 = (
    "f0fa476fd4be3765b1ed8c780e09d52037a0215554a7579dbd259d00cab02b60"
)
COMPONENT_SET_SHA256 = (
    "be6cece69ec9d6ea00ad7c953e49af3fb359ff0bc57c4f09bfca7c2895f51c37"
)
COMPONENT_ORDER_SHA256 = (
    "e10e1045e6ede63ade0cb436b684be7a6ab377d0d27f74480ac8721756045ccb"
)
EDGE_SET_SHA256 = (
    "0b4791ce986f98d6d36d2be0f4ba3cdf7cce3f0354850aa84e54e5c2de0dbc61"
)
MAPPING_SHA256 = (
    "c0726d114df69290d68d9f9e1a40857074ccb1a66e5ca41883caa004f2b0ea40"
)
COMPONENT_ASSIGNMENTS_SHA256 = (
    "1ce9837be1a7e60d6c6cef0768d68eb29c22f2fa326f3558b8d0395e3dbd1819"
)

EXPECTED_PYTHON = "3.9.11"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.13.1"
EXPECTED_HIGHS_WRAPPER_SHA256 = (
    "31df702fefbd1c5e5a4cbd8d7dbc012e6ad3ca07532ca233502b6709a99ff91d"
)

METADATA_ARRAY_KEYS = (
    "sample_indices",
    "targets",
    "source_stems",
    "image_paths",
)
FORBIDDEN_ARRAY_KEYS = (
    "folds",
    "keeper_probabilities",
    "model_boxes",
    "crop_boxes",
    "label_paths",
)
EXPECTED_ARCHIVE_KEYS = tuple(sorted((*METADATA_ARRAY_KEYS, *FORBIDDEN_ARRAY_KEYS)))
EXPECTED_METADATA_ARRAYS = {
    "sample_indices": {
        "shape": (763,),
        "dtype": "int64",
        "sha256": "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05",
    },
    "targets": {
        "shape": (763,),
        "dtype": "int64",
        "sha256": "15c43ecc7335c7a7febc4e0fbf622df7ad593fc52e5d9f14e5cc27acc16fc000",
    },
    "source_stems": {
        "shape": (763,),
        "dtype": "<U11",
        "sha256": "b127d4d6ef8d1ae6b7f6137f8b7ff461c105fd85de414afb2abdf678a3065b21",
    },
    "image_paths": {
        "shape": (763,),
        "dtype": "<U61",
        "sha256": "6c42e9e7d5d7a34ecffee72dd96edf5240c54eed291d1a6f86c712016cdfb002",
    },
}

ROWS = 763
UNIQUE_STEMS = 735
FOLDS = 5
IMAGE_N_WINDOW = 3
TARGET_COUNTS = {0: 158, 1: 541, 2: 54, 3: 0, 4: 10}
ROW_BOUNDS = (152, 153)
CLASS_BOUNDS = {
    0: (31, 32),
    1: (108, 109),
    2: (10, 11),
    3: (0, 0),
    4: (2, 2),
}
EXPECTED_COMPONENTS = 158
EXPECTED_SINGLETON_COMPONENTS = 64
EXPECTED_LARGEST_COMPONENT = 82
EXPECTED_EDGES = 1372
EXPECTED_FEASIBILITY_CALLS = 451

SOLVER_OPTIONS = {
    "mip_rel_gap": 0.0,
    "presolve": True,
    "random_seed": 0,
    "threads": 1,
    "time_limit": 30.0,
}

LEGACY_FOLD_DIAGNOSTICS = {
    "role": "context_only_not_read_or_used_by_this_builder",
    "source": "read_only_preimplementation_audit_20260729",
    "components_spanning_legacy_folds": 85,
    "components_total": 158,
    "rows_in_spanning_components": 679,
    "leakage_groups_spanning_legacy_folds": 150,
    "leakage_groups_total": 531,
    "sorted_consecutive_w3_links_crossing_legacy_folds": 464,
    "sorted_consecutive_w3_links_total": 558,
    "assignment_influence": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _target_counts(rows: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    materialized = list(rows)
    return {
        str(target): sum(int(row["target"]) == target for row in materialized)
        for target in range(5)
    }


def _histogram(values: Iterable[int]) -> Dict[str, int]:
    counts: Dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def _normalized_output_path(value: str) -> str:
    normalized = ntpath.normpath(value.strip()).replace("/", "\\")
    return normalized.casefold()


def _assert_expected_repository_path() -> None:
    actual = REPOSITORY_ROOT.resolve()
    expected = EXPECTED_REPOSITORY_ROOT.resolve()
    if actual != expected:
        raise RuntimeError(
            "Fold-manifest writes are restricted to the expected repository: "
            f"expected={expected}, actual={actual}"
        )


def _load_cohort_metadata(path: Path = COHORT_PATH) -> Dict[str, np.ndarray]:
    # The ZIP central-directory member names are inspected so schema drift
    # fails closed. Only the four metadata members are deserialized; legacy
    # folds, probabilities, boxes and label paths are never loaded or hashed.
    with np.load(path, allow_pickle=False) as archive:
        observed_keys = tuple(sorted(str(key) for key in archive.files))
        if observed_keys != EXPECTED_ARCHIVE_KEYS:
            raise ValueError(
                "Cohort archive member schema differs: "
                f"observed={observed_keys}, expected={EXPECTED_ARCHIVE_KEYS}"
            )
        arrays = {
            key: np.asarray(archive[key]).copy()
            for key in METADATA_ARRAY_KEYS
        }
    shapes = {key: tuple(value.shape) for key, value in arrays.items()}
    if any(len(shape) != 1 for shape in shapes.values()):
        raise ValueError(f"Cohort metadata arrays must be one-dimensional: {shapes}")
    lengths = {shape[0] for shape in shapes.values()}
    if len(lengths) != 1:
        raise ValueError(f"Cohort metadata lengths differ: {shapes}")
    observed_contract = {
        key: {
            "shape": tuple(int(value) for value in array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_sha256(array),
        }
        for key, array in arrays.items()
    }
    if observed_contract != EXPECTED_METADATA_ARRAYS:
        raise ValueError(
            "Cohort metadata member contract differs: "
            f"observed={observed_contract}, expected={EXPECTED_METADATA_ARRAYS}"
        )
    return arrays


def _load_yolo_manifest(
    path: Path = YOLO_MANIFEST_PATH,
) -> Dict[str, object]:
    by_output: Dict[str, Dict[str, str]] = {}
    group_splits: Dict[str, set] = {}
    group_rows: Dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "leakage_group", "output_image"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "YOLO manifest columns differ: "
                f"required={sorted(required)}, observed={reader.fieldnames}"
            )
        total_rows = 0
        for line_number, row in enumerate(reader, start=2):
            total_rows += 1
            raw_output_image = row.get("output_image")
            raw_leakage_group = row.get("leakage_group")
            raw_split = row.get("split")
            if (
                raw_output_image is None
                or raw_leakage_group is None
                or raw_split is None
            ):
                raise ValueError(
                    f"Malformed manifest row at CSV line {line_number}"
                )
            output_image = raw_output_image.strip()
            leakage_group = raw_leakage_group.strip()
            split = raw_split.strip().lower()
            if not output_image or not leakage_group or not split:
                raise ValueError(
                    f"Blank manifest identity field at CSV line {line_number}"
                )
            key = _normalized_output_path(output_image)
            if key in by_output:
                raise ValueError(
                    f"Duplicate manifest output_image at CSV line {line_number}: "
                    f"{output_image}"
                )
            by_output[key] = {
                "output_image": output_image,
                "leakage_group": leakage_group,
                "split": split,
            }
            group_splits.setdefault(leakage_group, set()).add(split)
            group_rows[leakage_group] = group_rows.get(leakage_group, 0) + 1
    return {
        "by_output": by_output,
        "group_splits": group_splits,
        "group_rows": group_rows,
        "total_rows": total_rows,
    }


def _canonical_rows(
    cohort: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_output = manifest["by_output"]
    group_splits = manifest["group_splits"]
    group_rows = manifest["group_rows"]
    if not isinstance(by_output, Mapping):
        raise TypeError("Manifest output index is malformed")
    if not isinstance(group_splits, Mapping) or not isinstance(group_rows, Mapping):
        raise TypeError("Manifest group audit is malformed")

    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for position in range(len(cohort["sample_indices"])):
        sample_index = int(cohort["sample_indices"][position])
        target = int(cohort["targets"][position])
        stem = str(cohort["source_stems"][position]).strip().casefold()
        image_path = str(cohort["image_paths"][position]).strip()
        manifest_row = by_output.get(_normalized_output_path(image_path))
        if manifest_row is None:
            missing.append(image_path)
            continue
        split = str(manifest_row["split"])
        if split != "train":
            raise ValueError(
                f"Cohort row is not in YOLO train split: {image_path} -> {split}"
            )
        match = re.fullmatch(r"image_(\d+)", stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot parse Image_N source stem: {stem}")
        output_basename = ntpath.basename(str(manifest_row["output_image"]))
        output_stem = ntpath.splitext(output_basename)[0].casefold()
        if output_stem != stem:
            raise ValueError(
                "Cohort stem differs from joined manifest output: "
                f"cohort={stem}, manifest={output_stem}"
            )
        rows.append(
            {
                "sample_index": sample_index,
                "stem": stem,
                "target": target,
                "image_n": int(match.group(1)),
                "leakage_group": str(manifest_row["leakage_group"]),
                "image_rel": f"images/train/{output_basename}",
            }
        )
    if missing:
        raise ValueError(
            f"Manifest join missing {len(missing)} cohort paths; first={missing[0]}"
        )
    rows.sort(key=lambda row: int(row["sample_index"]))
    sample_indices = [int(row["sample_index"]) for row in rows]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("sample_index is not unique")
    if any(int(row["target"]) not in range(5) for row in rows):
        raise ValueError("Target outside the locked five-class index range")

    cohort_groups = {str(row["leakage_group"]) for row in rows}
    crossing = {
        group: sorted(str(split) for split in group_splits[group])
        for group in cohort_groups
        if set(group_splits[group]) != {"train"}
    }
    if crossing:
        first = sorted(crossing)[0]
        raise ValueError(
            "A cohort leakage_group crosses YOLO output splits: "
            f"{first} -> {crossing[first]}"
        )
    cohort_group_sizes: Dict[str, int] = {}
    for row in rows:
        group = str(row["leakage_group"])
        cohort_group_sizes[group] = cohort_group_sizes.get(group, 0) + 1
    audit = {
        "manifest_rows": int(manifest["total_rows"]),
        "joined_rows": len(rows),
        "missing_rows": 0,
        "ambiguous_rows": 0,
        "cohort_output_splits": {"train": len(rows)},
        "cohort_leakage_groups": len(cohort_groups),
        "cohort_leakage_group_size_histogram": _histogram(
            cohort_group_sizes.values()
        ),
        "related_manifest_rows": sum(int(group_rows[group]) for group in cohort_groups),
        "related_groups_crossing_output_splits": 0,
    }
    return rows, audit


def _add_relation(
    edge_relations: MutableMapping[Tuple[int, int], set],
    left: int,
    right: int,
    relation: str,
) -> None:
    if left == right:
        return
    edge = (left, right) if left < right else (right, left)
    edge_relations.setdefault(edge, set()).add(relation)


def _build_relation_graph(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    count = len(rows)
    parent = list(range(count))
    sizes = [1] * count

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root] or (
            sizes[left_root] == sizes[right_root] and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    edge_relations: Dict[Tuple[int, int], set] = {}
    for field, relation in (
        ("stem", "exact_stem"),
        ("leakage_group", "exact_leakage_group"),
    ):
        groups: Dict[str, List[int]] = {}
        for position, row in enumerate(rows):
            groups.setdefault(str(row[field]), []).append(position)
        for positions in groups.values():
            for left, right in itertools.combinations(positions, 2):
                union(left, right)
                _add_relation(
                    edge_relations,
                    int(rows[left]["sample_index"]),
                    int(rows[right]["sample_index"]),
                    relation,
                )

    numeric = sorted(
        (int(row["image_n"]), position)
        for position, row in enumerate(rows)
    )
    for left_offset, (left_n, left_position) in enumerate(numeric):
        for right_n, right_position in numeric[left_offset + 1 :]:
            if right_n - left_n > IMAGE_N_WINDOW:
                break
            union(left_position, right_position)
            _add_relation(
                edge_relations,
                int(rows[left_position]["sample_index"]),
                int(rows[right_position]["sample_index"]),
                "image_n_w3",
            )

    by_root: Dict[int, List[int]] = {}
    for position in range(count):
        by_root.setdefault(find(position), []).append(position)
    components: List[Dict[str, object]] = []
    for positions in by_root.values():
        component_rows = sorted(
            (dict(rows[position]) for position in positions),
            key=lambda row: int(row["sample_index"]),
        )
        component_sha = _json_sha256(component_rows)
        components.append(
            {
                "component_sha256": component_sha,
                "rows_internal": component_rows,
            }
        )
    components.sort(key=lambda item: str(item["component_sha256"]))
    component_hashes = [str(item["component_sha256"]) for item in components]
    if len(component_hashes) != len(set(component_hashes)):
        raise ValueError("Canonical component SHA-256 collision")

    component_set = [
        {
            "component_sha256": str(component["component_sha256"]),
            "sample_indices": sorted(
                int(row["sample_index"])
                for row in component["rows_internal"]
            ),
        }
        for component in components
    ]
    edges = [
        {
            "left_sample_index": left,
            "right_sample_index": right,
            "relations": sorted(str(value) for value in relations),
        }
        for (left, right), relations in sorted(edge_relations.items())
    ]
    relation_counts = {
        relation: sum(relation in edge["relations"] for edge in edges)
        for relation in ("exact_stem", "exact_leakage_group", "image_n_w3")
    }
    component_sizes = [len(component["rows_internal"]) for component in components]
    return {
        "components_internal": components,
        "edges_internal": edges,
        "component_set_sha256": _json_sha256(component_set),
        "component_order_sha256": _json_sha256(component_hashes),
        "edge_set_sha256": _json_sha256(edges),
        "edge_count": len(edges),
        "relation_edge_counts": relation_counts,
        "component_count": len(components),
        "component_size_histogram": _histogram(component_sizes),
        "singleton_components": sum(size == 1 for size in component_sizes),
        "largest_component_rows": max(component_sizes, default=0),
    }


def _solve(
    objective: np.ndarray,
    integrality: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    constraint: LinearConstraint,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*",
            category=RuntimeWarning,
        )
        return milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraint,
            options=SOLVER_OPTIONS,
        )


def _assign_components(
    components: Sequence[Mapping[str, object]],
    *,
    fold_count: int = FOLDS,
    row_bounds: Tuple[int, int] = ROW_BOUNDS,
    class_bounds: Mapping[int, Tuple[int, int]] = CLASS_BOUNDS,
) -> Dict[str, object]:
    component_count = len(components)
    class_order = sorted(int(target) for target in class_bounds)
    component_rows = np.asarray(
        [len(component["rows_internal"]) for component in components],
        dtype=np.int64,
    )
    component_classes = np.asarray(
        [
            [
                sum(
                    int(row["target"]) == target
                    for row in component["rows_internal"]
                )
                for target in class_order
            ]
            for component in components
        ],
        dtype=np.int64,
    )
    total_rows = int(component_rows.sum())
    if not row_bounds[0] * fold_count <= total_rows <= row_bounds[1] * fold_count:
        raise ValueError("Global row total is incompatible with fold bounds")
    for class_position, target in enumerate(class_order):
        total = int(component_classes[:, class_position].sum())
        lower, upper = class_bounds[target]
        if not lower * fold_count <= total <= upper * fold_count:
            raise ValueError(
                f"Global class-{target} total is incompatible with fold bounds"
            )
    if component_count and int(component_rows.max()) > row_bounds[1]:
        raise ValueError("A transitive component exceeds the maximum fold size")
    for class_position, target in enumerate(class_order):
        if not component_count:
            continue
        largest_class_count = int(component_classes[:, class_position].max())
        if largest_class_count > class_bounds[target][1]:
            raise ValueError(
                f"A transitive component exceeds the class-{target} fold bound"
            )

    variable_count = component_count * fold_count
    blocks = []
    constraint_lower: List[float] = []
    constraint_upper: List[float] = []

    assignment = lil_matrix((component_count, variable_count), dtype=np.float64)
    for component in range(component_count):
        for fold in range(fold_count):
            assignment[component, component * fold_count + fold] = 1.0
    blocks.append(assignment.tocsr())
    constraint_lower.extend([1.0] * component_count)
    constraint_upper.extend([1.0] * component_count)

    rows_matrix = lil_matrix((fold_count, variable_count), dtype=np.float64)
    for fold in range(fold_count):
        for component in range(component_count):
            rows_matrix[fold, component * fold_count + fold] = component_rows[component]
    blocks.append(rows_matrix.tocsr())
    constraint_lower.extend([float(row_bounds[0])] * fold_count)
    constraint_upper.extend([float(row_bounds[1])] * fold_count)

    for class_position, target in enumerate(class_order):
        class_matrix = lil_matrix((fold_count, variable_count), dtype=np.float64)
        for fold in range(fold_count):
            for component in range(component_count):
                class_matrix[fold, component * fold_count + fold] = (
                    component_classes[component, class_position]
                )
        blocks.append(class_matrix.tocsr())
        lower, upper = class_bounds[target]
        constraint_lower.extend([float(lower)] * fold_count)
        constraint_upper.extend([float(upper)] * fold_count)

    constraint = LinearConstraint(
        vstack(blocks).tocsr(),
        np.asarray(constraint_lower, dtype=np.float64),
        np.asarray(constraint_upper, dtype=np.float64),
    )
    objective = np.zeros(variable_count, dtype=np.float64)
    integrality = np.ones(variable_count, dtype=np.int8)
    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.ones(variable_count, dtype=np.float64)

    calls = 0
    base = _solve(objective, integrality, lower, upper, constraint)
    calls += 1
    if int(base.status) != 0:
        raise RuntimeError(
            "Base fold MILP is not optimal: "
            f"status={base.status}, message={base.message}"
        )

    chosen: List[int] = []
    for component in range(component_count):
        accepted = False
        for fold in range(fold_count):
            candidate_lower = lower.copy()
            candidate_upper = upper.copy()
            variable = component * fold_count + fold
            candidate_lower[variable] = 1.0
            candidate_upper[variable] = 1.0
            result = _solve(
                objective,
                integrality,
                candidate_lower,
                candidate_upper,
                constraint,
            )
            calls += 1
            if int(result.status) == 0:
                lower[variable] = 1.0
                upper[variable] = 1.0
                chosen.append(fold)
                accepted = True
                break
            if int(result.status) != 2:
                raise RuntimeError(
                    "Lexicographic feasibility probe did not prove an answer: "
                    f"component={component}, fold={fold}, "
                    f"status={result.status}, message={result.message}"
                )
        if not accepted:
            raise RuntimeError(f"No feasible fold remains for component {component}")

    final = _solve(objective, integrality, lower, upper, constraint)
    calls += 1
    if int(final.status) != 0 or final.x is None:
        raise RuntimeError(
            "Final locked fold MILP is not optimal: "
            f"status={final.status}, message={final.message}"
        )
    integrality_error = float(np.max(np.abs(final.x - np.rint(final.x))))
    if integrality_error != 0.0:
        raise RuntimeError(
            f"Final fold assignment is not exactly integral: {integrality_error}"
        )
    return {
        "fold_by_component": chosen,
        "fold_vector": "".join(str(fold) for fold in chosen),
        "solver_calls": calls,
        "status": int(final.status),
        "status_text": str(final.message),
        "integrality_max_abs": integrality_error,
    }


def _materialize_assignment(
    components: Sequence[Mapping[str, object]],
    assignment: Mapping[str, object],
) -> Dict[str, object]:
    fold_by_component = [int(value) for value in assignment["fold_by_component"]]
    if len(fold_by_component) != len(components):
        raise RuntimeError("Fold vector length differs from component count")
    mapping: List[Dict[str, object]] = []
    component_payloads: List[Dict[str, object]] = []
    for component, fold in zip(components, fold_by_component):
        rows = list(component["rows_internal"])
        component_sha = str(component["component_sha256"])
        for row in rows:
            mapping.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "component_sha256": component_sha,
                    "fold": fold,
                }
            )
        component_payloads.append(
            {
                "component_sha256": component_sha,
                "fold": fold,
                "rows": len(rows),
                "target_counts": _target_counts(rows),
                "sample_indices": sorted(int(row["sample_index"]) for row in rows),
                "image_n_min": min(int(row["image_n"]) for row in rows),
                "image_n_max": max(int(row["image_n"]) for row in rows),
            }
        )
    mapping.sort(key=lambda row: int(row["sample_index"]))
    if len(mapping) != len({int(row["sample_index"]) for row in mapping}):
        raise RuntimeError("Final mapping does not contain each sample exactly once")

    fold_payloads: List[Dict[str, object]] = []
    for fold in range(FOLDS):
        current_components = [
            component
            for component in component_payloads
            if int(component["fold"]) == fold
        ]
        current_rows = [
            row
            for component, component_fold in zip(components, fold_by_component)
            if component_fold == fold
            for row in component["rows_internal"]
        ]
        fold_payloads.append(
            {
                "fold": fold,
                "rows": len(current_rows),
                "target_counts": _target_counts(current_rows),
                "components": len(current_components),
                "largest_component_rows": max(
                    (int(component["rows"]) for component in current_components),
                    default=0,
                ),
            }
        )
    return {
        "rows": mapping,
        "mapping_sha256": _json_sha256(mapping),
        "component_assignments_sha256": _json_sha256(
            [
                {
                    "component_sha256": component["component_sha256"],
                    "fold": component["fold"],
                }
                for component in component_payloads
            ]
        ),
        "components": component_payloads,
        "folds": fold_payloads,
    }


def _calibration_payload(
    rows: Sequence[Mapping[str, object]],
    fold_by_sample: Mapping[int, int],
    component_by_sample: Mapping[int, str],
) -> Dict[str, object]:
    outer_payloads: List[Dict[str, object]] = []
    for held_fold in range(FOLDS):
        calibration_fold = (held_fold + 1) % FOLDS
        held = [row for row in rows if fold_by_sample[int(row["sample_index"])] == held_fold]
        calibration = [
            row
            for row in rows
            if fold_by_sample[int(row["sample_index"])] == calibration_fold
        ]
        fit = [
            row
            for row in rows
            if fold_by_sample[int(row["sample_index"])]
            not in {held_fold, calibration_fold}
        ]
        if int(_target_counts(calibration)["4"]) != 2:
            raise RuntimeError("Simple calibration fold does not have class-4 support 2")
        if int(_target_counts(fit)["4"]) != 6:
            raise RuntimeError("Three-fold fit partition does not have class-4 support 6")
        component_sets = {
            "held": {
                component_by_sample[int(row["sample_index"])] for row in held
            },
            "calibration": {
                component_by_sample[int(row["sample_index"])]
                for row in calibration
            },
            "fit": {
                component_by_sample[int(row["sample_index"])] for row in fit
            },
        }
        overlap = sum(
            len(component_sets[left].intersection(component_sets[right]))
            for left, right in (
                ("fit", "calibration"),
                ("fit", "held"),
                ("calibration", "held"),
            )
        )
        if overlap != 0:
            raise RuntimeError("A component overlaps fit/calibration/held partitions")
        outer_payloads.append(
            {
                "held_fold": held_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(
                    set(range(FOLDS)) - {held_fold, calibration_fold}
                ),
                "held": {"rows": len(held), "target_counts": _target_counts(held)},
                "calibration": {
                    "rows": len(calibration),
                    "target_counts": _target_counts(calibration),
                },
                "fit": {"rows": len(fit), "target_counts": _target_counts(fit)},
                "candidate_model_rule": (
                    "fit one candidate model only on fit_folds; the same fitted "
                    "model predicts both calibration and held rows"
                ),
                "component_overlap_fit_calibration_held": overlap,
            }
        )
    return {
        "authorized_mode": "fixed_simple_one_fold_calibration_only",
        "mapping": "calibration_fold=(held_fold+1)%5",
        "fit_fold_count": 3,
        "alternative_calibration_modes_authorized": False,
        "outer_folds": outer_payloads,
        "limitations": [
            "The 763-row cohort has zero class-3 support.",
            "Do not claim full five-class calibration from this cohort.",
        ],
    }


def _runtime_payload(*, enforce: bool) -> Dict[str, object]:
    from scipy.optimize._highs import _highs_wrapper

    wrapper_path = Path(_highs_wrapper.__file__).resolve()
    payload = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "solver": "scipy.optimize.milp",
        "backend": "SciPy-vendored HiGHS",
        "highs_wrapper_filename": wrapper_path.name,
        "highs_wrapper_sha256": _sha256(wrapper_path),
        "solver_options": dict(SOLVER_OPTIONS),
    }
    if enforce:
        expected = {
            "python_version": EXPECTED_PYTHON,
            "numpy_version": EXPECTED_NUMPY,
            "scipy_version": EXPECTED_SCIPY,
            "highs_wrapper_sha256": EXPECTED_HIGHS_WRAPPER_SHA256,
        }
        mismatches = {
            key: {"expected": value, "observed": payload[key]}
            for key, value in expected.items()
            if payload[key] != value
        }
        if mismatches:
            raise RuntimeError(f"Pinned fold-solver runtime differs: {mismatches}")
    return payload


def _assert_real_profile(
    rows: Sequence[Mapping[str, object]],
    manifest_audit: Mapping[str, object],
    graph: Mapping[str, object],
    materialized: Mapping[str, object],
) -> None:
    observed_counts = {int(key): value for key, value in _target_counts(rows).items()}
    checks = {
        "rows": (len(rows), ROWS),
        "unique_stems": (len({str(row["stem"]) for row in rows}), UNIQUE_STEMS),
        "target_counts": (observed_counts, TARGET_COUNTS),
        "cohort_leakage_groups": (
            int(manifest_audit["cohort_leakage_groups"]),
            531,
        ),
        "component_count": (int(graph["component_count"]), EXPECTED_COMPONENTS),
        "edge_count": (int(graph["edge_count"]), EXPECTED_EDGES),
        "singleton_components": (
            int(graph["singleton_components"]),
            EXPECTED_SINGLETON_COMPONENTS,
        ),
        "largest_component_rows": (
            int(graph["largest_component_rows"]),
            EXPECTED_LARGEST_COMPONENT,
        ),
        "component_set_sha256": (
            str(graph["component_set_sha256"]),
            COMPONENT_SET_SHA256,
        ),
        "component_order_sha256": (
            str(graph["component_order_sha256"]),
            COMPONENT_ORDER_SHA256,
        ),
        "edge_set_sha256": (
            str(graph["edge_set_sha256"]),
            EDGE_SET_SHA256,
        ),
        "mapping_sha256": (
            str(materialized["mapping_sha256"]),
            MAPPING_SHA256,
        ),
        "component_assignments_sha256": (
            str(materialized["component_assignments_sha256"]),
            COMPONENT_ASSIGNMENTS_SHA256,
        ),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise RuntimeError(f"Locked real-cohort profile differs: {mismatches}")


def build_manifest() -> Dict[str, object]:
    _assert_expected_repository_path()
    predecessor_sha256 = _sha256(PREDECESSOR_OUTPUT_PATH)
    if predecessor_sha256 != PREDECESSOR_OUTPUT_SHA256:
        raise RuntimeError(
            "Predecessor fold manifest differs: "
            f"observed={predecessor_sha256}, "
            f"expected={PREDECESSOR_OUTPUT_SHA256}"
        )
    predecessor_sidecar_sha256 = _sha256(PREDECESSOR_SIDECAR_PATH)
    if predecessor_sidecar_sha256 != PREDECESSOR_SIDECAR_SHA256:
        raise RuntimeError(
            "Predecessor fold-manifest SHA sidecar differs: "
            f"observed={predecessor_sidecar_sha256}, "
            f"expected={PREDECESSOR_SIDECAR_SHA256}"
        )
    observed_input_hashes = {
        "cohort_container": _sha256(COHORT_PATH),
        "yolo_manifest_csv": _sha256(YOLO_MANIFEST_PATH),
        "cvat_exporter": _sha256(CVAT_EXPORTER_PATH),
    }
    expected_input_hashes = {
        "cohort_container": COHORT_CONTAINER_LINEAGE_SHA256,
        "yolo_manifest_csv": YOLO_MANIFEST_SHA256,
        "cvat_exporter": CVAT_EXPORTER_SHA256,
    }
    if observed_input_hashes != expected_input_hashes:
        raise RuntimeError(
            "Fold-manifest input hash differs: "
            f"observed={observed_input_hashes}, expected={expected_input_hashes}"
        )
    runtime = _runtime_payload(enforce=True)
    cohort = _load_cohort_metadata()
    cohort_member_contract = {
        key: {
            "shape": [int(value) for value in cohort[key].shape],
            "dtype": str(cohort[key].dtype),
            "sha256": _array_sha256(cohort[key]),
        }
        for key in METADATA_ARRAY_KEYS
    }
    selected_member_set_sha256 = _json_sha256(cohort_member_contract)
    yolo_manifest = _load_yolo_manifest()
    rows, join_audit = _canonical_rows(cohort, yolo_manifest)
    metadata_sha = _json_sha256(rows)
    if metadata_sha != CANONICAL_METADATA_SHA256:
        raise RuntimeError(
            "Canonical cohort metadata SHA differs: "
            f"observed={metadata_sha}, expected={CANONICAL_METADATA_SHA256}"
        )
    graph = _build_relation_graph(rows)
    assignment = _assign_components(graph["components_internal"])
    if int(assignment["status"]) != 0:
        raise RuntimeError("Pinned real-cohort MILP did not return status 0")
    if int(assignment["solver_calls"]) != EXPECTED_FEASIBILITY_CALLS:
        raise RuntimeError(
            "Lexicographic feasibility decision count differs: "
            f"observed={assignment['solver_calls']}, "
            f"expected={EXPECTED_FEASIBILITY_CALLS}"
        )
    materialized = _materialize_assignment(
        graph["components_internal"],
        assignment,
    )
    _assert_real_profile(rows, join_audit, graph, materialized)

    fold_by_sample = {
        int(row["sample_index"]): int(row["fold"])
        for row in materialized["rows"]
    }
    component_by_sample = {
        int(row["sample_index"]): str(row["component_sha256"])
        for row in materialized["rows"]
    }
    cross_fold_edges = sum(
        fold_by_sample[int(edge["left_sample_index"])]
        != fold_by_sample[int(edge["right_sample_index"])]
        for edge in graph["edges_internal"]
    )
    component_cross_fold = sum(
        len(
            {
                fold_by_sample[int(row["sample_index"])]
                for row in component["rows_internal"]
            }
        )
        != 1
        for component in graph["components_internal"]
    )
    if cross_fold_edges != 0 or component_cross_fold != 0:
        raise RuntimeError("Fold-disjoint graph audit failed")

    return {
        "schema": "trkh_pair_surface_ddf_v2_fold_manifest/v2",
        "state": (
            "frozen_v2r2_provenance_hardened_train_metadata_"
            "manifest_group_numeric_neighborhood_disjoint"
        ),
        "date": "2026-07-29",
        "builder": {
            "path": "scripts/build_trkh_pair_surface_ddf_v2_fold_manifest.py",
            "sha256": _sha256(Path(__file__).resolve()),
            "default_mode": "check_only",
            "write_mode": "one_shot_explicit_--write_only_refuses_existing_pair",
            "expected_repository_root": str(EXPECTED_REPOSITORY_ROOT),
        },
        "scope": {
            "train_only": True,
            "candidate_score_independent": True,
            "legacy_fold_independent": True,
            "archive_member_names_inspected": True,
            "metadata_array_members_deserialized": list(METADATA_ARRAY_KEYS),
            "forbidden_array_members_not_deserialized": list(
                FORBIDDEN_ARRAY_KEYS
            ),
            "whole_cohort_container_bytes_hashed_for_opaque_lineage": True,
            "whole_cohort_container_hash_used_as_assignment_feature": False,
            "keeper_probabilities_or_candidate_scores_deserialized": False,
            "legacy_folds_deserialized": False,
            "dataset_manifest_full_bytes_read": True,
            "dataset_manifest_columns_semantically_used": [
                "split",
                "leakage_group",
                "output_image",
            ],
            "cross_split_manifest_identity_metadata_read_for_group_audit": True,
            "validation_pixels_labels_scores_or_checkpoints_accessed": False,
            "test_pixels_labels_scores_or_checkpoints_accessed": False,
        },
        "inputs": {
            "predecessor_manifest": {
                "path": str(PREDECESSOR_OUTPUT_PATH),
                "sha256": predecessor_sha256,
                "sha_sidecar_path": str(PREDECESSOR_SIDECAR_PATH),
                "sha_sidecar_sha256": predecessor_sidecar_sha256,
                "role": "immutable lineage and identical-mapping comparison only",
                "assignment_influence": False,
            },
            "cohort_container": {
                "path": str(COHORT_PATH),
                "sha256": observed_input_hashes["cohort_container"],
                "role": "opaque_lineage_provenance_only",
                "bytes_hashed": True,
                "decoded_for_assignment": False,
            },
            "selected_npz_members": {
                "members": cohort_member_contract,
                "selected_member_set_sha256": selected_member_set_sha256,
                "assignment_inputs": True,
            },
            "yolo_manifest_csv": {
                "path": str(YOLO_MANIFEST_PATH),
                "sha256": observed_input_hashes["yolo_manifest_csv"],
            },
            "leakage_group_exporter": {
                "path": str(CVAT_EXPORTER_PATH),
                "sha256": observed_input_hashes["cvat_exporter"],
                "git_commit": CVAT_EXPORTER_GIT_COMMIT,
                "git_blob_oid": CVAT_EXPORTER_GIT_BLOB,
                "semantics": (
                    "transitive union of normalized source-family identity and "
                    "near-visual matches using 64-bit dHash Hamming distance <=3 "
                    "plus a quantized 16x16 colour-thumbnail threshold"
                ),
                "canonical_output": "the pinned yolo manifest bytes",
                "portability_caveat": (
                    "group IDs hash absolute resolved source paths and are not "
                    "promised byte-identical after dataset-root relocation"
                ),
            },
            "manifest_join_audit": join_audit,
        },
        "canonical_metadata": {
            "fields": [
                "sample_index",
                "stem",
                "target",
                "image_n",
                "leakage_group",
                "image_rel",
            ],
            "row_order": "sample_index ascending",
            "serialization": "JSON UTF-8 sort_keys separators=(',', ':') no newline",
            "sha256": metadata_sha,
            "rows": len(rows),
            "unique_sample_indices": len({int(row["sample_index"]) for row in rows}),
            "unique_stems": len({str(row["stem"]) for row in rows}),
            "target_counts": _target_counts(rows),
        },
        "relation_graph": {
            "union_rule": (
                "transitive connected components of exact stem OR exact "
                "leakage_group OR abs(Image_N_i-Image_N_j)<=3"
            ),
            "image_n_window": IMAGE_N_WINDOW,
            "edge_definition": "all qualifying unordered row pairs",
            "edge_count": graph["edge_count"],
            "relation_edge_counts": graph["relation_edge_counts"],
            "edge_set_sha256": graph["edge_set_sha256"],
            "component_count": graph["component_count"],
            "component_size_histogram": graph["component_size_histogram"],
            "singleton_components": graph["singleton_components"],
            "largest_component_rows": graph["largest_component_rows"],
            "component_order": "component_sha256 ascending",
            "component_order_sha256": graph["component_order_sha256"],
            "component_set_sha256": graph["component_set_sha256"],
        },
        "constraints": {
            "folds": FOLDS,
            "row_bounds_per_fold": list(ROW_BOUNDS),
            "class_bounds_per_fold": {
                str(target): list(bounds)
                for target, bounds in sorted(CLASS_BOUNDS.items())
            },
            "class4_exact_per_fold": 2,
            "each_component_exactly_one_fold": True,
        },
        "solver": {
            **runtime,
            "objective": "constant_zero_feasibility_only",
            "tie_break": (
                "lexicographically smallest fold vector in canonical component "
                "order; try folds 0..4 and lock the first remaining-feasible fold"
            ),
            "status": assignment["status"],
            "status_text": assignment["status_text"],
            "integrality_max_abs": assignment["integrality_max_abs"],
            "feasibility_calls": assignment["solver_calls"],
            "fold_vector": assignment["fold_vector"],
        },
        "assignment": {
            "mapping_key": "sample_index under the locked canonical metadata SHA",
            "mapping_sha256": materialized["mapping_sha256"],
            "component_assignments_sha256": materialized[
                "component_assignments_sha256"
            ],
            "cross_fold_relation_edges": cross_fold_edges,
            "cross_fold_components": component_cross_fold,
            "rows": materialized["rows"],
            "components": materialized["components"],
            "folds": materialized["folds"],
        },
        "calibration": _calibration_payload(
            rows,
            fold_by_sample,
            component_by_sample,
        ),
        "legacy_fold_diagnostics": dict(LEGACY_FOLD_DIAGNOSTICS),
        "failure_policy": {
            "mode": "fail_closed",
            "conditions": [
                "input/exporter, canonical metadata, component-set, or mapping SHA differs",
                "row, class, stem, leakage-group, or component profile differs",
                "manifest join is missing, duplicate, blank, non-train, or crosses output splits",
                "Image_N parsing fails",
                "a component exceeds a hard row or class bound",
                "MILP does not prove OPTIMAL/INFEASIBLE at every decision",
                "the final solution is not exactly integral",
                "any relation edge or connected component crosses folds",
                "fresh-process replay differs from the frozen JSON or SHA record",
                "candidate output, legacy fold, probability, or box data influences assignment",
                "fit/calibration/held components overlap or held data fits a calibrator",
                "the output JSON or SHA sidecar already exists in write mode",
            ],
        },
        "supersession": {
            "scope": "provenance disclosure and immutable-write semantics only",
            "predecessor_schema": "trkh_pair_surface_ddf_v2_fold_manifest/v1",
            "predecessor_sha256": predecessor_sha256,
            "predecessor_mapping_sha256": MAPPING_SHA256,
            "mapping_changed": False,
            "component_set_changed": False,
            "formal_or_replay_runs_consumed_before_supersession": 0,
            "candidate_scores_observed_before_supersession": False,
        },
        "identity_with_predecessor": {
            "canonical_metadata_sha256": CANONICAL_METADATA_SHA256,
            "edge_set_sha256": EDGE_SET_SHA256,
            "component_set_sha256": COMPONENT_SET_SHA256,
            "component_order_sha256": COMPONENT_ORDER_SHA256,
            "mapping_sha256": MAPPING_SHA256,
            "component_assignments_sha256": COMPONENT_ASSIGNMENTS_SHA256,
            "fold_vector": assignment["fold_vector"],
            "all_equal": True,
        },
        "write_contract": {
            "default_check_only": True,
            "one_shot_output_absent_required": True,
            "expected_repository_root_guard": True,
            "per_file_atomic_fsync_replace": True,
            "pair_transactional": False,
            "partial_pair_fails_closed": True,
            "recovery_requires_separate_reviewed_action": True,
            "effective_only_after_builder_artifact_tests_commit_and_push": True,
        },
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_manifest(manifest: Mapping[str, object]) -> str:
    _assert_expected_repository_path()
    existing = [
        str(path)
        for path in (OUTPUT_PATH, OUTPUT_SHA_PATH)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Frozen fold-manifest outputs already exist; one-shot write refuses "
            f"overwrite: {existing}"
        )
    payload = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(OUTPUT_PATH, payload)
    _atomic_write_bytes(
        OUTPUT_SHA_PATH,
        f"{digest}  {OUTPUT_PATH.name}\n".encode("ascii"),
    )
    return digest


def _assert_check_only_matches_frozen(manifest: Mapping[str, object]) -> None:
    if not OUTPUT_PATH.is_file() or not OUTPUT_SHA_PATH.is_file():
        raise FileNotFoundError(
            "Frozen fold manifest is missing; use explicit --write only at the "
            "expected repository path"
        )
    parts = OUTPUT_SHA_PATH.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[1] != OUTPUT_PATH.name:
        raise ValueError("Frozen fold-manifest SHA record is malformed")
    actual = _sha256(OUTPUT_PATH)
    if parts[0] != actual:
        raise ValueError(
            f"Frozen fold-manifest SHA differs: recorded={parts[0]}, actual={actual}"
        )
    frozen = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    generated = json.loads(json.dumps(manifest, ensure_ascii=False))
    if generated != frozen:
        raise ValueError(
            "Regenerated fold manifest differs from frozen output: "
            f"generated={_json_sha256(generated)}, frozen={_json_sha256(frozen)}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or replay-check the provenance-hardened deterministic TRKH "
            "Pair-Surface DDF v2R2 train-only fold manifest."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--write",
        action="store_true",
        help=(
            "Explicitly perform the one-shot write of the frozen JSON and SHA "
            "sidecar. Existing output is never overwritten. Without this flag "
            "the builder is check-only."
        ),
    )
    modes.add_argument(
        "--check-only",
        action="store_true",
        help="Explicit alias for the default non-writing replay check.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest()
    if args.write:
        digest = write_manifest(manifest)
        print(f"Wrote {OUTPUT_PATH}")
        print(f"SHA-256 {digest}")
        return
    _assert_check_only_matches_frozen(manifest)
    print(
        json.dumps(
            {
                "mode": "check_only",
                "rows": manifest["canonical_metadata"]["rows"],
                "components": manifest["relation_graph"]["component_count"],
                "mapping_sha256": manifest["assignment"]["mapping_sha256"],
                "fold_rows": [
                    fold["rows"] for fold in manifest["assignment"]["folds"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

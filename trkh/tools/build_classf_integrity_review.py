from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_SPLITS = ("train", "val")
MANIFEST_FIELDS = (
    "split",
    "relative_path",
    "class_id",
    "class_name",
    "source_image_name",
    "source",
    "source_split",
    "leakage_group",
    "source_prefix",
    "source_number",
    "source_session_candidate_id",
    "source_session_size",
    "source_session_splits",
    "sha256",
    "phash64",
    "was_relabelled",
    "relabel_count",
    "original_class_ids",
    "train_review_flags",
    "diagnostic_flags",
)
REVIEW_FIELDS = (
    "issue_id",
    "priority",
    "scope",
    "issue_type",
    "split_a",
    "path_a",
    "class_a",
    "split_b",
    "path_b",
    "class_b",
    "source_prefix",
    "source_number_delta",
    "phash_distance",
    "details",
    "recommended_action",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _normalise_relative(path: Path) -> str:
    return path.as_posix()


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


def _ensure_output_outside_dataset(data_root: Path, output_dir: Path) -> None:
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        return
    raise ValueError("output_dir must be outside the immutable dataset root")


def _source_parts(row: Mapping[str, str], relative_path: str) -> Tuple[str, Optional[int]]:
    source_stem = Path(row.get("source_image", "")).stem
    if not source_stem:
        source_stem = re.sub(r"_box\d+$", "", Path(relative_path).stem, flags=re.IGNORECASE)
    match = re.match(r"^(.*?)(\d+)$", source_stem)
    if not match:
        return source_stem.casefold(), None
    prefix = match.group(1).rstrip("_- ").casefold() or "source"
    return prefix, int(match.group(2))


def _stable_sample_identity(row: Mapping[str, str], relative_path: str) -> str:
    source_name = Path(row.get("source_image", "")).name.casefold()
    box_match = re.search(r"(_box\d+)$", Path(relative_path).stem, flags=re.IGNORECASE)
    box = box_match.group(1).casefold() if box_match else ""
    return "|".join(
        (
            row.get("split", "").casefold(),
            row.get("leakage_group", "").casefold(),
            source_name,
            box,
        )
    )


def _read_selected_manifest(data_root: Path, manifest_path: Path) -> Tuple[List[Dict[str, str]], str]:
    selected: List[Dict[str, str]] = []
    canonical_rows: List[str] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "output_image", "class_id", "class_name"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(f"manifest missing fields: {sorted(required - set(reader.fieldnames or ()))}")
        for raw in reader:
            split = str(raw.get("split", "")).strip().casefold()
            if split not in ALLOWED_SPLITS:
                continue
            row = {str(key): str(value or "") for key, value in raw.items()}
            output = Path(row["output_image"]).expanduser().resolve()
            try:
                relative = output.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(f"selected manifest output is outside data root: {output}") from exc
            if not relative.parts or relative.parts[0].casefold() != split:
                raise ValueError(f"manifest split/path mismatch: {split} vs {relative}")
            row["split"] = split
            row["relative_path"] = _normalise_relative(relative)
            selected.append(row)
            canonical_rows.append(_canonical_json(row))
    selected.sort(key=lambda row: row["relative_path"].casefold())
    filtered_hash = _sha256_bytes(("\n".join(sorted(canonical_rows)) + "\n").encode("utf-8"))
    return selected, filtered_hash


def _read_relabel_history(operations_path: Optional[Path]) -> Tuple[Dict[str, List[Dict[str, str]]], str]:
    histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    selected_rows: List[str] = []
    if operations_path is None or not operations_path.exists():
        return histories, _sha256_bytes(b"")
    with operations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                operation = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid operations JSON at line {line_number}") from exc
            if operation.get("status") != "committed" or operation.get("action") != "change_classification":
                continue
            for change in operation.get("manifest_changed_rows", ()):
                before = {str(k): str(v or "") for k, v in dict(change.get("before", {})).items()}
                after = {str(k): str(v or "") for k, v in dict(change.get("after", {})).items()}
                split = after.get("split", "").casefold()
                if split not in ALLOWED_SPLITS:
                    continue
                relative_hint = Path(after.get("output_image", "")).name
                identity = _stable_sample_identity(after, relative_hint)
                history = {
                    "operation_id": str(operation.get("operation_id", "")),
                    "timestamp": str(operation.get("timestamp", "")),
                    "old_class_id": before.get("class_id", ""),
                    "old_class_name": before.get("class_name", ""),
                    "new_class_id": after.get("class_id", ""),
                    "new_class_name": after.get("class_name", ""),
                }
                histories[identity].append(history)
                selected_rows.append(_canonical_json({"identity": identity, **history}))
    for values in histories.values():
        values.sort(key=lambda item: (item["timestamp"], item["operation_id"]))
    filtered_hash = _sha256_bytes(("\n".join(sorted(selected_rows)) + "\n").encode("utf-8"))
    return histories, filtered_hash


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(math.pi * (2.0 * positions[None, :] + 1.0) * frequencies / (2.0 * size))
    matrix[0, :] *= math.sqrt(1.0 / size)
    matrix[1:, :] *= math.sqrt(2.0 / size)
    return matrix


_DCT32 = _dct_matrix(32)
_POPCOUNT8 = tuple(bin(value).count("1") for value in range(256))


def _hamming64(left: int, right: int) -> int:
    difference = int(left) ^ int(right)
    return sum(_POPCOUNT8[(difference >> shift) & 0xFF] for shift in range(0, 64, 8))


def _phash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    low = (_DCT32 @ pixels @ _DCT32.T)[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    value = 0
    for bit, coefficient in enumerate(low[1:], start=1):
        if float(coefficient) > threshold:
            value |= 1 << bit
    return value


class _HammingBKTree:
    def __init__(self) -> None:
        self._root: Optional[List[object]] = None

    def add(self, value: int, index: int) -> None:
        if self._root is None:
            self._root = [value, [index], {}]
            return
        node = self._root
        while True:
            distance = _hamming64(int(node[0]), value)
            if distance == 0:
                node[1].append(index)  # type: ignore[union-attr]
                return
            children: MutableMapping[int, List[object]] = node[2]  # type: ignore[assignment]
            if distance not in children:
                children[distance] = [value, [index], {}]
                return
            node = children[distance]

    def query(self, value: int, radius: int) -> Iterable[Tuple[int, int]]:
        if self._root is None:
            return ()
        found: List[Tuple[int, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            distance = _hamming64(int(node[0]), value)
            if distance <= radius:
                found.extend((int(index), distance) for index in node[1])  # type: ignore[union-attr]
            children: Mapping[int, List[object]] = node[2]  # type: ignore[assignment]
            keys = [key for key in children if distance - radius <= key <= distance + radius]
            stack.extend(children[key] for key in sorted(keys, reverse=True))
        return found


def _make_issue(
    *,
    issue_type: str,
    priority: str,
    scope: str,
    record_a: Mapping[str, object],
    record_b: Optional[Mapping[str, object]] = None,
    source_number_delta: object = "",
    phash_distance: object = "",
    details: object = None,
) -> Dict[str, object]:
    record_b = record_b or {}
    identity = {
        "issue_type": issue_type,
        "path_a": record_a.get("relative_path", ""),
        "path_b": record_b.get("relative_path", ""),
        "details": details or {},
    }
    return {
        "issue_id": _sha256_bytes(_canonical_json(identity).encode("utf-8"))[:20],
        "priority": priority,
        "scope": scope,
        "issue_type": issue_type,
        "split_a": record_a.get("split", ""),
        "path_a": record_a.get("relative_path", ""),
        "class_a": record_a.get("class_id", ""),
        "split_b": record_b.get("split", ""),
        "path_b": record_b.get("relative_path", ""),
        "class_b": record_b.get("class_id", ""),
        "source_prefix": record_a.get("source_prefix", ""),
        "source_number_delta": source_number_delta,
        "phash_distance": phash_distance,
        "details": _canonical_json(details or {}),
        "recommended_action": "human_review_only_no_auto_relabel_or_move",
    }


def _pair_scope(left: Mapping[str, object], right: Mapping[str, object]) -> str:
    if left["split"] == right["split"] == "train":
        return "train_only_review"
    return "validation_diagnostic_only"


def _assign_source_sessions(records: List[Dict[str, object]], adjacency: int) -> None:
    cohorts: Dict[Tuple[str, str, str], Dict[int, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        number = record.get("source_number")
        if number is None:
            continue
        cohort = (str(record["source"]), str(record["source_split"]), str(record["source_prefix"]))
        cohorts[cohort][int(number)].append(record)
    for cohort, by_number in sorted(cohorts.items()):
        component: List[int] = []
        components: List[List[int]] = []
        for number in sorted(by_number):
            if component and number - component[-1] > adjacency:
                components.append(component)
                component = []
            component.append(number)
        if component:
            components.append(component)
        for numbers in components:
            members = [record for number in numbers for record in by_number[number]]
            key = f"{cohort}|{numbers[0]}|{numbers[-1]}|gap={adjacency}"
            session_id = _sha256_bytes(key.encode("utf-8"))[:16]
            splits = ";".join(sorted({str(record["split"]) for record in members}))
            for record in members:
                record["source_session_candidate_id"] = session_id
                record["source_session_size"] = len(members)
                record["source_session_splits"] = splits


def _source_adjacency_issues(
    records: List[Dict[str, object]], adjacency: int
) -> Tuple[List[Dict[str, object]], Dict[str, set[str]], Dict[str, set[str]]]:
    cohorts: Dict[Tuple[str, str, str], Dict[int, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        number = record.get("source_number")
        if number is not None:
            cohorts[(str(record["source"]), str(record["source_split"]), str(record["source_prefix"]))][
                int(number)
            ].append(record)
    issues: List[Dict[str, object]] = []
    train_flags: Dict[str, set[str]] = defaultdict(set)
    diagnostic_flags: Dict[str, set[str]] = defaultdict(set)
    for by_number in cohorts.values():
        numbers = sorted(by_number)
        for position, left_number in enumerate(numbers):
            for right_number in numbers[position:]:
                delta = right_number - left_number
                if delta > adjacency:
                    break
                left_items = by_number[left_number]
                right_items = by_number[right_number]
                for left_index, left in enumerate(left_items):
                    start = left_index + 1 if delta == 0 else 0
                    for right in right_items[start:]:
                        label_conflict = left["class_id"] != right["class_id"]
                        cross_split = left["split"] != right["split"]
                        if not label_conflict and not cross_split:
                            continue
                        pair = sorted((left, right), key=lambda row: str(row["relative_path"]).casefold())
                        scope = _pair_scope(pair[0], pair[1])
                        issue_type = "source_adjacent_label_conflict" if label_conflict else "source_adjacent_cross_split"
                        if label_conflict and cross_split:
                            issue_type = "source_adjacent_cross_split_label_conflict"
                        priority = "high" if label_conflict or cross_split else "medium"
                        issues.append(
                            _make_issue(
                                issue_type=issue_type,
                                priority=priority,
                                scope=scope,
                                record_a=pair[0],
                                record_b=pair[1],
                                source_number_delta=delta,
                                details={"label_conflict": label_conflict, "cross_split": cross_split},
                            )
                        )
                        for record in pair:
                            relative = str(record["relative_path"])
                            if scope == "train_only_review" and label_conflict:
                                train_flags[relative].add("source_adjacent_cross_label")
                            else:
                                diagnostic_flags[relative].add("source_adjacency_cross_split")
    return issues, train_flags, diagnostic_flags


def _phash_issues(
    records: List[Dict[str, object]], radius: int
) -> Tuple[List[Dict[str, object]], Dict[str, set[str]], Dict[str, set[str]]]:
    tree = _HammingBKTree()
    issues: List[Dict[str, object]] = []
    train_flags: Dict[str, set[str]] = defaultdict(set)
    diagnostic_flags: Dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        value = int(record["phash_value"])
        for previous_index, distance in tree.query(value, radius):
            previous = records[previous_index]
            label_conflict = previous["class_id"] != record["class_id"]
            cross_split = previous["split"] != record["split"]
            if not label_conflict and not cross_split:
                continue
            pair = sorted((previous, record), key=lambda row: str(row["relative_path"]).casefold())
            scope = _pair_scope(pair[0], pair[1])
            issue_type = "near_crop_label_conflict" if label_conflict else "near_crop_cross_split"
            if label_conflict and cross_split:
                issue_type = "near_crop_cross_split_label_conflict"
            issues.append(
                _make_issue(
                    issue_type=issue_type,
                    priority="critical" if label_conflict and distance <= 3 else "high",
                    scope=scope,
                    record_a=pair[0],
                    record_b=pair[1],
                    phash_distance=distance,
                    details={"label_conflict": label_conflict, "cross_split": cross_split},
                )
            )
            for item in pair:
                relative = str(item["relative_path"])
                if scope == "train_only_review" and label_conflict:
                    train_flags[relative].add("near_crop_cross_label")
                else:
                    diagnostic_flags[relative].add("near_crop_cross_split")
        tree.add(value, index)
    return issues, train_flags, diagnostic_flags


def build_classf_integrity_review(
    data_root: Path,
    output_dir: Path,
    *,
    manifest_path: Optional[Path] = None,
    operations_path: Optional[Path] = None,
    phash_distance: int = 3,
    source_adjacency: int = 1,
) -> Dict[str, object]:
    data_root = data_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _ensure_output_outside_dataset(data_root, output_dir)
    if phash_distance < 0 or phash_distance > 16:
        raise ValueError("phash_distance must be in [0, 16]")
    if source_adjacency < 0 or source_adjacency > 20:
        raise ValueError("source_adjacency must be in [0, 20]")
    manifest_path = (manifest_path or data_root / "manifest.csv").expanduser().resolve()
    operations_path = (
        operations_path
        or data_root / ".cvat_nhai_classification_archive" / "operations.jsonl"
    ).expanduser().resolve()
    manifest_rows, manifest_train_val_sha256 = _read_selected_manifest(data_root, manifest_path)
    histories, operations_train_val_sha256 = _read_relabel_history(operations_path)
    by_path: Dict[str, Dict[str, str]] = {}
    for row in manifest_rows:
        key = _path_key(data_root / row["relative_path"])
        if key in by_path:
            raise ValueError(f"duplicate selected manifest output: {row['relative_path']}")
        by_path[key] = row

    records: List[Dict[str, object]] = []
    seen_paths: set[str] = set()
    for split in ALLOWED_SPLITS:
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing required split directory: {split_root}")
        paths = sorted(
            (path for path in split_root.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
            key=lambda path: _normalise_relative(path.relative_to(data_root)).casefold(),
        )
        for path in paths:
            key = _path_key(path)
            row = by_path.get(key)
            if row is None:
                raise ValueError(f"train/val image absent from manifest: {path}")
            seen_paths.add(key)
            relative = _normalise_relative(path.relative_to(data_root))
            if path.parent.name != row["class_name"]:
                raise ValueError(f"class folder/manifest mismatch: {relative}")
            prefix, number = _source_parts(row, relative)
            sha256 = _sha256_file(path)
            phash_value = _phash64(path)
            identity = _stable_sample_identity(row, relative)
            history = histories.get(identity, [])
            original_ids = sorted({item["old_class_id"] for item in history if item["old_class_id"]})
            records.append(
                {
                    "split": split,
                    "relative_path": relative,
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "source_image_name": Path(row.get("source_image", "")).name,
                    "source": row.get("source", ""),
                    "source_split": row.get("source_split", ""),
                    "leakage_group": row.get("leakage_group", ""),
                    "source_prefix": prefix,
                    "source_number": number,
                    "source_session_candidate_id": "",
                    "source_session_size": 1,
                    "source_session_splits": split,
                    "sha256": sha256,
                    "phash64": f"{phash_value:016x}",
                    "phash_value": phash_value,
                    "was_relabelled": bool(history),
                    "relabel_count": len(history),
                    "original_class_ids": ";".join(original_ids),
                    "history": history,
                    "train_review_flags": set(),
                    "diagnostic_flags": set(),
                }
            )
    missing = sorted(row["relative_path"] for key, row in by_path.items() if key not in seen_paths)
    if missing:
        raise ValueError(f"manifest has {len(missing)} missing train/val outputs; first={missing[0]}")
    records.sort(key=lambda row: str(row["relative_path"]).casefold())
    _assign_source_sessions(records, source_adjacency)

    review: List[Dict[str, object]] = []
    for record in records:
        history = list(record["history"])
        if not history:
            continue
        scope = "train_only_review" if record["split"] == "train" else "validation_diagnostic_only"
        history_class_ids = {
            int(item["old_class_id"]) for item in history if str(item.get("old_class_id", "")).strip()
        }
        history_class_ids.add(int(record["class_id"]))
        review.append(
            _make_issue(
                issue_type="relabel_history",
                priority="high" if 1 in history_class_ids else "medium",
                scope=scope,
                record_a=record,
                details={"history": history},
            )
        )
        flag_target = record["train_review_flags"] if scope == "train_only_review" else record["diagnostic_flags"]
        flag_target.add("relabel_history")

    source_issues, source_train_flags, source_diagnostic_flags = _source_adjacency_issues(
        records, source_adjacency
    )
    phash_issues, phash_train_flags, phash_diagnostic_flags = _phash_issues(records, phash_distance)
    review.extend(source_issues)
    review.extend(phash_issues)
    for record in records:
        relative = str(record["relative_path"])
        record["train_review_flags"].update(source_train_flags.get(relative, ()))
        record["train_review_flags"].update(phash_train_flags.get(relative, ()))
        record["diagnostic_flags"].update(source_diagnostic_flags.get(relative, ()))
        record["diagnostic_flags"].update(phash_diagnostic_flags.get(relative, ()))
        record["train_review_flags"] = ";".join(sorted(record["train_review_flags"]))
        record["diagnostic_flags"] = ";".join(sorted(record["diagnostic_flags"]))
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    review.sort(
        key=lambda row: (
            priority_order.get(str(row["priority"]), 9),
            str(row["scope"]),
            str(row["issue_type"]),
            str(row["path_a"]).casefold(),
            str(row["path_b"]).casefold(),
            str(row["issue_id"]),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    derived_manifest_path = output_dir / "classf_train_val_integrity_manifest.csv"
    review_path = output_dir / "classf_human_review_queue.csv"
    _write_csv(derived_manifest_path, MANIFEST_FIELDS, records)
    _write_csv(review_path, REVIEW_FIELDS, review)
    image_attestation_rows = [
        f"{row['split']}\t{row['relative_path']}\t{row['class_id']}\t{row['sha256']}" for row in records
    ]
    issue_counts = Counter(str(row["issue_type"]) for row in review)
    scope_counts = Counter(str(row["scope"]) for row in review)
    sample_split_class_counts = Counter(f"{row['split']}:{row['class_id']}" for row in records)
    train_flagged_class_counts = Counter(
        str(row["class_id"]) for row in records if row["split"] == "train" and row["train_review_flags"]
    )
    relabel_transition_counts: Counter = Counter()
    for row in records:
        for history in row["history"]:
            relabel_transition_counts[f"{history['old_class_id']}->{history['new_class_id']}"] += 1
    issue_class_pair_counts = Counter(
        (
            f"{row['issue_type']}:"
            + "-".join(sorted((str(row["class_a"]), str(row["class_b"]))))
            if row["class_b"] != ""
            else f"{row['issue_type']}:{row['class_a']}"
        )
        for row in review
    )
    summary: Dict[str, object] = {
        "schema_version": "TRKH_CLASSF_INTEGRITY_REVIEW_V1",
        "policy": {
            "canonical_splits_read": list(ALLOWED_SPLITS),
            "canonical_test_accessed": False,
            "raw_dataset_modified": False,
            "model_predictions_used": False,
            "automatic_relabel_or_move": False,
            "training_flags_use_validation_evidence": False,
            "source_session_semantics": "candidate only: same provenance prefix joined by numeric adjacency",
        },
        "data_root": str(data_root),
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_train_val_sha256": manifest_train_val_sha256,
            "operations": str(operations_path) if operations_path.exists() else None,
            "operations_train_val_sha256": operations_train_val_sha256,
        },
        "parameters": {
            "phash": "64-bit DCT luminance hash",
            "phash_distance": phash_distance,
            "source_adjacency": source_adjacency,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
        "counts": {
            "samples": len(records),
            "samples_by_split": dict(sorted(Counter(str(row["split"]) for row in records).items())),
            "samples_by_split_class": dict(sorted(sample_split_class_counts.items())),
            "relabelled_samples": sum(bool(row["was_relabelled"]) for row in records),
            "relabel_transitions": dict(sorted(relabel_transition_counts.items())),
            "train_flagged_samples": sum(bool(row["train_review_flags"]) for row in records),
            "train_flagged_samples_by_class": dict(sorted(train_flagged_class_counts.items())),
            "diagnostic_flagged_samples": sum(bool(row["diagnostic_flags"]) for row in records),
            "review_issues": len(review),
            "issues_by_type": dict(sorted(issue_counts.items())),
            "issues_by_scope": dict(sorted(scope_counts.items())),
            "issues_by_type_class_pair": dict(sorted(issue_class_pair_counts.items())),
        },
        "attestation": {
            "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
            "train_val_image_tree_sha256": _sha256_bytes(("\n".join(image_attestation_rows) + "\n").encode("utf-8")),
            "derived_manifest_sha256": _sha256_file(derived_manifest_path),
            "review_queue_sha256": _sha256_file(review_path),
        },
    }
    summary_path = output_dir / "classf_integrity_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an immutable train/val-only class_f integrity manifest and human review queue."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--operations", type=Path, default=None)
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=3,
        help="High-confidence near-duplicate radius; use 8 only for a broad diagnostic queue.",
    )
    parser.add_argument("--source-adjacency", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_classf_integrity_review(
        args.data_root,
        args.output_dir,
        manifest_path=args.manifest,
        operations_path=args.operations,
        phash_distance=args.phash_distance,
        source_adjacency=args.source_adjacency,
    )
    print(json.dumps({"counts": summary["counts"], "attestation": summary["attestation"]}, indent=2))


if __name__ == "__main__":
    main()

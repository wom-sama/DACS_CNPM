from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _split_relative_key(path_text: str, *, split: str) -> str:
    parts = [
        part
        for part in str(path_text or "").strip().replace("\\", "/").split("/")
        if part and part != "."
    ]
    split_name = str(split).strip().casefold()
    positions = [
        index for index, part in enumerate(parts) if part.casefold() == split_name
    ]
    if positions:
        parts = parts[positions[-1] + 1 :]
    if len(parts) < 2:
        raise ValueError(
            "Manifest output_image must map to <class>/<image> under "
            f"{split_name}: {path_text!r}."
        )
    return "/".join(part.casefold() for part in parts)


def load_tempered_leakage_group_ids(
    manifest_path: Path,
    *,
    sample_paths: Sequence[Path],
    labels: Sequence[int],
    train_root: Path,
    class_names: Sequence[str],
) -> Tuple[List[str], Dict[str, object]]:
    """Map every final train row to one canonical leakage group, fail closed."""

    manifest_path = Path(manifest_path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing leakage-group manifest: {manifest_path}")
    if len(sample_paths) != len(labels):
        raise ValueError(
            "Leakage-group mapping needs one label per sample path: "
            f"paths={len(sample_paths)}, labels={len(labels)}."
        )

    required = {"split", "output_image", "class_id", "leakage_group"}
    train_entries: Dict[str, Tuple[str, int]] = {}
    group_splits: Dict[str, set[str]] = {}
    train_group_labels: Dict[str, set[int]] = {}
    manifest_rows = train_rows = skipped_non_train = 0

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {
            str(field).strip().casefold() for field in (reader.fieldnames or [])
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Leakage-group manifest is missing columns: {missing}.")

        for row_number, raw_row in enumerate(reader, start=2):
            manifest_rows += 1
            row = {
                str(key).strip().casefold(): str(value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            split = row.get("split", "").casefold()
            if split not in {"train", "val", "test"}:
                raise ValueError(
                    f"Invalid manifest split at row {row_number}: {split!r}."
                )
            output_image = row.get("output_image", "")
            group_id = row.get("leakage_group", "")
            if not output_image or not group_id:
                raise ValueError(
                    f"Empty output_image/leakage_group at row {row_number}."
                )
            try:
                class_id = int(row.get("class_id", ""))
            except ValueError as error:
                raise ValueError(
                    f"Invalid class_id at row {row_number}: "
                    f"{row.get('class_id', '')!r}."
                ) from error
            if not 0 <= class_id < len(class_names):
                raise ValueError(
                    f"class_id outside [0,{len(class_names)}) at row {row_number}."
                )

            group_splits.setdefault(group_id, set()).add(split)
            if split != "train":
                skipped_non_train += 1
                continue

            train_rows += 1
            key = _split_relative_key(output_image, split="train")
            if key.split("/", 1)[0] != str(class_names[class_id]).casefold():
                raise ValueError(
                    "Manifest class_id does not match output_image at "
                    f"row {row_number}: class_id={class_id}, key={key!r}."
                )
            if key in train_entries:
                raise ValueError(
                    f"Duplicate train output_image at row {row_number}: {key!r}."
                )
            train_entries[key] = (group_id, class_id)
            train_group_labels.setdefault(group_id, set()).add(class_id)

    cross_split_groups = sorted(
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    )
    if cross_split_groups:
        raise ValueError(
            "Leakage groups cross declared splits: "
            f"count={len(cross_split_groups)}, preview={cross_split_groups[:10]}."
        )
    if not train_entries:
        raise ValueError(f"Manifest contains no train rows: {manifest_path}")

    train_root = Path(train_root).resolve()
    observed_keys: set[str] = set()
    group_ids: List[str] = []
    member_counts: Counter[Tuple[int, str]] = Counter()
    for sample_index, (sample_path, label) in enumerate(zip(sample_paths, labels)):
        resolved_path = Path(sample_path).resolve()
        try:
            key = resolved_path.relative_to(train_root).as_posix().casefold()
        except ValueError as error:
            raise ValueError(
                f"Train sample is outside train root: {resolved_path}."
            ) from error
        if key in observed_keys:
            raise ValueError(f"Duplicate final train sample path: {key!r}.")
        observed_keys.add(key)

        entry = train_entries.get(key)
        if entry is None:
            raise ValueError(f"Train sample is missing from manifest: {key!r}.")
        group_id, manifest_class_id = entry
        if int(label) != manifest_class_id:
            raise ValueError(
                "Train label does not match manifest: "
                f"index={sample_index}, key={key!r}, dataset={label}, "
                f"manifest={manifest_class_id}."
            )
        if key.split("/", 1)[0] != str(class_names[int(label)]).casefold():
            raise ValueError(
                f"Train folder does not match dataset label: {key!r}."
            )
        group_ids.append(group_id)
        member_counts[(int(label), group_id)] += 1

    extra_keys = sorted(set(train_entries) - observed_keys)
    if extra_keys:
        raise ValueError(
            "Manifest train rows are absent from final dataset: "
            f"count={len(extra_keys)}, preview={extra_keys[:10]}."
        )

    group_counts = [
        len({group for (label, group) in member_counts if label == class_index})
        for class_index in range(len(class_names))
    ]
    max_group_sizes = [
        max(
            (
                count
                for (label, _group), count in member_counts.items()
                if label == class_index
            ),
            default=0,
        )
        for class_index in range(len(class_names))
    ]
    cardinalities = Counter(len(values) for values in train_group_labels.values())
    summary: Dict[str, object] = {
        "enabled": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_split": "train_only",
        "mapping_key": "casefolded_path_relative_to_train_root",
        "manifest_rows": manifest_rows,
        "train_rows": train_rows,
        "skipped_non_train_rows": skipped_non_train,
        "mapped_samples": len(group_ids),
        "unique_train_groups": len(train_group_labels),
        "group_counts_by_class": group_counts,
        "max_group_sizes_by_class": max_group_sizes,
        "mixed_label_group_count": sum(
            1 for values in train_group_labels.values() if len(values) > 1
        ),
        "group_label_cardinality_histogram": {
            str(cardinality): int(count)
            for cardinality, count in sorted(cardinalities.items())
        },
        "cross_split_group_count": 0,
        "leakage_guard": "exact train coverage; cross-split groups rejected",
    }
    return group_ids, summary

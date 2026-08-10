from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from trkh.tools.build_yolo_oof_folds import YoloImageSample, _scan_yolo_samples


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one source-disjoint train/holdout YOLO view from an "
            "existing per-object fold declaration. Raw files are hardlinked only."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--condition", type=str, default="clean")
    parser.add_argument("--expected-data-sha256", type=str, default="")
    parser.add_argument("--expected-declaration-sha256", type=str, default="")
    parser.add_argument("--expected-fit-index-sha256", type=str, default="")
    parser.add_argument("--expected-holdout-index-sha256", type=str, default="")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_index_sha256(indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def _verify_optional_hash(path: Path, expected: str, *, label: str) -> str:
    observed = _sha256(path)
    normalized = str(expected).strip().casefold()
    if normalized and observed.casefold() != normalized:
        raise ValueError(
            f"{label} SHA-256 differs: observed={observed}, expected={normalized}."
        )
    return observed


def _read_declaration(
    path: Path,
    *,
    condition: str,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "fold",
            "target_index",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fold declaration is missing columns: {sorted(missing)}")
        for raw in reader:
            if str(raw["condition"]).strip() != str(condition):
                continue
            source_stem = str(raw["source_stem"]).strip().casefold()
            if not source_stem:
                raise ValueError("Fold declaration contains an empty source_stem.")
            selected.append(
                {
                    "sample_index": int(raw["sample_index"]),
                    "source_stem": source_stem,
                    "image_path": str(Path(str(raw["image_path"])).resolve()),
                    "fold": int(raw["fold"]),
                    "target": int(raw["target_index"]),
                }
            )
    selected.sort(key=lambda row: int(row["sample_index"]))
    observed_indices = [int(row["sample_index"]) for row in selected]
    if observed_indices != list(range(len(selected))):
        raise ValueError(
            "Selected declaration rows must have ordered unique sample indices 0..N-1."
        )
    if not selected:
        raise ValueError(f"No declaration rows found for condition={condition!r}.")
    return selected


def _validate_source_alignment(
    rows: Sequence[Mapping[str, object]],
    samples: Sequence[YoloImageSample],
) -> Dict[str, Dict[str, object]]:
    rows_by_source: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_source[str(row["source_stem"])].append(row)

    samples_by_source: Dict[str, YoloImageSample] = {}
    for sample in samples:
        key = sample.stem.casefold()
        if key in samples_by_source:
            raise ValueError(f"Duplicate case-insensitive YOLO source stem: {sample.stem}")
        samples_by_source[key] = sample
    if set(rows_by_source) != set(samples_by_source):
        missing_declaration = sorted(set(samples_by_source).difference(rows_by_source))
        missing_dataset = sorted(set(rows_by_source).difference(samples_by_source))
        raise ValueError(
            "Declaration and YOLO source sets differ: "
            f"missing_declaration={missing_declaration[:5]}, "
            f"missing_dataset={missing_dataset[:5]}."
        )

    sources: Dict[str, Dict[str, object]] = {}
    for source_stem, source_rows in sorted(rows_by_source.items()):
        source_rows = sorted(source_rows, key=lambda row: int(row["sample_index"]))
        sample = samples_by_source[source_stem]
        folds = {int(row["fold"]) for row in source_rows}
        if len(folds) != 1:
            raise ValueError(f"Source {source_stem} crosses declaration folds: {folds}")
        targets = tuple(int(row["target"]) for row in source_rows)
        if targets != tuple(int(value) for value in sample.labels):
            raise ValueError(
                f"YOLO object order/labels differ for {source_stem}: "
                f"declaration={targets}, labels={sample.labels}."
            )
        expected_image = str(sample.image_path.resolve()).casefold()
        for row in source_rows:
            if str(row["image_path"]).casefold() != expected_image:
                raise ValueError(
                    f"Declaration image path differs for source {source_stem}: "
                    f"{row['image_path']} != {sample.image_path}."
                )
        sources[source_stem] = {
            "sample": sample,
            "fold": next(iter(folds)),
            "row_count": len(source_rows),
            "sample_indices": [int(row["sample_index"]) for row in source_rows],
            "targets": list(targets),
        }
    return sources


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {path}")
        resolved = path.resolve()
        cwd = Path.cwd().resolve()
        runs_root = (cwd / "runs").resolve()
        if runs_root not in resolved.parents or resolved == runs_root:
            raise ValueError(
                f"Refusing to overwrite output outside a child of {runs_root}: {resolved}"
            )
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=False)


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _write_yaml(
    output_root: Path,
    *,
    class_names: Sequence[str],
    declaration: Path,
    fold: int,
) -> Path:
    path = output_root / "data.yaml"
    payload = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": list(class_names),
        "format": "yolo",
        "class_name_mode": "raw",
        "note": (
            "Train-only declared fold view. test mirrors holdout only for loader "
            "compatibility and must not be evaluated."
        ),
        "fold_declaration": str(declaration.resolve()),
        "held_out_fold": int(fold),
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def build_declared_fold(
    *,
    data_yaml: Path,
    declaration: Path,
    output_root: Path,
    fold: int = 0,
    condition: str = "clean",
    expected_data_sha256: str = "",
    expected_declaration_sha256: str = "",
    expected_fit_index_sha256: str = "",
    expected_holdout_index_sha256: str = "",
    overwrite: bool = False,
) -> Dict[str, object]:
    data_yaml = data_yaml.resolve()
    declaration = declaration.resolve()
    output_root = output_root.resolve()
    data_sha256 = _verify_optional_hash(
        data_yaml,
        expected_data_sha256,
        label="data YAML",
    )
    declaration_sha256 = _verify_optional_hash(
        declaration,
        expected_declaration_sha256,
        label="fold declaration",
    )
    rows = _read_declaration(declaration, condition=condition)
    samples, class_names, images_dir, labels_dir = _scan_yolo_samples(
        data_yaml,
        source_split="train",
        max_images=0,
    )
    sources = _validate_source_alignment(rows, samples)
    fold = int(fold)
    fit_rows = [row for row in rows if int(row["fold"]) != fold]
    holdout_rows = [row for row in rows if int(row["fold"]) == fold]
    if not fit_rows or not holdout_rows:
        raise ValueError(f"Declared fold {fold} does not create nonempty fit/holdout rows.")
    fit_indices = [int(row["sample_index"]) for row in fit_rows]
    holdout_indices = [int(row["sample_index"]) for row in holdout_rows]
    fit_index_sha256 = _ordered_index_sha256(fit_indices)
    holdout_index_sha256 = _ordered_index_sha256(holdout_indices)
    if expected_fit_index_sha256 and (
        fit_index_sha256.casefold() != expected_fit_index_sha256.strip().casefold()
    ):
        raise ValueError("Fit sample-index SHA-256 differs from the locked declaration.")
    if expected_holdout_index_sha256 and (
        holdout_index_sha256.casefold()
        != expected_holdout_index_sha256.strip().casefold()
    ):
        raise ValueError("Holdout sample-index SHA-256 differs from the locked declaration.")

    fit_sources = {
        str(row["source_stem"])
        for row in fit_rows
    }
    holdout_sources = {
        str(row["source_stem"])
        for row in holdout_rows
    }
    overlap = fit_sources.intersection(holdout_sources)
    if overlap:
        raise ValueError(f"Declared fit/holdout source overlap: {sorted(overlap)[:5]}")

    _prepare_output(output_root, overwrite=overwrite)
    data_output = _write_yaml(
        output_root,
        class_names=class_names,
        declaration=declaration,
        fold=fold,
    )
    hardlink_count = 0
    for source_stem, source in sorted(sources.items()):
        sample = source["sample"]
        if not isinstance(sample, YoloImageSample):
            raise TypeError("Internal declared-fold sample type differs.")
        split = "val" if source_stem in holdout_sources else "train"
        for split_name in (split, "test") if split == "val" else (split,):
            _hardlink(
                sample.image_path,
                output_root / "images" / split_name / sample.image_path.name,
            )
            _hardlink(
                sample.label_path,
                output_root / "labels" / split_name / sample.label_path.name,
            )
            hardlink_count += 2

    fit_class_counts = [
        sum(int(row["target"]) == class_index for row in fit_rows)
        for class_index in range(len(class_names))
    ]
    holdout_class_counts = [
        sum(int(row["target"]) == class_index for row in holdout_rows)
        for class_index in range(len(class_names))
    ]
    source_fold_counts = Counter(int(value["fold"]) for value in sources.values())
    summary: Dict[str, object] = {
        "method": "declared_train_only_yolo_fold",
        "data_yaml": str(data_yaml),
        "data_sha256": data_sha256,
        "declaration": str(declaration),
        "declaration_sha256": declaration_sha256,
        "condition": str(condition),
        "held_out_fold": fold,
        "output_root": str(output_root),
        "output_data_yaml": str(data_output.resolve()),
        "source_images_dir": str(images_dir.resolve()),
        "source_labels_dir": str(labels_dir.resolve()),
        "class_names": list(class_names),
        "total_rows": len(rows),
        "total_sources": len(sources),
        "fit_rows": len(fit_rows),
        "holdout_rows": len(holdout_rows),
        "fit_sources": len(fit_sources),
        "holdout_sources": len(holdout_sources),
        "source_overlap": len(overlap),
        "fit_class_counts": fit_class_counts,
        "holdout_class_counts": holdout_class_counts,
        "source_fold_counts": {
            str(key): int(value) for key, value in sorted(source_fold_counts.items())
        },
        "fit_index_sha256": fit_index_sha256,
        "holdout_index_sha256": holdout_index_sha256,
        "hardlink_count": hardlink_count,
        "test_mirrors_holdout": True,
        "raw_data_modified": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.resolve())
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_declared_fold(
        data_yaml=args.data,
        declaration=args.declaration,
        output_root=args.output_root,
        fold=args.fold,
        condition=args.condition,
        expected_data_sha256=args.expected_data_sha256,
        expected_declaration_sha256=args.expected_declaration_sha256,
        expected_fit_index_sha256=args.expected_fit_index_sha256,
        expected_holdout_index_sha256=args.expected_holdout_index_sha256,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

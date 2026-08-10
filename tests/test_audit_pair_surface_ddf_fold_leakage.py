from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

from trkh.tools import audit_pair_surface_ddf_fold_leakage as audit


def _cohort_arrays() -> Dict[str, np.ndarray]:
    rows = audit.EXPECTED_ROWS
    source_stems = np.asarray(
        [f"isolated_{index}" for index in range(rows)],
        dtype="<U32",
    )
    folds = np.arange(rows, dtype=np.int64) % 5
    targets = np.arange(rows, dtype=np.int64) % 5

    source_stems[:6] = [
        "Image_10",
        "image_10",
        "Image_11",
        "Image_13",
        "Image_20",
        "Image_22",
    ]
    folds[:6] = [0, 1, 2, 0, 0, 1]
    targets[:6] = [1, 1, 0, 1, 1, 1]
    return {
        "sample_indices": np.arange(rows, dtype=np.int64),
        "targets": targets,
        "folds": folds,
        "source_stems": source_stems,
    }


def _unique_stems(source_stems: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for source_stem in source_stems:
        result.setdefault(str(source_stem).casefold(), str(source_stem))
    return result


def _leakage_group(stem_key: str) -> str:
    if stem_key in {"image_10", "image_11", "image_13"}:
        return "group_a"
    if stem_key in {"image_20", "image_22"}:
        return "group_b"
    return f"group_{stem_key}"


def _write_inputs(
    tmp_path: Path,
    *,
    omitted_stem: Optional[str] = None,
    non_train_stem: Optional[str] = None,
) -> Tuple[Path, Path]:
    arrays = _cohort_arrays()
    cohort_path = tmp_path / "cohort_arrays.npz"
    np.savez(cohort_path, **arrays)

    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.EXPECTED_MANIFEST_FIELDS)
        writer.writeheader()
        for stem_key, display_stem in _unique_stems(
            arrays["source_stems"].astype(str).tolist()
        ).items():
            if stem_key == omitted_stem:
                continue
            # Upper-case output stems prove that the join is casefold-exact.
            output_stem = display_stem.upper()
            split = "val" if stem_key == non_train_stem else "train"
            writer.writerow(
                {
                    "split": split,
                    "source_split": "source_train",
                    "leakage_group": _leakage_group(stem_key),
                    "source_image": f"D:\\source\\{output_stem}.jpg",
                    "output_image": (
                        f"D:\\dataset\\images\\{split}\\{output_stem}.JPG"
                    ),
                    "output_label": (
                        f"D:\\dataset\\labels\\{split}\\{output_stem}.txt"
                    ),
                }
            )
    return cohort_path, manifest_path


def test_metadata_audit_reports_all_locked_relations_without_assigning_folds(
    tmp_path: Path,
) -> None:
    cohort_path, manifest_path = _write_inputs(tmp_path)
    output_path = tmp_path / "fold_leakage.json"

    audit.main(
        [
            "--cohort-arrays",
            str(cohort_path),
            "--yolo-manifest",
            str(manifest_path),
            "--output-json",
            str(output_path),
        ]
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["cohort"]["rows"] == 763
    assert report["cohort"]["unique_casefold_stems"] == 762
    assert report["manifest_mapping"] == {
        "all_mapped_rows_are_train": True,
        "join": "casefold_exact_output_image_stem_to_source_stem",
        "manifest_rows_scanned": 762,
        "mapped_rows": 763,
        "mapped_unique_stems": 762,
        "unmapped_rows": 0,
        "unmapped_unique_stems": 0,
    }
    assert report["access_contract"] == {
        "candidate_score_arrays_read": 0,
        "declared_metadata_files_read": 2,
        "fold_assignment_performed": False,
        "image_or_label_paths_opened": 0,
        "pixel_files_opened": 0,
    }

    exact = report["old_fold_exact_stem_overlap"]
    assert exact["group_count"] == 762
    assert exact["cross_fold_group_count"] == 1
    assert exact["covered_row_count"] == 2
    assert exact["cross_fold_row_pairs"] == {
        "cross_label": 0,
        "same_label": 1,
        "total": 1,
    }
    assert exact["fold_span_histogram"] == {"1": 761, "2": 1}

    leakage = report["manifest_leakage_group_cross_fold"]
    assert leakage["group_count"] == 759
    assert leakage["cross_fold_group_count"] == 2
    assert leakage["covered_row_count"] == 6
    assert leakage["cross_fold_row_pairs"] == {
        "cross_label": 3,
        "same_label": 3,
        "total": 6,
    }
    assert leakage["fold_span_histogram"] == {"1": 757, "2": 1, "3": 1}

    numeric = report["numeric_image_n_cross_fold"]
    assert numeric["parsed_row_count"] == 6
    assert numeric["parsed_unique_stem_count"] == 5
    assert numeric["windows"]["1"]["cross_fold_unique_stem_pair_count"] == 1
    assert numeric["windows"]["1"]["cross_fold_row_pair_count"] == 2
    assert numeric["windows"]["3"]["cross_fold_unique_stem_pair_count"] == 4
    assert numeric["windows"]["3"]["cross_fold_row_pair_count"] == 5
    assert numeric["windows"]["3"]["cross_fold_row_pairs"] == {
        "cross_label": 3,
        "same_label": 2,
        "total": 5,
    }

    components = report["transitive_union_window_le_3"]
    assert components["fold_assignment_performed"] is False
    assert components["component_count"] == 759
    assert components["row_size"] == {"maximum": 4, "p95_nearest_rank": 1}
    assert components["unique_stem_size"] == {
        "maximum": 3,
        "p95_nearest_rank": 1,
    }
    assert components["cross_fold_component_count"] == 2
    assert components["cross_fold_covered_row_count"] == 6
    assert components["mixed_class_component_count"] == 1


def test_cohort_loader_never_reads_unrequested_archive_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _cohort_arrays()
    accessed = []

    class GuardedArchive:
        files = [
            *audit.REQUIRED_COHORT_ARRAYS,
            "keeper_probabilities",
            "model_srgb_uint8",
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            accessed.append(key)
            if key not in audit.REQUIRED_COHORT_ARRAYS:
                raise AssertionError(f"Forbidden archive member read: {key}")
            return arrays[key]

    monkeypatch.setattr(audit.np, "load", lambda *_args, **_kwargs: GuardedArchive())
    rows = audit._load_cohort(tmp_path / "guarded.npz")

    assert len(rows) == 763
    assert accessed == list(audit.REQUIRED_COHORT_ARRAYS)


def test_audit_fails_if_casefold_manifest_mapping_does_not_cover_all_rows(
    tmp_path: Path,
) -> None:
    cohort_path, manifest_path = _write_inputs(
        tmp_path,
        omitted_stem="image_10",
    )
    with pytest.raises(ValueError, match="must cover all 763 cohort rows"):
        audit.build_report(cohort_path, manifest_path)


def test_audit_fails_if_any_mapped_manifest_row_is_not_train(
    tmp_path: Path,
) -> None:
    cohort_path, manifest_path = _write_inputs(
        tmp_path,
        non_train_stem="image_11",
    )
    with pytest.raises(ValueError, match="Mapped manifest split must be train"):
        audit.build_report(cohort_path, manifest_path)


def test_audit_fails_closed_on_unexpected_cohort_or_manifest_format(
    tmp_path: Path,
) -> None:
    arrays = _cohort_arrays()
    arrays["folds"] = arrays["folds"].astype(np.int32)
    bad_cohort = tmp_path / "bad_cohort.npz"
    np.savez(bad_cohort, **arrays)
    _, manifest_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="Unexpected cohort folds dtype"):
        audit.build_report(bad_cohort, manifest_path)

    good_cohort = tmp_path / "cohort_arrays.npz"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("split,leakage_group,output_image\n")
    with pytest.raises(ValueError, match="Unexpected YOLO manifest fields"):
        audit.build_report(good_cohort, manifest_path)


def test_parse_args_requires_the_three_declared_metadata_paths() -> None:
    args = audit.parse_args(
        [
            "--cohort-arrays",
            "cohort_arrays.npz",
            "--yolo-manifest",
            "manifest.csv",
            "--output-json",
            "report.json",
        ]
    )
    assert args.cohort_arrays == Path("cohort_arrays.npz")
    assert args.yolo_manifest == Path("manifest.csv")
    assert args.output_json == Path("report.json")

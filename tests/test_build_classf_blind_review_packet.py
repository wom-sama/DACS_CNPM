from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import pytest
from PIL import Image

from trkh.tools.build_classf_blind_review_packet import build_packet


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 40), rgb).save(path, format="JPEG", quality=92)


def _fixture(tmp_path: Path) -> Dict[str, Path]:
    root = tmp_path / "dataset"
    for class_id in range(5):
        (root / "train" / f"C{class_id}").mkdir(parents=True, exist_ok=True)
    images = {
        "target": root / "train" / "C1" / "secret_target.jpg",
        "near": root / "train" / "C2" / "secret_near.jpg",
        "adjacent": root / "train" / "C0" / "secret_adjacent.jpg",
        "pair_a": root / "train" / "C3" / "secret_pair_a.jpg",
        "pair_b": root / "train" / "C4" / "secret_pair_b.jpg",
    }
    for index, path in enumerate(images.values(), start=1):
        _make_image(path, (20 * index, 30 * index, 40 * index))

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "format: classification_folder\npath: .\ntrain: train\nval: val\ntest: test\nnc: 5\n"
        "class_name_mode: raw\nnames:\n  0: C0\n  1: C1\n  2: C2\n  3: C3\n  4: C4\n",
        encoding="utf-8",
    )
    priority = tmp_path / "priority"
    near_json = json.dumps([{"relative_path": "train/C2/secret_near.jpg", "class_id": 2}])
    adjacent_json = json.dumps([{"relative_path": "train/C0/secret_adjacent.jpg", "class_id": 0}])
    _write_csv(
        priority / "priority_p1_class1_intersection.csv",
        ("review_id", "relative_path", "class_id", "sha256", "near_partners_json", "adjacent_partners_json"),
        [
            {
                "review_id": "P1_source_secret",
                "relative_path": "train/C1/secret_target.jpg",
                "class_id": 1,
                "sha256": _sha256(images["target"]),
                "near_partners_json": near_json,
                "adjacent_partners_json": adjacent_json,
            }
        ],
    )
    _write_csv(
        priority / "priority_p2_rgb_near_identical_pairs.csv",
        ("review_id", "path_a", "class_a", "sha256_a", "path_b", "class_b", "sha256_b"),
        [
            {
                "review_id": "P2_source_secret",
                "path_a": "train/C3/secret_pair_a.jpg",
                "class_a": 3,
                "sha256_a": _sha256(images["pair_a"]),
                "path_b": "train/C4/secret_pair_b.jpg",
                "class_b": 4,
                "sha256_b": _sha256(images["pair_b"]),
            }
        ],
    )
    integrity_manifest = tmp_path / "integrity" / "manifest.csv"
    image_classes = {"target": 1, "near": 2, "adjacent": 0, "pair_a": 3, "pair_b": 4}
    _write_csv(
        integrity_manifest,
        ("split", "relative_path", "class_id", "sha256"),
        [
            {
                "split": "train",
                "relative_path": path.relative_to(root).as_posix(),
                "class_id": image_classes[name],
                "sha256": _sha256(path),
            }
            for name, path in images.items()
        ],
    )
    p1_path = priority / "priority_p1_class1_intersection.csv"
    p2_path = priority / "priority_p2_rgb_near_identical_pairs.csv"
    review_summary = priority / "review_summary.json"
    review_summary.write_text(
        json.dumps(
            {
                "scope": {
                    "dataset_split_materialized": ["train"],
                    "validation_images_opened": False,
                    "test_images_opened": False,
                    "model_inference_run": False,
                    "raw_dataset_modified": False,
                    "automatic_relabel": False,
                },
                "outputs": {
                    p1_path.name: {"sha256": _sha256(p1_path), "rows": 1},
                    p2_path.name: {"sha256": _sha256(p2_path), "rows": 1},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "data_yaml": data_yaml,
        "priority": priority,
        "integrity_manifest": integrity_manifest,
        "review_summary": review_summary,
    }


def _snapshot(root: Path) -> Dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_packet_is_deterministic_complete_and_reviewer_blind(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    expected_inputs = {
        "data_yaml_sha256": _sha256(fixture["data_yaml"]),
        "integrity_manifest_sha256": _sha256(fixture["integrity_manifest"]),
        "p1_csv_sha256": _sha256(fixture["priority"] / "priority_p1_class1_intersection.csv"),
        "p2_csv_sha256": _sha256(fixture["priority"] / "priority_p2_rgb_near_identical_pairs.csv"),
        "review_summary_sha256": _sha256(fixture["review_summary"]),
    }
    kwargs = {
        "seed": 20260805,
        "integrity_manifest": fixture["integrity_manifest"],
        "review_summary": fixture["review_summary"],
        "expected_input_sha256": expected_inputs,
    }
    summary_a = build_packet(fixture["data_yaml"], fixture["priority"], out_a, **kwargs)
    summary_b = build_packet(fixture["data_yaml"], fixture["priority"], out_b, **kwargs)

    assert summary_a == summary_b
    assert _snapshot(out_a) == _snapshot(out_b)
    assert summary_a["scope"] == {
        "split": "train",
        "validation_opened": False,
        "test_opened": False,
        "model_inference_run": False,
        "raw_dataset_modified": False,
        "automatic_relabel": False,
    }
    assert summary_a["counts"] == {
        "source_groups": 2,
        "source_p1_groups": 1,
        "source_p2_groups": 1,
        "reviewer_display_groups_each": 4,
        "reviewer_p1_singletons_each": 3,
        "reviewer_p2_pairs_each": 1,
        "distinct_train_images": 5,
        "reviewer_visible_rows_each": 5,
    }
    assert summary_a["input_hash_contract_verified"] is True

    visible_paths = [
        path
        for reviewer in (1, 2)
        for path in (out_a / f"reviewer_{reviewer}").iterdir()
        if path.suffix.lower() in {".html", ".csv", ".txt"}
    ]
    visible_text = "\n".join(path.read_text(encoding="utf-8") for path in visible_paths)
    for forbidden in (
        "P1_source_secret",
        "P2_source_secret",
        "train/",
        "secret_target.jpg",
        "secret_near.jpg",
        "secret_adjacent.jpg",
        "secret_pair_a.jpg",
        "secret_pair_b.jpg",
        "current_class_id",
        "source_review_id",
        "b9_oof",
        "was_relabelled",
    ):
        assert forbidden not in visible_text

    sealed = json.loads((out_a / "ADMIN_DO_NOT_SHARE" / "sealed_mapping.json").read_text(encoding="utf-8"))
    assert len(sealed["mapping"]) == 10
    assert {row["relative_path"] for row in sealed["mapping"]} == {
        "train/C1/secret_target.jpg",
        "train/C2/secret_near.jpg",
        "train/C0/secret_adjacent.jpg",
        "train/C3/secret_pair_a.jpg",
        "train/C4/secret_pair_b.jpg",
    }
    first_group_sequence = {}
    for reviewer in (1, 2):
        sequence = []
        seen = set()
        for row in sealed["mapping"]:
            if row["reviewer"] == reviewer and row["blind_group_id"] not in seen:
                seen.add(row["blind_group_id"])
                sequence.append((row["cohort"], row["relative_path"], tuple(row["source_review_ids"])))
        first_group_sequence[reviewer] = sequence
    assert first_group_sequence[1] != first_group_sequence[2]
    for reviewer in (1, 2):
        with (out_a / f"reviewer_{reviewer}" / "pair_answers.csv").open(encoding="utf-8") as handle:
            pair_rows = list(csv.DictReader(handle))
        assert len(pair_rows) == 1
        assert pair_rows[0]["image_a_id"].endswith("-A")
        assert pair_rows[0]["image_b_id"].endswith("-B")


def test_packet_rejects_non_train_source_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    p1 = fixture["priority"] / "priority_p1_class1_intersection.csv"
    rows = list(csv.DictReader(p1.read_text(encoding="utf-8").splitlines()))
    rows[0]["relative_path"] = "val/C1/forbidden.jpg"
    _write_csv(p1, tuple(rows[0].keys()), rows)
    with pytest.raises(ValueError, match="TRAIN-relative"):
        build_packet(fixture["data_yaml"], fixture["priority"], tmp_path / "out", seed=1)


def test_packet_rejects_tampered_source_and_output_overlap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    expected_inputs = {
        "data_yaml_sha256": _sha256(fixture["data_yaml"]),
        "integrity_manifest_sha256": _sha256(fixture["integrity_manifest"]),
        "p1_csv_sha256": _sha256(fixture["priority"] / "priority_p1_class1_intersection.csv"),
        "p2_csv_sha256": _sha256(fixture["priority"] / "priority_p2_rgb_near_identical_pairs.csv"),
        "review_summary_sha256": _sha256(fixture["review_summary"]),
    }
    with pytest.raises(ValueError, match="overlaps protected input root"):
        build_packet(
            fixture["data_yaml"],
            fixture["priority"],
            fixture["root"] / "train" / "generated_packet",
            seed=1,
        )

    p1 = fixture["priority"] / "priority_p1_class1_intersection.csv"
    p1.write_text(p1.read_text(encoding="utf-8").replace("P1_source_secret", "P1_tampered_secret"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind|Locked input SHA256 mismatch"):
        build_packet(
            fixture["data_yaml"],
            fixture["priority"],
            tmp_path / "tampered_out",
            seed=1,
            integrity_manifest=fixture["integrity_manifest"],
            review_summary=fixture["review_summary"],
            expected_input_sha256=expected_inputs,
        )

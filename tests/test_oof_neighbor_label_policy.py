import csv
import json
from pathlib import Path

import numpy as np

from trkh.tools.build_oof_neighbor_label_policy import build_policy


def _write_data_yaml(root: Path) -> Path:
    train_root = root / "train"
    class_names = ["class_a", "class_b", "class_c", "class_d", "class_e"]
    for class_name in class_names:
        (train_root / class_name).mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        json.dumps(
            {
                "path": str(root),
                "train": "train",
                "val": "train",
                "test": "train",
                "format": "classification_folder",
                "names": {index: name for index, name in enumerate(class_names)},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def _write_cleanlab(path: Path, paths: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "sample_index",
                "true_index",
                "true_name",
                "suggested_index",
                "suggested_name",
                "is_label_issue",
                "issue_rank",
                "label_quality",
                "self_confidence",
                "top1_confidence",
                "top2_margin",
                "normalized_entropy",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": paths[0],
                "sample_index": 10,
                "true_index": 0,
                "true_name": "class_a",
                "suggested_index": 1,
                "suggested_name": "class_b",
                "is_label_issue": 1,
                "issue_rank": 1,
                "label_quality": 0.01,
                "self_confidence": 0.01,
                "top1_confidence": 0.95,
                "top2_margin": 0.90,
                "normalized_entropy": 0.05,
            }
        )
        writer.writerow(
            {
                "path": paths[3],
                "sample_index": 13,
                "true_index": 0,
                "true_name": "class_a",
                "suggested_index": 1,
                "suggested_name": "class_b",
                "is_label_issue": 1,
                "issue_rank": 2,
                "label_quality": 0.02,
                "self_confidence": 0.02,
                "top1_confidence": 0.94,
                "top2_margin": 0.89,
                "normalized_entropy": 0.05,
            }
        )


def test_build_policy_requires_cleanlab_and_neighbor_consensus(tmp_path: Path) -> None:
    data_yaml = _write_data_yaml(tmp_path / "data")
    paths = [
        str(tmp_path / "data" / "train" / "class_a" / "Image_1_box000.jpg"),
        str(tmp_path / "data" / "train" / "class_b" / "Image_2_box000.jpg"),
        str(tmp_path / "data" / "train" / "class_b" / "Image_3_box000.jpg"),
        str(tmp_path / "data" / "train" / "class_a" / "Image_4_box000.jpg"),
        str(tmp_path / "data" / "train" / "class_c" / "Image_5_box000.jpg"),
    ]
    for path in paths:
        Path(path).write_bytes(b"fake")
    cleanlab = tmp_path / "cleanlab.csv"
    _write_cleanlab(cleanlab, paths)
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.0, 0.99],
        ],
        dtype=np.float32,
    )
    feature_npz = tmp_path / "features_train_test_model.npz"
    np.savez(feature_npz, features=features, labels=np.asarray([0, 1, 1, 0, 2]), paths=np.asarray(paths))

    summary = build_policy(
        data_yaml=data_yaml,
        cleanlab_manifest=cleanlab,
        feature_npz=[feature_npz],
        output_dir=tmp_path / "out",
        top_k=2,
        min_vote_fraction=1.0,
        max_issue_rank=10,
        min_oof_confidence=0.70,
        max_self_confidence=0.20,
        exclude_same_image_id=True,
    )

    assert summary["candidate_rows"] == 2
    assert summary["strict_rows"] == 1
    with (tmp_path / "out" / "strict_relabel_or_ignore_candidates.csv").open("r", newline="", encoding="utf-8") as handle:
        strict_rows = list(csv.DictReader(handle))
    assert strict_rows[0]["image_path"] == paths[0]
    with (tmp_path / "out" / "strict_ignore_sample_weights_train_only.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        weight_rows = list(csv.DictReader(handle))
    assert weight_rows[0]["sample_weight"] == "0.05"
    assert weight_rows[0]["sample_index"] == "10"
    with (tmp_path / "out" / "strict_relabel_soft_targets_train_only.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        soft_rows = list(csv.DictReader(handle))
    assert soft_rows[0]["sample_index"] == "10"
    assert soft_rows[0]["soft_target_index"] == "1"
    assert soft_rows[0]["soft_0"] == "0.25"
    assert soft_rows[0]["soft_1"] == "0.75"


def test_build_policy_prefers_sample_index_when_paths_are_duplicate(tmp_path: Path) -> None:
    data_yaml = _write_data_yaml(tmp_path / "data")
    duplicate_path = str(tmp_path / "data" / "train" / "class_a" / "Image_1.jpg")
    Path(duplicate_path).write_bytes(b"fake")
    cleanlab = tmp_path / "cleanlab.csv"
    _write_cleanlab(cleanlab, [duplicate_path, duplicate_path, duplicate_path, duplicate_path, duplicate_path])
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.0, 0.99],
        ],
        dtype=np.float32,
    )
    feature_npz = tmp_path / "features_train_duplicate_paths.npz"
    np.savez(
        feature_npz,
        features=features,
        labels=np.asarray([0, 1, 1, 0, 0]),
        paths=np.asarray([duplicate_path, duplicate_path, duplicate_path, duplicate_path, duplicate_path]),
        sample_index=np.asarray([10, 11, 12, 13, 14]),
    )

    summary = build_policy(
        data_yaml=data_yaml,
        cleanlab_manifest=cleanlab,
        feature_npz=[feature_npz],
        output_dir=tmp_path / "out_dup",
        top_k=2,
        min_vote_fraction=1.0,
        max_issue_rank=10,
        min_oof_confidence=0.70,
        max_self_confidence=0.20,
        exclude_same_image_id=False,
    )

    assert summary["strict_rows"] == 1
    with (tmp_path / "out_dup" / "strict_relabel_or_ignore_candidates.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        strict_rows = list(csv.DictReader(handle))
    assert strict_rows[0]["sample_index"] == "10"

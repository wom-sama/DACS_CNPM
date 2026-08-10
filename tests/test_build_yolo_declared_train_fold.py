from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image
import pytest
import yaml

from trkh.tools.build_yolo_declared_train_fold import build_declared_fold


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tiny_yolo(root: Path) -> tuple[Path, Path]:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    labels = {
        "a": (0, 1),
        "b": (2,),
        "c": (1,),
        "d": (4,),
    }
    rows = []
    sample_index = 0
    folds = {"a": 0, "b": 1, "c": 1, "d": 0}
    for source_index, (stem, targets) in enumerate(labels.items()):
        image_path = image_dir / f"{stem}.png"
        Image.new("RGB", (12, 12), color=(20 + source_index, 30, 40)).save(image_path)
        label_path = label_dir / f"{stem}.txt"
        label_path.write_text(
            "".join(
                f"{target} 0.5 0.5 0.4 0.4\n"
                for target in targets
            ),
            encoding="utf-8",
        )
        for target in targets:
            rows.append(
                {
                    "condition": "clean",
                    "sample_index": sample_index,
                    "source_stem": stem,
                    "image_path": str(image_path.resolve()),
                    "fold": folds[stem],
                    "target_index": target,
                }
            )
            sample_index += 1
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "images/train",
                "val": "images/train",
                "test": "images/train",
                "nc": 5,
                "names": ["c0", "c1", "c2", "c3", "c4"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    declaration = root / "declaration.csv"
    with declaration.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data_yaml, declaration


def test_build_declared_fold_hardlinks_exact_source_disjoint_view(tmp_path: Path) -> None:
    data_yaml, declaration = _write_tiny_yolo(tmp_path / "raw")
    raw_hashes = {
        path: _sha256(path)
        for path in sorted((tmp_path / "raw").rglob("*"))
        if path.is_file()
    }
    output = tmp_path / "generated"
    summary = build_declared_fold(
        data_yaml=data_yaml,
        declaration=declaration,
        output_root=output,
        fold=0,
        expected_data_sha256=_sha256(data_yaml),
        expected_declaration_sha256=_sha256(declaration),
    )
    assert summary["total_rows"] == 5
    assert summary["total_sources"] == 4
    assert summary["fit_rows"] == 2
    assert summary["holdout_rows"] == 3
    assert summary["fit_sources"] == 2
    assert summary["holdout_sources"] == 2
    assert summary["source_overlap"] == 0
    assert summary["fit_class_counts"] == [0, 1, 1, 0, 0]
    assert summary["holdout_class_counts"] == [1, 1, 0, 0, 1]
    assert summary["hardlink_count"] == 12
    assert sorted(path.stem for path in (output / "images" / "train").iterdir()) == [
        "b",
        "c",
    ]
    assert sorted(path.stem for path in (output / "images" / "val").iterdir()) == [
        "a",
        "d",
    ]
    assert sorted(path.stem for path in (output / "images" / "test").iterdir()) == [
        "a",
        "d",
    ]
    generated_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert generated_yaml["held_out_fold"] == 0
    assert generated_yaml["test"] == "images/test"
    assert all(_sha256(path) == digest for path, digest in raw_hashes.items())
    with pytest.raises(FileExistsError):
        build_declared_fold(
            data_yaml=data_yaml,
            declaration=declaration,
            output_root=output,
            fold=0,
        )


def test_build_declared_fold_rejects_source_crossing_folds(tmp_path: Path) -> None:
    data_yaml, declaration = _write_tiny_yolo(tmp_path / "raw")
    rows = list(csv.DictReader(declaration.open("r", encoding="utf-8")))
    rows[1]["fold"] = "1"
    with declaration.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="crosses declaration folds"):
        build_declared_fold(
            data_yaml=data_yaml,
            declaration=declaration,
            output_root=tmp_path / "generated",
            fold=0,
        )

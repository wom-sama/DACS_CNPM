import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from trkh.data.dataset import QualityGroupDataset
from trkh.tools.build_quality_group_manifest import build_manifest
from trkh.training.train import _load_quality_group_manifest


class QualityGroupManifestTests(unittest.TestCase):
    def _write_data_yaml(self, root: Path) -> Path:
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    "format: classification_folder",
                    "path: .",
                    "train: train",
                    "val: val",
                    "test: test",
                    "nc: 2",
                    "class_name_mode: raw",
                    "names:",
                    "  0: c0",
                    "  1: c1",
                ]
            ),
            encoding="utf-8",
        )
        return data_yaml

    def _write_image(self, path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=color).save(path)

    def test_build_manifest_outputs_train_only_quality_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_yaml = self._write_data_yaml(root)
            train_a = root / "train" / "c0" / "a.jpg"
            train_b = root / "train" / "c1" / "b.jpg"
            self._write_image(train_a, (245, 245, 210))
            self._write_image(train_b, (35, 60, 25))

            predictions = root / "predictions.csv"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "target_index",
                        "prediction_index",
                        "confidence",
                        "correct",
                        "top2_probability",
                        "prob_0_c0",
                        "prob_1_c1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(train_a),
                        "target_index": 0,
                        "prediction_index": 1,
                        "confidence": 0.70,
                        "correct": 0,
                        "top2_probability": 0.20,
                        "prob_0_c0": 0.10,
                        "prob_1_c1": 0.70,
                    }
                )

            summary = build_manifest(
                data_yaml=data_yaml,
                output_dir=root / "quality",
                split="train",
                prediction_csvs=[predictions],
                group_key="class_quality",
                max_samples=0,
                copy_images=False,
                max_review_images_per_group=2,
                dry_run=False,
            )

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["prediction_matched_rows"], 1)
            self.assertGreaterEqual(summary["num_groups"], 1)
            manifest = root / "quality" / "quality_groups_train_only.csv"
            self.assertTrue(manifest.is_file())
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("train" in row["image_path"].replace("\\", "/") for row in rows))
            self.assertIn("hard_disagreement", {row["cartography_bucket"] for row in rows})

    def test_train_loader_skips_non_train_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "dataset" / "train" / "c0" / "a.jpg"
            val_path = root / "dataset" / "val" / "c0" / "b.jpg"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            val_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.write_bytes(b"x")
            val_path.write_bytes(b"x")
            manifest = root / "quality_groups.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "quality_group_index", "quality_group_name"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(train_path),
                        "quality_group_index": 2,
                        "quality_group_name": "train_group",
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(val_path),
                        "quality_group_index": 3,
                        "quality_group_name": "val_group",
                    }
                )

            groups, sample_groups, summary = _load_quality_group_manifest(str(manifest))

            self.assertEqual(summary["paths"], 1)
            self.assertEqual(summary["sample_indices"], 0)
            self.assertEqual(summary["skipped_non_train"], 1)
            self.assertEqual(list(groups.values()), [2])
            self.assertEqual(sample_groups, {})

    def test_quality_group_sample_index_does_not_collapse_duplicate_paths(self):
        class TwoObjectDataset(Dataset):
            def __init__(self, image_path: Path) -> None:
                self._paths = [image_path, image_path]

            def __len__(self) -> int:
                return 2

            def sample_paths(self):
                return list(self._paths)

            def __getitem__(self, index: int):
                return torch.zeros(3, 4, 4), int(index)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "dataset" / "train" / "c0" / "same.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"x")
            manifest = root / "quality_groups.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_index",
                        "image_path",
                        "quality_group_index",
                        "quality_group_name",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_index": 0,
                        "image_path": str(image_path),
                        "quality_group_index": 2,
                        "quality_group_name": "first_object",
                    }
                )
                writer.writerow(
                    {
                        "sample_index": 1,
                        "image_path": str(image_path),
                        "quality_group_index": 3,
                        "quality_group_name": "second_object",
                    }
                )

            groups, sample_groups, summary = _load_quality_group_manifest(str(manifest))
            dataset = QualityGroupDataset(
                TwoObjectDataset(image_path),
                groups,
                group_indices_by_sample_index=sample_groups,
            )

            self.assertEqual(groups, {})
            self.assertEqual(sample_groups, {0: 2, 1: 3})
            self.assertEqual(summary["key_mode"], "sample_index")
            self.assertEqual(dataset.quality_group_summary()["matched_samples"], 2)
            self.assertEqual(int(dataset[0][2]["quality_group_index"].item()), 2)
            self.assertEqual(int(dataset[1][2]["quality_group_index"].item()), 3)


if __name__ == "__main__":
    unittest.main()

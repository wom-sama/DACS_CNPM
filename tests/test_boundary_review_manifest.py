import csv
import tempfile
import unittest
from pathlib import Path

from trkh.tools.build_boundary_review_manifest import build_manifest


class BoundaryReviewManifestTests(unittest.TestCase):
    def _write_predictions(self, path: Path, train_image: Path, val_image: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_path",
                    "target_index",
                    "target_name",
                    "prediction_index",
                    "prediction_name",
                    "confidence",
                    "top2_index",
                    "top2_probability",
                    "prob_0_class0",
                    "prob_1_class1",
                    "prob_2_class2",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": str(train_image),
                    "target_index": 0,
                    "target_name": "class0",
                    "prediction_index": 1,
                    "prediction_name": "class1",
                    "confidence": 0.82,
                    "top2_index": 0,
                    "top2_probability": 0.10,
                    "prob_0_class0": 0.10,
                    "prob_1_class1": 0.82,
                    "prob_2_class2": 0.08,
                }
            )
            writer.writerow(
                {
                    "image_path": str(val_image),
                    "target_index": 1,
                    "target_name": "class1",
                    "prediction_index": 0,
                    "prediction_name": "class0",
                    "confidence": 0.80,
                    "top2_index": 1,
                    "top2_probability": 0.12,
                    "prob_0_class0": 0.80,
                    "prob_1_class1": 0.12,
                    "prob_2_class2": 0.08,
                }
            )

    def test_rejects_mixed_split_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_image = root / "class_f" / "train" / "class0" / "a.jpg"
            val_image = root / "class_f" / "val" / "class1" / "b.jpg"
            train_image.parent.mkdir(parents=True)
            val_image.parent.mkdir(parents=True)
            train_image.write_bytes(b"not-image")
            val_image.write_bytes(b"not-image")
            predictions = root / "predictions.csv"
            self._write_predictions(predictions, train_image, val_image)

            with self.assertRaises(ValueError):
                build_manifest(
                    predictions=predictions,
                    output_dir=root / "out",
                    split="auto",
                    image_stats_mode="none",
                )

    def test_allow_other_splits_keeps_train_and_writes_manual_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_image = root / "class_f" / "train" / "class0" / "a.jpg"
            val_image = root / "class_f" / "val" / "class1" / "b.jpg"
            train_image.parent.mkdir(parents=True)
            val_image.parent.mkdir(parents=True)
            train_image.write_bytes(b"not-image")
            val_image.write_bytes(b"not-image")
            predictions = root / "predictions.csv"
            self._write_predictions(predictions, train_image, val_image)

            summary = build_manifest(
                predictions=predictions,
                output_dir=root / "out",
                split="train",
                allow_other_splits=True,
                pairs="0-1,1-2",
                image_stats_mode="none",
                copy_images=False,
            )

            self.assertEqual(summary["split"], "train")
            self.assertEqual(summary["selected_rows"], 1)
            self.assertEqual(summary["skipped_other_split"], 1)
            self.assertEqual(summary["by_reason"], {"focus_false_positive": 1})

            with (root / "out" / "boundary_review_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0].keys())
            self.assertEqual(rows[0]["split"], "train")
            self.assertIn("manual_label_status", fieldnames)
            self.assertIn("quality_dirty_obstacle", fieldnames)
            self.assertIn("review_notes", fieldnames)

    def test_cartography_unseen_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seen_image = root / "class_f" / "train" / "class0" / "seen.jpg"
            unseen_image = root / "class_f" / "train" / "class0" / "unseen.jpg"
            seen_image.parent.mkdir(parents=True)
            seen_image.write_bytes(b"not-image")
            unseen_image.write_bytes(b"not-image")
            predictions = root / "cartography.csv"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "target_index",
                        "prediction_index",
                        "top2_index",
                        "top2_margin",
                        "seen_count",
                        "prob_0_class0",
                        "prob_1_class1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(unseen_image),
                        "target_index": 0,
                        "prediction_index": -1,
                        "top2_index": -1,
                        "top2_margin": 0.0,
                        "seen_count": 0,
                        "prob_0_class0": 0.0,
                        "prob_1_class1": 0.0,
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(seen_image),
                        "target_index": 0,
                        "prediction_index": 1,
                        "top2_index": 0,
                        "top2_margin": 0.05,
                        "seen_count": 1,
                        "prob_0_class0": 0.45,
                        "prob_1_class1": 0.55,
                    }
                )

            summary = build_manifest(
                predictions=predictions,
                output_dir=root / "out",
                split="train",
                pairs="0-1",
                image_stats_mode="none",
                copy_images=False,
            )

            self.assertEqual(summary["selected_rows"], 1)
            self.assertEqual(summary["skipped_unseen"], 1)


if __name__ == "__main__":
    unittest.main()

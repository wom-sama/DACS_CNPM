import csv
import tempfile
import unittest
from pathlib import Path

from trkh.tools.build_data_centric_sample_weights import build_manifest


class DataCentricSampleWeightTests(unittest.TestCase):
    def test_build_manifest_merges_base_weight_and_rejects_non_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "class_f" / "train" / "c0" / "a.jpg"
            val_path = root / "class_f" / "val" / "c0" / "b.jpg"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            val_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.write_bytes(b"not-an-image")
            val_path.write_bytes(b"not-an-image")

            predictions = root / "predictions.csv"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "target_index",
                        "prediction_index",
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
                        "image_path": str(train_path),
                        "target_index": 0,
                        "prediction_index": 1,
                        "confidence": 0.80,
                        "top2_index": 0,
                        "top2_probability": 0.10,
                        "prob_0_class0": 0.10,
                        "prob_1_class1": 0.80,
                        "prob_2_class2": 0.10,
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(val_path),
                        "target_index": 0,
                        "prediction_index": 1,
                        "confidence": 0.90,
                        "top2_index": 0,
                        "top2_probability": 0.05,
                        "prob_0_class0": 0.05,
                        "prob_1_class1": 0.90,
                        "prob_2_class2": 0.05,
                    }
                )

            base_manifest = root / "base.csv"
            with base_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "target_index", "prediction_index", "sample_weight", "reason"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(train_path),
                        "target_index": 0,
                        "prediction_index": 1,
                        "sample_weight": 2.0,
                        "reason": "base_weight",
                    }
                )

            summary = build_manifest(
                predictions=predictions,
                output_dir=root / "out",
                base_sample_weight_manifest=base_manifest,
                boundary_pairs="0-1,1-2",
                high_confidence_threshold=0.55,
                low_self_confidence_threshold=0.25,
                ambiguous_margin_threshold=0.06,
                high_confidence_multiplier=0.30,
                low_self_confidence_multiplier=0.45,
                ambiguous_multiplier=0.85,
                min_weight=0.25,
                max_weight=2.5,
                max_issues=20,
                max_per_reason=20,
                max_per_pair=20,
                allow_non_train_paths=False,
                copy_images=False,
                max_review_images_per_reason=5,
                dry_run=False,
            )

            self.assertEqual(summary["skipped_non_train"], 1)
            self.assertEqual(summary["selected_issues"], 1)
            self.assertEqual(summary["merged_manifest_rows"], 1)
            self.assertEqual(summary["by_reason"], {"high_confidence_disagreement": 1})

            with (root / "out" / "sample_weights_train_only.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["base_sample_weight"]), 2.0, places=5)
            self.assertAlmostEqual(float(rows[0]["issue_multiplier"]), 0.30, places=5)
            self.assertAlmostEqual(float(rows[0]["sample_weight"]), 0.60, places=5)

    def test_build_manifest_accepts_cleanlab_oof_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "class_f" / "train" / "c0" / "a.jpg"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.write_bytes(b"not-an-image")

            predictions = root / "cleanlab.csv"
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "path",
                        "true_index",
                        "true_name",
                        "pred_index",
                        "pred_name",
                        "suggested_index",
                        "suggested_name",
                        "is_label_issue",
                        "issue_rank",
                        "self_confidence",
                        "top1_confidence",
                        "top2_margin",
                        "prob_0_class0",
                        "prob_1_class1",
                        "prob_2_class2",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": str(train_path),
                        "true_index": 0,
                        "true_name": "class0",
                        "pred_index": 1,
                        "pred_name": "class1",
                        "suggested_index": 1,
                        "suggested_name": "class1",
                        "is_label_issue": 1,
                        "issue_rank": 1,
                        "self_confidence": 0.02,
                        "top1_confidence": 0.97,
                        "top2_margin": 0.94,
                        "prob_0_class0": 0.02,
                        "prob_1_class1": 0.97,
                        "prob_2_class2": 0.01,
                    }
                )

            summary = build_manifest(
                predictions=predictions,
                output_dir=root / "out",
                base_sample_weight_manifest=None,
                boundary_pairs="0-1,1-2",
                high_confidence_threshold=0.90,
                low_self_confidence_threshold=0.08,
                ambiguous_margin_threshold=0.04,
                high_confidence_multiplier=0.25,
                low_self_confidence_multiplier=0.45,
                ambiguous_multiplier=0.80,
                min_weight=0.20,
                max_weight=1.0,
                max_issues=20,
                max_per_reason=20,
                max_per_pair=20,
                allow_non_train_paths=False,
                copy_images=False,
                max_review_images_per_reason=5,
                dry_run=False,
            )

            self.assertEqual(summary["selected_issues"], 1)
            with (root / "out" / "sample_weights_train_only.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["prediction_index"], "1")
            self.assertEqual(rows[0]["prediction_name"], "class1")
            self.assertAlmostEqual(float(rows[0]["sample_weight"]), 0.25, places=5)


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path

from trkh.tools.build_clean_anchor_manifest import build_manifest


class CleanAnchorManifestTests(unittest.TestCase):
    def test_build_manifest_selects_balanced_train_only_correct_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train0 = root / "class_f" / "train" / "c0" / "a.jpg"
            train1a = root / "class_f" / "train" / "c1" / "b.jpg"
            train1b = root / "class_f" / "train" / "c1" / "c.jpg"
            val0 = root / "class_f" / "val" / "c0" / "d.jpg"
            for path in (train0, train1a, train1b, val0):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"img")

            cartography = root / "cartography.csv"
            with cartography.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "target_index",
                        "target_name",
                        "prediction_index",
                        "confidence_mean",
                        "correctness_mean",
                        "confidence_std",
                        "ce_loss_mean",
                        "top2_margin",
                        "prob_0",
                        "prob_1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(train0),
                        "target_index": 0,
                        "target_name": "c0",
                        "prediction_index": 0,
                        "confidence_mean": 0.72,
                        "correctness_mean": 1.0,
                        "confidence_std": 0.0,
                        "ce_loss_mean": 0.3,
                        "top2_margin": 0.42,
                        "prob_0": 0.72,
                        "prob_1": 0.28,
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(train1a),
                        "target_index": 1,
                        "target_name": "c1",
                        "prediction_index": 1,
                        "confidence_mean": 0.64,
                        "correctness_mean": 1.0,
                        "confidence_std": 0.0,
                        "ce_loss_mean": 0.45,
                        "top2_margin": 0.28,
                        "prob_0": 0.36,
                        "prob_1": 0.64,
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(train1b),
                        "target_index": 1,
                        "target_name": "c1",
                        "prediction_index": 0,
                        "confidence_mean": 0.80,
                        "correctness_mean": 0.0,
                        "confidence_std": 0.0,
                        "ce_loss_mean": 1.2,
                        "top2_margin": 0.50,
                        "prob_0": 0.80,
                        "prob_1": 0.20,
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(val0),
                        "target_index": 0,
                        "target_name": "c0",
                        "prediction_index": 0,
                        "confidence_mean": 0.99,
                        "correctness_mean": 1.0,
                        "confidence_std": 0.0,
                        "ce_loss_mean": 0.01,
                        "top2_margin": 0.90,
                        "prob_0": 0.99,
                        "prob_1": 0.01,
                    }
                )

            summary = build_manifest(
                cartography_csv=cartography,
                output_dir=root / "out",
                per_class=1,
                min_correctness=1.0,
                min_confidence=0.0,
                min_target_probability=0.0,
                max_confidence_std=-1.0,
                max_ce_loss=-1.0,
                sample_weight=1.4,
                allow_non_train_paths=False,
            )

            self.assertEqual(summary["selected"], 2)
            self.assertEqual(summary["selected_by_class"], {"0": 1, "1": 1})
            self.assertEqual(summary["skipped_non_train"], 1)
            self.assertEqual(summary["skipped_thresholds"], 1)

            with (root / "out" / "clean_anchor_hard_samples_train_only.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["target_index"] for row in rows], ["0", "1"])
            self.assertEqual({row["reason"] for row in rows}, {"clean_balanced_anchor"})
            self.assertTrue(all(float(row["sample_weight"]) == 1.4 for row in rows))


if __name__ == "__main__":
    unittest.main()

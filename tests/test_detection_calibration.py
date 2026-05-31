import csv
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

import model as model_module
from config import AugmentationConfig, ModelConfig, TrainConfig, load_data_spec
from evaluate import (
    _build_background_aware_classification_metrics,
    _build_detection_confidence_curve,
    _build_detection_confidence_curve_from_prepared,
    _compute_detection_metrics_at_threshold,
    _compute_prepared_detection_metrics_at_threshold,
    _detection_curve_metrics_at_index,
    _prepare_detection_records_for_confidence_curve,
    _resolve_adaptive_max_detections,
    _score_detection_queries,
)
from dataset import (
    RareClassRepeatDataset,
    build_rare_class_repeat_factors,
    build_train_collate_fn,
    build_train_transform,
)
from inference import post_process_detections, resolve_detection_output_limit
from loss import DETRSetCriterion
from matcher import HungarianMatcher
from model import create_model
from train import (
    apply_balance_file_auto_adjustment,
    _initial_training_progress_from_resume,
    _load_resume_configs_from_checkpoint,
    _load_training_checkpoint,
    _resolve_classification_overfit_guard,
    _resolve_eval_num_workers,
    _resolve_resume_checkpoint_path,
    _resolve_checkpoint_selection,
    _resolve_detection_stage,
    _save_interrupt_checkpoint,
    train_one_epoch,
)
from utils import append_csv_row, build_warmup_decay_scheduler, load_checkpoint, resolve_amp_dtype, save_checkpoint


class DetectionCalibrationTests(unittest.TestCase):
    def test_objectness_loss_balances_positive_and_negative_queries(self):
        criterion = DETRSetCriterion(
            num_classes=4,
            objectness_focal_alpha=0.75,
            objectness_focal_gamma=0.0,
        )
        logits = torch.zeros((1, 10), dtype=torch.float32)
        targets = torch.zeros_like(logits)
        targets[0, 0] = 1.0

        loss = criterion.objectness_loss(logits, targets, num_boxes=1)

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.item()), math.log(2.0), places=5)

    def test_matcher_can_use_objectness_cost(self):
        matcher = HungarianMatcher(
            cost_class=0.0,
            cost_bbox=0.0,
            cost_giou=0.0,
            cost_objectness=1.0,
        )
        model_output = {
            "logits": torch.zeros((1, 2, 4), dtype=torch.float32),
            "boxes": torch.full((1, 2, 4), 0.5, dtype=torch.float32),
            "objectness_logits": torch.tensor([[-5.0, 5.0]], dtype=torch.float32),
        }
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            }
        ]

        indices = matcher(model_output, targets)

        self.assertEqual(indices[0][0].tolist(), [1])
        self.assertEqual(indices[0][1].tolist(), [0])

    def test_matcher_sanitizes_non_finite_cost_entries(self):
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            cost_objectness=1.0,
        )
        model_output = {
            "logits": torch.tensor([[[float("nan"), 0.0], [float("inf"), -float("inf")]]], dtype=torch.float32),
            "boxes": torch.tensor([[[float("nan"), 0.5, 0.2, 0.2], [0.5, float("inf"), 0.2, 0.2]]], dtype=torch.float32),
            "objectness_logits": torch.tensor([[float("nan"), float("inf")]], dtype=torch.float32),
        }
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            }
        ]

        indices = matcher(model_output, targets)

        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0][0].numel(), 1)
        self.assertEqual(indices[0][1].tolist(), [0])

    def test_background_aware_metrics_record_calibrated_threshold(self):
        metrics = _build_background_aware_classification_metrics(
            targets=torch.tensor([0, 1, 1], dtype=torch.long),
            predictions_with_background=torch.tensor([0, 2, 1], dtype=torch.long),
            class_names=["a", "b"],
            threshold=0.08,
            score_mode="foreground",
        )

        self.assertEqual(metrics["threshold"], 0.08)
        self.assertEqual(metrics["background_predictions"], 1)
        self.assertAlmostEqual(metrics["background_prediction_rate"], 1.0 / 3.0)

    def test_objectness_detection_score_modes(self):
        record = {
            "scores": torch.tensor([0.18, 0.48], dtype=torch.float32),
            "class_scores": torch.tensor([0.9, 0.6], dtype=torch.float32),
            "objectness_scores": torch.tensor([0.2, 0.8], dtype=torch.float32),
        }

        torch.testing.assert_close(
            _score_detection_queries(record, "foreground"),
            torch.tensor([0.18, 0.48], dtype=torch.float32),
        )
        torch.testing.assert_close(
            _score_detection_queries(record, "class_only"),
            torch.tensor([0.9, 0.6], dtype=torch.float32),
        )
        torch.testing.assert_close(
            _score_detection_queries(record, "objectness"),
            torch.tensor([0.2, 0.8], dtype=torch.float32),
        )
        torch.testing.assert_close(
            _score_detection_queries(record, "class_objectness_mean"),
            torch.tensor([0.55, 0.70], dtype=torch.float32),
        )
        torch.testing.assert_close(
            _score_detection_queries(record, "class_objectness_min"),
            torch.tensor([0.2, 0.6], dtype=torch.float32),
        )

    def test_macro_detection_hmean_checkpoint_selection_penalizes_low_side(self):
        name, score, higher_is_better = _resolve_checkpoint_selection(
            ModelConfig(model_type="vit_registers_hybrid"),
            TrainConfig(best_metric="macro_detection_hmean"),
            {
                "macro_f1": 0.98,
                "bbox": {"mean_iou": 0.75},
                "detection_confidence_curve": {"best_f1_50": 0.70},
            },
        )

        self.assertEqual(name, "hmean_macro_f1_detection_f1")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(score, 2.0 * 0.98 * 0.70 / (0.98 + 0.70))

    def test_detection_confidence_curve_reuses_nms_without_metric_drift(self):
        records = [
            {
                "scores": torch.tensor([0.95, 0.90, 0.70, 0.20, 0.65], dtype=torch.float32),
                "classes": torch.tensor([0, 0, 1, 2, 1], dtype=torch.long),
                "boxes": torch.tensor(
                    [
                        [0.50, 0.50, 0.20, 0.20],
                        [0.51, 0.50, 0.20, 0.20],
                        [0.20, 0.20, 0.10, 0.10],
                        [0.80, 0.80, 0.10, 0.10],
                        [0.21, 0.20, 0.10, 0.10],
                    ],
                    dtype=torch.float32,
                ),
                "is_background_argmax": torch.tensor([False, False, False, False, True]),
                "target_labels": torch.tensor([0, 1], dtype=torch.long),
                "target_boxes": torch.tensor(
                    [
                        [0.50, 0.50, 0.20, 0.20],
                        [0.20, 0.20, 0.10, 0.10],
                    ],
                    dtype=torch.float32,
                ),
            }
        ]
        params = {
            "nms_iou_threshold": 0.5,
            "max_detections_per_image": 3,
            "require_foreground_argmax": True,
            "score_mode": "foreground",
        }
        prepared = _prepare_detection_records_for_confidence_curve(records, **params)

        for threshold in (0.0, 0.25, 0.75, 0.96):
            direct = _compute_detection_metrics_at_threshold(records, threshold=threshold, **params)
            reused = _compute_prepared_detection_metrics_at_threshold(
                prepared,
                threshold=threshold,
                **params,
            )
            for key in ("predicted_objects", "ground_truth_objects", "true_positive_50"):
                self.assertEqual(reused[key], direct[key])
            for key in ("precision_50", "recall_50", "f1_50"):
                self.assertAlmostEqual(float(reused[key]), float(direct[key]), places=6)

        curve = _build_detection_confidence_curve(records, num_thresholds=6, **params)
        curve_from_prepared = _build_detection_confidence_curve_from_prepared(
            prepared,
            num_thresholds=6,
            **params,
        )
        self.assertEqual(len(curve["thresholds"]), 6)
        self.assertGreaterEqual(float(curve["best_f1_50"]), 0.0)
        self.assertEqual(curve_from_prepared["predicted_objects"], curve["predicted_objects"])
        self.assertEqual(curve_from_prepared["true_positive_50"], curve["true_positive_50"])
        for index, threshold in enumerate(curve["thresholds"]):
            direct = _compute_detection_metrics_at_threshold(records, threshold=float(threshold), **params)
            self.assertAlmostEqual(curve["precision_50"][index], direct["precision_50"], places=6)
            self.assertAlmostEqual(curve["recall_50"][index], direct["recall_50"], places=6)
            self.assertAlmostEqual(curve["f1_50"][index], direct["f1_50"], places=6)
        threshold_zero = _detection_curve_metrics_at_index(curve, 0)
        direct_zero = _compute_detection_metrics_at_threshold(records, threshold=0.0, **params)
        self.assertEqual(threshold_zero["predicted_objects"], direct_zero["predicted_objects"])
        self.assertEqual(threshold_zero["true_positive_50"], direct_zero["true_positive_50"])

    def test_count_head_uses_cls_register_global_feature(self):
        model = create_model(
            num_classes=4,
            model_config={
                "model_type": "vit_registers_hybrid",
                "image_size": 32,
                "patch_size": 16,
                "stem_channels": 8,
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
                "num_registers": 2,
                "dropout": 0.0,
                "drop_path_rate": 0.0,
                "bbox_head_hidden_dim": 32,
                "num_queries": 5,
                "decoder_depth": 1,
                "decoder_num_heads": 4,
                "decoder_ffn_dim": 64,
                "decoder_dropout": 0.0,
                "head_pooling": "cls_register_mean",
                "count_head": True,
                "count_head_hidden_dim": 32,
                "count_head_dropout": 0.0,
                "count_head_prior": 1.2,
            },
        )
        model.eval()

        with torch.no_grad():
            outputs = model(torch.zeros((2, 3, 32, 32), dtype=torch.float32))

        self.assertIn("count_logits", outputs)
        self.assertEqual(tuple(outputs["count_logits"].shape), (2,))
        self.assertTrue(torch.isfinite(outputs["count_logits"]).all())

    def test_count_loss_is_weighted_in_detection_criterion(self):
        criterion = DETRSetCriterion(
            num_classes=4,
            cls_weight=0.0,
            objectness_weight=0.0,
            bbox_l1_weight=0.0,
            bbox_giou_weight=0.0,
            cardinality_weight=0.0,
            count_weight=2.0,
        )
        one_count_logit = torch.log(torch.expm1(torch.tensor(1.0, dtype=torch.float32)))
        model_output = {
            "logits": torch.zeros((2, 4, 4), dtype=torch.float32),
            "boxes": torch.full((2, 4, 4), 0.5, dtype=torch.float32),
            "objectness_logits": torch.zeros((2, 4), dtype=torch.float32),
            "count_logits": torch.stack((one_count_logit, one_count_logit)),
        }
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.long),
                "boxes": torch.full((1, 4), 0.5, dtype=torch.float32),
            },
            {
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
                "boxes": torch.full((3, 4), 0.5, dtype=torch.float32),
            },
        ]

        loss, details = criterion(model_output, targets, return_details=True)

        self.assertGreater(details["count_loss"], 0.0)
        self.assertAlmostEqual(float(loss.item()), 2.0 * details["count_loss"], places=5)

    def test_adaptive_max_detections_uses_count_head_or_objectness(self):
        record = {
            "scores": torch.ones((8,), dtype=torch.float32),
            "count_prediction": torch.tensor(2.2, dtype=torch.float32),
            "objectness_scores": torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32),
        }

        self.assertEqual(
            _resolve_adaptive_max_detections(
                record,
                adaptive_max_detections=True,
                count_source="count_head",
                count_margin=1,
                min_detections=1,
                max_detections_per_image=8,
            ),
            3,
        )
        self.assertEqual(
            _resolve_adaptive_max_detections(
                record,
                adaptive_max_detections=True,
                count_source="objectness",
                count_margin=1,
                min_detections=1,
                max_detections_per_image=8,
            ),
            3,
        )

    def test_adaptive_confidence_curve_caps_overpredicted_queries(self):
        record = {
            "scores": torch.tensor([0.90, 0.80, 0.70, 0.60], dtype=torch.float32),
            "classes": torch.tensor([0, 0, 0, 0], dtype=torch.long),
            "boxes": torch.tensor(
                [
                    [0.15, 0.15, 0.10, 0.10],
                    [0.40, 0.15, 0.10, 0.10],
                    [0.65, 0.15, 0.10, 0.10],
                    [0.90, 0.15, 0.10, 0.10],
                ],
                dtype=torch.float32,
            ),
            "count_prediction": torch.tensor(1.0, dtype=torch.float32),
            "is_background_argmax": torch.tensor([False, False, False, False]),
            "target_labels": torch.tensor([0], dtype=torch.long),
            "target_boxes": torch.tensor([[0.15, 0.15, 0.10, 0.10]], dtype=torch.float32),
        }

        fixed = _compute_detection_metrics_at_threshold(
            [record],
            threshold=0.0,
            nms_iou_threshold=0.1,
            max_detections_per_image=4,
        )
        adaptive = _compute_detection_metrics_at_threshold(
            [record],
            threshold=0.0,
            nms_iou_threshold=0.1,
            max_detections_per_image=4,
            adaptive_max_detections=True,
            adaptive_count_source="count_head",
            adaptive_count_margin=0,
            adaptive_min_detections=1,
        )

        self.assertEqual(fixed["predicted_objects"], 4)
        self.assertEqual(adaptive["predicted_objects"], 1)
        self.assertGreater(adaptive["precision_50"], fixed["precision_50"])

    def test_detection_output_limit_defaults_to_checkpoint_queries(self):
        checkpoint = {"model_config": {"num_queries": 40}}

        self.assertEqual(resolve_detection_output_limit(checkpoint, 0), 40)
        self.assertEqual(resolve_detection_output_limit(checkpoint, 5), 5)

    def test_postprocess_keeps_last_class_when_objectness_head_exists(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 8.0],
                    [8.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        boxes = torch.tensor(
            [
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.2, 0.2, 0.1, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
        objectness_logits = torch.tensor([[8.0, -8.0]], dtype=torch.float32)

        detections = post_process_detections(
            logits=logits,
            boxes=boxes,
            objectness_logits=objectness_logits,
            conf_threshold=0.1,
            max_detections=None,
            nms_iou_threshold=0.5,
        )[0]

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_index"], 3)

    def test_postprocess_keeps_last_class_for_more_than_four_classes(self):
        logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 9.0]]], dtype=torch.float32)
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32)
        objectness_logits = torch.tensor([[9.0]], dtype=torch.float32)

        detections = post_process_detections(
            logits=logits,
            boxes=boxes,
            objectness_logits=objectness_logits,
            conf_threshold=0.1,
            max_detections=None,
            nms_iou_threshold=0.5,
        )[0]

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_index"], 5)

    def test_data_spec_raw_mode_supports_more_than_four_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_yaml = f"""
path: {temp_path.as_posix()}
train: images/train
val: images/val
nc: 6
names:
  0: ripe
  1: unripe
  2: bruised
  3: rotten
  4: flower
  5: leaf
class_name_mode: raw
"""
            data_yaml_path = temp_path / "data.yaml"
            data_yaml_path.write_text(data_yaml, encoding="utf-8")

            spec = load_data_spec(data_yaml_path, expected_num_classes=6)

        self.assertEqual(spec.num_classes, 6)
        self.assertEqual(spec.class_name_mode, "raw")
        self.assertEqual(spec.class_names[-1], "leaf")

    def test_create_model_blocks_pretrained_and_external_weight_options(self):
        blocked_configs = [
            {"model_type": "vit_registers_hybrid", "pretrained": True},
            {"model_type": "resnet50", "weights": "DEFAULT"},
            {"model_type": "vit_b_16", "pretrained_weights": "imagenet"},
            {"model_type": "mobilenet_v3_large", "init_checkpoint": "external.pt"},
        ]

        for config in blocked_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "pretrained/external weights"):
                    create_model(num_classes=2, model_config=config)

    def test_torchvision_backbones_are_requested_with_weights_none(self):
        calls = []

        class FakeResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3, 4)

        class FakeMobileNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(3, 4))

        class FakeViT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 4)

        original_resnet50 = model_module.tv_models.resnet50
        original_mobilenet = model_module.tv_models.mobilenet_v3_large
        original_vit = model_module.tv_models.vit_b_16

        def fake_resnet50(*args, **kwargs):
            calls.append(("resnet50", kwargs.get("weights", "missing")))
            return FakeResNet()

        def fake_mobilenet_v3_large(*args, **kwargs):
            calls.append(("mobilenet_v3_large", kwargs.get("weights", "missing")))
            return FakeMobileNet()

        def fake_vit_b_16(*args, **kwargs):
            calls.append(("vit_b_16", kwargs.get("weights", "missing")))
            return FakeViT()

        try:
            model_module.tv_models.resnet50 = fake_resnet50
            model_module.tv_models.mobilenet_v3_large = fake_mobilenet_v3_large
            model_module.tv_models.vit_b_16 = fake_vit_b_16
            create_model(num_classes=2, model_config={"model_type": "resnet50"})
            create_model(num_classes=2, model_config={"model_type": "mobilenet_v3_large"})
            create_model(num_classes=2, model_config={"model_type": "vit_b_16", "image_size": 224})
        finally:
            model_module.tv_models.resnet50 = original_resnet50
            model_module.tv_models.mobilenet_v3_large = original_mobilenet
            model_module.tv_models.vit_b_16 = original_vit

        self.assertEqual(
            calls,
            [
                ("resnet50", None),
                ("mobilenet_v3_large", None),
                ("vit_b_16", None),
            ],
        )

    def test_amp_dtype_defaults_to_auto_bf16_when_supported(self):
        previous = os.environ.pop("TRKH_AMP_DTYPE", None)
        original_is_bf16_supported = torch.cuda.is_bf16_supported

        try:
            torch.cuda.is_bf16_supported = lambda: True
            self.assertEqual(resolve_amp_dtype(torch.device("cuda")), torch.bfloat16)

            os.environ["TRKH_AMP_DTYPE"] = "float16"
            self.assertEqual(resolve_amp_dtype(torch.device("cuda")), torch.float16)
        finally:
            torch.cuda.is_bf16_supported = original_is_bf16_supported
            os.environ.pop("TRKH_AMP_DTYPE", None)
            if previous is not None:
                os.environ["TRKH_AMP_DTYPE"] = previous

    def test_canbang_yaml_auto_configures_balance_without_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_yaml_path = temp_path / "data.yaml"
            data_yaml_path.write_text(
                f"""
path: {temp_path.as_posix()}
train: images/train
val: images/val
nc: 3
names:
  0: class_a
  1: class_b
  2: class_c
""",
                encoding="utf-8",
            )
            (temp_path / "canbang.yaml").write_text(
                """
dataset_balance:
  version_note: "Giu nguyen class ID goc, khong gop nhan"
  total_pairs: 1000
  train_ratio_config: 0.7
  val_ratio_config: 0.2
  test_ratio_config: 0.1
  classes:
    "0":
      count: 350
      ratio: 0.35
      percent: 35.0
    "1":
      count: 250
      ratio: 0.25
      percent: 25.0
    "2":
      count: 400
      ratio: 0.40
      percent: 40.0
""",
                encoding="utf-8",
            )

            spec = load_data_spec(data_yaml_path)
            train_config = TrainConfig(auto_tune_imbalance=False)
            augmentation_config = AugmentationConfig(
                class_aware_augmentation=False,
                rare_class_repeat=False,
                class_aware_photometric_augmentation=True,
            )
            summary = apply_balance_file_auto_adjustment(
                data_spec=spec,
                train_config=train_config,
                augmentation_config=augmentation_config,
            )

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["class_counts"], [350, 250, 400])
        self.assertGreater(summary["auto_repeat_factors"][1], summary["auto_repeat_factors"][0])
        self.assertEqual(summary["auto_repeat_factors"][2], 1.0)
        self.assertTrue(augmentation_config.class_aware_augmentation)
        self.assertTrue(augmentation_config.rare_class_repeat)
        self.assertFalse(augmentation_config.class_aware_photometric_augmentation)
        self.assertTrue(train_config.auto_tune_imbalance)

    def test_detection_mosaic_collate_creates_multi_object_targets(self):
        torch.manual_seed(7)
        collate = build_train_collate_fn(
            num_classes=4,
            batch_mix_probability=1.0,
            mosaic_probability=1.0,
            cutmix_probability=0.0,
            max_detection_objects=40,
        )
        batch = []
        for index in range(4):
            image = torch.full((3, 32, 32), float(index) / 4.0)
            target = {
                "labels": torch.tensor([index % 4], dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
            }
            batch.append((image, target))

        images, targets = collate(batch)

        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertTrue(all(target["labels"].numel() > 1 for target in targets))
        self.assertTrue(all(target["boxes"].shape[-1] == 4 for target in targets))

    def test_rare_class_repeat_targets_extreme_class_two_three_imbalance(self):
        class TinyLabelDataset(torch.utils.data.Dataset):
            def __init__(self, labels):
                self._labels = list(labels)

            def __len__(self):
                return len(self._labels)

            def __getitem__(self, index):
                return int(self._labels[index])

            def labels(self):
                return list(self._labels)

        generator = torch.Generator().manual_seed(123)
        class_two = torch.full((6,), 2, dtype=torch.long)
        class_three = torch.full((40,), 3, dtype=torch.long)
        labels = torch.cat((class_two, class_three), dim=0)
        labels = labels[torch.randperm(labels.numel(), generator=generator)].tolist()
        factors = build_rare_class_repeat_factors(
            [0, 0, 6, 40],
            repeat_power=0.5,
            max_factor=3.0,
            min_ratio=0.35,
        )

        repeated = RareClassRepeatDataset(
            TinyLabelDataset(labels),
            class_repeat_factors=factors,
            seed=123,
        )
        repeated_labels = repeated.labels()

        self.assertGreater(factors[2], 2.0)
        self.assertEqual(factors[3], 1.0)
        self.assertGreater(repeated_labels.count(2), labels.count(2))
        self.assertEqual(repeated_labels.count(3), labels.count(3))
        self.assertGreater(repeated_labels.count(2) / max(1, repeated_labels.count(3)), 0.25)

    def test_class_aware_scale_does_not_amplify_photometric_by_default(self):
        transform = build_train_transform(
            image_size=32,
            brightness=0.2,
            contrast=0.2,
            lighting_probability=0.15,
            scale_photometric_with_augmentation=False,
        )

        self.assertFalse(transform.scale_photometric_with_augmentation)

    def test_classification_overfit_guard_reduces_cls_weight_when_detection_lags(self):
        train_config = TrainConfig(
            classification_overfit_guard=True,
            classification_guard_macro_f1_threshold=0.95,
            classification_guard_detection_gap=0.20,
            classification_guard_min_cls_weight=0.25,
        )
        previous_metrics = {
            "macro_f1": 0.99,
            "detection_confidence_curve": {"best_f1_50": 0.50},
        }

        summary = _resolve_classification_overfit_guard(
            train_config=train_config,
            detection_mode=True,
            stage_name="stage2_full_detection",
            previous_val_metrics=previous_metrics,
        )

        self.assertTrue(summary["active"])
        self.assertLess(summary["multiplier"], 1.0)
        self.assertGreaterEqual(summary["multiplier"], 0.25)

    def test_eval_num_workers_can_be_overridden_for_cached_validation(self):
        previous = os.environ.pop("TRKH_EVAL_NUM_WORKERS", None)
        try:
            self.assertEqual(_resolve_eval_num_workers(TrainConfig(num_workers=6, eval_num_workers=-1)), 6)
            self.assertEqual(_resolve_eval_num_workers(TrainConfig(num_workers=6, eval_num_workers=0)), 0)
            os.environ["TRKH_EVAL_NUM_WORKERS"] = "0"
            self.assertEqual(_resolve_eval_num_workers(TrainConfig(num_workers=6, eval_num_workers=-1)), 0)
        finally:
            os.environ.pop("TRKH_EVAL_NUM_WORKERS", None)
            if previous is not None:
                os.environ["TRKH_EVAL_NUM_WORKERS"] = previous

    def test_detection_stage1_disables_auxiliary_detection_losses(self):
        train_config = TrainConfig(
            stage1_epochs=3,
            objectness_loss_weight=5.0,
            cardinality_loss_weight=1.0,
            count_loss_weight=0.8,
            quality_loss_weight=0.7,
            auxiliary_loss_weight=0.2,
            count_objectness_consistency_weight=0.1,
        )

        stage = _resolve_detection_stage(
            train_config=train_config,
            epoch_index=1,
            detection_mode=True,
        )

        self.assertEqual(stage["stage_name"], "stage1_cls_only")
        self.assertEqual(stage["objectness_weight"], 0.0)
        self.assertEqual(stage["cardinality_weight"], 0.0)
        self.assertEqual(stage["count_weight"], 0.0)
        self.assertEqual(stage["quality_weight"], 0.0)
        self.assertEqual(stage["auxiliary_weight"], 0.0)
        self.assertEqual(stage["count_objectness_consistency_weight"], 0.0)

    def test_detection_loss_casts_bf16_outputs_to_stable_fp32(self):
        criterion = DETRSetCriterion(
            num_classes=4,
            objectness_weight=5.0,
            cardinality_weight=0.5,
            count_weight=0.25,
            quality_weight=0.25,
            auxiliary_weight=0.1,
            count_objectness_consistency_weight=0.05,
        )
        logits = torch.randn((2, 6, 4), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        boxes = torch.rand((2, 6, 4), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        objectness_logits = torch.randn((2, 6), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        quality_logits = torch.randn((2, 6), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        count_logits = torch.randn((2,), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        aux_logits = torch.randn((2, 6, 4), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        aux_boxes = torch.rand((2, 6, 4), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        aux_objectness_logits = torch.randn((2, 6), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        aux_quality_logits = torch.randn((2, 6), dtype=torch.float32).to(torch.bfloat16).requires_grad_()
        targets = [
            {
                "labels": torch.tensor([0, 2], dtype=torch.long),
                "boxes": torch.tensor(
                    [[0.35, 0.35, 0.20, 0.20], [0.70, 0.60, 0.15, 0.18]],
                    dtype=torch.float32,
                ),
            },
            {
                "labels": torch.tensor([1], dtype=torch.long),
                "boxes": torch.tensor([[0.45, 0.55, 0.25, 0.25]], dtype=torch.float32),
            },
        ]
        model_output = {
            "logits": logits,
            "boxes": boxes,
            "objectness_logits": objectness_logits,
            "quality_logits": quality_logits,
            "count_logits": count_logits,
            "aux_outputs": [
                {
                    "logits": aux_logits,
                    "boxes": aux_boxes,
                    "objectness_logits": aux_objectness_logits,
                    "quality_logits": aux_quality_logits,
                }
            ],
        }

        loss, details = criterion(model_output, targets, return_details=True)
        loss.backward()

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(math.isfinite(details["loss"]))
        for tensor in (
            logits,
            boxes,
            objectness_logits,
            quality_logits,
            count_logits,
            aux_logits,
            aux_boxes,
            aux_objectness_logits,
            aux_quality_logits,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())

    def test_train_one_epoch_aborts_repeated_nonfinite_gradients(self):
        class InfGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.shape = tuple(value.shape)
                return value.sum() * 0.0 + 1.0

            @staticmethod
            def backward(ctx, grad_output):
                return torch.full(ctx.shape, float("inf"))

        class InfGradCriterion(torch.nn.Module):
            def forward(self, logits, labels):
                return InfGradient.apply(logits)

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        dataloader = [
            (torch.ones(1, 2), torch.tensor([0], dtype=torch.long)),
            (torch.ones(1, 2), torch.tensor([0], dtype=torch.long)),
        ]

        with self.assertRaisesRegex(RuntimeError, "Gradient norm became non-finite"):
            train_one_epoch(
                model=model,
                dataloader=dataloader,
                criterion=InfGradCriterion(),
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                device=torch.device("cpu"),
                amp=False,
                grad_clip_norm=1.0,
                epoch_index=1,
                max_nonfinite_grad_steps=2,
            )

    def test_resume_checkpoint_restores_training_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            resume_path = temp_path / "last.pt"
            model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                model.weight.fill_(0.25)
                model.bias.fill_(0.5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = build_warmup_decay_scheduler(
                optimizer=optimizer,
                warmup_epochs=1,
                warmup_start_factor=0.1,
                total_epochs=10,
                min_learning_rate=1e-5,
                decay_style="cosine",
            )
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step(3.0)
            expected_lr = scheduler.get_last_lr()[0]
            checkpoint = {
                "epoch": 3,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "class_names": ["a", "b"],
                "best_macro_f1": 0.8,
                "best_epoch": 2,
                "best_selection_metric": {
                    "name": "composite",
                    "value": 0.7,
                    "higher_is_better": True,
                },
                "epochs_without_improvement": 1,
                "metrics": {"macro_f1": 0.8},
            }
            save_checkpoint(resume_path, checkpoint)

            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            restored_scheduler = build_warmup_decay_scheduler(
                optimizer=restored_optimizer,
                warmup_epochs=1,
                warmup_start_factor=0.1,
                total_epochs=10,
                min_learning_rate=1e-5,
                decay_style="cosine",
            )
            summary, restored_checkpoint = _load_training_checkpoint(
                resume_path=resume_path,
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                scaler=None,
                device=torch.device("cpu"),
                class_names=["a", "b"],
            )

        self.assertEqual(summary["next_epoch"], 4)
        self.assertTrue(summary["optimizer_restored"])
        self.assertTrue(summary["scheduler_restored"])
        self.assertEqual(restored_checkpoint["best_epoch"], 2)
        self.assertAlmostEqual(restored_scheduler.get_last_lr()[0], expected_lr)
        for original, restored in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.allclose(original, restored))

    def test_auto_resume_resolves_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            last_path = run_dir / "checkpoints" / "last.pt"
            last_path.parent.mkdir(parents=True)
            save_checkpoint(last_path, {"epoch": 1, "model_state": {}})

            resolved = _resolve_resume_checkpoint_path(
                resume_path=None,
                auto_resume=True,
                run_dir=run_dir,
            )
            loaded = load_checkpoint(resolved)

        self.assertEqual(resolved, last_path.resolve())
        self.assertEqual(loaded["epoch"], 1)

    def test_disable_resume_overrides_explicit_and_auto_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            last_path = run_dir / "checkpoints" / "last.pt"
            last_path.parent.mkdir(parents=True)
            save_checkpoint(last_path, {"epoch": 1, "model_state": {}})

            resolved = _resolve_resume_checkpoint_path(
                resume_path=last_path,
                auto_resume=True,
                run_dir=run_dir,
                disable_resume=True,
            )

        self.assertIsNone(resolved)

    def test_resume_reset_epoch_starts_phase_progress_from_one(self):
        resume_summary = {
            "completed_epoch": 104,
            "next_epoch": 105,
        }
        checkpoint = {
            "epoch": 104,
            "best_macro_f1": 0.994,
            "best_epoch": 104,
            "best_selection_metric": {
                "name": "detection_f1",
                "value": 0.8518,
                "higher_is_better": True,
            },
            "epochs_without_improvement": 0,
            "metrics": {
                "macro_f1": 0.993,
                "detection_confidence_curve": {"best_f1_50": 0.8518},
            },
        }

        inherited = _initial_training_progress_from_resume(
            resume_summary,
            checkpoint,
            reset_epoch=False,
        )
        reset = _initial_training_progress_from_resume(
            resume_summary,
            checkpoint,
            reset_epoch=True,
        )

        self.assertEqual(inherited["start_epoch"], 105)
        self.assertAlmostEqual(inherited["best_selection_metric_value"], 0.8518)
        self.assertEqual(reset["start_epoch"], 1)
        self.assertEqual(reset["best_epoch"], 0)
        self.assertEqual(reset["best_macro_f1"], -1.0)
        self.assertIsNone(reset["best_selection_metric_value"])
        self.assertEqual(reset["source_completed_epoch"], 104)
        self.assertEqual(reset["source_best_epoch"], 104)
        self.assertIs(reset["previous_val_metrics"], checkpoint["metrics"])

    def test_resume_loads_configs_from_checkpoint_without_redeclaring_flags(self):
        checkpoint = {
            "data_yaml": "D:/data/data.yaml",
            "model_config": {
                "model_type": "vit_registers_hybrid",
                "image_size": 224,
                "num_queries": 40,
            },
            "train_config": {
                "epochs": 0,
                "batch_size": 8,
                "classification_overfit_guard": True,
            },
            "augmentation_config": {
                "resize_mode": "pad",
                "color_jitter_brightness": 0.05,
                "rare_class_repeat": True,
            },
        }

        model_config, train_config, augmentation_config, data_yaml = _load_resume_configs_from_checkpoint(
            checkpoint
        )

        self.assertEqual(model_config.image_size, 224)
        self.assertEqual(model_config.num_queries, 40)
        self.assertEqual(train_config.epochs, 0)
        self.assertEqual(train_config.batch_size, 8)
        self.assertEqual(augmentation_config.color_jitter_brightness, 0.05)
        self.assertEqual(str(data_yaml).replace("\\", "/"), "D:/data/data.yaml")

    def test_keyboard_interrupt_checkpoint_is_resumeable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            interrupt_path = temp_path / "interrupt.pt"
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = build_warmup_decay_scheduler(
                optimizer=optimizer,
                warmup_epochs=1,
                warmup_start_factor=0.1,
                total_epochs=10,
                min_learning_rate=1e-5,
                decay_style="cosine",
            )

            _save_interrupt_checkpoint(
                path=interrupt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=7,
                model_config=ModelConfig(image_size=224, num_queries=40),
                train_config=TrainConfig(epochs=0),
                augmentation_config=AugmentationConfig(color_jitter_brightness=0.05),
                data_yaml=temp_path / "data.yaml",
                class_names=["a", "b"],
                metrics={"macro_f1": 0.9},
                data_summary={"class_names": ["a", "b"]},
                imbalance_summary={"class_counts": [1, 1]},
                best_macro_f1=0.9,
                best_epoch=5,
                best_selection_metric_name="composite",
                best_selection_metric_value=0.8,
                best_selection_metric_higher_is_better=True,
                epochs_without_improvement=2,
            )
            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            restored_scheduler = build_warmup_decay_scheduler(
                optimizer=restored_optimizer,
                warmup_epochs=1,
                warmup_start_factor=0.1,
                total_epochs=10,
                min_learning_rate=1e-5,
                decay_style="cosine",
            )
            summary, restored_checkpoint = _load_training_checkpoint(
                resume_path=interrupt_path,
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                scaler=None,
                device=torch.device("cpu"),
                class_names=["a", "b"],
            )

        self.assertEqual(summary["next_epoch"], 8)
        self.assertEqual(restored_checkpoint["resume_state"]["checkpoint_kind"], "keyboard_interrupt")
        self.assertEqual(restored_checkpoint["best_epoch"], 5)

    def test_history_csv_expands_schema_when_timing_columns_are_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.csv"
            append_csv_row(
                history_path,
                row={"epoch": 1, "train_loss": 1.0},
                fieldnames=["epoch", "train_loss"],
            )
            append_csv_row(
                history_path,
                row={"epoch": 2, "train_loss": 0.9, "train_seconds": 12.5},
                fieldnames=["epoch", "train_loss", "train_seconds"],
            )

            with history_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertIn("train_seconds", reader.fieldnames or [])
        self.assertEqual(rows[0]["train_seconds"], "")
        self.assertEqual(rows[1]["train_seconds"], "12.5")


if __name__ == "__main__":
    unittest.main()

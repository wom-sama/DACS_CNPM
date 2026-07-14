import csv
import importlib.util
import json
import math
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from PIL import Image, ImageDraw

import trkh.models.model as model_module
from trkh.core.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AugmentationConfig,
    ModelConfig,
    TrainConfig,
    load_data_spec,
)
from trkh.evaluation.attention_viz import summarize_heatmap_focus
from trkh.evaluation.evaluate import (
    _build_prediction_records,
    _build_background_aware_classification_metrics,
    _checkpoint_data_path_mismatch,
    _build_detection_confidence_curve,
    _build_detection_confidence_curve_from_prepared,
    _compute_detection_metrics_at_threshold,
    _compute_prepared_detection_metrics_at_threshold,
    _detection_curve_metrics_at_index,
    _prepare_detection_records_for_confidence_curve,
    _resolve_adaptive_max_detections,
    _score_detection_queries,
    save_evaluation_artifacts,
)
from trkh.evaluation.metrics import build_metrics
from trkh.models.feature_hooks import (
    build_attention_heatmap,
    build_attention_rollout_heatmap,
    build_gradient_weighted_attention_rollout_heatmap,
    resolve_feature_hook,
    summarize_register_attention,
)
from trkh.data.dataset import (
    AmbiguousSoftTargetDataset,
    ClassificationFolderDataset,
    HardSampleRepeatDataset,
    IndexedSampleDataset,
    MangoYOLOCropDataset,
    RareClassRepeatDataset,
    SampleWeightDataset,
    StrictBalancedBatchSampler,
    TargetedMarginDataset,
    _apply_copypaste_detection_batch,
    _apply_foreground_background_mix_batch,
    _grabcut_foreground_mask_array,
    _pseudo_foreground_mask_array,
    _pseudo_foreground_mask_from_tensor_images,
    build_rare_class_repeat_factors,
    build_eval_transform,
    build_train_collate_fn,
    build_train_transform,
)
from trkh.inference.inference import (
    post_process_detections,
    render_prediction_image,
    resolve_detection_output_limit,
)
from trkh.tools.filter_classification_single_source import build_single_source_classification_dataset
from trkh.tools.build_classification_merged_classes import build_merged_classification_dataset
from trkh.training.loss import DETRSetCriterion
from trkh.training.losses import (
    BalancedSoftmaxFocalLoss,
    FocalCrossEntropyLoss,
    LDAMFocalLoss,
    SupervisedContrastiveLoss,
)
from trkh.training.matcher import HungarianMatcher
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    create_model,
)
from trkh.training.train import (
    ModelEMA,
    apply_balance_file_auto_adjustment,
    _initial_training_progress_from_resume,
    _load_resume_configs_from_checkpoint,
    _load_training_checkpoint,
    _apply_adaptive_detection_loss,
    _combine_class_target_scales,
    _resolve_adaptive_detection_loss,
    _resolve_classification_overfit_guard,
    _resolve_rare_class_recall_guard,
    _resolve_eval_num_workers,
    _resolve_resume_checkpoint_path,
    _resolve_checkpoint_selection,
    _resolve_detection_stage,
    _save_interrupt_checkpoint,
    _build_attention_guided_views,
    _attention_guided_score_map,
    _classification_loss_with_sample_weights,
    _background_counterfactual_images,
    _background_counterfactual_consistency_loss,
    _early_learning_regularization_loss,
    _angular_margin_loss_from_features,
    _boundary_contrastive_loss_from_features,
    _parse_boundary_contrastive_pairs,
    _bounded_attention_drop_mask,
    _forward_train_loss,
    _foreground_consistency_loss_from_features,
    _metric_learning_loss_from_features,
    _ordinal_maturity_loss_from_features,
    _ordinal_boundary_loss_from_logits,
    _pairwise_margin_loss_from_features,
    _topk_reassessment_auxiliary_loss_from_features,
    _register_diversity_loss_from_features,
    _pseudo_foreground_mask_from_normalized_images,
    _parse_metric_learning_sources,
    _pairwise_confusion_loss_from_features,
    _mutual_channel_loss_from_features,
    _complement_entropy_loss_from_logits,
    _load_ambiguous_soft_target_manifest,
    _load_sample_weight_manifest,
    _load_targeted_margin_manifest,
    _targeted_margin_loss_from_logits,
    train_one_epoch,
)
from trkh.core.utils import (
    append_csv_row,
    build_warmup_decay_scheduler,
    load_checkpoint,
    plot_dataset_color_audit,
    plot_per_class_validation_metric,
    plot_train_val_final_test_metrics,
    resolve_amp_dtype,
    save_checkpoint,
)


class DetectionCalibrationTests(unittest.TestCase):
    @staticmethod
    def _normalized_rgb_tensor(rgb: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(IMAGENET_MEAN, dtype=rgb.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=rgb.dtype).view(3, 1, 1)
        return (rgb - mean) / std

    def test_checkpoint_data_path_mismatch_detects_different_split_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_data = root / "dataset_a" / "data.yaml"
            requested_data = root / "dataset_b" / "data.yaml"
            mismatch = _checkpoint_data_path_mismatch(
                {"data_yaml": str(checkpoint_data)},
                requested_data,
            )
            self.assertIsNotNone(mismatch)
            self.assertEqual(
                Path(mismatch["checkpoint_data_yaml"]),
                checkpoint_data.resolve(strict=False),
            )
            self.assertIsNone(
                _checkpoint_data_path_mismatch(
                    {"data_yaml": str(checkpoint_data)},
                    checkpoint_data,
                )
            )

    def test_model_ema_uses_warmup_decay_and_restores_state(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.9)

        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)

        expected = 1.0 - (2.0 / 11.0)
        self.assertTrue(torch.allclose(ema.module.weight, torch.full_like(model.weight, expected)))
        restored = ModelEMA(model, decay=0.9)
        restored.load_state_dict(ema.state_dict())
        self.assertTrue(torch.equal(restored.module.weight, ema.module.weight))

    def test_attention_rollout_heatmap_has_requested_output_size(self):
        attention = torch.eye(6).unsqueeze(0).repeat(2, 1, 1)
        heatmap = build_attention_rollout_heatmap(
            attentions=[attention, attention],
            grid_size=(2, 2),
            prefix_tokens=2,
            output_size=(8, 8),
            query_tokens="cls_register_mean",
        )
        self.assertEqual(tuple(heatmap.shape), (8, 8))
        self.assertTrue(float(heatmap.min()) >= 0.0)
        self.assertTrue(float(heatmap.max()) <= 1.0)

    def test_gradient_weighted_rollout_accepts_attention_gradients(self):
        attention = torch.ones(1, 2, 6, 6, requires_grad=True)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        attention.retain_grad()
        attention[:, :, 0, 3].sum().backward()
        heatmap = build_gradient_weighted_attention_rollout_heatmap(
            attentions=[attention],
            grid_size=(2, 2),
            prefix_tokens=2,
            output_size=(8, 8),
            query_tokens="cls_register_mean",
        )

        self.assertEqual(tuple(heatmap.shape), (8, 8))
        self.assertTrue(float(heatmap.max()) <= 1.0)
        self.assertGreater(float(heatmap.max()), 0.0)

    def test_gradient_weighted_rollout_reports_missing_gradient_fallback(self):
        attention = torch.eye(6).unsqueeze(0).repeat(2, 1, 1)
        heatmap, provenance = build_gradient_weighted_attention_rollout_heatmap(
            attentions=[attention],
            grid_size=(2, 2),
            prefix_tokens=2,
            output_size=(8, 8),
            query_tokens="cls_register_mean",
            return_metadata=True,
        )

        self.assertEqual(tuple(heatmap.shape), (8, 8))
        self.assertEqual(provenance["gradient_layer_count"], 0)
        self.assertEqual(provenance["fallback_layer_count"], 1)
        self.assertEqual(provenance["missing_gradient_layer_count"], 1)
        self.assertFalse(provenance["all_layers_gradient_weighted"])

    def test_register_attention_summary_reports_similarity(self):
        attention = torch.zeros(1, 2, 6, 6)
        attention[:, :, :, :] = 1.0 / 6.0
        summary = summarize_register_attention([attention], prefix_tokens=2)

        self.assertIn("register_to_patch_attention_mean", summary)
        self.assertIn("cls_register_heatmap_similarity", summary)
        self.assertGreaterEqual(summary["cls_register_heatmap_similarity"], 0.0)

    def test_heatmap_focus_reports_border_attention(self):
        heatmap = torch.zeros(20, 20).numpy()
        heatmap[:2, :] = 1.0
        crop = Image.new("RGB", (20, 20), color=(128, 180, 64))
        summary = summarize_heatmap_focus(heatmap, crop)
        self.assertGreater(summary["border_mass"], 0.9)
        self.assertGreaterEqual(summary["background_mass"], 0.0)

    def test_resolve_feature_hook_can_target_stem_last(self):
        class PatchEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(3, 8, kernel_size=16, stride=16)

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 4, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, padding=1)),
                )
                self.patch_embed = PatchEmbed()

        model = ToyModel()
        spec = resolve_feature_hook(model, feature_source="stem_last")
        self.assertIs(spec.module, model.stem[2][0])
        self.assertTrue(spec.source.startswith("stem."))

    def test_resolve_feature_hook_can_target_complete_stem_output(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 4, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(4, 8, kernel_size=3, padding=1),
                )

        model = ToyModel()
        spec = resolve_feature_hook(model, feature_source="stem_output")
        self.assertIs(spec.module, model.stem)
        self.assertEqual(spec.source, "stem.output")

    def test_vit_registers_ignores_detection_only_model_config_keys(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                embed_dim=64,
                depth=1,
                num_heads=4,
                num_registers=2,
                num_queries=40,
                quality_head=True,
                count_head=True,
                auxiliary_decoder_outputs=True,
            ),
        )
        self.assertEqual(getattr(model, "model_type", ""), "vit_registers")
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_build_model_from_legacy_classifier_hybrid_checkpoint(self):
        base = create_model(
            num_classes=4,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        legacy_state = dict(base.state_dict())
        legacy_state["bbox_head.0.weight"] = torch.randn(16, 32)
        legacy_state["bbox_head.0.bias"] = torch.randn(16)
        legacy_state["bbox_head.1.weight"] = torch.ones(16)
        legacy_state["bbox_head.1.bias"] = torch.zeros(16)
        legacy_state["bbox_head.4.weight"] = torch.randn(4, 16)
        legacy_state["bbox_head.4.bias"] = torch.randn(4)
        legacy_state["classification_head.weight"] = legacy_state["head.weight"].clone()
        legacy_state["classification_head.bias"] = legacy_state["head.bias"].clone()
        checkpoint = {
            "class_names": ["a", "b", "c", "d"],
            "model_config": {
                "model_type": "vit_registers_hybrid",
                "image_size": 64,
                "patch_size": 16,
                "stem_channels": 8,
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
                "num_registers": 2,
                "bbox_head_hidden_dim": 16,
            },
            "model_state": legacy_state,
        }

        model = build_model_from_checkpoint(checkpoint)

        self.assertEqual(getattr(model, "model_type", ""), "vit_registers")
        self.assertFalse(hasattr(model, "bbox_head"))
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_cnn_feature_fusion_can_extend_existing_vit_checkpoint(self):
        base = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        fusion = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                cnn_feature_fusion=True,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        missing, unexpected = fusion.load_flexible_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(str(key).startswith("cnn_fusion_head.") for key in missing))

        base.eval()
        fusion.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), fusion(images), atol=1e-6))

    def test_fine_grained_pooling_can_extend_existing_vit_checkpoint(self):
        base = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        fine_grained = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                fine_grained_pooling=True,
                fine_grained_pooling_dropout=0.0,
            ),
        )
        missing, unexpected = fine_grained.load_flexible_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(str(key).startswith("fine_grained_pool.") for key in missing))

        base.eval()
        fine_grained.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), fine_grained(images), atol=1e-6))

    def test_dual_patch_norm_vit_registers_forward_shape(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                dual_patch_norm=True,
            ),
        )
        self.assertTrue(model.patch_embed.dual_patch_norm)
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_color_stat_fusion_can_extend_existing_vit_checkpoint(self):
        base = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        color_fusion = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                color_stat_fusion=True,
                color_stat_fusion_dropout=0.0,
            ),
        )
        missing, unexpected = color_fusion.load_flexible_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(str(key).startswith("color_fusion_head.") for key in missing))

        base.eval()
        color_fusion.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), color_fusion(images), atol=1e-6))

    def test_defect_stat_fusion_can_extend_existing_vit_checkpoint(self):
        base = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        defect_fusion = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                defect_stat_fusion=True,
                defect_stat_fusion_dropout=0.0,
            ),
        )
        missing, unexpected = defect_fusion.load_flexible_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(str(key).startswith("defect_fusion_head.") for key in missing))

        base.eval()
        defect_fusion.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), defect_fusion(images), atol=1e-6))

    def test_foreground_surface_fusion_can_extend_existing_vit_checkpoint(self):
        base = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        surface_fusion = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                foreground_surface_fusion=True,
                foreground_surface_fusion_dropout=0.0,
            ),
        )
        missing, unexpected = surface_fusion.load_flexible_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(str(key).startswith("foreground_surface_fusion_head.") for key in missing))

        base.eval()
        surface_fusion.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), surface_fusion(images), atol=1e-6))

    def test_bilinear_patch_fusion_can_extend_existing_vit_checkpoint(self):
        base = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                fine_grained_pooling=True,
            ),
        )
        bilinear = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                fine_grained_pooling=True,
                bilinear_patch_fusion=True,
                bilinear_patch_rank=16,
                bilinear_patch_dropout=0.0,
            ),
        )
        missing, unexpected = bilinear.load_flexible_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(any(str(key).startswith("bilinear_patch_fusion_head.") for key in missing))

        base.eval()
        bilinear.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), bilinear(images), atol=1e-6))

    def test_frequency_selective_pooling_blend_zero_preserves_checkpoint(self):
        base = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=32,
                patch_size=8,
                use_cnn_stem=False,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                dropout=0.0,
                attention_dropout=0.0,
                drop_path_rate=0.0,
            ),
        )
        selective = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=32,
                patch_size=8,
                use_cnn_stem=False,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                dropout=0.0,
                attention_dropout=0.0,
                drop_path_rate=0.0,
                frequency_selective_pooling=True,
                frequency_selective_top_k=1,
                frequency_selective_blend=0.0,
            ),
        )
        missing, unexpected = selective.load_flexible_state_dict(
            base.state_dict(),
            strict=False,
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        images = torch.randn(2, 3, 32, 32)
        base.eval()
        selective.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(images), selective(images), atol=1e-6))

    def test_frequency_selective_pooling_votes_are_normalized_and_traced(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=32,
                patch_size=8,
                use_cnn_stem=False,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                dropout=0.0,
                attention_dropout=0.0,
                drop_path_rate=0.0,
                frequency_selective_pooling=True,
                frequency_selective_top_k=2,
                frequency_selective_blend=1.0,
                frequency_selective_foreground_threshold=0.35,
            ),
        )
        images = torch.randn(2, 3, 32, 32)
        features = model.forward_features(images, return_trace=True)
        logits = classification_logits_from_features(model, features)
        votes = features["trace"]["frequency_selective_vote_fraction"]
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(votes.shape), (2, 16))
        self.assertTrue(
            torch.allclose(
                votes.sum(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.isfinite(features["frequency_selective_feature"]).all())

    def test_frequency_selective_pooling_respects_foreground_candidates(self):
        pool = model_module.FrequencySelectivePatchPooling(
            dim=8,
            top_k=2,
            foreground_threshold=0.5,
        )
        patches = torch.randn(1, 5, 8)
        foreground_prior = torch.tensor([[0.1, 0.8, 0.9, 0.2, 0.7]])
        _, trace = pool(
            patches,
            foreground_prior=foreground_prior,
            return_trace=True,
        )
        selected = set(trace["selected_indices"].flatten().tolist())
        self.assertTrue(selected.issubset({1, 2, 4}))

    def test_yolo_crop_dataset_exposes_sample_paths_for_color_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            image_path = images_dir / "sample.jpg"
            Image.new("RGB", (32, 32), (128, 180, 64)).save(image_path)
            (labels_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            dataset = MangoYOLOCropDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                classification_target=True,
                classification_object_crops=True,
                num_classes=1,
            )

            self.assertEqual(dataset.sample_paths(), [image_path])

    def test_multi_branch_token_fusion_forward_shape(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                multi_branch_fusion=True,
                branch_color_tokens=1,
                branch_edge_tokens=1,
                branch_cnn_tokens=1,
                branch_token_dropout=0.0,
                head_pooling="cls_branch_register_mean",
            ),
        )
        images = torch.randn(2, 3, 64, 64)
        logits = model(images)
        features = model.forward_features(images, return_attention=True)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(features["branch_tokens"].shape), (2, 3, 32))
        self.assertEqual(int(model.num_prefix_tokens), 6)
        self.assertEqual(tuple(features["grid_size"]), (4, 4))
        self.assertEqual(tuple(features["patches"].shape), (2, 16, 32))
        self.assertEqual(tuple(features["tokens"].shape), (2, 22, 32))

    def test_token_pruning_reduces_patch_count_and_preserves_xai_grid(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=6,
                num_heads=4,
                num_registers=2,
                detail_patch_enhancement=True,
                token_pruning=True,
                token_prune_layers="2,4",
                token_keep_rates="0.75,0.50",
            ),
        )
        images = torch.randn(2, 3, 64, 64)
        features = model.forward_features(images, return_trace=True)
        self.assertEqual(tuple(features["patches"].shape), (2, 8, 32))
        self.assertEqual(tuple(features["patch_indices"].shape), (2, 8))
        self.assertEqual(
            [
                (int(item["before_count"]), int(item["after_count"]))
                for item in features["trace"]["pruning"]
            ],
            [(16, 12), (12, 8)],
        )

        xai_features = model.forward_features(images, return_attention=True)
        self.assertEqual(tuple(xai_features["patches"].shape), (2, 16, 32))
        self.assertEqual(len(xai_features["attentions"]), 6)

    def test_fine_grained_pool_exposes_patch_attention_after_pruning(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=4,
                num_heads=4,
                num_registers=2,
                fine_grained_pooling=True,
                fine_grained_pooling_dropout=0.0,
                token_pruning=True,
                token_prune_layers="2",
                token_keep_rates="0.50",
            ),
        )
        images = torch.randn(2, 3, 64, 64)
        features = model.forward_features(images)
        logits = classification_logits_from_features(model, features)
        attention = features.get("fine_grained_attention")

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(attention.shape), (2, 8))
        self.assertTrue(
            torch.allclose(
                attention.sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )

    def test_fine_grained_pool_masks_invalid_patch_attention(self):
        pool = model_module.FineGrainedPatchPooling(dim=8, dropout=0.0)
        global_feature = torch.randn(2, 8)
        patch_tokens = torch.randn(2, 4, 8)
        valid_mask = torch.tensor(
            [[True, True, False, False], [False, False, False, False]],
            dtype=torch.bool,
        )
        attention = pool.attention_weights(
            global_feature,
            patch_tokens,
            valid_mask=valid_mask,
        )

        self.assertTrue(torch.allclose(attention[0, 2:], torch.zeros(2), atol=1e-7))
        self.assertTrue(torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-6))

    def test_attention_guided_crop_and_drop_build_valid_views(self):
        images = torch.randn(2, 3, 32, 32)
        features = {
            "patches": torch.randn(2, 4, 8),
            "grid_size": (2, 2),
            "patch_indices": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            "fine_grained_attention": torch.tensor(
                [[0.02, 0.03, 0.05, 0.90], [0.90, 0.05, 0.03, 0.02]]
            ),
            "foreground_prior": torch.tensor(
                [[0.0, 0.0, 0.2, 1.0], [1.0, 0.2, 0.0, 0.0]]
            ),
        }

        crop_views, crop_indices, crop_stats = _build_attention_guided_views(
            images=images,
            features=features,
            crop_probability=1.0,
            drop_probability=0.0,
            crop_threshold=0.55,
            drop_threshold=0.70,
            crop_padding_ratio=0.05,
            crop_min_area_ratio=0.25,
            foreground_weight=0.35,
            drop_blur_kernel=5,
        )
        drop_views, drop_indices, drop_stats = _build_attention_guided_views(
            images=images,
            features=features,
            crop_probability=0.0,
            drop_probability=1.0,
            crop_threshold=0.55,
            drop_threshold=0.70,
            crop_padding_ratio=0.05,
            crop_min_area_ratio=0.25,
            foreground_weight=0.35,
            drop_blur_kernel=5,
        )

        self.assertEqual(tuple(crop_views.shape), tuple(images.shape))
        self.assertEqual(tuple(drop_views.shape), tuple(images.shape))
        self.assertEqual(crop_indices.tolist(), [0, 1])
        self.assertEqual(drop_indices.tolist(), [0, 1])
        self.assertEqual(crop_stats["attention_crop_fraction"], 1.0)
        self.assertEqual(drop_stats["attention_drop_fraction"], 1.0)
        self.assertFalse(torch.allclose(crop_views, images))
        self.assertFalse(torch.allclose(drop_views, images))
        self.assertGreaterEqual(drop_stats["attention_drop_area_fraction"], 0.06)
        self.assertLessEqual(drop_stats["attention_drop_area_fraction"], 0.16)

    def test_attention_drop_mask_enforces_area_bounds(self):
        score = torch.zeros(2, 1, 20, 20)
        score[0, 0, 10, 10] = 1.0
        score[1, 0] = 1.0

        mask = _bounded_attention_drop_mask(
            score,
            threshold=0.70,
            dilation_kernel=5,
            min_area_ratio=0.06,
            max_area_ratio=0.16,
        )
        fractions = mask.float().flatten(1).mean(dim=1)

        self.assertTrue(torch.all(fractions >= 0.06))
        self.assertTrue(torch.all(fractions <= 0.16))

    def test_surface_detail_attention_score_uses_detail_map_and_foreground_gate(self):
        images = torch.zeros(1, 3, 32, 32)
        features = {
            "patches": torch.randn(1, 4, 8),
            "grid_size": (2, 2),
            "patch_indices": torch.tensor([[0, 1, 2, 3]]),
            "fine_grained_attention": torch.tensor([[0.90, 0.05, 0.03, 0.02]]),
            "foreground_prior": torch.tensor([[0.0, 0.0, 0.2, 1.0]]),
            "detail_map": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        }

        learned = _attention_guided_score_map(
            images=images,
            features=features,
            foreground_weight=0.8,
            score_source="learned_attention",
        )
        surface = _attention_guided_score_map(
            images=images,
            features=features,
            foreground_weight=0.8,
            score_source="surface_detail",
        )

        self.assertEqual(tuple(surface.shape), (1, 1, 32, 32))
        self.assertFalse(torch.allclose(learned, surface))
        self.assertGreater(
            float(surface[0, 0, 24, 24]),
            float(surface[0, 0, 8, 8]),
        )

    def test_attention_view_loss_is_finite_and_backpropagates(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                fine_grained_pooling=True,
                fine_grained_pooling_dropout=0.0,
            ),
        )
        images = torch.randn(3, 3, 64, 64)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        loss, _, _, loss_details, _ = _forward_train_loss(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            images=images,
            labels=None,
            targets=targets,
            device=torch.device("cpu"),
            amp=False,
            attention_view_loss_weight=0.5,
            attention_crop_probability=1.0,
            attention_drop_probability=0.0,
            attention_view_start_epoch=1,
            epoch_index=1,
            attention_drop_blur_kernel=5,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss_details["attention_view_loss"], 0.0)
        self.assertEqual(loss_details["attention_view_fraction"], 1.0)
        gradient_sum = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)

    def test_color_fusion_logits_are_used_by_feature_training_path(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                color_stat_fusion=True,
            ),
        )
        with torch.no_grad():
            model.color_fusion_head.net[-1].bias.fill_(0.25)
        images = torch.randn(2, 3, 64, 64)
        features = model.forward_features(images)
        logits = classification_logits_from_features(model, features)
        base_logits = model.head(model.head_input_from_features(features))
        self.assertTrue(torch.allclose(logits - base_logits, torch.full_like(logits, 0.25), atol=1e-5))

    def test_defect_fusion_logits_are_used_by_feature_training_path(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                defect_stat_fusion=True,
            ),
        )
        with torch.no_grad():
            model.defect_fusion_head.net[-1].bias.fill_(0.2)
        images = torch.randn(2, 3, 64, 64)
        features = model.forward_features(images)
        logits = classification_logits_from_features(model, features)
        base_logits = model.head(model.head_input_from_features(features))
        self.assertTrue(torch.allclose(logits - base_logits, torch.full_like(logits, 0.2), atol=1e-5))

    def test_foreground_surface_fusion_logits_are_used_and_traced(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                foreground_surface_fusion=True,
            ),
        )
        with torch.no_grad():
            model.foreground_surface_fusion_head.net[-1].bias.fill_(0.15)
        images = torch.randn(2, 3, 64, 64)
        features = model.forward_features(images, return_trace=True)
        logits = classification_logits_from_features(model, features)
        base_logits = model.head(model.head_input_from_features(features))
        self.assertTrue(torch.allclose(logits - base_logits, torch.full_like(logits, 0.15), atol=1e-5))
        self.assertIn("foreground_surface_stats", features["trace"])
        self.assertIn("foreground_surface_weight_map", features["trace"])
        self.assertEqual(tuple(features["trace"]["foreground_surface_stats"].shape), (2, 129))

    def test_bilinear_patch_fusion_logits_are_used_and_traced(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                fine_grained_pooling=True,
                bilinear_patch_fusion=True,
                bilinear_patch_rank=16,
                bilinear_patch_dropout=0.0,
                token_pruning=True,
                token_prune_layers="1",
                token_keep_rates="0.50",
            ),
        )
        with torch.no_grad():
            model.bilinear_patch_fusion_head.net[-1].bias.fill_(0.12)
        images = torch.randn(2, 3, 64, 64)
        features = model.forward_features(images, return_trace=True)
        logits = classification_logits_from_features(model, features)
        base_logits = model.head(model.head_input_from_features(features))

        self.assertTrue(torch.allclose(logits - base_logits, torch.full_like(logits, 0.12), atol=1e-5))
        self.assertEqual(tuple(features["trace"]["bilinear_patch_attention"].shape), (2, 8))
        self.assertEqual(tuple(features["trace"]["bilinear_patch_descriptor"].shape), (2, 288))
        self.assertTrue(
            torch.allclose(
                features["trace"]["bilinear_patch_attention"].sum(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )

    def test_pairwise_margin_head_adjusts_logits_and_has_auxiliary_loss(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                pairwise_margin_head=True,
                pairwise_margin_pairs="0-1,4-rest",
                pairwise_margin_logit_scale=1.0,
                pairwise_margin_dropout=0.0,
            ),
        )
        images = torch.randn(4, 3, 64, 64)
        features = model.forward_features(images)
        base_logits = model.head(model.head_input_from_features(features))
        with torch.no_grad():
            model.pairwise_margin_head.bias[0] = 1.0
            model.pairwise_margin_head.bias[1] = -0.5
        logits = classification_logits_from_features(model, features)
        pairwise_logits = features.get("pairwise_margin_logits")
        aux_loss = _pairwise_margin_loss_from_features(
            model=model,
            features=features,
            targets=torch.tensor([0, 1, 4, 3], dtype=torch.long),
        )

        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(pairwise_logits.shape), (4, 2))
        pair_delta = logits[:, 1] - logits[:, 0] - (base_logits[:, 1] - base_logits[:, 0])
        self.assertGreater(float(pair_delta.mean()), 1.9)
        self.assertTrue(torch.isfinite(aux_loss).item())

    def test_pairwise_margin_routing_only_opens_matching_ambiguous_expert(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=32,
                patch_size=8,
                use_cnn_stem=False,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                dropout=0.0,
                attention_dropout=0.0,
                drop_path_rate=0.0,
                pairwise_margin_head=True,
                pairwise_margin_pairs="0-1,1-2,4-rest",
                pairwise_margin_logit_scale=1.0,
                pairwise_margin_dropout=0.0,
                pairwise_margin_routing=True,
                pairwise_margin_route_max_probability_margin=0.20,
            ),
        )
        base_logits = torch.tensor(
            [
                [1.00, 0.95, 0.00, 0.00, 0.00],
                [0.00, 1.00, 0.95, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.95, 1.00],
            ]
        )
        pairwise_logits = torch.ones(3, 3)
        adjustment, routes = model.pairwise_margin_adjustment(
            pairwise_logits,
            base_logits,
            return_route_weights=True,
        )

        self.assertGreater(float(routes[0, 0].item()), 0.0)
        self.assertEqual(float(routes[0, 1].item()), 0.0)
        self.assertEqual(float(routes[0, 2].item()), 0.0)
        self.assertEqual(float(routes[1, 0].item()), 0.0)
        self.assertGreater(float(routes[1, 1].item()), 0.0)
        self.assertEqual(float(routes[1, 2].item()), 0.0)
        self.assertEqual(float(routes[2, 0].item()), 0.0)
        self.assertEqual(float(routes[2, 1].item()), 0.0)
        self.assertGreater(float(routes[2, 2].item()), 0.0)
        self.assertNotEqual(float(adjustment[0, 0].item()), 0.0)
        self.assertEqual(float(adjustment[0, 2].item()), 0.0)

    def test_pairwise_margin_routing_closes_on_high_confidence_margin(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=32,
                patch_size=8,
                use_cnn_stem=False,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                pairwise_margin_head=True,
                pairwise_margin_pairs="0-1",
                pairwise_margin_routing=True,
                pairwise_margin_route_max_probability_margin=0.05,
            ),
        )
        base_logits = torch.tensor([[5.0, 0.0, -1.0, -1.0, -1.0]])
        routes = model.pairwise_margin_route_weights(base_logits)
        self.assertTrue(torch.equal(routes, torch.zeros_like(routes)))

    def test_topk_reassessment_zero_init_is_neutral_and_routes_boundary(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=32,
                patch_size=8,
                use_cnn_stem=False,
                embed_dim=32,
                depth=2,
                num_heads=4,
                num_registers=2,
                dropout=0.0,
                attention_dropout=0.0,
                drop_path_rate=0.0,
                topk_reassessment_head=True,
                topk_reassessment_top_k=2,
                topk_reassessment_hidden_dim=32,
                topk_reassessment_dropout=0.0,
                topk_reassessment_logit_scale=1.0,
                topk_reassessment_routing=True,
                topk_reassessment_route_pairs="0-1,1-2",
                topk_reassessment_route_max_probability_margin=0.25,
            ),
        )
        images = torch.randn(2, 3, 32, 32)
        features = model.forward_features(images)
        base_logits = model.head(model.head_input_from_features(features))
        neutral_logits = classification_logits_from_features(model, features)
        self.assertTrue(torch.allclose(base_logits, neutral_logits, atol=1e-6))
        self.assertIn("topk_reassessment_base_logits", features)
        self.assertIn("topk_reassessment_route_weights", features)

        with torch.no_grad():
            final = model.topk_reassessment_head.net[-1]
            final.bias[1] = 1.0
        routed_base = torch.tensor(
            [
                [1.00, 0.96, 0.00, 0.00, 0.00],
                [2.00, 0.00, 0.00, 0.00, 1.80],
            ]
        )
        head_input = torch.randn(2, 32)
        reassessment_logits = model.topk_reassessment_logits_from_head_input(
            head_input,
            routed_base,
        )
        adjustment, routes = model.topk_reassessment_adjustment(
            reassessment_logits,
            routed_base,
            return_route_weights=True,
        )

        self.assertGreater(float(routes[0].item()), 0.0)
        self.assertAlmostEqual(float(routes[1].item()), 0.0, places=6)
        self.assertGreater(float(adjustment[0, 1].item()), 0.0)
        self.assertAlmostEqual(float(adjustment[1].abs().sum().item()), 0.0, places=6)

    def test_topk_reassessment_aux_loss_routes_only_adjustment(self):
        base_logits = torch.tensor(
            [
                [1.00, 0.96, 0.00, 0.00, 0.00],
                [2.00, 0.00, 0.00, 0.00, 1.80],
            ],
            requires_grad=True,
        )
        adjustment = torch.zeros_like(base_logits, requires_grad=True)
        features = {
            "topk_reassessment_base_logits": base_logits,
            "topk_reassessment_adjustment": adjustment,
            "topk_reassessment_route_weights": torch.tensor([0.8, 0.01]),
        }
        targets = torch.tensor([1, 4], dtype=torch.long)

        loss, fraction = _topk_reassessment_auxiliary_loss_from_features(
            features=features,
            targets=targets,
            route_min_weight=0.05,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertAlmostEqual(fraction, 0.5, places=6)
        self.assertIsNone(base_logits.grad)
        self.assertGreater(float(adjustment.grad[0].abs().sum().item()), 0.0)
        self.assertAlmostEqual(float(adjustment.grad[1].abs().sum().item()), 0.0, places=6)

    def test_ordinal_maturity_head_orders_non_defect_classes(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                ordinal_maturity_head=True,
                ordinal_maturity_classes="0,1,2,3",
                ordinal_maturity_logit_scale=1.0,
                ordinal_maturity_dropout=0.0,
            ),
        )
        images = torch.randn(4, 3, 64, 64)
        features = model.forward_features(images)
        base_logits = model.head(model.head_input_from_features(features))
        with torch.no_grad():
            model.ordinal_maturity_head.bias.fill_(1.0)
        logits = classification_logits_from_features(model, features)
        maturity_score = features.get("ordinal_maturity_score")
        aux_loss = _ordinal_maturity_loss_from_features(
            model=model,
            features=features,
            targets=torch.tensor([0, 1, 2, 4], dtype=torch.long),
        )

        self.assertEqual(tuple(maturity_score.shape), (4, 1))
        delta = logits - base_logits
        self.assertLess(float(delta[:, 0].mean()), float(delta[:, 1].mean()))
        self.assertLess(float(delta[:, 1].mean()), float(delta[:, 2].mean()))
        self.assertLess(float(delta[:, 2].mean()), float(delta[:, 3].mean()))
        self.assertTrue(torch.allclose(delta[:, 4], torch.zeros_like(delta[:, 4])))
        self.assertTrue(torch.isfinite(aux_loss).item())

    def test_angular_margin_loss_is_finite_and_backpropagates(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        images = torch.randn(4, 3, 64, 64)
        features = model.forward_features(images)
        loss = _angular_margin_loss_from_features(
            model=model,
            features=features,
            targets=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            margin=0.10,
            scale=8.0,
            classes="0,1,2,3",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(model.head.weight.grad)

    def test_angular_margin_loss_ignores_unselected_classes(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        images = torch.randn(3, 3, 64, 64)
        features = model.forward_features(images)
        selected_loss = _angular_margin_loss_from_features(
            model=model,
            features=features,
            targets=torch.tensor([0, 1, 4], dtype=torch.long),
            margin=0.10,
            scale=8.0,
            classes="0,1",
        )
        ignored_loss = _angular_margin_loss_from_features(
            model=model,
            features=features,
            targets=torch.tensor([4, 4, 4], dtype=torch.long),
            margin=0.10,
            scale=8.0,
            classes="0,1",
        )

        self.assertTrue(torch.isfinite(selected_loss).item())
        self.assertGreater(float(selected_loss.item()), 0.0)
        self.assertEqual(float(ignored_loss.item()), 0.0)

    def test_ordinal_boundary_loss_is_finite_and_backpropagates(self):
        logits = torch.tensor(
            [
                [2.0, 1.0, -1.0, -2.0, 0.0],
                [0.4, 1.3, 0.7, -0.5, 0.0],
                [-1.0, 0.3, 1.5, 0.5, 0.0],
                [-2.0, -0.5, 0.4, 1.6, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        loss = _ordinal_boundary_loss_from_logits(
            logits=logits,
            targets=targets,
            classes="0,1,2,3",
            threshold_weights="1.25,1.25,1.0",
            temperature=1.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(logits.grad)

    def test_ordinal_boundary_loss_ignores_non_ordinal_targets(self):
        logits = torch.zeros(3, 5, dtype=torch.float32, requires_grad=True)
        loss = _ordinal_boundary_loss_from_logits(
            logits=logits,
            targets=torch.tensor([4, 4, 4], dtype=torch.long),
            classes="0,1,2,3",
            threshold_weights="1.0",
            temperature=1.0,
        )

        self.assertEqual(float(loss.item()), 0.0)

    def test_pairwise_confusion_loss_is_finite_and_backpropagates(self):
        model = create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="vit_registers",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
            ),
        )
        images = torch.randn(4, 3, 64, 64)
        features = model.forward_features(images)
        logits = classification_logits_from_features(model, features)
        loss = _pairwise_confusion_loss_from_features(
            model=model,
            features=features,
            logits=logits,
            sources="head,patch,registers,logits",
            normalize=True,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(model.head.weight.grad)

    def test_mutual_channel_loss_is_finite_and_backpropagates(self):
        stem_features = torch.rand(6, 20, 4, 4, dtype=torch.float32, requires_grad=True)
        features = {"stem_features": stem_features}
        targets = torch.tensor([0, 1, 2, 3, 4, 1], dtype=torch.long)
        loss = _mutual_channel_loss_from_features(
            features=features,
            targets=targets,
            num_classes=5,
            top_k=3,
            diversity_weight=0.2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(stem_features.grad)

    def test_mutual_channel_loss_returns_zero_without_stem_features(self):
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        loss = _mutual_channel_loss_from_features(
            features={},
            targets=targets,
            num_classes=5,
            top_k=2,
            diversity_weight=0.2,
        )

        self.assertEqual(float(loss.item()), 0.0)

    def test_complement_entropy_loss_penalizes_collapsed_non_targets(self):
        collapsed_logits = torch.tensor(
            [[4.0, 3.5, -2.0, -2.0, -2.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        balanced_logits = torch.tensor(
            [[4.0, 0.2, 0.2, 0.2, 0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor([0], dtype=torch.long)

        collapsed_loss = _complement_entropy_loss_from_logits(
            logits=collapsed_logits,
            targets=targets,
            classes="all",
        )
        balanced_loss = _complement_entropy_loss_from_logits(
            logits=balanced_logits,
            targets=targets,
            classes="all",
        )
        collapsed_loss.backward()

        self.assertTrue(torch.isfinite(collapsed_loss).item())
        self.assertGreater(float(collapsed_loss.item()), float(balanced_loss.item()))
        self.assertIsNotNone(collapsed_logits.grad)

    def test_complement_entropy_loss_respects_target_class_filter(self):
        logits = torch.randn(3, 5, dtype=torch.float32)
        targets = torch.tensor([2, 3, 4], dtype=torch.long)
        loss = _complement_entropy_loss_from_logits(
            logits=logits,
            targets=targets,
            classes="0,1",
        )

        self.assertEqual(float(loss.item()), 0.0)

    def test_hybrid_model_accepts_classification_extension_config_keys(self):
        model = create_model(
            num_classes=3,
            model_config=ModelConfig(
                model_type="vit_registers_hybrid",
                image_size=64,
                patch_size=16,
                stem_channels=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                num_registers=2,
                num_queries=4,
                decoder_depth=1,
                decoder_num_heads=4,
                decoder_ffn_dim=64,
                color_stat_fusion=True,
                fine_grained_pooling=True,
            ),
        )
        output = model(torch.randn(2, 3, 64, 64))
        self.assertIn("logits", output)
        self.assertEqual(tuple(output["logits"].shape[:2]), (2, 4))

    def test_mixstyle_is_train_only_and_preserves_shape(self):
        mixstyle = model_module.MixStyle(probability=1.0, alpha=0.1)
        features = torch.zeros(4, 3, 8, 8, dtype=torch.float32)
        features[1] = 1.0
        features[2] = 2.0
        features[3] = 3.0

        torch.manual_seed(123)
        mixstyle.train()
        mixed = mixstyle(features)
        mixstyle.eval()
        restored = mixstyle(features)

        self.assertEqual(tuple(mixed.shape), tuple(features.shape))
        self.assertFalse(torch.allclose(mixed, features))
        self.assertTrue(torch.allclose(restored, features))

    def test_plot_dataset_color_audit_writes_png_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_paths = []
            train_labels = []
            val_paths = []
            val_labels = []
            for split_paths, split_labels, suffix in (
                (train_paths, train_labels, "train"),
                (val_paths, val_labels, "val"),
            ):
                red = root / f"red_{suffix}.png"
                green = root / f"green_{suffix}.png"
                Image.new("RGB", (16, 16), (220, 30, 30)).save(red)
                Image.new("RGB", (16, 16), (30, 180, 40)).save(green)
                split_paths.extend([red, green])
                split_labels.extend([0, 1])
            summary = plot_dataset_color_audit(
                class_names=["red", "green"],
                train_paths=train_paths,
                train_labels=train_labels,
                val_paths=val_paths,
                val_labels=val_labels,
                output_path=root / "color_audit.png",
                summary_path=root / "color_audit.json",
                max_images_per_class=2,
                max_pixels_per_image=32,
            )
            self.assertTrue((root / "color_audit.png").is_file())
            self.assertTrue((root / "color_audit.json").is_file())
            self.assertEqual(summary["classes"]["0"]["train"]["sampled_images"], 1)
            self.assertGreater(summary["classes"]["0"]["train"]["mean_rgb"][0], 0.5)

    def test_classification_dataset_target_feeds_vit_registers_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            Image.new("RGB", (96, 96), color=(128, 200, 64)).save(images_dir / "sample.jpg")
            (labels_dir / "sample.txt").write_text("1 0.5 0.5 0.6 0.6\n", encoding="utf-8")

            dataset = MangoYOLOCropDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                transform=build_eval_transform(image_size=64, resize_mode="pad"),
                crop_to_primary_object=True,
                num_classes=5,
                classification_target=True,
            )
            sample = dataset[0]
            image, label, metadata = sample
            self.assertEqual(label, 1)
            self.assertIn("image_mask", metadata)

            images, targets, batch_metadata = build_train_collate_fn(
                num_classes=5,
                batch_mix_probability=0.0,
            )(
                [sample, sample]
            )
            self.assertEqual(tuple(targets.shape), (2,))
            self.assertEqual(targets.tolist(), [1, 1])
            self.assertEqual(tuple(batch_metadata["image_mask"].shape), (2, 64, 64))

            model = create_model(
                num_classes=5,
                model_config={
                    "model_type": "vit_registers",
                    "image_size": 64,
                    "patch_size": 16,
                    "embed_dim": 64,
                    "depth": 1,
                    "num_heads": 4,
                    "num_registers": 2,
                },
            )
            criterion = LDAMFocalLoss(class_counts=[1, 2, 1, 1, 1])
            loss, *_ = _forward_train_loss(
                model=model,
                criterion=criterion,
                images=images,
                labels=None,
                targets=targets,
                device=torch.device("cpu"),
                amp=False,
            )
            self.assertTrue(torch.isfinite(loss))

    def test_eval_transform_can_normalize_illumination_and_suppress_background(self):
        image = Image.new("RGB", (96, 72), (25, 95, 180))
        image.paste((235, 235, 235), (0, 0, 24, 72))
        image.paste((24, 24, 24), (72, 0, 96, 72))
        image.paste((225, 175, 42), (30, 20, 66, 58))

        plain_transform = build_eval_transform(image_size=64, resize_mode="pad")
        filtered_transform = build_eval_transform(
            image_size=64,
            resize_mode="pad",
            illumination_normalization=True,
            illumination_normalization_strength=0.35,
            background_suppression_mode="desaturate_blur",
            background_suppression_margin=0.05,
            background_suppression_blur_radius=5.0,
        )

        plain = plain_transform(image)
        filtered, meta = filtered_transform(image, return_meta=True)

        self.assertEqual(tuple(filtered.shape), (3, 64, 64))
        self.assertEqual(meta["mode"], "pad")
        self.assertGreater(float(torch.mean(torch.abs(filtered - plain)).item()), 1e-3)

    def test_pseudo_foreground_mask_excludes_imagenet_padding(self):
        image = Image.new("RGB", (64, 64), (124, 116, 104))
        image.paste((220, 180, 38), (10, 20, 54, 44))
        mask = _pseudo_foreground_mask_array(image, margin=0.05)

        self.assertLess(float(mask[:12].mean()), 0.05)
        self.assertLess(float(mask[52:].mean()), 0.05)
        self.assertGreater(float(mask[24:40, 18:46].mean()), 0.80)

    def test_grabcut_foreground_mask_removes_border_background(self):
        if importlib.util.find_spec("cv2") is None:
            self.skipTest("OpenCV is not installed")
        image = Image.new("RGB", (96, 72), (70, 150, 72))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 12, 78, 62), fill=(170, 205, 70))
        draw.ellipse((44, 30, 58, 44), fill=(95, 120, 36))

        mask = _grabcut_foreground_mask_array(image, margin=0.05)

        self.assertLess(float(mask[:, :8].mean()), 0.20)
        self.assertLess(float(mask[:, -8:].mean()), 0.20)
        self.assertGreater(float(mask[26:48, 36:62].mean()), 0.65)
        self.assertLess(float(mask.mean()), 0.80)

    def test_training_pseudo_foreground_mask_excludes_imagenet_padding(self):
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        rgb = mean.expand(1, 3, 64, 64).clone()
        rgb[:, :, 20:44, 10:54] = torch.tensor([220 / 255, 180 / 255, 38 / 255]).view(1, 3, 1, 1)
        images = (rgb - mean) / std
        mask = _pseudo_foreground_mask_from_normalized_images(images, margin=0.05)

        self.assertLess(float(mask[:, :, :12].mean().item()), 0.05)
        self.assertLess(float(mask[:, :, 52:].mean().item()), 0.05)
        self.assertGreater(float(mask[:, :, 24:40, 18:46].mean().item()), 0.80)

    def test_soft_target_class_weights_affect_hard_label_loss(self):
        logits = torch.zeros(2, 2)
        targets = torch.tensor([0, 1], dtype=torch.long)
        unweighted = FocalCrossEntropyLoss(gamma=0.0, focal_mix=0.0)(logits, targets)
        weighted = FocalCrossEntropyLoss(
            weight=torch.tensor([1.0, 3.0]),
            gamma=0.0,
            focal_mix=0.0,
        )(logits, targets)
        criterion = FocalCrossEntropyLoss(
            weight=torch.tensor([1.0, 3.0]),
            gamma=0.0,
            focal_mix=0.0,
        )
        criterion.set_class_weight_multipliers(torch.tensor([1.0, 2.0]))
        boosted = criterion(logits, targets)

        self.assertGreater(float(weighted.item()), float(unweighted.item()))
        self.assertGreater(float(boosted.item()), float(weighted.item()))

    def test_rare_class_recall_guard_activates_for_classification_only(self):
        guard = _resolve_rare_class_recall_guard(
            train_config=TrainConfig(
                rare_class_recall_guard=True,
                rare_class_recall_target=0.9,
                rare_class_recall_guard_scale_threshold=1.5,
                rare_class_recall_guard_max_multiplier=2.0,
                rare_class_recall_guard_min_precision=0.5,
            ),
            detection_mode=False,
            stage_name="classification",
            previous_val_metrics={
                "per_class": [
                    {"class_index": 0, "precision": 0.95, "recall": 0.95, "support": 20},
                    {"class_index": 1, "precision": 0.8, "recall": 0.72, "support": 10},
                ]
            },
            class_target_scales=[1.0, 2.0],
        )

        self.assertTrue(guard["active"])
        self.assertEqual(guard["active_class_count"], 1)
        self.assertGreater(guard["multipliers"][1], 1.0)

    def test_balanced_macro_f1_selection_penalizes_large_class_gap(self):
        metric_name, score, higher_is_better = _resolve_checkpoint_selection(
            model_config=ModelConfig(model_type="vit_registers"),
            train_config=TrainConfig(best_metric="balanced_macro_f1"),
            metrics={
                "macro_f1": 0.92,
                "per_class": [
                    {"class_index": 0, "f1": 0.98},
                    {"class_index": 1, "f1": 0.78},
                    {"class_index": 2, "f1": 0.98},
                ],
            },
        )

        self.assertEqual(metric_name, "balanced_macro_f1_gap_penalty")
        self.assertTrue(higher_is_better)
        self.assertLess(score, 0.92)

    def test_fair_macro_f1_selection_uses_min_class_and_gap(self):
        metric_name, score, higher_is_better = _resolve_checkpoint_selection(
            model_config=ModelConfig(model_type="vit_registers"),
            train_config=TrainConfig(
                best_metric="fair_macro_f1",
                fair_f1_gap_target=0.05,
                fair_f1_gap_penalty=1.5,
                fair_f1_min_weight=0.25,
            ),
            metrics={
                "macro_f1": 0.92,
                "per_class": [
                    {"class_index": 0, "f1": 0.98},
                    {"class_index": 1, "f1": 0.78},
                    {"class_index": 2, "f1": 0.98},
                ],
            },
        )

        self.assertEqual(metric_name, "fair_macro_f1_min_class_gap_penalty")
        self.assertTrue(higher_is_better)
        self.assertLess(score, 0.92)

    def test_loss_aware_fair_macro_f1_penalizes_overfit_checkpoint(self):
        train_config = TrainConfig(
            best_metric="loss_aware_fair_macro_f1",
            fair_f1_gap_target=0.35,
            fair_f1_gap_penalty=0.35,
            fair_f1_min_weight=0.20,
            fair_f1_loss_weight=0.20,
        )
        model_config = ModelConfig(model_type="vit_registers")
        stable_name, stable_score, stable_higher = _resolve_checkpoint_selection(
            model_config=model_config,
            train_config=train_config,
            metrics={
                "macro_f1": 0.682,
                "loss": 0.715,
                "per_class": [
                    {"class_index": 0, "f1": 0.80},
                    {"class_index": 1, "f1": 0.19},
                    {"class_index": 2, "f1": 0.76},
                    {"class_index": 3, "f1": 0.82},
                    {"class_index": 4, "f1": 0.84},
                ],
            },
        )
        overfit_name, overfit_score, overfit_higher = _resolve_checkpoint_selection(
            model_config=model_config,
            train_config=train_config,
            metrics={
                "macro_f1": 0.657,
                "loss": 1.251,
                "per_class": [
                    {"class_index": 0, "f1": 0.79},
                    {"class_index": 1, "f1": 0.26},
                    {"class_index": 2, "f1": 0.74},
                    {"class_index": 3, "f1": 0.71},
                    {"class_index": 4, "f1": 0.77},
                ],
            },
        )

        self.assertEqual(stable_name, "loss_aware_fair_macro_f1_min_class_gap_loss_penalty")
        self.assertEqual(overfit_name, stable_name)
        self.assertTrue(stable_higher)
        self.assertTrue(overfit_higher)
        self.assertGreater(stable_score, overfit_score)

    def test_supervised_contrastive_loss_is_finite_with_positive_pairs(self):
        criterion = SupervisedContrastiveLoss(temperature=0.2)
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        targets = torch.tensor([0, 0, 1, 1])
        loss = criterion(embeddings, targets)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

    def test_boundary_contrastive_loss_is_finite_and_pair_limited(self):
        pairs = _parse_boundary_contrastive_pairs("0-1,1-2,4-rest", num_classes=5)
        self.assertEqual(pairs, [(0, 1), (1, 2), (-1, 4)])
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.0, 1.0],
                [0.1, 0.8],
            ],
            dtype=torch.float32,
        )
        features = {
            "pooled": embeddings,
            "patches": embeddings.unsqueeze(1),
        }
        targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        loss, stats = _boundary_contrastive_loss_from_features(
            model=nn.Module(),
            features=features,
            targets=targets,
            pairs="0-1,1-2",
            sources="head,patch",
            margin=0.12,
            temperature=0.20,
            max_pairs=2,
            num_classes=5,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(stats["used_sources"], ["head", "patch"])
        self.assertGreater(int(stats["terms"]), 0)
        self.assertLessEqual(int(stats["terms"]), 16)

    def test_balanced_softmax_loss_is_finite_and_backpropagates(self):
        logits = torch.zeros(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        criterion = BalancedSoftmaxFocalLoss(
            class_counts=[2, 1, 6],
            gamma=1.5,
            focal_mix=0.2,
            label_smoothing=0.01,
            prior_tau=1.0,
        )
        loss = criterion(logits, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_classification_folder_data_spec_preserves_yaml_class_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "val", "test"):
                for class_name in ("class_b", "class_a"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    Image.new("RGB", (32, 24), color=(120, 80, 40)).save(
                        class_dir / f"{split}_{class_name}.jpg"
                    )
            (root / "data.yaml").write_text(
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
                        "  0: class_a",
                        "  1: class_b",
                    ]
                ),
                encoding="utf-8",
            )

            data_spec = load_data_spec(root / "data.yaml", class_name_mode="raw", expected_num_classes=2)
            self.assertEqual(data_spec.data_format, "classification_folder")
            self.assertEqual(data_spec.class_names, ["class_a", "class_b"])

            dataset = ClassificationFolderDataset.from_data_spec(
                data_spec,
                split="train",
                transform=build_eval_transform(image_size=32, resize_mode="pad"),
            )
            self.assertEqual(dataset.class_counts(data_spec.num_classes), [1, 1])
            images, targets, metadata = build_train_collate_fn(
                num_classes=2,
                batch_mix_probability=0.0,
            )(
                [dataset[0], dataset[1]]
            )
            self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
            self.assertEqual(tuple(targets.shape), (2,))
            self.assertEqual(targets.tolist(), [0, 1])
            self.assertEqual(tuple(metadata["image_mask"].shape), (2, 32, 32))

    def test_build_classification_merged_classes_preserves_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            class_names = ["raw0", "raw1", "raw2", "raw3", "raw4"]
            for split in ("train", "val", "test"):
                for class_index, class_name in enumerate(class_names):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    Image.new(
                        "RGB",
                        (12, 10),
                        color=(20 + class_index * 20, 40, 80),
                    ).save(class_dir / f"{split}_{class_name}.jpg")
            (root / "data.yaml").write_text(
                "\n".join(
                    [
                        "format: classification_folder",
                        "path: .",
                        "train: train",
                        "val: val",
                        "test: test",
                        "nc: 5",
                        "class_name_mode: raw",
                        "names:",
                        "  0: raw0",
                        "  1: raw1",
                        "  2: raw2",
                        "  3: raw3",
                        "  4: raw4",
                    ]
                ),
                encoding="utf-8",
            )

            output_dir = Path(tmpdir) / "merged"
            summary = build_merged_classification_dataset(
                root / "data.yaml",
                output_dir,
                merge_classes=[0, 1],
                merged_name="raw0_or_raw1",
                class_name_mode="raw",
                expected_num_classes=5,
                link_mode="copy",
                overwrite=True,
            )

            self.assertEqual(summary["num_classes"], 4)
            self.assertEqual(summary["old_to_new"], {"0": 0, "1": 0, "2": 1, "3": 2, "4": 3})
            data_spec = load_data_spec(output_dir / "data.yaml", class_name_mode="raw", expected_num_classes=4)
            self.assertEqual(data_spec.class_names, ["raw0_or_raw1", "raw2", "raw3", "raw4"])
            self.assertEqual(data_spec.balance.class_counts(4), [6, 3, 3, 3])

            dataset = ClassificationFolderDataset.from_data_spec(data_spec, split="train")
            self.assertEqual(dataset.class_counts(4), [2, 1, 1, 1])
            with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(len(manifest_rows), 15)
            self.assertEqual(
                Counter(row["new_class_name"] for row in manifest_rows)["raw0_or_raw1"],
                6,
            )

    def test_filter_classification_single_source_removes_multi_box_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "src"
            class_names = ["raw0", "raw1"]
            image_specs = [
                ("train", 0, "Image_1_box000.jpg"),
                ("train", 0, "Image_2_box000.jpg"),
                ("train", 1, "Image_2_box001.jpg"),
                ("val", 1, "Image_3_box000.jpg"),
                ("test", 1, "Image_4_box000.jpg"),
            ]
            for split in ("train", "val", "test"):
                for class_name in class_names:
                    (root / split / class_name).mkdir(parents=True, exist_ok=True)
            manifest_rows = []
            for split, class_id, filename in image_specs:
                class_name = class_names[class_id]
                path = root / split / class_name / filename
                Image.new("RGB", (12, 10), color=(20 + class_id * 60, 40, 80)).save(path)
                source_id = filename.split("_box")[0]
                manifest_rows.append(
                    {
                        "split": split,
                        "original_split": split,
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_id": source_id,
                        "image_number": source_id.replace("Image_", ""),
                        "source_path": str(path),
                        "output_path": str(path),
                    }
                )
            (root / "data.yaml").write_text(
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
                        "  0: raw0",
                        "  1: raw1",
                    ]
                ),
                encoding="utf-8",
            )
            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)

            output_dir = Path(tmpdir) / "single"
            report = build_single_source_classification_dataset(
                root / "data.yaml",
                output_dir,
                class_name_mode="raw",
                expected_num_classes=2,
                link_mode="copy",
                overwrite=True,
            )

            self.assertEqual(report["kept_rows"], 3)
            self.assertEqual(report["removed_rows"], 2)
            self.assertTrue((output_dir / "train" / "raw0" / "Image_1_box000.jpg").is_file())
            self.assertFalse((output_dir / "train" / "raw0" / "Image_2_box000.jpg").exists())
            self.assertFalse((output_dir / "train" / "raw1" / "Image_2_box001.jpg").exists())
            data_spec = load_data_spec(output_dir / "data.yaml", class_name_mode="raw", expected_num_classes=2)
            self.assertEqual(data_spec.balance.class_counts(2), [1, 2])

    def test_hard_sample_repeat_dataset_repeats_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for class_name in ("class_a", "class_b"):
                (root / class_name).mkdir(parents=True)
            hard_image = root / "class_a" / "hard.jpg"
            easy_image = root / "class_b" / "easy.jpg"
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(hard_image)
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(easy_image)

            dataset = ClassificationFolderDataset(root, ["class_a", "class_b"])
            repeated = HardSampleRepeatDataset(dataset, [hard_image], repeat_factor=3.0, seed=1)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(repeated), 4)
            self.assertEqual(repeated.labels().count(0), 3)
            self.assertEqual(repeated.labels().count(1), 1)

    def test_save_evaluation_artifacts_writes_predictions_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            class_names = ["class_a", "class_b"]
            targets = torch.tensor([0, 1], dtype=torch.long)
            predictions = torch.tensor([0, 0], dtype=torch.long)
            probabilities = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32)
            metrics = build_metrics(
                targets=targets,
                predictions=predictions,
                class_names=class_names,
                probabilities=probabilities,
            )
            metrics["prediction_records"] = _build_prediction_records(
                targets=targets,
                predictions=predictions,
                probabilities=probabilities,
                class_names=class_names,
                sample_paths=["a.jpg", "b.jpg"],
            )
            save_evaluation_artifacts(metrics, class_names, output_dir)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "predictions.csv").exists())
            with (output_dir / "predictions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["image_path"], "a.jpg")
            self.assertEqual(rows[1]["correct"], "0")

    def test_render_prediction_image_writes_classification_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.jpg"
            output_path = root / "overlay.jpg"
            Image.new("RGB", (96, 72), color=(120, 180, 80)).save(input_path)
            result = {
                "top_prediction": {
                    "class_index": 1,
                    "class_name": "class_b",
                    "probability": 0.72,
                },
                "predictions": [
                    {"class_index": 1, "class_name": "class_b", "probability": 0.72},
                    {"class_index": 0, "class_name": "class_a", "probability": 0.28},
                ],
                "prediction_status": "accepted",
            }

            render_prediction_image(
                image_path=input_path,
                result=result,
                output_path=output_path,
                overlay_top_k=2,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_per_class_validation_metric_writes_f1_and_accuracy_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = root / "history.csv"
            f1_path = root / "per_class_val_f1.png"
            acc_path = root / "per_class_val_accuracy.png"
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "val_class_0_f1",
                        "val_class_0_recall",
                        "val_class_1_f1",
                        "val_class_1_recall",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "val_class_0_f1": 0.91,
                        "val_class_0_recall": 0.92,
                        "val_class_1_f1": 0.74,
                        "val_class_1_recall": 0.8,
                    }
                )
                writer.writerow(
                    {
                        "epoch": 2,
                        "val_class_0_f1": 0.94,
                        "val_class_0_recall": 0.95,
                        "val_class_1_f1": 0.79,
                        "val_class_1_recall": 0.84,
                    }
                )

            plot_per_class_validation_metric(
                history_path,
                ["class_a", "class_b"],
                f1_path,
                metric="f1",
                title="Per-Class Validation F1",
                ylabel="F1",
                target=0.97,
            )
            plot_per_class_validation_metric(
                history_path,
                ["class_a", "class_b"],
                acc_path,
                metric="accuracy",
                title="Per-Class Validation Accuracy",
                ylabel="Accuracy / TP-rate",
                target=0.97,
            )

            self.assertTrue(f1_path.exists())
            self.assertGreater(f1_path.stat().st_size, 0)
            self.assertTrue(acc_path.exists())
            self.assertGreater(acc_path.stat().st_size, 0)

    def test_plot_train_val_final_test_metrics_writes_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = root / "history.csv"
            output_path = root / "train_val_final_test_curves.png"
            metrics_path = root / "final_test" / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "learning_rate",
                        "train_loss",
                        "val_loss",
                        "val_accuracy",
                        "val_macro_f1",
                        "val_weighted_f1",
                        "val_class_0_f1",
                        "val_class_1_f1",
                        "epoch_seconds",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "learning_rate": 1e-4,
                        "train_loss": 1.2,
                        "val_loss": 1.0,
                        "val_accuracy": 0.6,
                        "val_macro_f1": 0.55,
                        "val_weighted_f1": 0.58,
                        "val_class_0_f1": 0.7,
                        "val_class_1_f1": 0.4,
                        "epoch_seconds": 12.0,
                    }
                )
                writer.writerow(
                    {
                        "epoch": 2,
                        "learning_rate": 8e-5,
                        "train_loss": 0.9,
                        "val_loss": 0.8,
                        "val_accuracy": 0.72,
                        "val_macro_f1": 0.68,
                        "val_weighted_f1": 0.7,
                        "val_class_0_f1": 0.8,
                        "val_class_1_f1": 0.56,
                        "epoch_seconds": 11.0,
                    }
                )
            metrics_path.write_text(
                json.dumps(
                    {
                        "accuracy": 0.74,
                        "macro_f1": 0.69,
                        "weighted_f1": 0.71,
                        "per_class": [
                            {"class_index": 0, "f1": 0.81},
                            {"class_index": 1, "f1": 0.58},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            plot_train_val_final_test_metrics(
                history_csv=history_path,
                final_test_metrics_json=metrics_path,
                output_path=output_path,
                class_names=["class_a", "class_b"],
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_evaluation_artifacts_writes_baseline_comparison_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            class_names = ["class_a", "class_b"]
            targets = torch.tensor([0, 1], dtype=torch.long)
            predictions = torch.tensor([0, 0], dtype=torch.long)
            probabilities = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32)
            metrics = build_metrics(
                targets=targets,
                predictions=predictions,
                class_names=class_names,
                probabilities=probabilities,
            )
            metrics["loss"] = 0.5
            metrics["prediction_records"] = _build_prediction_records(
                targets=targets,
                predictions=predictions,
                probabilities=probabilities,
                class_names=class_names,
                sample_paths=["a.jpg", "b.jpg"],
            )
            save_evaluation_artifacts(
                metrics,
                class_names,
                output_dir,
                comparison_summary={
                    "paper_name": "TRKH-Test",
                    "family": "TRKH",
                    "backend": "trkh",
                    "model": "vit_registers",
                    "pretrained": False,
                    "test_size": 2,
                    "test_loss": 0.5,
                    "metrics": {
                        "accuracy": 0.5,
                        "macro_precision": 0.25,
                        "macro_recall": 0.5,
                        "macro_f1": 0.3333333333,
                    },
                    "ci95": {},
                    "params": 123,
                    "inference_time_ms_per_image": 1.0,
                    "best_epoch": 1,
                    "classes": class_names,
                },
                comparison_prediction_rows=[
                    {
                        "path": "a.jpg",
                        "y_true": 0,
                        "y_pred": 0,
                        "true_name": "class_a",
                        "pred_name": "class_a",
                    },
                    {
                        "path": "b.jpg",
                        "y_true": 1,
                        "y_pred": 0,
                        "true_name": "class_b",
                        "pred_name": "class_a",
                    },
                ],
            )
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "metrics_detailed.json").exists())
            self.assertTrue((output_dir / "predictions.csv").exists())
            self.assertTrue((output_dir / "predictions_detailed.csv").exists())
            with (output_dir / "predictions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0].keys()),
                ["path", "y_true", "y_pred", "true_name", "pred_name"],
            )
            self.assertEqual(rows[1]["pred_name"], "class_a")

    def test_classification_object_crops_expand_multi_object_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            Image.new("RGB", (128, 96), color=(100, 180, 80)).save(images_dir / "multi.jpg")
            (labels_dir / "multi.txt").write_text(
                "1 0.25 0.50 0.30 0.60\n"
                "3 0.75 0.50 0.25 0.55\n",
                encoding="utf-8",
            )

            dataset = MangoYOLOCropDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                transform=build_eval_transform(image_size=64, resize_mode="pad"),
                crop_to_primary_object=True,
                num_classes=5,
                classification_target=True,
                classification_object_crops=True,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.labels(), [1, 3])
            self.assertEqual(dataset.class_counts(5), [0, 1, 0, 1, 0])
            self.assertEqual(dataset.quality_report()["ignored_object_count"], 0)

            _, first_label, first_metadata = dataset[0]
            _, second_label, second_metadata = dataset[1]
            self.assertEqual([first_label, second_label], [1, 3])
            self.assertIn("image_mask", first_metadata)
            self.assertIn("image_mask", second_metadata)

            repeated = RareClassRepeatDataset(
                dataset,
                class_repeat_factors=[1.0, 1.0, 1.0, 2.0, 1.0],
                seed=42,
            )
            self.assertIn(3, repeated.labels())

    def test_auto_class_crop_margin_targets_scaled_classes_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            Image.new("RGB", (100, 100), color=(100, 180, 80)).save(images_dir / "common.jpg")
            Image.new("RGB", (100, 100), color=(120, 180, 80)).save(images_dir / "rare.jpg")
            (labels_dir / "common.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (labels_dir / "rare.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            dataset = MangoYOLOCropDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                transform=None,
                crop_margin_ratio=0.05,
                crop_to_primary_object=True,
                num_classes=2,
                classification_target=True,
                class_crop_margin_scale_threshold=1.5,
                class_crop_margin_max_ratio=0.20,
                class_crop_margin_scales=[1.0, 2.0],
            )

            common_image, common_label = dataset[0]
            rare_image, rare_label = dataset[1]
            self.assertEqual([common_label, rare_label], [0, 1])
            self.assertGreater(rare_image.size[0], common_image.size[0])
            self.assertGreater(rare_image.size[1], common_image.size[1])

            crop_report = dataset.quality_report()["class_crop_margin"]
            self.assertEqual(crop_report["target_classes"], [1])
            self.assertAlmostEqual(crop_report["effective_ratios"][0], 0.05)
            self.assertAlmostEqual(crop_report["effective_ratios"][1], 0.10)

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

    def test_canbang_yaml_reports_balanced_sampling_without_mutating_augmentation(self):
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
        self.assertGreater(
            summary["balanced_sampling_factors"][1],
            summary["balanced_sampling_factors"][0],
        )
        self.assertEqual(summary["auto_repeat_factors"], [1.0, 1.0, 1.0])
        self.assertFalse(augmentation_config.class_aware_augmentation)
        self.assertFalse(augmentation_config.rare_class_repeat)
        self.assertTrue(augmentation_config.class_aware_photometric_augmentation)
        self.assertFalse(train_config.auto_tune_imbalance)
        self.assertFalse(summary["imbalance_auto_tune"])

    def test_canbang_yaml_prefers_train_split_counts_to_avoid_val_test_leakage(self):
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
      count: 450
      ratio: 0.45
      percent: 45.0
    "1":
      count: 50
      ratio: 0.05
      percent: 5.0
    "2":
      count: 500
      ratio: 0.50
      percent: 50.0
  splits:
    train:
      total_images: 700
      classes:
        "0": 300
        "1": 100
        "2": 300
""",
                encoding="utf-8",
            )

            spec = load_data_spec(data_yaml_path)
            train_config = TrainConfig(
                auto_tune_imbalance=False,
                balance_auto_max_repeat_factor=1.8,
            )
            augmentation_config = AugmentationConfig()
            summary = apply_balance_file_auto_adjustment(
                data_spec=spec,
                train_config=train_config,
                augmentation_config=augmentation_config,
            )

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["count_scope"], "train")
        self.assertEqual(summary["class_counts"], [300, 100, 300])
        self.assertEqual(summary["total_pairs"], 700)
        self.assertGreater(summary["balanced_sampling_factors"][1], 1.0)
        self.assertEqual(summary["raw_auto_repeat_factors"], [1.0, 1.0, 1.0])

    def test_foreground_consistency_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(13)
        images = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
        images[:, :, 8:24, 8:24] = 1.0
        patches = torch.randn((2, 16, 8), dtype=torch.float32, requires_grad=True)
        features = {"patches": patches, "grid_size": (4, 4)}

        mask = _pseudo_foreground_mask_from_normalized_images(images, margin=0.02)
        loss = _foreground_consistency_loss_from_features(
            images=images,
            features=features,
            margin=0.02,
        )
        loss.backward()

        self.assertEqual(tuple(mask.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(patches.grad)
        self.assertTrue(torch.isfinite(patches.grad).all())

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

    def test_detection_copy_paste_collate_adds_valid_objects(self):
        torch.manual_seed(11)
        collate = build_train_collate_fn(
            num_classes=5,
            batch_mix_probability=1.0,
            mosaic_probability=0.0,
            cutmix_probability=0.0,
            copy_paste_probability=1.0,
            copy_paste_max_objects=2,
            max_detection_objects=8,
        )
        batch = []
        for index in range(4):
            image = torch.zeros((3, 48, 48), dtype=torch.float32)
            image[:, 10 + index : 26 + index, 12 + index : 28 + index] = 0.2 + 0.1 * index
            target = {
                "labels": torch.tensor([index % 5], dtype=torch.long),
                "boxes": torch.tensor([[0.42, 0.40, 0.28, 0.28]], dtype=torch.float32),
            }
            batch.append((image, target))

        images, targets = collate(batch)

        self.assertEqual(tuple(images.shape), (4, 3, 48, 48))
        self.assertTrue(any(target["labels"].numel() > 1 for target in targets))
        for target in targets:
            self.assertEqual(target["boxes"].shape[-1], 4)
            self.assertTrue(torch.all(target["boxes"] >= 0.0))
            self.assertTrue(torch.all(target["boxes"] <= 1.0))

    def test_targeted_copy_paste_prefers_scaled_object_class(self):
        torch.manual_seed(7)
        images = torch.zeros((2, 3, 48, 48), dtype=torch.float32)
        images[1, :, 8:18, 8:18] = 0.5
        images[1, :, 28:38, 28:38] = 0.9
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.long),
                "boxes": torch.tensor([[0.75, 0.75, 0.16, 0.16]], dtype=torch.float32),
            },
            {
                "labels": torch.tensor([1, 2], dtype=torch.long),
                "boxes": torch.tensor(
                    [
                        [0.27, 0.27, 0.18, 0.18],
                        [0.69, 0.69, 0.18, 0.18],
                    ],
                    dtype=torch.float32,
                ),
            },
        ]

        _, pasted_targets = _apply_copypaste_detection_batch(
            images,
            targets,
            max_objects=8,
            max_paste_objects=1,
            source_weights=torch.tensor([0.0, 1.0], dtype=torch.float32),
            target_class_scales=torch.tensor([1.0, 2.2, 1.0], dtype=torch.float32),
            target_scale_threshold=1.5,
            target_probability=1.0,
        )

        pasted_labels = pasted_targets[0]["labels"].tolist()
        self.assertIn(1, pasted_labels)
        self.assertNotIn(2, pasted_labels)

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

    def test_strict_balanced_sampler_rotates_remainder_across_the_epoch(self):
        labels = [0] * 1941 + [1] * 541 + [2] * 1920 + [3] * 2520 + [4] * 2293
        sampler = StrictBalancedBatchSampler(
            labels=labels,
            batch_size=32,
            num_classes=5,
            epoch_multiplier=1.0,
            seed=42,
        )
        summary = sampler.exposure_summary()
        self.assertEqual(summary["total_samples"], 9216)
        self.assertLessEqual(
            summary["max_class_exposure"] - summary["min_class_exposure"],
            1,
        )
        self.assertEqual(len(summary["class_exposure_counts"]), 5)

        first_epoch = list(iter(sampler))
        repeated_epoch = list(iter(sampler))
        self.assertEqual(first_epoch, repeated_epoch)
        self.assertEqual(len(first_epoch), summary["num_batches"])
        for batch_indices in first_epoch:
            batch_counts = [0] * 5
            for sample_index in batch_indices:
                batch_counts[labels[sample_index]] += 1
            self.assertTrue(all(count > 0 for count in batch_counts))
            self.assertLessEqual(max(batch_counts) - min(batch_counts), 1)

    def test_class_aware_scale_does_not_amplify_photometric_by_default(self):
        transform = build_train_transform(
            image_size=32,
            brightness=0.2,
            contrast=0.2,
            lighting_probability=0.15,
            scale_photometric_with_augmentation=False,
        )

        self.assertFalse(transform.scale_photometric_with_augmentation)

    def test_foreground_background_mix_preserves_foreground_and_swaps_background(self):
        rgb0 = torch.full((3, 32, 32), 0.18, dtype=torch.float32)
        rgb1 = torch.full((3, 32, 32), 0.76, dtype=torch.float32)
        rgb0[:, 9:23, 9:23] = torch.tensor([0.28, 0.76, 0.12]).view(3, 1, 1)
        rgb1[:, 9:23, 9:23] = torch.tensor([0.80, 0.68, 0.10]).view(3, 1, 1)
        images = torch.stack(
            [self._normalized_rgb_tensor(rgb0), self._normalized_rgb_tensor(rgb1)],
            dim=0,
        )

        mask = _pseudo_foreground_mask_from_tensor_images(
            images,
            margin=0.05,
            min_fraction=0.04,
            max_fraction=0.90,
        )
        mixed = _apply_foreground_background_mix_batch(
            images,
            probability=1.0,
            margin=0.05,
            min_foreground_fraction=0.04,
            max_foreground_fraction=0.90,
            softness=0.0,
        )

        self.assertGreater(float(mask[:, :, 12:20, 12:20].float().mean().item()), 0.90)
        self.assertLess(float(mask[:, :, :4, :4].float().mean().item()), 0.05)
        self.assertTrue(torch.allclose(mixed[0, :, 12:20, 12:20], images[0, :, 12:20, 12:20]))
        self.assertTrue(torch.allclose(mixed[0, :, :4, :4], images[1, :, :4, :4]))

    def test_foreground_background_mix_can_use_bbox_mask(self):
        rgb0 = torch.full((3, 32, 32), 0.18, dtype=torch.float32)
        rgb1 = torch.full((3, 32, 32), 0.76, dtype=torch.float32)
        rgb0[:, 8:24, 8:24] = torch.tensor([0.28, 0.76, 0.12]).view(3, 1, 1)
        rgb1[:, 8:24, 8:24] = torch.tensor([0.80, 0.68, 0.10]).view(3, 1, 1)
        images = torch.stack(
            [self._normalized_rgb_tensor(rgb0), self._normalized_rgb_tensor(rgb1)],
            dim=0,
        )
        bboxes = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        mixed = _apply_foreground_background_mix_batch(
            images,
            probability=1.0,
            margin=0.0,
            min_foreground_fraction=0.05,
            max_foreground_fraction=0.90,
            softness=0.0,
            mask_source="bbox",
            bboxes=bboxes,
        )

        self.assertTrue(torch.allclose(mixed[0, :, 12:20, 12:20], images[0, :, 12:20, 12:20]))
        self.assertTrue(torch.allclose(mixed[0, :, :4, :4], images[1, :, :4, :4]))

    def test_foreground_background_mix_collate_preserves_metadata_labels(self):
        rgb0 = torch.full((3, 32, 32), 0.18, dtype=torch.float32)
        rgb1 = torch.full((3, 32, 32), 0.76, dtype=torch.float32)
        rgb0[:, 9:23, 9:23] = torch.tensor([0.28, 0.76, 0.12]).view(3, 1, 1)
        rgb1[:, 9:23, 9:23] = torch.tensor([0.80, 0.68, 0.10]).view(3, 1, 1)
        image0 = self._normalized_rgb_tensor(rgb0)
        image1 = self._normalized_rgb_tensor(rgb1)
        collate = build_train_collate_fn(
            num_classes=2,
            batch_mix_probability=0.0,
            foreground_background_mix_probability=1.0,
            foreground_background_mix_margin=0.05,
            foreground_background_mix_min_foreground_fraction=0.04,
            foreground_background_mix_max_foreground_fraction=0.90,
            foreground_background_mix_softness=0.0,
        )

        images, labels, metadata = collate(
            [
                (image0, 0, {"sample_index": 10, "sample_weight": 1.25}),
                (image1, 1, {"sample_index": 11, "sample_weight": 0.75}),
            ]
        )

        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(metadata["sample_index"].tolist(), [10, 11])
        self.assertTrue(torch.allclose(metadata["sample_weight"], torch.tensor([1.25, 0.75])))
        self.assertTrue(torch.allclose(images[0, :, 12:20, 12:20], image0[:, 12:20, 12:20]))
        self.assertTrue(torch.allclose(images[0, :, :4, :4], image1[:, :4, :4]))

    def test_foreground_background_mix_collate_can_use_crop_bbox_metadata(self):
        rgb0 = torch.full((3, 32, 32), 0.18, dtype=torch.float32)
        rgb1 = torch.full((3, 32, 32), 0.76, dtype=torch.float32)
        rgb0[:, 8:24, 8:24] = torch.tensor([0.28, 0.76, 0.12]).view(3, 1, 1)
        rgb1[:, 8:24, 8:24] = torch.tensor([0.80, 0.68, 0.10]).view(3, 1, 1)
        image0 = self._normalized_rgb_tensor(rgb0)
        image1 = self._normalized_rgb_tensor(rgb1)
        crop_bbox = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        full_bbox = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        collate = build_train_collate_fn(
            num_classes=2,
            batch_mix_probability=0.0,
            foreground_background_mix_probability=1.0,
            foreground_background_mix_margin=0.0,
            foreground_background_mix_min_foreground_fraction=0.05,
            foreground_background_mix_max_foreground_fraction=0.90,
            foreground_background_mix_softness=0.0,
            foreground_background_mix_mask_source="crop_bbox",
        )

        images, labels, metadata = collate(
            [
                (image0, 0, {"sample_index": 20, "bbox": full_bbox, "crop_bbox": crop_bbox}),
                (image1, 1, {"sample_index": 21, "bbox": full_bbox, "crop_bbox": crop_bbox}),
            ]
        )

        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(metadata["sample_index"].tolist(), [20, 21])
        self.assertTrue(torch.allclose(metadata["crop_bbox"], torch.stack([crop_bbox, crop_bbox])))
        self.assertTrue(torch.allclose(images[0, :, 12:20, 12:20], image0[:, 12:20, 12:20]))
        self.assertTrue(torch.allclose(images[0, :, :4, :4], image1[:, :4, :4]))

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

    def test_rare_class_recall_guard_boosts_only_scaled_low_recall_classes(self):
        train_config = TrainConfig(
            rare_class_recall_guard=True,
            rare_class_recall_target=0.70,
            rare_class_recall_guard_scale_threshold=1.5,
            rare_class_recall_guard_max_multiplier=2.0,
            rare_class_recall_guard_min_precision=0.35,
        )
        class_scales = _combine_class_target_scales(
            3,
            class_augmentation_scales=[1.0, 2.3, 1.1],
            rare_class_repeat_factors=[1.0, 2.0, 1.0],
        )
        previous_metrics = {
            "per_class": [
                {"class_index": 0, "precision": 0.9, "recall": 0.8, "support": 10},
                {"class_index": 1, "precision": 0.7, "recall": 0.35, "support": 12},
                {"class_index": 2, "precision": 0.9, "recall": 0.4, "support": 10},
            ]
        }

        summary = _resolve_rare_class_recall_guard(
            train_config=train_config,
            detection_mode=True,
            stage_name="stage2_full_detection",
            previous_val_metrics=previous_metrics,
            class_target_scales=class_scales,
        )

        self.assertTrue(summary["active"])
        self.assertEqual(summary["active_class_count"], 1)
        self.assertEqual(len(summary["multipliers"]), 3)
        self.assertEqual(summary["multipliers"][0], 1.0)
        self.assertGreater(summary["multipliers"][1], 1.0)
        self.assertEqual(summary["multipliers"][2], 1.0)

    def test_detr_class_weight_multipliers_are_resettable(self):
        criterion = DETRSetCriterion(
            num_classes=3,
            class_weights=torch.tensor([1.0, 1.5, 0.8], dtype=torch.float32),
            background_weight=0.2,
        )
        base_weight = criterion.empty_weight.clone()

        criterion.set_class_weight_multipliers(torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32))
        self.assertAlmostEqual(float(criterion.empty_weight[1].item()), 3.0, places=5)
        self.assertAlmostEqual(float(criterion.empty_weight[-1].item()), 0.2, places=5)

        criterion.reset_class_weight_multipliers()
        self.assertTrue(torch.allclose(criterion.empty_weight, base_weight))

    def test_adaptive_detection_loss_boosts_detection_weights_when_detection_lags(self):
        train_config = TrainConfig(
            adaptive_detection_loss=True,
            adaptive_detection_macro_f1_threshold=0.93,
            adaptive_detection_f1_target=0.90,
            adaptive_detection_gap_threshold=0.20,
            adaptive_detection_bbox_iou_target=0.70,
            adaptive_detection_max_multiplier=2.0,
        )
        previous_metrics = {
            "macro_f1": 0.96,
            "bbox": {"mean_iou": 0.52},
            "detection_confidence_curve": {"best_f1_50": 0.54},
        }

        summary = _resolve_adaptive_detection_loss(
            train_config=train_config,
            detection_mode=True,
            stage_name="stage2_full_detection",
            previous_val_metrics=previous_metrics,
        )
        stage_config = {
            "bbox_l1_weight": 1.0,
            "bbox_giou_weight": 0.7,
            "objectness_weight": 6.0,
            "cardinality_weight": 0.1,
        }
        _apply_adaptive_detection_loss(stage_config, summary)

        self.assertTrue(summary["active"])
        self.assertGreater(summary["multiplier"], 1.0)
        self.assertGreater(stage_config["bbox_l1_weight"], 1.0)
        self.assertGreater(stage_config["objectness_weight"], 6.0)

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

    def test_stage1_auto_advance_moves_next_epoch_to_stage2(self):
        train_config = TrainConfig(stage1_epochs=5, matcher_class_cost=0.0)

        stage1 = _resolve_detection_stage(
            train_config=train_config,
            epoch_index=2,
            detection_mode=True,
            stage1_auto_advance_epoch=None,
        )
        stage2 = _resolve_detection_stage(
            train_config=train_config,
            epoch_index=3,
            detection_mode=True,
            stage1_auto_advance_epoch=2,
        )

        self.assertEqual(stage1["stage_name"], "stage1_cls_only")
        self.assertEqual(stage1["matcher_class_cost"], 1.0)
        self.assertEqual(stage2["stage_name"], "stage2_full_detection")
        self.assertEqual(stage2["matcher_class_cost"], 0.0)

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
            "best_macro_f1_epoch": 103,
            "selected_checkpoint_macro_f1": 0.993,
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
        self.assertEqual(inherited["best_macro_f1_epoch"], 103)
        self.assertAlmostEqual(inherited["selected_checkpoint_macro_f1"], 0.993)
        self.assertEqual(reset["start_epoch"], 1)
        self.assertEqual(reset["best_epoch"], 0)
        self.assertEqual(reset["best_macro_f1"], -1.0)
        self.assertIsNone(reset["best_macro_f1_epoch"])
        self.assertIsNone(reset["selected_checkpoint_macro_f1"])
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

    def test_attention_heatmap_can_pool_cls_and_register_queries(self):
        attention = torch.zeros(1, 4, 8)
        attention[0, 0, 4:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        attention[0, 1, 4:] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        attention[0, 2, 4:] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        attention[0, 3, 4:] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        cls_heatmap = build_attention_heatmap(
            attention=attention,
            grid_size=(2, 2),
            prefix_tokens=4,
            reduction="mean",
            output_size=(2, 2),
            query_tokens="cls",
        )
        pooled_heatmap = build_attention_heatmap(
            attention=attention,
            grid_size=(2, 2),
            prefix_tokens=4,
            reduction="mean",
            output_size=(2, 2),
            query_tokens="cls_register_mean",
        )

        self.assertEqual(cls_heatmap.shape, (2, 2))
        self.assertEqual(pooled_heatmap.shape, (2, 2))
        self.assertFalse(bool((torch.tensor(cls_heatmap) == torch.tensor(pooled_heatmap)).all().item()))

    def test_metric_learning_multisource_loss_is_finite(self):
        class _FakeModel(torch.nn.Module):
            cnn_fusion_norm = torch.nn.Identity()

        criterion = SupervisedContrastiveLoss(temperature=0.2, class_balanced=True)
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        features = {
            "pooled": torch.randn(4, 8, requires_grad=True),
            "cnn_pooled": torch.randn(4, 8, requires_grad=True),
            "patches": torch.randn(4, 3, 8, requires_grad=True),
            "registers": torch.randn(4, 2, 8, requires_grad=True),
        }

        self.assertEqual(_parse_metric_learning_sources("all"), ["head", "cnn", "patch", "registers"])
        loss, used_sources = _metric_learning_loss_from_features(
            model=_FakeModel(),
            features=features,
            targets=targets,
            criterion=criterion,
            sources="head,cnn,patch,registers",
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(used_sources, ["head", "cnn", "patch", "registers"])

    def test_register_diversity_loss_penalizes_collapsed_registers(self):
        collapsed = torch.ones(2, 4, 8, requires_grad=True)
        diverse = torch.eye(4, 8).unsqueeze(0).repeat(2, 1, 1).requires_grad_(True)

        collapsed_loss = _register_diversity_loss_from_features({"registers": collapsed})
        diverse_loss = _register_diversity_loss_from_features({"registers": diverse})

        self.assertTrue(torch.isfinite(collapsed_loss).item())
        self.assertGreater(collapsed_loss.item(), diverse_loss.item())
        collapsed_loss.backward()
        self.assertIsNotNone(collapsed.grad)

    def test_indexed_sample_dataset_collate_preserves_sample_indices(self):
        class _TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return torch.full((3, 8, 8), float(index)), int(index % 2)

        wrapped = IndexedSampleDataset(_TinyDataset())
        collate = build_train_collate_fn(num_classes=2, batch_mix_probability=0.0)
        images, labels, metadata = collate([wrapped[0], wrapped[2]])

        self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
        self.assertTrue(torch.equal(labels, torch.tensor([0, 0], dtype=torch.long)))
        self.assertIn("sample_index", metadata)
        self.assertTrue(torch.equal(metadata["sample_index"], torch.tensor([0, 2], dtype=torch.long)))

    def test_early_learning_regularization_updates_target_state(self):
        logits = torch.tensor(
            [[2.0, 0.2, -0.4], [0.1, 1.5, -0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        sample_indices = torch.tensor([0, 3], dtype=torch.long)
        target_state = torch.zeros(5, 3, dtype=torch.float32)

        loss = _early_learning_regularization_loss(
            logits,
            sample_indices,
            target_state,
            beta=0.7,
            update_state=True,
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertLessEqual(float(loss.item()), 0.0)
        self.assertTrue(torch.allclose(target_state[0].sum(), torch.tensor(1.0), atol=1e-6))
        self.assertTrue(torch.allclose(target_state[3].sum(), torch.tensor(1.0), atol=1e-6))
        self.assertEqual(float(target_state[1].sum().item()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_sample_weight_dataset_and_weighted_loss(self):
        class _TinyDataset(torch.utils.data.Dataset):
            def __init__(self, root: Path):
                self.paths = [root / "a.jpg", root / "b.jpg"]
                for path in self.paths:
                    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)

            def __len__(self):
                return len(self.paths)

            def sample_paths(self):
                return list(self.paths)

            def __getitem__(self, index):
                return torch.zeros(3, 8, 8), int(index)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _TinyDataset(root)
            wrapped = SampleWeightDataset(
                dataset,
                {str(dataset.paths[1]): 2.5},
                default_weight=1.0,
                max_weight=3.0,
            )
            collate = build_train_collate_fn(num_classes=2, batch_mix_probability=0.0)
            _, labels, metadata = collate([wrapped[0], wrapped[1]])

        self.assertTrue(torch.equal(labels, torch.tensor([0, 1], dtype=torch.long)))
        self.assertIn("sample_weight", metadata)
        self.assertTrue(torch.allclose(metadata["sample_weight"], torch.tensor([1.0, 2.5])))

        logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]], requires_grad=True)
        criterion = FocalCrossEntropyLoss(gamma=0.0, focal_mix=0.0)
        unweighted = criterion(logits, labels)
        weighted, mean_weight, per_sample = _classification_loss_with_sample_weights(
            criterion,
            logits,
            labels,
            metadata["sample_weight"],
        )
        self.assertGreater(float(weighted.item()), float(unweighted.item()))
        self.assertTrue(torch.allclose(mean_weight, torch.tensor(1.75)))
        self.assertIsNotNone(per_sample)
        self.assertEqual(tuple(per_sample.shape), (2,))

    def test_sample_weight_dataset_prefers_sample_index_over_duplicate_path(self):
        class _DuplicatePathDataset(torch.utils.data.Dataset):
            def __init__(self, root: Path):
                self.path = root / "shared.jpg"
                Image.new("RGB", (8, 8), color=(128, 128, 128)).save(self.path)

            def __len__(self):
                return 2

            def sample_paths(self):
                return [self.path, self.path]

            def __getitem__(self, index):
                return torch.zeros(3, 8, 8), int(index), {"sample_index": torch.tensor(index)}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _DuplicatePathDataset(root)
            wrapped = SampleWeightDataset(
                dataset,
                {},
                sample_weights_by_sample_index={1: 2.5},
                default_weight=1.0,
                max_weight=3.0,
            )
            collate = build_train_collate_fn(num_classes=2, batch_mix_probability=0.0)
            _, labels, metadata = collate([wrapped[0], wrapped[1]])
            summary = wrapped.sample_weight_summary()

        self.assertTrue(torch.equal(labels, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(metadata["sample_index"], torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.allclose(metadata["sample_weight"], torch.tensor([1.0, 2.5])))
        self.assertEqual(summary["key_mode"], "sample_index")
        self.assertEqual(summary["weighted_sample_indices"], 1)
        self.assertEqual(summary["matched_samples"], 1)

    def test_sample_weight_manifest_sample_index_rows_do_not_create_path_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "train" / "class0" / "shared.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_path)
            manifest = root / "sample_weights.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_index", "image_path", "sample_weight", "reason"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_index": 1,
                        "image_path": str(image_path),
                        "sample_weight": 2.5,
                        "reason": "sample_index_only",
                    }
                )

            by_path, by_index, summary = _load_sample_weight_manifest(
                str(manifest),
                default_weight=1.0,
                max_weight=3.0,
            )

        self.assertEqual(by_path, {})
        self.assertEqual(by_index, {1: 2.5})
        self.assertEqual(summary["key_mode"], "sample_index")
        self.assertEqual(summary["paths"], 0)
        self.assertEqual(summary["sample_indices"], 1)

    def test_targeted_margin_dataset_loader_and_loss(self):
        class _TinyDataset(torch.utils.data.Dataset):
            def __init__(self, root: Path):
                self.paths = [
                    root / "train" / "class0" / "a.jpg",
                    root / "train" / "class1" / "b.jpg",
                ]
                for path in self.paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)

            def __len__(self):
                return len(self.paths)

            def sample_paths(self):
                return list(self.paths)

            def __getitem__(self, index):
                return torch.zeros(3, 8, 8), int(index)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _TinyDataset(root)
            manifest_path = root / "targeted_margin.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "target_index",
                        "negative_index",
                        "targeted_margin",
                        "targeted_margin_weight",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(dataset.paths[0]),
                        "target_index": 0,
                        "negative_index": 1,
                        "targeted_margin": 0.2,
                        "targeted_margin_weight": 1.5,
                    }
                )
            specs, summary = _load_targeted_margin_manifest(
                str(manifest_path),
                num_classes=2,
                default_margin=0.12,
                default_weight=1.0,
                max_weight=2.0,
            )
            wrapped = TargetedMarginDataset(
                dataset,
                specs,
                default_margin=0.12,
                default_weight=1.0,
                max_weight=2.0,
            )
            collate = build_train_collate_fn(num_classes=2, batch_mix_probability=0.0)
            _, labels, metadata = collate([wrapped[0], wrapped[1]])

            self.assertTrue(summary["enabled"])
            self.assertIn("targeted_margin_negative", metadata)
            self.assertTrue(torch.equal(metadata["targeted_margin_negative"], torch.tensor([1, -1])))
            self.assertTrue(torch.allclose(metadata["targeted_margin_weight"], torch.tensor([1.5, 0.0])))

            logits = torch.tensor([[0.1, 0.4], [0.2, 1.0]], dtype=torch.float32, requires_grad=True)
            loss, mean_weight, fraction = _targeted_margin_loss_from_logits(
                logits=logits,
                hard_labels=labels,
                targets=labels,
                negative_indices=metadata["targeted_margin_negative"],
                weights=metadata["targeted_margin_weight"],
                margins=metadata["targeted_margin_margin"],
                default_margin=0.12,
            )
            self.assertTrue(torch.isfinite(loss).item())
            self.assertAlmostEqual(float(loss.item()), 0.5, places=5)
            self.assertAlmostEqual(float(mean_weight.item()), 0.75, places=5)
            self.assertAlmostEqual(float(fraction), 0.5, places=5)
            loss.backward()
            self.assertIsNotNone(logits.grad)

            leak_manifest = root / "targeted_margin_leak.csv"
            leak_path = root / "test" / "class0" / "x.jpg"
            leak_path.parent.mkdir(parents=True, exist_ok=True)
            with leak_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "target_index", "negative_index"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(leak_path),
                        "target_index": 0,
                        "negative_index": 1,
                    }
                )
            with self.assertRaises(ValueError):
                _load_targeted_margin_manifest(
                    str(leak_manifest),
                    num_classes=2,
                    default_margin=0.12,
                    default_weight=1.0,
                    max_weight=2.0,
                )

    def test_ambiguous_soft_target_dataset_and_loader_rejects_non_train(self):
        class _TinyDataset(torch.utils.data.Dataset):
            def __init__(self, root: Path):
                self.paths = [
                    root / "train" / "class0" / "a.jpg",
                    root / "train" / "class1" / "b.jpg",
                ]
                for path in self.paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)

            def __len__(self):
                return len(self.paths)

            def sample_paths(self):
                return list(self.paths)

            def labels(self):
                return [0, 1]

            def __getitem__(self, index):
                return torch.zeros(3, 8, 8), int(index)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _TinyDataset(root)
            manifest_path = root / "ambiguous.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "sample_index",
                        "target_index",
                        "soft_target_index",
                        "alpha",
                        "prob_0",
                        "prob_1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(dataset.paths[0]),
                        "sample_index": 0,
                        "target_index": 0,
                        "soft_target_index": 1,
                        "alpha": 0.25,
                    }
                )
                writer.writerow(
                    {
                        "image_path": str(dataset.paths[0]),
                        "sample_index": 1,
                        "prob_0": 0.1,
                        "prob_1": 0.9,
                    }
                )
            soft_targets, summary = _load_ambiguous_soft_target_manifest(
                str(manifest_path),
                num_classes=2,
                default_alpha=0.20,
            )
            wrapped = AmbiguousSoftTargetDataset(
                dataset,
                soft_targets,
                num_classes=2,
                default_alpha=0.20,
            )
            collate = build_train_collate_fn(num_classes=2, batch_mix_probability=0.0)
            _, labels, metadata = collate([wrapped[0], wrapped[1]])

            self.assertTrue(summary["enabled"])
            self.assertEqual(summary["key_mode"], "sample_index")
            self.assertTrue(torch.equal(labels, torch.tensor([0, 1], dtype=torch.long)))
            self.assertIn("soft_target", metadata)
            self.assertTrue(
                torch.allclose(
                    metadata["soft_target"],
                    torch.tensor([[0.75, 0.25], [0.1, 0.9]], dtype=torch.float32),
                )
            )

            leak_manifest = root / "leak.csv"
            leak_path = root / "val" / "class0" / "x.jpg"
            leak_path.parent.mkdir(parents=True, exist_ok=True)
            with leak_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_path", "target_index", "soft_target_index"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": str(leak_path),
                        "target_index": 0,
                        "soft_target_index": 1,
                    }
                )
            with self.assertRaises(ValueError):
                _load_ambiguous_soft_target_manifest(
                    str(leak_manifest),
                    num_classes=2,
                    default_alpha=0.25,
                )

    def test_background_counterfactual_consistency_loss_is_finite(self):
        class _TinyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(3, 2)

            def forward(self, images):
                return self.head(images.mean(dim=(2, 3)))

        model = _TinyClassifier()
        images = torch.randn(4, 3, 16, 16)
        logits = model(images)
        counterfactual = _background_counterfactual_images(
            images,
            mode="desaturate_blur",
            margin=0.08,
            blur_kernel=5,
        )
        self.assertEqual(tuple(counterfactual.shape), tuple(images.shape))
        self.assertTrue(torch.isfinite(counterfactual).all().item())

        loss, fraction = _background_counterfactual_consistency_loss(
            model=model,
            images=images,
            logits=logits,
            probability=1.0,
            mode="desaturate_blur",
            margin=0.08,
            blur_kernel=5,
            temperature=1.0,
            amp=False,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(fraction, 1.0)

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse

import pytest
import torch

from trkh.tools.build_detection_classification_curriculum_checkpoint import (
    DETECTOR_RUNTIME_ARGUMENTS,
    ENCODER_TRANSFER_PREFIXES,
    detector_train_profile,
    fresh_checkpoint_payload,
    is_encoder_transfer_key,
    merge_matching_state,
)
from trkh.training.train import (
    _propagate_curriculum_transfer_metadata,
    _validate_resume_runtime_requirements,
)


def test_encoder_allowlist_includes_representation_but_excludes_heads() -> None:
    for prefix in ENCODER_TRANSFER_PREFIXES:
        assert is_encoder_transfer_key(prefix)
    assert is_encoder_transfer_key("blocks.0.attn.qkv.weight")
    assert is_encoder_transfer_key("stem.0.weight")
    assert not is_encoder_transfer_key("head.weight")
    assert not is_encoder_transfer_key("classification_head.weight")
    assert not is_encoder_transfer_key("bbox_head.net.0.weight")
    assert not is_encoder_transfer_key("pairwise_margin_head.weight")


def test_encoder_only_merge_preserves_classifier_heads() -> None:
    source = {
        "blocks.0.weight": torch.full((2, 2), 3.0),
        "head.weight": torch.full((2, 2), 9.0),
        "classification_head.weight": torch.full((2, 2), 8.0),
    }
    target = {
        "blocks.0.weight": torch.zeros(2, 2),
        "head.weight": torch.ones(2, 2),
    }
    merged, summary = merge_matching_state(source, target, encoder_only=True)
    assert torch.equal(merged["blocks.0.weight"], source["blocks.0.weight"])
    assert torch.equal(merged["head.weight"], target["head.weight"])
    assert summary["transferred_keys"] == ["blocks.0.weight"]
    assert summary["source_candidates_without_target"] == []


def test_full_matching_merge_reports_source_only_and_shape_mismatch() -> None:
    source = {
        "stem.weight": torch.ones(2, 2),
        "head.weight": torch.ones(3, 2),
        "head.bias": torch.ones(3),
    }
    target = {
        "stem.weight": torch.zeros(2, 2),
        "head.weight": torch.zeros(4, 2),
    }
    _, summary = merge_matching_state(source, target, encoder_only=False)
    assert summary["transferred_keys"] == ["stem.weight"]
    assert summary["shape_mismatches"] == ["head.weight"]
    assert summary["source_candidates_without_target"] == ["head.bias"]


def test_fresh_checkpoint_resets_training_state_and_keeps_metadata() -> None:
    reference = {
        "epoch": 9,
        "model_state": {"head.weight": torch.zeros(2, 2)},
        "train_model_state": {"head.weight": torch.ones(2, 2)},
        "optimizer_state": {"state": {1: {}}},
        "scheduler_state": {"last_epoch": 9},
        "scaler_state": {"scale": 1.0},
        "ema_model_state": {"head.weight": torch.ones(2, 2)},
        "metrics": {"test_macro_f1": 0.99},
        "best_epoch": 7,
        "resume_state": {"next_epoch": 10},
        "train_stage": "classification",
        "stage1_auto_advance": {"enabled": True},
        "checkpoint_weight_source": "ema",
        "validation_weight_source": "ema",
        "calibration": {"source_split": "val"},
        "data_summary": {"crop_to_primary_object": True},
        "imbalance_summary": {"sampler_type": "strict_balanced"},
        "class_names": ["a", "b"],
        "train_config": {"epochs": 30},
    }
    state = {"head.weight": torch.full((2, 2), 2.0)}
    payload = fresh_checkpoint_payload(
        reference,
        model_state=state,
        model_config={"model_type": "vit_registers"},
        transfer_summary={"mode": "test"},
        train_config={"epochs": 2, "stage1_epochs": 0},
    )
    assert payload["epoch"] == 0
    assert payload["optimizer_state"] is None
    assert payload["scheduler_state"] is None
    assert payload["scaler_state"] is None
    assert "train_model_state" not in payload
    assert "ema_model_state" not in payload
    assert "metrics" not in payload
    assert "best_epoch" not in payload
    assert "resume_state" not in payload
    assert "train_stage" not in payload
    assert "stage1_auto_advance" not in payload
    assert "checkpoint_weight_source" not in payload
    assert "validation_weight_source" not in payload
    assert "calibration" not in payload
    assert "data_summary" not in payload
    assert "imbalance_summary" not in payload
    assert payload["class_names"] == ["a", "b"]
    assert torch.equal(payload["model_state"]["head.weight"], state["head.weight"])
    assert payload["raw_dataset_touched"] is False
    assert payload["test_split_used"] is False
    assert payload["train_config"] == {"epochs": 2, "stage1_epochs": 0}


def test_detector_train_profile_starts_full_detection_with_light_localization() -> None:
    class Args:
        train_batch_size = 6
        train_grad_accum_steps = 2
        train_epochs = 3
        train_learning_rate = 2e-4
        train_backbone_lr_scale = 0.1
        train_min_learning_rate = 1e-6
        train_patience = 2
        num_workers = 4
        eval_num_workers = 2
        max_train_batches = 24
        max_val_batches = 3

    profile = detector_train_profile(
        {
            "stage1_epochs": 30,
            "bbox_l1_loss_weight": 1.0,
            "objectness_loss_weight": 5.0,
            "metric_learning_loss_weight": 0.04,
            "teacher_focus_binary_loss_weight": 0.015,
        },
        args=Args(),
    )
    assert profile["stage1_epochs"] == 0
    assert profile["bbox_l1_loss_weight"] == 0.20
    assert profile["bbox_giou_loss_weight"] == 0.10
    assert profile["objectness_loss_weight"] == 1.0
    assert profile["backbone_lr_scale"] == 0.1
    assert profile["metric_learning_loss_weight"] == 0.0
    assert profile["teacher_focus_binary_loss_weight"] == 0.0
    assert profile["max_train_batches"] == 24


def test_curriculum_detector_runtime_requirements_fail_closed() -> None:
    checkpoint = {
        "curriculum_transfer": {
            "required_runtime": {
                "full_image_detection": True,
                "skip_final_test": True,
                "trainer_arguments": list(DETECTOR_RUNTIME_ARGUMENTS),
            }
        }
    }
    args = argparse.Namespace(full_image_detection=False, skip_final_test=False)
    with pytest.raises(ValueError, match="--full-image-detection --skip-final-test"):
        _validate_resume_runtime_requirements(args, checkpoint)

    args.full_image_detection = True
    args.skip_final_test = True
    _validate_resume_runtime_requirements(args, checkpoint)


def test_resume_runtime_requirements_ignore_regular_checkpoint() -> None:
    args = argparse.Namespace(full_image_detection=False, skip_final_test=False)
    _validate_resume_runtime_requirements(args, {"model_state": {}})


def test_curriculum_metadata_propagates_to_descendant_checkpoint() -> None:
    source = {
        "curriculum_transfer": {
            "mode": "classifier_to_detector",
            "required_runtime": {
                "full_image_detection": True,
                "skip_final_test": True,
            },
        }
    }
    descendant = {"epoch": 1}
    _propagate_curriculum_transfer_metadata(descendant, source)
    assert descendant["curriculum_transfer"] == source["curriculum_transfer"]
    assert descendant["curriculum_transfer"] is not source["curriculum_transfer"]

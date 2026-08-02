from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from trkh.tools.export_aidt_predictions import parse_args as parse_aidt_args
from trkh.tools.export_timm_predictions import (
    align_imagefolder_class_order,
    build_export_model,
    checkpoint_eval_transform,
    checkpoint_export_metadata,
    parse_args as parse_timm_args,
)


def test_timm_export_requires_explicit_test_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_timm_predictions.py",
            "--checkpoint",
            "missing.pt",
            "--data",
            "missing_data",
            "--split",
            "test",
            "--output-dir",
            "missing_output",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_timm_args()

    assert exc_info.value.code == 2
    assert "--split test requires the explicit --allow-test acknowledgement" in capsys.readouterr().err

def test_timm_export_allows_validation_without_test_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_timm_predictions.py",
            "--checkpoint",
            "checkpoint.pt",
            "--data",
            "dataset",
            "--split",
            "val",
            "--output-dir",
            "output",
        ],
    )

    args = parse_timm_args()

    assert args.split == "val"
    assert args.allow_test is False


def test_timm_export_reads_current_trkh_checkpoint_schema() -> None:
    state = {"head.weight": object()}
    classes, model_name, resolved_state, selection = checkpoint_export_metadata(
        {
            "class_names": ["zero", "one"],
            "model_config": {
                "model_type": "timm_classifier",
                "timm_model_name": "vit_small_patch16_dinov3.lvd1689m",
            },
            "model_state": state,
        }
    )

    assert classes == ["zero", "one"]
    assert model_name == "vit_small_patch16_dinov3.lvd1689m"
    assert resolved_state is state
    assert selection["resolved_state_key"] == "model_state"


def test_timm_export_auto_uses_ema_that_produced_last_checkpoint_validation() -> None:
    train_state = {"weight": object()}
    ema_state = {"weight": object()}
    _, _, resolved_state, selection = checkpoint_export_metadata(
        {
            "class_names": ["zero", "one"],
            "model_config": {"timm_model_name": "model"},
            "model_state": train_state,
            "ema_model_state": ema_state,
            "validation_weight_source": "ema",
            "resume_state": {"checkpoint_kind": "last"},
        },
        weight_source="auto",
    )

    assert resolved_state is ema_state
    assert selection == {
        "requested": "auto",
        "resolved_state_key": "ema_model_state",
        "checkpoint_weight_source": "",
        "validation_weight_source": "ema",
    }


def test_timm_export_auto_uses_promoted_best_model_state() -> None:
    promoted_ema_state = {"weight": object()}
    stale_ema_copy = {"weight": object()}
    _, _, resolved_state, selection = checkpoint_export_metadata(
        {
            "class_names": ["zero", "one"],
            "model_config": {"timm_model_name": "model"},
            "model_state": promoted_ema_state,
            "ema_model_state": stale_ema_copy,
            "checkpoint_weight_source": "ema",
            "validation_weight_source": "ema",
        },
        weight_source="auto",
    )

    assert resolved_state is promoted_ema_state
    assert selection["resolved_state_key"] == "model_state"


def test_timm_export_explicit_weight_source_fails_when_ema_is_absent() -> None:
    with pytest.raises(ValueError, match="missing model/model_state weights"):
        checkpoint_export_metadata(
            {
                "class_names": ["zero", "one"],
                "model_config": {"timm_model_name": "model"},
                "model_state": {"weight": object()},
            },
            weight_source="ema",
        )


def test_timm_export_rebuilds_current_trkh_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_build_model(checkpoint: dict, num_classes: int) -> object:
        captured["checkpoint"] = checkpoint
        captured["num_classes"] = num_classes
        return sentinel

    monkeypatch.setattr(
        "trkh.tools.export_timm_predictions.build_model_from_checkpoint",
        fake_build_model,
    )
    checkpoint = {
        "model_config": {"model_type": "dinov3_surface_patch_hybrid_v2"},
        "model_state": {"weight": object()},
    }
    selected_state = {"weight": object()}
    model, source = build_export_model(
        checkpoint,
        model_name="vit_small_patch16_dinov3.lvd1689m",
        num_classes=5,
        state_dict=selected_state,
    )

    assert model is sentinel
    assert source == "trkh_checkpoint_architecture"
    assert captured["num_classes"] == 5
    assert captured["checkpoint"]["model_state"] is selected_state
    assert checkpoint["model_state"] is not selected_state


def test_timm_export_keeps_legacy_checkpoint_schema() -> None:
    state = {"head.weight": object()}
    classes, model_name, resolved_state, selection = checkpoint_export_metadata(
        {
            "classes": ["zero", "one"],
            "args": {"model": "legacy_model"},
            "model": state,
        }
    )

    assert classes == ["zero", "one"]
    assert model_name == "legacy_model"
    assert resolved_state is state
    assert selection["resolved_state_key"] == "model"


def test_timm_export_rebuilds_current_trkh_eval_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_build_eval_transform(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "trkh.tools.export_timm_predictions.build_eval_transform",
        fake_build_eval_transform,
    )
    transform, mean, std, metadata = checkpoint_eval_transform(
        {
            "model_config": {
                "image_size": 256,
                "input_mean": [0.1, 0.2, 0.3],
                "input_std": [0.4, 0.5, 0.6],
                "temporal_frames": 1,
            },
            "augmentation_config": {
                "resize_mode": "pad",
                "illumination_normalization": True,
                "illumination_normalization_strength": 0.25,
                "foreground_crop_mode": "green_mask",
                "foreground_crop_margin_ratio": 0.0,
                "foreground_crop_min_mask_area_ratio": 0.04,
                "foreground_crop_max_mask_area_ratio": 0.91,
                "foreground_crop_max_crop_area_ratio": 0.97,
                "background_suppression_mode": "desaturate_blur",
                "background_suppression_margin": 0.07,
                "background_suppression_blur_radius": 5.0,
                "surface_detail_amplification_mode": "unsharp",
                "surface_detail_amplification_strength": 0.2,
                "surface_detail_amplification_blur_radius": 1.5,
                "surface_detail_amplification_foreground_weight": 0.8,
                "eval_surface_detail_amplification": True,
            },
        },
        "ignored_for_trkh_metadata",
    )

    assert transform is sentinel
    assert mean == (0.1, 0.2, 0.3)
    assert std == (0.4, 0.5, 0.6)
    assert captured == {
        "image_size": 256,
        "resize_mode": "pad",
        "illumination_normalization": True,
        "illumination_normalization_strength": 0.25,
        "foreground_crop_mode": "green_mask",
        "foreground_crop_margin_ratio": 0.0,
        "foreground_crop_min_mask_area_ratio": 0.04,
        "foreground_crop_max_mask_area_ratio": 0.91,
        "foreground_crop_max_crop_area_ratio": 0.97,
        "background_suppression_mode": "desaturate_blur",
        "background_suppression_margin": 0.07,
        "background_suppression_blur_radius": 5.0,
        "surface_detail_amplification_mode": "unsharp",
        "surface_detail_amplification_strength": 0.2,
        "surface_detail_amplification_blur_radius": 1.5,
        "surface_detail_amplification_foreground_weight": 0.8,
        "eval_surface_detail_amplification": True,
        "mean": (0.1, 0.2, 0.3),
        "std": (0.4, 0.5, 0.6),
    }
    assert metadata["source"] == "trkh_checkpoint"
    assert metadata["resize_mode"] == "pad"
    assert metadata["input_mean"] == [0.1, 0.2, 0.3]
    assert metadata["input_std"] == [0.4, 0.5, 0.6]


def test_timm_export_keeps_timm_preprocessing_for_legacy_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []

    def fake_build_transform_config(model_name: str) -> tuple[object, tuple[float, ...], tuple[float, ...]]:
        calls.append(model_name)
        return sentinel, (0.1, 0.2, 0.3), (0.4, 0.5, 0.6)

    monkeypatch.setattr(
        "trkh.tools.export_timm_predictions.build_transform_config",
        fake_build_transform_config,
    )
    transform, mean, std, metadata = checkpoint_eval_transform(
        {"classes": ["zero", "one"], "args": {"model": "legacy_model"}},
        "legacy_model",
    )

    assert transform is sentinel
    assert calls == ["legacy_model"]
    assert mean == (0.1, 0.2, 0.3)
    assert std == (0.4, 0.5, 0.6)
    assert metadata == {
        "source": "timm_model_config",
        "input_mean": [0.1, 0.2, 0.3],
        "input_std": [0.4, 0.5, 0.6],
    }


def test_timm_export_rejects_malformed_trkh_preprocessing_metadata() -> None:
    with pytest.raises(ValueError, match="augmentation_config must be a mapping"):
        checkpoint_eval_transform(
            {
                "model_config": {"image_size": 256},
                "augmentation_config": "invalid",
            },
            "unused",
        )


def test_timm_export_remaps_imagefolder_targets_to_checkpoint_order() -> None:
    dataset = SimpleNamespace(
        classes=["class_a", "class_b"],
        class_to_idx={"class_a": 0, "class_b": 1},
        samples=[("b.jpg", 1), ("a.jpg", 0)],
        imgs=[("b.jpg", 1), ("a.jpg", 0)],
        targets=[1, 0],
    )

    align_imagefolder_class_order(dataset, ["class_b", "class_a"])

    assert dataset.classes == ["class_b", "class_a"]
    assert dataset.class_to_idx == {"class_b": 0, "class_a": 1}
    assert dataset.samples == [("b.jpg", 0), ("a.jpg", 1)]
    assert dataset.imgs == dataset.samples
    assert dataset.targets == [0, 1]


def test_aidt_export_requires_explicit_test_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_aidt_predictions.py",
            "--checkpoint",
            "missing.pt",
            "--data",
            "missing_data",
            "--split",
            "test",
            "--output-dir",
            "missing_output",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_aidt_args()

    assert exc_info.value.code == 2
    assert "--split test requires the explicit --allow-test acknowledgement" in capsys.readouterr().err

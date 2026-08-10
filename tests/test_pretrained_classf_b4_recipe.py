from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import trkh.recipes.pretrained_classf_b4 as recipe
from trkh.recipes.pretrained_classf_b0 import (
    DINO_MODEL_NAME,
    DINO_SHA256,
    EXPECTED_CLASS_NAMES,
)
from trkh.training.train import build_configs, parse_args


def _build_args(tmp_path: Path, *, stage: str = "probe") -> list[str]:
    dino_checkpoint = tmp_path / "model.safetensors"
    dino_checkpoint.write_bytes(b"unit-test-placeholder")
    return recipe.build_train_args(
        data_yaml=tmp_path / "class_f_5class_dev.yaml",
        dino_checkpoint=dino_checkpoint,
        b2_checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "runs",
        stage=stage,
        run_tag="unit",
        batch_size=24,
        num_workers=2,
        eval_num_workers=1,
        seed=42,
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        dataset_image_tree_sha256=recipe.DATASET_IMAGE_TREE_SHA256,
    )


def _fake_b2_checkpoint(path: Path, *, data_yaml: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "class_names": list(EXPECTED_CLASS_NAMES),
            "data_yaml": str(data_yaml.resolve()),
            "epoch": 4,
            "best_epoch": 4,
            "checkpoint_weight_source": "ema",
            "validation_weight_source": "ema",
            "model_state": {},
            "model_config": {
                "model_type": "timm_classifier",
                "research_track": "pretrained",
                "pretrained": True,
                "timm_model_name": DINO_MODEL_NAME,
                "pretrained_checkpoint_sha256": DINO_SHA256,
            },
            "train_config": {
                "experiment_protocol_id": recipe.B2_TEMPERED_P05_PROTOCOL_ID,
                "source_commit": recipe.B2_SOURCE_COMMIT,
                "source_tree_sha256": recipe.B2_SOURCE_TREE_SHA256,
                "dataset_image_tree_sha256": (
                    recipe.DATASET_IMAGE_TREE_SHA256
                ),
                "recipe_train_contract_sha256": (
                    recipe.B2_RECIPE_TRAIN_CONTRACT_SHA256
                ),
                "balanced_epoch_sampling": False,
                "tempered_class_sampling_power": 0.5,
                "classification_loss": "ldam_focal",
                "ldam_max_margin": 0.3,
                "ldam_scale": 18.0,
                "focal_loss_gamma": 1.0,
                "focal_loss_mix": 0.1,
                "label_smoothing": 0.02,
                "batch_size": 24,
                "grad_accum_steps": 2,
                "epochs": 5,
                "scheduler_total_epochs": 5,
                "max_train_batches": 120,
                "max_val_batches": 0,
                "warmup_epochs": 1,
            },
        },
        path,
    )


def test_b4_probe_recipe_is_locked_lora_only_b2_continuation(
    tmp_path: Path,
) -> None:
    args = _build_args(tmp_path)
    parsed = parse_args(args)
    model_config, train_config, augmentation_config = build_configs(parsed)

    assert parsed.run_name == "pretrained_dinov3_classf_b4_qv_lora_r4_probe_unit"
    assert parsed.resume == (tmp_path / "best.pt").resolve()
    assert parsed.resume_use_cli_config is True
    assert parsed.resume_weight_source == "selected"
    assert parsed.resume_reset_epoch is True
    assert parsed.resume_reset_optimizer is True
    assert parsed.resume_reset_scheduler is True
    assert parsed.resume_reset_scaler is True
    assert parsed.skip_final_test is True

    assert model_config.model_type == "timm_classifier"
    assert model_config.research_track == "pretrained"
    assert model_config.pretrained is True
    assert model_config.timm_model_name == DINO_MODEL_NAME
    assert model_config.pretrained_checkpoint_sha256 == DINO_SHA256
    assert model_config.timm_qv_lora is True
    assert model_config.timm_qv_lora_layers == "8,9,10,11"
    assert model_config.timm_qv_lora_rank == 4
    assert model_config.timm_qv_lora_alpha == 8.0
    assert model_config.timm_qv_lora_dropout == pytest.approx(0.05)

    assert train_config.trainable_module_prefixes.split(",") == list(
        recipe.TRAINABLE_PREFIXES
    )
    assert train_config.learning_rate == pytest.approx(2e-4)
    assert train_config.backbone_lr_scale == pytest.approx(1.0)
    assert train_config.epochs == 5
    assert train_config.scheduler_total_epochs == 5
    assert train_config.warmup_epochs == 1
    assert train_config.max_train_batches == 120
    assert train_config.max_val_batches == 0
    assert train_config.batch_size == 24
    assert train_config.grad_accum_steps == 2
    assert train_config.tempered_class_sampling_power == pytest.approx(0.5)
    assert train_config.balanced_epoch_sampling is False
    assert train_config.classification_loss == "ldam_focal"
    assert train_config.ldam_max_margin == pytest.approx(0.3)
    assert train_config.ldam_scale == pytest.approx(18.0)
    assert train_config.focal_loss_gamma == pytest.approx(1.0)
    assert train_config.focal_loss_mix == pytest.approx(0.1)
    assert train_config.label_smoothing == pytest.approx(0.02)
    assert train_config.experiment_protocol_id == recipe.PROTOCOL_ID

    assert augmentation_config.resize_mode == "pad"
    assert augmentation_config.random_resized_crop_scale_min == pytest.approx(
        0.90
    )
    assert augmentation_config.random_resized_crop_probability == pytest.approx(
        0.30
    )
    assert augmentation_config.color_jitter_brightness == pytest.approx(0.05)
    assert augmentation_config.color_jitter_contrast == pytest.approx(0.05)
    assert augmentation_config.color_jitter_saturation == pytest.approx(0.03)
    assert augmentation_config.color_jitter_hue == pytest.approx(0.01)


def test_b4_smoke_is_bounded_and_test_locked(tmp_path: Path) -> None:
    parsed = parse_args(_build_args(tmp_path, stage="smoke"))

    assert parsed.epochs == 1
    assert parsed.scheduler_total_epochs == 5
    assert parsed.warmup_epochs == 0
    assert parsed.max_train_batches == 4
    assert parsed.max_val_batches == 2
    assert parsed.skip_final_test is True


def test_b4_contract_digest_covers_resume_and_adapter_delta(
    tmp_path: Path,
) -> None:
    args = _build_args(tmp_path)
    digest_index = args.index("--recipe-train-contract-sha256")
    assert digest_index == len(args) - 2
    observed = args[digest_index + 1]
    expected = hashlib.sha256(
        json.dumps(
            args[:digest_index],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert observed == expected
    int(observed, 16)
    assert "--resume" in args[:digest_index]
    assert "--timm-qv-lora" in args[:digest_index]
    assert "--trainable-module-prefixes" in args[:digest_index]


def test_b4_exact_trainable_prefixes_have_24576_parameters() -> None:
    assert recipe.TRAINABLE_PREFIXES == (
        "blocks.8.attn.qkv.q_lora",
        "blocks.8.attn.qkv.v_lora",
        "blocks.9.attn.qkv.q_lora",
        "blocks.9.attn.qkv.v_lora",
        "blocks.10.attn.qkv.q_lora",
        "blocks.10.attn.qkv.v_lora",
        "blocks.11.attn.qkv.q_lora",
        "blocks.11.attn.qkv.v_lora",
    )
    dino_width = 384
    parameters_per_projection = (
        dino_width * recipe.LORA_RANK
        + recipe.LORA_RANK * dino_width
    )
    observed = len(recipe.TRAINABLE_PREFIXES) * parameters_per_projection
    assert observed == recipe.EXPECTED_LORA_TRAINABLE_PARAMETERS == 24_576


def test_b4_train_arg_validator_rejects_trainable_head_drift(
    tmp_path: Path,
) -> None:
    args = _build_args(tmp_path)
    summary = recipe.validate_b4_train_args(args)
    assert summary["effective_batch_size"] == 48
    assert summary["test_locked"] is True
    assert summary["expected_trainable_parameters"] == 24_576

    prefix_index = args.index("--trainable-module-prefixes") + 1
    args[prefix_index] += ",head"
    with pytest.raises(ValueError, match="train-argument contract mismatch"):
        recipe.validate_b4_train_args(args)


def test_b4_locked_b2_checkpoint_accepts_only_exact_path_hash_and_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_yaml = tmp_path / "configs" / "class_f_5class_dev.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("format: classification_folder\n", encoding="utf-8")
    relative_path = Path("locked") / "best.pt"
    checkpoint_path = tmp_path / relative_path
    _fake_b2_checkpoint(checkpoint_path, data_yaml=data_yaml)
    monkeypatch.setattr(recipe, "B2_BEST_RELATIVE_PATH", relative_path)
    monkeypatch.setattr(
        recipe,
        "B2_BEST_SHA256",
        recipe._sha256(checkpoint_path),
    )

    summary = recipe.validate_locked_b2_checkpoint(
        checkpoint_path,
        repo_root=tmp_path,
        training_data_yaml=data_yaml,
    )

    assert summary["weight_source"] == "selected_ema_model_state"
    assert summary["epoch"] == 4
    assert summary["source_commit"] == recipe.B2_SOURCE_COMMIT
    assert summary["test_inference_performed"] is False

    copied_path = tmp_path / "copy.pt"
    copied_path.write_bytes(checkpoint_path.read_bytes())
    with pytest.raises(ValueError, match="resume path"):
        recipe.validate_locked_b2_checkpoint(
            copied_path,
            repo_root=tmp_path,
            training_data_yaml=data_yaml,
        )


def test_b4_locked_b2_checkpoint_rejects_sha_and_provenance_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_yaml = tmp_path / "class_f_5class_dev.yaml"
    data_yaml.write_text("format: classification_folder\n", encoding="utf-8")
    relative_path = Path("locked") / "best.pt"
    checkpoint_path = tmp_path / relative_path
    _fake_b2_checkpoint(checkpoint_path, data_yaml=data_yaml)
    monkeypatch.setattr(recipe, "B2_BEST_RELATIVE_PATH", relative_path)
    monkeypatch.setattr(recipe, "B2_BEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        recipe.validate_locked_b2_checkpoint(
            checkpoint_path,
            repo_root=tmp_path,
            training_data_yaml=data_yaml,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["train_config"]["source_commit"] = "f" * 40
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(
        recipe,
        "B2_BEST_SHA256",
        recipe._sha256(checkpoint_path),
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        recipe.validate_locked_b2_checkpoint(
            checkpoint_path,
            repo_root=tmp_path,
            training_data_yaml=data_yaml,
        )


def test_b4_branch_contract_is_exact_and_probe_requires_clean() -> None:
    good = {
        "available": True,
        "branch": recipe.REQUIRED_BRANCH,
        "commit": "a" * 40,
        "clean": True,
    }
    recipe.validate_branch_contract(good, require_clean=True)

    with pytest.raises(RuntimeError, match="requires branch"):
        recipe.validate_branch_contract(
            {**good, "branch": "classification-only-research"},
            require_clean=False,
        )
    with pytest.raises(RuntimeError, match="clean worktree"):
        recipe.validate_branch_contract(
            {**good, "clean": False},
            require_clean=True,
        )
    with pytest.raises(RuntimeError, match="auditable Git"):
        recipe.validate_branch_contract(
            {"available": False},
            require_clean=False,
        )

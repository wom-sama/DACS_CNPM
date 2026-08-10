from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import trkh.recipes.pretrained_classf_b4 as b4
import trkh.recipes.pretrained_classf_b5 as recipe
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


def _build_b4_args(tmp_path: Path, *, stage: str = "probe") -> list[str]:
    return b4.build_train_args(
        data_yaml=tmp_path / "class_f_5class_dev.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
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


def _drop_value_option(args: list[str], option: str) -> None:
    index = args.index(option)
    del args[index : index + 2]


def _refresh_digest(args: list[str]) -> None:
    _drop_value_option(args, "--recipe-train-contract-sha256")
    args.extend(
        [
            "--recipe-train-contract-sha256",
            b4._train_contract_sha256(args),
        ]
    )


def test_b5_probe_is_exact_b4_plus_locked_posf_delta(tmp_path: Path) -> None:
    b5_args = _build_args(tmp_path)
    parsed = parse_args(b5_args)
    model_config, train_config, augmentation_config = build_configs(parsed)

    assert parsed.run_name == "pretrained_dinov3_classf_b5_posf_qv_lora_r4_probe_unit"
    assert parsed.resume == (tmp_path / "best.pt").resolve()
    assert parsed.resume_weight_source == "selected"
    assert parsed.resume_use_cli_config is True
    assert parsed.resume_reset_epoch is True
    assert parsed.resume_reset_optimizer is True
    assert parsed.resume_reset_scheduler is True
    assert parsed.resume_reset_scaler is True
    assert parsed.skip_final_test is True

    assert model_config.timm_qv_lora is True
    assert model_config.timm_qv_lora_layers == recipe.LORA_LAYERS
    assert model_config.timm_qv_lora_rank == recipe.LORA_RANK
    assert model_config.timm_qv_lora_alpha == recipe.LORA_ALPHA
    assert model_config.timm_qv_lora_dropout == pytest.approx(recipe.LORA_DROPOUT)
    assert model_config.ordinal_maturity_head is False
    assert model_config.cumulative_ordinal_head is False

    assert train_config.trainable_module_prefixes.split(",") == list(
        recipe.TRAINABLE_PREFIXES
    )
    assert train_config.semantic_attribute_loss_weight == pytest.approx(0.006)
    assert train_config.semantic_attribute_specs == recipe.SEMANTIC_ATTRIBUTE_SPECS
    assert train_config.ordinal_maturity_loss_weight == 0.0
    assert train_config.cumulative_ordinal_loss_weight == 0.0
    assert train_config.ordinal_distribution_loss_weight == 0.0
    assert train_config.ordinal_boundary_loss_weight == 0.0
    assert train_config.focus_auc_rank_loss_weight == 0.0
    assert train_config.focus_partial_auc_loss_weight == 0.0
    assert train_config.experiment_protocol_id == recipe.PROTOCOL_ID

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

    assert augmentation_config.resize_mode == "pad"
    assert augmentation_config.random_resized_crop_scale_min == pytest.approx(0.90)
    assert augmentation_config.random_resized_crop_probability == pytest.approx(0.30)

    normalized_b5 = list(b5_args)
    normalized_b4 = _build_b4_args(tmp_path)
    _drop_value_option(normalized_b5, "--recipe-train-contract-sha256")
    _drop_value_option(normalized_b4, "--recipe-train-contract-sha256")
    _drop_value_option(normalized_b5, "--semantic-attribute-loss-weight")
    _drop_value_option(normalized_b5, "--semantic-attribute-specs")
    normalized_b5[normalized_b5.index("--run-name") + 1] = normalized_b4[
        normalized_b4.index("--run-name") + 1
    ]
    normalized_b5[normalized_b5.index("--experiment-protocol-id") + 1] = (
        b4.PROTOCOL_ID
    )
    assert normalized_b5 == normalized_b4


def test_b5_smoke_is_bounded_and_test_locked(tmp_path: Path) -> None:
    parsed = parse_args(_build_args(tmp_path, stage="smoke"))
    assert parsed.epochs == 1
    assert parsed.scheduler_total_epochs == 5
    assert parsed.warmup_epochs == 0
    assert parsed.max_train_batches == 4
    assert parsed.max_val_batches == 2
    assert parsed.skip_final_test is True


def test_b5_contract_digest_covers_posf_and_inherited_lora(tmp_path: Path) -> None:
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
    assert "--semantic-attribute-loss-weight" in args[:digest_index]
    assert "--semantic-attribute-specs" in args[:digest_index]


def test_b5_validator_locks_only_posf_delta(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    summary = recipe.validate_b5_train_args(args)
    assert summary["protocol"] == recipe.PROTOCOL_ID
    assert summary["expected_trainable_parameters"] == 24_576
    assert summary["semantic_attribute_loss_weight"] == pytest.approx(0.006)
    assert summary["semantic_attribute_tasks"] == 3
    assert summary["test_locked"] is True

    digest_drift = list(args)
    digest_drift[-1] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        recipe.validate_b5_train_args(digest_drift)

    weight_drift = list(args)
    weight_drift[weight_drift.index("--semantic-attribute-loss-weight") + 1] = "0.01"
    _refresh_digest(weight_drift)
    with pytest.raises(ValueError, match="B5 train-argument contract mismatch"):
        recipe.validate_b5_train_args(weight_drift)

    spec_drift = list(args)
    spec_drift[spec_drift.index("--semantic-attribute-specs") + 1] = (
        "maturity:0,1|2,3|4"
    )
    _refresh_digest(spec_drift)
    with pytest.raises(ValueError, match="B5 train-argument contract mismatch"):
        recipe.validate_b5_train_args(spec_drift)

    forbidden_pauc = list(args)
    forbidden_pauc.extend(["--focus-partial-auc-loss-weight", "0.01"])
    _refresh_digest(forbidden_pauc)
    with pytest.raises(ValueError, match="B5 train-argument contract mismatch"):
        recipe.validate_b5_train_args(forbidden_pauc)


def _control_metrics() -> dict[str, object]:
    return {
        "accuracy": 0.8900,
        "macro_f1": 0.8520,
        "class1_precision": 0.612,
        "class1_recall": 0.730,
        "class1_f1": 0.666,
        "class1_tp": 115,
        "restricted_fp": 69,
        "class0_to_1": 16,
        "class2_to_1": 39,
        "class4_to_1": 14,
    }


def _passing_candidate_metrics() -> dict[str, object]:
    return {
        "accuracy": 0.8910,
        "macro_f1": 0.8530,
        "class1_precision": 0.630,
        "class1_recall": 0.720,
        "class1_f1": 0.675,
        "class1_tp": 113,
        "restricted_fp": 64,
        "class0_to_1": 15,
        "class2_to_1": 36,
        "class4_to_1": 13,
        "validation_support": 2479,
        "nonfinite_steps": 0,
        "train_semantic_attribute_tasks": 3,
        "train_semantic_attribute_loss": 0.72,
        "trainable_parameters": 24_576,
        "test_inference_performed": False,
    }


def test_b5_probe_gate_is_conjunctive_and_matched_to_b4() -> None:
    passing = recipe.evaluate_probe_gate(
        _passing_candidate_metrics(),
        _control_metrics(),
    )
    assert passing["passed"] is True
    assert all(passing["checks"].values())
    assert passing["dynamic_thresholds"]["class1_f1_min"] == pytest.approx(
        0.671
    )
    assert passing["dynamic_thresholds"]["restricted_fp_max"] == pytest.approx(
        64
    )

    failing_candidate = _passing_candidate_metrics()
    failing_candidate["class2_to_1"] = 37
    failing = recipe.evaluate_probe_gate(failing_candidate, _control_metrics())
    assert failing["passed"] is False
    assert failing["checks"]["class2_to_1_gate"] is False


def test_b5_cli_has_no_full_train_mode() -> None:
    with pytest.raises(SystemExit):
        recipe.parse_args(
            [
                "--mode",
                "full",
                "--data",
                "class_f.yaml",
                "--dino-checkpoint",
                "model.safetensors",
            ]
        )

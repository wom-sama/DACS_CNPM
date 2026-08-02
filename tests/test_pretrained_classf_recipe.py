from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import trkh.recipes.pretrained_classf_b0 as recipe
from trkh.data.dataset import TemperedClassBatchSampler
from trkh.core.utils import resolve_amp_dtype
from trkh.recipes.pretrained_classf_b0 import (
    B1_MARGIN0_PROTOCOL_ID,
    B1_NATURAL_PROTOCOL_ID,
    B2_TEMPERED_P05_PROTOCOL_ID,
    B9_B2_REFERENCE_COMPLETION_PROTOCOL_ID,
    B10_GROUP_TEMPERED_P05_PROTOCOL_ID,
    DINO_MODEL_NAME,
    DINO_SHA256,
    EXPECTED_CLASS_NAMES,
    HYBRID_V2_GENERIC_EXPECTED_PARAMETER_COUNT,
    HYBRID_V2_GENERIC_PROTOCOL_ID,
    HYBRID_V2_LOCAL_SURFACE_EXPECTED_PARAMETER_COUNT,
    HYBRID_V2_LOCAL_SURFACE_PROTOCOL_ID,
    HYBRID_V2_LOCAL_SURFACE_RANDOMINIT_PROTOCOL_ID,
    PRMR_R1_CONTROL_PROTOCOL_ID,
    PRMR_R1_PROTOCOL_ID,
    build_train_args,
    load_canonical_attestation,
    run_name,
    source_tree_sha256,
    validate_auto_resume_checkpoint,
    validate_development_data_yaml,
)
from trkh.training.train import (
    OPTIMIZER_TELEMETRY_HISTORY_FIELDS,
    build_configs,
    parse_args,
    train_one_epoch,
)


def _args(tmp_path: Path, *, stage: str = "full") -> list[str]:
    return build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage=stage,
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
    )


def _strip_lineage_only_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    skip_value_for = {
        "--experiment-protocol-id",
        "--recipe-train-contract-sha256",
        "--run-name",
    }
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in skip_value_for:
            index += 2
            continue
        normalized.append(argument)
        index += 1
    return normalized


def _strip_hybrid_v2_model_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    skip_value_for = {
        "--model-type",
        "--dinov3-surface-hybrid-mode",
        "--dinov3-surface-initial-gate-scale",
        "--dinov3-surface-max-gate-scale",
    }
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in skip_value_for:
            index += 2
            continue
        normalized.append(argument)
        index += 1
    return normalized


def _strip_external_initialization_args(args: list[str]) -> list[str]:
    normalized = list(args)
    if "--pretrained" in normalized:
        normalized.remove("--pretrained")
    for option in (
        "--pretrained-checkpoint-path",
        "--pretrained-checkpoint-sha256",
        "--pretrained-source-url",
        "--pretrained-source-revision",
        "--pretrained-source-license",
    ):
        if option in normalized:
            index = normalized.index(option)
            del normalized[index : index + 2]
    return normalized


def test_classf_b0_recipe_is_direct_pretrained_hard_label_only(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    parsed = parse_args(args)

    assert parsed.model_type == "timm_classifier"
    assert parsed.timm_model_name == DINO_MODEL_NAME
    assert parsed.research_track == "pretrained"
    assert parsed.pretrained is True
    assert parsed.pretrained_checkpoint_path.name == "model.safetensors"
    assert parsed.image_size == 256
    assert parsed.batch_size == 16
    assert parsed.grad_accum_steps == 3
    assert parsed.skip_final_test is True
    assert parsed.distillation_teacher_csv is None
    assert parsed.pretrained_distillation is False
    assert parsed.distillation_weight == 0.0
    assert parsed.teacher_focus_binary_loss_weight == 0.0
    assert parsed.max_train_batches == 0
    assert parsed.max_val_batches == 0
    assert parsed.resume is None
    assert parsed.classification_folder_yolo_data is None
    assert "yolo_f" not in " ".join(args).lower()


def test_classf_b0_smoke_is_bounded_but_keeps_test_locked(
    tmp_path: Path,
) -> None:
    parsed = parse_args(_args(tmp_path, stage="smoke"))

    assert parsed.epochs == 1
    assert parsed.scheduler_total_epochs == 5
    assert parsed.max_train_batches == 4
    assert parsed.max_val_batches == 2
    assert parsed.skip_final_test is True


def test_amp_init_scale_default_preserves_recipe_contract(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    implicit_args = _args(tmp_path)
    explicit_none_args = build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="full",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
        amp_init_scale=None,
    )
    parsed = parse_args(implicit_args)
    _, train_config, _ = build_configs(parsed)

    assert implicit_args == explicit_none_args
    assert "--amp-init-scale" not in implicit_args
    assert parsed.amp_init_scale == pytest.approx(65536.0)
    assert train_config.amp_init_scale == pytest.approx(65536.0)


def test_amp_init_scale_is_hashed_before_recipe_contract(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    default_args = _args(tmp_path)
    custom_args = build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="full",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
        amp_init_scale=1024,
    )
    scale_index = custom_args.index("--amp-init-scale")
    contract_index = custom_args.index("--recipe-train-contract-sha256")
    observed_contract = custom_args[contract_index + 1]
    expected_contract = hashlib.sha256(
        json.dumps(
            custom_args[:contract_index],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    parsed = parse_args(custom_args)
    _, train_config, _ = build_configs(parsed)

    assert scale_index < contract_index
    assert custom_args[scale_index + 1] == "1024.0"
    assert observed_contract == expected_contract
    assert observed_contract != default_args[
        default_args.index("--recipe-train-contract-sha256") + 1
    ]
    assert parsed.amp_init_scale == pytest.approx(1024.0)
    assert train_config.amp_init_scale == pytest.approx(1024.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_amp_init_scale_rejects_non_positive_or_nonfinite_values(
    tmp_path: Path,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="amp_init_scale must be finite and > 0"):
        build_train_args(
            data_yaml=tmp_path / "class_f" / "data.yaml",
            dino_checkpoint=tmp_path / "model.safetensors",
            output_dir=tmp_path / "runs",
            stage="full",
            run_tag="unit",
            amp_init_scale=value,
        )


def test_amp_dtype_is_explicit_in_recipe_hash_and_train_config(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    common = {
        "data_yaml": tmp_path / "data.yaml",
        "dino_checkpoint": tmp_path / "model.safetensors",
        "output_dir": tmp_path / "runs",
        "stage": "probe",
        "run_tag": "amp_dtype_unit",
        "experiment": "b10-group-tempered-p05",
    }
    bf16_args = build_train_args(**common, amp_dtype="bf16")
    fp16_args = build_train_args(**common, amp_dtype="fp16")
    bf16 = parse_args(bf16_args)
    _, bf16_train, _ = build_configs(bf16)
    fp16 = parse_args(fp16_args)

    assert bf16_args[bf16_args.index("--amp-dtype") + 1] == "bf16"
    assert "--deterministic" in bf16_args
    assert bf16.amp_dtype == "bf16"
    assert bf16_train.amp_dtype == "bf16"
    assert bf16_train.deterministic is True
    assert fp16.amp_dtype == "fp16"
    assert bf16.recipe_train_contract_sha256 != fp16.recipe_train_contract_sha256


def test_random_init_arm_does_not_require_or_hash_unused_dino_file(
    tmp_path: Path,
) -> None:
    args = build_train_args(
        data_yaml=tmp_path / "data.yaml",
        dino_checkpoint=None,
        output_dir=tmp_path / "runs",
        stage="smoke",
        run_tag="random_without_weights",
        experiment="hybrid-v2-local-surface-randominit",
    )
    parsed = parse_args(args)

    assert parsed.pretrained is False
    assert parsed.pretrained_checkpoint_path is None
    assert parsed.pretrained_checkpoint_sha256 == ""
    assert "--pretrained-source-url" not in args


def test_explicit_bf16_fails_closed_on_unsupported_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRKH_AMP_DTYPE", "bf16")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="does not support BF16"):
        resolve_amp_dtype(torch.device("cuda"))


def test_train_epoch_reports_successful_optimizer_updates() -> None:
    assert OPTIMIZER_TELEMETRY_HISTORY_FIELDS == (
        "train_optimizer_step_attempts",
        "train_optimizer_updates_successful",
        "train_optimizer_steps_skipped_nonfinite",
        "train_nonfinite_loss_batches",
        "train_amp_optimizer_steps_skipped",
    )
    torch.manual_seed(1)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [
        (torch.tensor([[1.0, 0.0]]), torch.tensor([0])),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([1])),
    ]

    _, stats, _ = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device=torch.device("cpu"),
        amp=False,
        grad_clip_norm=1.0,
        epoch_index=1,
        grad_accum_steps=2,
    )

    assert stats["optimizer_step_attempts"] == 1.0
    assert stats["optimizer_updates_successful"] == 1.0
    assert stats["optimizer_steps_skipped_nonfinite"] == 0.0
    assert stats["nonfinite_loss_batches"] == 0.0


def test_train_epoch_reports_nonfinite_gradient_skip() -> None:
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _, stats, _ = train_one_epoch(
        model=model,
        dataloader=[(torch.ones(1, 2), torch.tensor([0]))],
        criterion=InfGradCriterion(),
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device=torch.device("cpu"),
        amp=False,
        grad_clip_norm=1.0,
        epoch_index=1,
        max_nonfinite_grad_steps=0,
    )

    assert stats["optimizer_step_attempts"] == 1.0
    assert stats["optimizer_updates_successful"] == 0.0
    assert stats["optimizer_steps_skipped_nonfinite"] == 1.0
    assert stats["nonfinite_loss_batches"] == 0.0


def test_classf_b0_run_name_is_versioned() -> None:
    assert (
        run_name("full", "20260731")
        == "pretrained_dinov3_classf_direct_full_20260731"
    )


def test_classf_b1_natural_changes_only_sampler_and_lineage(
    tmp_path: Path,
) -> None:
    b0_args = _args(tmp_path, stage="probe")
    b1_args = build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
        experiment="b1-natural",
    )
    b0 = parse_args(b0_args)
    b1 = parse_args(b1_args)

    assert b0.disable_balanced_epoch_sampling is False
    assert b1.disable_balanced_epoch_sampling is True
    assert b1.experiment_protocol_id == B1_NATURAL_PROTOCOL_ID
    assert b1.classification_loss == b0.classification_loss == "ldam_focal"
    assert b1.focal_loss_gamma == b0.focal_loss_gamma == 1.0
    assert b1.focal_loss_mix == b0.focal_loss_mix == 0.1
    assert b1.ldam_max_margin == b0.ldam_max_margin == 0.3
    assert b1.ldam_scale == b0.ldam_scale == 18.0
    assert b1.disable_class_weights is b0.disable_class_weights is True
    assert b1.max_train_batches == b0.max_train_batches == 120
    assert (
        _strip_lineage_only_args(b1_args)
        == _strip_lineage_only_args(b0_args)
        + ["--disable-balanced-epoch-sampling"]
    )
    assert (
        run_name("probe", "unit", experiment="b1-natural")
        == "pretrained_dinov3_classf_natural_probe_unit"
    )


def test_classf_b1_margin0_changes_only_ldam_margin_and_lineage(
    tmp_path: Path,
) -> None:
    b0_args = _args(tmp_path, stage="probe")
    margin_args = build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
        experiment="b1-margin0",
    )
    b0 = parse_args(b0_args)
    margin = parse_args(margin_args)

    assert margin.disable_balanced_epoch_sampling is False
    assert margin.experiment_protocol_id == B1_MARGIN0_PROTOCOL_ID
    assert margin.classification_loss == b0.classification_loss == "ldam_focal"
    assert margin.focal_loss_gamma == b0.focal_loss_gamma == 1.0
    assert margin.focal_loss_mix == b0.focal_loss_mix == 0.1
    assert margin.ldam_max_margin == 0.0
    assert b0.ldam_max_margin == 0.3
    assert margin.ldam_scale == b0.ldam_scale == 18.0
    assert margin.disable_ldam is b0.disable_ldam is False
    expected_args = _strip_lineage_only_args(b0_args)
    margin_index = expected_args.index("--ldam-max-margin") + 1
    expected_args[margin_index] = "0.0"
    assert _strip_lineage_only_args(margin_args) == expected_args
    assert (
        run_name("probe", "unit", experiment="b1-margin0")
        == "pretrained_dinov3_classf_margin0_probe_unit"
    )


def test_classf_b2_tempered_changes_only_train_sampling_prior_and_lineage(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    b0_args = _args(tmp_path, stage="probe")
    tempered_args = build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
        experiment="b2-tempered-p05",
    )
    b0 = parse_args(b0_args)
    tempered = parse_args(tempered_args)
    _, tempered_train_config, _ = build_configs(tempered)

    assert tempered.disable_balanced_epoch_sampling is False
    assert tempered.tempered_class_sampling_power == pytest.approx(0.5)
    assert tempered_train_config.balanced_epoch_sampling is False
    assert tempered_train_config.tempered_class_sampling_power == pytest.approx(0.5)
    assert tempered.experiment_protocol_id == B2_TEMPERED_P05_PROTOCOL_ID
    assert tempered.classification_loss == b0.classification_loss == "ldam_focal"
    assert tempered.focal_loss_gamma == b0.focal_loss_gamma == 1.0
    assert tempered.focal_loss_mix == b0.focal_loss_mix == 0.1
    assert tempered.ldam_max_margin == b0.ldam_max_margin == 0.3
    assert tempered.ldam_scale == b0.ldam_scale == 18.0
    assert tempered.disable_class_weights is b0.disable_class_weights is True
    assert tempered.max_train_batches == b0.max_train_batches == 120
    assert (
        _strip_lineage_only_args(tempered_args)
        == _strip_lineage_only_args(b0_args)
        + ["--tempered-class-sampling-power", "0.5"]
    )
    assert (
        run_name("probe", "unit", experiment="b2-tempered-p05")
        == "pretrained_dinov3_classf_tempered_p05_probe_unit"
    )


def test_classf_b2_tempered_probe_exposure_is_locked() -> None:
    counts = [1987, 497, 1326, 2080, 2388]
    labels = [
        class_index
        for class_index, class_count in enumerate(counts)
        for _ in range(class_count)
    ]
    sampler = TemperedClassBatchSampler(
        labels=labels,
        batch_size=24,
        num_classes=5,
        power=0.5,
        seed=42,
    )
    summary = sampler.exposure_summary(num_batches=120)

    assert summary["class_exposure_counts"] == [649, 325, 530, 664, 712]
    assert summary["total_samples"] == 2880
    assert summary["max_prefix_absolute_quota_error"] < 1.0


def test_b9_reference_completion_is_b2_training_semantic_clone(
    tmp_path: Path,
) -> None:
    common = {
        "data_yaml": tmp_path / "class_f" / "data.yaml",
        "dino_checkpoint": tmp_path / "model.safetensors",
        "output_dir": tmp_path / "runs",
        "stage": "full",
        "run_tag": "unit",
        "batch_size": 16,
        "num_workers": 2,
        "eval_num_workers": 1,
        "seed": 42,
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "dataset_image_tree_sha256": "c" * 64,
    }
    b2_args = build_train_args(**common, experiment="b2-tempered-p05")
    b9_args = build_train_args(
        **common, experiment="b9-b2-reference-completion"
    )
    b2 = parse_args(b2_args)
    b9 = parse_args(b9_args)

    # The complete trainer command is byte-for-byte equivalent after removing
    # only the deliberately distinct experiment identity/provenance fields.
    assert _strip_lineage_only_args(b9_args) == _strip_lineage_only_args(
        b2_args
    )
    assert (
        b9.tempered_class_sampling_power
        == b2.tempered_class_sampling_power
        == 0.5
    )
    assert (
        b9.disable_balanced_epoch_sampling
        is b2.disable_balanced_epoch_sampling
        is False
    )
    assert b9.classification_loss == b2.classification_loss == "ldam_focal"
    assert b9.ldam_max_margin == b2.ldam_max_margin == 0.3
    assert b9.epochs == b2.epochs == 30
    assert b9.scheduler_total_epochs == b2.scheduler_total_epochs == 30
    assert b9.skip_final_test is b2.skip_final_test is True
    assert b9.experiment_protocol_id == B9_B2_REFERENCE_COMPLETION_PROTOCOL_ID
    assert b9.experiment_protocol_id != B2_TEMPERED_P05_PROTOCOL_ID
    assert run_name(
        "full", "unit", experiment="b9-b2-reference-completion"
    ) == "pretrained_dinov3_classf_b9_b2_reference_completion_full_unit"


def test_b9_reference_completion_provenance_is_explicitly_non_promotable() -> None:
    b2 = recipe.experiment_spec("b2-tempered-p05")
    b9 = recipe.experiment_spec("b9-b2-reference-completion")

    assert b9.reference_experiment == b2.key
    assert b9.research_role == "exploratory_canonical_reference_completion"
    assert b9.promotion_eligible is False
    assert b9.full_train_authorized is True
    assert b9.known_probe_class1_f1 == pytest.approx(0.653409)
    assert b9.required_probe_class1_f1 == pytest.approx(0.66)
    assert b9.known_probe_class1_f1 < b9.required_probe_class1_f1
    assert (
        b9.balanced_epoch_sampling,
        b9.tempered_class_sampling_power,
        b9.ldam_max_margin,
        b9.single_semantic_delta_from_b0,
    ) == (
        b2.balanced_epoch_sampling,
        b2.tempered_class_sampling_power,
        b2.ldam_max_margin,
        b2.single_semantic_delta_from_b0,
    )


def test_b10_group_tempered_is_single_train_sampling_delta_and_probe_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    common = {
        "data_yaml": tmp_path / "class_f" / "data.yaml",
        "dino_checkpoint": tmp_path / "model.safetensors",
        "output_dir": tmp_path / "runs",
        "stage": "probe",
        "run_tag": "unit",
        "batch_size": 16,
        "num_workers": 2,
        "eval_num_workers": 1,
    }
    control_args = build_train_args(
        **common,
        experiment="b2-tempered-p05",
    )
    candidate_args = build_train_args(
        **common,
        experiment="b10-group-tempered-p05",
    )
    control = parse_args(control_args)
    candidate = parse_args(candidate_args)
    _, candidate_config, _ = build_configs(candidate)
    candidate_spec = recipe.experiment_spec("b10-group-tempered-p05")
    manifest_path = str(
        (common["data_yaml"].resolve().parent / "manifest.csv").resolve()
    )

    assert candidate.experiment_protocol_id == B10_GROUP_TEMPERED_P05_PROTOCOL_ID
    assert candidate_spec.reference_experiment == "b2-tempered-p05"
    assert candidate_spec.tempered_leakage_group_sampling is True
    assert candidate_spec.full_train_authorized is False
    assert candidate_spec.promotion_eligible is False
    assert control.tempered_leakage_group_manifest == ""
    assert candidate.tempered_leakage_group_manifest == manifest_path
    assert candidate_config.tempered_leakage_group_manifest == manifest_path
    assert candidate.tempered_class_sampling_power == control.tempered_class_sampling_power == 0.5
    assert _strip_lineage_only_args(candidate_args) == (
        _strip_lineage_only_args(control_args)
        + ["--tempered-leakage-group-manifest", manifest_path]
    )
    assert run_name(
        "probe",
        "unit",
        experiment="b10-group-tempered-p05",
    ) == "pretrained_dinov3_classf_b10_group_tempered_p05_probe_unit"


def test_hybrid_v2_probe_pair_inherits_b2_and_differs_only_by_model_mode(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"unit-test-placeholder")
    common = {
        "data_yaml": tmp_path / "class_f" / "data.yaml",
        "dino_checkpoint": checkpoint,
        "output_dir": tmp_path / "runs",
        "stage": "probe",
        "run_tag": "unit",
        "batch_size": 16,
        "num_workers": 2,
        "eval_num_workers": 1,
    }
    b2_args = build_train_args(
        **common,
        experiment="b2-tempered-p05",
    )
    generic_args = build_train_args(
        **common,
        experiment="hybrid-v2-generic",
    )
    local_args = build_train_args(
        **common,
        experiment="hybrid-v2-local-surface",
    )
    random_args = build_train_args(
        **common,
        experiment="hybrid-v2-local-surface-randominit",
    )
    generic = parse_args(generic_args)
    local = parse_args(local_args)
    random_control = parse_args(random_args)
    _, generic_train, _ = build_configs(generic)
    _, local_train, _ = build_configs(local)
    generic_spec = recipe.experiment_spec("hybrid-v2-generic")
    local_spec = recipe.experiment_spec("hybrid-v2-local-surface")
    random_spec = recipe.experiment_spec(
        "hybrid-v2-local-surface-randominit"
    )
    assert generic.experiment_protocol_id == HYBRID_V2_GENERIC_PROTOCOL_ID
    assert local.experiment_protocol_id == HYBRID_V2_LOCAL_SURFACE_PROTOCOL_ID
    assert random_control.experiment_protocol_id == (
        HYBRID_V2_LOCAL_SURFACE_RANDOMINIT_PROTOCOL_ID
    )
    assert generic.model_type == local.model_type == (
        "dinov3_surface_patch_hybrid_v2"
    )
    assert generic.dinov3_surface_hybrid_mode == "generic_token_adapter"
    assert local.dinov3_surface_hybrid_mode == "local_surface"
    assert generic.dinov3_surface_initial_gate_scale == pytest.approx(0.05)
    assert local.dinov3_surface_initial_gate_scale == pytest.approx(0.05)
    assert generic.dinov3_surface_max_gate_scale == pytest.approx(0.25)
    assert local.dinov3_surface_max_gate_scale == pytest.approx(0.25)
    assert generic.pretrained is local.pretrained is True
    assert random_control.pretrained is False
    assert str(random_control.pretrained_checkpoint_path or "") == ""
    assert random_control.pretrained_checkpoint_sha256 == ""
    assert generic.timm_model_name == local.timm_model_name == DINO_MODEL_NAME
    assert generic.pretrained_checkpoint_path == checkpoint.absolute()
    assert local.pretrained_checkpoint_path == checkpoint.absolute()
    assert generic.pretrained_checkpoint_sha256 == local.pretrained_checkpoint_sha256 == DINO_SHA256
    assert generic.pretrained_source_url == local.pretrained_source_url == recipe.DINO_SOURCE_URL
    assert generic.pretrained_source_revision == local.pretrained_source_revision == recipe.DINO_SOURCE_REVISION
    assert generic.pretrained_source_license == local.pretrained_source_license == recipe.DINO_SOURCE_LICENSE
    assert generic.tempered_class_sampling_power == local.tempered_class_sampling_power == pytest.approx(0.5)
    assert generic.tempered_leakage_group_manifest == ""
    assert local.tempered_leakage_group_manifest == ""
    assert generic_train.tempered_leakage_group_manifest == ""
    assert local_train.tempered_leakage_group_manifest == ""

    # Removing only identity and architecture selectors leaves the complete
    # optimizer/loss/augmentation/sampler schedule byte-for-byte matched to B2.
    normalized_b2 = _strip_hybrid_v2_model_args(
        _strip_lineage_only_args(b2_args)
    )
    assert _strip_hybrid_v2_model_args(
        _strip_lineage_only_args(generic_args)
    ) == normalized_b2
    assert _strip_hybrid_v2_model_args(
        _strip_lineage_only_args(local_args)
    ) == normalized_b2
    assert _strip_external_initialization_args(
        _strip_lineage_only_args(random_args)
    ) == _strip_external_initialization_args(
        _strip_lineage_only_args(local_args)
    )

    assert generic_spec.reference_experiment == "b2-tempered-p05"
    assert local_spec.reference_experiment == "b2-tempered-p05"
    assert generic_spec.tempered_leakage_group_sampling is False
    assert local_spec.tempered_leakage_group_sampling is False
    assert generic_spec.full_train_authorized is False
    assert local_spec.full_train_authorized is False
    assert generic_spec.promotion_eligible is False
    assert local_spec.promotion_eligible is False
    assert random_spec.reference_experiment == "hybrid-v2-local-surface"
    assert random_spec.external_initialization_used is False
    assert random_spec.full_train_authorized is False
    assert generic_spec.expected_parameter_count == (
        HYBRID_V2_GENERIC_EXPECTED_PARAMETER_COUNT
    )
    assert local_spec.expected_parameter_count == (
        HYBRID_V2_LOCAL_SURFACE_EXPECTED_PARAMETER_COUNT
    )


def test_group_manifest_can_be_bound_to_canonical_root_with_dev_yaml(
    tmp_path: Path,
) -> None:
    canonical_manifest = tmp_path / "canonical" / "manifest.csv"
    args = build_train_args(
        data_yaml=tmp_path / "configs" / "class_f_5class_dev.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="unit",
        experiment="b10-group-tempered-p05",
        tempered_leakage_group_manifest=canonical_manifest,
    )

    parsed = parse_args(args)
    assert parsed.tempered_leakage_group_manifest == str(
        canonical_manifest.resolve()
    )


def test_hybrid_v2_auto_resume_validates_model_mode_and_gate(
    tmp_path: Path,
) -> None:
    data_yaml = tmp_path / "class_f_dev.yaml"
    data_yaml.write_text("format: classification_folder\n", encoding="utf-8")
    expected_args = build_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="resume_hybrid",
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        dataset_image_tree_sha256="c" * 64,
        experiment="hybrid-v2-local-surface",
    )
    expected = parse_args(expected_args)
    train_config = {
        "experiment_protocol_id": expected.experiment_protocol_id,
        "source_commit": expected.source_commit,
        "source_tree_sha256": expected.source_tree_sha256,
        "dataset_image_tree_sha256": expected.dataset_image_tree_sha256,
        "recipe_train_contract_sha256": expected.recipe_train_contract_sha256,
    }
    for field in (
        "illumination_consistency_loss_weight",
        "illumination_consistency_probability",
        "illumination_consistency_brightness",
        "illumination_consistency_contrast",
        "illumination_consistency_gamma",
        "illumination_consistency_temperature",
        "illumination_consistency_mode",
        "illumination_consistency_focus_class",
        "illumination_consistency_negative_classes",
        "illumination_consistency_margin_retention",
        "illumination_consistency_start_epoch",
    ):
        train_config[field] = getattr(expected, field)
    checkpoint = {
        "class_names": list(EXPECTED_CLASS_NAMES),
        "data_yaml": str(data_yaml),
        "model_config": {
            "model_type": "dinov3_surface_patch_hybrid_v2",
            "research_track": "pretrained",
            "pretrained": True,
            "timm_model_name": DINO_MODEL_NAME,
            "pretrained_checkpoint_sha256": DINO_SHA256,
            "dinov3_surface_hybrid_mode": "local_surface",
            "dinov3_surface_initial_gate_scale": 0.05,
            "dinov3_surface_max_gate_scale": 0.25,
        },
        "train_config": train_config,
        "epoch": 2,
    }
    checkpoint_path = tmp_path / "hybrid_last.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = validate_auto_resume_checkpoint(
        checkpoint_path,
        training_data_yaml=data_yaml,
        expected_train_args=expected_args,
    )
    assert summary["model_type"] == "dinov3_surface_patch_hybrid_v2"

    checkpoint["model_config"]["dinov3_surface_hybrid_mode"] = (
        "generic_token_adapter"
    )
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="pretrained contract mismatch"):
        validate_auto_resume_checkpoint(
            checkpoint_path,
            training_data_yaml=data_yaml,
            expected_train_args=expected_args,
        )


def test_pretrained_checkpoint_sha_extraction_supports_direct_and_hybrid_schema() -> None:
    direct = torch.nn.Linear(2, 2)
    direct.pretrained_provenance = {
        "initialization": {"checkpoint": {"sha256": DINO_SHA256}}
    }
    hybrid = torch.nn.Linear(2, 2)
    hybrid.pretrained_provenance = {
        "primary_backbone": {
            "initialization": {"checkpoint": {"sha256": DINO_SHA256}}
        }
    }

    assert recipe._pretrained_checkpoint_sha256(direct) == DINO_SHA256
    assert recipe._pretrained_checkpoint_sha256(hybrid) == DINO_SHA256

    hybrid.pretrained_provenance["initialization"] = {
        "checkpoint": {"sha256": "f" * 64}
    }
    with pytest.raises(RuntimeError, match="conflicting pretrained checkpoint"):
        recipe._pretrained_checkpoint_sha256(hybrid)


def test_prmr_r1_probe_pair_is_matched_and_train_only(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    common = dict(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
    )
    control_args = build_train_args(**common, experiment="prmr-r1-control")
    candidate_args = build_train_args(**common, experiment="prmr-r1")
    control = parse_args(control_args)
    candidate = parse_args(candidate_args)
    _, candidate_config, _ = build_configs(candidate)

    assert control.experiment_protocol_id == PRMR_R1_CONTROL_PROTOCOL_ID
    assert candidate.experiment_protocol_id == PRMR_R1_PROTOCOL_ID
    assert control.tempered_class_sampling_power == pytest.approx(0.5)
    assert candidate.tempered_class_sampling_power == pytest.approx(0.5)
    assert control.illumination_consistency_loss_weight == 0.0
    assert control.illumination_consistency_mode == "symmetric_kl"
    assert candidate.illumination_consistency_loss_weight == pytest.approx(0.15)
    assert candidate.illumination_consistency_probability == pytest.approx(0.5)
    assert candidate.illumination_consistency_brightness == pytest.approx(0.25)
    assert candidate.illumination_consistency_contrast == pytest.approx(0.10)
    assert candidate.illumination_consistency_gamma == 0.0
    assert candidate.illumination_consistency_mode == "pairwise_margin_retention"
    assert candidate.illumination_consistency_focus_class == 1
    assert candidate.illumination_consistency_negative_classes == "0,2,4"
    assert candidate.illumination_consistency_margin_retention == pytest.approx(0.8)
    assert candidate.illumination_consistency_start_epoch == 3
    assert candidate_config.illumination_consistency_mode == "pairwise_margin_retention"
    assert candidate_config.illumination_consistency_start_epoch == 3
    assert candidate.skip_final_test is control.skip_final_test is True
    assert _strip_lineage_only_args(candidate_args) == (
        _strip_lineage_only_args(control_args)
        + list(recipe.experiment_spec("prmr-r1").extra_train_args)
    )
    assert recipe.experiment_spec("prmr-r1").full_train_authorized is False
    assert recipe.experiment_spec("prmr-r1-control").full_train_authorized is False

    smoke_args = build_train_args(
        **{**common, "stage": "smoke"},
        experiment="prmr-r1",
    )
    smoke = parse_args(smoke_args)
    assert smoke.illumination_consistency_start_epoch == 1


def test_prmr_config_canonicalizes_rivals_and_rejects_ignored_transform_args(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"unit-test-placeholder")
    candidate_args = build_train_args(
        data_yaml=tmp_path / "class_f" / "data.yaml",
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="unit",
        batch_size=16,
        num_workers=2,
        eval_num_workers=1,
        experiment="prmr-r1",
    )

    semicolon_args = list(candidate_args)
    rival_index = semicolon_args.index(
        "--illumination-consistency-negative-classes"
    ) + 1
    semicolon_args[rival_index] = "0;2;4"
    _, normalized, _ = build_configs(parse_args(semicolon_args))
    assert normalized.illumination_consistency_negative_classes == "0,2,4"

    gamma_args = list(candidate_args)
    gamma_index = gamma_args.index("--illumination-consistency-gamma") + 1
    gamma_args[gamma_index] = "0.1"
    with pytest.raises(ValueError, match="khong dung gamma"):
        build_configs(parse_args(gamma_args))

    brightness_args = list(candidate_args)
    brightness_index = brightness_args.index(
        "--illumination-consistency-brightness"
    ) + 1
    brightness_args[brightness_index] = "0.96"
    with pytest.raises(ValueError, match="brightness"):
        build_configs(parse_args(brightness_args))

    nonfinite_args = list(candidate_args)
    probability_index = nonfinite_args.index(
        "--illumination-consistency-probability"
    ) + 1
    nonfinite_args[probability_index] = "nan"
    with pytest.raises(ValueError, match="must be finite"):
        build_configs(parse_args(nonfinite_args))


def test_full_train_authorization_is_fail_closed() -> None:
    b0 = recipe.experiment_spec("b0")
    b1_natural = recipe.experiment_spec("b1-natural")
    b1_margin0 = recipe.experiment_spec("b1-margin0")
    b2 = recipe.experiment_spec("b2-tempered-p05")
    b9 = recipe.experiment_spec("b9-b2-reference-completion")

    assert b0.research_role == "matched_pretrained_backbone_control"
    assert b0.full_train_authorized is True
    assert b0.promotion_eligible is False
    assert b9.full_train_authorized is True
    for closed in (b1_natural, b1_margin0, b2):
        assert closed.full_train_authorized is False
        assert closed.promotion_eligible is False
    assert b2.research_role == "closed_probe_hypothesis"
    assert b2.known_probe_class1_f1 == pytest.approx(0.653409)
    assert b2.required_probe_class1_f1 == pytest.approx(0.66)


def test_closed_b2_full_and_b9_probe_are_rejected_before_io(
    tmp_path: Path,
) -> None:
    common = [
        "--data",
        str(tmp_path / "data.yaml"),
        "--dino-checkpoint",
        str(tmp_path / "model.safetensors"),
    ]
    with pytest.raises(RuntimeError, match="full training is closed"):
        recipe.main(
            [
                "--experiment",
                "b2-tempered-p05",
                "--mode",
                "full",
                "--confirm-full",
                *common,
            ]
        )
    with pytest.raises(RuntimeError, match="only preflight or full"):
        recipe.main(
            [
                "--experiment",
                "b9-b2-reference-completion",
                "--mode",
                "probe",
                *common,
            ]
        )


def test_classf_b0_development_yaml_is_test_locked_and_canonical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "class_f"
    root.mkdir()
    data_yaml = tmp_path / "class_f_dev.yaml"
    names = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(EXPECTED_CLASS_NAMES)
    )
    data_yaml.write_text(
        "\n".join(
            (
                "format: classification_folder",
                f"path: {root.as_posix()}",
                "train: train",
                "val: val",
                "nc: 5",
                "class_name_mode: raw",
                "names:",
                names,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeDataset:
        def __init__(self, counts: list[int]) -> None:
            self._counts = counts

        def class_counts(self, num_classes: int) -> list[int]:
            assert num_classes == 5
            return list(self._counts)

        def __len__(self) -> int:
            return sum(self._counts)

    monkeypatch.setattr(
        recipe.ClassificationFolderDataset,
        "from_data_spec",
        classmethod(
            lambda _cls, _spec, split: _FakeDataset(
                list(recipe.EXPECTED_DEVELOPMENT_COUNTS[split])
            )
        ),
    )
    contract = validate_development_data_yaml(
        data_yaml,
        canonical_contract={"root": str(root)},
    )

    assert contract["has_test_split"] is False
    assert contract["split_totals"] == {"train": 8278, "val": 2479}


def test_classf_prmr_r1_resume_checkpoint_is_bound_to_full_semantic_contract(
    tmp_path: Path,
) -> None:
    data_yaml = tmp_path / "class_f_dev.yaml"
    data_yaml.write_text("format: classification_folder\n", encoding="utf-8")
    expected_args = build_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="resume_unit",
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        dataset_image_tree_sha256="c" * 64,
        experiment="prmr-r1",
    )
    expected = parse_args(expected_args)
    checkpoint_path = tmp_path / "last.pt"
    torch.save(
        {
            "class_names": list(EXPECTED_CLASS_NAMES),
            "data_yaml": str(data_yaml),
            "model_config": {
                "model_type": "timm_classifier",
                "research_track": "pretrained",
                "pretrained": True,
                "timm_model_name": DINO_MODEL_NAME,
                "pretrained_checkpoint_sha256": DINO_SHA256,
            },
            "train_config": {
                "experiment_protocol_id": expected.experiment_protocol_id,
                "source_commit": expected.source_commit,
                "source_tree_sha256": expected.source_tree_sha256,
                "dataset_image_tree_sha256": expected.dataset_image_tree_sha256,
                "recipe_train_contract_sha256": (
                    expected.recipe_train_contract_sha256
                ),
                "illumination_consistency_loss_weight": (
                    expected.illumination_consistency_loss_weight
                ),
                "illumination_consistency_probability": (
                    expected.illumination_consistency_probability
                ),
                "illumination_consistency_brightness": (
                    expected.illumination_consistency_brightness
                ),
                "illumination_consistency_contrast": (
                    expected.illumination_consistency_contrast
                ),
                "illumination_consistency_gamma": (
                    expected.illumination_consistency_gamma
                ),
                "illumination_consistency_temperature": (
                    expected.illumination_consistency_temperature
                ),
                "illumination_consistency_mode": (
                    expected.illumination_consistency_mode
                ),
                "illumination_consistency_focus_class": (
                    expected.illumination_consistency_focus_class
                ),
                "illumination_consistency_negative_classes": (
                    expected.illumination_consistency_negative_classes
                ),
                "illumination_consistency_margin_retention": (
                    expected.illumination_consistency_margin_retention
                ),
                "illumination_consistency_start_epoch": (
                    expected.illumination_consistency_start_epoch
                ),
            },
            "epoch": 3,
        },
        checkpoint_path,
    )

    summary = validate_auto_resume_checkpoint(
        checkpoint_path,
        training_data_yaml=data_yaml,
        expected_train_args=expected_args,
    )

    assert summary["epoch"] == 3
    assert summary["pretrained_checkpoint_sha256"] == DINO_SHA256
    assert summary["lineage"]["source_commit"] == "a" * 40
    assert summary["illumination_contract"]["illumination_consistency_mode"] == (
        "pairwise_margin_retention"
    )

    mismatched_args = build_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="probe",
        run_tag="resume_unit",
        source_commit="d" * 40,
        source_tree_sha256="b" * 64,
        dataset_image_tree_sha256="c" * 64,
        experiment="prmr-r1",
    )
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_auto_resume_checkpoint(
            checkpoint_path,
            training_data_yaml=data_yaml,
            expected_train_args=mismatched_args,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["train_config"]["illumination_consistency_margin_retention"] = 0.7
    tampered_checkpoint_path = tmp_path / "tampered_last.pt"
    torch.save(checkpoint, tampered_checkpoint_path)
    with pytest.raises(ValueError, match="semantic mismatch"):
        validate_auto_resume_checkpoint(
            tampered_checkpoint_path,
            training_data_yaml=data_yaml,
            expected_train_args=expected_args,
        )


def test_classf_b0_source_tree_digest_is_stable_shape() -> None:
    digest = source_tree_sha256(Path(__file__).resolve().parents[1])

    assert len(digest) == 64
    int(digest, 16)


def test_classf_b0_source_tree_digest_is_line_ending_portable(
    tmp_path: Path,
) -> None:
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    for root, newline in ((lf_root, b"\n"), (crlf_root, b"\r\n")):
        (root / "trkh").mkdir(parents=True)
        (root / "configs").mkdir()
        (root / "trkh" / "module.py").write_bytes(
            newline.join((b"x = 1", b"y = 2", b""))
        )
        (root / "configs" / "data.yaml").write_bytes(
            newline.join((b"nc: 5", b"train: train", b""))
        )

    assert source_tree_sha256(lf_root) == source_tree_sha256(crlf_root)


def test_classf_b0_reuses_only_exact_canonical_attestation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_yaml = tmp_path / "class_f" / "data.yaml"
    data_yaml.parent.mkdir()
    data_yaml.write_text("canonical\n", encoding="utf-8")
    (data_yaml.parent / "manifest.csv").write_text("manifest\n", encoding="utf-8")
    (data_yaml.parent / "stats.json").write_text("{}\n", encoding="utf-8")
    file_hashes = {
        name: recipe._sha256(data_yaml.parent / name)
        for name in ("data.yaml", "manifest.csv", "stats.json")
    }
    monkeypatch.setattr(recipe, "CANONICAL_FILE_SHA256", file_hashes)
    monkeypatch.setattr(recipe, "CANONICAL_IMAGE_TREE_SHA256", "a" * 64)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        recipe.json.dumps(
            {
                "contract": "TRKH_CLASS_F_CANONICAL_V1_20260730",
                "status": "passed",
                "data_yaml": str(data_yaml.resolve()),
                "file_sha256": file_hashes,
                "image_tree_sha256": "a" * 64,
                "image_tree_file_count": 12019,
                "test_model_inference_performed": False,
                "test_metrics_read": False,
            }
        ),
        encoding="utf-8",
    )

    payload = load_canonical_attestation(attestation, data_yaml=data_yaml)
    assert payload["reused_without_rehashing_images"] is True
    assert payload["attestation_sha256"] == recipe._sha256(attestation)

    document = recipe.json.loads(attestation.read_text(encoding="utf-8"))
    document["image_tree_sha256"] = "b" * 64
    attestation.write_text(recipe.json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="attestation mismatch"):
        load_canonical_attestation(attestation, data_yaml=data_yaml)

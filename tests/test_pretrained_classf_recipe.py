from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import trkh.recipes.pretrained_classf_b0 as recipe
from trkh.data.dataset import TemperedClassBatchSampler
from trkh.recipes.pretrained_classf_b0 import (
    B1_MARGIN0_PROTOCOL_ID,
    B1_NATURAL_PROTOCOL_ID,
    B2_TEMPERED_P05_PROTOCOL_ID,
    DINO_MODEL_NAME,
    DINO_SHA256,
    EXPECTED_CLASS_NAMES,
    build_train_args,
    load_canonical_attestation,
    run_name,
    source_tree_sha256,
    validate_auto_resume_checkpoint,
    validate_development_data_yaml,
)
from trkh.training.train import build_configs, parse_args, train_one_epoch


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


def test_train_epoch_reports_successful_optimizer_updates() -> None:
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


def test_classf_b0_resume_checkpoint_is_bound_to_data_and_pretrain(
    tmp_path: Path,
) -> None:
    data_yaml = tmp_path / "class_f_dev.yaml"
    data_yaml.write_text("format: classification_folder\n", encoding="utf-8")
    expected_args = build_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="full",
        run_tag="resume_unit",
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        dataset_image_tree_sha256="c" * 64,
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

    mismatched_args = build_train_args(
        data_yaml=data_yaml,
        dino_checkpoint=tmp_path / "model.safetensors",
        output_dir=tmp_path / "runs",
        stage="full",
        run_tag="resume_unit",
        source_commit="d" * 40,
        source_tree_sha256="b" * 64,
        dataset_image_tree_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_auto_resume_checkpoint(
            checkpoint_path,
            training_data_yaml=data_yaml,
            expected_train_args=mismatched_args,
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

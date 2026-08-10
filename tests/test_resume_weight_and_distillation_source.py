from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trkh.core.config import TrainConfig
from trkh.training.train import (
    _load_training_checkpoint,
    _validate_canonical_classf_teacher_lock,
    _validate_final_distillation_config,
    _validate_resume_data_source,
)


def _filled_linear(value: float) -> torch.nn.Linear:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float(value))
    return model


def _load_weights(
    checkpoint_path: Path,
    *,
    weight_source: str,
    restore_optimizer: bool,
) -> tuple[torch.nn.Linear, dict[str, object]]:
    model = _filled_linear(0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    summary, _ = _load_training_checkpoint(
        resume_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device=torch.device("cpu"),
        class_names=("a", "b"),
        restore_optimizer=restore_optimizer,
        restore_scheduler=False,
        restore_scaler=False,
        weight_source=weight_source,
    )
    return model, summary


def test_resume_auto_uses_selected_weights_for_new_phase(tmp_path: Path) -> None:
    selected = _filled_linear(1.0)
    train = _filled_linear(2.0)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 3,
            "class_names": ["a", "b"],
            "model_state": selected.state_dict(),
            "train_model_state": train.state_dict(),
        },
        checkpoint_path,
    )

    restored, summary = _load_weights(
        checkpoint_path,
        weight_source="auto",
        restore_optimizer=False,
    )

    assert summary["resolved_weight_source"] == "selected"
    assert summary["weight_state_key"] == "model_state"
    for parameter in restored.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_resume_explicit_train_uses_train_model_state(tmp_path: Path) -> None:
    selected = _filled_linear(1.0)
    train = _filled_linear(2.0)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 3,
            "class_names": ["a", "b"],
            "model_state": selected.state_dict(),
            "train_model_state": train.state_dict(),
        },
        checkpoint_path,
    )

    restored, summary = _load_weights(
        checkpoint_path,
        weight_source="train",
        restore_optimizer=False,
    )

    assert summary["resolved_weight_source"] == "train"
    assert summary["weight_state_key"] == "train_model_state"
    for parameter in restored.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2.0))


def test_selected_resume_rejects_stale_optimizer_bundle(
    tmp_path: Path,
) -> None:
    selected = _filled_linear(1.0)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 3,
            "class_names": ["a", "b"],
            "model_state": selected.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="cannot be paired with optimizer"):
        _load_weights(
            checkpoint_path,
            weight_source="selected",
            restore_optimizer=True,
        )


def test_final_resume_merge_rejects_two_global_kd_sources(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"checkpoint")
    teacher_csv = tmp_path / "teacher.csv"
    teacher_csv.write_text("path,prob_0,prob_1\n", encoding="utf-8")
    config = TrainConfig(
        pretrained_distillation=True,
        distillation_teacher_checkpoint=str(checkpoint),
        distillation_teacher_csv=str(teacher_csv),
    )

    with pytest.raises(ValueError, match="exactly one global distillation source"):
        _validate_final_distillation_config(config)


def test_resume_rejects_silent_cross_dataset_override(tmp_path: Path) -> None:
    canonical = tmp_path / "class_f" / "data.yaml"
    historical = tmp_path / "yolo_f" / "data.yaml"

    with pytest.raises(ValueError, match="cannot cross dataset semantics"):
        _validate_resume_data_source(
            cli_data_yaml=canonical,
            checkpoint_data_yaml=historical,
            allow_cross_dataset_resume=False,
        )


def test_resume_cross_dataset_override_is_explicitly_recorded(
    tmp_path: Path,
) -> None:
    summary = _validate_resume_data_source(
        cli_data_yaml=tmp_path / "class_f" / "data.yaml",
        checkpoint_data_yaml=tmp_path / "yolo_f" / "data.yaml",
        allow_cross_dataset_resume=True,
    )

    assert summary["matches"] is False
    assert summary["cross_dataset_override"] is True


def test_resume_missing_dataset_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="missing or differs"):
        _validate_resume_data_source(
            cli_data_yaml=tmp_path / "class_f" / "data.yaml",
            checkpoint_data_yaml=None,
            allow_cross_dataset_resume=False,
        )

    summary = _validate_resume_data_source(
        cli_data_yaml=tmp_path / "class_f" / "data.yaml",
        checkpoint_data_yaml=None,
        allow_cross_dataset_resume=True,
    )
    assert summary["matches"] is False
    assert summary["cross_dataset_override"] is True


def test_canonical_classf_rejects_unproven_teacher_cache(
    tmp_path: Path,
) -> None:
    data_spec = SimpleNamespace(
        data_format="classification_folder",
        class_names=[
            "Xoai_Song_Chua_KhoDap",
            "Xoai_Song_ChuaNhe_CoNguyCo",
            "Xoai_Chin_NgotThanh_DeDap",
            "Xoai_ChinGia_NgotGat_KhongVanChuyen",
            "Xoai_Hu_KhongAnDuoc",
        ],
    )
    config = TrainConfig(
        distillation_teacher_csv=str(tmp_path / "legacy_teacher.csv"),
    )

    with pytest.raises(ValueError, match="fail-closed"):
        _validate_canonical_classf_teacher_lock(data_spec, config)


def test_canonical_classf_rejects_patch_router_teacher_cache(
    tmp_path: Path,
) -> None:
    data_spec = SimpleNamespace(
        data_format="classification_folder",
        class_names=[
            "Xoai_Song_Chua_KhoDap",
            "Xoai_Song_ChuaNhe_CoNguyCo",
            "Xoai_Chin_NgotThanh_DeDap",
            "Xoai_ChinGia_NgotGat_KhongVanChuyen",
            "Xoai_Hu_KhongAnDuoc",
        ],
    )
    config = TrainConfig(
        patch_evidence_router_teacher_csv=str(
            tmp_path / "legacy_patch_router_teacher.csv"
        ),
    )

    with pytest.raises(ValueError, match="patch_evidence_router_teacher_csv"):
        _validate_canonical_classf_teacher_lock(data_spec, config)

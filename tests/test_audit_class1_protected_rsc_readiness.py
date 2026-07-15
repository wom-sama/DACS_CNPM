from pathlib import Path
import os

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import create_model
from trkh.tools.audit_class1_protected_rsc_readiness import (
    _locked_args_exact,
    assess_stage_a,
    parse_args,
)


def _comparison(
    *,
    macro: float = 0.001,
    f1: float = 0.006,
    precision: float = 0.006,
    recall: float = 0.0,
    fp_reduction: int = 2,
    rescues: int = 1,
    breaks: int = 0,
    corrections: int = 3,
    harms: int = 1,
) -> dict:
    return {
        "control": {"predicted_support": [100, 100, 100, 100, 100]},
        "candidate": {"predicted_support": [100, 98, 100, 100, 100]},
        "delta": {
            "macro_f1": macro,
            "class1_f1": f1,
            "class1_precision": precision,
            "class1_recall": recall,
        },
        "transitions": {
            "restricted_focus_fp_reduction": fp_reduction,
            "focus_fn_rescue": rescues,
            "focus_tp_break": breaks,
            "candidate_correction": corrections,
            "candidate_harm": harms,
        },
        "maximum_nonfocus_f1_drop": 0.005,
    }


def test_default_arguments_match_locked_protocol() -> None:
    args = parse_args([])

    assert _locked_args_exact(args)
    assert args.checkpoint == Path(
        "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt"
    )
    assert args.eligible_classes == "0,2,3,4"
    assert args.max_train_batches == 60
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_native_position_grid_skips_nondeterministic_bicubic(monkeypatch) -> None:
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
            token_pruning=False,
        ),
    )

    def fail_interpolate(*args, **kwargs):
        raise AssertionError("native grid must not invoke bicubic interpolation")

    monkeypatch.setattr("trkh.models.model.F.interpolate", fail_interpolate)
    positional = model.get_interpolated_pos_embed(model.patch_embed.base_grid_size)
    positional.sum().backward()

    assert torch.equal(positional, model.pos_embed)
    assert model.pos_embed.grad is not None


def test_non_native_position_grid_still_interpolates() -> None:
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
            token_pruning=False,
        ),
    )

    positional = model.get_interpolated_pos_embed((5, 3))
    positional.sum().backward()

    expected_tokens = (
        1 + (model.num_registers if model.register_positional_embedding else 0) + 15
    )
    assert positional.shape == (1, expected_tokens, 32)
    assert model.pos_embed.grad is not None


def test_assessment_authorizes_only_when_every_gate_passes() -> None:
    clean = _comparison()
    illumination = [_comparison(f1=0.0, precision=0.0) for _ in range(3)]

    decision = assess_stage_a(
        structural_checks={"source": True, "mechanism": True},
        clean=clean,
        illumination=illumination,
    )

    assert decision["all_gates_passed"]
    assert decision["stage_b_smoke_authorized"]
    assert not decision["full_train_authorized"]


def test_assessment_rejects_precision_from_recall_suppression() -> None:
    clean = _comparison(precision=0.10, f1=-0.05, recall=-0.20, fp_reduction=10)
    illumination = [_comparison(f1=0.0, precision=0.0) for _ in range(3)]

    decision = assess_stage_a(
        structural_checks={"source": True},
        clean=clean,
        illumination=illumination,
    )

    assert not decision["stage_b_smoke_authorized"]
    assert "class1_f1_delta_gte_0p005" in decision["failed_checks"]
    assert "class1_recall_delta_gte_minus_0p005" in decision["failed_checks"]

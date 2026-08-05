from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from trkh.tools.run_dinov3_convpass_b21_train_fold import (
    ModelEMA,
    classification_metrics,
    parameter_role,
    parse_args,
    promotion_gate,
    state_sha256,
)


def _metrics(
    *,
    macro: float,
    c1_f1: float,
    c1_tp: int,
    restricted_fp: int,
    two_to_one: int,
) -> dict:
    return {
        "macro_f1": macro,
        "class1": {"f1": c1_f1, "tp": c1_tp},
        "restricted_fp_into_class1": restricted_fp,
        "transitions_into_class1": {"0->1": 2, "2->1": two_to_one, "3->1": 1, "4->1": 1},
    }


def test_classification_metrics_reports_directional_class1_errors() -> None:
    labels = [0, 0, 1, 1, 1, 2, 2, 3, 4]
    predictions = [0, 1, 1, 1, 2, 1, 2, 3, 1]

    result = classification_metrics(labels, predictions)

    assert result["confusion_matrix"][0][1] == 1
    assert result["confusion_matrix"][2][1] == 1
    assert result["confusion_matrix"][4][1] == 1
    assert result["restricted_fp_into_class1"] == 3
    assert result["class1"]["tp"] == 2
    assert result["class1"]["fn"] == 1
    assert result["class1"]["fp"] == 3
    assert result["accuracy"] == pytest.approx(5 / 9)


def test_gate_passes_on_class1_gain_with_all_safety_checks() -> None:
    direct = _metrics(macro=0.85, c1_f1=0.63, c1_tp=100, restricted_fp=80, two_to_one=40)
    spatial = _metrics(macro=0.849, c1_f1=0.636, c1_tp=99, restricted_fp=78, two_to_one=39)

    gate = promotion_gate(direct, spatial, residual_p95=0.02, finite_updates=True)

    assert gate["passed"] is True
    assert gate["checks"]["quality_signal"] is True
    assert gate["meaning"] == "pass_authorizes_fixed_train_oof_with_dephased_control_only"


def test_gate_can_use_fp_route_but_rejects_two_to_one_growth() -> None:
    direct = _metrics(macro=0.85, c1_f1=0.63, c1_tp=100, restricted_fp=80, two_to_one=40)
    spatial = _metrics(macro=0.85, c1_f1=0.631, c1_tp=99, restricted_fp=75, two_to_one=41)

    gate = promotion_gate(direct, spatial, residual_p95=0.02, finite_updates=True)

    assert gate["checks"]["quality_signal"] is True
    assert gate["checks"]["two_to_one_nonincrease"] is False
    assert gate["passed"] is False


@pytest.mark.parametrize("residual", (0.0, 1e-5, 0.1, float("inf")))
def test_gate_rejects_inactive_or_unbounded_branch(residual: float) -> None:
    direct = _metrics(macro=0.85, c1_f1=0.63, c1_tp=100, restricted_fp=80, two_to_one=40)
    spatial = _metrics(macro=0.85, c1_f1=0.64, c1_tp=100, restricted_fp=75, two_to_one=39)

    gate = promotion_gate(direct, spatial, residual_p95=residual, finite_updates=True)

    assert gate["checks"]["branch_active_bounded"] is False
    assert gate["passed"] is False


def test_parameter_roles_keep_adapter_and_head_out_of_backbone_lr() -> None:
    assert parameter_role("backbone.blocks.0.adapter_attn.up.weight") == "adapter"
    assert parameter_role("backbone.blocks.11.adapter_mlp.conv.weight") == "adapter"
    assert parameter_role("backbone.head.weight") == "head"
    assert parameter_role("backbone.blocks.0.attn.qkv.weight") == "backbone"


def test_state_hash_is_order_independent_and_content_sensitive() -> None:
    first = {
        "b": torch.tensor([2.0]),
        "a": torch.tensor([1.0]),
    }
    second = {"a": first["a"].clone(), "b": first["b"].clone()}
    changed = copy.deepcopy(second)
    changed["a"].add_(1.0)

    assert state_sha256(first) == state_sha256(second)
    assert state_sha256(first) != state_sha256(changed)


def test_model_ema_uses_training_warmup_and_copies_integer_state() -> None:
    model = nn.BatchNorm1d(2)
    ema = ModelEMA(model, decay=0.995)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.num_batches_tracked.fill_(7)
    ema.update(model)

    expected_decay = 2.0 / 11.0
    assert torch.allclose(
        ema.module.weight,
        torch.full_like(ema.module.weight, expected_decay * 1.0 + (1.0 - expected_decay) * 2.0),
    )
    assert int(ema.module.num_batches_tracked.item()) == 7


def test_parser_exposes_no_epoch_or_validation_override() -> None:
    parsed = parse_args(
        [
            "--output-dir",
            "out",
            "--assignment-csv",
            "folds.csv",
            "--dino-weight",
            "model.safetensors",
            "--preflight-only",
        ]
    )

    assert parsed.preflight_only is True
    assert parsed.preflight_artifact is None
    assert not hasattr(parsed, "epochs")
    assert not hasattr(parsed, "validation")

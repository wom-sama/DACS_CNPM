import copy

import pytest
import torch
from torch import nn

from trkh.tools import run_b38_relit_bicar_fold2 as b38
from trkh.tools.run_dinov3_convpass_b21_train_fold import gradient_alignment_by_role


def _condition(
    *,
    accuracy: float,
    macro: float,
    class1: float,
    pair: float,
    tp: int,
    restricted_fp: int,
    zero_to_one: int,
) -> dict[str, object]:
    confusion = [[0] * 5 for _ in range(5)]
    confusion[0][1] = zero_to_one
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "class1_f1": class1,
        "mean_pair_auroc": pair,
        "class1_tp": tp,
        "restricted_fp": restricted_fp,
        "confusion_matrix": confusion,
    }


def test_b38_gate_requires_clean_and_relighting_improvement() -> None:
    control = {
        "clean": _condition(
            accuracy=0.92,
            macro=0.89,
            class1=0.70,
            pair=0.98,
            tp=70,
            restricted_fp=30,
            zero_to_one=10,
        ),
        "lighting_dim": _condition(
            accuracy=0.88,
            macro=0.84,
            class1=0.60,
            pair=0.96,
            tp=65,
            restricted_fp=45,
            zero_to_one=20,
        ),
        "lighting_bright": _condition(
            accuracy=0.85,
            macro=0.80,
            class1=0.50,
            pair=0.93,
            tp=60,
            restricted_fp=70,
            zero_to_one=40,
        ),
    }
    candidate = copy.deepcopy(control)
    candidate["clean"].update(
        accuracy=0.921,
        macro_f1=0.891,
        class1_f1=0.708,
        class1_tp=71,
        restricted_fp=29,
    )
    candidate["lighting_dim"]["class1_f1"] = 0.615
    candidate["lighting_bright"]["class1_f1"] = 0.525
    candidate["lighting_bright"]["confusion_matrix"][0][1] = 35

    result = b38._gate(control, candidate)

    assert result["passed"] is True
    assert result["next_permission"] == "confirm_relit_bicar_on_remaining_train_folds"

    candidate["lighting_bright"]["confusion_matrix"][0][1] = 38
    failed = b38._gate(control, candidate)
    assert failed["passed"] is False
    assert "bright_zero_to_one_reduction" in failed["failed"]


def test_b38_objective_is_inactive_before_epoch_three() -> None:
    objective = b38.RelitBiCARObjective(class_counts=(20, 10, 20, 20, 20))
    logits = torch.randn(4, 5, requires_grad=True)
    loss, stats = objective(
        nn.Identity(),
        torch.randn(4, 3, 8, 8),
        torch.tensor([0, 1, 2, 4]),
        logits,
        b38.START_EPOCH - 2,
        0,
    )
    assert loss.item() == 0.0
    assert stats == {"active": 0.0}
    assert objective.state.updates == 0


def test_b38_oracle_penalizes_both_class_one_confusion_directions() -> None:
    oracle = b38._oracle_directions((200, 50, 130, 210, 240))
    assert oracle["passed"] is True
    assert all(oracle["checks"].values())


class _ToyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.head = nn.Linear(2, 2, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.backbone.head(values)


def test_gradient_alignment_reports_weighted_auxiliary_scale() -> None:
    model = _ToyHead()
    output = model(torch.tensor([[1.0, -1.0]]))
    task = output.square().sum()
    auxiliary = 0.25 * output.square().sum()

    alignment = gradient_alignment_by_role(model, task, auxiliary)

    assert alignment["head"]["task_norm"] > 0.0
    assert alignment["head"]["auxiliary_over_task"] == pytest.approx(0.25)
    assert alignment["head"]["cosine"] == pytest.approx(1.0)


def test_b38_matched_warmup_requires_exact_epoch_two_state() -> None:
    rows = [
        {
            "epoch": epoch,
            "mean_loss": 1.0 / epoch,
            "mean_task_loss": 1.0 / epoch,
            "optimizer_updates_total": epoch * 3,
            "learning_rates": [1e-4],
            "update_over_parameter_norm": {"backbone": 0.01 * epoch},
            "model_state_sha256": "model-2" if epoch == 2 else None,
            "ema_state_sha256": "ema-2" if epoch == 2 else None,
        }
        for epoch in (1, 2)
    ]
    control = {"initial_primary_state_sha256": "initial", "epochs": rows}
    candidate = copy.deepcopy(control)

    assert b38._matched_warmup(control, candidate)["passed"] is True

    candidate["epochs"][1]["model_state_sha256"] = "changed"
    failed = b38._matched_warmup(control, candidate)
    assert failed["passed"] is False
    assert failed["checks"]["epoch2_model_state_sha256"] is False

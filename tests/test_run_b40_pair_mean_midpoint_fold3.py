import torch
from torch import nn

from trkh.tools import run_b40_pair_mean_midpoint_fold3 as b40
from trkh.training.confusion_spectral import ConfusionSpectralEMAState


def test_b40_pair_result_updates_both_unequal_directions() -> None:
    logits = torch.tensor(
        [
            [0.0, 3.0, -1.0, -2.0, -3.0],
            [1.0, 0.0, -1.0, -2.0, -3.0],
        ],
        requires_grad=True,
    )
    result = b40._pair_result(
        logits,
        torch.tensor([0, 1]),
        pair=b40.BRIGHT_PAIR,
        state=ConfusionSpectralEMAState(),
    )
    gradient = torch.autograd.grad(result.loss, logits)[0]
    assert gradient[0, 0] < 0 and gradient[0, 1] > 0
    assert gradient[1, 0] > 0 and gradient[1, 1] < 0


def test_b40_objective_is_inactive_before_epoch_three() -> None:
    objective = b40.StratifiedPairMeanObjective(
        class_counts=(100, 25, 75, 100, 100)
    )
    logits = torch.randn(5, 5, requires_grad=True)
    loss, stats = objective(
        nn.Identity(),
        torch.randn(5, 3, 8, 8),
        torch.arange(5),
        logits,
        b40.START_EPOCH - 2,
        0,
    )
    assert loss.item() == 0.0
    assert stats == {"active": 0.0}
    assert objective.active_calls == 0
    assert objective.bright_state.updates == 0
    assert objective.dim_state.updates == 0


def test_b40_oracle_covers_both_unequal_pair_directions() -> None:
    oracle = b40._pair_oracle_directions()
    assert oracle["passed"] is True
    assert all(oracle["checks"].values())
    assert min(oracle["row_gradient_min_over_max"].values()) >= 0.10


def test_b40_gate_uses_its_own_follow_up_language(monkeypatch) -> None:
    monkeypatch.setattr(
        b40.b38,
        "_gate",
        lambda _control, _candidate: {
            "passed": False,
            "next_permission": "stale_b38_value",
        },
    )
    gate = b40._gate({}, {})
    assert gate["next_permission"] == "close_exact_pair_mean_midpoint"


def test_b40_scale_is_mechanically_locked() -> None:
    contract = b40._objective_contract()
    assert contract["outer_weight"] == 0.15
    assert contract["pair_mix_weights"] == [0.5, 0.5]
    assert contract["frequency_weighting"] is False

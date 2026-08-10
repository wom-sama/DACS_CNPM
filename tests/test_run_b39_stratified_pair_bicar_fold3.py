import torch
from torch import nn

from trkh.tools import run_b39_stratified_pair_bicar_fold3 as b39
from trkh.training.confusion_spectral import ConfusionSpectralEMAState


def test_b39_pair_result_updates_both_confusion_directions() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0, -2.0, -3.0],
            [0.0, 2.0, -1.0, -2.0, -3.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])
    state = ConfusionSpectralEMAState()
    result = b39._pair_result(
        logits,
        labels,
        pair=b39.BRIGHT_PAIR,
        pair_counts=(100, 25),
        state=state,
    )
    gradient = torch.autograd.grad(result.loss, logits)[0]
    assert gradient[0, 0] < 0 and gradient[0, 1] > 0
    assert gradient[1, 0] > 0 and gradient[1, 1] < 0


def test_b39_objective_is_inactive_before_epoch_three() -> None:
    objective = b39.StratifiedPairBiCARObjective(
        class_counts=(100, 25, 75, 100, 100)
    )
    logits = torch.randn(5, 5, requires_grad=True)
    loss, stats = objective(
        nn.Identity(),
        torch.randn(5, 3, 8, 8),
        torch.arange(5),
        logits,
        b39.START_EPOCH - 2,
        0,
    )
    assert loss.item() == 0.0
    assert stats == {"active": 0.0}
    assert objective.active_calls == 0
    assert objective.bright_state.updates == 0
    assert objective.dim_state.updates == 0


def test_b39_pair_oracle_covers_bright_and_dim_pairs() -> None:
    oracle = b39._pair_oracle_directions((200, 50, 130, 210, 240))
    assert oracle["passed"] is True
    assert all(oracle["checks"].values())


def test_b39_gate_uses_its_own_follow_up_language(monkeypatch) -> None:
    monkeypatch.setattr(
        b39.b38,
        "_gate",
        lambda _control, _candidate: {
            "passed": False,
            "checks": {"clean": False},
            "next_permission": "stale_b38_value",
        },
    )
    gate = b39._gate({}, {})
    assert gate["next_permission"] == "close_exact_stratified_pair_midpoint"

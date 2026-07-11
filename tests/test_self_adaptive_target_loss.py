import torch

from trkh.training.train import (
    _initialize_self_adaptive_target_state,
    _self_adaptive_target_loss,
)


def test_initialize_self_adaptive_target_state_is_one_hot() -> None:
    state = _initialize_self_adaptive_target_state(
        [0, 1, 3],
        num_classes=4,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(state, expected)


def test_self_adaptive_target_loss_updates_state_by_sample_index() -> None:
    state = _initialize_self_adaptive_target_state(
        [0, 1, 1],
        num_classes=3,
        device=torch.device("cpu"),
    )
    original = state.clone()
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.5, 1.4]], requires_grad=True)
    targets = torch.tensor([0, 1])
    sample_indices = torch.tensor([0, 2])

    loss, stats = _self_adaptive_target_loss(
        logits,
        targets,
        sample_indices,
        state,
        beta=0.5,
        hard_target_weight=0.1,
        confidence_power=1.0,
        min_confidence=0.0,
        update_state=True,
    )

    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert stats["fraction"] == 1.0
    assert stats["confidence"] > 0.0
    assert not torch.equal(state[0], original[0])
    assert torch.equal(state[1], original[1])
    assert not torch.equal(state[2], original[2])
    assert torch.allclose(state.sum(dim=1), torch.ones(3), atol=1e-6)


def test_self_adaptive_target_loss_can_skip_state_update_for_replay() -> None:
    state = _initialize_self_adaptive_target_state(
        [0, 1],
        num_classes=2,
        device=torch.device("cpu"),
    )
    original = state.clone()

    _self_adaptive_target_loss(
        torch.tensor([[0.2, 1.0], [1.1, 0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        state,
        beta=0.5,
        hard_target_weight=0.1,
        confidence_power=0.0,
        min_confidence=0.0,
        update_state=False,
    )

    assert torch.equal(state, original)

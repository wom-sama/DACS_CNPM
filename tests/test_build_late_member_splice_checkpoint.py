from pathlib import Path

import pytest
import torch

from trkh.tools.build_late_member_splice_checkpoint import (
    READOUT_PREFIXES,
    build_late_member_splice_checkpoint,
    splice_prefixes,
)


def _state(offset: float) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "stem.weight": torch.tensor([offset + 1.0]),
        "stem.scalar": torch.tensor(offset + 2.0),
    }
    for index in range(8):
        state[f"blocks.{index}.weight"] = torch.tensor([offset + 10.0 + index])
    for index, prefix in enumerate(READOUT_PREFIXES):
        state[f"{prefix}weight"] = torch.tensor([offset + 100.0 + index])
    return state


def _checkpoint(offset: float) -> dict[str, object]:
    return {
        "model_config": {
            "depth": 8,
            "token_prune_layers": "2,5",
        },
        "model_state": _state(offset),
        "ema_model_state": _state(offset + 1000.0),
        "optimizer_state": {"large": True},
        "metrics": {"stale": True},
    }


def _build(
    keeper: dict[str, object],
    candidate: dict[str, object],
):
    return build_late_member_splice_checkpoint(
        keeper,
        candidate,
        keeper_path=Path("keeper.pt"),
        candidate_path=Path("candidate.pt"),
        keeper_sha256="a" * 64,
        candidate_sha256="b" * 64,
        fork_after_block=6,
    )


def test_splice_copies_only_late_blocks_and_readout() -> None:
    keeper = _checkpoint(0.0)
    candidate = _checkpoint(1000.0)

    output, summary = _build(keeper, candidate)
    output_state = output["model_state"]
    assert isinstance(output_state, dict)
    assert torch.equal(output_state["blocks.5.weight"], torch.tensor([15.0]))
    assert torch.equal(output_state["blocks.6.weight"], torch.tensor([1016.0]))
    assert torch.equal(output_state["head.weight"], torch.tensor([1101.0]))
    assert torch.equal(output_state["stem.weight"], torch.tensor([1.0]))
    assert summary["untouched_tensors_bit_identical"] is True
    assert summary["fork_after_block"] == 6
    assert "optimizer_state" not in output
    assert "ema_model_state" not in output
    assert "metrics" not in output


def test_splice_prefixes_use_zero_based_state_after_one_based_fork() -> None:
    prefixes = splice_prefixes(depth=8, fork_after_block=6)
    assert prefixes[:2] == ("blocks.6.", "blocks.7.")
    assert prefixes[2:] == READOUT_PREFIXES


def test_splice_rejects_state_schema_mismatch() -> None:
    keeper = _checkpoint(0.0)
    candidate = _checkpoint(1000.0)
    del candidate["model_state"]["head.weight"]

    with pytest.raises(ValueError, match="state schemas differ"):
        _build(keeper, candidate)


def test_splice_rejects_shape_mismatch() -> None:
    keeper = _checkpoint(0.0)
    candidate = _checkpoint(1000.0)
    candidate["model_state"]["blocks.7.weight"] = torch.zeros(2)

    with pytest.raises(ValueError, match="shape mismatch"):
        _build(keeper, candidate)


def test_splice_rejects_pruning_after_fork() -> None:
    keeper = _checkpoint(0.0)
    candidate = _checkpoint(1000.0)
    keeper["model_config"]["token_prune_layers"] = "2,7"

    with pytest.raises(ValueError, match="all token pruning"):
        _build(keeper, candidate)


def test_splice_rejects_missing_required_readout_prefix() -> None:
    keeper = _checkpoint(0.0)
    candidate = _checkpoint(1000.0)
    del keeper["model_state"]["fine_grained_pool.weight"]
    del candidate["model_state"]["fine_grained_pool.weight"]

    with pytest.raises(ValueError, match="prefixes have no state tensors"):
        _build(keeper, candidate)


def test_splice_rejects_nonfinite_tensor() -> None:
    keeper = _checkpoint(0.0)
    candidate = _checkpoint(1000.0)
    candidate["model_state"]["head.weight"] = torch.tensor([float("nan")])

    with pytest.raises(ValueError, match="nonfinite"):
        _build(keeper, candidate)

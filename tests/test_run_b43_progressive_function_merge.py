from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from trkh.tools import run_b43_progressive_function_merge as b43


def _assignment() -> dict[str, object]:
    paths: list[str] = []
    labels: list[int] = []
    folds: list[int] = []
    for label in range(5):
        for row in range(8):
            paths.append(f"train/{label}/image_{label}_{row}.jpg")
            labels.append(label)
            folds.append(row % 5)
    return {
        "relative_paths": paths,
        "labels": np.asarray(labels, dtype=np.int64),
        "folds": np.asarray(folds, dtype=np.int64),
    }


def test_calibration_selection_is_balanced_deterministic_and_excludes_fold3() -> None:
    assignment = _assignment()
    first = b43._select_calibration_positions(assignment)
    second = b43._select_calibration_positions(assignment)
    labels = np.asarray(assignment["labels"])[first]
    folds = np.asarray(assignment["folds"])[first]
    assert first == second
    assert np.bincount(labels, minlength=5).tolist() == [2, 2, 2, 2, 2]
    assert 3 not in folds.tolist()


def test_calibration_selection_rejects_missing_class_support() -> None:
    assignment = _assignment()
    assignment["folds"] = np.where(
        np.asarray(assignment["labels"]) == 4,
        b43.SOURCE_FOLD,
        np.asarray(assignment["folds"]),
    )
    with pytest.raises(b43.B43ContractError, match="Class 4"):
        b43._select_calibration_positions(assignment)


class _ToyBlock(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        rope: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del attn_mask, is_causal
        return value + (0.0 if rope is None else rope)


class _ToyModel(nn.Module):
    def __init__(self, block_count: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock() for _ in range(block_count)])
        self.register_buffer("rope", torch.ones(1, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value, rope=self.rope)
        return value


def test_capture_block_io_preserves_order_and_rope() -> None:
    model = _ToyModel()
    images = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    inputs, outputs, rope = b43._capture_block_io(
        model,
        images,
        block_index=1,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert torch.equal(inputs, images + 1.0)
    assert torch.equal(outputs, images + 2.0)
    assert torch.equal(rope, torch.ones(1, 2))


def test_block_state_and_rng_helpers_are_strict() -> None:
    state = {
        "backbone.blocks.0.weight": torch.tensor([1.0]),
        "backbone.blocks.1.weight": torch.tensor([2.0]),
    }
    assert b43._block_state(state, 1) == {"weight": torch.tensor([2.0])}
    before = b43._rng_snapshot()
    after = b43._rng_snapshot()
    assert b43._same_rng(before, after)
    torch.rand(1)
    changed = b43._rng_snapshot()
    assert not b43._same_rng(before, changed)


def test_formal_balanced_indices_and_block_replacement() -> None:
    indices = b43._balanced_source_indices(
        80,
        8,
        16,
        device=torch.device("cpu"),
    )
    assert indices.tolist() == list(range(8, 16)) + list(range(88, 96))
    with pytest.raises(ValueError, match="range"):
        b43._balanced_source_indices(80, 16, 8, device=torch.device("cpu"))

    state = {
        "backbone.blocks.0.weight": torch.tensor([1.0]),
        "backbone.blocks.0.bias": torch.tensor([2.0]),
        "backbone.blocks.1.weight": torch.tensor([3.0]),
    }
    b43._replace_block_state(
        state,
        0,
        {"weight": torch.tensor([4.0]), "bias": torch.tensor([5.0])},
    )
    assert state["backbone.blocks.0.weight"].item() == 4.0
    assert state["backbone.blocks.0.bias"].item() == 5.0
    assert state["backbone.blocks.1.weight"].item() == 3.0


def test_capture_all_formal_block_outputs() -> None:
    model = _ToyModel(block_count=12)
    images = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    outputs = b43._capture_all_block_outputs(
        model,
        images,
        batch_size=3,
        device=torch.device("cpu"),
    )
    assert len(outputs) == 12
    for index, output in enumerate(outputs, start=1):
        assert torch.equal(output, images + float(index))

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from trkh.tools.average_checkpoints import average_state_dicts, build_average_checkpoint


def test_average_state_dicts_weighted_float_and_non_float_copy() -> None:
    first = OrderedDict(
        weight=torch.tensor([1.0, 3.0]),
        counter=torch.tensor(4, dtype=torch.long),
    )
    second = OrderedDict(
        weight=torch.tensor([3.0, 7.0]),
        counter=torch.tensor(4, dtype=torch.long),
    )
    averaged, stats = average_state_dicts([first, second], [0.25, 0.75])

    assert torch.allclose(averaged["weight"], torch.tensor([2.5, 6.0]))
    assert torch.equal(averaged["counter"], torch.tensor(4, dtype=torch.long))
    assert stats["copied_non_float"] == 1
    assert stats["copied_different_non_float"] == 0


def test_build_average_checkpoint_rejects_shape_mismatch(tmp_path) -> None:
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    torch.save({"model_state": OrderedDict(weight=torch.ones(2))}, ckpt_a)
    torch.save({"model_state": OrderedDict(weight=torch.ones(3))}, ckpt_b)

    with pytest.raises(ValueError, match="shape mismatch"):
        build_average_checkpoint([ckpt_a, ckpt_b], weights=[0.5, 0.5])

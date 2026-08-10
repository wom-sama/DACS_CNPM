from pathlib import Path

import pytest
import torch

from trkh.tools.extract_ema_checkpoint import build_ema_checkpoint


def _checkpoint() -> dict:
    return {
        "epoch": 3,
        "ema_updates": 12,
        "model_state": {
            "weight": torch.tensor([1.0, 2.0]),
            "counter": torch.tensor(2),
        },
        "ema_model_state": {
            "weight": torch.tensor([3.0, 4.0]),
            "counter": torch.tensor(5),
        },
        "optimizer_state": {"state": "large"},
    }


def test_build_ema_checkpoint_promotes_ema_and_removes_resume_payload() -> None:
    output, summary = build_ema_checkpoint(
        _checkpoint(),
        source_path=Path("last.pt"),
        source_sha256="abc",
    )

    assert torch.equal(output["model_state"]["weight"], torch.tensor([3.0, 4.0]))
    assert "ema_model_state" not in output
    assert "optimizer_state" not in output
    assert output["checkpoint_kind"] == "ema_export"
    assert output["checkpoint_weight_source"] == "ema"
    assert summary["tensor_count"] == 2


def test_build_ema_checkpoint_rejects_shape_mismatch() -> None:
    checkpoint = _checkpoint()
    checkpoint["ema_model_state"]["weight"] = torch.zeros(3)

    with pytest.raises(ValueError, match="shape mismatch"):
        build_ema_checkpoint(
            checkpoint,
            source_path=Path("last.pt"),
            source_sha256="abc",
        )

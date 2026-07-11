import torch
from torch import nn

from trkh.tools.evaluate_test_time_adaptation import (
    build_adapted_checkpoint_payload,
    collect_adaptation_parameters,
    reliable_entropy_mask,
)


class TinyAdaptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.body = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)


def test_reliable_entropy_mask_keeps_low_entropy_confident_samples() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.1],
        ]
    )

    mask = reliable_entropy_mask(
        logits,
        selection_fraction=1.0,
        min_confidence=0.50,
    )

    assert mask.tolist() == [True, False, True, False]


def test_collect_adaptation_parameters_limits_trainable_state() -> None:
    model = TinyAdaptModel()

    parameters, names = collect_adaptation_parameters(model, "layernorm_head_bias")

    assert set(names) == {"norm.weight", "norm.bias", "head.bias"}
    assert sum(parameter.numel() for parameter in parameters) == 11
    assert model.norm.weight.requires_grad
    assert model.norm.bias.requires_grad
    assert model.head.bias.requires_grad
    assert not model.head.weight.requires_grad
    assert not model.body.weight.requires_grad


def test_build_adapted_checkpoint_payload_drops_stale_training_state(tmp_path) -> None:
    model = TinyAdaptModel()
    source_checkpoint = {
        "model_state": {"old": torch.tensor([1.0])},
        "model_config": {"model_type": "tiny"},
        "class_names": ["a", "b", "c"],
        "optimizer_state": {"stale": True},
        "ema_model_state": {"stale": torch.tensor([2.0])},
    }

    args = type(
        "Args",
        (),
        {
            "checkpoint": tmp_path / "source.pt",
            "data": tmp_path / "data.yaml",
            "split": "val",
        },
    )()
    args.checkpoint.write_bytes(b"placeholder")
    args.data.write_text("names: []\n", encoding="utf-8")

    payload = build_adapted_checkpoint_payload(
        source_checkpoint=source_checkpoint,
        model=model,
        adapt_summary={"selected_samples": 2},
        args=args,
    )

    assert "optimizer_state" not in payload
    assert "ema_model_state" not in payload
    assert payload["test_time_adaptation"]["selected_samples"] == 2
    assert set(payload["model_state"]) == set(model.state_dict())

import torch
from torch import nn

from trkh.models.model import SubCenterProxyHead
from trkh.training.train import _subcenter_proxy_loss_from_features


class _DummySubcenterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 3, bias=False)
        self.subcenter_proxy_head = SubCenterProxyHead(
            embed_dim=2,
            num_classes=3,
            subcenters=2,
            dropout=0.0,
            init_std=0.001,
        )

    def head_input_from_features(self, features):
        return features["pooled"]

    def subcenter_proxy_logits_from_head_input(self, head_input):
        return self.subcenter_proxy_head(head_input)


def _set_proxy_directions(model: _DummySubcenterModel, directions) -> None:
    with torch.no_grad():
        proxies = torch.tensor(directions, dtype=torch.float32)
        model.subcenter_proxy_head.proxies.copy_(proxies[:, None, :].expand(-1, 2, -1))


def test_subcenter_proxy_loss_lower_for_aligned_proxy() -> None:
    model = _DummySubcenterModel()
    features = {"pooled": torch.tensor([[1.0, -1.0], [-1.0, 1.0]])}
    targets = torch.tensor([0, 1])

    _set_proxy_directions(
        model,
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
    )
    aligned_loss, aligned_stats = _subcenter_proxy_loss_from_features(
        model=model,
        features=features,
        targets=targets,
        margin=0.05,
        scale=8.0,
        classes="0,1",
    )

    _set_proxy_directions(
        model,
        [
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
    )
    swapped_loss, _ = _subcenter_proxy_loss_from_features(
        model=model,
        features=features,
        targets=targets,
        margin=0.05,
        scale=8.0,
        classes="0,1",
    )

    assert aligned_loss.item() < swapped_loss.item()
    assert aligned_stats["valid_fraction"] == 1.0
    assert aligned_stats["positive_negative_margin"] > 0.0
    assert aligned_stats["selected_subcenter_count"] >= 1.0


def test_subcenter_proxy_loss_respects_class_filter() -> None:
    model = _DummySubcenterModel()
    features = {"pooled": torch.tensor([[1.0, -1.0]])}
    targets = torch.tensor([2])

    loss, stats = _subcenter_proxy_loss_from_features(
        model=model,
        features=features,
        targets=targets,
        margin=0.05,
        scale=8.0,
        classes="0,1",
    )

    assert loss.item() == 0.0
    assert stats["valid_fraction"] == 0.0

from types import SimpleNamespace

import torch

from trkh.models.model import (
    VisionTransformerWithRegisters,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.training.train import (
    _cumulative_ordinal_loss_from_features,
    _ordinal_distribution_loss_from_logits,
)


def test_cumulative_ordinal_head_zero_init_is_neutral():
    torch.manual_seed(7)
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        in_channels=3,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        cumulative_ordinal_head=True,
        cumulative_ordinal_classes="0,1,2,3",
        cumulative_ordinal_logit_scale=0.35,
        cumulative_ordinal_dropout=0.0,
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    features = model.forward_features(images)
    head_input = extract_head_input_from_features(model, features)
    base_logits = model.head(head_input)
    logits = classification_logits_from_features(model, features)
    assert "cumulative_ordinal_logits" in features
    assert features["cumulative_ordinal_logits"].shape == (2, 3)
    torch.testing.assert_close(logits, base_logits, atol=1e-6, rtol=1e-6)


def test_cumulative_ordinal_loss_prefers_correct_threshold_direction():
    model = SimpleNamespace(cumulative_ordinal_classes=[0, 1, 2, 3])
    targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [-5.0, 0.0, 0.0],
            [5.0, -5.0, 0.0],
            [5.0, 5.0, -5.0],
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )
    bad_logits = -good_logits
    good_loss = _cumulative_ordinal_loss_from_features(
        model=model,
        features={"cumulative_ordinal_logits": good_logits},
        targets=targets,
    )
    bad_loss = _cumulative_ordinal_loss_from_features(
        model=model,
        features={"cumulative_ordinal_logits": bad_logits},
        targets=targets,
    )
    assert torch.isfinite(good_loss)
    assert good_loss.item() < bad_loss.item()


def test_ordinal_distribution_loss_prefers_ordered_probability_and_ignores_other_class():
    targets = torch.tensor([1, 1, 4], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [-1.0, 4.0, -1.0, -3.0, 2.0],
            [-1.0, 3.5, -0.5, -3.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
        ]
    )
    bad_logits = torch.tensor(
        [
            [-2.0, -1.0, -0.5, 4.0, 2.0],
            [-2.0, -1.0, -0.5, 3.5, 1.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
        ]
    )
    good_loss = _ordinal_distribution_loss_from_logits(
        logits=good_logits,
        targets=targets,
        classes="0,1,2,3",
    )
    bad_loss = _ordinal_distribution_loss_from_logits(
        logits=bad_logits,
        targets=targets,
        classes="0,1,2,3",
    )
    ignored_loss = _ordinal_distribution_loss_from_logits(
        logits=good_logits[-1:],
        targets=targets[-1:],
        classes="0,1,2,3",
    )
    assert torch.isfinite(good_loss)
    assert good_loss.item() < bad_loss.item()
    assert ignored_loss.item() == 0.0

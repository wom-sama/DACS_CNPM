import torch

from trkh.models.model import VisionTransformerWithRegisters, classification_logits_from_features
from trkh.training.train import _class_independent_loss_from_features


def test_class_independent_head_adds_ova_logits_feature():
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=2,
        class_independent_head=True,
    )
    images = torch.randn(3, 3, 32, 32)
    features = model.forward_features(images)
    logits = classification_logits_from_features(model, features)

    assert logits.shape == (3, 5)
    assert features["class_independent_logits"].shape == (3, 5)


def test_class_independent_loss_is_finite_and_tracks_margin():
    features = {
        "class_independent_logits": torch.tensor(
            [
                [2.0, -1.0, -1.0],
                [-1.0, 2.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )
    }
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    loss, stats = _class_independent_loss_from_features(
        features=features,
        targets=targets,
        positive_weight=2.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert stats["positive_count"].item() == 3
    assert stats["positive_probability"].item() > stats["negative_probability"].item()
    assert stats["probability_margin"].item() > 0.0

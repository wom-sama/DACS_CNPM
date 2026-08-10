import torch

from trkh.models.model import VisionTransformerWithRegisters, classification_logits_from_features
from trkh.training.train import _focus_class_auxiliary_loss_from_features


def test_focus_class_head_adds_binary_logit_feature():
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=16,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=2,
        focus_class_head=True,
        focus_class_index=1,
        focus_class_logit_scale=0.2,
    )
    images = torch.randn(3, 3, 32, 32)
    features = model.forward_features(images)
    logits = classification_logits_from_features(model, features)

    assert logits.shape == (3, 5)
    assert features["focus_class_logits"].shape == (3,)
    assert model.focus_class_route_weights(logits).shape == (3,)


def test_focus_class_auxiliary_loss_is_finite():
    class DummyModel:
        focus_class_index = 1

    features = {"focus_class_logits": torch.tensor([-1.0, 1.0, 0.25])}
    targets = torch.tensor([0, 1, 1], dtype=torch.long)

    loss = _focus_class_auxiliary_loss_from_features(
        model=DummyModel(),
        features=features,
        targets=targets,
        positive_weight=1.25,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

import torch

from trkh.training.train import (
    _boundary_center_margin_loss_for_embedding,
    _boundary_center_margin_loss_from_features,
    _teacher_reliability_weights_for_targets,
)


class DummyHeadModel:
    def head_input_from_features(self, features):
        return features["pooled"]


def test_boundary_center_margin_lower_for_separated_embeddings():
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    separated = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=torch.float32,
    )
    mixed = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )

    separated_loss, separated_terms = _boundary_center_margin_loss_for_embedding(
        separated,
        targets,
        pairs=[(0, 1)],
        margin=0.1,
        temperature=0.2,
        compactness_weight=0.1,
    )
    mixed_loss, mixed_terms = _boundary_center_margin_loss_for_embedding(
        mixed,
        targets,
        pairs=[(0, 1)],
        margin=0.1,
        temperature=0.2,
        compactness_weight=0.1,
    )

    assert separated_terms == 4
    assert mixed_terms == 4
    assert separated_loss.item() < mixed_loss.item()


def test_boundary_center_margin_from_features_uses_requested_sources():
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    features = {
        "pooled": torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [-1.0, 0.0],
                [-0.95, -0.05],
            ],
            dtype=torch.float32,
        )
    }

    loss, stats = _boundary_center_margin_loss_from_features(
        model=DummyHeadModel(),
        features=features,
        targets=targets,
        pairs="0-1",
        sources="head",
        margin=0.1,
        temperature=0.2,
        compactness_weight=0.1,
        num_classes=2,
    )

    assert torch.isfinite(loss)
    assert stats["terms"] == 4
    assert stats["used_sources"] == ["head"]


def test_boundary_center_margin_uses_sample_weights_for_centers_and_anchors():
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],  # noisy class-0 outlier, should be masked out
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=torch.float32,
    )

    unweighted_loss, unweighted_terms = _boundary_center_margin_loss_for_embedding(
        embeddings,
        targets,
        pairs=[(0, 1)],
        margin=0.1,
        temperature=0.2,
        compactness_weight=0.0,
    )
    weighted_loss, weighted_terms = _boundary_center_margin_loss_for_embedding(
        embeddings,
        targets,
        pairs=[(0, 1)],
        margin=0.1,
        temperature=0.2,
        compactness_weight=0.0,
        sample_weights=torch.tensor([1.0, 0.0, 1.0, 1.0]),
    )

    assert unweighted_terms == 4
    assert weighted_terms == 3
    assert weighted_loss.item() < unweighted_loss.item()


def test_teacher_reliability_weights_require_agreement_and_confidence():
    targets = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.60, 0.25, 0.15],
            [0.20, 0.55, 0.25],
            [0.48, 0.47, 0.05],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float32,
    )

    weights = _teacher_reliability_weights_for_targets(
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        teacher_min_confidence=0.50,
        require_agreement=True,
        weight_mode="confidence_margin",
        num_classes=3,
    )

    assert torch.allclose(
        weights,
        torch.tensor([0.21, 0.165, 0.0, 0.35], dtype=torch.float32),
        atol=1e-6,
    )

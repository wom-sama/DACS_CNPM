from __future__ import annotations

import torch
from torch import nn

from trkh.training.losses import SupervisedContrastiveLoss
from trkh.training.train import (
    TeacherGuidedContrastiveMemoryQueue,
    _teacher_guided_contrastive_loss_from_features,
)


def test_teacher_guided_contrastive_filters_teacher_disagreement() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.92, 0.04, 0.02, 0.01, 0.01],
            [0.88, 0.06, 0.03, 0.02, 0.01],
            [0.05, 0.90, 0.03, 0.01, 0.01],
            [0.60, 0.30, 0.06, 0.02, 0.02],
            [0.10, 0.80, 0.05, 0.03, 0.02],
        ],
        dtype=torch.float32,
    )
    criterion = SupervisedContrastiveLoss(temperature=0.2, class_balanced=True)

    loss, stats = _teacher_guided_contrastive_loss_from_features(
        model=nn.Module(),
        features={"pooled": embeddings},
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        criterion=criterion,
        sources="head",
        classes="0,1,2",
        teacher_min_confidence=0.70,
        require_agreement=True,
        num_classes=5,
    )

    assert stats["count"] == 3
    assert stats["fraction"] == 0.6
    assert stats["used_sources"] == ["head"]
    assert torch.isfinite(loss).item()
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all().item()


def test_teacher_guided_contrastive_zero_when_no_positive_pair() -> None:
    embeddings = torch.eye(3, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.03, 0.01, 0.01],
            [0.05, 0.90, 0.03, 0.01, 0.01],
            [0.05, 0.03, 0.90, 0.01, 0.01],
        ],
        dtype=torch.float32,
    )

    loss, stats = _teacher_guided_contrastive_loss_from_features(
        model=nn.Module(),
        features={"pooled": embeddings},
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        criterion=SupervisedContrastiveLoss(temperature=0.2),
        sources="head",
        classes="0,1,2",
        teacher_min_confidence=0.70,
        require_agreement=True,
        num_classes=5,
    )

    assert stats["count"] == 3
    assert torch.isfinite(loss).item()
    assert float(loss.item()) == 0.0


def test_supervised_contrastive_accepts_sample_weights() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 0.4, 0.9, 0.2], dtype=torch.float32)

    loss = SupervisedContrastiveLoss(temperature=0.2)(embeddings, targets, sample_weights=weights)

    assert torch.isfinite(loss).item()
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all().item()


def test_teacher_guided_contrastive_confidence_weighted_stochastic() -> None:
    torch.manual_seed(7)
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.70, 0.20, 0.05, 0.03, 0.02],
            [0.66, 0.22, 0.07, 0.03, 0.02],
            [0.18, 0.68, 0.08, 0.03, 0.03],
            [0.22, 0.63, 0.08, 0.04, 0.03],
        ],
        dtype=torch.float32,
    )

    loss, stats = _teacher_guided_contrastive_loss_from_features(
        model=nn.Module(),
        features={"pooled": embeddings},
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        criterion=SupervisedContrastiveLoss(temperature=0.2),
        sources="head",
        classes="0,1",
        teacher_min_confidence=0.0,
        require_agreement=True,
        weight_mode="confidence_margin",
        stochastic_std=0.05,
        min_reliability=0.0,
        teacher_confidence_power=0.5,
        num_classes=5,
    )

    assert stats["count"] == 4
    assert stats["used_sources"] == ["head"]
    assert stats["reliability_mean"] > 0.0
    assert stats["stochastic_std"] == 0.05
    assert torch.isfinite(loss).item()
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all().item()


def test_teacher_guided_contrastive_memory_queue_backpropagates() -> None:
    queue = TeacherGuidedContrastiveMemoryQueue(
        queue_size=8,
        temperature=0.2,
        class_balanced=True,
        min_memory_count=2,
    )
    queue.enqueue(
        "head",
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    anchors = torch.tensor(
        [
            [0.95, 0.05, 0.0],
            [0.05, 0.95, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss, stats = queue.contrastive_loss(
        source="head",
        anchors=anchors,
        targets=torch.tensor([0, 1], dtype=torch.long),
    )

    assert stats["anchor_count"] == 2
    assert stats["positive_count"] == 4
    assert stats["memory_size"] == 4
    assert torch.isfinite(loss).item()
    loss.backward()
    assert anchors.grad is not None
    assert torch.isfinite(anchors.grad).all().item()


def test_teacher_guided_contrastive_uses_memory_after_enqueue() -> None:
    queue = TeacherGuidedContrastiveMemoryQueue(
        queue_size=8,
        temperature=0.2,
        class_balanced=True,
        min_memory_count=2,
    )
    first_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.03, 0.01, 0.01],
            [0.88, 0.07, 0.03, 0.01, 0.01],
            [0.05, 0.90, 0.03, 0.01, 0.01],
            [0.06, 0.88, 0.04, 0.01, 0.01],
        ],
        dtype=torch.float32,
    )
    criterion = SupervisedContrastiveLoss(temperature=0.2, class_balanced=True)
    _, first_stats = _teacher_guided_contrastive_loss_from_features(
        model=nn.Module(),
        features={"pooled": first_embeddings},
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        criterion=criterion,
        sources="head",
        classes="0,1",
        teacher_min_confidence=0.70,
        require_agreement=True,
        num_classes=5,
        memory_queue=queue,
    )

    assert first_stats["memory_anchor_count"] == 0
    assert queue.size("head") == 4

    second_embeddings = torch.tensor(
        [
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    second_targets = torch.tensor([0, 1], dtype=torch.long)
    second_teacher_probabilities = torch.tensor(
        [
            [0.84, 0.10, 0.03, 0.02, 0.01],
            [0.10, 0.84, 0.03, 0.02, 0.01],
        ],
        dtype=torch.float32,
    )
    loss, second_stats = _teacher_guided_contrastive_loss_from_features(
        model=nn.Module(),
        features={"pooled": second_embeddings},
        targets=second_targets,
        teacher_probabilities=second_teacher_probabilities,
        criterion=criterion,
        sources="head",
        classes="0,1",
        teacher_min_confidence=0.70,
        require_agreement=True,
        num_classes=5,
        memory_queue=queue,
    )

    assert second_stats["count"] == 2
    assert second_stats["memory_anchor_count"] == 2
    assert second_stats["memory_positive_count"] == 4
    assert second_stats["memory_size"] == 4
    assert torch.isfinite(loss).item()
    loss.backward()
    assert second_embeddings.grad is not None
    assert torch.isfinite(second_embeddings.grad).all().item()

from __future__ import annotations

import unittest

import torch

from trkh.training.train import (
    _distillation_loss,
    _teacher_non_target_distillation_loss_from_logits,
)


class PretrainedDistillationTest(unittest.TestCase):
    def test_distillation_loss_is_finite_and_backpropagates(self) -> None:
        student_logits = torch.tensor(
            [[1.0, 0.0, -1.0], [0.1, 0.2, 0.3]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_logits = torch.tensor(
            [[0.5, 1.5, -0.5], [0.0, -0.5, 1.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor([1, 2], dtype=torch.int64)

        loss = _distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            temperature=2.0,
            focus_class_index=1,
            focus_class_weight=1.5,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(student_logits.grad)
        self.assertTrue(torch.isfinite(student_logits.grad).all().item())
        self.assertGreater(float(student_logits.grad.abs().sum().item()), 0.0)

    def test_teacher_non_target_distillation_masks_hard_target(self) -> None:
        labels = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
        teacher_probabilities = torch.tensor(
            [
                [0.80, 0.05, 0.10, 0.00, 0.05],
                [0.08, 0.75, 0.12, 0.00, 0.05],
                [0.10, 0.05, 0.80, 0.00, 0.05],
                [0.05, 0.06, 0.09, 0.00, 0.80],
            ],
            dtype=torch.float32,
        )
        matching_logits = torch.tensor(
            [
                [4.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 4.0, 2.0, 0.0, 0.0],
                [2.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        mismatched_logits = torch.tensor(
            [
                [4.0, 2.0, 0.0, 0.0, 0.0],
                [2.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 4.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
        )

        matching_loss, fraction, mean_confidence = (
            _teacher_non_target_distillation_loss_from_logits(
                logits=matching_logits,
                teacher_probabilities=teacher_probabilities,
                hard_labels=labels,
                targets=labels,
                classes="0,1,2,4",
                temperature=2.0,
                teacher_min_confidence=0.70,
                require_agreement=True,
            )
        )
        mismatched_loss, mismatched_fraction, _ = (
            _teacher_non_target_distillation_loss_from_logits(
                logits=mismatched_logits,
                teacher_probabilities=teacher_probabilities,
                hard_labels=labels,
                targets=labels,
                classes="0,1,2,4",
                temperature=2.0,
                teacher_min_confidence=0.70,
                require_agreement=True,
            )
        )

        self.assertTrue(torch.isfinite(matching_loss).item())
        self.assertLess(float(matching_loss.item()), float(mismatched_loss.item()))
        self.assertEqual(fraction, 1.0)
        self.assertEqual(mismatched_fraction, 1.0)
        self.assertGreater(float(mean_confidence.item()), 0.0)
        matching_loss.backward()
        self.assertIsNotNone(matching_logits.grad)
        self.assertTrue(torch.isfinite(matching_logits.grad).all().item())


if __name__ == "__main__":
    unittest.main()

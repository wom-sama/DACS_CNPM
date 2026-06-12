from __future__ import annotations

import unittest

import torch

from trkh.training.train import _distillation_loss


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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from trkh.tools.calibrate_classification_logits import (
    fit_logit_bias,
    metrics_for_bias,
)


class LogitBiasCalibrationTest(unittest.TestCase):
    def test_validation_bias_improves_shifted_class_boundary(self) -> None:
        targets = torch.tensor([0] * 20 + [1] * 10 + [2] * 20, dtype=torch.int64)
        probabilities = torch.full((50, 3), 0.05, dtype=torch.float64)
        probabilities[:20, 0] = 0.55
        probabilities[:20, 1] = 0.40
        probabilities[20:30, 0] = 0.48
        probabilities[20:30, 1] = 0.47
        probabilities[30:, 2] = 0.90
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        log_probabilities = probabilities.log()
        class_names = ["zero", "one", "two"]

        baseline = metrics_for_bias(
            log_probabilities,
            targets,
            class_names,
            [0.0, 0.0, 0.0],
        )
        bias, adjusted, trace = fit_logit_bias(
            log_probabilities,
            targets,
            class_names,
            objective="macro_class1",
            class1_index=1,
            class1_weight=0.25,
            reference_class=2,
            max_abs_bias=0.5,
            search_steps=(0.1, 0.02),
        )

        self.assertTrue(trace)
        self.assertEqual(bias[2], 0.0)
        self.assertGreater(adjusted["macro_f1"], baseline["macro_f1"])
        self.assertGreater(
            adjusted["per_class"][1]["f1"],
            baseline["per_class"][1]["f1"],
        )


if __name__ == "__main__":
    unittest.main()

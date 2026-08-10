from __future__ import annotations

import unittest

import torch

from trkh.tools.fit_class1_reroute_calibration import (
    RerouteRule,
    apply_reroute_rule,
    fit_reroute_rule,
)


class Class1RerouteCalibrationTest(unittest.TestCase):
    def test_rejects_low_margin_focus_false_positive(self) -> None:
        probabilities = torch.tensor(
            [
                [0.42, 0.43, 0.05, 0.05, 0.05],
                [0.10, 0.62, 0.10, 0.08, 0.10],
                [0.05, 0.25, 0.24, 0.22, 0.24],
            ],
            dtype=torch.float64,
        )
        rule = RerouteRule(focus_class_index=1, retain_threshold=0.30, retain_margin=0.02)

        predictions, delta = apply_reroute_rule(probabilities, rule)

        self.assertEqual(predictions.tolist(), [0, 1, 2])
        self.assertEqual(delta["changed_from_focus"], 2)
        self.assertEqual(delta["changed_to_focus"], 0)

    def test_fit_rule_improves_focus_precision(self) -> None:
        class_names = ["zero", "one", "two", "three", "four"]
        targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4], dtype=torch.int64)
        probabilities = torch.tensor(
            [
                [0.42, 0.43, 0.05, 0.05, 0.05],
                [0.44, 0.45, 0.04, 0.03, 0.04],
                [0.05, 0.62, 0.11, 0.11, 0.11],
                [0.10, 0.58, 0.12, 0.10, 0.10],
                [0.10, 0.22, 0.40, 0.14, 0.14],
                [0.10, 0.23, 0.38, 0.14, 0.15],
                [0.05, 0.08, 0.12, 0.65, 0.10],
                [0.05, 0.08, 0.10, 0.12, 0.65],
            ],
            dtype=torch.float64,
        )

        rule, adjusted, trace = fit_reroute_rule(
            targets,
            probabilities,
            class_names,
            focus_class_index=1,
            retain_thresholds=(None, 0.30),
            retain_margins=(None, 0.02),
            enable_rescue=False,
        )

        self.assertTrue(trace)
        self.assertEqual(rule.focus_class_index, 1)
        self.assertGreaterEqual(adjusted["per_class"][1]["precision"], 0.99)
        self.assertGreater(adjusted["macro_f1"], 0.70)


if __name__ == "__main__":
    unittest.main()

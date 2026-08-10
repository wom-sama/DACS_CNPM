from trkh.tools.evaluate_counterfactual_color_guard import (
    GuardConfig,
    apply_guard_predictions,
    compute_classification_metrics,
    sweep_guard_configs,
)


def test_apply_guard_routes_only_focus_predictions_with_sufficient_drop():
    rows = [
        {
            "target_index": 0,
            "clean_pred": 1,
            "desat_pred": 0,
            "clean_margin": 0.10,
            "clean_confidence": 0.36,
            "clean_prob_1": 0.36,
            "desat_prob_1": 0.20,
        },
        {
            "target_index": 1,
            "clean_pred": 1,
            "desat_pred": 0,
            "clean_margin": 0.30,
            "clean_confidence": 0.50,
            "clean_prob_1": 0.50,
            "desat_prob_1": 0.30,
        },
        {
            "target_index": 2,
            "clean_pred": 2,
            "desat_pred": 1,
            "clean_margin": 0.02,
            "clean_confidence": 0.40,
            "clean_prob_1": 0.10,
            "desat_prob_1": 0.30,
        },
    ]

    predictions = apply_guard_predictions(
        rows,
        config=GuardConfig(drop_threshold=0.12, max_clean_margin=0.20, max_clean_confidence=0.40),
        focus_class=1,
        num_classes=3,
    )

    assert predictions == [0, 1, 2]


def test_sweep_guard_configs_finds_focus_false_positive_fix():
    rows = [
        {
            "target_index": 0,
            "clean_pred": 1,
            "desat_pred": 0,
            "clean_margin": 0.10,
            "clean_confidence": 0.36,
            "clean_prob_1": 0.40,
            "desat_prob_1": 0.20,
        },
        {
            "target_index": 1,
            "clean_pred": 1,
            "desat_pred": 0,
            "clean_margin": 0.30,
            "clean_confidence": 0.52,
            "clean_prob_1": 0.52,
            "desat_prob_1": 0.30,
        },
        {
            "target_index": 2,
            "clean_pred": 2,
            "desat_pred": 2,
            "clean_margin": 0.30,
            "clean_confidence": 0.80,
            "clean_prob_1": 0.05,
            "desat_prob_1": 0.04,
        },
    ]

    config, metrics = sweep_guard_configs(
        rows,
        focus_class=1,
        num_classes=3,
        focus_weight=0.25,
        drop_thresholds=[0.0, 0.12],
        max_margins=[0.12, 1.0],
        max_confidences=[0.40, 1.0],
    )

    assert config.max_clean_margin <= 0.12
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"][1]["f1"] == 1.0


def test_compute_classification_metrics_reports_focus_class_f1():
    metrics = compute_classification_metrics(
        [0, 1, 1, 2],
        [0, 1, 0, 2],
        num_classes=3,
    )

    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert metrics["per_class"][1]["recall"] == 0.5

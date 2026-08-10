from trkh.tools.audit_inceptionnext_atto_postsmoke import assess_postsmoke


def _robustness(delta: float = 0.0, recall_delta: float = 0.0):
    return [
        {"macro_f1_delta": delta, "class1_recall_delta": recall_delta}
        for _ in range(5)
    ]


def _xai_means():
    return {
        "attention_foreground_mass": 0.75,
        "gradcam_foreground_mass": 0.82,
        "gradcam_border_mass": 0.20,
        "object_desaturate_original_prediction_drop": 0.06,
        "object_desaturate_target_probability_drop": 0.05,
        "background_gray_original_prediction_drop": 0.006,
        "background_gray_target_probability_drop": 0.005,
        "background_blur_original_prediction_drop": 0.003,
        "background_blur_target_probability_drop": 0.004,
    }


def test_postsmoke_gate_accepts_locked_robust_surface_evidence():
    result = assess_postsmoke(
        pair_gate_passed=True,
        robustness_rows=_robustness(),
        trace_passed=True,
        candidate_xai_means=_xai_means(),
        attention_sources={"forward_features.return_attention.blocks[7].mhsa_probability": 16},
        grad_rollout_fallback_cases=0,
    )
    assert result["stage_b_passed"] is True
    assert result["five_epoch_permission"] is True


def test_postsmoke_gate_rejects_weak_robustness_and_silent_fallback():
    rows = _robustness(delta=-0.01, recall_delta=-0.051)
    means = _xai_means()
    means["gradcam_foreground_mass"] = 0.40
    result = assess_postsmoke(
        pair_gate_passed=False,
        robustness_rows=rows,
        trace_passed=False,
        candidate_xai_means=means,
        attention_sources={"fallback.square_tokens": 16},
        grad_rollout_fallback_cases=1,
    )
    assert result["stage_b_passed"] is False
    assert "pair_metric_gate_passed" in result["failed_checks"]
    assert "robustness_three_of_five" in result["failed_checks"]
    assert "grad_rollout_no_fallback" in result["failed_checks"]
    assert "stem_gradcam_foreground_focused" in result["failed_checks"]

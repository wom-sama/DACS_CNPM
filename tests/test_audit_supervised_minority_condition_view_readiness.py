from __future__ import annotations

import copy
import os

import torch

from trkh.tools.audit_supervised_minority_condition_view_readiness import (
    CONDITIONS,
    EXPECTED_PROBE_PARAMETERS,
    EXPECTED_REPRESENTATION_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS,
    LOCKED_CONTRASTIVE_SCHEDULE_SHA256,
    LOCKED_PROBE_SCHEDULE_SHA256,
    STAGE1_ETA_MIN,
    STAGE1_LEARNING_RATE,
    STAGE1_WARMUP_START,
    ConditionViewVariant,
    _IsolatedCandidateExport,
    _counterfactual_tensors,
    _equation_diagnostics,
    _positive_mask,
    _required_xai_events,
    _stage1_learning_rate,
    assess_pretraining_selectivity,
    assess_representation_mechanism,
    assess_train_holdout_behavior,
    independent_contrastive_loss,
    parse_args,
    supervised_contrastive_loss,
)


def test_protocol_defaults_and_locked_schedule_hashes() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == 32
    assert args.num_workers == 4
    assert args.xai_batch_size == 4
    assert args.seed == 42 and args.fold == 0
    assert LOCKED_CONTRASTIVE_SCHEDULE_SHA256 == (
        "d73ba25938afb8ba69061ea7c1ae79ac58b254d8a7c9dbb952566b130da1b52a"
    )
    assert LOCKED_PROBE_SCHEDULE_SHA256 == (
        "bf5568009a9e0480aec7f6f8d7f3cce7f08409ac74325930747f5d022a4f10f0"
    )


def test_stage1_learning_rate_has_locked_endpoints() -> None:
    assert _stage1_learning_rate(0) == STAGE1_WARMUP_START
    assert _stage1_learning_rate(1) == STAGE1_LEARNING_RATE
    assert _stage1_learning_rate(2) == STAGE1_LEARNING_RATE
    assert abs(_stage1_learning_rate(19) - STAGE1_ETA_MIN) <= 1e-12


def test_variant_schema_zero_residual_and_rng_isolation() -> None:
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        variant = ConditionViewVariant()
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)
    representation = sum(
        parameter.numel() for parameter in variant.representation_parameters()
    )
    probe = sum(parameter.numel() for parameter in variant.probe.parameters())
    assert representation == EXPECTED_REPRESENTATION_PARAMETERS == 132_288
    assert probe == EXPECTED_PROBE_PARAMETERS == 1_285
    assert representation + probe == EXPECTED_TOTAL_PARAMETERS == 133_573
    pooled = torch.randn(7, 256)
    raw_logits = torch.randn(7, 5)
    assert torch.equal(variant.adapted_feature(pooled), pooled)
    assert torch.equal(variant.deployed_logits(pooled, raw_logits), raw_logits)


def test_supervised_minority_and_standard_masks_have_exact_positive_counts() -> None:
    labels = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    minority = _positive_mask(labels, supervised_minority=True).repeat(2, 2)
    standard = _positive_mask(labels, supervised_minority=False).repeat(2, 2)
    self_mask = torch.eye(2 * labels.numel(), dtype=torch.bool)
    minority = minority.masked_fill(self_mask, 0.0)
    standard = standard.masked_fill(self_mask, 0.0)
    anchors = labels.repeat(2)
    assert torch.equal(
        minority.sum(dim=1)[anchors.eq(0)],
        torch.ones(2 * int(labels.eq(0).sum().item())),
    )
    assert torch.equal(
        minority.sum(dim=1)[anchors.eq(1)],
        torch.full((2 * int(labels.eq(1).sum().item()),), 3.0),
    )
    assert torch.equal(
        standard.sum(dim=1)[anchors.eq(0)],
        torch.full((2 * int(labels.eq(0).sum().item()),), 5.0),
    )
    assert torch.equal(
        standard.sum(dim=1)[anchors.eq(1)],
        torch.full((2 * int(labels.eq(1).sum().item()),), 3.0),
    )


def test_vectorized_losses_match_independent_flat_equation() -> None:
    torch.manual_seed(7)
    features = torch.randn(9, 2, 13)
    labels = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0])
    for supervised_minority in (False, True):
        observed, diagnostics = supervised_contrastive_loss(
            features,
            labels,
            supervised_minority=supervised_minority,
        )
        expected = independent_contrastive_loss(
            features,
            labels,
            supervised_minority=supervised_minority,
        )
        assert torch.allclose(observed, expected, atol=1e-6, rtol=0.0)
        assert diagnostics["finite"] is True


def test_equation_diagnostics_match_official_source_and_independent_replay() -> None:
    diagnostics = _equation_diagnostics()
    assert diagnostics["matches_independent_and_official_within_1e_6"] is True
    assert diagnostics["maximum_absolute_error"] <= 1e-6
    assert diagnostics["supmin_nonclass1_positive_count_exactly_one"] is True
    assert diagnostics["supmin_class1_positive_count_exact"] is True


def _prediction_row(
    sample_index: int, target: int, prediction: int, class1_probability: float
) -> dict[str, object]:
    remainder = (1.0 - class1_probability) / 4.0
    row: dict[str, object] = {
        "sample_index": sample_index,
        "target": target,
        "prediction": prediction,
    }
    for class_index in range(5):
        row[f"prob_{class_index}"] = (
            class1_probability if class_index == 1 else remainder
        )
    return row


def test_xai_event_selection_covers_clean_changes_and_shifted_fp_events() -> None:
    predictions = {}
    for condition in CONDITIONS:
        raw = [
            _prediction_row(10, 1, 1, 0.60),
            _prediction_row(11, 0, 1, 0.55),
            _prediction_row(12, 2, 2, 0.20),
        ]
        candidate = copy.deepcopy(raw)
        if condition == "clean":
            candidate[0] = _prediction_row(10, 1, 0, 0.35)
        if condition == "lighting_dim":
            candidate[1] = _prediction_row(11, 0, 0, 0.25)
            candidate[2] = _prediction_row(12, 2, 1, 0.52)
        predictions[condition] = {
            "raw": raw,
            "ce_identity": copy.deepcopy(raw),
            "standard_supcon": copy.deepcopy(raw),
            "supervised_minority": candidate,
        }
    events, categories, required = _required_xai_events(predictions, minimum_rows=0)
    assert ("clean", 10) in events
    assert "class1_tp_break" in categories["clean:10"]
    assert "clean_class1_related_change" in categories["clean:10"]
    assert "restricted_fp_removal" in categories["lighting_dim:11"]
    assert "restricted_fp_creation" in categories["lighting_dim:12"]
    assert required == set(categories)


def test_bbox_counterfactuals_are_finite_and_region_specific() -> None:
    image = torch.zeros(3, 8, 8)
    image[0] = 1.0
    image[1, :, 4:] = 0.5
    bbox = torch.tensor([0.5, 0.5, 0.5, 0.5])
    variants, contract = _counterfactual_tensors(
        image=image,
        bbox=bbox,
        image_valid_mask=torch.ones(8, 8, dtype=torch.bool),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    assert variants.shape == (3, 3, 8, 8)
    assert torch.isfinite(variants).all()
    assert contract["names"] == [
        "object_desaturate",
        "background_blur",
        "background_gray",
    ]
    assert not torch.equal(variants[0, :, 3, 3], image[:, 3, 3])
    assert torch.equal(variants[0, :, 0, 0], image[:, 0, 0])
    assert torch.equal(variants[2, :, 3, 3], image[:, 3, 3])
    assert not torch.equal(variants[2, :, 0, 0], image[:, 0, 0])


def test_isolated_deployment_wrapper_excludes_projector() -> None:
    variant = ConditionViewVariant().eval()
    wrapper = _IsolatedCandidateExport(variant).eval()
    assert all("projector" not in name for name, _parameter in wrapper.named_parameters())
    pooled = torch.randn(4, 256)
    logits = torch.randn(4, 5)
    with torch.inference_mode():
        assert torch.equal(wrapper(pooled, logits), logits)


def _selectivity_metrics(*, passing: bool = True) -> dict[str, object]:
    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = {
            "finite": True,
            "tp_hard_auroc": 0.66,
            "tp_hard_neighbor_gap": 0.04,
            "class1_cac": 0.62,
            "clean_to_condition_saa": 0.91,
        }
    if not passing:
        conditions["lighting_dim"]["tp_hard_auroc"] = 0.59
    return {"conditions": conditions}


def test_pretraining_selectivity_gate_is_all_or_nothing() -> None:
    passing = assess_pretraining_selectivity(_selectivity_metrics())
    assert passing["representation_training_authorized"] is True
    rejected = assess_pretraining_selectivity(_selectivity_metrics(passing=False))
    assert rejected["representation_training_authorized"] is False
    assert "dim_tp_hard_auroc_gte_0p60" in rejected["failed_checks"]


def _representation_metrics(*, candidate: bool = False) -> dict[str, object]:
    output = {"conditions": {}}
    for condition in CONDITIONS:
        output["conditions"][condition] = {
            "class1_cac": 0.70 if candidate else 0.66,
            "nonclass1_cac": 0.96 if candidate else 0.97,
            "tp_hard_auroc": 0.72 if candidate else 0.68,
            "clean_to_condition_saa": 0.91 if candidate else 0.92,
            "class1_mean_pairwise_cosine": 0.31 if candidate else 0.27,
            "nonclass1_mean_off_instance_cosine": 0.10 if candidate else 0.09,
        }
    return output


def test_representation_gate_requires_candidate_mechanism() -> None:
    representations = {
        "raw": _representation_metrics(),
        "standard_supcon": _representation_metrics(),
        "supervised_minority": _representation_metrics(candidate=True),
    }
    representations["standard_supcon"]["conditions"]["clean"][
        "tp_hard_auroc"
    ] = 0.70
    training = {
        "supervised_minority": {
            "final_to_first_loss_ratio": 0.80,
            "projection_diagnostics": {
                "adapter_residual_rms": 0.2,
                "projection_all_dimensions_finite_variance": True,
                "projection_effective_rank": 30.0,
            },
        }
    }
    result = assess_representation_mechanism(
        representations=representations,
        representation_training=training,
    )
    assert result["all_gates_passed"] is True
    failed_training = copy.deepcopy(training)
    failed_training["supervised_minority"]["projection_diagnostics"][
        "projection_effective_rank"
    ] = 15.9
    result = assess_representation_mechanism(
        representations=representations,
        representation_training=failed_training,
    )
    assert result["all_gates_passed"] is False
    assert "projection_effective_rank_gte_16" in result["failed_checks"]


def _comparison(
    *,
    macro: float = 0.002,
    f1: float = 0.006,
    precision: float = 0.012,
    recall: float = 0.0,
    fp_reduction: int = 3,
    corrections: int = 5,
    harms: int = 1,
    breaks: int = 0,
    rescues: int = 0,
) -> dict[str, object]:
    return {
        "delta": {
            "macro_f1": macro,
            "class1_f1": f1,
            "class1_precision": precision,
            "class1_recall": recall,
        },
        "transitions": {
            "focus_tp_break": breaks,
            "focus_fn_rescue": rescues,
            "restricted_focus_fp_reduction": fp_reduction,
            "candidate_correction": corrections,
            "candidate_harm": harms,
        },
        "maximum_nonfocus_f1_drop": 0.004,
    }


def _passing_comparisons() -> dict[str, object]:
    output = {}
    for condition in CONDITIONS:
        raw = _comparison()
        if condition != "clean":
            raw = _comparison(f1=0.004, precision=0.005, fp_reduction=1)
        output[condition] = {
            "raw_candidate": raw,
            "ce_candidate": _comparison(
                macro=0.001, f1=0.004, precision=0.004, fp_reduction=1
            ),
            "standard_candidate": _comparison(
                macro=0.001, f1=0.004, precision=0.004, fp_reduction=1
            ),
        }
    return output


def test_behavior_gate_rejects_precision_gain_from_tp_suppression() -> None:
    passing = assess_train_holdout_behavior(_passing_comparisons())
    assert passing["all_gates_passed"] is True
    comparisons = _passing_comparisons()
    comparisons["clean"]["raw_candidate"] = _comparison(
        f1=0.006,
        precision=0.030,
        recall=-0.004,
        fp_reduction=8,
        corrections=9,
        harms=2,
        breaks=2,
        rescues=0,
    )
    rejected = assess_train_holdout_behavior(comparisons)
    assert rejected["all_gates_passed"] is False
    assert "clean_tp_breaks_lte_1" in rejected["failed_checks"]
    assert "clean_fn_rescues_gte_tp_breaks" in rejected["failed_checks"]

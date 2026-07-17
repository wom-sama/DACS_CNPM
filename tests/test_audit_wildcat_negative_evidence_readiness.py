from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trkh.tools.audit_wildcat_negative_evidence_readiness import (
    ALPHA,
    BATCH_SIZE,
    EMBED_DIM,
    EPOCHS,
    EXPECTED_HEAD_PARAMETERS,
    GRID_HEIGHT,
    GRID_WIDTH,
    K_FRACTION,
    MAPS_PER_CLASS,
    PREFIX_TOKENS,
    PROJECTED_CHANNELS,
    SELECTED_CELLS,
    DenseBlock2Capture,
    WildcatSpatialHead,
    _bbox_mask,
    _finite_difference_error,
    _independent_pool_oracle,
    _official_source_classes,
    _prepare_export_inputs,
    _replay_interrupted_pre_xai,
    _saliency_repeat_diagnostics,
    _signed_overlay,
    _source_paths,
    _summarize_mechanism_rows,
    _train_heads,
    _verify_interrupted_reference,
    _xai_selection,
    assess_stage_a,
    class_wise_pool,
    locked_training_order,
    parse_args,
    positive_k,
    spatial_pool_components,
    wildcat_pool,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 32
    assert args.epochs == EPOCHS == 20
    assert args.maps_per_class == MAPS_PER_CLASS == 4
    assert args.k_fraction == K_FRACTION == 0.2
    assert args.alpha == ALPHA == 0.7
    assert SELECTED_CELLS == 51
    assert args.erratum.name.endswith("IMPLEMENTATION_ERRATUM_20260717.md")
    assert "interrupted_xai_determinism" in args.interrupted_reference.name


def test_interrupted_reference_matches_locked_erratum_hashes() -> None:
    paths = _source_paths(parse_args([]))
    reference = _verify_interrupted_reference(paths)
    assert reference["artifact_count"] == 5
    assert reference["contains_summary"] is False
    assert reference["contains_xai"] is False
    assert reference["forbidden_checkpoint_count"] == 0
    replay = _replay_interrupted_pre_xai(
        output_dir=paths["interrupted_reference"],
        reference_dir=paths["interrupted_reference"],
    )
    assert replay["all_exact"] is True


def test_positive_k_matches_official_rounding_contract() -> None:
    assert positive_k(0.2, 256) == 51
    assert positive_k(0.5, 7) == 4
    assert positive_k(3, 7) == 3
    assert positive_k(99, 7) == 7
    assert positive_k(0, 7) == 0


def test_classwise_and_spatial_pool_match_independent_oracle_and_gradients() -> None:
    torch.manual_seed(9)
    projected = torch.randn(
        2,
        PROJECTED_CHANNELS,
        GRID_HEIGHT,
        GRID_WIDTH,
        dtype=torch.float64,
        requires_grad=True,
    )
    maps = class_wise_pool(projected)
    observed = wildcat_pool(maps)
    expected, expected_top, expected_bottom = _independent_pool_oracle(
        projected, alpha=ALPHA
    )
    components = spatial_pool_components(maps)
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)
    assert torch.equal(components["top_mean"], expected_top)
    assert torch.allclose(
        components["bottom_mean"], expected_bottom, atol=1e-15, rtol=0.0
    )
    observed_gradient = torch.autograd.grad(
        observed.square().mean(), projected, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected.square().mean(), projected)[0]
    assert torch.allclose(observed_gradient, expected_gradient, atol=1e-15, rtol=0.0)


def test_top_only_and_same_weight_ablation_have_locked_scaling() -> None:
    maps = torch.arange(2 * 5 * 16 * 16, dtype=torch.float32).reshape(2, 5, 16, 16)
    components = spatial_pool_components(maps)
    assert torch.equal(wildcat_pool(maps, alpha=0.0), components["top_mean"])
    full = wildcat_pool(maps, alpha=ALPHA)
    assert torch.equal(
        full,
        (components["top_mean"] + ALPHA * components["bottom_mean"]) / 2.0,
    )
    assert torch.equal(components["top_mean"] / 2.0, components["top_mean"] * 0.5)


def test_official_source_extraction_matches_candidate_equation() -> None:
    source = _source_paths(parse_args([]))["official_pooling"]
    assert Path(source).is_file()
    official_wildcat, official_classwise = _official_source_classes(source)
    projected = torch.randn(2, PROJECTED_CHANNELS, 16, 16)
    classwise = official_classwise()
    classwise.num_maps = MAPS_PER_CLASS
    official_maps = classwise.forward(projected)
    official = official_wildcat()
    official.kmax = K_FRACTION
    official.kmin = K_FRACTION
    official.alpha = ALPHA
    expected = wildcat_pool(class_wise_pool(projected))
    assert torch.allclose(official.forward(official_maps), expected, atol=2e-7, rtol=0.0)


def test_finite_difference_error_is_within_protocol() -> None:
    assert _finite_difference_error() <= 1e-4


def test_head_parameter_and_output_contracts() -> None:
    torch.manual_seed(42)
    prototype = WildcatSpatialHead("gap")
    state = {key: value.clone() for key, value in prototype.state_dict().items()}
    features = torch.randn(3, EMBED_DIM, GRID_HEIGHT, GRID_WIDTH)
    for mode in ("gap", "top_only", "wildcat"):
        head = WildcatSpatialHead(mode)
        head.load_state_dict(state)
        assert sum(parameter.numel() for parameter in head.parameters()) == EXPECTED_HEAD_PARAMETERS
        assert head(features).shape == (3, 5)
    candidate = WildcatSpatialHead("wildcat")
    candidate.load_state_dict(state)
    assert candidate.no_bottom_logits(features).shape == (3, 5)


class _TupleBlock(nn.Module):
    def forward(self, value: torch.Tensor):
        return value, torch.ones(1)


class _CaptureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_prefix_tokens = PREFIX_TOKENS
        self.blocks = nn.ModuleList((nn.Identity(), _TupleBlock()))


def test_dense_block2_capture_extracts_tuple_output_suffix() -> None:
    model = _CaptureModel()
    tokens = torch.randn(2, PREFIX_TOKENS + 256, EMBED_DIM)
    with DenseBlock2Capture(model) as capture:
        model.blocks[1](tokens)
        dense = capture.take()
    assert dense.shape == (2, EMBED_DIM, GRID_HEIGHT, GRID_WIDTH)
    assert capture.call_count == 1
    assert capture.value is None
    assert capture.output_shape == (2, PREFIX_TOKENS + 256, EMBED_DIM)


def test_bbox_mask_uses_cell_centers_and_never_returns_empty() -> None:
    boxes = torch.tensor(((0.5, 0.5, 0.5, 0.5), (0.01, 0.01, 0.001, 0.001)))
    mask = _bbox_mask(boxes, height=16, width=16)
    assert mask.shape == (2, 16, 16)
    assert bool(mask.flatten(1).any(dim=1).all())
    assert int(mask[0].sum()) == 64


def test_locked_training_order_is_deterministic_and_epoch_complete() -> None:
    first = locked_training_order((1, 2, 3, 4), seed=42, epochs=3)
    second = locked_training_order((1, 2, 3, 4), seed=42, epochs=3)
    assert first == second
    assert len(first) == 12
    assert all(sorted(first[start : start + 4]) == [1, 2, 3, 4] for start in range(0, 12, 4))


def test_prepare_export_inputs_uses_normal_inference_bbox_and_mask() -> None:
    images = torch.randn(2, 3, 8, 8)
    bbox = torch.tensor(((0.1, 0.2, 0.3, 0.4), (0.2, 0.3, 0.4, 0.5)))
    crop_bbox = torch.ones_like(bbox)
    image_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    image_mask[:, 1:-1, 1:-1] = True
    prepared_images, prepared_bbox, prepared_mask = _prepare_export_inputs(
        (
            images,
            torch.tensor((0, 1)),
            {"bbox": bbox, "crop_bbox": crop_bbox, "image_mask": image_mask},
        )
    )
    assert torch.equal(prepared_images, images[:1])
    assert torch.equal(prepared_bbox, bbox[:1])
    assert not torch.equal(prepared_bbox, crop_bbox[:1])
    assert torch.equal(prepared_mask, image_mask[:1])
    assert prepared_mask.dtype == torch.bool


def test_small_cpu_head_training_records_optimizer_and_logit_telemetry(
    tmp_path: Path,
) -> None:
    args = parse_args([])
    args.epochs = 1
    args.batch_size = 2
    sample_indices = np.asarray((1, 3, 5, 7), dtype=np.int64)
    targets = np.asarray((0, 1, 2, 4), dtype=np.int64)
    cache_path = tmp_path / "features.f16"
    cache = np.memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(4, EMBED_DIM, GRID_HEIGHT, GRID_WIDTH),
    )
    generator = np.random.default_rng(17)
    cache[:] = generator.normal(0.0, 0.1, size=cache.shape).astype(np.float16)
    cache.flush()
    torch.manual_seed(42)
    prototype = WildcatSpatialHead("gap")
    state = prototype.state_dict()
    prototypes = {}
    for mode in ("gap", "top_only", "wildcat"):
        head = WildcatSpatialHead(mode)
        head.load_state_dict(state)
        prototypes[mode] = head
    order = locked_training_order(sample_indices.tolist(), seed=42, epochs=1)
    _trained, summary = _train_heads(
        prototypes=prototypes,
        cache=cache,
        targets=targets,
        sample_indices=sample_indices,
        order=order,
        args=args,
        device=torch.device("cpu"),
    )
    assert summary["updates"] == {"gap": 2, "top_only": 2, "wildcat": 2}
    assert summary["optimizer_configurations_equal"] is True
    assert summary["all_gradients_seen_nonzero"] is True
    assert summary["all_final_parameters_finite"] is True
    for mode in ("gap", "top_only", "wildcat"):
        row = summary["history"][mode][0]
        assert row["mean_absolute_logit"] > 0.0
        assert row["rms_logit"] > 0.0
        assert row["maximum_absolute_logit"] > 0.0
    del cache


def _selection_row(
    sample_index: int, target: int, prediction: int, probability: float
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "target": target,
        "prediction": prediction,
        "prob_1": probability,
    }


def test_xai_selection_keeps_all_required_events_but_bounds_secondary_rows() -> None:
    top_rows = []
    candidate_rows = []
    no_bottom_rows = []
    for index in range(20):
        if index == 19:
            target, top_prediction, candidate_prediction = 0, 1, 0
        else:
            target, top_prediction, candidate_prediction = 3, 2, 3
        top_rows.append(
            _selection_row(index, target, top_prediction, 0.8 - index / 100.0)
        )
        candidate_rows.append(
            _selection_row(index, target, candidate_prediction, index / 100.0)
        )
        no_bottom_rows.append(
            _selection_row(index, target, candidate_prediction, index / 100.0)
        )
    selected, categories, required = _xai_selection(
        {
            "top_only": {"clean": top_rows},
            "wildcat": {"clean": candidate_rows},
            "no_bottom": {"clean": no_bottom_rows},
        }
    )
    assert required == [19]
    assert selected == [*range(15), 19]
    assert "vs_top_restricted_fp_remove" in categories[19]
    assert "vs_top_correction" in categories[19]


def _mechanism_row(
    *, condition: str, sample_index: int, target: int, prediction: int, top_bbox: float
) -> dict[str, object]:
    row: dict[str, object] = {
        "condition": condition,
        "sample_index": sample_index,
        "target": target,
        "wildcat_prediction": prediction,
        "gap_direction_margin": float(target == 1),
        "top_only_direction_margin": float(target == 1),
        "wildcat_direction_margin": float(target == 1),
        "no_bottom_direction_margin": float(target == 1),
        "wildcat_bottom_margin": 1.0 if target == 1 else -1.0,
        "class1_top_mean": 0.5,
        "class1_bottom_mean": -0.2,
        "class1_top_bbox_fraction": top_bbox,
        "class1_bottom_bbox_fraction": 0.2,
        "bbox_grid_fraction": 0.4,
        "class1_map_entropy": 0.8,
        "class_map_effective_rank": 3.0,
    }
    for class_index in range(5):
        row[f"class_{class_index}_top_mean"] = 0.5 + class_index
        row[f"class_{class_index}_bottom_mean"] = -0.2 - class_index
    return row


def test_mechanism_bbox_summary_uses_candidate_true_positives_only() -> None:
    values = []
    for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast"):
        values.extend(
            (
                _mechanism_row(
                    condition=condition,
                    sample_index=1,
                    target=1,
                    prediction=1,
                    top_bbox=0.8,
                ),
                _mechanism_row(
                    condition=condition,
                    sample_index=2,
                    target=1,
                    prediction=0,
                    top_bbox=0.0,
                ),
                _mechanism_row(
                    condition=condition,
                    sample_index=3,
                    target=0,
                    prediction=0,
                    top_bbox=0.1,
                ),
            )
        )
    summary = _summarize_mechanism_rows(values)
    assert summary["clean"]["candidate_class1_tp_rows"] == 1
    assert summary["clean"]["tp_top_bbox_fraction"] == 0.8
    assert summary["clean"]["tp_top_alignment_above_chance"] == 0.4
    assert summary["clean"]["bottom_margin_auroc"] == 1.0
    assert len(summary["clean"]["mean_top_by_class"]) == 5


def test_signed_overlay_preserves_shape_and_distinguishes_sign() -> None:
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    signed = np.zeros((2, 2), dtype=np.float32)
    signed[0, 0] = -1.0
    signed[1, 1] = 1.0
    overlay = np.asarray(_signed_overlay(rgb, signed))
    assert overlay.shape == rgb.shape
    assert not np.array_equal(overlay[0, 0], overlay[-1, -1])


def test_saliency_repeat_diagnostics_enforces_erratum_thresholds() -> None:
    saliency = torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8)
    class_maps = torch.randn(1, 5, 16, 16)
    top_indices = torch.arange(51).reshape(1, 1, 51).expand(1, 5, 51)
    bottom_indices = torch.arange(205, 256).reshape(1, 1, 51).expand(1, 5, 51)
    exact = _saliency_repeat_diagnostics(
        first_saliency=saliency,
        second_saliency=saliency.clone(),
        first_class_maps=class_maps,
        second_class_maps=class_maps.clone(),
        first_top_indices=top_indices,
        second_top_indices=top_indices.clone(),
        first_bottom_indices=bottom_indices,
        second_bottom_indices=bottom_indices.clone(),
    )
    assert exact["passed"] is True
    changed = saliency.clone()
    changed[0, 0, 0] += 1e-4
    rejected = _saliency_repeat_diagnostics(
        first_saliency=saliency,
        second_saliency=changed,
        first_class_maps=class_maps,
        second_class_maps=class_maps.clone(),
        first_top_indices=top_indices,
        second_top_indices=top_indices.clone(),
        first_bottom_indices=bottom_indices,
        second_bottom_indices=bottom_indices.clone(),
    )
    assert rejected["passed"] is False
    assert rejected["saliency_max_abs_error"] > 1e-9


def _comparison() -> dict[str, object]:
    metrics = {
        "macro_f1": 0.91,
        "per_class_precision": [0.9, 0.80, 0.9, 0.9, 0.9],
        "per_class_recall": [0.9, 0.78, 0.9, 0.9, 0.9],
        "per_class_f1": [0.9, 0.79, 0.9, 0.9, 0.9],
    }
    return {
        "candidate": metrics,
        "control": metrics,
        "delta": {
            "macro_f1": 0.004,
            "class1_precision": 0.02,
            "class1_recall": 0.0,
            "class1_f1": 0.01,
        },
        "maximum_nonfocus_f1_drop": 0.0,
        "transitions": {
            "restricted_focus_fp_reduction": 3,
            "focus_tp_break": 0,
            "focus_fn_rescue": 1,
            "candidate_correction": 5,
            "candidate_harm": 1,
            "focus_fp_remove_correct": 3,
            "focus_fp_create": 0,
        },
    }


def _passing_gate_inputs():
    controls = {
        condition: {
            "raw_candidate": _comparison(),
            "control_candidate": _comparison(),
        }
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }
    ablations = {
        condition: {"raw_candidate": _comparison()}
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }
    comparisons = {
        "controls": controls,
        "same_weight_ablation": ablations,
        "raw_context": {},
    }
    mechanism = {
        condition: {
            "direction_auroc": {
                "gap": 0.72,
                "top_only": 0.73,
                "wildcat": 0.75,
                "no_bottom": 0.73,
            },
            "bottom_margin_auroc": 0.70,
            "tp_top_alignment_above_chance": 0.15,
            "all_finite": True,
        }
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }
    resources = {
        "runtime_ratio": 1.05,
        "peak_vram_ratio": 1.05,
        "extra_peak_vram_gib": 0.05,
        "capture_raw_logit_max_abs_error": 0.0,
        "capture_raw_prediction_mismatches": 0,
        "branch_finite": True,
    }
    onnx = {
        "succeeded": True,
        "finite": True,
        "argmax_match": True,
        "maximum_absolute_error": 1e-6,
    }
    exports = {
        "isolated_onnx": onnx,
        "full_onnx": onnx,
        "tensorrt": {"available": False},
    }
    xai = {
        "selection_order_exact": True,
        "required_event_coverage_exact": True,
        "all_finite": True,
        "defined_maps_nonzero": True,
        "selected_rows": 16,
        "page_count": 2,
        "pages": [str(Path(__file__)), str(Path(__file__))],
        "npz_keys_exact": True,
        "npz_replay_finite": True,
        "npz_key_count": 80,
        "npz_replay_max_abs_error": 0.0,
        "saliency_repeat": {"passed": True},
        "saliency_backward_determinism_scope": "warn_only",
        "saliency_repeats_per_batch": 2,
        "hook_calls": 8,
        "expected_hook_calls": 8,
        "strict_determinism_before_scope": True,
        "warn_only_before_scope": False,
        "strict_determinism_state_restored": True,
    }
    return comparisons, mechanism, resources, exports, xai


def test_gate_accepts_precision_safe_signal_and_rejects_recall_loss() -> None:
    comparisons, mechanism, resources, exports, xai = _passing_gate_inputs()
    result = assess_stage_a(
        structural_checks={"structure": True},
        comparisons=comparisons,
        mechanism=mechanism,
        resources=resources,
        exports=exports,
        xai=xai,
        visual_review_passed=True,
    )
    assert result["all_gates_passed"] is True
    comparisons["controls"]["lighting_dim"]["control_candidate"]["delta"][
        "class1_recall"
    ] = -0.10
    result = assess_stage_a(
        structural_checks={"structure": True},
        comparisons=comparisons,
        mechanism=mechanism,
        resources=resources,
        exports=exports,
        xai=xai,
        visual_review_passed=True,
    )
    assert result["all_gates_passed"] is False
    assert "worst_recall_delta_vs_top_gte_minus_0p030" in result["failed_checks"]

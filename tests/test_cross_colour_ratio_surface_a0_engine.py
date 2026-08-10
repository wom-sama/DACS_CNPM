from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trkh.tools.cross_colour_ratio_surface_a0_engine import (
    CLASS_ORDER,
    COLOUR_RATIO_MODE,
    CROSS_COLOUR_RATIO_MODE,
    DESCRIPTOR_CHANNEL_NAMES,
    EPOCHS,
    INPUT_MEAN,
    INPUT_STD,
    LOG_EPSILON,
    ROLE_NAMES,
    SIGNED_RESPONSE_LIMIT,
    CrossColourRatioSurfacePipeline,
    FixedAreaResize2d,
    apply_keeper_suppression,
    array_sha256,
    build_dephase_offsets,
    build_epoch_orders,
    calibrate_class1_retention_threshold,
    class1_evidence_score,
    direct_cross_colour_log_ratio,
    gaussian_derivative_kernels,
    initialize_role_head,
    keeper_margin,
    learning_rate,
    log_difference_cross_colour_ratio,
    map_targets_to_head_indices,
    margin_crop_box,
    model_state_arrays,
    parameter_contract,
    role_seed,
    signed_colour_derivatives,
    spatially_dephase,
    state_arrays_sha256,
)


REPOSITORY = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    REPOSITORY
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_LOCK_20260725.json"
)
GEOMETRY_PATH = (
    REPOSITORY
    / "runs"
    / "audit_attention_maxsep_prototype_a0_20260721"
    / "cohort_geometry.npz"
)


def _lock() -> dict[str, object]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _numpy_derivative_oracle(
    linear_rgb: np.ndarray,
    reliability: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    logs = np.log(np.maximum(linear_rgb, LOG_EPSILON))
    if mode == COLOUR_RATIO_MODE:
        planes = logs
    elif mode == CROSS_COLOUR_RATIO_MODE:
        planes = np.stack(
            (
                logs[:, 0] - logs[:, 1],
                logs[:, 0] - logs[:, 2],
                logs[:, 1] - logs[:, 2],
            ),
            axis=1,
        )
    else:
        raise ValueError(mode)

    kernels = gaussian_derivative_kernels()[:, 0]
    responses: list[np.ndarray] = []
    for plane_index in range(3):
        padded = np.pad(
            planes[:, plane_index],
            ((0, 0), (3, 3), (3, 3)),
            mode="constant",
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (7, 7),
            axis=(1, 2),
        )
        for axis_index in range(2):
            kernel = kernels[2 * plane_index + axis_index]
            responses.append(
                np.einsum("bhwij,ij->bhw", windows, kernel, optimize=True)
            )
    raw = np.stack(responses, axis=1)
    clipped = (~np.isfinite(raw)) | (np.abs(raw) > SIGNED_RESPONSE_LIMIT)
    bounded = np.nan_to_num(
        raw,
        nan=0.0,
        posinf=SIGNED_RESPONSE_LIMIT,
        neginf=-SIGNED_RESPONSE_LIMIT,
    )
    bounded = np.clip(
        bounded,
        -SIGNED_RESPONSE_LIMIT,
        SIGNED_RESPONSE_LIMIT,
    )
    descriptors = bounded * reliability
    denominator = np.maximum(
        reliability.sum(axis=(1, 2, 3)) * 6.0,
        1.0,
    )
    clip_fraction = (
        clipped * np.broadcast_to(reliability, clipped.shape)
    ).sum(axis=(1, 2, 3)) / denominator
    return descriptors, clip_fraction


def test_equations_match_direct_ratio_and_independent_numpy_oracle() -> None:
    generator = torch.Generator().manual_seed(20260725)
    first = (
        torch.rand(37, 3, generator=generator, dtype=torch.float64) * 0.8
        + 0.1
    )
    second = (
        torch.rand(37, 3, generator=generator, dtype=torch.float64) * 0.8
        + 0.1
    )
    direct = direct_cross_colour_log_ratio(first, second)
    difference = log_difference_cross_colour_ratio(first, second)
    torch.testing.assert_close(direct, difference, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        difference,
        -log_difference_cross_colour_ratio(second, first),
        atol=1e-12,
        rtol=0.0,
    )

    linear_rgb = (
        torch.rand(
            2,
            3,
            13,
            15,
            generator=generator,
            dtype=torch.float64,
        )
        * 0.8
        + 0.1
    )
    reliability = torch.ones(2, 1, 13, 15, dtype=torch.float64)
    reliability[0, :, :2, :] = 0.0
    for mode in (COLOUR_RATIO_MODE, CROSS_COLOUR_RATIO_MODE):
        actual, actual_clip = signed_colour_derivatives(
            linear_rgb,
            reliability,
            mode=mode,
        )
        expected, expected_clip = _numpy_derivative_oracle(
            linear_rgb.numpy(),
            reliability.numpy(),
            mode=mode,
        )
        np.testing.assert_allclose(
            actual.numpy(),
            expected,
            rtol=0.0,
            atol=2e-15,
        )
        np.testing.assert_array_equal(actual_clip.numpy(), expected_clip)


def test_cross_colour_derivatives_have_the_locked_physical_invariances() -> None:
    generator = torch.Generator().manual_seed(71)
    base = (
        torch.rand(
            1,
            3,
            31,
            35,
            generator=generator,
            dtype=torch.float64,
        )
        * 0.18
        + 0.18
    )
    y_axis = torch.linspace(0.82, 1.08, 31, dtype=torch.float64)
    x_axis = torch.linspace(0.88, 1.06, 35, dtype=torch.float64)
    common_shading = (y_axis[:, None] * x_axis[None, :])[None, None]
    channel_gain = torch.tensor(
        [0.82, 1.0, 1.13],
        dtype=torch.float64,
    ).view(1, 3, 1, 1)
    transformed = base * common_shading * channel_gain
    reliability = torch.zeros(1, 1, 31, 35, dtype=torch.float64)
    reliability[:, :, 3:-3, 3:-3] = 1.0

    candidate_base, _ = signed_colour_derivatives(
        base,
        reliability,
        mode=CROSS_COLOUR_RATIO_MODE,
    )
    candidate_transformed, _ = signed_colour_derivatives(
        transformed,
        reliability,
        mode=CROSS_COLOUR_RATIO_MODE,
    )
    torch.testing.assert_close(
        candidate_base,
        candidate_transformed,
        atol=2e-15,
        rtol=0.0,
    )

    control_base, _ = signed_colour_derivatives(
        base,
        reliability,
        mode=COLOUR_RATIO_MODE,
    )
    control_transformed, _ = signed_colour_derivatives(
        transformed,
        reliability,
        mode=COLOUR_RATIO_MODE,
    )
    assert float((control_base - control_transformed).abs().mean()) > 1e-4


def test_fixed_area_resize_matches_pytorch_area_interpolation() -> None:
    generator = torch.Generator().manual_seed(109)
    inputs = torch.randn(2, 3, 256, 256, generator=generator)
    actual = FixedAreaResize2d(256, 96)(inputs)
    expected = F.interpolate(inputs, size=(96, 96), mode="area")
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=0.0)


def test_locked_dephase_offsets_and_epoch_orders_reproduce_the_lock() -> None:
    lock = _lock()
    with np.load(GEOMETRY_PATH, allow_pickle=False) as archive:
        sample_indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        folds = np.asarray(archive["folds"], dtype=np.int64)

    offsets = build_dephase_offsets(sample_indices)
    assert list(offsets.shape) == lock["descriptor"]["dephase_offsets_shape"]
    assert (
        array_sha256(offsets)
        == lock["descriptor"]["dephase_offsets_sha256"]
    )
    assert int(offsets.min()) == lock["descriptor"]["dephase_minimum_offset"]
    assert int(offsets.max()) == lock["descriptor"]["dephase_maximum_offset"]

    image = torch.arange(96 * 96, dtype=torch.float32).reshape(1, 1, 96, 96)
    descriptors = image.repeat(2, 6, 1, 1)
    dephased = spatially_dephase(descriptors, offsets[:2])
    for row in range(2):
        for channel in range(6):
            torch.testing.assert_close(
                torch.sort(dephased[row, channel].flatten()).values,
                torch.sort(descriptors[row, channel].flatten()).values,
                atol=0.0,
                rtol=0.0,
            )
    assert not torch.equal(dephased[:, 0], dephased[:, 1])

    for fold_record in lock["folds"]:
        outer_fold = int(fold_record["outer_fold"])
        calibration_fold = int(fold_record["calibration_fold"])
        fit_indices = np.flatnonzero(
            (folds != outer_fold) & (folds != calibration_fold)
        ).astype(np.int64)
        for role, order_key in (
            ("cross_colour_ratio_candidate", "primary_orders"),
            ("cross_colour_ratio_seed_repeat", "repeat_orders"),
        ):
            orders = build_epoch_orders(
                fit_indices,
                seed=role_seed(role, outer_fold),
            )
            assert len(orders) == EPOCHS
            assert [
                array_sha256(order) for order in orders
            ] == fold_record[order_key]["per_epoch_sha256"]
            assert array_sha256(np.concatenate(orders)) == fold_record[
                order_key
            ]["all_epochs_sha256"]


def test_roles_are_scratch_initialized_with_one_locked_parameter_budget() -> None:
    device = torch.device("cpu")
    contracts: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for role in ROLE_NAMES:
        model = initialize_role_head(role, fold=3, device=device)
        contracts[role] = parameter_contract(model)
        hashes[role] = state_arrays_sha256(model_state_arrays(model))
    assert set(DESCRIPTOR_CHANNEL_NAMES) == {
        "rg_x",
        "rg_y",
        "rb_x",
        "rb_y",
        "gb_x",
        "gb_y",
    }
    assert all(contract == contracts[ROLE_NAMES[0]] for contract in contracts.values())
    assert contracts[ROLE_NAMES[0]]["trainable_parameter_count"] == 3004
    assert (
        hashes["colour_ratio_control"]
        == hashes["cross_colour_ratio_candidate"]
        == hashes["cross_colour_ratio_spatial_dephased_control"]
    )
    assert (
        hashes["cross_colour_ratio_seed_repeat"]
        != hashes["cross_colour_ratio_candidate"]
    )

    model = initialize_role_head(
        "cross_colour_ratio_candidate",
        fold=3,
        device=device,
    )
    initial_hash = state_arrays_sha256(model_state_arrays(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    descriptors = torch.randn(4, 6, 96, 96)
    reliability = torch.ones(4, 1, 96, 96)
    logits, _, _, _ = model(descriptors, reliability)
    loss = F.cross_entropy(
        logits,
        map_targets_to_head_indices(torch.tensor(CLASS_ORDER)),
    )
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert (
        state_arrays_sha256(model_state_arrays(model))
        != initial_hash
    )


def test_schedule_scores_threshold_ties_and_suppression_are_locked() -> None:
    assert math.isclose(learning_rate(0), 5e-4)
    assert math.isclose(learning_rate(1), 1e-3)
    assert math.isclose(learning_rate(2), 1e-3)
    assert math.isclose(learning_rate(EPOCHS - 1), 0.0, abs_tol=1e-15)

    logits = np.asarray(
        [
            [0.0, 2.0, 1.0, -1.0],
            [2.0, 0.5, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    numpy_scores = class1_evidence_score(logits)
    torch_scores = class1_evidence_score(torch.from_numpy(logits))
    np.testing.assert_allclose(
        numpy_scores,
        torch_scores.numpy(),
        rtol=0.0,
        atol=2e-16,
    )

    threshold = calibrate_class1_retention_threshold(
        np.asarray([0.1, 0.2, 0.2, 0.3, 0.9], dtype=np.float64),
        np.ones(5, dtype=np.bool_),
        minimum_retention=0.6,
    )
    assert threshold["threshold"] == 0.2
    assert threshold["allowed_class1_breaks"] == 2
    assert threshold["observed_class1_breaks"] == 1
    assert threshold["class1_retention"] == 0.8

    probabilities = np.asarray(
        [
            [0.10, 0.50, 0.10, 0.29, 0.01],
            [0.10, 0.50, 0.10, 0.29, 0.01],
            [0.60, 0.20, 0.10, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    actions = apply_keeper_suppression(
        probabilities,
        np.asarray([0.1, 0.2, 0.1], dtype=np.float64),
        threshold=0.2,
    )
    assert actions["suppressed"].tolist() == [True, False, False]
    assert actions["predictions"].tolist() == [3, 1, 0]

    high_class3 = np.asarray(
        [[0.10, 0.40, 0.08, 0.39, 0.03]],
        dtype=np.float64,
    )
    expected_margin = math.log(0.40) - math.log(0.10)
    assert math.isclose(keeper_margin(high_class3)[0], expected_margin)


def test_margin_crop_matches_the_frozen_production_geometry() -> None:
    assert margin_crop_box(100, 80, (0.5, 0.5, 0.4, 0.5)) == (
        28,
        18,
        72,
        62,
    )
    assert margin_crop_box(100, 80, (0.05, 0.05, 0.1, 0.1)) == (
        0,
        0,
        11,
        9,
    )


def test_full_pipeline_has_standard_onnx_and_dynamic_batch_replay(
    tmp_path: Path,
) -> None:
    import onnx
    import onnxruntime as ort

    torch.manual_seed(20260725)
    model = CrossColourRatioSurfacePipeline().eval()
    srgb = torch.rand(1, 3, 256, 256) * 0.9 + 0.05
    mean = torch.tensor(INPUT_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(INPUT_STD).view(1, 3, 1, 1)
    model_input = (srgb - mean) / std
    valid_mask = torch.ones(1, 1, 256, 256)
    valid_mask[:, :, :17, :] = 0.0
    with torch.inference_mode():
        expected = model(model_input, valid_mask)

    path = tmp_path / "cross_colour_ratio_surface_a0.onnx"
    torch.onnx.export(
        model,
        (model_input, valid_mask),
        str(path),
        opset_version=17,
        input_names=("model_input", "image_valid_mask"),
        output_names=("probabilities", "pair_maps"),
        dynamic_axes={
            "model_input": {0: "batch"},
            "image_valid_mask": {0: "batch"},
            "probabilities": {0: "batch"},
            "pair_maps": {0: "batch"},
        },
        do_constant_folding=True,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert {node.domain for node in graph.graph.node} <= {"", "ai.onnx"}

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        None,
        {
            "model_input": model_input.numpy(),
            "image_valid_mask": valid_mask.numpy(),
        },
    )
    assert float(np.max(np.abs(actual[0] - expected[0].numpy()))) <= 1e-5
    assert float(np.max(np.abs(actual[1] - expected[1].numpy()))) <= 1e-5

    batch2_outputs = session.run(
        None,
        {
            "model_input": model_input.repeat(2, 1, 1, 1).numpy(),
            "image_valid_mask": valid_mask.repeat(2, 1, 1, 1).numpy(),
        },
    )
    assert batch2_outputs[0].shape == (2, 4)
    assert batch2_outputs[1].shape == (2, 3, 24, 24)

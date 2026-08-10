from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.pair_surface_ddf_v2_engine import (
    BATCH_SIZE,
    CAUSAL_CHANNEL_NEUTRAL,
    CAUSAL_SPATIAL_NEUTRAL,
    EPOCHS,
    ROLE_NAMES,
    MaskedBatchNorm2d,
    MaskedPopulationAccumulator,
    PairSurfaceDDFV2Sidecar,
    PreparedSurfaceInputs,
    apply_ddf_standard,
    array_sha256,
    build_adamw,
    build_epoch_orders,
    build_task_supervision,
    displace_spatial_filters_nonwrap,
    filter_normalize,
    initialize_sidecar,
    input_normalization_contract,
    learning_rate,
    literal_zero_mask,
    masked_spatial_softmax,
    neutralize_spatial_outside_support,
    numpy_ddf,
    numpy_filter_normalize,
    pair_surface_loss,
    parameter_contract,
    prepare_cached_model_inputs,
    rasterize_model_boxes_16,
    recalibrate_masked_batch_norms,
    role_initialization_seed,
    role_order_seed,
    tensor_initialization_seed,
    unpack_valid_masks_little,
    validity_weighted_channel_pool,
)


def _valid_pyramid(
    batch: int,
    *,
    fully_valid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid64 = torch.zeros(batch, 1, 64, 64, dtype=torch.bool)
    if fully_valid:
        valid64[:] = True
    else:
        valid64[:, :, 3:59, 5:61] = True
        valid64[:, :, 21:26, 28:33] = False
    valid32 = torch.nn.functional.max_pool2d(
        valid64.float(),
        kernel_size=3,
        stride=2,
        padding=1,
    ) > 0
    valid16 = torch.nn.functional.max_pool2d(
        valid32.float(),
        kernel_size=3,
        stride=2,
        padding=1,
    ) > 0
    return valid64, valid32, valid16


def _numpy_masked_moments(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = mask[:, 0].astype(bool)
    channel_values = [
        values[:, channel][selected]
        for channel in range(values.shape[1])
    ]
    mean = np.asarray([item.mean(dtype=np.float64) for item in channel_values])
    variance = np.asarray(
        [item.var(dtype=np.float64, ddof=0) for item in channel_values]
    )
    return mean, variance


def test_valid_cache_decoder_is_mandatorily_little_endian() -> None:
    packed = np.zeros((1, 8192), dtype=np.uint8)
    packed[0, 0] = np.uint8(0b10000001)
    unpacked = unpack_valid_masks_little(packed)

    assert unpacked.shape == (1, 256, 256)
    assert unpacked.dtype == np.bool_
    assert bool(unpacked[0, 0, 0])
    assert bool(unpacked[0, 0, 7])
    assert not bool(unpacked[0, 0, 1])
    with pytest.raises(ValueError, match="8192"):
        unpack_valid_masks_little(np.zeros((1, 8), dtype=np.uint8))


def test_validity_aware_input_prepare_has_exact_invalid_fill_invariance() -> None:
    generator = torch.Generator().manual_seed(2026072901)
    images = torch.randint(
        0,
        256,
        (2, 3, 256, 256),
        dtype=torch.uint8,
        generator=generator,
    )
    valid = torch.zeros(2, 1, 256, 256, dtype=torch.bool)
    valid[:, :, 7:247, 11:241] = True
    valid[:, :, 80:101, 120:147] = False
    changed_fill = images.clone()
    noise = torch.randint(
        0,
        256,
        images.shape,
        dtype=torch.uint8,
        generator=generator,
    )
    changed_fill[~valid.expand_as(images)] = noise[~valid.expand_as(images)]

    first = prepare_cached_model_inputs(images, valid)
    second = prepare_cached_model_inputs(changed_fill, valid)

    assert torch.equal(first.values, second.values)
    assert torch.equal(first.valid64, second.valid64)
    assert torch.equal(first.valid32, second.valid32)
    assert torch.equal(first.valid16, second.valid16)
    assert torch.equal(
        first.values[~first.valid64.expand_as(first.values)],
        torch.zeros_like(first.values[~first.valid64.expand_as(first.values)]),
    )
    assert not torch.signbit(
        first.values[~first.valid64.expand_as(first.values)]
    ).any()

    row, output_y, output_x = 0, 3, 4
    y0, x0 = output_y * 4, output_x * 4
    block_valid = valid[row, 0, y0 : y0 + 4, x0 : x0 + 4]
    raw = images[row, :, y0 : y0 + 4, x0 : x0 + 4].float() / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    expected = ((raw - mean) / std)[:, block_valid].mean(dim=1)
    assert torch.allclose(
        first.values[row, :, output_y, output_x],
        expected,
        atol=1e-7,
        rtol=0.0,
    )
    model = initialize_sidecar("ddf_full", fold=0).eval()
    first_trace = model.forward_with_trace(
        first.values,
        first.valid64,
        first.valid32,
        first.valid16,
    )
    second_trace = model.forward_with_trace(
        second.values,
        second.valid64,
        second.valid32,
        second.valid16,
    )
    for name in ("scores", "features", "evidence_maps", "attention_maps"):
        assert torch.equal(first_trace[name], second_trace[name])


def test_seed_endianness_fp32_constants_and_global_rng_are_pinned() -> None:
    expected_seeds = {
        (20260729, "conv1.weight"): 8061816417627997939,
        (20260729, "bn1_pre.weight"): 7335466851678255501,
        (20261129, "block2.channel_scale"): 5347873788593879758,
        (20360729, "attention.bias"): 9046612744460380978,
    }
    for (base_seed, name), expected in expected_seeds.items():
        assert tensor_initialization_seed(base_seed, name) == expected
        assert 0 <= expected < (1 << 63)

    mean, std = input_normalization_contract()
    assert mean.view(np.uint32).tolist() == [0x3EF851EC, 0x3EE978D5, 0x3ECFDF3B]
    assert std.view(np.uint32).tolist() == [0x3E6A7EFA, 0x3E656042, 0x3E666666]
    assert array_sha256(mean) == "64adba3379585c990c33ac8a768652797811c460965970493bacbc150b4709e3"
    assert array_sha256(std) == "c917c2b282ce5e9dafe8456bfca484a079a72b4eff1a4587f571bc0e2f4f2bfa"

    torch.manual_seed(9173)
    state_before = torch.random.get_rng_state().clone()
    first = initialize_sidecar("ddf_full", fold=2)
    state_after = torch.random.get_rng_state().clone()
    second = initialize_sidecar("ddf_full", fold=2)
    assert torch.equal(state_before, state_after)
    assert torch.equal(state_after, torch.random.get_rng_state())
    assert all(
        torch.equal(first.state_dict()[name], value)
        for name, value in second.state_dict().items()
    )


def test_masked_pool_and_softmax_match_independent_numpy_oracles() -> None:
    rng = np.random.default_rng(2026072902)
    values = rng.normal(size=(2, 3, 4, 5))
    mask = rng.uniform(size=(2, 1, 4, 5)) > 0.35
    pooled = validity_weighted_channel_pool(
        torch.from_numpy(values),
        torch.from_numpy(mask),
    ).numpy()
    expected_pool = np.stack(
        [
            np.asarray(
                [values[row, channel][mask[row, 0]].mean() for channel in range(3)]
            )
            for row in range(2)
        ]
    ).reshape(2, 3, 1, 1)
    assert np.max(np.abs(pooled - expected_pool)) <= 2e-16

    logits = rng.normal(size=(2, 4, 4, 5))
    observed = masked_spatial_softmax(
        torch.from_numpy(logits),
        torch.from_numpy(mask),
    ).numpy()
    expected = np.zeros_like(logits)
    for row in range(2):
        selected = mask[row, 0]
        for task in range(4):
            shifted = logits[row, task, selected]
            weights = np.exp(shifted - shifted.max())
            expected[row, task, selected] = weights / weights.sum()
    assert np.max(np.abs(observed - expected)) <= 2e-16
    invalid_values = observed[~np.broadcast_to(mask, observed.shape)]
    assert np.all(invalid_values == 0.0)
    assert not np.signbit(invalid_values).any()
    assert np.max(np.abs(observed.sum(axis=(2, 3)) - 1.0)) <= 5e-16


def test_masked_batch_norm_train_eval_and_buffers_match_numpy() -> None:
    rng = np.random.default_rng(2026072903)
    values = rng.normal(size=(3, 2, 4, 5))
    mask = rng.uniform(size=(3, 1, 4, 5)) > 0.4
    values[~np.broadcast_to(mask, values.shape)] = 1e6
    module = MaskedBatchNorm2d(2).double()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.4, -0.7], dtype=torch.float64))
        module.bias.copy_(torch.tensor([0.3, 0.2], dtype=torch.float64))
        module.running_mean.copy_(torch.tensor([9.0, 8.0], dtype=torch.float64))
        module.running_var.copy_(torch.tensor([7.0, 6.0], dtype=torch.float64))

    running_mean_before = module.running_mean.clone()
    running_var_before = module.running_var.clone()
    module.train()
    observed_train = module(
        torch.from_numpy(values),
        torch.from_numpy(mask),
    ).detach().numpy()
    mean, variance = _numpy_masked_moments(values, mask)
    expected_train = (
        (values - mean.reshape(1, 2, 1, 1))
        / np.sqrt(variance.reshape(1, 2, 1, 1) + 1e-5)
        * np.asarray([1.4, -0.7]).reshape(1, 2, 1, 1)
        + np.asarray([0.3, 0.2]).reshape(1, 2, 1, 1)
    )
    expected_train = np.where(mask, expected_train, 0.0)
    assert np.max(np.abs(observed_train - expected_train)) <= 9e-16
    assert torch.equal(module.running_mean, running_mean_before)
    assert torch.equal(module.running_var, running_var_before)
    assert int(module.num_batches_tracked) == 1

    module.eval()
    observed_eval = module(
        torch.from_numpy(values),
        torch.from_numpy(mask),
    ).detach().numpy()
    expected_eval = (
        (values - np.asarray([9.0, 8.0]).reshape(1, 2, 1, 1))
        / np.sqrt(np.asarray([7.0, 6.0]).reshape(1, 2, 1, 1) + 1e-5)
        * np.asarray([1.4, -0.7]).reshape(1, 2, 1, 1)
        + np.asarray([0.3, 0.2]).reshape(1, 2, 1, 1)
    )
    expected_eval = np.where(mask, expected_eval, 0.0)
    assert np.max(np.abs(observed_eval - expected_eval)) <= 2e-15
    assert int(module.num_batches_tracked) == 1


def test_masked_population_accumulator_is_fp64_and_batch_size_invariant() -> None:
    rng = np.random.default_rng(2026072904)
    values = rng.normal(size=(7, 3, 4, 5)).astype(np.float32)
    mask = rng.uniform(size=(7, 1, 4, 5)) > 0.25
    accumulator = MaskedPopulationAccumulator(3)
    for start, stop in ((0, 2), (2, 6), (6, 7)):
        accumulator.update(
            torch.from_numpy(values[start:stop]),
            torch.from_numpy(mask[start:stop]),
        )
    mean, variance = accumulator.finalize()
    expected_mean, expected_variance = _numpy_masked_moments(values, mask)
    assert mean.dtype == torch.float64
    assert variance.dtype == torch.float64
    assert np.max(np.abs(mean.numpy() - expected_mean)) <= 2e-16
    assert np.max(np.abs(variance.numpy() - expected_variance)) <= 4e-16


def test_filter_normalization_and_ddf_match_fp64_numpy_oracles() -> None:
    rng = np.random.default_rng(2026072905)
    spatial = rng.normal(size=(2, 9, 5, 4))
    channel = rng.normal(size=(2, 3, 9))
    scale = rng.normal(size=(1, 3, 9))
    observed_spatial = filter_normalize(
        torch.from_numpy(spatial),
        tap_dimension=1,
        scale=math.sqrt(2.0) / 3.0,
    ).numpy()
    observed_channel = filter_normalize(
        torch.from_numpy(channel),
        tap_dimension=2,
        scale=torch.from_numpy(scale),
    ).numpy()
    expected_spatial = numpy_filter_normalize(
        spatial,
        tap_axis=1,
        scale=math.sqrt(2.0) / 3.0,
    )
    expected_channel = numpy_filter_normalize(
        channel,
        tap_axis=2,
        scale=scale,
    )
    assert np.max(np.abs(observed_spatial - expected_spatial)) <= 5e-16
    assert np.max(np.abs(observed_channel - expected_channel)) <= 1e-15

    inputs = rng.normal(size=(2, 3, 5, 4))
    observed_ddf = apply_ddf_standard(
        torch.from_numpy(inputs),
        torch.from_numpy(channel),
        torch.from_numpy(spatial),
    ).numpy()
    expected_ddf = numpy_ddf(inputs, channel, spatial)
    assert np.max(np.abs(observed_ddf - expected_ddf)) <= 4e-15


def test_all_roles_preserve_parameter_oracles_and_mask_every_output() -> None:
    inputs = torch.randn(2, 3, 64, 64)
    valid64, valid32, valid16 = _valid_pyramid(2)
    expected_counts = {
        "ddf_full": 9380,
        "static_matched": 9435,
        "ddf_spatial_only": 9380,
        "ddf_channel_only": 9380,
        "ddf_full_repeat": 9380,
    }
    for role in ROLE_NAMES:
        model = initialize_sidecar(role, fold=0).eval()
        trace = model.forward_with_trace(inputs, valid64, valid32, valid16)
        assert parameter_contract(model)["parameter_count"] == expected_counts[role]
        assert trace["scores"].shape == (2, 4)
        assert trace["evidence_maps"].shape == (2, 4, 16, 16)
        assert trace["attention_maps"].shape == (2, 4, 16, 16)
        assert torch.isfinite(trace["scores"]).all()
        invalid = ~valid16.expand(2, 4, 16, 16)
        assert torch.equal(trace["evidence_maps"][invalid], torch.zeros_like(trace["evidence_maps"][invalid]))
        assert torch.equal(trace["attention_maps"][invalid], torch.zeros_like(trace["attention_maps"][invalid]))
        assert not torch.signbit(trace["evidence_maps"][invalid]).any()
        assert not torch.signbit(trace["attention_maps"][invalid]).any()
        feature_invalid = ~valid16.expand(2, 32, 16, 16)
        assert torch.equal(
            trace["features"][feature_invalid],
            torch.zeros_like(trace["features"][feature_invalid]),
        )
        assert not torch.signbit(trace["features"][feature_invalid]).any()
        assert torch.allclose(
            trace["attention_maps"].sum(dim=(2, 3)),
            torch.ones(2, 4),
            atol=1e-6,
            rtol=0.0,
        )
        if role != "static_matched":
            spatial_filters = trace["spatial_filters"]
            assert isinstance(spatial_filters, list)
            for spatial_filter, valid in zip(spatial_filters, (valid32, valid16)):
                outside = ~valid.expand_as(spatial_filter)
                assert torch.equal(
                    spatial_filter[outside],
                    torch.ones_like(spatial_filter[outside]),
                )


def test_primary_roles_share_matching_tensors_and_repeat_seed_differs() -> None:
    primary_roles = (
        "ddf_full",
        "static_matched",
        "ddf_spatial_only",
        "ddf_channel_only",
    )
    states = {
        role: initialize_sidecar(role, fold=3).state_dict()
        for role in primary_roles
    }
    shared_names = set.intersection(*(set(state) for state in states.values()))
    shared_names = {
        name
        for name in shared_names
        if all(
            states[role][name].shape == states["ddf_full"][name].shape
            for role in primary_roles
        )
    }
    assert shared_names
    for name in shared_names:
        reference = states["ddf_full"][name]
        assert all(
            torch.equal(reference, states[role][name])
            for role in primary_roles[1:]
        )
    repeat = initialize_sidecar("ddf_full_repeat", fold=3).state_dict()
    assert any(
        not torch.equal(states["ddf_full"][name], repeat[name])
        for name in states["ddf_full"]
        if states["ddf_full"][name].is_floating_point()
    )
    assert role_order_seed("ddf_full", 3) == role_initialization_seed(
        "ddf_full",
        3,
    ) + 1


def test_batch_shapes_are_finite_and_full_role_groups_update() -> None:
    for batch in (1, 2, 32):
        valid64, valid32, valid16 = _valid_pyramid(batch)
        model = initialize_sidecar("ddf_full", fold=1).eval()
        scores = model(
            torch.randn(batch, 3, 64, 64),
            valid64,
            valid32,
            valid16,
        )
        assert scores.shape == (batch, 4)
        assert torch.isfinite(scores).all()

    model = initialize_sidecar("ddf_full", fold=2).train()
    batch = 8
    valid64, valid32, valid16 = _valid_pyramid(batch, fully_valid=True)
    targets = torch.tensor([1, 0, 1, 2, 1, 4, 1, 0], dtype=torch.long)
    bbox = torch.zeros(batch, 16, 16, dtype=torch.bool)
    bbox[:, 3:13, 2:14] = True
    supervision = build_task_supervision(targets, bbox.flatten(1).any(1))
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    running_before = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if "running_" in name
    }
    optimizer = build_adamw(model, lr=learning_rate(0))
    optimizer.zero_grad(set_to_none=True)
    trace = model.forward_with_trace(
        torch.randn(batch, 3, 64, 64),
        valid64,
        valid32,
        valid16,
    )
    losses = pair_surface_loss(
        trace["scores"],
        trace["attention_maps"],
        supervision,
        bbox,
        valid16,
    )
    losses["total"].backward()
    groups = {
        "stem": ("conv1.", "conv2."),
        "normalization": ("bn1_", "bn2_"),
        "spatial": ("block1.spatial_projection", "block2.spatial_projection"),
        "channel": ("block1.channel_", "block2.channel_"),
        "heads": ("evidence.", "attention."),
    }
    named = dict(model.named_parameters())
    for prefixes in groups.values():
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if name.startswith(prefixes) and parameter.requires_grad
        ]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0001)
    assert optimizer.param_groups[0]["params"] == [
        parameter for _, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert all(
        torch.equal(running_before[name], value)
        for name, value in model.named_buffers()
        if name in running_before
    )
    assert all(
        int(module.num_batches_tracked) == 1
        for module in model.masked_batch_norms()
    )
    optimizer.step()
    for prefixes in groups.values():
        changed = [
            not torch.equal(before[name], parameter.detach())
            for name, parameter in named.items()
            if name.startswith(prefixes) and parameter.requires_grad
        ]
        assert changed and any(changed)


def test_explicit_task_activity_and_fixed_denominator_loss() -> None:
    targets = torch.tensor([1, 1, 1, 0, 0, 2, 4], dtype=torch.long)
    usable = torch.ones(7, dtype=torch.bool)
    supervision = build_task_supervision(targets, usable)
    rows = float(targets.numel())
    for task in range(4):
        positive = supervision.labels[:, task] == 1
        negative = (
            supervision.labels[:, task] == 0
        ) & supervision.active[:, task]
        assert float(supervision.classification_weights[positive, task].sum()) == pytest.approx(rows / 2)
        assert float(supervision.classification_weights[negative, task].sum()) == pytest.approx(rows / 2)
        assert torch.equal(
            supervision.classification_weights[~supervision.active[:, task], task],
            torch.zeros_like(
                supervision.classification_weights[~supervision.active[:, task], task]
            ),
        )
    assert supervision.active[targets == 1].all()
    assert not supervision.active[targets == 0, 2:].any()
    assert not supervision.active[targets == 2, (1, 3)].any()
    assert not supervision.active[targets == 4, 1:3].any()

    scores = torch.zeros(7, 4, requires_grad=True)
    attention = torch.full((7, 4, 16, 16), 1.0 / 256.0)
    bbox = torch.ones(7, 16, 16, dtype=torch.bool)
    valid16 = torch.ones(7, 1, 16, 16, dtype=torch.bool)
    losses = pair_surface_loss(
        scores,
        attention,
        supervision,
        bbox,
        valid16,
    )
    expected_task = rows * math.log(2.0) / float(BATCH_SIZE)
    assert torch.allclose(
        losses["classification_by_task"],
        torch.full((4,), expected_task),
        atol=1e-7,
        rtol=0.0,
    )
    assert float(losses["bbox_total"]) == pytest.approx(0.0, abs=1e-12)
    assert float(losses["total"]) == pytest.approx(
        1.5 * expected_task,
        abs=1e-7,
    )
    losses["total"].backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    partly_usable = torch.tensor([False, True, True, True, True, True, True])
    adjusted = build_task_supervision(targets, partly_usable)
    assert torch.equal(adjusted.bbox_weights[0], torch.zeros(4))


def test_loss_slices_fit_frozen_weights_without_short_batch_rescaling() -> None:
    targets = torch.tensor([1, 0, 1, 2, 1, 4, 1, 0], dtype=torch.long)
    full = build_task_supervision(targets, torch.ones(8, dtype=torch.bool))
    indices = torch.tensor([0, 1, 3], dtype=torch.long)
    batch = full.take(indices)
    scores = torch.zeros(3, 4)
    attention = torch.full((3, 4, 16, 16), 1.0 / 256.0)
    bbox = torch.ones(3, 16, 16, dtype=torch.bool)
    valid16 = torch.ones(3, 1, 16, 16, dtype=torch.bool)
    losses = pair_surface_loss(scores, attention, batch, bbox, valid16)
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            scores,
            batch.labels,
            reduction="none",
        )
        * batch.classification_weights
    ).sum(dim=0) / 64.0
    assert torch.equal(losses["classification_by_task"], expected)


def test_nonzero_bbox_loss_matches_independent_fixed_weight_oracle() -> None:
    targets = torch.tensor([1, 1, 1, 0, 0, 2, 4, 1], dtype=torch.long)
    bbox_usable = torch.tensor(
        [False, True, True, True, True, True, True, True],
        dtype=torch.bool,
    )
    full = build_task_supervision(targets, bbox_usable)
    assert full.classification_weights[0].gt(0).all()
    assert torch.equal(full.bbox_weights[0], torch.zeros(4))
    assert torch.equal(
        full.classification_weights[~full.active],
        torch.zeros_like(full.classification_weights[~full.active]),
    )
    assert torch.equal(
        full.bbox_weights[~full.active],
        torch.zeros_like(full.bbox_weights[~full.active]),
    )

    masses = torch.tensor(
        [
            [0.15, 0.20, 0.25, 0.30],
            [0.31, 0.37, 0.41, 0.43],
            [0.47, 0.49, 0.53, 0.59],
            [0.23, 0.29, 0.31, 0.37],
            [0.19, 0.27, 0.33, 0.39],
            [0.17, 0.21, 0.45, 0.35],
            [0.13, 0.18, 0.22, 0.55],
            [0.61, 0.63, 0.67, 0.71],
        ],
        dtype=torch.float64,
    )
    attention = torch.empty(8, 4, 16, 16, dtype=torch.float64)
    for row in range(8):
        for task in range(4):
            attention[row, task].fill_((1.0 - masses[row, task]) / 255.0)
            attention[row, task, 0, 0] = masses[row, task]
    bbox = torch.zeros(8, 16, 16, dtype=torch.bool)
    bbox[bbox_usable, 0, 0] = True
    valid16 = torch.ones(8, 1, 16, 16, dtype=torch.bool)
    scores = torch.zeros(8, 4, dtype=torch.float64)
    losses = pair_surface_loss(scores, attention, full, bbox, valid16)
    expected_bbox = (
        -torch.log(masses.clamp_min(1e-8))
        * full.bbox_weights.to(dtype=torch.float64)
    ).sum(dim=0) / 64.0
    assert torch.allclose(
        losses["bbox_by_task"],
        expected_bbox,
        atol=2e-16,
        rtol=0.0,
    )
    assert float(losses["bbox_total"]) > 0.0

    short_indices = torch.tensor([0, 1, 3], dtype=torch.long)
    short = full.take(short_indices)
    short_losses = pair_surface_loss(
        scores.index_select(0, short_indices),
        attention.index_select(0, short_indices),
        short,
        bbox.index_select(0, short_indices),
        valid16.index_select(0, short_indices),
    )
    short_expected = (
        -torch.log(masses.index_select(0, short_indices).clamp_min(1e-8))
        * short.bbox_weights.to(dtype=torch.float64)
    ).sum(dim=0) / 64.0
    assert torch.allclose(
        short_losses["bbox_by_task"],
        short_expected,
        atol=2e-16,
        rtol=0.0,
    )


def test_bbox_rasterization_and_usability_are_explicit() -> None:
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.25],
            [0.0, 0.0, 0.0, 0.0],
            [1.2, 1.1, 0.2, 0.4],
        ],
        dtype=torch.float64,
    )
    masks = rasterize_model_boxes_16(boxes)
    assert masks.shape == (3, 16, 16)
    assert masks[0, 6:10, 4:12].all()
    assert int(masks[0].sum()) == 32
    assert int(masks[1].sum()) == 1
    assert int(masks[2].sum()) >= 1


def test_lr_orders_and_optimizer_contract_are_v2_specific() -> None:
    assert learning_rate(0) == pytest.approx(0.003)
    assert learning_rate(19) > 0.0
    assert learning_rate(20) == pytest.approx(0.0, abs=1e-18)
    assert all(learning_rate(index) > learning_rate(index + 1) for index in range(20))
    with pytest.raises(ValueError, match="ascending"):
        build_epoch_orders(np.asarray([2, 1, 3]), seed=123)

    indices = np.asarray([2, 7, 11, 19, 23], dtype=np.int64)
    observed = build_epoch_orders(indices, seed=20260730)
    rng = np.random.Generator(np.random.PCG64(20260730))
    expected = [rng.permutation(indices) for _ in range(EPOCHS)]
    assert len(observed) == EPOCHS
    assert all(np.array_equal(left, right) for left, right in zip(observed, expected))
    assert not np.array_equal(observed[0], observed[1])


def test_support_neutralization_and_nonwrap_direction_are_caller_locked() -> None:
    spatial = torch.arange(1, 1 + 9 * 3 * 4, dtype=torch.float32).reshape(
        1,
        9,
        3,
        4,
    )
    victim = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    source = victim.clone()
    source[:, :, 1, 2] = False
    supported = neutralize_spatial_outside_support(spatial, victim, source)
    assert torch.equal(supported[:, :, 1, 2], torch.ones(1, 9))
    assert torch.equal(supported[:, :, 0, 0], spatial[:, :, 0, 0])

    with pytest.raises(ValueError, match="source_to_destination"):
        displace_spatial_filters_nonwrap(
            spatial,
            torch.tensor([[0, 1]]),
            direction="",
        )
    moved = displace_spatial_filters_nonwrap(
        spatial,
        torch.tensor([[0, 1]]),
        direction="source_to_destination",
    )
    assert torch.equal(moved[:, :, :, 0], torch.ones(1, 9, 3))
    assert torch.equal(moved[:, :, :, 1:], spatial[:, :, :, :-1])


def test_same_weight_causal_paths_preserve_state_and_require_direction() -> None:
    model = initialize_sidecar("ddf_full", fold=1).eval()
    batch = 3
    inputs = torch.randn(batch, 3, 64, 64)
    valid64, valid32, valid16 = _valid_pyramid(batch)
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    clean = model.forward_with_trace(inputs, valid64, valid32, valid16)
    spatial = clean["spatial_filters"]
    channel = clean["channel_filters"]
    assert isinstance(spatial, list) and isinstance(channel, list)
    replay = model.forward_with_trace(
        inputs,
        valid64,
        valid32,
        valid16,
        filter_overrides=list(zip(spatial, channel)),
        override_valid_masks=[valid32, valid16],
    )
    assert torch.equal(clean["scores"], replay["scores"])
    spatial_neutral = model.forward_with_trace(
        inputs,
        valid64,
        valid32,
        valid16,
        causal_mode=CAUSAL_SPATIAL_NEUTRAL,
    )
    channel_neutral = model.forward_with_trace(
        inputs,
        valid64,
        valid32,
        valid16,
        causal_mode=CAUSAL_CHANNEL_NEUTRAL,
    )
    assert not torch.equal(clean["scores"], spatial_neutral["scores"])
    assert not torch.equal(clean["scores"], channel_neutral["scores"])
    offsets = torch.tensor(
        [
            [[[0, 1], [1, 0]]],
            [[[1, 1], [-1, 0]]],
            [[[-1, -1], [0, -1]]],
        ],
        dtype=torch.long,
    ).reshape(batch, 2, 2)
    with pytest.raises(ValueError, match="direction"):
        model.forward_with_trace(
            inputs,
            valid64,
            valid32,
            valid16,
            spatial_offsets=offsets,
        )
    dephased = model.forward_with_trace(
        inputs,
        valid64,
        valid32,
        valid16,
        spatial_offsets=offsets,
        spatial_offset_direction="source_to_destination",
    )
    assert not torch.equal(clean["scores"], dephased["scores"])
    assert all(
        torch.equal(state_before[name], value)
        for name, value in model.state_dict().items()
    )


def test_sequential_recalibration_matches_final_numpy_site_oracles() -> None:
    generator = torch.Generator().manual_seed(2026072906)
    images = torch.randint(
        0,
        256,
        (5, 3, 256, 256),
        dtype=torch.uint8,
        generator=generator,
    )
    valid256 = torch.zeros(5, 1, 256, 256, dtype=torch.bool)
    valid256[:, :, 5:250, 9:244] = True
    valid256[1, :, 70:130, 90:140] = False
    prepared = prepare_cached_model_inputs(images, valid256)
    model = initialize_sidecar("ddf_full", fold=4)
    for module in model.masked_batch_norms():
        module.num_batches_tracked.fill_(EPOCHS * 8)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    records = recalibrate_masked_batch_norms(
        model,
        prepared,
        sample_indices=np.asarray([3, 8, 11, 19, 24], dtype=np.int64),
        batch_size=2,
    )
    assert [record["site_name"] for record in records] == list(
        model.NORMALIZATION_NAMES
    )
    assert all(record["valid_count"] > 1 for record in records)
    assert all(
        int(module.num_batches_tracked) == EPOCHS * 8
        for module in model.masked_batch_norms()
    )
    assert all(
        torch.equal(parameters_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    model.eval()
    trace = model.forward_with_trace(
        prepared.values,
        prepared.valid64,
        prepared.valid32,
        prepared.valid16,
    )
    normalization_inputs = trace["normalization_inputs"]
    normalization_masks = trace["normalization_masks"]
    assert isinstance(normalization_inputs, list)
    assert isinstance(normalization_masks, list)
    for module, values, mask in zip(
        model.masked_batch_norms(),
        normalization_inputs,
        normalization_masks,
    ):
        mean, variance = _numpy_masked_moments(
            values.detach().numpy(),
            mask.detach().numpy(),
        )
        assert np.max(np.abs(module.running_mean.numpy() - mean)) <= 2e-7
        assert np.max(np.abs(module.running_var.numpy() - variance)) <= 2e-7


def test_recalibration_rejects_wrong_training_call_count() -> None:
    values = torch.zeros(2, 3, 64, 64)
    valid64, valid32, valid16 = _valid_pyramid(2, fully_valid=True)
    prepared = PreparedSurfaceInputs(values, valid64, valid32, valid16)
    model = initialize_sidecar("ddf_full", fold=0)
    with pytest.raises(RuntimeError, match="counters"):
        recalibrate_masked_batch_norms(
            model,
            prepared,
            sample_indices=np.asarray([1, 2], dtype=np.int64),
        )


@pytest.mark.parametrize("role", ("ddf_full", "static_matched"))
def test_standard_op_onnx_dynamic_batch_replays(
    tmp_path: Path,
    role: str,
) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = initialize_sidecar(role, fold=0).eval()
    example = (torch.randn(2, 3, 64, 64), *_valid_pyramid(2))
    path = tmp_path / f"pair_surface_ddf_v2_{role}.onnx"
    torch.onnx.export(
        model,
        example,
        path,
        opset_version=17,
        input_names=["images", "valid64", "valid32", "valid16"],
        output_names=["scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "valid64": {0: "batch"},
            "valid32": {0: "batch"},
            "valid16": {0: "batch"},
            "scores": {0: "batch"},
        },
        dynamo=False,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert {node.domain for node in graph.graph.node} == {""}
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    for batch in (1, 2):
        masks = _valid_pyramid(batch)
        inputs = torch.randn(batch, 3, 64, 64)
        expected = model(inputs, *masks).detach().numpy()
        observed = session.run(
            None,
            {
                "images": inputs.numpy(),
                "valid64": masks[0].numpy(),
                "valid32": masks[1].numpy(),
                "valid16": masks[2].numpy(),
            },
        )[0]
        assert observed.shape == (batch, 4)
        assert np.max(np.abs(observed - expected)) <= 1e-5
    assert array_sha256(np.asarray(observed)) == array_sha256(
        np.asarray(observed).copy()
    )


def test_static_role_rejects_dynamic_causal_overrides() -> None:
    model = PairSurfaceDDFV2Sidecar("static_matched").eval()
    inputs = torch.randn(2, 3, 64, 64)
    valid64, valid32, valid16 = _valid_pyramid(2)
    with pytest.raises(ValueError, match="Static role"):
        model.forward_with_trace(
            inputs,
            valid64,
            valid32,
            valid16,
            filter_overrides=[
                (torch.ones(2, 9, 32, 32), torch.ones(2, 16, 9)),
                (torch.ones(2, 9, 16, 16), torch.ones(2, 32, 9)),
            ],
        )

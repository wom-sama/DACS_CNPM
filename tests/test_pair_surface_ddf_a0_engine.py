from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.pair_surface_ddf_a0_engine import (
    CAUSAL_CHANNEL_NEUTRAL,
    CAUSAL_SPATIAL_NEUTRAL,
    ROLE_NAMES,
    PairSurfaceDDFSidecar,
    apply_ddf_standard,
    array_sha256,
    filter_normalize,
    initialize_sidecar,
    learning_rate,
    numpy_ddf,
    numpy_filter_normalize,
    pair_surface_loss,
    parameter_contract,
    prepare_cached_model_inputs,
    task_targets_and_weights,
)


def test_filter_normalization_matches_independent_numpy_fp64() -> None:
    rng = np.random.default_rng(20260725)
    spatial = rng.normal(size=(2, 9, 5, 4))
    channel = rng.normal(size=(2, 3, 9))
    scale = rng.normal(size=(1, 3, 9))

    spatial_torch = filter_normalize(
        torch.from_numpy(spatial),
        tap_dimension=1,
        scale=np.sqrt(2.0) / 3.0,
    ).numpy()
    channel_torch = filter_normalize(
        torch.from_numpy(channel),
        tap_dimension=2,
        scale=torch.from_numpy(scale),
    ).numpy()
    official_channel = (
        (
            torch.from_numpy(channel)
            - torch.from_numpy(channel).mean(dim=2, keepdim=True)
        )
        / (
            torch.from_numpy(channel).std(
                dim=2,
                correction=1,
                keepdim=True,
            )
            + 1e-10
        )
        * torch.from_numpy(scale)
    ).numpy()

    spatial_numpy = numpy_filter_normalize(
        spatial,
        tap_axis=1,
        scale=np.sqrt(2.0) / 3.0,
    )
    channel_numpy = numpy_filter_normalize(
        channel,
        tap_axis=2,
        scale=scale,
    )
    assert np.max(np.abs(spatial_torch - spatial_numpy)) <= 5e-16
    assert np.array_equal(channel_torch, official_channel)
    assert np.max(np.abs(channel_torch - channel_numpy)) <= 1e-15


def test_standard_ddf_matches_independent_numpy_loop_fp64() -> None:
    rng = np.random.default_rng(20260726)
    inputs = rng.normal(size=(2, 3, 5, 6))
    channel = rng.normal(size=(2, 3, 9))
    spatial = rng.normal(size=(2, 9, 5, 6))

    observed = apply_ddf_standard(
        torch.from_numpy(inputs),
        torch.from_numpy(channel),
        torch.from_numpy(spatial),
    ).numpy()
    expected = numpy_ddf(inputs, channel, spatial)

    assert np.max(np.abs(observed - expected)) <= 4e-15


def test_all_roles_have_locked_shapes_parameters_and_finite_outputs() -> None:
    inputs = torch.randn(2, 3, 64, 64)
    expected_counts = {
        "ddf_full": 9380,
        "static_matched": 9435,
        "ddf_spatial_only": 9380,
        "ddf_channel_only": 9380,
        "ddf_full_repeat": 9380,
    }
    for role in ROLE_NAMES:
        model = initialize_sidecar(role, fold=0)
        trace = model.forward_with_trace(inputs)
        contract = parameter_contract(model)
        assert contract["parameter_count"] == expected_counts[role]
        assert trace["scores"].shape == (2, 4)
        assert trace["evidence_maps"].shape == (2, 4, 16, 16)
        assert trace["attention_maps"].shape == (2, 4, 16, 16)
        assert torch.isfinite(trace["scores"]).all()
        assert torch.isfinite(trace["attention_maps"]).all()
        assert torch.allclose(
            trace["attention_maps"].sum(dim=(2, 3)),
            torch.ones(2, 4),
            atol=1e-6,
            rtol=0.0,
        )


def test_primary_roles_share_matching_initial_tensors_and_repeat_differs() -> None:
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
        if all(states[role][name].shape == states["ddf_full"][name].shape for role in primary_roles)
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


def test_full_candidate_groups_receive_gradients_and_update() -> None:
    torch.manual_seed(20260727)
    model = initialize_sidecar("ddf_full", fold=2)
    model.train()
    inputs = torch.randn(16, 3, 64, 64)
    targets = torch.tensor(
        [1, 0, 1, 2, 1, 4, 1, 0, 1, 2, 1, 4, 1, 0, 1, 2],
        dtype=torch.long,
    )
    labels, weights = task_targets_and_weights(targets)
    bbox = torch.zeros(16, 16, 16, dtype=torch.bool)
    bbox[:, 3:13, 2:14] = True
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0001,
    )
    trace = model.forward_with_trace(inputs)
    losses = pair_surface_loss(
        trace["scores"],
        trace["attention_maps"],
        labels,
        weights,
        bbox,
    )
    losses["total"].backward()

    groups = {
        "stem": ("conv1.", "conv2."),
        "spatial": ("block1.spatial_projection", "block2.spatial_projection"),
        "channel": (
            "block1.channel_",
            "block2.channel_",
        ),
        "heads": ("evidence.", "attention."),
    }
    named = dict(model.named_parameters())
    for prefixes in groups.values():
        gradients = [
            named[name].grad
            for name in named
            if name.startswith(prefixes) and named[name].requires_grad
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0

    optimizer.step()
    for prefixes in groups.values():
        changed = [
            not torch.equal(before[name], named[name].detach())
            for name in named
            if name.startswith(prefixes) and named[name].requires_grad
        ]
        assert changed and any(changed)


def test_same_weight_causal_paths_change_scores_without_state_change() -> None:
    torch.manual_seed(20260728)
    model = initialize_sidecar("ddf_full", fold=1).eval()
    inputs = torch.randn(4, 3, 64, 64)
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    clean = model.forward_with_trace(inputs)
    spatial = clean["spatial_filters"]
    channel = clean["channel_filters"]
    assert isinstance(spatial, list)
    assert isinstance(channel, list)

    replay = model.forward_with_trace(
        inputs,
        filter_overrides=list(zip(spatial, channel)),
    )
    crossed = model.forward_with_trace(
        inputs,
        filter_overrides=[
            (spatial[0].roll(1, 0), channel[0].roll(1, 0)),
            (spatial[1].roll(1, 0), channel[1].roll(1, 0)),
        ],
    )
    spatial_neutral = model.forward_with_trace(
        inputs,
        causal_mode=CAUSAL_SPATIAL_NEUTRAL,
    )
    channel_neutral = model.forward_with_trace(
        inputs,
        causal_mode=CAUSAL_CHANNEL_NEUTRAL,
    )
    rolls = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[2, 1], [4, 3]],
            [[3, 2], [5, 1]],
            [[4, 1], [1, 5]],
        ],
        dtype=torch.long,
    )
    dephased = model.forward_with_trace(inputs, spatial_rolls=rolls)

    assert torch.equal(clean["scores"], replay["scores"])
    for candidate in (
        crossed,
        spatial_neutral,
        channel_neutral,
        dephased,
    ):
        assert not torch.equal(clean["scores"], candidate["scores"])
    assert all(
        torch.equal(state_before[name], value)
        for name, value in model.state_dict().items()
    )


def test_balanced_task_weights_and_loss_are_finite() -> None:
    targets = torch.tensor([1, 1, 1, 0, 0, 2, 4], dtype=torch.long)
    labels, weights = task_targets_and_weights(targets)
    rows = float(targets.numel())
    for task in range(4):
        positive = labels[:, task] == 1
        negative = (labels[:, task] == 0) & (weights[:, task] > 0)
        assert float(weights[positive, task].sum()) == pytest.approx(rows / 2)
        assert float(weights[negative, task].sum()) == pytest.approx(rows / 2)

    scores = torch.randn(7, 4, requires_grad=True)
    attention = torch.softmax(torch.randn(7, 4, 256), dim=2).reshape(
        7, 4, 16, 16
    )
    bbox = torch.zeros(7, 16, 16, dtype=torch.bool)
    bbox[:, 5:11, 4:12] = True
    losses = pair_surface_loss(
        scores,
        attention,
        labels,
        weights,
        bbox,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_cached_input_transform_and_cosine_schedule_are_locked() -> None:
    values = torch.full((2, 3, 256, 256), 128, dtype=torch.uint8)
    transformed = prepare_cached_model_inputs(values)
    expected = (
        torch.full((1, 3), 128.0 / 255.0)
        - torch.tensor([[0.485, 0.456, 0.406]])
    ) / torch.tensor([[0.229, 0.224, 0.225]])
    assert transformed.shape == (2, 3, 64, 64)
    assert transformed.dtype == torch.float32
    assert torch.allclose(
        transformed[:, :, 0, 0],
        expected.expand(2, -1),
        atol=1e-6,
        rtol=0.0,
    )
    assert learning_rate(0) == pytest.approx(0.003)
    assert learning_rate(19) == pytest.approx(0.0, abs=1e-18)
    assert all(
        learning_rate(index) >= learning_rate(index + 1)
        for index in range(19)
    )


def test_standard_op_onnx_dynamic_batch_replays(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = initialize_sidecar("ddf_full", fold=0).eval()
    example = torch.randn(2, 3, 64, 64)
    path = tmp_path / "pair_surface_ddf.onnx"
    torch.onnx.export(
        model,
        example,
        path,
        opset_version=17,
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
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
        inputs = torch.randn(batch, 3, 64, 64)
        expected = model(inputs).detach().numpy()
        observed = session.run(None, {"images": inputs.numpy()})[0]
        assert observed.shape == (batch, 4)
        assert np.max(np.abs(observed - expected)) <= 1e-5
    assert array_sha256(np.asarray(observed)) == array_sha256(
        np.asarray(observed).copy()
    )


def test_static_role_rejects_dynamic_causal_overrides() -> None:
    model = PairSurfaceDDFSidecar("static_matched").eval()
    inputs = torch.randn(2, 3, 64, 64)
    fake_spatial = torch.ones(2, 9, 32, 32)
    fake_channel = torch.ones(2, 16, 9)
    with pytest.raises(ValueError, match="Static role"):
        model.forward_with_trace(
            inputs,
            filter_overrides=[
                (fake_spatial, fake_channel),
                (torch.ones(2, 9, 16, 16), torch.ones(2, 32, 9)),
            ],
        )

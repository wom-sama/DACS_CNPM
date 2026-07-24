from __future__ import annotations

import numpy as np
import torch

from trkh.tools import restricted_negative_learnable_isda_a0_engine as engine


def test_hash_encodings_include_array_metadata_and_string_boundaries() -> None:
    assert engine.array_sha256(
        np.asarray([1, 2], dtype=np.int64)
    ) != engine.array_sha256(np.asarray([1, 2], dtype=np.int32))
    assert engine.string_sequence_sha256(
        ["ab", "c"]
    ) != engine.string_sequence_sha256(["a", "bc"])
    assert engine.string_sequence_sha256(
        ["ab", "c"]
    ) == engine.string_sequence_sha256(["ab", "c"])


def test_restricted_equation_matches_numpy_oracle() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(9, 6)).astype(np.float64)
    labels = np.asarray([0, 1, 2, 3, 4, 0, 2, 4, 1], dtype=np.int64)
    weight = generator.normal(size=(5, 6)).astype(np.float64)
    bias = generator.normal(size=5).astype(np.float64)
    covariance = generator.uniform(0.01, 0.3, size=(9, 6)).astype(np.float64)
    strength = 2.75

    augmented, clean, delta = engine.isda_logits(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
        weight=torch.from_numpy(weight),
        bias=torch.from_numpy(bias),
        covariance=torch.from_numpy(covariance),
        lambda_strength=strength,
    )
    expected_clean = features @ weight.T + bias
    expected_delta = np.zeros((labels.size, 5), dtype=np.float64)
    for row, target in enumerate(labels.tolist()):
        if target not in {0, 2, 4}:
            continue
        difference = weight[1] - weight[target]
        expected_delta[row, 1] = (
            0.5
            * strength
            * np.sum(difference**2 * covariance[row])
        )
    np.testing.assert_allclose(clean.numpy(), expected_clean, atol=1e-12)
    np.testing.assert_allclose(delta.numpy(), expected_delta, atol=1e-12)
    np.testing.assert_allclose(
        augmented.numpy(), expected_clean + expected_delta, atol=1e-12
    )


def test_restricted_masks_leave_class1_and_class3_clean() -> None:
    torch.manual_seed(11)
    features = torch.randn(10, 8)
    labels = torch.tensor([0, 1, 2, 3, 4, 1, 3, 0, 2, 4])
    weight = torch.randn(5, 8)
    bias = torch.randn(5)
    covariance = torch.rand_like(features)
    _, _, delta = engine.isda_logits(
        features=features,
        labels=labels,
        weight=weight,
        bias=bias,
        covariance=covariance,
        lambda_strength=5.0,
    )
    assert torch.count_nonzero(delta[labels == 1]).item() == 0
    assert torch.count_nonzero(delta[labels == 3]).item() == 0
    assert torch.count_nonzero(delta[:, [0, 2, 3, 4]]).item() == 0


def test_zero_output_initializes_exact_classwise_covariance() -> None:
    predictor = engine.CovariancePredictor(
        input_dim=8, hidden_dim=4, output_dim=8, seed=17
    )
    features = torch.randn(12, 8)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
    base = torch.rand(5, 8).clamp_min(0.01)
    covariance, scales = engine.predict_covariance(
        predictor, features, labels, base
    )
    torch.testing.assert_close(scales, torch.ones_like(scales))
    torch.testing.assert_close(covariance, base[labels])


def test_functional_sgd_matches_real_sgd_with_and_without_buffer() -> None:
    state = engine.initialize_linear_head(
        feature_dim=6,
        class_count=5,
        seed=23,
        device=torch.device("cpu"),
    )
    weight_gradient = torch.randn_like(state.weight)
    bias_gradient = torch.randn_like(state.bias)
    expected_weight, expected_bias = engine.functional_sgd_parameters(
        state,
        weight_gradient,
        bias_gradient,
        learning_rate=0.03,
    )
    engine.real_sgd_step(
        state,
        weight_gradient,
        bias_gradient,
        learning_rate=0.03,
    )
    torch.testing.assert_close(state.weight, expected_weight)
    torch.testing.assert_close(state.bias, expected_bias)

    next_weight_gradient = torch.randn_like(state.weight)
    next_bias_gradient = torch.randn_like(state.bias)
    expected_weight, expected_bias = engine.functional_sgd_parameters(
        state,
        next_weight_gradient,
        next_bias_gradient,
        learning_rate=0.01,
    )
    engine.real_sgd_step(
        state,
        next_weight_gradient,
        next_bias_gradient,
        learning_rate=0.01,
    )
    torch.testing.assert_close(state.weight, expected_weight)
    torch.testing.assert_close(state.bias, expected_bias)


def test_meta_gradient_reaches_output_first_and_hidden_second() -> None:
    torch.manual_seed(29)
    predictor = engine.CovariancePredictor(
        input_dim=8, hidden_dim=4, output_dim=8, seed=31
    )
    head = engine.initialize_linear_head(
        feature_dim=8,
        class_count=5,
        seed=37,
        device=torch.device("cpu"),
    )
    head.weight_momentum = torch.randn_like(head.weight) * 0.02
    head.bias_momentum = torch.randn_like(head.bias) * 0.02
    train_features = torch.randn(32, 8)
    meta_features = torch.randn(32, 8)
    train_labels = torch.tensor(([0, 1, 2, 3, 4] * 7)[:32])
    meta_labels = torch.tensor(([4, 3, 2, 1, 0] * 7)[:32])
    base = torch.rand(5, 8) * 0.3 + 0.05

    first = engine.meta_covnet_step(
        predictor=predictor,
        head=head,
        train_features=train_features,
        train_labels=train_labels,
        meta_features=meta_features,
        meta_labels=meta_labels,
        covariance_input=train_features,
        base_variance=base,
        head_learning_rate=0.03,
        covnet_learning_rate=0.01,
        lambda_strength=5.0,
        eligible_classes=(0, 2, 4),
        rival_mode="focus",
    )
    assert first["output_gradient_norm"] > 0.0
    assert first["hidden_gradient_norm"] == 0.0

    second = engine.meta_covnet_step(
        predictor=predictor,
        head=head,
        train_features=train_features,
        train_labels=train_labels,
        meta_features=meta_features,
        meta_labels=meta_labels,
        covariance_input=train_features,
        base_variance=base,
        head_learning_rate=0.03,
        covnet_learning_rate=0.01,
        lambda_strength=5.0,
        eligible_classes=(0, 2, 4),
        rival_mode="focus",
    )
    assert second["output_gradient_norm"] > 0.0
    assert second["hidden_gradient_norm"] > 0.0


def test_derangement_is_label_preserving_source_disjoint_permutation() -> None:
    labels = np.repeat(np.arange(5, dtype=np.int64), 12)
    indices = np.arange(labels.size, dtype=np.int64)
    sources = np.asarray(
        [f"source_{class_index}_{row}" for class_index in range(5) for row in range(12)]
    )
    mapping = engine.build_same_label_source_derangement(
        indices,
        labels,
        sources,
        outer_fold=2,
        partition_name="a",
    )
    np.testing.assert_array_equal(labels[indices], labels[mapping[indices]])
    assert np.all(sources[indices] != sources[mapping[indices]])
    assert len(np.unique(mapping[indices])) == len(indices)
    repeated = engine.build_same_label_source_derangement(
        indices,
        labels,
        sources,
        outer_fold=2,
        partition_name="a",
    )
    np.testing.assert_array_equal(mapping, repeated)


def test_balanced_meta_schedule_is_exact_deterministic_and_cyclic() -> None:
    labels = np.tile(np.arange(5, dtype=np.int64), 40)
    indices = np.arange(labels.size, dtype=np.int64)
    sources = np.asarray([f"source_{index // 2}" for index in indices])
    schedule = engine.build_balanced_meta_schedule(
        indices,
        labels,
        sources,
        outer_fold=2,
        partition_name="b",
        slot_count=9,
    )
    repeated = engine.build_balanced_meta_schedule(
        indices,
        labels,
        sources,
        outer_fold=2,
        partition_name="b",
        slot_count=9,
    )
    np.testing.assert_array_equal(schedule, repeated)
    assert schedule.shape == (9, 30)
    for row in schedule:
        assert len(np.unique(row)) == 30
        assert len(np.unique(sources[row])) == 30
        np.testing.assert_array_equal(
            np.bincount(labels[row], minlength=5),
            np.full(5, 6, dtype=np.int64),
        )
    class_1_usage = np.bincount(
        schedule[labels[schedule] == 1],
        minlength=len(labels),
    )
    class_1_usage = class_1_usage[labels == 1]
    assert int(class_1_usage.max() - class_1_usage.min()) <= 1
    assert (
        engine.balanced_meta_max_usage_spread(
            schedule,
            indices,
            labels,
        )
        <= 1
    )
    assert engine.balanced_meta_slot_count(6) == 12


def test_outer_holdout_swap_donors_are_locked_and_source_disjoint() -> None:
    labels = np.arange(60, dtype=np.int64) % 5
    sources = np.asarray([f"source_{index}" for index in range(60)])
    fit = np.arange(50, dtype=np.int64)
    held = np.arange(50, 60, dtype=np.int64)
    donors = engine.build_outer_holdout_swap_donors(
        fit,
        held,
        labels,
        sources,
        outer_fold=3,
    )
    repeated = engine.build_outer_holdout_swap_donors(
        fit,
        held,
        labels,
        sources,
        outer_fold=3,
    )
    np.testing.assert_array_equal(donors, repeated)
    assert np.all(np.isin(donors, fit))
    np.testing.assert_array_equal(labels[held], labels[donors])
    assert np.all(sources[held] != sources[donors])


def test_semantic_draw_and_cosine_proxy_are_deterministic() -> None:
    feature = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    covariance = np.asarray([0.25, 0.5, 1.0], dtype=np.float32)
    child, epsilon = engine.draw_semantic_feature(
        feature,
        covariance,
        outer_fold=2,
        sample_index=41,
        role="candidate",
        draw_index=1,
    )
    repeated, repeated_epsilon = engine.draw_semantic_feature(
        feature,
        covariance,
        outer_fold=2,
        sample_index=41,
        role="candidate",
        draw_index=1,
    )
    np.testing.assert_array_equal(child, repeated)
    np.testing.assert_array_equal(epsilon, repeated_epsilon)
    np.testing.assert_allclose(
        child,
        feature + np.sqrt(5.0 * covariance).astype(np.float32) * epsilon,
    )

    features = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    sources = np.asarray(["anchor", "other_1", "other_2", "other_3"])
    proxy, distance = engine.cosine_nearest_proxy(
        np.asarray([1.0, 0.0], dtype=np.float32),
        features,
        np.asarray([2, 1, 3, 0], dtype=np.int64),
        sources,
        excluded_source="anchor",
    )
    assert proxy == 1
    assert abs(distance) <= 1e-7


def test_paper_schedule_and_semantic_lambda_are_locked() -> None:
    first = engine.paper_learning_rate(
        initial_lr=0.03,
        epoch=0,
        step=0,
        steps_per_epoch=114,
    )
    warmup_last = engine.paper_learning_rate(
        initial_lr=0.03,
        epoch=4,
        step=113,
        steps_per_epoch=114,
    )
    cosine_first = engine.paper_learning_rate(
        initial_lr=0.03,
        epoch=5,
        step=0,
        steps_per_epoch=114,
    )
    assert first == 0.03 / (1 + 5 * 114)
    assert warmup_last < 0.03
    assert cosine_first == 0.03
    assert engine.semantic_lambda(9) == 0.0
    assert engine.semantic_lambda(10) == 0.25
    assert engine.semantic_lambda(29) == 5.0


def test_joint_training_updates_covnet_while_detached_candidate_does_not() -> None:
    torch.manual_seed(41)
    features = torch.randn(32, 8)
    labels = torch.tensor(([0, 1, 2, 3, 4] * 7)[:32])
    base = torch.rand(5, 8) * 0.2 + 0.01

    def state(role: str) -> tuple[engine.LinearHeadState, engine.CovariancePredictor]:
        head = engine.initialize_linear_head(
            feature_dim=8,
            class_count=5,
            seed=43,
            device=torch.device("cpu"),
        )
        predictor = engine.CovariancePredictor(
            input_dim=8, hidden_dim=4, output_dim=8, seed=47
        )
        before = torch.cat(
            [parameter.detach().flatten() for parameter in predictor.parameters()]
        )
        engine.real_head_step(
            role=role,
            predictor=predictor,
            head=head,
            features=features,
            labels=labels,
            covariance_input=features,
            base_variance=base,
            learning_rate=0.03,
            covnet_learning_rate=0.001,
            lambda_strength=5.0,
        )
        after = torch.cat(
            [parameter.detach().flatten() for parameter in predictor.parameters()]
        )
        return head, predictor, before, after

    _, _, detached_before, detached_after = state("rn_lisda_candidate")
    _, _, joint_before, joint_after = state("rn_lisda_joint_no_meta")
    torch.testing.assert_close(detached_before, detached_after)
    assert not torch.equal(joint_before, joint_after)


def test_full_fold_loop_locks_update_counts_and_clean_holdout_shape() -> None:
    torch.manual_seed(53)
    rows = 480
    features = torch.randn(rows, 16)
    labels = torch.as_tensor(np.tile(np.arange(5), rows // 5), dtype=torch.long)
    sources = np.asarray([f"source_{index}" for index in range(rows)])
    partition_a = np.arange(0, 200, dtype=np.int64)
    partition_b = np.arange(200, 400, dtype=np.int64)
    holdout = np.arange(400, 480, dtype=np.int64)

    result = engine.train_role_fold(
        role="rn_lisda_candidate",
        features=features,
        labels=labels,
        source_stems=sources,
        outer_fold=0,
        partition_a=partition_a,
        partition_b=partition_b,
        holdout=holdout,
    )
    assert result["paired_steps"] == 6
    assert result["head_update_count"] == 30 * 6 * 2
    assert result["meta_update_count"] == 12 * 2
    assert result["balanced_meta_slot_count"] == 12
    assert result["balanced_meta_schedule_a_max_class_usage_spread"] <= 1
    assert result["balanced_meta_schedule_b_max_class_usage_spread"] <= 1
    assert len(result["balanced_meta_schedule_a_sha256"]) == 64
    assert len(result["balanced_meta_schedule_b_sha256"]) == 64
    assert result["holdout_score_calls"] == 1
    assert result["holdout_score_epoch"] == 30
    assert result["partition_a_indices_sha256"] == engine.array_sha256(
        partition_a
    )
    assert result["partition_b_indices_sha256"] == engine.array_sha256(
        partition_b
    )
    assert result["holdout_indices_sha256"] == engine.array_sha256(holdout)
    assert result["probabilities"].shape == (80, 5)
    np.testing.assert_allclose(
        result["probabilities"].sum(axis=1),
        np.ones(80),
        atol=1e-6,
    )
    assert len(result["trace"]) == 30
    assert all(
        len(row["partition_a_consumed_indices_sha256"]) == 64
        and len(row["partition_b_consumed_indices_sha256"]) == 64
        for row in result["trace"]
    )
    assert len(result["meta_trace"]) == 24
    assert all(
        row["meta_class_counts"] == [6, 6, 6, 6, 6]
        and row["meta_unique_row_count"] == 30
        and row["meta_unique_source_count"] == 30
        and row["train_meta_source_overlap_count"] == 0
        and len(row["meta_indices_sha256"]) == 64
        for row in result["meta_trace"]
    )
    assert result["meta_trace"][0]["hidden_gradient_norm"] == 0.0
    assert result["meta_trace"][0]["output_gradient_norm"] > 0.0
    assert result["meta_trace"][1]["hidden_gradient_norm"] > 0.0

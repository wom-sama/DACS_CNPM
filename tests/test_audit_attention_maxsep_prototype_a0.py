from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools import audit_attention_maxsep_prototype_a0 as audit


def _synthetic_inputs(
    rows: int = 3, *, dtype: np.dtype = np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(91)
    features = generator.normal(
        0.0,
        0.2,
        size=(rows, audit.FEATURE_CHANNELS, audit.FEATURE_SIZE, audit.FEATURE_SIZE),
    ).astype(dtype)
    valid = np.ones((rows, audit.FEATURE_SIZE, audit.FEATURE_SIZE), dtype=np.bool_)
    valid[:, :2] = False
    bbox = np.zeros_like(valid)
    bbox[:, 3:14, 4:13] = True
    keeper = generator.dirichlet(np.ones(audit.NUM_CLASSES), size=rows).astype(dtype)
    targets = np.arange(rows, dtype=np.int64) % audit.NUM_CLASSES
    return features, valid, bbox, keeper, targets


def _numpy_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "attention": outputs["evidence_attention"][:, 0].detach().cpu().numpy(),
        "attention_logits": outputs["attention_logits"].detach().cpu().numpy(),
        "prototype_logits": outputs["prototype_logits"].detach().cpu().numpy(),
        "class_evidence_max": outputs["class_evidence_max"].detach().cpu().numpy(),
        "class_evidence_mean": outputs["class_evidence_mean"].detach().cpu().numpy(),
        "class_min_distance": outputs["class_min_distance"].detach().cpu().numpy(),
        "prototype_evidence": outputs["prototype_evidence"].detach().cpu().numpy(),
        "projected_mean": outputs["projected"].mean(dim=(2, 3)).detach().cpu().numpy(),
    }


def test_parse_defaults_and_locked_arguments() -> None:
    args = audit.parse_args([])
    audit._validate_locked_args(args)
    assert args.batch_size == audit.BATCH_SIZE
    assert args.num_workers == audit.NUM_WORKERS
    args.batch_size += 1
    with pytest.raises(ValueError, match="differ from prospective lock"):
        audit._validate_locked_args(args)


def test_head_shapes_padding_and_fixed_connections() -> None:
    features, valid, _, _, _ = _synthetic_inputs(rows=2, dtype=np.float32)
    head = audit.AttentionMaxSepPrototypeHead().eval()
    with torch.inference_mode():
        outputs = head(torch.from_numpy(features), torch.from_numpy(valid))
    assert outputs["prototype_logits"].shape == (2, audit.NUM_CLASSES)
    assert outputs["distances"].shape == (
        2,
        audit.NUM_PROTOTYPES,
        audit.FEATURE_SIZE,
        audit.FEATURE_SIZE,
    )
    assert torch.equal(outputs["attention"], outputs["evidence_attention"])
    assert float((outputs["attention"][:, 0] * (~torch.from_numpy(valid))).abs().max()) == 0.0
    assert torch.equal(
        head.classifier_weight,
        audit._classifier_weights(head.prototype_class_identity),
    )
    assert head.trainable_parameter_count == 19_526


def test_independent_numpy_forward_and_gradient_oracles() -> None:
    result = audit.engineering_checks()
    assert result["passed"]
    assert max(result["equation_errors"].values()) <= 1e-10
    assert result["prototype_gradient_max_abs_error"] <= 1e-10


def test_matched_heads_and_dephasing_are_rng_local_and_exact() -> None:
    before = audit._global_rng_snapshot()
    heads = audit.build_matched_heads(fold=3)
    after = audit._global_rng_snapshot()
    assert audit._global_rng_equal(before, after)
    assert len({audit._model_state_sha256(head) for head in heads.values()}) == 1
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(
        2,
        audit.FEATURE_CHANNELS,
        audit.FEATURE_SIZE,
        audit.FEATURE_SIZE,
        generator=generator,
    )
    indices = audit.dephase_gather_indices(3)
    first = audit.channel_dephase(features, indices)
    second = audit.channel_dephase(features, indices)
    assert torch.equal(first, second)
    assert not torch.equal(first, features)
    assert torch.equal(
        first.flatten(2).sort(dim=2).values,
        features.flatten(2).sort(dim=2).values,
    )


def test_roll_and_cycle_are_active_same_weight_interventions() -> None:
    features, valid, _, _, _ = _synthetic_inputs(rows=2, dtype=np.float32)
    head = audit.AttentionMaxSepPrototypeHead().eval()
    with torch.inference_mode():
        aligned = head(torch.from_numpy(features), torch.from_numpy(valid))
        rolled = head(
            torch.from_numpy(features),
            torch.from_numpy(valid),
            attention_roll=audit.ATTENTION_ROLL,
        )
        cycled = head(
            torch.from_numpy(features),
            torch.from_numpy(valid),
            class_cycle=True,
        )
    assert not torch.equal(
        aligned["prototype_logits"], rolled["prototype_logits"]
    )
    assert not torch.equal(
        aligned["prototype_logits"], cycled["prototype_logits"]
    )
    assert torch.equal(aligned["attention"], rolled["attention"])
    assert not torch.equal(rolled["attention"], rolled["evidence_attention"])


def test_torch_and_numpy_learned_descriptors_match_fp64() -> None:
    features, valid, bbox, keeper, _ = _synthetic_inputs(rows=3)
    head = audit.AttentionMaxSepPrototypeHead().double().eval()
    with torch.inference_mode():
        outputs = head(torch.from_numpy(features), torch.from_numpy(valid))
        observed = audit.learned_descriptor_torch(
            outputs,
            torch.from_numpy(valid),
            torch.from_numpy(bbox),
            torch.from_numpy(keeper),
            include_keeper=True,
        ).numpy()
    expected = audit.learned_descriptor(
        _numpy_outputs(outputs), valid, bbox, keeper, include_keeper=True
    )
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=0.0)
    assert observed.shape == (3, 38)


def test_descriptor_bank_has_all_locked_roles_and_dimensions() -> None:
    features, valid, bbox, keeper, _ = _synthetic_inputs(rows=4, dtype=np.float32)
    head = audit.AttentionMaxSepPrototypeHead().eval()
    with torch.inference_mode():
        outputs = _numpy_outputs(
            head(torch.from_numpy(features), torch.from_numpy(valid))
        )
    learned = {role: outputs for role in audit.LEARNED_DESCRIPTOR_ROLES}
    descriptors = audit.build_role_descriptors(
        learned, features, valid, bbox, keeper
    )
    assert set(descriptors) == set(audit.ROLE_NAMES)
    assert descriptors[audit.CANDIDATE_ROLE].shape == (4, 38)
    assert descriptors[audit.WITHOUT_KEEPER_ROLE].shape == (4, 33)
    assert descriptors[audit.GAP_KEEPER_ROLE].shape == (4, 517)
    assert all(np.isfinite(value).all() for value in descriptors.values())


def test_oof_readout_states_replay_scores_actions_and_thresholds() -> None:
    generator = np.random.default_rng(55)
    rows_per_fold = 12
    rows = rows_per_fold * len(audit.FOLDS)
    folds = np.repeat(np.asarray(audit.FOLDS), rows_per_fold)
    labels = np.tile(np.asarray([0, 1]), rows // 2)
    signal = labels[:, None] * 1.2 + generator.normal(0.0, 0.4, size=(rows, 1))
    descriptors = {
        role: np.concatenate(
            (signal, generator.normal(size=(rows, 4))), axis=1
        )
        for role in audit.ROLE_NAMES
    }
    sources = [f"source_{fold}_{index}" for fold, index in zip(folds, range(rows))]
    scores, actions, states = audit.fit_oof_readouts(
        descriptors, labels, folds, sources
    )
    replay_scores, replay_actions = audit.apply_readout_states(
        descriptors, folds, states
    )
    for role in audit.ROLE_NAMES:
        np.testing.assert_allclose(scores[role], replay_scores[role], atol=1e-12)
        assert np.array_equal(actions[role], replay_actions[role])
    assert audit._verify_readout_thresholds(
        descriptors, folds, labels, states
    ) <= 1e-12


def test_readout_score_torch_matches_persisted_equation() -> None:
    descriptor = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    state = {
        "scaler_mean": [0.2, -0.1],
        "scaler_scale": [0.5, 2.0],
        "coefficient": [[0.7, -0.3]],
        "intercept": [0.15],
    }
    observed = audit.readout_score_torch(descriptor, state).numpy()
    decision = (
        (descriptor.numpy() - np.asarray(state["scaler_mean"]))
        / np.asarray(state["scaler_scale"])
        * np.asarray(state["coefficient"])[0]
    ).sum(axis=1) + state["intercept"][0]
    expected = 1.0 / (1.0 + np.exp(-decision))
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_state_mapping_hash_includes_names_shapes_and_values() -> None:
    first = {"a": torch.tensor([1.0, 2.0]), "b": torch.ones(2, 2)}
    reordered = {"b": torch.ones(2, 2), "a": torch.tensor([1.0, 2.0])}
    changed = {"a": torch.tensor([1.0, 3.0]), "b": torch.ones(2, 2)}
    assert audit._state_mapping_sha256(first) == audit._state_mapping_sha256(
        reordered
    )
    assert audit._state_mapping_sha256(first) != audit._state_mapping_sha256(changed)


def test_epoch_orders_are_deterministic_complete_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "EPOCHS", 3)
    positions = np.asarray([2, 4, 7, 9, 11, 15], dtype=np.int64)
    first, first_hash = audit._epoch_orders(positions, fold=3)
    second, second_hash = audit._epoch_orders(positions, fold=3)
    assert first_hash == second_hash
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(np.array_equal(np.sort(order), positions) for order in first)
    assert len({order.tobytes() for order in first}) == 3


def test_fixed_xai_positions_select_tp_fn_fp_per_fold() -> None:
    rows = []
    sample_index = 0
    for fold in audit.FOLDS:
        for target, prediction in ((1, 1), (1, 0), (0, 1)):
            rows.append(
                audit.CleanTrainRow(
                    sample_index=sample_index,
                    source_stem=f"source_{sample_index}",
                    image_path=Path(f"train/source_{sample_index}.jpg"),
                    fold=fold,
                    target=target,
                    keeper_prediction=prediction,
                    keeper_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
                )
            )
            sample_index += 1
    positions = audit.fixed_xai_positions(rows)
    assert positions == list(range(15))


def test_feature_cache_cleanup_closes_and_removes_memmap(tmp_path: Path) -> None:
    path = tmp_path / "cache.npy"
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(2, 3))
    cache[:] = 1.0
    extraction: dict[str, object] = {
        "features": cache,
        "cache": {"path": str(path)},
    }
    result = audit._close_delete_feature_cache(extraction)
    assert result["deleted"]
    assert not path.exists()
    assert "features" not in extraction

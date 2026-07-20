from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools import audit_deepbdc_stem_joint_dependence_a0 as audit


def test_torch_bdc_matches_independent_numpy_fp64_oracle() -> None:
    value = np.random.default_rng(12).normal(size=(3, 9, 17))
    tau = np.log(1.0 / 34.0)
    matrix = audit.bdc_matrix_torch(torch.from_numpy(value), tau).numpy()
    descriptor = audit.bdc_descriptor_torch(torch.from_numpy(value), tau).numpy()
    assert np.max(np.abs(matrix - audit.bdc_matrix_numpy(value, tau))) <= 1e-11
    assert (
        np.max(np.abs(descriptor - audit.bdc_descriptor_numpy(value, tau)))
        <= 1e-11
    )


def test_bdc_engineering_checks_cover_locked_equation_properties() -> None:
    result = audit.engineering_checks()
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["errors"]["finite_difference_gradient"] <= 2e-5
    assert result["errors"]["descriptor_bf16"] <= 2e-2


def test_bdc_preserves_singleton_batch_and_rejects_invalid_input() -> None:
    value = torch.randn(1, 7, 13)
    descriptor = audit.bdc_descriptor_torch(value, np.log(1.0 / 26.0))
    assert descriptor.shape == (1, 28)
    with pytest.raises(ValueError, match="shape"):
        audit.bdc_matrix_torch(torch.randn(7, 13), 0.0)
    with pytest.raises(ValueError, match="finite"):
        invalid = value.clone()
        invalid[0, 0, 0] = torch.nan
        audit.bdc_matrix_torch(invalid, 0.0)


def test_dephasing_is_deterministic_marginal_preserving_and_rng_local() -> None:
    before = audit._global_rng_snapshot()
    features = torch.arange(
        2 * audit.PROJECTION_DIM * audit.SPATIAL_TOKENS, dtype=torch.float32
    ).reshape(2, audit.PROJECTION_DIM, audit.SPATIAL_TOKENS)
    indices = torch.tensor([91, 203], dtype=torch.long)
    first = audit.dephase_projected(features, indices)
    second = audit.dephase_projected(features, indices)
    after = audit._global_rng_snapshot()
    assert torch.equal(first, second)
    assert audit._global_rng_equal(before, after)
    assert torch.equal(
        torch.sort(first, dim=2).values,
        torch.sort(features, dim=2).values,
    )
    assert not torch.equal(first, features)


def test_valid_crop_removes_padding_and_resizes_to_locked_shape() -> None:
    stem = torch.arange(2 * 256 * 32 * 32, dtype=torch.float32).reshape(
        2, 256, 32, 32
    )
    mask = torch.zeros(2, 1, 256, 256, dtype=torch.bool)
    mask[0, :, 32:224, 64:192] = True
    mask[1, :, 64:192, 16:240] = True
    cropped, geometry = audit.crop_valid_stem_maps(stem, mask)
    assert cropped.shape == (2, 256, 16, 16)
    assert geometry[0]["stem_x0"] == 8
    assert geometry[0]["stem_x1_exclusive"] == 24
    assert geometry[0]["stem_y0"] == 4
    assert geometry[0]["stem_y1_exclusive"] == 28
    assert geometry[1]["stem_x0"] == 2
    assert geometry[1]["stem_x1_exclusive"] == 30


def test_matched_heads_share_projection_without_changing_global_rng() -> None:
    before = audit._global_rng_snapshot()
    heads = audit.make_matched_heads(
        fold=3,
        base_mean=np.zeros(5, dtype=np.float32),
        base_std=np.ones(5, dtype=np.float32),
    )
    after = audit._global_rng_snapshot()
    assert audit._global_rng_equal(before, after)
    states = [
        head.projection.state_dict()
        for head in heads.values()
        if head.projection is not None
    ]
    for state in states[1:]:
        assert all(torch.equal(state[key], states[0][key]) for key in states[0])


@pytest.mark.parametrize(
    ("role", "expected_dim"),
    [
        ("base_logprob", 5),
        ("mean32_base", 37),
        ("cov32_base", 533),
        ("bdc32_dephased_base", 533),
        ("bdc32_aligned_base", 533),
        ("bdc32_aligned_only", 528),
    ],
)
def test_head_roles_emit_locked_feature_dimensions(
    role: str, expected_dim: int
) -> None:
    head = audit.StemJointHead(
        role,
        base_mean=np.zeros(5, dtype=np.float32),
        base_std=np.ones(5, dtype=np.float32),
    ).eval()
    maps = torch.randn(2, 256, 16, 16)
    probabilities = torch.softmax(torch.randn(2, 5), dim=1)
    indices = torch.tensor([11, 12])
    features, _ = head.feature_vector(maps, probabilities, indices)
    assert features.shape == (2, expected_dim)
    assert head(maps, probabilities, indices).shape == (2,)
    assert torch.isfinite(features).all()


def test_candidate_backward_reaches_projection_tau_and_classifier() -> None:
    head = audit.StemJointHead(
        "bdc32_aligned_base",
        base_mean=np.zeros(5, dtype=np.float32),
        base_std=np.ones(5, dtype=np.float32),
    ).train()
    maps = torch.randn(4, 256, 16, 16)
    probabilities = torch.softmax(torch.randn(4, 5), dim=1)
    indices = torch.tensor([4, 5, 6, 7])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(maps, probabilities, indices), labels
    )
    loss.backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().max()) > 0.0, name


def test_parameter_hash_supports_scalar_tau() -> None:
    head = audit.StemJointHead(
        "bdc32_aligned_base",
        base_mean=np.zeros(5, dtype=np.float32),
        base_std=np.ones(5, dtype=np.float32),
    )
    first = audit._parameter_sha256(head)
    second = audit._parameter_dict_sha256(
        {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    )
    assert first == second


def _synthetic_oof_state(rows_per_fold: int = 6):
    rows = len(audit.FOLDS) * rows_per_fold
    folds = np.repeat(np.asarray(audit.FOLDS, dtype=np.int64), rows_per_fold)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), rows // 2)
    sample_indices = np.arange(100, 100 + rows, dtype=np.int64)
    generator = np.random.default_rng(81)
    heads = {}
    thresholds = {}
    features = {
        role: np.empty((rows, audit._role_feature_dim(role)), dtype=np.float64)
        for role in audit.ROLE_NAMES
    }
    scores = {
        role: np.full(rows, np.nan, dtype=np.float64) for role in audit.ROLE_NAMES
    }
    actions = {
        role: np.zeros(rows, dtype=np.bool_) for role in audit.ROLE_NAMES
    }
    for fold in audit.FOLDS:
        fold_heads = audit.make_matched_heads(
            fold=fold,
            base_mean=np.zeros(5, dtype=np.float32),
            base_std=np.ones(5, dtype=np.float32),
        )
        heads[fold] = fold_heads
        thresholds[fold] = {}
        held = folds == fold
        for role in audit.ROLE_NAMES:
            source = (
                "bdc32_aligned_base" if role == audit.SAME_WEIGHT_ROLE else role
            )
            matrix = generator.normal(size=(held.sum(), audit._role_feature_dim(role)))
            features[role][held] = matrix
            head = fold_heads[source]
            weight = head.classifier.weight.detach().double().numpy().reshape(-1)
            bias = float(head.classifier.bias.detach().double().item())
            role_scores = audit._score_from_features(matrix, weight, bias)
            scores[role][held] = role_scores
            thresholds[fold][role] = 0.5
            actions[role][held] = role_scores >= 0.5
    return {
        "features": features,
        "heads": heads,
        "thresholds": thresholds,
        "labels": labels,
        "folds": folds,
        "sample_indices": sample_indices,
        "candidate_projected_channel_variance": np.ones(
            audit.PROJECTION_DIM, dtype=np.float64
        ),
        "scores": scores,
        "actions": actions,
    }


def test_serialized_head_state_replays_scores_and_actions(tmp_path: Path) -> None:
    oof = _synthetic_oof_state()
    artifacts = audit._save_replay_state_with_axes(tmp_path, oof=oof)
    scores, actions = audit.replay_oof_from_saved_state(
        feature_path=Path(artifacts["features"]["path"]),
        state_path=Path(artifacts["head_states"]["path"]),
        metadata_path=Path(artifacts["metadata"]["path"]),
    )
    assert audit._maximum_score_difference(oof["scores"], scores) <= 1e-12
    assert all(
        np.array_equal(oof["actions"][role], actions[role])
        for role in audit.ROLE_NAMES
    )


def test_review_selection_is_fixed_unique_sixteen_rows() -> None:
    labels = np.asarray([1] * 20 + [0] * 20, dtype=np.int64)
    actions = np.asarray(
        [True] * 10 + [False] * 10 + [True] * 10 + [False] * 10,
        dtype=np.bool_,
    )
    scores = np.linspace(0.0, 1.0, 40)
    positions = audit._review_positions(labels, actions, scores)
    assert len(positions) == 16
    assert len(set(positions)) == 16
    assert sum(labels[positions] == 1) == 8
    assert sum(actions[positions]) == 8


def test_manifest_detects_artifact_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("locked", encoding="utf-8")
    audit._write_manifest(tmp_path)
    assert audit._verify_manifest(tmp_path)["passed"] is True
    payload.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        audit._verify_manifest(tmp_path)


def test_visual_pass_cannot_rescue_automatic_failure(tmp_path: Path) -> None:
    contact = tmp_path / "contact.png"
    contact.write_bytes(b"fixed-contact")
    summary_path = tmp_path / "summary.json"
    audit._write_json(
        summary_path,
        {
            "mode": audit.MODE,
            "contact_sheet": {
                "path": str(contact.resolve()),
                "sha256": audit._sha256(contact),
            },
            "external_replay": {"completed": True, "passed": True},
            "automated_gate_passed": False,
        },
    )
    audit._write_manifest(tmp_path)
    result = audit.finalize_visual_review(
        summary_path,
        result="pass",
        expected_summary_sha256=audit._sha256(summary_path),
    )
    assert result["a0_passed"] is False
    assert result["trainer_integration_authorized"] is False
    assert result["matched_short_smoke_authorized"] is False
    assert result["full_train_authorized"] is False
    updated = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated["visual_review"]["passed"] is True
    assert updated["a0_passed"] is False


def test_locked_argument_override_is_rejected() -> None:
    args = audit.parse_args([])
    args.num_workers = 2
    with pytest.raises(ValueError, match="locks batch-size"):
        audit._validate_locked_args(args)

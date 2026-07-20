from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from trkh.tools import audit_ielt_mhv_signal_a0 as mhv


def test_protocol_and_locked_geometry_are_exact() -> None:
    protocol = (
        mhv.REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_IELT_MHV_SIGNAL_A0_PROTOCOL_20260720.md"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == mhv.LOCKED_PROTOCOL_SHA256
    assert [mhv.scaled_vote_count(value) for value in (256, 218, 167)] == [8, 7, 5]
    assert list(mhv.LAYER_QUOTAS) == [5, 4, 3, 3, 3, 3, 4, 5]
    assert sum(mhv.LAYER_QUOTAS) == 30
    assert len(set(mhv.DEPHASE_OFFSETS)) == mhv.HEAD_COUNT
    assert all(offset != (0, 0) for offset in mhv.DEPHASE_OFFSETS)


@pytest.mark.parametrize("patches", [256, 218, 167])
def test_torch_mhv_matches_independent_numpy_oracle(patches: int) -> None:
    prefix = 7
    generator = torch.Generator().manual_seed(91 + patches)
    attention = torch.rand(
        (3, mhv.HEAD_COUNT, prefix + patches, prefix + patches),
        generator=generator,
    )
    attention = attention / attention.sum(dim=-1, keepdim=True)
    indices = torch.arange(mhv.PATCH_COUNT).reshape(1, -1)[:, :patches]
    indices = indices.expand(3, -1).contiguous()
    attention[:, 0, 0, prefix : prefix + 10] = 0.5
    local = mhv.selector_bank_torch(
        attention, indices, prefix_count=prefix, quota=3
    )
    oracle = mhv.selector_bank_numpy(
        local["cls_attention"].numpy(), indices.numpy(), quota=3
    )
    for role in ("vote_raw", "vote_dephased_smooth", "mhv_smooth"):
        assert np.array_equal(local["scores"][role].numpy(), oracle["scores"][role])
        assert np.array_equal(
            local["selected_original"][role].numpy(),
            oracle["selected_original"][role],
        )
    expected_votes = 3 * mhv.HEAD_COUNT * mhv.scaled_vote_count(patches)
    assert int(local["active_votes"].sum()) == expected_votes


def test_ties_use_ascending_original_patch_index() -> None:
    scores = torch.zeros((1, 8), dtype=torch.float32)
    indices = torch.tensor([[21, 3, 17, 9, 2, 18, 6, 14]], dtype=torch.long)
    order = mhv._stable_descending_local(scores, indices)
    observed = indices.gather(1, order)[0].tolist()
    assert observed == sorted(indices[0].tolist())


def test_dephased_placebo_is_deterministic_and_preserves_vote_mass() -> None:
    prefix = 2
    patches = 256
    generator = torch.Generator().manual_seed(301)
    attention = torch.rand(
        (2, mhv.HEAD_COUNT, prefix + patches, prefix + patches), generator=generator
    )
    indices = torch.arange(patches).reshape(1, -1).expand(2, -1)
    first = mhv.selector_bank_torch(
        attention, indices, prefix_count=prefix, quota=5
    )
    second = mhv.selector_bank_torch(
        attention, indices, prefix_count=prefix, quota=5
    )
    assert torch.equal(
        first["scores"]["vote_dephased_smooth"],
        second["scores"]["vote_dephased_smooth"],
    )
    assert int(first["full_votes"].sum()) == (
        2 * mhv.HEAD_COUNT * mhv.scaled_vote_count(patches)
    )
    assert not torch.equal(
        first["scores"]["mhv_smooth"],
        first["scores"]["vote_dephased_smooth"],
    )


def test_projection_and_block_descriptor_are_locked_and_finite() -> None:
    projection = mhv._projection_matrix(3)
    assert projection.shape == (mhv.FEATURE_DIM, mhv.PROJECTION_DIM)
    assert torch.allclose(
        projection.T @ projection,
        torch.eye(mhv.PROJECTION_DIM),
        atol=2e-6,
        rtol=0.0,
    )
    tokens = torch.randn(4, 20, mhv.FEATURE_DIM)
    selected = torch.tensor([[0, 2, 4], [1, 3, 5], [2, 4, 6], [3, 5, 7]])
    descriptor = mhv.block_role_descriptor(tokens, selected, block_index=3)
    assert descriptor.shape == (4, mhv.PROJECTION_DIM * 2)
    assert torch.isfinite(descriptor).all()


def _synthetic_binary_features():
    rng = np.random.default_rng(20260720)
    rows_per_fold = 40
    folds = np.repeat(np.asarray(mhv.FIT_FOLDS), rows_per_fold)
    labels = np.tile(np.asarray([1] * 28 + [0] * 12), len(mhv.FIT_FOLDS))
    signal = (labels * 2 - 1).reshape(-1, 1)
    features = {}
    for role in mhv.ALL_READOUT_ROLES:
        width = 5 if role == "base_only" else mhv.ROLE_DIM + 5
        values = rng.normal(0.0, 0.5, size=(labels.size, width))
        values[:, :1] += 1.4 * signal
        features[role] = values
    return features, labels, folds


def test_oof_readout_scores_every_row_and_serialized_state_replays() -> None:
    features, labels, folds = _synthetic_binary_features()
    scores, actions, states = mhv.fit_clean_oof_readouts(
        features, labels, folds
    )
    applied_scores, applied_actions = mhv.apply_clean_readout_states(
        features, folds, states
    )
    assert mhv._states_converged(states)
    for role in mhv.ALL_READOUT_ROLES:
        assert np.isfinite(scores[role]).all()
        assert np.max(np.abs(scores[role] - applied_scores[role])) <= 1e-10
        assert np.array_equal(actions[role], applied_actions[role])
        for fold_state in states[role]["folds"]:
            assert fold_state["fit_tp_retention"] >= mhv.MIN_FIT_TP_RETENTION


def test_positive_probability_expands_missing_positive_class() -> None:
    class MissingPositiveModel:
        classes_ = np.asarray([0], dtype=np.int64)

        @staticmethod
        def predict_proba(features: np.ndarray) -> np.ndarray:
            return np.ones((features.shape[0], 1), dtype=np.float64)

    observed = mhv._positive_probability(
        MissingPositiveModel(), np.zeros((6, 3), dtype=np.float64)
    )
    assert np.array_equal(observed, np.zeros(6, dtype=np.float64))


def test_analysis_passes_only_for_candidate_with_fp_control() -> None:
    positives_per_fold = 30
    negatives_per_fold = 20
    labels = []
    targets = []
    folds = []
    for fold in mhv.FIT_FOLDS:
        labels.extend([1] * positives_per_fold + [0] * negatives_per_fold)
        targets.extend(
            [1] * positives_per_fold
            + [0, 2, 4, 0, 2] * (negatives_per_fold // 5)
        )
        folds.extend([fold] * (positives_per_fold + negatives_per_fold))
    labels_array = np.asarray(labels, dtype=np.int64)
    targets_array = np.asarray(targets, dtype=np.int64)
    folds_array = np.asarray(folds, dtype=np.int64)
    rows = labels_array.size
    base_scores = np.linspace(0.25, 0.75, rows)
    candidate_scores = np.where(labels_array == 1, 0.90, 0.10)
    scores = {
        "base_only": base_scores,
        "cls_mean_smooth": base_scores + 0.005,
        "vote_raw": base_scores + 0.010,
        "vote_dephased_smooth": base_scores + 0.015,
        "mhv_smooth": candidate_scores,
    }
    actions = {
        role: np.ones(rows, dtype=np.bool_) for role in mhv.ALL_READOUT_ROLES
    }
    actions["mhv_smooth"] = labels_array.astype(np.bool_)
    stats = {
        role: {
            "effective_rank": 32.0,
            "blocks": [
                {"block": block, "constant": False}
                for block in range(mhv.BLOCK_COUNT)
            ],
        }
        for role in mhv.ROLE_NAMES
    }
    selector = {
        "roles": {
            role: {
                "top_score_tie_fraction": 0.0,
                "alignment_score": 0.4 if role == "mhv_smooth" else 0.2,
            }
            for role in mhv.ROLE_NAMES
        }
    }
    analysis = mhv.build_analysis(
        labels=labels_array,
        targets=targets_array,
        folds=folds_array,
        scores=scores,
        actions=actions,
        descriptor_stats=stats,
        selector_summary=selector,
    )
    assert analysis["mechanism_gates_passed"] is True
    assert all(analysis["mechanism_gates"].values())


class _DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_dropout = nn.Dropout(0.0)


class _DummyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _DummyAttention()

    def forward(self, value):
        return value


class _DummyKeeper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(mhv.BLOCK_COUNT)])
        self.num_prefix_tokens = 7


def test_capture_close_restores_exact_hook_counts() -> None:
    model = _DummyKeeper()
    capture = mhv.FrozenMHVCapture(model)
    lifecycle = capture.close()
    assert lifecycle["no_leak"] is True
    assert lifecycle["before"] == lifecycle["after"]


def test_manifest_detects_payload_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("locked\n", encoding="utf-8")
    mhv._write_manifest(tmp_path)
    assert mhv._verify_manifest(tmp_path)["passed"] is True
    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest payload differs"):
        mhv._verify_manifest(tmp_path)


def test_source_keeps_model_trainer_and_current_best_commands_read_only() -> None:
    source = Path(mhv.__file__).read_text(encoding="utf-8")
    assert "train_model" not in source
    assert "split=\"val\"" not in source
    assert "split=\"test\"" not in source
    assert "current_command_update_authorized\": False" in source

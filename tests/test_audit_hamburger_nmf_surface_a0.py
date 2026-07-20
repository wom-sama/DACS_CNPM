from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools import audit_hamburger_nmf_surface_a0 as audit


def _synthetic_surface(rows: int = 23) -> np.ndarray:
    return np.random.default_rng(41).random(
        (rows, audit.PROJECTION_DIM), dtype=np.float64
    )


def _fold_inputs(rows_per_fold: int = 8):
    rows = len(audit.FOLDS) * rows_per_fold
    folds = np.repeat(np.asarray(audit.FOLDS, dtype=np.int64), rows_per_fold)
    indices = np.arange(1000, 1000 + rows, dtype=np.int64)
    sources = [f"source_{fold}_{position}" for fold, position in zip(folds, indices)]
    margins = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    descriptors = np.random.default_rng(13).normal(
        size=(rows, audit.SURFACE_DESCRIPTOR_DIM)
    )
    return descriptors, folds, margins, sources, indices


def test_fixed_projection_matches_prospective_hash_and_geometry() -> None:
    projection = audit.fixed_projection()
    diagnostics = audit.projection_diagnostics(projection)
    assert diagnostics["passed"] is True
    assert diagnostics["sha256"] == audit.LOCKED_PROJECTION_SHA256
    assert projection.shape == (audit.PROJECTION_INPUT_DIM, audit.PROJECTION_DIM)


def test_project_surface_tokens_is_nonnegative_and_rms_normalized() -> None:
    tokens = np.random.default_rng(17).normal(
        size=(31, audit.PROJECTION_INPUT_DIM)
    ).astype(np.float32)
    projected = audit.project_surface_tokens(tokens, audit.fixed_projection())
    assert projected.shape == (31, audit.PROJECTION_DIM)
    assert np.isfinite(projected).all()
    assert float(projected.min()) >= 0.0
    active = np.sqrt(np.mean(np.square(projected.astype(np.float64)), axis=0)) > 1e-5
    assert np.allclose(
        np.sqrt(np.mean(np.square(projected.astype(np.float64)), axis=0))[active],
        1.0,
        atol=2e-5,
    )


def test_transformed_crop_bbox_prior_is_aligned_and_normalized() -> None:
    crop_bbox = torch.tensor(
        [
            [0.501953125, 0.501953125, 0.59765625, 0.91015625],
            [0.505859375, 0.501953125, 0.91015625, 0.15234375],
        ],
        dtype=torch.float32,
    )
    patch_indices = torch.arange(256, dtype=torch.long).repeat(2, 1)
    prior = audit.build_transformed_crop_bbox_prior(
        crop_bbox, patch_indices, grid_size=(16, 16)
    )
    assert prior.shape == (2, 256)
    assert torch.allclose(prior.amax(dim=1), torch.ones(2))
    assert int((prior[0] >= audit.BBOX_PRIOR_THRESHOLD).sum()) >= 16
    assert int((prior[1] >= audit.BBOX_PRIOR_THRESHOLD).sum()) >= 16


def test_torch_hamburger_nmf_matches_independent_numpy_oracle() -> None:
    surface = _synthetic_surface()
    production = audit.hamburger_nmf_torch(surface, sample_index=321)
    oracle = audit.hamburger_nmf_numpy(surface, sample_index=321)
    for key in ("reconstruction", "basis", "coef", "objectives"):
        assert np.max(np.abs(production[key] - oracle[key])) <= 1e-10


def test_hamburger_nmf_is_nonnegative_and_objective_monotonic() -> None:
    result = audit.hamburger_nmf_torch(_synthetic_surface(), sample_index=99)
    assert audit._objectives_monotonic(result["objectives"])
    assert len(result["objectives"]) == audit.NMF_STEPS + 2
    assert float(result["basis"].min()) >= -1e-12
    assert float(result["coef"].min()) >= -1e-12
    assert float(result["reconstruction"].min()) >= -1e-12


def test_surface_descriptor_is_equal_dimensional_for_all_roles() -> None:
    surface = _synthetic_surface()
    svd = audit.svd_rank_reconstruction(surface)
    nmf = audit.hamburger_nmf_numpy(surface, sample_index=17)["reconstruction"]
    raw_descriptor = audit.surface_descriptor(surface, surface)
    svd_descriptor = audit.surface_descriptor(surface, svd)
    nmf_descriptor = audit.surface_descriptor(surface, nmf)
    assert raw_descriptor.shape == svd_descriptor.shape == nmf_descriptor.shape
    assert raw_descriptor.shape == (audit.SURFACE_DESCRIPTOR_DIM,)
    assert np.allclose(raw_descriptor[128:256], 0.0)
    assert not np.array_equal(raw_descriptor, svd_descriptor)
    assert not np.array_equal(svd_descriptor, nmf_descriptor)


def test_svd_control_has_rank_at_most_eight() -> None:
    reconstruction = audit.svd_rank_reconstruction(_synthetic_surface(37))
    assert np.linalg.matrix_rank(reconstruction, tol=1e-10) <= audit.NMF_RANK


def test_source_derangement_is_fold_local_source_safe_and_deterministic() -> None:
    descriptors, folds, margins, sources, indices = _fold_inputs()
    first, first_assigned, first_mapping = audit.build_source_derangement(
        descriptors,
        folds=folds,
        clean_margins=margins,
        source_stems=sources,
        sample_indices=indices,
    )
    second, second_assigned, second_mapping = audit.build_source_derangement(
        descriptors,
        folds=folds,
        clean_margins=margins,
        source_stems=sources,
        sample_indices=indices,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_assigned, second_assigned)
    assert first_mapping == second_mapping
    assert np.array_equal(folds, folds[first_assigned])
    assert all(sources[row] != sources[source] for row, source in enumerate(first_assigned))


def test_readout_features_have_locked_dimensions() -> None:
    descriptors, folds, margins, sources, indices = _fold_inputs()
    logits = np.stack(
        (margins, -margins, margins * 0.2, margins * -0.3, margins * 0.1), axis=1
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    descriptor_bank = {
        role: descriptors + offset
        for role, offset in (
            ("raw_surface_control", 0.0),
            ("svd_rank8_control", 0.1),
            ("nmf_candidate", 0.2),
            ("nmf_seed_repeat", 0.3),
        )
    }
    features, assigned, mapping = audit.build_readout_features(
        logits=logits,
        probabilities=probabilities,
        descriptors=descriptor_bank,
        folds=folds,
        source_stems=sources,
        sample_indices=indices,
    )
    assert features["base_only"].shape == (len(indices), audit.BASE_DIM)
    assert all(
        features[role].shape == (len(indices), audit.ROLE_DIM)
        for role in audit.ROLE_NAMES
        if role != "base_only"
    )
    assert assigned.shape == (len(indices),)
    assert len(mapping) == len(indices)


def test_oof_readout_state_replay_is_exact() -> None:
    rows_per_fold = 16
    rows = len(audit.FOLDS) * rows_per_fold
    folds = np.repeat(np.asarray(audit.FOLDS, dtype=np.int64), rows_per_fold)
    sources = [f"readout_source_{fold}_{position}" for fold, position in zip(folds, range(rows))]
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), rows // 2)
    generator = np.random.default_rng(71)
    features = {}
    for role in audit.ROLE_NAMES:
        dimensions = audit.BASE_DIM if role == "base_only" else audit.ROLE_DIM
        value = generator.normal(scale=0.2, size=(rows, dimensions))
        value[:, 0] += labels * 1.5
        features[role] = value
    scores, actions, states = audit.fit_clean_oof_readouts(
        features, labels, folds, sources
    )
    applied_scores, applied_actions = audit.apply_readout_states(
        features, folds, states
    )
    assert audit._maximum_role_difference(scores, applied_scores) <= 1e-10
    assert all(
        np.array_equal(actions[role], applied_actions[role])
        for role in audit.ROLE_NAMES
    )


def test_decomposition_bank_uses_only_local_rng() -> None:
    rows = 3
    counts = np.asarray([19, 23, 17], dtype=np.int16)
    tokens = np.zeros(
        (rows, audit.MAX_PATCHES, audit.PROJECTION_DIM), dtype=np.float32
    )
    generator = np.random.default_rng(18)
    for row, count in enumerate(counts):
        tokens[row, : int(count)] = generator.random(
            (int(count), audit.PROJECTION_DIM), dtype=np.float32
        )
    bank = audit.build_decomposition_bank(
        tokens,
        counts,
        np.asarray([10, 20, 30], dtype=np.int64),
        oracle=False,
    )
    assert bank["diagnostics"]["local_rng_only"] is True
    assert bank["diagnostics"]["candidate_objectives_monotonic_all_rows"] is True
    assert bank["diagnostics"]["repeat_objectives_monotonic_all_rows"] is True


def test_manifest_detects_artifact_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "payload.txt"
    artifact.write_text("locked", encoding="utf-8")
    audit._write_manifest(tmp_path)
    assert audit._verify_manifest(tmp_path)["passed"] is True
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        audit._verify_manifest(tmp_path)


def test_contact_sheet_renders_fixed_component_and_residual_views(
    tmp_path: Path,
) -> None:
    sample_indices = np.arange(200, 212, dtype=np.int64)
    targets = [1, 1, 1, 0, 0, 0, 2, 2, 2, 4, 4, 4]
    cohort = [
        audit.CleanTrainRow(
            sample_index=int(sample_index),
            source_stem=f"sheet_source_{sample_index}",
            image_path=tmp_path / f"{sample_index}.jpg",
            fold=int(position % 5),
            target=int(targets[position]),
            keeper_prediction=1,
            keeper_probabilities=(0.1, 0.6, 0.1, 0.1, 0.1),
        )
        for position, sample_index in enumerate(sample_indices)
    ]
    projected = np.zeros(
        (len(cohort), audit.MAX_PATCHES, audit.PROJECTION_DIM), dtype=np.float32
    )
    generator = np.random.default_rng(112)
    projected[:, : audit.MIN_SURFACE_TOKENS] = generator.random(
        (len(cohort), audit.MIN_SURFACE_TOKENS, audit.PROJECTION_DIM),
        dtype=np.float32,
    )
    token_cache = {
        "review_indices": sample_indices,
        "review_rgb": generator.integers(
            0, 256, size=(len(cohort), 256, 256, 3), dtype=np.uint8
        ),
        "counts": np.full(len(cohort), audit.MIN_SURFACE_TOKENS, dtype=np.int16),
        "projected_tokens": projected,
        "patch_indices": np.tile(
            np.concatenate(
                (
                    np.arange(audit.MIN_SURFACE_TOKENS, dtype=np.int16),
                    np.full(
                        audit.MAX_PATCHES - audit.MIN_SURFACE_TOKENS,
                        -1,
                        dtype=np.int16,
                    ),
                )
            ),
            (len(cohort), 1),
        ),
    }
    output = tmp_path / "sheet.png"
    summary = audit.render_contact_sheet(
        output, token_cache=token_cache, cohort=cohort
    )
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert summary["rows"] == 12
    assert len(summary["columns"]) == 5
    assert summary["sha256"] == audit._sha256(output)


def test_visual_pass_cannot_rescue_automated_failure(tmp_path: Path) -> None:
    contact = tmp_path / "contact.png"
    contact.write_bytes(b"fixed-contact")
    summary_path = tmp_path / "summary.json"
    summary = {
        "mode": audit.MODE,
        "contact_sheet": {
            "path": str(contact.resolve()),
            "sha256": audit._sha256(contact),
        },
        "external_replay": {"completed": True, "passed": True},
        "automated_gate_passed": False,
        "visual_review": {"required": True, "completed": False, "passed": False},
    }
    audit._write_json(summary_path, summary)
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
    args.batch_size = audit.BATCH_SIZE // 2
    with pytest.raises(ValueError, match="locks batch-size"):
        audit._validate_locked_args(args)

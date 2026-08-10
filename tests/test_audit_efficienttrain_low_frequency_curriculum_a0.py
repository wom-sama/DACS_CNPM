from __future__ import annotations

from pathlib import Path

import torch

from trkh.tools.audit_efficienttrain_low_frequency_curriculum_a0 import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    NUM_WORKERS,
    SEED,
    _finalize_geometry,
    _fit_thresholds,
    _load_official_freq_crop,
    _masked_rms,
    _new_geometry_accumulator,
    _declaration_mismatch_policy,
    _summarize_known_near_tie_exclusion,
    _summarize_known_near_tie_mechanism_exclusion,
    _summarize_mechanism,
    _summarize_views,
    _update_geometry,
    _write_manifest,
    _write_mechanism,
    _write_predictions,
    assess_information_gate,
    assess_mechanism_gate,
    efficienttrain_freq_crop,
    independent_frequency_crop_oracle,
    independent_same_size_reconstruction,
    parse_args,
    same_size_low_frequency_reconstruction,
)
from trkh.tools.replay_efficienttrain_low_frequency_curriculum_a0 import (
    binary_auroc,
    canonical_close,
    empirical_quantile_higher,
    replay_artifacts,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == NUM_WORKERS == 4
    assert args.seed == SEED == 42
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 3
    assert "EfficientTrain_ICCV2023" in str(args.official_root)
    assert str(args.output_dir).endswith("20260719")
    assert not args.preflight_only
    assert not args.finalize_visual_review


def test_frequency_crop_matches_independent_oracle_and_has_locked_shapes() -> None:
    generator = torch.Generator().manual_seed(11)
    images = torch.randn((2, 3, 32, 32), generator=generator)
    candidate = efficienttrain_freq_crop(images, 16)
    oracle = independent_frequency_crop_oracle(images, 16)
    assert candidate.shape == (2, 3, 16, 16)
    assert torch.equal(candidate, oracle)
    assert torch.isfinite(candidate).all()


def test_b256_identity_is_same_tensor_and_byte_exact() -> None:
    images = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    identity = efficienttrain_freq_crop(images, 16)
    assert identity is images
    assert identity.data_ptr() == images.data_ptr()
    assert torch.equal(identity, images)


def test_same_size_low_frequency_residual_matches_independent_oracle() -> None:
    generator = torch.Generator().manual_seed(13)
    images = torch.randn((1, 3, 32, 32), generator=generator)
    low, residual = same_size_low_frequency_reconstruction(images, 16)
    oracle_low, oracle_residual = independent_same_size_reconstruction(images, 16)
    assert torch.equal(low, oracle_low)
    assert torch.equal(residual, oracle_residual)
    assert torch.allclose(low + residual, images, atol=1e-6, rtol=0.0)


def test_official_function_is_ast_isolated_and_matches_candidate(tmp_path: Path) -> None:
    source = tmp_path / "utils.py"
    source.write_text(
        "import unavailable_dependency\n\n"
        "def freq_crop(input_t, band_width):\n"
        "    img_size = input_t.size(-1)\n"
        "    band_w = band_width // 2\n"
        "    img_f = torch.fft.fft2(input_t)\n"
        "    img_crop = torch.empty([input_t.size(0), input_t.size(1), "
        "band_w * 2, band_w * 2], dtype=img_f.dtype, device=img_f.device)\n"
        "    img_crop[:, :, :band_w, :band_w] = img_f[:, :, :band_w, :band_w]\n"
        "    img_crop[:, :, -band_w:, :band_w] = img_f[:, :, -band_w:, :band_w]\n"
        "    img_crop[:, :, :band_w, -band_w:] = img_f[:, :, :band_w, -band_w:]\n"
        "    img_crop[:, :, -band_w:, -band_w:] = img_f[:, :, -band_w:, -band_w:]\n"
        "    img_crop = img_crop * ((band_w * 2 / img_size) ** 2)\n"
        "    return torch.real(torch.fft.ifft2(img_crop))\n",
        encoding="utf-8",
    )
    official = _load_official_freq_crop(source)
    images = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    assert torch.equal(official(images, 16), efficienttrain_freq_crop(images, 16))


def _clean_threshold_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    position = 0
    fold_counts = {
        1: (112, 45),
        2: (100, 48),
        3: (101, 52),
        4: (108, 41),
    }
    for fold, (tp_count, fp_count) in fold_counts.items():
        tp_score = 10.0 if fold == 1 else float(fold) / 100.0
        for index in range(tp_count):
            rows.append(
                {
                    "condition": "clean",
                    "cohort_position": position,
                    "cohort": "tp",
                    "fold": fold,
                    "early_b176_suppression_score": tp_score + index * 1e-8,
                }
            )
            position += 1
        for index in range(fp_count):
            rows.append(
                {
                    "condition": "clean",
                    "cohort_position": position,
                    "cohort": "fp",
                    "fold": fold,
                    "early_b176_suppression_score": 20.0 + index,
                }
            )
            position += 1
    for canonical_position, row in enumerate(rows):
        row["cohort_position"] = canonical_position
    return rows


def test_fold_threshold_excludes_held_fold_tp_values() -> None:
    thresholds = _fit_thresholds(_clean_threshold_rows(), "early_b176")
    assert thresholds[1] < 1.0
    assert thresholds[2] > 9.0
    assert thresholds[3] > 9.0
    assert thresholds[4] > 9.0


def test_independent_quantile_and_auroc_handle_ties() -> None:
    values = [0.0, 1.0, 2.0, 3.0]
    assert empirical_quantile_higher(values, 0.5) == 2.0
    assert binary_auroc([0, 0, 1, 1], [0.0, 0.5, 0.5, 1.0]) == 0.875


def test_declaration_policy_allows_only_the_locked_known_near_tie() -> None:
    known = {
        "expected_sample_index": 3657,
        "observed_sample_index": 3657,
        "expected_target": 2,
        "observed_target": 2,
        "expected_prediction": 1,
        "observed_prediction": 2,
    }
    assert _declaration_mismatch_policy([known])["policy_passed"]
    unknown = dict(known)
    unknown["observed_sample_index"] = 3658
    assert not _declaration_mismatch_policy([unknown])["policy_passed"]
    assert not _declaration_mismatch_policy([])["policy_passed"]


def _condition_metrics(
    auroc: float = 0.70,
    tp_retention: float = 0.98,
    fp_rejection: float = 0.15,
) -> dict[str, object]:
    return {
        "auroc_fp_vs_tp": auroc,
        "tp_retention": tp_retention,
        "restricted_fp_rejection": fp_rejection,
        "tp_breaks": 8,
        "restricted_fp_removals": 28,
    }


def _passing_view_metrics() -> dict[str, object]:
    early_conditions = {
        "clean": _condition_metrics(),
        "lighting_dim": _condition_metrics(0.66, 0.95, 0.10),
        "lighting_bright": _condition_metrics(0.67, 0.96, 0.11),
        "low_contrast": _condition_metrics(0.68, 0.95, 0.09),
    }
    middle_conditions = {
        name: _condition_metrics(0.64, 0.99, 0.08)
        for name in early_conditions
    }
    return {
        "early_b176": {
            "conditions": early_conditions,
            "positive_clean_fp_rejection_fold_count": 4,
            "hard_decisions_clean": {
                "corrections": 8,
                "harms": 1,
                "class1_tp_breaks": 1,
            },
        },
        "middle_b224": {"conditions": middle_conditions},
    }


def test_information_gate_accepts_selective_precision_signal_and_rejects_tp_loss() -> None:
    passing = _passing_view_metrics()
    assert assess_information_gate(passing)["passed"]
    failing = _passing_view_metrics()
    failing["early_b176"]["conditions"]["lighting_dim"]["tp_retention"] = 0.80
    result = assess_information_gate(failing)
    assert not result["passed"]
    assert "every_shifted_tp_retention_gte_0p93" in result["failed_checks"]


def _passing_mechanism() -> dict[str, object]:
    conditions = {}
    for index, name in enumerate(
        ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    ):
        conditions[name] = {
            "all_regions_nonempty": True,
            "object_auroc_fp_vs_tp": 0.66 + index * 0.01,
            "outside_auroc_fp_vs_tp": 0.55,
            "object_fp_minus_tp_gap": 0.08,
            "outside_fp_minus_tp_gap": 0.01,
        }
    return {"views": {"early_b176": {"conditions": conditions}}}


def test_mechanism_gate_requires_object_specific_signal() -> None:
    passing = _passing_mechanism()
    assert assess_mechanism_gate(passing)["passed"]
    failing = _passing_mechanism()
    failing["views"]["early_b176"]["conditions"]["clean"][
        "outside_auroc_fp_vs_tp"
    ] = 0.65
    result = assess_mechanism_gate(failing)
    assert not result["passed"]
    assert (
        "every_condition_object_auroc_exceeds_outside_by_0p03"
        in result["failed_checks"]
    )


class _DynamicModel:
    num_prefix_tokens = 3
    num_registers = 2
    register_positional_embedding = True

    @staticmethod
    def get_interpolated_pos_embed(grid: tuple[int, int]) -> torch.Tensor:
        return torch.zeros((1, 1 + 2 + grid[0] * grid[1], 8))


def test_dynamic_geometry_tracks_prefix_grid_and_pruning_indices() -> None:
    accumulator = _new_geometry_accumulator()
    features = {
        "grid_size": (11, 11),
        "patches": torch.zeros((2, 80, 8)),
        "tokens": torch.zeros((2, 83, 8)),
        "patch_indices": torch.arange(80).repeat(2, 1),
    }
    _update_geometry(
        accumulator,
        model=_DynamicModel(),
        features=features,
        logits=torch.zeros((2, 5)),
    )
    empty = _new_geometry_accumulator()
    accumulators = {
        "early_b176": accumulator,
        "middle_b224": empty,
        "native_b256": empty,
    }
    # Finalization requires each view to have one observation, so mirror valid
    # metadata with the corresponding immutable grid sizes.
    for view, grid in (("middle_b224", 14), ("native_b256", 16)):
        patch_count = grid * grid
        _update_geometry(
            accumulators[view],
            model=_DynamicModel(),
            features={
                "grid_size": (grid, grid),
                "patches": torch.zeros((1, patch_count, 8)),
                "tokens": torch.zeros((1, patch_count + 3, 8)),
                "patch_indices": torch.arange(patch_count).reshape(1, -1),
            },
            logits=torch.zeros((1, 5)),
        )
    result = _finalize_geometry(accumulators, _DynamicModel())
    assert result["early_b176"]["grid_exact"]
    assert result["early_b176"]["prefix_count_exact"]
    assert result["early_b176"]["indices_in_grid"]


def test_masked_rms_and_manifest_forbid_model_artifacts(tmp_path: Path) -> None:
    activation = torch.ones((2, 3, 4, 4))
    mask = torch.zeros((2, 4, 4), dtype=torch.bool)
    mask[:, :2, :2] = True
    assert torch.equal(_masked_rms(activation, mask), torch.ones(2))
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "forbidden.pt").write_bytes(b"not-a-checkpoint")
    manifest = _write_manifest(tmp_path)
    assert manifest["forbidden_artifact_count"] == 1


def test_full_csv_contract_replays_thresholds_hard_counts_and_mechanism(
    tmp_path: Path,
) -> None:
    identities = _clean_threshold_rows()
    records: list[dict[str, object]] = []
    mechanism_rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(
        ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    ):
        for source in identities:
            cohort = str(source["cohort"])
            position = int(source["cohort_position"])
            fold = int(source["fold"])
            target = 1 if cohort == "tp" else 0
            sample_index = 3657 if position == 337 else position
            native_p1 = 0.80
            early_score = (
                0.010 + position * 1e-7
                if cohort == "tp"
                else 0.100 + position * 1e-7
            )
            middle_score = early_score * 0.75
            row: dict[str, object] = {
                "condition": condition,
                "cohort_position": position,
                "sample_index": sample_index,
                "source_stem": f"sample_{position}",
                "image_path": f"D:/train/sample_{position}.jpg",
                "fold": fold,
                "target": target,
                "cohort": cohort,
            }
            for view, p1, prediction in (
                ("native_b256", native_p1, 1),
                (
                    "early_b176",
                    native_p1 - early_score,
                    target if cohort == "fp" and position % 3 == 0 else 1,
                ),
                ("middle_b224", native_p1 - middle_score, 1),
            ):
                for class_index in range(5):
                    row[f"{view}_logit_{class_index}"] = 0.0
                    row[f"{view}_prob_{class_index}"] = (
                        p1 if class_index == 1 else (1.0 - p1) / 4.0
                    )
                row[f"{view}_prediction"] = prediction
            row["early_b176_suppression_score"] = early_score
            row["middle_b224_suppression_score"] = middle_score
            records.append(row)
            for view, scale in (("early_b176", 1.0), ("middle_b224", 0.8)):
                mechanism_rows.append(
                    {
                        "condition": condition,
                        "view": view,
                        "cohort_position": position,
                        "sample_index": sample_index,
                        "fold": fold,
                        "target": target,
                        "cohort": cohort,
                        "object_residual_rms": scale
                        * (0.20 if cohort == "fp" else 0.10)
                        + condition_index * 1e-4,
                        "outside_residual_rms": scale
                        * (0.11 if cohort == "fp" else 0.10)
                        + condition_index * 1e-4,
                        "object_pixels": 100,
                        "outside_pixels": 200,
                    }
                )
    expected_views = _summarize_views(records)
    expected_mechanism = _summarize_mechanism(mechanism_rows)
    expected_prediction_sensitivity = _summarize_known_near_tie_exclusion(
        records, expected_views
    )
    expected_mechanism_sensitivity = (
        _summarize_known_near_tie_mechanism_exclusion(mechanism_rows)
    )
    predictions_path = tmp_path / "predictions_all_conditions.csv"
    mechanism_path = tmp_path / "frequency_mechanism.csv"
    _write_predictions(predictions_path, records)
    _write_mechanism(mechanism_path, mechanism_rows)
    replay = replay_artifacts(predictions_path, mechanism_path)
    assert canonical_close(expected_views, replay["view_metrics"])
    assert canonical_close(expected_mechanism, replay["mechanism_summary"])
    assert canonical_close(
        expected_prediction_sensitivity,
        replay["known_near_tie_exclusion_sensitivity"]["prediction"],
    )
    assert canonical_close(
        expected_mechanism_sensitivity,
        replay["known_near_tie_exclusion_sensitivity"]["mechanism"],
    )
    assert replay["prediction_rows"] == 4 * 607
    assert replay["mechanism_rows"] == 4 * 2 * 607

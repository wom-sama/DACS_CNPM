from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow
from trkh.tools.audit_spatial_worst_shift_robust_optimization_a0 import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    EXPECTED_PROBE_ROWS,
    PHOTOMETRIC_CONDITIONS,
    TRANSFORMS,
    VARIANTS,
    SpatialBatch,
    _maximum_numeric_difference,
    _metrics_with_focus,
    _process_snapshot,
    _read_prediction_csv,
    _shift_batch,
    _summarize_all,
    _write_prediction_csv,
    assess_probe_signal,
    assess_spatial_a0,
    equation_diagnostics,
    official_worst_indices,
    official_worst_indices_reference,
    parse_args,
    random_shift_indices,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 32
    assert args.num_workers == 4
    assert args.updates == 32
    assert args.learning_rate == pytest.approx(1e-5)
    assert args.weight_decay == pytest.approx(0.05)
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 40
    assert TRANSFORMS == (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )


def test_equation_diagnostics_cover_translation_selector_and_random_replay() -> None:
    result = equation_diagnostics()
    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert all(result["checks"].values())


def test_official_selector_prioritizes_wrong_transform_then_largest_ce() -> None:
    losses = torch.tensor(
        [
            [4.0, 1.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [1.0, 0.5],
            [0.5, 0.4],
            [0.4, 0.3],
            [0.3, 0.2],
            [0.2, 0.1],
            [0.1, 0.0],
        ]
    )
    targets = torch.tensor([1, 2])
    predictions = targets.view(1, -1).repeat(len(TRANSFORMS), 1)
    predictions[2, 0] = 4
    predictions[3, 0] = 0
    observed = official_worst_indices(losses, predictions, targets)
    assert observed.tolist() == [2, 1]
    assert torch.equal(
        observed,
        official_worst_indices_reference(losses, predictions, targets),
    )


def test_random_selector_is_sample_stable_and_batch_order_independent() -> None:
    indices = torch.tensor([101, 202, 303, 404])
    first = random_shift_indices(indices, step=5)
    second = random_shift_indices(indices, step=5)
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = random_shift_indices(indices[permutation], step=5)
    assert torch.equal(first, second)
    assert torch.equal(first[permutation], permuted)
    assert not torch.equal(first, random_shift_indices(indices, step=6))


def test_shift_batch_moves_image_mask_and_bboxes_together() -> None:
    batch = SpatialBatch(
        images=torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3),
        targets=torch.tensor([2]),
        bbox=torch.tensor([[0.5, 0.5, 0.4, 0.6]]),
        crop_bbox=torch.tensor([[0.4, 0.6, 0.3, 0.5]]),
        image_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        sample_indices=torch.tensor([19]),
    )
    shifted = _shift_batch(batch, TRANSFORMS.index((1, -1)))
    assert shifted.images[0, 0, 1].tolist() == [6.0, 6.0, 7.0]
    assert shifted.image_mask[:, :, 0].sum().item() == 0
    assert shifted.image_mask[:, -1, :].sum().item() == 0
    assert shifted.bbox[0, 0].item() == pytest.approx(0.5 + 1.0 / 256.0)
    assert shifted.bbox[0, 1].item() == pytest.approx(0.5 - 1.0 / 256.0)
    assert shifted.crop_bbox[0, 0].item() == pytest.approx(0.4 + 1.0 / 256.0)
    assert torch.equal(shifted.targets, batch.targets)
    assert torch.equal(shifted.sample_indices, batch.sample_indices)


def _passing_probe_rows() -> list[dict[str, object]]:
    rows = []
    for sample_index in range(EXPECTED_PROBE_ROWS):
        target = 1 if sample_index == 0 else 0
        selected = 1 if sample_index == 0 else sample_index % len(TRANSFORMS)
        losses = [0.1] * len(TRANSFORMS)
        losses[selected] = 1.0
        predictions = [target] * len(TRANSFORMS)
        predictions[selected] = (target + 1) % 5
        if sample_index == 1:
            predictions[0] = 1
        row: dict[str, object] = {
            "official_index": selected,
            "target": target,
            "target_loss_span": 0.9,
        }
        for index in range(len(TRANSFORMS)):
            row[f"loss_{index}"] = losses[index]
            row[f"pred_{index}"] = predictions[index]
        rows.append(row)
    return rows


def test_probe_gate_is_conjunctive_and_requires_replay() -> None:
    rows = _passing_probe_rows()
    result = assess_probe_signal(rows)
    assert result["passed"] is True
    failed = assess_probe_signal(rows, output_replay_exact=False)
    assert failed["passed"] is False
    assert failed["checks"]["output_replay_exact"] is False


def _metric(
    *,
    macro: float,
    f1: float,
    precision: float,
    tp: int,
    restricted_fp: int,
) -> dict[str, object]:
    return {
        "macro_f1": macro,
        "predicted_support": [100, 100, 100, 100, 100],
        "class1": {
            "f1": f1,
            "precision": precision,
            "recall": 0.9,
            "tp": tp,
            "fp": restricted_fp,
            "fn": 5,
            "restricted_fp": restricted_fp,
        },
    }


def _passing_assessment_payload() -> tuple[dict, dict]:
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    evidence: dict[str, dict[str, dict[str, object]]] = {}
    for photo in PHOTOMETRIC_CONDITIONS:
        summaries[photo] = {
            "keeper": {
                "clean": _metric(
                    macro=0.80, f1=0.80, precision=0.80, tp=100, restricted_fp=10
                ),
                "official_worst": _metric(
                    macro=0.75, f1=0.75, precision=0.75, tp=95, restricted_fp=14
                ),
                "shift_consistency": {"all_nine_same_rate": 0.80},
            },
            "random_control": {
                "clean": _metric(
                    macro=0.801, f1=0.801, precision=0.801, tp=100, restricted_fp=9
                ),
                "official_worst": _metric(
                    macro=0.751, f1=0.751, precision=0.751, tp=96, restricted_fp=13
                ),
                "shift_consistency": {"all_nine_same_rate": 0.81},
            },
            "worst_shift": {
                "clean": _metric(
                    macro=0.82, f1=0.83, precision=0.84, tp=101, restricted_fp=5
                ),
                "official_worst": _metric(
                    macro=0.78, f1=0.79, precision=0.80, tp=97, restricted_fp=8
                ),
                "shift_consistency": {"all_nine_same_rate": 0.90},
            },
        }
        pair = {
            "transitions": {"candidate_correction": 10, "candidate_harm": 2},
            "restricted_fp_corrected_to_target": 5,
            "broken_control_class1_tp": 1,
        }
        evidence[photo] = {
            mode: {
                "keeper_vs_candidate": pair,
                "random_vs_candidate": pair,
            }
            for mode in ("clean", "official_worst")
        }
    return summaries, evidence


def test_spatial_assessment_requires_structural_clean_worst_and_lighting_gates() -> None:
    summaries, evidence = _passing_assessment_payload()
    result = assess_spatial_a0(
        structural_checks={"structure": True},
        summaries=summaries,
        evidence=evidence,
    )
    assert result["all_nonvisual_gates_passed"] is True
    summaries["clean"]["worst_shift"]["clean"]["class1"]["precision"] = 0.80
    failed = assess_spatial_a0(
        structural_checks={"structure": True},
        summaries=summaries,
        evidence=evidence,
    )
    assert failed["all_nonvisual_gates_passed"] is False
    assert any("class1_precision" in name for name in failed["failed_checks"])


def _tiny_evaluations() -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    output: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for photo in PHOTOMETRIC_CONDITIONS:
        output[photo] = {}
        for variant in VARIANTS:
            targets = torch.tensor([0, 1])
            predictions = targets.view(1, -1).repeat(len(TRANSFORMS), 1)
            probabilities = torch.zeros(len(TRANSFORMS), 2, 5)
            probabilities[:, 0, 0] = 1.0
            probabilities[:, 1, 1] = 1.0
            losses = torch.arange(len(TRANSFORMS), dtype=torch.float32).view(-1, 1)
            losses = losses.repeat(1, 2) / 10.0
            output[photo][variant] = {
                "targets": targets,
                "sample_indices": torch.tensor([0, 1]),
                "predictions": predictions,
                "probabilities": probabilities,
                "losses": losses,
                "official_indices": torch.tensor([8, 8]),
                "mean_predictions": targets.clone(),
            }
    return output


def test_prediction_csv_replay_recomputes_indices_and_metrics(tmp_path: Path) -> None:
    evaluations = _tiny_evaluations()
    holdout = [
        CleanTrainRow(
            sample_index=index,
            source_stem=f"source_{index}",
            image_path=Path(f"train/{index}.jpg"),
            fold=0,
            target=index,
            keeper_prediction=index,
            keeper_probabilities=tuple(
                1.0 if class_index == index else 0.0 for class_index in range(5)
            ),
        )
        for index in range(2)
    ]
    path = tmp_path / "predictions.csv"
    written = _write_prediction_csv(path, evaluations=evaluations, holdout=holdout)
    replay, audit = _read_prediction_csv(path)
    assert written == len(PHOTOMETRIC_CONDITIONS) * len(VARIANTS) * len(TRANSFORMS) * 2
    assert audit["official_indices_recomputed_exact"] is True
    assert audit["mean_predictions_recomputed_exact"] is True
    assert _maximum_numeric_difference(
        _summarize_all(evaluations), _summarize_all(replay)
    ) == 0.0


def test_focus_metrics_count_restricted_false_positives() -> None:
    result = _metrics_with_focus(
        np.asarray([0, 1, 1, 2, 3, 4]),
        np.asarray([1, 1, 0, 1, 3, 4]),
    )
    assert result["class1"] == {
        "precision": pytest.approx(1.0 / 3.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.4),
        "tp": 1,
        "fp": 2,
        "fn": 1,
        "restricted_fp": 2,
    }


def test_process_snapshot_owns_current_python_process() -> None:
    snapshot = _process_snapshot()
    current = [
        row for row in snapshot["processes"] if row["pid"] == snapshot["current_pid"]
    ]
    assert len(current) == 1
    assert current[0]["current_auditor"] is True
    assert current[0]["current_auditor_owned"] is True

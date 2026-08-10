from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

import trkh.tools.audit_c4_rotation_consensus_a0 as c4


def _synthetic_orbit() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = []
    folds = []
    probabilities = []

    def add(target: int, fold: int, views: list[list[float]]) -> None:
        targets.append(target)
        folds.append(fold)
        probabilities.append(views)

    for fold in range(5):
        for _ in range(8):
            add(1, fold, [[0.04, 0.80, 0.04, 0.06, 0.06]] * 4)
        for _ in range(2):
            add(
                1,
                fold,
                [
                    [0.55, 0.40, 0.01, 0.02, 0.02],
                    [0.10, 0.80, 0.02, 0.04, 0.04],
                    [0.10, 0.80, 0.02, 0.04, 0.04],
                    [0.10, 0.80, 0.02, 0.04, 0.04],
                ],
            )
        for target in (0, 2):
            add(
                target,
                fold,
                [
                    [0.40 if target == 0 else 0.01, 0.55, 0.40 if target == 2 else 0.01, 0.02, 0.02],
                    [0.82 if target == 0 else 0.01, 0.10, 0.82 if target == 2 else 0.01, 0.035, 0.035],
                    [0.82 if target == 0 else 0.01, 0.10, 0.82 if target == 2 else 0.01, 0.035, 0.035],
                    [0.82 if target == 0 else 0.01, 0.10, 0.82 if target == 2 else 0.01, 0.035, 0.035],
                ],
            )
        for target in (0, 2, 3, 4):
            stable = np.full(5, 0.025, dtype=np.float64)
            stable[target] = 0.90
            for _ in range(4):
                add(target, fold, [stable.tolist()] * 4)
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


def test_rotate_appearance_rotates_image_and_mask_but_freezes_bbox() -> None:
    images = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    mask = torch.arange(12).reshape(1, 3, 4) > 4
    bbox = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    rotated, metadata = c4.rotate_appearance_batch(
        images, {"image_mask": mask, "bbox": bbox}, k=1
    )
    assert torch.equal(rotated, torch.rot90(images, k=1, dims=(-2, -1)))
    assert torch.equal(metadata["image_mask"], torch.rot90(mask, k=1, dims=(-2, -1)))
    assert metadata["bbox"] is bbox


def test_c4_statistics_matches_probability_mean_and_zero_identity_js() -> None:
    identity = np.asarray([[[0.7, 0.3]] * 4], dtype=np.float64)
    stats = c4.c4_statistics(identity)
    np.testing.assert_allclose(stats["consensus_probabilities"], [[0.7, 0.3]])
    assert float(stats["mean_js_divergence"][0]) == pytest.approx(0.0, abs=1e-12)
    assert int(stats["top1_agreement_count"][0]) == 4
    assert bool(stats["top1_all_agree"][0]) is True


def test_c4_statistics_rejects_non_normalized_probabilities() -> None:
    with pytest.raises(ValueError, match="not normalized"):
        c4.c4_statistics(np.ones((2, 4, 3), dtype=np.float64))


def test_analysis_passes_only_for_selective_fp_removal() -> None:
    targets, folds, probabilities = _synthetic_orbit()
    analysis = c4.analyze_predictions(
        targets, folds, probabilities, peak_cuda_bytes=1024
    )
    assert analysis["mechanism_gates_passed"] is True
    assert analysis["directional"]["restricted_fp_net_removal"] == 10
    assert analysis["directional"]["candidate_harm"] == 0
    assert analysis["direction"]["delta_p1_auroc_fp_vs_tp"] == pytest.approx(1.0)
    assert all(analysis["mechanism_gates"].values())


def test_analysis_rejects_broad_class1_contraction() -> None:
    targets, folds, probabilities = _synthetic_orbit()
    baseline_predictions = probabilities[:, 0].argmax(axis=1)
    true_positive = (targets == 1) & (baseline_predictions == 1)
    probabilities[true_positive, 1:, :] = np.asarray([0.90, 0.025, 0.025, 0.025, 0.025])
    analysis = c4.analyze_predictions(
        targets, folds, probabilities, peak_cuda_bytes=1024
    )
    assert analysis["mechanism_gates_passed"] is False
    assert analysis["mechanism_gates"]["class1_recall_delta_ge_minus0005"] is False
    assert analysis["mechanism_gates"]["focus_tp_break_not_above_rescue"] is False


def test_read_cidt_clean_enforces_order_and_probabilities(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(c4, "EXPECTED_TRAIN_ROWS", 2)
    path = tmp_path / "cidt.csv"
    fieldnames = [
        "condition",
        "sample_index",
        "target_index",
        "keeper_prediction",
        "fold",
        "image_path",
        "source_stem",
        "keeper_prob_0",
        "keeper_prob_1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "condition": "clean",
                "sample_index": 1,
                "target_index": 1,
                "keeper_prediction": 1,
                "fold": 1,
                "image_path": "b.jpg",
                "source_stem": "b",
                "keeper_prob_0": 0.2,
                "keeper_prob_1": 0.8,
            }
        )
        writer.writerow(
            {
                "condition": "clean",
                "sample_index": 0,
                "target_index": 0,
                "keeper_prediction": 0,
                "fold": 0,
                "image_path": "a.jpg",
                "source_stem": "a",
                "keeper_prob_0": 0.9,
                "keeper_prob_1": 0.1,
            }
        )
        writer.writerow(
            {
                "condition": "dim",
                "sample_index": 0,
                "target_index": 0,
                "keeper_prediction": 0,
                "fold": 0,
                "image_path": "a.jpg",
                "source_stem": "a",
                "keeper_prob_0": 0.8,
                "keeper_prob_1": 0.2,
            }
        )
    loaded = c4._read_cidt_clean(path, num_classes=2)
    assert loaded["indices"].tolist() == [0, 1]
    assert loaded["targets"].tolist() == [0, 1]
    np.testing.assert_allclose(
        loaded["probabilities"], [[0.9, 0.1], [0.2, 0.8]]
    )


def test_output_directory_must_be_empty(tmp_path) -> None:
    output = tmp_path / "audit"
    assert c4._require_empty_output(output) == output.resolve()
    (output / "payload.txt").write_text("locked", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        c4._require_empty_output(output)


def test_manifest_and_replay_recompute_analysis(tmp_path) -> None:
    targets, folds, probabilities = _synthetic_orbit()
    class_names = [f"class_{index}" for index in range(5)]
    paths = [tmp_path / f"image_{index}.jpg" for index in range(targets.size)]
    sources = [path.stem for path in paths]
    rows = c4._prediction_rows(
        targets=targets,
        folds=folds,
        paths=paths,
        source_stems=sources,
        angle_probabilities=probabilities,
        class_names=class_names,
    )
    c4._write_csv(tmp_path / "predictions.csv", rows)
    analysis = c4.analyze_predictions(
        targets, folds, probabilities, peak_cuda_bytes=1024
    )
    c4._write_json(
        tmp_path / "summary.json",
        {
            "class_names": class_names,
            "runtime": {"peak_cuda_bytes": 1024},
            "analysis": analysis,
        },
    )
    c4._write_manifest(tmp_path)
    replay = c4.replay_summary(tmp_path / "summary.json")
    assert replay["replay_passed"] is True
    assert replay["rows"] == targets.size
    assert replay["maximum_numeric_difference"] <= 1e-12
    assert replay["payload_count"] == 2


def test_manifest_detects_payload_mutation(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"value": 1}), encoding="utf-8")
    c4._write_manifest(tmp_path)
    payload.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        c4._verify_manifest(tmp_path)

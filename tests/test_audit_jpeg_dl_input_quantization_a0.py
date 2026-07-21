from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools import audit_jpeg_dl_input_quantization_a0 as audit


def _probabilities(prediction: int, confidence: float = 0.8) -> list[float]:
    values = np.full(audit.NUM_CLASSES, (1.0 - confidence) / (audit.NUM_CLASSES - 1))
    values[prediction] = confidence
    return values.tolist()


def _prediction_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_index = 0
    for fold in (0, 1):
        for target in range(audit.NUM_CLASSES):
            raw_prediction = audit.FOCUS_CLASS if target in (0, 1) else target
            candidate_prediction = target
            record: dict[str, object] = {
                "condition": "clean",
                "sample_index": sample_index,
                "source_stem": f"source_{fold}_{target}",
                "fold": fold,
                "target": target,
                "psnr": 36.0,
                "mean_absolute_rgb_change": 0.01,
                "max_abs_mean_channel_shift": 0.005,
                "out_of_range_fraction": 0.0,
                "minimum_channel_variance": 0.02,
                "candidate_constant": False,
            }
            for role in audit.EVALUATED_ROLES:
                prediction = (
                    candidate_prediction
                    if role == audit.CANDIDATE_ROLE
                    else raw_prediction
                )
                for class_index, value in enumerate(_probabilities(prediction)):
                    record[f"prob_{role}_{class_index}"] = value
            records.append(record)
            sample_index += 1
    return records


def _q_training(candidate_q: np.ndarray) -> dict[str, object]:
    scalar = torch.tensor([[[0.95]], [[1.05]]], dtype=torch.float32)
    return {
        "states": {
            audit.PAPER_ROLE: {"q": torch.from_numpy(candidate_q.copy()).float()},
            audit.SCALAR_ROLE: {"q": scalar},
            audit.CANDIDATE_ROLE: {"q": torch.from_numpy(candidate_q.copy()).float()},
        }
    }


def test_parse_defaults_and_locked_arguments() -> None:
    args = audit.parse_args([])
    audit._validate_locked_args(args)
    assert args.batch_size == audit.BATCH_SIZE
    assert args.num_workers == audit.NUM_WORKERS
    assert args.seed == audit.SEED
    args.num_workers += 1
    with pytest.raises(ValueError, match="differ from prospective lock"):
        audit._validate_locked_args(args)


def test_layer_shapes_dtype_roundtrip_and_parameter_counts() -> None:
    full = audit.JpegDlInputLayer(mean=(0.4, 0.5, 0.6), std=(0.2, 0.3, 0.4)).double()
    scalar = audit.JpegDlInputLayer(
        mean=(0.4, 0.5, 0.6),
        std=(0.2, 0.3, 0.4),
        scalar_tables=True,
    )
    assert full.trainable_parameter_count == 128
    assert scalar.trainable_parameter_count == 2
    images = torch.linspace(-1.0, 1.0, 2 * 3 * 17 * 23, dtype=torch.float64).reshape(
        2, 3, 17, 23
    )
    bypass = full.bypass(images)
    assert bypass.dtype == torch.float64
    torch.testing.assert_close(bypass, images, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(
        full.bypass(images, phase_shifted=True), images, atol=1e-10, rtol=0.0
    )


def test_independent_numeric_and_gradient_oracles() -> None:
    result = audit.engineering_checks()
    assert result["passed"]
    assert max(result["errors"].values()) <= 5e-4
    assert result["checks"]["global_rng_unchanged"]


def test_frequency_permutation_is_exact_active_same_weight_placebo() -> None:
    layer = audit.JpegDlInputLayer(mean=(0.5,) * 3, std=(0.25,) * 3).eval()
    with torch.no_grad():
        values = torch.linspace(0.55, 1.75, 128).reshape(2, 8, 8)
        layer.q.copy_(values)
    aligned = layer.q_tables()
    permuted = layer.q_tables(frequency_permuted=True)
    assert torch.equal(aligned[:, 0, 0], permuted[:, 0, 0])
    assert torch.equal(aligned.flatten(1).sort(dim=1).values, permuted.flatten(1).sort(dim=1).values)
    assert not torch.equal(aligned, permuted)
    image = torch.linspace(-1.2, 1.2, 3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.inference_mode():
        direct = layer(image)
        placebo = layer(image, frequency_permuted=True)
    assert not torch.equal(direct, placebo)


def test_precision_loss_selects_restricted_fp_and_protected_tp() -> None:
    logits = torch.tensor(
        [
            [0.1, 1.4, 0.2, 0.0, -0.1],
            [0.0, 1.2, 0.1, -0.2, -0.3],
            [0.2, 0.1, 1.0, 0.0, -0.1],
            [0.0, 0.4, 0.1, 1.1, -0.2],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2, 3])
    keeper_predictions = torch.tensor([1, 1, 2, 1])
    loss, components = audit.precision_loss(logits, targets, keeper_predictions)
    loss.backward()
    assert int(components["restricted_rows"]) == 1
    assert int(components["protected_rows"]) == 1
    assert float(components["restricted_fp"]) > 0.0
    assert float(components["protected_tp"]) > 0.0
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_epoch_orders_are_deterministic_complete_and_distinct() -> None:
    positions = np.asarray([1, 3, 8, 9, 12, 19, 27], dtype=np.int64)
    first, first_hash = audit._epoch_orders(positions, fold=2)
    second, second_hash = audit._epoch_orders(positions, fold=2)
    assert first_hash == second_hash
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(np.array_equal(np.sort(order), positions) for order in first)
    assert len({order.tobytes() for order in first}) == audit.EPOCHS


def test_reconstruction_summary_accepts_healthy_and_rejects_collapse() -> None:
    healthy = _prediction_records()
    result = audit.reconstruction_summary(healthy)
    assert result["passed"]
    collapsed = [dict(row) for row in healthy]
    collapsed[0]["candidate_constant"] = True
    collapsed[0]["minimum_channel_variance"] = 0.0
    assert not audit.reconstruction_summary(collapsed)["passed"]


def test_q_diagnostics_detects_active_and_collapsed_tables() -> None:
    pattern = np.linspace(-0.08, 0.08, 128, dtype=np.float32).reshape(2, 8, 8)
    active = audit.q_diagnostics({0: _q_training(1.0 + pattern)})
    assert active["passed"]
    collapsed = audit.q_diagnostics({0: _q_training(np.ones((2, 8, 8), dtype=np.float32))})
    assert not collapsed["passed"]
    assert not collapsed["checks"]["all_candidate_tables_active_noncollapsed"]


def test_prediction_csv_roundtrip_preserves_authoritative_analysis(tmp_path: Path) -> None:
    records = _prediction_records()
    expected = audit.analyze_records(records)
    path = tmp_path / "predictions.csv"
    audit._write_csv(path, records)
    observed_records = audit._read_prediction_csv(path)
    observed = audit.analyze_records(observed_records)
    assert audit._canonical_sha256(observed) == audit._canonical_sha256(expected)
    candidate = observed["conditions"]["clean"]["roles"][audit.CANDIDATE_ROLE]
    keeper = observed["conditions"]["clean"]["roles"][audit.KEEPER_ROLE]
    assert candidate["fp_removed"] == 2
    assert candidate["class1_precision"] > keeper["class1_precision"]


def test_fixed_xai_rows_select_tp_fn_fp_per_executed_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[audit.CleanTrainRow] = []
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
                    keeper_probabilities=(0.1, 0.6, 0.1, 0.1, 0.1),
                )
            )
            sample_index += 1
    monkeypatch.setattr(audit, "locked_cohort", lambda values: list(values))
    selected = audit.fixed_xai_rows(rows, audit.FOLDS)
    assert [row.sample_index for row in selected] == list(range(15))


def test_fold_state_and_manifest_hash_verification(tmp_path: Path) -> None:
    q = np.linspace(0.9, 1.1, 128, dtype=np.float32).reshape(2, 8, 8)
    states = _q_training(q)["states"]
    hashes = {
        role: audit._state_mapping_sha256(state)
        for role, state in states.items()
    }
    training = {
        "fold": 0,
        "states": states,
        "optimizer_state": {"step": 2},
        "occurrence_sha256": "0" * 64,
        "final_hashes": hashes,
        "all_gradient_roles_live": True,
    }
    public, artifact = audit._persist_fold_training(tmp_path, training)
    assert "states" not in public
    assert Path(artifact["path"]).is_file()
    loaded = audit._load_persisted_fold(Path(artifact["path"]))
    assert loaded["final_hashes"] == hashes
    manifest = audit._write_manifest(tmp_path)
    assert manifest.is_file()
    assert audit._verify_manifest(tmp_path)["passed"]
    Path(artifact["path"]).write_bytes(b"tampered")
    verification = audit._verify_manifest(tmp_path)
    assert not verification["passed"]
    assert Path(artifact["path"]).name in verification["mismatched"]

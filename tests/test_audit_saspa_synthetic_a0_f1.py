from __future__ import annotations

import inspect
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from trkh.tools import audit_saspa_synthetic_a0_f1 as audit


def test_f1_implementation_and_f0_hashes_are_locked() -> None:
    assert (
        audit._sha256(audit.IMPLEMENTATION_LOCK_PATH)
        == audit.LOCKED_IMPLEMENTATION_JSON_SHA256
    )
    assert (
        audit._sha256(audit.IMPLEMENTATION_LOCK_MD_PATH)
        == audit.LOCKED_IMPLEMENTATION_MD_SHA256
    )
    assert (
        audit._sha256(audit.IMPLEMENTATION_LOCK_SHA_PATH)
        == audit.LOCKED_IMPLEMENTATION_SHA_FILE_SHA256
    )
    assert (
        audit._sha256(audit.F0_ROOT / "summary.json")
        == audit.LOCKED_F0_SUMMARY_SHA256
    )
    assert (
        audit._sha256(audit.F0_ROOT / "source_selection.json")
        == audit.LOCKED_SOURCE_SELECTION_SHA256
    )
    assert (
        audit._sha256(audit.NO_OUTPUT_RETRY_LOCK_PATH)
        == audit.LOCKED_NO_OUTPUT_RETRY_SHA256
    )
    assert (
        audit._sha256(audit.FAILED_NO_OUTPUT_DIR / "failure.json")
        == audit.LOCKED_NO_OUTPUT_FAILURE_SHA256
    )
    assert audit.DEFAULT_OUTPUT_DIR.name == "f1_tiny_output_v2"


def _official_hwc3_fixture(value: np.ndarray) -> np.ndarray:
    assert value.dtype == np.uint8
    if value.ndim == 2:
        value = value[:, :, None]
    height, width, channels = value.shape
    assert height > 0 and width > 0
    assert channels in (1, 3, 4)
    if channels == 3:
        return value
    if channels == 1:
        return np.concatenate([value, value, value], axis=2)
    color = value[:, :, :3].astype(np.float32)
    alpha = value[:, :, 3:4].astype(np.float32) / 255.0
    return (color * alpha + 255.0 * (1.0 - alpha)).clip(
        0, 255
    ).astype(np.uint8)


def _official_canny_fixture(value: np.ndarray) -> np.ndarray:
    image = _official_hwc3_fixture(value)
    height, width, unused_channels = image.shape
    height = float(height)
    width = float(width)
    scale = 512.0 / min(height, width)
    height *= scale
    width *= scale
    if height * width > 1_200_000:
        scale = math.sqrt(1_200_000 / (height * width))
        height *= scale
        width *= scale
    height = int(np.round(height / 64.0)) * 64
    width = int(np.round(width / 64.0)) * 64
    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA,
    )
    canny = cv2.Canny(resized, 120, 200)
    return np.concatenate([canny[:, :, None]] * 3, axis=2)


def test_canny_transcription_is_byte_identical_on_locked_fixtures() -> None:
    rng = np.random.default_rng(20260724)
    fixtures = [
        rng.integers(0, 256, size=(73, 121, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(1500, 900), dtype=np.uint8),
        rng.integers(0, 256, size=(95, 67, 4), dtype=np.uint8),
    ]
    for fixture in fixtures:
        expected = _official_canny_fixture(fixture)
        actual = audit.make_locked_canny(fixture)
        assert actual.shape == expected.shape
        assert actual.dtype == np.uint8
        assert actual.tobytes() == expected.tobytes()


def test_fidelity_identity_bbox_and_padding_normalization() -> None:
    rng = np.random.default_rng(41)
    image = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    metrics = audit.fidelity_statistics(image, image.copy())
    assert metrics == {
        "lab_histogram_js": 0.0,
        "hsv_hue_circular_w1": 0.0,
        "lbp_texture_chi_square": 0.0,
        "edge_density_ratio": 1.0,
    }

    grid = np.zeros((10, 20, 3), dtype=np.uint8)
    crop = audit._bbox_crop(grid, [0.5, 0.5, 0.31, 0.41])
    assert crop.shape == (6, 8, 3)

    padded = np.full((100, 120, 3), 128, dtype=np.uint8)
    padded[10:90, 15:105] = (20, 140, 60)
    trimmed = audit._trim_subject_padding(padded)
    assert trimmed.shape == (80, 90, 3)
    assert np.all(trimmed == np.array([20, 140, 60], dtype=np.uint8))


def test_final_decode_hook_observes_decoder_output() -> None:
    class FakeVae:
        def __init__(self) -> None:
            self.decoder = torch.nn.Identity()

    class FakePipe:
        def __init__(self) -> None:
            self.vae = FakeVae()

    pipe = FakePipe()
    state, handle, api = audit._install_final_decode_finiteness_hook(
        pipe,
        torch,
    )
    assert api == "pipeline.vae.decoder.register_forward_hook"
    pipe.vae.decoder(torch.ones(1))
    assert state["detected"] is False
    pipe.vae.decoder(torch.tensor([float("nan")]))
    assert state["detected"] is True
    handle.remove()


def test_generation_resource_gates_are_conjunctive() -> None:
    protocol = audit._read_json(audit.PROTOCOL_PATH)
    limits = protocol["phase_f1_tiny_output"]["resource_gates"]
    gib = 1024 ** 3
    healthy = audit._generation_resource_checks(
        limits,
        generation_seconds=100.0,
        output_index=0,
        torch_peak_allocated_bytes=5 * gib,
        torch_peak_reserved_bytes=6 * gib,
        telemetry_summary={
            "max_nvml_used_mib": 7000,
            "max_virtual_used_fraction": 0.5,
            "external_python_or_trtexec_pids": [],
        },
        nonfinite_detected=False,
    )
    assert healthy["passed"] is True
    rejected = audit._generation_resource_checks(
        limits,
        generation_seconds=200.0,
        output_index=0,
        torch_peak_allocated_bytes=7 * gib,
        torch_peak_reserved_bytes=8 * gib,
        telemetry_summary={
            "max_nvml_used_mib": 8000,
            "max_virtual_used_fraction": 0.95,
            "external_python_or_trtexec_pids": [123],
        },
        nonfinite_detected=True,
    )
    assert rejected["passed"] is False
    assert all(value is False for value in rejected["checks"].values())


def test_generation_worker_contract_is_train_only_and_named() -> None:
    source = inspect.getsource(audit._generation_worker)
    assert 'torch.Generator(device="cpu")' in source
    assert "condtioning_image=" in source
    assert "reference_image=" in source
    assert "num_inference_steps=" in source
    assert "pipeline_invoked = True" in source
    assert "synthetic_pixels_generated = True" in source
    assert "_install_final_decode_finiteness_hook" in source
    assert 'immutable["yolo_train_images"]' in source
    assert 'immutable["classification_train_root"]' in source
    assert "yolo_val" not in source
    assert "yolo_test" not in source
    assert '.to("cuda")' not in source


def test_parent_detects_output_saved_before_worker_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(command, **unused_kwargs):
        worker_input = Path(command[command.index("--worker-input") + 1])
        worker_result = Path(command[command.index("--worker-result") + 1])
        payload = audit._read_json(worker_input)
        attempt_dir = Path(payload["attempt_dir"])
        output_dir = attempt_dir / "outputs"
        output_dir.mkdir(parents=True)
        (output_dir / "partial.png").write_bytes(b"partial")
        audit._write_json(
            worker_result,
            {
                "schema": audit.WORKER_SCHEMA,
                "passed": False,
                "saved_output_count": 0,
                "pipeline_invoked": True,
                "synthetic_pixels_generated": True,
            },
        )
        return subprocess.CompletedProcess(command, 1, "", "failed")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    result = audit._run_generation_attempt(
        protocol={"protocol_id": "fixture"},
        rows=[],
        snapshot_path=tmp_path / "snapshot",
        output_dir=tmp_path,
        offload_mode="enable_model_cpu_offload",
        timeout_seconds=10.0,
    )
    assert result["passed"] is False
    assert result["reported_saved_output_count"] == 0
    assert result["observed_saved_output_count"] == 1
    assert result["saved_output_count"] == 1
    assert result["synthetic_pixels_generated"] is True


def test_blind_mapping_is_deterministic_but_not_written_early() -> None:
    protocol = {"protocol_id": "fixture"}
    selected = [{"output_id": "output-{}".format(index)} for index in range(10)]
    provenance = [
        {
            "output_id": row["output_id"],
            "output_sha256": "{:064x}".format(index),
        }
        for index, row in enumerate(selected)
    ]
    first = audit._locked_blind_mapping(protocol, selected, provenance)
    second = audit._locked_blind_mapping(protocol, selected, provenance)
    assert first == second
    assert [row["blind_id"] for row in first] == list("ABCDEFGHIJ")
    assert {row["output_id"] for row in first} == {
        row["output_id"] for row in selected
    }
    contact_source = inspect.getsource(audit._build_contact_sheets)
    assert "blind_review_mapping.sealed.json" not in contact_source
    assert '"target_labels_exposed": False' in contact_source


def test_model_snapshot_verification_is_hash_and_path_locked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = snapshot / "a.bin"
    second = snapshot / "nested" / "b.json"
    second.parent.mkdir()
    first.write_bytes(b"weights")
    second.write_bytes(b"config")
    rows = [
        {
            "path": "a.bin",
            "size_bytes": first.stat().st_size,
            "sha256": audit._sha256(first),
        },
        {
            "path": "nested/b.json",
            "size_bytes": second.stat().st_size,
            "sha256": audit._sha256(second),
        },
    ]
    total = sum(int(row["size_bytes"]) for row in rows)
    monkeypatch.setattr(audit.f0, "LOCKED_MODEL_REMOTE_BYTES", total)
    result = audit.verify_model_snapshot(
        {
            "snapshot_path": str(snapshot),
            "files": rows,
            "file_manifest_sha256": audit._canonical_sha256(rows),
            "total_bytes": total,
        }
    )
    assert result["passed"] is True
    assert result["file_count"] == 2
    assert result["total_bytes"] == total
    assert result["files_sha256"] == audit._canonical_sha256(rows)


def test_model_snapshot_rejects_parent_traversal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    rows = [
        {
            "path": "../outside.bin",
            "size_bytes": outside.stat().st_size,
            "sha256": audit._sha256(outside),
        }
    ]
    monkeypatch.setattr(
        audit.f0,
        "LOCKED_MODEL_REMOTE_BYTES",
        outside.stat().st_size,
    )
    try:
        audit.verify_model_snapshot(
            {
                "snapshot_path": str(snapshot),
                "files": rows,
                "file_manifest_sha256": audit._canonical_sha256(rows),
                "total_bytes": outside.stat().st_size,
            }
        )
    except RuntimeError as exc:
        assert "manifest path is unsafe" in str(exc)
    else:
        raise AssertionError("Parent traversal was not rejected")


def test_locked_source_order_and_replay_have_no_premature_authorization() -> None:
    selection = audit._read_json(audit.F0_ROOT / "source_selection.json")
    rows = sorted(
        selection["rows"],
        key=lambda row: (int(row["class_index"]), int(row["edge_rank"])),
    )
    assert [
        (int(row["class_index"]), int(row["edge_rank"])) for row in rows
    ] == [(class_index, rank) for class_index in range(5) for rank in range(2)]
    replay_source = inspect.getsource(audit.replay_generation)
    assert '"a1_authorized": False' in replay_source
    assert '"validation_test_pixels_opened": False' in replay_source

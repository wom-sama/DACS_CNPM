from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from trkh.tools import audit_saspa_synthetic_a0_f1_duplicates as audit


def test_duplicate_lock_and_generation_artifacts_are_immutable() -> None:
    assert audit._sha256(audit.LOCK_PATH) == audit.LOCKED_LOCK_SHA256
    lock = audit._read_json(audit.LOCK_PATH)
    for filename, expected in lock["locked_generation_artifacts"].items():
        assert (
            audit._sha256(audit.GENERATION_ROOT / filename) == expected
        )
    assert lock["enumeration"]["total_reference_files"] == 24996
    assert lock["phash"]["bit_rule"].startswith("coefficient > median")


def test_phash_matches_locked_equation_and_is_symmetric() -> None:
    rng = np.random.default_rng(20260724)
    image = rng.integers(0, 256, size=(91, 137, 3), dtype=np.uint8)
    gray = Image.fromarray(image).convert("L").resize(
        (32, 32),
        Image.Resampling.LANCZOS,
    )
    coefficients = cv2.dct(np.asarray(gray, dtype=np.float32))[:8, :8]
    expected = (
        coefficients > float(np.median(coefficients))
    ).reshape(-1)
    actual = audit._phash_bits(image)
    assert np.array_equal(actual, expected)
    assert audit._phash_hamming(actual, actual.copy()) == 0
    altered = image[:, ::-1].copy()
    reverse = audit._phash_bits(altered)
    assert audit._phash_hamming(actual, reverse) == audit._phash_hamming(
        reverse,
        actual,
    )


def test_decoded_rgb_hash_includes_shape_contract() -> None:
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    descriptor = audit._decoded_rgb_descriptor(image)
    assert descriptor["height"] == 7
    assert descriptor["width"] == 11
    assert len(descriptor["decoded_rgb_sha256"]) == 64


def test_reference_enumeration_is_count_locked_and_globally_sorted(
    tmp_path: Path,
) -> None:
    first = tmp_path / "z_scope"
    second = tmp_path / "a_scope"
    first.mkdir()
    second.mkdir()
    (first / "z.jpg").write_bytes(b"z")
    (second / "a.png").write_bytes(b"a")
    lock = {
        "reference_scopes": [
            {
                "scope": "z",
                "root": str(first),
                "locked_file_count": 1,
            },
            {
                "scope": "a",
                "root": str(second),
                "locked_file_count": 1,
            },
        ],
        "enumeration": {"total_reference_files": 2},
    }
    rows, scopes = audit._enumerate_reference_paths(lock)
    assert len(scopes) == 2
    assert [str(row["path"]) for row in rows] == sorted(
        [str(first / "z.jpg"), str(second / "a.png")],
        key=str.lower,
    )


def test_snapshot_manifest_requires_locked_safetensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = snapshot / "model.safetensors"
    config = snapshot / "config.json"
    model.write_bytes(b"locked-model")
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "MODEL_SAFETENSORS_BYTES",
        model.stat().st_size,
    )
    rows = audit._snapshot_files(snapshot)
    assert [row["path"] for row in rows] == [
        "config.json",
        "model.safetensors",
    ]
    assert rows[1]["sha256"] == audit._sha256(model)


def test_reference_record_persists_only_allowed_nearest_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.jpg"
    path.write_bytes(b"sample")
    row = audit._reference_record(
        {
            "scope": "yolo_f/val",
            "relative_path": "sample.jpg",
            "path": path,
        },
        {
            "decoded_rgb_sha256": "a" * 64,
            "height": 10,
            "width": 20,
        },
        distance_key="phash_hamming",
        distance_value=9,
    )
    assert set(row) == {
        "scope",
        "relative_path",
        "file_sha256",
        "decoded_rgb_sha256",
        "height",
        "width",
        "phash_hamming",
    }
    assert "label" not in row
    assert "embedding" not in row


def test_every_unordered_synthetic_pair_is_evaluated() -> None:
    rows = []
    for index in range(10):
        bits = np.zeros(64, dtype=np.bool_)
        bits[index : index + 16] = True
        rows.append(
            {
                "output_id": "output-{}".format(index),
                "output_sha256": "{:064x}".format(index),
                "decoded": {
                    "height": 512,
                    "width": 512,
                    "decoded_rgb_sha256": "{:064x}".format(index + 20),
                },
                "phash": bits,
            }
        )
    embeddings = np.eye(10, dtype=np.float32)
    pairs = audit._synthetic_pairs(rows, embeddings)
    assert len(pairs) == 45
    assert len(
        {
            (row["left_output_id"], row["right_output_id"])
            for row in pairs
        }
    ) == 45


def test_formal_audit_is_offline_and_never_authorizes_training() -> None:
    source = inspect.getsource(audit.run_formal_audit)
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "AutoModel.from_pretrained" in source
    assert "local_files_only=True" in source
    assert '"reference_embeddings_persisted": False' in source
    assert '"validation_test_labels_opened": False' in source
    assert '"a1_authorized": False' in source
    assert '"training_authorized": False' in source

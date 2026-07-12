import hashlib
import json
from pathlib import Path

import pytest

from trkh.tools.compact_rejected_run_evidence import (
    _verify_payload_manifest,
    compact_rejected_run_evidence,
    refresh_compacted_evidence_metadata,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compactor_verifies_payloads_and_deletes_only_declared_sources(tmp_path):
    runs_root = tmp_path / "runs"
    source_train = runs_root / "rejected_train"
    source_eval = runs_root / "rejected_eval"
    protected = runs_root / "keeper"
    (source_train / "checkpoints").mkdir(parents=True)
    source_eval.mkdir(parents=True)
    protected.mkdir(parents=True)
    (source_train / "metrics.json").write_text('{"macro_f1": 0.5}\n', encoding="utf-8")
    (source_train / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    (source_eval / "predictions.csv").write_text("sample_index,prediction\n0,1\n", encoding="utf-8")
    (source_eval / "summary.json").write_text('{"split": "val"}\n', encoding="utf-8")
    (protected / "best.pt").write_bytes(b"keeper")

    evidence = runs_root / "evidence_rejected"
    cleanup_manifest = runs_root / "cleanup_manifest.json"
    summary = compact_rejected_run_evidence(
        runs_root=runs_root,
        output_dir=evidence,
        sources=(("train", source_train), ("eval", source_eval)),
        notes=("no test", "rejected gate"),
        test_data_used=False,
        protected_paths=(protected,),
        delete_sources=True,
        cleanup_manifest=cleanup_manifest,
        created_at="2026-07-12T00:00:00+07:00",
    )

    assert not source_train.exists()
    assert not source_eval.exists()
    assert (protected / "best.pt").read_bytes() == b"keeper"
    assert (evidence / "train" / "metrics.json").is_file()
    assert (evidence / "eval" / "predictions.csv").is_file()
    assert not (evidence / "train" / "checkpoints" / "best.pt").exists()
    inventory = json.loads((evidence / "source_inventory.json").read_text(encoding="utf-8"))
    excluded = [
        row
        for source in inventory["sources"]
        for row in source["files"]
        if row["excluded"]
    ]
    assert [row["relative_path"] for row in excluded] == ["checkpoints/best.pt"]
    manifest = evidence / "file_manifest.sha256.txt"
    assert _verify_payload_manifest(evidence, manifest) == summary["verified_payloads"]
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "eval/summary.json" in manifest_text
    assert not any(line.endswith("  summary.json") for line in manifest_text.splitlines())
    assert summary["payload_manifest_sha256"] == _sha256(manifest)
    cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    assert cleanup["status"] == "completed"
    assert cleanup["deletion_verified"] is True
    assert cleanup["compacted_names"] == ["rejected_train", "rejected_eval"]
    refreshed = refresh_compacted_evidence_metadata(evidence, cleanup_manifest)
    assert refreshed["verified_payloads"] == summary["verified_payloads"]
    refreshed_cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    assert refreshed_cleanup["evidence_manifest_sha256"] == _sha256(manifest)


def test_compactor_rejects_protected_or_outside_paths(tmp_path):
    runs_root = tmp_path / "runs"
    source = runs_root / "candidate"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected"):
        compact_rejected_run_evidence(
            runs_root=runs_root,
            output_dir=runs_root / "evidence",
            sources=(("candidate", source),),
            test_data_used=False,
            protected_paths=(source,),
        )

    with pytest.raises(ValueError, match="strict child"):
        compact_rejected_run_evidence(
            runs_root=runs_root,
            output_dir=tmp_path / "outside",
            sources=(("candidate", source),),
            test_data_used=False,
        )

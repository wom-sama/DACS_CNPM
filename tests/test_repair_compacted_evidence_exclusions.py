import json
from pathlib import Path

import pytest

from trkh.tools.compact_rejected_run_evidence import (
    _verify_payload_manifest,
    compact_rejected_run_evidence,
)
from trkh.tools.repair_compacted_evidence_exclusions import (
    repair_compacted_evidence_exclusions,
)


def _build_buggy_compaction(tmp_path: Path):
    runs_root = tmp_path / "runs"
    source = runs_root / "rejected"
    (source / "checkpoints").mkdir(parents=True)
    (source / "metrics.json").write_text('{"macro_f1": 0.5}\n', encoding="utf-8")
    (source / "plot.png").write_bytes(b"png")
    (source / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    evidence = runs_root / "evidence"
    cleanup = runs_root / "cleanup.json"
    compact_rejected_run_evidence(
        runs_root=runs_root,
        output_dir=evidence,
        sources=(("candidate", source),),
        exclude_globs=("*.png",),
        test_data_used=False,
        delete_sources=True,
        cleanup_manifest=cleanup,
        created_at="2026-07-15T00:00:00+07:00",
    )
    return runs_root, evidence, cleanup


def test_repair_removes_hash_verified_binary_and_refreshes_metadata(tmp_path):
    runs_root, evidence, cleanup = _build_buggy_compaction(tmp_path)

    result = repair_compacted_evidence_exclusions(
        runs_root=runs_root,
        evidence_dir=evidence,
        cleanup_manifest=cleanup,
        repaired_at="2026-07-15T01:00:00+07:00",
    )

    assert (evidence / "candidate" / "metrics.json").is_file()
    assert not (evidence / "candidate" / "plot.png").exists()
    assert not (evidence / "candidate" / "checkpoints" / "best.pt").exists()
    assert result["copied_files"] == 1
    assert result["excluded_files"] == 2
    assert result["exclusion_repair"]["removed_files"] == 1
    assert result["exclusion_repair"]["removed_bytes"] == len(b"checkpoint")
    inventory = json.loads((evidence / "source_inventory.json").read_text(encoding="utf-8"))
    rows = {row["relative_path"]: row for row in inventory["sources"][0]["files"]}
    assert rows["plot.png"]["excluded"] is True
    assert rows["checkpoints/best.pt"]["excluded"] is True
    assert "evidence_path" not in rows["checkpoints/best.pt"]
    assert rows["metrics.json"]["excluded"] is False
    assert _verify_payload_manifest(
        evidence, evidence / "file_manifest.sha256.txt"
    ) == result["verified_payloads"]
    cleanup_payload = json.loads(cleanup.read_text(encoding="utf-8"))
    assert cleanup_payload["status"] == "completed"
    assert cleanup_payload["excluded_files"] == 2
    assert cleanup_payload["exclusion_repair"]["sha256_verified_before_delete"] is True


def test_repair_refuses_to_delete_tampered_binary(tmp_path):
    runs_root, evidence, cleanup = _build_buggy_compaction(tmp_path)
    checkpoint = evidence / "candidate" / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="(size|hash) mismatch"):
        repair_compacted_evidence_exclusions(
            runs_root=runs_root,
            evidence_dir=evidence,
            cleanup_manifest=cleanup,
        )

    assert checkpoint.is_file()

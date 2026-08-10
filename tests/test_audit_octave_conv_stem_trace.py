from __future__ import annotations

import json
from pathlib import Path

from trkh.tools.audit_tokenizer_trace import run_audit


def _trace_sample(class_id: int, root: Path) -> dict[str, object]:
    return {
        "class_id": class_id,
        "source_image": str(root / "images" / "train" / f"sample_{class_id}.jpg"),
        "stem_shape": [1, 256, 32, 32],
        "patch_embedding_shape": [1, 256, 256],
        "grid_size": [16, 16],
        "block_token_shapes": [[1, 263, 256] for _ in range(8)],
        "pruning": [{"layer": 2}, {"layer": 5}],
    }


def test_octave_trace_audit_accepts_baseline_matched_stem_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    method = "octave_conv_parameter_matched_frequency_stem"
    trace = {
        "split_used": "train",
        "weights": str(checkpoint),
        "model_config": {
            "stem_architecture": "octave_conv",
            "pretrained": False,
            "image_size": 256,
            "patch_size": 16,
            "depth": 8,
            "num_heads": 8,
            "num_registers": 4,
            "token_pruning": True,
            "token_prune_layers": "2,5",
        },
        "samples": [_trace_sample(index, tmp_path) for index in range(5)],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    stage_a_path = tmp_path / "stage_a.json"
    stage_a_path.write_text(
        json.dumps({"method": method, "gate": {"smoke_permission": True}}),
        encoding="utf-8",
    )

    summary = run_audit(
        trace_summary=trace_path,
        checkpoint=checkpoint,
        stage_a_summary=stage_a_path,
        output_dir=tmp_path / "audit",
        expected_method=method,
        expected_stem="octave_conv",
        expected_stem_channels=256,
        expected_stem_spatial_size=32,
        mode="octave_conv_architecture_trace_audit",
    )

    assert summary["passed"] is True
    assert summary["failed_checks"] == []

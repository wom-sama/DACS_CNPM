from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


EXPECTED_STEM = "inceptionnext_atto_tokenizer"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def run_audit(
    *,
    trace_summary: Path,
    checkpoint: Path,
    stage_a_summary: Path,
    output_dir: Path,
) -> Dict[str, object]:
    trace_summary = Path(trace_summary).resolve()
    checkpoint = Path(checkpoint).resolve()
    stage_a_summary = Path(stage_a_summary).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Trace audit output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = _load_json(trace_summary)
    stage_a = _load_json(stage_a_summary)
    model = trace.get("model_config", {})
    samples = trace.get("samples", [])
    if not isinstance(model, Mapping) or not isinstance(samples, list):
        raise ValueError("Trace is missing model_config or samples")
    stage_gate = stage_a.get("gate", {})
    if stage_a.get("method") != "inceptionnext_atto_surface_tokenizer":
        raise ValueError("Stage-A method mismatch")
    if not isinstance(stage_gate, Mapping) or not bool(stage_gate.get("smoke_permission")):
        raise ValueError("Stage-A did not grant smoke permission")

    sample_checks = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("Trace sample must be an object")
        pruning = sample.get("pruning", [])
        source_parts = {part.casefold() for part in Path(str(sample.get("source_image", ""))).parts}
        sample_checks.append(
            {
                "class_id": int(sample.get("class_id", -1)),
                "train_source": "train" in source_parts
                and "val" not in source_parts
                and "test" not in source_parts,
                "stem_shape": sample.get("stem_shape") == [1, 160, 16, 16],
                "patch_embedding_shape": sample.get("patch_embedding_shape")
                == [1, 256, 256],
                "grid_size": sample.get("grid_size") == [16, 16],
                "eight_transformer_blocks": len(sample.get("block_token_shapes", [])) == 8,
                "locked_pruning_layers": [int(row.get("layer", -1)) for row in pruning]
                == [2, 5],
            }
        )

    checks = {
        "train_split_only": str(trace.get("split_used", "")) == "train",
        "checkpoint_path_exact": Path(str(trace.get("weights", ""))).resolve() == checkpoint,
        "candidate_stem_exact": str(model.get("stem_architecture", "")) == EXPECTED_STEM,
        "scratch_only": not bool(model.get("pretrained", True)),
        "image_patch_grid_locked": int(model.get("image_size", 0)) == 256
        and int(model.get("patch_size", 0)) == 16,
        "transformer_locked": int(model.get("depth", 0)) == 8
        and int(model.get("num_heads", 0)) == 8
        and int(model.get("num_registers", 0)) == 4,
        "token_pruning_locked": bool(model.get("token_pruning"))
        and str(model.get("token_prune_layers", "")) == "2,5",
        "five_class_examples": sorted(row["class_id"] for row in sample_checks)
        == [0, 1, 2, 3, 4],
        "all_sample_shapes_and_sources": len(sample_checks) == 5
        and all(
            all(value for key, value in row.items() if key != "class_id")
            for row in sample_checks
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    summary: Dict[str, object] = {
        "mode": "inceptionnext_atto_architecture_trace_audit",
        "checks": {**checks, "test_used": False},
        "failed_checks": failed,
        "passed": not failed,
        "sample_checks": sample_checks,
        "sources": {
            "trace_summary": str(trace_summary),
            "trace_summary_sha256": _sha256(trace_summary),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "stage_a_summary": str(stage_a_summary),
            "stage_a_summary_sha256": _sha256(stage_a_summary),
        },
        "raw_dataset_modified": False,
        "test_used": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a locked InceptionNeXt-Atto architecture trace.")
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(
        trace_summary=args.trace_summary,
        checkpoint=args.checkpoint,
        stage_a_summary=args.stage_a_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

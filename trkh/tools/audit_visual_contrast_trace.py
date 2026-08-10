from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Mapping, Optional, Sequence


TRACE_FIELDS = (
    "visual_contrast_lambda_stage1",
    "visual_contrast_lambda_stage2",
    "visual_contrast_stage1_positive_mass",
    "visual_contrast_stage1_negative_mass",
    "visual_contrast_stage2_positive_mass",
    "visual_contrast_stage2_negative_mass",
    "visual_contrast_stage1_contrast_norm",
    "visual_contrast_stage2_contrast_norm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(sample: Mapping[str, object], field: str, layers: int) -> List[float]:
    raw = sample.get(field)
    if not isinstance(raw, list) or len(raw) != layers:
        raise ValueError(f"{field} must contain exactly {layers} layer values")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} contains a non-finite value")
    return values


def audit_trace_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    split = str(payload.get("split_used", "") or "").casefold()
    if split == "test":
        raise ValueError("test-split architecture traces are forbidden for VCA development")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("trace summary must contain at least one sample")
    first_layers = samples[0].get("visual_contrast_attention_layers")
    if not isinstance(first_layers, list) or not first_layers:
        raise ValueError("visual contrast attention layers are missing")
    layers = len(first_layers)
    traces: Dict[str, List[List[float]]] = {field: [] for field in TRACE_FIELDS}
    class_ids = []
    for sample in samples:
        sample_layers = sample.get("visual_contrast_attention_layers")
        if sample_layers != first_layers:
            raise ValueError("visual contrast layer indices differ across trace samples")
        class_ids.append(int(sample["class_id"]))
        for field in TRACE_FIELDS:
            traces[field].append(_values(sample, field, layers))

    layer_metrics: List[Dict[str, object]] = []
    for layer_offset, layer_index in enumerate(first_layers):
        row: Dict[str, object] = {"layer_index": int(layer_index)}
        for field in TRACE_FIELDS:
            row[f"{field}_mean"] = fmean(
                sample_values[layer_offset] for sample_values in traces[field]
            )
        row["stage1_mass_gap_mean"] = fmean(
            abs(
                traces["visual_contrast_stage1_positive_mass"][sample_index][
                    layer_offset
                ]
                - traces["visual_contrast_stage1_negative_mass"][sample_index][
                    layer_offset
                ]
            )
            for sample_index in range(len(samples))
        )
        row["stage2_mass_gap_mean"] = fmean(
            abs(
                traces["visual_contrast_stage2_positive_mass"][sample_index][
                    layer_offset
                ]
                - traces["visual_contrast_stage2_negative_mass"][sample_index][
                    layer_offset
                ]
            )
            for sample_index in range(len(samples))
        )
        layer_metrics.append(row)

    all_stage1_norms = [
        value
        for sample_values in traces["visual_contrast_stage1_contrast_norm"]
        for value in sample_values
    ]
    all_stage2_norms = [
        value
        for sample_values in traces["visual_contrast_stage2_contrast_norm"]
        for value in sample_values
    ]
    stage1_balanced = [
        abs(positive - negative) < 0.05
        for positive_values, negative_values in zip(
            traces["visual_contrast_stage1_positive_mass"],
            traces["visual_contrast_stage1_negative_mass"],
        )
        for positive, negative in zip(positive_values, negative_values)
    ]
    stage2_balanced = [
        abs(positive - negative) < 0.05
        for positive_values, negative_values in zip(
            traces["visual_contrast_stage2_positive_mass"],
            traces["visual_contrast_stage2_negative_mass"],
        )
        for positive, negative in zip(positive_values, negative_values)
    ]
    return {
        "mode": "visual_contrast_architecture_trace_audit",
        "split_used": payload.get("split_used"),
        "sample_count": len(samples),
        "class_ids": class_ids,
        "layer_count": layers,
        "layer_indices": [int(index) for index in first_layers],
        "checks": {
            "all_values_finite": True,
            "all_layers_present": True,
            "stage1_contrast_nonzero": min(all_stage1_norms) > 1e-8,
            "stage2_contrast_nonzero": min(all_stage2_norms) > 1e-8,
            "test_used": False,
        },
        "diagnostics": {
            "stage1_contrast_norm_min": min(all_stage1_norms),
            "stage1_contrast_norm_max": max(all_stage1_norms),
            "stage2_contrast_norm_min": min(all_stage2_norms),
            "stage2_contrast_norm_max": max(all_stage2_norms),
            "mass_near_balance_threshold": 0.05,
            "stage1_near_balanced_fraction": sum(stage1_balanced)
            / len(stage1_balanced),
            "stage2_near_balanced_fraction": sum(stage2_balanced)
            / len(stage2_balanced),
            "mass_balance_is_diagnostic_not_a_collapse_gate": True,
        },
        "layer_metrics": layer_metrics,
        "raw_dataset_modified": False,
    }


def run_audit(*, trace_summary: Path, output_dir: Path) -> Dict[str, object]:
    trace_summary = Path(trace_summary).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Trace audit output already exists and is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(trace_summary.read_text(encoding="utf-8"))
    summary = audit_trace_payload(payload)
    summary["source"] = str(trace_summary)
    summary["source_sha256"] = _sha256(trace_summary)

    layer_path = output_dir / "layer_metrics.csv"
    layer_rows = summary["layer_metrics"]
    with layer_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    summary["layer_metrics_csv"] = str(layer_path)
    summary["layer_metrics_sha256"] = _sha256(layer_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate finite/nonzero and mass-balance diagnostics from a VCA trace."
    )
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(trace_summary=args.trace_summary, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

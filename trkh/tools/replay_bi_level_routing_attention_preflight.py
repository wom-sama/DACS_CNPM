from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.models.model import create_model
from trkh.tools.audit_bi_level_routing_attention_preflight import (
    EXPECTED_HOLDOUT_ROWS,
    _bra_config,
    _build_dataset,
    _condition_summary,
    _load_json,
    _sha256,
    _trace_parity,
    _unpack_batch,
    _write_manifest,
)


METHOD = "bi_level_routing_attention_a1_postflight_replay"
LOCKED_FORMAL_SUMMARY_SHA256 = (
    "e8f0c54000d2f4ce05199cc3fc7ee2f59f1643598c36c627174bcb2ac80f8e0b"
)
LOCKED_SELECTIVITY_ROWS_SHA256 = (
    "86c1086ed872c50cfc4253ec1b3c70588021efbbd2228eb1bdea9094bcbf728e"
)
NUMERIC_COLUMNS = (
    "foreground_gain",
    "far_background_reduction",
    "routed_foreground_fraction",
    "all_region_foreground_fraction",
    "routed_far_background_fraction",
    "all_region_far_background_fraction",
    "distinct_route_sets",
    "pairwise_route_jaccard",
    "nonlocal_route_fraction",
    "probability_mae",
    "selected_affinity_margin",
    "bbox_area",
    "bbox_edge_gap",
    "bbox_center_edge_distance",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only replay of BRA preflight reporting defects. This tool can "
            "never grant pair, validation, or test permission."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/audit_bra_preflight_20260716/summary.json"),
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path(
            "runs/audit_bra_preflight_20260716/route_selectivity_rows.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/audit_bra_preflight_20260716/postflight_replay.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _parse_selectivity_rows(path: Path) -> Dict[str, list[Dict[str, object]]]:
    grouped: Dict[str, list[Dict[str, object]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "valid_object_query",
            "clean_route_jaccard",
            *NUMERIC_COLUMNS,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"BRA selectivity CSV lacks columns: {sorted(missing)}")
        for raw in reader:
            row: Dict[str, object] = dict(raw)
            row["sample_index"] = int(raw["sample_index"])
            row["valid_object_query"] = (
                str(raw["valid_object_query"]).strip().lower() == "true"
            )
            for name in NUMERIC_COLUMNS:
                row[name] = float(raw[name])
            row["clean_route_jaccard"] = (
                ""
                if str(raw["clean_route_jaccard"]).strip() == ""
                else float(raw["clean_route_jaccard"])
            )
            grouped.setdefault(str(raw["condition"]), []).append(row)
    return grouped


def _corrected_selectivity(
    formal: Mapping[str, object], grouped: Mapping[str, Sequence[Mapping[str, object]]]
) -> Dict[str, object]:
    conditions = {
        name: _condition_summary(rows) for name, rows in grouped.items()
    }
    expected_conditions = {
        "clean",
        "lighting_dim",
        "lighting_bright",
        "low_contrast",
    }
    if set(conditions) != expected_conditions:
        raise ValueError("BRA postflight condition inventory differs from the protocol.")
    if any(int(value["rows"]) != EXPECTED_HOLDOUT_ROWS for value in conditions.values()):
        raise ValueError("BRA postflight condition row count differs from the protocol.")
    clean = conditions["clean"]
    shifted = sorted(expected_conditions.difference({"clean"}))
    checks = {
        "all_object_queries_valid": all(
            int(value["valid_object_query_rows"]) == EXPECTED_HOLDOUT_ROWS
            for value in conditions.values()
        ),
        "all_rows_finite": all(
            int(value["finite_rows"]) == EXPECTED_HOLDOUT_ROWS
            for value in conditions.values()
        ),
        "clean_foreground_gain": float(clean["foreground_gain_mean"]) >= 0.020,
        "clean_positive_gain_fraction": float(
            clean["positive_foreground_gain_fraction"]
        )
        >= 0.55,
        "clean_far_background_reduction": float(
            clean["far_background_reduction_mean"]
        )
        >= 0.020,
        "shift_foreground_gain_positive": all(
            float(conditions[name]["foreground_gain_mean"]) > 0.0
            for name in shifted
        ),
        "shift_far_background_not_above_control": all(
            float(conditions[name]["far_background_reduction_mean"]) >= 0.0
            for name in shifted
        ),
        "condition_route_stability": all(
            float(conditions[name]["clean_route_jaccard_mean"]) >= 0.65
            for name in shifted
        ),
    }
    invalid_indices = sorted(
        {
            int(row["sample_index"])
            for rows in grouped.values()
            for row in rows
            if not bool(row["valid_object_query"])
        }
    )
    original_selectivity = formal.get("selectivity")
    if not isinstance(original_selectivity, Mapping):
        raise ValueError("Formal BRA summary lacks selectivity evidence.")
    return {
        "conditions": conditions,
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "invalid_object_query_sample_indices": invalid_indices,
        "original_rows_sha256": original_selectivity.get("rows_csv_sha256"),
    }


def _corrected_trace_parity(
    formal: Mapping[str, object], *, seed: int
) -> Dict[str, object]:
    sources = formal.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("Formal BRA summary lacks source paths.")
    resolved_config_path = Path(str(sources["resolved_config"]["path"]))
    fold_data_path = Path(str(sources["fold_data"]["path"]))
    resolved_config = _load_json(resolved_config_path)
    source_config = resolved_config.get("model_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("BRA resolved config lacks model_config.")
    config = _bra_config(source_config, topk=4)
    set_seed(seed, deterministic=True)
    model = create_model(num_classes=5, model_config=config).cuda().eval()
    dataset = _build_dataset(
        fold_data=fold_data_path,
        model_config=config,
        resolved_config=resolved_config,
    )
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=0,
        requested_pin_memory=False,
        context="bra_postflight_trace_parity",
        persistent_workers=False,
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
        )
    )
    images_cpu, _labels, metadata_cpu = _unpack_batch(batch)
    images = images_cpu.cuda()
    metadata = {key: value.cuda() for key, value in metadata_cpu.items()}
    parity = _trace_parity(model, images, metadata)
    parity["loader"] = loader_summary
    parity["trace_contract"] = "return_trace_true_return_attention_false_pruning_enabled"
    return parity


def run_replay(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.seed) != 42:
        raise ValueError("BRA postflight replay is locked to seed 42.")
    if not torch.cuda.is_available():
        raise RuntimeError("BRA postflight trace replay requires CUDA.")
    summary_path = Path(args.summary).resolve()
    rows_path = Path(args.rows).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"BRA postflight replay cannot overwrite: {output_path}")
    if _sha256(summary_path) != LOCKED_FORMAL_SUMMARY_SHA256:
        raise ValueError("BRA formal summary differs from the signed visual-review state.")
    if _sha256(rows_path) != LOCKED_SELECTIVITY_ROWS_SHA256:
        raise ValueError("BRA selectivity rows differ from the formal preflight.")
    formal = _load_json(summary_path)
    gate = formal.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("BRA formal summary lacks a gate.")
    if bool(gate.get("formal_pair_permission")) or bool(gate.get("automated_pass")):
        raise ValueError("BRA postflight replay requires the rejected formal artifact.")
    grouped = _parse_selectivity_rows(rows_path)
    corrected_selectivity = _corrected_selectivity(formal, grouped)
    trace_parity = _corrected_trace_parity(formal, seed=int(args.seed))
    model = formal.get("model")
    resource = formal.get("resource")
    visual = formal.get("visual_review")
    if not isinstance(model, Mapping) or not isinstance(resource, Mapping):
        raise ValueError("BRA formal summary lacks model/resource evidence.")
    role_rng_equal = bool(
        model["rng"]["control"] == model["rng"]["candidate"]
    )
    material_failures = {
        "inference_runtime_ratio": float(resource["inference_runtime_ratio"])
        <= 1.35,
        "all_object_queries_valid": bool(
            corrected_selectivity["checks"]["all_object_queries_valid"]
        ),
        "clean_positive_gain_fraction": bool(
            corrected_selectivity["checks"]["clean_positive_gain_fraction"]
        ),
        "condition_route_stability": bool(
            corrected_selectivity["checks"]["condition_route_stability"]
        ),
        "visual_review_passed": bool(
            isinstance(visual, Mapping) and visual.get("passed", False)
        ),
    }
    result: Dict[str, object] = {
        "method": METHOD,
        "formal_summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
            "automated_pass": False,
            "formal_pair_permission": False,
        },
        "selectivity_rows": {
            "path": str(rows_path),
            "sha256": _sha256(rows_path),
        },
        "reporting_corrections": {
            "constructor_rng_contract": {
                "topk16_topk4_equal": role_rng_equal,
                "baseline_comparison_was_not_required": True,
            },
            "trace_parity": trace_parity,
            "finite_geometry_aggregation": corrected_selectivity,
        },
        "material_gate_checks": material_failures,
        "material_failed_checks": sorted(
            name for name, passed in material_failures.items() if not passed
        ),
        "decision": {
            "route_closed_before_training": True,
            "formal_pair_permission": False,
            "validation_permission": False,
            "test_permission": False,
            "reason": (
                "Correcting reporting defects cannot override independent runtime, "
                "coverage-consistency, dim-route-stability, and visual failures."
            ),
        },
    }
    if not role_rng_equal or not bool(trace_parity["passed"]):
        raise RuntimeError("BRA postflight reporting correction did not replay.")
    if all(material_failures.values()):
        raise RuntimeError("BRA postflight unexpectedly found no material rejection.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_path.parent)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    result = run_replay(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

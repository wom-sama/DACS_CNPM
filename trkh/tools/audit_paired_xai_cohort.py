from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trkh.tools.audit_precision_ensemble_readiness import _sha256, _write_csv


HEATMAP_METHODS = ("attention", "rollout", "grad_rollout", "gradcam")
HEATMAP_METRICS = ("foreground_mass", "background_mass", "border_mass", "entropy")
ROBUSTNESS_PROBES = (
    "background_gray",
    "background_blur",
    "object_desaturate",
    "center_occlusion",
)


def _reject_test_path(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if re.search(r"(^|[/_.-])test([/_.-]|$)", normalized):
        raise ValueError(f"Paired XAI readiness is validation-only; refusing test path: {path}")


def _read_cohort(path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_index" not in reader.fieldnames:
            raise ValueError(f"Cohort CSV must contain sample_index: {path}")
        order: List[str] = []
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            key = str(row.get("sample_index", "") or "").strip()
            if not key:
                continue
            if key in rows:
                raise ValueError(f"Duplicate cohort sample_index={key}")
            order.append(key)
            rows[key] = dict(row)
    if not rows:
        raise ValueError(f"Cohort CSV is empty: {path}")
    return order, rows


def _read_xai_cases(root: Path) -> Dict[str, Dict[str, object]]:
    cases: Dict[str, Dict[str, object]] = {}
    for path in sorted(root.glob("case_*/case.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = str(payload.get("sample_index", ""))
        if not key:
            raise ValueError(f"XAI case has no sample_index: {path}")
        if key in cases:
            raise ValueError(f"Duplicate XAI sample_index={key} under {root}")
        payload["case_json"] = str(path.resolve())
        cases[key] = payload
    if not cases:
        raise ValueError(f"No case_*/case.json artifacts found under {root}")
    return cases


def _nested_float(payload: Mapping[str, object], keys: Sequence[str]) -> float:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return float("nan")
        value = value[key]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def flatten_xai_case(payload: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "sample_index": int(payload["sample_index"]),
        "target_index": int(payload["target_index"]),
        "prediction_index": int(payload["prediction_index"]),
        "confidence": float(payload.get("confidence", 0.0)),
        "margin": float(payload.get("margin", 0.0)),
        "correct": int(payload.get("correct", 0)),
        "attention_source": str(
            payload.get("viz", {}).get("attention_source", "")
            if isinstance(payload.get("viz"), Mapping)
            else ""
        ),
        "review_flags": ";".join(
            str(value) for value in payload.get("review_flags", [])
        ),
        "case_json": str(payload.get("case_json", "")),
    }
    for field in ("confidence", "margin"):
        if not math.isfinite(float(row[field])):
            raise ValueError(
                f"XAI case sample_index={row['sample_index']} has non-finite {field}."
            )
    for method in HEATMAP_METHODS:
        for metric in HEATMAP_METRICS:
            row[f"{method}_{metric}"] = _nested_float(
                payload,
                ("viz", "heatmap_focus", method, metric),
            )
    for field in (
        "selected_layer_count",
        "gradient_layer_count",
        "fallback_layer_count",
        "missing_gradient_layer_count",
        "zero_weight_fallback_layer_count",
    ):
        row[f"grad_rollout_{field}"] = _nested_float(
            payload,
            ("viz", "grad_rollout_provenance", field),
        )
    for probe in ROBUSTNESS_PROBES:
        for metric in ("original_prediction_drop", "target_probability_drop"):
            row[f"{probe}_{metric}"] = _nested_float(
                payload,
                ("robustness", probe, metric),
            )
    return row


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> Optional[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(field, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else None


def _known_attention_source(value: object) -> bool:
    source = str(value)
    return source in {"native_attention", "feature_map_fallback"} or re.fullmatch(
        r"forward_features\.return_attention\.(?:blocks|late_member_blocks)\[\d+\]"
        r"(?:\.(?:mhsa_probability|vca_effective_positive|mixed))?",
        source,
    ) is not None


def _case_json_set_sha256(cases: Mapping[str, Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(
        cases,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        path = Path(str(cases[key].get("case_json", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Missing XAI case JSON for sample_index={key}: {path}")
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _aggregate(rows: Sequence[Mapping[str, object]], model_prefix: str) -> Dict[str, object]:
    numeric_fields = [
        "confidence",
        "margin",
        *[
            f"{method}_{metric}"
            for method in HEATMAP_METHODS
            for metric in HEATMAP_METRICS
        ],
        *[
            f"{probe}_{metric}"
            for probe in ROBUSTNESS_PROBES
            for metric in ("original_prediction_drop", "target_probability_drop")
        ],
        "grad_rollout_selected_layer_count",
        "grad_rollout_gradient_layer_count",
        "grad_rollout_fallback_layer_count",
        "grad_rollout_missing_gradient_layer_count",
        "grad_rollout_zero_weight_fallback_layer_count",
    ]
    return {
        "rows": int(len(rows)),
        "accuracy": float(
            np.mean([int(row.get(f"{model_prefix}_correct", 0)) for row in rows])
        )
        if rows
        else 0.0,
        "means": {
            field: _mean(rows, f"{model_prefix}_{field}") for field in numeric_fields
        },
        "attention_sources": dict(
            Counter(str(row.get(f"{model_prefix}_attention_source", "")) for row in rows)
        ),
    }


def _delta_means(
    rows: Sequence[Mapping[str, object]],
    *,
    left_prefix: str,
    right_prefix: str,
) -> Dict[str, Optional[float]]:
    fields = [
        "confidence",
        "margin",
        *[
            f"{method}_{metric}"
            for method in HEATMAP_METHODS
            for metric in HEATMAP_METRICS
        ],
        *[
            f"{probe}_{metric}"
            for probe in ROBUSTNESS_PROBES
            for metric in ("original_prediction_drop", "target_probability_drop")
        ],
    ]
    result: Dict[str, Optional[float]] = {}
    for field in fields:
        deltas = []
        for row in rows:
            try:
                left = float(row[f"{left_prefix}_{field}"])
                right = float(row[f"{right_prefix}_{field}"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(left) and math.isfinite(right):
                deltas.append(right - left)
        result[field] = float(np.mean(deltas)) if deltas else None
    return result


def _viz_file(payload: Mapping[str, object], method: str) -> Optional[Path]:
    viz = payload.get("viz")
    if not isinstance(viz, Mapping):
        return None
    files = viz.get("files")
    if not isinstance(files, Mapping):
        return None
    if method == "crop":
        value = files.get("crop")
    else:
        method_files = files.get(method)
        value = method_files.get("overlay") if isinstance(method_files, Mapping) else None
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _image_tile(path: Optional[Path], size: Tuple[int, int]) -> Image.Image:
    if path is None:
        return Image.new("RGB", size, (235, 235, 235))
    with Image.open(path) as image:
        tile = image.convert("RGB")
    tile.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 248))
    canvas.paste(
        tile,
        ((size[0] - tile.width) // 2, (size[1] - tile.height) // 2),
    )
    return canvas


def _grad_rollout_title(
    name: str,
    cases: Mapping[str, Mapping[str, object]],
    keys: Sequence[str],
) -> str:
    fallback_cases = 0
    for key in keys:
        viz = cases[key].get("viz")
        provenance = (
            viz.get("grad_rollout_provenance") if isinstance(viz, Mapping) else None
        )
        fallback_layers = (
            provenance.get("fallback_layer_count")
            if isinstance(provenance, Mapping)
            else 0
        )
        try:
            fallback_cases += int(float(fallback_layers or 0) > 0.0)
        except (TypeError, ValueError):
            fallback_cases += 1
    if keys and fallback_cases == len(keys):
        return f"{name} rollout fallback"
    if fallback_cases:
        return f"{name} grad-rollout (mixed)"
    return f"{name} grad-rollout"


def _render_contact_sheet(
    *,
    category: str,
    keys: Sequence[str],
    cohort: Mapping[str, Mapping[str, str]],
    left_cases: Mapping[str, Mapping[str, object]],
    right_cases: Mapping[str, Mapping[str, object]],
    left_name: str,
    right_name: str,
    output_path: Path,
) -> Dict[str, object]:
    tile_size = (220, 220)
    label_height = 34
    header_height = 42
    columns = (
        ("crop", "crop", "crop"),
        (f"{left_name} native attention", "left", "attention"),
        (f"{right_name} native attention", "right", "attention"),
        (f"{left_name} Grad-CAM", "left", "gradcam"),
        (f"{right_name} Grad-CAM", "right", "gradcam"),
        (_grad_rollout_title(left_name, left_cases, keys), "left", "grad_rollout"),
        (_grad_rollout_title(right_name, right_cases, keys), "right", "grad_rollout"),
    )
    width = tile_size[0] * len(columns)
    height = header_height + (tile_size[1] + label_height) * len(keys)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column_index, (title, _side, _method) in enumerate(columns):
        draw.text(
            (column_index * tile_size[0] + 6, 12),
            title,
            fill=(20, 20, 20),
            font=font,
        )
    missing = 0
    left_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", str(left_name)).strip("_") or "left"
    right_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", str(right_name)).strip("_") or "right"
    for row_index, key in enumerate(keys):
        y = header_height + row_index * (tile_size[1] + label_height)
        meta = cohort[key]
        label = (
            f"idx={key} y={meta.get('target_index', '')} "
            f"{left_name}={meta.get(f'{left_prefix}_prediction_index', '')} "
            f"{right_name}={meta.get(f'{right_prefix}_prediction_index', '')}"
        )
        draw.text((6, y + 8), label, fill=(20, 20, 20), font=font)
        tile_y = y + label_height
        for column_index, (_title, side, method) in enumerate(columns):
            payload = left_cases[key] if side in {"crop", "left"} else right_cases[key]
            path = _viz_file(payload, method)
            if path is None:
                missing += 1
            tile = _image_tile(path, tile_size)
            sheet.paste(tile, (column_index * tile_size[0], tile_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path.resolve()),
        "rows": int(len(keys)),
        "columns": [title for title, _side, _method in columns],
        "missing_tiles": int(missing),
        "sha256": _sha256(output_path),
    }


def _validate_case_alignment(
    *,
    key: str,
    cohort_row: Mapping[str, str],
    left: Mapping[str, object],
    right: Mapping[str, object],
    left_prefix: str,
    right_prefix: str,
) -> None:
    if int(left["target_index"]) != int(right["target_index"]):
        raise ValueError(f"Target mismatch for sample_index={key}")
    cohort_target_value = str(cohort_row.get("target_index", "") or "").strip()
    if cohort_target_value:
        cohort_target = int(cohort_target_value)
        if int(left["target_index"]) != cohort_target:
            raise ValueError(
                f"XAI/cohort target mismatch for sample_index={key}: "
                f"xai={left['target_index']}, cohort={cohort_target}"
            )
    for prefix, flattened in ((left_prefix, left), (right_prefix, right)):
        expected_field = f"{prefix}_prediction_index"
        expected_value = str(cohort_row.get(expected_field, "") or "").strip()
        if expected_value and int(flattened["prediction_index"]) != int(expected_value):
            raise ValueError(
                f"XAI/cohort prediction mismatch for sample_index={key}, "
                f"model={prefix}: xai={flattened['prediction_index']}, "
                f"cohort={expected_value}"
            )
        expected_correct = int(
            int(flattened["prediction_index"]) == int(flattened["target_index"])
        )
        if int(flattened["correct"]) != expected_correct:
            raise ValueError(
                f"Inconsistent XAI correctness for sample_index={key}, model={prefix}"
            )


def run_paired_xai_cohort_audit(
    *,
    cohort_csv: Path,
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    left_name: str = "keeper",
    right_name: str = "candidate",
    expected_cases: int = 0,
) -> Dict[str, object]:
    cohort_csv = Path(cohort_csv)
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    for path in (cohort_csv, left_dir, right_dir):
        _reject_test_path(path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Paired XAI output already exists and is nonempty: {output_dir}"
            )
    order, cohort = _read_cohort(cohort_csv)
    left_cases = _read_xai_cases(left_dir)
    right_cases = _read_xai_cases(right_dir)
    expected = set(order)
    if set(left_cases) != expected or set(right_cases) != expected:
        raise ValueError(
            "Paired XAI sample mismatch: "
            f"cohort={len(expected)}, left={len(left_cases)}, right={len(right_cases)}, "
            f"left_missing={len(expected - set(left_cases))}, "
            f"right_missing={len(expected - set(right_cases))}"
        )
    if int(expected_cases) > 0 and len(order) != int(expected_cases):
        raise ValueError(f"Expected {expected_cases} cases, found {len(order)}")
    left_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", str(left_name)).strip("_") or "left"
    right_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", str(right_name)).strip("_") or "right"
    if left_prefix == right_prefix:
        raise ValueError(
            f"Model names collide after sanitization: {left_name!r}, {right_name!r}"
        )
    paired_rows: List[Dict[str, object]] = []
    for key in order:
        left = flatten_xai_case(left_cases[key])
        right = flatten_xai_case(right_cases[key])
        _validate_case_alignment(
            key=key,
            cohort_row=cohort[key],
            left=left,
            right=right,
            left_prefix=left_prefix,
            right_prefix=right_prefix,
        )
        row: Dict[str, object] = dict(cohort[key])
        for field, value in left.items():
            if field == "sample_index":
                continue
            row[f"{left_prefix}_{field}"] = value
        for field, value in right.items():
            if field == "sample_index":
                continue
            row[f"{right_prefix}_{field}"] = value
        paired_rows.append(row)

    categories = ["all"] + sorted(
        {str(row.get("xai_category", "") or "uncategorized") for row in paired_rows}
    )
    category_summary: Dict[str, object] = {}
    for category in categories:
        selected = (
            paired_rows
            if category == "all"
            else [
                row
                for row in paired_rows
                if str(row.get("xai_category", "") or "uncategorized") == category
            ]
        )
        category_summary[category] = {
            left_prefix: _aggregate(selected, left_prefix),
            right_prefix: _aggregate(selected, right_prefix),
            f"{right_prefix}_minus_{left_prefix}": _delta_means(
                selected,
                left_prefix=left_prefix,
                right_prefix=right_prefix,
            ),
        }
    all_summary = category_summary["all"]
    attention_sources_valid = all(
        all(_known_attention_source(source) for source in all_summary[prefix]["attention_sources"])
        for prefix in (left_prefix, right_prefix)
    )
    fallback_present = any(
        int(all_summary[prefix]["attention_sources"].get("feature_map_fallback", 0)) > 0
        for prefix in (left_prefix, right_prefix)
    )
    grad_rollout_fallback_cases = {
        prefix: int(
            sum(
                float(row.get(f"{prefix}_grad_rollout_fallback_layer_count", 0.0))
                > 0.0
                for row in paired_rows
            )
        )
        for prefix in (left_prefix, right_prefix)
    }
    if not attention_sources_valid:
        raise ValueError("Paired XAI contains missing or unknown attention provenance.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_xai_cases.csv"
    _write_csv(paired_path, paired_rows)
    contact_sheets: Dict[str, object] = {}
    for category in sorted(
        {str(row.get("xai_category", "") or "uncategorized") for row in paired_rows}
    ):
        category_keys = [
            str(row["sample_index"])
            for row in paired_rows
            if str(row.get("xai_category", "") or "uncategorized") == category
        ]
        safe_category = re.sub(r"[^a-zA-Z0-9_-]+", "_", category).strip("_")
        contact_sheets[category] = _render_contact_sheet(
            category=category,
            keys=category_keys,
            cohort=cohort,
            left_cases=left_cases,
            right_cases=right_cases,
            left_name=left_name,
            right_name=right_name,
            output_path=output_dir / f"paired_xai_{safe_category}.png",
        )
    summary: Dict[str, object] = {
        "mode": "validation_only_paired_xai_cohort",
        "cohort": str(cohort_csv.resolve()),
        "cohort_sha256": _sha256(cohort_csv),
        "cases": int(len(order)),
        "left": {
            "name": left_name,
            "dir": str(left_dir.resolve()),
            "case_json_count": int(len(left_cases)),
            "case_json_set_sha256": _case_json_set_sha256(left_cases),
        },
        "right": {
            "name": right_name,
            "dir": str(right_dir.resolve()),
            "case_json_count": int(len(right_cases)),
            "case_json_set_sha256": _case_json_set_sha256(right_cases),
        },
        "categories": category_summary,
        "attention_provenance": {
            "known_sources_only": attention_sources_valid,
            "feature_map_fallback_present": fallback_present,
            "interpretation": (
                "Fallback heatmaps are structural feature maps, not native Transformer attention. "
                "Use Grad-CAM and grad-rollout as the primary attribution evidence when fallback is present."
            ),
        },
        "grad_rollout_provenance": {
            "fallback_case_count": grad_rollout_fallback_cases,
            "fallback_present": any(grad_rollout_fallback_cases.values()),
            "interpretation": (
                "A fallback layer uses ordinary attention rollout because its returned "
                "attention tensor has no usable gradient path to the selected logit. "
                "Do not interpret fallback cases as gradient-weighted attribution."
            ),
        },
        "contact_sheets": contact_sheets,
        "test_data_used": False,
        "artifacts": {
            "paired_cases": "paired_xai_cases.csv",
            "paired_cases_sha256": _sha256(paired_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two FP32 XAI runs over an exact validation cohort."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-name", default="keeper")
    parser.add_argument("--right-name", default="candidate")
    parser.add_argument("--expected-cases", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_paired_xai_cohort_audit(
        cohort_csv=args.cohort,
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        output_dir=args.output_dir,
        left_name=args.left_name,
        right_name=args.right_name,
        expected_cases=args.expected_cases,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "cases": summary["cases"],
                "feature_map_fallback_present": summary["attention_provenance"][
                    "feature_map_fallback_present"
                ],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

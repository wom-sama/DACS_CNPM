from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _transition(case: Mapping[str, object]) -> str:
    return f"{int(case.get('target_index', 0))}->{int(case.get('prediction_index', 0))}"


def _nested(mapping: Mapping[str, object], keys: Sequence[str]) -> object:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _robustness_drop(case: Mapping[str, object], probe: str, field: str) -> float | None:
    value = _nested(case, ("robustness", probe, field))
    if value is None:
        return None
    return _safe_float(value)


def _heatmap_metric(case: Mapping[str, object], method: str, field: str) -> float | None:
    value = _nested(case, ("viz", "heatmap_focus", method, field))
    if value is None:
        return None
    return _safe_float(value)


def _append_if_present(bucket: Dict[str, List[float]], key: str, value: float | None) -> None:
    if value is not None:
        bucket[key].append(float(value))


def _summarize_cases(cases: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "cases": 0,
            "flags": Counter(),
            "values": defaultdict(list),
        }
    )
    for case in cases:
        transition = _transition(case)
        entry = grouped[transition]
        entry["cases"] = int(entry["cases"]) + 1
        entry["flags"].update(str(flag) for flag in case.get("review_flags", []) or [])
        values: Dict[str, List[float]] = entry["values"]  # type: ignore[assignment]
        for probe in ("background_blur", "background_gray", "object_desaturate", "center_occlusion"):
            _append_if_present(
                values,
                f"{probe}_original_prediction_drop",
                _robustness_drop(case, probe, "original_prediction_drop"),
            )
            _append_if_present(
                values,
                f"{probe}_target_probability_drop",
                _robustness_drop(case, probe, "target_probability_drop"),
            )
        for method in ("attention", "rollout", "grad_rollout", "gradcam"):
            for field in ("foreground_mass", "background_mass", "border_mass", "entropy"):
                _append_if_present(
                    values,
                    f"{method}_{field}",
                    _heatmap_metric(case, method, field),
                )

    rows: List[Dict[str, object]] = []
    for transition, entry in sorted(grouped.items()):
        values = entry["values"]
        flags: Counter[str] = entry["flags"]  # type: ignore[assignment]
        row: Dict[str, object] = {
            "transition": transition,
            "cases": int(entry["cases"]),
            "top_flags": ";".join(f"{name}:{count}" for name, count in flags.most_common(8)),
        }
        for key in sorted(values):
            row[f"{key}_mean"] = _mean(values[key])
        rows.append(row)
    return rows


def _case_image_path(case: Mapping[str, object], image_kind: str) -> Optional[Path]:
    files = _nested(case, ("viz", "files"))
    if not isinstance(files, Mapping):
        return None
    if image_kind == "crop":
        path_value = files.get("crop")
    else:
        section = files.get(image_kind)
        path_value = section.get("overlay") if isinstance(section, Mapping) else None
    if not path_value:
        return None
    path = Path(str(path_value))
    return path if path.is_file() else None


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_resized(source: Path, sheet: Image.Image, xy: tuple[int, int], size: int) -> None:
    image = Image.open(source).convert("RGB")
    image.thumbnail((size, size))
    sheet.paste(image, xy)


def _write_contact_sheet(
    *,
    cases: Sequence[Mapping[str, object]],
    output_path: Path,
    columns: int,
    thumbnail_size: int,
    include_images: Sequence[str],
) -> Dict[str, object]:
    selected_images = [item for item in include_images if item]
    if not selected_images:
        raise ValueError("At least one contact-sheet image kind is required.")
    columns = max(1, int(columns))
    thumbnail_size = max(32, int(thumbnail_size))
    label_height = 42
    image_gap = 4
    cell_width = len(selected_images) * thumbnail_size + max(0, len(selected_images) - 1) * image_gap + 8
    cell_height = thumbnail_size + label_height + 10
    rows = max(1, int(math.ceil(len(cases) / columns)))
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = _load_font(14)
    rendered = 0
    missing_images = 0
    for index, case in enumerate(cases, start=1):
        row_index, column_index = divmod(index - 1, columns)
        x0 = column_index * cell_width
        y0 = row_index * cell_height
        transition = _transition(case)
        sample_index = case.get("sample_index", "")
        confidence = _safe_float(case.get("confidence"), 0.0)
        draw.text(
            (x0 + 4, y0 + 4),
            f"#{index} s{sample_index} {transition} conf {confidence:.3f}",
            fill="black",
            font=font,
        )
        for image_index, image_kind in enumerate(selected_images):
            image_path = _case_image_path(case, image_kind)
            px = x0 + 4 + image_index * (thumbnail_size + image_gap)
            py = y0 + label_height
            if image_path is None:
                missing_images += 1
                draw.rectangle((px, py, px + thumbnail_size - 1, py + thumbnail_size - 1), outline="red")
                draw.text((px + 4, py + 4), f"missing {image_kind}", fill="red", font=font)
                continue
            _draw_resized(image_path, sheet, (px, py), thumbnail_size)
            draw.text((px + 2, py + thumbnail_size - 16), image_kind, fill="black", font=font)
            rendered += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return {
        "contact_sheet": str(output_path.resolve()),
        "cases": int(len(cases)),
        "image_kinds": list(selected_images),
        "rendered_images": int(rendered),
        "missing_images": int(missing_images),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_xai_transitions(
    *,
    xai_summary_json: Path,
    output_dir: Path,
    contact_sheet: bool = True,
    contact_sheet_name: str = "xai_transition_contact_sheet.png",
    contact_sheet_columns: int = 4,
    thumbnail_size: int = 180,
    include_images: Sequence[str] = ("crop", "gradcam", "rollout"),
    leakage_guard: str = (
        "XAI transition summary only. This tool writes no sample-weight, soft-target, "
        "targeted-margin, relabel, or raw-data files."
    ),
) -> Dict[str, object]:
    source_path = Path(xai_summary_json)
    output_dir = Path(output_dir)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"XAI summary has no case list: {source_path}")
    transition_rows = _summarize_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transition_xai_summary.csv"
    _write_csv(csv_path, transition_rows)
    contact_summary: Dict[str, object] = {}
    if contact_sheet:
        contact_summary = _write_contact_sheet(
            cases=cases,
            output_path=output_dir / contact_sheet_name,
            columns=contact_sheet_columns,
            thumbnail_size=thumbnail_size,
            include_images=include_images,
        )
    summary: Dict[str, object] = {
        "source_xai_summary": str(source_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_cases": int(len(cases)),
        "transition_count": int(len(transition_rows)),
        "transition_rows": transition_rows,
        "transition_summary_csv": str(csv_path.resolve()),
        "contact_sheet": contact_summary,
        "raw_dataset_touched": False,
        "test_split_used": str(payload.get("split", "")).lower() == "test",
        "trainable_manifest_written": False,
        "guardrail": leakage_guard,
    }
    (output_dir / "transition_xai_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# XAI Transition Summary",
                "",
                f"Source: `{source_path}`",
                "",
                "This artifact summarizes existing XAI cases by target->prediction transition.",
                "It is diagnostic-only and must not be used as a trainable manifest.",
                "",
                f"Cases: `{len(cases)}`",
                f"Transitions: `{len(transition_rows)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _split_values(value: str | Sequence[str]) -> List[str]:
    chunks = [value] if isinstance(value, str) else list(value)
    values: List[str] = []
    for chunk in chunks:
        for part in str(chunk).replace(";", ",").split(","):
            item = part.strip()
            if item:
                values.append(item)
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize XAI audit cases by class transition.")
    parser.add_argument("--xai-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--contact-sheet-name", type=str, default="xai_transition_contact_sheet.png")
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--thumbnail-size", type=int, default=180)
    parser.add_argument(
        "--include-images",
        nargs="*",
        default=("crop", "gradcam", "rollout"),
        help="Contact-sheet image kinds, comma-separated or repeated. Valid examples: crop,gradcam,rollout.",
    )
    parser.add_argument("--leakage-guard", type=str, default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = summarize_xai_transitions(
        xai_summary_json=args.xai_summary_json,
        output_dir=args.output_dir,
        contact_sheet=not args.no_contact_sheet,
        contact_sheet_name=args.contact_sheet_name,
        contact_sheet_columns=args.contact_sheet_columns,
        thumbnail_size=args.thumbnail_size,
        include_images=_split_values(args.include_images),
        leakage_guard=args.leakage_guard
        or (
            "XAI transition summary only. This tool writes no sample-weight, soft-target, "
            "targeted-margin, relabel, or raw-data files."
        ),
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REVIEW_IMAGE_NAME = re.compile(r"^([A-Za-z]+_\d{5})_")


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _safe_float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "") or "").strip())
    except ValueError:
        return float(default)


def _sort_rows(rows: Sequence[Mapping[str, str]]) -> List[Mapping[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            -_safe_float(row, "severity", 0.0),
            _safe_float(row, "top2_margin", 1.0),
            str(row.get("review_id", "")),
        ),
    )


def _find_review_images(root: Path) -> Dict[str, Path]:
    image_root = Path(root) / "review_images"
    if not image_root.is_dir():
        return {}
    mapping: Dict[str, Path] = {}
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = REVIEW_IMAGE_NAME.match(image_path.name)
        review_id = match.group(1) if match is not None else image_path.name.split("_", 1)[0]
        if review_id and review_id not in mapping:
            mapping[review_id] = image_path
    return mapping


def _image_src(path: Path, output_html: Path) -> str:
    try:
        return html.escape(path.resolve().relative_to(output_html.parent.resolve()).as_posix())
    except ValueError:
        try:
            return html.escape(path.resolve().as_uri())
        except ValueError:
            return html.escape(str(path))


def _source_image_src(row: Mapping[str, str]) -> str:
    image_path = Path(str(row.get("image_path", "") or ""))
    if not image_path.is_file():
        return ""
    try:
        return html.escape(image_path.resolve().as_uri())
    except ValueError:
        return html.escape(str(image_path))


def render_boundary_review_html(
    *,
    manifest: Path,
    output_html: Path,
    title: str,
    max_rows: int,
    image_width: int,
) -> Dict[str, object]:
    rows, _ = _read_csv(Path(manifest))
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    image_map = _find_review_images(Path(manifest).parent)
    selected = _sort_rows(rows)
    if int(max_rows) > 0:
        selected = selected[: int(max_rows)]

    by_reason: Dict[str, int] = {}
    by_pair: Dict[str, int] = {}
    by_bucket: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "") or "")
        pair = str(row.get("boundary_pair", "") or "")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_pair[pair] = by_pair.get(pair, 0) + 1
        for bucket in str(row.get("buckets", "") or "").split(";"):
            if bucket:
                by_bucket[bucket] = by_bucket.get(bucket, 0) + 1

    def cell(row: Mapping[str, str], key: str) -> str:
        return html.escape(str(row.get(key, "") or ""))

    table_rows: List[str] = []
    for row in selected:
        review_id = str(row.get("review_id", "") or "")
        image_path = image_map.get(review_id)
        image_src = _image_src(image_path, output_html) if image_path is not None else _source_image_src(row)
        image_html = (
            f'<img src="{image_src}" width="{int(image_width)}" loading="lazy" />'
            if image_src
            else "<span class=\"missing\">missing image</span>"
        )
        table_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td>{image_html}</td>",
                    f"<td><b>{cell(row, 'review_id')}</b><br>{cell(row, 'split')}<br>{cell(row, 'reason')}</td>",
                    f"<td>{cell(row, 'target_index')}<br>{cell(row, 'target_name')}</td>",
                    f"<td>{cell(row, 'prediction_index')}<br>{cell(row, 'prediction_name')}</td>",
                    f"<td>conf={cell(row, 'confidence')}<br>target_p={cell(row, 'target_probability')}<br>margin={cell(row, 'top2_margin')}</td>",
                    f"<td>{cell(row, 'boundary_pair')}<br>{cell(row, 'buckets')}</td>",
                    f"<td>{cell(row, 'manual_label_status')}<br>{cell(row, 'review_notes')}</td>",
                    "</tr>",
                ]
            )
        )

    payload = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; font-size: 13px; }}
    th {{ background: #f3f5f7; position: sticky; top: 0; }}
    img {{ object-fit: contain; background: #111; }}
    pre {{ background: #f6f8fa; padding: 10px; overflow-x: auto; }}
    .missing {{ color: #b00020; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p><b>Manifest:</b> {html.escape(str(Path(manifest).resolve()))}</p>
  <p><b>Rows:</b> {len(rows)} total, {len(selected)} rendered. Manual decisions must be written back to CSV, not this HTML.</p>
  <h2>Summary</h2>
  <pre>{html.escape(json.dumps({"by_reason": by_reason, "by_boundary_pair": by_pair, "by_bucket": by_bucket}, indent=2, ensure_ascii=False))}</pre>
  <h2>Cases</h2>
  <table>
    <thead>
      <tr>
        <th>Image</th>
        <th>Review</th>
        <th>Target</th>
        <th>Prediction</th>
        <th>Scores</th>
        <th>Boundary / Buckets</th>
        <th>Manual</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_html.write_text(payload, encoding="utf-8")
    summary = {
        "manifest": str(Path(manifest).resolve()),
        "output_html": str(output_html.resolve()),
        "input_rows": int(len(rows)),
        "rendered_rows": int(len(selected)),
        "copied_image_matches": int(sum(1 for row in selected if str(row.get("review_id", "") or "") in image_map)),
        "by_reason": by_reason,
        "by_boundary_pair": by_pair,
        "by_bucket": by_bucket,
    }
    (output_html.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a boundary review CSV as a local HTML contact sheet.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Boundary Review")
    parser.add_argument("--max-rows", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = render_boundary_review_html(
        manifest=args.manifest,
        output_html=args.output_html,
        title=args.title,
        max_rows=args.max_rows,
        image_width=args.image_width,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

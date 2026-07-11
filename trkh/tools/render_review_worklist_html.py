from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _split_columns(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = [value]
    else:
        chunks = list(value)
    columns: List[str] = []
    for chunk in chunks:
        for part in str(chunk).split(","):
            column = part.strip()
            if column:
                columns.append(column)
    return columns


def _safe_float(value: object) -> float | None:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return None


def _natural_sort_value(row: Mapping[str, str], column: str) -> tuple[int, object]:
    value = str(row.get(column, "") or "").strip()
    numeric = _safe_float(value)
    if numeric is not None:
        return (0, numeric)
    return (1, value.lower())


def _sort_rows(rows: Sequence[Mapping[str, str]], sort_by: Sequence[str]) -> List[Mapping[str, str]]:
    selected = list(rows)
    for raw_column in reversed(list(sort_by)):
        reverse = raw_column.startswith("-")
        column = raw_column[1:] if reverse else raw_column
        selected.sort(key=lambda row, col=column: _natural_sort_value(row, col), reverse=reverse)
    return selected


def _image_src(row: Mapping[str, str], image_column: str) -> str:
    image_value = str(row.get(image_column, "") or "").strip()
    if not image_value:
        return ""
    image_path = Path(image_value)
    if not image_path.is_file():
        return ""
    try:
        return html.escape(image_path.resolve().as_uri())
    except ValueError:
        return html.escape(str(image_path))


def _count_by(rows: Sequence[Mapping[str, str]], column: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, "") or "").strip()
        if not value:
            value = "<blank>"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _escape_cell(row: Mapping[str, str], column: str) -> str:
    return html.escape(str(row.get(column, "") or ""))


def render_review_worklist_html(
    *,
    csv_path: Path,
    output_html: Path,
    title: str,
    columns: Sequence[str],
    group_by: Sequence[str] = (),
    sort_by: Sequence[str] = (),
    image_column: str = "image_path",
    id_column: str = "review_id",
    manual_field: str = "manual_label_status",
    max_rows: int = 0,
    image_width: int = 180,
    leakage_guard: str = "",
) -> Dict[str, object]:
    rows, fieldnames = _read_csv(Path(csv_path))
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"Columns not found in {csv_path}: {', '.join(missing)}")
    if image_column not in fieldnames:
        raise ValueError(f"Image column not found in {csv_path}: {image_column}")

    ordered_rows = _sort_rows(rows, sort_by) if sort_by else list(rows)
    if int(max_rows) > 0:
        ordered_rows = ordered_rows[: int(max_rows)]

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    manual_filled = 0
    if manual_field in fieldnames:
        manual_filled = sum(1 for row in rows if str(row.get(manual_field, "") or "").strip())

    group_counts = {column: _count_by(rows, column) for column in group_by if column in fieldnames}
    missing_image_count = sum(1 for row in ordered_rows if not _image_src(row, image_column))

    cards: List[str] = []
    for index, row in enumerate(ordered_rows, start=1):
        image_src = _image_src(row, image_column)
        image_html = (
            f'<img src="{image_src}" width="{int(image_width)}" loading="lazy" />'
            if image_src
            else '<span class="missing">missing image</span>'
        )
        case_id = _escape_cell(row, id_column) if id_column in fieldnames else str(index)
        field_rows = "\n".join(
            f"<tr><th>{html.escape(column)}</th><td>{_escape_cell(row, column)}</td></tr>"
            for column in columns
        )
        cards.append(
            "\n".join(
                [
                    '<section class="case-card">',
                    f'<div class="image-box">{image_html}</div>',
                    '<div class="case-fields">',
                    f"<h2>{index}. {case_id}</h2>",
                    "<table>",
                    field_rows,
                    "</table>",
                    "</div>",
                    "</section>",
                ]
            )
        )

    summary = {
        "source_csv": str(Path(csv_path).resolve()),
        "output_html": str(output_html.resolve()),
        "input_rows": int(len(rows)),
        "rows_rendered": int(len(ordered_rows)),
        "manual_field": manual_field,
        "manual_label_status_filled": int(manual_filled),
        "missing_images_rendered": int(missing_image_count),
        "columns": list(columns),
        "group_counts": group_counts,
        "leakage_guard": leakage_guard,
    }

    payload = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; color: #222; }}
    .meta {{ background: #f6f8fa; border: 1px solid #ddd; padding: 10px; margin-bottom: 16px; }}
    .case-card {{ display: grid; grid-template-columns: {int(image_width) + 24}px minmax(0, 1fr); gap: 12px; border-top: 1px solid #ddd; padding: 12px 0; }}
    .image-box {{ min-height: 80px; display: flex; align-items: flex-start; justify-content: center; background: #111; padding: 6px; }}
    img {{ max-width: 100%; height: auto; object-fit: contain; }}
    h1 {{ font-size: 22px; margin: 0 0 10px; }}
    h2 {{ font-size: 15px; margin: 0 0 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 6px; vertical-align: top; font-size: 13px; }}
    th {{ width: 190px; background: #f3f5f7; text-align: left; }}
    pre {{ margin: 0; white-space: pre-wrap; }}
    .missing {{ color: #ffb4b4; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">
    <p><b>CSV:</b> {html.escape(str(Path(csv_path).resolve()))}</p>
    <p><b>Rows:</b> {len(rows)} total, {len(ordered_rows)} rendered. Manual decisions must be written back to CSV, not this HTML.</p>
    <p><b>Guard:</b> {html.escape(leakage_guard)}</p>
    <pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  </div>
  {''.join(cards)}
</body>
</html>
"""
    output_html.write_text(payload, encoding="utf-8")
    output_html.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a generic CSV review worklist as local HTML.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Review Worklist")
    parser.add_argument("--columns", nargs="+", required=True, help="Columns to display, comma-separated or repeated.")
    parser.add_argument("--group-by", nargs="*", default=(), help="Summary group columns, comma-separated or repeated.")
    parser.add_argument("--sort-by", nargs="*", default=(), help="Sort columns, prefix with '-' for descending.")
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--id-column", type=str, default="review_id")
    parser.add_argument("--manual-field", type=str, default="manual_label_status")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=180)
    parser.add_argument("--leakage-guard", type=str, default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = render_review_worklist_html(
        csv_path=args.csv,
        output_html=args.output_html,
        title=args.title,
        columns=_split_columns(args.columns),
        group_by=_split_columns(args.group_by),
        sort_by=_split_columns(args.sort_by),
        image_column=args.image_column,
        id_column=args.id_column,
        manual_field=args.manual_field,
        max_rows=args.max_rows,
        image_width=args.image_width,
        leakage_guard=args.leakage_guard,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

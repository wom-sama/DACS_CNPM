from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from trkh.core.config import load_data_spec


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_CROP_NAME = re.compile(r"^(?P<source>.+)_box(?P<box>\d+)$", re.IGNORECASE)
KNOWN_SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _path_value(row: Mapping[str, str]) -> str:
    for key in ("image_path", "path", "sample_path"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _split_from_path(path_text: str) -> str:
    parts = [part.lower() for part in str(path_text or "").replace("\\", "/").split("/") if part]
    for part in parts:
        if part in KNOWN_SPLITS:
            return KNOWN_SPLITS[part]
    return "unknown"


def _row_split(row: Mapping[str, str]) -> str:
    split = str(row.get("split", "") or "").strip().lower()
    if split in KNOWN_SPLITS:
        return KNOWN_SPLITS[split]
    return _split_from_path(_path_value(row))


def _safe_int(row: Mapping[str, str], keys: Sequence[str], default: int = -1) -> int:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return int(default)


def _safe_float(row: Mapping[str, str], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return float(parsed)
    return float(default)


def _safe_name(text: object) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return value.strip("._") or "unknown"


def _source_from_crop_path(path: Path) -> Tuple[str, int]:
    match = CLASS_CROP_NAME.match(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse classification crop name: {path.name}")
    return str(match.group("source")), int(match.group("box"))


def _find_image(images_dir: Path, source_id: str) -> Optional[Path]:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{source_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _load_yolo_objects(label_path: Path) -> List[Dict[str, object]]:
    objects: List[Dict[str, object]] = []
    if not label_path.is_file():
        return objects
    for index, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            label = int(parts[0])
            bbox = tuple(float(value) for value in parts[1:5])
        except ValueError:
            continue
        if any(value < 0.0 or value > 1.0 for value in bbox) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        objects.append({"object_index": int(index), "label": int(label), "bbox": bbox})
    return objects


def _bbox_xywh_to_xyxy_pixels(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(value) for value in bbox]
    x1 = max(0.0, (cx - bw / 2.0) * float(width))
    y1 = max(0.0, (cy - bh / 2.0) * float(height))
    x2 = min(float(width), (cx + bw / 2.0) * float(width))
    y2 = min(float(height), (cy + bh / 2.0) * float(height))
    return (
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    )


def _crop_with_margin(
    image: Image.Image,
    bbox: Sequence[float],
    margin_ratio: float,
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = _bbox_xywh_to_xyxy_pixels(bbox, width=width, height=height)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = int(math.ceil(float(box_width) * max(0.0, float(margin_ratio))))
    margin_y = int(math.ceil(float(box_height) * max(0.0, float(margin_ratio))))
    left = max(0, x1 - margin_x)
    top = max(0, y1 - margin_y)
    right = min(width, x2 + margin_x)
    bottom = min(height, y2 + margin_y)
    if right <= left or bottom <= top:
        return image.copy()
    return image.crop((left, top, right, bottom))


def _fit_image(image: Image.Image, size: Tuple[int, int], fill: Tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, fill)
    src = image.convert("RGB")
    src.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - src.width) // 2
    y = (size[1] - src.height) // 2
    canvas.paste(src, (x, y))
    return canvas


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_source_boxes(
    image: Image.Image,
    objects: Sequence[Mapping[str, object]],
    selected_index: int,
    class_names: Sequence[str],
) -> Image.Image:
    drawn = image.convert("RGB").copy()
    draw = ImageDraw.Draw(drawn)
    font = _font(14)
    width, height = drawn.size
    for obj in objects:
        bbox = obj.get("bbox", ())
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        object_index = int(obj.get("object_index", -1))
        label = int(obj.get("label", -1))
        color = (255, 203, 79) if object_index == int(selected_index) else (130, 190, 255)
        x1, y1, x2, y2 = _bbox_xywh_to_xyxy_pixels(bbox, width=width, height=height)
        for offset in range(3 if object_index == int(selected_index) else 1):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        label_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        tag = f"box{object_index:03d} c{label} {label_name}"
        text_box = draw.textbbox((x1 + 3, y1 + 3), tag, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1 + 3, y1 + 3), tag, fill=color, font=font)
    return drawn


def _render_composite(
    *,
    source_image: Path,
    crop_image: Path,
    yolo_crop: Image.Image,
    objects: Sequence[Mapping[str, object]],
    selected_index: int,
    class_names: Sequence[str],
    metadata_lines: Sequence[str],
    output_path: Path,
    source_panel_width: int,
    crop_panel_width: int,
) -> None:
    with Image.open(source_image) as src_handle:
        source = src_handle.convert("RGB")
    with Image.open(crop_image) as crop_handle:
        class_crop = crop_handle.convert("RGB")

    source_drawn = _draw_source_boxes(source, objects, selected_index, class_names)
    panel_height = int(max(260, min(520, round(source_panel_width * 0.75))))
    source_panel = _fit_image(source_drawn, (int(source_panel_width), panel_height), (12, 15, 18))
    class_crop_panel = _fit_image(class_crop, (int(crop_panel_width), panel_height), (18, 18, 18))
    yolo_crop_panel = _fit_image(yolo_crop, (int(crop_panel_width), panel_height), (18, 18, 18))

    title_height = 28
    meta_width = int(source_panel_width) + int(crop_panel_width) * 2
    font = _font(14)
    small_font = _font(13)
    scratch = Image.new("RGB", (meta_width, 20), (255, 255, 255))
    scratch_draw = ImageDraw.Draw(scratch)
    wrapped: List[str] = []
    for line in metadata_lines:
        wrapped.extend(_wrap_text(scratch_draw, line, small_font, meta_width - 24))
    meta_height = max(88, 18 * len(wrapped) + 22)

    canvas = Image.new("RGB", (meta_width, title_height + panel_height + meta_height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    headers = [
        ("YOLO source + boxes", 0, int(source_panel_width)),
        ("class_f crop", int(source_panel_width), int(crop_panel_width)),
        ("YOLO crop from selected box", int(source_panel_width) + int(crop_panel_width), int(crop_panel_width)),
    ]
    for text, x, w in headers:
        draw.rectangle((x, 0, x + w, title_height), fill=(32, 40, 48))
        draw.text((x + 8, 7), text, fill=(255, 255, 255), font=font)
    y = title_height
    canvas.paste(source_panel, (0, y))
    canvas.paste(class_crop_panel, (int(source_panel_width), y))
    canvas.paste(yolo_crop_panel, (int(source_panel_width) + int(crop_panel_width), y))
    meta_y = y + panel_height
    draw.rectangle((0, meta_y, meta_width, meta_y + meta_height), fill=(255, 255, 255))
    text_y = meta_y + 10
    for line in wrapped:
        draw.text((12, text_y), line, fill=(26, 32, 38), font=small_font)
        text_y += 18
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _focus_match(row: Mapping[str, str], focus_class_index: int) -> bool:
    if focus_class_index < 0:
        return True
    values = {
        _safe_int(row, ("target_index", "true_index"), -999),
        _safe_int(row, ("prediction_index", "pred_index"), -999),
        _safe_int(row, ("suggested_index",), -999),
        _safe_int(row, ("top2_index",), -999),
    }
    if int(focus_class_index) in values:
        return True
    boundary_pair = str(row.get("boundary_pair", "") or "")
    pair_values = {part.strip() for part in re.split(r"[-_/;,\s]+", boundary_pair) if part.strip()}
    return str(int(focus_class_index)) in pair_values


def _sort_key(row: Mapping[str, str]) -> Tuple[float, float, float, str]:
    issue_rank = _safe_float(row, ("cleanlab_issue_rank", "issue_rank"), default=1_000_000.0)
    severity = _safe_float(row, ("severity",), default=0.0)
    label_quality = _safe_float(row, ("cleanlab_label_quality", "label_quality"), default=1.0)
    review_id = str(row.get("review_id", "") or row.get("path", "") or row.get("image_path", ""))
    return (issue_rank, -severity, label_quality, review_id)


def _html_image_src(path: Path, output_html: Path) -> str:
    try:
        return html.escape(path.resolve().relative_to(output_html.parent.resolve()).as_posix())
    except ValueError:
        try:
            return html.escape(path.resolve().as_uri())
        except ValueError:
            return html.escape(str(path))


def _render_html(
    *,
    manifest: Path,
    output_html: Path,
    title: str,
    rows: Sequence[Mapping[str, object]],
    image_width: int,
) -> None:
    def cell(row: Mapping[str, object], key: str) -> str:
        value = row.get(key, "")
        if value is None:
            return ""
        return html.escape(str(value))

    table_rows: List[str] = []
    for row in rows:
        image_path = Path(str(row.get("context_image_path", "") or ""))
        if image_path.is_file():
            image_html = f'<img src="{_html_image_src(image_path, output_html)}" width="{int(image_width)}" loading="lazy" />'
        else:
            image_html = '<span class="missing">missing image</span>'
        table_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td>{image_html}</td>",
                    f"<td><b>{cell(row, 'review_id')}</b><br>{cell(row, 'split')}<br>{cell(row, 'reason')}</td>",
                    f"<td>{cell(row, 'target_index')}<br>{cell(row, 'target_name')}</td>",
                    f"<td>{cell(row, 'prediction_index')}<br>{cell(row, 'prediction_name')}</td>",
                    f"<td>{cell(row, 'source_id')} box{cell(row, 'box_index')}<br>YOLO c{cell(row, 'yolo_label_index')}<br>match={cell(row, 'class_label_matches_yolo')}</td>",
                    f"<td>{cell(row, 'manual_label_status')}<br>{cell(row, 'review_notes')}</td>",
                    "</tr>",
                ]
            )
        )

    summary = {
        "rows": len(rows),
        "by_boundary_pair": dict(Counter(str(row.get("boundary_pair", "") or "") for row in rows)),
        "by_reason": dict(Counter(str(row.get("reason", "") or "") for row in rows)),
        "label_mismatch_count": sum(1 for row in rows if str(row.get("class_label_matches_yolo", "")) == "False"),
    }
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
  <p><b>Manifest:</b> {html.escape(str(manifest.resolve()))}</p>
  <p>This is a train-only source-context review artifact. Write manual decisions back to the CSV if they become training policy.</p>
  <pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <table>
    <thead>
      <tr>
        <th>Source Context</th>
        <th>Review</th>
        <th>Target</th>
        <th>Prediction</th>
        <th>YOLO Mapping</th>
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


def build_yolo_source_context_review(
    *,
    review_csv: Path,
    yolo_data: Path,
    output_dir: Path,
    split: str = "train",
    focus_class_index: int = 1,
    max_rows: int = 160,
    crop_margin_ratio: float = 0.05,
    source_panel_width: int = 520,
    crop_panel_width: int = 260,
    image_width_html: int = 760,
) -> Dict[str, object]:
    rows, fieldnames = _read_csv(Path(review_csv))
    data_spec = load_data_spec(yolo_data, class_name_mode="raw", expected_num_classes=None)
    split_name = KNOWN_SPLITS.get(str(split).strip().lower(), str(split).strip().lower())
    images_dir = data_spec.split_images_dir(split_name)
    labels_dir = data_spec.split_labels_dir(split_name)
    output_dir = Path(output_dir)
    review_images_dir = output_dir / "source_context_images"

    selected: List[Mapping[str, str]] = []
    skipped_non_split = 0
    skipped_non_focus = 0
    for row in rows:
        if _row_split(row) != split_name:
            skipped_non_split += 1
            continue
        if not _focus_match(row, int(focus_class_index)):
            skipped_non_focus += 1
            continue
        selected.append(row)
    selected = sorted(selected, key=_sort_key)
    if int(max_rows) > 0:
        selected = selected[: int(max_rows)]
    if not selected:
        raise ValueError("No rows selected for YOLO source-context review.")

    output_rows: List[Dict[str, object]] = []
    skipped_missing = 0
    label_mismatch = 0
    for index, row in enumerate(selected, start=1):
        crop_path = Path(_path_value(row))
        try:
            source_id, box_index = _source_from_crop_path(crop_path)
        except ValueError:
            skipped_missing += 1
            continue
        source_image = _find_image(images_dir, source_id)
        label_path = labels_dir / f"{source_id}.txt"
        objects = _load_yolo_objects(label_path)
        if source_image is None or not crop_path.is_file() or box_index >= len(objects):
            skipped_missing += 1
            continue

        selected_object = objects[box_index]
        yolo_label = int(selected_object["label"])
        selected_bbox = selected_object["bbox"]  # type: ignore[assignment]
        bbox_area = float(selected_bbox[2]) * float(selected_bbox[3])  # type: ignore[index]
        target_index = _safe_int(row, ("target_index", "true_index"), default=-1)
        label_matches = bool(target_index == yolo_label)
        if not label_matches:
            label_mismatch += 1
        with Image.open(source_image) as source_handle:
            source_width, source_height = source_handle.size
            yolo_crop = _crop_with_margin(
                source_handle.convert("RGB"),
                selected_bbox,  # type: ignore[arg-type]
                margin_ratio=crop_margin_ratio,
            )

        review_id = str(row.get("review_id", "") or f"{split_name}_{index:05d}")
        pair = str(row.get("boundary_pair", "") or "unknown")
        reason = str(row.get("reason", "") or str(row.get("buckets", "") or "review"))
        output_image = (
            review_images_dir
            / _safe_name(pair)
            / f"{index:04d}_{_safe_name(review_id)}_{source_id}_box{box_index:03d}.jpg"
        )
        target_name = str(row.get("target_name", "") or row.get("true_name", ""))
        pred_name = str(row.get("prediction_name", "") or row.get("pred_name", ""))
        metadata_lines = [
            f"{review_id} | source={source_id} box={box_index:03d} | pair={pair} | reason={reason}",
            f"target={target_index} {target_name} | prediction={_safe_int(row, ('prediction_index', 'pred_index'), -1)} {pred_name}",
            f"confidence={_safe_float(row, ('confidence', 'top1_confidence', 'cleanlab_top1_confidence'), 0.0):.6f} "
            f"target_p={_safe_float(row, ('target_probability', 'self_confidence', 'cleanlab_self_confidence'), 0.0):.6f} "
            f"margin={_safe_float(row, ('top2_margin',), 0.0):.6f}",
            f"yolo_label={yolo_label} {data_spec.class_names[yolo_label] if 0 <= yolo_label < len(data_spec.class_names) else ''} "
            f"| label_matches_class_f={label_matches}",
            str(crop_path),
        ]
        _render_composite(
            source_image=source_image,
            crop_image=crop_path,
            yolo_crop=yolo_crop,
            objects=objects,
            selected_index=box_index,
            class_names=data_spec.class_names,
            metadata_lines=metadata_lines,
            output_path=output_image,
            source_panel_width=int(source_panel_width),
            crop_panel_width=int(crop_panel_width),
        )

        enriched = dict(row)
        enriched.update(
            {
                "review_id": review_id,
                "split": split_name,
                "source_id": source_id,
                "box_index": int(box_index),
                "source_image_path": str(source_image.resolve()),
                "source_label_path": str(label_path.resolve()),
                "context_image_path": str(output_image.resolve()),
                "source_width": int(source_width),
                "source_height": int(source_height),
                "yolo_object_count": int(len(objects)),
                "yolo_label_index": int(yolo_label),
                "yolo_label_name": (
                    data_spec.class_names[yolo_label] if 0 <= yolo_label < len(data_spec.class_names) else ""
                ),
                "yolo_bbox_xywh": " ".join(f"{float(value):.8f}" for value in selected_bbox),  # type: ignore[arg-type]
                "yolo_bbox_area": f"{bbox_area:.8f}",
                "class_label_matches_yolo": bool(label_matches),
                "manual_label_status": row.get("manual_label_status", ""),
                "manual_expected_class": row.get("manual_expected_class", ""),
                "quality_lighting": row.get("quality_lighting", ""),
                "quality_dirty_obstacle": row.get("quality_dirty_obstacle", ""),
                "quality_partial_fruit": row.get("quality_partial_fruit", ""),
                "quality_background_mask": row.get("quality_background_mask", ""),
                "review_notes": row.get("review_notes", ""),
            }
        )
        output_rows.append(enriched)

    if not output_rows:
        raise ValueError("Selected rows could not be mapped to YOLO source images.")

    extra_fields = [
        "source_id",
        "box_index",
        "source_image_path",
        "source_label_path",
        "context_image_path",
        "source_width",
        "source_height",
        "yolo_object_count",
        "yolo_label_index",
        "yolo_label_name",
        "yolo_bbox_xywh",
        "yolo_bbox_area",
        "class_label_matches_yolo",
    ]
    output_fieldnames = list(fieldnames)
    for field in extra_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)
    for field in (
        "manual_label_status",
        "manual_expected_class",
        "quality_lighting",
        "quality_dirty_obstacle",
        "quality_partial_fruit",
        "quality_background_mask",
        "review_notes",
    ):
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    manifest_path = output_dir / "source_context_review_manifest.csv"
    output_html = output_dir / "source_context_review.html"
    _write_csv(manifest_path, output_rows, output_fieldnames)
    _render_html(
        manifest=manifest_path,
        output_html=output_html,
        title="YOLO Source Context Boundary Review",
        rows=output_rows,
        image_width=int(image_width_html),
    )

    summary = {
        "review_csv": str(Path(review_csv).resolve()),
        "yolo_data": str(Path(yolo_data).resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "html": str(output_html.resolve()),
        "input_rows": int(len(rows)),
        "selected_rows": int(len(selected)),
        "output_rows": int(len(output_rows)),
        "skipped_non_split": int(skipped_non_split),
        "skipped_non_focus": int(skipped_non_focus),
        "skipped_missing_mapping": int(skipped_missing),
        "label_mismatch_count": int(label_mismatch),
        "by_boundary_pair": dict(Counter(str(row.get("boundary_pair", "") or "") for row in output_rows)),
        "by_reason": dict(Counter(str(row.get("reason", "") or "") for row in output_rows)),
        "note": "Train-only source-context review artifact; do not use val/test rows to build training policy.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render train-only boundary review rows with YOLO source image, bbox, class_f crop, and YOLO crop."
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=160)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.05)
    parser.add_argument("--source-panel-width", type=int, default=520)
    parser.add_argument("--crop-panel-width", type=int, default=260)
    parser.add_argument("--image-width-html", type=int, default=760)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_yolo_source_context_review(
        review_csv=args.review_csv,
        yolo_data=args.yolo_data,
        output_dir=args.output_dir,
        split=args.split,
        focus_class_index=args.focus_class_index,
        max_rows=args.max_rows,
        crop_margin_ratio=args.crop_margin_ratio,
        source_panel_width=args.source_panel_width,
        crop_panel_width=args.crop_panel_width,
        image_width_html=args.image_width_html,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
from pathlib import Path

from trkh.tools.render_boundary_review_html import render_boundary_review_html


def test_render_boundary_review_html_maps_copied_review_image(tmp_path: Path) -> None:
    manifest = tmp_path / "boundary_review_manifest.csv"
    image_dir = tmp_path / "review_images" / "train" / "0-1" / "focus_false_positive"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "train_00001_t0_p1_m0.010.jpg"
    image_path.write_bytes(b"fake-jpg")
    fieldnames = [
        "review_id",
        "split",
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "confidence",
        "target_probability",
        "top2_margin",
        "boundary_pair",
        "reason",
        "severity",
        "buckets",
        "manual_label_status",
        "review_notes",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "review_id": "train_00001",
                "split": "train",
                "image_path": str(image_path),
                "target_index": "0",
                "target_name": "class0",
                "prediction_index": "1",
                "prediction_name": "class1",
                "confidence": "0.31",
                "target_probability": "0.21",
                "top2_margin": "0.01",
                "boundary_pair": "0-1",
                "reason": "focus_false_positive",
                "severity": "5.0",
                "buckets": "0->1;over_bright_or_glare",
            }
        )

    output_html = tmp_path / "boundary_review_report.html"
    summary = render_boundary_review_html(
        manifest=manifest,
        output_html=output_html,
        title="Test Review",
        max_rows=10,
        image_width=96,
    )

    html_text = output_html.read_text(encoding="utf-8")
    assert summary["input_rows"] == 1
    assert summary["copied_image_matches"] == 1
    assert "train_00001_t0_p1_m0.010.jpg" in html_text
    assert "focus_false_positive" in html_text

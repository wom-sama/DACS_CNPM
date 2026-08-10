import csv
from pathlib import Path

import pytest

from trkh.tools.render_review_worklist_html import render_review_worklist_html


def test_render_review_worklist_html_counts_manual_and_groups(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-jpg")
    csv_path = tmp_path / "review.csv"
    fieldnames = [
        "review_id",
        "image_path",
        "transition",
        "strict_review_side",
        "manual_label_status",
        "effv2_review_order",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "review_id": "case_b",
                "image_path": str(image_path),
                "transition": "0->1",
                "strict_review_side": "fp_suppressor",
                "manual_label_status": "",
                "effv2_review_order": "2",
            }
        )
        writer.writerow(
            {
                "review_id": "case_a",
                "image_path": str(image_path),
                "transition": "1->0",
                "strict_review_side": "recall_protector",
                "manual_label_status": "correct",
                "effv2_review_order": "1",
            }
        )

    output_html = tmp_path / "review.html"
    summary = render_review_worklist_html(
        csv_path=csv_path,
        output_html=output_html,
        title="Generic Review",
        columns=["review_id", "transition", "strict_review_side", "manual_label_status"],
        group_by=["strict_review_side"],
        sort_by=["effv2_review_order"],
        image_width=96,
        leakage_guard="review-only",
    )

    html_text = output_html.read_text(encoding="utf-8")
    assert summary["input_rows"] == 2
    assert summary["rows_rendered"] == 2
    assert summary["manual_label_status_filled"] == 1
    assert summary["missing_images_rendered"] == 0
    assert summary["group_counts"]["strict_review_side"]["recall_protector"] == 1
    assert html_text.index("case_a") < html_text.index("case_b")
    assert "review-only" in html_text


def test_render_review_worklist_html_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    csv_path.write_text("review_id,image_path\ncase_a,missing.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Columns not found"):
        render_review_worklist_html(
            csv_path=csv_path,
            output_html=tmp_path / "review.html",
            title="Generic Review",
            columns=["review_id", "transition"],
        )

import csv
from pathlib import Path

from trkh.tools.build_spectral_margin_manifest import build_manifest


def _write_scores(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "ambiguity_score",
        "neighbor_same_label_fraction",
        "neighbor_top_rival_label",
        "centroid_margin",
    ]
    rows = [
        {
            "sample_index": "10",
            "image_path": "train/a.jpg",
            "target_index": "0",
            "base_prediction": "1",
            "ambiguity_score": "0.91",
            "neighbor_same_label_fraction": "0.0",
            "neighbor_top_rival_label": "1",
            "centroid_margin": "-0.2",
        },
        {
            "sample_index": "11",
            "image_path": "train/b.jpg",
            "target_index": "1",
            "base_prediction": "0",
            "ambiguity_score": "0.82",
            "neighbor_same_label_fraction": "0.1",
            "neighbor_top_rival_label": "0",
            "centroid_margin": "-0.1",
        },
        {
            "sample_index": "12",
            "image_path": "train/c.jpg",
            "target_index": "4",
            "base_prediction": "4",
            "ambiguity_score": "0.90",
            "neighbor_same_label_fraction": "0.2",
            "neighbor_top_rival_label": "2",
            "centroid_margin": "-0.1",
        },
        {
            "sample_index": "13",
            "image_path": "train/d.jpg",
            "target_index": "0",
            "base_prediction": "0",
            "ambiguity_score": "0.30",
            "neighbor_same_label_fraction": "1.0",
            "neighbor_top_rival_label": "1",
            "centroid_margin": "0.5",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_spectral_margin_manifest_uses_sample_index_and_pair_caps(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    output = tmp_path / "manifest.csv"
    _write_scores(scores)

    summary = build_manifest(
        scores=scores,
        output=output,
        min_ambiguity_nonfocus=0.78,
        min_ambiguity_focus=0.74,
        max_samples=10,
    )

    assert summary["rows"] == 2
    assert summary["by_target_negative_pair"] == {"0->1": 1, "1->0": 1}
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_index"] for row in rows] == ["10", "11"]
    assert rows[0]["reason"] == "spectral_focus_false_positive_risk"
    assert rows[1]["reason"] == "spectral_focus_false_negative_risk"

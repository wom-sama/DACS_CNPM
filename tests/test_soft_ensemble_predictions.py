import csv
from pathlib import Path

from trkh.tools.soft_ensemble_predictions import run_soft_ensemble


def _write_predictions(path: Path, rows):
    fieldnames = [
        "sample_index",
        "path",
        "target_index",
        "target_name",
        "prob_0_A",
        "prob_1_B",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_soft_ensemble_aligns_by_sample_index_and_weights_probabilities(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_predictions(
        first,
        [
            {
                "sample_index": "0",
                "path": "image0.jpg",
                "target_index": "1",
                "target_name": "B",
                "prob_0_A": "0.80",
                "prob_1_B": "0.20",
            },
            {
                "sample_index": "1",
                "path": "image1.jpg",
                "target_index": "0",
                "target_name": "A",
                "prob_0_A": "0.90",
                "prob_1_B": "0.10",
            },
        ],
    )
    _write_predictions(
        second,
        [
            {
                "sample_index": "1",
                "path": "image1_alt.jpg",
                "target_index": "0",
                "target_name": "A",
                "prob_0_A": "0.70",
                "prob_1_B": "0.30",
            },
            {
                "sample_index": "0",
                "path": "image0_alt.jpg",
                "target_index": "1",
                "target_name": "B",
                "prob_0_A": "0.10",
                "prob_1_B": "0.90",
            },
        ],
    )

    summary = run_soft_ensemble(
        inputs=[f"first={first}", f"second={second}"],
        weights_text="first:0.25,second:0.75",
        output_dir=tmp_path / "ensemble",
    )

    assert summary["records"] == 2
    assert summary["metrics"]["accuracy"] == 1.0
    assert summary["metrics"]["per_class"][1]["f1"] == 1.0

    output_rows = list(
        csv.DictReader((tmp_path / "ensemble" / "predictions_detailed.csv").open())
    )
    assert [row["sample_index"] for row in output_rows] == ["0", "1"]
    assert output_rows[0]["prediction_index"] == "1"
    assert abs(float(output_rows[0]["prob_1_B"]) - 0.725) < 1e-12


def test_soft_ensemble_rejects_duplicate_keys(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    _write_predictions(
        duplicate,
        [
            {
                "sample_index": "0",
                "path": "a.jpg",
                "target_index": "0",
                "target_name": "A",
                "prob_0_A": "1.0",
                "prob_1_B": "0.0",
            },
            {
                "sample_index": "0",
                "path": "b.jpg",
                "target_index": "0",
                "target_name": "A",
                "prob_0_A": "1.0",
                "prob_1_B": "0.0",
            },
        ],
    )

    try:
        run_soft_ensemble(
            inputs=[f"dup={duplicate}"],
            weights_text="",
            output_dir=tmp_path / "ensemble",
        )
    except ValueError as exc:
        assert "Duplicate sample_index" in str(exc)
    else:
        raise AssertionError("expected duplicate key rejection")

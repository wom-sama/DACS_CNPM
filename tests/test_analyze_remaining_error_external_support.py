import csv
from pathlib import Path

from trkh.tools.analyze_remaining_error_external_support import parse_args, run


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_remaining_error_external_support_uses_sample_index_and_counts_consensus(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    remaining = tmp_path / "remaining.csv"
    ext_a = tmp_path / "ext_a.csv"
    ext_b = tmp_path / "ext_b.csv"
    rows = [
        {"sample_index": 0, "target_index": 1, "prediction_index": 0, "confidence": 0.6, "prob_0": 0.6, "prob_1": 0.4},
        {"sample_index": 1, "target_index": 0, "prediction_index": 1, "confidence": 0.7, "prob_0": 0.3, "prob_1": 0.7},
        {"sample_index": 2, "target_index": 0, "prediction_index": 0, "confidence": 0.8, "prob_0": 0.8, "prob_1": 0.2},
    ]
    _write_csv(base, rows)
    _write_csv(
        remaining,
        [
            {"sample_index": 0, "target_index": 1, "prediction_index": 0, "margin": 0.2},
            {"sample_index": 1, "target_index": 0, "prediction_index": 1, "margin": 0.4},
        ],
    )
    _write_csv(
        ext_a,
        [
            {"sample_index": 0, "target_index": 1, "prediction_index": 1, "prob_0": 0.1, "prob_1": 0.9},
            {"sample_index": 1, "target_index": 0, "prediction_index": 0, "prob_0": 0.8, "prob_1": 0.2},
            {"sample_index": 2, "target_index": 0, "prediction_index": 1, "prob_0": 0.4, "prob_1": 0.6},
        ],
    )
    _write_csv(
        ext_b,
        [
            {"sample_index": 0, "target_index": 1, "prediction_index": 1, "prob_0": 0.2, "prob_1": 0.8},
            {"sample_index": 1, "target_index": 0, "prediction_index": 0, "prob_0": 0.7, "prob_1": 0.3},
            {"sample_index": 2, "target_index": 0, "prediction_index": 0, "prob_0": 0.9, "prob_1": 0.1},
        ],
    )
    out = tmp_path / "out"

    summary = run(
        parse_args(
            [
                "--base-csv",
                str(base),
                "--remaining-errors-csv",
                str(remaining),
                "--external",
                f"a={ext_a}",
                "--external",
                f"b={ext_b}",
                "--output-dir",
                str(out),
                "--focus-class-index",
                "1",
                "--consensus-min-votes",
                "2",
            ]
        )
    )

    assert summary["remaining_error_support_by_bucket"]["focus_false_negative"]["consensus_correct"] == 1
    assert summary["remaining_error_support_by_bucket"]["focus_false_positive"]["consensus_correct"] == 1
    assert summary["diagnostic_focus_consensus_route"]["stats"]["corrections"] == 2
    assert summary["diagnostic_focus_consensus_route"]["stats"]["harms"] == 0
    assert (out / "summary.json").is_file()
    assert (out / "remaining_error_external_support.csv").is_file()

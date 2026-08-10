import csv
import json

import pytest

from trkh.tools.audit_prediction_transitions import audit_prediction_transitions


FIELDS = [
    "sample_index",
    "image_path",
    "target_index",
    "prediction_index",
    "prob_1_focus",
]


def _write(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_prediction_transition_audit_counts_focus_outcomes(tmp_path) -> None:
    base = tmp_path / "base.csv"
    candidate = tmp_path / "candidate.csv"
    image_paths = [str(tmp_path / f"image_{index}.jpg") for index in range(6)]
    targets = [1, 1, 0, 4, 2, 3]
    base_predictions = [0, 1, 1, 4, 0, 3]
    candidate_predictions = [1, 0, 0, 1, 2, 3]
    _write(
        base,
        [
            {
                "sample_index": index,
                "image_path": image_paths[index],
                "target_index": targets[index],
                "prediction_index": base_predictions[index],
                "prob_1_focus": 0.2,
            }
            for index in range(6)
        ],
    )
    _write(
        candidate,
        [
            {
                "sample_index": index,
                "image_path": image_paths[index],
                "target_index": targets[index],
                "prediction_index": candidate_predictions[index],
                "prob_1_focus": 0.3,
            }
            for index in range(6)
        ],
    )

    summary = audit_prediction_transitions(
        base_predictions=base,
        candidate_predictions=candidate,
        output_dir=tmp_path / "audit",
        focus_class=1,
    )

    assert summary["rows"] == 6
    assert summary["changed"] == 5
    assert summary["corrections"] == 3
    assert summary["harms"] == 2
    assert summary["wrong_to_wrong"] == 0
    assert summary["focus"] == {
        "base_tp": 1,
        "base_fp": 1,
        "base_fn": 1,
        "candidate_tp": 1,
        "candidate_fp": 1,
        "candidate_fn": 1,
        "fn_rescued": 1,
        "tp_broken": 1,
        "fp_removed": 1,
        "fp_created": 1,
    }
    persisted = json.loads((tmp_path / "audit" / "summary.json").read_text())
    assert persisted["net_corrections"] == 1


def test_prediction_transition_audit_rejects_misaligned_rows(tmp_path) -> None:
    base = tmp_path / "base.csv"
    candidate = tmp_path / "candidate.csv"
    _write(
        base,
        [
            {
                "sample_index": 0,
                "image_path": str(tmp_path / "a.jpg"),
                "target_index": 0,
                "prediction_index": 0,
                "prob_1_focus": 0.1,
            }
        ],
    )
    _write(
        candidate,
        [
            {
                "sample_index": 1,
                "image_path": str(tmp_path / "a.jpg"),
                "target_index": 0,
                "prediction_index": 0,
                "prob_1_focus": 0.1,
            }
        ],
    )

    with pytest.raises(ValueError, match="do not align"):
        audit_prediction_transitions(
            base_predictions=base,
            candidate_predictions=candidate,
            output_dir=tmp_path / "audit",
        )

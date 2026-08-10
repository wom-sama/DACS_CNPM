from __future__ import annotations

import csv

import pytest

from trkh.tools import replay_bi_level_routing_attention_preflight as replay


def _row(*, sample_index: int, valid: bool = True) -> dict[str, object]:
    return {
        "condition": "clean",
        "sample_index": sample_index,
        "valid_object_query": valid,
        "foreground_gain": 0.04 if valid else float("nan"),
        "far_background_reduction": 0.03 if valid else float("nan"),
        "routed_foreground_fraction": 0.30 if valid else float("nan"),
        "all_region_foreground_fraction": 0.26,
        "routed_far_background_fraction": 0.55 if valid else float("nan"),
        "all_region_far_background_fraction": 0.58,
        "distinct_route_sets": 6.0,
        "pairwise_route_jaccard": 0.50,
        "nonlocal_route_fraction": 0.40,
        "probability_mae": 0.003,
        "clean_route_jaccard": "",
        "selected_affinity_margin": 1.0,
        "bbox_area": 0.10,
        "bbox_edge_gap": 0.05,
        "bbox_center_edge_distance": 0.20,
    }


def test_parse_selectivity_rows_preserves_invalid_geometry(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    rows = [_row(sample_index=7), _row(sample_index=9, valid=False)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = replay._parse_selectivity_rows(path)

    assert list(grouped) == ["clean"]
    assert grouped["clean"][0]["sample_index"] == 7
    assert grouped["clean"][0]["valid_object_query"] is True
    assert grouped["clean"][1]["valid_object_query"] is False
    assert grouped["clean"][1]["foreground_gain"] != grouped["clean"][1][
        "foreground_gain"
    ]


def test_corrected_selectivity_uses_finite_rows_but_keeps_coverage_gate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(replay, "EXPECTED_HOLDOUT_ROWS", 2)
    clean = [_row(sample_index=7), _row(sample_index=9, valid=False)]
    shifted = []
    for index in range(2):
        row = _row(sample_index=index)
        row["clean_route_jaccard"] = 0.70
        shifted.append(row)
    grouped = {
        "clean": clean,
        "lighting_dim": shifted,
        "lighting_bright": shifted,
        "low_contrast": shifted,
    }

    result = replay._corrected_selectivity(
        {"selectivity": {"rows_csv_sha256": "locked"}}, grouped
    )

    assert result["conditions"]["clean"]["finite_rows"] == 1
    assert result["conditions"]["clean"]["foreground_gain_mean"] == pytest.approx(
        0.04
    )
    assert result["checks"]["clean_positive_gain_fraction"] is True
    assert result["checks"]["all_object_queries_valid"] is False
    assert result["checks"]["all_rows_finite"] is False
    assert result["invalid_object_query_sample_indices"] == [9]

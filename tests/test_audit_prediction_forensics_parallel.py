from __future__ import annotations

import csv
import time

import pytest

from trkh.tools import audit_prediction_forensics as module


def test_ordered_image_stats_preserves_input_order(monkeypatch) -> None:
    delays = {"slow": 0.03, "fast": 0.0, "medium": 0.01}

    def fake_image_stats(path: str, foreground_margin: float, mode: str):
        time.sleep(delays[path])
        return {
            "image_status": "ok",
            "path": path,
            "margin": foreground_margin,
            "mode": mode,
        }

    monkeypatch.setattr(module, "_image_stats", fake_image_stats)
    rows = module._ordered_image_stats(
        ["slow", "fast", "medium"],
        foreground_margin=0.08,
        mode="foreground",
        max_rows=0,
        workers=3,
    )

    assert [row["path"] for row in rows] == ["slow", "fast", "medium"]
    assert all(row["margin"] == 0.08 for row in rows)
    assert all(row["mode"] == "foreground" for row in rows)


def test_ordered_image_stats_respects_limit_without_reading_tail(monkeypatch) -> None:
    seen = []

    def fake_image_stats(path: str, foreground_margin: float, mode: str):
        seen.append(path)
        return {"image_status": "ok", "path": path}

    monkeypatch.setattr(module, "_image_stats", fake_image_stats)
    rows = module._ordered_image_stats(
        ["first", "second", "third"],
        foreground_margin=0.08,
        mode="basic",
        max_rows=2,
        workers=1,
    )

    assert seen == ["first", "second"]
    assert rows[-1] == {"image_status": "skipped_by_limit"}


def _write_cache(path, rows) -> None:
    fieldnames = ["sample_index", "image_path", *module.IMAGE_STAT_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_image_stats_cache_reuses_exact_ordered_cohort(tmp_path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    cache = tmp_path / "cache.csv"
    source_rows = [
        {"sample_index": "7", "image_path": str(first)},
        {"sample_index": "9", "image_path": str(second)},
    ]
    cached_rows = []
    for index, source in enumerate(source_rows):
        cached_rows.append(
            {
                **source,
                **{field: f"{field}-{index}" for field in module.IMAGE_STAT_FIELDS},
            }
        )
    _write_cache(cache, cached_rows)

    loaded = module._load_image_stats_cache(cache, source_rows)

    assert loaded[0]["brightness_mean"] == "brightness_mean-0"
    assert loaded[1]["image_status"] == "image_status-1"


def test_image_stats_cache_rejects_sample_mismatch(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    cache = tmp_path / "cache.csv"
    source_rows = [{"sample_index": "1", "image_path": str(image)}]
    _write_cache(
        cache,
        [
            {
                "sample_index": "2",
                "image_path": str(image),
                **{field: "0" for field in module.IMAGE_STAT_FIELDS},
            }
        ],
    )

    with pytest.raises(ValueError, match="sample_index mismatch"):
        module._load_image_stats_cache(cache, source_rows)

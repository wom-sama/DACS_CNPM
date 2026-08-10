import json
from pathlib import Path

from PIL import Image

from trkh.tools.summarize_xai_transitions import summarize_xai_transitions


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_summarize_xai_transitions_groups_metrics_and_contact_sheet(tmp_path: Path) -> None:
    image_paths = {}
    for index, color in enumerate(((240, 30, 30), (30, 180, 30), (30, 30, 220)), start=1):
        crop = tmp_path / f"case{index}" / "crop.png"
        gradcam = tmp_path / f"case{index}" / "gradcam_overlay.png"
        rollout = tmp_path / f"case{index}" / "rollout_overlay.png"
        _write_image(crop, color)
        _write_image(gradcam, color)
        _write_image(rollout, color)
        image_paths[index] = {"crop": crop, "gradcam": gradcam, "rollout": rollout}

    xai_summary = tmp_path / "xai_audit_summary.json"
    xai_summary.write_text(
        json.dumps(
            {
                "split": "val",
                "cases": [
                    {
                        "sample_index": 10,
                        "target_index": 1,
                        "prediction_index": 2,
                        "confidence": 0.31,
                        "review_flags": ["gradcam_border_attention", "object_color_sensitive"],
                        "robustness": {
                            "background_blur": {"original_prediction_drop": 0.01},
                            "background_gray": {"original_prediction_drop": 0.02},
                            "object_desaturate": {"original_prediction_drop": 0.20},
                        },
                        "viz": {
                            "heatmap_focus": {
                                "gradcam": {"foreground_mass": 0.91, "border_mass": 0.30},
                                "rollout": {"border_mass": 0.20},
                            },
                            "files": {
                                "crop": str(image_paths[1]["crop"]),
                                "gradcam": {"overlay": str(image_paths[1]["gradcam"])},
                                "rollout": {"overlay": str(image_paths[1]["rollout"])},
                            },
                        },
                    },
                    {
                        "sample_index": 11,
                        "target_index": 1,
                        "prediction_index": 2,
                        "confidence": 0.28,
                        "review_flags": ["gradcam_border_attention"],
                        "robustness": {
                            "background_blur": {"original_prediction_drop": 0.03},
                            "object_desaturate": {"original_prediction_drop": 0.10},
                        },
                        "viz": {
                            "heatmap_focus": {
                                "gradcam": {"foreground_mass": 0.95, "border_mass": 0.40},
                                "rollout": {"border_mass": 0.25},
                            },
                            "files": {
                                "crop": str(image_paths[2]["crop"]),
                                "gradcam": {"overlay": str(image_paths[2]["gradcam"])},
                                "rollout": {"overlay": str(image_paths[2]["rollout"])},
                            },
                        },
                    },
                    {
                        "sample_index": 12,
                        "target_index": 2,
                        "prediction_index": 1,
                        "confidence": 0.34,
                        "review_flags": ["rollout_background_attention"],
                        "robustness": {
                            "background_blur": {"original_prediction_drop": -0.01},
                            "object_desaturate": {"original_prediction_drop": 0.15},
                        },
                        "viz": {
                            "heatmap_focus": {
                                "gradcam": {"foreground_mass": 0.80, "border_mass": 0.10},
                                "rollout": {"border_mass": 0.22},
                            },
                            "files": {
                                "crop": str(image_paths[3]["crop"]),
                                "gradcam": {"overlay": str(image_paths[3]["gradcam"])},
                                "rollout": {"overlay": str(image_paths[3]["rollout"])},
                            },
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = summarize_xai_transitions(
        xai_summary_json=xai_summary,
        output_dir=tmp_path / "out",
        thumbnail_size=64,
        contact_sheet_columns=2,
        leakage_guard="diagnostic-only",
    )

    assert summary["selected_cases"] == 3
    assert summary["transition_count"] == 2
    assert summary["test_split_used"] is False
    rows = {row["transition"]: row for row in summary["transition_rows"]}
    assert rows["1->2"]["cases"] == 2
    assert rows["1->2"]["background_blur_original_prediction_drop_mean"] == 0.02
    assert rows["1->2"]["object_desaturate_original_prediction_drop_mean"] == 0.15000000000000002
    assert rows["1->2"]["gradcam_foreground_mass_mean"] == 0.9299999999999999
    assert "gradcam_border_attention:2" in rows["1->2"]["top_flags"]
    assert rows["2->1"]["cases"] == 1
    assert Path(summary["transition_summary_csv"]).is_file()
    assert Path(summary["contact_sheet"]["contact_sheet"]).is_file()
    assert summary["contact_sheet"]["missing_images"] == 0
    assert "diagnostic-only" in (tmp_path / "out" / "transition_xai_summary.json").read_text(encoding="utf-8")

import csv

from trkh.tools.build_patch_oof_sample_weights import build_manifest


def test_build_manifest_selects_confident_oof_disagreements(tmp_path) -> None:
    teacher = tmp_path / "teacher.csv"
    with teacher.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "path", "target_index", "prob_0", "prob_1", "prob_2"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_index": 0,
                "path": "a.jpg",
                "target_index": 0,
                "prob_0": 0.20,
                "prob_1": 0.80,
                "prob_2": 0.0,
            }
        )
        writer.writerow(
            {
                "sample_index": 1,
                "path": "b.jpg",
                "target_index": 1,
                "prob_0": 0.55,
                "prob_1": 0.45,
                "prob_2": 0.0,
            }
        )
        writer.writerow(
            {
                "sample_index": 2,
                "path": "c.jpg",
                "target_index": 1,
                "prob_0": 0.10,
                "prob_1": 0.90,
                "prob_2": 0.0,
            }
        )

    summary = build_manifest(
        teacher_csv=teacher,
        output_dir=tmp_path / "out",
        pair=(0, 1),
        disagreement_confidence_threshold=0.65,
        disagreement_weight=0.35,
    )

    assert summary["selected_rows"] == 1
    assert summary["by_target_prediction"] == {"0->1": 1}
    with (tmp_path / "out" / "patch_oof_sample_weights_train_only.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_index"] == "0"
    assert rows[0]["sample_weight"] == "0.35"
    assert rows[0]["reason"] == "patch_oof_confident_disagreement"

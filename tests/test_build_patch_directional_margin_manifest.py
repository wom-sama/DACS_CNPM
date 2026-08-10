import csv
from pathlib import Path

from trkh.tools.build_patch_directional_margin_manifest import build_manifest
from trkh.training.train import _load_targeted_margin_manifest


def _write_changed_cases(path: Path) -> None:
    path.parent.mkdir(parents=True)
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "change_type",
        "change_pair",
        "direction",
        "verifier_confidence",
        "pair_min_probability",
        "pair_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "sample_index": "7",
                "image_path": str(path.parent / "images" / "train" / "a.jpg"),
                "target_index": "0",
                "base_prediction": "1",
                "final_prediction": "0",
                "change_type": "correction",
                "change_pair": "0-1",
                "direction": "1->0",
                "verifier_confidence": "0.91",
                "pair_min_probability": "0.22",
                "pair_margin": "0.03",
            }
        )
        writer.writerow(
            {
                "split": "train",
                "sample_index": "8",
                "image_path": str(path.parent / "images" / "train" / "b.jpg"),
                "target_index": "1",
                "base_prediction": "1",
                "final_prediction": "0",
                "change_type": "harm",
                "change_pair": "0-1",
                "direction": "1->0",
                "verifier_confidence": "0.88",
                "pair_min_probability": "0.23",
                "pair_margin": "0.01",
            }
        )
        writer.writerow(
            {
                "split": "val",
                "sample_index": "9",
                "image_path": str(path.parent / "images" / "val" / "c.jpg"),
                "target_index": "0",
                "base_prediction": "1",
                "final_prediction": "0",
                "change_type": "correction",
                "change_pair": "0-1",
                "direction": "1->0",
                "verifier_confidence": "0.95",
                "pair_min_probability": "0.21",
                "pair_margin": "0.02",
            }
        )


def test_build_patch_directional_margin_manifest_is_train_only_and_sample_indexed(tmp_path):
    changed_cases = tmp_path / "gate" / "train" / "changed_cases_for_xai.csv"
    output = tmp_path / "targeted_margin.csv"
    _write_changed_cases(changed_cases)

    summary = build_manifest(
        changes_csv=changed_cases,
        output=output,
        focus_class=1,
        directions="1->0",
        suppressor_margin=0.06,
        suppressor_weight=1.5,
        protector_margin=0.04,
        protector_weight=0.8,
        include_protectors=True,
    )

    assert summary["rows"] == 2
    assert summary["by_reason"] == {
        "patch_gate_false_positive_suppressor": 1,
        "patch_gate_class1_recall_protector": 1,
    }
    assert summary["skipped"] == {"non_train": 1}

    by_path, by_sample_index, load_summary = _load_targeted_margin_manifest(
        str(output),
        num_classes=5,
        default_margin=0.12,
        default_weight=1.0,
        max_weight=2.0,
        return_sample_index_specs=True,
    )

    assert by_path == {}
    assert set(by_sample_index) == {7, 8}
    assert by_sample_index[7]["negative_index"] == 1
    assert by_sample_index[8]["negative_index"] == 0
    assert load_summary["key_mode"] == "sample_index"

from pathlib import Path
from types import SimpleNamespace

import torch

from trkh.evaluation.evaluate import _build_prediction_records, _dataset_sample_metadata


def test_prediction_records_include_yolo_object_metadata() -> None:
    dataset = SimpleNamespace(
        samples=[
            SimpleNamespace(
                image_path=Path("fold/images/val/img_001.jpg"),
                label_path=Path("fold/labels/val/img_001.txt"),
                primary_label=1,
                primary_object_index=2,
                objects=[
                    SimpleNamespace(label=0, object_index=0, bbox=(0.1, 0.2, 0.3, 0.4)),
                    SimpleNamespace(label=1, object_index=2, bbox=(0.5, 0.6, 0.2, 0.3)),
                ],
            )
        ]
    )

    metadata = _dataset_sample_metadata(dataset)
    records = _build_prediction_records(
        targets=torch.tensor([1]),
        predictions=torch.tensor([1]),
        probabilities=torch.tensor([[0.1, 0.8, 0.05, 0.03, 0.02]]),
        class_names=["c0", "c1", "c2", "c3", "c4"],
        sample_paths=["fold/images/val/img_001.jpg"],
        sample_metadata=metadata,
    )

    assert records[0]["source_stem"] == "img_001"
    assert records[0]["object_index"] == 2
    assert records[0]["primary_label"] == 1
    assert records[0]["label_path"].endswith("img_001.txt")
    assert records[0]["bbox_0"] == 0.5

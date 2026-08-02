from __future__ import annotations

import json
from pathlib import Path

import pytest

from trkh.tools.assess_pretrained_classf_probe import (
    assess_b10,
    assess_hybrid_v2,
    load_val_metrics,
    main,
)


def _metrics(
    *,
    macro: float,
    class1_f1: float,
    class1_precision: float = 0.70,
    class1_recall: float = 0.75,
    fp_0: int = 10,
    fp_2: int = 20,
    fp_4: int = 5,
) -> dict[str, object]:
    confusion = [[0] * 5 for _ in range(5)]
    for index in range(5):
        confusion[index][index] = 50
    confusion[0][1] = fp_0
    confusion[2][1] = fp_2
    confusion[4][1] = fp_4
    return {
        "macro_f1": macro,
        "per_class": [
            {
                "class_index": index,
                "precision": class1_precision if index == 1 else 0.9,
                "recall": class1_recall if index == 1 else 0.9,
                "f1": class1_f1 if index == 1 else 0.9,
            }
            for index in range(5)
        ],
        "confusion_matrix": confusion,
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_b10_gate_requires_class1_and_2_to1_gain_without_macro_loss(
    tmp_path: Path,
) -> None:
    control = load_val_metrics(
        _write(tmp_path / "best_val" / "b2.json", _metrics(macro=0.85, class1_f1=0.65))
    )
    candidate = load_val_metrics(
        _write(
            tmp_path / "best_val" / "b10.json",
            _metrics(macro=0.851, class1_f1=0.66, fp_2=18),
        )
    )

    assert assess_b10(candidate, control)["metric_gate_passed"] is True
    candidate["macro_f1"] = 0.849
    assert assess_b10(candidate, control)["metric_gate_passed"] is False


def test_hybrid_gate_is_conjunctive_against_both_controls(tmp_path: Path) -> None:
    pure = load_val_metrics(
        _write(tmp_path / "best_val" / "pure.json", _metrics(macro=0.86, class1_f1=0.70))
    )
    generic = load_val_metrics(
        _write(
            tmp_path / "best_val" / "generic.json",
            _metrics(macro=0.861, class1_f1=0.705, fp_0=9, fp_2=19, fp_4=5),
        )
    )
    candidate = load_val_metrics(
        _write(
            tmp_path / "best_val" / "local.json",
            _metrics(macro=0.87, class1_f1=0.72, fp_0=7, fp_2=14, fp_4=4),
        )
    )

    result = assess_hybrid_v2(
        candidate,
        {"pure_dino": pure, "generic_adapter": generic},
    )
    assert result["metric_gate_passed"] is True

    candidate["class1_f1"] = 0.719
    assert assess_hybrid_v2(
        candidate,
        {"pure_dino": pure, "generic_adapter": generic},
    )["metric_gate_passed"] is False


def test_cli_writes_hashed_val_only_verdict_and_never_authorizes_full(
    tmp_path: Path,
) -> None:
    control = _write(
        tmp_path / "best_val" / "b2.json",
        _metrics(macro=0.85, class1_f1=0.65),
    )
    candidate = _write(
        tmp_path / "best_val" / "b10.json",
        _metrics(macro=0.851, class1_f1=0.66, fp_2=18),
    )
    output = tmp_path / "audit" / "verdict.json"

    main(
        [
            "--protocol",
            "b10",
            "--candidate",
            str(candidate),
            "--control",
            f"b2={control}",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metric_gate_passed"] is True
    assert payload["test_split_opened"] is False
    assert payload["full_train_authorized_by_this_artifact"] is False
    assert len(payload["candidate"]["sha256"]) == 64


def test_test_path_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "test" / "metrics.json", _metrics(macro=0.8, class1_f1=0.6))
    with pytest.raises(ValueError, match="validation artifacts only"):
        load_val_metrics(path)

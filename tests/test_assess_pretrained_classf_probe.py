from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trkh.tools.assess_pretrained_classf_probe import (
    CANONICAL_CLASS_NAMES,
    CANONICAL_VAL_SUPPORTS,
    assess_b10,
    assess_hybrid_v2,
    assess_hybrid_v3,
    load_val_metrics,
    main,
    validate_hybrid_v3_comparison_contract,
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


def test_hybrid_v3_gate_locks_pair_control_safety_and_branch_contribution(
    tmp_path: Path,
) -> None:
    b2 = load_val_metrics(
        _write(tmp_path / "val_only" / "b2.json", _metrics(macro=0.85, class1_f1=0.68))
    )
    generic = load_val_metrics(
        _write(
            tmp_path / "val_only" / "generic.json",
            _metrics(macro=0.851, class1_f1=0.685, fp_0=10, fp_2=20, fp_4=5),
        )
    )
    branch_off = load_val_metrics(
        _write(
            tmp_path / "val_only" / "off.json",
            _metrics(macro=0.85, class1_f1=0.690, fp_0=10, fp_2=20, fp_4=5),
        )
    )
    candidate = load_val_metrics(
        _write(
            tmp_path / "val_only" / "candidate.json",
            _metrics(macro=0.852, class1_f1=0.720, fp_0=8, fp_2=16, fp_4=5),
        )
    )
    candidate["surface_pair_residual_ratio_p95"] = 0.039
    result = assess_hybrid_v3(
        candidate,
        {"b2": b2, "pair_generic": generic, "branch_off": branch_off},
    )
    assert result["metric_gate_passed"] is True

    candidate["class1_f1"] = 0.719999
    assert assess_hybrid_v3(
        candidate,
        {"b2": b2, "pair_generic": generic, "branch_off": branch_off},
    )["metric_gate_passed"] is False
    candidate["class1_f1"] = 0.720

    candidate["class2_to_1_fp"] = 17
    assert assess_hybrid_v3(
        candidate,
        {"b2": b2, "pair_generic": generic, "branch_off": branch_off},
    )["metric_gate_passed"] is False


def _canonical_v3_artifact(
    root: Path,
    *,
    name: str,
    checkpoint: Path,
    mode: str,
    branch_off: bool,
) -> dict[str, object]:
    artifact_dir = root / "val_only" / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    confusion = [[0] * 5 for _ in range(5)]
    rows = []
    for class_index, support in enumerate(CANONICAL_VAL_SUPPORTS):
        confusion[class_index][class_index] = int(support)
        for sample_index in range(int(support)):
            rows.append(
                {
                    "path": (
                        f"D:\\canonical\\val\\{CANONICAL_CLASS_NAMES[class_index]}"
                        f"\\image_{sample_index:04d}.jpg"
                    ),
                    "y_true": class_index,
                    "y_pred": class_index,
                    "true_name": CANONICAL_CLASS_NAMES[class_index],
                    "pred_name": CANONICAL_CLASS_NAMES[class_index],
                }
            )
    with (artifact_dir / "predictions_val.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "checkpoint": str(checkpoint),
        "split": "val",
        "samples": sum(CANONICAL_VAL_SUPPORTS),
        "classes": list(CANONICAL_CLASS_NAMES),
        "model_type": (
            "dinov3_surface_pair_hybrid_v3" if mode else "timm_classifier"
        ),
        "dinov3_surface_hybrid_mode": mode,
        "surface_pair_branch_off": branch_off,
        "surface_pair_routing_audit": {
            "statistics": {"residual_to_token_ratio": {"p95": 0.039}}
        },
        "macro_f1": 1.0,
        "per_class": [
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": CANONICAL_VAL_SUPPORTS[class_index],
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
            for class_index, class_name in enumerate(CANONICAL_CLASS_NAMES)
        ],
        "confusion_matrix": confusion,
    }
    return load_val_metrics(_write(artifact_dir / "metrics_val.json", payload))


def test_hybrid_v3_comparison_contract_binds_full_val_identity_and_branch_off(
    tmp_path: Path,
) -> None:
    candidate_checkpoint = tmp_path / "candidate.pt"
    generic_checkpoint = tmp_path / "generic.pt"
    b2_checkpoint = tmp_path / "b2.pt"
    candidate_checkpoint.write_bytes(b"candidate")
    generic_checkpoint.write_bytes(b"generic")
    b2_checkpoint.write_bytes(b"b2")
    candidate = _canonical_v3_artifact(
        tmp_path,
        name="candidate",
        checkpoint=candidate_checkpoint,
        mode="relative_surface_pair",
        branch_off=False,
    )
    controls = {
        "b2": _canonical_v3_artifact(
            tmp_path,
            name="b2",
            checkpoint=b2_checkpoint,
            mode="",
            branch_off=False,
        ),
        "pair_generic": _canonical_v3_artifact(
            tmp_path,
            name="generic",
            checkpoint=generic_checkpoint,
            mode="generic_token_pair",
            branch_off=False,
        ),
        "branch_off": _canonical_v3_artifact(
            tmp_path,
            name="branch_off",
            checkpoint=candidate_checkpoint,
            mode="relative_surface_pair",
            branch_off=True,
        ),
    }
    validate_hybrid_v3_comparison_contract(candidate, controls)
    assert len({item["sample_identity_sha256"] for item in [candidate, *controls.values()]}) == 1

    prediction_path = Path(str(controls["branch_off"]["path"])).with_name(
        "predictions_val.csv"
    )
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    prediction_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full canonical validation split"):
        validate_hybrid_v3_comparison_contract(candidate, controls)


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

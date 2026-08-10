from __future__ import annotations

from pathlib import Path

import torch

from trkh.tools.audit_dinov3_convnext_b24 import (
    FEATURE_DIM,
    IMAGE_SIZE,
    MODEL_WEIGHT_SHA256,
    _build_convnext,
    _sha256,
    assess_gate,
)


WEIGHT = Path(
    r"C:\Users\ADMIN\.cache\huggingface\hub\models--timm--convnext_tiny.dinov3_lvd1689m"
    r"\snapshots\39c7bb32f83c11bbb13cb23058f8852a4dac88ed\model.safetensors"
)


def _summary(*, macro: float, class1: float, pair: float, recall: float, fp: float):
    return {
        "macro_f1": macro,
        "class1_f1": class1,
        "mean_pair_auroc": pair,
        "class1_recall": recall,
        "restricted_fp_rate": fp,
        "folds": [{"fold": fold, "class1_f1": class1} for fold in range(5)],
    }


def test_strict_local_convnext_weight_and_feature_shape() -> None:
    assert WEIGHT.is_file()
    assert _sha256(WEIGHT) == MODEL_WEIGHT_SHA256
    model = _build_convnext(WEIGHT)
    with torch.inference_mode():
        feature = model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert tuple(feature.shape) == (1, FEATURE_DIM)
    assert torch.isfinite(feature).all()


def test_gate_requires_real_gain_and_all_guards() -> None:
    dino = _summary(macro=0.80, class1=0.52, pair=0.94, recall=0.70, fp=0.04)
    fusion = _summary(macro=0.805, class1=0.535, pair=0.944, recall=0.69, fp=0.042)
    bootstrap = {
        "intervals": {
            "class1_f1_delta": {"lower": -0.010},
            "macro_f1_delta": {"lower": -0.004},
            "mean_pair_auroc_delta": {"lower": -0.002},
        }
    }
    passed = assess_gate(
        dino=dino,
        fusion=fusion,
        bootstrap=bootstrap,
        integrity_complete=True,
        readouts_converged=True,
    )
    assert passed["signal_gate_passed"] is True
    assert passed["validation_permission"] is False
    assert passed["test_permission"] is False

    no_gain = dict(fusion)
    no_gain["macro_f1"] = 0.801
    failed = assess_gate(
        dino=dino,
        fusion=no_gain,
        bootstrap=bootstrap,
        integrity_complete=True,
        readouts_converged=True,
    )
    assert failed["signal_gate_passed"] is False
    assert "macro_gain" in failed["failed_checks"]

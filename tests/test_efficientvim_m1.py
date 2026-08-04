from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from trkh.models.efficientvim_m1 import (
    OFFICIAL_M1_E450_SHA256,
    EfficientViMM1,
    load_official_efficientvim_m1_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    os.environ.get(
        "TRKH_EFFICIENTVIM_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "runs"
            / "pretrained_assets"
            / "efficientvim_m1"
            / "official_EfficientViM_M1_e450.pth"
        ),
    )
)
OFFICIAL_REPOSITORY_ROOT = Path(
    os.environ.get(
        "TRKH_EFFICIENTVIM_OFFICIAL_ROOT",
        str(Path(os.environ.get("TEMP", "")) / "TRKH_EfficientViM_audit_20260803"),
    )
)
OFFICIAL_CLONE = OFFICIAL_REPOSITORY_ROOT / "classification" / "models"


def test_shapes_and_parameter_count() -> None:
    model = EfficientViMM1(num_classes=1000).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_679_458

    sample = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        features = model.forward_multistage_features(sample)
        final = model.forward_final_features(sample)
        logits = model(sample)

    assert tuple(feature.shape for feature in features) == (
        (1, 128),
        (1, 192),
        (1, 320),
        (1, 320),
    )
    assert final.shape == (1, 320)
    assert logits.shape == (1, 1000)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "shape",
    [
        (3, 224, 224),
        (1, 1, 224, 224),
        (1, 3, 224, 223),
        (1, 3, 192, 192),
    ],
)
def test_invalid_input_guard(shape: tuple[int, ...]) -> None:
    model = EfficientViMM1(num_classes=5).eval()
    with pytest.raises(ValueError, match="224"):
        model(torch.randn(shape))


def test_hidden_length_must_be_a_perfect_square() -> None:
    model = EfficientViMM1(num_classes=5).eval()
    mixer = model.stages[0].blocks[0].mixer
    with pytest.raises(ValueError, match="perfect square"):
        mixer(torch.randn(1, 128, 195))


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="official checkpoint is absent")
def test_safe_strict_official_checkpoint_load_has_no_persistent_globals() -> None:
    model = EfficientViMM1(num_classes=1000)
    safe_globals_before = list(torch.serialization.get_safe_globals())
    yacs_before = sys.modules.get("yacs")
    yacs_config_before = sys.modules.get("yacs.config")

    digest = load_official_efficientvim_m1_weights(model, CHECKPOINT)

    assert digest == OFFICIAL_M1_E450_SHA256
    assert list(torch.serialization.get_safe_globals()) == safe_globals_before
    assert sys.modules.get("yacs") is yacs_before
    assert sys.modules.get("yacs.config") is yacs_config_before
    assert len(model.state_dict()) == 307


def _load_official_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    package_name = "_trkh_test_official_efficientvim_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(OFFICIAL_CLONE)]
    monkeypatch.setitem(sys.modules, package_name, package)

    fvcore = types.ModuleType("fvcore")
    fvcore_nn = types.ModuleType("fvcore.nn")
    fvcore_nn.flop_count = lambda *args, **kwargs: ({}, {})
    fvcore.nn = fvcore_nn
    monkeypatch.setitem(sys.modules, "fvcore", fvcore)
    monkeypatch.setitem(sys.modules, "fvcore.nn", fvcore_nn)

    import timm.models

    monkeypatch.setattr(timm.models, "register_model", lambda function: function)
    for module_name, filename in (
        ("utils", "utils.py"),
        ("EfficientViM", "EfficientViM.py"),
    ):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, OFFICIAL_CLONE / filename
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import official EfficientViM {filename}.")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified_name, module)
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.EfficientViM"]


@pytest.mark.skipif(
    not CHECKPOINT.is_file() or not (OFFICIAL_CLONE / "EfficientViM.py").is_file(),
    reason="official checkpoint or temporary official clone is absent",
)
def test_official_synthetic_forward_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    official_module = _load_official_module(monkeypatch)
    candidate = EfficientViMM1(num_classes=1000).eval()
    load_official_efficientvim_m1_weights(candidate, CHECKPOINT)

    official = official_module.EfficientViM_M1(
        num_classes=1000, distillation=False
    ).eval()
    official.load_state_dict(candidate.state_dict(), strict=True)

    torch.manual_seed(20260803)
    sample = torch.randn(1, 3, 224, 224)
    captured: list[torch.Tensor] = []
    hook = official.heads[3].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    with torch.inference_mode():
        expected = official(sample)
        actual = candidate(sample)
        final = candidate.forward_final_features(sample)
    hook.remove()

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(final, captured[0], rtol=0, atol=1e-6)

from __future__ import annotations

import sys

import pytest

from trkh.tools.export_aidt_predictions import parse_args as parse_aidt_args
from trkh.tools.export_timm_predictions import parse_args as parse_timm_args


def test_timm_export_requires_explicit_test_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_timm_predictions.py",
            "--checkpoint",
            "missing.pt",
            "--data",
            "missing_data",
            "--split",
            "test",
            "--output-dir",
            "missing_output",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_timm_args()

    assert exc_info.value.code == 2
    assert "--split test requires the explicit --allow-test acknowledgement" in capsys.readouterr().err

def test_timm_export_allows_validation_without_test_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_timm_predictions.py",
            "--checkpoint",
            "checkpoint.pt",
            "--data",
            "dataset",
            "--split",
            "val",
            "--output-dir",
            "output",
        ],
    )

    args = parse_timm_args()

    assert args.split == "val"
    assert args.allow_test is False


def test_aidt_export_requires_explicit_test_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_aidt_predictions.py",
            "--checkpoint",
            "missing.pt",
            "--data",
            "missing_data",
            "--split",
            "test",
            "--output-dir",
            "missing_output",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_aidt_args()

    assert exc_info.value.code == 2
    assert "--split test requires the explicit --allow-test acknowledgement" in capsys.readouterr().err

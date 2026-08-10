from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ERRATUM = (
    ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_PREIMPLEMENTATION_ERRATUM_20260725.md"
)
ERRATUM_SHA = ERRATUM.with_suffix(".sha256")
LOCK = (
    ROOT
    / "docs"
    / "TRKH_5CLASS_CROSS_COLOUR_RATIO_SURFACE_A0_LOCK_20260725.json"
)
EXPECTED_PROTOCOL_SHA = (
    "1e408502c3fc20031d397d83bf24afb6c63aece539dda1bb8260160fc4eef347"
)
EXPECTED_LOCK_SHA = (
    "6c2a604deb4f976a3556ffb0ad8ddb60c7f376e81e374ae16272c6e6d49fc835"
)
EXPECTED_SOURCE_SHAS = {
    ROOT / "trkh" / "data" / "dataset.py": (
        "7ea29b5f3146995b63179600a5d31f338d154164221f1179b3352846ba4275d5"
    ),
    ROOT / "trkh" / "tools" / "build_precision_ensemble_checkpoint.py": (
        "481b7c1cdc0add86541e4be08175c3a42031d942cf2c86ed6d9e89320f43c7fe"
    ),
    ROOT / "trkh" / "evaluation" / "input_normalization.py": (
        "513f452063260c2f7724d40c621b4b34506e2463bfb78670f8cfb7d58c995e2f"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_ccr_erratum_is_bound_to_the_frozen_prospective_lock() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert lock["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA
    assert _sha256(LOCK) == EXPECTED_LOCK_SHA
    assert lock["state"] == "prospective_no_candidate_observation"
    assert all(value is None for value in lock["observations"].values())
    assert lock["repository"]["head_at_lock_build"] == lock["lock_parent_commit"]
    assert lock["repository"]["upstream_at_lock_build"] == lock["lock_parent_commit"]


def test_ccr_erratum_pins_exact_keeper_eval_semantics() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    config_path = Path(lock["immutable_inputs"]["keeper_config"]["path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model_config"]
    augmentation = config["augmentation_config"]

    assert model["image_size"] == 256
    assert model["input_mean"] == [0.485, 0.456, 0.406]
    assert model["input_std"] == [0.229, 0.224, 0.225]
    assert augmentation["crop_margin_ratio"] == 0.05
    assert augmentation["resize_mode"] == "pad"
    assert augmentation["illumination_normalization"] is True
    assert augmentation["illumination_normalization_strength"] == 0.35
    assert augmentation["background_suppression_mode"] == "desaturate_blur"
    assert augmentation["background_suppression_margin"] == 0.08
    assert augmentation["background_suppression_blur_radius"] == 7.0
    assert augmentation["foreground_crop_mode"] == "none"
    assert augmentation["surface_detail_amplification_mode"] == "none"
    assert augmentation["eval_surface_detail_amplification"] is False


def test_ccr_erratum_sources_and_digest_are_exact() -> None:
    for path, expected in EXPECTED_SOURCE_SHAS.items():
        assert _sha256(path) == expected

    parts = ERRATUM_SHA.read_text(encoding="ascii").split()
    assert parts == [_sha256(ERRATUM), ERRATUM.name]

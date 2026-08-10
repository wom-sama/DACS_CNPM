from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from trkh.core.utils import json_dump
from trkh.tools.audit_friendly_foreground_adversarial_readiness import sha256_file
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)


EXPECTED_SUMMARY_SHA256 = (
    "7ea4683bdb9edbd57cd3c7bb88a6ca29b6db13260de8588bcc6d293e4d0d1b57"
)
EXPECTED_ROWS_SHA256 = (
    "bfc051961520ebcf80cbdec9fc161f3ba49828218f53f3d30df7ded33d61b5b1"
)
EXPECTED_CONTACT_SHA256 = (
    "a374385511398349ee47fb4cb3830ac31a46fbaa309a6b19cb7157f693a6ce39"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash-locked manual closure for the LP-A3-inspired A0."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm-visual-review",
        action="store_true",
        default=False,
        help="Confirm that the fixed contact sheet was manually inspected.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def candidate_feature_difference_count(
    rows: Sequence[Mapping[str, str]],
) -> int:
    candidate = {
        int(row["sample_index"]): row
        for row in rows
        if str(row["variant"]) == "candidate"
    }
    feature = {
        int(row["sample_index"]): row
        for row in rows
        if str(row["variant"]) == "feature_only"
    }
    if candidate.keys() != feature.keys():
        raise ValueError("candidate and feature-only sample coverage differ")
    differences = 0
    for sample_index, candidate_row in candidate.items():
        feature_row = feature[sample_index]
        if (
            candidate_row["aug_prediction_index"]
            != feature_row["aug_prediction_index"]
            or candidate_row["selected_step"] != feature_row["selected_step"]
            or abs(
                float(candidate_row["feature_distance"])
                - float(feature_row["feature_distance"])
            )
            > 1e-8
        ):
            differences += 1
    return int(differences)


def finalize(args: argparse.Namespace) -> Mapping[str, object]:
    if not bool(args.confirm_visual_review):
        raise ValueError("--confirm-visual-review is required")
    audit_dir = Path(args.audit_dir).resolve()
    summary_path = audit_dir / "summary.json"
    rows_path = audit_dir / "hard_positive_rows.csv"
    contact_path = audit_dir / "contact_sheet.png"
    hashes = {
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
        "contact_sheet_sha256": sha256_file(contact_path),
    }
    expected = {
        "summary_sha256": EXPECTED_SUMMARY_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "contact_sheet_sha256": EXPECTED_CONTACT_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"formal artifact hashes differ: {hashes}")
    summary = _load_json(summary_path)
    if summary.get("validation_split_used") is not False:
        raise ValueError("formal summary does not prove validation isolation")
    if summary.get("test_split_used") is not False:
        raise ValueError("formal summary does not prove test isolation")
    gate = summary.get("gate")
    if not isinstance(gate, Mapping) or gate.get("automated_a0_permission") is not False:
        raise ValueError("formal automated gate is not the expected rejection")
    variants = summary.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("variant summaries are missing")
    candidate = variants.get("candidate")
    feature_only = variants.get("feature_only")
    if not isinstance(candidate, Mapping) or not isinstance(feature_only, Mapping):
        raise ValueError("candidate/control summaries are missing")
    rows = _load_rows(rows_path)
    difference_count = candidate_feature_difference_count(rows)
    if difference_count != 6:
        raise ValueError(
            f"expected six candidate-vs-feature-only differences, got {difference_count}"
        )

    visual_review = {
        "mode": "lpa3_inspired_hard_positive_a0_manual_visual_review",
        "formal_hashes": hashes,
        "confirmed": True,
        "reviewed_rows": 12,
        "natural_scale_identity_preserved": True,
        "gross_rendering_artifacts": False,
        "structured_surface_or_lesion_signal": False,
        "delta_character": (
            "Dense high-frequency speckle spread through the valid bbox; no "
            "stable concentration on lesions, ripeness patches, stems, or boundaries."
        ),
        "transition_review": (
            "Fixed rows include useful restricted-FP corrections but also class-1 "
            "TP breaks and clean-rival-to-class1 creations."
        ),
        "xai_escalation": (
            "Not opened: the prospective causal and safety gates already fail, "
            "and the amplified input-space attribution is diffuse."
        ),
        "passed": False,
    }
    json_dump(audit_dir / "visual_review.json", dict(visual_review))
    decision = {
        "mode": "lpa3_inspired_hard_positive_a0_final_decision",
        "decision": "Rejected",
        "formal_hashes": hashes,
        "automated_gate": {
            "passed": int(gate["passed"]),
            "total": int(gate["total"]),
        },
        "candidate": {
            "class1_tp_retention": float(candidate["class1_tp_retention"]),
            "restricted_rival_retention": float(
                candidate["restricted_rival_retention"]
            ),
            "restricted_fp_rejection": float(
                candidate["restricted_fp_rejection"]
            ),
            "class1_tp_broken": int(candidate["class1_tp_broken"]),
            "restricted_fp_removed": int(candidate["restricted_fp_removed"]),
            "restricted_fp_created": int(candidate["restricted_fp_created"]),
        },
        "causal_control": {
            "candidate_feature_only_differing_rows": difference_count,
            "support": int(candidate["support"]),
            "candidate_tp_minus_feature_only": float(
                candidate["class1_tp_retention"]
            )
            - float(feature_only["class1_tp_retention"]),
            "candidate_fp_rejection_minus_feature_only": float(
                candidate["restricted_fp_rejection"]
            )
            - float(feature_only["restricted_fp_rejection"]),
        },
        "failed_locked_gates": [
            "restricted_rival_retention",
            "absolute_restricted_fp_rejection",
            "label_hinge_causal_tp_delta",
            "source_fold_stability",
            "manual_structured_signal_review",
        ],
        "closed_nearby_variants": [
            "epsilon",
            "steps",
            "label_margin",
            "lagrange_schedule",
            "initial_noise",
            "pooled_feature_layer",
            "bbox_erosion",
        ],
        "authorized_next_stage": None,
        "trainer_integration": False,
        "validation_or_test": False,
        "smoke_or_probe": False,
        "full_train": False,
        "current_best_update": False,
    }
    json_dump(audit_dir / "final_decision.json", decision)
    _write_artifact_manifest(
        audit_dir,
        mode="lpa3_inspired_hard_positive_a0_final_manifest",
    )
    return decision


def main() -> None:
    decision = finalize(parse_args())
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

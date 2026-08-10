from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch

from trkh.core.config import to_serializable
from trkh.tools.audit_soft_moe_patch_adapter_readiness import (
    _equation_diagnostics,
    _sha256,
)


LOCKED_SOURCE_SUMMARY_SHA256 = (
    "1a2fe8909e9e73539779fdbfee88f161ca9847bf5ee82a4a1629e92d78bd64ec"
)
EQUATION_CHECK = "fp32_bf16_equation_and_normalization"
MAX_EQUATION_ERROR = 1e-5


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct the Soft-MoE Stage-A mixed-precision equation assessment "
            "without changing decision evidence or authorizing Stage B."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/audit_soft_moe_patch_adapter_stage_a_20260715/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/audit_soft_moe_patch_adapter_stage_a_20260715/"
            "equation_assessment_correction.json"
        ),
    )
    return parser.parse_args(argv)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        to_serializable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _equation_passed(equation: Mapping[str, object]) -> bool:
    rows = []
    for precision in ("fp32", "bf16"):
        value = equation.get(precision)
        if not isinstance(value, Mapping):
            return False
        rows.append(value)
    errors = [
        float(value)
        for row in rows
        for key, value in row.items()
        if str(key).endswith("_error")
    ]
    return bool(errors) and max(errors) <= MAX_EQUATION_ERROR and all(
        bool(row.get("prefix_bit_exact", False))
        and bool(row.get("finite", False))
        for row in rows
    )


def build_correction(
    summary: Mapping[str, object],
    *,
    equation: Mapping[str, object],
    source_summary_sha256: str,
    original_auditor_sha256: str,
    corrected_auditor_sha256: str,
) -> Dict[str, object]:
    if bool(summary.get("validation_data_used", True)):
        raise ValueError("Source summary used validation data.")
    if bool(summary.get("test_data_used", True)):
        raise ValueError("Source summary used test data.")
    gate = summary.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Source summary lacks a gate payload.")
    structural = gate.get("structural_checks")
    failed = gate.get("failed_checks")
    if not isinstance(structural, Mapping) or not isinstance(failed, list):
        raise ValueError("Source gate payload is invalid.")
    if bool(structural.get(EQUATION_CHECK, True)) or EQUATION_CHECK not in failed:
        raise ValueError("Source summary does not contain the locked equation failure.")
    if not _equation_passed(equation):
        raise ValueError("Corrected same-precision equation diagnostics did not pass.")

    corrected_gate = copy.deepcopy(dict(gate))
    corrected_structural = dict(structural)
    corrected_structural[EQUATION_CHECK] = True
    corrected_failed = [str(value) for value in failed if value != EQUATION_CHECK]
    corrected_gate["structural_checks"] = corrected_structural
    corrected_gate["failed_checks"] = corrected_failed
    corrected_gate["all_gates_passed"] = not corrected_failed
    corrected_gate["stage_b_smoke_authorized"] = not corrected_failed
    corrected_gate["full_train_authorized"] = False

    decision_payload = {
        "adapted_candidate_vs_raw": summary.get("adapted_candidate_vs_raw"),
        "adapted_candidate_vs_control": summary.get(
            "adapted_candidate_vs_control"
        ),
        "illumination_candidate_vs_raw": summary.get(
            "illumination_candidate_vs_raw"
        ),
        "illumination_candidate_vs_control": summary.get(
            "illumination_candidate_vs_control"
        ),
    }
    return {
        "method": "soft_moe_patch_adapter",
        "correction": "same_precision_bf16_equation_recomposition",
        "source_summary_sha256": str(source_summary_sha256),
        "original_auditor_sha256": str(original_auditor_sha256),
        "corrected_auditor_sha256": str(corrected_auditor_sha256),
        "validation_data_used": False,
        "test_data_used": False,
        "original_gate": gate,
        "corrected_equation": equation,
        "corrected_gate": corrected_gate,
        "decision_evidence_sha256": _canonical_sha256(decision_payload),
        "decision_evidence_changed": False,
        "outcome_changed": bool(gate.get("stage_b_smoke_authorized", False))
        != bool(corrected_gate["stage_b_smoke_authorized"]),
        "note": (
            "The original BF16 slot reference was recomposed outside autocast. "
            "The corrected reference uses the same precision as the module. "
            "Behavioral failures remain and Stage B stays forbidden."
        ),
    }


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Correction output already exists: {output_path}")
    source_sha = _sha256(summary_path)
    if source_sha != LOCKED_SOURCE_SUMMARY_SHA256:
        raise ValueError(
            f"Source summary SHA mismatch: {source_sha} != "
            f"{LOCKED_SOURCE_SUMMARY_SHA256}"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Correction requires BF16-capable CUDA.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit_path = (
        Path(__file__).resolve().parent
        / "audit_soft_moe_patch_adapter_readiness.py"
    )
    original_auditor_sha = str(
        summary["sources"]["implementation_sha256"]["auditor"]
    )
    corrected_auditor_sha = _sha256(audit_path)
    equation = _equation_diagnostics(torch.device("cuda"), torch.bfloat16)
    correction = build_correction(
        summary,
        equation=equation,
        source_summary_sha256=source_sha,
        original_auditor_sha256=original_auditor_sha,
        corrected_auditor_sha256=corrected_auditor_sha,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_serializable(correction), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": _sha256(output_path),
                "failed_checks": correction["corrected_gate"]["failed_checks"],
                "stage_b_smoke_authorized": correction["corrected_gate"][
                    "stage_b_smoke_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

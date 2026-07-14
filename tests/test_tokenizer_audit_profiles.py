from __future__ import annotations

from trkh.tools.audit_tokenizer_postsmoke import parse_args as parse_postsmoke_args
from trkh.tools.audit_tokenizer_trace import parse_args as parse_trace_args
from trkh.tools.build_tokenizer_xai_cohort import parse_args as parse_cohort_args


def test_starnet_trace_profile_is_explicit() -> None:
    args = parse_trace_args(
        [
            "--trace-summary",
            "trace.json",
            "--checkpoint",
            "best.pt",
            "--stage-a-summary",
            "stage_a.json",
            "--output-dir",
            "out",
            "--expected-method",
            "starnet_s2_local_multiplicative_tokenizer",
            "--expected-stem",
            "starnet_s2_tokenizer",
            "--expected-stem-channels",
            "128",
            "--mode",
            "starnet_s2_architecture_trace_audit",
        ]
    )

    assert args.expected_method == "starnet_s2_local_multiplicative_tokenizer"
    assert args.expected_stem == "starnet_s2_tokenizer"
    assert args.expected_stem_channels == 128
    assert args.mode == "starnet_s2_architecture_trace_audit"


def test_starnet_cohort_and_postsmoke_modes_are_explicit() -> None:
    cohort = parse_cohort_args(
        [
            "--changed-cases",
            "changed.csv",
            "--output-dir",
            "cohort",
            "--mode",
            "validation_only_starnet_s2_changed_case_cohort",
        ]
    )
    postsmoke = parse_postsmoke_args(
        [
            "--pair-summary",
            "pair.json",
            "--trace-audit",
            "trace.json",
            "--paired-xai",
            "xai.json",
            "--robustness-control",
            "robustness_control",
            "--robustness-candidate",
            "robustness_candidate",
            "--forensics-control",
            "forensics_control.json",
            "--forensics-candidate",
            "forensics_candidate.json",
            "--confusions-control",
            "confusions_control.json",
            "--confusions-candidate",
            "confusions_candidate.json",
            "--output-dir",
            "postsmoke",
            "--expected-method",
            "starnet_s2_local_multiplicative_tokenizer",
            "--mode",
            "starnet_s2_postsmoke_closure",
        ]
    )

    assert cohort.mode == "validation_only_starnet_s2_changed_case_cohort"
    assert postsmoke.expected_method == "starnet_s2_local_multiplicative_tokenizer"
    assert postsmoke.mode == "starnet_s2_postsmoke_closure"

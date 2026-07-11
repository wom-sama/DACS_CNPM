import numpy as np
import pandas as pd

from trkh.tools.stack_patch_feature_verifiers import (
    _proposal_and_candidates,
    _target_for_mode,
)


def test_proposal_prefers_patch_when_both_verifiers_change() -> None:
    df = pd.DataFrame(
        {
            "patch_base_prediction": [1, 1, 0],
            "patch_final_prediction": [0, 0, 0],
            "feature_final_prediction": [0, 1, 1],
        }
    )

    proposal, candidates, source = _proposal_and_candidates(df)

    assert proposal.tolist() == [0, 0, 1]
    assert candidates.tolist() == [True, True, True]
    assert source.tolist() == [3, 2, 1]


def test_class1_benefit_target_counts_corrections_and_false_class1_suppression() -> None:
    targets = np.array([1, 0, 4, 1])
    base = np.array([0, 1, 1, 1])
    proposals = np.array([1, 0, 0, 0])

    target = _target_for_mode(
        mode="class1_benefit",
        targets=targets,
        base_predictions=base,
        proposals=proposals,
    )

    assert target.tolist() == [1, 1, 1, 0]

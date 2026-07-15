import itertools

import numpy as np

from trkh.tools.audit_ip_dpp_information_gate import (
    DPPY_DETERMINANT_TOLERANCE,
    K_PER_CLASS,
    assess_gate,
    balanced_random_subset,
    dense_kernel,
    determinant_log_upper_bound,
    evaluate_subset,
    jaccard,
    kernel_components,
    subset_log_determinant,
    verify_rank_one_equations,
)


def test_rank_one_logdet_matches_dense_kernel() -> None:
    probabilities = np.asarray([0.1, 0.3, 0.55, 0.8, 0.9], dtype=np.float64)
    subset = np.asarray([0, 2, 4], dtype=np.int64)
    for mode in ("paper_n", "code_n2"):
        matrix = dense_kernel(probabilities, mode)
        sign, expected = np.linalg.slogdet(matrix[np.ix_(subset, subset)])
        assert sign > 0
        observed = subset_log_determinant(
            kernel_components(probabilities, mode), subset
        )
        np.testing.assert_allclose(observed, expected, atol=1e-13, rtol=0.0)


def test_determinant_upper_bound_is_conservative() -> None:
    probabilities = np.asarray([0.12, 0.24, 0.51, 0.72, 0.88], dtype=np.float64)
    components = kernel_components(probabilities, "paper_n")
    observed_maximum = max(
        subset_log_determinant(components, np.asarray(subset, dtype=np.int64))
        for subset in itertools.combinations(range(probabilities.size), 3)
    )
    assert determinant_log_upper_bound(components, 3) >= observed_maximum


def test_kernel_is_symmetric_stochastic_and_positive() -> None:
    audit = verify_rank_one_equations()
    assert audit["passed"] is True
    assert audit["minimum_eigenvalue"] > 0.0
    assert audit["maximum_eigenvalue"] <= 1.0 + 1e-12


def test_code_kernel_is_much_closer_to_identity() -> None:
    probabilities = np.linspace(0.12, 0.88, 31)
    paper = dense_kernel(probabilities, "paper_n")
    code = dense_kernel(probabilities, "code_n2")
    identity = np.eye(probabilities.size)
    assert np.linalg.norm(code - identity) < np.linalg.norm(paper - identity) / 10


def test_jaccard_ignores_order() -> None:
    assert jaccard(np.asarray([1, 2, 3]), np.asarray([3, 2, 1])) == 1.0
    assert jaccard(np.asarray([1, 2]), np.asarray([2, 3])) == 1 / 3


def _synthetic_data():
    labels = np.repeat(np.arange(5, dtype=np.int64), [7, 5, 8, 6, 9])
    folds = np.arange(labels.size, dtype=np.int64) % 5
    sources = np.asarray([f"source_{index}" for index in range(labels.size)])
    probabilities = {}
    for model in ("candidate", "keeper"):
        probabilities[model] = {}
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast"):
            values = np.full((labels.size, 5), 0.025, dtype=np.float64)
            values[np.arange(labels.size), labels] = 0.9
            probabilities[model][condition] = values
    return {
        "labels": labels,
        "folds": folds,
        "sources": sources,
        "probabilities": probabilities,
    }


def test_balanced_random_subset_is_deterministic(monkeypatch) -> None:
    data = _synthetic_data()
    monkeypatch.setattr(
        "trkh.tools.audit_ip_dpp_information_gate.K_PER_CLASS", 5
    )
    first = balanced_random_subset(data, 42)
    second = balanced_random_subset(data, 42)
    np.testing.assert_array_equal(first, second)
    selected_labels = data["labels"][first]
    assert np.bincount(selected_labels, minlength=5).tolist() == [5, 5, 5, 5, 5]


def test_subset_metrics_include_fold_restricted_fp_counts() -> None:
    data = _synthetic_data()
    indices = np.arange(data["labels"].size, dtype=np.int64)
    metrics = evaluate_subset(data, "candidate", indices)
    assert all(
        f"clean_fold_{fold}_restricted_focus_fp" in metrics for fold in range(5)
    )
    assert sum(
        int(metrics[f"clean_fold_{fold}_restricted_focus_fp"])
        for fold in range(5)
    ) == int(metrics["clean_restricted_focus_fp"])


def test_gate_is_fail_closed_on_source_and_sampler_discrepancies() -> None:
    conditions = ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    null_summary = {}
    official_metrics = {}
    official = {}
    for model in ("candidate", "keeper"):
        null_summary[model] = {
            "derived": {"code_q95_q05_determinant_odds": 1.001},
            "unique_sources": {"q05": 2500.0},
            **{f"fold_{fold}_rows": {"q05": 500.0, "q95": 580.0} for fold in range(5)},
            **{
                f"{condition}_{metric}": {"q95": 10.0}
                for condition in conditions
                for metric in ("mean_self_information", "restricted_focus_fp")
            },
        }
        official_metrics[model] = {
            "selected_rows": 5 * K_PER_CLASS,
            "class1_rows": K_PER_CLASS,
            "duplicate_samples": 0,
            "unique_sources": 2600,
            **{f"fold_{fold}_rows": 541 for fold in range(5)},
            **{
                f"{condition}_{metric}": 5.0
                for condition in conditions
                for metric in ("mean_self_information", "restricted_focus_fp")
            },
        }
        indices = np.arange(5 * K_PER_CLASS)
        official[model] = {
            "first_indices": indices,
            "uniform_indices": indices.copy(),
            "deterministic_replay": True,
        }
    kernel_rows = [
        {
            "kernel_mode": "paper_n",
            "class_count": 1000,
            "dppy_initialization_possible_at_tol": False,
            "psd_eigenvalue_lower_bound": 0.8,
            "row_stochastic_eigenvalue_max": 1.0,
        }
    ]
    result = assess_gate(
        equation_audit={"passed": True},
        kernel_rows=kernel_rows,
        official=official,
        official_metrics=official_metrics,
        null_summary=null_summary,
    )
    assert result["trainer_integration_authorized"] is False
    assert "official_default_changes_majority_selection" in result["failed_checks"]
    assert "official_first_state_is_not_uniform_initialization" in result["failed_checks"]
    assert "paper_and_code_kernel_scale_agree" in result["failed_checks"]
    assert "paper_kernel_dppy_initialization_feasible" in result["failed_checks"]


def test_locked_cardinality_exceeds_dppy_tolerance_only_when_log_bound_allows() -> None:
    probabilities = np.full(100, 0.8, dtype=np.float64)
    components = kernel_components(probabilities, "paper_n")
    bound = determinant_log_upper_bound(components, 80)
    assert not bool(bound > np.log(DPPY_DETERMINANT_TOLERANCE))

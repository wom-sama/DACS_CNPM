import numpy as np

from trkh.tools.probe_pairwise_feature_verifier import (
    _fit_pair_models,
    build_verifier_feature_names,
    pair_model_export_payload,
)
from trkh.tools.probe_patch_evidence_mil import build_patch_verifier_feature_names


def test_logistic_pair_model_export_matches_sklearn_predict_proba():
    rng = np.random.default_rng(123)
    features = rng.normal(size=(48, 5)).astype(np.float32)
    labels = np.array([0, 1] * 24, dtype=np.int64)
    features[labels == 1, 0] += 1.5
    features[labels == 0, 0] -= 1.5

    models, _summaries, _oof = _fit_pair_models(
        features,
        labels,
        [(0, 1)],
        model_type="logistic",
        c_value=1.0,
        max_iter=200,
        seed=7,
        oof_folds=3,
    )
    feature_names = [f"feature_{index}" for index in range(features.shape[1])]
    payload = pair_model_export_payload(
        models,
        feature_dim=features.shape[1],
        feature_names=feature_names,
    )

    exported = payload["pairs"][0]
    assert payload["feature_names"] == feature_names
    assert exported["status"] == "exported"
    raw_coef = np.asarray(exported["raw_coef"], dtype=np.float64)
    raw_intercept = float(exported["raw_intercept"])
    logits = features.astype(np.float64) @ raw_coef + raw_intercept
    exported_prob_class1 = 1.0 / (1.0 + np.exp(-logits))

    sklearn_prob_class1 = models[(0, 1)].predict_proba(features)[:, 1]
    np.testing.assert_allclose(exported_prob_class1, sklearn_prob_class1, rtol=1e-6, atol=1e-6)


def test_verifier_feature_name_schema_matches_feature_layout():
    names = build_verifier_feature_names(embedding_dim=4, class_count=3)
    assert names == [
        "embedding_0",
        "embedding_1",
        "embedding_2",
        "embedding_3",
        "prob_0",
        "prob_1",
        "prob_2",
        "log_prob_0",
        "log_prob_1",
        "log_prob_2",
        "prob_margin_top1_top2",
        "prob_confidence_top1",
    ]

    patch_names = build_patch_verifier_feature_names(
        embedding_dim=256,
        class_count=5,
        pairs=[(0, 1)],
        spatial_evidence_features=False,
    )
    assert len(patch_names) == 318
    assert patch_names[:3] == ["embedding_0", "embedding_1", "embedding_2"]
    assert patch_names[-10] == "pair_0_1_margin_mean_all"
    assert patch_names[-1] == "pair_0_1_positive_fraction_bbox"

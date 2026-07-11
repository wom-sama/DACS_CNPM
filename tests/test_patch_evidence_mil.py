import numpy as np
import torch

from trkh.tools.probe_patch_evidence_mil import (
    _append_source_domain_feature,
    _write_pair_teacher_csv,
    build_patch_verifier_features,
    summarize_patch_evidence,
)
from trkh.tools.probe_pairwise_feature_verifier import parse_pairs


def test_summarize_patch_evidence_prefers_bbox_topk_region() -> None:
    local_logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        ]
    )
    bbox_prior = torch.tensor([[0.0, 1.0, 0.0]])
    patch_features, method_logits, diagnostics = summarize_patch_evidence(
        local_logits,
        bbox_prior=bbox_prior,
        pairs=parse_pairs("0-1"),
        top_k=1,
    )
    assert patch_features.shape[0] == 1
    assert torch.equal(diagnostics["bbox_mask"], torch.tensor([[False, True, False]]))
    assert int(method_logits["patch_max"].argmax(dim=1).item()) == 2
    assert int(method_logits["bbox_topk_mean"].argmax(dim=1).item()) == 1


def test_summarize_patch_evidence_uses_valid_mask() -> None:
    local_logits = torch.tensor(
        [
            [
                [0.0, 5.0],
                [8.0, 0.0],
            ]
        ]
    )
    key_padding_mask = torch.tensor([[False, True]])
    _, method_logits, diagnostics = summarize_patch_evidence(
        local_logits,
        key_padding_mask=key_padding_mask,
        top_k=1,
    )
    assert torch.equal(diagnostics["valid_mask"], torch.tensor([[True, False]]))
    assert int(method_logits["patch_max"].argmax(dim=1).item()) == 1


def test_summarize_patch_evidence_spatial_features_split_interior_and_border() -> None:
    local_logits = torch.zeros((1, 9, 3), dtype=torch.float32)
    local_logits[0, 0, 2] = 9.0
    local_logits[0, 4, 1] = 5.0
    bbox_prior = torch.ones((1, 9), dtype=torch.float32)

    patch_features, method_logits, diagnostics = summarize_patch_evidence(
        local_logits,
        bbox_prior=bbox_prior,
        pairs=parse_pairs("0-1"),
        top_k=1,
        spatial_evidence_features=True,
        spatial_interior_erode=1,
    )

    assert patch_features.shape[0] == 1
    assert int(method_logits["patch_max"].argmax(dim=1).item()) == 2
    assert int(method_logits["interior_topk_mean"].argmax(dim=1).item()) == 1
    assert int(method_logits["border_topk_mean"].argmax(dim=1).item()) == 2
    assert diagnostics["interior_mask"].sum().item() == 1
    assert diagnostics["border_mask"].sum().item() == 8


def test_build_patch_verifier_features_appends_probability_and_patch_context() -> None:
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    probabilities = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)
    patch_features = np.array([[0.5, -0.25, 1.5]], dtype=np.float32)
    features = build_patch_verifier_features(embeddings, probabilities, patch_features)
    assert features.shape == (1, 13)
    assert np.allclose(features[0, :2], embeddings[0])
    assert np.allclose(features[0, 2:5], probabilities[0])
    assert np.isclose(features[0, -5], 0.5)
    assert np.allclose(features[0, -3:], patch_features[0])


def test_append_source_domain_feature_is_optional() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    unchanged = _append_source_domain_feature(features, source_value=1.0, enabled=False)
    with_source = _append_source_domain_feature(features, source_value=2.0, enabled=True)

    assert unchanged.shape == (2, 2)
    assert with_source.shape == (2, 3)
    assert np.allclose(with_source[:, :2], features)
    assert np.allclose(with_source[:, 2], 2.0)


def test_write_pair_teacher_csv_uses_oof_pair_probs_and_hard_other_classes(tmp_path) -> None:
    labels = np.array([0, 1, 2], dtype=np.int64)
    pair_probs = {
        (0, 1): np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [np.nan, np.nan],
            ],
            dtype=np.float32,
        )
    }
    path = tmp_path / "teacher.csv"

    summary = _write_pair_teacher_csv(
        path,
        labels=labels,
        paths=["a.jpg", "b.jpg", "c.jpg"],
        pair_probabilities=pair_probs,
        num_classes=3,
        split="train",
        smoothing=0.0,
    )

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("split,sample_index,path,target_index,prob_0")
    assert summary["rows"] == 3
    assert summary["pair_rows"] == 2
    assert summary["finite_pair_rows"] == 2
    assert summary["teacher_label_agreement"] == 1.0
    assert "0.800000" in rows[1] or "0.8" in rows[1]
    assert rows[3].endswith("0.0,0.0,1.0")

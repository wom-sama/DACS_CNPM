from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch import nn

from trkh.models.pretrained_semantic_branch import (
    PretrainedBackboneContractError,
    PretrainedSemanticResidualBranch,
    PretrainedWeightVerificationError,
    load_verified_local_timm_model,
    sha256_file,
)


class FakeTokenBackbone(nn.Module):
    num_prefix_tokens = 2
    num_features = 4

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(3, self.num_features, bias=False)
        self.prefix_tokens = nn.Parameter(
            torch.linspace(-0.4, 0.4, self.num_prefix_tokens * self.num_features).reshape(
                1,
                self.num_prefix_tokens,
                self.num_features,
            )
        )
        self.patch_offsets = nn.Parameter(
            torch.tensor(
                [
                    [-0.30, -0.10, 0.10, 0.30],
                    [0.20, -0.20, 0.40, -0.40],
                    [0.05, 0.15, -0.05, -0.15],
                ],
                dtype=torch.float32,
            )
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        image_summary = images.mean(dim=(-2, -1))
        projected = self.input_projection(image_summary)
        patches = projected.unsqueeze(1) + self.patch_offsets.unsqueeze(0)
        prefix = self.prefix_tokens.expand(int(images.size(0)), -1, -1)
        return torch.cat((prefix, patches), dim=1)


def _write_fake_checkpoint(path: Path, model: nn.Module) -> str:
    torch.save({"state_dict": model.state_dict()}, path)
    return sha256_file(path)


def _loader_kwargs(path: Path, digest: str) -> Dict[str, Any]:
    return {
        "model_name": "fake_vit_small",
        "checkpoint_path": path,
        "expected_sha256": digest,
        "source_repository": "local/fake-v1",
        "source_revision": "revision-for-test",
        "license_id": "test-only-license",
        "checkpoint_key": "state_dict",
    }


def test_verified_loader_fails_hash_before_constructing_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fake_weights.pt"
    _write_fake_checkpoint(checkpoint, FakeTokenBackbone())
    factory_calls: List[Dict[str, Any]] = []

    def factory(model_name: str, **kwargs: Any) -> nn.Module:
        factory_calls.append({"model_name": model_name, **kwargs})
        return FakeTokenBackbone()

    with pytest.raises(PretrainedWeightVerificationError, match="SHA-256 mismatch"):
        load_verified_local_timm_model(
            **_loader_kwargs(checkpoint, "0" * 64),
            model_factory=factory,
        )

    assert factory_calls == []


def test_verified_loader_uses_pretrained_false_and_records_provenance(
    tmp_path: Path,
) -> None:
    source = FakeTokenBackbone()
    with torch.no_grad():
        for parameter_index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(0.125 * parameter_index)
    checkpoint = tmp_path / "fake_weights.pt"
    digest = _write_fake_checkpoint(checkpoint, source)
    factory_calls: List[Dict[str, Any]] = []

    def factory(model_name: str, **kwargs: Any) -> nn.Module:
        factory_calls.append({"model_name": model_name, **kwargs})
        return FakeTokenBackbone()

    loaded, provenance = load_verified_local_timm_model(
        **_loader_kwargs(checkpoint, digest),
        model_kwargs={"num_classes": 0, "img_size": 8},
        model_factory=factory,
    )

    assert factory_calls == [
        {
            "model_name": "fake_vit_small",
            "pretrained": False,
            "num_classes": 0,
            "img_size": 8,
        }
    ]
    for key, expected in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)
    assert provenance["offline_verified_load"] is True
    assert provenance["network_access_required"] is False
    assert provenance["checkpoint"]["sha256"] == digest
    assert provenance["checkpoint"]["path"] == str(checkpoint.resolve())
    assert provenance["checkpoint"]["tensor_count"] == len(source.state_dict())
    assert provenance["factory"]["pretrained_argument"] is False
    assert provenance["source_repository"] == "local/fake-v1"
    assert loaded.pretrained_source_provenance == provenance


def test_verified_loader_preserves_logical_safetensors_suffix_for_symlink(
    tmp_path: Path,
) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    source = FakeTokenBackbone()
    blob = tmp_path / "blobs" / "extensionless-weight-blob"
    blob.parent.mkdir(parents=True)
    safetensors_torch.save_file(source.state_dict(), str(blob))
    logical_path = tmp_path / "snapshots" / "revision" / "model.safetensors"
    logical_path.parent.mkdir(parents=True)
    try:
        logical_path.symlink_to(blob)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Local platform does not permit test symlinks: {exc}")
    digest = sha256_file(logical_path)

    loaded, provenance = load_verified_local_timm_model(
        **{
            key: value
            for key, value in _loader_kwargs(logical_path, digest).items()
            if key != "checkpoint_key"
        },
        model_factory=lambda _name, **_kwargs: FakeTokenBackbone(),
    )

    for key, expected in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)
    assert provenance["checkpoint"]["path"] == str(logical_path.absolute())
    assert provenance["checkpoint"]["resolved_path"] == str(blob.resolve())
    assert provenance["checkpoint"]["format"] == "safetensors"
    assert provenance["checkpoint"]["is_symlink"] is True


def test_semantic_branch_splits_prefix_and_patches_and_is_zero_gated() -> None:
    backbone = FakeTokenBackbone()
    branch = PretrainedSemanticResidualBranch(
        backbone,
        output_dim=6,
        expected_patch_count=3,
        hidden_dim=8,
        dropout=0.0,
        max_scale=0.75,
        source_provenance={"checkpoint": {"sha256": "a" * 64}},
    ).eval()
    images = torch.randn(2, 3, 4, 4)
    base_feature = torch.randn(2, 6)

    fused, trace = branch(images, base_feature, return_trace=True)

    assert torch.equal(fused, base_feature)
    assert trace["token_shape"] == [2, 5, 4]
    assert trace["prefix_token_count"] == 2
    assert trace["patch_token_count"] == 3
    assert trace["descriptor_components"] == [
        "cls_token",
        "register_mean",
        "patch_mean",
        "patch_std",
    ]
    assert trace["descriptor"].shape == (2, 16)
    assert trace["residual"].shape == (2, 6)
    assert float(trace["raw_gate"].item()) == pytest.approx(0.0)
    assert float(trace["effective_gate"].item()) == pytest.approx(0.0)
    assert trace["configured_initial_scale"] == pytest.approx(0.0)
    assert trace["configured_max_scale"] == pytest.approx(0.75)
    assert torch.count_nonzero(trace["gated_residual_norm_ratio"]) == 0
    assert (
        trace["source_provenance"]["source"]["checkpoint"]["sha256"]
        == "a" * 64
    )

    with pytest.raises(PretrainedBackboneContractError, match="non-empty patch suffix"):
        branch.split_tokens(torch.zeros(2, 2, 4))

    wrong_patch_contract = PretrainedSemanticResidualBranch(
        FakeTokenBackbone(),
        output_dim=6,
        expected_patch_count=4,
        dropout=0.0,
    )
    with pytest.raises(PretrainedBackboneContractError, match="patch-token count changed"):
        wrong_patch_contract(images, base_feature)


def test_semantic_branch_nonzero_initial_scale_is_bounded_and_recorded() -> None:
    branch = PretrainedSemanticResidualBranch(
        FakeTokenBackbone(),
        output_dim=6,
        expected_patch_count=3,
        hidden_dim=8,
        dropout=0.0,
        initial_scale=0.20,
        max_scale=0.50,
    ).eval()
    images = torch.randn(2, 3, 4, 4)
    base_feature = torch.randn(2, 6)

    fused, trace = branch(images, base_feature, return_trace=True)

    assert float(trace["effective_gate"].item()) == pytest.approx(0.20, abs=1e-7)
    assert torch.allclose(
        fused,
        base_feature + 0.20 * trace["residual"],
        rtol=1e-6,
        atol=1e-7,
    )
    provenance = branch.provenance()["residual_contract"]
    assert provenance["initial_scale"] == pytest.approx(0.20)
    assert provenance["max_scale"] == pytest.approx(0.50)
    assert provenance["raw_gate_initial_value"] == pytest.approx(math.atanh(0.4))
    assert provenance["keeper_identity_at_initialization"] is False

    with pytest.raises(ValueError, match="strictly inside"):
        PretrainedSemanticResidualBranch(
            FakeTokenBackbone(),
            output_dim=6,
            initial_scale=0.5,
            max_scale=0.5,
        )


def test_semantic_branch_rezero_gradients_and_provenance() -> None:
    torch.manual_seed(7)
    backbone = FakeTokenBackbone()
    branch = PretrainedSemanticResidualBranch(
        backbone,
        output_dim=6,
        expected_patch_count=3,
        hidden_dim=9,
        dropout=0.0,
        max_scale=0.8,
        source_provenance={
            "model_name": "fake_vit_small",
            "checkpoint": {"sha256": "b" * 64},
        },
    )
    images = torch.randn(3, 3, 4, 4)
    base_feature = torch.randn(3, 6)

    fused, trace = branch(images, base_feature, return_trace=True)
    # Align the upstream gradient with the nonzero residual so the scalar
    # ReZero gate has a deterministic, strictly nonzero first-step gradient.
    loss = (fused * trace["residual"]).sum()
    loss.backward()

    assert branch.residual_gate.grad is not None
    assert float(branch.residual_gate.grad.abs().item()) > 0.0
    assert all(
        parameter.grad is None or int(torch.count_nonzero(parameter.grad).item()) == 0
        for parameter in branch.projector.parameters()
    )
    assert all(
        parameter.grad is None or int(torch.count_nonzero(parameter.grad).item()) == 0
        for parameter in branch.backbone.parameters()
    )

    branch.zero_grad(set_to_none=True)
    with torch.no_grad():
        branch.residual_gate.fill_(0.4)
    trained_fused = branch(images, base_feature)
    trained_fused.square().sum().backward()
    assert any(
        parameter.grad is not None
        and int(torch.count_nonzero(parameter.grad).item()) > 0
        for parameter in branch.projector.parameters()
    )
    assert any(
        parameter.grad is not None
        and int(torch.count_nonzero(parameter.grad).item()) > 0
        for parameter in branch.backbone.parameters()
    )

    provenance = branch.provenance()
    assert provenance["source"]["model_name"] == "fake_vit_small"
    assert provenance["token_contract"]["backbone_dim"] == 4
    assert provenance["token_contract"]["num_prefix_tokens"] == 2
    assert provenance["token_contract"]["expected_patch_count"] == 3
    assert provenance["residual_contract"]["output_dim"] == 6
    assert provenance["residual_contract"]["initial_scale"] == pytest.approx(0.0)
    assert provenance["residual_contract"]["max_scale"] == pytest.approx(0.8)
    assert provenance["residual_contract"]["keeper_identity_at_initialization"] is True
    assert provenance["parameter_count"] > 0
    assert provenance["trainable_parameter_count"] == provenance["parameter_count"]

    branch.set_backbone_trainable(False)
    branch.train()
    assert branch.backbone_frozen is True
    assert branch.backbone.training is False
    assert all(not parameter.requires_grad for parameter in branch.backbone.parameters())
    frozen_provenance = branch.provenance()
    assert frozen_provenance["backbone_frozen"] is True
    assert frozen_provenance["trainable_parameter_count"] < frozen_provenance["parameter_count"]

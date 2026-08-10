from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trkh.inference.deploy import (
    build_runtime_inputs,
    format_trt_shapes,
    resolve_runtime_input_shapes,
    validate_tensorrt_export_policy,
)
from trkh.inference.stream_infer_trt import predict_tensor_outputs_trt
from trkh.tools.audit_precision_ensemble_onnx import (
    _parse_sample_indices as _parse_onnx_sample_indices,
    _resolve_audit_indices as _resolve_onnx_audit_indices,
)
from trkh.tools.audit_precision_ensemble_trt import (
    _engine_schema,
    _parse_sample_indices,
    _resolve_audit_indices,
)
from trkh.evaluation.xai_audit import _select_xai_model
from trkh.models.model import (
    _onnx_exact_adaptive_avg_pool2d,
    build_model_from_checkpoint,
    create_model,
)
from trkh.models.precision_ensemble import PrecisionEnsembleClassifier


class _FixedMember(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone(), requires_grad=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(int(images.shape[0]), -1)


class _FakeMultiInputTensorRT:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.input_names = ["images", "image_valid_mask", "bbox"]
        self.input_dtypes = {
            "images": torch.float32,
            "image_valid_mask": torch.float32,
            "bbox": torch.float32,
        }
        self.input_dtype = torch.float32
        self.output_names = ["logits"]
        self.captured_inputs: dict[str, torch.Tensor] = {}

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.captured_inputs = inputs
        return {"logits": torch.zeros(1, 5)}


class _FakeTensorRTEngine:
    def __init__(self, shapes: dict[str, tuple[int, ...]]) -> None:
        self.shapes = shapes

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]


class _FakeAuditTensorRT:
    def __init__(self) -> None:
        self.input_names = ["images", "image_valid_mask", "bbox"]
        self.output_names = list(
            PrecisionEnsembleClassifier.deployment_output_names
        )
        self.input_dtypes = {name: torch.float32 for name in self.input_names}
        self.output_dtypes = {name: torch.float32 for name in self.output_names}
        self.engine = _FakeTensorRTEngine(
            {
                "images": (1, 3, 256, 256),
                "image_valid_mask": (1, 256, 256),
                "bbox": (1, 4),
                **{name: (1, 5) for name in self.output_names},
            }
        )


def _tiny_config() -> dict[str, object]:
    return {
        "model_type": "vit_registers",
        "image_size": 32,
        "patch_size": 8,
        "use_cnn_stem": False,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "num_registers": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "token_pruning": False,
        "head_pooling": "cls_register_mean",
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
    }


def test_precision_ensemble_matches_locked_probability_rule() -> None:
    keeper_logits = torch.tensor([[1.2, 1.0, -0.5, 0.1, -0.2]])
    candidate_logits = torch.tensor([[0.5, 1.7, -0.2, 0.0, -0.1]])
    model = PrecisionEnsembleClassifier(
        _FixedMember(keeper_logits),
        _FixedMember(candidate_logits),
        num_classes=5,
        candidate_weight=0.4,
        focus_class=1,
        focus_margin_offset=0.034,
    ).eval()
    images = torch.randn(3, 3, 8, 8)

    outputs = model.audit_outputs(images)
    deployment_outputs = model.deployment_outputs(images)
    keeper_probabilities = keeper_logits.softmax(1).expand(3, -1)
    candidate_probabilities = candidate_logits.softmax(1).expand(3, -1)
    raw_blend = keeper_probabilities * 0.6 + candidate_probabilities * 0.4
    decision_scores = raw_blend.clone()
    decision_scores[:, 1] -= 0.034
    expected_probabilities = decision_scores.clamp_min(1e-8)
    expected_probabilities /= expected_probabilities.sum(1, keepdim=True)

    assert torch.allclose(outputs["raw_blend"], raw_blend, atol=1e-7, rtol=0.0)
    assert torch.allclose(
        outputs["decision_scores"],
        decision_scores,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        outputs["probabilities"],
        expected_probabilities,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.equal(
        outputs["probabilities"].argmax(1),
        decision_scores.argmax(1),
    )
    assert len(deployment_outputs) == len(model.deployment_output_names)
    for name, tensor in zip(model.deployment_output_names, deployment_outputs):
        expected_name = "probabilities" if name == "deployment_probabilities" else name
        assert torch.equal(tensor, outputs[expected_name])
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_precision_ensemble_validates_spatial_metadata_shapes() -> None:
    model = PrecisionEnsembleClassifier(
        _FixedMember(torch.zeros(1, 5)),
        _FixedMember(torch.ones(1, 5)),
        num_classes=5,
        candidate_weight=0.4,
        focus_class=1,
        focus_margin_offset=0.034,
    )
    images = torch.randn(2, 3, 8, 8)
    assert model(images).shape == (2, 5)
    with pytest.raises(ValueError, match="image_valid_mask shape mismatch"):
        model(images, image_valid_mask=torch.ones(2, 7, 8))
    with pytest.raises(ValueError, match="bbox shape mismatch"):
        model(images, bbox=torch.ones(2, 5))


def test_precision_ensemble_checkpoint_roundtrip_is_strict() -> None:
    config = _tiny_config()
    keeper = create_model(num_classes=5, model_config=config)
    candidate = create_model(num_classes=5, model_config=config)
    source = PrecisionEnsembleClassifier(
        keeper,
        candidate,
        num_classes=5,
        candidate_weight=0.4,
        focus_class=1,
        focus_margin_offset=0.034,
    ).eval()
    checkpoint = {
        "class_names": [f"class_{index}" for index in range(5)],
        "model_config": {
            "model_type": "precision_ensemble",
            "image_size": 32,
            "temporal_frames": 1,
            "input_mean": [0.485, 0.456, 0.406],
            "input_std": [0.229, 0.224, 0.225],
            "precision_ensemble": {
                "candidate_weight": 0.4,
                "keeper_weight": 0.6,
                "focus_class": 1,
                "focus_margin_offset": 0.034,
                "minimum_probability": 1e-8,
                "member_model_configs": {
                    "keeper": config,
                    "candidate": config,
                },
            },
        },
        "model_state": source.state_dict(),
    }
    restored = build_model_from_checkpoint(checkpoint)
    assert isinstance(restored, PrecisionEnsembleClassifier)
    assert set(restored.state_dict()) == set(source.state_dict())
    images = torch.randn(2, 3, 32, 32)
    mask = torch.ones(2, 32, 32)
    bbox = torch.tensor([[0.5, 0.5, 0.8, 0.8]]).expand(2, -1)
    with torch.inference_mode():
        assert torch.allclose(
            source(images, mask, bbox),
            restored(images, mask, bbox),
            atol=0.0,
            rtol=0.0,
        )

    tampered = copy.deepcopy(checkpoint)
    tampered["model_state"]["candidate_weight"] = torch.tensor(0.5)
    with pytest.raises(ValueError, match="candidate-weight buffer"):
        build_model_from_checkpoint(tampered)


def test_precision_ensemble_runtime_shapes_include_metadata() -> None:
    model = PrecisionEnsembleClassifier(
        _FixedMember(torch.zeros(1, 5)),
        _FixedMember(torch.ones(1, 5)),
        num_classes=5,
        candidate_weight=0.4,
        focus_class=1,
        focus_margin_offset=0.034,
    )
    shapes = resolve_runtime_input_shapes(model, (2, 3, 32, 32))
    assert shapes == {
        "images": (2, 3, 32, 32),
        "image_valid_mask": (2, 32, 32),
        "bbox": (2, 4),
    }
    inputs = build_runtime_inputs(model, (2, 3, 32, 32), torch.device("cpu"))
    assert [tuple(tensor.shape) for tensor in inputs] == [
        (2, 3, 32, 32),
        (2, 32, 32),
        (2, 4),
    ]
    assert format_trt_shapes(shapes) == (
        "images:2x3x32x32,image_valid_mask:2x32x32,bbox:2x4"
    )
    assert model.supports_dynamic_batch is False
    assert model.certified_batch_sizes == (1,)
    assert model.supports_torchscript_deployment is False


@pytest.mark.parametrize(
    ("input_shape", "output_size"),
    [
        ((2, 7, 256, 256), (56, 56)),
        ((2, 3, 64, 64), (56, 56)),
        ((2, 1, 256, 256), (4, 4)),
        ((2, 4, 56, 56), 1),
    ],
)
def test_onnx_adaptive_pool_decomposition_matches_pytorch(
    input_shape: tuple[int, ...],
    output_size: int | tuple[int, int],
) -> None:
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(input_shape, generator=generator)
    expected = F.adaptive_avg_pool2d(inputs, output_size)
    actual = _onnx_exact_adaptive_avg_pool2d(inputs, output_size)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2.5e-7)


def test_tensorrt_prediction_binds_explicit_spatial_metadata() -> None:
    classifier = _FakeMultiInputTensorRT()
    images = torch.randn(1, 3, 8, 8)
    mask = torch.ones(1, 8, 8)
    bbox = torch.tensor([[0.4, 0.6, 0.3, 0.2]])
    outputs = predict_tensor_outputs_trt(
        classifier,
        images,
        image_valid_mask=mask,
        bbox=bbox,
    )
    assert set(classifier.captured_inputs) == {
        "images",
        "image_valid_mask",
        "bbox",
    }
    assert classifier.captured_inputs["image_valid_mask"] is mask
    assert classifier.captured_inputs["bbox"] is bbox
    assert outputs["spatial_metadata_mode"] == "certified_explicit_metadata"


def test_tensorrt_prediction_labels_missing_bbox_as_unverified() -> None:
    classifier = _FakeMultiInputTensorRT()
    outputs = predict_tensor_outputs_trt(
        classifier,
        torch.randn(1, 3, 8, 8),
        image_valid_mask=torch.ones(1, 8, 8),
    )
    assert outputs["spatial_metadata_mode"] == "unverified_full_frame_fallback"
    torch.testing.assert_close(
        classifier.captured_inputs["bbox"],
        torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
    )


def test_xai_requires_explicit_precision_ensemble_member(tmp_path) -> None:
    keeper = _FixedMember(torch.zeros(1, 5))
    candidate = _FixedMember(torch.ones(1, 5))
    model = PrecisionEnsembleClassifier(
        keeper,
        candidate,
        num_classes=5,
        candidate_weight=0.4,
        focus_class=1,
        focus_margin_offset=0.034,
    )
    checkpoint_path = tmp_path / "ensemble.pt"
    checkpoint_path.write_bytes(b"package")
    checkpoint = {
        "model_config": {"model_type": "precision_ensemble"},
        "precision_ensemble_provenance": {
            "candidate_packaged_state_sha256": "candidate-state",
            "candidate_checkpoint": "candidate.pt",
            "candidate_checkpoint_sha256": "candidate-source",
            "frozen_protocol_sha256": "protocol",
        },
    }
    with pytest.raises(ValueError, match="requires --ensemble-member"):
        _select_xai_model(
            model=model,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            ensemble_member=None,
        )
    selected, provenance = _select_xai_model(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        ensemble_member="candidate",
    )
    assert selected is candidate
    assert provenance["selected_member"] == "candidate"
    assert provenance["aggregate_attention_produced"] is False
    assert provenance["member_packaged_state_sha256"] == "candidate-state"
    assert provenance["xai_autograd"]["in_memory_only"] is True
    assert provenance["xai_autograd"]["optimizer_step_performed"] is False
    assert provenance["xai_autograd"]["requires_grad_reenabled"] is True
    assert all(parameter.requires_grad for parameter in selected.parameters())
    assert all(not parameter.requires_grad for parameter in keeper.parameters())


def test_tensorrt_audit_targeted_indices_are_strict() -> None:
    assert _parse_sample_indices("1094, 1571,2417") == [1094, 1571, 2417]
    assert _resolve_audit_indices(
        dataset_size=2606,
        max_samples=0,
        sample_indices=[1094, 1571, 2417],
    ) == [1094, 1571, 2417]
    assert _resolve_audit_indices(
        dataset_size=2606,
        max_samples=64,
        sample_indices=[],
    ) == list(range(64))
    with pytest.raises(ValueError, match="either"):
        _resolve_audit_indices(
            dataset_size=2606,
            max_samples=64,
            sample_indices=[1094],
        )
    with pytest.raises(ValueError, match="unique"):
        _resolve_audit_indices(
            dataset_size=2606,
            max_samples=0,
            sample_indices=[1094, 1094],
        )
    with pytest.raises(ValueError, match="outside"):
        _resolve_audit_indices(
            dataset_size=2606,
            max_samples=0,
            sample_indices=[2606],
        )


def test_onnx_audit_targeted_indices_are_strict() -> None:
    assert _parse_onnx_sample_indices("83,115, 1571") == [83, 115, 1571]
    assert _resolve_onnx_audit_indices(
        dataset_size=2606,
        max_samples=0,
        sample_indices=[83, 115, 1571],
    ) == [83, 115, 1571]
    with pytest.raises(ValueError, match="either"):
        _resolve_onnx_audit_indices(
            dataset_size=2606,
            max_samples=32,
            sample_indices=[83],
        )


def test_tensorrt_audit_schema_is_actually_validated() -> None:
    classifier = _FakeAuditTensorRT()
    schema = _engine_schema(classifier)
    assert schema["inputs"] == {
        "images": [1, 3, 256, 256],
        "image_valid_mask": [1, 256, 256],
        "bbox": [1, 4],
    }
    assert schema["outputs"] == {
        name: [1, 5]
        for name in PrecisionEnsembleClassifier.deployment_output_names
    }

    classifier.engine.shapes["bbox"] = (2, 4)
    with pytest.raises(ValueError, match="input schema mismatch"):
        _engine_schema(classifier)


def test_precision_ensemble_tensorrt_export_fails_closed() -> None:
    checkpoint = {"model_config": {"model_type": "precision_ensemble"}}
    with pytest.raises(ValueError, match="blocked by default"):
        validate_tensorrt_export_policy(
            checkpoint,
            skip_trt_engine=False,
            allow_uncertified_precision_ensemble_tensorrt=False,
        )

    validate_tensorrt_export_policy(
        checkpoint,
        skip_trt_engine=True,
        allow_uncertified_precision_ensemble_tensorrt=False,
    )
    validate_tensorrt_export_policy(
        checkpoint,
        skip_trt_engine=False,
        allow_uncertified_precision_ensemble_tensorrt=True,
    )

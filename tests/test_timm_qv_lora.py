from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from trkh.core.config import ModelConfig
from trkh.core.utils import build_optimizer_param_groups
from trkh.inference.deploy import export_dynamic_int8, export_torchscript
from trkh.inference.inference import export_onnx
from trkh.models.model import create_model
from trkh.models.timm_qv_lora import (
    TimmQKVLoRALinear,
    inject_timm_qv_lora,
    materialize_timm_qv_lora_,
    merge_timm_qv_lora_,
    parse_lora_layers,
    timm_qv_lora_parameter_prefixes,
    unmerge_timm_qv_lora_,
)
from trkh.training.train import (
    _load_model_state_allowing_extensions,
    build_configs,
    parse_args,
)


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width, bias=False)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = _Attention(width)


class _TimmLike(nn.Module):
    def __init__(self, width: int = 8, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks[-1].attn.qkv(inputs)


def test_parse_lora_layers_is_ordered_and_unique() -> None:
    assert parse_lora_layers("3,1,3;2") == [3, 1, 2]
    assert timm_qv_lora_parameter_prefixes("3,1") == [
        "blocks.3.attn.qkv.q_lora",
        "blocks.3.attn.qkv.v_lora",
        "blocks.1.attn.qkv.q_lora",
        "blocks.1.attn.qkv.v_lora",
    ]
    with pytest.raises(ValueError):
        parse_lora_layers("")


def test_zero_init_lora_is_bit_exact_and_qv_only() -> None:
    torch.manual_seed(7)
    source = nn.Linear(8, 24, bias=False)
    target = TimmQKVLoRALinear.from_linear(
        copy.deepcopy(source),
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    inputs = torch.randn(3, 5, 8)
    target.train()
    assert torch.equal(source(inputs), target(inputs))

    with torch.no_grad():
        target.q_lora.up.weight.fill_(0.2)
        target.v_lora.up.weight.fill_(-0.1)
    output = target(inputs)
    base = source(inputs)
    width = source.in_features
    assert not torch.equal(output[..., :width], base[..., :width])
    assert torch.equal(output[..., width : 2 * width], base[..., width : 2 * width])
    assert not torch.equal(output[..., 2 * width :], base[..., 2 * width :])


def test_lora_merge_and_unmerge_preserve_eval_output() -> None:
    torch.manual_seed(11)
    model = _TimmLike(width=8, depth=4).eval()
    inject_timm_qv_lora(
        model,
        layers="1,2,3",
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TimmQKVLoRALinear):
                module.q_lora.up.weight.normal_(std=0.02)
                module.v_lora.up.weight.normal_(std=0.02)
    inputs = torch.randn(2, 3, 8)
    before = [block.attn.qkv(inputs) for block in model.blocks]
    merge_summary = merge_timm_qv_lora_(model)
    after = [block.attn.qkv(inputs) for block in model.blocks]
    assert merge_summary == {"merged_modules": 3, "all_merged": True}
    for left, right in zip(before, after):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)

    unmerge_summary = unmerge_timm_qv_lora_(model)
    restored = [block.attn.qkv(inputs) for block in model.blocks]
    assert unmerge_summary == {"unmerged_modules": 3, "all_unmerged": True}
    for left, right in zip(before, restored):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_materialize_lora_is_plain_linear_and_preserves_eval_output() -> None:
    torch.manual_seed(13)
    model = _TimmLike(width=8, depth=4).eval()
    inject_timm_qv_lora(
        model,
        layers="1,2,3",
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TimmQKVLoRALinear):
                module.q_lora.up.weight.normal_(std=0.02)
                module.v_lora.up.weight.normal_(std=0.02)
    inputs = torch.randn(2, 3, 8)
    before = [block.attn.qkv(inputs) for block in model.blocks]
    state_keys_before = set(model.state_dict())

    summary = materialize_timm_qv_lora_(model)
    after = [block.attn.qkv(inputs) for block in model.blocks]
    state_keys_after = set(model.state_dict())

    assert summary == {
        "materialized_modules": 3,
        "module_names": [
            "blocks.1.attn.qkv",
            "blocks.2.attn.qkv",
            "blocks.3.attn.qkv",
        ],
        "remaining_lora_modules": 0,
        "source_dtype": "float32",
    }
    assert model.timm_qv_lora_enabled is False
    assert model.timm_qv_lora_summary["materialized"] is True
    assert model.timm_qv_lora_summary["wrapped_modules"] == []
    assert any("q_lora" in key for key in state_keys_before)
    assert all("lora" not in key for key in state_keys_after)
    assert all(
        type(model.blocks[index].attn.qkv) is nn.Linear for index in (1, 2, 3)
    )
    for left, right in zip(before, after):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_torchscript_and_dynamic_int8_exports_materialize_copy(tmp_path) -> None:
    torch.manual_seed(17)
    model = _TimmLike(width=8, depth=4).eval()
    inject_timm_qv_lora(
        model,
        layers="3",
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    with torch.no_grad():
        model.blocks[3].attn.qkv.q_lora.up.weight.normal_(std=0.02)
        model.blocks[3].attn.qkv.v_lora.up.weight.normal_(std=0.02)
    inputs = torch.randn(1, 2, 8)
    expected = model(inputs)

    torchscript_path = export_torchscript(
        model,
        (1, 2, 8),
        tmp_path / "candidate.pt",
    )
    scripted = torch.jit.load(str(torchscript_path)).eval()
    assert torch.allclose(expected, scripted(inputs), atol=1e-6, rtol=1e-5)

    quantized, quantized_path = export_dynamic_int8(
        model,
        (1, 2, 8),
        tmp_path / "candidate_int8.pt",
    )
    assert quantized_path.is_file()
    assert not any(
        isinstance(module, TimmQKVLoRALinear) for module in quantized.modules()
    )
    assert all("lora" not in key for key in quantized.state_dict())
    assert isinstance(model.blocks[3].attn.qkv, TimmQKVLoRALinear)


def test_onnx_export_materializes_copy_and_matches_runtime(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(19)
    model = _TimmLike(width=8, depth=4).eval()
    inject_timm_qv_lora(
        model,
        layers="3",
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    with torch.no_grad():
        model.blocks[3].attn.qkv.q_lora.up.weight.normal_(std=0.02)
        model.blocks[3].attn.qkv.v_lora.up.weight.normal_(std=0.02)
    output_path = export_onnx(
        model,
        {"model_config": {"image_size": 8, "temporal_frames": 1}},
        tmp_path / "candidate.onnx",
    )
    graph = onnx.load(str(output_path))
    graph_names = [
        value
        for node in graph.graph.node
        for value in (node.name, *node.input, *node.output)
    ]
    graph_names.extend(initializer.name for initializer in graph.graph.initializer)
    assert all("lora" not in str(name).lower() for name in graph_names)

    inputs = torch.randn(1, 3, 8, 8)
    expected = model(inputs).detach().cpu().numpy()
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    observed = session.run(None, {"images": inputs.numpy()})[0]
    assert float(abs(expected - observed).max()) <= 1e-5
    assert isinstance(model.blocks[3].attn.qkv, TimmQKVLoRALinear)


def test_inject_lora_reports_exact_parameter_count() -> None:
    model = _TimmLike(width=8, depth=4)
    summary = inject_timm_qv_lora(
        model,
        layers="0,2",
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert summary["layers"] == [0, 2]
    assert summary["lora_parameter_count"] == 2 * 2 * (8 * 2 + 2 * 8)
    assert isinstance(model.blocks[0].attn.qkv, TimmQKVLoRALinear)
    assert isinstance(model.blocks[2].attn.qkv, TimmQKVLoRALinear)
    assert isinstance(model.blocks[1].attn.qkv, nn.Linear)


def test_lora_keeps_experiment_lr_when_timm_backbone_scale_is_reduced() -> None:
    model = _TimmLike(width=8, depth=4)
    inject_timm_qv_lora(
        model,
        layers="1,2,3",
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    model.is_pretrained_timm_classifier = True
    model.pretrained_classifier_parameter_prefixes = ("head.",)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name
    groups = build_optimizer_param_groups(
        model,
        weight_decay=0.05,
        learning_rate=2e-4,
        backbone_lr_scale=0.1,
    )
    assert {group["name"] for group in groups} == {"head_decay"}
    optimizer = torch.optim.AdamW(groups, lr=2e-4)
    assert {group["lr"] for group in optimizer.param_groups} == {2e-4}


def test_resume_allows_only_new_lora_state() -> None:
    source = _TimmLike(width=8, depth=4)
    candidate = copy.deepcopy(source)
    inject_timm_qv_lora(
        candidate,
        layers="1,2,3",
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    summary = _load_model_state_allowing_extensions(
        candidate,
        source.state_dict(),
        allow_extensions=True,
    )
    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert len(summary["allowed_missing_keys"]) == 12
    assert all(
        ".attn.qkv." in key for key in summary["allowed_missing_keys"]
    )


def test_resume_rejects_incomplete_existing_lora_state() -> None:
    source = _TimmLike(width=8, depth=4)
    inject_timm_qv_lora(
        source,
        layers="1,2,3",
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    state = source.state_dict()
    state.pop("blocks.2.attn.qkv.v_lora.up.weight")
    candidate = copy.deepcopy(source)
    with pytest.raises(RuntimeError, match="Resume partial load"):
        _load_model_state_allowing_extensions(
            candidate,
            state,
            allow_extensions=True,
        )


def test_train_cli_records_locked_lora_config() -> None:
    args = parse_args(
        [
            "--model-type",
            "timm_classifier",
            "--timm-model-name",
            "vit_small_patch16_dinov3.lvd1689m",
            "--timm-qv-lora",
            "--timm-qv-lora-layers",
            "8,9,10,11",
            "--timm-qv-lora-rank",
            "4",
            "--timm-qv-lora-alpha",
            "8",
            "--timm-qv-lora-dropout",
            "0.05",
        ]
    )
    model_config, train_config, _augmentation_config = build_configs(args)
    assert model_config.timm_qv_lora is True
    assert model_config.timm_qv_lora_layers == "8,9,10,11"
    assert model_config.timm_qv_lora_rank == 4
    assert model_config.timm_qv_lora_alpha == 8.0
    assert model_config.timm_qv_lora_dropout == 0.05
    assert train_config.trainable_module_prefixes == ",".join(
        timm_qv_lora_parameter_prefixes("8,9,10,11")
    )


def test_lora_rejects_non_timm_model_type() -> None:
    with pytest.raises(ValueError, match="only by model_type=timm_classifier"):
        create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="resnet50",
                pretrained=False,
                timm_qv_lora=True,
            ),
        )


def test_lora_rejects_unverified_timm_model_contract() -> None:
    with pytest.raises(ValueError, match="fail-closed"):
        create_model(
            num_classes=5,
            model_config=ModelConfig(
                model_type="timm_classifier",
                timm_model_name="vit_tiny_patch16_224",
                pretrained=False,
                timm_qv_lora=True,
            ),
        )


def test_lora_rejects_attention_that_may_bypass_qkv_forward() -> None:
    model = _TimmLike(width=8, depth=4)
    model.blocks[2].attn.q_bias = nn.Parameter(torch.zeros(8))
    with pytest.raises(TypeError, match="bypass"):
        inject_timm_qv_lora(
            model,
            layers="2",
            rank=2,
            alpha=4.0,
            dropout=0.05,
        )

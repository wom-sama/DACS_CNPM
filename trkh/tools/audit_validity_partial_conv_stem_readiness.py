from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    HybridConvStem,
    ValidityPartialConv2d,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_bbox_logpolar_stem_a0 import _dataset_tree_identity
from trkh.tools.audit_moga_tokenizer_readiness import (
    _amp_forward_backward_benchmark,
    _move_batch,
)
from trkh.tools.audit_visual_contrast_attention_readiness import (
    _build_train_only_dataset,
    _forward_logits,
    _prepare_output_dir,
    _state_sha256,
)


METHOD = "validity_partial_conv_stem"
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_STEM_SHAPE = (32, 256, 32, 32)
EXPECTED_OFFICIAL_COMMIT = "610d373f35257887d45adae84c86d0ce7ad808ec"
EXPECTED_OFFICIAL_TREE = "39e37de8bdce67efd5ceee3008fe0c94bdd9cd5f"
MAX_EQUATION_ERROR = 1e-6
MAX_ALL_VALID_ERROR = 1e-6
MAX_FILL_INVARIANCE_ERROR = 1e-6
MIN_CONTROL_FILL_DELTA = 1e-6
MAX_ONNX_ERROR = 5e-5
MAX_RUNTIME_RATIO = 1.35
MAX_PEAK_VRAM_GIB = 7.5
BENCHMARK_REPEATS = 3

LOCKED_HASHES = {
    "checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "resolved_config": "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    "launcher_args": "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    "data_yaml": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "protocol": "1838135e00e2f4c38d9cbf1d19fbf851a6e715b2676f396236f2a6f010312cda",
    "padding_paper": "6b6d1ceee5bf69e7b0a6b93ef26b6681c8d479601ba70e16c9968bbf82ccc93f",
    "inpainting_paper": "9d902b164c4563c97cf9ca9f0a7d5508bb292ae4204ce4173cf3f8056009c73b",
    "official_source": "ca92d642523b7b18e56c981d6198e29e6b7cdbfa7469b3a07f5f93e741435116",
    "official_readme": "7578fea2adcbaab9ac35ad4f1f19b16296a40d868d60243523c262d5ad077dd0",
    "official_license": "8dc73b75a37967ada9d2a4b7643b155fad5b17e409319f3aa7c5d6092bf0d3d0",
    "current_best_command": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "current_best_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    keeper = Path(
        "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
        "bboxprior_120b_2e_20260701"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Train-only engineering readiness audit for the TRKH validity-aware "
            "partial-convolution stem. Validation and test access are forbidden."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=keeper / "checkpoints/best.pt")
    parser.add_argument("--resolved-config", type=Path, default=keeper / "resolved_config.json")
    parser.add_argument("--source-launcher-args", type=Path, default=keeper / "launcher_args.json")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/TRKH_5CLASS_VALIDITY_PARTIAL_CONV_STEM_PROTOCOL_20260720.md"),
    )
    parser.add_argument(
        "--padding-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\partial_convolution_padding_arxiv1811.11718.pdf"
        ),
    )
    parser.add_argument(
        "--inpainting-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\partial_convolution_inpainting_eccv2018.pdf"
        ),
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\NVIDIA_partialconv"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _git_value(repo: Path, expression: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", expression],
        text=True,
        encoding="utf-8",
    ).strip()


def _git_status(repo: Path, *, tracked_only: bool = False) -> list[str]:
    command = ["git", "-C", str(repo), "status", "--porcelain"]
    if tracked_only:
        command.append("--untracked-files=no")
    output = subprocess.check_output(command, text=True, encoding="utf-8")
    return [line for line in output.splitlines() if line.strip()]


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_repo).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "resolved_config": Path(args.resolved_config).resolve(),
        "launcher_args": Path(args.source_launcher_args).resolve(),
        "data_yaml": Path(args.data).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "padding_paper": Path(args.padding_paper).resolve(),
        "inpainting_paper": Path(args.inpainting_paper).resolve(),
        "official_source": official / "models/partialconv2d.py",
        "official_readme": official / "README.MD",
        "official_license": official / "LICENSE",
        "current_best_command": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "current_best_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _model_configs(source: Mapping[str, object]) -> tuple[Dict[str, object], Dict[str, object]]:
    control = dict(source)
    control.update(
        {
            "stem_architecture": "conv_pool",
            "stem_normalization": "batch",
            "stem_convolution": "standard",
            "stem_pooling_mode": "max",
            "pretrained": False,
        }
    )
    candidate = dict(control)
    candidate["stem_convolution"] = "validity_partial"
    return control, candidate


def _config_differences(
    left: Mapping[str, object], right: Mapping[str, object]
) -> list[str]:
    return sorted(
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    )


def _state_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return bool(
        list(left_state) == list(right_state)
        and all(torch.equal(value, right_state[name]) for name, value in left_state.items())
    )


def _module_type_differences(left: nn.Module, right: nn.Module) -> list[Dict[str, str]]:
    left_modules = dict(left.named_modules())
    right_modules = dict(right.named_modules())
    if set(left_modules) != set(right_modules):
        return [{"name": "<module-schema>", "control": "different", "candidate": "different"}]
    return [
        {
            "name": name,
            "control": type(left_modules[name]).__name__,
            "candidate": type(right_modules[name]).__name__,
        }
        for name in sorted(left_modules)
        if type(left_modules[name]) is not type(right_modules[name])
    ]


def _official_equation_oracle(source_path: Path) -> Dict[str, object]:
    namespace: Dict[str, object] = {}
    source = Path(source_path).read_text(encoding="utf-8")
    exec(compile(source, str(source_path), "exec"), namespace)
    official_class = namespace.get("PartialConv2d")
    if not isinstance(official_class, type):
        raise RuntimeError("Reviewed NVIDIA source did not define PartialConv2d.")

    set_seed(20260720)
    inputs = torch.randn(2, 2, 13, 15) * 0.2
    mask = torch.ones(2, 1, 13, 15)
    mask[0, :, 3:7, 4:8] = 0.0
    mask[1, :, 8:11, 9:13] = 0.0
    candidate = ValidityPartialConv2d(
        2, 3, kernel_size=3, stride=1, padding=1, bias=False
    ).eval()
    official = official_class(
        2,
        3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        multi_channel=False,
        return_mask=True,
    ).eval()
    with torch.no_grad():
        weights = torch.linspace(-0.05, 0.05, candidate.weight.numel()).reshape_as(
            candidate.weight
        )
        candidate.weight.copy_(weights)
        official.weight.copy_(weights)
        candidate_output, candidate_mask, candidate_count = candidate.forward_with_mask(
            inputs, mask.bool()
        )
        official_output, official_mask = official(inputs, mask)
    interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    interior_error = float(
        (candidate_output[interior] - official_output[interior]).abs().max().item()
    )
    border_error = float((candidate_output - official_output).abs().max().item())
    return {
        "reviewed_source_sha256": _sha256(source_path),
        "interior_output_max_abs_error": interior_error,
        "interior_mask_exact": bool(
            torch.equal(candidate_mask[interior], official_mask[interior].bool())
        ),
        "deliberate_tensor_border_delta": border_error,
        "candidate_count_minimum": float(candidate_count.min().item()),
        "candidate_count_maximum": float(candidate_count.max().item()),
        "finite": bool(
            torch.isfinite(candidate_output).all()
            and torch.isfinite(official_output).all()
        ),
    }


def _selected_gradient_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    names = [
        "stem.blocks.0.block.conv.weight",
        "stem.blocks.1.block.conv.weight",
        "stem.blocks.2.block.conv.weight",
        "patch_embed.proj.weight",
    ]
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(f"Missing locked gradient parameters: {missing}")
    return {name: parameters[name] for name in names}


def _summarize_gradients(model: nn.Module) -> Dict[str, object]:
    rows: Dict[str, Dict[str, object]] = {}
    for name, parameter in _selected_gradient_parameters(model).items():
        gradient = parameter.grad
        rows[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero_elements": (
                int(torch.count_nonzero(gradient).item()) if gradient is not None else 0
            ),
            "maximum_absolute": (
                float(gradient.detach().abs().max().item()) if gradient is not None else None
            ),
        }
    return {
        "parameters": rows,
        "passed": bool(
            all(
                row["present"]
                and row["finite"]
                and int(row["nonzero_elements"]) > 0
                for row in rows.values()
            )
        ),
    }


def _all_valid_equivalence(
    control: nn.Module,
    candidate: nn.Module,
    *,
    device: torch.device,
) -> Dict[str, object]:
    base = torch.linspace(
        -1.0,
        1.0,
        3 * 256 * 256,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 256, 256)
    mask = torch.ones(1, 256, 256, device=device, dtype=torch.bool)
    control.eval()
    candidate.eval()
    with torch.inference_mode():
        control_stem = control.stem(base)
        candidate_stem = candidate.stem(base, image_valid_mask=mask)
        control_features = control.forward_features(base, image_valid_mask=mask)
        candidate_features = candidate.forward_features(base, image_valid_mask=mask)
        control_logits = classification_logits_from_features(control, control_features)
        candidate_logits = classification_logits_from_features(candidate, candidate_features)

    def gradients(model: nn.Module) -> Dict[str, Tensor]:
        model.zero_grad(set_to_none=True)
        probe = base.detach().clone().requires_grad_(True)
        features = model.forward_features(probe, image_valid_mask=mask)
        logits = classification_logits_from_features(model, features)
        selected = _selected_gradient_parameters(model)
        values = torch.autograd.grad(
            logits.float().square().sum(),
            [probe, *selected.values()],
            retain_graph=False,
            create_graph=False,
        )
        return {
            "input": values[0].detach(),
            **{
                name: value.detach()
                for name, value in zip(selected, values[1:])
            },
        }

    control_gradients = gradients(control)
    candidate_gradients = gradients(candidate)
    gradient_errors = {
        name: float(
            (control_gradients[name] - candidate_gradients[name]).abs().max().item()
        )
        for name in control_gradients
    }
    gradients_finite_nonzero = bool(
        all(
            torch.isfinite(value).all() and int(torch.count_nonzero(value).item()) > 0
            for value in control_gradients.values()
        )
        and all(
            torch.isfinite(value).all() and int(torch.count_nonzero(value).item()) > 0
            for value in candidate_gradients.values()
        )
    )
    return {
        "stem_max_abs_error": float((control_stem - candidate_stem).abs().max().item()),
        "pooled_max_abs_error": float(
            (control_features["pooled"] - candidate_features["pooled"])
            .abs()
            .max()
            .item()
        ),
        "logits_max_abs_error": float((control_logits - candidate_logits).abs().max().item()),
        "stem_bit_exact": bool(torch.equal(control_stem, candidate_stem)),
        "logits_bit_exact": bool(torch.equal(control_logits, candidate_logits)),
        "gradient_max_abs_errors": gradient_errors,
        "maximum_gradient_error": float(max(gradient_errors.values())),
        "gradients_finite_nonzero": gradients_finite_nonzero,
    }


def _masked_fill_invariance(
    control: nn.Module,
    candidate: nn.Module,
    *,
    device: torch.device,
) -> Dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(20260721)
    base = torch.randn(1, 3, 256, 256, generator=generator).to(device)
    mask = torch.zeros(1, 1, 256, 256, device=device, dtype=torch.bool)
    mask[:, :, 31:229, 43:217] = True
    mask[:, :, 84:126, 43:77] = False
    first = torch.where(mask, base, torch.full_like(base, -4.0))
    second = torch.where(mask, base, torch.full_like(base, 6.0))
    metadata = {"image_mask": mask[:, 0]}
    control.eval()
    candidate.eval()
    with torch.inference_mode():
        candidate_stem_first = candidate.stem(first, image_valid_mask=mask)
        candidate_stem_second = candidate.stem(second, image_valid_mask=mask)
        control_stem_first = control.stem(first)
        control_stem_second = control.stem(second)
        candidate_logits_first, _ = _forward_logits(candidate, first, metadata)
        candidate_logits_second, _ = _forward_logits(candidate, second, metadata)
        control_logits_first, _ = _forward_logits(control, first, metadata)
        control_logits_second, _ = _forward_logits(control, second, metadata)
    return {
        "candidate_stem_max_abs_delta": float(
            (candidate_stem_first - candidate_stem_second).abs().max().item()
        ),
        "candidate_logits_max_abs_delta": float(
            (candidate_logits_first - candidate_logits_second).abs().max().item()
        ),
        "control_stem_max_abs_delta": float(
            (control_stem_first - control_stem_second).abs().max().item()
        ),
        "control_logits_max_abs_delta": float(
            (control_logits_first - control_logits_second).abs().max().item()
        ),
        "valid_fraction": float(mask.float().mean().item()),
        "all_outputs_finite": bool(
            all(
                torch.isfinite(value).all()
                for value in (
                    candidate_stem_first,
                    candidate_stem_second,
                    control_stem_first,
                    control_stem_second,
                    candidate_logits_first,
                    candidate_logits_second,
                    control_logits_first,
                    control_logits_second,
                )
            )
        ),
    }


def _mask_geometry(
    candidate: nn.Module,
    images: Tensor,
    image_mask: Tensor,
) -> Dict[str, object]:
    if not isinstance(candidate.stem, HybridConvStem):
        raise TypeError("Candidate does not contain HybridConvStem.")
    candidate.eval()
    with torch.inference_mode():
        stem_output, trace = candidate.stem(
            images,
            image_valid_mask=image_mask,
            return_mask_trace=True,
        )
    input_fraction = float(image_mask.float().mean().item())
    rows: list[Dict[str, object]] = []
    fractions = [input_fraction]
    expected_valid_shapes = [(256, 256), (128, 128), (64, 64)]
    expected_pooled_shapes = [(128, 128), (64, 64), (32, 32)]
    geometry_exact = True
    for index, block in enumerate(trace["blocks"]):
        valid_count = block["valid_count"]
        updated_mask = block["updated_mask"]
        pooled_mask = block["pooled_mask"]
        activation = block["activation"]
        pooled_fraction = float(pooled_mask.float().mean().item())
        fractions.append(pooled_fraction)
        geometry_exact = bool(
            geometry_exact
            and tuple(valid_count.shape[-2:]) == expected_valid_shapes[index]
            and tuple(updated_mask.shape[-2:]) == expected_valid_shapes[index]
            and tuple(pooled_mask.shape[-2:]) == expected_pooled_shapes[index]
            and tuple(activation.shape[-2:]) == expected_pooled_shapes[index]
        )
        rows.append(
            {
                "block": int(index + 1),
                "valid_count_shape": [int(value) for value in valid_count.shape],
                "valid_count_minimum": float(valid_count.min().item()),
                "valid_count_maximum": float(valid_count.max().item()),
                "valid_count_finite": bool(torch.isfinite(valid_count).all()),
                "updated_valid_pixels": int(updated_mask.sum().item()),
                "updated_valid_fraction": float(updated_mask.float().mean().item()),
                "pooled_valid_pixels": int(pooled_mask.sum().item()),
                "pooled_valid_fraction": pooled_fraction,
                "pooled_mask_nonempty": bool(pooled_mask.any()),
                "activation_shape": [int(value) for value in activation.shape],
                "activation_finite": bool(torch.isfinite(activation).all()),
            }
        )
    return {
        "stem_shape": [int(value) for value in stem_output.shape],
        "stem_finite": bool(torch.isfinite(stem_output).all()),
        "input_valid_fraction": input_fraction,
        "valid_fractions": fractions,
        "valid_fractions_monotonic": bool(
            all(right + 1e-12 >= left for left, right in zip(fractions, fractions[1:]))
        ),
        "geometry_exact": geometry_exact,
        "all_counts_finite_bounded": bool(
            all(
                row["valid_count_finite"]
                and float(row["valid_count_minimum"]) >= 0.0
                and float(row["valid_count_maximum"]) <= 9.0
                for row in rows
            )
        ),
        "all_masks_nonempty": bool(all(row["pooled_mask_nonempty"] for row in rows)),
        "blocks": rows,
    }


def _write_mechanism_sheet(
    *,
    output_path: Path,
    control: nn.Module,
    candidate: nn.Module,
    images: Tensor,
    image_mask: Tensor,
    labels: Tensor,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control.eval()
    candidate.eval()
    with torch.inference_mode():
        control_stem = control.stem(images)
        candidate_stem = candidate.stem(images, image_valid_mask=image_mask)
    control_energy = F.interpolate(
        control_stem.detach().float().abs().mean(dim=1, keepdim=True),
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    candidate_energy = F.interpolate(
        candidate_stem.detach().float().abs().mean(dim=1, keepdim=True),
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    difference = (candidate_energy - control_energy).abs()
    mean_tensor = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    display_images = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)

    figure, axes = plt.subplots(5, 5, figsize=(14, 14), constrained_layout=True)
    column_titles = ["Input", "Validity mask", "Standard energy", "Partial energy", "Abs delta"]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10)
    row_summaries = []
    for row in range(5):
        image = display_images[row].detach().cpu().permute(1, 2, 0).numpy()
        mask = image_mask[row].detach().cpu().bool().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        overlay = image.copy()
        overlay[~mask] = 0.25 * overlay[~mask] + np.array([0.75, 0.05, 0.05])
        control_map = control_energy[row].detach().cpu().numpy()
        candidate_map = candidate_energy[row].detach().cpu().numpy()
        delta_map = difference[row].detach().cpu().numpy()
        energy_max = max(float(control_map.max()), float(candidate_map.max()), 1e-8)
        delta_max = max(float(delta_map.max()), 1e-8)
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(overlay)
        axes[row, 2].imshow(control_map, cmap="viridis", vmin=0.0, vmax=energy_max)
        axes[row, 3].imshow(candidate_map, cmap="viridis", vmin=0.0, vmax=energy_max)
        axes[row, 4].imshow(delta_map, cmap="magma", vmin=0.0, vmax=delta_max)
        label = int(labels[row].item())
        axes[row, 0].set_ylabel(f"{label}: {class_names[label]}", fontsize=8)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        row_summaries.append(
            {
                "class_index": label,
                "valid_fraction": float(mask.mean()),
                "standard_energy_mean": float(control_map.mean()),
                "candidate_energy_mean": float(candidate_map.mean()),
                "absolute_delta_mean": float(delta_map.mean()),
                "absolute_delta_maximum": float(delta_map.max()),
            }
        )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "rows": row_summaries,
        "class_coverage": sorted(int(value) for value in labels.detach().cpu().tolist()),
        "written": output_path.is_file() and output_path.stat().st_size > 0,
    }


class _CandidateExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, image_valid_mask: Tensor, bbox: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            image_valid_mask=image_valid_mask,
            bbox_token_prior=bbox,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _export_candidate(model: nn.Module, output_dir: Path) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    export_path = output_dir / "validity_partial_keeper_dynamic_batch.onnx"
    wrapper = _CandidateExportWrapper(model.cpu().eval())
    image = torch.linspace(-1.0, 1.0, 3 * 256 * 256).reshape(1, 3, 256, 256)
    image_mask = torch.ones(1, 256, 256, dtype=torch.bool)
    image_mask[:, :, :37] = False
    bbox = torch.tensor([[0.10, 0.20, 0.90, 0.80]], dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (image, image_mask, bbox),
        export_path,
        input_names=["images", "image_valid_mask", "bbox"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "image_valid_mask": {0: "batch"},
            "bbox": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    graph = onnx.load(str(export_path))
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    custom_operators = [
        value for value in operators if value.startswith("ATen") or value.startswith("Python")
    ]
    session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    batch_images = torch.cat((image, image.flip(-1)), dim=0)
    batch_masks = torch.cat((image_mask, image_mask.flip(-1)), dim=0)
    batch_bboxes = torch.cat((bbox, torch.tensor([[0.15, 0.12, 0.84, 0.91]])), dim=0)
    with torch.inference_mode():
        expected = wrapper(batch_images, batch_masks, batch_bboxes).numpy()
    observed = session.run(
        None,
        {
            "images": batch_images.numpy(),
            "image_valid_mask": batch_masks.numpy(),
            "bbox": batch_bboxes.numpy(),
        },
    )[0]
    maximum_error = float(np.max(np.abs(expected - observed)))
    inputs = {
        value.name: {"shape": list(value.shape), "type": value.type}
        for value in session.get_inputs()
    }
    dynamic_batch = bool(
        all(value["shape"][0] == "batch" for value in inputs.values())
        and session.get_outputs()[0].shape[0] == "batch"
    )
    return {
        "path": str(export_path.resolve()),
        "sha256": _sha256(export_path),
        "size_bytes": int(export_path.stat().st_size),
        "operators": operators,
        "custom_operators": custom_operators,
        "inputs": inputs,
        "dynamic_batch": dynamic_batch,
        "image_valid_mask_present": "image_valid_mask" in inputs,
        "batch2_maximum_abs_error": maximum_error,
        "batch2_argmax_exact": bool(
            np.array_equal(np.argmax(expected, axis=1), np.argmax(observed, axis=1))
        ),
        "passed": bool(
            not custom_operators
            and dynamic_batch
            and "image_valid_mask" in inputs
            and maximum_error <= MAX_ONNX_ERROR
        ),
    }


def assess_readiness(
    *,
    checks: Mapping[str, bool],
    peak_vram_gib: float,
    runtime_ratio: float,
) -> Dict[str, object]:
    resolved = {str(name): bool(value) for name, value in checks.items()}
    resolved["peak_vram_within_budget"] = bool(
        math.isfinite(peak_vram_gib) and peak_vram_gib <= MAX_PEAK_VRAM_GIB
    )
    resolved["runtime_ratio_within_budget"] = bool(
        math.isfinite(runtime_ratio) and runtime_ratio <= MAX_RUNTIME_RATIO
    )
    failed = [name for name, passed in resolved.items() if not passed]
    return {
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": resolved,
        "failed_checks": failed,
        "thresholds": {
            "maximum_equation_error": MAX_EQUATION_ERROR,
            "maximum_all_valid_error": MAX_ALL_VALID_ERROR,
            "maximum_fill_invariance_error": MAX_FILL_INVARIANCE_ERROR,
            "minimum_control_fill_delta": MIN_CONTROL_FILL_DELTA,
            "maximum_onnx_error": MAX_ONNX_ERROR,
            "maximum_runtime_ratio": MAX_RUNTIME_RATIO,
            "maximum_peak_vram_gib": MAX_PEAK_VRAM_GIB,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _replay_gate_inputs(path: Path) -> Dict[str, object]:
    payload = _load_json(path)
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("gate_inputs.json lacks checks.")
    return assess_readiness(
        checks={str(key): bool(value) for key, value in checks.items()},
        peak_vram_gib=float(payload["peak_vram_gib"]),
        runtime_ratio=float(payload["runtime_ratio"]),
    )


def replay_summary(path: Path) -> Dict[str, object]:
    summary = _load_json(path)
    gate_inputs_path = Path(str(summary["gate_inputs_path"]))
    replayed = _replay_gate_inputs(gate_inputs_path)
    expected = summary.get("gate")
    exact = replayed == expected
    return {
        "summary_path": str(Path(path).resolve()),
        "gate_inputs_path": str(gate_inputs_path.resolve()),
        "expected_gate_sha256": hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "replayed_gate_sha256": hashlib.sha256(
            json.dumps(replayed, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "exact": exact,
        "replayed_gate": replayed,
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    gate = summary["gate"]
    runtime = summary["resources"]
    lines = [
        "# Validity Partial-Conv Stem Stage A",
        "",
        f"- Status: `{summary['status']}`",
        f"- Smoke permission: `{gate['smoke_permission']}`",
        f"- Failed checks: `{gate['failed_checks']}`",
        f"- Runtime ratio: `{runtime['runtime_ratio']:.6f}`",
        f"- Peak VRAM GiB: `{runtime['peak_vram_gib']:.6f}`",
        "- Validation used: `false`",
        "- Test used: `false`",
        "- Trainable checkpoint written: `false`",
        "",
        "This engineering audit does not promote current-best commands or authorize a full train.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output_dir: Path) -> Path:
    manifest_path = output_dir / "artifact_manifest.json"
    artifacts = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name.casefold()):
        if path == manifest_path or not path.is_file():
            continue
        artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    _write_json(
        manifest_path,
        {
            "artifacts": artifacts,
            "raw_dataset_modified": False,
            "validation_used": False,
            "test_used": False,
            "trainable_checkpoint_written": False,
        },
    )
    return manifest_path


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal validity-partial Stage A requires CUDA.")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("Locked batch sizes are AMP=32 and FP32=2.")
    if int(args.num_workers) != 4 or int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError("Locked settings are workers=4 and benchmark-repeats=3.")
    if int(args.seed) != 42:
        raise ValueError("Locked Stage-A seed is 42.")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for a formal audit.")

    output_dir = _prepare_output_dir(Path(args.output_dir))
    paths = _source_paths(args)
    dataset_identity_before = _dataset_tree_identity(paths["data_yaml"])
    observed_hashes = {name: _sha256(path) for name, path in paths.items()}
    hash_checks = {
        name: observed_hashes.get(name) == expected
        for name, expected in LOCKED_HASHES.items()
    }
    official_repo = Path(args.official_repo).resolve()
    official_commit = _git_value(official_repo, "HEAD")
    official_tree = _git_value(official_repo, "HEAD^{tree}")
    official_status_before = _git_status(official_repo)
    repository_head = _git_value(Path.cwd(), "HEAD")
    repository_status_tracked = _git_status(Path.cwd(), tracked_only=True)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    model_state = checkpoint.get("model_state")
    if (
        not isinstance(source_model_config, Mapping)
        or not isinstance(class_names, list)
        or not isinstance(model_state, Mapping)
    ):
        raise ValueError("Keeper lacks model_config, class_names, or model_state.")
    if len(class_names) != 5:
        raise ValueError(f"Expected five classes, got {len(class_names)}.")
    resolved_config = _load_json(paths["resolved_config"])
    control_config, candidate_config = _model_configs(source_model_config)
    config_differences = _config_differences(control_config, candidate_config)

    set_seed(int(args.seed))
    control = create_model(num_classes=5, model_config=control_config)
    set_seed(int(args.seed))
    candidate = create_model(num_classes=5, model_config=candidate_config)
    initial_state_equal = _state_equal(control, candidate)
    initial_control_sha = _state_sha256(control)
    initial_candidate_sha = _state_sha256(candidate)
    parameter_count_control = sum(parameter.numel() for parameter in control.parameters())
    parameter_count_candidate = sum(parameter.numel() for parameter in candidate.parameters())
    module_differences = _module_type_differences(control, candidate)

    default_config = dict(control_config)
    default_config.pop("stem_convolution", None)
    set_seed(int(args.seed))
    default_model = create_model(num_classes=5, model_config=default_config).eval()
    set_seed(int(args.seed))
    explicit_model = create_model(num_classes=5, model_config=control_config).eval()
    schema_probe = torch.linspace(-1.0, 1.0, 3 * 256 * 256).reshape(1, 3, 256, 256)
    with torch.inference_mode():
        default_logits, _ = _forward_logits(default_model, schema_probe, {})
        explicit_logits, _ = _forward_logits(explicit_model, schema_probe, {})
    default_state_equal = _state_equal(default_model, explicit_model)
    default_logits_equal = bool(torch.equal(default_logits, explicit_logits))
    del default_model, explicit_model, default_logits, explicit_logits, schema_probe

    control_load = load_model_state(control, dict(model_state), strict=True)
    candidate_load = load_model_state(candidate, dict(model_state), strict=True)
    loaded_state_equal = _state_equal(control, candidate)
    loaded_control_sha = _state_sha256(control)
    loaded_candidate_sha = _state_sha256(candidate)
    strict_load_empty = bool(tuple(control_load) == ([], []) and tuple(candidate_load) == ([], []))

    expected_module_differences = [
        {
            "name": f"stem.blocks.{index}.block.conv",
            "control": "Conv2d",
            "candidate": "ValidityPartialConv2d",
        }
        for index in range(3)
    ]
    candidate_convolutions = [block.block.conv for block in candidate.stem.blocks]
    placement_exact = bool(
        isinstance(control.stem, HybridConvStem)
        and isinstance(candidate.stem, HybridConvStem)
        and all(type(block.block.conv) is nn.Conv2d for block in control.stem.blocks)
        and all(isinstance(value, ValidityPartialConv2d) for value in candidate_convolutions)
        and module_differences == expected_module_differences
    )
    official_oracle = _official_equation_oracle(paths["official_source"])

    dataset = _build_train_only_dataset(
        data_yaml=paths["data_yaml"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    data_spec = load_data_spec(paths["data_yaml"])
    train_root = Path(data_spec.split_images_dir("train")).resolve()
    balanced_indices: Dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        balanced_indices.setdefault(int(sample.primary_label), int(index))
        if len(balanced_indices) == 5:
            break
    if set(balanced_indices) != set(range(5)):
        raise RuntimeError("Could not create a deterministic five-class train cohort.")
    ordered_indices = [balanced_indices[index] for index in range(5)]
    balanced_paths = [str(dataset.samples[index].image_path.resolve()) for index in ordered_indices]
    resource_paths = [
        str(dataset.samples[index].image_path.resolve()) for index in range(int(args.batch_size))
    ]
    train_only_paths = bool(
        all(Path(path).is_relative_to(train_root) for path in balanced_paths + resource_paths)
    )
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="validity_partial_stage_a_train_only",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **dataloader_kwargs,
    )
    resource_batch_cpu = next(iter(loader))
    balanced_batch_cpu = default_collate([dataset[index] for index in ordered_indices])
    del loader
    gc.collect()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    images, labels, metadata = _move_batch(resource_batch_cpu, device=device)
    balanced_images, balanced_labels, balanced_metadata = _move_batch(
        balanced_batch_cpu, device=device
    )
    image_mask = metadata.get("image_mask")
    balanced_mask = balanced_metadata.get("image_mask")
    if not torch.is_tensor(image_mask) or not torch.is_tensor(balanced_mask):
        raise RuntimeError("Train-only dataset did not emit image_mask metadata.")
    control = control.to(device).eval()
    candidate = candidate.to(device).eval()

    all_valid = _all_valid_equivalence(control, candidate, device=device)
    fill_invariance = _masked_fill_invariance(control, candidate, device=device)
    geometry = _mask_geometry(candidate, images, image_mask)
    mean, std = checkpoint_input_normalization(checkpoint)
    sheet = _write_mechanism_sheet(
        output_path=output_dir / "validity_partial_stem_mechanism_sheet.png",
        control=control,
        candidate=candidate,
        images=balanced_images,
        image_mask=balanced_mask,
        labels=balanced_labels,
        class_names=class_names,
        mean=mean,
        std=std,
    )

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    fp32_metadata = {
        key: value[: int(args.fp32_batch_size)] for key, value in metadata.items()
    }
    fp32_logits, _ = _forward_logits(
        candidate,
        images[: int(args.fp32_batch_size)],
        fp32_metadata,
    )
    fp32_loss = F.cross_entropy(fp32_logits, labels[: int(args.fp32_batch_size)])
    fp32_loss.backward()
    fp32_gradient = _summarize_gradients(candidate)
    fp32_peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    fp32 = {
        "batch_size": int(args.fp32_batch_size),
        "logits_finite": bool(torch.isfinite(fp32_logits).all()),
        "loss": float(fp32_loss.detach().item()),
        "loss_finite": bool(torch.isfinite(fp32_loss)),
        "gradient": fp32_gradient,
        "peak_vram_gib": fp32_peak,
    }

    load_model_state(candidate, dict(model_state), strict=True)
    candidate = candidate.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    control = control.to(device).train()
    control_benchmark = _amp_forward_backward_benchmark(
        model=control,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    control = control.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    candidate = candidate.to(device).train()
    candidate_benchmark = _amp_forward_backward_benchmark(
        model=candidate,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    amp_gradient = _summarize_gradients(candidate)
    runtime_ratio = float(candidate_benchmark["median_seconds"]) / max(
        float(control_benchmark["median_seconds"]), 1e-12
    )
    peak_vram_gib = max(
        fp32_peak,
        float(control_benchmark["peak_vram_gib"]),
        float(candidate_benchmark["peak_vram_gib"]),
    )

    load_model_state(candidate, dict(model_state), strict=True)
    candidate.zero_grad(set_to_none=True)
    export = _export_candidate(candidate, output_dir)
    official_status_after = _git_status(official_repo)
    dataset_identity_after = _dataset_tree_identity(paths["data_yaml"])
    dataset_identity_exact = dataset_identity_before == dataset_identity_after

    checks = {
        "all_locked_hashes": all(hash_checks.values()),
        "official_commit_tree_clean": bool(
            official_commit == EXPECTED_OFFICIAL_COMMIT
            and official_tree == EXPECTED_OFFICIAL_TREE
            and not official_status_before
            and not official_status_after
        ),
        "repository_tracked_clean": not repository_status_tracked,
        "only_config_difference": config_differences == ["stem_convolution"],
        "default_state_bit_identical": default_state_equal,
        "default_logits_bit_identical": default_logits_equal,
        "initial_state_bit_identical": initial_state_equal,
        "initial_state_hash_identical": initial_control_sha == initial_candidate_sha,
        "parameter_count_identical": parameter_count_control == parameter_count_candidate,
        "strict_keeper_load": strict_load_empty,
        "loaded_state_bit_identical": loaded_state_equal,
        "loaded_state_hash_identical": loaded_control_sha == loaded_candidate_sha,
        "exact_three_partial_convolutions": placement_exact,
        "official_equation_oracle": bool(
            official_oracle["finite"]
            and official_oracle["interior_mask_exact"]
            and float(official_oracle["interior_output_max_abs_error"])
            <= MAX_EQUATION_ERROR
        ),
        "all_valid_stem_equivalent": float(all_valid["stem_max_abs_error"])
        <= MAX_ALL_VALID_ERROR,
        "all_valid_logits_equivalent": float(all_valid["logits_max_abs_error"])
        <= MAX_ALL_VALID_ERROR,
        "all_valid_gradients_equivalent": bool(
            all_valid["gradients_finite_nonzero"]
            and float(all_valid["maximum_gradient_error"]) <= MAX_ALL_VALID_ERROR
        ),
        "candidate_masked_fill_invariant": bool(
            fill_invariance["all_outputs_finite"]
            and float(fill_invariance["candidate_stem_max_abs_delta"])
            <= MAX_FILL_INVARIANCE_ERROR
            and float(fill_invariance["candidate_logits_max_abs_delta"])
            <= MAX_FILL_INVARIANCE_ERROR
        ),
        "control_masked_fill_distinct": bool(
            float(fill_invariance["control_stem_max_abs_delta"])
            > MIN_CONTROL_FILL_DELTA
            and float(fill_invariance["control_logits_max_abs_delta"])
            > MIN_CONTROL_FILL_DELTA
        ),
        "dataset_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "train_only_paths": train_only_paths,
        "five_class_cohort": sorted(balanced_labels.detach().cpu().tolist())
        == [0, 1, 2, 3, 4],
        "mask_geometry": bool(
            geometry["stem_shape"] == list(EXPECTED_STEM_SHAPE)
            and geometry["stem_finite"]
            and geometry["valid_fractions_monotonic"]
            and geometry["geometry_exact"]
            and geometry["all_counts_finite_bounded"]
            and geometry["all_masks_nonempty"]
        ),
        "mechanism_sheet": bool(
            sheet["written"] and sheet["class_coverage"] == [0, 1, 2, 3, 4]
        ),
        "fp32_forward_backward": bool(
            fp32["logits_finite"] and fp32["loss_finite"] and fp32_gradient["passed"]
        ),
        "amp_forward_backward": bool(
            candidate_benchmark["logits_loss_finite"] and amp_gradient["passed"]
        ),
        "onnx_dynamic_batch_mask_input": bool(export["passed"]),
        "raw_dataset_identity_exact": dataset_identity_exact,
        "no_trainable_checkpoint_written": not any(output_dir.rglob("*.pt")),
        "validation_not_used": True,
        "test_not_used": True,
    }
    gate_inputs_path = output_dir / "gate_inputs.json"
    _write_json(
        gate_inputs_path,
        {
            "checks": checks,
            "peak_vram_gib": peak_vram_gib,
            "runtime_ratio": runtime_ratio,
        },
    )
    gate = _replay_gate_inputs(gate_inputs_path)
    replay = {
        "gate_inputs_path": str(gate_inputs_path.resolve()),
        "gate_inputs_sha256": _sha256(gate_inputs_path),
        "replayed_gate": gate,
        "exact": True,
    }
    replay_path = output_dir / "independent_replay.json"
    _write_json(replay_path, replay)

    summary: Dict[str, object] = {
        "method": METHOD,
        "status": "authorized" if gate["smoke_permission"] else "rejected",
        "repository_head": repository_head,
        "gate_inputs_path": str(gate_inputs_path.resolve()),
        "sources": {
            "paths": {name: str(path) for name, path in paths.items()},
            "observed_sha256": observed_hashes,
            "expected_sha256": LOCKED_HASHES,
            "hash_checks": hash_checks,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "official_status_before": official_status_before,
            "official_status_after": official_status_after,
            "validation_loaded": False,
            "test_loaded": False,
            "raw_dataset_modified": not dataset_identity_exact,
        },
        "configuration": {
            "differences": config_differences,
            "control_stem_convolution": control_config["stem_convolution"],
            "candidate_stem_convolution": candidate_config["stem_convolution"],
            "seed": int(args.seed),
        },
        "state_schema": {
            "initial_control_sha256": initial_control_sha,
            "initial_candidate_sha256": initial_candidate_sha,
            "loaded_control_sha256": loaded_control_sha,
            "loaded_candidate_sha256": loaded_candidate_sha,
            "initial_state_equal": initial_state_equal,
            "loaded_state_equal": loaded_state_equal,
            "parameter_count_control": parameter_count_control,
            "parameter_count_candidate": parameter_count_candidate,
            "strict_load_results_empty": strict_load_empty,
            "default_state_equal": default_state_equal,
            "default_logits_equal": default_logits_equal,
            "module_type_differences": module_differences,
            "placement_exact": placement_exact,
        },
        "official_equation_oracle": official_oracle,
        "all_valid_equivalence": all_valid,
        "masked_fill_invariance": fill_invariance,
        "mask_geometry": geometry,
        "mechanism_sheet": sheet,
        "selected_train_paths": {
            "train_root": str(train_root),
            "resource": resource_paths,
            "balanced": balanced_paths,
            "balanced_indices": ordered_indices,
            "train_only": train_only_paths,
        },
        "dataset": {
            "rows": len(dataset),
            "identity_before": dataset_identity_before,
            "identity_after": dataset_identity_after,
            "identity_exact": dataset_identity_exact,
        },
        "dataloader": dataloader_summary,
        "fp32": fp32,
        "resources": {
            "control_amp": control_benchmark,
            "candidate_amp": candidate_benchmark,
            "candidate_amp_gradient": amp_gradient,
            "runtime_ratio": runtime_ratio,
            "peak_vram_gib": peak_vram_gib,
            "export": export,
        },
        "gate": gate,
        "replay": replay,
        "validation_used": False,
        "test_used": False,
        "trainable_checkpoint_written": False,
        "current_best_command_updated": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    external_replay = replay_summary(summary_path)
    if not bool(external_replay["exact"]):
        raise RuntimeError("Disk replay differs from persisted Stage-A gate.")
    _write_report(output_dir / "report.md", summary)
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.replay_summary is not None:
        result = replay_summary(args.replay_summary)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
        if not bool(result["exact"]):
            raise SystemExit(2)
        return
    summary = run_audit(args)
    print(json.dumps(summary["gate"], indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    if not bool(summary["gate"]["smoke_permission"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PROBE_DEPTHS = (3, 7, 11)
QUERY_MARKER = "evidence_query."
RIVALS = (0, 2, 4)


class MultiDepthEvidenceQuery(nn.Module):
    """One-way class queries over frozen multi-depth DINO patch tokens.

    The primary DINO stream never sees a query token.  This preserves the
    inherited classifier exactly and makes the query decoder a bounded,
    class-specific evidence residual rather than another uniform token mixer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int = 5,
        query_dim: int = 64,
        num_heads: int = 4,
        probe_depths: tuple[int, ...] = PROBE_DEPTHS,
        residual_scale: float = 0.5,
        init_seed: int = 20260805,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.query_dim = int(query_dim)
        self.probe_depths = tuple(int(value) for value in probe_depths)
        self.residual_scale = float(residual_scale)
        self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 0))
        self.embed_dim = int(getattr(backbone, "embed_dim", 0))
        depth = len(getattr(backbone, "blocks", ()))
        if (
            self.num_classes != 5
            or self.embed_dim != 384
            or self.num_prefix_tokens != 5
            or depth != 12
            or self.probe_depths != PROBE_DEPTHS
            or self.query_dim <= 0
            or self.query_dim % int(num_heads) != 0
            or not 0.0 < self.residual_scale <= 1.0
        ):
            raise ValueError("B26 frozen DINO/query topology contract changed")
        if not callable(getattr(backbone, "forward_intermediates", None)):
            raise TypeError("B26 backbone must expose forward_intermediates")
        if str(getattr(backbone, "global_pool", "")).lower() != "avg":
            raise ValueError("B26 requires the B9 average-patch classifier")

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        with torch.random.fork_rng(
            devices=list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        ):
            torch.manual_seed(int(init_seed))
            self.evidence_query = nn.ModuleDict(
                {
                    "patch_projection": nn.Linear(
                        self.embed_dim, self.query_dim, bias=False
                    ),
                    "cross_attention": nn.MultiheadAttention(
                        self.query_dim,
                        int(num_heads),
                        dropout=0.0,
                        batch_first=True,
                    ),
                    "query_norm": nn.LayerNorm(self.query_dim),
                    "ffn_norm": nn.LayerNorm(self.query_dim),
                    "ffn": nn.Sequential(
                        nn.Linear(self.query_dim, 2 * self.query_dim),
                        nn.GELU(),
                        nn.Linear(2 * self.query_dim, self.query_dim),
                    ),
                    "output_norm": nn.LayerNorm(self.query_dim),
                    "output": nn.Linear(self.query_dim, 1),
                }
            )
            self.class_queries = nn.Parameter(
                torch.empty(self.num_classes, self.query_dim)
            )
            self.depth_embeddings = nn.Parameter(
                torch.empty(len(self.probe_depths), self.query_dim)
            )
            nn.init.trunc_normal_(self.class_queries, std=0.02)
            nn.init.trunc_normal_(self.depth_embeddings, std=0.02)
            nn.init.xavier_uniform_(self.evidence_query["patch_projection"].weight)
            nn.init.zeros_(self.evidence_query["output"].weight)
            nn.init.zeros_(self.evidence_query["output"].bias)

        self.model_type = "b26_multidepth_evidence_query"
        self.research_track = "pretrained"

    def train(self, mode: bool = True) -> MultiDepthEvidenceQuery:
        super().train(mode)
        self.backbone.eval()
        return self

    def query_parameters(self) -> Iterable[nn.Parameter]:
        yield self.class_queries
        yield self.depth_embeddings
        yield from self.evidence_query.parameters()

    @property
    def query_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.query_parameters())

    def _frozen_features(self, images: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        with torch.no_grad():
            final_tokens, intermediates = self.backbone.forward_intermediates(
                images,
                indices=list(self.probe_depths),
                return_prefix_tokens=False,
                norm=False,
                stop_early=False,
                output_fmt="NLC",
                intermediates_only=False,
            )
            base_logits = self.backbone.forward_head(final_tokens)
        if (
            not isinstance(final_tokens, Tensor)
            or tuple(final_tokens.shape[1:]) != (261, self.embed_dim)
            or len(intermediates) != len(self.probe_depths)
            or any(tuple(value.shape[1:]) != (256, self.embed_dim) for value in intermediates)
            or tuple(base_logits.shape[1:]) != (self.num_classes,)
        ):
            raise ValueError("B26 DINO token/logit geometry changed")
        return (
            final_tokens.detach(),
            [value.detach() for value in intermediates],
            base_logits.detach(),
        )

    def forward_with_trace(
        self, images: Tensor, *, return_attention: bool = True
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        _final_tokens, intermediates, base_logits = self._frozen_features(images)
        query = self.class_queries.unsqueeze(0).expand(int(images.size(0)), -1, -1)
        attention_maps: list[Tensor] = []
        for index, patches in enumerate(intermediates):
            normalized = F.layer_norm(patches, (self.embed_dim,))
            projected = self.evidence_query["patch_projection"](normalized)
            query = query + self.depth_embeddings[index].view(1, 1, -1)
            update, attention = self.evidence_query["cross_attention"](
                self.evidence_query["query_norm"](query),
                projected,
                projected,
                need_weights=bool(return_attention),
                average_attn_weights=False,
            )
            query = query + update
            query = query + self.evidence_query["ffn"](
                self.evidence_query["ffn_norm"](query)
            )
            if return_attention:
                attention_maps.append(attention)
        raw = self.evidence_query["output"](
            self.evidence_query["output_norm"](query)
        ).squeeze(-1)
        residual = self.residual_scale * torch.tanh(raw.float())
        logits = base_logits.float() + residual
        trace: dict[str, Tensor] = {
            "base_logits": base_logits.float(),
            "query_logits": raw.float(),
            "residual_logits": residual,
            "query_tokens": query.float(),
        }
        if attention_maps:
            trace["class_to_patch_attention"] = torch.stack(attention_maps, dim=1).float()
        return logits, trace

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_trace(images, return_attention=False)[0]


def is_query_parameter(name: str) -> bool:
    return name in {"class_queries", "depth_embeddings"} or name.startswith(
        QUERY_MARKER
    )


def configure_query_only(model: MultiDepthEvidenceQuery) -> dict[str, int]:
    counts = {"total": 0, "trainable": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        enabled = is_query_parameter(name)
        parameter.requires_grad_(enabled)
        count = int(parameter.numel())
        counts["total"] += count
        counts["trainable" if enabled else "frozen"] += count
    if counts["trainable"] != model.query_parameter_count:
        raise RuntimeError("B26 query-only parameter contract changed")
    return counts


def query_state_dict(model: MultiDepthEvidenceQuery) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
        if is_query_parameter(name)
    }


def load_query_state_dict(
    model: MultiDepthEvidenceQuery, state: Mapping[str, Tensor]
) -> None:
    expected = {name for name in model.state_dict() if is_query_parameter(name)}
    if set(state) != expected:
        raise ValueError("B26 query state keys changed")
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("B26 query load produced unexpected keys")
    missing = {name for name in incompatible.missing_keys if not is_query_parameter(name)}
    expected_missing = {
        name for name in model.state_dict() if not is_query_parameter(name)
    }
    if missing != expected_missing:
        raise RuntimeError("B26 query load did not omit exactly the backbone")


def boundary_guarded_loss(
    logits: Tensor,
    base_logits: Tensor,
    targets: Tensor,
    task_loss: Callable[[Tensor, Tensor], Tensor],
    *,
    retention_weight: float = 1.0,
    hard_negative_weight: float = 0.10,
    distillation_weight: float = 0.10,
    hard_negative_margin: float = 0.10,
    temperature: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if logits.shape != base_logits.shape or tuple(logits.shape[1:]) != (5,):
        raise ValueError("B26 student/base logits must both be [B,5]")
    if targets.shape != (int(logits.size(0)),):
        raise ValueError("B26 targets are not aligned")
    base = base_logits.detach()
    task = task_loss(logits, targets)

    positive = targets.eq(1)
    rival_index = torch.as_tensor(RIVALS, device=logits.device, dtype=torch.long)
    if bool(positive.any().item()):
        base_margin = base[:, 1] - base.index_select(1, rival_index).amax(dim=1)
        new_margin = logits[:, 1] - logits.index_select(1, rival_index).amax(dim=1)
        retention = F.relu(base_margin[positive] - new_margin[positive]).mean()
    else:
        retention = logits.sum() * 0.0

    hard_negative = torch.zeros_like(targets, dtype=torch.bool)
    for rival in RIVALS:
        hard_negative |= targets.eq(int(rival))
    if bool(hard_negative.any().item()):
        rows = torch.arange(int(targets.numel()), device=targets.device)
        true_logits = logits[rows, targets]
        rejection = F.relu(
            logits[:, 1] - true_logits + float(hard_negative_margin)
        )
        rejection = rejection[hard_negative].mean()
    else:
        rejection = logits.sum() * 0.0

    t = float(temperature)
    distillation = F.kl_div(
        F.log_softmax(logits / t, dim=1),
        F.softmax(base / t, dim=1),
        reduction="batchmean",
    ).clamp_min(0.0) * (t * t)
    total = (
        task
        + float(retention_weight) * retention
        + float(hard_negative_weight) * rejection
        + float(distillation_weight) * distillation
    )
    return total, {
        "task": task.detach(),
        "retention": retention.detach(),
        "hard_negative": rejection.detach(),
        "distillation": distillation.detach(),
        "total": total.detach(),
        "class1_rows": positive.sum().detach(),
        "hard_negative_rows": hard_negative.sum().detach(),
    }

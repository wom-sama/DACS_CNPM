from __future__ import annotations

import hashlib
import hmac
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STATE_DICT_KEYS = ("state_dict", "model_state", "model")
_BLOCKED_FACTORY_KWARGS = {
    "checkpoint_path",
    "pretrained",
    "pretrained_cfg",
    "pretrained_cfg_overlay",
}


class PretrainedWeightVerificationError(RuntimeError):
    """Raised when a local pretrained artifact cannot be verified exactly."""


class PretrainedBackboneContractError(RuntimeError):
    """Raised when a backbone does not expose the expected ViT token contract."""


def _normalized_sha256(value: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters.")
    return digest


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a local file without loading the full artifact into memory."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Local pretrained checkpoint does not exist: {resolved}")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_provenance_value(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must be recorded for a pretrained source.")
    return normalized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _load_local_checkpoint_payload(
    path: Path,
    *,
    format_suffix: Optional[str] = None,
) -> Mapping[str, Any]:
    suffix = str(format_suffix or path.suffix).strip().lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on optional runtime.
            raise ImportError(
                "Loading a .safetensors pretrained artifact requires safetensors."
            ) from exc
        payload = load_file(str(path), device="cpu")
    else:
        # weights_only=True is deliberately fail-closed. A scientific weight
        # artifact must not require arbitrary pickle code during local loading.
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise PretrainedWeightVerificationError(
            f"Local checkpoint must contain a mapping, got {type(payload)!r}."
        )
    return payload


def _extract_tensor_state_dict(
    payload: Mapping[str, Any],
    *,
    checkpoint_key: Optional[str],
    state_dict_prefix: str,
) -> Dict[str, Tensor]:
    selected: Any = payload
    resolved_key = str(checkpoint_key or "").strip()
    if resolved_key:
        if resolved_key not in payload:
            raise PretrainedWeightVerificationError(
                f"Checkpoint does not contain requested key {resolved_key!r}."
            )
        selected = payload[resolved_key]
    elif not payload or not all(torch.is_tensor(value) for value in payload.values()):
        selected = None
        for candidate_key in _STATE_DICT_KEYS:
            candidate = payload.get(candidate_key)
            if isinstance(candidate, Mapping):
                selected = candidate
                break
        if selected is None:
            raise PretrainedWeightVerificationError(
                "Checkpoint has no unambiguous tensor state_dict/model_state/model mapping."
            )
    if not isinstance(selected, Mapping) or not selected:
        raise PretrainedWeightVerificationError("Selected pretrained state_dict is empty.")

    prefix = str(state_dict_prefix or "")
    state_dict: Dict[str, Tensor] = {}
    for raw_key, value in selected.items():
        key = str(raw_key)
        if prefix:
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
        if not key:
            raise PretrainedWeightVerificationError(
                "state_dict_prefix produced an empty parameter name."
            )
        if not torch.is_tensor(value):
            raise PretrainedWeightVerificationError(
                f"Pretrained state entry {raw_key!r} is not a tensor."
            )
        state_dict[key] = value
    if not state_dict:
        raise PretrainedWeightVerificationError(
            f"No tensors remain after state_dict_prefix={prefix!r}."
        )
    return state_dict


def load_verified_local_timm_model(
    *,
    model_name: str,
    checkpoint_path: Path,
    expected_sha256: str,
    source_repository: str,
    source_revision: str,
    license_id: str,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    checkpoint_key: Optional[str] = None,
    state_dict_prefix: str = "",
    strict: bool = True,
    model_factory: Optional[Callable[..., nn.Module]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build a timm architecture offline and load only a hash-verified file.

    The artifact is hashed before model construction, the factory is always
    called with ``pretrained=False``, and the file is hashed again after state
    loading. This keeps model creation independent of a network/cache lookup
    while also detecting a file that changes during loading.

    ``model_factory`` is an injection point for tests. Production callers
    should omit it so the installed ``timm.create_model`` is used.
    """

    resolved_name = str(model_name or "").strip()
    if not resolved_name:
        raise ValueError("model_name must not be empty.")
    logical_path = Path(checkpoint_path).expanduser()
    if not logical_path.is_absolute():
        logical_path = Path.cwd() / logical_path
    # Do not resolve the logical path before choosing the loader. Hugging Face
    # snapshot entries such as `model.safetensors` are symlinks to extensionless
    # blobs; using the blob suffix would incorrectly route them to torch.load.
    logical_path = logical_path.absolute()
    if not logical_path.is_file():
        raise FileNotFoundError(
            f"Local pretrained checkpoint does not exist: {logical_path}"
        )
    resolved_path = logical_path.resolve(strict=True)
    expected_digest = _normalized_sha256(expected_sha256)
    repository = _require_provenance_value("source_repository", source_repository)
    revision = _require_provenance_value("source_revision", source_revision)
    resolved_license = _require_provenance_value("license_id", license_id)

    before_stat = resolved_path.stat()
    observed_digest = sha256_file(resolved_path)
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise PretrainedWeightVerificationError(
            "Local pretrained checkpoint SHA-256 mismatch: "
            f"observed={observed_digest}, expected={expected_digest}, "
            f"path={resolved_path}"
        )

    factory_kwargs = dict(model_kwargs or {})
    blocked = sorted(_BLOCKED_FACTORY_KWARGS.intersection(factory_kwargs))
    if blocked:
        raise ValueError(
            "model_kwargs may not override offline pretrained construction: "
            + ", ".join(blocked)
        )
    if model_factory is None:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise ImportError("Verified timm model loading requires the timm package.") from exc
        factory: Callable[..., nn.Module] = timm.create_model
        factory_identity = "timm.create_model"
        timm_version = str(getattr(timm, "__version__", "unknown"))
    else:
        factory = model_factory
        factory_identity = (
            f"{getattr(model_factory, '__module__', 'injected')}."
            f"{getattr(model_factory, '__qualname__', type(model_factory).__name__)}"
        )
        timm_version = "injected-test-factory"

    model = factory(resolved_name, pretrained=False, **factory_kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Model factory returned {type(model)!r}, expected torch.nn.Module.")
    # Load the exact resolved target that was hashed. The logical snapshot
    # suffix still selects the safe deserializer for extensionless HF blobs.
    payload = _load_local_checkpoint_payload(
        resolved_path,
        format_suffix=logical_path.suffix,
    )
    state_dict = _extract_tensor_state_dict(
        payload,
        checkpoint_key=checkpoint_key,
        state_dict_prefix=state_dict_prefix,
    )
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))

    if logical_path.resolve(strict=True) != resolved_path:
        raise PretrainedWeightVerificationError(
            f"Local pretrained checkpoint symlink changed while loading: {logical_path}"
        )
    after_stat = resolved_path.stat()
    after_digest = sha256_file(resolved_path)
    if (
        int(after_stat.st_size) != int(before_stat.st_size)
        or int(after_stat.st_mtime_ns) != int(before_stat.st_mtime_ns)
        or not hmac.compare_digest(after_digest, expected_digest)
    ):
        raise PretrainedWeightVerificationError(
            f"Local pretrained checkpoint changed while loading: {resolved_path}"
        )

    missing_keys = list(getattr(incompatible, "missing_keys", ()))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", ()))
    provenance: Dict[str, Any] = {
        "schema_version": 1,
        "model_name": resolved_name,
        "source_repository": repository,
        "source_revision": revision,
        "license_id": resolved_license,
        "offline_verified_load": True,
        "network_access_required": False,
        "factory": {
            "identity": factory_identity,
            "pretrained_argument": False,
            "model_kwargs": _json_safe(factory_kwargs),
            "timm_version": timm_version,
        },
        "checkpoint": {
            "path": str(logical_path),
            "resolved_path": str(resolved_path),
            "filename": logical_path.name,
            "format": logical_path.suffix.lower().lstrip(".") or "unknown",
            "is_symlink": bool(logical_path.is_symlink()),
            "bytes": int(after_stat.st_size),
            "sha256": after_digest,
            "checkpoint_key": str(checkpoint_key or ""),
            "state_dict_prefix": str(state_dict_prefix or ""),
            "tensor_count": int(len(state_dict)),
            "tensor_values": int(sum(int(value.numel()) for value in state_dict.values())),
        },
        "load_state_dict": {
            "strict": bool(strict),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
    }
    # A plain JSON-safe dictionary makes the provenance travel with a model
    # into a later checkpoint without importing this helper at serialization.
    setattr(model, "pretrained_source_provenance", deepcopy(provenance))
    return model, provenance


class PretrainedSemanticResidualBranch(nn.Module):
    """Fuse verified ViT semantics into an existing feature using ReZero.

    The backbone must return ``[B, prefix + patches, D]`` from
    ``forward_features``. The branch keeps CLS and register evidence separate,
    then concatenates CLS, register mean, patch mean, and patch standard
    deviation. It projects that descriptor to ``output_dim`` and adds it to
    ``base_feature`` through ``max_scale*tanh(residual_gate)``.
    With the default ``initial_scale=0``, an integrated keeper produces the
    same feature and logits before any optimization step. A nonzero initial
    scale remains explicit, bounded, and exactly recoverable from provenance.
    """

    descriptor_components = (
        "cls_token",
        "register_mean",
        "patch_mean",
        "patch_std",
    )

    def __init__(
        self,
        backbone: nn.Module,
        *,
        output_dim: int,
        backbone_dim: Optional[int] = None,
        num_prefix_tokens: Optional[int] = None,
        expected_patch_count: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.05,
        initial_scale: float = 0.0,
        max_scale: float = 1.0,
        freeze_backbone: bool = False,
        source_provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module.")
        self.backbone = backbone
        inferred_backbone_dim = int(
            getattr(backbone, "num_features", getattr(backbone, "embed_dim", 0)) or 0
        )
        resolved_backbone_dim = int(
            inferred_backbone_dim if backbone_dim is None else backbone_dim
        )
        if resolved_backbone_dim <= 0:
            raise ValueError("backbone_dim must be positive or inferable from the backbone.")
        if inferred_backbone_dim > 0 and inferred_backbone_dim != resolved_backbone_dim:
            raise ValueError(
                "backbone_dim does not match backbone.num_features/embed_dim: "
                f"{resolved_backbone_dim} != {inferred_backbone_dim}."
            )
        inferred_prefix = int(getattr(backbone, "num_prefix_tokens", 0) or 0)
        resolved_prefix = int(
            inferred_prefix if num_prefix_tokens is None else num_prefix_tokens
        )
        if resolved_prefix <= 0:
            raise ValueError(
                "num_prefix_tokens must be positive or inferable from the backbone."
            )
        resolved_patch_count = (
            None if expected_patch_count is None else int(expected_patch_count)
        )
        if resolved_patch_count is not None and resolved_patch_count <= 0:
            raise ValueError("expected_patch_count must be positive when provided.")
        resolved_output_dim = int(output_dim)
        if resolved_output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        resolved_hidden_dim = int(
            hidden_dim
            if hidden_dim is not None
            else max(resolved_output_dim, resolved_backbone_dim)
        )
        if resolved_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        resolved_dropout = float(dropout)
        if not 0.0 <= resolved_dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.max_scale = float(max_scale)
        self.initial_scale = float(initial_scale)
        if self.max_scale <= 0.0:
            raise ValueError("max_scale must be positive.")
        if not -self.max_scale < self.initial_scale < self.max_scale:
            raise ValueError("initial_scale must lie strictly inside (-max_scale, max_scale).")

        self.backbone_dim = resolved_backbone_dim
        self.output_dim = resolved_output_dim
        self.num_prefix_tokens = resolved_prefix
        self.expected_patch_count = resolved_patch_count
        self.hidden_dim = resolved_hidden_dim
        descriptor_dim = len(self.descriptor_components) * self.backbone_dim
        self.descriptor_norm = nn.LayerNorm(descriptor_dim)
        self.projector = nn.Sequential(
            nn.Linear(descriptor_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(resolved_dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        initial_raw_gate = math.atanh(self.initial_scale / self.max_scale)
        self.residual_gate = nn.Parameter(
            torch.tensor(initial_raw_gate, dtype=torch.float32)
        )
        inherited_provenance = getattr(backbone, "pretrained_source_provenance", {})
        selected_provenance = (
            source_provenance
            if source_provenance is not None
            else inherited_provenance
        )
        if selected_provenance and not isinstance(selected_provenance, Mapping):
            raise TypeError("source_provenance must be a mapping when provided.")
        self._source_provenance: Dict[str, Any] = deepcopy(
            _json_safe(dict(selected_provenance or {}))
        )
        self._freeze_backbone = False
        self.set_backbone_trainable(not bool(freeze_backbone))

    @property
    def backbone_frozen(self) -> bool:
        return bool(self._freeze_backbone)

    def set_backbone_trainable(self, enabled: bool) -> None:
        trainable = bool(enabled)
        self.backbone.requires_grad_(trainable)
        self._freeze_backbone = not trainable
        if self._freeze_backbone:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "PretrainedSemanticResidualBranch":
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _tokens_from_backbone_output(output: Any) -> Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, Mapping):
            for key in ("tokens", "x", "last_hidden_state", "features"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    return value
        raise PretrainedBackboneContractError(
            "backbone.forward_features must return a token tensor or a mapping/tuple "
            "containing one."
        )

    def encode_tokens(self, images: Tensor) -> Tensor:
        forward_features = getattr(self.backbone, "forward_features", None)
        if not callable(forward_features):
            raise PretrainedBackboneContractError(
                "Pretrained semantic backbone must implement forward_features."
            )
        tokens = self._tokens_from_backbone_output(forward_features(images))
        if tokens.ndim != 3:
            raise PretrainedBackboneContractError(
                f"Expected ViT tokens [B,N,D], got shape {tuple(tokens.shape)}."
            )
        if int(tokens.size(-1)) != self.backbone_dim:
            raise PretrainedBackboneContractError(
                "Backbone token dimension changed: "
                f"{int(tokens.size(-1))} != {self.backbone_dim}."
            )
        if int(tokens.size(1)) <= self.num_prefix_tokens:
            raise PretrainedBackboneContractError(
                "Backbone output must contain at least one patch token after "
                f"{self.num_prefix_tokens} prefix tokens; got {int(tokens.size(1))}."
            )
        self._validate_patch_count(int(tokens.size(1)) - self.num_prefix_tokens)
        return tokens

    def _validate_patch_count(self, observed_patch_count: int) -> None:
        if (
            self.expected_patch_count is not None
            and int(observed_patch_count) != self.expected_patch_count
        ):
            raise PretrainedBackboneContractError(
                "Backbone patch-token count changed: "
                f"{int(observed_patch_count)} != {self.expected_patch_count}."
            )

    def split_tokens(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or int(tokens.size(-1)) != self.backbone_dim:
            raise PretrainedBackboneContractError(
                f"Expected tokens [B,N,{self.backbone_dim}], got {tuple(tokens.shape)}."
            )
        if int(tokens.size(1)) <= self.num_prefix_tokens:
            raise PretrainedBackboneContractError(
                "Token sequence does not contain a non-empty patch suffix."
            )
        self._validate_patch_count(int(tokens.size(1)) - self.num_prefix_tokens)
        return (
            tokens[:, : self.num_prefix_tokens],
            tokens[:, self.num_prefix_tokens :],
        )

    def semantic_residual_from_tokens(
        self,
        tokens: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        prefix_tokens, patch_tokens = self.split_tokens(tokens)
        prefix_float = prefix_tokens.float()
        patch_float = patch_tokens.float()
        cls_token = prefix_float[:, 0]
        register_mean = (
            prefix_float[:, 1:].mean(dim=1)
            if int(prefix_float.size(1)) > 1
            else torch.zeros_like(cls_token)
        )
        patch_mean = patch_float.mean(dim=1)
        patch_std = patch_float.var(dim=1, unbiased=False).clamp_min(0.0).sqrt()
        descriptor = torch.cat(
            (cls_token, register_mean, patch_mean, patch_std),
            dim=-1,
        )
        projector_dtype = next(self.projector.parameters()).dtype
        descriptor_for_projection = descriptor.to(dtype=projector_dtype)
        residual = self.projector(self.descriptor_norm(descriptor_for_projection))
        details = {
            "cls_token": cls_token,
            "register_mean": register_mean,
            "patch_mean": patch_mean,
            "patch_std": patch_std,
            "descriptor": descriptor,
            "residual": residual,
        }
        return residual, details

    def semantic_residual(self, images: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        tokens = self.encode_tokens(images)
        residual, details = self.semantic_residual_from_tokens(tokens)
        return residual, tokens, details

    def effective_gate(self) -> Tensor:
        return torch.tanh(self.residual_gate) * self.max_scale

    def forward_from_tokens(
        self,
        base_feature: Tensor,
        tokens: Tensor,
        *,
        return_trace: bool = False,
    ):
        if base_feature.ndim != 2 or int(base_feature.size(1)) != self.output_dim:
            raise ValueError(
                f"base_feature must have shape [B,{self.output_dim}], "
                f"got {tuple(base_feature.shape)}."
            )
        if int(tokens.size(0)) != int(base_feature.size(0)):
            raise ValueError("Token and base_feature batch sizes must match.")
        residual, details = self.semantic_residual_from_tokens(tokens)
        gate = self.effective_gate().to(
            device=base_feature.device,
            dtype=base_feature.dtype,
        )
        typed_residual = residual.to(device=base_feature.device, dtype=base_feature.dtype)
        fused = base_feature + gate * typed_residual
        if not return_trace:
            return fused
        prefix_count = int(self.num_prefix_tokens)
        patch_count = int(tokens.size(1)) - prefix_count
        base_norm = base_feature.detach().float().norm(dim=-1)
        residual_norm = typed_residual.detach().float().norm(dim=-1)
        trace: Dict[str, Any] = {
            "token_shape": [int(value) for value in tokens.shape],
            "prefix_token_count": prefix_count,
            "patch_token_count": patch_count,
            "descriptor_components": list(self.descriptor_components),
            "cls_token": details["cls_token"].detach(),
            "register_mean": details["register_mean"].detach(),
            "patch_mean": details["patch_mean"].detach(),
            "patch_std": details["patch_std"].detach(),
            "descriptor": details["descriptor"].detach(),
            "residual": typed_residual.detach(),
            "raw_gate": self.residual_gate.detach().clone(),
            "effective_gate": gate.detach().clone(),
            "configured_initial_scale": float(self.initial_scale),
            "configured_max_scale": float(self.max_scale),
            "base_norm": base_norm,
            "residual_norm": residual_norm,
            "gated_residual_norm_ratio": (
                residual_norm * gate.detach().float().abs()
                / base_norm.clamp_min(1e-12)
            ),
            "source_provenance": self.provenance(),
        }
        return fused, trace

    def forward(
        self,
        images: Tensor,
        base_feature: Tensor,
        *,
        return_trace: bool = False,
    ):
        tokens = self.encode_tokens(images)
        return self.forward_from_tokens(
            base_feature,
            tokens,
            return_trace=return_trace,
        )

    def provenance(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "component": type(self).__name__,
            "source": deepcopy(self._source_provenance),
            "token_contract": {
                "backbone_dim": int(self.backbone_dim),
                "num_prefix_tokens": int(self.num_prefix_tokens),
                "expected_patch_count": self.expected_patch_count,
                "descriptor_components": list(self.descriptor_components),
                "descriptor_dim": int(
                    len(self.descriptor_components) * self.backbone_dim
                ),
            },
            "residual_contract": {
                "output_dim": int(self.output_dim),
                "hidden_dim": int(self.hidden_dim),
                "gate_parameterization": "max_scale*tanh(raw_gate)",
                "raw_gate_initial_value": float(
                    math.atanh(self.initial_scale / self.max_scale)
                ),
                "initial_scale": float(self.initial_scale),
                "max_scale": float(self.max_scale),
                "keeper_identity_at_initialization": self.initial_scale == 0.0,
            },
            "backbone_frozen": bool(self.backbone_frozen),
            "parameter_count": int(sum(parameter.numel() for parameter in self.parameters())),
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
            ),
        }

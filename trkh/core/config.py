from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LEGACY_REMOVED_CLASS_NAME = "Xoai_Song_ChuaNhe_CoNguyCoDap"
CANONICAL_MANGO_CLASS_NAMES = (
    "Xoai_Song_RatChua_KhoDap",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_Chin_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
CLASS_NAME_MODES = ("auto", "raw", "mango")


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_yaml() -> Path:
    return project_dir().parent / "dataset" / "data.yaml"


@dataclass
class BalanceClassSpec:
    class_index: int
    count: int
    ratio: float
    percent: float


@dataclass
class BalanceSpec:
    balance_yaml: Path
    version_note: str
    total_pairs: int
    train_ratio_config: float
    val_ratio_config: float
    test_ratio_config: float
    classes: Dict[int, BalanceClassSpec]

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0 for _ in range(max(1, int(num_classes)))]
        for class_index, class_spec in self.classes.items():
            if 0 <= int(class_index) < len(counts):
                counts[int(class_index)] = int(class_spec.count)
        return counts

    def class_ratios(self, num_classes: int) -> List[float]:
        ratios = [0.0 for _ in range(max(1, int(num_classes)))]
        for class_index, class_spec in self.classes.items():
            if 0 <= int(class_index) < len(ratios):
                ratios[int(class_index)] = float(class_spec.ratio)
        return ratios

    def auto_repeat_factors(self, num_classes: int) -> List[float]:
        counts = self.class_counts(num_classes)
        positive_counts = [count for count in counts if count > 0]
        if not positive_counts:
            return [1.0 for _ in counts]
        max_count = max(positive_counts)
        factors = []
        for count in counts:
            if count <= 0:
                factors.append(1.0)
                continue
            factors.append(float(max(1.0, math.sqrt(float(max_count) / float(count)))))
        return factors


@dataclass
class DataSpec:
    data_yaml: Path
    root: Path
    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path
    class_names: List[str]
    class_name_mode: str = "auto"
    data_format: str = "yolo"
    balance: Optional[BalanceSpec] = None
    test_images: Optional[Path] = None
    test_labels: Optional[Path] = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def has_test_split(self) -> bool:
        return self.test_images is not None and self.test_labels is not None

    def split_images_dir(self, split: str) -> Path:
        split_name = str(split).strip().lower()
        if split_name == "train":
            return self.train_images
        if split_name == "val":
            return self.val_images
        if split_name == "test" and self.test_images is not None:
            return self.test_images
        raise ValueError(f"Split khong ho tro hoac chua duoc khai bao: {split}")

    def split_labels_dir(self, split: str) -> Path:
        if self.data_format == "classification_folder":
            return self.split_images_dir(split)
        split_name = str(split).strip().lower()
        if split_name == "train":
            return self.train_labels
        if split_name == "val":
            return self.val_labels
        if split_name == "test" and self.test_labels is not None:
            return self.test_labels
        raise ValueError(f"Split khong ho tro hoac chua duoc khai bao: {split}")


@dataclass
class ModelConfig:
    model_type: str = "vit_registers_hybrid"
    pretrained: bool = False
    head_pooling: str = "cls_register_mean"
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    use_cnn_stem: bool = True
    stem_channels: int = 32
    cnn_feature_fusion: bool = False
    cnn_fusion_dropout: float = 0.1
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_registers: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    register_positional_embedding: bool = False
    gradient_checkpointing: bool = False
    temporal_frames: int = 1
    temporal_num_heads: int = 4
    temporal_dropout: float = 0.1
    temporal_kv_quant_bits: int = 8
    bbox_head_hidden_dim: int = 512
    num_queries: int = 40
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1
    decoder_memory_adapter: bool = False
    decoder_memory_adapter_dropout: float = 0.0
    learned_query_content: bool = True
    separate_objectness: bool = True
    objectness_prior_prob: float = 0.125
    quality_head: bool = False
    quality_prior_prob: float = 0.125
    auxiliary_decoder_outputs: bool = False
    query_denoising_noise: float = 0.0
    count_head: bool = False
    count_head_hidden_dim: int = 256
    count_head_dropout: float = 0.05
    count_head_prior: float = 1.2


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 8
    epochs: int = 80
    learning_rate: float = 5e-4
    backbone_lr_scale: float = 1.0
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 8
    warmup_start_factor: float = 0.1
    lr_scheduler: str = "cosine"
    scheduler_total_epochs: int = 0
    label_smoothing: float = 0.02
    grad_clip_norm: float = 1.0
    max_nonfinite_grad_steps: int = 8
    num_workers: int = 4
    eval_num_workers: int = -1
    train_image_cache_mb: int = -1
    eval_image_cache_mb: int = -1
    early_stopping_patience: int = 15
    seed: int = 42
    amp: bool = True
    deterministic: bool = False
    use_class_weights: bool = True
    class_weight_mode: str = "sqrt_inverse"
    class_weight_beta: float = 0.999
    use_weighted_sampler: bool = False
    weighted_sampler_power: float = 1.75
    weighted_sampler_epoch_multiplier: float = 2.0
    auto_tune_imbalance: bool = False
    imbalance_sampler_disable_threshold: float = 0.18
    save_last_checkpoint: bool = True
    max_train_batches: int = 0
    max_val_batches: int = 0
    focal_loss_gamma: float = 2.0
    focal_loss_mix: float = 0.35
    use_ldam: bool = True
    ldam_max_margin: float = 0.5
    ldam_scale: float = 30.0
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = False
    batch_mix_probability: float = 0.5
    mosaic_probability: float = 0.25
    mosaic_min_split: float = 0.35
    mosaic_max_split: float = 0.65
    mixup_probability: float = 0.0
    mixup_alpha: float = 0.4
    cutmix_probability: float = 0.5
    cutmix_alpha: float = 1.0
    copy_paste_probability: float = 0.0
    copy_paste_max_objects: int = 2
    targeted_copy_paste_scale_threshold: float = 1.5
    targeted_copy_paste_probability: float = 1.0
    eval_tta: bool = False
    tta_brightness_delta: float = 0.08
    log_artifact_stats: bool = True
    optimized_scheduler: bool = False
    stage1_epochs_optimized: int = 5
    multi_scale_training: bool = False
    multi_scale_epochs: int = 10
    stage1_epochs: int = 30
    stage1_auto_advance_macro_f1_threshold: float = 0.0
    stage1_auto_advance_min_epochs: int = 1
    stage1_bbox_l1_loss_weight: float = 0.0
    stage1_bbox_giou_loss_weight: float = 0.0
    best_metric: str = "composite"
    cls_loss_weight: float = 1.0
    classification_overfit_guard: bool = True
    classification_guard_macro_f1_threshold: float = 0.985
    classification_guard_detection_gap: float = 0.12
    classification_guard_min_cls_weight: float = 0.25
    adaptive_detection_loss: bool = True
    adaptive_detection_macro_f1_threshold: float = 0.93
    adaptive_detection_f1_target: float = 0.90
    adaptive_detection_gap_threshold: float = 0.20
    adaptive_detection_bbox_iou_target: float = 0.70
    adaptive_detection_max_multiplier: float = 2.0
    rare_class_recall_guard: bool = True
    rare_class_recall_target: float = 0.70
    rare_class_recall_guard_scale_threshold: float = 1.5
    rare_class_recall_guard_max_multiplier: float = 2.0
    rare_class_recall_guard_min_precision: float = 0.35
    hard_sample_manifest: str = ""
    hard_sample_repeat_factor: float = 1.0
    bbox_l1_loss_weight: float = 1.0
    bbox_giou_loss_weight: float = 0.5
    background_loss_weight: float = 0.3
    objectness_loss_weight: float = 5.0
    objectness_focal_alpha: float = 0.75
    objectness_focal_gamma: float = 0.5
    matcher_class_cost: float = 1.0
    matcher_objectness_cost: float = 1.0
    cardinality_loss_weight: float = 0.0
    count_loss_weight: float = 0.0
    quality_loss_weight: float = 0.0
    auxiliary_loss_weight: float = 0.0
    count_objectness_consistency_weight: float = 0.0
    eval_detection_nms_iou_threshold: float = 0.5
    eval_max_detections_per_image: int = 0
    eval_detection_score_mode: str = "foreground"
    eval_require_foreground_argmax: bool = False
    eval_adaptive_max_detections: bool = False
    eval_adaptive_count_source: str = "auto"
    eval_adaptive_count_margin: int = 1
    eval_adaptive_min_detections: int = 1


@dataclass
class AugmentationConfig:
    crop_margin_ratio: float = 0.05
    class_crop_margin_scale_threshold: float = 1.5
    class_crop_margin_max_ratio: float = 0.16
    resize_mode: str = "pad"
    random_resized_crop_scale_min: float = 0.8
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.0
    color_jitter_hue: float = 0.0
    random_erasing_probability: float = 0.2
    random_affine_degrees: float = 8.0
    random_affine_translate: float = 0.05
    random_affine_scale_min: float = 0.9
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.1
    rotate90_probability: float = 0.15
    lighting_probability: float = 0.15
    class_aware_augmentation: bool = True
    class_augmentation_power: float = 0.75
    class_augmentation_max_scale: float = 1.8
    class_aware_photometric_augmentation: bool = False
    class_aware_mix_probability_boost: float = 0.5
    class_aware_mix_source_power: float = 1.0
    rare_class_repeat: bool = True
    rare_class_repeat_power: float = 0.5
    rare_class_repeat_max_factor: float = 3.0
    rare_class_repeat_min_ratio: float = 0.35
    randaugment_num_ops: int = 0
    randaugment_magnitude: int = 0


def _normalize_names(names: Union[Sequence[str], Dict[Any, str]]) -> List[str]:
    if isinstance(names, dict):
        normalized = [value for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    else:
        normalized = list(names)
    return [str(name).strip() for name in normalized]


def _normalize_class_name_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "auto"
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "as_is": "raw",
        "asis": "raw",
        "preserve": "raw",
        "canonical_mango": "mango",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CLASS_NAME_MODES:
        allowed = ", ".join(CLASS_NAME_MODES)
        raise ValueError(f"class_name_mode khong hop le: {mode}. Gia tri hop le: {allowed}")
    return normalized


def _resolve_project_class_names(class_names: Sequence[str], mode: Optional[str] = "auto") -> List[str]:
    mode = _normalize_class_name_mode(mode)
    normalized = [str(name).strip() for name in class_names]
    canonical_set = set(CANONICAL_MANGO_CLASS_NAMES)

    if mode == "raw":
        return normalized

    has_legacy_mango_name = LEGACY_REMOVED_CLASS_NAME in normalized
    has_canonical_mango_name = bool(canonical_set.intersection(normalized))
    if has_legacy_mango_name and (mode == "mango" or has_canonical_mango_name):
        raise ValueError(
            "data.yaml van con lop legacy "
            f"'{LEGACY_REMOVED_CLASS_NAME}'. Hay cap nhat dataset ve bo 4 lop moi."
        )

    if mode == "mango":
        if len(normalized) != len(CANONICAL_MANGO_CLASS_NAMES) or set(normalized) != canonical_set:
            raise ValueError(
                "class_name_mode=mango yeu cau dung bo 4 lop mango canonical. "
                "Dung class_name_mode=raw cho dataset tuy bien hoac nhieu hon 4 lop."
            )
        return list(CANONICAL_MANGO_CLASS_NAMES)

    if len(normalized) == len(CANONICAL_MANGO_CLASS_NAMES) and set(normalized) == canonical_set:
        return list(CANONICAL_MANGO_CLASS_NAMES)
    return normalized


def _resolve_path(base: Path, value: Union[str, Path]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _images_to_labels(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Khong the suy ra labels dir tu duong dan: {images_dir}")
    image_index = parts.index("images")
    new_parts = parts[:]
    new_parts[image_index] = "labels"
    return Path(*new_parts)


def _normalize_data_format(value: object) -> str:
    normalized = str(value or "yolo").strip().lower().replace("-", "_")
    aliases = {
        "image_folder": "classification_folder",
        "imagefolder": "classification_folder",
        "classification": "classification_folder",
        "classification_dir": "classification_folder",
        "classification_directory": "classification_folder",
        "class_folder": "classification_folder",
        "classfolder": "classification_folder",
        "cls_folder": "classification_folder",
        "cls_crops": "classification_folder",
        "yolo_detection": "yolo",
        "yolo_labels": "yolo",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"yolo", "classification_folder"}:
        raise ValueError(
            "format/data_format khong hop le trong data.yaml: "
            f"{value!r}. Gia tri ho tro: yolo, classification_folder."
        )
    return normalized


def _infer_class_names_from_split_dir(split_dir: Path) -> List[str]:
    if not split_dir.exists():
        return []
    return [
        path.name
        for path in sorted(split_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir()
    ]


def _optional_balance_yaml_path(data_yaml_path: Path, raw: Dict[str, Any]) -> Optional[Path]:
    explicit_value = raw.get("balance_yaml", raw.get("balance_file", raw.get("canbang_yaml")))
    if explicit_value is False:
        return None
    if explicit_value is not None:
        return _resolve_path(data_yaml_path.parent, explicit_value)

    candidate = data_yaml_path.parent / "canbang.yaml"
    if candidate.exists():
        return candidate.resolve()
    return None


def _load_balance_spec(
    data_yaml_path: Path,
    raw: Dict[str, Any],
    num_classes: int,
) -> Optional[BalanceSpec]:
    balance_yaml_path = _optional_balance_yaml_path(data_yaml_path, raw)
    if balance_yaml_path is None:
        return None
    if not balance_yaml_path.exists():
        raise FileNotFoundError(f"Khong tim thay file can bang du lieu: {balance_yaml_path}")

    payload = yaml.safe_load(balance_yaml_path.read_text(encoding="utf-8")) or {}
    balance_payload = payload.get("dataset_balance", payload)
    if not isinstance(balance_payload, dict):
        raise ValueError(f"File can bang khong hop le: {balance_yaml_path}")

    raw_classes = balance_payload.get("classes", {})
    if not isinstance(raw_classes, dict):
        raise ValueError(f"dataset_balance.classes phai la mapping trong {balance_yaml_path}")

    classes: Dict[int, BalanceClassSpec] = {}
    total_pairs = int(balance_payload.get("total_pairs", 0) or 0)
    for raw_class_index, raw_class_payload in raw_classes.items():
        class_index = int(raw_class_index)
        if class_index < 0 or class_index >= int(num_classes):
            raise ValueError(
                "Class ID trong canbang.yaml nam ngoai khoang data.yaml: "
                f"class_id={class_index}, num_classes={num_classes}, file={balance_yaml_path}"
            )
        if not isinstance(raw_class_payload, dict):
            raise ValueError(f"Thong tin class {class_index} trong canbang.yaml khong hop le.")
        count = int(raw_class_payload.get("count", 0) or 0)
        total_pairs = max(total_pairs, 0)
        ratio = float(
            raw_class_payload.get(
                "ratio",
                float(count) / float(total_pairs) if total_pairs > 0 else 0.0,
            )
            or 0.0
        )
        percent = float(raw_class_payload.get("percent", ratio * 100.0) or 0.0)
        classes[class_index] = BalanceClassSpec(
            class_index=class_index,
            count=count,
            ratio=ratio,
            percent=percent,
        )

    if total_pairs <= 0:
        total_pairs = sum(class_spec.count for class_spec in classes.values())
    if total_pairs > 0:
        normalized_classes: Dict[int, BalanceClassSpec] = {}
        for class_index, class_spec in classes.items():
            ratio = class_spec.ratio if class_spec.ratio > 0.0 else float(class_spec.count) / float(total_pairs)
            normalized_classes[class_index] = BalanceClassSpec(
                class_index=class_index,
                count=class_spec.count,
                ratio=ratio,
                percent=class_spec.percent if class_spec.percent > 0.0 else ratio * 100.0,
            )
        classes = normalized_classes

    return BalanceSpec(
        balance_yaml=balance_yaml_path,
        version_note=str(balance_payload.get("version_note", "")),
        total_pairs=int(total_pairs),
        train_ratio_config=float(balance_payload.get("train_ratio_config", 0.0) or 0.0),
        val_ratio_config=float(balance_payload.get("val_ratio_config", 0.0) or 0.0),
        test_ratio_config=float(balance_payload.get("test_ratio_config", 0.0) or 0.0),
        classes=classes,
    )


def load_data_spec(
    data_yaml: Union[str, Path],
    class_name_mode: Optional[str] = None,
    expected_num_classes: Optional[int] = None,
) -> DataSpec:
    data_yaml_path = Path(data_yaml).resolve()
    raw = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    data_format = _normalize_data_format(raw.get("format", raw.get("data_format", "yolo")))
    root = _resolve_path(data_yaml_path.parent, raw.get("path", data_yaml_path.parent))
    train_images = _resolve_path(root, raw.get("train", "train"))
    val_images = _resolve_path(root, raw.get("val", "val"))
    test_value = raw.get("test")
    test_images = _resolve_path(root, test_value) if test_value else None
    yaml_class_name_mode = raw.get("class_name_mode", raw.get("class_names_mode", "auto"))
    resolved_class_name_mode = _normalize_class_name_mode(
        class_name_mode if class_name_mode is not None else yaml_class_name_mode
    )
    raw_names = raw.get("names")
    if raw_names is None:
        if data_format != "classification_folder":
            raise KeyError("data.yaml thieu truong names.")
        raw_names = _infer_class_names_from_split_dir(train_images)
        if not raw_names:
            raise ValueError(
                "Khong the tu suy ra names cho classification_folder vi train dir khong co thu muc class: "
                f"{train_images}"
            )
    class_names = _resolve_project_class_names(
        _normalize_names(raw_names),
        mode=resolved_class_name_mode,
    )
    nc = int(raw.get("nc", len(class_names)))
    if nc != len(class_names):
        raise ValueError(
            f"So class khong khop trong data.yaml: nc={nc}, len(names)={len(class_names)}"
        )
    if expected_num_classes is not None and int(expected_num_classes) != len(class_names):
        raise ValueError(
            "So class khong khop voi cau hinh yeu cau: "
            f"expected_num_classes={int(expected_num_classes)}, len(names)={len(class_names)}"
        )
    balance_spec = _load_balance_spec(data_yaml_path, raw, num_classes=len(class_names))

    return DataSpec(
        data_yaml=data_yaml_path,
        root=root,
        train_images=train_images,
        train_labels=train_images if data_format == "classification_folder" else _images_to_labels(train_images),
        val_images=val_images,
        val_labels=val_images if data_format == "classification_folder" else _images_to_labels(val_images),
        class_names=class_names,
        class_name_mode=resolved_class_name_mode,
        data_format=data_format,
        balance=balance_spec,
        test_images=test_images,
        test_labels=(
            test_images
            if data_format == "classification_folder" and test_images is not None
            else _images_to_labels(test_images) if test_images is not None else None
        ),
    )


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value

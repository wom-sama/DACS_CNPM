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
    count_scope: str = "all"

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
    timm_model_name: str = "mobilenetv3_large_100.ra_in1k"
    input_mean: Sequence[float] = IMAGENET_MEAN
    input_std: Sequence[float] = IMAGENET_STD
    head_pooling: str = "cls_register_mean"
    classification_mlp_head: bool = False
    classification_mlp_hidden_dim: int = 512
    classification_mlp_dropout: float = 0.08
    classification_mlp_residual_scale: float = 0.20
    subcenter_proxy_head: bool = False
    subcenter_proxy_subcenters: int = 3
    subcenter_proxy_dropout: float = 0.0
    subcenter_proxy_init_std: float = 0.01
    deep_abstention_head: bool = False
    deep_abstention_dropout: float = 0.05
    deep_abstention_initial_probability: float = 0.01
    teacher_feature_projection_adapter: bool = False
    teacher_feature_projection_teacher_dim: int = 0
    teacher_feature_projection_dim: int = 128
    teacher_feature_projection_dropout: float = 0.0
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    use_cnn_stem: bool = True
    stem_channels: int = 32
    stem_architecture: str = "conv_pool"
    stem_pooling_mode: str = "max"
    stem_softpool_blend: float = 0.15
    shifted_patch_tokenization: bool = False
    shifted_patch_shift: int = 1
    shifted_patch_residual_scale: float = 0.10
    dual_patch_norm: bool = False
    cnn_feature_fusion: bool = False
    cnn_fusion_dropout: float = 0.1
    color_stat_fusion: bool = False
    color_stat_fusion_dropout: float = 0.1
    defect_stat_fusion: bool = False
    defect_stat_fusion_dropout: float = 0.1
    foreground_surface_fusion: bool = False
    foreground_surface_fusion_dropout: float = 0.1
    foreground_surface_pairwise_head: bool = False
    foreground_surface_pairwise_pairs: str = "0-1,1-2,1-4,2-3"
    foreground_surface_pairwise_logit_scale: float = 0.18
    foreground_surface_pairwise_dropout: float = 0.05
    foreground_surface_pairwise_routing: bool = True
    foreground_surface_pairwise_route_max_probability_margin: float = 0.22
    interior_boundary_pairwise_head: bool = False
    interior_boundary_pairwise_pairs: str = "0-1,4-1"
    interior_boundary_pairwise_logit_scale: float = 0.14
    interior_boundary_pairwise_dropout: float = 0.05
    interior_boundary_pairwise_hidden_dim: int = 128
    interior_boundary_pairwise_erode_kernel: int = 9
    interior_boundary_pairwise_routing: bool = True
    interior_boundary_pairwise_route_max_probability_margin: float = 0.22
    bbox_spatial_fusion: bool = False
    bbox_spatial_fusion_hidden_dim: int = 64
    bbox_spatial_fusion_dropout: float = 0.05
    bbox_spatial_fusion_logit_scale: float = 0.20
    patch_objectness_guided_head: bool = False
    patch_objectness_hidden_dim: int = 128
    patch_objectness_dropout: float = 0.05
    patch_objectness_logit_scale: float = 0.12
    patch_objectness_temperature: float = 0.75
    bbox_prior_patch_context_head: bool = False
    bbox_prior_patch_context_hidden_dim: int = 128
    bbox_prior_patch_context_dropout: float = 0.05
    bbox_prior_patch_context_logit_scale: float = 0.12
    bbox_prior_patch_context_temperature: float = 0.50
    source_context_feature_fusion: bool = False
    source_context_fusion_hidden_dim: int = 128
    source_context_fusion_dropout: float = 0.05
    source_context_fusion_logit_scale: float = 0.20
    source_context_fusion_gate_bias: float = -2.0
    paired_view_feature_fusion: bool = False
    paired_view_fusion_hidden_dim: int = 128
    paired_view_fusion_dropout: float = 0.05
    paired_view_fusion_logit_scale: float = 0.12
    paired_view_fusion_gate_bias: float = -2.0
    bilinear_patch_fusion: bool = False
    bilinear_patch_rank: int = 32
    bilinear_patch_dropout: float = 0.1
    complementary_patch_suppression_head: bool = False
    complementary_patch_suppression_top_k: int = 6
    complementary_patch_suppression_hidden_dim: int = 128
    complementary_patch_suppression_dropout: float = 0.05
    complementary_patch_suppression_temperature: float = 0.20
    complementary_patch_suppression_strength: float = 0.85
    complementary_patch_suppression_bbox_weight: float = 0.35
    complementary_patch_suppression_logit_scale: float = 0.12
    micro_detail_patch_expert: bool = False
    micro_detail_top_k: int = 8
    micro_detail_hidden_dim: int = 128
    micro_detail_dropout: float = 0.08
    micro_detail_temperature: float = 0.12
    micro_detail_foreground_power: float = 1.0
    micro_detail_logit_scale: float = 0.18
    micro_detail_routing: bool = True
    micro_detail_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest"
    micro_detail_route_max_probability_margin: float = 0.25
    part_token_learner: bool = False
    part_token_count: int = 4
    part_token_hidden_dim: int = 128
    part_token_dropout: float = 0.08
    part_token_temperature: float = 0.70
    part_token_foreground_power: float = 1.0
    part_token_bbox_weight: float = 0.75
    part_token_logit_scale: float = 0.16
    part_token_routing: bool = True
    part_token_route_pairs: str = "0-1,1-2,4-1,2-3"
    part_token_route_max_probability_margin: float = 0.25
    part_token_pairwise_head: bool = False
    part_token_pairwise_pairs: str = "0-1,1-2,4-1,2-3"
    part_token_pairwise_logit_scale: float = 0.18
    part_token_pairwise_dropout: float = 0.08
    part_token_pairwise_routing: bool = True
    part_token_pairwise_route_max_probability_margin: float = 0.22
    patch_evidence_router_head: bool = False
    patch_evidence_router_pair: str = "0-1"
    patch_evidence_router_hidden_dim: int = 128
    patch_evidence_router_top_k: int = 4
    patch_evidence_router_bbox_weight: float = 0.75
    patch_evidence_router_dropout: float = 0.05
    patch_evidence_router_logit_scale: float = 0.12
    patch_evidence_router_margin_prior_mode: str = "none"
    patch_evidence_router_margin_prior_scale: float = 0.0
    patch_evidence_router_summary_stats: bool = False
    patch_evidence_router_routing: bool = True
    patch_evidence_router_route_max_probability_margin: float = 0.25
    patch_evidence_router_route_min_pair_probability: float = 0.02
    local_zoom_image_expert: bool = False
    local_zoom_crop_size: int = 128
    local_zoom_crop_scale: float = 0.48
    local_zoom_score_mode: str = "foreground_detail"
    local_zoom_hidden_dim: int = 128
    local_zoom_dropout: float = 0.08
    local_zoom_logit_scale: float = 0.16
    local_zoom_routing: bool = True
    local_zoom_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest"
    local_zoom_route_max_probability_margin: float = 0.25
    high_frequency_texture_expert: bool = False
    high_frequency_texture_hidden_dim: int = 128
    high_frequency_texture_dropout: float = 0.08
    high_frequency_texture_analysis_size: int = 96
    high_frequency_texture_logit_scale: float = 0.16
    high_frequency_texture_routing: bool = True
    high_frequency_texture_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest"
    high_frequency_texture_route_max_probability_margin: float = 0.25
    multi_granularity_aux_heads: bool = False
    multi_granularity_aux_layers: str = "2,5,8"
    multi_granularity_aux_dropout: float = 0.08
    self_boosting_attention_head: bool = False
    block_local_patch_mixer: bool = False
    block_local_patch_mixer_layers: str = "6,7,8"
    block_local_patch_mixer_dropout: float = 0.0
    block_local_patch_mixer_scale: float = 0.10
    locally_enhanced_ffn: bool = False
    locally_enhanced_ffn_layers: str = "1,2,3,4"
    locally_enhanced_ffn_kernel_size: int = 3
    concurrent_local_global_coupling: bool = False
    concurrent_local_global_layers: str = "1,2,3,4,5,6,7,8"
    concurrent_local_global_dim: int = 64
    concurrent_local_global_kernel_size: int = 3
    gated_relative_position_attention: bool = False
    gated_relative_position_attention_layers: str = "1,2,3,4"
    gated_relative_position_attention_max_mix: float = 0.25
    gated_relative_position_attention_locality_strength: float = 1.0
    layer_token_fusion: bool = False
    layer_token_fusion_layers: str = "2,4,6"
    layer_token_fusion_top_k: int = 4
    layer_token_fusion_blend: float = 0.12
    layer_token_fusion_attention_temperature: float = 0.20
    layer_token_fusion_bbox_weight: float = 0.20
    layer_token_fusion_foreground_weight: float = 0.10
    masked_reconstruction_head: bool = False
    masked_reconstruction_hidden_dim: int = 192
    frequency_selective_pooling: bool = False
    frequency_selective_top_k: int = 1
    frequency_selective_blend: float = 1.0
    frequency_selective_foreground_threshold: float = 0.35
    patch_memory_adapter: bool = False
    patch_memory_adapter_dropout: float = 0.0
    late_class_attention_pooling: bool = False
    late_class_attention_heads: int = 4
    late_class_attention_dropout: float = 0.05
    late_class_attention_mlp_ratio: float = 2.0
    late_class_attention_residual_scale: float = 0.10
    mixstyle: bool = False
    mixstyle_probability: float = 0.5
    mixstyle_alpha: float = 0.1
    fine_grained_pooling: bool = False
    fine_grained_pooling_dropout: float = 0.1
    multi_branch_fusion: bool = False
    branch_color_tokens: int = 1
    branch_edge_tokens: int = 1
    branch_cnn_tokens: int = 1
    branch_token_dropout: float = 0.1
    detail_patch_enhancement: bool = False
    detail_patch_dropout: float = 0.05
    token_pruning: bool = False
    token_prune_layers: str = "2,5"
    token_keep_rates: str = "0.75,0.50"
    token_prune_foreground_weight: float = 0.35
    token_prune_bbox_weight: float = 0.0
    token_prune_bbox_margin_ratio: float = 0.04
    early_token_mask_keep_rate: float = 1.0
    pairwise_margin_head: bool = False
    pairwise_margin_pairs: str = "0-1,2-3,4-rest"
    pairwise_margin_logit_scale: float = 0.35
    pairwise_margin_dropout: float = 0.05
    pairwise_margin_routing: bool = False
    pairwise_margin_route_max_probability_margin: float = 0.20
    topk_reassessment_head: bool = False
    topk_reassessment_top_k: int = 2
    topk_reassessment_hidden_dim: int = 128
    topk_reassessment_dropout: float = 0.05
    topk_reassessment_logit_scale: float = 0.15
    topk_reassessment_routing: bool = True
    topk_reassessment_route_pairs: str = "0-1,1-2,2-3,4-rest"
    topk_reassessment_route_max_probability_margin: float = 0.30
    focus_class_head: bool = False
    focus_class_index: int = 1
    focus_class_logit_scale: float = 0.20
    focus_class_dropout: float = 0.05
    focus_class_routing: bool = True
    focus_class_route_max_probability_margin: float = 0.35
    focus_class_route_min_probability: float = 0.08
    class_independent_head: bool = False
    class_independent_dropout: float = 0.05
    ordinal_maturity_head: bool = False
    ordinal_maturity_classes: str = "0,1,2,3"
    ordinal_maturity_logit_scale: float = 0.20
    ordinal_maturity_dropout: float = 0.05
    cumulative_ordinal_head: bool = False
    cumulative_ordinal_classes: str = "0,1,2,3"
    cumulative_ordinal_logit_scale: float = 0.25
    cumulative_ordinal_dropout: float = 0.05
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
    trainable_module_prefixes: str = ""
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
    balanced_epoch_sampling: bool = False
    balanced_epoch_multiplier: float = 1.0
    balanced_epoch_tolerance: float = 0.10
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
    classification_loss: str = "ldam_focal"
    balanced_softmax_tau: float = 1.0
    gce_q: float = 0.7
    ldr_margin: float = 2.0
    ldr_temperature: float = 1.0
    logit_norm_temperature: float = 0.04
    symmetric_ce_alpha: float = 0.1
    symmetric_ce_beta: float = 1.0
    symmetric_ce_epsilon: float = 1e-4
    seesaw_mitigation_power: float = 0.8
    seesaw_compensation_power: float = 2.0
    class_loss_multipliers: str = ""
    metric_learning_loss_weight: float = 0.0
    metric_learning_temperature: float = 0.12
    metric_learning_class_balanced: bool = True
    metric_learning_sources: str = "head"
    teacher_guided_contrastive_loss_weight: float = 0.0
    teacher_guided_contrastive_temperature: float = 0.16
    teacher_guided_contrastive_sources: str = "head,patch"
    teacher_guided_contrastive_classes: str = "0,1,2,4"
    teacher_guided_contrastive_teacher_min_confidence: float = 0.70
    teacher_guided_contrastive_require_agreement: bool = True
    teacher_guided_contrastive_class_balanced: bool = True
    teacher_guided_contrastive_weight_mode: str = "filter"
    teacher_guided_contrastive_stochastic_std: float = 0.0
    teacher_guided_contrastive_min_reliability: float = 0.0
    teacher_guided_contrastive_teacher_confidence_power: float = 1.0
    teacher_guided_contrastive_memory_queue_size: int = 0
    teacher_guided_contrastive_memory_min_count: int = 2
    boundary_contrastive_loss_weight: float = 0.0
    boundary_contrastive_pairs: str = "0-1,1-2,2-3,4-rest"
    boundary_contrastive_sources: str = "head,patch"
    boundary_contrastive_margin: float = 0.12
    boundary_contrastive_temperature: float = 0.20
    boundary_contrastive_max_pairs: int = 128
    boundary_center_loss_weight: float = 0.0
    boundary_center_pairs: str = "0-1,1-2,2-3,1-4"
    boundary_center_sources: str = "head,patch"
    boundary_center_margin: float = 0.10
    boundary_center_temperature: float = 0.20
    boundary_center_compactness_weight: float = 0.10
    boundary_center_teacher_min_confidence: float = 0.0
    boundary_center_require_agreement: bool = False
    boundary_center_teacher_weight_mode: str = "filter"
    foreground_consistency_loss_weight: float = 0.0
    foreground_consistency_margin: float = 0.08
    border_attention_suppression_loss_weight: float = 0.0
    border_attention_suppression_frame_width: int = 1
    border_attention_suppression_bbox_band: float = 0.12
    border_attention_suppression_bbox_weight: float = 0.5
    border_attention_suppression_temperature: float = 0.20
    border_attention_suppression_classes: str = "0,1,2,4"
    border_attention_suppression_start_epoch: int = 1
    foreground_surface_aux_loss_weight: float = 0.0
    foreground_surface_pairwise_loss_weight: float = 0.0
    interior_boundary_pairwise_loss_weight: float = 0.0
    micro_detail_aux_loss_weight: float = 0.0
    part_token_aux_loss_weight: float = 0.0
    part_token_pairwise_loss_weight: float = 0.0
    patch_objectness_loss_weight: float = 0.0
    patch_objectness_positive_weight: float = 1.0
    bbox_token_label_loss_weight: float = 0.0
    bbox_token_label_min_prior: float = 0.45
    bbox_token_label_prior_power: float = 1.0
    bbox_token_label_classes: str = "0,1,2,4"
    bbox_token_label_focus_class: int = 1
    bbox_token_label_focus_weight: float = 1.25
    bbox_token_label_detach_classifier: bool = True
    bbox_token_label_start_epoch: int = 1
    bbox_token_prior_source: str = "bbox"
    local_zoom_aux_loss_weight: float = 0.0
    high_frequency_texture_aux_loss_weight: float = 0.0
    high_frequency_texture_pairwise_loss_weight: float = 0.0
    high_frequency_texture_pairwise_pairs: str = "0-1,1-2,2-3,1-4"
    multi_granularity_aux_loss_weight: float = 0.0
    multi_granularity_refinement_loss_weight: float = 0.0
    multi_granularity_refinement_temperature: float = 32.0
    multi_granularity_contrastive_loss_weight: float = 0.0
    multi_granularity_contrastive_temperature: float = 0.18
    multi_granularity_contrastive_pairs: str = "0-1,1-2,1-4,2-3"
    multi_granularity_contrastive_teacher_min_confidence: float = 0.0
    multi_granularity_contrastive_require_agreement: bool = True
    multi_granularity_contrastive_weight_mode: str = "filter"
    multi_granularity_contrastive_teacher_confidence_power: float = 1.0
    self_boosting_attention_loss_weight: float = 0.0
    self_boosting_attention_temperature: float = 0.40
    self_boosting_attention_classes: str = "0,1,2,3,4"
    self_boosting_attention_start_epoch: int = 1
    focus_class_aux_loss_weight: float = 0.0
    focus_class_aux_positive_weight: float = 1.0
    focus_tversky_loss_weight: float = 0.0
    focus_tversky_class: int = 1
    focus_tversky_alpha: float = 0.70
    focus_tversky_beta: float = 0.30
    focus_tversky_gamma: float = 1.0
    focus_tversky_probability_power: float = 1.0
    focus_tversky_start_epoch: int = 1
    focus_auc_rank_loss_weight: float = 0.0
    focus_auc_rank_class: int = 1
    focus_auc_rank_negative_classes: str = "0,2,4"
    focus_auc_rank_margin: float = 0.04
    focus_auc_rank_temperature: float = 0.12
    focus_auc_rank_hard_fraction: float = 0.50
    focus_auc_rank_start_epoch: int = 1
    focus_partial_auc_loss_weight: float = 0.0
    focus_partial_auc_class: int = 1
    focus_partial_auc_negative_classes: str = "0,2,4"
    focus_partial_auc_margin: float = 0.04
    focus_partial_auc_temperature: float = 0.12
    focus_partial_auc_negative_fraction: float = 0.25
    focus_partial_auc_positive_fraction: float = 0.30
    focus_partial_auc_positive_weight: float = 0.45
    focus_partial_auc_min_negative_probability: float = 0.05
    focus_partial_auc_start_epoch: int = 1
    class_independent_loss_weight: float = 0.0
    class_independent_positive_weight: float = 1.0
    bbox_foreground_dropout_loss_weight: float = 0.0
    bbox_foreground_dropout_consistency_weight: float = 0.0
    bbox_foreground_dropout_probability: float = 0.0
    bbox_foreground_dropout_min_area_ratio: float = 0.04
    bbox_foreground_dropout_max_area_ratio: float = 0.14
    bbox_foreground_dropout_mode: str = "random"
    bbox_foreground_dropout_fill: str = "mean"
    bbox_foreground_dropout_temperature: float = 1.0
    bbox_object_erasure_negative_loss_weight: float = 0.0
    bbox_object_erasure_probability: float = 0.0
    bbox_object_erasure_margin_ratio: float = 0.04
    bbox_object_erasure_fill: str = "mean"
    bbox_object_erasure_blur_kernel: int = 15
    bbox_object_erasure_temperature: float = 1.0
    background_counterfactual_consistency_weight: float = 0.0
    background_counterfactual_probability: float = 0.0
    background_counterfactual_mode: str = "desaturate_blur"
    background_counterfactual_margin: float = 0.08
    background_counterfactual_blur_kernel: int = 15
    background_counterfactual_temperature: float = 1.0
    background_focus_suppression_loss_weight: float = 0.0
    background_focus_suppression_probability: float = 0.0
    background_focus_suppression_focus_class: int = 1
    background_focus_suppression_negative_classes: str = "0,4"
    background_focus_suppression_margin: float = 0.015
    background_focus_suppression_min_probability: float = 0.05
    background_focus_suppression_probability_power: float = 1.5
    source_context_focus_suppression_loss_weight: float = 0.0
    source_context_focus_suppression_probability: float = 0.0
    source_context_focus_suppression_focus_class: int = 1
    source_context_focus_suppression_negative_classes: str = "0,2,4"
    source_context_focus_suppression_margin: float = 0.015
    source_context_focus_suppression_min_probability: float = 0.05
    source_context_focus_suppression_probability_power: float = 1.5
    surface_counterfactual_consistency_weight: float = 0.0
    surface_counterfactual_probability: float = 0.0
    surface_counterfactual_mode: str = "foreground_luma"
    surface_counterfactual_strength: float = 0.18
    surface_counterfactual_blur_kernel: int = 7
    surface_counterfactual_temperature: float = 1.0
    surface_amplified_supervised_loss_weight: float = 0.0
    surface_amplified_boundary_margin_loss_weight: float = 0.0
    surface_amplified_probability: float = 0.0
    surface_amplified_mode: str = "foreground_luma"
    surface_amplified_strength: float = 0.22
    surface_amplified_blur_kernel: int = 7
    surface_amplified_boundary_pairs: str = "0-1,1-2,2-3,1-4,4-rest"
    surface_amplified_boundary_margin: float = 0.10
    paired_view_supervised_loss_weight: float = 0.0
    paired_view_consistency_weight: float = 0.0
    paired_view_feature_consistency_weight: float = 0.0
    paired_view_fusion_loss_weight: float = 0.0
    paired_view_fusion_consistency_weight: float = 0.0
    paired_view_temperature: float = 1.0
    paired_view_feature_source: str = "head"
    masked_reconstruction_loss_weight: float = 0.0
    masked_reconstruction_mask_ratio: float = 0.45
    masked_reconstruction_foreground_weight: float = 0.70
    masked_reconstruction_detail_weight: float = 0.25
    masked_reconstruction_bbox_weight: float = 0.35
    masked_reconstruction_bbox_margin_ratio: float = 0.04
    attention_view_loss_weight: float = 0.0
    attention_crop_probability: float = 0.50
    attention_drop_probability: float = 0.25
    attention_view_start_epoch: int = 2
    attention_crop_threshold: float = 0.55
    attention_drop_threshold: float = 0.70
    attention_crop_padding_ratio: float = 0.08
    attention_crop_min_area_ratio: float = 0.20
    attention_view_foreground_weight: float = 0.35
    attention_view_score_source: str = "learned_attention"
    attention_drop_blur_kernel: int = 15
    attention_drop_dilation_kernel: int = 5
    attention_drop_min_area_ratio: float = 0.06
    attention_drop_max_area_ratio: float = 0.16
    source_context_aux_loss_weight: float = 0.0
    source_context_aux_classification_weight: float = 0.0
    source_context_aux_consistency_weight: float = 0.0
    source_context_aux_bbox_margin_ratio: float = 0.04
    source_context_aux_attention_temperature: float = 0.20
    register_diversity_loss_weight: float = 0.0
    register_attention_alignment_loss_weight: float = 0.0
    register_attention_alignment_classes: str = "0,1,2,4"
    register_attention_alignment_bbox_margin_ratio: float = 0.0
    register_attention_alignment_agreement_weight: float = 1.0
    register_attention_alignment_foreground_weight: float = 0.25
    register_attention_alignment_start_epoch: int = 1
    pairwise_margin_loss_weight: float = 0.0
    patch_evidence_router_loss_weight: float = 0.0
    patch_evidence_router_positive_weight: float = 1.0
    patch_evidence_router_start_epoch: int = 1
    patch_evidence_router_teacher_csv: str = ""
    patch_evidence_router_teacher_loss_weight: float = 0.0
    patch_evidence_router_teacher_min_confidence: float = 0.60
    patch_evidence_router_teacher_min_pair_mass: float = 0.20
    patch_evidence_router_teacher_positive_weight: float = 1.0
    patch_evidence_linear_verifier_json: str = ""
    patch_evidence_linear_verifier_pair: str = "0-1"
    patch_evidence_linear_verifier_min_pair_probability: float = 0.02
    patch_evidence_linear_verifier_max_pair_margin: float = 0.40
    patch_evidence_linear_verifier_confidence_threshold: float = 0.60
    patch_evidence_linear_verifier_logit_boost: float = 0.01
    patch_evidence_linear_verifier_protect_right_min_probability: float = 0.0
    patch_evidence_linear_verifier_training_soft_adjustment: bool = False
    patch_evidence_linear_verifier_training_soft_logit_scale: float = 0.05
    patch_evidence_linear_verifier_training_soft_gate_temperature: float = 0.05
    patch_evidence_mil_loss_weight: float = 0.0
    patch_evidence_mil_pair: str = "0-1"
    patch_evidence_mil_top_k: int = 4
    patch_evidence_mil_bbox_threshold: float = 0.05
    patch_evidence_mil_positive_weight: float = 1.0
    patch_evidence_mil_start_epoch: int = 1
    topk_reassessment_aux_loss_weight: float = 0.0
    topk_reassessment_aux_route_min_weight: float = 0.05
    ordinal_maturity_loss_weight: float = 0.0
    cumulative_ordinal_loss_weight: float = 0.0
    cumulative_ordinal_threshold_weights: str = "1.4,1.4,1.0"
    ordinal_distribution_loss_weight: float = 0.0
    ordinal_distribution_classes: str = "0,1,2,3"
    ordinal_distribution_target_sigma: float = 0.0
    ordinal_distribution_start_epoch: int = 1
    angular_margin_loss_weight: float = 0.0
    angular_margin: float = 0.12
    angular_margin_scale: float = 16.0
    angular_margin_start_epoch: int = 2
    angular_margin_classes: str = "0,1,2,3"
    subcenter_proxy_loss_weight: float = 0.0
    subcenter_proxy_subcenters: int = 3
    subcenter_proxy_margin: float = 0.08
    subcenter_proxy_scale: float = 12.0
    subcenter_proxy_classes: str = "0,1,2,4"
    subcenter_proxy_start_epoch: int = 1
    subcenter_proxy_dropout: float = 0.0
    subcenter_proxy_init_std: float = 0.002
    deep_abstention_loss_weight: float = 0.0
    deep_abstention_penalty: float = 1.30
    deep_abstention_start_epoch: int = 2
    deep_abstention_dropout: float = 0.05
    deep_abstention_initial_probability: float = 0.01
    ordinal_boundary_loss_weight: float = 0.0
    ordinal_boundary_classes: str = "0,1,2,3"
    ordinal_boundary_threshold_weights: str = "1.25,1.25,1.0"
    ordinal_boundary_temperature: float = 1.0
    ordinal_boundary_start_epoch: int = 1
    pairwise_confusion_loss_weight: float = 0.0
    pairwise_confusion_sources: str = "head"
    pairwise_confusion_start_epoch: int = 1
    pairwise_confusion_normalize: bool = True
    mutual_channel_loss_weight: float = 0.0
    mutual_channel_top_k: int = 8
    mutual_channel_diversity_weight: float = 0.20
    mutual_channel_start_epoch: int = 1
    complement_entropy_loss_weight: float = 0.0
    complement_entropy_classes: str = "all"
    complement_entropy_start_epoch: int = 1
    pretrained_distillation: bool = False
    distillation_teacher_checkpoint: str = ""
    distillation_teacher_csv: str = ""
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    distillation_focus_class_index: int = 1
    distillation_focus_class_weight: float = 1.0
    teacher_non_target_distillation_loss_weight: float = 0.0
    teacher_non_target_distillation_classes: str = "0,1,2,4"
    teacher_non_target_distillation_temperature: float = 2.0
    teacher_non_target_distillation_teacher_min_confidence: float = 0.0
    teacher_non_target_distillation_require_agreement: bool = True
    teacher_focus_margin_loss_weight: float = 0.0
    teacher_focus_margin_focus_class: int = 1
    teacher_focus_margin_negative_classes: str = "0,2,4"
    teacher_focus_margin_teacher_max_probability: float = 0.20
    teacher_focus_margin_margin: float = 0.08
    teacher_focus_margin_min_probability: float = 0.05
    teacher_focus_margin_probability_power: float = 1.5
    teacher_focus_margin_require_agreement: bool = True
    teacher_focus_binary_loss_weight: float = 0.0
    teacher_focus_binary_focus_class: int = 1
    teacher_focus_binary_classes: str = "0,1,2,4"
    teacher_focus_binary_teacher_min_confidence: float = 0.0
    teacher_focus_binary_error_power: float = 0.0
    teacher_focus_binary_hard_target_blend: float = 0.0
    teacher_focus_binary_require_agreement: bool = True
    teacher_pairwise_margin_loss_weight: float = 0.0
    teacher_pairwise_margin_teacher_mass_threshold: float = 0.0
    teacher_pairwise_margin_error_power: float = 0.0
    teacher_pairwise_margin_hard_target_blend: float = 0.0
    teacher_pairwise_margin_require_agreement: bool = True
    teacher_feature_npz: str = ""
    teacher_feature_rkd_loss_weight: float = 0.0
    teacher_feature_rkd_distance_weight: float = 1.0
    teacher_feature_rkd_angle_weight: float = 0.0
    teacher_feature_rkd_source: str = "head"
    teacher_feature_rkd_pair_mode: str = "all"
    teacher_feature_rkd_pairs: str = "0-1,1-2,1-4,2-3"
    teacher_feature_contrastive_loss_weight: float = 0.0
    teacher_feature_contrastive_temperature: float = 0.20
    teacher_feature_contrastive_projection_dim: int = 128
    teacher_feature_contrastive_source: str = "head"
    teacher_feature_contrastive_pair_mode: str = "boundary"
    teacher_feature_contrastive_pairs: str = "0-1,1-2,1-4,2-3"
    teacher_feature_contrastive_start_epoch: int = 1
    teacher_feature_contrastive_use_projection_adapter: bool = False
    teacher_feature_contrastive_adapter_dropout: float = 0.0
    elr_loss_weight: float = 0.0
    elr_beta: float = 0.70
    elr_start_epoch: int = 2
    self_adaptive_target_loss_weight: float = 0.0
    self_adaptive_target_beta: float = 0.90
    self_adaptive_target_start_epoch: int = 2
    self_adaptive_target_hard_weight: float = 0.10
    self_adaptive_target_confidence_power: float = 1.0
    self_adaptive_target_min_confidence: float = 0.0
    rdrop_loss_weight: float = 0.0
    rdrop_temperature: float = 1.0
    augmix_consistency_loss_weight: float = 0.0
    augmix_consistency_probability: float = 0.0
    augmix_consistency_severity: float = 0.18
    augmix_consistency_width: int = 2
    augmix_consistency_depth: int = 2
    augmix_consistency_alpha: float = 1.0
    augmix_consistency_temperature: float = 1.0
    illumination_consistency_loss_weight: float = 0.0
    illumination_consistency_probability: float = 0.0
    illumination_consistency_brightness: float = 0.08
    illumination_consistency_contrast: float = 0.08
    illumination_consistency_gamma: float = 0.12
    illumination_consistency_temperature: float = 1.0
    foreground_chroma_consistency_loss_weight: float = 0.0
    foreground_chroma_consistency_probability: float = 0.0
    foreground_chroma_consistency_saturation_delta: float = 0.10
    foreground_chroma_consistency_hue_delta: float = 0.015
    foreground_chroma_consistency_bbox_margin_ratio: float = 0.02
    foreground_chroma_consistency_temperature: float = 1.0
    foreground_chroma_consistency_classes: str = "0,1,2,3,4"
    friendly_adversarial_loss_weight: float = 0.0
    friendly_adversarial_epsilon: float = 2.0 / 255.0
    friendly_adversarial_step_size: float = 1.0 / 255.0
    friendly_adversarial_steps: int = 2
    friendly_adversarial_bbox_erode_ratio: float = 0.10
    friendly_adversarial_focus_class: int = 1
    friendly_adversarial_negative_classes: str = "0,2,4"
    friendly_adversarial_max_per_direction: int = 8
    friendly_adversarial_start_epoch: int = 1
    semantic_attribute_loss_weight: float = 0.0
    semantic_attribute_specs: str = (
        "maturity:0,1|2,3|4;transport:0,2|1|3,4;quality:0,1,2|3|4"
    )
    confusion_pair_mixup_loss_weight: float = 0.0
    confusion_pair_mixup_alpha: float = 0.40
    confusion_pair_mixup_pairs: str = "0-1,1-2,1-4,2-3"
    confusion_pair_mixup_max_pairs: int = 64
    confusion_pair_mixup_start_epoch: int = 1
    foreground_snapmix_loss_weight: float = 0.0
    foreground_snapmix_probability: float = 0.0
    foreground_snapmix_alpha: float = 1.0
    foreground_snapmix_pairs: str = "0-1,1-2,1-4,2-3"
    foreground_snapmix_min_area_ratio: float = 0.06
    foreground_snapmix_max_area_ratio: float = 0.22
    foreground_snapmix_bbox_margin_ratio: float = 0.02
    foreground_counterexample_mix_loss_weight: float = 0.0
    foreground_counterexample_mix_probability: float = 0.0
    foreground_counterexample_mix_source_class: int = 1
    foreground_counterexample_mix_target_classes: str = "0,2,4"
    foreground_counterexample_mix_alpha: float = 1.0
    foreground_counterexample_mix_min_area_ratio: float = 0.04
    foreground_counterexample_mix_max_area_ratio: float = 0.12
    foreground_counterexample_mix_bbox_margin_ratio: float = 0.01
    foreground_counterexample_mix_start_epoch: int = 1
    dcl_region_shuffle_loss_weight: float = 0.0
    dcl_region_shuffle_probability: float = 0.0
    dcl_region_shuffle_grid_size: int = 4
    dcl_region_shuffle_bbox_margin_ratio: float = 0.02
    dcl_region_shuffle_classes: str = "0,1,2,3,4"
    dcl_region_shuffle_start_epoch: int = 1
    quantized_label_cpu_loss_weight: float = 0.0
    quantized_label_cpu_classes: str = "0,1,2,4"
    quantized_label_cpu_min_prior: float = 0.02
    quantized_label_cpu_max_prior: float = 0.60
    quantized_label_cpu_negative_weight: float = 1.0
    quantized_label_cpu_non_negative_beta: float = 0.0
    quantized_label_cpu_start_epoch: int = 1
    self_paced_loss_weight: float = 0.0
    self_paced_loss_percentile: float = 0.80
    self_paced_loss_gamma: float = 1.0
    self_paced_loss_min_weight: float = 0.35
    self_paced_loss_start_epoch: int = 1
    self_paced_loss_class_balanced: bool = False
    cyflod_loss_damping_weight: float = 0.0
    cyflod_loss_damping_delta: float = 0.25
    cyflod_loss_damping_cycle_epochs: int = 2
    cyflod_loss_damping_min_weight: float = 0.10
    cyflod_loss_damping_start_epoch: int = 1
    sample_weight_manifest: str = ""
    sample_weight_factor: float = 1.0
    sample_weight_max: float = 5.0
    quality_group_manifest: str = ""
    group_dro_loss_weight: float = 0.0
    group_dro_temperature: float = 0.35
    group_dro_min_samples: int = 1
    ambiguous_soft_target_manifest: str = ""
    ambiguous_soft_target_alpha: float = 0.25
    targeted_margin_manifest: str = ""
    targeted_margin_loss_weight: float = 0.0
    targeted_margin_default_margin: float = 0.12
    targeted_margin_default_weight: float = 1.0
    targeted_margin_max_weight: float = 3.0
    focus_neighbor_binary_manifest: str = ""
    focus_neighbor_binary_loss_weight: float = 0.0
    focus_neighbor_binary_focus_class: int = 1
    focus_neighbor_binary_neighbor_classes: str = "0,2,4"
    focus_neighbor_binary_default_weight: float = 1.0
    focus_neighbor_binary_max_weight: float = 3.0
    focused_false_positive_margin_loss_weight: float = 0.0
    focused_false_positive_class: int = 1
    focused_false_positive_negative_classes: str = "0,4"
    focused_false_positive_margin: float = 0.10
    focused_false_positive_min_probability: float = 0.05
    focused_false_positive_probability_power: float = 1.5
    data_cartography: bool = False
    data_cartography_output: str = ""
    balance_auto_max_repeat_factor: float = 0.0
    fair_f1_gap_target: float = 0.05
    fair_f1_gap_penalty: float = 1.5
    fair_f1_min_weight: float = 0.25
    fair_f1_loss_weight: float = 0.0
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = False
    model_ema: bool = False
    model_ema_decay: float = 0.999
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
    classification_source_context: bool = False
    classification_source_context_mode: str = "desaturate_blur"
    classification_source_context_layout: str = "full"
    classification_source_context_margin_ratio: float = 0.12
    classification_source_context_background_alpha: float = 0.35
    classification_source_context_blur_radius: float = 7.0
    classification_source_context_inset_scale: float = 0.34
    classification_source_context_aux: bool = False
    resize_mode: str = "pad"
    random_resized_crop_scale_min: float = 0.8
    random_resized_crop_probability: float = 1.0
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
    illumination_normalization: bool = False
    illumination_normalization_strength: float = 0.0
    foreground_crop_mode: str = "none"
    foreground_crop_probability: float = 0.0
    foreground_crop_margin_ratio: float = 0.08
    foreground_crop_min_mask_area_ratio: float = 0.03
    foreground_crop_max_mask_area_ratio: float = 0.92
    foreground_crop_max_crop_area_ratio: float = 0.98
    background_suppression_mode: str = "none"
    background_suppression_probability: float = 0.0
    background_suppression_margin: float = 0.08
    background_suppression_blur_radius: float = 7.0
    surface_detail_amplification_mode: str = "none"
    surface_detail_amplification_probability: float = 0.0
    surface_detail_amplification_strength: float = 0.0
    surface_detail_amplification_blur_radius: float = 1.25
    surface_detail_amplification_foreground_weight: float = 0.85
    eval_surface_detail_amplification: bool = False
    local_exposure_probability: float = 0.0
    local_exposure_strength: float = 0.25
    obstacle_probability: float = 0.0
    obstacle_max_area: float = 0.12
    foreground_background_mix_probability: float = 0.0
    foreground_background_mix_margin: float = 0.08
    foreground_background_mix_min_foreground_fraction: float = 0.06
    foreground_background_mix_max_foreground_fraction: float = 0.88
    foreground_background_mix_softness: float = 5.0
    foreground_background_mix_mask_source: str = "pseudo"
    class_aware_augmentation: bool = True
    class_augmentation_power: float = 0.75
    class_augmentation_max_scale: float = 1.8
    class_conditional_augmentation_scales: str = ""
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

    count_scope = "all"
    raw_classes = balance_payload.get("classes", {})
    raw_splits = balance_payload.get("splits", {})
    if isinstance(raw_splits, dict):
        train_split = raw_splits.get("train", {})
        if isinstance(train_split, dict) and isinstance(train_split.get("classes"), dict):
            raw_classes = train_split["classes"]
            count_scope = "train"
    if not isinstance(raw_classes, dict):
        raise ValueError(f"dataset_balance.classes phai la mapping trong {balance_yaml_path}")

    classes: Dict[int, BalanceClassSpec] = {}
    total_pairs = 0
    if count_scope == "train":
        train_split = raw_splits.get("train", {})
        total_pairs = int(
            train_split.get(
                "total_objects",
                train_split.get("total_images", 0),
            )
            or 0
        )
    if total_pairs <= 0:
        total_pairs = int(
            balance_payload.get(
                "total_pairs",
                balance_payload.get("total_objects", balance_payload.get("total_images", 0)),
            )
            or 0
        )
    for raw_class_index, raw_class_payload in raw_classes.items():
        class_index = int(raw_class_index)
        if class_index < 0 or class_index >= int(num_classes):
            raise ValueError(
                "Class ID trong canbang.yaml nam ngoai khoang data.yaml: "
                f"class_id={class_index}, num_classes={num_classes}, file={balance_yaml_path}"
            )
        if isinstance(raw_class_payload, (int, float)):
            raw_class_payload = {"count": int(raw_class_payload)}
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
        train_ratio_config=float(
            balance_payload.get(
                "train_ratio_config",
                (balance_payload.get("split_ratios", {}) or {}).get("train", 0.0),
            )
            or 0.0
        ),
        val_ratio_config=float(
            balance_payload.get(
                "val_ratio_config",
                (balance_payload.get("split_ratios", {}) or {}).get("val", 0.0),
            )
            or 0.0
        ),
        test_ratio_config=float(
            balance_payload.get(
                "test_ratio_config",
                (balance_payload.get("split_ratios", {}) or {}).get("test", 0.0),
            )
            or 0.0
        ),
        classes=classes,
        count_scope=count_scope,
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

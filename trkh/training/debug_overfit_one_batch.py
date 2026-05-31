from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from trkh.core.config import default_data_yaml, load_data_spec
from trkh.data.dataset import MangoYOLOCropDataset, build_eval_transform, build_train_collate_fn
from trkh.training.loss import HybridDetectionClassificationLoss
from trkh.models.model import DETRVisionTransformerWithRegisters
from trkh.core.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit DETR on one fixed batch for loss/matcher debugging.")
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default=None,
        help="Cach xu ly names trong data.yaml; dung raw cho dataset tuy bien hoac >4 lop.",
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=0,
        help="Neu > 0, validate so class trong data.yaml truoc debug.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "runs" / "debug_overfit_one_batch")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--indices", type=str, default="", help="Comma-separated dataset indices to overfit instead of random sampling.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--background-weight", type=float, default=0.1)
    parser.add_argument("--objectness-loss-weight", type=float, default=5.0)
    parser.add_argument("--objectness-focal-alpha", type=float, default=0.75)
    parser.add_argument("--objectness-focal-gamma", type=float, default=0.5)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.5)
    parser.add_argument("--bbox-l1-loss-weight", type=float, default=5.0)
    parser.add_argument("--bbox-giou-loss-weight", type=float, default=2.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def parse_indices(raw_indices: str) -> List[int] | None:
    raw_indices = raw_indices.strip()
    if not raw_indices:
        return None
    indices = [int(item.strip()) for item in raw_indices.split(",") if item.strip()]
    if not indices:
        return None
    return indices


def move_targets_to_device(targets: Sequence[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    moved: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        moved.append(
            {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return moved


def stack_image_masks(targets: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor | None:
    masks = []
    for target in targets:
        image_mask = target.get("image_mask")
        if not torch.is_tensor(image_mask):
            return None
        masks.append(image_mask.to(dtype=torch.bool))
    if not masks:
        return None
    return torch.stack(masks, dim=0)


def pairwise_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32, device=boxes1.device)
    boxes1 = boxes1.to(dtype=torch.float32)
    boxes2 = boxes2.to(device=boxes1.device, dtype=torch.float32)
    boxes1_xyxy = torch.cat((boxes1[:, :2] - boxes1[:, 2:] / 2.0, boxes1[:, :2] + boxes1[:, 2:] / 2.0), dim=-1)
    boxes2_xyxy = torch.cat((boxes2[:, :2] - boxes2[:, 2:] / 2.0, boxes2[:, :2] + boxes2[:, 2:] / 2.0), dim=-1)
    boxes1_xyxy = boxes1_xyxy.clamp(0.0, 1.0)
    boxes2_xyxy = boxes2_xyxy.clamp(0.0, 1.0)
    top_left = torch.maximum(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    bottom_right = torch.minimum(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0.0).prod(dim=-1)
    area1 = (boxes1_xyxy[:, 2:] - boxes1_xyxy[:, :2]).clamp(min=0.0).prod(dim=-1)
    area2 = (boxes2_xyxy[:, 2:] - boxes2_xyxy[:, :2]).clamp(min=0.0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def detection_query_scores(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = outputs["logits"].detach()
    probabilities = logits.softmax(dim=-1)
    objectness_logits = outputs.get("objectness_logits")
    if objectness_logits is not None:
        class_scores, classes = probabilities.max(dim=-1)
        objectness_scores = torch.sigmoid(objectness_logits.detach())
        scores = class_scores * objectness_scores
        background_scores = 1.0 - objectness_scores
        return scores, classes, background_scores, objectness_scores

    foreground_probabilities = probabilities[..., :-1]
    scores, classes = foreground_probabilities.max(dim=-1)
    background_scores = probabilities[..., -1]
    objectness_scores = 1.0 - background_scores
    return scores, classes, background_scores, objectness_scores


def detection_ap_f1(
    outputs: Dict[str, torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    logits = outputs["logits"].detach()
    boxes = outputs["boxes"].detach()
    scores, classes, background_scores, objectness_scores = detection_query_scores(outputs)

    predictions = []
    target_count = 0
    for batch_index, target in enumerate(targets):
        target_labels = target["labels"].to(device=logits.device, dtype=torch.long)
        target_boxes = target["boxes"].to(device=logits.device, dtype=boxes.dtype)
        target_count += int(target_labels.numel())
        for query_index in range(logits.shape[1]):
            predictions.append(
                {
                    "batch": batch_index,
                    "score": float(scores[batch_index, query_index].item()),
                    "class": int(classes[batch_index, query_index].item()),
                    "box": boxes[batch_index, query_index],
                }
            )

    predictions.sort(key=lambda item: item["score"], reverse=True)
    matched = [torch.zeros((target["labels"].numel(),), dtype=torch.bool, device=logits.device) for target in targets]
    tp_values: List[float] = []
    fp_values: List[float] = []
    for prediction in predictions:
        target = targets[prediction["batch"]]
        target_labels = target["labels"].to(device=logits.device, dtype=torch.long)
        target_boxes = target["boxes"].to(device=logits.device, dtype=boxes.dtype)
        if target_labels.numel() == 0:
            tp_values.append(0.0)
            fp_values.append(1.0)
            continue
        same_class = target_labels == int(prediction["class"])
        available = same_class & (~matched[prediction["batch"]])
        if not bool(available.any().item()):
            tp_values.append(0.0)
            fp_values.append(1.0)
            continue
        candidate_indices = torch.nonzero(available, as_tuple=False).flatten()
        candidate_ious = pairwise_iou_xywh(prediction["box"].unsqueeze(0), target_boxes[candidate_indices]).squeeze(0)
        best_local = int(torch.argmax(candidate_ious).item())
        best_iou = float(candidate_ious[best_local].item())
        if best_iou >= iou_threshold:
            matched[prediction["batch"]][candidate_indices[best_local]] = True
            tp_values.append(1.0)
            fp_values.append(0.0)
        else:
            tp_values.append(0.0)
            fp_values.append(1.0)

    if target_count == 0 or not predictions:
        return {
            "ap50": 0.0,
            "best_f1_50": 0.0,
            "best_threshold": 0.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
            "foreground_score_mean": float(scores.mean().item()),
            "background_score_mean": float(background_scores.mean().item()),
            "objectness_score_mean": float(objectness_scores.mean().item()),
        }

    tp = torch.as_tensor(tp_values, dtype=torch.float32)
    fp = torch.as_tensor(fp_values, dtype=torch.float32)
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    recall = cum_tp / float(target_count)
    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-12)
    recall_points = torch.cat((torch.zeros(1), recall.cpu(), torch.ones(1)))
    precision_points = torch.cat((torch.zeros(1), precision.cpu(), torch.zeros(1)))
    for index in range(precision_points.numel() - 2, -1, -1):
        precision_points[index] = torch.maximum(precision_points[index], precision_points[index + 1])
    changing = torch.nonzero(recall_points[1:] != recall_points[:-1], as_tuple=False).flatten()
    ap = float(((recall_points[changing + 1] - recall_points[changing]) * precision_points[changing + 1]).sum().item())

    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    best_index = int(torch.argmax(f1).item())
    return {
        "ap50": ap,
        "best_f1_50": float(f1[best_index].item()),
        "best_threshold": float(predictions[best_index]["score"]),
        "precision_at_best": float(precision[best_index].item()),
        "recall_at_best": float(recall[best_index].item()),
        "foreground_score_mean": float(scores.mean().item()),
        "background_score_mean": float(background_scores.mean().item()),
        "objectness_score_mean": float(objectness_scores.mean().item()),
    }


def describe_predictions(
    outputs: Dict[str, torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    logits = outputs["logits"].detach()
    boxes = outputs["boxes"].detach()
    scores, classes, background_scores, objectness_scores = detection_query_scores(outputs)
    full_argmax = classes
    if outputs.get("objectness_logits") is None:
        full_argmax = logits.softmax(dim=-1).argmax(dim=-1)

    reports: List[Dict[str, object]] = []
    for batch_index, target in enumerate(targets):
        target_labels = target["labels"].to(device=logits.device, dtype=torch.long)
        target_boxes = target["boxes"].to(device=logits.device, dtype=boxes.dtype)
        top_count = min(max(1, top_k), logits.shape[1])
        top_indices = torch.topk(scores[batch_index], k=top_count).indices

        top_predictions = []
        for query_index_tensor in top_indices:
            query_index = int(query_index_tensor.item())
            prediction_box = boxes[batch_index, query_index]
            ious = pairwise_iou_xywh(prediction_box.unsqueeze(0), target_boxes).squeeze(0)
            best_iou = float(ious.max().item()) if ious.numel() > 0 else 0.0
            top_predictions.append(
                {
                    "query": query_index,
                    "foreground_score": float(scores[batch_index, query_index].item()),
                    "background_score": float(background_scores[batch_index, query_index].item()),
                    "objectness_score": float(objectness_scores[batch_index, query_index].item()),
                    "foreground_class": int(classes[batch_index, query_index].item()),
                    "full_argmax": int(full_argmax[batch_index, query_index].item()),
                    "best_iou_any_target": best_iou,
                    "box": [float(value) for value in prediction_box.detach().cpu().tolist()],
                }
            )

        target_reports = []
        for target_index in range(int(target_labels.numel())):
            label = int(target_labels[target_index].item())
            target_box = target_boxes[target_index]
            ious = pairwise_iou_xywh(boxes[batch_index], target_box.unsqueeze(0)).squeeze(1)
            best_any_query = int(torch.argmax(ious).item()) if ious.numel() > 0 else -1

            same_class_mask = classes[batch_index] == label
            if bool(same_class_mask.any().item()):
                same_class_queries = torch.nonzero(same_class_mask, as_tuple=False).flatten()
                same_class_ious = ious[same_class_queries]
                best_local_index = int(torch.argmax(same_class_ious).item())
                best_same_query = int(same_class_queries[best_local_index].item())
                best_same_iou = float(same_class_ious[best_local_index].item())
            else:
                best_same_query = -1
                best_same_iou = 0.0

            target_reports.append(
                {
                    "target_index": target_index,
                    "target_label": label,
                    "target_box": [float(value) for value in target_box.detach().cpu().tolist()],
                    "best_any_query": best_any_query,
                    "best_any_iou": float(ious[best_any_query].item()) if best_any_query >= 0 else 0.0,
                    "best_any_foreground_class": int(classes[batch_index, best_any_query].item()) if best_any_query >= 0 else -1,
                    "best_any_foreground_score": float(scores[batch_index, best_any_query].item()) if best_any_query >= 0 else 0.0,
                    "best_any_background_score": float(background_scores[batch_index, best_any_query].item()) if best_any_query >= 0 else 0.0,
                    "best_any_objectness_score": float(objectness_scores[batch_index, best_any_query].item()) if best_any_query >= 0 else 0.0,
                    "best_same_class_query": best_same_query,
                    "best_same_class_iou": best_same_iou,
                    "best_same_class_score": float(scores[batch_index, best_same_query].item()) if best_same_query >= 0 else 0.0,
                    "best_same_class_background_score": float(background_scores[batch_index, best_same_query].item()) if best_same_query >= 0 else 0.0,
                    "best_same_class_objectness_score": float(objectness_scores[batch_index, best_same_query].item()) if best_same_query >= 0 else 0.0,
                }
            )

        reports.append(
            {
                "batch_index": batch_index,
                "target_count": int(target_labels.numel()),
                "target_labels": [int(value) for value in target_labels.detach().cpu().tolist()],
                "top_predictions": top_predictions,
                "targets": target_reports,
            }
        )
    return reports


def describe_matches(
    outputs: Dict[str, torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
    criterion: HybridDetectionClassificationLoss,
) -> List[Dict[str, object]]:
    logits = outputs["logits"].detach()
    boxes = outputs["boxes"].detach()
    scores, classes, background_scores, objectness_scores = detection_query_scores(outputs)
    indices = criterion.matcher({"logits": logits, "boxes": boxes}, targets)

    reports: List[Dict[str, object]] = []
    for batch_index, (query_indices, target_indices) in enumerate(indices):
        target_labels = targets[batch_index]["labels"].to(device=logits.device, dtype=torch.long)
        target_boxes = targets[batch_index]["boxes"].to(device=logits.device, dtype=boxes.dtype)
        matches = []
        for query_tensor, target_tensor in zip(query_indices, target_indices):
            query_index = int(query_tensor.item())
            target_index = int(target_tensor.item())
            iou = pairwise_iou_xywh(
                boxes[batch_index, query_index].unsqueeze(0),
                target_boxes[target_index].unsqueeze(0),
            ).squeeze().item()
            matches.append(
                {
                    "query": query_index,
                    "target_index": target_index,
                    "target_label": int(target_labels[target_index].item()),
                    "foreground_class": int(classes[batch_index, query_index].item()),
                    "foreground_score": float(scores[batch_index, query_index].item()),
                    "background_score": float(background_scores[batch_index, query_index].item()),
                    "objectness_score": float(objectness_scores[batch_index, query_index].item()),
                    "iou": float(iou),
                    "pred_box": [float(value) for value in boxes[batch_index, query_index].detach().cpu().tolist()],
                    "target_box": [float(value) for value in target_boxes[target_index].detach().cpu().tolist()],
                }
            )
        reports.append({"batch_index": batch_index, "matches": matches})
    return reports


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=args.split,
        transform=build_eval_transform(image_size=args.image_size, resize_mode="pad"),
        crop_margin_ratio=0.05,
        crop_to_primary_object=False,
        class_aware_augmentation=False,
    )
    requested_indices = parse_indices(args.indices)
    if requested_indices is None:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), k=min(args.batch_size, len(dataset)))
    else:
        if min(requested_indices) < 0 or max(requested_indices) >= len(dataset):
            raise ValueError(f"indices must be in [0, {len(dataset) - 1}]")
        indices = requested_indices
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
        collate_fn=build_train_collate_fn(num_classes=data_spec.num_classes, batch_mix_probability=0.0),
    )
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = move_targets_to_device(targets, device)
    image_valid_mask = stack_image_masks(targets)
    if image_valid_mask is not None:
        image_valid_mask = image_valid_mask.to(device)

    model = DETRVisionTransformerWithRegisters(
        image_size=args.image_size,
        patch_size=16,
        use_cnn_stem=True,
        stem_channels=32,
        num_classes=data_spec.num_classes,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        num_registers=4,
        dropout=0.1,
        attention_dropout=0.0,
        drop_path_rate=0.1,
        head_pooling="cls_register_mean",
        bbox_head_hidden_dim=512,
        num_queries=args.num_queries,
        decoder_depth=4,
        decoder_num_heads=8,
        decoder_ffn_dim=1024,
        decoder_dropout=0.1,
        decoder_memory_adapter=True,
        decoder_memory_adapter_dropout=0.05,
    ).to(device)
    criterion = HybridDetectionClassificationLoss(
        num_classes=data_spec.num_classes,
        label_smoothing=args.label_smoothing,
        cls_weight=args.cls_loss_weight,
        bbox_l1_weight=args.bbox_l1_loss_weight,
        bbox_giou_weight=args.bbox_giou_loss_weight,
        background_weight=args.background_weight,
        objectness_weight=args.objectness_loss_weight,
        objectness_focal_alpha=args.objectness_focal_alpha,
        objectness_focal_gamma=args.objectness_focal_gamma,
        cardinality_weight=args.cardinality_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    object_count = sum(int(target["labels"].numel()) for target in targets)
    batch_summary = {
        "indices": indices,
        "image_shape": list(images.shape),
        "object_count": object_count,
        "classes": [target["labels"].detach().cpu().tolist() for target in targets],
        "boxes": [target["boxes"].detach().cpu().tolist() for target in targets],
        "device": str(device),
        "loss_weights": {
            "cls": args.cls_loss_weight,
            "bbox_l1": args.bbox_l1_loss_weight,
            "bbox_giou": args.bbox_giou_loss_weight,
            "background": args.background_weight,
            "objectness": args.objectness_loss_weight,
            "objectness_focal_alpha": args.objectness_focal_alpha,
            "objectness_focal_gamma": args.objectness_focal_gamma,
            "cardinality": args.cardinality_loss_weight,
        },
    }
    print("BATCH", json.dumps(batch_summary, ensure_ascii=False), flush=True)

    history: List[Dict[str, float]] = []
    final_prediction_debug: List[Dict[str, object]] = []
    final_match_debug: List[Dict[str, object]] = []
    for iteration in range(1, args.iterations + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, image_valid_mask=image_valid_mask)
        loss, details = criterion(outputs, targets, return_details=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_log = iteration == 1 or iteration % max(1, args.log_every) == 0 or iteration == args.iterations
        if should_log:
            model.eval()
            with torch.no_grad():
                eval_outputs = model(images, image_valid_mask=image_valid_mask)
                eval_loss, eval_details = criterion(eval_outputs, targets, return_details=True)
                metrics = detection_ap_f1(eval_outputs, targets)
                if iteration == args.iterations:
                    final_prediction_debug = describe_predictions(eval_outputs, targets, top_k=args.top_k)
                    final_match_debug = describe_matches(eval_outputs, targets, criterion)
            row = {
                "iteration": iteration,
                **{key: float(value) for key, value in eval_details.items() if isinstance(value, (int, float))},
                **metrics,
            }
            history.append(row)
            print("ITER", json.dumps(row, ensure_ascii=False), flush=True)
            if iteration == args.iterations:
                print(
                    "FINAL_DEBUG",
                    json.dumps({"predictions": final_prediction_debug, "matches": final_match_debug}, ensure_ascii=False),
                    flush=True,
                )

    (output_dir / "overfit_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    (output_dir / "final_debug.json").write_text(
        json.dumps({"predictions": final_prediction_debug, "matches": final_match_debug}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

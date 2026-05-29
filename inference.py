from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from config import IMAGENET_MEAN, IMAGENET_STD, to_serializable
from dataset import bbox_xywh_to_xyxy, build_eval_transform
from debug_and_optimization import TestTimeAugmentation
from model import build_model_from_checkpoint, extract_bbox_from_model_output, extract_detection_from_model_output
from utils import autocast_context, ensure_dir, json_dump, load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suy dien 1 anh bang DETR ViT-Registers cho nhieu qua xoai.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="So detection toi da. Dat 0 de dung num_queries cua checkpoint.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--export-onnx", type=Path, default=None)
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--eval-tta", dest="tta", action="store_true", default=False)
    parser.add_argument("--tta-brightness-delta", type=float, default=0.08)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--disable-calibration", action="store_true", default=False)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    return parser.parse_args()


def iter_images(image_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in Path(image_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_with_yolo_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    crop_margin_ratio: float = 0.05,
) -> Image.Image:
    width, height = image.size
    x_center, y_center, box_width, box_height = bbox
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    margin_x = box_width * crop_margin_ratio
    margin_y = box_height * crop_margin_ratio

    x_min = max(0, int(round(x_center - box_width / 2.0 - margin_x)))
    y_min = max(0, int(round(y_center - box_height / 2.0 - margin_y)))
    x_max = min(width, int(round(x_center + box_width / 2.0 + margin_x)))
    y_max = min(height, int(round(y_center + box_height / 2.0 + margin_y)))

    if x_max <= x_min or y_max <= y_min:
        return image.copy()
    return image.crop((x_min, y_min, x_max, y_max))


def _build_tta_views(
    images: torch.Tensor,
    brightness_delta: float,
) -> List[Tuple[torch.Tensor, bool]]:
    channel_dim = images.ndim - 3
    view_shape = [1] * images.ndim
    view_shape[channel_dim] = 3
    mean = images.new_tensor(IMAGENET_MEAN).view(*view_shape)
    std = images.new_tensor(IMAGENET_STD).view(*view_shape)

    def adjust_brightness(batch: torch.Tensor, factor: float) -> torch.Tensor:
        raw = batch * std + mean
        raw = torch.clamp(raw * factor, 0.0, 1.0)
        return (raw - mean) / std

    delta = max(0.0, float(brightness_delta))
    views = [(images, False)]
    flipped = torch.flip(images, dims=(-1,))
    views.append((flipped, True))
    if delta > 0.0:
        views.append((adjust_brightness(images, 1.0 - delta), False))
        views.append((adjust_brightness(images, 1.0 + delta), False))
    return views


def ensure_temporal_input(images: torch.Tensor, temporal_frames: int) -> torch.Tensor:
    temporal_frames = max(1, int(temporal_frames))
    if temporal_frames == 1:
        if images.ndim == 5:
            return images[:, -1]
        return images
    if images.ndim == 4:
        return images.unsqueeze(1).repeat(1, temporal_frames, 1, 1, 1)
    if images.ndim != 5:
        raise ValueError("Model temporal chi ho tro input 4D hoac 5D.")
    if images.shape[1] == temporal_frames:
        return images
    if images.shape[1] > temporal_frames:
        return images[:, -temporal_frames:]
    pad_count = temporal_frames - images.shape[1]
    pad = images[:, :1].expand(-1, pad_count, -1, -1, -1)
    return torch.cat((pad, images), dim=1)


def invert_bbox_from_transform_meta(
    bbox: torch.Tensor,
    meta: Dict[str, object],
) -> Dict[str, List[float]]:
    output_size = int(meta.get("output_size", 224))
    bbox_values = [float(value) for value in bbox.detach().cpu().tolist()]
    x1, y1, x2, y2 = bbox_xywh_to_xyxy(
        bbox_values,
        width=output_size,
        height=output_size,
    )

    if meta.get("mode") == "pad":
        left, top, _, _ = meta.get("padding", (0, 0, 0, 0))
        scale = float(meta.get("scale", 1.0))
        x1 = (x1 - left) / max(scale, 1e-6)
        y1 = (y1 - top) / max(scale, 1e-6)
        x2 = (x2 - left) / max(scale, 1e-6)
        y2 = (y2 - top) / max(scale, 1e-6)
    else:
        crop_box = meta.get("crop_box", (0, 0, output_size, output_size))
        crop_left = float(crop_box[0])
        crop_top = float(crop_box[1])
        scale = float(meta.get("scale", 1.0))
        x1 = x1 / max(scale, 1e-6) + crop_left
        y1 = y1 / max(scale, 1e-6) + crop_top
        x2 = x2 / max(scale, 1e-6) + crop_left
        y2 = y2 / max(scale, 1e-6) + crop_top

    orig_width, orig_height = meta.get("orig_size", (output_size, output_size))
    x1 = float(min(max(x1, 0.0), orig_width))
    y1 = float(min(max(y1, 0.0), orig_height))
    x2 = float(min(max(x2, 0.0), orig_width))
    y2 = float(min(max(y2, 0.0), orig_height))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    return {
        "xyxy": [x1, y1, x2, y2],
        "xywh": [x1 + box_width / 2.0, y1 + box_height / 2.0, box_width, box_height],
    }


def valid_mask_from_transform_meta(meta: Dict[str, object]) -> torch.Tensor:
    output_size = int(meta.get("output_size", 224))
    if meta.get("mode") != "pad":
        return torch.ones((output_size, output_size), dtype=torch.bool)
    left, top, right, bottom = [int(value) for value in meta.get("padding", (0, 0, 0, 0))]
    mask = torch.zeros((output_size, output_size), dtype=torch.bool)
    x1 = min(max(0, left), output_size)
    y1 = min(max(0, top), output_size)
    x2 = min(max(x1, output_size - max(0, right)), output_size)
    y2 = min(max(y1, output_size - max(0, bottom)), output_size)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    else:
        mask[:, :] = True
    return mask


def predict_tensor_outputs(
    model: nn.Module,
    images: torch.Tensor,
    image_valid_mask: Optional[torch.Tensor] = None,
    amp: bool = True,
    tta: bool = False,
    tta_brightness_delta: float = 0.08,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    temporal_frames = int(getattr(model, "temporal_frames", 1))
    images = ensure_temporal_input(images, temporal_frames).to(device, non_blocking=True)
    if image_valid_mask is not None:
        image_valid_mask = image_valid_mask.to(device, non_blocking=True, dtype=torch.bool)

    logits_views = []
    merged_boxes = []
    objectness_views = []
    with torch.no_grad():
        if tta:
            with autocast_context(device, amp):
                tta_runner = TestTimeAugmentation(
                    brightness_delta=tta_brightness_delta,
                    contrast_delta=tta_brightness_delta,
                    saturation_delta=tta_brightness_delta,
                    num_aug=4,
                )
                tta_outputs = tta_runner.forward(model, images)
                result = {"logits": tta_outputs["logits"].float()}
                if tta_outputs.get("boxes") is not None:
                    result["boxes"] = tta_outputs["boxes"].float()
                if tta_outputs.get("objectness_logits") is not None:
                    result["objectness_logits"] = tta_outputs["objectness_logits"].float()
                return result

        with autocast_context(device, amp):
            if (
                image_valid_mask is not None
                and hasattr(model, "forward_features")
                and hasattr(model, "forward_heads")
                and hasattr(model, "num_registers")
            ):
                features = model.forward_features(images, image_valid_mask=image_valid_mask)
                model_output = model.forward_heads(features)
            else:
                model_output = model(images)
            logits, pred_boxes, objectness_logits = extract_detection_from_model_output(model_output)
            logits_views.append(logits.float())
            if pred_boxes is not None:
                merged_boxes.append(pred_boxes.float())
            if objectness_logits is not None:
                objectness_views.append(objectness_logits.float())
    result = {
        "logits": torch.stack(logits_views, dim=0).mean(dim=0),
    }
    if merged_boxes:
        result["boxes"] = torch.stack(merged_boxes, dim=0).mean(dim=0)
    if objectness_views:
        result["objectness_logits"] = torch.stack(objectness_views, dim=0).mean(dim=0)
    return result


def predict_tensor_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    image_valid_mask: Optional[torch.Tensor] = None,
    amp: bool = True,
    tta: bool = False,
    tta_brightness_delta: float = 0.08,
) -> torch.Tensor:
    outputs = predict_tensor_outputs(
        model=model,
        images=images,
        image_valid_mask=image_valid_mask,
        amp=amp,
        tta=tta,
        tta_brightness_delta=tta_brightness_delta,
    )
    return F.softmax(outputs["logits"], dim=-1)


def post_process_detections(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    objectness_logits: Optional[torch.Tensor] = None,
    conf_threshold: Optional[float] = 0.5,
    max_detections: Optional[int] = None,
    nms_iou_threshold: Optional[float] = 0.5,
    require_foreground_argmax: bool = False,
) -> List[List[Dict[str, object]]]:
    if logits.ndim != 3:
        raise ValueError("logits cho DETR post-process phai co shape [B, Q, C].")
    if boxes is None or boxes.ndim != 3:
        raise ValueError("boxes cho DETR post-process phai co shape [B, Q, 4].")

    probabilities = F.softmax(logits.float(), dim=-1)
    if objectness_logits is not None:
        class_scores, class_indices = probabilities.max(dim=-1)
        objectness_scores = torch.sigmoid(objectness_logits.float())
        confidence_scores = class_scores * objectness_scores
        best_class_indices = class_indices
        background_index = probabilities.shape[-1]
    else:
        background_index = probabilities.shape[-1] - 1
        foreground_probabilities = probabilities[..., :-1]
        confidence_scores, class_indices = foreground_probabilities.max(dim=-1)
        best_class_indices = probabilities.argmax(dim=-1)

    keep_mask = torch.ones_like(confidence_scores, dtype=torch.bool)
    if require_foreground_argmax:
        keep_mask = keep_mask & (best_class_indices != background_index)
    if conf_threshold is not None:
        keep_mask = keep_mask & (confidence_scores >= float(conf_threshold))

    all_detections: List[List[Dict[str, object]]] = []
    for batch_index in range(probabilities.shape[0]):
        detections: List[Dict[str, object]] = []
        kept_queries = torch.nonzero(keep_mask[batch_index], as_tuple=False).flatten()
        for query_index in kept_queries.tolist():
            detections.append(
                {
                    "query_index": int(query_index),
                    "class_index": int(class_indices[batch_index, query_index].item()),
                    "probability": float(confidence_scores[batch_index, query_index].item()),
                    "box": boxes[batch_index, query_index].detach().cpu().to(torch.float32),
                }
            )
        detections.sort(key=lambda item: float(item["probability"]), reverse=True)
        detections = _nms_detections(
            detections,
            iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )
        all_detections.append(detections)
    return all_detections


def _xywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)
    boxes = boxes.to(dtype=torch.float32)
    top_left = boxes[:, :2] - boxes[:, 2:] / 2.0
    bottom_right = boxes[:, :2] + boxes[:, 2:] / 2.0
    return torch.cat((top_left, bottom_right), dim=-1).clamp(0.0, 1.0)


def _pairwise_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2].view(1, 2), boxes[:, :2])
    bottom_right = torch.minimum(box[2:].view(1, 2), boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0.0).prod(dim=-1)
    box_area = (box[2:] - box[:2]).clamp(min=0.0).prod()
    boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0.0).prod(dim=-1)
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=1e-12)


def _nms_detections(
    detections: List[Dict[str, object]],
    iou_threshold: Optional[float],
    max_detections: Optional[int],
) -> List[Dict[str, object]]:
    if not detections:
        return detections
    detections = sorted(detections, key=lambda item: float(item["probability"]), reverse=True)
    if iou_threshold is None or float(iou_threshold) <= 0.0:
        return detections[: max(1, int(max_detections))] if max_detections is not None else detections

    boxes = torch.stack([item["box"].detach().cpu().to(torch.float32) for item in detections], dim=0)
    boxes_xyxy = _xywh_to_xyxy_tensor(boxes)
    classes = torch.tensor([int(item["class_index"]) for item in detections], dtype=torch.long)
    kept_indices: List[int] = []
    for detection_index in range(len(detections)):
        should_keep = True
        for kept_index in kept_indices:
            if int(classes[detection_index].item()) != int(classes[kept_index].item()):
                continue
            iou = _pairwise_iou_xyxy(boxes_xyxy[detection_index], boxes_xyxy[kept_index : kept_index + 1])
            if float(iou.item()) > float(iou_threshold):
                should_keep = False
                break
        if should_keep:
            kept_indices.append(detection_index)
            if max_detections is not None and len(kept_indices) >= max(1, int(max_detections)):
                break
    return [detections[index] for index in kept_indices]


def resolve_detection_output_limit(
    checkpoint: Dict[str, object],
    requested_limit: Optional[int] = None,
) -> Optional[int]:
    if requested_limit is not None and int(requested_limit) > 0:
        return int(requested_limit)
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        num_queries = int(model_config.get("num_queries", 0) or 0)
        if num_queries > 0:
            return num_queries
    return None


def resolve_confidence_threshold(
    checkpoint: Dict[str, object],
    explicit_threshold: Optional[float] = None,
    disable_calibration: bool = False,
) -> Optional[float]:
    if disable_calibration:
        return None
    if explicit_threshold is not None:
        return float(explicit_threshold)

    calibration = checkpoint.get("calibration", {})
    threshold = calibration.get("best_detection_f1_confidence")
    if threshold is None:
        threshold = calibration.get("best_macro_f1_confidence")
    if threshold is None:
        metrics = checkpoint.get("metrics", {})
        detection_curve = metrics.get("detection_confidence_curve", {}) if isinstance(metrics, dict) else {}
        threshold = detection_curve.get("best_f1_50_confidence")
    if threshold is None:
        metrics = checkpoint.get("metrics", {})
        confidence_curves = metrics.get("confidence_curves", {}) if isinstance(metrics, dict) else {}
        threshold = confidence_curves.get("best_macro_f1_confidence")
    if threshold is None:
        return None
    return float(threshold)


def build_prediction_payload(
    probabilities: torch.Tensor,
    class_names: List[str],
    top_k: int,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, object]:
    probabilities = probabilities.detach().cpu().to(torch.float32).view(-1)
    top_k = max(1, min(int(top_k), probabilities.numel()))
    values, indices = torch.topk(probabilities, k=top_k)

    predictions = []
    for score, index in zip(values.tolist(), indices.tolist()):
        predictions.append(
            {
                "class_index": int(index),
                "class_name": class_names[int(index)],
                "probability": float(score),
            }
        )

    top_prediction = predictions[0]
    calibrated_prediction = top_prediction
    prediction_status = "accepted"
    if confidence_threshold is not None and float(top_prediction["probability"]) < float(confidence_threshold):
        calibrated_prediction = None
        prediction_status = "low_confidence"

    return {
        "top_prediction": top_prediction,
        "predictions": predictions,
        "calibrated_prediction": calibrated_prediction,
        "prediction_status": prediction_status,
        "confidence_threshold": confidence_threshold,
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    override_image_size: Optional[int] = None,
):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_names = list(checkpoint["class_names"])
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        num_classes=len(class_names),
        override_image_size=override_image_size,
    )
    model.to(device)
    model.eval()
    return model, checkpoint, class_names


def predict(
    model: nn.Module,
    checkpoint: Dict[str, object],
    image_path: Path,
    device: torch.device,
    top_k: int = 0,
    amp: bool = True,
    tta: bool = False,
    tta_brightness_delta: float = 0.08,
    confidence_threshold: Optional[float] = None,
    nms_iou_threshold: Optional[float] = 0.5,
) -> Dict[str, object]:
    image_size = int(checkpoint["model_config"]["image_size"])
    transform = build_eval_transform(
        image_size=image_size,
        resize_mode=checkpoint.get("augmentation_config", {}).get("resize_mode", "pad"),
    )

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        tensor, meta = transform(image, return_meta=True)
        image_valid_mask = valid_mask_from_transform_meta(meta).unsqueeze(0)
        tensor = tensor.unsqueeze(0).to(device)

    prediction_outputs = predict_tensor_outputs(
        model=model,
        images=tensor,
        image_valid_mask=image_valid_mask.to(device),
        amp=amp,
        tta=tta,
        tta_brightness_delta=tta_brightness_delta,
    )
    if "boxes" not in prediction_outputs:
        class_top_k = int(top_k) if int(top_k) > 0 else len(checkpoint["class_names"])
        probabilities = F.softmax(prediction_outputs["logits"][0], dim=-1)
        result = build_prediction_payload(
            probabilities=probabilities,
            class_names=list(checkpoint["class_names"]),
            top_k=class_top_k,
            confidence_threshold=confidence_threshold,
        )
        result["image_path"] = str(image_path.resolve())
        return result

    raw_detections = post_process_detections(
        logits=prediction_outputs["logits"],
        boxes=prediction_outputs["boxes"],
        objectness_logits=prediction_outputs.get("objectness_logits"),
        conf_threshold=confidence_threshold,
        max_detections=resolve_detection_output_limit(checkpoint, top_k),
        nms_iou_threshold=nms_iou_threshold,
    )[0]
    detections: List[Dict[str, object]] = []
    for detection in raw_detections:
        bbox_payload = invert_bbox_from_transform_meta(
            detection["box"],
            meta=meta,
        )
        detections.append(
            {
                "query_index": int(detection["query_index"]),
                "class_index": int(detection["class_index"]),
                "class_name": str(checkpoint["class_names"][int(detection["class_index"])]),
                "probability": float(detection["probability"]),
                "bbox": bbox_payload,
            }
        )
    return {
        "image_path": str(image_path.resolve()),
        "detections": detections,
        "num_detections": len(detections),
        "max_detections": resolve_detection_output_limit(checkpoint, top_k),
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
    }


def export_onnx(
    model: nn.Module,
    checkpoint: Dict[str, object],
    output_path: Path,
    opset: int = 17,
) -> Path:
    image_size = int(checkpoint["model_config"]["image_size"])
    temporal_frames = max(1, int(checkpoint["model_config"].get("temporal_frames", 1)))
    if temporal_frames > 1:
        dummy = torch.randn(
            1,
            temporal_frames,
            3,
            image_size,
            image_size,
            device=next(model.parameters()).device,
        )
    else:
        dummy = torch.randn(1, 3, image_size, image_size, device=next(model.parameters()).device)

    class OnnxExportWrapper(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, images):
            logits, boxes, objectness_logits = extract_detection_from_model_output(self.base_model(images))
            if boxes is None:
                return logits
            if objectness_logits is not None:
                return logits, boxes, objectness_logits
            return logits, boxes

    export_model = OnnxExportWrapper(model).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logits, boxes, objectness_logits = extract_detection_from_model_output(model(dummy))
    output_names = ["logits"] if boxes is None else ["logits", "bbox"]
    if objectness_logits is not None:
        output_names.append("objectness_logits")
    dynamic_axes = {
        "images": {0: "batch"},
        "logits": {0: "batch"},
    }
    if boxes is not None:
        dynamic_axes["bbox"] = {0: "batch"}
    if objectness_logits is not None:
        dynamic_axes["objectness_logits"] = {0: "batch"}
    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return output_path


def main() -> None:
    args = parse_args()
    if (args.image is None) == (args.image_dir is None):
        raise ValueError("Can truyen dung mot trong hai tham so: --image hoac --image-dir.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, _ = load_model(
        args.checkpoint,
        device,
        override_image_size=args.override_image_size,
    )
    checkpoint["model_config"]["image_size"] = int(
        args.override_image_size or checkpoint["model_config"]["image_size"]
    )

    if args.export_onnx is not None:
        exported_path = export_onnx(
            model=model,
            checkpoint=checkpoint,
            output_path=args.export_onnx,
            opset=args.onnx_opset,
        )
        print({"exported_onnx": str(exported_path.resolve())})

    confidence_threshold = resolve_confidence_threshold(
        checkpoint=checkpoint,
        explicit_threshold=args.confidence_threshold,
        disable_calibration=args.disable_calibration,
    )
    if args.image is not None:
        result = predict(
            model=model,
            checkpoint=checkpoint,
            image_path=args.image,
            device=device,
            top_k=args.top_k,
            amp=not args.disable_amp,
            tta=args.tta,
            tta_brightness_delta=args.tta_brightness_delta,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )

        if args.output_json is not None:
            json_dump(args.output_json, to_serializable(result))
        print(result)
        return

    image_paths = iter_images(args.image_dir)
    output_dir = ensure_dir(
        args.output_dir
        if args.output_dir is not None
        else args.checkpoint.resolve().parent.parent / "result" / f"{args.image_dir.stem}_pt"
    )
    results = []
    for image_path in image_paths:
        result = predict(
            model=model,
            checkpoint=checkpoint,
            image_path=image_path,
            device=device,
            top_k=args.top_k,
            amp=not args.disable_amp,
            tta=args.tta,
            tta_brightness_delta=args.tta_brightness_delta,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )
        results.append(result)
        json_dump(output_dir / f"{image_path.stem}.json", to_serializable(result))

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "images": len(image_paths),
        "output_dir": str(output_dir.resolve()),
        "results": results,
    }
    if args.output_json is not None:
        json_dump(args.output_json, to_serializable(summary))
    else:
        json_dump(output_dir / "predictions.json", to_serializable(summary))
    print({key: value for key, value in summary.items() if key != "results"})


if __name__ == "__main__":
    main()

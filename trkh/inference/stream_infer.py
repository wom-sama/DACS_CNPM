from __future__ import annotations

import argparse
from collections import deque
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from trkh.data.dataset import build_eval_transform
from trkh.inference.inference import (
    invert_bbox_from_transform_meta,
    load_model,
    post_process_detections as inference_post_process_detections,
    predict_tensor_outputs,
    resolve_confidence_threshold,
    resolve_detection_output_limit,
)
from trkh.core.utils import PredictionDriftMonitor, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suy dien webcam/video cho DETR ViT-Registers multi-object.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="So detection toi da; 0 = dung num_queries.")
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--classify-every", type=int, default=1)
    parser.add_argument("--temporal-smoothing", choices=("none", "ema", "vote"), default="ema")
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--display-max-width", type=int, default=0)
    parser.add_argument("--display-max-height", type=int, default=0)
    parser.add_argument("--save-output", action="store_true", default=False)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--tta-brightness-delta", type=float, default=0.08)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--disable-calibration", action="store_true", default=False)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--drift-window-size", type=int, default=128)
    parser.add_argument("--drift-alert-threshold", type=float, default=0.25)
    parser.add_argument("--drift-log-interval", type=int, default=32)
    return parser.parse_args()


def prepare_transform(checkpoint, override_image_size: Optional[int] = None):
    image_size = int(override_image_size or checkpoint["model_config"]["image_size"])
    augmentation_config = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    return build_eval_transform(
        image_size=image_size,
        resize_mode=augmentation_config.get("resize_mode", "pad"),
        illumination_normalization=bool(augmentation_config.get("illumination_normalization", False)),
        illumination_normalization_strength=float(
            augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0
        ),
        foreground_crop_mode=str(augmentation_config.get("foreground_crop_mode", "none") or "none"),
        foreground_crop_margin_ratio=float(augmentation_config.get("foreground_crop_margin_ratio", 0.08) or 0.08),
        foreground_crop_min_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_min_mask_area_ratio", 0.03) or 0.03
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_max_mask_area_ratio", 0.92) or 0.92
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation_config.get("foreground_crop_max_crop_area_ratio", 0.98) or 0.98
        ),
        background_suppression_mode=str(augmentation_config.get("background_suppression_mode", "none") or "none"),
        background_suppression_margin=float(augmentation_config.get("background_suppression_margin", 0.08) or 0.08),
        background_suppression_blur_radius=float(augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0),
        surface_detail_amplification_mode=str(
            augmentation_config.get("surface_detail_amplification_mode", "none") or "none"
        ),
        surface_detail_amplification_strength=float(
            augmentation_config.get("surface_detail_amplification_strength", 0.0) or 0.0
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation_config.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation_config.get("surface_detail_amplification_foreground_weight", 0.85) or 0.85
        ),
        eval_surface_detail_amplification=bool(
            augmentation_config.get("eval_surface_detail_amplification", False)
        ),
    )


def resolve_capture_source(source: Optional[str], camera_index: int):
    if source is None or not str(source).strip():
        return camera_index

    source_text = str(source).strip()
    if source_text.lstrip("+-").isdigit():
        return int(source_text)
    return source_text


def resolve_display_limits(
    display_max_width: int,
    display_max_height: int,
) -> Tuple[Optional[int], Optional[int]]:
    max_width = int(display_max_width) if int(display_max_width) > 0 else None
    max_height = int(display_max_height) if int(display_max_height) > 0 else None
    if max_width is not None and max_height is not None:
        return max_width, max_height

    try:
        import ctypes

        user32 = ctypes.windll.user32
        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))
    except Exception:
        screen_width = 0
        screen_height = 0

    if screen_width > 0 and max_width is None:
        max_width = max(320, screen_width - 120)
    if screen_height > 0 and max_height is None:
        max_height = max(320, screen_height - 160)
    return max_width, max_height


def compute_preview_size(
    frame_width: int,
    frame_height: int,
    max_width: Optional[int],
    max_height: Optional[int],
) -> Tuple[int, int]:
    if frame_width <= 0 or frame_height <= 0:
        return max(1, frame_width), max(1, frame_height)

    scale_width = (max_width / frame_width) if max_width is not None else 1.0
    scale_height = (max_height / frame_height) if max_height is not None else 1.0
    scale = min(1.0, scale_width, scale_height)
    return (
        max(1, int(round(frame_width * scale))),
        max(1, int(round(frame_height * scale))),
    )


def infer_result_dir(reference_path: Path) -> Path:
    resolved = reference_path.resolve()
    if resolved.parent.name == "checkpoints":
        return resolved.parent.parent / "deploy" / "result"
    if resolved.parent.name == "deploy":
        return resolved.parent / "result"
    return resolved.parent / "result"


def sanitize_filename_component(text: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in text.strip())
    normalized = normalized.strip("_")
    return normalized or "stream"


def source_output_stem(capture_source) -> str:
    if isinstance(capture_source, int):
        return f"camera_{capture_source}"
    source_text = str(capture_source).strip()
    if not source_text:
        return "stream"
    source_path = Path(source_text)
    if source_path.suffix:
        return source_path.stem
    return source_text


def build_output_video_path(
    reference_path: Path,
    capture_source,
    suffix: str,
) -> Path:
    result_dir = ensure_dir(infer_result_dir(reference_path))
    source_stem = sanitize_filename_component(source_output_stem(capture_source))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return result_dir / f"{source_stem}_{suffix}_{timestamp}.mp4"


def resolve_capture_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1.0:
        return 30.0
    return fps


def create_video_writer(
    output_path: Path,
    frame_size: Tuple[int, int],
    fps: float,
) -> cv2.VideoWriter:
    ensure_dir(output_path.parent)
    safe_fps = max(1.0, float(fps))
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            safe_fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Khong tao duoc VideoWriter cho output: {output_path}")


def _truncate_label(text: str, max_chars: int = 26) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def draw_translucent_panel(
    frame_bgr: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)


def draw_box_label(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    if not label:
        return

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )
    label_top = max(0, y1 - text_height - baseline - 8)
    label_bottom = label_top + text_height + baseline + 8
    label_right = min(width - 1, x1 + text_width + 10)
    draw_translucent_panel(frame_bgr, (x1, label_top), (label_right, label_bottom), color, alpha=0.8)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 5, label_bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )


def draw_prediction_overlay(
    frame_bgr: np.ndarray,
    predictions: List[dict],
    fps: float,
    classify_every: int,
    smoothing_mode: str = "none",
) -> None:
    height, width = frame_bgr.shape[:2]
    panel_x = 16
    panel_y = 16
    panel_width = min(max(320, width // 2), width - 32)
    panel_height = min(height - 32, 96 + len(predictions) * 28)
    panel_right = min(width - 16, panel_x + panel_width)
    panel_bottom = min(height - 16, panel_y + panel_height)

    draw_translucent_panel(
        frame_bgr,
        (panel_x, panel_y),
        (panel_right, panel_bottom),
        color=(18, 18, 18),
        alpha=0.58,
    )

    cv2.putText(
        frame_bgr,
        "DETR ViT Registers Stream",
        (panel_x + 12, panel_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"fps {fps:.1f} | every {max(1, classify_every)}f | {smoothing_mode}",
        (panel_x + 12, panel_y + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    bar_left = panel_x + 118
    bar_right = panel_right - 68
    bar_width = max(20, bar_right - bar_left)
    start_y = panel_y + 84

    for rank, prediction in enumerate(predictions, start=1):
        row_y = start_y + (rank - 1) * 28
        if row_y + 10 > panel_bottom - 8:
            break
        probability = float(prediction["probability"])
        class_name = _truncate_label(str(prediction["class_name"]))
        color = (70, 215, 245) if rank == 1 else (80, 165, 255)
        fill_width = int(round(bar_width * max(0.0, min(1.0, probability))))

        cv2.putText(
            frame_bgr,
            f"{rank}. {class_name}",
            (panel_x + 12, row_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame_bgr, (bar_left, row_y - 11), (bar_left + bar_width, row_y + 3), (70, 70, 70), 1)
        if fill_width > 0:
            cv2.rectangle(
                frame_bgr,
                (bar_left, row_y - 11),
                (bar_left + fill_width, row_y + 3),
                color,
                -1,
            )
        cv2.putText(
            frame_bgr,
            f"{probability * 100:5.1f}%",
            (bar_left + bar_width + 8, row_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def build_classification_prediction_result(
    logits: torch.Tensor,
    class_names: List[str],
    top_k: int = 0,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, object]:
    logits = logits.detach().cpu().to(torch.float32).reshape(-1)
    if logits.numel() <= 0:
        raise ValueError("Classification logits must contain at least one class.")
    if len(class_names) != int(logits.numel()):
        raise ValueError(
            "Classification class_names/logits mismatch: "
            f"{len(class_names)} names for {int(logits.numel())} logits."
        )

    probabilities = torch.softmax(logits, dim=-1)
    class_count = int(probabilities.numel())
    display_count = int(top_k) if int(top_k) > 0 else min(5, class_count)
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=max(1, min(display_count, class_count)),
    )
    predictions = [
        {
            "class_index": int(class_index.item()),
            "class_name": str(class_names[int(class_index.item())]),
            "probability": float(probability.item()),
        }
        for probability, class_index in zip(top_probabilities, top_indices)
    ]
    accepted_predictions = [
        prediction
        for prediction in predictions
        if confidence_threshold is None
        or float(prediction["probability"]) >= float(confidence_threshold)
    ]
    return {
        "predictions": accepted_predictions or predictions[:1],
        "top_prediction": predictions[0] if predictions else None,
    }


class StreamSmoother:
    def __init__(self, mode: str, alpha: float, window_size: int) -> None:
        self.mode = str(mode)
        self.alpha = float(min(max(alpha, 0.0), 1.0))
        self.vote_history: deque = deque(maxlen=max(1, int(window_size)))
        self.ema_logits: Optional[torch.Tensor] = None
        self.ema_boxes: Optional[torch.Tensor] = None
        self.ema_objectness_logits: Optional[torch.Tensor] = None

    def smooth_query_outputs(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        objectness_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        logits = logits.detach().cpu().to(torch.float32)
        boxes = boxes.detach().cpu().to(torch.float32)
        objectness_logits = (
            objectness_logits.detach().cpu().to(torch.float32)
            if objectness_logits is not None
            else None
        )
        if self.mode == "none":
            return logits, boxes, objectness_logits
        if self.mode == "ema":
            if self.ema_logits is None or self.ema_boxes is None:
                self.ema_logits = logits
                self.ema_boxes = boxes
                self.ema_objectness_logits = objectness_logits
            else:
                self.ema_logits = self.alpha * logits + (1.0 - self.alpha) * self.ema_logits
                self.ema_boxes = self.alpha * boxes + (1.0 - self.alpha) * self.ema_boxes
                if objectness_logits is not None:
                    self.ema_objectness_logits = (
                        objectness_logits
                        if self.ema_objectness_logits is None
                        else self.alpha * objectness_logits + (1.0 - self.alpha) * self.ema_objectness_logits
                    )
            return self.ema_logits, self.ema_boxes, self.ema_objectness_logits

        probabilities = torch.softmax(logits, dim=-1)
        self.vote_history.append(probabilities)
        voted_probabilities = torch.stack(list(self.vote_history), dim=0).mean(dim=0)
        if self.ema_boxes is None:
            self.ema_boxes = boxes
            self.ema_objectness_logits = objectness_logits
        else:
            self.ema_boxes = self.alpha * boxes + (1.0 - self.alpha) * self.ema_boxes
            if objectness_logits is not None:
                self.ema_objectness_logits = (
                    objectness_logits
                    if self.ema_objectness_logits is None
                    else self.alpha * objectness_logits + (1.0 - self.alpha) * self.ema_objectness_logits
                )
        return voted_probabilities.clamp(min=1e-8).log(), self.ema_boxes, self.ema_objectness_logits

    def smooth_class_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.detach().cpu().to(torch.float32)
        if self.mode == "none":
            return logits
        if self.mode == "ema":
            if self.ema_logits is None:
                self.ema_logits = logits
            else:
                self.ema_logits = self.alpha * logits + (1.0 - self.alpha) * self.ema_logits
            return self.ema_logits

        probabilities = torch.softmax(logits, dim=-1)
        self.vote_history.append(probabilities)
        voted_probabilities = torch.stack(list(self.vote_history), dim=0).mean(dim=0)
        return voted_probabilities.clamp(min=1e-8).log()


def post_process_detections(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    objectness_logits: Optional[torch.Tensor],
    conf_threshold: Optional[float],
    max_detections: Optional[int] = None,
    nms_iou_threshold: Optional[float] = 0.5,
) -> List[List[Dict[str, object]]]:
    return inference_post_process_detections(
        logits=logits,
        boxes=boxes,
        objectness_logits=objectness_logits,
        conf_threshold=conf_threshold,
        max_detections=max_detections,
        nms_iou_threshold=nms_iou_threshold,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, _ = load_model(
        args.checkpoint,
        device,
        override_image_size=args.override_image_size,
    )
    transform = prepare_transform(checkpoint, override_image_size=args.override_image_size)
    confidence_threshold = resolve_confidence_threshold(
        checkpoint=checkpoint,
        explicit_threshold=args.confidence_threshold,
        disable_calibration=args.disable_calibration,
    )
    max_detections = resolve_detection_output_limit(checkpoint, args.top_k)
    drift_monitor = PredictionDriftMonitor(
        reference_class_counts=checkpoint.get("data_summary", {}).get("train_class_counts", []),
        class_names=list(checkpoint["class_names"]),
        window_size=args.drift_window_size,
        alert_threshold=args.drift_alert_threshold,
        log_interval=args.drift_log_interval,
    )
    smoother = StreamSmoother(
        mode=args.temporal_smoothing,
        alpha=args.smoothing_alpha,
        window_size=args.smoothing_window,
    )

    capture_source = resolve_capture_source(args.source, args.camera_index)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc source={capture_source}")

    save_output = args.save_output or args.output_path is not None
    output_path = (
        args.output_path
        if args.output_path is not None
        else build_output_video_path(args.checkpoint, capture_source, suffix="stream")
    )
    output_fps = resolve_capture_fps(capture)
    video_writer: Optional[cv2.VideoWriter] = None

    window_name = "DETR ViT Registers Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    display_limits = resolve_display_limits(args.display_max_width, args.display_max_height)

    frame_index = 0
    preview_source_size: Optional[Tuple[int, int]] = None
    last_prediction_result: Dict[str, object] = {"detections": []}
    smoothed_fps = 0.0
    last_tick = time.perf_counter()

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            current_source_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            if preview_source_size != current_source_size:
                preview_source_size = current_source_size
                preview_size = compute_preview_size(
                    frame_width=current_source_size[0],
                    frame_height=current_source_size[1],
                    max_width=display_limits[0],
                    max_height=display_limits[1],
                )
                cv2.resizeWindow(window_name, preview_size[0], preview_size[1])
                if save_output and video_writer is None:
                    video_writer = create_video_writer(output_path, current_source_size, output_fps)

            now = time.perf_counter()
            frame_fps = 1.0 / max(now - last_tick, 1e-6)
            smoothed_fps = frame_fps if smoothed_fps <= 0.0 else (0.9 * smoothed_fps + 0.1 * frame_fps)
            last_tick = now

            frame_index += 1
            should_classify = frame_index % max(1, args.classify_every) == 0

            if should_classify:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                tensor, meta = transform(pil_image, return_meta=True)
                outputs = predict_tensor_outputs(
                    model=model,
                    images=tensor.unsqueeze(0),
                    amp=torch.cuda.is_available(),
                    tta=args.tta,
                    tta_brightness_delta=args.tta_brightness_delta,
                )
                logits = outputs["logits"][0]
                boxes = outputs.get("boxes")
                if boxes is None:
                    smoothed_logits = smoother.smooth_class_logits(logits)
                    last_prediction_result = build_classification_prediction_result(
                        logits=smoothed_logits,
                        class_names=list(checkpoint["class_names"]),
                        top_k=args.top_k,
                        confidence_threshold=confidence_threshold,
                    )
                    drift_prediction = last_prediction_result.get("top_prediction")
                    drift_event = drift_monitor.update(
                        None if drift_prediction is None else int(drift_prediction.get("class_index", -1))
                    )
                    if drift_event is not None:
                        print(drift_event, flush=True)
                else:
                    objectness_logits = outputs.get("objectness_logits")
                    smoothed_logits, smoothed_boxes, smoothed_objectness_logits = smoother.smooth_query_outputs(
                        logits,
                        boxes[0],
                        objectness_logits[0] if objectness_logits is not None else None,
                    )
                    raw_detections = post_process_detections(
                        logits=smoothed_logits.unsqueeze(0),
                        boxes=smoothed_boxes.unsqueeze(0),
                        objectness_logits=(
                            smoothed_objectness_logits.unsqueeze(0)
                            if smoothed_objectness_logits is not None
                            else None
                        ),
                        conf_threshold=confidence_threshold,
                        max_detections=max_detections,
                        nms_iou_threshold=args.nms_iou_threshold,
                    )[0]
                    detections: List[Dict[str, object]] = []
                    for detection in raw_detections:
                        bbox_payload = invert_bbox_from_transform_meta(detection["box"], meta)
                        detections.append(
                            {
                                "query_index": int(detection["query_index"]),
                                "class_index": int(detection["class_index"]),
                                "class_name": str(checkpoint["class_names"][int(detection["class_index"])]),
                                "probability": float(detection["probability"]),
                                "bbox": bbox_payload,
                            }
                        )
                    last_prediction_result = {
                        "detections": detections,
                        "num_detections": len(detections),
                    }
                    drift_prediction = detections[0] if detections else None
                    drift_event = drift_monitor.update(
                        None if drift_prediction is None else int(drift_prediction.get("class_index", -1))
                    )
                    if drift_event is not None:
                        print(drift_event, flush=True)

            detections = list(last_prediction_result.get("detections", []))
            for detection in detections:
                label_parts = [
                    f"{_truncate_label(str(detection['class_name']), max_chars=22)} "
                    f"{float(detection['probability']):.2f}"
                ]
                draw_box_label(
                    frame_bgr,
                    tuple(int(round(value)) for value in detection["bbox"]["xyxy"]),
                    " | ".join(label_parts),
                    color=(30, 220, 30),
                )

            if detections:
                overlay_detections = detections if max_detections is None else detections[:max_detections]
                draw_prediction_overlay(
                    frame_bgr,
                    overlay_detections,
                    fps=smoothed_fps,
                    classify_every=args.classify_every,
                    smoothing_mode=args.temporal_smoothing,
                )
            elif last_prediction_result.get("predictions"):
                draw_prediction_overlay(
                    frame_bgr,
                    list(last_prediction_result.get("predictions", [])),
                    fps=smoothed_fps,
                    classify_every=args.classify_every,
                    smoothing_mode=args.temporal_smoothing,
                )

            cv2.putText(
                frame_bgr,
                f"source: {capture_source} | q: quit",
                (20, frame_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if video_writer is not None:
                video_writer.write(frame_bgr)
            cv2.imshow(window_name, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        if save_output and video_writer is not None:
            print({"saved_video": str(output_path.resolve())})


if __name__ == "__main__":
    main()

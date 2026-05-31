from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import tensorrt as trt
import torch
import torch.nn.functional as F
from PIL import Image

from trkh.inference.inference import (
    invert_bbox_from_transform_meta,
    post_process_detections,
    resolve_confidence_threshold,
    resolve_detection_output_limit,
)
from trkh.inference.stream_infer import (
    StreamSmoother,
    build_output_video_path,
    compute_preview_size,
    create_video_writer,
    draw_box_label,
    draw_prediction_overlay,
    prepare_transform,
    resolve_capture_fps,
    resolve_capture_source,
    resolve_display_limits,
)
from trkh.core.utils import PredictionDriftMonitor, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suy dien webcam/video bang TensorRT DETR ViT-Registers engine.")
    parser.add_argument("--engine", type=Path, required=True)
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
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--disable-calibration", action="store_true", default=False)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--drift-window-size", type=int, default=128)
    parser.add_argument("--drift-alert-threshold", type=float, default=0.25)
    parser.add_argument("--drift-log-interval", type=int, default=32)
    return parser.parse_args()


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"TensorRT dtype khong ho tro: {dtype}")
    return mapping[dtype]


class TensorRTHybridModel:
    def __init__(self, engine_path: Path, cuda_device: int = 0, profile_index: int = 0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engine can CUDA. Hay chay tren may co GPU CUDA.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Khong tim thay TensorRT engine: {self.engine_path}")

        self.cuda_device = int(cuda_device)
        self.profile_index = int(profile_index)
        self.device = torch.device(f"cuda:{self.cuda_device}")
        torch.cuda.set_device(self.device)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Khong deserialize duoc TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Khong tao duoc TensorRT execution context.")

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise NotImplementedError(f"Chi ho tro engine 1 input. Tim thay {len(self.input_names)} input.")

        self.input_name = self.input_names[0]
        self.input_dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtypes = {
            name: torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            for name in self.output_names
        }
        self.profile_selected = False

    def _ensure_profile_selected(self, stream_handle: int) -> None:
        num_profiles = int(getattr(self.engine, "num_optimization_profiles", 0))
        if num_profiles <= 1 or self.profile_selected:
            return
        if not 0 <= self.profile_index < num_profiles:
            raise ValueError(
                f"profile_index={self.profile_index} nam ngoai so profile cua engine ({num_profiles})."
            )
        ok = self.context.set_optimization_profile_async(self.profile_index, stream_handle)
        if not ok:
            raise RuntimeError(f"Khong set duoc optimization profile {self.profile_index}.")
        self.profile_selected = True

    def infer(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(device=self.device, non_blocking=True)
        if input_tensor.dtype != self.input_dtype:
            input_tensor = input_tensor.to(dtype=self.input_dtype)
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        stream = torch.cuda.current_stream(device=self.device)
        stream_handle = int(stream.cuda_stream)
        self._ensure_profile_selected(stream_handle)

        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"Khong set duoc input shape {input_shape} cho tensor {self.input_name}.")
        self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))

        outputs: Dict[str, torch.Tensor] = {}
        for output_name in self.output_names:
            output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(output_name))
            if any(dim < 0 for dim in output_shape):
                raise RuntimeError(f"Tensor output {output_name} van con dynamic shape: {output_shape}")
            output_tensor = torch.empty(
                output_shape,
                device=self.device,
                dtype=self.output_dtypes[output_name],
            )
            self.context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
            outputs[output_name] = output_tensor

        ok = self.context.execute_async_v3(stream_handle=stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 that bai.")
        return outputs


def load_metadata(
    checkpoint_path: Path,
    override_image_size: Optional[int] = None,
) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if override_image_size is not None:
        checkpoint["model_config"]["image_size"] = int(override_image_size)
    return checkpoint


def predict_tensor_outputs_trt(
    classifier: TensorRTHybridModel,
    input_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    input_tensor = input_tensor.to(
        device=classifier.device,
        dtype=classifier.input_dtype,
        non_blocking=True,
    )
    with torch.inference_mode():
        outputs = classifier.infer(input_tensor)
        logits = outputs.get("logits")
        bbox = outputs.get("bbox")
        objectness_logits = outputs.get("objectness_logits")
        if logits is None:
            first_name = classifier.output_names[0]
            logits = outputs[first_name]
        result = {
            "logits": logits.float(),
        }
        if bbox is not None:
            result["boxes"] = bbox.float()
        if objectness_logits is not None:
            result["objectness_logits"] = objectness_logits.float()
        return result


def main() -> None:
    args = parse_args()
    checkpoint = load_metadata(
        args.checkpoint,
        override_image_size=args.override_image_size,
    )
    checkpoint["resolved_confidence_threshold"] = resolve_confidence_threshold(
        checkpoint=checkpoint,
        explicit_threshold=args.confidence_threshold,
        disable_calibration=args.disable_calibration,
    )
    max_detections = resolve_detection_output_limit(checkpoint, args.top_k)
    classifier = TensorRTHybridModel(
        engine_path=args.engine,
        cuda_device=args.cuda_device,
        profile_index=args.profile_index,
    )
    transform = prepare_transform(checkpoint, override_image_size=args.override_image_size)
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
        else build_output_video_path(args.engine, capture_source, suffix="stream_trt")
    )
    output_fps = resolve_capture_fps(capture)
    video_writer: Optional[cv2.VideoWriter] = None

    window_name = "DETR ViT Registers Stream TRT"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    display_limits = resolve_display_limits(args.display_max_width, args.display_max_height)

    frame_index = 0
    preview_source_size = None
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

            if frame_index % max(1, args.classify_every) == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                tensor, meta = transform(pil_image, return_meta=True)
                outputs = predict_tensor_outputs_trt(classifier, tensor.unsqueeze(0))
                if "boxes" not in outputs:
                    raise RuntimeError("TensorRT stream moi yeu cau engine detection co output boxes.")
                objectness_logits = outputs.get("objectness_logits")
                smoothed_logits, smoothed_boxes, smoothed_objectness_logits = smoother.smooth_query_outputs(
                    outputs["logits"][0].detach().cpu(),
                    outputs["boxes"][0].detach().cpu(),
                    objectness_logits[0].detach().cpu() if objectness_logits is not None else None,
                )
                raw_detections = post_process_detections(
                    logits=smoothed_logits.unsqueeze(0),
                    boxes=smoothed_boxes.unsqueeze(0),
                    objectness_logits=(
                        smoothed_objectness_logits.unsqueeze(0)
                        if smoothed_objectness_logits is not None
                        else None
                    ),
                    conf_threshold=checkpoint.get("resolved_confidence_threshold"),
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
                    f"{str(detection['class_name'])[:22]} "
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

            cv2.putText(
                frame_bgr,
                f"source: {capture_source} | TRT | q: quit",
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

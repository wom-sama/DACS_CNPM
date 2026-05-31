from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from trkh.core.config import to_serializable
from trkh.inference.inference import (
    invert_bbox_from_transform_meta,
    post_process_detections,
    resolve_confidence_threshold,
    resolve_detection_output_limit,
)
from trkh.inference.stream_infer import prepare_transform
from trkh.inference.stream_infer_trt import TensorRTHybridModel, predict_tensor_outputs_trt
from trkh.core.utils import ensure_dir, json_dump, load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suy dien anh hoac thu muc anh bang TensorRT engine.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=0, help="So detection toi da; 0 = dung num_queries.")
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--disable-calibration", action="store_true", default=False)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--profile-index", type=int, default=0)
    return parser.parse_args()


def iter_images(image_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in Path(image_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_image(
    classifier: TensorRTHybridModel,
    checkpoint: Dict[str, object],
    image_path: Path,
    confidence_threshold: Optional[float],
    top_k: int,
    nms_iou_threshold: Optional[float],
    override_image_size: Optional[int],
) -> Dict[str, object]:
    transform = prepare_transform(checkpoint, override_image_size=override_image_size)
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        tensor, meta = transform(image, return_meta=True)

    with torch.inference_mode():
        outputs = predict_tensor_outputs_trt(classifier, tensor.unsqueeze(0))
    if "boxes" not in outputs:
        raise RuntimeError("TensorRT image infer yeu cau engine detection co output boxes.")

    raw_detections = post_process_detections(
        logits=outputs["logits"],
        boxes=outputs["boxes"],
        objectness_logits=outputs.get("objectness_logits"),
        conf_threshold=confidence_threshold,
        max_detections=resolve_detection_output_limit(checkpoint, top_k),
        nms_iou_threshold=nms_iou_threshold,
    )[0]

    detections: List[Dict[str, object]] = []
    class_names = list(checkpoint["class_names"])
    for detection in raw_detections:
        class_index = int(detection["class_index"])
        detections.append(
            {
                "query_index": int(detection["query_index"]),
                "class_index": class_index,
                "class_name": str(class_names[class_index]),
                "probability": float(detection["probability"]),
                "bbox": invert_bbox_from_transform_meta(detection["box"], meta),
            }
        )

    return {
        "image_path": str(image_path.resolve()),
        "detections": detections,
        "num_detections": len(detections),
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
    }


def main() -> None:
    args = parse_args()
    if (args.image is None) == (args.image_dir is None):
        raise ValueError("Can truyen dung mot trong hai tham so: --image hoac --image-dir.")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if args.override_image_size is not None:
        checkpoint.setdefault("model_config", {})["image_size"] = int(args.override_image_size)

    confidence_threshold = resolve_confidence_threshold(
        checkpoint=checkpoint,
        explicit_threshold=args.confidence_threshold,
        disable_calibration=args.disable_calibration,
    )
    classifier = TensorRTHybridModel(
        engine_path=args.engine,
        cuda_device=args.cuda_device,
        profile_index=args.profile_index,
    )

    if args.image is not None:
        result = predict_image(
            classifier=classifier,
            checkpoint=checkpoint,
            image_path=args.image,
            confidence_threshold=confidence_threshold,
            top_k=args.top_k,
            nms_iou_threshold=args.nms_iou_threshold,
            override_image_size=args.override_image_size,
        )
        if args.output_json is not None:
            json_dump(args.output_json, to_serializable(result))
        print(to_serializable(result))
        return

    image_paths = iter_images(args.image_dir)
    output_dir = ensure_dir(
        args.output_dir
        if args.output_dir is not None
        else args.engine.resolve().parent / "result" / f"{args.image_dir.stem}_engine"
    )
    results = []
    for image_path in image_paths:
        result = predict_image(
            classifier=classifier,
            checkpoint=checkpoint,
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            top_k=args.top_k,
            nms_iou_threshold=args.nms_iou_threshold,
            override_image_size=args.override_image_size,
        )
        results.append(result)
        json_dump(output_dir / f"{image_path.stem}.json", to_serializable(result))

    summary = {
        "engine": str(args.engine.resolve()),
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

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from trkh.core.config import IMAGENET_MEAN, load_data_spec
from trkh.data.dataset import (
    IMAGE_EXTENSIONS,
    ResizePadToSquare,
    _grabcut_foreground_mask_array,
    _imagenet_fill,
    _pseudo_foreground_mask_array,
    _suppress_background_image,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one-per-class pseudo-vs-GrabCut background audit images."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--blur-radius", type=float, default=7.0)
    return parser.parse_args()


def _overlay_mask(image: Image.Image, mask: np.ndarray, color=(0, 210, 80), alpha: int = 110) -> Image.Image:
    base = image.convert("RGBA")
    mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8) * int(alpha))
    overlay = Image.new("RGBA", base.size, tuple(int(v) for v in color) + (0,))
    overlay.putalpha(Image.fromarray(mask_uint8, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _border_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0.0
    height, width = mask.shape[:2]
    border = max(1, int(round(min(height, width) * 0.08)))
    pieces = [
        mask[:border, :],
        mask[-border:, :],
        mask[:, :border],
        mask[:, -border:],
    ]
    total = sum(int(piece.size) for piece in pieces)
    active = sum(int(piece.sum()) for piece in pieces)
    return float(active) / float(max(1, total))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if int(union) == 0:
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum()) / float(union)


def _class_image_paths(split_root: Path, class_name: str) -> List[Path]:
    class_dir = split_root / class_name
    if not class_dir.exists():
        return []
    return [
        path
        for path in sorted(class_dir.rglob("*"), key=lambda item: str(item).lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> int:
    args = _parse_args()
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
    )
    split_root = data_spec.split_images_dir(args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    resizer = ResizePadToSquare(
        image_size=int(args.image_size),
        fill=_imagenet_fill(IMAGENET_MEAN),
    )
    records: List[Dict[str, object]] = []
    for class_index, class_name in enumerate(data_spec.class_names):
        paths = _class_image_paths(split_root, class_name)
        if not paths:
            records.append(
                {
                    "class_index": int(class_index),
                    "class_name": class_name,
                    "status": "missing_class_images",
                }
            )
            continue
        image_path = rng.choice(paths)
        with Image.open(image_path) as img:
            original = img.convert("RGB").copy()
        resized, _, padding = resizer.apply_with_meta(original)
        pseudo = _pseudo_foreground_mask_array(resized, margin=float(args.margin))
        grabcut = _grabcut_foreground_mask_array(resized, margin=float(args.margin))

        class_dir = output_dir / f"class_{class_index}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        original.save(class_dir / "00_original.jpg", quality=95)
        resized.save(class_dir / "01_resized_pad.png")
        _overlay_mask(resized, pseudo).save(class_dir / "02_pseudo_overlay.png")
        _overlay_mask(resized, grabcut).save(class_dir / "03_grabcut_overlay.png")
        _suppress_background_image(
            resized,
            mode="grabcut_blur",
            margin=float(args.margin),
            blur_radius=float(args.blur_radius),
        ).save(class_dir / "04_grabcut_blur.png")
        _suppress_background_image(
            resized,
            mode="grabcut_gray",
            margin=float(args.margin),
            blur_radius=float(args.blur_radius),
        ).save(class_dir / "05_grabcut_gray.png")

        record = {
            "class_index": int(class_index),
            "class_name": class_name,
            "source_image": str(image_path),
            "output_dir": str(class_dir),
            "resize_padding": tuple(int(v) for v in padding),
            "pseudo_mask_fraction": float(np.asarray(pseudo, dtype=bool).mean()),
            "grabcut_mask_fraction": float(np.asarray(grabcut, dtype=bool).mean()),
            "pseudo_border_fraction": _border_fraction(pseudo),
            "grabcut_border_fraction": _border_fraction(grabcut),
            "pseudo_grabcut_iou": _mask_iou(pseudo, grabcut),
        }
        (class_dir / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)

    summary = {
        "data": str(args.data),
        "split": str(args.split),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "samples": len(records)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

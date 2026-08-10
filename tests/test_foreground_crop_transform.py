from PIL import Image, ImageDraw

from trkh.data.dataset import build_eval_transform


def _green_object_image() -> Image.Image:
    image = Image.new("RGB", (96, 64), (123, 116, 103))
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 10, 72, 58), fill=(115, 165, 42))
    return image


def test_foreground_crop_default_off_keeps_meta_empty():
    transform = build_eval_transform(image_size=64, resize_mode="pad")
    _, meta = transform(_green_object_image(), return_meta=True)
    assert meta["foreground_crop_box"] is None


def test_foreground_crop_pseudo_records_crop_box():
    transform = build_eval_transform(
        image_size=64,
        resize_mode="pad",
        foreground_crop_mode="pseudo",
        foreground_crop_margin_ratio=0.05,
    )
    tensor, meta = transform(_green_object_image(), return_meta=True)
    assert tuple(tensor.shape) == (3, 64, 64)
    assert meta["foreground_crop_box"] is not None

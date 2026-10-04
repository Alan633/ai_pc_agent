from __future__ import annotations

from PIL import Image, ImageChops, ImageStat


def normalized_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, min(width, round(left * width))),
            max(0, min(height, round(top * height))),
            max(0, min(width, round(right * width))),
            max(0, min(height, round(bottom * height))),
        )
    )


def image_change_score(
    previous: Image.Image,
    current: Image.Image,
    *,
    sample_size: tuple[int, int] = (192, 108),
) -> float:
    """返回 0~1 的平均灰度变化比例。"""
    if previous.size != current.size:
        return 1.0
    a = previous.convert("L").resize(sample_size)
    b = current.convert("L").resize(sample_size)
    diff = ImageChops.difference(a, b)
    mean = ImageStat.Stat(diff).mean[0]
    return float(mean) / 255.0

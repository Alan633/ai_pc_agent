from PIL import Image

from game_baque.recorder import image_change_score, normalized_crop


def test_image_change_score_identical_is_zero() -> None:
    image = Image.new("RGB", (200, 100), "black")
    assert image_change_score(image, image.copy()) == 0.0


def test_image_change_score_detects_large_change() -> None:
    black = Image.new("RGB", (200, 100), "black")
    white = Image.new("RGB", (200, 100), "white")
    assert image_change_score(black, white) > 0.95


def test_normalized_crop_uses_relative_coordinates() -> None:
    image = Image.new("RGB", (200, 100), "black")
    crop = normalized_crop(image, (0.25, 0.20, 0.75, 0.80))
    assert crop.size == (100, 60)

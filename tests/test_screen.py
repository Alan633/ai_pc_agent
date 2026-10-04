from __future__ import annotations

from pathlib import Path

import pytest

from tools.screen import map_image_coordinates, take_screenshot


def test_coordinate_mapping() -> None:
    assert map_image_coordinates(500, 250, 1000, 500, 2000, 1000) == (1000, 500)
    assert map_image_coordinates(9999, -5, 1000, 500, 2000, 1000) == (1999, 0)


def test_screenshot_creates_png(tmp_path: Path) -> None:
    try:
        result = take_screenshot(tmp_path)
    except Exception as exc:
        pytest.skip(f"当前会话无法捕获桌面：{exc}")
    assert result.ok
    assert result.artifact is not None and result.artifact.exists()
    assert result.artifact.suffix.lower() == ".png"
    assert result.data["width"] > 0 and result.data["height"] > 0


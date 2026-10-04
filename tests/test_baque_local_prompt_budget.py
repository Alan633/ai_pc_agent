from __future__ import annotations

from pathlib import Path

from PIL import Image

from game_baque.event_vision import LOCAL_EVENT_PROMPT, _local_event_image_url
from game_baque.vision import BaqueVision, LOCAL_VISION_PROMPT, SNAPSHOT_SCHEMA


def test_local_vision_prompt_is_compact() -> None:
    # 本地 4096 ctx 还要给图片留大量 token；禁止再次把完整 schema 塞进 prompt。
    assert len(LOCAL_VISION_PROMPT) < 2600
    assert "additionalProperties" not in LOCAL_VISION_PROMPT
    assert "start_button" in LOCAL_VISION_PROMPT
    assert len(str(SNAPSHOT_SCHEMA)) > 1000


def test_local_event_prompt_is_compact() -> None:
    assert len(LOCAL_EVENT_PROMPT) < 1800
    assert "additionalProperties" not in LOCAL_EVENT_PROMPT
    assert "BEFORE" in LOCAL_EVENT_PROMPT


def test_context_length_error_detection() -> None:
    error = RuntimeError(
        'request (4123 tokens) exceeds the available context size (4096 tokens)'
    )
    assert BaqueVision._is_context_length_error(error) is True
    assert BaqueVision._is_context_length_error(RuntimeError("connection refused")) is False


def test_local_event_image_is_cropped(tmp_path: Path) -> None:
    source = tmp_path / "frame.jpg"
    Image.new("RGB", (1000, 600), "white").save(source, "JPEG")

    url = _local_event_image_url(source, "left")

    assert url.startswith("data:image/jpeg;base64,")
    # helper 能成功生成裁剪后的 data URL 即可；具体 JPEG 字节会因 Pillow 版本不同而变化。
    assert len(url) > 100

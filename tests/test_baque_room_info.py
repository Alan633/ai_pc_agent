from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from game_baque.vision import ROOM_WIN_CAPS, BaqueVision


async def _parse(tmp_path: Path, title: str) -> tuple[float | None, list[float], float | None]:
    image_path = tmp_path / "table.png"
    Image.new("RGB", (1936, 1088), (30, 120, 100)).save(image_path)
    vision = BaqueVision("", "local", "http://127.0.0.1:11434/v1")

    async def fake_quick_json(_prompt: str, _url: str, label: str) -> dict[str, object]:
        if label == "场次底分":
            return {"text": title}
        return {"beans": ["27.2万", "0", "0"]}

    vision._quick_json = fake_quick_json  # type: ignore[method-assign]
    return await vision.parse_round_info(image_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base", "room"),
    [(10, "新手"), (35, "普通"), (240, "高级"), (800, "顶级")],
)
async def test_room_is_derived_from_base_score_when_title_is_covered(
    tmp_path: Path, base: int, room: str
) -> None:
    # 实图：“认输”标签挡住了“成金普”，标题只剩“点字通场”。
    parsed_base, balances, cap = await _parse(tmp_path, f"百变八雀·点字通场 底分：{base}")
    assert parsed_base == float(base)
    assert cap == ROOM_WIN_CAPS[room]
    assert balances == [272000.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_base_score_wins_over_conflicting_title_text(tmp_path: Path) -> None:
    _base, _balances, cap = await _parse(tmp_path, "百变八雀·点字成金新手场 底分：35")
    assert cap == ROOM_WIN_CAPS["普通"]


@pytest.mark.asyncio
async def test_unknown_base_falls_back_to_title_text(tmp_path: Path) -> None:
    _base, _balances, cap = await _parse(tmp_path, "百变八雀·点字成金高级场 底分：")
    assert cap == ROOM_WIN_CAPS["高级"]

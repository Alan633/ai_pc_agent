from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RoundReport:
    """一局的结果与过程统计，用于衡量胜率和排查丢失的回合。

    只统计程序自己能确定的事实：自己点了“胡”即为胜；局结束而自己没胡，
    记为“未胜”（对手胡牌或流局）。
    """

    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ended_at: str = ""
    room_cap: float | None = None
    base_score: float | None = None
    opponent_beans: list[float] = field(default_factory=list)
    gold_ranks: list[str] = field(default_factory=list)
    turns: int = 0
    draws: int = 0
    discards: int = 0
    lost_turns: int = 0
    lost_turn_reasons: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    opportunity_cards: int = 0
    choice_picks: int = 0
    tenpai_turns: int = 0
    best_waits: int = 0
    hu_available: int = 0
    passed_hu: int = 0
    hu_clicked: bool = False
    end_reason: str = ""
    # 本局截图所在文件夹。
    screenshots_dir: str = ""

    @property
    def won(self) -> bool:
        return self.hu_clicked

    def add_reason(self, bucket: dict[str, int], reason: str) -> None:
        key = _short_reason(reason)
        bucket[key] = bucket.get(key, 0) + 1

    def summary(self) -> str:
        result = "胜（自己胡牌）" if self.won else "未胜"
        lost = (
            "，".join(f"{reason}×{count}" for reason, count in self.lost_turn_reasons.items())
            or "无"
        )
        tenpai = f"最多听 {self.best_waits} 张" if self.best_waits else "未听牌"
        room = _room_label(self.room_cap)
        return (
            f"{result}｜{room} 底分 {_fmt(self.base_score)}｜回合 {self.turns}"
            f"（摸 {self.draws}、出 {self.discards}）｜丢失回合 {self.lost_turns}（{lost}）｜"
            f"{tenpai}（听牌回合 {self.tenpai_turns}）｜可胡 {self.hu_available} 次、放弃 {self.passed_hu} 次｜"
            f"?卡 {self.opportunity_cards}｜结束：{self.end_reason or '未知'}"
            + (f"｜截图：{self.screenshots_dir}" if self.screenshots_dir else "")
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["won"] = self.won
        return data


class RoundLog:
    """把每局结果追加写入 JSONL，并提供累计统计。"""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def append(self, report: RoundReport) -> None:
        LOGGER.info("八雀本局结果：%s", report.summary())
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            LOGGER.warning("八雀对局记录写入失败：%s", exc)

    def load(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def stats(self, last: int | None = None) -> dict[str, Any]:
        rows = self.load()
        if last is not None:
            rows = rows[-last:]
        total = len(rows)
        if total == 0:
            return {"rounds": 0}
        wins = sum(1 for row in rows if row.get("won"))
        turns = sum(int(row.get("turns", 0)) for row in rows)
        lost = sum(int(row.get("lost_turns", 0)) for row in rows)
        tenpai_rounds = sum(1 for row in rows if int(row.get("best_waits", 0)) > 0)
        reasons: Counter[str] = Counter()
        for row in rows:
            reasons.update(row.get("lost_turn_reasons", {}) or {})
        return {
            "rounds": total,
            "wins": wins,
            "win_rate": round(wins / total, 4),
            "avg_turns": round(turns / total, 2),
            "lost_turns": lost,
            "lost_turn_rate": round(lost / turns, 4) if turns else None,
            "tenpai_rounds": tenpai_rounds,
            "tenpai_rate": round(tenpai_rounds / total, 4),
            "hu_available_rounds": sum(1 for row in rows if int(row.get("hu_available", 0)) > 0),
            "top_lost_reasons": reasons.most_common(5),
            "recent": [_row_line(row) for row in rows[-5:]],
        }


def _short_reason(reason: str) -> str:
    text = reason.strip()
    for marker, label in (
        ("应识别到", "手牌张数识别不符"),
        ("金牌点数", "金牌识别不全"),
        ("低置信", "低置信度"),
        ("复核失败", "操作前复核失败"),
        ("过期", "画面过期"),
        ("未识别到", "按钮/牌堆未识别"),
        ("识别失败", "识别失败"),
    ):
        if marker in text:
            return label
    return text[:20] or "未知"


def _room_label(cap: float | None) -> str:
    return {
        20_000.0: "新手场",
        300_000.0: "普通场",
        3_000_000.0: "高级场",
        8_000_000.0: "顶级场",
    }.get(cap or -1.0, "场次未知")


def _fmt(value: float | None) -> str:
    if value is None:
        return "?"
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _row_line(row: dict[str, Any]) -> str:
    result = "胜" if row.get("won") else "负"
    waits = int(row.get("best_waits", 0))
    return (
        f"{str(row.get('started_at', ''))[5:16].replace('T', ' ')} {result} "
        f"{_room_label(row.get('room_cap'))} 回合{row.get('turns', 0)} "
        f"丢{row.get('lost_turns', 0)} {'听' + str(waits) + '张' if waits else '未听'}"
    )

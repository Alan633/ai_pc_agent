from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event

from agent.session import AgentSession
from game_balatro.capture import (
    CapturedFrame,
    capture_balatro_window,
    click_frame_point,
    focus_balatro_window,
)
from game_balatro.local_strategy import BalatroLocalStrategist
from game_balatro.models import (
    BalatroDecision,
    BalatroSnapshot,
    ScreenType,
)
from game_balatro.solver import BalatroSolver
from game_balatro.vision import BalatroVision


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BalatroStepResult:
    acted: bool
    message: str
    snapshot: BalatroSnapshot
    decision: BalatroDecision
    frame: CapturedFrame
    stop_loop: bool = False


class BalatroController:
    def __init__(
        self,
        *,
        vision: BalatroVision,
        solver: BalatroSolver,
        strategist: BalatroLocalStrategist | None = None,
        decision_mode: str = "python",
        screenshots_dir: Path,
        logs_dir: Path,
        session: AgentSession,
        window_title_keyword: str = "Balatro",
        action_interval_seconds: float = 1.2,
        auto_next_run: bool = False,
    ) -> None:
        self.vision = vision
        self.solver = solver
        self.strategist = strategist
        normalized_mode = decision_mode.strip().lower()
        if normalized_mode not in {"python", "hybrid"}:
            raise ValueError("Balatro decision_mode 必须是 python 或 hybrid")
        self.decision_mode = normalized_mode
        self.screenshots_dir = screenshots_dir / "balatro"
        self.logs_dir = logs_dir / "balatro"
        self.logs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.session = session
        self.window_title_keyword = window_title_keyword
        self.action_interval_seconds = max(
            0.4,
            float(action_interval_seconds),
        )
        self.auto_next_run = bool(auto_next_run)

        self.last_snapshot: BalatroSnapshot | None = None
        self.last_decision: BalatroDecision | None = None
        self.last_message = "尚未识别 Balatro"
        self.inspect_count = 0
        self.action_count = 0
        self.unknown_count = 0
        self.run_count = 0
        self.game_over_count = 0
        self.qwen_strategy_count = 0
        self.qwen_fallback_count = 0
        self.last_decision_source = "none"
        self.strategy_memory: deque[str] = deque(maxlen=12)
        self._last_action_fingerprint: (
            tuple[object, ...] | None
        ) = None
        self._last_action_at = 0.0

    async def inspect(
        self,
    ) -> tuple[
        CapturedFrame,
        BalatroSnapshot,
    ]:
        frame = await asyncio.to_thread(
            capture_balatro_window,
            self.screenshots_dir,
            self.window_title_keyword,
        )
        snapshot = await self.vision.parse(
            frame.path
        )
        self.inspect_count += 1
        if (
            snapshot.screen_type
            == ScreenType.UNKNOWN
        ):
            self.unknown_count += 1
        self.last_snapshot = snapshot
        self.last_message = self._snapshot_text(
            snapshot
        )
        return frame, snapshot

    def _snapshot_text(
        self,
        snapshot: BalatroSnapshot,
    ) -> str:
        hand = (
            " ".join(
                f"{card.rank}{card.suit}"
                for card in snapshot.hand
            )
            or "-"
        )
        jokers = (
            " | ".join(
                joker.name or "未知Joker"
                for joker in snapshot.jokers
            )
            or "-"
        )
        return (
            f"页面={snapshot.screen_type.value} "
            f"Ante={snapshot.ante} "
            f"Round={snapshot.round_no} "
            f"Blind={snapshot.blind_name or '-'} "
            f"分数={snapshot.current_score}/"
            f"{snapshot.target_score} "
            f"${snapshot.money} "
            f"Hands={snapshot.hands_left} "
            f"Discards={snapshot.discards_left} "
            f"手牌=[{hand}] "
            f"Jokers=[{jokers}] "
            f"{snapshot.message}"
        ).strip()

    def state_text(self) -> str:
        return self.last_message

    @staticmethod
    def _decision_fingerprint(
        snapshot: BalatroSnapshot,
        decision: BalatroDecision,
    ) -> tuple[object, ...]:
        return (
            snapshot.screen_type.value,
            snapshot.ante,
            snapshot.round_no,
            snapshot.current_score,
            snapshot.money,
            snapshot.hands_left,
            snapshot.discards_left,
            tuple(
                (
                    card.rank,
                    card.suit,
                    card.selected,
                )
                for card in snapshot.hand
            ),
            decision.action,
            decision.card_indices,
            (
                (
                    decision.target.x,
                    decision.target.y,
                )
                if decision.target
                else None
            ),
        )

    @staticmethod
    def _safe_card_click_point(
        snapshot: BalatroSnapshot,
        index: int,
    ) -> tuple[int, int]:
        """返回重叠手牌的安全点击点。

        Balatro 手牌是从左到右重叠绘制的。直接点击模型给出的几何中心，
        前面的牌很容易被右侧邻牌覆盖，实际命中邻牌。这里把 X 向每张牌
        左侧已露出的区域移动约半个牌间距；最后一张同样轻微左移，仍位于
        自身最上层区域。Y 保持模型识别值，避免选中牌抬升后再引入纵向误差。
        """
        if index < 0 or index >= len(snapshot.hand):
            raise IndexError(index)
        card = snapshot.hand[index]
        if len(snapshot.hand) <= 1:
            return card.x, card.y

        gaps = [
            snapshot.hand[pos + 1].x - snapshot.hand[pos].x
            for pos in range(len(snapshot.hand) - 1)
            if snapshot.hand[pos + 1].x > snapshot.hand[pos].x
        ]
        if not gaps:
            return card.x, card.y

        local_gap: int
        if index < len(snapshot.hand) - 1:
            candidate = snapshot.hand[index + 1].x - card.x
            local_gap = candidate if candidate > 0 else gaps[len(gaps) // 2]
        else:
            candidate = card.x - snapshot.hand[index - 1].x
            local_gap = candidate if candidate > 0 else gaps[len(gaps) // 2]

        shift = max(6, round(local_gap * 0.45))
        # 不允许因为模型异常间距把点击点推离当前牌太远。
        shift = min(shift, max(12, round(sum(gaps) / len(gaps) * 0.65)))
        return max(0, card.x - shift), card.y

    async def _click(
        self,
        frame: CapturedFrame,
        x: int,
        y: int,
    ) -> None:
        self.session.set_native_call(True)
        try:
            await asyncio.to_thread(
                click_frame_point,
                frame,
                x,
                y,
            )
        finally:
            self.session.set_native_call(False)

    async def _execute(
        self,
        frame: CapturedFrame,
        snapshot: BalatroSnapshot,
        decision: BalatroDecision,
    ) -> bool:
        if decision.action in {
            "wait",
            "stop",
        }:
            return False

        fingerprint = self._decision_fingerprint(
            snapshot,
            decision,
        )
        now = time.monotonic()
        if (
            fingerprint
            == self._last_action_fingerprint
            and now - self._last_action_at
            < max(
                2.0,
                self.action_interval_seconds,
            )
        ):
            LOGGER.warning(
                "Balatro 拒绝短时间重复执行同一动作：%s",
                decision.action,
            )
            return False

        if decision.action in {
            "play_hand",
            "discard",
        }:
            desired = set(
                decision.card_indices
            )
            for index in range(len(snapshot.hand) - 1, -1, -1):
                card = snapshot.hand[index]
                if (
                    card.selected
                    and index not in desired
                ):
                    safe_x, safe_y = self._safe_card_click_point(
                        snapshot,
                        index,
                    )
                    LOGGER.info(
                        "Balatro手牌安全点击 index=%d card=%s%s model=(%d,%d) safe=(%d,%d) selected=%s",
                        index,
                        card.rank,
                        card.suit,
                        card.x,
                        card.y,
                        safe_x,
                        safe_y,
                        card.selected,
                    )
                    await self._click(
                        frame,
                        safe_x,
                        safe_y,
                    )
                    await asyncio.sleep(0.08)

            # 手牌从左到右重叠绘制，先点右侧最上层牌，再向左，
            # 可以逐步暴露左侧牌，进一步降低命中邻牌的概率。
            for index in sorted(
                decision.card_indices,
                reverse=True,
            ):
                if (
                    index < 0
                    or index >= len(snapshot.hand)
                ):
                    raise RuntimeError(
                        "Balatro Solver 给出非法手牌索引："
                        f"{index}"
                    )
                card = snapshot.hand[index]
                if not card.selected:
                    safe_x, safe_y = self._safe_card_click_point(
                        snapshot,
                        index,
                    )
                    LOGGER.info(
                        "Balatro手牌安全点击 index=%d card=%s%s model=(%d,%d) safe=(%d,%d) selected=%s",
                        index,
                        card.rank,
                        card.suit,
                        card.x,
                        card.y,
                        safe_x,
                        safe_y,
                        card.selected,
                    )
                    await self._click(
                        frame,
                        safe_x,
                        safe_y,
                    )
                    await asyncio.sleep(0.08)

        target = decision.target
        if target is None:
            raise RuntimeError(
                f"Balatro 动作 {decision.action} "
                "缺少安全点击目标"
            )

        await self._click(
            frame,
            target.x,
            target.y,
        )
        self.action_count += 1
        self._last_action_fingerprint = (
            fingerprint
        )
        self._last_action_at = (
            time.monotonic()
        )
        self._write_decision(
            snapshot,
            decision,
            acted=True,
        )
        LOGGER.info(
            "Balatro执行 action=%s cards=%s target=%s reason=%s",
            decision.action,
            decision.card_indices,
            (
                (
                    decision.target.x,
                    decision.target.y,
                )
                if decision.target
                else None
            ),
            decision.reason,
        )
        return True

    def _write_decision(
        self,
        snapshot: BalatroSnapshot,
        decision: BalatroDecision,
        *,
        acted: bool,
    ) -> None:
        record = {
            "time": (
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="seconds"
                )
            ),
            "acted": acted,
            "screen_type": (
                snapshot.screen_type.value
            ),
            "ante": snapshot.ante,
            "round_no": snapshot.round_no,
            "blind": snapshot.blind_name,
            "score": snapshot.current_score,
            "target_score": (
                snapshot.target_score
            ),
            "money": snapshot.money,
            "hands_left": (
                snapshot.hands_left
            ),
            "discards_left": (
                snapshot.discards_left
            ),
            "hand": [
                f"{card.rank}{card.suit}"
                for card in snapshot.hand
            ],
            "jokers": [
                joker.name
                for joker in snapshot.jokers
            ],
            "action": decision.action,
            "card_indices": list(
                decision.card_indices
            ),
            "target": (
                {
                    "x": decision.target.x,
                    "y": decision.target.y,
                }
                if decision.target
                else None
            ),
            "reason": decision.reason,
            "expected_strength": (
                decision.expected_strength
            ),
            "decision_source": self.last_decision_source,
            "strategy_memory": list(self.strategy_memory),
            "metadata": decision.metadata,
        }
        path = (
            self.logs_dir
            / "decisions.jsonl"
        )
        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    async def _choose_decision(
        self,
        snapshot: BalatroSnapshot,
    ) -> BalatroDecision:
        top_n = (
            self.strategist.top_n
            if self.strategist is not None
            else 6
        )
        candidates = self.solver.rank_candidates(
            snapshot,
            top_n=top_n,
        )
        if not candidates:
            self.last_decision_source = "python"
            return self.solver.decide(snapshot)

        baseline = candidates[0]
        can_use_qwen = (
            self.decision_mode == "hybrid"
            and self.strategist is not None
            and len(candidates) >= 2
            and snapshot.screen_type
            in {
                ScreenType.PLAY,
                ScreenType.SHOP,
                ScreenType.PACK,
                ScreenType.BLIND_SELECT,
            }
        )
        if not can_use_qwen:
            self.last_decision_source = "python"
            return baseline

        result = await self.strategist.rerank(
            snapshot=snapshot,
            candidates=candidates,
            strategy_memory=tuple(self.strategy_memory),
        )
        if result.used_model:
            self.qwen_strategy_count += 1
            self.last_decision_source = "qwen_hybrid"
            if result.build_plan:
                memory = (
                    f"Ante {snapshot.ante}/Round {snapshot.round_no}: "
                    f"{result.build_plan}"
                )
                if (
                    not self.strategy_memory
                    or self.strategy_memory[-1] != memory
                ):
                    self.strategy_memory.append(memory)
            LOGGER.info(
                "Balatro HYBRID choice=%s confidence=%.2f plan=%s reason=%s",
                result.choice_id,
                result.confidence,
                result.build_plan,
                result.reason,
            )
            return result.decision

        self.qwen_fallback_count += 1
        self.last_decision_source = "python_fallback"
        LOGGER.info(
            "Balatro HYBRID回退Python confidence=%.2f reason=%s",
            result.confidence,
            result.reason,
        )
        return baseline

    async def step(
        self,
        *,
        execute: bool,
    ) -> BalatroStepResult:
        frame, snapshot = await self.inspect()
        decision = await self._choose_decision(
            snapshot
        )
        self.last_decision = decision
        acted = False
        if execute:
            acted = await self._execute(
                frame,
                snapshot,
                decision,
            )
        else:
            self._write_decision(
                snapshot,
                decision,
                acted=False,
            )

        message = (
            f"{self._snapshot_text(snapshot)}\n"
            f"决策={decision.action}："
            f"{decision.reason}"
        )
        self.last_message = message
        return BalatroStepResult(
            acted=acted,
            message=message,
            snapshot=snapshot,
            decision=decision,
            frame=frame,
            stop_loop=(
                decision.action == "stop"
            ),
        )

    async def prepare_auto(self) -> None:
        title = await asyncio.to_thread(
            focus_balatro_window,
            self.window_title_keyword,
        )
        LOGGER.info(
            "Balatro自动模式已聚焦窗口：%s",
            title,
        )
        await asyncio.sleep(0.4)

    async def run_loop(
        self,
        stop_signal: Event,
    ) -> str:
        self.run_count += 1
        unknown_streak = 0
        navigation_screen: ScreenType | None = None
        navigation_streak = 0
        while not stop_signal.is_set():
            try:
                result = await self.step(
                    execute=True
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception(
                    "Balatro自动循环单步失败"
                )
                return (
                    "Balatro 自动游戏停止："
                    f"{type(exc).__name__}: {exc}"
                )

            if (
                result.snapshot.screen_type
                == ScreenType.UNKNOWN
            ):
                unknown_streak += 1
            else:
                unknown_streak = 0

            if (
                result.acted
                and result.snapshot.screen_type
                in {
                    ScreenType.START,
                    ScreenType.BLIND_SELECT,
                }
            ):
                if (
                    result.snapshot.screen_type
                    == navigation_screen
                ):
                    navigation_streak += 1
                else:
                    navigation_screen = (
                        result.snapshot.screen_type
                    )
                    navigation_streak = 1
            else:
                navigation_screen = None
                navigation_streak = 0

            if navigation_streak >= 3:
                return (
                    "Balatro 连续 3 次点击同一导航页面仍未推进，"
                    "为避免重复误点已停止。"
                )

            if result.stop_loop:
                self.game_over_count += 1
                if not self.auto_next_run:
                    return (
                        "Balatro 本局已结束，"
                        "已停止自动模式。"
                    )
                unknown_streak = 0

            if unknown_streak >= 3:
                return (
                    "Balatro 连续 3 次无法可靠识别页面，"
                    "为避免误点已停止。"
                )

            await asyncio.sleep(
                self.action_interval_seconds
            )
        return "Balatro 自动游戏已停止。"

    def stats(
        self,
    ) -> dict[str, object]:
        return {
            "runs_started": self.run_count,
            "inspections": self.inspect_count,
            "unknown": self.unknown_count,
            "actions": self.action_count,
            "game_overs": self.game_over_count,
            "decision_mode": self.decision_mode,
            "last_decision_source": self.last_decision_source,
            "qwen_strategy_count": self.qwen_strategy_count,
            "qwen_fallback_count": self.qwen_fallback_count,
            "strategy_memory": list(self.strategy_memory),
            "last_state": self.state_text(),
            "last_decision": (
                {
                    "action": (
                        self.last_decision.action
                    ),
                    "reason": (
                        self.last_decision.reason
                    ),
                    "cards": list(
                        self.last_decision.card_indices
                    ),
                }
                if self.last_decision
                else None
            ),
        }

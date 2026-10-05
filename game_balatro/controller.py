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
from game_balatro.balatrobot import BalatroBotClient, BalatroBotError
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
        balatrobot: BalatroBotClient | None = None,
        control_mode: str = "mouse",
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
        normalized_control = control_mode.strip().lower()
        if normalized_control not in {"auto", "api", "mouse"}:
            raise ValueError(
                "Balatro control_mode 必须是 auto、api 或 mouse"
            )
        if normalized_control == "api" and balatrobot is None:
            raise ValueError("Balatro api 模式必须配置 BalatroBot 客户端")
        self.control_mode = normalized_control
        self.balatrobot = balatrobot

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
        self.api_state_count = 0
        self.api_action_count = 0
        self.api_failure_count = 0
        self.mouse_action_count = 0
        self.mouse_fallback_count = 0
        self.last_state_source = "none"
        self.last_control_backend = "none"
        self._api_retry_after = 0.0
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

        snapshot: BalatroSnapshot | None = None
        api_allowed = (
            self.balatrobot is not None
            and self.control_mode != "mouse"
            and (
                self.control_mode == "api"
                or time.monotonic() >= self._api_retry_after
            )
        )
        if api_allowed:
            try:
                assert self.balatrobot is not None
                snapshot = await asyncio.to_thread(
                    self.balatrobot.get_snapshot,
                    timeout_seconds=min(
                        1.5,
                        self.balatrobot.timeout_seconds,
                    ),
                )
                self.api_state_count += 1
                self.last_state_source = "balatrobot"
                self._api_retry_after = 0.0
            except BalatroBotError as exc:
                self.api_failure_count += 1
                if self.control_mode == "api":
                    raise RuntimeError(
                        f"BalatroBot API 状态读取失败：{exc}"
                    ) from exc
                # auto 模式短暂熔断，避免 BalatroBot 未安装时每一帧都等连接超时。
                self._api_retry_after = time.monotonic() + 15.0
                LOGGER.warning(
                    "BalatroBot不可用，15秒内回退视觉识别：%s",
                    exc,
                )

        if snapshot is None:
            snapshot = await self.vision.parse(
                frame.path
            )
            self.last_state_source = "vision"

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

    async def _execute_mouse(
        self,
        frame: CapturedFrame,
        snapshot: BalatroSnapshot,
        decision: BalatroDecision,
        fingerprint: tuple[object, ...],
    ) -> bool:
        if decision.action in {
            "wait",
            "stop",
            "cash_out",
        }:
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
        if target is None or not target.visible:
            raise RuntimeError(
                f"Balatro 动作 {decision.action} "
                "缺少可见的安全点击目标"
            )

        await self._click(
            frame,
            target.x,
            target.y,
        )
        self.action_count += 1
        self.mouse_action_count += 1
        self.last_control_backend = "mouse"
        self._last_action_fingerprint = fingerprint
        self._last_action_at = time.monotonic()
        self._write_decision(
            snapshot,
            decision,
            acted=True,
        )
        LOGGER.info(
            "Balatro鼠标执行 action=%s cards=%s target=%s reason=%s",
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

        use_api = (
            self.balatrobot is not None
            and self.control_mode != "mouse"
            and snapshot.metadata.get("source") == "balatrobot"
        )
        if use_api:
            assert self.balatrobot is not None
            try:
                command = self.balatrobot.command_for_decision(
                    snapshot,
                    decision,
                )
                if command is None:
                    raise BalatroBotError(
                        "当前候选缺少可验证的 BalatroBot 动作映射"
                    )
                method, params = command
                await asyncio.to_thread(
                    self.balatrobot.call,
                    method,
                    params,
                )
                self.action_count += 1
                self.api_action_count += 1
                self.last_control_backend = "balatrobot"
                self._last_action_fingerprint = fingerprint
                self._last_action_at = time.monotonic()
                self._write_decision(
                    snapshot,
                    decision,
                    acted=True,
                )
                LOGGER.info(
                    "BalatroBot执行 method=%s params=%s action=%s reason=%s",
                    method,
                    params,
                    decision.action,
                    decision.reason,
                )
                return True
            except BalatroBotError as exc:
                self.api_failure_count += 1
                if self.control_mode == "api":
                    raise RuntimeError(
                        f"BalatroBot API 动作失败，严格API模式停止：{exc}"
                    ) from exc

                LOGGER.warning(
                    "BalatroBot动作失败，切换视觉+安全鼠标兜底：%s",
                    exc,
                )
                # 不能使用 API Snapshot 的占位坐标，也不能使用 API 调用前的旧截图。
                # API 可能已经执行成功、只是响应超时；必须重新截取最新画面再求解，
                # 才能避免同一手牌/按钮被重复点击。
                fallback_frame = await asyncio.to_thread(
                    capture_balatro_window,
                    self.screenshots_dir,
                    self.window_title_keyword,
                )
                vision_snapshot = await self.vision.parse(
                    fallback_frame.path
                )
                fallback_decision = await self._choose_decision(
                    vision_snapshot
                )
                if fallback_decision.action in {"wait", "stop"}:
                    return False
                self.mouse_fallback_count += 1
                source = self.last_decision_source
                self.last_decision_source = (
                    f"{source}_mouse_fallback"
                )
                fallback_fingerprint = self._decision_fingerprint(
                    vision_snapshot,
                    fallback_decision,
                )
                return await self._execute_mouse(
                    fallback_frame,
                    vision_snapshot,
                    fallback_decision,
                    fallback_fingerprint,
                )

        # API 未启用、处于熔断期，或当前状态来自视觉时沿用原来的安全鼠标链路。
        return await self._execute_mouse(
            frame,
            snapshot,
            decision,
            fingerprint,
        )

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
            "state_source": self.last_state_source,
            "control_mode": self.control_mode,
            "control_backend": self.last_control_backend,
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
            "control_mode": self.control_mode,
            "last_state_source": self.last_state_source,
            "last_control_backend": self.last_control_backend,
            "last_decision_source": self.last_decision_source,
            "api_state_count": self.api_state_count,
            "api_action_count": self.api_action_count,
            "api_failure_count": self.api_failure_count,
            "mouse_action_count": self.mouse_action_count,
            "mouse_fallback_count": self.mouse_fallback_count,
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
